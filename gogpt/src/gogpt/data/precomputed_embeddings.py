# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Precomputed per-residue protein embeddings on disk.

The protein encoder is frozen, so its embeddings can be computed once
(scripts/precompute_embeddings.py) and read during training instead of running the
encoder every step. Layout:

    <dir>/
        emb_r{rank}_{n:03d}.bin   raw fp16 [rows, dim], C-contiguous
        index_r{rank}.jsonl       per-writer index, one line per protein
        meta_r{rank}.json         settings the writer ran with
        index.parquet             merged index, written by finalize()
        manifest.json             model / layer / dim / dtype / max_len / counts

A protein has exactly min(len(sequence), max_len) rows, one per residue, matching the
`protein_tokens` the datasets store. Reads use one os.preadv() per protein, which is
fast on both local disks and network filesystems. For training, put the directory on
local disk if you can: every epoch reads all of it in random order.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

MANIFEST_NAME = "manifest.json"
INDEX_NAME = "index.parquet"


class EmbeddingCacheError(RuntimeError):
    """The embeddings directory is missing, malformed or inconsistent with the run."""


@dataclass(frozen=True)
class IndexEntry:
    shard: str
    row_offset: int
    length: int


def shard_name(rank, n):
    return f"emb_r{rank}_{n:03d}.bin"


# --------------------------------------------------------------------------- writer


class EmbeddingShardWriter:
    """Append-only writer for one rank.

    Bytes are flushed before the index line is appended, so a killed job leaves at most
    trailing unindexed bytes; the next writer truncates them and resumes.
    """

    def __init__(self, cache_dir, rank, dim, meta, max_shard_bytes=16 << 30):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.dim = dim
        self.max_shard_bytes = max_shard_bytes
        self.jsonl_path = self.cache_dir / f"index_r{rank}.jsonl"
        self.done = {}
        self._shard_n = 0
        self._fh = None
        self._rows_in_shard = 0
        self._write_meta({**meta, "dim": dim})
        self._recover()

    def _write_meta(self, meta):
        path = self.cache_dir / f"meta_r{self.rank}.json"
        if path.exists():
            with open(path) as f:
                prior = json.load(f)
            conflicts = {k: (prior.get(k), v) for k, v in meta.items() if k in prior and prior[k] != v}
            if conflicts:
                raise EmbeddingCacheError(
                    f"rank {self.rank} was previously written with different settings {conflicts}; "
                    f"resuming would mix incompatible embeddings. Use a new output directory.")
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

    def _recover(self):
        """Replay this rank's index, then truncate shard bytes past its last entry."""
        if self.jsonl_path.exists():
            with open(self.jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        break  # torn final line from a hard kill
                    self.done[rec["protein_id"]] = IndexEntry(rec["shard"], rec["row_offset"], rec["length"])

        valid_rows = {}
        for e in self.done.values():
            valid_rows[e.shard] = max(valid_rows.get(e.shard, 0), e.row_offset + e.length)
        shards = sorted(self.cache_dir.glob(f"emb_r{self.rank}_*.bin"))
        for path in shards:
            keep = valid_rows.get(path.name, 0) * self.dim * 2
            if path.stat().st_size > keep:
                with open(path, "r+b") as f:
                    f.truncate(keep)
        if shards:
            last = shards[-1]
            self._shard_n = int(last.stem.split("_")[-1])
            if last.stat().st_size >= self.max_shard_bytes:
                self._shard_n += 1

    def _open_shard(self):
        if self._fh is None:
            path = self.cache_dir / shard_name(self.rank, self._shard_n)
            self._fh = open(path, "ab")
            self._rows_in_shard = path.stat().st_size // (self.dim * 2)
        return self._fh

    def write(self, protein_id, emb):
        if emb.dtype != np.float16:
            raise ValueError(f"{protein_id}: expected float16, got {emb.dtype}")
        if emb.ndim != 2 or emb.shape[1] != self.dim:
            raise ValueError(f"{protein_id}: expected [L, {self.dim}], got {emb.shape}")
        if not np.isfinite(emb).all():
            raise ValueError(f"{protein_id}: non-finite values in embedding")

        if self._fh is not None and self._fh.tell() >= self.max_shard_bytes:
            self._fh.close()
            self._fh = None
            self._shard_n += 1
        fh = self._open_shard()
        entry = IndexEntry(shard_name(self.rank, self._shard_n), self._rows_in_shard, emb.shape[0])
        fh.write(np.ascontiguousarray(emb).tobytes())
        fh.flush()
        os.fsync(fh.fileno())
        with open(self.jsonl_path, "a") as idx:
            idx.write(json.dumps({"protein_id": protein_id, "shard": entry.shard,
                                  "row_offset": entry.row_offset, "length": entry.length}) + "\n")
            idx.flush()
            os.fsync(idx.fileno())
        self._rows_in_shard += emb.shape[0]
        self.done[protein_id] = entry

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def read_rank_meta(cache_dir):
    """The settings every rank was written with; raises if ranks disagree."""
    metas = {}
    for path in sorted(Path(cache_dir).glob("meta_r*.json")):
        with open(path) as f:
            metas[path.name] = json.load(f)
    if not metas:
        raise EmbeddingCacheError(f"no meta_r*.json under {cache_dir}; nothing has been written yet")
    names = list(metas)
    for name in names[1:]:
        if metas[name] != metas[names[0]]:
            raise EmbeddingCacheError(
                f"ranks disagree on how embeddings were produced: {names[0]}={metas[names[0]]} "
                f"vs {name}={metas[name]}")
    return dict(metas[names[0]])


def finalize(cache_dir, manifest, expected_lengths=None):
    """Merge the per-rank indices into index.parquet and write manifest.json.

    expected_lengths maps protein_id -> min(len(sequence), max_len); every entry is
    checked against it.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    cache_dir = Path(cache_dir)
    entries = {}
    for path in sorted(cache_dir.glob("index_r*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # A protein written twice (e.g. by runs with different shard counts) keeps
                # its first entry; the other copy stays as unreferenced bytes.
                entries.setdefault(rec["protein_id"],
                                   IndexEntry(rec["shard"], rec["row_offset"], rec["length"]))
    if not entries:
        raise EmbeddingCacheError(f"no index_r*.jsonl records under {cache_dir}")

    dim = manifest["dim"]
    by_shard = {}
    for e in entries.values():
        by_shard.setdefault(e.shard, []).append(e)
    for shard, items in by_shard.items():
        items.sort(key=lambda e: e.row_offset)
        end = 0
        for e in items:
            if e.row_offset < end:
                raise EmbeddingCacheError(f"{shard}: overlapping entries at row {e.row_offset}; index is corrupted")
            end = e.row_offset + e.length
        if (cache_dir / shard).stat().st_size < end * dim * 2:
            raise EmbeddingCacheError(f"{shard}: index references more bytes than the file holds")

    if expected_lengths is not None:
        bad = [(p, e.length, expected_lengths[p]) for p, e in entries.items()
               if p in expected_lengths and e.length != expected_lengths[p]]
        if bad:
            raise EmbeddingCacheError(
                f"{len(bad)} proteins have a row count that disagrees with the sequence length, "
                f"e.g. {bad[:5]}")

    pids = sorted(entries)
    pq.write_table(pa.table({
        "protein_id": pa.array(pids, pa.string()),
        "shard": pa.array([entries[p].shard for p in pids], pa.string()),
        "row_offset": pa.array([entries[p].row_offset for p in pids], pa.int64()),
        "length": pa.array([entries[p].length for p in pids], pa.int32()),
    }), cache_dir / INDEX_NAME)

    manifest = dict(manifest)
    manifest["n_proteins"] = len(entries)
    manifest["n_rows"] = sum(e.length for e in entries.values())
    with open(cache_dir / MANIFEST_NAME, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


# --------------------------------------------------------------------------- reader


class EmbeddingCache:
    """Random-access reader. Safe to create before DataLoader workers fork: file
    descriptors are opened lazily, per process."""

    def __init__(self, cache_dir):
        import pyarrow.parquet as pq

        self.cache_dir = Path(cache_dir)
        manifest_path, index_path = self.cache_dir / MANIFEST_NAME, self.cache_dir / INDEX_NAME
        if not manifest_path.exists() or not index_path.exists():
            raise EmbeddingCacheError(
                f"incomplete embeddings directory {self.cache_dir}: expected {MANIFEST_NAME} and "
                f"{INDEX_NAME}. Run scripts/precompute_embeddings.py --finalize.")
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.dim = int(self.manifest["dim"])
        self.max_len = int(self.manifest["max_len"])

        cols = ["protein_id", "shard", "row_offset", "length"]
        t = pq.read_table(index_path, columns=cols).to_pydict()
        self.index = {p: IndexEntry(s, o, n) for p, s, o, n in zip(*(t[c] for c in cols))}
        self._fds = {}
        self._owner_pid = os.getpid()

    def __contains__(self, protein_id):
        return protein_id in self.index

    def __len__(self):
        return len(self.index)

    def _fd(self, shard):
        if os.getpid() != self._owner_pid:  # inherited across fork: reopen per process
            for fd in self._fds.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._fds = {}
            self._owner_pid = os.getpid()
        fd = self._fds.get(shard)
        if fd is None:
            fd = self._fds[shard] = os.open(self.cache_dir / shard, os.O_RDONLY)
        return fd

    def get(self, protein_id):
        """[L, dim] fp16 tensor for one protein."""
        entry = self.index.get(protein_id)
        if entry is None:
            raise EmbeddingCacheError(f"{protein_id} is not in {self.cache_dir}")
        out = np.empty((entry.length, self.dim), dtype=np.float16)
        view = out.reshape(-1).view(np.uint8)
        fd = self._fd(entry.shard)
        offset = entry.row_offset * self.dim * 2
        got = 0
        while got < view.nbytes:
            n = os.preadv(fd, [view[got:]], offset + got)
            if n == 0:
                raise EmbeddingCacheError(f"{protein_id}: unexpected end of file in {entry.shard}")
            got += n
        return torch.from_numpy(out)


def verify_cache(cache_dir, required_protein_ids, expect_dim=None, expect_model=None,
                 expect_layer=None, expect_max_len=None, expected_lengths=None,
                 deep_sample=64, verbose=True):
    """Check, before training starts, that the directory can serve this run.

    expected_lengths maps protein_id -> len(protein_tokens). Returns the open cache.
    """
    cache = EmbeddingCache(cache_dir)
    m = cache.manifest

    def check(field, want, got):
        if want is not None and got != want:
            raise EmbeddingCacheError(
                f"embeddings in {cache_dir} were built with {field}={got!r}, this run needs {want!r}")

    check("dim", expect_dim, cache.dim)
    check("model", expect_model, m.get("model"))
    check("layer", expect_layer, m.get("layer"))
    check("max_len", expect_max_len, cache.max_len)
    check("dtype", "float16", m.get("dtype"))

    required = set(required_protein_ids)
    missing = sorted(required - cache.index.keys())
    if missing:
        raise EmbeddingCacheError(
            f"{len(missing)} of {len(required)} proteins are missing from {cache_dir}, "
            f"e.g. {missing[:10]}")
    if expected_lengths:
        bad = [(p, cache.index[p].length, n) for p, n in expected_lengths.items()
               if cache.index[p].length != n]
        if bad:
            raise EmbeddingCacheError(
                f"{len(bad)} proteins have a cached row count that disagrees with protein_tokens "
                f"(the embeddings and the dataset were truncated differently), e.g. {bad[:5]}")
    if deep_sample and required:
        rng = np.random.default_rng(0)
        for pid in rng.choice(sorted(required), size=min(deep_sample, len(required)), replace=False):
            emb = cache.get(str(pid)).float()
            if not torch.isfinite(emb).all():
                raise EmbeddingCacheError(f"{pid}: non-finite cached values")
            if (emb.norm(dim=-1) == 0).any():
                raise EmbeddingCacheError(f"{pid}: all-zero rows (padding written to disk?)")
    if verbose:
        print(f"Embeddings OK: {len(cache)} proteins, dim {cache.dim}, {m.get('model')} "
              f"layer {m.get('layer')}, max_len {cache.max_len}")
    return cache
