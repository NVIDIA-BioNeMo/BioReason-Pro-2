# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Precompute ESM2 embeddings for every wanglab/bioreason-pro-v2 protein (optional).

ESM2 is frozen, so training can read its embeddings from disk instead of running it in
every step (scripts/train.py picks them up through data.embedding_cache_dir). This is
an optimization: training without it runs ESM2 on the fly and is otherwise identical.
Needs ~370 GB of disk for the full dataset (fp16, 2000-residue truncation).

One process per GPU, then merge:

    python scripts/precompute_embeddings.py --shard-index 0 --num-shards 4   # ... 1, 2, 3
    python scripts/precompute_embeddings.py --finalize
    python scripts/precompute_embeddings.py --check                        # optional

Re-running a killed shard resumes where it stopped. Embeddings are hidden_states[27] of
ESM2-650M without special tokens, computed under bf16 autocast on GPU and stored as
fp16: the same computation as gogpt.models.protein_encoder.ESM2Encoder.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gogpt.data.precomputed_embeddings import (  # noqa: E402
    EmbeddingShardWriter, finalize, read_rank_meta, verify_cache)
from gogpt.models.protein_encoder import ESM2Encoder  # noqa: E402


def read_proteins(data_dir, fasta=None):
    """{protein_id: sequence} over every split of proteins/ and proteins_extra/ (+ FASTA)."""
    import pyarrow.parquet as pq

    proteins = {}

    def add(pid, seq):
        if proteins.setdefault(pid, seq) != seq:
            raise SystemExit(f"{pid} appears with two different sequences")

    for sub in ("proteins", "proteins_extra"):
        files = sorted((data_dir / sub).glob("*.parquet"))
        if not files:
            raise SystemExit(f"no parquet files under {data_dir / sub}")
        for f in files:
            t = pq.read_table(f, columns=["protein_id", "sequence"]).to_pydict()
            for pid, seq in zip(t["protein_id"], t["sequence"]):
                if seq:  # deleted UniProt accessions have no sequence
                    add(pid, seq)
    if fasta:
        pid, chunks = None, []
        for line in open(fasta):
            line = line.strip()
            if line.startswith(">"):
                if pid:
                    add(pid, "".join(chunks))
                pid, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line)
        if pid:
            add(pid, "".join(chunks))
    return proteins


def embed_batch(encoder, sequences, device):
    ids, mask = encoder.batch_tokens(sequences, device=device)
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda"
                else torch.autocast("cpu", enabled=False))
    with autocast:
        hidden = encoder(ids, mask).float()
    return [hidden[i, : int(mask[i].sum())].cpu() for i in range(len(sequences))]


def self_test(encoder, sequences, device):
    """The shortest proteins embedded alone and as one padded batch must agree: padding
    must not leak into real positions."""
    solo = [embed_batch(encoder, [s], device)[0] for s in sequences]
    batched = embed_batch(encoder, sequences, device)
    tol = 3e-2 if device.type == "cuda" else 1e-4  # bf16 kernels vary with batch shape
    for s, b in zip(solo, batched):
        rel = ((s - b).norm() / s.norm()).item()
        if rel >= tol:
            raise SystemExit(f"self-test failed: solo vs batched relative L2 {rel:.2e} >= {tol}")
    print(f"self-test passed on {len(sequences)} proteins")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None,
                    help="local copy of wanglab/bioreason-pro-v2 (default: download from the Hub)")
    ap.add_argument("--fasta", default=None, help="extra proteins to embed, e.g. an evaluation set")
    ap.add_argument("--dataset", default=None,
                    help="only embed the proteins of this prepared dataset (scripts/prepare_data.py "
                         "output_dir), e.g. artifacts/preprocessed/v2")
    ap.add_argument("--output-dir", default="artifacts/embeddings/esm2_t33_650M_UR50D_l27")
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--layer", type=int, default=27, help="index into ESM2 hidden_states (0 = embeddings)")
    ap.add_argument("--max-len", type=int, default=2000, help="must match max_protein_length")
    ap.add_argument("--max-tokens-per-batch", type=int, default=8192)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="embed at most N proteins (smoke test)")
    ap.add_argument("--finalize", action="store_true", help="merge the shards' indices, then exit")
    ap.add_argument("--check", action="store_true", help="verify a finalized directory, then exit")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        from huggingface_hub import snapshot_download

        data_dir = Path(snapshot_download("wanglab/bioreason-pro-v2", repo_type="dataset",
                                          allow_patterns=["proteins/*.parquet", "proteins_extra/*.parquet"]))
    proteins = read_proteins(data_dir, args.fasta)
    if args.dataset:
        from datasets import load_from_disk

        ds_dir = Path(args.dataset) if Path(args.dataset).is_absolute() else ROOT / args.dataset
        ds = load_from_disk(str(ds_dir / "dataset"))
        keep = set().union(*(set(split["protein_id"]) for split in ds.values()))
        missing = keep - proteins.keys()
        if missing:
            raise SystemExit(f"{len(missing)} proteins of {args.dataset} have no sequence, e.g. {sorted(missing)[:5]}")
        proteins = {p: s for p, s in proteins.items() if p in keep}
    lengths = {p: min(len(s), args.max_len) for p, s in proteins.items()}
    manifest = {"model": args.model, "layer": args.layer, "max_len": args.max_len, "dtype": "float16"}

    if args.finalize:
        meta = read_rank_meta(out_dir)
        for key in ("model", "layer", "max_len"):
            if meta[key] != manifest[key]:
                raise SystemExit(f"shards were written with {key}={meta[key]!r}, arguments say {manifest[key]!r}")
        print(json.dumps(finalize(out_dir, {**manifest, **meta}, lengths), indent=2, sort_keys=True))
        return
    if args.check:
        verify_cache(out_dir, proteins, expect_model=args.model, expect_layer=args.layer,
                     expect_max_len=args.max_len, expected_lengths=lengths)
        return

    device = torch.device(args.device)
    encoder = ESM2Encoder(args.model, args.layer, args.max_len).to(device)
    from transformers import __version__ as transformers_version

    writer = EmbeddingShardWriter(out_dir, args.shard_index, encoder.hidden_size,
                                  meta={**manifest, "transformers_version": transformers_version})
    # Longest first, so an out-of-memory batch shows up at once; round-robin over the
    # length-sorted list gives every shard the same length distribution.
    ordered = sorted(proteins, key=lambda p: (-len(proteins[p]), p))
    mine = ordered[args.shard_index::args.num_shards][: args.limit]
    done = set()
    for idx in out_dir.glob("index_r*.jsonl"):  # proteins written by any shard
        for line in open(idx):
            try:
                done.add(json.loads(line)["protein_id"])
            except (json.JSONDecodeError, KeyError):
                break
    todo = [p for p in mine if p not in done]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(mine):,} proteins, {len(todo):,} to embed")
    if not todo:
        return
    self_test(encoder, [proteins[p][: args.max_len] for p in sorted(todo, key=lambda p: len(proteins[p]))[:3]], device)

    from tqdm import tqdm

    batches, current = [], []
    for pid in sorted(todo, key=lambda p: -lengths[p]):
        if current and lengths[current[0]] * (len(current) + 1) > args.max_tokens_per_batch:
            batches.append(current)
            current = []
        current.append(pid)
    if current:
        batches.append(current)

    with writer, tqdm(total=len(todo), unit="protein") as bar:
        pending = list(reversed(batches))
        while pending:
            batch = pending.pop()
            try:
                embs = embed_batch(encoder, [proteins[p] for p in batch], device)
            except torch.cuda.OutOfMemoryError:
                if len(batch) == 1:
                    raise
                torch.cuda.empty_cache()
                mid = len(batch) // 2
                pending += [batch[mid:], batch[:mid]]
                continue
            for pid, emb in zip(batch, embs):
                arr = emb.numpy().astype(np.float16)
                if arr.shape[0] != lengths[pid]:
                    raise SystemExit(f"{pid}: {arr.shape[0]} rows, expected {lengths[pid]}")
                writer.write(pid, arr)
            bar.update(len(batch))
    print(f"done; when every shard has finished run: python {Path(__file__).name} --finalize")


if __name__ == "__main__":
    main()
