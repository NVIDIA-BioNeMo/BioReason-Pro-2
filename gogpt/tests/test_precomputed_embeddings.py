# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import torch

from gogpt.data.precomputed_embeddings import (
    EmbeddingCache, EmbeddingCacheError, EmbeddingShardWriter, finalize, read_rank_meta, verify_cache)

META = {"model": "m", "layer": 27, "max_len": 2000, "dtype": "float16"}


def write(dirpath, rank, items, max_shard_bytes=16 << 30):
    with EmbeddingShardWriter(dirpath, rank, 4, META, max_shard_bytes=max_shard_bytes) as w:
        for pid, arr in items.items():
            w.write(pid, arr)


def rand(n, seed):
    return np.random.default_rng(seed).standard_normal((n, 4)).astype(np.float16)


def test_round_trip_across_ranks_and_shards(tmp_path):
    a = {f"P{i}": rand(i + 1, i) for i in range(5)}
    b = {f"Q{i}": rand(2 * i + 1, 10 + i) for i in range(4)}
    write(tmp_path, 0, a, max_shard_bytes=40)  # tiny shards: forces several files
    write(tmp_path, 1, b)
    assert len(list(tmp_path.glob("emb_r0_*.bin"))) > 1
    manifest = finalize(tmp_path, {**read_rank_meta(tmp_path)},
                        expected_lengths={p: v.shape[0] for p, v in {**a, **b}.items()})
    assert manifest["n_proteins"] == 9
    cache = verify_cache(tmp_path, list(a) + list(b), expect_dim=4, expect_layer=27, verbose=False)
    for pid, arr in {**a, **b}.items():
        assert torch.equal(cache.get(pid), torch.from_numpy(arr))


def test_resume_truncates_unindexed_tail(tmp_path):
    write(tmp_path, 0, {"P0": rand(3, 0)})
    shard = next(tmp_path.glob("emb_r0_*.bin"))
    with open(shard, "ab") as f:
        f.write(b"\x00" * 16)  # bytes from a write that never reached the index
    write(tmp_path, 0, {"P1": rand(2, 1)})
    finalize(tmp_path, read_rank_meta(tmp_path))
    cache = EmbeddingCache(tmp_path)
    assert torch.equal(cache.get("P1"), torch.from_numpy(rand(2, 1)))


def test_resume_with_different_settings_is_refused(tmp_path):
    write(tmp_path, 0, {"P0": rand(3, 0)})
    with pytest.raises(EmbeddingCacheError):
        EmbeddingShardWriter(tmp_path, 0, 4, {**META, "layer": 26})


def test_verify_catches_problems(tmp_path):
    write(tmp_path, 0, {"P0": rand(3, 0)})
    finalize(tmp_path, read_rank_meta(tmp_path))
    with pytest.raises(EmbeddingCacheError, match="missing"):
        verify_cache(tmp_path, ["P0", "P9"], verbose=False)
    with pytest.raises(EmbeddingCacheError, match="row count"):
        verify_cache(tmp_path, ["P0"], expected_lengths={"P0": 4}, verbose=False)
    with pytest.raises(EmbeddingCacheError, match="layer"):
        verify_cache(tmp_path, ["P0"], expect_layer=30, verbose=False)


def test_writer_rejects_bad_input(tmp_path):
    with EmbeddingShardWriter(tmp_path, 0, 4, META) as w:
        with pytest.raises(ValueError):
            w.write("P0", rand(3, 0).astype(np.float32))
        bad = rand(3, 0)
        bad[0, 0] = np.inf
        with pytest.raises(ValueError):
            w.write("P1", bad)
