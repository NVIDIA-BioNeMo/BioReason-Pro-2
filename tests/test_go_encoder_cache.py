# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The GO encoder's two inputs are constants; they must not be rebuilt per step.

`_get_unified_edge_index()` and `_get_precomputed_embeddings(all_go_terms)` depend
only on things fixed at construction, yet were recomputed on every forward -- 183 ms
and 164 ms respectively, against 8 ms for the entire GAT + cross-attention that
consumes them. That is ~21% of a 1.85 s training step spent rebuilding constants.

These tests pin the memoisation: same values, same dtypes, correct device handling,
and no growth in the checkpoint.

Run:  pytest tests/test_go_encoder_cache.py   or   python tests/test_go_encoder_cache.py
Needs the GO embeddings (GO_EMBEDDINGS_PATH, default assets/go_embeddings), and a GPU for the
device test; each is skipped without them.
"""
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from bioreason2.models.go_graph_encoder import create_go_graph_encoder_pipeline

OBO = os.path.join(REPO, "bioreason2", "dataset", "go-basic.obo")
EMB = os.environ.get("GO_EMBEDDINGS_PATH", os.path.join(REPO, "assets", "go_embeddings"))


def _mk():
    """The released GO encoder configuration, built from the downloaded embeddings.

    Returns:
        The encoder.
    """
    return create_go_graph_encoder_pipeline(
        go_obo_path=OBO, precomputed_embeddings_path=EMB,
        hidden_dim=512, num_gat_layers=3, num_heads=8,
        num_reduced_embeddings=200, embedding_dim=2560,
    )


@pytest.fixture(scope="module")
def go():
    """One encoder for the module, skipped when the GO embeddings are absent.

    Returns:
        The encoder.
    """
    if not os.path.isdir(EMB):
        pytest.skip(f"no GO embeddings at {EMB}")
    return _mk()


def test_edge_index_is_identical_and_cached(go):
    a = go._get_unified_edge_index()
    b = go._get_unified_edge_index()
    assert torch.equal(a, b), "edge index changed between calls"
    assert a is b, "edge index was rebuilt instead of returned from cache"
    assert a.dtype == torch.long


def test_embeddings_identical_and_cached(go):
    a = go._get_precomputed_embeddings(go.all_go_terms)
    b = go._get_precomputed_embeddings(go.all_go_terms)
    assert torch.equal(a, b), "gathered embeddings changed between calls"
    assert a is b, "embeddings were re-gathered instead of returned from cache"
    assert a.shape[0] == len(go.all_go_terms)


def test_cached_values_match_an_uncached_rebuild(go):
    """The cache must return what the original code path would have produced."""
    cached_x = go._get_precomputed_embeddings(go.all_go_terms)
    cached_e = go._get_unified_edge_index()
    # force the slow path by passing an equal-but-distinct list (identity check fails)
    fresh_x = go._get_precomputed_embeddings(list(go.all_go_terms))
    assert torch.equal(cached_x, fresh_x), "cache diverged from a fresh gather"
    assert cached_x.dtype == fresh_x.dtype, "cache changed the dtype"
    go._cached_edge_index = None
    fresh_e = go._get_unified_edge_index()
    assert torch.equal(cached_e, fresh_e), "cache diverged from a fresh edge build"


def test_partial_term_list_is_not_cached(go):
    """Only the whole-graph case is a constant; a subset must not hit the cache."""
    subset = list(go.all_go_terms[:10])
    out = go._get_precomputed_embeddings(subset)
    assert out.shape[0] == 10, "a subset request was served from the whole-graph cache"


def test_cache_not_in_state_dict(go):
    """221 MB must not land in every checkpoint, and old checkpoints must still load."""
    go._get_precomputed_embeddings(go.all_go_terms)
    go._get_unified_edge_index()
    keys = list(go.state_dict().keys())
    assert not [k for k in keys if "_cached" in k], \
        "cached constants leaked into the state dict: %s" % [k for k in keys if "_cached" in k]


def test_cache_follows_a_device_move(go):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    cpu_x = go._get_precomputed_embeddings(go.all_go_terms)
    assert cpu_x.device.type == "cpu"
    go.to("cuda")
    gpu_x = go._get_precomputed_embeddings(go.all_go_terms)
    assert gpu_x.device.type == "cuda", "cache returned a CPU tensor after .to('cuda')"
    assert torch.equal(cpu_x.cuda(), gpu_x), "values changed across the device move"
    gpu_e = go._get_unified_edge_index()
    assert gpu_e.device.type == "cuda", "edge index stayed on CPU after .to('cuda')"
    go.to("cpu")


if __name__ == "__main__":
    print("building the GO encoder (43,248 terms)...")
    _go = _mk()
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(_go); print("PASS  %s" % name)
            except pytest.skip.Exception as e:
                print("SKIP  %s (%s)" % (name, e))
            except AssertionError as e:
                fails += 1; print("FAIL  %s\n        %s" % (name, e))
            except Exception as e:
                fails += 1; print("ERROR %s\n        %s: %s" % (name, type(e).__name__, e))
    print("\n%s" % ("ALL PASS" if not fails else "%d FAILED" % fails))
    sys.exit(1 if fails else 0)
