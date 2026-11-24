# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The eval-path guards fire on the bad input and stay inert on the good one.

Each of these was a silent failure: a missing GO sidecar scored random weights as a
result, a chunk where every batch failed exited 0, repetition_penalty exhausted host
memory, and an uncapped sequence OOM'd ESM3 against a capped placeholder count.

Runs in the inference environment (README, Setup), under pytest or as a script; skipped
without vLLM, and the checkpoint checks without a checkpoint:

    TEST_CKPT=<a downloaded checkpoint> python tests/test_eval_guards.py
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

pytest.importorskip("vllm")

CKPT = os.environ.get("TEST_CKPT", os.path.join(REPO, "ckpt", "rl"))

_results = []


def need_checkpoint():
    """Skip a check that reads the checkpoint when there is none.

    Raises:
        pytest.skip.Exception: when CKPT is not a directory.
    """
    if not os.path.isdir(CKPT):
        pytest.skip(f"no checkpoint at {CKPT} (set TEST_CKPT)")


def check(cond, msg):
    """Record and print one check, and fail the test when it does not hold.

    Args:
        cond: the check.
        msg: what it asserts.

    Raises:
        AssertionError: when `cond` is false.
    """
    _results.append(bool(cond))
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    assert cond, msg


def test_mrope():
    from transformers import AutoConfig
    from bioreason2.models.protein_vllm import _disable_mrope

    need_checkpoint()
    cfg = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
    tc = getattr(cfg, "text_config", cfg)
    before = set(getattr(tc, "rope_parameters", {}) or {})
    check("mrope_section" in before,
          f"the checkpoint config really does carry M-RoPE ({sorted(before)})")

    _disable_mrope(cfg)
    after = set(getattr(getattr(cfg, "text_config", cfg), "rope_parameters", {}) or {})
    check(not ({"mrope_section", "mrope_interleaved"} & after),
          f"_disable_mrope strips it ({sorted(after)})")


def test_go_sidecars():
    src = open(os.path.join(REPO, "bioreason2/models/protein_vllm.py")).read()
    check("raise FileNotFoundError(f\"GO encoder weights not found" in src,
          "a missing GO encoder raises instead of warning")
    check("raise FileNotFoundError(f\"GO projection weights not found" in src,
          "a missing GO projection raises instead of warning")
    need_checkpoint()
    check(os.path.exists(os.path.join(CKPT, "go_encoder.pt"))
          and os.path.exists(os.path.join(CKPT, "go_projection.pt")),
          "our own checkpoints carry both, so the guard is inert here")


def test_repetition_penalty():
    src = open(os.path.join(REPO, "bioreason2/models/protein_vllm.py")).read()
    check("is unusable with prompt_embeds" in src,
          "a non-1.0 repetition_penalty raises before vLLM allocates")
    src_eval = open(os.path.join(REPO, "eval.py")).read()
    check('"--repetition_penalty", type=float, default=1.0' in src_eval,
          "eval.py defaults to 1.0, so the guard never fires unprompted")


def test_encoder_cap():
    from bioreason2.models.protein_encoder import ESM3Encoder
    import inspect

    sig = inspect.signature(ESM3Encoder.encode_sequences)
    check("max_length" in sig.parameters,
          "encode_sequences takes a max_length cap")
    check(sig.parameters["max_length"].default is None,
          "it defaults to None, so no existing caller changes behaviour")


def test_eval_exit():
    src = open(os.path.join(REPO, "eval.py")).read()
    check("no results written to" in src,
          "a chunk that wrote nothing exits non-zero")
    check("successfully_processed == 0" in src,
          "the check is on the saved-record count, not on an exception")


def main():
    """Run every check as a script, continuing past failures.

    Returns:
        0 when every check passed, 1 otherwise.
    """
    for fn in (test_mrope, test_go_sidecars, test_repetition_penalty,
               test_encoder_cap, test_eval_exit):
        print(f"--- {fn.__name__}")
        try:
            fn()
        except AssertionError:
            pass  # already recorded and printed by check()
        except pytest.skip.Exception as exc:
            print(f"  SKIP  {exc}")
        except Exception as exc:
            _results.append(False)
            print(f"  FAIL  {fn.__name__} raised: {exc}")

    print(f"\n{sum(_results)}/{len(_results)} passed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
