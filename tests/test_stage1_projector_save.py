# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The stage-1 projector handoff survives sharding.

Stage 1 used to save each projector's own `state_dict()`. That is right under DDP
and silently wrong under FSDP, where the projectors sit in the root flat parameter
and a submodule `state_dict()` returns this rank's shard -- a file of the right
name holding a truncated tensor, which stage 2 then loads as its warm start.

The save now slices the whole-model state dict. These checks hold that slice to
exactly what the old code wrote, so the DDP runs are unchanged, and hold the
sibling prefixes apart so `go_projection` cannot absorb `go_encoder`.

    pytest tests/test_stage1_projector_save.py      or      python tests/test_stage1_projector_save.py
"""
import os
import sys

import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from bioreason2.models.model_family import submodule_state_dict

TEXT_HIDDEN = 16
PROTEIN_DIM = 8
GO_DIM = 12
PROJECTORS = ("protein_projection", "go_projection", "go_encoder")

_results = []


def check(cond, msg):
    _results.append(bool(cond))
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    return cond


def build_model():
    """A stand-in carrying the real nesting: projectors hang off an inner `model`.

    Returns:
        Module whose state-dict keys carry the `model.<projector>.` prefixes that
        `STAGE1_ARTIFACTS` slices on.
    """
    inner = nn.Module()
    inner.protein_projection = nn.Sequential(
        nn.Linear(PROTEIN_DIM, TEXT_HIDDEN), nn.GELU(), nn.Linear(TEXT_HIDDEN, TEXT_HIDDEN)
    )
    inner.go_projection = nn.Sequential(nn.Linear(GO_DIM, TEXT_HIDDEN))
    inner.go_encoder = nn.Sequential(nn.Linear(GO_DIM, GO_DIM), nn.Linear(GO_DIM, GO_DIM))

    outer = nn.Module()
    outer.model = inner
    return outer


def main():
    """Run every check and report the tally.

    Returns:
        0 when all checks pass, 1 otherwise.
    """
    torch.manual_seed(23)
    outer = build_model()
    full = outer.state_dict()

    print("the slice reproduces what the pre-FSDP code saved")
    for attr in PROJECTORS:
        direct = getattr(outer.model, attr).state_dict()
        sliced = submodule_state_dict(full, f"model.{attr}.")
        check(set(sliced) == set(direct) and len(sliced) > 0,
              f"{attr}: same {len(direct)} keys, rebased")
        check(all(torch.equal(sliced[k], direct[k]) for k in direct),
              f"{attr}: every tensor identical")

    print("\nthe slice is what stage 2 loads")
    fresh = nn.Sequential(
        nn.Linear(PROTEIN_DIM, TEXT_HIDDEN), nn.GELU(), nn.Linear(TEXT_HIDDEN, TEXT_HIDDEN)
    )
    missing, unexpected = fresh.load_state_dict(
        submodule_state_dict(full, "model.protein_projection."), strict=True
    )
    check(not missing and not unexpected, "loads strict=True with nothing left over")
    check(all(torch.equal(fresh.state_dict()[k], v)
              for k, v in outer.model.protein_projection.state_dict().items()),
          "the loaded weights match the source")

    print("\nprefixes are exact, not substring")
    proj = submodule_state_dict(full, "model.go_projection.")
    enc = submodule_state_dict(full, "model.go_encoder.")
    check(len(proj) == len(outer.model.go_projection.state_dict())
          and len(enc) == len(outer.model.go_encoder.state_dict()),
          "go_projection and go_encoder each get only their own")
    check(len(proj) + len(enc) == sum(1 for k in full if k.startswith("model.go_")),
          "together they account for every model.go_* parameter")

    print("\na missing projector yields nothing, which the caller turns into a raise")
    check(submodule_state_dict(full, "model.no_such_module.") == {},
          "an unmatched prefix returns empty rather than a partial dict")

    print(f"\n{sum(_results)}/{len(_results)} passed")
    return 0 if all(_results) else 1


def test_all_checks() -> None:
    """Every check, under pytest."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
