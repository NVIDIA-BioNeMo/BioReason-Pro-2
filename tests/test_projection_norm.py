# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Each projection-norm variant changes the one thing it is supposed to change.

The variants separate two effects `nn.LayerNorm` bundles together -- output scale and
per-residue norm variation -- which only means anything if each variant really is
isolated. That is asserted here rather than assumed. The released models use `blocknorm`.

    pytest tests/test_projection_norm.py      or      python tests/test_projection_norm.py
"""
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from bioreason2.models import projection_norm as pn  # noqa: E402

#: Width and length of the synthetic protein block.
HIDDEN = 64
RESIDUES = 400

#: The measured baseline: projected blocks sit near RMS 7.4 with a 0.25 coefficient of
#: variation across residues, and the magnitude ramps along the chain.
BASELINE_RMS = 7.4
RAMP = (0.5, 1.5)

_results = []


def check(cond: bool, msg: str) -> bool:
    """Record and print one assertion.

    Args:
        cond: the condition.
        msg: what it asserts.

    Returns:
        The condition, so a caller can branch on it.
    """
    _results.append(bool(cond))
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    return bool(cond)


def block(ramped: bool) -> torch.Tensor:
    """A synthetic projected protein block.

    Args:
        ramped: whether per-residue magnitude ramps along the chain, as the trained
            baseline's does.

    Returns:
        (RESIDUES, HIDDEN) tensor at roughly ``BASELINE_RMS``.
    """
    torch.manual_seed(0)
    out = torch.randn(RESIDUES, HIDDEN) * BASELINE_RMS
    if ramped:
        out = out * torch.linspace(RAMP[0], RAMP[1], RESIDUES).unsqueeze(1)
    return out


def stats(tensor: torch.Tensor) -> tuple:
    """Mean per-residue RMS and its coefficient of variation.

    Args:
        tensor: (residues, hidden) block.

    Returns:
        (mean RMS, CV).
    """
    rms = tensor.pow(2).mean(dim=-1).sqrt()
    return float(rms.mean()), float(rms.std() / rms.mean())


def main() -> int:
    """Run every assertion and report.

    Returns:
        0 if all pass, 1 otherwise.
    """
    check(pn.resolve("", False) == "none", "an unset variant with the flag off is `none`")
    check(pn.resolve("", True) == "layernorm", "an unset variant with the flag on is `layernorm`")
    check(pn.build("none", HIDDEN) is None, "`none` adds no module, so old checkpoints load")
    try:
        pn.resolve("blocknorm", True)
        check(False, "a contradictory pair raises")
    except ValueError:
        check(True, "a contradictory pair raises")

    flat = block(ramped=False)
    ramped = block(ramped=True)
    base_rms, base_cv = stats(ramped)
    print("  input: RMS %.2f, CV %.3f" % (base_rms, base_cv))

    out = {name: (stats(pn.build(name, HIDDEN)(flat)),
                  stats(pn.build(name, HIDDEN)(ramped)))
           for name in ("layernorm", "blocknorm", "lnscale")}
    for name, ((frms, fcv), (rrms, rcv)) in out.items():
        print("  %-10s flat RMS %.2f CV %.3f | ramped RMS %.2f CV %.3f"
              % (name, frms, fcv, rrms, rcv))

    (ln_rms, _), (_, ln_cv) = out["layernorm"]
    (bn_rms, _), (_, bn_cv) = out["blocknorm"]
    (ls_rms, _), (_, ls_cv) = out["lnscale"]

    check(abs(ln_rms - 1.0) < 0.05, "layernorm lands at RMS 1")
    check(ln_cv < 0.02, "layernorm destroys the ramp (CV under 0.02)")

    check(abs(bn_rms - 1.0) < 0.05, "blocknorm lands at layernorm's scale")
    check(abs(bn_cv - base_cv) < 0.02,
          "blocknorm preserves the ramp, so it isolates variance from scale")

    check(abs(ls_rms - 7.4) < 0.4, "lnscale lands at the baseline's scale")
    check(ls_cv < 0.02, "lnscale destroys the ramp, so it isolates scale from variance")

    scalar = pn.build("blocknorm", HIDDEN)
    check(sum(p.numel() for p in scalar.parameters()) == 1,
          "blocknorm adds exactly one parameter")

    doubled = pn.build("blocknorm", HIDDEN)(ramped * 3.0)
    check(abs(stats(doubled)[0] - out["blocknorm"][1][0]) < 1e-3,
          "blocknorm is scale-invariant in its input, as a normaliser must be")

    pn.check_loaded(["text_model.lm_head.weight"], ["go_encoder.x"])
    check(True, "check_loaded ignores mismatches outside the projection")
    for missing, unexpected in ((["protein_projection.3.weight"], []),
                                ([], ["model.protein_projection.3.norm.weight"])):
        try:
            pn.check_loaded(missing, unexpected)
            check(False, "check_loaded raises on %s" % (missing or unexpected))
        except RuntimeError:
            check(True, "check_loaded raises on %s" % (missing or unexpected))

    print("\n%d/%d passed" % (sum(_results), len(_results)))
    return 0 if all(_results) else 1


def test_all_checks() -> None:
    """Every check, under pytest."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
