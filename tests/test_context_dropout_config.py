# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""ContextDropoutConfig: inert by default, enabled by any knob, rejects impossible splits.

train_protein_llm.py calls enabled() unconditionally at startup, so a config that raises
or misreports breaks every run, not just dropout runs. These tests pin the contract.

Run:  python tests/test_context_dropout_config.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from bioreason2.dataset.cafa5.context_dropout import ContextDropoutConfig


def test_default_config_is_disabled_and_does_not_raise():
    """Every probability at 0.0 is an exact no-op."""
    cfg = ContextDropoutConfig()
    assert cfg.enabled() is False, "a default config must be inert"


def test_every_knob_enables():
    """Any single non-zero knob turns dropout on."""
    for name in ("ppi_section", "ppi_item", "ppi_complete", "gogpt_section", "gogpt_item",
                 "features_section", "features_complete", "features_item",
                 "structure_section"):
        assert ContextDropoutConfig(**{name: 0.1}).enabled() is True, name


def test_impossible_split_raises():
    """none + complete above 1 would silently starve the partial branch."""
    for none_k, comp_k in (("ppi_section", "ppi_complete"),
                           ("features_section", "features_complete")):
        try:
            ContextDropoutConfig(**{none_k: 0.8, comp_k: 0.3})
        except ValueError:
            continue
        raise AssertionError(f"{none_k} + {comp_k} > 1 did not raise")


def test_describe_names_every_field():
    """describe() is the run log's record of the recipe."""
    text = ContextDropoutConfig(ppi_section=0.8, structure_section=0.5).describe()
    assert "ppi_section=0.8" in text and "structure_section=0.5" in text, text


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS  %s" % name)
            except AssertionError as e:
                fails += 1; print("FAIL  %s\n        %s" % (name, e))
            except Exception as e:
                fails += 1; print("ERROR %s\n        %s: %s" % (name, type(e).__name__, e))
    print("\n%s" % ("ALL PASS" if not fails else "%d FAILED" % fails))
    sys.exit(1 if fails else 0)
