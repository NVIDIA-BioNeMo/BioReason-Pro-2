# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Structure dropout fires at its configured rate and touches nothing else.

The structure reaches ESM3 rather than the prompt, so it is dropped in
`ContextDropoutDataset.__getitem__` by nulling `structure_path`. Two things have to
hold: the rate is what the config says on rows that HAVE a structure, and rows that
have none are unaffected. The question must be byte-identical either way.

    pytest tests/test_structure_dropout.py      or      python tests/test_structure_dropout.py
"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

N_ROWS = 400
EPOCHS = 10
RATE = 0.5

# Written rather than imported from scripts/data/question_template.py: this tests the wrapper,
# not the published prompt.
TEMPLATE_SRC = '''
ABSENT = "None annotated"
SCOPE_ALL = "every annotated feature"
SCOPE_SUBSET = "a subset, others withheld"
SCOPE_NONE = "none supplied"
PROMPT = """From organism {organism}.

Residue features, {features_scope}:
{features}

Initial GO term speculations:
{speculations}

Known interaction partners:
{partners}

Reason about the function."""
FIELDS = ("organism", "features", "features_scope", "speculations", "partners")
'''

_results = []


def check(cond, msg):
    _results.append(bool(cond))
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    return cond


def build_rows(template):
    rows = []
    for i in range(N_ROWS):
        ctx = {
            "organism": "Homo sapiens",
            "features": f"- 1-{10 + i} (DOMAIN): Globin [IPR000971]",
            "features_scope": template.SCOPE_ALL,
            "speculations": "Molecular Function (MF): GO:0003674 (molecular function)",
            "partners": template.ABSENT,
        }
        rows.append({
            "protein_id": f"P{i:05d}",
            "question": template.PROMPT.format(**ctx),
            "sequence": "MKTLLILAVV",
            # Every third row genuinely has no structure, as ~6% of the corpus does.
            "structure_path": None if i % 3 == 0 else f"/tmp/AF-P{i:05d}-F1.pdb",
            **{f"ctx_{k}": v for k, v in ctx.items()},
        })
    return rows


def main():
    import importlib.util
    from bioreason2.dataset.cafa5.context_dropout import (
        ContextDropoutConfig, ContextDropoutDataset)

    tmp = tempfile.mkdtemp()
    TEMPLATE = os.path.join(tmp, "question_template.py")
    with open(TEMPLATE, "w") as fh:
        fh.write(TEMPLATE_SRC)

    spec = importlib.util.spec_from_file_location("qt", TEMPLATE)
    template = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(template)

    rows = build_rows(template)
    cfg = ContextDropoutConfig(structure_section=RATE)
    check(cfg.enabled(), "structure_section alone makes the config enabled")

    ds = ContextDropoutDataset(rows, cfg=cfg, base_seed=23,
                               question_template_py=TEMPLATE)

    with_struct = kept = dropped = 0
    without_struct_changed = 0
    question_changed = 0
    for e in range(EPOCHS):
        ds.epoch_value = type("E", (), {"value": e})()
        for i, row in enumerate(rows):
            out = ds[i]
            if out["question"] != row["question"]:
                question_changed += 1
            if row["structure_path"] is None:
                if out["structure_path"] is not None:
                    without_struct_changed += 1
                continue
            with_struct += 1
            if out["structure_path"] is None:
                dropped += 1
            else:
                kept += 1

    rate = dropped / with_struct
    print(f"  accesses with a structure {with_struct:,}: kept {kept:,}, dropped {dropped:,}")
    check(abs(rate - RATE) < 0.04, f"realised drop rate {rate:.3f} (target {RATE})")
    check(without_struct_changed == 0, "rows with no structure are never given one")
    check(question_changed == 0, "the question is byte-identical -- text is untouched")

    ds.epoch_value = type("E", (), {"value": 0})()
    first = [ds[i]["structure_path"] for i in range(N_ROWS)]
    again = [ds[i]["structure_path"] for i in range(N_ROWS)]
    check(first == again, "the mask is reproducible for a given (epoch, index)")

    ds.epoch_value = type("E", (), {"value": 1})()
    check(first != [ds[i]["structure_path"] for i in range(N_ROWS)],
          "the mask changes between epochs")

    off = ContextDropoutDataset(rows, cfg=ContextDropoutConfig(structure_section=0.0),
                                base_seed=23, question_template_py=TEMPLATE)
    check(all(off[i]["structure_path"] == rows[i]["structure_path"]
              for i in range(N_ROWS)),
          "rate 0.0 is an exact no-op")

    print(f"\n{sum(_results)}/{len(_results)} passed")
    return 0 if all(_results) else 1


def test_all_checks() -> None:
    """Every check, under pytest."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
