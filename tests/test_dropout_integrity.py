# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Nothing the dropout is not supposed to touch may change.

The rate tests in `test_sft_question_pipeline.py` answer "does it fire at the
configured rate". This answers the harder question: **when it fires, is the
result still the same prompt minus the withheld lines, and nothing else.**

Both are needed and they fail differently. A dropout that rewrote a surviving
bullet, reordered the block, dropped a blank line or corrupted the GO
speculations would pass every rate check in the other file -- the rates would be
exactly right and the corpus would be wrong.

Every assertion is made against the RENDERED QUESTION, never against the `ctx_*`
columns, because the columns are what the dropout claims it did and the question
is what the model reads. That distinction is not theoretical: a dropout that
updated `prompt` but not `question` once made a 0.76 firing rate read as 0.23.

    BRP_SFT_DIR=<dir> python tests/test_dropout_integrity.py
"""
import os
import re
import sys
import importlib.util

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

SFT_DIR = os.environ.get("BRP_SFT_DIR",
                         os.path.join(REPO, "data", "sft", "v2-features"))
TEMPLATE = os.environ.get(
    "QUESTION_TEMPLATE_PY",
    os.path.join(REPO, "scripts", "data", "question_template.py"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
#: Rows to check. Every one is rendered EPOCHS times and each render is parsed,
#: so the full corpus is 130,452 x 10 = 1.3M parses -- roughly three hours, which
#: would make this a launch blocker rather than a launch gate. The sample is
#: STRIDED, not the first N: the table is written train-then-test in shard order,
#: so a prefix would check one shard and miss whatever is peculiar about the rest.
#: 0 means no cap.
MAX_ROWS = int(os.environ.get("MAX_ROWS", "2000"))

_fail = []


def bad(msg):
    _fail.append(msg)


def load_template():
    spec = importlib.util.spec_from_file_location("qt", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def split_sections(q, T):
    """The rendered question as {head, features_scope, features, spec, ppi, tail}.

    Parsed positionally off the template's own literal text rather than by
    guessing at blank lines, so a change to the template breaks this loudly
    instead of silently matching the wrong span.
    """
    m = re.search(r"^Residue features, (.+?):\n(.*?)\n\nInitial GO term "
                  r"speculations:\n(.*?)\n\nKnown interaction partners:\n(.*?)"
                  r"\n\nReason about the function", q, re.S | re.M)
    if not m:
        return None
    return {
        "head": q[:m.start()],
        "scope": m.group(1),
        "features": m.group(2),
        "spec": m.group(3),
        "ppi": m.group(4),
        "tail": q[m.end():],
    }


def bullets(block, T):
    return [] if block == T.ABSENT else block.splitlines()


def is_ordered_subset(shown, full):
    """`shown` must be `full` with lines deleted -- same lines, same order.

    Catches a rewritten bullet, a reordered block and a duplicated line in one
    check. A plain set comparison would miss the reordering, and a length
    comparison would miss all three.
    """
    it = iter(full)
    return all(any(line == f for f in it) for line in shown)


def main():
    if not os.path.isdir(os.path.join(SFT_DIR, "data")):
        print(f"SKIP: no built dataset at {SFT_DIR}")
        return 0

    from datasets import load_dataset
    from bioreason2.dataset.cafa5 import context_dropout as CD

    T = load_template()
    d = load_dataset(SFT_DIR)["train"]
    cfg = CD.ContextDropoutConfig(ppi_section=0.8, ppi_item=0.5,
                                  features_section=0.5, features_complete=0.1,
                                  features_item=0.25)
    ds = CD.ContextDropoutDataset(d, cfg=cfg, base_seed=23,
                                  question_template_py=TEMPLATE)
    ds.verify_no_op(0)

    idx = list(range(len(d)))
    if MAX_ROWS and len(idx) > MAX_ROWS:
        step = len(idx) / MAX_ROWS
        idx = [int(i * step) for i in range(MAX_ROWS)]
        print(f"sampling {len(idx):,} of {len(d):,} rows, strided "
              f"(MAX_ROWS={MAX_ROWS}; set 0 to check every row)")
    base = {i: split_sections(d[i]["question"], T) for i in idx}
    if any(b is None for b in base.values()):
        print("FAIL: the stored question does not match the template layout")
        return 1

    n = seen = 0
    modes = {"features": {}, "ppi": {}}
    for e in range(EPOCHS):
        ds.epoch_value = type("E", (), {"value": e})()
        for i in idx:
            row, orig = ds[i], base[i]
            got = split_sections(row["question"], T)
            n += 1
            if got is None:
                bad(f"row {i} epoch {e}: rendered question lost its layout")
                continue

            # 1. everything outside the two ablatable sections is byte-identical
            for key in ("head", "spec", "tail"):
                if got[key] != orig[key]:
                    bad(f"row {i} epoch {e}: `{key}` changed\n"
                        f"      was {orig[key][-90:]!r}\n"
                        f"      now {got[key][-90:]!r}")

            # 2. the target is never touched
            if (row["assistant_reasoning"] != d[i]["assistant_reasoning"]
                    or row["assistant_answer"] != d[i]["assistant_answer"]):
                bad(f"row {i} epoch {e}: the TARGET changed")

            # 3. each ablatable section is the source minus whole lines, in order
            for field, key in (("features", "features"), ("ppi", "ppi")):
                shown, full = got[key], orig[key]
                if shown == T.ABSENT:
                    mode = "none"
                elif shown == full:
                    mode = "complete"
                else:
                    mode = "partial"
                    sb, fb = bullets(shown, T), bullets(full, T)
                    if not is_ordered_subset(sb, fb):
                        bad(f"row {i} epoch {e}: {field} is not an ordered "
                            f"subset -- a line was rewritten, reordered or "
                            f"duplicated")
                    if len(sb) >= len(fb):
                        bad(f"row {i} epoch {e}: {field} labelled partial but "
                            f"kept {len(sb)} of {len(fb)} lines")
                    if any(not b.strip() for b in sb):
                        bad(f"row {i} epoch {e}: {field} has a blank line")
                    # A duplicate in the SHOWN block is only a defect if the
                    # dropout INTRODUCED it. It cannot claim more copies of a
                    # line than the source had -- but the source is not
                    # duplicate-free, and asserting that it is made this check
                    # fail on real data for a reason that has nothing to do with
                    # dropout.
                    #
                    # Measured over all 140,735 rows: 174 (0.12%) carry a
                    # repeated partner line, 548 lines of 1,156,215 (0.05%).
                    # A0A060YAF5 ships 10 partner lines, 7 unique -- the same
                    # partner description appears twice under different ids.
                    # `is_ordered_subset` above already passes on those, because
                    # keeping both copies IS an ordered subset. This is the
                    # multiset form of the same claim, and it is the one that
                    # holds.
                    from collections import Counter as _C
                    _cs, _cf = _C(sb), _C(fb)
                    _extra = [b for b, n in _cs.items() if n > _cf.get(b, 0)]
                    if _extra:
                        bad(f"row {i} epoch {e}: {field} shows {_cs[_extra[0]]} "
                            f"copies of a line the source had "
                            f"{_cf.get(_extra[0], 0)} -- dropout duplicated it")
                modes[field][mode] = modes[field].get(mode, 0) + 1
                if full == T.ABSENT and mode != "none":
                    bad(f"row {i} epoch {e}: {field} was already absent but "
                        f"rendered as {mode}")

            # 4. the scope label must describe the block beside it
            want = {"none": T.SCOPE_NONE, "complete": T.SCOPE_ALL,
                    "partial": T.SCOPE_SUBSET}[
                "none" if got["features"] == T.ABSENT else
                "complete" if got["features"] == orig["features"] else "partial"]
            if got["scope"] != want:
                bad(f"row {i} epoch {e}: scope says {got['scope']!r}, block is "
                    f"{want!r}")
            seen += 1

    # --- two properties the loop above cannot see --------------------------
    # 1. DDP ranks must agree. Each rank builds its OWN ContextDropoutDataset, so
    #    "reproducible" has to hold across independent objects, not just across
    #    two reads of one. If it did not, ranks would train on different text for
    #    the same (epoch, index) and the gradients would be averaging over
    #    different prompts without anything reporting it.
    def _fresh():
        return CD.ContextDropoutDataset(
            d, cfg=CD.ContextDropoutConfig(
                ppi_section=0.8, ppi_item=0.5, features_section=0.5,
                features_complete=0.1, features_item=0.25),
            base_seed=23, question_template_py=TEMPLATE)

    a, b = _fresh(), _fresh()
    probe = idx[:min(80, len(idx))]
    for o in (a, b):
        o.epoch_value = type("E", (), {"value": 7})()
    if any(a[i]["question"] != b[i]["question"] for i in probe):
        bad("two independently constructed datasets disagree at the same "
            "(epoch, index) -- DDP ranks would train on different prompts")
    a.epoch_value = type("E", (), {"value": 8})()
    if all(a[i]["question"] == b[i]["question"] for i in probe):
        bad("epoch 8 is identical to epoch 7 -- the mask is not re-rolling")

    print(f"checked {n:,} rendered questions over {EPOCHS} epochs "
          f"({len(idx):,} of {len(d):,} rows), plus DDP agreement")
    for field, m in modes.items():
        tot = sum(m.values()) or 1
        print(f"  {field:9s} " + "  ".join(
            f"{k}={v/tot:.3f}" for k, v in sorted(m.items())))
    if _fail:
        print(f"\n{len(_fail)} FAILURES (first 12):")
        for f in _fail[:12]:
            print("  " + f)
        return 1
    print("\nno violations: outside the two ablatable sections every rendered "
          "question is byte-identical to the corpus, and every surviving line "
          "is an unmodified line of the original in its original order")
    return 0


if __name__ == "__main__":
    sys.exit(main())
