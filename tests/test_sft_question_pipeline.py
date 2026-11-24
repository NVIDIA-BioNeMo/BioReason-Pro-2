# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The pre-rendered question must survive the whole data path unchanged.

`scripts/data/build_question.py` renders the student's question ONCE into
an SFT table, so training and evaluation read the same bytes for a row and cannot
diverge. That is the property this file pins, end to end:

    parquet -> load_dataset -> _format_reasoning_prompt -> format_cafa5_for_protein_llm

plus the thing that is allowed to change it -- context dropout at train time --
and the one thing that must never: a label column reaching the prompt.

It is worth pinning because the failure mode is silent in every direction. A
divergence between the training and eval prompt drops Fmax with no error. A
dropout that no-ops leaves a run training at full context while its config says
80%. And a stale `datasets` cache serves the previous build's schema without
saying so.

Run against a built dataset:

    BRP_SFT_DIR=<repo>/data/sft/v2-features python tests/test_sft_question_pipeline.py

Skips cleanly when that directory does not exist, so it costs nothing in a
checkout that has not built one.
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

SFT_DIR = os.environ.get("BRP_SFT_DIR",
                         os.path.join(REPO, "data", "sft", "v2-features"))
TEMPLATE = os.environ.get(
    "QUESTION_TEMPLATE_PY",
    os.path.join(REPO, "scripts", "data", "question_template.py"))

_results = []


def check(cond, msg):
    _results.append(bool(cond))
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    return cond


def _section(question, header, scope=False):
    """The body of one labelled section of a rendered question.

    Read back out of the finished prompt rather than off the row, deliberately:
    the row is what the dropout *says* it produced, and the prompt is what the
    model actually sees. Every assertion about a rate below is made against the
    prompt, so a dropout that updated `ctx_*` without reaching the question --
    the exact bug that made an 0.76 drop rate read as 0.23 -- fails here.
    """
    if scope:                       # the feature block, whose header varies
        start = question.index("Residue features, ")
        start = question.index(":\n", start) + 2
    else:
        start = question.index(header) + len(header) + 1
    end = question.index("\n\n", start)
    return question[start:end]


def main():
    if not os.path.isdir(os.path.join(SFT_DIR, "data")):
        print(f"SKIP: no built dataset at {SFT_DIR}")
        return 0

    from datasets import load_dataset
    from bioreason2.dataset.cafa5.load import _format_reasoning_prompt
    from bioreason2.dataset.cafa5.format import format_cafa5_for_protein_llm
    from bioreason2.dataset.cafa5 import context_dropout as CD

    d = load_dataset(SFT_DIR)
    check("train" in d and d["train"].num_rows,
          f"load_dataset -> { {k: v.num_rows for k, v in d.items()} }")
    row = dict(d["train"][0])

    # --- the question survives the loader untouched ------------------------
    p = _format_reasoning_prompt(row)["prompt"]
    check(p["user"] == row["question"] and p["system"] == "",
          "load.py uses the stored question verbatim, no system turn")
    check(p["assistant_answer"] == row["assistant_answer"],
          "target is the teacher's final_answer, unspliced")

    chat = format_cafa5_for_protein_llm(_format_reasoning_prompt(row))
    text = [c for c in chat["prompt"][0]["content"] if c["type"] == "text"][0]["text"]
    # An unconditional "system\n\nuser" join would prepend a blank line to every
    # prompt in the corpus once the system turn is empty.
    check(text == row["question"],
          "format.py user text == question exactly, no empty-system blank line")
    check([c["type"] for c in chat["prompt"][0]["content"]]
          == ["protein", "go_graph", "text"],
          "protein and go_graph placeholders precede the text")

    # --- no label reaches the prompt ---------------------------------------
    supplied = set(re.findall(r"GO:\d{7}", row["ctx_speculations"]))
    extra = set(re.findall(r"GO:\d{7}", text)) - supplied
    check(not extra, f"no GO id in the question beyond go_pred ({len(supplied)} supplied)")

    # --- the scope label ---------------------------------------------------
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("qt", TEMPLATE)
    T = _ilu.module_from_spec(_spec); _spec.loader.exec_module(T)
    ARM_HAS_FEATURES = any(r["ctx_features"] != T.ABSENT
                           for r in list(d["train"])[:50])
    check(f"Residue features, {row['ctx_features_scope']}:" in row["question"],
          f"the corpus render is labelled {row['ctx_features_scope']!r}")
    _n = min(len(d["train"]), 20000)
    # A template without SCOPE_ALL filters its block before rendering, so a
    # complete draw is still a subset. Same fallback the loader uses.
    _scope_present = getattr(T, "SCOPE_ALL", None) or T.SCOPE_SUBSET
    check(all(d["train"][i]["ctx_features_scope"] ==
              (T.SCOPE_NONE if d["train"][i]["ctx_features"] == T.ABSENT
               else _scope_present)
              for i in range(0, len(d["train"]), max(1, len(d["train"]) // _n))),
          "every stored scope matches its stored block (present / none)")

    # --- dropout: rebuilt from ctx_*, not edited as text -------------------
    cfg = CD.ContextDropoutConfig(ppi_section=0.8, ppi_item=0.5,
                                  features_section=0.5, features_complete=0.1,
                                  features_item=0.25)
    ds = CD.ContextDropoutDataset(d["train"], cfg=cfg, base_seed=23,
                                  question_template_py=TEMPLATE)
    ds.verify_no_op(0)
    check(True, "verify_no_op: rebuilding from ctx_* reproduces `question` exactly")

    # --- partners: 80% none / 20% partial@0.5, no complete branch ----------
    # Strided cap, same reasoning as test_dropout_integrity: 12 epochs over every
    # partner-carrying row is ~1.2M renders on the real corpus.
    CAP = int(os.environ.get("MAX_ROWS", "1500"))
    _all = [i for i, r in enumerate(d["train"]) if r["ctx_partners"] != T.ABSENT]
    have = (_all if len(_all) <= CAP
            else [_all[int(i * len(_all) / CAP)] for i in range(CAP)])
    if len(have) < len(_all):
        print(f"  (sampling {len(have):,} of {len(_all):,} partner-carrying rows)")
    EPOCHS = 12
    modes = {"none": 0, "complete": 0, "partial": 0}
    for e in range(EPOCHS):
        ds.epoch_value = type("E", (), {"value": e})()
        for i in have:
            shown = _section(ds[i]["question"], "Known interaction partners:")
            full = d["train"][i]["ctx_partners"]
            modes["none" if shown == T.ABSENT else
                  "complete" if shown == full else "partial"] += 1
    tot = sum(modes.values())
    none_r, comp_r = modes["none"] / tot, modes["complete"] / tot
    check(0.70 <= none_r <= 0.90,
          f"partners: none {none_r:.2f} (target 0.80, structural rows excluded)")
    check(comp_r <= 0.12,
          f"partners: complete {comp_r:.2f} -- no complete branch, so only "
          f"short lists that survived p=0.5 intact")

    # --- features: 50% none / 10% complete / 40% partial@0.25 --------------
    if ARM_HAS_FEATURES:
        _fa = [i for i in have if d["train"][i]["ctx_features"] != T.ABSENT]
        fhave = _fa
        fm = {"none": 0, "complete": 0, "partial": 0}
        lies = scope_ok = 0
        for e in range(EPOCHS):
            ds.epoch_value = type("E", (), {"value": e})()
            for i in fhave:
                q = ds[i]["question"]
                shown = _section(q, None, scope=True)
                full = d["train"][i]["ctx_features"]
                mode = ("none" if shown == T.ABSENT else
                        "complete" if shown == full else "partial")
                fm[mode] += 1
                want = {"none": T.SCOPE_NONE, "complete": _scope_present,
                        "partial": T.SCOPE_SUBSET}[mode]
                if f"Residue features, {want}:" in q:
                    scope_ok += 1
                else:
                    lies += 1
                # the at-least-one guarantee
                if mode == "partial":
                    n_full = len([l for l in full.splitlines() if l.startswith("- ")])
                    n_now = len([l for l in shown.splitlines() if l.startswith("- ")])
                    if n_now >= n_full:
                        lies += 1
        ftot = sum(fm.values())
        check(0.42 <= fm["none"] / ftot <= 0.58,
              f"features: none {fm['none']/ftot:.2f} (target 0.50)")
        check(0.05 <= fm["complete"] / ftot <= 0.16,
              f"features: complete {fm['complete']/ftot:.2f} (target 0.10)")
        check(0.32 <= fm["partial"] / ftot <= 0.48,
              f"features: partial {fm['partial']/ftot:.2f} (target 0.40)")
        check(lies == 0,
              f"the scope label never lies, and 'subset' always removed >=1 "
              f"({scope_ok}/{ftot} labelled correctly)")
    else:
        check(all(r["ctx_features_scope"] == T.SCOPE_NONE for r in d["train"]),
              "nofeatures arm: every row is labelled 'none supplied'")

    # --- go_pred is never touched, in either arm ---------------------------
    ds.epoch_value = type("E", (), {"value": 0})()
    check(all(_section(ds[i]["question"], "Initial GO term speculations:")
              == d["train"][i]["ctx_speculations"] for i in range(min(60, len(ds)))),
          "go_pred is supplied in full on every access -- no gogpt knob is set")

    class _Epoch:
        value = 0

    ds.epoch_value = _Epoch()
    e0 = [ds[i]["question"] for i in range(min(40, len(ds)))]
    ds.epoch_value.value = 1
    e1 = [ds[i]["question"] for i in range(min(40, len(ds)))]
    ds.epoch_value.value = 0
    check(e0 != e1, "the mask changes between epochs")
    check(e0 == [ds[i]["question"] for i in range(min(40, len(ds)))],
          "the mask is reproducible for a given (epoch, index), so DDP ranks agree")

    # The one silent no-op this design could produce.
    try:
        CD.ContextDropoutDataset(d["train"], cfg=cfg, base_seed=23)
        check(False, "dropout on + no template -> should raise")
    except RuntimeError as exc:
        check("silently do nothing" in str(exc),
              "dropout on + no template -> raises instead of no-opping")

    print(f"\n{sum(_results)}/{len(_results)} passed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
