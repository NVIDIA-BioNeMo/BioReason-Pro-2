#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Render the student question for every corpus row and write the prompt table SFT and eval read.

    python scripts/data/build_question.py --no-features --out data/sft/v2-nofeatures   # without InterPro
    python scripts/data/build_question.py --out data/sft/v2-features                   # with the feature block
    python scripts/data/build_question.py --check                                      # render all, write nothing
    python scripts/data/build_question.py --print P50945                               # one question

The reasoning traces were written by a teacher shown the protein's annotation AND its GO
labels (`synth/`). This builds the other half: the question the trained model is asked and
must answer from the chain alone.

    student context  =  teacher context  -  the label

The prompt itself is `question_template.py`, which holds the prompt and nothing else so it
can be read as a prompt. This file decides which column fills which placeholder, how the
validation split is drawn, and the size guard.

The question is rendered ONCE into a single `question` column, and `load.py` uses it
verbatim as the one user turn. Training and evaluation then read the same bytes and cannot
diverge, whatever any flag downstream says.

ONE TURN, NO BRANCHES

There is no system prompt -- the teacher had none -- so the student matches it.
`eval.py:_build_prompt_string` iterates whatever roles the conversation carries, so a lone
user turn needs no special handling.

And there are no branches: every section always renders. Branching templates are how one
field takes an unrelated one with it, so:

  * the partner section always renders. Absence is information and it is common: 22.3% of
    train rows have no partners;
  * all three GO aspects are always asked for. Naming only the aspects a row has ground
    truth for leaks the shape of the label;
  * there is no "summarize" instruction: the teacher's answer format opens on its own
    `- Summary:` bullet, so the shape is taught by the targets.

TWO ARMS, ONE FLAG

`--no-features` renders the residue-feature section as `None annotated` rather than
removing it, so the two arms differ in what the model is told and not in the shape of what
it reads:

    features      Residue features, every annotated feature:\\n- 1-100 (TOPO_DOM): ...
    no features   Residue features, none supplied:\\nNone annotated

There is ONE string for "nothing here", and that is deliberate. A field can be empty three
ways -- curation found nothing (1.8% of rows for features, 22.3% for partners), the arm
withheld it, or `p_drop_ppi` dropped it this epoch -- and the model should treat all three
identically, because at inference a protein with no partners and a protein whose partners
were dropped are the same situation. `context_dropout.ABSENT` is the same string, so all
three paths agree.

Only what InterProScan produces from a bare sequence reaches the feature block
(`_scan_only`): curated UniProt features need a curator and PDB features a solved
structure, and a novel protein has neither. Neither arm carries `interpro_formatted`: an
entry's span is a derived envelope, `min(start)..max(end)` over its member matches, and in
14.77% of entries it covers residues no signature matched. Under the features arm, domains
reach the model on the match that carries them; under the no-features arm they must come
out of the ESM3 embedding.

PARTNERS ARE DROPPED 80% OF THE TIME, AT TRAIN TIME ONLY

`--p_drop_ppi 0.8` in training, applied by `bioreason2/dataset/cafa5/context_dropout.py`
per access and reseeded per epoch, so a protein is seen with partners in some epochs and
without in others -- that is what teaches invariance, where baking the drop into this file
would merely partition the corpus into two fixed halves. Validation and evaluation run at
full context.

WHAT IS DELIBERATELY ABSENT

The label, which is larger than `go_*_leaf`. `subcellular_location` IS the
cellular-component answer written in prose, and `protein_names` states the molecular
function in the field a reader reaches first ("Steroid 5-alpha-reductase 1").
`assert_no_label_column_is_read` tests for all of them positively rather than trusting the
template to keep omitting them.
"""
import argparse
import collections
import glob
import os
import random
import re
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import question_template as T    # noqa: E402  the prompt
from bioreason2 import paths as _paths  # noqa: E402
from bioreason2.dataset import protein_fields as bp  # noqa: E402  the shared feature renderer

#: The corpus as `scripts/data/prepare_corpus.py` writes it. `bioreason2/paths.py` is the
#: single source of truth for every location; set BRP_DATA_DIR to move it, or
#: BRP_V2_ROOT to point this one script somewhere else.
DEFAULT_V2_ROOT = os.environ.get("BRP_V2_ROOT", _paths.PROMPT_CORPUS)

#: `eval.py` runs at `--max_model_len 16384 --max_new_tokens 8000`, and
#: `text_token_budget()` computes the tokenizer cap as
#: `max_model_len - 200 go-graph tokens - max_length_protein - 2` = 14,182.
#: That formula does NOT reserve room for the generation, and vLLM's window covers
#: prompt AND completion, so the honest ceiling is 8,000 lower:
#:
#:     16384 - 8000 - 200 - 2000 - 2  =  6182
#:
#: The feature block is trimmed to fit it; nothing else can reach it.
PROMPT_TOKEN_BUDGET = 16384 - 8000 - 200 - 2000 - 2

#: Characters per token, calibrated against real token counts. The feature block is
#: denser than prose because it is mostly digits, capitals and punctuation.
PROSE_CHARS_PER_TOKEN = 3.50
BLOCK_CHARS_PER_TOKEN = 2.60

#: Columns the question is built from. Every one is present at inference.
CONTEXT_COLUMNS = ("protein_id", "organism", "ppi_formatted", "go_pred", "features")

#: Columns that ARE the answer, in whole or in part, and may never reach the
#: question. Checked positively by `assert_no_label_column_is_read` -- omission is
#: what a refactor loses first.
LABEL_COLUMNS = (
    "go_ids", "go_mf", "go_bp", "go_cc",
    "go_mf_leaf", "go_bp_leaf", "go_cc_leaf",
    "go_mf_tier2", "go_bp_tier2", "go_cc_tier2",
    "go_mf_tier2_leaf", "go_bp_tier2_leaf", "go_cc_tier2_leaf",
    "go_mf_tier3", "go_bp_tier3", "go_cc_tier3",
    "go_mf_tier3_leaf", "go_bp_tier3_leaf", "go_cc_tier3_leaf",
    "go_mf_tier4", "go_bp_tier4", "go_cc_tier4",
    "go_mf_tier4_leaf", "go_bp_tier4_leaf", "go_cc_tier4_leaf",
    "protein_names", "protein_function", "subcellular_location",
    #: The teacher's `Additional context` block: eleven UniProt CC topics plus
    #: `subcellular_location`, and three of them ARE the answer: `CATALYTIC
    #: ACTIVITY` gives the reaction and EC number, which is the molecular function;
    #: `PATHWAY` is the biological process; `SUBCELLULAR LOCATION` is the cellular
    #: component. Listed so the sentinel probe covers it -- a column absent from
    #: this tuple is not tested.
    "comments",
    "reasoning", "final_answer",
)

#: Carried onto the SFT table unchanged, so it is self-contained and neither the
#: training nor the eval path needs the v2 parquet beside it. `sequence` and
#: `structure_path` are what ESM3 consumes; `go_*` and `scored_aspects` are what
#: `cafa_evals` scores against, and `scored_aspects` is the ONLY correct gate --
#: a row can carry a non-empty `go_mf` and still not be scorable in MF
#: (`immature_annotation`, `tier1_retracted_since_gaf226`, `train_subsequence`).
PASSTHROUGH_COLUMNS = ("sequence", "structure_path", "organism", "length",
                       "go_mf", "go_bp", "go_cc",
                       "scored_aspects", "excluded_reason", "uniprot_status")

#: Validation share of train, held out by protein identity. Composition saturates
#: around 2,000 rows -- aspect presence, no-PPI share and
#: median length all sit within a point of full train and do not move above it --
#: so a larger split buys nothing measurable and costs training data. Small enough
#: to validate on IN FULL every epoch, which is why `--limit_val_batches` is 1.0.
DEFAULT_VAL_FRAC = 0.02

#: What InterProScan emits from a bare sequence, which is all a novel protein has:
#: `uniprot` is curated and `pdb` needs a solved structure.
SCAN_SRC = ("interpro_match", "interpro_site")

#: Improbable in any protein annotation, and long enough not to collide.
_SENTINEL = "ZZQXLEAKPROBEXQZZ"

#: One shard per this many rows. 30k keeps each file near the ~100 MB the Hub
#: prefers and matches what v2 itself ships.
ROWS_PER_SHARD = 30_000

CARD = """---
license: cc-by-4.0
pretty_name: BioReason-Pro 2 SFT ({arm})
tags:
  - protein-function-prediction
  - gene-ontology
  - cafa
configs:
  - config_name: default
    data_files:
{data_files}---

# BioReason-Pro 2 SFT — {arm} arm

Built by `scripts/data/build_question.py` from `wanglab/bioreason-pro-v2`. A local dataset
directory that `load_dataset("<path>")` reads.

| split | rows |
|---|---:|
{counts}

## Columns

| column | is |
|---|---|
| `protein_id` | UniProt accession, the join key |
| `split` | `train` / `validation` / `test`; the directory splits on it too |
| `question` | the whole student prompt, one user turn, rendered once so training and evaluation cannot diverge |
| `assistant_reasoning` | the teacher's trace |
| `assistant_answer` | the teacher's Final Answer, verbatim |
| `ctx_organism`, `ctx_features`, `ctx_speculations`, `ctx_partners` | the four rendered values `question` was built from, so the loader can rebuild it with a section dropped (`p_drop_ppi`) and verify a no-op rebuild reproduces `question` exactly |
| `sequence`, `structure_path`, `length` | what ESM3 consumes |
| `go_mf`, `go_bp`, `go_cc` | propagated tier-1 ground truth |
| `scored_aspects` | **the only correct Fmax gate.** A row can carry a non-empty `go_mf` and still not be scorable in MF |
| `excluded_reason`, `uniprot_status` | why a test row is unscorable; whether it is 2026_02 or v1 vintage |

`validation` is carved out of v2's `train` by protein identity at seed 23, because
v2 ships no validation split.

## Provenance

```
{provenance}
```
"""


def _dedupe_lines(block):
    """Collapse repeated lines, keeping the first occurrence and the order.

    `ppi_formatted` is not deduplicated: the same partner DESCRIPTION arrives under
    two different interactor ids and both are rendered, on 174 rows (0.12%). A0A060YAF5
    is the clearest -- 10 partner lines, 7 distinct, three of them doubled. Shown to
    the model, that reads as more independent evidence than it is.

    ORDER-PRESERVING, and not a `set`. A set would reorder the block, and order
    carries meaning on the feature side (N to C) -- so the same helper stays safe
    if a later build applies it there.

    KNOWN ASYMMETRY: the TEACHER saw the duplicated list, because its prompt was
    built from the same column. So on those 174 proteins the student sees a slightly
    cleaner block than the trace was written against. No information is added or
    removed -- a repeated line carries none.

    Args:
        block: newline-separated lines.

    Returns:
        The block without repeated lines.
    """
    if not block:
        return block
    seen, out = set(), []
    for line in block.splitlines():
        if line.strip() and line in seen:
            continue
        if line.strip():
            seen.add(line)
        out.append(line)
    return "\n".join(out)


def _strip_refs(text):
    """Drop citations and evidence codes, exactly as the teacher prompt does.

    `protein_fields.prepare_fields` runs `strip_refs_block` over `organism`,
    `ppi_formatted` and `features_formatted` AFTER rendering them. This file calls
    `bp.render_features` directly and reads `ppi_formatted` straight off the row, so
    it sits upstream of that pass and would otherwise keep apparatus the teacher
    did not see -- a divergence in the fields the two halves are supposed to share
    exactly. One row in ~17,600 carries a `(PubMed:...)` in a MUTAGEN description.

    `strip_refs_block` returns a line that carries no reference untouched, so
    this cannot damage the cap marker or a description that never had one.

    Args:
        text: a rendered field.

    Returns:
        The field without citations.
    """
    return bp.strip_refs_block(text) if text and text != T.ABSENT else text


def _feature_src(f):
    """A feature record's `src`, from a dict or an attribute."""
    return f.get("src") if isinstance(f, dict) else getattr(f, "src", None)


def _scan_only(feats):
    """Drop every feature a novel sequence could not supply. Unconditional, not a flag."""
    if not feats:
        return feats
    return [f for f in feats if _feature_src(f) in SCAN_SRC]


def render_features(feats, other_fields):
    """The residue-feature block, trimmed to whatever the rest of the prompt leaves.

    Rendered by `bioreason2/dataset/protein_fields.py` rather than reimplemented:
    the point of supplying features to both halves of the pipeline is that they are
    the same object, and two renderers would drift. The cap, the ordering and the
    dedupe are all its.

    Trimming is by TOKEN budget, not by bullet count, because the tail is extreme --
    at the teacher's cap of 200 the largest block is ~8,500 tokens and 40 rows
    overflow the window. It is the only section that gives way, and `render_features`
    already orders by evidence tier then position and states what a cap left out, so
    a trimmed block still reads N to C and still says it is short.

    Args:
        feats: the row's scan-derived feature records.
        other_fields: the template's other values, to size the budget.

    Returns:
        The block, or ABSENT when there is none or it cannot fit.
    """
    if not feats:
        return T.ABSENT
    scaffold = T.PROMPT.format(**{**other_fields, "features": ""})
    budget = max(0, PROMPT_TOKEN_BUDGET - _est_tokens(scaffold))

    cap = bp.FEATURE_CAP
    block = _strip_refs(bp.render_features(feats, cap=cap))
    while cap > 1 and _est_tokens(block, BLOCK_CHARS_PER_TOKEN) > budget:
        cap //= 2
        block = _strip_refs(bp.render_features(feats, cap=cap))
    return block if _est_tokens(block, BLOCK_CHARS_PER_TOKEN) <= budget else T.ABSENT


def _est_tokens(text, chars_per_token=PROSE_CHARS_PER_TOKEN):
    """Character-based estimate, deliberately, so this runs without a tokenizer.

    Calibrated against real token counts (median error 5.2%). It reports prompt
    sizes and sizes the feature budget; nothing depends on it being exact.
    """
    return len(text or "") / chars_per_token


def context_fields(row, features=True):
    """The values the template is filled from, each already rendered.

    These go onto the SFT table beside the question, so the prompt can be REBUILT
    per access without the v2 parquet or this file. That is what makes train-time
    context dropout possible: `p_drop_ppi` mutates `ctx_partners` and the row
    re-renders. Storing the rendered strings rather than the raw columns keeps the
    table small and keeps the rendering decision here rather than in the loader.

    Args:
        row: one corpus row.
        features: render the residue-feature block.

    Returns:
        Template field -> rendered text.
    """
    fields = {
        "organism": _strip_refs((row.get("organism") or "").strip()) or "Unknown",
        # `go_pred` is GO-GPT output -- identifiers and names, never prose -- so
        # it carries no apparatus and the teacher does not strip it either.
        "speculations": (row.get("go_pred") or "").strip() or T.ABSENT,
        "partners": _dedupe_lines(
            _strip_refs((row.get("ppi_formatted") or "").strip())) or T.ABSENT,
    }
    # `features_scope` must be filled before rendering, because the block is
    # trimmed against the scaffold the rest of the prompt occupies and the scope
    # string is part of it. It is overwritten immediately below with the real
    # value; the placeholder only has to be the LONGEST of the three so the
    # budget is computed against the worst case.
    fields["features_scope"] = max(T.SCOPE_ALL, T.SCOPE_SUBSET, T.SCOPE_NONE,
                                   key=len)
    fields["features"] = (render_features(_scan_only(row.get("features")), fields)
                          if features else T.ABSENT)
    # The table stores the UN-DROPPED render, so the scope it carries is "all"
    # whenever a block is present. The dropout wrapper recomputes it per access.
    fields["features_scope"] = (T.SCOPE_NONE if fields["features"] == T.ABSENT
                                else T.SCOPE_ALL)
    return fields


def build_question(row, features=True):
    """The whole prompt for one corpus row, as one string.

    Deterministic in the row -- the same row always renders the same bytes, which is
    what lets the dropout wrapper assert that an un-ablated re-render reproduces the
    stored question.

    Args:
        row: one corpus row.
        features: render the residue-feature block.

    Returns:
        The question.

    Raises:
        SystemExit: the template and this builder disagree on the fields.
    """
    fields = context_fields(row, features)
    if set(fields) != set(T.FIELDS):
        raise SystemExit(f"template fields {sorted(T.FIELDS)} but this builder "
                         f"supplies {sorted(fields)}")
    return T.PROMPT.format(**fields)


def assert_no_label_column_is_read(features=True):
    """Poison every label column and check none of it survives into the question.

    This is a property of the CODE, not of a row, so it runs once rather than per
    protein. A verbatim-substring scan over the rendered text cannot do this job:
    InterPro entry names are legitimately near-copies of protein names, and that is
    public InterPro data available at inference, not a leak. Poisoning tests the
    wiring instead: if an edit reads a label column into a placeholder, the sentinel
    comes out the other end and this fails loudly.

    Args:
        features: render the residue-feature block.

    Raises:
        SystemExit: a label column reaches the question.
    """
    row = {c: _SENTINEL for c in LABEL_COLUMNS}
    row.update(protein_id="P00000", organism="Escherichia coli",
               go_pred="Molecular Function (MF): GO:0003674 (molecular function)",
               ppi_formatted="- Some partner", features=[])
    leaked = [c for c in LABEL_COLUMNS
              if _SENTINEL in build_question({**row, **{c: _SENTINEL}}, features)]
    if _SENTINEL in build_question(row, features):
        raise SystemExit(
            "LABEL LEAK: a label column reaches the question. Poisoned columns "
            f"that survive rendering: {leaked or LABEL_COLUMNS}")


def check_no_leak(question, row):
    """The one leak that IS data-dependent: a GO identifier the row did not supply.

    The answer is a set of `GO:\\d{7}` strings, and the question may carry only
    those `go_pred` supplies. `go_pred` is a GO-GPT prediction made from the
    sequence, so an id in both is a prediction that happens to be right, not a leak.

    Args:
        question: the rendered question.
        row: the row it was rendered from.

    Returns:
        One message per GO id the question carries and go_pred does not.
    """
    pid = row.get("protein_id")
    supplied = set(re.findall(r"GO:\d{7}", row.get("go_pred") or ""))
    return [f"{pid}: GO id {gid} in the question but not in go_pred"
            for gid in sorted(set(re.findall(r"GO:\d{7}", question)) - supplied)]


def build_sample(row, features=True):
    """One SFT row: the question, the target, and the identity to join on.

    The target is the teacher's `final_answer` verbatim; its format already opens
    on the `- Summary:` bullet the teacher wrote, so nothing is spliced in.

    Args:
        row: one corpus row, carrying `reasoning` and `final_answer`.
        features: render the residue-feature block.

    Returns:
        The table row.
    """
    ctx = context_fields(row, features)
    out = {
        "protein_id": row.get("protein_id"),
        "split": row.get("split"),
        "question": T.PROMPT.format(**ctx),
        "assistant_reasoning": row.get("reasoning") or "",
        "assistant_answer": row.get("final_answer") or "",
        # A grounding row is an opening paragraph with no answer; the collator reads
        # this to end the supervised span before </think> and <|im_end|>.
        "is_grounding": bool(row.get("is_grounding")),
    }
    # The pieces the question was built from, so the training loader can rebuild
    # it with a section dropped and assert that rebuilding with NOTHING dropped
    # reproduces `question` byte for byte.
    out.update({f"ctx_{k}": v for k, v in ctx.items()})
    for col in PASSTHROUGH_COLUMNS:
        out[col] = row.get(col)
    return out


def _purge_hf_cache(out_dir):
    """Drop any `datasets` cache entry for this dataset directory.

    `load_dataset` caches the INFERRED FEATURES beside the arrow, keyed on the
    directory's name. Rebuilding in place with a column added therefore fails with
    a `CastError`, and `download_mode="force_redownload"` does not help, because
    the stale features come from the builder metadata rather than the download.
    Since the caller has just rewritten the data, any cache entry for it is stale.

    Args:
        out_dir: the dataset directory just written.
    """
    try:
        import datasets.config as dc
    except Exception:
        return
    root = str(dc.HF_DATASETS_CACHE)
    name = os.path.basename(os.path.normpath(out_dir))
    for path in glob.glob(os.path.join(root, name)) + \
            glob.glob(os.path.join(root, f"*_{name}_*.lock")):
        shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) \
            else os.remove(path)
        print(f"  purged stale dataset cache: {path}")


def write_hf_dataset(rows, out_dir, arm, provenance):
    """Write a HuggingFace-loadable directory: sharded parquet plus a card.

    One file per split rather than one parquet with a `split` column, because a
    consumer calling `load_dataset(path)` should get three splits without having to
    know to filter. The column is kept on the rows as well.

    Args:
        rows: every table row.
        out_dir: the destination directory.
        arm: `features` or `nofeatures`, for the card.
        provenance: text recorded in the card.

    Returns:
        Split -> row count.
    """
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"] or "train", []).append(r)

    # The schema is imposed ONCE over the whole frame and then written per split.
    # Pin the columns that are null on train and populated on test: Arrow would
    # otherwise type them `null` on a train-only build, and `load_dataset` infers
    # its schema from the first shard it reads and then dies casting the next.
    schema = pa.Table.from_pylist(rows).schema
    for name, typ in (("scored_aspects", pa.list_(pa.string())),
                      ("excluded_reason", pa.string()),
                      ("structure_path", pa.string())):
        i = schema.get_field_index(name)
        if i >= 0 and pa.types.is_null(schema.field(i).type):
            schema = schema.set(i, pa.field(name, typ))
    data = os.path.join(out_dir, "data")
    os.makedirs(data, exist_ok=True)

    # PURGE THE DATA DIRECTORY FIRST. Shard NAMES carry the shard COUNT
    # (`train-00000-of-00005`), so a rebuild that changes the count does not
    # overwrite the previous build -- it sits beside it, `load_dataset` globs both,
    # and the table silently gains the old rows as duplicates.
    stale = sorted(glob.glob(os.path.join(data, "*.parquet")))
    if stale:
        print(f"purging {len(stale)} existing parquet file(s) from {data}")
        for f in stale:
            os.remove(f)

    files = {}
    for split in ("train", "validation", "test"):
        part = by_split.get(split) or []
        if not part:
            continue
        n = max(1, -(-len(part) // ROWS_PER_SHARD))
        files[split] = []
        for i in range(n):
            name = f"{split}-{i:05d}-of-{n:05d}.parquet"
            chunk = part[i * ROWS_PER_SHARD:(i + 1) * ROWS_PER_SHARD]
            pq.write_table(pa.Table.from_pylist(chunk, schema=schema),
                           os.path.join(data, name))
            files[split].append(name)

    _purge_hf_cache(out_dir)

    yaml = "".join(f"      - split: {s}\n        path: data/{s}-*.parquet\n"
                   for s in files)
    counts = "\n".join(f"| {s} | {len(by_split[s]):,} |" for s in files)
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write(CARD.format(arm=arm, data_files=yaml, counts=counts,
                             provenance=provenance))
    return {s: len(by_split[s]) for s in files}


def assign_validation(protein_ids, frac=DEFAULT_VAL_FRAC, seed=23):
    """Carve a validation split out of train, by protein identity.

    v2 ships `train` and `test` and no `validation`; without this, checkpoints would
    be selected on the test set.

    Args:
        protein_ids: every train protein id.
        frac: the validation share.
        seed: the shuffle seed.

    Returns:
        The validation ids.
    """
    ids = sorted(protein_ids)
    random.Random(seed).shuffle(ids)
    return set(ids[:round(frac * len(ids))])


def _iter_v2(root, split, columns):
    """Rows of one split of the corpus, reading only `columns`.

    Args:
        root: the corpus directory, holding `proteins/<split>-*.parquet`.
        split: `train` or `test`.
        columns: the columns to read.

    Yields:
        One dict per row.
    """
    for path in sorted(glob.glob(os.path.join(root, "proteins",
                                              f"{split}-*.parquet"))):
        for row in pq.read_table(path, columns=list(columns)).to_pylist():
            yield row


def main():
    """Check, print or build the prompt table.

    Returns:
        The process exit code.

    Raises:
        SystemExit: a leak, a split violation or an empty target; nothing is written.
    """
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2-root", default=DEFAULT_V2_ROOT)
    ap.add_argument("--out", help="destination dataset DIRECTORY (HF layout)")
    ap.add_argument("--no-features", dest="features", action="store_false",
                    help="no residue-feature block (the published prompt)")
    ap.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--keep-untraced", action="store_true",
                    help="keep train and validation rows without a trace, with empty targets, as test "
                         "rows are kept: a table for RL, which reads the question and labels alone. "
                         "SFT cannot train on it")
    ap.add_argument("--check", action="store_true",
                    help="render every row, report leaks and sizes, write nothing")
    ap.add_argument("--print", dest="print_id",
                    help="render one protein's question and exit")
    args = ap.parse_args()

    if args.print_id:
        for split in ("train", "test"):
            for row in _iter_v2(args.v2_root, split, CONTEXT_COLUMNS):
                if row["protein_id"] == args.print_id:
                    print(build_question(row, args.features))
                    return 0
        print(f"no such protein: {args.print_id}", file=sys.stderr)
        return 1

    if args.check:
        assert_no_label_column_is_read(args.features)
        print("label columns reaching the question: none")
        problems, sizes = [], []
        for split in ("train", "test"):
            for row in _iter_v2(args.v2_root, split, CONTEXT_COLUMNS):
                q = build_question(row, args.features)
                problems += check_no_leak(q, row)
                sizes.append(_est_tokens(q))
        sizes.sort()
        print(f"rendered {len(sizes):,} questions, "
              f"features={'on' if args.features else 'off'}")
        print(f"  est tokens  p50 {sizes[len(sizes) // 2]:.0f}  "
              f"p90 {sizes[int(.9 * len(sizes))]:.0f}  max {sizes[-1]:.0f}")
        print(f"  over the {PROMPT_TOKEN_BUDGET:,}-token budget "
              f"{sum(1 for s in sizes if s > PROMPT_TOKEN_BUDGET):,}")
        print(f"  leaks {len(problems)}")
        for p in problems[:10]:
            print("   ", p)
        return 1 if problems else 0

    if not args.out:
        ap.error("--out is required unless --check or --print")

    assert_no_label_column_is_read(args.features)
    print(f"traces: inline, from {args.v2_root}")

    val_ids = assign_validation(
        [r["protein_id"] for r in _iter_v2(args.v2_root, "train",
                                           ("protein_id",))],
        args.val_frac, args.seed)
    print(f"validation carved from train: {len(val_ids):,}")

    columns = (CONTEXT_COLUMNS + ("split", "is_grounding") + PASSTHROUGH_COLUMNS
               + ("reasoning", "final_answer"))
    out, missing, untraced = [], 0, 0
    for split in ("train", "test"):
        for row in _iter_v2(args.v2_root, split, columns):
            _r, _a = row.get("reasoning"), row.get("final_answer")
            # A grounding row is an opening paragraph and `final_answer` is NULL
            # on all 38,044 by design, so requiring both would drop every one.
            trace = (_r, _a or "") if (_r and (_a or row.get("is_grounding"))) else None
            if not trace:
                # TEST rows are kept WITHOUT a trace, by design: the student is
                # evaluated by GENERATING on test, and a teacher trace there would
                # be a target nothing reads. Dropping them instead would leave
                # nothing to evaluate. Empty targets are safe because training
                # reads train+validation only. --keep-untraced keeps train rows the same way, for RL.
                if split == "test" or args.keep_untraced:
                    row["reasoning"], row["final_answer"] = "", ""
                    untraced += 1
                else:
                    missing += 1
                    continue
            else:
                row["reasoning"], row["final_answer"] = trace
            _src_split = split           # the FILE it came from, not the column
            if split == "train" and row["protein_id"] in val_ids:
                row["split"] = "validation"
            sample = build_sample(row, args.features)
            sample["_src_split"] = _src_split
            leaks = check_no_leak(sample["question"], row)
            if leaks:
                raise SystemExit(f"label leak, refusing to write: {leaks[0]}")
            out.append(sample)

    if not out:
        raise SystemExit("no rows -- every protein in the v2 split lacked a usable "
                         "trace. Check --v2-root.")

    # ---- EVERY CHECK RUNS BEFORE THE WRITE ------------------------------
    # The next consumer reads the DIRECTORY, not this exit code, so a table that
    # fails a check must never land on disk.

    # Split integrity, keyed on which parquet the row came from rather than on
    # whether it has a trace, which stopped being a proxy for "is a train row"
    # once test rows could carry traces.
    _leak = [r["protein_id"] for r in out
             if r["_src_split"] == "test" and r["split"] != "test"]
    if _leak:
        raise SystemExit(f"SPLIT LEAK, refusing to write: {len(_leak)} rows from "
                         f"the test parquet landed in "
                         f"{sorted({r['split'] for r in out if r['_src_split'] == 'test'})}, "
                         f"e.g. {_leak[:3]}")
    _by_id = collections.Counter(r["protein_id"] for r in out)
    _dupe = [pid for pid, n in _by_id.items() if n > 1]
    if _dupe:
        raise SystemExit(f"DUPLICATE protein_id across splits, refusing to write: "
                         f"{len(_dupe)}, e.g. {_dupe[:3]}")
    _train_ids = {r["protein_id"] for r in out if r["split"] in ("train", "validation")}
    _test_ids = {r["protein_id"] for r in out if r["split"] == "test"}
    if _train_ids & _test_ids:
        raise SystemExit(f"TRAIN/TEST OVERLAP, refusing to write: "
                         f"{len(_train_ids & _test_ids)} ids")

    # A train or validation row must never carry an empty target: the loss mask
    # scans for an assistant turn, finds none, emits all -100, and the run
    # contributes no gradient for that sample with nothing in the log to say so.
    # A grounding row's target is its reasoning alone, so only that is required.
    _bad = [r["protein_id"] for r in out
            if r["split"] != "test" and not (r["assistant_reasoning"]
                                             and (r["assistant_answer"] or r["is_grounding"]))]
    if _bad and not args.keep_untraced:
        raise SystemExit(f"refusing to write: {len(_bad)} train/validation rows "
                         f"have an empty target, e.g. {_bad[:3]}")

    _ground = sum(1 for r in out if r["is_grounding"])
    _ground_bad = [r["protein_id"] for r in out if r["is_grounding"] and r["assistant_answer"]]
    if _ground_bad:
        raise SystemExit(f"refusing to write: {len(_ground_bad)} grounding rows carry "
                         f"an answer, e.g. {_ground_bad[:3]}")

    print(f"split integrity: {len(_train_ids):,} train+val ids, "
          f"{len(_test_ids):,} test ids, 0 overlap, 0 duplicates")
    print(f"grounding rows: {_ground:,} (reasoning only, no answer, no terminator)")
    for r in out:
        del r["_src_split"]

    # ---- only now is anything written -----------------------------------
    provenance = "\n".join([
        "traces      inline (reasoning/final_answer on the v2 rows)",
        f"v2 root     {args.v2_root}",
        f"arm         {'features' if args.features else 'nofeatures'}",
        f"val         {args.val_frac:.3f} of train, by protein identity, seed {args.seed}",
        "untraced    " + ("kept with empty targets (--keep-untraced, not for SFT)" if args.keep_untraced
                          else "train rows dropped"),
        f"builder     {os.path.abspath(__file__)}",
    ])
    counts = write_hf_dataset(out, args.out,
                              "features" if args.features else "nofeatures",
                              provenance)
    print(f"rows {len(out):,}  {counts}  "
          f"(train rows with no usable trace: {missing:,}; "
          f"rows kept with empty targets: {untraced:,})")
    print(f"wrote {args.out}/  ->  load_dataset({args.out!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
