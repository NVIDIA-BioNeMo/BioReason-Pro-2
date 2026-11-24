#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Does `wanglab/bioreason-pro-v2` still carry what the code reads from it?

    python tests/test_hub_dataset_contract.py [--structure-dir assets/structures]

A column rename on the Hub does not break anything loudly: `build_question.py` and the judge read
their fields with `.get(...)`, so a dropped column renders as an empty section and the only symptom
is a lower score. This checks the contract instead of trusting it, on the benchmark split that
scripts/data/fetch_corpus.py writes: config `proteins_extra`, split `test`, joined on `protein_id`
with config `proteins_extra_preds`.

Downloads the two test shards (about 11 MB), so it needs network access; plain `pytest` collects
nothing from it.
"""
import argparse
import os
import sys

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bioreason2 import paths as _paths  # noqa: E402

PROTEINS_FILE = "proteins_extra/test-00000-of-00001.parquet"
PREDS_FILE = "proteins_extra_preds/test-00000-of-00001.parquet"
BENCHMARK_ROWS = 6_112

#: Columns of the protein config the pipeline reads, and who reads them.
REQUIRED = {
    "protein_id": "every join, the eval resume filenames",
    "sequence": "ESM3, the numbered sequence the judge sees, grounding",
    "length": "build_question.py, carried onto the prompt table; grounding",
    "organism": "the prompt, the judge",
    "taxid": "grounding: the taxon's domain",
    "structure_path": "optional structure conditioning",
    "ppi_formatted": "the prompt's partner section, the judge",
    "features": "the prompt's InterPro block, the judge's feature block, grounding",
    "interpro_formatted": "the judge, grounding's InterPro entries",
    "interpro_ids": "grounding's InterPro entries",
    "go_mf": "ground truth, molecular function",
    "go_bp": "ground truth, biological process",
    "go_cc": "ground truth, cellular component",
    "go_mf_tier2": "tier-2 ground truth, unioned into the label set (cafa_evals.py --tier2_from)",
    "go_bp_tier2": "as above",
    "go_cc_tier2": "as above",
    "scored_aspects": "the scoring gate per (protein, aspect)",
    "excluded_reason": "why a row is unscorable",
    "uniprot_status": "carried onto the prompt table",
    "reviewed": "grounding",
    "protein_names": "the judge's evidence, grounding",
    "protein_function": "the judge's evidence",
    "subcellular_location": "the judge's evidence, grounding",
    "comments": "the judge's additional context, grounding",
    "reasoning": "build_question.py (empty here: the traces are not released)",
    "final_answer": "as above",
}

#: Columns of the predictions config: prepare_corpus.py renders the prompt's GO-GPT block from these.
REQUIRED_PREDS = ("protein_id", "go_mf_gogpt_ep75", "go_bp_gogpt_ep75", "go_cc_gogpt_ep75")


def main() -> int:
    """Check the benchmark shards' schema and join, and optionally structure coverage.

    Returns:
        0 when the schema satisfies every reader, 1 otherwise.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=_paths.HF_RAW_DATASET)
    ap.add_argument("--structure-dir", default=None,
                    help="flat directory of .pdb/.cif; checks structure_path coverage")
    args = ap.parse_args()

    proteins = pq.read_table(hf_hub_download(args.repo, PROTEINS_FILE, repo_type="dataset"))
    preds = pq.read_table(hf_hub_download(args.repo, PREDS_FILE, repo_type="dataset"))

    print("%s  %s + %s" % (args.repo, PROTEINS_FILE, PREDS_FILE))
    print("  rows %d / %d   columns %d / %d"
          % (proteins.num_rows, preds.num_rows, proteins.num_columns, preds.num_columns))
    missing = [c for c in REQUIRED if c not in proteins.column_names]
    missing += [c for c in REQUIRED_PREDS if c not in preds.column_names]
    for column, why in sorted(REQUIRED.items()):
        print("  %-24s %s" % (column, "ok" if column in proteins.column_names else "MISSING  <- " + why))
    for column in REQUIRED_PREDS[1:]:
        print("  %-24s %s" % (column, "ok" if column in preds.column_names else "MISSING  <- the GO-GPT block"))

    # fetch_corpus.py joins the two configs row for row; anything but a 1:1 join changes the benchmark.
    ids = proteins.column("protein_id").to_pylist()
    pred_ids = preds.column("protein_id").to_pylist()
    joined = len(ids) == len(set(ids)) == len(pred_ids) == len(set(pred_ids)) and set(ids) == set(pred_ids)
    print("  join on protein_id: %s, %d proteins (benchmark %d)"
          % ("1:1" if joined else "NOT 1:1", len(set(ids)), BENCHMARK_ROWS))

    scorable = sum(1 for r in proteins.column("scored_aspects").to_pylist() if r) \
        if "scored_aspects" in proteins.column_names else 0
    print("  scorable: %d of %d" % (scorable, proteins.num_rows))

    # A structure that does not resolve is not an error anywhere downstream: the
    # collator substitutes empty coordinates and the protein is encoded from
    # sequence alone. Partial coverage therefore looks exactly like a normal run.
    short = False
    if args.structure_dir:
        paths = [p for p in proteins.column("structure_path").to_pylist() if p]
        have = sum(1 for p in paths if os.path.exists(os.path.join(args.structure_dir, os.path.basename(p))))
        pct = 100.0 * have / len(paths) if paths else 0.0
        print("\n  structures  %d of %d resolve (%.1f%%) in %s"
              % (have, len(paths), pct, args.structure_dir))
        if pct < 99.0:
            print("  the rest will be encoded sequence-only, silently. Download every shard.")
            short = True

    if short:
        print("\nFAIL: incomplete structure coverage")
        return 1
    if missing:
        print("\nFAIL: %d required columns missing" % len(missing))
        return 1
    if not joined or len(ids) != BENCHMARK_ROWS:
        print("\nFAIL: the benchmark is not %d proteins joined 1:1 with their predictions" % BENCHMARK_ROWS)
        return 1
    print("\nPASS: the dataset satisfies every reader")
    return 0


if __name__ == "__main__":
    sys.exit(main())
