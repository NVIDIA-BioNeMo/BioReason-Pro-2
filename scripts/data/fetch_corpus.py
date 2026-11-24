#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Write the corpus the benchmark and training read, from `wanglab/bioreason-pro-v2`.

    python scripts/data/fetch_corpus.py                  # the benchmark: data/v2/proteins/test-*.parquet
    python scripts/data/fetch_corpus.py --split train    # the training proteins and their labels

The dataset keeps GO-GPT's predictions in separate `*_preds` configs. This joins them back on
`protein_id`, so every shard carries `go_{mf,bp,cc}_gogpt_ep{25,50,75}`, marks the `ground` split's
rows `is_grounding` (their trace is a grounding paragraph alone), renders `features` into the
`features_formatted` block the judges read, and writes the shards where every default looks
(`bioreason2.paths.CORPUS`):

- test: config `proteins_extra`, split `test`, the benchmark (6,112 proteins).
- train: config `proteins` split `train` and config `proteins_extra` splits `train` and `ground`, less the
  1,321 proteins of data/splits/train_excluded.txt (272,688 proteins). The `reasoning` and `final_answer`
  columns are empty, as the traces are not released: this gives SFT's proteins and labels, not its targets.

scripts/data/prepare_corpus.py and scripts/data/build_question.py then run on the result unchanged.
"""
import argparse
import glob
import os
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from bioreason2 import paths  # noqa: E402
from bioreason2.dataset.protein_fields import render_features  # noqa: E402

#: (config, split) of the dataset that make up each split of the corpus.
SOURCES = {
    "test": [("proteins_extra", "test")],
    "train": [("proteins", "train"), ("proteins_extra", "train"), ("proteins_extra", "ground")],
}
EXPECTED_ROWS = {"test": 6_112, "train": 272_688}
EXCLUDED = os.path.join(REPO, "data", "splits", "train_excluded.txt")
PREDS_SUFFIX = "_preds"
PRED_PREFIXES = ("go_mf_gogpt_ep", "go_bp_gogpt_ep", "go_cc_gogpt_ep")
GROUNDING_SPLIT = "ground"
#: What render_features returns for nothing to render; the corpus stored an empty block instead.
NO_FEATURES = "None annotated"
SHARD = "{split}-{index:05d}-of-{total:05d}.parquet"


def feature_block(features: list) -> str:
    """The residue-feature block of one protein, as the corpus's `features_formatted` holds it.

    Args:
        features: The protein's `features` records.

    Returns:
        The rendered block, or "" when there is nothing to render.
    """
    text = render_features(features) if features else ""
    return "" if text == NO_FEATURES else text


def source_shards(root: str, config: str, split: str) -> list:
    """The downloaded parquet shards of one config and split.

    Args:
        root: The local snapshot of the dataset.
        config: Config name.
        split: Split name.

    Returns:
        Sorted shard paths.

    Raises:
        SystemExit: if the snapshot holds none.
    """
    shards = sorted(glob.glob(os.path.join(root, config, f"{split}-*.parquet")))
    if not shards:
        raise SystemExit(f"no {config}/{split}-*.parquet in the {paths.HF_RAW_DATASET} snapshot")
    return shards


def with_predictions(table: pa.Table, preds: pa.Table, where: str) -> pa.Table:
    """A protein table with GO-GPT's id lists joined on `protein_id`, row order kept.

    Args:
        table: Rows of a protein config.
        preds: Rows of the matching `*_preds` config.
        where: Label for errors.

    Returns:
        The table with every `go_*_gogpt_ep*` column appended.

    Raises:
        SystemExit: if the join is not one prediction row per protein.
    """
    ids = preds.column("protein_id")
    if len(pc.unique(ids)) != len(ids):
        raise SystemExit(f"{where}: repeated protein_id in the predictions")
    index = pc.index_in(table.column("protein_id"), value_set=ids)
    if index.null_count:
        raise SystemExit(f"{where}: {index.null_count} proteins without predictions")
    rows = preds.take(index)
    for name in preds.column_names:
        if name.startswith(PRED_PREFIXES):
            if name in table.column_names:
                table = table.drop([name])
            table = table.append_column(name, rows.column(name))
    return table


def fetch(split: str, out_dir: str) -> int:
    """Download one split's shards, join their predictions and write them.

    Args:
        split: `test` or `train`.
        out_dir: Corpus directory; shards go to `<out_dir>/proteins`.

    Returns:
        Rows written.

    Raises:
        SystemExit: on a row count other than the expected one, or a repeated protein.
    """
    patterns = [f"{config}{suffix}/{name}-*.parquet"
                for config, name in SOURCES[split] for suffix in ("", PREDS_SUFFIX)]
    root = snapshot_download(paths.HF_RAW_DATASET, repo_type="dataset", allow_patterns=patterns)

    excluded = set()
    if split == "train":
        with open(EXCLUDED) as handle:
            excluded = {line.strip() for line in handle if line.strip()}

    tables = []
    for config, name in SOURCES[split]:
        preds = pa.concat_tables([pq.read_table(p) for p in source_shards(root, config + PREDS_SUFFIX, name)])
        for path in source_shards(root, config, name):
            table = with_predictions(pq.read_table(path), preds, f"{config}/{name}")
            # prepare_corpus.py takes the split from the file name; the dataset's own column calls the
            # grounding rows `ground`, which build_question.py would keep as a split of its own.
            if "split" in table.column_names:
                table = table.drop(["split"])
            if "is_grounding" not in table.column_names:
                table = table.append_column("is_grounding", pa.array([name == GROUNDING_SPLIT] * table.num_rows))
            if "features_formatted" not in table.column_names:
                blocks = [feature_block(f) for f in table.column("features").to_pylist()]
                table = table.append_column("features_formatted", pa.array(blocks, pa.string()))
            if excluded:
                table = table.filter(pc.invert(pc.is_in(table.column("protein_id"), pa.array(sorted(excluded)))))
            tables.append(table)

    ids = pa.chunked_array([t.column("protein_id") for t in tables])
    rows = len(ids)
    if len(pc.unique(ids)) != rows:
        raise SystemExit(f"{split}: a protein_id appears more than once")
    if rows != EXPECTED_ROWS[split]:
        raise SystemExit(f"{split}: {rows:,} proteins, expected {EXPECTED_ROWS[split]:,}")

    shard_dir = os.path.join(out_dir, "proteins")
    os.makedirs(shard_dir, exist_ok=True)
    # load_dataset and build_question glob every shard of a split, so an older build's would be read too.
    for stale in glob.glob(os.path.join(shard_dir, f"{split}-*.parquet")):
        os.remove(stale)
    for index, table in enumerate(tables):
        path = os.path.join(shard_dir, SHARD.format(split=split, index=index, total=len(tables)))
        pq.write_table(table, path)
        print(f"  {os.path.basename(path)}: {table.num_rows:,} rows")
    return rows


def main() -> int:
    """CLI.

    Returns:
        Process exit code.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=sorted(SOURCES), default="test",
                    help="test: the benchmark. train: SFT's proteins and labels (the traces are not released)")
    ap.add_argument("--out", default=paths.DATA_DIR, help="corpus directory (default: %(default)s)")
    args = ap.parse_args()

    rows = fetch(args.split, args.out)
    print(f"wrote {rows:,} {args.split} proteins to {os.path.join(args.out, 'proteins')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
