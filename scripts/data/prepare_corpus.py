#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Materialise the corpus for build_question.py: one GO-GPT epoch as `go_pred`, plus `split`.

    python scripts/data/prepare_corpus.py --src data/v2 --out data/v2-ep75

The corpus (as scripts/data/fetch_corpus.py writes it) carries GO-GPT's predictions for three
epochs as id lists (`go_{mf,bp,cc}_gogpt_ep{25,50,75}`) and has no `split` column: the split is the
file name.
build_question.py reads one rendered `go_pred` block and a `split` column, so this writes both.

The block is rendered from the id lists with GO-GPT's own renderer (`gogpt.utils.go_ontology`),
which is what `gogpt_api.py` uses on new proteins and what the training tables were built with.
The corpus's `go_pred_ep*` strings are not used: they spell the three root terms with underscores
(`molecular_function`), and every trained prompt spells them with spaces.
"""
import argparse
import glob
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "gogpt", "src"))

from gogpt.utils.go_ontology import format_predictions, load_go_names  # noqa: E402

GO_OBO = os.path.join(REPO, "bioreason2", "dataset", "go-basic.obo")
#: The GO-GPT epoch the released models trained and were scored with.
DEFAULT_EPOCH = 75
ASPECTS = ("MF", "BP", "CC")


def render_go_pred(table: pa.Table, epoch: int, go_names: dict) -> list:
    """The prompt's GO-GPT block for every row of a shard.

    Args:
        table: one corpus shard.
        epoch: GO-GPT epoch whose predictions to render.
        go_names: GO id -> name, from the repo's ontology.

    Returns:
        One rendered block per row.

    Raises:
        KeyError: when the shard has no predictions for that epoch.
    """
    columns = {a: table.column(f"go_{a.lower()}_gogpt_ep{epoch}").to_pylist() for a in ASPECTS}
    return [format_predictions({a: columns[a][i] or [] for a in ASPECTS}, go_names)
            for i in range(table.num_rows)]


def prepare_shard(path: str, out_dir: str, epoch: int, go_names: dict) -> int:
    """Write one shard with `go_pred` and `split` added.

    Args:
        path: source parquet, named `<split>-NNNNN-of-NNNNN.parquet`.
        out_dir: destination directory.
        epoch: GO-GPT epoch to render.
        go_names: GO id -> name.

    Returns:
        Rows written.
    """
    table = pq.read_table(path)
    if "go_pred" in table.column_names:
        table = table.drop(["go_pred"])
    table = table.append_column("go_pred", pa.array(render_go_pred(table, epoch, go_names)))
    if "split" not in table.column_names:
        split = os.path.basename(path).split("-")[0]
        table = table.append_column("split", pa.array([split] * table.num_rows))
    pq.write_table(table, os.path.join(out_dir, os.path.basename(path)))
    return table.num_rows


def main() -> int:
    """CLI.

    Returns:
        Process exit code.

    Raises:
        SystemExit: when the source holds no corpus shards.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(REPO, "data", "v2"),
                    help="the corpus as scripts/data/fetch_corpus.py writes it (holds proteins/*.parquet)")
    ap.add_argument("--out", default=os.path.join(REPO, "data", "v2-ep%d" % DEFAULT_EPOCH))
    ap.add_argument("--epoch", type=int, default=DEFAULT_EPOCH, choices=(25, 50, 75))
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.src, "proteins", "*.parquet")))
    if not shards:
        raise SystemExit(f"no corpus shards under {args.src}/proteins; run scripts/data/fetch_corpus.py "
                         "(README, Evaluate)")
    out_dir = os.path.join(args.out, "proteins")
    os.makedirs(out_dir, exist_ok=True)

    go_names = load_go_names(GO_OBO)
    for path in shards:
        print(f"{os.path.basename(path)}: {prepare_shard(path, out_dir, args.epoch, go_names):,} rows")
    print(f"wrote {out_dir} (go_pred = GO-GPT ep{args.epoch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
