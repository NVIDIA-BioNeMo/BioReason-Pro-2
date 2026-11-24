# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Mix two renders of the same RL rows so a share of prompts carries the residue-feature block.

SFT saw only the no-features prompt (`None annotated`), while the eval runs both contexts; the
released RL model trained with the block in half its prompts. Both inputs are `add_judge_fields`
outputs over the same proteins, one rendered from the no-features table and one from the features
table. A protein takes its with-features row when its hash falls under `--share`, so the whole
GRPO group sees one prompt and one epoch sees each protein once. With-features rows get their own
`data_source`, which SkyRL's evaluation reports separately, so validation shows both contexts.

`--keep_without` lists proteins whose with-features prompt is over the trainer's
`max_prompt_length`: SkyRL drops such rows silently, so they keep the no-features prompt instead.

    python scripts/rl/mix_features.py --without <parquet> --with <parquet> \\
        --share 0.5 --keep_without <ids> --output <parquet>
"""

import argparse
import hashlib
import os
import re
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

# Fixed: it decides which proteins carried the block in the released run.
SALT = "016-grpo-018-features"
DATA_SOURCE_WITH = "bioreason-pro-v2-feat"
# The no-features section as `scripts/data/question_template.py` renders it; the features table renders
# the same when a protein has no scan-derived features (1.8% of rows).
NO_FEATURES = "Residue features, none supplied:\nNone annotated"
# The feature section runs from its header to the next section of the template.
FEATURE_SECTION = re.compile(r"Residue features[^\n]*:\n.*?\n\n(?=Initial GO term speculations:)", re.S)
# The only fields the two renders may differ in.
PROMPT_KEYS = ("prompt", "data_source")
PROMPT_INFO_KEYS = ("question",)


def bucket(protein_id: str) -> float:
    """A stable uniform draw in [0, 1) for a protein.

    Args:
        protein_id: The id.

    Returns:
        The draw.
    """
    return int(hashlib.sha1(f"{SALT}:{protein_id}".encode()).hexdigest()[:12], 16) / 16 ** 12


def by_protein(path: str) -> dict[str, dict]:
    """Rows of a parquet keyed by protein id.

    Args:
        path: The parquet.

    Returns:
        protein id -> row.

    Raises:
        SystemExit: A protein appears twice.
    """
    rows = pq.read_table(path).to_pylist()
    out = {r["extra_info"]["protein_id"]: r for r in rows}
    if len(out) != len(rows):
        raise SystemExit(f"{path}: {len(rows) - len(out)} duplicate proteins")
    return out


def without_features(text: str) -> str:
    """A prompt text with its one feature section cut out.

    Args:
        text: A user turn or question.

    Returns:
        The text without the section.

    Raises:
        SystemExit: The text has no feature section, or more than one.
    """
    stripped, n = FEATURE_SECTION.subn("", text)
    if n != 1:
        raise SystemExit(f"expected one feature section, found {n}")
    return stripped


def check_pair(without: dict, with_: dict) -> None:
    """The two renders of one protein are the same template, differing only in the feature section.

    Args:
        without: The no-features row.
        with_: The features row.

    Raises:
        SystemExit: Any other field or any other byte of the prompt differs, or the no-features
            render carries a feature block.
    """
    pid = without["extra_info"]["protein_id"]
    for key in set(without) | set(with_):
        if key not in PROMPT_KEYS and key != "extra_info" and without.get(key) != with_.get(key):
            raise SystemExit(f"{pid}: field {key} differs between the renders")
    for key in set(without["extra_info"]) | set(with_["extra_info"]):
        if key not in PROMPT_INFO_KEYS and without["extra_info"].get(key) != with_["extra_info"].get(key):
            raise SystemExit(f"{pid}: extra_info.{key} differs between the renders")
    if NO_FEATURES not in without["extra_info"]["question"]:
        raise SystemExit(f"{pid}: the no-features render has a feature block")
    if [m["role"] for m in without["prompt"]] != [m["role"] for m in with_["prompt"]]:
        raise SystemExit(f"{pid}: the renders have different turns")
    for a, b in zip(without["prompt"], with_["prompt"]):
        if a["content"] != b["content"] and without_features(a["content"]) != without_features(b["content"]):
            raise SystemExit(f"{pid}: the {a['role']} turns differ outside the feature section")


def main() -> None:
    """Write the mixed parquet and print the share that carries the block.

    Raises:
        SystemExit: The inputs cover different proteins, or a pair fails `check_pair`.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--without", required=True, help="add_judge_fields output from the no-features table")
    parser.add_argument("--with", dest="with_", required=True, help="the same from the features table")
    parser.add_argument("--share", type=float, required=True, help="share of proteins given the features render")
    parser.add_argument("--keep_without", default="", help="file of protein ids that keep the no-features prompt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise SystemExit(f"{args.output} exists; refusing to overwrite")

    without, with_ = by_protein(args.without), by_protein(args.with_)
    if set(without) != set(with_):
        raise SystemExit(f"the renders cover different proteins: {len(set(without) ^ set(with_))} differ")
    keep_without = set()
    if args.keep_without:
        with open(args.keep_without) as handle:
            keep_without = {line.strip() for line in handle if line.strip()}

    rows, kinds = [], Counter()
    for pid, row in without.items():
        check_pair(row, with_[pid])
        if bucket(pid) >= args.share:
            kinds["none"] += 1
        elif pid in keep_without:
            kinds["kept_without"] += 1
        else:
            row = {**with_[pid], "data_source": DATA_SOURCE_WITH}
            kinds["features"] += 1
            kinds["annotated"] += NO_FEATURES not in row["extra_info"]["question"]
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), args.output)
    print(f"{args.output}: {len(rows)} rows; features render {kinds['features']} "
          f"({kinds['features'] / len(rows):.3f}), of which annotated {kinds['annotated']}; "
          f"drawn but over the length cap, kept without {kinds['kept_without']}; none {kinds['none']}")


if __name__ == "__main__":
    main()
