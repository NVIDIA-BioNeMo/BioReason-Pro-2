# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Add the LLM judge's inputs to an RL parquet, leaving the prompt and labels untouched.

The judge-reward env needs two things beyond the GO labels: the question the policy was served
and the protein's ground-truth fields. Each row's `extra_info` gains:

- `question`: the served user turn;
- `judge_truth`: `judge_reward.derive_truth` of the protein's corpus row: every curated field
  (UniProt function, location and comments, the full feature block, PDB ids, partners, the
  numbered sequence, the teacher's answer for the optional reference). Read from `--truth_corpus`
  when given (the corpus carries comments and the full feature block), else from the
  snapshot the row's labels came from (`label_source`).

    python -m bioreason2.rl.add_judge_fields --input rl_data/train_nofeatures.parquet \\
        --output rl_data/judge/train_nofeatures.parquet --truth_corpus <v2-data>/proteins
"""

import argparse
import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq

from bioreason2.rl.judge_reward import CORPUS_COLUMNS, TRUTH_KEYS, derive_truth


def load_truth(roots: set[str], pids: set[str]) -> dict[tuple[str, str], dict[str, str]]:
    """Judge text fields per (corpus root, protein).

    Args:
        roots: Corpus directories of parquet shards.
        pids: Proteins wanted.

    Returns:
        (root, protein_id) -> TRUTH_KEYS -> text ("" when the corpus has none).
    """
    out = {}
    for root in sorted(roots):
        for shard in sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)):
            names = set(pq.read_schema(shard).names)
            columns = ["protein_id", *(k for k in CORPUS_COLUMNS if k in names)]
            for row in pq.read_table(shard, columns=columns).to_pylist():
                if row["protein_id"] in pids:
                    out[(root, row["protein_id"])] = derive_truth(row)
    return out


def user_turn(prompt: list[dict]) -> str:
    """The served user message.

    Args:
        prompt: The row's chat messages.

    Returns:
        The user turn's content.

    Raises:
        ValueError: The prompt has no user turn.
    """
    for message in prompt:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    raise ValueError("prompt has no user turn")


def main() -> None:
    """Write the augmented parquet.

    Raises:
        FileExistsError: The output exists.
        SystemExit: Some rows have no corpus row to take the judge fields from.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--truth_corpus", default="")
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError(f"{args.output} exists; refusing to overwrite")

    rows = pq.read_table(args.input).to_pylist()
    pids = {r["extra_info"]["protein_id"] for r in rows}
    roots = {args.truth_corpus} if args.truth_corpus else {r["extra_info"]["label_source"] for r in rows}
    truth = load_truth(roots, pids)
    missing, empty = [], 0
    for row in rows:
        info = row["extra_info"]
        root = args.truth_corpus or info["label_source"]
        fields = truth.get((root, info["protein_id"]))
        if fields is None:
            missing.append(info["protein_id"])
            continue
        empty += not any(fields.values())
        info.update(question=user_turn(row["prompt"]), judge_truth=fields)
    if missing:
        raise SystemExit(f"{len(missing)} rows have no corpus row, e.g. {missing[:3]}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output)
    filled = {k: sum(bool(r["extra_info"]["judge_truth"][k]) for r in rows) for k in TRUTH_KEYS}
    print(f"{len(rows)} rows -> {args.output}; rows with no text truth at all: {empty}")
    print("rows with each field: " + ", ".join(f"{k} {v}" for k, v in filled.items()))


if __name__ == "__main__":
    main()
