# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Build a SkyRL GRPO parquet whose prompts are token-identical to the published eval.

Rows come from `eval.py`'s own `load_dataset`, driven by the flags `scripts/eval/eval.sh` passes,
so the prompt messages are exactly the ones the SFT checkpoint was scored on. Each row is then
changed in two ways only, both forced by how SkyRL injects a modality:

* the user turn becomes a string, rendered with the chat template's own rule for each part
  (`Protein: <|protein_pad|>\\n\\n`, `GO graph: ...\\n\\n`, then the text);
* the GO run is the 200 distinct `<|go_graph_i|>` tokens checkpoint prep baked into
  `embed_tokens`, instead of one `<|go_graph_pad|>` the processor would expand.

The protein stays as ONE `<|protein_pad|>`: SkyRL counts placeholder strings and expands each to
the payload's `n_tokens`, so writing the run out here would expand it n-squared times.
`n_tokens` comes from the split's ESM3 cache and must equal what eval's processor expands to,
`min(len(sequence), 2000) + 2`; rows where they differ are dropped and counted.

Before writing, the first `--parity_rows` rows are tokenized both ways -- eval's (template with
list content, then the processor's pad expansion) and RL's (template with string content, the
protein expanded, the GO ids mapped back to `<|go_graph_pad|>`) -- and must be identical.

Run in the eval environment, from the repo root::

    python -m bioreason2.rl.build_dataset --table data/sft/v2-nofeatures \\
        --split val --cache_dir <ESM3 layer-37 cache> --ids <pool ids> \\
        --checkpoint <prepared RL checkpoint> --output rl_data/train_nofeatures.parquet
"""

import argparse
import ast
import glob
import importlib
import os
from typing import Any

import pyarrow.parquet as pq
from datasets import Dataset
from safetensors import safe_open
from transformers import AutoTokenizer

from bioreason2 import paths
from bioreason2.rl.constants import (
    FIRST_GO_ID,
    GO_GRAPH_PAD_ID,
    GO_GRAPH_PAD_TOKEN,
    GO_TOKENS,
    MAX_LENGTH_PROTEIN,
    NUM_GO_TOKENS,
    PROTEIN_PAD_TOKEN,
)
from bioreason2.rl.handlers import CACHE_KEY
from bioreason2.rl.reward import ground_truth

#: SkyRL modality id and gym environment the rows are routed to.
MODALITY_ID = "protein_mod"
ENV_CLASS = "go_function"
DATA_SOURCE = "bioreason-pro-v2"

#: How the chat template renders each non-text part of a user turn.
PROTEIN_PART = f"Protein: {PROTEIN_PAD_TOKEN}\n\n"
GO_PART_SFT = f"GO graph: {GO_GRAPH_PAD_TOKEN}\n\n"
GO_PART_RL = "GO graph: " + "".join(GO_TOKENS) + "\n\n"

#: `eval.py`'s `--enable_thinking` default, which `eval.sh` leaves alone.
ENABLE_THINKING = True

#: Sample fields copied to `extra_info` for the reward: what eval writes into its records.
LABEL_FIELDS = (
    "go_aspect", "go_bp", "go_mf", "go_cc", "go_bp_leaf", "go_mf_leaf", "go_cc_leaf",
    "scored_aspects", "excluded_reason",
)

#: Tier-2 labels, which neither eval nor the SFT tables carry; `cafa_evals.py --tier2_from`
#: joins them from the corpus by protein_id, and so does this. Tier 1 is read alongside to
#: check the snapshot against the table.
TIER1_FIELDS = ("go_mf", "go_bp", "go_cc")
TIER2_FIELDS = ("go_mf_tier2", "go_bp_tier2", "go_cc_tier2")

#: `extra_info` key recording which corpus snapshot a row's labels came from.
LABEL_SOURCE = "label_source"


def eval_args(table: str, split: str, dataset_cache_dir: str) -> argparse.Namespace:
    """`eval.py`'s arguments, with the dataset flags `scripts/eval/eval.sh` passes.

    Args:
        table: Prompt table directory.
        split: `train`, `val` or `test`, as `eval.load_dataset` names them.
        dataset_cache_dir: HF datasets cache.

    Returns:
        The parsed namespace.
    """
    unused = "unused-by-build_dataset"
    flags = [
        "--ckpt_dir", unused, "--evals_path", unused, "--precomputed_embeddings_path", unused,
        "--go_obo_path", paths.GO_OBO,
        "--cafa5_dataset", table, "--cafa5_dataset_split", split,
        "--cafa5_dataset_name", "default", "--dataset_cache_dir", dataset_cache_dir,
        "--max_length_protein", str(MAX_LENGTH_PROTEIN), "--max_samples", "0",
        "--num_chunks", "1", "--seed", "23", "--debug", "False",
    ]
    return importlib.import_module("eval").setup_argument_parser().parse_args(flags)


def prompt_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """The turns eval sends: system and user messages up to the first assistant turn.

    Args:
        sample: A row from `eval.load_dataset`.

    Returns:
        The prompt messages, content parts unchanged.
    """
    messages = []
    for message in sample["prompt"]:
        if message["role"] == "assistant":
            break
        messages.append(message)
    return messages


def render_content(content: Any, go_part: str) -> str:
    """Render a user turn's parts as the chat template does, with a chosen GO span.

    Args:
        content: A string, or a list of `protein` / `go_graph` / `text` parts.
        go_part: Text standing in for the GO graph part.

    Returns:
        The rendered content.

    Raises:
        ValueError: A part has an unknown type.
    """
    if isinstance(content, str):
        return content
    pieces = []
    for part in content:
        kind = part.get("type")
        if kind == "protein":
            pieces.append(PROTEIN_PART)
        elif kind == "go_graph":
            pieces.append(go_part)
        elif kind == "text":
            pieces.append(part.get("text") or "")
        else:
            raise ValueError(f"unknown content part {kind!r}")
    return "".join(pieces)


def rl_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """String-content messages with the baked GO tokens.

    Args:
        messages: Eval's prompt messages.

    Returns:
        `{"role", "content"}` dicts SkyRL can template.
    """
    return [{"role": m["role"], "content": render_content(m["content"], GO_PART_RL)} for m in messages]


def as_terms(value: Any) -> list[str]:
    """Normalise a GO-term column value (list, stringified list, or null) to a sorted list.

    Args:
        value: The stored value.

    Returns:
        The terms, sorted.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = ast.literal_eval(value) if value.strip() else []
    return sorted(value)


def load_labels(labels_from: list[str]) -> dict[str, dict[str, list[str]]]:
    """Tier-1 and tier-2 GO labels per protein, read from corpus snapshots.

    `cafa_evals.py --tier2_from` joins tier 2 by protein_id; tier 1 is read too, only to check
    that the snapshot agrees with the labels the prompt table carries. A protein held by more
    than one snapshot must carry identical labels in each, so the order of `labels_from` can
    never decide a label.

    Args:
        labels_from: Corpus directories of parquet shards with `go_*` and `go_*_tier2` columns.

    Returns:
        protein_id -> {column: terms, LABEL_SOURCE: snapshot directory}.

    Raises:
        FileNotFoundError: A directory holds no parquet shard.
        ValueError: Two snapshots disagree on a protein's labels.
    """
    labels = {}
    for root in labels_from:
        shards = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
        if not shards:
            raise FileNotFoundError(f"no parquet shards under {root}")
        for shard in shards:
            for row in pq.read_table(shard, columns=["protein_id", *TIER1_FIELDS, *TIER2_FIELDS]).to_pylist():
                terms = {k: as_terms(row[k]) for k in (*TIER1_FIELDS, *TIER2_FIELDS)}
                seen = labels.get(row["protein_id"])
                if seen is None:
                    labels[row["protein_id"]] = {**terms, LABEL_SOURCE: root}
                elif any(seen[k] != terms[k] for k in terms):
                    raise ValueError(f"{row['protein_id']}: labels differ between {seen[LABEL_SOURCE]} and {root}")
    return labels


def cached_rows(cache_dir: str, protein_id: str) -> int | None:
    """Row count of a protein's cached span, read from the safetensors header only.

    Args:
        cache_dir: Split cache directory.
        protein_id: Cache key.

    Returns:
        The row count, or None if the protein is not cached.
    """
    path = os.path.join(cache_dir, f"{protein_id}.safetensors")
    if not os.path.exists(path):
        return None
    with safe_open(path, framework="pt") as handle:
        return int(handle.get_slice(CACHE_KEY).get_shape()[0])


def sft_token_ids(tokenizer: AutoTokenizer, messages: list[dict[str, Any]], n_tokens: int) -> list[int]:
    """Token ids exactly as eval builds them: template with list content, then pad expansion.

    Args:
        tokenizer: The prepared checkpoint's tokenizer.
        messages: Eval's prompt messages.
        n_tokens: Protein span length.

    Returns:
        The prompt's token ids.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=ENABLE_THINKING)
    text = text.replace(PROTEIN_PAD_TOKEN, PROTEIN_PAD_TOKEN * n_tokens, 1)
    text = text.replace(GO_GRAPH_PAD_TOKEN, GO_GRAPH_PAD_TOKEN * NUM_GO_TOKENS, 1)
    return tokenizer(text)["input_ids"]


def rl_token_ids(tokenizer: AutoTokenizer, messages: list[dict[str, str]], n_tokens: int) -> list[int]:
    """Token ids of the RL prompt, with the GO ids mapped back to `<|go_graph_pad|>`.

    Args:
        tokenizer: The prepared checkpoint's tokenizer.
        messages: String-content RL messages.
        n_tokens: Protein span length.

    Returns:
        The prompt's token ids in SFT's vocabulary.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=ENABLE_THINKING)
    text = text.replace(PROTEIN_PAD_TOKEN, PROTEIN_PAD_TOKEN * n_tokens, 1)
    go_ids = range(FIRST_GO_ID, FIRST_GO_ID + NUM_GO_TOKENS)
    return [GO_GRAPH_PAD_ID if i in go_ids else i for i in tokenizer(text)["input_ids"]]


def check_parity(tokenizer: AutoTokenizer, rows: list[tuple[list[dict[str, Any]], list[dict[str, str]], int]]) -> None:
    """Require RL and eval tokenizations to agree on every probed row.

    Args:
        tokenizer: The prepared checkpoint's tokenizer.
        rows: `(eval messages, RL messages, n_tokens)` per probed row.

    Raises:
        RuntimeError: A row tokenizes differently, or a GO run is not exactly the baked tokens.
    """
    for index, (sft_messages, messages, n_tokens) in enumerate(rows):
        expected = sft_token_ids(tokenizer, sft_messages, n_tokens)
        got = rl_token_ids(tokenizer, messages, n_tokens)
        if got != expected:
            first = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b), min(len(got), len(expected)))
            raise RuntimeError(
                f"row {index}: RL prompt differs from eval's at token {first} "
                f"(lengths {len(got)} vs {len(expected)}): {got[first:first + 5]} vs {expected[first:first + 5]}"
            )
        raw = tokenizer(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=ENABLE_THINKING))["input_ids"]
        start = raw.index(FIRST_GO_ID)
        if raw[start:start + NUM_GO_TOKENS] != list(range(FIRST_GO_ID, FIRST_GO_ID + NUM_GO_TOKENS)):
            raise RuntimeError(f"row {index}: the GO run is not the {NUM_GO_TOKENS} baked tokens in order")
    print(f"[parity] {len(rows)} rows: RL prompts are token-identical to eval's")


def build_row(sample: dict[str, Any], messages: list[dict[str, str]], n_tokens: int, split: str,
              tier2: dict[str, list[str]]) -> dict[str, Any]:
    """One SkyRL sample.

    Args:
        sample: Eval row.
        messages: RL prompt messages.
        n_tokens: Protein span length.
        split: Source split.
        tier2: This protein's tier-2 labels.

    Returns:
        The sample dict.
    """
    return {
        "data_source": DATA_SOURCE,
        "prompt": messages,
        "env_class": ENV_CLASS,
        "reward_spec": {"method": "rule", "ground_truth": sample["protein_id"]},
        "extra_info": {"split": split, "protein_id": sample["protein_id"],
                       **{k: sample.get(k) for k in LABEL_FIELDS}, **tier2},
        "modalities": {MODALITY_ID: [{"protein_id": sample["protein_id"], "n_tokens": n_tokens}]},
    }


def build(table: str, split: str, cache_dir: str, checkpoint: str, output: str, limit: int, parity_rows: int,
          dataset_cache_dir: str, labels_from: list[str], keep: set[str] | None = None) -> None:
    """Build and write one split.

    Args:
        table: Prompt table directory.
        split: `train`, `val` or `test`.
        cache_dir: ESM3 cache built for this split.
        checkpoint: Prepared RL checkpoint whose tokenizer RL uses.
        output: Destination parquet.
        limit: Maximum rows to write, 0 for all.
        parity_rows: Rows to check against eval's tokenization.
        dataset_cache_dir: HF datasets cache.
        labels_from: Corpus directories to join tier-2 labels from; snapshots that disagree on a
            protein raise.
        keep: Proteins to build, or None for the whole split.

    Raises:
        FileExistsError: `output` already exists.
        RuntimeError: The parity check fails or nothing is left to write.
    """
    if os.path.exists(output):
        raise FileExistsError(f"{output} exists; refusing to overwrite")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    labels = load_labels(labels_from)
    samples = importlib.import_module("eval").load_dataset(eval_args(table, split, dataset_cache_dir))

    rows, probes = [], []
    dropped = {"no_labels": 0, "tier1_mismatch": 0, "no_cache": 0, "length_mismatch": 0, "no_scorable_truth": 0}
    for sample in samples:
        if keep is not None and sample["protein_id"] not in keep:
            continue
        known = labels.get(sample["protein_id"])
        if known is None:
            dropped["no_labels"] += 1
            continue
        if any(as_terms(sample.get(k)) != known[k] for k in TIER1_FIELDS):
            dropped["tier1_mismatch"] += 1
            continue
        n_tokens = cached_rows(cache_dir, sample["protein_id"])
        if n_tokens is None:
            dropped["no_cache"] += 1
            continue
        if n_tokens != min(len(sample["sequence"]), MAX_LENGTH_PROTEIN) + 2:
            dropped["length_mismatch"] += 1
            continue

        sft_messages = prompt_messages(sample)
        messages = rl_messages(sft_messages)
        if len(probes) < parity_rows:
            probes.append((sft_messages, messages, n_tokens))
        row = build_row(sample, messages, n_tokens, split, {k: known[k] for k in (*TIER2_FIELDS, LABEL_SOURCE)})
        if not ground_truth(row["extra_info"])[0]:
            dropped["no_scorable_truth"] += 1
            continue
        rows.append(row)
        if limit and len(rows) >= limit:
            break

    if not rows:
        raise RuntimeError(f"no rows left from {table}:{split} (dropped {dropped})")
    check_parity(tokenizer, probes)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    Dataset.from_list(rows).to_parquet(output)
    print(f"[{split}] wrote {len(rows)} rows -> {output}; dropped {dropped}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", required=True, help="prompt table directory, as eval reads it")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--cache_dir", required=True, help="ESM3 layer-37 cache built for this split")
    parser.add_argument("--checkpoint", required=True, help="prepared RL checkpoint (its tokenizer)")
    parser.add_argument("--output", required=True, help="destination parquet")
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    parser.add_argument("--parity_rows", type=int, default=64)
    parser.add_argument("--dataset_cache_dir", default=os.path.join(paths.REPO_ROOT, "cache", "datasets"))
    parser.add_argument("--labels_from", nargs="+", default=[paths.CORPUS],
                        help="corpus snapshots to join go_*_tier2 labels from; disagreeing snapshots raise")
    parser.add_argument("--ids", default="", help="file of protein ids to build (one per line); default all")
    args = parser.parse_args()
    keep = None
    if args.ids:
        with open(args.ids) as handle:
            keep = {line.strip() for line in handle if line.strip()}
    build(args.table, args.split, args.cache_dir, args.checkpoint, args.output, args.limit, args.parity_rows,
          args.dataset_cache_dir, args.labels_from, keep)


if __name__ == "__main__":
    main()
