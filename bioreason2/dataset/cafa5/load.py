# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Load a prompt table into the chat format training and evaluation share.

A prompt table is what `scripts/data/build_question.py` writes: one row
per protein with the finished `question` (the single user turn) and, on traced rows, the
teacher's `assistant_reasoning` and `assistant_answer`. The question is read verbatim, so
training and evaluation see the same bytes for a row and no flag downstream can change them.
"""

import ast
import os
import traceback

from datasets import load_dataset

from bioreason2.dataset.cafa5.format import format_cafa5_for_protein_llm
from bioreason2.dataset.utils import truncate_protein

SPLITS = ("train", "validation", "test")
DEBUG_ROWS = 50


def _warn_if_structures_unresolved(dataset, structure_dir, sample=512):
    """Fail when structure_dir is set but the files are not actually there.

    Missing structures are handled silently downstream: collate.py falls back to
    empty coordinates, so a wrong structure_dir (or shards extracted without
    gunzipping, leaving .cif.gz names) degrades the model to sequence-only with
    no error anywhere.

    This RAISES rather than warns. A warning was not enough: a partial directory
    took a 24-shard run to 71% coverage, printed one line into one of 24 chunk
    logs, and produced 31,552 traces that looked healthy and were quietly
    sequence-only for 29% of proteins. Silence is the failure mode, so the check
    that guards it cannot itself be quiet.

    Set BRP_ALLOW_PARTIAL_STRUCTURES=1 to proceed anyway; sequence-only is a
    legitimate protocol, but it has to be chosen rather than stumbled into.

    Args:
        dataset: the split being loaded, carrying a ``structure_path`` column.
        structure_dir: directory the paths were resolved against.
        sample: how many rows to probe.

    Raises:
        RuntimeError: when fewer than 99% of sampled structures resolve.
    """
    try:
        paths = [p for p in dataset["structure_path"][:sample] if p]
    except Exception:
        return
    if not paths:
        return
    found = sum(1 for p in paths if os.path.exists(p))
    pct = 100.0 * found / len(paths)
    if pct >= 99.0:
        print(f"Structure files resolve: {found}/{len(paths)} sampled ({pct:.1f}%)")
        return

    msg = (
        f"only {found}/{len(paths)} ({pct:.1f}%) sampled structure files exist under "
        f"structure_dir={structure_dir}. Missing structures are SILENTLY ignored "
        f"(empty coordinates), so this run would be sequence-only for the rest and "
        f"would not be comparable to anything scored with structures. "
        f"Take EVERY shard: python scripts/data/download_assets.py --dest <dir>  "
        f"(a partial or --referenced-only mirror is the usual cause). "
        f"Set BRP_ALLOW_PARTIAL_STRUCTURES=1 to proceed deliberately, or pass "
        f"--use_structure False to run sequence-only on purpose."
    )
    if os.environ.get("BRP_ALLOW_PARTIAL_STRUCTURES") == "1":
        print("\n" + "!" * 78 + f"\n! PROCEEDING ANYWAY: {msg}\n" + "!" * 78 + "\n")
        return
    raise RuntimeError(msg)


def _add_structure_prefix(example, structure_dir):
    """Resolve a row's structure file against the structure directory.

    Args:
        example: one row.
        structure_dir: the flat structure directory.

    Returns:
        The row with an absolute `structure_path`, or unchanged when it has none.
    """
    if example["structure_path"] is not None:
        example["structure_path"] = os.path.join(structure_dir, example["structure_path"])
    return example


def _add_ground_truth_go_terms(example):
    """Add `ground_truth_go_terms`, the row's GO ids joined by spaces.

    Args:
        example: one row, with `go_ids` as a list or its string repr.

    Returns:
        The row with the field added.
    """
    go_ids = example.get("go_ids", [])
    if go_ids is None:
        go_ids = []
    if isinstance(go_ids, str):
        go_ids = ast.literal_eval(go_ids) if go_ids else []
    example["ground_truth_go_terms"] = " ".join(go_ids)
    return example


def _format_reasoning_prompt(example):
    """The row's prompt: its pre-rendered question as the one user turn, no system turn.

    Args:
        example: one row of a prompt table.

    Returns:
        The row with `prompt` and `go_aspect` set.

    Raises:
        ValueError: when the row has no `question`; a table without one was not built by
            build_question.py and has no prompt this code can train or score on.
    """
    if not example.get("question"):
        raise ValueError(
            f"row {example.get('protein_id')!r} has no `question`. Prompt tables come from "
            f"scripts/data/build_question.py (README, Evaluate on the benchmark).")
    return {
        **example,
        "prompt": {
            "system": "",
            "user": example["question"],
            "assistant_reasoning": example.get("assistant_reasoning") or "",
            "assistant_answer": example.get("assistant_answer") or "",
        },
        "go_aspect": example.get("go_aspect") or "all",
    }


def _truncate(example, max_length):
    """Clip the row's sequence to the protein budget.

    Args:
        example: one row.
        max_length: residues kept.

    Returns:
        The row, sequence truncated.
    """
    return truncate_protein(example, max_length)


def _process_split(dataset, max_length, return_as_chat_template, structure_dir, num_proc, debug):
    """Every transformation one split goes through, in order.

    Args:
        dataset: the split as loaded.
        max_length: residues kept per sequence.
        return_as_chat_template: convert the prompt to the chat-message list.
        structure_dir: flat structure directory, or None to leave paths as stored.
        num_proc: worker processes for `map`.
        debug: keep only the first DEBUG_ROWS rows.

    Returns:
        The processed split.
    """
    if debug and len(dataset) > DEBUG_ROWS:
        print(f"Debug mode: limiting split from {len(dataset)} to {DEBUG_ROWS} samples")
        dataset = dataset.select(range(DEBUG_ROWS))

    dataset = dataset.map(_add_ground_truth_go_terms, num_proc=num_proc,
                          desc="Adding ground truth GO terms")
    dataset = dataset.map(_format_reasoning_prompt, num_proc=num_proc,
                          desc="Formatting reasoning prompts")
    dataset = dataset.map(_truncate, num_proc=num_proc, desc="Truncating sequences",
                          fn_kwargs={"max_length": max_length})
    if return_as_chat_template:
        dataset = dataset.map(format_cafa5_for_protein_llm, num_proc=num_proc,
                              desc="Setting chat template")
    if structure_dir is not None:
        dataset = dataset.map(_add_structure_prefix, num_proc=num_proc, desc="Adding structure paths",
                              fn_kwargs={"structure_dir": structure_dir})
        _warn_if_structures_unresolved(dataset, structure_dir)
    return dataset


def load_cafa5_dataset(
    dataset: str = None,
    config: str = "default",
    max_length: int = 2000,
    cache_dir: str = None,
    structure_dir: str = None,
    num_proc: int = None,
    return_as_chat_template: bool = False,
    debug: bool = False,
):
    """Load a prompt table's splits, formatted for the Protein-LLM.

    Args:
        dataset: the prompt table, as a directory or a Hub id.
        config: the table's config name; build_question.py writes `default`.
        max_length: residues kept per sequence.
        cache_dir: HuggingFace datasets cache.
        structure_dir: flat structure directory the stored paths resolve against.
        num_proc: worker processes (None = all cores but two).
        return_as_chat_template: return each prompt as the chat-message list.
        debug: keep DEBUG_ROWS rows per split.

    Returns:
        (train, validation, test); a split the table does not carry is None.

    Raises:
        ValueError: when no table is named, or a row has no `question`.
    """
    # No default is correct: the arm's prompt is baked into the table, so
    # guessing one scores a checkpoint against the wrong prompt.
    if not dataset:
        raise ValueError(
            "load_cafa5_dataset() needs an explicit `dataset`: the SFT table "
            "from build_question.py, as a directory or a Hub id.")

    try:
        if num_proc is None:
            num_proc = os.cpu_count() - 2
        print(f"Using {num_proc} CPU cores for parallel processing")

        datasets = load_dataset(dataset, name=config, cache_dir=cache_dir)
        print(f"Loaded {dataset} ({config}): "
              + ", ".join(f"{s} {len(datasets[s])}" for s in datasets))

        splits = tuple(
            _process_split(datasets[name], max_length, return_as_chat_template, structure_dir,
                           num_proc, debug) if name in datasets else None
            for name in SPLITS)
        for name, split in zip(SPLITS, splits):
            print(f"  - {name}: {len(split) if split is not None else 'absent'}")
        return splits

    except Exception as e:
        # Returning empty lists here turned a transient HuggingFace read timeout into
        # a confusing "'list' object has no attribute 'shuffle'" further downstream,
        # and let callers proceed as though the dataset had loaded. Fail loudly.
        print(f"Failed to load dataset: {e}")
        traceback.print_exc()
        raise
