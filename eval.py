#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
CAFA Evaluation Script (batched inference)

Features:
- Batched inference: one vLLM call per batch
- Individual JSON file output per protein_id + go_aspect combination
- Resumable: a protein that already has a result file is skipped
- Multi-GPU safe concurrent execution via --num_chunks / --chunk_id
- Per-sample retry on OOM

Usage:
    python eval.py --ckpt_dir /path/to/checkpoint --evals_path /path/to/results [options]
"""

import argparse
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from bioreason2.models.protein_vllm import ProteinLLMModel
from bioreason2.dataset.cafa5.load import load_cafa5_dataset
from bioreason2.utils import str2bool

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STOP_TOKENS = ["<|im_end|>"]
EVAL_SPLITS = {"train": 0, "val": 1, "test": 2}

# GO Aspect mapping for cleaner filenames
GO_ASPECT_CODES = {"molecular_function": "MF", "cellular_component": "CC", "biological_process": "BP"}


def get_go_aspect_code(go_aspect: str) -> str:
    """Convert GO aspect to short code for cleaner filenames."""
    return GO_ASPECT_CODES.get(go_aspect, go_aspect)


def _get_ground_truth(sample: Dict[str, Any]) -> str:
    """Extracts the ground truth assistant reasoning and answer from the sample."""
    prompt_data = sample.get("prompt")
    if isinstance(prompt_data, list):
        for message in prompt_data:
            if message.get("role") == "assistant":
                reasoning = message.get("reasoning_content", "")
                answer = ""
                content = message.get("content", [])
                if isinstance(content, list) and content:
                    answer = content[0].get("text", "")
                return f"{reasoning}\n\n{answer}" if reasoning and answer else reasoning or answer
    return sample.get("answer", "")


def text_token_budget(args) -> int:
    """Tokenizer cap for the text part, so prompt + protein + GO fits max_model_len.

    The processor truncates at max_length_text + 200 GO tokens + max_length_protein + 2;
    letting that exceed the context window makes vLLM reject the request outright.
    """
    return max(1, args.max_model_len - 200 - args.max_length_protein - 2)


def initialize_model(args) -> ProteinLLMModel:
    """Initialize and return the ProteinLLMModel."""
    print(f"Loading ProteinLLMModel from checkpoint: {args.ckpt_dir}...")
    model = ProteinLLMModel(
        ckpt_dir=args.ckpt_dir,
        protein_model_name=args.protein_model_name,
        protein_embedding_layer=args.protein_embedding_layer,
        go_obo_path=args.go_obo_path,
        precomputed_embeddings_path=args.precomputed_embeddings_path,
        max_length_protein=args.max_length_protein,
        max_length_text=text_token_budget(args),
        max_model_len=args.max_model_len,
        protein_projection_layernorm=args.protein_projection_layernorm,
        protein_projection_norm=args.protein_projection_norm,
        protein_projection_norm_target=args.protein_projection_norm_target,
        go_hidden_dim=args.go_hidden_dim,
        go_num_gat_layers=args.go_num_gat_layers,
        go_num_heads=args.go_num_heads,
        go_num_reduced_embeddings=args.go_num_reduced_embeddings,
        go_embedding_dim=args.go_embedding_dim,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        text_model_finetune=False,
        go_model_finetune=False,
    )
    print("Model initialized successfully.")
    return model


def load_dataset(args):
    """Load the evaluation split, shuffled with the run's seed and cut to this chunk.

    Args:
        args: eval.py's parsed arguments.

    Returns:
        The chunk's rows, a datasets.Dataset.

    Raises:
        ValueError: when the table has no such split, or it is empty.
    """
    print(f"\nLoading dataset (split: {args.cafa5_dataset_split})...")
    splits = load_cafa5_dataset(
        dataset=args.cafa5_dataset,
        config=args.cafa5_dataset_name,
        cache_dir=args.dataset_cache_dir,
        max_length=args.max_length_protein,
        return_as_chat_template=True,
        # Sequence-only runs skip the structure-coverage check, which would otherwise
        # refuse a missing or partial mirror the run is not going to read.
        structure_dir=args.structure_dir if args.use_structure else None,
        debug=args.debug,
    )
    dataset = splits[EVAL_SPLITS[args.cafa5_dataset_split]]
    if dataset is None:
        raise ValueError(f"{args.cafa5_dataset} has no '{args.cafa5_dataset_split}' split.")

    _shuf = getattr(args, "shuffle_sequence", "none") or "none"
    if _shuf != "none":
        # The embedding cache is keyed by protein_id and validated on ROW COUNT,
        # which every shuffle mode preserves by construction. A cached run would
        # therefore serve the REAL protein's embeddings against a corrupted
        # sequence and report a null result that looks like a finding. Fatal.
        if os.environ.get("PROTEIN_EMBED_CACHE"):
            raise SystemExit(
                "--shuffle_sequence is incompatible with PROTEIN_EMBED_CACHE: the "
                "cache is keyed on protein_id and its only integrity check is the "
                "row count, which corruption preserves. It would silently return "
                "the intact protein's embeddings. Unset PROTEIN_EMBED_CACHE.")
        print(f"SHUFFLING SEQUENCES AT EVAL: mode={_shuf} seed={args.seed}")
        dataset = apply_sequence_shuffle(dataset, _shuf, args.seed)
    if not dataset or len(dataset) == 0:
        raise ValueError(f"Dataset split '{args.cafa5_dataset_split}' is empty or failed to load.")

    dataset = dataset.shuffle(seed=args.seed)
    n_samples = len(dataset) if args.max_samples <= 0 else min(args.max_samples, len(dataset))

    # Handle chunking for multi-GPU processing
    if args.num_chunks > 1:
        chunk_size = n_samples // args.num_chunks
        start_idx = args.chunk_id * chunk_size
        # Last chunk gets any remaining samples
        end_idx = n_samples if args.chunk_id == args.num_chunks - 1 else start_idx + chunk_size
        print(f"Processing chunk {args.chunk_id + 1}/{args.num_chunks}: samples {start_idx} to {end_idx - 1}")
        samples = dataset.select(range(start_idx, end_idx))
    else:
        print("Processing full dataset (no chunking)")
        samples = dataset.select(range(n_samples))

    print(f"Loaded {len(samples)} samples for evaluation.")
    return samples


def filter_unprocessed_samples(samples, evals_path: str):
    """Drop samples that already have a result file, returning a Dataset.

    Reads protein_id/go_aspect columnwise; row-by-row iteration would decode the
    full sequence and prompt just to read two strings.
    """
    os.makedirs(evals_path, exist_ok=True)
    processed_ids = set()

    for filename in os.listdir(evals_path):
        if filename.endswith(".json"):
            # Filename is {protein_id}_{go_aspect_code}_k{i:02d}.json
            parts = filename.split("_")
            if len(parts) >= 2:
                processed_ids.add(f"{parts[0]}_{parts[1]}")
    print(f"Found {len(processed_ids)} samples with at least one result file.")

    protein_ids = samples["protein_id"]
    go_aspects = samples["go_aspect"]
    keep = [
        i
        for i, (pid, aspect) in enumerate(zip(protein_ids, go_aspects))
        if f"{pid}_{get_go_aspect_code(aspect)}" not in processed_ids
    ]

    print(f"Total samples: {len(samples)}")
    print(f"Already processed: {len(samples) - len(keep)}")
    print(f"Remaining to process: {len(keep)}")
    return samples.select(keep)


def batch_slice(dataset, start: int, end: int) -> List[Dict[str, Any]]:
    """Rows [start, end) as a list of dicts, via a single Arrow slice."""
    columns = dataset[start:end]
    n = end - start
    return [{key: values[i] for key, values in columns.items()} for i in range(n)]


def save_result(result_record: Dict[str, Any], protein_id: str, go_aspect: str, evals_path: str, k_idx: int = 0) -> None:
    """Save individual result to its own JSON file using short GO aspect codes."""
    go_aspect_code = get_go_aspect_code(go_aspect)
    result_filename = f"{protein_id}_{go_aspect_code}_k{k_idx:02d}.json"
    with open(os.path.join(evals_path, result_filename), "w") as f:
        json.dump(result_record, f, indent=4)


def log_error(error_log_path: str, error_type: str, protein_id: str, go_aspect: str, error_msg: str = "") -> None:
    """Append an error record to the run's error log."""
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "error_type": error_type,
        "protein_id": protein_id,
        "go_aspect": go_aspect,
        "error_message": error_msg or ("Out of Memory" if error_type == "oom" else "Unknown error"),
    }

    errors = []
    if os.path.exists(error_log_path):
        try:
            with open(error_log_path) as f:
                errors = json.load(f)
        except Exception:
            errors = []
    errors.append(record)

    with open(error_log_path, "w") as f:
        json.dump(errors, f, indent=4)


def _build_prompt_string(model, sample: Dict[str, Any], args) -> Optional[str]:
    """Chat-template string for one sample, or None if it cannot be built."""
    conversation_data = sample.get("prompt")
    if conversation_data is None:
        return None

    # Keep system and user messages; stop at the first assistant turn
    user_conversation = []
    for message in conversation_data:
        if message.get("role") in ["system", "user"]:
            user_conversation.append(message)
        elif message.get("role") == "assistant":
            break

    return model.text_tokenizer.apply_chat_template(
        user_conversation,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )


def _batch_structure_coords(samples: List[Dict[str, Any]], max_length_protein: int) -> torch.Tensor:
    """Backbone N/CA/C coordinates per sample, padded to the batch max.

    Eval previously passed no structure at all: the processor has no structure
    track, so `structure_coords` was always None and every protein was encoded
    sequence-only -- even though training conditions ESM3 on structure via the
    collator. That train/eval mismatch is silent, because the encoder simply
    treats absent coordinates as sequence-only.

    This mirrors `collate.py` so eval sees structure the same way training did.
    A missing or unparseable file degrades to zero-length coordinates, which the
    encoder skips -- matching the collator rather than raising. Padding is NaN, so
    a protein without structure stays sequence-only in a batch with structured ones.
    """
    from bioreason2.dataset.cafa5.structure_io import backbone_resolving_alternates

    arrays = []
    for sample in samples:
        path = sample.get("structure_path")
        seq = (sample.get("sequence") or "")[:max_length_protein]
        coords = np.full((0, 3, 3), np.nan)
        if path and os.path.exists(path) and seq:
            try:
                # Same verification the trainer uses: confirm the structure is
                # this protein and align it, else fall back to sequence-only.
                aligned = backbone_resolving_alternates(path, seq)
                if aligned is not None:
                    coords = np.asarray(aligned)
            except Exception:
                coords = np.full((0, 3, 3), np.nan)
        if coords.ndim != 3 or coords.shape[1:] != (3, 3):
            coords = np.full((0, 3, 3), np.nan)
        arrays.append(coords[:max_length_protein])

    max_len = max((a.shape[0] for a in arrays), default=0)
    if max_len == 0:
        return torch.zeros((len(arrays), 0, 3, 3), dtype=torch.float32)
    padded = torch.full((len(arrays), max_len, 3, 3), float("nan"), dtype=torch.float32)
    for i, arr in enumerate(arrays):
        if arr.shape[0]:
            padded[i, : arr.shape[0]] = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32)
    return padded


# --- systematic sequence corruption (eval only) ----------------------------
# DIVERGENCE FROM UPSTREAM (additive; inactive unless --shuffle_sequence is set).
#
# `--ablate_context sequence` removes the protein tokens entirely. This instead
# keeps the token COUNT and destroys only the information they carry, which
# separates two things the deletion test cannot: whether the model IGNORES the
# encoder, or merely needs something in the slot.
#
# Every mode preserves len(sequence), so the prompt is token-for-token the same
# length as the intact baseline and the ONLY variable is embedding content.
# `swap` is the exception -- see below.
SHUFFLE_MODES = ("none", "shuffle", "block20", "reverse", "random", "swap")

# UniProt background amino-acid frequencies. `random` draws from these rather
# than uniformly: a uniform draw is detectable as synthetic on composition alone
# (real proteins are ~10% L and ~1% W), which would confound "no information"
# with "obviously not a protein".
_AA_BG = (("A", .0825), ("R", .0553), ("N", .0406), ("D", .0545), ("C", .0137),
          ("Q", .0393), ("E", .0675), ("G", .0707), ("H", .0227), ("I", .0596),
          ("L", .0966), ("K", .0584), ("M", .0242), ("F", .0386), ("P", .0470),
          ("S", .0656), ("T", .0534), ("W", .0108), ("Y", .0292), ("V", .0687))
_AA = [a for a, _ in _AA_BG]
_AA_W = [w for _, w in _AA_BG]


def _corrupt_sequence(seq, mode, rng):
    """One protein's sequence under `mode`. Length is always preserved."""
    if not seq or mode in ("none", ""):
        return seq
    if mode == "shuffle":                       # composition kept, all order gone
        a = list(seq); rng.shuffle(a); return "".join(a)
    if mode == "reverse":                       # local windows kept, reversed
        return seq[::-1]
    if mode == "block20":                       # motifs kept, domain order gone
        b = [seq[i:i + 20] for i in range(0, len(seq), 20)]
        rng.shuffle(b); return "".join(b)
    if mode == "random":                        # composition gone too
        return "".join(rng.choices(_AA, weights=_AA_W, k=len(seq)))
    raise ValueError("unknown --shuffle_sequence mode: %r" % mode)


def apply_sequence_shuffle(dataset, mode, seed):
    """Rewrite every row's `sequence` deterministically.

    Seeded per protein id (crc32), not per row index, so the corruption is
    identical however the dataset is ordered, chunked or resumed -- the same
    property `context_dropout` needs for DDP ranks to agree.
    """
    if mode in ("none", ""):
        return dataset
    if mode not in SHUFFLE_MODES:
        raise ValueError("--shuffle_sequence must be one of %s" % (SHUFFLE_MODES,))

    import zlib
    from random import Random

    if mode == "swap":
        # A real-but-WRONG protein: the confidently-misleading end of the ladder,
        # where every other mode is merely uninformative. Partners are adjacent
        # in length order so the token count barely moves (reported below); an
        # exact-length partner generally does not exist.
        seqs = dataset["sequence"]
        ids = dataset["protein_id"]
        order = sorted(range(len(seqs)), key=lambda i: (len(seqs[i] or ""), ids[i]))
        partner = {}
        for k in range(0, len(order) - 1, 2):
            a, b = order[k], order[k + 1]
            partner[ids[a]], partner[ids[b]] = seqs[b], seqs[a]
        if len(order) % 2:                      # odd one out keeps its own
            partner.setdefault(ids[order[-1]], seqs[order[-1]])
        deltas = [abs(len(partner[ids[i]] or "") - len(seqs[i] or "")) for i in range(len(seqs))]
        print("SEQUENCE SWAP: mean |delta length| = %.1f residues (max %d)"
              % (sum(deltas) / len(deltas), max(deltas)))

        def _swap(ex):
            return {"sequence": partner.get(ex["protein_id"], ex["sequence"])}

        return dataset.map(_swap, desc="Swapping sequences")

    def _one(ex):
        rng = Random(zlib.crc32(("%s|%s|%d" % (ex["protein_id"], mode, seed)).encode()))
        return {"sequence": _corrupt_sequence(ex["sequence"], mode, rng)}

    return dataset.map(_one, desc="Corrupting sequences (%s)" % mode)


_TRUNCATED = 0   # prompts clipped to the text budget, reported per chunk

# Text sections are ablated by building a prompt table without them; the sequence is
# the one part that lives outside the text.
ABLATABLE = {"sequence"}


def ablated_parts(args) -> set:
    """Parse --ablate_context once. Empty set means no ablation."""
    raw = getattr(args, "ablate_context", "") or ""
    parts = {p.strip().lower() for p in raw.split(",") if p.strip()}
    unknown = parts - ABLATABLE
    if unknown:
        raise ValueError(
            f"--ablate_context: unknown part(s) {sorted(unknown)}; "
            f"known: {sorted(ABLATABLE)}"
        )
    return parts


def prepare_batch_inputs(model: ProteinLLMModel, batch_samples: List[Dict[str, Any]], args) -> Optional[Dict[str, Any]]:
    """Tokenize a batch. Returns None when no sample in the batch is usable."""
    prompts, sequences, go_aspects, keep = [], [], [], []
    for sample in batch_samples:
        prompt = _build_prompt_string(model, sample, args)
        sequence = sample.get("sequence")
        if prompt is None or sequence is None:
            continue
        prompts.append(prompt)
        sequences.append(sequence)
        go_aspects.append(sample.get("go_aspect", "all"))
        keep.append(sample)

    if not prompts:
        return None

    # Left-pad so content sits at the end of every row. generate() strips padding
    # before handing embeddings to vLLM either way; this matches predict.py.
    _ablate_seq = "sequence" in ablated_parts(args)
    if _ablate_seq:
        # Passing an empty protein list stops the processor EXPANDING the
        # placeholder, but leaves the literal <|protein_pad|> token in the text --
        # one per sample, which then has no embedding to match and trips the
        # count assertion. Remove the placeholder from the prompt too, so the
        # ablated prompt genuinely contains no protein slot.
        from bioreason2.models.special_tokens import get_token
        _pad = get_token("protein_pad")
        prompts = [p.replace(_pad, "") for p in prompts]

    original_padding_side = model.text_tokenizer.padding_side
    model.text_tokenizer.padding_side = "left"
    try:
        processed_inputs = model.processor(
            # Copy: the processor expands <|protein_pad|> in place.
            text=list(prompts),
            # `sequence` ablation passes an empty protein list per sample, so the
            # processor expands zero <|protein_pad|> tokens and no embedding is
            # required. Skipping injection while leaving the tokens in place would
            # trip the model's token-count-vs-embedding-count assertion.
            batch_protein_sequences=[[] for _ in sequences] if _ablate_seq
            else [[s] for s in sequences],
            batch_go_aspects=go_aspects,
            max_length_text=model.max_length_text,
            max_length_protein=model.max_length_protein,
            return_tensors="pt",
        )
    finally:
        model.text_tokenizer.padding_side = original_padding_side

    # Report truncation instead of letting it happen silently. The text budget is
    # max_model_len - 200 - max_length_protein - 2, so a window that is too small
    # quietly clips the prompt and degrades predictions with no error anywhere.
    _ids = processed_inputs.get("input_ids")
    if _ids is not None:
        _budget = model.max_length_text
        _n_at_cap = int((_ids.ne(model.text_tokenizer.pad_token_id).sum(dim=1) >= _budget).sum())
        if _n_at_cap:
            global _TRUNCATED
            _TRUNCATED += _n_at_cap
            print(f"⚠️  {_n_at_cap} prompt(s) hit the {_budget}-token text budget "
                  f"(cumulative {_TRUNCATED}). Raise --max_model_len if this is nonzero.",
                  flush=True)

    return {
        "input_ids": processed_inputs.get("input_ids").to(DEVICE),
        "attention_mask": processed_inputs.get("attention_mask").to(DEVICE),
        # Built here, not by the processor: the processor has no structure track.
        "structure_coords": (
            _batch_structure_coords(keep, model.max_length_protein).to(DEVICE)
            if getattr(args, "use_structure", True)
            else None
        ),
        "sequences": sequences,
        "protein_ids": [x.get("protein_id") for x in keep],
        # generate() must not load embeddings for proteins whose tokens are no
        # longer in the prompt. Kept as a flag rather than by emptying
        # "sequences": that field is also the per-record metadata written to the
        # output JSON, and blanking it broke record writing with IndexError.
        "ablate_sequence": _ablate_seq,
        "go_aspects": go_aspects,
        "prompts": prompts,
        "samples": keep,
    }


def process_batch(model: ProteinLLMModel, batch_samples: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    """Run one batched generation and build a result record per sample."""
    batch_inputs = prepare_batch_inputs(model, batch_samples, args)
    if batch_inputs is None:
        return []

    with torch.inference_mode():
        generated_outputs = model.generate(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            protein_sequences=[] if batch_inputs["ablate_sequence"] else batch_inputs["sequences"],
            batch_idx_map=[] if batch_inputs["ablate_sequence"]
            else list(range(len(batch_inputs["sequences"]))),
            protein_ids=[] if batch_inputs["ablate_sequence"] else batch_inputs.get("protein_ids"),
            go_aspects=batch_inputs["go_aspects"],
            structure_coords=batch_inputs["structure_coords"],
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            stop=STOP_TOKENS,
        )

    results = []
    for i, sample in enumerate(batch_inputs["samples"]):
        text = generated_outputs[i] if i < len(generated_outputs) else ""
        sequence = batch_inputs["sequences"][i]
        results.append({
            "protein_id": sample.get("protein_id"),
            "go_aspect": batch_inputs["go_aspects"][i],
            "ground_truth": _get_ground_truth(sample),
            "generated_response": text,
            # Always True: marking an empty generation as failed makes
            # cafa_evals.py drop the protein from ground truth too, inflating Fmax.
            "success": True,
            "protein_sequence": sequence,
            "input_prompt": batch_inputs["prompts"][i],
            "sequence_length": len(sequence) if sequence else 0,
            "go_bp": sample.get("go_bp", ""),
            "go_mf": sample.get("go_mf", ""),
            "go_cc": sample.get("go_cc", ""),
            "go_bp_leaf": sample.get("go_bp_leaf", ""),
            "go_mf_leaf": sample.get("go_mf_leaf", ""),
            "go_cc_leaf": sample.get("go_cc_leaf", ""),
            # v2 GATING. cafa_evals reads `scored_aspects` to decide which
            # (protein, aspect) pairs may be scored at all -- a row can carry
            # tier-1 terms and still be unscorable: immature_annotation (382 test
            # rows), tier1_retracted_since_gaf226 (854), train_subsequence (244).
            #
            # cafa_evals was taught to read it and then never received it, because
            # this record did not carry the column. It silently fell back to its
            # ungated path ("any aspect with a non-empty go_* column") and scored ~6,800
            # proteins per aspect instead of the ~1,700 v2 permits, charging the
            # model against label sets v2 declares incomplete or leaked. A correct
            # gate fed nothing looks exactly like no gate at all.
            "scored_aspects": sample.get("scored_aspects", None),
            "excluded_reason": sample.get("excluded_reason", None),
        })
    return results


def print_final_statistics(newly_processed: int, empty_responses: int, total_time: float, evals_path: str) -> None:
    """Print final evaluation statistics."""
    total_files = len([f for f in os.listdir(evals_path) if f.endswith(".json")])

    print("\nEvaluation complete.")
    print(f"Processed {newly_processed} new samples in {total_time:.2f}s")
    if newly_processed > 0:
        print(f"Processing rate: {newly_processed / total_time:.2f} samples/s")
    if empty_responses:
        print(f"WARNING: {empty_responses} samples generated an empty response; "
              f"they are scored as a miss, not dropped.")
    print(f"Total result files: {total_files} in directory: {evals_path}")


def run_inference(args):
    """Orchestrate data loading, batched model inference, and result saving."""
    print("--- Starting batched CAFA inference ---")
    print(f"Batch size: {args.batch_size}")

    # Not .json: everything scanning evals_path for *.json treats those as results.
    error_log_path = os.path.join(args.evals_path, f"evaluation_errors_chunk{args.chunk_id:03d}.log")

    try:
        model = initialize_model(args)
        samples = load_dataset(args)
        unprocessed = filter_unprocessed_samples(samples, args.evals_path)

        n = len(unprocessed)
        if n == 0:
            print("All samples already processed. Nothing to do.")
            return

        bs = max(1, args.batch_size)
        num_batches = (n + bs - 1) // bs
        print(f"\nStarting inference: {n} samples, batch_size={bs}, {num_batches} batches, pass@{args.pass_at_k}")

        t_start = time.time()
        successfully_processed = 0
        empty_responses = 0

        for batch_idx in tqdm(range(num_batches), desc="Processing batches", unit="batch"):
            batch_samples = batch_slice(unprocessed, batch_idx * bs, min((batch_idx + 1) * bs, n))

            for k_idx in range(args.pass_at_k):
                try:
                    records = process_batch(model, batch_samples, args)

                except torch.cuda.OutOfMemoryError:
                    print(f"\nCUDA OOM on batch {batch_idx} (k={k_idx}); retrying its samples individually.")
                    torch.cuda.empty_cache()
                    records = []
                    for sample in batch_samples:
                        try:
                            records.extend(process_batch(model, [sample], args))
                        except Exception as exc:
                            log_error(error_log_path, "oom", sample.get("protein_id", "unknown"),
                                      sample.get("go_aspect", "all"), str(exc))
                            torch.cuda.empty_cache()

                except Exception as exc:
                    ids = ",".join(str(s.get("protein_id")) for s in batch_samples)
                    print(f"\nError on batch {batch_idx} (k={k_idx}) [{ids}]: {exc}")
                    traceback.print_exc()
                    for sample in batch_samples:
                        log_error(error_log_path, "batch_error", sample.get("protein_id", "unknown"),
                                  sample.get("go_aspect", "all"), str(exc))
                    continue

                for record in records:
                    save_result(record, record["protein_id"], record["go_aspect"], args.evals_path, k_idx=k_idx)
                    if not record["generated_response"].strip():
                        empty_responses += 1
                    if k_idx == 0:
                        successfully_processed += 1

        print_final_statistics(successfully_processed, empty_responses, time.time() - t_start, args.evals_path)

        # Every batch can fail and `continue`, leaving a chunk that wrote nothing and
        # still exited 0 -- which a job scheduler reports as Completed.
        if successfully_processed == 0:
            raise SystemExit(f"no results written to {args.evals_path}: every batch failed")

    except Exception as exc:
        # Must not swallow this: exiting 0 makes a dead shard look COMPLETED.
        print(f"Critical Error: {exc}")
        traceback.print_exc()
        raise SystemExit(1)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup and return the argument parser."""
    parser = argparse.ArgumentParser(description="Batched CAFA inference with ProteinLLMModel")

    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--ckpt_dir", type=str, required=True, help="Path to the ProteinLLMModel checkpoint directory."
    )
    model_group.add_argument(
        "--protein_model_name", type=str, default="esm3_sm_open_v1", help="Name of the protein encoder model."
    )
    model_group.add_argument(
        "--protein_embedding_layer",
        type=int,
        default=37,
        help="ESM3 layer to extract embeddings from. 37 is the released checkpoint; -1 takes the final output. A mismatch is SILENT: both are 1536-dim, so protein_projection loads cleanly and the model runs to completion producing noise."
    )
    model_group.add_argument("--go_obo_path", type=str, required=True, help="Path to GO ontology .obo file.")
    model_group.add_argument(
        "--precomputed_embeddings_path",
        type=str,
        required=True,
        help="Path to directory with precomputed GO embeddings.",
    )
    model_group.add_argument(
        "--protein_projection_layernorm",
        type=str2bool,
        default=False,
        help="Must match how the checkpoint was trained; it adds parameters.",
    )
    model_group.add_argument(
        "--protein_projection_norm",
        type=str,
        default="",
        help="Normalisation on the projection tail. Empty derives it from --protein_projection_layernorm, so every existing run is unchanged. blocknorm keeps per-residue magnitudes and fixes only the scale; lnscale does the opposite. Must match how the checkpoint was trained.",
    )
    model_group.add_argument(
        "--protein_projection_norm_target",
        type=float,
        default=0.0,
        help="Initial output RMS for the variants that take one; 0 uses the variant's default (1.0 for blocknorm, 7.4 for lnscale).",
    )
    model_group.add_argument("--max_model_len", type=int, default=16384,
                             help="Maximum context length for vLLM (prompt + generation). "
                                  "build_question.py sizes every question for 16384 with "
                                  "8000 new tokens.")
    model_group.add_argument(
        "--gpu_memory_utilization", type=float, default=0.5,
        help="Fraction of GPU memory vLLM may use. ESM3, the GO encoder and the "
             "batch's prompt embeddings are allocated outside it, so leave headroom.",
    )
    model_group.add_argument(
        "--max_num_seqs", type=int, default=256,
        help="Upper bound on sequences vLLM runs concurrently.",
    )
    model_group.add_argument(
        "--tensor_parallel_size", type=int, default=1,
        help="GPUs to shard the weights over. A 27B is ~55 GB of bf16 weights and "
             "does not fit one 80 GB card alongside its KV cache; use 2 or more.",
    )
    model_group.add_argument(
        "--go_hidden_dim", type=int, default=512, help="Hidden dimension for GO GAT layers (must match training)."
    )
    model_group.add_argument(
        "--go_num_gat_layers", type=int, default=3, help="Number of GAT layers in GO encoder (must match training)."
    )
    model_group.add_argument(
        "--go_num_heads", type=int, default=8, help="Number of attention heads in GO GAT (must match training)."
    )
    model_group.add_argument(
        "--go_num_reduced_embeddings",
        type=int,
        default=200,
        help="Number of reduced GO embeddings (must match training).",
    )
    model_group.add_argument(
        "--go_embedding_dim", type=int, default=2560, help="GO embedding dimension (must match training)."
    )

    # Dataset options
    dataset_group = parser.add_argument_group("Dataset Configuration")
    dataset_group.add_argument(
        "--cafa5_dataset", type=str, required=True,
        help="SFT table from build_question.py (directory or Hub id). The arm's "
             "prompt is baked into it, so there is no correct default.",
    )
    dataset_group.add_argument(
        "--cafa5_dataset_name", type=str, default="default",
        help="Config name. build_question.py writes one, `default`.",
    )
    dataset_group.add_argument(
        "--cafa5_dataset_split", type=str, default="test", choices=list(EVAL_SPLITS),
        help="Which split to evaluate; the v2 test set is `test`. A split the table "
             "does not carry raises.",
    )
    dataset_group.add_argument("--dataset_cache_dir", type=str, default=None)
    dataset_group.add_argument(
        "--structure_dir", type=str, default=None
    )
    dataset_group.add_argument(
        "--use_structure", type=str2bool, default=True,
        help="Condition ESM3 on backbone structure when a structure file exists. "
             "False encodes every protein sequence-only, as an older eval did while "
             "training used structure.",
    )
    dataset_group.add_argument("--seed", type=int, default=23)
    dataset_group.add_argument("--debug", type=str2bool, default=False)
    dataset_group.add_argument(
        "--max_length_protein", type=int, default=2000, help="Maximum length of protein sequences. Must match training or the protein is truncated differently than it was trained."
    )
    dataset_group.add_argument("--enable_thinking", type=str2bool, default=True)
    dataset_group.add_argument(
        "--shuffle_sequence", type=str, default="none",
        help="Corrupt the protein sequence at inference while preserving its LENGTH, "
             "so the protein-token count is unchanged and only the embedding content "
             "varies. none | shuffle (permute residues) | block20 (permute 20-mer "
             "blocks) | reverse | random (resample from UniProt background) | swap "
             "(a different, real protein of near-identical length).",
    )
    dataset_group.add_argument(
        "--ablate_context", type=str, default="",
        help="`sequence` removes the protein embeddings and their placeholder from the "
             "prompt. Empty = the full prompt.",
    )

    # Evaluation controls
    eval_group = parser.add_argument_group("Evaluation Configuration")
    eval_group.add_argument("--max_samples", type=int, default=-1, help="Max samples to process (-1 for all).")
    eval_group.add_argument(
        "--batch_size", type=int, default=16,
        help="Samples per vLLM generation call, and so the number of sequences "
             "decoded concurrently. Raise for throughput; set 1 for one at a time.",
    )
    eval_group.add_argument("--max_new_tokens", type=int, default=8000,
                            help="Must leave room for the model to finish reasoning and emit "
                                 "its GO summary; truncating produces empty predictions.")
    eval_group.add_argument("--temperature", type=float, default=0.0)
    eval_group.add_argument("--top_p", type=float, default=0.95)
    eval_group.add_argument("--repetition_penalty", type=float, default=1.0)
    eval_group.add_argument(
        "--pass_at_k",
        type=int,
        default=1,
        help="Number of inference attempts per sample for pass@k evaluation (default: 1). Use temperature > 0 for diversity."
    )

    # Data chunking (optional)
    chunk_group = parser.add_argument_group("Data Chunking (Optional)")
    chunk_group.add_argument(
        "--num_chunks",
        type=int,
        default=1,
        help="Total number of chunks for distributed processing. Default: 1 (no chunking).",
    )
    chunk_group.add_argument(
        "--chunk_id", type=int, default=0, help="ID of this chunk (0-indexed). Only used when num_chunks > 1."
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--evals_path", type=str, required=True, help="Directory path to save individual evaluation results."
    )

    return parser


if __name__ == "__main__":
    parser = setup_argument_parser()
    args = parser.parse_args()
    run_inference(args)
