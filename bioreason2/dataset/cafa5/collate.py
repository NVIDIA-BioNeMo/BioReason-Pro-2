# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import os
from typing import Dict, List

import torch
import numpy as np

from bioreason2.models.pl.processing_pl import PLProcessor
from trl.data_utils import maybe_apply_chat_template

from esm.sdk.api import ESMProtein

# BioPython imports for structure parsing
from Bio.PDB import MMCIFParser, is_aa


# Parsers live in structure_io so the eval environment can import them without
# pulling this module's training-only dependency (trl).
from bioreason2.dataset.cafa5.structure_io import (  # noqa: F401
    _coords_from_cif, _coords_from_pdb, backbone_resolving_alternates,
)


def _marker_positions(row: torch.Tensor, marker: torch.Tensor) -> List[int]:
    """Start offsets of every occurrence of `marker` in `row`.

    `unfold` builds all length-m sliding windows as a view, so the comparison is one
    C-level op rather than a Python loop over the sequence.

    Args:
        row: 1-D token ids for one sequence.
        marker: 1-D token ids to find.

    Returns:
        Ascending list of indices where `marker` starts.
    """
    m = marker.numel()
    if row.numel() < m:
        return []

    hits = (row.unfold(0, m, 1) == marker).all(dim=1)

    return hits.nonzero(as_tuple=False).squeeze(1).tolist()


def _truncate_after_assistant_start(text: str) -> str:
    """
    Keep everything up to and including the first '<|im_start|>assistant\n',
    drop any assistant answer that follows. Used for inference mode.
    """
    marker = "<|im_end|>\n<|im_start|>assistant\n"
    idx = text.find(marker)
    if idx != -1:
        return text[: idx + len(marker)]
    return text


def qwen_protein_collate_fn(
    examples: List[Dict],
    processor: PLProcessor,
    max_length_text: int,
    max_length_protein: int,
    return_answer_in_batch: bool = False,
    inference_mode: bool = False,
    supervise_eos: bool = True,
) -> Dict:
    """Custom collate function for Qwen Protein models.

    Creates a batch with proper labels for supervised fine-tuning where only
    the assistant responses contribute to the loss calculation. Handles both
    text and protein structure data.

    Args:
        examples (List[Dict]): List of example dictionaries containing text,
            protein sequences, structure paths, and answers.
        processor (PLProcessor): Protein language model processor for tokenization.
        max_length_text (int): Maximum length for text sequences.
        max_length_protein (int): Maximum length for protein sequences.
        return_answer_in_batch (bool): Whether to include answers in the batch.
        inference_mode (bool): If True, truncate inputs after the assistant start
            marker to prepare for generation. Defaults to False.

    Returns:
        Dict: Batch dictionary containing:
            - input_ids: Tokenized input sequences
            - attention_mask: Attention masks for the sequences
            - structure_coords: Protein structure coordinates as tensor
            - labels: Labels for supervised fine-tuning (only assistant responses)
            - answer: Ground truth answers (if return_answer_in_batch=True)
            - prompt: Text prompts (if inference_mode=True)
    """
    prompts_text = [maybe_apply_chat_template(example, processor)["prompt"] for example in examples]

    # Restore the assistant turn's terminator before tokenizing.
    #
    # `format_cafa5_for_protein_llm` emits {"prompt": [user, assistant]}, and TRL's
    # maybe_apply_chat_template renders a prompt-ONLY example as a CONTINUATION: the
    # assistant content is included but its closing <|im_end|> is not. Measured on the
    # real structure, the rendered text contains <|im_end|> exactly once -- closing the
    # USER turn -- and nothing after the answer. The stop token therefore never appeared
    # in a training target, could never be learned, and generation ran to max_new_tokens.
    #
    # The fix appends the terminator at the text level, which keeps the existing
    # assistant-span masking intact. Inference mode must NOT get a terminator: there
    # the text is a prompt to continue.
    # supervise_eos=False reproduces the PRE-FIX collator exactly, so a run can be
    # compared against a baseline trained before the stop token was supervised. It
    # re-introduces a known bug on purpose; never use it for a run that is not being
    # matched to such a baseline.
    if not inference_mode and supervise_eos:
        _eos = processor.tokenizer.eos_token
        if _eos:
            prompts_text = [t if t.rstrip().endswith(_eos) else (t + _eos) for t in prompts_text]
    batch_protein_sequences = [example["protein_sequences"] for example in examples]
    batch_protein_structures = [example["structure_path"] for example in examples]
    batch_go_aspects = [example.get("go_aspect") for example in examples]

    batch = processor(
        text=prompts_text,
        batch_protein_sequences=batch_protein_sequences,
        batch_go_aspects=batch_go_aspects,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        max_length_text=max_length_text,
        max_length_protein=max_length_protein,
    )

    # -------------------------------------------------------------------
    # Process protein structure coordinates (dynamic padding to batch max)
    # -------------------------------------------------------------------
    structure_coord_list = []
    for struct_path, _seqs in zip(batch_protein_structures, batch_protein_sequences):
        # One structure file per example; `protein_sequences` is a list per example
        # (usually length 1). Verification uses the first sequence.
        _seq = _seqs[0] if isinstance(_seqs, (list, tuple)) and _seqs else (_seqs or "")
        if struct_path is not None and os.path.exists(struct_path) and _seq:
            try:
                # Verify the structure IS this protein and align it. Returns None
                # for a different splice form / precursor / wrong file, which
                # degrades to sequence-only rather than embedding wrong residues.
                coords = backbone_resolving_alternates(struct_path, _seq)
                if coords is None:
                    coords = np.full((0, 3, 3), np.nan)
            except Exception:
                coords = np.full((0, 3, 3), np.nan)
        else:
            coords = np.full((0, 3, 3), np.nan)

        # Truncate if number of residues exceeds max_length_protein
        if coords.shape[0] > max_length_protein:
            coords = coords[:max_length_protein]

        structure_coord_list.append(coords)

    # Dynamic padding: pad to batch max length (more memory efficient)
    lengths = [arr.shape[0] for arr in structure_coord_list]
    max_len = max(lengths) if lengths else 0
    batch_size = len(structure_coord_list)

    # Build padded tensor
    if max_len == 0:
        padded = torch.zeros((batch_size, 0, 3, 3), dtype=torch.float32)
    else:
        # NaN, not zero: the encoder reads an all-NaN row set as "no structure", while zeros
        # would pass as a backbone at the origin for a protein batched with structured ones.
        padded = torch.full((batch_size, max_len, 3, 3), float("nan"), dtype=torch.float32)
        for i, arr in enumerate(structure_coord_list):
            L = arr.shape[0]
            if L > 0:
                padded[i, :L] = torch.from_numpy(arr).to(torch.float32)

    batch["structure_coords"] = padded.to(batch["input_ids"].device)

    # -------------------------------------------------------------------
    # Create labels for supervised fine-tuning
    # -------------------------------------------------------------------

    # Create labels tensor filled with -100 (ignored in loss calculation)
    labels = torch.full_like(batch["input_ids"], -100)

    # Get token IDs for special markers
    assistant_start_marker = "<|im_start|>assistant\n"
    im_end_marker = "<|im_end|>"

    assistant_start_token_ids = processor.tokenizer.encode(assistant_start_marker, add_special_tokens=False)
    im_end_token_ids = processor.tokenizer.encode(im_end_marker, add_special_tokens=False)
    think_end_token_ids = processor.tokenizer.encode("</think>", add_special_tokens=False)

    # Convert token arrays to tensors for faster comparison
    assistant_marker_tensor = torch.tensor(assistant_start_token_ids, device=batch["input_ids"].device)
    im_end_marker_tensor = torch.tensor(im_end_token_ids, device=batch["input_ids"].device)
    think_end_marker_tensor = torch.tensor(think_end_token_ids, device=batch["input_ids"].device)

    # Get dimensions for easier reference
    assistant_marker_len = len(assistant_start_token_ids)
    im_end_marker_len = len(im_end_token_ids)

    # For each sequence in the batch
    for i in range(batch["input_ids"].shape[0]):
        input_ids = batch["input_ids"][i]
        seq_len = input_ids.size(0)
        # An opening paragraph with no answer: supervise it, but no terminator.
        is_grounding = bool(examples[i].get("is_grounding", False))

        # Track assistant sections
        assistant_sections = []

        # Stored as the position AFTER the marker, where the assistant turn begins.
        start_positions = [p + assistant_marker_len
                           for p in _marker_positions(input_ids, assistant_marker_tensor)]
        end_positions = _marker_positions(input_ids, im_end_marker_tensor)

        think_ends = (_marker_positions(input_ids, think_end_marker_tensor)
                      if is_grounding else [])

        # Match start and end markers to create sections
        for start_pos in start_positions:
            # Find the next end marker after this start position
            valid_ends = [pos for pos in end_positions if pos > start_pos]
            if valid_ends:
                end_pos = min(valid_ends)  # Take the first end marker after start
                # INCLUDE the closing <|im_end|> in the supervised span. Excluding it
                # (the original `start_pos:end_pos`) meant the stop token never received
                # a single gradient, so the model never learned to stop and looped at
                # inference until max_new_tokens. Measured on a real batch: 0 labels
                # equal <|im_end|> before this change, 1 after. The span is still bounded
                # by seq_len below.
                if start_pos < end_pos:
                    _end = end_pos + im_end_marker_len if supervise_eos else end_pos
                    if is_grounding:
                        _end = min((p for p in think_ends if p > start_pos), default=end_pos)
                    assistant_sections.append((start_pos, _end))
            else:
                # If no end marker, assume the section runs to the end of the sequence
                _end = seq_len
                if is_grounding:
                    _end = min((p for p in think_ends if p > start_pos), default=seq_len)
                assistant_sections.append((start_pos, _end))

        # Set labels for all identified assistant sections
        for start_pos, end_pos in assistant_sections:
            if start_pos < end_pos and start_pos < seq_len:
                end_pos = min(end_pos, seq_len)  # Safety check
                labels[i, start_pos:end_pos] = input_ids[start_pos:end_pos]

    # Mask padding via the attention mask, NOT `input_ids == pad_token_id`.
    # protein_llm.py sets `pad_token = eos_token`, and for Qwen that is <|im_end|>
    # (id 248046) -- the very token the answer must end with. Masking by id therefore
    # erased the stop-token label everywhere, including the one just restored above.
    # Both bugs had to be fixed together; either alone still yields zero EOS gradient.
    if supervise_eos and "attention_mask" in batch and batch["attention_mask"] is not None:
        labels[batch["attention_mask"] == 0] = -100
    else:
        labels[batch["input_ids"] == processor.tokenizer.pad_token_id] = -100

    # Add labels to batch
    batch["labels"] = labels

    # Add answer to batch
    if return_answer_in_batch:
        batch["answer"] = [example["answer"].strip() for example in examples]

    # -------------------------------------------------------------------
    # Inference mode: truncate after assistant start marker
    # -------------------------------------------------------------------
    if inference_mode:
        device = batch["input_ids"].device
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = processor.tokenizer.eos_token_id

        # Composite marker to match _truncate_after_assistant_start
        composite = "<|im_end|>\n<|im_start|>assistant\n"
        comp_ids = processor.tokenizer.encode(composite, add_special_tokens=False)
        comp_t = torch.tensor(comp_ids, device=device)
        comp_len = len(comp_ids)

        B, L = batch["input_ids"].shape
        keep_lens: List[int] = []

        for i in range(B):
            ids = batch["input_ids"][i]
            keep = L  # default: keep all if marker not found
            # Scan for FIRST occurrence
            for j in range(0, L - comp_len + 1):
                if torch.all(ids[j : j + comp_len] == comp_t):
                    keep = j + comp_len
                    break
            keep_lens.append(keep)

        new_max = max(keep_lens) if keep_lens else 0

        # Allocate new left-padded tensors
        new_input_ids = torch.full((B, new_max), pad_id, dtype=batch["input_ids"].dtype, device=device)
        new_attention = torch.zeros((B, new_max), dtype=batch["attention_mask"].dtype, device=device)
        new_labels = torch.full((B, new_max), -100, dtype=batch["labels"].dtype, device=device)

        for i, k in enumerate(keep_lens):
            if k == 0:
                continue
            # Take the first k tokens (truncate from the RIGHT), then left-pad to new_max
            src_ids = batch["input_ids"][i, :k]
            src_attn = batch["attention_mask"][i, :k]
            src_lbls = batch["labels"][i, :k]

            new_input_ids[i, -k:] = src_ids
            new_attention[i, -k:] = src_attn
            new_labels[i, -k:] = src_lbls

        batch["input_ids"] = new_input_ids
        batch["attention_mask"] = new_attention
        batch["labels"] = new_labels

        # Also truncate the text prompts
        batch["prompt"] = [_truncate_after_assistant_start(p) for p in prompts_text]

    return batch
