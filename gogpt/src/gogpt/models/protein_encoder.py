# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Frozen ESM2 protein encoder.

One definition of how a sequence becomes per-residue embeddings, shared by training
(on-the-fly embeddings), scripts/precompute_embeddings.py (cached embeddings) and the
predictor, so the three cannot drift apart:

* tokenization without special tokens (no <cls>/<eos>): one token per residue,
* truncation to `max_length` residues,
* the output is `hidden_states[layer]` (0 = embedding output, i = output of layer i).
"""

import re

import torch
import torch.nn as nn

# Residue letters the ESM2 vocabulary has a token for. Anything else would become <unk>.
ESM2_RESIDUES = set("LAGVSERTIDPKQNFYMHWCXBUZO")


def clean_sequence(sequence):
    """Upper-case a sequence and drop whitespace; reject characters ESM2 cannot encode."""
    seq = re.sub(r"\s+", "", str(sequence)).upper()
    if not seq:
        raise ValueError("empty protein sequence")
    bad = sorted(set(seq) - ESM2_RESIDUES)
    if bad:
        raise ValueError(f"protein sequence contains characters ESM2 cannot encode: {bad}")
    return seq


class ESM2Encoder(nn.Module):
    def __init__(self, model_name, layer, max_length):
        super().__init__()
        from transformers import AutoTokenizer, EsmModel

        self.model_name = model_name
        self.layer = layer
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.esm = EsmModel.from_pretrained(model_name, add_pooling_layer=False)
        n_layers = self.esm.config.num_hidden_layers
        if not 0 <= layer <= n_layers:
            raise ValueError(f"layer {layer} out of range: {model_name} has hidden_states 0..{n_layers}")
        self.hidden_size = self.esm.config.hidden_size
        self.esm.requires_grad_(False)
        self.esm.eval()

    def train(self, mode=True):
        # Frozen: stays in eval mode (no dropout) whatever the parent module does.
        super().train(False)
        return self

    def tokenize(self, sequence):
        """Token ids for one sequence: one per residue, truncated to max_length."""
        return self.tokenizer(
            sequence, add_special_tokens=False, truncation=True,
            max_length=self.max_length, padding=False,
        )["input_ids"]

    def batch_tokens(self, sequences, device=None):
        """Right-padded [B, T] input ids and a bool attention mask."""
        ids = [self.tokenize(s) for s in sequences]
        T = max(len(i) for i in ids)
        input_ids = torch.full((len(ids), T), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(ids), T), dtype=torch.bool)
        for row, i in enumerate(ids):
            input_ids[row, : len(i)] = torch.tensor(i, dtype=torch.long)
            mask[row, : len(i)] = True
        if device is not None:
            input_ids, mask = input_ids.to(device), mask.to(device)
        return input_ids, mask

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        """[B, T, hidden_size] embeddings from hidden_states[layer]; padded rows are zero."""
        out = self.esm(input_ids=input_ids, attention_mask=attention_mask.long(),
                       output_hidden_states=True)
        hidden = out.hidden_states[self.layer]
        return hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
