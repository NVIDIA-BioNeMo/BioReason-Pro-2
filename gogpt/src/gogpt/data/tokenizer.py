# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path
from typing import Iterable, List

SPECIAL_TOKENS = ["<pad>", "<|MF_START|>", "<|MF_END|>", "<|BP_START|>", "<|BP_END|>",
                  "<|CC_START|>", "<|CC_END|>"]
ASPECTS = ("MF", "BP", "CC")


class GOTermTokenizer:
    """GO term vocabulary: 7 special tokens followed by the GO ids in sorted order.

    A sequence for one aspect is [<|X_START|>, GO ids..., <|X_END|>].
    """

    def __init__(self, go_terms: Iterable[str]):
        self.special_tokens = list(SPECIAL_TOKENS)
        all_tokens = self.special_tokens + sorted(set(go_terms))
        self.token_to_id = {token: idx for idx, token in enumerate(all_tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        self.pad_token_id = self.token_to_id["<pad>"]
        self.mf_start_token_id = self.token_to_id["<|MF_START|>"]
        self.mf_end_token_id = self.token_to_id["<|MF_END|>"]
        self.bp_start_token_id = self.token_to_id["<|BP_START|>"]
        self.bp_end_token_id = self.token_to_id["<|BP_END|>"]
        self.cc_start_token_id = self.token_to_id["<|CC_START|>"]
        self.cc_end_token_id = self.token_to_id["<|CC_END|>"]
        self.aspect_to_tokens = {
            "MF": (self.mf_start_token_id, self.mf_end_token_id),
            "BP": (self.bp_start_token_id, self.bp_end_token_id),
            "CC": (self.cc_start_token_id, self.cc_end_token_id),
        }

    def __len__(self):
        return len(self.token_to_id)

    @property
    def special_token_ids(self):
        return {self.token_to_id[t] for t in self.special_tokens}

    def encode(self, go_terms: List[str], aspect: str) -> List[int]:
        """[START, terms..., END] for one aspect. Every term must be in the vocabulary."""
        if aspect not in self.aspect_to_tokens:
            raise ValueError(f"aspect must be one of {ASPECTS}, got {aspect!r}")
        start, end = self.aspect_to_tokens[aspect]
        missing = [t for t in go_terms if t not in self.token_to_id]
        if missing:
            raise KeyError(f"GO terms not in the vocabulary: {missing[:5]}")
        return [start] + [self.token_to_id[t] for t in go_terms] + [end]

    def decode(self, ids: Iterable[int], aspect: str) -> List[str]:
        """GO terms from a generated sequence for one aspect.

        Reads from after the aspect's START token (if present) up to its END token,
        skips special tokens and drops repeats, keeping generation order.
        """
        start, end = self.aspect_to_tokens[aspect]
        ids = list(ids)
        if start in ids:
            ids = ids[ids.index(start) + 1:]
        if end in ids:
            ids = ids[: ids.index(end)]
        special = self.special_token_ids
        terms, seen = [], set()
        for i in ids:
            if i in special:
                continue
            term = self.id_to_token.get(i)
            if term is not None and term not in seen:
                seen.add(term)
                terms.append(term)
        return terms

    # ------------------------------------------------------------------ serialization

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump({"tokens": [self.id_to_token[i] for i in range(len(self))]}, f)

    @classmethod
    def from_json(cls, path):
        with open(Path(path)) as f:
            tokens = json.load(f)["tokens"]
        if tokens[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError(f"{path}: the first tokens must be {SPECIAL_TOKENS}")
        tok = cls(tokens[len(SPECIAL_TOKENS):])
        if [tok.id_to_token[i] for i in range(len(tok))] != tokens:
            raise ValueError(f"{path}: GO ids are not in sorted order; ids would be remapped")
        return tok
