# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from dataclasses import asdict, dataclass, fields


@dataclass
class GOGPTConfig:
    # Decoder
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int  # maximum protein + GO sequence length
    vocab_size: int
    organism_vocab_size: int
    dropout: float = 0.0
    bias: bool = True
    use_gated_attention: bool = True  # sigmoid-gated attention output
    use_rmsnorm: bool = True  # RMSNorm instead of LayerNorm
    use_qk_norm: bool = True  # per-head RMSNorm on queries and keys

    # GO vocabulary special tokens
    pad_token_id: int = 0
    mf_start_token_id: int = 1
    mf_end_token_id: int = 2
    bp_start_token_id: int = 3
    bp_end_token_id: int = 4
    cc_start_token_id: int = 5
    cc_end_token_id: int = 6

    # Protein encoder (frozen ESM2). protein_layer_index indexes the encoder's
    # hidden_states tuple: 0 is the embedding output, i is the output of layer i.
    embed_model_path: str = "facebook/esm2_t33_650M_UR50D"
    protein_embedding_dim: int = 1280
    protein_layer_index: int = 27
    max_protein_length: int = 2000

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Build from a dict, rejecting unknown keys so a config typo cannot pass silently."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown GOGPTConfig keys: {unknown}")
        return cls(**d)
