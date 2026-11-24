# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Ids and names shared by checkpoint prep, the SkyRL handlers, the vLLM model and the dataset.

Kept free of heavy imports so the vLLM plugin, the trainer and CPU-only tests can all read it.
"""

from bioreason2.models.special_tokens import GO_GRAPH_PAD_TOKEN, PROTEIN_PAD_TOKEN

#: SkyRL modality key and vLLM ``multi_modal_data`` key.
MODALITY = "protein"

#: Ids the SFT tokenizer assigns. Asserted, never derived, so a drifted tokenizer fails loudly.
PROTEIN_PAD_ID = 248077
GO_GRAPH_PAD_ID = 248078

#: The unified GO encoder emits exactly this many reduced embeddings.
NUM_GO_TOKENS = 200

#: One distinct token per GO row, written into unused padding rows of the 248320-row table.
GO_TOKENS = [f"<|go_graph_{i}|>" for i in range(NUM_GO_TOKENS)]
FIRST_GO_ID = 248079

#: Embedding table height of Qwen3.5 4B/9B; the GO rows must fit below it without a resize.
VOCAB_SIZE = 248320

#: ESM3 ``esm3_sm_open_v1`` layer-37 width.
ESM3_DIM = 1536

#: SFT/eval truncate at 2000 residues and ESM3 adds BOS/EOS.
MAX_LENGTH_PROTEIN = 2000
MAX_PROTEIN_TOKENS = MAX_LENGTH_PROTEIN + 2

#: Architecture the prepared checkpoint declares; must match ``vllm_plugin.ARCHITECTURE``.
ARCHITECTURE = "Qwen3_5ProteinForCausalLM"

#: Projector sidecar written by the SFT export.
PROJECTION_FILENAME = "protein_projection.pt"

__all__ = [
    "ARCHITECTURE",
    "ESM3_DIM",
    "FIRST_GO_ID",
    "GO_GRAPH_PAD_ID",
    "GO_GRAPH_PAD_TOKEN",
    "GO_TOKENS",
    "MAX_LENGTH_PROTEIN",
    "MAX_PROTEIN_TOKENS",
    "MODALITY",
    "NUM_GO_TOKENS",
    "PROJECTION_FILENAME",
    "PROTEIN_PAD_ID",
    "PROTEIN_PAD_TOKEN",
    "VOCAB_SIZE",
]
