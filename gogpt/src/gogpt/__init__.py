# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""GO-GPT: autoregressive Gene Ontology term prediction from protein sequence.

Importing the package is cheap: the predictor, model and training modules are loaded on
first use, so inference never needs the training dependencies (Lightning, Hydra).
"""

__version__ = "2.0.0"

_LAZY = {
    "GOGPTPredictor": ("gogpt.inference", "GOGPTPredictor"),
    "GOGPT": ("gogpt.models.gogpt", "GOGPT"),
    "GOGPTConfig": ("gogpt.config.model_config", "GOGPTConfig"),
    "ESM2Encoder": ("gogpt.models.protein_encoder", "ESM2Encoder"),
    "GOTermTokenizer": ("gogpt.data.tokenizer", "GOTermTokenizer"),
    "OrganismMapper": ("gogpt.utils.organism_mapper", "OrganismMapper"),
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'gogpt' has no attribute {name!r}")
