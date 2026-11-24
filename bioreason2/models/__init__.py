# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy re-exports.

`protein_llm` builds the full training model and imports its dependencies at module
level. Importing it eagerly here would load them for anything under this package, such
as `protein_encoder` on the inference path.

PEP 562 module __getattr__ keeps the public names available while deferring the import
until someone actually asks for them.
"""

__all__ = ["ProteinLLMModel"]


def __getattr__(name):
    if name in __all__:
        from .protein_llm import ProteinLLMModel

        return ProteinLLMModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
