# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

ABSENT = "None annotated"

SCOPE_ALL = "every annotated feature"
SCOPE_SUBSET = "a subset, others withheld"
SCOPE_NONE = "none supplied"

PROMPT = """\
You are a scientific assistant specialized in protein function prediction. The protein \
sequence is given above, from organism {organism}. Reason step by step to its molecular \
function, the biological processes it participates in, and where in the cell it acts.

Residue features, {features_scope}:
{features}

Initial GO term speculations:
{speculations}

Known interaction partners:
{partners}

Reason about the function of the protein and focus on its molecular function, biological \
process and cellular component.\
"""

FIELDS = ("organism", "features", "features_scope", "speculations", "partners")
