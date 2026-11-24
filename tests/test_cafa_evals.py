# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""What the Fmax scorer takes from a generation and from a row's labels."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.cafa import cafa_evals  # noqa: E402

ANSWER = """<think>GO:0000001 was considered and rejected.</think>
- Summary: a kinase.
- Confident GO Terms:
  - Molecular Function: GO:0004672 (protein kinase activity)
- Speculated GO Terms:
  - Biological Process: GO:0006468 (protein phosphorylation)
"""


def test_tracks_split_the_answer() -> None:
    """`confident` is the Confident section; `both` adds Speculated; `all` adds the reasoning."""
    assert cafa_evals.extract_go_terms(ANSWER, "confident") == {"GO:0004672"}
    assert cafa_evals.extract_go_terms(ANSWER, "both") == {"GO:0004672", "GO:0006468"}
    assert cafa_evals.extract_go_terms(ANSWER, "all") == {"GO:0000001", "GO:0004672", "GO:0006468"}


def test_missing_headers_read_the_answer() -> None:
    """An answer without the track headers is scored on the ids of its answer, never of its reasoning."""
    assert cafa_evals.extract_go_terms("binds GO:0005524", "both") == {"GO:0005524"}
    assert cafa_evals.extract_go_terms(
        "<think>GO:0000001 is ruled out.</think> It binds ATP, GO:0005524.", "both") == {"GO:0005524"}
    assert cafa_evals.extract_go_terms("<think>GO:0000001 might fit, but", "both") == set()


def test_scored_aspects_gate_the_ground_truth() -> None:
    """Only the aspects in `scored_aspects` are scored, even when another aspect has terms."""
    sample = {"go_mf": ["GO:0003824"], "go_bp": ["GO:0008150"], "go_cc": [], "scored_aspects": ["MF"]}
    _, aspects = cafa_evals.extract_reasoning_ground_truth(sample)
    assert aspects == {"MF"}
    _, aspects = cafa_evals.extract_reasoning_ground_truth(
        {k: v for k, v in sample.items() if k != "scored_aspects"})
    assert aspects == {"MF", "BP"}


def test_tier_two_joins_the_ground_truth() -> None:
    """Tier-2 terms are part of the label set, as they were part of the teacher's prompt."""
    sample = {"go_mf": ["GO:0003824"], "go_mf_tier2": ["GO:0016301"], "scored_aspects": ["MF"]}
    terms, _ = cafa_evals.extract_reasoning_ground_truth(sample)
    assert {"GO:0003824", "GO:0016301"} <= set(terms)
