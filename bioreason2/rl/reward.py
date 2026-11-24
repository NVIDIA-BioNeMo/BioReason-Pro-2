# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""GO parsing and ground truth for the RL reward, exactly as the published scoring derives them.

The reward is the LLM judge's (`judge_reward.py`). Two deterministic pieces around it come from
`evals/cafa/cafa_evals.py`, so RL reads an answer and a protein's labels the way the benchmark does:
the GO ids an answer commits to (the `both` track, which gates the judge call) and the ground truth
(tier 1 union tier 2, gated on `scored_aspects`), which decides whether a protein can be trained on.
"""

import importlib.util
import os
import types

from bioreason2 import paths

#: The commitment track the published numbers score (`cafa_evals.py --go_track both`).
TRACK = "both"

CAFA_EVALS_PATH = os.path.join(paths.REPO_ROOT, "evals", "cafa", "cafa_evals.py")


def load_cafa_evals() -> types.ModuleType:
    """Import `evals/cafa/cafa_evals.py` by path, so the reward never depends on the working directory.

    Returns:
        The module.
    """
    spec = importlib.util.spec_from_file_location("bioreason2_cafa_evals", CAFA_EVALS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAFA_EVALS = load_cafa_evals()


def ground_truth(labels: dict) -> tuple[set[str], set[str]]:
    """Ground-truth terms and scorable aspects, exactly as `cafa_evals.py` derives them.

    Args:
        labels: A row's `extra_info`: `go_{mf,bp,cc}`, `go_*_tier2`, `scored_aspects`.

    Returns:
        The tier-1 union tier-2 terms of the scorable aspects, and those aspects.
    """
    return CAFA_EVALS.extract_reasoning_ground_truth(labels)


def predicted_terms(text: str) -> set[str]:
    """The GO ids an answer commits to, exactly as `cafa_evals.py` parses them.

    Args:
        text: The generated response.

    Returns:
        GO ids on the scored track; empty when the answer names none.
    """
    return CAFA_EVALS.parse_prediction_format(text, final_answer_only=False, track=TRACK)


__all__ = ["ground_truth", "predicted_terms"]
