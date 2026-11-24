# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from typing import Any, Dict


def format_cafa5_for_protein_llm(example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a row's prompt dict into the chat-message list the Protein-LLM reads.

    Args:
        example: a row carrying `prompt` (system, user, assistant_reasoning,
            assistant_answer) and `sequence`.

    Returns:
        The fields the collator and eval read: `prompt` as messages, the sequence,
        structure path, GO aspect, answer and grounding flag.
    """
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "protein", "text": None},
                    {"type": "go_graph", "text": None},
                    {
                        "type": "text",
                        # Join only what is there. A pre-rendered question carries
                        # no system turn (the teacher has none either), and an
                        # unconditional join would prepend a blank line to every
                        # prompt in the corpus.
                        "text": "\n\n".join(
                            p for p in ((example["prompt"]["system"] or "").strip(),
                                        (example["prompt"]["user"] or "").strip()) if p
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                # A test row has no trace, so both are NULL rather than absent.
                "reasoning_content": (example["prompt"]["assistant_reasoning"] or "").strip(),
                "content": [
                    {
                        "type": "text",
                        "text": (example["prompt"]["assistant_answer"] or "").strip(),
                    },
                ],
            },
        ],
        "protein_sequences": [
            example["sequence"],
        ],
        "structure_path": example.get("structure_path"),
        "go_aspect": example.get("go_aspect"),
        "answer": example["prompt"]["assistant_answer"].strip(),
        "ground_truth_go_terms": example.get("ground_truth_go_terms", ""),
        "is_grounding": bool(example.get("is_grounding", False)),
    }
