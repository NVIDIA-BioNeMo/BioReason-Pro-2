# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""A grounding row is supervised on its paragraph and on no terminator.

`proteins_extra/ground` ships an opening paragraph with `final_answer` NULL. Trained
like an ordinary row it teaches the model to end the turn (`<|im_end|>`) or to stop
thinking (`</think>`) after one paragraph. Both are asserted here; the second is the
one that survives the obvious fix.

Runs in the training environment (README, Setup):

    pytest tests/test_grounding_target.py      or      python tests/test_grounding_target.py
"""
import os
import sys

REPO =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

MODEL = os.environ.get("TEST_TOKENIZER", "Qwen/Qwen3.5-4B")

PARAGRAPH = ("The chain opens with a signal peptide, 1 to 22, carrying its three parts "
             "in the textbook order: a short basic n-region over 1 to 6, a hydrophobic "
             "core over 7 to 17, and a polar c-region over 18 to 22.")
ANSWER = "- Summary: a secreted precursor.\n- MF: GO:0005515 (protein binding)"
QUESTION = "Reason step by step about this protein.\n\nResidue features, none supplied:\nNone annotated"

_results = []


def check(cond, msg):
    _results.append(bool(cond))
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    return cond


def _row(is_grounding):
    return {
        "protein_id": "GROUND1" if is_grounding else "NORMAL1",
        "question": QUESTION,
        "assistant_reasoning": PARAGRAPH,
        "assistant_answer": "" if is_grounding else ANSWER,
        "sequence": "MKTLLILAVVAAALAAPQE",
        "structure_path": None,
        "is_grounding": is_grounding,
    }


def main():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("SKIP: transformers not importable")
        return 0

    from bioreason2.dataset.cafa5.load import _format_reasoning_prompt
    from bioreason2.dataset.cafa5.format import format_cafa5_for_protein_llm
    from bioreason2.dataset.cafa5.collate import qwen_protein_collate_fn
    from bioreason2.models.pl.processing_pl import PLProcessor
    from bioreason2.models.pl.chat_template_pl import get_chat_template
    from bioreason2.models.special_tokens import get_all_special_tokens

    ground = format_cafa5_for_protein_llm(_format_reasoning_prompt(_row(True)))
    normal = format_cafa5_for_protein_llm(_format_reasoning_prompt(_row(False)))
    check(ground.get("is_grounding") is True and normal.get("is_grounding") is False,
          "is_grounding survives the loader and the formatter")

    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=os.environ.get("HF_HOME"))
    tok.chat_template = get_chat_template(MODEL)
    tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": get_all_special_tokens()})
    proc = PLProcessor(tokenizer=tok)

    batch = qwen_protein_collate_fn(examples=[ground, normal], processor=proc,
                                    max_length_text=4096, max_length_protein=2000)
    ids, labels = batch["input_ids"], batch["labels"]

    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    think_end = tok.encode("</think>", add_special_tokens=False)[0]

    for r, name in ((0, "grounding"), (1, "normal")):
        sup = labels[r] != -100
        text = tok.decode(ids[r][sup])
        n_im_end = int((labels[r] == im_end).sum())
        n_think = int((labels[r] == think_end).sum())
        print(f"  [{name}] supervised {int(sup.sum())} tokens, "
              f"im_end {n_im_end}, </think> {n_think}")
        print(f"  [{name}] tail: ...{text[-90:]!r}")

        check(int(sup.sum()) > 0, f"{name}: something is supervised")
        check("Reason step by step" not in text, f"{name}: the question is not supervised")

        if name == "grounding":
            check(n_im_end == 0, "grounding: <|im_end|> carries no label")
            check(n_think == 0, "grounding: </think> carries no label")
            check(PARAGRAPH[-40:] in text, "grounding: the paragraph is supervised to its end")
            check(ANSWER[:20] not in text, "grounding: no answer text leaks in")
        else:
            check(n_im_end >= 1, "normal: <|im_end|> still carries a label")
            check(ANSWER[:30] in text, "normal: the answer is supervised")

    print(f"\n{sum(_results)}/{len(_results)} passed")
    return 0 if all(_results) else 1


def test_all_checks() -> None:
    """Every check, under pytest; skipped outside the training environment."""
    import pytest  # the training environment may not have pytest, so not at module level

    pytest.importorskip("trl")
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
