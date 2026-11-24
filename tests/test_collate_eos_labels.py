# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The stop token must be supervised.

Two independent defects in `qwen_protein_collate_fn` each zeroed the gradient on
`<|im_end|>`, so the model never learned to stop and looped at inference:

  A. the supervised span was `labels[start:end]` where `end` is the POSITION of
     `<|im_end|>` -- a Python slice excludes it.
  B. `labels[input_ids == pad_token_id] = -100`, while protein_llm.py sets
     `pad_token = eos_token`, which for Qwen IS `<|im_end|>`.

Either alone is sufficient to break it, so both are asserted here. The collator's structure
padding is checked here too: a protein without a structure must reach ESM3 sequence-only.

Run:  python -m pytest tests/test_collate_eos_labels.py -q     (needs the Qwen tokenizer)
"""
import os
import sys
import tempfile

import numpy as np
import torch

try:                                    # the training env has no pytest
    import pytest
    _fixture = pytest.fixture(scope="module")
except ImportError:                     # standalone mode, see __main__ below
    pytest = None
    def _fixture(fn):
        return fn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

MODEL = os.environ.get("TEST_TOKENIZER", "Qwen/Qwen3.5-4B")


@_fixture
def processor():
    if pytest is not None:
        pytest.importorskip("trl")   # the collator's import chain needs the training environment
    from transformers import AutoTokenizer
    from bioreason2.models.pl.chat_template_pl import get_chat_template
    from bioreason2.models.pl.processing_pl import PLProcessor
    from bioreason2.models.special_tokens import get_all_special_tokens

    # The three things ProteinLLM's constructor does to the tokenizer.
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=os.environ.get("HF_HOME"))
    tok.chat_template = get_chat_template(MODEL)
    tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": get_all_special_tokens()})
    return PLProcessor(tokenizer=tok)


def _examples(n=1, answer="It is a kinase."):
    """EXACTLY the structure `format_cafa5_for_protein_llm` emits.

    This matters: an earlier version of this test used a plain
    [{"role": "user"...}, {"role": "assistant", "content": "..."}] list and proved
    nothing, because the real pipeline passes typed content parts plus
    `reasoning_content`. Keep this in sync with dataset/cafa5/format.py.
    """
    out = []
    for i in range(n):
        out.append({
            "prompt": [
                {"role": "user", "content": [
                    {"type": "protein", "text": None},
                    {"type": "go_graph", "text": None},
                    {"type": "text", "text": "SYSTEM TEXT\n\nWhat does this protein do?"},
                ]},
                {"role": "assistant",
                 "reasoning_content": "REASONING %d" % i,
                 "content": [{"type": "text", "text": "%s (%d)" % (answer, i)}]},
            ],
            "protein_sequences": [], "structure_path": None, "go_aspect": "all",
            "answer": answer,
        })
    return out


def _collate(processor, examples):
    from bioreason2.dataset.cafa5.collate import qwen_protein_collate_fn
    return qwen_protein_collate_fn(
        examples=examples, processor=processor,
        max_length_text=4096, max_length_protein=2000,
    )


def test_pad_token_really_is_eos(processor):
    """The precondition that makes defect B bite. If this ever stops holding, the
    masking-by-id would have been harmless -- so the test documents WHY."""
    tok = processor.tokenizer
    assert tok.pad_token_id == tok.eos_token_id
    assert tok.pad_token_id == tok.convert_tokens_to_ids("<|im_end|>")


def test_eos_is_supervised(processor):
    """Defect A: the closing <|im_end|> must carry a label, not -100."""
    tok = processor.tokenizer
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    batch = _collate(processor, _examples())
    labels = batch["labels"]
    assert (labels != -100).any(), "nothing was supervised at all"
    n_eos = int((labels == im_end).sum())
    assert n_eos >= 1, (
        "the stop token is never supervised -- the model cannot learn to stop. "
        "Supervised text was: %r" % tok.decode(labels[labels != -100])
    )


def test_padding_is_still_masked(processor):
    """The terminator fix must not leak padding into the loss. Two rows of very
    different length force real left-padding."""
    ex = _examples(1) + _examples(1, answer="A considerably longer answer " * 20)
    batch = _collate(processor, ex)
    labels, attn = batch["labels"], batch["attention_mask"]
    assert int((attn == 0).sum()) > 0, "no padding produced; test is not exercising it"
    assert bool((labels[attn == 0] == -100).all()), "padding leaked into the labels"


def test_terminator_present_in_every_row(processor):
    """Batched: every row must carry a supervised terminator, not just row 0."""
    tok = processor.tokenizer
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    batch = _collate(processor, _examples(n=3))
    labels = batch["labels"]
    for i in range(labels.shape[0]):
        assert int((labels[i] == im_end).sum()) >= 1, "row %d has no supervised stop token" % i


def test_inference_mode_has_no_terminator(processor):
    """The guard against over-correcting: at inference the text is a prompt to be
    CONTINUED, so appending a stop token would make the model emit one immediately."""
    from bioreason2.dataset.cafa5.collate import qwen_protein_collate_fn
    tok = processor.tokenizer
    batch = qwen_protein_collate_fn(
        examples=_examples(), processor=processor, max_length_text=4096,
        max_length_protein=2000, inference_mode=True,
    )
    text = tok.decode(batch["input_ids"][0], skip_special_tokens=False)
    assert not text.rstrip().endswith("<|im_end|>"), (
        "inference prompt was terminated; generation would stop instantly")


def test_prompt_is_not_supervised(processor):
    """The guard against over-correcting: user text must stay masked."""
    tok = processor.tokenizer
    batch = _collate(processor, _examples())
    text = tok.decode(batch["labels"][batch["labels"] != -100])
    assert "What does this protein do?" not in text, "the user turn leaked into the loss"


def test_structure_padding_is_nan(processor):
    """A protein without a structure, batched with one that has it, is padded with NaN, which the
    encoder reads as no structure; zeros would pass as a backbone at the origin."""
    from bioreason2.dataset.cafa5 import collate

    examples = _examples(2)
    examples[0]["protein_sequences"] = ["MKTAYIAKQR"]
    examples[1]["protein_sequences"] = ["MSSQQY"]
    parser = collate.backbone_resolving_alternates
    collate.backbone_resolving_alternates = lambda path, seq: np.ones((len(seq), 3, 3))
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb") as handle:
            examples[0]["structure_path"] = handle.name
            coords = _collate(processor, examples)["structure_coords"]
    finally:
        collate.backbone_resolving_alternates = parser
    assert coords.shape == (2, 10, 3, 3)
    assert torch.equal(coords[0].cpu(), torch.ones(10, 3, 3))
    assert torch.isnan(coords[1]).all()


# The training env has no pytest, so the file is runnable directly:
#   python tests/test_collate_eos_labels.py
if __name__ == "__main__":
    _proc = None
    if _proc is None:
        from transformers import AutoTokenizer
        from bioreason2.models.pl.processing_pl import PLProcessor
        _tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=os.environ.get("HF_HOME"))
        _tok.pad_token = _tok.eos_token
        _proc = PLProcessor(tokenizer=_tok)
    fails = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn(_proc)
                print("PASS  %s" % _name)
            except AssertionError as e:
                fails += 1
                print("FAIL  %s\n        %s" % (_name, e))
            except Exception as e:
                fails += 1
                print("ERROR %s\n        %s: %s" % (_name, type(e).__name__, e))
    print("\n%s" % ("ALL PASS" if not fails else "%d FAILED" % fails))
    sys.exit(1 if fails else 0)
