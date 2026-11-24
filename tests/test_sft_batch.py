# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""What actually reaches the model, from the built SFT dataset to `labels`.

`test_sft_question_pipeline.py` stops at the chat structure. This goes the rest of
the way -- real loader, real Qwen tokenizer, real collator -- and asserts on the
tensors a training step would see, because that is the only place several of these
failures are visible:

  * the loss mask. Only the assistant turn may be supervised. If the question is
    supervised the model is trained to reproduce its own prompt; if the answer is
    NOT supervised the run learns nothing and the loss still looks plausible.
  * the stop token. `<|im_end|>` must carry a label -- two independent defects in
    the collator each zeroed that gradient in earlier runs, and the symptom was
    generation running to `max_new_tokens` rather than an error.
  * the placeholders. `<|protein_pad|>` and `<|go_graph_pad|>` are substituted by
    embeddings before the forward pass and the model hard-fails if the token count
    and the embedding count disagree.
  * truncation. `max_length_text` silently clips, and a prompt clipped past the
    assistant marker removes the target entirely.

Needs a tokenizer, so it runs in the training environment (README, Setup):

    BRP_SFT_DIR=<dir> python tests/test_sft_batch.py

Skips cleanly without a built dataset or without transformers.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

SFT_DIR = os.environ.get("BRP_SFT_DIR",
                         os.path.join(REPO, "data", "sft", "v2-features"))
MODEL = os.environ.get("TEST_TOKENIZER", "Qwen/Qwen3.5-4B")
MAX_TEXT = int(os.environ.get("MAX_LENGTH_TEXT", "16384"))

_results = []


def check(cond, msg):
    _results.append(bool(cond))
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    return cond


def main():
    if not os.path.isdir(os.path.join(SFT_DIR, "data")):
        print(f"SKIP: no built dataset at {SFT_DIR}")
        return 0
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("SKIP: transformers not importable")
        return 0

    from bioreason2.dataset.cafa5.load import load_cafa5_dataset
    from bioreason2.dataset.cafa5.collate import qwen_protein_collate_fn
    from bioreason2.models.pl.processing_pl import PLProcessor

    # --- the loader, with the training invocation's own arguments ----------
    train, val, test = load_cafa5_dataset(
        dataset=SFT_DIR,
        config="default",
        cache_dir=os.environ.get("DATASET_CACHE_DIR", "/tmp/e2e-cache"),
        structure_dir=None,
        num_proc=1,
        return_as_chat_template=True,   # what train_protein_llm.py passes
    )
    check(train is not None and val is not None and len(train) and len(val),
          f"load_cafa5_dataset -> train {len(train)}, val {len(val)}, "
          f"test {len(test) if test is not None else 'absent'}")

    ex = train[0]
    turns = ex["prompt"]
    user = [c for c in turns[0]["content"] if c["type"] == "text"][0]["text"]
    check(turns[0]["role"] == "user" and turns[1]["role"] == "assistant"
          and len(turns) == 2, "two turns: one user, one assistant")
    check("Initial GO term speculations:" in user
          and "Known interaction partners:" in user,
          "the user turn is our question, section headings intact")
    check("- Summary:" in turns[1]["content"][0]["text"],
          "the assistant turn is the teacher's Final Answer")
    check(turns[1].get("reasoning_content"), "the reasoning trace is carried separately")

    check("InterPro annotations" not in user,
          "no section injected -- the stored question is used verbatim")

    # --- the collator ------------------------------------------------------
    # Exactly the three things ProteinLLM's constructor does to the tokenizer
    # (`ProteinLLM.__init__`). All three matter here: without the custom chat
    # template the `{"type": "protein"}` content part renders as the literal
    # string "None" instead of `<|protein_pad|>`, and without the added tokens the
    # placeholder does not tokenize to a single id.
    from bioreason2.models.pl.chat_template_pl import get_chat_template
    from bioreason2.models.special_tokens import get_all_special_tokens, get_token

    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=os.environ.get("HF_HOME"))
    tok.chat_template = get_chat_template(MODEL)
    tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": get_all_special_tokens()})
    proc = PLProcessor(tokenizer=tok)

    batch = qwen_protein_collate_fn(
        examples=[dict(train[i]) for i in range(4)], processor=proc,
        max_length_text=MAX_TEXT, max_length_protein=2000)
    ids, labels = batch["input_ids"], batch["labels"]
    check(ids.shape == labels.shape, f"batch {tuple(ids.shape)}, labels aligned")

    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    row_ids, row_lab = ids[0], labels[0]
    sup = (row_lab != -100)

    # --- the loss mask -----------------------------------------------------
    check(sup.sum().item() > 0, f"{sup.sum().item()} supervised tokens in row 0")
    frac = sup.sum().item() / int((row_ids != tok.pad_token_id).sum())
    check(0.05 < frac < 0.95, f"supervised fraction {frac:.2f} -- neither all nor none")

    supervised_text = tok.decode(row_ids[sup])
    answer = train[0]["prompt"][1]["content"][0]["text"]
    check(answer[:60] in supervised_text,
          "the answer's first 60 chars are inside the supervised span")
    check(train[0]["prompt"][1]["reasoning_content"][:60] in supervised_text,
          "the reasoning trace is supervised too")

    prompt_text = tok.decode(row_ids[~sup & (row_ids != tok.pad_token_id)])
    q_probe = user[:80]
    check(q_probe in prompt_text and q_probe not in supervised_text,
          "the question is in the UNsupervised span and nowhere in the supervised one")

    # --- the stop token --------------------------------------------------
    n_sup_imend = int(((row_ids == im_end) & sup).sum())
    check(n_sup_imend >= 1,
          f"<|im_end|> carries a label ({n_sup_imend} supervised occurrence(s))")
    check(supervised_text.rstrip().endswith("<|im_end|>")
          or row_ids[sup][-1].item() == im_end,
          "the supervised span ENDS on the stop token")

    # --- padding -----------------------------------------------------------
    pad_positions = (row_ids == tok.pad_token_id) & (batch["attention_mask"][0] == 0)
    check(int((row_lab[pad_positions] != -100).sum()) == 0,
          "every padded position is masked out")

    # --- the placeholders --------------------------------------------------
    # One protein token per residue (capped at max_length_protein, +2 for ESM's
    # own specials) and exactly 200 GO tokens -- the model substitutes embeddings
    # at these positions and hard-fails if the counts disagree.
    for key, expect in (("protein_pad", None), ("go_graph_pad", 200)):
        tid = tok.convert_tokens_to_ids(get_token(key))
        n0 = int((ids[0] == tid).sum())
        if expect is None:
            seq = len(train[0]["protein_sequences"][0])
            expect = min(seq, 2000) + 2
        check(n0 == expect,
              f"<|{key}|> row 0: {n0} tokens (expected {expect})")

    # --- truncation --------------------------------------------------------
    lens = [int((ids[i] != tok.pad_token_id).sum()) for i in range(ids.shape[0])]
    check(max(lens) < MAX_TEXT,
          f"no row reaches max_length_text={MAX_TEXT} (longest {max(lens)})")

    print("\n--- what the model sees, row 0 " + "-" * 46)
    print(f"tokens {int((row_ids != tok.pad_token_id).sum())}  "
          f"supervised {int(sup.sum())}  ({frac:.1%})")
    print("\nUNSUPERVISED (prompt) head:\n" + prompt_text[:420])
    print("\nSUPERVISED (target) head:\n" + supervised_text[:420])
    print("\nSUPERVISED tail:\n" + supervised_text[-260:])
    print("-" * 78)

    print(f"\n{sum(_results)}/{len(_results)} passed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
