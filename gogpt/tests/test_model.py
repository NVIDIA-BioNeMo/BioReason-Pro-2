# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Decoder behaviour on a small random model (CPU, fp32, no downloads)."""

import pytest
import torch
import torch.nn.functional as F

from gogpt.config.model_config import GOGPTConfig
from gogpt.models.gogpt import GOGPT

D_PROT = 24


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = GOGPTConfig(n_layer=2, n_head=4, n_embd=32, block_size=128, vocab_size=40,
                      organism_vocab_size=5, dropout=0.0, protein_embedding_dim=D_PROT)
    m = GOGPT(cfg).eval()
    # Random init makes END tokens unlikely; boost them so generation actually stops.
    with torch.no_grad():
        m.lm_head.weight[[2, 4, 6]] *= 3
    return m


def proteins(lengths, seed=1):
    g = torch.Generator().manual_seed(seed)
    T = max(lengths)
    emb = torch.randn(len(lengths), T, D_PROT, generator=g)
    mask = torch.zeros(len(lengths), T, dtype=torch.bool)
    for i, n in enumerate(lengths):
        mask[i, :n] = True
    return emb, mask


@torch.no_grad()
def stepwise_greedy(model, emb, mask, start, org, max_new_tokens):
    """Reference decoder: a full forward() over all tokens at every step."""
    go = start.clone()
    go_mask = torch.ones_like(go, dtype=torch.bool)
    finished = torch.zeros(go.size(0), dtype=torch.bool)
    for _ in range(max_new_tokens):
        logits, _ = model(go, emb, mask, go_mask=go_mask, organism_id=org)
        logits = logits[:, -1, :]
        logits[finished] = float("-inf")
        logits[finished, 0] = 0.0
        nxt = logits.argmax(-1, keepdim=True)
        go = torch.cat([go, nxt], 1)
        go_mask = torch.cat([go_mask, (~finished).unsqueeze(1)], 1)
        finished |= torch.isin(nxt.squeeze(1), torch.tensor([2, 4, 6]))
        if finished.all():
            break
    return go


def test_generate_matches_stepwise_forward(model):
    emb, mask = proteins([17, 9, 30])
    start = torch.tensor([[1], [3], [5]])
    org = torch.tensor([0, 2, 4])
    fast = model.generate(emb, mask, start, 25, temperature=0.0, organism_id=org)
    slow = stepwise_greedy(model, emb, mask, start, org, 25)
    assert torch.equal(fast, slow)


def test_padding_does_not_change_predictions(model):
    """A protein decoded alone and next to a longer, padded neighbour gives the same tokens."""
    emb, mask = proteins([11, 40])
    start = torch.tensor([[1], [1]])
    org = torch.tensor([1, 1])
    together = model.generate(emb, mask, start, 20, organism_id=org)
    alone = model.generate(emb[:1, :11], mask[:1, :11], start[:1], 20, organism_id=org[:1])
    assert torch.equal(together[0, : alone.size(1)], alone[0])
    assert (together[0, alone.size(1):] == 0).all()


def test_padded_protein_rows_are_ignored(model):
    emb, mask = proteins([8, 8])
    emb2 = emb.clone()
    mask[:, 6:] = False
    emb2[:, 6:] = 1e3  # garbage in padded rows
    go = torch.tensor([[1, 9, 10], [3, 11, 12]])
    a, _ = model(go, emb, mask, targets=go)
    b, _ = model(go, emb2, mask, targets=go)
    assert torch.allclose(a, b, atol=1e-6)


def test_padded_go_positions_are_ignored(model):
    emb, mask = proteins([10])
    go = torch.tensor([[1, 9, 10]])
    padded = torch.tensor([[1, 9, 10, 0, 0]])
    go_mask = torch.tensor([[True, True, True, False, False]])
    a, _ = model(go, emb, mask, targets=go)
    b, _ = model(padded, emb, mask, go_mask=go_mask, targets=padded)
    assert torch.allclose(a, b[:, :3], atol=1e-6)


def test_last_position_logits(model):
    emb, mask = proteins([12])
    go = torch.tensor([[1, 9, 10, 11]])
    full, _ = model(go, emb, mask, targets=go)
    last, loss = model(go, emb, mask)
    assert loss is None and torch.allclose(full[:, -1:], last, atol=1e-6)


def test_loss_ignores_pad_targets(model):
    emb, mask = proteins([12])
    go = torch.tensor([[1, 9, 10, 11]])
    targets = torch.tensor([[9, 10, 0, 0]])
    logits, loss = model(go, emb, mask, targets=targets)
    expected = F.cross_entropy(logits[0, :2], targets[0, :2])
    assert torch.allclose(loss, expected)


def test_beam_size_one_is_greedy(model):
    emb, mask = proteins([15, 7])
    start = torch.tensor([[1], [5]])
    greedy = model.generate(emb, mask, start, 20)
    beam = model.generate_beam_search(emb, mask, start, 20, beam_size=1, length_penalty=0.0)
    for g, b in zip(greedy, beam):
        g, b = g[g != 0].tolist(), b[b != 0].tolist()
        assert g == b


def test_beam_search_returns_finished_sequences(model):
    emb, mask = proteins([15, 7, 22])
    start = torch.tensor([[1], [3], [5]])
    out = model.generate_beam_search(emb, mask, start, 40, beam_size=4)
    for row, s in zip(out.tolist(), (1, 3, 5)):
        assert row[0] == s
        body = [t for t in row[1:] if t != 0]
        assert body and body[-1] in (2, 4, 6)


def test_block_size_is_enforced(model):
    emb, mask = proteins([126])
    with pytest.raises(ValueError, match="block_size"):
        model(torch.tensor([[1, 9, 10]]), emb, mask)
