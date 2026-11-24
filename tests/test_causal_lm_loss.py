# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The chunked training loss equals transformers' causal-LM loss, in value and gradients."""

from typing import Callable, Tuple

import pytest
import torch
from transformers.loss.loss_utils import ForCausalLMLoss

from bioreason2.models.protein_llm import causal_lm_loss

#: A small model; chunks of 3 positions force several chunks and a ragged last one.
VOCAB, HIDDEN, LENGTH, CHUNK = 50, 16, 23, 3

LossFn = Callable[[torch.Tensor, torch.nn.Module, torch.Tensor], torch.Tensor]


def inputs(seed: int) -> Tuple[torch.Tensor, torch.nn.Linear, torch.Tensor]:
    """Hidden states, an output projection and labels with unsupervised (-100) spans.

    Args:
        seed: random seed.

    Returns:
        (hidden, lm_head, labels)
    """
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(2, LENGTH, HIDDEN, generator=g)
    lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.randn(VOCAB, HIDDEN, generator=g))
    labels = torch.randint(0, VOCAB, (2, LENGTH), generator=g)
    labels[0, :7] = -100
    labels[1, 15:] = -100
    return hidden, lm_head, labels


def loss_and_grads(fn: LossFn, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A loss and its gradients with respect to the hidden states and the projection.

    Args:
        fn: (hidden, lm_head, labels) -> loss.
        seed: random seed for the inputs.

    Returns:
        (loss, hidden grad, weight grad)
    """
    hidden, lm_head, labels = inputs(seed)
    hidden.requires_grad_(True)
    loss = fn(hidden, lm_head, labels)
    loss.backward()
    return loss.detach(), hidden.grad, lm_head.weight.grad


def transformers_loss(hidden: torch.Tensor, lm_head: torch.nn.Module, labels: torch.Tensor) -> torch.Tensor:
    """transformers' loss over the full logits.

    Args:
        hidden: final hidden states.
        lm_head: the output projection.
        labels: token ids, -100 where unsupervised.

    Returns:
        The mean loss.
    """
    return ForCausalLMLoss(lm_head(hidden), labels, vocab_size=VOCAB)


def chunked_loss(hidden: torch.Tensor, lm_head: torch.nn.Module, labels: torch.Tensor) -> torch.Tensor:
    """The training loss, a few positions at a time.

    Args:
        hidden: final hidden states.
        lm_head: the output projection.
        labels: token ids, -100 where unsupervised.

    Returns:
        The mean loss.
    """
    return causal_lm_loss(hidden, lm_head, labels, chunk=CHUNK)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_transformers(seed: int) -> None:
    """Loss, hidden-state gradients and projection gradients all match transformers'."""
    ours = loss_and_grads(chunked_loss, seed)
    ref = loss_and_grads(transformers_loss, seed)
    for a, b in zip(ours, ref):
        torch.testing.assert_close(a, b)


def test_chunk_size_does_not_matter() -> None:
    """Small chunks and one whole chunk give the same loss and gradients."""
    small = loss_and_grads(chunked_loss, 3)
    whole = loss_and_grads(causal_lm_loss, 3)
    for a, b in zip(small, whole):
        torch.testing.assert_close(a, b)


def test_nothing_supervised() -> None:
    """With every label -100 the loss is zero and backward still runs, giving zero gradients."""
    hidden, lm_head, labels = inputs(0)
    hidden.requires_grad_(True)
    loss = causal_lm_loss(hidden, lm_head, torch.full_like(labels, -100), chunk=CHUNK)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.count_nonzero(hidden.grad) == 0
