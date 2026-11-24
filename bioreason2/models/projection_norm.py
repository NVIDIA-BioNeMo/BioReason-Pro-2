# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Normalisation options for the tail of the protein projection.

The projection's output is SUBSTITUTED for `<|protein_pad|>` token embeddings, so its
scale is compared against the LLM's own input embeddings whether or not anyone chose it.
Measured on the trained 9B checkpoints:

    variant          protein RMS   token RMS   ratio   per-token CV
    none                   7.41      0.0133    556x          0.250
    layernorm              1.02      0.0133     77x          0.012

`nn.LayerNorm` therefore changes TWO things at once, so a LayerNorm run alone cannot tell
which caused a change: it moves the scale 7x closer to the token embeddings, and it
collapses the per-residue norm variation by a factor of 20. The baseline's projected
block carries a magnitude ramp along the chain (6.98 at the N-terminus to 7.79 at 80% of
the way through) and a length-correlated mean; LayerNorm flattens both to 1.02 at every
position.

These variants separate the two factors, so the pair of runs is a 2x2 rather than two
more guesses:

    variant     scale            per-token variance   isolates
    none        ~7.4 (baseline)  kept                 --
    layernorm   ~1.0             destroyed            --
    blocknorm   ~1.0             KEPT                 variance, at layernorm's scale
    lnscale     ~7.4             destroyed            scale, at the baseline's

Placement follows Prot2Text-V2 (arXiv:2505.11194), whose projector is
``LN(W2 . GELU(W1 . E_p))`` -- normalisation after the second linear, which is what this
repo already does. What that paper adds and this one does not is a contrastive alignment
stage that trains the projector to match the decoder's intermediate distribution before
any generation loss; these variants are the non-contrastive way to set the same scale.
"""
import torch
from torch import nn

#: Guard against dividing by the norm of an all-zero block.
EPS = 1e-6

#: Matches protein-projection keys with or without Lightning's `model.` prefix.
PROJECTION_KEY = "protein_projection."


class BlockRMSNorm(nn.Module):
    """Normalise a protein block by ONE scalar, preserving per-residue magnitudes.

    `nn.LayerNorm` normalises each residue independently, which sets every token to the
    same norm and so discards the relative magnitudes across the chain. This divides the
    whole block by a single RMS taken over all its residues at once, so the block lands
    at a chosen scale while the ratio between any two residues is untouched.
    """

    def __init__(self, hidden_size: int, target: float = 1.0):
        """Build the block normaliser.

        Args:
            hidden_size: width of the projection output, kept for interface symmetry
                with ``nn.LayerNorm``.
            target: initial output RMS; learnable thereafter.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.tensor(float(target)))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Rescale one protein's residue embeddings to the learned target RMS.

        Args:
            hidden: (residues, hidden_size) projected embeddings for one protein.

        Returns:
            The same tensor scaled so its overall RMS equals ``weight``.
        """
        scale = hidden.float().pow(2).mean().sqrt().to(hidden.dtype)
        return hidden / (scale + EPS) * self.weight


class ScaledLayerNorm(nn.Module):
    """LayerNorm followed by one learnable scalar, so the output scale is a free choice.

    Isolates the scale half of what `nn.LayerNorm` changes: per-residue normalisation is
    applied exactly as in the `layernorm` variant, then the whole block is multiplied
    back up to the scale the unnormalised projection had.
    """

    def __init__(self, hidden_size: int, target: float = 1.0):
        """Build the scaled normaliser.

        Args:
            hidden_size: width of the projection output.
            target: initial output RMS; learnable thereafter.
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.weight = nn.Parameter(torch.tensor(float(target)))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Normalise each residue, then scale the block to the learned target.

        Args:
            hidden: (residues, hidden_size) projected embeddings for one protein.

        Returns:
            The normalised tensor scaled by ``weight``.
        """
        return self.norm(hidden) * self.weight


#: Variant name -> (builder, default target RMS). `none` is the released configuration
#: and adds no parameters, so its checkpoints stay interchangeable with every prior run.
VARIANTS = {
    "none": (None, 0.0),
    "layernorm": (lambda d, t: nn.LayerNorm(d), 1.0),
    "blocknorm": (BlockRMSNorm, 1.0),
    "lnscale": (ScaledLayerNorm, 7.4),
}


def build(variant: str, hidden_size: int, target: float = 0.0):
    """Return the normalisation module for a variant, or None for `none`.

    Args:
        variant: one of ``VARIANTS``.
        hidden_size: width of the projection output.
        target: initial output RMS; 0 means use the variant's default.

    Returns:
        An ``nn.Module``, or None if the variant adds no normalisation.

    Raises:
        ValueError: if the variant is unknown.
    """
    if variant not in VARIANTS:
        raise ValueError("protein_projection_norm must be one of %s, got %r"
                         % (sorted(VARIANTS), variant))
    builder, default = VARIANTS[variant]
    if builder is None:
        return None
    return builder(hidden_size, target or default)


def build_protein_projection(in_dim: int, hidden_size: int, variant: str = "none",
                             target: float = 0.0) -> nn.Sequential:
    """Build the protein projector: Linear -> GELU -> Linear, plus the variant's tail.

    The one definition shared by training, the vLLM wrappers and RL, so parameter names
    (`0.*`, `2.*`, `3.*`) always match `protein_projection.pt` without remapping.

    Args:
        in_dim: width of the protein encoder output (1536 for ESM3).
        hidden_size: width of the LLM's token embeddings.
        variant: one of ``VARIANTS``.
        target: initial output RMS for the tail; 0 means use the variant's default.

    Returns:
        The projector as an ``nn.Sequential``.

    Raises:
        ValueError: if the variant is unknown.
    """
    layers = [nn.Linear(in_dim, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size)]
    tail = build(variant, hidden_size, target)
    if tail is not None:
        layers.append(tail)
    return nn.Sequential(*layers)


def check_loaded(missing: list, unexpected: list) -> None:
    """Raise if a non-strict load left any protein-projection weight unrestored.

    Args:
        missing: ``missing_keys`` from ``load_state_dict``.
        unexpected: ``unexpected_keys`` from ``load_state_dict``.

    Raises:
        RuntimeError: naming the offending keys and the flag to check.
    """
    bad = [k for k in list(missing) + list(unexpected) if PROJECTION_KEY in k]
    if bad:
        raise RuntimeError(
            "protein projection did not load cleanly (%s); --protein_projection_norm "
            "and --protein_projection_norm_target must match the training run"
            % ", ".join(sorted(bad)[:6]))


def resolve(variant: str, legacy_layernorm: bool) -> str:
    """Reconcile the variant flag with the older boolean it replaces.

    `protein_projection_layernorm` is threaded through four call sites, and its
    checkpoints must keep loading, so it stays authoritative
    whenever the variant was not set explicitly.

    Args:
        variant: the requested variant, possibly empty.
        legacy_layernorm: the value of ``protein_projection_layernorm``.

    Returns:
        The variant name to build.

    Raises:
        ValueError: if both are set and they disagree.
    """
    if not variant:
        return "layernorm" if legacy_layernorm else "none"
    if legacy_layernorm and variant != "layernorm":
        raise ValueError(
            "protein_projection_layernorm=True conflicts with "
            "protein_projection_norm=%r; set only one" % variant)
    return variant
