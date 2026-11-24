# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SkyRL modality handlers for the protein span.

SkyRL scans `input_ids` for the placeholder id, turns each run into a `(start, length)` span,
and replaces it with `projector.project(encoder.encode(payload))`. These two handlers make that
path reproduce the SFT/eval injection exactly:

* `ProteinCachedEncoder` reads the precomputed ESM3 layer-37 span (`[L+2, 1536]` bf16, BOS/EOS
  included) from the split's safetensors cache, keyed by the payload's `protein_id`. ESM3 is
  frozen and never loaded under RL.
* `ProteinProjectionHandler` is the SFT projector, built by the shared
  `build_protein_projection`, so `protein_projection.pt` loads without remapping and the
  weight-sync name `modalities.<id>.projection.<name>` maps onto the vLLM model's
  `protein_projection.<name>` by a prefix swap.

The projector keeps **fp32 master weights** -- in bf16, AdamW steps at RL learning rates round
to nothing (the blocknorm scale is ~0.0133) -- but **computes in bf16 with bf16-rounded
weights**, which is exactly what eval does and what the vLLM engine does with the synced
weights. Trainer and engine therefore project identically at every step, and step 0
reproduces the published eval.

Registered via Hydra overrides::

    +modalities.protein_mod.placeholder_token="'<|protein_pad|>'"
    +modalities.protein_mod.max_placeholder_tokens=2002
    +modalities.protein_mod.encoder.target=bioreason2.rl.handlers:ProteinCachedEncoder
    +modalities.protein_mod.encoder.kwargs.cache_dir=<split cache>
    +modalities.protein_mod.projection.target=bioreason2.rl.handlers:ProteinProjectionHandler
    +modalities.protein_mod.projection.kwargs.projection_path=<rl ckpt>/protein_projection.pt
"""

import os
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn
from torch.func import functional_call

from bioreason2.models.projection_norm import build_protein_projection
from bioreason2.rl.constants import ESM3_DIM

#: Key of the embedding tensor inside each cache file, as eval.py writes it under
#: PROTEIN_EMBED_CACHE_WRITE.
CACHE_KEY = "emb"

#: Spans kept in memory per process. A step touches 32 proteins, each once per rollout, and an
#: unrepeated epoch never comes back to them, so reuse is within a step. Every trainer rank and
#: every engine actor holds its own copy, at up to 6 MB per 2002-row span.
DEFAULT_LRU_SIZE = 128

#: The dtype eval and the vLLM engine project in.
COMPUTE_DTYPE = torch.bfloat16


class ProteinCachedEncoder:
    """Reads one precomputed ESM3 span per payload from a per-split safetensors cache."""

    def __init__(
        self,
        modality_id: str,
        role: str,
        *,
        cache_dir: str,
        embedding_dim: int = ESM3_DIM,
        lru_size: int = DEFAULT_LRU_SIZE,
        **_: Any,
    ) -> None:
        """Bind the encoder to one cache directory.

        Args:
            modality_id: SkyRL modality id this handler serves.
            role: Handler role, always ``"encoder"``.
            cache_dir: Directory of ``<protein_id>.safetensors`` files for this RL split.
            embedding_dim: Expected ESM3 width, checked per span.
            lru_size: Spans kept in memory.

        Raises:
            NotADirectoryError: `cache_dir` does not exist.
        """
        if not os.path.isdir(cache_dir):
            raise NotADirectoryError(f"protein embedding cache {cache_dir} is not a directory")
        self.modality_id = modality_id
        self.role = role
        self.cache_dir = cache_dir
        self.embedding_dim = embedding_dim
        self.lru_size = lru_size
        self._lru: OrderedDict[str, torch.Tensor] = OrderedDict()
        # The engine actor can encode from more than one thread; OrderedDict reordering is not atomic.
        self._lock = threading.Lock()

    def _load(self, protein_id: str) -> torch.Tensor:
        """Return one span, from memory when possible.

        Args:
            protein_id: Cache key.

        Returns:
            The cached `[L+2, embedding_dim]` tensor.

        Raises:
            FileNotFoundError: The protein is not in the cache.
        """
        with self._lock:
            if protein_id in self._lru:
                self._lru.move_to_end(protein_id)
                return self._lru[protein_id]

        path = os.path.join(self.cache_dir, f"{protein_id}.safetensors")
        if not os.path.exists(path):
            raise FileNotFoundError(f"modality `{self.modality_id}`: no cached embedding for {protein_id} at {path}")
        span = load_file(path)[CACHE_KEY]

        with self._lock:
            self._lru[protein_id] = span
            if len(self._lru) > self.lru_size:
                self._lru.popitem(last=False)
        return span

    def encode(self, payloads: Sequence[Any]) -> list[torch.Tensor]:
        """Read one span per payload.

        Args:
            payloads: Mappings with ``protein_id`` and ``n_tokens``.

        Returns:
            One ``(n_tokens, embedding_dim)`` tensor per payload, in the cache's dtype.

        Raises:
            TypeError: A payload is not a mapping.
            ValueError: A span's shape disagrees with the payload or the expected width.
        """
        outputs = []
        for index, payload in enumerate(payloads):
            if not isinstance(payload, Mapping):
                raise TypeError(
                    f"modality `{self.modality_id}` expects mapping payloads; got "
                    f"{type(payload).__name__} at index {index}"
                )
            protein_id = payload["protein_id"]
            span = self._load(protein_id)
            expected = (int(payload["n_tokens"]), self.embedding_dim)
            if tuple(span.shape) != expected:
                raise ValueError(
                    f"modality `{self.modality_id}`: span for {protein_id} has shape {tuple(span.shape)}, "
                    f"the payload declares {expected}; the dataset and the cache disagree"
                )
            outputs.append(span)
        return outputs


class ProteinProjectionHandler(nn.Sequential):
    """The trainable SFT protein projector, fp32 master weights with bf16 compute."""

    def __init__(
        self,
        modality_id: str,
        role: str,
        *,
        projection_path: str | None = None,
        output_dim: int,
        embedding_dim: int = ESM3_DIM,
        norm: str = "none",
        norm_target: float = 0.0,
        dtype: str = "float32",
        **_: Any,
    ) -> None:
        """Build the projector and load the exported weights strictly.

        Args:
            modality_id: SkyRL modality id this handler serves.
            role: Handler role, always ``"projection"``.
            projection_path: Path to ``protein_projection.pt``.
            output_dim: LLM hidden size.
            embedding_dim: ESM3 width.
            norm: Projection-norm variant the checkpoint was trained with.
            norm_target: Its initial target RMS.
            dtype: Storage dtype of the master weights.

        Raises:
            ValueError: No path was given; a random projector would not error, it would look
                like an RL run that does not learn.
            RuntimeError: The checkpoint does not match the requested shape.
        """
        super().__init__(*build_protein_projection(embedding_dim, output_dim, norm, norm_target))
        self.modality_id = modality_id
        self.role = role

        if not projection_path:
            raise ValueError(f"modality `{modality_id}`: projection_path is required")
        state = torch.load(projection_path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"modality `{modality_id}`: {projection_path} does not match norm={norm!r}: "
                f"missing={list(missing)} unexpected={list(unexpected)}"
            )
        self.to(getattr(torch, dtype))

    def project(self, features: torch.Tensor) -> torch.Tensor:
        """Project one protein span exactly as eval does: bf16 weights, bf16 compute.

        Gradients flow through the casts into the fp32 master weights.

        Args:
            features: ``(n_tokens, embedding_dim)`` ESM3 span.

        Returns:
            ``(n_tokens, output_dim)`` bf16 tensor on the projector's device.
        """
        weights = {name: param.to(COMPUTE_DTYPE) for name, param in self.named_parameters()}
        features = features.to(device=self[0].weight.device, dtype=COMPUTE_DTYPE)
        return functional_call(self, weights, (features,))


__all__ = ["ProteinCachedEncoder", "ProteinProjectionHandler"]
