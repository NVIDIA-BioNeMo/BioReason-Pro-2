# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Protein encoder implementation for ESM3.

This module provides the interface for protein embedding generation.
"""

import os
import time

import torch
from abc import ABC, abstractmethod
from typing import List, Optional

# ESM imports
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig


# --- optional in-situ profiler ------------------------------------------------
# Set BRP_TIME_ESM3=1 to report what fraction of each step ESM3 actually costs.
# Measures the ESM3 region with CUDA events and divides by the wall time between
# consecutive calls (= one training/inference step), so the share is measured in
# situ rather than inferred from a standalone benchmark. Off by default and it
# changes no numerics.
_TIME_ESM3 = os.environ.get("BRP_TIME_ESM3") == "1"
_PROF = {"esm_ms": 0.0, "step_ms": 0.0, "n": 0, "last_call_start": None, "warm": 0}
_PROF_EVERY = int(os.environ.get("BRP_TIME_ESM3_EVERY", "20"))
_PROF_WARMUP = int(os.environ.get("BRP_TIME_ESM3_WARMUP", "5"))


def _prof_begin():
    if not _TIME_ESM3:
        return None
    now = time.perf_counter()
    prev = _PROF["last_call_start"]
    _PROF["last_call_start"] = now
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter(), (now - prev) * 1000.0 if prev is not None else None)


def _prof_end(state, n_proteins, n_residues):
    if not _TIME_ESM3 or state is None:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    esm_ms = (time.perf_counter() - state[0]) * 1000.0
    step_ms = state[1]
    _PROF["warm"] += 1
    if _PROF["warm"] <= _PROF_WARMUP or step_ms is None:
        return
    _PROF["esm_ms"] += esm_ms
    _PROF["step_ms"] += step_ms
    _PROF["n"] += 1
    if _PROF["n"] % _PROF_EVERY == 0:
        e = _PROF["esm_ms"] / _PROF["n"]
        st = _PROF["step_ms"] / _PROF["n"]
        print(
            f"[ESM3-PROF] n={_PROF['n']:5d}  esm3={e:7.1f} ms/step  "
            f"step={st:8.1f} ms  share={100.0 * e / st:5.1f}%  "
            f"(last call: {n_proteins} proteins, {n_residues} residues)",
            flush=True,
        )


class ProteinEncoder(ABC):
    """Abstract base class for protein encoders"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self._embedding_dim = None

    @abstractmethod
    def encode_sequences(
        self,
        protein_sequences: List[str],
        batch_idx_map: List[int],
        batch_size: int,
        structure_coords: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Encode protein sequences into embeddings.

        Args:
            protein_sequences: List of protein sequence strings
            batch_idx_map: Mapping of each sequence to its batch item
            batch_size: Number of items in the batch
            structure_coords: Optional tensor containing structure coordinates

        Returns:
            List of tensor embeddings for each batch item
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension"""
        pass

    @property
    @abstractmethod
    def supports_structure(self) -> bool:
        """Whether this encoder supports structure information"""
        pass

    def freeze(self):
        """Put the encoder in eval mode with no trainable parameters; it is never trained."""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False


class ESM3Encoder(ProteinEncoder):
    """ESM3 protein encoder with structure support"""

    def __init__(self, model_name: str, embedding_layer: int = -1):
        super().__init__(model_name)

        # from_pretrained() defaults to cuda:0 for every rank; load on CPU and let
        # Lightning place it. The SDK only casts to bf16 off-CPU, so cast here.
        self.model = ESM3.from_pretrained(model_name, device=torch.device("cpu"))
        self.model = self.model.to(torch.bfloat16)
        self.sampling_config = SamplingConfig
        self._embedding_dim = self.model.encoder.sequence_embed.embedding_dim
        
        # Layer selection for embeddings - ESM3 always has transformer.blocks
        self.total_blocks = len(self.model.transformer.blocks)
        
        self.embedding_layer = self._validate_embedding_layer(embedding_layer)
        self.layer_outputs = {}
        self.hooks = []

        self.freeze()

    def _validate_embedding_layer(self, embedding_layer: int) -> int:
        """
        Validate and normalize the embedding layer parameter.
        
        Args:
            embedding_layer: Layer index to extract embeddings from.
                           -1 means final output (default behavior)
                           0 to total_blocks-1 means specific transformer layer
                           
        Returns:
            Validated layer index
        """
        if embedding_layer == -1:
            return -1  # Final output
        elif 0 <= embedding_layer < self.total_blocks:
            return embedding_layer
        else:
            raise ValueError(
                f"embedding_layer must be -1 (final) or 0-{self.total_blocks-1}, got {embedding_layer}"
            )
    
    def _register_layer_hook(self, layer_idx: int):
        """Register a forward hook to capture output from a specific layer."""
        def hook_fn(module, input, output):
            # esm 3.4 blocks return (hidden states, attention weights); 3.2 returns the states.
            self.layer_outputs[layer_idx] = output[0] if isinstance(output, tuple) else output
        
        block = self.model.transformer.blocks[layer_idx]
        hook = block.register_forward_hook(hook_fn)
        self.hooks.append(hook)
        return hook
    
    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.layer_outputs.clear()

    def encode_sequences(
        self,
        protein_sequences: List[str],
        batch_idx_map: List[int],
        batch_size: int,
        structure_coords: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Encode sequences using ESM3 with optional layer-specific extraction"""

        # The processor caps placeholders at max_length_protein; match it here or ESM3 OOMs.
        if max_length:
            protein_sequences = [s[:max_length] for s in protein_sequences]

        # Initialize result list
        result = [[] for _ in range(batch_size)]

        _prof_state = _prof_begin()

        # Set up layer hook if using specific layer
        if self.embedding_layer != -1:
            self._register_layer_hook(self.embedding_layer)

        # Process each protein sequence individually (ESM3 doesn't support batching)
        for seq_idx, sequence in enumerate(protein_sequences):

            # Check if we have valid structure coordinates to use
            coords_truncated = None
            if structure_coords is not None and structure_coords.numel() > 0:
                coords = structure_coords[batch_idx_map[seq_idx]].to(dtype=torch.float32)
                seq_len = len(sequence)
                
                # Only use coordinates if we have enough and they contain valid data
                if (coords.shape[0] >= seq_len and 
                    not torch.isnan(coords[:seq_len]).all()):
                    coords_truncated = coords[:seq_len]
            
            # Create ESMProtein with or without structure coordinates
            if coords_truncated is not None:
                protein = ESMProtein(sequence=sequence, coordinates=coords_truncated)
            else:
                protein = ESMProtein(sequence=sequence)

            # Encode protein
            protein_tensor = self.model.encode(protein)

            # ESM3 is frozen; its own forward runs under no_grad too, so it could not be trained here.
            with torch.no_grad():

                if self.embedding_layer == -1:
                    # Standard behavior: get final embeddings
                    seq_embeddings = self.model.forward_and_sample(
                        protein_tensor,
                        self.sampling_config(return_per_residue_embeddings=True),
                    ).per_residue_embedding

                else:
                    # Layer-specific extraction: run forward pass and extract from hook
                    self.model.forward_and_sample(
                        protein_tensor,
                        self.sampling_config(return_per_residue_embeddings=True),
                    )
                    
                    # Extract embeddings from the specific layer - ESM3 blocks return direct tensors
                    seq_embeddings = self.layer_outputs[self.embedding_layer]
                    
                    # Layer outputs have extra batch dimension that needs squeezing
                    if seq_embeddings.dim() == 3 and seq_embeddings.shape[0] == 1:
                        seq_embeddings = seq_embeddings.squeeze(0)

            # Get the batch index for this sequence
            batch_idx = batch_idx_map[seq_idx]

            # Add to appropriate batch result
            result[batch_idx].append(seq_embeddings)

        # Clean up hooks
        self._clear_hooks()

        _prof_end(_prof_state, len(protein_sequences),
                  sum(len(x) for x in protein_sequences))

        # Concatenate embeddings for each batch item
        for i in range(batch_size):
            if result[i]:
                result[i] = torch.cat(result[i], dim=0)
            else:
                # Empty tensor for batch items with no proteins
                result[i] = torch.zeros((0, self.embedding_dim))

        return result

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def supports_structure(self) -> bool:
        return True


def create_protein_encoder(model_name: str, embedding_layer: int = -1) -> ProteinEncoder:
    """
    Factory function to create the protein encoder, frozen.

    Args:
        model_name: Name of the protein model ("esm3_sm_open_v1")
        embedding_layer: Layer to extract embeddings from (-1 for final output, 0-N for specific layers)

    Returns:
        The ESM3 encoder

    Raises:
        ValueError: for a model that is not ESM3.
    """
    if "esm3" in model_name.lower():
        return ESM3Encoder(model_name, embedding_layer)
    raise ValueError(f"Unknown protein model '{model_name}'; only ESM3 is supported")
