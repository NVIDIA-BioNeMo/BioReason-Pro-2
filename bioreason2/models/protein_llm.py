# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutputWithPast

from bioreason2.models import model_family as _mf

from typing import Optional, List, Union

from bioreason2.models.pl.processing_pl import PLProcessor
from bioreason2.models.pl.chat_template_pl import get_chat_template
from bioreason2.models.protein_encoder import create_protein_encoder
from bioreason2.models import projection_norm
from bioreason2.models.go_graph_encoder import create_go_graph_encoder_pipeline
from bioreason2.models.special_tokens import get_all_special_tokens, get_token

# Positions whose logits the training loss computes at once. The full logits of a 16k-token
# sample over Qwen3.5's 248k vocabulary are ~15 GiB in fp32, and backward needs them again.
LOSS_CHUNK = 4096


def _chunk_loss(hidden: torch.Tensor, lm_head: nn.Module, targets: torch.Tensor) -> torch.Tensor:
    """Summed cross-entropy of one chunk, logits in fp32 as transformers computes them.

    Args:
        hidden: final hidden states of the chunk's positions, [n, hidden].
        lm_head: the output projection.
        targets: next-token ids, [n].

    Returns:
        The summed loss, a scalar.
    """
    return nn.functional.cross_entropy(lm_head(hidden).float(), targets, reduction="sum")


def causal_lm_loss(hidden: torch.Tensor, lm_head: nn.Module, labels: torch.Tensor,
                   chunk: int = LOSS_CHUNK) -> torch.Tensor:
    """Mean next-token cross-entropy over the labelled positions, without the full logits.

    The quantity transformers' causal-LM loss computes (shift by one, skip -100, mean over the
    labelled tokens), a chunk of positions at a time, each chunk's logits recomputed in backward
    instead of stored.

    Args:
        hidden: final hidden states, [batch, length, hidden].
        lm_head: the output projection.
        labels: token ids, -100 where unsupervised, [batch, length].
        chunk: positions per chunk.

    Returns:
        The mean loss, a scalar.
    """
    targets = labels[:, 1:].reshape(-1)
    hidden = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    keep = targets != -100
    hidden, targets = hidden[keep], targets[keep]
    if targets.numel() == 0:
        # Nothing supervised (truncation took the whole answer): a zero loss that still reaches
        # every parameter, so backward and DDP's gradient all-reduce proceed.
        return hidden.sum().float() * 0.0
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk):
        end = start + chunk
        total = total + checkpoint(
            _chunk_loss, hidden[start:end], lm_head, targets[start:end], use_reentrant=False
        )
    return total / targets.numel()


class ProteinLLMModel(nn.Module):
    """
    A combined model that processes both protein sequences and text inputs.

    The model uses a protein encoder (ESM3) to extract features from protein sequences
    and a text model (LLM) to process text inputs and generate responses. The protein features are
    projected to the text model's embedding space and prepended to the text embeddings.
    """

    def __init__(
        self,
        text_model_name: str,
        protein_model_name: str = "esm3_sm_open_v1",
        cache_dir: Optional[str] = None,
        max_length_protein: int = 2048,
        max_length_text: int = 4096,
        text_model_finetune: bool = True,
        protein_embedding_layer: int = -1,
        go_model_finetune: bool = True,
        attn_implementation: str = "flash_attention_2",
        go_obo_path: Optional[str] = None,
        precomputed_embeddings_path: Optional[str] = None,
        go_hidden_dim: int = 512,
        go_num_gat_layers: int = 3,
        go_num_heads: int = 8,
        go_num_reduced_embeddings: int = 200,  # Update processing_pl.py to use this as well
        go_embedding_dim: int = 2560,
        quantization_config: Optional[object] = None,  # QLoRA quantization config
        model_family: Optional[str] = None,
        protein_projection_layernorm: bool = False,
        protein_projection_norm: str = "",
        protein_projection_norm_target: float = 0.0,
    ):
        """
        Initialize the ProteinLLMModel.

        Args:
            text_model_name: Name of the text model to be used.
            protein_model_name: Name of the protein model to be used (ESM3).
            cache_dir: Directory to cache the models.
            max_length_protein: Maximum length of protein sequences. Defaults to 2048.
            max_length_text: Maximum length of text sequences. Defaults to 4096.
            text_model_finetune: Whether to finetune the text model. Defaults to True.
            protein_embedding_layer: ESM3 layer to extract embeddings from. Use -1 for final output (default), 0-N for specific transformer layers. Only works with ESM3 models.
            go_model_finetune: Whether to finetune the GO graph encoder. Defaults to True.
            attn_implementation: Attention implementation to use. Defaults to "flash_attention_2".
            go_obo_path: Path to GO ontology OBO file. If None, GO encoder will be disabled.
            precomputed_embeddings_path: Directory with GO embeddings .safetensors files.
            go_hidden_dim: Hidden dimension for GO GAT layers. Defaults to 512.
            go_num_gat_layers: Number of GAT layers in GO encoder. Defaults to 3.
            go_num_heads: Number of attention heads in GO GAT. Defaults to 8.
            go_num_reduced_embeddings: Number of reduced GO embeddings. Defaults to 200.
            go_embedding_dim: GO embedding dimension. Defaults to 2560.
            quantization_config: QLoRA quantization configuration for 4-bit training. Defaults to None.
            protein_projection_layernorm: Append a LayerNorm to the protein projection. Defaults to False.
            protein_projection_norm: Normalisation variant for the projection tail, one of
                `bioreason2.models.projection_norm.VARIANTS`. Empty derives it from
                `protein_projection_layernorm`. Defaults to "".
            protein_projection_norm_target: Initial output RMS for the variants that take
                one; 0 uses the variant's default. Defaults to 0.0.
        """
        super().__init__()

        self.text_model_finetune = text_model_finetune
        self.protein_embedding_layer = protein_embedding_layer
        self.go_model_finetune = go_model_finetune
        self.max_length_protein = max_length_protein
        self.max_length_text = max_length_text

        # Qwen3.5 needs its own loader, nests its config and carries an unused vision
        # tower. Everything backbone-specific lives in bioreason2/models/model_family.py.
        self.model_family = model_family or _mf.detect_family(text_model_name, cache_dir)
        print(f"Text model family: {self.model_family} -- {_mf.describe(self.model_family)}")

        self.text_model, self.text_tokenizer = _mf.load_text_model(
            self.model_family,
            text_model_name,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )

        self.text_config = self.text_model.config
        self.text_tokenizer.chat_template = get_chat_template(text_model_name)

        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        # Add special tokens from centralized module
        all_special_tokens = get_all_special_tokens()
        self.text_tokenizer.add_special_tokens({"additional_special_tokens": all_special_tokens})
        self.protein_token_id = self.text_tokenizer.convert_tokens_to_ids(get_token("protein_pad"))
        self.go_token_id = self.text_tokenizer.convert_tokens_to_ids(get_token("go_graph_pad"))

        # Resize ONLY when the tokenizer outgrows the embedding matrix.
        #
        # Qwen3.5 ships a PADDED matrix (vocab_size 248320 vs 248077 real tokens), so
        # <|protein_pad|> and <|go_graph_pad|> land inside the existing rows and an
        # unconditional resize would SHRINK it 248320 -> 248079, silently discarding
        # 241 trained embedding rows and re-tying lm_head.
        _cur_rows = self.text_model.get_input_embeddings().weight.shape[0]
        if len(self.text_tokenizer) > _cur_rows:
            print(f"Resizing token embeddings {_cur_rows} -> {len(self.text_tokenizer)}")
            self.text_model.resize_token_embeddings(len(self.text_tokenizer))
        else:
            print(
                f"Token embeddings not resized: {_cur_rows} rows already cover "
                f"{len(self.text_tokenizer)} tokens (padded vocab)."
            )

        # Load the protein encoder (ESM3)
        self.protein_encoder = create_protein_encoder(protein_model_name, embedding_layer=protein_embedding_layer)
        self.protein_model = self.protein_encoder.model

        # Get embedding dimensions
        # Qwen3.5's top-level config is multimodal: `hidden_size` there is the VISION
        # tower's width. The language width lives under `text_config`.
        self.text_hidden_size = _mf.text_hidden_size(self.text_config)
        self.protein_hidden_size = self.protein_encoder.embedding_dim

        # Initialize GO graph encoder if paths are provided
        self.go_encoder = None
        if go_obo_path is not None and precomputed_embeddings_path is not None:
            self.go_encoder = create_go_graph_encoder_pipeline(
                go_obo_path=go_obo_path,
                precomputed_embeddings_path=precomputed_embeddings_path,
                hidden_dim=go_hidden_dim,
                num_gat_layers=go_num_gat_layers,
                num_heads=go_num_heads,
                num_reduced_embeddings=go_num_reduced_embeddings,
                embedding_dim=go_embedding_dim,
            )

            # Create projection layer for GO embeddings to text space
            self.go_projection = nn.Sequential(
                nn.Linear(go_embedding_dim, self.text_hidden_size),
                nn.GELU(),
                nn.Linear(self.text_hidden_size, self.text_hidden_size),
            )
        else:
            self.go_projection = None

        # Create projection layer to map protein embeddings to text model's embedding space
        # Optional trailing LayerNorm matches the scale of the token embeddings it replaces.
        self.protein_projection_norm = projection_norm.resolve(
            protein_projection_norm, protein_projection_layernorm)
        self.protein_projection_layernorm = protein_projection_layernorm
        self.protein_projection = projection_norm.build_protein_projection(
            self.protein_hidden_size, self.text_hidden_size,
            self.protein_projection_norm, protein_projection_norm_target,
        )

        # Initialize all models in eval mode with frozen parameters by default
        # Training setup will be handled by train_protein_llm.py
        self._setup_default_eval_mode()

        # Create processor for handling inputs
        self.processor = PLProcessor(tokenizer=self.text_tokenizer)

    def _setup_default_eval_mode(self):
        """
        Set all model components to eval mode with frozen parameters by default.
        Training setup will be handled by train_protein_llm.py.
        """
        # Text model: eval mode, frozen
        self.text_model.eval()
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        # Protein encoder: frozen, never trained
        self.protein_encoder.freeze()
        
        # Protein projection: eval mode, frozen
        self.protein_projection.eval()
        for param in self.protein_projection.parameters():
            param.requires_grad = False
            
        # GO encoder: eval mode, frozen
        if self.go_encoder is not None:
            self.go_encoder.eval()
            for param in self.go_encoder.parameters():
                param.requires_grad = False
        
        # GO projection: eval mode, frozen
        if self.go_projection is not None:
            self.go_projection.eval()
            for param in self.go_projection.parameters():
                param.requires_grad = False

    def process_protein_embeddings(
        self,
        protein_sequences: List[str],
        batch_idx_map: List[int],
        batch_size: int,
        structure_coords: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Process protein sequences and structures to obtain embeddings using the protein encoder.

        Args:
            protein_sequences: List of protein sequence strings
            batch_idx_map: Mapping of each sequence to its batch item
            batch_size: Number of items in the batch
            structure_coords: Optional tensor containing the structure coordinates

        Returns:
            List of tensor embeddings for each batch item
        """
        # Use the protein encoder to get embeddings
        batch_protein_embeddings = self.protein_encoder.encode_sequences(
            protein_sequences=protein_sequences,
            batch_idx_map=batch_idx_map,
            batch_size=batch_size,
            structure_coords=structure_coords,
        )

        # Project all embeddings to text embedding space
        for i in range(batch_size):
            if batch_protein_embeddings[i].numel() > 0:  # Check if tensor is not empty
                batch_protein_embeddings[i] = batch_protein_embeddings[i].to(
                    device=self.protein_projection[0].weight.device,
                    dtype=self.protein_projection[0].weight.dtype,
                )
                batch_protein_embeddings[i] = self.protein_projection(batch_protein_embeddings[i])
            else:
                # Ensure empty tensors have correct dimensions
                batch_protein_embeddings[i] = torch.zeros(
                    (0, self.text_hidden_size),
                    device=self.protein_projection[0].weight.device,
                    dtype=self.protein_projection[0].weight.dtype,
                )

        return batch_protein_embeddings

    def process_go_aspects(
        self,
        go_aspects: Optional[List[str]] = None,
        batch_size: int = 1,
    ) -> Optional[List[torch.Tensor]]:
        """
        GO embeddings for a batch: the whole ontology, encoded once and shared by every example.

        Args:
            go_aspects: One entry per batch item; None means the batch has no GO tokens.
            batch_size: Number of items in the batch

        Returns:
            Optional list of tensors with GO embeddings, one per batch item, each of shape
            (200, text_hidden_size). Returns None if no GO encoder is available.
        """
        if self.go_encoder is None or go_aspects is None:
            return None

        batch_go_embeddings = []

        reduced_embeddings = self.go_encoder()  # (200, 2560)

        # Project to text embedding space
        if self.go_projection is not None:
            reduced_embeddings = reduced_embeddings.to(
                device=self.go_projection[0].weight.device,
                dtype=self.go_projection[0].weight.dtype,
            )
            reduced_embeddings = self.go_projection(reduced_embeddings)  # (200, text_hidden_size)

        # Duplicate for all batch items
        for i in range(batch_size):
            batch_go_embeddings.append(reduced_embeddings)

        return batch_go_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        protein_sequences: Optional[List[str]] = None,
        batch_idx_map: Optional[List[int]] = None,
        structure_coords: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        go_aspects: Optional[List[str]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            protein_sequences: List of protein sequence strings
            batch_idx_map: Batch mapping for protein sequences
            structure_coords: Optional tensor containing the structure coordinates
            labels: Labels for supervised fine-tuning
            go_aspects: GO aspects for protein sequences
            **kwargs: Additional arguments

        Returns:
            Outputs from the text model; with labels, the loss alone
        """
        # Ensure required inputs are available
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")

        batch_size = input_ids.shape[0]

        # Get text embeddings from the model's embedding layer
        text_inputs_embeds = self.text_model.get_input_embeddings()(input_ids)

        # Process GO aspects if provided
        go_embeddings = None
        if go_aspects is not None:
            go_embeddings = self.process_go_aspects(go_aspects, batch_size)

        # Process protein sequences if provided
        if protein_sequences is not None and batch_idx_map is not None:
            # Find positions where protein tokens should be replaced
            mask = input_ids == self.protein_token_id
            n_protein_tokens = mask.sum().item()

            batch_protein_embeds = self.process_protein_embeddings(
                protein_sequences,
                batch_idx_map,
                batch_size,
                structure_coords=structure_coords,
            )
            protein_embeds_flat = torch.cat(batch_protein_embeds, dim=0)
            n_protein_features = protein_embeds_flat.shape[0]

            if n_protein_features != n_protein_tokens:
                raise ValueError(
                    f"Protein features and protein tokens do not match: "
                    f"features {n_protein_features}, tokens: {n_protein_tokens}"
                )

            # Replace protein tokens with actual protein embeddings (out-of-place to preserve autograd graph)
            if n_protein_tokens > 0:
                orig_shape = text_inputs_embeds.shape  # (B, L, H)
                hidden_size = orig_shape[-1]
                embeds_2d = text_inputs_embeds.view(-1, hidden_size)
                mask_flat = mask.view(-1)
                idx = mask_flat.nonzero(as_tuple=False).squeeze(1)
                # Exact out-of-place write. index_copy preserves the autograd graph
                # (it is not the in-place index_copy_) while writing the encoder's
                # vector bit-exactly; the older `a + (b - a)` scatter_add rounded twice
                # in bf16, landing every injected vector ~1 ULP off.
                embeds_2d = embeds_2d.index_copy(0, idx, protein_embeds_flat.to(embeds_2d.dtype))
                text_inputs_embeds = embeds_2d.view(orig_shape)

        # Process GO embeddings if provided
        if go_embeddings is not None:
            # Find positions where GO tokens should be replaced
            go_mask = input_ids == self.go_token_id

            # Count tokens and embeddings
            n_go_tokens = go_mask.sum().item()
            go_embeds_flat = torch.cat([emb for emb in go_embeddings if emb.numel() > 0], dim=0)
            n_go_features = go_embeds_flat.shape[0] if go_embeds_flat.numel() > 0 else 0

            if n_go_features != n_go_tokens:
                raise ValueError(
                    f"GO embeddings and GO tokens do not match: " f"embeddings {n_go_features}, tokens: {n_go_tokens}"
                )

            if n_go_tokens > 0:
                # Ensure GO embeddings have the same dtype as text embeddings
                go_embeds_flat = go_embeds_flat.to(dtype=text_inputs_embeds.dtype)

                # Replace GO tokens with actual GO embeddings (out-of-place)
                orig_shape = text_inputs_embeds.shape  # (B, L, H)
                hidden_size = orig_shape[-1]
                embeds_2d = text_inputs_embeds.view(-1, hidden_size)
                mask_flat = go_mask.view(-1)
                idx = mask_flat.nonzero(as_tuple=False).squeeze(1)
                embeds_2d = embeds_2d.index_copy(0, idx, go_embeds_flat.to(embeds_2d.dtype))
                text_inputs_embeds = embeds_2d.view(orig_shape)

        if labels is None:
            return self.text_model(inputs_embeds=text_inputs_embeds, attention_mask=attention_mask, **kwargs)

        # The loss is taken from the final hidden states rather than through the model's
        # labels, which materialise the full logits (see causal_lm_loss).
        base = self.text_model
        if hasattr(base, "get_base_model"):
            base = base.get_base_model()
        hidden = base.model(
            inputs_embeds=text_inputs_embeds, attention_mask=attention_mask, **kwargs
        ).last_hidden_state
        return CausalLMOutputWithPast(loss=causal_lm_loss(hidden, base.get_output_embeddings(), labels))

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        protein_sequences: Optional[List[str]] = None,
        batch_idx_map: Optional[List[int]] = None,
        structure_coords: Optional[torch.Tensor] = None,
        go_aspects: Optional[List[str]] = None,
        **generation_kwargs,
    ) -> Union[torch.Tensor, List[str]]:
        """
        Generate text based on protein and text inputs.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            protein_sequences: List of protein sequence strings
            batch_idx_map: Batch mapping for protein sequences
            structure_coords: Optional tensor containing the structure coordinates
            go_aspects: GO aspects for protein sequences
            **generation_kwargs: Additional arguments for generation

        Returns:
            Generated token IDs
        """
        # Ensure required inputs are available
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")

        batch_size = input_ids.shape[0]

        # Get text embeddings from the model's embedding layer
        text_inputs_embeds = self.text_model.get_input_embeddings()(input_ids)

        # Process GO aspects if provided
        go_embeddings = None
        if go_aspects is not None:
            go_embeddings = self.process_go_aspects(go_aspects, batch_size)

        # Process protein sequences if provided
        if protein_sequences is not None and batch_idx_map is not None:
            batch_protein_embeds = self.process_protein_embeddings(
                protein_sequences,
                batch_idx_map,
                batch_size,
                structure_coords=structure_coords,
            )

            # Find positions where protein tokens should be replaced
            mask = input_ids == self.protein_token_id

            # Count tokens and embeddings
            n_protein_tokens = mask.sum().item()
            protein_embeds_flat = torch.cat(batch_protein_embeds, dim=0)
            n_protein_features = protein_embeds_flat.shape[0]

            if n_protein_features != n_protein_tokens:
                raise ValueError(
                    f"Protein features and protein tokens do not match: "
                    f"features {n_protein_features}, tokens: {n_protein_tokens}"
                )

            # Ensure protein embeddings have the same dtype as text embeddings
            protein_embeds_flat = protein_embeds_flat.to(dtype=text_inputs_embeds.dtype)

            # Replace protein tokens with actual protein embeddings (out-of-place)
            if n_protein_tokens > 0:
                orig_shape = text_inputs_embeds.shape  # (B, L, H)
                hidden_size = orig_shape[-1]
                embeds_2d = text_inputs_embeds.view(-1, hidden_size)
                mask_flat = mask.view(-1)
                idx = mask_flat.nonzero(as_tuple=False).squeeze(1)
                embeds_2d = embeds_2d.index_copy(0, idx, protein_embeds_flat.to(embeds_2d.dtype))
                text_inputs_embeds = embeds_2d.view(orig_shape)

        # Process GO embeddings if provided
        if go_embeddings is not None:
            # Find positions where GO tokens should be replaced
            go_mask = input_ids == self.go_token_id

            # Count tokens and embeddings
            n_go_tokens = go_mask.sum().item()
            go_embeds_flat = torch.cat([emb for emb in go_embeddings if emb.numel() > 0], dim=0)
            n_go_features = go_embeds_flat.shape[0] if go_embeds_flat.numel() > 0 else 0

            if n_go_features != n_go_tokens:
                raise ValueError(
                    f"GO embeddings and GO tokens do not match: " f"embeddings {n_go_features}, tokens: {n_go_tokens}"
                )

            if n_go_tokens > 0:
                # Ensure GO embeddings have the same dtype as text embeddings
                go_embeds_flat = go_embeds_flat.to(dtype=text_inputs_embeds.dtype)

                # Replace GO tokens with actual GO embeddings (out-of-place)
                orig_shape = text_inputs_embeds.shape  # (B, L, H)
                hidden_size = orig_shape[-1]
                embeds_2d = text_inputs_embeds.view(-1, hidden_size)
                mask_flat = go_mask.view(-1)
                idx = mask_flat.nonzero(as_tuple=False).squeeze(1)
                embeds_2d = embeds_2d.index_copy(0, idx, go_embeds_flat.to(embeds_2d.dtype))
                text_inputs_embeds = embeds_2d.view(orig_shape)

        # Generation with embeddings
        text_inputs_embeds = text_inputs_embeds.to(input_ids.device)
        attention_mask = attention_mask.to(input_ids.device)

        with torch.inference_mode():
            outputs = self.text_model.generate(
                inputs_embeds=text_inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                **generation_kwargs,
            )

        return outputs  
