# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""vLLM multimodal model: Qwen3.5 text + a projected ESM3 protein span.

Rollouts must NOT go through `prompt_embeds`, which is how `protein_vllm.py` serves eval:
it sets `prompt_token_ids=None`, which disables `prompt_logprobs` and stops vLLM applying LoRA,
and the first SkyRL multimodal port traced a persistent 14-18% vLLM-vs-FSDP logprob gap to it.
Here the protein is a first-class multimodal input -- `TokensPrompt(prompt_token_ids,
multi_modal_data={"protein": esm3_span})` -- so `input_ids` stay intact. The model owns a copy
of `protein_projection`, projects each span in `embed_multimodal`, and vLLM scatters the result
over the run of `<|protein_pad|>` ids. The GO block needs nothing: checkpoint prep baked it into
`embed_tokens` behind 200 ordinary tokens.

Rules:

* **Subclass, don't wrap**, so weight names stay top-level (`model.*`, `lm_head.*`) and every
  interface of the base class is inherited. vLLM 0.20.2 neither registers `Qwen3_5ForCausalLM`
  nor marks it `IsHybrid` -- only `Qwen3_5ForConditionalGeneration` carries the GDN state
  classmethods -- so this class adds `IsHybrid` and delegates those three.
* The prompt update targets the **whole** placeholder run, or vLLM builds one length-1 range
  per token.
* Profiling uses a worst-case span: 2002 tokens.
* `load_weights` runs on every weight-sync RPC, so the projector sidecar is loaded **once**;
  afterwards the projector belongs to weight sync (`set_modality_parameter`).
* Every parameter must have been loaded, or the model raises instead of generating fluent text
  from a random module.

The projector runs in the model dtype (bf16), as eval's does; the trainer computes in bf16 from
the same bf16-rounded weights, so the two sides project identically.

Registered in every process by the `vllm.general_plugins` entry point in `vllm_plugin.py`.
"""

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn
from transformers import BatchFeature
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import IsHybrid, MultiModalEmbeddings, SupportsMultiModal
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    EmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)

from bioreason2.models.projection_norm import build_protein_projection
from bioreason2.rl.constants import (
    ESM3_DIM,
    MAX_PROTEIN_TOKENS,
    MODALITY,
    PROJECTION_FILENAME,
    PROTEIN_PAD_ID,
    PROTEIN_PAD_TOKEN,
)

#: Passthrough field carrying the raw spans from the processor to `embed_multimodal`.
EMBEDS_FIELD = "protein_embeds"


def check_protein_span(prompt_ids: Sequence[int], pad_id: int, expected_len: int) -> None:
    """Require exactly one contiguous run of `expected_len` protein pads.

    SFT renders `Protein: <|protein_pad|> x n` once per prompt, with no start/end marker tokens,
    so this is the whole span contract.

    Args:
        prompt_ids: Token ids of the prompt.
        pad_id: Id of `<|protein_pad|>`.
        expected_len: Rows of the span being injected.

    Raises:
        ValueError: The pads are missing, split into several runs, or the wrong length.
    """
    positions = [i for i, token in enumerate(prompt_ids) if token == pad_id]
    if not positions:
        raise ValueError("RL vLLM prompt carries a protein span but no <|protein_pad|> ids")
    if positions[-1] - positions[0] + 1 != len(positions):
        raise ValueError(f"<|protein_pad|> ids are not contiguous ({len(positions)} pads, first {positions[0]}, last {positions[-1]})")
    if len(positions) != expected_len:
        raise ValueError(f"prompt has {len(positions)} <|protein_pad|> ids for a {expected_len}-row span")


class ProteinEmbeddingItems(EmbeddingItems):
    """One precomputed ESM3 span per item, shape `(n_tokens, 1536)`.

    `expected_hidden_size` validates the raw ESM3 width at ingest; projection to the LLM width
    happens in `Qwen3_5ProteinForCausalLM.embed_multimodal`.
    """

    def __init__(self, data: torch.Tensor | list[torch.Tensor]) -> None:
        """Wrap the spans.

        Args:
            data: The spans, one per item.
        """
        super().__init__(data, MODALITY, expected_hidden_size=ESM3_DIM)


class ProteinDataParser(MultiModalDataParser):
    """Adds a `protein` sub-parser that treats a span as a precomputed embedding."""

    def _parse_protein_data(self, data: object) -> ModalityDataItems[Any, Any] | None:
        """Normalise the incoming span(s) to a list of 2-D tensors, keeping their dtype.

        Args:
            data: A `(n_tokens, 1536)` tensor, a list of them, or a 3-D batch.

        Returns:
            Wrapped items, or None when there is no protein data.

        Raises:
            ValueError: The data is not 2-D or 3-D.
        """
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            if data.ndim == 2:
                items = [data]
            elif data.ndim == 3:
                items = [data[i] for i in range(data.shape[0])]
            else:
                raise ValueError(f"protein tensor must be 2-D or 3-D, got {tuple(data.shape)}")
        elif isinstance(data, (list, tuple)):
            items = [torch.as_tensor(item) for item in data]
        else:
            items = [torch.as_tensor(data)]
        return ProteinEmbeddingItems(items)

    def _get_subparsers(self) -> Mapping[str, Any]:
        """Register the protein sub-parser next to vLLM's own.

        Returns:
            Modality -> parser.
        """
        parsers = dict(super()._get_subparsers())
        parsers[MODALITY] = self._parse_protein_data
        return parsers


class ProteinProcessingInfo(BaseProcessingInfo):
    """Processing info for a passthrough-embedding modality with no HF processor."""

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        """One protein per prompt, matching the SFT rendering.

        Returns:
            Modality -> maximum items per prompt.
        """
        return {MODALITY: 1}

    def get_data_parser(self) -> MultiModalDataParser:
        """Use the protein-aware parser.

        Returns:
            The parser.
        """
        return ProteinDataParser(expected_hidden_size=ESM3_DIM)

    def get_hf_processor(self, **kwargs: object) -> object:
        """Return the tokenizer; spans are precomputed, so there is no HF processor.

        Args:
            **kwargs: Unused.

        Returns:
            The tokenizer.
        """
        return self.get_tokenizer()

    def protein_pad_id(self) -> int:
        """Id of `<|protein_pad|>`, checked against the SFT assignment.

        Returns:
            The id.

        Raises:
            ValueError: The tokenizer does not map the placeholder to the SFT id, which would
                otherwise scatter spans over the wrong positions.
        """
        token_id = self.get_tokenizer().convert_tokens_to_ids(PROTEIN_PAD_TOKEN)
        if token_id != PROTEIN_PAD_ID:
            raise ValueError(
                f"tokenizer maps {PROTEIN_PAD_TOKEN!r} to {token_id!r}, SFT used {PROTEIN_PAD_ID}; "
                f"load the checkpoint written by bioreason2.rl.prepare_checkpoint"
            )
        return token_id


class ProteinDummyInputsBuilder(BaseDummyInputsBuilder[ProteinProcessingInfo]):
    """Worst-case dummy inputs for vLLM's memory profiling pass."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """A full-length placeholder run per dummy item.

        Args:
            mm_counts: Items per modality.

        Returns:
            The dummy prompt text.
        """
        return PROTEIN_PAD_TOKEN * MAX_PROTEIN_TOKENS * mm_counts.get(MODALITY, 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """One `(2002, 1536)` span per dummy item.

        Args:
            seq_len: Profiling sequence length (unused).
            mm_counts: Items per modality.
            mm_options: vLLM dummy-data overrides (unused).

        Returns:
            The dummy spans.
        """
        count = mm_counts.get(MODALITY, 0)
        return {MODALITY: [torch.zeros(MAX_PROTEIN_TOKENS, ESM3_DIM, dtype=torch.bfloat16) for _ in range(count)]}


class ProteinMultiModalProcessor(BaseMultiModalProcessor[ProteinProcessingInfo]):
    """Maps each protein span onto its run of placeholder tokens."""

    def _get_mm_fields_config(
        self,
        hf_inputs: object,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Keep one variable-length 2-D span per item.

        Args:
            hf_inputs: Unused.
            hf_processor_mm_kwargs: Unused.

        Returns:
            Field -> config.
        """
        return {EMBEDS_FIELD: MultiModalFieldConfig.batched(MODALITY)}

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Map each span's full placeholder run to itself.

        SkyRL already expanded the authored placeholder to `n_tokens` pads before tokenizing.
        Targeting a single pad id would make vLLM emit one length-1 range per pad, so the target
        is the whole run and the resulting range has exactly the span's length.

        Args:
            mm_items: Parsed multimodal items for this prompt.
            hf_processor_mm_kwargs: Unused.
            out_mm_kwargs: Unused.

        Returns:
            One replacement for the protein modality.
        """
        pad_id = self.info.protein_pad_id()
        items = mm_items.get_items(MODALITY, ProteinEmbeddingItems)
        span_ids = [[pad_id] * int(items.get(i).shape[0]) for i in range(len(items))]
        return [
            PromptReplacement(modality=MODALITY, target=span_ids.__getitem__, replacement=span_ids.__getitem__),
        ]

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: object,
        mm_prompt_updates: object,
        is_update_applied: bool,
    ) -> tuple[list[int], Mapping[str, list[Any]]]:
        """Map the placeholders, then check the prompt against the SFT span contract.

        Args:
            mm_items: Parsed multimodal items for this prompt.
            prompt_ids: Token ids of the prompt.
            mm_kwargs: Processed multimodal kwargs.
            mm_prompt_updates: The prompt updates from `_get_prompt_updates`.
            is_update_applied: Whether the prompt already carries the placeholder run.

        Returns:
            The prompt ids and placeholder ranges from the base implementation.

        Raises:
            ValueError: The prompt breaks the span contract.
        """
        prompt_ids, mm_placeholders = super()._maybe_apply_prompt_updates(
            mm_items=mm_items,
            prompt_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_prompt_updates=mm_prompt_updates,
            is_update_applied=is_update_applied,
        )
        if mm_items.get_all_counts().get(MODALITY, 0):
            span = mm_items.get_items(MODALITY, ProteinEmbeddingItems).get(0)
            check_protein_span(prompt_ids, self.info.protein_pad_id(), int(span.shape[0]))
        return prompt_ids, mm_placeholders

    def _apply_hf_processor_mm_only(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Return the passthrough spans directly.

        The base implementation calls the bare tokenizer on empty processor data and raises.

        Args:
            mm_items: Parsed multimodal items.
            hf_processor_mm_kwargs: Unused.
            tokenization_kwargs: Unused.

        Returns:
            The passthrough fields.
        """
        present = {key for key, count in mm_items.get_all_counts().items() if count > 0}
        _processor_data, passthrough_data = self._get_hf_mm_data(mm_items.select(present))
        return BatchFeature(dict(passthrough_data))


@MULTIMODAL_REGISTRY.register_processor(
    ProteinMultiModalProcessor,
    info=ProteinProcessingInfo,
    dummy_inputs=ProteinDummyInputsBuilder,
)
class Qwen3_5ProteinForCausalLM(Qwen3_5ForCausalLM, IsHybrid, SupportsMultiModal):
    """Qwen3.5 text model with a projected ESM3 span injected at `<|protein_pad|>`."""

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """The placeholder string for a modality.

        Args:
            modality: Requested modality.
            i: Item index (unused).

        Returns:
            `<|protein_pad|>`.

        Raises:
            ValueError: A modality other than `protein` was requested.
        """
        if modality == MODALITY:
            return PROTEIN_PAD_TOKEN
        raise ValueError(f"only the {MODALITY!r} modality is supported, got {modality!r}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Build the text model, then attach the protein projector in the model dtype.

        Args:
            vllm_config: vLLM config.
            prefix: Weight-name prefix.

        Raises:
            ValueError: The GDN state cache dtype differs from the checkpoint's `mamba_ssm_dtype`.
        """
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_text_config
        # The GDN state cache must run in the checkpoint's mamba_ssm_dtype (float32), as eval did.
        # vLLM sets it only through a hook keyed by architecture name, which vllm_plugin maps for
        # this class; without the plugin the cache falls back to bf16 and nothing says so.
        expected, actual = getattr(config, "mamba_ssm_dtype", None), vllm_config.cache_config.mamba_ssm_cache_dtype
        if expected is not None and actual != expected:
            raise ValueError(
                f"mamba_ssm_cache_dtype is {actual!r} but the checkpoint asks for {expected!r}; install the repo "
                f"(`pip install -e . --no-deps`) so vllm_plugin registers this architecture's config hook"
            )
        self._model_path = vllm_config.model_config.model
        self.protein_projection = build_protein_projection(
            config.protein_embedding_dim,
            config.hidden_size,
            config.protein_projection_norm,
            config.protein_projection_norm_target,
        ).to(vllm_config.model_config.dtype)

    def get_language_model(self) -> nn.Module:
        """The text stack.

        Returns:
            `self.model`; this class is the language model.
        """
        return self.model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Project each raw ESM3 span into the LLM hidden space, one protein at a time.

        Blocknorm takes one RMS per protein, so spans are never concatenated before projecting.

        Args:
            **kwargs: Must contain `protein_embeds`.

        Returns:
            One `(n_tokens, hidden)` tensor per item.

        Raises:
            ValueError: A span is not 2-D or 3-D.
        """
        spans = kwargs.get(EMBEDS_FIELD)
        if spans is None:
            return []
        if isinstance(spans, torch.Tensor):
            if spans.ndim == 2:
                spans = [spans]
            elif spans.ndim == 3:
                spans = [spans[i] for i in range(spans.shape[0])]
            else:
                raise ValueError(f"{EMBEDS_FIELD} must be 2-D or 3-D, got {spans.ndim}-D")

        weight = self.protein_projection[0].weight
        return tuple(self.protein_projection(span.to(device=weight.device, dtype=weight.dtype)) for span in spans)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed the text ids, then scatter the projected spans over the placeholders.

        Overrides the base class's single-argument version, which stays reachable as
        `self.model.embed_input_ids` and is what the text path below uses.

        Args:
            input_ids: Token ids for the batch.
            multimodal_embeddings: Projected spans from `embed_multimodal`.
            is_multimodal: Mask of placeholder positions.

        Returns:
            Input embeddings with the spans substituted in.
        """
        inputs_embeds = self._embed_text_input_ids(input_ids, self.model.embed_input_ids, is_multimodal=is_multimodal)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if is_multimodal is None:
            raise ValueError("is_multimodal is required to scatter protein spans")
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def set_modality_parameter(self, name: str, tensor: torch.Tensor) -> bool:
        """Apply one projector parameter pushed by RL weight sync.

        SkyRL names it `modalities.<modality_id>.projection.<rest>`; here it lives at
        `protein_projection.<rest>`. Without this the engine keeps the step-0 projector while
        the trainer moves on, and nothing errors.

        Args:
            name: Fully qualified SkyRL parameter name.
            tensor: The updated values.

        Returns:
            Whether the parameter was found and written.
        """
        marker = ".projection."
        if not name.startswith("modalities.") or marker not in name:
            return False
        suffix = name.split(marker, 1)[1]
        for param_name, param in self.protein_projection.named_parameters():
            if param_name == suffix:
                with torch.no_grad():
                    param.copy_(tensor.to(dtype=param.dtype, device=param.device))
                return True
        return False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the text weights, and on the first call only, the projector sidecar.

        Called again on every weight-sync RPC. Re-reading the sidecar there would overwrite the
        co-trained projector with the stale on-disk copy and inflate the loaded count that
        SkyRL's partial-load guard compares against.

        Args:
            weights: `(name, tensor)` pairs.

        Returns:
            The names loaded by this call.

        Raises:
            FileNotFoundError: The sidecar is missing on the first load.
            RuntimeError: The sidecar does not match, or some parameter was never loaded.
        """
        loaded = super().load_weights(weights)
        if getattr(self, "_projection_loaded", False):
            return loaded

        path = os.path.join(self._model_path, PROJECTION_FILENAME)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} not found; the protein projector would stay at random init")
        state = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.protein_projection.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"{path} does not match the projector: missing={list(missing)} unexpected={list(unexpected)}")
        loaded |= {f"protein_projection.{name}" for name, _ in self.protein_projection.named_parameters()}
        self._projection_loaded = True

        residual = sorted(name for name, _ in self.named_parameters() if name not in loaded)
        if residual:
            raise RuntimeError(f"{len(residual)} parameter(s) never loaded and still at random init, e.g. {residual[:6]}")
        print(f"[protein-mm] loaded projector from {path}; all {len(loaded)} parameters loaded", flush=True)
        return loaded

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
        """GDN state dtypes, from the class vLLM marks hybrid.

        Args:
            vllm_config: vLLM config.

        Returns:
            Conv and temporal state dtypes.
        """
        return Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig) -> tuple[tuple[int, int], tuple[int, int]]:
        """GDN state shapes, from the class vLLM marks hybrid.

        Args:
            vllm_config: vLLM config.

        Returns:
            Conv and temporal state shapes.
        """
        return Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple:
        """GDN state copy functions for prefix caching, from the class vLLM marks hybrid.

        Returns:
            One copy function per state.
        """
        return Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()


__all__ = ["Qwen3_5ProteinForCausalLM", "check_protein_span"]
