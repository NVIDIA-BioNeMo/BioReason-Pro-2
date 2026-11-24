# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""vLLM plugin that registers the protein multimodal architecture in every process.

A driver-side `ModelRegistry.register_model` is not enough: vLLM builds its engine core and its
TP workers in separate processes, which start with a clean registry and then reject
`Qwen3_5ProteinForCausalLM`. `load_general_plugins()` runs in every process and calls whatever
is advertised under the `vllm.general_plugins` entry point, so registering here reaches all of
them. Entry points come from installed distribution metadata, so the repo must be installed
(`pip install -e . --no-deps`) into the environment running vLLM; a bare PYTHONPATH makes the
module importable but invisible to plugin discovery.

Registering the class is not the whole job. vLLM applies per-architecture config hooks from
`MODELS_CONFIG_MAP`, keyed by architecture NAME, and Qwen3.5's hook is what sets the GDN state
cache to the checkpoint's `mamba_ssm_dtype` (float32). Under a new name that hook is skipped and
the state cache silently falls back to bf16 -- a different recurrence from the one SFT eval ran
and the trainer scores -- so the new name is mapped to the same hook.
"""

from bioreason2.rl.constants import ARCHITECTURE

#: Import path of the model, registered as a string so the import stays lazy in forked workers.
MODEL_PATH = "bioreason2.rl.vllm_protein_mm:Qwen3_5ProteinForCausalLM"


def register() -> None:
    """Register the architecture and give it Qwen3.5's config hook. Idempotent.

    Runs in EVERY vLLM process of any env this repo is installed into, eval envs on other vLLM
    versions included, so nothing here may raise. A failure is printed instead; it cannot pass
    silently where it matters, because an RL engine then either rejects the unknown architecture or
    the model refuses to build without its float32 GDN state.
    """
    try:
        from vllm import ModelRegistry

        if ARCHITECTURE not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(ARCHITECTURE, MODEL_PATH)
        from vllm.model_executor.models.config import MODELS_CONFIG_MAP, Qwen3_5ForConditionalGenerationConfig
    except (ImportError, AttributeError) as exc:
        print(f"[bioreason2.rl.vllm_plugin] not registered in this vLLM: {type(exc).__name__}: {exc}", flush=True)
        return
    MODELS_CONFIG_MAP.setdefault(ARCHITECTURE, Qwen3_5ForConditionalGenerationConfig)


__all__ = ["ARCHITECTURE", "MODEL_PATH", "register"]
