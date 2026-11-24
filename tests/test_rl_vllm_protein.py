# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The protein vLLM model's CPU-checkable contract: span check, weight sync, plugin, config.

Needs vLLM importable (the RL env); no GPU::

    python -m pytest tests/test_rl_vllm_protein.py -q
"""

import json
import os
import pathlib
import sys
import types

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

pytest.importorskip("vllm")

from transformers import AutoConfig, Qwen3_5TextConfig  # noqa: E402
from vllm import ModelRegistry  # noqa: E402
from vllm.config import VllmConfig  # noqa: E402
from vllm.model_executor.models.config import MODELS_CONFIG_MAP, Qwen3_5ForConditionalGenerationConfig  # noqa: E402

from bioreason2.models.projection_norm import build_protein_projection  # noqa: E402
from bioreason2.rl import vllm_plugin  # noqa: E402
from bioreason2.rl.constants import ARCHITECTURE, ESM3_DIM, PROTEIN_PAD_ID  # noqa: E402
from bioreason2.rl.vllm_protein_mm import Qwen3_5ProteinForCausalLM, check_protein_span  # noqa: E402

PAD = PROTEIN_PAD_ID


def test_span_check_accepts_one_run_of_the_right_length() -> None:
    """`Protein: <pad> x n` once, as SFT renders it."""
    check_protein_span([1, 2, PAD, PAD, PAD, 3], PAD, 3)


@pytest.mark.parametrize(
    "ids, length, message",
    [
        ([1, PAD, PAD, 2, PAD, 3], 3, "not contiguous"),
        ([1, PAD, PAD, 3], 3, "for a 3-row span"),
        ([1, 2, 3], 3, "no <|protein_pad|>"),
    ],
)
def test_span_check_refuses_a_broken_run(ids: list[int], length: int, message: str) -> None:
    """A split run, a wrong length, or no pads at all would scatter the span onto the wrong rows."""
    with pytest.raises(ValueError, match=message):
        check_protein_span(ids, PAD, length)


def engine_stub(norm: str = "blocknorm", target: float = 0.0133) -> types.SimpleNamespace:
    """Just enough of the vLLM model to call `set_modality_parameter`: its bf16 projector.

    Args:
        norm: Projection-norm variant.
        target: Its target RMS.

    Returns:
        A stub with a `protein_projection` attribute.
    """
    return types.SimpleNamespace(protein_projection=build_protein_projection(ESM3_DIM, 64, norm, target).to(torch.bfloat16))


@pytest.mark.parametrize("suffix", ["0.weight", "2.bias", "3.weight"])
def test_weight_sync_reaches_every_projector_parameter(suffix: str) -> None:
    """SkyRL's `modalities.<id>.projection.<name>` lands on `protein_projection.<name>`, cast to bf16."""
    stub = engine_stub()
    target = dict(stub.protein_projection.named_parameters())[suffix]
    update = torch.randn(target.shape, dtype=torch.float32)
    assert Qwen3_5ProteinForCausalLM.set_modality_parameter(stub, f"modalities.protein_mod.projection.{suffix}", update)
    assert torch.equal(target, update.to(torch.bfloat16))


@pytest.mark.parametrize("name", ["model.layers.0.mlp.up_proj.weight", "modalities.protein_mod.projection.9.weight"])
def test_weight_sync_reports_what_it_did_not_apply(name: str) -> None:
    """A name that is not a projector parameter returns False, so the fork's guard can raise."""
    assert not Qwen3_5ProteinForCausalLM.set_modality_parameter(engine_stub(), name, torch.zeros(1))


def hook_target() -> types.SimpleNamespace:
    """The slice of a VllmConfig that `try_verify_and_update_config` and Qwen3.5's hook touch.

    Returns:
        A fresh stub: our architecture, a float32 GDN dtype in the checkpoint, cache at "auto".
    """
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            architecture=ARCHITECTURE,
            is_hybrid=False,
            convert_type="none",
            hf_text_config=types.SimpleNamespace(mamba_ssm_dtype="float32"),
        ),
        cache_config=types.SimpleNamespace(mamba_ssm_cache_dtype="auto"),
    )


def test_without_the_hook_the_gdn_state_cache_stays_bf16() -> None:
    """The defect the plugin exists for: vLLM keys the hook by name, so a new name skips it."""
    previous = MODELS_CONFIG_MAP.pop(ARCHITECTURE, None)
    try:
        config = hook_target()
        VllmConfig.try_verify_and_update_config(config)
        assert config.cache_config.mamba_ssm_cache_dtype == "auto"
    finally:
        if previous is not None:
            MODELS_CONFIG_MAP[ARCHITECTURE] = previous


def test_plugin_registers_the_model_and_its_config_hook() -> None:
    """After `register()`, vLLM's own config pass sets the GDN state cache to float32."""
    vllm_plugin.register()
    vllm_plugin.register()
    assert ARCHITECTURE in ModelRegistry.get_supported_archs()
    assert MODELS_CONFIG_MAP[ARCHITECTURE] is Qwen3_5ForConditionalGenerationConfig
    config = hook_target()
    VllmConfig.try_verify_and_update_config(config)
    assert config.cache_config.mamba_ssm_cache_dtype == "float32"


def test_plugin_never_raises_in_a_vllm_without_the_hook(monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    """An eval env on another vLLM runs `register()` too; a moved hook must not break it."""
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.config", None)
    vllm_plugin.register()
    assert "not registered in this vLLM" in capsys.readouterr().out


def test_projector_fields_survive_the_text_config(tmp_path: pathlib.Path) -> None:
    """The fields the vLLM class builds its projector from load back through transformers."""
    config = Qwen3_5TextConfig(num_hidden_layers=4).to_dict()
    config.update(
        architectures=[ARCHITECTURE],
        protein_embedding_dim=ESM3_DIM,
        protein_projection_norm="blocknorm",
        protein_projection_norm_target=0.0133,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    loaded = AutoConfig.from_pretrained(str(tmp_path))
    assert loaded.model_type == "qwen3_5_text"
    assert (loaded.protein_embedding_dim, loaded.protein_projection_norm) == (ESM3_DIM, "blocknorm")
    assert loaded.protein_projection_norm_target == pytest.approx(0.0133)
