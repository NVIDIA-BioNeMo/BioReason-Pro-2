# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""What is specific to the Qwen3.5 backbone, kept in one place.

`protein_llm.py`, `protein_vllm.py` and the trainer call through here rather than assume
a plain causal LM, because Qwen3.5 is not one:

| | Qwen3.5 (`Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`) |
|---|---|
| HF class | `Qwen3_5ForConditionalGeneration` (a **VLM**) |
| loader | `AutoModelForImageTextToText` (`AutoModelForCausalLM` cannot build it) |
| hidden size | `config.text_config.hidden_size` |
| vision tower | present, unused, must be frozen **and** kept in eval |
| RoPE | **M-RoPE**, which vLLM cannot combine with `prompt_embeds` |
| transformers | **>=5.5** (`qwen3_5` model type does not exist before it) |

Where a Qwen3.5 specific is load-bearing, the reason is recorded at the call site.
"""

from __future__ import annotations

from typing import Any, Tuple

QWEN35 = "qwen3_5"
_SUPPORTED = (QWEN35,)


def detect_family(model_name: str, cache_dir: str | None = None) -> str:
    """Return the family for `model_name`.

    Prefers the HF config's `model_type` (authoritative, and survives local paths
    and mirrors under other names). Falls back to the string only when the config
    cannot be read, e.g. offline.

    Args:
        model_name: Hub id or local directory of the backbone or a checkpoint.
        cache_dir: HuggingFace cache.

    Returns:
        QWEN35.

    Raises:
        ValueError: for any other backbone.
    """
    model_type = None
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        model_type = getattr(cfg, "model_type", None)
    except Exception:
        pass

    if model_type and (model_type.startswith("qwen3_5") or model_type.startswith("qwen3.5")):
        return QWEN35
    lowered = str(model_name).lower().replace("-", ".").replace("_", ".")
    if model_type is None and ("qwen3.5" in lowered or "qwen3.6" in lowered):
        return QWEN35
    raise ValueError(
        f"{model_name!r} (model_type {model_type!r}) is not a supported backbone. "
        f"Supported: {_SUPPORTED}."
    )


def is_vlm(family: str) -> bool:
    """Qwen3.5 ships a vision tower even when used purely as a text model."""
    return family == QWEN35


def text_hidden_size(config: Any) -> int:
    """Hidden size of the *language* stack.

    Qwen3.5's top-level config is multimodal; the language dims live under
    `text_config`. Reading `config.hidden_size` there silently returns the vision
    tower's width, which would build projections of the wrong shape and only fail
    much later as a shape mismatch inside the forward pass.
    """
    return getattr(config, "text_config", config).hidden_size


def text_vocab_size(config: Any) -> int:
    """Vocabulary size of the language stack (same nesting caveat as above)."""
    return getattr(config, "text_config", config).vocab_size


def load_text_model(
    family: str,
    model_name: str,
    *,
    cache_dir: str | None = None,
    attn_implementation: str = "flash_attention_2",
    dtype: Any = None,
    quantization_config: Any = None,
) -> Tuple[Any, Any]:
    """Load the backbone and its tokenizer.

    Args:
        family: the backbone family, from `detect_family`.
        model_name: Hub id or local directory.
        cache_dir: HuggingFace cache.
        attn_implementation: attention kernel.
        dtype: weight dtype; bf16 when None.
        quantization_config: bitsandbytes config (QLoRA).

    Returns:
        `(model, tokenizer)`.
    """
    import torch
    # AutoModelForCausalLM cannot instantiate a conditional-generation VLM.
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    dtype = dtype or torch.bfloat16
    kwargs = {
        "cache_dir": cache_dir,
        "trust_remote_code": True,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config

    model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    # Left padding and the backbone's pad id in the generation config are what the released
    # checkpoints' tokenizer and generation files record; save_hf_ckpt.py writes both.
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True, padding_side="left"
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def set_gradient_checkpointing(model, enabled: bool):
    """Turn gradient checkpointing on or off for training.

    Non-reentrant, as the released multi-GPU runs used.

    Measured in the training container on one H100, Qwen3.5-4B + LoRA, at a FIXED
    seq 2024, fwd+bwd --

        checkpointing ON   369 ms/step, peak 12.2 GB
        checkpointing OFF  239 ms/step, peak 25.5 GB     1.55x, +13 GB

    That 1.55x is real but it is NOT safe to take by default, which a real run
    settled: with it off, an 8-GPU training job OOM'd at the optimizer step. The
    microbenchmark pinned sequence length at 2024; real batches run out to
    --max_length_text (16384), and it excluded ESM3, the GO encoder, optimizer states
    and DDP buffers. At batch 4 with it off, peak was already 70.5 GB, and batch 8
    OOM'd even in the microbenchmark.

    So checkpointing stays ON unless sequence length is capped well below the
    maximum, in which case --gradient_checkpointing False is worth ~1.5x.

    Args:
        model: the backbone, with or without adapters.
        enabled: checkpoint every decoder layer.

    Returns:
        The model.
    """
    if enabled:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return model
    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception as exc:  # never let a perf switch kill a run
            print(f"WARNING: gradient_checkpointing_disable() failed ({exc}); "
                  f"leaving checkpointing enabled")
    return model


def apply_lora(
    family: str,
    model: Any,
    *,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Any,
    seed: int = 23,
) -> Any:
    """Attach LoRA adapters to the language stack.

    PEFT's default initialisation, drawn right after seeding, as every released
    checkpoint trained with.

    Args:
        family: the backbone family.
        model: the backbone.
        r: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout.
        target_modules: module names to adapt, resolved by `target_modules_for`.
        seed: adapter init seed.

    Returns:
        The adapted model.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import set_seed

    # The protein/GO pad embeddings are overwritten on every forward and lm_head is
    # tied, so nothing goes in modules_to_save.
    cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    set_seed(seed)
    return get_peft_model(model, cfg)


def freeze_vision_tower(family: str, model: Any) -> bool:
    """Freeze and eval Qwen3.5's unused vision tower.

    Must run *after* `.train()`, which puts every submodule, the tower included, back
    in train mode.

    Returns True if a tower was found and frozen.
    """
    if not is_vlm(family):
        return False

    visual = None
    for path in (
        ("base_model", "model", "model", "visual"),   # PEFT-wrapped
        ("model", "model", "visual"),                 # PEFT-wrapped, shallower
        ("model", "visual"),                          # plain HF
        ("visual",),
    ):
        node = model
        try:
            for attr in path:
                node = getattr(node, attr)
        except AttributeError:
            continue
        visual = node
        break

    if visual is None:
        # Not fatal: a text-only export of Qwen3.5 legitimately has no tower.
        print("WARNING: Qwen3.5 vision tower not found; nothing to freeze.")
        return False

    visual.requires_grad_(False)
    visual.eval()
    return True


def describe(family: str) -> str:
    """One line naming how the family is loaded.

    Args:
        family: the backbone family.

    Returns:
        A description for the run log.
    """
    return {
        QWEN35: "Qwen3.5 (VLM, AutoModelForImageTextToText, nested text_config, vision tower frozen)",
    }.get(family, family)

_NEVER_ADAPT = ("lm_head",)

# --- LoRA target modules ------------------------------------------------------
#
# Qwen3.5-9B is a HYBRID. Measured module census (32 layers):
#     in_proj_qkv / in_proj_z / in_proj_b / in_proj_a / out_proj   x24  (linear attention)
#     q_proj / k_proj / v_proj / o_proj                            x8   (standard attention)
#     gate_proj / up_proj / down_proj                              x32  (all MLPs)
#
# A q/k/v/o + MLP list matches only 8*4 + 32*3 = 128 modules and leaves the attention of
# 24 of 32 layers (75%) completely unadapted -- silently. LoRA attaches, gradients flow,
# loss looks healthy. What must be held is "adapt every attention and MLP linear at
# r=128 / alpha=256 / dropout=0", which is why the trainer resolves the list from the
# live module tree.
_TARGETS = {
    QWEN35: [
        # standard-attention layers (8 of 32)
        "q_proj", "k_proj", "v_proj", "o_proj",
        # linear-attention layers (24 of 32)
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
        # every MLP
        "gate_proj", "up_proj", "down_proj",
    ],
}


def all_linear_targets(model: Any) -> list:
    """Every Linear leaf name in the language stack -- i.e. `"all-linear"`, computed.

    This is `target_modules="all-linear"` in substance, but resolved here instead of
    inside PEFT, which is strictly better than a hardcoded list: a new backbone
    with novel module names (Qwen3.5's `in_proj_qkv` was exactly that) is picked up
    automatically instead of being silently skipped.

    Excludes the vision tower (never fed) and `lm_head` (tied to input embeddings) --
    the same exclusions PEFT's own "all-linear" applies.
    """
    import torch.nn as nn

    names = set()
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if "visual" in name or "lora_" in name or name.endswith("base_layer"):
            continue
        leaf = name.split(".")[-1]
        if leaf in _NEVER_ADAPT:
            continue
        names.add(leaf)
    return sorted(names)


def target_modules_for(family: str, model: Any = None) -> list:
    """LoRA target module names.

    With `model`, returns true all-linear coverage resolved from the live module tree
    (preferred -- see `all_linear_targets`). Without it, falls back to the static
    per-family list, which is kept only so the targets can be inspected without
    instantiating a model.
    """
    if model is not None:
        return all_linear_targets(model)
    if family not in _TARGETS:
        raise ValueError(f"No LoRA targets registered for family {family!r}")
    return list(_TARGETS[family])


def decoder_layer_classes(model: Any) -> set:
    """Transformer block classes to shard, resolved from the live module tree.

    Same reason as `all_linear_targets`: a hardcoded class name silently wraps
    nothing on a new backbone, and FSDP that wraps nothing shards nothing while
    still reporting a healthy run. Vision blocks are excluded -- the tower is
    frozen and never fed.

    Args:
        model: the instantiated backbone.

    Returns:
        Set of module classes, suitable for FSDP's auto_wrap_policy.
    """
    classes = set()
    for _, mod in model.named_modules():
        name = type(mod).__name__
        if "DecoderLayer" in name and "Vision" not in name:
            classes.add(type(mod))
    return classes


def submodule_state_dict(full: dict, prefix: str) -> dict:
    """Slice one submodule's state dict out of a whole-model state dict.

    Rebases the keys so the result loads straight into the submodule, matching
    what `submodule.state_dict()` returns. Callers go through the whole-model
    dict because under FSDP a submodule's own `state_dict()` yields this rank's
    shard, not the parameter.

    Args:
        full: state dict of the enclosing module.
        prefix: dotted path of the submodule, trailing dot included.

    Returns:
        Mapping of submodule-relative name to tensor; empty when nothing matches.
    """
    return {k[len(prefix):]: v for k, v in full.items() if k.startswith(prefix)}


def lora_coverage(model: Any, target_modules) -> dict:
    """Audit LoRA coverage over the whole language stack.

    Two distinct questions, and only checking the first is CIRCULAR:

      1. Did every module *named in the target list* get an adapter?
      2. Is there any Linear in the language stack that the target list never
         mentions? -- i.e. a module we forgot. This is the one that catches a new
         architecture. Qwen3.5's `in_proj_qkv` was exactly this case: it is not in
         Qwen3's list, so a list-only check reported a clean bill of health while
         75% of attention went unadapted.

    `unmatched` answers (2): plain Linears in the language stack, excluding the vision
    tower, `lm_head`, and PEFT's own internal `base_layer` / `lora_*` children.
    """
    import torch.nn as nn

    targets = set(target_modules)
    adapted, plain, unmatched = {}, {}, {}
    for name, mod in model.named_modules():
        if "visual" in name:
            continue
        leaf = name.split(".")[-1]
        if leaf in targets:
            if hasattr(mod, "lora_A"):
                adapted[leaf] = adapted.get(leaf, 0) + 1
            elif isinstance(mod, nn.Linear):
                plain[leaf] = plain.get(leaf, 0) + 1
            continue
        # Anything else that is a real Linear and not PEFT plumbing.
        if (
            isinstance(mod, nn.Linear)
            and leaf not in _NEVER_ADAPT
            and "lora_" not in name
            and not name.endswith("base_layer")
        ):
            unmatched[leaf] = unmatched.get(leaf, 0) + 1
    return {
        "adapted": adapted,
        "unadapted": plain,
        "unmatched": unmatched,
        "n_adapted": sum(adapted.values()),
        "n_unadapted": sum(plain.values()),
        "n_unmatched": sum(unmatched.values()),
    }


def assert_lora_coverage(model: Any, target_modules, strict: bool = True) -> dict:
    """Fail loudly when a targetable Linear did not receive an adapter.

    Without this the mismatch is invisible: training proceeds, the loss is finite, and
    the run silently compares a fully-adapted model against a partially-adapted one.
    """
    cov = lora_coverage(model, target_modules)
    print(
        f"LoRA coverage: {cov['n_adapted']} adapted, {cov['n_unadapted']} unadapted, "
        f"{cov['n_unmatched']} not covered by the target list"
    )
    print(f"  by module: {cov['adapted']}")
    if cov["n_unmatched"]:
        msg = (
            f"LoRA target list does not mention {cov['n_unmatched']} Linear module(s) "
            f"in the language stack: {cov['unmatched']}. Either add them to "
            f"target_modules_for() or add them to _NEVER_ADAPT with a reason."
        )
        if strict:
            raise RuntimeError(msg)
        print("WARNING: " + msg)
    if cov["n_unadapted"]:
        msg = (
            f"LoRA coverage gap: {cov['n_unadapted']} targetable Linear module(s) "
            f"received no adapter: {cov['unadapted']}. The target list does not match "
            f"this architecture."
        )
        if strict:
            raise RuntimeError(msg)
        print("WARNING: " + msg)
    if cov["n_adapted"] == 0:
        raise RuntimeError("LoRA attached to NOTHING -- target_modules match no module.")
    return cov
