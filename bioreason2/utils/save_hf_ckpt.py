# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import torch
from bioreason2.models.protein_llm import ProteinLLMModel
from bioreason2.models import model_family as _mf
from bioreason2.models import projection_norm
from pathlib import Path
import argparse
from bioreason2.utils.argparse_utils import str2bool
from transformers import AutoConfig



# --- Qwen3.5 checkpoint layout -------------------------------------------------
# Two layout mismatches must be corrected or vLLM cannot load the result.
_HF_VISUAL_PFX = "model.language_model.visual."
_HF_LM_PFX = "model.language_model."


def _remap_qwen35_key(k: str):
    """Rename a Qwen3.5 key for vLLM, or return None to DROP it.

    HF `save_pretrained` writes text weights under `model.language_model.*`; vLLM
    expects `language_model.model.*`.

    Vision-tower weights are **dropped**, not remapped. Renaming them to `visual.*`
    works on vLLM 0.19.1, which instantiates the tower. We load with
    `language_model_only=True` on vLLM 0.27.1, which does not create a `visual` module
    at all, so those keys fail with:

        ValueError: There is no module or parameter named 'visual' in Qwen3_5Model

    Dropping them is correct for this model regardless of vLLM version: the tower is
    frozen throughout training, is never fed (no image ever enters the graph), and
    only inflates the checkpoint. Returns None for keys to omit.
    """
    # Substring match, not a prefix. The tower's exact path has varied with the PEFT
    # wrapping (`model.language_model.visual.*` here, `model.visual.*` in other
    # arrangements), and a prefix that silently fails to match leaves 333 vision keys in
    # the checkpoint -- which is precisely what happened and only surfaced as a vLLM
    # weight-loading error much later. "visual" is unambiguous in this model.
    if ".visual." in k or k.startswith("visual."):
        return None
    if k.startswith(_HF_LM_PFX):
        return f"language_model.model.{k[len(_HF_LM_PFX):]}"
    return k

def _setup_lora_for_checkpoint_loading(
    model: ProteinLLMModel,
    lora_rank: int = 128,
    lora_alpha: int = 256,
    lora_dropout: float = 0.05,
):
    """Rebuild the adapter topology EXACTLY as training created it.

    This must go through the same dispatch the trainer uses. It previously used a
    static 7-module list, which on Qwen3.5 produces 128 adapters against a
    checkpoint trained with 248 -- the extra 120
    `in_proj_*` LoRA tensors then arrive as "unexpected keys" and conversion aborts.

    Any mismatch here silently changes which weights get merged, so it is worth
    keeping this in lockstep with train_protein_llm.py rather than duplicating a list.

    Args:
        model: the freshly built model, adapters not yet attached.
        lora_rank: LoRA rank the checkpoint trained with.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout.
    """
    family = model.model_family
    target_modules = _mf.target_modules_for(family, model=model.text_model)
    print(f"🔧 LoRA rebuild (family={family}, rank={lora_rank}, alpha={lora_alpha}), "
          f"{len(target_modules)} target names")

    model.text_model = _mf.apply_lora(
        family,
        model.text_model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        seed=42,
    )
    cov = _mf.lora_coverage(model.text_model, target_modules)
    print(f"✅ LoRA rebuilt: {cov['n_adapted']} adapters, "
          f"{cov['n_unadapted']} unadapted, {cov['n_unmatched']} unmatched")


def save_lightning_ckpt(args):
    """Convert a PyTorch Lightning checkpoint to HuggingFace format."""
    checkpoint_path = Path(args.checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📥 Loading Lightning checkpoint: {checkpoint_path}")
    
    # Load checkpoint and extract state_dict
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    
    print(f"📤 Extracted state_dict with {len(state_dict)} keys")
    print(f"📊 Epoch: {checkpoint.get('epoch', 'N/A')} | Global step: {checkpoint.get('global_step', 'N/A')}")
    
    print("Building ProteinLLMModel...")
    
    # Create the model as training built it
    model = ProteinLLMModel(
        text_model_name=args.text_model_name,
        protein_model_name=args.protein_model_name,
        cache_dir=args.cache_dir,
        max_length_protein=args.max_length_protein,
        max_length_text=args.max_length_text,
        text_model_finetune=True,
        protein_embedding_layer=args.protein_embedding_layer,
        go_model_finetune=True,
        attn_implementation="flash_attention_2",
        go_obo_path=args.go_obo_path,
        precomputed_embeddings_path=args.precomputed_embeddings_path,
        go_hidden_dim=args.go_hidden_dim,
        go_num_gat_layers=args.go_num_gat_layers,
        go_num_heads=args.go_num_heads,
        go_num_reduced_embeddings=args.go_num_reduced_embeddings,
        go_embedding_dim=args.go_embedding_dim,
        protein_projection_layernorm=args.protein_projection_layernorm,
        protein_projection_norm=args.protein_projection_norm,
        protein_projection_norm_target=args.protein_projection_norm_target,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"📍 Model on {device}")
    
    # Rebuild the LoRA adapters
    _setup_lora_for_checkpoint_loading(
        model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # Remap keys: Lightning saves with "model." prefix, we need to remove it
    print("🔧 Remapping keys...")
    
    # Filter to only keys with "model." prefix (Lightning wrapper)
    # The checkpoint has duplicates: both "model.X" and "X" for each key
    # We only want the "model." prefixed ones (these are the actual saved weights)
    lightning_keys = {k: v for k, v in state_dict.items() if k.startswith("model.")}
    print(f"📊 Filtering to Lightning-wrapped keys: {len(state_dict)} → {len(lightning_keys)}")
    
    remapped = {}
    for k, v in lightning_keys.items():
        # Remove "model." prefix from Lightning checkpoint
        # The PEFT structure (base_model.model) is already present in the checkpoint
        new_k = k[6:]  # Remove "model." prefix

        # Left on CPU deliberately. `load_state_dict` copies into each parameter and
        # does the host-to-device transfer itself, one tensor at a time, so staging
        # the whole dict on the GPU only buys a second full copy of the weights
        # alongside the model already there. At 27B that is ~54 GB twice on an 80 GB
        # H100 and the conversion OOMs; the 9B fit only because ~18 GB doubles small.
        remapped[new_k] = v
    
    print(f"📊 After removing 'model.' prefix: {len(remapped)} keys")
    
    # Show sample remapped keys for validation
    print("\n🔍 Sample remapped keys:")
    text_model_keys_sample = [k for k in remapped.keys() if "text_model" in k]
    if text_model_keys_sample:
        for key in sorted(text_model_keys_sample)[:5]:
            print(f"  {key}")
        if len(text_model_keys_sample) > 5:
            print(f"  ... and {len(text_model_keys_sample) - 5} more text_model keys")
    
    # Load state dict
    result = model.load_state_dict(remapped, strict=False)
    print(f"\n📥 Loaded: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")
    
    # Analyze and validate loaded keys
    if result.missing_keys:
        print("\n⚠️  Missing keys analysis:")
        # Group by component
        missing_by_component = {}
        for key in result.missing_keys:
            component = key.split('.')[0] if '.' in key else key
            missing_by_component.setdefault(component, []).append(key)
        
        for component, keys in sorted(missing_by_component.items()):
            print(f"  {component}: {len(keys)} keys")
    
    if result.unexpected_keys:
        print("\n⚠️  Unexpected keys analysis:")
        # Group by component
        unexpected_by_component = {}
        for key in result.unexpected_keys:
            component = key.split('.')[0] if '.' in key else key
            unexpected_by_component.setdefault(component, []).append(key)
        
        for component, keys in sorted(unexpected_by_component.items()):
            print(f"  {component}: {len(keys)} keys")
            # Show samples for problematic components
            if component == "text_model" and len(keys) <= 5:
                for k in keys:
                    print(f"    - {k}")
    
    # Validate that critical text_model weights loaded correctly
    text_model_unexpected = [k for k in result.unexpected_keys if k.startswith('text_model.')]
    
    if text_model_unexpected:
        print(f"\n❌ ERROR: {len(text_model_unexpected)} text_model keys were unexpected!")
        print("These weights were NOT loaded from the checkpoint and will use random initialization:")
        for key in text_model_unexpected[:10]:
            print(f"  - {key}")
        if len(text_model_unexpected) > 10:
            print(f"  ... and {len(text_model_unexpected) - 10} more")
        raise RuntimeError(
            f"Failed to load {len(text_model_unexpected)} text_model weights from checkpoint. "
            "The model would contain untrained weights. Please check key remapping logic."
        )
    projection_norm.check_loaded(result.missing_keys, result.unexpected_keys)

    print("\n✅ All critical weights loaded successfully")
    
    # Save missing and unexpected keys to file (in current working directory)
    keys_log_path = "checkpoint_keys.txt"
    with open(keys_log_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Save directory: {args.save_dir}\n")
        f.write(f"Total checkpoint keys: {len(state_dict)}\n")
        f.write(f"Total remapped keys: {len(remapped)}\n\n")
        
        f.write(f"Missing keys ({len(result.missing_keys)}):\n")
        for key in result.missing_keys:
            f.write(f"  {key}\n")
        
        f.write(f"\nUnexpected keys ({len(result.unexpected_keys)}):\n")
        for key in result.unexpected_keys:
            f.write(f"  {key}\n")
    print(f"💾 Saved key mapping log to {keys_log_path}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=False)
    print(f"💾 Saving to {args.save_dir}")
    
    # Capture the FULL multimodal config BEFORE merging. `text_model.save_pretrained`
    # writes the text-only Qwen3_5TextConfig, but vLLM needs the wrapper Qwen3_5Config
    # to load with language_model_only=True. Without this the converted directory looks
    # fine and vLLM simply refuses it.
    _full_config = AutoConfig.from_pretrained(args.text_model_name, cache_dir=args.cache_dir)
    print(f"📐 Captured full multimodal config: {type(_full_config).__name__}")

    # Merge LoRA adapters
    print("🔗 Merging LoRA adapters...")
    model.text_model = model.text_model.merge_and_unload()
    # LoRA matrices can be fp32; cast so save_pretrained writes uniform bf16 rather
    # than a mixed-precision checkpoint.
    model.text_model = model.text_model.to(torch.bfloat16)
    print("✅ LoRA merged into base model (bf16)")
    model = model.cpu()

    # Save all components
    _raw = model.text_model.state_dict()
    _sd = {}
    _dropped = 0
    for k, v in _raw.items():
        nk = _remap_qwen35_key(k)
        if nk is None:
            _dropped += 1
            continue
        _sd[nk] = v
    print(f"🔧 Qwen3.5: {len(_sd)} keys remapped for vLLM, "
          f"{_dropped} vision-tower keys dropped (unused, and absent under "
          f"language_model_only=True)")
    leftover = [k for k in _sd if "visual" in k]
    if leftover:
        raise RuntimeError(
            f"{len(leftover)} vision keys survived the filter, e.g. {leftover[:3]}. "
            "vLLM with language_model_only=True will reject them."
        )
    model.text_model.save_pretrained(args.save_dir, state_dict=_sd)
    model.text_tokenizer.save_pretrained(args.save_dir)
    # Overwrite the text-only config.json that save_pretrained just wrote.
    _full_config.save_pretrained(args.save_dir)
    print("📐 config.json overwritten with the full multimodal config")
    torch.save(model.protein_projection.state_dict(), f"{args.save_dir}/protein_projection.pt")
    torch.save(model.go_projection.state_dict(), f"{args.save_dir}/go_projection.pt")
    torch.save(model.go_encoder.state_dict(), f"{args.save_dir}/go_encoder.pt")
    # ESM3 is not saved: it stays frozen, and inference loads it from the `esm` package.

    # Report parameter counts
    total = sum(p.numel() for p in model.parameters())
    text = sum(p.numel() for p in model.text_model.parameters())
    protein = sum(p.numel() for p in model.protein_model.parameters())
    go_enc = sum(p.numel() for p in model.go_encoder.parameters())
    go_proj = sum(p.numel() for p in model.go_projection.parameters())
    
    print(f"✅ Saved {total/1e6:.1f}M params (text {text/1e6:.1f}M • protein {protein/1e6:.1f}M • GO enc {go_enc/1e6:.1f}M • GO proj {go_proj/1e6:.1f}M)")


def main():
    parser = argparse.ArgumentParser(description="Convert a Lightning checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--text_model_name", type=str, required=True)
    parser.add_argument("--protein_model_name", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_length_text", type=int, default=16384)
    parser.add_argument("--max_length_protein", type=int, default=2000)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--protein_embedding_layer", type=int, default=37)
    parser.add_argument("--go_obo_path", type=str, default=None)
    parser.add_argument("--precomputed_embeddings_path", type=str, default=None)
    parser.add_argument("--go_hidden_dim", type=int, default=512)
    parser.add_argument("--go_num_gat_layers", type=int, default=3)
    parser.add_argument("--go_num_heads", type=int, default=8)
    parser.add_argument("--go_num_reduced_embeddings", type=int, default=200)
    parser.add_argument("--go_embedding_dim", type=int, default=2560)
    parser.add_argument("--protein_projection_layernorm", type=str2bool, default=False,
                        help="Must match how the checkpoint was trained; it adds parameters.")
    parser.add_argument("--protein_projection_norm", type=str, default="",
                        help="Normalisation on the projection tail. Empty derives it from --protein_projection_layernorm, so every existing run is unchanged. blocknorm keeps per-residue magnitudes and fixes only the scale; lnscale does the opposite. Must match how the checkpoint was trained.")
    parser.add_argument("--protein_projection_norm_target", type=float, default=0.0,
                        help="Initial output RMS for the variants that take one; 0 uses the variant's default (1.0 for blocknorm, 7.4 for lnscale).")
    
    save_lightning_ckpt(parser.parse_args())


if __name__ == "__main__":
    main()

