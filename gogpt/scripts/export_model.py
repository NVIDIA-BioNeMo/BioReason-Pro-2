# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Export a training checkpoint to the directory layout GOGPTPredictor loads.

    python scripts/export_model.py \
        --checkpoint artifacts/training/gogpt_v2/<time>/checkpoints/best/<name>.ckpt \
        --artifacts artifacts/preprocessed/v2/artifacts \
        --output-dir artifacts/export/gogpt-v2

Writes config.json, model.safetensors (decoder weights only, fp32), go_tokenizer.json
and organism_mapper.json, and checks that the exported model reproduces the
checkpoint's logits. Optimizer state and training metadata are not exported.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gogpt.config.model_config import GOGPTConfig  # noqa: E402
from gogpt.models.gogpt import GOGPT  # noqa: E402

GENERATION = {"decoding": "greedy", "max_new_tokens": 200}


def export(model_config, state_dict, artifacts_dir, output_dir):
    """Write the model directory; state_dict holds GOGPT keys (no "model." prefix)."""
    from safetensors.torch import load_model, save_model

    config = GOGPTConfig.from_dict(model_config)
    model = GOGPT(config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, str(output_dir / "model.safetensors"))  # stores the tied wte/lm_head once
    with open(output_dir / "config.json", "w") as f:
        json.dump({"format_version": 2, "model": config.to_dict(), "generation": GENERATION}, f, indent=2)
    for name in ("go_tokenizer.json", "organism_mapper.json"):
        shutil.copyfile(Path(artifacts_dir) / name, output_dir / name)

    # Round trip: the exported weights must give exactly the checkpoint's logits.
    reloaded = GOGPT(config)
    load_model(reloaded, str(output_dir / "model.safetensors"), strict=True)
    reloaded.eval()
    g = torch.Generator().manual_seed(0)
    emb = torch.randn(2, 37, config.protein_embedding_dim, generator=g)
    mask = torch.ones(2, 37, dtype=torch.bool)
    mask[1, 30:] = False
    go = torch.tensor([[config.mf_start_token_id, 10, 11], [config.bp_start_token_id, 12, 13]])
    org = torch.tensor([0, 1])
    with torch.no_grad():
        a, _ = model(go, emb, mask, organism_id=org, targets=go)
        b, _ = reloaded(go, emb, mask, organism_id=org, targets=go)
    if not torch.equal(a, b):
        raise SystemExit("exported model does not reproduce the checkpoint's logits")
    print(f"exported to {output_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Lightning checkpoint from scripts/train.py")
    ap.add_argument("--artifacts", required=True, help="artifacts/ directory written by prepare_data.py")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = dict(ckpt["hyper_parameters"]["model_config"])
    prefix = "model."
    state_dict = {k[len(prefix):]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)}
    other = [k for k in ckpt["state_dict"] if not k.startswith(prefix)]
    if other:
        raise SystemExit(f"unexpected non-decoder weights in the checkpoint: {other[:5]}")
    export(model_config, state_dict, args.artifacts, args.output_dir)


if __name__ == "__main__":
    main()
