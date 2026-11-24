# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Export a SkyRL GRPO checkpoint back into the SFT export layout the published eval reads.

RL trains only a LoRA adapter on the text model and the protein projector; everything else --
embeddings (including the baked GO rows), the GO encoder and projection, ESM3 -- is frozen. So
the exported model is the ORIGINAL SFT export with exactly two changes:

1. each LoRA target weight becomes ``W + (B @ A) * alpha / r``, merged in fp32 and written in the
   export's dtype, under the export's own names (``language_model.model.*``);
2. ``protein_projection.pt`` is the trained projector (fp32 masters, as the export stores it).

Every other file is copied unchanged, so ``eval.py`` loads it through the same
prompt_embeds path, GO encoder and tokenizer that produced the published numbers.

A GRPO checkpoint is per-rank DTensor shards, and a wrong reassembly loads cleanly and scores like
the base model. The shard handling uses numeric rank order, a world size derived from filenames and
DTensor stitching without a process group; the export refuses to write anything it cannot verify:
every LoRA pair must map onto an export weight of the right shape, the target count must match, and
the projector must load strictly.

    python -m bioreason2.rl.export_checkpoint --step_dir <run>/ckpt/global_step_90 \\
        --base <SFT export> --dst <export dir> --lora_alpha 128 --lora_rank 64

or in two stages, the first in the RL environment (the torch that wrote the shards) and the second
wherever the SFT export lives:

    python -m bioreason2.rl.export_checkpoint --step_dir <run>/ckpt/global_step_90 --save_trainable t.pt
    python -m bioreason2.rl.export_checkpoint --trainable t.pt --base <SFT export> --dst <export dir> \\
        --lora_alpha 128 --lora_rank 64
"""

import argparse
import glob
import json
import os
import re
import shutil

import torch
from torch.distributed.tensor import DTensor, Shard
from safetensors import safe_open
from safetensors.torch import save_file

from bioreason2.models.projection_norm import build_protein_projection
from bioreason2.rl.constants import ESM3_DIM, PROJECTION_FILENAME

LORA_A_SUFFIX = ".lora_A.default.weight"
LORA_B_SUFFIX = ".lora_B.default.weight"
PROJ_PREFIX = "_skyrl_modality_projections."
PEFT_PREFIX = "base_model.model."
BASE_LAYER_SUFFIX = ".base_layer.weight"

#: Text-model prefix in the SFT export; RL names the same weights `model.*`.
EXPORT_PREFIX = "language_model."
WEIGHTS_FILE = "model.safetensors"

#: Not carried over: a frozen copy of ESM3 that older conversions wrote. Inference loads ESM3 from
#: the `esm` package and never read it.
SKIPPED = ("protein_model",)

#: q/k/v/o and gate/up/down on every layer that SFT's LoRA targeted, as the runs asserted.
EXPECTED_LORA_TARGETS = 128
NORM_EPS = 1e-12


class ExportError(RuntimeError):
    """Raised when the export cannot be verified."""


def rank_shard_paths(policy_dir: str) -> list[str]:
    """The per-rank shards in numeric rank order, with the world size taken from the filenames.

    Args:
        policy_dir: A ``global_step_N/policy`` directory.

    Returns:
        One path per rank, rank 0 first.

    Raises:
        ExportError: No shards, mixed world sizes, or a missing rank.
    """
    paths = glob.glob(os.path.join(policy_dir, "model_world_size_*_rank_*.pt"))
    if not paths:
        raise ExportError(f"no model_world_size_*_rank_*.pt shards in {policy_dir}")
    sizes = {int(re.search(r"world_size_(\d+)_rank_", p).group(1)) for p in paths}
    if len(sizes) != 1:
        raise ExportError(f"mixed world sizes {sorted(sizes)} in {policy_dir}")
    world = sizes.pop()
    ranks = {int(re.search(r"_rank_(\d+)", p).group(1)): p for p in paths}
    if sorted(ranks) != list(range(world)):
        raise ExportError(f"world size {world} but ranks {sorted(ranks)[:8]}... in {policy_dir}")
    return [ranks[r] for r in range(world)]


def mesh_coords(rank: int, mesh_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Row-major coordinates of a rank in a device mesh, as DeviceMesh lays them out.

    Args:
        rank: Global rank.
        mesh_shape: Mesh shape, e.g. ``(32,)`` or ``(4, 8)``.

    Returns:
        The rank's coordinates.
    """
    coords = []
    for size in reversed(mesh_shape):
        coords.append(rank % size)
        rank //= size
    return tuple(reversed(coords))


def stitch(locals_by_rank: dict[int, torch.Tensor], mesh_shape: tuple[int, ...], placements: tuple,
           global_shape: tuple[int, ...]) -> torch.Tensor:
    """Rebuild a full tensor from per-rank DTensor locals, without a process group.

    Args:
        locals_by_rank: rank -> local shard.
        mesh_shape: Device mesh shape.
        placements: One placement per mesh dimension.
        global_shape: The DTensor's full shape, checked at the end.

    Returns:
        The full tensor.

    Raises:
        ExportError: More than one sharded mesh dimension, missing coordinates, or a wrong shape.
    """
    sharded =[(axis, p.dim) for axis, p in enumerate(placements) if isinstance(p, Shard)]
    if not sharded:
        return next(iter(locals_by_rank.values()))
    if len(sharded) > 1:
        raise ExportError(f"{len(sharded)} sharded mesh dims in {placements}")
    mesh_axis, dim = sharded[0]
    chosen = {}
    for rank, tensor in locals_by_rank.items():
        coords = mesh_coords(rank, mesh_shape)
        if all(c == 0 for axis, c in enumerate(coords) if axis != mesh_axis):
            chosen[coords[mesh_axis]] = tensor
    if sorted(chosen) != list(range(mesh_shape[mesh_axis])):
        raise ExportError(f"shard coords {sorted(chosen)} do not cover the mesh axis {mesh_shape[mesh_axis]}")
    full = torch.cat([chosen[i] for i in range(mesh_shape[mesh_axis])], dim=dim)
    if tuple(full.shape) != tuple(global_shape):
        raise ExportError(f"stitched {tuple(full.shape)} != declared {tuple(global_shape)}")
    return full


def trainable_state(paths: list[str]) -> dict[str, torch.Tensor]:
    """Full LoRA and projector tensors from the rank shards, streamed one rank at a time.

    Args:
        paths: Shards in rank order.

    Returns:
        key -> full tensor, for every LoRA A/B and projector entry.
    """
    plain, parts, meta = {}, {}, {}
    for rank, path in enumerate(paths):
        state = torch.load(path, map_location="cpu", weights_only=False)
        for key, value in state.items():
            if not (LORA_A_SUFFIX in key or LORA_B_SUFFIX in key or key.startswith(PROJ_PREFIX)):
                continue
            if isinstance(value, DTensor):
                parts.setdefault(key, {})[rank] = value.to_local().clone()
                meta.setdefault(key, (tuple(value.device_mesh.shape), value.placements, tuple(value.shape)))
            elif key not in plain:
                plain[key] = value.clone()
        del state
    for key, locals_by_rank in parts.items():
        plain[key] = stitch(locals_by_rank, *meta[key])
    return plain


def lora_pairs(state: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """LoRA (A, B) pairs keyed by the export's weight name.

    Args:
        state: Output of `trainable_state`.

    Returns:
        ``language_model.model.<...>.weight`` -> (A, B).

    Raises:
        ExportError: A pair is missing its A or B half.
    """
    halves: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        if LORA_A_SUFFIX in key:
            halves.setdefault(key.replace(LORA_A_SUFFIX, ".weight"), {})["A"] = value
        elif LORA_B_SUFFIX in key:
            halves.setdefault(key.replace(LORA_B_SUFFIX, ".weight"), {})["B"] = value
    broken = [k for k, v in halves.items() if set(v) != {"A", "B"}]
    if broken:
        raise ExportError(f"{len(broken)} LoRA pair(s) missing a half, e.g. {broken[:3]}")
    pairs = {}
    for key, v in halves.items():
        name = key.removeprefix(PEFT_PREFIX).replace(BASE_LAYER_SUFFIX, ".weight")
        pairs[EXPORT_PREFIX + name] = (v["A"], v["B"])
    return pairs


def projector_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The trained projector under the keys `protein_projection.pt` uses (`0.weight`, ...).

    Args:
        state: Output of `trainable_state`.

    Returns:
        name -> tensor.
    """
    return {re.sub(rf"^{re.escape(PROJ_PREFIX)}[^.]+\.", "", k): v for k, v in state.items() if k.startswith(PROJ_PREFIX)}


def export(step_dir: str, base: str, dst: str, scaling: float, expected_targets: int,
           trainable: str = "") -> None:
    """Write the merged export.

    Args:
        step_dir: ``global_step_N`` directory of the RL run.
        base: The SFT export RL started from.
        dst: Destination; must not exist.
        scaling: LoRA ``alpha / r``.
        expected_targets: LoRA target count the run asserted.
        trainable: A `--save_trainable` file to merge instead of reading the step's rank shards.

    Raises:
        FileExistsError: `dst` exists.
        ExportError: Any verification fails.
    """
    if os.path.exists(dst):
        raise FileExistsError(f"{dst} exists; refusing to overwrite")
    state = (torch.load(trainable, map_location="cpu", weights_only=True) if trainable
             else trainable_state(rank_shard_paths(os.path.join(step_dir, "policy"))))
    pairs = lora_pairs(state)
    projector = projector_state(state)
    if len(pairs) != expected_targets:
        raise ExportError(f"{len(pairs)} LoRA pairs, the run asserted {expected_targets}")

    with safe_open(os.path.join(base, WEIGHTS_FILE), framework="pt") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    missing = [k for k in pairs if k not in tensors]
    if missing:
        raise ExportError(f"{len(missing)} LoRA targets not in the export, e.g. {missing[:3]}")

    moved = []
    for key, (a, b) in pairs.items():
        weight = tensors[key]
        delta = (b.float() @ a.float()) * scaling
        if delta.shape != weight.shape:
            raise ExportError(f"{key}: delta {tuple(delta.shape)} vs weight {tuple(weight.shape)}")
        tensors[key] = (weight.float() + delta).to(weight.dtype)
        moved.append(float(delta.norm()))
    if max(moved) == 0.0:
        raise ExportError("every LoRA delta is zero; this is the untrained adapter")

    with open(os.path.join(base, "config.json")) as handle:
        text_config = json.load(handle).get("text_config", {})
    check = build_protein_projection(ESM3_DIM, text_config["hidden_size"],
                                     *norm_settings(base))
    reference = torch.load(os.path.join(base, PROJECTION_FILENAME), map_location="cpu", weights_only=True)
    check.load_state_dict({k: v.to(reference[k].dtype) for k, v in projector.items()}, strict=True)
    projector = {k: v.to(reference[k].dtype).contiguous() for k, v in projector.items()}
    shift = {k: float((projector[k].float() - reference[k].float()).norm() / (reference[k].float().norm() + NORM_EPS))
             for k in projector}

    os.makedirs(dst)
    for name in sorted(os.listdir(base)):
        source = os.path.join(base, name)
        if name in (WEIGHTS_FILE, PROJECTION_FILENAME) or name in SKIPPED:
            continue
        if os.path.isdir(source):
            shutil.copytree(source, os.path.join(dst, name))
        else:
            shutil.copy2(source, os.path.join(dst, name))
    save_file(tensors, os.path.join(dst, WEIGHTS_FILE), metadata={"format": "pt"})
    torch.save(projector, os.path.join(dst, PROJECTION_FILENAME))

    summary = {
        "step_dir": step_dir, "trainable": trainable, "base": base, "scaling": scaling, "lora_targets": len(pairs),
        "lora_delta_norm_mean": sum(moved) / len(moved), "lora_delta_norm_max": max(moved),
        "projector_relative_change": shift,
    }
    with open(os.path.join(dst, "rl_export.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


def norm_settings(base: str) -> tuple[str, float]:
    """The projection-norm variant of an SFT export, read from its projector's keys.

    `blocknorm` stores one scalar `3.weight`; `none` has no tail. Other variants are refused
    rather than guessed.

    Args:
        base: SFT export directory.

    Returns:
        (variant, target).

    Raises:
        ExportError: The tail is not one this export knows.
    """
    keys = torch.load(os.path.join(base, PROJECTION_FILENAME), map_location="cpu", weights_only=True)
    tail = {k: v for k, v in keys.items() if k.startswith("3.")}
    if not tail:
        return "none", 0.0
    if set(tail) == {"3.weight"} and tail["3.weight"].dim() == 0:
        return "blocknorm", 0.0
    raise ExportError(f"unrecognised projector tail {sorted(tail)}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step_dir", default="")
    parser.add_argument("--base", default="", help="the SFT export the RL checkpoint was prepared from")
    parser.add_argument("--dst", default="")
    parser.add_argument("--lora_alpha", type=float, default=0.0)
    parser.add_argument("--lora_rank", type=float, default=0.0)
    parser.add_argument("--expected_targets", type=int, default=EXPECTED_LORA_TARGETS)
    # Two stages for checkpoints on another filesystem than the base: write only the trained tensors
    # (LoRA pairs and projector, ~2% of the shards) where the checkpoint is, move that file, merge there.
    parser.add_argument("--save_trainable", default="", help="write the stitched LoRA + projector tensors and stop")
    parser.add_argument("--trainable", default="", help="merge a --save_trainable file instead of --step_dir")
    args = parser.parse_args()
    if args.save_trainable:
        state = trainable_state(rank_shard_paths(os.path.join(args.step_dir, "policy")))
        torch.save(state, args.save_trainable)
        print(f"wrote {len(state)} trained tensors -> {args.save_trainable}")
        return
    if not (args.base and args.dst and args.lora_alpha and args.lora_rank and (args.step_dir or args.trainable)):
        parser.error("a merge needs --base, --dst, --lora_alpha, --lora_rank and --step_dir or --trainable")
    export(args.step_dir, args.base, args.dst, args.lora_alpha / args.lora_rank, args.expected_targets,
           args.trainable)


if __name__ == "__main__":
    main()
