# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Turn a BioReason-Pro 2 SFT export into a self-describing, text-only RL checkpoint.

The export is read by three consumers under RL -- SkyRL's dataset planner, the FSDP training
model and the vLLM engine -- and every convention it leaves implicit fails silently in one of
them. This folds those conventions into the checkpoint:

1. **Layout.** The export declares `Qwen3_5ForConditionalGeneration` but stores the text model
   as `language_model.model.*` with no vision tower. That is the legacy layout that silently
   loaded a randomly initialised training model in the first SkyRL multimodal port. Weights are
   rewritten to the text-only `model.*` / `lm_head.weight` layout that `Qwen3_5ForCausalLM`
   expects in both transformers and vLLM.
2. **GO block.** With the unified GO encoder, `go_projection(go_encoder())` takes no
   protein input, so frozen it is one constant `[200, H]` block. It is computed once, exactly
   as eval computes it (bf16, on the GPU), and written into 200 unused padding rows of
   `embed_tokens` behind 200 new tokens. Prompts then carry those tokens instead of 200
   `<|go_graph_pad|>`, and the GO span needs no plumbing in vLLM or the trainer.
3. **Output head.** A tied model (the 4B) is untied first, with `lm_head` a copy of the
   embedding table *before* the GO rows are written, so the output side of every row stays
   what SFT trained against.
4. **Config.** `text_config` is hoisted to the top level as `qwen3_5_text`, the architecture
   points at the multimodal RL class, MTP is zeroed (speculative decoding would corrupt the
   on-policy logprobs), M-RoPE is dropped as eval already does, and the projector's shape is
   recorded so every consumer builds the same one.
5. **Tokenizer and generation config.** The GO tokens are registered and the SFT chat template
   saved onto the tokenizer, or SkyRL silently applies the stock Qwen3.5 template. Sampling
   defaults are stripped from `generation_config.json` so nothing truncates the distribution
   behind the trainer, and eos is `<|im_end|>`, the token SFT eval stops on.

Run in the eval environment (transformers 5.x plus the GO encoder's deps)::

    python -m bioreason2.rl.prepare_checkpoint \\
        --src <SFT export directory> --dst <RL checkpoint directory> \\
        --protein_projection_norm blocknorm --protein_projection_norm_target 0.0133
"""

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from bioreason2 import paths
from bioreason2.models import projection_norm
from bioreason2.models.go_graph_encoder import create_go_graph_encoder_pipeline
from bioreason2.models.pl.chat_template_pl import get_chat_template
from bioreason2.rl.constants import (
    ARCHITECTURE,
    ESM3_DIM,
    FIRST_GO_ID,
    GO_GRAPH_PAD_ID,
    GO_GRAPH_PAD_TOKEN,
    GO_TOKENS,
    NUM_GO_TOKENS,
    PROJECTION_FILENAME,
    PROTEIN_PAD_ID,
    PROTEIN_PAD_TOKEN,
    VOCAB_SIZE,
)

#: The export's text-model prefix, and the prefixes that must not reach a text-only model.
LEGACY_PREFIX = "language_model."
DROPPED_PREFIXES = ("visual.", "model.visual.", "mtp.")

EMBED_KEY = "model.embed_tokens.weight"
LM_HEAD_KEY = "lm_head.weight"

#: GO encoder shape, identical to `scripts/eval/eval.sh`; a mismatch fails the strict load.
GO_HIDDEN_DIM = 512
GO_NUM_GAT_LAYERS = 3
GO_NUM_HEADS = 8
GO_EMBEDDING_DIM = 2560

#: Cached GO encoder output that eval reads in place of running the encoder.
GO_EMBEDDING_CACHE = "go_embedding.pt"
GO_ENCODER_FILE = "go_encoder.pt"
GO_PROJECTION_FILE = "go_projection.pt"

#: Kept from the top level of the source config when `text_config` lacks them.
TOP_LEVEL_KEPT = ("dtype", "torch_dtype")

END_OF_TURN = "<|im_end|>"

#: `generation_config.json` fields vLLM turns into default sampling parameters.
SAMPLING_FIELDS = (
    "do_sample", "temperature", "top_p", "top_k", "min_p", "typical_p",
    "repetition_penalty", "presence_penalty", "frequency_penalty", "no_repeat_ngram_size",
)

#: `rope_parameters` keys that turn on M-RoPE, which eval already strips.
MROPE_KEYS = ("mrope_section", "mrope_interleaved")

WEIGHTS_FILE = "model.safetensors"
WEIGHTS_INDEX = "model.safetensors.index.json"

#: The baked GO block, kept beside the weights for the parity checks.
GO_ROWS_FILE = "go_rows.pt"


def load_tensors(src: str) -> dict[str, torch.Tensor]:
    """Read every tensor of a single-file or sharded safetensors checkpoint.

    Args:
        src: Checkpoint directory.

    Returns:
        Name -> tensor, on the CPU.

    Raises:
        FileNotFoundError: The directory holds no safetensors weights.
    """
    index_path = os.path.join(src, WEIGHTS_INDEX)
    if os.path.exists(index_path):
        with open(index_path) as handle:
            shards = sorted(set(json.load(handle)["weight_map"].values()))
    elif os.path.exists(os.path.join(src, WEIGHTS_FILE)):
        shards = [WEIGHTS_FILE]
    else:
        raise FileNotFoundError(f"no {WEIGHTS_FILE} or {WEIGHTS_INDEX} in {src}")

    tensors = {}
    for shard in shards:
        with safe_open(os.path.join(src, shard), framework="pt") as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)
    return tensors


def compute_go_rows(src: str, hidden_size: int, go_obo_path: str, go_embeddings_path: str,
                    device: str) -> torch.Tensor:
    """The projected GO block, computed the way `protein_vllm.ProteinLLMModel` computes it at eval.

    Same modules, same strict loads, same order (fp32 load, then bf16 on the device), so the
    baked rows are the vectors the published eval substituted for `<|go_graph_pad|>`. Eval
    prefers a cached encoder output (`go_embedding.pt`) over running the encoder, so this does
    too.

    Args:
        src: Export directory holding `go_projection.pt` and either `go_embedding.pt` or
            `go_encoder.pt`.
        hidden_size: LLM hidden size.
        go_obo_path: GO ontology file.
        go_embeddings_path: Directory of precomputed GO term embeddings.
        device: Device to run the encoder on.

    Returns:
        A `(NUM_GO_TOKENS, hidden_size)` bf16 tensor on the CPU.

    Raises:
        RuntimeError: The encoder does not return `NUM_GO_TOKENS` rows.
    """
    cached_path = os.path.join(src, GO_EMBEDDING_CACHE)
    if os.path.exists(cached_path):
        reduced = torch.load(cached_path, map_location=device, weights_only=True).to(device=device, dtype=torch.bfloat16)
        print(f"[prep] GO encoder output read from {cached_path}, as eval does")
    else:
        encoder = create_go_graph_encoder_pipeline(
            go_obo_path=go_obo_path,
            precomputed_embeddings_path=go_embeddings_path,
            hidden_dim=GO_HIDDEN_DIM,
            num_gat_layers=GO_NUM_GAT_LAYERS,
            num_heads=GO_NUM_HEADS,
            num_reduced_embeddings=NUM_GO_TOKENS,
            embedding_dim=GO_EMBEDDING_DIM,
            embeddings_load_to=device,
        )
        encoder.load_state_dict(torch.load(os.path.join(src, GO_ENCODER_FILE), map_location=device, weights_only=True),
                                strict=True)
        encoder = encoder.to(device=device, dtype=torch.bfloat16).eval()
        with torch.no_grad():
            reduced = encoder().to(device=device, dtype=torch.bfloat16)

    projection = nn.Sequential(
        nn.Linear(GO_EMBEDDING_DIM, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
    ).to(device)
    projection.load_state_dict(torch.load(os.path.join(src, GO_PROJECTION_FILE), map_location=device, weights_only=True),
                               strict=True)
    projection = projection.to(device=device, dtype=torch.bfloat16).eval()

    with torch.no_grad():
        rows = projection(reduced)

    if tuple(rows.shape) != (NUM_GO_TOKENS, hidden_size):
        raise RuntimeError(f"GO block has shape {tuple(rows.shape)}, expected {(NUM_GO_TOKENS, hidden_size)}")
    return rows.cpu()


def check_projection(src: str, hidden_size: int, norm: str, target: float) -> None:
    """Load the projector sidecar strictly into the shape the RL consumers will build.

    Args:
        src: Export directory holding `protein_projection.pt`.
        hidden_size: LLM hidden size.
        norm: Projection-norm variant the checkpoint was trained with.
        target: Its initial target RMS.

    Raises:
        RuntimeError: The sidecar does not match, i.e. the norm flags disagree with training.
    """
    module = projection_norm.build_protein_projection(ESM3_DIM, hidden_size, norm, target)
    state = torch.load(os.path.join(src, PROJECTION_FILENAME), map_location="cpu", weights_only=True)
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{PROJECTION_FILENAME} does not match --protein_projection_norm {norm!r}: "
            f"missing={list(missing)} unexpected={list(unexpected)}"
        )


def prepare_tokenizer(src: str, dst: str) -> AutoTokenizer:
    """Register the GO tokens, attach the SFT chat template, save, and verify the round trip.

    Args:
        src: Export directory.
        dst: Destination directory.

    Returns:
        The saved tokenizer, reloaded from `dst`.

    Raises:
        ValueError: An id differs from the SFT assignment, the GO tokens would not fit below
            `VOCAB_SIZE`, or the export's own chat template differs from the SFT one.
    """
    tokenizer = AutoTokenizer.from_pretrained(src)
    for token, expected in ((PROTEIN_PAD_TOKEN, PROTEIN_PAD_ID), (GO_GRAPH_PAD_TOKEN, GO_GRAPH_PAD_ID)):
        if tokenizer.convert_tokens_to_ids(token) != expected:
            raise ValueError(f"{token!r} has id {tokenizer.convert_tokens_to_ids(token)}, SFT used {expected}")
    if len(tokenizer) != FIRST_GO_ID:
        raise ValueError(f"tokenizer has {len(tokenizer)} entries; the GO rows assume {FIRST_GO_ID}")

    template = get_chat_template("")
    if tokenizer.chat_template and tokenizer.chat_template != template:
        raise ValueError("the export's chat template differs from bioreason2's SFT template")

    tokenizer.add_tokens(GO_TOKENS, special_tokens=True)
    tokenizer.chat_template = template
    tokenizer.save_pretrained(dst)

    reloaded = AutoTokenizer.from_pretrained(dst)
    ids = reloaded.convert_tokens_to_ids(GO_TOKENS)
    if ids != list(range(FIRST_GO_ID, FIRST_GO_ID + NUM_GO_TOKENS)) or max(ids) >= VOCAB_SIZE:
        raise ValueError(f"GO tokens landed on ids {ids[0]}..{ids[-1]}, expected {FIRST_GO_ID}.. below {VOCAB_SIZE}")
    if reloaded.chat_template != template:
        raise ValueError("the chat template did not survive save_pretrained")
    if reloaded.eos_token != END_OF_TURN:
        raise ValueError(f"tokenizer eos is {reloaded.eos_token!r}, SFT eval stops on {END_OF_TURN!r}")
    return reloaded


def rewrite_weights(tensors: dict[str, torch.Tensor], go_rows: torch.Tensor) -> dict[str, torch.Tensor]:
    """Move to the text-only layout, untie the head if needed, and write the GO rows.

    Args:
        tensors: The export's tensors.
        go_rows: `(NUM_GO_TOKENS, H)` GO block.

    Returns:
        The rewritten tensors.

    Raises:
        ValueError: An unexpected key survives, or the embedding table has the wrong height.
    """
    out = {}
    dropped = 0
    for key, value in tensors.items():
        if key.startswith(DROPPED_PREFIXES):
            dropped += 1
            continue
        out[key.removeprefix(LEGACY_PREFIX)] = value

    stray = [k for k in out if not (k.startswith("model.") or k == LM_HEAD_KEY)]
    if stray:
        raise ValueError(f"keys outside the text-only layout: {stray[:6]}")

    embed = out[EMBED_KEY]
    if embed.shape[0] != VOCAB_SIZE:
        raise ValueError(f"{EMBED_KEY} has {embed.shape[0]} rows, expected {VOCAB_SIZE}")

    if LM_HEAD_KEY not in out:
        out[LM_HEAD_KEY] = embed.clone()
        print(f"[prep] untied: {LM_HEAD_KEY} = copy of {EMBED_KEY}, taken before the GO rows")

    embed = embed.clone()
    embed[FIRST_GO_ID:FIRST_GO_ID + NUM_GO_TOKENS] = go_rows.to(embed.dtype)
    out[EMBED_KEY] = embed
    print(f"[prep] {len(out)} tensors, {dropped} dropped; GO rows written at {FIRST_GO_ID}..{FIRST_GO_ID + NUM_GO_TOKENS - 1}")
    return out


def rewrite_config(src: str, dst: str, norm: str, target: float, eos_id: int) -> int:
    """Write the text-only config the RL consumers read.

    Args:
        src: Export directory.
        dst: Destination directory.
        norm: Projection-norm variant.
        target: Its initial target RMS.
        eos_id: Id of the end-of-turn token.

    Returns:
        The LLM hidden size.
    """
    with open(os.path.join(src, "config.json")) as handle:
        source = json.load(handle)

    config = dict(source.get("text_config", source))
    for key in TOP_LEVEL_KEPT:
        if key in source and key not in config:
            config[key] = source[key]
    dropped = sorted(k for k in source if k not in config and k != "text_config")
    print(f"[prep] config: hoisted text_config; top-level keys dropped: {dropped}")

    config["architectures"] = [ARCHITECTURE]
    config["model_type"] = "qwen3_5_text"
    config["tie_word_embeddings"] = False
    config["mtp_num_hidden_layers"] = 0
    config["eos_token_id"] = eos_id
    config["rope_parameters"] = {k: v for k, v in config["rope_parameters"].items() if k not in MROPE_KEYS}
    config["protein_embedding_dim"] = ESM3_DIM
    config["protein_projection_norm"] = norm
    config["protein_projection_norm_target"] = target
    config["transformers_version"] = source.get("transformers_version", config.get("transformers_version"))

    with open(os.path.join(dst, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    return int(config["hidden_size"])


def rewrite_generation_config(src: str, dst: str, eos_ids: list[int]) -> None:
    """Strip sampling defaults and set eos, so vLLM samples exactly what SkyRL asks for.

    Args:
        src: Export directory.
        dst: Destination directory.
        eos_ids: Ids generation must stop on, end-of-turn first.
    """
    path = os.path.join(src, "generation_config.json")
    config = {}
    if os.path.exists(path):
        with open(path) as handle:
            config = json.load(handle)
    stripped = [k for k in SAMPLING_FIELDS if config.pop(k, None) is not None]
    config["eos_token_id"] = eos_ids
    with open(os.path.join(dst, "generation_config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    print(f"[prep] generation_config: eos {eos_ids}, stripped {stripped or 'nothing'}")


def prepare(src: str, dst: str, norm: str, target: float, go_obo_path: str, go_embeddings_path: str,
            device: str) -> None:
    """Build the RL checkpoint at `dst` from the export at `src`.

    Args:
        src: Exported HF checkpoint directory.
        dst: Destination; must not exist yet.
        norm: Projection-norm variant the checkpoint was trained with.
        target: Its initial target RMS.
        go_obo_path: GO ontology file.
        go_embeddings_path: Directory of precomputed GO term embeddings.
        device: Device for the GO encoder.

    Raises:
        FileExistsError: `dst` already exists.
    """
    if os.path.exists(dst):
        raise FileExistsError(f"{dst} exists; refusing to overwrite")
    os.makedirs(dst)

    tokenizer = prepare_tokenizer(src, dst)
    eos_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    hidden_size = rewrite_config(src, dst, norm, target, eos_id)
    check_projection(src, hidden_size, norm, target)

    go_rows = compute_go_rows(src, hidden_size, go_obo_path, go_embeddings_path, device)
    print(f"[prep] GO block {tuple(go_rows.shape)}, row RMS {go_rows.float().pow(2).mean(-1).sqrt().mean():.4f}")

    tensors = rewrite_weights(load_tensors(src), go_rows)
    save_file(tensors, os.path.join(dst, WEIGHTS_FILE), metadata={"format": "pt"})

    with open(os.path.join(src, "generation_config.json")) as handle:
        source_eos = json.load(handle).get("eos_token_id")
    extra = [i for i in (source_eos if isinstance(source_eos, list) else [source_eos]) if i not in (None, eos_id)]
    rewrite_generation_config(src, dst, [eos_id] + extra)

    shutil.copy2(os.path.join(src, PROJECTION_FILENAME), os.path.join(dst, PROJECTION_FILENAME))
    torch.save(go_rows, os.path.join(dst, GO_ROWS_FILE))
    check(src, dst)
    print(f"[prep] done -> {dst}")


def check(src: str, dst: str) -> None:
    """Reload the written checkpoint the way the trainer does and verify every edit.

    The CPU half of parity check 2: if transformers loads the result with nothing missing or
    unexpected, the GO rows and head are exactly what was intended, and the config says what
    the vLLM class reads, the remaining risk is numerical rather than structural.

    Args:
        src: Export directory the checkpoint was built from.
        dst: The prepared checkpoint.

    Raises:
        RuntimeError: Any check fails.
    """
    model, info = AutoModelForCausalLM.from_pretrained(dst, dtype=torch.bfloat16, output_loading_info=True)
    if info["missing_keys"] or info["unexpected_keys"]:
        raise RuntimeError(f"load of {dst}: missing={info['missing_keys'][:6]} unexpected={info['unexpected_keys'][:6]}")

    embed = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    if embed.data_ptr() == head.data_ptr():
        raise RuntimeError("lm_head is still tied to embed_tokens")

    go_rows = torch.load(os.path.join(dst, GO_ROWS_FILE), map_location="cpu")
    if not torch.equal(embed[FIRST_GO_ID:FIRST_GO_ID + NUM_GO_TOKENS], go_rows.to(embed.dtype)):
        raise RuntimeError("embed_tokens GO rows differ from go_rows.pt")

    source = load_tensors(src)
    source_head = source.get(LM_HEAD_KEY, source[LEGACY_PREFIX + EMBED_KEY])
    if not torch.equal(head, source_head.to(head.dtype)):
        raise RuntimeError("lm_head differs from the source's output head")
    untouched = torch.ones(embed.shape[0], dtype=torch.bool)
    untouched[FIRST_GO_ID:FIRST_GO_ID + NUM_GO_TOKENS] = False
    if not torch.equal(embed[untouched], source[LEGACY_PREFIX + EMBED_KEY][untouched].to(embed.dtype)):
        raise RuntimeError("embed_tokens rows outside the GO block changed")

    config = model.config
    expected = {"architectures": [ARCHITECTURE], "vocab_size": VOCAB_SIZE, "tie_word_embeddings": False,
                "mtp_num_hidden_layers": 0}
    wrong = {k: getattr(config, k, None) for k, v in expected.items() if getattr(config, k, None) != v}
    if wrong:
        raise RuntimeError(f"config fields differ from what the RL consumers expect: {wrong}")
    print(f"[check] {dst}: loads clean, GO rows and head exact, config {sorted(expected)} as expected")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="exported HF checkpoint directory")
    parser.add_argument("--dst", required=True, help="destination for the RL checkpoint")
    parser.add_argument("--protein_projection_norm", required=True, choices=sorted(projection_norm.VARIANTS))
    parser.add_argument("--protein_projection_norm_target", type=float, default=0.0)
    parser.add_argument("--go_obo_path", default=paths.GO_OBO)
    parser.add_argument("--go_embeddings_path", default=paths.GO_EMBEDDINGS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    prepare(args.src, args.dst, args.protein_projection_norm, args.protein_projection_norm_target,
            args.go_obo_path, args.go_embeddings_path, args.device)


if __name__ == "__main__":
    main()
