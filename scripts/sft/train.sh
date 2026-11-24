#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Train a BioReason-Pro 2 checkpoint: stage 1, then stage 2 warm-started from it.
#
#   bash scripts/sft/train.sh
#
# The released SFT model's recipe. Every path is an environment variable or repository-relative.

# ===================================================================================================
# SLURM Configuration
# Adjust these to match your cluster. The published runs used 4 nodes x 8 GPUs; any GPU count
# that divides 32 trains the same effective batch.
# ===================================================================================================
# #SBATCH --job-name=train_bioreason_pro
# #SBATCH --partition=your_gpu_partition
# #SBATCH --time=48:00:00
# #SBATCH --nodes=4
# #SBATCH --gpus-per-node=8
# #SBATCH --ntasks-per-node=8
# #SBATCH --cpus-per-task=16
# #SBATCH --mem=0
# #SBATCH --output=train_bioreason_pro_%j.out
# #SBATCH --error=train_bioreason_pro_%j.err

set -euo pipefail


# ===================================================================================================
# Environment Setup
# Activate the brp-train environment first (README, Setup).
# ===================================================================================================
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Triton 3.3 reads the CUDA version of the first ptxas on PATH and rejects 13.x (a CUDA 13
# toolkit, e.g. nvcr.io/nvidia/pytorch:26.02), so it is pointed at the ptxas it ships with.
_TRITON_PTXAS="$(python -c 'import os, triton; print(os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas"))' 2>/dev/null || true)"
if [ -z "${TRITON_PTXAS_PATH:-}" ] && [ -x "$_TRITON_PTXAS" ]; then export TRITON_PTXAS_PATH="$_TRITON_PTXAS"; fi

# Without an entity, run W&B offline rather than fall through to a code default.
WANDB_PROJECT="${WANDB_PROJECT:-bioreason-pro-finetune}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
if [ -z "$WANDB_ENTITY" ]; then export WANDB_MODE="${WANDB_MODE:-offline}"; fi


# ===================================================================================================
# Model and Data
# ===================================================================================================
BACKBONE="${BACKBONE:-Qwen/Qwen3.5-9B}"                    # or Qwen/Qwen3.5-4B
SFT_TABLE="${SFT_TABLE:-$REPO_ROOT/data/sft/v2-nofeatures}" # README, Evaluate on the benchmark

# The released model's protein projection: one learnable scale over the residue block.
PROTEIN_PROJECTION_NORM="${PROTEIN_PROJECTION_NORM:-blocknorm}"
PROTEIN_PROJECTION_NORM_TARGET="${PROTEIN_PROJECTION_NORM_TARGET:-0.0133}"

RUN_NAME="${RUN_NAME:-$(basename "$BACKBONE" | tr 'A-Z' 'a-z')-$PROTEIN_PROJECTION_NORM}"


# ===================================================================================================
# Paths
# ===================================================================================================
CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/ckpt/train}"
CACHE_DIR="${CACHE_DIR:-${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}}"   # the Hub's model cache
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-$REPO_ROOT/cache/datasets}"
GO_EMBEDDINGS_PATH="${GO_EMBEDDINGS_PATH:-${BRP_ASSETS_DIR:-$REPO_ROOT/assets}/go_embeddings}"
STRUCTURE_DIR="${STRUCTURE_DIR:-${BRP_ASSETS_DIR:-$REPO_ROOT/assets}/structures}"   # must be FLAT
GO_OBO_PATH="$REPO_ROOT/bioreason2/dataset/go-basic.obo"
QUESTION_TEMPLATE_PY="$REPO_ROOT/scripts/data/question_template.py"

STAGE1_DIR="$CKPT_ROOT/$RUN_NAME-stage1"
STAGE2_DIR="$CKPT_ROOT/$RUN_NAME"


# ===================================================================================================
# Recipe
# ===================================================================================================
MAX_LENGTH_TEXT="${MAX_LENGTH_TEXT:-16384}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-32}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
MAX_STEPS="${MAX_STEPS:-0}"          # > 0 stops each stage early, for a smoke run
CHECKPOINT_START_EPOCH="${CHECKPOINT_START_EPOCH:-0}"   # save every epoch from this one, as published
DEBUG="${DEBUG:-False}"              # True loads 50 rows per split

# Hold the effective batch at 32 whatever the GPU count. Under SLURM srun starts one task
# per GPU; on one machine Lightning starts one process per local GPU.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    NUM_NODES="${SLURM_NNODES:-1}"
    GPUS_PER_NODE="${SLURM_NTASKS_PER_NODE:-$(nvidia-smi -L | wc -l)}"
    LAUNCHER="srun"
else
    NUM_NODES=1
    # Lightning trains on the visible GPUs, so count those, not every GPU in the machine.
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        GPUS_PER_NODE="${GPUS_PER_NODE:-$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)}"
    else
        GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}"
    fi
    LAUNCHER=""
fi
GPUS=$((NUM_NODES * GPUS_PER_NODE))
if [ $((EFFECTIVE_BATCH % GPUS)) -ne 0 ]; then
    echo "Error: EFFECTIVE_BATCH=$EFFECTIVE_BATCH is not a multiple of $GPUS GPUs" >&2
    exit 1
fi
GRAD_ACCUM=$((EFFECTIVE_BATCH / GPUS))


# ===================================================================================================
# Preflight: fail here, not an hour into model loading
# ===================================================================================================
if [ ! -d "$SFT_TABLE/data" ]; then
    echo "Error: no prompt table at $SFT_TABLE (expected $SFT_TABLE/data/*.parquet)" >&2
    exit 1
fi
if [ -z "$(ls -A "$GO_EMBEDDINGS_PATH" 2>/dev/null)" ]; then
    echo "Error: no GO embeddings at $GO_EMBEDDINGS_PATH (scripts/data/download_assets.py)" >&2
    exit 1
fi
if [ ! -d "$STRUCTURE_DIR" ]; then
    echo "Error: no structures at $STRUCTURE_DIR; set STRUCTURE_DIR" >&2
    exit 1
fi
mkdir -p "$STAGE1_DIR" "$STAGE2_DIR" "$DATASET_CACHE_DIR"

echo "### $BACKBONE  table=$SFT_TABLE  projection=$PROTEIN_PROJECTION_NORM/$PROTEIN_PROJECTION_NORM_TARGET"
echo "### $NUM_NODES node(s) x $GPUS_PER_NODE GPU(s) x batch 1 x accum $GRAD_ACCUM = $EFFECTIVE_BATCH"

# Build the dataset cache once before srun starts every rank: ranks mapping the table
# concurrently into one cache directory race on its arrow shards.
if [ -n "$LAUNCHER" ]; then
    echo "### warming the dataset cache in $DATASET_CACHE_DIR"
    python -c "from bioreason2.dataset.cafa5.load import load_cafa5_dataset; load_cafa5_dataset(
        dataset='$SFT_TABLE', config='default', cache_dir='$DATASET_CACHE_DIR', max_length=2000,
        return_as_chat_template=True, structure_dir='$STRUCTURE_DIR', debug=$DEBUG)"
fi


# ===================================================================================================
# Shared Arguments
# DDP runs without the per-step buffer broadcast: ESM3 builds its structure codebook lazily, so
# ranks disagree on their buffer set until each has seen a structure, and the broadcast
# deadlocks. The buffers are constants; dropping the sync changes nothing else.
# ===================================================================================================
COMMON_ARGS=(
    --wandb_project "$WANDB_PROJECT" ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"}
    --cache_dir "$CACHE_DIR" --dataset_cache_dir "$DATASET_CACHE_DIR"
    --text_model_name "$BACKBONE" --protein_model_name esm3_sm_open_v1
    --strategy ddp_no_buffer_sync --use_qlora False
    --num_gpus -1 --num_nodes "$NUM_NODES" --batch_size 1 --gradient_accumulation_steps "$GRAD_ACCUM"
    --cafa5_dataset "$SFT_TABLE" --cafa5_dataset_name default
    --structure_dir "$STRUCTURE_DIR"
    --max_length_protein 2000 --max_length_text "$MAX_LENGTH_TEXT"
    --lora_rank 128 --lora_alpha 256 --lora_dropout 0
    --protein_embedding_layer 37 --go_model_finetune True
    --attn_implementation flash_attention_2
    --go_obo_path "$GO_OBO_PATH" --precomputed_embeddings_path "$GO_EMBEDDINGS_PATH"
    --go_hidden_dim 512 --go_num_gat_layers 3 --go_num_heads 8
    --go_num_reduced_embeddings 200 --go_embedding_dim 2560
    --protein_projection_norm "$PROTEIN_PROJECTION_NORM"
    --protein_projection_norm_target "$PROTEIN_PROJECTION_NORM_TARGET"
    --return_answer_in_batch False --num_workers 8 --weight_decay 0.01 --seed 23 --save_top_k 2
    --checkpoint_start_epoch "$CHECKPOINT_START_EPOCH"
    --log_every_n_steps 200
    --gradient_checkpointing True --val_check_interval 1.0
    --p_drop_ppi 0.8 --p_ppi_complete 0.0 --p_drop_ppi_item 0.5
    --p_drop_features 0.0 --p_features_complete 0.0 --p_drop_features_item 0.0
    --p_drop_gogpt 0.0 --p_drop_structure 0.5
    --question_template_py "$QUESTION_TEMPLATE_PY"
    --learning_rate 1e-4 --warmup_ratio 0.05 --every_n_train_steps 10000000000
    --max_steps "$MAX_STEPS" --debug "$DEBUG"
)


# ===================================================================================================
# Stage 1: projectors and GO encoder, the LLM frozen
# ===================================================================================================
echo "### stage 1"
$LAUNCHER python train_protein_llm.py "${COMMON_ARGS[@]}" \
    --run_name "$RUN_NAME-stage1" --checkpoint_dir "$STAGE1_DIR" \
    --training_stage 1 --text_model_finetune False --max_epochs "$STAGE1_EPOCHS"

for f in projector_weights.pt go_projection_weights.pt go_encoder_weights.pt; do
    if [ ! -s "$STAGE1_DIR/$f" ]; then
        echo "Error: stage 1 did not write $STAGE1_DIR/$f" >&2
        exit 1
    fi
done


# ===================================================================================================
# Stage 2: LoRA plus the same three components, warm-started from stage 1
# ===================================================================================================
echo "### stage 2, warm-starting from $STAGE1_DIR"
$LAUNCHER python train_protein_llm.py "${COMMON_ARGS[@]}" \
    --run_name "$RUN_NAME-stage2" --checkpoint_dir "$STAGE2_DIR" \
    --training_stage 2 --text_model_finetune True --max_epochs "$MAX_EPOCHS" \
    --projector_checkpoint_path "$STAGE1_DIR/projector_weights.pt" \
    --go_projection_checkpoint_path "$STAGE1_DIR/go_projection_weights.pt" \
    --go_encoder_checkpoint_path "$STAGE1_DIR/go_encoder_weights.pt"

echo "### done: epoch checkpoints in $STAGE2_DIR ($RUN_NAME-stage2-epoch=NN.ckpt)."
echo "### Convert one for inference with bioreason2/utils/save_hf_ckpt.py (README, Train)."
