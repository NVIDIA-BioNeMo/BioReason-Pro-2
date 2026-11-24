#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Generate with a BioReason-Pro 2 checkpoint on the benchmark (the corpus test split), then score it.
#
#   CKPT_DIR=ckpt/rl bash scripts/eval/eval.sh                                      # without InterPro
#   CKPT_DIR=ckpt/rl EVAL_TABLE=data/sft/v2-features bash scripts/eval/eval.sh      # with InterPro
#
# Every path is an environment variable or repository-relative.

# ===================================================================================================
# SLURM Configuration
# Adjust these to match your cluster. One GPU per process; shard with NUM_CHUNKS / CHUNK_ID.
# ===================================================================================================
# #SBATCH --job-name=eval_bioreason_pro
# #SBATCH --partition=your_gpu_partition
# #SBATCH --time=24:00:00
# #SBATCH --nodes=1
# #SBATCH --gpus=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=16
# #SBATCH --mem=128gb
# #SBATCH --output=eval_bioreason_pro_%j.out
# #SBATCH --error=eval_bioreason_pro_%j.err

set -euo pipefail


# ===================================================================================================
# Environment Setup
# Activate the brp-eval environment first (README, Setup).
# ===================================================================================================
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"


# ===================================================================================================
# Model Checkpoint
# A converted (HF-layout) checkpoint directory, e.g. one downloaded with `hf download`.
# The projection flags MUST match it: blocknorm / 0.0133 for both released models. A mismatch
# raises on load.
# ===================================================================================================
CKPT_DIR="${CKPT_DIR:?set CKPT_DIR to a checkpoint directory}"
PROTEIN_PROJECTION_NORM="${PROTEIN_PROJECTION_NORM:-blocknorm}"
PROTEIN_PROJECTION_NORM_TARGET="${PROTEIN_PROJECTION_NORM_TARGET:-0.0133}"


# ===================================================================================================
# Paths
# ===================================================================================================
EVAL_TABLE="${EVAL_TABLE:-$REPO_ROOT/data/sft/v2-nofeatures}"   # README, Evaluate on the benchmark
EVAL_SPLIT="${EVAL_SPLIT:-test}"                                 # the benchmark, 6,112 proteins
EVALS_PATH="${EVALS_PATH:-$REPO_ROOT/results/$(basename "$CKPT_DIR")}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-$REPO_ROOT/cache/datasets}"
GO_EMBEDDINGS_PATH="${GO_EMBEDDINGS_PATH:-${BRP_ASSETS_DIR:-$REPO_ROOT/assets}/go_embeddings}"
STRUCTURE_DIR="${STRUCTURE_DIR:-${BRP_ASSETS_DIR:-$REPO_ROOT/assets}/structures}"   # must be FLAT
GO_OBO_PATH="$REPO_ROOT/bioreason2/dataset/go-basic.obo"
TIER2_FROM="${TIER2_FROM:-${BRP_DATA_DIR:-$REPO_ROOT/data/v2}/proteins}"


# ===================================================================================================
# Evaluation Parameters
# The window and the question budget are one decision: build_question.py trims every question to
# 16384 - 8000 - 200 - 2000 - 2 tokens, so a smaller window truncates the generation.
# GPU_MEMORY_UTILIZATION is vLLM's share only; ESM3, the GO encoder and the prompt embeddings
# live outside it. 0.45 fits the 9B on an 80 GB GPU.
# ===================================================================================================
MAX_MODEL_LEN=16384
MAX_NEW_TOKENS=8000
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
BATCH_SIZE="${BATCH_SIZE:-64}"
USE_STRUCTURE="${USE_STRUCTURE:-True}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
NUM_CHUNKS="${NUM_CHUNKS:-1}"
CHUNK_ID="${CHUNK_ID:-0}"
SCORE="${SCORE:-True}"

# N processes on one machine share the cores, or OpenMP oversubscribes them.
if [ -z "${OMP_NUM_THREADS:-}" ]; then
    export OMP_NUM_THREADS=$(( $(nproc) / NUM_CHUNKS > 0 ? $(nproc) / NUM_CHUNKS : 1 ))
fi


# ===================================================================================================
# Preflight
# ===================================================================================================
if [ ! -f "$CKPT_DIR/config.json" ]; then
    echo "Error: no checkpoint at $CKPT_DIR" >&2
    exit 1
fi
if [ ! -d "$EVAL_TABLE/data" ]; then
    echo "Error: no prompt table at $EVAL_TABLE (README, Evaluate on the benchmark)" >&2
    exit 1
fi
if [ -z "$(ls -A "$GO_EMBEDDINGS_PATH" 2>/dev/null)" ]; then
    echo "Error: no GO embeddings at $GO_EMBEDDINGS_PATH (scripts/data/download_assets.py)" >&2
    exit 1
fi
mkdir -p "$EVALS_PATH"


# ===================================================================================================
# Execute Evaluation
# ===================================================================================================
echo "### $CKPT_DIR  projection=$PROTEIN_PROJECTION_NORM/$PROTEIN_PROJECTION_NORM_TARGET"
echo "### $EVAL_TABLE ($EVAL_SPLIT) -> $EVALS_PATH  chunk $CHUNK_ID/$NUM_CHUNKS"

python eval.py \
    --ckpt_dir "$CKPT_DIR" \
    --protein_model_name esm3_sm_open_v1 --protein_embedding_layer 37 \
    --protein_projection_norm "$PROTEIN_PROJECTION_NORM" \
    --protein_projection_norm_target "$PROTEIN_PROJECTION_NORM_TARGET" \
    --go_obo_path "$GO_OBO_PATH" --precomputed_embeddings_path "$GO_EMBEDDINGS_PATH" \
    --go_hidden_dim 512 --go_num_gat_layers 3 --go_num_heads 8 \
    --go_num_reduced_embeddings 200 --go_embedding_dim 2560 \
    --cafa5_dataset "$EVAL_TABLE" --cafa5_dataset_split "$EVAL_SPLIT" \
    --cafa5_dataset_name default --dataset_cache_dir "$DATASET_CACHE_DIR" \
    --structure_dir "$STRUCTURE_DIR" --use_structure "$USE_STRUCTURE" \
    --max_model_len "$MAX_MODEL_LEN" --max_new_tokens "$MAX_NEW_TOKENS" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" --batch_size "$BATCH_SIZE" --max_num_seqs 256 \
    --temperature 0 --top_p 0.95 --pass_at_k 1 --repetition_penalty 1.0 --seed 23 \
    --max_samples "$MAX_SAMPLES" --max_length_protein 2000 \
    --num_chunks "$NUM_CHUNKS" --chunk_id "$CHUNK_ID" \
    --evals_path "$EVALS_PATH"


# ===================================================================================================
# Score
# Once every chunk has written; with NUM_CHUNKS > 1 run the printed command after the last one.
# ===================================================================================================
IA_FILE="${IA_FILE:-${BRP_IA_FILE:-$REPO_ROOT/data/IA_CAFA6.tsv}}"   # not shipped: README, Setup
SCORE_CMD=(python evals/cafa/cafa_evals.py --input_dir "$EVALS_PATH" --go_track both
    --ontology "$GO_OBO_PATH" --ia_file "$IA_FILE" --tier2_from "$TIER2_FROM"
    --reasoning_mode True --output_dir "${EVALS_PATH}-scores")
if [ "$SCORE" != "True" ] || [ "$NUM_CHUNKS" -ne 1 ]; then
    echo "### score with: ${SCORE_CMD[*]}"
elif [ ! -f "$IA_FILE" ]; then
    echo "### no IA weights at $IA_FILE (README, Setup); score once they are there:"
    echo "###   ${SCORE_CMD[*]}"
else
    "${SCORE_CMD[@]}"
    echo "### scores: ${EVALS_PATH}-scores/best_f.tsv (f_w = IA-weighted Fmax, rc_w = recall)"
fi
