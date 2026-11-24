#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Judge-reward GRPO on one 8-GPU node, colocated, with the released RL model's recipe as the
# defaults. Run inside the NeMo 26.06 container (nvcr.io/nvidia/nemo:26.06) with the RL venv that
# scripts/rl/build_env.sh builds active. Every knob is an environment variable.
#
#   MODEL_PATH=<prepared checkpoint> TRAIN_FILE=<train.parquet> \
#   CACHE_DIR=<ESM3 layer-37 cache> OUT_DIR=<run directory> \
#   SKYRL_JUDGE_URL=<https://endpoint/v1/chat/completions> SKYRL_JUDGE_MODEL=<judge model> \
#   JUDGE_API_KEY=<key> bash scripts/rl/grpo.sh
#
# A run resumes from its latest checkpoint in OUT_DIR, so re-running the same command continues it.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to a prepared checkpoint (bioreason2/rl/prepare_checkpoint.py)}"
TRAIN_FILE="${TRAIN_FILE:?set TRAIN_FILE to the RL train parquet}"
CACHE_DIR="${CACHE_DIR:?set CACHE_DIR to the ESM3 layer-37 cache of every training protein}"
OUT="${OUT_DIR:?set OUT_DIR, the run directory}"
# Absolute, because the trainer starts from /tmp.
MODEL_PATH="$(realpath -m "${MODEL_PATH}")" TRAIN_FILE="$(realpath -m "${TRAIN_FILE}")"
CACHE_DIR="$(realpath -m "${CACHE_DIR}")" OUT="$(realpath -m "${OUT}")"
RUN_NAME="${RUN_NAME:-$(basename "${OUT}")}"
mkdir -p "${OUT}/wandb" "${OUT}/ckpt" "${OUT}/exports"
exec > >(tee -a "${OUT}/launch.log") 2>&1
echo "=== $(date -u) ${RUN_NAME} on $(hostname)"
# Written by the trainer when the reward source has gone (rollouts unscored for several steps).
# Delete the file to resume.
export SKYRL_STOP_FILE="${OUT}/STOP_unscored"
if [ -f "${SKYRL_STOP_FILE}" ]; then
    echo "stop file present, not starting:"; cat "${SKYRL_STOP_FILE}"; exit 0
fi
# Written when training finished. A later start would resume from the final checkpoint, restart
# the exhausted dataloader and train another epoch, so it ends here instead.
if [ -f "${OUT}/DONE" ]; then
    echo "run complete, not starting: $(cat "${OUT}/DONE")"; exit 0
fi

# ---------------------------------------------------------------- reward
export SKYRL_JUDGE_PROMPT_STYLE="${SKYRL_JUDGE_PROMPT_STYLE:-axes}"
export SKYRL_JUDGE_GROUP_SIZE="${SKYRL_JUDGE_GROUP_SIZE:-0}"
export SKYRL_JUDGE_REFERENCE="${SKYRL_JUDGE_REFERENCE:-0}"
export SKYRL_JUDGE_RUBRIC="$(realpath -m "${SKYRL_JUDGE_RUBRIC:-${REPO}/scripts/rl/judge_rubric.md}")"
[ -n "${SKYRL_JUDGE_KEY_FILE:-}" ] && export SKYRL_JUDGE_KEY_FILE="$(realpath -m "${SKYRL_JUDGE_KEY_FILE}")"
# No penalty to 13k characters, zero reward at 16k.
export SKYRL_JUDGE_LEN_FREE="${SKYRL_JUDGE_LEN_FREE:-13000}" SKYRL_JUDGE_LEN_CAP="${SKYRL_JUDGE_LEN_CAP:-16000}"
export SKYRL_JUDGE_LOG_PATH="$(realpath -m "${SKYRL_JUDGE_LOG_PATH:-${OUT}/judge_samples.jsonl}")"
: "${SKYRL_JUDGE_URL:?set SKYRL_JUDGE_URL, the judge endpoint's full /chat/completions URL}"
: "${SKYRL_JUDGE_MODEL:?set SKYRL_JUDGE_MODEL, the judge model's name at that endpoint}"
export SKYRL_JUDGE_URL SKYRL_JUDGE_MODEL
KEY_VAR="${SKYRL_JUDGE_KEY_VAR:-JUDGE_API_KEY}"
if [ -z "${!KEY_VAR:-}" ] && ! grep -q "^${KEY_VAR}=" "${SKYRL_JUDGE_KEY_FILE:-/dev/null}" 2>/dev/null; then
    echo "FATAL: no judge key: set ${KEY_VAR} or name a KEY=VALUE file in SKYRL_JUDGE_KEY_FILE"; exit 1
fi
[ -f "${SKYRL_JUDGE_RUBRIC}" ] || { echo "FATAL: no rubric at ${SKYRL_JUDGE_RUBRIC}"; exit 1; }
ENV_WORKERS="${ENV_WORKERS:-512}"
# Judge while generating: the async engine and the per-trajectory agent loop fire each rollout's
# env step, and so its judge call, the moment it finishes. Rollout logprobs are off on that path;
# skip_old_logprob_pass takes old = current under one mini-batch either way.
OVERLAP="${OVERLAP:-1}"
if [ "${OVERLAP}" = "1" ]; then
    GEN_FLAGS="generator.async_engine=true generator.batched=false generator.sampling_params.logprobs=null"
else
    GEN_FLAGS="generator.async_engine=false generator.batched=true generator.sampling_params.logprobs=0"
fi

# ---------------------------------------------------------------- model / LoRA
cfg() { python3 -c "import json; print(json.load(open('${MODEL_PATH}/config.json'))['$1'])" 2>/dev/null; }
HIDDEN="$(cfg hidden_size)"
NORM="$(cfg protein_projection_norm)"
NORM_TARGET="$(cfg protein_projection_norm_target)"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
TARGET_MODULES="${TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]}"
# The LoRA target count the fork must find (128 at 9B); a mismatch means the targets did not match.
export SKYRL_EXPECTED_LORA_TARGETS="${SKYRL_EXPECTED_LORA_TARGETS:-128}"

# ---------------------------------------------------------------- algorithm
LR="${LR:-2e-5}"
NSAMPLES="${NSAMPLES:-16}"
TRAIN_BATCH="${TRAIN_BATCH:-32}"
MAX_GEN="${MAX_GEN:-4096}"
MAX_PROMPT="${MAX_PROMPT:-8704}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
EPOCHS="${EPOCHS:-1}"
ALGO_FLAGS="trainer.algorithm.eps_clip_low=0.2 trainer.algorithm.eps_clip_high=0.28"
ALGO_FLAGS="${ALGO_FLAGS} trainer.algorithm.policy_loss_type=dual_clip trainer.algorithm.clip_ratio_c=3.0"
# Advantages divided by the group's reward sd.
NORM_BY_STD="${NORM_BY_STD:-true}"
ALGO_FLAGS="${ALGO_FLAGS} trainer.algorithm.loss_reduction=seq_mean_token_sum_norm trainer.algorithm.grpo_norm_by_std=${NORM_BY_STD}"
ALGO_FLAGS="${ALGO_FLAGS} trainer.algorithm.grpo_baseline_floor=0 trainer.algorithm.onpolicy_exact_ratio=true"
ALGO_FLAGS="${ALGO_FLAGS} trainer.algorithm.max_seq_len=${MAX_GEN}"
ALGO_FLAGS="${ALGO_FLAGS} trainer.algorithm.skip_old_logprob_pass=${SKIP_OLD_LOGPROBS:-true}"
export SKYRL_RESPONSE_LOGITS_ONLY="${SKYRL_RESPONSE_LOGITS_ONLY:-1}"
NUM_GPUS="${NUM_GPUS:-8}"
GPU_MEM="${GPU_MEM:-0.45}"
CKPT_INTERVAL="${CKPT_INTERVAL:-2}"

# ---------------------------------------------------------------- process environment
cd /tmp
python -c "import bioreason2, skyrl_gym, skyrl_train" \
    || { echo "FATAL: bioreason2 / skyrl_gym / skyrl_train do not import; activate the RL venv"; exit 1; }
# Eight engines compiling into one shared directory race; keep Triton's cache node-local.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton}"
mkdir -p "${TRITON_CACHE_DIR}"
export WANDB_DIR="${OUT}/wandb"
# One W&B run per RUN_NAME, across restarts: the id is kept in the run dir and every trainer resumes it.
[ -s "${OUT}/wandb_run_id" ] || printf '%s' "${RUN_NAME}" | md5sum | cut -c1-16 > "${OUT}/wandb_run_id"
export WANDB_RUN_ID="$(cat "${OUT}/wandb_run_id")" WANDB_RESUME=allow
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export SKYRL_DISABLE_META_INIT=1
# A generation that has not returned in this long marks its engine hung; the step then raises so
# the restart loop below resumes from the last checkpoint at once.
export SKYRL_GENERATE_TIMEOUT_S="${SKYRL_GENERATE_TIMEOUT_S:-600}" SKYRL_RESTART_ON_GENERATE_TIMEOUT=1
export SKYRL_MM_MODALITY=protein SKYRL_VLLM_ENABLE_LORA=1
export SKYRL_BALANCE_DP_TOKENS=1 SKYRL_BATCHED_METRIC_ALLREDUCE=1
# W&B when there is a key (the released run logged there), the console otherwise: SkyRL starts W&B
# inside a Ray actor, where a missing key fails the trainer rather than prompting.
LOGGER="${LOGGER:-$([ -n "${WANDB_API_KEY:-}" ] && echo wandb || echo console)}"
echo "logger ${LOGGER}$([ "${LOGGER}" = wandb ] && echo ", run id ${WANDB_RUN_ID}")"
echo "model=${MODEL_PATH} hidden=${HIDDEN} norm=${NORM}/${NORM_TARGET} train=${TRAIN_FILE}"
echo "norm_by_std=${NORM_BY_STD} judge=${SKYRL_JUDGE_MODEL} style=${SKYRL_JUDGE_PROMPT_STYLE} group=${SKYRL_JUDGE_GROUP_SIZE} reference=${SKYRL_JUDGE_REFERENCE} len=${SKYRL_JUDGE_LEN_FREE}/${SKYRL_JUDGE_LEN_CAP} rubric=${SKYRL_JUDGE_RUBRIC} overlap=${OVERLAP}"
[ -n "${HIDDEN}" ] && [ -n "${NORM}" ] && [ -n "${NORM_TARGET}" ] \
    || { echo "FATAL: cannot read the projector config from ${MODEL_PATH}/config.json"; exit 1; }

# Ray's ports, kept below the kernel's ephemeral range: a vLLM TCPStore port drawn inside Ray's
# worker range aborts start-up. Check `sysctl net.ipv4.ip_local_port_range` on a new machine.
RAY_PORTS="${RAY_PORTS:---port=6379 --object-manager-port=8006 --node-manager-port=8007 --metrics-export-port=8005 --runtime-env-agent-port=8004 --dashboard-agent-grpc-port=8002 --dashboard-agent-listen-port=8003 --min-worker-port=2000 --max-worker-port=5999}"
# The head's port, read back out of RAY_PORTS so an override moves the address with it.
RAY_HEAD_PORT="$(echo "${RAY_PORTS}" | grep -oE -- '--port=[0-9]+' | cut -d= -f2)"

# Stall watchdog: the log is written every few seconds while anything moves (rollout progress,
# judge and trainer phases), so a log silent for STALL_S means a hung engine. It stops the trainer
# after recording each GPU's load (a hung engine's GPU stands out from the idle rest).
STALL_S="${STALL_S:-1200}"


watchdog() {
    while sleep 60; do
        age=$(( $(date +%s) - $(stat -c %Y "${OUT}/launch.log") ))
        if [ "${age}" -gt "${STALL_S}" ]; then
            echo "=== $(date -u) STALL: launch.log silent for ${age}s; stopping the trainer"
            nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>&1 | sed 's/^/  gpu /'
            pkill -TERM -f skyrl_train.entrypoints.main_base
            return
        fi
    done
}


run_trainer() {
    python -m skyrl_train.entrypoints.main_base \
      data.train_data="['${TRAIN_FILE}']" \
      trainer.algorithm.advantage_estimator=grpo \
      trainer.policy.model.path="${MODEL_PATH}" \
      trainer.strategy=fsdp2 \
      trainer.placement.colocate_all=true \
      trainer.placement.policy_num_nodes=1 \
      trainer.placement.ref_num_nodes=1 \
      trainer.placement.policy_num_gpus_per_node=${NUM_GPUS} \
      trainer.placement.ref_num_gpus_per_node=${NUM_GPUS} \
      trainer.policy.model.lora.rank=${LORA_RANK} \
      trainer.policy.model.lora.alpha=${LORA_ALPHA} \
      trainer.policy.model.lora.dropout=0.0 \
      trainer.target_modules="${TARGET_MODULES}" \
      trainer.epochs=${EPOCHS} \
      trainer.train_batch_size=${TRAIN_BATCH} \
      trainer.policy_mini_batch_size=${TRAIN_BATCH} \
      trainer.micro_forward_batch_size_per_gpu=1 \
      trainer.micro_train_batch_size_per_gpu=1 \
      trainer.update_epochs_per_batch=1 \
      trainer.policy.optimizer_config.lr=${LR} \
      trainer.policy.optimizer_config.num_warmup_steps=0 \
      trainer.algorithm.use_kl_loss=true \
      trainer.algorithm.kl_loss_coef=0.001 \
      trainer.algorithm.kl_estimator_type=k3 \
      trainer.max_prompt_length=${MAX_PROMPT} \
      trainer.ckpt_interval=${CKPT_INTERVAL:-5} \
      trainer.eval_interval=-1 \
      trainer.export_path="${OUT}/exports" \
      trainer.ckpt_path="${OUT}/ckpt" \
      trainer.max_ckpts_to_keep=${MAX_CKPTS:-2} \
      trainer.resume_mode="${RESUME_MODE:-latest}" \
      trainer.seed=23 \
      trainer.logger="${LOGGER:-wandb}" \
      trainer.project_name="${WANDB_PROJECT:-bioreason-pro-rl}" \
      trainer.run_name="${RUN_NAME}" \
      ${ALGO_FLAGS} ${GEN_FLAGS} \
      environment.env_class=go_function \
      environment.skyrl_gym.max_env_workers=${ENV_WORKERS} \
      generator.backend=vllm \
      generator.run_engines_locally=true \
      generator.weight_sync_backend=nccl \
      generator.zero_reward_on_non_stop=true \
      generator.num_inference_engines=${NUM_GPUS} \
      generator.inference_engine_tensor_parallel_size=1 \
      generator.n_samples_per_prompt=${NSAMPLES} \
      generator.gpu_memory_utilization=${GPU_MEM} \
      generator.enforce_eager=false \
      generator.enable_prefix_caching=true \
      generator.sampling_params.max_generate_length=${MAX_GEN} \
      generator.sampling_params.temperature=1.0 \
      generator.sampling_params.top_p=1.0 \
      generator.max_num_seqs=128 \
      generator.model_dtype=bfloat16 \
      +generator.engine_init_kwargs.max_model_len=${MAX_MODEL_LEN} \
      +generator.engine_init_kwargs.limit_mm_per_prompt.protein=1 \
      +generator.engine_init_kwargs.enable_mm_embeds=true \
      +modalities.protein_mod.placeholder_token="'<|protein_pad|>'" \
      +modalities.protein_mod.max_placeholder_tokens=2002 \
      +modalities.protein_mod.encoder.target="bioreason2.rl.handlers:ProteinCachedEncoder" \
      +modalities.protein_mod.encoder.kwargs.cache_dir="${CACHE_DIR}" \
      +modalities.protein_mod.projection.target="bioreason2.rl.handlers:ProteinProjectionHandler" \
      +modalities.protein_mod.projection.kwargs.projection_path="${MODEL_PATH}/protein_projection.pt" \
      +modalities.protein_mod.projection.kwargs.output_dim=${HIDDEN} \
      +modalities.protein_mod.projection.kwargs.embedding_dim=1536 \
      +modalities.protein_mod.projection.kwargs.norm="${NORM}" \
      +modalities.protein_mod.projection.kwargs.norm_target=${NORM_TARGET} \
      +modalities.protein_mod.projection.kwargs.dtype=float32 \
      +modalities.protein_mod.trainable.encoder=false \
      +modalities.protein_mod.trainable.projection=true \
      "$@"
}


# A stalled or crashed trainer restarts from the last checkpoint, at most MAX_RESTARTS times (a
# deterministic crash should end the run), and not after the stop file (the reward source is down).
MAX_RESTARTS="${MAX_RESTARTS:-2}"
attempt=0
while :; do
    ray start --head --num-gpus="${NUM_GPUS}" --temp-dir=/tmp/ray ${RAY_PORTS} --disable-usage-stats >/dev/null
    export RAY_ADDRESS="127.0.0.1:${RAY_HEAD_PORT:-6379}"
    watchdog &
    WATCHDOG=$!
    run_trainer "$@"
    rc=$?
    kill "${WATCHDOG}" 2>/dev/null
    ray stop --force >/dev/null 2>&1
    [ "${rc}" -eq 0 ] && break
    [ -f "${SKYRL_STOP_FILE}" ] && break
    attempt=$((attempt + 1))
    [ "${attempt}" -gt "${MAX_RESTARTS}" ] && break
    echo "=== $(date -u) trainer exit ${rc}; restart ${attempt}/${MAX_RESTARTS}"
    sleep 20
done
if [ "${rc}" -eq 0 ]; then
    echo "finished at $(date -u)" > "${OUT}/DONE"
fi
echo "=== $(date -u) exit ${rc}"
exit ${rc}
