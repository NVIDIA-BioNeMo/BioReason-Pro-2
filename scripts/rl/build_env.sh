#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# The RL environment: a venv layered on the NeMo 26.06 container (nvcr.io/nvidia/nemo:26.06), run
# INSIDE that container. It is separate from the SFT and eval environments, whose pins it cannot
# meet: RL runs on the container's torch, transformers 5.x and vLLM, plus SkyRL with this
# repository's patch (skyrl/). Idempotent.
#
#   bash scripts/rl/build_env.sh <venv dir>
#
# SKYRL_DIR moves the patched SkyRL checkout (default skyrl/src, built by skyrl/fetch.sh). Then, on
# a GPU node, scripts/rl/verify_env.py checks CUDA, vllm._C, the GDN fast path and the protein vLLM
# plugin; on a CPU node those checks fail by design.
set -euo pipefail
# Absolute, because the final check runs from /tmp.
VENV="$(realpath -m "${1:?venv directory}")"
BRP="$(cd "$(dirname "$0")/../.." && pwd)"
SKYRL="$(realpath -m "${SKYRL_DIR:-${BRP}/skyrl/src}")"
bash "${BRP}/skyrl/fetch.sh" "${SKYRL}"
PY="${VENV}/bin/python"
SITE="${VENV}/lib/python3.12/site-packages"
OPT=/opt/venv/lib/python3.12/site-packages

[ -x "${PY}" ] || python -m venv --system-site-packages "${VENV}"

# The container's torch is a pre-release that satisfies no `torch>=X`, so without its versions as
# constraints pip pulls a PyPI torch into the venv and breaks vllm._C and causal_conv1d.
CONSTRAINTS="${VENV}/container-constraints.txt"
"${PY}" -m pip freeze --all 2>/dev/null | grep "==" | grep -viE "^(ray|peft)==" > "${CONSTRAINTS}"
grep -E "^(torch|transformers|vllm)==" "${CONSTRAINTS}" || true

"${PY}" -m pip install --no-cache-dir -q -c "${CONSTRAINTS}" \
    loguru tqdm tensorboard func_timeout "hydra-core==1.3.2" accelerate torchdata \
    omegaconf hf_transfer wandb tensordict jaxtyping polars s3fs fastapi uvicorn \
    "ray==2.51.1" "peft==0.19.1" colorama

# The Qwen3.5 GDN fast-path kernels ship in the container's /opt/venv, which the venv does not see.
for name in fla flash_linear_attention-0.4.2.dist-info fla_core-0.4.2.dist-info \
            causal_conv1d causal_conv1d-1.6.2.post1.dist-info; do
    ln -sfn "${OPT}/${name}" "${SITE}/${name}"
done
for so in "${OPT}"/causal_conv1d_cuda*.so; do ln -sfn "${so}" "${SITE}/$(basename "${so}")"; done

# Editable and --no-deps: the fork and this repo must not move the container's pins, and the vLLM
# plugin entry point (pyproject.toml) is read from installed metadata.
"${PY}" -m pip install --no-cache-dir -q --no-deps -e "${SKYRL}/skyrl-gym"
"${PY}" -m pip install --no-cache-dir -q --no-deps -e "${SKYRL}/skyrl-train"
"${PY}" -m pip install --no-cache-dir -q --no-deps -e "${BRP}"

cd /tmp
EXPECT_SKYRL="${SKYRL}" EXPECT_BRP="${BRP}" "${PY}" "${BRP}/scripts/rl/verify_env.py" || true
echo "built ${VENV}; on a CPU node the CUDA, vllm._C and GDN checks fail by design"
