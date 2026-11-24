#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Build the SkyRL that BioReason-Pro 2's RL runs on: upstream SkyRL at a pinned commit plus
# bioreason.patch, checked against the expected tree. Idempotent.
#
#   bash skyrl/fetch.sh [<dir>]        # default skyrl/src
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$(realpath -m "${1:-${HERE}/src}")"
UPSTREAM=https://github.com/NovaSky-AI/SkyRL.git
BASE=50a570c07e105b324778d357c1f304d4c03a8156
# git's tree hash of upstream BASE with the patch applied; any other result is not this patch. The
# released model was trained at tree f179f0c5, which differs only in the go_function env: it also
# offered a GO-F1 reward and scored validation rollouts on GO-F1. The released run's reward was the
# judge alone, which this tree keeps unchanged.
EXPECTED_TREE=fdc3372642b5d4eaf2e02165840763e171aa7dd4

if [ ! -d "${DEST}/.git" ]; then
    git init -q "${DEST}"
    git -C "${DEST}" remote add origin "${UPSTREAM}"
fi
if [ "$(git -C "${DEST}" rev-parse -q --verify HEAD^{tree} 2>/dev/null || true)" != "${EXPECTED_TREE}" ]; then
    git -C "${DEST}" fetch -q --depth 1 origin "${BASE}"
    git -C "${DEST}" checkout -q -f "${BASE}"
    git -C "${DEST}" clean -qfdx
    git -C "${DEST}" apply "${HERE}/bioreason.patch"
    git -C "${DEST}" add -A
    git -C "${DEST}" -c user.name=bioreason -c user.email=bioreason@localhost \
        commit -q -m "BioReason-Pro 2 patch on SkyRL ${BASE:0:8}"
fi

TREE="$(git -C "${DEST}" rev-parse HEAD^{tree})"
if [ "${TREE}" != "${EXPECTED_TREE}" ]; then
    echo "FATAL: ${DEST} is tree ${TREE}, expected ${EXPECTED_TREE}" >&2
    exit 1
fi
echo "SkyRL ${BASE:0:8} + bioreason.patch at ${DEST} (tree ${TREE:0:12})"
