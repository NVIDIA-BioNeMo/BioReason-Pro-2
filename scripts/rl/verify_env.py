# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Check the RL venv resolves every piece from the right tree and that the vLLM plugin is visible.

The expected source trees are EXPECT_SKYRL (the patched SkyRL, by default this repository's
skyrl/src) and EXPECT_BRP (this repository, by default the one holding this script), so the same
checks run on any layout.
"""

import importlib
import importlib.metadata
import os
import sys

import torch

FAILURES: list[str] = []

#: Where the NeMo image installs the compiled stack the layer must not shadow.
CONTAINER_SITE = "/usr/local/lib/python3.12/dist-packages"

#: This repository, the default for EXPECT_BRP.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def report(label: str, ok: bool, detail: str = "") -> None:
    """Print one check line and record a failure.

    Args:
        label: What is checked.
        ok: Whether it passed.
        detail: Value shown next to the verdict.
    """
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<52} {detail}")
    if not ok:
        FAILURES.append(label)


def origin(module: str) -> str:
    """Import a module and return the file it came from, or the error.

    Args:
        module: Dotted module name.

    Returns:
        Its `__file__`, or `ERROR <message>`.
    """
    try:
        return str(importlib.import_module(module).__file__)
    except Exception as exc:
        return f"ERROR {type(exc).__name__}: {exc}"


def main() -> None:
    """Run every check and exit non-zero if any failed."""
    brp = os.path.realpath(os.environ.get("EXPECT_BRP") or REPO)
    skyrl = os.path.realpath(os.environ.get("EXPECT_SKYRL") or os.path.join(brp, "skyrl", "src"))
    for name in ("torch", "vllm", "transformers", "ray", "peft", "flash_attn"):
        try:
            version = importlib.metadata.version(name.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            version = getattr(importlib.import_module(name), "__version__", "?")
        print(f"  {name:<14} {version}")

    report("CUDA available", torch.cuda.is_available(), torch.version.cuda or "")
    for name in ("torch", "transformers", "vllm"):
        report(f"{name} is the container's, not a layer copy", origin(name).startswith(CONTAINER_SITE), origin(name))
    report("skyrl_train is the patched SkyRL", origin("skyrl_train").startswith(skyrl), origin("skyrl_train"))
    report("skyrl_gym is the patched SkyRL", origin("skyrl_gym").startswith(skyrl), origin("skyrl_gym"))
    report("bioreason2 is this repository", origin("bioreason2").startswith(brp), origin("bioreason2"))
    report("vllm._C loads", not origin("vllm._C").startswith("ERROR"), origin("vllm._C"))

    plugins = {ep.name: ep.value for ep in importlib.metadata.entry_points(group="vllm.general_plugins")}
    report("vllm.general_plugins advertises the protein model",
           plugins.get("bioreason2_protein_mm") == "bioreason2.rl.vllm_plugin:register", str(plugins))
    report("RL model class imports", not origin("bioreason2.rl.vllm_protein_mm").startswith("ERROR"),
           origin("bioreason2.rl.vllm_protein_mm"))
    report("RL handlers import", not origin("bioreason2.rl.handlers").startswith("ERROR"), origin("bioreason2.rl.handlers"))
    report("reward env imports", not origin("skyrl_gym.envs.go_function.env").startswith("ERROR"),
           origin("skyrl_gym.envs.go_function.env"))

    report("fla (container 0.4.2) imports", not origin("fla").startswith("ERROR"), origin("fla"))
    report("causal_conv1d (container 1.6.2) imports", not origin("causal_conv1d").startswith("ERROR"), origin("causal_conv1d"))
    from transformers.utils import import_utils

    checks = {
        name: getattr(import_utils, name)()
        for name in ("is_flash_linear_attention_available", "is_causal_conv1d_available")
        if hasattr(import_utils, name)
    }
    report("transformers sees the GDN fast-path libraries", bool(checks) and all(checks.values()), str(checks))

    print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
