# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The TorchMetrics guard keeps the non-commercial EED module out of every Lightning process.

Each case runs in a fresh interpreter, so modules other tests imported cannot interfere.
"""

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# find_spec, not importorskip: importing Lightning here would load the real EED module into the
# pytest process before any guarded module, and that module's guard would then refuse to load.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pytorch_lightning") is None, reason="needs pytorch_lightning"
)

ROOT = Path(__file__).resolve().parents[1]
GUARD = "bioreason2.utils.torchmetrics_guard"

#: Guard first, then Lightning and every TorchMetrics path that reaches EED; prints OK if it held.
GUARDED = textwrap.dedent(
    f"""
    import sys
    import {GUARD}
    import pytorch_lightning
    import torchmetrics
    import torchmetrics.text
    from torchmetrics.functional.text import extended_edit_distance

    eed = sys.modules["torchmetrics.functional.text.eed"]
    assert getattr(eed, "__file__", None) is None, eed
    files = [getattr(m, "__file__", None) or "" for m in list(sys.modules.values())]
    real = [f for f in files if f.replace("\\\\", "/").endswith("torchmetrics/functional/text/eed.py")]
    assert not real, real
    try:
        extended_edit_distance(["a b"], [["a b"]])
    except RuntimeError as err:
        assert "not available" in str(err), err
        print("OK")
    """
)

#: TorchMetrics first, then the guard, which must refuse.
LATE = f"import torchmetrics\nimport {GUARD}\n"


def run(code: str) -> subprocess.CompletedProcess:
    """Run Python code in a fresh interpreter from the repository root.

    Args:
        code: the program.

    Returns:
        The finished process, with stdout and stderr captured as text.
    """
    return subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300
    )


def test_guard_keeps_eed_out() -> None:
    """With the guard imported first, Lightning loads and only the stand-in EED module exists."""
    proc = run(GUARDED)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("OK"), proc.stdout


def test_guard_refuses_after_torchmetrics() -> None:
    """Imported after TorchMetrics, the guard fails loudly instead of doing nothing."""
    proc = run(LATE)
    assert proc.returncode != 0
    assert "was imported before" in proc.stderr, proc.stderr
