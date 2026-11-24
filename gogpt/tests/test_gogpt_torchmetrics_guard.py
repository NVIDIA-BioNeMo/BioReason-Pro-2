# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The TorchMetrics guard keeps the non-commercial EED module out of GO-GPT training.

Each case runs in a fresh interpreter, so modules other tests imported cannot interfere.
"""

import importlib.util
import os
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

SRC = Path(__file__).resolve().parents[1] / "src"

#: The training module imports the guard before Lightning; prints OK if it held.
GUARDED = textwrap.dedent(
    """
    import sys
    import gogpt.models.gogpt_lightning
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
LATE = "import torchmetrics\nimport gogpt.torchmetrics_guard\n"


def run(code: str) -> subprocess.CompletedProcess:
    """Run Python code in a fresh interpreter with GO-GPT's sources importable.

    Args:
        code: the program.

    Returns:
        The finished process, with stdout and stderr captured as text.
    """
    path = os.pathsep.join(p for p in (str(SRC), os.environ.get("PYTHONPATH", "")) if p)
    return subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": path},
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_guard_keeps_eed_out() -> None:
    """Importing the Lightning module loads Lightning with only the stand-in EED module."""
    proc = run(GUARDED)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("OK"), proc.stdout


def test_guard_refuses_after_torchmetrics() -> None:
    """Imported after TorchMetrics, the guard fails loudly instead of doing nothing."""
    proc = run(LATE)
    assert proc.returncode != 0
    assert "was imported before" in proc.stderr, proc.stderr
