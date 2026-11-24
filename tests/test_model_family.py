# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""`detect_family` trusts a readable config over the model's name, and reads the name only without one.

Runs on CPU in either environment, under pytest:

    python -m pytest tests/test_model_family.py
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from bioreason2.models import model_family  # noqa: E402

pytest.importorskip("transformers")


def write_config(path: str, model_type: str) -> str:
    """A model directory holding only a config.json.

    Args:
        path: the directory to create.
        model_type: the config's `model_type`.

    Returns:
        The directory.
    """
    os.makedirs(path)
    with open(os.path.join(path, "config.json"), "w") as handle:
        json.dump({"model_type": model_type}, handle)
    return path


def test_config_overrides_a_qwen_name(tmp_path) -> None:
    """A Qwen3.5-named directory whose config declares another model is refused."""
    path = write_config(str(tmp_path / "Qwen3.5-9B-copy"), "llama")
    with pytest.raises(ValueError, match="llama"):
        model_family.detect_family(path)


def test_qwen_config_is_detected(tmp_path) -> None:
    """A Qwen3.5 config is detected whatever the directory is called."""
    path = write_config(str(tmp_path / "checkpoint"), "qwen3_5")
    assert model_family.detect_family(path) == model_family.QWEN35


def test_name_is_read_without_a_config(tmp_path) -> None:
    """With no readable config (offline, say), the name decides."""
    path = str(tmp_path / "Qwen3.5-9B")
    os.makedirs(path)
    assert model_family.detect_family(path) == model_family.QWEN35


def test_unknown_without_a_config(tmp_path) -> None:
    """With no config and no Qwen3.5 name, the backbone is refused."""
    path = str(tmp_path / "some-model")
    os.makedirs(path)
    with pytest.raises(ValueError):
        model_family.detect_family(path)
