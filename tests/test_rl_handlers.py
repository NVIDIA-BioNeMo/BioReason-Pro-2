# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The RL protein handlers read the right span and project it exactly as eval and the engine do.

Run in any env with torch and safetensors (no vLLM, no GPU)::

    python -m pytest tests/test_rl_handlers.py -q
"""

import os
import pathlib
import sys

import pytest
import torch
from safetensors.torch import save_file

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from bioreason2.models.projection_norm import build_protein_projection  # noqa: E402
from bioreason2.rl.handlers import ProteinCachedEncoder, ProteinProjectionHandler  # noqa: E402

#: Small widths keep the test fast; the handlers take both as arguments.
IN_DIM = 16
HIDDEN = 32
#: The released models' blocknorm target RMS.
BLOCKNORM_TARGET = 0.0133


def write_span(cache_dir: str, protein_id: str, rows: int, dim: int = IN_DIM) -> torch.Tensor:
    """Write one cache file the way eval.py's PROTEIN_EMBED_CACHE_WRITE does, and return its tensor.

    Args:
        cache_dir: Cache directory.
        protein_id: File stem.
        rows: Span length, i.e. residues + 2.
        dim: Embedding width.

    Returns:
        The stored bf16 span.
    """
    span = torch.randn(rows, dim).to(torch.bfloat16)
    save_file({"emb": span}, os.path.join(cache_dir, f"{protein_id}.safetensors"))
    return span


def write_sidecar(path: str, norm: str, target: float, seed: int = 0) -> dict[str, torch.Tensor]:
    """Write an fp32 projector sidecar with non-trivial weights.

    Args:
        path: Destination `.pt` file.
        norm: Projection-norm variant.
        target: Its target RMS.
        seed: Weight seed.

    Returns:
        The saved state dict.
    """
    torch.manual_seed(seed)
    module = build_protein_projection(IN_DIM, HIDDEN, norm, target)
    state = {name: value.clone() for name, value in module.state_dict().items()}
    torch.save(state, path)
    return state


def test_encoder_returns_the_cached_span(tmp_path: pathlib.Path) -> None:
    """The span comes back untouched, dtype included."""
    want = write_span(str(tmp_path), "P12345", 7)
    encoder = ProteinCachedEncoder("protein_mod", "encoder", cache_dir=str(tmp_path), embedding_dim=IN_DIM)
    (got,) = encoder.encode([{"protein_id": "P12345", "n_tokens": 7}])
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, want)


def test_encoder_refuses_a_length_the_dataset_did_not_declare(tmp_path: pathlib.Path) -> None:
    """A cache built at another truncation must fail here, not scatter a short span."""
    write_span(str(tmp_path), "P12345", 7)
    encoder = ProteinCachedEncoder("protein_mod", "encoder", cache_dir=str(tmp_path), embedding_dim=IN_DIM)
    with pytest.raises(ValueError, match="disagree"):
        encoder.encode([{"protein_id": "P12345", "n_tokens": 8}])


def test_encoder_refuses_a_missing_protein(tmp_path: pathlib.Path) -> None:
    """A protein outside the split's cache raises rather than being skipped."""
    encoder = ProteinCachedEncoder("protein_mod", "encoder", cache_dir=str(tmp_path), embedding_dim=IN_DIM)
    with pytest.raises(FileNotFoundError, match="Q99999"):
        encoder.encode([{"protein_id": "Q99999", "n_tokens": 3}])


def test_encoder_memory_is_bounded(tmp_path: pathlib.Path) -> None:
    """The LRU keeps at most `lru_size` spans per process."""
    for index in range(5):
        write_span(str(tmp_path), f"P{index}", 4)
    encoder = ProteinCachedEncoder("protein_mod", "encoder", cache_dir=str(tmp_path), embedding_dim=IN_DIM, lru_size=2)
    encoder.encode([{"protein_id": f"P{index}", "n_tokens": 4} for index in range(5)])
    assert list(encoder._lru) == ["P3", "P4"]


@pytest.mark.parametrize("norm, target", [("none", 0.0), ("blocknorm", BLOCKNORM_TARGET)])
def test_projection_matches_eval_bit_for_bit(tmp_path: pathlib.Path, norm: str, target: float) -> None:
    """The trainer's projection equals eval's bf16 module exactly, for both released variants."""
    path = str(tmp_path / "protein_projection.pt")
    state = write_sidecar(path, norm, target)
    handler = ProteinProjectionHandler(
        "protein_mod", "projection", projection_path=path, output_dim=HIDDEN, embedding_dim=IN_DIM, norm=norm, norm_target=target
    )
    reference = build_protein_projection(IN_DIM, HIDDEN, norm, target).to(torch.bfloat16)
    reference.load_state_dict(state)
    span = torch.randn(9, IN_DIM).to(torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(handler.project(span), reference(span))


def test_engine_copy_after_sync_projects_identically(tmp_path: pathlib.Path) -> None:
    """After a training step the engine holds bf16(W); it must project exactly as the trainer does."""
    path = str(tmp_path / "protein_projection.pt")
    write_sidecar(path, "blocknorm", BLOCKNORM_TARGET)
    handler = ProteinProjectionHandler(
        "protein_mod", "projection", projection_path=path, output_dim=HIDDEN, embedding_dim=IN_DIM, norm="blocknorm",
        norm_target=BLOCKNORM_TARGET,
    )
    with torch.no_grad():
        for param in handler.parameters():
            param.add_(1e-3 * torch.randn_like(param))
    engine = build_protein_projection(IN_DIM, HIDDEN, "blocknorm", BLOCKNORM_TARGET).to(torch.bfloat16)
    with torch.no_grad():
        for (_, param), (_, master) in zip(engine.named_parameters(), handler.named_parameters()):
            param.copy_(master.to(param.dtype))
        span = torch.randn(11, IN_DIM).to(torch.bfloat16)
        assert torch.equal(handler.project(span), engine(span))


def test_projection_trains_fp32_masters(tmp_path: pathlib.Path) -> None:
    """Gradients reach every fp32 master weight through the bf16 compute."""
    path = str(tmp_path / "protein_projection.pt")
    write_sidecar(path, "blocknorm", BLOCKNORM_TARGET)
    handler = ProteinProjectionHandler(
        "protein_mod", "projection", projection_path=path, output_dim=HIDDEN, embedding_dim=IN_DIM, norm="blocknorm",
        norm_target=BLOCKNORM_TARGET,
    )
    handler.project(torch.randn(6, IN_DIM)).float().pow(2).sum().backward()
    for name, param in handler.named_parameters():
        assert param.dtype == torch.float32, name
        assert param.grad is not None and param.grad.abs().sum() > 0, name


def test_projection_refuses_the_wrong_norm(tmp_path: pathlib.Path) -> None:
    """A sidecar trained with blocknorm cannot load into the no-tail shape, or the reverse."""
    path = str(tmp_path / "protein_projection.pt")
    write_sidecar(path, "blocknorm", BLOCKNORM_TARGET)
    with pytest.raises(RuntimeError, match="does not match"):
        ProteinProjectionHandler("protein_mod", "projection", projection_path=path, output_dim=HIDDEN, embedding_dim=IN_DIM, norm="none")


def test_projection_refuses_a_missing_path() -> None:
    """A random projector does not error later; it looks like RL that does not learn."""
    with pytest.raises(ValueError, match="projection_path"):
        ProteinProjectionHandler("protein_mod", "projection", output_dim=HIDDEN, embedding_dim=IN_DIM)
