#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""What the model reads at inference: batched structure, InterPro records and predict.py's ids.

- A protein without a structure, batched with one that has it, reaches ESM3 sequence-only, as it
  does alone: padding is NaN, which the encoder reads as no structure, never a zero backbone.
- InterPro-N predictions are dropped by their `source`, while InterProScan's own JSON, whose
  signatures carry a null type, keeps every match and site.
- predict.py refuses a protein_id that appears twice.

CPU only, in the inference environment (it imports eval.py, and with it vLLM and esm); no
checkpoint, network or data.

    pytest tests/test_inference_inputs.py      or      python tests/test_inference_inputs.py
"""
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

pytest.importorskip("vllm")

import eval as brp_eval  # noqa: E402
import interpro_api  # noqa: E402
import predict  # noqa: E402
from bioreason2.dataset.cafa5 import structure_io  # noqa: E402
from bioreason2.models.protein_encoder import ESM3Encoder  # noqa: E402

STRUCTURED = "MKTAYIAKQRQISFVKSHFSRQ"
BARE = "MSSQQY"
EMBED_DIM = 4


class RecordingESM3:
    """Stands in for ESM3: records each protein it is given and returns a zero embedding."""

    def __init__(self) -> None:
        self.proteins = []

    def encode(self, protein):
        """Record the protein.

        Args:
            protein: the ESMProtein the encoder built.

        Returns:
            The protein, as the protein tensor.
        """
        self.proteins.append(protein)
        return protein

    def forward_and_sample(self, protein, config) -> SimpleNamespace:
        """A zero per-residue embedding of the protein's length plus its two special tokens.

        Args:
            protein: the recorded protein.
            config: the sampling config, unused.

        Returns:
            An object with `per_residue_embedding`.
        """
        return SimpleNamespace(per_residue_embedding=torch.zeros(len(protein.sequence) + 2, EMBED_DIM))


def recording_encoder() -> ESM3Encoder:
    """An ESM3Encoder around RecordingESM3, built without loading ESM3.

    Returns:
        The encoder.
    """
    encoder = ESM3Encoder.__new__(ESM3Encoder)
    encoder.model = RecordingESM3()
    encoder.sampling_config = lambda **kwargs: None
    encoder.embedding_layer = -1
    encoder.layer_outputs = {}
    encoder.hooks = []
    encoder._embedding_dim = EMBED_DIM
    return encoder


def fake_backbone(path: str, sequence: str) -> np.ndarray:
    """Finite backbone coordinates for any sequence, in place of parsing a structure file.

    Args:
        path: the structure file, unused.
        sequence: the protein sequence.

    Returns:
        (len(sequence), 3, 3) coordinates, all ones.
    """
    return np.ones((len(sequence), 3, 3))


def batched_coords(monkeypatch) -> torch.Tensor:
    """eval.py's structure tensor for a batch of one structured and one structure-less protein.

    Args:
        monkeypatch: pytest's monkeypatch, to stand in for the structure parser.

    Returns:
        The padded (2, len(STRUCTURED), 3, 3) tensor.
    """
    monkeypatch.setattr(structure_io, "backbone_resolving_alternates", fake_backbone)
    with tempfile.NamedTemporaryFile(suffix=".pdb") as handle:
        samples = [{"structure_path": handle.name, "sequence": STRUCTURED},
                   {"structure_path": None, "sequence": BARE}]
        return brp_eval._batch_structure_coords(samples, max_length_protein=2000)


def test_batch_pads_with_nan(monkeypatch) -> None:
    """The structure-less row is all NaN, the structured one keeps its coordinates."""
    coords = batched_coords(monkeypatch)
    assert coords.shape == (2, len(STRUCTURED), 3, 3)
    assert torch.equal(coords[0], torch.ones(len(STRUCTURED), 3, 3))
    assert torch.isnan(coords[1]).all()


def test_structure_less_protein_is_encoded_sequence_only(monkeypatch) -> None:
    """In a mixed batch ESM3 gets the structured protein's backbone and nothing for the other."""
    coords = batched_coords(monkeypatch)
    encoder = recording_encoder()
    out = encoder.encode_sequences([STRUCTURED, BARE], batch_idx_map=[0, 1], batch_size=2,
                                   structure_coords=coords)
    with_structure, without = encoder.model.proteins
    assert with_structure.coordinates is not None and torch.equal(
        with_structure.coordinates, torch.ones(len(STRUCTURED), 3, 3))
    assert without.coordinates is None
    assert [tuple(t.shape) for t in out] == [(len(STRUCTURED) + 2, EMBED_DIM), (len(BARE) + 2, EMBED_DIM)]


def match(source: str, signature_type, sites: bool = False) -> dict:
    """One InterProScan match, PIRSR-shaped when it carries sites, else a Pfam domain.

    Args:
        source: the precomputed service's `source`, or "" for InterProScan's own JSON (no field).
        signature_type: the signature's type.
        sites: whether it is a PIRSR match with one residue site.

    Returns:
        The match.
    """
    library = "PIRSR" if sites else "PFAM"
    signature = {"accession": "PIRSR036517-1" if sites else "PF00042", "name": None, "description": None,
                 "type": signature_type, "signatureLibraryRelease": {"library": library, "version": "x"},
                 "entry": None if sites else {"accession": "IPR000971", "name": "Globin", "type": "DOMAIN"}}
    location = {"start": 20, "end": 158}
    if sites:
        location["sites"] = [{"description": "Iron (heme b proximal ligand)",
                              "siteLocations": [{"start": 109, "end": 109, "residue": "H"}]}]
    out = {"signature": signature, "locations": [location]}
    if source:
        out["source"] = source
    return out


def test_interpro_keeps_interproscan_matches_with_null_type() -> None:
    """InterProScan's own JSON: a null signature type no longer drops the match or its sites."""
    records = interpro_api.to_features([match("", None), match("", None, sites=True)])
    assert [(r["src"], r["member_db"], r["type"]) for r in records] == [
        ("interpro_match", "pfam", "domain"), ("interpro_site", "pirsr", "site")]
    assert records[1]["start"] == 109 and records[1]["residue"] == "H"


def test_interpro_drops_interpro_n_predictions() -> None:
    """The precomputed service: InterPro-N rows are dropped, the member database's own row kept."""
    records = interpro_api.to_features([match("Pfam", "domain"), match("InterPro-N", None)])
    assert [(r["src"], r["member_db"]) for r in records] == [("interpro_match", "pfam")]


def test_predict_refuses_repeated_ids() -> None:
    """Two rows with one protein_id would share every stage's results, so the input is refused."""
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as handle:
        handle.write("protein_id\torganism\tsequence\nP1\tHomo sapiens\tMKT\nP1\tHomo sapiens\tMKV\n")
    try:
        predict.read_input_tsv(handle.name)
    except ValueError as err:
        assert "more than once" in str(err)
    else:
        raise AssertionError("a repeated protein_id was accepted")
    finally:
        os.remove(handle.name)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
