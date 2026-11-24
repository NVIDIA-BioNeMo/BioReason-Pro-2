# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""GO tokenizer, organism mapper and GO name/format helpers (no downloads)."""

from pathlib import Path

import pytest

from gogpt.data.tokenizer import GOTermTokenizer
from gogpt.utils.go_ontology import format_predictions, load_go_names
from gogpt.utils.organism_mapper import OrganismMapper

OBO = Path(__file__).resolve().parents[1] / "data" / "go-basic.obo"


def test_tokenizer_json_round_trip(tmp_path):
    tok = GOTermTokenizer({"GO:0000003", "GO:0000001", "GO:0000002"})
    tok.to_json(tmp_path / "t.json")
    back = GOTermTokenizer.from_json(tmp_path / "t.json")
    assert back.token_to_id == tok.token_to_id
    assert tok.token_to_id["GO:0000001"] == 7  # 7 special tokens first, then sorted GO ids


def test_tokenizer_encode_decode():
    tok = GOTermTokenizer({"GO:0000001", "GO:0000002", "GO:0000003"})
    ids = tok.encode(["GO:0000002", "GO:0000001"], "BP")
    assert ids == [3, 8, 7, 4]
    # decode stops at the aspect END, skips special tokens and repeats, keeps order
    assert tok.decode([3, 8, 0, 7, 8, 4, 9], "BP") == ["GO:0000002", "GO:0000001"]
    with pytest.raises(KeyError):
        tok.encode(["GO:9999999"], "MF")
    with pytest.raises(ValueError):
        tok.encode([], "XX")


def test_organism_mapper(tmp_path):
    taxa = ["9606"] * 3 + ["10090"] * 2 + ["7227"]
    m = OrganismMapper(taxa, top_n_organisms=2,
                       names={"9606": "Homo sapiens (Human)", "10090": "Mus musculus (Mouse)"})
    assert m.organism_to_idx == {"<UNKNOWN>": 0, "10090": 1, "9606": 2}
    m.to_json(tmp_path / "m.json")
    m = OrganismMapper.from_json(tmp_path / "m.json")
    assert m.resolve(9606) == m.resolve("9606") == "9606"
    assert m.resolve("Homo sapiens (Human)") == m.resolve("homo sapiens") == "9606"
    assert m.resolve("7227") is None  # outside the top 2
    assert m.resolve("Escherichia coli") is None
    assert m.resolve(None) is None
    assert m.map_organism("7227") == 0


def test_load_go_names():
    names = load_go_names(OBO)
    assert names["GO:0003674"] == "molecular_function".replace("_", " ")
    assert names["GO:0005515"] == "protein binding"
    assert not any(n == "" for n in names.values())


def test_go_names_match_goatools():
    """BioReason-Pro 2 names GO terms with goatools; the prompt text must not differ."""
    obo_parser = pytest.importorskip("goatools.obo_parser")
    dag = obo_parser.GODag(str(OBO), prt=None)
    expected = {go_id: term.name.replace("_", " ") for go_id, term in dag.items()}
    assert load_go_names(OBO) == expected


def test_format_predictions():
    names = {"GO:0003674": "molecular function", "GO:0005488": "binding"}
    text = format_predictions({"MF": ["GO:0003674", "GO:0005488"], "BP": [], "CC": ["GO:0000001"]}, names)
    assert text == ("Molecular Function (MF): GO:0003674 (molecular function), GO:0005488 (binding)\n"
                    "Biological Process (BP): \n"
                    "Cellular Component (CC): GO:0000001 (Unknown)")
