#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The audited grounding scorer (evals/grounding/grounding.py) on fixtures.

Each case is one rule of the audit, scored against a synthetic protein, so the test needs no test-set parquet, no eval
results and no network.

    python -m pytest tests/test_grounding.py        # or: python tests/test_grounding.py
"""
import gzip
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from evals.grounding import grounding  # noqa: E402

LENGTH = 300
# Residues placed in an otherwise all-alanine chain; 13 to 28 are glycines for the counting-run case.
PLACED = {1: "M", 40: "T", 57: "H", 171: "S", 195: "S", 219: "D"}
GLYCINES = range(13, 29)
PROTEIN = "P0TEST"
REFERENCE = {"taxon_domain": {"9606": "Eukaryota", "562": "Bacteria"}, "virus_host_domain": {},
             "retired_interpro": {"IPR999999": ["PF00001"]}}
PARENTS = {"IPR000001": "IPR000002"}

# (text, [(aa, pos, category, correct)]): correct is None for a claim that is not counted.
RESIDUE_CASES = (
    ("The catalytic Ser195 attacks the substrate.", [("S", 195, "self", True)]),
    ("The catalytic Ser-196 is wrong here.", [("S", 196, "self", False)]),
    ("the catalytic aspartate at 219 activates water.", [("D", 219, "self", True)]),
    ("The mature chain begins with Thr1, the nucleophile.", [("T", 1, "self", True)]),
    ("It methylates histone H3 on Lys-27.", [("K", 27, "histone", None)]),
    ("It phosphorylates CsrA on Ser-171.", [("S", 171, "substrate", None)]),
    ("Phosphorylation of host STAT1 at Tyr-70 blocks signalling.", [("Y", 70, "substrate", None)]),
    ("It forms Lys48-linked chains on its substrates.", [("K", 48, "ubiquitin", None)]),
    ("a Cys2-His2 zinc finger binds DNA.", [("C", 2, "zinc", None), ("H", 2, "zinc", None)]),
    ("The zinc site at cysteines and histidines 33, 36, 50 and 52 holds the metal.",
     [("H", p, "mixed_list", None) for p in (33, 36, 50, 52)]),
    ("A list: " + ", ".join(f"Gly{p}" for p in GLYCINES) + ".", [("G", p, "loop", None) for p in GLYCINES]),
)

# (text, feature type, whether the regex reader says the protein has it).
PRESENCE_CASES = (
    ("A single transmembrane helix at 100 to 122 anchors the chain.", "transmembrane", True),
    ("There is no signal peptide and no transmembrane helix.", "transmembrane", False),
    ("Its partner is a transmembrane receptor.", "transmembrane", False),
    ("GO:0022857 transmembrane transporter activity", "transmembrane", False),
    ("One predictor calls 5 to 23 a transmembrane helix, but that is the hydrophobic core of the signal.",
     "transmembrane", False),
    ("The N-terminal stretch is a signal anchor rather than a cleavable signal peptide.", "transmembrane", True),
    ("The N-terminal stretch is a signal anchor rather than a cleavable signal peptide.", "signal_peptide", False),
    ("If this protein had a signal peptide, I would reconsider.", "signal_peptide", False),
    ("a coiled coil at 200 to 240 mediates dimerization.", "coiled_coil", True),
    ("His57 coordinates the zinc ion.", "binding_site", True),
)

# (text, feature type, start, end, distance with the site records): the nearest true feature's distance.
SPAN_CASES = (
    ("A single transmembrane helix at 100 to 122 anchors the chain.", "transmembrane", 100, 122, 0),
    ("A transmembrane helix at 140 to 160 anchors the chain.", "transmembrane", 140, 160, 40),
    ("a coiled coil at 200 to 240 mediates dimerization.", "coiled_coil", 200, 240, 0),
    ("The catalytic Ser-196 is wrong here.", "active_site", 196, 196, 1),
)

# (label, row overrides, feature type, has it, the truth note it leaves): the truth-side rules.
TRUTH_CASES = (
    ("TMHMM helix on a signal peptide is its h-region",
     {"features": [("uniprot", "SIGNAL", 1, 22, ""), ("interpro_match", "tmhmm", 5, 20, "")]}, "transmembrane", False,
     "tmhmm_on_signal_peptide"),
    ("SignalP's eukaryotic mode does not count for a bacterium",
     {"taxid": 562.0, "features": [("interpro_match", "signalp_e", 1, 22, "")]}, "signal_peptide", False,
     "signalp_mode"),
    ("SignalP's Gram mode counts for a bacterium",
     {"taxid": 562.0, "features": [("interpro_match", "signalp_g+", 1, 22, "")]}, "signal_peptide", True, None),
    ("a SIGNAL ending before residue 10 is a fragment's remnant",
     {"features": [("uniprot", "SIGNAL", 1, 3, "")]}, "signal_peptide", False, "signal_remnant"),
    ("a binding record for RNA is no ligand site",
     {"features": [("uniprot", "BINDING", 50, 55, "RNA")]}, "binding_site", False, "non_ligand_uniprot"),
    ("a binding record for ATP is a ligand site",
     {"features": [("uniprot", "BINDING", 50, 55, "ATP ChEBI:CHEBI:30616")]}, "binding_site", True, None),
    ("an EC number makes an enzyme", {"protein_names": "Trypsin (EC 3.4.21.4)"}, "active_site", True, None),
    ("the RING-type E3 acts through the E2's cysteine",
     {"protein_names": "E3 ubiquitin-protein ligase (EC 2.3.2.27)"}, "active_site", False, None),
    ("a single-pass membrane protein has a transmembrane segment",
     {"subcellular_location": "Cell membrane; Single-pass membrane protein."}, "transmembrane", True, None),
    ("so does one UniProt files under a typed location",
     {"subcellular_location": "Golgi apparatus membrane; Single-pass type II membrane protein."}, "transmembrane",
     True, None),
    ("a peripheral membrane protein has none",
     {"subcellular_location": "Cell membrane; Peripheral membrane protein."}, "transmembrane", False, None),
    ("a bZIP domain is a coiled coil",
     {"features": [("uniprot", "DOMAIN", 10, 60, "bZIP")]}, "coiled_coil", True, None),
    ("the CHCH domain is named for helices",
     {"features": [("uniprot", "DOMAIN", 10, 60, "CHCH (coiled coil-helix-coiled coil-helix)")]}, "coiled_coil",
     False, None),
)


def sequence() -> str:
    """
    The fixture protein's sequence.

    Returns:
        LENGTH residues: alanines, with PLACED residues and the GLYCINES run.
    """
    residues = ["A"] * LENGTH
    for pos, aa in PLACED.items():
        residues[pos - 1] = aa
    for pos in GLYCINES:
        residues[pos - 1] = "G"
    return "".join(residues)


def feature(src: str, kind: str, start: int, end: int, desc: str = "", acc: str | None = None,
            parent: str | None = None, member: str | None = None) -> dict:
    """
    One test-set feature record.

    Args:
        src: Source (uniprot, interpro_match, interpro_site).
        kind: Feature type.
        start: First residue.
        end: Last residue.
        desc: Description.
        acc: InterPro accession.
        parent: Parent InterPro accession.
        member: Member signature.

    Returns:
        The record, with the test set's field names.
    """
    return {"src": src, "type": kind, "start": start, "end": end, "desc": desc, "acc": acc, "parent_acc": parent,
            "member_acc": member}


def protein_row(**overrides) -> dict:
    """
    A test-set row for the fixture protein: a human single-pass membrane protease with a transmembrane helix at 100 to
    122, a catalytic Ser195, a zinc-binding His57, a coiled coil at 200 to 240, a mature chain from residue 40 (Thr) and
    InterPro entry IPR000001 (parent IPR000002, member signature PF00001).

    Args:
        **overrides: Fields to replace; features given as (src, type, start, end, desc) tuples.

    Returns:
        The row.
    """
    row = {"protein_id": PROTEIN, "sequence": sequence(), "length": float(LENGTH), "taxid": 9606.0,
           "reviewed": "Reviewed", "subcellular_location": "Membrane; Single-pass membrane protein.",
           "protein_names": "Test protease (EC 3.4.21.4)", "comments": [], "interpro_formatted": "",
           "interpro_formatted_iscan": "", "interpro_ids": ["IPR000001"],
           "features": [feature("uniprot", "CHAIN", 40, LENGTH), feature("uniprot", "TRANSMEM", 100, 122),
                        feature("uniprot", "ACT_SITE", 195, 195),
                        feature("uniprot", "BINDING", 57, 57, "Zn(2+) ChEBI:CHEBI:29105"),
                        feature("uniprot", "COILED", 200, 240),
                        feature("interpro_match", "domain", 130, 250, "Test domain", "IPR000001", "IPR000002",
                                "PF00001")]}
    for key, value in overrides.items():
        row[key] = [feature(*item) for item in value] if key == "features" else value
    return row


def record(text: str, length: int = LENGTH) -> dict:
    """
    An eval.py record of the fixture protein.

    Args:
        text: The generation.
        length: The sequence length the record reports.

    Returns:
        The record.
    """
    return {"protein_id": PROTEIN, "protein_sequence": sequence(), "sequence_length": length,
            "generated_response": text, "input_prompt": ""}


def truth() -> dict:
    """
    The fixture protein's truth.

    Returns:
        grounding.build_truth of protein_row().
    """
    return grounding.build_truth(protein_row(), PARENTS, REFERENCE)


def test_residue_rules() -> None:
    """Claims about another molecule, names and counting runs are not counted; the counted ones are checked."""
    for text, expected in RESIDUE_CASES:
        claims = grounding.residue_claims(record(text), truth())
        got = [(c["aa"], c["pos"], c["category"], c["correct"]) for c in claims]
        assert got == expected, (text, got)
    mature = grounding.residue_claims(record("The mature chain begins with Thr1."), truth())
    assert mature[0]["rule"] == "mature_numbering"


def test_presence_regex_reader() -> None:
    """Denials, partners, GO names, dismissed predictions and hypotheticals assert nothing."""
    for text, name, said in PRESENCE_CASES:
        assert grounding.presence(record(text), truth())[name]["said"] is said, (text, name)


def test_presence_truth() -> None:
    """The truth fixes and the protein-level evidence decide whether a protein has each presence type."""
    for label, overrides, name, has, note in TRUTH_CASES:
        found = grounding.build_truth(protein_row(**{"features": [], "protein_names": "Test protein",
                                                     "subcellular_location": "Secreted.", **overrides}),
                                      PARENTS, REFERENCE)
        assert grounding.presence_truth(found, name) is has, label
        if note:
            assert note in found["notes"].get(name, []), label


def test_spans() -> None:
    """A claimed span is scored by its distance to the nearest true feature of its type."""
    for text, name, start, end, distance in SPAN_CASES:
        scored, _ = grounding.spans(record(text), truth())
        found = [c for c in scored if (c["type"], c["start"], c["end"]) == (name, start, end)]
        assert found and found[0]["distance"] == distance, (text, [(c["type"], c["start"], c["distance"]) for c in
                                                                   scored])
    assert grounding.region_distance((8, 18), (1, 23)) == 0
    assert grounding.region_distance((90, 130), (100, 122)) == 10


def test_interpro() -> None:
    """A parent and a retired entry with a matching signature are right; degenerate runs and no-entry proteins not."""
    text = "Domains: IPR000001, IPR000002, IPR999999 and IPR000003."
    verdicts = {i["acc"]: i["verdict"] for i in grounding.interpro(record(text), truth(), REFERENCE)}
    assert verdicts == {"IPR000001": "matched", "IPR000002": "parent", "IPR999999": "retired", "IPR000003": "false"}
    assert all(i["verdict"] == "false" for i in grounding.interpro(record(text), None, REFERENCE))
    run = "Entries " + ", ".join(f"IPR0501{n:02d}" for n in range(20, 4, -1)) + "."
    assert grounding.interpro(record(run), truth(), REFERENCE) == []
    assert len(grounding.interpro(record(run), truth(), REFERENCE, skip_runs=False)) == 16


def test_go_id_after() -> None:
    """The GO-name check agrees with its regex, and a hyphen-joined sequence after a keyword returns at once."""
    cases = [" plasma membrane GO:0005886", " (GO:0005886)", "-binding GO:0003677", " foo-GO:0000001",
             " the membrane GO:0005886", " membrane, GO:0005886", "", " M-S-Q", "GO:0005886", " go:0005886"]
    for text in cases:
        assert grounding.go_id_after(text, 0) == bool(grounding.RE_GO_AFTER.match(text, 0)), text
    run = " " + "-".join("MSQKLV" * 700) + ". It binds DNA."
    assert grounding.go_id_after(run, 0) is False


def test_truncation_cap() -> None:
    """A protein at or past the truncation cap is scored for nothing but InterPro."""
    text = "The catalytic Ser195 sits in a transmembrane helix at 100 to 122."
    long_record = record(text, grounding.rc.TRUNCATION_CAP)
    assert not grounding.usable(long_record, truth())
    assert grounding.residue_claims(long_record, truth()) == []
    assert grounding.presence(long_record, truth()) == {}


def test_presence_llm_offline() -> None:
    """The frozen prompt, the reply parser and the cache, without calling the endpoint."""
    frozen = hashlib.md5((grounding.PROMPT_TEMPLATE + json.dumps(grounding.SCHEMA, sort_keys=True)).encode())
    assert grounding.PROMPT_VERSION == "p3" and frozen.hexdigest()[:8] == "27aaa9bd"
    prompt = grounding.build_prompt("<think>\nreasoning\n</think>\nanswer")
    assert "<think>" not in prompt and "<reasoning>\nreasoning\n</reasoning>\nanswer" in prompt
    verdict = {name: {"quote": "", "asserted": name == "transmembrane"} for name in grounding.TYPES}
    assert grounding.extract_json("```json\n" + json.dumps(verdict) + "\n```") == verdict
    assert grounding.validate(verdict) == [] and grounding.validate({}) != []
    with tempfile.TemporaryDirectory() as tmp:
        path = grounding.cache_file(Path(tmp), "run", PROTEIN, "max")
        path.parent.mkdir(parents=True)
        cached = {"prompt_md5": grounding.PROMPT_MD5, "generation_md5": grounding.text_md5("a text"),
                  "verdict": verdict, "problems": []}
        path.write_text(json.dumps(cached))
        said = grounding.cached_said("run", {PROTEIN: "a text"}, "max", Path(tmp))
        assert said == {PROTEIN: {name: name == "transmembrane" for name in grounding.TYPES}}
        assert grounding.cached_said("run", {PROTEIN: "another text"}, "max", Path(tmp)) == {}
        path.write_text(json.dumps({**cached, "prompt_md5": "00000000"}))
        try:
            grounding.cached_said("run", {PROTEIN: "a text"}, "max", Path(tmp))
        except RuntimeError:
            pass
        else:
            raise AssertionError("a reading from an edited prompt must not load")


def test_grounding_eval_tables() -> None:
    """One record's table row and the pooled metrics, with an LLM reading that says TM and coiled coil."""
    text = ("The catalytic Ser195 and the wrong Ser-196. A single transmembrane helix at 100 to 122, a coiled coil at "
            "200 to 240. Domains: IPR000001 and IPR000003.")
    flags = {name: name in ("transmembrane", "coiled_coil") for name in grounding.TOOL_TYPES}
    grounding.install({PROTEIN: truth()}, REFERENCE, {PROTEIN: flags})
    row, spans = grounding.score(record(text))
    assert (row["rip_n"], row["rip_hit"], row["ipr_said"], row["ipr_correct"], row["presence_read"]) == (2, 1, 2, 1, 1)
    assert (row["exist_transmembrane_tp"], row["exist_signal_peptide_said"], row["exist_active_site_said"]) == (1, 0, 0)
    assert row["regex_exist_active_site_tp"] == 1
    assert ([PROTEIN, "transmembrane", 100, 122, 0, "1", 0, "1"] in spans)
    values = grounding.summarise(grounding.table_arrays([row], spans), 0, 0)
    assert values["residue_accuracy"]["value"] == 0.5 and values["interpro_precision"]["value"] == 0.5
    assert values["presence_f1_tool"]["value"] == 1.0
    assert 0.0 < values["span_within_5"]["value"] <= 1.0


def run_command(argv: list[str]) -> int:
    """
    Run grounding.py's command line with these arguments.

    Args:
        argv: The arguments after the script name.

    Returns:
        The command's exit status.
    """
    saved = sys.argv
    sys.argv = ["grounding.py", *argv]
    try:
        return grounding.main()
    finally:
        sys.argv = saved


def test_commands() -> None:
    """`read` and `score` end to end on one exported record with a cached reading, so the endpoint is never called."""
    text = "A single transmembrane helix at 100 to 122, and a coiled coil at 200 to 240."
    verdict = {name: {"quote": "", "asserted": name in ("transmembrane", "coiled_coil")} for name in grounding.TYPES}
    saved_model = grounding.MODEL
    saved_env = {name: os.environ.get(name) for name in (grounding.API_KEY_VAR, grounding.BASE_URL_VAR)}
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        test_set = folder / "test-00000-of-00001.parquet"
        pq.write_table(pa.Table.from_pylist([protein_row()]), test_set)
        export = folder / "run.jsonl.gz"
        with gzip.open(export, "wt") as handle:
            handle.write(json.dumps(record(text)) + "\n")
        cache = folder / "cache"
        path = grounding.cache_file(cache, "run", PROTEIN, "max")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"prompt_md5": grounding.PROMPT_MD5, "generation_md5": grounding.text_md5(text),
                                    "verdict": verdict, "problems": []}))
        try:
            grounding.MODEL = "a-model"
            os.environ[grounding.API_KEY_VAR] = "unused"
            os.environ[grounding.BASE_URL_VAR] = "http://127.0.0.1:9/v1"
            assert run_command(["read", str(export), "--cache-dir", str(cache), "--workers", "1"]) == 0
        finally:
            grounding.MODEL = saved_model
            for name, value in saved_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        out = folder / "out"
        assert run_command(["score", str(export), "--cache-dir", str(cache), "--test-set", str(test_set),
                            "--out", str(out), "--workers", "1", "--bootstrap", "0"]) == 0
        summary = json.loads((out / "run.summary.json").read_text())
        assert summary["counts"] == {"records": 1, "test_proteins": 1, "usable": 1, "llm_read": 1}
        assert summary["metrics"]["presence_f1_transmembrane"]["value"] == 1.0
        assert (out / "run.csv.gz").exists() and (out / "run.spans.csv.gz").exists()


def test_load_records() -> None:
    """A results directory yields its first-sample unified-aspect files; a protein given twice is refused."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for name in ("P1_all_k00.json", "P2_all_k00.json", "P1_all_k01.json"):
            (folder / name).write_text(json.dumps({"protein_id": name.split("_")[0], "generated_response": "x"}))
        (folder / "evaluation_errors_chunk000.log").write_text("[]")
        assert [r["protein_id"] for r in grounding.load_records(folder)] == ["P1", "P2"]
        export = folder / "run.jsonl.gz"
        with gzip.open(export, "wt") as handle:
            handle.write(json.dumps({"protein_id": "P1"}) + "\n" + json.dumps({"protein_id": "P1"}) + "\n")
        try:
            grounding.load_records(export)
        except ValueError:
            pass
        else:
            raise AssertionError("a protein given twice must be refused")
        assert grounding.run_name(export) == "run"


if __name__ == "__main__":
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"{len(tests)} of {len(tests)} pass")
