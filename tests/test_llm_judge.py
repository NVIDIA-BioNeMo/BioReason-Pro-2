# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The eval judge's prompt, verdict parsing and retry passes."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.judge import llm_judge  # noqa: E402

ROW = {"protein_id": "P12345", "sequence_numbered": "1 M\n2 C\n3 K",
       "generation": "<think>\nThe chain is 3 residues.\n</think>\n\nCys2 is the nucleophile."}


def complete_verdict() -> dict:
    """A verdict scoring every axis.

    Returns:
        The verdict.
    """
    return {**{a: {"score": 7, "rationale": "x"} for a in llm_judge.AXES},
            "unsupported_claims": [], "summary": "ok"}


def test_prompt_shows_the_sequence_and_the_generation_once() -> None:
    """The prompt carries the accession, every numbered residue, and the generation once."""
    prompt = llm_judge.build_prompt(ROW, None, 60)
    assert "UniProt Accession: P12345" in prompt and "\n1 M\n2 C\n3 K\n" in prompt
    assert prompt.count("Cys2 is the nucleophile.") == 1


def test_prompt_has_no_think_tags() -> None:
    """The generation's reasoning tags, which Nemotron reads as its own, are renamed."""
    prompt = llm_judge.build_prompt(ROW, None, 60)
    assert "<think>" not in prompt and "</think>" not in prompt
    assert llm_judge.rename_think_tags(ROW["generation"]) in prompt
    assert "<reasoning> block" in prompt


def test_extract_json_tolerates_fences_and_preambles() -> None:
    """A chat endpoint may ignore response_format; the verdict is still found."""
    assert llm_judge.extract_json('{"a": 1}') == {"a": 1}
    assert llm_judge.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm_judge.extract_json('Here it is: {"a": {"b": "}"}} done') == {"a": {"b": "}"}}
    assert llm_judge.extract_json("no object here") is None


def test_validate_flags_missing_and_out_of_range_axes() -> None:
    """A complete verdict is clean; a missing axis or a score of 11 disqualifies it."""
    verdict = complete_verdict()
    assert llm_judge.validate(verdict) == []
    del verdict["mf"]
    verdict["bp"]["score"] = 11
    assert len(llm_judge.hard_problems(llm_judge.validate(verdict))) == 2


class StubJudge:
    """A judge stand-in, with `call_judge`'s signature, whose first verdict for P2 misses an axis."""

    def __init__(self) -> None:
        """Start with no calls."""
        self.calls = {}

    def __call__(self, prompt, model, effort, base_url=None, timeout=900, max_output_tokens=16000):
        """One verdict.

        Returns:
            (text, usage, reasoning), as `call_judge` returns them.
        """
        pid = prompt.split("UniProt Accession: ")[1].split("\n")[0]
        self.calls[pid] = self.calls.get(pid, 0) + 1
        verdict = complete_verdict()
        if pid == "P2" and self.calls[pid] == 1:
            del verdict["reliability"]
        return json.dumps(verdict), {"reasoning_tokens": 5}, "thinking"


def test_incomplete_verdicts_are_judged_again(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A verdict missing an axis is set aside and re-judged; a complete one is judged once."""
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps({"protein_id": p, "sequence_numbered": "1 M", "generation": "g"}) + "\n"
                            for p in ("P1", "P2")))
    judge = StubJudge()
    monkeypatch.setattr(llm_judge, "call_judge", judge)
    monkeypatch.setattr(sys, "argv", ["llm_judge.py", "--rows-jsonl", str(rows), "--out", str(tmp_path / "run"),
                                      "--model", "m", "--workers", "2"])
    assert llm_judge.main() == 0
    assert judge.calls == {"P1": 1, "P2": 2}
    assert sorted(os.listdir(tmp_path / "run" / "_retry")) == ["P2.pass1.json"]
    assert all(llm_judge.scores_every_axis(str(tmp_path / "run" / f"{p}.json")) for p in ("P1", "P2"))
