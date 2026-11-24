# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""End-to-end check of the released model (downloads ESM2 and the checkpoint, ~3.4 GB).

    pytest -m slow
    GOGPT_MODEL_DIR=/path/to/exported/model pytest -m slow    # a local export instead
"""

import os
import re

import pytest

pytestmark = pytest.mark.slow

# Human hemoglobin subunit alpha (P69905)
HBA = ("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSD"
       "LHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR")


@pytest.fixture(scope="module")
def predictor():
    from gogpt import GOGPTPredictor

    if os.environ.get("GOGPT_MODEL_DIR"):
        return GOGPTPredictor(os.environ["GOGPT_MODEL_DIR"], device="cpu")
    return GOGPTPredictor.from_pretrained(os.environ.get("GOGPT_MODEL_ID", "wanglab/gogpt-v2"), device="cpu")


def test_predicts_known_function(predictor):
    pred = predictor.predict(HBA, organism=9606)
    assert set(pred) == {"MF", "BP", "CC"}
    for terms in pred.values():
        assert all(re.fullmatch(r"GO:\d{7}", t) for t in terms)
        assert len(terms) == len(set(terms))
    # Hemoglobin: oxygen carrier activity (MF) and hemoglobin complex (CC)
    assert "GO:0005344" in pred["MF"]
    assert "GO:0005833" in pred["CC"]


def test_batch_matches_single(predictor):
    seqs = [HBA, HBA[:60], "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUAU"]
    orgs = [9606, "Homo sapiens", None]
    batch = predictor.predict_batch(seqs, orgs, batch_size=3)
    for s, o, b in zip(seqs, orgs, batch):
        assert predictor.predict(s, o) == b


def test_organism_name_and_taxid_agree(predictor):
    assert predictor.predict(HBA, "9606") == predictor.predict(HBA, "Homo sapiens (Human)")


def test_unknown_organism_warns(predictor):
    with pytest.warns(UserWarning, match="not one of the model's"):
        predictor.predict(HBA[:50], organism="Not an organism")


def test_rejects_invalid_sequence(predictor):
    with pytest.raises(ValueError, match="cannot encode"):
        predictor.predict("MKT1AYI")
