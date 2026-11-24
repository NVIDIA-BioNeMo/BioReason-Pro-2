# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The validation metric on a hand-built ontology, every value worked out by hand (no downloads)."""

import pytest

from gogpt.utils.cafa import evaluate_binary

#: MF: root 1 <- a 2 (alt id 9) <- b 3 (part_of), root 1 <- c 4, obsolete 5.
#: BP: root 10 <- x 11, whose part_of into MF crosses namespaces and is dropped.
OBO = """format-version: 1.2

[Term]
id: GO:0000001
name: mf root
namespace: molecular_function

[Term]
id: GO:0000002
name: a
namespace: molecular_function
alt_id: GO:0000009
is_a: GO:0000001 ! mf root

[Term]
id: GO:0000003
name: b
namespace: molecular_function
relationship: part_of GO:0000002 ! a

[Term]
id: GO:0000004
name: c
namespace: molecular_function
is_a: GO:0000001 ! mf root

[Term]
id: GO:0000005
name: old
namespace: molecular_function
is_obsolete: true

[Term]
id: GO:0000010
name: bp root
namespace: biological_process

[Term]
id: GO:0000011
name: x
namespace: biological_process
is_a: GO:0000010 ! bp root
relationship: part_of GO:0000002 ! a

[Typedef]
id: part_of
name: part of
namespace: external
"""

#: Roots carry no IA, so they never count in a weighted score.
IA = "GO:0000001\t0\nGO:0000002\t1.0\nGO:0000003\t2.0\nGO:0000004\t4.0\nGO:0000010\t0\nGO:0000011\t3.0\n"


@pytest.fixture
def files(tmp_path) -> tuple[str, str]:
    """The ontology and IA files.

    Args:
        tmp_path: pytest's fixture.

    Returns:
        (OBO path, IA path).
    """
    (tmp_path / "go.obo").write_text(OBO)
    (tmp_path / "ia.tsv").write_text(IA)
    return str(tmp_path / "go.obo"), str(tmp_path / "ia.tsv")


def test_propagation_alt_ids_and_dropped_ids(files) -> None:
    """P1 truth {1,2,3}, prediction {1,2} via the alt id; P2 has no prediction and counts in recall only."""
    gt = [("P1", "GO:0000003"), ("P2", "GO:0000004")]
    pred = [("P1", "GO:0000009"), ("P1", "GO:0000005"), ("P1", "GO:9999999")]
    res = evaluate_binary(gt, pred, *files)
    assert set(res) == {"MF"}
    assert res["MF"]["f"] == pytest.approx(0.5)        # pr 1, rc (2/3 + 0) / 2
    assert res["MF"]["f_w"] == pytest.approx(2 / 7)    # pr 1, rc (1/3 + 0) / 2


def test_namespace_without_prediction_is_left_out(files) -> None:
    """BP has ground truth and no prediction, so it is not reported."""
    res = evaluate_binary([("P1", "GO:0000002"), ("P3", "GO:0000011")], [("P1", "GO:0000002")], *files)
    assert set(res) == {"MF"} and res["MF"] == {"f": 1.0, "f_w": 1.0}


def test_zero_ia_prediction(files) -> None:
    """A prediction of the root alone counts unweighted but not weighted."""
    res = evaluate_binary([("P4", "GO:0000002")], [("P4", "GO:0000001")], *files)
    assert res["MF"]["f"] == pytest.approx(2 / 3)       # pr 1, rc 1/2
    assert res["MF"]["f_w"] == 0.0


def test_cross_namespace_edge_is_dropped(files) -> None:
    """x's part_of into MF does not propagate: a perfect BP prediction stays perfect, with no MF score."""
    res = evaluate_binary([("P5", "GO:0000011")], [("P5", "GO:0000011")], *files)
    assert res == {"BP": {"f": 1.0, "f_w": 1.0}}


def test_predictions_without_truth_are_ignored(files) -> None:
    """A prediction for a protein, or in a namespace, with no ground truth changes nothing."""
    gt = [("P1", "GO:0000003")]
    alone = evaluate_binary(gt, [("P1", "GO:0000002")], *files)
    extra = evaluate_binary(gt, [("P1", "GO:0000002"), ("P1", "GO:0000011"), ("P9", "GO:0000004")], *files)
    assert alone == extra
