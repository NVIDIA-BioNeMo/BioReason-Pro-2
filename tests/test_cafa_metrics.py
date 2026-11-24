# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The CAFA scorer on a hand-built ontology, every value worked out by hand."""

import math

import pytest

from evals.cafa import cafa_metrics

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

#: P1 {1,2,3}, P2 {1,4} in MF; P3 {10,11} in BP; P4's only term is obsolete, so P4 has no truth.
TRUTH = "P1\tGO:0000003\nP2\tGO:0000004\nP3\tGO:0000011\nP4\tGO:0000005\n"

#: P1 {1,2} via the alt id (its 0.5 line is under the threshold), P2 {1,2,3}; P3's id is unknown and
#: P9 has no truth, so neither counts and BP has no prediction in this file.
PREDICTIONS = ("P1\tGO:0000009\t1.0\nP1\tGO:0000004\t0.5\nP2\tGO:0000003\t1.0\n"
               "P3\tGO:9999999\t1.0\nP9\tGO:0000004\t1.0\n")


@pytest.fixture
def run(tmp_path):
    """Score PREDICTIONS (a.tsv) and a perfect BP prediction (sub/b.tsv) against TRUTH.

    Args:
        tmp_path: pytest's fixture.

    Returns:
        cafa_eval's (table, best rows).
    """
    (tmp_path / "go.obo").write_text(OBO)
    (tmp_path / "ia.tsv").write_text(IA)
    (tmp_path / "gt.tsv").write_text(TRUTH)
    (tmp_path / "pred" / "sub").mkdir(parents=True)
    (tmp_path / "pred" / "a.tsv").write_text(PREDICTIONS)
    (tmp_path / "pred" / "sub" / "b.tsv").write_text("P3\tGO:0000011\t1.0\n")
    return cafa_metrics.cafa_eval(str(tmp_path / "go.obo"), str(tmp_path / "pred"), str(tmp_path / "gt.tsv"),
                                  str(tmp_path / "ia.tsv"))


def test_unweighted_scores(run) -> None:
    """MF of a.tsv: P1 pred 2 / true 3 / hit 2, P2 pred 3 / true 2 / hit 1."""
    row = run[0].loc[("a.tsv", "molecular_function", 0.99)]
    assert row["n"] == 2 and row["cov"] == 1.0
    assert row["pr"] == pytest.approx((2 / 2 + 1 / 3) / 2)
    assert row["rc"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert row["f"] == pytest.approx(28 / 45)
    assert (row["tp"], row["fp"], row["fn"]) == (1.5, 1.0, 1.0)
    assert (row["mi"], row["ru"]) == (1.0, 1.0) and row["s"] == pytest.approx(math.sqrt(2))
    assert row["pr_micro"] == pytest.approx(0.6) and row["rc_micro"] == pytest.approx(0.6)
    assert row["f_micro"] == pytest.approx(0.6)


def test_weighted_scores(run) -> None:
    """Weighted MF: P1 pred {2}=1 / true {2,3}=3 / hit 1; P2 pred {2,3}=3 / true {4}=4 / hit 0."""
    row = run[0].loc[("a.tsv", "molecular_function", 0.99)]
    assert row["n_w"] == 2 and row["cov_w"] == 1.0
    assert row["pr_w"] == pytest.approx(0.5) and row["rc_w"] == pytest.approx(1 / 6)
    assert row["f_w"] == pytest.approx(0.25)
    assert (row["tp_w"], row["fp_w"], row["fn_w"]) == (0.5, 1.5, 3.0)
    assert row["s_w"] == pytest.approx(math.sqrt(3.0 ** 2 + 1.5 ** 2))
    assert row["pr_micro_w"] == pytest.approx(0.25) and row["rc_micro_w"] == pytest.approx(1 / 7)
    assert row["f_micro_w"] == pytest.approx(2 / 11)


def test_layout(run) -> None:
    """One row per (file, namespace with a prediction), the evaluator's columns, best rows sorted by key."""
    table, best = run
    assert list(table.index.names) == ["filename", "ns", "tau"]
    assert list(table.index) == [("a.tsv", "molecular_function", 0.99), ("sub_b.tsv", "biological_process", 0.99)]
    assert list(table.columns) == [*cafa_metrics.COLUMNS, *(f"{c}_w" for c in cafa_metrics.COLUMNS)]
    assert set(best) == {"f", "f_w", "s", "f_micro", "f_micro_w"}
    for metric, rows in best.items():
        assert list(rows.index) == list(table.index)
        assert list(rows["cov_max"]) == list(rows["cov_w" if metric.endswith("_w") else "cov"])


def test_cross_namespace_edge_is_dropped(run) -> None:
    """x's part_of into MF does not propagate: a perfect BP prediction scores 1 and adds nothing to MF."""
    row = run[0].loc[("sub_b.tsv", "biological_process", 0.99)]
    assert row["f"] == 1.0 and row["f_w"] == 1.0 and row["s"] == 0.0


def test_no_prediction_with_truth(tmp_path) -> None:
    """Nothing to score gives (None, {})."""
    (tmp_path / "go.obo").write_text(OBO)
    (tmp_path / "ia.tsv").write_text(IA)
    (tmp_path / "gt.tsv").write_text(TRUTH)
    (tmp_path / "pred").mkdir()
    (tmp_path / "pred" / "a.tsv").write_text("P9\tGO:0000004\t1.0\n")
    assert cafa_metrics.cafa_eval(str(tmp_path / "go.obo"), str(tmp_path / "pred"), str(tmp_path / "gt.tsv"),
                                  str(tmp_path / "ia.tsv")) == (None, {})
