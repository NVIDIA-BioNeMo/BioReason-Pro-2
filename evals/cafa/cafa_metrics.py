# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""CAFA evaluation of unscored GO predictions: precision, recall, F, S and their IA-weighted forms.

Every predicted term carries score 1.0, so the evaluation has one threshold. Ground truth and
predictions are propagated to their ancestors over `is_a` and `part_of` within a namespace;
alternative ids resolve to their primary terms; obsolete and unknown ids count for nothing. Per
namespace, over the proteins with ground truth there:

- precision is averaged over the proteins with a prediction, recall over all of them, and F is the
  harmonic mean of the two;
- remaining uncertainty (ru) and misinformation (mi) are the missed and the wrongly predicted terms
  per protein, and S is their Euclidean norm;
- the micro scores pool true and false positives over proteins before dividing;
- coverage (cov) is the share of proteins with a prediction.

The unweighted scores count every term once. The weighted ones (suffix `_w`) weigh each term by its
information accretion (IA) and count only terms with IA > 0. Per-protein sums run over dense rows in
term order, so every number equals the CAFA evaluator's to the last bit. The tables have its layout:
one row per (prediction file, namespace, threshold), and the best row per metric with `cov_max`.
"""

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import obonet
import pandas as pd

#: The relations propagation follows.
RELATIONS = ("is_a", "part_of")

#: The one threshold: every prediction is at 1.0, and the tables report it as the threshold column.
THRESHOLD = 0.99

#: The score columns, in table order; the weighted ones follow with the suffix `_w`.
COLUMNS = ("n", "tp", "fp", "fn", "pr", "rc", "cov", "mi", "ru", "f", "s", "pr_micro", "rc_micro", "f_micro")

#: metric -> whether larger is better, for the best-row tables.
BEST = {"f": True, "f_w": True, "s": False, "f_micro": True, "f_micro_w": True}

#: Proteins per dense block, bounding memory to a few hundred MB on the biggest namespace.
BLOCK = 512


@dataclass
class Namespace:
    """One GO namespace.

    Attributes:
        index: GO id -> term index, in the OBO file's order.
        alt: Alternative id -> the primary ids that list it.
        parents: Term index -> its parents' indices (same namespace only).
        ia: Term index -> IA; 0 where the IA file has none or a non-finite value.
        ancestors: Memo of `propagate`'s per-term closures.
    """

    index: dict[str, int] = field(default_factory=dict)
    alt: dict[str, set[str]] = field(default_factory=dict)
    parents: list[list[int]] = field(default_factory=list)
    ia: np.ndarray | None = None
    ancestors: dict[int, frozenset[int]] = field(default_factory=dict)


def read_ia(ia_file: str) -> dict[str, float]:
    """IA per GO id from a two-column, whitespace-separated file.

    Args:
        ia_file: The IA file.

    Returns:
        GO id -> IA, non-finite values as 0.
    """
    ia = {}
    with open(ia_file) as handle:
        for line in handle:
            if line.strip():
                term, value = line.split()
                ia[term] = float(value) if math.isfinite(float(value)) else 0.0
    return ia


@lru_cache(maxsize=4)
def load_ontology(obo_file: str, ia_file: str) -> dict[str, Namespace]:
    """Parse the ontology and IA once per process.

    Args:
        obo_file: GO in OBO format.
        ia_file: IA weights.

    Returns:
        Namespace name -> Namespace, in the order the OBO file first uses each namespace.
    """
    graph = obonet.read_obo(obo_file, ignore_obsolete=True)
    ia = read_ia(ia_file)
    namespaces: dict[str, Namespace] = {}
    weights: dict[str, list[float]] = {}
    for term, data in graph.nodes(data=True):
        name = data.get("namespace")
        if name is None or name == "external":
            continue
        ns = namespaces.setdefault(name, Namespace())
        ns.index[term] = len(ns.index)
        weights.setdefault(name, []).append(ia.get(term, 0.0))
        for alt in data.get("alt_id", []):
            ns.alt.setdefault(alt, set()).add(term)
    for name, ns in namespaces.items():
        ns.ia = np.array(weights[name])
        ns.parents = [[ns.index[parent] for _, parent, relation in graph.out_edges(term, keys=True)
                       if relation in RELATIONS and parent in ns.index]
                      for term in ns.index]
    return namespaces


def resolve(term: str, namespaces: dict[str, Namespace]) -> tuple[str | None, list[int]]:
    """A GO id's namespace and the term indices it stands for.

    Args:
        term: A GO id, primary or alternative.
        namespaces: The parsed ontology.

    Returns:
        (namespace, indices); (None, []) for an obsolete or unknown id.
    """
    for name, ns in namespaces.items():
        if term in ns.index:
            return name, [ns.index[term]]
        if term in ns.alt:
            return name, [ns.index[primary] for primary in sorted(ns.alt[term])]
    return None, []


def propagate(ns: Namespace, indices: set[int]) -> set[int]:
    """The terms and all their ancestors.

    Args:
        ns: The terms' namespace.
        indices: Term indices.

    Returns:
        The propagated index set.
    """
    out: set[int] = set()
    for index in indices:
        if index not in ns.ancestors:
            seen, stack = {index}, [index]
            while stack:
                for parent in ns.parents[stack.pop()]:
                    if parent not in seen:
                        seen.add(parent)
                        stack.append(parent)
            ns.ancestors[index] = frozenset(seen)
        out |= ns.ancestors[index]
    return out


def read_truth(path: str, namespaces: dict[str, Namespace]) -> dict[str, dict[str, set[int]]]:
    """Ground truth by namespace and protein, proteins in first-seen order.

    Args:
        path: `protein<TAB>GO id` lines.
        namespaces: The parsed ontology.

    Returns:
        Namespace -> protein -> term indices; ids that resolve to nothing are dropped.
    """
    truth: dict[str, dict[str, set[int]]] = {}
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 2:
                continue
            name, indices = resolve(fields[1], namespaces)
            if name is not None:
                truth.setdefault(name, {}).setdefault(fields[0], set()).update(indices)
    return truth


def read_predictions(path: str, namespaces: dict[str, Namespace],
                     truth: dict[str, dict[str, set[int]]]) -> dict[str, dict[str, set[int]]]:
    """Predictions at or above the threshold, for proteins with ground truth in the term's namespace.

    Args:
        path: `protein<TAB>GO id<TAB>score` lines.
        namespaces: The parsed ontology.
        truth: `read_truth`'s result.

    Returns:
        Namespace -> protein -> term indices.
    """
    predicted: dict[str, dict[str, set[int]]] = {}
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 3 or float(fields[2]) < THRESHOLD:
                continue
            name, indices = resolve(fields[1], namespaces)
            if name is not None and fields[0] in truth.get(name, {}):
                predicted.setdefault(name, {}).setdefault(fields[0], set()).update(indices)
    return predicted


def dense(rows: list[set[int]], n_terms: int) -> np.ndarray:
    """Boolean protein x term matrix.

    Args:
        rows: Each protein's term indices.
        n_terms: Terms in the namespace.

    Returns:
        The matrix.
    """
    out = np.zeros((len(rows), n_terms), dtype=bool)
    for i, row in enumerate(rows):
        out[i, sorted(row)] = True
    return out


def per_protein(truth: list[set[int]], predicted: list[set[int]], n_terms: int,
                columns: np.ndarray, weights: np.ndarray | None) -> dict[str, np.ndarray]:
    """Per-protein sizes of the predicted, true, true-positive, false-positive and missed sets.

    Args:
        truth: Propagated ground truth per protein.
        predicted: Propagated predictions per protein, empty where there are none.
        n_terms: Terms in the namespace.
        columns: The terms that count.
        weights: Their IA for the weighted sizes, or None to count.

    Returns:
        Size name -> one value per protein.
    """
    sizes: dict[str, list[np.ndarray]] = {k: [] for k in ("pred", "true", "tp", "fp", "fn")}
    for start in range(0, len(truth), BLOCK):
        g = dense(truth[start:start + BLOCK], n_terms)[:, columns]
        p = dense(predicted[start:start + BLOCK], n_terms)[:, columns]
        for name, block in (("pred", p), ("true", g), ("tp", p & g), ("fp", p & ~g), ("fn", ~p & g)):
            sizes[name].append(block.sum(axis=1) if weights is None else (block * weights).sum(axis=1))
    return {name: np.concatenate(parts) for name, parts in sizes.items()}


def harmonic(precision: float, recall: float) -> float:
    """F, the harmonic mean of precision and recall; 0 when both are 0.

    Args:
        precision: Precision.
        recall: Recall.

    Returns:
        F.
    """
    total = precision + recall
    return 2 * precision * recall / total if total != 0 else 0.0


def scores(sizes: dict[str, np.ndarray]) -> dict[str, float]:
    """The score columns of one namespace from its per-protein sizes.

    Args:
        sizes: `per_protein`'s result.

    Returns:
        Column -> value, for COLUMNS.
    """
    proteins = len(sizes["true"])
    has_pred, has_true = sizes["pred"] > 0, sizes["true"] > 0
    n = float(has_pred.sum())
    precision = np.divide(sizes["tp"], sizes["pred"], out=np.zeros(proteins), where=has_pred).sum()
    recall = np.divide(sizes["tp"], sizes["true"], out=np.zeros(proteins), where=has_true).sum()
    pr = precision / n if n > 0 else 0.0
    rc = recall / proteins
    tp, fp, fn = (float(sizes[k].sum()) / proteins for k in ("tp", "fp", "fn"))
    pr_micro = tp / (tp + fp) if tp + fp > 0 else 0.0
    rc_micro = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "pr": float(pr), "rc": float(rc), "cov": n / proteins,
            "mi": fp, "ru": fn, "f": float(harmonic(pr, rc)), "s": math.sqrt(fn * fn + fp * fp),
            "pr_micro": pr_micro, "rc_micro": rc_micro, "f_micro": harmonic(pr_micro, rc_micro)}


def evaluate_file(path: str, namespaces: dict[str, Namespace],
                  truth: dict[str, dict[str, set[int]]]) -> list[dict[str, float]]:
    """Score one prediction file in every namespace where some protein with ground truth has a prediction.

    Args:
        path: The prediction file.
        namespaces: The parsed ontology.
        truth: `read_truth`'s result.

    Returns:
        One row per namespace: `ns` plus the unweighted and the weighted score columns.
    """
    predicted = read_predictions(path, namespaces, truth)
    rows = []
    for name, ns in namespaces.items():
        if name not in predicted:
            continue
        proteins = list(truth[name])
        g = [propagate(ns, truth[name][p]) for p in proteins]
        p = [propagate(ns, predicted[name].get(protein, set())) for protein in proteins]
        weighted = np.flatnonzero(ns.ia > 0)
        plain = scores(per_protein(g, p, len(ns.index), np.arange(len(ns.index)), None))
        by_ia = scores(per_protein(g, p, len(ns.index), weighted, ns.ia[weighted]))
        rows.append({"ns": name, **plain, **{f"{k}_w": v for k, v in by_ia.items()}})
    return rows


def best_rows(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The best row per (file, namespace) for each metric in BEST, with the coverage at that row.

    Args:
        table: The full table.

    Returns:
        Metric -> its best rows, ordered by (file, namespace).
    """
    out = {}
    grouped = table.groupby(level=["filename", "ns"])
    for metric, larger in BEST.items():
        best = table.loc[grouped[metric].idxmax() if larger else grouped[metric].idxmin()].copy()
        best["cov_max"] = best["cov_w" if metric.endswith("_w") else "cov"]
        out[metric] = best
    return out


def cafa_eval(obo_file: str, pred_dir: str, gt_file: str, ia_file: str) -> tuple[pd.DataFrame | None, dict]:
    """Score every prediction file under a directory against one ground truth.

    Args:
        obo_file: GO in OBO format.
        pred_dir: Directory of prediction files, searched recursively.
        gt_file: The ground truth.
        ia_file: IA weights.

    Returns:
        (the table indexed by (filename, ns, tau), metric -> best rows), or (None, {}) when no
        file has a prediction for a protein with ground truth.
    """
    namespaces = load_ontology(str(obo_file), str(ia_file))
    truth = read_truth(gt_file, namespaces)
    root = os.path.normpath(pred_dir)
    rows = []
    for directory, _, files in sorted(os.walk(root)):
        for name in sorted(files):
            path = os.path.join(directory, name)
            label = os.path.relpath(path, root).replace(os.sep, "_")
            rows += [{"filename": label, "tau": THRESHOLD, **row} for row in evaluate_file(path, namespaces, truth)]
    if not rows:
        return None, {}
    columns = ["filename", "ns", "tau", *COLUMNS, *(f"{c}_w" for c in COLUMNS)]
    table = pd.DataFrame(rows, columns=columns).set_index(["filename", "ns", "tau"])
    return table, best_rows(table)
