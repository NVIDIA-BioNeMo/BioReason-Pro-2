# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""CAFA F1 and IA-weighted F1 of unscored GO predictions, the validation metric.

The CAFA evaluation at the one threshold a set of unscored predictions has. Ground truth and
predictions are propagated to their ancestors over `is_a` and `part_of` within a namespace;
alternative ids resolve to their primary terms; obsolete and unknown ids count for nothing. Per
namespace, precision is averaged over the proteins with a prediction there and recall over every
protein with ground truth there, and F1 is their harmonic mean. The weighted scores weigh each term
by its information accretion (IA) and count only terms with IA > 0. A namespace in which no protein
has a prediction is left out.
"""

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import obonet

NAMESPACE_TO_ASPECT = {
    "molecular_function": "MF",
    "biological_process": "BP",
    "cellular_component": "CC",
}

#: The relations propagation follows.
RELATIONS = ("is_a", "part_of")


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
        Namespace name -> Namespace, for the namespaces holding non-obsolete terms.
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


def by_namespace(pairs: list[tuple[str, str]], namespaces: dict[str, Namespace]) -> dict[str, dict[str, set[int]]]:
    """Group (target, GO id) pairs by namespace and target, keeping first-seen target order.

    Args:
        pairs: (target id, GO id) pairs.
        namespaces: The parsed ontology.

    Returns:
        Namespace -> target -> term indices; ids that resolve to nothing are dropped.
    """
    out: dict[str, dict[str, set[int]]] = {}
    for target, term in pairs:
        name, indices = resolve(term, namespaces)
        if name is not None:
            out.setdefault(name, {}).setdefault(target, set()).update(indices)
    return out


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


def mass(indices: set[int], ia: np.ndarray | None) -> float:
    """Size of a term set: its count, or its summed IA over terms with IA > 0.

    Args:
        indices: Term indices.
        ia: IA weights, or None to count.

    Returns:
        The size.
    """
    if ia is None:
        return float(len(indices))
    values = ia[np.array(sorted(indices), dtype=int)] if indices else np.zeros(0)
    return float(values[values > 0].sum())


def f_measure(pairs: list[tuple[set[int], set[int]]], ia: np.ndarray | None) -> float:
    """F1 of macro-averaged precision and recall over one namespace's proteins.

    Args:
        pairs: (propagated truth, propagated prediction) per protein with ground truth.
        ia: IA weights for the weighted score, or None.

    Returns:
        The F1; precision is 0 when no protein has a (weighted) prediction.
    """
    precision = np.zeros(len(pairs))
    recall = np.zeros(len(pairs))
    n_predicted = 0
    for i, (truth, predicted) in enumerate(pairs):
        hit = mass(truth & predicted, ia)
        size_pred, size_true = mass(predicted, ia), mass(truth, ia)
        if size_pred > 0:
            precision[i] = hit / size_pred
            n_predicted += 1
        if size_true > 0:
            recall[i] = hit / size_true
    pr = precision.sum() / n_predicted if n_predicted else 0.0
    rc = recall.sum() / len(pairs)
    return float(2 * pr * rc / (pr + rc)) if pr + rc > 0 else 0.0


def evaluate_binary(gt: list[tuple[str, str]], pred: list[tuple[str, str]], obo_file: str, ia_file: str) -> dict:
    """F1 and IA-weighted F1 per aspect for unscored predictions.

    Args:
        gt: (target id, GO id) pairs of the ground truth.
        pred: (target id, GO id) pairs predicted.
        obo_file: GO in OBO format.
        ia_file: IA weights.

    Returns:
        {"MF": {"f": .., "f_w": ..}, ...} for the aspects in which some protein has a prediction.
    """
    namespaces = load_ontology(str(obo_file), str(ia_file))
    truth = by_namespace(gt, namespaces)
    predicted = by_namespace(pred, namespaces)
    results = {}
    for name, targets in truth.items():
        ns = namespaces[name]
        guesses = predicted.get(name, {})
        pairs = [(propagate(ns, terms), propagate(ns, guesses.get(target, set()))) for target, terms in targets.items()]
        if not any(p for _, p in pairs):
            continue
        results[NAMESPACE_TO_ASPECT.get(name, name)] = {"f": f_measure(pairs, None), "f_w": f_measure(pairs, ns.ia)}
    return results
