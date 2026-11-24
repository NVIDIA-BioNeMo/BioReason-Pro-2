# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The rendered feature block must contain nothing a novel sequence could not supply.

`+feat` is the condition the benchmark reports and the one a deployment would use, so
everything in that block has to be computable from sequence alone -- InterProScan output
and nothing else. The `features` column it is rendered from is a MERGE: UniProt curated
records (BINDING, ACT_SITE, TRANSMEM, DISULFID, MOD_RES, ...), InterProScan matches, and
PDB structure records. `build_question._scan_only` filters that down, but the filter is a
property of the BUILDER, and a table on disk may have been built by another version.

So this checks the artefact, not the code: it reads the rendered `question` out of the SFT
tables themselves and asserts every coordinate line in the block traces to an
InterProScan-sourced record for that protein.

    BRP_TABLES=data/sft/v2-features,data/sft/v2-nofeatures BRP_PROTEINS=data/v2-ep75/proteins \\
        python tests/test_feature_block_provenance.py
"""
import collections
import glob
import os
import re
import sys

import pyarrow.dataset as ds

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from bioreason2 import paths  # noqa: E402
from bioreason2.dataset import protein_fields as bp  # noqa: E402

#: Sources a novel sequence CAN supply: InterProScan runs on the sequence.
SCAN_SRC = ("interpro_match", "interpro_site")

#: Default tables to audit, whichever of them exist.
DEFAULT_TABLES = os.environ.get("BRP_TABLES", ",".join([
    os.path.join(REPO, "data", "sft", "v2-features"),
    os.path.join(REPO, "data", "sft", "v2-nofeatures"),
]))

PROTEINS = os.environ.get("BRP_PROTEINS", os.path.join(paths.PROMPT_CORPUS, "proteins"))

#: A block line opens with its coordinates: "- 28-485 (FAMILY): ..." or "- 65 (SITE): ...".
LINE = re.compile(r"^- ([\d,\-]+) \(([A-Z_]+)\)", re.MULTILINE)

#: Where the block starts and stops inside the rendered question.
BLOCK_START = "Residue features, every annotated feature:"
BLOCK_END = "Initial GO term speculations:"

#: How many proteins to audit. The block is deterministic in the row, so a few hundred
#: settles it; the whole table is 267k rows and reading it proves nothing extra.
SAMPLE = int(os.environ.get("BRP_SAMPLE", "400"))

_results = []


def check(cond: bool, msg: str) -> bool:
    """Record and print one assertion.

    Args:
        cond: the condition.
        msg: what it asserts.

    Returns:
        The condition.
    """
    _results.append(bool(cond))
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    return bool(cond)


def load_features(sources: str) -> dict:
    """Read the raw `features` records for every protein, keyed by id.

    Args:
        sources: comma-separated directories of protein parquet shards.

    Returns:
        protein_id -> list of feature dicts.
    """
    out = {}
    for src in [s for s in sources.split(",") if s]:
        shards = sorted(glob.glob(os.path.join(src, "*.parquet")))
        if not shards:
            continue
        rows = ds.dataset(shards, format="parquet").to_table(
            columns=["protein_id", "features"]).to_pylist()
        for row in rows:
            out.setdefault(row["protein_id"], row["features"] or [])
    return out


def block_of(question: str) -> str:
    """Extract the residue-feature block from a rendered question.

    Args:
        question: the `question` column.

    Returns:
        The block text, or "" if the question carries no populated block.
    """
    start = question.find(BLOCK_START)
    if start < 0:
        return ""
    end = question.find(BLOCK_END, start)
    return question[start + len(BLOCK_START):end if end > 0 else None]


def audit(table: str, features: dict) -> dict:
    """Decide which render the stored block is, by producing both and comparing.

    Attributing individual bullets by coordinate fails: the renderer merges multi-position
    records ("46,263,459"), dedupes and caps, so a line often corresponds to no single
    record. Re-rendering is exact instead -- the block is deterministic in the row, so
    rendering the same features scan-only and in full answers the question outright.

    Args:
        table: an SFT table directory.
        features: protein_id -> raw feature records.

    Returns:
        Mapping with ``rows``, ``with_block``, ``scan`` (blocks equal to the scan-only
        render), ``full`` (equal to the all-sources render), ``neither``, and
        ``curated_lines`` summed over rows whose block is the full render.
    """
    shards = sorted(glob.glob(os.path.join(table, "data", "*.parquet")))
    if not shards:
        shards = sorted(glob.glob(os.path.join(table, "*.parquet")))
    rows = ds.dataset(shards, format="parquet").to_table(
        columns=["protein_id", "question"]).to_pylist()[:SAMPLE]

    out = collections.Counter()
    for row in rows:
        stored = block_of(row["question"]).strip()
        if not stored:
            continue
        out["with_block"] += 1
        raw = features.get(row["protein_id"], [])
        scan = [f for f in raw if f.get("src") in SCAN_SRC]
        rendered_scan = bp.render_features(scan, cap=bp.FEATURE_CAP).strip()
        rendered_full = bp.render_features(raw, cap=bp.FEATURE_CAP).strip()

        # `build_question._strip_refs` runs over the render before storage, and the cap
        # is retried downward against a token budget, so compare on the set of coordinate
        # lines rather than byte-for-byte.
        keys = lambda b: set(LINE.findall(b))
        stored_keys, scan_keys, full_keys = keys(stored), keys(rendered_scan), keys(rendered_full)
        if stored_keys <= scan_keys:
            out["scan"] += 1
        elif stored_keys <= full_keys:
            out["full"] += 1
            out["curated_lines"] += len(stored_keys - scan_keys)
        else:
            out["neither"] += 1
        out["lines"] += len(stored_keys)
        out["lines_not_in_scan"] += len(stored_keys - scan_keys)
    return {"rows": len(rows), **out}


def main() -> int:
    """Audit every table and report.

    Returns:
        0 if no curated source reached a rendered block, 1 otherwise.
    """
    features = load_features(PROTEINS)
    print("feature records for %d proteins\n" % len(features))

    src_census = collections.Counter()
    for records in features.values():
        for feature in records:
            src_census[feature.get("src")] += 1
    print("sources present in the `features` COLUMN: %s\n" % dict(src_census))

    for table in DEFAULT_TABLES.split(","):
        if not os.path.isdir(table):
            print("SKIP %s (absent)" % table)
            continue
        stats = audit(table, features)
        name = os.path.basename(table.rstrip("/"))
        print("%s: %d rows, %d with a populated block, %d block lines"
              % (name, stats["rows"], stats.get("with_block", 0), stats.get("lines", 0)))
        for key in ("scan", "full", "neither"):
            print("    blocks matching the %-8s render: %d" % (key, stats.get(key, 0)))
        print("    block lines absent from the scan-only render: %d (%.1f%%)"
              % (stats.get("lines_not_in_scan", 0),
                 100 * stats.get("lines_not_in_scan", 0) / max(1, stats.get("lines", 1))))
        check(stats.get("lines_not_in_scan", 0) == 0,
              "%s: every block line is reproducible from InterProScan alone" % name)
        print()

    print("%d/%d passed" % (sum(_results), len(_results)))
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
