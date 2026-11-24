# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Predict GO terms for the proteins in a FASTA file.

    python scripts/predict.py --fasta proteins.fasta --output predictions.tsv --organism 9606
    python scripts/predict.py --fasta proteins.fasta --organisms organisms.tsv --output predictions.tsv

FASTA ids are the first word of each header (for UniProt headers like
">sp|P69905|HBA_HUMAN ..." the accession is used). Organisms are NCBI taxonomy ids or
organism names: one for all proteins (--organism) or per protein from a two-column TSV
"protein_id<TAB>organism" (--organisms). Proteins without one use the <UNKNOWN>
organism embedding.

Output TSV columns: protein_id, aspect, go_id, go_name, rank (1 = first generated).
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gogpt.inference import DEFAULT_MODEL_ID, GOGPTPredictor  # noqa: E402
from gogpt.utils.go_ontology import load_go_names  # noqa: E402


def read_fasta(path):
    records, pid, chunks = [], None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if pid is not None:
                records.append((pid, "".join(chunks)))
            pid = line[1:].split()[0]
            parts = pid.split("|")
            if len(parts) >= 3 and parts[0] in ("sp", "tr"):
                pid = parts[1]
            chunks = []
        elif line:
            chunks.append(line)
    if pid is not None:
        records.append((pid, "".join(chunks)))
    ids = [r[0] for r in records]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{path}: duplicate protein ids")
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--output", required=True)
    org = ap.add_mutually_exclusive_group()
    org.add_argument("--organism", default=None, help="taxonomy id or name for every protein")
    org.add_argument("--organisms", default=None, help="TSV: protein_id<TAB>taxonomy id or name")
    ap.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hugging Face model id")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--model-dir", default=None, help="local model directory instead of --model")
    ap.add_argument("--embedding-cache-dir", default=None, help="precomputed ESM2 embeddings, used by protein id")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--decoding", choices=["greedy", "beam", "sample"], default="greedy")
    ap.add_argument("--device", default=None)
    ap.add_argument("--precision", choices=["auto", "bf16", "fp32"], default="auto")
    ap.add_argument("--obo", default=str(ROOT / "data" / "go-basic.obo"))
    args = ap.parse_args()

    records = read_fasta(args.fasta)
    if args.organisms:
        with open(args.organisms) as f:
            table = {row[0]: row[1] for row in csv.reader(f, delimiter="\t") if len(row) >= 2}
        organisms = [table.get(pid) for pid, _ in records]
    else:
        organisms = [args.organism] * len(records)

    kwargs = dict(device=args.device, precision=args.precision, embedding_cache_dir=args.embedding_cache_dir)
    if args.model_dir:
        predictor = GOGPTPredictor(args.model_dir, **kwargs)
    else:
        predictor = GOGPTPredictor.from_pretrained(args.model, revision=args.revision, **kwargs)

    predictions = predictor.predict_batch([s for _, s in records], organisms,
                                          protein_ids=[p for p, _ in records],
                                          batch_size=args.batch_size, decoding=args.decoding)
    names = load_go_names(args.obo)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["protein_id", "aspect", "go_id", "go_name", "rank"])
        for (pid, _), pred in zip(records, predictions):
            for aspect, terms in pred.items():
                for rank, go_id in enumerate(terms, 1):
                    w.writerow([pid, aspect, go_id, names.get(go_id, "Unknown"), rank])
    print(f"wrote predictions for {len(records)} proteins to {args.output}")


if __name__ == "__main__":
    main()
