#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Predict the function of new proteins with BioReason-Pro 2.

    python predict.py --input proteins.tsv --output predictions.tsv                      # RL model
    python predict.py --input proteins.tsv --output predictions.tsv --interpro --email you@your.org
    python predict.py --input proteins.tsv --output predictions.tsv --model sft

Up to three stages, each cached next to the output and reused with --resume:

  1. InterPro (only with --interpro or --interproscan_dir): InterProScan's member-database
     matches and residue sites, from InterPro's precomputed results or an EBI job
     (`--interpro`) or a local InterProScan (`--interproscan_dir`), converted by
     interpro_api.py into the feature records the model was trained on. Without it the
     prompt's residue-feature section reads "None annotated", which the model was also
     trained on.
  2. GO-GPT predicts GO terms from the sequence and organism: the prompt's "Initial GO term
     speculations".
  3. The question is rendered by scripts/data/build_question.py, the builder of every training
     prompt, and the model generates through eval.py's batched path.

Input TSV (tab-separated, with header): protein_id, organism, sequence; optional columns
`taxid` (NCBI taxonomy id, which GO-GPT prefers to the name), `partners` (known interaction
partners as names separated by "~") and `structure_path` (a PDB or mmCIF file, read by ESM3).
Output TSV: protein_id, organism, sequence_length, features, gogpt, question,
generated_response.

Runs in the inference environment (README, Setup) on one GPU.
"""

import argparse
import csv
import gc
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import eval as brp_eval  # noqa: E402
from bioreason2 import paths  # noqa: E402
from bioreason2.dataset.cafa5.format import format_cafa5_for_protein_llm  # noqa: E402
from bioreason2.dataset.cafa5.load import _format_reasoning_prompt  # noqa: E402
from gogpt_api import format_go_output, load_predictor  # noqa: E402
from interpro_api import scan_many  # noqa: E402

#: The prompt builder and template every published checkpoint was trained and scored with.
QUESTION_BUILDER = REPO_ROOT / "scripts" / "data" / "build_question.py"
MODELS = {"rl": paths.HF_CHECKPOINTS["rl"], "sft": paths.HF_CHECKPOINTS["sft"]}
MAX_LENGTH_PROTEIN = 2000
GOGPT_BATCH = 16
#: Absent from all 1.8M partner names in the corpus; ";" and "|" both occur inside names.
PARTNER_SEPARATOR = "~"
#: ESM3's residue alphabet. X (unknown) and U (selenocysteine) occur in the training data and are
#: kept; dropping them would shift every position after them.
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYXBUZO")
TSV_COLUMNS = ["protein_id", "organism", "sequence_length", "features", "gogpt", "question",
               "generated_response"]


def clean_sequence(seq: str) -> str:
    """Upper-case the sequence and drop whitespace and non-amino-acid characters.

    Args:
        seq: raw sequence.

    Returns:
        The cleaned sequence.
    """
    seq = re.sub(r"\s+", "", seq).upper()
    cleaned = "".join(c for c in seq if c in VALID_AA)
    if len(cleaned) != len(seq):
        print(f"  Warning: removed {len(seq) - len(cleaned)} non-amino-acid characters")
    return cleaned


def read_input_tsv(path: str) -> List[Dict[str, str]]:
    """Read and validate the input proteins.

    Args:
        path: input TSV.

    Returns:
        One dict per protein, sequence cleaned.

    Raises:
        ValueError: on an empty file, a missing column, a repeated protein_id, or an empty sequence.
    """
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError("Input TSV is empty")
    for col in ("protein_id", "organism", "sequence"):
        if col not in rows[0]:
            raise ValueError(f"Input TSV missing required column: '{col}'")
    seen = set()
    for row in rows:
        # Every stage's results are keyed by protein_id, so a repeat would take another row's.
        if row["protein_id"] in seen:
            raise ValueError(f"Protein '{row['protein_id']}' appears more than once; give each row its own id")
        seen.add(row["protein_id"])
        row["sequence"] = clean_sequence(row["sequence"])
        row["organism"] = row["organism"].strip()
        if not row["sequence"]:
            raise ValueError(f"Protein '{row['protein_id']}' has an empty sequence after cleaning")
    return rows


def completed_ids(output_path: str) -> set:
    """Protein ids that already have a generation in the output.

    Args:
        output_path: output TSV.

    Returns:
        The ids.
    """
    if not os.path.exists(output_path):
        return set()
    with open(output_path) as f:
        return {row["protein_id"] for row in csv.DictReader(f, delimiter="\t")
                if row.get("generated_response")}


def append_results(output_path: str, rows: List[Dict[str, str]]) -> None:
    """Append rows to the output TSV, writing the header first if the file is new.

    Args:
        output_path: output TSV.
        rows: result rows.
    """
    new = not os.path.exists(output_path)
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in TSV_COLUMNS})


def load_cache(path: str, resume: bool) -> Dict:
    """A stage's cached results.

    Args:
        path: the stage's JSON cache.
        resume: reuse it; otherwise start empty.

    Returns:
        protein_id -> result.
    """
    if resume and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def run_interpro(proteins: List[Dict[str, str]], cache_path: str, args) -> Dict[str, List[Dict]]:
    """InterPro feature records per protein.

    Args:
        proteins: input proteins.
        cache_path: JSON cache of finished proteins.
        args: predict.py's arguments.

    Returns:
        protein_id -> feature records. A protein InterPro could not be run for is absent and
        gets the no-feature prompt.
    """
    results = load_cache(cache_path, args.resume)
    todo = {p["protein_id"]: p["sequence"] for p in proteins if p["protein_id"] not in results}
    source = f"local InterProScan at {args.interproscan_dir}" if args.interproscan_dir else "online"
    print(f"\n[1/3] InterPro ({source}): {len(todo)} of {len(proteins)} proteins to scan")
    if todo:
        results.update(scan_many(todo, email=args.email, interproscan_dir=args.interproscan_dir,
                                 cpu=args.cpu))
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=1)
    missing = sum(p["protein_id"] not in results for p in proteins)
    if missing:
        print(f"  {missing} proteins without InterPro results take the no-feature prompt")
    return results


def run_gogpt(proteins: List[Dict[str, str]], cache_path: str, resume: bool,
              revision: str = paths.HF_GOGPT_REVISION) -> Dict[str, str]:
    """GO-GPT predictions per protein, rendered as the prompt's GO-term block.

    Args:
        proteins: input proteins.
        cache_path: JSON cache of finished predictions.
        resume: reuse the cache.
        revision: GO-GPT's Hub revision.

    Returns:
        protein_id -> rendered GO-GPT block.
    """
    results = load_cache(cache_path, resume)
    todo = [p for p in proteins if p["protein_id"] not in results]
    print(f"\n[2/3] GO-GPT ({revision}): {len(todo)} of {len(proteins)} proteins to predict")
    if todo:
        predictor = load_predictor(revision=revision)
        # A taxonomy id identifies the organism unambiguously; the name is the fallback.
        organisms = [(p.get("taxid") or "").strip() or p["organism"] for p in todo]
        predictions = predictor.predict_batch([p["sequence"] for p in todo], organisms=organisms,
                                              batch_size=GOGPT_BATCH)
        for p, pred in zip(todo, predictions):
            results[p["protein_id"]] = format_go_output(pred)
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=1)
        # The model takes the GPU next.
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
    return results


def load_question_builder():
    """scripts/data/build_question.py, the one writer of the training prompts.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location("brp_build_question", QUESTION_BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prompt_row(protein: Dict[str, str], go_pred: str, features: Optional[List[Dict]]) -> Dict:
    """The corpus-shaped row the question builder reads.

    Args:
        protein: one input protein.
        go_pred: its rendered GO-GPT block.
        features: its InterPro feature records, or None for the no-feature prompt.

    Returns:
        The row.
    """
    partners = [n.strip() for n in (protein.get("partners") or "").split(PARTNER_SEPARATOR) if n.strip()]
    return {"organism": protein["organism"], "go_pred": go_pred,
            "ppi_formatted": "\n".join(f"- {name}" for name in partners),
            "features": features or []}


def resolve_checkpoint(checkpoint: str) -> str:
    """A local checkpoint directory, downloading a Hub id first.

    Args:
        checkpoint: local directory or Hub repository id.

    Returns:
        The local directory.
    """
    if os.path.isdir(checkpoint):
        return checkpoint
    print(f"  Downloading {checkpoint} ...")
    return snapshot_download(repo_id=checkpoint)


def eval_arguments(args, ckpt_dir: str, work_dir: str) -> argparse.Namespace:
    """eval.py's arguments for this checkpoint, at eval.py's defaults.

    Args:
        args: predict.py's arguments.
        ckpt_dir: local checkpoint directory.
        work_dir: scratch directory eval.py requires.

    Returns:
        The parsed namespace.
    """
    return brp_eval.setup_argument_parser().parse_args([
        "--ckpt_dir", ckpt_dir, "--evals_path", work_dir, "--cafa5_dataset", "unused-by-predict",
        "--go_obo_path", paths.GO_OBO, "--precomputed_embeddings_path", args.go_embeddings_path,
        "--protein_projection_norm", "blocknorm",
        "--protein_projection_norm_target", str(paths.PROTEIN_PROJECTION_NORM_TARGET),
        "--gpu_memory_utilization", str(args.gpu_memory_utilization),
        "--batch_size", str(args.batch_size), "--max_length_protein", str(MAX_LENGTH_PROTEIN),
    ])


def run_bioreason(proteins: List[Dict[str, str]], go_preds: Dict[str, str],
                  features: Optional[Dict[str, List[Dict]]], args) -> None:
    """Generate with the checkpoint and append each batch to the output TSV.

    Args:
        proteins: input proteins.
        go_preds: protein_id -> rendered GO-GPT block.
        features: protein_id -> InterPro feature records, or None without InterPro.
        args: predict.py's arguments.

    Raises:
        SystemExit: when the GO embeddings are missing.
    """
    done = completed_ids(args.output) if args.resume else set()
    todo = [p for p in proteins if p["protein_id"] not in done]
    print(f"\n[3/3] BioReason-Pro 2: {len(todo)} of {len(proteins)} proteins to generate")
    if not todo:
        return
    if not os.path.isdir(args.go_embeddings_path):
        raise SystemExit(f"no GO embeddings at {args.go_embeddings_path} "
                         f"(python scripts/data/download_assets.py, README Setup)")

    builder = load_question_builder()
    work_dir = str(Path(args.output).with_suffix("")) + "_work"
    checkpoint = args.checkpoint or MODELS[args.model]
    eval_args = eval_arguments(args, resolve_checkpoint(checkpoint), work_dir)
    model = brp_eval.initialize_model(eval_args)

    for start in tqdm(range(0, len(todo), args.batch_size), desc="  BioReason-Pro 2", unit="batch"):
        batch = todo[start:start + args.batch_size]
        samples, rendered = [], {}
        for p in batch:
            pid = p["protein_id"]
            row = prompt_row(p, go_preds.get(pid, ""), (features or {}).get(pid))
            question = builder.build_question(row, features=True)
            rendered[pid] = (builder.context_fields(row, True)["features"], question)
            sample = {"protein_id": pid, "sequence": p["sequence"][:MAX_LENGTH_PROTEIN],
                      "structure_path": (p.get("structure_path") or "").strip() or None,
                      "question": question}
            samples.append({**sample, **format_cafa5_for_protein_llm(_format_reasoning_prompt(sample))})
        records = brp_eval.process_batch(model, samples, eval_args)
        by_id = {p["protein_id"]: p for p in batch}
        append_results(args.output, [{
            "protein_id": r["protein_id"], "organism": by_id[r["protein_id"]]["organism"],
            "sequence_length": len(by_id[r["protein_id"]]["sequence"]),
            "features": rendered[r["protein_id"]][0],
            "gogpt": go_preds.get(r["protein_id"], ""),
            "question": rendered[r["protein_id"]][1],
            "generated_response": r["generated_response"],
        } for r in records])
    print(f"  {len(completed_ids(args.output))}/{len(proteins)} proteins have a generation")


def main() -> int:
    """CLI.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="input TSV (protein_id, organism, sequence)")
    parser.add_argument("--output", required=True, help="output TSV")
    parser.add_argument("--model", choices=sorted(MODELS), default="rl",
                        help="rl (default, the best model) or sft (the model RL started from)")
    parser.add_argument("--checkpoint", default=None,
                        help="a local directory or Hub id, instead of --model")
    parser.add_argument("--interpro", action="store_true",
                        help="add InterProScan features to the prompt, online")
    parser.add_argument("--interproscan_dir", default=None,
                        help="add InterProScan features from this local InterProScan 5 installation")
    parser.add_argument("--email", default=None,
                        help="your email, for EBI InterProScan jobs (sequences InterPro has not scanned)")
    parser.add_argument("--cpu", type=int, default=8, help="InterProScan threads (local)")
    parser.add_argument("--gogpt_revision", default=paths.HF_GOGPT_REVISION,
                        help="GO-GPT's Hub revision: ep75 (default, the predictions the models were trained "
                             "with, also main) or ep89 (a later checkpoint, slightly better at GO on its own)")
    parser.add_argument("--go_embeddings_path", default=paths.GO_EMBEDDINGS)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.45,
                        help="vLLM's share only; ESM3 and the GO encoder allocate outside it")
    parser.add_argument("--resume", action="store_true",
                        help="reuse cached InterPro and GO-GPT results and skip finished proteins")
    args = parser.parse_args()

    proteins = read_input_tsv(args.input)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    stem = str(Path(args.output).with_suffix(""))
    print(f"BioReason-Pro 2: {len(proteins)} proteins, model {args.checkpoint or MODELS[args.model]}")
    features = None
    if args.interpro or args.interproscan_dir:
        features = run_interpro(proteins, stem + "_interpro.json", args)
    else:
        print("\n[1/3] InterPro: off (--interpro to add it)")
    go_preds = run_gogpt(proteins, stem + "_gogpt.json", args.resume, args.gogpt_revision)
    run_bioreason(proteins, go_preds, features, args)
    print(f"\nResults: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
