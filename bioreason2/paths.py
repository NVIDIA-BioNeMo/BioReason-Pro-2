#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Every Hub id and default location, in one place.

    python -m bioreason2.paths          # print what resolves, and what is missing

It lives in the package so that every module and script reaches it the same way,
`from bioreason2 import paths`. Local locations are relative to the clone, which the README installs
editable (`pip install -e .`), and each moves with an environment variable:

    BRP_DATA_DIR     the corpus (fetch_corpus.py)   default <repo>/data/v2
    BRP_ASSETS_DIR   GO embeddings and structures   default <repo>/assets
    BRP_IA_FILE      CAFA 6 IA weights              default <repo>/data/IA_CAFA6.tsv
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- published artifacts ----------------------------------------------------

#: The proteins and labels of the benchmark and of training (scripts/data/fetch_corpus.py), which the
#: traces were written from and GO-GPT was trained on (synth/, gogpt/).
HF_RAW_DATASET = "wanglab/bioreason-pro-v2"

#: The released models. Both take the `blocknorm` projection at PROTEIN_PROJECTION_NORM_TARGET.
HF_CHECKPOINTS = {
    "sft": "nvidia/BioReason-Pro-2-9B-SFT",
    "rl": "nvidia/BioReason-Pro-2-9B-RL",
}
PROTEIN_PROJECTION_NORM_TARGET = 0.0133

#: Required: the GO graph encoder will not build without these.
HF_GO_EMBEDDINGS = "wanglab/bioreason-pro-go-embeddings"

#: Optional: without structures ESM3 reads the sequence alone.
HF_STRUCTURES = "wanglab/bioreason-pro-structures"

#: The protein encoder. The `esm` package downloads it itself.
HF_PROTEIN_ENCODER = "biohub/esm3-sm-open-v1"

#: GO-GPT, the GO term predictor (gogpt/). predict.py runs it on new proteins.
HF_GOGPT = "wanglab/gogpt-v2"
#: The GO-GPT checkpoint whose predictions the models were trained and scored with (epoch 75), also the
#: repository's `main`. Its `ep89` branch holds a later checkpoint, slightly better at GO on its own.
HF_GOGPT_REVISION = "ep75"

# --- local locations --------------------------------------------------------

DATA_DIR = os.environ.get("BRP_DATA_DIR", os.path.join(REPO_ROOT, "data", "v2"))
ASSETS_DIR = os.environ.get("BRP_ASSETS_DIR", os.path.join(REPO_ROOT, "assets"))

#: Parquet shards of the corpus, as scripts/data/fetch_corpus.py writes them.
CORPUS = os.path.join(DATA_DIR, "proteins")
#: `scripts/data/prepare_corpus.py`'s output and `build_question.py`'s input: the corpus with
#: GO-GPT's epoch-75 predictions rendered as `go_pred` and a `split` column.
PROMPT_CORPUS = DATA_DIR + "-ep75"

GO_EMBEDDINGS = os.path.join(ASSETS_DIR, "go_embeddings")
STRUCTURES = os.path.join(ASSETS_DIR, "structures")

#: CAFA 6's information accretion, which every weighted score uses. CAFA publishes it without a
#: license, so it does not ship: `scripts/data/download_assets.py` fetches it.
IA_CAFA6 = os.environ.get("BRP_IA_FILE", os.path.join(REPO_ROOT, "data", "IA_CAFA6.tsv"))

# --- files that ship with the repository ------------------------------------

#: Pinned at releases/2023-01-01. The GO encoder vocabulary and every file in GO_EMBEDDINGS are
#: keyed to it.
GO_OBO = os.path.join(REPO_ROOT, "bioreason2", "dataset", "go-basic.obo")

#: The pinned 1,000-protein judge sample. Two models are only comparable on the same list.
JUDGE_SUBSET = os.path.join(REPO_ROOT, "data", "eval", "judge_subset.txt")


def describe() -> int:
    """Print every location and whether it is present.

    Returns:
        0 when every required local path exists, 1 otherwise.
    """
    print("repository  %s\n" % REPO_ROOT)
    print("Hugging Face")
    print("  %-22s %s  (benchmark and training proteins)" % ("corpus", HF_RAW_DATASET))
    for key, repo in HF_CHECKPOINTS.items():
        print("  %-22s %s" % ("model " + key, repo))
    print("  %-22s %s" % ("GO embeddings", HF_GO_EMBEDDINGS))
    print("  %-22s %s  (optional)" % ("structures", HF_STRUCTURES))
    print("  %-22s %s" % ("protein encoder", HF_PROTEIN_ENCODER))
    print("  %-22s %s  (revision %s)" % ("GO-GPT", HF_GOGPT, HF_GOGPT_REVISION))

    required = [("GO embeddings", GO_EMBEDDINGS), ("GO ontology", GO_OBO)]
    # predict.py needs neither: they serve the scorer and the judge.
    for_scoring = [("IA weights", IA_CAFA6), ("judge sample", JUDGE_SUBSET)]
    # The corpus is needed for the benchmark and training, not for predict.py.
    optional = [("corpus", CORPUS), ("structures", STRUCTURES)]

    print("\nLocal")
    missing = 0
    for label, path in required:
        ok = os.path.exists(path)
        missing += not ok
        print("  %-22s %-58s %s" % (label, path, "ok" if ok else "MISSING"))
    for label, path in for_scoring:
        print("  %-22s %-58s %s"
              % (label, path, "ok" if os.path.exists(path) else "absent (needed for scoring)"))
    for label, path in optional:
        print("  %-22s %-58s %s"
              % (label, path, "ok" if os.path.exists(path) else "absent (optional)"))

    if missing:
        print("\n%d required path(s) missing. See the README, section 'Setup'." % missing)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(describe())
