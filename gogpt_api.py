#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
GO-GPT API: predict Gene Ontology (GO) terms from a protein sequence with GO-GPT v2.

Usage (CLI):
    python gogpt_api.py --sequence <protein_sequence> --organism 9606
    python gogpt_api.py --sequence <protein_sequence> --organism "Homo sapiens"

Usage (import):
    from gogpt_api import load_predictor, predict_and_format

    predictor = load_predictor()
    text = predict_and_format(predictor, sequence, organism=9606)

The organism is an NCBI taxonomy id or an organism name. GO-GPT knows the 200 organisms
most frequent in its training data; others, or none, use a shared "unknown organism"
embedding (with a warning). See gogpt/README.md.
"""
import argparse
import functools
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "gogpt" / "src"))
sys.path.insert(0, str(REPO_ROOT))

from bioreason2.paths import HF_GOGPT, HF_GOGPT_REVISION  # noqa: E402
from gogpt import GOGPTPredictor  # noqa: E402
from gogpt.utils.go_ontology import format_predictions, load_go_names  # noqa: E402

# The ontology BioReason-Pro 2 names GO terms with (GO release 2023-01-01).
GO_OBO_PATH = REPO_ROOT / "bioreason2" / "dataset" / "go-basic.obo"

Organism = Optional[Union[int, str]]


@functools.lru_cache(maxsize=1)
def _go_names() -> Dict[str, str]:
    return load_go_names(GO_OBO_PATH)


def load_predictor(
    model_name: str = HF_GOGPT,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = HF_GOGPT_REVISION,
    device: Optional[str] = None,
) -> GOGPTPredictor:
    """Load GO-GPT from the Hugging Face Hub (or the local cache)."""
    return GOGPTPredictor.from_pretrained(model_name, revision=revision, cache_dir=cache_dir, device=device)


def predict_go_terms(predictor: GOGPTPredictor, sequence: str, organism: Organism = None) -> Dict[str, List[str]]:
    """GO terms per aspect: {"MF": [...], "BP": [...], "CC": [...]}, in generation order."""
    return predictor.predict(sequence, organism=organism)


def format_go_output(predictions: Dict[str, List[str]]) -> str:
    """Render predictions as the GO-term block of the BioReason-Pro 2 prompt.

    One line per aspect (MF, BP, CC), terms as "GO:xxxxxxx (name)" joined by ", ";
    an aspect without predictions keeps its label with nothing after ": ".
    """
    return format_predictions(predictions, _go_names())


def predict_and_format(predictor: GOGPTPredictor, sequence: str, organism: Organism = None) -> str:
    return format_go_output(predict_go_terms(predictor, sequence, organism))


def main():
    parser = argparse.ArgumentParser(description="Predict GO terms from a protein sequence using GO-GPT v2.")
    parser.add_argument("--sequence", required=True, help="Protein sequence (amino acids)")
    parser.add_argument("--organism", default=None,
                        help="NCBI taxonomy id (e.g. 9606) or organism name (e.g. 'Homo sapiens'); optional")
    parser.add_argument("--model", default=HF_GOGPT, help=f"Hugging Face model id (default: {HF_GOGPT})")
    parser.add_argument("--revision", default=HF_GOGPT_REVISION,
                        help=f"Hub revision (default: {HF_GOGPT_REVISION}, the one the models were trained with; "
                             "ep89 for the later checkpoint)")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory")
    args = parser.parse_args()

    predictor = load_predictor(args.model, args.cache_dir, args.revision)
    print(predict_and_format(predictor, args.sequence, args.organism))


if __name__ == "__main__":
    main()
