# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""High-level inference API.

    from gogpt import GOGPTPredictor

    predictor = GOGPTPredictor.from_pretrained()          # wanglab/gogpt-v2 from the Hub
    predictor.predict("MKTAYIAKQR...", organism=9606)
    # {"MF": ["GO:0003674", ...], "BP": [...], "CC": [...]}
"""

import contextlib
import json
import warnings
from pathlib import Path

import torch

from gogpt.config.model_config import GOGPTConfig
from gogpt.data.tokenizer import ASPECTS, GOTermTokenizer
from gogpt.models.gogpt import GOGPT
from gogpt.models.protein_encoder import clean_sequence
from gogpt.utils.organism_mapper import OrganismMapper

DEFAULT_MODEL_ID = "wanglab/gogpt-v2"
MODEL_FILES = ("config.json", "model.safetensors", "go_tokenizer.json", "organism_mapper.json")


class GOGPTPredictor:
    """Loads GO-GPT (decoder + frozen ESM2 encoder) and predicts GO terms for sequences.

    The default decoding reproduces the released predictions: greedy, up to 200 GO
    tokens per aspect, bf16 autocast on GPU (fp32 on CPU).
    """

    def __init__(self, model_dir, device=None, precision="auto", embedding_cache_dir=None):
        """
        Args:
            model_dir: directory holding config.json, model.safetensors, go_tokenizer.json
                and organism_mapper.json (see from_pretrained for the Hub).
            device: "cuda", "cpu", ... (default: cuda if available).
            precision: "auto" (bf16 autocast on CUDA, fp32 elsewhere), "bf16" or "fp32".
            embedding_cache_dir: optional directory of precomputed ESM2 embeddings
                (scripts/precompute_embeddings.py). Proteins found there by id are not
                re-encoded; all others are encoded on the fly.
        """
        from safetensors.torch import load_model

        model_dir = Path(model_dir)
        with open(model_dir / "config.json") as f:
            cfg = json.load(f)
        self.config = GOGPTConfig.from_dict(cfg["model"])
        self.generation_defaults = dict(cfg.get("generation", {}))
        self.tokenizer = GOTermTokenizer.from_json(model_dir / "go_tokenizer.json")
        self.organism_mapper = OrganismMapper.from_json(model_dir / "organism_mapper.json")
        if len(self.tokenizer) != self.config.vocab_size:
            raise ValueError(f"tokenizer has {len(self.tokenizer)} tokens, config says {self.config.vocab_size}")
        if self.organism_mapper.vocab_size != self.config.organism_vocab_size:
            raise ValueError("organism mapper size disagrees with the model config")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if precision == "auto":
            precision = "bf16" if self.device.type == "cuda" else "fp32"
        if precision not in ("bf16", "fp32"):
            raise ValueError(f"precision must be 'auto', 'bf16' or 'fp32', got {precision!r}")
        self.precision = precision

        self.model = GOGPT(self.config)
        load_model(self.model, str(model_dir / "model.safetensors"), strict=True)
        self.model.to(self.device).eval()

        self._encoder = None  # ESM2, loaded on first use
        self.embedding_cache = None
        if embedding_cache_dir is not None:
            from gogpt.data.precomputed_embeddings import EmbeddingCache

            cache = EmbeddingCache(embedding_cache_dir)
            m = cache.manifest
            expected = {"model": self.config.embed_model_path, "layer": self.config.protein_layer_index,
                        "max_len": self.config.max_protein_length, "dim": self.config.protein_embedding_dim}
            got = {"model": m.get("model"), "layer": m.get("layer"), "max_len": cache.max_len, "dim": cache.dim}
            if got != expected:
                raise ValueError(f"{embedding_cache_dir} was built with {got}; this model needs {expected}")
            self.embedding_cache = cache
        self._warned = set()

    @classmethod
    def from_pretrained(cls, model_id=DEFAULT_MODEL_ID, revision=None, cache_dir=None, token=None, **kwargs):
        """Download (or reuse the cached) model files from the Hugging Face Hub."""
        from huggingface_hub import hf_hub_download

        paths = [hf_hub_download(model_id, f, revision=revision, cache_dir=cache_dir, token=token)
                 for f in MODEL_FILES]
        return cls(Path(paths[0]).parent, **kwargs)

    # ------------------------------------------------------------------------ helpers

    @property
    def encoder(self):
        if self._encoder is None:
            from gogpt.models.protein_encoder import ESM2Encoder

            self._encoder = ESM2Encoder(self.config.embed_model_path, self.config.protein_layer_index,
                                        self.config.max_protein_length).to(self.device)
        return self._encoder

    def _autocast(self):
        if self.precision == "bf16":
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _warn_once(self, key, message):
        if key not in self._warned:
            self._warned.add(key)
            warnings.warn(message, stacklevel=3)

    def organism_index(self, organism):
        """Embedding index for an NCBI taxonomy id (int or str) or organism name.

        Organisms outside the model's organism vocabulary, and None, use the shared
        <UNKNOWN> embedding (index 0).
        """
        key = self.organism_mapper.resolve(organism)
        if key is None:
            if organism is None:
                self._warn_once(None, "no organism given; using the <UNKNOWN> organism embedding")
            else:
                self._warn_once(organism, f"organism {organism!r} is not one of the model's "
                                f"{self.organism_mapper.vocab_size - 1} organisms; using <UNKNOWN>")
            return 0
        return self.organism_mapper.map_organism(key)

    def _embed(self, sequences, protein_ids):
        """Right-padded [B, T, D] embeddings and a [B, T] bool mask."""
        L = self.config.max_protein_length
        rows = [None] * len(sequences)
        if self.embedding_cache is not None and protein_ids is not None:
            for i, pid in enumerate(protein_ids):
                if pid in self.embedding_cache:
                    emb = self.embedding_cache.get(pid)
                    if emb.shape[0] != min(len(sequences[i]), L):
                        raise ValueError(f"{pid}: cached embedding has {emb.shape[0]} rows but the "
                                         f"sequence has {len(sequences[i])} residues (max {L})")
                    rows[i] = emb.to(self.device)
        todo = [i for i, r in enumerate(rows) if r is None]
        if todo:
            ids, mask = self.encoder.batch_tokens([sequences[i] for i in todo], device=self.device)
            with self._autocast():
                hidden = self.encoder(ids, mask)
            # Rounded to fp16, the precision the training embeddings were stored in, so
            # on-the-fly and precomputed embeddings give the same predictions.
            for j, i in enumerate(todo):
                rows[i] = hidden[j, : int(mask[j].sum())].to(torch.float16)
        T = max(r.shape[0] for r in rows)
        emb = torch.zeros(len(rows), T, self.config.protein_embedding_dim, device=self.device)
        mask = torch.zeros(len(rows), T, dtype=torch.bool, device=self.device)
        for i, r in enumerate(rows):
            emb[i, : r.shape[0]] = r.float()
            mask[i, : r.shape[0]] = True
        return emb, mask

    # ------------------------------------------------------------------------ predict

    @torch.no_grad()
    def predict(self, sequence, organism=None, **kwargs):
        """GO terms for one protein: {"MF": [...], "BP": [...], "CC": [...]}."""
        return self.predict_batch([sequence], [organism], **kwargs)[0]

    @torch.no_grad()
    def predict_batch(self, sequences, organisms=None, protein_ids=None, aspects=ASPECTS,
                      batch_size=16, max_new_tokens=None, decoding="greedy", beam_size=5,
                      length_penalty=0.6, temperature=1.0, top_k=None):
        """GO terms for many proteins, one dict per input in input order.

        Args:
            sequences: amino-acid strings (whitespace and case are normalized).
            organisms: per-protein NCBI taxonomy id or organism name, or None.
            protein_ids: ids used to look proteins up in `embedding_cache_dir`.
            aspects: any of "MF", "BP", "CC".
            batch_size: proteins per forward pass.
            max_new_tokens: GO tokens generated per aspect (default from config: 200).
            decoding: "greedy" (default, deterministic), "beam" or "sample".
        Terms are listed in generation order (general to specific), without repeats.
        """
        n = len(sequences)
        organisms = list(organisms) if organisms is not None else [None] * n
        if len(organisms) != n or (protein_ids is not None and len(protein_ids) != n):
            raise ValueError("sequences, organisms and protein_ids must have the same length")
        if decoding not in ("greedy", "beam", "sample"):
            raise ValueError(f"decoding must be 'greedy', 'beam' or 'sample', got {decoding!r}")
        bad_aspects = [a for a in aspects if a not in ASPECTS]
        if bad_aspects:
            raise ValueError(f"unknown aspects {bad_aspects}; choose from {ASPECTS}")
        if max_new_tokens is None:
            max_new_tokens = self.generation_defaults.get("max_new_tokens", 200)

        sequences = [clean_sequence(s) for s in sequences]
        for s in sequences:
            if len(s) > self.config.max_protein_length:
                self._warn_once("truncation", f"sequences longer than {self.config.max_protein_length} "
                                "residues are truncated, as in training")
        org_idx = [self.organism_index(o) for o in organisms]

        results = [None] * n
        order = sorted(range(n), key=lambda i: -len(sequences[i]))  # less padding per batch
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            emb, mask = self._embed([sequences[i] for i in idx],
                                    [protein_ids[i] for i in idx] if protein_ids is not None else None)
            org = torch.tensor([org_idx[i] for i in idx], dtype=torch.long, device=self.device)
            out = [{} for _ in idx]
            for aspect in aspects:
                start_id = self.tokenizer.aspect_to_tokens[aspect][0]
                go = torch.full((len(idx), 1), start_id, dtype=torch.long, device=self.device)
                with self._autocast():
                    if decoding == "beam":
                        tokens = self.model.generate_beam_search(
                            emb, mask, go, max_new_tokens, beam_size=beam_size,
                            length_penalty=length_penalty, organism_id=org)
                    else:
                        tokens = self.model.generate(
                            emb, mask, go, max_new_tokens,
                            temperature=0.0 if decoding == "greedy" else temperature,
                            top_k=top_k, organism_id=org)
                for j, row in enumerate(tokens.tolist()):
                    out[j][aspect] = self.tokenizer.decode(row, aspect)
            for j, i in enumerate(idx):
                results[i] = out[j]
        return results
