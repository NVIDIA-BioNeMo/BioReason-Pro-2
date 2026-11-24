# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from gogpt.data.tokenizer import GOTermTokenizer
from gogpt.utils.organism_mapper import OrganismMapper

ESM_PAD_TOKEN_ID = 1


class PreprocessedProteinGODataset(Dataset):
    """Examples written by scripts/prepare_data.py: one (protein, aspect) per row.

    With `embedding_cache`, each item also carries the protein's precomputed embeddings
    and the model never runs the protein encoder.
    """

    def __init__(self, hf_dataset, embedding_cache=None):
        self.dataset = hf_dataset
        self.embedding_cache = embedding_cache

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        n_go = len(ex["go_input_tokens"])
        item = {
            "protein_tokens": torch.tensor(ex["protein_tokens"], dtype=torch.long),
            "protein_mask": torch.ones(len(ex["protein_tokens"]), dtype=torch.bool),
            "go_tokens": torch.tensor(ex["go_input_tokens"], dtype=torch.long),
            "targets": torch.tensor(ex["go_targets"], dtype=torch.long),
            "go_mask": torch.ones(n_go, dtype=torch.bool),
            "organism_id": torch.tensor(ex["organism_id"], dtype=torch.long),
            "protein_id": ex["protein_id"],
        }
        if "go_terms_list" in ex:  # validation ground truth (unfiltered by the vocabulary)
            item["go_terms_list"] = ex["go_terms_list"]
        if self.embedding_cache is not None:
            emb = self.embedding_cache.get(ex["protein_id"])
            if emb.shape[0] != len(ex["protein_tokens"]):
                raise ValueError(
                    f"{ex['protein_id']}: {emb.shape[0]} embedding rows but "
                    f"{len(ex['protein_tokens'])} protein tokens; the embeddings and the dataset "
                    "were built with different truncation lengths")
            item["protein_embedding"] = emb
        return item


def collate_batch(batch):
    out = {
        "protein_tokens": pad_sequence([b["protein_tokens"] for b in batch], batch_first=True,
                                       padding_value=ESM_PAD_TOKEN_ID),
        "protein_mask": pad_sequence([b["protein_mask"] for b in batch], batch_first=True,
                                     padding_value=False),
        "go_tokens": pad_sequence([b["go_tokens"] for b in batch], batch_first=True, padding_value=0),
        "go_mask": pad_sequence([b["go_mask"] for b in batch], batch_first=True, padding_value=False),
        "targets": pad_sequence([b["targets"] for b in batch], batch_first=True, padding_value=0),
        "organism_id": torch.stack([b["organism_id"] for b in batch]),
        "protein_ids": [b["protein_id"] for b in batch],
    }
    if "go_terms_list" in batch[0]:
        out["go_terms_list"] = [b["go_terms_list"] for b in batch]
    if "protein_embedding" in batch[0]:
        out["protein_embeddings"] = pad_sequence([b["protein_embedding"] for b in batch],
                                                 batch_first=True, padding_value=0.0)
    return out


def load_artifacts(artifacts_dir):
    """(GO tokenizer, organism mapper, dataset info) written by scripts/prepare_data.py."""
    artifacts_dir = Path(artifacts_dir)
    tokenizer = GOTermTokenizer.from_json(artifacts_dir / "go_tokenizer.json")
    mapper = OrganismMapper.from_json(artifacts_dir / "organism_mapper.json")
    with open(artifacts_dir / "dataset_info.json") as f:
        info = json.load(f)
    return tokenizer, mapper, info


def load_preprocessed_data(preprocessed_path, batch_size, num_workers=4, pin_memory=False,
                           persistent_workers=True, prefetch_factor=4, embedding_cache_dir=None,
                           expected_embedding_dim=None, expected_embedding_model=None,
                           expected_embedding_layer=None):
    """Train/validation loaders plus the GO tokenizer and dataset info.

    With embedding_cache_dir the precomputed embeddings are checked against both splits
    (coverage, row counts, model, layer, dim) before any loader is built.
    """
    from datasets import load_from_disk

    preprocessed_path = Path(preprocessed_path)
    dataset = load_from_disk(str(preprocessed_path / "dataset"))
    train_data, val_data = dataset["train"], dataset["validation"]
    tokenizer, _, info = load_artifacts(preprocessed_path / "artifacts")
    print(f"Loaded {len(train_data):,} train / {len(val_data):,} validation examples, "
          f"GO vocabulary {len(tokenizer):,}")

    cache = None
    if embedding_cache_dir is not None:
        from gogpt.data.precomputed_embeddings import verify_cache

        lengths = {}
        for split in (train_data, val_data):
            for pid, n in zip(split["protein_id"], split["protein_length"]):
                lengths[pid] = n
        cache = verify_cache(embedding_cache_dir, lengths.keys(),
                             expect_dim=expected_embedding_dim, expect_model=expected_embedding_model,
                             expect_layer=expected_embedding_layer, expected_lengths=lengths)

    loader_kwargs = dict(batch_size=batch_size, collate_fn=collate_batch, num_workers=num_workers,
                         pin_memory=pin_memory)
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
    train_loader = DataLoader(PreprocessedProteinGODataset(train_data, cache), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(PreprocessedProteinGODataset(val_data, cache), shuffle=False, **loader_kwargs)
    return tokenizer, train_loader, val_loader, info
