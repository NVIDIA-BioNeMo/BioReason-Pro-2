# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Build the GO-GPT training dataset from wanglab/bioreason-pro-v2.

Training proteins are the `train` splits of the `proteins` and `proteins_extra` configs
(proteins in `proteins/test` are always excluded). A validation set is held out from
both sources. Then, per (protein, aspect):

* labels: tier-1 ∪ tier-2 GO terms (configurable), already propagated to ancestors,
  restricted to the vocabulary and sorted by GO depth (root first), so the decoder
  generates general terms before specific ones;
* validation additionally keeps the raw tier-1 terms (`go_terms_list`) as the
  ground truth for the CAFA metric, unfiltered by the vocabulary;
* proteins are tokenized for ESM2 without special tokens and truncated to
  `max_protein_length` residues.

The vocabulary keeps, per aspect, the terms annotated on at least `min_freq` training
proteins. The organism vocabulary keeps the `top_n_organisms` most frequent NCBI
taxonomy ids; the rest share the <UNKNOWN> embedding.

    python scripts/prepare_data.py                       # downloads the dataset
    python scripts/prepare_data.py data_dir=/path/to/bioreason-pro-v2

Output: <output_dir>/{dataset/, artifacts/{go_tokenizer.json, organism_mapper.json,
dataset_info.json}}.
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import hydra
import numpy as np
import pyarrow.parquet as pq
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gogpt.data.tokenizer import ASPECTS, GOTermTokenizer  # noqa: E402
from gogpt.utils.organism_mapper import OrganismMapper  # noqa: E402


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def dataset_root(cfg):
    """Local directory holding proteins/ and proteins_extra/; downloads it if not given."""
    if cfg.data_dir:
        return resolve(cfg.data_dir)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(cfg.hf_dataset, repo_type="dataset", revision=cfg.hf_revision,
                                  allow_patterns=["proteins/*.parquet", "proteins_extra/*.parquet"]))


def aslist(v):
    return [] if v is None else list(v)


def taxid_key(taxid):
    if taxid is None or (isinstance(taxid, float) and np.isnan(taxid)):
        return None
    return str(int(taxid))


def load_source(directory, pattern, source, include_v1_vintage):
    cols = (["protein_id", "taxid", "organism", "sequence", "uniprot_status"]
            + [f"go_{a.lower()}" for a in ASPECTS] + [f"go_{a.lower()}_tier2" for a in ASPECTS])
    if source == "extra":
        cols.append("nn_identity_v2")
    files = sorted(directory.glob(pattern))
    if not files:
        raise SystemExit(f"no {pattern} under {directory}")
    rows, n_total = {}, 0
    for f in files:
        data = pq.read_table(f, columns=cols).to_pydict()
        for i, pid in enumerate(data["protein_id"]):
            n_total += 1
            if not include_v1_vintage and data["uniprot_status"][i] != "live":
                continue
            if pid in rows:
                raise SystemExit(f"{pid} appears twice in {source}")
            rows[pid] = {c: data[c][i] for c in cols}
            rows[pid]["source"] = source
    print(f"{source}: {len(rows):,} of {n_total:,} rows")
    return rows


def go_depth_function(obo_file):
    """Longest is_a + part_of path from a term to its namespace root."""
    from goatools.obo_parser import GODag

    dag = GODag(str(obo_file), optional_attrs=["relationship"], prt=None)
    cache = {}

    def parents(term):
        ps = set(term.parents)
        if "part_of" in getattr(term, "relationship", {}):
            ps.update(term.relationship["part_of"])
        return ps

    def depth(go_id):
        if go_id in cache:
            return cache[go_id]
        if go_id not in dag:
            raise KeyError(f"{go_id} is not in {obo_file}")
        best, seen, stack = 0, {}, [(dag[go_id], 0)]
        while stack:
            term, d = stack.pop()
            if seen.get(term.id, -1) >= d:
                continue
            seen[term.id] = d
            ps = parents(term)
            if not ps:
                best = max(best, d)
            stack.extend((p, d + 1) for p in ps)
        cache[go_id] = best
        return best

    return depth


@hydra.main(version_base=None, config_path="../configs/preprocess", config_name="base")
def main(cfg: DictConfig) -> None:
    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer

    print(OmegaConf.to_yaml(cfg))
    tiers = sorted(cfg.label_tiers)
    if tiers not in ([1], [1, 2]):
        raise SystemExit(f"label_tiers must be [1] or [1, 2], got {tiers}")

    # ------------------------------------------------------------------ load
    root = dataset_root(cfg)
    rows = load_source(root / "proteins", "train-*.parquet", "proteins", cfg.include_v1_vintage)
    extra = load_source(root / "proteins_extra", "train-*.parquet", "extra", cfg.include_v1_vintage)
    if rows.keys() & extra.keys():
        raise SystemExit("proteins and proteins_extra share protein ids")
    test_ids = set()
    for f in sorted((root / "proteins").glob("test-*.parquet")):
        test_ids.update(pq.read_table(f, columns=["protein_id"])["protein_id"].to_pylist())
    excluded = extra.keys() & test_ids
    if excluded:
        print(f"excluded {len(excluded):,} proteins/test ids from proteins_extra/train")
    rows.update({p: r for p, r in extra.items() if p not in excluded})
    del extra

    ids_by_source = {s: sorted(p for p, r in rows.items() if r["source"] == s) for s in ("proteins", "extra")}
    if cfg.limit_proteins:  # smoke tests only: the first N proteins of each source
        ids_by_source = {s: ids[: cfg.limit_proteins] for s, ids in ids_by_source.items()}
        rows = {p: rows[p] for ids in ids_by_source.values() for p in ids}
        print(f"limit_proteins={cfg.limit_proteins}: SMOKE TEST, not a training dataset")

    def labels(row, aspect, label_tiers):
        terms = aslist(row[f"go_{aspect.lower()}"])
        if 2 in label_tiers:
            terms = terms + aslist(row[f"go_{aspect.lower()}_tier2"])
        return terms

    # ------------------------------------------------------------ validation
    # Per source, a seeded sample of proteins with tier-1 labels. proteins_extra
    # candidates must also be < 95% identical to their nearest neighbour in the corpus
    # (nn_identity_v2), so validation cannot be inflated by near-duplicates.
    val_ids = set()
    for source, n_val in (("proteins", cfg.val_per_source.proteins), ("extra", cfg.val_per_source.extra)):
        cands = [p for p in ids_by_source[source] if any(labels(rows[p], a, [1]) for a in ASPECTS)]
        if source == "extra":
            cands = [p for p in cands if rows[p]["nn_identity_v2"] is not None
                     and rows[p]["nn_identity_v2"] < 0.95]
        random.Random(cfg.seed).shuffle(cands)
        if len(cands) < n_val:
            raise SystemExit(f"{source}: only {len(cands)} validation candidates for {n_val}")
        val_ids.update(cands[:n_val])
    train_ids = [p for p in ids_by_source["proteins"] + ids_by_source["extra"] if p not in val_ids]
    print(f"{len(train_ids):,} train proteins, {len(val_ids):,} validation proteins")

    # ------------------------------------------------------------ vocabulary
    depth = go_depth_function(resolve(cfg.obo_file))
    vocab = {}
    for aspect in ASPECTS:
        counts = Counter()
        for pid in train_ids:
            counts.update(set(labels(rows[pid], aspect, [1, 2])))
        vocab[aspect] = {t for t, c in counts.items() if c >= cfg.min_freq}
        print(f"{aspect}: {len(vocab[aspect]):,} of {len(counts):,} terms annotated on >= {cfg.min_freq} proteins")
    go_tokenizer = GOTermTokenizer(set().union(*vocab.values()))
    for term in go_tokenizer.token_to_id:
        if term.startswith("GO:"):
            depth(term)  # every vocabulary term must be in the ontology

    # --------------------------------------------------------------- examples
    def build(ids, validation):
        out = []
        for aspect in ASPECTS:
            for pid in sorted(ids):
                row = rows[pid]
                raw = labels(row, aspect, [1] if validation else tiers)
                terms = sorted({t for t in raw if t in vocab[aspect]}, key=lambda t: (depth(t), t))
                if not terms:
                    continue
                if len(terms) > cfg.max_go_terms:
                    raise SystemExit(f"{pid} {aspect}: {len(terms)} terms > max_go_terms={cfg.max_go_terms}")
                ex = {"protein_id": pid, "sequence": row["sequence"], "aspect": aspect,
                      "go_terms": terms, "organism": taxid_key(row["taxid"]), "source": row["source"]}
                if validation:
                    ex["go_terms_list"] = sorted(set(labels(row, aspect, [1])))
                out.append(ex)
        return out

    val_examples = build(val_ids, validation=True)
    train_examples = build(train_ids, validation=False)
    max_terms = max(len(ex["go_terms"]) for ex in train_examples + val_examples)
    if cfg.max_protein_length + max_terms + 1 > cfg.block_size:
        raise SystemExit(f"protein ({cfg.max_protein_length}) + GO ({max_terms + 1}) tokens exceed "
                         f"block_size {cfg.block_size}")

    # Organism vocabulary: per training protein, top N taxonomy ids. Display names (the
    # most common UniProt organism string per id) let users pass a name instead.
    taxa = [taxid_key(rows[p]["taxid"]) for p in train_ids]
    names = {}
    for (key, name), _ in Counter((taxid_key(rows[p]["taxid"]), rows[p]["organism"]) for p in train_ids).most_common():
        if key is not None and name:
            names.setdefault(key, name)
    organism_mapper = OrganismMapper(taxa, top_n_organisms=cfg.top_n_organisms, names=names)
    unnamed = set(organism_mapper.organism_to_idx) - set(organism_mapper.names) - {"<UNKNOWN>"}
    if unnamed:
        raise SystemExit(f"no organism name for taxonomy ids {sorted(unnamed)[:10]}")

    # ---------------------------------------------------------------- tokenize
    protein_tokenizer = AutoTokenizer.from_pretrained(cfg.embed_model_path)

    def tokenize(ex):
        ids = protein_tokenizer(ex["sequence"], add_special_tokens=False, truncation=True,
                                max_length=cfg.max_protein_length, padding=False)["input_ids"]
        go = go_tokenizer.encode(ex["go_terms"], aspect=ex["aspect"])
        out = {"protein_id": ex["protein_id"], "aspect": ex["aspect"], "source": ex["source"],
               "protein_tokens": ids, "protein_length": len(ids),
               "go_input_tokens": go[:-1], "go_targets": go[1:],
               "organism": ex["organism"], "organism_id": organism_mapper.map_organism(ex["organism"])}
        if "go_terms_list" in ex:
            out["go_terms_list"] = ex["go_terms_list"]
        return out

    splits = {}
    for name, examples in (("validation", val_examples), ("train", train_examples)):
        splits[name] = Dataset.from_list(examples).map(
            tokenize, remove_columns=["sequence", "go_terms"], num_proc=cfg.num_proc,
            desc=f"tokenizing {name}", load_from_cache_file=False)

    # -------------------------------------------------------------------- save
    out_dir = resolve(cfg.output_dir)
    DatasetDict({"train": splits["train"], "validation": splits["validation"]}).save_to_disk(str(out_dir / "dataset"))
    art = out_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    go_tokenizer.to_json(art / "go_tokenizer.json")
    organism_mapper.to_json(art / "organism_mapper.json")
    info = {
        "source": cfg.hf_dataset if not cfg.data_dir else str(cfg.data_dir),
        "label_tiers": tiers,
        "min_freq": cfg.min_freq,
        "vocab_size": len(go_tokenizer),
        "vocab_per_aspect": {a: len(vocab[a]) for a in ASPECTS},
        "organism_vocab_size": organism_mapper.get_vocab_size(),
        "max_protein_length": cfg.max_protein_length,
        "max_label_terms": max_terms,
        "embed_model_path": cfg.embed_model_path,
        "n_train_proteins": len(train_ids),
        "n_val_proteins": len(val_ids),
        "n_train_examples": len(splits["train"]),
        "n_val_examples": len(splits["validation"]),
        "seed": cfg.seed,
    }
    with open(art / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
