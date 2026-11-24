# GO-GPT v2

GO-GPT predicts Gene Ontology (GO) terms from a protein sequence. A frozen ESM2 protein language
model embeds the sequence; a GPT-style decoder then generates the GO terms of one aspect at a time
(Molecular Function, Biological Process or Cellular Component) as a token sequence, conditioned on
the protein and its organism. GO-GPT's predictions are one of the inputs of BioReason-Pro 2.

Model weights: [wanglab/gogpt-v2](https://huggingface.co/wanglab/gogpt-v2): `main` holds the epoch-75
checkpoint, whose predictions BioReason-Pro 2 was trained with, and the `ep89` branch a later one.
Training data: [wanglab/bioreason-pro-v2](https://huggingface.co/datasets/wanglab/bioreason-pro-v2).

## Changes from v1

| | v1 ([wanglab/gogpt](https://huggingface.co/wanglab/gogpt)) | v2 |
|---|---|---|
| training data | CAFA5-based | `wanglab/bioreason-pro-v2` `proteins` + `proteins_extra`, tier-1 and tier-2 labels |
| protein encoder | ESM2-3B, layer 30 | ESM2-650M, layer 27; sequences up to 2,000 residues |
| decoder | 12 layers, LayerNorm | 8 layers, 900 dims, RMSNorm, QK-norm, gated attention |
| training | cross-entropy | + label smoothing and masked GO inputs (MEAP); validation on CAFA wF1 |
| organism input | species name | NCBI taxonomy id or name; 200 organisms |
| default decoding | beam search | greedy (deterministic; what the results below use) |
| weights | pickled Lightning checkpoint | `model.safetensors` + JSON configs |

v2 also fixes three problems in the v1 inference code: it encoded proteins with ESM2's `<cls>`
and `<eos>` tokens although training never used them, it loaded weights with `strict=False`, and
species names that were not exact keys of the organism table silently fell back to the
"unknown organism" embedding.

## Installation

From the repository root:

```bash
pip install -e gogpt              # inference
pip install -e "gogpt[train]"     # + data preparation and training
```

Python 3.11+, PyTorch 2.6+ and transformers 4.51+. A GPU is recommended; the model also runs on
CPU. The first prediction downloads GO-GPT (0.75 GB) and ESM2-650M (2.6 GB) from the Hugging
Face Hub.

## Usage

```python
from gogpt import GOGPTPredictor

predictor = GOGPTPredictor.from_pretrained("wanglab/gogpt-v2")

predictor.predict("MERPEPELIRQSWRAVSRSPLEHGTVLFARLFALEPDLLPLFQYNCRQFSSPEDCLSSPEFLDHIRKVMLVIDAAVTNVEDLSSLEEYLASLGRKHRAVGVKLSSFSTVGESLLYMLEKCLGPAFTPATRAAWSQLYGAVVQAMSRGWDGE",
                  organism=9606)
# {"MF": ["GO:0003674", "GO:0003824", "GO:0005488", ...], "BP": [...], "CC": [...]}

predictor.predict_batch(sequences, organisms=[9606, "Mus musculus", None], batch_size=16)
```

- Terms are returned per aspect in generation order, which runs from general to specific. As in
  the training labels, the set includes ancestors of the specific terms.
- `organism` is an NCBI taxonomy id (`9606`, `"9606"`) or an organism name (`"Homo sapiens"`,
  `"Homo sapiens (Human)"`). GO-GPT has embeddings for the 200 organisms most frequent in its
  training data (`predictor.organism_mapper.names`); any other organism, or none, uses a shared
  "unknown organism" embedding and issues a warning.
- Sequences longer than 2,000 residues are truncated, as in training.
- Decoding: `decoding="greedy"` (default), `"beam"` (`beam_size`, `length_penalty`) or
  `"sample"` (`temperature`, `top_k`).
- On GPU the model runs under bf16 autocast, as in training; pass `precision="fp32"` to change
  that. On CPU it runs in fp32.

`gogpt.utils.go_ontology.format_predictions` renders predictions as the text block BioReason-Pro
2 reads in its prompt. [`notebooks/inference.ipynb`](notebooks/inference.ipynb) walks through
all of the above.

From the command line:

```bash
python scripts/predict.py --fasta proteins.fasta --organism 9606 --output predictions.tsv
python scripts/predict.py --fasta proteins.fasta --organisms organisms.tsv --output predictions.tsv
```

The output has one row per predicted term: `protein_id, aspect, go_id, go_name, rank`.

## Results

Greedy decoding, on the two held-out test sets of `wanglab/bioreason-pro-v2` (no test protein
was used in training). Each protein is scored on its `scored_aspects` against tier-1 labels with
the CAFA metric ([`src/gogpt/utils/cafa.py`](src/gogpt/utils/cafa.py), the one validation reports):
predictions and labels are propagated to their ancestors, and
wF1 is the F1 weighted by information accretion: CAFA 6's, which does not ship;
`scripts/data/download_assets.py` fetches it to `data/IA_CAFA6.tsv` at the repository root.

| test set | proteins (protein-aspect pairs) | checkpoint | MF wF1 | BP wF1 | CC wF1 | mean wF1 | mean F1 |
|---|---|---|---|---|---|---|---|
| `proteins_extra` / `test` | 6,112 (9,748) | ep75 (`main`) | 0.635 | 0.403 | 0.618 | **0.552** | 0.639 |
| | | ep89 | 0.644 | 0.415 | 0.628 | **0.562** | 0.649 |
| `proteins` / `test` | 3,726 (5,186) | ep75 (`main`) | 0.654 | 0.591 | 0.701 | **0.649** | 0.715 |
| | | ep89 | 0.668 | 0.614 | 0.715 | **0.666** | 0.728 |

Against tier-1 ∪ tier-2 labels (what the model is trained on) the mean wF1 is 0.618 and 0.691 for
ep75, 0.628 and 0.707 for ep89.

## Training

The released model was trained with the configuration in
[`configs/experiment/default.yaml`](configs/experiment/default.yaml) on 4 H100 GPUs for 100
epochs. `main` on the Hub is epoch index 74, the checkpoint BioReason-Pro 2's prompts were built
with; the `ep89` branch is the one with the best validation wF1 (epoch index 89).

**1. Prepare the data.** Downloads `wanglab/bioreason-pro-v2`, holds out 8,000 validation
proteins, builds the GO vocabulary (terms on at least 20 training proteins: 13,453 terms) and
tokenizes.

```bash
python scripts/prepare_data.py                     # -> artifacts/preprocessed/v2/
```

**2. Precompute ESM2 embeddings (optional).** ESM2 is frozen, so its embeddings can be computed
once instead of in every training step. Needs ~370 GB of disk. Without this step training runs
ESM2 on the fly.

```bash
for i in 0 1 2 3; do   # one process per GPU
    CUDA_VISIBLE_DEVICES=$i python scripts/precompute_embeddings.py --shard-index $i --num-shards 4 &
done; wait
python scripts/precompute_embeddings.py --finalize
```

**3. Train.** Uses the precomputed embeddings if `data.embedding_cache_dir` exists, ESM2 on the fly
otherwise. Outputs go to `artifacts/training/`.

```bash
python scripts/train.py
python scripts/train.py trainer.devices=1 data.batch_size=4      # Hydra overrides
python scripts/train.py resume.checkpoint_path=<path>/checkpoints/latest.ckpt
```

Validation decodes every validation protein greedily and reports CAFA F1 and IA-weighted F1
(wF1) per aspect against the tier-1 labels; `val_wf1` is their mean.

**4. Export** a checkpoint to the layout `GOGPTPredictor` loads:

```bash
python scripts/export_model.py --checkpoint <path>.ckpt \
    --artifacts artifacts/preprocessed/v2/artifacts --output-dir artifacts/export/my-model
```

```python
predictor = GOGPTPredictor("artifacts/export/my-model")
```

## Tests

```bash
pip install -e "gogpt[train,dev]"
cd gogpt && pytest            # unit tests, CPU, no downloads
pytest -m slow                # end to end with the released model (downloads ~3.4 GB)
```

## Model

| | |
|---|---|
| protein encoder | `facebook/esm2_t33_650M_UR50D`, frozen; `hidden_states[27]`, no special tokens |
| decoder | 8 layers, 12 heads, 900 dims; 198M parameters, 12M of them the GO token embeddings (tied with the output layer) |
| attention | protein positions attend to each other; GO tokens attend to all protein positions and to earlier GO tokens |
| GO vocabulary | 13,453 terms (MF 2,501, BP 9,493, CC 1,459) + 7 special tokens; GO release 2023-01-01 |
| organisms | 200 NCBI taxonomy ids + unknown |
| max length | 2,000 residues + 1,000 GO tokens |

## Layout

```
gogpt/
├── configs/                   # Hydra configs: preprocess/base.yaml, experiment/default.yaml
├── data/go-basic.obo          # GO release 2023-01-01
├── notebooks/inference.ipynb
├── scripts/                   # prepare_data, precompute_embeddings, train, export_model, predict
├── src/gogpt/
│   ├── inference.py           # GOGPTPredictor
│   ├── models/                # gogpt.py (decoder), protein_encoder.py (ESM2), gogpt_lightning.py
│   ├── data/                  # dataset, GO tokenizer, precomputed embeddings
│   └── utils/                 # organism mapper, GO names and formatting, CAFA metric
└── tests/
```
