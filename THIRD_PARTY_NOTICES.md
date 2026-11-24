# Third-party notices

This repository contains material from the projects below. Each keeps its own license; the rest of
the repository is under the MIT license in [`LICENSE`](LICENSE).

Material the code downloads at run time (checkpoints, datasets, the ESM3 encoder, Python packages)
is not distributed with this repository and is governed by its own terms; see the notice in the
README and, for the packages and models, [Run-time dependencies](#run-time-dependencies) below.

## BioReason-Pro v1

- Source: https://github.com/bowang-lab/BioReason-Pro
- License: MIT
- Copyright: Copyright (c) 2026 Adibvafa Fallahpour

Code files carried over from v1 and changed since (each carries v1's copyright header above the
NVIDIA one):

```
bioreason2/dataset/cafa5/collate.py
bioreason2/dataset/cafa5/format.py
bioreason2/dataset/cafa5/load.py
bioreason2/dataset/utils.py
bioreason2/models/__init__.py
bioreason2/models/go_graph_encoder.py
bioreason2/models/pl/chat_template_pl.py
bioreason2/models/pl/processing_pl.py
bioreason2/models/protein_encoder.py
bioreason2/models/protein_llm.py
bioreason2/models/protein_vllm.py
bioreason2/utils/save_hf_ckpt.py
eval.py
evals/cafa/cafa_evals.py
gogpt/scripts/prepare_data.py
gogpt/scripts/train.py
gogpt/src/gogpt/__init__.py
gogpt/src/gogpt/config/model_config.py
gogpt/src/gogpt/data/dataset.py
gogpt/src/gogpt/data/tokenizer.py
gogpt/src/gogpt/inference.py
gogpt/src/gogpt/models/gogpt.py
gogpt/src/gogpt/models/gogpt_lightning.py
gogpt/src/gogpt/utils/organism_mapper.py
gogpt_api.py
predict.py
scripts/data/download_assets.py
scripts/eval/eval.sh
scripts/sft/train.sh
train_protein_llm.py
```

Code files carried over from v1 unchanged (each carries v1's copyright header only):

```
bioreason2/__init__.py
bioreason2/dataset/__init__.py
bioreason2/dataset/cafa5/__init__.py
bioreason2/models/pl/__init__.py
bioreason2/models/special_tokens.py
bioreason2/utils/__init__.py
bioreason2/utils/argparse_utils.py
evals/__init__.py
```

Other files from v1: `gogpt/.gitignore` unchanged; `pyproject.toml`, the GO-GPT configs,
README and inference notebook changed. The chat template and the Gene Ontology file below also
came through v1.

```
MIT License

Copyright (c) 2026 Adibvafa Fallahpour

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Qwen3 chat template

- File: `bioreason2/models/pl/chat_template.jinja2`
- Source: the chat template of Qwen3 (https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507,
  `tokenizer_config.json`), modified in BioReason-Pro v1 to render the protein and GO-graph
  placeholders and list-form message content.
- License: Apache License 2.0, full text in [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)
- Copyright: Copyright (c) Alibaba Cloud (the Qwen team)

## SkyRL

- File: `skyrl/bioreason.patch`, a modification of SkyRL at commit
  `50a570c07e105b324778d357c1f304d4c03a8156` that carries lines of SkyRL's source as context. The files
  it changes are the ones it names; `skyrl/README.md` describes the changes.
- Source: https://github.com/NovaSky-AI/SkyRL
- License: Apache License 2.0, full text in [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)
- Copyright: Copyright (c) the SkyRL authors

## Gene Ontology

- Files: `bioreason2/dataset/go-basic.obo` and `gogpt/data/go-basic.obo` (identical copies),
  release 2023-01-01 (`data-version: releases/2023-01-01`), unmodified.
- Source: the Gene Ontology Consortium, https://geneontology.org
- License: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/
- Attribution: Gene Ontology data from the GO Consortium (geneontology.org), CC BY 4.0.

## UniProt

- Files: `data/examples/proteins.tsv` (protein accessions, organisms, taxonomy ids and
  sequences), `data/eval/judge_subset.txt`, `data/rl/*.txt` and `data/splits/train_excluded.txt` (UniProt accessions), the
  worked examples in `synth/v5/` and `synth/v6/` (UniProt annotation of the example proteins),
  and `evals/grounding/grounding_reference.json` (the domain of life of the test proteins'
  organisms and viral hosts, from UniProt taxonomy and UniProtKB).
  Derived from UniProtKB; changed by selection and formatting.
- Source: the UniProt Consortium, https://www.uniprot.org
- License: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/
- Attribution: UniProt Consortium, UniProt: the Universal Protein Knowledgebase,
  https://www.uniprot.org, CC BY 4.0.

## InterPro

- Files: the worked examples in `synth/v5/` and `synth/v6/` quote InterPro entries and
  InterProScan member-database matches of the example proteins, including predictions of the
  member tools Phobius and TMHMM, whose terms are set by their providers.
  `evals/grounding/grounding_reference.json` lists retired InterPro entries and the
  member-database signatures they held.
- Source: EMBL-EBI InterPro, https://www.ebi.ac.uk/interpro
- License: InterPro data are freely available (CC0 1.0).

## STRING

- Files: the worked examples in `synth/v5/` and `synth/v6/` list the example proteins' known
  interaction partners.
- Source: STRING, https://string-db.org
- License: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/
- Attribution: STRING database (string-db.org), CC BY 4.0.

## Goodfire banner

- Files: `docs/banner.png` (the banner at the top of the README).
- Source: Goodfire, from the BioReason-Pro 2 blog post,
  https://www.goodfire.com/research/building-stronger-biological-reasoning-models/
- License: © Goodfire, used with permission. Not covered by this repository's MIT license.

## Run-time dependencies

None of the following is in this repository: pip installs the packages and the code downloads the
models and data on first use, each under its own license. All of the packages allow commercial use.
Most are permissive (MIT, BSD, Apache-2.0). The table lists `esm`, whose license depends on its
version, and the packages that carry further conditions:

| package | used by | license |
|---|---|---|
| `esm` 3.4.0 | the ESM3 encoder, in training and inference | MIT (Copyright 2026 Chan Zuckerberg Biohub, Inc.); the weights it loads are listed below |
| `func_timeout` (RL, via SkyRL), `pycountry` (via vLLM) | as noted | LGPL-2.1 |
| `tqdm`, `certifi` | throughout | MPL-2.0 |
| NVIDIA CUDA wheels (`nvidia-*`, `cuda-python`) | GPU runtime | NVIDIA software license agreements |

The RL environment (`scripts/rl/build_env.sh`) runs on the NeMo container (NVIDIA's container terms)
and adds SkyRL with this repository's patch (Apache-2.0, above) and these packages: `ray`, `peft`,
`accelerate`, `hf_transfer` and `tensorboard` (Apache-2.0); `hydra-core`, `wandb`, `loguru`,
`tensordict`, `jaxtyping`, `polars` and `fastapi` (MIT); `omegaconf`, `torchdata`, `s3fs`, `uvicorn` and
`colorama` (BSD-3-Clause); `func_timeout` (LGPL-2.1, above).

Models and data:

| | Hub | license |
|---|---|---|
| Qwen3.5-9B (backbone) | `Qwen/Qwen3.5-9B` | Apache-2.0 |
| ESM3 open small (protein encoder) | `biohub/esm3-sm-open-v1` | MIT |
| ESM2-650M (GO-GPT's encoder) | `facebook/esm2_t33_650M_UR50D` | MIT |
| GO-GPT | `wanglab/gogpt-v2` | MIT |
| BioReason-Pro 2 SFT and RL | `nvidia/BioReason-Pro-2-9B-SFT`, `nvidia/BioReason-Pro-2-9B-RL` | OpenMDW-1.1 |
| raw corpus | `wanglab/bioreason-pro-v2` | CC BY 4.0 |
| GO embeddings | `wanglab/bioreason-pro-go-embeddings` | Apache-2.0 |
| structures | `wanglab/bioreason-pro-structures` | AlphaFold DB structures, CC BY 4.0 |
| CAFA 6 IA weights | Kaggle, `cafa-6-protein-function-prediction` | the competition's rules; no license stated |
