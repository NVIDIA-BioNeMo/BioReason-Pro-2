# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Train GO-GPT.

    python scripts/train.py                                   # configs/experiment/default.yaml
    python scripts/train.py trainer.devices=1 data.batch_size=4
    python scripts/train.py resume.checkpoint_path=artifacts/training/.../checkpoints/latest.ckpt

Prepare the data first (scripts/prepare_data.py). Precomputed ESM2 embeddings
(scripts/precompute_embeddings.py) are used when data.embedding_cache_dir exists;
otherwise ESM2 runs on the fly. Outputs go to artifacts/training/<experiment>/<time>/.
"""

import os
import sys
from pathlib import Path

# Padded batch shapes vary every step; expandable segments stop the CUDA caching
# allocator from fragmenting over long runs. Must be set before CUDA initializes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Before Lightning: keeps TorchMetrics' non-commercial EED module from loading.
import gogpt.torchmetrics_guard  # noqa: E402,F401
import hydra  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint  # noqa: E402


def resolve(path):
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    from gogpt.data.dataset import load_preprocessed_data
    from gogpt.models.gogpt_lightning import LightningGOGPT

    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.pytorch.matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = cfg.pytorch.enable_tf32
    torch.backends.cudnn.allow_tf32 = cfg.pytorch.enable_tf32

    cache_dir = resolve(cfg.data.embedding_cache_dir)
    use_cache = cache_dir is not None and cache_dir.exists()
    print(f"Protein embeddings: {'precomputed, from ' + str(cache_dir) if use_cache else 'ESM2 on the fly'}")

    tokenizer, train_loader, val_loader, info = load_preprocessed_data(
        resolve(cfg.data.preprocessed_path),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        embedding_cache_dir=cache_dir if use_cache else None,
        expected_embedding_dim=cfg.model.protein_embedding_dim,
        expected_embedding_model=cfg.model.embed_model_path,
        expected_embedding_layer=cfg.model.protein_layer_index,
    )
    if info["max_protein_length"] != cfg.model.max_protein_length:
        raise ValueError(f"dataset truncates proteins to {info['max_protein_length']} residues, "
                         f"model.max_protein_length is {cfg.model.max_protein_length}")

    model_config = OmegaConf.to_container(cfg.model)
    model_config.update(
        vocab_size=len(tokenizer),
        organism_vocab_size=info["organism_vocab_size"],
        pad_token_id=tokenizer.pad_token_id,
        mf_start_token_id=tokenizer.mf_start_token_id, mf_end_token_id=tokenizer.mf_end_token_id,
        bp_start_token_id=tokenizer.bp_start_token_id, bp_end_token_id=tokenizer.bp_end_token_id,
        cc_start_token_id=tokenizer.cc_start_token_id, cc_end_token_id=tokenizer.cc_end_token_id,
    )
    model = LightningGOGPT(
        model_config=model_config,
        tokenizer=tokenizer,
        obo_file=str(resolve(cfg.validation.obo_file)),
        ia_file=str(resolve(cfg.validation.ia_file)),
        use_precomputed_embeddings=use_cache,
        **OmegaConf.to_container(cfg.training),
    )
    print(f"GO-GPT decoder: {model.model.get_num_params() / 1e6:.1f}M parameters (excluding token embeddings)")

    callbacks = [ModelCheckpoint(**OmegaConf.to_container(c)) for c in cfg.checkpoints]
    logger = False
    if cfg.wandb.enabled:
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(project=cfg.wandb.project, name=cfg.wandb.name, save_dir=".")
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = pl.Trainer(callbacks=callbacks, logger=logger, default_root_dir=".",
                         **OmegaConf.to_container(cfg.trainer))
    ckpt = resolve(cfg.resume.checkpoint_path)
    trainer.fit(model, train_loader, val_loader, ckpt_path=str(ckpt) if ckpt else None)

    if trainer.is_global_zero:
        OmegaConf.save(cfg, "final_config.yaml")
        best = next((c for c in callbacks if isinstance(c, ModelCheckpoint) and c.monitor), None)
        if best is not None and best.best_model_path:
            print(f"Best checkpoint ({best.monitor}={best.best_model_score:.4f}): {best.best_model_path}")
        print(f"Outputs in {os.getcwd()}")


if __name__ == "__main__":
    train()
