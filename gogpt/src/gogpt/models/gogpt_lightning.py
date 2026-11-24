# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import math

import gogpt.torchmetrics_guard  # noqa: F401  (before the Lightning import below)
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LambdaLR

from gogpt.config.model_config import GOGPTConfig
from gogpt.models.gogpt import GOGPT
from gogpt.utils.cafa import evaluate_binary

ENCODER_PREFIX = "protein_encoder."


class LightningGOGPT(pl.LightningModule):
    """Training wrapper.

    Protein embeddings come from the batch when the dataloader serves precomputed ones,
    otherwise from a frozen ESM2 encoder run on the batch's protein tokens. The encoder
    is only built in the second case and is never written to checkpoints.

    Validation decodes every example greedily and reports CAFA F1 and IA-weighted F1
    (wF1) per aspect against the raw (vocabulary-unfiltered) ground truth; `val_wf1` is
    the mean over the three aspects.
    """

    def __init__(self, model_config, tokenizer, obo_file, ia_file, use_precomputed_embeddings,
                 learning_rate=3e-4, weight_decay=0.2, betas=(0.9, 0.95), warmup_fraction=0.05,
                 min_lr_ratio=0.1, label_smoothing=0.0, meap_mask_ratio=0.0, val_max_new_tokens=100):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.config = GOGPTConfig.from_dict(dict(model_config))
        self.model = GOGPT(self.config)
        self.tokenizer = tokenizer
        self.protein_encoder = None
        if not use_precomputed_embeddings:
            from gogpt.models.protein_encoder import ESM2Encoder

            self.protein_encoder = ESM2Encoder(self.config.embed_model_path,
                                               self.config.protein_layer_index,
                                               self.config.max_protein_length)
        self._val_gt, self._val_pred = [], []

    # ----------------------------------------------------------------------- helpers

    def _protein_embeddings(self, batch):
        if "protein_embeddings" in batch:
            return batch["protein_embeddings"]
        if self.protein_encoder is None:
            raise RuntimeError("batch has no protein_embeddings and no protein encoder was built")
        return self.protein_encoder(batch["protein_tokens"], batch["protein_mask"])

    def _aspect_of(self, start_token):
        for aspect, (start, _) in self.tokenizer.aspect_to_tokens.items():
            if start == start_token:
                return aspect
        raise ValueError(f"GO sequence does not begin with an aspect START token: {start_token}")

    # ---------------------------------------------------------------------- training

    def training_step(self, batch, batch_idx):
        go_tokens = batch["go_tokens"]
        if self.hparams.meap_mask_ratio > 0:
            # Mask-enhanced training: replace a random fraction of the GO *inputs* with
            # <pad> (never the START token or padding); the targets are unchanged.
            go_tokens = go_tokens.clone()
            r = torch.rand(go_tokens.shape, device=go_tokens.device)
            r[:, 0] = 1.0
            r[~batch["go_mask"]] = 1.0
            go_tokens[r < self.hparams.meap_mask_ratio] = self.config.pad_token_id
        _, loss = self.model(
            go_tokens=go_tokens,
            protein_embeddings=self._protein_embeddings(batch),
            protein_mask=batch["protein_mask"],
            go_mask=batch["go_mask"],
            organism_id=batch["organism_id"],
            targets=batch["targets"],
            label_smoothing=self.hparams.label_smoothing,
        )
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    # -------------------------------------------------------------------- validation

    def validation_step(self, batch, batch_idx):
        emb = self._protein_embeddings(batch)
        _, loss = self.model(
            go_tokens=batch["go_tokens"], protein_embeddings=emb, protein_mask=batch["protein_mask"],
            go_mask=batch["go_mask"], organism_id=batch["organism_id"], targets=batch["targets"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=batch["go_tokens"].size(0))

        start_tokens = batch["go_tokens"][:, :1]
        generated = self.model.generate(
            protein_embeddings=emb, protein_mask=batch["protein_mask"], go_tokens=start_tokens,
            max_new_tokens=self.hparams.val_max_new_tokens, temperature=0.0,
            organism_id=batch["organism_id"])
        for i, pid in enumerate(batch["protein_ids"]):
            aspect = self._aspect_of(start_tokens[i, 0].item())
            uid = f"{pid}_{aspect}"
            self._val_gt += [(uid, t) for t in batch["go_terms_list"][i] if t.startswith("GO:")]
            self._val_pred += [(uid, t) for t in self.tokenizer.decode(generated[i].tolist(), aspect)]

    def on_validation_epoch_end(self):
        gt, pred = self._val_gt, self._val_pred
        self._val_gt, self._val_pred = [], []
        if self.trainer.world_size > 1:
            import torch.distributed as dist

            gathered = [None] * self.trainer.world_size
            dist.all_gather_object(gathered, (gt, pred))
            gt = [pair for g, _ in gathered for pair in g]
            pred = [pair for _, p in gathered for pair in p]

        # [f_mf, wf_mf, f_bp, wf_bp, f_cc, wf_cc]; rank 0 computes, then broadcasts so
        # every rank logs the same values (ModelCheckpoint monitors them on all ranks).
        metrics = torch.zeros(6, device=self.device)
        if self.trainer.is_global_zero and gt:
            res = evaluate_binary(gt, pred, self.hparams.obo_file, self.hparams.ia_file)
            for j, aspect in enumerate(("MF", "BP", "CC")):
                metrics[2 * j] = res.get(aspect, {}).get("f", 0.0)
                metrics[2 * j + 1] = res.get(aspect, {}).get("f_w", 0.0)
        if self.trainer.world_size > 1:
            torch.distributed.broadcast(metrics, src=0)

        # Identical on every rank after the broadcast, so syncing is a no-op on the values.
        for j, aspect in enumerate(("mf", "bp", "cc")):
            self.log(f"val_f1_{aspect}", metrics[2 * j], sync_dist=True)
            self.log(f"val_wf1_{aspect}", metrics[2 * j + 1], sync_dist=True)
        self.log("val_f1", metrics[0::2].mean(), sync_dist=True)
        self.log("val_wf1", metrics[1::2].mean(), prog_bar=True, sync_dist=True)

    # -------------------------------------------------------------------- checkpoints

    def on_save_checkpoint(self, checkpoint):
        # The frozen encoder is reloaded from its pretrained weights; don't store it.
        sd = checkpoint["state_dict"]
        for k in [k for k in sd if k.startswith(ENCODER_PREFIX)]:
            del sd[k]

    def on_load_checkpoint(self, checkpoint):
        # Put the (unsaved) encoder weights back so strict loading still checks every
        # decoder key.
        if self.protein_encoder is not None:
            sd = checkpoint["state_dict"]
            for k, v in self.protein_encoder.state_dict().items():
                sd.setdefault(ENCODER_PREFIX + k, v)

    # ---------------------------------------------------------------------- optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay,
                                      betas=tuple(self.hparams.betas))
        total = self.trainer.estimated_stepping_batches
        warmup = int(total * self.hparams.warmup_fraction)
        floor = self.hparams.min_lr_ratio

        def lr_lambda(step):
            # Linear warmup, then cosine decay from 1 to min_lr_ratio.
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": LambdaLR(optimizer, lr_lambda), "interval": "step"}}
