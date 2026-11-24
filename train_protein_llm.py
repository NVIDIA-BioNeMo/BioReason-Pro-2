# SPDX-FileCopyrightText: Copyright (c) 2026 Adibvafa Fallahpour
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Set envioronment variables
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_CUMEM_ENABLE"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Before Lightning: keeps TorchMetrics' non-commercial EED module from loading.
import bioreason2.utils.torchmetrics_guard  # noqa: F401

# Standard library imports
import collections
import os
from argparse import ArgumentParser
from functools import partial
from typing import Optional, Dict

# Third-party imports
import pytorch_lightning as pl
import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    DeviceStatsMonitor,
    Callback,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy
from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from transformers import logging

# Local imports
import ctypes
import multiprocessing as mp

from bioreason2.dataset.cafa5.collate import qwen_protein_collate_fn
from bioreason2.models import model_family as _mf
from bioreason2.dataset.cafa5.context_dropout import (
    ContextDropoutConfig,
    ContextDropoutDataset,
    ContextDropoutEpochCallback,
)
from bioreason2.dataset.cafa5.load import load_cafa5_dataset
from bioreason2.models.pl.processing_pl import PLProcessor
from bioreason2.models.protein_llm import ProteinLLMModel
from bioreason2.utils import str2bool

# Set start method to 'spawn' for CUDA compatibility with multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
logging.set_verbosity_error()

GRADIENT_CLIP_VAL = 1.0

# Frozen ESM3 encoder: never trained, rebuilt from pretrained, excluded from saves.
PROTEIN_ENCODER_PREFIX = "model.protein_model."

# (state-dict prefix, filename, required) for the stage 1 -> stage 2 hand-off.
STAGE1_ARTIFACTS = (
    ("model.protein_projection.", "projector_weights.pt", True),
    ("model.go_projection.", "go_projection_weights.pt", False),
    ("model.go_encoder.", "go_encoder_weights.pt", False),
)


class EpochCheckpointFromN(Callback):
    """Custom callback to save checkpoints every epoch starting from a specific epoch."""
    
    def __init__(self, checkpoint_dir, run_name, start_epoch):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.start_epoch = start_epoch
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Save checkpoint at the end of each epoch if epoch >= start_epoch."""
        if trainer.current_epoch >= self.start_epoch:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{self.run_name}-epoch={trainer.current_epoch:02d}.ckpt"
            )
            trainer.save_checkpoint(checkpoint_path)
            print(f"✓ Saved epoch checkpoint: {checkpoint_path}")


class ProteinLLMFineTuner(pl.LightningModule):
    """
    PyTorch Lightning module for fine-tuning Protein-LLM models.
    """

    def __init__(self, hparams, train_dataset=None, val_dataset=None, test_dataset=None):
        """
        Initialize the ProteinLLMFineTuner.

        Args:
            hparams: Hyperparameters for the model and training
            train_dataset: Pre-loaded training dataset
            val_dataset: Pre-loaded validation dataset
            test_dataset: Pre-loaded test dataset
        """
        super().__init__()
        self.save_hyperparameters(hparams)

        self.text_model_name = self.hparams.text_model_name
        self.protein_model_name = self.hparams.protein_model_name
        self.cache_dir = self.hparams.cache_dir
        self.learning_rate = self.hparams.learning_rate
        self.weight_decay = self.hparams.weight_decay
        self.warmup_ratio = self.hparams.warmup_ratio
        self.text_model_finetune = self.hparams.text_model_finetune
        self.protein_embedding_layer = self.hparams.protein_embedding_layer
        self.go_model_finetune = self.hparams.go_model_finetune
        self.attn_implementation = self.hparams.attn_implementation
        self.go_obo_path = self.hparams.go_obo_path
        self.precomputed_embeddings_path = self.hparams.precomputed_embeddings_path
        self.go_hidden_dim = self.hparams.go_hidden_dim
        self.go_num_gat_layers = self.hparams.go_num_gat_layers
        self.go_num_heads = self.hparams.go_num_heads
        self.go_num_reduced_embeddings = self.hparams.go_num_reduced_embeddings
        self.go_embedding_dim = self.hparams.go_embedding_dim
        self.lora_rank = self.hparams.lora_rank
        self.lora_alpha = self.hparams.lora_alpha
        self.lora_dropout = self.hparams.lora_dropout
        self.max_length_protein = self.hparams.max_length_protein
        self.max_length_text = self.hparams.max_length_text
        self.return_answer_in_batch = self.hparams.return_answer_in_batch
        self.training_stage = self.hparams.training_stage
        self.projector_checkpoint_path = self.hparams.projector_checkpoint_path
        self.go_projection_checkpoint_path = self.hparams.go_projection_checkpoint_path
        self.go_encoder_checkpoint_path = self.hparams.go_encoder_checkpoint_path
        self.every_n_train_steps = self.hparams.every_n_train_steps
        self.protein_projection_layernorm = self.hparams.protein_projection_layernorm
        self.protein_projection_norm = self.hparams.protein_projection_norm
        self.protein_projection_norm_target = self.hparams.protein_projection_norm_target

        # Store datasets
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._test_dataset = test_dataset

        # Create quantization config if QLoRA is enabled
        quantization_config = None
        if self.hparams.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.hparams.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=self.hparams.bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=self.hparams.bnb_4bit_use_double_quant,
            )
            print(f"🔧 QLoRA enabled with {self.hparams.bnb_4bit_quant_type} quantization")

        # ESM3 materializes `_structure_encoder` lazily, on the first protein that
        # carries coordinates. It therefore EXISTS when a checkpoint is written and
        # is ABSENT in a freshly built model, so a strict resume rejects those keys
        # and every restart silently became a restart from step 0. The encoder is
        # frozen and never trained, so ignoring it on load is exactly right.
        self.strict_loading = False

        # Load model
        self.model = ProteinLLMModel(
            text_model_name=self.text_model_name,
            protein_model_name=self.protein_model_name,
            cache_dir=self.cache_dir,
            max_length_protein=self.max_length_protein,
            max_length_text=self.max_length_text,
            text_model_finetune=self.text_model_finetune,
            protein_embedding_layer=self.protein_embedding_layer,
            go_model_finetune=self.go_model_finetune,
            attn_implementation=self.attn_implementation,
            go_obo_path=self.go_obo_path,
            precomputed_embeddings_path=self.precomputed_embeddings_path,
            go_hidden_dim=self.go_hidden_dim,
            go_num_gat_layers=self.go_num_gat_layers,
            go_num_heads=self.go_num_heads,
            go_num_reduced_embeddings=self.go_num_reduced_embeddings,
            go_embedding_dim=self.go_embedding_dim,
            quantization_config=quantization_config,
            protein_projection_layernorm=self.protein_projection_layernorm,
            protein_projection_norm=self.protein_projection_norm,
            protein_projection_norm_target=self.protein_projection_norm_target,
        )

        # Load projector weights if provided (for stage 2)
        if self.training_stage == 2 and self.projector_checkpoint_path and not self.hparams.ckpt_path:
            print(f"Loading projector weights from: {self.projector_checkpoint_path}")
            projector_state_dict = torch.load(self.projector_checkpoint_path, map_location=self.device)
            self.model.protein_projection.load_state_dict(projector_state_dict)
            print("✓ Projector weights loaded successfully.")

            # Also load GO projection weights if available
            if (
                self.go_projection_checkpoint_path
                and os.path.exists(self.go_projection_checkpoint_path)
                and hasattr(self.model, "go_projection")
                and self.model.go_projection is not None
            ):
                print(f"Loading GO projection weights from: {self.go_projection_checkpoint_path}")
                go_projection_state_dict = torch.load(self.go_projection_checkpoint_path, map_location=self.device)
                self.model.go_projection.load_state_dict(go_projection_state_dict)
                print("✓ GO projection weights loaded successfully.")

            # Also load GO encoder weights if available
            if (
                self.go_encoder_checkpoint_path
                and os.path.exists(self.go_encoder_checkpoint_path)
                and hasattr(self.model, "go_encoder")
                and self.model.go_encoder is not None
            ):
                print(f"Loading GO encoder weights from: {self.go_encoder_checkpoint_path}")
                go_encoder_state_dict = torch.load(self.go_encoder_checkpoint_path, map_location=self.device)
                
                # Use strict=False to handle architecture changes (old vs new GO encoder)
                missing_keys, unexpected_keys = self.model.go_encoder.load_state_dict(go_encoder_state_dict, strict=False)

                # strict=False exists to tolerate the old-vs-new reducer, but it
                # also means a checkpoint from a DIFFERENT encoder loads nothing at
                # all and only prints a warning -- stage 2 would then run on random
                # GO weights while its log says it warm-started. Anything past a
                # partial mismatch is the wrong file, not an architecture change.
                _loaded = len(self.model.go_encoder.state_dict()) - len(missing_keys)
                if _loaded == 0:
                    raise RuntimeError(
                        f"GO encoder checkpoint {self.go_encoder_checkpoint_path} "
                        f"shares no keys with this encoder -- nothing was loaded. "
                        f"{len(missing_keys)} missing, {len(unexpected_keys)} unexpected.")
                print(f"✓ GO encoder: {_loaded}/{len(self.model.go_encoder.state_dict())} tensors loaded")
                
                if missing_keys:
                    print(f"⚠️  Missing keys in GO encoder checkpoint (will be randomly initialized): {len(missing_keys)}")
                    # Only show key names for new architecture components
                    new_arch_missing = [k for k in missing_keys if "all_cross_attention_reducer" in k]
                    if new_arch_missing:
                        print(f"   - New 'all' cross-attention reducer parameters: {len(new_arch_missing)} keys")
                    other_missing = [k for k in missing_keys if "all_cross_attention_reducer" not in k]
                    if other_missing:
                        print(f"   - Other missing keys: {other_missing[:3]}{'...' if len(other_missing) > 3 else ''}")
                
                if unexpected_keys:
                    print(f"⚠️  Unexpected keys in GO encoder checkpoint (ignored): {len(unexpected_keys)}")
                
                print("✓ GO encoder weights loaded successfully (with architecture compatibility).")

        self.text_model = self.model.text_model
        self.protein_model = self.model.protein_model
        self.protein_projection = self.model.protein_projection
        self.go_projection = self.model.go_projection
        self.go_encoder = self.model.go_encoder
        self.tokenizer = self.model.text_tokenizer
        self.lora_config = self._setup_training_strategy()

        # --- Detailed Parameter Count ---
        protein_trainable = sum(p.numel() for p in self.protein_model.parameters() if p.requires_grad)
        protein_total = sum(p.numel() for p in self.protein_model.parameters())
        projector_trainable = sum(p.numel() for p in self.protein_projection.parameters() if p.requires_grad)
        text_model_trainable = sum(p.numel() for p in self.text_model.parameters() if p.requires_grad)
        embed_tokens_trainable = sum(
            p.numel() for p in self.text_model.get_input_embeddings().weight if p.requires_grad
        )
        lm_head_trainable = sum(p.numel() for p in self.text_model.get_output_embeddings().weight if p.requires_grad)

        # Count GO encoder and GO projection parameters
        go_encoder_trainable = 0
        go_projection_trainable = 0
        if hasattr(self.model, "go_encoder") and self.model.go_encoder is not None:
            go_encoder_trainable = sum(p.numel() for p in self.model.go_encoder.parameters() if p.requires_grad)
        if hasattr(self.model, "go_projection") and self.model.go_projection is not None:
            go_projection_trainable = sum(p.numel() for p in self.model.go_projection.parameters() if p.requires_grad)

        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"--- Trainable Parameters (Stage {self.training_stage}) ---")
        if protein_total > 0:
            protein_pct = (protein_trainable / protein_total) * 100
            print(f"  - Protein Model: {protein_trainable:,} / {protein_total:,} ({protein_pct:.1f}%)")
        else:
            print(f"  - Protein Model: {protein_trainable:,}")
        print(f"  - Projector MLP: {projector_trainable:,}")
        print(f"  - GO Encoder: {go_encoder_trainable:,}")
        print(f"  - GO Projection: {go_projection_trainable:,}")
        print(f"  - Text Model (LoRA): {text_model_trainable:,}")
        print(f"  - Embed Tokens: {embed_tokens_trainable:,}")
        print(f"  - LM Head: {lm_head_trainable:,}")
        print("  ----------------------------------")
        print(f"  - Total Trainable: {total_trainable:,}")
        print("------------------------------------")

    def _setup_projection_training(self):
        """
        Set up projection layers for training (always trainable).
        """
        # Protein projection: always trainable
        self.model.protein_projection.train()
        for param in self.model.protein_projection.parameters():
            param.requires_grad = True
        print("  - Protein projection: trainable")
        
        # GO projection: always trainable if available
        if hasattr(self.model, "go_projection") and self.model.go_projection is not None:
            self.model.go_projection.train()
            for param in self.model.go_projection.parameters():
                param.requires_grad = True
            print("  - GO projection: trainable")

    def _setup_training_strategy(self) -> Optional[LoraConfig]:
        """
        Configures the training strategy based on the current training stage.
        This involves freezing/unfreezing model parts and setting up LoRA if needed.
        """
        print("✓ Setting up training configuration...")
        
        # ESM3 is frozen when the model is built and never trained
        print("  - Protein encoder: keeping frozen")

        # Setup GO encoder training
        if hasattr(self.model, "go_encoder") and self.model.go_encoder is not None:
            if self.go_model_finetune:
                print("  - Enabling GO encoder training")
                self.model.go_encoder.train()
                for param in self.model.go_encoder.parameters():
                    param.requires_grad = True
            else:
                print("  - GO encoder: keeping frozen")
        
        # Setup projection layers (always trainable)
        self._setup_projection_training()
        
        # Setup text model training
        if self.text_model_finetune:
            print("  - Enabling text model LoRA training")
        else:
            print("  - Text model: keeping frozen")

        # Stage 1: Train only the projectors
        if self.training_stage == 1:
            print("Setting up for Stage 1: Projector training only.")
            # FROZEN IS `requires_grad = False`. IT IS NOT `.eval()`, AND THE
            # DIFFERENCE COSTS 60 GB.
            #
            # HuggingFace guards gradient checkpointing with
            #     if self.gradient_checkpointing and self.training:   # modeling_layers.py:60
            # so `.eval()` makes checkpointing a SILENT no-op -- the flag stays
            # True, nothing warns, and every decoder layer stores its activations.
            # Backward still traverses all 32 layers here, because the gradient
            # has to reach the protein/GO embeddings substituted at the input, so
            # the LLM being frozen saves parameter grads and saves nothing on
            # activations.
            #
            # Measured on the real corpus, 4B, 32 GPUs, seq up to 8k:
            #     74.52 GiB in use of 79.32 GiB, then
            #     torch.OutOfMemoryError: Tried to allocate 80.00 MiB
            # on stage 1. A smoke run on a small table has no long sequences and
            # never reaches this.
            #
            # Dropping `.eval()` is safe for THIS backbone family and that is why
            # it is stated rather than assumed: Qwen3.5-4B and -9B both declare
            # `attention_dropout: 0.0`, the norms are RMSNorm (stateless, no
            # train/eval branch), and there is no batchnorm in the stack. So
            # train mode differs from eval mode only in that checkpointing now
            # actually engages. A backbone WITH dropout would need
            # `.train()` plus an explicit dropout disable instead.
            _mf.set_gradient_checkpointing(
                self.model.text_model,
                enabled=getattr(self.hparams, "gradient_checkpointing", False),
            )
            self.model.text_model.train()
            for param in self.model.text_model.parameters():
                param.requires_grad = False
            if hasattr(self.model.text_model, "config"):
                self.model.text_model.config.use_cache = False
            if hasattr(self.model.text_model, "generation_config"):
                self.model.text_model.generation_config.use_cache = False
            # PROVE IT, do not assert it. The failure this replaces was a flag
            # that read True while the feature was off, so print the two things
            # that actually determine whether checkpointing runs: the module's
            # training flag, and the per-layer gradient_checkpointing attribute.
            # If a future refactor re-introduces .eval() here, this line says so
            # in the first 30 seconds of the run instead of at the OOM.
            _tm = self.model.text_model
            _layers = [m for m in _tm.modules()
                       if getattr(m, "gradient_checkpointing", False)]
            print(f"     Text model: frozen (requires_grad=False), "
                  f"training={_tm.training}, "
                  f"layers with gradient_checkpointing={len(_layers)}")
            if not _tm.training and _layers:
                print("     WARNING: eval mode disables gradient checkpointing "
                      "(modeling_layers.py: `if self.gradient_checkpointing and "
                      "self.training`). Activations will NOT be checkpointed.")
            return None  # No LoRA config in stage 1

        # Stage 2: Full model fine-tuning (with optional LoRA)
        elif self.training_stage == 2:
            print("Setting up for Stage 2: Full model fine-tuning.")
            if self.text_model_finetune:
                # Qwen3.5 is a hybrid (24 linear-attention layers with in_proj_*, 8
                # standard with q/k/v/o), so a hardcoded q/k/v/o list would leave 75%
                # of its attention unadapted -- silently. Resolve "all-linear" from the
                # live module tree instead.
                family = self.model.model_family
                target_modules = _mf.target_modules_for(family, model=self.model.text_model)
                print(f"     LoRA targets ({family}, all-linear resolved): {target_modules}")

                self.model.text_model = _mf.apply_lora(
                    family,
                    self.model.text_model,
                    r=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    target_modules=target_modules,
                    seed=self.hparams.seed,
                )

                # See model_family.set_gradient_checkpointing for the measurement
                # behind keeping it on.
                _mf.set_gradient_checkpointing(
                    self.model.text_model,
                    enabled=getattr(self.hparams, "gradient_checkpointing", False),
                )

                self.model.text_model.train()
                # MUST follow .train(), which puts the vision tower back in train mode.
                if _mf.freeze_vision_tower(family, self.model.text_model):
                    print("     Vision tower: frozen + eval (unused by protein+text)")

                # Hard-fail on an adapter gap. A target list that does not match the
                # architecture attaches LoRA to a subset, trains happily, and produces
                # a run that is NOT comparable to its baseline. Nothing else catches it.
                _mf.assert_lora_coverage(self.model.text_model, target_modules)
            
            else:
                # Keep text model frozen and in eval mode
                print("     Text model remaining frozen")
                for param in self.model.text_model.parameters():
                    param.requires_grad = False
                self.model.text_model.eval()

            return None

        else:
            raise ValueError(f"Invalid training stage: {self.training_stage}")

    def _step(self, batch: Dict, batch_idx: int, prefix: str) -> torch.Tensor:
        """
        Performs a single step for training, validation, or testing.

        Args:
            batch: Dictionary containing the batch data
            batch_idx: Integer indicating the batch index
            prefix: String indicating the step type ('train', 'val', or 'test')

        Returns:
            torch.Tensor: The computed loss for this batch
        """
        # Get batch data from the collate function
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device) if "labels" in batch else None
        protein_sequences = batch.get("protein_sequences")
        batch_idx_map = batch.get("batch_idx_map")
        structure_coords = batch.get("structure_coords")
        go_aspects = batch.get("batch_go_aspects")

        # Forward pass through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            protein_sequences=protein_sequences,
            batch_idx_map=batch_idx_map,
            structure_coords=structure_coords,
            labels=labels,
            go_aspects=go_aspects,
        )

        # Get the loss from model outputs
        loss = outputs.loss

        # Logging metrics
        self.log(
            f"{prefix}_loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Rank-0 live per-step loss for progress bar without cross-GPU sync
        if self.trainer.is_global_zero:
            self.log(
                f"{prefix}_loss",
                loss.detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                sync_dist=False,
            )
            self.log(
                "lr",
                self.lr_schedulers().get_last_lr()[0],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=False,
            )
            self.log(
                "step",
                self.global_step,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )

        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Perform a single training step."""
        return self._step(batch, batch_idx, prefix="train")

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Perform a single validation step."""
        return self._step(batch, batch_idx, prefix="val")

    def on_save_checkpoint(self, checkpoint: Dict) -> None:
        """Drop the frozen protein encoder from the checkpoint.

        ESM3 is frozen, is rebuilt from its pretrained weights on every start, and
        is never trained -- so storing it wastes space on every save and is the
        source of the resume mismatch (its `_structure_encoder` is built lazily,
        so the saved keys have no counterpart in a freshly constructed model).

        Args:
            checkpoint: the checkpoint dict, modified in place.
        """
        sd = checkpoint.get("state_dict")
        if not sd:
            return

        dropped = [k for k in sd if k.startswith(PROTEIN_ENCODER_PREFIX)]
        for key in dropped:
            del sd[key]

        if dropped and self.trainer.is_global_zero:
            print(f"checkpoint: dropped {len(dropped)} frozen "
                  f"{PROTEIN_ENCODER_PREFIX}* tensors")

    def on_before_optimizer_step(self, optimizer) -> None:
        """Clip gradients under FSDP, where the Trainer cannot do it.

        Lightning raises `gradient_clip_algorithm='norm' is currently not
        supported for FSDPPrecision`, because each rank holds only a shard and a
        local norm is not the global one. FSDP's own `clip_grad_norm_` all-reduces
        across ranks, so this keeps the same global-norm clipping the DDP runs get
        rather than silently dropping it.

        Args:
            optimizer: the optimizer about to step; unused, the norm is global.
        """
        if not isinstance(self.trainer.strategy, FSDPStrategy):
            return

        # The clip returns the pre-clip global norm, so logging it is free and is
        # the only view of gradient health: a spike or a NaN here precedes the
        # loss moving, and nothing else in this run records it.
        total_norm = self.trainer.strategy.model.clip_grad_norm_(GRADIENT_CLIP_VAL)
        self.log("grad_norm", total_norm, on_step=True, on_epoch=False,
                 prog_bar=False, sync_dist=False)

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            Tuple[List, List]: A tuple containing a list of optimizers and schedulers
        """
        # In Stage 1, we optimize the projector and GO components (if available)
        if self.training_stage == 1:
            # Collect all trainable parameters for Stage 1
            trainable_params = list(self.model.protein_projection.parameters())
            components = ["protein projector"]

            # Add GO projection parameters if available
            if hasattr(self.model, "go_projection") and self.model.go_projection is not None:
                trainable_params.extend(list(self.model.go_projection.parameters()))
                components.append("GO projection")
            
            # Add GO encoder parameters if available and trainable
            if (hasattr(self.model, "go_encoder") and self.model.go_encoder is not None and 
                self.go_model_finetune):
                trainable_params.extend(list(self.model.go_encoder.parameters()))
                components.append("GO encoder")

            component_str = " + ".join(components)
            print(f"Optimizer configured for Stage 1 ({component_str}) with LR: {self.learning_rate}")

            optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)

        else:  # Stage 2 optimizes all trainable parameters (LoRA + projector)
            trainable_params = self.parameters()
            print(f"Optimizer configured for Stage 2 (full) with LR: {self.learning_rate}")
            optimizer = AdamW(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.warmup_ratio * total_steps)
        decay_steps = total_steps - warmup_steps

        warmup = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        decay = CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=self.learning_rate * 0.1,
        )
        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, decay],
            milestones=[warmup_steps],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


    def _create_dataloader(
        self, dataset, split: str, shuffle: bool = False, return_answers: bool = False
    ) -> DataLoader:
        """Helper function to create dataloaders with common logic."""
        if dataset is None or len(dataset) == 0:
            print("No dataset provided. Creating empty dataloader")
            return DataLoader([], batch_size=self.hparams.batch_size, shuffle=shuffle)

        try:
            print(f"Creating {split} dataloader: {len(dataset)} samples")

            # Create processor
            processor = PLProcessor(
                tokenizer=self.model.text_tokenizer,
                # protein_tokenizer=None,  # ESM3 handles this internally
            )

            # Create collate function
            collate_fn = partial(
                qwen_protein_collate_fn,
                processor=processor,
                max_length_text=self.max_length_text,
                max_length_protein=self.max_length_protein,
                return_answer_in_batch=return_answers or self.return_answer_in_batch,
                supervise_eos=getattr(self.hparams, "supervise_eos", True),
            )

            return DataLoader(
                dataset,
                batch_size=self.hparams.batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn,
                num_workers=self.hparams.num_workers,
                persistent_workers=True,
                pin_memory=True,
            )

        except Exception as e:
            print(f"Failed to create dataloader: {e}")
            return DataLoader([], batch_size=self.hparams.batch_size, shuffle=shuffle)

    def train_dataloader(self) -> DataLoader:
        """Create and return the training DataLoader."""
        return self._create_dataloader(self._train_dataset, split="train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation DataLoader."""
        return self._create_dataloader(self._val_dataset, split="val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Create and return the test DataLoader."""
        return self.val_dataloader()

    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        pass

    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to ignore missing/unexpected keys from protein model.
        Since we don't care about protein model weights (frozen anyway), we can safely ignore these.
        """
        # Filter out structure encoder keys that might not be present
        filtered_state_dict = {}
        unexpected_keys = []

        for key, value in state_dict.items():
            if "_structure_encoder" in key:
                # Skip structure encoder keys - these are from ESM3 and we don't need them
                unexpected_keys.append(key)
            else:
                filtered_state_dict[key] = value

        if unexpected_keys:
            print(f"⚠️  Ignoring {len(unexpected_keys)} unexpected structure encoder keys from checkpoint")
            print("   These keys are from ESM3's structure encoder which may not be initialized in current model")
            for key in sorted(unexpected_keys)[:5]:  # Show first 5 sorted
                print(f"   - {key}")

        # Call parent's load_state_dict with strict=False to handle any other mismatches gracefully
        result = super().load_state_dict(filtered_state_dict, strict=False)

        # Log any other unexpected issues
        if result.missing_keys:
            print(f"⚠️  Missing keys in checkpoint: {len(result.missing_keys)} keys")
        if result.unexpected_keys:
            print(f"⚠️  Other unexpected keys (non-structure encoder): {len(result.unexpected_keys)} keys")

        print(f"Missing keys:\n{result.missing_keys}")
        print(f"Unexpected keys:\n{result.unexpected_keys}")

        return result


def _unify_float_dtype(model, dtype=torch.bfloat16) -> int:
    """Cast every floating-point parameter and buffer to one dtype, for FSDP.

    FSDP flattens each wrapped unit into a single tensor and rejects mixed dtypes
    ("Must flatten tensors with uniform dtype"). `use_orig_params=True` relaxes
    only `requires_grad`, and a MixedPrecision policy does not help because the
    check runs on the original parameters.

    Args:
        model: the LightningModule to cast in place.
        dtype: the dtype every floating-point tensor is cast to.

    Returns:
        Number of tensors whose dtype changed.
    """
    before = collections.Counter()
    cast = 0

    for _, param in model.named_parameters():
        if param.is_floating_point() and param.dtype != dtype:
            before[str(param.dtype)] += 1
            param.data = param.data.to(dtype)
            cast += 1

    for _, buf in model.named_buffers():
        if buf.is_floating_point() and buf.dtype != dtype:
            before[str(buf.dtype)] += 1
            buf.data = buf.data.to(dtype)
            cast += 1

    if cast:
        print(f"FSDP: cast {cast} tensors to {dtype} for a uniform flat parameter "
              f"(from {dict(before)})")
    return cast


def _resolve_strategy(name: str, model, sharding: str = "FULL_SHARD",
                      num_nodes: int = 1, gpus_per_node: int = 8,
                      prefetch: bool = False):
    """Build the Lightning strategy, expanding "fsdp" into a sharded config.

    Any other value is passed through untouched, so the DDP default every
    released checkpoint used is unchanged.

    "ddp_no_buffer_sync" is that same DDP without the per-forward buffer
    broadcast. ESM3 builds its structure encoder lazily, on the first protein that
    carries coordinates, and its EMA codebook adds 1,052,672 buffer elements; a
    rank whose first proteins had none broadcasts a smaller buffer set than the
    rest, and every rank then waits on a collective of the wrong size until NCCL
    times out. A 32-GPU run hung that way, deterministically on ranks 11 and 28.
    Nothing here needs the broadcast: the codebook is never updated (it
    asserts in training mode), and the model holds no batch-norm statistics.

    Wrap classes come from the live module tree: a hardcoded class name wraps
    nothing on a new backbone, and FSDP that wraps nothing shards nothing while
    still reporting a healthy run.

    Args:
        name: the --strategy value.
        model: the constructed LightningModule.
        sharding: FSDP sharding strategy, e.g. FULL_SHARD or HYBRID_SHARD.
        num_nodes: nodes in the job, the replication dimension of the mesh.
        gpus_per_node: GPUs per node, the sharding dimension of the mesh.
        prefetch: overlap the next layer's all-gather with current compute.

    Returns:
        The original string, a DDPStrategy, or a configured FSDPStrategy.

    Raises:
        RuntimeError: when "fsdp" is asked for but no layer class matches, or
            HYBRID_SHARD is asked for on a single node.
    """
    if name == "ddp_no_buffer_sync":
        print("DDP: find_unused_parameters=False, broadcast_buffers=False")
        return DDPStrategy(find_unused_parameters=False, broadcast_buffers=False)
    if name != "fsdp":
        return name

    layers = _mf.decoder_layer_classes(model.model.text_model)
    if not layers:
        raise RuntimeError(
            "FSDP requested but no decoder-layer class was found to wrap; "
            "sharding nothing would look like a healthy run at full memory.")
    _unify_float_dtype(model)
    print(f"FSDP: sharding {sorted(c.__name__ for c in layers)} "
          f"strategy={sharding} nodes={num_nodes} gpus/node={gpus_per_node}")

    # No activation_checkpointing_policy: --gradient_checkpointing already checkpoints
    # these same classes (model_family.set_gradient_checkpointing), and wrapping them
    # again recomputes every block twice.
    #
    # use_orig_params: LoRA leaves the base frozen and the adapters trainable
    # inside one wrapped block, which flat-parameter FSDP rejects.
    kwargs = dict(
        auto_wrap_policy=layers,
        sharding_strategy=sharding,
        state_dict_type="full",
        use_orig_params=True,
    )

    # HYBRID_SHARD shards inside a node and replicates across nodes, so every
    # all-gather stays on NVLink instead of crossing InfiniBand. It needs the
    # (replication, sharding) mesh; without it FSDP cannot form the two groups.
    if sharding == "HYBRID_SHARD":
        if num_nodes < 2:
            raise RuntimeError(
                "HYBRID_SHARD needs >1 node; on one node it degrades to FULL_SHARD "
                "with extra machinery. Use FULL_SHARD instead.")
        kwargs["device_mesh"] = (num_nodes, gpus_per_node)

    # Overlap the next layer's all-gather with the current layer's compute.
    # Costs one extra layer of gathered parameters in memory, which the 27B has
    # room for once sharding is intra-node.
    if prefetch:
        kwargs["forward_prefetch"] = True
        kwargs["limit_all_gathers"] = False

    return FSDPStrategy(**kwargs)


def _save_stage1_projectors(trainer, model, checkpoint_dir: str) -> None:
    """Write the stage-1 projector and GO-encoder weights that stage 2 warm-starts from.

    Goes through the strategy's state dict rather than each submodule's own. Under
    FSDP the projectors sit in the root flat parameter, so
    `protein_projection.state_dict()` returns this rank's *shard* -- a file of the
    right name holding a truncated tensor. The strategy call is collective and
    unshards to rank 0, and is an ordinary state dict under DDP, so the default
    path is byte-identical to before.

    Args:
        trainer: the Lightning trainer, after `fit`.
        model: the LightningModule that was trained.
        checkpoint_dir: directory to write the .pt files into.

    Raises:
        RuntimeError: when a required prefix matches no parameter, which would
            otherwise write an empty file that stage 2 loads as random weights.
    """
    full = trainer.strategy.lightning_module_state_dict()
    if trainer.global_rank != 0:
        return

    for prefix, filename, required in STAGE1_ARTIFACTS:
        tensors = _mf.submodule_state_dict(full, prefix)
        if not tensors:
            if required:
                raise RuntimeError(
                    f"stage 1 finished but no parameter matched '{prefix}'; "
                    "stage 2 would warm-start from an empty file.")
            continue

        attr = prefix.split(".")[1]
        if getattr(model.model, attr, None) is None:
            continue

        path = os.path.join(checkpoint_dir, filename)
        torch.save(tensors, path)
        print(f"Saved {len(tensors)} tensors to {path}")


def main(args: ArgumentParser):
    """
    Main function to run the Protein-Text fine-tuning process.

    Args:
        args (ArgumentParser): Parsed command-line arguments
    """
    # Set random seed and environment variables
    pl.seed_everything(args.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset = load_cafa5_dataset(
        dataset=args.cafa5_dataset,
        config=args.cafa5_dataset_name,
        cache_dir=args.dataset_cache_dir,
        max_length=args.max_length_protein,
        return_as_chat_template=True,
        structure_dir=args.structure_dir,
        debug=args.debug,
    )
    if train_dataset is None or val_dataset is None:
        raise ValueError(f"{args.cafa5_dataset} needs a `train` and a `validation` split.")

    # Setup directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{args.wandb_project}-{args.text_model_name.split('/')[-1]}"

    # --- optional random context dropout (TRAIN ONLY) -------------------------
    # Validation is deliberately left at full context so val_loss stays comparable
    # across dropout configurations and against the no-dropout baseline.
    dropout_cfg = ContextDropoutConfig(
        ppi_section=args.p_drop_ppi,
        ppi_item=args.p_drop_ppi_item,
        ppi_complete=args.p_ppi_complete,
        features_section=args.p_drop_features,
        features_complete=args.p_features_complete,
        features_item=args.p_drop_features_item,
        gogpt_section=args.p_drop_gogpt,
        gogpt_item=args.p_drop_gogpt_item,
        structure_section=args.p_drop_structure,
    )
    dropout_epoch_value = None
    if dropout_cfg.enabled():
        # Shared across dataloader-worker processes: persistent_workers=True forks
        # them once, so a plain attribute set in this process would never be seen.
        dropout_epoch_value = mp.Value(ctypes.c_int, 0)
        train_dataset = ContextDropoutDataset(
            train_dataset,
            cfg=dropout_cfg,
            base_seed=args.seed,
            epoch_value=dropout_epoch_value,
            question_template_py=args.question_template_py,
        )
        # Fail here, not 19 hours in: proves the ctx_* pieces rebuild the stored
        # question exactly when nothing is dropped.
        train_dataset.verify_no_op()
        print(f"Context dropout ENABLED (train only): {dropout_cfg.describe()}")
        print("Context dropout no-op verification: PASSED (re-render matches cache)")
    else:
        print("Context dropout disabled (all probabilities 0.0)")

    # Initialize model with pre-loaded datasets
    model = ProteinLLMFineTuner(
        args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    # Setup callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
    ]
    if dropout_epoch_value is not None:
        callbacks.append(ContextDropoutEpochCallback(dropout_epoch_value))

    # Only enable model checkpointing for Stage 2, as we only need the projector weights from Stage 1
    if args.training_stage == 2:
        # 1) Keep the single best by lowest val_loss_epoch (saved at validation end)
        best_val_ckpt = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f"{run_name}-best-epoch={{epoch:02d}}-val={{val_loss_epoch:.4f}}",
            monitor="val_loss_epoch",
            mode="min",
            save_top_k=1,
            save_last=False,
            save_on_train_epoch_end=False,
            verbose=True,
        )
        # 2) Keep the most recent, saved every N training steps
        recent_ckpts = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f"{run_name}-recent-epoch={{epoch:02d}}-step={{step:06d}}",
            save_top_k=args.save_top_k,
            monitor="step",
            mode="max",
            save_last=True,
            every_n_train_steps=args.every_n_train_steps,
            save_on_train_epoch_end=False,
            save_weights_only=False,
            verbose=True,
        )
        # 3) Save every epoch from checkpoint_start_epoch onwards
        epoch_ckpt = EpochCheckpointFromN(
            checkpoint_dir=args.checkpoint_dir,
            run_name=run_name,
            start_epoch=args.checkpoint_start_epoch,
        )
        callbacks.extend([recent_ckpts, best_val_ckpt, epoch_ckpt])

    # Setup logger
    is_resuming = args.ckpt_path is not None
    logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        save_dir=args.log_dir,
        name=run_name,
        resume="allow" if is_resuming else None,  # Allow resuming existing run
        log_model=False,
    )

    # Configure Lightning AdvancedProfiler (simple and robust)
    profiler = None
    if args.enable_profiler:
        os.makedirs(args.profiler_dir, exist_ok=True)
        if args.profiler_type == "advanced":
            profiler = AdvancedProfiler(dirpath=args.profiler_dir, filename=args.profiler_filename)
        else:
            # SimpleProfiler reports per-hook wall time (get_train_batch,
            # optimizer_step, backward, ...), which is what identifies where a step
            # actually goes, and is much cheaper than cProfile. Default for that
            # reason -- NOT, as an earlier version of this comment claimed, because
            # AdvancedProfiler was broken: the "TypeError: '>' not supported between
            # instances of 'tuple' and 'float'" seen while adding this came from
            # ContextDropoutConfig.enabled(), which is unrelated and is now fixed.
            profiler = SimpleProfiler(dirpath=args.profiler_dir, filename=args.profiler_filename)

    # Optionally add device stats monitor
    if args.enable_device_stats_monitor:
        callbacks.append(DeviceStatsMonitor(cpu_stats=args.device_stats_cpu))

    # Initialize the PyTorch Lightning Trainer
    # FSDP does its own clipping in `on_before_optimizer_step`; leaving the
    # Trainer's set as well raises MisconfigurationException before step 1.
    # num_gpus is -1 ("all") in every launcher, so the mesh's sharding dimension
    # comes from the visible device count.
    _gpus_per_node = args.num_gpus if args.num_gpus > 0 else torch.cuda.device_count()
    _strategy = _resolve_strategy(args.strategy, model, args.fsdp_sharding,
                                  args.num_nodes, _gpus_per_node,
                                  args.fsdp_prefetch)
    _clip = None if isinstance(_strategy, FSDPStrategy) else GRADIENT_CLIP_VAL

    trainer = pl.Trainer(
        strategy=_strategy,
        max_epochs=args.max_epochs,
        max_steps=(args.max_steps if (args.max_steps is not None and args.max_steps > 0) else -1),
        accelerator="gpu",
        devices=args.num_gpus,
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=_clip,
        val_check_interval=args.val_check_interval,
        num_nodes=args.num_nodes,
        profiler=profiler,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=args.num_sanity_val_steps,
        # Stage 1 keeps only its projector files; Lightning's default checkpoint would also write
        # the whole model (~22 GB at 9B) into the logger directory.
        enable_checkpointing=args.training_stage == 2,
        enable_model_summary=True,
        enable_progress_bar=True,
        sync_batchnorm=False,
    )

    # Start the training process
    trainer.fit(model, ckpt_path=args.ckpt_path)

    # After stage 1, save the projector weights
    if args.training_stage == 1:
        _save_stage1_projectors(trainer, model, args.checkpoint_dir)

    # trainer.test(model, ckpt_path=args.ckpt_path if args.ckpt_path else "best")


if __name__ == "__main__":
    parser = ArgumentParser()

    # Add command-line arguments
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs.")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.expanduser("~/.cache/huggingface/hub"),
        help="Directory for HuggingFace model cache.",
    )
    parser.add_argument("--text_model_name", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--protein_model_name",
        type=str,
        default="esm3_sm_open_v1",
        choices=["esm3_sm_open_v1"],
        help="Protein model name: ESM3 open (esm3_sm_open_v1).",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--max_length_protein", type=int, default=2000)
    parser.add_argument("--max_length_text", type=int, default=16384)
    parser.add_argument("--text_model_finetune", type=str2bool, default=True)
    parser.add_argument(
        "--protein_embedding_layer",
        type=int,
        default=37,
        help="ESM3 layer to extract embeddings from. 37 is the released checkpoint; -1 takes the final output. Eval must be given the SAME value: both are 1536-dim, so a mismatch loads cleanly and silently produces noise."
    )
    parser.add_argument("--go_model_finetune", type=str2bool, default=True)

    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--go_obo_path", type=str, default=None)
    parser.add_argument("--precomputed_embeddings_path", type=str, default=None)
    parser.add_argument("--go_hidden_dim", type=int, default=512)
    parser.add_argument("--go_num_gat_layers", type=int, default=3)
    parser.add_argument("--go_num_heads", type=int, default=8)
    parser.add_argument("--go_num_reduced_embeddings", type=int, default=200)
    parser.add_argument("--go_embedding_dim", type=int, default=2560)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--use_qlora", type=str2bool, default=False, help="Enable QLoRA 4-bit quantization")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16", help="Compute dtype for 4-bit quantization")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", help="Quantization type (nf4 or fp4)")
    parser.add_argument("--bnb_4bit_use_double_quant", type=str2bool, default=True, help="Use double quantization")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument(
        "--fsdp_sharding", type=str, default="FULL_SHARD",
        choices=["FULL_SHARD", "SHARD_GRAD_OP", "HYBRID_SHARD", "NO_SHARD"],
        help="FSDP sharding strategy. HYBRID_SHARD shards within a node and "
             "replicates across nodes, keeping all-gathers on NVLink; it needs "
             "the per-rank shard to fit, which for a 27B is ~6.9 GB at 8 GPUs.",
    )
    parser.add_argument(
        "--fsdp_prefetch", type=str2bool, default=False,
        help="Overlap the next layer's all-gather with the current layer's "
             "compute. Trades one layer of gathered parameters for less stall.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="bioreason-pro-finetune",
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="WandB entity (username or team).",
    )
    parser.add_argument(
        "--cafa5_dataset", type=str, required=True,
        help="SFT table from build_question.py (directory or Hub id). The arm's "
             "prompt is baked into it, so there is no correct default.",
    )
    parser.add_argument(
        "--cafa5_dataset_name",
        type=str,
        default="default",
        help="Config name. build_question.py writes one, `default`.",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=None,
    )
    parser.add_argument("--structure_dir", type=str, default=None)
    parser.add_argument("--return_answer_in_batch", type=str2bool, default=False)
    parser.add_argument("--save_top_k", type=int, default=1)
    parser.add_argument("--val_check_interval", type=float, default=0.2)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    # Arguments for staged training
    parser.add_argument(
        "--training_stage",
        type=int,
        default=2,
        choices=[1, 2],
        help="Training stage: 1 for projector only, 2 for full model.",
    )
    parser.add_argument(
        "--projector_checkpoint_path",
        type=str,
        default=None,
        help="Path to a pretrained projector checkpoint for stage 2.",
    )
    parser.add_argument(
        "--go_projection_checkpoint_path",
        type=str,
        default=None,
        help="Path to a pretrained GO projection checkpoint for stage 2.",
    )
    parser.add_argument(
        "--go_encoder_checkpoint_path",
        type=str,
        default=None,
        help="Path to a pretrained GO encoder checkpoint for stage 2.",
    )
    parser.add_argument("--checkpoint_start_epoch", type=int, default=4, help="Epoch to start saving checkpoints from.")
    parser.add_argument("--every_n_train_steps", type=int, default=None)
    parser.add_argument(
        "--supervise_eos", type=str2bool, default=True,
        help="Supervise the assistant turn's closing <|im_end|>. False reproduces "
             "the pre-fix collator, for runs matched against a baseline trained "
             "before it. Do not set False otherwise -- it re-introduces the bug "
             "that stopped the model ever learning to stop.")
    parser.add_argument("--debug", type=str2bool, default=False)
    dropout_group = parser.add_argument_group("Context Dropout (train only)")
    dropout_group.add_argument(
        "--question_template_py", type=str, default=os.environ.get("QUESTION_TEMPLATE_PY") or None,
        help="path to scripts/data/question_template.py. Required when any "
             "p_drop_* is non-zero: the dropout rebuilds the question through it.")
    dropout_group.add_argument(
        "--p_drop_ppi", type=float, default=0.0,
        help="Probability of dropping the ENTIRE PPI block for a sample.")
    dropout_group.add_argument(
        "--p_drop_ppi_item", type=float, default=0.0,
        help="Probability of dropping each individual PPI partner line.")
    dropout_group.add_argument(
        "--p_ppi_complete", type=float, default=0.0,
        help="Probability of showing the PPI list EXACTLY as the corpus rendered "
             "it. The remainder after --p_drop_ppi and this is the partial "
             "branch, which drops each line at --p_drop_ppi_item. The published "
             "recipe uses 0.0: "
             "partners get none/partial only.")
    dropout_group.add_argument(
        "--p_drop_features", type=float, default=0.0,
        help="Probability of dropping the ENTIRE residue-feature block. Only "
             "reachable on the pre-rendered `question` path.")
    dropout_group.add_argument(
        "--p_features_complete", type=float, default=0.0,
        help="Probability of showing the feature block EXACTLY as the corpus "
             "rendered it, labelled as such. The remainder after "
             "--p_drop_features and this is the partial branch.")
    dropout_group.add_argument(
        "--p_drop_features_item", type=float, default=0.0,
        help="Probability of dropping each individual feature bullet in the "
             "partial branch. At least one bullet is always removed, so the "
             "'a subset, others withheld' label cannot lie; a one-bullet list "
             "is left whole and reported complete instead.")
    dropout_group.add_argument(
        "--p_drop_structure", type=float, default=0.0,
        help="Probability of withholding the structure from a row that has one.")
    dropout_group.add_argument(
        "--p_drop_gogpt", type=float, default=0.0,
        help="Probability of dropping the ENTIRE GO-GPT speculation block.")
    dropout_group.add_argument(
        "--p_drop_gogpt_item", type=float, default=0.0,
        help="Probability of dropping each GO-GPT aspect line (MF / BP / CC).")

    parser.add_argument("--run_name", type=str, default=None, help="Custom name for the WandB run.")
    parser.add_argument(
        "--protein_projection_layernorm",
        type=str2bool,
        default=False,
        help="Append a LayerNorm to the protein projection. False is every existing checkpoint.",
    )
    parser.add_argument(
        "--protein_projection_norm",
        type=str,
        default="",
        help="Normalisation on the projection tail. Empty derives it from --protein_projection_layernorm, so every existing run is unchanged. blocknorm keeps per-residue magnitudes and fixes only the scale; lnscale does the opposite. Must match how the checkpoint was trained.",
    )
    parser.add_argument(
        "--protein_projection_norm_target",
        type=float,
        default=0.0,
        help="Initial output RMS for the variants that take one; 0 uses the variant's default (1.0 for blocknorm, 7.4 for lnscale).",
    )
    # Profiling controls (AdvancedProfiler only)
    parser.add_argument(
        "--gradient_checkpointing", type=str2bool, default=True,
        help="Recompute activations in the backward instead of storing them. "
             "ON by default because OFF runs out of memory on real data. "
             "At a FIXED seq 2024 on one H100 (Qwen3.5-4B + LoRA) OFF is a genuine "
             "1.55x -- 369 ms/step at 12.2 GB vs 239 ms/step at 25.5 GB -- but real "
             "batches run to --max_length_text (16384), and an 8-GPU run with it OFF "
             "OOM'd at the optimizer step. Only turn it off if sequence length is "
             "capped well below the maximum.")
    parser.add_argument("--enable_profiler", type=str2bool, default=False)
    parser.add_argument("--profiler_type", type=str, default="simple",
                        choices=["simple", "advanced"])
    parser.add_argument("--profiler_dir", type=str, default="profiles")
    parser.add_argument("--profiler_filename", type=str, default="profile")
    # Short-run controls for profiling
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="If > 0, overrides to run a fixed number of steps",
    )
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--num_sanity_val_steps", type=int, default=2)
    # Device stats monitor controls
    parser.add_argument("--enable_device_stats_monitor", type=str2bool, default=False)
    parser.add_argument("--device_stats_cpu", type=str2bool, default=False)
    args = parser.parse_args()

    main(args)
