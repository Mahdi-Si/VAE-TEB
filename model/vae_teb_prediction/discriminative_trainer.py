"""PyTorch Lightning module and trainer for discriminative VAE-TEB fine-tuning.

This module provides:

* ``PlDiscriminativeSeqVae`` — Lightning wrapper that routes training /
  validation steps through ``DiscriminativeSeqVae.compute_loss()`` and
  builds a differential-LR optimizer (low LR for encoders, high LR for the
  classifier head).
* ``GraphModelDiscriminativeTrainer`` — Experiment scaffold that instantiates
  ``SeqVae``, loads a pretrained checkpoint, wraps it in
  ``DiscriminativeSeqVae``, applies phase freezing, and runs the Lightning
  training loop.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import lightning as pl
import numpy as np
import time
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.profilers import SimpleProfiler
from loguru import logger
from torch.optim import Optimizer

from hdf5_dataset.hdf5_dataset import create_optimized_dataloader
from model.vae_teb_prediction.discriminative_finetune_model import (
    DiscriminativeSeqVae,
)
from model.vae_teb_prediction.vae_teb_model_prediction import SeqVae
from train.callbacks import (
    HyperparameterLoggingCallback,
    LossPlotCallback,
    MetricsLoggingCallback,
)
from train.graph_model_base import GraphModelBase
from train.graph_models_utils import load_checkpoint_strict
from train.pl_model_base import LightningModelBase, MetricDict


class PlDiscriminativeSeqVae(LightningModelBase):
    """Lightning wrapper for discriminative VAE-TEB fine-tuning.

    Extends ``LightningModelBase`` with:

    * ``compute_loss_and_metrics`` that delegates to
      ``DiscriminativeSeqVae.compute_loss()`` and extracts centroid distances
      for monitoring.
    * ``build_optimizer`` that creates two parameter groups with differential
      learning rates (``lr_encoders`` for VAE encoders, ``lr`` for the
      classifier head).
    """

    prog_bar_metrics: Tuple[str, ...] = ("total_loss", "cls_accuracy")

    def __init__(
        self,
        base_model: DiscriminativeSeqVae,
        *,
        lr: float = 1e-3,
        lr_encoders: float = 1e-5,
        lr_milestones: Optional[List[int]] = None,
        weight_decay: float = 1e-4,
        training_phase: int = 1,
        module_name: Optional[str] = None,
    ) -> None:
        """Initialize the Lightning wrapper.

        Args:
            base_model: A ``DiscriminativeSeqVae`` instance with phase
                freezing already applied.
            lr: Learning rate for the classifier head.
            lr_encoders: Learning rate for VAE encoder parameters (used in
                phase 2 only).
            lr_milestones: Epoch milestones for LR decay.
            weight_decay: AdamW weight decay.
            training_phase: Current training phase (1 or 2), stored in hparams
                for logging.
            module_name: Friendly name for log messages.
        """
        super().__init__(
            base_model,
            lr=lr,
            lr_milestones=lr_milestones,
            weight_decay=weight_decay,
            module_name=module_name or "PlDiscriminativeSeqVae",
        )
        self.save_hyperparameters(ignore=["base_model"])
        self.hparams["lr_encoders"] = lr_encoders
        self.hparams["training_phase"] = training_phase

    def compute_loss_and_metrics(
        self,
        batch,
        batch_idx: int,
        stage: str,
    ) -> Tuple[torch.Tensor, MetricDict]:
        """Compute discriminative loss and collect monitoring metrics.

        Args:
            batch: Dataloader batch with attributes ``fhr_st``, ``fhr_ph``,
                ``fhr_up_ph``, ``fhr``, and ``target``.
            batch_idx: Index of the current batch.
            stage: One of ``'train'``, ``'val'``, ``'test'``.

        Returns:
            Tuple of (total_loss, metrics_dict) where metrics_dict contains
            all individual loss components plus classification accuracy and
            centroid distances.
        """
        y_st = batch.fhr_st       # (B, T, 43)
        y_ph = batch.fhr_ph       # (B, T, 44)
        x_ph = batch.fhr_up_ph    # (B, T, 137)
        y_raw = batch.fhr         # (B, 4800)
        target_seq = batch.target  # (B, T) values in {1, 2, 3}

        # Aggregate to single label per sample (take max across sequence)
        labels = target_seq.max(dim=1)[0]  # (B,) values in {1, 2, 3}

        # Forward pass
        forward_outputs = self.model(y_st=y_st, y_ph=y_ph, x_ph=x_ph)

        # Compute combined loss
        kld_beta = float(self.hparams.get("kld_beta", 0.05))
        loss_dict = self._orig_model.compute_loss(
            forward_outputs=forward_outputs,
            y_st=y_st,
            y_ph=y_ph,
            y_raw=y_raw,
            labels=labels,
            beta=kld_beta,
        )

        total_loss = loss_dict["total_loss"]

        # Build metrics
        metrics: MetricDict = {
            "total_loss": total_loss,
            "nll_loss": loss_dict["nll_loss"],
            "kld_loss": loss_dict["kld_loss"],
            "center_loss": loss_dict["center_loss"],
            "cls_loss": loss_dict["cls_loss"],
            "cls_accuracy": loss_dict["cls_accuracy"],
            "kld_beta": loss_dict["kld_beta"],
        }

        # Log centroid distances (only on validation to avoid overhead)
        if stage == "val":
            center_dists = self._orig_model.get_center_distances()
            for i in range(center_dists.shape[0]):
                for j in range(i + 1, center_dists.shape[1]):
                    metrics[f"center_dist_{i}_{j}"] = center_dists[i, j]

        return total_loss, metrics

    def build_optimizer(
        self,
        trainable_params: Iterable[torch.nn.Parameter],
    ) -> Optimizer:
        """Build AdamW optimizer with differential learning rates.

        Creates two parameter groups:

        1. **Encoder parameters** — VAE source, target, and conditional
           encoders at ``lr_encoders`` (very low, e.g. 1e-5).
        2. **Classifier head parameters** — at ``lr`` (higher, e.g. 1e-3).

        If phase 1 (all VAE frozen), the encoder group will be empty and only
        the head group receives gradients.

        Args:
            trainable_params: All trainable parameters (unused; we build
                groups from the underlying model directly).

        Returns:
            Configured AdamW optimizer.
        """
        lr_head = float(self.hparams.get("lr", 1e-3))
        lr_enc = float(self.hparams.get("lr_encoders", 1e-5))
        weight_decay = float(self.hparams.get("weight_decay", 1e-4))

        encoder_params = self._orig_model.get_encoder_params()
        head_params = self._orig_model.get_head_params()

        param_groups = []
        if encoder_params:
            param_groups.append({
                "params": encoder_params,
                "lr": lr_enc,
                "name": "encoders",
            })
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": lr_head,
                "name": "classifier_head",
            })

        if not param_groups:
            logger.warning("No trainable parameters found!")
            param_groups = [{"params": list(trainable_params), "lr": lr_head}]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
            eps=1e-8,
            betas=(0.9, 0.95),
        )

        for group in optimizer.param_groups:
            n_params = sum(p.numel() for p in group["params"])
            logger.info(
                f"Optimizer group '{group.get('name', '?')}': "
                f"{n_params:,} params, lr={group['lr']}"
            )

        return optimizer


class GraphModelDiscriminativeTrainer(GraphModelBase):
    """Experiment scaffold for discriminative VAE-TEB fine-tuning.

    Follows the same pattern as ``GraphModelVaeTebSmallTrainer`` and
    ``GraphModelClassifierTrainer``:

    1. ``create_model()`` instantiates SeqVae, loads pretrained weights, wraps
       in ``DiscriminativeSeqVae``, applies phase freezing, then wraps in the
       Lightning module.
    2. ``train_model()`` configures callbacks and runs the Lightning trainer.
    """

    def __init__(self, config_file_path: str | None = None) -> None:
        """Initialize the trainer.

        Args:
            config_file_path: Path to the discriminative fine-tuning YAML
                config file.
        """
        super().__init__(config_file_path)

    def create_model(self) -> None:
        """Create the DiscriminativeSeqVae and wrap in Lightning.

        Reads configuration from ``model_config.discriminative`` and
        ``model_config.core_model_checkpoint`` to build the full training
        pipeline.
        """
        model_config = self.config.get("model_config", {})
        disc_config = model_config.get("discriminative", {})
        vae_checkpoint = model_config.get("core_model_checkpoint")

        if vae_checkpoint is None:
            raise ValueError(
                "model_config.core_model_checkpoint must be provided"
            )

        # 1. Instantiate and load pretrained SeqVae
        vae_model = SeqVae()
        load_checkpoint_strict(model=vae_model, checkpoint=vae_checkpoint)
        logger.info(f"Pretrained VAE loaded from: {vae_checkpoint}")

        # 2. Wrap in DiscriminativeSeqVae
        self.pytorch_model = DiscriminativeSeqVae(
            vae_model=vae_model,
            num_classes=disc_config.get("num_classes", 3),
            classifier_hidden_dim=disc_config.get("classifier_hidden_dim", 32),
            center_ema_decay=disc_config.get("center_ema_decay", 0.99),
            alpha_recon=disc_config.get("alpha_recon", 1.0),
            alpha_kld=disc_config.get("alpha_kld", 1.0),
            alpha_center=disc_config.get("alpha_center", 0.1),
            alpha_cls=disc_config.get("alpha_cls", 0.5),
        )

        # 3. Apply phase freezing
        training_phase = disc_config.get("training_phase", 1)
        self.pytorch_model.freeze_for_phase(training_phase)

        # 4. Wrap in Lightning module
        lr_encoders = disc_config.get("lr_encoders", 1e-5)
        kld_beta = model_config.get("VAE_model", {}).get("kld_beta", 0.05)

        trainer_hparams = {
            "lr": self.lr,
            "lr_milestones": self.lr_milestones,
            "lr_encoders": lr_encoders,
            "kld_beta": kld_beta,
            "training_phase": training_phase,
        }

        self.pl_model = PlDiscriminativeSeqVae(
            self.pytorch_model,
            lr=self.lr,
            lr_encoders=lr_encoders,
            lr_milestones=self.lr_milestones,
            training_phase=training_phase,
        )

        self.apply_config_hyperparameters(trainer_hparams, self.pl_model)

    def train_model(
        self,
        train_dataloader,
        validation_dataloader,
    ) -> pl.Trainer:
        """Run the discriminative fine-tuning training loop.

        Args:
            train_dataloader: Training dataloader.
            validation_dataloader: Validation dataloader.

        Returns:
            The fitted ``pl.Trainer`` instance.
        """
        callbacks_cfg = self.config.get("advanced_config", {}).get("callbacks", {})

        # Callbacks
        metrics_callback = MetricsLoggingCallback()
        loss_plot_callback = LossPlotCallback(
            output_dir=self.train_results_dir,
            plot_frequency=self.config["general_config"].get("plot_frequency", 1),
            mlflow_logger=self.mlflow_logger,
        )
        hyperparam_callback = HyperparameterLoggingCallback(
            output_dir=self.train_results_dir,
            plot_frequency=10,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_checkpoint_dir,
            monitor="val/total_loss",
            filename="disc-finetune-epoch={epoch:02d}-loss={val/total_loss:.4f}",
            save_top_k=callbacks_cfg.get("model_checkpoint", {}).get("save_top_k", 3),
            mode="min",
        )
        early_stopping_callback = EarlyStopping(
            monitor="val/total_loss",
            patience=callbacks_cfg.get("early_stopping", {}).get("patience", 30),
            mode="min",
            verbose=True,
        )

        callback_list = [
            metrics_callback,
            loss_plot_callback,
            hyperparam_callback,
            checkpoint_callback,
            early_stopping_callback,
        ]

        # Trainer config
        trainer_cfg = self.config.get("advanced_config", {}).get("trainer", {})
        precision = trainer_cfg.get("precision", "32-true")
        gradient_clip_val = trainer_cfg.get("gradient_clip_val")
        gradient_clip_algorithm = trainer_cfg.get("gradient_clip_algorithm", "norm")
        logger_reference = self.lightning_loggers if self.lightning_loggers else True

        trainer_kwargs = {
            "max_epochs": self.epochs_num,
            "callbacks": callback_list,
            "default_root_dir": self.train_results_dir,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "precision": precision,
            "deterministic": trainer_cfg.get("deterministic", False),
            "benchmark": trainer_cfg.get("benchmark", True),
            "gradient_clip_val": gradient_clip_val,
            "gradient_clip_algorithm": gradient_clip_algorithm,
            "enable_checkpointing": True,
            "log_every_n_steps": 1,
            "num_sanity_val_steps": 0,
            "use_distributed_sampler": True,
            "sync_batchnorm": len(self.cuda_devices) > 1,
            "enable_progress_bar": True,
            "profiler": SimpleProfiler(dirpath=self.train_results_dir),
            "logger": logger_reference,
        }

        if torch.cuda.is_available():
            trainer_kwargs.update({
                "accelerator": "gpu",
                "devices": self.cuda_devices,
                "strategy": "ddp" if len(self.cuda_devices) > 1 else "auto",
            })
        else:
            trainer_kwargs.update({"accelerator": "cpu", "devices": 1})

        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(self.pl_model, train_dataloader, validation_dataloader)
        return trainer


def main() -> None:
    """Entry point for discriminative fine-tuning."""
    np.random.seed(42)
    torch.manual_seed(42)

    start_time = time.time()

    with open(r"config_discriminative.yaml") as f:
        config = yaml.safe_load(f)

    # Setup dataset
    dataset_config = config.get("dataset_config")
    dataloader_config = dataset_config.get("dataloader_config")
    dataset_kwargs = dataloader_config.get("dataset_kwargs")
    normalized_fields = dataloader_config.get("normalize_fields")
    stat_path = dataset_config.get("stat_path")

    if stat_path is None:
        raise ValueError("stat_path must be provided")

    logger.info(f"Normalized fields: {normalized_fields}")

    train_dataloader = create_optimized_dataloader(
        hdf5_files=dataset_config.get("vae_train_datasets", []),
        batch_size=config["general_config"]["batch_size"]["train"],
        num_workers=dataloader_config.get("num_workers", 4),
        shuffle=True,
        stats_path=stat_path,
        normalize_fields=normalized_fields,
        prefetch_factor=dataloader_config.get("prefetch_factor", 2),
        pin_memory=True,
        rank=0,
        world_size=1,
        **dataset_kwargs,
    )

    validation_dataloader = create_optimized_dataloader(
        hdf5_files=dataset_config.get("vae_test_datasets", []),
        batch_size=config["general_config"]["batch_size"]["test"],
        num_workers=dataloader_config.get("num_workers", 4),
        shuffle=False,
        stats_path=stat_path,
        normalize_fields=normalized_fields,
        prefetch_factor=dataloader_config.get("prefetch_factor", 2),
        rank=0,
        world_size=1,
        **dataset_kwargs,
    )

    graph_model = GraphModelDiscriminativeTrainer(
        config_file_path=r"config_discriminative.yaml"
    )
    graph_model.setup_config()
    graph_model.create_model()
    trainer = graph_model.train_model(train_dataloader, validation_dataloader)

    end_time = time.time()
    logger.info(f"Training completed in {(end_time - start_time) / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
