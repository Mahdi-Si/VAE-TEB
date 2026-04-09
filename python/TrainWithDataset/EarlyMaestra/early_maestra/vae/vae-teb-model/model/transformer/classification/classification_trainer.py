"""Training pipeline for the transformer-based GRU classifier.

Provides the PyTorch Lightning wrapper (``PlClassifier``) and the
``GraphModelBase``-derived trainer (``GraphModelClassificationTrainer``)
that orchestrate single-fold training of the :class:`TimeAwareGRUClassifier`.

Typical workflow::

    trainer = GraphModelClassificationTrainer(
        config_file_path="config_classification.yaml",
    )
    trainer.setup_config()
    trainer.create_model()
    lightning_trainer = trainer.train_model(train_loader, val_loader)

Or use the convenience :func:`train_fold` to train a single fold
end-to-end from a config dict.
"""

from __future__ import annotations

import gc
import json
import os
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.profilers import SimpleProfiler
from loguru import logger

from train.graph_model_base import GraphModelBase
from train.pl_model_base import LightningModelBase
from train.callbacks import (
    LossPlotCallback,
    HyperparameterLoggingCallback,
    MetricsLoggingCallback,
)
from train.graph_models_utils import load_checkpoint_strict
from model.transformer.classification.classification_model import (
    TimeAwareGRUClassifier,
)


# ---------------------------------------------------------------------------
#  Lightning Module
# ---------------------------------------------------------------------------


class PlClassifier(LightningModelBase):
    """PyTorch Lightning wrapper for :class:`TimeAwareGRUClassifier`.

    Inherits from :class:`LightningModelBase` and overrides
    :meth:`compute_loss_and_metrics` to handle GUID-sequence batches.

    Supports phased unfreezing: when ``freeze_mode="phased"`` and the
    current epoch equals ``unfreeze_after_epoch``, the transformer is
    unfrozen and added to the optimizer with a separate learning rate.

    Important:
        Overrides ``__init__`` to bypass ``torch.compile`` because
        ``TimeAwareGRUClassifier.forward()`` contains dynamic control
        flow (custom GRU loop, chunked encoding, conditional branches)
        that is incompatible with ``torch.compile`` graph tracing.

    Args:
        base_model: A :class:`TimeAwareGRUClassifier` instance.
        lr: Learning rate for the classifier head.
        lr_milestones: Epochs for learning-rate decay.
        weight_decay: AdamW weight-decay coefficient.
        freeze_mode: ``"frozen"`` | ``"trainable"`` | ``"phased"``.
        unfreeze_after_epoch: Epoch at which to unfreeze the
            transformer (only used when ``freeze_mode="phased"``).
        transformer_lr: Separate learning rate for transformer params
            after unfreezing.
    """

    prog_bar_metrics = ("loss", "accuracy")

    def __init__(
        self,
        base_model: nn.Module,
        *,
        lr: float = 1e-3,
        lr_milestones: Optional[Sequence[int]] = None,
        weight_decay: float = 1e-4,
        freeze_mode: str = "frozen",
        unfreeze_after_epoch: int = 50,
        transformer_lr: float = 1e-5,
    ) -> None:
        # Bypass torch.compile entirely — same reasoning as
        # PlTemporalClassifier in the guid_classifier pipeline.
        pl.LightningModule.__init__(self)
        self.save_hyperparameters(ignore=["base_model"])
        self._orig_model = base_model
        self._wrapper_name = self.__class__.__name__
        self.model = base_model  # Eager mode — no torch.compile

        self._freeze_mode = freeze_mode
        self._unfreeze_after_epoch = unfreeze_after_epoch
        self._transformer_lr = transformer_lr
        self._unfrozen = freeze_mode == "trainable"

    # ------------------------------------------------------------------ #
    #  Phased Unfreezing                                                   #
    # ------------------------------------------------------------------ #

    def on_train_epoch_start(self) -> None:
        """Unfreeze the transformer at the configured epoch."""
        if (
            self._freeze_mode == "phased"
            and not self._unfrozen
            and self.current_epoch >= self._unfreeze_after_epoch
        ):
            self.model.unfreeze_transformer()
            self._unfrozen = True

            # Add transformer params to the existing optimizer.
            optimizer = self.optimizers()
            if hasattr(optimizer, "param_groups"):
                transformer_params = [
                    p for p in self.model.transformer.parameters()
                    if p.requires_grad
                ]
                if transformer_params:
                    optimizer.add_param_group({
                        "params": transformer_params,
                        "lr": self._transformer_lr,
                    })
                    logger.info(
                        "Epoch {}: unfroze transformer, added {} params "
                        "with lr={}",
                        self.current_epoch,
                        sum(p.numel() for p in transformer_params),
                        self._transformer_lr,
                    )

    # ------------------------------------------------------------------ #
    #  Optimizer                                                           #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self):
        """Configure AdamW with optional discriminative learning rates.

        Returns:
            Dict with ``optimizer`` and ``lr_scheduler`` keys.
        """
        lr = self.hparams.get("lr", 1e-3)
        weight_decay = self.hparams.get("weight_decay", 1e-4)
        milestones = self.hparams.get("lr_milestones") or []

        # Separate parameter groups.
        param_groups = []

        # Classifier head parameters (always trainable).
        classifier_params = []
        transformer_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("transformer."):
                transformer_params.append(param)
            else:
                classifier_params.append(param)

        if classifier_params:
            param_groups.append({
                "params": classifier_params,
                "lr": lr,
            })

        # Transformer params (only if already unfrozen at init).
        if transformer_params:
            param_groups.append({
                "params": transformer_params,
                "lr": self._transformer_lr,
            })

        if not param_groups:
            param_groups = [{"params": [torch.zeros(1, requires_grad=True)]}]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(milestones),
            gamma=0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    # ------------------------------------------------------------------ #
    #  Training / Validation Steps                                         #
    # ------------------------------------------------------------------ #

    def training_step(self, batch, batch_idx):
        """Override to log training metrics at epoch-level only.

        The per-step values are noisy due to variable batch composition
        from the bucket sampler.  Logging with ``on_step=False`` ensures
        ``callback_metrics["train/loss"]`` contains the epoch average.
        """
        loss, metrics = self.compute_loss_and_metrics(
            batch, batch_idx, stage="train"
        )
        self._log_metrics(metrics, stage="train", on_step=False)
        return loss

    def compute_loss_and_metrics(
        self,
        batch: Dict,
        batch_idx: int,
        stage: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run forward pass and compute classification loss.

        Args:
            batch: Dict from ``sequence_collate_fn``.
            batch_idx: Index of the current batch.
            stage: One of ``'train'``, ``'val'``, ``'test'``.

        Returns:
            Tuple of ``(loss, metrics_dict)``.
        """
        outputs = self.model(batch)
        loss_dict = self._orig_model.compute_loss(outputs, batch)

        loss = loss_dict["loss"]
        metrics = {
            "loss": loss,
            "accuracy": loss_dict["accuracy"],
            "class_0_acc": loss_dict["class_0_acc"],
            "class_1_acc": loss_dict["class_1_acc"],
        }
        return loss, metrics


# ---------------------------------------------------------------------------
#  Class Balance Estimation
# ---------------------------------------------------------------------------


def estimate_class_balance(
    dataset,
    loss_type: str = "bce",
) -> Dict:
    """Estimate class balance weights from GUID-level labels.

    Args:
        dataset: An already-instantiated ``SignalSequenceDataset`` or
            ``PrecomputedEmbeddingDataset``.
        loss_type: ``"bce"`` or ``"ce"``.

    Returns:
        Dict with:
            - ``pos_weight``: float (for BCE) or ``None``
            - ``class_weights``: list of floats (for CE) or ``None``
    """
    weights, counts = dataset.estimate_class_weights(num_classes=2)

    if counts.sum() == 0:
        logger.warning(
            "No GUIDs found for class weight estimation, "
            "returning uniform weights"
        )
        return {"pos_weight": None, "class_weights": None}

    logger.info(
        "Class balance — GUID counts (healthy={}, unhealthy={}), "
        "weights={}",
        counts[0].item(), counts[1].item(), weights.tolist(),
    )

    if loss_type == "bce":
        n_neg = counts[0].item()
        n_pos = counts[1].item()
        pos_weight = n_neg / max(n_pos, 1.0)
        logger.info("BCE pos_weight: {:.4f}", pos_weight)
        return {"pos_weight": pos_weight, "class_weights": None}
    else:
        class_weights = weights.tolist()
        logger.info("CE class_weights: {}", class_weights)
        return {"pos_weight": None, "class_weights": class_weights}


# ---------------------------------------------------------------------------
#  Best Checkpoint Metrics
# ---------------------------------------------------------------------------


def resolve_best_checkpoint_metrics(
    trainer: pl.Trainer,
    pl_model: pl.LightningModule,
    validation_dataloader,
    checkpoint_callback: ModelCheckpoint,
) -> Dict[str, float]:
    """Resolve validation metrics for the actual best checkpoint.

    Lightning's ``trainer.callback_metrics`` reflects the most recent
    epoch, not necessarily the checkpoint selected by
    ``ModelCheckpoint``.  This helper validates the best checkpoint to
    recover accurate metrics.

    Args:
        trainer: Fitted Lightning trainer.
        pl_model: Lightning module used during training.
        validation_dataloader: Validation dataloader for the fold.
        checkpoint_callback: The ModelCheckpoint callback.

    Returns:
        Dict with ``best_checkpoint_path``, ``best_val_loss``,
        ``best_val_accuracy``.
    """
    def _to_float(value, default: float) -> float:
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    callback_metrics = getattr(trainer, "callback_metrics", {}) or {}
    best_model_path = (
        getattr(checkpoint_callback, "best_model_path", "") or ""
    )
    best_model_score = getattr(checkpoint_callback, "best_model_score", None)

    if best_model_score is not None:
        best_val_loss = _to_float(best_model_score, float("inf"))
    else:
        best_val_loss = _to_float(
            callback_metrics.get("val/loss"), float("inf")
        )

    best_val_accuracy = _to_float(
        callback_metrics.get("val/accuracy"), 0.0
    )

    if best_model_path:
        try:
            best_results = trainer.validate(
                model=pl_model,
                dataloaders=validation_dataloader,
                ckpt_path=best_model_path,
                verbose=False,
            )
            if best_results:
                best_metrics = best_results[0]
                if "val/loss" in best_metrics:
                    best_val_loss = _to_float(
                        best_metrics["val/loss"], best_val_loss
                    )
                if "val/accuracy" in best_metrics:
                    best_val_accuracy = _to_float(
                        best_metrics["val/accuracy"], best_val_accuracy
                    )
        except Exception as exc:
            logger.warning(
                "Failed to validate best checkpoint {}: {}",
                best_model_path, exc,
            )

    return {
        "best_checkpoint_path": best_model_path,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
    }


# ---------------------------------------------------------------------------
#  GraphModel Trainer
# ---------------------------------------------------------------------------


class GraphModelClassificationTrainer(GraphModelBase):
    """Training pipeline for the transformer GRU classifier.

    Reads configuration from ``config_classification.yaml``, creates a
    :class:`TimeAwareGRUClassifier` with a frozen (or fine-tunable)
    transformer, wraps it in :class:`PlClassifier`, and runs training
    via PyTorch Lightning.

    Args:
        config_file_path: Path to the YAML configuration file.
    """

    def __init__(self, config_file_path: Optional[str] = None) -> None:
        if config_file_path is None:
            config_file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "config_classification.yaml",
            )
        super().__init__(config_file_path)

    def create_model(
        self,
        pos_weight: Optional[float] = None,
        class_weights: Optional[List[float]] = None,
    ) -> None:
        """Instantiate :class:`TimeAwareGRUClassifier` and wrap in Lightning.

        Steps:
            1. Load pre-trained transformer checkpoint.
            2. Build :class:`TimeAwareGRUClassifier` from config.
            3. Log trainable vs frozen parameter counts.
            4. Wrap in :class:`PlClassifier`.

        Args:
            pos_weight: Positive class weight for BCE loss.
            class_weights: Per-class weights for CE loss.
        """
        model_cfg = self.config.get("model_config", {})

        # ----- 1. Load transformer ---------------------------------------- #
        transformer_checkpoint = model_cfg.get("transformer_checkpoint")
        precompute_mode = model_cfg.get("precompute_embeddings", False)

        transformer_model = None
        if not precompute_mode:
            if transformer_checkpoint is None:
                raise ValueError(
                    "transformer_checkpoint must be provided in model_config "
                    "when precompute_embeddings is False"
                )

            from model.transformer.model.model import (
                CausalMultimodalTransformer,
            )
            from model.transformer.tr_testing.base import (
                TransformerTestRunner,
            )

            logger.info(
                "Loading transformer from: {}", transformer_checkpoint
            )
            ckpt = torch.load(
                transformer_checkpoint, map_location="cpu",
                weights_only=False,
            )
            tr_config = TransformerTestRunner._extract_config(ckpt)
            transformer_model = CausalMultimodalTransformer(tr_config)
            loaded = load_checkpoint_strict(transformer_model, ckpt)
            if loaded is None:
                raise RuntimeError(
                    "Strict transformer checkpoint loading failed. "
                    f"Checkpoint: {transformer_checkpoint}"
                )
            logger.info("Transformer loaded successfully.")

        # ----- 2. Build classifier model ---------------------------------- #
        emb_cfg = model_cfg.get("segment_embedding", {})
        time_cfg = model_cfg.get("time_features", {})
        cls_cfg = model_cfg.get("classifier", {})
        loss_cfg = model_cfg.get("loss", {})
        freeze_cfg = model_cfg.get("freeze_strategy", {})

        self.pytorch_model = TimeAwareGRUClassifier(
            transformer_model=transformer_model,
            d_embedding=emb_cfg.get("d_embedding", 416),
            time_embed_dim=time_cfg.get("embed_dim", 32),
            input_proj_dim=cls_cfg.get("input_proj_dim", 256),
            gru_hidden_dim=cls_cfg.get("gru_hidden_dim", 256),
            dropout=cls_cfg.get("dropout", 0.1),
            loss_type=loss_cfg.get("type", "bce"),
            pos_weight=pos_weight,
            class_weights=class_weights,
            label_smoothing=loss_cfg.get("label_smoothing", 0.0),
            transformer_chunk_size=model_cfg.get(
                "transformer_chunk_size", 16
            ),
            freeze_strategy=freeze_cfg.get("mode", "frozen"),
            pooling=emb_cfg.get("pooling", "mean"),
            anchor_step=emb_cfg.get("anchor_step", 5),
            nominal_gap_minutes=time_cfg.get("nominal_gap_minutes", 20.0),
            gap_threshold_minutes=time_cfg.get(
                "gap_threshold_minutes", 22.0
            ),
        )

        # ----- 3. Log parameter counts ----------------------------------- #
        total_params = sum(
            p.numel() for p in self.pytorch_model.parameters()
        )
        trainable_params = sum(
            p.numel()
            for p in self.pytorch_model.parameters()
            if p.requires_grad
        )
        frozen_params = total_params - trainable_params
        logger.info("Total parameters: {:,}", total_params)
        logger.info("Trainable parameters: {:,}", trainable_params)
        logger.info("Frozen parameters: {:,}", frozen_params)

        # ----- 4. Wrap in Lightning --------------------------------------- #
        self.pl_model = PlClassifier(
            self.pytorch_model,
            lr=self.lr,
            lr_milestones=self.lr_milestones,
            freeze_mode=freeze_cfg.get("mode", "frozen"),
            unfreeze_after_epoch=freeze_cfg.get("unfreeze_after_epoch", 50),
            transformer_lr=freeze_cfg.get("transformer_lr", 1e-5),
        )

        trainer_hparams = {
            "lr": self.lr,
            "lr_milestones": self.lr_milestones,
        }
        self.apply_config_hyperparameters(trainer_hparams, self.pl_model)

    def train_model(
        self,
        train_dataloader,
        validation_dataloader,
    ) -> pl.Trainer:
        """Train the classifier using PyTorch Lightning.

        Args:
            train_dataloader: Training DataLoader.
            validation_dataloader: Validation DataLoader.

        Returns:
            The Lightning Trainer instance after fitting.
        """
        callbacks_cfg = (
            self.config.get("advanced_config", {}).get("callbacks", {})
        )

        # --- Callbacks ---------------------------------------------------- #
        self.metrics_callback = MetricsLoggingCallback()
        self.loss_plot_callback = LossPlotCallback(
            output_dir=self.train_results_dir,
            plot_frequency=self.config["general_config"].get(
                "plot_frequency", 10
            ),
            mlflow_logger=self.mlflow_logger,
        )
        self.hyperparam_callback = HyperparameterLoggingCallback(
            output_dir=self.train_results_dir,
            plot_frequency=10,
        )

        ckpt_cfg = callbacks_cfg.get("model_checkpoint", {})
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_checkpoint_dir,
            monitor=ckpt_cfg.get("monitor", "val/loss"),
            filename="cls-model-{epoch:02d}",
            save_top_k=ckpt_cfg.get("save_top_k", 3),
            mode="min",
            auto_insert_metric_name=False,
        )

        callback_list = [
            self.metrics_callback,
            self.loss_plot_callback,
            self.hyperparam_callback,
            self.checkpoint_callback,
        ]

        es_cfg = callbacks_cfg.get("early_stopping", {})
        if es_cfg.get("enabled", True):
            self.early_stopping_callback = EarlyStopping(
                monitor=es_cfg.get("monitor", "val/loss"),
                patience=es_cfg.get("patience", 30),
                mode="min",
                verbose=True,
            )
            callback_list.append(self.early_stopping_callback)

        # --- Trainer config ----------------------------------------------- #
        trainer_cfg = (
            self.config.get("advanced_config", {}).get("trainer", {})
        )
        precision = trainer_cfg.get("precision", "32-true")
        gradient_clip_val = trainer_cfg.get("gradient_clip_val")
        gradient_clip_algorithm = trainer_cfg.get(
            "gradient_clip_algorithm", "norm"
        )

        logger_reference = (
            self.lightning_loggers if self.lightning_loggers else True
        )

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
            "use_distributed_sampler": False,
            "sync_batchnorm": False,
            "enable_progress_bar": True,
            "profiler": SimpleProfiler(dirpath=self.train_results_dir),
            "logger": logger_reference,
        }

        if torch.cuda.is_available():
            trainer_kwargs.update({
                "accelerator": "gpu",
                "devices": self.cuda_devices,
                "strategy": (
                    "ddp" if len(self.cuda_devices) > 1 else "auto"
                ),
            })
        else:
            trainer_kwargs.update({"accelerator": "cpu", "devices": 1})

        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(self.pl_model, train_dataloader, validation_dataloader)
        self._trainer = trainer

        return trainer


# ---------------------------------------------------------------------------
#  Single-Fold Training Function
# ---------------------------------------------------------------------------


def train_fold(
    fold_id: int,
    config: Dict,
    gpu_id: int = 0,
) -> Tuple[str, GraphModelClassificationTrainer]:
    """Train the classifier on a single fold.

    This is the primary entry point for single-fold training.  It
    handles dataset loading, class weight estimation, config writing,
    model creation, and training.

    Args:
        fold_id: Fold number (1-10).
        config: Full configuration dict from
            ``config_classification.yaml``.
        gpu_id: GPU device to use.

    Returns:
        Tuple of ``(checkpoint_dir, trainer_instance)``.
    """
    import random
    import numpy as np

    logger.info("Starting classification fold {} on GPU {}", fold_id, gpu_id)

    # Seed for reproducibility.
    seed = 42 + fold_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Dataset paths ---------------------------------------------------- #
    from model.vae_teb_prediction.kfold_classifier_trainer import (
        get_fold_datasets,
    )

    dataset_cfg = config.get("dataset_config", {})
    kfold_base_path = dataset_cfg["kfold_base_path"]
    test_mode = dataset_cfg.get("test_mode", None)
    fold_datasets = get_fold_datasets(
        kfold_base_path, fold_id, test_mode=test_mode
    )

    dataloader_cfg = dataset_cfg.get("dataloader_config", {})
    dataset_kwargs = dataloader_cfg.get("dataset_kwargs", {})
    stat_path = dataset_cfg.get("stat_path")
    normalize_fields = dataloader_cfg.get("normalize_fields")

    bucket_cfg = dataset_cfg.get("bucket_sampler", {})
    use_bucketing = bucket_cfg.get("enabled", True)

    # --- Dataloader params ------------------------------------------------ #
    num_workers = dataloader_cfg.get("num_workers", 0)
    prefetch = dataloader_cfg.get("prefetch_factor", 2)
    pin_mem = dataloader_cfg.get("pin_memory", False)
    seg_duration = dataloader_cfg.get("segment_duration", 1200.0)
    guid_cache = dataloader_cfg.get("guid_cache_size", 128)

    batch_size_train = config["general_config"]["batch_size"]["train"]
    batch_size_test = config["general_config"]["batch_size"]["test"]

    model_cfg = config.get("model_config", {})
    use_precomputed = model_cfg.get("precompute_embeddings", False)

    # --- Create dataloaders ----------------------------------------------- #
    if use_precomputed:
        from model.transformer.classification.precompute_embeddings import (
            create_precomputed_embedding_dataloader,
        )

        precomputed_dir = model_cfg.get("precomputed_dir", "")
        transformer_checkpoint = model_cfg.get("transformer_checkpoint")

        common_kwargs = dict(
            num_workers=num_workers,
            segment_duration=seg_duration,
            guid_cache_size=guid_cache,
            stats_path=stat_path,
            normalize_fields=normalize_fields,
            prefetch_factor=prefetch,
            pin_memory=pin_mem,
            seed=seed,
            transformer_checkpoint=transformer_checkpoint,
            **dataset_kwargs,
        )

        train_precomputed = os.path.join(
            precomputed_dir, f"precomputed_fold_{fold_id}_train.hdf5"
        )
        val_precomputed = os.path.join(
            precomputed_dir, f"precomputed_fold_{fold_id}_val.hdf5"
        )

        bucket_ranges = bucket_cfg.get("bucket_ranges")
        train_loader, train_dataset = (
            create_precomputed_embedding_dataloader(
                precomputed_path=train_precomputed,
                hdf5_files=fold_datasets["train"],
                batch_size=batch_size_train,
                bucket_ranges=bucket_ranges,
                shuffle=bucket_cfg.get("shuffle", True),
                **common_kwargs,
            )
        )
        val_loader, val_dataset = (
            create_precomputed_embedding_dataloader(
                precomputed_path=val_precomputed,
                hdf5_files=fold_datasets["val"],
                batch_size=batch_size_test,
                bucket_ranges=bucket_ranges,
                shuffle=False,
                **common_kwargs,
            )
        )

    elif use_bucketing:
        from model.vae_teb_prediction.guid_classifier.length_bucket_sampler import (
            create_bucketed_sequence_dataloader,
        )

        logger.info(
            "Fold {}: Creating bucketed sequence dataloaders...", fold_id
        )

        bucket_ranges = bucket_cfg.get("bucket_ranges")
        common_dl_kwargs = dict(
            num_workers=num_workers,
            segment_duration=seg_duration,
            guid_cache_size=guid_cache,
            stats_path=stat_path,
            normalize_fields=normalize_fields,
            prefetch_factor=prefetch,
            pin_memory=pin_mem,
            seed=seed,
            **dataset_kwargs,
        )

        train_loader, train_dataset = create_bucketed_sequence_dataloader(
            hdf5_files=fold_datasets["train"],
            batch_size=batch_size_train,
            bucket_ranges=bucket_ranges,
            shuffle=bucket_cfg.get("shuffle", True),
            **common_dl_kwargs,
        )
        val_loader, val_dataset = create_bucketed_sequence_dataloader(
            hdf5_files=fold_datasets["val"],
            batch_size=batch_size_test,
            bucket_ranges=bucket_ranges,
            shuffle=False,
            **common_dl_kwargs,
        )

    else:
        from hdf5_dataset.guid_hdf5_dataset import (
            SignalSequenceDataset,
            sequence_collate_fn,
        )
        from torch.utils.data import DataLoader

        logger.info(
            "Fold {}: Creating standard sequence dataloaders...", fold_id
        )

        _ds_common = dict(
            segment_duration=seg_duration,
            guid_cache_size=guid_cache,
            stats_path=stat_path,
            normalize_fields=normalize_fields,
            pin_memory=pin_mem,
            **dataset_kwargs,
        )
        _dl_common = dict(
            num_workers=num_workers,
            collate_fn=sequence_collate_fn,
            drop_last=False,
            prefetch_factor=prefetch if num_workers > 0 else None,
            multiprocessing_context="spawn" if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            pin_memory=False,
        )

        train_dataset = SignalSequenceDataset(
            paths=fold_datasets["train"], **_ds_common
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size_train,
            shuffle=True, **_dl_common,
        )
        val_dataset = SignalSequenceDataset(
            paths=fold_datasets["val"], **_ds_common
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size_test,
            shuffle=False, **_dl_common,
        )

    logger.info(
        "Fold {}: train={} GUIDs, val={} GUIDs",
        fold_id, len(train_dataset), len(val_dataset),
    )

    # --- Class balance weights -------------------------------------------- #
    loss_cfg = model_cfg.get("loss", {})
    balance_cfg = loss_cfg.get("class_balance", {})
    loss_type = loss_cfg.get("type", "bce")

    pos_weight = None
    class_weights_list = None

    if balance_cfg.get("enabled", True):
        method = balance_cfg.get("method", "auto")
        if method == "auto":
            logger.info("Fold {}: Estimating class balance...", fold_id)
            balance = estimate_class_balance(train_dataset, loss_type)
            pos_weight = balance["pos_weight"]
            class_weights_list = balance["class_weights"]
        elif method == "manual":
            if loss_type == "bce":
                pos_weight = balance_cfg.get("manual_pos_weight", 1.0)
                logger.info(
                    "Fold {}: Using manual pos_weight={}", fold_id, pos_weight
                )
            else:
                class_weights_list = balance_cfg.get(
                    "manual_class_weights", [1.0, 1.0]
                )
                logger.info(
                    "Fold {}: Using manual class_weights={}",
                    fold_id, class_weights_list,
                )
    else:
        logger.info(
            "Fold {}: Class weighting disabled", fold_id
        )

    # --- Fold-specific config --------------------------------------------- #
    fold_output_dir = (
        Path(config["general_config"]["folders_config"]["out_dir_base"])
        / f"fold_{fold_id}"
    )
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    fold_config = config.copy()
    fold_config["general_config"] = {**config["general_config"]}
    fold_config["general_config"]["cuda_devices"] = [0]
    fold_config["general_config"]["folders_config"] = {
        "out_dir_base": str(fold_output_dir),
    }

    fold_config_path = fold_output_dir / "config.yaml"
    with open(fold_config_path, "w") as f:
        yaml.dump(fold_config, f, default_flow_style=False)
    logger.info("Fold {}: config saved to {}", fold_id, fold_config_path)

    # --- Create trainer and model ----------------------------------------- #
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logger.info(
        "Fold {}: Loading transformer and creating model...", fold_id
    )

    graph_model = GraphModelClassificationTrainer(
        config_file_path=str(fold_config_path)
    )
    graph_model.setup_config()
    graph_model.create_model(
        pos_weight=pos_weight,
        class_weights=class_weights_list,
    )
    logger.info("Fold {}: Model created on GPU {}", fold_id, gpu_id)

    # --- Train ------------------------------------------------------------ #
    logger.info("Fold {}: Starting training...", fold_id)
    start_time = time.time()
    trainer = graph_model.train_model(train_loader, val_loader)
    training_time_min = (time.time() - start_time) / 60.0
    logger.info(
        "Fold {}: Training completed in {:.2f} minutes",
        fold_id, training_time_min,
    )

    # Resolve best checkpoint metrics.
    best_metrics = resolve_best_checkpoint_metrics(
        trainer=trainer,
        pl_model=graph_model.pl_model,
        validation_dataloader=val_loader,
        checkpoint_callback=graph_model.checkpoint_callback,
    )
    best_val_loss = best_metrics["best_val_loss"]
    best_val_acc = best_metrics["best_val_accuracy"]
    best_ckpt_path = best_metrics["best_checkpoint_path"]

    logger.info(
        "Fold {}: best val_loss={:.4f}, val_accuracy={:.4f}, "
        "checkpoint={}",
        fold_id, best_val_loss, best_val_acc, best_ckpt_path,
    )

    # Cleanup.
    del train_loader, val_loader
    if hasattr(train_dataset, "clear_cache"):
        train_dataset.clear_cache()
    if hasattr(val_dataset, "clear_cache"):
        val_dataset.clear_cache()
    del train_dataset, val_dataset
    gc.collect()

    fold_results = {
        "fold_id": fold_id,
        "training_time_minutes": training_time_min,
        "best_val_loss_training": best_val_loss,
        "best_val_accuracy_training": best_val_acc,
        "best_checkpoint_path": best_ckpt_path,
        "status": "success",
    }

    results_path = fold_output_dir / "fold_results.json"
    with open(results_path, "w") as f:
        json.dump(fold_results, f, indent=2)

    return str(graph_model.model_checkpoint_dir), graph_model


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Single-fold training entry point.

    Reads ``config_classification.yaml`` from the same directory,
    trains fold 1 by default.
    """
    import numpy as np

    np.random.seed(42)
    torch.manual_seed(42)

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "config_classification.yaml",
    )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_cfg = config.get("dataset_config", {})
    fold_ids = dataset_cfg.get("fold_ids") or [1]
    gpu_ids = config["general_config"].get("cuda_devices", [0])

    for idx, fold_id in enumerate(fold_ids):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        checkpoint_dir, trainer = train_fold(
            fold_id=fold_id,
            config=config,
            gpu_id=gpu_id,
        )
        logger.info("Fold {} checkpoints at: {}", fold_id, checkpoint_dir)


if __name__ == "__main__":
    main()
