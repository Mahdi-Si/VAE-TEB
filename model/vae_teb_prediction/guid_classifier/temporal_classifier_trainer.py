"""Training pipeline for the temporal VAE classifier.

Provides the PyTorch Lightning wrapper (``PlTemporalClassifier``) and the
``GraphModelBase``-derived trainer (``GraphModelTemporalTrainer``) that
orchestrate single-fold training of the :class:`TemporalVaeClassifier`.

Typical workflow::

    trainer = GraphModelTemporalTrainer(config_file_path="config_temporal.yaml")
    trainer.setup_config()
    trainer.create_model()
    lightning_trainer = trainer.train_model(train_loader, val_loader)

Or use the convenience :func:`train_fold` to train a single fold end-to-end
from a config dict.
"""

from __future__ import annotations

import gc
import os
import time
import json
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
from model.vae_teb_prediction.guid_classifier.temporal_classification_model import (
    TemporalVaeClassifier,
)


# ---------------------------------------------------------------------------
#  Lightning Module
# ---------------------------------------------------------------------------


class PlTemporalClassifier(LightningModelBase):
    """PyTorch Lightning wrapper for :class:`TemporalVaeClassifier`.

    Inherits from :class:`LightningModelBase` and overrides
    :meth:`compute_loss_and_metrics` to handle GUID-sequence batches from
    ``sequence_collate_fn``.

    Important:
        Overrides ``__init__`` to bypass ``torch.compile`` because the
        ``TemporalVaeClassifier.forward()`` contains dynamic control flow
        (chunked VAE encoding, variable-length packed sequences, conditional
        feature encoding) that is incompatible with ``torch.compile`` graph
        tracing.

    Args:
        base_model: A :class:`TemporalVaeClassifier` instance.
        lr: Learning rate.
        lr_milestones: Epochs for learning-rate decay.
        class_weights: Optional per-class weights for cross-entropy loss.
        weight_decay: AdamW weight-decay coefficient.
    """

    prog_bar_metrics = ("loss", "accuracy")

    def __init__(
        self,
        base_model: nn.Module,
        *,
        lr: float = 1e-3,
        lr_milestones: Optional[Sequence[int]] = None,
        class_weights: Optional[List[float]] = None,
        weight_decay: float = 1e-4,
    ) -> None:
        # Bypass torch.compile entirely.  LightningModelBase.__init__ calls
        # torch.compile(base_model) which can fail on Windows and produces
        # graph-break errors with dynamic control flow
        # (pack_padded_sequence, chunked VAE encoding, conditional branches).
        #
        # We replicate the base __init__ manually, skipping the compile step.
        pl.LightningModule.__init__(self)
        self.save_hyperparameters(ignore=["base_model"])
        self._orig_model = base_model
        self._wrapper_name = self.__class__.__name__
        self.model = base_model  # Eager mode — no torch.compile

        if class_weights is not None:
            logger.info(
                "PlTemporalClassifier: class weights {} (applied by inner model)",
                class_weights,
            )

    def training_step(self, batch, batch_idx):
        """Override to log training metrics at epoch-level only.

        The base class logs with ``on_step=True`` which puts the **last step
        value** into ``callback_metrics["train/loss"]``.  For the temporal
        classifier the per-step values are very noisy (variable batch
        composition from the bucket sampler), making the training loss plot
        look like random noise.  Logging with ``on_step=False`` ensures
        ``callback_metrics["train/loss"]`` contains the epoch average,
        matching the behaviour of validation metrics.
        """
        loss, metrics = self.compute_loss_and_metrics(batch, batch_idx, stage="train")
        self._log_metrics(metrics, stage="train", on_step=False)
        return loss

    def compute_loss_and_metrics(
        self, batch: Dict, batch_idx: int, stage: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run forward pass and compute masked cross-entropy loss.

        Args:
            batch: Dict from ``sequence_collate_fn`` with keys ``fhr_st``,
                ``fhr_ph``, ``fhr_up_ph``, ``delta_t``, ``mask``, ``lengths``,
                ``target``, etc.
            batch_idx: Index of the current batch within the epoch.
            stage: One of ``'train'``, ``'val'``, ``'test'``.

        Returns:
            Tuple of ``(loss, metrics_dict)`` where ``metrics_dict`` keys are
            short names (``loss``, ``accuracy``, ``class_0_acc``,
            ``class_1_acc``).  The base class prefixes them with ``stage/``.
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
        # Capture optional CausalMIL-specific decomposed loss components.
        for extra_key in ("loss_guid", "loss_mono"):
            if extra_key in loss_dict:
                metrics[extra_key] = loss_dict[extra_key]
        return loss, metrics


# ---------------------------------------------------------------------------
#  Class Weight Estimation (GUID-level)
# ---------------------------------------------------------------------------


def estimate_temporal_class_weights(
    dataset,
) -> torch.Tensor:
    """Estimate class weights from GUID-level labels using an existing dataset.

    Delegates to ``SignalSequenceDataset.estimate_class_weights()`` which
    reads only the ``target`` field from HDF5 in bulk — no redundant dataset
    creation or full signal data loading.

    Args:
        dataset: An already-instantiated ``SignalSequenceDataset``.

    Returns:
        Tensor of shape ``(2,)`` — ``[healthy_weight, unhealthy_weight]``.
        Weights are inverse-frequency normalised so they sum to
        ``num_classes = 2``.
    """
    weights, counts = dataset.estimate_class_weights(num_classes=2)

    if counts.sum() == 0:
        logger.warning("No GUIDs found for class weight estimation, returning uniform weights")
        return torch.ones(2)

    logger.info(
        "Temporal class weights — GUID counts (healthy={}, unhealthy={}), weights={}",
        counts[0].item(), counts[1].item(), weights.tolist(),
    )
    return weights


def resolve_best_checkpoint_metrics(
    trainer: pl.Trainer,
    pl_model: pl.LightningModule,
    validation_dataloader,
    checkpoint_callback: ModelCheckpoint,
) -> Dict[str, float]:
    """Resolve validation metrics for the actual best checkpoint.

    Lightning's ``trainer.callback_metrics`` reflects the most recent epoch,
    not necessarily the checkpoint selected by ``ModelCheckpoint``.  This
    helper uses the callback's ``best_model_score`` for the monitored metric
    and, when possible, runs a targeted validation pass on the best checkpoint
    to recover the corresponding accuracy and any other validation metrics.

    Args:
        trainer: Fitted Lightning trainer.
        pl_model: Lightning module used during training.
        validation_dataloader: Validation dataloader for the current fold.
        checkpoint_callback: The ModelCheckpoint instance attached to the run.

    Returns:
        Dict containing ``best_checkpoint_path``, ``best_val_loss``, and
        ``best_val_accuracy``.
    """
    def _to_float(value, default: float) -> float:
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    callback_metrics = getattr(trainer, "callback_metrics", {}) or {}
    best_model_path = getattr(checkpoint_callback, "best_model_path", "") or ""
    best_model_score = getattr(checkpoint_callback, "best_model_score", None)

    if best_model_score is not None:
        best_val_loss = _to_float(best_model_score, float("inf"))
    else:
        best_val_loss = _to_float(
            callback_metrics.get("val/loss"),
            float("inf"),
        )

    best_val_accuracy = _to_float(callback_metrics.get("val/accuracy"), 0.0)

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
                    best_val_loss = _to_float(best_metrics["val/loss"], best_val_loss)
                if "val/accuracy" in best_metrics:
                    best_val_accuracy = _to_float(
                        best_metrics["val/accuracy"],
                        best_val_accuracy,
                    )
        except Exception as exc:
            logger.warning(
                "Failed to validate best checkpoint {} for metric recovery: {}",
                best_model_path,
                exc,
            )

    return {
        "best_checkpoint_path": best_model_path,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
    }


# ---------------------------------------------------------------------------
#  GraphModel Trainer
# ---------------------------------------------------------------------------


class GraphModelTemporalTrainer(GraphModelBase):
    """Training pipeline for the temporal VAE classifier.

    Reads configuration from ``config_temporal.yaml``, creates a
    :class:`TemporalVaeClassifier` with a frozen VAE encoder, wraps it in
    :class:`PlTemporalClassifier`, and runs training via PyTorch Lightning.

    Inherits directory management, logging, and config loading from
    :class:`GraphModelBase`.

    Args:
        config_file_path: Path to the YAML configuration file.  Defaults to
            ``config_temporal.yaml`` in the same directory as this module.
    """

    def __init__(self, config_file_path: Optional[str] = None) -> None:
        if config_file_path is None:
            config_file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "config_temporal.yaml",
            )
        super().__init__(config_file_path)

    def create_model(
        self,
        class_weights: Optional[List[float]] = None,
    ) -> None:
        """Instantiate :class:`TemporalVaeClassifier` and wrap in Lightning.

        Steps:
            1. Load pre-trained VAE checkpoint.
            2. Build :class:`TemporalVaeClassifier` from ``model_config``.
            3. Log trainable vs frozen parameter counts.
            4. Wrap in :class:`PlTemporalClassifier`.

        Args:
            class_weights: Optional per-class weights.  If ``None``, no
                class weighting is applied to the cross-entropy loss.
        """
        model_cfg = self.config.get("model_config", {})

        # ----- 1. Load VAE ------------------------------------------------ #
        vae_checkpoint = model_cfg.get("vae_checkpoint")
        if vae_checkpoint is None:
            raise ValueError(
                "vae_checkpoint must be provided in model_config"
            )

        from model.vae_teb_prediction.vae_teb_model_prediction import SeqVae

        vae_model = SeqVae()
        loaded_vae = load_checkpoint_strict(vae_model, checkpoint=vae_checkpoint)
        if loaded_vae is None:
            raise RuntimeError(
                "Strict VAE checkpoint loading failed during temporal "
                f"training setup. Checkpoint: {vae_checkpoint}"
            )
        logger.info("VAE model loaded from checkpoint: {}", vae_checkpoint)

        # ----- 2. Build classifier model ---------------------------------- #
        architecture_type = model_cfg.get("architecture_type", "temporal_lstm")
        seg_cfg = model_cfg.get("segment_encoder", {})
        lstm_cfg = model_cfg.get("temporal_lstm", {})
        feat_cfg = model_cfg.get("temporal_features", {})
        head_cfg = model_cfg.get("classifier_head", {})

        seg_idx_cfg = feat_cfg.get("segment_index", {})
        tlo_cfg = feat_cfg.get("time_from_labor_onset", {})
        dt_cfg = feat_cfg.get("delta_t", {})

        if architecture_type in ("abmil", "transmil", "causal_mil"):
            # --- MIL-based architecture --- #
            from model.vae_teb_prediction.guid_classifier.mil_classification_model import (
                ABMILClassifier,
                TransMILClassifier,
                CausalMILClassifier,
            )

            _MIL_MAP = {
                "abmil": ABMILClassifier,
                "transmil": TransMILClassifier,
                "causal_mil": CausalMILClassifier,
            }
            mil_cfg = model_cfg.get("mil_config", {})
            cls = _MIL_MAP[architecture_type]

            self.pytorch_model = cls(
                vae_model=vae_model,
                segment_encoder_type=seg_cfg.get("type", "simple"),
                d_seg=seg_cfg.get("d_seg", 128),
                delta_t_embed_dim=(
                    dt_cfg.get("embed_dim", 8) if dt_cfg.get("enabled", True) else 0
                ),
                delta_t_dropout=dt_cfg.get("dropout", 0.1),
                position_embed_dim=(
                    seg_idx_cfg.get("embed_dim", 8) if seg_idx_cfg.get("enabled", False) else 0
                ),
                max_position_index=seg_idx_cfg.get("max_index", 40),
                tlo_enabled=tlo_cfg.get("enabled", False),
                tlo_embed_dim=tlo_cfg.get("embed_dim", 0),
                tlo_dropout=tlo_cfg.get("dropout", 0.1),
                num_classes=head_cfg.get("num_classes", 2),
                class_weights=class_weights,
                vae_chunk_size=model_cfg.get("vae_chunk_size", 32),
                use_posterior=model_cfg.get("use_posterior", True),
                freeze_vae=model_cfg.get("freeze_vae", True),
                rich_conv_channels=seg_cfg.get("rich_conv_channels", [32, 64, 128]),
                rich_kernel_sizes=seg_cfg.get("rich_kernel_sizes", [5, 7, 11]),
                rich_dilations=seg_cfg.get("rich_dilations", [1, 2, 4]),
                **mil_cfg,
            )
            logger.info("Architecture: {} (MIL)", architecture_type)
        else:
            # --- Original temporal LSTM architecture (default) --- #
            self.pytorch_model = TemporalVaeClassifier(
                vae_model=vae_model,
                segment_encoder_type=seg_cfg.get("type", "mean_pool"),
                d_seg=seg_cfg.get("d_seg", 64),
                temporal_lstm_hidden=lstm_cfg.get("hidden_dim", 128),
                temporal_lstm_layers=lstm_cfg.get("num_layers", 2),
                temporal_lstm_dropout=lstm_cfg.get("dropout", 0.1),
                gap_encoding=model_cfg.get("gap_encoding", "concat"),
                position_embed_dim=(
                    seg_idx_cfg.get("embed_dim", 8) if seg_idx_cfg.get("enabled", False) else 0
                ),
                max_position_index=seg_idx_cfg.get("max_index", 40),
                tlo_enabled=tlo_cfg.get("enabled", False),
                tlo_embed_dim=tlo_cfg.get("embed_dim", 0),
                tlo_dropout=tlo_cfg.get("dropout", 0.1),
                delta_t_embed_dim=dt_cfg.get("embed_dim", 0),
                delta_t_dropout=dt_cfg.get("dropout", 0.1),
                persist_segment_state=seg_cfg.get("persist_state", False),
                segment_state_decay=seg_cfg.get("state_decay", True),
                temporal_lstm_residual=lstm_cfg.get("residual", False),
                num_classes=head_cfg.get("num_classes", 2),
                classifier_dropout=head_cfg.get("dropout", 0.1),
                mlp_multiplier=head_cfg.get("mlp_multiplier", 2.0),
                classifier_num_residual_blocks=head_cfg.get("num_residual_blocks", 0),
                classifier_bottleneck_dim=head_cfg.get("bottleneck_dim", 64),
                output_dropout=head_cfg.get("output_dropout", 0.0),
                class_weights=class_weights,
                vae_chunk_size=model_cfg.get("vae_chunk_size", 32),
                use_posterior=model_cfg.get("use_posterior", True),
                freeze_vae=model_cfg.get("freeze_vae", True),
                cnn_kernel=seg_cfg.get("cnn_kernel", 7),
            )
            logger.info("Architecture: temporal_lstm (default)")

        # ----- 3. Log parameter counts ----------------------------------- #
        total_params = sum(p.numel() for p in self.pytorch_model.parameters())
        trainable_params = sum(
            p.numel() for p in self.pytorch_model.parameters() if p.requires_grad
        )
        frozen_params = total_params - trainable_params
        logger.info("Total parameters: {:,}", total_params)
        logger.info("Trainable parameters: {:,}", trainable_params)
        logger.info("Frozen parameters: {:,}", frozen_params)

        # ----- 4. Wrap in Lightning --------------------------------------- #
        self.pl_model = PlTemporalClassifier(
            self.pytorch_model,
            lr=self.lr,
            lr_milestones=self.lr_milestones,
            class_weights=class_weights,
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
        """Train the temporal classifier using PyTorch Lightning.

        Sets up callbacks (ModelCheckpoint, EarlyStopping, LossPlot,
        MetricsLogging, HyperparameterLogging), creates a Lightning Trainer
        with ``use_distributed_sampler=False`` (we supply our own
        ``LengthBucketSampler``), and calls ``trainer.fit()``.

        Args:
            train_dataloader: Training DataLoader (bucketed sequence loader).
            validation_dataloader: Validation DataLoader.

        Returns:
            The Lightning :class:`~lightning.pytorch.Trainer` instance after
            fitting completes.
        """
        callbacks_cfg = self.config.get("advanced_config", {}).get("callbacks", {})

        # --- Callbacks ---------------------------------------------------- #
        self.metrics_callback = MetricsLoggingCallback()
        self.loss_plot_callback = LossPlotCallback(
            output_dir=self.train_results_dir,
            plot_frequency=self.config["general_config"].get("plot_frequency", 10),
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
            filename="temporal-model-{epoch:02d}",
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
            # CRITICAL: We use our own LengthBucketSampler — disable
            # Lightning's automatic DistributedSampler wrapping.
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
                "strategy": "ddp" if len(self.cuda_devices) > 1 else "auto",
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
) -> Tuple[str, GraphModelTemporalTrainer]:
    """Train the temporal classifier on a single fold.

    This is the primary entry point for single-fold training.  It handles
    dataset loading, class weight estimation, config writing, model creation,
    and training.

    Args:
        fold_id: Fold number (1-10).
        config: Full configuration dict from ``config_temporal.yaml``.
        gpu_id: GPU device to use.

    Returns:
        Tuple of ``(checkpoint_dir, trainer_instance)``.

    Raises:
        ValueError: If required config keys are missing.
    """
    import random
    import numpy as np

    logger.info("Starting temporal fold {} on GPU {}", fold_id, gpu_id)

    # Seed for reproducibility
    seed = 42 + fold_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Dataset paths ---------------------------------------------------- #
    from model.vae_teb_prediction.kfold_classifier_trainer import get_fold_datasets
    from model.vae_teb_prediction.guid_classifier.length_bucket_sampler import (
        create_bucketed_sequence_dataloader,
    )

    dataset_cfg = config.get("dataset_config", {})
    kfold_base_path = dataset_cfg["kfold_base_path"]
    fold_datasets = get_fold_datasets(kfold_base_path, fold_id)

    dataloader_cfg = dataset_cfg.get("dataloader_config", {})
    dataset_kwargs = dataloader_cfg.get("dataset_kwargs", {})
    stat_path = dataset_cfg.get("stat_path")
    normalize_fields = dataloader_cfg.get("normalize_fields")

    bucket_cfg = dataset_cfg.get("bucket_sampler", {})
    use_bucketing = bucket_cfg.get("enabled", True)

    # --- Create dataloaders ----------------------------------------------- #
    num_workers = dataloader_cfg.get("num_workers", 0)
    prefetch = dataloader_cfg.get("prefetch_factor", 2)
    pin_mem = dataloader_cfg.get("pin_memory", False)
    seg_duration = dataloader_cfg.get("segment_duration", 1200.0)
    guid_cache = dataloader_cfg.get("guid_cache_size", 128)

    batch_size_train = config["general_config"]["batch_size"]["train"]
    batch_size_test = config["general_config"]["batch_size"]["test"]

    if use_bucketing:
        # --- Bucketed sampling: groups GUIDs by segment count ------------- #
        logger.info("Fold {}: Creating bucketed sequence dataloaders...", fold_id)

        bucket_ranges = bucket_cfg.get("bucket_ranges")
        bucket_shuffle = bucket_cfg.get("shuffle", True)

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
            shuffle=bucket_shuffle,
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
        # --- Standard random sampling: full diversity per batch ----------- #
        logger.info("Fold {}: Creating standard (non-bucketed) sequence dataloaders...", fold_id)

        from hdf5_dataset.guid_hdf5_dataset import (
            SignalSequenceDataset,
            sequence_collate_fn,
        )
        from torch.utils.data import DataLoader

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

        train_dataset = SignalSequenceDataset(paths=fold_datasets["train"], **_ds_common)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size_train, shuffle=True, **_dl_common,
        )

        val_dataset = SignalSequenceDataset(paths=fold_datasets["val"], **_ds_common)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size_test, shuffle=False, **_dl_common,
        )

    logger.info(
        "Fold {}: train={} GUIDs, val={} GUIDs",
        fold_id, len(train_dataset), len(val_dataset),
    )

    # --- Class weights ---------------------------------------------------- #
    head_cfg = config.get("model_config", {}).get("classifier_head", {})
    use_class_weights = head_cfg.get("use_class_weights", True)

    if use_class_weights:
        logger.info("Fold {}: Estimating class weights...", fold_id)
        class_weights = estimate_temporal_class_weights(train_dataset)
        class_weights_list = class_weights.tolist()
        logger.info("Fold {}: Class weights estimated: {}", fold_id, class_weights_list)
    else:
        class_weights_list = None
        logger.info("Fold {}: Class weighting disabled, using uniform CE loss", fold_id)

    # --- Fold-specific config --------------------------------------------- #
    fold_output_dir = Path(config["general_config"]["folders_config"]["out_dir_base"]) / f"fold_{fold_id}"
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
    # Set CUDA_VISIBLE_DEVICES so PyTorch sees only the assigned GPU.
    # When called from kfold_temporal_trainer the caller already set this,
    # but when called standalone (main()) this is the only place it happens.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logger.info("Fold {}: Loading VAE checkpoint and creating model...", fold_id)

    graph_model = GraphModelTemporalTrainer(config_file_path=str(fold_config_path))
    graph_model.setup_config()
    graph_model.create_model(class_weights=class_weights_list)
    logger.info("Fold {}: Model created on GPU {}", fold_id, gpu_id)

    # --- Train ------------------------------------------------------------ #
    logger.info("Fold {}: Starting training...", fold_id)
    start_time = time.time()
    trainer = graph_model.train_model(train_loader, val_loader)
    training_time_min = (time.time() - start_time) / 60.0
    logger.info("Fold {}: Training completed in {:.2f} minutes", fold_id, training_time_min)

    # Resolve metrics for the actual best checkpoint before tearing down the
    # validation dataloader.
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
        "Fold {}: best val_loss={:.4f}, val_accuracy={:.4f}, checkpoint={}",
        fold_id,
        best_val_loss,
        best_val_acc,
        best_ckpt_path,
    )

    # Cleanup training dataloaders and dataset caches.
    del train_loader, val_loader
    train_dataset.clear_cache()
    val_dataset.clear_cache()
    del train_dataset, val_dataset
    gc.collect()
    logger.info("Fold {}: Training dataloaders and caches cleaned up", fold_id)

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

    Reads ``config_temporal.yaml`` from the same directory as this module,
    trains fold 1 by default.
    """
    import numpy as np

    np.random.seed(42)
    torch.manual_seed(42)

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "config_temporal.yaml",
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
