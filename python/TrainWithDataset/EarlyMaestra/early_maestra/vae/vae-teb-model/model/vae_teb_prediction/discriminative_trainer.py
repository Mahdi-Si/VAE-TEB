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

from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

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


def map_labels(raw_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Map raw dataset labels ``{1, 2, 3}`` to 0-indexed class indices.

    Args:
        raw_labels: Per-sample labels of shape ``(B,)`` with values in
            ``{1, 2, 3}`` (HEALTHY=1, ACIDOSIS=2, HIE=3).
        num_classes: Number of target classes.

            * ``2`` — Binary: HEALTHY (1) -> 0, UNHEALTHY (2 or 3) -> 1.
            * ``3`` — Three-class: ``(raw_labels - 1)`` giving {0, 1, 2}.

    Returns:
        Integer tensor of shape ``(B,)`` with values in
        ``{0, ..., num_classes - 1}``.

    Raises:
        ValueError: If ``num_classes`` is not 2 or 3.
    """
    if num_classes == 2:
        return (raw_labels > 1).long()
    elif num_classes == 3:
        return (raw_labels - 1).long()
    else:
        raise ValueError(f"num_classes must be 2 or 3, got {num_classes}")


def compute_class_weights_from_dataloader(
    dataloader,
    num_classes: int = 3,
    max_batches: int = 50,
) -> list[float]:
    """Scan training data to compute inverse-frequency class weights.

    Iterates over up to ``max_batches`` batches from the dataloader,
    counts samples per class, and returns weights proportional to
    ``1 / frequency``, normalized so the smallest weight equals 1.0.

    Args:
        dataloader: Training dataloader whose batches have a ``.target``
            attribute of shape ``(B, T)`` with values in ``{1, 2, 3}``.
        num_classes: Number of distinct classes (labels 1..num_classes).
        max_batches: Maximum number of batches to scan. Set to 0 or
            negative to scan the entire dataloader.

    Returns:
        List of ``num_classes`` floats — per-class weights ordered by
        class index (class 1 first, class num_classes last).
    """
    counts = torch.zeros(num_classes, dtype=torch.long)
    n_batches = 0

    for n_batches, batch in enumerate(dataloader, start=1):
        if 0 < max_batches < n_batches:
            break
        target_seq = batch.target  # (B, T)
        raw_labels = target_seq.max(dim=1)[0]  # (B,) values in {1, 2, 3}
        mapped = map_labels(raw_labels, num_classes)  # (B,) values in {0..C-1}
        for c in range(num_classes):
            counts[c] += (mapped == c).sum().item()

    total = counts.sum().item()
    if total == 0:
        logger.warning("No samples found when computing class weights, using uniform weights")
        return [1.0] * num_classes

    # Inverse frequency: weight_c = total / (num_classes * count_c)
    weights = []
    for c in range(num_classes):
        if counts[c] == 0:
            weights.append(1.0)
            logger.warning(f"Class {c + 1} has 0 samples in scanned batches")
        else:
            weights.append(total / (num_classes * counts[c].item()))

    # Normalize so the smallest weight is 1.0
    min_w = min(weights)
    weights = [w / min_w for w in weights]

    logger.info(
        f"Auto class weights (scanned {n_batches} batches, "
        f"{total} samples): counts={counts.tolist()}, weights={[f'{w:.2f}' for w in weights]}"
    )
    return weights


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

        # Aggregate to single label per sample and map to 0-indexed classes
        raw_labels = target_seq.max(dim=1)[0]  # (B,) values in {1, 2, 3}
        num_classes = self._orig_model.num_classes
        labels = map_labels(raw_labels, num_classes)  # (B,) values in {0..C-1}

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

        # Per-class accuracies (only present when class has samples in batch)
        for c in range(self._orig_model.num_classes):
            key = f"cls_acc_class_{c}"
            if key in loss_dict:
                metrics[key] = loss_dict[key]

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


class _RunnerAdapter:
    """Minimal stand-in for ``TestRunner`` used by the sample-plot callback.

    ``_plot_all_single_sample_plots`` accesses ``runner.model``,
    ``runner.warmup_steps``, and ``runner.decimation_factor``.  This adapter
    exposes the underlying ``SeqVae`` with those attributes so the plotting
    function works without importing the full testing infrastructure.
    """

    def __init__(self, vae_model: SeqVae) -> None:
        self.model = vae_model
        self.warmup_steps: int = int(vae_model.warmup_period)
        self.decimation_factor: int = int(vae_model.decimation_factor)


class SamplePlotCallback(pl.Callback):
    """Lightning callback that generates single-sample VAE plots during training.

    Every ``plot_frequency`` epochs, draws ``n_samples`` from the validation
    dataloader, runs a forward pass through the current model, and generates
    the same comprehensive reconstruction-analysis figure produced by
    ``plot_single_samples.py``.  Plots are saved under
    ``<output_dir>/sample_plots/epoch_XXXX/``.

    Attributes:
        validation_dataloader: Held reference to the validation dataloader.
        output_dir: Root directory for sample plot output.
        plot_frequency: Generate plots every N epochs.
        n_samples: Number of samples to plot per epoch.
    """

    def __init__(
        self,
        validation_dataloader: Any,
        output_dir: str,
        plot_frequency: int = 10,
        n_samples: int = 3,
    ) -> None:
        """Initialize the sample-plot callback.

        Args:
            validation_dataloader: Validation dataloader to draw samples from.
            output_dir: Directory where ``sample_plots/`` will be created.
            plot_frequency: Plot every N epochs.
            n_samples: Number of samples to plot each time.
        """
        super().__init__()
        self.validation_dataloader = validation_dataloader
        self.output_dir = Path(output_dir) / "sample_plots"
        self.plot_frequency = plot_frequency
        self.n_samples = n_samples
        self._stats: Any = None

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Generate sample plots at the configured frequency.

        Args:
            trainer: The Lightning trainer instance.
            pl_module: The Lightning module being trained.
        """
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_frequency != 0:
            return

        try:
            self._generate_plots(trainer, pl_module)
        except Exception as exc:
            logger.warning(f"Sample plotting failed at epoch {epoch}: {exc}")

    def _generate_plots(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Run forward pass on validation samples and generate plots.

        Args:
            trainer: The Lightning trainer instance.
            pl_module: The Lightning module being trained.
        """
        # Lazy imports to avoid hard dependency on the testing module
        from model.vae_teb_prediction.testing.plot_single_samples import (
            _get_normalization_stats,
            _plot_all_single_sample_plots,
            _sanitize_folder_name,
        )
        from model.vae_teb_prediction.testing.collectors import (
            _extract_epoch,
            _extract_guid,
        )

        device = pl_module.device
        disc_model: DiscriminativeSeqVae = pl_module._orig_model
        vae_model = disc_model.vae_model
        adapter = _RunnerAdapter(vae_model)

        # Cache normalization stats on first call
        if self._stats is None:
            self._stats = _get_normalization_stats(self.validation_dataloader)

        # Grab one batch from validation
        batch = next(iter(self.validation_dataloader))

        # Forward pass (no grad, eval mode)
        was_training = disc_model.training
        disc_model.eval()
        with torch.no_grad():
            outputs = disc_model(
                y_st=batch.fhr_st.to(device),
                y_ph=batch.fhr_ph.to(device),
                x_ph=batch.fhr_up_ph.to(device),
            )
        if was_training:
            disc_model.train()

        # Create per-epoch output directory
        epoch_dir = self.output_dir / f"epoch_{trainer.current_epoch:04d}"

        n = min(self.n_samples, batch.fhr_st.size(0))
        for idx in range(n):
            guid = _extract_guid(batch, idx)
            epoch_val = _extract_epoch(batch, idx)
            sample_name = _sanitize_folder_name(
                guid or f"sample_{idx}", epoch_val or 0.0,
            )
            _plot_all_single_sample_plots(
                runner=adapter,
                batch=batch,
                idx=idx,
                outputs=outputs,
                sample_dir=epoch_dir,
                sample_name=sample_name,
                stats=self._stats,
            )

        logger.info(
            f"Plotted {n} validation samples at epoch "
            f"{trainer.current_epoch} -> {epoch_dir}"
        )


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

    def create_model(
        self,
        class_weights: list[float] | None = None,
    ) -> None:
        """Create the DiscriminativeSeqVae and wrap in Lightning.

        Reads configuration from ``model_config.discriminative`` and
        ``model_config.core_model_checkpoint`` to build the full training
        pipeline.

        Args:
            class_weights: Pre-computed class weights (e.g. from
                ``compute_class_weights_from_dataloader``).  If provided,
                overrides the ``class_weights`` value in the config file.
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
        # Use passed-in class_weights (from auto-compute) or fall back to config
        if class_weights is None:
            cfg_weights = disc_config.get("class_weights")
            if isinstance(cfg_weights, list):
                class_weights = [float(w) for w in cfg_weights]

        self.pytorch_model = DiscriminativeSeqVae(
            vae_model=vae_model,
            num_classes=disc_config.get("num_classes", 3),
            classifier_hidden_dim=disc_config.get("classifier_hidden_dim", 32),
            center_ema_decay=disc_config.get("center_ema_decay", 0.99),
            alpha_recon=disc_config.get("alpha_recon", 1.0),
            alpha_kld=disc_config.get("alpha_kld", 1.0),
            alpha_center=disc_config.get("alpha_center", 0.1),
            alpha_cls=disc_config.get("alpha_cls", 0.5),
            class_weights=class_weights,
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
            filename="disc-finetune-epoch={epoch:02d}",
            save_top_k=callbacks_cfg.get("model_checkpoint", {}).get("save_top_k", 3),
            mode="min",
        )
        early_stopping_callback = EarlyStopping(
            monitor="val/total_loss",
            patience=callbacks_cfg.get("early_stopping", {}).get("patience", 30),
            mode="min",
            verbose=True,
        )

        sample_plot_callback = SamplePlotCallback(
            validation_dataloader=validation_dataloader,
            output_dir=self.train_results_dir,
            plot_frequency=self.plot_every_epoch,
            n_samples=3,
        )

        callback_list = [
            metrics_callback,
            loss_plot_callback,
            hyperparam_callback,
            checkpoint_callback,
            early_stopping_callback,
            sample_plot_callback,
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

    # Auto-compute class weights from training data if configured
    disc_config = config.get("model_config", {}).get("discriminative", {})
    num_classes = disc_config.get("num_classes", 3)
    cfg_class_weights = disc_config.get("class_weights")

    class_weights = None
    if cfg_class_weights == "auto":
        logger.info("Computing class weights automatically from training data...")
        class_weights = compute_class_weights_from_dataloader(
            train_dataloader, num_classes=num_classes,
        )
    elif isinstance(cfg_class_weights, list):
        class_weights = [float(w) for w in cfg_class_weights]
        logger.info(f"Using manual class weights from config: {class_weights}")

    graph_model = GraphModelDiscriminativeTrainer(
        config_file_path=r"config_discriminative.yaml"
    )
    graph_model.setup_config()
    graph_model.create_model(class_weights=class_weights)
    graph_model.train_model(train_dataloader, validation_dataloader)

    end_time = time.time()
    logger.info(f"Training completed in {(end_time - start_time) / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
