from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple, Union
from abc import ABC, abstractmethod
from pathlib import Path

import lightning as L
import torch

from loguru import logger
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

MetricDict = Dict[str, torch.Tensor]
"""Convenience alias used for the metric dictionaries each step returns."""


class LightningModelBase(L.LightningModule, ABC):
    """Base Lightning wrapper providing a repeatable training/validation skeleton.

    The class standardizes boilerplate required to convert an ordinary
    ``torch.nn.Module`` into a fully functional Lightning module. Sub-classes are
    encouraged to implement only the domain-specific logic, primarily
    ``compute_loss_and_metrics`` and optionally optimizer/scheduler builders or
    epoch hooks. Everything else—compiling the model, saving hyperparameters,
    logging learning-rate/metrics, and filtering trainable parameters—is handled
    centrally here so downstream Lightning modules remain concise.

    Metric logging workflow:

    * ``compute_loss_and_metrics`` must return ``(loss, metrics_dict)`` where keys
      are short metric names (``total_loss``, ``kld_loss``, ``beta`` ...).
    * ``LightningModelBase`` prefixes those keys with the trainer stage
      (``train/``, ``val/``, ``test/``) inside ``_log_metrics`` so every callback
      sees consistent names in ``trainer.callback_metrics``.
    * Built-in callbacks such as ``LossPlotCallback`` or Lightning's
      ``ModelCheckpoint`` simply reference those keys via their ``monitor``
      argument (e.g., ``monitor='val/total_loss'``) and need no extra wiring.
    * Any metric logged via ``self.log(...)`` or returned in ``metrics_dict`` is
      immediately available to externally supplied callbacks, so you can monitor
      reconstruction losses, hyper-parameters, or custom scalars just by adding
      them to the metrics dict.
    """

    prog_bar_metrics: Tuple[str, ...] = ("total_loss",)
    """Metric suffixes that should surface in Lightning's progress bar."""

    sync_dist_stages: Tuple[str, ...] = ("val", "test")
    """Trainer stages that require distributed metric synchronization."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        lr: float = 1e-4,
        lr_milestones: Optional[Iterable[int]] = None,
        weight_decay: float = 1e-4,
        module_name: Optional[str] = None,
    ) -> None:
        """
        Args:
            base_model: The raw ``nn.Module`` that performs inference and loss work.
            lr: Default learning rate stored in ``self.hparams``.
            lr_milestones: Optional milestone epochs for the scheduler helper.
            weight_decay: AdamW weight decay applied across parameters.
            module_name: Friendly name used in logs/debug messages.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['base_model'])
        self._orig_model = base_model  # Reference to the original module before compilation/wrapping
        self._wrapper_name = module_name or self.__class__.__name__  # Used in logs to identify this wrapper
        self.model = torch.compile(base_model) # you can consider mode="max-autotune" for long trainings

    @property
    def orig_model(self) -> nn.Module:
        """Return the underlying, non-Lightning module."""
        return self._orig_model

    def forward(self, *args, **kwargs):
        """Delegate forwards to the wrapped PyTorch module."""
        return self.model(*args, **kwargs)

    def on_train_epoch_start(self) -> None:
        """Expose LR telemetry and let subclasses refresh state before a new epoch."""
        self._log_learning_rate()
        self._on_train_epoch_start_hook()

    def training_step(self, batch, batch_idx):
        """Shared training step that delegates to ``compute_loss_and_metrics``."""
        return self._dispatch_stage_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        """Shared validation step mirroring ``training_step`` without grads."""
        return self._dispatch_stage_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        """Shared test step using the same metric dispatch path."""
        return self._dispatch_stage_step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        """Create the optimizer + optional scheduler using helper builders."""
        trainable_params = self._trainable_parameters()
        self._log_parameter_overview(trainable_params)
        optimizer = self.build_optimizer(trainable_params)
        scheduler = self.build_lr_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @abstractmethod
    def compute_loss_and_metrics(self, batch, batch_idx: int, stage: str) -> Tuple[torch.Tensor, MetricDict]:
        """Perform the forward/loss computation for the current stage.

        Implementations should:

        1. Run the wrapped model forward pass given the ``batch`` data.
        2. Compute the scalar loss tensor to backpropagate (typically named
            ``total_loss`` or similar).
        3. Build a metric dictionary where each value is either a tensor or a
            float/int convertible to tensor. Metrics can include the loss itself
            (e.g., ``{'total_loss': loss}``) or any auxiliary quantities.
        4. Return ``(loss, metrics)`` where ``loss`` participates in gradient
            calculation and ``metrics`` feeds the unified logging helper.

        Args:
            batch: The Lightning batch object passed into the stage step.
            batch_idx: Index of the batch within the epoch.
            stage: Literal string ``'train'``, ``'val'``, or ``'test'`` used to
                scope metric names (e.g., ``train/total_loss``).

        Returns:
            A tuple containing the scalar loss tensor and a dictionary of metrics
            to log. Missing metrics or ``None`` values are ignored gracefully.

        Example:
            >>> loss, metrics = self.compute_loss_and_metrics(batch, batch_idx, "train")
            >>> metrics
            {
                "total_loss": loss,
                "recon_loss": recon_loss,
                "kld_loss": kld,
                "beta": beta_value,  # logged as train/beta
            }
            The helper will automatically prefix each key with the current stage
            unless the name already contains '/'.
        Example Implementation:
        ```Python
        (
            y_st,        # scattering inputs
            y_ph,        # phase-harmonic inputs
            x_ph,        # cross-phase inputs (if used)
            y_raw,       # raw waveform target
            meta,        # optional extra info (guid, epoch, etc.)
        ) = batch

        # forward pass through SeqVaeTeb (compiled handle for speed)
        outputs = self.model(
            y_st=y_st,
            y_ph=y_ph,
            x_ph=x_ph,
            meta=meta,
        )

        # SeqVaeTeb already exposes a loss helper
        loss_dict = self.orig_model.compute_loss(
            forward_outputs=outputs,
            y_raw=y_raw,
            y_st=y_st,
            y_ph=y_ph,
            beta_override=self.hparams.get("kld_beta"),
            log_forecast_metrics=self.hparams.get("log_forecast_metrics", True),
        )

        total_loss = loss_dict["total_loss"]

        metrics = {
            "total_loss": total_loss,
            "reconstruction_loss": loss_dict.get("reconstruction_loss"),
            "kld_loss": loss_dict.get("kld_loss"),
            "forecast_loss": loss_dict.get("forecast_loss"),
            "beta": loss_dict.get("beta"),
        }
        return total_loss, metrics
        ```
        """

    def build_optimizer(self, trainable_params: Iterable[torch.nn.Parameter]) -> Optimizer:
        """Construct the optimizer; override for custom optimizers.

        The default uses ``torch.optim.AdamW`` with the learning-rate and
        weight-decay pulled from ``self.hparams``. Sub-classes can override this
        method to:

        * Swap in entirely different optimizers (SGD, Adam, Lion, etc.).
        * Group parameters with different hyperparameters.
        * Introduce optimizer-specific keyword arguments.

        Always return an ``Optimizer`` instance ready to be consumed by
        Lightning's ``configure_optimizers`` flow.
        """
        lr = float(getattr(self.hparams, "lr", 1e-4))
        weight_decay = float(getattr(self.hparams, "weight_decay", 1e-4))
        return torch.optim.AdamW(
            list(trainable_params),
            lr=lr,
            weight_decay=weight_decay,
            eps=1e-8,
            betas=(0.9, 0.95),
        )

    def build_lr_scheduler(
        self,
        optimizer: Optimizer,
    ) -> Optional[Union[_LRScheduler, Dict[str, Union[_LRScheduler, str, int]]]]:
        """Optional MultiStepLR scheduler builder.

        Uses milestones configured via ``self.hparams.lr_milestones``. Override
        this method to return ``None`` (no scheduler), a plain scheduler
        instance, or the richer dict structure Lightning expects when extra
        metadata (e.g., interval or frequency) is required.
        """
        milestones = getattr(self.hparams, "lr_milestones", None)
        if not milestones:
            return None
        from torch.optim.lr_scheduler import MultiStepLR

        gamma = float(getattr(self.hparams, "lr_gamma", 0.1))
        scheduler = MultiStepLR(optimizer, milestones=list(milestones), gamma=gamma)
        return {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }

    def _dispatch_stage_step(self, batch, batch_idx: int, stage: str):
        """Helper shared by train/val/test steps for consistent logging.

        Centralizes the boilerplate of calling ``compute_loss_and_metrics`` and
        logging the returned metrics. Sub-classes generally should not override
        ``training_step``/``validation_step``/``test_step`` directly unless they
        need non-standard behavior—customization typically happens inside
        ``compute_loss_and_metrics``.
        """
        loss, metrics = self.compute_loss_and_metrics(batch, batch_idx, stage)
        self._log_metrics(metrics, stage=stage, on_step=True)
        return loss

    def _log_learning_rate(self) -> None:
        """Report the first parameter group's LR once per epoch."""
        optimizer = self.optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]
        if not optimizer or not optimizer.param_groups:
            return
        lr_value = optimizer.param_groups[0].get("lr")
        if lr_value is None:
            return
        self.log("lr", lr_value, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def _log_metrics(self, metrics: MetricDict, *, stage: str, on_step: bool) -> None:
        """Unified metric logger framing keys as ``stage/name``."""
        if not metrics:
            return
        for raw_name, value in metrics.items():
            if value is None:
                continue
            name = raw_name if "/" in raw_name else f"{stage}/{raw_name}"
            metric_tensor = self._as_tensor(value)
            prog_bar = self._should_log_on_prog_bar(name)
            sync_dist = stage in self.sync_dist_stages
            self.log(name, metric_tensor, on_step=on_step, on_epoch=True, prog_bar=prog_bar, logger=True, sync_dist=sync_dist)

    def _trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        """Collect parameters with ``requires_grad`` for optimizer construction."""
        return [param for param in self.parameters() if param.requires_grad]

    def _log_parameter_overview(self, trainable_params: Iterable[torch.nn.Parameter]) -> None:
        """Emit a short breakdown of trainable vs total parameters."""
        trainable_params = list(trainable_params)
        total_params = sum(param.numel() for param in self.parameters())
        trainable_count = sum(param.numel() for param in trainable_params)
        frozen_count = total_params - trainable_count
        if total_params == 0:
            logger.warning("[{}] No parameters detected in model", self._wrapper_name)
            return
        logger.info("=" * 80)
        logger.info("[{}] Parameter overview", self._wrapper_name)
        logger.info("  Total parameters: {:,}", total_params)
        logger.info("  Trainable parameters: {:,} ({:.2f}%)", trainable_count, 100.0 * trainable_count / total_params)
        logger.info("  Frozen parameters: {:,} ({:.2f}%)", frozen_count, 100.0 * frozen_count / total_params)
        logger.info("=" * 80)

    def _should_log_on_prog_bar(self, name: str) -> bool:
        """Check whether the metric suffix is part of ``prog_bar_metrics``."""
        metric_name = name.split("/")[-1]
        return metric_name in self.prog_bar_metrics

    def _as_tensor(self, value) -> torch.Tensor:
        """Convert scalars/None into detached tensors for ``self.log``."""
        if isinstance(value, torch.Tensor):
            tensor = value
        elif isinstance(value, (float, int)):
            tensor = torch.tensor(float(value), device=self.device)
        else:
            tensor = torch.tensor(0.0, device=self.device)
        return tensor.detach()

    def _on_train_epoch_start_hook(self) -> None:
        """Optional hook for subclasses that need to refresh schedulers or state."""
