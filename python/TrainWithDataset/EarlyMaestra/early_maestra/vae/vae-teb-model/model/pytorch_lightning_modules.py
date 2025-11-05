import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYDEVD_USE_CYTHON'] = "NO"

import lightning as L
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
import torch
# Disable interactive mode and set thread-safe parameters

from typing import Dict, Optional, Tuple

from vae_teb_model import SeqVaeTeb, ensure_compiled_module, is_compiled_module

def _log_metric_tensor(value: Optional[torch.Tensor], reference: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    elif value is None:
        tensor = reference.new_tensor(default)
    else:
        tensor = reference.new_tensor(float(value))
    return tensor.detach()

torch.backends.cudnn.enabled = True

from loguru import logger


# ------------------------------------------------------------------------------------------------------------------------------------------
# Lightning Module
# ------------------------------------------------------------------------------------------------------------------------------------------
# Note: Callbacks have been moved to model/callbacks.py for better organization
# Import them from: from model.callbacks import (
#     LossPlotCallback, ScatteringForecastMetricsCallback, MetricsLoggingCallback,
#     MemoryMonitorCallback, ReconstructionPlotCallback
# )

class LightSeqVaeTeb(L.LightningModule):
    """
    PyTorch Lightning module for the SeqVaeTeb model.

    This module handles the training, validation, and optimization loops,
    including learning rate scheduling and KLD beta annealing.
    Supports both standard TEB and beta-TCVAE training modes.
    """

    _FORECAST_METRIC_KEYS = (
        'forecast_mse',
        'forecast_rmse',
        'forecast_nll',
    )

    def __init__(
        self,
        seqvae_teb_model: SeqVaeTeb,
        lr: float = 1e-4,
        lr_milestones: list = None,
        beta_schedule: str = "linear",
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        beta_anneal_epochs: int = 100,
        beta_cycle_len: int = 1000,
        beta_const_val: float = 1.0,
        predictive_horizon: Optional[int] = None,
        predictive_max_anchors: Optional[int] = None,
        scattering_forecast_weight: Optional[float] = None,
        scattering_discount_gamma: float = 1.0,
        log_forecast_metrics: bool = True,
        enable_forecaster: Optional[bool] = None,
    ):
        """Lightning wrapper for SeqVaeTeb with optional scattering forecasting losses."""
        super().__init__()

        model_horizon = getattr(seqvae_teb_model, 'horizon_len', None)
        if predictive_horizon is None:
            predictive_horizon = model_horizon if model_horizon is not None else 1
        if predictive_max_anchors is None:
            predictive_max_anchors = 0
        if scattering_forecast_weight is None:
            scattering_forecast_weight = 0.0

        forecaster_available = getattr(seqvae_teb_model, 'has_forecaster', lambda: False)()
        if enable_forecaster is None:
            enable_forecaster = forecaster_available

        self.save_hyperparameters(ignore=['seqvae_teb_model'])
        self.model = seqvae_teb_model
        self._forecaster_enabled = bool(enable_forecaster and forecaster_available)
        self.hparams.enable_forecaster = self._forecaster_enabled
        if not self._forecaster_enabled:
            self.hparams.scattering_forecast_weight = 0.0
            self.hparams.log_forecast_metrics = False

        self._orig_model = self.model

        if not is_compiled_module(self.model):
            self.model, self._model_compiled = ensure_compiled_module(
                self.model,
                module_name='SeqVaeTeb Lightning wrapper',
            )
            if self._model_compiled and hasattr(self.model, '_orig_mod'):
                self._orig_model = self.model._orig_mod
        else:
            self._model_compiled = True
            if hasattr(self.model, '_orig_mod'):
                self._orig_model = self.model._orig_mod
            logger.info('[LightSeqVaeTeb] Model already compiled, skipping compilation')

    def forward(self, y_st, y_ph, x_ph):
        """Forward pass through the SeqVaeTeb model."""
        return self.model(y_st, y_ph, x_ph)

    def _has_forecaster(self) -> bool:
        return getattr(self.model, "has_forecaster", lambda: True)()

    def _calculate_beta(self):
        """Calculates the KLD weight (beta) based on the current epoch and schedule."""
        schedule = self.hparams.beta_schedule
        epoch = self.current_epoch

        if schedule == 'linear':
            # Linear annealing from beta_start to beta_end
            progress = min(1.0, epoch / self.hparams.beta_anneal_epochs)
            beta = self.hparams.beta_start + (self.hparams.beta_end - self.hparams.beta_start) * progress
        elif schedule == 'cyclic':
            # Cyclic annealing
            cycle_progress = (epoch % self.hparams.beta_cycle_len) / self.hparams.beta_cycle_len
            beta = self.hparams.beta_start + (self.hparams.beta_end - self.hparams.beta_start) * cycle_progress
        elif schedule == 'constant':
            beta = self.hparams.beta_const_val
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

        # Update beta in the underlying model
        return beta

    def on_train_epoch_start(self):
        """Called at the beginning of each training epoch."""
        # Preserve frozen core's eval mode after Lightning's automatic train() call
        if hasattr(self.model, 'is_core_frozen'):
            model = self.model
            # Handle compiled models
            orig_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            if orig_model.is_core_frozen():
                orig_model.core.eval()

        self.hparams.beta = self._calculate_beta()
        self.log('kld_beta', self.hparams.beta, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('hyperparams/beta', self.hparams.beta, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        # Log learning rate at the start of each epoch
        try:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr', lr, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('hyperparams/lr', lr, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        except IndexError:
            # This can happen if the optimizer is not yet configured
            pass

        # Validate hyperparameters are correctly set (first epoch only)
        if self.current_epoch == 0:
            self._validate_hyperparameters()

    def _validate_hyperparameters(self):
        """Validate that hyperparameters are correctly set from config."""
        logger.info(" Validating hyperparameters...")
        logger.info(f"  Current beta_schedule: {self.hparams.beta_schedule}")
        logger.info(f"  Current beta_const_val: {self.hparams.beta_const_val}")
        logger.info(f"  Current beta_start: {self.hparams.beta_start}")
        logger.info(f"  Current beta_end: {self.hparams.beta_end}")
        logger.info(f"  Current lr: {self.hparams.lr}")
        logger.info(f"  Current lr_milestones: {self.hparams.lr_milestones}")
        logger.info(" Hyperparameter validation complete")

    def _compute_losses_and_metrics(self, batch, stage: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Run forward pass, compute losses, and optional forecast metrics using new API."""
        y_st = batch.fhr_st
        y_ph = batch.fhr_ph
        x_ph = batch.fhr_up_ph
        y_raw = batch.fhr

        use_forecaster = self._has_forecaster() and bool(self.hparams.enable_forecaster)

        # Determine loss weights based on configuration
        decoder_weight = 1.0  # Always compute reconstruction during training
        forecaster_weight = float(self.hparams.scattering_forecast_weight) if use_forecaster else 0.0
        kld_weight = self.hparams.beta  # Use current beta value
        gamma = float(getattr(self.hparams, 'scattering_discount_gamma', 1.0))

        # Compute combined loss using new API
        loss_dict = self.model.compute_combined_loss(
            y_st=y_st,
            y_ph=y_ph,
            x_ph=x_ph,
            y_raw=y_raw,
            decoder_weight=decoder_weight,
            forecaster_weight=forecaster_weight,
            kld_weight=kld_weight,
            gamma=gamma,
        )

        # Compute auxiliary metrics if enabled (validation only)
        aux_metrics: Dict[str, torch.Tensor] = {}
        if use_forecaster and self.hparams.log_forecast_metrics and forecaster_weight > 0.0 and stage != 'train':
            with torch.no_grad():
                mu_future_full = loss_dict.get('mu_future')
                logvar_future_full = loss_dict.get('logvar_future')
                timesteps = loss_dict.get('forecast_timesteps')
                if (
                    isinstance(mu_future_full, torch.Tensor)
                    and isinstance(logvar_future_full, torch.Tensor)
                    and isinstance(timesteps, torch.Tensor)
                    and timesteps.numel() > 0
                ):
                    idx = timesteps.long()
                    target_stph = torch.cat([y_st, y_ph], dim=-1)
                    mu_slice = mu_future_full[:, idx, :, :]
                    logvar_slice = logvar_future_full[:, idx, :, :]
                    metrics = self.model.scattering_forecast_metrics(mu_slice, logvar_slice, target_stph, idx)
                    aux_metrics = {f'forecast_{k}': v for k, v in metrics.items()}

        return loss_dict, aux_metrics

    def training_step(self, batch, batch_idx):
        """Defines the training loop with memory optimization."""
        loss_dict, aux_metrics = self._compute_losses_and_metrics(batch, stage="train")
        total_loss = loss_dict['total_loss']

        # Core reconstruction / KL logging
        self.log('train/total_loss', total_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.log('train/recon_loss', loss_dict['reconstruction_loss'], on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('train/mse_loss', loss_dict['mse_loss'], on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('train/nll_loss', loss_dict['nll_loss'], on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('train/kld_loss', loss_dict['kld_loss'], on_epoch=True, prog_bar=True, logger=True, sync_dist=False)

        forecast_loss = _log_metric_tensor(loss_dict.get('forecasting_loss'), reference=total_loss, default=0.0)
        scattering_nll = _log_metric_tensor(loss_dict.get('scattering_nll', loss_dict.get('forecasting_loss')), reference=total_loss, default=0.0)
        scattering_mse = _log_metric_tensor(loss_dict.get('scattering_mse', 0.0), reference=total_loss, default=0.0)
        valid_steps = _log_metric_tensor(loss_dict.get('valid_steps', 0.0), reference=total_loss, default=0.0)

        self.log('train/forecast_loss', forecast_loss, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('train/scattering_nll', scattering_nll, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('train/scattering_mse', scattering_mse, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log('train/valid_steps', valid_steps, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)

        # Auxiliary metrics (if any were computed)
        for name, value in sorted(aux_metrics.items()):
            metric_value = _log_metric_tensor(value, reference=total_loss, default=0.0)
            self.log(f'train/{name}', metric_value, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Defines the validation loop with memory optimization."""
        loss_dict, aux_metrics = self._compute_losses_and_metrics(batch, stage="val")
        total_loss = loss_dict['total_loss']

        # Core reconstruction / KL logging
        self.log('val/total_loss', total_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/recon_loss', loss_dict['reconstruction_loss'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val/mse_loss', loss_dict['mse_loss'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val/nll_loss', loss_dict['nll_loss'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val/kld_loss', loss_dict['kld_loss'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        forecast_loss = _log_metric_tensor(loss_dict.get('forecasting_loss'), reference=total_loss, default=0.0)
        scattering_nll = _log_metric_tensor(loss_dict.get('scattering_nll', loss_dict.get('forecasting_loss')), reference=total_loss, default=0.0)
        valid_steps = _log_metric_tensor(loss_dict.get('valid_steps', 0.0), reference=total_loss, default=0.0)

        self.log('val/forecast_loss', forecast_loss, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val/scattering_nll', scattering_nll, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val/valid_steps', valid_steps, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        for name in self._FORECAST_METRIC_KEYS:
            metric_value = _log_metric_tensor(aux_metrics.get(name), reference=total_loss, default=0.0)
            self.log(f'val/{name}', metric_value, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Minimal cleanup after each training batch - removed frequent cache clearing for multi-GPU."""
        # Only clean up batch references - no cache clearing for better multi-GPU performance
        del batch

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        """Minimal cleanup after each validation batch - removed frequent cache clearing for multi-GPU."""
        # Only clean up batch references - no cache clearing for better multi-GPU performance
        del batch

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers with SOTA optimizations."""
        # Only optimize parameters that require gradients (respects frozen core)
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        # Log parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        frozen_count = total_params - trainable_count

        logger.info("=" * 80)
        logger.info(f"Optimizer configuration:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_count:,} ({100.0 * trainable_count / total_params:.2f}%)")
        logger.info(f"  Frozen parameters: {frozen_count:,} ({100.0 * frozen_count / total_params:.2f}%)")
        logger.info("=" * 80)

        # OPTIMIZATION: Use AdamW with gradient clipping compatibility
        optimizer = torch.optim.AdamW(
            trainable_params,  # Only optimize trainable parameters
            lr=self.hparams.lr,
            weight_decay=1e-4,     # L2 regularization
            eps=1e-8,              # Numerical stability
            betas=(0.9, 0.95),     # SOTA: Slightly higher beta2 for better convergence
            amsgrad=False,         # Standard AdamW
            # foreach=True,          # SOTA: Vectorized optimizer updates (faster)
            maximize=False,
            capturable=False,      # Standard mode for compatibility
            differentiable=False,
            fused=False,           # Disable fused for gradient clipping compatibility
        )

        if self.hparams.lr_milestones:
            # Use simple milestone-based learning rate scheduler
            from torch.optim.lr_scheduler import MultiStepLR
            scheduler = MultiStepLR(
                optimizer,
                milestones=self.hparams.lr_milestones,
                gamma=0.1  # Decay factor at each milestone
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # Epoch-wise for milestone scheduling
                    "frequency": 1,
                },
            }
        return optimizer
