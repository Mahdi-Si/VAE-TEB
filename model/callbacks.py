"""Reusable Lightning callbacks for SeqVAE training utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lightning.pytorch.loggers import MLFlowLogger

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from loguru import logger


class LossPlotCallback(Callback):
    """Collects logged losses and produces simple matplotlib plots."""

    def __init__(self, output_dir: Path | str, plot_frequency: int = 5, max_history_size: int = 1000, *, mlflow_logger: "MLFlowLogger" | None = None) -> None:
        super().__init__()
        self._mlflow_logger = mlflow_logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_frequency = max(1, int(plot_frequency))
        self.max_history_size = max(1, int(max_history_size))
        self._loss_dir = self.output_dir / "loss_plots"
        self._loss_dir.mkdir(parents=True, exist_ok=True)
        self.metric_keys: List[str] = [
            "train/total_loss",
            "train/recon_loss",
            "train/mse_loss",
            "train/nll_loss",
            "train/kld_loss",
            "train/forecast_loss",
            "train/scattering_nll",
            "train/scattering_mse",
            "train/valid_steps",
            "val/total_loss",
            "val/recon_loss",
            "val/mse_loss",
            "val/nll_loss",
            "val/kld_loss",
            "val/forecast_loss",
            "val/scattering_nll",
            "val/scattering_mse",
            "val/valid_steps",
            "val/forecast_mse",
            "val/forecast_rmse",
            "val/forecast_nll",
            "kld_beta",
            "lr",
        ]
        self.history: Dict[str, List[float]] = {"epoch": []}
        self.history.update({key: [] for key in self.metric_keys})

    @staticmethod
    def _to_float(value: torch.Tensor | float | None) -> float:
        if value is None:
            return float("nan")
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                try:
                    return float(value.item())
                except RuntimeError:
                    return float(value.detach().cpu().item())
            return float("nan")
        return float(value)

    def _trim_history(self) -> None:
        if len(self.history["epoch"]) <= self.max_history_size:
            return
        trim = len(self.history["epoch"]) - self.max_history_size
        for key in self.history:
            self.history[key] = self.history[key][trim:]
    def _log_artifact(self, path: Path) -> None:
        if self._mlflow_logger is None:
            return
        try:
            self._mlflow_logger.experiment.log_artifact(self._mlflow_logger.run_id, str(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to log artifact {path} to MLflow: {exc}")


    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        self.history["epoch"].append(epoch)
        for key in self.metric_keys:
            self.history[key].append(self._to_float(metrics.get(key)))
        self._trim_history()
        if not trainer.is_global_zero:
            return
        if (epoch + 1) % self.plot_frequency != 0:
            return
        self._plot_losses()
        self._plot_hyperparameters()

    def _plot_losses(self) -> None:
        epochs = np.array(self.history["epoch"], dtype=float)
        if epochs.size == 0:
            return
        train_keys = [k for k in self.metric_keys if k.startswith("train/")]
        val_keys = [k for k in self.metric_keys if k.startswith("val/")]
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        self._plot_group(axes[0], epochs, train_keys, title="Training metrics")
        self._plot_group(axes[1], epochs, val_keys, title="Validation metrics")
        axes[1].set_xlabel("Epoch")
        fig.tight_layout()
        save_path = self._loss_dir / f"losses_epoch_{int(epochs[-1]):04d}.png"
        fig.savefig(save_path, dpi=150)
        logger.info(f"Saved loss plot to {save_path}")
        plt.close(fig)

    def _plot_group(self, axis: plt.Axes, epochs: np.ndarray, keys: Iterable[str], *, title: str) -> None:
        axis.set_title(title)
        for key in keys:
            values = np.array(self.history[key], dtype=float)
            if np.all(np.isnan(values)):
                continue
            axis.plot(epochs, values, label=key)
        axis.legend(loc="upper right", frameon=False)
        axis.grid(True, alpha=0.3)

    def _plot_hyperparameters(self) -> None:
        epochs = np.array(self.history["epoch"], dtype=float)
        if epochs.size == 0:
            return
        beta_values = np.array(self.history["kld_beta"], dtype=float)
        lr_values = np.array(self.history["lr"], dtype=float)
        if np.all(np.isnan(beta_values)) and np.all(np.isnan(lr_values)):
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        if not np.all(np.isnan(beta_values)):
            axes[0].plot(epochs, beta_values, color="tab:red")
        axes[0].set_title("KLD beta")
        axes[0].set_xlabel("Epoch")
        axes[0].grid(True, alpha=0.3)
        if not np.all(np.isnan(lr_values)):
            axes[1].plot(epochs, lr_values, color="tab:blue")
        axes[1].set_title("Learning rate")
        axes[1].set_xlabel("Epoch")
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        save_path = self._loss_dir / f"hyperparams_epoch_{int(epochs[-1]):04d}.png"
        fig.savefig(save_path, dpi=150)
        logger.info(f"Saved hyperparameter plot to {save_path}")
        plt.close(fig)


class ScatteringForecastMetricsCallback(Callback):
    """
    Logs scattering forecast and reconstruction metrics on validation epochs.

    This callback computes additional diagnostic metrics for the forecasting model:
    - Reconstruction RMSE on validation samples
    - Forecast MSE/RMSE/NLL for future scattering coefficients

    Only runs when forecaster is available and at specified frequency.
    """

    def __init__(self, log_every_n_epochs: int = 1) -> None:
        super().__init__()
        self.log_every_n_epochs = max(1, log_every_n_epochs)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch
        if (epoch % self.log_every_n_epochs) != 0 or not trainer.is_global_zero:
            return

        # Get validation dataloader
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            dataloader = trainer.datamodule.val_dataloader()
            if isinstance(dataloader, list):
                dataloader = dataloader[0]
        else:
            dataloader = trainer.val_dataloaders
            if isinstance(dataloader, list):
                dataloader = dataloader[0]

        try:
            batch = next(iter(dataloader))
        except Exception as exc:
            logger.warning(f"ScatteringForecastMetricsCallback: unable to fetch validation batch ({exc})")
            return

        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
        model = pl_module.model
        orig_model = model._orig_mod if hasattr(model, "_orig_mod") else model

        with torch.no_grad():
            # Compute reconstruction metrics
            recon = orig_model(y_st=batch.fhr_st, y_ph=batch.fhr_ph, x_ph=batch.fhr_up_ph)
            mu_pr = recon.get("mu_pr")
            if mu_pr is not None:
                raw = batch.fhr
                if raw.dim() == 3 and raw.size(-1) == 1:
                    raw = raw.squeeze(-1)
                recon_rmse = torch.sqrt(torch.nn.functional.mse_loss(mu_pr, raw))
                pl_module.log("val/sample_recon_rmse", recon_rmse, prog_bar=False, logger=True, sync_dist=False)

            # Compute forecasting metrics if forecaster available
            if hasattr(orig_model, 'has_forecaster') and orig_model.has_forecaster():
                if hasattr(orig_model, 'forecast_scattering'):
                    forecast = orig_model.forecast_scattering(
                        y_st=batch.fhr_st,
                        y_ph=batch.fhr_ph,
                        x_ph=batch.fhr_up_ph,
                        timesteps=None,
                        use_posterior_mean=True,
                    )
                    mu_future = forecast.get("mu_future")
                    logvar_future = forecast.get("logvar_future")
                    timesteps = forecast.get("timesteps")
                    if mu_future is not None and timesteps is not None:
                        target_stph = torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1)
                        if hasattr(orig_model, 'scattering_forecast_metrics'):
                            metrics = orig_model.scattering_forecast_metrics(mu_future, logvar_future, target_stph, timesteps)
                            for key, value in metrics.items():
                                pl_module.log(f"val/sample_{key}", value, prog_bar=False, logger=True, sync_dist=False)

        # Preserve frozen core eval mode
        if hasattr(model, "is_core_frozen") and model.is_core_frozen():
            orig_model.core.eval()


class MetricsLoggingCallback(Callback):
    """
    Simple callback to collect loss history for both training and validation.

    Useful for post-training analysis and plotting.
    """

    def __init__(self):
        super().__init__()
        self.train_loss_history = []
        self.val_loss_history = []

    def on_train_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        train_loss = logs.get("train/total_loss")
        self.train_loss_history.append(train_loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        val_loss = logs.get("val/total_loss")
        self.val_loss_history.append(val_loss)


class MemoryMonitorCallback(Callback):
    """
    Callback to monitor GPU memory usage and automatically clear cache when needed.
    Optimized for multi-GPU training with reduced monitoring frequency.

    Args:
        threshold_gb: GPU memory threshold in GB above which cache is cleared
        log_frequency: Frequency (in batches) to log memory usage
    """

    def __init__(self, threshold_gb: float = 12.0, log_frequency: int = 200):
        super().__init__()
        self.threshold_gb = threshold_gb
        self.log_frequency = log_frequency
        self.batch_count = 0

    def _log_memory_usage(self, prefix: str = "") -> float:
        """Log current GPU memory usage for all devices."""
        if torch.cuda.is_available():
            total_allocated = 0.0
            device_count = torch.cuda.device_count()
            for device_id in range(device_count):
                allocated = torch.cuda.memory_allocated(device_id) / 1024 ** 3  # GB
                reserved = torch.cuda.memory_reserved(device_id) / 1024 ** 3  # GB
                logger.info(f"{prefix} GPU {device_id}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                total_allocated += allocated
            return total_allocated
        return 0.0

    def _clear_memory_if_needed(self) -> bool:
        """Clear GPU memory on all devices if usage exceeds threshold."""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            cleared_any = False
            for device_id in range(device_count):
                allocated = torch.cuda.memory_allocated(device_id) / 1024 ** 3  # GB
                if allocated > self.threshold_gb:
                    logger.warning(
                        f"GPU {device_id} memory usage ({allocated:.2f}GB) exceeds threshold ({self.threshold_gb}GB). Clearing cache...")
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()
                    cleared_any = True
            return cleared_any
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Monitor memory after each training batch."""
        self.batch_count += 1

        # Log memory usage periodically
        if self.batch_count % self.log_frequency == 0:
            self._log_memory_usage(f"Train batch {batch_idx}")

        # Clear memory if needed
        self._clear_memory_if_needed()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Monitor memory after each validation batch."""
        # Clear memory if needed during validation
        self._clear_memory_if_needed()

    def on_train_epoch_start(self, trainer, pl_module):
        """Log memory at the start of each epoch."""
        self._log_memory_usage(f"Epoch {trainer.current_epoch} start")

    def on_train_epoch_end(self, trainer, pl_module):
        """Log usage at the end of each epoch - reduced cache clearing for multi-GPU."""
        self._log_memory_usage(f"Epoch {trainer.current_epoch} end")
        # Only clear cache at epoch end, not during training for better multi-GPU performance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ReconstructionPlotCallback(Callback):
    """Plots reconstructions from the validation set every few epochs."""

    def __init__(self, output_dir: Path | str, plot_frequency: int = 5, num_examples: int = 3, *, mlflow_logger: "MLFlowLogger" | None = None) -> None:
        super().__init__()
        self._mlflow_logger = mlflow_logger
        self.output_dir = Path(output_dir) / "reconstructions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_frequency = max(1, int(plot_frequency))
        self.num_examples = max(1, int(num_examples))

    def _log_artifact(self, path: Path) -> None:
        if getattr(self, "_mlflow_logger", None) is None:
            return
        try:
            self._mlflow_logger.experiment.log_artifact(self._mlflow_logger.run_id, str(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to log artifact {path} to MLflow: {exc}")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_frequency != 0:
            return
        val_loader = trainer.val_dataloaders
        if isinstance(val_loader, list):
            val_loader = val_loader[0] if val_loader else None
        if val_loader is None:
            logger.warning("ReconstructionPlotCallback could not access validation dataloader")
            return
        try:
            batch = next(iter(val_loader))
        except StopIteration:
            logger.warning("Validation dataloader is empty; skipping reconstruction plot")
            return
        device = pl_module.device
        outputs = None
        try:
            batch = pl_module.transfer_batch_to_device(batch, device, dataloader_idx=0)
            pl_module.eval()
            with torch.no_grad():
                outputs = pl_module.model(batch.fhr_st, batch.fhr_ph, batch.fhr_up_ph)
        finally:
            pl_module.train()
        if outputs is None:
            logger.warning('Failed to produce model outputs for reconstruction plot')
            return
        mu_pr = outputs.get("mu_pr")
        logvar_pr = outputs.get("logvar_pr")
        if mu_pr is None:
            logger.warning("Model output missing 'mu_pr'; skipping reconstruction plot")
            return
        y_raw = batch.fhr.detach().cpu()
        mu_np = mu_pr.detach().cpu()
        std_np = None
        if logvar_pr is not None:
            std_np = torch.exp(0.5 * logvar_pr).detach().cpu()
        count = min(self.num_examples, y_raw.size(0))
        del batch
        fig, axes = plt.subplots(count, 1, figsize=(14, 3 * count), sharex=True)
        if count == 1:
            axes = [axes]
        time_axis = np.arange(y_raw.shape[1])
        for idx in range(count):
            axis = axes[idx]
            axis.plot(time_axis, y_raw[idx].numpy(), label="target", color="tab:blue", linewidth=1.2)
            axis.plot(time_axis, mu_np[idx].numpy(), label="reconstruction", color="tab:orange", linewidth=1.0)
            if std_np is not None:
                mean = mu_np[idx].numpy()
                std = std_np[idx].numpy()
                axis.fill_between(time_axis, mean - 2 * std, mean + 2 * std, color="tab:orange", alpha=0.2)
            axis.set_ylabel(f"Sample {idx}")
            axis.grid(True, alpha=0.3)
        axes[0].set_title("Validation reconstructions")
        axes[-1].set_xlabel("Time index")
        axes[0].legend(loc="upper right", frameon=False)
        fig.tight_layout()
        save_path = self.output_dir / f"reconstruction_epoch_{epoch:04d}.png"
        fig.savefig(save_path, dpi=150)
        logger.info(f"Saved reconstruction plot to {save_path}")
        plt.close(fig)
