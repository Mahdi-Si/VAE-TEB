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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from lightning.pytorch.callbacks import Callback
from loguru import logger


class LossPlotCallback(Callback):
    """Collects logged losses and produces interactive Plotly HTML plots."""

    def __init__(self, output_dir: Path | str, plot_frequency: int = 5, max_history_size: int = 1000, *, mlflow_logger: "MLFlowLogger" | None = None) -> None:
        """
        Args:
            output_dir: Directory where the loss plot HTML files will be saved.
            plot_frequency: Frequency (in epochs) to generate the loss plot.
            max_history_size: Maximum number of epochs to keep in history to prevent memory issues.
            mlflow_logger: Optional MLflow logger for artifact logging.
        """
        super().__init__()
        self._mlflow_logger = mlflow_logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_frequency = max(1, int(plot_frequency))
        self.max_history_size = max(1, int(max_history_size))

        # Standard TEB metrics (cleaned up - removed legacy aliases)
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            # Core VAE losses
            "train/total_loss": [],
            "train/recon_loss": [],
            "train/mse_loss": [],
            "train/nll_loss": [],
            "train/kld_loss": [],
            # Scattering forecaster losses
            "train/scattering_nll": [],
            "train/scattering_mse": [],
            "train/forecast_loss": [],
            "train/forecast_nll": [],
            "train/forecast_mse": [],
            "train/forecast_rmse": [],
            # Auxiliary metrics
            "train/valid_steps": [],
            # Validation core losses
            "val/total_loss": [],
            "val/recon_loss": [],
            "val/mse_loss": [],
            "val/nll_loss": [],
            "val/kld_loss": [],
            # Validation forecaster losses
            "val/scattering_nll": [],
            "val/scattering_mse": [],
            "val/forecast_loss": [],
            "val/forecast_nll": [],
            "val/forecast_mse": [],
            "val/forecast_rmse": [],
            # Validation metrics
            "val/valid_steps": [],
            "val/sample_recon_rmse": [],
            # Hyperparameters
            "hyperparams/beta": [],
            "hyperparams/lr": [],
            # Legacy compatibility
            "kld_beta": [],
            "lr": [],
        }

    @staticmethod
    def _to_float(value: torch.Tensor | float | None) -> float:
        """Convert a tensor or value to float, handling None and multi-element tensors."""
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
        """Trim history to prevent unlimited memory growth."""
        if len(self.history["epoch"]) <= self.max_history_size:
            return
        trim_size = len(self.history["epoch"]) - self.max_history_size
        for key in self.history:
            self.history[key] = self.history[key][trim_size:]

    def _log_artifact(self, path: Path) -> None:
        """Log artifact to MLflow if logger is available."""
        if self._mlflow_logger is None:
            return
        try:
            self._mlflow_logger.experiment.log_artifact(self._mlflow_logger.run_id, str(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to log artifact {path} to MLflow: {exc}")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        """Collect metrics and generate plots at validation epoch end."""
        # Extract the current epoch number
        epoch = trainer.current_epoch

        # Retrieve logged metrics from the trainer
        metrics = trainer.callback_metrics

        # Store losses in history
        self.history["epoch"].append(epoch)
        for key in self.history:
            if key != "epoch":
                self.history[key].append(self._to_float(metrics.get(key)))

        # Trim history to prevent memory issues
        self._trim_history()

        # Check if it's time to plot the losses and only do so on the main process
        if (epoch + 1) % self.plot_frequency == 0 and trainer.is_global_zero:
            self.plot_losses()
            self.plot_hyperparameters()

    def plot_losses(self) -> None:
        """Generate interactive Plotly loss plot with all metrics."""
        import gc

        # Create a Plotly figure and add a trace for each metric
        fig = go.Figure()

        for key, values in self.history.items():
            if key == "epoch" or not any(v is not None and not np.isnan(v) for v in values):
                continue

            fig.add_trace(go.Scatter(
                x=self.history["epoch"],
                y=values,
                mode='lines+markers',
                name=key.replace('/', ' ').title()
            ))

        # Customize layout
        fig.update_layout(
            title="Training and Validation Losses",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            legend_title="Metrics",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Save the figure as an HTML file
        plot_path = self.output_dir / "loss_plot_epoch.html"
        fig.write_html(str(plot_path))
        logger.info(f"Loss plot saved to {plot_path}")

        # Clean up figure to free memory
        del fig
        gc.collect()

    def plot_hyperparameters(self) -> None:
        """Create an interactive Plotly plot of hyperparameter evolution."""
        import gc

        if len(self.history["epoch"]) == 0:
            return

        # Check if we have hyperparameter data (try both new and legacy keys)
        beta_values = self.history.get("hyperparams/beta", [])
        lr_values = self.history.get("hyperparams/lr", [])

        # Fallback to legacy keys if new ones are empty
        if not any(v is not None and not np.isnan(v) for v in beta_values):
            beta_values = self.history.get("kld_beta", [])
        if not any(v is not None and not np.isnan(v) for v in lr_values):
            lr_values = self.history.get("lr", [])

        has_beta = any(v is not None and not np.isnan(v) for v in beta_values)
        has_lr = any(v is not None and not np.isnan(v) for v in lr_values)

        if not (has_beta or has_lr):
            logger.info("No hyperparameter data available for plotting")
            return

        # Create subplots for different hyperparameters
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Beta (KLD Weight)', 'Learning Rate'),
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )

        # Plot Beta if available
        if has_beta:
            fig.add_trace(
                go.Scatter(
                    x=self.history["epoch"],
                    y=beta_values,
                    mode='lines+markers',
                    name='Beta',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )

        # Plot Learning Rate if available
        if has_lr:
            fig.add_trace(
                go.Scatter(
                    x=self.history["epoch"],
                    y=lr_values,
                    mode='lines+markers',
                    name='Learning Rate',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=2
            )

        # Update layout
        fig.update_layout(
            title="Training Hyperparameters Evolution",
            showlegend=False,
            template="plotly_white",
            height=400
        )

        # Update axes labels
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)

        fig.update_yaxes(title_text="Beta Value", row=1, col=1)
        fig.update_yaxes(title_text="Learning Rate", row=1, col=2, type="log")

        # Save the figure
        plot_path = self.output_dir / "hyperparameters_evolution.html"
        fig.write_html(str(plot_path))
        logger.info(f"Hyperparameters plot saved to {plot_path}")

        # Clean up
        del fig
        gc.collect()


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




class ScatteringForecastVisualizationCallback(Callback):
    """Log qualitative plots for scattering forecasts and raw reconstructions."""

    def __init__(self, output_dir: Path | str, plot_every_epoch: int = 5, *, num_examples: int = 2, max_anchors: int = 2) -> None:
        super().__init__()
        self.output_dir = Path(output_dir) / 'forecast_visualizations'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_every_epoch = max(1, int(plot_every_epoch))
        self.num_examples = max(1, int(num_examples))
        self.max_anchors = max(1, int(max_anchors))

    @staticmethod
    def _first_batch(trainer):
        if hasattr(trainer, 'datamodule') and trainer.datamodule is not None:
            dataloader = trainer.datamodule.val_dataloader()
            if isinstance(dataloader, list):
                dataloader = dataloader[0]
        else:
            dataloader = trainer.val_dataloaders
            if isinstance(dataloader, list):
                dataloader = dataloader[0]
        try:
            return next(iter(dataloader))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Forecast viz callback could not fetch batch: {exc}")
            return None

    @staticmethod
    def _select_indices(total: int, count: int) -> list[int]:
        if total <= count:
            return list(range(total))
        step = max(1, total // count)
        return [idx for idx in range(0, total, step)][:count]

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        epoch = trainer.current_epoch
        if not trainer.is_global_zero or (epoch + 1) % self.plot_every_epoch != 0:
            return
        batch = self._first_batch(trainer)
        if batch is None:
            return
        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
        model = pl_module.model
        orig_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        with torch.no_grad():
            forward_out = orig_model(batch.fhr_st, batch.fhr_ph, batch.fhr_up_ph)
        mu_pr = forward_out.get('mu_pr')
        if mu_pr is not None:
            self._plot_raw_recon(epoch, batch.fhr, mu_pr)
        if not getattr(orig_model, 'has_forecaster', lambda: False)():
            return
        with torch.no_grad():
            forecast = orig_model.forecast_scattering(
                y_st=batch.fhr_st,
                y_ph=batch.fhr_ph,
                x_ph=batch.fhr_up_ph,
                timesteps=None,
                use_posterior_mean=True,
            )
        mu_future = forecast.get('mu_future')
        timesteps = forecast.get('timesteps')
        if mu_future is None or timesteps is None or timesteps.numel() == 0:
            logger.warning('Forecast viz callback found no valid scattering predictions.')
            return
        target_stph = torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1)
        self._plot_scattering(epoch, mu_future, timesteps, target_stph)

    def _plot_raw_recon(self, epoch: int, y_raw: torch.Tensor, mu_pr: torch.Tensor) -> None:
        import matplotlib.pyplot as plt
        gt = y_raw.detach().cpu()
        pred = mu_pr.detach().cpu()
        if gt.dim() == 3 and gt.size(-1) == 1:
            gt = gt[..., 0]
        if pred.dim() == 3 and pred.size(-1) == 1:
            pred = pred[..., 0]
        sample_indices = self._select_indices(gt.size(0), self.num_examples)
        fig, axes = plt.subplots(len(sample_indices), 1, figsize=(12, 3 * len(sample_indices)), sharex=True)
        if len(sample_indices) == 1:
            axes = [axes]
        time_axis = torch.arange(gt.size(-1), dtype=torch.float32) / 4.0
        for axis, s_idx in zip(axes, sample_indices):
            axis.plot(time_axis, gt[s_idx].numpy(), label='raw', linewidth=1.2)
            axis.plot(time_axis, pred[s_idx].numpy(), label='recon', linewidth=1.0)
            axis.set_title(f'Raw reconstruction | sample {s_idx}')
            axis.set_ylabel('FHR (bpm)')
            axis.legend(loc='upper right', frameon=False)
            axis.grid(alpha=0.3)
        axes[-1].set_xlabel('Time (s)')
        fig.tight_layout()
        save_path = self.output_dir / f'reconstruction_epoch_{epoch:04d}.png'
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info(f'Saved raw reconstruction plot to {save_path}')

    def _plot_scattering(self, epoch: int, mu_future: torch.Tensor, timesteps: torch.Tensor, target_stph: torch.Tensor) -> None:
        import matplotlib.pyplot as plt
        B, _, horizon, _ = mu_future.shape
        sample_indices = self._select_indices(B, self.num_examples)
        anchor_indices = self._select_indices(timesteps.numel(), self.max_anchors)
        for sample_idx in sample_indices:
            fig, axes = plt.subplots(len(anchor_indices), 2, figsize=(12, 4 * len(anchor_indices)))
            if len(anchor_indices) == 1:
                axes = [axes]
            for row_axes, anchor_pos in zip(axes, anchor_indices):
                anchor = int(timesteps[anchor_pos].item())
                pred_slice = mu_future[sample_idx, anchor_pos].detach().cpu()
                slices = []
                for offset in range(pred_slice.size(0)):
                    step_idx = anchor + 1 + offset
                    if step_idx >= target_stph.size(1):
                        break
                    slices.append(target_stph[sample_idx, step_idx].detach().cpu())
                if not slices:
                    continue
                target_slice = torch.stack(slices, dim=0)
                self._imshow(row_axes[0], target_slice, f'Target | anchor {anchor}')
                self._imshow(row_axes[1], pred_slice, f'Predicted | anchor {anchor}')
            fig.suptitle(f'Scattering forecast | sample {sample_idx}')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = self.output_dir / f'scattering_epoch_{epoch:04d}_sample_{sample_idx}.png'
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            logger.info(f'Saved scattering forecast plot to {save_path}')

    @staticmethod
    def _imshow(axis, data: torch.Tensor, title: str) -> None:
        axis.imshow(data.numpy().T, aspect='auto', origin='lower', cmap='viridis')
        axis.set_title(title)
        axis.set_ylabel('Channel')
        axis.set_xlabel('Horizon step')

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


class PlottingCallBack(Callback):
    """
    Comprehensive plotting callback for VAE-TEB model validation.

    Generates detailed visualizations of:
    - Reconstruction quality (FHR/UP signals, latents)
    - Latent trajectories and statistics
    - Forecast results (when forecaster is enabled)
    - Batch-aggregated forecasts

    All plots use professional scientific styling with high-quality PDFs.
    """

    def __init__(self, output_dir: Path | str, plot_every_epoch: int = 5, predictive_horizon: Optional[int] = None) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_every_epoch = plot_every_epoch
        self.predictive_horizon = predictive_horizon

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        if trainer.current_epoch % self.plot_every_epoch != 0 or not trainer.is_global_zero:
            return

        logger.info(f"Starting comprehensive plotting callback for epoch {trainer.current_epoch}")

        # Close all existing matplotlib figures before starting
        plt.close('all')

        try:
            # Get validation dataloader
            if hasattr(trainer, 'datamodule') and trainer.datamodule is not None:
                val_dataloader = trainer.datamodule.val_dataloader()
            else:
                val_dataloader = trainer.val_dataloaders
                if isinstance(val_dataloader, list):
                    val_dataloader = val_dataloader[0]

            batch = next(iter(val_dataloader))
            logger.info("Successfully fetched batch from validation dataloader")
        except (StopIteration, AttributeError, IndexError) as e:
            logger.warning(f"Could not get a batch from validation dataloader for plotting: {e}")
            return

        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)

        pl_module.eval()
        try:
            with torch.no_grad():
                logger.info("Accessing batch data...")
                y_st, y_ph, x_ph = batch.fhr_st, batch.fhr_ph, batch.fhr_up_ph
                y_raw_normalized = batch.fhr
                up_raw_normalized = batch.up

                model_outputs = pl_module.model(y_st, y_ph, x_ph)
                mu_pr = model_outputs.get("mu_pr")
                logvar_pr = model_outputs.get("logvar_pr")
                latent_z_full = model_outputs.get("z")
                mu_post_full = model_outputs.get("mu_post")
                mu_prior_full = model_outputs.get("mu_prior")

                self._plot_reconstruction_overview(
                    y_raw_normalized=y_raw_normalized,
                    up_raw_normalized=up_raw_normalized,
                    mu_pr=mu_pr,
                    logvar_pr=logvar_pr,
                    latent_z=latent_z_full,
                    mu_post=mu_post_full,
                    mu_prior=mu_prior_full,
                    epoch=trainer.current_epoch,
                )

                # Get uncompiled model for utility methods
                model = pl_module.model
                orig_model = model._orig_mod if hasattr(model, '_orig_mod') else model

                # Check if model has forecaster
                has_forecaster = False
                if hasattr(orig_model, 'has_forecaster'):
                    has_forecaster = orig_model.has_forecaster()

                if has_forecaster:
                    # Generate forecasts using the forecast() method (returns latent predictions)
                    try:
                        forecast_out = orig_model.forecast(
                            y_st, y_ph, x_ph, anchors=None, use_posterior_mean=True
                        )

                        anchors = forecast_out.get("anchors")
                        mu_future = forecast_out.get("mu_future")          # Predicted scattering coefficients (B,N,480)
                        logvar_future = forecast_out.get("logvar_future")  # Uncertainty in scattering predictions
                        z_future = forecast_out.get("z_future")            # Predicted latent trajectories
                        latent_logvar_future = forecast_out.get("latent_logvar_future")
                        enc = forecast_out.get("enc")

                        # Aggregate forecasts to raw signal canvas
                        canvas_mu, mean_mu = orig_model.aggregate_forecasts_to_canvas(
                            mu_future, anchors, total_len=y_raw_normalized.shape[1], stride=orig_model.decimation_factor
                        )

                        var_future = logvar_future.exp()
                        _, mean_var = orig_model.aggregate_forecasts_to_canvas(
                            var_future, anchors, total_len=y_raw_normalized.shape[1], stride=orig_model.decimation_factor
                        )
                        std_mu = mean_var.clamp_min(1e-8).sqrt()

                        # Plot latent forecast diagnostics (prediction accuracy in scattering domain)
                        self._plot_latent_forecast_samples(
                            mu_post_sequence=enc.get("mu_post"),
                            z_future=z_future,
                            latent_logvar_future=latent_logvar_future,
                            anchors=anchors,
                            epoch=trainer.current_epoch,
                        )

                        self._plot_channel_forecasts(
                            mu_post_sequence=enc.get("mu_post"),
                            z_future=z_future,
                            latent_logvar_future=latent_logvar_future,
                            anchors=anchors,
                            epoch=trainer.current_epoch,
                        )

                        self._plot_latent_trajectory_analysis(
                            mu_post_sequence=enc.get("mu_post"),
                            mu_prior_sequence=enc.get("mu_prior"),
                            epoch=trainer.current_epoch,
                        )

                        # Plot reconstructed raw signal from predictions
                        self._plot_forecast_results_scattering(
                            y_raw_normalized,
                            mean_mu,
                            std_mu,
                            canvas_mu,
                            anchors,
                            enc.get('mu_post'),
                            trainer.current_epoch
                        )

                        try:
                            self._plot_batch_aggregated_forecast(
                                y_raw_batch=y_raw_normalized,
                                mean_mu_batch=mean_mu,
                                std_mu_batch=std_mu,
                                epoch=trainer.current_epoch,
                            )
                        except Exception as e:
                            logger.warning(f"Failed to plot batch aggregated forecast: {e}")

                    except Exception as e:
                        logger.warning(f"Failed to generate forecasts for plotting: {e}")

                # Always plot latent statistics
                self._plot_latent_statistics(
                    mu_prior=mu_prior_full,
                    mu_post=mu_post_full,
                    epoch=trainer.current_epoch,
                )

        except Exception as e:
            logger.error(f"Error during plotting: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            # Ensure all figures are closed to prevent threading issues
            plt.close('all')
            import gc
            gc.collect()

            # Restore training mode
            pl_module.train()

            # Preserve frozen core's eval mode if applicable
            model = pl_module.model
            orig_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            if hasattr(orig_model, 'is_core_frozen') and orig_model.is_core_frozen():
                orig_model.core.eval()
                logger.debug("Preserved frozen core eval mode after plotting callback")

    def _plot_reconstruction_overview(
        self,
        y_raw_normalized: torch.Tensor,
        up_raw_normalized: Optional[torch.Tensor],
        mu_pr: Optional[torch.Tensor],
        logvar_pr: Optional[torch.Tensor],
        latent_z: Optional[torch.Tensor],
        mu_post: Optional[torch.Tensor],
        mu_prior: Optional[torch.Tensor],
        epoch: int,
    ):
        """Plot ground-truth vs reconstruction along with latent diagnostics."""
        import gc

        if y_raw_normalized is None or mu_pr is None or logvar_pr is None:
            return

        batch_idx = 0
        try:
            y_raw = y_raw_normalized[batch_idx].detach().cpu().numpy()
            recon = mu_pr[batch_idx].detach().cpu().numpy()
            logvar = logvar_pr[batch_idx].detach().cpu().numpy()
        except Exception:
            return

        std = np.exp(0.5 * logvar)
        diff = y_raw - recon

        up_raw = None
        if up_raw_normalized is not None:
            try:
                up_raw = up_raw_normalized[batch_idx].detach().cpu().numpy()
            except Exception:
                up_raw = None

        latent_matrix = None
        if latent_z is not None:
            try:
                latent_matrix = latent_z[batch_idx].detach().cpu().numpy()
            except Exception:
                latent_matrix = None
        if latent_matrix is None and mu_post is not None:
            try:
                latent_matrix = mu_post[batch_idx].detach().cpu().numpy()
            except Exception:
                latent_matrix = None

        prior_matrix = None
        if mu_prior is not None:
            try:
                prior_matrix = mu_prior[batch_idx].detach().cpu().numpy()
            except Exception:
                prior_matrix = None

        Fs = 4.0
        t = np.arange(y_raw.shape[0]) / Fs

        corr = float('nan')
        if np.std(y_raw) > 1e-8 and np.std(recon) > 1e-8:
            try:
                corr = float(np.corrcoef(y_raw, recon)[0, 1])
            except Exception:
                corr = float('nan')

        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))

        colors = {
            'fhr': "#055C9A",
            'up': "#0DD8A2",
            'gt': '#456882',
            'recon': '#BB3E00',
            'uncertainty': '#F7AD45',
            'background': '#F9F3EF'
        }

        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'axes.linewidth': 0.7,
            'axes.edgecolor': "#9E9D9D",
            'axes.facecolor': colors['background'],
            'grid.color': "#838383",
            'grid.linewidth': 0.4,
            'grid.alpha': 0.6,
            'legend.frameon': True,
            'legend.fancybox': False,
            'legend.shadow': False,
            'legend.framealpha': 0.95,
            'legend.edgecolor': '#A2B9A7',
            'legend.facecolor': colors['background'],
            'figure.facecolor': 'white',
            'savefig.facecolor': 'white',
            'savefig.dpi': 300
        })

        n_rows = 4
        fig, ax = plt.subplots(
            nrows=n_rows, ncols=2, figsize=(20, n_rows * 3.2),
            gridspec_kw={"width_ratios": [80, 1]}, constrained_layout=True)

        for i in range(n_rows):
            ax[i, 0].grid(True, linestyle='-', alpha=0.35, linewidth=0.4, color='#D2C1B6')
            ax[i, 0].grid(True, which='minor', linestyle=':', alpha=0.25, linewidth=0.3, color='#D2C1B6')
            ax[i, 0].minorticks_on()
            ax[i, 0].set_axisbelow(True)
            ax[i, 0].spines['top'].set_visible(False)
            ax[i, 0].spines['right'].set_visible(False)
            ax[i, 0].spines['left'].set_color('#A2B9A7')
            ax[i, 0].spines['bottom'].set_color('#A2B9A7')
            ax[i, 0].spines['left'].set_linewidth(0.7)
            ax[i, 0].spines['bottom'].set_linewidth(0.7)

        ax[0, 1].set_axis_off()
        ax[0, 0].plot(t, y_raw, linewidth=1.2, color=colors['fhr'], label='FHR', alpha=0.9)
        if up_raw is not None:
            ax[0, 0].plot(t, up_raw, linewidth=1.0, color=colors['up'], label='UP', alpha=0.75)
        ax[0, 0].set_ylabel('Amplitude')
        ax[0, 0].set_title('Raw FHR/UP Signals')
        ax[0, 0].autoscale(enable=True, axis='x', tight=True)
        ax[0, 0].legend(loc='upper right', framealpha=0.95)

        ax[1, 1].set_axis_off()
        ax[1, 0].plot(t, y_raw, linewidth=1.5, color=colors['gt'], label='Ground Truth', alpha=0.9, zorder=3)
        ax[1, 0].plot(t, recon, linewidth=1.3, color=colors['recon'], label='Reconstruction', alpha=0.85, zorder=2)
        ax[1, 0].fill_between(t, recon - std, recon + std, color=colors['uncertainty'], alpha=0.25, label='+/- 1 sigma')
        ax[1, 0].set_ylabel('FHR (bpm)')
        ax[1, 0].set_title('Ground Truth vs Reconstruction')
        ax[1, 0].legend(loc='upper right', framealpha=0.95)
        ax[1, 0].autoscale(enable=True, axis='x', tight=True)
        ax[1, 0].text(
            0.01,
            0.92,
            f"RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nCorr: {corr:.3f}",
            transform=ax[1, 0].transAxes,
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

        ax[2, 0].set_axis_off()
        ax[2, 1].set_axis_off()

        if prior_matrix is not None and latent_matrix is not None and ax.shape[1] > 1:
            ax[2, 1].set_axis_on()
            step_idx = np.arange(latent_matrix.shape[0])
            prior_norm = np.linalg.norm(prior_matrix, axis=1)
            post_norm = np.linalg.norm(latent_matrix, axis=1)
            delta_norm = np.linalg.norm(latent_matrix - prior_matrix, axis=1)
            ax[2, 1].plot(step_idx, prior_norm, color='#7F8C8D', linewidth=1.0, label='||mu_prior||')
            ax[2, 1].plot(step_idx, post_norm, color='#BB3E00', linewidth=1.2, label='||mu_post||')
            ax[2, 1].plot(step_idx, delta_norm, color='#2E86AB', linewidth=1.2, linestyle='--', label='||delta||')
            ax[2, 1].set_title('Latent Norm Dynamics')
            ax[2, 1].set_xlabel('Decimated step')
            ax[2, 1].grid(True, alpha=0.3)
            ax[2, 1].legend(loc='upper right', fontsize=9, framealpha=0.9)

        ax[3, 1].set_axis_off()
        if latent_matrix is not None:
            img = ax[3, 0].imshow(latent_matrix.T, aspect='auto', cmap='RdBu_r', origin='lower')
            ax[3, 0].set_ylabel('Latent dim')
            ax[3, 0].set_xlabel('Decimated step')
            title_suffix = ''
            if prior_matrix is not None:
                mean_delta = float(np.mean(np.abs(latent_matrix - prior_matrix)))
                title_suffix = f' | mean |delta| = {mean_delta:.3f}'
            ax[3, 0].set_title(f'Posterior Latent Trajectory{title_suffix}')
            cbar = fig.colorbar(img, cax=ax[3, 1])
            cbar.ax.tick_params(labelsize=9, colors='#666666')
            cbar.set_label('Activation', fontsize=10, color='#666666')
        else:
            ax[3, 0].text(0.5, 0.5, 'Latents not available', ha='center', va='center', fontsize=12)
            ax[3, 0].set_axis_off()

        fig.suptitle(f'Reconstruction Overview - Epoch {epoch}', fontsize=14, color='#456882')
        save_path = self.output_dir / f'reconstruction_overview_epoch_{epoch}.pdf'
        plt.savefig(str(save_path), bbox_inches='tight', dpi=300)
        plt.close(fig)
        gc.collect()
        logger.info(f"Reconstruction plot saved to {save_path}")

    def _plot_latent_statistics(
        self,
        mu_prior: Optional[torch.Tensor],
        mu_post: Optional[torch.Tensor],
        epoch: int,
    ):
        """Plot summary statistics for latent trajectories (prior vs posterior)."""
        import gc

        if mu_post is None:
            return

        batch_idx = 0
        try:
            mu_post_np = mu_post[batch_idx].detach().cpu().numpy()
        except Exception:
            return

        if mu_post_np.ndim != 2:
            return

        if mu_prior is not None:
            try:
                mu_prior_np = mu_prior[batch_idx].detach().cpu().numpy()
            except Exception:
                mu_prior_np = np.zeros_like(mu_post_np)
        else:
            mu_prior_np = np.zeros_like(mu_post_np)

        delta = mu_post_np - mu_prior_np
        prior_norm = np.linalg.norm(mu_prior_np, axis=1)
        post_norm = np.linalg.norm(mu_post_np, axis=1)
        delta_norm = np.linalg.norm(delta, axis=1)
        steps = np.arange(mu_post_np.shape[0])

        with np.errstate(all='ignore'):
            corr = np.corrcoef(mu_post_np.T)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

        delta_heatmap = delta.T
        energy_per_dim = np.sum(delta ** 2, axis=0)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

        axes[0, 0].plot(steps, prior_norm, color='#7F8C8D', linewidth=1.0, label='||mu_prior||')
        axes[0, 0].plot(steps, post_norm, color='#BB3E00', linewidth=1.2, label='||mu_post||')
        axes[0, 0].plot(steps, delta_norm, color='#2E86AB', linewidth=1.2, linestyle='--', label='||delta||')
        axes[0, 0].set_title('Latent Norms over Time')
        axes[0, 0].set_xlabel('Decimated step')
        axes[0, 0].set_ylabel('Norm')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(loc='upper right', framealpha=0.9)

        im_corr = axes[0, 1].imshow(corr, aspect='auto', origin='lower', cmap='Spectral', vmin=-1.0, vmax=1.0)
        axes[0, 1].set_title('Posterior Latent Correlation')
        axes[0, 1].set_xlabel('Latent dim')
        axes[0, 1].set_ylabel('Latent dim')
        fig.colorbar(im_corr, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im_delta = axes[1, 0].imshow(delta_heatmap, aspect='auto', origin='lower', cmap='RdBu_r')
        axes[1, 0].set_title('delta = mu_post - mu_prior')
        axes[1, 0].set_xlabel('Decimated step')
        axes[1, 0].set_ylabel('Latent dim')
        fig.colorbar(im_delta, ax=axes[1, 0], fraction=0.046, pad=0.04)

        axes[1, 1].bar(np.arange(delta_heatmap.shape[0]), energy_per_dim, color='#3C6E71')
        axes[1, 1].set_title('Energy per Latent Dimension')
        axes[1, 1].set_xlabel('Latent dim')
        axes[1, 1].set_ylabel('Energy')
        axes[1, 1].grid(True, axis='y', alpha=0.3)

        fig.suptitle(f'Latent Statistics - Epoch {epoch}', fontsize=14, color='#456882')
        save_path = self.output_dir / f'latent_statistics_epoch_{epoch}.pdf'
        plt.savefig(str(save_path), bbox_inches='tight', dpi=300)
        plt.close(fig)
        gc.collect()
        logger.info(f"Latent statistics plot saved to {save_path}")

    def _plot_forecast_results_scattering(
        self,
        y_raw_normalized: torch.Tensor,
        mean_mu: torch.Tensor,
        std_mu: torch.Tensor,
        canvas_mu: torch.Tensor,
        timesteps: torch.Tensor,
        latent_z: torch.Tensor,
        epoch: int,
    ):
        """Plot forecast results: ground truth vs aggregated forecast with uncertainty."""
        import gc

        batch_idx = 0

        y_raw = y_raw_normalized[batch_idx].cpu().numpy()
        pred_mean = mean_mu[batch_idx].cpu().numpy()
        pred_std = std_mu[batch_idx].cpu().numpy()
        z_latent = None
        if latent_z is not None:
            z_latent = latent_z[batch_idx].detach().cpu().numpy()

        mask = ~np.isnan(pred_mean)
        coverage = float(np.mean(mask)) if mask.size > 0 else float('nan')

        if np.any(mask):
            gt_masked = y_raw.copy()
            pred_masked = pred_mean.copy()
            gt_masked[~mask] = np.nan
            pred_masked[~mask] = np.nan
            sample_mse = float(np.nanmean((pred_masked - gt_masked) ** 2))
            sample_mae = float(np.nanmean(np.abs(pred_masked - gt_masked)))
            if np.sum(mask) > 2:
                sample_corr = float(np.corrcoef(gt_masked[mask], pred_masked[mask])[0, 1])
            else:
                sample_corr = float('nan')
        else:
            sample_mse = float('nan')
            sample_mae = float('nan')
            sample_corr = float('nan')

        Fs = 4
        N = len(y_raw)
        t_in = np.arange(0, N) / Fs

        colors = {
            'gt': '#456882',
            'recon': '#BB3E00',
            'uncertainty': '#F7AD45',
            'background': '#F9F3EF'
        }

        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'font.size': 11,
            'axes.facecolor': colors['background'],
            'figure.facecolor': 'white',
            'savefig.dpi': 300
        })

        n_rows = 2
        fig, ax = plt.subplots(
            nrows=n_rows, ncols=2, figsize=(20, n_rows * 3.5),
            gridspec_kw={"width_ratios": [80, 1]}, constrained_layout=True)

        for i in range(n_rows):
            ax[i, 0].grid(True, alpha=0.4)
            ax[i, 0].spines['top'].set_visible(False)
            ax[i, 0].spines['right'].set_visible(False)

        ax[0, 1].set_axis_off()
        ax[0, 0].plot(t_in, y_raw, linewidth=1.5, color=colors['gt'], label='Ground Truth', alpha=0.9)
        ax[0, 0].plot(t_in, pred_mean, linewidth=1.2, color='#BB3E00', label='Forecast (mean)', alpha=0.9)
        ax[0, 0].fill_between(
            t_in,
            pred_mean - pred_std,
            pred_mean + pred_std,
            alpha=0.3,
            color=colors['uncertainty'],
            label='Uncertainty band',
        )
        ax[0, 0].set_ylabel('FHR (bpm)')
        ax[0, 0].set_title('Raw FHR vs Aggregated Forecast')
        ax[0, 0].legend(loc='upper right', framealpha=0.95)
        ax[0, 0].autoscale(enable=True, axis='x', tight=True)

        if z_latent is not None:
            imgplot = ax[1, 0].imshow(z_latent.T, aspect='auto', cmap='bwr', origin='lower')
            ax[1, 1].set_axis_on()
            cbar = fig.colorbar(imgplot, cax=ax[1, 1])
            cbar.ax.tick_params(labelsize=10, colors='#666666')
            cbar.set_label('Activation', fontsize=11, color='#666666')
            ax[1, 0].set_ylabel('Latent Dimensions')
            ax[1, 0].set_xlabel('Decimated steps')
            ax[1, 0].set_title('Latent mu_post (T x D)')
        else:
            ax[1, 0].text(0.5, 0.5, 'Latents not available', ha='center', va='center')
            ax[1, 1].set_axis_off()

        diag_str = f"MSE={sample_mse:.4f} | MAE={sample_mae:.4f} | Corr={sample_corr:.3f} | Cov={coverage:.2%}"
        fig.suptitle(
            f'Forecasting Results - Epoch {epoch}\n{diag_str}',
            fontsize=14,
            color='#456882'
        )

        save_path = self.output_dir / f'forecast_results_epoch_{epoch}.pdf'
        plt.savefig(str(save_path), bbox_inches='tight', dpi=300)
        plt.close(fig)
        gc.collect()
        logger.info(f"Forecast results plot saved to {save_path}")

    def _plot_batch_aggregated_forecast(
        self,
        y_raw_batch: torch.Tensor,
        mean_mu_batch: torch.Tensor,
        std_mu_batch: torch.Tensor,
        epoch: int,
    ):
        """Plot average aggregated forecast across validation batch with uncertainty band."""
        import gc

        mask = ~torch.isnan(mean_mu_batch)
        pred_mean = torch.nanmean(mean_mu_batch, dim=0)
        gt_masked = y_raw_batch.masked_fill(~mask, float('nan'))
        gt_mean = torch.nanmean(gt_masked, dim=0)

        e_var = torch.nanmean(std_mu_batch.pow(2), dim=0)
        mean_centered = mean_mu_batch - pred_mean.unsqueeze(0)
        mean_centered = mean_centered.masked_fill(~mask, float('nan'))
        var_e = torch.nanmean(mean_centered.pow(2), dim=0)
        total_var = (e_var + var_e).clamp_min(0.0)
        pred_std = total_var.sqrt()

        t = np.arange(pred_mean.shape[0]) / 4.0
        pm = pred_mean.detach().cpu().numpy()
        ps = pred_std.detach().cpu().numpy()
        gm = gt_mean.detach().cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(16, 4.5), constrained_layout=True)
        ax.plot(t, gm, color='#2E86AB', label='GT (batch mean)', linewidth=1.5)
        ax.plot(t, pm, color='#BB3E00', label='Forecast (batch mean)', linewidth=1.2)
        ax.fill_between(
            t,
            pm - ps,
            pm + ps,
            color='#F5B7B1',
            alpha=0.4,
            label='Total uncertainty band',
        )
        ax.set_title('Batch-Aggregated Forecast vs Ground Truth')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('FHR (bpm)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        save_path = self.output_dir / f'forecast_results_epoch_{epoch}_avg.pdf'
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        plt.close(fig)
        gc.collect()
        logger.info(f"Batch-aggregated forecast plot saved to {save_path}")
    def _plot_latent_forecast_samples(
        self,
        mu_post_sequence: Optional[torch.Tensor],
        z_future: Optional[torch.Tensor],
        latent_logvar_future: Optional[torch.Tensor],
        anchors: Optional[torch.Tensor],
        epoch: int,
    ):
        """Visualize forecasted latent trajectories against ground truth for spaced anchors."""
        import gc

        if mu_post_sequence is None or z_future is None or anchors is None or anchors.numel() == 0:
            return

        batch_idx = 0
        try:
            mu_post_np = mu_post_sequence[batch_idx].detach().cpu().numpy()
            z_future_np = z_future[batch_idx].detach().cpu().numpy()
            anchors_np = anchors.detach().cpu().numpy().astype(int)
            latent_std_np = None
            if latent_logvar_future is not None:
                latent_std_np = np.sqrt(
                    np.exp(latent_logvar_future[batch_idx].detach().cpu().numpy())
                )
        except Exception:
            return

        if mu_post_np.ndim != 2 or z_future_np.ndim != 3:
            return

        horizon = z_future_np.shape[1]
        step = 30
        selected = []
        last_anchor = -step
        for idx, anchor in enumerate(anchors_np):
            if anchor - last_anchor >= step and anchor + 1 + horizon <= mu_post_np.shape[0]:
                selected.append((idx, anchor))
                last_anchor = anchor
            if len(selected) >= 4:
                break

        if not selected:
            for idx, anchor in enumerate(anchors_np[:4]):
                if anchor + 1 + horizon <= mu_post_np.shape[0]:
                    selected.append((idx, anchor))

        if not selected:
            return

        global_max = 1e-6
        valid_segments = []
        for idx, anchor in selected:
            start = anchor + 1
            end = start + horizon
            gt = mu_post_np[start:end]
            if gt.shape[0] != horizon:
                continue
            pred = z_future_np[idx]
            global_max = max(global_max, np.abs(gt).max(), np.abs(pred).max())
            std_seg = latent_std_np[idx] if latent_std_np is not None else None
            valid_segments.append((idx, anchor, gt, pred, std_seg))

        if not valid_segments:
            return

        n_rows = len(valid_segments)
        fig, axes = plt.subplots(
            n_rows, 4, figsize=(22, n_rows * 3.2),
            gridspec_kw={'width_ratios': [1, 1, 1, 1.3]},
            constrained_layout=True,
        )
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for row, (idx, anchor, gt, pred, std_segment) in enumerate(valid_segments):
            err = pred - gt

            im0 = axes[row, 0].imshow(gt.T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-global_max, vmax=global_max)
            axes[row, 0].set_title(f'Ground truth mu_post | anchor={anchor}')
            axes[row, 0].set_ylabel('Latent dim')
            axes[row, 0].set_xlabel('Forecast step')

            im1 = axes[row, 1].imshow(pred.T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-global_max, vmax=global_max)
            axes[row, 1].set_title('Forecast mu_future')
            axes[row, 1].set_xlabel('Forecast step')

            abs_err = np.abs(err)
            im2 = axes[row, 2].imshow(abs_err.T, aspect='auto', origin='lower', cmap='magma')
            axes[row, 2].set_title('Absolute error')
            axes[row, 2].set_xlabel('Forecast step')

            for c in range(3):
                axes[row, c].grid(False)

            if abs_err.size > 0:
                per_channel_energy = abs_err.sum(axis=0)
                best_channel = int(np.argmax(per_channel_energy))
            else:
                best_channel = 0
            time_axis = np.arange(horizon)
            axes[row, 3].plot(time_axis, gt[:, best_channel], color='#2E86AB', linewidth=1.4, label='GT')
            axes[row, 3].plot(time_axis, pred[:, best_channel], color='#BB3E00', linewidth=1.2, linestyle='--', label='Forecast')
            if std_segment is not None and best_channel < std_segment.shape[1]:
                std_vec = std_segment[:, best_channel]
                upper = pred[:, best_channel] + 1.96 * std_vec
                lower = pred[:, best_channel] - 1.96 * std_vec
                axes[row, 3].fill_between(
                    time_axis,
                    lower,
                    upper,
                    color='#BB3E00',
                    alpha=0.18,
                    label='Forecast +/- 1.96 std' if row == 0 else None,
                )
            axes[row, 3].fill_between(
                time_axis,
                gt[:, best_channel],
                pred[:, best_channel],
                color='#F5B7B1',
                alpha=0.3,
            )
            axes[row, 3].set_title(f'Latent dim {best_channel} trajectory')
            axes[row, 3].set_xlabel('Forecast step')
            axes[row, 3].set_ylabel('Activation')
            axes[row, 3].grid(True, alpha=0.3)
            axes[row, 3].legend(loc='upper right', fontsize=8, framealpha=0.85)

            if row == 0:
                fig.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)
                fig.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

        save_path = self.output_dir / f'latent_forecast_samples_epoch_{epoch}.pdf'
        fig.suptitle(f'Latent Forecast Diagnostics - Epoch {epoch}', fontsize=14, color='#456882')
        plt.savefig(str(save_path), bbox_inches='tight', dpi=300)
        plt.close(fig)
        gc.collect()
        logger.info(f"Latent forecast comparison saved to {save_path}")

    def _plot_channel_forecasts(
        self,
        mu_post_sequence: Optional[torch.Tensor],
        z_future: Optional[torch.Tensor],
        latent_logvar_future: Optional[torch.Tensor],
        anchors: Optional[torch.Tensor],
        epoch: int,
    ) -> None:
        """Plot per-dimension latent trajectories for specific anchors (75 and 224)."""
        import gc

        if mu_post_sequence is None or z_future is None or anchors is None or anchors.numel() == 0:
            return

        batch_idx = 0
        try:
            mu_post_np = mu_post_sequence[batch_idx].detach().cpu().numpy()
            z_future_np = z_future[batch_idx].detach().cpu().numpy()
            anchors_np = anchors.detach().cpu().numpy().astype(int)
            std_future_np = None
            if latent_logvar_future is not None:
                std_future_np = np.sqrt(
                    np.exp(latent_logvar_future[batch_idx].detach().cpu().numpy())
                )
        except Exception:
            return

        if mu_post_np.ndim != 2 or z_future_np.ndim != 3:
            return

        horizon = z_future_np.shape[1]
        latent_dim = z_future_np.shape[2]
        desired_anchors = [75, 224]
        anchor_pairs = []
        for anchor_val in desired_anchors:
            if anchor_val in anchors_np:
                idx_anchor = int(np.where(anchors_np == anchor_val)[0][0])
                start = anchor_val + 1
                end = start + horizon
                if end <= mu_post_np.shape[0]:
                    std_slice = std_future_np[idx_anchor] if std_future_np is not None else None
                    anchor_pairs.append((anchor_val, mu_post_np[start:end], z_future_np[idx_anchor], std_slice))

        if len(anchor_pairs) != len(desired_anchors):
            return

        fig, axes = plt.subplots(
            latent_dim,
            len(anchor_pairs),
            figsize=(len(anchor_pairs) * 5.5, latent_dim * 1.6),
            sharex=True,
            constrained_layout=True,
        )
        if latent_dim == 1:
            axes = axes.reshape(1, -1)

        time_axis = np.arange(horizon)
        colors = {
            'gt': '#2E86AB',
            'pred': '#BB3E00',
        }

        for col, (anchor_val, gt, pred, std_block) in enumerate(anchor_pairs):
            for row in range(latent_dim):
                ax = axes[row, col]
                ax.plot(time_axis, gt[:, row], color=colors['gt'], linewidth=1.2, label='GT')
                ax.plot(time_axis, pred[:, row], color=colors['pred'], linewidth=1.0, linestyle='--', label='Forecast')
                if std_block is not None:
                    std_vec = std_block[:, row]
                    upper = pred[:, row] + 1.96 * std_vec
                    lower = pred[:, row] - 1.96 * std_vec
                    ax.fill_between(
                        time_axis,
                        lower,
                        upper,
                        color=colors['pred'],
                        alpha=0.18,
                        label='Forecast +/- 1.96 std' if (row == 0 and col == 0) else None,
                    )
                ax.grid(True, alpha=0.3)
                if row == 0:
                    ax.set_title(f'Anchor {anchor_val}')
                if col == 0:
                    ax.set_ylabel(f'Latent {row}')
                if row == latent_dim - 1:
                    ax.set_xlabel('Forecast step')

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', ncol=2)

        save_path = self.output_dir / f'latent_forecast_channels_epoch_{epoch}.pdf'
        fig.suptitle(f'Per-channel Latent Forecasts - Epoch {epoch}', fontsize=14, color='#456882')
        plt.savefig(str(save_path), bbox_inches='tight', dpi=300)
        plt.close(fig)
        gc.collect()
        logger.info(f"Latent forecast per-channel plot saved to {save_path}")

    def _plot_latent_trajectory_analysis(
        self,
        mu_post_sequence: Optional[torch.Tensor],
        mu_prior_sequence: Optional[torch.Tensor],
        epoch: int,
    ) -> None:
        """Comprehensive latent trajectory diagnostics for a single validation sample."""
        import gc

        if mu_post_sequence is None or mu_post_sequence.numel() == 0:
            return

        batch_idx = 0
        try:
            mu_post_np = mu_post_sequence[batch_idx].detach().cpu().numpy()
        except Exception:
            return

        if mu_post_np.ndim != 2:
            return

        prior_np = None
        if mu_prior_sequence is not None:
            try:
                prior_np = mu_prior_sequence[batch_idx].detach().cpu().numpy()
                if prior_np.ndim != 2 or prior_np.shape != mu_post_np.shape:
                    prior_np = None
            except Exception:
                prior_np = None

        time_axis = np.arange(mu_post_np.shape[0])
        latent_dim = mu_post_np.shape[1]
        if latent_dim == 0:
            return

        if prior_np is not None:
            delta_np = mu_post_np - prior_np
            ranking_signal = delta_np.var(axis=0)
        else:
            delta_np = None
            ranking_signal = mu_post_np.var(axis=0)
        order = np.argsort(ranking_signal)[::-1]
        top_k = int(min(latent_dim, 8))
        total_rows = top_k + 1

        fig, axes = plt.subplots(
            total_rows,
            2,
            figsize=(12, 2.2 * total_rows),
            constrained_layout=True,
        )
        axes = np.asarray(axes)

        colors = {'posterior': '#1f77b4', 'prior': '#ff7f0e', 'delta': '#2ca02c'}
        legend_handles = []
        legend_labels = []

        for row, dim in enumerate(order[:top_k]):
            ax_left = axes[row, 0]
            line_post, = ax_left.plot(time_axis, mu_post_np[:, dim], color=colors['posterior'], linewidth=1.2, label='mu_post')
            if 'mu_post' not in legend_labels:
                legend_handles.append(line_post)
                legend_labels.append('mu_post')
            if prior_np is not None:
                line_prior, = ax_left.plot(time_axis, prior_np[:, dim], color=colors['prior'], linewidth=1.0, linestyle='--', label='mu_prior')
                if 'mu_prior' not in legend_labels:
                    legend_handles.append(line_prior)
                    legend_labels.append('mu_prior')
            ax_left.grid(True, alpha=0.3)
            if row == 0:
                ax_left.set_title('Latent trajectory')
            if row == top_k - 1:
                ax_left.set_xlabel('Decimated step')
            ax_left.set_ylabel(f'z[{dim}]')

            ax_right = axes[row, 1]
            if delta_np is not None:
                line_delta, = ax_right.plot(time_axis, delta_np[:, dim], color=colors['delta'], linewidth=1.2, label='delta')
                if row == 0:
                    ax_right.set_title('Delta (mu_post - mu_prior)')
            else:
                centered = mu_post_np[:, dim] - mu_post_np[:, dim].mean()
                line_delta, = ax_right.plot(time_axis, centered, color=colors['delta'], linewidth=1.2, label='delta')
                if row == 0:
                    ax_right.set_title('Centered trajectory')
            if 'delta' not in legend_labels:
                legend_handles.append(line_delta)
                legend_labels.append('delta')
            ax_right.grid(True, alpha=0.3)
            if row == top_k - 1:
                ax_right.set_xlabel('Decimated step')

        energy = np.sum(mu_post_np ** 2, axis=0)
        summary_ax = axes[top_k, 0]
        summary_ax.bar(np.arange(latent_dim), energy, color='#345995')
        summary_ax.set_title('Posterior energy per latent')
        summary_ax.set_xlabel('Latent dim')
        summary_ax.set_ylabel('Energy')
        summary_ax.grid(True, axis='y', alpha=0.3)

        heat_ax = axes[top_k, 1]
        heat_data = mu_post_np[:, order[:top_k]].T
        im = heat_ax.imshow(heat_data, aspect='auto', origin='lower', cmap='RdBu_r')
        heat_ax.set_title('Posterior heatmap (top dims)')
        heat_ax.set_xlabel('Decimated step')
        heat_ax.set_yticks(range(top_k))
        heat_ax.set_yticklabels([f'z[{dim}]' for dim in order[:top_k]])
        fig.colorbar(im, ax=heat_ax, fraction=0.046, pad=0.04, label='Activation')

        if legend_handles:
            fig.legend(legend_handles, legend_labels, loc='upper center', ncol=len(legend_handles))

        overall_energy = float(np.linalg.norm(mu_post_np))
        if delta_np is not None:
            delta_energy = float(np.linalg.norm(delta_np))
            stats_text = f'||mu_post||_2={overall_energy:.2f}  |  ||delta||_2={delta_energy:.2f}'
        else:
            stats_text = f'||mu_post||_2={overall_energy:.2f}'

        save_path = self.output_dir / f'latent_trajectory_analysis_epoch_{epoch}.pdf'
        fig.suptitle(f'Latent Trajectory Analysis - Epoch {epoch}\n{stats_text}', fontsize=14, color='#456882')
        plt.savefig(str(save_path), bbox_inches='tight', dpi=300)
        plt.close(fig)
        gc.collect()
        logger.info(f'Latent trajectory analysis plot saved to {save_path}')
