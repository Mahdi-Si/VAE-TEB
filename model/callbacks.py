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
                    # Generate forecasts
                    if hasattr(orig_model, 'forecast_scattering'):
                        forecast_out = orig_model.forecast_scattering(
                            y_st=y_st, y_ph=y_ph, x_ph=x_ph, timesteps=None, use_posterior_mean=True
                        )
                    else:
                        logger.warning("Model has forecaster but no forecast_scattering method")
                        forecast_out = None

                    if forecast_out is not None:
                        mu_future = forecast_out.get("mu_future")
                        logvar_future = forecast_out.get("logvar_future")
                        timesteps = forecast_out.get("timesteps")

                        # Plot forecast visualizations
                        if mu_future is not None and timesteps is not None:
                            # Aggregate forecasts to canvas for visualization
                            if hasattr(orig_model, 'aggregate_scattering_forecasts'):
                                canvas_mu, mean_mu = orig_model.aggregate_scattering_forecasts(
                                    mu_future, timesteps, total_len=y_raw_normalized.shape[1]
                                )
                                var_future = logvar_future.exp() if logvar_future is not None else None
                                if var_future is not None:
                                    _, mean_var = orig_model.aggregate_scattering_forecasts(
                                        var_future, timesteps, total_len=y_raw_normalized.shape[1]
                                    )
                                    std_mu = mean_var.clamp_min(1e-8).sqrt()
                                else:
                                    std_mu = torch.zeros_like(mean_mu)

                                self._plot_forecast_results_scattering(
                                    y_raw_normalized,
                                    mean_mu,
                                    std_mu,
                                    canvas_mu,
                                    timesteps,
                                    mu_post_full,
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
