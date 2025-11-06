"""Reusable Lightning callbacks for SeqVAE training utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

import math

if TYPE_CHECKING:
    from lightning.pytorch.loggers import MLFlowLogger

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from loguru import logger


def _resolve_validation_dataloader(trainer):
    if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
        dataloader = trainer.datamodule.val_dataloader()
    else:
        dataloader = trainer.val_dataloaders
    if isinstance(dataloader, list):
        return dataloader[0] if dataloader else None
    return dataloader


def _first_validation_batch(trainer):
    dataloader = _resolve_validation_dataloader(trainer)
    if dataloader is None:
        return None
    iterator = iter(dataloader)
    return next(iterator, None)


def _metric_to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float("nan")
        try:
            return float(value.detach().cpu().item())
        except Exception:  # noqa: BLE001
            return float("nan")
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float("nan")


class LossPlotCallback(Callback):
    """Plot training/validation losses with Plotly and optional MLflow logging."""

    def __init__(
        self,
        output_dir: Path | str,
        plot_frequency: int = 5,
        max_history_size: int = 1000,
        *,
        mlflow_logger: "MLFlowLogger" | None = None,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_frequency = max(1, int(plot_frequency))
        self.max_history_size = max(1, int(max_history_size))
        self._mlflow_logger = mlflow_logger
        self._plotly_available = True
        self.history: Dict[str, List[float]] = {"epoch": []}
        self.metric_keys: List[str] = [
            "train/total_loss",
            "train/recon_loss",
            "train/mse_loss",
            "train/nll_loss",
            "train/kld_loss",
            "train/forecast_loss",
            "train/forecast_nll",
            "train/latent_nll_loss",
            "train/predictive_kl_loss",
            "train/stability_penalty",
            "train/agg_mse",
            "train/scattering_nll",
            "train/scattering_mse",
            "train/valid_steps",
            "train/agg_mae",
            "val/total_loss",
            "val/recon_loss",
            "val/mse_loss",
            "val/nll_loss",
            "val/kld_loss",
            "val/forecast_loss",
            "val/forecast_mse",
            "val/forecast_rmse",
            "val/forecast_nll",
            "val/latent_nll_loss",
            "val/predictive_kl_loss",
            "val/stability_penalty",
            "val/scattering_nll",
            "val/scattering_mse",
            "val/valid_steps",
            "val/agg_mse",
            "val/agg_mae",
            "val/agg_corr",
            "val/agg_std",
            "val/agg_coverage",
            "hyperparams/beta",
            "hyperparams/lr",
            "kld_beta",
            "lr",
            "learning_rate",
        ]
        for key in self.metric_keys:
            self.history[key] = []

    def _trim_history(self) -> None:
        history_len = len(self.history["epoch"])
        if history_len <= self.max_history_size:
            return
        trim = history_len - self.max_history_size
        for key in self.history:
            self.history[key] = self.history[key][trim:]

    def _log_artifact(self, path: Path) -> None:
        if self._mlflow_logger is None:
            return
        try:
            self._mlflow_logger.experiment.log_artifact(self._mlflow_logger.run_id, str(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to log artifact {path} to MLflow: {exc}")

    def on_validation_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        self.history["epoch"].append(int(epoch))
        for key in self.metric_keys:
            self.history[key].append(_metric_to_float(metrics.get(key)))
        self._trim_history()
        if (epoch + 1) % self.plot_frequency != 0 or not trainer.is_global_zero:
            return
        self.plot_losses()
        self.plot_hyperparameters()

    def plot_losses(self) -> None:
        if not self._plotly_available:
            return
        try:
            import plotly.graph_objects as go
        except ImportError:
            self._plotly_available = False
            logger.warning("Plotly is not installed; skipping loss plotting.")
            return
        epochs = self.history["epoch"]
        if not epochs:
            return
        fig = go.Figure()
        for key, values in self.history.items():
            if key == "epoch":
                continue
            if key.startswith("hyperparams/") or key in {"kld_beta", "lr", "learning_rate"}:
                continue
            array = np.array(values, dtype=float)
            if array.size == 0 or np.all(np.isnan(array)):
                continue
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=array,
                    mode="lines+markers",
                    name=key.replace("/", " ").title(),
                )
            )
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
                x=1.0,
            ),
        )
        path = self.output_dir / "loss_plot_epoch.html"
        fig.write_html(str(path))
        logger.info(f"Loss plot saved to {path}")
        self._log_artifact(path)

    def plot_hyperparameters(self) -> None:
        if not self._plotly_available:
            return
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            self._plotly_available = False
            logger.warning("Plotly is not installed; skipping hyperparameter plotting.")
            return
        epochs = np.array(self.history["epoch"], dtype=float)
        if epochs.size == 0:
            return
        beta = np.array(self.history.get("hyperparams/beta", []), dtype=float)
        lr = np.array(self.history.get("hyperparams/lr", []), dtype=float)
        has_beta = beta.size and not np.all(np.isnan(beta))
        has_lr = lr.size and not np.all(np.isnan(lr))
        if not (has_beta or has_lr):
            return
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Beta (KLD Weight)", "Learning Rate"),
            horizontal_spacing=0.10,
        )
        if has_beta:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=beta,
                    mode="lines+markers",
                    name="Beta",
                    line=dict(color="firebrick", width=2),
                ),
                row=1,
                col=1,
            )
        if has_lr:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=lr,
                    mode="lines+markers",
                    name="Learning Rate",
                    line=dict(color="steelblue", width=2),
                ),
                row=1,
                col=2,
            )
        fig.update_layout(
            title="Training Hyperparameters Evolution",
            showlegend=False,
            template="plotly_white",
            height=400,
        )
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Beta Value", row=1, col=1)
        fig.update_yaxes(title_text="Learning Rate", row=1, col=2, type="log")
        path = self.output_dir / "hyperparameters_evolution.html"
        fig.write_html(str(path))
        logger.info(f"Hyperparameters plot saved to {path}")
        self._log_artifact(path)


class HyperparameterLoggingCallback(Callback):
    """Track beta and learning rate schedules across epochs."""

    def __init__(self) -> None:
        super().__init__()
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "beta": [],
            "lr": [],
        }

    def on_train_epoch_start(self, trainer, pl_module):  # type: ignore[override]
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch
        try:
            beta_value = float(_metric_to_float(pl_module._calculate_beta()))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            beta_value = float("nan")
        lr_value = float("nan")
        optimizers = getattr(trainer, "optimizers", None)
        if optimizers:
            if not isinstance(optimizers, (list, tuple)):
                optimizers = [optimizers]
            if optimizers:
                try:
                    lr_value = float(optimizers[0].param_groups[0].get("lr", float("nan")))
                except (IndexError, KeyError, TypeError):  # noqa: BLE001
                    lr_value = float("nan")
        self.history["epoch"].append(epoch)
        self.history["beta"].append(beta_value)
        self.history["lr"].append(lr_value)
        logger.info(f"Epoch {epoch}: beta={beta_value:.4f}, lr={lr_value:.6f}")

    def plot_hyperparameters(self, output_dir: Path | str) -> None:
        if not self.history["epoch"]:
            return
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.warning("Plotly is not installed; skipping hyperparameter logging plot.")
            return
        epochs = np.array(self.history["epoch"], dtype=float)
        beta = np.array(self.history["beta"], dtype=float)
        lr = np.array(self.history["lr"], dtype=float)
        has_beta = beta.size and not np.all(np.isnan(beta))
        has_lr = lr.size and not np.all(np.isnan(lr))
        if not (has_beta or has_lr):
            return
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Beta (KLD Weight)", "Learning Rate"),
            horizontal_spacing=0.10,
        )
        if has_beta:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=beta,
                    mode="lines+markers",
                    name="Beta",
                    line=dict(color="firebrick", width=2),
                ),
                row=1,
                col=1,
            )
        if has_lr:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=lr,
                    mode="lines+markers",
                    name="Learning Rate",
                    line=dict(color="steelblue", width=2),
                ),
                row=1,
                col=2,
            )
        fig.update_layout(
            title="Training Hyperparameters Evolution",
            showlegend=False,
            template="plotly_white",
            height=400,
        )
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Beta Value", row=1, col=1)
        fig.update_yaxes(title_text="Learning Rate", row=1, col=2, type="log")
        path = output_path / "hyperparameters_evolution.html"
        fig.write_html(str(path))
        logger.info(f"Hyperparameters plot saved to {path}")


class MetricsLoggingCallback(Callback):
    """Collect train/validation loss histories for post-analysis."""

    def __init__(self) -> None:
        super().__init__()
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.beta_history: List[float] = []
        self.lr_history: List[float] = []

    def on_validation_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        metrics = trainer.callback_metrics
        self.train_loss_history.append(_metric_to_float(metrics.get("train/total_loss")))
        self.val_loss_history.append(_metric_to_float(metrics.get("val/total_loss")))
        self.beta_history.append(_metric_to_float(metrics.get("kld_beta")))
        self.lr_history.append(_metric_to_float(metrics.get("learning_rate")))


class ScatteringForecastMetricsCallback(Callback):
    """Compute reconstruction and forecaster diagnostics on validation epochs."""

    def __init__(self, log_every_n_epochs: int = 5, *, num_examples: int = 1) -> None:
        super().__init__()
        self.log_every_n_epochs = max(1, int(log_every_n_epochs))
        self.num_examples = max(1, int(num_examples))

    def on_validation_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        epoch = trainer.current_epoch
        if not trainer.is_global_zero:
            return
        if (epoch + 1) % self.log_every_n_epochs != 0:
            return
        batch = _first_validation_batch(trainer)
        if batch is None:
            logger.warning("ScatteringForecastMetricsCallback: validation batch unavailable; skipping metrics")
            return
        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
        model = pl_module.model
        orig_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        pl_module.eval()
        try:
            with torch.no_grad():
                outputs = orig_model(batch.fhr_st, batch.fhr_ph, batch.fhr_up_ph)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"ScatteringForecastMetricsCallback: forward pass failed: {exc}")
            return
        finally:
            pl_module.train()
        self._log_reconstruction_metrics(outputs, batch, pl_module)
        if getattr(orig_model, "has_forecaster", lambda: False)():
            with torch.no_grad():
                forecast = orig_model.forecast_scattering(
                    y_st=batch.fhr_st,
                    y_ph=batch.fhr_ph,
                    x_ph=batch.fhr_up_ph,
                    timesteps=None,
                    use_posterior_mean=True,
                )
            self._log_forecast_metrics(forecast, batch, pl_module)

    def _log_reconstruction_metrics(self, outputs: Dict[str, Any], batch, pl_module) -> None:
        mu_pr = outputs.get("mu_pr")
        target = getattr(batch, "fhr", None)
        if not isinstance(mu_pr, torch.Tensor) or not isinstance(target, torch.Tensor):
            return
        count = min(self.num_examples, mu_pr.shape[0], target.shape[0])
        if count == 0:
            return
        diff = (mu_pr[:count] - target[:count]).detach()
        if diff.numel() == 0:
            return
        mse = torch.mean(diff.pow(2))
        if torch.isfinite(mse):
            rmse = torch.sqrt(torch.clamp(mse, min=0.0))
            pl_module.log("val/sample_recon_rmse", rmse.detach(), on_epoch=True, prog_bar=False, logger=True, sync_dist=False)

    def _log_forecast_metrics(self, forecast: Dict[str, Any], batch, pl_module) -> None:
        mu_future = forecast.get("mu_future")
        logvar_future = forecast.get("logvar_future")
        timesteps = forecast.get("timesteps")
        target_st = getattr(batch, "fhr_st", None)
        target_ph = getattr(batch, "fhr_ph", None)
        if not isinstance(mu_future, torch.Tensor) or not isinstance(timesteps, torch.Tensor):
            return
        if not isinstance(target_st, torch.Tensor) or not isinstance(target_ph, torch.Tensor):
            return
        if mu_future.dim() < 3 or timesteps.numel() == 0:
            return
        target = torch.cat([target_st, target_ph], dim=-1)
        metrics = self._compute_forecast_metrics(mu_future, logvar_future, timesteps, target)
        if not metrics:
            return
        for name, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value):
                    continue
                pl_module.log(f"val/sample_forecast_{name}", value.detach(), on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
            else:
                pl_module.log(f"val/sample_forecast_{name}", float(value), on_epoch=True, prog_bar=False, logger=True, sync_dist=False)

    def _compute_forecast_metrics(
        self,
        mu_future: torch.Tensor,
        logvar_future: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        sample_limit = min(self.num_examples, mu_future.shape[0], target.shape[0])
        if sample_limit == 0:
            return {}
        mse_sum = torch.zeros((), device=mu_future.device, dtype=mu_future.dtype)
        mse_count = 0
        nll_sum = torch.zeros((), device=mu_future.device, dtype=mu_future.dtype) if isinstance(logvar_future, torch.Tensor) else None
        nll_count = 0
        for sample_idx in range(sample_limit):
            anchors = timesteps
            for anchor_idx in range(min(mu_future.shape[1], anchors.shape[0])):
                anchor = int(anchors[anchor_idx].item())
                if anchor < 0:
                    continue
                pred = mu_future[sample_idx, anchor_idx]
                horizon = pred.shape[0]
                target_slice = target[sample_idx, anchor + 1: anchor + 1 + horizon]
                if target_slice.shape[0] == 0:
                    continue
                min_len = min(target_slice.shape[0], pred.shape[0])
                pred = pred[:min_len]
                target_slice = target_slice[:min_len]
                diff = pred - target_slice
                if diff.numel() == 0:
                    continue
                mse_val = torch.mean(diff.pow(2))
                if torch.isfinite(mse_val):
                    mse_sum = mse_sum + mse_val
                    mse_count += 1
                if isinstance(logvar_future, torch.Tensor):
                    logvar_slice = logvar_future[sample_idx, anchor_idx][:min_len]
                    if logvar_slice.shape != diff.shape:
                        min_h = min(logvar_slice.shape[0], diff.shape[0])
                        logvar_slice = logvar_slice[:min_h]
                        diff = diff[:min_h]
                    if logvar_slice.numel() == 0:
                        continue
                    var = torch.exp(logvar_slice)
                    log_two_pi = logvar_slice.new_tensor(2.0 * math.pi).log()
                    nll_tensor = 0.5 * (log_two_pi + logvar_slice + diff.pow(2) / torch.clamp(var, min=1e-8))
                    nll_val = torch.mean(nll_tensor)
                    if torch.isfinite(nll_val):
                        nll_sum = nll_sum + nll_val if nll_sum is not None else None
                        nll_count += 1
        metrics: Dict[str, torch.Tensor] = {}
        if mse_count > 0:
            avg_mse = mse_sum / mse_count
            metrics["mse"] = avg_mse.detach()
            metrics["rmse"] = torch.sqrt(torch.clamp(avg_mse, min=0.0)).detach()
        if nll_sum is not None and nll_count > 0:
            metrics["nll"] = (nll_sum / nll_count).detach()
        return metrics


class ReconstructionPlotCallback(Callback):
    """Plot reconstruction quality for a small set of validation examples."""

    _SUPPORTED_FORMATS = {"png", "pdf", "jpg", "jpeg"}

    def __init__(
        self,
        output_dir: Path | str,
        plot_frequency: int = 5,
        num_examples: int = 3,
        *,
        file_format: str = "png",
        mlflow_logger: "MLFlowLogger" | None = None,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir) / "reconstructions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_frequency = max(1, int(plot_frequency))
        self.num_examples = max(1, int(num_examples))
        file_format = file_format.lower()
        if file_format not in self._SUPPORTED_FORMATS:
            logger.warning(f"ReconstructionPlotCallback: unsupported file format '{file_format}', falling back to 'png'")
            file_format = "png"
        self.file_format = file_format
        self._mlflow_logger = mlflow_logger

    def _log_artifact(self, path: Path) -> None:
        if self._mlflow_logger is None:
            return
        try:
            self._mlflow_logger.experiment.log_artifact(self._mlflow_logger.run_id, str(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to log artifact {path} to MLflow: {exc}")

    def on_validation_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_frequency != 0:
            return
        batch = _first_validation_batch(trainer)
        if batch is None:
            logger.warning("ReconstructionPlotCallback: validation batch unavailable; skipping plots")
            return
        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
        model = pl_module.model
        orig_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        pl_module.eval()
        try:
            with torch.no_grad():
                outputs = orig_model(batch.fhr_st, batch.fhr_ph, batch.fhr_up_ph)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"ReconstructionPlotCallback: forward pass failed: {exc}")
            return
        finally:
            pl_module.train()
        mu_pr = outputs.get("mu_pr")
        logvar_pr = outputs.get("logvar_pr")
        target = getattr(batch, "fhr", None)
        if not isinstance(mu_pr, torch.Tensor) or not isinstance(target, torch.Tensor):
            logger.warning("ReconstructionPlotCallback: required tensors missing; skipping plots")
            return
        count = min(self.num_examples, mu_pr.shape[0], target.shape[0])
        if count == 0:
            return
        target_np = target[:count].detach().cpu().numpy()
        recon_np = mu_pr[:count].detach().cpu().numpy()
        std_np = None
        if isinstance(logvar_pr, torch.Tensor):
            std_np = torch.exp(0.5 * logvar_pr[:count]).detach().cpu().numpy()
        fig, axes = plt.subplots(count, 1, figsize=(14, 3 * count), sharex=True)
        if count == 1:
            axes = [axes]
        for idx in range(count):
            axis = axes[idx]
            series_target = np.squeeze(target_np[idx])
            series_recon = np.squeeze(recon_np[idx])
            length = min(series_target.shape[-1], series_recon.shape[-1])
            time_axis = np.arange(length)
            axis.plot(time_axis, series_target[..., :length], label="target", color="tab:blue", linewidth=1.2)
            axis.plot(time_axis, series_recon[..., :length], label="reconstruction", color="tab:orange", linewidth=1.0)
            if std_np is not None:
                series_std = np.squeeze(std_np[idx])
                series_std = series_std[..., :length]
                series_mean = series_recon[..., :length]
                axis.fill_between(time_axis, series_mean - 2 * series_std, series_mean + 2 * series_std, color="tab:orange", alpha=0.2)
            axis.set_ylabel(f"Example {idx}")
            axis.grid(True, alpha=0.3)
        axes[0].set_title("Validation reconstructions")
        axes[-1].set_xlabel("Time index")
        axes[0].legend(loc="upper right", frameon=False)
        fig.tight_layout()
        path = self.output_dir / f"reconstruction_epoch_{epoch:04d}.{self.file_format}"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved reconstruction plot to {path}")
        self._log_artifact(path)


class ComprehensiveForecastPlotCallback(Callback):
    """Comprehensive visualization of reconstructions and scattering forecasts."""

    def __init__(self, output_dir: Path | str, plot_every_epoch: int = 5, *, predictive_horizon: Optional[int] = None, num_examples: int = 1, max_anchors: int = 3) -> None:
        super().__init__()
        self.output_dir = Path(output_dir) / 'forecast_visualizations'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_every_epoch = max(1, int(plot_every_epoch))
        self.predictive_horizon = predictive_horizon
        self.num_examples = max(1, int(num_examples))
        self.max_anchors = max(1, int(max_anchors))

    @staticmethod
    def _first_batch(trainer):
        batch = _first_validation_batch(trainer)
        if batch is None:
            logger.warning("ComprehensiveForecastPlotCallback: would plot but validation batch is unavailable")
        return batch

    @staticmethod
    def _select_indices(total: int, count: int) -> list[int]:
        if total <= count:
            return list(range(total))
        step = max(1, total // count)
        return [idx for idx in range(0, total, step)][:count]

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_every_epoch != 0 or not trainer.is_global_zero:
            return
        batch = self._first_batch(trainer)
        if batch is None:
            return
        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
        model = pl_module.model
        orig_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        pl_module.eval()
        try:
            with torch.no_grad():
                outputs = orig_model(batch.fhr_st, batch.fhr_ph, batch.fhr_up_ph)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"ComprehensiveForecastPlotCallback: forward pass failed: {exc}")
            return
        finally:
            pl_module.train()
        mu_pr = outputs.get('mu_pr')
        logvar_pr = outputs.get('logvar_pr')
        mu_post = outputs.get('mu_post')
        mu_prior = outputs.get('mu_prior')
        latent_z = outputs.get('z')

        self._plot_reconstruction_overview(
            y_raw=batch.fhr,
            up_raw=getattr(batch, 'up', None),
            mu_pr=mu_pr,
            logvar_pr=logvar_pr,
            latent_z=latent_z,
            mu_post=mu_post,
            mu_prior=mu_prior,
            epoch=epoch,
        )

        forecast_full = getattr(orig_model, "forecast_full", None)
        if callable(forecast_full):
            try:
                with torch.no_grad():
                    forecast_dict = forecast_full(
                        y_st=batch.fhr_st,
                        y_ph=batch.fhr_ph,
                        x_ph=batch.fhr_up_ph,
                        anchors=None,
                        use_posterior_mean=True,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"ComprehensiveForecastPlotCallback: forecast_full failed: {exc}")
            else:
                mean_mu = forecast_dict.get("mean_mu")
                std_mu = forecast_dict.get("std_mu")
                canvas_mu = forecast_dict.get("canvas_mu")
                anchors = forecast_dict.get("anchors")
                latent_mu_future = forecast_dict.get("latent_mu_future")
                latent_logvar_future = forecast_dict.get("latent_logvar_future")
                if (
                    isinstance(mean_mu, torch.Tensor)
                    and isinstance(std_mu, torch.Tensor)
                    and isinstance(canvas_mu, torch.Tensor)
                    and isinstance(anchors, torch.Tensor)
                    and mean_mu.numel() > 0
                ):
                    self._plot_forecast_results(
                        y_raw_normalized=batch.fhr,
                        mean_mu=mean_mu,
                        std_mu=std_mu,
                        canvas_mu=canvas_mu,
                        anchors=anchors,
                        latent_z=latent_z,
                        epoch=epoch,
                    )
                    self._plot_latent_forecast(
                        latent_sequence=latent_z,
                        latent_mu_future=latent_mu_future,
                        latent_logvar_future=latent_logvar_future,
                        anchors=anchors,
                        enc=forecast_dict.get("enc"),
                        epoch=epoch,
                    )
                    try:
                        self._plot_batch_aggregated_forecast(
                            y_raw_batch=batch.fhr,
                            mean_mu_batch=mean_mu,
                            std_mu_batch=std_mu,
                            epoch=epoch,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f"ComprehensiveForecastPlotCallback: failed batch aggregated forecast plot: {exc}")
                else:
                    logger.warning("ComprehensiveForecastPlotCallback: forecast_full did not return all required tensors; skipping FHR plots")
        if getattr(orig_model, 'has_forecaster', lambda: False)():
            with torch.no_grad():
                forecast = orig_model.forecast_scattering(
                    y_st=batch.fhr_st,
                    y_ph=batch.fhr_ph,
                    x_ph=batch.fhr_up_ph,
                    timesteps=None,
                    use_posterior_mean=True,
                )
            mu_future = forecast.get('mu_future')
            logvar_future = forecast.get('logvar_future')
            timesteps = forecast.get('timesteps')
            target_stph = torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1)
            self._plot_scattering_forecast(epoch, mu_future, logvar_future, timesteps, target_stph)
        self._plot_latent_summary(epoch, mu_post, mu_prior)

    def _plot_reconstruction_overview(
        self,
        y_raw: torch.Tensor,
        up_raw: Optional[torch.Tensor],
        mu_pr: Optional[torch.Tensor],
        logvar_pr: Optional[torch.Tensor],
        latent_z: Optional[torch.Tensor],
        mu_post: Optional[torch.Tensor],
        mu_prior: Optional[torch.Tensor],
        epoch: int,
    ) -> None:
        import gc
        import matplotlib.pyplot as plt
        import numpy as np

        if y_raw is None or mu_pr is None or logvar_pr is None:
            return

        sample_idx = 0
        try:
            y_tensor = y_raw[sample_idx].detach().cpu()
            recon_tensor = mu_pr[sample_idx].detach().cpu()
            logvar_tensor = logvar_pr[sample_idx].detach().cpu()
        except Exception:
            return

        y_np = np.reshape(y_tensor.numpy(), -1)
        recon_np = np.reshape(recon_tensor.numpy(), -1)
        logvar_np = np.reshape(logvar_tensor.numpy(), -1)

        length = min(len(y_np), len(recon_np), len(logvar_np))
        if length == 0:
            return
        y_np = y_np[:length]
        recon_np = recon_np[:length]
        logvar_np = logvar_np[:length]
        std_np = np.sqrt(np.maximum(np.exp(logvar_np), 1e-8))
        diff_np = y_np - recon_np

        up_np = None
        if up_raw is not None:
            try:
                up_np = np.reshape(up_raw[sample_idx].detach().cpu().numpy(), -1)[:length]
            except Exception:
                up_np = None

        latent_matrix = None
        if isinstance(latent_z, torch.Tensor):
            try:
                latent_matrix = latent_z[sample_idx].detach().cpu().numpy()
            except Exception:
                latent_matrix = None
        if latent_matrix is not None and latent_matrix.ndim == 1:
            latent_matrix = latent_matrix[:, None]

        posterior_matrix = None
        if isinstance(mu_post, torch.Tensor):
            try:
                posterior_matrix = mu_post[sample_idx].detach().cpu().numpy()
            except Exception:
                posterior_matrix = None
        if posterior_matrix is not None and posterior_matrix.ndim == 1:
            posterior_matrix = posterior_matrix[:, None]

        prior_matrix = None
        if isinstance(mu_prior, torch.Tensor):
            try:
                prior_matrix = mu_prior[sample_idx].detach().cpu().numpy()
            except Exception:
                prior_matrix = None
        if prior_matrix is not None and prior_matrix.ndim == 1:
            prior_matrix = prior_matrix[:, None]

        Fs = 4.0
        time_axis = np.arange(length) / Fs

        corr = float('nan')
        if np.std(y_np) > 1e-8 and np.std(recon_np) > 1e-8:
            try:
                corr = float(np.corrcoef(y_np, recon_np)[0, 1])
            except Exception:
                corr = float('nan')
        rmse = float(np.sqrt(np.mean(diff_np ** 2))) if diff_np.size else float('nan')
        mae = float(np.mean(np.abs(diff_np))) if diff_np.size else float('nan')

        colors = {
            'fhr': '#055C9A',
            'up': '#0DD8A2',
            'gt': '#456882',
            'recon': '#BB3E00',
            'uncertainty': '#F7AD45',
            'samples': '#4BD605',
            'background': '#F9F3EF',
        }

        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'axes.linewidth': 0.7,
            'axes.edgecolor': '#9E9D9D',
            'axes.facecolor': colors['background'],
            'grid.color': '#838383',
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
            'savefig.dpi': 300,
        })

        n_rows = 4
        fig, ax = plt.subplots(
            nrows=n_rows,
            ncols=2,
            figsize=(20, n_rows * 3.2),
            gridspec_kw={'width_ratios': [80, 1]},
            constrained_layout=True,
        )

        for row in range(n_rows):
            ax[row, 0].grid(True, linestyle='-', alpha=0.35, linewidth=0.4, color='#D2C1B6')
            ax[row, 0].grid(True, which='minor', linestyle=':', alpha=0.25, linewidth=0.3, color='#D2C1B6')
            ax[row, 0].minorticks_on()
            ax[row, 0].set_axisbelow(True)
            ax[row, 0].spines['top'].set_visible(False)
            ax[row, 0].spines['right'].set_visible(False)
            ax[row, 0].spines['left'].set_color('#A2B9A7')
            ax[row, 0].spines['bottom'].set_color('#A2B9A7')
            ax[row, 0].spines['left'].set_linewidth(0.7)
            ax[row, 0].spines['bottom'].set_linewidth(0.7)
            ax[row, 1].set_axis_off()

        ax[0, 0].plot(time_axis, y_np, linewidth=1.2, color=colors['fhr'], label='FHR', alpha=0.9)
        if up_np is not None:
            ax[0, 0].plot(time_axis, up_np, linewidth=1.0, color=colors['up'], label='UP', alpha=0.75)
        ax[0, 0].set_ylabel('Amplitude')
        ax[0, 0].set_title('Raw FHR and UP Signals')
        ax[0, 0].legend(loc='upper right', framealpha=0.95)
        ax[0, 0].autoscale(enable=True, axis='x', tight=True)

        ax[1, 0].plot(time_axis, y_np, linewidth=1.5, color=colors['gt'], label='Ground Truth', alpha=0.9, zorder=3)
        ax[1, 0].plot(time_axis, recon_np, linewidth=1.3, color=colors['recon'], label='Reconstruction', alpha=0.85, zorder=2)
        ax[1, 0].fill_between(time_axis, recon_np - std_np, recon_np + std_np, color=colors['uncertainty'], alpha=0.25, label='plusminus 1 std', zorder=1)
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
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
        )

        mu_samples_np = recon_tensor.numpy()
        if mu_samples_np.ndim <= 1:
            sample_series = [np.reshape(mu_samples_np, -1)[:length]]
        else:
            flat = mu_samples_np.reshape(mu_samples_np.shape[0], -1)
            sample_series = [flat[idx, :length] for idx in range(min(6, flat.shape[0]))]

        ax[2, 0].plot(time_axis, y_np, linewidth=1.2, color=colors['gt'], label='Ground Truth', alpha=0.9)
        for idx, series in enumerate(sample_series):
            alpha_val = 0.45 + 0.1 * idx
            ax[2, 0].plot(time_axis, series, linewidth=1.0, color=colors['samples'], alpha=min(alpha_val, 0.9), label='Model sample' if idx == 0 else None)
        ax[2, 0].set_ylabel('FHR (bpm)')
        ax[2, 0].set_title('Sample Reconstructions')
        ax[2, 0].autoscale(enable=True, axis='x', tight=True)
        ax[2, 0].legend(loc='upper right', framealpha=0.95)

        if prior_matrix is not None and posterior_matrix is not None:
            ax[2, 1].set_axis_on()
            prior_arr = np.asarray(prior_matrix)
            post_arr = np.asarray(posterior_matrix)
            if prior_arr.ndim == 1:
                prior_arr = prior_arr[:, None]
            if post_arr.ndim == 1:
                post_arr = post_arr[:, None]
            steps = np.arange(post_arr.shape[0])
            ax[2, 1].plot(steps, np.linalg.norm(prior_arr, axis=1), color='#7F8C8D', linewidth=1.0, label='||mu_prior||')
            ax[2, 1].plot(steps, np.linalg.norm(post_arr, axis=1), color='#BB3E00', linewidth=1.2, label='||mu_post||')
            delta = post_arr - prior_arr
            ax[2, 1].plot(steps, np.linalg.norm(delta, axis=1), color='#2E86AB', linewidth=1.0, linestyle='--', label='||delta||')
            ax[2, 1].set_title('Latent Norm Dynamics')
            ax[2, 1].set_xlabel('Decimated step')
            ax[2, 1].set_ylabel('Norm')
            ax[2, 1].grid(True, alpha=0.3)
            ax[2, 1].legend(loc='upper right', fontsize=9, framealpha=0.9)

        if latent_matrix is not None and latent_matrix.size > 0:
            if latent_matrix.ndim == 1:
                latent_matrix = latent_matrix[:, None]
            img = ax[3, 0].imshow(latent_matrix.T, aspect='auto', cmap='RdBu_r', origin='lower')
            ax[3, 0].set_ylabel('Latent dim')
            ax[3, 0].set_xlabel('Decimated step')
            ax[3, 0].set_title('Posterior Latent Trajectory')
            ax[3, 1].set_axis_on()
            cbar = fig.colorbar(img, cax=ax[3, 1])
            cbar.ax.tick_params(labelsize=9, colors='#666666')
            cbar.set_label('Activation', fontsize=10, color='#666666')
        else:
            ax[3, 0].text(0.5, 0.5, 'Latent trajectory unavailable', ha='center', va='center', fontsize=12)
            ax[3, 0].set_axis_off()

        fig.suptitle(f'Reconstruction Overview - Epoch {epoch}', fontsize=14, color='#456882')
        save_path = self.output_dir / f'reconstruction_overview_epoch_{epoch:04d}.pdf'
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        gc.collect()
        logger.info(f'Reconstruction plot saved to {save_path}')

    def _plot_forecast_results(
        self,
        y_raw_normalized: torch.Tensor,
        mean_mu: torch.Tensor,
        std_mu: torch.Tensor,
        canvas_mu: torch.Tensor,
        anchors: torch.Tensor,
        latent_z: Optional[torch.Tensor],
        epoch: int,
    ) -> None:
        """Plot aggregated forecast curves together with uncertainty and latent heatmaps."""
        import gc
        import matplotlib.pyplot as plt
        import numpy as np

        if any(arg is None for arg in (y_raw_normalized, mean_mu, std_mu, canvas_mu, anchors)):
            return

        batch_idx = 0
        try:
            y_raw = y_raw_normalized[batch_idx].detach().cpu().numpy()
            pred_mean = mean_mu[batch_idx].detach().cpu().numpy()
            pred_std = std_mu[batch_idx].detach().cpu().numpy()
        except Exception:  # noqa: BLE001
            return

        y_raw = np.reshape(y_raw, -1)
        pred_mean = np.reshape(pred_mean, -1)
        pred_std = np.reshape(pred_std, -1)

        z_latent = None
        if isinstance(latent_z, torch.Tensor):
            try:
                z_latent = latent_z[batch_idx].detach().cpu().numpy()
            except Exception:  # noqa: BLE001
                z_latent = None

        mask = ~np.isnan(pred_mean)
        coverage = float(np.mean(mask)) if mask.size > 0 else float("nan")
        if np.any(mask):
            gt_masked = y_raw.copy()
            pred_masked = pred_mean.copy()
            gt_masked[~mask] = np.nan
            pred_masked[~mask] = np.nan
            sample_mse = float(np.nanmean((pred_masked - gt_masked) ** 2))
            sample_mae = float(np.nanmean(np.abs(pred_masked - gt_masked)))
            sample_corr = float(np.corrcoef(gt_masked[mask], pred_masked[mask])[0, 1]) if np.sum(mask) > 2 else float("nan")
        else:
            sample_mse = float("nan")
            sample_mae = float("nan")
            sample_corr = float("nan")
        coverage_display = f"{coverage:.2%}" if not np.isnan(coverage) else "nan"
        horizon_tag = self.predictive_horizon if self.predictive_horizon is not None else "model"
        logger.info(
            "[ComprehensiveForecastPlotCallback] Epoch %s sample forecast diagnostics | horizon=%s | MSE=%.4f | MAE=%.4f | Corr=%.4f | Coverage=%s",
            epoch,
            horizon_tag,
            sample_mse,
            sample_mae,
            sample_corr,
            coverage_display,
        )

        Fs = 4
        t_in = np.arange(len(y_raw)) / Fs
        colors = {
            "fhr": "#055C9A",
            "up": "#0DD8A2",
            "gt": "#456882",
            "recon": "#BB3E00",
            "uncertainty": "#F7AD45",
            "samples": "#4BD605",
            "background": "#F9F3EF",
        }

        plt.style.use("default")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#9E9D9D",
            "axes.facecolor": colors["background"],
            "grid.color": "#838383",
            "grid.linewidth": 0.4,
            "grid.alpha": 0.6,
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.shadow": False,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#A2B9A7",
            "legend.facecolor": colors["background"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
        })

        n_rows = 3
        fig, ax = plt.subplots(
            nrows=n_rows,
            ncols=2,
            figsize=(20, n_rows * 3.5),
            gridspec_kw={"width_ratios": [80, 1]},
            constrained_layout=True,
        )
        for row in range(n_rows):
            ax[row, 0].grid(True, linestyle="-", alpha=0.4, linewidth=0.4, color="#D2C1B6")
            ax[row, 0].grid(True, which="minor", linestyle=":", alpha=0.25, linewidth=0.3, color="#D2C1B6")
            ax[row, 0].minorticks_on()
            ax[row, 0].set_axisbelow(True)
            ax[row, 0].spines["top"].set_visible(False)
            ax[row, 0].spines["right"].set_visible(False)
            ax[row, 0].spines["left"].set_color("#A2B9A7")
            ax[row, 0].spines["bottom"].set_color("#A2B9A7")
            ax[row, 0].spines["left"].set_linewidth(0.7)
            ax[row, 0].spines["bottom"].set_linewidth(0.7)
            ax[row, 1].set_axis_off()

        ax[0, 0].plot(t_in, y_raw, linewidth=1.5, color=colors["gt"], label="Ground Truth", alpha=0.9)
        ax[0, 0].plot(t_in, pred_mean, linewidth=1.2, color=colors["recon"], label="Forecast (mean)", alpha=0.9)
        ax[0, 0].fill_between(
            t_in,
            pred_mean - pred_std,
            pred_mean + pred_std,
            alpha=0.3,
            color=colors["uncertainty"],
            label="Uncertainty band",
        )
        ax[0, 0].set_ylabel("FHR (bpm)")
        ax[0, 0].set_title("Raw FHR vs Aggregated Forecast")
        ax[0, 0].legend(loc="upper right", framealpha=0.95)
        ax[0, 0].autoscale(enable=True, axis="x", tight=True)

        ax[1, 0].plot(t_in, y_raw, color=colors["gt"], alpha=0.4, linewidth=1.0)
        if isinstance(anchors, torch.Tensor) and anchors.numel() > 0 and isinstance(canvas_mu, torch.Tensor):
            anc = anchors.detach().cpu().numpy()
            cmu = canvas_mu[batch_idx].detach().cpu().numpy()
            if cmu.ndim == 1:
                cmu = cmu[np.newaxis, ...]
            pick_indices = [0, len(anc) // 2, len(anc) - 1] if len(anc) >= 3 else list(range(len(anc)))
            seen: set[int] = set()
            for idx in pick_indices:
                idx = int(idx)
                if idx < 0 or idx >= cmu.shape[0] or idx in seen:
                    continue
                seen.add(idx)
                window = cmu[idx]
                ax[1, 0].plot(t_in, window, color="#D7263D", linewidth=1.0, alpha=0.8)
        ax[1, 0].set_ylabel("FHR (bpm)")
        ax[1, 0].set_title("Sample Forecast Windows")
        ax[1, 0].autoscale(enable=True, axis="x", tight=True)

        if z_latent is not None:
            img = ax[2, 0].imshow(z_latent.T, aspect="auto", cmap="bwr", origin="lower")
            ax[2, 1].set_axis_on()
            cbar = fig.colorbar(img, cax=ax[2, 1])
            cbar.ax.tick_params(labelsize=10, colors="#666666")
            cbar.set_label("Activation", fontsize=11, color="#666666")
            cbar.outline.set_color("#A2B9A7")
            cbar.outline.set_linewidth(0.7)
            ax[2, 0].set_ylabel("Latent Dimensions")
            ax[2, 0].set_xlabel("Decimated steps")
            ax[2, 0].set_title("Latent mu_post (T x D)")
        else:
            ax[2, 0].text(0.5, 0.5, "Latents not available", ha="center", va="center")

        diag_str = (
            f"H={horizon_tag} | "
            f"MSE={sample_mse:.4f} | MAE={sample_mae:.4f} | Corr={sample_corr:.4f} | Cov={coverage_display}"
        )
        fig.suptitle(f"Forecasting Results - Epoch {epoch}\n{diag_str}", fontsize=14, color="#456882")
        save_path = self.output_dir / f"forecast_results_epoch_{epoch:04d}.pdf"
        fig.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white", edgecolor="none")
        plt.close(fig)
        gc.collect()
        logger.info(f"Forecast results plot saved to {save_path}")

    def _plot_latent_forecast(
        self,
        latent_sequence: Optional[torch.Tensor],
        latent_mu_future: Optional[torch.Tensor],
        latent_logvar_future: Optional[torch.Tensor],
        anchors: Optional[torch.Tensor],
        enc: Optional[Dict[str, torch.Tensor]],
        epoch: int,
    ) -> None:
        """Visualize latent predictions against ground truth for selected anchors."""
        import gc
        import matplotlib.pyplot as plt
        import numpy as np

        if latent_mu_future is None or anchors is None or anchors.numel() == 0:
            logger.warning("[ComprehensiveForecastPlotCallback] Skipping latent forecast plot: no latent predictions available")
            return

        batch_idx = 0
        anchors_np = anchors.detach().cpu().numpy()
        target_steps = [60, 150, 169]
        selected: list[tuple[int, int]] = []
        for t in target_steps:
            matches = np.where(anchors_np == t)[0]
            if matches.size > 0:
                selected.append((t, int(matches[0])))
        if not selected:
            logger.warning("[ComprehensiveForecastPlotCallback] Skipping latent forecast plot: none of the requested anchors were found")
            return

        gt_source = None
        if isinstance(enc, dict):
            mu_post = enc.get("mu_post")
            if isinstance(mu_post, torch.Tensor):
                gt_source = mu_post.detach().cpu().numpy()
        if gt_source is None and isinstance(latent_sequence, torch.Tensor):
            gt_source = latent_sequence.detach().cpu().numpy()
        if gt_source is None:
            logger.warning("[ComprehensiveForecastPlotCallback] Skipping latent forecast plot: no latent targets available")
            return
        if gt_source.ndim == 2:
            gt_source = gt_source[None, ...]
        if batch_idx >= gt_source.shape[0]:
            logger.warning("[ComprehensiveForecastPlotCallback] Skipping latent forecast plot: batch index out of range")
            return

        gt_full = gt_source[batch_idx]
        pred_full = latent_mu_future[batch_idx].detach().cpu().numpy()
        var_source = latent_logvar_future
        if var_source is None and isinstance(enc, dict):
            var_source = enc.get("latent_logvar_future")

        columns: list[tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
        for anchor_t, anchor_idx in selected:
            if anchor_idx >= pred_full.shape[0]:
                continue
            pred_window = pred_full[anchor_idx]
            if pred_window.ndim == 1:
                pred_window = pred_window[:, None]
            if isinstance(var_source, torch.Tensor) and anchor_idx < var_source.shape[1]:
                var_window = var_source[batch_idx, anchor_idx].detach().cpu().numpy()
                if var_window.ndim == 1:
                    var_window = var_window[:, None]
            else:
                var_window = None
            window_len = pred_window.shape[0]
            start = anchor_t + 1
            end = start + window_len
            if start >= gt_full.shape[0]:
                continue
            end = min(end, gt_full.shape[0])
            gt_window = gt_full[start:end]
            if gt_window.shape[0] != pred_window.shape[0]:
                window_len = min(gt_window.shape[0], pred_window.shape[0])
                pred_window = pred_window[:window_len]
                gt_window = gt_window[:window_len]
                if var_window is not None:
                    var_window = var_window[:window_len]
            if window_len == 0:
                continue
            columns.append((anchor_t, pred_window, gt_window, var_window))

        if not columns:
            logger.warning("[ComprehensiveForecastPlotCallback] Skipping latent forecast plot: no usable windows after alignment")
            return

        n_cols = len(columns)
        latent_dim = columns[0][1].shape[1]
        time_axis = np.arange(columns[0][1].shape[0])
        fig_height = max(latent_dim * 1.1, 6.0)
        fig_width = max(n_cols * 4.0, 8.0)
        fig, axes = plt.subplots(
            nrows=latent_dim,
            ncols=n_cols,
            figsize=(fig_width, fig_height),
            sharex=True,
            constrained_layout=True,
        )
        if latent_dim == 1:
            axes = np.expand_dims(axes, axis=0)
        if n_cols == 1:
            axes = np.expand_dims(axes, axis=1)

        mse_summary: list[tuple[int, float]] = []
        for col_idx, (anchor_t, pred_window, gt_window, var_window) in enumerate(columns):
            mse_val = float(np.mean((pred_window - gt_window) ** 2))
            mse_summary.append((anchor_t, mse_val))
            for dim_idx in range(latent_dim):
                axis = axes[dim_idx, col_idx]
                axis.plot(time_axis, gt_window[:, dim_idx], label="GT" if col_idx == 0 and dim_idx == 0 else None, color="#055C9A", linewidth=1.2)
                axis.plot(time_axis, pred_window[:, dim_idx], label="Pred" if col_idx == 0 and dim_idx == 0 else None, color="#BB3E00", linewidth=1.0, linestyle="--")
                if var_window is not None:
                    std_vals = np.sqrt(np.maximum(var_window[:, dim_idx], 1e-8))
                    axis.fill_between(time_axis, pred_window[:, dim_idx] - std_vals, pred_window[:, dim_idx] + std_vals, color="#BB3E00", alpha=0.15, linewidth=0)
                axis.grid(alpha=0.3, linewidth=0.4)
                if col_idx == 0:
                    axis.set_ylabel(f"z{dim_idx}", fontsize=8)
                else:
                    axis.set_yticklabels([])
                if dim_idx == latent_dim - 1:
                    axis.set_xlabel("Forecast step", fontsize=8)
                if dim_idx == 0:
                    axis.set_title(f"t = {anchor_t}", fontsize=10)
        if latent_dim > 0 and n_cols > 0:
            axes[0, 0].legend(loc="upper right", fontsize=8)

        summary_str = " | ".join([f"t={t}: MSE={m:.4f}" for t, m in mse_summary])
        fig.suptitle(f"Latent Forecast vs Ground Truth | Epoch {epoch}\n{summary_str}", fontsize=14, color="#456882")
        save_path = self.output_dir / f"latent_forecast_epoch_{epoch:04d}.pdf"
        fig.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white", edgecolor="none")
        plt.close(fig)
        gc.collect()
        logger.info(f"Latent forecast comparison plot saved to {save_path}")

    def _plot_batch_aggregated_forecast(
        self,
        y_raw_batch: torch.Tensor,
        mean_mu_batch: torch.Tensor,
        std_mu_batch: torch.Tensor,
        epoch: int,
    ) -> None:
        """Plot average aggregated forecast across validation batch with uncertainty band."""
        import matplotlib.pyplot as plt
        import numpy as np

        if not isinstance(mean_mu_batch, torch.Tensor) or mean_mu_batch.numel() == 0:
            return

        mask = ~torch.isnan(mean_mu_batch)
        pred_mean = torch.nanmean(mean_mu_batch, dim=0)
        gt_masked = y_raw_batch.masked_fill(~mask, float("nan"))
        gt_mean = torch.nanmean(gt_masked, dim=0)

        e_var = torch.nanmean(std_mu_batch.pow(2), dim=0)
        mean_centered = mean_mu_batch - pred_mean.unsqueeze(0)
        mean_centered = mean_centered.masked_fill(~mask, float("nan"))
        var_e = torch.nanmean(mean_centered.pow(2), dim=0)
        total_var = (e_var + var_e).clamp_min(0.0)
        pred_std = total_var.sqrt()

        t = np.arange(pred_mean.shape[0]) / 4.0
        pm = pred_mean.detach().cpu().numpy()
        ps = pred_std.detach().cpu().numpy()
        gm = gt_mean.detach().cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(16, 4.5), constrained_layout=True)
        ax.plot(t, gm, color="#2E86AB", label="GT (batch mean)", linewidth=1.5)
        ax.plot(t, pm, color="#BB3E00", label="Forecast (batch mean)", linewidth=1.2)
        ax.fill_between(
            t,
            pm - ps,
            pm + ps,
            color="#F5B7B1",
            alpha=0.4,
            label="Total uncertainty band",
        )
        ax.set_title("Batch-Aggregated Forecast vs Ground Truth")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("FHR (bpm)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        save_path = self.output_dir / f"forecast_results_epoch_{epoch:04d}_avg.pdf"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Batch-aggregated forecast plot saved to {save_path}")

    def _plot_scattering_forecast(self, epoch: int, mu_future: Optional[torch.Tensor], logvar_future: Optional[torch.Tensor], timesteps: Optional[torch.Tensor], target_stph: torch.Tensor) -> None:
        import matplotlib.pyplot as plt
        import numpy as np
        if mu_future is None or timesteps is None or timesteps.numel() == 0:
            logger.warning('ComprehensiveForecastPlotCallback: no scattering predictions available for plotting.')
            return
        sample_idx = 0
        if sample_idx >= mu_future.shape[0]:
            return
        mu_np = mu_future[sample_idx].detach().cpu().numpy()
        logvar_np = None
        if isinstance(logvar_future, torch.Tensor) and sample_idx < logvar_future.shape[0]:
            logvar_np = logvar_future[sample_idx].detach().cpu().numpy()
        anchors = timesteps.detach().cpu().numpy()
        selected_idx = self._select_indices(len(anchors), self.max_anchors)
        rows = len(selected_idx)
        if rows == 0:
            return
        fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows), constrained_layout=True)
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)
        mse_values = []
        stph_np = target_stph[sample_idx].detach().cpu().numpy()
        samples: List[Dict[str, Any]] = []
        for row, anchor_idx in enumerate(selected_idx):
            anchor = int(anchors[anchor_idx])
            if anchor < 0 or anchor_idx >= mu_np.shape[0]:
                continue
            pred = mu_np[anchor_idx]
            std_arr = None
            if logvar_np is not None and anchor_idx < logvar_np.shape[0]:
                std_arr = np.sqrt(np.maximum(np.exp(logvar_np[anchor_idx]), 1e-8))
            horizon = pred.shape[0]
            target = stph_np[anchor + 1: anchor + 1 + horizon]
            if target.shape[0] != pred.shape[0]:
                min_len = min(target.shape[0], pred.shape[0])
                pred = pred[:min_len]
                target = target[:min_len]
                if std_arr is not None:
                    std_arr = std_arr[:min_len]
            if target.size == 0 or pred.size == 0:
                continue
            diff = pred - target
            mse = float(np.mean(diff ** 2)) if target.size else float('nan')
            mse_values.append((anchor, mse))
            im0 = axes[row, 0].imshow(target.T, aspect='auto', origin='lower', cmap='viridis')
            axes[row, 0].set_title(f'Target scattering | t={anchor}')
            plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.01)
            im1 = axes[row, 1].imshow(pred.T, aspect='auto', origin='lower', cmap='viridis')
            axes[row, 1].set_title('Predicted scattering')
            plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.01)
            im2 = axes[row, 2].imshow(np.abs(diff).T, aspect='auto', origin='lower', cmap='magma')
            axes[row, 2].set_title(f'|Error| (MSE={mse:.4f})')
            plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.01)
            for col in range(3):
                axes[row, col].set_ylabel('Channel index')
                axes[row, col].set_xlabel('Horizon step')
            samples.append(
                {
                    'anchor': anchor,
                    'pred': np.array(pred, copy=True),
                    'target': np.array(target, copy=True),
                    'std': np.array(std_arr, copy=True) if isinstance(std_arr, np.ndarray) else None,
                }
            )
        fig.suptitle('Scattering forecast vs target', fontsize=14)
        save_path = self.output_dir / f'scattering_forecast_epoch_{epoch:04d}.pdf'
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        if mse_values:
            summary = ', '.join(f"t={t}: MSE={m:.4f}" for t, m in mse_values)
            logger.info(f'Scattering forecast summary (epoch {epoch}): {summary}')
        self._plot_channel_average_trajectories(epoch, samples)
        self._plot_band_error_heatmap(epoch, samples)
        self._plot_forecast_fan_chart(epoch, samples)

    def _plot_channel_average_trajectories(self, epoch: int, samples: List[Dict[str, Any]]) -> None:
        import matplotlib.pyplot as plt
        import numpy as np
        valid = [s for s in samples if isinstance(s.get('pred'), np.ndarray) and s['pred'].ndim >= 2 and s['pred'].size > 0]
        if not valid:
            return
        fig, axes = plt.subplots(len(valid), 1, figsize=(12, 3 * len(valid)), sharex=True)
        if len(valid) == 1:
            axes = [axes]
        for ax, sample in zip(axes, valid):
            pred = sample['pred']
            target = sample['target']
            horizon = min(pred.shape[0], target.shape[0])
            if horizon == 0:
                continue
            pred_mean = np.mean(pred[:horizon], axis=-1)
            target_mean = np.mean(target[:horizon], axis=-1)
            x = np.arange(horizon)
            ax.plot(x, target_mean, label='target mean', color='tab:blue')
            ax.plot(x, pred_mean, label='prediction mean', color='tab:orange')
            ax.set_ylabel(f't={sample["anchor"]}')
            ax.grid(alpha=0.3)
        axes[0].set_title('Channel-averaged scattering trajectories')
        axes[-1].set_xlabel('Horizon step')
        axes[0].legend(loc='upper right', frameon=False)
        fig.tight_layout()
        path = self.output_dir / f'channel_mean_forecast_epoch_{epoch:04d}.pdf'
        fig.savefig(path, dpi=200)
        plt.close(fig)
        logger.info(f"Saved channel-averaged forecast plot to {path}")

    def _plot_band_error_heatmap(self, epoch: int, samples: List[Dict[str, Any]]) -> None:
        import matplotlib.pyplot as plt
        import numpy as np
        valid = [s for s in samples if isinstance(s.get('pred'), np.ndarray) and s['pred'].ndim >= 2 and s['pred'].shape[-1] > 0]
        if not valid:
            return
        channels = valid[0]['pred'].shape[-1]
        band_count = min(6, channels)
        if band_count == 0:
            return
        bands = [band for band in np.array_split(np.arange(channels), band_count) if band.size > 0]
        if not bands:
            return
        heatmap = np.full((len(valid), len(bands)), np.nan, dtype=float)
        anchor_labels: List[int] = []
        for row, sample in enumerate(valid):
            anchor_labels.append(int(sample['anchor']))
            pred = sample['pred']
            target = sample['target']
            horizon = min(pred.shape[0], target.shape[0])
            if horizon == 0 or pred.shape[-1] != channels or target.shape[-1] != channels:
                continue
            diff = pred[:horizon] - target[:horizon]
            for col, band in enumerate(bands):
                band_diff = diff[..., band]
                if band_diff.size == 0:
                    continue
                heatmap[row, col] = float(np.sqrt(np.mean(np.square(band_diff))))
        if not np.isfinite(heatmap).any():
            return
        fig, ax = plt.subplots(figsize=(12, 0.8 * len(valid) + 2))
        im = ax.imshow(heatmap, aspect='auto', origin='lower', cmap='magma')
        ax.set_title('Band-wise forecast RMSE')
        ax.set_xlabel('Channel band')
        ax.set_ylabel('Anchor t')
        ax.set_xticks(np.arange(len(bands)))
        band_labels = [f'B{idx + 1}\n[{band[0]}-{band[-1]}]' for idx, band in enumerate(bands)]
        ax.set_xticklabels(band_labels)
        ax.set_yticks(np.arange(len(valid)))
        ax.set_yticklabels([str(anchor) for anchor in anchor_labels])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='RMSE')
        fig.tight_layout()
        path = self.output_dir / f'band_error_heatmap_epoch_{epoch:04d}.pdf'
        fig.savefig(path, dpi=200)
        plt.close(fig)
        logger.info(f"Saved band-wise error heatmap to {path}")

    def _plot_forecast_fan_chart(self, epoch: int, samples: List[Dict[str, Any]]) -> None:
        import matplotlib.pyplot as plt
        import numpy as np
        valid = [s for s in samples if isinstance(s.get('pred'), np.ndarray) and s['pred'].ndim >= 2 and s['pred'].shape[0] > 0]
        if not valid:
            return
        fig, axes = plt.subplots(len(valid), 1, figsize=(12, 3 * len(valid)), sharex=True)
        if len(valid) == 1:
            axes = [axes]
        for ax, sample in zip(axes, valid):
            pred = sample['pred']
            target = sample['target']
            horizon = min(pred.shape[0], target.shape[0])
            if horizon == 0:
                continue
            pred_mean = np.mean(pred[:horizon], axis=-1)
            target_mean = np.mean(target[:horizon], axis=-1)
            x = np.arange(horizon)
            ax.plot(x, target_mean, label='target mean', color='tab:blue')
            ax.plot(x, pred_mean, label='prediction mean', color='tab:orange')
            std_matrix = sample.get('std')
            if isinstance(std_matrix, np.ndarray) and std_matrix.ndim >= 2:
                std_matrix = std_matrix[:horizon]
                std_agg = np.sqrt(np.mean(std_matrix ** 2, axis=-1))
                ax.fill_between(x, pred_mean - std_agg, pred_mean + std_agg, color='tab:orange', alpha=0.2, label='prediction mean +/- 1 std')
                ax.fill_between(x, pred_mean - 2 * std_agg, pred_mean + 2 * std_agg, color='tab:orange', alpha=0.1, label='prediction mean +/- 2 std')
            ax.set_ylabel(f't={sample["anchor"]}')
            ax.grid(alpha=0.3)
        axes[0].set_title('Forecast fan chart (channel-averaged)')
        axes[-1].set_xlabel('Horizon step')
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            dedup_handles = []
            dedup_labels = []
            for handle, label in zip(handles, labels):
                if label in dedup_labels:
                    continue
                dedup_handles.append(handle)
                dedup_labels.append(label)
            axes[0].legend(dedup_handles, dedup_labels, loc='upper right', frameon=False)
        fig.tight_layout()
        path = self.output_dir / f'forecast_fan_chart_epoch_{epoch:04d}.pdf'
        fig.savefig(path, dpi=200)
        plt.close(fig)
        logger.info(f"Saved forecast fan chart to {path}")

    def _plot_latent_summary(self, epoch: int, mu_post: Optional[torch.Tensor], mu_prior: Optional[torch.Tensor]) -> None:
        import matplotlib.pyplot as plt
        import numpy as np
        if mu_post is None or mu_prior is None:
            return
        sample_idx = 0
        post = mu_post[sample_idx].detach().cpu().numpy()
        prior = mu_prior[sample_idx].detach().cpu().numpy()
        diff = post - prior
        fig, axes = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)
        im0 = axes[0].imshow(prior.T, aspect='auto', origin='lower', cmap='RdBu_r')
        axes[0].set_title('Prior mean')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.01)
        im1 = axes[1].imshow(post.T, aspect='auto', origin='lower', cmap='RdBu_r')
        axes[1].set_title('Posterior mean')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.01)
        im2 = axes[2].imshow(diff.T, aspect='auto', origin='lower', cmap='coolwarm')
        axes[2].set_title('Posterior - Prior')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.01)
        for ax in axes:
            ax.set_xlabel('Decimated step')
            ax.set_ylabel('Latent dim')
        fig.suptitle(f'Latent summary - epoch {epoch}', fontsize=14)
        save_path = self.output_dir / f'latent_summary_epoch_{epoch:04d}.pdf'
        fig.savefig(save_path, dpi=200)
        plt.close(fig)


__all__ = [
    "LossPlotCallback",
    "HyperparameterLoggingCallback",
    "MetricsLoggingCallback",
    "ScatteringForecastMetricsCallback",
    "ReconstructionPlotCallback",
    "ComprehensiveForecastPlotCallback",
]
