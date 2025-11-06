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
    """Plot training/validation losses and optionally push to MLflow."""

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
        self._loss_dir = self.output_dir / "loss_plots"
        self._loss_dir.mkdir(parents=True, exist_ok=True)
        self.plot_frequency = max(1, int(plot_frequency))
        self.max_history_size = max(1, int(max_history_size))
        self._mlflow_logger = mlflow_logger
        self.history: Dict[str, List[float]] = {"epoch": []}
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
            "val/forecast_mse",
            "val/forecast_rmse",
            "val/forecast_nll",
            "val/valid_steps",
            "val/agg_mse",
            "val/agg_mae",
            "val/agg_corr",
            "val/agg_std",
            "val/agg_coverage",
            "kld_beta",
            "learning_rate",
        ]
        for key in self.metric_keys:
            self.history[key] = []

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

    def on_validation_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        self.history["epoch"].append(epoch)
        for key in self.metric_keys:
            self.history[key].append(_metric_to_float(metrics.get(key)))
        self._trim_history()
        if not trainer.is_global_zero:
            return
        if (epoch + 1) % self.plot_frequency != 0:
            return
        self._plot_losses()
        self._plot_hparams()

    def _plot_losses(self) -> None:
        epochs = np.array(self.history["epoch"], dtype=float)
        if epochs.size == 0:
            return
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        train_keys = [k for k in self.metric_keys if k.startswith("train/")]
        val_keys = [k for k in self.metric_keys if k.startswith("val/")]
        self._plot_group(axes[0], epochs, train_keys, "Training metrics")
        self._plot_group(axes[1], epochs, val_keys, "Validation metrics")
        axes[1].set_xlabel("Epoch")
        fig.tight_layout()
        path = self._loss_dir / f"losses_epoch_{int(epochs[-1]):04d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved loss plot to {path}")
        self._log_artifact(path)

    def _plot_group(self, axis: plt.Axes, epochs: np.ndarray, keys: Iterable[str], title: str) -> None:
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        for key in keys:
            values = np.array(self.history.get(key, []), dtype=float)
            if values.size == 0 or np.all(np.isnan(values)):
                continue
            axis.plot(epochs, values, label=key)
        if axis.lines:
            axis.legend(loc="upper right", frameon=False)

    def _plot_hparams(self) -> None:
        epochs = np.array(self.history["epoch"], dtype=float)
        if epochs.size == 0:
            return
        beta = np.array(self.history.get("kld_beta", []), dtype=float)
        lr = np.array(self.history.get("learning_rate", []), dtype=float)
        if (beta.size == 0 or np.all(np.isnan(beta))) and (lr.size == 0 or np.all(np.isnan(lr))):
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        axes[0].set_title("KLD beta")
        axes[0].grid(True, alpha=0.3)
        if beta.size and not np.all(np.isnan(beta)):
            axes[0].plot(epochs, beta, color="tab:red")
        axes[0].set_xlabel("Epoch")
        axes[1].set_title("Learning rate")
        axes[1].grid(True, alpha=0.3)
        if lr.size and not np.all(np.isnan(lr)):
            axes[1].plot(epochs, lr, color="tab:blue")
        axes[1].set_xlabel("Epoch")
        axes[1].set_yscale("log")
        fig.tight_layout()
        path = self._loss_dir / f"hyperparams_epoch_{int(epochs[-1]):04d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved hyperparameter plot to {path}")
        self._log_artifact(path)


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
    "MetricsLoggingCallback",
    "ScatteringForecastMetricsCallback",
    "ReconstructionPlotCallback",
    "ComprehensiveForecastPlotCallback",
]
