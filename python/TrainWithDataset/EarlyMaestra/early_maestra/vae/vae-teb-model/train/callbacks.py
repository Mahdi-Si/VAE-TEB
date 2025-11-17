"""Project-specific Lightning callbacks for plotting and metric logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import fnmatch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from loguru import logger

if TYPE_CHECKING:
    from lightning.pytorch.loggers import MLFlowLogger


def _resolve_validation_dataloader(trainer):
    """Return the first validation dataloader regardless of trainer setup."""
    if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
        dataloader = trainer.datamodule.val_dataloader()
    else:
        dataloader = trainer.val_dataloaders
    if isinstance(dataloader, list):
        return dataloader[0] if dataloader else None
    return dataloader


def _first_validation_batch(trainer):
    """Fetch the very first validation batch for qualitative callbacks."""
    dataloader = _resolve_validation_dataloader(trainer)
    if dataloader is None:
        return None
    iterator = iter(dataloader)
    return next(iterator, None)


def _metric_to_float(value: Any) -> float:
    """Convert tensors or scalars to float, returning NaN when conversion fails."""
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
    """Plot scalar metrics logged during training/validation."""

    def __init__(
        self,
        output_dir: Path | str,
        plot_frequency: int = 5,
        max_history_size: int = 1000,
        *,
        metric_filters: Optional[Iterable[str]] = None,
        hyperparam_keys: Optional[Iterable[str]] = None,
        mlflow_logger: "MLFlowLogger" | None = None,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir) / "loss_plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_frequency = max(1, int(plot_frequency))
        self.max_history_size = max(1, int(max_history_size))
        self._mlflow_logger = mlflow_logger
        self._plotly_available = True
        self.history: Dict[str, List[float]] = {"epoch": []}
        self.metric_filters: Tuple[str, ...] = tuple(metric_filters or ("train/*", "val/*"))
        self.hyperparam_keys: Tuple[str, ...] = tuple(
            hyperparam_keys or ("hyperparams/beta", "hyperparams/lr", "kld_beta", "lr", "learning_rate")
        )
        for key in self.hyperparam_keys:
            self.history.setdefault(key, [])
        self._tracked_metrics: List[str] = []

    def _should_track_metric(self, name: str) -> bool:
        if name.endswith("_step") or name in self.hyperparam_keys or name == "epoch":
            return False
        if not self.metric_filters:
            return True
        return any(fnmatch.fnmatch(name, pattern) for pattern in self.metric_filters)

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
        current_index = len(self.history["epoch"]) - 1
        epoch_values: Dict[str, float] = {}
        for name, value in metrics.items():
            if not self._should_track_metric(name):
                continue
            epoch_values[name] = _metric_to_float(value)
            if name not in self.history:
                self.history[name] = [float("nan")] * current_index
            if name not in self._tracked_metrics:
                self._tracked_metrics.append(name)
        for name in self._tracked_metrics:
            metric_value = epoch_values.get(name, float("nan"))
            self.history[name].append(metric_value)
        for key in self.hyperparam_keys:
            if key not in self.history:
                self.history[key] = [float("nan")] * current_index
            self.history[key].append(_metric_to_float(metrics.get(key)))
        self._trim_history()
        if not trainer.is_global_zero:
            return
        if (epoch + 1) % self.plot_frequency != 0:
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
            if key == "epoch" or key in self.hyperparam_keys:
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
            yaxis_title="Value",
            legend_title="Metrics",
            template="plotly_white",
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
    """Track arbitrary hyper-parameters (learning rates, annealing weights, etc.)."""

    def __init__(self, tracked_keys: Optional[Iterable[str]] = None) -> None:
        super().__init__()
        self.tracked_keys: Tuple[str, ...] = tuple(
            tracked_keys or ("hyperparams/beta", "kld_beta", "hyperparams/lr", "lr", "learning_rate")
        )
        self.history: Dict[str, List[float]] = {"epoch": []}
        for key in self.tracked_keys:
            self.history[key] = []

    def on_train_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        if not trainer.is_global_zero:
            return
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        self.history["epoch"].append(int(epoch))
        for key in self.tracked_keys:
            self.history[key].append(_metric_to_float(metrics.get(key)))

    def plot_hyperparameters(self, output_dir: Path | str) -> None:
        if not self.history["epoch"]:
            return
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("Plotly is not installed; skipping hyperparameter logging plot.")
            return
        epochs = np.array(self.history["epoch"], dtype=float)
        if epochs.size == 0:
            return
        fig = go.Figure()
        plotted = False
        for key in self.tracked_keys:
            values = np.array(self.history.get(key, []), dtype=float)
            if values.size == 0 or np.all(np.isnan(values)):
                continue
            plotted = True
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=values,
                    mode="lines+markers",
                    name=key,
                )
            )
        if not plotted:
            return
        fig.update_layout(
            title="Hyper-parameter Evolution",
            xaxis_title="Epoch",
            yaxis_title="Value",
            template="plotly_white",
        )
        path = output_path / "hyperparameters_evolution.html"
        fig.write_html(str(path))
        logger.info(f"Hyperparameters plot saved to {path}")


class MetricsLoggingCallback(Callback):
    """Collect arbitrary metrics each validation epoch for post-analysis."""

    def __init__(self, tracked_metrics: Optional[Iterable[str]] = None) -> None:
        super().__init__()
        self.tracked_metrics: Tuple[str, ...] = tuple(
            tracked_metrics or ("train/total_loss", "val/total_loss", "kld_beta", "learning_rate")
        )
        self.history: Dict[str, List[float]] = {name: [] for name in self.tracked_metrics}

    def on_validation_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        metrics = trainer.callback_metrics
        for name in self.tracked_metrics:
            self.history[name].append(_metric_to_float(metrics.get(name)))

    def as_dict(self) -> Dict[str, List[float]]:
        return self.history


class ReconstructionPlotCallback(Callback):
    """Plot reconstructions for a handful of validation samples."""

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
        self.file_format = file_format
        self._mlflow_logger = mlflow_logger

    def _log_artifact(self, path: Path) -> None:
        if self._mlflow_logger is None:
            return
        self._mlflow_logger.experiment.log_artifact(self._mlflow_logger.run_id, str(path))

    @staticmethod
    def _extract(batch, name):
        if isinstance(batch, dict):
            return batch.get(name)
        return getattr(batch, name, None)

    @staticmethod
    def _guid_from_batch(guid_field, index: int = 0) -> str:
        if guid_field is None:
            return "unknown"
        if isinstance(guid_field, (list, tuple)):
            if not guid_field:
                return "unknown"
            value = guid_field[index % len(guid_field)]
            return str(value)
        if isinstance(guid_field, torch.Tensor):
            try:
                return str(guid_field[index].item())
            except Exception:  # noqa: BLE001
                return "unknown"
        return str(guid_field)

    def on_validation_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_frequency != 0:
            return
        batch = _first_validation_batch(trainer)
        if batch is None:
            return
        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
        forward_module = getattr(pl_module, "model", pl_module)
        y_st = self._extract(batch, "fhr_st")
        y_ph = self._extract(batch, "fhr_ph")
        x_ph = self._extract(batch, "fhr_up_ph")
        target = self._extract(batch, "fhr")
        if not isinstance(y_st, torch.Tensor) or not isinstance(y_ph, torch.Tensor) or not isinstance(target, torch.Tensor):
            return
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            outputs = forward_module(y_st=y_st, y_ph=y_ph, x_ph=x_ph)
        if was_training:
            pl_module.train()
        mu_pr = outputs.get("mu_pr")
        logvar_pr = outputs.get("logvar_pr")
        if not isinstance(mu_pr, torch.Tensor):
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
            target_series = np.squeeze(target_np[idx])
            recon_series = np.squeeze(recon_np[idx])
            length = min(target_series.shape[-1], recon_series.shape[-1])
            t_axis = np.arange(length)
            axis.plot(t_axis, target_series[..., :length], label="target", color="tab:blue", linewidth=1.2)
            axis.plot(t_axis, recon_series[..., :length], label="reconstruction", color="tab:orange", linewidth=1.0)
            if std_np is not None:
                std_series = np.squeeze(std_np[idx])[..., :length]
                axis.fill_between(
                    t_axis,
                    recon_series[..., :length] - 2 * std_series,
                    recon_series[..., :length] + 2 * std_series,
                    color="tab:orange",
                    alpha=0.2,
                )
            axis.set_ylabel(f"Example {idx + 1}")
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


class PlottingCallBack(Callback):
    """Simple qualitative figure for the first validation batch."""

    def __init__(self, output_dir: Path | str, plot_frequency: int = 5):
        super().__init__()
        self.output_dir = Path(output_dir) / "analysis_plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_frequency = max(1, int(plot_frequency))

    @staticmethod
    def _get(batch, name):
        if isinstance(batch, dict):
            return batch.get(name)
        return getattr(batch, name, None)

    @staticmethod
    def _guid_from_batch(guid_field, index: int = 0) -> str:
        if guid_field is None:
            return "unknown"
        if isinstance(guid_field, (list, tuple)):
            if not guid_field:
                return "unknown"
            return str(guid_field[index % len(guid_field)])
        if isinstance(guid_field, torch.Tensor):
            try:
                return str(guid_field[index].item())
            except Exception:  # noqa: BLE001
                return "unknown"
        return str(guid_field)

    def on_validation_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_frequency != 0:
            return
        batch = _first_validation_batch(trainer)
        if batch is None:
            return
        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
        y_st = self._get(batch, "fhr_st")
        y_ph = self._get(batch, "fhr_ph")
        x_ph = self._get(batch, "fhr_up_ph")
        y_raw = self._get(batch, "fhr")
        up_raw = self._get(batch, "up")
        guid_field = self._get(batch, "guid")
        if not all(isinstance(t, torch.Tensor) for t in (y_st, y_ph, x_ph, y_raw, up_raw)):
            return
        model = getattr(pl_module, "model", pl_module)
        with torch.no_grad():
            outputs = model(y_st=y_st, y_ph=y_ph, x_ph=x_ph)
        mu_pr = outputs.get("mu_pr")
        logvar_pr = outputs.get("logvar_pr")
        latent_z = outputs.get("z")
        if not all(isinstance(t, torch.Tensor) for t in (mu_pr, logvar_pr, latent_z)):
            return
        guid = self._guid_from_batch(guid_field)
        self._plot_sample(y_raw, up_raw, mu_pr, logvar_pr, latent_z, epoch, guid)

    def _plot_sample(self, y_raw, up_raw, mu_pr, logvar_pr, latent_z, epoch: int, guid: str) -> None:
        idx = 0
        y = y_raw[idx].detach().cpu().numpy()
        up = up_raw[idx].detach().cpu().numpy()
        recon = mu_pr[idx].detach().cpu().numpy()
        logvar = logvar_pr[idx].detach().cpu().numpy()
        latent = latent_z[idx].detach().cpu().numpy()
        std = np.sqrt(np.exp(logvar))
        time_axis = np.arange(y.shape[-1]) / 4.0

        colors = {
            "fhr": "#055C9A",
            "up": "#0DD8A2",
            "gt": "#456882",
            "recon": "#BB3E00",
            "uncertainty": "#F7AD45",
            "background": "#F9F3EF",
        }

        plt.style.use("default")
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 11,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "axes.edgecolor": "#A2B9A7",
                "axes.linewidth": 1.0,
                "figure.facecolor": "#F7F7F7",
            }
        )
        fig, ax = plt.subplots(4, 2, figsize=(16, 12), gridspec_kw={"height_ratios": [1.3, 1.3, 1.3, 1.6]})
        fig.subplots_adjust(hspace=0.35, wspace=0.25)
        fig.patch.set_facecolor("#F7F7F7")
        for axis_row in ax:
            for axis in axis_row:
                axis.set_facecolor(colors["background"])
                axis.grid(color="#C6D8D3", linestyle="--", linewidth=0.5, alpha=0.4)
                axis.tick_params(axis="x", colors="#666666")
                axis.tick_params(axis="y", colors="#666666")

        Fs = 4
        zoom = Fs * 120

        ax[0, 0].plot(time_axis, y, label="FHR signal (target)", color=colors["fhr"], linewidth=1.2)
        ax[0, 0].plot(time_axis, up, label="UP signal", color=colors["up"], linewidth=1.0, alpha=0.8)
        ax[0, 0].set_ylabel("Normalized amplitude")
        ax[0, 0].set_title("Input FHR and UP signals", pad=12)
        ax[0, 0].legend(loc="upper right", framealpha=0.9)

        ax[0, 1].plot(time_axis[:zoom], y[:zoom], color=colors["gt"], linewidth=1.2)
        ax[0, 1].set_title("FHR signal (first 2 minutes)", pad=12)
        ax[0, 1].set_ylabel("Amplitude")

        ax[1, 0].plot(time_axis, recon, color=colors["recon"], linewidth=1.2, label="Model mean reconstruction")
        ax[1, 0].fill_between(time_axis, recon - 2 * std, recon + 2 * std, color=colors["uncertainty"], alpha=0.35, label="±2σ")
        ax[1, 0].set_ylabel("Amplitude")
        ax[1, 0].set_title("Model mean reconstruction with uncertainty", pad=12)
        ax[1, 0].legend(loc="upper right", framealpha=0.9)

        ax[1, 1].plot(time_axis[:zoom], recon[:zoom], color=colors["recon"], linewidth=1.2)
        ax[1, 1].fill_between(time_axis[:zoom], (recon - 2 * std)[:zoom], (recon + 2 * std)[:zoom], color=colors["uncertainty"], alpha=0.35)
        ax[1, 1].set_title("Reconstruction detail (first 2 minutes)", pad=12)
        ax[1, 1].set_ylabel("Amplitude")

        ax[2, 0].plot(time_axis, y, label="Ground truth", color=colors["gt"], linewidth=1.2)
        ax[2, 0].plot(time_axis, recon, label="Model mean", color=colors["recon"], linewidth=1.2, alpha=0.9)
        ax[2, 0].set_ylabel("Normalized amplitude")
        ax[2, 0].set_title("FHR vs model reconstructions", pad=12)
        ax[2, 0].legend(loc="upper right", framealpha=0.9)
        ax[2, 1].axis("off")

        imgplot = ax[3, 0].imshow(latent.T, aspect="auto", cmap="bwr", origin="lower")
        ax[3, 0].set_ylabel("Latent dimensions")
        ax[3, 0].set_xlabel("Time steps")
        ax[3, 0].set_title("Latent space representation", pad=12)
        fig.colorbar(imgplot, ax=ax[3, 1], fraction=0.046, pad=0.04).set_label("Activation", fontsize=11, color="#666666")

        fig.suptitle(f"Model performance analysis – Epoch {epoch} | guid: {guid}", fontsize=14, y=0.98, color=colors["gt"])
        path = self.output_dir / f"analysis_epoch_{epoch:04d}.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=250)
        plt.close(fig)


__all__ = [
    "LossPlotCallback",
    "HyperparameterLoggingCallback",
    "MetricsLoggingCallback",
    "ReconstructionPlotCallback",
    "PlottingCallBack",
]
