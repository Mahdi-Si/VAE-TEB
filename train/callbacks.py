
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING
from lightning.pytorch.callbacks import Callback
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from loguru import logger
from pathlib import Path
import numpy as np
import matplotlib
import fnmatch
import torch

from plotly.subplots import make_subplots
import plotly.graph_objects as go

if TYPE_CHECKING:
    from lightning.pytorch.loggers import MLFlowLogger

matplotlib.use("Agg")

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
        self.output_dir = Path(output_dir)
        self.plot_frequency = max(1, int(plot_frequency))
        self.max_history_size = max(1, int(max_history_size))
        self._mlflow_logger = mlflow_logger
        self.history: Dict[str, List[float]] = {"epoch": []}
        self.metric_filters: Tuple[str, ...] = tuple(metric_filters or ("train/*", "val/*"))
        self.hyperparam_keys: Tuple[str, ...] = tuple(
            hyperparam_keys or ("hyperparams/beta", "hyperparams/lr", "hyperparams/kld_beta", "lr", "learning_rate", "kld_beta", "beta")
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
        self._mlflow_logger.experiment.log_artifact(self._mlflow_logger.run_id, str(path))

    def on_validation_epoch_end(self, trainer, pl_module):
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

    def __init__(
        self,
        tracked_keys: Optional[Iterable[str]] = None,
        *,
        output_dir: Path | str,
        plot_frequency: int = 10,
    ) -> None:
        super().__init__()
        self.tracked_keys: Tuple[str, ...] = tuple(
            tracked_keys or ("hyperparams/beta", "kld_beta", "hyperparams/lr", "lr", "learning_rate")
        )
        self.history: Dict[str, List[float]] = {"epoch": []}
        for key in self.tracked_keys:
            self.history[key] = []
        self.output_dir = Path(output_dir)
        self.plot_frequency = max(1, int(plot_frequency))
        self._plotly_available = True

    @staticmethod
    def _current_lr(trainer) -> Optional[float]:
        optimizers = getattr(trainer, "optimizers", None)
        if not optimizers:
            return None
        optimizer = optimizers[0]
        if not optimizer.param_groups:
            return None
        return float(optimizer.param_groups[0].get("lr", 0.0))

    def on_train_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        if not trainer.is_global_zero:
            return
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        self.history["epoch"].append(int(epoch))
        lr_value = self._current_lr(trainer)
        for key in self.tracked_keys:
            value = metrics.get(key)
            if value is None and lr_value is not None and "lr" in key:
                value = lr_value
            self.history[key].append(_metric_to_float(value))
        if (epoch + 1) % self.plot_frequency == 0:
            self.plot_hyperparameters()

    def plot_hyperparameters(self) -> None:
        if not self._plotly_available:
            return
        if not self.history["epoch"]:
            return
        try:
            import plotly.graph_objects as go
        except ImportError:
            self._plotly_available = False
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
        path = self.output_dir / f"hyperparameters.html"
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
    """Reproduce the original SeqVAE qualitative plotting callback."""

    def __init__(self, output_dir: Path | str, plot_every_epoch: int = 5, input_channel_num: int = 0):
        super().__init__()
        self.output_dir = Path(output_dir) / "analysis_plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_every_epoch = max(1, int(plot_every_epoch))
        self.input_channel_num = input_channel_num
        self._plotly_available = True

    @staticmethod
    def _get(batch, name):
        if isinstance(batch, dict):
            return batch.get(name)
        return getattr(batch, name, None)

    @staticmethod
    def _guid(batch, index: int = 0) -> str:
        guid_field = None
        if isinstance(batch, dict):
            guid_field = batch.get("guid")
        elif hasattr(batch, "guid"):
            guid_field = getattr(batch, "guid")
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
        if (epoch + 1) % self.plot_every_epoch != 0:
            return

        batch = _first_validation_batch(trainer)
        if batch is None:
            logger.warning("PlottingCallBack: could not fetch validation batch")
            return

        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
        module = getattr(pl_module, "model", pl_module)

        pl_module.eval()
        try:
            with torch.no_grad():
                y_st, y_ph, x_ph = batch.fhr_st, batch.fhr_ph, batch.fhr_up_ph
                y_raw = batch.fhr
                up_raw = batch.up
                outputs = module(y_st, y_ph, x_ph)
                latent_z = outputs["z"]
                mu_pr = outputs["mu_pr"]
                logvar_pr = outputs["logvar_pr"]
            guid = self._guid(batch)
            self._plot_results(
                y_raw,
                up_raw,
                mu_pr,
                logvar_pr,
                latent_z,
                epoch,
                guid,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"PlottingCallBack failed: {exc}")
        finally:
            pl_module.train()

    def _plot_results(
        self,
        y_raw_normalized,
        up_raw_normalized,
        mu_pr,
        logvar_pr,
        latent_z,
        epoch: int,
        guid: str,
    ) -> None:
        batch_idx = 0
        y_raw = y_raw_normalized[batch_idx].detach().cpu().numpy()
        up_raw = up_raw_normalized[batch_idx].detach().cpu().numpy()
        mu_samples = mu_pr[batch_idx].detach().cpu().numpy()
        logvar_samples = logvar_pr[batch_idx].detach().cpu().numpy()
        z_latent = latent_z[batch_idx].detach().cpu().numpy()
        time_axis = np.arange(0, len(y_raw)) / 4.0

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
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 11,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "axes.linewidth": 0.7,
                "axes.edgecolor": "#9E9D9D",
                "axes.facecolor": colors["background"],
                "grid.color": "#838383",
                "grid.linewidth": 0.4,
                "grid.alpha": 0.6,
                "figure.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.dpi": 300,
            }
        )

        n_rows = 4
        fig, ax = plt.subplots(
            nrows=n_rows,
            ncols=2,
            figsize=(20, n_rows * 3.5),
            gridspec_kw={"width_ratios": [80, 1]},
            constrained_layout=True,
        )
        for i in range(n_rows):
            ax[i, 0].grid(True, linestyle="-", alpha=0.4, linewidth=0.4, color="#D2C1B6")
            ax[i, 0].grid(True, which="minor", linestyle=":", alpha=0.25, linewidth=0.3, color="#D2C1B6")
            ax[i, 0].minorticks_on()
            ax[i, 0].set_axisbelow(True)
            ax[i, 0].spines["top"].set_visible(False)
            ax[i, 0].spines["right"].set_visible(False)
            ax[i, 0].spines["left"].set_color("#A2B9A7")
            ax[i, 0].spines["bottom"].set_color("#A2B9A7")
            ax[i, 0].spines["left"].set_linewidth(0.7)
            ax[i, 0].spines["bottom"].set_linewidth(0.7)
            ax[i, 1].set_axis_off()

        ax[0, 0].plot(time_axis, y_raw, linewidth=1.2, color=colors["fhr"], label="FHR", alpha=0.85)
        ax[0, 0].plot(time_axis, up_raw, linewidth=1.2, color=colors["up"], label="UP", alpha=0.85)
        ax[0, 0].set_ylabel("Amplitude")
        ax[0, 0].set_title("Raw FHR and UP Signals", pad=12)
        ax[0, 0].legend(loc="upper right", framealpha=0.95)

        ax[1, 0].plot(time_axis, y_raw, linewidth=1.5, color=colors["gt"], label="Ground Truth", alpha=0.85, zorder=3)
        ax[1, 0].plot(time_axis, mu_samples, linewidth=1.5, color=colors["recon"], label="Reconstruction", alpha=0.85, zorder=2)
        std_dev = np.exp(0.5 * logvar_samples)
        ax[1, 0].fill_between(
            time_axis,
            mu_samples - std_dev,
            mu_samples + std_dev,
            alpha=0.3,
            color=colors["uncertainty"],
            label="Uncertainty (±1σ)",
            zorder=1,
        )
        ax[1, 0].set_ylabel("FHR (bpm)")
        ax[1, 0].set_title("FHR Reconstruction with Uncertainty", pad=12)
        ax[1, 0].legend(loc="upper right", framealpha=0.95)

        ax[2, 0].plot(time_axis, y_raw, linewidth=1.5, color=colors["gt"], label="Ground Truth", alpha=0.85, zorder=2)
        ax[2, 0].plot(time_axis, mu_samples, linewidth=1.5, color=colors["samples"], label="Model Prediction", alpha=0.85, zorder=1)
        ax[2, 0].set_ylabel("FHR (bpm)")
        ax[2, 0].set_title("FHR vs Model Reconstructions", pad=12)
        ax[2, 0].legend(loc="upper right", framealpha=0.95)

        imgplot = ax[3, 0].imshow(z_latent.T, aspect="auto", cmap="bwr", origin="lower")
        ax[3, 0].set_ylabel("Latent Dimensions")
        ax[3, 0].set_xlabel("Time Steps")
        ax[3, 0].set_title("Latent Space Representation", pad=12)
        ax[3, 1].set_axis_on()
        cbar = fig.colorbar(imgplot, cax=ax[3, 1])
        cbar.ax.tick_params(labelsize=10, colors="#666666")
        cbar.set_label("Activation", fontweight="normal", fontsize=11, color="#666666")

        fig.suptitle(f"Model Performance Analysis — Epoch {epoch} | guid: {guid}", fontsize=14, y=0.97, color="#456882")
        save_path = self.output_dir / f"model_results_epoch_{epoch:04d}.pdf"
        fig.savefig(save_path, bbox_inches="tight", orientation="landscape", dpi=300, facecolor="white", edgecolor="none")
        plt.close(fig)



class PlottingAvgPredCallBack(Callback):
    """Plot averaged raw prediction (post-warmup) versus ground truth with uncertainty, latent, and UP."""

    def __init__(self, output_dir: Path | str, plot_every_epoch: int = 5, input_channel_num: int = 0):
        super().__init__()
        self.output_dir = Path(output_dir) / "analysis_plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_every_epoch = max(1, int(plot_every_epoch))
        self.input_channel_num = input_channel_num
        self._plotly_available = True

    @staticmethod
    def _get(batch, name):
        if isinstance(batch, dict):
            return batch.get(name)
        return getattr(batch, name, None)

    @staticmethod
    def _guid(batch, index: int = 0) -> str:
        guid_field = None
        if isinstance(batch, dict):
            guid_field = batch.get("guid")
        elif hasattr(batch, "guid"):
            guid_field = getattr(batch, "guid")
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
        if (epoch + 1) % self.plot_every_epoch != 0:
            return

        batch = _first_validation_batch(trainer)
        if batch is None:
            logger.warning("PlottingCallBack: could not fetch validation batch")
            return

        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
        module = getattr(pl_module, "model", pl_module)

        pl_module.eval()
        try:
            with torch.no_grad():
                y_st, y_ph, x_ph = batch.fhr_st, batch.fhr_ph, batch.fhr_up_ph
                y_raw = batch.fhr
                up_raw = getattr(batch, "up", None)
                outputs = module(y_st, y_ph, x_ph)
                mu_pr = outputs["mu_pr"]
                logvar_pr = outputs.get("logvar_pr")
                avg_pred = module.average_raw_prediction(mu_pr)
                avg_std = None
                if isinstance(logvar_pr, torch.Tensor):
                    std_pr = torch.exp(0.5 * logvar_pr)
                    avg_std = self._average_segments(
                        std_pr,
                        decimation_factor=getattr(module, "decimation_factor", 16),
                        warmup=getattr(module, "warmup_period", 0),
                        raw_len=y_raw.shape[-1],
                    )
                linear_output = outputs.get("linear_output")
                latent_z = outputs.get("z")
            guid = self._guid(batch)
            self._plot_results(
                y_raw,
                up_raw,
                avg_pred,
                avg_std,
                linear_output,
                latent_z,
                epoch,
                guid,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"PlottingCallBack failed: {exc}")
        finally:
            pl_module.train()

    @staticmethod
    def _average_segments(segments: torch.Tensor, decimation_factor: int, warmup: int, raw_len: int) -> torch.Tensor:
        """Average per-step horizon segments onto the raw axis (mirrors average_raw_prediction)."""
        if segments.dim() != 3:
            return torch.full((segments.size(0), raw_len), float("nan"), device=segments.device, dtype=segments.dtype)
        B, T, H = segments.shape
        s = int(decimation_factor)
        warmup = int(warmup)
        device = segments.device
        dtype = segments.dtype

        avg = torch.full((B, raw_len), float("nan"), device=device, dtype=dtype)
        max_valid_t = max(0, min(T, (raw_len - H) // s + 1))
        start_t = min(warmup, max_valid_t)
        if start_t >= max_valid_t:
            return avg
        valid_steps = torch.arange(start_t, max_valid_t, device=device)
        start_idx = valid_steps[:, None] * s
        h_idx = torch.arange(H, device=device)[None, :]
        idx = start_idx + h_idx  # (T_valid, H)
        mask = (idx < raw_len).to(dtype)
        idx_clamped = idx.clamp(max=raw_len - 1)

        pred_sum = torch.zeros((B, raw_len), device=device, dtype=dtype)
        count = torch.zeros((B, raw_len), device=device, dtype=dtype)

        flat_idx = idx_clamped.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
        flat_seg = segments[:, valid_steps, :].reshape(B, -1)
        flat_mask = mask.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)

        pred_sum.scatter_add_(1, flat_idx, flat_seg * flat_mask)
        count.scatter_add_(1, flat_idx, flat_mask)

        avg = pred_sum / count.clamp_min(1.0)
        avg = avg.masked_fill(count == 0, float("nan"))
        return avg

    def _plot_results(
        self,
        y_raw_normalized,
        up_raw_normalized,
        avg_pred,
        avg_std,
        linear_output,
        latent_z,
        epoch: int,
        guid: str,
    ) -> None:
        batch_idx = 0
        y_raw = y_raw_normalized[batch_idx].detach().cpu().numpy()
        up_raw = None
        if isinstance(up_raw_normalized, torch.Tensor):
            up_raw = up_raw_normalized[batch_idx].detach().cpu().numpy()
        avg_pred_np = avg_pred[batch_idx].detach().cpu().numpy()
        avg_std_np = None
        if isinstance(avg_std, torch.Tensor):
            avg_std_np = avg_std[batch_idx].detach().cpu().numpy()
        linear_np = None
        if isinstance(linear_output, torch.Tensor):
            linear_np = linear_output[batch_idx].detach().cpu().numpy()
        latent_np = None
        if isinstance(latent_z, torch.Tensor):
            latent_np = latent_z[batch_idx].detach().cpu().numpy()
        time_axis = np.arange(0, len(y_raw)) / 4.0

        colors = {
            "fhr": "#055C9A",
            "avg": "#BB3E00",
            "up": "#0DD8A2",
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
                "axes.linewidth": 0.7,
                "axes.edgecolor": "#9E9D9D",
                "axes.facecolor": colors["background"],
                "grid.color": "#838383",
                "grid.linewidth": 0.4,
                "grid.alpha": 0.6,
                "figure.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.dpi": 300,
            }
        )

        fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=False)

        ax0 = axes[0]
        ax0.grid(True, linestyle="-", alpha=0.4, linewidth=0.4, color="#D2C1B6")
        ax0.grid(True, which="minor", linestyle=":", alpha=0.25, linewidth=0.3, color="#D2C1B6")
        ax0.minorticks_on()
        ax0.plot(time_axis, y_raw, linewidth=1.2, color=colors["fhr"], label="FHR", alpha=0.85)
        if up_raw is not None:
            ax0.plot(time_axis, up_raw, linewidth=1.0, color=colors["up"], label="UP", alpha=0.8)
        ax0.set_ylabel("Amplitude")
        ax0.set_title("Raw FHR/UP")
        ax0.legend(loc="upper right", framealpha=0.95)

        ax1 = axes[1]
        ax1.grid(True, linestyle="-", alpha=0.4, linewidth=0.4, color="#D2C1B6")
        ax1.grid(True, which="minor", linestyle=":", alpha=0.25, linewidth=0.3, color="#D2C1B6")
        ax1.minorticks_on()
        ax1.plot(time_axis, y_raw, linewidth=1.2, color=colors["fhr"], label="Ground Truth", alpha=0.85)
        ax1.plot(time_axis, avg_pred_np, linewidth=1.0, color=colors["avg"], label="Avg Prediction", alpha=0.9)
        if avg_std_np is not None:
            ax1.fill_between(
                time_axis,
                avg_pred_np - 2 * avg_std_np,
                avg_pred_np + 2 * avg_std_np,
                color=colors["uncertainty"],
                alpha=0.25,
                label="Uncertainty (±2σ)",
            )
        ax1.set_ylabel("Amplitude")
        ax1.set_xlabel("Time (s)")
        ax1.set_title("Average Raw Prediction (post-warmup)")
        ax1.legend(loc="upper right", framealpha=0.95)

        ax2 = axes[2]
        if latent_np is not None:
            im = ax2.imshow(latent_np.T, aspect="auto", cmap="bwr", origin="lower")
            ax2.set_ylabel("Latent dims")
            ax2.set_xlabel("Time steps (decimated)")
            ax2.set_title("Latent Representation")
            fig.colorbar(im, ax=ax2, fraction=0.015, pad=0.02)
        else:
            ax2.set_visible(False)

        ax3 = axes[3]
        if linear_np is not None:
            im2 = ax3.imshow(linear_np.T, aspect="auto", cmap="viridis", origin="lower")
            ax3.set_ylabel("Linear Output Ch")
            ax3.set_xlabel("Time steps (decimated)")
            ax3.set_title("Linear Output (B,T,40)")
            fig.colorbar(im2, ax=ax3, fraction=0.015, pad=0.02)
        else:
            ax3.set_visible(False)

        fig.suptitle(f"Avg Prediction — Epoch {epoch} | guid: {guid}", fontsize=12, y=0.98, color="#456882")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save_path = self.output_dir / f"avg_prediction_epoch_{epoch:04d}.pdf"
        fig.savefig(save_path, bbox_inches="tight", orientation="landscape", dpi=300, facecolor="white", edgecolor="none")
        plt.close(fig)

__all__ = [
    "LossPlotCallback",
    "HyperparameterLoggingCallback",
    "MetricsLoggingCallback",
    "ReconstructionPlotCallback",
    "PlottingCallBack",
    "PlottingAvgPredCallBack",
]
