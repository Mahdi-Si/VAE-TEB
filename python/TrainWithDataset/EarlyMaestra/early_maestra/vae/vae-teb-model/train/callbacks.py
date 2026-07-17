
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING
from lightning.pytorch.callbacks import Callback
import plotly.graph_objects as go
from loguru import logger
from pathlib import Path
import numpy as np
import fnmatch
import os
import torch

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# The rank-0 artifact seam is a leaf helper with no framework coupling, so it lives in
# utils/ and is shared with the SeqVAE plotters. train/ -> utils/ is the allowed
# direction; see train/tests/test_layering.py.
from utils.mlflow_utils import log_artifact_to_mlflow

if TYPE_CHECKING:
    from lightning.pytorch.loggers import MLFlowLogger


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
        self.plot_losses(trainer)
        self.plot_hyperparameters(trainer)

    def plot_losses(self, trainer=None) -> None:
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
        log_artifact_to_mlflow(self._mlflow_logger, path, trainer)

    def plot_hyperparameters(self, trainer=None) -> None:
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
        log_artifact_to_mlflow(self._mlflow_logger, path, trainer)


class HyperparameterLoggingCallback(Callback):
    """Track arbitrary hyper-parameters (learning rates, annealing weights, etc.)."""

    def __init__(
        self,
        tracked_keys: Optional[Iterable[str]] = None,
        *,
        output_dir: Path | str,
        plot_frequency: int = 10,
        mlflow_logger: "MLFlowLogger" | None = None,
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
        self._mlflow_logger = mlflow_logger
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
            self.plot_hyperparameters(trainer)

    def plot_hyperparameters(self, trainer=None) -> None:
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
        log_artifact_to_mlflow(self._mlflow_logger, path, trainer)


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


#: Metric names ``LightningModelBase`` emits **without** a stage prefix. Every other name it
#: logs is framed as ``{stage}/{name}`` (``pl_model_base.py:556``), so a tracked name that is
#: bare and not in this tuple can never match a key in ``trainer.callback_metrics`` -- it
#: produces an all-NaN column rather than an error. ``lr`` is here because
#: ``_log_learning_rate`` logs that literal key (``pl_model_base.py:547``).
_BARE_METRIC_KEYS: Tuple[str, ...] = ("lr",)


def _unreachable_metric_names(names: Iterable[str]) -> Tuple[str, ...]:
    """Return the tracked names the framework can never emit.

    Args:
        names: Metric names a consumer asked to track.

    Returns:
        Those which are unprefixed and not framework-bare, in input order.
    """
    return tuple(n for n in names if "/" not in n and n not in _BARE_METRIC_KEYS)


class MetricsHistoryCsvCallback(Callback):
    """Write :class:`MetricsLoggingCallback`'s in-memory history to a CSV.

    ``MetricsLoggingCallback`` accumulates ``self.history`` and never writes it anywhere, so a
    run's metric history dies with the process unless something serialises it. This is that
    something. It rewrites the file every validation epoch, so a killed run still leaves the
    history it had reached rather than nothing at all.

    On construction it warns about tracked names the framework can never emit. Those are not an
    error -- the CSV is still written -- but they are silent otherwise: the column appears, and
    it is NaN for every epoch of every run. Naming a metric ``kld_beta`` rather than
    ``train/kld_beta`` is the way to make that happen (see :data:`_BARE_METRIC_KEYS`).
    """

    def __init__(self, source: MetricsLoggingCallback, output_dir: str) -> None:
        """Initialize.

        Args:
            source: The ``MetricsLoggingCallback`` whose history to serialise.
            output_dir: Directory receiving ``metrics_history.csv``. Created on first write.
        """
        super().__init__()
        self.source = source
        self.output_dir = output_dir
        # The epoch the most recent collected row belongs to, so the fit-end rewrite anchors to the
        # same number the per-epoch rewrite used rather than re-deriving it from a Trainer counter
        # that has already moved past the last epoch by then.
        self._last_epoch: Optional[int] = None
        unreachable = _unreachable_metric_names(source.tracked_metrics)
        if unreachable:
            logger.warning(
                "MetricsHistoryCsvCallback: tracked metrics {} are unprefixed and are not "
                "logged bare by the framework, so their columns will be all-NaN. Prefix them "
                "with a stage, e.g. 'train/{}'.",
                list(unreachable),
                unreachable[0],
            )

    def _write(self, last_epoch: Optional[int] = None) -> Optional[str]:
        """Write the history CSV.

        Args:
            last_epoch: The epoch the final history row belongs to. Earlier rows are numbered
                backwards from it, which is what keeps the column aligned with MLflow and the
                checkpoint filenames even when the source collected rows this writer did not see.
                ``None`` numbers the rows positionally from zero.

        Returns:
            The path written, or ``None`` when the history is empty.
        """
        # Imported here rather than at module scope: pandas is needed only by this writer, and
        # importing train.callbacks must stay cheap for the many consumers that never use it.
        import pandas as pd

        history = self.source.as_dict()
        if not history:
            return None
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "metrics_history.csv")
        frame = pd.DataFrame(history)
        # Anchored to the real epoch, not to the row index. MetricsLoggingCallback has no
        # sanity-check guard, so with num_sanity_val_steps > 0 its first row is the sanity pass --
        # a row this writer never wrote, and one that would otherwise shift every subsequent epoch
        # number by one against MLflow, the checkpoint filenames and the loss plots.
        first_epoch = 0 if last_epoch is None else last_epoch - (len(frame) - 1)
        frame.insert(0, "epoch", range(first_epoch, first_epoch + len(frame)))
        frame.to_csv(path, index=False)
        return path

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        """Rewrite the CSV so a killed run still leaves the history it had."""
        if trainer.is_global_zero and not trainer.sanity_checking:
            self._last_epoch = trainer.current_epoch
            self._write(last_epoch=self._last_epoch)

    def on_fit_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        """Write the final history CSV on rank zero."""
        if trainer.is_global_zero:
            path = self._write(last_epoch=self._last_epoch)
            if path:
                logger.info(f"metrics history written to {path}")


class MLflowRunLoggingCallback(Callback):
    """Log the model architecture and the final trained model to the MLflow run.

    Attached by the trainer builder only when tracking is enabled, so it is model
    agnostic — it relies solely on ``nn.Module`` APIs (``repr`` and ``parameters()``)
    and the portable eager ``orig_model``. Every write goes through the run-bound
    client API, is guarded to rank 0 (duplicate cross-rank writes corrupt a run), and
    is fail-closed: a tracking-server error is warned and swallowed so it can never
    lose a finished training run.
    """

    def __init__(self, mlflow_logger, experiment_tag: str, log_model: bool = False) -> None:
        """
        Args:
            mlflow_logger: The run-bound ``MLFlowLogger`` (``None`` disables the hooks).
            experiment_tag: Registry name the final model is registered under.
            log_model: When ``True``, ``on_fit_end`` logs and registers the final model;
                architecture logging in ``on_fit_start`` always runs when tracking is on.
        """
        super().__init__()
        self._mlflow_logger = mlflow_logger
        self._experiment_tag = experiment_tag
        self._log_model = bool(log_model)

    def _eager_module(self, pl_module):
        """Return the portable eager module (never the ``torch.compile`` wrapper)."""
        return getattr(pl_module, "orig_model", pl_module)

    def on_fit_start(self, trainer, pl_module) -> None:
        """Record the architecture ``repr`` and parameter counts (rank 0, fail-closed)."""
        if self._mlflow_logger is None or not trainer.is_global_zero:
            return
        module = self._eager_module(pl_module)
        run_id = self._mlflow_logger.run_id
        experiment = self._mlflow_logger.experiment
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        try:
            experiment.log_text(run_id, repr(module), "model/model_architecture.txt")
            experiment.log_param(run_id, "model_class", type(module).__name__)
            experiment.log_metric(run_id, "params_total", float(total))
            experiment.log_metric(run_id, "params_trainable", float(trainable))
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow architecture logging failed (continuing): {}", exc)

    def on_fit_end(self, trainer, pl_module) -> None:
        """Log + register the final eager model on the training run (rank 0, fail-closed).

        The model is logged **without** a signature: signature inference would have to
        run the model on its multi-tensor keyword batch, which ``infer_signature``
        cannot represent. The logged ``state_dict`` carries no ``_orig_mod.`` prefix
        (it is the eager module), so it reloads via ``mlflow.pytorch.load_model``.
        """
        if self._mlflow_logger is None or not trainer.is_global_zero or not self._log_model:
            return
        module = self._eager_module(pl_module)
        run_id = self._mlflow_logger.run_id
        try:
            import mlflow

            # Bind to the training run: MLFlowLogger creates the run via the client and
            # never calls start_run, so a bare log_model would orphan into a new run.
            with mlflow.start_run(run_id=run_id):
                mlflow.pytorch.log_model(
                    module,
                    name="model",
                    registered_model_name=self._experiment_tag,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow model logging failed (continuing): {}", exc)


__all__ = [
    "LossPlotCallback",
    "HyperparameterLoggingCallback",
    "MetricsLoggingCallback",
    "MetricsHistoryCsvCallback",
    "MLflowRunLoggingCallback",
]
