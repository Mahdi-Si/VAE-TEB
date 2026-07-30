r"""Validation-epoch diagnostic callback for the raw-signal lag-attention VAE.

The page itself -- seven rows, from a single forward pass -- lives in
:mod:`teb_vae.lag_attn_rws.sample_page` at the package root, because the evaluation draws exactly
the same page and may import neither Lightning nor this module. :func:`build_diagnostic_figure` is
re-exported here so the callback, and every caller and test that reached for
``plotting.build_diagnostic_figure``, are unchanged.

What is left here is the callback: when to draw, which samples, and where the files go. It never
raises into the training loop -- generation is wrapped in a broad ``try/except`` that warns and
closes any leaked figures -- and every saved file goes to MLflow through the rank-0 artifact seam
:func:`utils.mlflow_utils.log_artifact_to_mlflow`. This module lives in the model layer rather
than under ``nets/``, so it may depend on matplotlib, Lightning and ``utils``.

The publication style is applied **once**, when the callback is constructed. It mutates global
``rcParams``, so applying it per figure restyled the whole process on every validation epoch and
made the timing of any other figure's appearance decide how it looked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

import torch
from lightning.pytorch.callbacks import Callback
from loguru import logger

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Re-exported rather than defined here: the evaluation's per-sample pages draw the same figure and
# may not import this module. Bound as a module attribute so ``monkeypatch.setattr(plotting,
# "build_diagnostic_figure", ...)`` still intercepts the callback's call.
from teb_vae.lag_attn_rws.sample_page import build_diagnostic_figure  # noqa: E402,F401
from utils.mlflow_utils import log_artifact_to_mlflow  # noqa: E402
from utils.style import SAVE_DPI, apply_publication_style, save_figure  # noqa: E402

__all__ = ["LagAttnRwsPlotCallback", "build_diagnostic_figure"]


# =============================================================================
# Small helpers
# =============================================================================
def _first_validation_batch(trainer: Any) -> Optional[Any]:
    """Fetch the first batch of the first validation dataloader, or ``None``.

    Args:
        trainer: The Lightning trainer.

    Returns:
        A batch, or ``None`` if no validation loader is attached or it is empty.
    """
    loaders = getattr(trainer, "val_dataloaders", None)
    if loaders is None:
        return None
    loader = loaders[0] if isinstance(loaders, (list, tuple)) else loaders
    try:
        return next(iter(loader))
    except StopIteration:
        return None


def _first_validation_dataset(trainer: Any) -> Optional[Any]:
    """Return the dataset behind the first validation dataloader, or ``None``.

    Args:
        trainer: The Lightning trainer.

    Returns:
        The dataset object, or ``None`` if there is no validation loader.
    """
    loaders = getattr(trainer, "val_dataloaders", None)
    if loaders is None:
        return None
    loader = loaders[0] if isinstance(loaders, (list, tuple)) else loaders
    return getattr(loader, "dataset", None)


def normalization_stats_of(trainer: Any) -> Optional[Dict[str, Any]]:
    """Reach the loader's normalization statistics, for rendering the target in bpm.

    The run's trainer is handed plain dataloaders rather than a datamodule, so the statistics
    are reached through the dataloader's dataset. Failing to find them is not an error: the
    figure falls back to z-units and says so on the axis.

    Args:
        trainer: The Lightning trainer.

    Returns:
        The statistics dict keyed by field name, or ``None``.
    """
    dataset = _first_validation_dataset(trainer)
    getter = getattr(dataset, "get_normalization_stats", None)
    if not callable(getter):
        return None
    try:
        return cast(Optional[Dict[str, Any]], getter())
    except Exception as exc:  # noqa: BLE001 - a diagnostic figure is not worth a failed run
        logger.debug(f"LagAttnRwsPlotCallback: normalization statistics unavailable: {exc}")
        return None


def _get_field(batch: Any, name: str) -> Optional[Any]:
    """Pull ``name`` from a dict batch or an attribute-style batch.

    Args:
        batch: A batch from the data module.
        name: Field name.

    Returns:
        The field, or ``None`` if absent.
    """
    if isinstance(batch, dict):
        return batch.get(name)
    return getattr(batch, name, None)


def _guid_of(batch: Any, index: int = 0) -> str:
    """Extract a printable recording identifier for sample ``index``.

    Args:
        batch: A batch from the data module.
        index: Sample index within the batch.

    Returns:
        The identifier, or ``'unknown'`` when the field is absent or unreadable.
    """
    field = _get_field(batch, "guid")
    if field is None:
        return "unknown"
    if isinstance(field, (list, tuple)):
        return str(field[index % len(field)]) if field else "unknown"
    if isinstance(field, torch.Tensor):
        try:
            return str(field[index].item())
        except Exception:  # noqa: BLE001
            return "unknown"
    return str(field)


def _source_delay_steps(model: Any) -> int:
    """Return the model's causal input delay in decimated steps.

    Zero unless a reach budget has been resolved and applied to the source channels. Read off
    the model rather than assumed, because the reported lag is wrong by exactly this amount when
    a delay is configured and the figure ignores it.

    Args:
        model: The net.

    Returns:
        The delay $\\delta$ in steps.
    """
    return int(getattr(model, "source_delay_steps", 0) or 0)


# =============================================================================
# Callback
# =============================================================================
class LagAttnRwsPlotCallback(Callback):
    """Writes the validation diagnostic figure and routes it to MLflow."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        plot_frequency: int = 1,
        num_examples: int = 2,
        *,
        file_format: str = "pdf",
        mlflow_logger: Any = None,
        forecast_anchor_frac: float = 0.6,
        subdir: str = "lag_attn_rws_diagnostics",
    ) -> None:
        """Initialize the callback.

        Args:
            output_dir: Run results directory; figures land in ``output_dir/subdir``.
            plot_frequency: Plot every this many epochs.
            num_examples: Samples of the first validation batch to draw.
            file_format: Figure extension, e.g. ``'pdf'`` or ``'png'``.
            mlflow_logger: The run's MLflow logger, or ``None`` to skip artifact upload.
            forecast_anchor_frac: Where in the trained-anchor range to place the forecast zoom.
            subdir: Subdirectory name under ``output_dir``.
        """
        super().__init__()
        # Once, here, rather than inside the builder. It mutates global rcParams, so a per-figure
        # call restyled the whole process on every validation epoch -- and made how any other
        # figure in the run looked depend on whether this callback had drawn yet.
        apply_publication_style()
        self.output_dir = Path(output_dir) / subdir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_frequency = max(1, int(plot_frequency))
        self.num_examples = max(1, int(num_examples))
        self.file_format = file_format.lower().lstrip(".")
        self.forecast_anchor_frac = float(forecast_anchor_frac)
        self._mlflow_logger = mlflow_logger

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Draw the figures for this epoch, if this rank and epoch should.

        Args:
            trainer: The Lightning trainer.
            pl_module: The task.
        """
        if not getattr(trainer, "is_global_zero", True):
            return
        # The sanity pass runs before epoch 0 and would write a figure numbered like a real
        # epoch's.
        if getattr(trainer, "sanity_checking", False):
            return
        epoch = int(getattr(trainer, "current_epoch", 0))
        if (epoch + 1) % self.plot_frequency != 0:
            return
        # The fetch is inside the guard, not before it: it builds a fresh iterator over the
        # validation loader, whose worker processes and HDF5 handles can raise anything at all,
        # and an exception escaping here would abort a multi-day fit for the sake of a figure.
        try:
            batch = _first_validation_batch(trainer)
            if batch is None:
                logger.debug("LagAttnRwsPlotCallback: no validation batch available.")
                return
            self._generate_plots(trainer, batch, pl_module, epoch)
        except Exception as exc:  # noqa: BLE001 - a figure is never worth failing a fit for
            plt.close("all")
            logger.warning(f"LagAttnRwsPlotCallback failed: {exc}")

    @torch.no_grad()
    def _generate_plots(self, trainer: Any, batch: Any, pl_module: Any, epoch: int) -> None:
        """Run one forward pass and write one figure per requested sample.

        The streams are assembled through the task's own builders and the loss through the net's
        own ``compute_loss``, so a figure cannot quietly disagree with the objective it
        illustrates about what the model was fed or what it scored.

        Args:
            trainer: The Lightning trainer.
            batch: The validation batch to draw from.
            pl_module: The task.
            epoch: The current epoch.
        """
        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
        # orig_model, not model: the latter may be a compiled wrapper without the net's methods.
        model = pl_module.orig_model

        y_st, y_ph = pl_module._build_target_streams(batch)
        u_stream = pl_module._build_source_stream(batch)
        fhr_raw, weight = pl_module._build_raw_target(batch)

        was_training = pl_module.training
        pl_module.eval()
        try:
            outs = model(y_st, y_ph, u_stream)
            # The schedule's value for this epoch, not hparams['kld_beta']: under any warm-up the
            # raw hyperparameter is the endpoint and the figure would report a constant.
            beta = float(pl_module._resolve_beta(pl_module.current_epoch))
            scalars = model.compute_loss(
                outs,
                fhr_raw,
                weight=weight,
                beta=beta,
                lambda_full=float(pl_module.hparams.get("lambda_full", 1.0)),
                lambda_base=float(pl_module.hparams.get("lambda_base", 1.0)),
                likelihood=str(pl_module.hparams.get("likelihood", "gaussian_nll")),
                free_bits=float(pl_module.hparams.get("free_bits", 0.0)),
            )["metrics"]
            kld_per_dim = model.kld_tensor(
                mu_prior=outs["mu_prior"],
                logvar_prior=outs["logvar_prior"],
                mu_post=outs["mu_post"],
                logvar_post=outs["logvar_post"],
            )
        finally:
            if was_training:
                pl_module.train()

        stats = normalization_stats_of(trainer)
        for index in range(min(self.num_examples, int(y_st.shape[0]))):
            guid = _guid_of(batch, index)
            figure = build_diagnostic_figure(
                outs=outs,
                kld_per_dim=kld_per_dim,
                fhr_raw=fhr_raw,
                geometry=model.geometry,
                sample_index=index,
                epoch=epoch,
                guid=guid,
                beta=beta,
                scalars={name: float(value) for name, value in scalars.items()},
                normalization_stats=stats,
                delay_steps=_source_delay_steps(model),
                forecast_anchor_frac=self.forecast_anchor_frac,
            )
            path = self.output_dir / (
                f"lag_attn_rws_epoch{epoch:04d}_sample{index}_{guid[:16]}.{self.file_format}"
            )
            save_figure(figure, path, dpi=SAVE_DPI, close=True)
            log_artifact_to_mlflow(self._mlflow_logger, path, trainer)
