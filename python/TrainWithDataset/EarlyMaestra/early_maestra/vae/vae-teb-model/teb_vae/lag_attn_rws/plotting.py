r"""Validation-epoch diagnostic figure for the raw-signal lag-attention VAE.

One figure per selected validation sample, from a single forward pass, in seven rows:

1. The raw target trace **in bpm**, with the plotted anchor's forecast window marked.
2. That window zoomed: the true future against the base ($z^p$) and full ($z^q$) forecasts and
   their $\mu \pm 2\sigma$ bands. This is the panel the whole model exists to produce -- the two
   curves are the two predictions whose log-score difference is the coupling readout -- and it is
   the reason the normalization statistics are plumbed this far: a forecast drawn in z-units
   cannot be checked against physiology by eye.
3. $\mu^p_t$ over $\mu^q_t - \mu^p_t$: the target-only latent state, and the additional
   source-derived shift, on one colour scale so their relative size is visible. The design's
   claim is that the second is *small but useful*; a delta as large as the state itself means the
   posterior is doing the prior's job.
4. The per-step per-dimension KL, which is where a collapse into one or two dimensions shows up.
5. The total per-step KL $K_t$.
6. The lag-attention matrix, head-averaged, with its per-step argmax overlaid.
7. The source-conditioned KL attributed across lags, $\widetilde K_{t,\ell}$.

Both lag panels carry a secondary axis in **compensated** seconds -- $4\,(\ell + \delta)$, the
residual physiological lag on the mechanically aligned timeline -- and say so in the label. The
uncorrected sensor-file figure is $20$ s larger and is deliberately not what a figure shows.

The callback never raises into the training loop: generation is wrapped in a broad ``try/except``
that warns and closes any leaked figures, and every saved file goes to MLflow through the rank-0
artifact seam :func:`utils.mlflow_utils.log_artifact_to_mlflow`. This module lives in the model
layer rather than under ``nets/``, so it may depend on matplotlib, Lightning and ``utils``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from loguru import logger

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from teb_vae.lag_attn.figure_primitives import (  # noqa: E402
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_LIGHT_GRAY,
    COLOR_ORANGE,
    COLOR_VERMILLION,
    attach_lag_seconds_axis,
    safe_vabs,
    shade_warmup,
    time_axes,
    to_numpy,
)
from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry  # noqa: E402
from teb_vae.lag_attn_rws.nets.lag_report import (  # noqa: E402
    COMPENSATED_LAG_AXIS_LABEL,
    SECONDS_PER_STEP,
    lag_compensated_seconds,
)
from train.graph_models_utils import denormalize_signal_data  # noqa: E402
from utils.mlflow_utils import log_artifact_to_mlflow  # noqa: E402
from utils.style import (  # noqa: E402
    SAVE_DPI,
    apply_publication_style,
    save_figure,
    style_axes,
)

__all__ = ["LagAttnRwsPlotCallback"]

#: Raw sampling rate of the target signal, in Hz. With ``decimation = 16`` this is the $4$ s
#: decimated step the whole geometry is built on.
_FS_RAW = 4.0

#: Band width drawn around each forecast mean, in standard deviations.
_BAND_SIGMAS = 2.0

#: Vertical strip reserved above the first panel for the two-line title, in inches.
_HEADER_INCHES = 0.75


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


def _in_bpm(
    values: torch.Tensor, normalization_stats: Optional[Dict[str, Any]]
) -> Tuple[np.ndarray, str]:
    """Invert the loader's z-scoring on a target-signal tensor, when the statistics are known.

    Args:
        values: Target-signal values in loader units.
        normalization_stats: The loader's statistics dict, or ``None``.

    Returns:
        ``(array, unit_label)`` -- the values and the unit to put on the axis. The label is the
        honest one: without statistics the figure stays in z-units rather than mislabelling
        them.
    """
    if normalization_stats is not None and "fhr" in normalization_stats:
        return to_numpy(denormalize_signal_data(values, "fhr", normalization_stats)), "bpm"
    return to_numpy(values), "normalised"


# =============================================================================
# Figure builder
# =============================================================================
def build_diagnostic_figure(
    *,
    outs: Dict[str, Any],
    kld_per_dim: torch.Tensor,
    fhr_raw: torch.Tensor,
    geometry: TrimmedRawGeometry,
    sample_index: int,
    epoch: int,
    guid: str,
    beta: float,
    scalars: Dict[str, float],
    normalization_stats: Optional[Dict[str, Any]] = None,
    delay_steps: int = 0,
    forecast_anchor_frac: float = 0.6,
) -> Any:
    r"""Build the seven-row diagnostic figure for one sample.

    Args:
        outs: The model's forward dict.
        kld_per_dim: Per-step per-dimension KL $(B, T, d_z)$, from the model's own
            ``kld_tensor`` so the drawn number and the trained number share one formula.
        fhr_raw: The raw target $(B, L_{\mathrm{raw}})$ in loader units.
        geometry: The model's trimmed-grid geometry.
        sample_index: Which sample of the batch to draw.
        epoch: Current epoch, for the title.
        guid: Recording identifier, for the title.
        beta: The KL weight **resolved for this epoch**, not the raw hyperparameter.
        scalars: Loss readouts for the title (``nll_base_block``, ``nll_full_block``,
            ``pred_gap``, ``source_conditioned_kl_raw``); missing keys are skipped.
        normalization_stats: The loader's statistics, so the target renders in bpm.
        delay_steps: The causal input delay $\delta$, for the compensated lag axes.
        forecast_anchor_frac: Where in the trained-anchor range to place the forecast zoom.

    Returns:
        The matplotlib ``Figure``. The caller saves and closes it.
    """
    apply_publication_style()

    i = int(sample_index)
    t_steps, horizon, raw_per_step = geometry.t, geometry.horizon, geometry.r
    warmup, t_valid = geometry.warmup, geometry.t_valid
    time_raw, time_dec, t_max = time_axes(t_steps, geometry.raw_len, fs_raw=_FS_RAW)

    # The anchor whose forecast is drawn: a fraction into the trained range, so it is never the
    # warm-up edge (untrained) nor the final anchor (whose window ends exactly at the segment
    # end and shows no continuation).
    anchor = warmup + int(round(forecast_anchor_frac * max(0, t_valid - 1 - warmup)))
    anchor = int(min(max(anchor, warmup), t_valid - 1))
    window_start = geometry.future_block_start(anchor)
    window_stop = window_start + horizon * raw_per_step

    fhr_np, unit = _in_bpm(fhr_raw[i], normalization_stats)
    fhr_np = np.asarray(fhr_np).ravel()

    # Both forecasts and both bands, denormalized through the same affine map as the truth, so
    # the three curves in the zoom panel are directly comparable.
    def _forecast(branch: str) -> List[np.ndarray]:
        """Return ``[mean, lower, upper]`` of one branch's forecast at ``anchor``, in ``unit``."""
        mean = outs[f"mu_{branch}"][i, anchor].reshape(-1)
        sigma = torch.exp(0.5 * outs[f"logvar_{branch}"][i, anchor].reshape(-1))
        curves = [mean, mean - _BAND_SIGMAS * sigma, mean + _BAND_SIGMAS * sigma]
        return [_in_bpm(curve, normalization_stats)[0].ravel() for curve in curves]

    base_mean, base_lo, base_hi = _forecast("base")
    full_mean, full_lo, full_hi = _forecast("full")

    mu_prior_np = to_numpy(outs["mu_prior"][i])                       # (T, d_z)
    delta_mu_np = to_numpy(outs["mu_post"][i] - outs["mu_prior"][i])  # (T, d_z)
    kld_dims_np = to_numpy(kld_per_dim[i])                            # (T, d_z)
    kld_total_np = to_numpy(outs["kld_per_t"][i])                     # (T,)
    alpha_np = to_numpy(outs["attn_weights"][i]).mean(axis=1)         # (T, L)
    kl_lag_np = to_numpy(outs["source_kl_lag_map"][i])                # (T, L)
    d_z, n_lags = mu_prior_np.shape[1], alpha_np.shape[1]

    row_specs = [
        ("raw", 0.9),
        ("forecast", 1.1),
        ("latent", 1.2),
        ("kld_dims", 1.1),
        ("kld_total", 0.85),
        ("lag_attn", 1.2),
        ("kl_lag_map", 1.2),
    ]
    height_ratios = [height for _, height in row_specs]
    figure_height = sum(height_ratios) * 2.6
    fig = plt.figure(figsize=(14, figure_height))
    # The two-line suptitle needs a fixed *physical* strip, not a fixed fraction: as a fraction
    # it shrinks with every row added and the title lands on top of the first panel's own.
    header_frac = _HEADER_INCHES / figure_height
    # Two columns: every main axes sits in column 0 so all rows are exactly as wide, with a
    # narrow reserved colorbar column beside it. Line rows hide their unused cax rather than
    # skipping it, which would make those rows wider than the heatmap rows.
    grid = GridSpec(
        len(row_specs), 2, figure=fig,
        height_ratios=height_ratios, width_ratios=[1.0, 0.022],
        left=0.065, right=0.93, top=1.0 - header_frac, bottom=0.03,
        # The gutter before the colorbar column has to hold the lag panels' secondary axis --
        # its ticks *and* its label. Too narrow and matplotlib still draws the label, underneath
        # the colorbar, where the one thing the axis has to say is invisible.
        hspace=0.55, wspace=0.09,
    )
    row_of = {name: index for index, (name, _) in enumerate(row_specs)}

    def row_axes(name: str) -> Tuple[Any, Any]:
        """Return the ``(main, cax)`` axes pair of a named row."""
        index = row_of[name]
        return fig.add_subplot(grid[index, 0]), fig.add_subplot(grid[index, 1])

    def attach_cbar(cax: Any, image: Any, label: str) -> None:
        """Attach a colorbar for ``image`` onto a row's reserved cax."""
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label(label, fontsize=8, color=COLOR_BLACK)
        cbar.ax.tick_params(labelsize=7, colors=COLOR_BLACK)

    def heatmap_spines(ax: Any) -> None:
        """Draw all four spines on a heatmap axes."""
        ax.grid(False)
        for spine in ("top", "bottom", "left", "right"):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color(COLOR_BLACK)
            ax.spines[spine].set_linewidth(0.6)

    def finalise_time_axis(ax: Any) -> None:
        """Pin the shared physical-time axis and shade the untrained warm-up."""
        ax.set_xlim(0.0, t_max)
        shade_warmup(ax, warmup, t_max, t_steps)

    def lag_panel(ax: Any, values: np.ndarray, title: str, cmap: str) -> Any:
        """Draw a ``(T, L)`` lag map with a compensated-seconds secondary axis."""
        image = ax.imshow(
            values.T, aspect="auto", cmap=cmap, origin="lower",
            extent=[0.0, t_max, -0.5, n_lags - 0.5],
        )
        ax.set_title(title, fontsize=9, pad=6)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Lag $\\ell$ (steps)", fontsize=8)
        heatmap_spines(ax)
        secondary = attach_lag_seconds_axis(
            ax,
            step_seconds=SECONDS_PER_STEP,
            # The whole compensation, expressed as the axis offset: lag 0 already sits at
            # 4*delta seconds once the source channels are read delta steps stale.
            delta_up_seconds=float(lag_compensated_seconds(0, delay_steps=delay_steps)),
        )
        if secondary is not None:
            # Overriding the primitive's generic label: which of the two lag quantities is drawn
            # is exactly the thing a reader must not have to guess.
            secondary.set_ylabel(COMPENSATED_LAG_AXIS_LABEL, fontsize=8)
        finalise_time_axis(ax)
        return image

    # ---- Row: the raw target trace ----------------------------------------
    ax, cax = row_axes("raw")
    ax.plot(time_raw, fhr_np, color=COLOR_BLUE, linewidth=0.7)
    ax.axvspan(
        window_start / _FS_RAW, window_stop / _FS_RAW,
        color=COLOR_ORANGE, alpha=0.25, zorder=0,
    )
    ax.set_title(
        f"Raw target FHR — anchor {anchor} forecast window shaded "
        f"(raw [{window_start}, {window_stop}))",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel(f"FHR ({unit})", fontsize=8)
    style_axes(ax, grid="both")
    finalise_time_axis(ax)
    cax.set_visible(False)

    # ---- Row: the forecast window, zoomed ---------------------------------
    ax, cax = row_axes("forecast")
    window_time = time_raw[window_start:window_stop]
    ax.plot(
        window_time, fhr_np[window_start:window_stop],
        color=COLOR_BLACK, linewidth=1.1, label="true $Y^{+}$",
    )
    ax.fill_between(window_time, base_lo, base_hi, color=COLOR_GRAY, alpha=0.22, linewidth=0)
    ax.plot(
        window_time, base_mean, color=COLOR_GRAY, linewidth=1.0, linestyle="--",
        label="base ($z^p$, target-only)",
    )
    ax.fill_between(
        window_time, full_lo, full_hi, color=COLOR_VERMILLION, alpha=0.18, linewidth=0
    )
    ax.plot(
        window_time, full_mean, color=COLOR_VERMILLION, linewidth=1.0,
        label="full ($z^q$, source-conditioned)",
    )
    ax.set_title(
        f"Forecast at anchor {anchor} — mean $\\pm$ {_BAND_SIGMAS:.0f}$\\sigma$ "
        f"({horizon}$\\times${raw_per_step} = {horizon * raw_per_step} raw samples)",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel(f"FHR ({unit})", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.95)
    style_axes(ax, grid="both")
    ax.set_xlim(float(window_time[0]), float(window_time[-1]))
    cax.set_visible(False)

    # ---- Row: prior mean over the source-derived delta ---------------------
    ax, cax = row_axes("latent")
    latent_stack = np.concatenate([mu_prior_np.T, delta_mu_np.T], axis=0)  # (2*d_z, T)
    vabs = safe_vabs(latent_stack)
    image = ax.imshow(
        latent_stack, aspect="auto", cmap="bwr", origin="upper",
        vmin=-vabs, vmax=vabs, extent=[0.0, t_max, 2 * d_z - 0.5, -0.5],
    )
    ax.axhline(d_z - 0.5, color="white", linewidth=1.2, linestyle="--")
    ax.set_yticks([d_z // 2, d_z + d_z // 2])
    ax.set_yticklabels(["$\\mu^p$", "$\\mu^q-\\mu^p$"])
    ax.set_title(
        "Target-only latent state and the source-derived shift (shared colour scale)",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    heatmap_spines(ax)
    attach_cbar(cax, image, "value")
    finalise_time_axis(ax)

    # ---- Row: per-dimension KL --------------------------------------------
    ax, cax = row_axes("kld_dims")
    image = ax.imshow(
        kld_dims_np.T, aspect="auto", cmap="magma", origin="lower",
        extent=[0.0, t_max, -0.5, d_z - 0.5],
    )
    ax.set_title("Per-dimension source-conditioned KL (nats)", fontsize=9, pad=6)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Latent dim", fontsize=8)
    heatmap_spines(ax)
    attach_cbar(cax, image, "nats")
    finalise_time_axis(ax)

    # ---- Row: total KL per step -------------------------------------------
    ax, cax = row_axes("kld_total")
    ax.plot(time_dec, kld_total_np, color=COLOR_VERMILLION, linewidth=0.9)
    # The tail H anchors carry no reconstruction term, so the KL there is untrained and must not
    # be read as coupling fading away at the end of the recording.
    ax.axvspan(
        float(t_valid) * (t_max / float(t_steps)), t_max,
        color=COLOR_LIGHT_GRAY, alpha=0.35, zorder=0,
    )
    ax.set_title(
        "$K_t$ — total source-conditioned KL per step (warm-up and untrained tail shaded)",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("nats", fontsize=8)
    style_axes(ax, grid="both")
    finalise_time_axis(ax)
    cax.set_visible(False)

    # ---- Row: lag attention with its argmax --------------------------------
    ax, cax = row_axes("lag_attn")
    image = lag_panel(ax, alpha_np, "Lag attention (head-averaged) with per-step argmax", "viridis")
    ax.plot(
        time_dec, alpha_np.argmax(axis=1),
        color=COLOR_ORANGE, linewidth=0.7, alpha=0.85,
    )
    attach_cbar(cax, image, "attention")

    # ---- Row: the KL attributed across lags --------------------------------
    ax, cax = row_axes("kl_lag_map")
    image = lag_panel(
        ax, kl_lag_np, "$\\widetilde K_{t,\\ell}$ — source-conditioned KL by lag", "magma"
    )
    attach_cbar(cax, image, "nats")

    readouts = "  ".join(
        f"{name}={float(scalars[name]):.4g}"
        for name in ("nll_base_block", "nll_full_block", "pred_gap", "source_conditioned_kl_raw")
        if name in scalars
    )
    fig.suptitle(
        f"epoch {epoch} — sample {i} — guid {guid} — beta={beta:.4g}\n{readouts}",
        fontsize=10, y=1.0 - 0.1 / figure_height, va="top",
    )
    return fig


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
