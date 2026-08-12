r"""The seven-row per-sample diagnostic page, and nothing else.

One figure per sample, from a single forward pass:

1. The raw target FHR **in bpm** and the raw source UP **in mmHg**, on one time axis with one
   y-axis each. UP shares the row rather than getting one of its own because rows 6 and 7 are
   statements *about* this trace: a contraction has to be findable in the same column of the page
   as the response it is claimed to drive.
2. The forecast over the whole recording, tiled into **consecutive non-overlapping** windows: the
   true future against the base ($z^p$) and full ($z^q$) forecasts and their $\mu \pm 2\sigma$
   bands, with a thin dashed vertical at every window edge. This is the panel the whole model
   exists to produce -- the two curves are the two predictions whose log-score difference is the
   coupling readout -- and it is the reason the normalization statistics are plumbed this far: a
   forecast drawn in z-units cannot be checked against physiology by eye. The tiling is this
   module's alone; see :func:`build_diagnostic_figure`.
3. $\mu^p_t$ over $\mu^q_t - \mu^p_t$: the target-only latent state, and the additional
   source-derived shift, on one colour scale so their relative size is visible. The design's
   claim is that the second is *small but useful*; a delta as large as the state itself means the
   posterior is doing the prior's job.
4. The per-step per-dimension KL, which is where a collapse into one or two dimensions shows up.
5. The total per-step KL $K_t$.
6. The lag-attention matrix, head-averaged, with its per-step argmax overlaid.
7. The source-conditioned KL attributed across lags, $\widetilde K_{t,\ell}$.

Between rows $2$ and $3$ the page draws **one optional row per gated input stream** -- the target
and the source exactly as the encoders receive them, surviving channels only, each already shifted
by its own causal delay. They are what the seven rows below are conditioned on, and they are drawn
from :class:`InputStreamPanel` objects the caller supplies; a caller that supplies none gets the
seven-row page unchanged. See :mod:`teb_vae.lag_attn_rws.input_budget`, which builds them and which
also draws the run-level companion figure: the same channels against the forecast window, which is
where "what may this model be asked to predict" is answered.

**Rows 3-7 are drawn over the trained anchors only.** The warm-up prefix $[0, w)$ is excluded from
every one of them, and the tail $[T - H, T)$ from all but the attention: those columns are not
merely uninteresting, they carry no gradient at all -- the tail is neither decoded nor inside the
KL support -- and while they stayed in the arrays they set the colour scale, so a warm-up transient
compressed the whole trained region into the bottom of the colormap. They are removed from the
*panel's copy* of the data, not shaded over it; the axes still span the full recording, so every
row stays column-aligned with rows 1 and 2, and the empty margins are marked in grey.

**Everything here is a drawing decision.** This module cuts and re-lays-out arrays it is handed and
does nothing else: it takes the forward dict as given, never re-runs the model, never re-scores
anything, and is imported by no part of the training loop or of any metric. The tiling below in
particular is *not* how the model forecasts -- the forward pass decodes every valid anchor at
stride $1$, and the objective and every reported number are computed over all of them.

Both lag panels carry a secondary axis in **compensated** seconds -- $4\,(\ell + \delta)$, the
residual physiological lag on the mechanically aligned timeline -- and say so in the label. The
uncorrected sensor-file figure is $20$ s larger and is deliberately not what a figure shows.

**Why this lives at the package root.** Two consumers draw this page and they sit on opposite
sides of the layering: the training callback in :mod:`~teb_vae.lag_attn_rws.plotting`, which is a
Lightning callback, and the evaluation's per-sample pages, which may import neither Lightning nor
``plotting``. A home under ``eval/`` would make the *training* path import the evaluation package,
inverting the dependency; a home in ``plotting.py`` puts Lightning in the evaluation's import
graph. At the root, beside ``channel_reach.py``, both reach it and neither reaches the other.
``plotting.py`` re-exports the name, so the callback and the tests that monkeypatch it are
unchanged.

This module is matplotlib-only: no Lightning, no MLflow, no config, no loader. It does **not**
call :func:`~utils.style.apply_publication_style`; that mutates global ``rcParams`` and is called
once by whoever owns the process -- the callback at construction, the evaluation at run start --
rather than on every figure this builds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from teb_vae.lag_attn.figure_primitives import (  # noqa: E402
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_LIGHT_GRAY,
    COLOR_ORANGE,
    COLOR_VERMILLION,
    attach_lag_seconds_axis,
    concat_single_forecasts,
    safe_vabs,
    shade_warmup,
    time_axes,
    to_numpy,
)
from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry  # noqa: E402
from teb_vae.lag_attn.nets.lag_report import (  # noqa: E402
    COMPENSATED_LAG_AXIS_LABEL,
    SECONDS_PER_STEP,
    lag_compensated_seconds,
)
from utils.style import style_axes  # noqa: E402

__all__ = [
    "BAND_SIGMAS",
    "ForecastRowInputs",
    "InputStreamPanel",
    "annotate_channel_frequencies",
    "build_diagnostic_figure",
    "raw_context_row",
    "raw_forecast_rows",
]

#: The two row names the forecast seam owns. Named constants rather than literals repeated in a
#: sibling package: the layout's row list is this module's, and a name that disagrees with it
#: raises a ``KeyError`` from inside a callback that swallows exceptions to protect the fit.
RAW_ROW = "raw"
FORECAST_ROW = "forecast"

#: Raw sampling rate of the target signal, in Hz. With ``decimation = 16`` this is the $4$ s
#: decimated step the whole geometry is built on.
_FS_RAW = 4.0

#: Band width drawn around each forecast mean, in standard deviations. Public because a sibling
#: page draws the same band around a forecast in another domain, and two pages of one family
#: quoting different intervals under the same $\pm k\sigma$ caption is a difference nobody would
#: look for.
BAND_SIGMAS = 2.0

#: Physical unit of each raw signal the page draws, for the axis labels. Only reachable when the
#: loader's statistics are known -- see :func:`_denormalised`, which falls back to z-units rather
#: than putting one of these on an axis it cannot honour.
_SIGNAL_UNITS = {"fhr": "bpm", "up": "mmHg"}

#: Vertical strip reserved above the first panel for the two-line title, in inches.
_HEADER_INCHES = 0.75

#: Interpolation for every heatmap on the page. ``'none'`` rather than matplotlib's default
#: ``'antialiased'``: a resampled heatmap invents intermediate values between two anchors or two
#: latent dimensions, and on a lag map that is exactly the axis a reader is trying to read a peak
#: off. It also makes a rendered PDF's cells match the array's, which is what lets one be checked
#: against the other.
_IMSHOW_INTERPOLATION = "none"


def _denormalised(
    values: torch.Tensor, field: str, normalization_stats: Optional[Dict[str, Any]]
) -> Tuple[np.ndarray, str]:
    """Invert the loader's z-scoring on one raw signal, when the statistics are known.

    Args:
        values: Raw-signal values in loader units.
        field: The loader field name, ``'fhr'`` or ``'up'``.
        normalization_stats: The loader's statistics dict, or ``None``.

    Returns:
        ``(array, unit_label)`` -- the values and the unit to put on the axis. The label is the
        honest one: without statistics for *this* field the values stay in z-units rather than
        being mislabelled. The two signals are resolved independently, so a run whose statistics
        carry only the target draws that one in bpm and the source in z-units.
    """
    from train.graph_models_utils import denormalize_signal_data

    if normalization_stats is not None and field in normalization_stats:
        return (
            to_numpy(denormalize_signal_data(values, field, normalization_stats)),
            _SIGNAL_UNITS[field],
        )
    return to_numpy(values), "normalised"


@dataclass(frozen=True)
class InputStreamPanel:
    r"""One of the model's input streams, **after the causal guard**, ready to draw.

    The page's other rows are all statements about what the model produced; this one is the only
    statement about what it was given. It is drawn from the tensor the encoder actually receives
    -- the surviving channels only, each already shifted by its own delay $\delta_c$ -- rather
    than from the stored block, because the difference between those two is precisely the guard
    the row exists to make visible.

    Assembled by :func:`~teb_vae.lag_attn_rws.input_budget.stream_panels`, which owns the filter
    bank and the reach arithmetic. Passed in as plain arrays so this module stays a drawing
    module: nothing here imports ``kymatio``, and the evaluation, which draws the same page, does
    not acquire it by importing this one.

    Attributes:
        name: Stream name, ``'target'`` or ``'source'``. Becomes the row key ``input_<name>``.
        values: The gated stream of the sample being drawn, $(T, C_{\mathrm{kept}})$, in the
            channel order the encoder sees.
        delays: One delay $\delta_c$ in decimated steps per surviving channel, same order. All
            zero for an unguarded run, which draws as a flat staircase at $t = 0$.
        center_hz: One representative centre frequency per surviving channel, same order;
            ``nan`` where the channel has none (the order-$0$ scattering low-pass).
        blocks: ``(name, start, stop)`` per stored block, as half-open ranges **into the
            surviving channels**, for the row's dividers and its y-axis labels.
        title: The panel title, which is where the budget, the surviving counts and the delay
            range are stated.
    """

    name: str
    values: np.ndarray
    delays: np.ndarray
    center_hz: np.ndarray
    blocks: Tuple[Tuple[str, int, int], ...]
    title: str


def annotate_channel_frequencies(ax: Any, center_hz: np.ndarray, *, count: int = 8) -> Any:
    """Label a channel axis with the centre frequencies of a sample of its channels.

    A twin axis carrying tick *labels* at channel positions, not a frequency scale: the channel
    axis is not monotone in frequency -- it descends within a block and restarts at the next one
    -- so a continuous secondary scale would be a straight misreading. Shared by the per-sample
    input rows and the run-level budget figure so a channel is annotated identically in both.

    Args:
        ax: The axes whose y-axis is the channel index.
        center_hz: One centre frequency per channel, in Hz; ``nan`` where the channel has none.
        count: How many channels to label.

    Returns:
        The twin axes, or ``None`` when there are no channels to label.
    """
    values = np.asarray(center_hz, dtype=float)
    if not values.size:
        return None
    sampled = np.unique(np.linspace(0, values.size - 1, num=min(count, values.size), dtype=int))
    secondary = ax.twinx()
    secondary.set_ylim(ax.get_ylim())
    secondary.set_yticks(sampled)
    secondary.set_yticklabels(
        [
            "$S_0$" if not np.isfinite(values[index]) else f"{values[index]:.3g}"
            for index in sampled
        ],
        fontsize=6,
    )
    secondary.set_ylabel("channel centre freq. (Hz)", fontsize=7)
    secondary.grid(False)
    return secondary


def _input_stream_row(ax: Any, panel: InputStreamPanel, *, t_max: float, seconds_per_step: float):
    """Draw one gated input stream as a channel-by-time heatmap, and return the image.

    Three things are marked on it, and each is a property of the guard rather than of the sample:
    the block dividers, the per-channel centre frequency on a right-hand axis, and the staircase
    at $t = \\delta_c$ before which the channel carries the guard's zero fill rather than data.

    Args:
        ax: The row's main axes.
        panel: The stream to draw.
        t_max: The recording's length in seconds, so the row shares the page's time axis.
        seconds_per_step: $\\Delta$, for placing the delay staircase in physical time.

    Returns:
        The ``AxesImage``, for the caller's colorbar.
    """
    values = np.asarray(panel.values, dtype=float)
    n_channels = values.shape[1]

    # Robust limits rather than min/max: these are z-scored wavelet coefficients, and one
    # heavy-tailed channel otherwise sets the scale for all of them and flattens the rest to a
    # single colour. NaN-aware because a gated stream may carry non-finite values from the loader.
    finite = values[np.isfinite(values)]
    low, high = (
        (float(np.percentile(finite, 1.0)), float(np.percentile(finite, 99.0)))
        if finite.size
        else (0.0, 1.0)
    )
    if not high > low:
        high = low + 1.0

    image = ax.imshow(
        values.T, aspect="auto", cmap="viridis", origin="lower",
        vmin=low, vmax=high, extent=[0.0, t_max, -0.5, n_channels - 0.5],
        interpolation=_IMSHOW_INTERPOLATION,
    )

    # Block dividers and their names. Drawn from the panel's own spans rather than from a channel
    # count restated here: under a reach budget the blocks lose different numbers of channels, so
    # the boundary is not where the declared widths would put it.
    ticks, labels = [], []
    for index, (name, start, stop) in enumerate(panel.blocks):
        if index:
            ax.axhline(start - 0.5, color="white", linewidth=1.0, linestyle="--")
        ticks.append(0.5 * (start + stop) - 0.5)
        labels.append(name)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)

    annotate_channel_frequencies(ax, panel.center_hz)

    # Where each channel's data begins. Under the guard the first $\delta_c$ steps of a channel
    # have no source and are emitted as zero, and the whole staircase must sit inside the shaded
    # warm-up -- that requirement is what `resolve_channel_budget` enforces, and this is where a
    # reader can see that it holds.
    if n_channels and int(np.max(panel.delays)) > 0:
        ax.step(
            np.asarray(panel.delays, dtype=float) * seconds_per_step,
            np.arange(n_channels), where="mid",
            color=COLOR_ORANGE, linewidth=0.9,
            label="guard: first step with data, $\\delta_c$",
        )
        ax.legend(loc="lower right", fontsize=6, framealpha=0.9)

    ax.set_title(panel.title, fontsize=9, pad=6)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Input channel", fontsize=8)
    return image


@dataclass(frozen=True)
class ForecastRowInputs:
    r"""Everything the page's first two rows need, and the layout hooks they draw through.

    The two rows are the only ones that depend on what is being forecast. Rows $3$ to $7$ -- the
    latent state, the per-dimension KL, $K_t$ and the two lag maps -- are statements about the
    latent and the attention, and are identical whatever the target is. So the page hands these
    two out behind a seam and keeps the GridSpec, the row cuts, the shared time axis and the
    caption in one place; a sibling forecasting another domain writes a replacement for
    :func:`raw_forecast_rows` and inherits the rest of the page unedited.

    ``row_axes`` and ``finalise_time_axis`` are passed rather than reimplemented because they are
    what makes a column of the page one instant on all seven rows: an implementation that set its
    own limits would break the alignment that the whole figure is read by.

    Attributes:
        outs: The model's forward dict.
        target: The forecast target the builder was handed, in loader units. Raw samples for the
            raw-signal models; a feature block for a model forecasting coefficients.
        batch: The loader batch, or ``None``. Present so an implementation whose ``target`` is
            *not* the raw signal can still draw the raw traces for physiological context, which
            it cannot recover from ``target``.
        geometry: The model's trimmed-grid geometry.
        sample_index: Which sample of the batch to draw.
        normalization_stats: The loader's statistics, or ``None``.
        up_raw: The raw source, or ``None``.
        time_raw: The raw-grid time axis in seconds.
        t_max: The recording's length in seconds; the right edge of every row.
        row_axes: Maps a row name to its ``(main, cax)`` axes pair. The two names these rows own
            are :data:`RAW_ROW` and :data:`FORECAST_ROW`.
        finalise_time_axis: Pins the shared time axis and shades the warm-up.
    """

    outs: Dict[str, Any]
    target: torch.Tensor
    batch: Any
    geometry: TrimmedRawGeometry
    sample_index: int
    normalization_stats: Optional[Dict[str, Any]]
    up_raw: Optional[torch.Tensor]
    time_raw: Any
    t_max: float
    row_axes: Callable[[str], Tuple[Any, Any]]
    finalise_time_axis: Callable[..., None]


def raw_context_row(rows: ForecastRowInputs, fhr_values: torch.Tensor) -> Tuple[np.ndarray, str]:
    """Draw :data:`RAW_ROW`: the raw target trace, and the raw source on a twin axis beside it.

    Shared by every page in the family rather than owned by the raw one, because the row means the
    same thing whatever the model forecasts -- it is physiological context, not a readout, and rows
    $6$ and $7$ are statements *about* the UP trace it draws. What differs between pages is only
    where the FHR values come from: the raw page already holds them as its target, a feature-domain
    page has to reach into the batch for them, and neither can recover the other's.

    Args:
        rows: The row inputs and the layout hooks.
        fhr_values: The raw target of the sample being drawn, $(L_{\\mathrm{raw}},)$ in loader
            units. Already indexed by sample -- this function draws one recording.

    Returns:
        ``(values, unit)`` -- the trace as drawn and the unit it is in, so a caller rendering the
        same signal again in another row converts it once. The unit is ``'normalised'`` when the
        loader's statistics are unavailable, never a physical unit the values are not in.
    """
    ax, cax = rows.row_axes(RAW_ROW)
    fhr_np, unit = _denormalised(fhr_values, "fhr", rows.normalization_stats)
    fhr_np = np.asarray(fhr_np).ravel()

    # The source trace, on the target's own axis. Skipped rather than fatal when it is absent or
    # on another grid: a page that cannot draw UP is still the page every other row needs.
    up_np: Optional[np.ndarray] = None
    up_unit = "normalised"
    if rows.up_raw is not None:
        candidate, up_unit = _denormalised(
            rows.up_raw[rows.sample_index], "up", rows.normalization_stats
        )
        candidate = np.asarray(candidate).ravel()
        if candidate.size == rows.time_raw.size:
            up_np = candidate

    ax.plot(rows.time_raw, fhr_np, color=COLOR_BLUE, linewidth=0.7, label=f"FHR ({unit})")
    ax.set_title("Raw target FHR and raw source UP", fontsize=9, pad=6)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel(f"FHR ({unit})", fontsize=8, color=COLOR_BLUE)
    style_axes(ax, grid="both")
    rows.finalise_time_axis(ax)
    if up_np is not None:
        # Its own y-axis, because the two signals share neither unit nor scale and a single axis
        # would flatten whichever has the smaller range into a line. `twinx` makes the new axes'
        # patch transparent, so the warm-up shading below still shows through.
        twin = ax.twinx()
        twin.plot(rows.time_raw, up_np, color=COLOR_GREEN, linewidth=0.7, label=f"UP ({up_unit})")
        twin.set_ylabel(f"UP ({up_unit})", fontsize=8, color=COLOR_GREEN)
        twin.tick_params(axis="y", labelsize=7, colors=COLOR_GREEN)
        twin.grid(False)
        twin.set_xlim(0.0, rows.t_max)
        handles = list(ax.get_lines()) + list(twin.get_lines())
        ax.legend(
            handles, [handle.get_label() for handle in handles],
            loc="upper right", fontsize=7, framealpha=0.95,
        )
    cax.set_visible(False)
    return fhr_np, unit


def raw_forecast_rows(rows: ForecastRowInputs) -> None:
    r"""Draw the raw-signal page's first two rows: the two raw traces, then the tiled forecast.

    The forecast row is tiled rather than zoomed, and the tiling is a **choice of what to draw**,
    not a change to what the model does: ``outs`` already carries a forecast for every valid anchor
    at stride $1$, all of which the objective scored, and this row selects a subset of them. One
    anchor predicts $H \cdot R$ raw samples, so the anchors whose forecasts abut without overlapping
    are spaced exactly $H$ apart; the tiling starts at the first trained anchor $w$ and runs while
    an anchor is still valid, i.e. ``range(warmup, t_valid, horizon)``. Plotted at the model's own
    stride instead, adjacent windows would overlap by $(H-1)/H$ and the row would show each instant
    $H$ times over, from $H$ different latents. Two spans are therefore blank by construction:
    everything before $w$'s own window, and whatever tail is left when the recording is not an exact
    number of windows -- reaching the segment end would need the final anchor, whose window overlaps
    the last tiled one. At production geometry that is $8$ windows of $120$ s covering $124$ s to
    $1084$ s, leaving $[0, 124)$ s and $[1084, 1200)$ s undrawn.

    Args:
        rows: The row inputs and the layout hooks.
    """
    geometry = rows.geometry
    i, time_raw = rows.sample_index, rows.time_raw
    t_steps, horizon, raw_per_step = geometry.t, geometry.horizon, geometry.r
    warmup, t_valid = geometry.warmup, geometry.t_valid
    stats = rows.normalization_stats

    # ---- Row: the two raw traces ------------------------------------------
    # The target *is* the raw trace here, so the context row and the forecast row draw the same
    # signal and the conversion is done once, by the row that owns it.
    fhr_np, unit = raw_context_row(rows, rows.target[i])

    # The tiling's anchors, and the raw-second position of every window edge -- one per window
    # plus the last window's end. Mirrors `concat_single_forecasts`, which walks the identical
    # set: it stops at `t + 1 + H <= T`, which is `t < T - H = t_valid`.
    tile_anchors = list(range(warmup, t_valid, horizon))
    window_edges = [geometry.future_block_start(t) / _FS_RAW for t in tile_anchors]
    if tile_anchors:
        window_edges.append(
            (geometry.future_block_start(tile_anchors[-1]) + horizon * raw_per_step) / _FS_RAW
        )

    # Both forecasts and both bands, denormalized through the same affine map as the truth, so
    # the three curves in the forecast panel are directly comparable, then tiled onto the raw
    # grid. `concat_single_forecasts` is the shared helper for exactly this walk; its trailing
    # axis is "channels" for the decimated model and raw-samples-per-token here, and its `(T, R)`
    # result flattens to the raw grid because horizon token h of anchor t is decimated step
    # t + 1 + h, i.e. raw `[R(t + 1 + h), ...)` = `future_block_start(t) + R*h`. Uncovered
    # positions come back NaN and render as gaps rather than as a fabricated continuation.
    def _tiled(branch: str) -> List[np.ndarray]:
        """Return ``[mean, lower, upper]`` of one branch, tiled onto the raw grid, in ``unit``."""
        mean = rows.outs[f"mu_{branch}"][i]
        sigma = torch.exp(0.5 * rows.outs[f"logvar_{branch}"][i])
        curves = [mean, mean - BAND_SIGMAS * sigma, mean + BAND_SIGMAS * sigma]
        return [
            concat_single_forecasts(
                _denormalised(curve, "fhr", stats)[0], t_steps, horizon, warmup
            ).reshape(-1)
            for curve in curves
        ]

    base_mean, base_lo, base_hi = _tiled("base")
    full_mean, full_lo, full_hi = _tiled("full")
    # The truth restricted to the tiled support, so the forecast panel is about the predicted
    # windows alone and the uncovered spans read as absent rather than as unpredicted.
    truth_tiled = np.where(np.isfinite(full_mean), fhr_np, np.nan)

    # ---- Row: the forecast, tiled into non-overlapping windows -------------
    ax, cax = rows.row_axes(FORECAST_ROW)
    # The truth first, so it stays ``ax.lines[0]`` -- the tests read it from there.
    ax.plot(time_raw, truth_tiled, color=COLOR_BLACK, linewidth=0.7, label="true $Y^{+}$")
    ax.fill_between(time_raw, base_lo, base_hi, color=COLOR_GRAY, alpha=0.22, linewidth=0)
    ax.plot(
        time_raw, base_mean, color=COLOR_GRAY, linewidth=0.8, linestyle="--",
        label="base ($z^p$, target-only)",
    )
    ax.fill_between(
        time_raw, full_lo, full_hi, color=COLOR_VERMILLION, alpha=0.18, linewidth=0
    )
    ax.plot(
        time_raw, full_mean, color=COLOR_VERMILLION, linewidth=0.8,
        label="full ($z^q$, source-conditioned)",
    )
    # Where one forecast ends and the next begins. Without them the tiling reads as a single
    # continuous prediction, which is exactly what it is not: each window is decoded from one
    # latent and never sees the window before it.
    for edge in window_edges:
        ax.axvline(edge, color=COLOR_GRAY, linewidth=0.5, linestyle="--", alpha=0.7, zorder=1)
    window_seconds = horizon * raw_per_step / _FS_RAW
    ax.set_title(
        f"Forecast — {len(tile_anchors)} consecutive non-overlapping {window_seconds:.0f} s "
        f"windows from the first trained anchor, mean $\\pm$ {BAND_SIGMAS:.0f}$\\sigma$ "
        f"({horizon}$\\times${raw_per_step} = {horizon * raw_per_step} raw samples each; "
        f"dashed: window edges)",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel(f"FHR ({unit})", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.95)
    style_axes(ax, grid="both")
    rows.finalise_time_axis(ax)
    cax.set_visible(False)


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
    up_raw: Optional[torch.Tensor] = None,
    normalization_stats: Optional[Dict[str, Any]] = None,
    delay_steps: int = 0,
    forecast_rows: Optional[Callable[[ForecastRowInputs], None]] = None,
    batch: Any = None,
    input_streams: Optional[Sequence[InputStreamPanel]] = None,
) -> Any:
    r"""Build the seven-row diagnostic figure for one sample.

    This function owns the layout: the row heights, the GridSpec with its reserved colorbar
    column, the two boundaries the maps are cut at, the shared time axis every row is pinned to,
    and the caption. Rows $3$ to $7$ -- the latent state, the per-dimension KL, $K_t$, the lag
    attention and the KL-by-lag map -- are drawn here, because they are statements about the
    latent and the attention and read the same whatever is being forecast.

    Rows $1$ and $2$ are the ones that depend on the target domain, and they go through
    ``forecast_rows``; see :func:`raw_forecast_rows`, the default, for what the raw-signal page
    draws and why its forecast row is tiled.

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
        up_raw: The raw source $(B, L_{\mathrm{raw}})$ in loader units, or ``None``. The model
            never sees it -- it consumes the decimated UP feature blocks -- so it is passed in
            beside the target rather than read off the forward dict.
        normalization_stats: The loader's statistics, so the two traces render in bpm and mmHg.
        delay_steps: The causal input delay $\delta$, for the compensated lag axes.
        forecast_rows: Draws rows $1$ and $2$. ``None`` -- the default, so every existing caller
            is unaffected -- uses :func:`raw_forecast_rows`.
        batch: The loader batch, passed straight to ``forecast_rows``. Unused by the raw page,
            which reads its traces off ``fhr_raw``, and the only route to them for an
            implementation whose target is not the raw signal.
        input_streams: The model's gated input streams, one row each, drawn between the forecast
            and the latent. ``None`` or empty -- the default, so every existing caller is
            unaffected -- draws the seven rows alone.

    Returns:
        The matplotlib ``Figure``. The caller saves and closes it.
    """
    i = int(sample_index)
    t_steps = geometry.t
    warmup, t_valid = geometry.warmup, geometry.t_valid
    time_raw, time_dec, t_max = time_axes(t_steps, geometry.raw_len, fs_raw=_FS_RAW)
    seconds_per_step = t_max / float(t_steps)

    # The two boundaries rows 3-7 are cut at. Both are properties of the objective, not of the
    # figure: nothing before `warmup_sec` enters any loss, and nothing after `tail_sec` is either
    # decoded or inside the KL support, so a latent there carries no gradient at all.
    warmup_sec = float(warmup) * seconds_per_step
    tail_sec = float(t_valid) * seconds_per_step

    mu_prior_np = to_numpy(outs["mu_prior"][i])                       # (T, d_z)
    delta_mu_np = to_numpy(outs["mu_post"][i] - outs["mu_prior"][i])  # (T, d_z)
    kld_dims_np = to_numpy(kld_per_dim[i])                            # (T, d_z)
    kld_total_np = to_numpy(outs["kld_per_t"][i])                     # (T,)
    alpha_np = to_numpy(outs["attn_weights"][i]).mean(axis=1)         # (T, L)
    kl_lag_np = to_numpy(outs["source_kl_lag_map"][i])                # (T, L)
    d_z, n_lags = mu_prior_np.shape[1], alpha_np.shape[1]

    # One row per gated input stream, between what the model produced and the latent it produced
    # it from. Built into the row list rather than reserved unconditionally, so a caller that
    # passes none gets exactly the page it got before.
    panels = tuple(input_streams or ())
    row_specs = [
        ("raw", 0.9),
        # Taller than the other line rows: it now carries the whole recording rather than one
        # 480-sample window, and two forecasts with their bands on top of it.
        ("forecast", 1.3),
        *((f"input_{panel.name}", 1.25) for panel in panels),
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

    def finalise_time_axis(ax: Any, *, tail: bool = False) -> None:
        """Pin the shared physical-time axis and mark the spans this row draws nothing over.

        Args:
            ax: The axes to finalise.
            tail: Whether this row also stops at the last trained anchor.
        """
        # Every row spans the whole recording whatever it draws, so a column of the page is the
        # same instant on all seven of them.
        ax.set_xlim(0.0, t_max)
        shade_warmup(ax, warmup, t_max, t_steps)
        if tail:
            ax.axvspan(tail_sec, t_max, color=COLOR_LIGHT_GRAY, alpha=0.35, zorder=0)

    def trained_columns(values: np.ndarray, *, drop_tail: bool) -> Tuple[np.ndarray, float]:
        """Cut a time-first array down to the anchors the objective actually scored.

        Args:
            values: An array whose first axis is the decimated step.
            drop_tail: Whether to also drop the tail $H$ anchors, which are neither decoded nor
                inside the KL support and so receive no gradient at all.

        Returns:
            ``(cut, stop_seconds)`` -- the surviving columns and the second the last one ends at,
            which is the right edge of the ``imshow`` extent that draws them.
        """
        stop = t_valid if drop_tail else t_steps
        return values[warmup:stop], (tail_sec if drop_tail else t_max)

    def lag_panel(
        ax: Any, values: np.ndarray, title: str, cmap: str, *, drop_tail: bool
    ) -> Any:
        """Draw a ``(T, L)`` lag map, trained columns only, with a compensated-seconds axis."""
        trained, stop_sec = trained_columns(values, drop_tail=drop_tail)
        image = ax.imshow(
            trained.T, aspect="auto", cmap=cmap, origin="lower",
            extent=[warmup_sec, stop_sec, -0.5, n_lags - 0.5],
            interpolation=_IMSHOW_INTERPOLATION,
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
        finalise_time_axis(ax, tail=drop_tail)
        return image

    # ---- Rows 1-2: whatever this model forecasts ---------------------------
    # Behind a seam, because these two are the only rows that depend on the target domain.
    # Everything below -- the cuts, the shared axis, the lag panels, the caption -- is a
    # statement about the latent and the attention and is the same page whatever is forecast.
    (forecast_rows or raw_forecast_rows)(
        ForecastRowInputs(
            outs=outs,
            target=fhr_raw,
            batch=batch,
            geometry=geometry,
            sample_index=i,
            normalization_stats=normalization_stats,
            up_raw=up_raw,
            time_raw=time_raw,
            t_max=t_max,
            row_axes=row_axes,
            finalise_time_axis=finalise_time_axis,
        )
    )

    # ---- Rows: the gated input streams, as the encoders receive them --------
    # Drawn over the whole recording, not cut to the trained anchors like the rows below: the
    # warm-up columns are where the guard's zero fill lives, and cutting them would remove the
    # one span the delay staircase exists to be checked against.
    for panel in panels:
        ax, cax = row_axes(f"input_{panel.name}")
        image = _input_stream_row(
            ax, panel, t_max=t_max, seconds_per_step=seconds_per_step
        )
        heatmap_spines(ax)
        attach_cbar(cax, image, "value")
        finalise_time_axis(ax)

    # ---- Row: prior mean over the source-derived delta ---------------------
    ax, cax = row_axes("latent")
    # Stacked time-first so the cut below is one slice of one axis, then transposed for imshow.
    latent_stack, latent_stop = trained_columns(
        np.concatenate([mu_prior_np, delta_mu_np], axis=1), drop_tail=True
    )
    vabs = safe_vabs(latent_stack)
    image = ax.imshow(
        latent_stack.T, aspect="auto", cmap="bwr", origin="upper",
        vmin=-vabs, vmax=vabs, extent=[warmup_sec, latent_stop, 2 * d_z - 0.5, -0.5],
        interpolation=_IMSHOW_INTERPOLATION,
    )
    ax.axhline(d_z - 0.5, color="white", linewidth=1.2, linestyle="--")
    ax.set_yticks([d_z // 2, d_z + d_z // 2])
    ax.set_yticklabels(["$\\mu^p$", "$\\mu^q-\\mu^p$"])
    ax.set_title(
        "Target-only latent state and the source-derived shift (shared colour scale, "
        "trained anchors only)",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    heatmap_spines(ax)
    attach_cbar(cax, image, "value")
    finalise_time_axis(ax, tail=True)

    # ---- Row: per-dimension KL --------------------------------------------
    ax, cax = row_axes("kld_dims")
    kld_dims_trained, kld_dims_stop = trained_columns(kld_dims_np, drop_tail=True)
    image = ax.imshow(
        kld_dims_trained.T, aspect="auto", cmap="magma", origin="lower",
        extent=[warmup_sec, kld_dims_stop, -0.5, d_z - 0.5],
        interpolation=_IMSHOW_INTERPOLATION,
    )
    ax.set_title(
        "Per-dimension source-conditioned KL (nats, trained anchors only)", fontsize=9, pad=6
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Latent dim", fontsize=8)
    heatmap_spines(ax)
    attach_cbar(cax, image, "nats")
    finalise_time_axis(ax, tail=True)

    # ---- Row: total KL per step -------------------------------------------
    ax, cax = row_axes("kld_total")
    # Cut like the two KL heatmaps rather than merely shaded: the warm-up carries the encoders'
    # settling transient, and left in it sets the y-scale so the trained range flattens.
    kld_trained, _ = trained_columns(kld_total_np, drop_tail=True)
    kld_time, _ = trained_columns(time_dec, drop_tail=True)
    ax.plot(kld_time, kld_trained, color=COLOR_VERMILLION, linewidth=0.9)
    ax.set_title(
        "$K_t$ — total source-conditioned KL per step (trained anchors only)",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("nats", fontsize=8)
    style_axes(ax, grid="both")
    finalise_time_axis(ax, tail=True)
    cax.set_visible(False)

    # ---- Row: lag attention with its argmax --------------------------------
    ax, cax = row_axes("lag_attn")
    # The tail stays here alone among the five: attention is a property of the source stream and
    # is defined at every step, whereas the KL panels above and below it are identically zero
    # there by construction of the mask.
    image = lag_panel(
        ax, alpha_np, "Lag attention (head-averaged) with per-step argmax", "viridis",
        drop_tail=False,
    )
    attn_trained, _ = trained_columns(alpha_np, drop_tail=False)
    attn_time, _ = trained_columns(time_dec, drop_tail=False)
    ax.plot(
        attn_time, attn_trained.argmax(axis=1),
        color=COLOR_ORANGE, linewidth=0.7, alpha=0.85,
    )
    attach_cbar(cax, image, "attention")

    # ---- Row: the KL attributed across lags --------------------------------
    ax, cax = row_axes("kl_lag_map")
    image = lag_panel(
        ax, kl_lag_np, "$\\widetilde K_{t,\\ell}$ — source-conditioned KL by lag", "magma",
        drop_tail=True,
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
