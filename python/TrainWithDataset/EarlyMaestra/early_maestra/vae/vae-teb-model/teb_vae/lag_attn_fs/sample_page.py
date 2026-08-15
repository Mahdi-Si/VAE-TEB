r"""The two rows of the diagnostic page that depend on what this model forecasts.

The page is the sibling's. :func:`~teb_vae.lag_attn_rws.sample_page.build_diagnostic_figure` owns
the GridSpec, the row heights, the cut at the trained anchors, the shared physical-time axis and
the caption, and draws rows $3$ to $7$ -- the latent state, the per-dimension KL, $K_t$, the lag
attention and the KL-by-lag map -- because every one of those is a statement about the latent and
the attention and reads the same whatever is being forecast. What is here is the replacement for
its first two rows, reached through the ``forecast_rows`` seam.

**Row 1 is unchanged in meaning and shared in code.** It is the raw FHR and UP traces, drawn by
:func:`~teb_vae.lag_attn_rws.sample_page.raw_context_row`, the sibling page's own implementation.
This model never reads either trace -- it consumes the decimated feature blocks and forecasts
decimated coefficients -- but rows $6$ and $7$ are statements *about* the UP trace, and a reader
judging whether a forecast is plausible needs the physiology in the same column of the page. The
one difference is where the trace comes from: the raw page holds it as its target, and here the
target is a $(B, T, c_y)$ feature block, so the values are read from the batch.

**Row 2 is the one that is genuinely new**, and it answers a question one curve cannot. The
reconstruction is summed over $H \cdot C_{\mathrm{keep}} = 2340$ coefficients, so a model that
forecasts three easy low-frequency channels well is indistinguishable, in every scalar the run
reports, from one that is uniformly mediocre. The row therefore draws two things:

* three target channels chosen **by the data** -- worst, middle and best calibrated, through
  :func:`~teb_vae.lag_attn.figure_primitives.select_forecast_channels` -- as three offset lanes
  carrying the truth, the base ($z^p$) and full ($z^q$) forecasts and their $\mu \pm 2\sigma$
  bands, tiled into consecutive non-overlapping windows exactly as the raw page tiles its one
  signal;
* a $(C_{\mathrm{keep}} \times H)$ absolute-error map for **one** anchor, so a per-channel failure
  is visible without any channel having been chosen by hand.

The channels are never hard-coded. ``lag_attn`` carried a ``forecast_channels`` config key until
the stored phase-harmonic block went from $44$ to $66$ channels and every inherited index silently
began naming a different coefficient; the rule replaces the key rather than re-tuning it.

**The error map is an inset rather than a panel of its own.** Its x-axis is the horizon step
$\tau$, not physical time, so a side-by-side split of the row would leave the forecast curves
narrower than the other six rows and break the property the whole page is read by -- that a column
is one instant on every row. Drawn inside the row's own axes it costs a corner of the forecast
panel and nothing else, and the window it describes is shaded on the curves so the two are tied.

Nothing here is denormalised. The forecast target is the loader's ``normalize_fields`` output used
as delivered -- there is deliberately no second normalisation and no per-channel statistics blob --
so the coefficients have no physical unit to be restored to, and the axis says ``normalised``. Row
$1$ is the exception and converts, because its two signals do have units and a clinician cannot
check a forecast drawn in z-units.

Like the sibling page this module is matplotlib-only -- no Lightning, no MLflow, no config, no
loader -- and it never re-runs the model or re-scores anything: it cuts and lays out arrays it is
handed.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
from matplotlib.ticker import MaxNLocator  # noqa: E402

from teb_vae.lag_attn.figure_primitives import (  # noqa: E402
    COLOR_BLACK,
    COLOR_GRAY,
    COLOR_ORANGE,
    COLOR_VERMILLION,
    concat_single_forecasts,
    safe_vabs,
    select_forecast_channels,
    time_axes,
    to_numpy,
)
from teb_vae.lag_attn_rws.sample_page import (  # noqa: E402
    BAND_SIGMAS,
    FORECAST_ROW,
    RAW_ROW,
    ForecastRowInputs,
    raw_context_row,
    top_down_extent,
)
from utils.style import style_axes  # noqa: E402

__all__ = ["FORECAST_CHANNELS", "feature_forecast_rows"]

#: Target channels the forecast row draws as offset lanes. Three is what fits legibly at the
#: sibling's row height; *which* three is the data's to say, never a configured index.
FORECAST_CHANNELS = 3

#: Lane spacing as a multiple of the widest lane's own drawn extent, so neighbouring bands cannot
#: touch and be read as one signal. Anything above $1$ separates them; the margin is what keeps the
#: gap visible rather than merely non-negative.
_LANE_HEADROOM = 1.15

#: Where the error map sits inside **this** page's forecast axes, as ``[x0, y0, width, height]``
#: in axes coordinates. Far right, because this tiling starts at the first trained anchor and
#: leaves the recording's tail undrawn by construction -- reaching the segment end would need an
#: anchor whose window overlaps the last tiled one -- so the corner it covers is the emptiest one
#: on the row. A page whose blank corner is elsewhere passes its own ``box``: the causal sibling's
#: tiling starts at a floor of $133$ and runs to the end, so its right margin is the one place on
#: the row an inset would hide data.
_ERROR_MAP_BOX = (0.775, 0.06, 0.215, 0.90)

#: Interpolation for the error map, matching the sibling page's heatmaps. ``'none'`` rather than
#: matplotlib's default: a resampled map invents values between two channels or two horizon steps,
#: and per-channel resolution is the entire reason this panel exists.
_IMSHOW_INTERPOLATION = "none"


def _batch_field(batch: Any, name: str) -> Optional[Any]:
    """Pull ``name`` from a dict batch or an attribute-style batch.

    Args:
        batch: A batch from the data module, or ``None``.
        name: Field name.

    Returns:
        The field, or ``None`` if the batch is absent or carries no such field.
    """
    if batch is None:
        return None
    if isinstance(batch, dict):
        return batch.get(name)
    return getattr(batch, name, None)


def _resolved_keep_index(keep_index: Optional[Sequence[int]], width: int) -> np.ndarray:
    """Resolve the decoder's output channels to their positions in the declared target stream.

    Args:
        keep_index: The reach budget's surviving target channels, or ``None`` for an unguarded
            model, whose decoder emits every declared channel in order.
        width: The decoder's output width, i.e. how many channels the forecast carries.

    Returns:
        A ``(width,)`` integer array of declared channel indices.

    Raises:
        ValueError: If a keep-index was supplied whose length is not the decoder's width -- which
            would gather the wrong truth for every lane and still draw a plausible figure.
    """
    if keep_index is None:
        return np.arange(width, dtype=int)
    resolved = np.asarray(to_numpy(keep_index), dtype=int).ravel()
    if resolved.size != width:
        raise ValueError(
            f"keep_index has {resolved.size} entries but the forecast carries {width} channels; "
            f"the index is positional into the declared target stream, so a mismatch would draw "
            f"the wrong channel's truth against each forecast"
        )
    return resolved


def _tiled_branch(rows: ForecastRowInputs, branch: str) -> Tuple[np.ndarray, np.ndarray]:
    r"""Tile one branch's forecast onto the decimated grid, for every channel it emits.

    Non-overlapping and stride-$H$, the sibling page's construction and for its reason: the model
    decodes every valid anchor at stride $1$, so plotted at its own stride the row would show each
    instant $H$ times over from $H$ different latents. The overlap-averaging alternative in
    :mod:`~teb_vae.lag_attn.figure_primitives` blends those $H$ latents into one curve, which is a
    third object neither the model nor the objective ever produces.

    Args:
        rows: The row inputs.
        branch: ``'base'`` or ``'full'``.

    Returns:
        ``(mean, sigma)``, each $(T, C_{\mathrm{keep}})$ on the decimated grid, ``NaN`` where no
        tiled window covers the step.
    """
    geometry = rows.geometry
    index = rows.sample_index

    def tiled(values: np.ndarray) -> np.ndarray:
        """Walk the non-overlapping windows of one $(T_{\\mathrm{valid}}, H, C)$ array."""
        return concat_single_forecasts(values, geometry.t, geometry.horizon, geometry.warmup)

    mean = to_numpy(rows.outs[f"mu_{branch}"][index])
    sigma = np.exp(0.5 * to_numpy(rows.outs[f"logvar_{branch}"][index]))
    return tiled(mean), tiled(sigma)


def _draw_context_row(rows: ForecastRowInputs) -> None:
    """Draw row $1$: the raw traces from the batch, or a note saying the batch carried none.

    Args:
        rows: The row inputs and the layout hooks.
    """
    fhr_raw = _batch_field(rows.batch, "fhr")
    if fhr_raw is not None:
        raw_context_row(rows, fhr_raw[rows.sample_index])
        return
    # The page is still worth drawing without it -- six of the seven rows say nothing about the raw
    # grid -- so the row is annotated rather than raised over. It keeps its title, its axis and its
    # place in the layout, so the rows below stay column-aligned and the gap is visible.
    ax, cax = rows.row_axes(RAW_ROW)
    ax.set_title("Raw target FHR and raw source UP", fontsize=9, pad=6)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.text(
        0.5, 0.5, "raw traces unavailable in this batch",
        transform=ax.transAxes, ha="center", va="center", fontsize=8, color=COLOR_GRAY,
    )
    style_axes(ax, grid="both")
    rows.finalise_time_axis(ax)
    cax.set_visible(False)


def _draw_error_map(
    ax: Any,
    cax: Any,
    error: np.ndarray,
    *,
    first_block_channels: int,
    anchor_seconds: float,
    box: Sequence[float] = _ERROR_MAP_BOX,
) -> None:
    r"""Draw the per-channel absolute-error map for one anchor, inset into the forecast axes.

    Args:
        ax: The forecast row's main axes; the map is inset into it.
        cax: The row's reserved colorbar axes.
        error: $(C_{\mathrm{keep}}, H)$ absolute error of the full forecast at one anchor.
        first_block_channels: How many of the surviving channels came from the first stored block,
            so the boundary between the two is drawn on the channel axis. The two blocks' filters
            have different reaches and therefore different blends, and it is the split the run
            reports ``pred_gap_st`` and ``pred_gap_ph`` over. ``0`` draws no line.
        anchor_seconds: The anchor's own position in the recording, for the panel's label.
        box: ``[x0, y0, width, height]`` in the row's axes coordinates, defaulting to
            :data:`_ERROR_MAP_BOX`. A parameter because *which corner is blank* is a property of
            the caller's tiling, not of this panel: the two-sided page leaves its tail undrawn and
            the causal one leaves the span below its anchor floor undrawn, and an inset over the
            other page's drawn span hides exactly the forecast it is a detail of.
    """
    n_channels, horizon = error.shape
    inset = ax.inset_axes(list(box))
    image = inset.imshow(
        error,
        aspect="auto",
        cmap="magma",
        origin="upper",
        # Cell centres at integer $\tau$ and integer channel, so a cell read off the rendered
        # figure is the array's cell rather than an interpolated neighbourhood of it. Channel $0$
        # at the top, the page's one channel-axis convention -- a map whose channel axis ran the
        # other way from the input rows two rows above it would be read wrongly by anyone who
        # carried a channel between them.
        extent=top_down_extent(-0.5, horizon - 0.5, n_channels),
        vmin=0.0,
        vmax=safe_vabs(error),
        interpolation=_IMSHOW_INTERPOLATION,
    )
    if 0 < first_block_channels < n_channels:
        inset.axhline(first_block_channels - 0.5, color="white", linewidth=0.8, linestyle="--")
    # No title, deliberately: every *titled* axes on this page spans the whole recording on the
    # shared time axis, and this one is the horizon of a single anchor. Labelled instead, so the
    # property that lets a reader carry an instant down the page stays exactly checkable.
    inset.set_xlabel(f"$\\tau$ @ t={anchor_seconds:.0f} s", fontsize=7)
    inset.set_ylabel("target channel", fontsize=7)
    # Both axes count things. A fractional tick on either would name a horizon step or a channel
    # that does not exist, and at the tiny horizons this page is tested at that is what the
    # default locator produces.
    inset.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    inset.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    inset.tick_params(labelsize=6)
    inset.grid(False)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color(COLOR_BLACK)
        spine.set_linewidth(0.6)

    cbar = ax.figure.colorbar(image, cax=cax)
    cbar.set_label("$|Y^{+}-\\mu^{q}|$ (normalised)", fontsize=8, color=COLOR_BLACK)
    cbar.ax.tick_params(labelsize=7, colors=COLOR_BLACK)


def feature_forecast_rows(
    rows: ForecastRowInputs,
    *,
    keep_index: Optional[Sequence[int]] = None,
    block_split: Optional[int] = None,
) -> None:
    r"""Draw the feature-domain page's first two rows.

    Bound to a model's channel facts by the task and handed to
    :func:`~teb_vae.lag_attn_rws.sample_page.build_diagnostic_figure` as its ``forecast_rows``
    seam. Those facts are the two things the page cannot recover from the arrays it is given:
    which declared channel each forecast channel *is* -- needed to gather the truth and to label a
    lane with a number that means something outside this run -- and where the two stored blocks
    meet.

    Args:
        rows: The row inputs and the layout hooks. ``rows.target`` is the concatenated target
            stream $(B, T, c_y)$ the loss was computed against, and ``rows.batch`` is the only
            route to the raw traces row $1$ draws.
        keep_index: The reach budget's surviving target channels, positional into the declared
            $c_y$. ``None`` for an unguarded model, whose decoder emits all of them in order.
        block_split: How many declared channels belong to the first stored block, for the error
            map's boundary line. ``None`` draws no boundary.

    Raises:
        ValueError: If ``keep_index`` does not have one entry per forecast channel.
    """
    index, geometry = rows.sample_index, rows.geometry

    _draw_context_row(rows)

    base_mean, base_sigma = _tiled_branch(rows, "base")
    full_mean, full_sigma = _tiled_branch(rows, "full")
    keep = _resolved_keep_index(keep_index, full_mean.shape[-1])

    # The truth on the decimated grid, gathered to the channels the decoder emits and restricted
    # to the tiled support, so an uncovered span reads as absent rather than as unpredicted.
    stream = to_numpy(rows.target[index])
    truth = np.where(np.isfinite(full_mean), stream[:, keep], np.nan)

    # Which channels to draw, from this batch's own calibration. The same rule at any channel
    # count: no index in this module names a coefficient.
    lanes, coverage = select_forecast_channels(
        truth, full_mean, full_sigma, count=FORECAST_CHANNELS, n_sigmas=BAND_SIGMAS
    )

    ax, cax = rows.row_axes(FORECAST_ROW)
    # Through the page builder's own axis helper rather than re-deriving the arithmetic: the
    # decimated grid has to land on the same seconds the five rows below it are drawn against.
    _, time_dec, _ = time_axes(geometry.t, geometry.raw_len)
    seconds_per_step = rows.t_max / float(geometry.t)

    # One offset per lane, from the drawn extent rather than a constant: the bands are
    # $2\sigma$ wide and $\sigma$ is a learned per-coefficient quantity, so a fixed offset either
    # overlaps early in training or flattens every lane late in it.
    def lane_extent(channel: int) -> float:
        """Total vertical span the widest artist of one lane needs.

        Both branches are measured, not just the source-conditioned one. The lane draws the
        target-only band too, and $\\sigma^{p} > \\sigma^{q}$ is the ordinary case -- sizing the
        stride off the narrower branch alone lets the wider band run into the lane above, so a
        reader attributes one channel's uncertainty to the channel labelled there.
        """
        half = BAND_SIGMAS * full_sigma[:, channel]
        base_half = BAND_SIGMAS * base_sigma[:, channel]
        stacked = np.concatenate(
            [full_mean[:, channel] - half, full_mean[:, channel] + half,
             base_mean[:, channel] - base_half, base_mean[:, channel] + base_half,
             truth[:, channel]]
        )
        finite = stacked[np.isfinite(stacked)]
        return float(finite.max() - finite.min()) if finite.size else 0.0

    stride = _LANE_HEADROOM * max([lane_extent(int(channel)) for channel in lanes] + [0.0])
    if not np.isfinite(stride) or stride <= 0.0:
        stride = 1.0

    for lane, channel in enumerate(int(value) for value in lanes):
        offset = lane * stride
        # The truth first in every lane, as on the sibling page, so the lane's first line is the
        # thing the two forecasts are being judged against.
        ax.plot(
            time_dec, truth[:, channel] + offset, color=COLOR_BLACK, linewidth=0.7,
            label="true $Y^{+}$" if lane == 0 else None,
        )
        for mean_all, sigma_all, colour, alpha, style, label in (
            (base_mean, base_sigma, COLOR_GRAY, 0.22, "--", "base ($z^p$, target-only)"),
            (full_mean, full_sigma, COLOR_VERMILLION, 0.18, "-",
             "full ($z^q$, source-conditioned)"),
        ):
            mean = mean_all[:, channel] + offset
            half = BAND_SIGMAS * sigma_all[:, channel]
            ax.fill_between(
                time_dec, mean - half, mean + half, color=colour, alpha=alpha, linewidth=0
            )
            ax.plot(
                time_dec, mean, color=colour, linewidth=0.8, linestyle=style,
                label=label if lane == 0 else None,
            )

    # Each lane named by its **declared** channel and its calibration, so the number on the axis
    # survives a change of reach budget: the positional index would not.
    ax.set_yticks([lane * stride for lane in range(len(lanes))])
    ax.set_yticklabels(
        [f"ch {int(keep[channel])}\n{coverage[int(channel)]:.0%}" for channel in lanes],
        fontsize=7,
    )

    # The anchor the error map describes: the middle tiled window, clear of the warm-up transient
    # at one end and of the undrawn tail at the other, and a fixed choice rather than the best or
    # worst window the run happens to contain. Its target is decimated steps $[t+1, t+1+H)$.
    tile_anchors = list(range(geometry.warmup, geometry.t_valid, geometry.horizon))
    anchor = tile_anchors[len(tile_anchors) // 2] if tile_anchors else geometry.warmup
    anchor_start = float(anchor + 1) * seconds_per_step
    ax.axvspan(
        anchor_start, anchor_start + geometry.horizon * seconds_per_step,
        color=COLOR_ORANGE, alpha=0.14, zorder=0,
    )

    ax.set_title(
        f"Forecast — {len(lanes)} of {len(keep)} target channels by "
        f"{BAND_SIGMAS:.0f}$\\sigma$ calibration (worst, middle, best), lanes offset by "
        f"{stride:.3g}, mean $\\pm$ {BAND_SIGMAS:.0f}$\\sigma$; shaded: the anchor the error map "
        f"draws",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("target coefficient (normalised)", fontsize=8)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.95)
    style_axes(ax, grid="both")
    rows.finalise_time_axis(ax)

    # $(C, H)$ rather than $(H, C)$: imshow's first axis is the vertical one, and the channel is
    # what a reader scans for a failure.
    anchor_truth = stream[anchor + 1 : anchor + 1 + geometry.horizon, keep]
    anchor_mean = to_numpy(rows.outs["mu_full"][index, anchor])
    _draw_error_map(
        ax,
        cax,
        np.abs(anchor_truth - anchor_mean).T,
        first_block_channels=(
            0 if block_split is None else int(np.count_nonzero(keep < int(block_split)))
        ),
        anchor_seconds=anchor_start,
    )
