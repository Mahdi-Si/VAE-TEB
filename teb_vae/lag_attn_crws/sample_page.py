r"""The two rows of the diagnostic page that depend on a tiled anchor set over a raw target.

The page is the raw-signal sibling's. :func:`~teb_vae.lag_attn_rws.sample_page.build_diagnostic_figure`
owns the GridSpec, the row heights, the cut at the trained anchors, the shared physical-time axis and
the caption, and draws the five rows below -- the latent state, the per-dimension KL, $K_t$, the lag
attention and the KL-by-lag map -- because each of those is a statement about the latent and the
attention and reads the same whatever is being forecast. What is here is the one thing this cell's
geometry makes different, and it is a single seam rather than the causal-feature sibling's three.

**The forecast rows**, through ``forecast_rows``. The raw sibling's version cannot be reused, and
the reason is not the target domain -- the target *is* its target -- but the anchor axis. That
implementation tiles with :func:`~teb_vae.lag_attn.figure_primitives.concat_single_forecasts`, which
walks ``range(warmup, t_valid, horizon)`` and reads ``mu_pred[t]`` at each **anchor** $t$; this
model's forecast is $(A_{\max}, H, R)$ indexed by *position in the decoded set*. At the shipped
geometry that is $136$ dense positions read at anchors $134 \dots 269$, so the first index is already
out of range and the page dies inside a handler that warns and continues -- a whole run with an empty
diagnostics directory and one log line. Where it does not raise it is worse: at a smaller floor it
draws a real forecast at the wrong time, with no exception anywhere in it. The tiling is therefore
walked through ``anchor_index`` and ``anchor_valid``, which the forward returns for exactly this
reason: the figure and the objective read one anchor set or they are two pictures of two models.

**Two rows, not the causal-feature page's eight.** That page adds six stitched field rows because
three of $98$ target channels drawn as offset lanes is not a picture of a $98$-channel forecast.
Here the decoder emits $R = 16$ raw samples per horizon token of **one** signal, so the forecast is a
curve on the raw grid and the shipped two-row layout is the right one: the trace, then the tiling
drawn over it. Nothing is reserved through ``forecast_extra_rows`` and the task declares no such
seam -- a name reserved and not drawn is a blank row on every page of the run.

**The input rows and the run-level budget figure are the causal-feature cell's**, imported by the
task rather than rebuilt here. Those streams really are the same three tensors -- this cell differs
from that one in what the decoder emits, not in what the encoders read -- so a second implementation
could only differ from the first by being wrong. The shipped builders cannot draw them for either
cell: both consult the production two-sided Morlet bank, which did not produce these coefficients and
refuses these channel widths, inside handlers that warn and continue.

**The lag axis is stored-coefficient time on the input side, and only on the input side -- and this
is the one cell where that is a repairable statement.** The target is a raw sample, so it has no
warm-up and no group delay and the anchor is exact: anchor $t$ predicts raw samples
$[16(t+1), 16(t+1) + H R)$, which are those samples and no others. The *inputs* are one-sided
coefficients and still lag by their own composed group delay, $13$ to $791$ s as the bank declares
them. Unaligned, that bias is indexed by a channel *pair* and no single number labels the axis; with
the source channels shifted onto one reference $\tau^u_{\mathrm{ref}}$ it collapses to a constant,
and because $\tau^y \equiv 0$ here the lead time
$\Delta(\ell + 1 + h) + \tau^u_{\mathrm{ref}} - 20$ s is a delay between *signals* rather than
between two filters' reports of them. That is what the footnote states, and the arithmetic behind it
is :func:`~teb_vae.lag_attn.nets.lag_report.physical_lag_seconds` rather than a second copy here.

The footnote is where it is said because the two lag panels belong to the shared builder, and the
six shipped models drawn by that builder must not acquire a caption about a transform they do not
use.

Like both sibling pages this module is matplotlib-only -- no Lightning, no MLflow, no config, no
loader -- and it never re-runs the model or re-scores anything: it cuts and lays out arrays it is
handed. The anchor walk and the anchor overlay are the causal-feature page's own functions, reached
by import: both take an anchor set, a validity vector and a horizon and name no channel, no
coefficient and no target at all, so a copy of either here could only drift from the tiling the
sibling draws.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")

from teb_vae.lag_attn.figure_primitives import (  # noqa: E402
    COLOR_BLACK,
    COLOR_GRAY,
    COLOR_VERMILLION,
    to_numpy,
)
from teb_vae.lag_attn.nets.lag_report import (  # noqa: E402
    MECHANICAL_SHIFT_SECONDS,
    SECONDS_PER_STEP,
)
from teb_vae.lag_attn_cfs.sample_page import (  # noqa: E402
    _draw_anchor_overlay,
    _tiling_anchors,
)
from teb_vae.lag_attn_rws.sample_page import (  # noqa: E402
    BAND_SIGMAS,
    FORECAST_ROW,
    ForecastRowInputs,
    _denormalised,
    raw_context_row,
)
from utils.style import style_axes  # noqa: E402

__all__ = ["LAG_TIME_CAVEAT", "causal_raw_forecast_rows"]

#: The caveat every lag-resolved panel on this page is read under. Stated once, as a page footnote,
#: because the two lag panels are the shared builder's and the six shipped models must not gain a
#: caption about a transform they do not use. Asserted as a string by the suite, so it cannot be
#: dropped by an edit that keeps the figure rendering.
#:
#: It is deliberately **one-sided** where the causal-feature page's is two-sided: the correction to a
#: physical delay has a term for each side of the attention, and this cell's target side contributes
#: none -- a raw sample is at the instant it is at. What is left is the source channel's own
#: composed group delay, plus the shift the preprocessing already removed.
#:
#: It states the **identity** and not a number, because $\tau^u_{\mathrm{ref}}$ is a decision of the
#: run rather than a property of this module and nothing the page is handed carries it: the rows are
#: given arrays, a geometry and the loader's statistics, and the model's own ``source_delay_steps``
#: is the largest *stored-step* shift -- attained by the channel furthest from the reference -- and
#: is emphatically not it. The resolved value travels in the run's own record instead, which the
#: sentence names so a reader of a page can find the constant that completes it.
#:
#: $\Delta$ and the acquisition shift are interpolated from
#: :mod:`~teb_vae.lag_attn.nets.lag_report` rather than typed, so the caption and the function a
#: consumer would evaluate it with cannot state two different corrections.
LAG_TIME_CAVEAT = (
    f"Lag axes are stored-coefficient time on the input side, not physical delay: the raw target "
    f"carries no group delay, so $\\tau^y \\equiv 0$ and the anchor is exact, while a causal input "
    f"coefficient lags by its own composed group delay (13-791 s as declared). Aligned onto one "
    f"source reference $\\tau^u_{{\\mathrm{{ref}}}}$ -- the run's own, logged as "
    f"source_reference_delay_s -- that bias is a single constant, and a peak at lag $\\ell$, "
    f"horizon element $h$, is a physical lead time of "
    f"${SECONDS_PER_STEP:.0f}(\\ell + 1 + h) + \\tau^u_{{\\mathrm{{ref}}}} - "
    f"{MECHANICAL_SHIFT_SECONDS:.0f}$ s, the $-{MECHANICAL_SHIFT_SECONDS:.0f}$ s being the "
    f"acquisition shift preprocessing already removed from the source trace. The correction is "
    f"one-sided here: there is no target-side $\\tau^y$ term to subtract. Unaligned, "
    f"$\\tau^u_{{\\mathrm{{ref}}}}$ is replaced by each channel's own $\\tau^u_c$ and no single "
    f"number labels the axis."
)


def _tiled_branch(
    rows: ForecastRowInputs,
    branch: str,
    anchors: np.ndarray,
    positions: Sequence[int],
) -> List[np.ndarray]:
    r"""Tile one branch's forecast onto the **raw** grid: mean and both band edges.

    Horizon token $h$ of anchor $t$ holds raw samples
    $[\,\mathrm{start}(t) + R h,\ \mathrm{start}(t) + R(h+1)\,)$ with
    $\mathrm{start}(t) = \texttt{future\_block\_start}(t)$, so an anchor's $(H, R)$ block flattens to
    its $H R$ contiguous raw samples in order and the window is one slice assignment. This is the
    same identity :func:`~teb_vae.lag_attn.figure_primitives.concat_single_forecasts` relies on; what
    it cannot do here is find the anchor, because its index is a position in the decoded set.

    Args:
        rows: The row inputs, for the forward dict, the geometry and the loader statistics.
        branch: ``'base'`` or ``'full'``.
        anchors: The decoded anchor indices of the sample being drawn, $(A_{\max},)$.
        positions: Which positions of the anchor axis to draw, from ``_tiling_anchors``.

    Returns:
        ``[mean, lower, upper]``, each $(L_{\mathrm{raw}},)$ in the unit the context row drew,
        ``NaN`` where no drawn window covers the sample -- which renders as a gap rather than as a
        fabricated continuation.
    """
    index, geometry = rows.sample_index, rows.geometry
    mean = rows.outs[f"mu_{branch}"][index]
    sigma = torch.exp(0.5 * rows.outs[f"logvar_{branch}"][index])
    block = int(geometry.horizon) * int(geometry.r)

    tiled: List[np.ndarray] = []
    for curve in (mean, mean - BAND_SIGMAS * sigma, mean + BAND_SIGMAS * sigma):
        # Denormalized through the same affine map as the truth, so the three curves in the panel
        # are directly comparable and a forecast can be checked against physiology by eye.
        values = np.asarray(
            _denormalised(curve, "fhr", rows.normalization_stats)[0], dtype=np.float64
        ).reshape(int(curve.shape[0]), block)
        out = np.full(int(geometry.raw_len), np.nan, dtype=np.float64)
        for position in positions:
            start = geometry.future_block_start(int(anchors[position]))
            out[start : start + block] = values[position]
        tiled.append(out)
    return tiled


def causal_raw_forecast_rows(rows: ForecastRowInputs, *, training_stride: int = 1) -> None:
    r"""Draw this page's first two rows, over the anchors the forward actually decoded.

    Bound to the net's tiling by the task and handed to
    :func:`~teb_vae.lag_attn_rws.sample_page.build_diagnostic_figure` as its ``forecast_rows`` seam.
    Row $1$ is the shared raw-context row -- this model's target *is* the raw trace, so it is drawn
    once and both rows read the same array in the same unit. Row $2$ is the tiling, and it is the
    row this whole module exists for.

    The tiling is a **choice of what to draw**, not a change to what the model does: ``outs`` already
    carries a forecast for every anchor the forward decoded, all of which the objective scored, and
    this row selects the subset whose windows abut without overlapping. The page is produced at the
    evaluation resolution -- every valid anchor, at stride $1$ -- so consecutive decoded windows
    overlap $(H-1)/H$ and plotted at that stride the row would show each instant $H$ times over, from
    $H$ different latents.

    Args:
        rows: The row inputs and the layout hooks. ``rows.outs`` must carry ``anchor_index`` and
            ``anchor_valid``: the forecast tensors are indexed by *position in the decoded set*, not
            by anchor, so without them a window is drawn at the wrong time with no shape error
            anywhere in it.
        training_stride: $S$, so the overlay can mark the tile grid a training step would use beside
            the dense set this page is produced at. Bound by the task from the net's own
            ``anchor_stride``.

    Raises:
        KeyError: If the forward dict carries no anchor set.
    """
    index, geometry = rows.sample_index, rows.geometry
    horizon, raw_per_step = int(geometry.horizon), int(geometry.r)
    block = horizon * raw_per_step

    # ---- Row: the two raw traces ------------------------------------------
    # The target *is* the raw trace here, so the context row and the forecast row draw the same
    # signal and the conversion is done once, by the row that owns it.
    fhr_np, unit = raw_context_row(rows, rows.target[index])

    anchors = to_numpy(rows.outs["anchor_index"][index]).astype(int).ravel()
    valid = to_numpy(rows.outs["anchor_valid"][index]).astype(bool).ravel()
    positions = _tiling_anchors(anchors, valid, horizon)

    base_mean, base_lo, base_hi = _tiled_branch(rows, "base", anchors, positions)
    full_mean, full_lo, full_hi = _tiled_branch(rows, "full", anchors, positions)
    # The truth restricted to the drawn support, so the row is about the predicted windows alone and
    # an uncovered span reads as absent rather than as unpredicted.
    truth_tiled = np.where(np.isfinite(full_mean), fhr_np, np.nan)

    # Seconds per raw sample, taken from the axis the page was built on rather than from a sampling
    # rate restated here: every mark below has to land on `rows.time_raw`, which is that axis.
    seconds_per_sample = rows.t_max / float(geometry.raw_len)
    seconds_per_step = rows.t_max / float(geometry.t)

    # ---- Row: the forecast, tiled into non-overlapping windows -------------
    ax, cax = rows.row_axes(FORECAST_ROW)
    # The truth first, so it stays ``ax.lines[0]`` -- the sibling pages' tests read it from there.
    ax.plot(
        rows.time_raw, truth_tiled, color=COLOR_BLACK, linewidth=0.7, label="true $Y^{+}$"
    )
    for mean, low, high, colour, alpha, style, label in (
        (base_mean, base_lo, base_hi, COLOR_GRAY, 0.22, "--", "base ($z^p$, target-only)"),
        (
            full_mean, full_lo, full_hi, COLOR_VERMILLION, 0.18, "-",
            "full ($z^q$, source-conditioned)",
        ),
    ):
        ax.fill_between(rows.time_raw, low, high, color=colour, alpha=alpha, linewidth=0)
        ax.plot(
            rows.time_raw, mean, color=colour, linewidth=0.8, linestyle=style, label=label
        )

    # Where one forecast ends and the next begins. Without them the tiling reads as a single
    # continuous prediction, which is exactly what it is not: each window is decoded from one latent
    # and never sees the window before it. Walked off the drawn anchors rather than off the geometry,
    # for the same reason every other mark on this row is, and deduplicated because consecutive
    # drawn windows abut: one boundary is one line, drawn once, at one alpha.
    window_edges = sorted(
        {
            edge
            for position in positions
            for edge in (
                geometry.future_block_start(int(anchors[position])),
                geometry.future_block_start(int(anchors[position])) + block,
            )
        }
    )
    for edge in window_edges:
        ax.axvline(
            edge * seconds_per_sample, color=COLOR_GRAY, linewidth=0.5, linestyle="--",
            alpha=0.7, zorder=1,
        )

    window_seconds = block * seconds_per_sample
    ax.set_title(
        f"Forecast — {len(positions)} of {int(valid.sum())} decoded anchors drawn, as consecutive "
        f"non-overlapping {window_seconds:.0f} s windows, mean $\\pm$ {BAND_SIGMAS:.0f}$\\sigma$ "
        f"({horizon}$\\times${raw_per_step} = {block} raw samples each; dashed: window edges).\n"
        f"Drawn at the evaluation resolution — every valid anchor — so the ticks are what this page "
        f"decoded; the dotted grid is the sparser set a training step tiles.",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel(f"FHR ({unit})", fontsize=8)
    style_axes(ax, grid="both")
    rows.finalise_time_axis(ax)
    _draw_anchor_overlay(ax, rows, anchors, valid, seconds_per_step, int(training_stride))
    ax.legend(loc="upper right", fontsize=6, framealpha=0.95, ncol=2)
    cax.set_visible(False)

    # The footnote the lag panels are read under. Added here rather than to their titles because
    # those panels are the shared builder's and are drawn for six other models whose transform this
    # caveat is not about. Placed inside the GridSpec's bottom margin, so it costs no row.
    ax.figure.text(
        0.5, 0.004, LAG_TIME_CAVEAT,
        ha="center", va="bottom", fontsize=7, color=COLOR_GRAY, wrap=True,
    )
