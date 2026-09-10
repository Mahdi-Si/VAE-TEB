r"""The per-sample diagnostic page for the lag-residual causal-feature forecaster.

One figure per drawn sample, from a single forward pass. The page is the causal sibling's read
against an **anchor-indexed** latent, with the two attention rows replaced by four this
architecture can actually compute.

**Why this package owns a page at all.** The shared builder in
:mod:`teb_vae.lag_attn_rws.sample_page` reads ``attn_weights`` and ``source_kl_lag_map`` before it
consults its own row filter, and this model emits neither. Its lower rows also slice a time-first
array over $[F, T_{\mathrm{valid}})$, which is right for the sibling -- whose latent is produced at
every stored step and gathered at anchors only for the decoder -- and silently wrong here, where
**every** latent tensor carries an anchor axis. A page that merely dropped two rows would still
draw the remaining three at the wrong columns, with no shape error to say so.

**What is reused rather than rewritten.** The tiling, the stitching, the five field rows, the
anchor overlay, the inset error map, the per-channel profiles and the input-stream panels are the
causal package's own and are imported, not copied: two pages of one recording that disagreed about
which anchors were drawn would be two pictures that only look aligned. What is written here is the
layout, the forecast lane row, the window score and the seven rows below the input rows.

**The window score is this module's and it has to be.** The causal page's version applies neither
the channel weighting nor the horizon weighting, and this cell configures both, so its curves
would not be the ``pred_gap`` the page's own title and the training curve carry.
:func:`_weighted_window_scores` reduces through the objective's own
:func:`~teb_vae.lag_attn_rws.nets.losses.masked_raw_block_per_anchor`, handed the model's own
target gather and pooled validity, so a value on the row is a value in the log.

**Every model row is at the anchor step; only the raw row is physical.** One column is one anchor
on the forecast, input, latent, divergence and lag rows alike, and each row states its clock on its
x label. Nothing is shifted to compensate for a group delay: the stored timeline is canonical.

**The four lag rows are a suppression reading, not an attribution.** There is no nonnegative
per-lag split of the divergence for a summed update, because the cross terms can reinforce or
cancel. What is drawn instead is what removing a lag does to the fitted computation, which is a
different and answerable question --
:data:`~teb_vae.lag_slot_transformer_cfs.nets.controls.SUPPRESSION_QUALIFICATION` states its limit
and is written into the figure rather than only into this docstring. The rows exist only where the
lags are summed; under the comparator's normalised aggregation there is no per-lag term to remove
and they are omitted rather than faked.

This module is matplotlib-only: no Lightning, no MLflow, no config, no loader. It does not call
:func:`~utils.style.apply_publication_style`, which mutates global ``rcParams`` and is called once
by whoever owns the process.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from teb_vae.lag_attn.figure_primitives import (  # noqa: E402
    COLOR_BLACK,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_LIGHT_GRAY,
    COLOR_ORANGE,
    COLOR_VERMILLION,
    attach_lag_seconds_axis,
    safe_vabs,
    sample_cell_edges,
    select_forecast_channels,
    shade_warmup,
    time_axes,
    to_numpy,
)
from teb_vae.lag_attn.nets.lag_report import COMPENSATED_LAG_AXIS_LABEL  # noqa: E402
from teb_vae.lag_attn_cfs.sample_page import (  # noqa: E402
    LAG_TIME_CAVEAT,
    _ERROR_MAP_VERTICAL,
    _LANE_HEADROOM,
    _NOMINAL_COVERAGE,
    _PROFILE_VERTICAL,
    _channel_profile,
    _draw_anchor_overlay,
    _draw_field_rows,
    _forecast_time_label,
    _prefix_boxes,
    _profile_inset,
    _scored_clock_view,
    _Stitched,
    _tail_anchor,
    _tiled_branch,
    _tiling_anchors,
    causal_stream_panels,
)
from teb_vae.lag_attn_fs.sample_page import (  # noqa: E402
    FORECAST_CHANNELS,
    _batch_field,
    _draw_context_row,
    _draw_error_map,
    _resolved_keep_index,
)
from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor  # noqa: E402
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask  # noqa: E402
from teb_vae.lag_attn_rws.sample_page import (  # noqa: E402
    BAND_SIGMAS,
    FORECAST_ROW,
    ForecastRowInputs,
    InputStreamPanel,
    _input_stream_row,
    _raw_overlay,
    top_down_extent,
)
from teb_vae.lag_slot_transformer_cfs.nets.controls import (  # noqa: E402
    SUPPRESSION_QUALIFICATION,
    suppressed_parameters,
)
from utils.style import style_axes  # noqa: E402

__all__ = [
    "LAG_ROWS",
    "PROPOSAL_ROWS",
    "ResidualLagPanels",
    "build_residual_page",
    "causal_stream_panels",
    "residual_forecast_rows",
    "residual_lag_panels",
    "wrapped_caption",
]

#: Raw sampling rate of the stored record, in Hz. The page's own $\Delta$ is re-derived from it and
#: from the decimated length rather than taken from a module constant, so a run at another
#: decimation draws the axis it actually has.
_FS_RAW = 4.0

#: Fixed physical strip for the two-line suptitle. A fraction would shrink with every row added and
#: put the title on top of the first panel's own.
_HEADER_INCHES = 0.75

#: Fixed physical strip for the two footnotes, which is wider than the sibling's because this page
#: carries two: the family's lag-time caveat and this architecture's suppression qualification. A
#: floor rather than a replacement; see :func:`build_residual_page`.
_FOOTER_INCHES = 1.7

#: Where each footnote's baseline sits, in inches from the figure's bottom edge. In inches and not
#: in figure fractions for the reason the header strip is: a fraction moves with the row count, and
#: this page's row count depends on the arm.
_CAVEAT_INCHES = 0.65
_QUALIFICATION_INCHES = 0.12

#: Characters per line of a footnote, and the reason this page wraps its own captions rather than
#: asking matplotlib to.
#:
#: ``Text(wrap=True)`` finds its break points by **measuring**: it re-measures the line so far once
#: per word, and every one of those measurements re-parses whatever mathtext the line contains, so
#: the cost grows with the square of the caption's length. MEASURED at the production geometry:
#: the two captions below cost $7.9$ s of a $20.7$ s page, for two blocks of prose. Wrapped here
#: they cost nothing measurable.
#:
#: Whitespace is a safe break point for these two strings and stays safe: no ``$...$`` span in
#: either contains a space, so a break can never land inside one, and
#: :func:`_test_captions_break_only_between_math_spans` in the package's page tests holds that.
#: The count suits $7$ pt over this page's $14$ inch width with a margin for the mathtext, which
#: sets wider than the prose around it.
_CAPTION_CHARS = 235

#: Never interpolate a heatmap on this page: a cell is one anchor by one channel or one lag, and a
#: smoothed edge invents a value between two the model produced.
_IMSHOW_INTERPOLATION = "none"

#: The four rows that replace the sibling's attention matrix and divergence-by-lag map, with their
#: height ratios. Present only on the arm that sums per-lag updates.
LAG_ROWS: Tuple[Tuple[str, float], ...] = (
    ("lag_proposal", 1.2),
    ("lag_exposure", 1.2),
    ("lag_suppression", 1.2),
    ("cancellation", 0.85),
)

#: The three rows about the latent itself, which every arm has.
PROPOSAL_ROWS: Tuple[Tuple[str, float], ...] = (
    ("latent", 1.2),
    ("kld_dims", 1.1),
    ("kld_total", 0.85),
)


def wrapped_caption(text: str) -> str:
    """Break one caption into lines here, rather than leaving it to the renderer.

    See :data:`_CAPTION_CHARS` for why. Breaking on whitespace cannot split a ``$...$`` span in
    either caption this page draws, because neither contains one with a space in it.

    Args:
        text: The caption, as one paragraph.

    Returns:
        The same text with newlines at the break points.
    """
    return "\n".join(textwrap.wrap(text, width=_CAPTION_CHARS))


@dataclass(frozen=True)
class ResidualLagPanels:
    r"""The per-lag readouts of one sample, as plain arrays.

    Built by :func:`residual_lag_panels`, which is the only thing here that touches the net, so
    the drawing below stays a drawing module. Every array's first axis is the anchor and its
    second, where it has one, the candidate lag $\ell$.

    Attributes:
        proposal_norm: $\| r^\mu_{t,\ell} \|_2$, $(A, L)$ -- what the head emitted before
            the sum, the limiter and the prior-relative parameters.
        exposure: Available source channels per anchor-lag pair, $(A, L)$. The denominator of
            every row beside it: a lag that carried two channels of the declared width and a lag
            that carried all of them are not the same measurement.
        suppression: $K_t - K_t^{\setminus \ell}$, $(A, L)$ -- how much the anchor's divergence
            falls when that lag's proposals alone are removed and everything else is held fixed.
        cancellation_ratio: $\kappa_t$, $(A,)$.
        cancellation_numerator: $\| \sum_\ell r_{t,\ell} \|_2$, $(A,)$.
        cancellation_denominator: $\sum_\ell \| r_{t,\ell} \|_2$, $(A,)$.
        lag_valid: Whether the anchor-lag pair carried any available channel at all, $(A, L)$.
            The two maps above are masked to it, so an out-of-range lag reads as absent rather
            than as a measured zero.
    """

    proposal_norm: np.ndarray
    exposure: np.ndarray
    suppression: np.ndarray
    cancellation_ratio: np.ndarray
    cancellation_numerator: np.ndarray
    cancellation_denominator: np.ndarray
    lag_valid: np.ndarray


@torch.no_grad()
def residual_lag_panels(
    model: Any, outs: Dict[str, Any], *, sample_index: int = 0
) -> Optional[ResidualLagPanels]:
    r"""Build one sample's per-lag readouts from a forward taken with the proposals retained.

    **The suppression map is recomputed from the cached proposals**, through the same
    :func:`~teb_vae.lag_slot_transformer_cfs.nets.controls.suppressed_parameters` the evaluation's
    band margins use. Removal is a subtraction from the matched raw update, so no encoder and no
    head runs again and the whole map costs $L$ passes of cheap tensor arithmetic over one
    sample's proposals.

    Args:
        model: The net, for the fusion arm and the two residual bounds.
        outs: A forward dict taken with ``return_proposals=True``.
        sample_index: Which sample of the batch to read.

    Returns:
        The panels, or ``None`` on an arm with no per-lag terms to remove -- the comparator's
        normalised aggregation, or a target-only model with no source pathway at all. A caller
        that gets ``None`` draws the page without the four lag rows rather than with four empty
        ones.
    """
    if str(getattr(model, "lag_fusion", "local")) != "local":
        return None
    if "mean_proposals" not in outs or "lag_valid" not in outs:
        return None

    index = int(sample_index)
    proposals = outs["mean_proposals"]  # (B, A, L, d_z)
    lag_valid = outs["lag_valid"]  # (B, A, L)
    n_lags = int(proposals.shape[2])

    matched = outs["kld_per_anchor"][index]  # (A,)
    suppression = torch.empty(
        (matched.shape[0], n_lags), dtype=matched.dtype, device=matched.device
    )
    for lag in range(n_lags):
        # One lag at a time rather than a band: the row's question is what this lag alone
        # contributes to the fitted computation, and a band would answer a coarser one.
        removed = torch.zeros(n_lags, dtype=torch.bool, device=proposals.device)
        removed[lag] = True
        suppression[:, lag] = matched - suppressed_parameters(model, outs, removed)[
            "kld_per_anchor"
        ][index]

    # The gathered per-channel availability where the forward kept it, and the coarser anchor-lag
    # indicator otherwise. Both answer "how much source did this lag have", at the two resolutions
    # a forward can be asked for.
    channel_mask = outs.get("source_channel_mask")
    exposure = (
        lag_valid[index].to(torch.float64)
        if channel_mask is None
        else channel_mask[index].to(torch.float64).sum(dim=-1)
    )

    return ResidualLagPanels(
        proposal_norm=to_numpy(proposals[index].norm(dim=-1)),
        exposure=to_numpy(exposure),
        suppression=to_numpy(suppression),
        cancellation_ratio=to_numpy(outs["cancellation_ratio_mean"][index]),
        cancellation_numerator=to_numpy(outs["cancellation_numerator_mean"][index]),
        cancellation_denominator=to_numpy(outs["cancellation_denominator_mean"][index]),
        lag_valid=to_numpy(lag_valid[index]).astype(bool),
    )


def _weighted_window_scores(
    rows: ForecastRowInputs,
    stitched: _Stitched,
    *,
    likelihood: str,
    coverage_floor: float,
    forecast_target: Callable[..., torch.Tensor],
    scored_weight: Callable[[torch.Tensor], torch.Tensor],
    channel_weight: Optional[torch.Tensor],
    horizon_weight: Optional[torch.Tensor],
) -> Optional[Dict[str, np.ndarray]]:
    r"""Each drawn window's block score, reduced exactly as the objective reduces it.

    The causal page's own version sums the elementwise term against the mask and applies neither
    weighting. This cell configures both a channel weighting and a horizon weighting, so that
    version's curves would be in different units from the ``nll_base_block``, ``nll_full_block``
    and ``pred_gap`` the page's title carries and the training curve plots -- a diagnostic
    disagreeing with the run it diagnoses.

    Both the gather and the pooled validity are the **model's own methods**, handed in bound, so
    the block scored here is the block the objective scored rather than a second construction of
    it.

    Args:
        rows: The row inputs; ``rows.batch`` is the only route to the validity signal and
            ``rows.target`` is the declared-width target stream.
        stitched: The drawn tiling, for which positions to report.
        likelihood: The objective's own likelihood.
        coverage_floor: The model's own anchor coverage floor.
        forecast_target: The model's target gather, ``(stream, anchors) -> (B, A, H, C_keep)``.
        scored_weight: The model's pooled validity, ``(B, T) -> (B, T)``.
        channel_weight: Per-channel weight on the block's last axis, or ``None``.
        horizon_weight: Per-step weight on the horizon axis, or ``None``.

    Returns:
        ``{'base': (W,), 'full': (W,)}`` over the drawn windows, or ``None`` when the batch
        carries no validity signal -- the objective's mask is a function of it, and scoring an
        invalid span would put a spike on the row the objective never saw.
    """
    weight = _batch_field(rows.batch, "weight")
    if weight is None or not stitched.positions:
        return None

    index = rows.sample_index
    positions = list(stitched.positions)
    with torch.no_grad():
        mask, _coverage = forecast_mask(
            scored_weight(weight),
            rows.geometry,
            coverage_floor=float(coverage_floor),
            anchors=rows.outs["anchor_index"],
            anchor_valid=rows.outs["anchor_valid"],
        )
        target = forecast_target(rows.target, rows.outs["anchor_index"])
        scores: Dict[str, np.ndarray] = {}
        for branch in ("base", "full"):
            block, _contributing = masked_raw_block_per_anchor(
                rows.outs[f"mu_{branch}"],
                target,
                mask,
                likelihood=likelihood,
                logvar=rows.outs[f"logvar_{branch}"],
                channel_weight=channel_weight,
                horizon_weight=horizon_weight,
            )
            scores[branch] = to_numpy(block[index])[positions].astype(float)
    return scores


def _draw_weighted_gap_row(
    rows: ForecastRowInputs,
    stitched: _Stitched,
    *,
    scores: Optional[Dict[str, np.ndarray]],
    likelihood: str,
    coverage_floor: float,
    seconds_per_step: float,
    weighted: bool,
) -> None:
    r"""Draw ``pred_gap``: each drawn window's block score, base against full, plus the profiles.

    The causal page's row with the score handed in rather than computed inside, which is what lets
    this one carry the objective's weighted reduction. Everything else -- the signed shaded gap,
    the two profile insets, the fallback when no score can be built -- is that row's behaviour.

    Args:
        rows: The row inputs and the layout hooks.
        stitched: The drawn tiling.
        scores: ``{'base', 'full'}`` over the drawn windows, or ``None``.
        likelihood: The objective's own likelihood, for the axis label and the title.
        coverage_floor: The model's own anchor coverage floor, for the title.
        seconds_per_step: $\Delta$ in seconds, for placing a window in physical time.
        weighted: Whether a channel or horizon weighting is in force, which decides whether the
            row may call its unit nats at all.
    """
    ax, cax = rows.row_axes("pred_gap")
    cax.set_visible(False)
    ax.set_xlabel(stitched.time_label, fontsize=8)
    # A weighted block score is not a log density, and saying nats over one would invite a reader
    # to compare it with an unweighted run's columns.
    unit = "nats" if likelihood == "gaussian_nll" else "sq. error"
    if weighted:
        unit = f"weighted {unit}"
    ax.set_ylabel(f"block score ({unit})", fontsize=8)
    style_axes(ax, grid="both")

    if scores is None:
        # The row keeps its title, its axis and its place, so the rows below stay column-aligned
        # and the gap is visible.
        ax.set_title("Per-window forecast score", fontsize=9, pad=6)
        ax.text(
            0.5, 0.5, "no validity signal in this batch, so the objective's mask cannot be built",
            transform=ax.transAxes, ha="center", va="center", fontsize=8, color=COLOR_GRAY,
        )
        rows.finalise_time_axis(ax)
        return

    horizon = int(rows.geometry.horizon)
    # A window's mark sits at the centre of the span it scores, so it lands over the same columns
    # the field rows above drew it in.
    centres = np.asarray(
        [
            (int(stitched.anchors[position]) + 1 + 0.5 * horizon) * seconds_per_step
            for position in stitched.positions
        ],
        dtype=float,
    )
    base, full = scores["base"], scores["full"]

    for condition, colour, label in (
        (base >= full, COLOR_GREEN, "source helps ($D_0 > D_1$)"),
        (base < full, COLOR_VERMILLION, "source hurts"),
    ):
        ax.fill_between(
            centres, base, full, where=condition, color=colour, alpha=0.25, linewidth=0,
            interpolate=True, label=label,
        )
    ax.plot(
        centres, base, color=COLOR_GRAY, linewidth=0.9, linestyle="--", marker="o", markersize=2.5,
        label="$D_0$ base ($z^p$, target-only)",
    )
    ax.plot(
        centres, full, color=COLOR_VERMILLION, linewidth=0.9, marker="o", markersize=2.5,
        label="$D_1$ full ($z^q$, source-conditioned)",
    )

    gap = float(np.mean(base - full)) if base.size else float("nan")
    ax.set_title(
        f"Per-window forecast score, reduced as the objective reduces it "
        f"('{likelihood}', {'channel- and horizon-weighted' if weighted else 'unweighted'}, "
        f"masked at the model's coverage floor {coverage_floor:g}) — "
        f"{base.size} windows, mean gap $D_0-D_1$ = {gap:.4g} over the drawn set.\n"
        f"Insets: per-channel error and $\\pm${BAND_SIGMAS:.0f}$\\sigma$ coverage over the same "
        f"windows, on the channel axis of the rows above.",
        fontsize=9, pad=6,
    )
    ax.legend(loc="upper right", fontsize=6, framealpha=0.95, ncol=2)
    rows.finalise_time_axis(ax)

    base_rmse, base_coverage = _channel_profile(stitched, "base")
    full_rmse, full_coverage = _channel_profile(stitched, "full")
    error_box, coverage_box = _prefix_boxes(rows, 2, _PROFILE_VERTICAL)
    _profile_inset(
        ax, error_box, stitched,
        (
            (base_rmse, COLOR_GRAY, "--", "base"),
            (full_rmse, COLOR_VERMILLION, "-", "full"),
        ),
        xlabel="RMSE",
    )
    _profile_inset(
        ax, coverage_box, stitched,
        (
            (base_coverage, COLOR_GRAY, "--", "base"),
            (full_coverage, COLOR_VERMILLION, "-", "full"),
        ),
        xlabel=f"{BAND_SIGMAS:.0f}$\\sigma$ coverage",
        reference=_NOMINAL_COVERAGE,
    )


def residual_forecast_rows(
    rows: ForecastRowInputs,
    *,
    keep_index: Optional[Sequence[int]] = None,
    block_split: Optional[int] = None,
    training_stride: int = 1,
    likelihood: str = "gaussian_nll",
    coverage_floor: float = 0.0,
    target_forecast_shift: Optional[Sequence[int]] = None,
    forecast_clock_delay_s: Optional[float] = None,
    forecast_target: Optional[Callable[..., torch.Tensor]] = None,
    scored_weight: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    channel_weight: Optional[torch.Tensor] = None,
    horizon_weight: Optional[torch.Tensor] = None,
) -> None:
    r"""Draw this page's forecast rows, over the anchors the forward decoded.

    The causal sibling's rows, with one difference: the per-window score is the objective's own
    weighted reduction rather than the unweighted sum, because this cell configures a channel
    weighting and a horizon weighting and the sibling's row applies neither.

    Row $1$ is the shared raw-context row, drawn from the batch because this model's target is a
    feature block. Row $2$ is the stitched forecast with the training-tile fan over it. The six
    rows below it are :data:`~teb_vae.lag_attn_cfs.sample_page.CAUSAL_EXTRA_ROWS`, five of which
    are the sibling's own field rows, unchanged.

    Args:
        rows: The row inputs and the layout hooks.
        keep_index: Declared channel index of each decoder output lane, or ``None`` to take the
            lanes as declared.
        block_split: Declared width of the first stored target block, for the channel divider.
        training_stride: $S$, so the overlay can draw the sparser grid a training step tiles
            beside the dense set this page draws.
        likelihood: The objective's own likelihood.
        coverage_floor: The model's own anchor coverage floor, so a window the objective dropped
            is dropped from the score row too.
        target_forecast_shift: $s_c$ per kept channel, the model's own forecast clock. ``None``
            -- the stored clock -- draws the stream as stored.
        forecast_clock_delay_s: $\tau$ of that clock in seconds, so the forecast rows' axis can
            state how far before the scored step their content sits. Nothing drawn moves with it.
        forecast_target: The model's own target gather. ``None`` leaves the score row on its
            fallback, which is what a hand-built task with no net gets.
        scored_weight: The model's own pooled validity. ``None`` behaves as above.
        channel_weight: The objective's per-channel weight, or ``None``.
        horizon_weight: The objective's per-horizon-step weight, or ``None``.

    Raises:
        KeyError: If the forward dict carries no anchor set.
        ValueError: If ``keep_index`` does not have one entry per forecast channel.
    """
    index, geometry = rows.sample_index, rows.geometry
    _draw_context_row(rows)

    # The footnote the lag rows are read under, in the GridSpec's bottom margin so it costs no
    # row, and written before the early return: a page that keeps the lag rows and drops the
    # forecast rows is exactly the page that leads with a lag axis. Positioned from the figure's
    # own height rather than at a fixed fraction, because this page's row count depends on the arm
    # and a fraction that clears the last row on one lands inside it on another.
    rows.figure.text(
        0.5, _CAVEAT_INCHES / float(rows.figure.get_figheight()),
        wrapped_caption(LAG_TIME_CAVEAT),
        ha="center", va="bottom", fontsize=7, color=COLOR_GRAY,
    )

    if not rows.wants(FORECAST_ROW):
        return

    anchors = to_numpy(rows.outs["anchor_index"][index]).astype(int).ravel()
    valid = to_numpy(rows.outs["anchor_valid"][index]).astype(bool).ravel()
    positions = _tail_anchor(
        anchors, valid, _tiling_anchors(anchors, valid, int(geometry.horizon)),
        int(geometry.horizon),
    )

    base_mean, base_sigma = _tiled_branch(rows, "base", anchors, positions)
    full_mean, full_sigma = _tiled_branch(rows, "full", anchors, positions)
    keep = _resolved_keep_index(keep_index, full_mean.shape[-1])

    # Re-indexed once and shared with the error map below: a second gather from the stored clock
    # is how an inset comes to score a physical-clock forecast against stored-clock truth.
    stream = to_numpy(rows.target[index])
    scored_stream = _scored_clock_view(stream[:, keep], target_forecast_shift)
    truth = np.where(np.isfinite(full_mean), scored_stream, np.nan)

    # The model's own anchor ceiling, re-derived as the net derives it: under an advancing clock
    # the trailing anchors read past the stored record and are never built, so the overlay's tile
    # grid and ceiling mark must stop where the anchors do.
    anchor_ceiling = int(geometry.t_valid) - max(
        [0, *(int(shift) for shift in target_forecast_shift or ())]
    )

    stitched = _Stitched(
        truth=truth,
        base_mean=base_mean,
        base_sigma=base_sigma,
        full_mean=full_mean,
        full_sigma=full_sigma,
        keep=keep,
        block_split=(
            0 if block_split is None else int(np.count_nonzero(keep < int(block_split)))
        ),
        anchors=anchors,
        positions=positions,
        time_label=_forecast_time_label(target_forecast_shift, forecast_clock_delay_s),
    )

    lanes, coverage = select_forecast_channels(
        truth, full_mean, full_sigma, count=FORECAST_CHANNELS, n_sigmas=BAND_SIGMAS
    )

    ax, cax = rows.row_axes(FORECAST_ROW)
    _, time_dec, _ = time_axes(geometry.t, geometry.raw_len)
    seconds_per_step = rows.t_max / float(geometry.t)

    def lane_extent(channel: int) -> float:
        """Total vertical span the widest artist of one lane needs.

        Both branches are measured. The lane draws the target-only band too, and a prior wider
        than the full distribution is the ordinary case: sizing the stride off the narrower
        branch alone lets the wider band run into the lane above, so a reader attributes one
        channel's uncertainty to the channel labelled there.
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

    # The forecasts a TRAINING step would have scored, as a fan. The stitched lanes above show at
    # most one latent per step, which is a fraction of what a training step tiles; the fan is
    # where the rest become visible and where a forecast's drift with lead time can be read
    # against the same truth.
    tile_positions = [
        position for position, (anchor, is_valid) in enumerate(zip(anchors, valid))
        if is_valid and (int(anchor) - int(geometry.warmup)) % int(training_stride) == 0
    ]
    fan_mean = to_numpy(rows.outs["mu_full"][index])
    for lane, channel in enumerate(int(value) for value in lanes):
        offset = lane * stride
        for count, position in enumerate(tile_positions):
            start = int(anchors[position]) + 1
            stop = min(start + int(geometry.horizon), geometry.t)
            ax.plot(
                time_dec[start:stop], fan_mean[position, : stop - start, channel] + offset,
                color=COLOR_VERMILLION, linewidth=0.5, alpha=0.45, zorder=1.5,
                label=(
                    f"training-tile forecasts ($\\mu^q$, $S$={int(training_stride)}, "
                    f"$\\varphi$=0)"
                    if lane == 0 and count == 0
                    else None
                ),
            )

    # Each lane named by its **declared** channel, so the number on the axis survives a change of
    # budget: the positional index among the survivors would not.
    ax.set_yticks([lane * stride for lane in range(len(lanes))])
    ax.set_yticklabels(
        [f"ch {int(keep[channel])}\n{coverage[int(channel)]:.0%}" for channel in lanes],
        fontsize=7,
    )

    # The anchor the error map describes: the middle drawn window, a fixed choice rather than the
    # best or worst the run happens to contain.
    position = positions[len(positions) // 2] if positions else 0
    anchor = int(anchors[position])
    anchor_start = float(anchor + 1) * seconds_per_step
    ax.axvspan(
        anchor_start, anchor_start + geometry.horizon * seconds_per_step,
        color=COLOR_ORANGE, alpha=0.14, zorder=0,
    )

    ax.set_title(
        f"Forecast — {len(lanes)} of {len(keep)} target channels by "
        f"{BAND_SIGMAS:.0f}$\\sigma$ calibration (worst, middle, best), lanes offset by "
        f"{stride:.3g}, mean $\\pm$ {BAND_SIGMAS:.0f}$\\sigma$; {len(positions)} of "
        f"{int(valid.sum())} decoded anchors stitched, first anchor wins; shaded: the anchor the "
        f"error map draws.\nDrawn at the evaluation resolution — every valid anchor — so the "
        f"ticks are what this page decoded; the dotted grid is the sparser set a training step "
        f"tiles.",
        fontsize=9, pad=6,
    )
    ax.set_xlabel(stitched.time_label, fontsize=8)
    ax.set_ylabel("target coefficient (normalised)", fontsize=8)
    style_axes(ax, grid="both")
    rows.finalise_time_axis(ax)
    _draw_anchor_overlay(
        ax, rows, anchors, valid, seconds_per_step, int(training_stride), anchor_ceiling
    )
    ax.legend(loc="upper left", fontsize=6, framealpha=0.95, ncol=2)

    # From the SCORED stream, not the stored one: the forecast being subtracted is on the model's
    # own clock, and a stored gather would show every shifted channel a phantom error of
    # truth-minus-differently-timed-truth.
    anchor_truth = scored_stream[anchor + 1 : anchor + 1 + geometry.horizon]
    anchor_mean = to_numpy(rows.outs["mu_full"][index, position])
    _draw_error_map(
        ax,
        cax,
        np.abs(anchor_truth - anchor_mean).T,
        first_block_channels=stitched.block_split,
        anchor_seconds=anchor_start,
        # The blank span on this tiling is the prefix below the anchor floor, not the tail: the
        # windows run to the end of the recording.
        box=_prefix_boxes(rows, 1, _ERROR_MAP_VERTICAL)[0],
    )

    _draw_field_rows(rows, stitched)

    scores = (
        None
        if forecast_target is None or scored_weight is None
        else _weighted_window_scores(
            rows,
            stitched,
            likelihood=likelihood,
            coverage_floor=coverage_floor,
            forecast_target=forecast_target,
            scored_weight=scored_weight,
            channel_weight=channel_weight,
            horizon_weight=horizon_weight,
        )
    )
    _draw_weighted_gap_row(
        rows,
        stitched,
        scores=scores,
        likelihood=likelihood,
        coverage_floor=coverage_floor,
        seconds_per_step=seconds_per_step,
        weighted=channel_weight is not None or horizon_weight is not None,
    )


def build_residual_page(
    *,
    outs: Dict[str, Any],
    target_features: torch.Tensor,
    geometry: Any,
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
    forecast_extra_rows: Sequence[Tuple[str, float]] = (),
    lag_panels: Optional[ResidualLagPanels] = None,
    rows: Optional[Sequence[str]] = None,
) -> Any:
    r"""Draw one sample's page and return the figure, which the caller saves and closes.

    The parameter names are the shared builder's wherever they mean the same thing, so a reader
    who knows that page knows this one. What differs is the tensor contract: every latent quantity
    read here carries an **anchor** axis, and each is drawn at the stored step its anchor sits at
    rather than at its position in the decoded set.

    Args:
        outs: The forward dict, taken with ``return_proposals=True`` where the lag rows are wanted.
        target_features: The declared-width target stream $(B, T, c_y)$, which is what this
            model's task calls its raw target.
        geometry: The trimmed-grid geometry.
        sample_index: Which sample of the batch to draw.
        epoch: For the title.
        guid: For the title.
        beta: The divergence weight **resolved for this epoch**, not the raw hyperparameter.
        scalars: The **epoch's** own loss readouts for the title, keyed as the objective names
            them, and not this sample's: the caller takes them from what the run already logged
            rather than recomputing them, because this architecture's objective ends in a
            collective and the page is drawn on one rank. Missing keys are skipped, and an empty
            mapping leaves the title's second line off entirely.
        up_raw: The raw source in loader units, or ``None``.
        normalization_stats: The loader's statistics, so the raw row renders in physical units.
        delay_steps: The causal input delay $\delta$, for the lag rows' compensated axis.
        forecast_rows: Draws the forecast rows. ``None`` uses :func:`residual_forecast_rows` with
            its own defaults, which is the page a hand-built task gets.
        batch: The loader batch, passed straight to ``forecast_rows``.
        input_streams: The gated input streams, one row each, between the forecast and the latent.
        forecast_extra_rows: ``(name, height_ratio)`` per additional row ``forecast_rows`` draws,
            reserved immediately below the forecast row.
        lag_panels: The per-lag readouts, or ``None`` to draw no lag rows -- which is the honest
            page on an arm that sums no per-lag updates.
        rows: The subset of the page's rows to build, by name, or ``None`` for all of them. An
            unrecognised name raises rather than being ignored: a page quietly missing the panel
            it was rendered for is invisible in the output.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        ValueError: If ``rows`` names a row this page does not reserve.
    """
    index = int(sample_index)
    t_steps = geometry.t
    time_raw, _time_dec, t_max = time_axes(t_steps, geometry.raw_len, fs_raw=_FS_RAW)
    seconds_per_step = t_max / float(t_steps)
    warmup, t_valid = geometry.warmup, geometry.t_valid

    # The anchors this sample actually decoded, in stored steps. Every row below the input rows is
    # drawn against these rather than against a contiguous range: the latent exists at these
    # anchors and nowhere else.
    anchors = to_numpy(outs["anchor_index"][index]).astype(int).ravel()
    valid = to_numpy(outs["anchor_valid"][index]).astype(bool).ravel()
    kept = np.flatnonzero(valid)

    mu_prior_np = to_numpy(outs["mu_prior"][index])[kept]
    delta_mu_np = to_numpy(outs["mu_post"][index] - outs["mu_prior"][index])[kept]
    kld_dims_np = to_numpy(outs["kld_per_anchor_dim"][index])[kept]
    kld_total_np = to_numpy(outs["kld_per_anchor"][index])[kept]
    anchor_steps = anchors[kept]
    d_z = mu_prior_np.shape[1]

    panels = tuple(input_streams or ())
    row_specs: List[Tuple[str, float]] = [
        ("raw", 0.9),
        ("forecast", 1.3),
        *((str(name), float(height)) for name, height in forecast_extra_rows),
        *((f"input_{panel.name}", 1.25) for panel in panels),
        *PROPOSAL_ROWS,
    ]
    if lag_panels is not None:
        row_specs += list(LAG_ROWS)

    if rows is not None:
        # Resolved against the full list, so a name that is legal only on a page with other seams
        # is refused here rather than silently producing a shorter page than the caller asked for.
        reserved = {name for name, _ in row_specs}
        requested = list(dict.fromkeys(str(name) for name in rows))
        unknown = [name for name in requested if name not in reserved]
        if unknown:
            raise ValueError(
                f"unknown page row(s) {unknown}; this page reserves {sorted(reserved)}"
            )
        keep_rows = set(requested)
        row_specs = [spec for spec in row_specs if spec[0] in keep_rows]

    included: FrozenSet[str] = frozenset(name for name, _ in row_specs)
    height_ratios = [height for _, height in row_specs]
    figure_height = sum(height_ratios) * 2.6
    fig = plt.figure(figsize=(14, figure_height))
    try:
        header_frac = _HEADER_INCHES / figure_height
        # A floor, not a replacement: a page long enough for the fraction to be the wider margin
        # keeps the fraction.
        bottom_frac = max(0.03, _FOOTER_INCHES / figure_height)
        grid = GridSpec(
            len(row_specs), 2, figure=fig,
            height_ratios=height_ratios, width_ratios=[1.0, 0.022],
            left=0.065, right=0.93, top=1.0 - header_frac, bottom=bottom_frac,
            # The gutter before the colorbar column has to hold the lag rows' secondary axis, its
            # ticks and its label; too narrow and the label lands under the colorbar.
            hspace=0.55, wspace=0.09,
        )
        row_of = {name: position for position, (name, _) in enumerate(row_specs)}

        def row_axes(name: str) -> Tuple[Any, Any]:
            """Return the ``(main, cax)`` axes pair of a named row."""
            position = row_of[name]
            return fig.add_subplot(grid[position, 0]), fig.add_subplot(grid[position, 1])

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
            """Pin the shared axis and mark the spans this row draws nothing over.

            Args:
                ax: The axes to finalise.
                tail: Whether this row also stops at the last decoded anchor.
            """
            # Every row spans the whole recording whatever it draws, so a column of the page is
            # the same stored step on all of them.
            ax.set_xlim(0.0, t_max)
            shade_warmup(ax, warmup, t_max, t_steps)
            if tail:
                ax.axvspan(
                    float(t_valid) * seconds_per_step, t_max,
                    color=COLOR_LIGHT_GRAY, alpha=0.35, zorder=0,
                )

        def anchor_edges(count: int) -> Tuple[float, float]:
            """The left and right edges of an anchor-column image, in seconds.

            The columns are the decoded anchors, whose spacing is the run's stride rather than one
            stored step, so the cell width is taken from the anchors themselves rather than
            assumed. A single-column page falls back to one step, which is the only spacing a
            lone anchor can be said to have.

            Args:
                count: How many anchor columns the image carries.

            Returns:
                ``(left, right)`` in seconds.
            """
            spacing = seconds_per_step
            if anchor_steps.size > 1:
                spacing = float(np.median(np.diff(anchor_steps))) * seconds_per_step
            first = float(anchor_steps[0]) * seconds_per_step if anchor_steps.size else 0.0
            return sample_cell_edges(max(count, 1), spacing, first=first)

        def lag_row(
            ax: Any,
            values: np.ndarray,
            title: str,
            cmap: str,
            *,
            available: Optional[np.ndarray] = None,
        ) -> Any:
            """Draw one anchors-by-lags map with a compensated-seconds secondary axis.

            Args:
                ax: The row's axes.
                values: The $(A, L)$ map, over the decoded anchors.
                title: The row title.
                cmap: The colormap's name.
                available: Which anchor-lag pairs carried any source at all. Where given, the
                    rest are painted rather than left to the colormap's low end, which is what
                    keeps an unavailable lag from reading as a measured zero.

            Returns:
                The image, for the caller's colorbar.
            """
            drawn = np.asarray(values, dtype=float)
            colormap: Any = cmap
            if available is not None:
                drawn = np.where(available, drawn, np.nan)
                colormap = matplotlib.colormaps[cmap].with_extremes(bad=COLOR_LIGHT_GRAY)
            left, right = anchor_edges(drawn.shape[0])
            image = ax.imshow(
                drawn.T, aspect="auto", cmap=colormap, origin="lower",
                extent=[left, right, -0.5, drawn.shape[1] - 0.5],
                interpolation=_IMSHOW_INTERPOLATION,
            )
            ax.set_title(title, fontsize=9, pad=6)
            ax.set_xlabel("Anchor step (s)", fontsize=8)
            ax.set_ylabel("Lag $\\ell$ (steps)", fontsize=8)
            heatmap_spines(ax)
            secondary = attach_lag_seconds_axis(
                ax,
                # The run's own step, derived from the raw length and the decimated length, rather
                # than a module constant: the primary axis is in stored steps and the secondary
                # one must be the same steps in seconds.
                step_seconds=seconds_per_step,
                # Lag zero already sits at delta steps once the source channels are read that
                # stale, which is the whole of the compensation.
                offset_seconds=float(delay_steps) * seconds_per_step,
            )
            if secondary is not None:
                secondary.set_ylabel(COMPENSATED_LAG_AXIS_LABEL, fontsize=8)
            finalise_time_axis(ax, tail=True)
            return image

        # ---- Rows: whatever this model forecasts -------------------------------
        (forecast_rows or residual_forecast_rows)(
            ForecastRowInputs(
                outs=outs,
                target=target_features,
                batch=batch,
                geometry=geometry,
                sample_index=index,
                normalization_stats=normalization_stats,
                up_raw=up_raw,
                time_raw=time_raw,
                t_max=t_max,
                row_axes=row_axes,
                finalise_time_axis=finalise_time_axis,
                attach_cbar=attach_cbar,
                heatmap_spines=heatmap_spines,
                figure=fig,
                included_rows=included,
            )
        )

        # ---- Rows: the gated input streams, as the encoders receive them --------
        # Drawn over the whole recording rather than cut to the decoded anchors: the warm-up
        # columns are where the guard's zero fill lives, and cutting them would remove the one
        # span the staircase exists to be checked against.
        for panel in panels:
            if f"input_{panel.name}" not in included:
                continue
            ax, cax = row_axes(f"input_{panel.name}")
            image = _input_stream_row(
                ax, panel, t_max=t_max, seconds_per_step=seconds_per_step,
                overlay=_raw_overlay(
                    panel, batch=batch, fhr_raw=None, up_raw=up_raw,
                    sample_index=index, time_raw=time_raw,
                    normalization_stats=normalization_stats,
                ),
            )
            heatmap_spines(ax)
            attach_cbar(cax, image, "value")
            finalise_time_axis(ax)

        # ---- Row: prior mean over the source-derived delta ---------------------
        if "latent" in included:
            ax, cax = row_axes("latent")
            latent_stack = np.concatenate([mu_prior_np, delta_mu_np], axis=1)
            vabs = safe_vabs(latent_stack)
            left, right = anchor_edges(latent_stack.shape[0])
            image = ax.imshow(
                latent_stack.T, aspect="auto", cmap="bwr", origin="upper",
                vmin=-vabs, vmax=vabs,
                extent=[left, right, 2 * d_z - 0.5, -0.5],
                interpolation=_IMSHOW_INTERPOLATION,
            )
            ax.axhline(d_z - 0.5, color="white", linewidth=1.2, linestyle="--")
            ax.set_yticks([d_z // 2, d_z + d_z // 2])
            ax.set_yticklabels(["$\\mu^p$", "$\\mu^q-\\mu^p$"])
            ax.set_title(
                "Target-only latent state and the source-derived shift, at the decoded anchors "
                "(shared colour scale). The design's claim is that the second is small beside "
                "the first.",
                fontsize=9, pad=6,
            )
            ax.set_xlabel("Anchor step (s)", fontsize=8)
            heatmap_spines(ax)
            attach_cbar(cax, image, "value")
            finalise_time_axis(ax, tail=True)

        # ---- Row: per-dimension divergence -------------------------------------
        if "kld_dims" in included:
            ax, cax = row_axes("kld_dims")
            left, right = anchor_edges(kld_dims_np.shape[0])
            # Top-down, matching the latent row directly above: both are indexed by the same
            # latent coordinate, and drawing it upward on one and downward on the other makes the
            # two impossible to read against each other.
            image = ax.imshow(
                kld_dims_np.T, aspect="auto", cmap="magma", origin="upper",
                extent=top_down_extent(left, right, d_z),
                interpolation=_IMSHOW_INTERPOLATION,
            )
            ax.set_title(
                "Per-coordinate source-conditioned divergence (nats), at the decoded anchors — "
                "where a collapse into one or two coordinates shows up",
                fontsize=9, pad=6,
            )
            ax.set_xlabel("Anchor step (s)", fontsize=8)
            ax.set_ylabel("Latent dim", fontsize=8)
            heatmap_spines(ax)
            attach_cbar(cax, image, "nats")
            finalise_time_axis(ax, tail=True)

        # ---- Row: total divergence per anchor ----------------------------------
        if "kld_total" in included:
            ax, cax = row_axes("kld_total")
            ax.plot(
                anchor_steps * seconds_per_step, kld_total_np,
                color=COLOR_VERMILLION, linewidth=0.9,
            )
            ax.set_title(
                "$K_t$ — total source-conditioned divergence per decoded anchor",
                fontsize=9, pad=6,
            )
            ax.set_xlabel("Anchor step (s)", fontsize=8)
            ax.set_ylabel("nats", fontsize=8)
            style_axes(ax, grid="both")
            finalise_time_axis(ax, tail=True)
            cax.set_visible(False)

        if lag_panels is not None:
            # Cut to the drawn anchors once, here, so the four rows below cannot disagree about
            # which anchors they are about.
            drawn_valid = lag_panels.lag_valid[kept]

            # ---- Row: what the head emitted, before the sum --------------------
            if "lag_proposal" in included:
                ax, cax = row_axes("lag_proposal")
                image = lag_row(
                    ax, lag_panels.proposal_norm[kept],
                    "$\\| r^{\\mu}_{t,\\ell} \\|_2$ — per-lag proposal magnitude, "
                    "before the sum, the scaling and the limiter",
                    "viridis",
                )
                attach_cbar(cax, image, "magnitude")

            # ---- Row: how much source each lag actually had --------------------
            if "lag_exposure" in included:
                ax, cax = row_axes("lag_exposure")
                image = lag_row(
                    ax, lag_panels.exposure[kept],
                    "Available source channels per anchor-lag pair — the denominator the two "
                    "rows around this one are read against",
                    "cividis",
                )
                attach_cbar(cax, image, "channels")

            # ---- Row: what removing a lag does ---------------------------------
            if "lag_suppression" in included:
                ax, cax = row_axes("lag_suppression")
                image = lag_row(
                    ax, lag_panels.suppression[kept],
                    "$K_t - K_t^{\\setminus \\ell}$ — fall in the divergence when that lag's "
                    "proposals alone are removed. A reliance reading, not an attribution; "
                    "grey carried no source at all.",
                    "magma", available=drawn_valid,
                )
                attach_cbar(cax, image, "nats")

            # ---- Row: the cancellation ratio, with both of its parts ------------
            if "cancellation" in included:
                ax, cax = row_axes("cancellation")
                seconds = anchor_steps * seconds_per_step
                ax.plot(
                    seconds, lag_panels.cancellation_ratio[kept],
                    color=COLOR_VERMILLION, linewidth=0.9, label="$\\kappa_t$ (ratio)",
                )
                twin = ax.twinx()
                for values, colour, style, label in (
                    (lag_panels.cancellation_numerator, COLOR_BLACK, "-",
                     "$\\| \\sum_\\ell r_{t,\\ell} \\|_2$"),
                    (lag_panels.cancellation_denominator, COLOR_GRAY, "--",
                     "$\\sum_\\ell \\| r_{t,\\ell} \\|_2$"),
                ):
                    twin.plot(
                        seconds, values[kept], color=colour, linewidth=0.7, linestyle=style,
                        alpha=0.8, label=label,
                    )
                twin.set_ylabel("magnitude", fontsize=8)
                twin.tick_params(labelsize=7)
                twin.grid(False)
                ax.set_title(
                    "$\\kappa_t$ — how much per-lag proposal mass survives the sum, with both "
                    "of its parts. The ratio alone cannot separate proposals that cancel from "
                    "proposals that are all near zero; the denominator can.",
                    fontsize=9, pad=6,
                )
                ax.set_xlabel("Anchor step (s)", fontsize=8)
                ax.set_ylabel("$\\kappa_t$", fontsize=8)
                ax.set_ylim(0.0, 1.05)
                style_axes(ax, grid="both")
                handles, labels = ax.get_legend_handles_labels()
                twin_handles, twin_labels = twin.get_legend_handles_labels()
                ax.legend(
                    handles + twin_handles, labels + twin_labels,
                    loc="upper right", fontsize=6, framealpha=0.95, ncol=3,
                )
                finalise_time_axis(ax, tail=True)
                cax.set_visible(False)

            # The qualification travels with the figure rather than only with the design note: a
            # caveat that lives in a planning document is one edit away from being dropped from
            # the thing a reader actually opens.
            fig.text(
                0.5, _QUALIFICATION_INCHES / figure_height,
                wrapped_caption(SUPPRESSION_QUALIFICATION),
                ha="center", va="bottom", fontsize=7, color=COLOR_GRAY,
            )

        readouts = "  ".join(
            f"{name}={float(scalars[name]):.4g}"
            for name in (
                "nll_base_block", "nll_full_block", "pred_gap",
                "source_conditioned_kl_raw", "prior_rate",
            )
            if name in scalars
        )
        # Labelled, because these are the epoch's own readouts over the whole validation set and
        # the rows below are one recording's. Unlabelled, a reader would take them for this
        # sample's and find that the gap row does not average to the gap printed above it.
        fig.suptitle(
            f"epoch {epoch} — sample {index} — guid {guid} — beta={beta:.4g}"
            + (f"\nvalidation epoch: {readouts}" if readouts else ""),
            fontsize=10, y=1.0 - 0.1 / figure_height, va="top",
        )
        return fig
    except BaseException:
        plt.close(fig)
        raise
