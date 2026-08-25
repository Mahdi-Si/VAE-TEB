r"""The rows of the diagnostic page that depend on one-sided coefficients and tiled anchors.

The page is the raw-signal sibling's. :func:`~teb_vae.lag_attn_rws.sample_page.build_diagnostic_figure`
owns the GridSpec, the row heights, the cut at the trained anchors, the shared physical-time axis
and the caption, and draws the five rows below -- the latent state, the per-dimension KL, $K_t$, the
lag attention and the KL-by-lag map -- because each of those is a statement about the latent and the
attention and reads the same whatever is being forecast. What is here is what this model's own
geometry makes different, and there are two seams rather than the sibling's one.

**The forecast rows**, through ``forecast_rows``. The two-sided sibling's version cannot be reused:
it walks a dense $(T_{\mathrm{valid}}, H, C)$ block with
:func:`~teb_vae.lag_attn.figure_primitives.concat_single_forecasts`, indexing an *anchor* into the
first axis, and this model's forecast is $(A_{\max}, H, C)$ indexed by *position in the decoded
set*. Feeding one to the other draws a real forecast at the wrong time -- there is no shape error
anywhere in it. The tiling is therefore walked through ``anchor_index`` and ``anchor_valid``, which
the forward returns for exactly this reason: the figure and the objective read one anchor set or
they are two pictures of two models.

**Six rows beyond the sibling's two**, reserved through :data:`CAUSAL_EXTRA_ROWS` and drawn from
the same stitched tiling, because three of $98$ channels drawn as offset lanes is not a picture of
this forecast. A model that predicts a handful of easy low-frequency coefficients well and the
rest not at all is indistinguishable, in the lane row and in every scalar the run reports, from
one that is uniformly mediocre. The six are the truth, the two branches' means, their signed
skill difference, the full branch's $\sigma$, and the per-window score the whole model is read by:

* ``pred_truth`` / ``pred_base`` / ``pred_full`` -- $Y^{+}$, $\mu^p$ and $\mu^q$ over every kept
  channel, on **one** colour scale taken from the truth, so an over- or under-shooting branch
  reads as saturation rather than being silently rescaled into looking right;
* ``pred_skill`` -- $|Y^{+}-\mu^p| - |Y^{+}-\mu^q|$, which is ``pred_gap`` resolved per channel
  and per step: red where conditioning on the source helped, blue where it hurt;
* ``pred_sigma`` -- $\sigma^q$, beside the error it is supposed to predict;
* ``pred_gap`` -- each drawn window's own block score under the **objective's own** likelihood,
  base against full, with the two per-channel profiles (error and $2\sigma$ coverage) as insets.

The scores on that last row are not a second implementation of the loss: they go through
:func:`~teb_vae.lag_attn_rws.nets.losses.raw_sample_score` under
:func:`~teb_vae.lag_attn_rws.nets.raw_masks.forecast_mask`, the two functions the objective itself
reduces, so a window's height on the row is in the same nats as the ``nll_base_block`` and
``nll_full_block`` printed in the page's own title. This is the one place the module computes
rather than lays out, and it is deliberate: an error curve drawn from a re-derived formula is the
one diagnostic that can disagree with the run it is diagnosing.

**Every channel axis reads top-down**, through
:func:`~teb_vae.lag_attn_rws.sample_page.top_down_extent`: coefficient $0$ at the top, the
scattering block above the phase block, on the input rows, the error map and all five field rows
alike. A reader carries a channel index down the page and it is in the same place on every row.

**Every inset sits in the warm-up prefix**, through :func:`_prefix_boxes` -- the per-anchor error
map and the two per-channel profiles alike. The two-sided sibling puts its error map in the right
margin, and that is right for its tiling and wrong for this one: both pages leave one span of the
forecast row undrawn, but its tiling stops short of the recording's end and this one runs to it,
so the blank corner is the tail there and the prefix below the anchor floor here. Inheriting the
box unedited put the panel over the last windows of the very forecast it is a detail of.

**The input rows**, through ``input_stream_panels``. The shipped builder calls
``describe_streams``, which raises on this family's channel widths -- inside a handler that warns
and continues, so the cost of not replacing it is a green suite and a page missing two rows. It also
draws ``gate(values)``, and on this model the gate is a pure gather: the warm-up mask lives one
layer further on, inside the availability adapter, so the gate's output is *not* what the encoder
reads. What is drawn here is the adapter's own availability buffer applied to the gated stream, so
the zeros on the row and the staircase over them are one tensor rather than two that agree.

**What the staircase means here is not what it means on a two-sided page.** There it is a delay: the
channel is read $\delta_c$ steps late and its leading steps are the guard's zero fill. Here it is a
*warm-up*: the coefficient at $t < W'_c$ exists and is a perfectly ordinary float, but it is a
function of assumed pre-recording history rather than of the recording, and it was normalised with
constants accumulated while excluding exactly that region. The row says so in its own legend, which
is why :class:`~teb_vae.lag_attn_rws.sample_page.InputStreamPanel` carries the label as a field.

**The lag axis is stored-coefficient time, not physical delay.** One-sidedness and zero latency are
different properties and this family buys only the first: beyond its warm-up a causal channel still
lags by its composed group delay, up to $791$ s, and nothing compensates for it. The forecast claim
survives that untouched -- a coefficient at $t$ is a function of $\{x(s) : s \le t\}$, so predicting
$t + 1 + \tau$ from history up to $t$ is a genuine forecast whatever the internal latency -- but a
peak at lag $\ell$ is an attribution over stored coefficients, not a physiological delay. The page
carries that caveat as a footnote, because the two lag panels belong to the shared builder and the
four shipped models must not acquire a caption about a transform they do not use.

Like both sibling pages this module is matplotlib-only -- no Lightning, no MLflow, no config, no
loader -- and it never re-runs the model or re-scores anything: it cuts and lays out arrays it is
handed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
from matplotlib.ticker import MaxNLocator  # noqa: E402

from teb_vae.lag_attn.figure_primitives import (  # noqa: E402
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_LIGHT_GRAY,
    COLOR_ORANGE,
    COLOR_VERMILLION,
    select_forecast_channels,
    time_axes,
    to_numpy,
)
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP  # noqa: E402
from teb_vae.lag_attn_cfs.causal_warmup import SOURCE_BLOCKS, TARGET_BLOCKS  # noqa: E402
from teb_vae.lag_attn_fs.sample_page import (  # noqa: E402
    FORECAST_CHANNELS,
    _batch_field,
    _draw_context_row,
    _draw_error_map,
    _resolved_keep_index,
)
from teb_vae.lag_attn_rws.nets.losses import raw_sample_score  # noqa: E402
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask  # noqa: E402
from teb_vae.lag_attn_rws.sample_page import (  # noqa: E402
    BAND_SIGMAS,
    FORECAST_ROW,
    ForecastRowInputs,
    InputStreamPanel,
    top_down_extent,
)
from utils.style import style_axes  # noqa: E402

__all__ = [
    "CAUSAL_EXTRA_ROWS",
    "COMPACT_PAGE_ROWS",
    "LAG_TIME_CAVEAT",
    "WARMUP_STAIRCASE_LABEL",
    "causal_forecast_rows",
    "causal_stream_panels",
]

#: What the input rows' staircase is on a one-sided stream. Not a delay: nothing is shifted, and
#: the region behind it holds real values on no defined scale rather than a zero fill.
WARMUP_STAIRCASE_LABEL = "warm-up: first honest step, $W'_c$"

#: The caveat every lag-resolved panel on this page is read under. Stated once, as a page footnote,
#: because the two lag panels are the shared builder's and the four shipped models must not gain a
#: caption about a transform they do not use. Asserted as a string by the suite, so it cannot be
#: dropped by an edit that keeps the figure rendering.
LAG_TIME_CAVEAT = (
    "Lag axes are stored-coefficient time, not physical delay: a causal channel still lags by its "
    "own composed group delay (13-791 s), so the physical lag is $\\Delta\\ell$ plus a "
    "channel-dependent $\\tau^u_c - \\tau^y_{c'}$ plus the $-20$ s preprocessing shift."
)

#: Lane spacing as a multiple of the widest lane's own drawn extent, the two-sided page's rule and
#: for its reason: the bands are $2\sigma$ wide and $\sigma$ is learned, so a fixed offset either
#: overlaps early in training or flattens every lane late in it.
_LANE_HEADROOM = 1.15

#: Where the anchor rug sits inside the forecast axes, as a fraction of the drawn y-range. Low
#: enough to stay clear of the lanes, which start at offset $0$ and stack upwards.
_RUG_POSITION = 0.02

#: The rows this page draws **beyond** the two every page in the family reserves, as
#: ``(name, height_ratio)`` in drawing order, handed to
#: :func:`~teb_vae.lag_attn_rws.sample_page.build_diagnostic_figure` through its
#: ``forecast_extra_rows`` argument. Reserved and drawn from one constant on purpose: a name here
#: that nothing draws is a blank row, and a name drawn that is not here is a ``KeyError`` raised
#: inside a callback that swallows exceptions to protect the fit -- i.e. a silently missing page.
#: They sit directly below the forecast row, so everything the model *produced* stays contiguous
#: and the input rows still sit against the latent they feed.
CAUSAL_EXTRA_ROWS: Tuple[Tuple[str, float], ...] = (
    ("pred_truth", 1.25),
    ("pred_base", 1.25),
    ("pred_full", 1.25),
    ("pred_skill", 1.25),
    ("pred_sigma", 1.25),
    # Taller than the page's other line rows: it carries two curves, the shaded gap between them,
    # and the two per-channel profiles as insets.
    ("pred_gap", 1.5),
)

#: The rows the **reduced** page keeps, in the full page's own order, handed to
#: :func:`~teb_vae.lag_attn_rws.sample_page.build_diagnostic_figure` through its ``rows``
#: argument. Everything a reader needs to answer "what did this recording's latent and
#: attention do": the physiological context, the target block as the encoder receives it, the
#: latent state and its source-derived shift, $K_t$, and the lag attention. What it drops is the
#: forecast itself -- eight rows of what the model predicted, which is the other question.
#:
#: ``input_target`` is the name the layout derives from :func:`causal_stream_panels`'s first
#: panel, which is why this constant lives here rather than beside the layout: the shared
#: builder does not know what this cell's streams are called.
COMPACT_PAGE_ROWS: Tuple[str, ...] = (
    "raw",
    "input_target",
    "latent",
    "kld_total",
    "lag_attn",
)

#: Coverage a correctly calibrated $\mu \pm 2\sigma$ band attains under the Gaussian the
#: likelihood assumes, drawn as the reference line the per-channel coverage profile is read
#: against. Derived from :data:`~teb_vae.lag_attn_rws.sample_page.BAND_SIGMAS` rather than written
#: as $0.9545$, so a page that widened its bands cannot keep quoting the old target.
_NOMINAL_COVERAGE = float(
    torch.erf(torch.tensor(BAND_SIGMAS / np.sqrt(2.0), dtype=torch.float64))
)

#: Margin around and between the insets this page puts in a row's blank warm-up prefix, in axes
#: fractions.
_PREFIX_MARGIN = 0.03

#: Smallest fraction of a row the prefix insets are allowed to shrink to, for a geometry whose
#: warm-up prefix is too short to hold them. Below this they are unreadable, and a row is better
#: off overlapping a little of its own left edge than carrying illegible boxes.
_PREFIX_MIN_SPAN = 0.30

#: Vertical placement of the two profile insets inside the gap row, as ``(y0, height)`` in axes
#: fractions. Nearly the full height: that row's own content is eleven markers on one curve.
_PROFILE_VERTICAL = (0.10, 0.84)

#: Vertical placement of the per-anchor error map inside the forecast row. Stops well below the
#: top because the lane row's legend sits at its upper left -- in the same blank prefix, and it is
#: the legend that names which of the three curves in a lane is which.
_ERROR_MAP_VERTICAL = (0.06, 0.60)

#: Robust colour-limit percentiles for the field rows, the input rows' own rule and for its
#: reason: these are z-scored wavelet coefficients and one heavy-tailed channel otherwise sets the
#: scale for all of them.
_ROBUST_PERCENTILES = (1.0, 99.0)

#: Interpolation for the field rows, matching both sibling pages. ``'none'`` rather than
#: matplotlib's default: a resampled map invents values between two channels or two steps, and
#: per-channel resolution is the entire reason these rows exist.
_IMSHOW_INTERPOLATION = "none"


# =================================================================================================
# The input rows
# =================================================================================================
def _stream_blocks(
    names: Sequence[str], split: int, keep_index: np.ndarray, declared_width: int
) -> Tuple[Tuple[str, int, int], ...]:
    """Re-express a stream's two stored blocks as spans over its **surviving** channels.

    Under a budget the blocks lose different numbers of channels, so the boundary in the gathered
    stream is not where the declared widths put it -- which is the same reason
    ``StreamChannels.kept_block_spans`` exists on the two-sided side.

    Args:
        names: The stored block names, in concatenation order. A stream built without its first
            stored block carries the second name **alone**, which is the ``use_up_st: false``
            source, so the count is read rather than assumed.
        split: How many *declared* channels belong to the first block. ``0`` means the stream was
            built without it, so the remaining block is the whole stream.
        keep_index: Surviving channel indices into the declared width, ascending.
        declared_width: The stream's declared channel count.

    Returns:
        One ``(name, start, stop)`` per block that kept at least one channel, half-open and in
        surviving-channel coordinates.
    """
    # One name is one span over the whole declared width. Indexing ``names[1]`` unconditionally
    # made a legal stream an IndexError instead of a row -- and the page callback swallows it, so
    # the cost was both input rows behind a single warning rather than a failure.
    bounds = (
        ((names[0], 0, declared_width),)
        if len(names) == 1
        else ((names[0], 0, split), (names[1], split, declared_width))
    )
    spans: List[Tuple[str, int, int]] = []
    for name, start, stop in bounds:
        inside = int(np.count_nonzero((keep_index >= start) & (keep_index < stop)))
        if not inside:
            continue
        offset = int(np.count_nonzero(keep_index < start))
        spans.append((name, offset, offset + inside))
    return tuple(spans)


def _stream_title(
    name: str,
    blocks: Tuple[Tuple[str, int, int], ...],
    kept: int,
    declared: int,
    warmup: np.ndarray,
) -> str:
    """One line stating what the budget kept, what it dropped, and how long the survivors wait.

    Args:
        name: ``'target'`` or ``'source'``.
        blocks: The stream's surviving block spans, for the per-block counts.
        kept: Surviving channel count.
        declared: Declared channel count.
        warmup: $W'_c$ per survivor, in decimated steps.

    Returns:
        The row title.
    """
    per_block = ", ".join(f"{block} {stop - start}" for block, start, stop in blocks)
    # The **realised** budget, $\max_c W'_c$ over the survivors, not the configured threshold: the
    # two differ wherever the staircase has a gap, and it is the realised one the anchor floor has
    # to clear. The model carries only the survivors' vector, which is exactly this quantity.
    span = (
        f"budget $B$={int(warmup.max())} steps, warm-up {int(warmup.min())}-"
        f"{int(warmup.max())} steps "
        f"({warmup.min() * SECONDS_PER_STEP:g}-{warmup.max() * SECONDS_PER_STEP:g} s)"
        if warmup.size
        else "no warm-up"
    )
    return (
        f"Model input — {name} stream as the encoder receives it: {per_block}; "
        f"{kept}/{declared} channels kept, {declared - kept} dropped by the budget; {span}"
    )


def causal_stream_panels(
    model: Any,
    forward_inputs: Sequence[torch.Tensor],
    *,
    sample_index: int = 0,
) -> List[InputStreamPanel]:
    r"""Build the per-sample input rows from the tensors the net was actually fed.

    Replaces :func:`~teb_vae.lag_attn_rws.input_budget.stream_panels` for this family, which cannot
    be reused for two independent reasons. It calls ``describe_streams``, which builds the
    production two-sided Morlet bank and refuses a model whose declared width disagrees with it --
    and these widths do, because seven scattering channels per block were dropped at write time. And
    it draws ``gate(values)``: on this model the gate is a pure gather and the warm-up mask lives
    inside the availability adapter, so the gate's output is one layer short of the encoder's input.

    What is drawn is the gated stream multiplied by the **adapter's own** availability buffer -- the
    same tensor the adapter masks and announces with -- so the zeros on the row and the staircase
    over them cannot describe different regions.

    Args:
        model: The net, for its gates, its adapters and its resolved warm-up vectors.
        forward_inputs: The task's forward inputs, ``(y_st, y_ph, u_stream, ...)``. Only the first
            three are read; the tiling arguments beyond them carry no channels.
        sample_index: Which sample of the batch to draw.

    Returns:
        One panel per stream, target first.

    Raises:
        ValueError: If fewer than three tensors arrive, or if a stream's width disagrees with the
            model's declared one -- the block spans are positional into that width, so a mismatch
            would label one model's data with another's channels.
    """
    if len(forward_inputs) < 3:
        raise ValueError(
            f"expected at least the (y_st, y_ph, u_stream) inputs of the causal feature "
            f"representation, got {len(forward_inputs)} tensors"
        )
    y_st, y_ph, u_stream = forward_inputs[:3]
    streams = {"target": torch.cat([y_st, y_ph], dim=-1), "source": u_stream}

    panels: List[InputStreamPanel] = []
    for name, blocks, split, declared in (
        ("target", TARGET_BLOCKS, int(model.TARGET_BLOCK_SPLIT), int(model.c_y)),
        (
            "source",
            SOURCE_BLOCKS if model.use_up_st else SOURCE_BLOCKS[1:],
            int(model.SOURCE_BLOCK_SPLIT) if model.use_up_st else 0,
            int(model.c_u),
        ),
    ):
        values = streams[name]
        if values.dim() != 3 or int(values.shape[-1]) != declared:
            raise ValueError(
                f"the {name} stream is {tuple(values.shape)} but this model declares {declared} "
                f"channels. The block spans and the warm-up vector are positional into that "
                f"width, so drawing them against this tensor would mislabel every channel."
            )

        gate = getattr(model, f"{name}_gate", None)
        adapter = getattr(model, f"{name}_adapter")
        keep_index = (
            np.arange(declared)
            if gate is None
            else to_numpy(gate.keep_index).astype(int).ravel()
        )
        warmup = np.asarray(getattr(model, f"{name}_warmup_steps") or (), dtype=int)
        if not warmup.size:
            warmup = np.zeros(keep_index.size, dtype=int)
        # Plus the gate's own shift, which is the vector the adapter was built at. A gathered and
        # delayed channel is honest only once the step index has reached both, so a staircase drawn
        # from the warm-up alone would sit up to 97 steps left of the mask below it -- and the row
        # would show a zeroed region the drawn boundary said was real. Zero on every unaligned
        # model, where the gate is a pure gather.
        if gate is not None:
            warmup = warmup + to_numpy(gate.delay.delay_steps).astype(int).ravel()

        with torch.no_grad():
            gathered = values if gate is None else gate(values)
            # The adapter's buffer, taken through the adapter's **own** slice, not a mask rebuilt
            # from the warm-up vector: it is the tensor the encoder's input was multiplied by, and
            # a second construction of "the same" pattern is how a figure comes to draw a region
            # the model did not mask. Absent exactly when the adapter built no availability term,
            # which is the ungated model.
            availability = getattr(adapter, "availability", None)
            if availability is not None:
                available = adapter._slice(availability, int(gathered.shape[1]))
                gathered = gathered * available.to(gathered.dtype)

        spans = _stream_blocks(blocks, split, keep_index, declared)
        panels.append(
            InputStreamPanel(
                name=name,
                values=to_numpy(gathered[sample_index]),
                delays=warmup,
                # No centre frequencies: they are a property of the filter bank, and the one that
                # produced these coefficients is one-sided and is not the bank any figure module in
                # this tree carries. An empty vector draws no secondary axis, which is the honest
                # outcome; a fabricated one would label every channel wrongly.
                center_hz=np.empty(0, dtype=float),
                blocks=spans,
                title=_stream_title(name, spans, keep_index.size, declared, warmup),
                delay_label=WARMUP_STAIRCASE_LABEL,
            )
        )
    return panels


# =================================================================================================
# The forecast rows
# =================================================================================================
def _tiling_anchors(anchors: np.ndarray, valid: np.ndarray, horizon: int) -> List[int]:
    """Pick the decoded anchors whose forecast windows abut without overlapping.

    The page is drawn at the **validation** resolution, where every valid anchor is decoded at
    stride $1$ and consecutive windows overlap $(H-1)/H$; plotted at that stride the row would show
    each instant $H$ times over, from $H$ different latents. Walked out of the decoded set rather
    than recomputed from the geometry, so a row cannot draw a window the forward never produced.

    Args:
        anchors: The decoded anchor indices $(A_{\\max},)$.
        valid: Which of them are real $(A_{\\max},)$.
        horizon: $H$, the forecast length in decimated steps.

    Returns:
        Positions into the anchor axis, ascending, whose windows do not overlap.
    """
    chosen: List[int] = []
    next_free = -1
    for position, (anchor, is_valid) in enumerate(zip(anchors.tolist(), valid.tolist())):
        if is_valid and anchor >= next_free:
            chosen.append(position)
            next_free = anchor + horizon
    return chosen


def _tiled_branch(
    rows: ForecastRowInputs,
    branch: str,
    anchors: np.ndarray,
    positions: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Tile one branch's forecast onto the decimated grid, for every channel it emits.

    Args:
        rows: The row inputs.
        branch: ``'base'`` or ``'full'``.
        anchors: The decoded anchor indices of the sample being drawn.
        positions: Which positions of the anchor axis to draw, from :func:`_tiling_anchors`.

    Returns:
        ``(mean, sigma)``, each $(T, C_{\mathrm{keep}})$ on the decimated grid, ``NaN`` where no
        drawn window covers the step -- which renders as a gap rather than as a fabricated
        continuation.
    """
    index, geometry = rows.sample_index, rows.geometry
    mean = to_numpy(rows.outs[f"mu_{branch}"][index])
    sigma = np.exp(0.5 * to_numpy(rows.outs[f"logvar_{branch}"][index]))

    horizon, channels = int(geometry.horizon), int(mean.shape[-1])
    tiled = [np.full((geometry.t, channels), np.nan, dtype=np.float64) for _ in range(2)]
    for position in positions:
        start = int(anchors[position]) + 1
        stop = min(start + horizon, geometry.t)
        for target, source in zip(tiled, (mean, sigma)):
            target[start:stop] = source[position][: stop - start]
    return tiled[0], tiled[1]


def _draw_anchor_overlay(
    ax: Any,
    rows: ForecastRowInputs,
    anchors: np.ndarray,
    valid: np.ndarray,
    seconds_per_step: float,
    training_stride: int,
) -> None:
    r"""Mark the floor, the anchors this page decoded, and the tile grid training would use.

    Two anchor sets are drawn because two exist and they are not the same one. The page is produced
    at the validation resolution -- every valid anchor, at stride $1$ -- which is what makes it
    reproducible; training decodes $\mathcal A(\varphi) = \{F + \varphi + kS\}$, roughly a
    fifteenth as many, at a phase derived per segment per epoch. A page showing only the dense set
    would say nothing about the geometry the gradients were computed at, and one showing only a
    tile grid would be a picture of a phase this page did not draw.

    Args:
        ax: The forecast row's axes.
        rows: The row inputs, for the geometry and the layout hooks.
        anchors: The decoded anchor indices of the sample being drawn.
        valid: Which of them are real.
        seconds_per_step: $\Delta$ in seconds, for placing an anchor in physical time.
        training_stride: $S$, the stride a training step would tile at.
    """
    geometry = rows.geometry
    floor = int(geometry.warmup)
    decoded = anchors[valid.astype(bool)]

    low, high = ax.get_ylim()
    ax.axvline(
        floor * seconds_per_step, color=COLOR_BLUE, linewidth=1.0, linestyle="-",
        label=f"anchor floor $F$={floor}",
    )
    # A rug rather than one line per anchor: at the validation resolution there are 136 of them,
    # and 136 vertical lines is a shaded band that hides the forecast underneath it.
    ax.plot(
        decoded * seconds_per_step,
        np.full(decoded.size, low + _RUG_POSITION * (high - low)),
        marker="|", linestyle="none", markersize=4.0, color=COLOR_GREEN,
        label=f"decoded anchors ({decoded.size})",
    )
    # The training grid at phase 0, which is the one a reader can check against the geometry; the
    # phase a given segment gets in a given epoch is derived from its own identity and is not a
    # property of this figure.
    edges = range(floor, geometry.t_valid, training_stride)
    for position, anchor in enumerate(edges):
        ax.axvline(
            anchor * seconds_per_step, color=COLOR_GRAY, linewidth=0.5, linestyle=":",
            alpha=0.8, zorder=1,
            label=(
                f"training tiles, $S$={training_stride}, $\\varphi$=0"
                if position == 0
                else None
            ),
        )
    ax.set_ylim(low, high)


# =================================================================================================
# The stitched field rows and the per-window score
# =================================================================================================
@dataclass(frozen=True)
class _Stitched:
    r"""One sample's drawn tiling, in the form every row below the lane row reads it.

    Assembled once by :func:`causal_forecast_rows` and passed down, rather than re-derived per
    row: the tiling is a *choice of which decoded anchors to draw*, and six rows drawn from six
    independently made choices would be six pictures that only look aligned.

    Attributes:
        truth: $Y^{+}$ on the decimated grid $(T, C_{\mathrm{keep}})$, ``NaN`` outside the drawn
            windows.
        base_mean: $\mu^p$, same grid and same ``NaN`` support.
        base_sigma: $\sigma^p$, same.
        full_mean: $\mu^q$, same.
        full_sigma: $\sigma^q$, same.
        keep: Declared channel index of each decoder output lane, ascending.
        block_split: How many of the kept channels came from the first stored block; $0$ when the
            caller declared no split, which draws no divider.
        anchors: The decoded anchor indices of this sample $(A_{\max},)$.
        positions: Positions into that axis whose windows are drawn, ascending and
            non-overlapping.
    """

    truth: np.ndarray
    base_mean: np.ndarray
    base_sigma: np.ndarray
    full_mean: np.ndarray
    full_sigma: np.ndarray
    keep: np.ndarray
    block_split: int
    anchors: np.ndarray
    positions: Sequence[int]

    @property
    def block_spans(self) -> Tuple[Tuple[str, int, int], ...]:
        """The stored blocks as spans over the kept channels, for the field rows' dividers."""
        kept = int(self.keep.size)
        if not 0 < self.block_split < kept:
            return ((TARGET_BLOCKS[0], 0, kept),)
        return (
            (TARGET_BLOCKS[0], 0, self.block_split),
            (TARGET_BLOCKS[1], self.block_split, kept),
        )


def _prefix_boxes(
    rows: ForecastRowInputs, count: int, vertical: Tuple[float, float]
) -> Tuple[Tuple[float, float, float, float], ...]:
    r"""``count`` side-by-side inset boxes filling a row's blank **warm-up prefix**.

    Where an inset goes on this page, and it is not where it goes on the two-sided one. Both pages
    leave one span of the forecast row undrawn, but not the same span: the two-sided tiling starts
    at the first trained anchor and stops short of the recording's end, so its blank corner is the
    tail; this tiling starts at the anchor floor $F$ and runs to the end, so its blank corner is
    the *prefix*. The prefix is blank by construction rather than by luck -- no anchor exists below
    $F$, and a floor below either half of the pairing is refused at construction -- and at the
    shipped geometry it is $134$ of
    $300$ steps, comfortably the widest empty span on the row.

    Args:
        rows: The row inputs, for the geometry the prefix is read off.
        count: How many boxes to lay side by side.
        vertical: ``(y0, height)`` in axes fractions, shared by all of them.

    Returns:
        ``count`` boxes, left to right, each ``[x0, y0, width, height]`` in axes coordinates.
    """
    prefix = float(rows.geometry.warmup) / float(rows.geometry.t)
    span = max(prefix - 2.0 * _PREFIX_MARGIN, _PREFIX_MIN_SPAN)
    width = (span - (count - 1) * _PREFIX_MARGIN) / count
    y0, height = vertical
    return tuple(
        (_PREFIX_MARGIN + index * (width + _PREFIX_MARGIN), y0, width, height)
        for index in range(count)
    )


def _robust_limits(values: np.ndarray) -> Tuple[float, float]:
    """Colour limits from a field's finite values, at :data:`_ROBUST_PERCENTILES`.

    Args:
        values: Any array; non-finite entries are ignored.

    Returns:
        ``(vmin, vmax)``, always with ``vmax > vmin`` so ``imshow`` has a usable scale even for a
        constant or entirely non-finite field.
    """
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0, 1.0
    low, high = (float(np.percentile(finite, edge)) for edge in _ROBUST_PERCENTILES)
    return (low, high) if high > low else (low, low + 1.0)


def _field_row(
    rows: ForecastRowInputs,
    row_name: str,
    field: np.ndarray,
    stitched: _Stitched,
    *,
    title: str,
    cmap: str,
    limits: Tuple[float, float],
    cbar_label: str,
) -> None:
    r"""Draw one $(T, C_{\mathrm{keep}})$ field as a channel-by-time heatmap on its own row.

    The construction is :func:`~teb_vae.lag_attn_rws.sample_page._input_stream_row`'s -- top-down
    channel axis, white dashed block divider, one y-tick per block at its centre -- so a channel on
    a forecast row and the same channel on an input row are at the same height of the page and are
    labelled the same way.

    Args:
        rows: The row inputs and the layout hooks.
        row_name: Which reserved row to draw into; one of :data:`CAUSAL_EXTRA_ROWS`.
        field: The values $(T, C_{\mathrm{keep}})$, ``NaN`` where no drawn window covers the step.
        stitched: The drawn tiling, for the channel count and the block spans.
        title: The row title.
        cmap: Colormap name. Given ``NaN`` support explicitly, so an uncovered span reads as a
            gap rather than as whatever the colormap puts at its bad-value default.
        limits: ``(vmin, vmax)``.
        cbar_label: Label for the reserved colorbar.
    """
    ax, cax = rows.row_axes(row_name)
    n_channels = int(stitched.keep.size)
    image = ax.imshow(
        field.T,
        aspect="auto",
        cmap=matplotlib.colormaps[cmap].with_extremes(bad=COLOR_LIGHT_GRAY),
        origin="upper",
        vmin=limits[0],
        vmax=limits[1],
        extent=top_down_extent(0.0, rows.t_max, n_channels),
        interpolation=_IMSHOW_INTERPOLATION,
    )

    ticks, labels = [], []
    for index, (name, start, stop) in enumerate(stitched.block_spans):
        if index:
            ax.axhline(start - 0.5, color="white", linewidth=1.0, linestyle="--")
        ticks.append(0.5 * (start + stop) - 0.5)
        labels.append(name)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)

    ax.set_title(title, fontsize=9, pad=6)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Target channel", fontsize=8)
    rows.heatmap_spines(ax)
    rows.attach_cbar(cax, image, cbar_label)
    rows.finalise_time_axis(ax)


def _draw_field_rows(rows: ForecastRowInputs, stitched: _Stitched) -> None:
    r"""Draw the five field rows: the truth, both branches' means, the skill map and $\sigma^q$.

    Args:
        rows: The row inputs and the layout hooks.
        stitched: The drawn tiling.
    """
    windows = len(stitched.positions)
    kept = int(stitched.keep.size)
    # One scale for the three fields that are the same quantity, taken from the **truth**: scaled
    # each to its own range instead, a branch that predicts a flat line and one that tracks the
    # signal would render as equally structured pictures.
    shared = _robust_limits(stitched.truth)
    for row_name, field, quantity in (
        ("pred_truth", stitched.truth, "true $Y^{+}$"),
        ("pred_base", stitched.base_mean, "base $\\mu^p$ — target-only"),
        ("pred_full", stitched.full_mean, "full $\\mu^q$ — source-conditioned"),
    ):
        _field_row(
            rows, row_name, field, stitched,
            title=(
                f"{quantity} over all {kept} target channels — {windows} consecutive "
                f"non-overlapping windows, shared colour scale with the two rows beside it; grey: "
                f"no drawn window"
            ),
            cmap="viridis",
            limits=shared,
            cbar_label="normalised",
        )

    # Where the source earned its keep. The same subtraction ``pred_gap`` is, resolved on both
    # axes it is summed over -- and signed, because the interesting failure is the region where
    # conditioning on UP makes the forecast *worse*, which no scalar on the page can show.
    skill = np.abs(stitched.truth - stitched.base_mean) - np.abs(
        stitched.truth - stitched.full_mean
    )
    # Symmetric, so zero is the colormap's own centre -- but at the field rows' robust percentile
    # rather than at ``safe_vabs``'s maximum: the difference of two absolute errors is heavy-tailed
    # in exactly the way one outlying cell washes the whole map to white. ``_robust_limits``
    # returns a strictly positive upper edge for a non-negative field, which is what this is.
    _, vabs = _robust_limits(np.abs(skill))
    _field_row(
        rows, "pred_skill", skill, stitched,
        title=(
            "Source skill — $|Y^{+}-\\mu^p| - |Y^{+}-\\mu^q|$, the per-channel per-step "
            "decomposition of the gap: red where conditioning on the source helped, blue where "
            "it hurt"
        ),
        cmap="bwr",
        limits=(-vabs, vabs),
        cbar_label="normalised",
    )

    # Sigma on its own row rather than as a band, because at 98 channels a band is unreadable and
    # a variance collapse -- sigma pinned at its floor while the error above it is not -- is the
    # failure this family has actually had.
    _field_row(
        rows, "pred_sigma", stitched.full_sigma, stitched,
        title=(
            "Predicted $\\sigma^q$ of the source-conditioned forecast — read against the error "
            "two rows above: a flat map under a structured error is a collapsed variance head"
        ),
        cmap="magma",
        limits=(0.0, _robust_limits(stitched.full_sigma)[1]),
        cbar_label="normalised",
    )


def _window_block_scores(
    rows: ForecastRowInputs,
    stitched: _Stitched,
    *,
    likelihood: str,
    coverage_floor: float,
) -> Optional[Dict[str, np.ndarray]]:
    r"""Each drawn window's own block score, under the objective's likelihood and its mask.

    Reduced exactly as :func:`~teb_vae.lag_attn_rws.nets.losses.masked_raw_block_per_anchor`
    reduces it -- the elementwise term summed over the anchor's masked $H \cdot C_{\mathrm{keep}}$
    block -- so a value here is in the same nats as the ``nll_base_block`` and ``nll_full_block``
    the page's own title carries, and their difference is ``pred_gap`` restricted to this window.
    Both the term and the mask are the objective's own functions rather than copies: a curve drawn
    from a re-derived formula is the one diagnostic that can disagree with the run it diagnoses.

    Args:
        rows: The row inputs; ``rows.batch`` is the only route to the validity signal.
        stitched: The drawn tiling.
        likelihood: ``'mse'`` or ``'gaussian_nll'``, the objective's own.
        coverage_floor: The model's own floor, so an anchor this drops is dropped here too.

    Returns:
        ``{'base': (W,), 'full': (W,)}`` over the drawn windows, or ``None`` when the batch
        carries no ``weight`` -- the mask is a function of it, and scoring an invalid span would
        put a spike on the row that the objective never saw.
    """
    weight = _batch_field(rows.batch, "weight")
    if weight is None or not stitched.positions:
        return None

    index, geometry = rows.sample_index, rows.geometry
    horizon = int(geometry.horizon)
    with torch.no_grad():
        mask, _coverage = forecast_mask(
            weight,
            geometry,
            coverage_floor=float(coverage_floor),
            anchors=rows.outs["anchor_index"],
            anchor_valid=rows.outs["anchor_valid"],
        )
        stream = rows.target[index]
        keep = torch.as_tensor(stitched.keep, dtype=torch.long, device=stream.device)

        scores: Dict[str, np.ndarray] = {}
        for branch in ("base", "full"):
            mean = rows.outs[f"mu_{branch}"][index]
            logvar = rows.outs[f"logvar_{branch}"][index]
            window_scores = []
            for position in stitched.positions:
                anchor = int(stitched.anchors[position])
                # The anchor's own target block, gathered to the decoder's lanes: anchor $t$
                # predicts decimated steps $t+1 \dots t+H$, the forward's own convention.
                target = stream[anchor + 1 : anchor + 1 + horizon].index_select(1, keep)
                per_element = raw_sample_score(
                    mean[position], target, likelihood=likelihood, logvar=logvar[position]
                )
                window_scores.append(
                    float((per_element * mask[index, position][:, None]).sum())
                )
            scores[branch] = np.asarray(window_scores, dtype=float)
    return scores


def _channel_profile(
    stitched: _Stitched, branch: str
) -> Tuple[np.ndarray, np.ndarray]:
    r"""One branch's per-channel error and $2\sigma$ coverage over the drawn windows.

    Both are computed over the drawn support alone: outside it there is no forecast, and counting
    an uncovered step as a miss would read as a calibration failure of the model rather than as an
    absence of the figure.

    Args:
        stitched: The drawn tiling.
        branch: ``'base'`` or ``'full'``.

    Returns:
        ``(rmse, coverage)``, each $(C_{\mathrm{keep}},)$; ``NaN`` for a channel no drawn window
        covers.
    """
    mean = stitched.base_mean if branch == "base" else stitched.full_mean
    sigma = stitched.base_sigma if branch == "base" else stitched.full_sigma
    error = stitched.truth - mean
    drawn = np.isfinite(error)

    with np.errstate(invalid="ignore"):
        rmse = np.sqrt(_column_mean(np.where(drawn, error**2, np.nan)))
        inside = np.where(drawn, (np.abs(error) <= BAND_SIGMAS * sigma).astype(float), np.nan)
    return rmse, _column_mean(inside)


def _column_mean(values: np.ndarray) -> np.ndarray:
    """Mean down the time axis, ignoring ``NaN``, without warning on an all-``NaN`` channel.

    Args:
        values: A $(T, C)$ array.

    Returns:
        The $(C,)$ column means; ``NaN`` where a column is entirely ``NaN``.
    """
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    totals = np.where(finite, values, 0.0).sum(axis=0)
    return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)


def _profile_inset(
    ax: Any,
    box: Tuple[float, float, float, float],
    stitched: _Stitched,
    curves: Sequence[Tuple[np.ndarray, str, str, str]],
    *,
    xlabel: str,
    reference: Optional[float] = None,
) -> None:
    r"""Draw one per-channel profile as an inset, on the page's own top-down channel axis.

    Untitled, deliberately: every *titled* axes on this page spans the whole recording on the
    shared time axis, and this one's x-axis is a per-channel statistic. Labelled instead, which is
    the rule :func:`~teb_vae.lag_attn_fs.sample_page._draw_error_map` already established.

    Args:
        ax: The row the inset goes into.
        box: ``[x0, y0, width, height]`` in that row's axes coordinates.
        stitched: The drawn tiling, for the channel count and the block divider.
        curves: ``(values, colour, style, label)`` per curve, each $(C_{\mathrm{keep}},)$.
        xlabel: The statistic's label.
        reference: A vertical reference line, or ``None``.
    """
    inset = ax.inset_axes(list(box))
    channels = np.arange(int(stitched.keep.size))
    for values, colour, style, label in curves:
        inset.plot(values, channels, color=colour, linewidth=0.8, linestyle=style, label=label)
    if reference is not None:
        inset.axvline(reference, color=COLOR_BLACK, linewidth=0.7, linestyle=":")
    if 0 < stitched.block_split < channels.size:
        inset.axhline(stitched.block_split - 0.5, color=COLOR_GRAY, linewidth=0.7, linestyle="--")

    # The page's channel direction, set on the limits because this axes carries lines rather than
    # an image and so has no extent to take it from.
    inset.set_ylim(channels.size - 0.5, -0.5)
    inset.set_xlabel(xlabel, fontsize=6)
    inset.set_ylabel("target channel", fontsize=6)
    inset.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    inset.xaxis.set_major_locator(MaxNLocator(nbins=3))
    inset.tick_params(labelsize=5)
    inset.legend(loc="lower right", fontsize=5, framealpha=0.9)
    inset.grid(False)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color(COLOR_BLACK)
        spine.set_linewidth(0.6)


def _draw_gap_row(
    rows: ForecastRowInputs,
    stitched: _Stitched,
    *,
    likelihood: str,
    coverage_floor: float,
    seconds_per_step: float,
) -> None:
    r"""Draw ``pred_gap``: each drawn window's block score, base against full, plus the profiles.

    Args:
        rows: The row inputs and the layout hooks.
        stitched: The drawn tiling.
        likelihood: The objective's own likelihood.
        coverage_floor: The model's own anchor coverage floor.
        seconds_per_step: $\Delta$ in seconds, for placing a window in physical time.
    """
    ax, cax = rows.row_axes("pred_gap")
    cax.set_visible(False)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel(f"block score ({'nats' if likelihood == 'gaussian_nll' else 'sq. error'})",
                  fontsize=8)
    style_axes(ax, grid="both")

    scores = _window_block_scores(
        rows, stitched, likelihood=likelihood, coverage_floor=coverage_floor
    )
    if scores is None:
        # The row keeps its title, its axis and its place, so the rows below stay column-aligned
        # and the gap is visible -- the fallback the context row already uses.
        ax.set_title("Per-window forecast score", fontsize=9, pad=6)
        ax.text(
            0.5, 0.5, "no validity signal in this batch, so the objective's mask cannot be built",
            transform=ax.transAxes, ha="center", va="center", fontsize=8, color=COLOR_GRAY,
        )
        rows.finalise_time_axis(ax)
        return

    horizon = int(rows.geometry.horizon)
    # A window's mark sits at the centre of the span it scores, $[t+1, t+H]$, so it lands over
    # the same columns the field rows above drew it in.
    centres = np.asarray(
        [(int(stitched.anchors[p]) + 1 + 0.5 * horizon) * seconds_per_step
         for p in stitched.positions],
        dtype=float,
    )
    base, full = scores["base"], scores["full"]

    # The gap as the area between the two curves, signed by which one is lower. The shaded area is
    # `pred_gap` itself, window by window: nothing else on the page resolves it in time.
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
        f"Per-window forecast score under the objective's own likelihood "
        f"('{likelihood}', masked at the model's coverage floor {coverage_floor:g}) — "
        f"{base.size} windows, mean gap $D_0-D_1$ = {gap:.4g} over the drawn set.\n"
        f"Insets: per-channel error and $\\pm${BAND_SIGMAS:.0f}$\\sigma$ coverage over the same "
        f"windows, on the channel axis of the rows above.",
        fontsize=9, pad=6,
    )
    # Upper *right*: the two insets below take the left of the row, which is the span the anchor
    # floor leaves blank.
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


def causal_forecast_rows(
    rows: ForecastRowInputs,
    *,
    keep_index: Optional[Sequence[int]] = None,
    block_split: Optional[int] = None,
    training_stride: int = 1,
    likelihood: str = "gaussian_nll",
    coverage_floor: float = 0.0,
) -> None:
    r"""Draw the causal-feature page's forecast rows, over the anchors the forward decoded.

    Bound to a model's channel facts and its tiling by the task and handed to
    :func:`~teb_vae.lag_attn_rws.sample_page.build_diagnostic_figure` as its ``forecast_rows`` seam.
    Row $1$ is the shared raw-context row, drawn from the batch because this model's target is a
    feature block; row $2$ is the three-lane forecast, which is where the anchor axis makes this
    different from the two-sided sibling's version; the six rows below it are
    :data:`CAUSAL_EXTRA_ROWS`, which the same seam reserves and which draw the same tiling over
    every kept channel.

    On a page built for a subset of the rows -- see :data:`COMPACT_PAGE_ROWS` -- only the raw
    context row and the lag footnote are drawn, and the function returns before the forecast is
    stitched at all, so none of the per-window scoring below is paid for a page that has nowhere
    to put it.

    Args:
        rows: The row inputs and the layout hooks. ``rows.outs`` must carry ``anchor_index`` and
            ``anchor_valid``: the forecast tensors are indexed by *position in the decoded set*,
            not by anchor, so without them a window would be drawn at the wrong time with no shape
            error anywhere in it.
        keep_index: The budget's surviving target channels, positional into the declared $c_y$.
            ``None`` for an ungated model, whose decoder emits every declared channel in order.
        block_split: How many declared channels belong to the first stored block, for the error
            map's and the field rows' boundary line. ``None`` draws no boundary.
        training_stride: $S$, so the overlay can mark the tile grid a training step would use
            beside the dense set this page draws.
        likelihood: The objective's own likelihood, for the per-window score row. Bound by the
            task from the same hyperparameter the callback passes to ``compute_loss``, so the
            curve and the scalar beside it in the title are the same quantity.
        coverage_floor: The model's own anchor coverage floor, so a window the objective dropped
            is dropped from the score row too.

    Raises:
        KeyError: If the forward dict carries no anchor set.
        ValueError: If ``keep_index`` does not have one entry per forecast channel.
    """
    index, geometry = rows.sample_index, rows.geometry
    _draw_context_row(rows)

    # The footnote the lag panels are read under. Added here rather than to their titles because
    # those panels are the shared builder's and are drawn for six other models whose transform
    # this caveat is not about. Placed inside the GridSpec's bottom margin, so it costs no row --
    # and written before the early return below, because a page that keeps the lag rows and drops
    # the forecast rows is exactly the page that leads with a lag axis.
    rows.figure.text(
        0.5, 0.004, LAG_TIME_CAVEAT,
        ha="center", va="bottom", fontsize=7, color=COLOR_GRAY, wrap=True,
    )

    # Everything below belongs to the forecast row and the six rows reserved under it. A page
    # built without them has no axes for any of it, and ``row_axes`` raises rather than
    # inventing one -- inside a callback that swallows exceptions, which would cost the page.
    if not rows.wants(FORECAST_ROW):
        return

    anchors = to_numpy(rows.outs["anchor_index"][index]).astype(int).ravel()
    valid = to_numpy(rows.outs["anchor_valid"][index]).astype(bool).ravel()
    positions = _tiling_anchors(anchors, valid, int(geometry.horizon))

    base_mean, base_sigma = _tiled_branch(rows, "base", anchors, positions)
    full_mean, full_sigma = _tiled_branch(rows, "full", anchors, positions)
    keep = _resolved_keep_index(keep_index, full_mean.shape[-1])

    # The truth on the decimated grid, gathered to the channels the decoder emits and restricted to
    # the drawn windows, so an uncovered span reads as absent rather than as unpredicted.
    stream = to_numpy(rows.target[index])
    truth = np.where(np.isfinite(full_mean), stream[:, keep], np.nan)

    # The one resolved description of what this page draws, shared by the lane row, the error map
    # and the six rows below: a second walk of the anchor set, or a second count of the surviving
    # first-block channels, is how a page comes to draw two tilings that only look aligned.
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
    )

    lanes, coverage = select_forecast_channels(
        truth, full_mean, full_sigma, count=FORECAST_CHANNELS, n_sigmas=BAND_SIGMAS
    )

    ax, cax = rows.row_axes(FORECAST_ROW)
    _, time_dec, _ = time_axes(geometry.t, geometry.raw_len)
    seconds_per_step = rows.t_max / float(geometry.t)

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
        f"{int(valid.sum())} decoded anchors drawn, non-overlapping; shaded: the anchor the error "
        f"map draws.\nDrawn at the evaluation resolution — every valid anchor — so the ticks are "
        f"what this page decoded; the dotted grid is the sparser set a training step tiles.",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("target coefficient (normalised)", fontsize=8)
    style_axes(ax, grid="both")
    rows.finalise_time_axis(ax)
    _draw_anchor_overlay(ax, rows, anchors, valid, seconds_per_step, int(training_stride))
    ax.legend(loc="upper left", fontsize=6, framealpha=0.95, ncol=2)

    # $(C, H)$ rather than $(H, C)$: imshow's first axis is the vertical one, and the channel is
    # what a reader scans for a failure.
    anchor_truth = stream[anchor + 1 : anchor + 1 + geometry.horizon, keep]
    anchor_mean = to_numpy(rows.outs["mu_full"][index, position])
    _draw_error_map(
        ax,
        cax,
        np.abs(anchor_truth - anchor_mean).T,
        first_block_channels=stitched.block_split,
        anchor_seconds=anchor_start,
        # Not the two-sided page's right margin: this tiling runs to the end of the recording, so
        # an inset there covers the last windows of the very forecast it is a detail of. The blank
        # span here is the prefix below the anchor floor.
        box=_prefix_boxes(rows, 1, _ERROR_MAP_VERTICAL)[0],
    )

    _draw_field_rows(rows, stitched)
    _draw_gap_row(
        rows,
        stitched,
        likelihood=likelihood,
        coverage_floor=coverage_floor,
        seconds_per_step=seconds_per_step,
    )
