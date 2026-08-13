r"""The three rows of the diagnostic page that depend on one-sided coefficients and tiled anchors.

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

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")

from teb_vae.lag_attn.figure_primitives import (  # noqa: E402
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_GREEN,
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
    _draw_context_row,
    _draw_error_map,
    _resolved_keep_index,
)
from teb_vae.lag_attn_rws.sample_page import (  # noqa: E402
    BAND_SIGMAS,
    FORECAST_ROW,
    ForecastRowInputs,
    InputStreamPanel,
)
from utils.style import style_axes  # noqa: E402

__all__ = [
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
    # A rug rather than one line per anchor: at the validation resolution there are 152 of them,
    # and 152 vertical lines is a shaded band that hides the forecast underneath it.
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


def causal_forecast_rows(
    rows: ForecastRowInputs,
    *,
    keep_index: Optional[Sequence[int]] = None,
    block_split: Optional[int] = None,
    training_stride: int = 1,
) -> None:
    r"""Draw the causal-feature page's first two rows, over the anchors the forward decoded.

    Bound to a model's channel facts and its tiling by the task and handed to
    :func:`~teb_vae.lag_attn_rws.sample_page.build_diagnostic_figure` as its ``forecast_rows`` seam.
    Row $1$ is the shared raw-context row, drawn from the batch because this model's target is a
    feature block; row $2$ is the forecast, which is where the anchor axis makes this different
    from the two-sided sibling's version.

    Args:
        rows: The row inputs and the layout hooks. ``rows.outs`` must carry ``anchor_index`` and
            ``anchor_valid``: the forecast tensors are indexed by *position in the decoded set*,
            not by anchor, so without them a window would be drawn at the wrong time with no shape
            error anywhere in it.
        keep_index: The budget's surviving target channels, positional into the declared $c_y$.
            ``None`` for an ungated model, whose decoder emits every declared channel in order.
        block_split: How many declared channels belong to the first stored block, for the error
            map's boundary line. ``None`` draws no boundary.
        training_stride: $S$, so the overlay can mark the tile grid a training step would use
            beside the dense set this page draws.

    Raises:
        KeyError: If the forward dict carries no anchor set.
        ValueError: If ``keep_index`` does not have one entry per forecast channel.
    """
    index, geometry = rows.sample_index, rows.geometry
    _draw_context_row(rows)

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

    lanes, coverage = select_forecast_channels(
        truth, full_mean, full_sigma, count=FORECAST_CHANNELS, n_sigmas=BAND_SIGMAS
    )

    ax, cax = rows.row_axes(FORECAST_ROW)
    _, time_dec, _ = time_axes(geometry.t, geometry.raw_len)
    seconds_per_step = rows.t_max / float(geometry.t)

    def lane_extent(channel: int) -> float:
        """Total vertical span the widest artist of one lane needs."""
        half = BAND_SIGMAS * full_sigma[:, channel]
        stacked = np.concatenate(
            [full_mean[:, channel] - half, full_mean[:, channel] + half, truth[:, channel]]
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
        first_block_channels=(
            0 if block_split is None else int(np.count_nonzero(keep < int(block_split)))
        ),
        anchor_seconds=anchor_start,
    )

    # The footnote the lag panels are read under. Added here rather than to their titles because
    # those panels are the shared builder's and are drawn for six other models whose transform this
    # caveat is not about. Placed inside the GridSpec's bottom margin, so it costs no row.
    ax.figure.text(
        0.5, 0.004, LAG_TIME_CAVEAT,
        ha="center", va="bottom", fontsize=7, color=COLOR_GRAY, wrap=True,
    )
