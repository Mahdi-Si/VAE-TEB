r"""The figures of one scoring pass and of one acceptance record, drawn from the artifacts alone.

**Every builder here takes the summary's results block, never a tensor.** A figure is a picture of
a number the results already carry -- an interval, a curve, a margin -- so a figure and the number
beside it cannot disagree, and the whole set can be redrawn from a finished directory on a box
with no checkpoint, no shard and no ``torch``. That is also what makes each builder testable
against a hand-written block. The run's figures are written by this cell's own analyses, each
into its own subdirectory; what is here is the drawing.

**What the figures are, and the rule behind each.**

* One axis per panel, always. A score and a count are never drawn on two y-axes of one frame; a
  quantity in a different unit gets its own panel, and panels that share an axis share it
  explicitly.
* Every interval that exists is drawn. A margin without its paired interval is a point that reads
  as a finding; the summary carries the interval, so the figure does.
* Identity is carried by text, colour by family. On a dot plot the y label names the arm and the
  colour says whether it is a matched branch, a lag band or a source control; on a curve figure
  the declared bands take fixed colours in declaration order and a legend names them.
* An empty panel says so. A block the run did not produce -- a target-only arm has no bands, a
  normalised fusion has no latent profile -- draws :data:`EMPTY_NOTE` rather than an empty frame.
* Every lag axis is stored-coefficient time, labelled as such, with the qualification the lag
  readouts must be read with drawn under the figure rather than left in the summary.

The style is the shared evaluation publication style -- open frames, a 7 pt serif scale,
unframed legends, panel letters, a double-column width -- applied once per process by
:func:`configure_figure_style`; importing this module restyles nothing. Titles are short noun
phrases; what a panel means is written in its axis labels and in the figure guide beside this
package, not in a sentence across its top.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from teb_vae.lag_attn.eval.figures import (  # noqa: E402
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_LIGHT_GRAY,
    COLOR_ORANGE,
    COLOR_PURPLE,
    COLOR_VERMILLION,
    EMPTY_NOTE,
    FIGURE_WIDTH,
    footnote,
    layout_rect,
    histogram_panel,
    legend_with_headroom,
    render_figure,
    style_axes,
)
from teb_vae.lag_attn.eval.figures import configure_figure_style as _configure_shared_style  # noqa: E402
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP  # noqa: E402
from teb_vae.lag_slot_transformer_cfs.eval.lag_metrics import SUPPRESSION_PREFIX  # noqa: E402
from teb_vae.lag_slot_transformer_cfs.nets.controls import SUPPRESSION_QUALIFICATION  # noqa: E402

__all__ = [
    "ACCEPTANCE_FIGURES",
    "BAND_COLORS",
    "build_acceptance_arms_figure",
    "build_acceptance_bands_figure",
    "build_acceptance_comparisons_figure",
    "build_block_figure",
    "build_calibration_figure",
    "build_gap_distribution_figure",
    "build_headline_figure",
    "build_horizon_figure",
    "build_lag_profile_figure",
    "configure_figure_style",
    "render_acceptance_figures",
]

#: Fixed colours for the declared lag bands, assigned in declaration order and never cycled: a
#: fifth band, should a configuration declare one, takes a shade from a sequential map rather than
#: reusing the first band's hue.
BAND_COLORS: Tuple[str, ...] = (COLOR_ORANGE, COLOR_GREEN, COLOR_PURPLE, COLOR_VERMILLION)

#: Colour by **family** on the dot plots: the two matched branches and the reference identities,
#: the lag bands, and the source controls. The y label carries the identity.
FAMILY_COLORS: Mapping[str, str] = {
    "branch": COLOR_BLUE,
    "reference": COLOR_GRAY,
    "band": COLOR_GREEN,
    "control": COLOR_ORANGE,
}

#: The two matched branches, drawn in a fixed pair of colours wherever both appear.
BRANCH_COLORS: Mapping[str, str] = {"base": COLOR_GRAY, "full": COLOR_BLUE}

#: Line weight of an interval bar and of a curve, and marker size of a point estimate. Thin marks,
#: so the data reads over the frame rather than under it.
_INTERVAL_WIDTH = 0.8
_CURVE_WIDTH = 0.7
_MARKER_SIZE = 2.8

#: Weight of a reference line -- a zero, a diagonal, a uniform value -- drawn under the data.
_REFERENCE_WIDTH = 0.5

#: Gap in points between the end of an interval bar and the value written beside it.
_VALUE_GAP_POINTS = 4.0

#: Height in inches of one row of a dot plot, and of the margin the axis labels and title take.
_DOT_ROW_HEIGHT = 0.28
_DOT_MARGIN_HEIGHT = 1.1

#: The prefix a band's column carries, from the readout module that names the arms.
_SUPPRESSION_PREFIX = SUPPRESSION_PREFIX

#: The order the arms of a run are listed in on the headline figure, by family; every arm the
#: summary carries and this order does not name is appended after them, so nothing is dropped.
_ARM_ORDER: Tuple[Tuple[str, str], ...] = (
    ("base", "branch"),
    ("full", "branch"),
    ("silence", "reference"),
    ("replace:zeros", "control"),
    ("replace:constant", "control"),
    ("permute", "control"),
)

#: The names an acceptance record's figures are written under.
ACCEPTANCE_FIGURES: Tuple[str, ...] = (
    "acceptance_comparisons",
    "acceptance_arms",
    "acceptance_bands",
)


def configure_figure_style(figure_format: Optional[str] = None) -> None:
    """Apply the repository's publication style once and fix the run's figure format.

    Args:
        figure_format: The format every figure of this run is written in, or ``None`` to leave the
            active format standing.
    """
    _configure_shared_style(figure_format)


# =============================================================================
# Primitives
# =============================================================================
def _note_empty(ax: Any, note: str = EMPTY_NOTE) -> None:
    """Mark an axes that had nothing to draw, so an empty panel is legible rather than puzzling.

    Args:
        ax: The axes.
        note: What to write.
    """
    ax.text(
        0.5, 0.5, note, transform=ax.transAxes, ha="center", va="center",
        fontsize=plt.rcParams["axes.labelsize"], fontstyle="italic", color=COLOR_GRAY,
    )
    style_axes(ax)


def _finite(value: Any) -> Optional[float]:
    """Return ``value`` as a float when it is a finite number, else ``None``.

    Args:
        value: Anything the summary might carry in a numeric slot.

    Returns:
        The float, or ``None``.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _curve_at(
    record: Optional[Mapping[str, Any]], index: int
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """One position of a curve record as ``(point, lo, hi)``, ``None`` where it is absent.

    Args:
        record: A curve bootstrap record, or ``None``.
        index: The position to read.

    Returns:
        The three values, each ``None`` when the record lacks that list or it is too short.
    """
    values = []
    for key in ("point", "lo", "hi"):
        series = (record or {}).get(key) or []
        values.append(_finite(series[index]) if index < len(series) else None)
    return values[0], values[1], values[2]


def _finish(fig: Any, *, footnote_text: Optional[str] = None) -> Any:
    """Lay the figure out, leaving room for a footnote when one was written.

    The footnote goes through the shared :func:`~teb_vae.lag_attn.eval.figures.footnote`, which
    wraps it to the figure width and reserves the room the wrapped block needs; the layout here is
    then done above that room, so a long qualification pushes the axes up rather than crossing
    their x label.

    Args:
        fig: The figure.
        footnote_text: The footnote, or ``None``.

    Returns:
        The same figure.
    """
    if footnote_text:
        footnote(fig, footnote_text)
    try:
        fig.tight_layout(rect=layout_rect(fig))
        fig.align_ylabels()
    except Exception:  # noqa: BLE001 - a layout warning must not lose a completed figure
        pass
    return fig


def _dot_intervals(
    ax: Any,
    rows: Sequence[Tuple[str, Optional[float], Optional[float], Optional[float], str]],
    *,
    xlabel: str,
    title: str = "",
    zero: bool = False,
    annotate: bool = True,
) -> int:
    """Draw one point estimate with its interval per row, rows labelled on the y axis.

    Args:
        ax: The axes.
        rows: ``(label, point, lo, hi, colour)`` per row, top to bottom. A row whose point is
            ``None`` is drawn as its label and a dash, so a missing measurement stays visible.
        xlabel: The x label.
        title: The panel title.
        zero: Whether to draw a reference line at zero.
        annotate: Whether to write each point's value beside its row.

    Returns:
        How many rows carried a point.
    """
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if not rows:
        _note_empty(ax)
        return 0
    positions = np.arange(len(rows), dtype=float)[::-1]
    drawn = 0
    for position, (label, point, lo, hi, colour) in zip(positions, rows):
        if point is None:
            # In the middle of the row rather than at its left edge, where it would sit on the
            # tick labels; italic so it reads as a note rather than as a category.
            ax.text(
                0.5, position, "not measured", transform=ax.get_yaxis_transform(),
                fontsize=plt.rcParams["legend.fontsize"], fontstyle="italic", color=COLOR_GRAY,
                ha="center", va="center",
            )
            continue
        right = point
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [position, position], color=colour, linewidth=_INTERVAL_WIDTH,
                    solid_capstyle="butt")
            right = max(hi, point)
        ax.plot(point, position, marker="o", color=colour, markersize=_MARKER_SIZE,
                markeredgecolor="white", markeredgewidth=0.3, linestyle="none", zorder=3)
        if annotate:
            # Beside the bar it belongs to, offset in points so it never lands on the frame or on
            # the neighbouring row whatever the axis range turns out to be.
            ax.annotate(
                f"{point:.3g}", xy=(right, position), xytext=(_VALUE_GAP_POINTS, 0.0),
                textcoords="offset points", fontsize=plt.rcParams["legend.fontsize"],
                color=COLOR_GRAY, ha="left", va="center", annotation_clip=False,
            )
        drawn += 1
    ax.set_yticks(positions)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    if zero:
        ax.axvline(0.0, color=COLOR_GRAY, linestyle=":", linewidth=_REFERENCE_WIDTH, zorder=0)
    style_axes(ax, grid="major")
    ax.grid(False, axis="y")
    ax.tick_params(axis="y", length=0)
    if annotate and drawn:
        # Room on the right for the values, which sit outside the data range by construction.
        ax.margins(x=0.12)
    return drawn


def _dot_figure_height(rows: int) -> float:
    """Height in inches of a dot-plot figure with ``rows`` rows."""
    return _DOT_ROW_HEIGHT * max(int(rows), 4) + _DOT_MARGIN_HEIGHT


def _ribbon(
    ax: Any,
    x: Any,
    record: Mapping[str, Any],
    *,
    colour: str,
    label: str,
) -> bool:
    """Draw one bootstrapped curve: the point as a line, the interval as a ribbon.

    Args:
        ax: The axes.
        x: The positions.
        record: A :func:`~teb_vae.lag_slot_transformer_cfs.eval.lag_metrics.bootstrap_curve`
            record, or anything with ``point``, ``lo`` and ``hi`` lists.
        colour: The curve's colour.
        label: The legend label.

    Returns:
        Whether anything finite was drawn.
    """
    point = np.asarray(record.get("point") or [], dtype=np.float64)
    if point.size == 0 or not np.isfinite(point).any():
        return False
    axis = np.asarray(list(x), dtype=np.float64)[: point.size]
    lo = np.asarray(record.get("lo") or [np.nan] * point.size, dtype=np.float64)
    hi = np.asarray(record.get("hi") or [np.nan] * point.size, dtype=np.float64)
    if np.isfinite(lo).any() and np.isfinite(hi).any():
        ax.fill_between(axis, lo, hi, color=colour, alpha=0.18, linewidth=0)
    ax.plot(axis, point, color=colour, linewidth=_CURVE_WIDTH, label=label)
    return True


def _band_colour(index: int, count: int) -> Any:
    """The colour of the band declared at ``index`` of ``count``.

    Fixed for the first few, then a sequential shade rather than a repeat of the first hue.

    Args:
        index: The band's position in declaration order.
        count: How many bands were declared.

    Returns:
        A colour matplotlib accepts.
    """
    if index < len(BAND_COLORS):
        return BAND_COLORS[index]
    extra = max(count - len(BAND_COLORS), 1)
    return plt.get_cmap("cividis")(0.2 + 0.6 * (index - len(BAND_COLORS)) / extra)


def _declared_bands(results: Mapping[str, Any]) -> List[str]:
    """The lag bands a summary declares, in declaration order and without the two identities.

    Args:
        results: The results block of the summary.

    Returns:
        The band names.
    """
    bands = (results.get("lag_readouts") or {}).get("band_suppression") or {}
    return [name for name in bands if name not in ("none", "all")]


def _lag_seconds_axis(ax: Any, step_seconds: float, delay_steps: int) -> None:
    """Attach a top axis in stored-coefficient seconds to a lag axis in stored steps.

    Args:
        ax: The axes whose x axis is the lag index.
        step_seconds: Seconds per stored step.
        delay_steps: The model's own input delay in stored steps, which the axis compensates for.
    """
    step, delay = float(step_seconds), float(delay_steps)
    if step <= 0.0:
        return
    try:
        top = ax.secondary_xaxis(
            "top", functions=(lambda lag: step * (lag + delay), lambda s: s / step - delay)
        )
        top.set_xlabel("lag (s, stored-coefficient time)", fontsize=plt.rcParams["axes.labelsize"])
        top.tick_params(labelsize=plt.rcParams["xtick.labelsize"])
    except Exception:  # noqa: BLE001 - an axis decoration must not lose the figure
        return


def _shade_bands(ax: Any, results: Mapping[str, Any], *, label: bool) -> None:
    """Shade the declared lag bands on a lag axis and name them along the top.

    The band edges come from the run's own override delta as the summary records them under the
    horizon block's band names and the lag profile's axis; when the edges are not carried, the
    bands are named without shading rather than at guessed edges.

    Args:
        ax: The axes whose x axis is the lag index.
        results: The results block of the summary.
        label: Whether to write the band names.
    """
    edges = (results.get("lag_readouts") or {}).get("band_edges") or {}
    names = _declared_bands(results)
    for index, name in enumerate(names):
        span = edges.get(name)
        if not span:
            continue
        lo, hi = float(span[0]) - 0.5, float(span[1]) + 0.5
        ax.axvspan(lo, hi, color=_band_colour(index, len(names)), alpha=0.10, linewidth=0,
                   zorder=0)
        if label:
            # Just inside the top of the panel, in the band's own colour, so the name is read
            # against its shading and not across a curve.
            ax.annotate(
                name, xy=(0.5 * (lo + hi), 1.0), xycoords=ax.get_xaxis_transform(),
                xytext=(0.0, -2.0), textcoords="offset points", ha="center", va="top",
                fontsize=plt.rcParams["legend.fontsize"], color=_band_colour(index, len(names)),
            )


# =============================================================================
# The run's figures
# =============================================================================
def build_headline_figure(results: Mapping[str, Any]) -> Any:
    """Every scored arm's predictive score, and every margin against the matched branch.

    Two panels. The left lists each arm's own equal-recording score with its interval, in a fixed
    order by family; the right lists each intervened arm's margin against the full branch with the
    **paired** interval of the per-recording differences, which is the interval a claim about a
    margin rests on. The gap itself is the first row of the right panel.

    Args:
        results: The results block of the summary.

    Returns:
        The figure.
    """
    headline = results.get("arm_scores") or {}
    readouts = results.get("lag_readouts") or {}
    bands = readouts.get("band_suppression") or {}
    controls = results.get("source_controls") or {}

    ordered: List[Tuple[str, str]] = list(_ARM_ORDER)
    for name in bands:
        ordered.insert(2, (f"{_SUPPRESSION_PREFIX}{name}", "band"))
    known = {arm for arm, _family in ordered}
    for column in sorted(headline):
        arm = column[len("nll_"):]
        if column.startswith("nll_") and arm not in known:
            ordered.append((arm, "control"))

    score_rows = []
    for arm, family in ordered:
        record = headline.get(f"nll_{arm}")
        if record is None:
            continue
        score_rows.append(
            (arm, _finite(record.get("point")), _finite(record.get("lo")),
             _finite(record.get("hi")), FAMILY_COLORS[family])
        )

    margin_rows = []
    gap = headline.get("pred_gap") or {}
    margin_rows.append(
        ("pred_gap (base $-$ full)", _finite(gap.get("point")), _finite(gap.get("lo")),
         _finite(gap.get("hi")), FAMILY_COLORS["branch"])
    )
    for name, block in bands.items():
        interval = block.get("margin_interval") or {}
        margin_rows.append(
            (f"{_SUPPRESSION_PREFIX}{name}", _finite(block.get("margin_nats")),
             _finite(interval.get("lo")), _finite(interval.get("hi")),
             FAMILY_COLORS["reference" if name in ("none", "all") else "band"])
        )
    for control in ("silence", "replace_zeros", "replace_constant", "permute"):
        interval = controls.get(f"{control}_margin_interval") or {}
        margin_rows.append(
            (control, _finite(controls.get(f"{control}_margin_nats")),
             _finite(interval.get("lo")), _finite(interval.get("hi")),
             FAMILY_COLORS["reference" if control == "silence" else "control"])
        )

    height = _dot_figure_height(max(len(score_rows), len(margin_rows)))
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, height))
    _dot_intervals(
        axes[0], score_rows, xlabel="predictive score (nats per anchor; lower is better)",
        title="Score per arm",
    )
    _dot_intervals(
        axes[1], margin_rows, xlabel="paired margin against the full branch (nats per anchor)",
        title="Margins, paired over recordings", zero=True,
    )
    return _finish(
        fig,
        footnote_text=(
            "A positive margin means the fitted model predicts worse under the intervention. The "
            "silence, suppress:none and suppress:all rows are identities that verify the "
            "intervention path; the gap is necessary and not sufficient, and is read beside an "
            "independently trained target-only reference."
        ),
    )


def build_gap_distribution_figure(
    results: Mapping[str, Any], per_recording: Mapping[str, Mapping[str, Any]]
) -> Any:
    """How the gap and the effective draw count are spread across recordings.

    An interval on the mean can exclude zero on a population where a third of the recordings sit
    on the other side, so the per-recording distribution is drawn beside it. The effective draw
    count says how many of the $K$ draws each recording's score rested on: a mass near one is a
    warning that $K$ was too small for those recordings.

    Args:
        results: The results block of the summary.
        per_recording: ``{recording: {column: value}}`` from the per-recording table.

    Returns:
        The figure.
    """
    gaps = [row.get("pred_gap") for row in per_recording.values()]
    concentration = [row.get("draw_concentration_full") for row in per_recording.values()]
    draws = (results.get("draws") or {}).get("num_mc_samples")

    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 2.6))
    histogram_panel(
        axes[0], [value for value in gaps if value not in (None, "")],
        title="Predictive gap per recording", xlabel="base $-$ full (nats per anchor)",
        color=COLOR_BLUE, reference=0.0, reference_label="no gap",
    )
    histogram_panel(
        axes[1], [value for value in concentration if value not in (None, "")],
        title="Effective draw count per recording (full branch)",
        xlabel="$1 / \\sum_k w_k^2$ over the $K$ draws",
        color=COLOR_PURPLE,
        reference=None if draws is None else float(draws),
        reference_label="" if draws is None else f"K = {int(draws)}",
    )
    return _finish(fig)


def build_band_figure(results: Mapping[str, Any]) -> Any:
    """Each declared band's suppression margin with its paired interval, and its exposure.

    Two panels on one band axis: the margin, and the usable anchor count behind it. The second is
    what says whether a small margin is a small effect or a band with little source to remove.

    Args:
        results: The results block of the summary.

    Returns:
        The figure.
    """
    bands = (results.get("lag_readouts") or {}).get("band_suppression") or {}
    names = [name for name in bands if name != "none"]
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, _dot_figure_height(len(names))))
    rows = []
    exposure_rows = []
    for index, name in enumerate(names):
        block = bands[name]
        interval = block.get("margin_interval") or {}
        colour = FAMILY_COLORS["reference"] if name == "all" else _band_colour(index, len(names))
        rows.append(
            (name, _finite(block.get("margin_nats")), _finite(interval.get("lo")),
             _finite(interval.get("hi")), colour)
        )
        exposure_rows.append((name, _finite(block.get("band_anchors")), None, None, colour))
    _dot_intervals(
        axes[0], rows, xlabel="paired suppression margin (nats per anchor)",
        title="Band suppression", zero=True,
    )
    _dot_intervals(
        axes[1], exposure_rows, xlabel="scored anchor-lag pairs with an available channel",
        title="Exposure behind each margin",
    )
    return _finish(fig, footnote_text=SUPPRESSION_QUALIFICATION)


def build_lag_profile_figure(results: Mapping[str, Any]) -> Any:
    """The lag axis at every candidate lag: exposure, the latent profile, the predictive margin.

    Four panels sharing the lag axis, top to bottom: the fraction of scored anchors at which the
    lag carried any available channel and the mean fraction of channels available at those
    anchors; the proposal norm and the shift removing the lag makes to the bounded mean update;
    the drop removing the lag makes to the divergence; and the predictive margin of removing the
    lag alone, with its paired interval, on the segments the profile cap admitted. The declared
    bands are shaded across every panel.

    Args:
        results: The results block of the summary.

    Returns:
        The figure.
    """
    readouts = results.get("lag_readouts") or {}
    exposure = readouts.get("exposure") or {}
    axis = readouts.get("lag_axis") or {}
    profile = readouts.get("lag_profile") or {}
    latent = profile.get("latent") or {}
    predictive = profile.get("predictive") or {}

    anchors = np.asarray(exposure.get("per_lag_anchors") or [], dtype=np.float64)
    channels = np.asarray(exposure.get("per_lag_channels") or [], dtype=np.float64)
    n_source = len(exposure.get("per_source_channel") or [])
    scored = float(max(anchors.max(), 1.0)) if anchors.size else 1.0
    lags = np.arange(anchors.size, dtype=np.float64)

    fig, axes = plt.subplots(4, 1, figsize=(FIGURE_WIDTH, 8.0), sharex=True)
    if anchors.size == 0:
        for ax in axes:
            _note_empty(ax, "no lag was read: this arm has no source pathway")
        return _finish(fig, footnote_text=SUPPRESSION_QUALIFICATION)

    # Exposure, as two fractions on one axis.
    ax = axes[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        live = anchors / scored
        mean_channels = np.where(anchors > 0, channels / np.maximum(anchors, 1.0), np.nan)
        channel_fraction = mean_channels / float(max(n_source, 1))
    ax.plot(lags, live, color=COLOR_GRAY, linewidth=_CURVE_WIDTH, label="live anchors")
    ax.plot(lags, channel_fraction, color=COLOR_BLUE, linewidth=_CURVE_WIDTH,
            label="available channels at live anchors")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("fraction")
    ax.set_title("Exposure per lag")
    legend_with_headroom(ax, ncol=2, below=True)
    style_axes(ax)
    _shade_bands(ax, results, label=True)

    # The latent profile: what the head emitted, and what removing it does to the update.
    ax = axes[1]
    drawn = False
    for name, colour, label in (
        ("proposal_norm", COLOR_GRAY, "proposal norm"),
        ("update_shift", COLOR_BLUE, "mean-update shift when removed"),
        ("scale_proposal_norm", COLOR_PURPLE, "scale proposal norm"),
    ):
        values = latent.get(name)
        if not values:
            continue
        curve = np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)
        if np.isfinite(curve).any():
            ax.plot(lags[: curve.size], curve, color=colour, linewidth=_CURVE_WIDTH, label=label)
            drawn = True
    if drawn:
        ax.set_ylabel("latent units")
        ax.set_title("Latent profile per lag")
        legend_with_headroom(ax, ncol=3)
        style_axes(ax)
        _shade_bands(ax, results, label=False)
    else:
        _note_empty(ax, "no latent profile: this arm sums no per-lag updates")

    ax = axes[2]
    values = latent.get("divergence_drop")
    if values:
        curve = np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)
        ax.plot(lags[: curve.size], curve, color=COLOR_VERMILLION, linewidth=_CURVE_WIDTH)
        ax.axhline(0.0, color=COLOR_GRAY, linestyle=":", linewidth=_REFERENCE_WIDTH)
        ax.set_ylabel("nats per anchor")
        ax.set_title("Divergence drop, lag removed alone (signed)")
        style_axes(ax)
        _shade_bands(ax, results, label=False)
    else:
        _note_empty(ax, "no latent profile: this arm sums no per-lag updates")

    ax = axes[3]
    margin = predictive.get("margin_nats") or {}
    if predictive.get("status") == "READ" and _ribbon(
        ax, lags, margin, colour=COLOR_GREEN,
        label=(
            f"single-lag suppression margin, paired over {margin.get('n', '?')} recordings "
            f"({predictive.get('n_segments', '?')} segments)"
        ),
    ):
        ax.axhline(0.0, color=COLOR_GRAY, linestyle=":", linewidth=_REFERENCE_WIDTH)
        ax.set_ylabel("nats per anchor")
        ax.set_title("Predictive margin, lag removed alone")
        legend_with_headroom(ax)
        style_axes(ax)
        _shade_bands(ax, results, label=False)
    else:
        _note_empty(ax, f"predictive lag profile {predictive.get('status', 'absent')}")
    ax.set_xlabel("lag (stored steps back from the anchor)")
    _lag_seconds_axis(
        axes[0], float(axis.get("seconds_per_step", SECONDS_PER_STEP)),
        int(axis.get("delay_steps", 0) or 0),
    )
    return _finish(fig, footnote_text=SUPPRESSION_QUALIFICATION)


def build_horizon_figure(results: Mapping[str, Any]) -> Any:
    """The gap and every margin resolved by horizon step, each with its interval.

    Three panels sharing the horizon axis: the gap; the band margins, one fixed colour per
    declared band; and the control margins. Every curve is a marginal mixture of that step's own
    likelihood factors, so the steps do not sum to the block score and the axis is read for its
    shape rather than its total.

    Args:
        results: The results block of the summary.

    Returns:
        The figure.
    """
    block = results.get("horizon_resolved") or {}
    positions = list(block.get("positions") or [])
    fig, axes = plt.subplots(3, 1, figsize=(FIGURE_WIDTH, 6.6), sharex=True)
    if not positions:
        for ax in axes:
            _note_empty(ax, "no horizon-resolved score was collected")
        return _finish(fig)

    unit = str(block.get("unit", "nats per anchor per horizon step"))
    ax = axes[0]
    if _ribbon(ax, positions, block.get("pred_gap") or {}, colour=COLOR_BLUE, label="pred_gap"):
        ax.axhline(0.0, color=COLOR_GRAY, linestyle=":", linewidth=_REFERENCE_WIDTH)
        ax.set_title("Predictive gap (base $-$ full) by horizon step")
        ax.set_ylabel(unit)
        legend_with_headroom(ax)
        style_axes(ax)
    else:
        _note_empty(ax)

    ax = axes[1]
    names = _declared_bands(results)
    band_margins = block.get("band_margins") or {}
    drawn = 0
    for index, name in enumerate(names):
        if _ribbon(ax, positions, band_margins.get(name) or {},
                   colour=_band_colour(index, len(names)), label=f"suppress:{name}"):
            drawn += 1
    if drawn:
        ax.axhline(0.0, color=COLOR_GRAY, linestyle=":", linewidth=_REFERENCE_WIDTH)
        ax.set_title("Band suppression margins by horizon step, paired over recordings")
        ax.set_ylabel(unit)
        legend_with_headroom(ax, ncol=min(max(drawn, 1), 4))
        style_axes(ax)
    else:
        _note_empty(ax, "no band was suppressed")

    ax = axes[2]
    control_margins = block.get("control_margins") or {}
    drawn = 0
    for name, colour in (
        ("replace_zeros", COLOR_ORANGE), ("replace_constant", COLOR_PURPLE),
        ("permute", COLOR_VERMILLION),
    ):
        if _ribbon(ax, positions, control_margins.get(name) or {}, colour=colour, label=name):
            drawn += 1
    if drawn:
        ax.axhline(0.0, color=COLOR_GRAY, linestyle=":", linewidth=_REFERENCE_WIDTH)
        ax.set_title("Source control margins by horizon step, paired over recordings")
        ax.set_ylabel(unit)
        legend_with_headroom(ax, ncol=3)
        style_axes(ax)
    else:
        _note_empty(ax, "no source control ran")
    ax.set_xlabel("horizon step")
    ax.set_xticks(positions)
    return _finish(
        fig,
        footnote_text=(
            "Each step is the marginal mixture of its own likelihood factors under the shared "
            "draws; the steps do not sum to the joint block score. Positive margins mean the "
            "fitted model predicts that step worse under the intervention."
        ),
    )


def build_block_figure(results: Mapping[str, Any]) -> Any:
    """The gap and the margins resolved by stored target block.

    One panel per block, each a dot plot of the gap and every margin with its paired interval.
    The channel count each block holds is written in its title, because the two blocks are not
    the same size and a nat summed over more channels is a larger number for that reason alone.

    Args:
        results: The results block of the summary.

    Returns:
        The figure.
    """
    block = results.get("block_resolved") or {}
    positions = list(block.get("positions") or [])
    counts = block.get("channels_per_block") or {}
    if not positions:
        fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH * 0.5, 2.2))
        _note_empty(ax, "no block-resolved score was collected")
        return _finish(fig)

    names = _declared_bands(results)
    fig, axes = plt.subplots(
        1, len(positions), figsize=(min(FIGURE_WIDTH, 3.6 * len(positions)), 2.8), sharex=True,
    )
    axes = np.atleast_1d(axes)
    for index, name in enumerate(positions):
        rows = [("pred_gap", *_curve_at(block.get("pred_gap"), index), FAMILY_COLORS["branch"])]
        for band_index, band in enumerate(names):
            record = (block.get("band_margins") or {}).get(band)
            rows.append((f"suppress:{band}", *_curve_at(record, index),
                         _band_colour(band_index, len(names))))
        for control in ("replace_zeros", "replace_constant", "permute"):
            record = (block.get("control_margins") or {}).get(control)
            rows.append((control, *_curve_at(record, index), FAMILY_COLORS["control"]))
        held = counts.get(name)
        _dot_intervals(
            axes[index], rows, xlabel="nats per anchor, summed over the block",
            title=f"Block {name}" + ("" if held is None else f" ({int(held)} channels)"),
            zero=True,
        )
    return _finish(fig)


def build_calibration_figure(results: Mapping[str, Any]) -> Any:
    """Mixture calibration of both branches: central coverage against nominal, and the transform.

    Left, the coverage of each central interval against its nominal level, with the diagonal a
    calibrated forecast sits on. Right, the mean and variance of the probability integral
    transform against the uniform reference the summary carries beside them.

    Args:
        results: The results block of the summary.

    Returns:
        The figure.
    """
    calibration = results.get("mixture_calibration") or {}
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 2.8))

    ax = axes[0]
    drawn = False
    for branch, colour in BRANCH_COLORS.items():
        block = calibration.get(branch) or {}
        coverage = block.get("coverage") or {}
        if not coverage:
            continue
        levels = sorted(float(level) for level in coverage)
        observed = [float(coverage[f"{level:g}"]) for level in levels]
        ax.plot(levels, observed, marker="o", color=colour, linewidth=_CURVE_WIDTH,
                markersize=_MARKER_SIZE, markeredgecolor="white", markeredgewidth=0.3,
                label=branch)
        drawn = True
    if drawn:
        ax.plot([0.0, 1.0], [0.0, 1.0], color=COLOR_GRAY, linewidth=_REFERENCE_WIDTH, linestyle=":",
                zorder=0, label="calibrated")
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(0.0, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("nominal central coverage")
        ax.set_ylabel("observed coverage")
        ax.set_title("Central coverage of the mixture")
        ax.legend(loc="upper left")
        style_axes(ax)
    else:
        _note_empty(ax)

    ax = axes[1]
    rows = []
    for branch, colour in BRANCH_COLORS.items():
        block = calibration.get(branch) or {}
        rows.append((f"{branch} PIT mean", _finite(block.get("pit_mean")), None, None, colour))
        rows.append((f"{branch} PIT variance", _finite(block.get("pit_var")), None, None, colour))
    reference_mean = _finite((calibration.get("full") or {}).get("uniform_pit_mean")) or 0.5
    reference_var = _finite((calibration.get("full") or {}).get("uniform_pit_var")) or 1.0 / 12.0
    _dot_intervals(ax, rows, xlabel="value", title="Probability integral transform")
    # The two uniform references go in a legend rather than as labels over the frame, where
    # they collided with the title.
    ax.axvline(reference_mean, color=COLOR_GRAY, linestyle=":", linewidth=_REFERENCE_WIDTH, zorder=0,
               label="uniform mean")
    ax.axvline(reference_var, color=COLOR_GRAY, linestyle="--", linewidth=_REFERENCE_WIDTH, zorder=0,
               label="uniform variance")
    ax.legend(loc="center")
    return _finish(fig)


# =============================================================================
# The acceptance record's figures
# =============================================================================
def _record_rows(
    entries: Mapping[str, Mapping[str, Any]], key: str, colour: str
) -> List[Tuple[str, Optional[float], Optional[float], Optional[float], str]]:
    """Dot-plot rows from ``{label: {key: bootstrap record}}``.

    Args:
        entries: The mapping.
        key: The interval's key inside each entry.
        colour: The rows' colour.

    Returns:
        The rows, in the mapping's order.
    """
    rows = []
    for label, entry in entries.items():
        record = (entry or {}).get(key) or {}
        rows.append((label, _finite(record.get("point")), _finite(record.get("lo")),
                     _finite(record.get("hi")), colour))
    return rows


def build_acceptance_comparisons_figure(record: Mapping[str, Any]) -> Any:
    """The declared primary comparisons, each a paired difference with its interval.

    Args:
        record: The acceptance record, or one of its ``selection`` / ``confirmation`` blocks.

    Returns:
        The figure.
    """
    block = record.get("selection", record)
    comparisons = block.get("primary_comparisons") or {}
    rows = []
    for name, entry in comparisons.items():
        interval = entry.get("difference_nats") or {}
        label = f"{name}\n{entry.get('left_arm')} - {entry.get('right_arm')}"
        colour = FAMILY_COLORS["branch"] if entry.get("status") == "READ" else COLOR_GRAY
        rows.append((label, _finite(interval.get("point")), _finite(interval.get("lo")),
                     _finite(interval.get("hi")), colour))
    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, 0.5 * max(len(rows), 3) + _DOT_MARGIN_HEIGHT))
    _dot_intervals(
        ax, rows, xlabel="left $-$ right, nll_full (nats per anchor)\nnegative favours the left",
        title="Primary comparisons, paired over recordings and averaged over seeds",
        zero=True,
    )
    return _finish(
        fig,
        footnote_text=(
            "A comparison is READ at the declared draw count when both arms carry the seed "
            "minimum; a grey row is one that was not read, and its interval, where drawn, is "
            "below the seed minimum or unmatched."
        ),
    )


def build_acceptance_arms_figure(record: Mapping[str, Any]) -> Any:
    """Every arm's internal gap, and its two differences against the frozen reference.

    Three panels sharing the arm axis. The internal gap; the full branch against the reference,
    which is the comparison the internal gap cannot make; and the base branch against the
    reference, which is the one that can fail.

    Args:
        record: The acceptance record, or one of its blocks.

    Returns:
        The figure.
    """
    block = record.get("selection", record)
    per_arm = block.get("per_arm") or {}
    arms = list(per_arm)
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH, _dot_figure_height(len(arms))), sharey=True)
    _dot_intervals(
        axes[0], _record_rows(per_arm, "internal_gap_nats", FAMILY_COLORS["branch"]),
        xlabel="base $-$ full (nats per anchor)", title="Internal gap", zero=True,
    )
    against = {arm: (entry.get("against_reference") or {}) for arm, entry in per_arm.items()}
    _dot_intervals(
        axes[1], _record_rows(against, "full_minus_reference_nats", FAMILY_COLORS["control"]),
        xlabel="full $-$ reference (nats per anchor)\nnegative favours the arm",
        title="Full branch against the frozen reference", zero=True,
    )
    _dot_intervals(
        axes[2], _record_rows(against, "base_minus_reference_nats", FAMILY_COLORS["reference"]),
        xlabel="base $-$ reference (nats per anchor)\npositive is a degraded base",
        title="Base branch against the frozen reference", zero=True,
    )
    return _finish(fig)


def build_acceptance_bands_figure(record: Mapping[str, Any]) -> Any:
    """Every arm's band margins at the nominal and the family-adjusted level.

    One panel per arm that searched the bands. The nominal interval is drawn in the band's colour
    and the family-adjusted one, which is the one a claim about the peak rests on, as a thinner
    grey bar behind it.

    Args:
        record: The acceptance record, or one of its blocks.

    Returns:
        The figure.
    """
    block = record.get("selection", record)
    bands_block = block.get("exploratory_bands") or {}
    searched = {arm: entry for arm, entry in bands_block.items() if entry.get("status") == "READ"}
    if not searched:
        fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH * 0.5, 2.2))
        _note_empty(ax, "no arm searched the lag bands")
        return _finish(fig, footnote_text=SUPPRESSION_QUALIFICATION)
    fig, axes = plt.subplots(
        1, len(searched), figsize=(min(FIGURE_WIDTH, 3.6 * len(searched)), 2.8),
        sharey=True, squeeze=False,
    )
    for ax, (arm, entry) in zip(axes[0], searched.items()):
        names = list(entry.get("searched_bands") or [])
        rows = []
        positions = np.arange(len(names), dtype=float)[::-1]
        for index, name in enumerate(names):
            band = (entry.get("bands") or {}).get(name) or {}
            adjusted = band.get("margin_nats_family_adjusted") or {}
            lo, hi = _finite(adjusted.get("lo")), _finite(adjusted.get("hi"))
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [positions[index], positions[index]], color=COLOR_LIGHT_GRAY,
                        linewidth=_INTERVAL_WIDTH * 3.0, solid_capstyle="butt", zorder=1)
            nominal = band.get("margin_nats") or {}
            rows.append((name, _finite(nominal.get("point")), _finite(nominal.get("lo")),
                         _finite(nominal.get("hi")), _band_colour(index, len(names))))
        peak = entry.get("peak_band")
        _dot_intervals(
            ax, rows, xlabel="paired suppression margin (nats per anchor)",
            title=f"{arm}" + ("" if peak is None else f" (peak band: {peak})"), zero=True,
        )
    return _finish(
        fig,
        footnote_text=(
            "Coloured bars are nominal intervals; the grey bar behind each is the family-adjusted "
            "interval covering every searched band at once, which is the one a claim about the "
            "peak rests on. " + SUPPRESSION_QUALIFICATION
        ),
    )


def render_acceptance_figures(record: Mapping[str, Any], directory: Any) -> Dict[str, str]:
    """Draw every figure of one acceptance record into a directory.

    Args:
        record: The acceptance record.
        directory: Where to write the figures.

    Returns:
        ``{name: path}`` of what was written.
    """
    target = Path(str(directory))
    target.mkdir(parents=True, exist_ok=True)
    builders = {
        "acceptance_comparisons": build_acceptance_comparisons_figure,
        "acceptance_arms": build_acceptance_arms_figure,
        "acceptance_bands": build_acceptance_bands_figure,
    }
    return {
        name: str(render_figure(builders[name](record), target / name, tight=False))
        for name in ACCEPTANCE_FIGURES
    }
