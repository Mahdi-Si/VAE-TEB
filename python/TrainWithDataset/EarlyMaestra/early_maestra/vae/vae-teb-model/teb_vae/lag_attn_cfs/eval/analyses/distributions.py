r"""What each metric's distribution over 20-minute segments looks like, cohort by cohort.

Every other analysis in this pipeline reduces to **one value per recording** before it reports
anything, and for good reason: consecutive anchors' forecast windows overlap in $H - 1$ of their $H$
steps and one GUID contributes many segments, so a per-segment $p$-value is anticonservative
by roughly that factor. What that reduction hides is the *shape* -- whether a cohort's higher mean
forecast error is a uniform shift, a heavier tail, or a handful of segments the model fails on
completely. Three distributions with the same mean are three different findings, and no figure
before this one could tell them apart.

So this analysis is deliberately the one place the per-segment population is described directly,
and it is **descriptive by construction**: no test, no confidence interval, no $p$-value, nothing
registered in the headline block. A visible separation here is a reason to look, not a result;
``cross_subgroup`` remains the only analysis that adjudicates a cohort difference, and it does so
on per-recording values.

**Both levels are drawn on the same axes, and that is the point.** The filled density is one value
per segment; the median / inter-quartile / range **strip** above it is one value per recording.
Their difference *is* the pseudo-replication -- if the strip is far narrower than the density
beneath it, most of the spread on display is within-recording variation rather than between-baby
variation, and the density is showing roughly thirty views of the same delivery. Drawing both makes
that visible where a sentence in a document would be read once and forgotten.

The two levels are drawn in two *different forms* deliberately, and :func:`draw_density_panel`
states why: a forty-bin density over the six recordings of a small cohort is a row of spikes that
estimates nothing, takes the panel's y-limit with it, and squashes the distribution the reader came
for.

**Four presentation choices, each load-bearing:**

* **Density, not counts.** The healthy cohort contributes an order of magnitude more segments than
  HIE. On a count axis every panel would show one tall healthy curve and two flat lines, which is
  a statement about the cohort sizes rather than about the metric. Each curve's $n$ travels in the
  legend at both levels instead.
* **One bin grid per panel**, computed from the pooled values across the cohorts drawn there. Two
  histograms on two different grids are not a comparison, and the difference between them can be
  the binning.
* **A faint fill under a hairline outline, with every fill beneath every outline.** The cohorts
  are drawn on top of one another, and the whole subject is where they differ -- so the encoding
  has to survive three or four of them overlapping. A solid fill hides whatever is behind it, and
  a heavy one blends with its neighbours into a colour no legend explains; what stays legible
  through a stack is a **line**. So the fill is dropped to a tint that only locates a cohort's
  mass, the shape is carried by an outline at full opacity and the lightest weight in the
  package's scale, and the two are drawn in separate passes so that no cohort's outline is washed
  out by a later cohort's fill. See :data:`FILL_ALPHA` and :data:`FILL_ZORDER`.
* **The error metrics are rooted per segment, and rooting is the only transform applied**, because
  a distribution of squared $z$-units is unreadable while a distribution of $z$-units is not. That
  rooting is legitimate here and would not be elsewhere: ``residual``'s rule is that *averaging*
  finished roots is Jensen-biased low, and the object drawn here is the distribution itself rather
  than its mean. The recording-level series still roots **after** the per-recording mean, exactly
  as the rest of the pipeline does -- so the two levels differ by that bias as well as by the
  aggregation, and the emitted record says so rather than leaving the two to be compared as though
  they were the same arithmetic.

**Nothing here is converted into a clinical unit, and there is no unit to convert into.** The
sibling pipeline's fourth presentation choice was rooting *and* a $z$-to-bpm conversion, because
its target is a heart rate and a clinician thinks in heartbeats. This one forecasts $C_{\mathrm{keep}}$
wavelet-modulus and phase-harmonic coefficients: there is no bpm for one of them, and inverting
the loader's per-channel statistics would put the channels on scales spanning orders of magnitude
-- which would destroy the pooled distribution this analysis is entirely about, since every panel
here draws one metric pooled over all $C_{\mathrm{keep}}$. So every axis is labelled ``normalised``,
:mod:`~teb_vae.lag_attn_cfs.eval.metrics` exports no conversion to reach for, and
``tests/test_eval_units.py`` asserts the absence rather than the repointing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from teb_vae.lag_attn_cfs.eval import cohort
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.frames import (
    describe,
    finite_column,
    per_recording_means,
    scored_sample_count,
)
from teb_vae.lag_attn_cfs.eval.metrics import NORMALISED_UNIT

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "distributions"

#: What it writes.
SUMMARY_FILENAME = "distribution_summary.csv"
PER_SEGMENT_FILENAME = "per_segment_metrics.csv"

#: The figures, named as ``FIGURE_GUIDE.md`` names them.
#:
#: **The ``<stem>_by_clinical_class.pdf`` / ``<stem>_by_subgroup.pdf`` shape is reserved** for the
#: runner's grouped-variant fan-out, and a figure named into it disappears: ``test_eval_smoke.py``
#: normalises those two suffixes out of the manifest as a *family*, so such a file is never
#: recorded, never documented, and reads to an operator as one of the violin figures it is not.
#: These are named the other way round for that reason.
CLASS_FIGURE = "class_histograms"
SUBGROUP_FIGURE = "subgroup_histograms"

#: Bins per panel. A module constant rather than an ``eval_config`` key, for the reason
#: ``TRAJECTORY_BIN_HOURS`` and the significance level are not keys either: an operator who could
#: widen the bins could merge two modes until a difference appeared or disappeared.
HISTOGRAM_BINS = 40

#: The two aggregation levels drawn in every panel, and written as a column of the summary table.
SEGMENT_LEVEL = "segment"
RECORDING_LEVEL = "recording"
LEVELS: Tuple[str, ...] = (SEGMENT_LEVEL, RECORDING_LEVEL)

#: The pooled row's axis name in the summary table -- the whole split, ignoring cohort.
POOLED_AXIS = "pooled"
POOLED_GROUP = "all"

#: Written into the record rather than only into this docstring, because it is the one way the two
#: levels are not the same arithmetic applied to two populations.
PER_SEGMENT_ROOT_NOTE = (
    "the segment-level series of a rooted metric is the root of each segment's own mean square, "
    "while the recording-level series roots after the per-recording mean as the rest of this "
    "pipeline does; by Jensen the first sits at or below the second, so the mean of the filled "
    "distribution is not this pipeline's headline RMSE and must not be quoted as one"
)

#: The standing on this analysis's output, carried in the record beside every number it writes.
DESCRIPTIVE_ONLY_NOTE = (
    "descriptive only: these are distributions, not tests. No p-value, interval or verdict is "
    "computed here and nothing from this analysis enters the headline block -- a visible "
    "separation between cohorts is a reason to look rather than a result, and cross_subgroup is "
    "the analysis that decides whether one survives being asked properly, on per-recording values"
)


@dataclass(frozen=True)
class Metric:
    """One distribution drawn, and how to get it from the per-sample table.

    Attributes:
        name: The name the figure, the summary table and the derived CSV all use.
        column: The source column on ``per_sample.csv``.
        root: Whether the source is an unrooted mean square that must be rooted before it is drawn.
            Rooting is the *only* transform this analysis applies, and the unit does not change
            under it: the root of a mean square of $z$-units is $z$-units.
        unit: The unit the metric is drawn in, and the one written into the summary table. Fixed
            per metric rather than resolved at run time, because nothing here depends on the
            loader's statistics being available.
        reference: A vertical line worth drawing -- a null the distribution should sit away from,
            or a calibrated value. ``None`` draws none.
        meaning: One line, carried into the summary table so a row reads without this module.
    """

    name: str
    column: str
    root: bool
    unit: str
    reference: Optional[float]
    meaning: str


#: The eight distributions drawn, two per question this pipeline asks. Deliberately a short
#: explicit list rather than every numeric column on the table: the per-sample frame carries about
#: thirty-five, and a figure with thirty-five rows is one nobody reads. The choice of *which* is an
#: editorial claim about which distributions matter, and it belongs in one visible table.
METRICS: Tuple[Metric, ...] = (
    # --- Forecast fidelity ---------------------------------------------------
    Metric(
        "rmse_full", "sq_error_full", True, NORMALISED_UNIT, None,
        "per-segment RMSE of the source-conditioned forecast over its scored target coefficients",
    ),
    Metric(
        "rmse_base", "sq_error_base", True, NORMALISED_UNIT, None,
        "the same for the target-only forecast; the pair says how much of a cohort's error the "
        "source pathway removed",
    ),
    Metric(
        "nll_full_block", "nll_full_block", False, "nats per anchor", None,
        "the source-conditioned block score: accuracy and confidence together, summed over the "
        "H*C_keep coefficients of one anchor's forecast block",
    ),
    # --- Coupling ------------------------------------------------------------
    Metric(
        "mc_pred_gap", "mc_pred_gap", False, "nats per anchor", 0.0,
        "the headline coupling readout per segment; zero is the null and the reference line "
        "marks it",
    ),
    Metric(
        "source_conditioned_kl_raw", "source_conditioned_kl_raw", False, "nats per anchor", None,
        "the unfloored KL between the two latents; readable as a rate only while the prior "
        "variance is off its clamp",
    ),
    # --- Latent and calibration ----------------------------------------------
    Metric(
        "delta_mu_rms", "delta_mu_rms", False, NORMALISED_UNIT, None,
        "per-element RMS of mu_post - mu_prior: how far the source moved the belief. Already "
        "rooted by the collection pass, so it is drawn as it is stored",
    ),
    Metric(
        "mean_logvar_full", "mean_logvar_full", False, "log z-units", 0.0,
        "the decoder's mean log-variance per segment; the reference marks sigma = 1 z-unit",
    ),
    Metric(
        "attention_entropy_nats", "attention_entropy_nats", False, "nats", None,
        "how spread the lag attention was; readable against the attainable ceiling rather than "
        "on its own",
    ),
)

#: The identity columns the derived per-segment CSV carries, so the figures are reproducible from
#: it alone. Read from the per-sample table where present -- an older run's tables may not carry
#: all of them, and a missing identity column is not a reason to fail.
IDENTITY_COLUMNS: Tuple[str, ...] = (
    "guid", "epoch", labels.CLASS_COLUMN, labels.SUBGROUP_COLUMN, "n_anchors",
)


# =============================================================================
# From the per-sample table to the two drawn frames
# =============================================================================
def build_frames(
    per_sample: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Derive the segment-level and recording-level frames in the units they are drawn in.

    The two are built from the same source columns by two deliberately different routes, and the
    difference is the subject of :data:`PER_SEGMENT_ROOT_NOTE`: a rooted metric roots **per
    segment** on the left and **after the per-recording mean** on the right, which is the form
    every other analysis reports.

    The sibling takes the loader's statistics here and converts the rooted metrics into bpm. This
    one takes no such argument, and the missing parameter is the point rather than an omission: a
    conversion that can be called will be, and there is no clinical unit for a wavelet modulus to
    be converted into. See the module docstring.

    Args:
        per_sample: The collected per-sample table.

    Returns:
        ``(segment, recording, units)``. Both frames carry the cohort columns and one column per
        entry of :data:`METRICS`, named for the metric rather than for its source. ``units`` maps
        each metric to its declared unit; it is a fixed mapping here rather than a resolved one,
        and is returned at all so the axis labels and the summary table read it from one place.
    """
    source_columns = [metric.column for metric in METRICS]
    per_guid = per_recording_means(per_sample, source_columns)

    segment = pd.DataFrame(index=per_sample.index)
    recording = pd.DataFrame(index=per_guid.index)
    units: Dict[str, str] = {}

    for metric in METRICS:
        segment_values = finite_column(per_sample, metric.column)
        recording_values = finite_column(per_guid, metric.column)
        if metric.root:
            # Clipped before the root rather than after: a mean of squares is non-negative, so a
            # negative value here is float noise, and ``sqrt`` of it would turn that noise into a
            # NaN indistinguishable from a segment that measured nothing. NaN passes the clip.
            segment_values = np.sqrt(np.clip(segment_values, 0.0, None))
            recording_values = np.sqrt(np.clip(recording_values, 0.0, None))
        segment[metric.name] = segment_values
        recording[metric.name] = recording_values
        units[metric.name] = metric.unit

    for name in IDENTITY_COLUMNS:
        if name in getattr(per_sample, "columns", []):
            segment[name] = per_sample[name].to_numpy()
    for name in labels.GROUP_COLUMNS:
        if name in getattr(per_guid, "columns", []):
            recording[name] = per_guid[name]

    return segment, recording, units


def cohorts_present(frame: pd.DataFrame, axis: str) -> List[str]:
    """The cohorts on one axis, in this pipeline's clinical order.

    Args:
        frame: Either drawn frame.
        axis: The cohort column.

    Returns:
        The labels present, worst-first; empty when the frame does not carry the axis.
    """
    if axis not in getattr(frame, "columns", []):
        return []
    return cohort.ordered_groups(labels.distinct_groups(list(frame[axis])), axis)


def subgroups_of_class(frame: pd.DataFrame, class_name: str) -> List[str]:
    """The subgroups appearing under one clinical class, in canonical order.

    Read from the data rather than from the stem's prefix: the class comes from the target tensor
    and the subgroup from the shard basename, so the mapping between them is a property of the
    split being evaluated rather than of a naming convention this module would then encode twice.

    Args:
        frame: The segment-level frame.
        class_name: The clinical class.

    Returns:
        The subgroups, canonically ordered; empty when either column is absent.
    """
    columns = getattr(frame, "columns", [])
    if labels.CLASS_COLUMN not in columns or labels.SUBGROUP_COLUMN not in columns:
        return []
    cut = frame[frame[labels.CLASS_COLUMN].astype(str) == str(class_name)]
    return cohort.ordered_groups(
        labels.distinct_groups(list(cut[labels.SUBGROUP_COLUMN])), labels.SUBGROUP_COLUMN
    )


def _series(frame: pd.DataFrame, axis: Optional[str], group: Optional[str], name: str) -> np.ndarray:
    """One cohort's finite values of one metric.

    Args:
        frame: Either drawn frame.
        axis: The cohort column, or ``None`` for the pooled series.
        group: The cohort, ignored when ``axis`` is ``None``.
        name: The metric column.

    Returns:
        The finite values as ``float64``, possibly empty.
    """
    cut = (
        frame[frame[axis].astype(str) == str(group)]
        if axis is not None and axis in getattr(frame, "columns", [])
        else frame
    )
    values = finite_column(cut, name)
    return values[np.isfinite(values)]


# =============================================================================
# The summary table
# =============================================================================
def build_summary_rows(
    segment: pd.DataFrame, recording: pd.DataFrame, units: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Describe every (axis, cohort, metric, level) cell the figures draw.

    Both levels for every cell, because the comparison between them is what says how much of a
    distribution's width is between recordings rather than within one -- and a reader who has only
    the CSV should be able to make that comparison without the figure.

    Args:
        segment: The segment-level frame.
        recording: The recording-level frame.
        units: Metric to the unit it ended up in.

    Returns:
        Long-form rows: the axis, the cohort, the metric, its unit and meaning, the level, and the
        count-mean-quartile description of that cell.
    """
    rows: List[Dict[str, Any]] = []
    axes: List[Tuple[str, List[Optional[str]]]] = [(POOLED_AXIS, [None])]
    for axis in labels.GROUP_COLUMNS:
        present = cohorts_present(segment, axis)
        if present:
            axes.append((axis, list(present)))

    for axis, groups in axes:
        for group in groups:
            for metric in METRICS:
                for level, frame in ((SEGMENT_LEVEL, segment), (RECORDING_LEVEL, recording)):
                    values = _series(
                        frame, None if axis == POOLED_AXIS else axis, group, metric.name
                    )
                    rows.append(
                        {
                            "group_column": axis,
                            "group": POOLED_GROUP if group is None else str(group),
                            "level": level,
                            "unit": units.get(metric.name, metric.unit),
                            "source_column": metric.column,
                            "meaning": metric.meaning,
                            **describe(values, name=metric.name),
                        }
                    )
    return rows


# =============================================================================
# The figures
# =============================================================================
#: Fraction of the panel height reserved above the densities for the per-recording strip.
RECORDING_STRIP_FRACTION = 0.28

#: Further headroom above the strip, left empty for the legend. Without it the legend box sits on
#: the strip band and hides the right-hand end of the widest cohort's range -- which is the one
#: value a reader is most likely to be looking for.
LEGEND_HEADROOM_FRACTION = 0.24

#: The y label, which is where the two levels are named. Not the title: the nested figure puts
#: three cells across a 13-inch page, and a title long enough to explain both levels overflows its
#: column and collides with its neighbours. A label is compact, vertical, and on every panel.
DENSITY_YLABEL = "density (bars); recordings (strip)"

#: How opaque a cohort's filled body is. Faint on purpose: up to four cohorts are drawn on top of
#: one another in a single panel, and a fill heavy enough to read on its own hides whatever is
#: behind it and blends with its neighbours into a colour the legend does not explain. At this
#: value the fill does one job -- locating where a cohort's mass is -- and the outline does the
#: rest, which is why it is safe to make it this light.
#:
#: The outline is drawn in the cohort's own colour rather than a darkened one, which was tried and
#: rejected: it evens out the contrast of the light end of a class's subgroup tints, but it also
#: turns the amber of ``acidosis`` brown, and the severity hue is the one thing on these figures
#: that is read without the legend.
FILL_ALPHA = 0.18

#: Every fill is drawn beneath every outline, rather than each cohort being drawn whole in turn.
#: That is what makes the overlaps readable at all: within one z-level matplotlib draws in call
#: order, so a cohort's outline would be veiled by the fill of every cohort drawn after it, and the
#: first entry in the legend would be the hardest curve to trace. That is an artefact of the draw
#: order reading as a property of the data. Two levels fix it -- the bodies below, the shapes above.
#:
#: Both sit around the Line2D default of $2.0$, so the reference line still draws over the fills
#: and the outlines still draw over it: a null line a cohort's outline crosses stays a null line.
FILL_ZORDER = 1.6
OUTLINE_ZORDER = 2.4


def draw_density_panel(
    ax: Any,
    segment_by_group: Dict[str, np.ndarray],
    recording_by_group: Dict[str, np.ndarray],
    groups: Sequence[str],
    *,
    title: str,
    xlabel: str,
    reference: Optional[float] = None,
) -> int:
    """Draw the segment-level density per cohort, with the recording-level spread above it.

    No shared primitive draws this: ``figures.histogram_panel`` is single-series, and a grouped
    variant of it would need the group ordering, the shared bin grid and the two levels -- all of
    which are this analysis's own. It is composed from the same ``new_figure`` / ``style_axes``
    surface every other hand-drawn panel in this package uses.

    **The two levels are drawn in two different forms, and that is deliberate.** The segment level
    is a density, because there are hundreds to thousands of segments per cohort and the shape is
    the subject. The recording level is a **median / inter-quartile / range strip**, because there
    are six to forty recordings -- and a forty-bin density over six values is a row of spikes that
    estimates nothing, sets the y-limit for the whole panel, and squashes the distribution the
    reader came for. The strip says the one thing the recording level is there to say: how wide the
    between-recording spread is against the between-segment one drawn beneath it.

    **The densities are drawn as outlines over tints, in two passes rather than one per cohort**,
    because the panel's subject is where cohorts *differ* and they are drawn on top of one another.
    A cohort's body is filled at :data:`FILL_ALPHA`, faint enough that a stack of them still reads
    as tints of the cohorts in the legend; its shape is carried by a hairline outline at full
    opacity; and every fill is drawn beneath every outline -- see :data:`FILL_ZORDER` -- so which
    cohort is legible is not decided by which was drawn last.

    Args:
        ax: Target axes.
        segment_by_group: Cohort to its finite segment-level values.
        recording_by_group: Cohort to its finite recording-level values.
        groups: The cohorts, in the order they should be drawn and legended.
        title: Panel title.
        xlabel: X-axis label, carrying the unit.
        reference: Optional vertical line -- a null, or a calibrated value.

    Returns:
        The number of cohorts that contributed at least one finite segment value. Zero draws the
        empty note instead, so a metric this checkpoint never produced reads as unmeasured rather
        than as a plotting failure.
    """
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(DENSITY_YLABEL)

    pooled = np.concatenate(
        [segment_by_group.get(group, np.zeros(0)) for group in groups] + [np.zeros(0)]
    )
    if pooled.size == 0:
        ax.text(
            0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes,
            ha="center", va="center", fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY,
        )
        figures.style_axes(ax)
        return 0

    # One grid for every cohort in this panel. Two histograms on two grids are not a comparison,
    # and the difference between them can be the binning rather than the data.
    edges = np.histogram_bin_edges(pooled, bins=HISTOGRAM_BINS)
    colours = figures.group_colors(groups)

    drawn: List[str] = []
    for group in groups:
        values = segment_by_group.get(group, np.zeros(0))
        if values.size == 0:
            continue
        colour = colours.get(group, figures.COLOR_BLUE)
        # Two passes over the one grid, for the reason ``FILL_ZORDER`` states: the bodies of every
        # cohort first, then their outlines above all of them.
        #
        # Density rather than counts: the cohorts differ in size by an order of magnitude, and on
        # a count axis every panel would report that rather than the metric.
        #
        # The translucency is on the *face colour* rather than on the artist, which is what lets
        # the outline stay opaque -- ``alpha`` would fade the border along with the fill and give
        # back the soft-edged blur this encoding exists to replace. This pass carries the legend
        # entry, so the swatch is the mark actually on the page rather than half of it.
        ax.hist(
            values, bins=edges, density=True, histtype="stepfilled",
            color=to_rgba(colour, FILL_ALPHA), edgecolor=colour,
            linewidth=figures.LINE_HAIRLINE, zorder=FILL_ZORDER,
            label=(
                f"{group} ({values.size} seg / "
                f"{recording_by_group.get(group, np.zeros(0)).size} rec)"
            ),
        )
        # The same staircase re-stroked above every fill, and deliberately unlabelled so the legend
        # keeps one row per cohort. Identical geometry, colour and weight to the border above, so
        # it restores that border where a later cohort covered it rather than adding a second mark.
        ax.hist(
            values, bins=edges, density=True, histtype="step",
            color=colour, linewidth=figures.LINE_HAIRLINE, zorder=OUTLINE_ZORDER,
        )
        drawn.append(group)

    _draw_recording_strip(ax, recording_by_group, drawn, colours)

    if reference is not None and np.isfinite(reference):
        ax.axvline(
            float(reference), color=figures.COLOR_GRAY, linestyle=":", linewidth=figures.LINE_REGULAR,
            label=f"reference {float(reference):g}",
        )
    ax.legend(fontsize=figures.FONT_SMALL, loc="upper right", framealpha=0.85)
    figures.style_axes(ax)
    return len(drawn)


def _draw_recording_strip(
    ax: Any,
    recording_by_group: Dict[str, np.ndarray],
    groups: Sequence[str],
    colours: Dict[str, str],
) -> None:
    """Lay each cohort's per-recording median, quartiles and range across the top of the panel.

    Placed above the densities rather than over them so neither obscures the other, and drawn in
    the cohort's own colour so the strip and the density beneath it need no second legend.

    Args:
        ax: The axes the densities have already been drawn into.
        recording_by_group: Cohort to its finite recording-level values.
        groups: The cohorts that were drawn, in order.
        colours: The cohort palette.
    """
    top = float(ax.get_ylim()[1])
    if not groups or top <= 0.0:
        return
    # Reserve headroom, then space the cohorts evenly inside it. Done by rescaling the axis rather
    # than by an axes-fraction transform, so the densities keep their own scale and the strip
    # cannot land on top of a tall bar.
    strip = top * RECORDING_STRIP_FRACTION
    ax.set_ylim(0.0, top + strip + top * LEGEND_HEADROOM_FRACTION)

    for index, group in enumerate(groups):
        values = recording_by_group.get(group, np.zeros(0))
        if values.size == 0:
            continue
        height = top + strip * (index + 1) / (len(groups) + 1)
        colour = colours.get(group, figures.COLOR_BLUE)
        ax.hlines(height, float(values.min()), float(values.max()),
                  color=colour, linewidth=figures.LINE_THIN, alpha=0.9)
        if values.size >= 2:
            ax.hlines(
                height, float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75)),
                color=colour, linewidth=figures.LINE_HEAVY, alpha=0.9,
            )
        ax.plot(
            [float(np.median(values))], [height], marker="o", markersize=3.0,
            color=figures.COLOR_BLACK, markerfacecolor=colour, markeredgewidth=0.5,
        )


def _panel_title(metric: Metric, prefix: str = "") -> str:
    """The title a panel carries: the metric, and on the nested figure the class it is cut to."""
    return f"{prefix}{metric.name}"


def build_class_figure(
    segment: pd.DataFrame, recording: pd.DataFrame, units: Dict[str, str]
) -> Any:
    """One panel per metric, the three clinical classes overlaid.

    Args:
        segment: The segment-level frame.
        recording: The recording-level frame.
        units: Metric to unit, for the axis labels.

    Returns:
        The figure; the caller renders and closes it.
    """
    axis = labels.CLASS_COLUMN
    groups = cohorts_present(segment, axis)
    figure, axes = figures.new_figure(len(METRICS), 1, height_per_row=3.0)
    for row, metric in enumerate(METRICS):
        draw_density_panel(
            axes[row, 0],
            {group: _series(segment, axis, group, metric.name) for group in groups},
            {group: _series(recording, axis, group, metric.name) for group in groups},
            groups,
            title=_panel_title(metric),
            xlabel=f"{metric.name} ({units.get(metric.name, metric.unit)})",
            reference=metric.reference,
        )
    return figure


def build_subgroup_figure(
    segment: pd.DataFrame, recording: pd.DataFrame, units: Dict[str, str]
) -> Any:
    """One panel per (metric, clinical class), that class's subgroups overlaid inside each.

    Nested rather than flat: eight densities on one axes is unreadable, and the subgroup axis is
    already a subdivision of the class axis. One column per class puts at most four curves in a
    cell, and they are the four tints of that class's own colour -- so the layout and the palette
    say the same thing.

    Args:
        segment: The segment-level frame.
        recording: The recording-level frame.
        units: Metric to unit, for the axis labels.

    Returns:
        The figure; the caller renders and closes it.
    """
    classes = cohorts_present(segment, labels.CLASS_COLUMN)
    axis = labels.SUBGROUP_COLUMN
    figure, axes = figures.new_figure(
        len(METRICS), max(len(classes), 1), height_per_row=3.0, width=13.0
    )
    for row, metric in enumerate(METRICS):
        for column, class_name in enumerate(classes):
            groups = subgroups_of_class(segment, class_name)
            draw_density_panel(
                axes[row, column],
                {group: _series(segment, axis, group, metric.name) for group in groups},
                {group: _series(recording, axis, group, metric.name) for group in groups},
                groups,
                title=_panel_title(metric, prefix=f"{class_name}: "),
                xlabel=f"{metric.name} ({units.get(metric.name, metric.unit)})",
                reference=metric.reference,
            )
        if not classes:
            draw_density_panel(
                axes[row, 0], {}, {}, [],
                title=_panel_title(metric),
                xlabel=f"{metric.name} ({units.get(metric.name, metric.unit)})",
                reference=metric.reference,
            )
    return figure


# =============================================================================
# The registry entry point
# =============================================================================
def run_distributions_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Describe each metric's per-segment distribution, cut by clinical class and by subgroup.

    Args:
        context: The analysis context, read for the per-sample table. The loader statistics the
            sibling reads here are deliberately not consulted: nothing is converted.
        eval_config: The validated block. Unused: every choice this analysis makes is a module
            constant, deliberately -- see :data:`HISTOGRAM_BINS`.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the metric table, the units resolved, how many cohorts each
        figure drew, and the two standing notes -- the descriptive-only statement and the
        per-segment rooting caveat -- so both travel in ``summary.json`` rather than only in the
        documentation.
    """
    collection = context.collection
    per_sample = collection.per_sample
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    segment, recording, units = build_frames(per_sample)
    segment.to_csv(directory / PER_SEGMENT_FILENAME, index=False)
    rows = build_summary_rows(segment, recording, units)
    pd.DataFrame(rows).to_csv(directory / SUMMARY_FILENAME, index=False)

    # ``render_figure`` closes the figure it saves, which is why nothing here does: a production
    # pass draws two figures per analysis and matplotlib holds every unclosed one alive.
    written = [
        str(figures.render_figure(builder(segment, recording, units), directory / filename).name)
        for filename, builder in (
            (CLASS_FIGURE, build_class_figure), (SUBGROUP_FIGURE, build_subgroup_figure),
        )
    ]

    # Segments carrying a subgroup but no clinical class fall outside the nested figure entirely,
    # because its columns are the classes. Counted rather than silently omitted: a non-zero value
    # means the two labelling routes disagree, which is a dataset question and not a drawing one.
    classified = (
        int(segment[labels.CLASS_COLUMN].notna().sum())
        if labels.CLASS_COLUMN in segment.columns else 0
    )
    subgrouped = (
        int(segment[labels.SUBGROUP_COLUMN].notna().sum())
        if labels.SUBGROUP_COLUMN in segment.columns else 0
    )

    # No ``grouped_frames`` declaration, and its absence is deliberate rather than an oversight.
    # The runner's fan-out draws violins documented as holding one value per *recording*; handing
    # it this per-segment frame would produce a per-segment violin that reads as a per-recording
    # one, which is the exact confusion this analysis exists to make visible.
    return {
        "n_samples": scored_sample_count(per_sample, "nll_full_block"),
        "composition": {
            "n_segments": int(len(segment)),
            "n_recordings": int(len(recording)),
            "n_segments_with_class": classified,
            "n_segments_with_subgroup": subgrouped,
        },
        "plan": {"capped": False, "bins": HISTOGRAM_BINS},
        "metrics": [
            {
                "metric": metric.name,
                "source_column": metric.column,
                "unit": units.get(metric.name, metric.unit),
                "rooted_per_segment": bool(metric.root),
                "meaning": metric.meaning,
            }
            for metric in METRICS
        ],
        "levels": list(LEVELS),
        "cohorts": {
            axis: cohorts_present(segment, axis) for axis in labels.GROUP_COLUMNS
        },
        "descriptive_only": DESCRIPTIVE_ONLY_NOTE,
        "per_segment_root_note": PER_SEGMENT_ROOT_NOTE,
        "files": [PER_SEGMENT_FILENAME, SUMMARY_FILENAME, *written],
    }
