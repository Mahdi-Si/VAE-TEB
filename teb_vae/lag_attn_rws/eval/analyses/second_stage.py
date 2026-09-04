r"""Does the coupling change around the onset of the second stage of labour?

The same two readouts the delivery clock resolves, resolved against a **different clinical
landmark**. Delivery is the end of a process; the onset of the second stage is the event inside it
that a clinician actually acts on, and two recordings four hours before delivery can be at
completely different points of labour. Nothing else in this pipeline aligns recordings on it.

**The axis is signed and is not negated.** The shard stores
$\texttt{second\_stage\_onset} = \texttt{domain\_start} - t_{\mathrm{SSO}}$, so
$h = \texttt{second\_stage\_onset}/3600$ is already negative before onset and positive after it --
the mirror of ``epoch``, which is stored as time *before* delivery and therefore is negated. The
arithmetic lives in :func:`~teb_vae.lag_attn_rws.eval.cohort.add_second_stage_bins` and is pinned
there by a known-answer test, because negating this one as well would run every trajectory
backwards through the second stage with nothing raising. For the same reason the figures are drawn
in the **natural** orientation with a line at zero, rather than inverted the way the delivery
clock's are, and the axis label names the sign convention outright.

**This analysis scores a different population, and says so.** A recording with no recorded onset
cannot be placed on this axis at all, so it is dropped and counted; the analysis therefore reports
fewer segments than every analysis beside it and declares itself ``capped``, which is the existing
mechanism for "excluded from the population comparison" rather than a disagreement about who was
evaluated.

**Two further ways a stored onset can be wrong are counted and filtered nowhere.** An onset falling
at delivery itself is what a pipeline writes when it substitutes zero for a missing time, and an
onset that moves across a recording's own segments can only come from a broken write. Both are
reported per recording in the eligibility table and in the record; neither excludes anything,
because excluding a recording changes the population every number is computed over and a count does
not.

**Three layers of inference, and the family is this clock's own.** Kruskal-Wallis across the classes
per window, Holm across *these* windows as one family, pairwise Mann-Whitney with Cliff's delta on
the survivors. The correction is **not** joint with the delivery clock's: the two are different
alignments of an overlapping population, so a window significant on one and not the other is a
statement about alignment, and the family-wise error rate each correction controls is within its own
clock. A reader combining a claim from both clocks is making two comparisons and is told so here.

.. note::

    lean-limit: the per-window cohort split below is written out here as well as in the
    delivery clock's analysis, because an analysis may not import another and the two differ in the
    window columns, the orientation and the eligibility rule; move it into ``cohort.py``,
    parameterised on the clock, when a third clock needs it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn_rws.eval import cohort
from teb_vae.lag_attn_rws.eval import figures_seam as figures
from teb_vae.lag_attn_rws.eval._reuse import labels, stats as shared_stats

#: This analysis's own subdirectory inside the results directory. Its own rather than a section of
#: the delivery clock's, because it scores a *different population* and has to record who was
#: dropped and why -- which is a directory rather than a footnote in one named for another clock.
ANALYSIS_DIRNAME = "second_stage"

#: What it writes.
TRAJECTORY_FILENAME = "second_stage_trajectory.csv"
PER_RECORDING_FILENAME = "second_stage_per_recording.csv"
SIGNIFICANCE_FILENAME = "second_stage_significance.csv"
PAIRWISE_FILENAME = "second_stage_pairwise.csv"
ELIGIBILITY_FILENAME = "second_stage_eligibility.csv"

#: The figure stems, named as ``FIGURE_GUIDE.md`` names them. Each is a **stem**, not a filename:
#: every readout gets both pages to itself, written as ``<stem>_<readout slug>``, exactly as
#: the delivery clock's are and for the reason
#: :class:`~teb_vae.lag_attn_rws.eval.cohort.ClockReadout`
#: gives. The trajectory is the shape of a readout against this clock; the windows page is what
#: that shape is made of.
TRAJECTORY_FIGURE = "second_stage_trajectory"
WINDOWS_FIGURE = "second_stage_windows"


def figure_stem(base: str, readout: cohort.ClockReadout) -> str:
    """Return the filename stem one readout's copy of a figure is written under.

    Written out here as well as on the delivery clock, because an analysis may not import another
    and the two stems differ; the *rule* is one line either way.

    Args:
        base: :data:`TRAJECTORY_FIGURE` or :data:`WINDOWS_FIGURE`.
        readout: The readout the figure resolves.

    Returns:
        ``'<base>_<slug>'``, without a suffix -- the renderer adds it.
    """
    return f"{base}_{readout.slug}"

#: Width of a window, in hours. The **same** grid the delivery clock uses, bound from the layer
#: below rather than restated, so a window on one clock's figure is the same duration as a window on
#: the other's. It is not an ``eval_config`` key, for the reason it is not one there.
TRAJECTORY_BIN_HOURS = cohort.TRAJECTORY_BIN_HOURS

#: Family-wise error rate the Holm correction across **this clock's** windows controls.
DEFAULT_ALPHA = 0.05

#: The two readouts, bound from the layer below rather than restated: both clocks resolve exactly
#: these, and two copies of the tuple would be two answers to what a clock figure shows.
READOUTS: Tuple[cohort.ClockReadout, ...] = cohort.CLOCK_READOUTS

#: The per-sample columns this analysis reduces, in the order the tables carry them.
VALUE_COLUMNS: Tuple[str, ...] = cohort.CLOCK_VALUE_COLUMNS

#: The x-axis label of both figures. It names the sign convention explicitly rather than leaving
#: "hours from onset" to be read either way round: a reader who takes a negative value for "after"
#: reads the whole trajectory backwards, and nothing on the page would contradict them.
AXIS_LABEL = "Hours from second-stage onset (negative = before onset, positive = after)"

#: The method sentence written into every record, so a $p$-value here is readable without this
#: module -- including the fact that its family stops at this clock.
METHOD = (
    "Per window of signed hours from second-stage onset: Kruskal-Wallis across clinical classes "
    "over one value per recording, Holm step-down correction across this clock's windows as one "
    "family, pairwise two-sided Mann-Whitney U with Cliff's delta for the windows significant "
    "after Holm only. Every pair is oriented from the more severe class to the less severe one, "
    "so a positive Cliff's delta means the more severe class's values run higher. "
    "Non-parametric "
    "throughout. Classes with fewer than "
    f"{shared_stats.MIN_GROUP_SIZE} recordings in a window are excluded from it and recorded. "
    "The Holm family is this clock's windows alone and is NOT corrected jointly with the "
    "time-before-delivery clock's: the two clocks are different alignments of an overlapping "
    "population, so a window significant on one and not the other is a statement about alignment, "
    "and the family-wise error rate each correction controls is within its own clock. The pooled "
    "row ignores the clock and is confounded by unequal class coverage of the axis."
)


# =============================================================================
# Who can be placed on this clock at all
# =============================================================================
def eligible_rows(per_sample: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the per-sample table into its eligibility record and the rows that record admits.

    Args:
        per_sample: The collected per-sample table.

    Returns:
        ``(eligibility, rows)`` -- the per-recording eligibility table from
        :func:`~teb_vae.lag_attn_rws.eval.cohort.second_stage_eligibility`, and the subset of
        ``per_sample`` belonging to the recordings it marks eligible. **One rule drops a
        recording: it has no onset.** The at-delivery and inconsistent-onset diagnostics travel on
        the table and exclude nothing.
    """
    eligibility = cohort.second_stage_eligibility(per_sample)
    if eligibility.empty or "guid" not in getattr(per_sample, "columns", []):
        return eligibility, per_sample.iloc[:0]
    admitted = sorted(set(eligibility.loc[eligibility["eligible"].astype(bool), "guid"].astype(str)))
    return eligibility, per_sample[per_sample["guid"].astype(str).isin(admitted)]


def eligibility_summary(eligibility: pd.DataFrame) -> Dict[str, Any]:
    """Count the population this analysis kept, the one it dropped, and the two diagnostics.

    Args:
        eligibility: The per-recording eligibility table.

    Returns:
        Counts over **recordings**, never segments: how many there were, how many carry an onset,
        how many were dropped for not carrying one, and how many of those kept are flagged by
        either diagnostic. The tolerance the second diagnostic is measured at travels with them,
        so the number is readable without this module.
    """
    if eligibility.empty:
        kept = dropped = at_delivery = inconsistent = 0
    else:
        admitted = eligibility["eligible"].astype(bool)
        kept = int(admitted.sum())
        dropped = int((~admitted).sum())
        at_delivery = int(eligibility.loc[admitted, "onset_at_delivery"].astype(bool).sum())
        inconsistent = int(eligibility.loc[admitted, "inconsistent_onset"].astype(bool).sum())
    return {
        "n_recordings": int(len(eligibility)),
        "n_eligible": kept,
        "n_dropped_no_onset": dropped,
        # Counted, never filtered -- see the module docstring.
        "n_onset_at_delivery": at_delivery,
        "n_inconsistent_onset": inconsistent,
        "onset_consistency_tolerance_s": float(cohort.ONSET_CONSISTENCY_TOLERANCE_S),
    }


# =============================================================================
# Binning and the trajectory tables
# =============================================================================
def build_per_recording(
    frame: pd.DataFrame, *, width: float = TRAJECTORY_BIN_HOURS
) -> Dict[str, pd.DataFrame]:
    """Reduce the eligible rows to one row per (cohort, window, recording), per axis.

    Args:
        frame: The eligible subset of the per-sample table.
        width: Window width in hours.

    Returns:
        Cohort axis to its per-recording-per-window frame, cut on **this clock's** window columns.
        Both axes are built because a reader wants both cuts; only the class axis is tested.
    """
    binned = cohort.add_second_stage_bins(frame, width=width)
    return {
        axis: cohort.per_recording_in_bins(
            binned,
            VALUE_COLUMNS,
            group_column=axis,
            bin_column=cohort.SECOND_STAGE_BIN_COLUMN,
            center_column=cohort.SECOND_STAGE_BIN_CENTER_COLUMN,
        )
        for axis in labels.GROUP_COLUMNS
    }


def build_trajectory_rows(per_recording: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    """Summarise every (axis, cohort, window) cell of both readouts, over its recordings.

    Args:
        per_recording: The per-axis frames from :func:`build_per_recording`.

    Returns:
        Long-form rows carrying the axis, the cohort, the window, the recording count and the
        quartiles. The window keys stay ``time_bin`` and ``bin_center_h`` on this clock too: they
        name a window and its centre, which is what they are on either, and renaming them per clock
        would fork every consumer of the table.
    """
    rows: List[Dict[str, Any]] = []
    for axis, frame in per_recording.items():
        for readout in READOUTS:
            for row in cohort.trajectory_rows(
                frame,
                readout.column,
                metric=readout.name,
                bin_column=cohort.SECOND_STAGE_BIN_COLUMN,
                center_column=cohort.SECOND_STAGE_BIN_CENTER_COLUMN,
            ):
                rows.append(
                    {"group_column": axis, "source_column": readout.column, **row}
                )
    return rows


# =============================================================================
# Significance across the classes, window by window
# =============================================================================
def _class_values(frame: pd.DataFrame, column: str) -> Dict[str, np.ndarray]:
    """Split one window's rows into per-class finite vectors, dropping none of them.

    Args:
        frame: One window's per-recording rows.
        column: The value column.

    Returns:
        Every class present, in the canonical HIE / acidosis / healthy order, mapped to its finite
        values. **Nothing is filtered here**: a class too small to test is still a class a figure
        must show, because a cohort thinning out toward the edge of the axis is the explanation for
        the window beside it -- and on this clock the positive side is short by construction, so
        that thinning is the common case rather than the exception.
    """
    values_by_class: Dict[str, np.ndarray] = {}
    for group in cohort.ordered_groups(
        sorted(set(frame["group"].astype(str))), labels.CLASS_COLUMN
    ):
        values = np.asarray(
            frame.loc[frame["group"].astype(str) == group, column], dtype=np.float64
        )
        values_by_class[group] = values[np.isfinite(values)]
    return values_by_class


def _class_samples(
    frame: pd.DataFrame, column: str
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Split one window's rows into per-class finite vectors, dropping classes too small to test.

    Args:
        frame: One window's per-recording rows.
        column: The value column.

    Returns:
        ``(usable, excluded)`` -- the classes with at least ``MIN_GROUP_SIZE`` recordings, and the
        sizes of those without, both in the canonical order. The exclusion is returned rather than
        dropped because "this class had two recordings in this window" is the explanation for a
        window the test skipped, and a reader who cannot see it will assume the comparison was
        made.
    """
    values_by_class = _class_values(frame, column)
    return (
        {
            group: values for group, values in values_by_class.items()
            if values.size >= shared_stats.MIN_GROUP_SIZE
        },
        {
            group: int(values.size) for group, values in values_by_class.items()
            if values.size < shared_stats.MIN_GROUP_SIZE
        },
    )


def window_samples(
    per_recording: pd.DataFrame, column: str
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, Dict[str, Any]]]:
    """Split a per-recording frame into the cells each window is tested and drawn from.

    One function rather than two because the figure and the test must describe the *same* cells: a
    violin drawn from one set of values under a $p$-value computed from another is a page that
    disagrees with itself and looks entirely ordinary.

    Args:
        per_recording: The class-axis per-recording-per-window frame.
        column: The value column.

    Returns:
        ``(samples, meta)`` keyed by window, in ascending window order -- which on this clock runs
        from before the onset to after it, because the bin index is signed rather than clipped.
        ``samples`` holds **every** class present in that window, unfiltered;
        :func:`testable_windows` is what applies the floor, because the figure and the test want
        different halves of the same split. ``meta`` holds what each window publishes beside its
        test -- the centre, the classes excluded as too small, and the floor they were excluded at.
    """
    samples: Dict[int, Dict[str, np.ndarray]] = {}
    meta: Dict[int, Dict[str, Any]] = {}
    columns = getattr(per_recording, "columns", [])
    if per_recording.empty or cohort.SECOND_STAGE_BIN_COLUMN not in columns:
        return samples, meta
    for bin_index in sorted(
        int(value) for value in per_recording[cohort.SECOND_STAGE_BIN_COLUMN].unique()
    ):
        cell = per_recording[per_recording[cohort.SECOND_STAGE_BIN_COLUMN] == bin_index]
        values_by_class = _class_values(cell, column)
        samples[int(bin_index)] = values_by_class
        meta[int(bin_index)] = {
            "bin_center_h": float(cell[cohort.SECOND_STAGE_BIN_CENTER_COLUMN].iloc[0]),
            "groups_excluded_as_too_small": {
                group: int(values.size) for group, values in values_by_class.items()
                if values.size < shared_stats.MIN_GROUP_SIZE
            },
            "min_group_size": shared_stats.MIN_GROUP_SIZE,
        }
    return samples, meta


def testable_windows(
    samples: Dict[int, Dict[str, np.ndarray]]
) -> Dict[int, Dict[str, np.ndarray]]:
    """Drop the classes too small to test, keeping every window.

    The window itself is kept even when nothing in it survives: a window that could not be tested
    has to reach the output as such, or a reader cannot tell it from a window nobody looked at.

    Args:
        samples: The unfiltered cells from :func:`window_samples`.

    Returns:
        The same windows, each holding only the classes with at least ``MIN_GROUP_SIZE``
        recordings.
    """
    return {
        window: {
            group: values for group, values in cell.items()
            if values.size >= shared_stats.MIN_GROUP_SIZE
        }
        for window, cell in samples.items()
    }


def analyse_windows(
    per_recording: pd.DataFrame, column: str, *, alpha: float = DEFAULT_ALPHA
) -> Dict[str, Any]:
    """Test whether the class trajectories of one readout differ, window by window.

    Args:
        per_recording: The class-axis per-recording-per-window frame.
        column: The value column to test.
        alpha: Family-wise error rate the Holm correction controls, **within this clock**.

    Returns:
        The record: whether the test could run at all, the per-window omnibus results with their
        Holm-adjusted $p$-values, the pairwise comparisons for the windows that survived, and the
        pooled context row.
    """
    groups = (
        cohort.ordered_groups(
            sorted(set(per_recording["group"].astype(str))), labels.CLASS_COLUMN
        )
        if not per_recording.empty and "group" in per_recording.columns
        else []
    )
    pooled = _pooled_test(per_recording, column)
    if len(groups) < 2:
        return {
            "metric_column": column,
            "group_column": labels.CLASS_COLUMN,
            "tested": False,
            "reason": f"fewer than two clinical classes present ({groups or 'none'})",
            "alpha": float(alpha),
            "bin_width_hours": float(TRAJECTORY_BIN_HOURS),
            "per_window": [],
            "pairwise": {},
            "pooled": pooled,
            "method": METHOD,
        }

    # The cohort half of the job is this analysis's own: which classes are in a window, in which
    # order, and which were too small to enter the test. That order is load-bearing rather than
    # cosmetic: the pairwise sweep names each pair in the order it receives, so passing the
    # classes severity-descending is what makes every comparison read HIE against acidosis, HIE
    # against healthy, acidosis against healthy -- more severe against less severe, never the
    # reverse.
    # The arithmetic half -- the omnibus, Holm across the windows and the pairwise sweep on the
    # survivors -- is ``stats.windowed_group_comparisons``, shared with every other trajectory
    # analysis in the family, so that "significant" has one definition across the repository
    # rather than one per clock.
    samples_by_window, meta_by_window = window_samples(per_recording, column)

    outcome = shared_stats.windowed_group_comparisons(
        testable_windows(samples_by_window), meta_by_window=meta_by_window, alpha=alpha
    )
    per_window = outcome["per_window"]
    significant = [record for record in per_window if record["significant"]]
    return {
        "metric_column": column,
        "group_column": labels.CLASS_COLUMN,
        "tested": True,
        "alpha": float(alpha),
        "bin_width_hours": float(TRAJECTORY_BIN_HOURS),
        "classes": groups,
        "n_windows": len(per_window),
        "n_windows_tested": outcome["n_windows_tested"],
        "n_significant_windows": len(significant),
        "significant_bin_centers_h": [record["bin_center_h"] for record in significant],
        "per_window": per_window,
        "pairwise": outcome["pairwise"],
        "pooled": pooled,
        "method": METHOD,
    }


def _pooled_test(per_recording: pd.DataFrame, column: str) -> Dict[str, Any]:
    """One clock-ignoring Kruskal-Wallis across the classes, flagged as confounded.

    Args:
        per_recording: The class-axis per-recording-per-window frame.
        column: The value column.

    Returns:
        The omnibus record, tagged ``confounded_by_time`` with the reason, so it cannot be read as
        the trajectory result. Nothing in this pipeline consumes it.
    """
    if per_recording.empty or "group" not in per_recording.columns:
        usable: Dict[str, np.ndarray] = {}
    else:
        # One value per recording first. The frame is keyed per (recording, window), so a recording
        # spanning several windows would otherwise enter the test several times: the p-value would
        # be pseudo-replicated by the windows-per-recording factor and the reported `n` would be a
        # row count wearing a recording's name. It also lets duplicated rows clear the
        # `MIN_GROUP_SIZE` floor that exists to keep tiny cohorts out.
        pooled_frame = per_recording.groupby(["group", "guid"], as_index=False)[column].mean()
        usable, _ = _class_samples(pooled_frame, column)
    record = shared_stats.kruskal_across_groups(usable)
    record["confounded_by_time"] = True
    record["note"] = (
        "pooled across every window, so a difference here can be an artifact of the classes "
        "covering the second-stage axis unequally rather than a difference in coupling; it is "
        "context and is consumed by nothing"
    )
    return record


# =============================================================================
# Emission
# =============================================================================
def significance_frame(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten the per-window omnibus results of every readout into one table."""
    rows: List[Dict[str, Any]] = []
    for record in records:
        for window in record.get("per_window") or []:
            rows.append(
                {
                    "metric_column": record["metric_column"],
                    "time_bin": window["time_bin"],
                    "bin_center_h": window["bin_center_h"],
                    "n_classes": window.get("n_groups"),
                    "n_recordings": sum((window.get("n_per_group") or {}).values()),
                    "statistic": window.get("statistic"),
                    "p_value": window.get("p_value"),
                    "p_holm": window.get("p_holm", float("nan")),
                    "significant": window.get("significant", False),
                    "alpha": window.get("alpha", float("nan")),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "metric_column", "time_bin", "bin_center_h", "n_classes", "n_recordings",
            "statistic", "p_value", "p_holm", "significant", "alpha",
        ],
    )


def pairwise_frame(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten the surviving windows' pairwise comparisons into one table."""
    rows: List[Dict[str, Any]] = []
    for record in records:
        centres = {
            int(window["time_bin"]): window["bin_center_h"]
            for window in record.get("per_window") or []
        }
        for key, comparisons in (record.get("pairwise") or {}).items():
            for item in comparisons:
                rows.append(
                    {
                        "metric_column": record["metric_column"],
                        "time_bin": int(key),
                        "bin_center_h": centres.get(int(key), float("nan")),
                        "left": item["left"],
                        "right": item["right"],
                        "n_left": item["n_left"],
                        "n_right": item["n_right"],
                        "p_value": item["p_value"],
                        "cliffs_delta": item["cliffs_delta"],
                        "magnitude": item["magnitude"],
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "metric_column", "time_bin", "bin_center_h", "left", "right", "n_left", "n_right",
            "p_value", "cliffs_delta", "magnitude",
        ],
    )


def build_trajectory_figure(
    rows: Sequence[Dict[str, Any]], axis: str, readout: cohort.ClockReadout
) -> Any:
    """Draw one readout's class trajectories against this clock, with each window's $n$ annotated.

    One readout per figure rather than one panel each on a shared page, exactly as the delivery
    clock draws them: the two are both in nats per anchor and routinely orders of magnitude apart.

    The count is annotated per window rather than reported once for the analysis because it is what
    a trajectory hides -- and on this clock it hides more of it than on the other: in the sampled
    onset table the second stage begins a couple of hours before delivery, so the windows after
    onset hold far fewer recordings than the windows before it, and a median there will wander for
    reasons that have nothing to do with the coupling.

    Args:
        rows: The long-form trajectory rows, of every readout; this selects its own.
        axis: The cohort axis to draw.
        readout: The readout this figure resolves.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(1, height_per_row=3.2)
    _draw_panel(
        axes[0, 0],
        [
            row for row in rows
            if row["group_column"] == axis and row["metric"] == readout.name
        ],
        axis,
        title=f"{readout.name} against time from second-stage onset, by {axis}",
    )
    return figure


def _draw_panel(ax: Any, rows: Sequence[Dict[str, Any]], axis: str, *, title: str) -> int:
    """Draw one readout's trajectories, median with inter-quartile ribbon, per cohort.

    Args:
        ax: Target axes.
        rows: The rows of one readout on one axis.
        axis: The cohort axis, for the colour and order conventions.
        title: Panel title.

    Returns:
        The number of cohorts drawn. Zero draws the empty note instead.
    """
    groups = cohort.ordered_groups([row["group"] for row in rows], axis) if rows else []
    if not groups:
        ax.text(
            0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes,
            ha="center", va="center", fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY,
        )
        ax.set_title(title)
        figures.style_axes(ax)
        return 0

    # From this package's one cohort palette, so a class is the same green / amber / red here as on
    # every other figure this evaluation draws of it.
    colours = figures.group_colors(groups)
    for group in groups:
        cell = sorted(
            (row for row in rows if row["group"] == group), key=lambda row: row["bin_center_h"]
        )
        if not cell:
            continue
        x = np.array([row["bin_center_h"] for row in cell], dtype=np.float64)
        colour = colours.get(group, figures.COLOR_BLUE)
        ax.fill_between(
            x,
            np.array([row["q25"] for row in cell], dtype=np.float64),
            np.array([row["q75"] for row in cell], dtype=np.float64),
            color=colour, alpha=0.15, linewidth=0,
        )
        ax.plot(
            x, np.array([row["median"] for row in cell], dtype=np.float64),
            marker="o", markersize=3, color=colour, linewidth=figures.LINE_EMPHASIS,
            label=f"{group} (n={int(cell[0].get('n_recordings_total', 0))} deliveries)",
        )
        for row in cell:
            ax.annotate(
                str(int(row["n_recordings"])),
                (float(row["bin_center_h"]), float(row["median"])),
                textcoords="offset points", xytext=(0, 5), ha="center",
                fontsize=figures.FONT_TINY, color=colour,
            )
    ax.set_title(title)
    ax.set_xlabel(AXIS_LABEL)
    ax.set_ylabel("nats per anchor")
    # The axis is **not** inverted, which is the whole difference from the delivery clock: this
    # coordinate is signed and reads naturally left to right, and the line marks the landmark
    # itself so the two halves of the axis are told apart by the page rather than by the label.
    ax.axvline(
        0.0, color=figures.COLOR_GRAY, linestyle=":", linewidth=figures.LINE_REGULAR, zorder=0
    )
    ax.legend(fontsize=figures.FONT_LABEL, loc="best")
    figures.style_axes(ax)
    return len(groups)


def build_windows_figure(
    class_frame: pd.DataFrame, record: Dict[str, Any], readout: cohort.ClockReadout
) -> Any:
    """Draw what one readout's trajectory is made of: the distributions, the significance, the
    effects.

    The trajectory figure draws one number per (window, class) cell and the tests were run on the
    values behind it; this draws both, on one axis, so a reader is not asked to hold a median in
    mind while opening a CSV. The cells come from :func:`window_samples`, which is also where the
    tested cells come from. One readout per page, for the reason the trajectory figure is one per
    page.

    Args:
        class_frame: The class-axis per-recording-per-window frame.
        record: That readout's significance record.
        readout: The readout this page resolves.

    Returns:
        The figure; the caller renders and closes it.
    """
    present = (
        sorted(set(class_frame["group"].astype(str)))
        if len(class_frame) and "group" in class_frame.columns
        else []
    )
    samples, _ = window_samples(class_frame, readout.column)
    # Aligned with the record's own window list by construction rather than by agreement, so a
    # window the test skipped cannot shift the cells drawn under the windows after it.
    order = [int(row["time_bin"]) for row in record.get("per_window") or []]

    return figures.windowed_comparison_figure(
        [(readout.name, [samples.get(key, {}) for key in order], record)],
        groups=cohort.ordered_groups(present, labels.CLASS_COLUMN),
        bin_width=TRAJECTORY_BIN_HOURS,
        # The same floor the test excludes a cell at, passed rather than defaulted, so the page and
        # the p-values beneath it agree about which cells carry evidence.
        min_body_size=shared_stats.MIN_GROUP_SIZE,
        xlabel=AXIS_LABEL,
        ylabel="nats per anchor",
        # Natural orientation with the onset marked, not the delivery clock's inversion.
        delivery_orientation=False,
    )


def _skip(reason: str, n_segments: int) -> Dict[str, Any]:
    """Return the recorded skip, and log it.

    Args:
        reason: Why there is no second-stage trajectory.
        n_segments: How many segments the table held.

    Returns:
        The protocol's keys with ``n_samples`` ``None`` -- this analysis scored no population, and
        a zero would enter the coverage block as a disagreement with every analysis that did.
    """
    logger.warning(f"{ANALYSIS_DIRNAME}: skipped -- {reason}")
    return {
        "n_samples": None,
        "composition": {},
        "plan": {"capped": False},
        "skipped": True,
        "reason": reason,
        "n_segments": int(n_segments),
        "files": [],
    }


def run_second_stage_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Bin both coupling readouts against signed hours from second-stage onset and test the classes.

    Args:
        context: The analysis context, read for the collected per-sample table.
        eval_config: The validated block. Unused: neither the window width nor the significance
            level is configurable, for the reason stated on each constant.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused: the population is the set of recordings carrying
            an onset on every segment, which only the table knows.

    Returns:
        The protocol's keys plus the eligibility record, the trajectory summary, the per-window
        significance of both readouts, and the paths written. A recorded skip -- naming its cause --
        on an empty table, a table collected before the onset column existed, a cohort with no
        onset at all, a cohort whose readouts are all non-finite, or a single-class split.
    """
    per_sample = context.collection.per_sample
    # The run's horizon, applied before anything is binned: it bounds the
    # population on the segment's own start, so every clock in the run answers
    # for the same segments. ``None`` leaves the frame untouched.
    per_sample = cohort.within_horizon(
        per_sample, eval_config.get("max_hours_before_delivery")
    )
    if per_sample.empty:
        return _skip("the collected per-sample table was empty", 0)
    if cohort.SECOND_STAGE_COLUMN not in per_sample.columns:
        return _skip(
            f"the collected table carries no '{cohort.SECOND_STAGE_COLUMN}' column, so no "
            f"recording can be placed on the second-stage clock; a run directory collected before "
            f"that column existed has to be collected again to gain it",
            len(per_sample),
        )

    eligibility, eligible = eligible_rows(per_sample)
    population = eligibility_summary(eligibility)
    if eligible.empty:
        return _skip(
            f"no recording carried a second-stage onset on every one of its segments "
            f"({population['n_dropped_no_onset']} of {population['n_recordings']} recording(s) "
            f"dropped), so this cohort cannot be placed on the second-stage clock",
            len(per_sample),
        )

    per_recording = build_per_recording(eligible)
    rows = build_trajectory_rows(per_recording)
    if not rows:
        return _skip(
            "no eligible segment carried both a finite readout and a cohort label, so there is no "
            "second-stage trajectory to draw",
            len(per_sample),
        )

    class_frame = per_recording.get(labels.CLASS_COLUMN, pd.DataFrame())
    classes = (
        cohort.ordered_groups(
            sorted(set(class_frame["group"].astype(str))), labels.CLASS_COLUMN
        )
        if len(class_frame) and "group" in class_frame.columns
        else []
    )
    if len(classes) < 2:
        # A skip rather than an untested trajectory, unlike the delivery clock: this analysis
        # exists for the class contrast on a subset population, and with one class it would report
        # a shape the other clock already draws over more recordings.
        return _skip(
            f"fewer than two clinical classes among the recordings that carry a second-stage "
            f"onset ({classes or 'none'})",
            len(per_sample),
        )

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    # Written before anything else: it is the record of *who was dropped and why*, and it is the
    # one table that describes the recordings this analysis does not otherwise mention.
    eligibility.to_csv(directory / ELIGIBILITY_FILENAME, index=False)
    pd.DataFrame(rows).to_csv(directory / TRAJECTORY_FILENAME, index=False)
    # One frame, both axes, so a reader can recompute any cell of the tables above from the
    # per-recording values that produced it.
    tall = pd.concat(
        [frame.assign(group_column=axis) for axis, frame in per_recording.items() if len(frame)],
        ignore_index=True,
    )
    tall.to_csv(directory / PER_RECORDING_FILENAME, index=False)

    significance = [analyse_windows(class_frame, readout.column) for readout in READOUTS]
    significance_frame(significance).to_csv(directory / SIGNIFICANCE_FILENAME, index=False)
    pairwise_frame(significance).to_csv(directory / PAIRWISE_FILENAME, index=False)

    # Four figures rather than two: each readout gets its trajectory and its windows page to
    # itself, so neither is drawn on the other's scale.
    figure_names = [
        str(
            figures.render_figure(
                builder(readout, record), directory / figure_stem(stem, readout)
            ).name
        )
        for readout, record in zip(READOUTS, significance)
        for stem, builder in (
            (
                TRAJECTORY_FIGURE,
                lambda readout, _record: build_trajectory_figure(
                    rows, labels.CLASS_COLUMN, readout
                ),
            ),
            (
                WINDOWS_FIGURE,
                lambda readout, record: build_windows_figure(class_frame, record, readout),
            ),
        )
    ]

    n_windows = int(tall[cohort.SECOND_STAGE_BIN_COLUMN].nunique()) if len(tall) else 0
    binned = cohort.add_second_stage_bins(eligible)
    logger.info(
        f"{ANALYSIS_DIRNAME}: {population['n_eligible']} of {population['n_recordings']} "
        f"recording(s) placed on the second-stage clock over {n_windows} window(s) of "
        f"{TRAJECTORY_BIN_HOURS:g} h; {population['n_dropped_no_onset']} dropped for having no "
        f"onset, {population['n_onset_at_delivery']} kept with an onset at delivery and "
        f"{population['n_inconsistent_onset']} with an onset that moves within the recording"
    )
    return {
        # The eligible segments, which is fewer than every other analysis scores -- hence `capped`
        # below, so the coverage block reads this as a different population by design rather than
        # as two analyses disagreeing about who was evaluated.
        "n_samples": int(len(binned)),
        "composition": {
            "n_recordings": population["n_eligible"],
            "n_windows": n_windows,
        },
        "plan": {
            "capped": True,
            "reason": (
                "scored over the recordings that carry a second-stage onset only, which is a "
                "subset of the evaluated cohort"
            ),
        },
        "bin_width_hours": float(TRAJECTORY_BIN_HOURS),
        "eligibility": population,
        "readouts": [
            {
                "metric": readout.name,
                "source_column": readout.column,
                "meaning": readout.meaning,
            }
            for readout in READOUTS
        ],
        "significance": [
            {
                key: value
                for key, value in record.items()
                # The per-window and pairwise detail is on the two CSVs; what belongs in the
                # summary is the headline of each test plus the pooled row and its flag.
                if key not in ("per_window", "pairwise")
            }
            for record in significance
        ],
        # No grouped variants are declared here, and that is not an omission: this analysis is
        # already cut by cohort, so its frame carries one row per (cohort, window, recording).
        # Fanning the by-class and by-subgroup emitter over it would resolve a cut by a cut.
        "files": [
            ELIGIBILITY_FILENAME, TRAJECTORY_FILENAME, PER_RECORDING_FILENAME,
            SIGNIFICANCE_FILENAME, PAIRWISE_FILENAME, *figure_names,
        ],
    }
