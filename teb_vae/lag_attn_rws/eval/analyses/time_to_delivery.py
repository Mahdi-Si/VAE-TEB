r"""Does the coupling change as delivery approaches, and does it change differently by class?

Every other reading of the coupling collapses a recording to one number. This one resolves the
same two readouts against **time before delivery** and asks whether the trajectories differ
between the clinical classes -- which is the form the clinical question is actually asked in.

**Both readouts travel, not just the KL.** The sibling pipeline tracks the KL alone; here
``pred_gap`` is tracked beside it, because the two fail differently. ``pred_gap`` is in the
decoder's own units and is immune to the prior-variance inflation, while
``source_conditioned_kl_raw`` is multiplied by an arbitrary factor whenever the prior variance
sits on its clamp. A trajectory visible in one and absent from the other is a finding about which
of the two is being read, and a run tracking only the KL cannot see it.

**The unit is the recording, inside a window as well as across the split.** A window's value for
a recording is the mean over that recording's segments falling in it; the tests then run over one
value per recording. Skipping that step would let a recording contributing eleven segments to a
window outvote one contributing two.

**Three layers of inference, in this order.**

* **Per window**, a Kruskal-Wallis across the classes present in it. Per window rather than
  pooled is what makes this a statement about the *trajectory*: it localises where the classes
  differ instead of collapsing gestation into one number. Non-parametric because these
  distributions are skewed and heavy-tailed.
* **Holm across the windows**, which form one family. Twenty-five windows at $\alpha = 0.05$
  produce a false positive with near certainty by construction. Holm rather than Bonferroni
  because it is uniformly more powerful at the same family-wise error rate.
* **Pairwise Mann-Whitney with Cliff's delta**, for the windows that survived Holm only. Running
  the pairwise tests on a window whose omnibus found nothing is the multiple-comparison problem
  with extra steps.

A **pooled** row is reported beside them and consumed by nothing: it ignores time, and the classes
do not cover the time axis equally, so a pooled difference can be a coverage artifact rather than
a difference in coupling. It carries ``confounded_by_time`` in the output for that reason.
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

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "time_to_delivery"

#: What it writes.
TRAJECTORY_FILENAME = "time_to_delivery_trajectory.csv"
PER_RECORDING_FILENAME = "time_to_delivery_per_recording.csv"
SIGNIFICANCE_FILENAME = "time_to_delivery_significance.csv"
PAIRWISE_FILENAME = "time_to_delivery_pairwise.csv"

#: The figure, named as ``FIGURE_GUIDE.md`` names it.
TRAJECTORY_FIGURE = "time_to_delivery_trajectory.pdf"

#: Width of a time-before-delivery window, in hours. Bound from the layer below rather than
#: restated: the lag structure is cut on the same windows and an analysis may not import another,
#: so one constant defines the grid both are read on. It is **not** an ``eval_config`` key -- an
#: operator who could widen it could merge two windows until a difference appeared or disappeared.
TRAJECTORY_BIN_HOURS = cohort.TRAJECTORY_BIN_HOURS

#: Family-wise error rate the Holm correction across the windows controls. Not configurable, for
#: the reason the bin width is not.
DEFAULT_ALPHA = 0.05

#: The two readouts tracked, as ``(reported name, per-sample column, what it is)``. Both, because
#: they fail differently -- see the module docstring.
READOUTS: Tuple[Tuple[str, str, str], ...] = (
    (
        "pred_gap_mc_nats",
        "mc_pred_gap",
        "Monte Carlo marginalised D_base - D_full in nats per anchor; in the decoder's own units "
        "and immune to the prior-variance inflation",
    ),
    (
        "source_conditioned_kl_raw_nats",
        "source_conditioned_kl_raw",
        "the unfloored KL between the two latents; inflated by an arbitrary factor whenever the "
        "prior variance sits on its clamp, which is why it is not read alone",
    ),
)

#: The per-sample columns this analysis reduces, in the order the tables carry them.
VALUE_COLUMNS: Tuple[str, ...] = tuple(column for _, column, _ in READOUTS)

#: The method sentence written into every record, so a $p$-value here is readable without this
#: module.
METHOD = (
    "Per time-before-delivery window: Kruskal-Wallis across clinical classes over one value per "
    "recording, Holm step-down correction across the windows as one family, pairwise two-sided "
    "Mann-Whitney U with Cliff's delta for the windows significant after Holm only. "
    "Non-parametric throughout. Classes with fewer than "
    f"{shared_stats.MIN_GROUP_SIZE} recordings in a window are excluded from it and recorded. The "
    "pooled row ignores time and is confounded by unequal class coverage of the axis."
)


# =============================================================================
# Binning and the trajectory tables
# =============================================================================
def build_per_recording(
    per_sample: pd.DataFrame, *, width: float = TRAJECTORY_BIN_HOURS
) -> Dict[str, pd.DataFrame]:
    """Reduce the per-sample table to one row per (cohort, window, recording), per axis.

    Args:
        per_sample: The collected per-sample table.
        width: Window width in hours.

    Returns:
        Cohort axis to its per-recording-per-window frame. Both axes are built because a reader
        wants both cuts; only the class axis is tested.
    """
    binned = cohort.add_time_bins(per_sample, width=width)
    return {
        axis: cohort.per_recording_in_bins(binned, VALUE_COLUMNS, group_column=axis)
        for axis in labels.GROUP_COLUMNS
    }


def build_trajectory_rows(per_recording: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    """Summarise every (axis, cohort, window) cell of both readouts, over its recordings.

    Args:
        per_recording: The per-axis frames from :func:`build_per_recording`.

    Returns:
        Long-form rows carrying the axis, the cohort, the window, the recording count and the
        quartiles -- one table a reader can pivot rather than four.
    """
    rows: List[Dict[str, Any]] = []
    for axis, frame in per_recording.items():
        for name, column, _ in READOUTS:
            for row in cohort.trajectory_rows(frame, column, metric=name):
                rows.append({"group_column": axis, "source_column": column, **row})
    return rows


# =============================================================================
# Significance across the classes, window by window
# =============================================================================
def _class_samples(frame: pd.DataFrame, column: str) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Split one window's rows into per-class finite vectors, dropping classes too small to test.

    Args:
        frame: One window's per-recording rows.
        column: The value column.

    Returns:
        ``(usable, excluded)`` -- the classes with at least ``MIN_GROUP_SIZE`` recordings, and the
        sizes of those without. The exclusion is returned rather than dropped because "this class
        had two recordings in this window" is the explanation for a window the test skipped, and a
        reader who cannot see it will assume the comparison was made.
    """
    usable: Dict[str, np.ndarray] = {}
    excluded: Dict[str, int] = {}
    for group in sorted(set(frame["group"].astype(str))):
        values = np.asarray(
            frame.loc[frame["group"].astype(str) == group, column], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        if finite.size < shared_stats.MIN_GROUP_SIZE:
            excluded[group] = int(finite.size)
            continue
        usable[group] = finite
    return usable, excluded


def analyse_windows(
    per_recording: pd.DataFrame, column: str, *, alpha: float = DEFAULT_ALPHA
) -> Dict[str, Any]:
    """Test whether the class trajectories of one readout differ, window by window.

    Args:
        per_recording: The class-axis per-recording-per-window frame.
        column: The value column to test.
        alpha: Family-wise error rate the Holm correction controls.

    Returns:
        The record: whether the test could run at all, the per-window omnibus results with their
        Holm-adjusted $p$-values, the pairwise comparisons for the windows that survived, and the
        pooled context row. ``tested`` is ``False`` with a reason on a split carrying fewer than
        two classes -- the ordinary outcome on the healthy-only pretraining split, not a failure.
    """
    groups = (
        sorted(set(per_recording["group"].astype(str)))
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

    per_window: List[Dict[str, Any]] = []
    samples_by_window: Dict[int, Dict[str, np.ndarray]] = {}
    for bin_index in sorted(int(value) for value in per_recording[cohort.BIN_COLUMN].unique()):
        cell = per_recording[per_recording[cohort.BIN_COLUMN] == bin_index]
        usable, excluded = _class_samples(cell, column)
        record: Dict[str, Any] = {
            "time_bin": int(bin_index),
            "bin_center_h": float(cell[cohort.BIN_CENTER_COLUMN].iloc[0]),
            "groups_excluded_as_too_small": excluded,
            "min_group_size": shared_stats.MIN_GROUP_SIZE,
        }
        if len(usable) >= 2:
            record.update(shared_stats.kruskal_across_groups(usable))
            samples_by_window[int(bin_index)] = usable
        else:
            record.update(
                {
                    "test": "kruskal-wallis",
                    "n_groups": len(usable),
                    "n_per_group": {name: int(values.size) for name, values in usable.items()},
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "note": "fewer than two classes had enough recordings in this window",
                }
            )
        per_window.append(record)

    adjusted = shared_stats.holm_adjust([record["p_value"] for record in per_window])
    n_tested = sum(1 for record in per_window if np.isfinite(record["p_value"]))
    for record, value in zip(per_window, adjusted):
        record["p_holm"] = float(value)
        record["alpha"] = float(alpha)
        record["correction"] = "holm"
        record["n_windows_in_family"] = n_tested
        record["significant"] = bool(np.isfinite(value) and value < float(alpha))

    pairwise = {
        str(record["time_bin"]): shared_stats.pairwise_comparisons(
            samples_by_window[int(record["time_bin"])]
        )
        for record in per_window
        if record["significant"]
    }
    significant = [record for record in per_window if record["significant"]]
    return {
        "metric_column": column,
        "group_column": labels.CLASS_COLUMN,
        "tested": True,
        "alpha": float(alpha),
        "bin_width_hours": float(TRAJECTORY_BIN_HOURS),
        "classes": groups,
        "n_windows": len(per_window),
        "n_windows_tested": n_tested,
        "n_significant_windows": len(significant),
        "significant_bin_centers_h": [record["bin_center_h"] for record in significant],
        "per_window": per_window,
        "pairwise": pairwise,
        "pooled": pooled,
        "method": METHOD,
    }


def _pooled_test(per_recording: pd.DataFrame, column: str) -> Dict[str, Any]:
    """One time-ignoring Kruskal-Wallis across the classes, flagged as confounded.

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
        # One value per recording first. The frame is keyed per (recording, window), so a
        # recording spanning eleven windows would otherwise enter the test eleven times: the
        # p-value would be pseudo-replicated by the windows-per-recording factor and the reported
        # `n` would be a row count wearing a recording's name. It also lets duplicated rows clear
        # the `MIN_GROUP_SIZE` floor that exists to keep tiny cohorts out.
        pooled_frame = per_recording.groupby(["group", "guid"], as_index=False)[column].mean()
        usable, _ = _class_samples(pooled_frame, column)
    record = shared_stats.kruskal_across_groups(usable)
    record["confounded_by_time"] = True
    record["note"] = (
        "pooled across every window, so a difference here can be an artifact of the classes "
        "covering the time-to-delivery axis unequally rather than a difference in coupling; it is "
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


def build_trajectory_figure(rows: Sequence[Dict[str, Any]], axis: str) -> Any:
    """Draw one panel per readout: the class trajectories, with each window's $n$ annotated.

    The count is annotated per window rather than reported once for the analysis because it is
    what a trajectory hides: a window's median can move because the cohort changed rather than
    because the coupling did, and the only thing that says which is the number of recordings
    behind that point.

    Args:
        rows: The long-form trajectory rows.
        axis: The cohort axis to draw.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(len(READOUTS), height_per_row=3.2)
    selected = [row for row in rows if row["group_column"] == axis]
    for index, (name, _, _) in enumerate(READOUTS):
        _draw_panel(
            axes[index, 0],
            [row for row in selected if row["metric"] == name],
            axis,
            title=f"{name} against time before delivery, by {axis}",
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
            ha="center", va="center", fontsize=9, color=figures.COLOR_GRAY,
        )
        ax.set_title(title)
        figures.style_axes(ax)
        return 0

    # From the one shared mapping, so a cohort is the same colour here as on every other figure
    # this repository draws of it.
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
            marker="o", markersize=3, color=colour, linewidth=1.4,
            label=f"{group} (n={int(sum(row['n_recordings'] for row in cell))})",
        )
        for row in cell:
            ax.annotate(
                str(int(row["n_recordings"])),
                (float(row["bin_center_h"]), float(row["median"])),
                textcoords="offset points", xytext=(0, 5), ha="center",
                fontsize=5, color=colour,
            )
    ax.set_title(title)
    ax.set_xlabel("Time before delivery (hours)")
    ax.set_ylabel("nats per anchor")
    # Delivery sits at the right, so the eye reads left to right toward it.
    ax.invert_xaxis()
    ax.legend(fontsize=7, loc="best")
    figures.style_axes(ax)
    return len(groups)


def _skip(reason: str, n_segments: int) -> Dict[str, Any]:
    """Return the recorded skip, and log it.

    Args:
        reason: Why there is no trajectory.
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


def run_time_to_delivery_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Bin both coupling readouts against time before delivery and test the class trajectories.

    Args:
        context: The analysis context, read for the collected per-sample table.
        eval_config: The validated block. Unused: neither the window width nor the significance
            level is configurable, for the reason stated on each constant.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused: the population is the set of segments carrying a
            finite ``epoch``, which only the table knows.

    Returns:
        The protocol's keys plus the trajectory summary, the per-window significance of both
        readouts, and the paths written. A recorded skip on a split with no ``epoch`` or no
        cohort labels.
    """
    per_sample = context.collection.per_sample
    if per_sample.empty:
        return _skip("the collected per-sample table was empty", 0)
    if "epoch" not in per_sample.columns:
        return _skip(
            "the batches carried no 'epoch' field, so time before delivery is unavailable",
            len(per_sample),
        )

    per_recording = build_per_recording(per_sample)
    rows = build_trajectory_rows(per_recording)
    if not rows:
        return _skip(
            "no segment carried both a finite epoch and a cohort label, so there is no "
            "trajectory to draw",
            len(per_sample),
        )

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(directory / TRAJECTORY_FILENAME, index=False)
    # One frame, both axes, so a reader can recompute any cell of the tables above from the
    # per-recording values that produced it.
    tall = pd.concat(
        [frame.assign(group_column=axis) for axis, frame in per_recording.items() if len(frame)],
        ignore_index=True,
    )
    tall.to_csv(directory / PER_RECORDING_FILENAME, index=False)

    class_frame = per_recording.get(labels.CLASS_COLUMN, pd.DataFrame())
    significance = [
        analyse_windows(class_frame, column) for _, column, _ in READOUTS
    ]
    significance_frame(significance).to_csv(directory / SIGNIFICANCE_FILENAME, index=False)
    pairwise_frame(significance).to_csv(directory / PAIRWISE_FILENAME, index=False)

    figure_name = str(
        figures.render_to_pdf(
            build_trajectory_figure(rows, labels.CLASS_COLUMN), directory / TRAJECTORY_FIGURE
        ).name
    )

    n_recordings = int(tall["guid"].nunique()) if "guid" in tall.columns else 0
    logger.info(
        f"{ANALYSIS_DIRNAME}: {len(rows)} cell(s) over "
        f"{int(tall[cohort.BIN_COLUMN].nunique()) if len(tall) else 0} window(s) of "
        f"{TRAJECTORY_BIN_HOURS:g} h from {n_recordings} recording(s)"
    )
    return {
        "n_samples": int(len(cohort.add_time_bins(per_sample))),
        "composition": {
            "n_recordings": n_recordings,
            "n_windows": int(tall[cohort.BIN_COLUMN].nunique()) if len(tall) else 0,
        },
        "plan": {"capped": False},
        "bin_width_hours": float(TRAJECTORY_BIN_HOURS),
        "readouts": [
            {"metric": name, "source_column": column, "meaning": note}
            for name, column, note in READOUTS
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
            TRAJECTORY_FILENAME, PER_RECORDING_FILENAME, SIGNIFICANCE_FILENAME,
            PAIRWISE_FILENAME, figure_name,
        ],
    }
