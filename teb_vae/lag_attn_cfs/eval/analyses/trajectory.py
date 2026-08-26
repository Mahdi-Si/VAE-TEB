r"""The two coupling readouts against time -- inside a segment, and across a whole delivery.

Everything else in this pipeline reduces a segment to one number. The per-anchor table exists so
that two questions can be asked that a per-segment mean cannot answer, and this is the analysis
that asks them.

**Within a segment**: how do $K_t$ and ``pred_gap`` behave against time-in-segment? The answer is
structural as much as physiological, and **the structure differs from the raw cells' in a way that
changes the expected shape of the figure**.

There, the profile spans the whole segment and droops at both ends: the warm-up prefix carries no
loss term, the lag support is truncated until $t \ge L - 1$, and the final $H$ anchors have no
fully observed future. Here, only the last of those three is still visible:

* **Nothing below the anchor floor exists at all.** The causal front end forces
  $F = 133$ of $T = 300$, and the forward decodes *no* anchor below it -- there is no row on the
  per-anchor table to profile, rather than a row scoring a warm-up value. So the profile **starts
  at $F$** and the reader sees no warm-up droop, because the region a droop would live in was never
  decoded. The emitted record carries the floor read off the run's own geometry beside the first
  anchor actually observed, so "the profile starts here" is checkable rather than asserted.
* **The lag truncation is inert.** The furthest searched lag is $L - 1 = 90$ and the floor is
  $133$, so every lag is causally valid at every scored anchor and there is no early-anchor
  truncation to see. This is a consequence of the floor rather than of the lag axis, and an arm
  that lowered the floor below $90$ would reintroduce it.
* **The final $H$ anchors are still never scored**, exactly as in the raw cells, and $H$ is $15$
  here rather than $30$.

So the expected shape is a profile that begins at $F$ and is otherwise flat, and a profile that
is *not* is the finding.

**Across a delivery**: a recording contributes tens of consecutive twenty-minute segments, and
laid end to end they are a trajectory through labour. Assembling it is the one thing the legacy
pipeline did that nothing else here reproduces, and it needs three things done right:

* **Overlapping timesteps are averaged, not duplicated.** Segments overlap by construction, so the
  same absolute second is scored by two of them; concatenating would put that second on the
  trajectory twice, at two different values, and every downstream mean would weight the overlap
  double.
* **A gap breaks the line.** A recording missing an hour has no trajectory across that hour.
  Joining the two ends would draw an interpolation through nothing and read as a slow trend.
* **The segment joins are recorded.** A discontinuity at a join is an artifact of assembly rather
  than a physiological event, and ``epoch_boundaries`` is what lets a reader tell the two apart.

The absolute coordinate is $t_{\mathrm{abs}} = \mathrm{epoch} + 4t$ seconds, negative before
delivery: ``epoch`` is the segment's own start on that axis and an anchor is $4$ s of it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn_cfs.eval import cohort
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "trajectory"

#: What it writes. The whole-delivery table is parquet for the reason the per-anchor table is: it
#: has one row per scored *second* of every recording, which is the same order of magnitude as the
#: anchor table it is built from and an order of magnitude past what a CSV should carry.
WITHIN_SEGMENT_FILENAME = "within_segment_profile.csv"
WHOLE_DELIVERY_FILENAME = "whole_delivery.parquet"
BOUNDARIES_FILENAME = "whole_delivery_boundaries.csv"
SUMMARY_FILENAME = "whole_delivery_summary.csv"

#: The figure, named as ``FIGURE_GUIDE.md`` names it.
PROFILE_FIGURE = "trajectory_profile"

#: The per-anchor columns profiled, as ``(reported name, column, axis label)``. Both readouts,
#: because they fail differently -- the KL is inflated by an arbitrary factor whenever the prior
#: variance sits on its clamp, and ``pred_gap`` is not.
READOUTS: Tuple[Tuple[str, str, str], ...] = (
    ("kld_per_t", "kld_per_t", "nats per anchor"),
    ("pred_gap_mc_nats", "mc_pred_gap", "nats per anchor"),
)

#: A join between two segments of one recording is a break when the gap exceeds one anchor step.
#: Equality is not a break: consecutive anchors are exactly ``SECONDS_PER_STEP`` apart.
BREAK_TOLERANCE_S = SECONDS_PER_STEP * 1.5


def anchor_floor(record: Dict[str, Any]) -> Optional[int]:
    r"""The anchor floor $F$ the run decoded at, or ``None``.

    Read off the collection record's geometry rather than assumed, for the same reason the block
    width is: it is a property of the checkpoint that was scored -- the causal front end resolves
    it from the warm-up budget -- and a constant here would state something about a different
    model whenever an arm moved it.

    It is reported rather than enforced. The profile's own first anchor comes from the per-anchor
    table and would still be the truth if the two disagreed; what a disagreement means is that the
    tables and the record describe different runs, and saying so is more useful than raising after
    the expensive pass has already happened.

    Args:
        record: The collection record. An offline re-run against a directory whose record carries
            no geometry block passes an empty mapping.

    Returns:
        $F$, or ``None`` when the geometry is absent.
    """
    floor = (record.get("geometry") or {}).get("anchor_floor")
    return None if floor is None else int(floor)


# =============================================================================
# Within a segment
# =============================================================================
def within_segment_profile(per_anchor: pd.DataFrame) -> pd.DataFrame:
    """Profile each readout against time-in-segment, on per-recording units.

    The reduction is the pipeline's chain applied along the anchor axis: within an anchor, average
    over a recording's segments; across recordings, take the quartiles. Pooling segments directly
    would let a recording contributing thirty-seven of them decide the shape.

    Args:
        per_anchor: The per-anchor table.

    Returns:
        One row per anchor: the anchor, its seconds into the segment, how many recordings scored
        it, and each readout's mean and quartiles across those recordings. Empty with the columns
        present when the table carries no anchors.
    """
    columns = ["anchor", "seconds_in_segment", "metric", "n_recordings", "mean", "q25", "median", "q75"]
    present = [column for _, column, _ in READOUTS if column in getattr(per_anchor, "columns", [])]
    if per_anchor.empty or "anchor" not in per_anchor.columns or not present:
        return pd.DataFrame(columns=columns)

    keys = ["anchor", "guid"] if "guid" in per_anchor.columns else ["anchor"]
    per_recording = per_anchor.groupby(keys)[present].mean().reset_index()
    rows: List[Dict[str, Any]] = []
    for anchor, cell in per_recording.groupby("anchor", sort=True):
        for name, column, _ in READOUTS:
            if column not in present:
                continue
            values = np.asarray(cell[column], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            rows.append(
                {
                    "anchor": int(anchor),
                    "seconds_in_segment": float(anchor) * float(SECONDS_PER_STEP),
                    "metric": name,
                    "n_recordings": int(values.size),
                    "mean": float(values.mean()),
                    "q25": float(np.percentile(values, 25)),
                    "median": float(np.percentile(values, 50)),
                    "q75": float(np.percentile(values, 75)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


# =============================================================================
# Across a delivery
# =============================================================================
def whole_delivery(per_anchor: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""Assemble each recording's segments into one trajectory on the absolute time axis.

    Args:
        per_anchor: The per-anchor table, carrying ``guid``, ``epoch`` and ``anchor``.

    Returns:
        ``(trajectory, boundaries)``. The trajectory holds one row per ``(guid, t_abs_sec)`` --
        overlapping anchors from adjacent segments **averaged** into that one row and counted in
        ``n_contributing`` -- with ``gap_before_s`` giving the distance to the previous timestep,
        so a break is visible as a value above :data:`BREAK_TOLERANCE_S` rather than being drawn
        through. The boundaries name, per segment, the row index its first anchor landed on.
    """
    trajectory_columns = ["guid", "t_abs_sec", "hours_before_delivery", "n_contributing",
                          "gap_before_s"]
    boundary_columns = ["guid", "epoch", "index", "t_abs_sec"]
    present = [column for _, column, _ in READOUTS if column in getattr(per_anchor, "columns", [])]
    required = {"guid", "epoch", "anchor"}
    if per_anchor.empty or not required.issubset(set(per_anchor.columns)) or not present:
        return (
            pd.DataFrame(columns=trajectory_columns + present),
            pd.DataFrame(columns=boundary_columns),
        )

    frame = per_anchor[["guid", "epoch", "anchor", *present]].copy()
    frame["t_abs_sec"] = (
        np.asarray(frame["epoch"], dtype=np.float64)
        + np.asarray(frame["anchor"], dtype=np.float64) * float(SECONDS_PER_STEP)
    )

    pieces: List[pd.DataFrame] = []
    boundaries: List[Dict[str, Any]] = []
    for guid, cell in frame.groupby("guid", sort=True):
        grouped = cell.groupby("t_abs_sec", sort=True)
        # The mean is what makes an overlap one timestep rather than two rows at one second, and
        # the count beside it is what says the averaging happened.
        merged = grouped[present].mean().reset_index()
        merged["n_contributing"] = grouped.size().to_numpy()
        merged.insert(0, "guid", guid)
        times = np.asarray(merged["t_abs_sec"], dtype=np.float64)
        merged["hours_before_delivery"] = -times / cohort.SECONDS_PER_HOUR
        gaps = np.full(times.size, np.nan, dtype=np.float64)
        if times.size > 1:
            gaps[1:] = np.diff(times)
        merged["gap_before_s"] = gaps
        pieces.append(merged)

        # Where each segment's first anchor landed in this recording's own row order. A join is a
        # property of the assembly, and a step at one is an artifact rather than a finding.
        starts = cell.groupby("epoch")["t_abs_sec"].min()
        for epoch, start in starts.items():
            index = int(np.searchsorted(times, float(start)))
            boundaries.append(
                {
                    "guid": str(guid),
                    "epoch": float(epoch),
                    "index": index,
                    "t_abs_sec": float(start),
                }
            )

    trajectory = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=trajectory_columns + present
    )
    ordered = [name for name in trajectory_columns if name in trajectory.columns]
    trajectory = trajectory.reindex(columns=ordered + present)
    return trajectory, pd.DataFrame(boundaries, columns=boundary_columns)


def delivery_summary(trajectory: pd.DataFrame, boundaries: pd.DataFrame) -> pd.DataFrame:
    """Summarise each recording's assembled trajectory.

    Args:
        trajectory: The whole-delivery table.
        boundaries: The segment joins.

    Returns:
        One row per recording: how many timesteps it holds, how many of them merged an overlap,
        how many breaks it carries, its span in hours, and how many segments it was built from.
    """
    columns = ["guid", "n_timesteps", "n_overlapping", "n_breaks", "span_hours", "n_segments"]
    if trajectory.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, Any]] = []
    segments = (
        boundaries.groupby("guid").size() if len(boundaries) else pd.Series(dtype=np.int64)
    )
    for guid, cell in trajectory.groupby("guid", sort=True):
        gaps = np.asarray(cell["gap_before_s"], dtype=np.float64)
        times = np.asarray(cell["t_abs_sec"], dtype=np.float64)
        rows.append(
            {
                "guid": str(guid),
                "n_timesteps": int(len(cell)),
                "n_overlapping": int((np.asarray(cell["n_contributing"]) > 1).sum()),
                "n_breaks": int(np.sum(np.isfinite(gaps) & (gaps > BREAK_TOLERANCE_S))),
                "span_hours": float((times.max() - times.min()) / cohort.SECONDS_PER_HOUR)
                if times.size else float("nan"),
                "n_segments": int(segments.get(guid, 0)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


# =============================================================================
# The figure
# =============================================================================
def build_profile_figure(
    profile: pd.DataFrame, trajectory: pd.DataFrame, *, guid: Optional[str] = None
) -> Any:
    """Draw the within-segment profiles and one recording's assembled trajectory.

    Args:
        profile: The within-segment table.
        trajectory: The whole-delivery table.
        guid: Which recording to draw. The longest one when omitted, chosen because a short
            recording shows neither an overlap nor a break.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(len(READOUTS) + 1, height_per_row=2.8)
    for index, (name, _, ylabel) in enumerate(READOUTS):
        cell = profile[profile["metric"] == name] if len(profile) else profile
        _draw_within_segment(axes[index, 0], cell, title=f"{name} against time in segment",
                             ylabel=ylabel)
    _draw_whole_delivery(axes[len(READOUTS), 0], trajectory, guid=guid)
    return figure


def _draw_within_segment(ax: Any, cell: pd.DataFrame, *, title: str, ylabel: str) -> None:
    """Draw one readout's median-with-IQR profile against time in segment."""
    if not len(cell):
        ax.text(
            0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes,
            ha="center", va="center", fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY,
        )
        ax.set_title(title)
        figures.style_axes(ax)
        return
    ordered = cell.sort_values("seconds_in_segment")
    x = np.asarray(ordered["seconds_in_segment"], dtype=np.float64)
    ax.fill_between(
        x, np.asarray(ordered["q25"], dtype=np.float64),
        np.asarray(ordered["q75"], dtype=np.float64),
        color=figures.COLOR_BLUE, alpha=0.2, linewidth=0, label="IQR over recordings",
    )
    ax.plot(
        x, np.asarray(ordered["median"], dtype=np.float64),
        color=figures.COLOR_BLUE, linewidth=figures.LINE_EMPHASIS, label="median over recordings",
    )
    ax.set_title(title)
    ax.set_xlabel("Time in segment (s)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=figures.FONT_LABEL, loc="best")
    figures.style_axes(ax)


def _draw_whole_delivery(ax: Any, trajectory: pd.DataFrame, *, guid: Optional[str]) -> None:
    """Draw one recording's assembled trajectory, with its breaks left as breaks.

    The gap is inserted as ``NaN`` rather than left to the line: matplotlib joins consecutive
    points whatever their spacing, so a recording missing an hour would otherwise be drawn as a
    straight interpolation across it and read as a slow trend.
    """
    if trajectory.empty:
        ax.text(
            0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes,
            ha="center", va="center", fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY,
        )
        ax.set_title("Whole-delivery trajectory")
        figures.style_axes(ax)
        return

    chosen = guid if guid is not None else str(
        trajectory.groupby("guid").size().sort_values(ascending=False).index[0]
    )
    cell = trajectory[trajectory["guid"].astype(str) == str(chosen)].sort_values("t_abs_sec")
    hours = np.asarray(cell["hours_before_delivery"], dtype=np.float64)
    gaps = np.asarray(cell["gap_before_s"], dtype=np.float64)
    # A break is the gap *before* a sample, so overwriting that sample with NaN to draw the break
    # discards a measured value -- the first point after every gap was never drawn. Inserting the
    # NaN instead breaks the line between the two samples and keeps both. Loop-invariant, so it is
    # computed once here rather than per readout.
    breaks = np.flatnonzero(np.isfinite(gaps) & (gaps > BREAK_TOLERANCE_S))
    hours_with_breaks = np.insert(hours, breaks, np.nan)
    for name, column, _ in READOUTS:
        if column not in cell.columns:
            continue
        values = np.insert(np.asarray(cell[column], dtype=np.float64), breaks, np.nan)
        ax.plot(hours_with_breaks, values, linewidth=figures.LINE_REGULAR, label=name)
    ax.set_title(f"Whole-delivery trajectory: {chosen}")
    ax.set_xlabel("Time before delivery (hours)")
    ax.set_ylabel("nats per anchor")
    ax.invert_xaxis()
    ax.legend(fontsize=figures.FONT_LABEL, loc="best")
    figures.style_axes(ax)


# =============================================================================
# The registry entry point
# =============================================================================
def run_trajectory_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Profile both readouts within a segment and assemble each recording's whole trajectory.

    Args:
        context: The analysis context, read for the per-anchor table -- this analysis's only
            input, and the table's first general consumer -- and for the collection record's
            anchor floor, which is what makes "the profile starts at $F$" checkable.
        eval_config: The validated block. Unused: nothing here is tunable.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the assembly's accounting -- how many timesteps merged an
        overlap, how many breaks the recordings carry -- and the paths written. A recorded skip
        when the run carried no per-anchor table.
    """
    per_anchor = getattr(context.collection, "per_anchor", None)
    if per_anchor is None or per_anchor.empty:
        reason = (
            "the run carried no per-anchor rows, so there is neither a within-segment profile "
            "nor a trajectory to assemble"
        )
        logger.warning(f"{ANALYSIS_DIRNAME}: skipped -- {reason}")
        return {
            "n_samples": None,
            "composition": {},
            "plan": {"capped": False},
            "skipped": True,
            "reason": reason,
            "files": [],
        }

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    floor = anchor_floor(dict(getattr(context.collection, "record", None) or {}))
    profile = within_segment_profile(per_anchor)
    profile.to_csv(directory / WITHIN_SEGMENT_FILENAME, index=False)
    trajectory, boundaries = whole_delivery(per_anchor)
    trajectory.to_parquet(directory / WHOLE_DELIVERY_FILENAME, index=False)
    boundaries.to_csv(directory / BOUNDARIES_FILENAME, index=False)
    summary = delivery_summary(trajectory, boundaries)
    summary.to_csv(directory / SUMMARY_FILENAME, index=False)

    figure_name = str(
        figures.render_figure(
            build_profile_figure(profile, trajectory), directory / PROFILE_FIGURE
        ).name
    )

    n_recordings = int(len(summary))
    logger.info(
        f"{ANALYSIS_DIRNAME}: assembled {int(summary['n_timesteps'].sum()) if n_recordings else 0} "
        f"timestep(s) over {n_recordings} recording(s) from {len(per_anchor)} anchor row(s)"
    )
    return {
        # Segments, so this analysis's population is comparable with every other analysis's.
        "n_samples": (
            int(per_anchor["sample_index"].nunique())
            if "sample_index" in per_anchor.columns else None
        ),
        "composition": {"n_recordings": n_recordings, "n_anchor_rows": int(len(per_anchor))},
        "plan": {"capped": False},
        "within_segment": {
            "n_anchors": int(profile["anchor"].nunique()) if len(profile) else 0,
            "seconds_per_anchor": float(SECONDS_PER_STEP),
            # The three numbers the structural caveat rests on. ``first_anchor`` is what the
            # tables actually carry; ``anchor_floor`` is what the model says it decoded at; and
            # ``starts_at_floor`` is the two agreeing. They differ only when the tables and the
            # record describe different runs, which is worth surfacing rather than collapsing
            # into one number a reader would then have to trust.
            "first_anchor": int(profile["anchor"].min()) if len(profile) else None,
            "anchor_floor": floor,
            "starts_at_floor": (
                None if floor is None or not len(profile)
                else bool(int(profile["anchor"].min()) == floor)
            ),
        },
        "whole_delivery": {
            "n_timesteps": int(len(trajectory)),
            "n_overlapping_timesteps": int(summary["n_overlapping"].sum()) if n_recordings else 0,
            "n_breaks": int(summary["n_breaks"].sum()) if n_recordings else 0,
            "break_tolerance_s": float(BREAK_TOLERANCE_S),
            "n_boundaries": int(len(boundaries)),
            "max_span_hours": float(summary["span_hours"].max()) if n_recordings else float("nan"),
        },
        "files": [
            WITHIN_SEGMENT_FILENAME, WHOLE_DELIVERY_FILENAME, BOUNDARIES_FILENAME,
            SUMMARY_FILENAME, figure_name,
        ],
    }
