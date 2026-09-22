r"""The pooled lag profile of this architecture, and its shape per segment, recording and cohort.

This cell's reading of the question the lag-attentive cells ask through ``attention`` and
``lag_kl``: where on the lag axis does the fitted model's source pathway sit, how concentrated is
it, and does that differ by cohort. The quantities are this architecture's own -- the proposal
norm and the signed divergence drop the collection pass writes per segment -- and none of them is
an attention distribution or an allocation of the divergence; every artifact here says so.

Four tables and two figures:

* ``proposal_profile_segments.csv`` -- one row per segment: the shape statistics of both
  profiles and their mass in each declared band, from
  :func:`~teb_vae.lag_slot_transformer_cfs.eval.lag_structure.segment_table`.
* ``proposal_profile_per_recording.csv`` -- the same reduced to one row per recording, which the
  runner fans out by clinical class and by subgroup.
* ``proposal_profile_stratified.csv`` -- the pooled profile of each cohort on both cohort axes,
  each recording weighted once, as a share and in the profile's own unit.
* ``proposal_profile_peaks.csv`` -- where each pooled profile peaks, with the family's
  degeneracy guard beside the position: a flat profile still has a confident argmax.
* ``proposal_profile`` -- the pooled profiles with their inter-quartile spread over recordings,
  the declared bands shaded and the guarded peak marked.
* ``proposal_profile_stratified`` -- the normalised profile per class and per subgroup, for both
  readouts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn_cfs.eval import cohort
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.frames import grouped_frame_entry, scored_sample_count
from teb_vae.lag_slot_transformer_cfs.eval import lag_structure

#: Where this analysis writes, inside the results directory.
ANALYSIS_DIRNAME = "proposal_profile"

#: What it writes.
SEGMENTS_FILENAME = "proposal_profile_segments.csv"
PER_RECORDING_FILENAME = "proposal_profile_per_recording.csv"
STRATIFIED_FILENAME = "proposal_profile_stratified.csv"
PEAKS_FILENAME = "proposal_profile_peaks.csv"
PROFILE_FIGURE = "proposal_profile"
STRATIFIED_FIGURE = "proposal_profile_stratified"

#: The per-recording columns the runner fans out by cohort: the position, the concentration and
#: the magnitude of each profile. Six rather than thirty-two, because a grouped page per column
#: is a page a reader opens, and these are the ones a cohort contrast is asked in.
GROUPED_STATISTICS: tuple = ("centroid", "entropy", "effective_support", "total_nats")


def grouped_columns(sources: Sequence[lag_structure.ProfileSource]) -> List[str]:
    """The per-recording columns fanned out by cohort, for the profiles present.

    Args:
        sources: The profiles the segment table carries.

    Returns:
        The column names, in profile-then-statistic order.
    """
    columns = [
        lag_structure.statistic_column(statistic, source)
        for source in sources
        for statistic in GROUPED_STATISTICS
    ]
    # The discarded negative mass of the signed profile is a reading of its own: a cohort whose
    # proposals cancel more is a different finding from one whose proposals are smaller.
    columns.extend(f"net_{source.key}" for source in sources if source.signed)
    return columns


def stratified_rows(
    table: pd.DataFrame,
    matrices: Mapping[str, np.ndarray],
    axis: lag_structure.LagAxis,
) -> List[Dict[str, Any]]:
    """The pooled profile of every cohort on both cohort axes, plus the whole population.

    Args:
        table: The segment table, for the cohort labels and recording identifiers.
        matrices: ``{profile key: (n, L)}`` per-sample profiles in the table's row order.
        axis: The lag axis.

    Returns:
        Long-form rows: one per (axis, cohort, profile, lag), carrying the mean value in the
        profile's unit, its share of the cohort's rectified mass, the quartiles and the
        recording count.
    """
    rows: List[Dict[str, Any]] = []
    cuts: List[tuple] = [("all", "all", np.ones(len(table), dtype=bool))]
    for group_column in labels.GROUP_COLUMNS:
        if group_column not in table.columns:
            continue
        present = cohort.ordered_groups(
            sorted(set(table[group_column].dropna().astype(str))), group_column
        )
        for group in present:
            cuts.append((group_column, group, table[group_column].astype(str).to_numpy() == group))
    for group_column, group, selector in cuts:
        guids = table.loc[selector, "guid"].astype(str).tolist()
        for key, matrix in matrices.items():
            pooled = lag_structure.pooled_profile(matrix[selector], guids)
            mean = np.asarray(pooled["mean"], dtype=np.float64)
            positive = np.where(np.isfinite(mean), np.maximum(mean, 0.0), 0.0)
            total = float(positive.sum())
            for lag in range(axis.n_lags):
                rows.append({
                    "group_column": group_column,
                    "group": group,
                    "profile": key,
                    "n_recordings": int(pooled["n_recordings"]),
                    "lag_step": lag,
                    "seconds": float(axis.seconds[lag]),
                    "mean": float(mean[lag]),
                    "share": float(positive[lag] / total) if total > 0.0 else float("nan"),
                    "q25": float(pooled["q25"][lag]),
                    "median": float(pooled["median"][lag]),
                    "q75": float(pooled["q75"][lag]),
                })
    return rows


def peak_rows(stratified: Sequence[Mapping[str, Any]], axis: lag_structure.LagAxis) -> List[Dict[str, Any]]:
    """Locate the guarded peak of every pooled profile the stratified rows carry.

    Args:
        stratified: The rows :func:`stratified_rows` returned.
        axis: The lag axis.

    Returns:
        One row per (axis, cohort, profile).
    """
    frame = pd.DataFrame(stratified)
    rows: List[Dict[str, Any]] = []
    if frame.empty:
        return rows
    for (group_column, group, key), cell in frame.groupby(
        ["group_column", "group", "profile"], sort=False
    ):
        ordered = cell.sort_values("lag_step")
        record = lag_structure.peak_record(ordered["mean"].to_numpy(), axis.seconds)
        rows.append({
            "group_column": group_column, "group": group, "profile": key,
            "n_recordings": int(ordered["n_recordings"].iloc[0]), **record,
        })
    return rows


# =============================================================================
# The figures
# =============================================================================
def build_profile_figure(
    stratified: Sequence[Mapping[str, Any]],
    peaks: Sequence[Mapping[str, Any]],
    sources: Sequence[lag_structure.ProfileSource],
    bands: Mapping[str, tuple],
    axis: lag_structure.LagAxis,
) -> Any:
    """The pooled profiles over the whole population, one panel per readout.

    Each panel draws the median over recordings with its inter-quartile ribbon, shades the
    declared bands, and marks the guarded peak -- or says the profile is degenerate, so a peak
    position is not read off a flat line.

    Args:
        stratified: The stratified rows, of which the ``all`` cut is drawn.
        peaks: The peak rows, for the mark.
        sources: The profiles present.
        bands: The declared bands.
        axis: The lag axis.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(max(len(sources), 1), height_per_row=2.4)
    frame = pd.DataFrame(stratified)
    for row, source in enumerate(sources):
        ax = axes[row, 0]
        cell = (
            frame[(frame["group_column"] == "all") & (frame["profile"] == source.key)]
            .sort_values("lag_step")
            if not frame.empty else frame
        )
        if cell.empty or not np.isfinite(cell["median"].to_numpy(dtype=np.float64)).any():
            ax.set_title(f"{source.label}, pooled over recordings")
            ax.text(
                0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes, ha="center", va="center",
                fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY, fontstyle="italic",
            )
            figures.style_axes(ax)
            continue
        colour = figures.COLOR_BLUE if not source.signed else figures.COLOR_VERMILLION
        ax.fill_between(
            axis.seconds, cell["q25"].to_numpy(), cell["q75"].to_numpy(),
            color=colour, alpha=0.18, linewidth=0, label="inter-quartile range over recordings",
        )
        ax.plot(
            axis.seconds, cell["median"].to_numpy(), color=colour,
            linewidth=figures.LINE_REGULAR, label="median over recordings",
        )
        ax.plot(
            axis.seconds, cell["mean"].to_numpy(), color=figures.COLOR_BLACK,
            linewidth=figures.LINE_THIN, linestyle="--", label="mean over recordings",
        )
        if source.signed:
            ax.axhline(0.0, color=figures.COLOR_GRAY, linestyle=":", linewidth=figures.LINE_THIN)
        peak = next(
            (
                record for record in peaks
                if record["group_column"] == "all" and record["profile"] == source.key
            ),
            None,
        )
        if peak is not None and peak.get("argmax_seconds") is not None:
            if peak.get("degenerate"):
                # As a legend entry rather than free text, so it sits with the other reading
                # aids instead of on top of the band names or the legend itself.
                ax.plot(
                    [], [], linestyle="none",
                    label="peak degenerate: no position read",
                )
            else:
                ax.axvline(
                    float(peak["argmax_seconds"]), color=figures.COLOR_GRAY, linestyle="--",
                    linewidth=figures.LINE_THIN,
                    label=f"peak at {float(peak['argmax_seconds']):g} s "
                          f"(share {float(peak['peak_share']):.2f})",
                )
        lag_structure.shade_bands(ax, bands, axis.seconds)
        ax.set_title(
            f"{source.label}, pooled over {int(cell['n_recordings'].iloc[0])} recordings"
        )
        ax.set_ylabel(source.unit)
        figures.legend_with_headroom(ax, ncol=2)
        figures.style_axes(ax)
    axes[-1, 0].set_xlabel(figures.COEFFICIENT_LAG_AXIS_LABEL)
    figures.caveat_note(figure, lag_structure.QUALIFICATION)
    return figure


def build_stratified_figure(
    stratified: Sequence[Mapping[str, Any]],
    sources: Sequence[lag_structure.ProfileSource],
    bands: Mapping[str, tuple],
    axis: lag_structure.LagAxis,
) -> Any:
    """The normalised profile of every cohort, one row per readout and one column per axis.

    Shares rather than magnitudes, so a cohort with a larger source pathway does not sit above
    the others for that reason alone; the magnitude is on the per-recording table and its
    grouped pages.

    Args:
        stratified: The stratified rows.
        sources: The profiles present.
        bands: The declared bands.
        axis: The lag axis.

    Returns:
        The figure; the caller renders and closes it.
    """
    axes_columns = list(labels.GROUP_COLUMNS)
    figure, axes = figures.new_figure(
        max(len(sources), 1), len(axes_columns), height_per_row=2.4
    )
    frame = pd.DataFrame(stratified)
    for row, source in enumerate(sources):
        for column, group_column in enumerate(axes_columns):
            ax = axes[row, column]
            cell = (
                frame[(frame["group_column"] == group_column) & (frame["profile"] == source.key)]
                if not frame.empty else frame
            )
            groups = (
                cohort.ordered_groups(sorted(set(cell["group"].astype(str))), group_column)
                if not cell.empty else []
            )
            ax.set_title(f"{source.label}: share by lag, by {group_column}")
            if not groups:
                ax.text(
                    0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes, ha="center",
                    va="center", fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY,
                    fontstyle="italic",
                )
                figures.style_axes(ax)
                continue
            colours = figures.group_colors(groups)
            for group in groups:
                part = cell[cell["group"].astype(str) == group].sort_values("lag_step")
                ax.plot(
                    axis.seconds, part["share"].to_numpy(), color=colours.get(group),
                    linewidth=figures.LINE_REGULAR,
                    label=f"{group} (n={int(part['n_recordings'].iloc[0])})",
                )
            lag_structure.shade_bands(ax, bands, axis.seconds)
            ax.set_ylabel("share of the rectified profile")
            if row == len(sources) - 1:
                ax.set_xlabel(figures.COEFFICIENT_LAG_AXIS_LABEL)
            figures.legend_with_headroom(ax, ncol=min(len(groups), 4))
            figures.style_axes(ax)
    figures.caveat_note(figure, lag_structure.QUALIFICATION)
    return figure


def _skip(reason: str) -> Dict[str, Any]:
    """The recorded skip, logged.

    Args:
        reason: Why nothing was written.

    Returns:
        The protocol's keys with ``n_samples`` ``None``.
    """
    logger.warning(f"{ANALYSIS_DIRNAME}: skipped -- {reason}")
    return {
        "n_samples": None, "composition": {}, "plan": {"capped": False},
        "skipped": True, "reason": reason, "files": [],
    }


def run_proposal_profile_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Shape both per-lag profiles per segment, pool them by cohort, and draw them.

    Args:
        context: The analysis context, read for the per-sample table, the vector sidecar and the
            results block.
        eval_config: The validated block. Unused: nothing here is capped or configurable.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys, the reducer's census per profile, the guarded peaks of the pooled
        profiles, the grouped-frame declaration and the files written. A recorded skip on an
        arm with no lag profile.
    """
    del eval_config, probe
    collection = context.collection
    results = dict(getattr(collection, "results", None) or {})
    table, record = lag_structure.segment_table(collection, results)
    sources = lag_structure.present_sources(record)
    axis = lag_structure.lag_axis_of(results)
    if table.empty or not sources or axis is None:
        return _skip(str(record.get("reason") or "no lag profile was collected"))
    bands = lag_structure.declared_bands(results)
    matrices = {
        source.key: lag_structure.profile_matrix(collection, source) for source in sources
    }

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(directory / SEGMENTS_FILENAME, index=False)

    columns = grouped_columns(sources)
    per_guid = lag_structure.per_recording_table(table, columns)
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)

    stratified = stratified_rows(table, matrices, axis)
    pd.DataFrame(stratified).to_csv(directory / STRATIFIED_FILENAME, index=False)
    peaks = peak_rows(stratified, axis)
    pd.DataFrame(peaks).to_csv(directory / PEAKS_FILENAME, index=False)

    written = [
        str(figures.render_figure(
            build_profile_figure(stratified, peaks, sources, bands, axis),
            directory / PROFILE_FIGURE,
        ).name),
        str(figures.render_figure(
            build_stratified_figure(stratified, sources, bands, axis),
            directory / STRATIFIED_FIGURE,
        ).name),
    ]
    logger.info(
        f"{ANALYSIS_DIRNAME}: {len(table)} segment(s), {len(per_guid)} recording(s), "
        f"{len(sources)} profile(s) over {axis.n_lags} lag(s)"
    )
    return {
        "n_samples": scored_sample_count(table, columns[0]) if columns else int(len(table)),
        "composition": {"n_recordings": int(len(per_guid)), "n_lags": axis.n_lags},
        "plan": {"capped": False},
        "profiles": {
            source.key: {
                "label": source.label, "unit": source.unit, "signed": source.signed,
                **(record["sources"].get(source.key) or {}),
            }
            for source in sources
        },
        "peaks": [row for row in peaks if row["group_column"] == "all"],
        "qualification": lag_structure.QUALIFICATION,
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, columns),
        ],
        "files": [
            SEGMENTS_FILENAME, PER_RECORDING_FILENAME, STRATIFIED_FILENAME, PEAKS_FILENAME,
            *written,
        ],
    }
