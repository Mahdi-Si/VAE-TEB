r"""The shape and the magnitude of the lag profiles, followed through labour on both clocks.

This cell's reading of the question the lag-attentive cells ask through ``lag_clocks`` and
``lag_kld_scaled``: does the location, the spread or the concentration of the lag structure move
as delivery approaches, does the magnitude in each declared band move, and does either differ by
class. The profiles are this architecture's own -- the proposal norm and the signed divergence
drop -- and their shape is reduced by the family's own statistics, so a centroid here is computed
as a centroid there, of a different readout and under a different reading.

**Per recording inside a window, and tested per window.** A window's value for a recording is
the mean over its segments in the window; a Kruskal-Wallis across the classes runs per window,
Holm corrects across the windows of one clock and one readout, and pairwise Mann-Whitney with
Cliff's delta runs on the survivors only. **Only the two centroids are tested** -- one per
profile, of the rectified divergence drop for the signed one -- which keeps the families at two
per clock; every other statistic and every band mass is drawn and tabled but carries no
$p$-value, and the record says so.

Three tables and three figures per clock: the per-recording rows, the trajectory cells, the
significance and pairwise tables; ``proposal_<clock>`` (the share of each class's profile by lag
and window, and the tested centroids), ``proposal_<clock>_windows`` (the distributions behind the
tested cells, their corrected significance and the effects), and ``proposal_<clock>_bands``
(each declared band's mass of each profile, per class, in the profile's own unit).
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
from teb_vae.lag_attn_cfs.eval.frames import scored_sample_count
from teb_vae.lag_slot_transformer_cfs.eval import lag_structure
from teb_vae.lag_slot_transformer_cfs.eval.figures import BAND_COLORS

#: Where this analysis writes, inside the results directory.
ANALYSIS_DIRNAME = "proposal_clocks"

#: What it writes.
PER_RECORDING_FILENAME = "proposal_clocks_per_recording.csv"
TRAJECTORY_FILENAME = "proposal_clocks_trajectory.csv"
SIGNIFICANCE_FILENAME = "proposal_clocks_significance.csv"
PAIRWISE_FILENAME = "proposal_clocks_pairwise.csv"

#: The figure stems per clock.
PROFILE_STEM = "proposal"
WINDOWS_SUFFIX = "windows"
BANDS_SUFFIX = "bands"

#: The statistics followed on the clocks, per profile. The tested one first.
TRACKED_STATISTICS: tuple = (
    "centroid", "spread", "median", "entropy", "effective_support", "peak_mass", "total_nats",
)

#: The one statistic per profile that is tested. One rather than seven, so each clock carries
#: two Holm families and not fourteen; a trajectory of an untested statistic that looks
#: separated is a hypothesis, not a claim.
TESTED_STATISTIC = "centroid"


def figure_stem(clock: lag_structure.Clock, suffix: str = "") -> str:
    """The stem one clock's figure is written under.

    Args:
        clock: The clock.
        suffix: ``''``, :data:`WINDOWS_SUFFIX` or :data:`BANDS_SUFFIX`.

    Returns:
        ``proposal_<clock>`` or ``proposal_<clock>_<suffix>``.
    """
    stem = f"{PROFILE_STEM}_{clock.name}"
    return f"{stem}_{suffix}" if suffix else stem


def tracked_columns(
    sources: Sequence[lag_structure.ProfileSource], bands: Mapping[str, tuple]
) -> Dict[str, List[str]]:
    """The columns followed on the clocks: the statistics per profile and the band masses.

    Args:
        sources: The profiles present.
        bands: The declared bands.

    Returns:
        ``{'statistics': [...], 'bands': [...], 'tested': [...]}``.
    """
    statistics = [
        lag_structure.statistic_column(statistic, source)
        for source in sources for statistic in TRACKED_STATISTICS
    ]
    band_columns = [
        lag_structure.band_column(band, source) for source in sources for band in bands
    ]
    tested = [lag_structure.statistic_column(TESTED_STATISTIC, source) for source in sources]
    return {"statistics": statistics, "bands": band_columns, "tested": tested}


# =============================================================================
# The figures
# =============================================================================
def build_profile_figure(
    clock: lag_structure.Clock,
    shares: Mapping[str, Any],
    windows: Sequence[int],
    centres: Sequence[float],
    rows: Sequence[Mapping[str, Any]],
    tested: Sequence[str],
    axis: lag_structure.LagAxis,
    source: lag_structure.ProfileSource,
) -> Any:
    """Where the profile sits, window by window, per class, and how the centroids move.

    One heatmap per class -- lag down, clock across, colour the class's share of the profile in
    that lag -- on one shared colour scale, then one panel per tested centroid with the median
    over recordings and its inter-quartile ribbon.

    Args:
        clock: The clock.
        shares: Per class, the ``share`` field $(L, W)$ and the recording count.
        windows: The window indices, ascending.
        centres: Their centres in hours.
        rows: This clock's trajectory rows on the class axis.
        tested: The tested columns, one panel each.
        axis: The lag axis.
        source: The profile the heatmaps draw.

    Returns:
        The figure; the caller renders and closes it.
    """
    fields = list(shares.items())
    figure, axes = figures.new_figure(max(len(fields), 1) + len(tested), height_per_row=2.6)
    limit = max(
        (float(np.nanmax(field["share"])) for _group, field in fields
         if np.isfinite(field["share"]).any()),
        default=0.0,
    )
    half = lag_structure.BIN_HOURS / 2.0
    half_lag = float(axis.seconds[1] - axis.seconds[0]) / 2.0 if axis.n_lags > 1 else 0.5
    extent = (
        (float(centres[0]) - half, float(centres[-1]) + half,
         float(axis.seconds[0]) - half_lag, float(axis.seconds[-1]) + half_lag)
        if centres else None
    )
    for index, (group, field) in enumerate(fields):
        ax = axes[index, 0]
        figures.heatmap_with_colorbar(
            figure, ax, np.asarray(field["share"])[::-1],
            title=(
                f"{group}: share of the {source.label} by lag and window "
                f"(n={int(field['n_recordings'])} recordings)"
            ),
            ylabel=figures.COEFFICIENT_LAG_AXIS_LABEL,
            symmetric=False,
            vlimits=(0.0, limit) if limit > 0.0 else None,
            colorbar_label="share of the profile",
            extent=extent,
            interpolation="none",
        )
        if clock.inverted:
            ax.invert_xaxis()
        else:
            ax.axvline(0.0, color=figures.COLOR_LIGHT_GRAY, linestyle=":", linewidth=figures.LINE_THIN)
        ax.set_xlabel(clock.axis_label)
    if not fields:
        ax = axes[0, 0]
        ax.text(
            0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes, ha="center", va="center",
            fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY, fontstyle="italic",
        )
        figures.style_axes(ax)
    for offset, column in enumerate(tested):
        lag_structure.draw_trajectory_panel(
            axes[max(len(fields), 1) + offset, 0],
            [row for row in rows if row["metric"] == column],
            axis=labels.CLASS_COLUMN, clock=clock,
            title=f"{column}: median over recordings per window (tested)",
            ylabel="stored-coefficient seconds",
        )
    figures.caveat_note(figure, lag_structure.QUALIFICATION)
    return figure


def build_bands_figure(
    clock: lag_structure.Clock,
    rows: Sequence[Mapping[str, Any]],
    sources: Sequence[lag_structure.ProfileSource],
    bands: Mapping[str, tuple],
) -> Any:
    """Each declared band's mass of each profile, per class, against the clock.

    One column per profile and one row per band, in the profile's own unit -- the magnitude the
    share heatmaps divide out. A line moving between rows is the informative past moving; a line
    moving within a row is that band's magnitude changing.

    Args:
        clock: The clock.
        rows: This clock's trajectory rows on the class axis.
        sources: The profiles present.
        bands: The declared bands.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(
        max(len(bands), 1), max(len(sources), 1), height_per_row=2.2
    )
    for column, source in enumerate(sources):
        for row, band in enumerate(bands):
            name = lag_structure.band_column(band, source)
            lag_structure.draw_trajectory_panel(
                axes[row, column],
                [item for item in rows if item["metric"] == name],
                axis=labels.CLASS_COLUMN, clock=clock,
                title=f"{band}: {source.label} in the band",
                ylabel=source.unit, zero=source.signed,
            )
            axes[row, column].set_title(
                f"{band}: {source.label} in the band", color=BAND_COLORS[row % len(BAND_COLORS)]
            )
    if not bands or not sources:
        axes[0, 0].text(
            0.5, 0.5, figures.EMPTY_NOTE, transform=axes[0, 0].transAxes, ha="center",
            va="center", fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY, fontstyle="italic",
        )
        figures.style_axes(axes[0, 0])
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


def run_proposal_clocks_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Follow the shape and the band masses of both profiles through labour on both clocks.

    Args:
        context: The analysis context, read for the per-sample table, the vector sidecar and the
            results block.
        eval_config: The validated block, for ``max_hours_before_delivery``.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys, the population each clock admitted, the headline of every test and
        the files written. A recorded skip on an arm with no lag profile.
    """
    del probe
    collection = context.collection
    results = dict(getattr(collection, "results", None) or {})
    table, record = lag_structure.segment_table(collection, results)
    sources = lag_structure.present_sources(record)
    axis = lag_structure.lag_axis_of(results)
    if table.empty or not sources or axis is None:
        return _skip(str(record.get("reason") or "no lag profile was collected"))
    table = cohort.within_horizon(table, eval_config.get("max_hours_before_delivery"))
    if table.empty:
        return _skip("no segment lies within the configured horizon before delivery")
    bands = lag_structure.declared_bands(results)
    columns = tracked_columns(sources, bands)
    value_columns = [*columns["statistics"], *columns["bands"]]
    matrices = {
        source.key: lag_structure.profile_matrix(collection, source) for source in sources
    }

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    tall: List[pd.DataFrame] = []
    rows: List[Dict[str, Any]] = []
    tests: List[Dict[str, Any]] = []
    clocks: Dict[str, Any] = {}
    written: List[str] = []
    for clock in lag_structure.CLOCKS:
        binned, population = lag_structure.clock_rows(table, clock)
        per_recording = lag_structure.per_recording_by_axis(binned, value_columns, clock)
        clock_rows = lag_structure.trajectory_rows(per_recording, value_columns, clock)
        rows.extend(clock_rows)
        for group_axis, frame in per_recording.items():
            if len(frame):
                tall.append(frame.assign(clock=clock.name, group_column=group_axis))
        class_frame = per_recording.get(labels.CLASS_COLUMN, pd.DataFrame())
        class_rows = [row for row in clock_rows if row["group_column"] == labels.CLASS_COLUMN]
        records = [
            lag_structure.windowed_tests(class_frame, column, clock) for column in columns["tested"]
        ]
        tests.extend(records)
        windows, centres, shares = lag_structure.windowed_shares(
            binned, matrices[sources[0].key], clock
        )
        written.append(str(figures.render_figure(
            build_profile_figure(
                clock, shares, windows, centres, class_rows, columns["tested"], axis, sources[0]
            ),
            directory / figure_stem(clock),
        ).name))
        written.append(str(figures.render_figure(
            lag_structure.windows_page(
                class_frame,
                [(column, record, column) for column, record in zip(columns["tested"], records)],
                clock,
            ),
            directory / figure_stem(clock, WINDOWS_SUFFIX),
        ).name))
        written.append(str(figures.render_figure(
            build_bands_figure(clock, class_rows, sources, bands),
            directory / figure_stem(clock, BANDS_SUFFIX),
        ).name))
        clocks[clock.name] = {
            **population,
            "n_windows": int(binned[clock.bin_column].nunique()) if len(binned) else 0,
            "n_segments": int(len(binned)),
        }

    pd.DataFrame(rows).to_csv(directory / TRAJECTORY_FILENAME, index=False)
    (pd.concat(tall, ignore_index=True) if tall else pd.DataFrame()).to_csv(
        directory / PER_RECORDING_FILENAME, index=False
    )
    lag_structure.significance_frame(tests).to_csv(directory / SIGNIFICANCE_FILENAME, index=False)
    lag_structure.pairwise_frame(tests).to_csv(directory / PAIRWISE_FILENAME, index=False)
    logger.info(
        f"{ANALYSIS_DIRNAME}: {len(value_columns)} column(s) of {len(sources)} profile(s) on "
        f"{len(lag_structure.CLOCKS)} clock(s); {sum(int(t.get('n_significant_windows', 0)) for t in tests)} "
        f"significant window(s) after Holm"
    )
    return {
        "n_samples": scored_sample_count(table, columns["tested"][0]),
        "composition": clocks,
        "plan": {"capped": True, "reason": "the second-stage clock admits eligible recordings only"},
        "bin_width_hours": lag_structure.BIN_HOURS,
        "n_lags": axis.n_lags,
        "tracked": columns,
        "tested": columns["tested"],
        "untested_note": (
            "Every statistic but the centroid, and every band mass, is drawn and tabled without "
            "a p-value; a separated-looking trajectory there is a hypothesis."
        ),
        "significance": [lag_structure.summary_of(record) for record in tests],
        "qualification": lag_structure.QUALIFICATION,
        # Already cut by cohort; no grouped variants.
        "files": [
            TRAJECTORY_FILENAME, PER_RECORDING_FILENAME, SIGNIFICANCE_FILENAME,
            PAIRWISE_FILENAME, *written,
        ],
    }
