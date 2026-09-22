r"""The band suppression margins and the source controls, followed through labour on both clocks.

This cell's reading of the question the lag-attentive cells' ``occlusion`` clock page asks: did
the informative past move, as an intervention on the fitted model rather than as a profile. Every
intervened arm's paired margin against the matched branch travels on the family's per-sample
table per segment; here those margins are binned on the two clinical clocks, per recording
inside a window, and drawn per clinical class.

**Descriptive only.** A margin in a half-hour window rests on the recordings that happened to
fall in it, and most (class, window) cells are below the minimum a rank test needs; a $p$-value
here would be a correction over cells that mostly could not be tested. The tests of the whole
population are on the arms page, and the tests of the lag *structure* on the clocks are in the
proposal-clocks analysis.

Two tables and two figures: ``band_clocks_per_recording.csv`` (one row per clock, cohort axis,
window and recording), ``band_clocks_trajectory.csv`` (the quartiles of every cell), and one
figure per clock with one panel per declared band and one for the controls.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from loguru import logger

from teb_vae.lag_attn_cfs.eval import cohort
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.frames import scored_sample_count
from teb_vae.lag_slot_transformer_cfs.eval import lag_structure

#: Where this analysis writes, inside the results directory.
ANALYSIS_DIRNAME = "band_clocks"

#: What it writes.
PER_RECORDING_FILENAME = "band_clocks_per_recording.csv"
TRAJECTORY_FILENAME = "band_clocks_trajectory.csv"

#: The figure stem per clock: ``band_margins_<clock>``.
FIGURE_STEM = "band_margins"

#: The per-sample column prefix of a band margin and the control margins drawn beside them, as
#: the collection pass names them. The prefix is a literal here rather than imported from the
#: collection module, which an analysis may not import; the smoke suite pins the two agree.
MARGIN_PREFIX = "margin_"
SUPPRESSION_MARGIN_PREFIX = "margin_suppress_"
CONTROL_MARGINS: tuple = (
    "margin_replace_zeros", "margin_replace_constant", "margin_permute",
)

#: The two reference identities, whose margins are zero and the gap by construction and are not
#: drawn as bands.
REFERENCE_BANDS: tuple = ("none", "all")


def figure_stem(clock: lag_structure.Clock) -> str:
    """The stem one clock's figure is written under.

    Args:
        clock: The clock.

    Returns:
        ``band_margins_<clock>``.
    """
    return f"{FIGURE_STEM}_{clock.name}"


def margin_columns(per_sample: pd.DataFrame) -> Dict[str, List[str]]:
    """The band and control margin columns the per-sample table carries.

    Args:
        per_sample: The family's per-sample table.

    Returns:
        ``{'bands': [...], 'controls': [...]}`` in the table's column order.
    """
    bands = [
        name for name in per_sample.columns
        if str(name).startswith(SUPPRESSION_MARGIN_PREFIX)
        and str(name)[len(SUPPRESSION_MARGIN_PREFIX):] not in REFERENCE_BANDS
    ]
    controls = [name for name in CONTROL_MARGINS if name in per_sample.columns]
    return {"bands": bands, "controls": controls}


def build_clock_figure(
    rows: Sequence[Dict[str, Any]],
    columns: Dict[str, List[str]],
    clock: lag_structure.Clock,
) -> Any:
    """One panel per declared band and one for the controls, per class, on one clock.

    Args:
        rows: This clock's trajectory rows on the class axis.
        columns: The band and control columns.
        clock: The clock.

    Returns:
        The figure; the caller renders and closes it.
    """
    panels = [(name, name[len(SUPPRESSION_MARGIN_PREFIX):]) for name in columns["bands"]]
    figure, axes = figures.new_figure(
        len(panels) + (1 if columns["controls"] else 0) or 1, height_per_row=2.4
    )
    for index, (column, band) in enumerate(panels):
        lag_structure.draw_trajectory_panel(
            axes[index, 0],
            [row for row in rows if row["metric"] == column and row["group"] != "all"],
            axis=labels.CLASS_COLUMN, clock=clock,
            title=f"suppress:{band} margin per window, by {labels.CLASS_COLUMN}",
            ylabel="paired margin (nats per anchor)", zero=True,
        )
    if columns["controls"]:
        ax = axes[len(panels), 0]
        drawn = 0
        for column, colour in zip(
            columns["controls"],
            (figures.COLOR_ORANGE, figures.COLOR_PURPLE, figures.COLOR_VERMILLION),
        ):
            cells = sorted(
                (row for row in rows if row["metric"] == column and row["group"] == "all"),
                key=lambda row: row["bin_center_h"],
            )
            if not cells:
                continue
            x = [row["bin_center_h"] for row in cells]
            ax.fill_between(
                x, [row["q25"] for row in cells], [row["q75"] for row in cells],
                color=colour, alpha=0.15, linewidth=0,
            )
            ax.plot(
                x, [row["median"] for row in cells], color=colour, marker="o",
                markersize=figures.MARKER_SMALL, linewidth=figures.LINE_REGULAR,
                label=column[len(MARGIN_PREFIX):],
            )
            drawn += 1
        ax.set_title("source control margins per window, all classes")
        if drawn:
            ax.axhline(0.0, color=figures.COLOR_GRAY, linestyle=":", linewidth=figures.LINE_THIN)
            if clock.inverted:
                ax.invert_xaxis()
            ax.set_xlabel(clock.axis_label)
            ax.set_ylabel("paired margin (nats per anchor)")
            ax.legend(fontsize=figures.FONT_SMALL, loc="best")
        else:
            ax.text(
                0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes, ha="center", va="center",
                fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY, fontstyle="italic",
            )
        figures.style_axes(ax)
    if not panels and not columns["controls"]:
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


def run_band_clocks_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Bin every band and control margin on both clinical clocks and draw them by class.

    Args:
        context: The analysis context, read for the per-sample table.
        eval_config: The validated block, for ``max_hours_before_delivery``.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys, the population each clock admitted, the columns drawn and the files
        written. A recorded skip when the table carries no margin column.
    """
    del probe
    per_sample = cohort.within_horizon(
        context.collection.per_sample, eval_config.get("max_hours_before_delivery")
    )
    columns = margin_columns(per_sample)
    value_columns = [*columns["bands"], *columns["controls"]]
    if per_sample.empty or not value_columns:
        return _skip(
            "the per-sample table carries no margin column: this arm has no source pathway, "
            "or the directory was collected before the margins were written per segment"
        )
    # All classes pooled as a cohort of its own, so the control panel has one trajectory per
    # control rather than one per class per control.
    with_all = per_sample.copy()
    with_all["all"] = "all"

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    tall: List[pd.DataFrame] = []
    rows: List[Dict[str, Any]] = []
    clocks: Dict[str, Any] = {}
    written: List[str] = []
    for clock in lag_structure.CLOCKS:
        binned, population = lag_structure.clock_rows(with_all, clock)
        per_recording = lag_structure.per_recording_by_axis(binned, value_columns, clock)
        per_recording["all"] = cohort.per_recording_in_bins(
            binned, value_columns, group_column="all",
            bin_column=clock.bin_column, center_column=clock.center_column,
        )
        clock_rows = lag_structure.trajectory_rows(per_recording, value_columns, clock)
        rows.extend(clock_rows)
        for axis, frame in per_recording.items():
            if len(frame):
                tall.append(frame.assign(clock=clock.name, group_column=axis))
        class_rows = [
            row for row in clock_rows
            if row["group_column"] in (labels.CLASS_COLUMN, "all")
        ]
        written.append(str(figures.render_figure(
            build_clock_figure(class_rows, columns, clock), directory / figure_stem(clock)
        ).name))
        clocks[clock.name] = {
            **population,
            "n_windows": int(binned[clock.bin_column].nunique()) if len(binned) else 0,
        }
    pd.DataFrame(rows).to_csv(directory / TRAJECTORY_FILENAME, index=False)
    (pd.concat(tall, ignore_index=True) if tall else pd.DataFrame()).to_csv(
        directory / PER_RECORDING_FILENAME, index=False
    )
    logger.info(
        f"{ANALYSIS_DIRNAME}: {len(columns['bands'])} band(s) and {len(columns['controls'])} "
        f"control(s) on {len(lag_structure.CLOCKS)} clock(s)"
    )
    return {
        "n_samples": scored_sample_count(per_sample, value_columns[0]),
        "composition": clocks,
        # Capped in the family's sense: the second clock scores the recordings that carry an
        # onset only, so its population is not the whole split's.
        "plan": {"capped": True, "reason": "the second-stage clock admits eligible recordings only"},
        "bin_width_hours": lag_structure.BIN_HOURS,
        "bands": [name[len(SUPPRESSION_MARGIN_PREFIX):] for name in columns["bands"]],
        "controls": [name[len(MARGIN_PREFIX):] for name in columns["controls"]],
        "descriptive_only": True,
        "note": (
            "Quartiles over recordings per window; no test is run on these cells. A positive "
            "margin means the fitted model predicts worse with the band's proposals removed."
        ),
        "qualification": lag_structure.QUALIFICATION,
        "files": [TRAJECTORY_FILENAME, PER_RECORDING_FILENAME, *written],
    }
