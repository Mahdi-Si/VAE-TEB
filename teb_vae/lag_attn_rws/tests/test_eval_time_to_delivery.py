r"""The coupling readouts against time before delivery, on recordings and on a fixed grid.

Four things are pinned, and each is a way this analysis could be wrong while looking right.

**The grid is not a setting.** The window width is a module constant and is absent from the
configuration schema, because an operator who could widen it could merge two windows until a
difference appeared or disappeared. Both halves are asserted mechanically rather than described.

**The binning is arithmetic with a sign in it.** ``epoch`` is negative before delivery, so hours
before delivery is $-\mathrm{epoch}/3600$; getting the sign wrong produces a trajectory running
backwards through labour with nothing raising.

**The unit is the recording, inside a window.** A window's value for a recording is the mean over
that recording's segments in it, so a recording contributing eleven segments to a window cannot
outvote one contributing two -- and the count reported per window is a count of recordings.

**Both readouts travel.** ``pred_gap`` and the unfloored KL fail differently, and an
implementation tracking only the KL would report a trajectory that is an artifact of the prior
variance sitting on its clamp.
"""
from __future__ import annotations

import types
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_rws.eval.figures_seam import figure_filename
from teb_vae.lag_attn_rws.eval import cohort
from teb_vae.lag_attn_rws.eval._reuse import labels
from teb_vae.lag_attn_rws.eval.analyses import time_to_delivery as analysis

#: One hour, in the seconds ``epoch`` is stored in.
_HOUR = 3600.0


def _per_sample(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """A per-sample table carrying only what this analysis reads."""
    frame = pd.DataFrame(rows)
    for column in analysis.VALUE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.full(len(frame), np.nan)
    return frame


def _context(per_sample: pd.DataFrame) -> Any:
    """An analysis context built by hand, with no model and no collection pass."""
    from teb_vae.lag_attn_rws.eval.analyses import AnalysisContext

    collection = types.SimpleNamespace(
        per_sample=per_sample, per_anchor=pd.DataFrame(), record={}, retained={}, results={},
        vectors={},
    )
    return AnalysisContext(collection=collection, config={})


def _cohort_rows(
    *, n_per_class: int = 4, segments: int = 2, offsets: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """Two classes, several recordings each, several segments per recording across two windows."""
    offsets = offsets if offsets is not None else [0.0, 5.0]
    rows: List[Dict[str, Any]] = []
    for class_index, (name, offset) in enumerate(zip(("healthy", "acidosis"), offsets)):
        for recording in range(n_per_class):
            for segment in range(segments):
                # Two windows: one an hour before delivery, one three hours before.
                hours = 1.0 if segment == 0 else 3.0
                rows.append(
                    {
                        "guid": f"{name}_{recording:02d}",
                        "epoch": -hours * _HOUR,
                        labels.CLASS_COLUMN: name,
                        labels.SUBGROUP_COLUMN: f"shard_{class_index}",
                        "mc_pred_gap": offset + float(recording) + 0.1 * segment,
                        "source_conditioned_kl_raw": offset + float(recording),
                    }
                )
    return rows


# =============================================================================
# The grid, and that it is not a setting
# =============================================================================
def test_the_bin_width_is_a_module_constant_and_not_a_config_key() -> None:
    """Both halves of the mechanical non-configurability assertion."""
    from teb_vae.lag_attn_rws.eval import config_schema

    assert analysis.TRAJECTORY_BIN_HOURS == pytest.approx(0.5)
    assert "trajectory_bin_hours" not in config_schema.VALID_KEYS
    # Bound from the layer below rather than restated, so the lag structure is cut on the same
    # windows this analysis reports.
    assert analysis.TRAJECTORY_BIN_HOURS is cohort.TRAJECTORY_BIN_HOURS


def test_a_negative_epoch_becomes_a_positive_time_before_delivery() -> None:
    """The sign. ``epoch`` is negative before delivery, so a wrong sign runs the trajectory
    backwards through labour with nothing raising."""
    frame = pd.DataFrame({"epoch": [-3600.0, -7200.0, -1800.0]})

    binned = cohort.add_time_bins(frame)

    assert list(binned[cohort.HOURS_COLUMN]) == [1.0, 2.0, 0.5]
    # 0.5 h windows: 1.0 h lands in bin 2, 2.0 h in bin 4, 0.5 h in bin 1.
    assert list(binned[cohort.BIN_COLUMN]) == [2, 4, 1]
    assert list(binned[cohort.BIN_CENTER_COLUMN]) == [1.25, 2.25, 0.75]


def test_a_non_finite_epoch_is_dropped_rather_than_binned() -> None:
    frame = pd.DataFrame({"epoch": [-3600.0, np.nan, float("inf")]})

    assert len(cohort.add_time_bins(frame)) == 1


# =============================================================================
# The unit inside a window
# =============================================================================
def test_a_windows_value_for_a_recording_is_the_mean_over_its_own_segments() -> None:
    """Three segments of one recording in one window count once, at their mean -- not three
    times, which is the pseudo-replication the whole chain exists to keep out."""
    frame = _per_sample(
        [
            {"guid": "a", "epoch": -3600.0, labels.CLASS_COLUMN: "healthy",
             "mc_pred_gap": value}
            for value in (1.0, 2.0, 6.0)
        ]
    )

    per_recording = cohort.per_recording_in_bins(
        cohort.add_time_bins(frame), ["mc_pred_gap"], group_column=labels.CLASS_COLUMN
    )

    assert len(per_recording) == 1
    assert float(per_recording["mc_pred_gap"].iloc[0]) == pytest.approx(3.0)


def test_the_reported_count_is_recordings_not_segments() -> None:
    frame = _per_sample(_cohort_rows(n_per_class=4, segments=3))

    rows = analysis.build_trajectory_rows(analysis.build_per_recording(frame))
    healthy = [
        row for row in rows
        if row["group"] == "healthy" and row["group_column"] == labels.CLASS_COLUMN
        and row["metric"] == "pred_gap_mc_nats"
    ]

    assert healthy, "the class axis produced no rows"
    assert {row["n_recordings"] for row in healthy} == {4}
    assert len(frame) == 24, "the fixture must hold more segments than recordings"


def test_a_segment_with_no_cohort_belongs_to_no_trajectory() -> None:
    """Folding the unlabelled segments together would create a cohort named after the absence."""
    frame = _per_sample(
        [
            {"guid": "a", "epoch": -3600.0, labels.CLASS_COLUMN: None, "mc_pred_gap": 1.0},
            {"guid": "b", "epoch": -3600.0, labels.CLASS_COLUMN: "healthy", "mc_pred_gap": 2.0},
        ]
    )

    per_recording = cohort.per_recording_in_bins(
        cohort.add_time_bins(frame), ["mc_pred_gap"], group_column=labels.CLASS_COLUMN
    )

    assert list(per_recording["group"]) == ["healthy"]


# =============================================================================
# Both readouts, and the tests over the windows
# =============================================================================
def test_both_readouts_produce_a_trajectory() -> None:
    """The sibling pipeline tracks only the KL; ``pred_gap`` is in the decoder's own units and is
    immune to the prior-variance inflation, so a trajectory in one and not the other is itself a
    finding about which readout is being believed."""
    rows = analysis.build_trajectory_rows(analysis.build_per_recording(_per_sample(_cohort_rows())))

    assert {row["metric"] for row in rows} == {
        "pred_gap_mc_nats", "source_conditioned_kl_raw_nats"
    }
    assert {row["group_column"] for row in rows} == set(labels.GROUP_COLUMNS)


def test_separated_classes_are_significant_in_the_windows_they_are_separated_in() -> None:
    """A known answer: two classes drawn five nats apart in every window must survive Holm."""
    frame = _per_sample(_cohort_rows(n_per_class=5, offsets=[0.0, 50.0]))
    per_recording = analysis.build_per_recording(frame)[labels.CLASS_COLUMN]

    record = analysis.analyse_windows(per_recording, "mc_pred_gap")

    assert record["tested"] is True
    assert record["n_windows_tested"] == 2
    assert record["n_significant_windows"] == 2
    assert set(record["pairwise"])


def test_overlapping_classes_are_not_significant() -> None:
    """Non-vacuity for the case above: an implementation reporting significance unconditionally
    passes it."""
    frame = _per_sample(_cohort_rows(n_per_class=5, offsets=[0.0, 0.0]))
    per_recording = analysis.build_per_recording(frame)[labels.CLASS_COLUMN]

    record = analysis.analyse_windows(per_recording, "mc_pred_gap")

    assert record["n_significant_windows"] == 0
    assert record["pairwise"] == {}


def test_the_holm_family_is_the_windows() -> None:
    frame = _per_sample(_cohort_rows(n_per_class=5, offsets=[0.0, 50.0]))
    per_recording = analysis.build_per_recording(frame)[labels.CLASS_COLUMN]

    record = analysis.analyse_windows(per_recording, "mc_pred_gap")

    for window in record["per_window"]:
        assert window["correction"] == "holm"
        assert window["n_windows_in_family"] == record["n_windows_tested"]
        assert window["p_holm"] >= window["p_value"]


def test_the_pooled_row_carries_its_confounded_flag_and_is_consumed_by_nothing() -> None:
    """The classes do not cover the time axis equally, so a pooled difference can be a coverage
    artifact. It is context, and the flag is what stops it being read as the result."""
    frame = _per_sample(_cohort_rows(n_per_class=5, offsets=[0.0, 50.0]))
    per_recording = analysis.build_per_recording(frame)[labels.CLASS_COLUMN]

    record = analysis.analyse_windows(per_recording, "mc_pred_gap")

    assert record["pooled"]["confounded_by_time"] is True
    assert "artifact" in record["pooled"]["note"]
    assert "significant" not in record["pooled"]


def test_the_pooled_row_counts_recordings_rather_than_recording_windows() -> None:
    """A recording spanning several windows must enter the pooled test once.

    The frame it is built from is keyed per (recording, window), so counting rows would
    pseudo-replicate the p-value by the windows-per-recording factor and report an ``n`` that is
    not a recording count. It also lets duplicated rows of one recording clear the
    ``MIN_GROUP_SIZE`` floor that exists to keep untestably small cohorts out.
    """
    frame = _per_sample(_cohort_rows(n_per_class=4, offsets=[0.0, 50.0]))
    per_recording = analysis.build_per_recording(frame)[labels.CLASS_COLUMN]

    record = analysis.analyse_windows(per_recording, "mc_pred_gap")

    # Each recording contributes two segments in two different windows, so the frame the pooled
    # test reads carries twice as many rows as there are recordings.
    assert len(per_recording) == 16
    assert record["pooled"]["n_per_group"] == {"healthy": 4, "acidosis": 4}


def test_the_pooled_row_refuses_a_cohort_too_small_to_test() -> None:
    """Non-vacuity for the count above: with two recordings per class the pooled test must be
    unable to run, exactly as every per-window test is. Counting (recording, window) rows instead
    would give it four per class and publish a p-value for a two-versus-two comparison."""
    frame = _per_sample(_cohort_rows(n_per_class=2, offsets=[0.0, 50.0]))
    per_recording = analysis.build_per_recording(frame)[labels.CLASS_COLUMN]

    record = analysis.analyse_windows(per_recording, "mc_pred_gap")

    assert record["n_windows_tested"] == 0, "the per-window tests correctly refuse"
    assert record["pooled"].get("n_groups", 0) < 2, (
        f"and so must the pooled row, got {record['pooled']}"
    )


def test_a_single_class_split_is_not_tested_and_says_why() -> None:
    """The ordinary outcome on the healthy-only pretraining split."""
    rows = [row for row in _cohort_rows() if row[labels.CLASS_COLUMN] == "healthy"]
    per_recording = analysis.build_per_recording(_per_sample(rows))[labels.CLASS_COLUMN]

    record = analysis.analyse_windows(per_recording, "mc_pred_gap")

    assert record["tested"] is False
    assert "fewer than two clinical classes" in record["reason"]


def test_a_window_with_too_few_recordings_records_the_exclusion() -> None:
    """"This class had two recordings in this window" is the explanation for a skipped window."""
    frame = _per_sample(_cohort_rows(n_per_class=2, offsets=[0.0, 50.0]))
    per_recording = analysis.build_per_recording(frame)[labels.CLASS_COLUMN]

    record = analysis.analyse_windows(per_recording, "mc_pred_gap")

    excluded = [window["groups_excluded_as_too_small"] for window in record["per_window"]]
    assert all(item == {"healthy": 2, "acidosis": 2} for item in excluded)
    assert record["n_windows_tested"] == 0


# =============================================================================
# What it writes, and the figure
# =============================================================================
def test_the_analysis_writes_its_tables_and_its_figure(tmp_path) -> None:
    result = analysis.run_time_to_delivery_analysis(
        _context(_per_sample(_cohort_rows(n_per_class=5, offsets=[0.0, 50.0]))),
        eval_config={}, output_dir=tmp_path, probe=None,
    )

    directory = tmp_path / analysis.ANALYSIS_DIRNAME
    for name in (
        analysis.TRAJECTORY_FILENAME, analysis.PER_RECORDING_FILENAME,
        analysis.SIGNIFICANCE_FILENAME, analysis.PAIRWISE_FILENAME,
        # Four figures, not two: each readout gets its trajectory and its windows page to itself.
        *(
            figure_filename(analysis.figure_stem(stem, readout))
            for readout in analysis.READOUTS
            for stem in (analysis.TRAJECTORY_FIGURE, analysis.WINDOWS_FIGURE)
        ),
    ):
        assert (directory / name).is_file(), name
        assert name in result["files"], name
    trajectory = pd.read_csv(directory / analysis.TRAJECTORY_FILENAME)
    assert set(trajectory["metric"]) == {"pred_gap_mc_nats", "source_conditioned_kl_raw_nats"}
    assert result["bin_width_hours"] == pytest.approx(0.5)
    assert [record["metric_column"] for record in result["significance"]] == list(
        analysis.VALUE_COLUMNS
    )
    # The summary carries the headline of each test, not the per-window detail, which is on disk.
    assert all("per_window" not in record for record in result["significance"])


def test_a_table_with_no_epoch_is_a_recorded_skip(tmp_path) -> None:
    frame = _per_sample([{"guid": "a", labels.CLASS_COLUMN: "healthy", "mc_pred_gap": 1.0}])

    result = analysis.run_time_to_delivery_analysis(
        _context(frame), eval_config={}, output_dir=tmp_path, probe=None
    )

    assert result["skipped"] is True
    assert "epoch" in result["reason"]
    assert result["n_samples"] is None
    assert not (tmp_path / analysis.ANALYSIS_DIRNAME).exists()


def test_the_figure_annotates_each_window_with_its_recording_count_in_the_cohort_colour() -> None:
    r"""The count is the point of the figure as much as the line: a window's median can move
    because the cohort changed rather than because the coupling did. The colours are asserted
    against the single mapping rather than eyeballed -- a figure whose classes are coloured by
    whatever order they arrived in cannot be compared with any other figure in the repository."""
    from teb_vae.lag_attn.eval import figures as shared_figures
    from teb_vae.lag_attn_rws.eval import figures_seam

    rows = analysis.build_trajectory_rows(
        analysis.build_per_recording(_per_sample(_cohort_rows(n_per_class=4)))
    )

    figure = analysis.build_trajectory_figure(rows, labels.CLASS_COLUMN, analysis.READOUTS[0])
    try:
        axis = figure.axes[0]
        annotations = sorted(text.get_text() for text in axis.texts)
        colours = {
            line.get_label().split(" ")[0]: line.get_color()
            for line in axis.lines
            if not line.get_label().startswith("_")
        }
    finally:
        shared_figures.plt.close(figure)

    # Four recordings in each of two windows, for each of two classes.
    assert annotations == ["4"] * 4
    assert colours == {
        name: colour
        for name, colour in figures_seam.group_colors(["healthy", "acidosis"]).items()
    }


def _window_rows(
    *,
    n_by_class=(("healthy", 5), ("acidosis", 5)),
    window_offsets=((1.0, 0.0), (3.0, 50.0)),
    segments: int = 1,
) -> List[Dict[str, Any]]:
    """Two classes over two windows, each window carrying its own separation between them.

    Args:
        n_by_class: ``(class, recordings)`` pairs. The first class is the reference; every other
            is shifted by the window's offset.
        window_offsets: ``(hours before delivery, separation)`` per window.
        segments: Segments each recording contributes to each window, so a test can tell a count
            of recordings from a count of segments.
    """
    rows: List[Dict[str, Any]] = []
    for index, (name, count) in enumerate(n_by_class):
        for recording in range(count):
            for hours, offset in window_offsets:
                for segment in range(segments):
                    value = float(recording) + (float(offset) if index else 0.0)
                    rows.append(
                        {
                            "guid": f"{name}_{recording:02d}",
                            "epoch": -hours * _HOUR,
                            labels.CLASS_COLUMN: name,
                            labels.SUBGROUP_COLUMN: f"{name}_cs",
                            "mc_pred_gap": value + 0.1 * segment,
                            "source_conditioned_kl_raw": value,
                        }
                    )
    return rows


def _windows_figure(rows: List[Dict[str, Any]]):
    """Build the first readout's windows page from a hand-made per-sample table, with the
    significance records of **both** readouts.

    One readout per page is the emitted layout, so the page under test is the one an operator
    opens; the second record travels beside it because several tests below assert on the record
    rather than on the drawing.
    """
    class_frame = analysis.build_per_recording(_per_sample(rows))[labels.CLASS_COLUMN]
    records = [analysis.analyse_windows(class_frame, column) for column in analysis.VALUE_COLUMNS]
    figure = analysis.build_windows_figure(class_frame, records[0], analysis.READOUTS[0])
    return figure, records


def _violin_centres(ax) -> List[float]:
    """The x coordinate of every violin body on one axes, ascending."""
    from matplotlib.collections import PolyCollection

    centres = []
    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            for path in collection.get_paths():
                x = np.asarray(path.vertices, dtype=np.float64)[:, 0]
                centres.append((float(x.min()) + float(x.max())) / 2.0)
    return sorted(centres)


def test_the_windows_page_stacks_a_strip_under_each_readouts_violins() -> None:
    """The layout is the content: the strip has to share the violins' x axis, or a reader is
    asked to carry a window's coordinate between two pages."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    figure, _ = _windows_figure(_window_rows())
    try:
        # One readout: a violin row, its strip, then the effect-size heatmap -- plus the colourbar
        # axes the heatmap attaches.
        assert len(figure.axes) == 4
        assert figure.axes[0].get_xlim() == pytest.approx(figure.axes[1].get_xlim())
        # Delivery at the right, on every panel that carries the clock.
        for index in range(2):
            low, high = figure.axes[index].get_xlim()
            assert low > high, f"panel {index} is not inverted"
    finally:
        shared_figures.plt.close(figure)


def test_the_strip_clears_alpha_in_the_window_the_classes_are_separated_in() -> None:
    """A known answer in both directions on one page: the same two classes, apart in one window
    and identical in the other. An implementation that drew significance unconditionally passes
    the first half and fails the second."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    figure, records = _windows_figure(_window_rows())
    try:
        strip = figure.axes[1]
        bars = sorted(
            [
                (float(patch.get_x() + patch.get_width() / 2.0), float(patch.get_height()))
                for patch in strip.patches
            ]
        )
        threshold = float(-np.log10(analysis.DEFAULT_ALPHA))

        assert [centre for centre, _ in bars] == pytest.approx([1.25, 3.25])
        # 1.25 h: the classes coincide exactly. 3.25 h: fifty nats apart.
        assert bars[0][1] < threshold
        assert bars[1][1] > threshold
    finally:
        shared_figures.plt.close(figure)

    assert records[0]["n_significant_windows"] == 1
    assert records[0]["significant_bin_centers_h"] == pytest.approx([3.25])


def test_the_violin_cells_are_drawn_in_the_canonical_order_and_the_cohort_colours() -> None:
    r"""Acidosis sits left of healthy inside every window -- the cohort axis runs worst first --
    and each is the colour this evaluation paints it everywhere else. A figure whose classes are
    placed or coloured by whatever order they arrived in cannot be compared with any other figure
    in the repository."""
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgba

    from teb_vae.lag_attn.eval import figures as shared_figures
    from teb_vae.lag_attn_rws.eval import figures_seam

    figure, _ = _windows_figure(_window_rows())
    try:
        violins = figure.axes[0]
        # Two windows of two classes: the dodge puts each class a third of a window either side.
        slot = analysis.TRAJECTORY_BIN_HOURS / 3.0
        assert _violin_centres(violins) == pytest.approx(
            sorted([1.25 - slot / 2, 1.25 + slot / 2, 3.25 - slot / 2, 3.25 + slot / 2])
        )
        faces = [
            tuple(collection.get_facecolor()[0][:3])
            for collection in violins.collections
            if isinstance(collection, PolyCollection)
        ]
        palette = figures_seam.group_colors(["acidosis", "healthy"])
        # Drawn cohort by cohort in the axis order, which runs worst first: the first two bodies
        # are acidosis's and the last two healthy's.
        assert faces[:2] == [to_rgba(palette["acidosis"])[:3]] * 2
        assert faces[2:] == [to_rgba(palette["healthy"])[:3]] * 2
    finally:
        shared_figures.plt.close(figure)


def test_a_cohort_below_the_test_floor_is_drawn_as_its_own_values_and_carries_no_verdict() -> None:
    """The two halves of one rule. A cell with two recordings is drawn -- a cohort thinning out is
    the explanation for the window beside it -- but drawn as points rather than as a density the
    smoother invented, and the window it sits in is marked untestable rather than tested."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    figure, records = _windows_figure(
        _window_rows(n_by_class=(("healthy", 5), ("acidosis", 2)))
    )
    try:
        violins, strip = figure.axes[0], figure.axes[1]
        bodies = _violin_centres(violins)
        points = [
            line for line in violins.lines
            if line.get_linestyle() == "None" and len(line.get_xdata())
        ]
        # Healthy has five recordings per window and gets a body; acidosis has two and does not.
        assert len(bodies) == 2
        assert sum(len(np.atleast_1d(line.get_xdata())) for line in points) == 4
        # And nothing was tested, so every window carries the untestable mark and no bar.
        assert len(strip.patches) == 0
        crosses = [line for line in strip.lines if line.get_marker() == "x"]
        assert sorted(np.atleast_1d(crosses[0].get_xdata())) == pytest.approx([1.25, 3.25])
    finally:
        shared_figures.plt.close(figure)

    assert records[0]["n_windows_tested"] == 0
    assert all(
        window["groups_excluded_as_too_small"] == {"acidosis": 2}
        for window in records[0]["per_window"]
    )


def test_the_count_on_each_cell_is_recordings_rather_than_segments() -> None:
    """The same rule the trajectory figure follows, and the reason the count is drawn at all: a
    cell's body can move because the cohort changed rather than because the coupling did."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    figure, _ = _windows_figure(_window_rows(segments=3))
    try:
        annotations = sorted(text.get_text() for text in figure.axes[0].texts)
    finally:
        shared_figures.plt.close(figure)

    # Two classes of five recordings in each of two windows -- never the fifteen segments behind.
    assert annotations == ["5"] * 4


def test_the_windows_page_of_a_single_class_split_draws_notes_rather_than_raising() -> None:
    """The ordinary outcome on the healthy-only pretraining split: nothing is tested, so there is
    nothing to draw, and the page has to say so rather than fail at the run's final step."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    rows = [row for row in _window_rows() if row[labels.CLASS_COLUMN] == "healthy"]
    figure, records = _windows_figure(rows)
    try:
        notes = [
            text.get_text() for axis in figure.axes for text in axis.texts
        ]
        assert notes.count(shared_figures.EMPTY_NOTE) >= 2
    finally:
        shared_figures.plt.close(figure)

    assert records[0]["tested"] is False


def test_a_single_cohort_population_draws_the_empty_note_rather_than_one_line() -> None:
    """One line invites a comparison there is nothing to compare against."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    figure = analysis.build_trajectory_figure([], labels.CLASS_COLUMN, analysis.READOUTS[0])
    try:
        notes = [text.get_text() for text in figure.axes[0].texts]
        n_lines = len(figure.axes[0].lines)
    finally:
        shared_figures.plt.close(figure)

    assert notes == [shared_figures.EMPTY_NOTE]
    assert n_lines == 0


# =============================================================================
# End to end, on a real run
# =============================================================================
def test_the_real_run_bins_the_generated_epochs(evaluated) -> None:
    """The fixture's segments span several hours before delivery, so a real run has more than one
    window to put them in -- which is what makes the binning load-bearing rather than decorative."""
    block = evaluated["summary"]["results"]["time_to_delivery"]
    trajectory = pd.read_csv(
        evaluated["results_dir"] / analysis.ANALYSIS_DIRNAME / analysis.TRAJECTORY_FILENAME
    )

    assert block.get("skipped") is not True
    assert block["composition"]["n_windows"] > 1
    assert block["bin_width_hours"] == pytest.approx(0.5)
    assert set(trajectory["group_column"]) == set(labels.GROUP_COLUMNS)
    # Every cell reports the recordings behind it, never the segments.
    assert (trajectory["n_recordings"] > 0).all()
