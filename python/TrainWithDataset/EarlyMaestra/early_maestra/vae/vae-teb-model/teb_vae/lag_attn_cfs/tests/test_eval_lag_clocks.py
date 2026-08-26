r"""The lag structure against the two clinical clocks: the arithmetic, the population, the pages.

Six things are pinned here, and each is a way this analysis could be wrong while looking right.

**The two scalars are a known answer.** A centre of mass and a spread are easy to write in a way
that is off by a bin, off by the compensated delay, or normalised over the wrong axis -- and every
such version still produces a smooth trajectory that a reader would believe. They are asserted
against profiles whose answer can be computed by hand.

**A profile that carries no evidence must not become a number.** A segment that scored no anchors
is all-``NaN`` and a negative bin cannot occur in a non-negative attribution; both yield ``NaN`` and
a count rather than a value averaged into a cohort.

**The two clocks differ in their sign and in their population, and that is the whole point.**
``epoch`` is negated and ``second_stage_onset`` is not, so a segment recorded before the onset lands
in a *negative* window; and a recording with no onset is dropped from the second clock while staying
on the first. Getting either wrong runs a trajectory backwards or over the wrong cohort with nothing
raising.

**The inference is the shared one, per family.** Four families -- two clocks times two tested
readouts -- corrected within themselves and never jointly, on a known-answer separation in both
directions.

**Every comparison reads less severe against worse**, because the sweep names each pair in the
order it receives the classes and this analysis hands them over in the canonical one.

**And the page a reader opens shows the cells the test used**: one heatmap per class on a shared
colour scale, the share summing to one within every window.
"""
from __future__ import annotations

import types
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn.eval import figures as shared_figures
from teb_vae.lag_attn_cfs.eval import cohort
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.analyses import lag_clocks as analysis
from teb_vae.lag_attn_cfs.eval.lag_axis import (
    COEFFICIENT_LAG_AXIS_LABEL,
    compensated_seconds_axis,
)
from teb_vae.lag_attn_cfs.eval.lag_shape import profile_statistics

#: One hour, in the seconds both clock fields are stored in.
_HOUR = 3600.0

#: A short lag axis: eleven bins at 4 s, so a hand-computed centroid is a small number.
_N_LAGS = 11

#: The three classes, and where each one's attribution sits in the far window. The gaps are wide
#: enough that a rank test on six recordings separates them, which is what makes the verdict
#: assertions a property of the fixture rather than a hope.
#:
#: Every centre is kept at least $3\sigma$ inside the lag axis **in both windows**, so no bump has a
#: truncated tail: a truncated tail pulls the centroid back toward the axis and the fixture would
#: then contain a smaller shift than the one it applied, which reads exactly like an arithmetic bug.
_FAR_CENTRE = {"healthy": 8.0, "acidosis": 6.0, "hie": 4.0}

#: How far the centre moves toward the anchor in the near window.
_SHIFT = 1.5


def _bump(centre: float, width: float = 0.6) -> np.ndarray:
    """A normalised bump over the lag axis, centred where the caller says."""
    lags = np.arange(_N_LAGS, dtype=np.float64)
    values = np.exp(-0.5 * ((lags - float(centre)) / width) ** 2)
    return values / values.sum()


def _collection(
    *,
    separated: bool = True,
    n_recordings: int = 6,
    classes: Tuple[str, ...] = ("hie", "acidosis", "healthy"),
    missing_onset: Tuple[str, ...] = (),
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Build a per-sample table and its vector sidecar.

    Args:
        separated: Whether the classes sit at different lags. ``False`` gives every class the same
            profile, which is the non-vacuity case for every verdict assertion below.
        n_recordings: Recordings per class.
        classes: The classes present, entered worst-first so no assertion below can be satisfied by
            an implementation that returns its input order.
        missing_onset: Recordings (by index within their class) whose second-stage offset is
            absent, as ``"<class>_<index>"``.

    Returns:
        ``(per_sample, vectors)`` -- two windows per recording on each clock, two segments per
        window, and a profile whose centre moves toward the anchor in the near window.
    """
    rows: List[Dict[str, Any]] = []
    attribution: List[np.ndarray] = []
    attention: List[np.ndarray] = []
    for name in classes:
        for recording in range(n_recordings):
            guid = f"{name}_{recording:02d}"
            for hours in (1.0, 3.0):
                for segment in range(2):
                    onset = np.nan if guid in missing_onset else (1.0 - hours) * _HOUR
                    rows.append(
                        {
                            "guid": guid,
                            "epoch": -hours * _HOUR,
                            cohort.SECOND_STAGE_COLUMN: onset,
                            labels.CLASS_COLUMN: name,
                            labels.SUBGROUP_COLUMN: f"{name}_no_cs",
                        }
                    )
                    base = _FAR_CENTRE[name] if separated else 5.0
                    centre = base - (0.0 if hours > 2.0 else _SHIFT) + 0.01 * segment
                    # The attribution is the attention times a per-recording KL, so the two
                    # profiles have the same shape and different totals -- which is the case a
                    # normalisation bug survives.
                    attention.append(_bump(centre))
                    attribution.append(_bump(centre) * (1.0 + recording))
    return (
        pd.DataFrame(rows),
        {
            "lag_profile_untruncated": np.asarray(attribution, dtype=np.float64),
            "attention_profile_support_corrected": np.asarray(attention, dtype=np.float64),
        },
    )


def _context(per_sample: pd.DataFrame, vectors: Dict[str, np.ndarray], *, n_lags: int = _N_LAGS,
             delay_steps: int = 0) -> Any:
    """An analysis context built by hand, with no model and no collection pass."""
    collection = types.SimpleNamespace(
        per_sample=per_sample,
        per_anchor=pd.DataFrame(),
        vectors=vectors,
        retained={},
        results={"lag": {"n_lags": int(n_lags), "delay_steps": int(delay_steps)}},
        record={},
    )
    return types.SimpleNamespace(collection=collection, config={}, task=None, loader=None)


def _run(context: Any, tmp_path: Any) -> Tuple[Dict[str, Any], Any]:
    """Run the analysis and return its record beside the directory it wrote."""
    record = analysis.run_lag_clocks_analysis(
        context, eval_config={}, output_dir=tmp_path, probe=None
    )
    return record, tmp_path / analysis.ANALYSIS_DIRNAME


# =================================================================================================
# The per-segment scalars
# =================================================================================================
def test_the_centroid_and_spread_are_the_hand_computed_answer() -> None:
    r"""Three profiles whose answers can be written down.

    All the mass in bin $3$ puts the centre at $\tau_3 = 12$ s with no spread. Half in bin $0$ and
    half in bin $10$ puts it at $20$ s with a spread of $20$ s, which is the case a version
    reporting the argmax or the median would get wrong. The delay enters as $4\delta$, so the same
    profile read at $\delta = 2$ answers $8$ s higher -- a version that forgot the compensation
    would answer the same number twice.
    """
    seconds = compensated_seconds_axis(_N_LAGS, 0)
    single = np.zeros(_N_LAGS)
    single[3] = 2.0
    split = np.zeros(_N_LAGS)
    split[0] = split[10] = 1.0

    statistics, record = profile_statistics(np.stack([single, split]), seconds)

    assert statistics["centroid"] == pytest.approx([12.0, 20.0])
    assert statistics["spread"] == pytest.approx([0.0, 20.0])
    assert (record["n_rows"], record["n_usable"]) == (2, 2)

    delayed, _ = profile_statistics(
        np.stack([single, split]), compensated_seconds_axis(_N_LAGS, 2)
    )
    assert delayed["centroid"] == pytest.approx([20.0, 28.0])


def test_a_profile_carrying_no_evidence_is_nan_and_counted() -> None:
    """An all-``NaN`` row is a segment that scored no anchors; a negative bin cannot occur in a
    non-negative attribution and is a defect. Neither may become a number a cohort mean absorbs,
    and both are counted rather than dropped silently."""
    seconds = compensated_seconds_axis(_N_LAGS, 0)
    good = _bump(4.0)
    empty = np.full(_N_LAGS, np.nan)
    negative = good.copy()
    negative[7] = -1.0

    statistics, record = profile_statistics(np.stack([good, empty, negative]), seconds)
    centroid, spread = statistics["centroid"], statistics["spread"]

    assert np.isfinite(centroid[0]) and np.isfinite(spread[0])
    assert not np.isfinite(centroid[1]) and not np.isfinite(centroid[2])
    assert (record["n_usable"], record["n_empty"], record["n_negative"]) == (1, 1, 1)


def test_a_profile_of_the_wrong_width_is_refused_rather_than_reshaped() -> None:
    """A vector that does not match the lag axis is a mis-assembled profile, and padding it into a
    plausible wrong answer is what this refuses."""
    statistics, record = profile_statistics(
        np.ones((3, _N_LAGS - 2)), compensated_seconds_axis(_N_LAGS, 0)
    )

    assert not np.isfinite(statistics["centroid"]).any()
    assert record["n_usable"] == 0 and "mis-assembled" in record["note"]


def test_the_features_are_nan_where_the_sidecar_has_no_profile() -> None:
    """A run directory collected before a readout existed is a partial input, not a broken one."""
    per_sample, _ = _collection(n_recordings=3)

    frame, record = analysis.add_feature_columns(
        per_sample, {}, compensated_seconds_axis(_N_LAGS, 0)
    )

    assert set(analysis.FEATURE_COLUMNS) <= set(frame.columns)
    assert not frame[list(analysis.FEATURE_COLUMNS)].notna().to_numpy().any()
    assert all("carries no" in entry["reason"] for entry in record.values())


# =================================================================================================
# The two clocks: their signs, and their populations
# =================================================================================================
def test_the_second_clock_is_signed_and_is_not_negated() -> None:
    """``epoch`` is stored as time *before* delivery and is negated; ``second_stage_onset`` is
    already signed and is not. Negating it as well would run every trajectory backwards through the
    second stage with nothing raising, so both coordinates are asserted on one segment."""
    per_sample, vectors = _collection(n_recordings=3)
    seconds = compensated_seconds_axis(_N_LAGS, 0)
    featured, _ = analysis.add_feature_columns(per_sample, vectors, seconds)
    delivery, second_stage = analysis.CLOCKS

    # The segment three hours before delivery is two hours *before* the onset in this fixture.
    delivery_rows, _ = analysis.clock_rows(delivery, featured)
    onset_rows, _ = analysis.clock_rows(second_stage, featured)
    far = delivery_rows[delivery_rows["epoch"] == -3.0 * _HOUR]
    same = onset_rows[onset_rows["epoch"] == -3.0 * _HOUR]

    assert float(far[delivery.center_column].iloc[0]) == pytest.approx(3.25)
    assert float(same[second_stage.center_column].iloc[0]) == pytest.approx(-1.75)


def test_a_recording_with_no_onset_is_dropped_from_the_second_clock_and_counted() -> None:
    """One rule drops a recording from that clock: it has no onset. It stays on the delivery clock,
    which is what makes the two populations different by design rather than by accident."""
    per_sample, vectors = _collection(n_recordings=4, missing_onset=("hie_00",))
    featured, _ = analysis.add_feature_columns(
        per_sample, vectors, compensated_seconds_axis(_N_LAGS, 0)
    )
    delivery, second_stage = analysis.CLOCKS

    delivery_rows, delivery_population = analysis.clock_rows(delivery, featured)
    onset_rows, onset_population = analysis.clock_rows(second_stage, featured)

    assert "hie_00" in set(delivery_rows["guid"])
    assert "hie_00" not in set(onset_rows["guid"])
    assert onset_population["n_dropped_no_onset"] == 1
    assert onset_population["n_eligible"] == 11
    # The delivery clock has no eligibility rule at all, so it reports none.
    assert "n_dropped_no_onset" not in delivery_population


def test_a_windows_value_for_a_recording_is_the_mean_over_its_own_segments() -> None:
    """The aggregation chain applied *inside* a window: without it a recording contributing eleven
    segments to a window would outvote one contributing two."""
    per_sample, vectors = _collection(n_recordings=3)
    featured, _ = analysis.add_feature_columns(
        per_sample, vectors, compensated_seconds_axis(_N_LAGS, 0)
    )
    clock = analysis.CLOCKS[0]
    binned, _ = analysis.clock_rows(clock, featured)

    frames = analysis.per_recording_frames(clock, binned)
    class_frame = frames[labels.CLASS_COLUMN]

    # Three classes, three recordings each, two windows: one row per (class, window, recording).
    assert len(class_frame) == 3 * 3 * 2
    assert set(labels.GROUP_COLUMNS) == set(frames)


# =================================================================================================
# The inference
# =================================================================================================
def test_separated_classes_are_significant_in_the_windows_they_are_separated_in(tmp_path) -> None:
    """A known answer, and the shape of the whole analysis: four families, every window tested,
    every window separated."""
    per_sample, vectors = _collection()

    record, directory = _run(_context(per_sample, vectors), tmp_path)

    # The whole population reached a clock: eighteen recordings, four segments each.
    assert record["n_samples"] == 3 * 6 * 4
    families = record["significance"]
    assert len(families) == len(analysis.CLOCKS) * len(analysis.READOUTS) == 4
    for family in families:
        assert family["tested"] is True, family["metric_column"]
        assert family["n_windows_tested"] == 2
        assert family["n_significant_windows"] == 2
    assert sorted(path.name for path in directory.iterdir()) == [
        analysis.PAIRWISE_FILENAME,
        analysis.PER_RECORDING_FILENAME,
        analysis.PROFILE_FILENAME,
        analysis.SIGNIFICANCE_FILENAME,
        analysis.TRAJECTORY_FILENAME,
        "lag_second_stage.pdf",
        "lag_second_stage_features.pdf",
        "lag_second_stage_windows.pdf",
        "lag_time_to_delivery.pdf",
        "lag_time_to_delivery_features.pdf",
        "lag_time_to_delivery_windows.pdf",
    ]


def test_classes_that_do_not_differ_are_not_significant(tmp_path) -> None:
    """Non-vacuity for the assertion above: an implementation reporting significance
    unconditionally passes that one and fails this."""
    per_sample, vectors = _collection(separated=False)

    record, _ = _run(_context(per_sample, vectors), tmp_path)

    for family in record["significance"]:
        assert family["n_significant_windows"] == 0, family["metric_column"]


def test_the_centroid_moves_toward_the_anchor_as_the_landmark_approaches(tmp_path) -> None:
    r"""The reading the analysis exists to support, on a fixture built to contain it: the profile
    sits $1.5$ bins -- $6$ s -- nearer the anchor in the near window, on both clocks and for every
    class."""
    per_sample, vectors = _collection()

    _, directory = _run(_context(per_sample, vectors), tmp_path)
    trajectory = pd.read_csv(directory / analysis.TRAJECTORY_FILENAME)

    rows = trajectory[
        (trajectory["metric"] == "lag_centroid_kl_s")
        & (trajectory["group_column"] == labels.CLASS_COLUMN)
    ]
    for clock in analysis.CLOCKS:
        for group in ("healthy", "acidosis", "hie"):
            cell = rows[(rows["clock"] == clock.name) & (rows["group"] == group)]
            cell = cell.sort_values("bin_center_h")
            values = list(cell["median"])
            assert len(values) == 2, (clock.name, group)
            # The near window is the later centre on the second-stage axis and the *earlier* one on
            # the delivery axis, which counts down toward the event.
            near, far = (values[0], values[1]) if clock.inverted else (values[1], values[0])
            assert far - near == pytest.approx(6.0, abs=0.5), (clock.name, group)


def test_every_comparison_runs_from_the_less_severe_class_to_the_worse_one(tmp_path) -> None:
    """The sweep names each pair in the order it receives the classes, and this analysis hands them
    over in the canonical one -- so a positive Cliff's delta means the less severe class's centroid
    sits further back in the past, on every pair of every window of every family."""
    per_sample, vectors = _collection()

    _, directory = _run(_context(per_sample, vectors), tmp_path)
    pairs = pd.read_csv(directory / analysis.PAIRWISE_FILENAME)

    assert len(pairs)
    for _, cell in pairs.groupby(["clock", "metric_column", "time_bin"]):
        assert list(zip(cell["left"], cell["right"])) == [
            ("healthy", "acidosis"), ("healthy", "hie"), ("acidosis", "hie"),
        ]
    # Healthy sits furthest back in this fixture, so every delta is positive: one sign convention
    # across every pair rather than one per pair depending on how the names sorted.
    assert (pairs["cliffs_delta"] > 0).all()


def test_each_clock_and_readout_is_its_own_holm_family(tmp_path) -> None:
    """Four families, corrected within themselves. A joint correction would divide by the total
    number of windows across both clocks, so the family size each window records is what says which
    correction it actually received."""
    per_sample, vectors = _collection()

    record, _ = _run(_context(per_sample, vectors), tmp_path)

    keys = {(family["clock"], family["metric_column"]) for family in record["significance"]}
    assert len(keys) == 4
    for family in record["significance"]:
        assert "NOT corrected jointly" in family["method"]


# =================================================================================================
# What the record and the page carry
# =================================================================================================
def test_the_record_carries_the_axis_caveat_and_guards_the_peak(tmp_path) -> None:
    """The caveat travels in ``summary.json`` as well as under the figures, because the summary is
    the artifact that gets quoted.

    And the peak this analysis *does* report cannot reach a table without its guard beside it.
    ``entmax15`` assigns lags exactly zero, so a flat profile still has a perfectly confident
    argmax; the mechanical criterion that says whether the position means anything is a column on
    the same row, and the thresholds it was judged against are in the record. Asserted structurally
    rather than as wording: a peak column emitted without its degeneracy column is the failure, and
    it would read as an ordinary trajectory."""
    per_sample, vectors = _collection(n_recordings=3)

    record, directory = _run(_context(per_sample, vectors), tmp_path)

    assert "stored-coefficient time" in record["axis_caveat"]
    assert "lag_peak_degenerate" in record["peak_reference"]
    # The pooled positional reading still belongs to lag_kl, and the record still says so.
    assert "lag_kl_stratified_peaks.csv" in record["peak_reference"]
    assert record["plan"]["capped"] is True
    thresholds = record["statistic_thresholds"]
    assert thresholds["degenerate_peak_to_median"] == analysis.DEGENERATE_PEAK_TO_MEDIAN
    assert thresholds["degenerate_zero_fraction"] == analysis.DEGENERATE_ZERO_FRACTION

    for source in analysis.PROFILE_SOURCES:
        columns = set(pd.read_csv(directory / analysis.PER_RECORDING_FILENAME).columns)
        assert f"lag_peak_{source.key}_s" in columns
        assert f"lag_peak_degenerate_{source.key}" in columns
    metrics = set(pd.read_csv(directory / analysis.TRAJECTORY_FILENAME)["metric"])
    for source in analysis.PROFILE_SOURCES:
        assert f"lag_peak_{source.key}_s" in metrics
        assert f"lag_peak_degenerate_{source.key}" in metrics


def test_no_untested_statistic_reaches_the_significance_tables(tmp_path) -> None:
    """Twelve of the fourteen are drawn and tabled but carry no $p$-value, and that is what keeps
    each clock's Holm family at two. A statistic that quietly entered the inference would multiply
    the families without anything on the page saying the correction had changed."""
    per_sample, vectors = _collection(n_recordings=3)

    _, directory = _run(_context(per_sample, vectors), tmp_path)

    tested = {feature.column for feature in analysis.READOUTS}
    assert tested == {"lag_centroid_kl_s", "lag_centroid_attn_s"}
    for name in (analysis.SIGNIFICANCE_FILENAME, analysis.PAIRWISE_FILENAME):
        frame = pd.read_csv(directory / name)
        assert set(frame["metric_column"]) <= tested, name


def test_the_share_field_is_a_distribution_over_lags_in_every_window() -> None:
    """The heatmap answers *where* the attribution sits rather than how much of it there is, which
    is what ``time_to_delivery`` already draws. Normalising per window is what separates the two,
    and every class is laid out on the same window axis so the panels can be read against each
    other column by column."""
    per_sample, vectors = _collection()
    seconds = compensated_seconds_axis(_N_LAGS, 0)
    featured, _ = analysis.add_feature_columns(per_sample, vectors, seconds)
    clock = analysis.CLOCKS[0]
    binned, _ = analysis.clock_rows(clock, featured)

    windows, centres, fields = analysis.window_profiles(
        clock, binned, vectors["lag_profile_untruncated"], _N_LAGS
    )

    assert len(windows) == len(centres) == 2
    assert [field.group for field in fields] == ["healthy", "acidosis", "hie"]
    for field in fields:
        assert field.share.shape == (_N_LAGS, len(windows))
        assert np.nansum(field.share, axis=0) == pytest.approx([1.0, 1.0])
        # The mean is the un-normalised quantity and is emitted beside the share: a recording with
        # a large total attribution must not be able to move the share field.
        assert np.nansum(field.mean, axis=0).max() > 1.0


def test_the_profile_page_draws_one_panel_per_class_on_one_colour_scale() -> None:
    """Three panels each scaled to its own extremes would paint the same colour for three different
    shares while every colourbar stayed correct, so the scale is asserted rather than eyeballed."""
    per_sample, vectors = _collection()
    seconds = compensated_seconds_axis(_N_LAGS, 0)
    featured, _ = analysis.add_feature_columns(per_sample, vectors, seconds)
    clock = analysis.CLOCKS[0]
    binned, _ = analysis.clock_rows(clock, featured)
    frames = analysis.per_recording_frames(clock, binned)
    windows, centres, fields = analysis.window_profiles(
        clock, binned, vectors["lag_profile_untruncated"], _N_LAGS
    )

    figure = analysis.build_profile_figure(
        clock, windows, centres, fields, analysis.trajectory_rows(clock, frames), seconds
    )
    try:
        images = [image for axis in figure.axes for image in axis.get_images()]
        limits = {image.get_clim() for image in images}
    finally:
        shared_figures.plt.close(figure)

    assert len(images) == len(fields) == 3
    assert len(limits) == 1


def test_the_features_page_draws_every_untested_statistic_with_both_profiles() -> None:
    """One panel per drawn statistic, and on each of them the attribution *and* the attention.

    Pairing the two profiles on one axis is the analysis's central argument -- the attribution is
    $K_t$ times the attention and inherits an inflation the attention is immune to, so a statistic
    that moves in one and not the other is a finding about which is being read, and it is only
    visible when the two share an axis. A page drawing one of them would look entirely ordinary.
    """
    per_sample, vectors = _collection()
    seconds = compensated_seconds_axis(_N_LAGS, 0)
    featured, _ = analysis.add_feature_columns(per_sample, vectors, seconds)
    clock = analysis.CLOCKS[0]
    binned, _ = analysis.clock_rows(clock, featured)
    rows = analysis.trajectory_rows(clock, analysis.per_recording_frames(clock, binned))

    figure = analysis.build_features_figure(clock, rows)
    try:
        panels = [axis for axis in figure.axes if axis.get_title()]
        styles = {
            axis.get_title(): {line.get_linestyle() for line in axis.get_lines()}
            for axis in panels
        }
        labelled = {axis.get_title(): axis.get_ylabel() for axis in panels}
    finally:
        shared_figures.plt.close(figure)

    assert len(panels) == len(analysis.DRAWN_STATISTICS)
    for title, dashes in styles.items():
        assert "--" in dashes, title
    # Every panel names its own unit: half of what is drawn here is not in seconds, and a panel
    # inheriting the lag axis label would state the wrong one rather than none.
    for statistic in analysis.DRAWN_STATISTICS:
        title = f"{statistic.key} against the clock, by {labels.CLASS_COLUMN}"
        assert labelled[title] == (
            statistic.unit or COEFFICIENT_LAG_AXIS_LABEL
        ), title


def test_the_windows_page_rows_read_less_severe_against_worse() -> None:
    """The page an operator opens beside the CSV. Sorting the row labels would put
    ``acidosis vs hie`` above ``healthy vs acidosis`` while the violins above run healthy-first."""
    per_sample, vectors = _collection()
    seconds = compensated_seconds_axis(_N_LAGS, 0)
    featured, _ = analysis.add_feature_columns(per_sample, vectors, seconds)
    clock = analysis.CLOCKS[0]
    binned, _ = analysis.clock_rows(clock, featured)
    class_frame = analysis.per_recording_frames(clock, binned)[labels.CLASS_COLUMN]
    records = [
        analysis.analyse_windows(clock, class_frame, feature.column)
        for feature in analysis.READOUTS
    ]

    figure = analysis.build_windows_figure(clock, class_frame, records)
    try:
        drawn = [
            [text.get_text() for text in axis.get_yticklabels()]
            for axis in figure.axes
            if any(" vs " in text.get_text() for text in axis.get_yticklabels())
        ]
    finally:
        shared_figures.plt.close(figure)

    expected = [
        f"{feature.column}: {left} vs {right}"
        for feature in analysis.READOUTS
        for left, right in (("healthy", "acidosis"), ("healthy", "hie"), ("acidosis", "hie"))
    ]
    assert drawn == [expected]


# =================================================================================================
# The skip paths
# =================================================================================================
@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda frame: frame.iloc[:0], "empty"),
        (lambda frame: frame.assign(epoch=np.nan, second_stage_onset=np.nan), "neither clock"),
    ],
)
def test_a_table_that_cannot_be_placed_records_a_skip(mutate, expected, tmp_path) -> None:
    """A skip names its cause and reports ``n_samples`` ``None``: this analysis then scored no
    population, and a zero would enter the coverage block as a disagreement with every analysis
    that did."""
    per_sample, vectors = _collection(n_recordings=3)

    record, _ = _run(_context(mutate(per_sample), vectors), tmp_path)

    assert record["skipped"] is True
    assert record["n_samples"] is None
    assert expected in record["reason"]


def test_a_run_with_no_lag_geometry_records_a_skip(tmp_path) -> None:
    """The lag axis comes from the collection record. Without it there is no axis to resolve a
    profile against, and a default width would report an axis this run never had."""
    per_sample, vectors = _collection(n_recordings=3)
    context = _context(per_sample, vectors)
    context.collection.results = {}

    record, _ = _run(context, tmp_path)

    assert record["skipped"] is True and "lag geometry" in record["reason"]


def test_a_single_class_split_is_not_tested_and_says_why(tmp_path) -> None:
    """The ordinary outcome on the healthy-only pretraining split. The figures are still drawn --
    the trajectory of one class is a reading -- and every family records why it was not tested."""
    per_sample, vectors = _collection(classes=("healthy",))

    record, _ = _run(_context(per_sample, vectors), tmp_path)

    for family in record["significance"]:
        assert family["tested"] is False
        assert "fewer than two clinical classes" in family["reason"]
