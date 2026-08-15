r"""The per-segment distributions: the one analysis that describes segments rather than recordings.

Every other analysis reduces to one value per recording before reporting anything, so this one is
the exception and the tests are written around exactly what makes it an exception:

* it draws **two levels on one axes**, and the arithmetic behind them is deliberately different --
  a rooted metric roots per segment on the left and after the per-recording mean on the right;
* it is **descriptive**, so it must declare no grouped frame and register no headline number, and
  both are asserted rather than assumed;
* it must survive the degenerate splits, because a single-cohort population is the ordinary case
  on the pretraining shards and an absent column is the ordinary case on an older run's tables.

**And one thing the sibling's copy of this file spends four tests on does not exist here.** There
it converts the rooted metrics into bpm and the assertions are about getting a *spread* conversion
right rather than a *level* one. A wavelet modulus has no clinical unit, so nothing here converts
at all: :func:`~teb_vae.lag_attn_cfs.eval.analyses.distributions.build_frames` takes no statistics
argument, every emitted unit is ``normalised``, and the tests below assert the absence rather than
the correctness of a conversion.

The happy path is asserted first and by hand-computed value, because every degenerate-input test
below would pass with the analysis unimplemented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection

from teb_vae.lag_attn.eval import figures as shared_figures
from teb_vae.lag_attn_cfs.eval import figures_seam
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.analyses import distributions
from teb_vae.lag_attn_cfs.eval.metrics import NORMALISED_UNIT

from .test_eval_grouped import GROUPED_SUFFIXES


def _frame(**overrides) -> pd.DataFrame:
    """A per-sample table with three classes, four subgroups and known values.

    ``healthy`` deliberately contributes more segments per recording than the other two, so the
    segment-level and recording-level series are genuinely different frames rather than the same
    numbers twice.
    """
    rows = (
        [("g_h1", "healthy", "healthy_bg_cs")] * 6
        + [("g_h2", "healthy", "healthy_no_bg_no_cs")] * 6
        + [("g_a1", "acidosis", "acidosis_cs")] * 2
        + [("g_i1", "hie", "hie_no_cs")] * 2
    )
    base = pd.DataFrame(
        {
            "guid": [row[0] for row in rows],
            "epoch": np.arange(len(rows), dtype=np.float64) * -600.0,
            labels.CLASS_COLUMN: [row[1] for row in rows],
            labels.SUBGROUP_COLUMN: [row[2] for row in rows],
            # The dense anchor count of this cell, not the raw cells' 240.
            "n_anchors": 152,
        }
    )
    for metric in distributions.METRICS:
        base[metric.column] = np.linspace(0.25, 4.0, len(rows))
    for name, values in overrides.items():
        base[name] = values
    return base


@pytest.fixture
def per_sample() -> pd.DataFrame:
    return _frame()


# =============================================================================
# The two levels, and the arithmetic that separates them
# =============================================================================
def test_a_rooted_metric_is_rooted_and_nothing_else(per_sample) -> None:
    r"""Rooting is the whole transform. The root of a mean square of $z$-units is $z$-units, so
    the unit does not change under it and the values are the plain roots of the source column --
    no scale, no offset, and no dependency on any statistics being available."""
    segment, _, units = distributions.build_frames(per_sample)

    squares = np.asarray(per_sample["sq_error_full"], dtype=np.float64)

    assert units["rmse_full"] == NORMALISED_UNIT
    assert np.allclose(np.asarray(segment["rmse_full"]), np.sqrt(squares))


def test_the_two_levels_root_at_different_points(per_sample) -> None:
    r"""The segment series roots each segment's own square; the recording series roots after the
    per-recording mean, as the rest of the pipeline does. By Jensen the first sits at or below the
    second, and the analysis carries that sentence rather than leaving the two to be compared as
    though they were one arithmetic."""
    segment, recording, _ = distributions.build_frames(per_sample)

    guid = "g_h1"
    squares = per_sample.loc[per_sample["guid"] == guid, "sq_error_full"]
    hand_rooted_once = np.sqrt(float(squares.mean()))
    mean_of_roots = float(
        np.mean(segment.loc[segment["guid"] == guid, "rmse_full"])
    )

    assert float(recording.loc[guid, "rmse_full"]) == pytest.approx(hand_rooted_once)
    # Strictly below, because this recording's squares are not all equal -- which is what makes
    # the assertion a check on the ordering rather than on a tie.
    assert mean_of_roots < hand_rooted_once
    assert "Jensen" in distributions.PER_SEGMENT_ROOT_NOTE


def test_an_unrooted_metric_passes_through_in_its_own_unit(per_sample) -> None:
    """Only the two error metrics are squares. A metric the collection pass already rooted --
    ``delta_mu_rms`` -- must not be rooted twice."""
    segment, _, units = distributions.build_frames(per_sample)

    assert units["delta_mu_rms"] == NORMALISED_UNIT
    assert np.allclose(
        np.asarray(segment["delta_mu_rms"]), np.asarray(per_sample["delta_mu_rms"])
    )
    assert units["mc_pred_gap"] == "nats per anchor"


# =============================================================================
# Nothing is converted, and nothing can be
# =============================================================================
def test_no_metric_is_drawn_in_a_clinical_unit(per_sample) -> None:
    """The inverted half of the sibling's file. Every declared unit is the loader's own scale or a
    nats/log axis, and the resolved units are exactly the declared ones -- there is no branch that
    could upgrade one, which is what "deleted rather than repointed" has to mean to hold."""
    _, _, units = distributions.build_frames(per_sample)

    assert units == {metric.name: metric.unit for metric in distributions.METRICS}
    assert all("bpm" not in unit for unit in units.values())
    assert set(units.values()) == {NORMALISED_UNIT, "nats per anchor", "log z-units", "nats"}


def test_the_frame_builder_takes_no_statistics_to_convert_with() -> None:
    """Asserted on the signature rather than on an output, because the failure this prevents is
    a *reintroduction*: a second argument here is a conversion waiting for a caller, and the two
    error metrics are the ones a mechanical port would convert."""
    import inspect

    parameters = list(inspect.signature(distributions.build_frames).parameters)

    assert parameters == ["per_sample"]


def test_the_readout_module_exports_nothing_this_analysis_could_convert_through() -> None:
    """Non-vacuity for the two above: they would both pass while a conversion sat one import
    away. ``tests/test_eval_units.py`` owns this claim in full; it is restated here because this
    analysis is the sibling's only caller of it."""
    from teb_vae.lag_attn_cfs.eval import metrics

    assert not hasattr(metrics, "sigma_to_bpm")
    assert not hasattr(metrics, "to_bpm")


# =============================================================================
# The summary table
# =============================================================================
def test_the_summary_carries_both_levels_for_every_cell(per_sample) -> None:
    segment, recording, units = distributions.build_frames(per_sample)
    table = pd.DataFrame(distributions.build_summary_rows(segment, recording, units))

    assert set(table["level"]) == set(distributions.LEVELS)
    # pooled + 3 classes + 4 subgroups, times 8 metrics, times 2 levels.
    assert len(table) == (1 + 3 + 4) * len(distributions.METRICS) * 2

    healthy_segments = table[
        (table["group"] == "healthy")
        & (table["metric"] == "rmse_full")
        & (table["level"] == distributions.SEGMENT_LEVEL)
    ].iloc[0]
    healthy_recordings = table[
        (table["group"] == "healthy")
        & (table["metric"] == "rmse_full")
        & (table["level"] == distributions.RECORDING_LEVEL)
    ].iloc[0]

    # The denominators are the whole point of drawing both: twelve segments, two recordings.
    assert int(healthy_segments["n"]) == 12
    assert int(healthy_recordings["n"]) == 2


def test_the_summary_is_written_in_the_canonical_cohort_order(per_sample) -> None:
    """Same rule as every other cohort-bearing table in this pipeline."""
    segment, recording, units = distributions.build_frames(per_sample)
    table = pd.DataFrame(distributions.build_summary_rows(segment, recording, units))

    classes = table[table["group_column"] == labels.CLASS_COLUMN]
    assert list(dict.fromkeys(classes["group"])) == ["healthy", "acidosis", "hie"]
    subgroups = table[table["group_column"] == labels.SUBGROUP_COLUMN]
    assert list(dict.fromkeys(subgroups["group"])) == [
        "healthy_no_bg_no_cs", "healthy_bg_cs", "acidosis_cs", "hie_no_cs",
    ]


def test_every_summary_row_names_the_unit_it_is_in(per_sample) -> None:
    """A pooled distribution over 98 channels in z units reads exactly like one over bpm unless
    something says which, and the CSV is what an offline reader has."""
    segment, recording, units = distributions.build_frames(per_sample)
    table = pd.DataFrame(distributions.build_summary_rows(segment, recording, units))

    assert table["unit"].notna().all()
    assert set(table.loc[table["metric"] == "rmse_full", "unit"]) == {NORMALISED_UNIT}


# =============================================================================
# The panel
# =============================================================================
def _panel(segment_by_group, recording_by_group, groups, **kwargs):
    """Draw one panel on a throwaway figure and return ``(drawn, axes, figure)``."""
    figure, axes = shared_figures.new_figure(1, 1)
    # The callers close the returned figure in their own ``finally``; they never get one if the
    # draw raises, so the failure path is closed here rather than left to pyplot's registry.
    try:
        drawn = distributions.draw_density_panel(
            axes[0, 0], segment_by_group, recording_by_group, groups,
            title=kwargs.pop("title", "t"), xlabel=kwargs.pop("xlabel", "x"), **kwargs
        )
    except BaseException:
        shared_figures.plt.close(figure)
        raise
    return drawn, axes[0, 0], figure


def test_the_cohorts_share_one_bin_grid() -> None:
    """Two histograms on two grids are not a comparison, and the difference between them can be
    the binning rather than the data. Asserted on disjoint ranges, where independent grids would
    be obvious and identical grids are the only way both curves span the same axis."""
    low = np.linspace(0.0, 1.0, 200)
    high = np.linspace(10.0, 11.0, 200)

    drawn, ax, figure = _panel(
        {"healthy": low, "acidosis": high}, {"healthy": low[:5], "acidosis": high[:5]},
        ["healthy", "acidosis"],
    )
    try:
        # A ``stepfilled`` histogram is one Polygon whose x vertices are its bin edges, so the two
        # cohorts sharing a grid is an equality between two edge sets rather than an inference.
        grids = [
            tuple(np.round(np.unique(patch.get_xy()[:, 0]), 9))
            for patch in ax.patches if patch.get_fill()
        ]
    finally:
        shared_figures.plt.close(figure)

    assert drawn == 2
    assert len(grids) == 2 and grids[0] == grids[1]
    # And the shared grid spans both cohorts, which two independent grids would not.
    assert min(grids[0]) == pytest.approx(0.0) and max(grids[0]) == pytest.approx(11.0)
    assert len(grids[0]) == distributions.HISTOGRAM_BINS + 1


def test_the_fill_is_translucent_and_its_border_is_not() -> None:
    """The one thing that makes overlapping cohorts readable, and the one way it regresses.

    The transparency is carried by the **face colour**, not by the artist's ``alpha``: an artist
    alpha fades the border along with the fill, which gives back the soft-edged blur the outline
    exists to replace. So the property asserted is not "it is transparent somewhere" but that the
    two are transparent *differently* -- a faint body inside an opaque hairline.
    """
    rng = np.random.default_rng(1)
    series = {"healthy": rng.normal(0.0, 1.0, 400), "acidosis": rng.normal(0.5, 1.0, 300)}

    _, ax, figure = _panel(
        series, {name: values[:6] for name, values in series.items()}, list(series)
    )
    try:
        filled = [patch for patch in ax.patches if patch.get_fill()]
        faces = [patch.get_facecolor() for patch in filled]
        edges = [patch.get_edgecolor() for patch in filled]
        artist_alphas = [patch.get_alpha() for patch in filled]
        widths = [float(patch.get_linewidth()) for patch in filled]
    finally:
        shared_figures.plt.close(figure)

    assert len(filled) == 2
    assert all(face[3] == pytest.approx(distributions.FILL_ALPHA) for face in faces)
    assert all(edge[3] == pytest.approx(1.0) for edge in edges)
    # An artist alpha would have been applied to the edge as well, and the assertion above would
    # still pass on the *stored* colour while the drawn border faded.
    assert all(alpha is None for alpha in artist_alphas)
    assert all(width == pytest.approx(figures_seam.LINE_HAIRLINE) for width in widths)


def test_every_outline_is_drawn_above_every_fill() -> None:
    """Which cohort is legible must not be decided by which was drawn last.

    Within one z-level matplotlib draws in call order, so a single pass per cohort would leave the
    first cohort's outline veiled by the fill of every cohort after it -- and the first entry in
    the legend would be the hardest curve to trace, which is an artefact of the draw order reading
    as a property of the data. The second pass is deliberately unlabelled, so the legend keeps one
    row per cohort rather than two.
    """
    rng = np.random.default_rng(2)
    series = {name: rng.normal(float(index), 1.0, 300) for index, name in enumerate(
        ("healthy", "acidosis", "hie")
    )}

    drawn, ax, figure = _panel(
        series, {name: values[:5] for name, values in series.items()}, list(series)
    )
    try:
        fills = [patch.get_zorder() for patch in ax.patches if patch.get_fill()]
        outlines = [patch.get_zorder() for patch in ax.patches if not patch.get_fill()]
        legend_rows = len(ax.get_legend().get_texts())
    finally:
        shared_figures.plt.close(figure)

    assert drawn == 3
    assert len(fills) == len(outlines) == 3
    assert min(outlines) > max(fills)
    assert legend_rows == 3


def test_the_panel_is_a_density_so_a_larger_cohort_is_not_a_taller_curve() -> None:
    """The healthy cohort contributes an order of magnitude more segments than HIE. On a count
    axis every panel would report that rather than the metric."""
    rng = np.random.default_rng(0)
    small = rng.normal(0.0, 1.0, 50)
    large = rng.normal(0.0, 1.0, 5000)

    drawn, ax, figure = _panel(
        {"healthy": large, "hie": small}, {"healthy": large[:9], "hie": small[:3]},
        ["healthy", "hie"],
    )
    try:
        top = float(ax.get_ylim()[1])
    finally:
        shared_figures.plt.close(figure)

    assert drawn == 2
    # A density of a unit normal peaks near 0.4; a count axis would reach into the hundreds.
    assert top < 5.0


def test_both_levels_reach_the_panel() -> None:
    """The filled histogram is the segments and the strip above it is the recordings. A panel that
    dropped either would look entirely healthy."""
    values = np.linspace(0.0, 1.0, 100)
    recordings = np.array([0.30, 0.40, 0.50, 0.60, 0.70])

    drawn, ax, figure = _panel({"healthy": values}, {"healthy": recordings}, ["healthy"])
    try:
        filled = [p for p in ax.patches if p.get_fill()]
        # The strip is drawn with ``hlines``, which is a LineCollection -- the median marker is a
        # Line2D and the histogram contributes neither.
        strips = [c for c in ax.collections if isinstance(c, LineCollection)]
        top = float(ax.get_ylim()[1])
        # A ``stepfilled`` histogram is one Polygon, so its tallest bar is the largest y vertex.
        density_top = max(float(patch.get_xy()[:, 1].max()) for patch in filled)
    finally:
        shared_figures.plt.close(figure)

    assert drawn == 1
    assert filled, "the segment-level histogram is missing"
    # Range line and inter-quartile bar.
    assert len(strips) == 2, "the recording-level strip is missing"
    # And the strip sits *above* the densities rather than over them, with legend headroom left.
    assert top > density_top * (1.0 + distributions.RECORDING_STRIP_FRACTION)


def test_the_strip_marks_the_median_quartiles_and_range() -> None:
    """The three things it exists to show, on values whose statistics are known by hand."""
    recordings = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    _, ax, figure = _panel(
        {"healthy": np.linspace(0.0, 6.0, 60)}, {"healthy": recordings}, ["healthy"]
    )
    try:
        spans = sorted(
            (float(segment[0][0]), float(segment[-1][0]))
            for collection in ax.collections if isinstance(collection, LineCollection)
            for segment in collection.get_segments()
        )
        medians = [
            float(line.get_xdata()[0]) for line in ax.lines
            if line.get_marker() == "o" and len(line.get_xdata()) == 1
        ]
    finally:
        shared_figures.plt.close(figure)

    assert spans == [(1.0, 5.0), (2.0, 4.0)]  # full range, then the inter-quartile bar
    assert medians == [3.0]


def test_a_single_recording_still_gets_a_median_without_a_quartile_bar() -> None:
    """``np.quantile`` of one value is that value, so an IQR bar would be a zero-length line
    pretending to be a spread. The median dot alone is the honest mark."""
    _, ax, figure = _panel(
        {"healthy": np.linspace(0.0, 1.0, 20)}, {"healthy": np.array([0.5])}, ["healthy"]
    )
    try:
        bars = [
            segment for collection in ax.collections
            if isinstance(collection, LineCollection) for segment in collection.get_segments()
        ]
        medians = [line for line in ax.lines if line.get_marker() == "o"]
    finally:
        shared_figures.plt.close(figure)

    assert len(bars) == 1 and len(medians) == 1


def test_a_cohort_with_no_finite_value_is_skipped_and_the_rest_are_drawn() -> None:
    drawn, _, figure = _panel(
        {"healthy": np.linspace(0.0, 1.0, 20), "hie": np.zeros(0)},
        {"healthy": np.zeros(3), "hie": np.zeros(0)},
        ["healthy", "hie"],
    )
    shared_figures.plt.close(figure)

    assert drawn == 1


def test_an_empty_panel_says_so_rather_than_drawing_an_empty_frame() -> None:
    drawn, ax, figure = _panel({}, {}, [])
    try:
        notes = [text.get_text() for text in ax.texts]
    finally:
        shared_figures.plt.close(figure)

    assert drawn == 0
    assert shared_figures.EMPTY_NOTE in notes


# =============================================================================
# The two figures
# =============================================================================
def test_the_class_figure_draws_one_row_per_metric_in_clinical_order(per_sample) -> None:
    segment, recording, units = distributions.build_frames(per_sample)

    figure = distributions.build_class_figure(segment, recording, units)
    try:
        assert len(figure.axes) == len(distributions.METRICS)
        legend = figure.axes[0].get_legend()
        drawn = [text.get_text().split(" (")[0] for text in legend.get_texts()]
    finally:
        shared_figures.plt.close(figure)

    assert drawn == ["healthy", "acidosis", "hie"]


def test_the_class_figure_uses_the_clinical_palette(per_sample) -> None:
    """The same green/amber/red every other cohort figure in this evaluation is drawn in.

    Read off the face colour with ``to_hex``, which drops the alpha channel -- so this asserts the
    hue and stays indifferent to how translucent the fill is, which is its own choice with its own
    test above.
    """
    from matplotlib.colors import to_hex

    segment, recording, units = distributions.build_frames(per_sample)

    figure = distributions.build_class_figure(segment, recording, units)
    try:
        filled = [p for p in figure.axes[0].patches if p.get_fill()]
        faces = {to_hex(patch.get_facecolor()) for patch in filled}
    finally:
        shared_figures.plt.close(figure)

    expected = {
        colour.lower()
        for colour in figures_seam.group_colors(["healthy", "acidosis", "hie"]).values()
    }
    assert faces == expected


def test_the_axis_label_carries_the_unit_the_panel_is_drawn_in(per_sample) -> None:
    """The label is where a reader of the figure learns the scale, and it is the only place: there
    is no clinical unit and no colour bar on these panels."""
    segment, recording, units = distributions.build_frames(per_sample)

    figure = distributions.build_class_figure(segment, recording, units)
    try:
        label = figure.axes[0].get_xlabel()
    finally:
        shared_figures.plt.close(figure)

    assert label == f"rmse_full ({NORMALISED_UNIT})"


def test_the_subgroup_figure_nests_each_class_in_its_own_column(per_sample) -> None:
    """Eight densities on one axes is unreadable, and the subgroup axis is already a subdivision
    of the class axis -- so a column is a class and a cell holds only that class's subgroups."""
    segment, recording, units = distributions.build_frames(per_sample)

    figure = distributions.build_subgroup_figure(segment, recording, units)
    try:
        assert len(figure.axes) == len(distributions.METRICS) * 3
        # The first row's three cells, left to right.
        titles = [figure.axes[index].get_title() for index in range(3)]
        healthy_cell = figure.axes[0].get_legend()
        healthy_labels = [text.get_text().split(" (")[0] for text in healthy_cell.get_texts()]
    finally:
        shared_figures.plt.close(figure)

    assert [title.split(":")[0] for title in titles] == ["healthy", "acidosis", "hie"]
    assert healthy_labels == ["healthy_no_bg_no_cs", "healthy_bg_cs"]


def test_a_subgroup_is_placed_under_the_class_its_own_rows_carry(per_sample) -> None:
    """Read from the data rather than from the stem's prefix: the class comes from the target
    tensor and the subgroup from the shard basename, so their pairing is a property of the split
    being evaluated."""
    assert distributions.subgroups_of_class(per_sample, "acidosis") == ["acidosis_cs"]
    assert distributions.subgroups_of_class(per_sample, "healthy") == [
        "healthy_no_bg_no_cs", "healthy_bg_cs",
    ]
    assert distributions.subgroups_of_class(per_sample, "not_a_class") == []


# =============================================================================
# The analysis end to end, and the degenerate splits
# =============================================================================
class _Collection:
    """The two attributes this analysis reads off the collection."""

    def __init__(self, per_sample, record=None):
        self.per_sample = per_sample
        self.record = {} if record is None else record


class _Context:
    def __init__(self, collection):
        self.collection = collection
        self.config = {}


def _run(frame, tmp_path, record=None):
    """Run the analysis."""
    return distributions.run_distributions_analysis(
        _Context(_Collection(frame, record)), eval_config={}, output_dir=tmp_path, probe=None
    )


def test_the_analysis_writes_both_tables_and_both_figures(per_sample, tmp_path) -> None:
    result = _run(per_sample, tmp_path)
    directory = tmp_path / distributions.ANALYSIS_DIRNAME

    for name in (
        distributions.PER_SEGMENT_FILENAME, distributions.SUMMARY_FILENAME,
        distributions.CLASS_FIGURE, distributions.SUBGROUP_FIGURE,
    ):
        assert (directory / name).is_file() and (directory / name).stat().st_size > 0

    assert result["composition"]["n_segments"] == len(per_sample)
    assert result["composition"]["n_recordings"] == 4
    assert result["plan"]["capped"] is False
    assert result["cohorts"][labels.CLASS_COLUMN] == ["healthy", "acidosis", "hie"]


def test_the_derived_csv_is_the_figures_reproducible_from_disk(per_sample, tmp_path) -> None:
    """The drawn columns are not on ``per_sample.csv`` under these names -- the rooted metrics are
    derived here -- so an offline reader needs them written down rather than recomputable only
    through this module."""
    _run(per_sample, tmp_path)
    written = pd.read_csv(
        tmp_path / distributions.ANALYSIS_DIRNAME / distributions.PER_SEGMENT_FILENAME
    )

    for metric in distributions.METRICS:
        assert metric.name in written.columns
    assert {"guid", "epoch", labels.CLASS_COLUMN, labels.SUBGROUP_COLUMN} <= set(written.columns)
    assert len(written) == len(per_sample)


def test_the_figures_do_not_collide_with_the_grouped_variant_naming() -> None:
    """The trap this analysis fell into once, kept from recurring.

    ``*_by_clinical_class.pdf`` and ``*_by_subgroup.pdf`` are the runner's grouped-variant
    violins, and the smoke test normalises them out of the figure manifest as a *family*. A figure
    of any other analysis named into that shape is therefore never recorded in the manifest and
    never documented -- it simply vanishes, while reading to an operator as one of the violin
    figures it is not. Which is the exact confusion this analysis exists to prevent.
    """
    for filename in (distributions.CLASS_FIGURE, distributions.SUBGROUP_FIGURE):
        assert not filename.endswith(GROUPED_SUFFIXES), filename


def test_the_analysis_declares_no_grouped_frame(per_sample, tmp_path) -> None:
    """The runner's fan-out draws violins documented as holding one value per **recording**.
    Handing it this per-segment frame would produce a per-segment violin that reads as a
    per-recording one -- the exact confusion this analysis exists to make visible."""
    result = _run(per_sample, tmp_path)

    assert "grouped_frames" not in result


def test_the_analysis_computes_no_test_interval_or_headline(per_sample, tmp_path) -> None:
    """Descriptive by construction: no $p$-value, no interval, no verdict, and nothing that the
    headline registry could dig a scalar out of. A separation visible here is a reason to look,
    and ``cross_subgroup`` is what decides whether one survives being asked properly."""
    result = _run(per_sample, tmp_path)

    forbidden = ("p_value", "ci_lo", "ci_hi", "headline", "verdict", "bootstrap_resamples")
    assert not any(key in result for key in forbidden)
    table = pd.read_csv(
        tmp_path / distributions.ANALYSIS_DIRNAME / distributions.SUMMARY_FILENAME
    )
    assert not any(name in table.columns for name in forbidden)


def test_the_record_carries_both_standing_notes(per_sample, tmp_path) -> None:
    """Both travel in ``summary.json`` rather than only in the documentation: a caveat a reader of
    the output cannot see is a caveat that does not apply."""
    result = _run(per_sample, tmp_path)

    assert "descriptive only" in result["descriptive_only"]
    assert "cross_subgroup" in result["descriptive_only"]
    assert result["per_segment_root_note"]


def test_a_single_cohort_split_still_draws_rather_than_skipping(tmp_path) -> None:
    """Unlike the grouped violins, a one-cohort population is *not* a skip here: one histogram of
    the whole split is a perfectly good description, and there is no comparison being invited."""
    frame = _frame()
    frame[labels.CLASS_COLUMN] = "healthy"
    frame[labels.SUBGROUP_COLUMN] = "healthy_bg_cs"

    result = _run(frame, tmp_path)

    assert result["cohorts"][labels.CLASS_COLUMN] == ["healthy"]
    assert (
        tmp_path / distributions.ANALYSIS_DIRNAME / distributions.CLASS_FIGURE
    ).stat().st_size > 0


def test_an_absent_metric_column_is_unmeasured_rather_than_a_failure(per_sample, tmp_path) -> None:
    """An older run's tables may not carry every column this analysis draws."""
    result = _run(per_sample.drop(columns=["mc_pred_gap", "attention_entropy_nats"]), tmp_path)
    table = pd.read_csv(
        tmp_path / distributions.ANALYSIS_DIRNAME / distributions.SUMMARY_FILENAME
    )

    absent = table[table["metric"] == "mc_pred_gap"]
    assert len(absent) and int(absent["n"].sum()) == 0
    assert result["composition"]["n_segments"] == len(per_sample)


def test_an_all_nan_metric_is_unmeasured_rather_than_a_failure(per_sample, tmp_path) -> None:
    frame = per_sample.copy()
    frame["source_conditioned_kl_raw"] = np.nan

    result = _run(frame, tmp_path)

    assert result["composition"]["n_segments"] == len(frame)


def test_an_empty_table_produces_empty_figures_rather_than_raising(tmp_path) -> None:
    """A fully masked split is a run that measured nothing, not a run that failed."""
    result = _run(_frame().iloc[:0], tmp_path)

    assert result["composition"]["n_segments"] == 0
    assert result["cohorts"][labels.CLASS_COLUMN] == []
    assert (
        tmp_path / distributions.ANALYSIS_DIRNAME / distributions.SUBGROUP_FIGURE
    ).stat().st_size > 0


def test_a_run_whose_record_carries_no_normalisation_block_still_completes(
    per_sample, tmp_path
) -> None:
    """The sibling needs that block and degrades its units without it. This one never reads it, so
    an empty record must change nothing at all -- which is a stronger statement than "it does not
    raise" and is what this asserts."""
    with_record = _run(per_sample, tmp_path / "with", record={"normalization": {"fhr_st": {}}})
    without = _run(per_sample, tmp_path / "without", record={})

    assert with_record["metrics"] == without["metrics"]
    assert {entry["unit"] for entry in without["metrics"]} == {
        metric.unit for metric in distributions.METRICS
    }


# =============================================================================
# Against a finished run, with no checkpoint
#
# The property that makes this analysis cheap to iterate on, and the one the hand-built frames
# above cannot check: the table it needs is the one already on disk, read back through a CSV round
# trip. A column that arrived as a string, or a cohort label that came back as the float ``nan``
# rather than as ``None``, would pass every test above and fail here.
# =============================================================================
@pytest.mark.slow
def test_it_runs_offline_against_a_finished_directory_with_no_model(
    collected_run, tmp_path, monkeypatch
) -> None:
    """``--only distributions`` against a finished run: no checkpoint, no GPU, no forward pass.

    The finished run is copied rather than re-entered so this pass cannot disturb the fixture
    every other file in the suite questions.
    """
    import json
    import shutil

    from teb_vae.lag_attn_cfs.eval import run as run_module
    from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs

    run_dir = tmp_path / "rerun"
    shutil.copytree(collected_run["results_dir"].parent, run_dir)

    def _explode(*args, **kwargs):
        raise AssertionError("the model was built and forwarded on an offline re-run")

    monkeypatch.setattr(SeqVaeLagAttnCfs, "forward", _explode)

    exit_code = run_module.main(None, run_dir, only="distributions", device="cpu")

    results_dir = run_dir / run_module.RESULTS_DIRNAME
    summary = json.loads((results_dir / run_module.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    record = summary["results"]["distributions"]
    directory = results_dir / distributions.ANALYSIS_DIRNAME

    assert exit_code == 0 and summary["checkpoint"] is None
    for name in (
        distributions.PER_SEGMENT_FILENAME, distributions.SUMMARY_FILENAME,
        distributions.CLASS_FIGURE, distributions.SUBGROUP_FIGURE,
    ):
        assert (directory / name).is_file() and (directory / name).stat().st_size > 0

    # The cohorts survived the CSV round trip as labels rather than as the string ``'nan'``, which
    # is what an unlabelled column comes back as if it is stringified before the null test.
    assert record["cohorts"][labels.CLASS_COLUMN], "no clinical class survived the round trip"
    assert "nan" not in record["cohorts"][labels.CLASS_COLUMN]
    assert record["composition"]["n_segments"] > 0

    # And the numbers are real: every metric the run produced has finite values at both levels.
    table = pd.read_csv(directory / distributions.SUMMARY_FILENAME)
    pooled = table[
        (table["group_column"] == distributions.POOLED_AXIS)
        & (table["level"] == distributions.SEGMENT_LEVEL)
    ]
    assert int(pooled["n"].sum()) > 0
    assert set(pooled["metric"]) == {metric.name for metric in distributions.METRICS}
