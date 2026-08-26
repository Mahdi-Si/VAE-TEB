r"""The cohort axis: the order every figure and table is read in, and the population record.

Two things live in ``cohort.py`` and they are one subject -- who was evaluated, and the one time
axis every clinical reading resolves them against. Neither is visual polish, and both fail
silently, which is why they are pinned here rather than eyeballed on a rendered PDF.

**The order.** Cohorts run HIE, acidosis, healthy -- worst first -- and the eight subgroups run
in the reverse of their canonical order. The default everywhere is alphabetical, and alphabetical is
wrong in a *specific* way that looks fine: it puts ``acidosis`` left of ``hie`` on every class
figure, and on the subgroup axis it interleaves the three classes -- ``acidosis_cs``,
``acidosis_no_cs``, ``healthy_bg_cs``, ... -- so neither the severity ordering nor the
background/caesarean structure is visible. A reader comparing two figures drawn from different
cohort subsets would be comparing different columns without either figure saying so.

**The order is read off the shared labelling rather than restated.** ``CLASS_NAMES`` is keyed by
the dataset's own class codes and ``CANONICAL_SUBGROUPS`` is written in the intended order, so a
subgroup added to the dataset appears in these figures without an edit here -- and this package and
the raw cells' cannot come to disagree about what a cohort *is*.

**And the order is not only presentational: it decides every comparison's orientation.**
``stats.pairwise_comparisons`` names each pair in the order it receives the cohorts, so the same
ordering that puts HIE leftmost on a figure makes every significance test read *more severe against
less severe* -- HIE vs acidosis, HIE vs healthy, acidosis vs healthy -- with Cliff's delta signed
that way in every window of every readout. That is the direction the clinical question is asked in:
the cohort a readout exists to detect is named first. Alphabetical would name the first clinical
pair ``acidosis vs healthy`` and sign it against the axis its own figure draws it on, which is a
sign error a reader has no way to see.

**The time axis and the bin width are constants rather than settings.** An operator who could widen
the trajectory bin could merge two windows until a difference appeared or disappeared, which is the
same argument that keeps the significance level out of the configuration.

**The horizon, by contrast, *is* a setting -- and the difference is worth stating.**
``eval_config.max_hours_before_delivery`` bounds how far before delivery a segment may be recorded
and still be evaluated. Widening a *bin* merges two windows and can make a difference appear or
disappear while the population stays put, which is why the width is fixed. A horizon does the
opposite: it narrows the population openly, applies to the delivery clock before anything is
binned so every clock answers for the same segments, and lands in the resolved config dumped into
the run directory -- so a bounded run says so in its own artifacts. The tests below pin both the
filter's edge behaviour and the fact that every clock analysis reads the key.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn.eval import figures as shared_figures
from teb_vae.lag_attn_cfs.eval import cohort, figures_seam
from teb_vae.lag_attn_cfs.eval._reuse import labels, stats as shared_stats
from teb_vae.lag_attn_cfs.eval.analyses import cross_subgroup, time_to_delivery
from teb_vae.lag_attn_cfs.eval.report_seam import emit_grouped_variants

#: The order the two axes are read in, written out rather than derived, so this file states the
#: contract instead of restating the code that implements it.
EXPECTED_CLASS_ORDER = ["hie", "acidosis", "healthy"]
EXPECTED_SUBGROUP_ORDER = [
    "hie_cs",
    "hie_no_cs",
    "acidosis_cs",
    "acidosis_no_cs",
    "healthy_bg_cs",
    "healthy_bg_no_cs",
    "healthy_no_bg_cs",
    "healthy_no_bg_no_cs",
]

#: Two metrics, so a row count of ``n_groups x n_metrics`` cannot coincide with either factor.
_METRICS = ["pred_gap", "source_conditioned_kl_raw"]


@pytest.fixture
def eight_subgroup_frame() -> pd.DataFrame:
    """Three recordings in each of the eight subgroups, in the *reverse* of the expected order.

    Reversed deliberately: a frame already in the right order would pass every assertion below
    with the ordering unimplemented, and alphabetical order is neither, so neither a no-op nor a
    plain ``sorted`` reproduces the expected answer.
    """
    subgroups = [name for name in reversed(EXPECTED_SUBGROUP_ORDER) for _ in range(3)]
    return pd.DataFrame(
        {
            "guid": [f"g{index:02d}" for index in range(len(subgroups))],
            labels.SUBGROUP_COLUMN: subgroups,
            labels.CLASS_COLUMN: [name.split("_")[0] for name in subgroups],
            "pred_gap": np.linspace(1.0, 4.0, len(subgroups)),
            "source_conditioned_kl_raw": np.linspace(0.5, 2.0, len(subgroups)),
        }
    )


# =================================================================================================
# The order, at its source
# =================================================================================================
def test_the_canonical_order_is_the_stated_one_on_both_axes() -> None:
    """Against a shuffled input, so a function returning its argument unchanged fails."""
    shuffled_classes = ["healthy", "hie", "acidosis"]
    shuffled_subgroups = sorted(EXPECTED_SUBGROUP_ORDER)

    assert cohort.ordered_groups(shuffled_classes, labels.CLASS_COLUMN) == EXPECTED_CLASS_ORDER
    assert cohort.ordered_groups(
        shuffled_subgroups, labels.SUBGROUP_COLUMN
    ) == EXPECTED_SUBGROUP_ORDER
    # Alphabetical is a genuinely different answer on both axes, which is what makes the two
    # assertions above discriminating rather than accidentally satisfied.
    assert sorted(EXPECTED_CLASS_ORDER) != EXPECTED_CLASS_ORDER
    assert sorted(EXPECTED_SUBGROUP_ORDER) != EXPECTED_SUBGROUP_ORDER


def test_the_order_is_read_off_the_shared_labelling_rather_than_restated() -> None:
    """The two orderings this package draws in are the shared tables' own read worst-first, which
    is what keeps one definition of a cohort across the family rather than one per package."""
    assert EXPECTED_CLASS_ORDER == [
        labels.CLASS_NAMES[code] for code in sorted(labels.CLASS_NAMES, reverse=True)
    ]
    assert EXPECTED_SUBGROUP_ORDER == list(reversed(labels.CANONICAL_SUBGROUPS))


def test_a_partial_cohort_keeps_the_order_of_the_ones_present() -> None:
    """The ordinary case: no split carries all eight, and the order must not depend on which are
    there -- otherwise two figures of overlapping cohorts cannot be compared column by column."""
    present = ["hie_cs", "healthy_bg_cs", "acidosis_no_cs"]

    assert cohort.ordered_groups(present, labels.SUBGROUP_COLUMN) == [
        "hie_cs", "acidosis_no_cs", "healthy_bg_cs",
    ]


def test_an_unrecognised_cohort_is_appended_rather_than_dropped() -> None:
    """A non-canonical shard stem is a dataset question. Dropping it here would remove a cohort
    from a figure with nothing saying so; it sorts after every cohort the order knows."""
    ordered = cohort.ordered_groups(
        ["zzz_unknown", "hie", "healthy", "aaa_unknown"], labels.CLASS_COLUMN
    )

    assert ordered == ["hie", "healthy", "aaa_unknown", "zzz_unknown"]


def test_an_axis_that_is_neither_cohort_column_falls_through_to_alphabetical() -> None:
    """The lag readout resolves by time window, which is an axis neither canonical order knows.
    Falling through is the previous behaviour rather than a new one, and it must not raise."""
    windows = ["6-8h", "0-2h", "2-4h"]

    assert cohort.ordered_groups(windows, "time_window") == sorted(windows)


# =================================================================================================
# The order, where a reader meets it
# =================================================================================================
def _emit_and_capture(frame, tmp_path, monkeypatch):
    """Emit both grouped variants, capturing what reaches the figure builder.

    Returns:
        ``{axis: (groups, colors)}`` -- the arguments the violin figure was actually built from,
        which is the thing an operator sees, rather than an intermediate this file chose.
    """
    captured = {}
    original = shared_figures.grouped_violin_figure

    def _spy(values_by_metric, groups, **kwargs):
        captured[len(captured)] = (list(groups), dict(kwargs.get("colors") or {}))
        return original(values_by_metric, groups, **kwargs)

    monkeypatch.setattr(shared_figures, "grouped_violin_figure", _spy)
    emit_grouped_variants(frame, tmp_path, value_columns=_METRICS)
    # ``GROUP_COLUMNS`` is the emission order, so the two captures key back onto their axes.
    return dict(zip(labels.GROUP_COLUMNS, [captured[index] for index in sorted(captured)]))


def test_both_grouped_figures_are_drawn_in_the_canonical_order(
    eight_subgroup_frame, tmp_path, monkeypatch
) -> None:
    captured = _emit_and_capture(eight_subgroup_frame, tmp_path, monkeypatch)

    assert captured[labels.CLASS_COLUMN][0] == EXPECTED_CLASS_ORDER
    assert captured[labels.SUBGROUP_COLUMN][0] == EXPECTED_SUBGROUP_ORDER


def test_both_grouped_figures_are_drawn_in_the_clinical_palette(
    eight_subgroup_frame, tmp_path, monkeypatch
) -> None:
    """The colours must reach the figure, not merely exist in the seam: the shared builder falls
    back to its own palette when none is passed, and that fallback paints healthy blue."""
    captured = _emit_and_capture(eight_subgroup_frame, tmp_path, monkeypatch)

    for axis, (groups, colors) in captured.items():
        assert colors == figures_seam.group_colors(groups), axis
    assert captured[labels.CLASS_COLUMN][1]["healthy"] == figures_seam.CLINICAL_CLASS_COLORS[
        "healthy"
    ]


def test_the_grouped_table_is_written_in_the_same_order_as_its_figure(
    eight_subgroup_frame, tmp_path
) -> None:
    """A table ordered alphabetically beside a figure ordered clinically is the configuration in
    which a reader reads the third row against the first violin."""
    emit_grouped_variants(eight_subgroup_frame, tmp_path, value_columns=_METRICS)

    for axis, expected in (
        (labels.CLASS_COLUMN, EXPECTED_CLASS_ORDER),
        (labels.SUBGROUP_COLUMN, EXPECTED_SUBGROUP_ORDER),
    ):
        table = pd.read_csv(tmp_path / f"per_sample_by_{axis}.csv")
        assert list(dict.fromkeys(table["group"])) == expected, axis
        # And the table is still the whole summary, not a reordered subset of it.
        assert len(table) == len(expected) * len(_METRICS)


# =================================================================================================
# The order, where the statistics meet it
# =================================================================================================
#: The three comparisons of the class axis, in the order every test must report them: worst
#: first, so each reads more severe against less severe.
EXPECTED_CLASS_PAIRS = [("hie", "acidosis"), ("hie", "healthy"), ("acidosis", "healthy")]

#: Hours in the seconds ``epoch`` is stored in, for the clock frame below.
_HOUR = 3600.0


def _severity_ordered_recordings(classes=("healthy", "acidosis", "hie")) -> pd.DataFrame:
    """A per-recording frame whose metric grows with severity, the classes entered healthy-first.

    Healthy-first deliberately: a frame already in the intended order would pass the assertions
    below with the ordering unimplemented, and alphabetical is a third order again -- so neither a
    no-op nor a plain ``sorted`` reproduces the expected answer.
    """
    rows: List[Dict[str, Any]] = []
    for name in classes:
        offset = {"healthy": 0.0, "acidosis": 10.0, "hie": 20.0}[name]
        for recording in range(4):
            rows.append({
                "guid": f"{name}_{recording:02d}",
                labels.CLASS_COLUMN: name,
                "pred_gap": offset + float(recording),
            })
    return pd.DataFrame(rows)


def _clock_per_sample() -> pd.DataFrame:
    """A per-sample table with the three classes separated in both windows of the delivery clock."""
    rows: List[Dict[str, Any]] = []
    for name, offset in (("healthy", 0.0), ("acidosis", 50.0), ("hie", 100.0)):
        for recording in range(5):
            for segment, hours in enumerate((1.0, 3.0)):
                rows.append({
                    "guid": f"{name}_{recording:02d}",
                    "epoch": -hours * _HOUR,
                    labels.CLASS_COLUMN: name,
                    labels.SUBGROUP_COLUMN: f"{name}_no_cs",
                    "mc_pred_gap": offset + float(recording) + 0.1 * segment,
                    "source_conditioned_kl_raw": offset + float(recording),
                })
    return pd.DataFrame(rows)


def test_every_comparison_runs_from_the_more_severe_cohort_to_the_less_severe_one() -> None:
    """The class axis of ``cross_subgroup``, through the two functions that decide it: this
    analysis picks the cohorts and their order, the shared sweep names each pair in that order.

    ``sorted`` would answer ``acidosis vs healthy``, ``acidosis vs hie``, ``healthy vs hie`` --
    two of the same three comparisons reversed, which is a *sign* difference in the Cliff's delta
    beside each and nothing in the output would say so.
    """
    frame = _severity_ordered_recordings()

    usable, _ = cross_subgroup.usable_groups(frame, "pred_gap", labels.CLASS_COLUMN)
    pairs = shared_stats.pairwise_comparisons(usable)

    assert [(item["left"], item["right"]) for item in pairs] == EXPECTED_CLASS_PAIRS
    # The metric grows with severity, so the more severe cohort of every pair runs higher and each
    # delta is positive. That is the property the orientation exists to make readable: one sign
    # convention across every pair, rather than one per pair depending on how the names sorted.
    assert all(item["cliffs_delta"] > 0.0 for item in pairs)
    assert sorted(EXPECTED_CLASS_PAIRS) != EXPECTED_CLASS_PAIRS


def test_the_subgroup_comparisons_run_in_the_canonical_order_too(eight_subgroup_frame) -> None:
    """Twenty-eight pairs on the wider axis, and the canonical order decides all of them: the HIE
    subgroups are the left of every pair they appear in, the healthy ones the right."""
    usable, _ = cross_subgroup.usable_groups(
        eight_subgroup_frame, "pred_gap", labels.SUBGROUP_COLUMN
    )
    pairs = shared_stats.pairwise_comparisons(usable)

    assert [(item["left"], item["right"]) for item in pairs] == list(
        itertools.combinations(EXPECTED_SUBGROUP_ORDER, 2)
    )


def test_the_clock_windows_orient_their_comparisons_the_same_way() -> None:
    """The trajectory clocks reach the sweep through ``windowed_group_comparisons`` rather than
    directly, so the orientation has to survive that layer as well -- and it is per window, which
    is where a per-call ``sorted`` would be least visible."""
    per_recording = time_to_delivery.build_per_recording(_clock_per_sample())[labels.CLASS_COLUMN]

    record = time_to_delivery.analyse_windows(per_recording, "mc_pred_gap")

    # Non-vacuity: the classes are fifty nats apart, so both windows must survive Holm and be
    # swept. An implementation that tested nothing would satisfy the loop below trivially.
    assert record["n_significant_windows"] == 2
    for comparisons in record["pairwise"].values():
        assert [(item["left"], item["right"]) for item in comparisons] == EXPECTED_CLASS_PAIRS
        assert all(item["cliffs_delta"] > 0.0 for item in comparisons)


def test_the_effect_heatmap_rows_follow_the_cohort_order_rather_than_their_labels() -> None:
    """The page an operator actually reads. Sorting the row *labels* would put
    ``acidosis vs healthy`` above ``hie vs acidosis`` while the violins above run worst-first,
    which is the configuration in which a reader compares a row against the wrong column."""
    per_recording = time_to_delivery.build_per_recording(_clock_per_sample())[labels.CLASS_COLUMN]
    readout = time_to_delivery.READOUTS[0]
    record = time_to_delivery.analyse_windows(per_recording, readout.column)

    figure = time_to_delivery.build_windows_figure(per_recording, record, readout)
    try:
        # Found by content rather than by position: the colourbar is an axes of its own, so the
        # heatmap is not reliably the figure's last.
        drawn = [
            [text.get_text() for text in axis.get_yticklabels()]
            for axis in figure.axes
            if any(" vs " in text.get_text() for text in axis.get_yticklabels())
        ]
    finally:
        shared_figures.plt.close(figure)

    expected = [
        f"{readout.name}: {left} vs {right}" for left, right in EXPECTED_CLASS_PAIRS
    ]
    assert drawn == [expected]
    assert sorted(expected) != expected


# =================================================================================================
# The time axis
# =================================================================================================
def test_epoch_becomes_hours_before_delivery_and_bins_on_the_fixed_grid() -> None:
    r"""``epoch`` is negative before delivery, so $h = -\mathrm{epoch}/3600$ is non-negative for a
    segment recorded before it. The bin width is a module constant and not an ``eval_config`` key.
    """
    frame = pd.DataFrame({"guid": ["A", "B"], "epoch": [-3600.0, -5400.0]})

    binned = cohort.add_time_bins(frame)

    assert cohort.TRAJECTORY_BIN_HOURS == 0.5
    assert list(binned[cohort.HOURS_COLUMN]) == [1.0, 1.5]
    assert list(binned[cohort.BIN_COLUMN]) == [2, 3]
    assert list(binned[cohort.BIN_CENTER_COLUMN]) == [1.25, 1.75]


def test_a_segment_at_or_after_delivery_lands_in_the_first_bin() -> None:
    """A negative index would sort before every real window on every trajectory figure."""
    frame = pd.DataFrame({"guid": ["A"], "epoch": [600.0]})

    binned = cohort.add_time_bins(frame)

    assert int(binned[cohort.BIN_COLUMN].iloc[0]) == 0


def test_a_frame_with_no_usable_epoch_is_empty_with_the_columns_a_caller_will_group_on() -> None:
    frame = pd.DataFrame({"guid": ["A"], "epoch": [np.nan]})

    binned = cohort.add_time_bins(frame)

    assert len(binned) == 0
    added = {cohort.HOURS_COLUMN, cohort.BIN_COLUMN, cohort.BIN_CENTER_COLUMN}
    assert added <= set(binned.columns)


def test_a_recordings_value_in_a_window_is_the_mean_over_its_own_segments_there() -> None:
    """The aggregation chain applied *inside* a window. Without it a recording contributing eleven
    segments to a window would outvote one contributing two, which is the pseudo-replication the
    whole chain exists to keep out."""
    frame = pd.DataFrame(
        {
            "guid": ["A", "A", "B"],
            "epoch": [-3600.0, -3700.0, -3600.0],
            labels.CLASS_COLUMN: ["healthy", "healthy", "healthy"],
            "pred_gap": [1.0, 3.0, 10.0],
        }
    )

    reduced = cohort.per_recording_in_bins(
        cohort.add_time_bins(frame), ["pred_gap"], group_column=labels.CLASS_COLUMN
    )

    assert len(reduced) == 2
    assert sorted(reduced["pred_gap"]) == [2.0, 10.0]
    rows = cohort.trajectory_rows(reduced, "pred_gap", metric="pred_gap")
    # Recordings, not segments: the unit every statistic on this table is computed on.
    assert [row["n_recordings"] for row in rows] == [2]


# =================================================================================================
# The population record
# =================================================================================================
def test_disjointness_is_computed_from_the_two_resolved_lists_rather_than_asserted() -> None:
    """A constant outlives the configuration that made it true, and the out-of-distribution
    sentence is a consequence of this flag rather than a standing claim."""
    config = {
        "dataset_config": {
            "vae_train_datasets": ["/data/train/healthy_bg_cs.hdf5"],
            "vae_test_datasets": ["/data/test/hie_cs.hdf5"],
        }
    }
    frame = pd.DataFrame(
        {"guid": ["A"], labels.SUBGROUP_COLUMN: ["hie_cs"], labels.CLASS_COLUMN: ["hie"]}
    )

    block = cohort.build_cohort_block(frame, config)

    assert block["training_cohort_disjoint"] is True
    assert block["training_cohort_overlap"] == []
    assert block["out_of_distribution"] == cohort.OUT_OF_DISTRIBUTION_SENTENCE
    assert block["non_comparability"] == cohort.NON_COMPARABILITY_SENTENCE


def test_an_overlapping_split_reports_the_overlap_and_no_leakage_free_claim() -> None:
    """The same file in both lists, written with different slashes, is still the same file."""
    config = {
        "dataset_config": {
            "vae_train_datasets": ["/data/test/hie_cs.hdf5"],
            "vae_test_datasets": ["\\data\\test\\hie_cs.hdf5"],
        }
    }

    block = cohort.build_cohort_block(pd.DataFrame({"guid": []}), config)

    assert block["training_cohort_disjoint"] is False
    assert len(block["training_cohort_overlap"]) == 1
    assert "out_of_distribution" not in block


def test_a_run_that_named_no_training_set_cannot_claim_disjointness() -> None:
    """``False`` there would read as "they overlap", which is a different statement from
    "unknown"."""
    block = cohort.build_cohort_block(
        pd.DataFrame({"guid": []}),
        {"dataset_config": {"vae_test_datasets": ["/data/test/hie_cs.hdf5"]}},
    )

    assert block["training_cohort_disjoint"] is None


def test_the_unseen_subgroups_are_wider_than_the_two_unhealthy_classes() -> None:
    """The pretraining split is drawn from the healthy *with-background* subgroups only, so the two
    healthy no-background subgroups are unseen as well -- which is what the out-of-distribution
    sentence applies to and is easy to state one class too narrowly."""
    block = cohort.build_cohort_block(pd.DataFrame({"guid": []}), {})

    assert set(block["pretraining_subgroups"]) == {"healthy_bg_cs", "healthy_bg_no_cs"}
    assert set(block["unseen_subgroups"]) == set(labels.CANONICAL_SUBGROUPS) - {
        "healthy_bg_cs", "healthy_bg_no_cs"
    }
    assert "healthy_no_bg_cs" in block["unseen_subgroups"]


def test_the_missing_labour_onset_rows_are_counted_rather_than_dropped() -> None:
    """The field is NaN wherever the recording is absent from the labour-onset table, and a summary
    that quietly dropped those rows would report a mean over a population it does not name."""
    frame = pd.DataFrame({"guid": ["A", "B"], "time_from_labor_onset": [3600.0, np.nan]})

    record = cohort.labor_onset_readout(frame)

    assert (record["n_rows"], record["n_finite"], record["n_nan"]) == (2, 1, 1)
    assert record["nan_fraction"] == pytest.approx(0.5)
    assert record["mean_hours"] == pytest.approx(1.0)
    assert cohort.labor_onset_readout(pd.DataFrame({"guid": ["A"]}))["present"] is False


def test_both_count_levels_are_reported_and_both_are_in_the_canonical_order() -> None:
    """A subgroup with many segments and three recordings is one whose statistics have $n = 3$, and
    only the second count says so."""
    frame = pd.DataFrame(
        {
            "guid": ["A", "A", "B", "C"],
            labels.CLASS_COLUMN: ["hie", "hie", "healthy", "acidosis"],
        }
    )

    counts = cohort.cohort_counts(frame, labels.CLASS_COLUMN)

    assert counts["segments"] == {"hie": 2, "acidosis": 1, "healthy": 1}
    assert counts["recordings"] == {"hie": 1, "acidosis": 1, "healthy": 1}
    assert list(counts["segments"]) == EXPECTED_CLASS_ORDER


# =================================================================================================
# The evaluation horizon
# =================================================================================================
def test_no_horizon_returns_the_frame_untouched() -> None:
    """``None`` is the shipped setting, so this is the path every current run takes."""
    frame = pd.DataFrame({"epoch": [-3600.0, -36000.0], "guid": ["a", "b"]})

    assert cohort.within_horizon(frame, None) is frame


def test_the_horizon_is_inclusive_at_its_own_boundary() -> None:
    """A segment recorded exactly at the bound is inside it -- ``4.0`` means "the last four
    hours", and a boundary segment falling out would be an off-by-one nobody could see."""
    frame = pd.DataFrame({"epoch": [-4 * 3600.0, -4 * 3600.0 - 1.0], "guid": ["at", "beyond"]})

    kept = cohort.within_horizon(frame, 4.0)

    assert kept["guid"].tolist() == ["at"]


def test_a_segment_at_or_after_delivery_is_never_cut_by_the_horizon() -> None:
    """The horizon bounds the *far* side only. A non-negative ``epoch`` is at or after delivery,
    which the binner clips into the first window; cutting it here would silently change which
    segments that clip is applied to."""
    frame = pd.DataFrame({"epoch": [0.0, 600.0], "guid": ["at", "after"]})

    assert cohort.within_horizon(frame, 4.0)["guid"].tolist() == ["at", "after"]


def test_a_non_finite_epoch_survives_the_horizon_and_is_dropped_by_the_binner() -> None:
    """Two different reasons for a row to disappear, kept separable: the horizon means "out of
    scope" and the binner's finite check means "no usable coordinate". A row removed here would be
    attributed to the wrong one in the excluded counts."""
    frame = pd.DataFrame({"epoch": [np.nan, -36000.0], "guid": ["nan", "far"]})

    kept = cohort.within_horizon(frame, 4.0)

    assert kept["guid"].tolist() == ["nan"]
    assert cohort.add_time_bins(kept).empty


def test_the_horizon_tolerates_a_frame_with_no_epoch_column() -> None:
    """``add_second_stage_bins`` is handed frames assembled elsewhere; a missing column is not the
    horizon's failure to report."""
    frame = pd.DataFrame({"guid": ["a"]})

    assert cohort.within_horizon(frame, 4.0) is frame
    assert cohort.within_horizon(pd.DataFrame(), 4.0).empty


def test_every_clock_analysis_reads_the_horizon_key() -> None:
    """The anti-omission direction: a clock analysis added later that never reads the key would
    silently evaluate the whole split while the run's config said otherwise, and no assertion on
    the existing analyses would notice."""
    import re
    from pathlib import Path

    root = Path(cohort.__file__).resolve().parent / "analyses"
    clocks = [
        name for name in ("trajectory.py", "time_to_delivery.py", "second_stage.py",
                          "lag_kl.py", "lag_clocks.py")
        if (root / name).is_file()
    ]
    assert clocks, "no clock analysis found to check"

    missing = [
        name for name in clocks
        if not re.search(r"max_hours_before_delivery", (root / name).read_text(encoding="utf-8"))
    ]

    assert missing == [], (
        f"{missing} bin on a clinical clock but never read "
        f"eval_config.max_hours_before_delivery, so a bounded run would not bound them"
    )
