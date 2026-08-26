r"""The two presentation conventions every cohort figure and table in this evaluation obeys.

Neither is visual polish, and both fail silently -- which is why they are pinned here rather than
eyeballed on a rendered PDF.

**The order.** Cohorts run HIE, acidosis, healthy -- worst first -- and the eight subgroups run
in the reverse of their canonical order. That order is not only presentational: the shared pairwise
sweep names each pair in the order it receives the cohorts, so it is also what makes every
significance test read *more severe against less severe*. The default everywhere is alphabetical,
and alphabetical is wrong in a *specific* way that looks fine: it puts ``acidosis`` left of ``hie``
on every class figure, and on the subgroup axis it interleaves the three classes --
``acidosis_cs``, ``acidosis_no_cs``, ``healthy_bg_cs``, ... -- so neither the severity ordering nor
the background/caesarean structure is visible. A reader comparing two figures drawn from different
cohort subsets would be comparing different columns without either figure saying so.

**The palette.** Green for healthy, amber for acidosis, red for HIE, with each subgroup a shade of
its class. Asserted twice over: by the literals, so a change is deliberate, and by the *property*
the literals exist for -- the green channel dominates healthy and the red channel dominates HIE --
so a future palette that keeps three distinct colours while losing the clinical reading fails
here rather than shipping.

The two are tested together because they are one decision: a figure whose violins are ordered one
way and coloured by another convention is worse than either alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn.eval import figures as shared_figures
from teb_vae.lag_attn_rws.eval import cohort, figures_seam
from teb_vae.lag_attn_rws.eval._reuse import labels
from teb_vae.lag_attn_rws.eval.report_seam import emit_grouped_variants

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
    """Three recordings in each of the eight subgroups, in *reverse* canonical order.

    Reversed deliberately: a frame already in the right order would pass every assertion below
    with the ordering unimplemented, and alphabetical order is not the reverse of canonical, so
    neither a no-op nor a plain ``sorted`` reproduces the expected answer.
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


# =============================================================================
# The order, at its source
# =============================================================================
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


# =============================================================================
# The order, where a reader meets it
# =============================================================================
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


def test_the_conditioned_coupling_violins_are_ordered_and_coloured_by_class() -> None:
    """``events`` draws its own violins rather than going through the grouped fan-out, so it is
    the analysis that would drift back to ``groupby``'s alphabetical order unnoticed."""
    from teb_vae.lag_attn_rws.eval.analyses import events as events_analysis

    per_recording = pd.DataFrame(
        {
            "guid": [f"g{index}" for index in range(9)],
            "metric": ["pred_gap_mc_nats"] * 9,
            labels.CLASS_COLUMN: ["healthy"] * 3 + ["hie"] * 3 + ["acidosis"] * 3,
            "difference": np.linspace(-1.0, 1.0, 9),
        }
    )

    figure = events_analysis.build_conditioned_figure(per_recording)
    try:
        drawn = [text.get_text() for text in figure.axes[0].get_xticklabels()]
    finally:
        shared_figures.plt.close(figure)

    assert drawn == EXPECTED_CLASS_ORDER


def test_the_effect_heatmap_columns_run_in_the_canonical_cohort_order() -> None:
    """``cross_subgroup``'s x axis is cohort *pairs*, and the shared pairwise test names each pair
    in the order it receives the cohorts -- the canonical one, so a pair reads more severe against
    less severe. What this analysis chooses is the column order, keyed on where the two cohorts
    fall on that same axis; a shuffled input must still come back worst-first."""
    from teb_vae.lag_attn_rws.eval.analyses import cross_subgroup

    pairs = pd.DataFrame(
        {
            "left": ["acidosis", "hie", "hie"],
            "right": ["healthy", "healthy", "acidosis"],
        }
    )

    ordered = cross_subgroup._ordered_pair_labels(pairs, labels.CLASS_COLUMN)

    # Keyed on (position of left, position of right): hie is 0, so its pairs come first.
    assert ordered == ["hie vs acidosis", "hie vs healthy", "acidosis vs healthy"]


def test_every_comparison_runs_from_the_more_severe_cohort_to_the_less_severe_one() -> None:
    """The orientation the column order above is read against, at the two functions that decide
    it: this analysis picks the cohorts and their order, and the shared sweep names each pair in
    the order it receives them. ``sorted`` would answer ``acidosis vs hie`` -- the same comparison
    with its Cliff's delta reversed, and nothing in the output would say so.
    """
    from teb_vae.lag_attn_rws.eval.analyses import cross_subgroup
    from teb_vae.lag_attn_rws.eval._reuse import stats as shared_stats

    # Entered healthy-first, and the metric grows with severity: neither a no-op nor a plain
    # ``sorted`` reproduces the expected answer.
    frame = pd.DataFrame(
        [
            {
                "guid": f"{name}_{recording:02d}",
                labels.CLASS_COLUMN: name,
                "pred_gap": offset + float(recording),
            }
            for name, offset in (("healthy", 0.0), ("acidosis", 10.0), ("hie", 20.0))
            for recording in range(4)
        ]
    )

    usable, _ = cross_subgroup.usable_groups(frame, "pred_gap", labels.CLASS_COLUMN)
    comparisons = shared_stats.pairwise_comparisons(usable)

    assert [(item["left"], item["right"]) for item in comparisons] == [
        ("hie", "acidosis"), ("hie", "healthy"), ("acidosis", "healthy"),
    ]
    # One sign convention across every pair: the more severe cohort of each runs higher here.
    assert all(item["cliffs_delta"] > 0.0 for item in comparisons)


# =============================================================================
# The palette
# =============================================================================
def test_each_clinical_class_carries_its_conventional_colour() -> None:
    """Both the literal and the property behind it. The literal makes a change deliberate; the
    channel test is what a replacement palette still has to satisfy to be the right one."""
    colours = figures_seam.CLINICAL_CLASS_COLORS

    assert colours == {"healthy": "#2E8B57", "acidosis": "#E8A33D", "hie": "#C0392B"}
    red, green, blue = (
        {name: int(value[index:index + 2], 16) for name, value in colours.items()}
        for index in (1, 3, 5)
    )
    assert green["healthy"] > red["healthy"] and green["healthy"] > blue["healthy"]
    assert red["hie"] > green["hie"] and red["hie"] > blue["hie"]
    # Amber is red-plus-green with little blue, which is what separates it from both neighbours.
    assert red["acidosis"] > blue["acidosis"] and green["acidosis"] > blue["acidosis"]


def test_every_canonical_subgroup_is_a_shade_of_its_own_class() -> None:
    """The property the eight-cohort figures are readable because of: a violin's hue says which
    class it belongs to before its label is read."""
    for group in labels.CANONICAL_SUBGROUPS:
        shade = figures_seam.SUBGROUP_COLORS[group]
        red, green, blue = (int(shade[index:index + 2], 16) for index in (1, 3, 5))
        if group.startswith("healthy"):
            assert green > red and green > blue, group
        elif group.startswith("acidosis"):
            assert red > blue and green > blue, group
        else:
            assert red > green and red > blue, group


def test_the_subgroups_of_one_class_are_distinguishable_from_each_other() -> None:
    """A shading range that collapsed would give four identical green violins, which is worse
    than four unrelated hues: the figure would read as one cohort drawn four times."""
    # Read in the *canonical* order rather than the reading order: the ramp is assigned along
    # ``CANONICAL_SUBGROUPS``, so that -- not the worst-first order the figures are drawn in -- is
    # the axis the luminance is monotone along.
    healthy = [
        figures_seam.SUBGROUP_COLORS[name]
        for name in labels.CANONICAL_SUBGROUPS
        if name.startswith("healthy")
    ]

    assert len(set(healthy)) == len(healthy) == 4
    # Monotone in luminance across the canonical order, so the shading itself carries the order.
    luminance = [sum(int(value[index:index + 2], 16) for index in (1, 3, 5)) for value in healthy]
    assert luminance == sorted(luminance, reverse=True)


def test_the_palette_is_a_table_rather_than_an_assignment_pass() -> None:
    """Order-independence is what lets two figures of overlapping cohorts be compared. The shared
    palette assigns colours in arrival order for anything it does not know, and that is exactly
    the failure this replaces for the eleven labels it does."""
    every = list(figures_seam.CLINICAL_CLASS_COLORS) + list(EXPECTED_SUBGROUP_ORDER)
    resolved = figures_seam.group_colors(every)

    assert figures_seam.group_colors(list(reversed(every))) == resolved
    for name in every:
        assert figures_seam.group_colors([name]) == {name: resolved[name]}


def test_an_unknown_cohort_still_receives_a_colour() -> None:
    """A non-canonical shard stem must be drawn, not dropped, so it falls back to the shared
    palette rather than to ``None`` -- which matplotlib would read as "use the default"."""
    resolved = figures_seam.group_colors(["healthy", "not_a_canonical_shard"])

    assert set(resolved) == {"healthy", "not_a_canonical_shard"}
    assert resolved["not_a_canonical_shard"].startswith("#")

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
