r"""Trace selection, axis labelling, and the report generator's refusal to invent a number.

Everything here is text and small arrays. Nothing renders a figure, builds a model or opens a run
directory: the figure functions are exercised by the user's smoke stage, and what this file checks
is the part of the generator that decides *what a reader is told* -- which recording a trace comes
from, what an explained-variance label says when the variance is not finite, and what appears in a
table cell that was never measured.

The two properties worth stating outright, because they are the ones a report is wrong in a way
nobody notices:

* :func:`~latent_pilot.report.select_traces` never reads a score. The tests give it a score column
  and shuffle it; the selection does not move.
* :func:`~latent_pilot.report.build_report` emits no number that was not in the record it was
  handed. An empty record produces a report full of ``not measured``, not a report full of zeros.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn.eval import labels
from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, report
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError


class _Projection:
    """The one attribute the captions read off a fitted projection."""

    def __init__(self, ratios):
        self.explained_variance_ratio = np.asarray(ratios, dtype=np.float64)


def _cohort(n_healthy=4, n_adverse=3, scores=None):
    """A recording table with a class column and, optionally, a score column to be ignored."""
    rows = []
    for index in range(n_healthy):
        rows.append({
            data.GUID_COLUMN: f"h{index}",
            data.OUTCOME_COLUMN: 0,
            labels.CLASS_COLUMN: "healthy",
        })
    for index in range(n_adverse):
        rows.append({
            data.GUID_COLUMN: f"a{index}",
            data.OUTCOME_COLUMN: 1,
            labels.CLASS_COLUMN: "acidosis",
        })
    frame = pd.DataFrame(rows)
    if scores is not None:
        frame["score"] = list(scores)
    return frame


# =============================================================================
# The trajectory axis
# =============================================================================
def test_signed_hours_put_delivery_at_zero_on_the_right():
    """Six half-hour bins over three hours, bin 0 nearest delivery, every coordinate negative."""
    x = report.signed_bin_hours(bin_hours=0.5, preservation_hours=3.0)
    assert x.size == 6
    assert x[0] == pytest.approx(-0.25)
    assert x[-1] == pytest.approx(-2.75)
    assert (x < 0.0).all()
    assert (np.diff(x) < 0.0).all()


def test_bin_axis_matches_the_data_module_edges():
    """The figure's x coordinate is the midpoint of the bin the data module defines, not a guess."""
    edges = data.bin_edges(bin_hours=0.5, preservation_hours=3.0)
    x = report.signed_bin_hours(bin_hours=0.5, preservation_hours=3.0)
    for index, (low, high) in enumerate(edges):
        assert x[index] == pytest.approx(-0.5 * (low + high))


# =============================================================================
# Trace selection: seeded, stratified, and blind to every score
# =============================================================================
def test_trace_selection_is_reproducible_from_the_seed():
    frame = _cohort()
    assert report.select_traces(frame, n=4, seed=7) == report.select_traces(frame, n=4, seed=7)


def test_a_different_seed_can_choose_differently():
    frame = _cohort(n_healthy=8, n_adverse=8)
    picks = {tuple(report.select_traces(frame, n=4, seed=seed)) for seed in range(6)}
    assert len(picks) > 1


def test_selection_never_reads_a_score():
    """The same GUIDs whatever the scores say, including when the row order follows them."""
    plain = _cohort()
    scored = _cohort(scores=[9.0, -9.0, 3.0, -3.0, 8.0, -8.0, 0.0])
    reordered = scored.sort_values("score").reset_index(drop=True)
    chosen = report.select_traces(plain, n=3, seed=1)
    assert report.select_traces(scored, n=3, seed=1) == chosen
    assert report.select_traces(reordered, n=3, seed=1) == chosen


def test_selection_takes_the_classes_round_robin():
    """A rare class is represented rather than sampled away by a majority class."""
    frame = _cohort(n_healthy=20, n_adverse=2)
    chosen = report.select_traces(frame, n=4, seed=3)
    assert sum(1 for guid in chosen if guid.startswith("a")) == 2


def test_selection_caps_at_the_cohort_and_returns_unique_recordings():
    frame = _cohort(n_healthy=2, n_adverse=1)
    chosen = report.select_traces(frame, n=10, seed=0)
    assert len(chosen) == 3
    assert len(set(chosen)) == 3


def test_selection_of_nothing_is_empty_rather_than_an_error():
    assert report.select_traces(_cohort(), n=0, seed=0) == []
    assert report.select_traces(pd.DataFrame({data.GUID_COLUMN: []}), n=3, seed=0) == []


def test_selection_without_a_guid_column_refuses_by_name():
    with pytest.raises(PilotConfigError, match="guid"):
        report.select_traces(pd.DataFrame({"score": [1.0, 2.0]}), n=2, seed=0)


def test_selection_without_a_class_column_still_works_as_one_stratum():
    frame = pd.DataFrame({data.GUID_COLUMN: ["a", "b", "c"]})
    chosen = report.select_traces(frame, n=2, seed=0)
    assert len(chosen) == 2
    assert set(chosen) <= {"a", "b", "c"}


def test_an_unoccupied_bin_stays_a_gap_rather_than_becoming_a_zero():
    """A bin a group never occupied is ``nan`` in the curve, which matplotlib draws as a break.
    Filling it with zero would draw a measurement that was never made."""
    x = report.signed_bin_hours(bin_hours=0.5, preservation_hours=3.0)
    bands = pd.DataFrame([
        {"group": "healthy", data.BIN_COLUMN: 0, "mean": 1.0, "lo": 0.5, "hi": 1.5,
         "n_recordings": 4},
        {"group": "healthy", data.BIN_COLUMN: 4, "mean": 2.0, "lo": 1.0, "hi": 3.0,
         "n_recordings": 2},
    ])
    curves = report._band_curves(bands, x)
    assert set(curves) == {"healthy"}
    entry = curves["healthy"]
    assert entry["mean"][0] == 1.0 and entry["mean"][4] == 2.0
    assert np.isnan(entry["mean"][[1, 2, 3, 5]]).all()
    assert entry["n"].tolist() == [4, 0, 0, 0, 2, 0]


def test_a_bin_outside_the_axis_is_dropped_rather_than_wrapped():
    """A negative bin index is the data module's "outside the window" marker; drawing it at
    position -1 would put an out-of-window summary on the last bin."""
    x = report.signed_bin_hours(bin_hours=0.5, preservation_hours=3.0)
    bands = pd.DataFrame([
        {"group": "healthy", data.BIN_COLUMN: -1, "mean": 9.0, "lo": 8.0, "hi": 10.0,
         "n_recordings": 3},
    ])
    entry = report._band_curves(bands, x)["healthy"]
    assert not np.isfinite(entry["mean"]).any()
    assert entry["n"].tolist() == [0] * 6


# =============================================================================
# Captions and labels
# =============================================================================
def test_variance_caption_states_both_shares():
    caption = report.variance_caption(_Projection([0.412, 0.189]))
    assert "PC1 41.2%" in caption and "PC2 18.9%" in caption


def test_variance_caption_does_not_invent_a_share_it_does_not_have():
    caption = report.variance_caption(_Projection([np.nan, np.nan]))
    assert caption == "explained variance unavailable"
    assert "%" not in caption


def test_axis_label_falls_back_to_the_bare_component_name():
    assert report.axis_label(_Projection([0.5, 0.25]), 0) == "PC1 (50.0%)"
    assert report.axis_label(_Projection([np.nan]), 0) == "PC1"
    assert report.axis_label(_Projection([0.5]), 1) == "PC2"


def test_score_scale_note_names_the_model_and_refuses_the_misreading():
    note = report.score_scale_note("adapted")
    assert "adapted" in note
    assert "not latent or physiological motion" in note


def test_the_panel_sized_disclosure_still_says_the_scales_are_not_comparable():
    short = report.score_scale_note("adapted", short=True)
    assert "adapted" in short
    assert "not comparable across models" in short
    assert len(short) < len(report.score_scale_note("adapted"))


def test_two_groups_never_share_one_fallback_colour():
    """Grouping by the binary outcome must not draw both lines in the same grey."""
    palette = report._palette(["0", "1"])
    assert palette["0"] != palette["1"]
    for name in report.PLOT_CLASSES:
        assert palette[name] == report.class_palette()[name]


def test_every_figure_has_a_caption():
    assert set(report.CAPTIONS) == {
        report.FIGURE_LATENT_SPACE,
        report.FIGURE_COVERAGE_SPACE,
        report.FIGURE_SUPERVISED_AXIS,
        report.FIGURE_TRAJECTORIES,
    }


# =============================================================================
# The coverage control's labels
# =============================================================================
def test_ascertainment_labels_name_both_flags():
    frame = pd.DataFrame([
        {"bg_label": True, "cs_label": False},
        {"bg_label": False, "cs_label": True},
        {"bg_label": None, "cs_label": None},
    ])
    strata = report.coverage_groupings(frame)["ascertainment"]
    assert strata == ["BG / no CS", "no BG / CS", "no BG / no CS"]


def test_coverage_labels_split_the_cohort_into_thirds():
    frame = pd.DataFrame({"last_anchor_hours": [0.1, 0.2, 0.5, 0.6, 0.9, 1.0]})
    observed = report.coverage_groupings(frame)["coverage"]
    assert observed[0].endswith("nearest delivery")
    assert observed[-1].endswith("earliest")
    assert len(set(observed)) == 3


def test_a_cohort_too_small_to_stratify_says_so_rather_than_inventing_thirds():
    frame = pd.DataFrame({"last_anchor_hours": [0.4, 0.4, 0.4, 0.4]})
    observed = report.coverage_groupings(frame)["coverage"]
    assert set(observed) == {"last observed: not stratified"}
    assert set(report.coverage_groupings(pd.DataFrame({"a": [1, 2]}))["coverage"]) == {
        "last observed: not stratified"
    }


def test_an_unknown_last_observed_time_is_its_own_group():
    frame = pd.DataFrame({"last_anchor_hours": [0.1, 0.5, 0.9, float("nan")]})
    observed = report.coverage_groupings(frame)["coverage"]
    assert observed[-1] == "last observed: unknown"


def test_class_names_come_from_the_repository_class_table():
    assert report.PLOT_CLASSES == tuple(labels.CLASS_NAMES.values())


# =============================================================================
# Attaching labels to label-blind summaries
# =============================================================================
def test_unknown_guids_stay_unlabelled_rather_than_guessed():
    summaries = pd.DataFrame({data.GUID_COLUMN: ["h0", "zz"], "n_segments": [3, 2]})
    joined = report.with_class_names(summaries, _cohort())
    assert joined[labels.CLASS_COLUMN].tolist()[0] == "healthy"
    assert pd.isna(joined[labels.CLASS_COLUMN].tolist()[1])
    assert pd.isna(joined[data.OUTCOME_COLUMN].tolist()[1])


# =============================================================================
# Rendering measured values, and refusing to render unmeasured ones
# =============================================================================
def test_absent_and_non_finite_values_render_as_missing():
    assert report._num(None) == report.MISSING
    assert report._num(float("nan")) == report.MISSING
    assert report._num(float("inf")) == report.MISSING
    assert report._int(None) == report.MISSING
    assert report._text("") == report.MISSING
    assert report._text(None) == report.MISSING


def test_measured_values_render_as_themselves():
    assert report._num(0.5, 3) == "0.500"
    assert report._int(np.int64(7)) == "7"
    assert report._text({"revision": "abc", "dirty": None}) == "dirty=not measured, revision=abc"


def test_an_interval_without_bounds_carries_the_estimator_reason():
    rendered = report._ci({
        "point": 0.5, "lo": float("nan"), "hi": float("nan"), "note": "only 2 usable draws",
    })
    assert rendered.startswith("0.5000")
    assert "only 2 usable draws" in rendered
    assert "[" not in rendered


def test_an_interval_with_bounds_prints_them():
    assert report._ci({"point": 0.71, "lo": 0.6, "hi": 0.82}, 2) == "0.71 [0.60, 0.82]"


def test_an_absent_interval_is_missing_rather_than_zero():
    assert report._ci(None) == report.MISSING
    assert report._ci({}) == report.MISSING


def test_a_table_marks_the_cells_it_has_no_value_for():
    rendered = report._table([{"a": 1, "b": None}, {"a": float("nan"), "b": "x"}])
    assert rendered.count(report.MISSING) == 2
    assert "| a | b |" in rendered


def test_an_empty_table_says_so_instead_of_rendering_a_header():
    assert report.MISSING in report._table([])


def test_a_table_escapes_a_pipe_so_one_cell_cannot_become_two():
    assert "\\|" in report._table([{"note": "a|b"}])


# =============================================================================
# The written report
# =============================================================================
def test_the_temporal_heading_follows_the_recorded_window_not_the_shipped_one():
    """The window is a config value, so a report that names three hours must have measured them."""
    record = {"protocol": {"settings": {"windows": {"preservation_hours": 3.0}}}, "temporal": {}}
    assert "## Change over the last 3 hours" in report.build_report(record)

    record["protocol"]["settings"]["windows"]["preservation_hours"] = 4.5
    assert "## Change over the last 4.5 hours" in report.build_report(record)

    # And with nothing recorded, a geometry-free heading rather than an invented default.
    assert "## Change before delivery" in report.build_report({"temporal": {}})


def test_an_empty_run_reports_nothing_measured_rather_than_zeros():
    text = report.build_report({})
    for heading in (
        "## Provenance", "## Cohort", "## Fitting and selection", "## Preservation",
        "## Held-out discrimination", "## Controls", "## Change before delivery",
        "## Prespecified subgroups", "## Figures", "## Limitations", "## Reproduction",
    ):
        assert heading in text
    assert text.count(report.MISSING) > 10
    assert "0.0000" not in text


def test_measured_metrics_reach_the_report_unchanged():
    text = report.build_report({
        "metrics": {
            "models": {
                "frozen": {"auroc": 0.6123, "n_recordings": 40},
                "adapted": {"auroc": 0.7456, "n_recordings": 40},
            },
            "bootstrap": {
                "paired": {"adapted - frozen": {"auroc": {"point": 0.13, "lo": -0.02, "hi": 0.29}}},
                "grouping": "guid",
                "grouping_disclosure": "no patient mapping was supplied",
                "resamples": 1000,
                "n_units": 40,
                "n_undefined_draws": 0,
                "method": "outcome-stratified paired percentile bootstrap over guids",
                "note": "conditional on this fold and seed",
            },
        },
    })
    assert "0.6123" in text and "0.7456" in text
    assert "0.1300 [-0.0200, 0.2900]" in text
    assert "no patient mapping was supplied" in text


def test_a_retained_frozen_model_is_reported_as_the_result_it_is():
    text = report.build_report({"selection": {"adaptation": {"selected_epoch": 0}}})
    assert "retained the frozen model" in text
    assert "not a failure to select" in text


def test_a_selected_epoch_is_named():
    text = report.build_report({"selection": {"adaptation": {"selected_epoch": 4}}})
    assert "chose epoch 4" in text


def test_gate_failures_and_warnings_are_both_shown_and_kept_apart():
    text = report.build_report({
        "preservation": {"gate": {
            "passed": False,
            "reasons": ["forecast: MSE rose 22.00%"],
            "warnings": ["healthy-only forecast MSE rose 14.00%"],
        }},
    })
    assert "**Gate failure:** forecast: MSE rose 22.00%" in text
    assert "*Reported, not gated:* healthy-only forecast MSE rose 14.00%" in text


def test_unknown_exposure_becomes_a_limitation_rather_than_a_clean_holdout():
    text = report.build_report({
        "exposure": {
            "pretraining": {"known": False, "n_exposed": None, "note": "no list was supplied"},
            "selection": {"known": False, "n_exposed": None, "note": "no list was supplied"},
            "statistics": {"population": None},
            "clean_holdout_supported": False,
        },
    })
    assert "Clean-holdout claim supported: **False**" in text
    assert "exploratory reuse" in text


def test_a_control_run_is_never_described_as_a_permutation_p_value():
    text = report.build_report({
        "controls": {"disclosure": {
            "shuffled_label_note": "1 shuffled-label fit(s): a leakage and overfitting sanity check",
            "prior_probe_note": "the frozen mu_prior probe ran",
            "permutation_p_value": False,
            "combined_branch_claim_supported": True,
        }},
    })
    assert "Permutation p-value: **False**" in text
    assert "sanity check" in text


def test_fixed_limitations_are_always_present():
    text = report.build_report({})
    assert "One fold, one checkpoint, one seed" in text
    assert "not a calibrated clinical risk" in text


def test_reproduction_commands_come_from_the_run_that_was_actually_launched():
    commands = report.reproduction_commands({
        "run_args": {"config_path": "configs/pilot.yaml"},
        "run_directory": "runs/fold_1/seed_42/abc",
    })
    assert commands[0].endswith("--config configs/pilot.yaml --stage all")
    assert "--run-dir runs/fold_1/seed_42/abc" in commands[1]
    assert all(report.RUNNER_MODULE in command for command in commands)


def test_no_reproduction_command_is_invented_when_none_was_recorded():
    assert report.reproduction_commands(None) == []
    assert report.reproduction_commands({}) == []
    assert report.MISSING in report.build_report({}).split("## Reproduction")[1]


def test_figures_are_listed_with_the_caption_they_carry():
    text = report.build_report({
        "figures": {report.FIGURE_TRAJECTORIES: "runs/x/figures/figure3_trajectories.pdf"},
    })
    assert "runs/x/figures/figure3_trajectories.pdf" in text
    assert report.CAPTIONS[report.FIGURE_TRAJECTORIES] in text


def test_the_report_is_written_where_it_is_asked_for(tmp_path):
    target = report.write_report({}, tmp_path)
    assert target.name == report.REPORT_FILENAME
    assert target.read_text(encoding="utf-8").startswith("# Latent-class fine-tuning pilot")
