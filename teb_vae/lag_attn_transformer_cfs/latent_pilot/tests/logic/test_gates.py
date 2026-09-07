r"""The preservation gate's arithmetic and the fixed subset it is measured on.

Hand-checkable numbers throughout. Nothing here runs a forward, decodes a forecast or opens a
checkpoint: a gate decision is a comparison between two records, and the subset draw is a
permutation over a small table. The pass that produces those records needs the real net and is
checked separately, on the execution machine.

Importing the evaluation module pulls the model-facing half of this package with it, which is the
one weight this file carries; it constructs nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, evaluate


def _recording(guid, *, outcome, bg_label=True, split="val", eligible=True, reason=""):
    """One row of the recording table, with only what the subset draw reads."""
    return {
        data.GUID_COLUMN: guid,
        data.SPLIT_COLUMN: split,
        data.OUTCOME_COLUMN: outcome,
        "bg_label": bg_label,
        data.EXCLUSION_COLUMN: reason,
        "eligible": eligible,
    }


def _cohort(**counts):
    """A recording table with the requested number of recordings per stratum."""
    rows = []
    for index in range(counts.get("adverse", 0)):
        rows.append(_recording(f"ADV-{index:02d}", outcome=1))
    for index in range(counts.get("healthy_bg", 0)):
        rows.append(_recording(f"HBG-{index:02d}", outcome=0, bg_label=True))
    for index in range(counts.get("healthy_no_bg", 0)):
        rows.append(_recording(f"HNB-{index:02d}", outcome=0, bg_label=False))
    return pd.DataFrame(rows)


def _record(**overrides):
    """A preservation record carrying the fields a gate decision reads."""
    record = {
        "mse_full": 1.0,
        "mse_full_healthy": 1.0,
        "delta_mu_sat_pp": 2.0,
        "finite": True,
        "n_latent_coordinates_varying": 8,
        "support_digest": "abcdef0123456789",
        "n_recordings": 12,
        "n_retained_anchors": 480,
    }
    record.update(overrides)
    return record


def _decide(baseline, candidate, **overrides):
    """Decide one gate at the protocol's declared tolerances."""
    settings = {"forecast_mse_max_increase": 0.10, "saturation_max_increase_pp": 5.0}
    settings.update(overrides)
    return evaluate.gate_decision(baseline, candidate, **settings)


# =============================================================================
# The subset
# =============================================================================
def test_the_subset_is_the_same_for_one_seed_and_moves_with_another():
    """Determinism is the property the whole before/after comparison rests on."""
    cohort = _cohort(adverse=6, healthy_bg=6, healthy_no_bg=6)

    first = evaluate.gate_subset(cohort, size=9, seed=42)
    second = evaluate.gate_subset(cohort.sample(frac=1.0, random_state=7), size=9, seed=42)
    other = evaluate.gate_subset(cohort, size=9, seed=43)

    # A shuffled input table chooses the same recordings: the draw sorts before it permutes.
    assert first["guids"] == second["guids"]
    assert first["guids"] != other["guids"]


def test_both_classes_appear_before_either_is_exhausted():
    """Round-robin across the strata, so a small subset is not one class's subset."""
    cohort = _cohort(adverse=2, healthy_bg=20, healthy_no_bg=20)

    record = evaluate.gate_subset(cohort, size=6, seed=42)

    assert record["n_recordings"] == 6
    assert record["counts_by_stratum"]["adverse"] == 2
    assert record["counts_by_stratum"]["healthy_bg"] == 2
    assert record["counts_by_stratum"]["healthy_no_bg"] == 2


def test_a_subset_larger_than_the_cohort_takes_the_cohort():
    cohort = _cohort(adverse=1, healthy_bg=2)

    record = evaluate.gate_subset(cohort, size=50, seed=42)

    assert record["n_recordings"] == 3
    assert record["requested_size"] == 50
    # The absent stratum is named rather than left to be inferred from a zero count.
    assert record["strata_unavailable"] == ["healthy_no_bg"]


def test_only_eligible_recordings_of_the_named_split_are_drawn():
    """Training recordings would put the gate on the data the candidate was fitted to."""
    cohort = pd.concat([
        _cohort(adverse=2, healthy_bg=2),
        pd.DataFrame([
            _recording("TRAIN-01", outcome=1, split="train"),
            _recording("TEST-01", outcome=1, split="test"),
            _recording("INELIGIBLE-01", outcome=1, eligible=False),
            _recording("EXCLUDED-01", outcome=None, reason=data.EXCLUDED_NO_CLASS),
        ]),
    ], ignore_index=True)

    record = evaluate.gate_subset(cohort, size=20, seed=42)

    assert record["n_recordings"] == 4
    assert all(guid.startswith(("ADV-", "HBG-")) for guid in record["guids"])


def test_a_split_with_no_usable_outcome_refuses_rather_than_returning_nothing():
    """An empty subset would make every candidate eligible with nothing checked."""
    cohort = pd.DataFrame([_recording("VAL-01", outcome=None, reason=data.EXCLUDED_NO_CLASS)])

    with pytest.raises(evaluate.GateEvaluationError, match="usable binary outcome"):
        evaluate.gate_subset(cohort, size=4, seed=42)


def test_the_subset_round_trips_through_a_run_directory(tmp_path):
    record = evaluate.gate_subset(_cohort(adverse=2, healthy_bg=2), size=4, seed=42)
    evaluate.save_gate_subset(record, tmp_path)

    assert evaluate.load_gate_subset(tmp_path) == record


def test_a_missing_subset_is_never_silently_redrawn(tmp_path):
    with pytest.raises(FileNotFoundError, match="drawn once"):
        evaluate.load_gate_subset(tmp_path)


def test_the_outcome_map_skips_recordings_with_no_label():
    cohort = pd.concat([
        _cohort(adverse=1, healthy_bg=1),
        pd.DataFrame([_recording("BAD-01", outcome=None, reason=data.EXCLUDED_NO_CLASS)]),
    ], ignore_index=True)

    mapping = evaluate.outcome_map(cohort)

    assert mapping == {"ADV-00": 1, "HBG-00": 0}


# =============================================================================
# The forecast gate
# =============================================================================
def test_an_unchanged_model_passes_every_gate():
    """The identity case: a candidate equal to the baseline cannot fail its own tolerances."""
    baseline = _record()

    result = _decide(baseline, dict(baseline))

    assert result.passed
    assert result.reasons == []
    assert result.record["mse_full_increase"] == pytest.approx(0.0)
    assert result.record["delta_mu_sat_increase_pp"] == pytest.approx(0.0)


def test_the_forecast_boundary_is_inclusive():
    """Stated on binary-exact values, so the assertion is about the comparison and not about
    whether 1.1 / 1.0 - 1 rounds above or below a tenth."""
    baseline = _record(mse_full=2.0, mse_full_healthy=2.0)
    at_the_edge = _record(mse_full=2.25, mse_full_healthy=2.25)

    result = _decide(baseline, at_the_edge, forecast_mse_max_increase=0.125)

    assert result.passed
    assert result.record["mse_full_increase"] == 0.125


def test_a_forecast_past_the_declared_tolerance_fails():
    baseline = _record(mse_full=1.0, mse_full_healthy=1.0)

    assert _decide(baseline, _record(mse_full=1.05, mse_full_healthy=1.05)).passed
    failed = _decide(baseline, _record(mse_full=1.2, mse_full_healthy=1.2))

    assert not failed.passed
    assert "forecast" in failed.reasons[0]
    assert failed.record["mse_full_increase"] == pytest.approx(0.2)


def test_a_forecast_that_improved_passes():
    """The gate is a ceiling on degradation, not a band around the baseline."""
    result = _decide(_record(mse_full=1.0), _record(mse_full=0.5, mse_full_healthy=0.5))

    assert result.passed
    assert result.record["mse_full_increase"] == pytest.approx(-0.5)


def test_a_zero_baseline_falls_back_to_a_named_rule_instead_of_dividing():
    """A relative tolerance is undefined at zero, and an infinity is not a gate decision."""
    baseline = _record(mse_full=0.0, mse_full_healthy=0.0)

    failed = _decide(baseline, _record(mse_full=1e-9, mse_full_healthy=1e-9))
    assert not failed.passed
    assert failed.record["rule"] == evaluate.ZERO_BASELINE_RULE
    assert failed.record["mse_full_increase"] is None

    passed = _decide(baseline, _record(mse_full=0.0, mse_full_healthy=0.0))
    assert passed.passed
    assert passed.record["rule"] == evaluate.ZERO_BASELINE_RULE


def test_a_non_finite_baseline_is_a_broken_measurement_not_a_failed_candidate():
    with pytest.raises(evaluate.GateEvaluationError, match="nothing to gate against"):
        _decide(_record(mse_full=float("nan")), _record())


# =============================================================================
# Saturation, finiteness and collapse
# =============================================================================
def test_saturation_is_gated_in_percentage_points_above_the_baseline():
    baseline = _record(delta_mu_sat_pp=2.0)

    assert _decide(baseline, _record(delta_mu_sat_pp=7.0)).passed
    failed = _decide(baseline, _record(delta_mu_sat_pp=7.5))
    assert not failed.passed
    assert "saturation" in failed.reasons[0]
    assert failed.record["delta_mu_sat_increase_pp"] == pytest.approx(5.5)


def test_a_non_finite_candidate_fails_rather_than_comparing_as_false():
    result = _decide(_record(), _record(finite=False))

    assert not result.passed
    assert any("finiteness" in reason for reason in result.reasons)


def test_a_collapsed_latent_fails_even_when_its_forecast_is_perfect():
    """Discrimination cannot come from a latent that no longer separates recordings."""
    result = _decide(_record(), _record(mse_full=0.1, n_latent_coordinates_varying=0))

    assert not result.passed
    assert any("collapse" in reason for reason in result.reasons)


def test_every_failed_gate_is_reported_not_only_the_first():
    result = _decide(
        _record(),
        _record(mse_full=2.0, mse_full_healthy=2.0, delta_mu_sat_pp=90.0, finite=False),
    )

    assert not result.passed
    assert len(result.reasons) == 3


# =============================================================================
# What is reported rather than gated
# =============================================================================
def test_healthy_only_drift_warns_without_blocking_selection():
    """The protocol declares one forecast tolerance; the smaller stratum is a disclosure."""
    result = _decide(
        _record(mse_full=1.0, mse_full_healthy=1.0),
        _record(mse_full=1.05, mse_full_healthy=1.40),
    )

    assert result.passed
    assert result.reasons == []
    assert any("healthy-only" in warning for warning in result.warnings)
    assert result.record["mse_full_healthy_increase"] == pytest.approx(0.40)


def test_a_subset_with_no_healthy_recording_says_so_rather_than_reporting_a_zero():
    result = _decide(
        _record(mse_full_healthy=None), _record(mse_full_healthy=None)
    )

    assert result.passed
    assert any("not measured" in warning for warning in result.warnings)


# =============================================================================
# The support the comparison is paired by
# =============================================================================
def test_two_readings_over_different_anchors_are_refused_rather_than_compared():
    """A change of population must never be reported as a change of forecast error."""
    with pytest.raises(evaluate.GateEvaluationError, match="different anchors"):
        _decide(_record(), _record(support_digest="0123456789abcdef"))


def test_the_draw_seed_is_keyed_to_the_examples_and_not_to_the_batch_order():
    """Common random numbers: the same segments draw the same noise in either pass."""
    guids, epochs = ["A", "B"], [-3600.0, -2400.0]

    assert evaluate._sample_seed(42, guids, epochs) == evaluate._sample_seed(42, guids, epochs)
    assert evaluate._sample_seed(42, guids, epochs) != evaluate._sample_seed(43, guids, epochs)
    assert evaluate._sample_seed(42, guids, epochs) != evaluate._sample_seed(
        42, ["A", "C"], epochs
    )
    assert evaluate._sample_seed(42, guids, epochs) != evaluate._sample_seed(
        42, guids, [-3600.0, -2401.0]
    )
    assert 0 <= evaluate._sample_seed(42, guids, epochs) < 2**63


# =============================================================================
# The Monte Carlo convergence check
# =============================================================================
def test_the_convergence_check_reports_the_movement_of_the_gap_it_is_read_from():
    small = {"support_digest": "d", "mc_draws": 8, "nll_full": -10.0, "nll_base": -8.0}
    large = {"support_digest": "d", "mc_draws": 64, "nll_full": -10.5, "nll_base": -8.2}

    record = evaluate.nll_convergence(small, large)

    assert record["nll_full_movement"] == pytest.approx(-0.5)
    assert record["nll_gap_small"] == pytest.approx(-2.0)
    assert record["nll_gap_large"] == pytest.approx(-2.3)
    assert record["nll_gap_movement"] == pytest.approx(-0.3)


def test_the_convergence_check_refuses_two_different_supports():
    with pytest.raises(evaluate.GateEvaluationError, match="different anchors"):
        evaluate.nll_convergence(
            {"support_digest": "a", "nll_full": 1.0, "nll_base": 1.0},
            {"support_digest": "b", "nll_full": 1.0, "nll_base": 1.0},
        )


# =============================================================================
# The relative-increase helper the two forecast paths share
# =============================================================================
def test_the_relative_increase_uses_the_baseline_magnitude():
    assert evaluate._relative_increase(2.0, 3.0) == pytest.approx(0.5)
    assert evaluate._relative_increase(2.0, 1.0) == pytest.approx(-0.5)
    assert evaluate._relative_increase(0.0, 1.0) is None
    assert np.isfinite(evaluate._relative_increase(1e-12, 1e-12))
