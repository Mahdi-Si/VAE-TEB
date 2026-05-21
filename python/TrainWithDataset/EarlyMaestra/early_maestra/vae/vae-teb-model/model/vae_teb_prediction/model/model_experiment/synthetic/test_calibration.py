"""Unit tests for the calibration helpers.

Tests the pure functions in :mod:`calibration` -- slope fit, beta
selector, the benchmark dispatcher, and the CLI override semantics --
without invoking the heavy orchestrator (which trains models and is
exercised by CI smoke / manual runs, not pytest).
"""
from __future__ import annotations

import math

import pytest

from model.vae_teb_prediction.model.model_experiment.synthetic.calibration import (
    _CALIBRATION_BUILDERS,
    _apply_overrides,
    _get_calibration_spec,
    build_g1_calibration_caches,
    build_g2_calibration_caches,
    fit_calibration_slope,
    select_beta_by_calibration,
)


# ----------------------------------------------------------------------
# fit_calibration_slope
# ----------------------------------------------------------------------

def test_slope_recovers_identity_on_y_equals_x():
    """Perfect identity ``k_bar = te_true`` -> alpha=0, gamma=1, R^2=1."""
    table = [(0.5, 0.5), (2.0, 2.0), (4.0, 4.0)]
    fit = fit_calibration_slope(table)
    assert fit["alpha"] == pytest.approx(0.0, abs=1e-9)
    assert fit["gamma"] == pytest.approx(1.0, abs=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
    assert fit["n"] == 3


def test_slope_recovers_affine_relationship():
    """``k_bar = 0.5 + 0.7 * te_true`` -> alpha=0.5, gamma=0.7."""
    xs = [0.1, 1.5, 3.0, 4.7]
    alpha_true, gamma_true = 0.5, 0.7
    table = [(x, alpha_true + gamma_true * x) for x in xs]
    fit = fit_calibration_slope(table)
    assert fit["alpha"] == pytest.approx(alpha_true, abs=1e-9)
    assert fit["gamma"] == pytest.approx(gamma_true, abs=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
    assert fit["n"] == 4


def test_slope_handles_three_points_minimum():
    """The headline calibration plan uses n=3 points; OLS must work there."""
    table = [(1.0, 0.8), (2.0, 1.8), (3.0, 2.7)]
    fit = fit_calibration_slope(table)
    # Hand-computed: cov(x,y)/var(x) for these three points.
    assert fit["gamma"] == pytest.approx(0.95, abs=1e-3)
    assert fit["n"] == 3
    assert 0.0 <= fit["r2"] <= 1.0


def test_slope_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least 2"):
        fit_calibration_slope([(1.0, 1.0)])


def test_slope_rejects_constant_te():
    """All te_true coincide -> singular OLS, helpful error."""
    with pytest.raises(ValueError, match="coincide"):
        fit_calibration_slope([(2.0, 0.5), (2.0, 0.8), (2.0, 1.1)])


def test_slope_rejects_malformed_input():
    with pytest.raises(ValueError, match="pairs"):
        fit_calibration_slope([(1.0, 1.0, 99.0), (2.0, 2.0, 99.0)])


# ----------------------------------------------------------------------
# select_beta_by_calibration
# ----------------------------------------------------------------------

def test_selector_picks_gamma_closest_to_one():
    table = [
        {"beta": 1e-4, "alpha": 0.1, "gamma": 0.3, "r2": 0.9},
        {"beta": 1e-3, "alpha": 0.1, "gamma": 0.98, "r2": 0.99},   # winner
        {"beta": 1e-2, "alpha": 0.1, "gamma": 1.5, "r2": 0.9},
    ]
    sel = select_beta_by_calibration(table, alpha_penalty=0.0)
    assert sel["beta"] == pytest.approx(1e-3)
    assert sel["gamma"] == pytest.approx(0.98)


def test_selector_uses_alpha_penalty():
    """Two beta have the same |gamma-1|; the one with smaller |alpha| wins."""
    table = [
        {"beta": 1e-4, "alpha": 0.5, "gamma": 0.8, "r2": 0.99},
        {"beta": 1e-3, "alpha": 0.0, "gamma": 1.2, "r2": 0.99},   # winner
    ]
    sel = select_beta_by_calibration(table, alpha_penalty=0.05)
    assert sel["beta"] == pytest.approx(1e-3)


def test_selector_tie_breaks_on_r2():
    """Identical (alpha, gamma) -> the higher R^2 wins."""
    table = [
        {"beta": 1e-4, "alpha": 0.0, "gamma": 0.9, "r2": 0.80},
        {"beta": 1e-3, "alpha": 0.0, "gamma": 0.9, "r2": 0.95},   # winner
    ]
    sel = select_beta_by_calibration(table, alpha_penalty=0.05)
    assert sel["beta"] == pytest.approx(1e-3)


def test_selector_skips_nonfinite_cells():
    """A cell with NaN gamma must be skipped."""
    table = [
        {"beta": 1e-4, "alpha": float("nan"), "gamma": float("nan"), "r2": float("nan")},
        {"beta": 1e-3, "alpha": 0.0, "gamma": 1.0, "r2": 1.0},
    ]
    sel = select_beta_by_calibration(table)
    assert sel["beta"] == pytest.approx(1e-3)


def test_selector_returns_empty_when_no_valid_cells():
    table = [
        {"beta": 1e-4, "alpha": float("nan"), "gamma": float("nan"), "r2": 0.0},
        {"beta": 1e-3, "alpha": float("inf"), "gamma": float("inf"), "r2": 0.0},
    ]
    sel = select_beta_by_calibration(table)
    assert sel == {}


def test_selector_returns_finite_score_in_rationale():
    """The rationale string mentions the alpha penalty used."""
    table = [
        {"beta": 1e-3, "alpha": 0.0, "gamma": 1.0, "r2": 1.0},
    ]
    sel = select_beta_by_calibration(table, alpha_penalty=0.05)
    assert "alpha" in sel["rationale"]
    assert math.isfinite(sel["score"])


# ----------------------------------------------------------------------
# Benchmark dispatcher
# ----------------------------------------------------------------------


def test_dispatcher_resolves_g1_to_state_space_builder():
    """G1 -> ``build_g1_calibration_caches`` + ``B_y`` knob."""
    spec = _get_calibration_spec("G1")
    assert spec.builder is build_g1_calibration_caches
    assert spec.knob_name == "B_y"
    assert spec.meta_key == "B_y"


def test_dispatcher_resolves_g2_to_arx_builder():
    """G2 -> ``build_g2_calibration_caches`` + ``c`` knob."""
    spec = _get_calibration_spec("G2")
    assert spec.builder is build_g2_calibration_caches
    assert spec.knob_name == "c"
    assert spec.meta_key == "c"


def test_dispatcher_rejects_unknown_benchmark():
    """An unregistered benchmark must raise ``ValueError`` with the valid set."""
    with pytest.raises(ValueError, match="unsupported"):
        _get_calibration_spec("G3")
    with pytest.raises(ValueError, match="unsupported"):
        _get_calibration_spec("nonsense")


def test_dispatcher_registry_covers_known_benchmarks():
    """The registry currently exposes G1 and G2; if the test breaks, update
    the headline figure + docs."""
    assert set(_CALIBRATION_BUILDERS) == {"G1", "G2"}


def test_g1_builder_rejects_g2_config():
    """``build_g1_calibration_caches`` only accepts G1 configs."""
    cfg = {
        "experiment": {"benchmark": "G2"},
        "data": {"M": 4, "oscillators": [[0.99, 0.05]], "target_ar": 0.95,
                 "delays": [60], "sigma2_y": 1.0, "sigma2_eta": 0.01},
        "model": {"horizon": 30},
    }
    with pytest.raises(ValueError, match="G1"):
        build_g1_calibration_caches(cfg, te_per_step_targets=[0.05])


def test_g2_builder_rejects_g1_config():
    """``build_g2_calibration_caches`` only accepts G2 configs."""
    cfg = {
        "experiment": {"benchmark": "G1"},
        "data": {"rho_u": 0.99, "rho_y": 0.95, "sigma2_eta": 1.0,
                 "sigma2_eps": 1.0, "delay": 60},
        "model": {"horizon": 30},
    }
    with pytest.raises(ValueError, match="G2"):
        build_g2_calibration_caches(cfg, te_per_step_targets=[0.05])


# ----------------------------------------------------------------------
# CLI overrides
# ----------------------------------------------------------------------


def test_apply_overrides_benchmark_updates_both_keys():
    """``--benchmark G2`` updates ``experiment.benchmark`` AND
    ``calibration.benchmark`` so the orchestrator and the active block agree."""
    cfg = {
        "experiment": {"benchmark": "G1", "seed": 0},
        "runtime": {"device": "auto"},
        "calibration": {"benchmark": "G1", "tag_prefix": "G1_te"},
    }
    _apply_overrides(cfg, {"benchmark": "G2"})
    assert cfg["experiment"]["benchmark"] == "G2"
    assert cfg["calibration"]["benchmark"] == "G2"
    # The G1 tag_prefix must be dropped so G2 picks up its default ("G2_te").
    assert "tag_prefix" not in cfg["calibration"]


def test_apply_overrides_benchmark_creates_calibration_block():
    """Missing ``calibration`` block must be created when the override fires."""
    cfg = {
        "experiment": {"benchmark": "G1", "seed": 0},
        "runtime": {"device": "auto"},
    }
    _apply_overrides(cfg, {"benchmark": "G2"})
    assert cfg["calibration"]["benchmark"] == "G2"


def test_apply_overrides_passes_through_seed_and_device():
    """Non-benchmark overrides still work."""
    cfg = {
        "experiment": {"benchmark": "G1", "seed": 0},
        "runtime": {"device": "auto"},
    }
    _apply_overrides(cfg, {"seed": 42, "device": "cpu"})
    assert cfg["experiment"]["seed"] == 42
    assert cfg["runtime"]["device"] == "cpu"


def test_apply_overrides_no_op_when_none():
    """``None`` values must leave the config untouched."""
    cfg = {
        "experiment": {"benchmark": "G1", "seed": 0},
        "runtime": {"device": "auto"},
        "calibration": {"benchmark": "G1", "tag_prefix": "G1_te"},
    }
    _apply_overrides(cfg, {"benchmark": None, "seed": None, "device": None})
    assert cfg["experiment"]["benchmark"] == "G1"
    assert cfg["calibration"]["benchmark"] == "G1"
    assert cfg["calibration"]["tag_prefix"] == "G1_te"
