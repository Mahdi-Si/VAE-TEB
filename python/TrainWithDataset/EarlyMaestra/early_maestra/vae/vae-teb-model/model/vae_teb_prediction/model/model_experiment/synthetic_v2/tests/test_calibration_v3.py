r"""S5-T01/T02/T03: the ``calibration`` stage and its TE stratification.

The stage is a *bridge* to the kernels shipped in ``model/vae_teb_prediction/testing/``, so what
needs pinning is not the arithmetic of NLL or CRPS -- those are tested there -- but the four
seams where this pipeline could silently diverge from them:

1. :func:`coverage_by_horizon` is the one reduction ``testing/`` does not export (its
   ``compute_interval_coverage`` pools the horizon axis away). Averaging our per-horizon
   coverage over :math:`h` must reproduce the shipped ``coverage`` **exactly**; that identity
   is the only thing anchoring the derivation.
2. The scalar summary folds into ``metrics.json`` under ``calibration_predictive``, leaving the
   pre-existing ``calibration`` block -- the $\bar K = \alpha + \gamma\,\mathrm{TE}$ fit --
   untouched. The name collision is a real trap.
3. A checkpoint trained with a fixed ``sigma_obs`` has no ``logvar_full``; the shipped analysis
   must degrade to an ``error`` entry, not raise.
4. Coverage on the $\mathrm{TE}_{\mathrm{inj}} = 0$ null cells is gated against nominal. It is
   asserted here on a **synthetic, perfectly-calibrated** stub, never on a pilot checkpoint:
   at 400 steps the source pathway has not switched on (Sprint 3/4), so the pilot's real
   coverage is a training-progress readout and is recorded in the spec, not asserted.
"""
from __future__ import annotations

import json
import types

import numpy as np
import pytest
import torch

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import calibration_v3 as cal
from model.vae_teb_prediction.testing.metrics import compute_interval_coverage

_B, _T, _H, _C = 4, 20, 3, 6
_WARMUP = 4
_LEVELS = (0.5, 0.8, 0.9, 0.95)


def _triplet(seed: int = 0, sigma: float = 1.0):
    r"""A ``(mu_full, logvar_full, y_plus)`` triple with a known observation scale."""
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(_B, _T, _H, _C, generator=g)
    logvar = torch.full((_B, _T, _H, _C), 2.0 * float(np.log(sigma)))
    noise = torch.randn(_B, _T - _H, _H, _C, generator=g) * sigma
    y_plus = mu[:, : _T - _H] + noise
    return mu, logvar, y_plus


# ---------------------------------------------------------------------------
# 1. The derived kernel is pinned to the shipped one.
# ---------------------------------------------------------------------------
def test_coverage_by_horizon_repools_to_the_shipped_kernel() -> None:
    r"""Averaging our $(B, H_d, n_{\rm lv})$ coverage over $h$ reproduces ``coverage`` exactly."""
    mu, logvar, y_plus = _triplet()
    ours = cal.coverage_by_horizon(mu, logvar, y_plus, _WARMUP, _H, levels=_LEVELS)
    assert ours.shape == (_B, _H, len(_LEVELS))

    shipped = compute_interval_coverage(
        mu, logvar, y_plus, _WARMUP, _H, levels=_LEVELS
    )["coverage"]
    assert torch.allclose(ours.mean(dim=1), shipped, atol=1e-6), (
        "the per-horizon derivation has drifted from testing.metrics"
    )


def test_coverage_by_horizon_is_calibrated_on_gaussian_draws() -> None:
    r"""With the truth drawn from the forecast itself, coverage tracks nominal."""
    mu, logvar, y_plus = _triplet(seed=3, sigma=1.0)
    cov = cal.coverage_by_horizon(mu, logvar, y_plus, _WARMUP, _H, levels=_LEVELS)
    pooled = cov.mean(dim=(0, 1))
    for j, level in enumerate(_LEVELS):
        assert abs(float(pooled[j]) - level) < 0.06, f"level {level} miscovered"


def test_coverage_by_horizon_detects_overconfidence() -> None:
    r"""Halving the predicted $\sigma$ must show up as systematic under-coverage."""
    mu, logvar, y_plus = _triplet(seed=5, sigma=1.0)
    cov = cal.coverage_by_horizon(
        mu, logvar - 2.0 * float(np.log(2.0)), y_plus, _WARMUP, _H, levels=_LEVELS
    )
    pooled = cov.mean(dim=(0, 1))
    for j, level in enumerate(_LEVELS):
        assert float(pooled[j]) < level, "an over-confident sigma still covered at nominal"


def test_coverage_by_horizon_rejects_bad_levels() -> None:
    mu, logvar, y_plus = _triplet()
    with pytest.raises(ValueError):
        cal.coverage_by_horizon(mu, logvar, y_plus, _WARMUP, _H, levels=(0.0, 0.9))


def test_empty_valid_window_returns_zeros() -> None:
    r"""A warm-up that swallows every anchor yields zeros, not a crash."""
    mu, logvar, y_plus = _triplet()
    cov = cal.coverage_by_horizon(mu, logvar, y_plus, _T, _T, levels=_LEVELS)
    assert cov.shape == (_B, _T, len(_LEVELS))
    assert torch.all(cov == 0.0)


# ---------------------------------------------------------------------------
# 2. metrics.json folding leaves the gamma-vs-TE block alone.
# ---------------------------------------------------------------------------
def test_fold_metrics_json_preserves_the_calibration_key(tmp_path) -> None:
    r"""``calibration_predictive`` is added; ``calibration`` survives byte-for-byte."""
    original = {
        "calibration": {"gamma_inj": 0.31, "alpha_inj": 0.02, "kld_variants": {"kbar": {}}},
        "prediction_controls": {"overall": {"feat_loss": 1.0}},
    }
    (tmp_path / "metrics.json").write_text(json.dumps(original), encoding="utf-8")

    existed = cal._fold_metrics_json(
        tmp_path, "calibration_predictive", {"nll_mean": np.float64(1.25)}
    )
    assert existed is True

    data = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert data["calibration"] == original["calibration"]
    assert data["prediction_controls"] == original["prediction_controls"]
    assert data["calibration_predictive"] == {"nll_mean": 1.25}


def test_fold_metrics_json_creates_when_absent(tmp_path) -> None:
    r"""An ungraded split gets a new file carrying only this block, plus a warning."""
    assert cal._fold_metrics_json(tmp_path, "calibration_predictive", {"a": 1}) is False
    data = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert data == {"calibration_predictive": {"a": 1}}


# ---------------------------------------------------------------------------
# 3/4. The stratified pass, over a stub runner.
# ---------------------------------------------------------------------------
class _StubModel:
    r"""Emits a Gaussian forecast whose spread is honest, or deliberately over-confident."""

    def __init__(self, *, emit_logvar: bool = True, sigma_scale: float = 1.0) -> None:
        self.emit_logvar = emit_logvar
        self.sigma_scale = float(sigma_scale)

    def eval(self):
        return self

    def __call__(self, **kwargs):
        mu = torch.zeros(_B, _T, _H, _C)
        out = {"mu_full": mu}
        if self.emit_logvar:
            out["logvar_full"] = torch.full(
                (_B, _T, _H, _C), 2.0 * float(np.log(self.sigma_scale))
            )
        return out


class _StubRunner:
    r"""The slice of ``TestRunner`` that :func:`collect_calibration_by_te` actually touches."""

    def __init__(self, batches, model) -> None:
        self._batches = batches
        self.model = model
        self.warmup_steps = _WARMUP
        self.horizon = _H

    def inference_mode(self):
        import contextlib

        return contextlib.nullcontext()

    def iter_batches(self, loader, max_samples=None):
        seen = 0
        for b in self._batches:
            if max_samples is not None and seen >= max_samples:
                break
            yield b
            seen += int(b.fhr_st.shape[0])

    def forward(self, batch):
        return self.model()

    def build_future_target(self, batch):
        return batch.y_plus


def _make_batch(te_level: float, *, seed: int, sigma: float, delay: int = 3):
    r"""One stub batch whose targets are exact Gaussian draws around ``mu = 0``.

    ``delay`` defaults below ``_WARMUP + 1`` so the two clean-window conventions coincide,
    mirroring the real benchmark (``warmup = 30``, :math:`D \le 20`).
    """
    g = torch.Generator().manual_seed(seed)
    y_plus = torch.randn(_B, _T - _H, _H, _C, generator=g) * sigma
    return types.SimpleNamespace(
        fhr_st=torch.zeros(_B, _T, 1),
        y_plus=y_plus,
        te_true=torch.full((_B,), float(te_level)),
        cell_id=torch.full((_B,), int(te_level * 10)),
        delay=torch.full((_B,), int(delay)),
    )


def test_snap_to_grid_pools_the_lags_of_one_nominal_level() -> None:
    r"""``te_true`` carries the *realised* TE, which differs per cell within a nominal level.

    ``build_dataset_v2`` stamps ``te_block_realised``, so the three cells nominally at
    :math:`\mathrm{TE}_{\mathrm{inj}} = 0.5` (one per lag) carry ``0.499121`` / ``0.499343`` /
    ``0.500882``. Grouping on the raw float would split every level by lag and defeat the
    stratification.
    """
    grid = [0.0, 0.5, 1.0, 2.0, 3.0]
    for realised in (0.499121, 0.499343, 0.500882):
        assert cal._snap_to_grid(realised, grid) == 0.5
    assert cal._snap_to_grid(0.0, grid) == 0.0
    assert cal._snap_to_grid(1.99764, grid) == 2.0
    assert cal._snap_to_grid(2.9987, grid) == 3.0
    # Without a grid, fall back to coarse rounding rather than splitting on float noise.
    assert cal._snap_to_grid(0.499121, None) == 0.5


def test_te_grid_read_from_config() -> None:
    config = {"benchmarks": {"G1_raw": {"mix": {"target_te_grid": [0.0, 0.5, 3.0]}}}}
    assert cal._te_grid(config, "G1_raw") == [0.0, 0.5, 3.0]
    assert cal._te_grid({}, "G1_raw") is None


def test_stratified_pass_groups_realised_te_onto_the_nominal_grid() -> None:
    r"""Three cells at nominally 0.5 (different lags) must pool into one row, not three."""
    batches = [
        _make_batch(0.499121, seed=1, sigma=1.0),
        _make_batch(0.499343, seed=2, sigma=1.0),
        _make_batch(0.500882, seed=3, sigma=1.0),
        _make_batch(0.0, seed=4, sigma=1.0),
    ]
    runner = _StubRunner(batches, _StubModel())
    res = cal.collect_calibration_by_te(
        runner, None, None, levels=_LEVELS, te_grid=[0.0, 0.5, 1.0, 2.0, 3.0]
    )
    assert sorted(res["per_te_summary"]) == ["0", "0.5"]
    assert res["per_te_summary"]["0.5"]["n_samples"] == 3 * _B
    # Provenance: the realised mean is kept even though the key is the nominal level.
    assert res["per_te_summary"]["0.5"]["te_realised_mean"] == pytest.approx(0.499782, abs=1e-5)


def test_stratified_table_has_the_full_key_grid() -> None:
    r"""One row per ``(te_level, horizon, level)``, and the TE levels are the grouping keys."""
    batches = [
        _make_batch(0.0, seed=1, sigma=1.0),
        _make_batch(2.0, seed=2, sigma=1.0),
        _make_batch(0.0, seed=3, sigma=1.0),
    ]
    runner = _StubRunner(batches, _StubModel())
    res = cal.collect_calibration_by_te(runner, None, None, levels=_LEVELS)

    df = res["by_te"]
    assert set(df.columns) >= {"te_level", "horizon", "level", "coverage", "nll", "crps"}
    assert sorted(df["te_level"].unique()) == [0.0, 2.0]
    assert len(df) == 2 * _H * len(_LEVELS)
    assert not df.duplicated(subset=["te_level", "horizon", "level"]).any()

    # 8 samples at TE=0 (two batches), 4 at TE=2.
    assert res["per_te_summary"]["0"]["n_samples"] == 2 * _B
    assert res["per_te_summary"]["2"]["n_samples"] == _B
    assert res["n_samples"] == 3 * _B
    assert res["warmup_ok"] is True


def test_stratified_coverage_is_nominal_on_the_null_cells() -> None:
    r"""On a perfectly-calibrated stub, coverage at 0.9 sits at nominal on the TE=0 cells.

    Asserted on synthetic draws with a *known* answer, not on a pilot checkpoint: the 400-step
    pilot has not switched its source pathway on, so its coverage is a training-progress
    readout (recorded in the spec's Section 11), not a gate.
    """
    batches = [_make_batch(0.0, seed=s, sigma=1.0) for s in range(12)]
    runner = _StubRunner(batches, _StubModel(sigma_scale=1.0))
    res = cal.collect_calibration_by_te(runner, None, None, levels=_LEVELS)

    gate = cal._te_null_coverage_gate(res, 0.9, 0.10)
    assert gate is not None
    assert gate["nominal"] == 0.9
    assert gate["pass"] is True, f"coverage {gate['empirical']:.3f} off nominal"
    assert gate["abs_error"] < 0.10


def test_null_cell_gate_fails_on_an_overconfident_variance() -> None:
    r"""Halve the predicted $\sigma$: the gate must catch it."""
    batches = [_make_batch(0.0, seed=s, sigma=1.0) for s in range(12)]
    runner = _StubRunner(batches, _StubModel(sigma_scale=0.5))
    res = cal.collect_calibration_by_te(runner, None, None, levels=_LEVELS)

    gate = cal._te_null_coverage_gate(res, 0.9, 0.05)
    assert gate is not None and gate["pass"] is False
    assert gate["empirical"] < 0.9


def test_null_cell_gate_absent_without_a_null_cell() -> None:
    batches = [_make_batch(1.0, seed=1, sigma=1.0)]
    runner = _StubRunner(batches, _StubModel())
    res = cal.collect_calibration_by_te(runner, None, None, levels=_LEVELS)
    assert cal._te_null_coverage_gate(res, 0.9, 0.1) is None


def test_missing_logvar_raises_for_the_shipped_skip_path() -> None:
    r"""A fixed-``sigma_obs`` checkpoint raises here; the shipped analysis turns it into ``error``."""
    runner = _StubRunner([_make_batch(0.0, seed=1, sigma=1.0)], _StubModel(emit_logvar=False))
    with pytest.raises(RuntimeError, match="logvar_full"):
        cal.collect_calibration_by_te(runner, None, None, levels=_LEVELS)


def test_max_samples_is_honoured() -> None:
    batches = [_make_batch(0.0, seed=s, sigma=1.0) for s in range(5)]
    runner = _StubRunner(batches, _StubModel())
    res = cal.collect_calibration_by_te(runner, None, _B, levels=_LEVELS)
    assert res["n_samples"] == _B


def test_warmup_mismatch_is_flagged() -> None:
    r"""``warmup < max(D) - 1`` makes the shipped window wider than eval's; say so."""
    batches = [_make_batch(0.0, seed=1, sigma=1.0, delay=_WARMUP + 40)]
    runner = _StubRunner(batches, _StubModel())
    res = cal.collect_calibration_by_te(runner, None, None, levels=_LEVELS)
    assert res["warmup_ok"] is False


# ---------------------------------------------------------------------------
# Stage + section registration.
# ---------------------------------------------------------------------------
def test_stage_is_registered_non_fatal_and_arm_scoped() -> None:
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import run_pipeline_v2 as rp

    spec = rp._STAGE_REGISTRY["calibration"]
    assert spec.model_dependent is True
    assert spec.fatal is False, "a failed calibration fit must never abort a headline run"
    assert spec.run is cal.run_calibration_stage
    assert "calibration" in rp.stage_names()


def test_report_section_renders_na_without_metrics(tmp_path) -> None:
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import final_report_v2 as fr

    ctx = fr.SectionContext(config={}, benchmark="G1_raw", results_dir=tmp_path, metrics=None)
    lines = cal._render_calibration_section(ctx)
    assert any("n/a" in ln for ln in lines)

    ctx_err = fr.SectionContext(
        config={}, benchmark="G1_raw", results_dir=tmp_path,
        metrics={"calibration_predictive": {"error": "no logvar_full"}},
    )
    assert any("no logvar_full" in ln for ln in cal._render_calibration_section(ctx_err))


def test_report_section_renders_the_gate_and_te_table(tmp_path) -> None:
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import final_report_v2 as fr

    metrics = {
        "calibration_predictive": {
            "n_samples": 128, "nll_mean": 1.5, "crps_mean": 0.4, "sharpness_mean": 0.9,
            "constant_sigma": 1.0, "nll_gain_over_constant": 0.05,
            "coverage_90": 0.87, "coverage_90_error": -0.03,
            "null_cell_coverage": {"nominal": 0.9, "empirical": 0.87, "abs_error": 0.03,
                                   "tolerance": 0.1, "pass": True, "n_samples": 32},
            "by_te": {"0": {"n_samples": 32, "nll_mean": 1.4, "crps_mean": 0.4,
                            "sharpness_mean": 0.9, "coverage_90": 0.9},
                      "2": {"n_samples": 32, "nll_mean": 1.6, "crps_mean": 0.5,
                            "sharpness_mean": 0.8, "coverage_90": 0.82}},
        }
    }
    ctx = fr.SectionContext(
        config={}, benchmark="G1_raw", results_dir=tmp_path, metrics=metrics
    )
    text = "\n".join(cal._render_calibration_section(ctx))
    assert "Predictive calibration" in text
    assert "pass" in text
    assert "0.82" in text, "the TE-stratified table did not render"
