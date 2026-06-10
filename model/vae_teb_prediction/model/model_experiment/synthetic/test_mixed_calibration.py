r"""Pure-logic unit tests for the mixed-population $\beta$-calibration pipeline.

These tests deliberately avoid building datasets or training (so they run fast
and dodge the Windows ``spawn`` + ``ProcessPoolExecutor`` build path): they
exercise the grid/tag resolution, the per-$\beta$ table assembly + selection
(``mixed_calibration``), the null-subtracted / per-$M$ calibration helpers
(``mixed_eval``), and the ``gpu_pool`` ``mix_beta`` cell enumeration with
``build=False``.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from model.vae_teb_prediction.model.model_experiment.synthetic import (
    gpu_pool as gp,
    mixed_calibration as mc,
    mixed_eval as me,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (
    load_config,
    resolve_active_benchmark,
)

_CFG = Path(__file__).resolve().parent / "config_synth.yaml"


@pytest.fixture(scope="module")
def g1mix_config():
    cfg = load_config(_CFG)
    cfg["experiment"]["benchmark"] = "G1_mix"
    resolve_active_benchmark(cfg)
    return cfg


# -----------------------------------------------------------------------------
# Grid + tag helpers
# -----------------------------------------------------------------------------

def test_resolve_beta_grid_fallback_chain():
    base = {"mix_calibration": {}, "calibration": {}, "beta_sweep": {}}
    # beta_sweep.grid only
    cfg = deepcopy(base); cfg["beta_sweep"]["grid"] = [0.1, 1.0]
    assert mc._resolve_beta_grid(cfg) == [0.1, 1.0]
    # calibration.beta_grid overrides beta_sweep.grid
    cfg = deepcopy(base)
    cfg["beta_sweep"]["grid"] = [0.1]; cfg["calibration"]["beta_grid"] = [0.2, 2.0]
    assert mc._resolve_beta_grid(cfg) == [0.2, 2.0]
    # mix_calibration.beta_grid wins over both
    cfg = deepcopy(base)
    cfg["beta_sweep"]["grid"] = [0.1]; cfg["calibration"]["beta_grid"] = [0.2]
    cfg["mix_calibration"]["beta_grid"] = [1.0, 3.0]
    assert mc._resolve_beta_grid(cfg) == [1.0, 3.0]
    # nothing set -> empty
    assert mc._resolve_beta_grid({}) == []


def test_beta_run_tag_format():
    assert mc._beta_run_tag(0.001) == "mixed_calibration/beta_1.0e-03"
    assert mc._beta_run_tag(3.0) == "mixed_calibration/beta_3.0e+00"


def test_config_has_nat_scale_beta_grid(g1mix_config):
    grid = mc._resolve_beta_grid(g1mix_config)
    assert grid, "G1_mix should resolve a non-empty beta grid"
    assert max(grid) >= 1.0, "nat-scale calibration needs beta up to O(1)"


# -----------------------------------------------------------------------------
# Per-beta table assembly + selection
# -----------------------------------------------------------------------------

def _synthetic_metrics(beta, gamma_by_M, alpha=0.05):
    by_M = {str(m): {"alpha": alpha, "gamma": g, "r2": 0.98, "n": 3}
            for m, g in gamma_by_M.items()}
    overall_g = float(np.mean(list(gamma_by_M.values())))
    return {"calibration": {"in_mix": {
        "overall": {"alpha": alpha, "gamma": overall_g, "r2": 0.98, "n": 9},
        "by_M": by_M,
        "by_band": {"mid": {"alpha": alpha, "gamma": overall_g, "r2": 0.98, "n": 3}},
    }}}


def test_fit_per_beta_calibration_shapes():
    mbb = {
        0.01: _synthetic_metrics(0.01, {8: 2.0, 16: 2.2, 32: 2.5}),
        0.1:  _synthetic_metrics(0.1,  {8: 1.0, 16: 1.05, 32: 0.95}),
        1.0:  _synthetic_metrics(1.0,  {8: 0.3, 16: 0.2, 32: 0.1}),
    }
    tables = mc.fit_per_beta_calibration(mbb)
    assert set(tables["per_M"].keys()) == {"8", "16", "32"}
    assert len(tables["overall"]) == 3
    assert len(tables["pooled_M"]) == 3
    # each per-M table has one row per beta
    for rows in tables["per_M"].values():
        assert len(rows) == 3
    # pooled score is smallest at beta=0.1 (gammas closest to 1)
    pooled = {r["beta"]: r["mean_abs_gamma_dev"] for r in tables["pooled_M"]}
    assert pooled[0.1] < pooled[0.01] and pooled[0.1] < pooled[1.0]


def test_select_beta_picks_best_calibrated():
    mbb = {
        0.01: _synthetic_metrics(0.01, {8: 2.0, 16: 2.2, 32: 2.5}),
        0.1:  _synthetic_metrics(0.1,  {8: 1.0, 16: 1.05, 32: 0.95}),
        1.0:  _synthetic_metrics(1.0,  {8: 0.3, 16: 0.2, 32: 0.1}),
    }
    tables = mc.fit_per_beta_calibration(mbb)
    sel = mc.select_beta(tables, alpha_penalty=0.05)
    assert sel["beta_star"] == 0.1
    assert sel["primary"] == "pooled_M"
    # per-M selections present for every M
    assert set(sel["selected_by_M"].keys()) == {"8", "16", "32"}


def test_select_beta_masking_robustness():
    # A pooled SLOPE mean would call (gamma=0.5, gamma=1.5) "perfect" (mean 1.0);
    # the per-M score must reject it in favour of a genuinely calibrated beta.
    mbb = {
        0.1: _synthetic_metrics(0.1, {8: 0.5, 16: 1.5}),     # mean gamma == 1.0 but both off
        1.0: _synthetic_metrics(1.0, {8: 0.95, 16: 1.05}),   # both close to 1
    }
    tables = mc.fit_per_beta_calibration(mbb)
    sel = mc.select_beta(tables, alpha_penalty=0.05)
    assert sel["beta_star"] == 1.0


# -----------------------------------------------------------------------------
# mixed_eval calibration helpers (null-subtraction + per-M primary)
# -----------------------------------------------------------------------------

def _toy_arrs():
    cell_ids, kbar, kbar_shuf, te, M = [], [], [], [], []
    for cid, (tval, base) in enumerate([(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]):
        for _ in range(4):
            cell_ids.append(cid)
            kbar.append(0.06 + base)       # floor 0.06 + true signal
            kbar_shuf.append(0.06)         # shuffled -> only the floor
            te.append(tval)
            M.append(8)
    return {
        "cell_id": np.array(cell_ids), "kbar": np.array(kbar, float),
        "kbar_shuffle": np.array(kbar_shuf, float),
        "te_true": np.array(te, float), "M": np.array(M),
    }, {c: {"M": 8, "band": "mid"} for c in range(3)}


def test_nullsub_drives_intercept_to_zero():
    arrs, cells = _toy_arrs()
    raw = me.fit_calibration_slices(arrs, cells)
    nsub = me.fit_calibration_slices_nullsub(arrs, cells, control="shuffle")
    assert raw["overall"]["alpha"] > 0.05
    assert abs(nsub["overall"]["alpha"]) < abs(raw["overall"]["alpha"])
    # gamma essentially unchanged by the constant-floor subtraction
    assert abs(nsub["overall"]["gamma"] - raw["overall"]["gamma"]) < 0.05


def test_nullsub_absent_control_returns_empty():
    arrs, cells = _toy_arrs()
    arrs.pop("kbar_shuffle")
    assert me.fit_calibration_slices_nullsub(arrs, cells, control="shuffle") == {}


def test_calibration_primary_summary():
    arrs, cells = _toy_arrs()
    raw = me.fit_calibration_slices(arrs, cells)
    prim = me.calibration_primary_summary(raw, gamma_tol=0.2)
    assert prim["n_M"] == 1 and "8" in prim["gamma_by_M"]
    assert 0.0 <= prim["frac_M_calibrated"] <= 1.0
    assert "mean_gamma" in prim and "gamma_spread" in prim


# -----------------------------------------------------------------------------
# gpu_pool mix_beta enumeration (build=False, no training)
# -----------------------------------------------------------------------------

def test_cells_mix_beta_enumeration(g1mix_config):
    betas = mc._resolve_beta_grid(g1mix_config)
    cells = gp._cells_mix_beta(g1mix_config, build=False)
    assert len(cells) == len(betas)
    tag = str(g1mix_config["experiment"]["tag"])
    seen_betas = []
    for cell in cells:
        assert cell.benchmark == "G1_mix"
        assert cell.data_tag == tag           # all share the one pool
        assert cell.run_tag.startswith("mixed_calibration/beta_")
        assert cell.patches["loss.likelihood"] == "gaussian_nll"
        assert cell.patches["loss.sigma_obs"] == "learned"
        seen_betas.append(cell.patches["loss.kld_beta"])
    assert sorted(seen_betas) == sorted(float(b) for b in betas)


def test_mix_beta_in_registries():
    assert "mix_beta" in gp._VALID_MODES
    assert "mix_beta" in gp._ENUMERATORS
