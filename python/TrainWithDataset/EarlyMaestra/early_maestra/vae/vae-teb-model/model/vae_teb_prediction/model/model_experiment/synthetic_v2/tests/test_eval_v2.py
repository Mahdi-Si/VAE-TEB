r"""Tests for the Sprint 6 evaluation gates (S6-T01..T05).

Covers :mod:`eval_v2`'s evaluation layer: the per-sample $\bar K$ clean-window
collection (S6-T01), the $\gamma$-calibration vs $\mathrm{TE}_{\mathrm{inj}}$ and
$\mathrm{TE}_{\mathrm{scat}}$ (S6-T02), the attention lag recovery (S6-T03), the
null controls + ``metrics.json`` assembly (S6-T04), and an end-to-end integration
smoke that composes build -> r0_realizability -> train -> eval -> report on a tiny
real grid (S6-T05).

Design: the calibration / lag / null unit gates use a lightweight **stub model**
returning controlled forward outputs (``kld_per_t`` / ``te_lag_map`` /
``attn_weights``) plus a tiny fake loader, so they run fast on CPU with no real
transform. The ``run_eval`` metrics test builds a **tiny fixture cache** (random
normalised features) + a tiny checkpoint. Only the ``e2e`` test runs the real
scattering transform (a handful of samples on a shared adapter). Run with the
project interpreter ``.venv/Scripts/python.exe``. See
``SYNTHETIC_V2_SPEC_AND_SPRINTS.md`` Sprint 6.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import yaml

# Force the repo root ahead of the sibling ``model/vae_teb_prediction`` on ``sys.path``
# so the model / ``train.graph_models_utils`` imports resolve under pytest (the same
# guard ``pl_module_v2`` / ``eval_v2.run_eval`` apply). See test_train_v2.
_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    dataset_v2 as ds2,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    eval_v2,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (  # noqa: E402
    resolve_cache_dir,
)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_synth_v2.yaml"

# Channel counts fixed by the transform / model contract.
_C_FHR_ST, _C_FHR_PH, _C_UP_ST, _C_UP_PH = 43, 44, 43, 58

# Tiny model for the fixture-cache eval test: production channel counts, everything
# else shrunk for a fast CPU run (mirrors test_train_v2's tiny model, T=32).
_T = 32
_HORIZON = 4
_WARMUP = 2
_MAX_LAG = 8
_DELAY = 4

_TINY_MODEL: Dict[str, Any] = {
    "sequence_length": _T,
    "d_model": 16,
    "d_z": 4,
    "horizon": _HORIZON,
    "warmup_period": _WARMUP,
    "c_y": _C_FHR_ST + _C_FHR_PH,   # 87
    "c_u": _C_UP_ST + _C_UP_PH,     # 101
    "use_up_st": True,
    "max_lag": _MAX_LAG,
    "num_heads": 2,
    "d_head": 8,                    # num_heads * d_head == d_model
    "lstm_layers": 1,
    "dropout": 0.0,
    "decoder_hidden": 16,
    "logvar_clamp": [-5.0, 3.0],
    "mu_scale": 5.0,
    "delta_mu_scale": 3.0,
    "latent_stats_momentum": 0.01,
    "use_entmax": False,
    "attention_grad_checkpoint": False,
    "head_structured_latent": False,
    "lag_bias_init": "normal",
    "horizon_depth": 1,
    "horizon_kernel": 3,
    "horizon_film": False,
    "encoder_extra_dilations": [],
}


# ---------------------------------------------------------------------------
# Stub model + fake loader (for the fast kbar / lag gates)
# ---------------------------------------------------------------------------
class _StubModel:
    r"""A minimal stand-in for :class:`SeqVaeLagAttnV1` returning controlled outputs.

    ``kld_per_t`` is a constant ``kld_value`` per step (so the clean-window mean is
    exactly ``kld_value``); ``te_lag_map`` / ``attn_weights`` are one-hot at ``lag_peak``.
    The source stream is ignored, so null-control forwards return the same ``kld``.

    Forecast heads ``mu_full`` / ``mu_base`` are ``(B, T, H_d, C_y)``. By default both are
    zeros (so the prediction gain $\Delta L = L_{\mathrm{base}} - L_{\mathrm{full}} = 0$);
    with ``perfect_full=True`` the full head is the exact future block (a perfect forecast)
    while the baseline stays zero, giving a strictly positive $\Delta L$.
    """

    def __init__(self, T: int, L: int, *, kld_value: float = 1.0, lag_peak: int = 2,
                 horizon: int = _HORIZON, c_y: int = _C_FHR_ST + _C_FHR_PH,
                 perfect_full: bool = False):
        self._T = int(T)
        self._L = int(L)
        self._kld = float(kld_value)
        self._peak = int(lag_peak)
        self._Hd = int(horizon)
        self._C = int(c_y)
        self._perfect = bool(perfect_full)

    def eval(self):
        return self

    def __call__(self, y_st, y_ph, u_stream):
        B = int(y_st.shape[0])
        kld = torch.full((B, self._T), self._kld)
        lagmap = torch.zeros(B, self._T, self._L)
        lagmap[:, :, self._peak] = 1.0
        attn = torch.zeros(B, self._T, 2, self._L)
        attn[:, :, :, self._peak] = 0.5
        mu_base = torch.zeros(B, self._T, self._Hd, self._C)
        mu_full = torch.zeros(B, self._T, self._Hd, self._C)
        if self._perfect:
            # Perfect full forecast: fill mu_full with the true unfolded future block.
            Y = torch.cat([y_st, y_ph], dim=-1)                       # (B, T, C_y)
            T_valid = self._T - self._Hd
            if T_valid > 0:
                Y_plus = Y[:, 1:, :].unfold(1, self._Hd, 1).permute(0, 1, 3, 2)
                mu_full[:, :T_valid] = Y_plus
        return {
            "kld_per_t": kld,
            "te_lag_map": lagmap,
            "attn_weights": attn,
            "warmup_mask": torch.ones(self._T, dtype=torch.bool),
            "mu_base": mu_base,
            "mu_full": mu_full,
        }


def _fake_batch(n: int, *, T: int, cell_id: int, te_inj: float, te_scat: float,
                frac_phi: float, delay: int, seed: int) -> ds2.AttributeDict:
    r"""One batched :class:`AttributeDict` with the native fields + v2 provenance."""
    g = torch.Generator().manual_seed(seed)
    b = ds2.AttributeDict()
    b["fhr_st"] = torch.randn(n, T, _C_FHR_ST, generator=g)
    b["fhr_ph"] = torch.randn(n, T, _C_FHR_PH, generator=g)
    b["up_st"] = torch.randn(n, T, _C_UP_ST, generator=g)
    b["up_ph"] = torch.randn(n, T, _C_UP_PH, generator=g)
    b["weight"] = torch.ones(n, T)
    b["cell_id"] = torch.full((n,), cell_id, dtype=torch.long)
    b["delay"] = torch.full((n,), delay, dtype=torch.long)
    b["te_true"] = torch.full((n,), te_inj, dtype=torch.float32)
    b["te_scat"] = torch.full((n,), te_scat, dtype=torch.float32)
    b["frac_phi"] = torch.full((n,), frac_phi, dtype=torch.float32)
    b["held_out"] = torch.zeros(n, dtype=torch.long)
    return b


# ---------------------------------------------------------------------------
# S6-T01: per-sample K-bar collection
# ---------------------------------------------------------------------------
def test_kbar_clean_window_mean() -> None:
    r"""``_clean_window_mean`` averages ``kld_per_t`` over ``[max(w, D-1), T-H)``."""
    T, warmup, horizon = 20, 3, 5
    # kld[b, t] = t, so the window mean is the mean of the integer window indices.
    kld = torch.arange(T, dtype=torch.float32).unsqueeze(0).repeat(2, 1)
    delay = torch.tensor([2, 9])   # floors: max(3, 1)=3 and max(3, 8)=8; hi = 15
    kbar, valid = eval_v2._clean_window_mean(kld, delay, warmup=warmup, horizon=horizon)
    exp0 = float(np.mean(np.arange(3, 15)))     # window [3, 15)
    exp1 = float(np.mean(np.arange(8, 15)))     # window [8, 15)
    assert kbar[0].item() == pytest.approx(exp0)
    assert kbar[1].item() == pytest.approx(exp1)
    assert valid[0].sum().item() == 12 and valid[1].sum().item() == 7


def test_kbar_collect_grouping_and_keys() -> None:
    r"""``collect_per_sample_kbar`` groups by cell, carries provenance + controls,
    and drops the v1 M/band keys."""
    T, L = 16, _MAX_LAG + 1
    model = _StubModel(T, L, kld_value=2.0, lag_peak=3)
    loader = [
        _fake_batch(4, T=T, cell_id=0, te_inj=0.0, te_scat=-0.1, frac_phi=float("nan"),
                    delay=4, seed=1),
        _fake_batch(3, T=T, cell_id=1, te_inj=2.0, te_scat=1.8, frac_phi=0.9,
                    delay=4, seed=2),
    ]
    arrs = eval_v2.collect_per_sample_kbar(
        model, loader, torch.device("cpu"),
        warmup=_WARMUP, horizon=_HORIZON, controls=["shuffle", "reverse"],
    )
    assert arrs["n"] == 7 and arrs["T"] == T
    # constant kld_value -> every clean-window mean is exactly 2.0
    assert np.allclose(arrs["kbar"], 2.0)
    for key in ("kbar", "te_inj", "te_scat", "frac_phi", "cell_id", "delay",
                "held_out", "kbar_shuffle", "kbar_reverse", "pred_gain", "uplift_rel"):
        assert key in arrs, key
    # v2 drops the v1 grouping axes.
    for absent in ("M", "band_id", "delay_max", "delay_min"):
        assert absent not in arrs
    # mu_full == mu_base (default stub) -> prediction gain is exactly zero.
    assert np.allclose(arrs["pred_gain"], 0.0)
    # grouping: two cells, each with a lag profile peaked at lag 3.
    assert set(arrs["lag_profiles"]) == {0, 1}
    for cid, prof in arrs["lag_profiles"].items():
        assert int(np.argmax(prof)) == 3
    assert arrs["lag_counts"] == {0: 4, 1: 3}
    # per-cell KLD-over-time profiles retained, one (T,) trace per cell.
    assert set(arrs["kbar_over_time"]) == {0, 1}
    for prof in arrs["kbar_over_time"].values():
        assert prof.shape == (T,) and np.allclose(prof, 2.0)


def test_pred_gain_sign() -> None:
    r"""A full forecast that beats the (zero) baseline yields a positive prediction gain."""
    T, L = 16, _MAX_LAG + 1
    model = _StubModel(T, L, kld_value=1.0, lag_peak=3, perfect_full=True)
    loader = [
        _fake_batch(3, T=T, cell_id=1, te_inj=2.0, te_scat=1.8, frac_phi=0.9,
                    delay=4, seed=7),
    ]
    arrs = eval_v2.collect_per_sample_kbar(
        model, loader, torch.device("cpu"), warmup=_WARMUP, horizon=_HORIZON,
    )
    # mu_base == 0, mu_full == Y_plus (perfect) -> L_base > 0, L_full == 0 -> gain > 0.
    assert np.all(arrs["pred_gain"] > 0.0)
    assert np.all(arrs["uplift_rel"] > 0.0)


# ---------------------------------------------------------------------------
# S6-T02: gamma-calibration
# ---------------------------------------------------------------------------
def test_calib_slope_recovers_known() -> None:
    r"""``fit_calibration_slope`` recovers a known affine relation."""
    pts = [(0.0, 0.2), (1.0, 0.7), (2.0, 1.2), (3.0, 1.7)]   # kbar = 0.2 + 0.5*TE
    fit = eval_v2.fit_calibration_slope(pts)
    assert fit["gamma"] == pytest.approx(0.5)
    assert fit["alpha"] == pytest.approx(0.2)
    assert fit["r2"] == pytest.approx(1.0)
    assert fit["n"] == 4
    with pytest.raises(ValueError):
        eval_v2.fit_calibration_slope([(1.0, 0.5), (1.0, 0.9)])   # constant TE


def test_calib_both_te_and_monotonic() -> None:
    r"""``fit_calibration`` fits vs both TEs and flags monotonicity."""
    # 4 cells; kbar increases with TE_inj (and TE_scat ~ 1.1*TE_inj).
    cell_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    te_inj = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    te_scat = te_inj * 1.1 - 0.05
    kbar = 0.3 + 0.6 * te_inj + np.array([0.01, -0.01] * 4)     # slope ~0.6
    arrs = {
        "cell_id": cell_ids, "te_inj": te_inj, "te_scat": te_scat,
        "kbar": kbar, "delay": np.full(8, 4), "frac_phi": np.full(8, 1.1),
        "held_out": np.zeros(8),
    }
    cal = eval_v2.fit_calibration(arrs)
    assert cal["n_cells"] == 4
    assert cal["gamma_inj"] == pytest.approx(0.6, abs=0.02)
    assert cal["gamma_scat"] is not None
    assert cal["monotonic_inj"] is True and cal["monotonic_scat"] is True
    assert len(cal["per_cell"]) == 4


# ---------------------------------------------------------------------------
# S6-T03: lag recovery
# ---------------------------------------------------------------------------
def test_lag_score_profile_in_band_and_out() -> None:
    r"""``score_lag_profile``: in-band peak scores mass ~1; out-of-band fails tolerance."""
    L = 12
    band = [4, 5, 6]
    prof_in = np.zeros(L); prof_in[5] = 10.0; prof_in[4] = 2.0
    s_in = eval_v2.score_lag_profile(prof_in, band, tolerance=1)
    assert s_in["lag_mass"] == pytest.approx(1.0)
    assert s_in["peak_lag"] == 5 and s_in["peak_lag_err"] == 0 and s_in["within_tol"]

    prof_out = np.zeros(L); prof_out[10] = 5.0
    s_out = eval_v2.score_lag_profile(prof_out, band, tolerance=1)
    assert s_out["lag_mass"] == pytest.approx(0.0)
    assert s_out["peak_lag"] == 10 and s_out["peak_lag_err"] == 4
    assert s_out["within_tol"] is False

    # +-1 tolerance boundary: peak one step outside the band still passes.
    prof_edge = np.zeros(L); prof_edge[7] = 5.0   # band max is 6
    s_edge = eval_v2.score_lag_profile(prof_edge, band, tolerance=1)
    assert s_edge["peak_lag_err"] == 1 and s_edge["within_tol"] is True


def test_lag_recover_cells() -> None:
    r"""``recover_lags`` builds L* per cell and aggregates over signal cells."""
    horizon = 30
    # cell 0 null (te_inj 0), cell 1 signal (D=8 -> band {0..7}, peak in band).
    L = 91
    prof_signal = np.zeros(L); prof_signal[3] = 1.0
    prof_null = np.ones(L)                       # diffuse -> low in-band mass
    lag_profiles = {0: prof_null, 1: prof_signal}
    cells_by_id = {
        0: {"delay": 8, "te_inj": 0.0},
        1: {"delay": 8, "te_inj": 2.0},
    }
    rec = eval_v2.recover_lags(lag_profiles, cells_by_id, horizon=horizon,
                               tolerance=1, threshold=0.8)
    assert rec["per_cell"][1]["true_band"] == list(range(0, 8))
    assert rec["per_cell"][1]["lag_mass"] == pytest.approx(1.0)
    assert rec["per_cell"][0]["is_null"] is True
    # aggregate uses only the signal cell.
    assert rec["mean_lag_mass"] == pytest.approx(1.0)
    assert rec["mean_lag_mass_pass"] is True


# ---------------------------------------------------------------------------
# S6-T04: null controls + metrics.json
# ---------------------------------------------------------------------------
def test_null_ratios_math() -> None:
    r"""``null_ratios`` computes per-cell and signal-averaged null ratios."""
    arrs = {
        "cell_id": np.array([0, 0, 1, 1]),
        "te_inj": np.array([0.0, 0.0, 2.0, 2.0]),
        "kbar": np.array([1.0, 1.0, 2.0, 2.0]),
        "kbar_shuffle": np.array([0.9, 0.9, 0.4, 0.4]),   # cell1 ratio 0.2
    }
    out = eval_v2.null_ratios(arrs, ["shuffle"])
    assert out["shuffle"]["per_cell"][1]["null_ratio"] == pytest.approx(0.2)
    # signal-cell average excludes the null cell (cell 0).
    assert out["shuffle"]["mean_ratio"] == pytest.approx(0.2)


@pytest.fixture
def force_cpu(monkeypatch) -> None:
    r"""Force torch onto CPU (deterministic, no GPU contention)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _write_eval_split(path: Path, cells: List[Dict[str, Any]], *, per_cell: int,
                      T: int, seed: int) -> None:
    r"""Write one split ``.npz`` split evenly across ``cells`` (native + provenance)."""
    rng = np.random.default_rng(seed)
    n = per_cell * len(cells)
    arrays: Dict[str, np.ndarray] = {
        "fhr_st": rng.standard_normal((n, T, _C_FHR_ST)).astype(np.float32),
        "fhr_ph": rng.standard_normal((n, T, _C_FHR_PH)).astype(np.float32),
        "up_st": rng.standard_normal((n, T, _C_UP_ST)).astype(np.float32),
        "up_ph": rng.standard_normal((n, T, _C_UP_PH)).astype(np.float32),
        "weight": np.ones((n, T), dtype=np.float32),
        "true_lag_tt": np.zeros((n, T), dtype=np.float32),
    }
    te_true = np.empty(n, np.float32); te_scat = np.empty(n, np.float32)
    frac = np.empty(n, np.float32); delay = np.empty(n, np.int16)
    cid = np.empty(n, np.int16); held = np.zeros(n, np.int8)
    for i, cell in enumerate(cells):
        sl = slice(i * per_cell, (i + 1) * per_cell)
        te_true[sl] = cell["te_inj"]; te_scat[sl] = cell["te_scat"]
        frac[sl] = cell["frac_phi"]; delay[sl] = cell["delay"]; cid[sl] = cell["cell_id"]
        arrays["true_lag_tt"][sl] = cell["delay"]
    arrays.update({
        "sample_te_true": te_true, "sample_te_scat": te_scat,
        "sample_frac_phi": frac, "sample_delay": delay,
        "sample_cell_id": cid, "sample_held_out": held,
    })
    np.savez(path, **arrays)


@pytest.fixture
def eval_fixture(tmp_path):
    r"""A tiny config + fixture cache (2 cells) + a saved tiny checkpoint for run_eval."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import pl_module_v2 as plm

    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
        cfg = copy.deepcopy(yaml.safe_load(handle))
    cfg["model"] = copy.deepcopy(_TINY_MODEL)
    cfg["experiment"]["tag"] = "test_eval"
    cfg["experiment"]["benchmark"] = "G1_raw"
    cfg["paths"]["data_dir"] = str(tmp_path / "data")
    cfg["paths"]["results_dir"] = str(tmp_path / "results")
    cfg["dataset"] = {"num_workers": 0, "pin_memory": False,
                      "persistent_workers": False, "mmap": "auto"}
    cfg["optim"]["batch_size"] = 4

    cells = [
        {"cell_id": 0, "te_inj": 0.0, "te_scat": -0.1, "frac_phi": float("nan"), "delay": _DELAY},
        {"cell_id": 1, "te_inj": 2.0, "te_scat": 1.8, "frac_phi": 0.9, "delay": _DELAY},
    ]
    cache_dir = resolve_cache_dir(cfg, benchmark="G1_raw")
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_eval_split(cache_dir / "train.npz", cells, per_cell=6, T=_T, seed=0)
    _write_eval_split(cache_dir / "val.npz", cells, per_cell=3, T=_T, seed=1)
    _write_eval_split(cache_dir / "test.npz", cells, per_cell=4, T=_T, seed=2)
    with open(cache_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump({
            "te_true": 1.0, "tag": "test_eval", "benchmark": "G1_raw",
            "true_lag_band": list(range(max(0, _DELAY - _HORIZON), _DELAY)),
        }, handle)

    results_dir = Path(cfg["paths"]["results_dir"]) / "test_eval"
    results_dir.mkdir(parents=True, exist_ok=True)
    model, kwargs = plm.build_model(cfg["model"], torch.device("cpu"))
    plm.save_checkpoint_v2(
        results_dir / "final.ckpt", model=model, model_kwargs=kwargs,
        config=cfg, data_meta={}, epoch=1, val_loss=float("nan"),
        loss_settings={"beta": 1e-3}, latent_stats_fitted=False,
    )
    return cfg, results_dir


def test_null_run_eval_writes_metrics(eval_fixture) -> None:
    r"""``run_eval`` grades the checkpoint and writes a complete ``metrics.json``."""
    cfg, results_dir = eval_fixture
    metrics = eval_v2.run_eval(
        cfg, benchmark="G1_raw", split="test", out_dir=results_dir,
        batch_size=4, device="cpu",
    )
    assert (results_dir / "metrics.json").is_file()
    with open(results_dir / "metrics.json", "r", encoding="utf-8") as handle:
        on_disk = json.load(handle)

    for gate in ("calibration", "lag_recovery", "null_controls", "null_probe",
                 "frac_phi", "per_cell", "per_cell_profiles"):
        assert gate in on_disk, gate
    assert on_disk["n_cells"] == 2
    assert on_disk["split"] == "test"
    for key in ("gamma_inj", "gamma_scat", "alpha_inj", "monotonic_inj"):
        assert key in on_disk["calibration"]
    # both null controls evaluated (config default [shuffle, reverse]).
    assert set(on_disk["null_controls"]) == {"shuffle", "reverse"}
    # null probe reads the null cell's dressing-only TE_scat.
    assert on_disk["null_probe"]["null_cell_ids"] == [0]
    # every per-cell row carries the prediction-gap fields.
    for row in on_disk["per_cell"]:
        assert "pred_gain" in row and "uplift_rel" in row
    # per-cell profiles persisted: a lag_profile + kbar_over_time per cell (JSON str keys).
    profiles = on_disk["per_cell_profiles"]
    assert set(profiles) == {"0", "1"}
    for prof in profiles.values():
        assert prof["lag_profile"] is not None and len(prof["lag_profile"]) > 0
        assert len(prof["kbar_over_time"]) == _T
        assert prof["lag_count"] > 0

    # the minimal report renders from the metrics.
    report_path = eval_v2.write_report(metrics, results_dir)
    assert Path(report_path).is_file()
    text = Path(report_path).read_text(encoding="utf-8")
    assert "Calibration" in text and "Lag recovery" in text and "Null controls" in text


# ---------------------------------------------------------------------------
# S6-T05: end-to-end integration smoke (real transform, tiny grid)
# ---------------------------------------------------------------------------
_E2E_MODEL: Dict[str, Any] = {
    **_TINY_MODEL,
    "sequence_length": 300,   # the real cache T after trim
    "horizon": 8,
    "warmup_period": 4,
    "max_lag": 16,
}


@pytest.mark.slow
def test_e2e_build_train_eval_report(tmp_path, force_cpu) -> None:
    r"""Compose build -> r0_realizability -> train -> eval -> report on a tiny real grid.

    The one heavy test: it runs the real scattering transform on a handful of samples
    (shared adapter) and asserts every stage's artifact is produced.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import pl_module_v2 as plm
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (
        build_all,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.scattering_adapter import (
        ScatteringAdapter,
    )

    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
        cfg = copy.deepcopy(yaml.safe_load(handle))
    cfg["model"] = copy.deepcopy(_E2E_MODEL)
    cfg["experiment"]["tag"] = "test_e2e"
    cfg["experiment"]["benchmark"] = "G1_raw"
    cfg["paths"]["data_dir"] = str(tmp_path / "data")
    cfg["paths"]["results_dir"] = str(tmp_path / "results")
    cfg["dataset"] = {"num_workers": 0, "pin_memory": False,
                      "persistent_workers": False, "mmap": "auto"}
    cfg["optim"]["batch_size"] = 4
    # cheap inverter + a tiny pilot realizability grid.
    cfg["benchmarks"]["G1_raw"]["mix"]["inverter"]["n_samples"] = 4000
    cfg["benchmarks"]["G1_raw"]["eval"]["realizability"]["pilot"] = {
        "target_te_grid": [0.0, 2.0], "lag_grid": [8], "n_per_cell": 12,
    }

    adapter = ScatteringAdapter(cfg, benchmark="G1_raw")
    results_dir = Path(cfg["paths"]["results_dir"]) / "test_e2e"

    # 1) build a tiny real cache.
    cache_dir = build_all(
        cfg, benchmark="G1_raw",
        grid_override={"target_te_grid": [0.0, 2.0], "lag_grid": [8]},
        n_override={"train": 12, "val": 6, "test": 6}, adapter=adapter,
    )
    for split in ("train", "val", "test"):
        assert (cache_dir / f"{split}.npz").is_file()

    # 2) r0_realizability pre-flight (non-fatal) -> realizability.json.
    eval_v2.run_realizability_preflight(
        cfg, benchmark="G1_raw", pilot=True, out_dir=results_dir,
        adapter=adapter, print_table=False,
    )
    assert (results_dir / "realizability.json").is_file()

    # 3) train (1 epoch, CPU) -> checkpoint + loss curves.
    train_res = plm.train_v2(
        cfg,
        {"epochs": 1, "limit_train_batches": 2, "limit_val_batches": 1,
         "batch_size": 4, "devices": 1, "progress_bar": False},
        benchmark="G1_raw",
    )
    assert Path(train_res["checkpoint"]).is_file()

    # 4) eval -> metrics.json.
    metrics = eval_v2.run_eval(
        cfg, benchmark="G1_raw", split="test", out_dir=results_dir,
        batch_size=4, device="cpu",
    )
    assert (results_dir / "metrics.json").is_file()
    assert metrics["n_cells"] >= 1

    # 5) report -> report.md.
    report_path = eval_v2.write_report(metrics, results_dir)
    assert Path(report_path).is_file()
