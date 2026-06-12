r"""Unit tests for the ``mixed_eval`` per-group evaluator.

Splits into (i) pure-function tests -- calibration slices, per-cell grouping,
peak-lag error, generalization gaps -- driven by hand-built arrays, and (ii)
eval-mechanics tests that run an **untrained** :class:`SeqVaeLagAttnV1` over a
tiny mixed cache (no training needed) to exercise the per-sample $\bar K$
windowing, grouping by ``cell_id`` and the per-cell LOLO wiring.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from model.vae_teb_prediction.model.model_experiment.synthetic import (
    mixed_dataset as MD,
    mixed_eval as ME,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.dataset import (
    SyntheticTEDataset,
    make_dataloader,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (
    load_config,
    resolve_active_benchmark,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1

_CONFIG = Path(__file__).resolve().parent / "config_synth.yaml"


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def _synthetic_arrs(alpha: float, gamma: float):
    """Build per-sample arrays whose per-cell means lie on K = a + g*TE."""
    cells = {
        0: {"cell_id": 0, "M": 8, "target_te": 0.5, "band": "mid",
            "delay_min": 1, "delay_max": 15, "B_y_scalar": 0.01},
        1: {"cell_id": 1, "M": 16, "target_te": 1.5, "band": "mid",
            "delay_min": 1, "delay_max": 15, "B_y_scalar": 0.02},
        2: {"cell_id": 2, "M": 32, "target_te": 3.0, "band": "short",
            "delay_min": 1, "delay_max": 8, "B_y_scalar": 0.03},
    }
    te_by_cell = {0: 0.5, 1: 1.5, 2: 3.0}
    cid, te, kb, M, band, dmax, held = [], [], [], [], [], [], []
    rng = np.random.default_rng(0)
    for c in (0, 1, 2):
        for _ in range(20):
            cid.append(c)
            te.append(te_by_cell[c])
            kb.append(alpha + gamma * te_by_cell[c] + rng.normal(0, 1e-6))
            M.append(cells[c]["M"])
            band.append({"short": 0, "mid": 1}[cells[c]["band"]])
            dmax.append(cells[c]["delay_max"])
            held.append(0)
    arrs = {
        "cell_id": np.array(cid), "te_true": np.array(te, float),
        "kbar": np.array(kb, float), "M": np.array(M),
        "band_id": np.array(band), "delay_max": np.array(dmax),
        "held_out": np.array(held),
    }
    return arrs, cells


def test_fit_calibration_slices_recovers_slope():
    """An exact K = 0.2 + 0.8*TE pool yields gamma=0.8, alpha=0.2 overall."""
    arrs, cells = _synthetic_arrs(alpha=0.2, gamma=0.8)
    slices = ME.fit_calibration_slices(arrs, cells)
    assert slices["overall"]["gamma"] == pytest.approx(0.8, abs=1e-4)
    assert slices["overall"]["alpha"] == pytest.approx(0.2, abs=1e-4)
    # Per-M slices each have a single TE point -> singular -> None.
    assert all(v is None for v in slices["by_M"].values())


def test_group_recovery_te_pred_and_nulls():
    """TE_pred = (kbar - alpha)/gamma recovers the cell TE; null ratios read."""
    arrs, cells = _synthetic_arrs(alpha=0.0, gamma=1.0)
    arrs["kbar_shuffle"] = arrs["kbar"] * 0.1
    rows = ME.group_recovery(arrs, cells, alpha=0.0, gamma=1.0,
                             controls=("shuffle",))
    assert len(rows) == 3
    for r in rows:
        assert r["te_pred_mean"] == pytest.approx(r["te_true"], abs=1e-4)
        assert r["te_rmse"] == pytest.approx(0.0, abs=1e-4)
        assert r["null_shuffle_ratio"] == pytest.approx(0.1, abs=1e-3)


def test_peak_lag_err():
    band = [0, 1, 2, 3]
    a = [0.0] * 10
    a[2] = 1.0
    assert ME._peak_lag_err(a, band) == 0.0
    a2 = [0.0] * 10
    a2[7] = 1.0
    assert ME._peak_lag_err(a2, band) == 4.0     # |7 - 3|
    assert np.isnan(ME._peak_lag_err([], band))
    assert np.isnan(ME._peak_lag_err([0.0, 0.0], band))


def test_generalization_gaps():
    rows_in = [
        {"M": 8, "te_rmse": 0.10, "lag_mass_lolo": 0.80},
        {"M": 8, "te_rmse": 0.20, "lag_mass_lolo": 0.60},
    ]
    rows_ho = [
        {"M": 8, "target_te": 1.5, "band": "mid",
         "te_rmse": 0.30, "lag_mass_lolo": 0.50},
    ]
    gaps = ME._generalization_gaps(rows_in, rows_ho)
    cell = gaps["cells"][0]
    assert cell["te_rmse_ref"] == pytest.approx(0.15)     # mean(0.10, 0.20)
    assert cell["te_rmse_gap"] == pytest.approx(0.15)     # 0.30 - 0.15
    assert cell["lag_mass_ref"] == pytest.approx(0.70)
    assert cell["lag_mass_gap"] == pytest.approx(-0.20)


def test_generalization_gaps_empty_holdout():
    assert ME._generalization_gaps([{"M": 8, "te_rmse": 0.1}], []) == {
        "cells": [], "summary": {}}


# ---------------------------------------------------------------------------
# Eval mechanics on an untrained model + a tiny cache
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cache_and_model(tmp_path_factory):
    """A tiny 2-cell mixed cache + a fresh (untrained) model on CPU."""
    tmp = tmp_path_factory.mktemp("g1mix_eval")
    cfg = load_config(_CONFIG)
    cfg["experiment"]["benchmark"] = "G1_mix"
    resolve_active_benchmark(cfg)
    mix = cfg["benchmarks"]["G1_mix"]["mix"]
    mix["build_workers"] = 1  # serial build in tests (cf. test_mixed_dataset);
    # a spawned pool worker re-imports the package fresh, which is fragile
    # under pytest's import mode and pointless for a 2-cell grid.
    mix["m_grid"] = [8, 16]
    mix["target_te_grid"] = [1.5]
    mix["lag_bands"] = {"mid": [1, 15]}
    mix["holdout"] = []
    mix["n_per_cell_train"] = 2
    mix["n_per_cell_val"] = 2
    mix["n_per_cell_test"] = 4
    # te_n_samples=1500 keeps the MC floor below the (16, 1.5) per-channel
    # target (0.094) so both M cells survive and the pool genuinely mixes M.
    mix["inverter"] = {"n_samples": 1500, "lo": 1e-4, "hi": 10.0,
                       "tol": 0.05, "max_iter": 8}
    cfg["data"]["te_n_samples"] = 1500
    cfg["data"]["sequence_length"] = 120
    cfg["model"]["sequence_length"] = 120
    cfg["model"]["horizon"] = 30
    cfg["paths"]["data_dir"] = str(tmp)
    cfg["experiment"]["tag"] = "tiny"
    MD.build_g1_mix(copy.deepcopy(cfg), force=True)

    model = SeqVaeLagAttnV1(
        sequence_length=120, d_model=128, d_z=24, horizon=30, warmup_period=30,
        c_y=87, c_u=101, use_up_st=True, max_lag=90, num_heads=4, d_head=32,
    ).eval()
    ds = SyntheticTEDataset(tmp / "G1_mix" / "tiny" / "test.npz")
    return {"cfg": cfg, "ds": ds, "model": model,
            "test_npz": tmp / "G1_mix" / "tiny" / "test.npz"}


def test_collect_per_sample_kbar_shapes_and_controls(cache_and_model):
    ds = cache_and_model["ds"]
    model = cache_and_model["model"]
    loader = make_dataloader(ds, batch_size=4, shuffle=False)
    arrs = ME.collect_per_sample_kbar(
        model, loader, torch.device("cpu"), warmup=30, horizon=30,
        controls=("shuffle", "reverse"),
    )
    n = len(ds)
    for key in ("kbar", "te_true", "M", "delay_max", "band_id", "cell_id",
                "held_out", "kbar_shuffle", "kbar_reverse"):
        assert arrs[key].shape == (n,), key
    assert np.all(np.isfinite(arrs["kbar"]))
    assert set(arrs["M"].tolist()) <= {8, 16}


def test_per_sample_kbar_window_matches_manual(cache_and_model):
    """The per-sample window mean equals the masked mean of the same KL tensor.

    Compares ``_per_sample_kld``'s scalar ``kbar`` against a hand-computed masked
    mean of *its own* returned full-sequence ``kld_bt`` -- so the check is exact
    and self-consistent (no separate, dropout-perturbed encoder pass).
    """
    ds = cache_and_model["ds"]
    model = cache_and_model["model"]
    model.eval()
    loader = make_dataloader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    from model.vae_teb_prediction.model.model_experiment.synthetic.dataset import (
        build_u_stream,
    )
    got, _, _, kld_bt = ME._per_sample_kld(
        model, batch.fhr_st, batch.fhr_ph, build_u_stream(batch),
        batch["delay_max"], warmup=30, horizon=30)
    T = kld_bt.shape[1]
    for i in range(kld_bt.shape[0]):
        lo = max(30, int(batch["delay_max"][i]) - 1)
        manual = float(kld_bt[i, lo:T - 30].mean())
        assert got[i] == pytest.approx(manual, abs=1e-5)


def test_per_cell_lag_recovery_runs(cache_and_model):
    ds = cache_and_model["ds"]
    model = cache_and_model["model"]
    with np.load(cache_and_model["test_npz"]) as npz:
        cellids = np.asarray(npz["sample_cell_id"], int)
    manifest = ds.meta["mixture"]
    cells_by_id = {int(c["cell_id"]): c for c in manifest["cells"]}
    out = ME.per_cell_lag_recovery(
        model, ds, cells_by_id, cellids, torch.device("cpu"),
        horizon=30, T=120, max_lag=90, warmup=30,
        loss_settings={"kld_beta": 0.001, "lambda_full": 1.0, "lambda_base": 0.5},
        eval_cfg={"window_width": 8, "lag_grid_step": 10,
                  "fine_lag_grid_step": 2, "n_lolo_per_cell": 4, "batch_size": 4},
    )
    assert set(out) == set(cells_by_id)
    for res in out.values():
        assert "lag_mass_lolo" in res
        assert "A_lag" in res
        assert "peak_lag_err" in res


def test_evaluate_mixed_end_to_end(cache_and_model, tmp_path):
    """evaluate_mixed on an untrained model writes the full artifact set."""
    cfg = copy.deepcopy(cache_and_model["cfg"])
    # Persist the untrained model as a checkpoint the evaluator can load.
    model = cache_and_model["model"]
    results_root = tmp_path / "results"
    run_dir = results_root / "G1_mix" / "untrained"
    run_dir.mkdir(parents=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_kwargs": {
            "sequence_length": 120, "d_model": 128, "d_z": 24, "horizon": 30,
            "warmup_period": 30, "c_y": 87, "c_u": 101, "use_up_st": True,
            "max_lag": 90, "num_heads": 4, "d_head": 32,
        },
        "loss_settings": {"kld_beta": 0.001, "lambda_full": 1.0,
                          "lambda_base": 0.5, "likelihood": "gaussian_nll"},
    }, run_dir / "final.ckpt")
    cfg["paths"]["results_dir"] = str(results_root)
    cfg["runtime"]["device"] = "cpu"
    cfg["benchmarks"]["G1_mix"]["eval"]["n_lolo_per_cell"] = 4

    metrics = ME.evaluate_mixed(
        cfg, run_tag="untrained", in_mix_tag="tiny", holdout_tag=None,
    )
    out = run_dir / "mixed_eval"
    assert (out / "per_cell.csv").is_file()
    assert (out / "per_sample.csv").is_file()
    assert (out / "calibration.json").is_file()
    assert (out / "metrics.json").is_file()
    assert (out / "kld_vs_te.pdf").is_file()
    assert (out / "calibration_health.pdf").is_file()
    assert metrics["n_cells_in_mix"] >= 1

    # --- Workstream A additions: new metric blocks, columns and figure --------
    assert "lag_recovery_summary" in metrics
    assert "lag_walk" in metrics
    lrs = metrics["lag_recovery_summary"]["in_mix"]
    assert lrs["threshold"] == pytest.approx(0.8)
    assert "frac_cells_lolo_ge" in lrs and "frac_cells_attn_ge" in lrs
    header = (out / "per_cell.csv").read_text(encoding="utf-8").splitlines()[0]
    for col in ("lag_mass_attn", "lag_mass_attn_ratio", "peak_lag_err_mean",
                "peak_in_band_frac", "lag_walk_mae", "lag_walk_within1_frac"):
        assert col in header, col

    # --- Per-sample scatter suite + evaluate_te-style figures -----------------
    ps_header = (out / "per_sample.csv").read_text(
        encoding="utf-8").splitlines()[0]
    assert ps_header.endswith("held_out,split")
    assert (out / "per_sample_scatter.pdf").is_file()
    assert (out / "per_sample_null_scatter.pdf").is_file()
    # evaluate_te-style cross-cell suite rendered via the row adapter:
    assert (out / "kbar_vs_te.pdf").is_file()
    assert (out / "kbar_vs_te__byM.pdf").is_file()
    # The tiny G1 cache carries the per-step lag walk -> the lag-walk figure and
    # the per-cell lag-walk metric are produced.
    with np.load(cache_and_model["test_npz"]) as npz:
        has_walk = "true_lag_tt" in npz.files
    if has_walk:
        assert (out / "lag_walk.pdf").is_file()
        assert metrics["lag_walk"]["in_mix"], "lag_walk summary should be non-empty"


# ---------------------------------------------------------------------------
# Workstream A pure functions: lag-mass rollup, lag-walk summary, cell meta
# ---------------------------------------------------------------------------

def test_lag_recovery_summary_thresholds():
    """Per-cell LagMass rolls up to the criterion-#4 pass fractions."""
    rows = [
        {"M": 8, "band": "mid", "lag_mass_lolo": 0.90, "lag_mass_attn": 0.85},
        {"M": 8, "band": "mid", "lag_mass_lolo": 0.50, "lag_mass_attn": 0.70},
        {"M": 16, "band": "short", "lag_mass_lolo": 0.95,
         "lag_mass_attn": float("nan")},
    ]
    s = ME.lag_recovery_summary(rows, threshold=0.8)
    assert s["n_cells"] == 3
    assert s["frac_cells_lolo_ge"] == pytest.approx(2.0 / 3.0)
    # attn: only the two finite values {0.85, 0.70} count -> 1/2 >= 0.8.
    assert s["frac_cells_attn_ge"] == pytest.approx(0.5)
    assert s["by_M"]["8"]["frac_lolo_ge"] == pytest.approx(0.5)
    assert s["by_M"]["16"]["frac_lolo_ge"] == pytest.approx(1.0)
    assert ME.lag_recovery_summary([]) == {}


def test_lag_walk_summary():
    """Aggregates per-cell lag-walk MAE / within-1 / corr; empty when absent."""
    rows = [
        {"lag_walk_mae": 1.0, "lag_walk_within1_frac": 0.8, "lag_walk_corr": 0.5},
        {"lag_walk_mae": 3.0, "lag_walk_within1_frac": 0.4, "lag_walk_corr": 0.7},
    ]
    s = ME._lag_walk_summary(rows)
    assert s["n_cells"] == 2
    assert s["mean_mae"] == pytest.approx(2.0)
    assert s["max_mae"] == pytest.approx(3.0)
    assert s["mean_within1_frac"] == pytest.approx(0.6)
    assert s["mean_corr"] == pytest.approx(0.6)
    assert ME._lag_walk_summary([{"M": 8}]) == {}


def test_pearson_helper():
    """Streaming Pearson matches numpy on a perfectly correlated stream."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x + 1.0
    n = float(x.size)
    got = ME._pearson(
        float(x.sum()), float(y.sum()), float((x * x).sum()),
        float((y * y).sum()), float((x * y).sum()), n,
    )
    assert got == pytest.approx(1.0, abs=1e-9)
    assert np.isnan(ME._pearson(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))  # n<2 -> nan


def test_cell_eval_meta_superset():
    """``_cell_eval_meta`` extends ``_cell_meta`` with the single-cell scalars."""
    cell = {"cell_id": 1, "M": 16, "target_te": 1.5, "te_cell_realised": 1.42,
            "band": "mid", "delay_min": 1, "delay_max": 15, "B_y_scalar": 0.02}
    meta = ME._cell_eval_meta(cell, horizon=30, T=120)
    # base (_cell_meta) fields
    assert meta["true_lag_band"] == list(range(15))
    assert meta["clean_anchor_range"] == [14, 90]
    assert meta["horizon"] == 30 and meta["sequence_length"] == 120
    # added scalars
    assert meta["te_true"] == pytest.approx(1.42)
    assert meta["te_per_step"] == pytest.approx(1.42 / 30.0)
    assert meta["M"] == 16 and meta["delay_max"] == 15 and meta["delay"] == 15
    assert meta["B_y"] == pytest.approx(0.02)


def test_collect_per_cell_attn_diag_runs(cache_and_model):
    """Attention LagMass / peak-lag / lag-walk run per cell on the tiny cache."""
    ds = cache_and_model["ds"]
    model = cache_and_model["model"]
    with np.load(cache_and_model["test_npz"]) as npz:
        cellids = np.asarray(npz["sample_cell_id"], int)
        has_walk = "true_lag_tt" in npz.files
    cells_by_id = {int(c["cell_id"]): c for c in ds.meta["mixture"]["cells"]}
    out = ME.collect_per_cell_attn_diag(
        model, ds, cells_by_id, cellids, torch.device("cpu"),
        warmup=30, horizon=30, T=120, max_lag=90,
        eval_cfg={"n_lolo_per_cell": 4, "batch_size": 4}, do_lag_walk=True,
    )
    assert set(out) == set(cells_by_id)
    for res in out.values():
        assert np.isfinite(res["lag_mass_attn_uniform"])
        assert "lag_mass_attn_ratio" in res
        assert "peak_lag_err_mean" in res and "peak_in_band_frac" in res
        if has_walk:
            assert "lag_walk_mae" in res
            assert len(res["lag_walk_true_mean"]) == 120
            assert len(res["lag_walk_pred_mean"]) == 120


def test_collect_per_cell_attn_diag_no_walk_when_disabled(cache_and_model):
    """``do_lag_walk=False`` omits the lag-walk fields but keeps LagMass."""
    ds = cache_and_model["ds"]
    model = cache_and_model["model"]
    with np.load(cache_and_model["test_npz"]) as npz:
        cellids = np.asarray(npz["sample_cell_id"], int)
    cells_by_id = {int(c["cell_id"]): c for c in ds.meta["mixture"]["cells"]}
    out = ME.collect_per_cell_attn_diag(
        model, ds, cells_by_id, cellids, torch.device("cpu"),
        warmup=30, horizon=30, T=120, max_lag=90,
        eval_cfg={"n_lolo_per_cell": 4, "batch_size": 4}, do_lag_walk=False,
    )
    for res in out.values():
        assert "lag_mass_attn" in res
        assert "lag_walk_mae" not in res


# ---------------------------------------------------------------------------
# Per-sample CSV round-trip + per-sample scatter suite (no model / GPU)
# ---------------------------------------------------------------------------

def _arrs_with_controls(alpha: float, gamma: float):
    """``_synthetic_arrs`` plus a fake shuffle control (10% of clean kbar)."""
    arrs, cells = _synthetic_arrs(alpha, gamma)
    arrs["kbar_shuffle"] = 0.1 * arrs["kbar"]
    return arrs, cells


def test_write_and_read_per_sample_csv_roundtrip(tmp_path):
    """Both splits round-trip through the CSV with held_out / split intact."""
    arrs, _ = _arrs_with_controls(0.2, 0.8)
    arrs_ho, _ = _arrs_with_controls(0.2, 0.8)
    arrs_ho["held_out"] = np.ones_like(arrs_ho["held_out"])
    path = tmp_path / "per_sample.csv"
    ME._write_per_sample_csv(
        path, [("in_mix", arrs), ("holdout", arrs_ho)], ("shuffle",))
    got = ME._read_per_sample_csv(path)
    n = arrs["kbar"].size
    assert got["kbar"].size == 2 * n
    assert np.allclose(got["kbar"][:n], arrs["kbar"])
    assert np.allclose(got["kbar_shuffle"][n:], arrs_ho["kbar_shuffle"])
    assert np.array_equal(got["M"][:n], arrs["M"])
    assert set(got["split"][:n]) == {"in_mix"}
    assert set(got["split"][n:]) == {"holdout"}
    assert np.all(got["held_out"][n:] == 1)
    # Empty splits are skipped without error.
    ME._write_per_sample_csv(path, [("in_mix", arrs), ("holdout", {})], ())
    assert ME._read_per_sample_csv(path)["kbar"].size == n


def test_read_per_sample_csv_legacy_format(tmp_path):
    """A pre-split CSV (no held_out/split columns) defaults to in_mix rows."""
    path = tmp_path / "per_sample.csv"
    path.write_text(
        "cell_id,M,band_id,delay_max,te_true,kbar\n"
        "0,8,1,15,0.5,1.0\n"
        "1,16,1,15,1.5,2.0\n",
        encoding="utf-8",
    )
    got = ME._read_per_sample_csv(path)
    assert np.array_equal(got["held_out"], [0, 0])
    assert set(got["split"]) == {"in_mix"}
    assert np.allclose(got["kbar"], [1.0, 2.0])


def test_per_sample_scatter_figures_smoke(tmp_path):
    """Every per-sample figure renders from hand-built arrays (no model)."""
    arrs, cells = _arrs_with_controls(0.2, 0.8)
    slices = ME.fit_calibration_slices(arrs, cells)
    slices_ns = ME.fit_calibration_slices_nullsub(arrs, cells, control="shuffle")
    rows = ME.group_recovery(arrs, cells, alpha=0.2, gamma=0.8,
                             controls=("shuffle",))
    arrs_ho, _ = _arrs_with_controls(0.2, 0.8)
    arrs_ho["held_out"] = np.ones_like(arrs_ho["held_out"])

    ME._fig_per_sample_scatter(
        tmp_path / "per_sample_scatter", rows, [], arrs, slices, ("shuffle",),
        arrs_ho=arrs_ho)
    ME._fig_per_sample_nullsub(
        tmp_path / "per_sample_nullsub", rows, [], arrs, slices, ("shuffle",),
        slices_nullsub=slices_ns)
    ME._fig_per_sample_te_error(
        tmp_path / "per_sample_te_error", rows, [], arrs, slices, ("shuffle",))
    ME._fig_per_sample_ecdf(
        tmp_path / "per_sample_kbar_ecdf", rows, [], arrs, slices, ("shuffle",))
    ME._fig_per_sample_null_scatter(
        tmp_path / "per_sample_null_scatter", rows, [], arrs, slices,
        ("shuffle",))
    for name in ("per_sample_scatter", "per_sample_nullsub",
                 "per_sample_te_error", "per_sample_kbar_ecdf",
                 "per_sample_null_scatter"):
        assert (tmp_path / f"{name}.pdf").is_file(), name


def test_evalte_suite_adapter_smoke(tmp_path):
    """The evaluate_te renderers run on adapted mixed per-cell rows."""
    arrs, cells = _arrs_with_controls(0.0, 1.0)
    # Per-cell per-dim KL so the by-cell heatmap also renders.
    arrs["per_dim_kl_by_cell"] = {
        c: (np.linspace(0.0, 1.0, 8) * (c + 1)).tolist() for c in (0, 1, 2)
    }
    slices = ME.fit_calibration_slices(arrs, cells)
    rows = ME.group_recovery(arrs, cells, alpha=0.0, gamma=1.0,
                             controls=("shuffle",))
    for r in rows:
        r["delta_L"] = 0.01 * r["te_true"]

    ME._fig_evalte_suite(tmp_path / "evalte_suite", rows, [], arrs, slices, ())
    for name in ("kbar_vs_te", "kbar_vs_B_y", "predgap_vs_kbar",
                 "kbar_vs_te__byM"):
        assert (tmp_path / f"{name}.pdf").is_file(), name
    ME._fig_per_dim_kl_by_cell_heatmap(
        tmp_path / "per_dim_kl_by_cell", rows, [], arrs, slices, ())
    assert (tmp_path / "per_dim_kl_by_cell.pdf").is_file()
    ME._fig_null_control_bars(
        tmp_path / "null_control_bars", rows, [], arrs, slices, ("shuffle",))
    assert (tmp_path / "null_control_bars.pdf").is_file()


def test_render_combined_per_sample_scatter(tmp_path):
    """The replot-only combined renderer pools CSVs across eval passes."""
    import json

    arrs, cells = _arrs_with_controls(0.2, 0.8)
    rows = ME.group_recovery(arrs, cells, alpha=0.2, gamma=0.8,
                             controls=("shuffle",))
    arrs_ho, _ = _arrs_with_controls(0.2, 0.8)
    arrs_ho["held_out"] = np.ones_like(arrs_ho["held_out"])

    run_dir = tmp_path / "run"
    base = run_dir / "mixed_eval"
    base.mkdir(parents=True)
    ME._write_per_sample_csv(
        base / "per_sample.csv",
        [("in_mix", arrs), ("holdout", arrs_ho)], ("shuffle",))
    ME._write_per_cell_csv(base / "per_cell.csv", rows, [], ("shuffle",))
    slices = ME.fit_calibration_slices(arrs, cells)
    with open(base / "calibration.json", "w", encoding="utf-8") as fh:
        json.dump({
            "alpha": 0.2, "gamma": 0.8,
            "in_mix": slices,
            "in_mix_nullsub": ME.fit_calibration_slices_nullsub(
                arrs, cells, control="shuffle"),
        }, fh)

    # One extrapolation pass at an untrained M=4.
    arrs_m4, _ = _arrs_with_controls(0.2, 0.8)
    arrs_m4["M"] = np.full_like(arrs_m4["M"], 4)
    arrs_m4["held_out"] = np.ones_like(arrs_m4["held_out"])
    extrap = run_dir / "mixed_eval_extrap_m4"
    extrap.mkdir(parents=True)
    ME._write_per_sample_csv(
        extrap / "per_sample.csv",
        [("in_mix", arrs), ("holdout", arrs_m4)], ("shuffle",))

    written = ME.render_combined_per_sample_scatter(run_dir)
    assert written, "expected at least one combined figure"
    out = run_dir / "combined_figures"
    assert (out / "per_sample_scatter_all.pdf").is_file()
    assert (out / "per_sample_kbar_ecdf_all.pdf").is_file()
    # evaluate_te-style suite pooled from per_cell.csv:
    assert (out / "kbar_vs_te.pdf").is_file()

    # A missing run dir is a graceful no-op.
    assert ME.render_combined_per_sample_scatter(tmp_path / "nope") is None
