r"""Sprint 6 tests: the raw-domain ground-truth grader ``eval_v4``.

Fabricated-fixture estimator tests (``calibration`` / ``null`` / ``pred_control`` / ``lag``) validate
the gate MATH without a trained model; the ``collect`` / ``run_eval`` tests exercise the real raw
forward path on a small-prod checkpoint + the tiny synthetic cache. The empirical $\gamma>0$ from a
trained model is the separate S8-T01 gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

pytestmark = pytest.mark.v4


# ---------------------------------------------------------------------------
# Shared: a small-prod checkpoint + a live runner/loader over the tiny cache.
# ---------------------------------------------------------------------------
def _small_prod_kwargs() -> Dict[str, Any]:
    from model.vae_teb_prediction.model.model_raw.testing.conftest import (
        SMALL_PROD_FRONTEND,
        SMALL_PROD_V3_KWARGS,
    )

    return dict(frontend=dict(SMALL_PROD_FRONTEND), raw_len=5280, decimation=16,
                **SMALL_PROD_V3_KWARGS)


def _save_ckpt(path: Path) -> None:
    import torch

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import SeqVaeRawV4
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.trainer_v4 import (
        SyntheticSeqVaeRawV4Pl,
    )

    kwargs = _small_prod_kwargs()
    model = SeqVaeRawV4(**kwargs)
    pl_module = SyntheticSeqVaeRawV4Pl(model, arm="prod", render_mode="direct", lr=1e-3,
                                       model_kwargs=kwargs)
    checkpoint: Dict[str, Any] = {"state_dict": pl_module.state_dict()}
    pl_module.on_save_checkpoint(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(path))


def _runner_loader(tiny_cache_v4, tmp_path):
    import torch

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_runner_v4 import (
        _build_runner_and_loader_v4,
    )

    ckpt = tmp_path / "prod" / "final.ckpt"
    _save_ckpt(ckpt)
    return _build_runner_and_loader_v4(
        ckpt, tiny_cache_v4["config"], benchmark="G1_raw_v4",
        cache_dir=tiny_cache_v4["cache_dir"], output_dir=tmp_path / "_eval",
        batch_size=2, split="val", device=torch.device("cpu"),
    )


# ---------------------------------------------------------------------------
# S6-T01: collect_per_sample_kbar_v4
# ---------------------------------------------------------------------------
def test_collect_te_lag_sum_equals_kld_and_losses_finite(tiny_cache_v4, tmp_path):
    r"""te_lag_map summed over lags == kld_per_t; l_full/l_base are finite masked raw MSE."""
    import torch

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v4 import (
        collect_per_sample_kbar_v4,
    )

    runner, loader = _runner_loader(tiny_cache_v4, tmp_path)

    # The lag-sum identity, checked directly on a forward pass.
    batch = next(iter(loader))
    with runner.inference_mode():
        out = runner.forward(batch)
    lag_sum = out["te_lag_map"].sum(dim=-1)                 # (B, T)
    assert torch.allclose(lag_sum, out["kld_per_t"], atol=1e-4, rtol=1e-3)

    arrs = collect_per_sample_kbar_v4(
        runner, loader, warmup=int(runner.warmup_steps), horizon=int(runner.horizon))
    assert arrs["n"] >= 1
    for key in ("kbar", "kbar_postwarm", "feat_loss", "base_loss", "feat_loss_shuffled"):
        assert key in arrs
        assert np.isfinite(np.asarray(arrs[key])).all(), f"{key} not finite"
    # in-band + out-band == kbar is a definitional identity (out-band := kbar - in-band), so it
    # alone cannot catch a wrong band mask. The bound below adds real coverage: the band is a 0/1 lag
    # subset over a non-negative te_lag_map, so the in-band mass must be a non-negative fraction of the
    # total, 0 <= kbar_inband <= kbar elementwise. This catches a sign-flipped / wrong tensor being
    # summed or a non-binary band mask; it does NOT constrain band *placement* (any lag subset
    # satisfies the bound) -- that is covered by the te_lag_map-sum identity above and the lag-recovery
    # test.
    assert np.allclose(arrs["kbar_inband"] + arrs["kbar_outband"], arrs["kbar"], atol=1e-5)
    assert np.all(arrs["kbar_inband"] >= -1e-6)
    assert np.all(arrs["kbar_inband"] <= arrs["kbar"] + 1e-6)
    assert isinstance(arrs["lag_profiles"], dict) and arrs["lag_profiles"]


# ---------------------------------------------------------------------------
# S6-T02: fit_calibration_v4 (estimator test on the fabricated fixture)
# ---------------------------------------------------------------------------
def _arrs_from_kbar(kbar, te_true) -> Dict[str, Any]:
    r"""Wrap ``(kbar, te_true)`` into a minimal ``arrs`` dict (cell_id per TE level)."""
    te = np.asarray(te_true, dtype=np.float64)
    levels = {v: i for i, v in enumerate(sorted(np.unique(te)))}
    cell_id = np.asarray([levels[v] for v in te], dtype=np.int64)
    return {
        "kbar": np.asarray(kbar, dtype=np.float64),
        "te_inj": te,
        "te_scat": np.full(te.shape, np.nan),
        "cell_id": cell_id,
        "delay": np.full(te.shape, 8, dtype=np.int64),
        "held_out": np.zeros(te.shape, dtype=np.int64),
    }


def test_calibration_recovers_positive_slope(signal_kbar_fixture):
    r"""On K = gamma*TE + noise the recovered slope is >0 and matches gamma (estimator test)."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v4 import (
        fit_calibration_v4,
    )

    data = signal_kbar_fixture(gamma=0.7, noise=0.05, reps=60)
    arrs = _arrs_from_kbar(data["kbar"], data["te_true"])
    # Provide kbar_full so fit_calibration_v4 actually populates its kld_variants entry; without it the
    # out_of_support assertion below would be vacuous (the ``in kv`` guard never fires).
    arrs["kbar_full"] = np.asarray(data["kbar"], dtype=np.float64)
    cal = fit_calibration_v4(arrs, kld_support="anchor")

    assert cal["gamma"] is not None and cal["gamma"] > 0.0
    assert abs(cal["gamma"] - 0.7) < 0.1
    assert cal["spearman"] is not None and cal["monotonic"] is True
    # kbar_full is flagged out_of_support under anchor KL support (S6-T02).
    kv = cal["kld_variants"]
    assert "kbar_full" in kv
    assert kv["kbar_full"]["out_of_support"] is True


# ---------------------------------------------------------------------------
# S6-T03: null-cell gate
# ---------------------------------------------------------------------------
def test_null_cell_gate_passes_on_zero_te(signal_kbar_fixture):
    r"""With no coupling the null-cell mean K-bar sits below the loose ceiling."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v4 import (
        null_cell_gate_v4,
    )

    data = signal_kbar_fixture(gamma=0.2, noise=0.02, reps=60)
    arrs = _arrs_from_kbar(data["kbar"], data["te_true"])
    gate = null_cell_gate_v4(arrs, ceiling=0.5)
    assert gate["pass"] is True
    assert gate["n_cells"] >= 1 and "mean" in gate


def test_null_cell_gate_fails_when_null_kbar_exceeds_ceiling():
    r"""A null cell whose mean $\bar K$ sits above the ceiling must FAIL the gate (failing direction).

    Mirrors :func:`test_null_cell_gate_passes_on_zero_te` in the opposite direction so a
    ``null_cell_gate_v4`` that could never return ``pass=False`` (e.g. the ``mean_null < ceiling``
    comparison regressing to a constant) is caught rather than silently keeping the suite green.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v4 import (
        null_cell_gate_v4,
    )

    rng = np.random.default_rng(3)
    reps = 40
    te = np.repeat([0.0, 1.0, 2.0], reps)
    # The null cell (te==0) carries an inflated K-bar well above the ceiling; signal cells higher still.
    kbar = np.where(te == 0.0, 0.8, 1.5) + rng.normal(0.0, 0.01, size=te.shape)
    arrs = _arrs_from_kbar(kbar, te)
    gate = null_cell_gate_v4(arrs, ceiling=0.5)
    assert gate["pass"] is False
    assert gate["mean"] > 0.5


# ---------------------------------------------------------------------------
# S6-T04: raw prediction-space source control (fabricated source-exploiting pair)
# ---------------------------------------------------------------------------
def test_prediction_control_ordering_holds(source_exploiting_outputs):
    r"""L_feat < L_base < L_feat^pi(U) and shuffle_penalty>0 on the source-exploiting fixture."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v4 import (
        prediction_controls_v4,
    )

    fx = source_exploiting_outputs
    target = fx["target"]

    def _mse(mu):
        return ((np.asarray(mu) - target) ** 2).mean(axis=(1, 2, 3))

    b = target.shape[0]
    arrs = {
        "cell_id": np.zeros(b, dtype=np.int64),
        "te_inj": np.full(b, 2.0),                       # signal cell
        "feat_loss": _mse(fx["clean"]["mu_full"]),
        "base_loss": _mse(fx["clean"]["mu_base"]),
        "feat_loss_shuffled": _mse(fx["permuted"]["mu_full"]),
    }
    res = prediction_controls_v4(arrs)
    ov = res["overall"]
    assert ov["ordering_pass"] is True
    assert ov["feat_loss"] < ov["base_loss"] < ov["feat_loss_shuffled"]
    assert ov["shuffle_penalty_shuffled"] > 0.0


# ---------------------------------------------------------------------------
# S6-T05: lag recovery vs planted D
# ---------------------------------------------------------------------------
def test_lag_recovery_matches_planted_d(planted_lag_te_lag_map):
    r"""The recovered argmax lag matches the planted D within tolerance; a null cell does not peak."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v4 import (
        recover_lag_v4,
    )

    fx = planted_lag_te_lag_map
    D = int(fx["planted_lag"])
    te_lag = np.asarray(fx["te_lag_map"])                 # (B, T, L)
    signal_profile = te_lag.mean(axis=(0, 1))            # (L,)
    L = signal_profile.shape[0]
    uniform_profile = np.full(L, 1.0 / L)

    # Two cells: cell 0 signal (planted bump), cell 1 null (flat profile), both at delay D.
    arrs = {
        "cell_id": np.array([0, 1], dtype=np.int64),
        "te_inj": np.array([2.0, 0.0]),
        "te_scat": np.array([np.nan, np.nan]),
        "kbar": np.array([1.0, 0.0]),
        "delay": np.array([D, D], dtype=np.int64),
        "held_out": np.array([0, 0], dtype=np.int64),
        "lag_profiles": {0: signal_profile, 1: uniform_profile},
    }
    rec = recover_lag_v4(arrs, horizon=30, tolerance=1)
    sig = rec["per_cell"][0]
    assert abs(int(sig["peak_lag"]) - D) <= 1, f"recovered {sig['peak_lag']} != planted {D}"
    # The uniform null profile has no concentrated peak: its band mass is far below the signal's.
    nul = rec["per_cell"][1]
    assert nul["lag_mass"] < sig["lag_mass"]


# ---------------------------------------------------------------------------
# S6-T06: run_eval_v4 assembly (metrics.json + per_sample_eval.npz)
# ---------------------------------------------------------------------------
def test_run_eval_writes_metrics(tiny_cache_v4, tmp_path, monkeypatch):
    r"""The eval stage writes metrics.json (all gate keys) + per_sample_eval.npz off a checkpoint."""
    import functools
    import json

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import eval_v4
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_runner_v4 import (
        _build_runner_and_loader_v4,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v4 import (
        StageContextV4,
    )

    # Place the checkpoint where _resolve_eval_checkpoint looks (run_dir/final.ckpt).
    config = tiny_cache_v4["config"]
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v2 import (
        _run_dir,
        _results_dir,
    )

    # Redirect results under tmp_path so the stage writes into an isolated tree.
    config = dict(config)
    config.setdefault("paths", {})
    config["paths"] = {**config.get("paths", {}), "results_dir": str(tmp_path / "results")}

    run_dir = _run_dir(config, "G1_raw_v4", "prod")
    _save_ckpt(run_dir / "final.ckpt")

    # Point the runner/loader at the tiny cache (run_eval_v4 does not thread a cache_dir).
    monkeypatch.setattr(
        eval_v4, "_build_runner_and_loader_v4",
        functools.partial(_build_runner_and_loader_v4, cache_dir=tiny_cache_v4["cache_dir"]))

    # Pre-write a realizability.json so the te_raw gate reads it instead of recomputing.
    results_dir = _results_dir(config, "G1_raw_v4")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "realizability.json", "w", encoding="utf-8") as handle:
        json.dump({"gate": {"passed": True}, "constants": {}}, handle)

    ctx = StageContextV4(config=config, benchmark="G1_raw_v4", arm="prod", pilot=True)
    rc = eval_v4.run_eval_v4(ctx)
    assert rc == 0

    metrics_path = run_dir / "metrics.json"
    assert metrics_path.is_file()
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    for key in ("calibration", "null_cell_gate", "prediction_controls", "lag_recovery",
                "te_raw_gate", "model_class", "arm", "render_mode"):
        assert key in metrics, f"metrics.json missing {key}"
    assert metrics["arm"] == "prod"
    assert (run_dir / "per_sample_eval.npz").is_file()
