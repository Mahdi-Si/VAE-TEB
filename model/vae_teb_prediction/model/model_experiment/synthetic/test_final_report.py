"""Pytest checks for the Sprint 6.6 ``final_report`` extensions.

Covers the v2 benchmark set (``G1`` / ``G2`` / ``G3``), the new null-control
rows (shuffle / reverse / wrong-delay / zero-coupling) and the claim-tier
``null_controls`` criterion. The headline figure is skipped here -- the
plot is exercised by hand at the end of the runtime gate.

The tests synthesise the per-phase JSON artifacts directly so they need no
trained model and run in milliseconds.

Run from the repo root with ``python -m pytest``.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from model.vae_teb_prediction.model.model_experiment.synthetic.final_report import (
    _BENCHMARKS,
    _benchmark_rows,
    _claim_tier,
    _collate,
    _status_info,
    build_final_report,
)


_DEFAULT_BENCHES = ("G1", "G2", "G3")


def test_benchmarks_constant_is_v2():
    """Sprint 6.6 swap: the headline iterates over the v2 benchmark set."""
    assert _BENCHMARKS == _DEFAULT_BENCHES


def test_status_info_helper():
    """``_status_info`` returns INFO for finite values, DEFERRED otherwise."""
    assert _status_info(0.0) == "INFO"
    assert _status_info(0.42) == "INFO"
    assert _status_info(None) == "DEFERRED"
    assert _status_info(float("nan")) == "DEFERRED"
    assert _status_info(float("inf")) == "DEFERRED"


# --- Synthetic artifact helpers ----------------------------------------------

def _eval_metrics(*, te_zero_k_bar: float = 0.001, k_shuf: float = 0.001,
                  k_rev: float = 0.001, gamma: float = 1.0,
                  rho: float = 0.99) -> Dict[str, Any]:
    """Build a stand-in ``eval_te/metrics.json`` payload."""
    return {
        "n_settings": 5,
        "metrics": {
            "metric1_null": {
                "E_0": te_zero_k_bar,
                "smallest_signal_k_bar": 0.05,
                "null_signal_ratio": 0.02,
                "k_bar_shuffled_mean": k_shuf,
                "k_bar_reversed_mean": k_rev,
            },
            "metric2_spearman": rho,
            "metric3_calibration": {"alpha": 0.0, "gamma": gamma, "r2": 0.98},
            "metric4_pred_gain": {
                "mean_pred_gap_te0": 0.0001,
                "mean_pred_gap_te_pos": 0.05,
                "verdict_te0_near_zero": True,
                "verdict_te_pos_positive": True,
            },
        },
    }


def _lag_metrics(*, ratio: float = 5.0, lolo: float = 0.9) -> Dict[str, Any]:
    """Build a stand-in ``lag_recovery/metrics.json`` payload."""
    return {
        "task_5_2_lag_mass_attn": {"ratio_to_uniform": ratio},
        "task_5_4_lolo": {"lag_mass_lolo": lolo},
    }


def _null_controls_metrics(*, wrong_delay_k: float = 0.1,
                           zero_coupling_k: float = 0.001) -> Dict[str, Any]:
    """Build a stand-in ``null_controls/metrics.json`` payload."""
    return {
        "source_benchmark": "G2",
        "source_run_tag": "test_run",
        "source_ckpt": "<test>",
        "controls": {
            "wrong_delay":   {"k_bar": wrong_delay_k, "te_true": 0.5},
            "zero_coupling": {"k_bar": zero_coupling_k, "te_true": 0.0},
        },
        "skipped": [],
    }


def _directionality_metrics(*, ratio: float = 10.0,
                            verdict: bool = True) -> Dict[str, Any]:
    """Build a stand-in ``directionality/metrics.json`` payload."""
    return {
        "comparison": {
            "k_bar_forward": 0.1,
            "k_bar_reverse": 0.1 / ratio,
            "directionality_ratio": ratio,
            "verdict_direction_specific": verdict,
            "te_true_forward": 0.5,
            "te_true_reverse": 0.0,
        },
        "rows": [],
        "skipped": [],
    }


def _calibration_metrics(
    *, benchmark: str = "G1",
    gamma: float = 1.0,
    alpha: float = 0.0,
    beta_star: float = 1e-3,
    te_points: tuple = (1.5, 4.5, 9.0),
    k_bar_points: tuple = (1.5, 4.5, 9.0),
) -> Dict[str, Any]:
    """Build a stand-in ``calibration/calibration.json`` payload.

    Mirrors the slim ``cells`` + ``selected`` block that the headline figure
    reads from. Defaults give a perfect calibration ($\\gamma=1$, $\\alpha=0$,
    perfect y=x identity) so panel (i) plots a non-degenerate scatter.
    """
    return {
        "benchmark": benchmark,
        "te_points": [
            {"te_per_step_target": float(te / 30.0), "data_tag": f"x_{i}",
             "knob_name": "B_y", "knob_value": 0.5,
             "te_block_target": float(te), "te_block_achieved": float(te),
             "te_per_step_achieved": float(te / 30.0)}
            for i, te in enumerate(te_points)
        ],
        "betas": [beta_star],
        "table": [{"beta": beta_star, "alpha": alpha, "gamma": gamma,
                   "r2": 1.0, "n": len(te_points)}],
        "selected": {"beta": beta_star, "alpha": alpha, "gamma": gamma,
                     "r2": 1.0, "score": abs(gamma - 1.0),
                     "rationale": "test"},
        "skipped": [],
        "n_rows": len(te_points),
        "cells": [
            {"te_per_step_target": float(te / 30.0),
             "data_tag": f"x_{i}", "knob_value": 0.5,
             "te_true_block": float(te), "beta": float(beta_star),
             "k_bar": float(k)}
            for i, (te, k) in enumerate(zip(te_points, k_bar_points))
        ],
    }


# Per-subdir filename map. ``calibration`` writes ``calibration.json`` (not
# ``metrics.json``); ``beta_sweep`` writes ``analysis.json``. Every other
# subdir uses the default ``metrics.json``.
_SUBDIR_FILENAME = {
    "beta_sweep": "analysis.json",
    "calibration": "calibration.json",
}


def _write_results_tree(
    root: Path,
    *,
    benches: Dict[str, Dict[str, Any]],
    directionality: Dict[str, Any] | None = None,
) -> None:
    """Materialise a synthetic ``results/`` directory tree of JSON artifacts."""
    for bench, parts in benches.items():
        for sub, payload in parts.items():
            target = root / bench / sub / _SUBDIR_FILENAME.get(
                sub, "metrics.json"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload), encoding="utf-8")
    if directionality is not None:
        target = root / "directionality" / "metrics.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(directionality), encoding="utf-8")


# --- _benchmark_rows tests ---------------------------------------------------

def test_benchmark_rows_emit_four_control_metrics():
    """A G2 row block contains shuffle / reverse / wrong_delay / zero_coupling rows."""
    art = {
        "eval_te": _eval_metrics(),
        "lag_recovery": _lag_metrics(),
        "beta_sweep": None,
        "null_controls": _null_controls_metrics(),
    }
    rows = _benchmark_rows("G2", art)
    metrics = {r["metric"] for r in rows}
    assert {"k_bar_shuffled", "k_bar_reversed",
            "k_bar_wrong_delay", "k_bar_zero_coupling"}.issubset(metrics)
    # Status sanity: collapse rows PASS at <= 0.05, wrong_delay is INFO.
    by_metric = {r["metric"]: r for r in rows}
    assert by_metric["k_bar_shuffled"]["status"] == "PASS"
    assert by_metric["k_bar_reversed"]["status"] == "PASS"
    assert by_metric["k_bar_zero_coupling"]["status"] == "PASS"
    assert by_metric["k_bar_wrong_delay"]["status"] == "INFO"


def test_benchmark_rows_fail_when_collapse_breaks():
    """A non-collapsing K_bar (>0.05) reports FAIL on the gated rows."""
    art = {
        "eval_te": _eval_metrics(k_shuf=0.2, k_rev=0.2),
        "lag_recovery": _lag_metrics(),
        "beta_sweep": None,
        "null_controls": _null_controls_metrics(zero_coupling_k=0.3),
    }
    rows = _benchmark_rows("G2", art)
    by_metric = {r["metric"]: r for r in rows}
    assert by_metric["k_bar_shuffled"]["status"] == "FAIL"
    assert by_metric["k_bar_reversed"]["status"] == "FAIL"
    assert by_metric["k_bar_zero_coupling"]["status"] == "FAIL"
    # Wrong-delay stays INFO regardless of the value.
    assert by_metric["k_bar_wrong_delay"]["status"] == "INFO"


def test_benchmark_rows_defer_when_artifacts_missing():
    """All control rows DEFER when their source JSONs are not present."""
    art: Dict[str, Any] = {
        "eval_te": None, "lag_recovery": None,
        "beta_sweep": None, "null_controls": None,
    }
    rows = _benchmark_rows("G2", art)
    by_metric = {r["metric"]: r for r in rows}
    for metric in ("k_bar_shuffled", "k_bar_reversed",
                   "k_bar_wrong_delay", "k_bar_zero_coupling"):
        assert by_metric[metric]["status"] == "DEFERRED", metric


# --- _claim_tier tests -------------------------------------------------------

def _full_pass_table() -> List[Dict[str, Any]]:
    """Assemble a fully-PASS table for G1 / G2 / G3 across every gated metric."""
    table: List[Dict[str, Any]] = []
    for b in _DEFAULT_BENCHES:
        art = {
            "eval_te": _eval_metrics(),
            "lag_recovery": _lag_metrics(),
            "beta_sweep": None,
            "null_controls": _null_controls_metrics(),
        }
        table.extend(_benchmark_rows(b, art))
    return table


def test_claim_tier_strong_requires_null_controls_pass():
    """A fully-PASS table with directionality returns ``strong``."""
    table = _full_pass_table()
    claim = _claim_tier(table, _directionality_metrics())
    assert claim["tier"] == "strong"
    assert claim["criteria"]["null_controls"] is True
    assert claim["criteria"]["directionality"] is True


def test_claim_tier_moderate_when_lag_fails_but_controls_pass():
    """Lag failing with everything else passing yields ``moderate``."""
    # Build a table where lag_mass_ratio_to_uniform FAILS but the rest PASS.
    table: List[Dict[str, Any]] = []
    for b in _DEFAULT_BENCHES:
        art = {
            "eval_te": _eval_metrics(),
            "lag_recovery": _lag_metrics(ratio=0.5, lolo=0.1),   # FAIL
            "beta_sweep": None,
            "null_controls": _null_controls_metrics(),
        }
        table.extend(_benchmark_rows(b, art))
    # Directionality PASS so it doesn't drag the tier down.
    claim = _claim_tier(table, _directionality_metrics())
    assert claim["criteria"]["lag"] is False
    assert claim["criteria"]["null_controls"] is True
    assert claim["tier"] == "moderate"


def test_claim_tier_deferred_when_table_empty():
    """A table with no decided criteria yields ``deferred``."""
    claim = _claim_tier([], None)
    assert claim["tier"] == "deferred"


# --- _collate + build_final_report end-to-end --------------------------------

def test_collate_reads_null_controls(tmp_path):
    """``_collate`` discovers the per-benchmark ``null_controls/metrics.json``."""
    _write_results_tree(
        tmp_path,
        benches={
            "G2": {
                "eval_te": _eval_metrics(),
                "lag_recovery": _lag_metrics(),
                "null_controls": _null_controls_metrics(),
            }
        },
        directionality=_directionality_metrics(),
    )
    collated = _collate(tmp_path)
    assert collated["benchmarks"]["G2"]["null_controls"] is not None
    assert collated["directionality"]["comparison"]["directionality_ratio"] == 10.0
    # Missing benchmarks come back as None slots, not absent keys.
    assert collated["benchmarks"]["G1"]["null_controls"] is None


def test_build_final_report_end_to_end(tmp_path):
    """``build_final_report`` emits ``report_table.csv`` + ``report.json`` for v2."""
    results_root = tmp_path / "results"
    _write_results_tree(
        results_root,
        benches={
            b: {
                "eval_te": _eval_metrics(),
                "lag_recovery": _lag_metrics(),
                "null_controls": _null_controls_metrics(),
            } for b in _DEFAULT_BENCHES
        },
        directionality=_directionality_metrics(),
    )
    # The headline figure import (matplotlib) is heavy; we test via the
    # public ``build_final_report`` and accept the figure render too -- it
    # exercises the v2 ``_BENCHMARKS`` panel iteration.
    config = {"paths": {"results_dir": "."}}  # resolved via results_root override
    result = build_final_report(config, results_root=results_root)

    assert "table" in result and "claim_tier" in result
    out_dir = Path(result["out_dir"])
    assert (out_dir / "report_table.csv").is_file()
    assert (out_dir / "report.json").is_file()
    # The table must contain rows for every (benchmark, control_metric) pair.
    metrics_seen = {(r["benchmark"], r["metric"]) for r in result["table"]}
    for b in _DEFAULT_BENCHES:
        for m in ("k_bar_shuffled", "k_bar_reversed",
                  "k_bar_wrong_delay", "k_bar_zero_coupling"):
            assert (b, m) in metrics_seen, (b, m)
    # The directionality row is keyed under the literal "directionality" tag.
    assert any(
        r["benchmark"] == "directionality"
        and r["metric"] == "directionality_ratio"
        for r in result["table"]
    )
    # Claim tier should land on a real value -- with the controlled inputs
    # above it is "strong".
    assert result["claim_tier"]["tier"] in ("strong", "moderate")
    assert result["claim_tier"]["criteria"]["null_controls"] is True


# --- Headline figure: calibration JSON collation -----------------------------

def test_collate_reads_calibration(tmp_path):
    """``_collate`` discovers the per-benchmark ``calibration/calibration.json``."""
    _write_results_tree(
        tmp_path,
        benches={
            "G1": {"calibration": _calibration_metrics(benchmark="G1")},
            "G2": {"calibration": _calibration_metrics(benchmark="G2",
                                                       beta_star=1e-2)},
        },
    )
    collated = _collate(tmp_path)
    assert collated["benchmarks"]["G1"]["calibration"] is not None
    assert collated["benchmarks"]["G1"]["calibration"]["selected"]["gamma"] == 1.0
    assert collated["benchmarks"]["G2"]["calibration"] is not None
    # G3 has no calibration JSON; the slot must still exist and be None.
    assert collated["benchmarks"]["G3"]["calibration"] is None


def test_build_final_report_present_flag_for_calibration(tmp_path):
    """``report.json.collated_present.<b>.calibration`` reflects the file presence."""
    results_root = tmp_path / "results"
    _write_results_tree(
        results_root,
        benches={
            "G1": {
                "eval_te": _eval_metrics(),
                "lag_recovery": _lag_metrics(),
                "null_controls": _null_controls_metrics(),
                "calibration": _calibration_metrics(benchmark="G1"),
            },
            "G2": {
                "eval_te": _eval_metrics(),
                "lag_recovery": _lag_metrics(),
                "null_controls": _null_controls_metrics(),
            },
            "G3": {
                "eval_te": _eval_metrics(),
                "lag_recovery": _lag_metrics(),
                "null_controls": _null_controls_metrics(),
            },
        },
        directionality=_directionality_metrics(),
    )
    config = {"paths": {"results_dir": "."}}
    result = build_final_report(config, results_root=results_root)
    report = json.loads(
        (Path(result["out_dir"]) / "report.json").read_text(encoding="utf-8")
    )
    presence = report["collated_present"]
    assert presence["G1"]["calibration"] is True
    assert presence["G2"]["calibration"] is False
    assert presence["G3"]["calibration"] is False


def test_headline_renders_with_calibration_inputs(tmp_path):
    """End-to-end smoke: headline.pdf is produced when calibration JSONs are
    present and not crash when they are missing."""
    results_root = tmp_path / "results"
    _write_results_tree(
        results_root,
        benches={
            "G1": {
                "eval_te": _eval_metrics(),
                "lag_recovery": _lag_metrics(),
                "null_controls": _null_controls_metrics(),
                "calibration": _calibration_metrics(benchmark="G1"),
            },
            "G2": {
                "eval_te": _eval_metrics(),
                "lag_recovery": _lag_metrics(),
                "null_controls": _null_controls_metrics(),
                "calibration": _calibration_metrics(benchmark="G2",
                                                    beta_star=1e-2),
            },
            "G3": {
                "eval_te": _eval_metrics(),
                "lag_recovery": _lag_metrics(),
                "null_controls": _null_controls_metrics(),
            },
        },
        directionality=_directionality_metrics(),
    )
    config = {"paths": {"results_dir": "."}}
    result = build_final_report(config, results_root=results_root)
    out_dir = Path(result["out_dir"])
    assert (out_dir / "headline.pdf").is_file()
    assert (out_dir / "headline.png").is_file()


def test_headline_renders_when_all_inputs_deferred(tmp_path):
    """An empty results tree must produce a 4-panel deferred headline PDF."""
    results_root = tmp_path / "results"
    results_root.mkdir(parents=True)
    config = {"paths": {"results_dir": "."}}
    result = build_final_report(config, results_root=results_root)
    out_dir = Path(result["out_dir"])
    assert (out_dir / "headline.pdf").is_file()
    # The claim tier is "deferred" because every criterion is None.
    assert result["claim_tier"]["tier"] == "deferred"
