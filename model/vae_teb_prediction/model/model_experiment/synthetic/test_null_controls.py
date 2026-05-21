"""Pytest checks for ``null_controls`` and the ``benchmark_override`` plumbing.

Sprint 6.4 / 6.5 coverage:

    * the two new benchmark blocks (``G2_wrong_delay`` / ``G2_zero_coupling``)
      dispatch to ``gen_smooth_arx`` with the YAML overrides applied;
    * ``gen_smooth_arx`` short-circuits ``te_true`` to ``0`` when $c = 0$;
    * ``evaluate_checkpoint(benchmark_override=...)`` resolves the right
      cache directory and never warns on the (intended) ``te_true`` divergence;
    * ``run_null_controls`` produces ``summary.csv`` + ``metrics.json`` with
      one row per requested control on a stand-in untrained checkpoint.

Run from the repo root with ``python -m pytest``.
"""

import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from model.vae_teb_prediction.model.model_experiment.synthetic.build_dataset import (
    _GENERATORS,
    build_dataset,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.dataset import (
    SyntheticTEDataset,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.evaluate_te import (
    evaluate_checkpoint,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.null_controls import (
    _DEFAULT_CONTROLS,
    _SUMMARY_FIELDS,
    _find_source_ckpt,
    run_null_controls,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (
    resolve_active_benchmark,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1

_CONFIG_PATH = Path(__file__).resolve().parent / "config_synth.yaml"
_N_TRAIN, _N_VAL, _N_TEST = 4, 2, 2


def _tiny_config(
    benchmark: str, tmp_path: Path, *, tag: str | None = None,
) -> Dict[str, Any]:
    """Load ``config_synth.yaml`` resolved at a tiny per-test cache size."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["experiment"]["benchmark"] = benchmark
    raw["experiment"]["tag"] = tag or f"test_{benchmark}"
    config = resolve_active_benchmark(raw)
    config["paths"]["data_dir"] = str(tmp_path / "data")
    config["paths"]["results_dir"] = str(tmp_path / "results")
    config["data"]["n_train"] = _N_TRAIN
    config["data"]["n_val"] = _N_VAL
    config["data"]["n_test"] = _N_TEST
    config["optim"]["batch_size"] = 2
    return config


def _save_stub_ckpt(
    config: Dict[str, Any], cache_meta: Dict[str, Any], run_tag: str,
) -> Path:
    """Save a minimal untrained ``final.ckpt`` mimicking ``train_minimal.save_checkpoint``."""
    ckpt_path = (
        Path(config["paths"]["results_dir"])
        / str(config["experiment"]["benchmark"])
        / run_tag / "final.ckpt"
    )
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = SeqVaeLagAttnV1()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": {},  # defaults match V2-D1 native shapes
            "data_meta": dict(cache_meta),
            "config": {
                "experiment": {
                    "benchmark": str(config["experiment"]["benchmark"]),
                    "tag": str(config["experiment"]["tag"]),
                },
                "optim": {"batch_size": 2},
            },
            "loss_settings": {
                "beta": float(config["loss"]["kld_beta"]),
                "lambda_full": float(config["loss"]["lambda_full"]),
                "lambda_base": float(config["loss"]["lambda_base"]),
                "likelihood": str(config["loss"].get("likelihood", "mse")),
                "sigma_obs": config["loss"].get("sigma_obs", 1.0),
                "free_bits": float(config["loss"].get("free_bits", 0.0)),
            },
            "epoch": 0,
            "latent_stats_fitted": False,
        },
        ckpt_path,
    )
    return ckpt_path


# --- Dispatch + generator semantics -----------------------------------------

def test_control_benchmarks_registered():
    """The two new v2 benchmarks dispatch to ``gen_smooth_arx`` like G2."""
    from model.vae_teb_prediction.model.model_experiment.synthetic.generators import (
        gen_smooth_arx,
    )
    assert _GENERATORS["G2_wrong_delay"] is gen_smooth_arx
    assert _GENERATORS["G2_zero_coupling"] is gen_smooth_arx


def test_g2_zero_coupling_te_true_is_zero(tmp_path):
    """Building the zero-coupling cache yields ``te_true == 0`` by construction."""
    config = _tiny_config("G2_zero_coupling", tmp_path, tag="zc_test")
    out_dir = build_dataset(config, force=True)
    ds = SyntheticTEDataset(out_dir / "test.npz")
    assert ds.meta["benchmark"] == "G2_zero_coupling"
    assert ds.meta["te_true"] == 0.0
    assert ds.meta["c"] == 0.0


def test_g2_wrong_delay_uses_overridden_delay(tmp_path):
    """The wrong-delay cache records ``delay`` >> ``max_lag + horizon``."""
    config = _tiny_config("G2_wrong_delay", tmp_path, tag="wd_test")
    out_dir = build_dataset(config, force=True)
    ds = SyntheticTEDataset(out_dir / "test.npz")
    assert ds.meta["benchmark"] == "G2_wrong_delay"
    assert ds.meta["delay"] == 200
    # te_true is well-defined (closed form ARX) and positive (c != 0); we just
    # check it's a finite scalar -- the headline test is the model evaluation,
    # not the analytic number.
    assert isinstance(ds.meta["te_true"], float)
    assert torch.isfinite(torch.tensor(ds.meta["te_true"]))


# --- evaluate_checkpoint(benchmark_override=...) ----------------------------

def test_benchmark_override_resolves_control_cache(tmp_path, capsys):
    """A G2 ckpt evaluated with ``benchmark_override='G2_zero_coupling'`` loads the right cache."""
    # Build BOTH the source G2 cache (for the stub ckpt's data_meta) and the
    # G2_zero_coupling control cache (the override target).
    g2_cfg = _tiny_config("G2", tmp_path, tag="src_test")
    g2_dir = build_dataset(g2_cfg, force=True)
    g2_meta: Dict[str, Any] = json.loads(
        (g2_dir / "meta.json").read_text(encoding="utf-8"),
    )
    ctrl_cfg = _tiny_config("G2_zero_coupling", tmp_path, tag="zc_test")
    build_dataset(ctrl_cfg, force=True)

    # Re-resolve the G2 config so the stub ckpt records G2 as its benchmark
    # (the build above mutated `config["experiment"]["benchmark"]`).
    g2_cfg["experiment"]["benchmark"] = "G2"
    g2_cfg["experiment"]["tag"] = "src_test"
    ckpt_path = _save_stub_ckpt(g2_cfg, g2_meta, run_tag="src_test_run")

    # Use the SAME shared-paths config (so the override re-uses tmp_path/data).
    row_override = evaluate_checkpoint(
        ckpt_path, g2_cfg, device=torch.device("cpu"),
        data_tag="zc_test", batch_size=2,
        benchmark_override="G2_zero_coupling",
    )
    # The row's benchmark should reflect the override and the te_true should
    # come from the control cache (0.0), not the checkpoint's recorded value.
    assert row_override["benchmark"] == "G2_zero_coupling"
    assert row_override["data_tag"] == "zc_test"
    assert row_override["te_true"] == 0.0

    # The te_true divergence-warning must NOT have fired (Sprint 6.4.3a
    # explicitly skips it under the override).
    out = capsys.readouterr().out
    assert "[warn] checkpoint te_true=" not in out


# --- run_null_controls orchestrator ------------------------------------------

def test_find_source_ckpt_returns_newest(tmp_path):
    """``_find_source_ckpt`` picks the most-recent ``final.ckpt`` when no tag is given."""
    root = tmp_path / "results"
    bench = "G2"
    (root / bench / "older").mkdir(parents=True)
    (root / bench / "newer").mkdir(parents=True)
    older = root / bench / "older" / "final.ckpt"
    newer = root / bench / "newer" / "final.ckpt"
    older.write_bytes(b"x")
    newer.write_bytes(b"x")
    import os
    os.utime(older, (1_000_000_000, 1_000_000_000))
    os.utime(newer, (2_000_000_000, 2_000_000_000))
    chosen = _find_source_ckpt(root, bench, None)
    assert chosen == newer

    # Explicit tag bypasses the newest-pick logic.
    chosen_old = _find_source_ckpt(root, bench, "older")
    assert chosen_old == older

    # Missing run tag returns None.
    assert _find_source_ckpt(root, bench, "nonexistent") is None


def test_run_null_controls_smoke(tmp_path):
    """End-to-end orchestrator pass: re-evaluate the source ckpt on pre-built tiny control caches."""
    # Step 1: build a tiny G2 source cache and save a stub ckpt under it.
    g2_cfg = _tiny_config("G2", tmp_path, tag="src_test")
    g2_dir = build_dataset(g2_cfg, force=True)
    g2_meta: Dict[str, Any] = json.loads(
        (g2_dir / "meta.json").read_text(encoding="utf-8"),
    )
    g2_cfg["experiment"]["benchmark"] = "G2"
    g2_cfg["experiment"]["tag"] = "src_test"
    _save_stub_ckpt(g2_cfg, g2_meta, run_tag="src_test_run")

    # Step 2: pre-build the two control caches at tiny per-test sizes. The
    # orchestrator's ``build_missing=True`` path would otherwise read the
    # n_train/n_val/n_test defaults from `config_synth.yaml` (8000/1000/2000)
    # and spend many minutes per cache -- impractical for a unit test. By
    # building the caches here at tiny sizes, the orchestrator just needs to
    # find them on disk and do the (cheap) re-evaluation step.
    for ctrl_label, ctrl_bench, ctrl_tag in _DEFAULT_CONTROLS:
        ctrl_cfg = _tiny_config(ctrl_bench, tmp_path, tag=ctrl_tag)
        # Reuse the same data_dir so the orchestrator sees the caches.
        build_dataset(ctrl_cfg, force=True)

    # Step 3: invoke the orchestrator with build_missing=False so it just
    # re-evaluates the source ckpt on the pre-built tiny caches.
    g2_cfg["null_controls"] = {
        "source_benchmark": "G2",
        "source_run_tag": "src_test_run",
        "controls": [list(c) for c in _DEFAULT_CONTROLS],
        "out_dir": "null_controls",
    }
    result = run_null_controls(
        g2_cfg, device=torch.device("cpu"), build_missing=False,
    )
    assert len(result["rows"]) == 2
    assert result["skipped"] == []
    assert result["source_benchmark"] == "G2"

    # Step 3: assert the two artifacts exist and parse correctly.
    out_dir = Path(result["out_dir"])
    assert (out_dir / "summary.csv").is_file()
    assert (out_dir / "metrics.json").is_file()
    metrics = json.loads(
        (out_dir / "metrics.json").read_text(encoding="utf-8"),
    )
    assert set(metrics["controls"].keys()) == {"wrong_delay", "zero_coupling"}
    for label in ("wrong_delay", "zero_coupling"):
        for key in (
            "k_bar", "k_bar_shuffled", "k_bar_reversed", "pred_gap", "te_true",
        ):
            assert key in metrics["controls"][label], f"{label}/{key}"
    # Zero-coupling te_true is exactly 0 (closed-form short-circuit).
    assert metrics["controls"]["zero_coupling"]["te_true"] == 0.0
    # Schema sanity: the row dict matches the _SUMMARY_FIELDS contract.
    for row in result["rows"]:
        for field in _SUMMARY_FIELDS:
            assert field in row, field


def test_run_null_controls_no_source_ckpt(tmp_path):
    """Missing source checkpoint yields an empty result + JSON error record."""
    g2_cfg = _tiny_config("G2", tmp_path, tag="src_test")
    g2_cfg["null_controls"] = {
        "source_benchmark": "G2",
        "source_run_tag": None,
        "controls": [list(c) for c in _DEFAULT_CONTROLS],
        "out_dir": "null_controls",
    }
    result = run_null_controls(
        g2_cfg, device=torch.device("cpu"), build_missing=False,
    )
    assert result["rows"] == []
    assert result["source_ckpt"] is None
    assert set(result["skipped"]) == {"wrong_delay", "zero_coupling"}
    metrics = json.loads(
        (Path(result["out_dir"]) / "metrics.json").read_text(encoding="utf-8"),
    )
    assert "error" in metrics
    assert metrics["controls"] == {}
