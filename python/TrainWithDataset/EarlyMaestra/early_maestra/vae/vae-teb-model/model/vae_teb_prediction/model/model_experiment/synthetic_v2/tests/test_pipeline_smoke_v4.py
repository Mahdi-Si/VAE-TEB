r"""S7-T07: the full-pipeline smoke test (slow).

Drives the real ``run_pipeline_v4`` driver end to end on a tiny concentrated cell-set and **two**
arms in pilot mode -- build -> train -> eval -> arms_report -- and asserts the cross-arm
``arms_report_v4.md`` is produced with exactly two arm rows. This is the orchestration wiring test
(stage chaining, per-arm dispatch, split fan-out, cross-arm aggregation); the model-quality gates
are the manual S8-T01 headline sweep.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.v4, pytest.mark.slow]

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_synth_v4.yaml"

#: The two arms the smoke sweeps: the causal headline arm and the no-source ablation.
_SMOKE_ARMS = ("prod", "disable_source")


def _smoke_config(tmp_path: Path) -> dict:
    r"""A tiny, self-contained config: tmp data/results dirs, two arms, mlflow off."""
    config = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["experiment"] = {**config["experiment"], "tag": "smoke_v4",
                            "data_tag": "smoke_v4_direct"}
    config["paths"] = {"data_dir": str(tmp_path / "data"),
                       "results_dir": str(tmp_path / "results")}
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path / "train_out")
    # Two arms only -> the report tabulates exactly two rows.
    config["arms"] = {
        "prod": {},
        "disable_source": {"model_config": {"VAE_model": {"disable_source": True}}},
    }
    # No MLflow server on the dev/CI box.
    config["advanced_config"]["tracking"]["mlflow"]["enabled"] = False
    return config


def test_pipeline_smoke_build_train_eval_arms_report(tmp_path):
    r"""build -> train (2 arms) -> eval (2 arms) -> arms_report yields a 2-row arms_report_v4.md."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import run_pipeline_v4 as rp
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v4 import (
        build_all_v4,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import (
        resolve_cache_dir,
    )

    config = _smoke_config(tmp_path)
    cfg_path = tmp_path / "config_smoke_v4.yaml"
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    # --- build: a tiny cache at the config-resolved path (one null + one signal cell) ---------
    cache_dir = Path(resolve_cache_dir(config, benchmark="G1_raw_v4"))
    build_all_v4(config, benchmark="G1_raw_v4", out_dir=cache_dir,
                 grid_override={"target_te_grid": [0.0, 2.0], "lag_grid": [8]},
                 n_override={"train": 8, "val": 4, "test": 4})
    assert (cache_dir / "train.npz").is_file()

    argv_base = ["--config", str(cfg_path)]

    # --- train each arm in-process (explicit --arm bypasses the DDP subprocess sweep) ---------
    for arm in _SMOKE_ARMS:
        rc = rp.main(argv_base + ["--stage", "train", "--arm", arm, "--pilot"])
        assert rc == 0, f"train failed for {arm}"
        run_dir = rp._run_dir(config, "G1_raw_v4", arm)
        assert (run_dir / "final.ckpt").is_file(), f"no checkpoint for {arm}"

    # Pre-write a realizability.json so eval's te_raw gate reads it instead of recomputing.
    import json

    results_root = rp._results_dir(config, "G1_raw_v4")
    results_root.mkdir(parents=True, exist_ok=True)
    with open(results_root / "realizability.json", "w", encoding="utf-8") as handle:
        json.dump({"gate": {"passed": True}, "constants": {}}, handle)

    # --- eval each arm on the val split -> per-arm/<split>/metrics.json -----------------------
    for arm in _SMOKE_ARMS:
        rc = rp.main(argv_base + ["--stage", "eval", "--arm", arm, "--pilot", "--split", "val"])
        assert rc == 0, f"eval failed for {arm}"
        metrics = rp._run_dir(config, "G1_raw_v4", arm) / "val" / "metrics.json"
        assert metrics.is_file(), f"no metrics for {arm}"

    # --- arms_report: one cross-arm table at the tag root ------------------------------------
    rc = rp.main(argv_base + ["--stage", "arms_report", "--split", "val"])
    assert rc == 0

    report = rp._results_dir(config, "G1_raw_v4") / "arms_report_v4.md"
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    for arm in _SMOKE_ARMS:
        assert f"[`{arm}`]" in text, f"{arm} missing from arms_report_v4.md"
    # Exactly two arm rows (each arm's row links its report.md).
    assert text.count("/val/report.md") == 2
