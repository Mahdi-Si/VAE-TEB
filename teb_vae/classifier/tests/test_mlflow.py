"""T-F4: MLflow parent/child nesting (SPEC §10.10.5, F8-F10) on a local sqlite store, and the fail-closed parent (F9).
Fold 1 of the fixture tree, one epoch per unit."""
from __future__ import annotations

import json
import sys

import pytest
import yaml

from teb_vae.classifier.tests.conftest import SMOKE_CONFIG

FAST = ["classifier.run.folds=[1]", "classifier.train.max_epochs=1"]


def _restore_loguru() -> None:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr)


def _on(uri: str, *extra: str) -> list:
    return ["advanced_config.tracking.mlflow.enabled=true", f"advanced_config.tracking.mlflow.tracking_uri={uri}",
            "advanced_config.tracking.mlflow.experiment_name=t-f4", *extra]


def test_unit_configs_nest_under_the_parent(smoke_overrides, tmp_path):
    from teb_vae.classifier.config import load, unit_config

    cfg = load(SMOKE_CONFIG, list(smoke_overrides) + _on("sqlite:///x.db"))
    for kind in ("model", "shuffled"):
        m = unit_config(cfg, run_dir=tmp_path, fold=2, seed=42, kind=kind,
                        mlflow_parent_id="parent-1")["advanced_config"]["tracking"]["mlflow"]
        assert m["tags"] == {"mlflow.parentRunId": "parent-1", "fold": "2", "seed": "42", "kind": kind}
        assert (m["enabled"], m["log_model"], m["log_checkpoints"]) == (True, False, False)  # F10
        assert m["run_name"] == f"fold2-seed42-{kind}"


@pytest.fixture(scope="module")
def tracked(smoke_overrides, tmp_path_factory):
    """Every stage on fold 1 with MLflow on a fresh sqlite store: ``(tracking uri, run dir)``."""
    from teb_vae.classifier import run

    tmp = tmp_path_factory.mktemp("mlflow")
    uri = f"sqlite:///{tmp / 'mlflow.db'}"
    overrides = list(smoke_overrides) + FAST + _on(uri, f"advanced_config.tracking.mlflow.artifact_location="
                                                        f"{tmp / 'artifacts'}")
    try:
        run.main(config=str(SMOKE_CONFIG), stage="all", overrides=overrides, run_dir=str(tmp / "run"))
    finally:
        _restore_loguru()
    return uri, tmp / "run"


@pytest.mark.slow
def test_one_parent_one_nested_child_per_unit_and_results_on_the_parent(tracked):
    from mlflow import MlflowClient

    uri, run_dir = tracked
    client = MlflowClient(tracking_uri=uri)
    runs = client.search_runs([client.get_experiment_by_name("t-f4").experiment_id], max_results=100)
    parents = [r for r in runs if r.data.tags.get("kind") == "parent"]
    assert len(parents) == 1 and len(runs) == 3
    parent, pid = parents[0], parents[0].info.run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["mlflow"] == {"enabled": True, "status": "ok", "parent_run_id": pid, "reason": None}

    children = [r for r in runs if r.info.run_id != pid]
    assert sorted((r.data.tags["fold"], r.data.tags["seed"], r.data.tags["kind"]) for r in children) == [
        ("1", "42", "model"), ("1", "42", "shuffled")]
    assert all(r.data.tags["mlflow.parentRunId"] == pid for r in children)
    for r in children:  # Lightning's epoch metrics, then the val selections of §10.10.1 step 4
        assert {"val/guid_logloss", "val/thr_np30", "val/sens_np30", "val/fpr_np30", "calib_temperature",
                "val/thr_youden"} <= set(r.data.metrics)
        assert r.data.params["model_config.classifier.labels.task"] == "adverse_vs_healthy"
        assert not client.search_model_versions(f"run_id='{r.info.run_id}'")  # F10

    tags = parent.data.tags
    assert {"git_commit", "torch_version", "host", "precision"} <= set(tags)  # GraphModelBase provenance
    assert (tags["task"], tags["source_kind"], tags["n_folds"], tags["seeds"]) == ("adverse_vs_healthy", "hdf5", "1",
                                                                                   "42")
    assert parent.data.params["classifier.labels.task"] == "adverse_vs_healthy"
    assert {"pooled_test/guid/auroc", "foldmean_test/guid/auroc", "pooled_test/guid/sens_at_np30",
            "pooled_test/guid/fpr_at_np30", "pooled_test/segment/auroc", "probe/pooled_test/guid/auroc",
            "shortcut/pooled_test/guid/auroc", "shuffled/pooled_test/guid/auroc"} <= set(parent.data.metrics)
    artifacts = {a.path for a in client.list_artifacts(pid)}
    assert {"config.resolved.yaml", "summary.md", "summary.json", "tables", "figures"} <= artifacts
    assert parent.info.status == "FINISHED"


@pytest.mark.slow
def test_parent_is_fail_closed(smoke_overrides, tmp_path, monkeypatch):
    """The tracking client raises: the run trains with MLflow off everywhere and records why (F9)."""
    import mlflow

    from teb_vae.classifier import run
    from teb_vae.classifier.config import unit_dir

    class Unreachable:
        def __init__(self, *args, **kwargs):
            raise ConnectionError("tracking server down")

    monkeypatch.setattr(mlflow, "MlflowClient", Unreachable)
    overrides = list(smoke_overrides) + FAST + ["classifier.baselines.shuffled_control=false"] + _on(
        "http://127.0.0.1:9")
    try:
        for stage in ("cohort", "extract", "train"):
            run.main(config=str(SMOKE_CONFIG), stage=stage, overrides=overrides, run_dir=str(tmp_path / "run"))
    finally:
        _restore_loguru()
    run_dir = tmp_path / "run"
    status = json.loads((run_dir / "manifest.json").read_text())["mlflow"]
    assert (status["enabled"], status["status"], status["parent_run_id"]) == (False, "unreachable", None)
    assert "tracking server down" in status["reason"]
    assert "F9: MLflow parent run could not be created" in (run_dir / "run.log").read_text()
    train = json.loads((run_dir / "stage_state.json").read_text())["train"]
    assert (train["status"], train["exit_code"], train["units"]["model|42|1"]["status"]) == ("done", 0, "done")
    out = unit_dir(run_dir, 1, 42, "model")
    unit = yaml.safe_load((out / "model_checkpoints" / "resolved_config.yaml").read_text())
    tracking = unit["advanced_config"]["tracking"]["mlflow"]
    assert tracking["enabled"] is False and "mlflow.parentRunId" not in tracking["tags"]
    assert (out / "selection_lock.json").is_file()
    assert json.loads((out / "fold_results.json").read_text())["mlflow_run_id"] is None
