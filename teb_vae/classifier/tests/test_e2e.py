"""The one end-to-end test (SPEC §15 T-X1, T-L2, §14.4): every stage on the three-fold fixture tree, run fold-parallel
over two CPU slots, then the selection lock and the dead-fold record on the finished run."""
from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from teb_vae.classifier.tests.conftest import SMOKE_CONFIG

#: smoke.yaml trains the noind ablation: the fixture's has_tlo missingness is flagged in folds 2-3 (§7.3.4, L10).
UNITS = [f"{kind}|42|{k}" for k in (1, 2, 3) for kind in ("model", "shuffled", "noind")]


def _restore_loguru() -> None:
    from loguru import logger  # run.main replaced loguru's sinks with files under tmp

    logger.remove()
    logger.add(sys.stderr)


def _auroc(m: pd.DataFrame, model_id: str, *, pooled: bool) -> float:
    """Test GUID AUROC: the pooled OOF value, or the per-fold mean (the primary AUROC, SPEC §11.7)."""
    v = m[(m["model_id"] == model_id) & (m["split"] == "test") & (m["level"] == "guid")
          & (m["metric"] == "auroc") & m["subgroup"].isna() & m["policy_id"].isna() & (m["point"] == "n/a")]
    v = v[v["fold"] == "pooled"] if pooled else v[v["fold"] != "pooled"]
    assert len(v) == (1 if pooled else 3), model_id
    return float(v["value"].mean())


@pytest.mark.slow
def test_end_to_end_fold_parallel(smoke_overrides, tmp_path):
    """``--stage all --devices cpu,cpu`` (three folds over two fold processes): every stage succeeds, every unit is
    locked, the planted signal is found and the shuffled-label control is not, verify passes, and each fold keeps its
    own logs with no result part left behind. On the finished run: a unit without its lock refuses predict (T-L2), and
    a fold whose process died before its result leaves that unit ``failed``, so predict lists it missing."""
    from teb_vae.classifier import config, run, verify
    from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import RunStateError

    overrides = list(smoke_overrides) + ["classifier.run.folds=[1,2,3]"]
    try:
        run_dir = run.main(config=str(SMOKE_CONFIG), stage="all", overrides=overrides, devices="cpu,cpu",
                           run_dir=str(tmp_path / "run"))
    finally:
        _restore_loguru()
    state = json.loads((run_dir / "stage_state.json").read_text())
    assert list(state) == list(run.STAGES)
    assert all(rec["status"] == "done" and rec["exit_code"] == 0 for rec in state.values()), state
    assert sorted(state["train"]["units"]) == sorted(UNITS)
    assert all(u["status"] == "done" for u in state["train"]["units"].values())
    progress = (run_dir / "kfold_progress.log").read_text().splitlines()
    assert len(progress) == 1 + len(UNITS) and all(" | done | " in line for line in progress[1:])

    m = pd.read_parquet(run_dir / "evaluation" / "tables" / "metrics.parquet")
    assert _auroc(m, "model", pooled=False) > 0.9 and _auroc(m, "probe", pooled=False) > 0.9  # the planted late ST
    assert _auroc(m, "shuffled", pooled=True) < 0.65  # the shuffled-label control (§10.9.3)
    prov = json.loads((run_dir / "predictions" / "provenance.json").read_text())
    assert all(u["lock_written_at"] < u["test_written_at"] for u in prov["units"]) and not prov["missing_units"]
    assert verify.main([str(run_dir)]) == 0

    for k in (1, 2, 3):
        for job in ("train", "predict"):
            work = run_dir / "folds" / f"fold_{k}" / "worker" / job
            assert (work / "output.log").is_file() and (work / "fold.log").is_file(), work
            assert not (work / "part.pkl").exists()
        assert json.loads((run_dir / "folds" / f"fold_{k}" / "worker" / "train" / "part.json").read_text())[
            "device"] == "cpu"

    cfg = config.load(SMOKE_CONFIG, overrides)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    (run_dir / "folds" / "fold_2" / "seed_42" / "selection_lock.json").unlink()
    try:
        with pytest.raises(RunStateError, match="L5: .* is missing"):  # T-L2: no lock, no test rows
            run.neural_predict(cfg, run_dir, manifest)
        empty = tmp_path / "dead_worker"
        empty.mkdir()
        part = run._merge_train(cfg, run_dir, manifest, 2, -9, empty)  # the fold process died (SIGKILL)
        rec = json.loads((run_dir / "folds" / "fold_2" / "seed_42" / "fold_results.json").read_text())
        assert rec["status"] == "failed" and "exited with code -9" in rec["error"]
        assert [r["kind"] for r in part["trained"]] == ["model"] and len(part["records"]) == 3
        units = json.loads((run_dir / "stage_state.json").read_text())["train"]["units"]
        assert units["model|42|2"]["status"] == "failed" and units["shuffled|42|2"]["status"] == "done"
        assert "model|42|2" in run.neural_predict(cfg, run_dir, manifest)[4]
    finally:
        _restore_loguru()
