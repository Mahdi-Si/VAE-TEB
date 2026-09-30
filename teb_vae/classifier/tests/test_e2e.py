"""Smoke end to end (SPEC §15 T-X1 for both scopes, T-X3): every stage on the three-fold fixture tree, the neural
units beside the P2 baselines, then resume semantics."""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import pytest

from teb_vae.classifier.tests.conftest import SMOKE_CONFIG

SCOPES = ("sequence", "segment")
#: smoke.yaml trains the noind ablation: the fixture's has_tlo missingness is flagged in folds 2-3 (§7.3.4, L10).
UNITS = [f"{kind}|42|{k}" for k in (1, 2, 3) for kind in ("model", "shuffled", "noind")]


def _restore_loguru() -> None:
    from loguru import logger  # run.main replaced loguru's sinks with files under tmp

    logger.remove()
    logger.add(sys.stderr)


@pytest.fixture(scope="module")
def smoke_runs(smoke_overrides, tmp_path_factory):
    """``get(scope) -> (run_dir, overrides, seconds)``: ``--stage all`` on 3 folds, once per scope (lazy)."""
    from teb_vae.classifier import run

    done = {}

    def get(scope):
        if scope not in done:
            overrides = list(smoke_overrides) + ["classifier.run.folds=[1,2,3]", f"classifier.model.scope={scope}"]
            started = time.perf_counter()
            try:
                run_dir = run.main(config=str(SMOKE_CONFIG), stage="all", overrides=overrides,
                                   run_dir=str(tmp_path_factory.mktemp(f"e2e_{scope}") / "run"))
            finally:
                _restore_loguru()
            done[scope] = run_dir, overrides, time.perf_counter() - started
        return done[scope]

    return get


def _auroc(m: pd.DataFrame, model_id: str, *, pooled: bool) -> float:
    """Test GUID AUROC: the pooled OOF value, or the per-fold mean (the primary AUROC, SPEC §11.7)."""
    v = m[(m["model_id"] == model_id) & (m["split"] == "test") & (m["level"] == "guid")
          & (m["metric"] == "auroc") & m["subgroup"].isna() & m["policy_id"].isna() & (m["point"] == "n/a")]
    v = v[v["fold"] == "pooled"] if pooled else v[v["fold"] != "pooled"]
    assert len(v) == (1 if pooled else 3), model_id
    return float(v["value"].mean())


@pytest.mark.slow
@pytest.mark.parametrize("scope", SCOPES)
def test_t_x1_smoke_end_to_end(smoke_runs, scope):
    from teb_vae.classifier import report, run, verify

    run_dir, overrides, seconds = smoke_runs(scope)
    print(f"T-X1 {scope}: --stage all in {seconds:.0f} s")
    state = json.loads((run_dir / "stage_state.json").read_text())
    assert list(state) == list(run.STAGES)
    assert all(rec["status"] == "done" and rec["exit_code"] == 0 for rec in state.values()), state
    assert sorted(state["train"]["units"]) == sorted(UNITS)
    assert all(u["status"] == "done" for u in state["train"]["units"].values())

    try:
        run.main(config=str(SMOKE_CONFIG), stage="all", overrides=overrides, run_dir=str(run_dir))  # resume: no-op
        assert json.loads((run_dir / "stage_state.json").read_text()) == state
        (run_dir / "evaluation" / "verify.json").unlink()
        argv = ["--config", str(SMOKE_CONFIG), "--run-dir", str(run_dir), "--stage", "verify"]
        assert run._cli(argv + [x for o in overrides for x in ("--set", o)]) == 0  # named: re-runs
    finally:
        _restore_loguru()
    assert (run_dir / "evaluation" / "verify.json").is_file()
    assert json.loads((run_dir / "stage_state.json").read_text())["train"] == state["train"]

    unit_files = ("model_checkpoints/best.ckpt", "model_checkpoints/resolved_config.yaml", "setup.json", "scaler.json",
                  "calibration.json", "thresholds.json", "selection_lock.json", "fold_results.json",
                  "train_results/metrics_history.csv", "train_results/epoch_summary.jsonl")
    for rel in ("cohort/segments.parquet", "cohort/guids.parquet", "predictions/segments.parquet",
                "predictions/guids.parquet", "predictions/thresholds.json", "predictions/provenance.json",
                "evaluation/tables/metrics.parquet", "evaluation/tables/thresholds.parquet",
                "evaluation/tables/roc_points.parquet", "evaluation/summary.json", "summary.md",
                "kfold_progress.log", "kfold_summary.json",
                *(f"baselines/fold_{k}/{f}" for k in (1, 2, 3) for f in
                  ("probe.skops", "shortcut.skops", "probe_fit.json", "shortcut_fit.json", "selection_lock.json")),
                *(f"folds/fold_{k}/seed_42/{sub}{f}" for k in (1, 2, 3) for sub in ("", "shuffled/", "noind/")
                  for f in unit_files)):
        assert (run_dir / rel).is_file(), rel
    progress = (run_dir / "kfold_progress.log").read_text().splitlines()
    assert len(progress) == 1 + len(UNITS) and all(" | done | " in line for line in progress[1:])
    summary = json.loads((run_dir / "kfold_summary.json").read_text())
    assert not summary["missing_units"] and set(summary["across_folds"]) == {"model|42", "shuffled|42", "noind|42"}

    m = pd.read_parquet(run_dir / "evaluation" / "tables" / "metrics.parquet")
    auroc = {k: _auroc(m, k, pooled=False) for k in ("model", "probe")}
    pooled = {k: _auroc(m, k, pooled=True) for k in ("model", "shuffled", "probe", "shortcut")}
    print(f"T-X1 {scope}: fold-mean test GUID AUROC {auroc}; pooled {pooled}")
    # the planted late-ST signal, on the primary AUROC. Pooled calibrated AUROC is secondary: scale-only
    # temperature calibration leaves each fold's intercept, so pooling can mis-rank folds (segment scope: 1.0 per
    # fold, 0.889 pooled on the fixture).
    assert auroc["model"] > 0.9 and auroc["probe"] > 0.9
    assert pooled["shuffled"] < 0.65  # the shuffled-label control (§10.9.3), as in the sanity check

    prov = json.loads((run_dir / "predictions" / "provenance.json").read_text())
    assert sorted((u["model_id"], u["seed"], u["fold"]) for u in prov["units"]) == sorted(
        [(m_, "na", k) for m_ in ("probe", "shortcut") for k in (1, 2, 3)]
        + [(m_, "42", k) for m_ in ("model", "shuffled", "noind") for k in (1, 2, 3)])
    assert all(u["lock_written_at"] < u["test_written_at"] for u in prov["units"]) and not prov["missing_units"]
    thr = json.loads((run_dir / "predictions" / "thresholds.json").read_text())
    assert thr["shortcut|na|1"]["guid"]["inst30_1h"] == {"skipped": "guid-level model"}
    assert thr["shortcut|na|1"]["guid"]["np30"]["basis"] == "guid_final"
    assert thr["model|42|2"]["guid"]["inst30_1h"]["basis"] == "instantaneous" and "shuffled|42|3" in thr
    guids = pd.read_parquet(run_dir / "predictions" / "guids.parquet")
    model = guids[guids["model_id"] == "model"]
    assert model[["score_final", "score_final_cal", "p_c0"]].notna().all().all()
    if scope == "segment":  # max, the primary aggregator, is score_final
        assert model[["score_mean_cal", "score_lse_cal", "score_last_cal", "score_topk_mean_cal"]].notna().all().all()

    cfg = json.loads((run_dir / "evaluation" / "summary.json").read_text())["results"]["config"]
    figures = run_dir / "evaluation" / "figures"
    missing = [s for s in report.FIGURE_REGISTRY(cfg) for fmt in cfg["eval"]["figure_formats"]
               if not (figures / f"{s}.{fmt}").is_file()]
    assert not missing
    assert verify.main([str(run_dir)]) == 0


@pytest.mark.slow
def test_t_x3_resume_retrains_only_unlocked_units_and_refuses_a_new_digest(smoke_runs):
    """An interrupted train (one unit without its lock, the stage not done) resumes that unit alone; a changed
    config in the same run directory is refused. Runs after T-X1 (it retrains a unit of the sequence run)."""
    from teb_vae.classifier import run
    from teb_vae.classifier.config import unit_dir

    run_dir, overrides, _ = smoke_runs("sequence")
    target = unit_dir(run_dir, 2, 42, "model")
    ckpts = {p: p.stat().st_mtime_ns for p in run_dir.glob("folds/**/best.ckpt")}
    assert len(ckpts) == len(UNITS)
    (target / "selection_lock.json").unlink()
    state = json.loads((run_dir / "stage_state.json").read_text())
    state["train"]["status"] = "failed"
    (run_dir / "stage_state.json").write_text(json.dumps(state))
    try:
        run.main(config=str(SMOKE_CONFIG), stage="train", overrides=overrides, run_dir=str(run_dir))
        with pytest.raises(ValueError, match="different config"):
            run.main(config=str(SMOKE_CONFIG), stage="train", overrides=overrides + ["classifier.train.max_epochs=7"],
                     run_dir=str(run_dir))
    finally:
        _restore_loguru()
    assert [p for p, t in ckpts.items() if p.stat().st_mtime_ns != t] == [target / "model_checkpoints" / "best.ckpt"]
    assert (target / "selection_lock.json").is_file()
    progress = (run_dir / "kfold_progress.log").read_text().splitlines()
    assert len(progress) == 2 + len(UNITS) and progress[-1].split(" | ")[1:5] == ["2", "42", "model", "done"]
    train = json.loads((run_dir / "stage_state.json").read_text())["train"]
    assert train["status"] == "done" and train["exit_code"] == 0 and sorted(train["units"]) == sorted(UNITS)


@pytest.mark.slow
def test_a_failed_unit_is_recorded_and_evaluate_needs_allow_partial(smoke_overrides, tmp_path, monkeypatch):
    """§10.10.1: a unit failure (here in the val selection step) is recorded, the other units still run, train exits
    1; predict lists the unit as missing; evaluate refuses the run unless ``--allow-partial``. Retrained with
    ``--stage train`` and re-predicted, verify fails the older evaluation (1b); a partial ``evaluate --only C`` leaves
    the stage pending, so the next ``--stage all`` re-runs evaluate and every later stage (§14.1), no unit missing."""
    from teb_vae.classifier import run, verify

    real = run.lock_unit

    def flaky(cfg, out, unit, labels, kind, *args):
        if kind == "model":
            raise RuntimeError("planted selection failure")
        return real(cfg, out, unit, labels, kind, *args)

    monkeypatch.setattr(run, "lock_unit", flaky)
    overrides = list(smoke_overrides) + ["classifier.run.folds=[1]", "classifier.train.max_epochs=1"]
    kw = dict(config=str(SMOKE_CONFIG), overrides=overrides, run_dir=str(tmp_path / "run"))
    try:
        with pytest.raises(RuntimeError, match="allow-partial"):
            run.main(stage="all", **kw)
        run.main(stage="evaluate", allow_partial=True, **kw)
    finally:
        _restore_loguru()
    run_dir = tmp_path / "run"
    state = json.loads((run_dir / "stage_state.json").read_text())
    assert state["train"]["exit_code"] == 1 and state["predict"]["status"] == "done"
    assert {k: u["status"] for k, u in state["train"]["units"].items()} == {"model|42|1": "failed",
                                                                         "shuffled|42|1": "done"}
    assert state["evaluate"]["status"] == "done"
    fold_results = json.loads((run_dir / "folds/fold_1/seed_42/fold_results.json").read_text())
    assert fold_results["status"] == "failed" and "planted selection failure" in fold_results["error"]
    assert (run_dir / "kfold_progress.log").read_text().splitlines()[1].endswith(
        "| RuntimeError: planted selection failure")
    assert json.loads((run_dir / "kfold_summary.json").read_text())["missing_units"] == ["model|42|1"]
    assert json.loads((run_dir / "predictions" / "provenance.json").read_text())["missing_units"] == ["model|42|1"]
    models = set(pd.read_parquet(run_dir / "predictions" / "guids.parquet")["model_id"])
    assert models == {"probe", "shortcut", "shuffled"}

    monkeypatch.setattr(run, "lock_unit", real)
    try:
        run.main(stage="train", **kw)  # retrains the failed unit only
        pending = json.loads((run_dir / "stage_state.json").read_text())
        run.main(stage="predict", **kw)
        stale = verify.main([str(run_dir), "--json-out", str(tmp_path / "stale.json")])
        before = (run_dir / "stage_state.json").read_text()
        argv = ["--config", kw["config"], "--run-dir", kw["run_dir"], "--stage", "evaluate", "--only", "C",
                *[a for o in overrides for a in ("--set", o)]]
        assert run._cli(argv) == 0  # a partial look: stage_state is untouched, so evaluate stays pending
        assert (run_dir / "stage_state.json").read_text() == before and any((run_dir / "evaluation").glob("partial_*"))
        run.main(stage="all", **kw)
    finally:
        _restore_loguru()
    assert pending["train"]["exit_code"] == 0
    assert [pending[s]["status"] for s in ("predict", "evaluate")] == ["pending"] * 2  # were done: stale now
    got = json.loads((tmp_path / "stale.json").read_text())["criteria"]  # the evaluation predates the predictions
    assert stale == 1 and got["1b_evaluation_current"]["verdict"] == verify.FAIL
    assert got["2_units"]["verdict"] == verify.INCONCLUSIVE and got["2_units"]["missing_units"] == ["model|42|1"]
    assert json.loads(before)["evaluate"]["status"] == "pending"
    state = json.loads((run_dir / "stage_state.json").read_text())
    assert [state[s]["status"] for s in ("predict", "evaluate", "report", "verify")] == ["done"] * 4
    assert json.loads((run_dir / "predictions" / "provenance.json").read_text())["missing_units"] == []
    assert set(pd.read_parquet(run_dir / "predictions" / "guids.parquet")["model_id"]) == models | {"model"}
    assert "model" in set(pd.read_parquet(run_dir / "evaluation" / "tables" / "metrics.parquet")["model_id"])
    final = json.loads((run_dir / "evaluation" / "verify.json").read_text())["criteria"]
    assert final["1b_evaluation_current"]["verdict"] == final["2_units"]["verdict"] == verify.PASS
    # 1 fold x 1 epoch leaves the shuffled control's AUROC to chance (0.71 here); T-X1 gates criterion 7 on 3 folds
    assert [k for k, v in final.items() if v["verdict"] == verify.FAIL] in ([], ["7_shuffled_control"])


@pytest.mark.slow
def test_covariates_end_to_end_with_covariates_off_and_the_noind_ablation(covariate_overrides, tmp_path):
    """P5 (§7.3): smoke on fold 1 with the fixture's covariates (film fusion), ``eval.covariates_off`` and the
    missingness ablation forced (cap 0). Every stage through report succeeds; the ``noind`` unit trains without the
    missing flags; ``model_covoff`` / ``noind_covoff`` reuse their unit's thresholds and are evaluated like any model."""
    from teb_vae.classifier import run
    from teb_vae.classifier.config import unit_dir
    from teb_vae.classifier.train import load_net

    overrides = list(covariate_overrides) + [
        "classifier.run.folds=[1]", "classifier.train.max_epochs=2", "classifier.baselines.shuffled_control=false",
        "classifier.context.fusion=film", "classifier.eval.covariates_off=true",
        "classifier.context.auto_ablate_missing=true", "classifier.context.missing_confound_max=0.0"]
    try:
        run_dir = run.main(config=str(SMOKE_CONFIG), stage="all", overrides=overrides, run_dir=str(tmp_path / "run"))
    finally:
        _restore_loguru()
    state = json.loads((run_dir / "stage_state.json").read_text())
    assert all(state[s]["exit_code"] == 0 for s in ("cohort", "extract", "train", "predict", "evaluate", "report"))
    assert sorted(state["train"]["units"]) == ["model|42|1", "noind|42|1"]
    confound = json.loads((run_dir / "cohort" / "confound.json").read_text())["1"]
    assert set(confound) == {"has_tlo", "parity", "temp_c"}
    assert len(pd.read_parquet(run_dir / "cohort" / "covariate_availability.parquet")) > 0

    net, _ = load_net(unit_dir(run_dir, 1, 42, "noind") / "model_checkpoints" / "best.ckpt")
    plain, _ = load_net(unit_dir(run_dir, 1, 42, "model") / "model_checkpoints" / "best.ckpt")
    assert (plain.n_ctx, plain.n_cov, plain.fusion) == (7, 7, "film") and (net.n_ctx, net.n_cov) == (6, 5)

    guids = pd.read_parquet(run_dir / "predictions" / "guids.parquet")
    ids = {"model", "model_covoff", "noind", "noind_covoff"}
    assert ids <= set(guids["model_id"])
    a, b = (guids[guids["model_id"] == m].set_index(["split", "guid"])["score_final"] for m in ("model", "model_covoff"))
    assert a.index.equals(b.index) and not np.allclose(a, b)  # the same GUIDs, scored without covariates
    thr = json.loads((run_dir / "predictions" / "thresholds.json").read_text())
    assert thr["model_covoff|42|1"] == thr["model|42|1"] and "noind_covoff|42|1" in thr
    m = pd.read_parquet(run_dir / "evaluation" / "tables" / "metrics.parquet")
    assert ids <= set(m["model_id"])
    for model_id in ids:
        assert np.isfinite(_auroc(m, model_id, pooled=True)), model_id


@pytest.mark.slow
def test_three_class_multiclass_end_to_end(smoke_overrides, tmp_path):
    """P6 (§6.3, §10.8, §11.1, §11.3 T5, §11.4): a multiclass ``three_class`` smoke on the 3 folds, every stage through
    report (the verify verdict is not asserted: the planted signal is binary, not HIE vs acidosis). Val calibration,
    the 3-class prediction columns, per-class OvR thresholds, the 3-class metric rows and monitor columns."""
    import numpy as np

    from teb_vae.classifier import run
    from teb_vae.classifier.baselines import THREE_CLASS_COLUMNS
    from teb_vae.classifier.train import tracked_metrics

    overrides = list(smoke_overrides) + [
        "classifier.run.folds=[1,2,3]", "classifier.labels.task=three_class", "classifier.labels.head=multiclass",
        "classifier.labels.aux_3class_weight=0", "classifier.train.loss.name=weighted_ce",
        "classifier.train.loss.weighting=sqrt_inverse"]
    try:
        run_dir = run.main(config=str(SMOKE_CONFIG), stage="all", overrides=overrides, run_dir=str(tmp_path / "run"))
    finally:
        _restore_loguru()
    state = json.loads((run_dir / "stage_state.json").read_text())
    assert all(state[s]["exit_code"] == 0 for s in ("cohort", "extract", "train", "predict", "evaluate", "report"))
    cal = json.loads((run_dir / "folds/fold_2/seed_42/calibration.json").read_text())
    assert cal["head"] == "multiclass" and 0.01 <= cal["temperature"] <= 100

    guids, segs = (pd.read_parquet(run_dir / "predictions" / f"{t}.parquet") for t in ("guids", "segments"))
    assert set(THREE_CLASS_COLUMNS) <= set(guids) and set(THREE_CLASS_COLUMNS) <= set(segs)
    assert set(guids["y"]) == {0, 1} and set(guids["class_code"]) == {1, 2, 3}  # y: the collapsed adverse target
    for m in ("model", "shuffled", "probe"):
        g = guids[guids["model_id"] == m]
        P = g[[f"p_c{k}_cal" for k in range(3)]].to_numpy()
        assert np.isfinite(P).all() and np.allclose(P.sum(1), 1) and g["ord_score"].isna().all()
        np.testing.assert_allclose(1 / (1 + np.exp(-g["score_final_cal"])), 1 - P[:, 0], atol=1e-6)
    thr = json.loads((run_dir / "predictions" / "thresholds.json").read_text())
    assert all(set(v["ovr"]) == {"0", "1", "2"} for v in thr.values())
    assert thr["model|42|1"]["ovr"]["2"]["guid"]["emp30"]["n_pos"] > 0
    t = pd.read_parquet(run_dir / "evaluation" / "tables" / "thresholds.parquet")
    assert set(t["class"].dropna()) == {0, 1, 2} and t["class"].isna().any()

    m = pd.read_parquet(run_dir / "evaluation" / "tables" / "metrics.parquet")
    pooled = m[(m["model_id"] == "model") & (m["split"] == "test") & (m["fold"] == "pooled") & (m["level"] == "guid")
               & m["subgroup"].isna() & m["policy_id"].isin([None, "argmax"])].set_index("metric")["value"]
    for name in ("auroc_macro", "auroc_hand_till", "auroc_ovr_c2", "auprc_ovr_c1", "rps", "qwk", "macro_f1",
                 "bal_acc", "recall_c0", "adverse_vs_healthy/auroc", "hie_vs_rest/auroc"):
        assert np.isfinite(pooled[name]), name
    n_test = guids[(guids["model_id"] == "model") & (guids["split"] == "test")]["guid"].nunique()
    assert sum(pooled[f"confusion_t{i}_p{j}"] for i in range(3) for j in range(3)) == n_test

    history = pd.read_csv(run_dir / "folds/fold_1/seed_42/train_results/metrics_history.csv")
    cfg = run.load(SMOKE_CONFIG, overrides)
    three = [c for c in tracked_metrics(cfg.classifier, cfg.advanced_config) if "class" in c or "macro" in c]
    assert len(three) == 14 and history[three].notna().any().all()
    assert "| model | 42 |" in (run_dir / "summary.md").read_text().split("## 7. 3-class")[1].split("## 8.")[0]


@pytest.mark.slow
def test_no_segment_head_skips_segment_basis_policies(smoke_overrides, tmp_path):
    """``model.segment_head: false`` with a ``{basis: segment}`` policy: the causal sequence model has no segment score,
    so that policy is recorded ``skipped`` (level ``segment``) and every unit trains, locks and predicts (it used to
    raise in the val selection after training)."""
    from teb_vae.classifier import run
    from teb_vae.classifier.config import unit_dir

    overrides = list(smoke_overrides) + [
        "classifier.run.folds=[1]", "classifier.train.max_epochs=1", "classifier.model.segment_head=false",
        "classifier.context.auto_ablate_missing=false",
        "classifier.eval.thresholds=[{id: np30, policy: fpr_cap, alpha: 0.3, method: np_umbrella, delta: 0.05, "
        "allow_fallback: true, basis: committed_overall, at: end}, {id: seg30, policy: fpr_cap, alpha: 0.3, "
        "method: empirical, basis: segment}]"]
    try:
        for stage in ("cohort", "extract", "train", "predict"):
            run_dir = run.main(config=str(SMOKE_CONFIG), stage=stage, overrides=overrides, run_dir=str(tmp_path / "run"))
    finally:
        _restore_loguru()
    state = json.loads((run_dir / "stage_state.json").read_text())
    assert state["train"]["exit_code"] == 0 and state["predict"]["status"] == "done"
    thr = json.loads((unit_dir(run_dir, 1, 42, "model") / "thresholds.json").read_text())
    assert thr["segment"]["seg30"] == {"skipped": "no segment score"} and "threshold" in thr["guid"]["np30"]
    probe = json.loads((run_dir / "predictions" / "thresholds.json").read_text())
    assert "threshold" in next(v for k, v in probe.items() if k.startswith("probe"))["segment"]["seg30"]
