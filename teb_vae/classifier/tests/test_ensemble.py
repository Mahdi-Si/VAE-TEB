"""P8 (SPEC §10.7, §11.8, §16): the multi-seed ensemble (``seed='ens'``: mean raw logits, one calibration and one
threshold selection on val, its own lock), its resume semantics (T-X3 with ensembles), and ``compare`` on two smoke
runs (DeLong and Nadeau-Bengio p-values)."""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import pytest

from teb_vae.classifier.tests.conftest import SMOKE_CONFIG

SEEDS = ("42", "43")
FOLDS = (1, 2, 3)


def _restore_loguru() -> None:
    from loguru import logger  # run.main replaced loguru's sinks with files under tmp

    logger.remove()
    logger.add(sys.stderr)


def _frames(a, s, P):
    seg = pd.DataFrame({"split": "val", "guid": ["g", "g", "h"], "seg_pos": [0, 1, 0], "logit_seg": a,
                        "logit_online": a, "attn_entropy": a, "seq_attn_final": np.nan, "y": [1, 1, 0]})
    gd = pd.DataFrame({"split": "val", "guid": ["g", "h"], "score_final": s, "score_max": s, "ord_score": s,
                       **{f"p_c{k}": P[:, k] for k in range(3)}, "y": [1, 0]})
    return seg, gd


def test_ensemble_scores_average_raw_logits_and_class_logits():
    from teb_vae.classifier import run

    P1, P2 = np.array([[.7, .2, .1], [.2, .3, .5]]), np.array([[.5, .25, .25], [.1, .1, .8]])
    seg, gd = run.ensemble_scores([_frames([1.0, 2, 3], [1.0, -1], P1), _frames([3.0, 4, 5], [3.0, 1], P2)])
    assert np.allclose(seg[["logit_seg", "logit_online", "attn_entropy"]], [[2] * 3, [3] * 3, [4] * 3])
    assert seg["seq_attn_final"].isna().all() and seg["y"].tolist() == [1, 1, 0]  # labels untouched
    assert np.allclose(gd[["score_final", "score_max", "ord_score"]], [[2] * 3, [0] * 3])
    L = (np.log(P1) + np.log(P2)) / 2  # softmax of the mean log-probability = of the mean class logits
    assert np.allclose(gd[["p_c0", "p_c1", "p_c2"]], np.exp(L) / np.exp(L).sum(1, keepdims=True))
    a, b = _frames([1.0, 2, 3], [1.0, -1], P1), _frames([1.0, 2, 3], [1.0, -1], P1)
    with pytest.raises(ValueError, match="different rows"):
        run.ensemble_scores([a, (b[0].iloc[::-1], b[1])])


def test_ensembles_group_the_kinds_with_several_seeds(smoke_config):
    from teb_vae.classifier import run

    one = smoke_config.model_copy(update={"run": smoke_config.run.model_copy(update={"seeds": [42]})})
    two = smoke_config.model_copy(update={"run": smoke_config.run.model_copy(update={"seeds": [42, 43]})})
    assert run.ensembles(one) == {}
    assert run.ensembles(two) == {"model": [42, 43]}  # shuffled / noind: first seed only
    online = two.model_copy(update={"train": two.train.model_copy(update={"regime": "partial"})})
    assert run.ensembles(online) == {"model": [42, 43], "frozen": [42, 43]}
    assert run.ens_dir("r", 2, "model").as_posix() == "r/folds/fold_2/ens"
    assert run.ens_dir("r", 2, "frozen").as_posix() == "r/folds/fold_2/ens/frozen"


@pytest.fixture(scope="module")
def smoke_pair(smoke_overrides, tmp_path_factory):
    """``get(name) -> (run_dir, overrides)``: ``--stage all`` on the three fixture folds, lazily: ``ens`` (sequence
    scope, seeds 42 and 43) and ``segment`` (segment scope, one seed), the same cohort. Three folds as in T-X1: on
    folds 1-2 alone the shuffled control's pooled AUROC is 0.66 (verify criterion 7), a fixture artifact."""
    from teb_vae.classifier import run

    arms = {"ens": ["classifier.run.seeds=[42,43]"], "segment": ["classifier.model.scope=segment"]}
    done = {}

    def get(name):
        if name not in done:
            overrides = list(smoke_overrides) + ["classifier.run.folds=[1,2,3]"] + arms[name]
            started = time.perf_counter()
            try:
                run_dir = run.main(config=str(SMOKE_CONFIG), stage="all", overrides=overrides,
                                   run_dir=str(tmp_path_factory.mktemp(f"p8_{name}") / "run"))
            finally:
                _restore_loguru()
            print(f"P8 smoke {name}: --stage all in {time.perf_counter() - started:.0f} s")
            done[name] = run_dir, overrides
        return done[name]

    return get


@pytest.mark.slow
def test_multi_seed_smoke_writes_a_locked_calibrated_ensemble(smoke_pair):
    from teb_vae.classifier import metrics, report, verify
    from teb_vae.classifier.train import apply_calibration, fit_calibration

    run_dir, _ = smoke_pair("ens")
    state = json.loads((run_dir / "stage_state.json").read_text())
    assert all(rec["status"] == "done" and rec["exit_code"] == 0 for rec in state.values()), state
    assert {f"model|ens|{k}" for k in FOLDS} <= set(state["train"]["units"])
    guids = pd.read_parquet(run_dir / "predictions" / "guids.parquet")
    model = guids[guids["model_id"] == "model"]
    assert set(model["seed"]) == {*SEEDS, "ens"}
    assert {(s, sp) for s, sp in zip(model["seed"], model["split"]) if s == "ens"} == {("ens", "val"), ("ens", "test")}
    assert (guids.loc[guids["model_id"] == "shuffled", "seed"] == "42").all()  # controls: first seed, no ensemble

    raw = model.pivot_table(index=["fold", "split", "guid"], columns="seed", values="score_final")
    assert np.allclose(raw["ens"], raw[list(SEEDS)].mean(1))  # the mean of the members' raw logits
    prov = json.loads((run_dir / "predictions" / "provenance.json").read_text())
    for k in FOLDS:
        out = run_dir / "folds" / f"fold_{k}" / "ens"
        cal = json.loads((out / "calibration.json").read_text())
        val = model[(model["fold"] == k) & (model["split"] == "val")].pivot_table(
            index="guid", columns="seed", values=["score_final", "y"])
        refit = fit_calibration(val["score_final"][list(SEEDS)].mean(1), val["y"]["ens"], "temperature")
        assert cal["temperature"] == pytest.approx(refit["temperature"], rel=1e-6)  # fitted once, on the ens val rows
        ens = model[(model["fold"] == k) & (model["seed"] == "ens")]
        assert np.allclose(ens["score_final_cal"], apply_calibration(ens["score_final"], cal))
        lock = json.loads((out / "selection_lock.json").read_text())
        assert set(lock["sha256"]) == {"calibration.json", "thresholds.json",
                                       *(f"../seed_{s}/selection_lock.json" for s in SEEDS)}
        unit = next(u for u in prov["units"] if (u["model_id"], u["seed"], u["fold"]) == ("model", "ens", k))
        assert unit["lock_written_at"] == lock["locked_utc"] < unit["test_written_at"]  # L5
    thr = json.loads((run_dir / "predictions" / "thresholds.json").read_text())
    assert thr["model|ens|1"] == json.loads((run_dir / "folds" / "fold_1" / "ens" / "thresholds.json").read_text())

    summary = json.loads((run_dir / "evaluation" / "summary.json").read_text())
    assert any(h["seed"] == "ens" for h in summary["results"]["headline"].values())
    assert metrics.primary_model([("model", "42"), ("model", "43"), ("model", "ens"), ("probe", "na")]) == \
        ("model", "ens")
    assert (run_dir / "evaluation" / "figures" / f"{report.SEED_SPREAD}.pdf").is_file()  # H2
    assert verify.main([str(run_dir)]) == 0


@pytest.mark.slow
def test_t_x3_resume_relocks_only_the_stale_ensemble(smoke_pair):
    """A member retrained on resume re-locks, so its fold's ensemble is re-selected; the other folds' are skipped."""
    from teb_vae.classifier import run
    from teb_vae.classifier.baselines import lock_problem
    from teb_vae.classifier.config import load, digest, unit_dir

    run_dir, overrides = smoke_pair("ens")
    want = digest(load(SMOKE_CONFIG, overrides))
    locks = {k: json.loads((run_dir / "folds" / f"fold_{k}" / "ens" / "selection_lock.json").read_text())
             for k in FOLDS}
    ckpts = {p: p.stat().st_mtime_ns for p in run_dir.glob("folds/**/best.ckpt")}
    target = unit_dir(run_dir, 1, 43, "model")
    (target / "selection_lock.json").unlink()
    state = json.loads((run_dir / "stage_state.json").read_text())
    state["train"]["status"] = "failed"
    (run_dir / "stage_state.json").write_text(json.dumps(state))
    try:
        run.main(config=str(SMOKE_CONFIG), stage="train", overrides=overrides, run_dir=str(run_dir))
    finally:
        _restore_loguru()
    assert [p for p, t in ckpts.items() if p.stat().st_mtime_ns != t] == [target / "model_checkpoints" / "best.ckpt"]
    after = {k: json.loads((run_dir / "folds" / f"fold_{k}" / "ens" / "selection_lock.json").read_text())
             for k in FOLDS}
    assert after[1]["locked_utc"] > locks[1]["locked_utc"] and all(after[k] == locks[k] for k in FOLDS[1:])
    assert all(lock_problem(run_dir / "folds" / f"fold_{k}" / "ens", want) is None for k in FOLDS)
    train = json.loads((run_dir / "stage_state.json").read_text())["train"]
    assert train["status"] == "done" and train["exit_code"] == 0 and train["units"]["model|ens|1"]["status"] == "done"


@pytest.mark.slow
def test_compare_two_smoke_runs_reports_delong_and_nadeau_bengio(smoke_pair, tmp_path):
    """P8 acceptance: ``compare`` of two real smoke runs (the multi-seed ensemble against a segment-scope run)."""
    from teb_vae.classifier import run

    a, b = smoke_pair("ens")[0], smoke_pair("segment")[0]
    try:
        code = run._cli(["compare", "--runs", str(a), str(b), "--out", str(tmp_path / "cmp")])
    finally:
        _restore_loguru()
    assert code == 0
    q = pd.read_parquet(tmp_path / "cmp" / "comparison.parquet")
    whole = q[(q["metric"] == "delta_auroc") & q["subgroup"].isna()]
    assert len(whole) == 1 and whole["reference_model"].iloc[0] == "model|ens"
    assert np.isfinite(whole[["value_raw", "p_delong", "p_nb"]].to_numpy(float)).all()
    assert ((0 <= whole[["p_delong", "p_nb"]]) & (whole[["p_delong", "p_nb"]] <= 1)).all(axis=None)
    assert (tmp_path / "cmp" / "comparison.md").is_file()
