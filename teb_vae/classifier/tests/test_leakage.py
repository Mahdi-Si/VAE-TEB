"""Leakage guards at model level (SPEC §12 L1, L5; §15 T-L1, T-L2): one trained sequence-scope unit on fold 1."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from teb_vae.classifier.tests.conftest import SMOKE_CONFIG

SCORES = ["logit_seg", "logit_online", "p_c0", "p_c1", "p_c2"]


def _restore_loguru() -> None:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr)


@pytest.fixture(scope="module")
def trained(covariate_overrides, tmp_path_factory):
    """cohort -> extract -> train on fold 1 (one ``model`` unit, 2 epochs, ``elapsed`` on so every clock-derived
    context feature is exercised, and the fixture's covariates, whose timed table carries a clock of its own):
    ``(cfg, overrides, run_dir, manifest)``."""
    from teb_vae.classifier import run
    from teb_vae.classifier.config import load

    tmp = tmp_path_factory.mktemp("leakage")
    overrides = list(covariate_overrides) + [
        "classifier.run.folds=[1]", f"classifier.source.cache_root={tmp / 'cache'}", "classifier.train.max_epochs=2",
        "classifier.baselines.shuffled_control=false", "classifier.context.elapsed.enabled=true",
        "classifier.context.auto_ablate_missing=false"]  # the covariates' confound would add a noind unit
    try:
        for stage in ("cohort", "extract", "train"):
            run.main(config=str(SMOKE_CONFIG), stage=stage, overrides=overrides, run_dir=str(tmp / "run"))
    finally:
        _restore_loguru()
    manifest = json.loads((tmp / "run" / "manifest.json").read_text())
    return load(SMOKE_CONFIG, overrides), overrides, tmp / "run", manifest


def _unit_dir(trained):
    from teb_vae.classifier.config import unit_dir

    return unit_dir(trained[2], 1, 42, "model")


@pytest.mark.slow
def test_t_l1_model_predictions_ignore_forbidden_inputs(trained, tmp_path):
    """T-L1 at model level: the trained unit's val/test scores are bit-identical on a cohort and cache whose
    forbidden columns are permuted or randomised, covariates re-joined with ``time_s`` moved with each GUID's clocks
    (the batch-level test's perturbation)."""
    from teb_vae.classifier.data import build_unit
    from teb_vae.classifier.tests.test_data import perturb_forbidden
    from teb_vae.classifier.train import score_split

    cfg, _, run_dir, manifest = trained
    perturbed_run, perturbed_cache = perturb_forbidden(run_dir, manifest["source"], tmp_path,
                                                       cfg.classifier.context.covariates)
    a, b = build_unit(cfg, run_dir, manifest["source"], 1), build_unit(cfg, perturbed_run, perturbed_cache, 1)
    assert not a.frames["test"]["t_end_s"].equals(b.frames["test"]["t_end_s"]) and a.n_cov == 7  # really perturbed
    for split in ("val", "test"):
        (sa, ga), (sb, gb) = score_split(_unit_dir(trained), a, split), score_split(_unit_dir(trained), b, split)
        np.testing.assert_array_equal(sa[SCORES].to_numpy(), sb[SCORES].to_numpy())
        np.testing.assert_array_equal(ga[["score_final", "p_c0", "p_c1", "p_c2"]].to_numpy(),
                                      gb[["score_final", "p_c0", "p_c1", "p_c2"]].to_numpy())


@pytest.mark.slow
def test_t_l2_test_rows_need_the_units_lock(trained):
    """T-L2: without the neural unit's ``selection_lock.json`` the predict stage refuses (after the baselines,
    which are locked), writes nothing, and records the stage as failed; restored, it predicts."""
    from teb_vae.classifier import run
    from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import RunStateError

    cfg, overrides, run_dir, manifest = trained
    lock = _unit_dir(trained) / "selection_lock.json"
    saved = lock.read_bytes()
    lock.unlink()
    try:
        with pytest.raises(RunStateError, match="L5: .*selection_lock.json is missing"):
            run.main(config=str(SMOKE_CONFIG), stage="predict", overrides=overrides, run_dir=str(run_dir))
    finally:
        lock.write_bytes(saved)
        _restore_loguru()
    assert not (run_dir / "predictions" / "segments.parquet").exists()
    assert json.loads((run_dir / "stage_state.json").read_text())["predict"]["status"] == "failed"
    _, _, thr, units, missing = run.neural_predict(cfg, run_dir, manifest)
    assert [(u["model_id"], u["seed"], u["fold"]) for u in units] == [("model", "42", 1)] and not missing
    assert set(thr) == {"model|42|1"}


@pytest.mark.slow
@pytest.mark.parametrize("name", ["model_checkpoints/best.ckpt", "scaler.json", "calibration.json",
                                  "thresholds.json"])
def test_lock_refuses_a_file_changed_after_locking(trained, name):
    from teb_vae.classifier import run
    from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import RunStateError

    cfg, _, run_dir, manifest = trained
    path = _unit_dir(trained) / name
    saved = path.read_bytes()
    path.write_bytes(saved + b"\n")
    try:
        with pytest.raises(RunStateError, match=f"L5: .*changed or missing since: \\['{name}'\\]"):
            run.neural_predict(cfg, run_dir, manifest)
    finally:
        path.write_bytes(saved)


@pytest.mark.slow
def test_lock_refuses_another_config(trained):
    from teb_vae.classifier import run
    from teb_vae.classifier.config import load
    from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import RunStateError

    _, overrides, run_dir, manifest = trained
    other = load(SMOKE_CONFIG, overrides + ["classifier.calibration.method=platt"])
    with pytest.raises(RunStateError, match="L5: .*was locked under config"):
        run.neural_predict(other, run_dir, manifest)
