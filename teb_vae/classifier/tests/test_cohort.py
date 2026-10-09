"""Cohort tables, tasks, labeling strategies, fold checks and the cohort stage.

T-C1..T-C5 of SPEC §15 on the generated fixture, plus the P0 sample-shard acceptance (§16).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from teb_vae.classifier import cohort, config
from teb_vae.classifier.tests.fixtures import make_fixture as fx
from teb_vae.lag_attn_transformer_cfs.latent_pilot.data import PilotConfigError

# ---- the fixture tree --------------------------------------------------------------------------
# ---- T-C1 --------------------------------------------------------------------------------------
def test_tc1_training_window_moves_only_the_train_split(fixture_segments, smoke_config):
    """L6: ``train_epoch_min_s`` narrows the train split's window; val and test keep ``epoch_min_s``."""
    base, data = fixture_segments[0], smoke_config.data
    floor = float(base["epoch_s"].median())
    shards = {(k, s): p for k in (1, 2, 3) for s, p in cohort.fold_shards(data, k).items()}
    seg, _ = cohort.segment_table(shards, trim_minutes=1.0, stride_s=data.stride_s, epoch_min_s=data.epoch_min_s,
                                  train_epoch_min_s=floor, min_valid_frac=data.min_valid_frac)
    train = seg["split"] == "train"
    early = train & (seg["epoch_s"] < floor) & (base["exclusion_reason"] != "duplicate_epoch")
    assert early.any() and (seg.loc[early, "exclusion_reason"] == "outside_window").all()
    unchanged = ~train | (seg["epoch_s"] >= floor)
    pd.testing.assert_series_equal(seg.loc[unchanged, "exclusion_reason"], base.loc[unchanged, "exclusion_reason"])


# ---- T-C2 --------------------------------------------------------------------------------------
@pytest.mark.parametrize("task", sorted(config.TASKS))
def test_tc2_task_targets_and_exclusions(fixture_segments, smoke_config, task):
    labels = smoke_config.labels.model_copy(update={"task": task})
    seg, guids = cohort.guid_table(fixture_segments[0], labels, smoke_config.cohort)
    mapping = config.TASKS[task]
    expected = guids["cs"].astype(int) if task == "cs_outcome" else guids["class_code"].map(mapping)
    retained = ~guids["excluded"]
    assert (guids.loc[retained, "y"] == expected[retained]).all()
    dropped = guids["exclusion_reason"] == "task_exclude"
    assert set(guids.loc[dropped, "class_code"]) == ({1, 2, 3} - set(mapping) if mapping else set())
    assert (seg.loc[seg["guid"].isin(guids.loc[dropped, "guid"]), "exclusion_reason"]
            == "task_exclude").all()


# ---- T-C3 --------------------------------------------------------------------------------------
DELTA = np.array([3.0, 2.0, 1.5, 1.0, 0.5])


def _weights(strategy, y, **overrides):
    settings = {"horizon_h": 1.0, "decay_halflife_h": 0.5, **overrides}
    pos = np.arange(len(DELTA))
    return cohort.strategy_weights(DELTA, np.full(len(DELTA), y), pos == pos[-1], strategy=strategy, **settings)


def test_tc3_strategy_weights_known_answers():
    np.testing.assert_allclose(_weights("horizon", 1), [0, 0, 0, 1, 1])
    np.testing.assert_allclose(_weights("horizon_decay", 1), [2 ** -4, 2 ** -2, 0.5, 1, 1])
    np.testing.assert_allclose(_weights("horizon_decay", 2, decay_halflife_h=1.0),
                               [0.25, 0.5, 2 ** -0.5, 1, 1])
    np.testing.assert_allclose(_weights("horizon_decay", 1, horizon_h=2.0), [0.25, 1, 1, 1, 1])
    np.testing.assert_allclose(_weights("propagate", 1), 1.0)
    for strategy in ("propagate", "horizon", "horizon_decay"):
        np.testing.assert_allclose(_weights(strategy, 0), 1.0)  # negatives: y = 0, ω = 1
    for strategy in ("horizon", "horizon_decay"):  # labels.time_matched: negatives take the positives' schedule
        np.testing.assert_allclose(_weights(strategy, 0, time_matched=True), _weights(strategy, 1))
    np.testing.assert_allclose(_weights("final_only", 1), [0, 0, 0, 0, 1])  # k_warm: data.GuidDataset's w_pos
    np.testing.assert_allclose(_weights("mil", 1), 0.0)


def test_tc3_eval_windows_are_time_matched(fixture_segments, smoke_config):
    """§6.6: the same window for positive and negative GUIDs."""
    for window in ("all", "bins", "horizon", "stage:first", "stage:second"):
        labels = smoke_config.labels.model_copy(update={"eval_window": window})
        seg, _ = cohort.guid_table(fixture_segments[0], labels, smoke_config.cohort)
        kept = ~seg["excluded"]
        expected = {"all": kept, "bins": kept, "horizon": kept & (seg["hours_to_delivery"] <= labels.horizon_h),
                    "stage:first": kept & (seg["stage"] == "first"),
                    "stage:second": kept & (seg["stage"] == "second")}[window]
        assert (seg["in_eval_window"] == expected).all(), window
        assert set(seg.loc[seg["in_eval_window"], "y"]) == {0, 1}, window


# ---- T-C4 --------------------------------------------------------------------------------------
def test_tc4_split_overlap_raises(fixture_guids):
    guids = fixture_guids[1]
    leak = guids[(guids["fold"] == 1) & (guids["split"] == "train")].iloc[[0]].assign(split="val")
    with pytest.raises(PilotConfigError):
        cohort.validate_folds(pd.concat([guids, leak]), task="adverse_vs_healthy")


def test_tc4_patient_overlap_raises(fixture_guids, tmp_path):
    guids = fixture_guids[1]
    fold1 = guids[guids["fold"] == 1]
    a = fold1[fold1["split"] == "train"]["guid"].iloc[0]
    b = fold1[fold1["split"] == "test"]["guid"].iloc[0]
    patient_map = tmp_path / "patients.json"
    patient_map.write_text(json.dumps({a: "P1", b: "P1"}))
    with pytest.raises(PilotConfigError):
        cohort.validate_folds(guids, task="adverse_vs_healthy", patient_map=str(patient_map))
    # joined on guid_norm like every external table: the normalised spelling of the same map raises too
    patient_map.write_text(json.dumps({cohort.normalize_guid(a): "P1", cohort.normalize_guid(b): "P1"}))
    assert cohort.normalize_guid(a) != a
    with pytest.raises(PilotConfigError):
        cohort.validate_folds(guids, task="adverse_vs_healthy", patient_map=str(patient_map))
    patient_map.write_text(json.dumps({a: "P1", cohort.normalize_guid(a): "P2"}))  # one GUID, two patients
    with pytest.raises(PilotConfigError, match="another patient"):
        cohort.validate_folds(guids, task="adverse_vs_healthy", patient_map=str(patient_map))


def test_tc4_pretraining_exposure(fixture_guids, fixture_tree, smoke_config, tmp_path):
    guids = fixture_guids[1]
    assert cohort.pretrain_exposure(guids, smoke_config.source, allow_overlap=False) == {
        "applicable": False,
        "note": "source.kind is hdf5: no pretrained encoder, so no pretraining exposure"}
    checkpoint = tmp_path / "model_checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"")
    test_shard = str(fx.Path(fixture_tree["root"]) / "fold_1" / "test" / "acidosis_no_cs.hdf5")
    (checkpoint.parent / "resolved_config.yaml").write_text(yaml.safe_dump(
        {"dataset_config": {"vae_train_datasets": [test_shard], "vae_test_datasets": []}}))
    source = smoke_config.source.model_copy(update={
        "kind": "vae",
        "vae": smoke_config.source.vae.model_copy(update={"checkpoint": str(checkpoint)})})
    with pytest.raises(ValueError, match="L3"):
        cohort.pretrain_exposure(guids, source, allow_overlap=False)
    record = cohort.pretrain_exposure(guids, source, allow_overlap=True)
    assert record["folds"]["1"]["test_overlap"] and record["folds"]["1"]["pretraining"]["known"]
    assert not record["folds"]["1"]["selection"]["known"]  # `vae_test_datasets: []` names no population
    for dataset in ({}, {"vae_train_datasets": None, "vae_test_datasets": []}):  # unnamed lists: UNKNOWN, not empty
        (checkpoint.parent / "resolved_config.yaml").write_text(yaml.safe_dump({"dataset_config": dataset}))
        fold = cohort.pretrain_exposure(guids, source, allow_overlap=False)["folds"]["1"]
        assert not fold["pretraining"]["known"] and not fold["selection"]["known"]
        assert fold["clean_holdout_supported"] is False and fold["test_overlap"] == []


# ---- T-C5 --------------------------------------------------------------------------------------
#: Segment ends (s rel. delivery) per guid_norm, and timed observations (guid spelt as in a CSV, time_s, value).
SEG_ENDS = pd.DataFrame({"guid_norm": ["A"] * 5 + ["B", "C", "D"],
                         "t_end_s": [-10800.0, -7200.0, -3600.0, -1800.0, -60.0, -60.0, -60.0, -60.0]})
OBS = [("a", -14400, 36.5), ("a", -7200, 37.0), ("a", -3000, 38.0), ("a", -30, 45.0),  # -30: after the last end
       ("B", -7260, 36.9), ("c", -7261, 36.8)]  # ages at -60: exactly 2 h, and 2 h + 1 s


def _covariates(tmp_path, *, static=None, timed=None, variables=(("temp_c", "numeric"),), **settings):
    """A ``CovariatesCfg`` over CSVs written from ``static`` / ``timed`` rows (dicts)."""
    paths = {}
    for key, rows in (("static_csv", static), ("timed_csv", timed)):
        if rows is not None:
            paths[key] = str(tmp_path / f"{key}.csv")
            pd.DataFrame(rows).to_csv(paths[key], index=False)
    return config.CovariatesCfg(**{
        "static_csv": None, "timed_csv": None, "max_age_h": 2.0, "age_feature": True, "missing": "indicator",
        "dropout_p": 0.2, "block_dropout_p": 0.1, **paths, **settings,
        "variables": [{"name": n, "kind": k, "available_at": "prospective"} for n, k in variables]})


def _timed(obs, shift=None):
    shift = shift or {}
    return [{"guid": g, "time_s": t + shift.get(g.upper(), 0.0), "variable": "temp_c", "value": v} for g, t, v in obs]


def test_tc5_asof_join_is_causal_and_respects_max_age(tmp_path):
    got = cohort.join_covariates(SEG_ENDS, _covariates(tmp_path, timed=_timed(OBS)))
    assert list(got) == ["cov:temp_c", "cov_age_h:temp_c"]  # time_s never leaves the join
    np.testing.assert_array_equal(got["cov:temp_c"], [36.5, 37.0, 37.0, 38.0, 38.0, 36.9, np.nan, np.nan])
    np.testing.assert_allclose(got["cov_age_h:temp_c"], [1.0, 0.0, 1.0, 1200 / 3600, 2940 / 3600, 2.0, np.nan, np.nan])
    assert not (got["cov:temp_c"] == 45.0).any()  # the observation after every segment end is never used
    short = cohort.join_covariates(SEG_ENDS, _covariates(tmp_path, timed=_timed(OBS), max_age_h=0.5))
    np.testing.assert_array_equal(short["cov:temp_c"], [np.nan, 37.0, np.nan, 38.0, np.nan, np.nan, np.nan, np.nan])


# ---- confound check, context columns, fold summary ---------------------------------------------
def test_context_hides_whether_second_stage_is_reached():
    """§7.1: pre-onset segments of a GUID that later reaches second stage and of a GUID whose onset is unknown
    have the same context (an ``unknown`` flag would be ``~has_ss`` from segment 0)."""
    seg = pd.DataFrame({"tlo_end_s": 7200.0, "epoch_s": -9000.0, "t_end_s": -7740.0, "valid_frac": 0.8,
                        "ss_rel_s": [-5000.0, np.nan, -600.0, 100.0]})
    seg["stage"] = cohort.stage_of(seg["ss_rel_s"], 60.0, 1260.0)
    assert list(seg["stage"]) == ["first", "unknown", "straddle", "second"]
    ctx = cohort.context_features(seg)
    pd.testing.assert_series_equal(ctx.iloc[0], ctx.iloc[1], check_names=False)
    assert list(ctx["in_ss"]) == [0.0, 0.0, 1.0, 1.0]  # one flag: wholly or partly in second stage


def test_context_tlo_hides_time_until_onset():
    """``context.tlo.pre_onset: clip`` (default): pre-onset segments look alike whatever their time until onset
    (future information, like time until second stage, §2.4); ``signed`` (ablation) keeps ψ of the negative hours."""
    seg = pd.DataFrame({"tlo_end_s": [-3600.0, -30000.0, 3600.0], "epoch_s": -9000.0, "t_end_s": -7740.0,
                        "valid_frac": 0.8, "ss_rel_s": np.nan, "stage": "unknown"})
    np.testing.assert_allclose(cohort.context_features(seg)["tlo_psi"], [0.0, 0.0, np.log(2.0)])
    np.testing.assert_allclose(cohort.context_features(seg, tlo_pre_onset="signed")["tlo_psi"],
                               [-np.log(2.0), -np.log1p(30000.0 / 3600.0), np.log(2.0)])


# ---- the cohort stage through run.py -----------------------------------------------------------
# ---- P0 acceptance on the sample shard ---------------------------------------------------------
