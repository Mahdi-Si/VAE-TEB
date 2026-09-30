"""Cohort tables, tasks, labeling strategies, fold checks and the cohort stage.

T-C1..T-C5 of SPEC §15 on the generated fixture, plus the P0 sample-shard acceptance (§16).
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from teb_vae.classifier import cohort, config
from teb_vae.classifier.tests.fixtures import make_fixture as fx
from teb_vae.lag_attn_transformer_cfs.latent_pilot.data import PilotConfigError

SAMPLE_SHARD = config.REPO_ROOT / "tmp" / "data" / "hie_cs.hdf5"
TRF_CFS_SHARD = config.REPO_ROOT / "teb_vae" / "lag_attn" / "tests" / "fixtures" / "tiny_shard_causal.hdf5"
WIDTHS = {"fhr_st": 36, "fhr_ph": 66, "up_st": 36, "up_ph": 15}
REASONS = ["crosses_delivery", "duplicate_epoch", "low_valid_frac", "outside_window"]


# ---- the fixture tree --------------------------------------------------------------------------
def test_fixture_loads_with_its_stats(fixture_tree):
    import torch
    from hdf5_dataset.hdf5_dataset import CombinedHDF5Dataset

    dataset = CombinedHDF5Dataset(
        fixture_tree["shards"][1], stats_path=fixture_tree["stats_path"],
        normalize_fields=list(WIDTHS), trim_minutes=1.0, pin_memory=False, cache_size=0)
    sample = dataset[0]
    assert dataset.normalization_enabled and dataset.transform == "causal"
    assert {k: tuple(sample[k].shape) for k in WIDTHS} == {k: (300, c) for k, c in WIDTHS.items()}
    assert all(torch.isfinite(sample[k]).all() for k in WIDTHS)
    assert len(list(fx.Path(fixture_tree["root"]).glob("fold_*/*/*.hdf5"))) == 36


def test_fixture_matches_the_trf_cfs_causal_shard(fixture_tree):
    with h5py.File(TRF_CFS_SHARD) as ref, h5py.File(fixture_tree["shards"][1][0]) as ours:
        for key in (*WIDTHS, "fhr", "up", "weight", "target"):
            assert ref[key].shape[1:] == ours[key].shape[1:], key
        for key in WIDTHS:
            for attr in ("causal_warmup_steps", "causal_delay_s"):
                np.testing.assert_array_equal(ref[key].attrs[attr], ours[key].attrs[attr])
        for attr in ("transform", "causal_leg_alignment"):
            assert ref.attrs[attr] == ours.attrs[attr]
        assert (ref.attrs.get("causal_phase_operator", "ratio_power_v0")
                == ours.attrs.get("causal_phase_operator", "ratio_power_v0"))


def test_fixture_plants_a_late_st_shift_in_positives(fixture_tree):
    by_class = {}
    for path in fixture_tree["shards"][1]:
        with h5py.File(path) as handle:
            code = int(handle["target"][0].max() / handle["weight"][0].max())
            late = handle["epoch"][()] + 1260 >= -fx.PLANT_WINDOW_S
            level = np.log(np.maximum(handle["fhr_st"][:, 1:, 15:-15], 1e-6)).mean((1, 2))
        by_class.setdefault(code, []).append(pd.DataFrame({"late": late, "level": level}))
    shift = {code: (lambda f: f.level[f.late].mean() - f.level[~f.late].mean())(pd.concat(frames))
             for code, frames in by_class.items()}
    assert abs(shift[1]) < 0.3 and 0.5 < shift[2] < shift[3]


# ---- T-C1 --------------------------------------------------------------------------------------
def test_tc1_stride_slot_clocks(fixture_segments):
    seg, info = fixture_segments
    assert info["stride_s"] == 660.0 and info["window_s"] == [60.0, 1260.0]
    first = seg.groupby(cohort.KEYS)["epoch_s"].transform("min")
    np.testing.assert_array_equal(seg["slot"], (seg["epoch_s"] - first) / 660.0)
    np.testing.assert_allclose(seg["t_end_s"], seg["epoch_s"] + 1260.0)
    np.testing.assert_allclose(seg["hours_to_delivery"], -seg["t_end_s"] / 3600.0)
    assert seg["guid"].nunique() == 60 and seg["guid"].dtype == "str"


def test_metadata_pass_reads_columns_exactly_like_the_loader(smoke_config):
    """The metadata pass reads each column once per file (the per-sample loader takes ~3 ms a segment, ~40 min over a
    real 10-fold cohort); its frame is the one the loader's per-sample pass gives, row for row (``ds_index``)."""
    from torch.utils.data import DataLoader

    from hdf5_dataset.hdf5_dataset import CombinedHDF5Dataset, attribute_dict_collate
    from teb_vae.lag_attn_transformer_cfs.latent_pilot.data import segment_frame

    for split, paths in cohort.fold_shards(smoke_config.data, 1).items():
        fast = cohort._read_split(paths, fold=1, split=split, trim_minutes=1.0)[0]
        ds = CombinedHDF5Dataset(list(paths), load_fields=cohort.META_FIELDS, cache_size=0, pin_memory=False,
                                 trim_minutes=1.0)
        ref = segment_frame(DataLoader(ds, batch_size=64, collate_fn=attribute_dict_collate), split=split)
        pd.testing.assert_frame_equal(fast.drop(columns=["valid_frac", "fold"]), ref)
        assert np.allclose(fast["valid_frac"], [float(ds[i]["weight"].double().mean()) for i in range(len(ds))])


def test_tc1_stride_inference_refuses_a_mixed_grid():
    def frame(epochs):
        return pd.DataFrame({"fold": 0, "split": "x", "guid": [g for g, e in epochs for _ in e],
                             "epoch_s": [v for _, e in epochs for v in e]})
    assert cohort.infer_stride(frame([("a", [0, 660, 1980]), ("b", [0, 660])])) == 660.0
    with pytest.raises(ValueError, match="not unimodal"):
        cohort.infer_stride(frame([("a", [0, 660, 1320, 1980]), ("b", [0, 1000])]))
    with pytest.raises(ValueError, match="not unimodal"):
        cohort.infer_stride(frame([("a", [0, 660, 1320]), ("b", [0, 1200, 2400])]))


def test_tc1_stage_at_trim_aware_boundaries(fixture_segments):
    assert list(cohort.stage_of([-1260.0, -1259.5, -60.5, -60.0, np.nan], 60.0, 1260.0)) == [
        "first", "straddle", "straddle", "second", "unknown"]
    seg, info = fixture_segments
    assert list(seg["stage"]) == list(cohort.stage_of(seg["ss_rel_s"], 60.0, 1260.0))
    assert set(seg["stage"]) == set(cohort.STAGES)


def test_tc1_sentinel_is_unknown_and_counted(fixture_segments):
    seg, info = fixture_segments
    sentinel = seg["guid"] == fx.SENTINEL_GUID
    assert info["sentinel_guids"] == [fx.SENTINEL_GUID]
    assert info["n_sentinel_segments"] == int(sentinel.sum()) > 0
    assert seg.loc[sentinel, "ss_rel_s"].isna().all()
    assert (seg.loc[sentinel, "stage"] == "unknown").all()


def test_tc1_exclusions_counted(fixture_segments):
    seg, _ = fixture_segments
    excluded = seg[seg["excluded"]]
    assert (excluded["guid"] == fx.SPECIAL_GUID).all()
    for fold, part in excluded.groupby("fold"):
        assert sorted(part["exclusion_reason"]) == REASONS, fold
    special = seg[(seg["guid"] == fx.SPECIAL_GUID) & (seg["fold"] == 1)]
    doubled = special.loc[special["exclusion_reason"] == "duplicate_epoch", "epoch_s"].item()
    copies = special[special["epoch_s"] == doubled].sort_values("ds_index")
    assert list(copies["excluded"]) == [False, True]  # the first stored copy is kept


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


def test_tc2_cohort_filters(fixture_segments, smoke_config):
    cohort_cfg = smoke_config.cohort.model_copy(
        update={"include_healthy_no_bg": False, "min_segments_per_guid": 4})
    seg, guids = cohort.guid_table(fixture_segments[0], smoke_config.labels, cohort_cfg)
    no_bg = guids["source_file"] == "healthy_no_bg_no_cs"
    assert (guids.loc[no_bg, "exclusion_reason"] == "healthy_no_bg").all()
    short = ~no_bg & (guids["n_segments"] < 4)
    assert short.any() and (guids.loc[short, "exclusion_reason"] == "min_segments").all()
    assert not guids.loc[~no_bg & ~short, "excluded"].any()


def test_tc2_conflicting_labels_exclude_the_guid(fixture_segments, smoke_config):
    seg = fixture_segments[0].copy()
    victim = (seg["guid"] == "acidosis_no_cs-03") & (seg["fold"] == 1)
    seg.loc[victim.idxmax(), "class_code"] = 1
    seg, guids = cohort.guid_table(seg, smoke_config.labels, smoke_config.cohort)
    row = guids[(guids["guid"] == "acidosis_no_cs-03") & (guids["fold"] == 1)]
    assert row["exclusion_reason"].item() == "conflicting_class_codes"
    assert (seg.loc[victim, "exclusion_reason"] == "label_conflict").all()
    assert (seg.loc[victim, "seg_pos"] == -1).all() and (seg.loc[victim, "label_weight"] == 0).all()


def test_tc2_missing_class_code_is_an_exclusion(fixture_segments, smoke_config):
    seg = fixture_segments[0].copy()
    seg["class_code"] = seg["class_code"].astype("Int64")
    whole = (seg["guid"] == "acidosis_no_cs-03") & (seg["fold"] == 1)
    one = seg.index[(seg["guid"] == "hie_cs-05") & (seg["fold"] == 1) & ~seg["excluded"]][1]
    seg.loc[whole, "class_code"] = pd.NA
    seg.loc[one, "class_code"] = pd.NA
    seg, guids = cohort.guid_table(seg, smoke_config.labels, smoke_config.cohort)
    reason = guids[guids["fold"] == 1].set_index("guid")["exclusion_reason"]
    assert reason["acidosis_no_cs-03"] == "no_valid_class_code" and reason["hie_cs-05"] == ""
    assert (seg.loc[whole, "exclusion_reason"] == "label_conflict").all()
    assert seg.loc[one, "exclusion_reason"] == "label_conflict"  # the GUID keeps its code, not the code-less segment
    kept = seg[(seg["guid"] == "hie_cs-05") & (seg["fold"] == 1) & ~seg["excluded"]]
    assert kept["class_code"].notna().all() and list(kept["seg_pos"]) == list(range(len(kept)))
    row = guids[(guids["guid"] == "hie_cs-05") & (guids["fold"] == 1)].iloc[0]  # the GUID stats follow the kept
    assert (row["n_segments"], row["first_epoch_s"], row["last_t_end_s"]) == (
        len(kept), kept["epoch_s"].min(), kept["t_end_s"].max())


def test_tc2_min_segments_counts_segments_with_a_class_code(fixture_segments, smoke_config):
    """A GUID whose segments lose their code down to fewer than ``min_segments_per_guid`` is excluded."""
    seg = fixture_segments[0].copy()
    seg["class_code"] = seg["class_code"].astype("Int64")
    idx = seg.index[(seg["guid"] == "hie_cs-05") & (seg["fold"] == 1) & ~seg["excluded"]]
    seg.loc[idx[1:], "class_code"] = pd.NA  # one retained segment keeps its code
    cfg = smoke_config.cohort.model_copy(update={"min_segments_per_guid": 2})
    seg, guids = cohort.guid_table(seg, smoke_config.labels, cfg)
    row = guids[(guids["guid"] == "hie_cs-05") & (guids["fold"] == 1)].iloc[0]
    assert row["n_segments"] == 1 and row["exclusion_reason"] == "min_segments"
    assert (seg.loc[idx, "excluded"]).all()


def test_tc2_guid_table_columns(fixture_guids):
    seg, guids = fixture_guids
    kept = seg[~seg["excluded"]]
    positions = kept.groupby(cohort.KEYS)["seg_pos"].agg(list)
    assert all(p == list(range(len(p))) for p in positions)
    assert (seg.loc[seg["excluded"], "seg_pos"] == -1).all()
    counts = kept.groupby(cohort.KEYS).size()
    assert (guids.set_index(cohort.KEYS)["n_segments"] == counts.reindex(
        guids.set_index(cohort.KEYS).index)).all()
    assert not guids["excluded"].any() and (guids["guid"].dtype, guids["split"].dtype) == ("str", "str")


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
    np.testing.assert_allclose(_weights("final_only", 1), [0, 0, 0, 0, 1])  # k_warm: data.GuidDataset's w_pos
    np.testing.assert_allclose(_weights("mil", 1), 0.0)


def test_tc3_k_warm_auto_is_3_for_propagate_and_sequence_scope_only(smoke_config):
    labels = smoke_config.labels
    got = {(strategy, scope): cohort.warm_positions(labels.model_copy(update={"strategy": strategy}), scope)
           for strategy in ("propagate", "horizon", "horizon_decay", "final_only", "mil")
           for scope in ("sequence", "segment")}
    assert got == {(s, scope): 3 if (s, scope) == ("propagate", "sequence") else 0 for s, scope in got}
    assert cohort.warm_positions(labels.model_copy(update={"k_warm": 2}), "sequence") == 2


@pytest.mark.parametrize("strategy", ["propagate", "horizon", "horizon_decay", "final_only", "mil"])
def test_tc3_table_weights_every_strategy(fixture_segments, smoke_config, strategy):
    """ω on the fixture's segment table, known answers per strategy (horizon 1 h, half-life 0.5 h); ω never carries
    ``k_warm``, which drops positions from the per-position loss only (``data.GuidDataset``'s ``w_pos``)."""
    labels = smoke_config.labels.model_copy(update={"strategy": strategy})
    seg, _ = cohort.guid_table(fixture_segments[0], labels, smoke_config.cohort)
    kept = seg[~seg["excluded"]]
    pos, delta, n = (kept["y"] > 0).to_numpy(), kept["hours_to_delivery"].to_numpy(), kept["seg_pos"].to_numpy()
    last = n == kept.groupby(cohort.KEYS)["seg_pos"].transform("max").to_numpy()
    expected = {
        "propagate": np.ones(len(kept)),
        "horizon": np.where(pos, (delta <= 1.0).astype(float), 1.0),
        "horizon_decay": np.where(pos, 2.0 ** (-np.maximum(0.0, delta - 1.0) / 0.5), 1.0),
        "final_only": last.astype(float),
        "mil": np.zeros(len(kept)),
    }[strategy]
    np.testing.assert_allclose(kept["label_weight"], expected)
    assert (seg.loc[seg["excluded"], "label_weight"] == 0).all()
    assert 0 < expected.sum() or strategy == "mil"


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


def test_tc3_table_weights_follow_the_strategy(fixture_guids, smoke_config):
    seg, _ = fixture_guids
    kept = seg[~seg["excluded"]]
    labels = smoke_config.labels
    expected = np.where(kept["y"] > 0, 2.0 ** (-np.maximum(
        0, kept["hours_to_delivery"] - labels.horizon_h) / labels.decay_halflife_h), 1.0)
    np.testing.assert_allclose(kept["label_weight"], expected)
    assert (seg["in_eval_window"] == ~seg["excluded"]).all()  # eval_window: all


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


def test_tc4_class_presence_raises(fixture_guids):
    guids = fixture_guids[1]
    no_positive_val = guids[~((guids["fold"] == 1) & (guids["split"] == "val") & (guids["y"] == 1))]
    with pytest.raises(PilotConfigError):
        cohort.validate_folds(no_positive_val, task="adverse_vs_healthy")


def test_tc4_shared_test_detected(fixture_guids):
    guids, record = cohort.validate_folds(fixture_guids[1], task="adverse_vs_healthy")
    assert set(guids.loc[guids["shared_test"], "guid"]) == {fx.SHARED_TEST_GUID}
    assert record == {"patient_grouping": "guid", "n_shared_test_guids": 1}


def test_tc4_patient_column(fixture_guids, tmp_path):
    guids = fixture_guids[1]
    plain, _ = cohort.validate_folds(guids, task="adverse_vs_healthy")
    assert plain["patient"].dtype == "str" and (plain["patient"] == plain["guid"]).all()  # no map: the GUID
    splits = guids.sort_values("fold").groupby("guid")["split"].agg("|".join)  # same split in every fold
    a, b = splits.index[splits == splits[splits.duplicated()].iloc[0]][:2]
    patient_map = tmp_path / "patients.json"
    patient_map.write_text(json.dumps({a: "P1", b: "P1"}))
    mapped, record = cohort.validate_folds(guids, task="adverse_vs_healthy", patient_map=str(patient_map))
    pair = mapped["guid"].isin([a, b])
    assert set(mapped.loc[pair, "patient"]) == {"P1"}
    assert (mapped.loc[~pair, "patient"] == mapped.loc[~pair, "guid"]).all()
    assert record["patient_grouping"]["1"]["n_recordings_with_patient_id"] == 2


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


def test_tc5_join_sees_time_only_relative_to_the_segment_end(tmp_path):
    """Shifting every clock of a GUID (its segment ends and its observations' ``time_s``) changes nothing."""
    shift = {"A": 4850.0, "B": -97.0, "C": 12.0, "D": -3000.0}
    moved = SEG_ENDS.assign(t_end_s=SEG_ENDS["t_end_s"] + SEG_ENDS["guid_norm"].map(shift))
    (tmp_path / "moved").mkdir()
    pd.testing.assert_frame_equal(
        cohort.join_covariates(SEG_ENDS, _covariates(tmp_path, timed=_timed(OBS))),
        cohort.join_covariates(moved, _covariates(tmp_path / "moved", timed=_timed(OBS, shift))))


def test_tc5_static_join_on_guid_norm_and_table_checks(tmp_path):
    seg = pd.DataFrame({"guid_norm": ["AB12", "CD34", "EF56"], "t_end_s": -60.0})
    parity = (("parity", "categorical"),)
    got = cohort.join_covariates(seg, _covariates(tmp_path, static=[{"guid": "ab-12", "parity": "1"},
                                                                    {"guid": "CD34", "parity": None}],
                                                  variables=parity))
    assert list(got) == ["cov:parity"] and got["cov:parity"].iloc[0] == "1"  # static: no age column
    assert got["cov:parity"].isna().tolist() == [False, True, True]
    with pytest.raises(ValueError, match="exactly one covariate table"):
        cohort.join_covariates(seg, _covariates(tmp_path, static=[{"guid": "AB12", "parity": "1"}]))  # temp_c: none
    with pytest.raises(ValueError, match="exactly one covariate table"):
        cohort.join_covariates(seg, _covariates(tmp_path, static=[{"guid": "AB12", "temp_c": 37.0}],
                                                timed=_timed(OBS)))
    with pytest.raises(ValueError, match="one row per GUID"):
        cohort.join_covariates(seg, _covariates(tmp_path, static=[{"guid": "ab-12", "parity": "1"},
                                                                  {"guid": "AB12", "parity": "2"}], variables=parity))


def test_tc5_fixture_join(fixture_guids, fixture_tree, covariate_overrides):
    """On the fixture: no retained segment sees the reading after its GUID's last segment end, every age is within
    ``max_age_h``, and GUIDs the CSVs spell in their ``guid_norm`` form still join."""
    cov = config.load(config.REPO_ROOT / "teb_vae/classifier/configs/smoke.yaml",
                      covariate_overrides).classifier.context.covariates
    seg = fixture_guids[0]
    kept = cohort.join_covariates(seg, cov)[~seg["excluded"]]
    timed = pd.read_csv(fixture_tree["timed_csv"]).assign(guid_norm=lambda f: f["guid"].map(cohort.normalize_guid))
    future = timed[timed["value"] == fx.FUTURE_TEMP_C]
    last_end = seg[~seg["excluded"]].groupby("guid_norm")["t_end_s"].max()
    assert len(future) and (future["time_s"] > future["guid_norm"].map(last_end)).all()
    assert kept["cov:temp_c"].notna().mean() > 0.3 and not (kept["cov:temp_c"] == fx.FUTURE_TEMP_C).any()
    assert kept["cov_age_h:temp_c"].dropna().between(0.0, cov.max_age_h).all()
    static = pd.read_csv(fixture_tree["static_csv"], dtype=str)
    respelt = static[~static["guid"].isin(seg["guid"]) & static["parity"].notna()].set_index("guid")["parity"]
    rows = seg.loc[~seg["excluded"], "guid_norm"]
    assert len(respelt) and (kept.loc[rows.isin(respelt.index), "cov:parity"]
                             == rows[rows.isin(respelt.index)].map(respelt)).all()


def test_covariate_availability_and_confound_known_answer(tmp_path):
    seg = pd.DataFrame({"fold": 1, "split": "train", "guid": ["a", "a", "b", "c", "d"],
                        "excluded": [False, False, False, True, False], "cov:t": [np.nan, 1.0, np.nan, 2.0, 3.0]})
    guids = pd.DataFrame({"fold": 1, "split": "train", "guid": ["a", "b", "c", "d"],
                          "excluded": [False, False, True, False], "y": [0, 0, 1, 1], "has_tlo": True})
    avail = cohort.covariate_availability(seg, guids, _covariates(tmp_path, variables=(("t", "numeric"),)))
    assert list(avail) == cohort.AVAILABILITY_COLUMNS and avail["available"].dtype == bool
    assert avail["available"].tolist() == [True, False, False, True]  # c: only an excluded segment has a value
    record = cohort.confound_check(guids, 0.10, avail)
    assert record["1"]["t"] == {"missing_rate_by_class": {"0": 0.5, "1": 0.0}, "delta": 0.5, "flagged": True}
    assert not record["1"]["has_tlo"]["flagged"] and cohort.confound_flagged(record)
    assert not cohort.confound_flagged({"1": {"has_tlo": record["1"]["has_tlo"]}}) and not cohort.confound_flagged(None)
    empty = cohort.covariate_availability(seg, guids, _covariates(tmp_path, variables=()))
    assert list(empty) == cohort.AVAILABILITY_COLUMNS and empty.empty


# ---- confound check, context columns, fold summary ---------------------------------------------
def test_confound_check_flags_class_dependent_missingness():
    guids = pd.DataFrame({"fold": 1, "split": "train", "excluded": False,
                          "y": [0, 0, 0, 0, 1, 1, 1, 1],
                          "has_tlo": [True] * 4 + [False, False, True, True]})
    record = cohort.confound_check(guids, 0.10)["1"]["has_tlo"]
    assert record == {"missing_rate_by_class": {"0": 0.0, "1": 0.5}, "delta": 0.5, "flagged": True}


def test_context_features_known_answer():
    hours = np.e - 1.0  # psi(e - 1) = 1
    seg = pd.DataFrame({"tlo_end_s": [hours * 3600, np.nan], "epoch_s": [-5000.0, -3000.0],
                        "t_end_s": [-3740.0, -1740.0], "ss_rel_s": [hours * 3600 - 1260, np.nan],
                        "stage": ["second", "unknown"], "valid_frac": [0.9, 0.5],
                        "hours_to_delivery": [1.0, 0.5], "cs": [True, False]})
    ctx = cohort.context_features(seg)
    np.testing.assert_allclose(ctx["tlo_psi"], [1.0, 0.0])
    np.testing.assert_allclose(ctx["tlo_missing"], [0.0, 1.0])
    np.testing.assert_allclose(ctx["time_in_ss"], [1.0, 0.0])
    assert [c for c in ctx if c.startswith("stage_")] == ["stage_straddle", "stage_second"]
    assert list(ctx["stage_second"]) == [1.0, 0.0] and list(ctx["stage_straddle"]) == [0.0, 0.0]
    assert not set(ctx) & {"epoch_s", "t_end_s", "hours_to_delivery", "cs", "bg", "source_file",
                           "class_code", "ss_rel_s"}


def test_context_hides_whether_second_stage_is_reached():
    """§7.1: pre-onset segments of a GUID that later reaches second stage and of a GUID whose onset is unknown
    have the same context (an ``unknown`` flag would be ``~has_ss`` from segment 0)."""
    seg = pd.DataFrame({"tlo_end_s": 7200.0, "epoch_s": -9000.0, "t_end_s": -7740.0, "valid_frac": 0.8,
                        "ss_rel_s": [-5000.0, np.nan]})
    seg["stage"] = cohort.stage_of(seg["ss_rel_s"], 60.0, 1260.0)
    assert list(seg["stage"]) == ["first", "unknown"]
    ctx = cohort.context_features(seg)
    pd.testing.assert_series_equal(ctx.iloc[0], ctx.iloc[1], check_names=False)


def test_context_tlo_hides_time_until_onset():
    """``context.tlo.pre_onset: clip`` (default): pre-onset segments look alike whatever their time until onset
    (future information, like time until second stage, §2.4); ``signed`` (ablation) keeps ψ of the negative hours."""
    seg = pd.DataFrame({"tlo_end_s": [-3600.0, -30000.0, 3600.0], "epoch_s": -9000.0, "t_end_s": -7740.0,
                        "valid_frac": 0.8, "ss_rel_s": np.nan, "stage": "unknown"})
    np.testing.assert_allclose(cohort.context_features(seg)["tlo_psi"], [0.0, 0.0, np.log(2.0)])
    np.testing.assert_allclose(cohort.context_features(seg, tlo_pre_onset="signed")["tlo_psi"],
                               [-np.log(2.0), -np.log1p(30000.0 / 3600.0), np.log(2.0)])


def test_fold_summary_counts_add_up(fixture_guids):
    seg, guids = fixture_guids
    summary = cohort.fold_summary(seg, guids)
    assert summary.loc[summary["level"] == "segment", "n"].sum() == len(seg)
    assert summary.loc[summary["level"] == "guid", "n"].sum() == len(guids)
    assert set(summary["status"]) == {"retained", *REASONS}


# ---- the cohort stage through run.py -----------------------------------------------------------
def test_run_cohort_stage(smoke_overrides, tmp_path):
    import sys

    from loguru import logger

    from teb_vae.classifier import run

    smoke = str(config.REPO_ROOT / "teb_vae" / "classifier" / "configs" / "smoke.yaml")
    try:
        run_dir = run.main(config=smoke, stage="cohort", overrides=smoke_overrides,
                           run_dir=str(tmp_path / "run"))
        for name in ("config.resolved.yaml", "manifest.json", "stage_state.json", "run.log",
                     "run.jsonl", "cohort/segments.parquet", "cohort/guids.parquet",
                     "cohort/fold_summary.csv", "cohort/exposure.json", "cohort/confound.json",
                     "cohort/covariate_availability.parquet"):
            assert (run_dir / name).is_file(), name
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["config_digest"] == config.digest(config.load(smoke, smoke_overrides))
        assert len(manifest["shards"]) == 24
        assert all(s["size"] > 0 and s["source_guid_digest"] for s in manifest["shards"])
        assert manifest["exposure"]["applicable"] is False
        assert manifest["cohort"]["n_shared_test_guids"] == 1 and "1" in manifest["confound"]
        assert manifest["software"]["revision"] and manifest["versions"]["pandas"]
        state = json.loads((run_dir / "stage_state.json").read_text())
        assert state["cohort"]["status"] == "done"
        segments = pd.read_parquet(run_dir / "cohort" / "segments.parquet")
        assert list(segments) == cohort.SEGMENT_COLUMNS and set(segments["fold"]) == {1, 2}
        assert list(pd.read_parquet(run_dir / "cohort" / "guids.parquet")) == cohort.GUID_COLUMNS

        run.main(config=smoke, stage="cohort", overrides=smoke_overrides, run_dir=str(run_dir),
                 device="cpu")  # resume: skipped, and the device is not part of the digest
        assert json.loads((run_dir / "stage_state.json").read_text()) == state
        with pytest.raises(ValueError, match="different config"):
            run.main(config=smoke, stage="cohort", run_dir=str(run_dir),
                     overrides=smoke_overrides + ["classifier.labels.horizon_h=2"])
    finally:  # run.main replaced loguru's sinks with files under tmp_path
        logger.remove()
        logger.add(sys.stderr)


# ---- P0 acceptance on the sample shard ---------------------------------------------------------
@pytest.mark.skipif(not SAMPLE_SHARD.is_file(), reason="tmp/data/hie_cs.hdf5 is not on this machine")
def test_p0_sample_shard():
    seg, info = cohort.segment_table({(0, "test"): [str(SAMPLE_SHARD)]}, trim_minutes=1.0)
    assert info["stride_s"] == 660.0
    assert (seg["guid"].nunique(), len(seg)) == (15, 614)
    assert info["stage_counts"] == {"first": 170, "straddle": 4, "second": 44, "unknown": 396}
    assert round(info["tlo_nan_rate"], 3) == 0.135
    assert info["sentinel_guids"] == [] and not seg["excluded"].any()
