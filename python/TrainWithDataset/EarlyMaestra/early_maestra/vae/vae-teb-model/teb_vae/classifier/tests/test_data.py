"""data.py: the P3/P4 batch contract, samplers, segment dropout, the shuffled-label control, covariates (encoding,
ModDrop, the covariates-off pass), L4, and T-L1 at batch level (SPEC §7.1, §7.3, §10.4, §10.9.3, §12, §15), on
fold 1 of the fixture tree."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.classifier.tests.conftest import SMOKE_CONFIG

SEGMENT_KEYS = {"x", "step_mask", "ctx", "y", "y3", "w", "guid", "row"}
@pytest.fixture(scope="module")
def env(smoke_overrides, tmp_path_factory):
    """cohort -> extract (fold 1) into a private tmp tree: ``(overrides, run_dir, cache record)``."""
    from loguru import logger

    from teb_vae.classifier import run

    tmp = tmp_path_factory.mktemp("data")
    overrides = list(smoke_overrides) + ["classifier.run.folds=[1]", f"classifier.source.cache_root={tmp / 'cache'}"]
    try:
        for stage in ("cohort", "extract"):
            run.main(config=str(SMOKE_CONFIG), stage=stage, overrides=overrides, run_dir=str(tmp / "run"))
    finally:  # run.main replaced loguru's sinks with files under tmp
        logger.remove()
        logger.add(sys.stderr)
    return overrides, tmp / "run", json.loads((tmp / "run" / "manifest.json").read_text())["source"]


@pytest.fixture(scope="module")
def cov_env(env, covariate_overrides, tmp_path_factory):
    """``env`` with the fixture's covariates: its own cohort; extract reuses ``env``'s cache (same segments)."""
    from loguru import logger

    from teb_vae.classifier import run

    tmp = tmp_path_factory.mktemp("data_cov")
    overrides = list(covariate_overrides) + ["classifier.run.folds=[1]",
                                             f"classifier.source.cache_root={Path(env[2]['cache_dir']).parent}"]
    try:
        for stage in ("cohort", "extract"):
            run.main(config=str(SMOKE_CONFIG), stage=stage, overrides=overrides, run_dir=str(tmp / "run"))
    finally:
        logger.remove()
        logger.add(sys.stderr)
    source = json.loads((tmp / "run" / "manifest.json").read_text())["source"]
    assert source["cache_dir"] == env[2]["cache_dir"]
    return overrides, tmp / "run", source


def _cfg(env, *sets):
    from teb_vae.classifier.config import load

    return load(SMOKE_CONFIG, env[0] + list(sets)).classifier


def _unit(env, *sets, run_dir=None, cache=None, shuffle_seed=None):
    from teb_vae.classifier.data import build_unit

    return build_unit(_cfg(env, *sets), run_dir or env[1], cache or env[2], 1, shuffle_seed=shuffle_seed)


def _loader(unit, split, train, seed=0):
    from teb_vae.classifier.data import make_loader

    return make_loader(unit, split, train=train, seed=seed)


def _log1p_dt(t):
    return np.log1p(np.diff(t, prepend=t[0]) / 3600.0)


# ---- columns, allow-list, L4 -------------------------------------------------------------------
def test_allow_list_class_counts_priors_and_train_only_fits(env):
    from teb_vae.classifier import data

    unit = _unit(env)
    assert all(list(f.columns) == data.ALLOWED_COLUMNS for f in unit.frames.values())
    assert not {"hours_to_delivery", "cs", "bg", "source_file", "n_segments", "first_epoch_s"} & set(data.ALLOWED_COLUMNS)
    g = unit.frames["train"].groupby("guid")[["y", "class_code"]].first()
    assert unit.class_counts == np.bincount(g["y"].astype(int), minlength=2).tolist()
    np.testing.assert_allclose(unit.priors["main"], np.array(unit.class_counts) / len(g))
    np.testing.assert_allclose(unit.priors["aux3"], np.bincount(g["class_code"].astype(int) - 1, minlength=3) / len(g))
    assert unit.scaler.record["population"] == "train" and unit.scaler.record["n_guids"] == len(g)
    assert (unit.n_values, unit.n_attn, unit.n_ctx) == (153, 0, 7)
    with pytest.raises(ValueError, match="L4"):
        data.train_counts(unit.frames["val"], 2)


# ---- batch contract ----------------------------------------------------------------------------
def test_sequence_batch_contract_and_padding(env):
    unit = _unit(env)
    frame, loader = unit.frames["val"], _loader(unit, "val", False)
    t_end, n_seen = frame["t_end_s"].to_numpy(), 0
    for batch in loader:
        assert set(batch) == SEGMENT_KEYS | {"seg_mask", "t_h", "w_pos"}
        b, n = batch["seg_mask"].shape
        assert batch["x"].shape == (b, n, unit.store.step_mask.shape[1], unit.n_values)
        assert batch["step_mask"].shape == batch["x"].shape[:3] and batch["ctx"].shape == (b, n, unit.n_ctx)
        assert all(batch[k].shape == (b, n) for k in ("t_h", "w", "w_pos", "row"))
        assert batch["t_h"].dtype == batch["w"].dtype == torch.float32 and batch["row"].dtype == torch.int64
        assert all(batch[k].shape == (b,) for k in ("y", "y3", "guid")) and batch["y"].dtype == torch.float32
        pad = ~batch["seg_mask"]
        assert (batch["row"][pad] == -1).all() and (batch["w"][pad] == 0).all() and (batch["t_h"][pad] == 0).all()
        assert (batch["w_pos"][pad] == 0).all()
        assert (batch["x"][pad] == 0).all() and not batch["step_mask"][pad].any() and (batch["ctx"][pad] == 0).all()
        for i in range(b):
            rows = batch["row"][i][batch["seg_mask"][i]].numpy()
            f = frame.iloc[rows]
            assert f["guid"].nunique() == 1 and f["guid"].iloc[0] == loader.dataset.guids[int(batch["guid"][i])]
            assert (f["seg_pos"].to_numpy() == np.arange(len(f))).all()
            assert batch["y"][i] == f["y"].iloc[0] and batch["y3"][i] == f["class_code"].iloc[0] - 1
            np.testing.assert_array_equal(batch["w"][i, :len(f)], f["label_weight"].to_numpy(np.float32))  # raw ω
            t = t_end[rows]
            np.testing.assert_allclose(batch["t_h"][i, :len(f)], (t - t[0]) / 3600.0, rtol=1e-6)
            np.testing.assert_allclose(batch["ctx"][i, :len(f), -1], _log1p_dt(t), rtol=1e-6)  # delta_t
            n_seen += len(f)
    assert n_seen == len(frame)


# ---- samplers ----------------------------------------------------------------------------------
# ---- shuffled-label control (§10.9.3) ----------------------------------------------------------
def test_shuffled_labels_stay_within_train(env):
    from teb_vae.classifier.config import TASKS

    plain, shuffled = _unit(env), _unit(env, shuffle_seed=7)
    for split in ("val", "test"):
        pd.testing.assert_frame_equal(plain.frames[split], shuffled.frames[split])
    a, b = plain.frames["train"], shuffled.frames["train"]
    labels = ["y", "class_code"]
    pd.testing.assert_frame_equal(a.drop(columns=labels), b.drop(columns=labels))  # ω too: the cohort's own
    ga, gb = (f.groupby("guid")[["y", "class_code"]].first() for f in (a, b))
    assert sorted(ga["class_code"]) == sorted(gb["class_code"]) and (ga["y"] != gb["y"]).any()
    assert (gb["y"] == gb["class_code"].map(TASKS["adverse_vs_healthy"])).all()  # pairs permuted together
    assert shuffled.priors == plain.priors and shuffled.class_counts == plain.class_counts
    assert _unit(env, shuffle_seed=7).frames["train"].equals(b)
    assert not _unit(env, shuffle_seed=8).frames["train"]["y"].equals(b["y"])


# ---- covariates (§7.3) -------------------------------------------------------------------------
# ---- T-L1 at batch level (§12 L1, §15) ---------------------------------------------------------
def perturb_forbidden(run_dir, source, dest, covariates=None):
    """A copy of ``run_dir/cohort`` and the feature cache under ``dest`` with every forbidden input (§7.1, L1)
    permuted or randomised: absolute clocks shifted per GUID (within-GUID differences are allowed), the negative
    part of ``ss_rel_s`` and whether it is known (first vs unknown stage), the magnitude of a negative (pre-onset)
    ``tlo_end_s``, ``hours_to_delivery``,
    ``cs``/``bg``/``source_file`` and the GUID metadata. With ``covariates`` (the run's ``CovariatesCfg``), the
    covariate columns are re-joined on the shifted clocks from a copy of the timed table whose ``time_s`` moves with
    its GUID, so covariate time can only matter relative to the segment end (§7.3). Returns ``(perturbed run dir,
    perturbed cache record)``; also used by the model-level T-L1 (test_leakage.py)."""
    rng = np.random.default_rng(0)
    seg = pd.read_parquet(run_dir / "cohort" / "segments.parquet")
    gd = pd.read_parquet(run_dir / "cohort" / "guids.parquet")
    index = pd.read_parquet(source["cache_dir"] + "/index.parquet")
    guids = seg["guid"].unique()
    shift = pd.Series(rng.integers(-50, 50, len(guids)) * 97.0, index=guids)  # per GUID: Δt is allowed

    window = seg["t_end_s"] - seg["epoch_s"]
    # Not (known to be) in second stage yet: first and unknown must look alike (§7.1, has_ss), so each such
    # segment gets an unknown onset or a random future one.
    pending = (seg["ss_rel_s"].isna() | (seg["ss_rel_s"] + window < 0)).to_numpy()
    unknown = pending & (rng.random(len(seg)) < 0.5)
    stage = seg["stage"].to_numpy()
    assert (unknown & (stage == "first")).any() and (pending & ~unknown & (stage == "unknown")).any()
    seg["ss_rel_s"] = seg["ss_rel_s"].mask(pending, -window - rng.uniform(1.0, 1e5, len(seg))).mask(unknown, np.nan)
    seg["stage"] = seg["stage"].mask(pending, np.where(unknown, "unknown", "first"))
    for frame in (seg, index):
        frame["epoch_s"] += frame["guid"].map(shift)
    seg["t_end_s"] += seg["guid"].map(shift)
    if covariates is not None and covariates.variables and covariates.timed_csv:
        from teb_vae.classifier.cohort import join_covariates, normalize_guid

        timed = pd.read_csv(covariates.timed_csv, dtype={"guid": str, "variable": str, "value": str})
        moved = timed["time_s"] + timed["guid"].map(normalize_guid).map(seg.groupby("guid_norm")["guid"].first()
                                                                         .map(shift)).fillna(0.0)
        assert (moved != timed["time_s"]).any()
        dest.mkdir(parents=True, exist_ok=True)
        timed.assign(time_s=moved).to_csv(dest / "timed.csv", index=False)
        seg = seg.assign(**join_covariates(seg, covariates.model_copy(update={"timed_csv": str(dest / "timed.csv")})))
    seg["hours_to_delivery"] = rng.normal(0.0, 10.0, len(seg))
    pre_onset = (seg["tlo_end_s"] < 0).to_numpy()  # time UNTIL onset is forbidden (context.tlo.pre_onset: clip)
    assert pre_onset.any() and (seg["tlo_end_s"] > 0).any()
    seg["tlo_end_s"] = seg["tlo_end_s"].mask(pre_onset, -rng.uniform(1.0, 1e5, len(seg)))
    for frame, cols in ((seg, ("cs", "bg", "source_file", "slot", "guid_norm")),
                        (gd, ("cs", "bg", "source_file", "has_tlo", "has_ss", "shared_test", "patient")),
                        (index, ("source_file",))):
        for col in cols:
            frame[col] = rng.permutation(frame[col].to_numpy())
    gd["n_segments"] = rng.integers(1, 99, len(gd))
    gd["first_epoch_s"] = rng.uniform(-1e5, 0.0, len(gd))
    gd["last_t_end_s"] = rng.uniform(-1e4, 0.0, len(gd))
    (dest / "run" / "cohort").mkdir(parents=True)
    seg.to_parquet(dest / "run" / "cohort" / "segments.parquet", index=False)
    gd.to_parquet(dest / "run" / "cohort" / "guids.parquet", index=False)
    shutil.copytree(source["cache_dir"], dest / "cache")
    index.to_parquet(dest / "cache" / "index.parquet", index=False)
    return dest / "run", {**source, "cache_dir": str(dest / "cache")}


@pytest.mark.parametrize("which", ["env", "cov_env"])
def test_t_l1_forbidden_columns_never_reach_a_batch(request, which, tmp_path):
    env = request.getfixturevalue(which)
    _, run_dir, source = env
    _, perturbed = perturb_forbidden(run_dir, source, tmp_path, _cfg(env).context.covariates)
    for sets in (["classifier.model.scope=segment", "classifier.context.elapsed.enabled=true"],
                 ["classifier.context.elapsed.enabled=true", "classifier.train.sampler=class_balanced"]):
        a = _unit(env, *sets)
        b = _unit(env, *sets, run_dir=tmp_path / "run", cache=perturbed)
        assert a.n_cov == (7 if which == "cov_env" else 0)
        assert not a.frames["train"]["t_end_s"].equals(b.frames["train"]["t_end_s"])  # really perturbed
        for split, train in (("train", True), ("val", False), ("test", False)):
            la, lb = _loader(a, split, train, seed=11), _loader(b, split, train, seed=11)
            for _ in range(2):  # two epochs: shuffling, balanced draws and segment dropout
                xs, ys = list(la), list(lb)
                assert len(xs) == len(ys)
                for x, y in zip(xs, ys):
                    assert x.keys() == y.keys()
                    for k in x:
                        assert x[k].dtype == y[k].dtype and torch.equal(x[k], y[k]), f"{sets} {split}: {k!r} moved"
