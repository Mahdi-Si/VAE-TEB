"""baselines.py: probe C on val, online/final/segment scores, shortcut inputs, train-only scaler and L5
(SPEC §10.9, §12)."""
from __future__ import annotations

import json
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

from teb_vae.classifier.tests.conftest import SMOKE_CONFIG


@pytest.fixture(scope="module")
def trained(smoke_overrides, tmp_path_factory):
    """cohort -> extract -> train (fold 1) on the fixture; ``(cfg, run_dir, features, fit_scaler calls)``."""
    from loguru import logger

    from teb_vae.classifier import baselines, run
    from teb_vae.classifier.config import load

    overrides = list(smoke_overrides) + ["classifier.run.folds=[1]"]
    run_dir, calls = tmp_path_factory.mktemp("baselines") / "run", []
    real = baselines.fit_scaler
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(baselines, "fit_scaler",
                       lambda index, *a, **k: calls.append(index.copy()) or real(index, *a, **k))
            for stage in ("cohort", "extract", "train"):
                run.main(config=str(SMOKE_CONFIG), stage=stage, overrides=overrides, run_dir=str(run_dir))
    finally:  # run.main replaced loguru's sinks with files under tmp_path
        logger.remove()
        logger.add(sys.stderr)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return load(SMOKE_CONFIG, overrides), run_dir, baselines.load_features(manifest["source"]), calls


def _last_features(cfg, run_dir, feats, split):
    from teb_vae.classifier import baselines
    from teb_vae.classifier.sources import Scaler

    frame = baselines.fold_frame(run_dir, 1, feats.index, feats.count)
    scaler = Scaler.load(baselines.fold_dir(run_dir, 1) / "scaler.json")
    online, _ = baselines.probe_features(frame, feats, scaler, cfg.classifier.baselines.probe_last_n)
    pick = ((frame["seg_pos"] == frame.groupby(["split", "guid"])["seg_pos"].transform("max"))
            & (frame["split"] == split)).to_numpy()
    return online[pick], frame["y"][pick].to_numpy(int)


def test_probe_c_chosen_on_val_logloss(trained):
    import skops.io as sio
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    from teb_vae.classifier import baselines

    cfg, run_dir, feats, _ = trained
    out = baselines.fold_dir(run_dir, 1)
    fit = json.loads((out / "probe_fit.json").read_text())
    (X_tr, y_tr), (X_va, y_va) = (_last_features(cfg, run_dir, feats, s) for s in ("train", "val"))
    loss = {C: log_loss(y_va, LogisticRegression(C=C, max_iter=5000).fit(X_tr, y_tr).predict_proba(X_va)[:, 1])
            for C in baselines.PROBE_C}
    assert fit["val_logloss_by_C"] == pytest.approx({str(C): v for C, v in loss.items()}, rel=1e-4)
    assert fit["C"] == min(loss, key=loss.get) == sio.load(out / "probe.skops").C
    assert fit["n_by_class"]["val"] == {str(k): int((y_va == k).sum()) for k in (0, 1)}


def test_probe_online_final_and_segment_scores(trained):
    import skops.io as sio

    from teb_vae.classifier import baselines
    from teb_vae.classifier.sources import Scaler, read_rows

    cfg, run_dir, feats, _ = trained
    seg, gd, lock = baselines.score_fold(cfg, run_dir, 1, feats, ["val"])
    assert lock == {} and set(seg["split"]) == set(gd["split"]) == {"val"}
    p, g = seg[seg["model_id"] == "probe"], gd[gd["model_id"] == "probe"].set_index("guid")
    last = p.loc[p.groupby("guid")["seg_pos"].idxmax()].set_index("guid")
    assert np.allclose(last["logit_online"], g.loc[last.index, "score_final"])  # score_final = online at the end
    first = p[p["seg_pos"] == 0]
    assert np.allclose(first["logit_seg"], first["logit_online"])  # a one-segment window is the segment
    assert (p["logit_seg_cal"] == p["logit_seg"]).all() and (g["score_final_cal"] == g["score_final"]).all()

    # Independent recomputation from the cache: pooled steps of the last 3 segments vs one segment.
    out, L = baselines.fold_dir(run_dir, 1), cfg.classifier.baselines.probe_last_n
    probe, scaler = sio.load(out / "probe.skops"), Scaler.load(out / "scaler.json")
    frame = baselines.fold_frame(run_dir, 1, feats.index, feats.count)
    frame = frame[frame["split"] == "val"]
    guid = frame.groupby("guid").size().idxmax()
    rows = frame[frame["guid"] == guid].sort_values("seg_pos")
    assert len(rows) > L
    f = read_rows(feats.cache_dir, rows["row"].to_numpy()[-L:])
    x, m = scaler.apply(f.values.numpy()), f.step_mask.numpy()

    def score(x, m):
        return probe.decision_function(np.r_[x[m].mean(0), x[m].max(0)][None])[0]

    mine = p[p["guid"] == guid].sort_values("seg_pos")
    assert score(x, m) == pytest.approx(mine["logit_online"].iloc[-1], abs=1e-4)
    assert score(x[-1:], m[-1:]) == pytest.approx(mine["logit_seg"].iloc[-1], abs=1e-4)
    assert not np.isclose(mine["logit_seg"].iloc[-1], mine["logit_online"].iloc[-1])


def test_shortcut_scores_only_metadata(trained):
    import skops.io as sio

    from teb_vae.classifier import baselines

    cfg, run_dir, feats, _ = trained
    out = baselines.fold_dir(run_dir, 1)
    model = sio.load(out / "shortcut.skops")
    assert json.loads((out / "shortcut_fit.json").read_text())["features"] == baselines.SHORTCUT_FEATURES
    assert model.n_features_in_ == len(baselines.SHORTCUT_FEATURES) == 5
    seg, gd, _ = baselines.score_fold(cfg, run_dir, 1, feats, ["val"])
    s = seg[seg["model_id"] == "shortcut"]
    assert s[["logit_seg", "logit_seg_cal", "logit_online", "logit_online_cal"]].isna().all().all()

    co = run_dir / "cohort"
    cg, cs = pd.read_parquet(co / "guids.parquet"), pd.read_parquet(co / "segments.parquet")
    cg = cg[(cg["fold"] == 1) & (cg["split"] == "val") & ~cg["excluded"]].set_index("guid")
    vf = cs[(cs["fold"] == 1) & (cs["split"] == "val") & ~cs["excluded"]].groupby("guid")["valid_frac"].mean()
    X = np.c_[cg["n_segments"], (cg["last_t_end_s"] - cg["first_epoch_s"]) / 3600, cg["has_tlo"], cg["has_ss"],
              vf.loc[cg.index]]
    g = gd[gd["model_id"] == "shortcut"].set_index("guid").loc[cg.index]
    assert np.allclose(g["score_final"], model.decision_function(X))


def test_scaler_fit_on_train_only(trained):
    from teb_vae.classifier import baselines

    _, run_dir, _, calls = trained
    guids = pd.read_parquet(run_dir / "cohort" / "guids.parquet")
    train = set(guids["guid"][(guids["fold"] == 1) & (guids["split"] == "train") & ~guids["excluded"]])
    assert len(calls) == 1 and set(calls[0]["split"]) == {"train"} and set(calls[0]["guid"]) == train
    record = json.loads((baselines.fold_dir(run_dir, 1) / "scaler.json").read_text())["record"]
    assert record["population"] == "train" and record["n_guids"] == len(train)


def test_segments_without_a_valid_cached_step_are_excluded(trained, tmp_path):
    """A retained segment whose cached step_mask is all False leaves the one population of the baselines and the
    neural models (``no_valid_steps``), with a WARNING and per-split counts; ``seg_pos`` stays 0..n-1."""
    import h5py
    from loguru import logger

    from teb_vae.classifier import baselines
    from teb_vae.classifier.data import build_unit

    cfg, run_dir, feats, _ = trained
    frame = baselines.fold_frame(run_dir, 1, feats.index, feats.count)
    none = {s: 0 for s in ("test", "train", "val")}
    assert frame.attrs["no_valid_steps"] == {"segments": none, "guids": none, "min_segments": none}
    fit = json.loads((baselines.fold_dir(run_dir, 1) / "probe_fit.json").read_text())
    assert fit["no_valid_steps"] == frame.attrs["no_valid_steps"]  # the persistent per-fold record

    val, train = frame[frame["split"] == "val"], frame[frame["split"] == "train"]
    middle = val.loc[(val["guid"] == val.groupby("guid").size().idxmax()) & (val["seg_pos"] == 1), "row"]
    lost = train.loc[train["guid"] == train["guid"].iloc[0], "row"]  # every segment of one train GUID
    shutil.copytree(feats.cache_dir, tmp_path / "cache")
    with h5py.File(tmp_path / "cache" / "features.h5", "a") as h5:
        mask = h5["step_mask"][()]
        mask[np.r_[middle, lost]] = False
        h5["step_mask"][...] = mask
    cache = {"cache_dir": str(tmp_path / "cache"), "fingerprint": None}
    planted, messages = baselines.load_features(cache), []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        got = baselines.fold_frame(run_dir, 1, planted.index, planted.count)
    finally:
        logger.remove(sink)
    assert any("no_valid_steps" in m for m in messages)
    assert got.attrs["no_valid_steps"] == {"segments": none | {"train": len(lost), "val": 1},
                                           "guids": none | {"train": 1}, "min_segments": none}
    assert set(frame["row"]) - set(got["row"]) == set(middle) | set(lost) and len(got) == len(frame) - len(lost) - 1
    g = got.groupby(["split", "guid"])
    assert (g.cumcount() == got["seg_pos"]).all() and (g["seg_pos"].transform("size") == got["n_segments"]).all()
    assert (baselines.label_rows(run_dir, 1, planted.index, ["val", "test"], planted.count)["n_valid_steps"] > 0).all()
    unit = build_unit(cfg, run_dir, cache, 1)
    assert set(pd.concat(unit.frames.values())["row"]) == set(got["row"])

    big = val.groupby("guid").size()  # min_segments_per_guid is re-applied after the drop
    n, name = big.max(), big.idxmax()
    short = baselines.fold_frame(run_dir, 1, planted.index, planted.count, min_segments=n)
    assert name not in set(short["guid"]) and (short["n_segments"] >= n).all()
    assert short.attrs["no_valid_steps"]["min_segments"]["val"] >= 1


def test_prediction_rows_carry_the_patient(trained, tmp_path):
    """The shared row builder (``label_rows``, also the neural units') carries the cohort's ``patient``."""
    from teb_vae.classifier import baselines

    cfg, run_dir, feats, _ = trained
    seg, gd, _ = baselines.score_fold(cfg, run_dir, 1, feats, ["val"])
    assert "patient" in baselines.SEGMENT_COLUMNS and "patient" in baselines.GUID_COLUMNS
    assert (seg["patient"] == seg["guid"]).all() and (gd["patient"] == gd["guid"]).all()  # no patient map
    shutil.copytree(run_dir / "cohort", tmp_path / "run" / "cohort")
    guids = pd.read_parquet(tmp_path / "run" / "cohort" / "guids.parquet")
    pair = guids.loc[(guids["fold"] == 1) & (guids["split"] == "val") & ~guids["excluded"], "guid"].iloc[:2]
    guids.assign(patient=guids["patient"].mask(guids["guid"].isin(pair), "P1")).to_parquet(
        tmp_path / "run" / "cohort" / "guids.parquet", index=False)
    rows = baselines.label_rows(tmp_path / "run", 1, feats.index, ["val"], feats.count)
    assert rows.loc[rows["patient"] == "P1", "guid"].nunique() == 2
    assert (rows.loc[rows["patient"] != "P1", "patient"] == rows.loc[rows["patient"] != "P1", "guid"]).all()


def test_l2_test_rows_need_the_lock(trained):
    """T-L2 style (L5): no lock or a file changed after locking -> test rows are refused."""
    from teb_vae.classifier import baselines
    from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import RunStateError

    cfg, run_dir, feats, _ = trained
    out = baselines.fold_dir(run_dir, 1)
    lock, thr = out / baselines.LOCK, out / "thresholds_probe.json"
    stamp, text = lock.read_text(), thr.read_text()
    shutil.move(lock, out / "aside.json")
    try:
        with pytest.raises(RunStateError, match="L5"):
            baselines.predict(cfg, run_dir, {"cache_dir": feats.cache_dir, "fingerprint": None})
        baselines.score_fold(cfg, run_dir, 1, feats, ["val"])  # val never needs the lock
    finally:
        shutil.move(out / "aside.json", lock)
    thr.write_text(text + " ")
    try:
        with pytest.raises(RunStateError, match="thresholds_probe.json"):
            baselines.score_fold(cfg, run_dir, 1, feats, ["val", "test"])
    finally:
        thr.write_text(text)
    seg, _, got = baselines.score_fold(cfg, run_dir, 1, feats, ["val", "test"])
    assert set(seg["split"]) == {"val", "test"} and got["locked_utc"] < baselines.utc_now()
    baselines.train(cfg, run_dir, {"cache_dir": feats.cache_dir, "fingerprint": None})  # resume: locked, skipped
    assert lock.read_text() == stamp


def test_three_class_probe_probabilities_keep_ovr_logits_finite():
    """A multinomial probe's ``predict_proba`` can be exactly 0 or 1; stored ``p_c<k>_cal`` are clipped like the neural
    ones, so the OvR thresholds' ``logit(p_c<k>_cal)`` stays finite."""
    from scipy.special import logit

    from teb_vae.classifier import baselines

    cols = baselines._three_class(np.array([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5]]), 2)
    P = np.stack([cols[f"p_c{k}_cal"] for k in range(3)], 1)
    assert np.isfinite(logit(P)).all() and np.allclose(P.sum(1), 1.0)
    assert np.isnan(baselines._three_class(None, 2)["p_c0_cal"]).all()


def test_final_only_weight_follows_a_dropped_last_segment(trained):
    """``labels.strategy: final_only``: when ``no_valid_steps`` drops a GUID's last segment, ω moves to the kept last
    one (the cohort's ω would leave that GUID with no weight at all)."""
    from teb_vae.classifier import baselines

    _, run_dir, feats, _ = trained
    val = baselines.fold_frame(run_dir, 1, feats.index, feats.count).query("split == 'val'")
    guid = val.groupby("guid").size().idxmax()
    count = np.array(feats.count, copy=True)
    count[val.loc[val["guid"] == guid, "row"].iloc[-1]] = 0  # its last segment has no valid cached step
    got = baselines.fold_frame(run_dir, 1, feats.index, count, strategy="final_only").query("split == 'val'")
    assert (got.groupby("guid")["label_weight"].sum() == 1).all()
    mine = got[got["guid"] == guid]
    assert list(mine["label_weight"]) == [0.0] * (len(mine) - 1) + [1.0]
