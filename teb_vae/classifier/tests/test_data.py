"""data.py: the P3/P4 batch contract, samplers, segment dropout, the shuffled-label control, covariates (encoding,
ModDrop, the covariates-off pass), L4, and T-L1 at batch level (SPEC §7.1, §7.3, §10.4, §10.9.3, §12, §15), on
fold 1 of the fixture tree."""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.classifier.tests.conftest import SMOKE_CONFIG

SEGMENT_KEYS = {"x", "step_mask", "ctx", "y", "y3", "w", "guid", "row"}
BASE_CTX = ["tlo_psi", "tlo_missing", "stage_straddle", "stage_second", "time_in_ss", "valid_frac"]


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


def _same(xs, ys) -> bool:
    return len(xs) == len(ys) and all(
        a.keys() == b.keys() and all(a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]) for k in a)
        for a, b in zip(xs, ys))


def _log1p_dt(t):
    return np.log1p(np.diff(t, prepend=t[0]) / 3600.0)


# ---- columns, allow-list, L4 -------------------------------------------------------------------
def test_context_columns_follow_the_toggles(env):
    from teb_vae.classifier.data import CONTEXT_COLUMNS

    assert CONTEXT_COLUMNS(_cfg(env)) == BASE_CTX + ["delta_t"]  # smoke = sequence scope
    assert CONTEXT_COLUMNS(_cfg(env, "classifier.model.scope=segment")) == BASE_CTX
    assert CONTEXT_COLUMNS(_cfg(env, "classifier.context.tlo.missing=no_indicator",
                                "classifier.context.stage.enabled=false",
                                "classifier.context.elapsed.enabled=true")) == [
        "tlo_psi", "time_in_ss", "valid_frac", "delta_t", "elapsed"]


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
@pytest.mark.parametrize("head", ["binary", "multiclass"])
def test_segment_batch_contract(env, head):
    from teb_vae.classifier.sources import read_rows

    loss = "ce" if head == "multiclass" else "bce"  # the head's loss family (config check)
    unit = _unit(env, "classifier.model.scope=segment", f"classifier.labels.head={head}",
                 f"classifier.train.loss.name={loss}")
    batch = next(iter(_loader(unit, "train", True)))
    b, t = len(batch["row"]), unit.store.step_mask.shape[1]
    assert set(batch) == SEGMENT_KEYS
    assert batch["x"].shape == (b, t, unit.n_values) and batch["x"].dtype == torch.float32
    assert batch["step_mask"].shape == (b, t) and batch["step_mask"].dtype == torch.bool
    assert batch["ctx"].shape == (b, unit.n_ctx) and batch["ctx"].dtype == torch.float32
    assert batch["y"].dtype == (torch.float32 if head == "binary" else torch.int64)
    assert all(batch[k].dtype == torch.int64 and batch[k].shape == (b,) for k in ("y3", "guid", "row"))
    assert batch["w"].dtype == torch.float32 and batch["w"].shape == (b,)

    frame = unit.frames["train"].iloc[batch["row"].numpy()]
    assert (batch["x"][~batch["step_mask"]] == 0).all()
    cached = read_rows(env[2]["cache_dir"], frame["row"])
    expected = np.where(cached.step_mask.numpy()[..., None], unit.scaler.apply(cached.values.numpy()), 0.0)
    np.testing.assert_allclose(batch["x"].numpy(), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(batch["y"].numpy(), frame["y"].to_numpy(float))
    np.testing.assert_array_equal(batch["y3"].numpy(), frame["class_code"].to_numpy(int) - 1)
    ctx = pd.DataFrame(batch["ctx"].numpy(), columns=BASE_CTX)
    np.testing.assert_array_equal(ctx["tlo_missing"], frame["tlo_end_s"].isna().to_numpy(float))
    np.testing.assert_array_equal(ctx["valid_frac"], frame["valid_frac"].to_numpy(np.float32))
    for stage in ("straddle", "second"):  # first and unknown: all zero (§7.1)
        np.testing.assert_array_equal(ctx[f"stage_{stage}"], (frame["stage"] == stage).to_numpy(float))


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


def test_attention_cues_are_scaled_and_keyed_apart(env, tmp_path):
    source = env[2]
    shutil.copytree(source["cache_dir"], tmp_path / "cache")
    with h5py.File(tmp_path / "cache" / "features.h5", "a") as h5:  # attn := the first two value channels
        h5["attn"] = h5["values"][()][..., :2]
        h5.attrs["channels"] = json.dumps(json.loads(h5.attrs["channels"]) + ["cue0", "cue1"])
    for scope in ("segment", "sequence"):
        unit = _unit(env, f"classifier.model.scope={scope}", cache={**source, "cache_dir": str(tmp_path / "cache")})
        assert unit.attn_channels == ("cue0", "cue1") and unit.n_values == 153
        batch = next(iter(_loader(unit, "val", False)))
        assert batch["attn"].shape == batch["x"].shape[:-1] + (2,) and batch["attn"].dtype == torch.float32
        assert torch.equal(batch["attn"], batch["x"][..., :2])  # same data -> same train scaling


def test_w_is_guid_normalised_for_segments_and_raw_for_sequences(env):
    from teb_vae.classifier.data import GuidDataset, SegmentDataset

    unit = _unit(env, "classifier.model.scope=segment")
    ds = SegmentDataset(unit, "train")
    np.testing.assert_allclose(pd.Series(ds.w).groupby(ds.codes).sum(), 1.0, rtol=1e-6)
    omega = unit.frames["train"]["label_weight"].to_numpy()
    assert (omega < 1).any()  # horizon_decay down-weights early positive segments: the test is not trivial
    zeroed = unit.frames["train"].assign(label_weight=np.where(ds.codes == 0, 0.0, omega))
    ds0 = SegmentDataset(dataclasses.replace(unit, frames={**unit.frames, "train": zeroed}), "train")
    assert (ds0.w[ds.codes == 0] == 0).all() and np.isfinite(ds0.w).all()  # Σω = 0: no segment term, no NaN

    seq = GuidDataset(_unit(env), "train")
    for i in range(len(seq)):
        item = seq[i]
        np.testing.assert_array_equal(item["w"], omega[item["row"]].astype(np.float32))


def test_k_warm_drops_early_positions_from_the_position_term_only(env):
    """§6.5/§10.3: ``k_warm`` (auto: 3 under ``propagate``) zeroes positions ``< k`` of the per-position weight
    ``w_pos`` only; the segment-local term keeps ω (``w``), so its loss ignores ``k_warm``."""
    from teb_vae.classifier.data import GuidDataset
    from teb_vae.classifier.losses import compute_loss

    unit = _unit(env, "classifier.labels.strategy=propagate", "classifier.labels.aux_3class_weight=0")
    seq = GuidDataset(unit, "train")
    item = max((seq[i] for i in range(len(seq))), key=lambda it: len(it["row"]))
    pos = np.arange(len(item["row"]))
    assert len(pos) > 3 and (item["w"][:3] > 0).all()  # the cohort's ω carries no k_warm
    np.testing.assert_array_equal(item["w_pos"], np.where(pos < 3, 0.0, item["w"]))

    g = torch.Generator().manual_seed(0)
    n = len(pos)
    out = {"pos": torch.randn(1, n, 1, generator=g), "seg": torch.randn(1, n, 1, generator=g)}
    batch = {"y": torch.ones(1), "y3": torch.ones(1, dtype=torch.long), "seg_mask": torch.ones(1, n, dtype=torch.bool),
             "w": torch.from_numpy(item["w"])[None]}
    kw = dict(scope="sequence", labels_cfg=unit.cfg.labels, train_cfg=unit.cfg.train)
    warm = compute_loss(out, batch | {"w_pos": torch.from_numpy(item["w_pos"])[None]}, **kw)[1]
    cold = compute_loss(out, batch, **kw)[1]  # no w_pos: every position weighs ω
    assert torch.equal(warm["loss_segment"], cold["loss_segment"])
    assert not torch.equal(warm["loss_positions"], cold["loss_positions"])


def test_segment_dropout_recomputes_relative_time_train_only(env):
    unit = _unit(env, "classifier.train.segment_dropout=0.5", "classifier.context.elapsed.enabled=true")
    cols, t_end = unit.context_columns, unit.frames["train"]["t_end_s"].to_numpy()
    loader, dropped = _loader(unit, "train", True, seed=3), 0
    for _ in range(2):
        for batch in loader:
            for i in range(len(batch["guid"])):
                g, rows = int(batch["guid"][i]), batch["row"][i][batch["seg_mask"][i]].numpy()
                full = np.arange(loader.dataset.starts[g], loader.dataset.starts[g] + loader.dataset.lengths[g])
                assert rows[-1] == full[-1] and np.isin(rows, full).all()  # the final segment is always kept
                dropped += len(full) - len(rows)
                t = t_end[rows]
                t_h = batch["t_h"][i, :len(rows)].numpy()
                np.testing.assert_allclose(t_h, (t - t[0]) / 3600.0, rtol=1e-6)
                ctx = batch["ctx"][i, :len(rows)].numpy()
                np.testing.assert_allclose(ctx[:, cols.index("delta_t")], _log1p_dt(t), rtol=1e-6)
                np.testing.assert_allclose(ctx[:, cols.index("elapsed")], np.log1p(t_h), rtol=1e-6)
    assert dropped > 0
    for split in ("train", "val"):  # eval loaders never drop
        n = sum(int(b["seg_mask"].sum()) for b in _loader(unit, split, False))
        assert n == len(unit.frames[split])


# ---- samplers ----------------------------------------------------------------------------------
@pytest.mark.parametrize("scope", ["segment", "sequence"])
def test_loaders_are_seeded_and_eval_is_unshuffled(env, scope):
    unit = _unit(env, f"classifier.model.scope={scope}")
    for split in ("val", "test"):
        loader = _loader(unit, split, False)
        first = list(loader)
        assert _same(first, list(loader)) and _same(first, list(_loader(unit, split, False, seed=99)))
        rows = torch.cat([b["row"][b["seg_mask"]] if "seg_mask" in b else b["row"] for b in first]).numpy()
        if scope == "segment":
            assert (rows == np.arange(len(unit.frames[split]))).all()
        else:
            assert sorted(rows) == list(range(len(unit.frames[split])))
            assert all((np.diff(b["guid"].numpy()) > 0).all() for b in first)  # frame order within a bucket
    train = _loader(unit, "train", True, seed=5)
    e0, e1 = list(train), list(train)
    again = _loader(unit, "train", True, seed=5)
    assert _same(list(again), e0) and _same(list(again), e1) and not _same(e0, e1)
    train.batch_sampler.set_epoch(0)
    assert _same(list(train), e0) and not _same(list(_loader(unit, "train", True, seed=6)), e0)
    assert train.batch_sampler.sampler is train.batch_sampler  # Lightning's _set_sampler_epoch reaches it


def test_store_is_memory_mapped_not_loaded(env):
    """The cache is memory-mapped (a real 10-fold cache exceeds a workstation's RAM), with the stored values."""
    from teb_vae.classifier.data import load_store

    path = Path(env[2]["cache_dir"]) / "features.h5"
    store = load_store(str(path.parent))
    with h5py.File(path, "r") as h5:
        for name in ("values", "step_mask"):
            assert isinstance(getattr(store, name), np.memmap), name
            assert np.array_equal(getattr(store, name), h5[name][()])


def test_workers_and_lightning_epochs_do_not_change_batches(env):
    from lightning.fabric.utilities.data import _set_sampler_epoch

    single, unit = _loader(_unit(env), "train", True, seed=4), _unit(env, "classifier.run.num_workers=2")
    forked = _loader(unit, "train", True, seed=4)
    assert forked.persistent_workers and not any(_loader(unit, s, False).persistent_workers for s in ("train", "val"))
    for epoch in (3, 4):
        for loader in (single, forked):
            _set_sampler_epoch(loader, epoch)  # what Lightning's fit loop calls before each epoch
            assert loader.batch_sampler.epoch == epoch
        assert _same(list(single), list(forked))


def test_online_reader_is_fork_safe(env):
    """Online loaders with workers (default ``run.num_workers: 4``) read the raw rows through handles of their own
    process: a forked worker swaps the inherited datasets for handle-free copies, and 2 workers give the 0-worker
    batches even after the parent process has opened (and read through) every shard."""
    from teb_vae.classifier.cohort import fold_shards
    from teb_vae.classifier.data import OnlineReader, make_loader
    from teb_vae.classifier.sources import Hdf5Source

    c = _cfg(env)
    reader = OnlineReader(Hdf5Source(c.source, fold_shards(c.data, 1)["train"][0]), _unit(env))

    def batches(workers):
        unit = _unit(env, f"classifier.run.num_workers={workers}")
        return [{k: v for k, v in b.items() if k != "vae"} | {f"vae.{k}": v for k, v in b["vae"].items()
                                                                if torch.is_tensor(v)}
                for b in make_loader(unit, "val", train=False, seed=0, reader=reader)]

    single = batches(0)  # reads in this process: its handles are now open
    assert single and "vae.weight" in single[0]
    parent = reader.datasets[(1, "val")]
    assert any(h is not None for h in parent.file_handles)
    assert _same(single, batches(2))

    reader.pid = -1  # as a forked worker sees it: every dataset is replaced by a copy without open handles
    child = reader._dataset(1, "val")
    assert child is not parent and all(h is None for h in child.file_handles) and reader.pid == os.getpid()
    torch.testing.assert_close(child[0]["weight"], parent[0]["weight"])


@pytest.mark.parametrize("scope", ["segment", "sequence"])
def test_class_balanced_sampler_balances_guid_classes(env, scope):
    unit = _unit(env, f"classifier.model.scope={scope}", "classifier.train.sampler=class_balanced")
    train = unit.frames["train"]
    positives = train["guid"][train["y"] == 1].unique()
    imbalanced = train[~train["guid"].isin(positives[:len(positives) * 2 // 3])].reset_index(drop=True)
    unit = dataclasses.replace(unit, frames={**unit.frames, "train": imbalanced})
    natural = dataclasses.replace(unit, cfg=_cfg(env, f"classifier.model.scope={scope}"))

    def positive_share(u):  # 40 epochs of draws; the batch sampler alone decides them
        loader = _loader(u, "train", True, seed=1)
        drawn = [i for _ in range(40) for batch in loader.batch_sampler for i, _ in batch]
        return float(loader.dataset.y[drawn].mean())

    guid_share = imbalanced.groupby("guid")["y"].first().mean()
    assert guid_share < 0.3
    assert abs(positive_share(unit) - 0.5) < 0.05 and positive_share(natural) < 0.35


@pytest.mark.parametrize("bucketed", [False, True])  # segment / sequence scope
@pytest.mark.parametrize("balanced", [False, True])
def test_batch_count_is_fixed_across_epochs(bucketed, balanced):
    """Lightning reads len() once: every epoch yields exactly that many batches. Lengths span the buckets (the
    fixture's GUIDs all fit the first), where re-bucketed balanced draws gave 135..155 batches around 152."""
    from teb_vae.classifier.data import EpochBatches, bucket_sizes

    rng = np.random.default_rng(0)
    y, lengths = (rng.random(200) < 0.25).astype(int), rng.integers(1, 80, 200)
    batches = EpochBatches(y, np.arange(200), batch_size=4, lengths=lengths, shuffle=True, balanced=balanced,
                           buckets=bucket_sizes(4) if bucketed else None, seed=1)
    n = len(batches)
    assert [len(list(batches)) for _ in range(8)] == [n] * 8 and len(batches) == n


def test_class_balanced_with_loss_weighting_warns(env):
    from loguru import logger

    messages = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        _loader(_unit(env, "classifier.train.sampler=class_balanced"), "train", True)
        unit = _unit(env, "classifier.train.sampler=class_balanced", "classifier.train.loss.weighting=inverse")
        _loader(unit, "val", False)  # eval loaders do not sample
        assert not any("corrected twice" in m for m in messages)
        _loader(unit, "train", True)
    finally:
        logger.remove(sink)
    assert any("corrected twice" in m for m in messages)


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
def test_fit_covariates_known_answer_and_train_only():
    from teb_vae.classifier.config import CovariatesCfg
    from teb_vae.classifier.data import fit_covariates

    cfg = CovariatesCfg(static_csv="s.csv", timed_csv="t.csv", max_age_h=2.0, age_feature=True, missing="indicator",
                        dropout_p=0.2, block_dropout_p=0.1,
                        variables=[{"name": "t", "kind": "numeric", "available_at": "prospective"},
                                   {"name": "p", "kind": "categorical", "available_at": "prospective"}])
    train = pd.DataFrame({"split": "train", "cov:t": [1.0, 3.0, np.nan], "cov_age_h:t": [0.0, np.e - 1, np.nan],
                          "cov:p": pd.Series(["b", "a", None], dtype="str")})
    enc = fit_covariates(train, cfg)
    assert enc.columns == ("t", "t:missing", "t:age_h", "p=a", "p=b", "p:missing")
    assert enc.groups.tolist() == [0, 0, 0, 1, 1, 1] and enc.variables[0]["mean"] == 2.0 and enc.variables[0]["std"] == 1
    np.testing.assert_allclose(enc.encode(train), [[-1, 0, 0, 0, 1, 0], [1, 0, 1, 1, 0, 0], [0, 1, 0, 0, 0, 1]])
    np.testing.assert_array_equal(enc.missing_row, [0, 1, 0, 0, 0, 1])
    unseen = train.assign(**{"cov:p": pd.Series(["c", "a", "a"], dtype="str")})
    np.testing.assert_array_equal(enc.encode(unseen)[:, 3:], [[0, 0, 0], [1, 0, 0], [1, 0, 0]])  # unseen: observed, 0s
    bare = fit_covariates(train, cfg.model_copy(update={"missing": "no_indicator", "age_feature": False}))
    assert bare.columns == ("t", "p=a", "p=b") and not bare.missing_row.any()
    with pytest.raises(ValueError, match="L4"):
        fit_covariates(train.assign(split="val"), cfg)
    assert fit_covariates(train, cfg.model_copy(update={"variables": []})) is None


@pytest.mark.parametrize("scope", ["segment", "sequence"])
def test_covariate_batches_dropout_is_train_only_and_seeded(cov_env, scope):
    from teb_vae.classifier.data import covariates_off

    unit = _unit(cov_env, f"classifier.model.scope={scope}", "classifier.train.segment_dropout=0",
                 "classifier.context.covariates.dropout_p=0.5", "classifier.context.covariates.block_dropout_p=0.2")
    enc, seq = unit.covariates, scope == "sequence"
    assert unit.n_cov == len(enc.columns) == 7  # parity: 0 | 1 | 2+ | missing; temp_c: value | missing | age

    def by_row(loader):  # the cov rows of one epoch, in frame order
        rows, covs = [], []
        for batch in loader:
            real = batch["seg_mask"] if seq else torch.ones_like(batch["row"], dtype=torch.bool)
            assert batch["cov"].shape == (*real.shape, unit.n_cov) and batch["cov"].dtype == torch.float32
            assert (batch["cov"][~real] == 0).all()  # padding
            rows.append(batch["row"][real].numpy())
            covs.append(batch["cov"][real].numpy())
        rows, covs = np.concatenate(rows), np.concatenate(covs)
        assert sorted(rows) == list(range(len(rows)))
        return covs[np.argsort(rows)]

    for split in ("train", "val", "test"):  # eval loaders: the plain train-fitted encoding
        np.testing.assert_array_equal(by_row(_loader(unit, split, False)), enc.encode(unit.frames[split]))
    plain, train = enc.encode(unit.frames["train"]), _loader(unit, "train", True, seed=2)
    e0, e1 = by_row(train), by_row(train)
    assert not np.array_equal(e0, plain) and not np.array_equal(e0, e1)  # dropped, and anew each epoch
    np.testing.assert_array_equal(by_row(_loader(unit, "train", True, seed=2)), e0)  # seeded
    moved = e0 != plain
    assert (e0[moved] == np.broadcast_to(enc.missing_row, e0.shape)[moved]).all()  # dropped = set missing
    if seq:  # one draw per GUID: temp_c is dropped at all or none of a GUID's positions where it was observed
        g = enc.groups == 1
        observed = (plain[:, g] != enc.missing_row[g]).any(1)
        dropped = pd.Series((e0[:, g] == enc.missing_row[g]).all(1)[observed])
        assert dropped.groupby(unit.frames["train"]["guid"].to_numpy()[observed]).nunique().le(1).all()
        assert dropped.any() and not dropped.all()
    off = covariates_off(unit)
    assert (by_row(_loader(off, "val", False)) == enc.missing_row).all() and off.covariates is enc
    none = _unit(cov_env, f"classifier.model.scope={scope}", "classifier.context.covariates.dropout_p=0",
                 "classifier.context.covariates.block_dropout_p=0", "classifier.train.segment_dropout=0")
    np.testing.assert_array_equal(by_row(_loader(none, "train", True, seed=2)), plain)


def test_without_indicators_drops_the_missing_flags(cov_env):
    from teb_vae.classifier.data import build_unit, without_indicators

    unit = build_unit(without_indicators(_cfg(cov_env)), cov_env[1], cov_env[2], 1)
    assert "tlo_missing" not in unit.context_columns and unit.n_ctx == 6
    assert unit.covariates.columns == ("parity=0", "parity=1", "parity=2+", "temp_c", "temp_c:age_h")


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
