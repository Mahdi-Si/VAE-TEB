"""T-S1..T-S4 (SPEC §15): feature sources, feature cache and scaler, on the §15 fixture tree.

The VAE is a tiny ``lag_attn_transformer_cfs`` model built the way that package's tests build theirs
(``make_task`` on ``TINY_KWARGS``), widened to the fixture's geometry (T = 300, c_y = 102, c_u = 51)
and saved with its ``resolved_config.yaml`` into tmp_path. Its posterior delta heads are perturbed,
because they are zero-initialised and ``delta_mu`` / the KL would otherwise be exactly zero.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from hdf5_dataset.hdf5_dataset import attribute_dict_collate
from teb_vae.classifier import cohort, sources
from teb_vae.classifier.config import load
from teb_vae.classifier.tests.conftest import BINS, D_Z, HEADS, SMOKE_CONFIG

def _cfg(smoke_overrides, *extra):
    return load(SMOKE_CONFIG, list(smoke_overrides) + list(extra)).classifier


def _batch(source, paths, n=8):
    dataset = source.dataset(paths)
    return attribute_dict_collate([dataset[i] for i in range(n)]), dataset


@pytest.fixture(scope="module")
def fold1(smoke_config):
    return {split: paths for split, paths in cohort.fold_shards(smoke_config.data, 1).items()}


# ---- T-S1 VaeSource ----------------------------------------------------------------------------
def test_vae_source_shapes_mask_and_derived_keys(vae_overrides, fold1):
    source = sources.VaeSource(_cfg(vae_overrides).source)
    batch, _ = _batch(source, fold1["train"])
    batch["weight"][0, 200:210] = 0
    feats = source(batch)
    n, t = len(batch["guid"]), torch.arange(300)

    assert source.trainable is False and not source.model.training
    assert not feats.values.requires_grad
    assert len(feats.channels) == feats.values.shape[-1] + 1  # one name per value column, then kld_per_t
    mask = feats.step_mask  # warm-up and weight; causal_all keeps [270, 300)
    assert torch.equal(mask, (batch["weight"] > 0) & (t >= 134))
    assert not mask[0, 200:210].any() and mask[:, 270:].all()
    assert (feats.values[~mask] == 0).all() and (feats.attn[~mask] == 0).all()

    with torch.no_grad():
        out = source.model(*source.task._build_forward_inputs(batch))
    torch.testing.assert_close(feats.values[..., :D_Z][mask], out["mu_prior"][mask])
    delta = feats.values[..., D_Z:2 * D_Z][mask]
    torch.testing.assert_close(delta, (out["mu_post"] - out["mu_prior"])[mask])
    assert delta.abs().max() > 0  # the perturbation took
    torch.testing.assert_close(torch.expm1(feats.attn[..., 0][mask]), out["kld_per_t"][mask])
    masses = feats.values[..., 3 * D_Z:].reshape(n, 300, HEADS, 2 + len(BINS))[..., 2:].sum(-1)
    torch.testing.assert_close(masses[mask], torch.ones_like(masses[mask]))

    supervised = sources.VaeSource(
        _cfg(vae_overrides, "classifier.source.vae.step_support=supervised").source)
    assert supervised.ceiling == 270  # T - horizon, from the model built off model_kwargs
    assert torch.equal(supervised(batch).step_mask, (batch["weight"] > 0) & (t >= 134) & (t < 270))


def test_load_task_keeps_every_checkpoint_hyperparameter(vae_checkpoint, tmp_path):
    """The task is rebuilt with every hyperparameter its constructor chain takes (the cfs ``seed`` through
    ``**kwargs``, the rws loss weights), so co-training optimises the pretraining objective (§10.6); never compiled."""
    blob = torch.load(vae_checkpoint, map_location="cpu", weights_only=False)
    blob["hyper_parameters"] |= {"lambda_ms": 0.1, "lambda_deriv": 0.2, "lambda_boundary": 0.05, "seed": 7,
                                 "compile_model": True, "lr_gamma": 0.5}  # lr_gamma: LightningModelBase's, not taken
    torch.save(blob, tmp_path / "best.ckpt")
    task, _ = sources.load_task(tmp_path / "best.ckpt", "lag_attn_transformer_cfs", "cpu")
    hp = task.hparams
    assert (hp["lambda_ms"], hp["lambda_deriv"], hp["lambda_boundary"], hp["seed"]) == (0.1, 0.2, 0.05, 7)
    assert task.model is task.orig_model  # eager
    assert {"seed", "lambda_ms", "kld_beta"} <= sources.task_parameters(type(task))


# ---- T-S2 Hdf5Source ---------------------------------------------------------------------------
# ---- T-S3 cache --------------------------------------------------------------------------------
def test_cache_once_per_segment_aligned_resumable_and_refused(fixture_segments, smoke_overrides,
                                                               tmp_path):
    segments = fixture_segments[0]
    cfg = _cfg(smoke_overrides, "classifier.source.hdf5.fields=[up_ph]",
               "classifier.source.hdf5.min_step=null")
    shards = {(k, s): p for k in (1, 2, 3) for s, p in cohort.fold_shards(cfg.data, k).items()}
    calls = []

    class Counting(sources.Hdf5Source):
        def __call__(self, batch):
            calls.append(len(batch["guid"]))
            return super().__call__(batch)

    source = Counting(cfg.source, shards[(1, "train")][0])
    record = sources.extract(source, segments, shards, cache_root=tmp_path, batch_size=50)
    kept = segments[~segments["excluded"]]
    n_unique = len(kept.drop_duplicates(["guid", "epoch_s"]))
    assert sum(calls) == record["n_unique"] == record["n_extracted"] == n_unique
    assert record["n_rows"] == len(kept) == 3 * n_unique  # every segment is in every fold
    assert record["bytes"] > 0 and record["segments_per_s"] > 0

    index = sources.open_cache(record["cache_dir"], record["fingerprint"])
    assert len(index) == len(kept) and sorted(index["row"].unique()) == list(range(n_unique))
    assert index.groupby(["guid", "epoch_s"])["row"].nunique().eq(1).all()
    later = index[(index["fold"] == 3) & (index["split"] == "val")].head(8)  # mostly read in fold 1
    assert (index[index["row"].isin(later["row"])]["fold"] != 3).any()
    dataset = source.dataset(shards[(3, "val")])
    batch = attribute_dict_collate([dataset[int(i)] for i in later["ds_index"]])
    direct, cached = source(batch), sources.read_rows(record["cache_dir"], later["row"])
    assert cached.channels == direct.channels and torch.equal(cached.step_mask, direct.step_mask)
    torch.testing.assert_close(cached.values, direct.values.half().float())
    assert batch["guid"] == later["guid"].tolist()

    again = sources.extract(source, segments, shards, cache_root=tmp_path)
    assert again["cache_dir"] == record["cache_dir"] and again["n_extracted"] == 0
    with h5py.File(Path(record["cache_dir"]) / "features.h5", "a") as h5:
        h5["done"][5] = False  # an interrupted chunk
    with pytest.raises(ValueError, match="incomplete"):
        sources.open_cache(record["cache_dir"])
    assert sources.extract(source, segments, shards, cache_root=tmp_path)["n_extracted"] == 1

    with pytest.raises(ValueError, match="another fingerprint"):
        sources.open_cache(record["cache_dir"], {**record["fingerprint"], "min_step": 7})
    stored = Path(record["cache_dir"]) / "fingerprint.json"
    stored.write_text(json.dumps({**record["fingerprint"], "trim_minutes": 0.5}))
    with pytest.raises(ValueError, match="another fingerprint"):
        sources.extract(source, segments, shards, cache_root=tmp_path)


# ---- T-S4 scaler -------------------------------------------------------------------------------
def test_scaler_train_only_recording_weighted_floored():
    # GUID a: 3 segments at 0, GUID b: 1 segment at 10 on channel 0 -> recording-weighted mean 5,
    # std 5 (a pooled mean would be 2.5). Channel 1 is constant (dropped), channel 2 barely varies
    # (floored). Invalid steps carry 1e6 and must not count.
    values = np.zeros((4, 3, 3))
    values[3, :, 0] = 10.0
    values[..., 1] = 7.0
    values[..., 2] = 1.0 + 1e-6 * np.arange(3)
    mask = np.array([[True, True, False]] * 4)
    values[~mask] = 1e6
    index = pd.DataFrame({"split": "train", "fold": 1, "guid": ["a", "a", "a", "b"]})
    channels = ["x", "const", "flat"]

    scaler = sources.fit_scaler(index, values, mask, channels)
    assert scaler.center[0] == pytest.approx(5.0) and scaler.scale[0] == pytest.approx(5.0)
    assert scaler.keep.tolist() == [True, False, True] and scaler.record["dropped"] == ["const"]
    floor = max(1e-3, 0.1 * float(np.median([5.0, 5e-7])))
    assert scaler.scale[2] == pytest.approx(floor) and scaler.record["n_at_floor"] == 1
    assert scaler.apply(values).shape == (4, 3, 2)

    with pytest.raises(ValueError, match="L4"):
        sources.fit_scaler(index.assign(split=["train", "train", "val", "train"]), values, mask,
                           channels)
