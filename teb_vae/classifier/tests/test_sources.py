"""T-S1..T-S4 (SPEC §15): feature sources, feature cache and scaler, on the §15 fixture tree.

The VAE is a tiny ``lag_attn_transformer_cfs`` model built the way that package's tests build theirs
(``make_task`` on ``TINY_KWARGS``), widened to the fixture's geometry (T = 300, c_y = 102, c_u = 51)
and saved with its ``resolved_config.yaml`` into tmp_path. Its posterior delta heads are perturbed,
because they are zero-initialised and ``delta_mu`` / the KL would otherwise be exactly zero.
"""
from __future__ import annotations

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
import yaml

from hdf5_dataset.hdf5_dataset import attribute_dict_collate
from teb_vae.classifier import cohort, sources
from teb_vae.classifier.config import REPO_ROOT, load
from teb_vae.classifier.tests.conftest import BINS, D_Z, HEADS, SMOKE_CONFIG, save_checkpoint as _save_checkpoint

FAMILIES = ["lag_attn_rws", "lag_attn_fs", "lag_attn_cfs", "lag_attn_crws",
            "lag_attn_transformer_rws", "lag_attn_transformer_fs", "lag_attn_transformer_cfs",
            "lag_attn_transformer_crws", "lag_attn_transformer_e2e", "lag_slot_transformer_cfs"]


def _cfg(smoke_overrides, *extra):
    return load(SMOKE_CONFIG, list(smoke_overrides) + list(extra)).classifier


def _batch(source, paths, n=8):
    dataset = source.dataset(paths)
    return attribute_dict_collate([dataset[i] for i in range(n)]), dataset


def _with_stat_path(checkpoint: Path, root: Path, stat_path: str) -> Path:
    """A copy of ``checkpoint``'s run under ``root`` whose resolved config names ``stat_path``."""
    shutil.copytree(checkpoint.parent, root / "model_checkpoints")
    config_path = root / "model_checkpoints" / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["dataset_config"]["stat_path"] = stat_path
    config_path.write_text(yaml.safe_dump(config))
    return root / "model_checkpoints" / "best.ckpt"


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
    assert feats.values.shape == (n, 300, 3 * D_Z + HEADS * (2 + len(BINS)))
    assert feats.attn.shape == (n, 300, 1) and feats.channels[-1] == "kld_per_t"
    assert len(feats.channels) == feats.values.shape[-1] + 1
    assert feats.channels[3 * D_Z:3 * D_Z + 3] == (
        "attn_summary[h0.lag]", "attn_summary[h0.entropy]", "attn_summary[h0.bin0]")
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


def test_vae_source_slot_family_scatters_anchors(vae_overrides, fold1, fixture_tree, tmp_path):
    from teb_vae.lag_slot_transformer_cfs.tests.conftest import tiny_model_kwargs

    kwargs = tiny_model_kwargs(sequence_length=300, c_y=102, c_u=51, warmup_period=134, horizon=30,
                               anchor_stride=30, target_keep_index=None, target_warmup_steps=None,
                               source_keep_index=None, source_warmup_steps=None)
    torch.manual_seed(0)
    model = sources.trainer_class("lag_slot_transformer_cfs").MODEL_CLS(**kwargs)
    checkpoint = _save_checkpoint(tmp_path, model, kwargs, fixture_tree["stats_path"])
    source = sources.VaeSource(_cfg(
        vae_overrides, "classifier.source.vae.package=lag_slot_transformer_cfs",
        f"classifier.source.vae.checkpoint={checkpoint}",
        "classifier.source.vae.keys=[{name: mu_prior}, {name: target_state}, "
        "{name: kld_per_t, role: attention}]").source)
    batch, _ = _batch(source, fold1["train"], n=4)
    feats = source(batch)
    mask, t = feats.step_mask, torch.arange(300)
    assert torch.equal(mask, (batch["weight"] > 0) & (t >= 134) & (t < 270))  # anchor_valid on T
    with torch.no_grad():
        out = source.model(*source.task._build_forward_inputs(batch))
    d_z, on_anchor = out["mu_prior"].shape[-1], mask[:, 134:270]  # dense anchors 134..269
    anchored = feats.values[:, 134:270]
    torch.testing.assert_close(anchored[..., :d_z][on_anchor], out["mu_prior"][on_anchor])
    torch.testing.assert_close(feats.attn[:, 134:270, 0][on_anchor],
                               out["kld_per_anchor"][on_anchor])
    torch.testing.assert_close(feats.values[..., d_z:][mask], out["target_state"][mask])  # T axis


def test_sample_z_draws_the_values_but_keeps_the_kl_keys_on_the_means(vae_overrides, fold1):
    """``sample_z`` (``frozen_online`` augmentation) draws the ``mu_post`` / ``delta_mu`` values only: ``kld_per_dim``
    and ``kld_excess`` read the posterior means, drawn or not."""
    source = sources.VaeSource(_cfg(vae_overrides, "classifier.source.vae.keys=[{name: delta_mu}, {name: kld_per_dim}, "
                                                   "{name: kld_excess, role: attention}]").source, sample_z=True)
    batch, _ = _batch(source, fold1["train"], n=4)
    mean, drawn = source(batch), source(batch, sample_z=True)
    mask, d = mean.step_mask, len([c for c in mean.channels if c.startswith("delta_mu")])
    assert not torch.allclose(drawn.values[..., :d][mask], mean.values[..., :d][mask])  # the draw reaches delta_mu
    torch.testing.assert_close(drawn.values[..., d:], mean.values[..., d:])  # kld_per_dim on the means
    torch.testing.assert_close(drawn.attn, mean.attn)  # kld_excess too


def test_vae_source_kld_excess_is_kld_minus_the_source_null_kld(vae_overrides, fold1):
    from teb_vae.lag_attn_rws.nets.controls import source_null_forward_outputs

    source = sources.VaeSource(_cfg(vae_overrides, "classifier.source.vae.keys=[{name: kld_per_t}, "
                                                   "{name: kld_excess, role: attention}]").source)
    batch, _ = _batch(source, fold1["train"], n=4)
    feats = source(batch)
    mask = feats.step_mask
    with torch.no_grad():
        inputs = source.task._build_forward_inputs(batch)
        out = source.model(*inputs)
        null = source_null_forward_outputs(source.model, out, inputs[2])
        kl = {name: source.model.kld_tensor(mu_prior=out["mu_prior"], logvar_prior=out["logvar_prior"],
                                            mu_post=o["mu_post"], logvar_post=o["logvar_post"]).sum(-1)
              for name, o in (("matched", out), ("null", null))}
    torch.testing.assert_close(kl["matched"], out["kld_per_t"])  # kld_per_t is the d_z-summed KL
    assert feats.channels == ("kld_per_t", "kld_excess")
    torch.testing.assert_close(feats.attn[..., 0][mask], (out["kld_per_t"] - kl["null"])[mask])
    assert (feats.attn[~mask] == 0).all() and feats.attn[..., 0][mask].abs().max() > 0


def test_attn_summary_gradient_is_finite_at_exact_zeros(vae_overrides, fold1):
    """entmax lag attention has exact zeros; ``xlogy``'s backward is NaN there, so a trainable encoder upstream of the
    attention got NaN gradients from a finite loss. The entropy value is unchanged."""
    alpha = torch.tensor([[0.0, 0.7, 0.3, 0.0], [0.25, 0.25, 0.25, 0.25]], requires_grad=True)
    out = sources.attn_summary(alpha, [(0, 1), (2, 3)])
    torch.testing.assert_close(out[..., 1], -torch.special.xlogy(alpha, alpha).sum(-1).detach())
    out.sum().backward()
    assert torch.isfinite(alpha.grad).all()

    source = sources.VaeSource(_cfg(vae_overrides).source, unfreeze=("target_encoder",))  # upstream of the attention
    batch, _ = _batch(source, fold1["train"], n=4)
    source.model.train()
    features = source(batch)
    assert (features.values[..., [i for i, c in enumerate(features.channels) if c.startswith("attn_summary")]]
            .requires_grad)
    features.values.pow(2).sum().backward()
    grads = [p.grad for _, p in source.allowlist() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


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


def test_vae_source_refusals(vae_overrides, fixture_tree, tmp_path):
    from teb_vae.lag_slot_transformer_cfs.tests.conftest import tiny_model_kwargs

    kwargs = tiny_model_kwargs(sequence_length=300, c_y=102, c_u=51, warmup_period=134, horizon=30,
                               anchor_stride=30, target_keep_index=None, target_warmup_steps=None,
                               source_keep_index=None, source_warmup_steps=None)
    slot = sources.trainer_class("lag_slot_transformer_cfs").MODEL_CLS(**kwargs)
    checkpoint = _save_checkpoint(tmp_path, slot, kwargs, fixture_tree["stats_path"])
    with pytest.raises(NotImplementedError, match="kld_excess needs a source-null"):
        sources.VaeSource(_cfg(vae_overrides, "classifier.source.vae.package=lag_slot_transformer_cfs",
                               f"classifier.source.vae.checkpoint={checkpoint}",
                               "classifier.source.vae.keys=[{name: kld_excess}]").source)
    with pytest.raises(ValueError, match="match no parameter"):
        sources.VaeSource(_cfg(vae_overrides).source, unfreeze=["posterior_head", "no_such_module"])
    with pytest.raises(NotImplementedError, match="lag_attn"):
        sources.VaeSource(_cfg(vae_overrides, "classifier.source.vae.package=lag_attn").source)
    with pytest.raises(ValueError, match="L13"):  # classifier trim != checkpoint loader/stats trim
        sources.VaeSource(_cfg(vae_overrides, "classifier.source.hdf5.trim_minutes=0.5").source)


def test_vae_source_stats_path_resolves_like_its_fingerprint(vae_overrides, vae_checkpoint, fold1,
                                                             fixture_tree, tmp_path, monkeypatch):
    reference = sources.VaeSource(_cfg(vae_overrides).source)
    # Repo-relative, as checkpoints store it; run from a cwd deeper than REPO_ROOT, so the same
    # relative path cannot land on the file by clamping at "/".
    relative = os.path.relpath(fixture_tree["stats_path"], REPO_ROOT)
    checkpoint = _with_stat_path(vae_checkpoint, tmp_path / "relative", relative)
    cwd = tmp_path.joinpath(*["d"] * len(REPO_ROOT.parts))
    cwd.mkdir(parents=True)
    monkeypatch.chdir(cwd)
    assert not Path(relative).exists()
    moved = sources.VaeSource(
        _cfg(vae_overrides, f"classifier.source.vae.checkpoint={checkpoint}").source)
    assert moved.dataset_kwargs["stats_path"] == moved.fingerprint["stats_path"]
    torch.testing.assert_close(moved(_batch(moved, fold1["train"])[0]).values,
                               reference(_batch(reference, fold1["train"])[0]).values)

    missing = _with_stat_path(vae_checkpoint, tmp_path / "missing", str(tmp_path / "none.hdf5"))
    with pytest.raises(FileNotFoundError):
        sources.VaeSource(_cfg(vae_overrides, f"classifier.source.vae.checkpoint={missing}").source)
    broken = tmp_path / "broken_stats.hdf5"  # trim intact, a field unloadable: loader disables
    shutil.copy(fixture_tree["stats_path"], broken)
    with h5py.File(broken, "a") as handle:
        del handle["fhr"].attrs["shape"]
    disabled = sources.VaeSource(_cfg(vae_overrides, "classifier.source.vae.checkpoint="
                                      f"{_with_stat_path(vae_checkpoint, tmp_path / 'b', str(broken))}"
                                      ).source)
    with pytest.raises(ValueError, match="did not load"):
        disabled.dataset(fold1["train"])


@pytest.mark.parametrize("package", FAMILIES)
def test_every_family_resolves_its_binding(package):
    trainer = sources.trainer_class(package)
    assert hasattr(trainer.MODEL_CLS, "kld_tensor")
    assert hasattr(trainer.TASK_CLS, "_build_forward_inputs")


# ---- T-S2 Hdf5Source ---------------------------------------------------------------------------
def test_hdf5_source_channel_counts_and_cold_cells(smoke_overrides, fold1):
    cfg = _cfg(smoke_overrides, "classifier.source.hdf5.fields=[fhr_st, up_ph, raw_fhr]",
               "classifier.source.hdf5.min_step=null")
    source = sources.Hdf5Source(cfg.source, fold1["train"][0])
    with h5py.File(fold1["train"][0], "r") as handle:
        stored = {"fhr_st": handle["fhr_st"].shape[1], "up_ph": handle["up_ph"].shape[1]}
    assert source.widths == {**stored, "raw_fhr": 16}
    assert source.fingerprint["widths"] == source.widths
    batch, dataset = _batch(source, fold1["train"])
    feats = source(batch)
    assert feats.values.shape == (8, 300, 36 + 15 + 16) and feats.attn is None

    warmup = np.r_[dataset.causal_warmup_steps["fhr_st"], dataset.causal_warmup_steps["up_ph"]]
    for channel, cold in enumerate(warmup):
        assert (feats.values[:, :cold, channel] == 0).all()
    assert torch.equal(feats.step_mask, batch["weight"] > 0)  # raw channels are never cold
    warm = feats.step_mask[:, :, None] & (torch.arange(300)[:, None] >= torch.from_numpy(warmup))
    assert (feats.values[..., :51][warm] != 0).float().mean() > 0.99
    torch.testing.assert_close(feats.values[..., 51:][feats.step_mask],
                               batch["fhr"].reshape(8, 300, 16)[feats.step_mask])

    up_ph = sources.Hdf5Source(_cfg(smoke_overrides, "classifier.source.hdf5.fields=[up_ph]",
                                    "classifier.source.hdf5.min_step=null").source,
                               fold1["train"][0])
    first_warm = int(dataset.causal_warmup_steps["up_ph"].min())
    assert first_warm > 0  # every up_ph channel is cold before it: the step is masked
    expected = (batch["weight"] > 0) & (torch.arange(300) >= first_warm)
    assert torch.equal(up_ph(batch).step_mask, expected)

    auto = sources.Hdf5Source(_cfg(smoke_overrides).source, fold1["train"][0])  # all 4 blocks, auto
    assert auto.min_step == max(int(w.max()) for w in dataset.causal_warmup_steps.values())
    mask = auto(_batch(auto, fold1["train"])[0]).step_mask
    assert not mask[:, :auto.min_step].any() and mask[:, auto.min_step:].any()


def test_time_pool_is_a_masked_mean():
    values = torch.tensor([[[1.0], [100.0], [3.0], [5.0]]])
    feats = sources._finish(values, torch.tensor([[True, False, False, False]]), None, ["x"], 2)
    assert feats.values.flatten().tolist() == [1.0, 0.0]
    assert feats.step_mask.tolist() == [[True, False]]


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


def test_cache_key_follows_code_content(fixture_segments, smoke_overrides, vae_overrides,
                                        tmp_path, monkeypatch):
    on_path = tmp_path / "on_path.py"  # stands in for a file on the extraction path
    on_path.write_text("SCALE = 1\n")
    monkeypatch.setattr(sources, "CODE_FILES", (*sources.CODE_FILES, on_path))
    cfg = _cfg(smoke_overrides, "classifier.source.hdf5.fields=[up_ph]",
               "classifier.source.hdf5.min_step=null")
    shards = {(k, s): p for k in (1, 2, 3) for s, p in cohort.fold_shards(cfg.data, k).items()}
    segments = fixture_segments[0].loc[lambda d: ~d["excluded"]].head(20)

    def extract():
        source = sources.Hdf5Source(cfg.source, shards[(1, "train")][0])
        return sources.extract(source, segments, shards, cache_root=tmp_path / "cache")

    first = extract()
    assert extract()["fingerprint_hash"] == first["fingerprint_hash"]  # untouched tree: reused
    real = sources.software_record
    monkeypatch.setattr(sources, "software_record", lambda: real() | {"revision": "0" * 40})  # a new commit
    moved = extract()
    assert moved["fingerprint_hash"] == first["fingerprint_hash"] and moved["n_extracted"] == 0  # the SHA is info
    assert moved["fingerprint"]["code_revision"] == "0" * 40
    on_path.write_text("SCALE = 2\n")  # an uncommitted edit: git HEAD is unchanged
    edited = extract()
    assert edited["fingerprint_hash"] != first["fingerprint_hash"]
    assert edited["n_extracted"] == edited["n_unique"] > 0

    sources.VaeSource(_cfg(vae_overrides).source)  # the family and its siblings are now imported
    files = {str(p.relative_to(REPO_ROOT)) for p in sources._vae_code_files()}
    assert {"teb_vae/lag_attn_transformer_cfs/nets/model.py", "teb_vae/lag_attn_cfs/trainer.py",
            "teb_vae/lag_attn_transformer_rws/trainer.py"} <= files
    assert not any("/tests/" in f or f.startswith("teb_vae/classifier/") for f in files)


def test_code_digest_ignores_the_checkout_location(tmp_path, monkeypatch):
    """Two clones of the same tree digest alike; a changed file does not."""
    for clone in ("a", "b"):
        (tmp_path / clone / "pkg").mkdir(parents=True)
        (tmp_path / clone / "pkg" / "x.py").write_text("SCALE = 1\n")
    digests = []
    for clone in ("a", "b"):
        monkeypatch.setattr(sources, "REPO_ROOT", tmp_path / clone)
        digests.append(sources._code_sha256([tmp_path / clone / "pkg" / "x.py"]))
    assert digests[0] == digests[1]
    (tmp_path / "b" / "pkg" / "x.py").write_text("SCALE = 2\n")
    assert sources._code_sha256([tmp_path / "b" / "pkg" / "x.py"]) != digests[1]


def test_extract_stage_through_run(smoke_overrides, tmp_path):
    from loguru import logger

    from teb_vae.classifier import run

    overrides = list(smoke_overrides) + [f"classifier.source.cache_root={tmp_path / 'cache'}"]
    try:
        assert list(run.STAGES)[:2] == ["cohort", "extract"]
        run_dir = run.main(config=str(SMOKE_CONFIG), stage="cohort", overrides=overrides,
                           run_dir=str(tmp_path / "run"))
        run.main(config=str(SMOKE_CONFIG), stage="extract", overrides=overrides,
                 run_dir=str(run_dir))
        assert json.loads((run_dir / "stage_state.json").read_text())["extract"]["status"] == "done"
        record = json.loads((run_dir / "manifest.json").read_text())["source"]
        index = sources.open_cache(record["cache_dir"], record["fingerprint"])
        segments = pd.read_parquet(run_dir / "cohort" / "segments.parquet")
        assert record["n_rows"] == len(index) == int((~segments["excluded"]).sum())
        assert record["fingerprint"]["kind"] == "hdf5" and record["bytes"] > 0
        assert Path(record["cache_dir"]).parent == tmp_path / "cache"
    finally:  # run.main replaced loguru's sinks with files under tmp_path
        logger.remove()
        logger.add(sys.stderr)


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


def test_scaler_json_round_trip(tmp_path):
    values = np.random.default_rng(0).normal(size=(6, 5, 2))
    index = pd.DataFrame({"split": "train", "guid": list("aabbcc")})
    scaler = sources.fit_scaler(index, values, np.ones((6, 5), bool), ["p", "q"])
    scaler.save(tmp_path / "scaler.json")
    loaded = sources.Scaler.load(tmp_path / "scaler.json")
    np.testing.assert_allclose(loaded.apply(values), scaler.apply(values))
    assert loaded.channels == ("p", "q") and loaded.record == scaler.record
