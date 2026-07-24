r"""The committed shard really produces what the trimmed-grid geometry assumes.

This is the only place the geometry meets the loader. Everything between the HDF5 file and the
model -- the channel-first on-disk layout, the per-sample transpose, the symmetric trim, the
stats-file normalization -- fails silently: a stats-schema mismatch disables normalization with
a warning and hands back correctly-shaped, wrongly-scaled tensors. For this model the stakes are
higher than for its feature-target sibling, because the *raw* signal is the reconstruction
target: without ``'fhr'`` in ``normalize_fields`` the target arrives in bpm, the Gaussian NLL
is computed against ~140-scale values under a z-scale variance model -- meaningless -- and
nothing raises anywhere.

The loader configuration used here is this module's own ``configs/tiny.yaml``, resolved through
its ``base:`` chain -- the exact configuration the smoke fit trains under, so the contract
asserted here is the contract the model actually sees.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import pytest
import torch

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry
from teb_vae.lag_attn_rws.nets.raw_targets import build_future_target
from train.data_module import GraphDataModule

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TINY_CONFIG = _REPO_ROOT / "teb_vae" / "lag_attn_rws" / "configs" / "tiny.yaml"

#: The production geometry the trimmed loader must realise: 5280 - 2*240 raw samples,
#: 330 - 2*15 decimated steps.
_GEOMETRY = TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=30, warmup=30)


@pytest.fixture(scope="module")
def config():
    """The resolved tiny config, with dataset paths made absolute.

    The config's paths are repo-root-relative because entry points run from the repo root;
    pytest may not.
    """
    resolved = load_config(str(_TINY_CONFIG))
    dataset = resolved["dataset_config"]
    for key in ("vae_train_datasets", "vae_test_datasets"):
        dataset[key] = [str(_REPO_ROOT / path) for path in dataset[key]]
    dataset["stat_path"] = str(_REPO_ROOT / dataset["stat_path"])
    return resolved


@pytest.fixture(scope="module")
def batch(config):
    return next(iter(GraphDataModule(config).train_dataloader()))


def test_the_raw_target_matches_the_geometry(batch):
    assert batch.fhr.shape[-1] == _GEOMETRY.raw_len
    assert batch.weight.shape[-1] == _GEOMETRY.t
    assert batch.fhr.shape[-1] == _GEOMETRY.decimation * batch.weight.shape[-1]


def test_the_future_gather_binds_to_the_real_raw_signal(batch):
    """Anchor 0's window on real data is exactly ``fhr[:, 16:496]`` -- the trimmed-grid
    formula, one minute earlier than the untrimmed grid's ``[256, 736)``."""
    target = build_future_target(batch.fhr, _GEOMETRY)
    assert target.shape == (batch.fhr.shape[0], 270, 30, 16)
    assert torch.equal(target[:, 0].reshape(batch.fhr.shape[0], -1), batch.fhr[:, 16:496])


def test_the_feature_blocks_carry_the_declared_widths(batch):
    assert batch.fhr_st.shape == (2, _GEOMETRY.t, 43)
    assert batch.fhr_ph.shape == (2, _GEOMETRY.t, 66)
    assert batch.up_st.shape == (2, _GEOMETRY.t, 43)
    assert batch.up_ph.shape == (2, _GEOMETRY.t, 15)


def test_the_cross_channel_block_is_never_loaded(config):
    """``fhr_up_ph`` mixes both signals in one coefficient; loading it would destroy the
    separation between the target-only prior and the source-conditioned posterior."""
    load_fields = config["dataset_config"]["dataloader_config"]["dataset_kwargs"]["load_fields"]
    assert "fhr_up_ph" not in load_fields
    for field in ("fhr", "fhr_st", "fhr_ph", "up_st", "up_ph", "weight"):
        assert field in load_fields


def test_the_raw_target_is_normalized(config, batch):
    """``'fhr'`` must sit in both ``load_fields`` and ``normalize_fields``. Without the latter
    the raw target arrives unnormalized (~140 bpm scale), the Gaussian NLL is meaningless, and
    nothing raises -- the numeric check is what makes this operational."""
    dataloader = config["dataset_config"]["dataloader_config"]
    assert "fhr" in dataloader["dataset_kwargs"]["load_fields"]
    assert "fhr" in dataloader["normalize_fields"]
    assert abs(float(batch.fhr.mean())) < 5.0, (
        "fhr looks unnormalized -- the stats file was probably rejected and normalization "
        "silently disabled"
    )


def test_the_stats_file_trim_matches_the_loader_trim(config):
    """A mismatch only warns in the loader, so it is pinned here instead."""
    loader_trim = config["dataset_config"]["dataloader_config"]["dataset_kwargs"][
        "trim_minutes"
    ]
    with h5py.File(config["dataset_config"]["stat_path"], "r") as stats:
        stats_trim = float(stats.attrs.get("trim_minutes", -1.0))
    assert stats_trim == float(loader_trim)


def test_the_weight_field_is_binary_on_the_committed_fixture(batch):
    """The >= 1.0 validity threshold and > 0 agree on binary weights; this pins that the
    fixture cannot distinguish them, which is why the threshold decision rests on the shard
    writer's construction (see ``nets/raw_masks.py``) rather than on this data."""
    assert set(torch.unique(batch.weight).tolist()) <= {0.0, 1.0}
