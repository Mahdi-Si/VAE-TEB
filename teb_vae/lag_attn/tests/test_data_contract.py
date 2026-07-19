r"""The committed shard really produces the batch the model's forward signature expects.

The model takes ``y_st (B,T,43)``, ``y_ph (B,T,66)`` and a source stream assembled from ``up_st``
and ``up_ph``. Between the HDF5 file and those tensors sit a channel-first on-disk layout, a
per-sample transpose, a symmetric trim, and a normalization step that reads a separate stats file --
and almost every failure in that chain is silent. A missing index field yields "No samples match the
specified filters"; a stats-schema mismatch disables normalization with a warning and hands back
correctly-shaped, wrongly-scaled tensors. So the contract is asserted here rather than discovered
three steps into a training run.

Regenerate the fixtures with ``python scripts/make_tiny_shard.py``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn.config import load_config
from train.data_module import GraphDataModule

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TINY_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"

#: What the trim arithmetic must produce from the committed shard: 330 - 2*(240//16) = 300.
_EXPECTED_T = 300
#: 16x decimation: 5280 - 2*240 = 4800.
_EXPECTED_RAW = 4800


@pytest.fixture(scope="module")
def config():
    """The resolved tiny config, with dataset paths made absolute.

    The config's paths are repo-root-relative because entry points run from the repo root; pytest
    may not.
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


def test_the_fixtures_are_committed():
    """A silently-absent shard would make every test below fail with an unhelpful message."""
    fixtures = Path(__file__).resolve().parent / "fixtures"
    assert (fixtures / "tiny_shard.hdf5").is_file()
    assert (fixtures / "tiny_stats.hdf5").is_file()


def test_the_fixtures_are_small_enough_to_live_in_the_repo():
    fixtures = Path(__file__).resolve().parent / "fixtures"
    total = sum((fixtures / name).stat().st_size for name in ("tiny_shard.hdf5", "tiny_stats.hdf5"))
    assert total < 4 * 1024 * 1024, f"fixtures grew to {total / 1e6:.1f} MB"


def test_the_batch_carries_the_model_input_contract(batch):
    """Channel counts are the contract; the task checks them against every batch."""
    assert batch.fhr_st.shape == (2, _EXPECTED_T, 43)
    assert batch.fhr_ph.shape == (2, _EXPECTED_T, 66)
    assert batch.up_st.shape == (2, _EXPECTED_T, 43)
    assert batch.up_ph.shape == (2, _EXPECTED_T, 15)
    assert batch.weight.shape == (2, _EXPECTED_T)


def test_the_feature_fields_arrive_time_major(batch):
    """The on-disk layout is (N, C, T) and the dataset transposes on read.

    A model that permuted again would silently train on a (channels, time) tensor of the right
    rank. The assertion that catches it is that the last axis is the channel count -- and 43 != 300
    is the only reason it can be caught at all.
    """
    assert batch.fhr_st.shape[-1] == 43 != batch.fhr_st.shape[-2]


def test_the_source_stream_concatenates_to_the_configured_width(batch, config):
    """What the task builds and hands to the model as ``u_stream``."""
    u_stream = torch.cat([batch.up_st, batch.up_ph], dim=-1)
    assert u_stream.shape[-1] == config["model_config"]["VAE_model"]["c_u"] == 58


def test_the_target_stream_concatenates_to_the_configured_width(batch, config):
    assert batch.fhr_st.shape[-1] + batch.fhr_ph.shape[-1] == config["model_config"]["VAE_model"]["c_y"]


def test_the_configured_widths_match_the_committed_shard(batch, config):
    r"""The check the net's constructor used to make against a constant, made against data.

    This is the replacement for the old ``c_u == (101 if use_up_st else 58)`` assertion in
    ``test_config_load.py``: same intent, but it reads the widths off a real HDF5 instead of
    off a second copy of the config, so it cannot go stale the way that one did.

    Limitation worth naming: it pins the config against the *tiny* shard, while ``default.yaml``
    points at production HDF5. It is therefore only as good as ``scripts/make_tiny_shard.py``'s
    ``CHANNELS`` tracking ``hdf5_dataset/new_pipeline/create_new_pipeline.py`` -- which is why
    that dict carries a comment pointing back at it.
    """
    vae = config["model_config"]["VAE_model"]
    assert batch.fhr_st.shape[-1] + batch.fhr_ph.shape[-1] == vae["c_y"]
    expected_c_u = (
        batch.up_st.shape[-1] + batch.up_ph.shape[-1]
        if vae["use_up_st"]
        else batch.up_ph.shape[-1]
    )
    assert expected_c_u == vae["c_u"]


def test_the_raw_signals_keep_the_sixteen_fold_decimation_ratio(batch):
    """The plots put the raw trace and the feature grid on one time axis."""
    assert batch.fhr.shape == (2, _EXPECTED_RAW)
    assert batch.up.shape == (2, _EXPECTED_RAW)
    assert batch.fhr.shape[-1] == 16 * batch.fhr_st.shape[-2]


def test_the_trim_removes_the_configured_minutes_from_each_end(batch, config):
    """300 on-disk-330 is not an arbitrary fixture size; it is what trim_minutes: 1.0 leaves.

    A stats file built at a different trim only warns, so the geometry is pinned here instead.
    """
    trim_minutes = config["dataset_config"]["dataloader_config"]["dataset_kwargs"]["trim_minutes"]
    trim_decimated = int(4 * 60 * trim_minutes) // 16
    assert batch.fhr_st.shape[1] == 330 - 2 * trim_decimated == _EXPECTED_T


def test_guid_is_a_list_of_strings_not_a_tensor(batch):
    """So the usual ``{k: v.to(device) for k, v in batch.items()}`` would crash here."""
    assert isinstance(batch.guid, list)
    assert all(isinstance(guid, str) for guid in batch.guid)


def test_the_batch_supports_both_attribute_and_item_access(batch):
    assert torch.equal(batch.fhr_st, batch["fhr_st"])


def test_every_model_input_is_finite(batch):
    """Normalization log-transforms 42 of the 43 scattering channels; a non-positive sample there
    would survive as a finite but absurd value, and a NaN would poison the first backward."""
    for field in ("fhr_st", "fhr_ph", "up_st", "up_ph", "weight"):
        assert torch.isfinite(batch[field]).all(), f"{field} carries a non-finite value"


def test_normalization_actually_happened(batch):
    """The silent failure this fixture exists to rule out.

    Every path into the stats reader is wrapped in a warn-and-continue, so a schema mismatch leaves
    the batch correctly shaped and entirely unnormalized. The shard is written with FHR around 140
    bpm; a mean anywhere near that means the stats file was ignored.
    """
    assert abs(float(batch.fhr.mean())) < 5.0, (
        "fhr looks unnormalized -- the stats file was probably rejected and normalization "
        "silently disabled"
    )


def test_the_val_loader_reads_the_held_out_list(config):
    """`val` and `test` both read `vae_test_datasets`; there is no in-process split."""
    data_module = GraphDataModule(config)
    assert next(iter(data_module.val_dataloader())).fhr_st.shape[1] == _EXPECTED_T


def test_a_null_stat_path_disables_normalization_rather_than_raising(config):
    """Why the entry point guards `stat_path` itself.

    The loader passes ``None`` straight through and the dataset merely skips normalization, so a
    typo'd key (the config key is ``stat_path``; the loader parameter is ``stats_path``) trains a
    model on raw-scale inputs and reports nothing. Nothing below the entry point will catch it.
    """
    unguarded = dict(config, dataset_config=dict(config["dataset_config"], stat_path=None))

    batch = next(iter(GraphDataModule(unguarded).train_dataloader()))

    assert abs(float(batch.fhr.mean())) > 100.0  # raw bpm scale: normalization never ran
