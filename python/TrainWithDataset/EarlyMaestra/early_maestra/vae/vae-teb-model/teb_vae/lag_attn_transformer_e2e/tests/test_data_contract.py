r"""The committed shard really produces what this model's geometry and front ends assume.

This is the only place the geometry meets the loader, and it is the loader-side counterpart to the
config half of the pre-flight: a config check can say what was *asked* for, and only a batch can say
what arrived. Everything between the HDF5 file and the model fails silently -- the symmetric trim,
the stats-file normalization, the per-sample transpose -- and a stats-schema mismatch disables
normalization with a warning and hands back correctly-shaped, wrongly-scaled tensors.

The stakes are higher here than for either sibling, because **both** model inputs are raw signals
that the front ends consume exactly as the loader produced them. Without ``'fhr'`` in
``normalize_fields`` the reconstruction target arrives in bpm and the Gaussian NLL is meaningless;
without ``'up'`` the source stream arrives in raw contraction units and every coupling number the
model reports is measured at an operating point nobody chose. Neither raises anywhere.

The loader configuration used here is this package's own ``configs/tiny.yaml``, resolved through its
``base:`` chain -- the exact configuration the smoke fit trains under, so the contract asserted here
is the contract the model actually sees.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import pytest
import torch

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from train.data_module import GraphDataModule

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TINY_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"

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


def test_the_model_this_config_builds_accepts_this_batch(config, batch):
    """The batch the loader produces is one the net's own input guards admit: both raw signals at
    the trimmed ``sequence_length * raw_per_step`` and the weight on the trimmed $(B, T)$ grid.
    Those guards name ``trim_minutes`` when they fire, which is the failure this asserts the
    absence of."""
    from teb_vae.lag_attn_transformer_e2e.trainer import LagAttnTrfE2ETrainer
    import tempfile

    import yaml

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        kwargs = LagAttnTrfE2ETrainer(config_file_path=str(path))._build_model_kwargs()

    model = SeqVaeLagAttnTrfE2E(**kwargs).eval()
    with torch.no_grad():
        outputs = model(batch.fhr, batch.up, batch.weight)

    assert outputs["mu_prior"].shape[:2] == (batch.fhr.shape[0], model.sequence_length)


def test_every_declared_load_field_is_present_on_the_batch(config, batch):
    """The other direction. The loader skips a field it cannot find in the shard *silently*, so a
    field named in the config and absent from the HDF5 produces a batch that is missing it and a
    task that fails one layer further in."""
    load_fields = config["dataset_config"]["dataloader_config"]["dataset_kwargs"]["load_fields"]

    for field in load_fields:
        assert getattr(batch, field, None) is not None, f"{field} never reached the batch"


def test_both_raw_signals_are_normalized(config, batch):
    """Both must sit in ``load_fields`` and ``normalize_fields``. The numeric check is what makes
    this operational: the stats file being rejected disables normalization with a warning, and
    correctly-shaped tensors at the wrong scale look exactly like correct ones."""
    dataloader = config["dataset_config"]["dataloader_config"]

    for field in ("fhr", "up"):
        assert field in dataloader["dataset_kwargs"]["load_fields"]
        assert field in dataloader["normalize_fields"]
        assert abs(float(getattr(batch, field).mean())) < 5.0, (
            f"{field} looks unnormalized -- the stats file was probably rejected and "
            f"normalization silently disabled"
        )


def test_the_stats_file_trim_matches_the_loader_trim(config):
    """A mismatch only warns in the loader, so it is pinned here instead."""
    loader_trim = config["dataset_config"]["dataloader_config"]["dataset_kwargs"][
        "trim_minutes"
    ]
    with h5py.File(config["dataset_config"]["stat_path"], "r") as stats:
        stats_trim = float(stats.attrs.get("trim_minutes", -1.0))
    assert stats_trim == float(loader_trim)
