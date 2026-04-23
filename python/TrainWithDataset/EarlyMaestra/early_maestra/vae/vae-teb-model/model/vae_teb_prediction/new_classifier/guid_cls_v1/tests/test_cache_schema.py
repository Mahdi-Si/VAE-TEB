"""Pure-h5py / numpy smoke tests for the precompute cache HDF5 schema.

These run without torch and validate only the on-disk layout produced by
:mod:`precompute_latents`. A synthetic cache is written by the helper in
``test_precompute_roundtrip`` (imported here) and checked against the schema
documented in the PRD.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest


from model.vae_teb_prediction.new_classifier.guid_cls_v1.tests.synthetic_cache import (
    D_MODEL,
    D_Z,
    L,
    T,
    write_synthetic_cache,
)


@pytest.fixture()
def cache_path(tmp_path: Path) -> Path:
    out = tmp_path / "fold_1" / "train.hdf5"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_synthetic_cache(out, num_guids=3, segments_per_guid=4)
    return out


def test_top_level_attrs_present(cache_path: Path) -> None:
    """All schema-v1 top-level attrs must be serialised."""
    with h5py.File(cache_path, "r", libver="latest") as fh:
        for key in (
            "schema_version",
            "vae_checkpoint_sha256",
            "vae_checkpoint_path",
            "cache_input_signature",
            "cache_input_summary_json",
            "use_up_st",
            "d_z",
            "d_model",
            "L",
            "T",
            "warmup_period",
            "partition",
            "fold_id",
            "mu_post_mean",
            "mu_post_var",
            "latent_stats_count",
        ):
            assert key in fh.attrs, f"missing attr {key!r}"
        assert fh.attrs["schema_version"] == "v1"
        assert int(fh.attrs["d_z"]) == D_Z
        assert int(fh.attrs["d_model"]) == D_MODEL
        assert int(fh.attrs["L"]) == L
        assert int(fh.attrs["T"]) == T
        mean = np.asarray(fh.attrs["mu_post_mean"])
        var = np.asarray(fh.attrs["mu_post_var"])
        assert mean.shape == (D_Z,)
        assert var.shape == (D_Z,)
        assert np.all(var > 0.0)


def test_per_guid_datasets_shapes(cache_path: Path) -> None:
    """Each GUID must store the schema-mandated per-segment datasets."""
    with h5py.File(cache_path, "r", libver="latest") as fh:
        guids_grp = fh["guids"]
        assert set(guids_grp.keys()) == {f"GUID_{g:03d}" for g in range(3)}
        for guid in guids_grp.keys():
            grp = guids_grp[guid]
            S = grp.attrs["S"]
            assert grp["h_y"].shape == (S, T, D_MODEL)
            assert grp["mu_prior"].shape == (S, T, D_Z)
            assert grp["mu_post"].shape == (S, T, D_Z)
            assert grp["kld_per_t"].shape == (S, T)
            assert grp["mean_alpha"].shape == (S, T, L)
            assert grp["weight"].shape == (S, T)
            assert grp["target"].shape == (S, T)
            assert grp["epoch"].shape == (S,)
            assert grp["time_from_labor_onset"].shape == (S,)
            assert grp["second_stage_onset"].shape == (S,)
            assert grp["cs_label"].shape == (S,)
            assert grp["bg_label"].shape == (S,)


def test_epochs_are_monotonic_per_guid(cache_path: Path) -> None:
    """The writer must store segments sorted ascending by epoch."""
    with h5py.File(cache_path, "r", libver="latest") as fh:
        guids_grp = fh["guids"]
        for guid in guids_grp.keys():
            epochs = guids_grp[guid]["epoch"][()]
            assert np.all(np.diff(epochs) > 0)
