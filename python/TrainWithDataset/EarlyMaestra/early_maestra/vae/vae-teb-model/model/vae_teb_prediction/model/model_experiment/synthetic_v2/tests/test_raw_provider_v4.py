r"""S3-T04 tests for ``make_raw_provider_v4`` -- the untrimmed on-demand raw regenerator."""

from __future__ import annotations

import numpy as np

import pytest

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v4 import (
    make_raw_provider_v4,
)

pytestmark = pytest.mark.v4


def test_raw_provider_v4_roundtrip_and_length(tiny_cache_v4):
    r"""Regenerated (physical) raw matches the normalised cache after re-normalisation; length 5280."""
    config = tiny_cache_v4["config"]
    cache_dir = tiny_cache_v4["cache_dir"]

    with np.load(cache_dir / "norm_stats.npz") as z:
        fmean, fstd = float(z["fhr_mean"]), float(z["fhr_std"])
        umean, ustd = float(z["up_mean"]), float(z["up_std"])
    with np.load(cache_dir / "train.npz") as z:
        cell_ids = np.asarray(z["sample_cell_id"])
        raw_idx = np.asarray(z["sample_raw_index"])
        fhr_cached = np.asarray(z["fhr"])
        up_cached = np.asarray(z["up"])

    provider = make_raw_provider_v4(config, "train", benchmark="G1_raw_v4", cache_dir=cache_dir)
    assert provider.window_length == 5280

    for row in range(len(cell_ids)):
        fhr_phys, up_phys = provider(int(cell_ids[row]), int(raw_idx[row]))
        assert fhr_phys.shape == (5280,)
        assert up_phys.shape == (5280,)
        fhr_norm = (fhr_phys - fmean) / fstd
        up_norm = (up_phys - umean) / ustd
        np.testing.assert_allclose(fhr_norm, fhr_cached[row], atol=1e-4)
        np.testing.assert_allclose(up_norm, up_cached[row], atol=1e-4)


def test_raw_provider_v4_unknown_cell_returns_nan(tiny_cache_v4):
    r"""An unknown cell / out-of-range row degrades to a NaN pair rather than raising."""
    provider = make_raw_provider_v4(
        tiny_cache_v4["config"], "train", benchmark="G1_raw_v4",
        cache_dir=tiny_cache_v4["cache_dir"],
    )
    fhr, up = provider(9999, 0)
    assert fhr.shape == (5280,)
    assert np.isnan(fhr).all()
    assert np.isnan(up).all()
