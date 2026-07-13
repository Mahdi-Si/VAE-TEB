r"""S3-T01 tests for ``SyntheticRawDatasetV4`` (raw cache loader)."""

from __future__ import annotations

import torch

import pytest

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.dataset_v4 import (
    SyntheticRawDatasetV4,
)

pytestmark = pytest.mark.v4


def test_dataset_v4_getitem_shapes(tiny_cache_v4):
    r"""``__getitem__`` returns the raw fields + provenance with correct shapes/dtypes."""
    cache_dir = tiny_cache_v4["cache_dir"]
    ds = SyntheticRawDatasetV4(cache_dir / "train.npz")
    n_expected = len(tiny_cache_v4["cells"]) * tiny_cache_v4["n_override"]["train"]
    assert len(ds) == n_expected

    sample = ds[0]
    assert sample["fhr"].shape == (5280,)
    assert sample["up"].shape == (5280,)
    assert sample["weight"].shape == (330,)
    assert sample["target"].shape == (330,)
    assert sample["fhr"].dtype == torch.float32
    assert sample["up"].dtype == torch.float32
    assert sample["true_lag_tt"].shape == (330,)
    assert sample["true_lag_tt"].dtype == torch.int64

    # Provenance present per item.
    assert isinstance(sample["te_true"], float)
    assert isinstance(sample["delay"], int)
    assert isinstance(sample["cell_id"], int)
    assert isinstance(sample["held_out"], int)
    assert isinstance(sample["raw_index"], int)
    assert isinstance(sample["guid"], str)


def test_dataset_v4_te_true_matches_cell(tiny_cache_v4):
    r"""Per-sample ``te_true`` takes one of the ladder values (0.0 or the signal cell's TE)."""
    ds = SyntheticRawDatasetV4(tiny_cache_v4["cache_dir"] / "train.npz")
    te_values = {round(ds[i]["te_true"], 3) for i in range(len(ds))}
    # Two cells: null (0.0) and one signal level.
    assert 0.0 in te_values
    assert any(v > 0.0 for v in te_values)


def test_dataset_v4_eager_matches_mmap(tiny_cache_v4):
    r"""Eager and memory-mapped backing return identical tensors."""
    npz = tiny_cache_v4["cache_dir"] / "train.npz"
    ds_mmap = SyntheticRawDatasetV4(npz, mmap=True)
    ds_eager = SyntheticRawDatasetV4(npz, mmap=False)
    a = ds_mmap[1]
    b = ds_eager[1]
    assert torch.allclose(a["fhr"], b["fhr"])
    assert torch.allclose(a["weight"], b["weight"])
