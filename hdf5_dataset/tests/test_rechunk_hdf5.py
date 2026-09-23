"""One round trip of the committed causal fixture through the re-chunk tool.

The converted shard has to be the same dataset under a different chunk layout: the chunk shapes are
what the tool changes, and everything else -- attributes, values, the resolved warm-up and the
tensors the loader serves -- is what it must not.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from hdf5_dataset.hdf5_dataset import CombinedHDF5Dataset, read_causal_warmup
from hdf5_dataset.rechunk_hdf5 import rechunk

FIXTURES = Path(__file__).resolve().parents[2] / "teb_vae" / "lag_attn" / "tests" / "fixtures"
SHARD = FIXTURES / "tiny_shard_causal_int.hdf5"
STATS = FIXTURES / "tiny_stats_causal_int.hdf5"
TRIM_MINUTES = 1.0


@pytest.fixture(scope="module")
def rechunked(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return rechunk(SHARD, tmp_path_factory.mktemp("rechunk") / "shard.hdf5")


def _attrs_equal(left: h5py.AttributeManager, right: h5py.AttributeManager) -> bool:
    return set(left) == set(right) and all(np.array_equal(left[k], right[k]) for k in left)


def test_chunks_are_one_sample_deep_and_one_d_datasets_keep_theirs(rechunked: Path) -> None:
    with h5py.File(SHARD, "r") as src, h5py.File(rechunked, "r") as dst:
        for name, dataset in src.items():
            if dataset.ndim >= 2:
                assert dst[name].chunks == (1, *dataset.shape[1:]), name
            else:
                assert dst[name].chunks == dataset.chunks, name
            assert dst[name].maxshape == dataset.maxshape, name


def test_attributes_and_values_survive(rechunked: Path) -> None:
    with h5py.File(SHARD, "r") as src, h5py.File(rechunked, "r") as dst:
        assert set(src) == set(dst)
        assert _attrs_equal(src.attrs, dst.attrs)
        for name, dataset in src.items():
            assert _attrs_equal(dataset.attrs, dst[name].attrs), name
            assert dst[name].dtype == dataset.dtype, name
            if dataset.dtype.kind == "O":
                assert dst[name][()].tolist() == dataset[()].tolist()
            else:
                assert np.array_equal(dst[name][()], dataset[()], equal_nan=True), name


def test_the_warm_up_and_the_loader_resolve_identically(rechunked: Path) -> None:
    before = read_causal_warmup([str(SHARD)], TRIM_MINUTES)
    after = read_causal_warmup([str(rechunked)], TRIM_MINUTES)
    for item in fields(before):
        if item.name == "paths":
            continue
        left, right = getattr(before, item.name), getattr(after, item.name)
        if isinstance(left, dict):
            assert set(left) == set(right), item.name
            assert all(np.array_equal(left[k], right[k]) for k in left), item.name
        else:
            assert left == right, item.name

    def dataset(path: Path) -> CombinedHDF5Dataset:
        return CombinedHDF5Dataset(
            paths=[str(path)], cache_size=0, pin_memory=False,
            trim_minutes=TRIM_MINUTES, stats_path=str(STATS),
        )

    original, converted = dataset(SHARD), dataset(rechunked)
    assert len(original) == len(converted) > 0
    for index in range(len(original)):
        left, right = original[index], converted[index]
        assert set(left) == set(right)
        for key in left:
            if key.startswith("source_file"):
                continue  # the fields that name the file a sample came from
            if isinstance(left[key], torch.Tensor):
                assert torch.equal(left[key].nan_to_num(), right[key].nan_to_num()), key
                assert torch.equal(left[key].isnan(), right[key].isnan()), key
            else:
                assert left[key] == right[key], key


def test_an_existing_destination_is_refused(rechunked: Path) -> None:
    with pytest.raises(FileExistsError, match=str(rechunked.name)):
        rechunk(SHARD, rechunked)
