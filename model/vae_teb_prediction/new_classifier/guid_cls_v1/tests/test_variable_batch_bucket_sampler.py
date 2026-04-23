"""Tests for true per-bucket batch sizing in the train loader sampler."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from hdf5_dataset.length_bucket_sampler import VariableBatchBucketSampler


def test_variable_batch_bucket_sampler_respects_bucket_sizes() -> None:
    """Each yielded batch should use the size configured for its bucket."""
    lengths = [2, 3, 4, 8, 9, 15, 16, 17, 30]
    bucket_batch_sizes = [
        ((1, 5), 3),
        ((6, 12), 2),
        ((13, 20), 1),
        ((21, 40), 4),
    ]

    sampler = VariableBatchBucketSampler(
        lengths=lengths,
        bucket_batch_sizes=bucket_batch_sizes,
        shuffle=False,
    )
    batches = list(sampler)

    assert batches == [
        [0, 1, 2],  # bucket [1,5], bs=3
        [3, 4],     # bucket [6,12], bs=2
        [5],        # bucket [13,20], bs=1
        [6],
        [7],
        [8],        # bucket [21,40], remainder under bs=4
    ]
    assert len(sampler) == len(batches)
