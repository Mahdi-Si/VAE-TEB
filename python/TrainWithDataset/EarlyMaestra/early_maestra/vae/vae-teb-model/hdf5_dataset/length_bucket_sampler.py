"""Length-aware bucket sampler for efficient batching of variable-length sequences.

Groups GUID indices by segment count into configurable buckets so that GUIDs
within the same batch have similar lengths. This minimises padding waste when
used with temporal collate functions that pad to the per-batch maximum.

The sampler is reusable across different GUID-level pipelines; it only needs
a per-GUID segment-count array and a batch size.

Typical usage::

    from hdf5_dataset.length_bucket_sampler import LengthBucketSampler

    lengths = [dataset.num_segments(i) for i in range(len(dataset))]
    sampler = LengthBucketSampler(
        lengths=lengths,
        batch_size=8,
        bucket_ranges=[[1, 5], [6, 12], [13, 20], [21, 40]],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        sampler=sampler,
        collate_fn=my_collate_fn,
        shuffle=False,   # sampler controls order
    )
"""

from __future__ import annotations

import random
from typing import Any, Iterator, List, Optional, Sequence, Tuple

from torch.utils.data import DataLoader, Sampler


class LengthBucketSampler(Sampler[int]):
    """Sampler that groups indices by length into buckets for batch efficiency.

    Consecutive ``batch_size`` yielded indices belong to the same bucket,
    ensuring minimal padding waste when used with a ``DataLoader`` that has
    matching ``batch_size``.

    Args:
        lengths: Per-item length (e.g. number of segments per GUID) used to
            assign each index to a bucket.
        batch_size: Must match the ``DataLoader``'s ``batch_size`` for the
            bucket chunking to line up with batch boundaries.
        bucket_ranges: List of ``[lo, hi]`` inclusive ranges. An item with
            ``lo <= length <= hi`` falls into that bucket. Items whose length
            exceeds every range overflow into the last bucket. Defaults to
            ``[[1, 5], [6, 12], [13, 20], [21, 40]]``.
        shuffle: Whether to shuffle indices within buckets and shuffle chunk
            order each epoch.
        seed: Base random seed. Actual seed per epoch is ``seed + epoch``.
    """

    _DEFAULT_BUCKET_RANGES: List[List[int]] = [[1, 5], [6, 12], [13, 20], [21, 40]]

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        bucket_ranges: Optional[List[List[int]]] = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.lengths: List[int] = list(lengths)
        self.batch_size = int(batch_size)
        self.bucket_ranges = bucket_ranges or self._DEFAULT_BUCKET_RANGES
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0
        self._buckets = self._assign_buckets()

    def _assign_buckets(self) -> List[List[int]]:
        """Assign each index to a bucket based on its length.

        Returns:
            List of lists (one per bucket) holding the indices that fall in
            that bucket. Items longer than the final range are appended to
            the last bucket rather than dropped.
        """
        num_buckets = len(self.bucket_ranges)
        buckets: List[List[int]] = [[] for _ in range(num_buckets)]

        for idx, length in enumerate(self.lengths):
            assigned = False
            for b_idx, (lo, hi) in enumerate(self.bucket_ranges):
                if lo <= length <= hi:
                    buckets[b_idx].append(idx)
                    assigned = True
                    break
            if not assigned:
                buckets[-1].append(idx)

        return buckets

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used for deterministic shuffling.

        Args:
            epoch: Current epoch number.
        """
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        """Yield flat indices, grouped by bucket in ``batch_size`` chunks.

        Auto-increments the internal epoch counter so each call (each training
        epoch) produces a different shuffle order without requiring an
        explicit :meth:`set_epoch` call.

        Yields:
            Dataset indices such that every consecutive ``batch_size`` indices
            come from the same bucket.
        """
        epoch = self._epoch
        self._epoch += 1
        rng = random.Random(self.seed + epoch)
        all_chunks: List[List[int]] = []

        for bucket_indices in self._buckets:
            if not bucket_indices:
                continue

            indices = list(bucket_indices)
            if self.shuffle:
                rng.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                chunk = indices[start: start + self.batch_size]
                all_chunks.append(chunk)

        if self.shuffle:
            rng.shuffle(all_chunks)

        for chunk in all_chunks:
            yield from chunk

    def __len__(self) -> int:
        """Return the total number of items across all buckets.

        Returns:
            Total item count (sum of bucket sizes, equal to ``len(lengths)``).
        """
        return len(self.lengths)


class VariableBatchBucketSampler(Sampler[List[int]]):
    """Batch sampler with bucket-specific batch sizes.

    Each yielded batch contains indices from a single length bucket, but
    different buckets may use different batch sizes. This is useful when long
    GUID sequences need smaller batches to stay within memory limits.

    Args:
        lengths: Per-item length used for bucket assignment.
        bucket_batch_sizes: Ordered ``[((lo, hi), batch_size), ...]``.
        shuffle: Whether to shuffle within buckets and shuffle batch order.
        seed: Base random seed. Actual seed per epoch is ``seed + epoch``.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        bucket_batch_sizes: Sequence[Tuple[Tuple[int, int], int]],
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not bucket_batch_sizes:
            raise ValueError("bucket_batch_sizes must not be empty")
        self.lengths: List[int] = list(lengths)
        self.bucket_batch_sizes = [
            ((int(lo), int(hi)), int(bs))
            for (lo, hi), bs in bucket_batch_sizes
        ]
        if any(bs <= 0 for _, bs in self.bucket_batch_sizes):
            raise ValueError("all bucket batch sizes must be positive")
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0
        self._buckets = self._assign_buckets()

    def _assign_buckets(self) -> List[List[int]]:
        """Assign each index to a bucket based on its length."""
        num_buckets = len(self.bucket_batch_sizes)
        buckets: List[List[int]] = [[] for _ in range(num_buckets)]

        for idx, length in enumerate(self.lengths):
            assigned = False
            for b_idx, ((lo, hi), _batch_size) in enumerate(self.bucket_batch_sizes):
                if lo <= length <= hi:
                    buckets[b_idx].append(idx)
                    assigned = True
                    break
            if not assigned:
                buckets[-1].append(idx)

        return buckets

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used for deterministic shuffling."""
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        """Yield one same-bucket batch at a time."""
        epoch = self._epoch
        self._epoch += 1
        rng = random.Random(self.seed + epoch)
        all_batches: List[List[int]] = []

        for bucket_indices, (_bucket_range, batch_size) in zip(
            self._buckets, self.bucket_batch_sizes
        ):
            if not bucket_indices:
                continue

            indices = list(bucket_indices)
            if self.shuffle:
                rng.shuffle(indices)

            for start in range(0, len(indices), batch_size):
                all_batches.append(indices[start: start + batch_size])

        if self.shuffle:
            rng.shuffle(all_batches)

        yield from all_batches

    def __len__(self) -> int:
        """Return the number of batches yielded per epoch."""
        total_batches = 0
        for bucket_indices, (_bucket_range, batch_size) in zip(
            self._buckets, self.bucket_batch_sizes
        ):
            if not bucket_indices:
                continue
            total_batches += (len(bucket_indices) + batch_size - 1) // batch_size
        return total_batches


def create_bucketed_sequence_dataloader(
    hdf5_files: List[str],
    batch_size: int = 8,
    bucket_ranges: Optional[List[List[int]]] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    segment_duration: Optional[float] = None,
    guid_cache_size: int = 128,
    stats_path: Optional[str] = None,
    normalize_fields: Optional[Sequence[str]] = None,
    trim_minutes: Optional[float] = None,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    seed: int = 42,
    **dataset_kwargs: Any,
) -> Tuple[DataLoader, Any]:
    """Create a DataLoader with length-bucket sampling for GUID sequences.

    Thin convenience wrapper that pairs :class:`LengthBucketSampler` with
    :class:`hdf5_dataset.guid_hdf5_dataset.SignalSequenceDataset` and the
    ``sequence_collate_fn`` collate function. Intended for the legacy temporal
    pipeline; new code building a custom dataset should construct the sampler
    directly.

    Args:
        hdf5_files: Paths to one or more HDF5 dataset files.
        batch_size: Number of GUIDs per batch. Must match the sampler's
            ``batch_size`` (enforced internally).
        bucket_ranges: Bucket ``[lo, hi]`` ranges for the sampler. Defaults to
            ``[[1, 5], [6, 12], [13, 20], [21, 40]]``.
        shuffle: Whether to shuffle within buckets each epoch.
        num_workers: Number of DataLoader worker processes.
        segment_duration: Segment duration in seconds. When ``None`` the
            dataset's default is used.
        guid_cache_size: GUID-level cache capacity (0 disables).
        stats_path: Path to HDF5 stats file for normalisation.
        normalize_fields: Fields to normalise (``None`` = all with stats).
        trim_minutes: Minutes to trim from each end of every segment.
        prefetch_factor: DataLoader prefetch factor (only when workers > 0).
        pin_memory: Pin per-segment tensors for faster GPU transfer.
        seed: Random seed for the bucket sampler.
        **dataset_kwargs: Extra kwargs forwarded to
            :class:`hdf5_dataset.hdf5_dataset.CombinedHDF5Dataset`.

    Returns:
        Tuple of ``(DataLoader, SignalSequenceDataset)`` so callers can access
        ``dataset.guid_lengths`` and related attributes.
    """
    # Lazy import to avoid a hard dependency on guid_hdf5_dataset for callers
    # that only want the sampler class.
    from hdf5_dataset.guid_hdf5_dataset import (  # noqa: WPS433
        DEFAULT_SEGMENT_DURATION,
        SignalSequenceDataset,
        sequence_collate_fn,
    )

    if segment_duration is None:
        segment_duration = DEFAULT_SEGMENT_DURATION

    dataset = SignalSequenceDataset(
        segment_duration=segment_duration,
        guid_cache_size=guid_cache_size,
        paths=hdf5_files,
        stats_path=stats_path,
        normalize_fields=normalize_fields,
        trim_minutes=trim_minutes,
        pin_memory=pin_memory,
        **dataset_kwargs,
    )

    lengths = dataset.guid_lengths
    sampler = LengthBucketSampler(
        lengths=lengths,
        batch_size=batch_size,
        bucket_ranges=bucket_ranges,
        shuffle=shuffle,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,  # Sampler controls order; DataLoader shuffle must be off.
        num_workers=num_workers,
        collate_fn=sequence_collate_fn,
        drop_last=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=False,
    )

    return dataloader, dataset


__all__ = [
    "LengthBucketSampler",
    "VariableBatchBucketSampler",
    "create_bucketed_sequence_dataloader",
]
