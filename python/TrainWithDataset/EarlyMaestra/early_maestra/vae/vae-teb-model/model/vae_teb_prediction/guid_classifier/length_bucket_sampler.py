"""Length-aware bucket sampler for efficient batching of variable-length sequences.

Groups GUID indices by segment count into configurable buckets, so that GUIDs
within the same batch have similar lengths.  This minimises padding waste in
``sequence_collate_fn``.

Typical usage::

    from model.vae_teb_prediction.guid_classifier.length_bucket_sampler import (
        LengthBucketSampler,
        create_bucketed_sequence_dataloader,
    )

    dataloader, dataset = create_bucketed_sequence_dataloader(
        hdf5_files=["fold_1/train/healthy_bg_cs.hdf5"],
        batch_size=8,
    )
"""

import random
from typing import Iterator, List, Optional, Sequence, Tuple, Any

from torch.utils.data import DataLoader, Sampler

from hdf5_dataset.guid_hdf5_dataset import (
    SignalSequenceDataset,
    sequence_collate_fn,
    DEFAULT_SEGMENT_DURATION,
)


class LengthBucketSampler(Sampler[int]):
    """Sampler that groups indices by length into buckets for batch efficiency.

    Consecutive ``batch_size`` yielded indices belong to the same bucket,
    ensuring minimal padding waste when used with a ``DataLoader`` that has
    matching ``batch_size``.

    Args:
        lengths: Number of segments per GUID, from ``dataset.guid_lengths``.
        batch_size: Must match the ``DataLoader``'s ``batch_size``.
        bucket_ranges: List of ``[lo, hi]`` inclusive ranges.  A GUID with
            ``lo <= length <= hi`` falls into that bucket.  GUIDs exceeding
            all ranges overflow into the last bucket.  Defaults to
            ``[[1, 5], [6, 12], [13, 20], [21, 40]]``.
        shuffle: Whether to shuffle indices within buckets and shuffle
            chunk order each epoch.
        seed: Base random seed.  Actual seed per epoch is ``seed + epoch``.
    """

    _DEFAULT_BUCKET_RANGES = [[1, 5], [6, 12], [13, 20], [21, 40]]

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        bucket_ranges: Optional[List[List[int]]] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.bucket_ranges = bucket_ranges or self._DEFAULT_BUCKET_RANGES
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        # Pre-assign indices to buckets.
        self._buckets = self._assign_buckets()

    def _assign_buckets(self) -> List[List[int]]:
        """Assign each GUID index to a bucket based on its segment count.

        Returns:
            List of lists, one per bucket, containing GUID indices.
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
                # Overflow: assign to the last bucket.
                buckets[-1].append(idx)

        return buckets

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling.

        Args:
            epoch: Current epoch number.
        """
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        """Yield flat GUID indices, grouped by bucket in batch-sized chunks.

        Yields:
            GUID indices such that consecutive ``batch_size`` indices share
            the same length bucket.
        """
        rng = random.Random(self.seed + self._epoch)
        all_chunks: List[List[int]] = []

        for bucket_indices in self._buckets:
            if not bucket_indices:
                continue

            indices = list(bucket_indices)
            if self.shuffle:
                rng.shuffle(indices)

            # Split into batch-sized chunks.
            for start in range(0, len(indices), self.batch_size):
                chunk = indices[start: start + self.batch_size]
                all_chunks.append(chunk)

        if self.shuffle:
            rng.shuffle(all_chunks)

        for chunk in all_chunks:
            yield from chunk

    def __len__(self) -> int:
        """Total number of GUID indices (all buckets combined).

        Returns:
            Number of GUIDs in the dataset.
        """
        return len(self.lengths)


def create_bucketed_sequence_dataloader(
    hdf5_files: List[str],
    batch_size: int = 8,
    bucket_ranges: Optional[List[List[int]]] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    segment_duration: float = DEFAULT_SEGMENT_DURATION,
    guid_cache_size: int = 128,
    stats_path: Optional[str] = None,
    normalize_fields: Optional[Sequence[str]] = None,
    trim_minutes: Optional[float] = None,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    seed: int = 42,
    **dataset_kwargs: Any,
) -> Tuple[DataLoader, SignalSequenceDataset]:
    """Create a DataLoader with length-bucket sampling for GUID sequences.

    Mirrors ``create_sequence_dataloader`` but uses a ``LengthBucketSampler``
    to group GUIDs with similar segment counts into the same batch, reducing
    padding waste.

    Args:
        hdf5_files: Paths to one or more HDF5 dataset files.
        batch_size: Number of GUIDs per batch.  Must match the sampler's
            ``batch_size`` (enforced internally).
        bucket_ranges: Bucket ``[lo, hi]`` ranges for the sampler.  Defaults
            to ``[[1, 5], [6, 12], [13, 20], [21, 40]]``.
        shuffle: Whether to shuffle within buckets each epoch.
        num_workers: Number of DataLoader worker processes.
        segment_duration: Segment duration for positional indices (seconds).
        guid_cache_size: GUID-level cache capacity (0 disables).
        stats_path: Path to HDF5 stats file for normalization.
        normalize_fields: Fields to normalize (``None`` = all with stats).
        trim_minutes: Minutes to trim from each end of every segment.
        prefetch_factor: DataLoader prefetch factor (only when workers > 0).
        pin_memory: Pin per-segment tensors for faster GPU transfer.
        seed: Random seed for the bucket sampler.
        **dataset_kwargs: Extra kwargs forwarded to ``CombinedHDF5Dataset``
            (``load_fields``, ``allowed_guids``, ``cs_label``, ``bg_label``,
            ``epoch_min``, ``epoch_max``, ``label``, ``cache_size``, ``dtype``).

    Returns:
        Tuple of ``(DataLoader, SignalSequenceDataset)`` so callers can
        access ``dataset.guid_lengths``, ``dataset.get_stats()``, etc.
    """
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
        shuffle=False,  # Sampler handles ordering; shuffle must be False.
        num_workers=num_workers,
        collate_fn=sequence_collate_fn,
        drop_last=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=False,  # Pinning handled at tensor level by inner dataset.
    )

    return dataloader, dataset
