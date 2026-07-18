"""
GUID-grouped HDF5 Dataset for temporal sequence modeling.

Groups per-segment HDF5 data by GUID (recording) into variable-length sequences
sorted by epoch, suitable for LSTM / Transformer / temporal models that operate
on full recording trajectories.

Uses ``CombinedHDF5Dataset`` internally for all HDF5 I/O, normalization, trimming,
and per-segment caching.  This module adds a GUID-level grouping layer on top.

Architecture::

    ┌─────────────────────────────────────┐
    │      SignalSequenceDataset          │  len() = #GUIDs
    │  ┌───────────────────────────────┐  │
    │  │   CombinedHDF5Dataset         │  │  len() = #segments
    │  │   (HDF5 I/O, norm, trim,      │  │
    │  │    cache, multi-worker safe)  │  │
    │  └───────────────────────────────┘  │
    └─────────────────────────────────────┘

Quick start — dataset only::

    from hdf5_dataset.guid_hdf5_dataset import SignalSequenceDataset

    dataset = SignalSequenceDataset(
        paths=["fold_1/train/healthy_bg_cs.hdf5",
               "fold_1/train/acidosis_cs.hdf5"],
        stats_path="fold_1/train/stats.hdf5",
        trim_minutes=1.0,
        segment_duration=1200.0,      # 20-min effective window after trim
    )

    sample = dataset[0]
    # sample['fhr_st']                    -> (S_i, 300, 43)   segments x timesteps x channels
    # sample['target']                    -> (S_i, 300)
    # sample['delta_t']                   -> (S_i,)            seconds between segments
    # sample['segment_indices']           -> (S_i,)            ordinal grid slot
    # sample['time_from_labor_onset']     -> (S_i,)            seconds since labor onset (NaN if unavailable)
    # sample['guid']                      -> str

Quick start — dataloader with padding::

    from hdf5_dataset.guid_hdf5_dataset import create_sequence_dataloader

    loader = create_sequence_dataloader(
        hdf5_files=["fold_1/train/healthy_bg_cs.hdf5", ...],
        batch_size=4,
        num_workers=2,
        stats_path="fold_1/train/stats.hdf5",
        trim_minutes=1.0,
    )

    for batch in loader:
        fhr_st  = batch['fhr_st']                    # (B, S_max, 300, 43)
        mask    = batch['mask']                      # (B, S_max)  True=valid
        delta_t = batch['delta_t']                   # (B, S_max)
        target  = batch['target']                    # (B, S_max, 300)  pad=-1
        weight  = batch['weight']                    # (B, S_max, 300)  pad=0.0
        tflo    = batch['time_from_labor_onset']     # (B, S_max)  pad=0.0, NaN if unavailable
        lengths = batch['lengths']                   # (B,)
        guids   = batch['guid']                      # list[str]

Padding values (set in ``sequence_collate_fn``)::

    target          -> -1   (use ignore_index=-1 in loss)
    weight          -> 0.0
    segment_indices -> -1
    all features    -> 0.0

Trimming interaction:
    When ``trim_minutes`` is set (e.g. 1.0), the inner ``CombinedHDF5Dataset``
    trims ``trim_minutes`` from each end of every segment.  This affects:

    - Raw signals (fhr, up): 5280 -> 4800 samples  (at trim=1.0)
    - Feature sequences (fhr_st, fhr_ph, fhr_up_ph, up_st, up_ph, target,
      weight): 330 -> 300 timesteps  (at trim=1.0, decimation=16)

    ``segment_duration`` should match the effective window after trimming
    (default 1200.0 s = 20 min).
"""

import os
import gc
import atexit
import threading
import traceback
import warnings
import numpy as np
from typing import Sequence, List, Tuple, Dict, Any, Optional
from collections import defaultdict, OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader

from hdf5_dataset.hdf5_dataset import (
    CombinedHDF5Dataset,
    AttributeDict,
    create_initial_hdf5,
    append_sample,
)

# Default effective segment duration in seconds (20 min after 1-min trim each side).
DEFAULT_SEGMENT_DURATION = 1200.0

# Fields that are per-segment metadata strings/bools — not stacked into tensors.
_META_FIELDS = frozenset({
    'guid', 'cs_label', 'bg_label',
    'source_file', 'source_file_basename', 'source_file_index',
})

# Padding values used by the collate function.
_PAD_VALUES: Dict[str, float] = {
    'target': -1.0,
    'weight': 0.0,
}
_DEFAULT_PAD = 0.0
_SEGMENT_INDICES_PAD = -1


# ======================================================================
# Dataset
# ======================================================================


class SignalSequenceDataset(Dataset):
    """PyTorch Dataset that groups HDF5 segments by GUID into temporal sequences.

    Each item represents one full recording (GUID) with all its segments sorted
    by ``epoch`` (seconds relative to delivery, ascending).  This turns the
    flat per-segment view of ``CombinedHDF5Dataset`` into a variable-length
    sequence view for temporal models.

    Internally wraps a ``CombinedHDF5Dataset`` instance, reusing its HDF5 I/O,
    filtering, normalization, trimming, and per-segment caching logic.

    Multi-worker safety:
        Implements ``__getstate__``/``__setstate__`` to properly clear
        unpicklable state (threading locks, GUID-level cache) before the
        dataset is sent to DataLoader worker processes.  Each worker rebuilds
        its own locks and cache lazily.  The inner ``CombinedHDF5Dataset``
        handles its own file-handle and lock lifecycle independently.

    Args:
        segment_duration: Effective duration of one analysis segment in seconds,
            used to compute ``segment_indices`` for positional encoding.
            Should match the post-trim window length.  Defaults to 1200.0
            (20 min = 300 timesteps at 4 Hz / 16 decimation after 1-min trim).
        guid_cache_size: Number of GUID-level samples to keep in an LRU cache.
            Set to 0 to disable.  Each cached entry holds the stacked tensors
            for one GUID, so memory usage is proportional to
            ``guid_cache_size * max_segments * tensor_size``.  Defaults to 128.
        **kwargs: All remaining keyword arguments are forwarded verbatim to
            ``CombinedHDF5Dataset`` (``paths``, ``stats_path``, ``trim_minutes``,
            ``load_fields``, ``allowed_guids``, ``cs_label``, ``bg_label``,
            ``epoch_min``, ``epoch_max``, ``label``, ``cache_size``,
            ``pin_memory``, ``dtype``, ``normalize_fields``).
    """

    def __init__(
        self,
        segment_duration: float = DEFAULT_SEGMENT_DURATION,
        guid_cache_size: int = 128,
        **kwargs: Any,
    ):
        self.segment_duration = segment_duration
        self.guid_cache_size = guid_cache_size

        # Inner per-segment dataset — owns all HDF5 file handles and locks.
        self.inner_dataset = CombinedHDF5Dataset(**kwargs)

        # GUID-level cache and its lock (rebuilt after unpickling).
        self._guid_cache: OrderedDict[int, AttributeDict] = OrderedDict()
        self._guid_cache_lock = threading.Lock()
        self._access_count = 0

        # Build GUID -> sorted segment indices mapping (bulk metadata read).
        self._guid_list, self._guid_to_inner_indices = self._build_guid_mapping()

        # Register cleanup so open handles are closed even on abnormal exit.
        atexit.register(self._cleanup)

        print(
            f"SignalSequenceDataset: {len(self._guid_list)} GUIDs, "
            f"{len(self.inner_dataset)} total segments"
        )
        if self.guid_cache_size > 0:
            effective = min(self.guid_cache_size, len(self._guid_list))
            print(f"GUID-level caching enabled: up to {effective} GUIDs")

    # ------------------------------------------------------------------
    # Pickling (multi-worker DataLoader support)
    # ------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        """Exclude unpicklable threading locks and GUID cache for multiprocessing.

        The inner ``CombinedHDF5Dataset`` is pickled via its own
        ``__getstate__``, which clears its file handles and locks.
        """
        state = self.__dict__.copy()
        # Threading lock cannot be pickled; each worker rebuilds it.
        state['_guid_cache_lock'] = None
        # Each worker builds its own GUID cache from scratch.
        state['_guid_cache'] = OrderedDict()
        state['_access_count'] = 0
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Recreate threading lock after unpickling in a worker process."""
        self.__dict__.update(state)
        self._guid_cache_lock = threading.Lock()
        # inner_dataset's __setstate__ handles its own lock/handle recreation.

    # ------------------------------------------------------------------
    # Lifecycle cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        """Release GUID cache memory."""
        self.clear_cache()

    def __del__(self) -> None:
        """Cleanup when dataset is garbage collected."""
        if hasattr(self, "_guid_cache_lock"):
            self._cleanup()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _build_guid_mapping(self) -> Tuple[List[str], Dict[str, List[int]]]:
        """Read guid/epoch metadata in bulk and build per-GUID index lists.

        Iterates over the inner dataset's ``index_map`` grouped by file for
        efficient sequential HDF5 reads (sorted sample indices per file).
        Does *not* call ``__getitem__`` — only reads the lightweight ``guid``
        and ``epoch`` datasets.

        Returns:
            Tuple of:
                - Alphabetically sorted list of unique GUID strings.
                - Dict mapping each GUID to a list of inner-dataset indices,
                  sorted by ascending ``epoch`` value.
        """
        index_map = self.inner_dataset.index_map  # List[(file_idx, sample_idx)]
        n_total = len(index_map)

        # Group inner-dataset indices by HDF5 file for sequential access.
        file_groups: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for inner_idx, (f_idx, s_idx) in enumerate(index_map):
            file_groups[f_idx].append((inner_idx, s_idx))

        guid_arr: List[str] = [''] * n_total
        epoch_arr = np.empty(n_total, dtype=np.float32)

        for f_idx, pairs in file_groups.items():
            handle = self.inner_dataset._open_handle(f_idx)
            # Sort by sample index within the file for sequential disk access.
            pairs_sorted = sorted(pairs, key=lambda p: p[1])
            sample_indices = [p[1] for p in pairs_sorted]
            inner_indices = [p[0] for p in pairs_sorted]

            guids_raw = handle['guid'][sample_indices]
            epochs_raw = handle['epoch'][sample_indices]

            for i, inner_idx in enumerate(inner_indices):
                g = guids_raw[i]
                guid_arr[inner_idx] = (
                    g.decode('utf-8') if isinstance(g, bytes) else str(g)
                )
                epoch_arr[inner_idx] = epochs_raw[i]

        # Group by GUID and sort each group by epoch (ascending).
        guid_to_epoch_idx: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
        for inner_idx, guid in enumerate(guid_arr):
            guid_to_epoch_idx[guid].append((epoch_arr[inner_idx], inner_idx))

        guid_to_sorted: Dict[str, List[int]] = {}
        for guid, pairs in guid_to_epoch_idx.items():
            pairs.sort(key=lambda p: p[0])
            guid_to_sorted[guid] = [p[1] for p in pairs]

        guid_list = sorted(guid_to_sorted.keys())
        return guid_list, guid_to_sorted

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of unique GUIDs (recordings)."""
        return len(self._guid_list)

    def __getitem__(self, idx: int) -> AttributeDict:
        """Load all segments for one GUID and return as a stacked sequence.

        Retrieves each segment via the inner ``CombinedHDF5Dataset.__getitem__``
        (which applies normalization, trimming, caching, and field selection),
        then stacks tensor fields along a new leading segments dimension.

        Args:
            idx: GUID index (0-based into the alphabetically sorted GUID list).

        Returns:
            ``AttributeDict`` containing:
                - **Feature tensors** (``fhr_st``, ``fhr_ph``, ``fhr_up_ph``,
                  ``fhr``, ``up``, etc.): shape ``(S_i, ...)``, where ``S_i``
                  is the number of segments for this GUID and ``...`` is the
                  per-segment shape from the inner dataset (e.g. ``(300, 43)``
                  for ``fhr_st`` with ``trim_minutes=1.0``).
                - **target**: ``(S_i, seq_len)`` per-segment class targets.
                - **weight**: ``(S_i, seq_len)`` per-segment validity masks.
                - **epoch**: ``(S_i,)`` raw epoch values (seconds relative to
                  delivery).
                - **delta_t**: ``(S_i,)`` time gaps in seconds between
                  consecutive segments.  ``delta_t[0] = 0.0``.
                - **time_from_labor_onset**: ``(S_i,)`` seconds since labor
                  onset for each segment.  ``NaN`` if labor onset data was
                  not available for this GUID during dataset creation.
                - **segment_indices**: ``(S_i,)`` long — ordinal position on a
                  uniform grid with spacing ``segment_duration``.  Computed as
                  ``round((epoch[j] - epoch[0]) / segment_duration)``.
                - **num_segments**: int — the value ``S_i``.
                - **guid**: str — recording identifier.
                - **cs_label**: bool — caesarean section flag (from first segment).
                - **bg_label**: bool — blood gas flag (from first segment).
        """
        # Check GUID-level cache.
        if self.guid_cache_size > 0:
            with self._guid_cache_lock:
                if idx in self._guid_cache:
                    self._guid_cache.move_to_end(idx)
                    self._access_count += 1
                    return self._guid_cache[idx]

        guid = self._guid_list[idx]
        inner_indices = self._guid_to_inner_indices[guid]
        n_segments = len(inner_indices)

        # Fetch all segments via the inner dataset (reuses norm/trim/cache).
        segments = []
        seg_pos = 0
        try:
            for seg_pos, inner_idx in enumerate(inner_indices):
                segments.append(self.inner_dataset[inner_idx])
        except Exception as e:
            msg = (
                f"Failed loading segment {seg_pos}/{n_segments} "
                f"(inner_idx={inner_indices[seg_pos]}) for GUID '{guid}'"
            )
            warnings.warn(f"{msg}:\n{traceback.format_exc()}")
            raise RuntimeError(msg) from e

        # Collect raw epoch values before building derived fields.
        epochs = torch.tensor(
            [
                seg['epoch'].item()
                if torch.is_tensor(seg['epoch'])
                else float(seg['epoch'])
                for seg in segments
            ],
            dtype=torch.float32,
        )

        # delta_t: seconds between consecutive segments (first is 0).
        delta_t = torch.zeros(n_segments, dtype=torch.float32)
        if n_segments > 1:
            delta_t[1:] = epochs[1:] - epochs[:-1]
            assert (delta_t[1:] >= 0).all(), (
                f"GUID '{guid}': non-monotonic epochs detected. "
                f"delta_t has negative values: {delta_t.tolist()}"
            )

        # segment_indices: ordinal slot on a uniform grid of segment_duration.
        segment_indices = torch.round(
            (epochs - epochs[0]) / self.segment_duration
        ).long()

        # Stack all tensor fields along a new leading segments dimension.
        out: Dict[str, Any] = {}
        for key in segments[0].keys():
            if key in _META_FIELDS:
                continue
            val = segments[0][key]
            if torch.is_tensor(val):
                out[key] = torch.stack([seg[key] for seg in segments], dim=0)

        # Override / add derived and metadata fields.
        out['epoch'] = epochs
        out['delta_t'] = delta_t
        out['segment_indices'] = segment_indices
        out['num_segments'] = n_segments
        out['guid'] = guid
        out['cs_label'] = segments[0]['cs_label']
        out['bg_label'] = segments[0]['bg_label']

        sample = AttributeDict(out)

        # Populate GUID-level cache (LRU eviction).
        if self.guid_cache_size > 0:
            with self._guid_cache_lock:
                if len(self._guid_cache) >= self.guid_cache_size:
                    self._guid_cache.popitem(last=False)  # Evict LRU
                self._guid_cache[idx] = sample

        self._access_count += 1
        return sample

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_guid_list(self) -> List[str]:
        """Return the alphabetically sorted list of unique GUIDs.

        Returns:
            List of GUID strings.
        """
        return list(self._guid_list)

    @property
    def guid_lengths(self) -> List[int]:
        """Number of segments per GUID, in same order as ``__getitem__`` indexing.

        Returns:
            List of segment counts, one per GUID, ordered to match the
            alphabetically sorted GUID list used by ``__getitem__``.
        """
        return [len(self._guid_to_inner_indices[g]) for g in self._guid_list]

    def get_guid_segment_counts(self) -> Dict[str, int]:
        """Return the number of segments per GUID.

        Returns:
            Dict mapping GUID string to its segment count.
        """
        return {
            guid: len(indices)
            for guid, indices in self._guid_to_inner_indices.items()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return dataset-level statistics for monitoring.

        Returns:
            Dict with keys: ``num_guids``, ``total_segments``,
            ``guid_cache_size``, ``guid_cache_used``, ``access_count``,
            ``segment_duration``, ``inner_stats`` (from ``CombinedHDF5Dataset``).
        """
        counts = self.get_guid_segment_counts()
        seg_counts = list(counts.values())
        return {
            'num_guids': len(self._guid_list),
            'total_segments': len(self.inner_dataset),
            'min_segments_per_guid': min(seg_counts) if seg_counts else 0,
            'max_segments_per_guid': max(seg_counts) if seg_counts else 0,
            'mean_segments_per_guid': float(np.mean(seg_counts)) if seg_counts else 0.0,
            'guid_cache_size': self.guid_cache_size,
            'guid_cache_used': len(self._guid_cache),
            'access_count': self._access_count,
            'segment_duration': self.segment_duration,
            'inner_stats': self.inner_dataset.get_stats(),
        }

    def clear_cache(self) -> None:
        """Clear both the GUID-level cache and the inner per-segment cache."""
        if self._guid_cache_lock is not None:
            with self._guid_cache_lock:
                self._guid_cache.clear()
        self.inner_dataset.clear_cache()
        gc.collect()

    def estimate_class_weights(
        self, num_classes: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate inverse-frequency class weights from GUID-level labels.

        Reads only the ``target`` field from HDF5 in bulk (no signal data),
        computes the per-segment max, and counts per-class at the GUID level
        using the first segment of each GUID.

        Args:
            num_classes: Number of output classes.  Binary (2) by default:
                0 = healthy, 1 = unhealthy (target max > 1).

        Returns:
            Tuple of ``(weights, counts)`` where:
                - ``weights``: shape ``(num_classes,)`` — inverse-frequency
                  weights normalised so they sum to ``num_classes``.
                - ``counts``: shape ``(num_classes,)`` — per-class GUID counts.
        """
        index_map = self.inner_dataset.index_map

        # Collect only the first inner index per GUID — one read per GUID
        # instead of reading ALL ~49K segments.
        first_inner_indices = set()
        for guid in self._guid_list:
            first_inner_indices.add(self._guid_to_inner_indices[guid][0])

        # Group only these first-segment indices by file.
        file_groups: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for inner_idx in first_inner_indices:
            f_idx, s_idx = index_map[inner_idx]
            file_groups[f_idx].append((inner_idx, s_idx))

        # Read only the target field for first segments and compute max.
        target_max: Dict[int, float] = {}
        for f_idx, pairs in file_groups.items():
            handle = self.inner_dataset._open_handle(f_idx)
            pairs_sorted = sorted(pairs, key=lambda p: p[1])
            sample_indices = [p[1] for p in pairs_sorted]
            inner_indices = [p[0] for p in pairs_sorted]

            targets_raw = handle['target'][sample_indices]  # (N, seq_len)
            seg_max = np.max(targets_raw, axis=-1)  # (N,)
            for i, inner_idx in enumerate(inner_indices):
                target_max[inner_idx] = seg_max[i]

        # Count per class at GUID level (first segment determines label).
        counts = torch.zeros(num_classes, dtype=torch.long)
        for guid in self._guid_list:
            first_inner_idx = self._guid_to_inner_indices[guid][0]
            label_val = target_max[first_inner_idx]
            binary_label = int(label_val > 1)  # 0=healthy, 1=unhealthy
            counts[min(binary_label, num_classes - 1)] += 1

        total = counts.sum().float()
        if total == 0:
            return torch.ones(num_classes), counts

        weights = total / (num_classes * counts.float())
        weights = torch.where(
            torch.isinf(weights), torch.ones_like(weights), weights,
        )
        return weights, counts


# ======================================================================
# Collate function
# ======================================================================


def sequence_collate_fn(batch: List[AttributeDict]) -> AttributeDict:
    """Collate variable-length GUID sequences into a right-padded batch.

    Takes a list of samples from ``SignalSequenceDataset.__getitem__``
    (each with a different number of segments ``S_i``) and pads all tensor
    fields to the maximum length ``S_max = max(S_i)`` in the batch.

    Padding strategy:
        - **Right-padding**: real segments are left-aligned at index 0;
          padding fills the tail.
        - **target**: padded with ``-1`` (use ``ignore_index=-1`` in loss).
        - **weight**: padded with ``0.0`` (masked positions have zero weight).
        - **segment_indices**: padded with ``-1`` (distinguishable from
          real index 0).
        - **All other tensors** (features, epoch, delta_t): padded with ``0.0``.

    Args:
        batch: List of ``AttributeDict`` samples, length ``B``.

    Returns:
        ``AttributeDict`` containing:
            - **Padded tensor fields**: shape ``(B, S_max, ...)`` for each
              tensor that had shape ``(S_i, ...)`` in individual samples.
            - **mask**: ``torch.bool`` of shape ``(B, S_max)``.
              ``True`` for valid segment positions, ``False`` for padding.
            - **lengths**: ``torch.long`` of shape ``(B,)``.
              Number of real segments per sample.
            - **guid**: ``list[str]`` of length ``B`` (not tensorised).
            - **cs_label**: ``torch.bool`` of shape ``(B,)``.
            - **bg_label**: ``torch.bool`` of shape ``(B,)``.
    """
    B = len(batch)
    lengths = torch.tensor(
        [s['num_segments'] for s in batch], dtype=torch.long,
    )
    S_max = int(lengths.max().item())

    # Validity mask: True at positions with real segments.
    mask = torch.zeros(B, S_max, dtype=torch.bool)
    for i, L in enumerate(lengths):
        mask[i, :L] = True

    out: Dict[str, Any] = {
        'mask': mask,
        'lengths': lengths,
        'guid': [s['guid'] for s in batch],
        'cs_label': torch.tensor(
            [s['cs_label'] for s in batch], dtype=torch.bool,
        ),
        'bg_label': torch.tensor(
            [s['bg_label'] for s in batch], dtype=torch.bool,
        ),
    }

    # Identify tensor fields to pad (skip non-tensor / already-handled fields).
    skip_keys = {'num_segments', 'guid', 'cs_label', 'bg_label'}
    tensor_keys = [
        k for k in batch[0].keys()
        if k not in skip_keys and torch.is_tensor(batch[0][k])
    ]

    for key in tensor_keys:
        sample_tensors = [s[key] for s in batch]
        tail_shape = sample_tensors[0].shape[1:]  # shape after segments dim

        pad_val: float = _PAD_VALUES.get(key, _DEFAULT_PAD)
        if key == 'segment_indices':
            pad_val = _SEGMENT_INDICES_PAD

        padded = torch.full(
            (B, S_max, *tail_shape),
            fill_value=pad_val,
            dtype=sample_tensors[0].dtype,
        )
        for i, t in enumerate(sample_tensors):
            padded[i, :t.shape[0]] = t

        out[key] = padded

    return AttributeDict(out)


# ======================================================================
# DataLoader factory
# ======================================================================


def create_sequence_dataloader(
    hdf5_files: List[str],
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = True,
    segment_duration: float = DEFAULT_SEGMENT_DURATION,
    guid_cache_size: int = 128,
    stats_path: Optional[str] = None,
    normalize_fields: Optional[Sequence[str]] = None,
    trim_minutes: Optional[float] = None,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    # TODO: Add a LengthBucketSampler that groups GUIDs with similar segment
    #       counts into batches to minimise padding waste.  For now the default
    #       shuffle / sequential ordering is used.
    **dataset_kwargs: Any,
) -> DataLoader:
    """Create a DataLoader yielding padded per-GUID sequence batches.

    Convenience factory that instantiates a ``SignalSequenceDataset`` and
    wraps it in a ``DataLoader`` with ``sequence_collate_fn``.

    Args:
        hdf5_files: Paths to one or more HDF5 dataset files.
        batch_size: Number of GUIDs per batch.
        num_workers: Number of DataLoader worker processes.  Each worker gets
            its own copy of the dataset with independent file handles and caches
            (handled by ``__getstate__``/``__setstate__``).
        shuffle: Whether to shuffle GUIDs each epoch.
        segment_duration: Segment duration for positional indices (seconds).
        guid_cache_size: GUID-level cache capacity (0 disables).
        stats_path: Path to HDF5 stats file for normalization.
        normalize_fields: Fields to normalize (``None`` = all with stats).
        trim_minutes: Minutes to trim from each end of every segment.
        prefetch_factor: DataLoader prefetch factor (only when workers > 0).
        pin_memory: Pin per-segment tensors for faster GPU transfer.
        **dataset_kwargs: Extra kwargs forwarded to ``CombinedHDF5Dataset``
            (``load_fields``, ``allowed_guids``, ``cs_label``, ``bg_label``,
            ``epoch_min``, ``epoch_max``, ``label``, ``cache_size``, ``dtype``).

    Returns:
        A ``DataLoader`` that yields ``AttributeDict`` batches produced by
        ``sequence_collate_fn``.
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

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=sequence_collate_fn,
        drop_last=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context='spawn' if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=False,  # pinning handled at tensor level by inner dataset
    )


# ======================================================================
# Verification / smoke test
# ======================================================================


def _run_smoke_test() -> None:
    """Create a dummy HDF5, verify dataset + collate + dataloader end-to-end.

    Creates 3 GUIDs with 3/7/5 segments and irregular epoch gaps, then
    checks shapes, ``delta_t`` correctness, and padding values.
    """
    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp(prefix="guid_seq_test_")
    hdf5_path = os.path.join(tmpdir, "test.hdf5")
    print(f"Temp dir: {tmpdir}")

    # LEGACY pre-split layout (22-min segments): fhr_up_ph holds cross-phase
    # concatenated with UP self-phase and there is no up_ph dataset. The
    # current pipeline (new_pipeline/create_new_pipeline.py) produces
    # fhr_ph=66 / fhr_up_ph=79 / up_ph=15 instead, so this fixture does NOT
    # exercise the shape CombinedHDF5Dataset sees in production — in
    # particular nothing here covers the up_ph path.
    len_signal = 5280
    len_sequence = 330
    n_cross_phase_channels = 137

    create_initial_hdf5(
        hdf5_path,
        len_signal=len_signal,
        n_channels=44 + n_cross_phase_channels,
        len_sequence=len_sequence,
        n_cross_phase_channels=n_cross_phase_channels,
        n_up_st_channels=43,
    )

    rng = np.random.RandomState(42)

    # (guid, n_segments, epoch_start, (step_lo, step_hi))
    guid_configs = [
        ("GUID_A", 3, -36000.0, (1200.0, 1400.0)),
        ("GUID_B", 7, -44000.0, (1200.0, 2400.0)),
        ("GUID_C", 5, -30000.0, (1200.0, 1800.0)),
    ]

    for guid, n_segs, epoch_start, (step_lo, step_hi) in guid_configs:
        epoch = epoch_start
        for _ in range(n_segs):
            append_sample(
                hdf5_path,
                fhr=rng.randn(len_signal).astype(np.float32) * 10 + 140,
                up=rng.randn(len_signal).astype(np.float32) * 5 + 20,
                fhr_st=rng.randn(43, len_sequence).astype(np.float32),
                fhr_ph=rng.randn(44, len_sequence).astype(np.float32),
                fhr_up_ph=rng.randn(n_cross_phase_channels, len_sequence).astype(np.float32),
                up_st=rng.randn(43, len_sequence).astype(np.float32),
                target=np.ones(len_sequence, dtype=np.float32),
                weight=np.ones(len_sequence, dtype=np.float32),
                guid=guid,
                epoch=epoch,
                cs_label=(guid == "GUID_A"),
                bg_label=True,
            )
            epoch += rng.uniform(step_lo, step_hi)

    # --- Test dataset ---
    print("\n=== Testing SignalSequenceDataset ===")
    ds = SignalSequenceDataset(
        paths=[hdf5_path],
        cache_size=0,
        guid_cache_size=0,
        pin_memory=False,
    )
    assert len(ds) == 3, f"Expected 3 GUIDs, got {len(ds)}"
    print(f"len(dataset) = {len(ds)}  [OK]")

    sample = ds[0]
    guid = sample['guid']
    n_segs = sample['num_segments']
    print(f"\nSample GUID={guid}, num_segments={n_segs}")
    print(f"  fhr_st shape:         {sample['fhr_st'].shape}")
    print(f"  fhr_ph shape:         {sample['fhr_ph'].shape}")
    print(f"  fhr_up_ph shape:      {sample['fhr_up_ph'].shape}")
    print(f"  target shape:         {sample['target'].shape}")
    print(f"  weight shape:         {sample['weight'].shape}")
    print(f"  epoch shape:          {sample['epoch'].shape}")
    print(f"  delta_t:              {sample['delta_t']}")
    print(f"  segment_indices:      {sample['segment_indices']}")

    # Verify delta_t
    assert sample['delta_t'][0].item() == 0.0, "delta_t[0] should be 0.0"
    if n_segs > 1:
        expected_dt = sample['epoch'][1:] - sample['epoch'][:-1]
        assert torch.allclose(sample['delta_t'][1:], expected_dt, atol=1e-3), \
            "delta_t mismatch"
    print("  delta_t:              [OK]")

    # Verify segment_indices
    expected_si = torch.round(
        (sample['epoch'] - sample['epoch'][0]) / DEFAULT_SEGMENT_DURATION
    ).long()
    assert torch.equal(sample['segment_indices'], expected_si), \
        "segment_indices mismatch"
    print("  segment_indices:      [OK]")

    # Verify stats utility
    stats = ds.get_stats()
    assert stats['num_guids'] == 3
    assert stats['total_segments'] == 15
    print(f"  get_stats():          [OK] (max segs/guid={stats['max_segments_per_guid']})")

    # --- Test collate ---
    print("\n=== Testing sequence_collate_fn ===")
    loader = DataLoader(
        ds, batch_size=3, shuffle=False, collate_fn=sequence_collate_fn,
    )
    batch = next(iter(loader))

    B = 3
    S_max = max(
        len(ds._guid_to_inner_indices[ds._guid_list[i]]) for i in range(B)
    )
    print(f"Batch size={B}, S_max={S_max}")
    print(f"  mask shape:           {batch['mask'].shape}  (expect [{B}, {S_max}])")
    print(f"  lengths:              {batch['lengths']}")
    print(f"  fhr_st shape:         {batch['fhr_st'].shape}")
    print(f"  target shape:         {batch['target'].shape}")
    print(f"  weight shape:         {batch['weight'].shape}")
    print(f"  delta_t shape:        {batch['delta_t'].shape}")
    print(f"  segment_indices shape:{batch['segment_indices'].shape}")
    print(f"  guids:                {batch['guid']}")
    print(f"  cs_label:             {batch['cs_label']}")
    print(f"  bg_label:             {batch['bg_label']}")
    print(f"  mask:\n{batch['mask']}")

    # Assert padding values.
    for i in range(B):
        L = batch['lengths'][i].item()
        if L < S_max:
            target_pad = batch['target'][i, L:]
            assert (target_pad == -1.0).all(), \
                f"GUID {i}: target padding not -1. Got: {target_pad.unique()}"

            weight_pad = batch['weight'][i, L:]
            assert (weight_pad == 0.0).all(), \
                f"GUID {i}: weight padding not 0. Got: {weight_pad.unique()}"

            si_pad = batch['segment_indices'][i, L:]
            assert (si_pad == -1).all(), \
                f"GUID {i}: segment_indices padding not -1. Got: {si_pad.unique()}"

    print("\nAll padding assertions passed!")

    # --- Test dataloader factory ---
    print("\n=== Testing create_sequence_dataloader ===")
    loader2 = create_sequence_dataloader(
        hdf5_files=[hdf5_path],
        batch_size=2,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        cache_size=0,
        guid_cache_size=0,
    )
    batch2 = next(iter(loader2))
    print(
        f"Factory loader batch: {len(batch2['guid'])} GUIDs, "
        f"S_max={batch2['mask'].shape[1]}"
    )

    # --- Test trimming ---
    print("\n=== Testing trim_minutes interaction ===")
    ds_trimmed = SignalSequenceDataset(
        paths=[hdf5_path],
        cache_size=0,
        guid_cache_size=0,
        pin_memory=False,
        trim_minutes=1.0,
    )
    sample_t = ds_trimmed[0]
    # With trim_minutes=1.0: raw 5280->4800, features 330->300, target/weight 330->300
    trimmed_raw = len_signal - 2 * int(4 * 60 * 1.0)        # 5280 - 480 = 4800
    trimmed_dec = len_sequence - 2 * (int(4 * 60 * 1.0) // 16)  # 330 - 30 = 300
    assert sample_t['fhr'].shape[-1] == trimmed_raw, \
        f"fhr raw trim: expected {trimmed_raw}, got {sample_t['fhr'].shape[-1]}"
    assert sample_t['fhr_st'].shape[1] == trimmed_dec, \
        f"fhr_st dec trim: expected {trimmed_dec}, got {sample_t['fhr_st'].shape[1]}"
    assert sample_t['target'].shape[-1] == trimmed_dec, \
        f"target trim: expected {trimmed_dec}, got {sample_t['target'].shape[-1]}"
    assert sample_t['weight'].shape[-1] == trimmed_dec, \
        f"weight trim: expected {trimmed_dec}, got {sample_t['weight'].shape[-1]}"
    print(f"  fhr raw:              {sample_t['fhr'].shape}  (expect [..., {trimmed_raw}])")
    print(f"  fhr_st trimmed:       {sample_t['fhr_st'].shape}  (expect [..., {trimmed_dec}, 43])")
    print(f"  target trimmed:       {sample_t['target'].shape}  (expect [..., {trimmed_dec}])")
    print(f"  weight trimmed:       {sample_t['weight'].shape}  (expect [..., {trimmed_dec}])")
    print("  Trimming:             [OK]")

    # --- Test GUID-level cache ---
    print("\n=== Testing GUID-level cache ===")
    ds_cached = SignalSequenceDataset(
        paths=[hdf5_path],
        cache_size=0,
        guid_cache_size=10,
        pin_memory=False,
    )
    _ = ds_cached[0]
    _ = ds_cached[0]  # should hit cache
    assert ds_cached._access_count == 2
    assert len(ds_cached._guid_cache) == 1
    ds_cached.clear_cache()
    assert len(ds_cached._guid_cache) == 0
    print("  GUID cache:           [OK]")

    # Cleanup.
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("\nAll tests passed.")


if __name__ == "__main__":
    _run_smoke_test()
