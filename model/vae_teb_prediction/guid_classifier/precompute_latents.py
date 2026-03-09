"""Pre-compute and cache VAE latent representations for fast training.

This module provides utilities to pre-compute the frozen VAE encoder's
posterior mean (``mu_post``) for every segment in the dataset and cache
the results to HDF5 files on disk.  During training the cached latents
are loaded directly, bypassing the expensive VAE forward pass entirely
and typically reducing per-epoch wall-clock time by >50%.

Components:

- :func:`precompute_fold_latents` — Pre-compute latents for one fold
  (train/val/test partitions).
- :class:`PrecomputedLatentDataset` — Dataset wrapper that injects
  ``mu_post_precomputed`` into each sample so the model's
  :meth:`~TemporalVaeClassifier.forward` skips VAE encoding.
- :func:`create_precomputed_sequence_dataloader` — Factory that builds
  a ``DataLoader`` with pre-computed latents and length-bucket sampling.

Cache invalidation is handled by storing a SHA-256 hash of the VAE
checkpoint inside the pre-computed HDF5 file.  If the checkpoint changes,
:class:`PrecomputedLatentDataset` raises a ``ValueError`` at init time.

Example — pre-compute from CLI::

    python -m model.vae_teb_prediction.guid_classifier.precompute_latents \\
        --config model/vae_teb_prediction/guid_classifier/config_temporal.yaml \\
        --fold_ids 1 2 3 \\
        --output_dir /data/precomputed_latents \\
        --device cuda:0

Example — use in training::

    from model.vae_teb_prediction.guid_classifier.precompute_latents import (
        create_precomputed_sequence_dataloader,
    )

    loader, dataset = create_precomputed_sequence_dataloader(
        precomputed_path="precomputed_fold_1_train.hdf5",
        hdf5_files=train_hdf5_files,
        batch_size=8,
    )
    # Batches now contain 'mu_post_precomputed' — model.forward() uses it
    # automatically and skips VAE encoding.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from loguru import logger



# ====================================================================== #
#  Utility                                                                 #
# ====================================================================== #


def _compute_file_hash(filepath: str) -> str:
    """Compute the SHA-256 hash of a file for cache invalidation.

    Args:
        filepath: Absolute or relative path to the file.

    Returns:
        Hexadecimal SHA-256 digest string (64 characters).
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


# ====================================================================== #
#  Pre-computation                                                         #
# ====================================================================== #


def precompute_fold_latents(
    fold_id: int,
    kfold_base_path: str,
    vae_checkpoint: str,
    output_dir: str,
    config: Dict[str, Any],
    device: str = "cuda:0",
    chunk_size: int = 32,
) -> Dict[str, str]:
    """Pre-compute VAE ``mu_post`` for all segments in one fold.

    For each partition (train / val / test):

    1. Loads segments via :class:`SignalSequenceDataset` (GUID-grouped).
    2. Runs :meth:`SeqVae.encode_only` in chunks of ``chunk_size``.
    3. Saves per-GUID ``mu_post`` to an HDF5 file.
    4. Stores metadata (VAE checkpoint path + SHA-256 hash) for cache
       invalidation.

    The output HDF5 layout::

        precomputed_fold_{fold_id}_{partition}.hdf5
        ├── attrs: vae_checkpoint_hash, vae_checkpoint_path, ...
        └── guids/
            ├── GUID_001/
            │   ├── mu_post   (S_1, 300, 16)  float32
            │   └── epochs    (S_1,)           float64
            ├── GUID_002/
            │   ├── mu_post   (S_2, 300, 16)  float32
            │   └── epochs    (S_2,)           float64
            └── ...

    Args:
        fold_id: Fold number (1-based, e.g. 1–10).
        kfold_base_path: Root directory of the k-fold dataset
            (contains ``fold_1/``, ``fold_2/``, …).
        vae_checkpoint: Path to the pre-trained VAE ``.ckpt`` file.
        output_dir: Directory where HDF5 output files are saved.
        config: Full config dict loaded from ``config_temporal.yaml``.
        device: CUDA device string (e.g. ``'cuda:0'``).
        chunk_size: Number of segments processed through the VAE per
            forward pass.  Larger values use more GPU memory but are
            faster.

    Returns:
        Dict mapping partition names (``'train'``, ``'val'``, ``'test'``)
        to the absolute paths of the generated HDF5 files.

    Raises:
        FileNotFoundError: If ``vae_checkpoint`` or ``kfold_base_path``
            does not exist.
    """
    from hdf5_dataset.guid_hdf5_dataset import SignalSequenceDataset
    from model.vae_teb_prediction.kfold_classifier_trainer import (
        get_fold_datasets,
    )
    from model.vae_teb_prediction.vae_teb_model_prediction import SeqVae
    from train.graph_models_utils import load_checkpoint_strict

    # ---- Load and freeze VAE ------------------------------------------- #
    logger.info("Loading VAE checkpoint: {}", vae_checkpoint)
    vae_model = SeqVae()
    load_checkpoint_strict(vae_model, checkpoint=vae_checkpoint)
    vae_model = vae_model.to(device)
    vae_model.eval()
    for param in vae_model.parameters():
        param.requires_grad = False

    vae_hash = _compute_file_hash(vae_checkpoint)
    use_posterior = config.get("model_config", {}).get("use_posterior", True)

    # ---- Resolve fold file lists --------------------------------------- #
    fold_datasets = get_fold_datasets(kfold_base_path, fold_id)

    # ---- Dataset construction kwargs from config ----------------------- #
    dl_cfg = config.get("dataset_config", {}).get("dataloader_config", {})
    ds_kwargs: Dict[str, Any] = dict(dl_cfg.get("dataset_kwargs", {}))
    stats_path = config.get("dataset_config", {}).get("stat_path")
    normalize_fields = dl_cfg.get("normalize_fields")
    segment_duration = dl_cfg.get("segment_duration", 1200.0)
    trim_minutes = ds_kwargs.pop("trim_minutes", None)

    os.makedirs(output_dir, exist_ok=True)
    output_paths: Dict[str, str] = {}

    for partition in ("train", "val", "test"):
        hdf5_files = fold_datasets.get(partition, [])
        if not hdf5_files:
            logger.warning(
                "No HDF5 files for fold {} partition '{}' — skipping.",
                fold_id,
                partition,
            )
            continue

        output_path = os.path.join(
            output_dir,
            f"precomputed_fold_{fold_id}_{partition}.hdf5",
        )

        logger.info(
            "Pre-computing latents: fold={}, partition={}, "
            "n_files={}, output={}",
            fold_id,
            partition,
            len(hdf5_files),
            output_path,
        )

        # ---- Create GUID-grouped sequence dataset ---------------------- #
        seq_ds = SignalSequenceDataset(
            segment_duration=segment_duration,
            guid_cache_size=0,  # No caching needed for one-pass iteration.
            paths=hdf5_files,
            stats_path=stats_path,
            normalize_fields=normalize_fields,
            trim_minutes=trim_minutes,
            **ds_kwargs,
        )

        total_segments = 0
        total_guids = len(seq_ds)

        with h5py.File(output_path, "w") as f:
            # -- Metadata ------------------------------------------------ #
            f.attrs["vae_checkpoint_path"] = vae_checkpoint
            f.attrs["vae_checkpoint_hash"] = vae_hash
            f.attrs["use_posterior"] = use_posterior
            f.attrs["creation_timestamp"] = datetime.now().isoformat()
            f.attrs["fold_id"] = fold_id
            f.attrs["partition"] = partition

            guids_grp = f.create_group("guids")

            for guid_idx in range(total_guids):
                sample = seq_ds[guid_idx]
                guid: str = sample["guid"]
                num_segs: int = sample["num_segments"]

                # VAE input tensors: (S_i, T, C) on GPU.
                fhr_st = sample["fhr_st"].to(device)
                fhr_ph = sample["fhr_ph"].to(device)
                fhr_up_ph = sample["fhr_up_ph"].to(device)

                # ---- Chunked VAE encoding ------------------------------ #
                mu_chunks: List[torch.Tensor] = []
                for start in range(0, num_segs, chunk_size):
                    end = min(start + chunk_size, num_segs)
                    with torch.no_grad():
                        enc = vae_model.encode_only(
                            y_st=fhr_st[start:end],
                            y_ph=fhr_ph[start:end],
                            x_ph=fhr_up_ph[start:end],
                            sample_z=False,
                        )
                    key = "mu_post" if use_posterior else "mu_prior"
                    mu_chunks.append(enc[key].cpu())

                mu_post = torch.cat(mu_chunks, dim=0)  # (S_i, 300, 16)

                # ---- Write per-GUID HDF5 group ------------------------- #
                g = guids_grp.create_group(guid)
                g.create_dataset(
                    "mu_post",
                    data=mu_post.numpy(),
                    dtype="float32",
                    compression="gzip",
                    compression_opts=4,
                )
                g.create_dataset(
                    "epochs",
                    data=sample["epoch"].numpy(),
                    dtype="float64",
                )

                total_segments += num_segs

                if (guid_idx + 1) % 100 == 0 or guid_idx + 1 == total_guids:
                    logger.info(
                        "  [{}/{}] GUIDs processed ({} segments so far)",
                        guid_idx + 1,
                        total_guids,
                        total_segments,
                    )

            # -- Final metadata ------------------------------------------ #
            f.attrs["total_segments"] = total_segments
            f.attrs["total_guids"] = total_guids

        logger.info(
            "Saved: {} GUIDs, {} segments → {}",
            total_guids,
            total_segments,
            output_path,
        )
        output_paths[partition] = output_path

    return output_paths


# ====================================================================== #
#  Dataset wrapper                                                         #
# ====================================================================== #


class PrecomputedLatentDataset(Dataset):
    """Wraps ``SignalSequenceDataset`` and injects pre-computed VAE latents.

    On each ``__getitem__`` call the underlying sequence dataset returns a
    GUID sample.  This wrapper additionally loads the pre-computed
    ``mu_post`` for that GUID from the HDF5 cache and adds it under the
    key ``'mu_post_precomputed'``.  The existing
    ``sequence_collate_fn`` pads this field to ``(B, S_max, 300, 16)``
    just like any other tensor field, and
    :meth:`TemporalVaeClassifier.forward` detects its presence and skips
    VAE encoding.

    Cache invalidation is enforced at init time by comparing the SHA-256
    hash of the VAE checkpoint against the hash stored in the HDF5 file.

    Note:
        The HDF5 file is re-opened in each DataLoader worker process via
        ``__getstate__``/``__setstate__`` to ensure multi-process safety.

    Args:
        precomputed_path: Path to the HDF5 file produced by
            :func:`precompute_fold_latents`.
        vae_checkpoint: Path to the VAE ``.ckpt`` file for hash
            validation.  Pass ``None`` to skip validation (not
            recommended for production).
        segment_duration: Forwarded to ``SignalSequenceDataset``.
        guid_cache_size: Forwarded to ``SignalSequenceDataset``.
        **dataset_kwargs: All remaining kwargs are forwarded to
            ``SignalSequenceDataset`` (and in turn to
            ``CombinedHDF5Dataset``).  Must include ``paths`` or
            ``hdf5_files``.

    Raises:
        ValueError: If the VAE checkpoint hash does not match the hash
            stored in the pre-computed HDF5 file.
        FileNotFoundError: If ``precomputed_path`` does not exist.
    """

    def __init__(
        self,
        precomputed_path: str,
        *,
        vae_checkpoint: Optional[str] = None,
        segment_duration: float = 1200.0,
        guid_cache_size: int = 128,
        **dataset_kwargs: Any,
    ) -> None:
        from hdf5_dataset.guid_hdf5_dataset import SignalSequenceDataset

        self._precomputed_path = precomputed_path
        self._vae_checkpoint = vae_checkpoint

        # Validate cache hash FIRST (before expensive dataset creation).
        self._h5_file: Optional[h5py.File] = None
        self._open_and_validate()

        # Underlying sequence dataset (returns per-GUID samples).
        self.seq_dataset = SignalSequenceDataset(
            segment_duration=segment_duration,
            guid_cache_size=guid_cache_size,
            **dataset_kwargs,
        )

    def _open_and_validate(self) -> None:
        """Open the HDF5 file and validate the VAE checkpoint hash.

        Raises:
            ValueError: On hash mismatch.
        """
        self._h5_file = h5py.File(self._precomputed_path, "r")

        if self._vae_checkpoint is not None:
            stored_hash = str(
                self._h5_file.attrs.get("vae_checkpoint_hash", "")
            )
            expected_hash = _compute_file_hash(self._vae_checkpoint)
            if stored_hash != expected_hash:
                self._h5_file.close()
                self._h5_file = None
                raise ValueError(
                    f"VAE checkpoint hash mismatch!  "
                    f"Stored: {stored_hash[:16]}…  "
                    f"Expected: {expected_hash[:16]}…  "
                    f"The pre-computed latents at '{self._precomputed_path}' "
                    f"were generated with a different VAE checkpoint.  "
                    f"Re-run precompute_fold_latents() with the current "
                    f"checkpoint."
                )

        logger.info(
            "Loaded precomputed latents: {} GUIDs, {} segments from {}",
            self._h5_file.attrs.get("total_guids", "?"),
            self._h5_file.attrs.get("total_segments", "?"),
            self._precomputed_path,
        )

    # ---- Pickling (DataLoader multi-process support) ------------------- #

    def __getstate__(self) -> Dict[str, Any]:
        """Prepare state for pickling (close HDF5 handle)."""
        state = self.__dict__.copy()
        state["_h5_file"] = None  # h5py.File is not picklable.
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state after unpickling (reopen HDF5 in worker)."""
        self.__dict__.update(state)
        self._open_and_validate()

    def _get_h5(self) -> h5py.File:
        """Return the open HDF5 handle, reopening if needed."""
        if self._h5_file is None:
            self._open_and_validate()
        assert self._h5_file is not None
        return self._h5_file

    # ---- Dataset interface --------------------------------------------- #

    def __len__(self) -> int:
        """Number of GUIDs in the dataset."""
        return len(self.seq_dataset)

    def __getitem__(self, idx: int) -> Any:
        """Return a GUID sample with pre-computed ``mu_post`` injected.

        The returned dict has the same keys as
        ``SignalSequenceDataset.__getitem__`` plus
        ``'mu_post_precomputed'`` of shape ``(S_i, 300, 16)``.

        Args:
            idx: GUID index (0-based).

        Returns:
            ``AttributeDict`` with ``mu_post_precomputed`` added.
        """
        sample = self.seq_dataset[idx]
        guid: str = sample["guid"]

        h5 = self._get_h5()
        guid_grp = h5["guids"][guid]

        stored_mu = torch.from_numpy(guid_grp["mu_post"][:])  # (S, 300, 16)
        stored_epochs = guid_grp["epochs"][:]  # (S,)

        sample_epochs = sample["epoch"].numpy()  # (S_i,)

        # Fast path: lengths match and epochs are aligned.
        if len(stored_mu) == len(sample_epochs) and np.allclose(
            stored_epochs, sample_epochs, atol=0.1
        ):
            sample["mu_post_precomputed"] = stored_mu
        else:
            # Slow path: epoch-based alignment for robustness.
            epoch_to_idx = {
                round(float(e), 1): i for i, e in enumerate(stored_epochs)
            }
            aligned: List[torch.Tensor] = []
            for e in sample_epochs:
                key = round(float(e), 1)
                if key not in epoch_to_idx:
                    raise KeyError(
                        f"Epoch {e} for GUID '{guid}' not found in "
                        f"precomputed latents.  Available: "
                        f"{stored_epochs.tolist()}"
                    )
                aligned.append(stored_mu[epoch_to_idx[key]])
            sample["mu_post_precomputed"] = torch.stack(aligned)

        return sample

    # ---- Forwarded properties & methods -------------------------------- #

    @property
    def guid_lengths(self) -> List[int]:
        """Number of segments per GUID (delegates to inner dataset)."""
        return self.seq_dataset.guid_lengths

    def get_guid_list(self) -> List[str]:
        """Return the sorted GUID list (delegates to inner dataset)."""
        return self.seq_dataset.get_guid_list()

    def get_stats(self) -> Dict[str, Any]:
        """Return dataset statistics augmented with precomputed info."""
        stats = self.seq_dataset.get_stats()
        stats["precomputed_path"] = self._precomputed_path
        h5 = self._get_h5()
        stats["precomputed_total_guids"] = int(
            h5.attrs.get("total_guids", 0)
        )
        stats["precomputed_total_segments"] = int(
            h5.attrs.get("total_segments", 0)
        )
        return stats


# ====================================================================== #
#  DataLoader factory                                                      #
# ====================================================================== #


def create_precomputed_sequence_dataloader(
    precomputed_path: str,
    hdf5_files: List[str],
    batch_size: int = 8,
    bucket_ranges: Optional[List[List[int]]] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    segment_duration: float = 1200.0,
    guid_cache_size: int = 128,
    stats_path: Optional[str] = None,
    normalize_fields: Optional[Sequence[str]] = None,
    trim_minutes: Optional[float] = None,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    seed: int = 42,
    vae_checkpoint: Optional[str] = None,
    **dataset_kwargs: Any,
) -> Tuple[DataLoader, PrecomputedLatentDataset]:
    """Create a DataLoader with pre-computed latents and bucket sampling.

    Mirrors :func:`~length_bucket_sampler.create_bucketed_sequence_dataloader`
    but wraps the underlying ``SignalSequenceDataset`` with
    :class:`PrecomputedLatentDataset` so that each batch contains a
    ``mu_post_precomputed`` tensor.

    Args:
        precomputed_path: Path to the HDF5 file from
            :func:`precompute_fold_latents`.
        hdf5_files: Paths to the original HDF5 dataset files.
        batch_size: Number of GUIDs per batch.
        bucket_ranges: Bucket ``[lo, hi]`` ranges for the sampler.
        shuffle: Whether to shuffle within buckets.
        num_workers: DataLoader worker processes.
        segment_duration: Segment duration for positional indices.
        guid_cache_size: GUID-level cache capacity.
        stats_path: HDF5 stats file for normalisation.
        normalize_fields: Fields to normalise.
        trim_minutes: Minutes to trim from each segment end.
        prefetch_factor: DataLoader prefetch factor (workers > 0 only).
        pin_memory: Pin memory for faster GPU transfer.
        seed: Random seed for the bucket sampler.
        vae_checkpoint: Path to VAE ``.ckpt`` for hash validation.
        **dataset_kwargs: Extra kwargs forwarded to
            ``CombinedHDF5Dataset``.

    Returns:
        Tuple of ``(DataLoader, PrecomputedLatentDataset)``.
    """
    from hdf5_dataset.guid_hdf5_dataset import sequence_collate_fn
    from model.vae_teb_prediction.guid_classifier.length_bucket_sampler import (
        LengthBucketSampler,
    )

    # ---- Build dataset ------------------------------------------------- #
    ds_kwargs: Dict[str, Any] = {}
    if stats_path is not None:
        ds_kwargs["stats_path"] = stats_path
    if normalize_fields is not None:
        ds_kwargs["normalize_fields"] = normalize_fields
    if trim_minutes is not None:
        ds_kwargs["trim_minutes"] = trim_minutes
    ds_kwargs.update(dataset_kwargs)

    dataset = PrecomputedLatentDataset(
        precomputed_path=precomputed_path,
        vae_checkpoint=vae_checkpoint,
        segment_duration=segment_duration,
        guid_cache_size=guid_cache_size,
        paths=hdf5_files,
        pin_memory=pin_memory,
        **ds_kwargs,
    )

    # ---- Build sampler ------------------------------------------------- #
    sampler = LengthBucketSampler(
        lengths=dataset.guid_lengths,
        batch_size=batch_size,
        bucket_ranges=bucket_ranges,
        shuffle=shuffle,
        seed=seed,
    )

    # ---- Build DataLoader ---------------------------------------------- #
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,  # Sampler handles ordering.
        num_workers=num_workers,
        collate_fn=sequence_collate_fn,
        drop_last=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=False,  # Pinning managed by inner dataset.
    )

    return dataloader, dataset


# ====================================================================== #
#  CLI entry point                                                         #
# ====================================================================== #


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pre-computation script."""
    parser = argparse.ArgumentParser(
        description="Pre-compute VAE latents for the temporal classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to config_temporal.yaml.  If not given, looks for the "
            "file next to this script."
        ),
    )
    parser.add_argument(
        "--fold_ids",
        type=int,
        nargs="+",
        default=None,
        help="Fold IDs to pre-compute (e.g. 1 2 3).  Defaults to all.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output directory for HDF5 files.  Defaults to "
            "<out_dir_base>/precomputed_latents/."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device for VAE encoding.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=32,
        help="Segments per VAE forward pass chunk.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point: pre-compute VAE latents for specified folds."""
    import yaml

    args = _parse_args()

    # ---- Locate and load config ---------------------------------------- #
    if args.config is not None:
        config_path = args.config
    else:
        config_path = str(
            Path(__file__).parent / "config_temporal.yaml"
        )

    logger.info("Loading config from: {}", config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # ---- Resolve paths from config ------------------------------------- #
    ds_cfg = config.get("dataset_config", {})
    kfold_base_path = ds_cfg.get("kfold_base_path", "")
    num_folds = ds_cfg.get("num_folds", 10)
    vae_checkpoint = config.get("model_config", {}).get("vae_checkpoint", "")

    fold_ids = args.fold_ids
    if fold_ids is None:
        fold_ids_cfg = ds_cfg.get("fold_ids")
        if fold_ids_cfg is not None:
            fold_ids = fold_ids_cfg
        else:
            fold_ids = list(range(1, num_folds + 1))

    output_dir = args.output_dir
    if output_dir is None:
        out_base = (
            config.get("general_config", {})
            .get("folders_config", {})
            .get("out_dir_base", ".")
        )
        output_dir = os.path.join(out_base, "precomputed_latents")

    # ---- Pre-compute each fold ----------------------------------------- #
    logger.info(
        "Pre-computing latents for folds {} on device {}",
        fold_ids,
        args.device,
    )

    all_results: Dict[int, Dict[str, str]] = {}
    for fold_id in fold_ids:
        fold_output = os.path.join(output_dir, f"fold_{fold_id}")
        result = precompute_fold_latents(
            fold_id=fold_id,
            kfold_base_path=kfold_base_path,
            vae_checkpoint=vae_checkpoint,
            output_dir=fold_output,
            config=config,
            device=args.device,
            chunk_size=args.chunk_size,
        )
        all_results[fold_id] = result

    # ---- Print summary ------------------------------------------------- #
    logger.info("=" * 60)
    logger.info("Pre-computation complete!")
    logger.info("=" * 60)
    for fold_id, paths in all_results.items():
        logger.info("Fold {}:", fold_id)
        for partition, path in paths.items():
            size_mb = os.path.getsize(path) / (1024 * 1024)
            logger.info("  {} → {} ({:.1f} MB)", partition, path, size_mb)


if __name__ == "__main__":
    main()
