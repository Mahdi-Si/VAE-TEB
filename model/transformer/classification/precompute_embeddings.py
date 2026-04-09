"""Pre-compute and cache transformer segment embeddings for fast training.

This module provides utilities to pre-compute the frozen transformer's
416-dim segment embeddings for every segment in the dataset and cache the
results to HDF5 files on disk.  During training the cached embeddings are
loaded directly, bypassing the expensive transformer forward pass.

Embedding composition (per segment)::

    e = [mean_pool(H_F)(192) | mean_pool(H_FU)(192) |
         mean(TE_mus)(16) | std(TE_mus)(16)]  = 416

Components:

- :func:`precompute_fold_embeddings` — Pre-compute embeddings for one
  fold (train/val/test partitions).
- :class:`PrecomputedEmbeddingDataset` — Dataset wrapper that injects
  ``embeddings_precomputed`` into each sample.
- :func:`create_precomputed_embedding_dataloader` — Factory that builds
  a ``DataLoader`` with pre-computed embeddings and bucket sampling.

Cache invalidation is handled by storing a SHA-256 hash of the
transformer checkpoint inside the pre-computed HDF5 file.

Example — pre-compute from CLI::

    python -m model.transformer.classification.precompute_embeddings \\
        --config model/transformer/classification/config_classification.yaml \\
        --fold_ids 1 2 3 \\
        --output_dir /data/precomputed_embeddings \\
        --device cuda:0

Example — use in training::

    from model.transformer.classification.precompute_embeddings import (
        create_precomputed_embedding_dataloader,
    )

    loader, dataset = create_precomputed_embedding_dataloader(
        precomputed_path="precomputed_fold_1_train.hdf5",
        hdf5_files=train_hdf5_files,
        batch_size=8,
    )
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


def precompute_fold_embeddings(
    fold_id: int,
    kfold_base_path: str,
    transformer_checkpoint: str,
    output_dir: str,
    config: Dict[str, Any],
    device: str = "cuda:0",
    chunk_size: int = 16,
) -> Dict[str, str]:
    """Pre-compute transformer segment embeddings for one fold.

    For each partition (train / val / test):

    1. Loads segments via :class:`SignalSequenceDataset` (GUID-grouped).
    2. Runs the frozen transformer in chunks of ``chunk_size`` segments.
    3. Extracts 416-dim embeddings: [pool(H_F) | pool(H_FU) |
       mean_TE | std_TE].
    4. Saves per-GUID embeddings to an HDF5 file with metadata for
       cache invalidation.

    The output HDF5 layout::

        precomputed_fold_{fold_id}_{partition}.hdf5
        ├── attrs: transformer_checkpoint_hash, d_embedding, ...
        └── guids/
            ├── GUID_001/
            │   ├── embeddings  (S_1, 416)  float32
            │   └── epochs      (S_1,)      float64
            └── ...

    Args:
        fold_id: Fold number (1-based, e.g. 1-10).
        kfold_base_path: Root directory of the k-fold dataset.
        transformer_checkpoint: Path to the pre-trained transformer
            checkpoint file.
        output_dir: Directory where HDF5 output files are saved.
        config: Full config dict from ``config_classification.yaml``.
        device: CUDA device string (e.g. ``'cuda:0'``).
        chunk_size: Number of segments processed per forward pass.

    Returns:
        Dict mapping partition names to generated HDF5 file paths.
    """
    from hdf5_dataset.guid_hdf5_dataset import SignalSequenceDataset
    from model.transformer.model.config import TransformerConfig
    from model.transformer.model.model import CausalMultimodalTransformer
    from model.transformer.tr_testing.base import TransformerTestRunner
    from model.vae_teb_prediction.kfold_classifier_trainer import (
        get_fold_datasets,
    )
    from train.graph_models_utils import load_checkpoint_strict

    # ---- Load and freeze transformer ------------------------------------- #
    logger.info("Loading transformer checkpoint: {}", transformer_checkpoint)
    ckpt = torch.load(transformer_checkpoint, map_location="cpu",
                       weights_only=False)
    tr_config = TransformerTestRunner._extract_config(ckpt)
    transformer = CausalMultimodalTransformer(tr_config)
    load_checkpoint_strict(transformer, ckpt)
    transformer = transformer.to(device)
    transformer.eval()
    for param in transformer.parameters():
        param.requires_grad = False

    ckpt_hash = _compute_file_hash(transformer_checkpoint)

    # ---- Embedding settings from config ---------------------------------- #
    model_cfg = config.get("model_config", {})
    emb_cfg = model_cfg.get("segment_embedding", {})
    anchor_step = emb_cfg.get("anchor_step", 5)
    d_embedding = emb_cfg.get("d_embedding", 416)
    pooling = emb_cfg.get("pooling", "mean")

    # Build dense TE anchor grid.
    anchor_grid = torch.arange(
        tr_config.valid_anchor_start,
        tr_config.valid_anchor_end + 1,
        anchor_step,
        device=device,
    )
    K = anchor_grid.shape[0]

    # ---- Resolve fold file lists ----------------------------------------- #
    ds_cfg = config.get("dataset_config", {})
    test_mode = ds_cfg.get("test_mode", None)
    fold_datasets = get_fold_datasets(kfold_base_path, fold_id,
                                       test_mode=test_mode)

    # ---- Dataset construction kwargs from config ------------------------- #
    dl_cfg = ds_cfg.get("dataloader_config", {})
    ds_kwargs: Dict[str, Any] = dict(dl_cfg.get("dataset_kwargs", {}))
    stats_path = ds_cfg.get("stat_path")
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
                fold_id, partition,
            )
            continue

        output_path = os.path.join(
            output_dir,
            f"precomputed_fold_{fold_id}_{partition}.hdf5",
        )

        logger.info(
            "Pre-computing embeddings: fold={}, partition={}, "
            "n_files={}, anchor_step={}, output={}",
            fold_id, partition, len(hdf5_files), anchor_step, output_path,
        )

        # ---- Create GUID-grouped sequence dataset ------------------------ #
        seq_ds = SignalSequenceDataset(
            segment_duration=segment_duration,
            guid_cache_size=0,
            paths=hdf5_files,
            stats_path=stats_path,
            normalize_fields=normalize_fields,
            trim_minutes=trim_minutes,
            **ds_kwargs,
        )

        total_segments = 0
        total_guids = len(seq_ds)

        with h5py.File(output_path, "w") as f:
            # -- Metadata -------------------------------------------------- #
            f.attrs["transformer_checkpoint_path"] = transformer_checkpoint
            f.attrs["transformer_checkpoint_hash"] = ckpt_hash
            f.attrs["d_embedding"] = d_embedding
            f.attrs["anchor_step"] = anchor_step
            f.attrs["pooling"] = pooling
            f.attrs["creation_timestamp"] = datetime.now().isoformat()
            f.attrs["fold_id"] = fold_id
            f.attrs["partition"] = partition

            guids_grp = f.create_group("guids")

            for guid_idx in range(total_guids):
                sample = seq_ds[guid_idx]
                guid: str = sample["guid"]
                num_segs: int = sample["num_segments"]

                fhr_st = sample["fhr_st"].to(device)  # (S_i, 300, 43)
                up_st = sample["up_st"].to(device)     # (S_i, 300, 43)

                # ---- Chunked transformer encoding ------------------------ #
                emb_chunks: List[torch.Tensor] = []
                for start in range(0, num_segs, chunk_size):
                    end = min(start + chunk_size, num_segs)
                    Y = fhr_st[start:end]
                    U = up_st[start:end]
                    C = Y.shape[0]

                    grid = anchor_grid.unsqueeze(0).expand(C, -1)

                    with torch.no_grad():
                        outputs = transformer(Y, U, anchor_indices=grid)

                    H_F = outputs["H_F"]          # (C, 300, 192)
                    H_FU = outputs["H_FU"]        # (C, 300, 192)
                    mu_post = outputs["mu_post"]  # (C * K, 16)

                    # Pool H_F and H_FU (full-sequence).
                    s_F = H_F.mean(dim=1)         # (C, 192)
                    s_FU = H_FU.mean(dim=1)       # (C, 192)

                    # TE statistics from dense anchor grid.
                    te_mus = mu_post.view(C, K, -1)
                    mean_te = te_mus.mean(dim=1)  # (C, 16)
                    std_te = te_mus.std(dim=1)    # (C, 16)

                    emb = torch.cat(
                        [s_F, s_FU, mean_te, std_te], dim=-1
                    ).cpu()
                    emb_chunks.append(emb)

                embeddings = torch.cat(emb_chunks, dim=0)  # (S_i, 416)

                # ---- Write per-GUID HDF5 group --------------------------- #
                g = guids_grp.create_group(guid)
                g.create_dataset(
                    "embeddings",
                    data=embeddings.numpy(),
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
                        guid_idx + 1, total_guids, total_segments,
                    )

            # -- Final metadata -------------------------------------------- #
            f.attrs["total_segments"] = total_segments
            f.attrs["total_guids"] = total_guids

        logger.info(
            "Saved: {} GUIDs, {} segments → {}",
            total_guids, total_segments, output_path,
        )
        output_paths[partition] = output_path

    return output_paths


# ====================================================================== #
#  Dataset wrapper                                                         #
# ====================================================================== #


class PrecomputedEmbeddingDataset(Dataset):
    """Wraps ``SignalSequenceDataset`` and injects pre-computed embeddings.

    On each ``__getitem__`` call the underlying sequence dataset returns a
    GUID sample.  This wrapper additionally loads the pre-computed 416-dim
    embedding for that GUID from the HDF5 cache and adds it under the key
    ``'embeddings_precomputed'``.  The existing ``sequence_collate_fn``
    pads this field to ``(B, S_max, 416)`` automatically, and
    ``TimeAwareGRUClassifier.forward()`` detects its presence and skips
    transformer encoding.

    Cache invalidation is enforced at init time by comparing the SHA-256
    hash of the transformer checkpoint against the hash stored in the HDF5.

    Note:
        The HDF5 file is re-opened in each DataLoader worker process via
        ``__getstate__``/``__setstate__`` to ensure multi-process safety.

    Args:
        precomputed_path: Path to the HDF5 file produced by
            :func:`precompute_fold_embeddings`.
        transformer_checkpoint: Path to the transformer checkpoint for
            hash validation.  Pass ``None`` to skip validation.
        segment_duration: Forwarded to ``SignalSequenceDataset``.
        guid_cache_size: Forwarded to ``SignalSequenceDataset``.
        **dataset_kwargs: All remaining kwargs are forwarded to
            ``SignalSequenceDataset``.
    """

    def __init__(
        self,
        precomputed_path: str,
        *,
        transformer_checkpoint: Optional[str] = None,
        segment_duration: float = 1200.0,
        guid_cache_size: int = 128,
        **dataset_kwargs: Any,
    ) -> None:
        from hdf5_dataset.guid_hdf5_dataset import SignalSequenceDataset

        self._precomputed_path = precomputed_path
        self._transformer_checkpoint = transformer_checkpoint

        # Validate cache hash FIRST.
        self._h5_file: Optional[h5py.File] = None
        self._open_and_validate()

        # Underlying sequence dataset.
        self.seq_dataset = SignalSequenceDataset(
            segment_duration=segment_duration,
            guid_cache_size=guid_cache_size,
            **dataset_kwargs,
        )

    def _open_and_validate(self) -> None:
        """Open the HDF5 file and validate the checkpoint hash.

        Raises:
            ValueError: On hash mismatch.
        """
        self._h5_file = h5py.File(self._precomputed_path, "r")

        if self._transformer_checkpoint is not None:
            stored_hash = str(
                self._h5_file.attrs.get("transformer_checkpoint_hash", "")
            )
            expected_hash = _compute_file_hash(self._transformer_checkpoint)
            if stored_hash != expected_hash:
                self._h5_file.close()
                self._h5_file = None
                raise ValueError(
                    f"Transformer checkpoint hash mismatch!  "
                    f"Stored: {stored_hash[:16]}…  "
                    f"Expected: {expected_hash[:16]}…  "
                    f"The pre-computed embeddings at "
                    f"'{self._precomputed_path}' were generated with a "
                    f"different checkpoint.  Re-run "
                    f"precompute_fold_embeddings() with the current "
                    f"checkpoint."
                )

        logger.info(
            "Loaded precomputed embeddings: {} GUIDs, {} segments from {}",
            self._h5_file.attrs.get("total_guids", "?"),
            self._h5_file.attrs.get("total_segments", "?"),
            self._precomputed_path,
        )

    # ---- Pickling (DataLoader multi-process support) --------------------- #

    def __getstate__(self) -> Dict[str, Any]:
        """Prepare state for pickling (close HDF5 handle)."""
        state = self.__dict__.copy()
        state["_h5_file"] = None
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

    # ---- Dataset interface ----------------------------------------------- #

    def __len__(self) -> int:
        """Number of GUIDs in the dataset."""
        return len(self.seq_dataset)

    def __getitem__(self, idx: int) -> Any:
        """Return a GUID sample with pre-computed embeddings injected.

        The returned dict has the same keys as
        ``SignalSequenceDataset.__getitem__`` plus
        ``'embeddings_precomputed'`` of shape ``(S_i, 416)``.

        Args:
            idx: GUID index (0-based).

        Returns:
            ``AttributeDict`` with ``embeddings_precomputed`` added.
        """
        sample = self.seq_dataset[idx]
        guid: str = sample["guid"]

        h5 = self._get_h5()
        guid_grp = h5["guids"][guid]

        stored_emb = torch.from_numpy(
            guid_grp["embeddings"][:]
        )                                           # (S, 416)
        stored_epochs = guid_grp["epochs"][:]       # (S,)

        sample_epochs = sample["epoch"].numpy()     # (S_i,)

        # Fast path: lengths and epochs are aligned.
        if len(stored_emb) == len(sample_epochs) and np.allclose(
            stored_epochs, sample_epochs, atol=0.1
        ):
            sample["embeddings_precomputed"] = stored_emb
        else:
            # Slow path: epoch-based alignment for robustness.
            epoch_to_idx = {
                round(float(e), 1): i
                for i, e in enumerate(stored_epochs)
            }
            aligned: List[torch.Tensor] = []
            for e in sample_epochs:
                key = round(float(e), 1)
                if key not in epoch_to_idx:
                    raise KeyError(
                        f"Epoch {e} for GUID '{guid}' not found in "
                        f"precomputed embeddings.  Available: "
                        f"{stored_epochs.tolist()}"
                    )
                aligned.append(stored_emb[epoch_to_idx[key]])
            sample["embeddings_precomputed"] = torch.stack(aligned)

        return sample

    # ---- Forwarded properties & methods ---------------------------------- #

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

    def estimate_class_weights(self, num_classes: int = 2):
        """Delegate class weight estimation to the inner dataset."""
        return self.seq_dataset.estimate_class_weights(
            num_classes=num_classes
        )

    def clear_cache(self) -> None:
        """Clear the inner dataset's cache."""
        self.seq_dataset.clear_cache()


# ====================================================================== #
#  DataLoader factory                                                      #
# ====================================================================== #


def create_precomputed_embedding_dataloader(
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
    transformer_checkpoint: Optional[str] = None,
    **dataset_kwargs: Any,
) -> Tuple[DataLoader, PrecomputedEmbeddingDataset]:
    """Create a DataLoader with pre-computed embeddings and bucket sampling.

    Mirrors :func:`~length_bucket_sampler.create_bucketed_sequence_dataloader`
    but wraps the underlying ``SignalSequenceDataset`` with
    :class:`PrecomputedEmbeddingDataset` so that each batch contains an
    ``embeddings_precomputed`` tensor.

    Args:
        precomputed_path: Path to the HDF5 file from
            :func:`precompute_fold_embeddings`.
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
        prefetch_factor: DataLoader prefetch factor.
        pin_memory: Pin memory for faster GPU transfer.
        seed: Random seed for the bucket sampler.
        transformer_checkpoint: Path to checkpoint for hash validation.
        **dataset_kwargs: Extra kwargs forwarded to the dataset.

    Returns:
        Tuple of ``(DataLoader, PrecomputedEmbeddingDataset)``.
    """
    from hdf5_dataset.guid_hdf5_dataset import sequence_collate_fn
    from model.vae_teb_prediction.guid_classifier.length_bucket_sampler import (
        LengthBucketSampler,
    )

    # ---- Build dataset --------------------------------------------------- #
    ds_kwargs: Dict[str, Any] = {}
    if stats_path is not None:
        ds_kwargs["stats_path"] = stats_path
    if normalize_fields is not None:
        ds_kwargs["normalize_fields"] = normalize_fields
    if trim_minutes is not None:
        ds_kwargs["trim_minutes"] = trim_minutes
    ds_kwargs.update(dataset_kwargs)

    dataset = PrecomputedEmbeddingDataset(
        precomputed_path=precomputed_path,
        transformer_checkpoint=transformer_checkpoint,
        segment_duration=segment_duration,
        guid_cache_size=guid_cache_size,
        paths=hdf5_files,
        pin_memory=pin_memory,
        **ds_kwargs,
    )

    # ---- Build sampler --------------------------------------------------- #
    sampler = LengthBucketSampler(
        lengths=dataset.guid_lengths,
        batch_size=batch_size,
        bucket_ranges=bucket_ranges,
        shuffle=shuffle,
        seed=seed,
    )

    # ---- Build DataLoader ------------------------------------------------ #
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=sequence_collate_fn,
        drop_last=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=False,
    )

    return dataloader, dataset


# ====================================================================== #
#  CLI entry point                                                         #
# ====================================================================== #


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pre-computation script."""
    parser = argparse.ArgumentParser(
        description=(
            "Pre-compute transformer segment embeddings for the "
            "classification pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help=(
            "Path to config_classification.yaml.  If not given, looks "
            "for the file next to this script."
        ),
    )
    parser.add_argument(
        "--fold_ids", type=int, nargs="+", default=None,
        help="Fold IDs to pre-compute (e.g. 1 2 3).  Defaults to all.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help=(
            "Output directory for HDF5 files.  Defaults to "
            "<out_dir_base>/precomputed_embeddings/."
        ),
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="CUDA device for transformer encoding.",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=16,
        help="Segments per transformer forward pass chunk.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point: pre-compute embeddings for specified folds."""
    import yaml

    args = _parse_args()

    # ---- Locate and load config ------------------------------------------ #
    if args.config is not None:
        config_path = args.config
    else:
        config_path = str(
            Path(__file__).parent / "config_classification.yaml"
        )

    logger.info("Loading config from: {}", config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # ---- Resolve paths from config --------------------------------------- #
    ds_cfg = config.get("dataset_config", {})
    kfold_base_path = ds_cfg.get("kfold_base_path", "")
    num_folds = ds_cfg.get("num_folds", 10)
    transformer_checkpoint = (
        config.get("model_config", {}).get("transformer_checkpoint", "")
    )

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
        output_dir = os.path.join(out_base, "precomputed_embeddings")

    # ---- Pre-compute each fold ------------------------------------------- #
    logger.info(
        "Pre-computing embeddings for folds {} on device {}",
        fold_ids, args.device,
    )

    for fid in fold_ids:
        precompute_fold_embeddings(
            fold_id=fid,
            kfold_base_path=kfold_base_path,
            transformer_checkpoint=transformer_checkpoint,
            output_dir=output_dir,
            config=config,
            device=args.device,
            chunk_size=args.chunk_size,
        )

    logger.info("All folds pre-computed successfully.")


if __name__ == "__main__":
    main()
