r"""``SyntheticRawDataModuleV4`` -- the (DDP-capable) datamodule over the ``synthetic_v4`` raw cache.

Wraps the cached raw splits written by :func:`build_dataset_v4.build_all_v4` for training and
evaluation of :class:`SeqVaeRawV4`, mirroring :class:`datamodule_v2.SyntheticTEDataModuleV2` but for
the raw (untrimmed $5280$) fields. The base trainer's ``train_model`` *receives* its loaders as
arguments, so handing it this datamodule's loaders needs no ``model_raw`` change (Sprint 4).

The v4 cache is already globally z-scored at build time, so this datamodule is a **pure cache
reader**: it does **not** apply ``dataloader_config.normalize_fields`` (that is a real-HDF5-loader
knob; re-normalising here would double-normalise). ``norm_stats.npz`` exists only for overlay
denorm in grading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import lightning as pl
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.dataset_v4 import (
    SyntheticRawDatasetV4,
    attribute_dict_collate,
    make_dataloader_v4,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import resolve_cache_dir


class SyntheticRawDataModuleV4(pl.LightningDataModule):
    r"""Serve the cached ``synthetic_v4`` raw splits as ``model_raw``-contract batches.

    Args:
        config: The (benchmark-resolved) config dict. Reads ``experiment.{benchmark,tag,data_tag}``,
            ``paths.data_dir`` and the ``dataset_config.dataloader_config`` block
            (``num_workers`` / ``dataset_kwargs.pin_memory``).
        batch_size: **Per-GPU** batch size (under DDP the global batch is
            ``batch_size * world_size``).
        benchmark: Active benchmark key (defaults to ``experiment.benchmark``).
        cache_dir: Optional explicit cache directory (overrides :func:`resolve_cache_dir`); useful
            for tests that point at a fixture cache.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        batch_size: int,
        benchmark: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._batch_size = int(batch_size)
        self._benchmark = str(
            benchmark if benchmark is not None
            else config.get("experiment", {}).get("benchmark", "G1_raw_v4")
        )
        self._cache_dir: Path = (
            Path(cache_dir) if cache_dir is not None
            else resolve_cache_dir(config, benchmark=self._benchmark)
        )

        # model_raw 4-key schema: loader knobs live under dataset_config.dataloader_config (NOT the
        # v2/v3 top-level ``dataset`` key). ``normalize_fields`` is deliberately ignored (the cache
        # is already normalised); ``persistent_workers`` / ``mmap`` are not in the schema, so they
        # take safe defaults.
        dl_cfg = (config.get("dataset_config") or {}).get("dataloader_config") or {}
        ds_kwargs = dl_cfg.get("dataset_kwargs") or {}
        self._num_workers = int(dl_cfg.get("num_workers", 0))
        self._pin_memory = bool(ds_kwargs.get("pin_memory", False))
        self._persistent_workers = bool(dl_cfg.get("persistent_workers", False))
        self._mmap = dl_cfg.get("mmap", "auto")

        self._train_ds: Optional[SyntheticRawDatasetV4] = None
        self._val_ds: Optional[SyntheticRawDatasetV4] = None
        self._test_ds: Optional[SyntheticRawDatasetV4] = None
        self._data_meta: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        r"""Open the train / optional val / optional test splits on every rank.

        Raises:
            FileNotFoundError: If ``train.npz`` is missing (with a build hint).
        """
        train_npz = self._cache_dir / "train.npz"
        val_npz = self._cache_dir / "val.npz"
        test_npz = self._cache_dir / "test.npz"
        if not train_npz.is_file():
            raise FileNotFoundError(
                f"cached raw dataset not found: {train_npz}\n"
                f"Build it first, e.g.:\n"
                f"  python -m ...synthetic_v2.run_pipeline_v4 --stage build --pilot"
            )
        self._train_ds = SyntheticRawDatasetV4(train_npz, mmap=self._mmap)
        self._data_meta = dict(self._train_ds.meta)
        if val_npz.is_file():
            self._val_ds = SyntheticRawDatasetV4(val_npz, mmap=self._mmap)
        if test_npz.is_file():
            self._test_ds = SyntheticRawDatasetV4(test_npz, mmap=self._mmap)

    def train_dataloader(self) -> DataLoader:
        r"""Shuffled training loader; Lightning injects the DistributedSampler under DDP."""
        assert self._train_ds is not None, "call setup() first"
        return make_dataloader_v4(
            self._train_ds, self._batch_size, shuffle=True, drop_last=True,
            num_workers=self._num_workers, pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        r"""Ordered validation loader, or ``None`` when no ``val.npz`` is cached."""
        if self._val_ds is None:
            return None
        return make_dataloader_v4(
            self._val_ds, self._batch_size, shuffle=False, drop_last=False,
            num_workers=self._num_workers, pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        r"""Ordered test loader, or ``None`` when no ``test.npz`` is cached."""
        if self._test_ds is None:
            return None
        return make_dataloader_v4(
            self._test_ds, self._batch_size, shuffle=False, drop_last=False,
            num_workers=self._num_workers, pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
        )

    # ------------------------------------------------------------------
    # Latent-stats pass (run outside the Trainer)
    # ------------------------------------------------------------------
    @property
    def data_meta(self) -> Dict[str, Any]:
        r"""The training split's ``meta.json`` (analytic TE, seeds, per-cell manifest)."""
        return self._data_meta

    def make_plain_train_loader(self) -> DataLoader:
        r"""Full-train, unshuffled loader for a post-fit latent-stats pass.

        When a process group is initialised, the loader is sharded with an explicit
        ``DistributedSampler(shuffle=False, drop_last=False)``; otherwise a plain ordered loader.

        .. note::
            With ``drop_last=False`` and ``len(dataset)`` not divisible by ``world_size``,
            :class:`~torch.utils.data.distributed.DistributedSampler` pads the last shard by
            repeating a few early indices, so up to ``world_size - 1`` rows are served on two ranks.
            A cross-rank latent-stats aggregation must dedup by ``guid`` (or use ``drop_last=True``).
        """
        assert self._train_ds is not None, "call setup() first"
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(self._train_ds, shuffle=False, drop_last=False)
            persistent = self._persistent_workers and self._num_workers > 0
            return DataLoader(
                self._train_ds,
                batch_size=self._batch_size,
                sampler=sampler,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
                persistent_workers=persistent,
                drop_last=False,
                collate_fn=attribute_dict_collate,
            )
        return make_dataloader_v4(
            self._train_ds, self._batch_size, shuffle=False, drop_last=False,
            num_workers=self._num_workers, pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
        )
