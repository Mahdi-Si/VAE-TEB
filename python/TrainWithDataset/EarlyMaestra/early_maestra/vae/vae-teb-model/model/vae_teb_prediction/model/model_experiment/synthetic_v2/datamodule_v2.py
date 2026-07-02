r"""Lightning ``DataModule`` over the cached ``synthetic_v2`` ``.npz`` benchmark.

Copied/adapted from ``synthetic/datamodule_synth.py`` (v2 owns the code). It wraps the
local :class:`dataset_v2.SyntheticTEDatasetV2` + :func:`dataset_v2.make_dataloader` in
the :class:`lightning.pytorch.LightningDataModule` contract so a Lightning
:class:`~lightning.pytorch.Trainer` built with ``use_distributed_sampler=True`` shards
each epoch across ranks automatically (the Sprint 5 training path). Two loaders are
exposed:

* :meth:`train_dataloader` / :meth:`val_dataloader` -- shuffled / ordered loaders. Under
  DDP, Lightning replaces their samplers with a ``DistributedSampler`` (preserving the
  ``shuffle`` flag); we deliberately do **not** add one here to avoid double-wrapping.
* :meth:`make_plain_train_loader` -- a full-train, unshuffled loader with an *explicit*
  ``DistributedSampler(shuffle=False, drop_last=False)`` for a post-fit latent-stats
  pass run outside the Trainer (so the sampler must be constructed here).

The cache directory is resolved by :func:`build_dataset_v2.resolve_cache_dir`
(``<data_dir>/<benchmark>/<tag>``), the same convention the build writes to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import lightning as pl
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .build_dataset_v2 import resolve_cache_dir
from .dataset_v2 import (
    SyntheticTEDatasetV2,
    attribute_dict_collate,
    make_dataloader,
)


class SyntheticTEDataModuleV2(pl.LightningDataModule):
    r"""Wrap the cached ``synthetic_v2`` splits for (DDP-capable) training.

    Args:
        config: The (benchmark-resolved) config dict. Reads
            ``experiment.{benchmark,tag}``, ``paths.data_dir`` and the optional
            ``dataset`` block (``num_workers`` / ``pin_memory`` /
            ``persistent_workers`` / ``mmap``).
        batch_size: **Per-GPU** batch size (under DDP the global batch is
            ``batch_size * world_size``).
        benchmark: Active benchmark key (defaults to ``experiment.benchmark``).
    """

    def __init__(
        self, config: Dict[str, Any], *, batch_size: int, benchmark: Optional[str] = None
    ) -> None:
        super().__init__()
        self._config = config
        self._batch_size = int(batch_size)
        self._benchmark = str(
            benchmark
            if benchmark is not None
            else config.get("experiment", {}).get("benchmark", "G1_raw")
        )
        self._cache_dir: Path = resolve_cache_dir(config, benchmark=self._benchmark)

        ds_cfg = config.get("dataset") or {}
        self._num_workers = int(ds_cfg.get("num_workers", 0))
        self._pin_memory = bool(ds_cfg.get("pin_memory", False))
        self._persistent_workers = bool(ds_cfg.get("persistent_workers", False))
        self._mmap = ds_cfg.get("mmap", "auto")

        self._train_ds: Optional[SyntheticTEDatasetV2] = None
        self._val_ds: Optional[SyntheticTEDatasetV2] = None
        self._test_ds: Optional[SyntheticTEDatasetV2] = None
        self._data_meta: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        r"""Open the train / optional val / optional test splits on every rank.

        With the default ``dataset.mmap: auto`` each rank (and DataLoader worker) maps
        the **same** uncompressed ``.npz`` regions, so the OS page cache holds one
        physical copy of the pool for the whole box (the multi-rank DDP RAM fix).

        Raises:
            FileNotFoundError: If ``train.npz`` is missing (with a build hint).
        """
        train_npz = self._cache_dir / "train.npz"
        val_npz = self._cache_dir / "val.npz"
        test_npz = self._cache_dir / "test.npz"
        if not train_npz.is_file():
            raise FileNotFoundError(
                f"cached dataset not found: {train_npz}\n"
                f"Build it first, e.g.:\n"
                f"  python .../synthetic_v2/run_pipeline_v2.py --stage build --full"
            )
        self._train_ds = SyntheticTEDatasetV2(train_npz, mmap=self._mmap)
        self._data_meta = dict(self._train_ds.meta)
        if val_npz.is_file():
            self._val_ds = SyntheticTEDatasetV2(val_npz, mmap=self._mmap)
        if test_npz.is_file():
            self._test_ds = SyntheticTEDatasetV2(test_npz, mmap=self._mmap)

    def train_dataloader(self) -> DataLoader:
        """Shuffled training loader; Lightning injects the DistributedSampler."""
        assert self._train_ds is not None, "call setup() first"
        return make_dataloader(
            self._train_ds, self._batch_size, shuffle=True, drop_last=True,
            num_workers=self._num_workers, pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Ordered validation loader, or ``None`` when no ``val.npz`` is cached."""
        if self._val_ds is None:
            return None
        return make_dataloader(
            self._val_ds, self._batch_size, shuffle=False, drop_last=False,
            num_workers=self._num_workers, pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Ordered test loader, or ``None`` when no ``test.npz`` is cached."""
        if self._test_ds is None:
            return None
        return make_dataloader(
            self._test_ds, self._batch_size, shuffle=False, drop_last=False,
            num_workers=self._num_workers, pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
        )

    # ------------------------------------------------------------------
    # Latent-stats pass (run outside the Trainer)
    # ------------------------------------------------------------------
    @property
    def data_meta(self) -> Dict[str, Any]:
        """The training split's ``meta.json`` (analytic TE, seeds, per-cell manifest)."""
        return self._data_meta

    def make_plain_train_loader(self) -> DataLoader:
        r"""Full-train, unshuffled loader for a post-fit latent-stats pass.

        When a process group is initialised, the loader is sharded with an explicit
        ``DistributedSampler(shuffle=False, drop_last=False)`` (v1-parity); otherwise it
        falls back to a plain ordered loader.

        .. note::
            With ``drop_last=False`` and ``len(dataset)`` not divisible by ``world_size``,
            :class:`~torch.utils.data.distributed.DistributedSampler` pads the last shard
            by repeating a few early indices, so up to ``world_size - 1`` rows are served
            on two ranks. A cross-rank latent-stats aggregation must therefore dedup by
            ``guid`` (or use ``drop_last=True``) rather than assume perfectly disjoint
            shards. This matches the v1 ``datamodule_synth`` behaviour and is finalised in
            the Sprint 5 training path.
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
        return make_dataloader(
            self._train_ds, self._batch_size, shuffle=False, drop_last=False,
            num_workers=self._num_workers, pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
        )
