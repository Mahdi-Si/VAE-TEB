r"""Lightning ``DataModule`` over the cached synthetic-TE ``.npz`` benchmark.

This is the data side of the **multi-GPU (DDP) training path** for the single
large ``G1_mix`` final-model run (see :mod:`train_ddp`). The single-GPU loop
:mod:`train_minimal` builds its loaders inline via
:func:`train_minimal.make_dataloaders`; this module wraps the **same**
:class:`dataset.SyntheticTEDataset` + :func:`dataset.make_dataloader` in the
:class:`lightning.pytorch.LightningDataModule` contract so a
:class:`lightning.pytorch.Trainer` built with ``use_distributed_sampler=True``
can shard each epoch across ranks automatically.

Two loaders are exposed:

* :meth:`train_dataloader` / :meth:`val_dataloader` -- plain shuffled / ordered
  loaders. Under DDP, Lightning replaces their samplers with a
  :class:`~torch.utils.data.distributed.DistributedSampler` (preserving the
  ``shuffle`` flag) because the Trainer sets ``use_distributed_sampler=True``.
  **We deliberately do not** add a ``DistributedSampler`` here -- doing so on
  top of Lightning's injection would double-wrap and silently drop samples.
* :meth:`make_plain_train_loader` -- a full-train, **unshuffled** loader with an
  *explicit* ``DistributedSampler(shuffle=False, drop_last=False)`` for the
  post-fit :meth:`SeqVaeLagAttnV1.fit_latent_stats` pass. That pass is run
  outside the Trainer (Lightning auto-injection is no longer active), so the
  sampler must be constructed here. Sharding (rather than a full copy per rank)
  keeps ``fit_latent_stats``'s cross-rank ``all_reduce`` count truthful: every
  valid time step is summed **exactly once**, not ``world_size`` times.

Mirrors :func:`train_minimal.make_dataloaders` for the cache-path resolution and
the DataLoader knobs (``dataset.num_workers`` / ``pin_memory`` /
``persistent_workers``), so single- and multi-GPU runs read the identical cache.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import lightning as pl
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.vae_teb_prediction.model.model_experiment.synthetic.dataset import (
    SyntheticTEDataset,
    attribute_dict_collate,
    make_dataloader,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (
    resolve_user_path,
)


class SyntheticTEDataModule(pl.LightningDataModule):
    r"""Wrap the cached synthetic benchmark splits for DDP training.

    Args:
        config: The (post-override, benchmark-resolved) config dict. Reads
            ``experiment.{benchmark,tag}``, ``paths.data_dir`` and the optional
            ``dataset`` block.
        batch_size: **Per-GPU** batch size. Under DDP the global batch is
            ``batch_size * world_size`` (the locked "bigger global batch"
            policy); LR scaling lives in :mod:`train_ddp`, not here.
    """

    def __init__(self, config: Dict[str, Any], *, batch_size: int) -> None:
        super().__init__()
        self._config = config
        self._batch_size = int(batch_size)

        exp = config["experiment"]
        data_root = resolve_user_path(config["paths"]["data_dir"])
        self._cache_dir: Path = data_root / str(exp["benchmark"]) / str(exp["tag"])

        ds_cfg = config.get("dataset") or {}
        self._num_workers = int(ds_cfg.get("num_workers", 0))
        self._pin_memory = bool(ds_cfg.get("pin_memory", False))
        self._persistent_workers = bool(ds_cfg.get("persistent_workers", False))

        self._train_ds: Optional[SyntheticTEDataset] = None
        self._val_ds: Optional[SyntheticTEDataset] = None
        self._data_meta: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        """Load the in-memory train / optional val splits on every rank.

        ``SyntheticTEDataset`` reads the whole ``.npz`` into RAM and closes the
        handle in ``__init__`` (spawn-safe, picklable), so each rank holds its
        own copy. Raises the same guidance as
        :func:`train_minimal.make_dataloaders` when the cache is missing.
        """
        train_npz = self._cache_dir / "train.npz"
        val_npz = self._cache_dir / "val.npz"
        if not train_npz.is_file():
            raise FileNotFoundError(
                f"cached dataset not found: {train_npz}\n"
                f"Build it first, e.g.:\n"
                f"  python -m model.vae_teb_prediction.model.model_experiment."
                f"synthetic.mixed_dataset --tag {self._config['experiment']['tag']}"
            )
        self._train_ds = SyntheticTEDataset(train_npz)
        self._data_meta = dict(self._train_ds.meta)
        if val_npz.is_file():
            self._val_ds = SyntheticTEDataset(val_npz)

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

    # ------------------------------------------------------------------
    # Latent-stats pass (run outside the Trainer)
    # ------------------------------------------------------------------
    @property
    def data_meta(self) -> Dict[str, Any]:
        """The training split's ``meta.json`` (analytic TE, seeds, layout)."""
        return self._data_meta

    def make_plain_train_loader(self) -> DataLoader:
        r"""Full-train, unshuffled loader for :meth:`fit_latent_stats`.

        When a process group is initialised, the loader is sharded with an
        explicit ``DistributedSampler(shuffle=False, drop_last=False)`` so each
        rank sees a disjoint slice and ``fit_latent_stats``'s ``all_reduce``
        counts every valid time step once. With no process group (single-GPU /
        CPU) it falls back to a plain ordered loader -- identical iteration to
        :func:`train_minimal.make_dataloaders`' train loader without shuffle.
        """
        assert self._train_ds is not None, "call setup() first"
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(
                self._train_ds, shuffle=False, drop_last=False
            )
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
