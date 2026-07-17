"""Optional data-loading lifecycle for the shared training framework.

Provides :class:`GraphDataModule`, a thin :class:`lightning.LightningDataModule`
wrapper around :func:`create_optimized_dataloader`. It centralises loader construction
from ``dataset_config`` and hands the ``Trainer`` **plain** (non-distributed)
dataloaders, so Lightning's built-in distributed-sampler injection owns the DDP sampler.

Adoption is optional and no training entry point is converted to it here; the module
exists so a consumer can opt in to a clean, DDP-correct data lifecycle when it is ready.
"""
from __future__ import annotations

from typing import Any, Dict, List

import lightning as L
from torch.utils.data import DataLoader

from hdf5_dataset.hdf5_dataset import create_optimized_dataloader


class GraphDataModule(L.LightningDataModule):
    r"""Config-driven dataloaders that let Lightning own the DDP sampler.

    The module builds **plain** map-style dataloaders — it never constructs a
    :class:`~torch.utils.data.DistributedSampler` itself — and relies on the Lightning
    ``Trainer``'s ``use_distributed_sampler`` (default ``True``) to wrap them for
    distributed training. For a map-style dataset this is the recommended approach: on
    each rank Lightning injects a ``DistributedSampler`` with the correct
    ``num_replicas``/``rank``, uses ``shuffle=True`` for training and ``shuffle=False``
    for validation/test, and — critically — calls ``sampler.set_epoch(epoch)`` every
    epoch so the training shuffle actually differs across epochs.

    Owning the sampler manually is worse on two counts that Lightning gets right: a
    hand-built ``DistributedSampler`` repeats the *same* ordering every epoch unless
    ``set_epoch`` is called by hand (effectively ``shuffle=False`` across epochs), and it
    drops the evaluation tail via ``drop_last`` instead of padding it. Deferring to
    Lightning avoids both.

    Because the module builds no sampler of its own, an adopting consumer keeps
    ``advanced_config.trainer.use_distributed_sampler`` at its default ``true`` — there is
    nothing for Lightning to collide with, so no ``use_distributed_sampler: false`` opt-out
    is needed.

    Splits come from separate HDF5 file lists in ``dataset_config``
    (``vae_train_datasets`` for training, ``vae_test_datasets`` for the held-out set).
    There is a single held-out list, so validation and test map to the same files.

    ``lean-limit``: this is a 1:1 loader wrapper, not a split manager; add a
    randomized-split path only when a dataset ships a single combined file that must be
    partitioned in-process.

    Args:
        config: The resolved experiment config mapping (the same dict the graph model
            loads). ``dataset_config`` and ``general_config.batch_size`` are read from
            it; nothing is mutated.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self._dataset_config = config.get("dataset_config", {})
        self._dataloader_config = self._dataset_config.get("dataloader_config", {})
        batch_size = config.get("general_config", {}).get("batch_size", {})
        self._batch_size_train = batch_size.get("train", 32)
        self._batch_size_test = batch_size.get("test", 32)

    def _make_loader(
        self, hdf5_files: List[str], *, batch_size: int, shuffle: bool
    ) -> DataLoader:
        """Build one split's plain (non-distributed) loader through the shared factory.

        The factory is called at its default ``world_size=1`` — no ``rank``/``world_size``
        is passed — so it builds **no** ``DistributedSampler``; under DDP the Lightning
        ``Trainer`` wraps the returned loader with one (see the class docstring).

        Args:
            hdf5_files: HDF5 file paths for this split.
            batch_size: Per-process batch size.
            shuffle: ``True`` for training, ``False`` for validation/test. Under DDP
                Lightning derives the injected sampler's shuffle from the stage, so this
                governs the single-process / non-distributed case.

        Returns:
            The configured :class:`~torch.utils.data.DataLoader`.
        """
        dl_config = self._dataloader_config
        # Copy so the splat below cannot mutate the shared config dict.
        dataset_kwargs = dict(dl_config.get("dataset_kwargs") or {})
        return create_optimized_dataloader(
            hdf5_files=hdf5_files,
            batch_size=batch_size,
            num_workers=dl_config.get("num_workers", 4),
            shuffle=shuffle,
            stats_path=self._dataset_config.get("stat_path"),
            normalize_fields=dl_config.get("normalize_fields"),
            prefetch_factor=dl_config.get("prefetch_factor", 2),
            **dataset_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        """Training loader over ``dataset_config.vae_train_datasets`` (shuffled)."""
        return self._make_loader(
            self._dataset_config.get("vae_train_datasets", []),
            batch_size=self._batch_size_train,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation loader over ``dataset_config.vae_test_datasets`` (unshuffled)."""
        return self._make_loader(
            self._dataset_config.get("vae_test_datasets", []),
            batch_size=self._batch_size_test,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Test loader over the same held-out ``vae_test_datasets`` (unshuffled)."""
        return self._make_loader(
            self._dataset_config.get("vae_test_datasets", []),
            batch_size=self._batch_size_test,
            shuffle=False,
        )
