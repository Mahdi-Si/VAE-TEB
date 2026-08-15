"""``GraphDataModule`` builds plain loaders and lets Lightning own the DDP sampler.

The loader factory (``create_optimized_dataloader``) is spied — no HDF5 file is opened
and no real ``DataLoader`` is constructed. The tests assert on the captured factory
kwargs: the module reproduces the shipped-config loader arguments and, crucially, never
asks the factory to build an internal ``DistributedSampler`` (it passes no
``rank``/``world_size``, so the factory stays single-process). Under DDP the Lightning
``Trainer`` wraps these plain loaders with a correctly-managed ``DistributedSampler``
(per-epoch ``set_epoch``), which is why the module builds none of its own.
"""
import yaml

import train.data_module as data_module
from train.data_module import GraphDataModule


class _Sentinel:
    """Placeholder return value for the spied factory (never used as a real loader)."""


def _spy(monkeypatch):
    """Replace the factory with a recorder; return the list of captured kwarg dicts."""
    calls = []

    def fake_factory(**kwargs):
        calls.append(kwargs)
        return _Sentinel()

    monkeypatch.setattr(data_module, "create_optimized_dataloader", fake_factory)
    return calls


def _load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_reproduces_config_loaders(config_path, monkeypatch):
    calls = _spy(monkeypatch)
    config = _load_config(config_path)
    ds = config["dataset_config"]
    dl = ds["dataloader_config"]

    dm = GraphDataModule(config)
    dm.train_dataloader()
    dm.val_dataloader()
    assert len(calls) == 2
    train_kw, val_kw = calls

    # Split files + shuffle direction.
    assert train_kw["hdf5_files"] == ds["vae_train_datasets"]
    assert train_kw["shuffle"] is True
    assert val_kw["hdf5_files"] == ds["vae_test_datasets"]
    assert val_kw["shuffle"] is False

    # Batch sizes come from general_config, not dataset_config.
    assert train_kw["batch_size"] == config["general_config"]["batch_size"]["train"]
    assert val_kw["batch_size"] == config["general_config"]["batch_size"]["test"]

    # Dataloader knobs + normalization threaded from dataloader_config / dataset_config.
    assert train_kw["num_workers"] == dl["num_workers"]
    assert train_kw["stats_path"] == ds["stat_path"]
    assert train_kw["normalize_fields"] == dl["normalize_fields"]

    # dataset_kwargs are forwarded verbatim (splatted) to the factory.
    for key, value in dl["dataset_kwargs"].items():
        assert train_kw[key] == value


def test_builds_plain_loaders_so_lightning_owns_the_ddp_sampler(config_path, monkeypatch):
    # The module must not construct an internal DistributedSampler: it passes no
    # rank/world, so the factory stays at its single-process default (world_size=1 ->
    # sampler=None), leaving DDP sharding to the Trainer's use_distributed_sampler. This
    # guards against re-introducing a hand-built sampler, which would repeat the shuffle
    # every epoch (no set_epoch) and double-shard against Lightning's own sampler.
    calls = _spy(monkeypatch)
    dm = GraphDataModule(_load_config(config_path))
    dm.train_dataloader()
    dm.val_dataloader()
    dm.test_dataloader()
    assert len(calls) == 3
    for kw in calls:
        assert kw.get("world_size", 1) == 1
        assert kw.get("rank", 0) == 0


def test_test_dataloader_uses_held_out_files(config_path, monkeypatch):
    calls = _spy(monkeypatch)
    ds = _load_config(config_path)["dataset_config"]
    dm = GraphDataModule(_load_config(config_path))
    dm.test_dataloader()
    assert calls[-1]["hdf5_files"] == ds["vae_test_datasets"]
    assert calls[-1]["shuffle"] is False
