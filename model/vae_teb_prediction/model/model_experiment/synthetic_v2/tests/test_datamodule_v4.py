r"""S3-T02 tests for ``SyntheticRawDataModuleV4``."""

from __future__ import annotations

import copy
from typing import Any, Dict

import pytest

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.datamodule_v4 import (
    SyntheticRawDataModuleV4,
)

pytestmark = pytest.mark.v4


def _single_process_config(config: Dict[str, Any]) -> Dict[str, Any]:
    r"""A deep copy with ``num_workers=0`` (multi-worker spawn is fragile under pytest on Windows)."""
    cfg = copy.deepcopy(config)
    cfg.setdefault("dataset_config", {}).setdefault("dataloader_config", {})["num_workers"] = 0
    return cfg


def test_datamodule_v4_batch_shapes(tiny_cache_v4):
    r"""A collated batch exposes the raw-model contract fields with the right shapes."""
    cfg = _single_process_config(tiny_cache_v4["config"])
    dm = SyntheticRawDataModuleV4(
        cfg, batch_size=2, benchmark="G1_raw_v4", cache_dir=tiny_cache_v4["cache_dir"],
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch.fhr.shape == (2, 5280)
    assert batch.up.shape == (2, 5280)
    assert batch.weight.shape == (2, 330)
    assert batch.target.shape == (2, 330)


def test_datamodule_v4_val_test_loaders(tiny_cache_v4):
    r"""Val / test loaders yield ordered batches of the same contract."""
    cfg = _single_process_config(tiny_cache_v4["config"])
    dm = SyntheticRawDataModuleV4(
        cfg, batch_size=2, benchmark="G1_raw_v4", cache_dir=tiny_cache_v4["cache_dir"],
    )
    dm.setup("fit")
    for loader in (dm.val_dataloader(), dm.test_dataloader()):
        assert loader is not None
        batch = next(iter(loader))
        assert batch.fhr.shape[1] == 5280
        assert batch.weight.shape[1] == 330


def test_datamodule_v4_missing_cache_raises(tiny_cache_v4, tmp_path):
    r"""A missing cache raises a clear ``FileNotFoundError`` with a build hint."""
    cfg = _single_process_config(tiny_cache_v4["config"])
    dm = SyntheticRawDataModuleV4(
        cfg, batch_size=2, benchmark="G1_raw_v4", cache_dir=tmp_path / "does_not_exist",
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        dm.setup("fit")
    assert "build" in str(excinfo.value).lower()
