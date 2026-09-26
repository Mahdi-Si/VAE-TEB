r"""The committed shard delivers the two target blocks at the loader's normalised scale.

In the raw sibling ``fhr_st`` and ``fhr_ph`` are only inputs; here they are also the reconstruction
target, so their scale sets the scale of every reported nat. A stats file the loader rejects
disables normalisation with a warning and hands back correctly shaped, wrongly scaled tensors --
and a Gaussian NLL against those is meaningless with nothing raising anywhere. The check runs
against the committed shard, through the real loader and the sibling's tiny config (the loader
configuration this package inherits).

The reach budget's resolution, the survivor counts and the loader's declared widths are pinned in
the packages that own them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from teb_vae.lag_attn_fs.tests.conftest import absolutize_dataset_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The sibling's tiny config, resolved through its own ``base:`` chain: the same shards, the same
#: trim and the same ``normalize_fields`` this package inherits.
_TINY_CONFIG = _REPO_ROOT / "teb_vae" / "lag_attn_rws" / "configs" / "tiny.yaml"


@pytest.fixture(scope="module")
def real_batch():
    """One batch from the committed shard, through the real loader and the shipped trim."""
    from teb_vae.lag_attn.config import load_config
    from train.data_module import GraphDataModule

    config = absolutize_dataset_paths(load_config(str(_TINY_CONFIG)))
    return next(iter(GraphDataModule(config).train_dataloader()))


@pytest.mark.slow
def test_the_target_blocks_are_normalized_by_the_loader(real_batch):
    """Both stored target blocks arrive near zero mean and order-one spread, not at stored scale."""
    for block in (real_batch.fhr_st, real_batch.fhr_ph):
        assert abs(float(block.mean())) < 5.0
        assert float(block.std()) < 20.0
