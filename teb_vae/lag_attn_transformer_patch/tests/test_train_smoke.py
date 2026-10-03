"""P4-08: one real 3-epoch fit of ``tiny.yaml`` through the entry point.

The committed shard has no FHR gap and no non-finite UP sample, so neither ``missing`` token would
ever see a gradient (AdamW decay alone still nudges it, which ``torch.equal`` would call "trained").
The fit therefore reads a tmp copy of that shard with one FHR gap (``weight`` 0) and one UP dropout
(NaN, which ``source_validity: finite`` masks), and movement is judged against a floor decay cannot
reach.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_transformer_patch import trainer as trainer_module
from teb_vae.lag_attn_transformer_patch.nets.model import SeqVaeLagAttnTrfPatch
from teb_vae.lag_attn_transformer_patch.nets.patching import PatchEmbedding
from train.graph_models_utils import load_checkpoint_strict

from .conftest import absolutize_dataset_paths, relative_change

_TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"

#: Relative movement a trained tensor must exceed. Decay alone moves one by ~lr * wd * steps
#: (~2e-7 here); the least-moved trained PatchEmbedding tensor moves ~4e-4.
TRAINED_FLOOR = 1e-5


@pytest.mark.slow
def test_tiny_fit_is_finite_and_trains_every_patch_embedding_tensor(tmp_path) -> None:
    config = absolutize_dataset_paths(load_config(str(_TINY)))
    shard = tmp_path / "gapped.hdf5"
    shutil.copy(config["dataset_config"]["vae_train_datasets"][0], shard)
    with h5py.File(shard, "r+") as handle:  # steps 100..109 (raw 1600..1759): past trim and warm-up
        handle["weight"][:, 100:110] = 0.0
        handle["fhr"][:, 1600:1760] = 0.0
        handle["up"][:, 2400:2560] = np.nan
    dataset = config["dataset_config"]
    dataset["vae_train_datasets"] = dataset["vae_test_datasets"] = [str(shard)]
    config["general_config"]["epochs"] = 3
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path)
    config["advanced_config"]["trainer"]["profiler"] = None
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    trainer_module.main(str(config_path))

    (metrics_path,) = tmp_path.rglob("metrics_history.csv")
    metrics = pd.read_csv(metrics_path)
    assert math.isfinite(metrics["train/total_loss"].iloc[-1])
    grad_norm = metrics["train/grad_norm"].dropna()
    assert len(grad_norm) == 3 and np.isfinite(grad_norm).all() and (grad_norm > 0).all(), grad_norm

    checkpoints = list(tmp_path.rglob("*.ckpt"))
    assert checkpoints and all(p.name.startswith("lag-attn-trf-patch-") for p in checkpoints)
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    blob = torch.load(latest, map_location="cpu", weights_only=False)
    trained = SeqVaeLagAttnTrfPatch(**blob["model_kwargs"])
    assert load_checkpoint_strict(trained, blob) is not None
    torch.manual_seed(config["general_config"]["seed"])  # reproduces the run's init exactly
    fresh = SeqVaeLagAttnTrfPatch(**blob["model_kwargs"])

    moved = {
        f"{name}.{pname}": relative_change(
            p.detach(), trained.get_parameter(f"{name}.{pname}").detach()
        )
        for name, module in fresh.named_modules()
        if isinstance(module, PatchEmbedding)
        for pname, p in module.named_parameters()
    }
    assert {"target_adapter.missing", "source_adapter.missing"} <= moved.keys()
    assert {k: v for k, v in moved.items() if v <= TRAINED_FLOOR} == {}
