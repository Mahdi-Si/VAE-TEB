"""The three pieces of ``adverse_finetune`` that can be wrong silently."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_transformer_cfs import adverse_finetune as af


def _planted_shard(path: Path, delay_s: float) -> None:
    """Two recordings whose FHR dips ``delay_s`` after every UA bump, at random bump times."""
    rng = np.random.default_rng(0)
    n, rows = 5280, 16
    t = np.arange(n) / af.FS_RAW
    up, fhr = np.zeros((rows, n)), np.zeros((rows, n))
    for i in range(rows):
        bumps = np.cumsum(rng.uniform(100, 200, size=12))
        bumps = bumps[bumps < t[-1]]
        ua = sum(np.exp(-0.5 * ((t - c) / 15.0) ** 2) for c in bumps)
        up[i] = 20 + 40 * ua + rng.normal(0, 1, n)
        fhr[i] = 140 - 20 * np.interp(t - delay_s, t, ua, left=0) + rng.normal(0, 2, n)
    with h5py.File(path, "w") as f:
        f["up"], f["fhr"] = up.astype("f4"), fhr.astype("f4")
        f["weight"] = np.ones((rows, 330), dtype="f4")
        f["epoch"] = np.linspace(-20000, -3000, rows).astype("f4")
        f.create_dataset("guid", data=np.repeat(np.array(list("abcd"), dtype=object), 4),
                         dtype=h5py.string_dtype())


def test_premise_recovers_a_planted_delay(tmp_path: Path) -> None:
    folder = tmp_path / "kfold" / "fold_1" / "train"
    folder.mkdir(parents=True)
    _planted_shard(folder / "acidosis_cs.hdf5", delay_s=40.0)
    out = af.run_premise(kfold_root=str(tmp_path / "kfold"), fold=1, splits=["train"],
                         subgroups=None, out_dir=tmp_path / "premise", max_lag_s=120.0,
                         min_lag_s=-60.0, fs_out=1.0, detrend_s=300.0, min_valid_frac=0.5,
                         min_pairs_s=300.0, hour_edges=(0, 1, 2, 4, 13), n_boot=50, seed=0)
    summary = pd.read_csv(out / "summary.csv").set_index(["group", "stratum_kind", "stratum"])
    row = summary.loc[("acidosis", "all", "all")]
    assert abs(row["d_min_lag_s"] - 40) <= 2   # FHR dips 40 s after UA: a negative coupling there
    assert row["d_min"] < -0.5 and row["d_min_ci_hi"] < 0
    curves = np.load(out / "curves.npz")
    # Mismatched recordings carry no coupling: the pooled null stays near zero at every lag.
    assert np.nanmax(np.abs(curves["acidosis|all|all|r0_mean"])) < 0.1
    assert (out / "xcorr.png").is_file()


def test_finetune_config_holds_beta_and_keeps_data_disjoint(tmp_path: Path) -> None:
    base = {"general_config": {"tag": "base", "lr": 3e-4},
            "model_config": {"VAE_model": {"d_z": 64, "beta_schedule": {
                "kind": "linear_warmup", "start": 0.0, "end": 0.7, "warmup_epochs": 50}}},
            "dataset_config": {"vae_train_datasets": ["pre.hdf5"], "stat_path": "s.hdf5"}}
    beta = af.final_beta(base)
    assert beta == 0.7
    cfg = af.finetune_config(base, pretrained=Path("p.ckpt"), train_files=[Path("tr.hdf5")],
                             val_files=[Path("va.hdf5")], tag="t", out_dir_base=tmp_path,
                             lr=3e-5, lr_warmup_steps=200, beta=beta, patience=10, max_epochs=5,
                             cuda_devices=None, extra={"general_config": {"lr": 1e-5}})
    vae = cfg["model_config"]["VAE_model"]
    assert vae["beta_schedule"] == {"kind": "constant", "value": 0.7} and vae["d_z"] == 64
    assert cfg["dataset_config"]["vae_train_datasets"] == ["tr.hdf5"]   # pretraining shards replaced
    assert cfg["dataset_config"]["stat_path"] == "s.hdf5"              # statistics kept
    assert cfg["general_config"]["lr"] == 1e-5                         # the extra delta wins
    with pytest.raises(ValueError):
        af.finetune_config(base, pretrained=Path("p.ckpt"), train_files=[Path("x.hdf5")],
                           val_files=[Path("x.hdf5")], tag="t", out_dir_base=tmp_path, lr=1.0,
                           lr_warmup_steps=0, beta=1.0, patience=1, max_epochs=1, cuda_devices=None)


def test_best_checkpoint_reads_the_primary_monitor(tmp_path: Path) -> None:
    def save(name: str, monitor: str, best: str) -> None:
        key = f"ModelCheckpoint{{'monitor': '{monitor}', 'mode': 'min'}}"
        torch.save({"callbacks": {key: {"best_model_path": f"/elsewhere/{best}"}}}, tmp_path / name)

    save("lag-attn-trf-cfs-epoch=03.ckpt", "val/total_loss", "lag-attn-trf-cfs-epoch=03.ckpt")
    save("lag-attn-trf-cfs-epoch=05.ckpt", "val/total_loss", "lag-attn-trf-cfs-epoch=03.ckpt")
    save("lag-attn-trf-cfs-val_nll-epoch=07.ckpt", "val/nll_full_block",
         "lag-attn-trf-cfs-val_nll-epoch=07.ckpt")
    assert af.best_checkpoint(tmp_path, "val/total_loss").name == "lag-attn-trf-cfs-epoch=03.ckpt"
