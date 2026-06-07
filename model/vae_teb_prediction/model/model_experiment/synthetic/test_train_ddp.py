r"""Smoke test for the DDP training entry point :mod:`train_ddp`.

Runs a single-device (``--devices 1``) 1-epoch train on a tiny synthetic cache
with a shrunken model, then asserts the synthetic-format checkpoint bridge is
intact. This is the single-GPU / CPU verification path (Windows dev); the real
8-GPU DDP run on Linux is validated separately (see ``model_validation_v2``).

What it guards:

* ``final.ckpt`` is written in the :func:`train_minimal.save_checkpoint` format.
* The stored ``model_state_dict`` keys match a fresh
  ``SeqVaeLagAttnV1(**model_kwargs)`` **exactly** (the strict-loader contract
  that :func:`train.graph_models_utils.load_checkpoint_strict` enforces).
* ``latent_stats_fitted`` is ``True`` (the post-fit ``fit_latent_stats`` ran).
* ``loss_settings`` carries both ``beta`` and ``kld_beta`` plus the lambda /
  likelihood / sigma_obs / free_bits knobs that ``mixed_eval`` reads.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1
from model.vae_teb_prediction.model.model_experiment.synthetic.train_ddp import (
    train_ddp,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (
    load_config,
)
from train.graph_models_utils import load_checkpoint_strict

_PKG_DIR = Path(__file__).resolve().parent
_CONFIG = _PKG_DIR / "config_synth.yaml"

# Tiny dimensions that keep the native channel layout (c_y=87, c_u=101) so the
# model runs unchanged, while shrinking everything else for a fast CPU/GPU pass.
_T = 40
_N_TRAIN = 8
_N_VAL = 4
_SMALL_MODEL = {
    "sequence_length": _T,
    "d_model": 16,
    "d_z": 4,
    "horizon": 8,
    "warmup_period": 8,
    "c_y": 87,
    "c_u": 101,
    "use_up_st": True,
    "max_lag": 10,
    "num_heads": 2,
    "d_head": 8,
    "lstm_layers": 1,
    "decoder_hidden": 16,
}


def _write_cache(cache_dir: Path, n: int, stem: str) -> None:
    """Write one tiny ``{stem}.npz`` + ``meta.json`` with the native layout."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0 if stem == "train" else 1)
    np.savez(
        cache_dir / f"{stem}.npz",
        fhr_st=rng.standard_normal((n, _T, 43)).astype(np.float32),
        fhr_ph=rng.standard_normal((n, _T, 44)).astype(np.float32),
        up_st=rng.standard_normal((n, _T, 43)).astype(np.float32),
        up_ph=rng.standard_normal((n, _T, 58)).astype(np.float32),
        weight=np.ones((n, _T), dtype=np.float32),
    )
    if not (cache_dir / "meta.json").is_file():
        with open(cache_dir / "meta.json", "w", encoding="utf-8") as fh:
            json.dump({"tag": "ddp_smoke", "te_true": 0.5,
                       "true_lag_band": [1, 2, 3]}, fh)


def _run(tmp: Path) -> Tuple[Path, Dict]:
    """Build caches, run a 1-epoch single-device train_ddp, return ckpt + state."""
    tag = "ddp_smoke"
    data_dir = tmp / "data"
    results_dir = tmp / "results"
    cache_dir = data_dir / "G1_mix" / tag
    _write_cache(cache_dir, _N_TRAIN, "train")
    _write_cache(cache_dir, _N_VAL, "val")

    config = load_config(_CONFIG)
    config["model"].update(_SMALL_MODEL)        # shrink the architecture

    train_ddp(
        config,
        overrides={
            "data_tag": tag,
            "run_tag": tag,
            "devices": 1,
            "epochs": 1,
            "batch_size": 4,
            "data_dir": str(data_dir),
            "results_dir": str(results_dir),
        },
    )
    ckpt_path = results_dir / "G1_mix" / tag / "final.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt_path, ckpt


def test_train_ddp_smoke() -> None:
    """One-epoch single-device run produces a loadable synthetic-format ckpt."""
    with tempfile.TemporaryDirectory() as _tmp:
        ckpt_path, ckpt = _run(Path(_tmp))

        assert ckpt_path.is_file(), f"final.ckpt missing at {ckpt_path}"

        # State-dict keys must match a fresh model exactly (strict-loader rule).
        fresh = SeqVaeLagAttnV1(**ckpt["model_kwargs"])
        assert set(ckpt["model_state_dict"]) == set(fresh.state_dict()), (
            "checkpoint state_dict keys do not match SeqVaeLagAttnV1(**model_kwargs)"
        )
        # Exercise the exact downstream contract mixed_eval / evaluate_te use.
        loaded = load_checkpoint_strict(fresh, ckpt, map_location="cpu")
        assert loaded is not None, "load_checkpoint_strict rejected the ckpt"

        assert ckpt["latent_stats_fitted"] is True, "fit_latent_stats did not run"

        ls = ckpt["loss_settings"]
        for key in ("beta", "kld_beta", "lambda_full", "lambda_base",
                    "likelihood", "sigma_obs", "free_bits"):
            assert key in ls, f"loss_settings missing {key!r}"
        assert ls["kld_beta"] == ls["beta"], "kld_beta must mirror beta"
        # G1_mix overlay pins the nat-scale likelihood.
        assert ls["likelihood"] == "gaussian_nll"
        print("[test_train_ddp] OK -- final.ckpt bridge verified")


if __name__ == "__main__":
    test_train_ddp_smoke()
