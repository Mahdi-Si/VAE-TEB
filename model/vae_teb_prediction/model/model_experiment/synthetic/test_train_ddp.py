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

import csv
import json
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1
from model.vae_teb_prediction.model.model_experiment.synthetic.train_ddp import (
    train_ddp,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (
    _FIELDNAMES,
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


def _run(tmp: Path) -> Path:
    """Build caches, run a 2-epoch single-device train_ddp, return the run dir.

    Two epochs with ``plot_every=1`` force at least one periodic loss-curve
    refresh (in addition to the unconditional ``on_fit_end`` render), so the
    figure assertions exercise the scheduled path rather than only the final one.
    """
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
            "epochs": 2,
            "batch_size": 4,
            "plot_every": 1,
            "data_dir": str(data_dir),
            "results_dir": str(results_dir),
        },
    )
    return results_dir / "G1_mix" / tag


def _assert_loss_settings(ls: Dict) -> None:
    """Assert the ``loss_settings`` block carries the full downstream contract."""
    for key in ("beta", "kld_beta", "lambda_full", "lambda_base",
                "likelihood", "sigma_obs", "free_bits"):
        assert key in ls, f"loss_settings missing {key!r}"
    assert ls["kld_beta"] == ls["beta"], "kld_beta must mirror beta"
    # G1_mix overlay pins the nat-scale likelihood.
    assert ls["likelihood"] == "gaussian_nll"


def _assert_strict_loadable(ckpt: Dict) -> None:
    """Assert the ckpt rebuilds + strict-loads into a fresh model exactly."""
    fresh = SeqVaeLagAttnV1(**ckpt["model_kwargs"])
    assert set(ckpt["model_state_dict"]) == set(fresh.state_dict()), (
        "checkpoint state_dict keys do not match SeqVaeLagAttnV1(**model_kwargs)"
    )
    # Exercise the exact downstream contract mixed_eval / evaluate_te use.
    loaded = load_checkpoint_strict(fresh, ckpt, map_location="cpu")
    assert loaded is not None, "load_checkpoint_strict rejected the ckpt"


def test_train_ddp_smoke() -> None:
    """Single-device run produces loadable ckpts + single-GPU-style figures."""
    with tempfile.TemporaryDirectory() as _tmp:
        run_dir = _run(Path(_tmp))

        # --- final.ckpt: post latent-stats fit, full synthetic-format bridge ---
        final_path = run_dir / "final.ckpt"
        assert final_path.is_file(), f"final.ckpt missing at {final_path}"
        final = torch.load(final_path, map_location="cpu", weights_only=False)
        _assert_strict_loadable(final)
        assert final["latent_stats_fitted"] is True, "fit_latent_stats did not run"
        _assert_loss_settings(final["loss_settings"])

        # --- best.ckpt: synthetic-format mirror of the Lightning best snapshot --
        best_path = run_dir / "best.ckpt"
        assert best_path.is_file(), f"best.ckpt missing at {best_path}"
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        _assert_strict_loadable(best)
        assert best["latent_stats_fitted"] is False, (
            "best.ckpt must predate the latent-stats fit (latent_stats_fitted=False)"
        )
        _assert_loss_settings(best["loss_settings"])
        assert "benchmark" in best["data_meta"] or "tag" in best["data_meta"], (
            "best.ckpt lost data_meta"
        )

        # No Lightning-format file is ever named best.ckpt / final.ckpt.
        for cons in (final, best):
            assert "model_kwargs" in cons, "consumable ckpt lost model_kwargs"

        # --- loss-curve artifacts (single-GPU look + location) -----------------
        csv_path = run_dir / "metrics.csv"
        assert csv_path.is_file(), f"metrics.csv missing at {csv_path}"
        with open(csv_path, newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        assert header == _FIELDNAMES, (
            f"metrics.csv header diverges from single-GPU schema:\n"
            f"  got: {header}\n  want: {_FIELDNAMES}"
        )
        assert (run_dir / "loss_plot_epoch.html").is_file(), "loss HTML missing"
        assert (run_dir / "training_curves.png").is_file(), "curves PNG missing"

        print("[test_train_ddp] OK -- final/best ckpt bridge + figures verified")


if __name__ == "__main__":
    test_train_ddp_smoke()
