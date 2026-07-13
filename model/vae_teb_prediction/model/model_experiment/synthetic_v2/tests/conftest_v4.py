r"""S0-T06: fabricated fixtures for the ``synthetic_v4`` (raw-model) test suite.

These fixtures are **model-free** fabrications that let the Sprint 6/7 estimator/gate math be
validated *before* any model is trained (the empirical $\gamma>0$ from a trained model is the
S8-T01 gate). They are exposed suite-wide by a ``from conftest_v4 import *`` in the sibling
``conftest.py`` (the ``v4`` marker is registered there too). All heavy imports (torch,
``model_raw``) are deferred into fixture bodies so importing this module at collection time
stays cheap and never perturbs the v2/v3 suites.

Fixtures:
    ``signal_kbar_fixture``   -- factory -> $(\bar K = \gamma\,\mathrm{te} + \text{noise},\
                                 \mathrm{te\_true})$ arrays (validates ``fit_calibration_v4``).
    ``source_exploiting_outputs`` -- clean vs permuted-UP forward-output dicts + target, built so
                                 the prediction-space ordering $\mathcal L_{\mathrm{feat}} <
                                 \mathcal L_{\mathrm{base}} < \mathcal L_{\mathrm{feat}}^{\pi(U)}$
                                 holds by construction.
    ``planted_lag_te_lag_map`` -- a $(B,T,L)$ ``te_lag_map`` whose lag mass peaks at a known $D$.
    ``tiny_raw_checkpoint``   -- a real tiny ``SeqVaeRawV4`` checkpoint (reuses the ``model_raw``
                                 tiny-model helpers), returned as ``(path, kwargs)``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pytest

__all__ = [
    "signal_kbar_fixture",
    "source_exploiting_outputs",
    "planted_lag_te_lag_map",
    "tiny_raw_checkpoint",
    "tiny_cache_v4",
    "synth_metrics_v4",
]

#: The default injected-TE ladder shared by the fabricated fixtures (matches the config grid).
_TE_LADDER: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0)


@pytest.fixture
def signal_kbar_fixture() -> Callable[..., Dict[str, object]]:
    r"""Return a factory building a linear $\bar K$-vs-TE dataset with a known slope.

    The factory signature is ``make(gamma, noise, *, reps=40, te_ladder=_TE_LADDER, seed=0)``
    and returns ``{"kbar": (n,), "te_true": (n,), "gamma": float, "noise": float}`` with

    $$\bar K_i = \gamma \, \mathrm{te}_i + \varepsilon_i,\quad
      \varepsilon_i \sim \mathcal N(0, \text{noise}^2).$$

    ``reps`` copies of each ladder level give $n = \text{reps}\cdot|\text{ladder}|$ points, enough
    for a stable OLS slope. Deterministic in ``seed``.
    """

    def _make(gamma: float, noise: float, *, reps: int = 40,
              te_ladder: Tuple[float, ...] = _TE_LADDER, seed: int = 0) -> Dict[str, object]:
        rng = np.random.default_rng(seed)
        te_true = np.repeat(np.asarray(te_ladder, dtype=float), reps)
        eps = rng.normal(0.0, noise, size=te_true.shape)
        kbar = gamma * te_true + eps
        return {"kbar": kbar, "te_true": te_true, "gamma": float(gamma), "noise": float(noise)}

    return _make


@pytest.fixture
def source_exploiting_outputs() -> Dict[str, object]:
    r"""A clean-vs-permuted-UP pair of forward-output dicts plus the raw future target.

    Built so the prediction-space control holds *by construction* on raw-forecast MSE:

    $$\mathcal L_{\mathrm{feat}}^{\text{clean}} < \mathcal L_{\mathrm{base}}
      < \mathcal L_{\mathrm{feat}}^{\pi(U)}.$$

    ``clean.mu_full`` explains most of the target (source-exploiting), ``mu_base`` is the
    UP-independent baseline, and ``permuted.mu_full`` is the clean prediction with the batch
    (source) axis shuffled, so it decorrelates from the target and scores *worse* than the
    baseline. Shapes are the raw 4-D $(B,T,H,R)$ layout, kept tiny.
    """
    rng = np.random.default_rng(0)
    B, T, H, R = 4, 8, 3, 4
    target = rng.normal(size=(B, T, H, R))

    # Clean prediction recovers 90% of the target -> small residual (source-exploiting).
    clean_mu_full = 0.9 * target
    # UP-independent baseline: the per-(H,R) mean over batch+time (no source information).
    mu_base = np.broadcast_to(target.mean(axis=(0, 1), keepdims=True), target.shape).copy()
    # Permuted-UP: clean prediction with the batch axis shuffled -> decorrelated from target.
    perm = rng.permutation(B)
    permuted_mu_full = clean_mu_full[perm]

    clean = {"mu_full": clean_mu_full, "mu_base": mu_base}
    permuted = {"mu_full": permuted_mu_full, "mu_base": mu_base}
    return {"clean": clean, "permuted": permuted, "target": target}


@pytest.fixture
def planted_lag_te_lag_map() -> Dict[str, object]:
    r"""A $(B,T,L)$ ``te_lag_map`` whose per-anchor lag mass peaks at a known planted lag $D$.

    Returns ``{"te_lag_map": (B,T,L), "planted_lag": int, "kld_per_t": (B,T)}`` where
    ``kld_per_t`` is the lag-sum of ``te_lag_map`` (the identity the model satisfies: the per-step
    surrogate $\bar K$ equals the sum over lags of the TE lag map). The map is a small positive
    floor plus a Gaussian bump at lag $D$, so ``argmax_l mean_{b,t} te_lag_map`` recovers $D$.
    """
    rng = np.random.default_rng(1)
    B, T, L, planted = 4, 300, 91, 8
    lags = np.arange(L)
    bump = np.exp(-0.5 * ((lags - planted) / 1.5) ** 2)          # (L,) peak at lag=D
    floor = 0.02
    base = floor + bump[None, None, :]                            # (1,1,L)
    noise = 0.005 * rng.random((B, T, L))
    te_lag_map = np.clip(base + noise, 0.0, None)                 # (B,T,L) strictly positive
    kld_per_t = te_lag_map.sum(axis=2)                            # (B,T) == lag-sum identity
    return {"te_lag_map": te_lag_map, "planted_lag": planted, "kld_per_t": kld_per_t}


@pytest.fixture
def tiny_raw_checkpoint(tmp_path: Path) -> Tuple[Path, dict]:
    r"""A real tiny :class:`SeqVaeRawV4` checkpoint, returned as ``(path, model_kwargs)``.

    Reuses the ``model_raw`` tiny-model helpers (``tiny_raw_kwargs`` / ``make_tiny_raw_model`` /
    ``make_raw_batch``): builds the tiny model, takes one optimiser step, stamps the checkpoint
    via ``SeqVaeRawV4Pl.on_save_checkpoint`` (so ``model_class`` + ``model_kwargs`` are carried),
    and saves it under ``tmp_path``. Used by the Sprint 4-6 eval-runner tests.
    """
    import torch

    from model.vae_teb_prediction.model.model_raw.testing.conftest import (
        make_raw_batch,
        make_tiny_raw_model,
        tiny_raw_kwargs,
    )
    from model.vae_teb_prediction.model.model_raw.trainer_raw_v4 import SeqVaeRawV4Pl

    kwargs = tiny_raw_kwargs()
    model = make_tiny_raw_model()
    pl_module = SeqVaeRawV4Pl(model, lr=1e-3, model_kwargs=kwargs)

    fhr_raw, up_raw, mask = make_raw_batch(batch_size=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.zero_grad()
    out = model.forward(fhr_raw, up_raw, mask)
    loss = model.compute_loss(out, fhr_raw, mask, beta=0.1, free_bits=0.1)["total_loss"]
    loss.backward()
    opt.step()

    checkpoint: dict = {"state_dict": pl_module.state_dict()}
    pl_module.on_save_checkpoint(checkpoint)
    path = tmp_path / "tiny_raw_v4.ckpt"
    torch.save(checkpoint, path)
    return path, kwargs


@pytest.fixture
def synth_metrics_v4(tmp_path: Path) -> Dict[str, object]:
    r"""Fabricate a v4 ``metrics.json`` + ``per_sample_eval.npz`` pair on disk (S7-T04/T05).

    Builds a *schema-faithful* eval artifact set -- the exact top-level keys and nested structure
    :func:`eval_v4.run_eval_v4` writes (un-suffixed ``calibration.gamma``, top-level
    ``null_cell_gate`` with a ``ceiling``, ``prediction_controls.overall`` with the ``shuffled``
    control, ``lag_recovery.per_cell``, ``te_raw_gate``) -- populated from a linear
    $\bar K = \gamma\,\mathrm{TE}_{\mathrm{inj}}$ signal so the report/visualizer tests can render
    every figure and section from a fixture rather than a trained model. No model is loaded.

    Returns:
        ``{"metrics": dict, "per_sample": {arrays}, "metrics_path": Path, "npz_path": Path,
        "run_dir": Path}`` -- both files are written under ``tmp_path / "prod"``.
    """
    rng = np.random.default_rng(7)
    gamma, alpha, reps = 0.8, 0.02, 24
    ladder = _TE_LADDER
    te = np.repeat(np.asarray(ladder, dtype=float), reps)
    cell_id = np.repeat(np.arange(len(ladder), dtype=np.int64), reps)
    delay = np.full(te.shape, 8, dtype=np.int64)
    kbar = alpha + gamma * te + rng.normal(0.0, 0.05, size=te.shape)
    # Source-exploiting forecast losses: feat < base, and the permuted-source loss is worse.
    base_loss = 1.0 - 0.15 * te + rng.normal(0.0, 0.01, size=te.shape)
    feat_loss = base_loss - 0.20 * te + rng.normal(0.0, 0.01, size=te.shape)
    feat_loss_shuffled = base_loss + 0.10 * te + rng.normal(0.0, 0.01, size=te.shape)
    per_sample: Dict[str, np.ndarray] = {
        "kbar": kbar, "kbar_postwarm": kbar, "te_inj": te,
        "te_scat": np.full(te.shape, np.nan), "cell_id": cell_id, "delay": delay,
        "held_out": np.zeros(te.shape, dtype=np.int64),
        "feat_loss": feat_loss, "base_loss": base_loss,
        "pred_gain": base_loss - feat_loss, "feat_loss_shuffled": feat_loss_shuffled,
    }

    # Per-cell reductions (one row per ladder level), matching calibration.per_cell.
    per_cell = []
    for k, te_lvl in enumerate(ladder):
        sel = cell_id == k
        per_cell.append({"cell_id": int(k), "te_inj": float(te_lvl),
                         "kbar": float(kbar[sel].mean()), "delay": 8, "n": int(sel.sum())})

    signal_ids = [c["cell_id"] for c in per_cell if c["te_inj"] > 0]
    metrics: Dict[str, object] = {
        "model_class": "SeqVaeRawV4", "arm": "prod", "render_mode": "direct",
        "kld_support": "anchor", "n_samples": int(te.size),
        "calibration": {
            "kld_support": "anchor", "gamma": gamma, "alpha": alpha, "r2": 0.98,
            "spearman": 1.0, "monotonic": True, "n_cells": len(per_cell),
            "n_samples": int(te.size), "gamma_sample": gamma, "alpha_sample": alpha,
            "r2_sample": 0.97,
            "by_lag": {"8": {"gamma": gamma, "alpha": alpha, "r2": 0.97, "n": int(te.size)}},
            "kld_variants": {}, "per_cell": per_cell,
        },
        "null_cell_gate": {"mean": float(kbar[cell_id == 0].mean()), "std": 0.05,
                           "ci_lo": 0.0, "ci_hi": 0.05, "n_cells": 1, "ceiling": 0.05,
                           "pass": True},
        "prediction_controls": {
            "controls": ["shuffled"], "n_signal_cells": len(signal_ids),
            "overall": {
                "feat_loss": float(feat_loss.mean()), "base_loss": float(base_loss.mean()),
                "feat_loss_shuffled": float(feat_loss_shuffled.mean()),
                "shuffle_penalty_shuffled": float((feat_loss_shuffled - feat_loss).mean()),
                "ordering_pass_shuffled": True, "ordering_pass": True, "ordering_pass_frac": 1.0,
            },
            "per_cell": {},
        },
        "lag_recovery": {
            "per_cell": {
                str(c["cell_id"]): {
                    "D": 8, "delay_lo": 8, "delay_hi": 8, "true_band": list(range(0, 8)),
                    "is_null": c["te_inj"] == 0.0,
                    "lag_mass": 0.05 if c["te_inj"] == 0.0 else 0.9,
                    "peak_lag": None if c["te_inj"] == 0.0 else 8,
                    "peak_lag_err": None if c["te_inj"] == 0.0 else 0,
                    "within_tol": None if c["te_inj"] == 0.0 else True,
                }
                for c in per_cell
            },
            "mean_lag_mass": 0.9, "frac_within_tol": 1.0, "lag_mass_threshold": 0.8,
            "tolerance": 1, "mean_lag_mass_pass": True,
        },
        "te_raw_gate": {"gate": {"passed": True}, "constants": {"frac_threshold": 0.30},
                        "source": "realizability.json"},
    }

    run_dir = tmp_path / "prod"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    npz_path = run_dir / "per_sample_eval.npz"
    np.savez_compressed(str(npz_path), **per_sample)
    return {"metrics": metrics, "per_sample": per_sample, "metrics_path": metrics_path,
            "npz_path": npz_path, "run_dir": run_dir}


@pytest.fixture(scope="session")
def tiny_cache_v4(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, object]:
    r"""Build a tiny but **full-geometry** ($5280$) v4 raw cache once for the whole session.

    Loads ``config_synth_v4.yaml`` and calls :func:`build_dataset_v4.build_all_v4` with a small
    ``grid_override`` (one null + one signal cell) and ``n_override`` (a handful of rows per split)
    into a session ``tmp`` dir. The raw waveforms are the real $5280$-sample length on the $330$-step
    decimated grid; only the cell count and ``n`` shrink, so the cache is cheap yet
    geometry-compatible with a production-geometry model
    (``model_raw.testing.conftest.make_small_prod_raw_model``). Shared by the Sprint 3-6 dataset /
    datamodule / batch-contract / eval tests.

    Returns:
        ``{"config": dict, "cache_dir": Path, "cells": list, "grid": dict, "n_override": dict}``.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v4 import (
        build_all_v4,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.cells_v4 import (
        enumerate_cells_v4,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import load_config

    config_path = Path(__file__).resolve().parents[1] / "config_synth_v4.yaml"
    config = load_config(str(config_path))
    grid = {"target_te_grid": [0.0, 2.0], "lag_grid": [8]}
    n_override = {"train": 8, "val": 4, "test": 4}

    cache_dir = tmp_path_factory.mktemp("tiny_cache_v4")
    build_all_v4(
        config, benchmark="G1_raw_v4", out_dir=cache_dir,
        grid_override=grid, n_override=n_override,
    )
    cells, _ = enumerate_cells_v4(
        config, benchmark="G1_raw_v4",
        target_te_grid=grid["target_te_grid"], lag_grid=grid["lag_grid"],
    )
    return {
        "config": config,
        "cache_dir": Path(cache_dir),
        "cells": cells,
        "grid": grid,
        "n_override": n_override,
    }
