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

from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pytest

__all__ = [
    "signal_kbar_fixture",
    "source_exploiting_outputs",
    "planted_lag_te_lag_map",
    "tiny_raw_checkpoint",
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
