"""Shared pytest configuration for the SeqVaeLagAttnV3 model unit tests.

Adds the repository root to ``sys.path`` (so the absolute ``model.vae_teb_prediction``
imports resolve no matter which directory pytest is invoked from -- mirroring the shim in
``model_experiment/synthetic_v2/tests/conftest.py``) and exposes small tiny-model fixtures
used across the v3 test suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# tests/ -> model/ -> vae_teb_prediction/ -> model/ -> <repo root>
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

# There are two ``utils`` packages in this repo: the real one at <repo root>/utils (holding
# ``utils.style``) and a near-empty one at model/vae_teb_prediction/utils. pytest prepends the
# latter's parent directory to ``sys.path`` when collecting these tests, which shadows the
# former and breaks ``from utils.style import ...`` inside the plotting callbacks. Binding the
# repo-root package into ``sys.modules`` now -- while ``_REPO_ROOT`` is still first -- pins its
# ``__path__`` for every later ``utils.<submodule>`` import.
import importlib  # noqa: E402

importlib.import_module("utils")

# Tiny but structurally faithful model config (all invariants satisfied:
# num_heads * d_head == d_model, d_z % num_heads == 0, warmup < T - horizon).
TINY_KWARGS = dict(
    sequence_length=16,
    d_model=32,
    d_z=8,
    horizon=4,
    warmup_period=2,
    c_y=87,
    c_u=101,
    use_up_st=True,
    max_lag=8,
    num_heads=4,
    d_head=8,
    dropout=0.0,
)

BATCH = 2
SEQ_LEN = TINY_KWARGS["sequence_length"]

# The v3 production path: every G0-G6 flag on. `lambda_perm`/`perm_every_n_batches` are
# small so the trainer tests can observe the schedule inside a handful of steps.
PROD_KWARGS = dict(
    TINY_KWARGS,
    causal_norm=True,          # G0
    posterior_logvar="residual",  # G1
    logvar_bound="smooth",     # G2
    kld_support="anchor",      # G3
    lag_bias_init="alibi_decay",  # G5
    lambda_perm=0.1,           # G6
    perm_every_n_batches=2,
    freeze_unused_attn_proj=True,  # only bites when head_structured_latent=True
)

#: hparams the Lightning wrapper reads; mirrors ``config_lag_attn_v3.yaml``'s VAE_model block.
PROD_HPARAMS = dict(
    lambda_full=1.0,
    lambda_base=0.5,
    likelihood="gaussian_nll",
    sigma_obs="learned",       # G7
    free_bits=0.1,
    detach_baseline_in_full=True,
    lambda_lag=1.0e-3,
    kld_beta=0.01,
    beta_schedule=None,
    loss_spike_skip={},
)


@pytest.fixture
def tiny_kwargs() -> dict:
    """A fresh copy of the tiny-model constructor kwargs (safe to mutate)."""
    return dict(TINY_KWARGS)


@pytest.fixture
def prod_kwargs() -> dict:
    """A fresh copy of the tiny-model kwargs with every v3 production flag enabled."""
    return dict(PROD_KWARGS)


@pytest.fixture
def inputs():
    """Seeded ``(y_st, y_ph, u_stream)`` tensors matching the tiny config."""
    g = torch.Generator().manual_seed(0)
    y_st = torch.randn(BATCH, SEQ_LEN, 43, generator=g)
    y_ph = torch.randn(BATCH, SEQ_LEN, 44, generator=g)
    u = torch.randn(BATCH, SEQ_LEN, 101, generator=g)
    return y_st, y_ph, u


def make_stub_batch(batch_size: int = 4, seq_len: int = SEQ_LEN, seed: int = 0):
    """A batch object exposing the fields ``SeqVaeLagAttnPl._build_source_stream`` reads."""
    import types

    g = torch.Generator().manual_seed(seed)
    return types.SimpleNamespace(
        fhr_st=torch.randn(batch_size, seq_len, 43, generator=g),
        fhr_ph=torch.randn(batch_size, seq_len, 44, generator=g),
        up_st=torch.randn(batch_size, seq_len, 43, generator=g),
        up_ph=torch.randn(batch_size, seq_len, 58, generator=g),
        weight=torch.ones(batch_size, seq_len),
    )


@pytest.fixture
def stub_batch():
    """A four-sample stub batch (large enough to derange)."""
    return make_stub_batch()


@pytest.fixture
def make_stub_batch_fn():
    """Factory fixture returning :func:`make_stub_batch`."""
    return make_stub_batch


def _make_v3_pl(model_kwargs: dict | None = None, hparams: dict | None = None):
    """Build a ``SeqVaeLagAttnV3`` wrapped in its Lightning module, with hparams applied.

    The trainer module is imported lazily so the pure-model tests never pay for Lightning,
    matplotlib, and the HDF5 dataloader.
    """
    from model.vae_teb_prediction.model.trainer_lag_attn_v3 import SeqVaeLagAttnV3Pl
    from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3

    kwargs = dict(PROD_KWARGS if model_kwargs is None else model_kwargs)
    torch.manual_seed(0)
    model = SeqVaeLagAttnV3(**kwargs)
    pl_module = SeqVaeLagAttnV3Pl(model, lr=1e-3, model_kwargs=kwargs)
    for key, value in dict(PROD_HPARAMS, **(hparams or {})).items():
        pl_module.hparams[key] = value
    return pl_module


@pytest.fixture
def v3_pl():
    """Factory fixture: ``v3_pl(model_kwargs=None, hparams=None) -> SeqVaeLagAttnV3Pl``."""
    return _make_v3_pl


@pytest.fixture
def perturb_posterior():
    """Factory fixture that breaks a model's zero-init so its KL terms become nonzero.

    At initialisation both posterior delta heads output exactly 0, so every KL -- true and
    shuffled alike -- is 0 and any assertion about them would pass vacuously.
    """

    def _perturb(model, seed: int = 3, scale: float = 0.1) -> None:
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in model.posterior_head.parameters():
                p.add_(torch.randn(p.shape, generator=g) * scale)

    return _perturb
