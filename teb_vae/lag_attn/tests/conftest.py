"""Shared pytest configuration for the lag-attention VAE tests.

Puts the repository root on ``sys.path`` so the absolute ``teb_vae.lag_attn`` imports resolve no
matter which directory pytest is invoked from, and exposes the tiny-model fixtures the suite is
built on. Mirrors ``train/tests/conftest.py``.

The fixtures here are deliberately small. A structurally faithful model at $d_{model} = 32$ and
$T = 16$ exercises every code path a production-scale one does, in milliseconds and on a CPU.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch

# teb_vae/lag_attn/tests/conftest.py -> parents[0]=tests, [1]=lag_attn, [2]=teb_vae, [3]=repo root.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

# There are two ``utils`` packages in this repo: the real one at <repo root>/utils and a
# near-empty one at model/vae_teb_prediction/utils. On a repository-wide run another conftest can
# put the latter's parent first on ``sys.path``, shadowing the real one. Binding the repo-root
# package now -- while ``_REPO_ROOT`` is still first -- pins its ``__path__`` for every later
# ``utils.<submodule>`` import.
try:
    importlib.import_module("utils")
except Exception:
    pass

# Tiny but structurally faithful: num_heads * d_head == d_model, d_z % num_heads == 0, and
# warmup < T - horizon, so every invariant the constructor enforces is satisfied.
TINY_KWARGS = dict(
    sequence_length=16,
    d_model=32,
    d_z=8,
    horizon=4,
    warmup_period=2,
    c_y=109,
    c_u=58,
    use_up_st=True,
    max_lag=8,
    num_heads=4,
    d_head=8,
    dropout=0.0,
)

BATCH = 2
SEQ_LEN = int(TINY_KWARGS["sequence_length"])

# The flag set the original's own test suite called "prod", ported under its original name for
# continuity with that suite. `lambda_perm` and `perm_every_n_batches` are small so a schedule
# is observable inside a handful of steps.
#
# It is a misnomer, and worth knowing why: it leaves four flags the shipped config sets at their
# constructor defaults. See SHIPPED_KWARGS. Smooth log-variance bounding and a residual
# posterior log-variance are absent for a different reason -- they are no longer choices, so
# passing them would be a TypeError.
PROD_KWARGS = dict(
    TINY_KWARGS,
    causal_norm=True,
    kld_support="anchor",
    lag_bias_init="alibi_decay",
    lambda_perm=0.1,
    perm_every_n_batches=2,
    freeze_unused_attn_proj=True,  # only bites when head_structured_latent=True
)

# What configs/default.yaml actually ships, at the tiny geometry.
#
# Each of these four gates real modules that PROD_KWARGS never builds: the per-head posterior,
# the FiLM generator, a third refine block, and two extra conv blocks per encoder. Testing only
# PROD_KWARGS leaves roughly a third of the production model's parameters unexercised -- and
# `freeze_unused_attn_proj` inert, since it requires head structure to do anything.
SHIPPED_KWARGS = dict(
    PROD_KWARGS,
    use_entmax=True,
    head_structured_latent=True,
    horizon_depth=3,
    horizon_film=True,
    encoder_extra_dilations=(8, 16),
)


@pytest.fixture
def tiny_kwargs() -> dict:
    """A fresh copy of the tiny-model constructor kwargs (safe to mutate)."""
    return dict(TINY_KWARGS)


@pytest.fixture
def prod_kwargs() -> dict:
    """A fresh copy of the original suite's flag set (safe to mutate)."""
    return dict(PROD_KWARGS)


@pytest.fixture
def shipped_kwargs() -> dict:
    """A fresh copy of the flags the shipped config actually sets (safe to mutate)."""
    return dict(SHIPPED_KWARGS)


# The loss hyperparameters the shipped config sets, as the task's constructor takes them.
# `beta_schedule=None` means the constant `kld_beta` applies, which keeps beta out of the way of
# tests that are not about the schedule.
PROD_HPARAMS = dict(
    lambda_full=1.0,
    lambda_base=0.5,
    likelihood="gaussian_nll",
    sigma_obs="learned",
    free_bits=0.1,
    detach_baseline_in_full=True,
    lambda_lag=1.0e-3,
    kld_beta=0.01,
    beta_schedule=None,
)


def make_stub_batch(batch_size: int = 4, seq_len: int = SEQ_LEN, seed: int = 0):
    """Build a batch exposing the fields the task reads.

    A ``SimpleNamespace`` rather than the real ``AttributeDict``: the task reads batch fields as
    attributes, and standing up an HDF5 loader to test a loss would couple every task test to the
    data layer. The real batch contract is asserted against the committed shard in
    ``test_data_contract.py``, which is where that belongs.

    Args:
        batch_size: Samples in the batch. Must be at least 2 to be derangeable.
        seq_len: Sequence length.
        seed: Seed, so a batch is reproducible.

    Returns:
        An object with ``fhr_st``, ``fhr_ph``, ``up_st``, ``up_ph`` and ``weight``.
    """
    import types

    generator = torch.Generator().manual_seed(seed)
    return types.SimpleNamespace(
        fhr_st=torch.randn(batch_size, seq_len, 43, generator=generator),
        fhr_ph=torch.randn(batch_size, seq_len, 66, generator=generator),
        up_st=torch.randn(batch_size, seq_len, 43, generator=generator),
        up_ph=torch.randn(batch_size, seq_len, 15, generator=generator),
        weight=torch.ones(batch_size, seq_len),
    )


@pytest.fixture
def stub_batch():
    """A four-sample stub batch, large enough to derange."""
    return make_stub_batch()


@pytest.fixture
def make_stub_batch_fn():
    """Factory fixture returning :func:`make_stub_batch`."""
    return make_stub_batch


def _make_task(model_kwargs: dict | None = None, hparams: dict | None = None, **task_kwargs):
    """Build a model wrapped in its task, with the production loss hparams applied.

    Imported lazily so the pure-net tests never pay for Lightning.

    Args:
        model_kwargs: Net constructor kwargs. Defaults to ``PROD_KWARGS``.
        hparams: Loss hparam overrides on top of ``PROD_HPARAMS``.
        **task_kwargs: Passed through to the task's constructor.

    Returns:
        A ``SeqVaeLagAttnTask`` with ``setup()`` already called, so the permutation generator
        exists exactly as it would under a real fit.
    """
    from teb_vae.lag_attn.nets.model import SeqVaeLagAttn
    from teb_vae.lag_attn.task import SeqVaeLagAttnTask

    kwargs = dict(PROD_KWARGS if model_kwargs is None else model_kwargs)
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**kwargs)
    task = SeqVaeLagAttnTask(
        model,
        lr=1e-3,
        model_kwargs=kwargs,
        **dict(PROD_HPARAMS, **(hparams or {})),
        **task_kwargs,
    )
    task.setup("fit")  # seeds the permutation generator; Lightning would call this itself
    return task


@pytest.fixture
def task():
    """Factory fixture: ``task(model_kwargs=None, hparams=None, **task_kwargs)``."""
    return _make_task


@pytest.fixture
def inputs():
    """Seeded ``(y_st, y_ph, u_stream)`` tensors matching the tiny config.

    The channel counts are the dataset's, and are independent of model size: $43$ FHR scattering,
    $66$ FHR phase-harmonic, and $58$ for the concatenated UP stream ``[up_st(43), up_ph(15)]``.
    They track ``hdf5_dataset/new_pipeline/create_new_pipeline.py``; when its phase-harmonic
    selection changes, these move with it.
    """
    generator = torch.Generator().manual_seed(0)
    y_st = torch.randn(BATCH, SEQ_LEN, 43, generator=generator)
    y_ph = torch.randn(BATCH, SEQ_LEN, 66, generator=generator)
    u_stream = torch.randn(BATCH, SEQ_LEN, 58, generator=generator)
    return y_st, y_ph, u_stream


@pytest.fixture
def perturb_posterior():
    """Factory fixture that breaks a model's zero-init so its KL terms become nonzero.

    Load-bearing. The posterior delta heads are zero-initialised, so at init the posterior
    equals the prior exactly and every KL -- true and shuffled alike -- is $0$. Any assertion
    about a KL therefore passes vacuously on an untouched model, including on a model that is
    completely wrong. Perturb first, then assert.
    """

    def _perturb(model, seed: int = 3, scale: float = 0.1) -> None:
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for parameter in model.posterior_head.parameters():
                parameter.add_(torch.randn(parameter.shape, generator=generator) * scale)

    return _perturb
