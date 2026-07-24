"""Shared pytest configuration for the raw-signal lag-attention VAE tests.

Puts the repository root on ``sys.path`` so the absolute ``teb_vae.*`` imports resolve no matter
which directory pytest is invoked from, and exposes the tiny-model fixtures the suite is built
on. Mirrors ``teb_vae/lag_attn/tests/conftest.py``, including its ``utils`` pre-import pin.

The fixtures are deliberately small. A structurally faithful model at $d_{model} = 32$ and
$T = 16$ exercises every code path a production-scale one does, in milliseconds and on a CPU.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

# teb_vae/lag_attn_rws/tests/conftest.py -> parents[0]=tests, [1]=lag_attn_rws, [2]=teb_vae,
# [3]=repo root.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

# There are two ``utils`` packages in this repo: the real one at <repo root>/utils and a
# near-empty one at model/vae_teb_prediction/utils. On a repository-wide run another conftest
# can put the latter's parent first on ``sys.path``, shadowing the real one. Binding the
# repo-root package now -- while ``_REPO_ROOT`` is still first -- pins its ``__path__`` for
# every later ``utils.<submodule>`` import.
try:
    importlib.import_module("utils")
except Exception:
    pass

# The perturbation fixture is shared with the sibling model: the posterior delta heads are
# zero-initialised in both, so at init every KL assertion passes vacuously, and both suites
# need the same "perturb first, then assert" escape from that trap. Importing the fixture
# binds it in this conftest's namespace, which is all pytest needs to serve it here.
from teb_vae.lag_attn.tests.conftest import perturb_posterior  # noqa: E402,F401

# Tiny but structurally faithful: num_heads * d_head == d_model, d_z % num_heads == 0 (the
# posterior is head-structured), and warmup < T - horizon, so every invariant the constructor
# enforces is satisfied. Raw length is sequence_length * raw_per_step = 256.
#
# horizon_film is on, matching the shipped config: per-block FiLM is hardcoded in the net, so the
# horizon core is built with film=horizon_film and film_per_block=True, and horizon_film=false
# would fail fast at construction. Keeping it on here is what makes every contract test exercise
# the per-block-FiLM decoder the production model actually runs.
TINY_KWARGS = dict(
    sequence_length=16,
    d_model=32,
    d_z=8,
    horizon=4,
    raw_per_step=16,
    warmup_period=2,
    c_y=109,
    c_u=58,
    use_up_st=True,
    max_lag=8,
    num_heads=4,
    d_head=8,
    horizon_film=True,
    dropout=0.0,
)

BATCH = 2
SEQ_LEN = int(TINY_KWARGS["sequence_length"])

# What configs/default.yaml sets, at full production geometry. Unlike the tiny set this builds
# the real thing -- 300 steps, d_z = 48, entmax attention, causal norms, the extra encoder
# dilations -- so construction-time invariants are checked against the model that actually
# trains, not a miniature of it. Forward passes stay on TINY_KWARGS for speed.
SHIPPED_KWARGS = dict(
    sequence_length=300,
    d_model=128,
    d_z=48,
    horizon=30,
    raw_per_step=16,
    warmup_period=30,
    c_y=109,
    c_u=58,
    use_up_st=True,
    max_lag=90,
    num_heads=4,
    d_head=32,
    lstm_layers=2,
    dropout=0.1,
    decoder_hidden=128,
    horizon_depth=3,
    horizon_kernel=3,
    horizon_film=True,
    horizon_embed_std=0.8,
    head_init_calibration=True,
    a_head_gain=2.0,
    encoder_extra_dilations=(8, 16),
    encoder_extra_kernel=15,
    conv_norm_groups=None,
    logvar_clamp=(-5.0, 3.0),
    mu_scale=5.0,
    delta_mu_scale=3.0,
    delta_logvar_scale=2.0,
    use_entmax=True,
    attention_grad_checkpoint=False,
    lag_bias_init="alibi_decay",
    query_uses_logvar=False,
    causal_norm=True,
    coverage_floor=0.9,
)

#: The decimated step index of the deliberate gap every stub batch carries. Chosen inside the
#: tiny trained-anchor range [warmup, T - H) = [2, 12), so the gap is visible to every mask.
STUB_GAP_STEP = 10


def absolutize_dataset_paths(config: dict) -> dict:
    """Rewrite the tiny config's shard and statistics paths to absolute, in place.

    The shipped paths are repo-root-relative because the entry points run from the repo root;
    a test that drives the loader from pytest's working directory needs them absolute. Shared
    rather than repeated per test file: a renamed dataset key would otherwise have to be fixed in
    every copy, and a miss surfaces as the loader's opaque "No samples match the specified
    filters" rather than as a path error.

    Args:
        config: A loaded config dict.

    Returns:
        The same dict.
    """
    dataset = config["dataset_config"]
    for key in ("vae_train_datasets", "vae_test_datasets"):
        dataset[key] = [str(Path(_REPO_ROOT) / path) for path in dataset[key]]
    dataset["stat_path"] = str(Path(_REPO_ROOT) / dataset["stat_path"])
    return config


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``slow`` marker; there is no repo-wide pytest configuration to declare it."""
    config.addinivalue_line(
        "markers", "slow: long-running empirical validation, excluded from the default run"
    )


def make_stub_batch(batch_size: int = BATCH, seq_len: int = SEQ_LEN, seed: int = 0):
    """Build a batch exposing the fields the task reads, raw target included.

    A ``SimpleNamespace`` rather than the real ``AttributeDict``: the task reads batch fields
    as attributes, and standing up an HDF5 loader to test a loss would couple every task test
    to the data layer. The real batch contract is asserted against the committed shard in
    ``test_data_contract.py``.

    The ``weight`` carries a deliberate gap at :data:`STUB_GAP_STEP`. That gap is load-bearing:
    a uniformly valid weight would leave every mask test green whether or not the masks work.

    Args:
        batch_size: Samples in the batch. Must be at least 2 to be derangeable.
        seq_len: Decimated sequence length; the raw signals are ``16 * seq_len`` long.
        seed: Seed, so a batch is reproducible.

    Returns:
        An object with ``fhr_st``, ``fhr_ph``, ``up_st``, ``up_ph``, ``fhr``, ``up`` and
        ``weight``.
    """
    generator = torch.Generator().manual_seed(seed)
    weight = torch.ones(batch_size, seq_len)
    weight[:, STUB_GAP_STEP] = 0.0
    return types.SimpleNamespace(
        fhr_st=torch.randn(batch_size, seq_len, 43, generator=generator),
        fhr_ph=torch.randn(batch_size, seq_len, 66, generator=generator),
        up_st=torch.randn(batch_size, seq_len, 43, generator=generator),
        up_ph=torch.randn(batch_size, seq_len, 15, generator=generator),
        fhr=torch.randn(batch_size, 16 * seq_len, generator=generator),
        up=torch.randn(batch_size, 16 * seq_len, generator=generator),
        weight=weight,
    )


@pytest.fixture
def tiny_kwargs() -> dict:
    """A fresh copy of the tiny-model constructor kwargs (safe to mutate)."""
    return dict(TINY_KWARGS)


@pytest.fixture
def shipped_kwargs() -> dict:
    """A fresh copy of the production constructor kwargs (safe to mutate)."""
    return dict(SHIPPED_KWARGS)


@pytest.fixture
def stub_batch():
    """A two-sample stub batch with the deliberate weight gap."""
    return make_stub_batch()


@pytest.fixture
def make_stub_batch_fn():
    """Factory fixture returning :func:`make_stub_batch`."""
    return make_stub_batch


# The loss hyperparameters the shipped config sets, as the task's constructor takes them.
# `beta_schedule=None` means the constant `kld_beta` applies, which keeps beta out of the way of
# tests that are not about the schedule. `free_bits` is genuinely 0.0 in the shipped config;
# tests about the raw/train KL split override it per-test.
TASK_HPARAMS = dict(
    lambda_full=1.0,
    lambda_base=1.0,
    likelihood="gaussian_nll",
    free_bits=0.0,
    kld_beta=1.0,
    beta_schedule=None,
)


def _make_task(model_kwargs: dict | None = None, hparams: dict | None = None, **task_kwargs):
    """Build a model wrapped in its task, with the production loss hparams applied.

    Imported lazily so the pure-net tests never pay for Lightning.

    Args:
        model_kwargs: Net constructor kwargs. Defaults to ``TINY_KWARGS``.
        hparams: Loss hparam overrides on top of ``TASK_HPARAMS``.
        **task_kwargs: Passed through to the task's constructor.

    Returns:
        A ``SeqVaeLagAttnRwsTask`` with ``setup()`` already called, so the permutation
        generator exists exactly as it would under a real fit.
    """
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
    from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask

    kwargs = dict(TINY_KWARGS if model_kwargs is None else model_kwargs)
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**kwargs)
    task = SeqVaeLagAttnRwsTask(
        model,
        lr=1e-3,
        model_kwargs=kwargs,
        **dict(TASK_HPARAMS, **(hparams or {})),
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
    """Seeded ``(y_st, y_ph, u_stream)`` tensors matching the tiny geometry.

    The channel counts are the dataset's, and are independent of model size: $43$ FHR
    scattering, $66$ FHR phase-harmonic, and $58$ for the concatenated UP stream
    ``[up_st(43), up_ph(15)]``. They track
    ``hdf5_dataset/new_pipeline/create_new_pipeline.py``; when its phase-harmonic selection
    changes, these move with it.
    """
    generator = torch.Generator().manual_seed(0)
    y_st = torch.randn(BATCH, SEQ_LEN, 43, generator=generator)
    y_ph = torch.randn(BATCH, SEQ_LEN, 66, generator=generator)
    u_stream = torch.randn(BATCH, SEQ_LEN, 58, generator=generator)
    return y_st, y_ph, u_stream
