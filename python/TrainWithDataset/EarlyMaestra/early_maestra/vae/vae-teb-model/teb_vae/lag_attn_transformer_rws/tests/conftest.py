r"""Shared pytest configuration for the causal conv-Transformer lag-attention VAE tests.

Puts the repository root on ``sys.path`` so the absolute ``teb_vae.*`` imports resolve no matter
which directory pytest is invoked from, and exposes the fixtures the suite is built on. Mirrors
``teb_vae/lag_attn_rws/tests/conftest.py``, including its ``utils`` pre-import pin.

Three things are imported from the sibling suite rather than restated: ``perturb_posterior``,
``make_stub_batch`` and ``absolutize_dataset_paths``. They describe the *data* and the *trap*, both
of which are shared -- the batch contract is the same one, and the posterior delta heads are
zero-initialised in both models, so at initialisation every KL assertion passes vacuously in both.

The constructor keyword sets are defined fresh, because they are the one thing that is not shared:
this model has no ``lstm_layers``, ``encoder_extra_dilations``, ``encoder_extra_kernel``,
``conv_norm_groups`` or ``causal_norm``, and it has seven encoder keys the sibling has never heard
of. A copy of the sibling's set would not construct.

The causality probe lives here too. Every invariant in this suite is measured the same way -- resample
the strict future, require bit-stability at the cut *and* visible movement at the end -- and the second
half is the negative control without which a dead layer passes every causality test in the package.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pytest
import torch

# teb_vae/lag_attn_transformer_rws/tests/conftest.py -> parents[0]=tests,
# [1]=lag_attn_transformer_rws, [2]=teb_vae, [3]=repo root.
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

# Importing the fixture binds it in this conftest's namespace, which is all pytest needs to serve
# it to the tests in this directory.
from teb_vae.lag_attn.tests.conftest import perturb_posterior  # noqa: E402,F401
from teb_vae.lag_attn_rws.tests.conftest import (  # noqa: E402,F401
    BATCH,
    STUB_GAP_STEP,
    absolutize_dataset_paths,
    make_stub_batch,
)

# Tiny but structurally faithful: ``num_heads * d_head == d_model``, ``d_z % num_heads == 0`` (the
# posterior is head-structured), ``warmup < T - horizon``, and the encoder head width
# ``d_model // encoder_num_heads`` is even, which rotary position encoding requires. Raw length is
# ``sequence_length * raw_per_step = 256``, matching the stub batch.
#
# The encoder geometry is shrunk with one property preserved that the shipped one has and a naive
# miniature would lose: the source encoder's receptive-field bound stays strictly inside the
# sequence. Stem reach is $1 + (3-1)\cdot 1 + (3-1)\cdot 2 = 7$ steps, so the source bound is
# $7 + 2(4-1) = 13 < 16$. A bound that clamped at $T$ would make the measured-bound probe vacuous.
#
# ``horizon_film`` is on, matching the shipped config: per-block FiLM is hardcoded in the net, so
# the horizon core is built with ``film=horizon_film`` and ``film_per_block=True``, and
# ``horizon_film=false`` would fail fast at construction.
SEQ_LEN = 16

TINY_KWARGS: Dict[str, Any] = dict(
    sequence_length=SEQ_LEN,
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
    encoder_conv_kernels=(3, 3),
    encoder_conv_dilations=(1, 2),
    encoder_num_heads=4,
    encoder_d_ff=64,
    target_attention_blocks=2,
    source_attention_blocks=2,
    source_attention_window=4,
)

# What configs/default.yaml sets, at full production geometry. Unlike the tiny set this describes
# the real thing -- 300 steps, $d_z = 48$, entmax attention, the $(5, 9)$ stem, four full-context
# target blocks and three source blocks at a 16-step window -- so construction-time invariants are
# checked against the model that actually trains, not a miniature of it. Forward passes in this
# suite stay on TINY_KWARGS for speed.
#
# Five of the sibling's keys are deliberately absent: ``lstm_layers``,
# ``encoder_extra_dilations``, ``encoder_extra_kernel``, ``conv_norm_groups`` and ``causal_norm``.
# There is no recurrent branch, no extra dilation schedule and no time-pooling normaliser left to
# causalise, so each of them would reach nothing.
SHIPPED_KWARGS: Dict[str, Any] = dict(
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
    dropout=0.1,
    decoder_hidden=128,
    horizon_depth=3,
    horizon_kernel=3,
    horizon_film=True,
    horizon_embed_std=0.8,
    head_init_calibration=True,
    a_head_gain=2.0,
    logvar_clamp=(-5.0, 3.0),
    mu_scale=5.0,
    delta_mu_scale=3.0,
    delta_logvar_scale=2.0,
    use_entmax=True,
    attention_grad_checkpoint=False,
    lag_bias_init="alibi_decay",
    query_uses_logvar=False,
    coverage_floor=0.9,
    encoder_conv_kernels=(5, 9),
    encoder_conv_dilations=(1, 2),
    encoder_num_heads=4,
    encoder_d_ff=256,
    target_attention_blocks=4,
    source_attention_blocks=3,
    source_attention_window=16,
)

#: Relative movement below which two float32 activations are the same computation run twice.
#: float32 round-off on $O(1)$ activations through a handful of layers.
CAUSALITY_TOL = 1e-5

#: Relative movement a probe must *exceed* to count as having reached the module under test. The
#: paired half of every causality assertion: without it a module returning zeros, or a residual
#: branch that was never connected, passes every bit-stability check in this suite.
MOVEMENT_TOL = 1e-3


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``slow`` marker; there is no repo-wide pytest configuration to declare it."""
    config.addinivalue_line(
        "markers", "slow: long-running empirical validation, excluded from the default run"
    )


def resample_after(x: torch.Tensor, cut: int, *, seed: int = 0) -> torch.Tensor:
    """Return a copy of ``x`` whose steps strictly after ``cut`` are drawn afresh.

    A random resample rather than a constant offset, deliberately. The input adapter upstream of
    both encoders ends in a ``LayerNorm``, which removes a uniform channel shift outright, so a
    constant-offset probe would report causality that was never tested. ``RMSNorm`` inside the
    encoder does not centre, but the probe runs through both.

    Args:
        x: Sequence-major tensor $(B, T, C)$.
        cut: Last index left untouched. Everything at ``cut + 1`` and beyond is replaced.
        seed: Seed for the replacement draw, so a probe is reproducible.

    Returns:
        A new tensor shaped like ``x``, identical up to and including ``cut``.
    """
    generator = torch.Generator().manual_seed(seed)
    perturbed = x.clone()
    tail_shape = perturbed[:, cut + 1 :].shape
    perturbed[:, cut + 1 :] = torch.randn(tail_shape, generator=generator, dtype=x.dtype)
    return perturbed


def relative_change(reference: torch.Tensor, other: torch.Tensor) -> float:
    """Return $\\lVert a - b \\rVert / \\lVert a \\rVert$, or the absolute norm if ``a`` is zero.

    Args:
        reference: The unperturbed tensor.
        other: The tensor computed from the perturbed input.

    Returns:
        The relative movement as a Python float.
    """
    denominator = float(reference.norm())
    difference = float((reference - other).norm())
    return difference / denominator if denominator > 0.0 else difference


def assert_token_causal(
    forward: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    cut: int,
    *,
    label: str = "module",
    seed: int = 0,
) -> None:
    """Assert $H_t = f(X_{\\le t})$ by a paired future-perturbation probe.

    Resamples the input strictly after ``cut`` and requires two things at once: the output at
    ``cut`` must be unmoved, and the output at the last step must have moved. The second half is
    the negative control. This architecture has no time-pooling normaliser to flip -- the trick the
    sibling's leak test uses -- so the control is positional instead, and needs no switch in
    production code that exists only for tests.

    Args:
        forward: Maps $(B, T, C)$ to a sequence-major tensor.
        x: The unperturbed input.
        cut: The cutoff $t_0$. Must leave at least one step after it.
        label: Named in the failure message, so a parametrised run says which block failed.
        seed: Seed for the resample.
    """
    assert cut < x.shape[1] - 1, f"cut {cut} leaves no future to perturb in {x.shape[1]} steps"

    reference = forward(x)
    perturbed = forward(resample_after(x, cut, seed=seed))

    at_cut = relative_change(reference[:, cut], perturbed[:, cut])
    at_end = relative_change(reference[:, -1], perturbed[:, -1])
    assert at_cut < CAUSALITY_TOL, (
        f"{label}: output at t={cut} moved by {at_cut:.3e} when only the strict future changed"
    )
    assert at_end > MOVEMENT_TOL, (
        f"{label}: output at the last step moved by only {at_end:.3e} -- the perturbation never "
        f"reached the module, so the bit-stability above proves nothing"
    )


def build_stream_encoder(stream: str, kwargs: Optional[dict] = None, **overrides):
    """Build one stream's encoder from a constructor keyword set, the way the model will.

    The seven encoder keys map onto the two encoders in exactly one way, and it is the same map in
    every structural test file, so it is written here once. The target takes the full causal prefix
    and the source a bounded window; that asymmetry is the architecture, not a setting these tests
    are free to choose.

    Args:
        stream: ``"target"`` or ``"source"``.
        kwargs: A constructor keyword set. Defaults to :data:`TINY_KWARGS`.
        **overrides: Passed through to the encoder, overriding anything derived above. Dropout
            defaults to $0$ here rather than to the keyword set's value, because every structural
            probe in this suite compares two forward passes.

    Returns:
        A ``CausalConvTransformerEncoder`` in eval mode.

    Raises:
        ValueError: If ``stream`` is neither of the two.
    """
    from teb_vae.lag_attn_transformer_rws.nets.encoders import CausalConvTransformerEncoder

    if stream not in ("target", "source"):
        raise ValueError(f"stream must be 'target' or 'source', got {stream!r}")
    source = dict(TINY_KWARGS if kwargs is None else kwargs)
    settings = dict(
        d_model=int(source["d_model"]),
        sequence_length=int(source["sequence_length"]),
        conv_kernels=source["encoder_conv_kernels"],
        conv_dilations=source["encoder_conv_dilations"],
        num_attention_blocks=int(source[f"{stream}_attention_blocks"]),
        num_heads=int(source["encoder_num_heads"]),
        d_ff=int(source["encoder_d_ff"]),
        attention_window=(
            None if stream == "target" else int(source["source_attention_window"])
        ),
        dropout=0.0,
    )
    settings.update(overrides)
    return CausalConvTransformerEncoder(**settings).eval()


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
    """A two-sample stub batch at the tiny geometry, with the deliberate weight gap."""
    return make_stub_batch(BATCH, SEQ_LEN)


@pytest.fixture
def make_stub_batch_fn():
    """Factory fixture returning :func:`make_stub_batch`."""
    return make_stub_batch


# The loss hyperparameters the shipped config sets, as the task's constructor takes them.
# ``beta_schedule=None`` means the constant ``kld_beta`` applies, which keeps beta out of the way of
# tests that are not about the schedule. ``free_bits`` is genuinely 0.0 in the shipped config; tests
# about the raw/train KL split override it per-test.
TASK_HPARAMS: Dict[str, Any] = dict(
    lambda_full=1.0,
    lambda_base=1.0,
    likelihood="gaussian_nll",
    free_bits=0.0,
    kld_beta=1.0,
    beta_schedule=None,
)


def _make_task(model_kwargs: Optional[dict] = None, hparams: Optional[dict] = None, **task_kwargs):
    """Build a model wrapped in its task, with the production loss hparams applied.

    Imported lazily so the pure-net tests never pay for Lightning.

    Args:
        model_kwargs: Net constructor kwargs. Defaults to :data:`TINY_KWARGS`.
        hparams: Loss hparam overrides on top of :data:`TASK_HPARAMS`.
        **task_kwargs: Passed through to the task's constructor.

    Returns:
        A ``SeqVaeLagAttnTrfRwsTask`` with ``setup()`` already called, so the permutation
        generator exists exactly as it would under a real fit.
    """
    from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
    from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask

    kwargs = dict(TINY_KWARGS if model_kwargs is None else model_kwargs)
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**kwargs)
    task = SeqVaeLagAttnTrfRwsTask(
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

    The channel counts are the dataset's and are independent of model size: $43$ FHR scattering,
    $66$ FHR phase-harmonic, and $58$ for the concatenated UP stream. They track
    ``hdf5_dataset/new_pipeline/create_new_pipeline.py``; when its phase-harmonic selection
    changes, these move with it.
    """
    generator = torch.Generator().manual_seed(0)
    y_st = torch.randn(BATCH, SEQ_LEN, 43, generator=generator)
    y_ph = torch.randn(BATCH, SEQ_LEN, 66, generator=generator)
    u_stream = torch.randn(BATCH, SEQ_LEN, 58, generator=generator)
    return y_st, y_ph, u_stream
