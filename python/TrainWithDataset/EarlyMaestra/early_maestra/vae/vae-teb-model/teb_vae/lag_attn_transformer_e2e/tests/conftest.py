r"""Shared pytest configuration for the end-to-end causal conv-Transformer lag-attention VAE.

Puts the repository root on ``sys.path`` so the absolute ``teb_vae.*`` imports resolve no matter
which directory pytest is invoked from, and exposes the fixtures the suite is built on. Mirrors
``teb_vae/lag_attn_transformer_rws/tests/conftest.py``, including its ``utils`` pre-import pin.

Almost nothing here is new. The data fixtures are the sibling suites' -- ``perturb_posterior``,
``make_stub_batch`` and ``absolutize_dataset_paths`` -- because they describe the *data* and the
*trap*, neither of which is a property of any model: the batch contract is the same one, and the
posterior delta heads are zero-initialised in all three models, so at initialisation every KL
assertion passes vacuously in all three. The token-causality probe and its two tolerances are
imported for the same reason: every structural invariant in this package is measured the same way
the sibling's are, and a second copy of a probe is a second thing to get wrong.

What is local is the one thing that is genuinely different. This model's history states are claimed
causal at **raw-sample** resolution rather than at token resolution, so the probe that measures
that claim -- :func:`assert_raw_causal` -- takes a raw, time-last input and a raw cut index, and
compares one *token* of the output. It is written here because there is nothing like it in either
sibling: their inputs arrive already on the token grid, so their probes cannot express a cut that
falls between two tokens.

The constructor keyword sets are defined fresh too. This model has no ``c_y``, ``c_u`` or
``use_up_st`` -- there are no feature blocks to declare a width for -- and it has a front end whose
kernels the tiny geometry shrinks. A copy of either sibling's set would not construct.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pytest
import torch

# teb_vae/lag_attn_transformer_e2e/tests/conftest.py -> parents[0]=tests,
# [1]=lag_attn_transformer_e2e, [2]=teb_vae, [3]=repo root.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

# There are two ``utils`` packages in this repo: the real one at <repo root>/utils and a near-empty
# one at model/vae_teb_prediction/utils. On a repository-wide run another conftest can put the
# latter's parent first on ``sys.path``, shadowing the real one. Binding the repo-root package now
# -- while ``_REPO_ROOT`` is still first -- pins its ``__path__`` for every later ``utils.<sub>``
# import.
try:
    importlib.import_module("utils")
except Exception:
    pass

# Importing a fixture binds it in this conftest's namespace, which is all pytest needs to serve it
# to the tests in this directory. The probe helpers and their tolerances are imported rather than
# restated for the same reason the data fixtures are: they describe how a causality claim is
# measured in this repository, and two copies could drift into two different claims.
from teb_vae.lag_attn.tests.conftest import perturb_posterior  # noqa: E402,F401
from teb_vae.lag_attn_rws.tests.conftest import (  # noqa: E402,F401
    BATCH,
    STUB_GAP_STEP,
    absolutize_dataset_paths,
    make_stub_batch,
)
from teb_vae.lag_attn_transformer_rws.tests.conftest import (  # noqa: E402,F401
    CAUSALITY_TOL,
    MOVEMENT_TOL,
    assert_token_causal,
    relative_change,
    resample_after,
)

#: Decimated sequence length of the tiny geometry. The raw signals the stub batch carries are
#: ``16 * SEQ_LEN`` long, which is what this model's front ends consume.
SEQ_LEN = 16

#: The tiny front-end kernel schedule, one gated depthwise kernel per stride-2 stage.
#:
#: Its own, not a shrunken copy of anything: the front end is the one component with no sibling.
#: Widest first, narrowing with depth, because stage $1$ runs at the raw $4$ Hz rate where the
#: structure is finest and stage $4$ at the $0.25$ Hz token rate where it is coarsest.
TINY_FRONTEND_KERNELS = (5, 3, 3, 3)

#: Warm-up steps of the tiny geometry, which is also the front end's reach budget: a front end may
#: reach no further back than ``warmup_period * raw_per_step`` raw samples, or a trained anchor
#: would read the zero-padded convolution transient at the segment's start.
#:
#: Six rather than the siblings' two, and the difference is forced. A four-stage stride-2 cascade
#: reaches
#:
#:     $$R = 2 + \sum_{i=1}^{4} (k_i + \tau - 2)\, 2^{\,i-1}$$
#:
#: raw samples, where $k_i$ is stage $i$'s kernel and $\tau$ the anti-alias filter's tap count: the
#: featurisation's first difference costs $2$, and each stage's cost is multiplied by the stride
#: already accumulated below it. The decimation alone therefore costs $2 + (\tau - 1) \cdot 15$,
#: which is already the siblings' whole budget of $2 \times 16 = 32$ raw samples at a three-tap
#: filter -- leaving room for no kernel wider than a single sample. At the kernels above the reach
#: is $64$ raw samples for a three-tap filter, against a budget of $6 \times 16 = 96$: margin
#: enough that a wider filter does not move the geometry.
#:
#: Legal because the constructor invariant is ``warmup_period < sequence_length - horizon``, which
#: is $16 - 4 = 12$ here; and the trained-anchor range $[6, 12)$ still contains the stub batch's
#: planted weight gap, without which every mask test in the suite would be vacuous.
TINY_WARMUP_PERIOD = 6

# Tiny but structurally faithful: ``num_heads * d_head == d_model``, ``d_z % num_heads == 0`` (the
# posterior is head-structured), ``warmup < T - horizon``, and the encoder head width
# ``d_model // encoder_num_heads`` is even, which rotary position encoding requires. Raw length is
# ``sequence_length * raw_per_step = 256``, matching the stub batch.
#
# The encoder geometry is the sibling's, shrunk the same way and with the same property preserved:
# the source encoder's receptive-field bound stays strictly inside the sequence. Stem reach is
# $1 + (3-1)\cdot 1 + (3-1)\cdot 2 = 7$ steps, so the source bound is $7 + 2(4-1) = 13 < 16$. A
# bound that clamped at $T$ would make the measured-bound probe vacuous.
#
# ``horizon_film`` is on, matching the shipped config: per-block FiLM is hardcoded in the net, so
# the horizon core is built with ``film=horizon_film`` and ``film_per_block=True``, and
# ``horizon_film=false`` would fail fast at construction.
TINY_KWARGS: Dict[str, Any] = dict(
    sequence_length=SEQ_LEN,
    d_model=32,
    d_z=8,
    horizon=4,
    raw_per_step=16,
    warmup_period=TINY_WARMUP_PERIOD,
    max_lag=8,
    num_heads=4,
    d_head=8,
    horizon_film=True,
    dropout=0.0,
    frontend_kernels=TINY_FRONTEND_KERNELS,
    encoder_conv_kernels=(3, 3),
    encoder_conv_dilations=(1, 2),
    encoder_num_heads=4,
    encoder_d_ff=64,
    target_attention_blocks=2,
    source_attention_blocks=2,
    source_attention_window=4,
)

# What configs/default.yaml sets, at full production geometry -- 300 steps, $d_z = 64$, entmax
# attention, the $(5, 9)$ stem, six full-context target blocks and three source blocks at a
# 16-step window, a $256$-wide decoder core four refine blocks deep with two horizon-attention
# blocks on top -- so construction-time invariants are checked against the model that actually
# trains rather than a miniature of it. Forward passes in this suite stay on TINY_KWARGS for speed.
#
# Leaf for leaf the sibling's production set minus three keys, which is the comparison this package
# exists to make: ``c_y``, ``c_u`` and ``use_up_st`` declare the widths of stored feature blocks
# this model does not load. No front-end key appears either -- the stage widths are derived from
# ``d_model`` and the production kernels are the constructor's own default -- so there is nothing
# for a config to set and nothing for a signature sweep to drop in silence.
SHIPPED_KWARGS: Dict[str, Any] = dict(
    sequence_length=300,
    d_model=128,
    d_z=64,
    horizon=30,
    raw_per_step=16,
    warmup_period=30,
    max_lag=90,
    num_heads=4,
    d_head=32,
    dropout=0.1,
    decoder_hidden=256,
    horizon_depth=4,
    horizon_kernel=3,
    horizon_film=True,
    horizon_attention_blocks=2,
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
    encoder_d_ff=512,
    target_attention_blocks=6,
    source_attention_blocks=3,
    source_attention_window=16,
)


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``slow`` marker; there is no repo-wide pytest configuration to declare it."""
    config.addinivalue_line(
        "markers", "slow: long-running empirical validation, excluded from the default run"
    )


# ---------------------------------------------------------------------------------------
# The raw-resolution causality probe
# ---------------------------------------------------------------------------------------
def resample_raw_after(x: torch.Tensor, cut: int, *, seed: int = 0) -> torch.Tensor:
    """Return a copy of ``x`` whose samples strictly after ``cut`` on the **last** axis are redrawn.

    The last axis, not axis $1$, which is what separates this from the imported
    :func:`resample_after`. A raw input is time-last -- $(B, L)$ as the model takes it, $(B, C, L)$
    once featurised -- while the sibling's token-major $(B, T, C)$ probe perturbs axis $1$. Applied
    to a raw batch that would resample *channels*, and every causality assertion built on it would
    be about nothing.

    A random resample rather than a constant offset, for the sibling's reason: a normaliser that
    removes a uniform shift would report causality that was never tested.

    Args:
        x: A time-last tensor, $(B, L)$ or $(B, C, L)$.
        cut: Last raw index left untouched. Everything at ``cut + 1`` and beyond is replaced.
        seed: Seed for the replacement draw, so a probe is reproducible.

    Returns:
        A new tensor shaped like ``x``, identical up to and including ``cut``.
    """
    generator = torch.Generator().manual_seed(seed)
    perturbed = x.clone()
    tail = perturbed[..., cut + 1 :]
    perturbed[..., cut + 1 :] = torch.randn(
        tail.shape, generator=generator, dtype=x.dtype
    )
    return perturbed


def assert_raw_causal(
    forward: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    cut: int,
    stride: int,
    *,
    label: str = "module",
    seed: int = 0,
) -> None:
    r"""Assert that output token $\lfloor \mathrm{cut} / \mathrm{stride} \rfloor$ reads no raw
    sample after ``cut``.

    The raw-resolution counterpart of :func:`assert_token_causal`, and the assertion this package's
    central claim is measured by. Resamples the raw input strictly after ``cut`` and requires two
    things at once: the output token whose causal endpoint is ``cut`` must be **bitwise** unmoved,
    and the last token must have moved. The second half is the negative control, without which a
    dead stage or an unconnected residual branch passes every causality assertion in the package.

    Bitwise rather than thresholded, because a threshold at this boundary is a statement about
    float32 round-off rather than about causality; drive the probe in float64 with a large
    amplitude and the two halves separate by many orders of magnitude.

    The token index is derived rather than passed: with a right-offset decimation of total stride
    $s$, token $t$'s causal endpoint is $s(t+1) - 1$, and $\lfloor (s(t+1)-1)/s \rfloor = t$
    exactly. So a caller states the raw cut it cares about and cannot pair it with the wrong token.

    Args:
        forward: Maps a time-last raw tensor to a token-major output, $(B, T, C)$.
        x: The unperturbed raw input, $(B, L)$ or $(B, C, L)$.
        cut: Last raw index the asserted token may depend on. Must leave at least one sample after
            it, and must fall inside the token range the output actually covers.
        stride: Total raw samples per output token.
        label: Named in the failure message, so a parametrised run says which case failed.
        seed: Seed for the resample.
    """
    assert stride >= 1, f"stride must be at least 1, got {stride}"
    assert cut < x.shape[-1] - 1, (
        f"cut {cut} leaves no future to perturb in {x.shape[-1]} raw samples"
    )
    token = cut // stride

    reference = forward(x)
    perturbed = forward(resample_raw_after(x, cut, seed=seed))

    assert token < reference.shape[1], (
        f"cut {cut} at stride {stride} names token {token}, beyond the "
        f"{reference.shape[1]} the output has"
    )
    assert torch.equal(reference[:, token], perturbed[:, token]), (
        f"{label}: token {token} changed when only raw samples after {cut} were perturbed, so "
        f"it reads its own future"
    )
    at_end = relative_change(reference[:, -1], perturbed[:, -1])
    assert at_end > MOVEMENT_TOL, (
        f"{label}: the last token moved by only {at_end:.3e} -- the perturbation never reached "
        f"the module, so the bit-stability above proves nothing"
    )


def build_frontend(kwargs: Dict[str, Any], **overrides):
    """Build one stream's front end from a constructor keyword set, the way the model will.

    The four front-end settings are derived from the model's own keys in exactly one way, and it is
    the same map in every front-end test file, so it is written here once. In particular the reach
    budget is ``warmup_period * raw_per_step`` -- not a caller's choice and not a configuration key,
    but a fact about the geometry -- and a test that invented its own would be measuring the guard
    against itself.

    Args:
        kwargs: A constructor keyword set, :data:`TINY_KWARGS` or :data:`SHIPPED_KWARGS`. The
            production kernels are the front end's own default, so a set without
            ``frontend_kernels`` gets them.
        **overrides: Passed through to the front end, overriding anything derived above.

    Returns:
        A ``CausalRawFrontend`` in eval mode -- every structural probe here compares two forward
        passes, and dropout would make that a coin flip.
    """
    from teb_vae.lag_attn_transformer_e2e.nets.frontend import CausalRawFrontend

    raw_per_step = int(kwargs["raw_per_step"])
    settings: Dict[str, Any] = dict(
        d_model=int(kwargs["d_model"]),
        raw_per_step=raw_per_step,
        reach_budget=int(kwargs["warmup_period"]) * raw_per_step,
    )
    if "frontend_kernels" in kwargs:
        settings["kernels"] = tuple(kwargs["frontend_kernels"])
    settings.update(overrides)
    return CausalRawFrontend(**settings).eval()


# ---------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------
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
# tests that are not about the schedule. ``free_bits`` is genuinely 0.0 in the shipped config.
TASK_HPARAMS: Dict[str, Any] = dict(
    lambda_full=1.0,
    lambda_base=1.0,
    likelihood="gaussian_nll",
    free_bits=0.0,
    kld_beta=1.0,
    beta_schedule=None,
)


@pytest.fixture
def raw_inputs():
    """Seeded ``(fhr, up, weight)`` tensors matching the tiny geometry.

    What this model's forward takes, in place of the siblings' three feature blocks: two raw
    signals at ``16 * SEQ_LEN`` and the decimated validity weight, with the same planted gap the
    stub batch carries -- so a test that reaches for tensors rather than a batch still exercises
    the masked path rather than a uniformly valid one.
    """
    batch = make_stub_batch(BATCH, SEQ_LEN)
    return batch.fhr, batch.up, batch.weight


def _make_task(model_kwargs: Optional[dict] = None, hparams: Optional[dict] = None, **task_kwargs):
    """Build a model wrapped in its task, with the production loss hparams applied.

    Imported lazily so the pure-net tests never pay for Lightning.

    Args:
        model_kwargs: Net constructor kwargs. Defaults to :data:`TINY_KWARGS`.
        hparams: Loss hparam overrides on top of :data:`TASK_HPARAMS`.
        **task_kwargs: Passed through to the task's constructor.

    Returns:
        A task with ``setup()`` already called, so the permutation generator exists exactly as it
        would under a real fit.
    """
    from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
    from teb_vae.lag_attn_transformer_e2e.task import SeqVaeLagAttnTrfE2ETask

    kwargs = dict(TINY_KWARGS if model_kwargs is None else model_kwargs)
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfE2E(**kwargs)
    task = SeqVaeLagAttnTrfE2ETask(
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
    """Factory fixture: ``task(model_kwargs=None, hparams=None, **task_kwargs)``.

    The net and the task it imports do not exist yet, so a test that asks for this fixture and
    calls it fails at the import rather than passing on nothing.
    """
    return _make_task
