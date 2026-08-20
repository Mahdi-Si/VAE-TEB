r"""Shared pytest configuration for the causal-input raw-target conv-Transformer VAE tests.

Puts the repository root on ``sys.path`` so the absolute ``teb_vae.*`` and ``hdf5_dataset.*``
imports resolve no matter which directory pytest is invoked from, and exposes the fixtures the suite
is built on. Follows the ``teb_vae/lag_attn_transformer_cfs/tests/conftest.py`` precedent, including
its ``utils`` pre-import pin.

**This conftest is spliced from two siblings, and which half comes from which is not
interchangeable.**

*The constructor keyword sets are written here, at the conv-Transformer keyword schema.* That
architecture has no ``lstm_layers``, ``encoder_extra_dilations``, ``encoder_extra_kernel``,
``conv_norm_groups`` or ``causal_norm``, and it has seven encoder keys the conv-LSTM cell has never
heard of. Taking the conv-LSTM cell's sets instead is the specific mistake worth naming, because it
fails *asymmetrically*: its ``TINY_KWARGS`` carries only one of the five absent keys, so the tiny
path would fail on that one keyword and the shipped path on three more, and every failure would name
a keyword rather than the conftest.

*The target and input half comes from the conv-LSTM cell.* The committed causal shard and the
two-sided one beside it, the configuration builder every refusal test starts from, the tiny warm-up
staircase and its resolved keep-indices, the budget resolver, the stub batch carrying ``guid`` and
``epoch`` -- the two fields the anchor tiling's phase is keyed on -- the seeded input streams at the
one-sided channel widths and the seeded raw target signal are all imported rather than restated.
They describe the *dataset*, the *target domain* and the *anchor geometry*, none of which is a
property of an encoder, and a second copy of any of them would be free to describe a boundary the
data no longer has.

*The causality probe and its two tolerances come from the conv-Transformer sibling.* They describe
the *architecture*: this encoder stack has no time-pooling normaliser to flip, so its control is
positional rather than the conv-LSTM cell's, and a local re-implementation would be a second
definition of what "causal" means for these blocks.

The two halves meet at the geometry keys :data:`SHARED_GEOMETRY_KEYS` names. The imported batch
machinery, the imported budget resolution, the imported raw-signal builder and every anchor
expectation in this suite close over them while the models here are built from the sets below, so
the splice is sound only while the two agree. ``test_fixtures.py`` asserts that they do, which is
what turns the paragraph above into a measurement -- and here it is load-bearing rather than
decorative, because the two conftests are independently maintained.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pytest
import torch

# teb_vae/lag_attn_transformer_crws/tests/conftest.py -> parents[0]=tests,
# [1]=lag_attn_transformer_crws, [2]=teb_vae, [3]=repo root.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

# There are two ``utils`` packages in this repo: the real one at <repo root>/utils and a near-empty
# one at model/vae_teb_prediction/utils. On a repository-wide run another conftest can put the
# latter's parent first on ``sys.path``, shadowing the real one. Binding the repo-root package now
# -- while ``_REPO_ROOT`` is still first -- pins its ``__path__`` for every later
# ``utils.<submodule>`` import.
try:
    importlib.import_module("utils")
except Exception:
    pass

#: Every name this conftest takes from the conv-Transformer raw-signal suite, as a literal rather
#: than as whatever the import statement happens to say. All six describe how a *causality* claim is
#: measured against this encoder stack, which is a property of the architecture and not of what it
#: reads or emits.
IMPORTED_FROM_ARCHITECTURE = (
    "CAUSALITY_TOL",
    "MOVEMENT_TOL",
    "assert_token_causal",
    "build_stream_encoder",
    "relative_change",
    "resample_after",
)

#: Every name this conftest takes from the conv-LSTM causal-input cell's suite. Those in turn are
#: mostly that suite's own imports from the causal-feature cell, so the objects here are the
#: family's single copies rather than a second hop's worth of copies -- which is what makes the
#: identity assertions in ``test_fixtures.py`` meaningful at either end of the chain.
IMPORTED_FROM_CRWS = (
    "BATCH",
    "CAUSAL_C_U",
    "CAUSAL_C_Y",
    "CAUSAL_PH_WIDTH",
    "CAUSAL_SHARD",
    "CAUSAL_ST_WIDTH",
    "SHIPPED_BUDGET_STEPS",
    "SHIPPED_HORIZON",
    "SHIPPED_SEQUENCE_LENGTH",
    "SHIPPED_TRIM_MINUTES",
    "SHIPPED_WARMUP_PERIOD",
    "TASK_HPARAMS",
    "TINY_HORIZON",
    "TINY_SEQ_LEN",
    "TINY_SOURCE_KEEP_INDEX",
    "TINY_SOURCE_WARMUP_STEPS",
    "TINY_STRIDE",
    "TINY_ALIGNED_WARMUP_PERIOD",
    "TINY_SOURCE_ALIGN_DELAYS",
    "TINY_TARGET_ALIGN_DELAYS",
    "TINY_TARGET_KEEP_INDEX",
    "TINY_TARGET_WARMUP_STEPS",
    "TINY_WARMUP_PERIOD",
    "TWO_SIDED_SHARD",
    "WARMUP_MODEL_KWARGS",
    "absolutize_dataset_paths",
    "causal_config",
    "hand_seeding_offenders",
    "make_raw_signal",
    "make_stub_batch",
    "make_streams",
    "perturb_posterior",
    "stored_warmup",
    "warmup_model_kwargs",
)

# ---------------------------------------------------------------------------------------
# The architecture half: the causality probe and its two tolerances.
#
# Every invariant in this suite is measured the same way -- resample the strict future, require
# bit-stability at the cut *and* visible movement at the end -- and the second half is the negative
# control without which a dead layer passes every causality test in the package. Importing a fixture
# binds it in this conftest's namespace, which is all pytest needs to serve it to the tests here.
# ---------------------------------------------------------------------------------------
from teb_vae.lag_attn_transformer_rws.tests.conftest import (  # noqa: E402,F401
    CAUSALITY_TOL,
    MOVEMENT_TOL,
    assert_token_causal,
    build_stream_encoder,
    relative_change,
    resample_after,
)

# ---------------------------------------------------------------------------------------
# The input and target half: the committed causal shard, the config builder, the tiny warm-up
# staircase, the stub batch carrying the two phase-key fields and the seeded raw target signal.
#
# ``perturb_posterior`` originates in ``lag_attn``; it is taken through the conv-LSTM causal-input
# suite because that is where the rest of this half comes from, and because the posterior delta
# heads are zero-initialised in every model of the family -- so at initialisation every KL assertion
# in this suite would pass vacuously without it.
# ---------------------------------------------------------------------------------------
from teb_vae.lag_attn_crws.tests.conftest import (  # noqa: E402,F401
    BATCH,
    CAUSAL_C_U,
    CAUSAL_C_Y,
    CAUSAL_PH_WIDTH,
    CAUSAL_SHARD,
    CAUSAL_ST_WIDTH,
    SHIPPED_BUDGET_STEPS,
    SHIPPED_HORIZON,
    SHIPPED_SEQUENCE_LENGTH,
    SHIPPED_TRIM_MINUTES,
    SHIPPED_WARMUP_PERIOD,
    TASK_HPARAMS,
    TINY_HORIZON,
    TINY_SEQ_LEN,
    TINY_SOURCE_KEEP_INDEX,
    TINY_SOURCE_WARMUP_STEPS,
    TINY_STRIDE,
    TINY_ALIGNED_WARMUP_PERIOD,
    TINY_SOURCE_ALIGN_DELAYS,
    TINY_TARGET_ALIGN_DELAYS,
    TINY_TARGET_KEEP_INDEX,
    TINY_TARGET_WARMUP_STEPS,
    TINY_WARMUP_PERIOD,
    TWO_SIDED_SHARD,
    WARMUP_MODEL_KWARGS,
    absolutize_dataset_paths,
    causal_config,
    hand_seeding_offenders,
    make_raw_signal,
    make_stub_batch,
    make_streams,
    perturb_posterior,
    stored_warmup,
    warmup_model_kwargs,
)

#: The geometry keys the imported input-and-target half closes over. They are read off the conv-LSTM
#: cell's sets -- by its budget resolution, its stub batch, its raw-signal builder and every anchor
#: count this suite asserts -- while every model here is built from the sets below, so the splice is
#: sound only while the two agree. A disagreement builds a model neither parent's suite tests: no
#: shape would differ, because $A_{\max}$ and the raw block width are geometry constants either way,
#: and the numbers would simply be another model's.
SHARED_GEOMETRY_KEYS = (
    "sequence_length",
    "c_y",
    "c_u",
    "horizon",
    "warmup_period",
    "raw_per_step",
    "use_up_st",
    "anchor_stride",
    "max_lag",
)

#: The five conv-LSTM-only constructor keywords. Absent from every keyword set and every config this
#: package builds: there is no recurrent branch, no extra dilation schedule and no time-pooling
#: normaliser left to causalise, so each of them would reach nothing -- and each raises ``TypeError``
#: at the constructor rather than being ignored, which is why the assertion is worth making once
#: here and reading everywhere.
CONV_LSTM_ONLY_KEYS = (
    "lstm_layers",
    "encoder_extra_dilations",
    "encoder_extra_kernel",
    "conv_norm_groups",
    "causal_norm",
)

# Tiny but structurally faithful, and larger than the conv-Transformer raw-signal sibling's
# $16$-step fixture for the tiling's reason: a tiling needs a floor, a stride and room for more than
# one tile, and at $T = 16$ with any usable floor there is only one anchor per phase -- which would
# leave every padding assertion in the suite passing without a padded slot ever existing.
#
# The encoder geometry is shrunk with one property preserved that the shipped one has and a naive
# miniature would lose: the source encoder's receptive-field bound stays strictly inside the
# sequence. Stem reach is $1 + (3-1)\cdot 1 + (3-1)\cdot 2 = 7$ steps, so the source bound is
# $7 + 2(4-1) = 13 < 24$. A bound that clamped at $T$ would make the measured-bound probe vacuous.
#
# ``horizon_film`` is on, matching the shipped config: per-block FiLM is hardcoded in the net, so
# the horizon core is built with ``film=horizon_film`` and ``film_per_block=True``, and
# ``horizon_film=false`` would fail fast at construction.
TINY_KWARGS: Dict[str, Any] = dict(
    sequence_length=TINY_SEQ_LEN,
    d_model=32,
    d_z=8,
    horizon=TINY_HORIZON,
    # The decoder's own width, unlike in the feature-target cells where it is the target gate's
    # surviving-channel count: a horizon token emits $R$ raw samples, so the raw block is
    # $H \cdot R$ and no configuration can put the decoder and the target on different widths.
    raw_per_step=16,
    warmup_period=TINY_WARMUP_PERIOD,
    c_y=CAUSAL_C_Y,
    c_u=CAUSAL_C_U,
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

# What configs/default.yaml sets, at full production geometry: the causal window and channel widths
# -- $300$ steps, $c_y = 102$, $c_u = 51$, a one-minute horizon tiled at $S = H$ from a floor of
# $133$ -- over the conv-Transformer encoders. Construction-time invariants are checked against the
# model that actually trains, not a miniature of it; forward passes in this suite stay on
# ``TINY_KWARGS`` for speed.
#
# Written out rather than derived from the conv-LSTM cell's set: the two are the two halves of the
# splice, and a set derived from the other could not disagree with it, which would make
# ``test_fixtures.py``'s agreement check a tautology.
SHIPPED_KWARGS: Dict[str, Any] = dict(
    sequence_length=SHIPPED_SEQUENCE_LENGTH,
    d_model=128,
    d_z=64,
    horizon=SHIPPED_HORIZON,
    raw_per_step=16,
    warmup_period=SHIPPED_WARMUP_PERIOD,
    c_y=CAUSAL_C_Y,
    c_u=CAUSAL_C_U,
    use_up_st=True,
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
    # Stated rather than defaulted: consecutive forecast windows partition the timeline instead of
    # overlapping, so no raw sample is scored twice in a step.
    anchor_stride=SHIPPED_HORIZON,
    encoder_conv_kernels=(5, 9),
    encoder_conv_dilations=(1, 2),
    encoder_num_heads=4,
    encoder_d_ff=512,
    target_attention_blocks=6,
    source_attention_blocks=3,
    source_attention_window=16,
)


def tiny_warmup_kwargs(
    kwargs: Optional[Mapping[str, Any]] = None, **overrides: Any
) -> Dict[str, Any]:
    """Add the tiny warm-up guard to a constructor keyword set.

    The conv-LSTM cell's identical helper is deliberately not imported: it defaults to *its own*
    ``TINY_KWARGS``, so a call here that omitted the first argument would silently build the wrong
    architecture's keyword set and fail at ``lstm_layers`` rather than saying so.

    Args:
        kwargs: The keyword set to guard. Defaults to :data:`TINY_KWARGS`.
        **overrides: Applied after the guard, so a test can move one leaf of it.

    Returns:
        A fresh dict carrying the guard on both streams.
    """
    guarded = dict(
        TINY_KWARGS if kwargs is None else kwargs,
        target_keep_index=TINY_TARGET_KEEP_INDEX,
        target_warmup_steps=TINY_TARGET_WARMUP_STEPS,
        source_keep_index=TINY_SOURCE_KEEP_INDEX,
        source_warmup_steps=TINY_SOURCE_WARMUP_STEPS,
    )
    guarded.update(overrides)
    return guarded


def tiny_align_kwargs(
    kwargs: Optional[Mapping[str, Any]] = None, **overrides: Any
) -> Dict[str, Any]:
    """Add the tiny warm-up guard **and** the tiny alignment to a constructor keyword set.

    The shift vectors are the causal-feature suite's own, bound above rather than rebuilt: every
    cell of this grid gates the same two stored streams at the same tiny widths, and a second
    definition of the same vectors is one that can drift. The anchor floor moves with the shift
    because the two are one decision -- a floor left at the unaligned value is refused by the
    constructor, which is itself asserted rather than worked around.

    Args:
        kwargs: The keyword set to guard. Defaults to :data:`TINY_KWARGS`.
        **overrides: Applied after the alignment, so a test can move one leaf of it.

    Returns:
        A fresh dict carrying the guard and the shift on both streams.
    """
    aligned = tiny_warmup_kwargs(
        kwargs,
        target_align_delays=TINY_TARGET_ALIGN_DELAYS,
        source_align_delays=TINY_SOURCE_ALIGN_DELAYS,
        warmup_period=TINY_ALIGNED_WARMUP_PERIOD,
    )
    aligned.update(overrides)
    return aligned


def shipped_warmup_kwargs(
    model_cls: Optional[type] = None, *, align: bool = True, **overrides: Any
) -> Dict[str, Any]:
    """The production constructor call: :data:`SHIPPED_KWARGS` plus the budget the shards resolve.

    Resolved through :func:`~teb_vae.lag_attn_cfs.causal_warmup.resolve_warmup_budget` against the
    committed causal fixture rather than written out, so the surviving-channel count comes from the
    data every time and a rebuilt fixture moves the model instead of failing an unrelated literal.

    The ``align`` flag names the **unaligned comparison arm**: the same geometry resolved with
    ``causal_align_reference`` removed, which is the model that shipped before the common clock and
    is what every "what did the alignment cost" assertion is stated against. It is a flag rather
    than a second builder because the two must not be allowed to drift in anything else.

    Args:
        model_cls: The class these kwargs will construct, which is what
            :func:`~teb_vae.lag_attn_cfs.model_kwargs.warmup_model_kwargs` refuses on: an
            architecture that cannot mask its inputs drops the two warm-up vectors at the signature
            sweep and trains to completion having read the assumed pre-recording history as signal.
            ``None`` names this package's own model, which is what almost every caller wants.
        align: ``False`` resolves the budget with no alignment reference, so the two shift
            vectors are absent and every source channel survives.
        **overrides: Applied last.

    Returns:
        Constructor kwargs. No ``decoder_out_channels``: it is not a keyword of this constructor,
        and the raw block's width follows ``raw_per_step`` regardless.
    """
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget
    from teb_vae.lag_attn_transformer_crws.nets.model import SeqVaeLagAttnTrfCrws

    resolved = resolve_warmup_budget(
        causal_config() if align else causal_config(causal_align_reference=None)
    )
    assert resolved is not None
    kwargs = dict(
        SHIPPED_KWARGS,
        **warmup_model_kwargs(
            resolved, SeqVaeLagAttnTrfCrws if model_cls is None else model_cls
        ),
    )
    # Updated rather than splatted into the call above: an override naming a key the set already
    # carries -- ``horizon`` is the one every arm moves -- is a TypeError there and a replacement
    # here, which is what "applied last" has to mean to be usable at all.
    kwargs.update(overrides)
    return kwargs


def build(kwargs: Mapping[str, Any], seed: int = 0):
    """Construct this package's model under a fixed seed, so two builds are the same weights.

    Args:
        kwargs: Constructor keyword set.
        seed: Global seed applied immediately before construction.

    Returns:
        The model, in whatever mode ``nn.Module`` defaults to.
    """
    from teb_vae.lag_attn_transformer_crws.nets.model import SeqVaeLagAttnTrfCrws

    torch.manual_seed(seed)
    return SeqVaeLagAttnTrfCrws(**dict(kwargs))


def make_task(
    model_kwargs: Optional[Mapping[str, Any]] = None,
    hparams: Optional[Mapping[str, Any]] = None,
    **task_kwargs: Any,
):
    """Build this model wrapped in this package's task, with the production loss hparams applied.

    This package's own task rather than either parent's: what the diamond adds is the tiling phase,
    the five-argument forward and the source-null readout on one side and the step-granular
    learning-rate ramp on the other, and every contract asserted through this factory is a contract
    of the pair a real run assembles. Imported lazily so the pure-net tests never pay for Lightning.

    Args:
        model_kwargs: Net constructor kwargs. Defaults to the guarded tiny set at the tiling
            stride, because the guard and the tiling are what this package is about: a task built
            at the constructor's inert stride of $1$ would resolve the same anchor set on every
            stage, and every phase assertion made against it would hold vacuously.
        hparams: Loss hparam overrides on top of :data:`TASK_HPARAMS`.
        **task_kwargs: Passed through to the task's constructor -- ``seed`` is the one that matters.

    Returns:
        A ``SeqVaeLagAttnTrfCrwsTask`` with ``setup()`` already called, so the permutation generator
        exists exactly as it would under a real fit.
    """
    from teb_vae.lag_attn_transformer_crws.task import SeqVaeLagAttnTrfCrwsTask

    kwargs = dict(
        tiny_warmup_kwargs(anchor_stride=TINY_STRIDE) if model_kwargs is None else model_kwargs
    )
    task = SeqVaeLagAttnTrfCrwsTask(
        build(kwargs),
        lr=1e-3,
        model_kwargs=kwargs,
        **dict(TASK_HPARAMS, **(hparams or {})),
        **task_kwargs,
    )
    task.setup("fit")  # seeds the permutation generator; Lightning would call this itself
    return task


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``slow`` marker; there is no repo-wide pytest configuration to declare it."""
    config.addinivalue_line(
        "markers", "slow: long-running empirical validation, excluded from the default run"
    )


@pytest.fixture
def config() -> Dict[str, Any]:
    """A fresh configuration at the shipped causal geometry (safe to mutate)."""
    return causal_config()


@pytest.fixture
def budget():
    """The resolved warm-up budget at the shipped threshold, against the committed fixture.

    Shipped means **aligned**: both streams carry a shift, the source is 47 channels wide, and
    ``reference_delay_s`` is set. :func:`unaligned_budget` is the comparison arm.
    """
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget

    resolved = resolve_warmup_budget(causal_config())
    assert resolved is not None
    return resolved


@pytest.fixture
def unaligned_budget():
    """The same budget with the alignment off: no shift, no reference, every source channel kept.

    The comparison arm that stays reachable at one key, and what every assertion phrased as "the
    warm-up budget alone decides this" is stated against.
    """
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget

    resolved = resolve_warmup_budget(causal_config(causal_align_reference=None))
    assert resolved is not None
    return resolved


@pytest.fixture
def tiny_kwargs() -> Dict[str, Any]:
    """A fresh copy of the ungated tiny constructor kwargs (safe to mutate)."""
    return dict(TINY_KWARGS)


@pytest.fixture
def shipped_kwargs() -> Dict[str, Any]:
    """A fresh copy of the ungated production constructor kwargs (safe to mutate)."""
    return dict(SHIPPED_KWARGS)


@pytest.fixture
def tiny_warmup() -> Dict[str, Any]:
    """A fresh copy of the tiny kwargs carrying the tiny warm-up guard (safe to mutate)."""
    return tiny_warmup_kwargs()


@pytest.fixture
def tiny_align() -> Dict[str, Any]:
    """A fresh copy of the tiny kwargs carrying the guard and the alignment (safe to mutate)."""
    return tiny_align_kwargs()


@pytest.fixture
def streams():
    """Seeded ``(y_st, y_ph, u_stream)`` at the tiny geometry and the causal widths."""
    return make_streams(TINY_KWARGS)


@pytest.fixture
def raw_signal():
    """A seeded raw target signal at the tiny geometry."""
    return make_raw_signal(TINY_KWARGS)


@pytest.fixture
def stub_batch():
    """A two-sample stub batch at the tiny geometry, with the deliberate weight gap."""
    return make_stub_batch()


@pytest.fixture
def task():
    """Factory fixture: ``task(model_kwargs=None, hparams=None, **task_kwargs)``."""
    return make_task
