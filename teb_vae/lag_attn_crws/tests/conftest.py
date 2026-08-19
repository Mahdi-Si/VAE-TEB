r"""Shared pytest configuration for the causal-input raw-target lag-attention VAE tests.

Puts the repository root on ``sys.path`` so the absolute ``teb_vae.*`` and ``hdf5_dataset.*`` imports
resolve no matter which directory pytest is invoked from, and exposes the fixtures the suite is built
on. Follows the ``teb_vae/lag_attn_transformer_cfs/tests/conftest.py`` precedent, including its
``utils`` pre-import pin.

**This conftest is spliced from two places, and which half comes from where is not
interchangeable.**

*The causal-input half is imported from the causal-feature cell's suite.* The committed causal shard
and the two-sided one beside it, the configuration builder every refusal test starts from, the tiny
warm-up staircase and its resolved keep-indices, and the shipped channel widths all describe the
*dataset*, which is not a property of what a model forecasts from it. A second copy of any of them
would be free to describe a boundary the data no longer has -- exactly what reading the shards rather
than declaring the vectors exists to prevent. :data:`IMPORTED_FROM_CAUSAL` names every one of them,
and ``test_fixtures.py`` asserts the list is a subset of that suite's namespace, so a name that
stopped being exported fails here rather than tempting an edit to that package.

*The constructor keyword sets are local.* They are the conv-LSTM schema at the causal channel widths
-- the same architecture the causal-feature cell builds and the same geometry it builds it at -- and
they are still written here rather than imported, because the geometry is a *choice* this package
makes rather than a fact it inherits. Nothing about a raw target forces $H = 15$, $F = 133$ or
$S = 15$: a raw sample is honest at every step, so no validity constraint ties the floor to the
budget at all. Holding all three at the causal-feature cell's values is what leaves exactly one
variable between the two cells -- what the decoder emits -- and a local set is what lets an arm
move one of them without touching the sibling.

Where a value *is* shared it is referenced rather than restated, so the two cannot come to disagree:
the widths, the sequence length, the horizon and the floor below are the imported constants, not
copies of them. The imported stub batch and stream builders close over the same objects.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pytest

# teb_vae/lag_attn_crws/tests/conftest.py -> parents[0]=tests, [1]=lag_attn_crws, [2]=teb_vae,
# [3]=repo root.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

# There are two ``utils`` packages in this repo: the real one at <repo root>/utils and a near-empty
# one at model/vae_teb_prediction/utils. On a repository-wide run another conftest can put the
# latter's parent first on ``sys.path``, shadowing the real one. Binding the repo-root package now
# -- while ``_REPO_ROOT`` is still first -- pins its ``__path__`` for every later ``utils.<submodule>``
# import.
try:
    importlib.import_module("utils")
except Exception:
    pass

#: Every name this conftest takes from the causal-feature cell's suite, as a literal rather than as
#: whatever the import statement below happens to say. ``test_fixtures.py`` asserts it is a subset of
#: that suite's namespace: a name that stopped being exported there then fails against a stated
#: intention, rather than surfacing as an ``ImportError`` in whichever test happened to collect
#: first -- and rather than tempting a re-export to be added to that package, which this one may not
#: edit.

#: This package's shipped forecast horizon, in decimated steps, **owned here** rather than imported
#: from the causal-feature suite.
#:
#: The two packages agreed on 15 for as long as both shipped it, and the import was the smaller
#: line. It was also wrong: a horizon is a property of a TARGET DOMAIN, and this one forecasts a raw
#: FHR window while that one forecasts stored coefficients whose warm-up bounds the anchor floor.
#: When `lag_attn_cfs` moved to 30 first, the shared constant silently re-pointed this suite's whole
#: shipped geometry at a horizon this package did not yet ship, and the failure surfaced as an
#: unrelated shape mismatch three files away. Both ship 30 now, and agreement between two configs is
#: not a reason to have one constant.
SHIPPED_HORIZON = 30

IMPORTED_FROM_CAUSAL = (
    "BATCH",
    "CAUSAL_C_U",
    "CAUSAL_C_Y",
    "CAUSAL_PH_WIDTH",
    "CAUSAL_SHARD",
    "CAUSAL_ST_WIDTH",
    "SHIPPED_BUDGET_STEPS",
    "SHIPPED_SEQUENCE_LENGTH",
    "SHIPPED_TRIM_MINUTES",
    "SHIPPED_WARMUP_PERIOD",
    "TASK_HPARAMS",
    "TINY_HORIZON",
    "TINY_SEQ_LEN",
    "TINY_SOURCE_KEEP_INDEX",
    "TINY_SOURCE_WARMUP_STEPS",
    "TINY_STRIDE",
    "TINY_TARGET_KEEP_INDEX",
    "TINY_TARGET_WARMUP_STEPS",
    "TINY_WARMUP_PERIOD",
    "TWO_SIDED_SHARD",
    "absolutize_dataset_paths",
    "causal_config",
    "hand_seeding_offenders",
    "make_stub_batch",
    "make_streams",
    "perturb_posterior",
    "stored_warmup",
)

# Importing a fixture binds it in this conftest's namespace, which is all pytest needs to serve it to
# the tests here. ``perturb_posterior`` originates in ``lag_attn`` and is taken through the causal
# suite because that is where the rest of this half comes from -- and because the posterior delta
# heads are zero-initialised in every model of the family, so at initialisation every KL assertion in
# this suite would pass vacuously without it.
from teb_vae.lag_attn_cfs.tests.conftest import (  # noqa: E402,F401
    BATCH,
    CAUSAL_C_U,
    CAUSAL_C_Y,
    CAUSAL_PH_WIDTH,
    CAUSAL_SHARD,
    CAUSAL_ST_WIDTH,
    SHIPPED_BUDGET_STEPS,
    SHIPPED_SEQUENCE_LENGTH,
    SHIPPED_TRIM_MINUTES,
    SHIPPED_WARMUP_PERIOD,
    TASK_HPARAMS,
    TINY_HORIZON,
    TINY_SEQ_LEN,
    TINY_SOURCE_KEEP_INDEX,
    TINY_SOURCE_WARMUP_STEPS,
    TINY_STRIDE,
    TINY_TARGET_KEEP_INDEX,
    TINY_TARGET_WARMUP_STEPS,
    TINY_WARMUP_PERIOD,
    TWO_SIDED_SHARD,
    absolutize_dataset_paths,
    causal_config,
    hand_seeding_offenders,
    make_stub_batch,
    make_streams,
    perturb_posterior,
    stored_warmup,
)

# Bound by reference rather than copied, which is what keeps "no edit to any existing package" true
# while leaving no second definition to drift: the threshold names no constructor argument, so what
# the network takes is the four concrete channel tuples it resolves to, and there is exactly one
# translation of the one into the other in the repository. ``test_warmup_budget.py`` asserts both are
# the sibling's own objects.
from teb_vae.lag_attn_cfs.model_kwargs import (  # noqa: E402,F401
    WARMUP_MODEL_KWARGS,
    warmup_model_kwargs,
)

# =================================================================================================
# The constructor keyword sets
# =================================================================================================
#: The tiny constructor keyword set, **ungated**: no budget, no gate, no warm-up mask. The guarded
#: variant is built from it by :func:`tiny_warmup_kwargs`, so the two differ in the guard alone.
#:
#: The window is $24$ decimated steps rather than the two-sided cells' $16$ for the causal cell's
#: reason, which transfers unchanged: a tiling needs a floor, a stride and room for more than one
#: tile, and at $T = 16$ with any usable floor there is one anchor per phase -- so every padding
#: assertion would pass without a padded slot ever existing.
TINY_KWARGS: Dict[str, Any] = dict(
    sequence_length=TINY_SEQ_LEN,
    d_model=32,
    d_z=8,
    horizon=TINY_HORIZON,
    # The decoder's own width here, unlike in the feature-target cells where it is the target gate's
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
    lstm_layers=1,
    horizon_film=True,
    dropout=0.0,
)

#: What the production configuration builds, **ungated**: the causal window and channel widths --
#: $300$ steps, $c_y = 102$, $c_u = 51$ -- with a one-minute horizon tiled at $S = H$ from a floor of
#: $133$, over the conv-LSTM encoders. Construction-time invariants are checked against the model
#: that actually trains, not a miniature of it; forward passes stay on :data:`TINY_KWARGS` for speed.
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
    lstm_layers=2,
    dropout=0.1,
    decoder_hidden=256,
    horizon_depth=4,
    horizon_kernel=3,
    horizon_film=True,
    horizon_attention_blocks=2,
    horizon_embed_std=0.8,
    head_init_calibration=True,
    a_head_gain=2.0,
    encoder_extra_dilations=(8, 16),
    encoder_extra_kernel=15,
    logvar_clamp=(-5.0, 3.0),
    mu_scale=5.0,
    delta_mu_scale=3.0,
    delta_logvar_scale=2.0,
    use_entmax=True,
    lag_bias_init="alibi_decay",
    causal_norm=True,
    coverage_floor=0.9,
    # Stated rather than defaulted: consecutive forecast windows partition the timeline instead of
    # overlapping, so no raw sample is scored twice in a step.
    anchor_stride=SHIPPED_HORIZON,
)


def tiny_warmup_kwargs(
    kwargs: Optional[Mapping[str, Any]] = None, **overrides: Any
) -> Dict[str, Any]:
    """Add the tiny warm-up guard to a constructor keyword set.

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


def shipped_warmup_kwargs(
    model_cls: Optional[type] = None, **overrides: Any
) -> Dict[str, Any]:
    """The production constructor call: :data:`SHIPPED_KWARGS` plus the budget the shards resolve.

    Resolved through :func:`~teb_vae.lag_attn_cfs.causal_warmup.resolve_warmup_budget` against the
    committed causal fixture rather than written out, so the surviving-channel count comes from the
    data every time and a rebuilt fixture moves the model instead of failing an unrelated literal.

    Args:
        model_cls: The class these kwargs will construct, which is what
            :func:`~teb_vae.lag_attn_cfs.model_kwargs.warmup_model_kwargs` refuses on: an
            architecture that cannot mask its inputs drops the two warm-up vectors at the signature
            sweep and trains to completion having read the assumed pre-recording history as signal.
            ``None`` names this package's own model, which is what almost every caller wants; a test
            *about* that refusal names its own class instead, and so is checking the refusal rather
            than whichever class a default had chosen for it.
        **overrides: Applied last.

    Returns:
        Constructor kwargs. No ``decoder_out_channels``: the raw block's width follows
        ``raw_per_step``, which the architecture resolves for itself.
    """
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget
    from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws

    resolved = resolve_warmup_budget(causal_config())
    assert resolved is not None
    kwargs = dict(
        SHIPPED_KWARGS,
        **warmup_model_kwargs(resolved, SeqVaeLagAttnCrws if model_cls is None else model_cls),
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
    import torch

    from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws

    torch.manual_seed(seed)
    return SeqVaeLagAttnCrws(**dict(kwargs))


def make_raw_signal(kwargs: Mapping[str, Any], batch: int = BATCH, seed: int = 0):
    """A seeded raw target signal $(B, L_{\\mathrm{raw}})$ for a keyword set.

    Local rather than imported: the causal-feature suite never needs one, because its target is a
    stored feature block the model is already shown. The length is $T \\cdot R$ from the same two
    keywords the model builds its geometry from, so a keyword set that moved either produces a
    signal the geometry still accepts.

    Args:
        kwargs: A constructor keyword set, read for its sequence length and raw decimation.
        batch: Samples in the batch.
        seed: Seed, so a signal is reproducible.

    Returns:
        The raw signal tensor.
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    length = int(kwargs["sequence_length"]) * int(kwargs["raw_per_step"])
    return torch.randn(batch, length, generator=generator)


def make_task(
    model_kwargs: Optional[Mapping[str, Any]] = None,
    hparams: Optional[Mapping[str, Any]] = None,
    **task_kwargs: Any,
):
    """Build this model wrapped in its own task, with the production loss hparams applied.

    Args:
        model_kwargs: Net constructor kwargs. Defaults to the guarded tiny set at the tiling
            stride, because the guard and the tiling are what this package is about: a task built
            at the constructor's inert stride of $1$ would resolve the same anchor set on every
            stage, and every phase assertion made against it would hold vacuously.
        hparams: Loss hparam overrides on top of :data:`TASK_HPARAMS`.
        **task_kwargs: Passed through to the task's constructor -- ``seed`` is the one that matters.

    Returns:
        A ``SeqVaeLagAttnCrwsTask`` with ``setup()`` already called, so the permutation generator
        exists exactly as it would under a real fit.
    """
    from teb_vae.lag_attn_crws.task import SeqVaeLagAttnCrwsTask

    kwargs = dict(
        tiny_warmup_kwargs(anchor_stride=TINY_STRIDE) if model_kwargs is None else model_kwargs
    )
    task = SeqVaeLagAttnCrwsTask(
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
    """The resolved warm-up budget at the shipped threshold, against the committed fixture."""
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget

    resolved = resolve_warmup_budget(causal_config())
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
