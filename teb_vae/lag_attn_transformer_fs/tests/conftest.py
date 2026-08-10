r"""Shared pytest configuration for the conv-Transformer feature-domain lag-attention VAE tests.

Puts the repository root on ``sys.path`` so the absolute ``teb_vae.*`` imports resolve no matter
which directory pytest is invoked from, and exposes the fixtures the suite is built on. Mirrors
``teb_vae/lag_attn_transformer_rws/tests/conftest.py``, including its ``utils`` pre-import pin.

**This is the only conftest in the family spliced from two siblings, and which half comes from
which is not interchangeable.**

*The constructor keyword sets come from the conv-Transformer sibling.* That model has no
``lstm_layers``, ``encoder_extra_dilations``, ``encoder_extra_kernel``, ``conv_norm_groups`` or
``causal_norm``, and it has seven encoder keys the feature sibling has never heard of. Taking the
feature suite's sets instead is the specific mistake worth naming, because it fails *asymmetrically*:
its ``TINY_KWARGS`` carries none of the five absent keys, so the tiny path would pass and only the
shipped path would fail, and the failure would name a keyword rather than the conftest.

*The batch machinery comes from the feature sibling.* ``make_patterned_batch`` plants a known
per-$(t, c)$ value in the two stored target blocks, and every target assertion in this suite is made
against it. The conv-Transformer sibling's stub batch draws them from ``randn``, which is fine for a
model that reads features and forecasts raw signal and useless for one whose target *is* those
features: against random data a transposed gather, an off-by-one anchor and a
delayed-instead-of-gathered target all produce correctly shaped tensors and pass every shape check
there is.

The two halves meet at five geometry keys the feature batch machinery closes over -- ``c_y``,
``c_u``, ``warmup_period``, ``raw_per_step`` and ``use_up_st``. ``resolve_target_budget`` reads them
off the *feature* suite's shipped set while the models here are built from the transformer's, so the
splice is only sound while the two agree. ``test_fixtures.py`` asserts that they do, which is what
turns the paragraph above into a measurement.

What is local is the pairing of the two: the gated keyword sets. The feature suite's
``shipped_gated_kwargs`` builds on its own shipped set and would carry the five absent keys, so the
budget's four resolved tuples are joined to the transformer's sets here instead.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import torch

# teb_vae/lag_attn_transformer_fs/tests/conftest.py -> parents[0]=tests,
# [1]=lag_attn_transformer_fs, [2]=teb_vae, [3]=repo root.
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

# ---------------------------------------------------------------------------------------
# The architecture half: keyword sets, tolerances and the causality probe.
#
# Every name here describes the conv-Transformer model. Importing a fixture binds it in this
# conftest's namespace, which is all pytest needs to serve it to the tests in this directory.
# ---------------------------------------------------------------------------------------
from teb_vae.lag_attn_transformer_rws.tests.conftest import (  # noqa: E402,F401
    BATCH,
    CAUSALITY_TOL,
    MOVEMENT_TOL,
    SEQ_LEN,
    SHIPPED_KWARGS,
    STUB_GAP_STEP,
    TASK_HPARAMS,
    TINY_KWARGS,
    absolutize_dataset_paths,
    assert_token_causal,
    inputs,
    make_stub_batch,
    relative_change,
    resample_after,
)

# ---------------------------------------------------------------------------------------
# The target half: the planted pattern, the small guard's tuples, the budget resolver.
#
# ``perturb_posterior`` originates in ``lag_attn``; it is taken through the feature suite because
# that is where the rest of this half comes from, and because the delta heads are zero-initialised
# in every model of the family -- so at initialisation every KL assertion in this suite would pass
# vacuously without it.
# ---------------------------------------------------------------------------------------
from teb_vae.lag_attn_fs.tests.conftest import (  # noqa: E402,F401
    PATTERN_STEP_SCALE,
    SHIPPED_REACH_BUDGET_S,
    TINY_DELAYS,
    TINY_KEEP_INDEX,
    TINY_SOURCE_DELAYS,
    TINY_SOURCE_KEEP_INDEX,
    build_target_gate,
    make_patterned_batch,
    patterned_feature_stream,
    perturb_posterior,
    resolve_target_budget,
)

#: The five geometry keys the imported feature-side batch machinery and budget resolver close over.
#: They are read off the feature suite's shipped set while every model here is built from the
#: conv-Transformer one, so the splice is sound only while the two agree -- asserted in
#: ``test_fixtures.py`` rather than assumed here.
SHARED_GEOMETRY_KEYS = ("c_y", "c_u", "warmup_period", "raw_per_step", "use_up_st")


def tiny_gated_kwargs() -> Dict[str, Any]:
    """The tiny constructor kwargs with a small causal guard on both streams.

    The shipped budget cannot be resolved at the tiny geometry -- its maximum delay is $30$ steps
    against a warm-up of $2$, which the resolver refuses -- so the gated path is exercised here at a
    hand-made guard. Its delays are non-zero and distinct, which is what makes the
    never-delayed-target assertions specific: a target built through the gate would be wrong by a
    different number of steps in each channel.

    Returns:
        Constructor kwargs. No ``decoder_out_channels``: it is not a keyword of this constructor.
    """
    return dict(
        TINY_KWARGS,
        target_keep_index=TINY_KEEP_INDEX,
        target_delays=TINY_DELAYS,
        source_keep_index=TINY_SOURCE_KEEP_INDEX,
        source_delays=TINY_SOURCE_DELAYS,
    )


def shipped_gated_kwargs(
    budget_s: Optional[float] = SHIPPED_REACH_BUDGET_S,
) -> Dict[str, Any]:
    """The production constructor call: this architecture's widths plus the resolved reach budget.

    What the trainer assembles, minus the trainer -- ``_build_model_kwargs`` forwards the config
    block and then updates it with exactly these four tuples. Built here rather than written out so
    the surviving-channel counts come from the filter bank every time, and built on the
    conv-Transformer sibling's shipped set rather than on the feature sibling's, which carries five
    keywords this constructor refuses.

    Args:
        budget_s: The reach budget in seconds, or ``None`` for the unguarded arm, which adds no
            tuples at all and leaves the model with no gate.

    Returns:
        Constructor kwargs. No ``decoder_out_channels``: the width is the model's to resolve.
    """
    budget = resolve_target_budget(budget_s)
    if budget is None:
        return dict(SHIPPED_KWARGS)
    return dict(
        SHIPPED_KWARGS,
        target_keep_index=budget.target_keep_index,
        target_delays=budget.target_delays,
        source_keep_index=budget.source_keep_index,
        source_delays=budget.source_delays,
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``slow`` marker; there is no repo-wide pytest configuration to declare it."""
    config.addinivalue_line(
        "markers", "slow: long-running empirical validation, excluded from the default run"
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
def tiny_gated() -> dict:
    """A fresh copy of the tiny kwargs with the small causal guard (safe to mutate)."""
    return tiny_gated_kwargs()


@pytest.fixture
def shipped_gated() -> dict:
    """A fresh copy of the production kwargs with the shipped reach budget (safe to mutate)."""
    return shipped_gated_kwargs()


@pytest.fixture
def stub_batch():
    """A two-sample stub batch at the tiny geometry, with the deliberate weight gap."""
    return make_stub_batch(BATCH, SEQ_LEN)


@pytest.fixture
def patterned_batch():
    """A two-sample batch at the tiny geometry whose target blocks carry the known pattern."""
    return make_patterned_batch(BATCH, SEQ_LEN)


def _make_task(model_kwargs: Optional[dict] = None, hparams: Optional[dict] = None, **task_kwargs):
    """Build this model wrapped in this package's task, with the production loss hparams applied.

    This package's own task rather than the shared one: what the diamond adds is the feature
    target's batch-to-target builder and the page-row seam on one side and the step-granular
    learning-rate ramp on the other, and every contract asserted through this factory is a contract
    of the pair a real run assembles. Imported lazily so the pure-net tests never pay for Lightning.

    Args:
        model_kwargs: Net constructor kwargs. Defaults to :data:`TINY_KWARGS`.
        hparams: Loss hparam overrides on top of :data:`TASK_HPARAMS`.
        **task_kwargs: Passed through to the task's constructor.

    Returns:
        A ``SeqVaeLagAttnTrfFsTask`` wrapping a ``SeqVaeLagAttnTrfFs``, with ``setup()`` already
        called so the permutation generator exists exactly as it would under a real fit.
    """
    from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs
    from teb_vae.lag_attn_transformer_fs.task import SeqVaeLagAttnTrfFsTask

    kwargs = dict(TINY_KWARGS if model_kwargs is None else model_kwargs)
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfFs(**kwargs)
    task = SeqVaeLagAttnTrfFsTask(
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
