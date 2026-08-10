r"""Shared pytest configuration for the feature-domain lag-attention VAE tests.

Puts the repository root on ``sys.path`` so the absolute ``teb_vae.*`` imports resolve no matter
which directory pytest is invoked from, and exposes the fixtures the suite is built on. Mirrors
``teb_vae/lag_attn_rws/tests/conftest.py``, including its ``utils`` pre-import pin.

**Almost everything is imported rather than restated.** The constructor keyword sets, the stub
batch, the posterior perturbation and the dataset-path helper all come from the sibling suites.
That is a stronger statement here than it was for the conv-Transformer sibling, which had to
define its own keyword sets because its net takes seven keys the raw model has never heard of:
this model's net is a *subclass* whose constructor schema is unchanged, so a keyword set of its
own could only ever be a copy that drifts. ``raw_per_step`` therefore stays in the tiny set (it
remains a geometry input; it is simply no longer the decoder width) and no
``decoder_out_channels`` is added, so every construction here exercises the width the model
derives for itself.

Two things in the sibling conftest are deliberately **not** imported. The two session-wide budget
shrinkers (``suite_oracle_budget``, ``suite_page_budget``) are autouse fixtures that import the
evaluation package to rebind its module globals, and this package has no evaluation pipeline for
them to shrink -- importing them would drag ``eval`` into every run of this suite to no effect.
The generated multi-class shard writer goes with them for the same reason: nothing here reads
generated shards, and the one test that touches real data reads the committed one.

What is local is the one thing that is not shared: a batch whose feature blocks carry a **known
per-$(t, c)$ value**. The sibling's stub batch draws them from ``randn``, which is fine for a
model that reads features and forecasts something else, and useless for a model whose target *is*
those features -- against random data a transposed gather, an off-by-one anchor and a
delayed-instead-of-gathered target all produce correctly shaped tensors and pass every shape
check there is.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import torch

# teb_vae/lag_attn_fs/tests/conftest.py -> parents[0]=tests, [1]=lag_attn_fs, [2]=teb_vae,
# [3]=repo root.
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

# Importing a fixture binds it in this conftest's namespace, which is all pytest needs to serve it
# to the tests in this directory. The delta heads are zero-initialised in every model of the
# family, so at initialisation every KL assertion passes vacuously in all of them, and the escape
# from that trap is one fixture rather than three.
from teb_vae.lag_attn.tests.conftest import perturb_posterior  # noqa: E402,F401
from teb_vae.lag_attn_rws.tests.conftest import (  # noqa: E402,F401
    BATCH,
    SEQ_LEN,
    SHIPPED_KWARGS,
    STUB_GAP_STEP,
    TASK_HPARAMS,
    TINY_KWARGS,
    absolutize_dataset_paths,
    inputs,
    make_stub_batch,
)

#: The reach budget the shipped configuration runs at, in seconds. It is the sibling's
#: ``configs/default.yaml`` value and stays unchanged in this package -- the survivor set is what
#: makes input and target live under one causal budget, which is the reason the target is the
#: gated subset rather than all $109$ channels.
SHIPPED_REACH_BUDGET_S = 120.0

#: Multiplier separating the step index from the channel index in the planted pattern below. It
#: only has to exceed the widest channel count the suite builds ($109$), and $1000$ keeps the
#: value readable in a failure message: $t = 7$, $c = 52$ reads as $7052$.
PATTERN_STEP_SCALE = 1000.0


def patterned_feature_stream(batch_size: int, seq_len: int, channels: int) -> torch.Tensor:
    r"""A feature stream whose every element names its own $(b, t, c)$ position.

    Element $(b, t, c)$ is $b\,S T + t\,S + c$ with $S$ = :data:`PATTERN_STEP_SCALE`, so the value
    is unique across the whole tensor and the position it came from can be read straight off a
    mismatch. Every value is an exact integer well inside float32's $2^{24}$ exactly-representable
    range at the geometries this suite builds, so comparisons against it are ``torch.equal``
    rather than ``allclose``.

    Args:
        batch_size: Samples in the batch.
        seq_len: Decimated sequence length $T$.
        channels: Channel count of the concatenated stream.

    Returns:
        The stream ``(batch_size, seq_len, channels)``, float32.
    """
    sample = torch.arange(batch_size, dtype=torch.float32).view(-1, 1, 1)
    step = torch.arange(seq_len, dtype=torch.float32).view(1, -1, 1)
    channel = torch.arange(channels, dtype=torch.float32).view(1, 1, -1)
    return sample * (PATTERN_STEP_SCALE * seq_len) + step * PATTERN_STEP_SCALE + channel


def make_patterned_batch(batch_size: int = BATCH, seq_len: int = SEQ_LEN, seed: int = 0):
    """Build the sibling's stub batch, then plant the known pattern in the two target blocks.

    The sibling's batch is reused rather than rebuilt so the fields this model does not forecast
    -- ``up_st``, ``up_ph``, ``fhr``, ``up`` and the deliberate ``weight`` gap -- stay exactly what
    every other suite in the family sees. Only ``fhr_st`` and ``fhr_ph`` are overwritten, and they
    are overwritten with **one** pattern split at the block boundary, so the value at channel $c$
    of the *concatenated* target stream is $c$ -- which is the index the reach budget's keep-index
    indexes into.

    Args:
        batch_size: Samples in the batch.
        seq_len: Decimated sequence length $T$.
        seed: Seed for the untouched fields, so a batch is reproducible.

    Returns:
        The stub batch, with patterned ``fhr_st`` and ``fhr_ph``.
    """
    batch = make_stub_batch(batch_size, seq_len, seed)
    st_channels = int(batch.fhr_st.shape[-1])
    ph_channels = int(batch.fhr_ph.shape[-1])
    stream = patterned_feature_stream(batch_size, seq_len, st_channels + ph_channels)
    batch.fhr_st = stream[..., :st_channels].contiguous()
    batch.fhr_ph = stream[..., st_channels:].contiguous()
    return batch


def resolve_target_budget(budget_s: Optional[float] = SHIPPED_REACH_BUDGET_S):
    """Resolve the causal reach budget at the production widths.

    The configuration block is assembled from :data:`SHIPPED_KWARGS` rather than written out, so a
    change to the production geometry reaches this suite instead of being shadowed by a copy.

    Args:
        budget_s: The budget in seconds, or ``None`` for the unguarded arm.

    Returns:
        The resolved ``ChannelBudget``, or ``None`` when ``budget_s`` is ``None``.
    """
    from teb_vae.lag_attn.channel_reach import resolve_stream_budgets

    config: Dict[str, Any] = {
        "causal_reach_budget_s": budget_s,
        "use_up_st": SHIPPED_KWARGS["use_up_st"],
        "warmup_period": SHIPPED_KWARGS["warmup_period"],
        "c_y": SHIPPED_KWARGS["c_y"],
        "c_u": SHIPPED_KWARGS["c_u"],
    }
    return resolve_stream_budgets(config)


def build_target_gate(budget_s: Optional[float] = SHIPPED_REACH_BUDGET_S):
    """Build the target stream's input guard exactly as the model builds it.

    Through the model class's own factory rather than by constructing a
    :class:`~teb_vae.lag_attn.nets.delays.ChannelGate` here: the factory decides what an absent
    budget means (no gate at all, not an identity one), and a second decision about that in the
    test suite is a second thing to keep in step.

    Args:
        budget_s: The budget in seconds, or ``None`` for the unguarded arm.

    Returns:
        The gate, or ``None`` when no budget is configured.
    """
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    budget = resolve_target_budget(budget_s)
    if budget is None:
        return SeqVaeLagAttnRws._build_channel_gate(SHIPPED_KWARGS["c_y"], None, None)
    return SeqVaeLagAttnRws._build_channel_gate(
        SHIPPED_KWARGS["c_y"], budget.target_keep_index, budget.target_delays
    )


#: A hand-made target guard for the tiny geometry, and the model width it implies. The shipped
#: budget cannot be resolved at the tiny geometry -- its maximum delay is $30$ steps against a
#: warm-up of $2$, which :func:`resolve_channel_budget` refuses -- so the gated path is exercised
#: here at a guard chosen to fit. The delays are non-zero and distinct: a target that applied them
#: would be wrong by a different amount in each channel, which is the failure the target builder's
#: gather-not-gate rule exists to prevent.
TINY_KEEP_INDEX = (0, 5, 9)
TINY_DELAYS = (0, 1, 2)
TINY_SOURCE_KEEP_INDEX = (2, 7)
TINY_SOURCE_DELAYS = (0, 2)


def tiny_gated_kwargs() -> Dict[str, Any]:
    """The tiny constructor kwargs with a small causal guard on both streams."""
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
    """The production constructor call: the shipped widths plus the resolved reach budget.

    What the trainer assembles, minus the trainer -- ``_build_model_kwargs`` forwards the config
    block and then updates it with exactly these four tuples. Built here rather than written out
    so the surviving-channel counts come from the filter bank every time.

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
def stub_batch():
    """A two-sample stub batch at the tiny geometry, with the deliberate weight gap."""
    return make_stub_batch(BATCH, SEQ_LEN)


@pytest.fixture
def patterned_batch():
    """A two-sample batch at the tiny geometry whose target blocks carry the known pattern."""
    return make_patterned_batch(BATCH, SEQ_LEN)


@pytest.fixture
def make_stub_batch_fn():
    """Factory fixture returning the sibling's :func:`make_stub_batch`."""
    return make_stub_batch


def _make_task(model_kwargs: dict | None = None, hparams: dict | None = None, **task_kwargs):
    """Build this model wrapped in its task, with the production loss hparams applied.

    The sibling's own factory is deliberately not reused: it names the sibling's model and task
    classes, and what every test below is about is that *this* pair behaves as that one does.
    Imported lazily so the pure-net tests never pay for Lightning.

    Args:
        model_kwargs: Net constructor kwargs. Defaults to :data:`TINY_KWARGS`.
        hparams: Loss hparam overrides on top of :data:`TASK_HPARAMS`.
        **task_kwargs: Passed through to the task's constructor.

    Returns:
        A ``SeqVaeLagAttnFsTask`` with ``setup()`` already called, so the permutation generator
        exists exactly as it would under a real fit.
    """
    from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
    from teb_vae.lag_attn_fs.task import SeqVaeLagAttnFsTask

    kwargs = dict(TINY_KWARGS if model_kwargs is None else model_kwargs)
    torch.manual_seed(0)
    model = SeqVaeLagAttnFs(**kwargs)
    task = SeqVaeLagAttnFsTask(
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
def tiny_gated():
    """A fresh copy of the tiny kwargs with the small causal guard (safe to mutate)."""
    return tiny_gated_kwargs()


@pytest.fixture
def shipped_gated():
    """A fresh copy of the production kwargs with the shipped reach budget (safe to mutate)."""
    return shipped_gated_kwargs()
