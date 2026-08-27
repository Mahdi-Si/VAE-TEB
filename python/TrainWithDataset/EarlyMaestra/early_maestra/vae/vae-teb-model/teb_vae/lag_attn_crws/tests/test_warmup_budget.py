r"""The warm-up budget resolves from the shards and reaches a constructor, or refuses by name.

This package resolves nothing itself. The budget is
:func:`~teb_vae.lag_attn_cfs.causal_warmup.resolve_warmup_budget`'s and the translation into
constructor keywords is :func:`~teb_vae.lag_attn_cfs.model_kwargs.warmup_model_kwargs`'s, both
reached by reference, so what is under test here is the **contract this package depends on** rather
than an implementation it owns. Three parts of that contract would each fail silently:

*The channel widths.* The keep-index and the warm-up vector are positional into the declared stream
width, so a boundary read against the wrong shards, the wrong trim or the wrong transform gathers
the wrong channels and waits the wrong number of steps for each -- with every shape intact. The
expected counts below are therefore derived from the committed fixture's own stored attributes; the
shipped row is additionally pinned as literals, because that row is the configuration every number
this package reports is produced at.

*The class.* The driver's config-to-constructor sweep drops any key naming no constructor argument.
The raw-signal architecture accepts both keep-indices and both **delay** vectors but neither warm-up
vector, so three quarters of the mapping lands and the two that matter are dropped in silence: an
ungated model that trains to completion, having read the assumed pre-recording history as signal on
coefficients whose normalisation constants excluded exactly that region. That partial match is the
concrete trap this package inherits, and it is asserted here rather than assumed -- it is also what
makes a model class of this package's own necessary at all.

*The binding.* Both names are the sibling's own objects rather than copies. A copy would be a second
declaration of which four keywords a resolved budget produces, free to fall out of step with the
constructor that takes them.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pytest

from hdf5_dataset.hdf5_dataset import decimated_trim_steps
from teb_vae.lag_attn_cfs import model_kwargs as sibling_model_kwargs
from teb_vae.lag_attn_cfs.causal_warmup import SOURCE_BLOCKS, TARGET_BLOCKS, resolve_warmup_budget

from . import conftest as local
from .conftest import (
    CAUSAL_C_U,
    CAUSAL_C_Y,
    CAUSAL_SHARD,
    SHIPPED_BUDGET_STEPS,
    SHIPPED_TRIM_MINUTES,
    TWO_SIDED_SHARD,
    ALIGN_MODEL_KWARGS,
    WARMUP_MODEL_KWARGS,
    causal_config,
    shipped_warmup_kwargs,
    stored_warmup,
    warmup_model_kwargs,
)

_PACKAGE_DIR = Path(__file__).resolve().parents[1]

#: What the shipped configuration must resolve to on the committed fixture, as literals. Everything
#: else below is derived from the shard's own attributes so a rebuild at another
#: ``causal_warmup_quantile`` re-derives rather than passes stale; this one row is pinned so that a
#: silent change to it fails here instead of quietly moving a training curve.
#: What the ALIGNMENT leaves, at this cell's own 42.21 s reference.
_SHIPPED_TARGET_KEPT = 38
_SHIPPED_SOURCE_KEPT = 17

#: What the WARM-UP BUDGET alone leaves, which is the unaligned arm's target width. Split from
#: the constant above because the two rules cut by different amounts once the reference moved
#: off `target_max`: the budget still keeps 98, the alignment then takes a further 60.
_BUDGET_TARGET_KEPT = 98


def _rebased_vector(blocks: Tuple[str, ...]) -> np.ndarray:
    r"""One stream's stored warm-up in the trimmed window's coordinates, blocks concatenated.

    Read straight off the shard's attributes and rebased as $W' = \max(W - \mathrm{trim}, 0)$, so
    the resolver is checked against the file rather than against itself.

    Args:
        blocks: The stored blocks making up the stream, in the model's concatenation order.

    Returns:
        The declared-width warm-up vector.
    """
    _, trim = decimated_trim_steps(SHIPPED_TRIM_MINUTES)
    stored = stored_warmup(CAUSAL_SHARD)
    return np.concatenate([np.maximum(stored[name] - trim, 0) for name in blocks])


class _WarmupModel:
    """A stand-in for this package's network, accepting exactly the keywords the real one will.

    A stub rather than the network itself: this file is about the *mapping*, and there is no model
    class in this package yet -- which is precisely the state in which the refusal below has to be
    established, since the refusal is the reason the class cannot simply be the raw-signal one.
    """

    def __init__(
        self,
        sequence_length: int = 300,
        horizon: int = 15,
        warmup_period: int = 134,
        c_y: int = CAUSAL_C_Y,
        c_u: int = CAUSAL_C_U,
        use_up_st: bool = True,
        target_keep_index: Optional[Sequence[int]] = None,
        target_warmup_steps: Optional[Sequence[int]] = None,
        source_keep_index: Optional[Sequence[int]] = None,
        source_warmup_steps: Optional[Sequence[int]] = None,
        target_align_delays: Optional[Sequence[int]] = None,
        source_align_delays: Optional[Sequence[int]] = None,
    ) -> None:
        self.kwargs: Dict[str, Any] = dict(locals())


# =================================================================================================
# The binding
# =================================================================================================
def test_both_names_are_the_siblings_own_objects():
    """Identity, so drift is structurally impossible rather than merely detected. A copy of either
    would be a second declaration of which four keywords a resolved budget produces."""
    assert local.warmup_model_kwargs is sibling_model_kwargs.warmup_model_kwargs
    assert local.WARMUP_MODEL_KWARGS is sibling_model_kwargs.WARMUP_MODEL_KWARGS


def test_this_package_ships_no_copy_of_the_resolver_or_the_mapping():
    """The other direction. Both modules sit above the net layer in the package that owns them --
    one opens HDF5 files, the other introspects a constructor -- and this package reaches them by
    reference, so a file of either name arriving here is a fork rather than a refactor."""
    for name in ("causal_warmup.py", "model_kwargs.py"):
        assert not (_PACKAGE_DIR / name).exists(), name


# =================================================================================================
# What the budget resolves to on the committed fixture
# =================================================================================================
def test_the_shipped_guard_keeps_ninety_eight_target_channels_and_forty_seven_source(
    budget, unaligned_budget
):
    """The shipped row, as literals, at the geometry this package's configuration declares.

    The two counts come from two different rules and both are stated. The **warm-up budget** takes
    four channels off the target and none off the source, which the unaligned arm shows directly;
    the **alignment reference** takes the four source channels slower than it and none off the
    target, because the reference is the target's own maximum.
    """
    assert budget.budget_steps == SHIPPED_BUDGET_STEPS
    assert budget.target.declared_width == CAUSAL_C_Y == 102
    assert budget.target.kept_width == _SHIPPED_TARGET_KEPT
    assert budget.source.declared_width == CAUSAL_C_U
    assert budget.source.kept_width == _SHIPPED_SOURCE_KEPT

    assert unaligned_budget.target.kept_width == _BUDGET_TARGET_KEPT
    assert unaligned_budget.source.kept_width == CAUSAL_C_U


def test_the_kept_set_is_exactly_the_channels_at_or_below_the_threshold(
    unaligned_budget, budget
):
    """Derived from the shard's own stored attributes, so a fixture rebuilt at another quantile
    re-derives the expectation instead of failing against a stale constant.

    Read off the UNALIGNED arm, for the reason the source test below gives: **two** rules narrow
    this stream and only one of them is the threshold. The alignment's further cut is asserted
    separately, so a passing threshold assertion can never stand in for a passing alignment one --
    which is exactly what would have happened had this kept reading the aligned budget once the
    reference moved off `target_max`.
    """
    declared = _rebased_vector(TARGET_BLOCKS)
    expected = np.flatnonzero(declared <= SHIPPED_BUDGET_STEPS)

    assert unaligned_budget.target.declared_warmup_steps == tuple(
        int(step) for step in declared
    )
    assert unaligned_budget.target.keep_index == tuple(int(index) for index in expected)
    assert unaligned_budget.target.warmup_steps == tuple(
        int(declared[index]) for index in expected
    )

    # The alignment keeps a strict subset of what the threshold keeps, and strictly fewer.
    assert set(budget.target.keep_index) < set(unaligned_budget.target.keep_index)
    assert budget.target.kept_width == _SHIPPED_TARGET_KEPT < _BUDGET_TARGET_KEPT


def test_the_source_is_never_gated_by_the_budget_however_slow_its_channels_are(unaligned_budget):
    r"""Its keep-index is the identity by construction rather than by arithmetic that happens to
    keep everything. Its slowest channels carry the contraction envelope, and the check is only
    meaningful because that channel is genuinely slower than the target's slowest survivor -- an
    ungated source that waited no longer would make it vacuous.

    Read off the unaligned arm, because the alignment reference *does* drop four source channels,
    for the unrelated reason that they are slower than the reference and could reach it only by
    being read from a later stored step. Mixing the two would make a passing assertion about the
    budget indistinguishable from one about the reference."""
    declared = _rebased_vector(SOURCE_BLOCKS)

    assert unaligned_budget.source.keep_index == tuple(range(len(declared)))
    assert unaligned_budget.source.declared_warmup_steps == tuple(int(step) for step in declared)
    assert unaligned_budget.source.max_warmup > unaligned_budget.target.max_warmup


def test_the_floor_this_package_declares_clears_both_streams_shifted_maxima(
    budget, unaligned_budget
):
    r"""The pairing the anchor floor is retained under, and which of its two halves now binds.

    Unaligned it was read off the target **survivors'** maximum -- a budget of $151$ keeps the
    identical channels -- as an input-warmth policy rather than a validity requirement: the first
    forecast step is $t + 1$, so $F \ge B - 1$ said every kept *target-stream* channel is warm by
    it, and the source stream was left cold for hundreds of steps deliberately, since gating it
    would cost the contraction envelope.

    Under the alignment the source is no longer left behind: every channel slower than the
    reference is dropped and every survivor is shifted onto it, so $\max_c(W'_c + d_c)$ is the same
    number on both streams.

    At this cell's $42.21$ s reference that number is small. The alignment keeps only fast
    channels, so $B = \max_c W'_c$ over the survivors is $1$ and $\max_c(W'_c + d_c)$ is $6$,
    giving a requirement of $\max(B - 1, 6) = 6$ against a shipped ``warmup_period`` of $134$.
    **The floor is therefore no longer a constraint but a retained anchor-cost policy**, which is
    what the assertions below state: the requirement is a lower bound the shipped value clears by
    more than twenty times, not an equality, because nothing now binds it.
    """
    assert budget.target.max_warmup == 1
    assert local.SHIPPED_KWARGS["warmup_period"] == SHIPPED_BUDGET_STEPS
    for stream in (budget.target, budget.source):
        assert max(stream.combined_steps) == 6, stream.name
        assert local.SHIPPED_KWARGS["warmup_period"] > max(stream.combined_steps), stream.name
    # And the reading that was retired: over the UNALIGNED source, a policy read across every input
    # channel would have put the floor at a much larger number, which is why it was not read there.
    assert local.SHIPPED_KWARGS["warmup_period"] < unaligned_budget.source.max_warmup - 1


# =================================================================================================
# The mapping into constructor keywords
# =================================================================================================
def test_the_four_resolved_tuples_reach_a_constructor_that_accepts_them(unaligned_budget):
    """The positive path, against the stub: the four tuples arrive under the warm-up names, paired
    positionally, with no delay keyword anywhere near them.

    Read off the unaligned arm so the set is exactly the four; the aligned arm adds the two shift
    vectors under their own names, which the test below covers."""
    budget = unaligned_budget
    mapped = warmup_model_kwargs(budget, _WarmupModel)

    assert set(mapped) == set(WARMUP_MODEL_KWARGS)
    assert mapped["target_keep_index"] == budget.target.keep_index
    assert mapped["target_warmup_steps"] == budget.target.warmup_steps
    assert mapped["source_keep_index"] == budget.source.keep_index
    assert mapped["source_warmup_steps"] == budget.source.warmup_steps
    assert len(mapped["target_keep_index"]) == len(mapped["target_warmup_steps"])
    assert not {"target_delays", "source_delays"} & set(mapped)

    built = _WarmupModel(**mapped)
    assert built.kwargs["target_warmup_steps"] == budget.target.warmup_steps


def test_the_shipped_keyword_set_carries_the_budget_and_the_declared_widths(budget):
    """What a real construction call looks like: the production geometry plus the four tuples, with
    the widths still the **declared** ones -- the gate is the model's own, so a set narrowed to the
    survivors would build a model that could not read the stream it is given."""
    kwargs = shipped_warmup_kwargs(_WarmupModel)

    assert set(WARMUP_MODEL_KWARGS) <= set(kwargs)
    assert kwargs["target_keep_index"] == budget.target.keep_index
    assert kwargs["c_y"] == CAUSAL_C_Y and kwargs["c_u"] == CAUSAL_C_U
    assert len(kwargs["target_keep_index"]) < kwargs["c_y"]
    assert kwargs["horizon"] == 30 and kwargs["warmup_period"] == 134
    # Applied last, which is what makes an arm expressible as one keyword at the call site.
    assert shipped_warmup_kwargs(_WarmupModel, horizon=15)["horizon"] == 15


def test_the_two_shift_vectors_reach_the_same_constructor_under_their_own_names(budget):
    """The alignment travels beside the warm-up rather than inside it, and is positionally paired
    with the keep-index resolved beside it: a shift vector is one entry per SURVIVING channel."""
    mapped = warmup_model_kwargs(budget, _WarmupModel)

    assert set(WARMUP_MODEL_KWARGS) | set(ALIGN_MODEL_KWARGS) == set(mapped)
    assert mapped["target_align_delays"] == budget.target.align_delays
    assert mapped["source_align_delays"] == budget.source.align_delays
    assert len(mapped["target_align_delays"]) == len(mapped["target_keep_index"])
    assert len(mapped["source_align_delays"]) == len(mapped["source_keep_index"])


def test_no_budget_adds_no_keys():
    """An unguarded run gets no gate and no warm-up mask, not an identity one."""
    assert warmup_model_kwargs(None, _WarmupModel) == {}
    assert resolve_warmup_budget(causal_config(causal_warmup_budget_steps=None)) is None


def test_the_raw_signal_architecture_is_refused_naming_both_warm_up_keywords(budget):
    """The refusal this package exists downstream of, and the reason it needs a model class of its
    own rather than the raw-signal one.

    It is a **partial** match, not a miss: that constructor accepts both keep-indices, so a driver
    pointed at it would gate the channels and skip the masking, and the run would report nothing
    about the region it read as signal.
    """
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    accepted = set(inspect.signature(SeqVaeLagAttnRws.__init__).parameters)
    assert "target_keep_index" in accepted, "the trap this guards is a partial match, not a miss"
    assert "source_keep_index" in accepted
    assert not {"target_warmup_steps", "source_warmup_steps"} & accepted

    with pytest.raises(ValueError) as error:
        warmup_model_kwargs(budget, SeqVaeLagAttnRws)
    message = str(error.value)

    assert "SeqVaeLagAttnRws" in message
    assert "target_warmup_steps" in message and "source_warmup_steps" in message


def test_the_refusal_does_not_fire_when_no_budget_is_configured():
    """The negative control: with nothing to route, the model class is nobody's business."""
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    assert warmup_model_kwargs(None, SeqVaeLagAttnRws) == {}


# =================================================================================================
# The refusals this package's configuration must not be able to trip
# =================================================================================================
def test_a_reach_budget_beside_a_warm_up_budget_is_refused_naming_both_keys():
    """The two-sided guard measures a forward reach on the production Morlet bank, which did not
    produce these coefficients -- and it resolves *delays*, which shift, so the model would read
    every gated channel late on top of its warm-up."""
    with pytest.raises(ValueError) as error:
        resolve_warmup_budget(causal_config(causal_reach_budget_s=120.0))
    message = str(error.value)

    assert "causal_warmup_budget_steps" in message
    assert "causal_reach_budget_s" in message
    assert "120.0" in message


def test_a_two_sided_shard_is_refused_naming_it():
    """It has no warm-up at all, and reading that absence as "fully valid" is the claim this cell
    exists to deny: a two-sided coefficient at step $t$ already contains $t$'s own future."""
    with pytest.raises(ValueError) as error:
        resolve_warmup_budget(causal_config(paths=[TWO_SIDED_SHARD]))
    message = str(error.value)

    assert TWO_SIDED_SHARD.name in message
    assert "two_sided" in message


@pytest.mark.parametrize("trim", (2.0, 0.5, None))
def test_a_trim_that_is_not_the_loaders_is_refused_naming_both_config_paths(trim):
    r"""The one failure mode no warm-fraction readout can see. A uniformly wrong rebase moves the
    anchor floor and the validity boundary by the same amount, so every "warm at every scored step"
    readout still reports exactly $1.0$ while the model reads pad. The cross-check is against the
    other declaration of the same geometry: $T$ is the stored length minus twice the trim."""
    with pytest.raises(ValueError) as error:
        resolve_warmup_budget(causal_config(trim_minutes=trim))
    message = str(error.value)

    assert "dataset_config.dataloader_config.dataset_kwargs.trim_minutes" in message
    assert "model_config.VAE_model.sequence_length" in message


def test_the_shipped_configuration_is_the_negative_control_for_all_three(config, budget):
    """Only one leaf moved in each refusal above; unmoved, this package's own configuration
    resolves."""
    assert resolve_warmup_budget(config) is not None
    assert budget.trim_minutes == SHIPPED_TRIM_MINUTES
    assert config["model_config"]["VAE_model"]["causal_reach_budget_s"] is None
