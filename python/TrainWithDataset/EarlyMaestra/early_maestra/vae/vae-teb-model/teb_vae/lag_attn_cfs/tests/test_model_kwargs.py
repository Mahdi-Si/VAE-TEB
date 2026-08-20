r"""The resolved budget reaches the constructor, under names that mean what they say.

Two silent failures live here and nothing else guards either.

The first is the **name**. ``target_delays`` and ``source_delays`` already exist, are already
forwarded by the family's config-to-constructor sweep, and reach ``ChannelDelay``, which *shifts*:
$\mathrm{out}[t,c] = x[t - \delta_c, c]$. A warm-up masks a leading region and leaves the rest at
its own index. Routed under the delay names, the resolved vectors would train a different model
with every shape intact, and a checkpoint's ``model_kwargs`` would be ambiguous between two
families under one key.

The second is the **class**. The sweep drops any key naming no constructor argument, so a driver
whose model class does not take the warm-up keywords builds an ungated model and trains it to
completion, having read the assumed pre-recording history as signal on coefficients whose
normalisation constants excluded exactly that region.

The mapping is asserted at dict level against a stub trainer rather than through a real driver:
there is no driver for this architecture yet, and one built early would inherit the raw model's
class attribute and build the wrong network for anyone who ran it.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Sequence

import pytest

from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget
from teb_vae.lag_attn_cfs.model_kwargs import (
    ALIGN_MODEL_KWARGS,
    NOVELTY_MODEL_KWARGS,
    WARMUP_MODEL_KWARGS,
    warmup_model_kwargs,
)
from teb_vae.lag_attn_cfs.tests.conftest import CAUSAL_C_U, CAUSAL_C_Y, causal_config

#: The names that must never carry a warm-up vector, whatever else changes.
DELAY_KWARGS = ("target_delays", "source_delays")


class _WarmupModel:
    """A stand-in for the network, accepting exactly the keywords the real one will.

    A stub rather than the network itself: this file is about the *mapping*, and pinning it to a
    class that does not exist yet would make the mapping untestable until it does.
    """

    def __init__(
        self,
        sequence_length: int = 300,
        horizon: int = 15,
        warmup_period: int = 133,
        c_y: int = CAUSAL_C_Y,
        c_u: int = CAUSAL_C_U,
        use_up_st: bool = True,
        target_keep_index: Optional[Sequence[int]] = None,
        target_warmup_steps: Optional[Sequence[int]] = None,
        source_keep_index: Optional[Sequence[int]] = None,
        source_warmup_steps: Optional[Sequence[int]] = None,
    ) -> None:
        self.kwargs = locals()


class _AlignedModel:
    """The same stand-in, with the two alignment keywords the real constructors gained.

    A second class rather than two more arguments on :class:`_WarmupModel`, and written out rather
    than subclassed with ``**kwargs``: the refusal below needs a class that takes the warm-up
    keywords and *not* the alignment ones -- exactly the partial match nothing else would catch --
    and the driver sweep reads ``inspect.signature``, where a ``**kwargs`` signature would silently
    stop forwarding every geometry key.
    """

    def __init__(
        self,
        sequence_length: int = 300,
        horizon: int = 15,
        warmup_period: int = 133,
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
        self.kwargs = locals()


class _FeatureTargetModel:
    """The aligned stand-in plus the readout keyword only a *feature*-target model takes.

    A third class rather than a flag, for the reason there are already two: the emission below is
    decided by ``inspect.signature``, and the negative control it needs is a class that takes
    everything else and not this -- which is exactly what the two raw-target cells are.
    """

    def __init__(
        self,
        sequence_length: int = 300,
        horizon: int = 15,
        warmup_period: int = 133,
        c_y: int = CAUSAL_C_Y,
        c_u: int = CAUSAL_C_U,
        use_up_st: bool = True,
        target_keep_index: Optional[Sequence[int]] = None,
        target_warmup_steps: Optional[Sequence[int]] = None,
        source_keep_index: Optional[Sequence[int]] = None,
        source_warmup_steps: Optional[Sequence[int]] = None,
        target_align_delays: Optional[Sequence[int]] = None,
        source_align_delays: Optional[Sequence[int]] = None,
        target_novelty_frac: Optional[Sequence[float]] = None,
    ) -> None:
        self.kwargs = locals()


class _StubTrainer:
    """The driver's config-to-constructor sweep, and nothing else of a driver.

    The sweep is what makes the ``causal_warmup_budget_steps`` claim below testable: the threshold
    names no constructor argument, so the filter drops it with no special case, and asserting that
    is stronger than implementing a second exclusion list that could fall out of step with the
    constructor.
    """

    MODEL_CLS = _AlignedModel

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def build_model_kwargs(self) -> Dict[str, Any]:
        """Forward every config key naming a constructor argument, then add the resolved budget."""
        vae_config = (self.config.get("model_config", {}) or {}).get("VAE_model", {}) or {}
        valid = set(inspect.signature(self.MODEL_CLS.__init__).parameters)
        kwargs = {
            name: value
            for name, value in vae_config.items()
            if name in valid and value is not None
        }
        kwargs.update(
            warmup_model_kwargs(resolve_warmup_budget(self.config), self.MODEL_CLS)
        )
        return kwargs


@pytest.fixture
def kwargs() -> Dict[str, Any]:
    """The constructor kwargs the shipped causal configuration produces."""
    return _StubTrainer(causal_config()).build_model_kwargs()


def test_the_four_resolved_tuples_reach_the_constructor(kwargs, budget) -> None:
    """And carry exactly what the resolver produced, positionally paired."""
    assert set(WARMUP_MODEL_KWARGS) <= set(kwargs)
    assert kwargs["target_keep_index"] == budget.target.keep_index
    assert kwargs["target_warmup_steps"] == budget.target.warmup_steps
    assert kwargs["source_keep_index"] == budget.source.keep_index
    assert kwargs["source_warmup_steps"] == budget.source.warmup_steps
    assert len(kwargs["target_keep_index"]) == len(kwargs["target_warmup_steps"])
    assert len(kwargs["source_keep_index"]) == len(kwargs["source_warmup_steps"])


def test_the_delay_keywords_stay_absent(kwargs) -> None:
    """Those names reach ``ChannelDelay`` and would apply a shift on top of the warm-up mask."""
    for name in DELAY_KWARGS:
        assert name not in kwargs, name
    assert not set(DELAY_KWARGS) & set(WARMUP_MODEL_KWARGS)


def test_the_threshold_itself_names_no_constructor_argument(kwargs) -> None:
    """So the sweep drops it for free, with no exclusion list to keep in step with the network."""
    assert "causal_warmup_budget_steps" not in inspect.signature(
        _WarmupModel.__init__
    ).parameters
    assert "causal_warmup_budget_steps" not in kwargs
    # The geometry keys around it are ordinary constructor arguments and must still be forwarded,
    # or the assertion above would also pass on a sweep that forwarded nothing at all.
    assert kwargs["c_y"] == CAUSAL_C_Y
    assert kwargs["horizon"] == causal_config()["model_config"]["VAE_model"]["horizon"]


def test_no_budget_adds_no_keys() -> None:
    """An unguarded run gets no gate and no warm-up mask, not an identity one."""
    assert warmup_model_kwargs(None, _WarmupModel) == {}
    kwargs = _StubTrainer(
        causal_config(causal_warmup_budget_steps=None)
    ).build_model_kwargs()
    assert not set(WARMUP_MODEL_KWARGS) & set(kwargs)
    assert kwargs["c_y"] == CAUSAL_C_Y


def test_a_model_class_without_the_warm_up_keywords_is_refused_naming_it(budget) -> None:
    """The failure this guard replaces is a full training run that reports nothing about it.

    The raw-signal model is the concrete case: it accepts both keep-indices and both delay vectors,
    so three quarters of the mapping would land and the two vectors that matter would be dropped by
    the signature sweep in silence.
    """
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    accepted = set(inspect.signature(SeqVaeLagAttnRws.__init__).parameters)
    assert "target_keep_index" in accepted, "the trap this guards is a partial match, not a miss"

    with pytest.raises(ValueError) as error:
        warmup_model_kwargs(budget, SeqVaeLagAttnRws)
    message = str(error.value)
    assert "SeqVaeLagAttnRws" in message
    assert "target_warmup_steps" in message and "source_warmup_steps" in message


def test_the_refusal_does_not_fire_when_no_budget_is_configured() -> None:
    """The negative control: with nothing to route, the model class is nobody's business."""
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    assert warmup_model_kwargs(None, SeqVaeLagAttnRws) == {}


# =================================================================================================
# The alignment travels under two more names, in a tuple of its own
#
# It IS a shift, so unlike the warm-up it legitimately reaches ``ChannelDelay`` -- but through its
# own keywords, which is what keeps a checkpoint's ``model_kwargs`` unambiguous about which of the
# two quantities produced a delayed stream. The tuple is separate from the warm-up's because one
# consumer subtracts that one to build the ungated comparison arm, and a model that kept its shifts
# while losing its mask is not that arm.
# =================================================================================================
@pytest.fixture
def aligned_kwargs() -> Dict[str, Any]:
    """The constructor kwargs the shipped configuration produces with the alignment on."""
    return _StubTrainer(
        causal_config(causal_align_reference="target_max")
    ).build_model_kwargs()


def test_the_two_alignment_vectors_reach_the_constructor(aligned_kwargs) -> None:
    """Positionally paired against the keep-indices resolved beside them."""
    resolved = resolve_warmup_budget(causal_config(causal_align_reference="target_max"))
    assert resolved is not None
    assert set(ALIGN_MODEL_KWARGS) <= set(aligned_kwargs)
    assert aligned_kwargs["target_align_delays"] == resolved.target.align_delays
    assert aligned_kwargs["source_align_delays"] == resolved.source.align_delays
    assert len(aligned_kwargs["target_align_delays"]) == len(
        aligned_kwargs["target_keep_index"]
    )
    assert len(aligned_kwargs["source_align_delays"]) == len(
        aligned_kwargs["source_keep_index"]
    )


def test_an_unaligned_run_emits_no_alignment_keys_at_all() -> None:
    """Absent rather than zero-filled: a vector of zeros is a shift that ran and did nothing, and
    it would build a ``ChannelDelay`` where the unaligned arm has none."""
    unaligned = _StubTrainer(
        causal_config(causal_align_reference=None)
    ).build_model_kwargs()

    assert not set(ALIGN_MODEL_KWARGS) & set(unaligned)
    assert set(WARMUP_MODEL_KWARGS) <= set(unaligned)


def test_the_two_tuples_stay_separate_and_disjoint() -> None:
    """Folding them together would make "ungated" mean two things at once.

    ``tests/test_docs.py`` builds the ungated comparison arm by removing
    :data:`WARMUP_MODEL_KWARGS` from a production keyword set. A six-entry tuple there would strip
    the shifts too, and the number it pins would stop describing the model it names.
    """
    assert not set(WARMUP_MODEL_KWARGS) & set(ALIGN_MODEL_KWARGS)
    assert not set(DELAY_KWARGS) & set(ALIGN_MODEL_KWARGS)
    assert len(ALIGN_MODEL_KWARGS) == 2


def test_the_reference_itself_names_no_constructor_argument(aligned_kwargs) -> None:
    """Like the threshold beside it, so the sweep drops it with no exclusion list to maintain."""
    assert "causal_align_reference" not in inspect.signature(
        _AlignedModel.__init__
    ).parameters
    assert "causal_align_reference" not in aligned_kwargs
    assert "causal_leg_alignment" not in aligned_kwargs


def test_a_model_class_that_cannot_shift_is_refused_naming_it() -> None:
    """A class taking the warm-up keywords but not the alignment ones passes the first refusal.

    That is the whole reason for a second one with its own sentence: the vectors would be dropped
    by the signature sweep and the run would read each stream's channel vector as one instant when
    its entries are stale by anything from 13.3 s to the reference, with every width correct.
    """
    resolved = resolve_warmup_budget(causal_config(causal_align_reference="target_max"))
    assert resolved is not None
    accepted = set(inspect.signature(_WarmupModel.__init__).parameters)
    assert set(WARMUP_MODEL_KWARGS) <= accepted, "the trap here is a partial match, not a miss"

    with pytest.raises(ValueError) as error:
        warmup_model_kwargs(resolved, _WarmupModel)
    message = str(error.value)
    assert "_WarmupModel" in message
    assert "target_align_delays" in message and "source_align_delays" in message
    assert "402.1604" in message


def test_that_refusal_does_not_fire_on_an_unaligned_budget(unaligned_budget) -> None:
    """The negative control: with no reference there is nothing to route, so a class that cannot
    shift is a perfectly good class."""
    mapped = warmup_model_kwargs(unaligned_budget, _WarmupModel)
    assert set(mapped) == set(WARMUP_MODEL_KWARGS)


# =================================================================================================
# The novelty vector: emitted to a feature target, and to nothing else
# =================================================================================================
def test_the_novelty_vector_reaches_a_feature_target_at_the_declared_width(budget) -> None:
    r"""One share per **declared** target channel, not per survivor.

    The width is the claim. The model gathers this vector through ``target_keep_index``, exactly as
    it does the per-block channel weights, so that the ungated comparison arm -- built by removing
    every resolved channel tuple and keeping the readouts -- still receives a vector positional
    against a width it has.
    """
    mapped = warmup_model_kwargs(budget, _FeatureTargetModel)

    assert set(NOVELTY_MODEL_KWARGS) <= set(mapped)
    vector = mapped["target_novelty_frac"]
    assert len(vector) == budget.target.declared_width == CAUSAL_C_Y
    assert vector == budget.target.declared_novelty_frac
    assert all(0.0 <= share <= 1.0 for share in vector)


def test_a_raw_target_class_gets_no_novelty_vector_and_is_not_refused(budget) -> None:
    """The asymmetry is real rather than an oversight, so it is a signature check and not a refusal.

    A raw target's last axis counts samples of one trace, every one of which lies strictly after the
    anchor: there is no per-channel novelty share to take, and a zero or a one would both be a
    fabricated column. The class that does not accept the keyword simply does not get it.
    """
    mapped = warmup_model_kwargs(budget, _AlignedModel)

    assert not set(NOVELTY_MODEL_KWARGS) & set(mapped)
    assert set(mapped) == set(WARMUP_MODEL_KWARGS) | set(ALIGN_MODEL_KWARGS)


def test_a_feature_target_on_shards_without_the_vector_is_refused_naming_it(tmp_path) -> None:
    """The one place the missing attribute must stop a run, and the reason it is here.

    The budget itself resolves against such a shard -- the novelty is a readout, not a guard -- so
    nothing earlier can refuse. What cannot be allowed is a feature cell reporting three tertile
    columns built from a default, which would rank the channels by declared index and read as a
    measurement.
    """
    from teb_vae.lag_attn_cfs.tests.conftest import CAUSAL_SHARD, write_variant

    def _strip(handle) -> None:
        for block in ("fhr_st", "fhr_ph", "up_st", "up_ph"):
            del handle[block].attrs["causal_novelty_frac"]

    legacy = write_variant(CAUSAL_SHARD, tmp_path / "no_novelty.hdf5", _strip)
    resolved = resolve_warmup_budget(causal_config(paths=[legacy]))
    assert resolved is not None, "the budget must still resolve; only the mapping refuses"

    # The negative control, on the same shard: a class with no such keyword maps cleanly.
    assert warmup_model_kwargs(resolved, _AlignedModel)

    with pytest.raises(ValueError) as error:
        warmup_model_kwargs(resolved, _FeatureTargetModel)
    message = str(error.value)
    assert "_FeatureTargetModel" in message
    assert "causal_novelty_frac" in message
    assert "fhr_st" in message and "fhr_ph" in message


def test_the_three_tuples_stay_separate_and_disjoint() -> None:
    """Three tuples because they answer three questions: what is masked, what is shifted, and what
    is only reported. The ungated comparison arm removes the first two and keeps the third."""
    names = (WARMUP_MODEL_KWARGS, ALIGN_MODEL_KWARGS, NOVELTY_MODEL_KWARGS)
    assert sum(len(group) for group in names) == len(set().union(*map(set, names)))
    assert not set(DELAY_KWARGS) & set(NOVELTY_MODEL_KWARGS)
    assert len(NOVELTY_MODEL_KWARGS) == 1
