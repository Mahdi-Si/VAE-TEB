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
from teb_vae.lag_attn_cfs.model_kwargs import WARMUP_MODEL_KWARGS, warmup_model_kwargs
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


class _StubTrainer:
    """The driver's config-to-constructor sweep, and nothing else of a driver.

    The sweep is what makes the ``causal_warmup_budget_steps`` claim below testable: the threshold
    names no constructor argument, so the filter drops it with no special case, and asserting that
    is stronger than implementing a second exclusion list that could fall out of step with the
    constructor.
    """

    MODEL_CLS = _WarmupModel

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
