r"""The resolved warm-up budget, as the constructor keywords that carry it into a checkpoint.

The experiment driver's config-to-constructor sweep forwards any ``model_config.VAE_model`` key
naming a real constructor argument. ``causal_warmup_budget_steps`` names none: it is a *threshold*,
and what the network takes is the four concrete channel tuples it resolves to. This module is that
translation, and it is a free function rather than a driver method for two reasons -- it can be
asserted before a driver for this architecture exists, and the driver that will call it does so
from ``_build_model_kwargs`` rather than from ``create_model``, so the tuples land in the
``model_kwargs`` written into every checkpoint. The input adapters' widths depend on them, so a
checkpoint recording only the threshold could not be rebuilt without re-reading the shards.

**The keywords are new names, not the reach guard's.** ``target_delays`` and ``source_delays``
reach ``ChannelDelay``, which *shifts*: $\mathrm{out}[t,c] = x[t - \delta_c, c]$, leaving content
permanently late. A warm-up masks a leading region and leaves the rest at its own index. Routing
one under the other's name would train a different model in silence, and would make a checkpoint's
``model_kwargs`` ambiguous between two families under one key.

**The channel alignment travels under two more names, in a second tuple.** It *is* a shift, and it
does reach ``ChannelDelay`` -- but through ``target_align_delays`` and ``source_align_delays``,
which the four causal cells accept and translate, rather than through the reach guard's names,
which they still refuse. Two tuples rather than one six-entry tuple because they answer different
questions and one consumer subtracts the warm-up tuple to build the *ungated* comparison arm: a
model with its warm-up vectors removed and its shifts left in place is not that arm, and folding
the two together would make "ungated" mean two things at once.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

from teb_vae.lag_attn_cfs.causal_warmup import WarmupBudget

#: The constructor keywords a resolved warm-up budget produces, in stream then role order. Named
#: here because two other places need the same list -- the startup log abbreviates them, since each
#: is one entry per channel and written out they bury every other kwarg, and the resolved-config
#: record writes them in full -- and a second copy is a list that can go stale.
WARMUP_MODEL_KWARGS = (
    "target_keep_index",
    "target_warmup_steps",
    "source_keep_index",
    "source_warmup_steps",
)

#: The two the channel alignment produces, in the same stream order. **Separate** from the tuple
#: above rather than appended to it: ``tests/test_docs.py`` builds the ungated comparison arm by
#: removing :data:`WARMUP_MODEL_KWARGS` from a production keyword set, and a model that kept its
#: shifts while losing its warm-up mask is not the arm that number is meant to name. Emitted only
#: when a reference is configured, so an unaligned run's kwargs dict is byte for byte what it was.
ALIGN_MODEL_KWARGS = ("target_align_delays", "source_align_delays")

#: The forecast clock's signed shift, in a fourth tuple for a fourth reason: it moves neither an
#: input (the two tuples above) nor a readout (the one below) but the **question itself** -- which
#: stored step each kept target channel is scored at. Emitted only when the configured clock is not
#: ``'stored'``, so a stored-clock run's kwargs dict -- and therefore every checkpoint written
#: before the key existed -- is byte for byte what it was.
FORECAST_ALIGN_MODEL_KWARGS = ("target_forecast_shift",)

#: The readout vector, in a third tuple for a third reason: it is neither a guard nor a shift. It
#: partitions the *scored target* by the shard's stored novelty proxy -- the fixed-horizon,
#: stored-clock envelope-mass share of each channel after the anchor, a label rather than an exact
#: value fraction (CFS-08) -- and it changes no width, no mask and no parameter, which is exactly
#: why the ungated comparison arm keeps it while dropping the two tuples above.
#:
#: Emitted only to a class that accepts it, which is the **feature**-target pair. The two
#: raw-target cells forecast a raw signal: its last axis counts samples of one trace, every one of
#: which lies strictly after the anchor, so a per-channel novelty share is not a smaller number
#: there but an undefined one. That is a real asymmetry between the architectures rather than a
#: keyword one of them forgot, so it is a signature check and not a refusal.
NOVELTY_MODEL_KWARGS = ("target_novelty_frac",)


def warmup_model_kwargs(
    budget: Optional[WarmupBudget], model_cls: type
) -> Dict[str, Any]:
    """Translate a resolved warm-up budget into constructor kwargs for ``model_cls``.

    The model class is a parameter rather than an assumption because the failure it guards against
    is silent for a long time: a driver whose ``MODEL_CLS`` still points at an architecture without
    the warm-up keywords would drop them at the signature sweep and build an *ungated* model that
    trains to completion, reporting nothing about the region it read as signal.

    Args:
        budget: The resolved budget, or ``None`` when none is configured.
        model_cls: The network class these kwargs will construct.

    Returns:
        The four channel tuples, plus the two alignment vectors when the budget carries a
        reference, plus the declared novelty vector when ``model_cls`` is a feature-target model;
        or an empty dict when ``budget`` is ``None`` -- which leaves the model with no gate and no
        warm-up mask, exactly as an unguarded run wants.

    Raises:
        ValueError: If ``model_cls`` does not accept the warm-up keywords, or -- when a reference
            is configured -- the alignment ones, or -- when it is a feature-target model -- if the
            configured shards carry no novelty vector to split its block score by.
    """
    if budget is None:
        return {}

    accepted = set(inspect.signature(model_cls.__init__).parameters)
    missing = tuple(name for name in WARMUP_MODEL_KWARGS if name not in accepted)
    if missing:
        raise ValueError(
            f"a causal warm-up budget is configured but {model_cls.__module__}."
            f"{model_cls.__qualname__} accepts none of {missing}. That architecture cannot mask "
            f"its inputs at the warm-up, so it would read the assumed pre-recording history as "
            f"signal -- on coefficients whose normalisation constants excluded exactly that "
            f"region, so those values are on no defined scale."
        )

    mapped: Dict[str, Any] = {
        "target_keep_index": budget.target.keep_index,
        "target_warmup_steps": budget.target.warmup_steps,
        "source_keep_index": budget.source.keep_index,
        "source_warmup_steps": budget.source.warmup_steps,
    }

    # The novelty vector, for a feature target only, and refused rather than defaulted when the
    # shards do not carry it. There is no share a missing vector could stand in as: a zero would
    # report every channel as pure history and a one as pure forecast, and either would render as
    # three plausible tertile columns that measured the channel index. The attribute is written by
    # the current writer on every block, so a shard without it is one built by an earlier one and
    # the fix is to name the rebuilt variant.
    if any(name in accepted for name in NOVELTY_MODEL_KWARGS):
        if budget.target.declared_novelty_frac is None:
            raise ValueError(
                f"{model_cls.__module__}.{model_cls.__qualname__} splits its forecast gap by "
                f"novelty tertile, but the configured shards carry no causal_novelty_frac on at "
                f"least one of the target blocks {budget.target.block_spans[0][0]!r} / "
                f"{budget.target.block_spans[-1][0]!r}. That attribute is written by the current "
                f"pipeline on every causal block, so these shards predate it; point the run at the "
                f"rebuilt variant rather than at a default, which would report the channel index "
                f"as a novelty split."
            )
        mapped["target_novelty_frac"] = budget.target.declared_novelty_frac

    # The forecast clock's shift, before the alignment gate below: the two are independent -- the
    # physical clock is resolvable on an unaligned run -- so its emission cannot sit behind the
    # reference check. Refused rather than dropped for the reason both refusals above give: a model
    # without the keyword would score every channel at its stored index while the run's config
    # states another clock, with every shape correct and the anchor count the only witness.
    if budget.target_forecast_shift is not None:
        if FORECAST_ALIGN_MODEL_KWARGS[0] not in accepted:
            raise ValueError(
                f"a {budget.target_forecast_clock!r} forecast clock is configured but "
                f"{model_cls.__module__}.{model_cls.__qualname__} does not accept "
                f"{FORECAST_ALIGN_MODEL_KWARGS[0]!r}. That architecture can only score each "
                f"target channel at its own stored index, so it would silently answer the stored "
                f"clock's question while the run records another one."
            )
        mapped["target_forecast_shift"] = budget.target_forecast_shift

    if budget.reference_delay_s is None:
        return mapped

    # A separate refusal from the one above, with its own sentence, because the two failures are
    # not the same: an architecture that cannot mask reads pad as signal, while one that cannot
    # shift reads a vector whose entries describe thirteen minutes of different instants as though
    # they described one. Both are silent, and neither message would explain the other.
    missing = tuple(name for name in ALIGN_MODEL_KWARGS if name not in accepted)
    if missing:
        raise ValueError(
            f"a causal alignment reference of {budget.reference_delay_s:.4f} s is configured but "
            f"{model_cls.__module__}.{model_cls.__qualname__} accepts none of {missing}. That "
            f"architecture cannot shift its input channels onto a common clock, so it would read "
            f"each stream's channel vector as one instant when its entries are stale by anything "
            f"from 13.3 s to the reference -- and every width would still be correct."
        )
    mapped["target_align_delays"] = budget.target.align_delays
    mapped["source_align_delays"] = budget.source.align_delays
    return mapped
