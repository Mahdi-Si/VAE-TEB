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
        The four channel tuples, or an empty dict when ``budget`` is ``None`` -- which leaves the
        model with no gate and no warm-up mask, exactly as an unguarded run wants.

    Raises:
        ValueError: If ``model_cls`` does not accept the warm-up keywords.
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

    return {
        "target_keep_index": budget.target.keep_index,
        "target_warmup_steps": budget.target.warmup_steps,
        "source_keep_index": budget.source.keep_index,
        "source_warmup_steps": budget.source.warmup_steps,
    }
