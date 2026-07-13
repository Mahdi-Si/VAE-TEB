r"""S0-T04: arm resolution for ``synthetic_v4``.

Arms are a deep-merge of ``arms.<name>`` over the *whole* config, resolved by the reused,
schema-agnostic :func:`resolve_arm` / ``_deep_merge`` machinery. Because v4 is authored in the
``model_raw`` config schema and builds :class:`SeqVaeRawV4` through the trainer's ``create_model``
(not through ``pl_module_v2.build_model``), arm deltas author **directly** under the ``model_raw``
paths -- ``single_stride`` sets ``model_config.VAE_model.frontend.stages: [16]``,
``disable_source`` sets ``model_config.VAE_model.disable_source: true``, ``am_carrier_prod`` points
``experiment.data_tag`` at the am cache -- with **no** ``model.v4`` overlay.

The one arm that cannot be expressed through config is ``frontend_noncausal``: a forbidden
time-pooling norm would make ``SeqVaeRawV4.__init__`` refuse to build (via
``assert_no_time_pooling_norm``). It is therefore selected **by class** -- its ``arms`` entry
carries the ``_leaky_class: true`` marker, which S4-T02 reads to substitute the synthetic-side
``LeakyRawFrontendSeqVaeRawV4`` (which replaces ``frontend_y``/``frontend_u`` *after* a valid
causal ``super().__init__()``). This module exposes that marker check so the training driver never
hard-codes the arm name.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import resolve_arm

#: The ``arms.<name>`` key that selects the leaky (G0 negative-control) front-end model class.
LEAKY_CLASS_KEY: str = "_leaky_class"


def list_arms(config: Dict[str, Any]) -> List[str]:
    r"""Return the configured arm names (empty list when there is no ``arms`` block)."""
    return sorted((config.get("arms") or {}).keys())


def resolve_arm_v4(config: Dict[str, Any], arm: Optional[str]) -> Dict[str, Any]:
    r"""Deep-merge ``arms.<arm>`` over the whole config (reuses the v2 resolver verbatim).

    Args:
        config: The parsed ``config_synth_v4.yaml`` tree (with an ``arms`` block).
        arm: The arm name, or ``None`` for the base config.

    Returns:
        A new arm-resolved config (or the input unchanged when ``arm`` is ``None``).

    Raises:
        ValueError: If ``arm`` is given but absent from the ``arms`` block.
    """
    return resolve_arm(config, arm)


def arm_uses_leaky_frontend(config: Dict[str, Any], arm: Optional[str]) -> bool:
    r"""Whether ``arm`` selects the leaky (time-pooling) front-end control class.

    This is the single source of truth the S4-T02 driver consults to decide between
    :class:`SeqVaeRawV4` and the synthetic-side ``LeakyRawFrontendSeqVaeRawV4``; it never
    hard-codes the ``frontend_noncausal`` name.

    Args:
        config: The parsed config tree.
        arm: The arm name, or ``None``.

    Returns:
        ``True`` iff ``arms.<arm>`` carries a truthy ``_leaky_class`` marker.

    Raises:
        ValueError: If ``arm`` is given but absent from the ``arms`` block.
    """
    if arm is None:
        return False
    arms = config.get("arms") or {}
    if arm not in arms:
        raise ValueError(f"unknown arm {arm!r}; configured arms: {sorted(arms)}")
    return bool((arms[arm] or {}).get(LEAKY_CLASS_KEY, False))
