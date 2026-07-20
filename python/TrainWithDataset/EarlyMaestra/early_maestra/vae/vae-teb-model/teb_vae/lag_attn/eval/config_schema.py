r"""Validation of the ``eval_config`` block, run before anything is built.

YAML has no schema, and a mapping silently absorbs whatever it is given: ``max_sample``
instead of ``max_samples`` parses cleanly, is never read, and means "no cap" -- a run that
took four hours instead of twenty minutes and reported nothing about why. The same class of
failure the model's documentation records for misspelled constructor kwargs.

So the block is validated at load, before the model, the loader or the output directory
exist, and a rejection names both the offending key and the valid set.

What is *not* validated here, deliberately: the names inside ``caps``. That set grows with
every analysis added, and a table listing analyses that do not exist yet would be wrong more
often than right. Cap values are checked; an unread cap name is inert rather than dangerous,
because a cap only ever narrows what an analysis retains.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

#: Every key the block accepts. Anything else raises -- see the module docstring.
VALID_KEYS = frozenset(
    {
        "seed",
        "max_samples",
        "caps",
        "bands",
        "up_shift_secs",
        "health_probe_floor",
        "saturation_flag_threshold",
        "ablation_batch_size",
    }
)

#: Values used when a key is absent. A partial block is legitimate -- a variant config
#: overriding one cap should not have to restate the rest -- but a *misspelled* key is not,
#: which is why absence is defaulted and an unknown key raises.
DEFAULTS: Dict[str, Any] = {
    "seed": 42,
    "max_samples": None,
    "caps": {},
    "bands": {},
    "up_shift_secs": 0.0,
    "health_probe_floor": 0.0,
    "saturation_flag_threshold": 0.05,
    "ablation_batch_size": None,
}

#: Upper bound on the seed: ``numpy.random.seed`` rejects anything outside $[0, 2^{32})$,
#: and torch's own seeding is unbounded, so this is the binding constraint of the three.
_MAX_SEED = 2**32


def _require_int(value: Any, name: str, *, minimum: int) -> int:
    """Return ``value`` as an int, raising if it is not one or is below ``minimum``.

    ``bool`` is rejected explicitly: it is an ``int`` subclass in Python, so ``caps: {x: true}``
    would otherwise validate and then cap that analysis at one sample.

    Args:
        value: The configured value.
        name: Dotted config path, used in the error message.
        minimum: Smallest acceptable value, inclusive.

    Returns:
        The value as an ``int``.

    Raises:
        ValueError: If the value is not an integer or is below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"eval_config.{name} must be an integer, got {value!r} "
            f"({type(value).__name__})."
        )
    if value < minimum:
        raise ValueError(f"eval_config.{name} must be >= {minimum}, got {value}.")
    return int(value)


def _validate_caps(caps: Any) -> Dict[str, Optional[int]]:
    """Validate the per-analysis retention caps.

    Args:
        caps: The ``eval_config.caps`` mapping.

    Returns:
        The caps, with every value either ``None`` or a positive int.

    Raises:
        ValueError: If ``caps`` is not a mapping, or any cap is not a positive integer.
    """
    if not isinstance(caps, Mapping):
        raise ValueError(
            f"eval_config.caps must be a mapping of analysis name to sample cap, got "
            f"{type(caps).__name__}."
        )
    validated: Dict[str, Optional[int]] = {}
    for name, value in caps.items():
        if value is None:
            validated[str(name)] = None
            continue
        # A cap of 0 would retain nothing and produce an empty analysis that still reports
        # success, which is indistinguishable in the output from an analysis that found
        # nothing. Use None to mean "no cap" instead.
        validated[str(name)] = _require_int(value, f"caps.{name}", minimum=1)
    return validated


def _validate_bands(bands: Any, max_lag: int) -> Dict[str, tuple]:
    r"""Validate the lag-ablation bands against the model's lag window.

    A band is an inclusive ``[lo, hi]`` pair in model-lag units, so a band is *empty* exactly
    when ``lo > hi``. An empty band would produce an all-``False`` keep mask, which removes
    every causally valid lag at every anchor -- and ``entmax15``, which the shipped config
    enables, raises on a zero-support row rather than degrading like ``softmax``.

    Args:
        bands: The ``eval_config.bands`` mapping.
        max_lag: The model's ``max_lag``, so the window is $L = \mathrm{max\_lag} + 1$ wide.

    Returns:
        The bands as ``{name: (lo, hi)}``.

    Raises:
        ValueError: If ``bands`` is not a mapping, a band is not a pair of integers, a band is
            empty, or a band's lags fall outside $[0, \mathrm{max\_lag}]$.
    """
    if not isinstance(bands, Mapping):
        raise ValueError(
            f"eval_config.bands must be a mapping of band name to an inclusive [lo, hi] lag "
            f"pair, got {type(bands).__name__}."
        )
    validated: Dict[str, tuple] = {}
    for name, span in bands.items():
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError(
                f"eval_config.bands.{name} must be an inclusive [lo, hi] lag pair, got "
                f"{span!r}."
            )
        low = _require_int(span[0], f"bands.{name}[0]", minimum=0)
        high = _require_int(span[1], f"bands.{name}[1]", minimum=0)
        if low > high:
            raise ValueError(
                f"eval_config.bands.{name} = [{low}, {high}] is empty (lo > hi). An empty band "
                f"masks every lag, leaving the attention with no valid support at any anchor."
            )
        if high > max_lag:
            raise ValueError(
                f"eval_config.bands.{name} = [{low}, {high}] exceeds the model's max_lag="
                f"{max_lag}; lags run over [0, {max_lag}] (L = {max_lag + 1})."
            )
        validated[str(name)] = (low, high)
    return validated


def validate_eval_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    r"""Validate ``config['eval_config']`` and return it with defaults filled in.

    Call this immediately after :func:`teb_vae.lag_attn.config.load_config` and before any
    model, loader or output directory is built, so a misconfigured run costs a parse rather
    than a checkpoint load and a first pass over the shards.

    ``max_lag`` is read from ``model_config.VAE_model``. That is the config's claim about the
    geometry, not the checkpoint's -- the authoritative width comes from the rebuilt model and
    is re-checked in preflight -- but it is what is available this early, and a band outside
    it is already wrong.

    Args:
        config: The merged run config.

    Returns:
        The validated ``eval_config`` block. ``bands`` values are ``(lo, hi)`` tuples and
        every cap is ``None`` or a positive int.

    Raises:
        ValueError: If the block is not a mapping, carries an unknown key, or any value fails
            its range or type check.
    """
    block = config.get("eval_config")
    if block is None:
        block = {}
    if not isinstance(block, Mapping):
        raise ValueError(
            f"eval_config must be a mapping, got {type(block).__name__}."
        )

    unknown = sorted(set(block) - VALID_KEYS)
    if unknown:
        raise ValueError(
            f"unknown eval_config key(s): {', '.join(repr(key) for key in unknown)}. "
            f"Valid keys are: {', '.join(sorted(VALID_KEYS))}. Nothing reads an unrecognised "
            f"key, so a misspelling here would silently disable whatever it was meant to set."
        )

    resolved: Dict[str, Any] = dict(DEFAULTS)
    resolved["caps"] = dict(DEFAULTS["caps"])
    resolved["bands"] = dict(DEFAULTS["bands"])
    resolved.update(block)

    resolved["seed"] = _require_int(resolved["seed"], "seed", minimum=0)
    if resolved["seed"] >= _MAX_SEED:
        raise ValueError(
            f"eval_config.seed must be < {_MAX_SEED} (numpy's bound), got {resolved['seed']}."
        )

    if resolved["max_samples"] is not None:
        resolved["max_samples"] = _require_int(resolved["max_samples"], "max_samples", minimum=1)

    resolved["caps"] = _validate_caps(resolved["caps"])

    vae_config = (config.get("model_config") or {}).get("VAE_model") or {}
    max_lag = _require_int(vae_config.get("max_lag", 0), "bands (model_config.VAE_model.max_lag)", minimum=0)
    resolved["bands"] = _validate_bands(resolved["bands"], max_lag)

    shift = resolved["up_shift_secs"]
    if isinstance(shift, bool) or not isinstance(shift, (int, float)) or not math.isfinite(float(shift)):
        raise ValueError(
            f"eval_config.up_shift_secs must be a finite number of seconds, got {shift!r}."
        )
    resolved["up_shift_secs"] = float(shift)

    floor = resolved["health_probe_floor"]
    if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(float(floor)):
        raise ValueError(
            f"eval_config.health_probe_floor must be a finite number, got {floor!r}."
        )
    if float(floor) < 0.0:
        # The floor gates a ratio of RMS magnitudes, which is non-negative by construction; a
        # negative floor could never fire and would read as an active check that is not one.
        raise ValueError(
            f"eval_config.health_probe_floor must be >= 0, got {floor}. It gates a ratio of RMS "
            f"magnitudes, so a negative floor can never fire."
        )
    resolved["health_probe_floor"] = float(floor)

    saturation = resolved["saturation_flag_threshold"]
    if (
        isinstance(saturation, bool)
        or not isinstance(saturation, (int, float))
        or not math.isfinite(float(saturation))
        or not 0.0 <= float(saturation) <= 1.0
    ):
        # It gates a fraction of saturated elements, so a value outside [0, 1] is either an
        # always-firing flag or one that can never fire -- both read as an active check that is
        # not one.
        raise ValueError(
            f"eval_config.saturation_flag_threshold must be a fraction in [0, 1], got "
            f"{saturation!r}."
        )
    resolved["saturation_flag_threshold"] = float(saturation)

    if resolved["ablation_batch_size"] is not None:
        # The lag ablation runs one forward per band on top of the attention window's dense
        # clone, so its peak memory is the band count times a normal forward's. This splits the
        # loader's batch into micro-batches rather than changing the loader, which every other
        # analysis shares.
        resolved["ablation_batch_size"] = _require_int(
            resolved["ablation_batch_size"], "ablation_batch_size", minimum=1
        )

    return resolved
