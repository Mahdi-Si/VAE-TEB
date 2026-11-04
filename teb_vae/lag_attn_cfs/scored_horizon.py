r"""The per-channel scored horizon $H_c$: how many leading forecast steps of each target channel count.

A single horizon $H$ is the wrong length for a block that mixes envelopes and phase products.
Measured held-out (``tmp/cfs_channel_horizon``), every ``fhr_st`` envelope stays forecastable from
FHR history across the whole horizon, while an ``fhr_ph`` coefficient whose slow leg sits above
the contraction band is a zero-mean random sign after a few steps: the relative phase of two
band-limited processes decorrelates within about the inverse bandwidth, and the low-pass
$\phi$ only holds it for its own support. Scoring those cells charges the objective for noise
nothing -- the source included -- can predict, and at a long horizon they are most of the block.

So the rule, stated in two configuration keys under ``model_config.VAE_model``:

$$H_c = \begin{cases} H_{\mathrm{fast}} & c \in \texttt{fhr\_ph},\ \xi_{i(c)} > f_{\mathrm{cut}} \\
H & \text{otherwise,} \end{cases}$$

with $\xi_{i(c)}$ the slow-leg centre frequency the shard records per phase channel
(``sel_xi_i_hz``). Contraction-locked decelerations put their energy at or below the cutoff, which
is where the source's effect lands, so those channels keep the full horizon.

``target_phase_fast_cutoff_hz`` and ``target_phase_fast_horizon`` both ``null`` (or absent)
resolves to ``None`` -- every cell scored -- which is every cell of the family but the one that
opts in.

**The scored start $\tau^{0}_c$.** A one-sided coefficient at step $t + 1 + \tau$ is a weighted sum
of raw samples, and the shards record, per channel, the share of that weight lying after the
anchor $t$ (``causal_novelty_curve``, column $k$ for the element $k$ steps after the anchor). A
slow channel's coefficient one step ahead is about $99\%$ samples the anchor has already seen, so
forecasting it is reproducing the past, its innovation is tiny, and its log-variance sits on the
clamp floor. With ``target_min_novelty`` $\varepsilon$ set, the first scored step of channel $c$ is

$$\tau^{0}_c = \min\{\tau \in [0, H) : \nu_c(\tau + 1) \ge \varepsilon\},$$

and $H$ when no step qualifies, so the channel is not scored. ``null`` (or absent) resolves to
``None`` and every channel starts at step $0$.

The threshold may also be a mapping from target block to threshold, such as
``{fhr_st: 0.05, fhr_ph: null}``. A block that is absent from the mapping, or mapped to ``null``,
starts every channel at step $0$. The per-block form exists because the rule separates the two
blocks differently. At the epoch-524 checkpoint, a threshold of $0.05$ removed scattering cells of
which $76\%$ sat on the log-variance floor and kept cells of which $6\%$ did. On the phase block
the removed and the kept cells were on the floor at about the same rate ($52\%$ and $37\%$), and
the rule removed three quarters of the phase cells.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import h5py
import numpy as np

CUTOFF_KEY = "target_phase_fast_cutoff_hz"
FAST_HORIZON_KEY = "target_phase_fast_horizon"
MIN_NOVELTY_KEY = "target_min_novelty"

#: The two stored target blocks, in the declared order the model concatenates them.
TARGET_BLOCKS = ("fhr_st", "fhr_ph")


def resolve_target_scored_horizon(config: Mapping[str, Any]) -> Optional[Tuple[int, ...]]:
    r"""Resolve $H_c$ per **declared** target channel from the configuration and the shards.

    Args:
        config: The full run configuration. ``model_config.VAE_model`` supplies the two rule keys,
            ``horizon`` and ``c_y``; ``dataset_config.vae_train_datasets`` (or, failing that,
            ``vae_test_datasets``) supplies the shard whose phase-leg frequencies are read.

    Returns:
        One integer per declared target channel, in $[1, H]$, or ``None`` when neither rule key is
        set.

    Raises:
        ValueError: If only one of the two keys is set, if the fast horizon is not in $[1, H]$, if
            no shard is configured, if the shard's phase block carries no ``sel_xi_i_hz``, or if the
            two stored blocks' widths do not add up to the declared ``c_y``.
    """
    vae = (config.get("model_config") or {}).get("VAE_model") or {}
    cutoff, fast = vae.get(CUTOFF_KEY), vae.get(FAST_HORIZON_KEY)
    if cutoff is None and fast is None:
        return None
    if cutoff is None or fast is None:
        raise ValueError(
            f"{CUTOFF_KEY} and {FAST_HORIZON_KEY} are a pair: set both or neither "
            f"(got {cutoff!r} and {fast!r})"
        )
    horizon = int(vae["horizon"])
    fast = int(fast)
    if not 1 <= fast <= horizon:
        raise ValueError(f"{FAST_HORIZON_KEY}={fast} must lie in [1, H={horizon}]")

    dataset = config.get("dataset_config") or {}
    paths = list(dataset.get("vae_train_datasets") or dataset.get("vae_test_datasets") or [])
    if not paths:
        raise ValueError(
            f"{CUTOFF_KEY} is set but dataset_config names no shard to read the phase-leg "
            f"frequencies from"
        )
    with h5py.File(paths[0], "r") as handle:
        width_st = int(handle[TARGET_BLOCKS[0]].shape[1])
        attrs = handle[TARGET_BLOCKS[1]].attrs
        if "sel_xi_i_hz" not in attrs:
            raise ValueError(
                f"{paths[0]}: the {TARGET_BLOCKS[1]} block carries no sel_xi_i_hz, so the phase "
                f"channels' slow-leg frequencies are unknown; rebuild the shard with the current "
                f"writer or unset {CUTOFF_KEY}"
            )
        xi_slow = np.asarray(attrs["sel_xi_i_hz"], dtype=float)

    c_y = int(vae["c_y"])
    if width_st + xi_slow.size != c_y:
        raise ValueError(
            f"the shard's target blocks hold {width_st} + {xi_slow.size} channels but c_y={c_y}"
        )
    phase = tuple(fast if xi > float(cutoff) else horizon for xi in xi_slow)
    return (horizon,) * width_st + phase


def resolve_target_scored_start(config: Mapping[str, Any]) -> Optional[Tuple[int, ...]]:
    r"""Resolve $\tau^{0}_c$ per **declared** target channel from ``target_min_novelty`` and the shards.

    Args:
        config: The full run configuration. ``model_config.VAE_model`` supplies
            ``target_min_novelty`` (a number, or a mapping from target block to a number or
            ``null``), ``horizon``, ``c_y`` and ``causal_target_forecast_clock``;
            ``dataset_config.vae_train_datasets`` (or, failing that, ``vae_test_datasets``) supplies
            the shard whose ``causal_novelty_curve`` is read.

    Returns:
        One integer per declared target channel, in $[0, H]$, or ``None`` when
        ``target_min_novelty`` is unset.

    Raises:
        ValueError: If a threshold is outside $(0, 1]$, if the mapping names a block that is not a
            target block, if the forecast clock is not ``stored``
            (the curve column would then depend on each channel's forecast shift), if no shard is
            configured, if a target block carries no ``causal_novelty_curve``, if the curve is
            tabulated to fewer than $H$ steps, or if the blocks' widths do not add up to ``c_y``.
    """
    vae = (config.get("model_config") or {}).get("VAE_model") or {}
    thresholds = _block_thresholds(vae.get(MIN_NOVELTY_KEY))
    if thresholds is None:
        return None
    clock = vae.get("causal_target_forecast_clock", "stored") or "stored"
    if clock != "stored":
        raise ValueError(
            f"{MIN_NOVELTY_KEY} is defined on the stored forecast clock only, got "
            f"causal_target_forecast_clock={clock!r}"
        )
    horizon = int(vae["horizon"])

    dataset = config.get("dataset_config") or {}
    paths = list(dataset.get("vae_train_datasets") or dataset.get("vae_test_datasets") or [])
    if not paths:
        raise ValueError(
            f"{MIN_NOVELTY_KEY} is set but dataset_config names no shard to read the novelty "
            f"curve from"
        )
    starts = []
    with h5py.File(paths[0], "r") as handle:
        for block in TARGET_BLOCKS:
            width = int(handle[block].shape[1])
            threshold = thresholds.get(block)
            if threshold is None:
                starts.extend([0] * width)
                continue
            attrs = handle[block].attrs
            if "causal_novelty_curve" not in attrs:
                raise ValueError(
                    f"{paths[0]}: the {block} block carries no causal_novelty_curve; rebuild the "
                    f"shard with the current writer or unset {MIN_NOVELTY_KEY}"
                )
            curve = np.asarray(attrs["causal_novelty_curve"], dtype=float)
            if curve.shape[0] != width:
                raise ValueError(
                    f"{paths[0]}: the {block} novelty curve has {curve.shape[0]} rows for "
                    f"{width} channels"
                )
            if curve.shape[1] < horizon + 1:
                raise ValueError(
                    f"the {block} novelty curve is tabulated to {curve.shape[1] - 1} steps, "
                    f"fewer than H={horizon}"
                )
            # Column tau + 1 is the scored element at horizon step tau on the stored clock.
            novel = curve[:, 1 : horizon + 1] >= threshold
            starts.extend(int(np.argmax(row)) if row.any() else horizon for row in novel)
    c_y = int(vae["c_y"])
    if len(starts) != c_y:
        raise ValueError(f"the shard's target blocks hold {len(starts)} channels but c_y={c_y}")
    return tuple(starts)


def _block_thresholds(value: Any) -> Optional[Dict[str, Optional[float]]]:
    """Normalise ``target_min_novelty`` to a per-block mapping.

    Args:
        value: ``None``, a number, or a mapping from target block to a number or ``None``.

    Returns:
        ``None`` when no block is filtered; otherwise ``{block: threshold or None}`` for every
        target block.

    Raises:
        ValueError: If a threshold is outside $(0, 1]$ or the mapping names an unknown block.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - set(TARGET_BLOCKS))
        if unknown:
            raise ValueError(
                f"{MIN_NOVELTY_KEY} names {unknown}, which are not target blocks {TARGET_BLOCKS}"
            )
        thresholds = {block: value.get(block) for block in TARGET_BLOCKS}
    else:
        thresholds = {block: value for block in TARGET_BLOCKS}
    resolved: Dict[str, Optional[float]] = {}
    for block, threshold in thresholds.items():
        if threshold is not None:
            threshold = float(threshold)
            if not 0.0 < threshold <= 1.0:
                raise ValueError(f"{MIN_NOVELTY_KEY}[{block}]={threshold} must lie in (0, 1]")
        resolved[block] = threshold
    if all(threshold is None for threshold in resolved.values()):
        return None
    return resolved
