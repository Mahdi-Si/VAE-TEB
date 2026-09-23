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
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import h5py
import numpy as np

CUTOFF_KEY = "target_phase_fast_cutoff_hz"
FAST_HORIZON_KEY = "target_phase_fast_horizon"

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
