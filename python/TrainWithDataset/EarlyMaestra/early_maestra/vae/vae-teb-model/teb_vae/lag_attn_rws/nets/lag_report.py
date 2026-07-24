r"""Turning an attention lag index into the two seconds figures that may be reported.

An attention peak at lag index $\ell$ is not by itself a physiological delay, for two reasons
that pull in opposite directions and are easy to conflate.

**The mechanical shift.** The preprocessing pipeline advances the uterine-pressure channel by
$20$ s to compensate a *known sensor* delay, so the streams the model sees are already
mechanically aligned. On that compensated timeline the residual delay the attention discovers is
the physiological one, and reporting it with the $20$ s added back would double-count a
correction that was made deliberately. The uncorrected figure is still worth storing -- it is
what maps a finding back to the original sensor files -- but it is a *different quantity* and is
named as such.

**The causal input delay.** When a channel is read at $t - \delta$ rather than at $t$ (the guard
that bounds how far an input feature reads into its own future), the source memory the attention
queries is itself $\delta$ steps stale. A peak at lag $\ell$ then refers to source content
$\ell + \delta$ steps back. With no guard configured, $\delta = 0$ and the correction vanishes.

Together, with $\Delta = 4$ s per decimated step:

$$
\tau_{\mathrm{compensated}} = \Delta\,(\ell + \delta), \qquad
\tau_{\mathrm{sensor}} = \tau_{\mathrm{compensated}} + 20\ \mathrm{s}.
$$

A standalone module with no model import, so a figure and an evaluation can share the arithmetic
without either reaching into the network.
"""
from __future__ import annotations

from typing import Union

import torch

#: Seconds per decimated step: the features are decimated by $16$ from a $4$ Hz raw grid.
SECONDS_PER_STEP = 4.0

#: The mechanical sensor delay the preprocessing pipeline already removes from the source
#: channel, in seconds. Added back only to map a lag onto the *uncorrected* sensor files.
MECHANICAL_SHIFT_SECONDS = 20.0

#: Axis label for a lag axis drawn in compensated seconds. A constant rather than a string in
#: each figure, so a plot and the number beside it cannot disagree about which quantity is shown.
COMPENSATED_LAG_AXIS_LABEL = "lag (s, mechanically compensated)"

#: Accepted lag types. Scalars for a single reported peak, tensors for a whole axis.
LagValue = Union[int, float, torch.Tensor]


def _validate_delay(delay_steps: int) -> int:
    """Return ``delay_steps`` as an int, having refused a negative one.

    Args:
        delay_steps: The causal input delay $\\delta$ in decimated steps.

    Returns:
        The delay as an ``int``.

    Raises:
        ValueError: If the delay is negative. A negative delay would mean the source memory is
            read from the future, which is the opposite of what the guard exists to prevent, and
            it would shorten the reported lag rather than lengthen it.
    """
    delay = int(delay_steps)
    if delay < 0:
        raise ValueError(
            f"delay_steps is the causal input delay in steps and must be >= 0, got {delay}"
        )
    return delay


def lag_compensated_seconds(
    lag_step: LagValue,
    *,
    delay_steps: int = 0,
    seconds_per_step: float = SECONDS_PER_STEP,
) -> LagValue:
    r"""The residual physiological lag: $\Delta\,(\ell + \delta)$ seconds.

    This is the quantity to report as *the* lag. The streams are mechanically aligned before the
    model sees them, so what the attention finds is the delay the physiology adds on top of the
    sensor delay that was already removed.

    Args:
        lag_step: Attention lag index $\ell$, a scalar or a whole axis as a tensor.
        delay_steps: The causal input delay $\delta$ applied to the source channels, in
            decimated steps. Zero when no reach budget is configured.
        seconds_per_step: Seconds per decimated step $\Delta$.

    Returns:
        The compensated lag in seconds, of the same kind as ``lag_step``.

    Raises:
        ValueError: If ``delay_steps`` is negative.
    """
    return float(seconds_per_step) * (lag_step + _validate_delay(delay_steps))


def lag_original_sensor_seconds(
    lag_step: LagValue,
    *,
    delay_steps: int = 0,
    seconds_per_step: float = SECONDS_PER_STEP,
    mechanical_shift_seconds: float = MECHANICAL_SHIFT_SECONDS,
) -> LagValue:
    r"""The same lag on the **uncorrected** sensor timeline: $\Delta\,(\ell + \delta) + 20$ s.

    Use this only to locate a finding in the original sensor files. It is not the physiological
    lag: it carries the $20$ s mechanical delay that preprocessing deliberately removed.

    Args:
        lag_step: Attention lag index $\ell$, a scalar or a whole axis as a tensor.
        delay_steps: The causal input delay $\delta$, in decimated steps.
        seconds_per_step: Seconds per decimated step $\Delta$.
        mechanical_shift_seconds: The sensor delay preprocessing removed, added back here.

    Returns:
        The uncorrected-timeline lag in seconds, of the same kind as ``lag_step``.

    Raises:
        ValueError: If ``delay_steps`` is negative.
    """
    compensated = lag_compensated_seconds(
        lag_step, delay_steps=delay_steps, seconds_per_step=seconds_per_step
    )
    return compensated + float(mechanical_shift_seconds)
