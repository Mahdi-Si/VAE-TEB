r"""Turning an attention lag index into the two seconds figures that may be reported.

An attention peak at lag index $\ell$ is not by itself a physiological delay, for two reasons
that pull in opposite directions and are easy to conflate.

**The mechanical shift.** The preprocessing pipeline advances the uterine-pressure channel by
$20$ s to compensate a *known sensor* delay, so the streams the model sees are already
mechanically aligned: stored position $g$ holds what the sensor recorded at $g + 20$ s. On that
compensated timeline the residual delay the attention discovers is the physiological one, and
adding the $20$ s back would double-count a correction that was made deliberately. The
uncorrected figure is still worth storing -- it is what maps a finding back to the original sensor
files -- but it is a *different quantity*, and it is reached by **subtracting** the shift rather
than adding it, because the shift pulled UP earlier.

**The causal input delay.** When a channel is read at $t - \delta$ rather than at $t$ (the guard
that bounds how far an input feature reads into its own future), the source memory the attention
queries is itself $\delta$ steps stale. A peak at lag $\ell$ then refers to source content
$\ell + \delta$ steps back. With no guard configured, $\delta = 0$ and the correction vanishes.

Together, with $\Delta = 4$ s per decimated step:

$$
\tau_{\mathrm{compensated}} = \Delta\,(\ell + \delta), \qquad
\tau_{\mathrm{sensor}} = \tau_{\mathrm{compensated}} - 20\ \mathrm{s}.
$$

**The group delay, when both streams are on one clock.** The two figures above are lags between
*stored coefficients*. A one-sided channel's content is itself stale by that channel's own composed
group delay $\tau_c$, so a lag between two of them is biased by $\tau^{u}_c - \tau^{y}_{c'}$ --
indexed by a channel *pair*, and therefore not a number a pooled state can carry. Once each stream
is shifted onto a common reference the pair index collapses to one constant, and the lead time
between the physical epochs the two coefficients summarise becomes reportable:

$$
\tau^{\mathrm{phys}}_{\ell,h} = \Delta\,(\ell + 1 + h)
  + \bigl(\tau^{u}_{\mathrm{ref}} - \tau^{y}_{\mathrm{ref}}\bigr)
  - 20\ \mathrm{s}.
$$

This is a *third* quantity, not a refinement of the first two, and it is built from a different
input: the two references are delays in **seconds**, resolved from the shards, and none of them is
the model's ``source_delay_steps``. That scalar is the largest shift the gate applies, attained by
the channel *furthest* from the reference, and using it here would be wrong by the whole reference.

A standalone module with no model import, so a figure and an evaluation can share the arithmetic
without either reaching into the network.
"""
from __future__ import annotations

from typing import Union

import torch

#: Seconds per decimated step: the features are decimated by $16$ from a $4$ Hz raw grid.
SECONDS_PER_STEP = 4.0

#: The mechanical sensor delay the preprocessing pipeline already removes from the source
#: channel, in seconds. Used only to map a lag onto the *uncorrected* sensor files, and
#: **subtracted** there rather than added: the preprocessing *advanced* UP, so undoing it moves the
#: reported figure down. See :func:`lag_original_sensor_seconds`.
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
    r"""The same lag on the **uncorrected** sensor timeline: $\Delta\,(\ell + \delta) - 20$ s.

    Use this only to locate a finding in the original sensor files. It is not the physiological
    lag: it carries the $20$ s mechanical shift that preprocessing deliberately removed.

    **Subtracted, not added.** Preprocessing *advanced* UP by $20$ s -- ``mimo_adaptor.py`` pulls
    the trace 80 samples earlier -- so stored position $g$ holds what the sensor recorded at
    $g + 20$ s. Undoing that to reach the sensor timeline moves the lag *down*. This returned
    ``compensated + 20`` until it was corrected, which put it $40$ s away from
    :func:`physical_lag_seconds` in this same module, with the sign reversed.

    Args:
        lag_step: Attention lag index $\ell$, a scalar or a whole axis as a tensor.
        delay_steps: The causal input delay $\delta$, in decimated steps.
        seconds_per_step: Seconds per decimated step $\Delta$.
        mechanical_shift_seconds: The sensor delay preprocessing removed, **subtracted** here to
            undo it. Preprocessing advanced UP, so reaching the uncorrected timeline moves the
            lag down; see the note above.

    Returns:
        The uncorrected-timeline lag in seconds, of the same kind as ``lag_step``.

    Raises:
        ValueError: If ``delay_steps`` is negative.
    """
    compensated = lag_compensated_seconds(
        lag_step, delay_steps=delay_steps, seconds_per_step=seconds_per_step
    )
    return compensated - float(mechanical_shift_seconds)


def _validate_reference(reference_s: float, name: str) -> float:
    """Return a stream's reference delay as a float, having refused a negative one.

    Args:
        reference_s: The reference $\\tau_{\\mathrm{ref}}$ in seconds.
        name: Which stream it is, for the message.

    Returns:
        The reference as a ``float``.

    Raises:
        ValueError: If it is negative. A reference is a composed group delay -- how far *back* a
            coefficient's content sits -- and a one-sided kernel supported on $[0, \\infty)$ has
            no negative one. A negative value here would shorten the reported lead time rather
            than fail, which is the same silent direction ``_validate_delay`` refuses.
    """
    value = float(reference_s)
    if value < 0.0:
        raise ValueError(
            f"{name} is a composed one-sided group delay in seconds and must be >= 0, got {value}"
        )
    return value


def physical_lag_seconds(
    lag_step: LagValue,
    *,
    source_reference_s: float,
    target_reference_s: float = 0.0,
    horizon_element: LagValue = 0,
    seconds_per_step: float = SECONDS_PER_STEP,
    mechanical_shift_seconds: float = MECHANICAL_SHIFT_SECONDS,
) -> LagValue:
    r"""The lead time between the physical epochs two aligned coefficients summarise, in seconds.

    $$\tau^{\mathrm{phys}}_{\ell,h} = \Delta\,(\ell + 1 + h)
      + \bigl(\tau^{u}_{\mathrm{ref}} - \tau^{y}_{\mathrm{ref}}\bigr) - \tau_{\mathrm{pre}}.$$

    Read off the two coefficients' stamps: the forecast element for horizon step $h$ of anchor $t$
    summarises target content centred at $\Delta(t + 1 + h) - \tau^{y}_{\mathrm{ref}}$, the source
    at lag $\ell$ summarises content centred at $\Delta(t - \ell) - \tau^{u}_{\mathrm{ref}}
    + \tau_{\mathrm{pre}}$, and the difference is the expression above -- the anchor $t$ cancels,
    which is why one number describes the whole page.

    **Only valid on aligned streams.** The identity is written with one reference per stream, so
    it presumes every channel of that stream has already been shifted onto it. Unaligned, the
    bias is $\tau^{u}_c - \tau^{y}_{c'}$ -- a channel-*pair*-indexed quantity spanning over
    $1100$ s on the shipped bank -- and no per-stream constant stands in for it. This function is
    given the references rather than resolving them for exactly that reason: what it may be
    handed is a decision about the run, not a property of this module.

    **A nonzero target reference changes what the answer is about.** With
    $\tau^{y}_{\mathrm{ref}} = 0$ -- the raw-target case, where the target passes through no filter
    at all -- the result is a lead time between the *uterine-activity signal* and the *fetal heart
    rate signal*. With a feature target it is a lead time between two **coefficient epochs**: both
    sides are then band-limited envelopes carrying their own intra-band group-delay dispersion, so
    the number is a lag between what two filters report and not between the signals underneath
    them.

    Args:
        lag_step: Attention lag index $\ell$, a scalar or a whole axis as a tensor.
        source_reference_s: $\tau^{u}_{\mathrm{ref}}$, the common clock the source stream's
            channels were shifted onto, in seconds.
        target_reference_s: $\tau^{y}_{\mathrm{ref}}$, the same for the target stream. Defaults to
            $0$, which is exact rather than conventional in the raw-target cells: a raw sample is
            at the instant it is at.
        horizon_element: $h$, which element of the forecast block the lag is reported against.
            Zero is the first predicted step, $\Delta$ seconds past the anchor.
        seconds_per_step: Seconds per decimated step $\Delta$.
        mechanical_shift_seconds: $\tau_{\mathrm{pre}}$, the sensor delay preprocessing already
            removed from the source trace. **Subtracted** here, exactly as
            :func:`lag_original_sensor_seconds` subtracts it -- the two agree on the sign because
            they undo the same advance. They remain different quantities: there the question is
            where a finding sits in the uncorrected files, here it is how far apart two physical
            epochs are on the corrected timeline the model was trained on.

    Returns:
        The physical lead time in seconds, of the same kind as ``lag_step``.

    Raises:
        ValueError: If either reference is negative.
    """
    source = _validate_reference(source_reference_s, "source_reference_s")
    target = _validate_reference(target_reference_s, "target_reference_s")
    grid = float(seconds_per_step) * (lag_step + 1 + horizon_element)
    return grid + (source - target) - float(mechanical_shift_seconds)
