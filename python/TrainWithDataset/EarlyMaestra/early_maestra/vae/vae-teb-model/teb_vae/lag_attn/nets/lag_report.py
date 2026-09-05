r"""Turning an attention lag index into the seconds figures that may be reported.

**The stored timeline is canonical.** The dataset builder shifts the uterine-pressure channel
when it creates the HDF5 shards. That shift is part of how the stored signals *are*: every
downstream consumer -- this module, every lag figure, every evaluation record, every simulation --
treats the stored UP and FHR traces as if they had been recorded exactly as they sit on disk. There
is no "sensor timeline", no "mechanical correction" and no "acquisition shift" to add back, subtract,
budget or interpret anywhere downstream. Earlier revisions of this module carried a
``MECHANICAL_SHIFT_SECONDS`` constant and a ``lag_original_sensor_seconds`` helper that undid the
builder's shift; both were removed on purpose, and neither must return under another name.

An attention peak at lag index $\ell$ is still not by itself a physiological delay, for two
reasons that are easy to conflate.

**The causal input delay.** When a channel is read at $t - \delta$ rather than at $t$ (the guard
that bounds how far an input feature reads into its own future), the source memory the attention
queries is itself $\delta$ steps stale. A peak at lag $\ell$ then refers to source content
$\ell + \delta$ steps back. With no guard configured, $\delta = 0$ and the correction vanishes.
With $\Delta = 4$ s per decimated step:

$$
\tau_{\mathrm{compensated}} = \Delta\,(\ell + \delta).
$$

**The group delay, when both streams are on one clock.** The figure above is a lag between
*stored coefficients*. A one-sided channel's content is itself stale by that channel's own composed
group delay $\tau_c$, so a lag between two of them is biased by $\tau^{u}_c - \tau^{y}_{c'}$ --
indexed by a channel *pair*, and therefore not a number a pooled state can carry. Once each stream
is shifted onto a common reference the pair index collapses to one constant, and the lead time
between the physical epochs the two coefficients summarise becomes reportable:

$$
\tau^{\mathrm{phys}}_{\ell,h} = \Delta\,(\ell + 1 + h)
  + \kappa\bigl(\tau^{u}_{\mathrm{ref}} - \tau^{y}_{\mathrm{ref}}\bigr).
$$

This is a *second* quantity, not a refinement of the first, and it is built from a different
input: the two references are delays in **seconds**, resolved from the shards, and none of them is
the model's ``source_delay_steps``. That scalar is the largest shift the gate applies, attained by
the channel *furthest* from the reference, and using it here would be wrong by the whole reference.
$\kappa$ is an explicitly approximate content-delay convention (the one-sided bank's energy-centroid
factor); the identity is a coefficient-content lead, not an exact physiological timestamp.

A standalone module with no model import, so a figure and an evaluation can share the arithmetic
without either reaching into the network.
"""
from __future__ import annotations

from typing import Union

import torch

#: Seconds per decimated step: the features are decimated by $16$ from a $4$ Hz raw grid.
SECONDS_PER_STEP = 4.0

#: Axis label for a lag axis drawn in seconds with the causal input delay $\delta$ added back. A
#: constant rather than a string in each figure, so a plot and the number beside it cannot disagree
#: about which quantity is shown. It names the input-delay compensation only: the stored timeline
#: is canonical and no other correction is ever applied to a lag axis.
COMPENSATED_LAG_AXIS_LABEL = "lag (s, input-delay compensated)"

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
    r"""The lag on the stored timeline with the input delay added back: $\Delta\,(\ell + \delta)$ s.

    This is the quantity to report as *the* lag between stored positions. The stored timeline is
    canonical, so nothing else is added or subtracted.

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
    realised_delay_factor: float = 1.0,
) -> LagValue:
    r"""The lead time between the content epochs two aligned coefficients summarise, in seconds.

    $$\tau^{\mathrm{phys}}_{\ell,h} = \Delta\,(\ell + 1 + h)
      + \kappa\bigl(\tau^{u}_{\mathrm{ref}} - \tau^{y}_{\mathrm{ref}}\bigr).$$

    Read off the two coefficients' stamps on the canonical stored timeline: the forecast element
    for horizon step $h$ of anchor $t$ summarises target content centred at
    $\Delta(t + 1 + h) - \kappa\tau^{y}_{\mathrm{ref}}$, the source at lag $\ell$ summarises content
    centred at $\Delta(t - \ell) - \kappa\tau^{u}_{\mathrm{ref}}$, and the difference is the
    expression above -- the anchor $t$ cancels, which is why one number describes the whole page.
    $\kappa$ is the fraction of a reference's reported group delay its content is *taken* to sit
    at -- the energy centroid of the one-sided bank's impulse response, $1 - 1/(2\gamma) = 0.875$
    at gammatone order $4$, the same factor the alignment shift is computed with. It is a
    convention rather than a measured universal (a narrowband modulation realises close to the
    full group delay), so the result is an approximate content lead, never an exact physiological
    timestamp. It is a property of the bank rather than of this module, so it is passed in;
    ``1.0`` reads the references as realised delays already.

    **Only valid on aligned streams.** The identity is written with one reference per stream, so
    it presumes every channel of that stream has already been shifted onto it. Unaligned, the
    bias is $\tau^{u}_c - \tau^{y}_{c'}$ -- a channel-*pair*-indexed quantity spanning over
    $1100$ s on the shipped bank -- and no per-stream constant stands in for it. This function is
    given the references rather than resolving them for exactly that reason: what it may be
    handed is a decision about the run, not a property of this module.

    **The target reference must be the clock of the *scored* target**, which under a
    ``'physical'`` forecast clock is the fastest kept channel's $\tau_{\min}$, not the target
    encoder's input reference. Passing the input reference there is wrong by the whole difference
    between the two clocks.

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
        target_reference_s: $\tau^{y}_{\mathrm{ref}}$, the clock of the scored target stream.
            Defaults to $0$, which is exact rather than conventional in the raw-target cells: a
            raw sample is at the instant it is at.
        horizon_element: $h$, which element of the forecast block the lag is reported against.
            Zero is the first predicted step, $\Delta$ seconds past the anchor.
        seconds_per_step: Seconds per decimated step $\Delta$.
        realised_delay_factor: $\kappa$, the fraction of each reference its content is taken to
            sit at. The one-sided cells pass their bank's factor; ``1.0`` treats both references
            as realised delays already.

    Returns:
        The content lead time in seconds, of the same kind as ``lag_step``.

    Raises:
        ValueError: If either reference is negative.
    """
    source = _validate_reference(source_reference_s, "source_reference_s")
    target = _validate_reference(target_reference_s, "target_reference_s")
    grid = float(seconds_per_step) * (lag_step + 1 + horizon_element)
    return grid + float(realised_delay_factor) * (source - target)
