r"""Forecast and KL validity masks on the decimated grid.

The loader's decimated ``weight`` is the authoritative gap signal: gaps in the raw signal are
stored as $0$ bpm, which after z-scoring is roughly $-11\sigma$ and is not a detectable
sentinel. Because the trimmed grid has no crop and the model's inputs are features rather than
raw samples, validity is constant within each decimated step: future raw sample $(t, \tau, r)$
lies in decimated step $t + 1 + \tau$ for *every* $r$, so the forecast mask lives on the
decimated $(B, T_{\mathrm{valid}}, H)$ grid and is broadcast over $r$ -- there is no
raw-resolution mask stack here because nothing consumes one.

The KL mask is *derived from* the forecast mask rather than rebuilt from ``weight``, so the two
supports are the same anchor set by construction. Charging the KL on an anchor the
reconstruction does not score leaves the posterior with nothing pulling it off the prior.

**Validity threshold.** A step is valid iff ``weight >= 1.0``. Evidence for the choice: the
shard writer's per-step weight is binary by construction
(``mimo/MIMO_Sequence_Trainer/mimo/sequence/mimo_sequence.py::calc_sample_weights`` computes
``1.0 * (1 - np.all(block_target - padded_target_val == 0, axis=2))``, values in $\{0, 1\}$, and
``hdf5_dataset/new_pipeline/create_new_pipeline.py`` stores it unmodified), so ``>= 1.0`` and
``> 0`` agree on every shard that writer produced. ``>= 1.0`` is still the form used, because if
a fractional weight ever appears it means a *partially* valid step -- and a partially valid step
still contains raw samples at $\approx -11\sigma$, which would dominate a summed $480$-sample
NLL. Partial steps must be excluded, not admitted.

Pure torch: no config, no batch schema.
"""
from __future__ import annotations

from typing import Tuple

import torch

from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry

#: A decimated step counts as valid only when fully valid; see the module docstring.
VALID_THRESHOLD = 1.0


def _validate_weight(weight: torch.Tensor, geometry: TrimmedRawGeometry) -> torch.Tensor:
    """Check the weight tensor against the geometry and binarize it.

    Args:
        weight: Decimated validity signal $(B, T)$.
        geometry: The trimmed-grid geometry.

    Returns:
        The float validity $(B, T)$, $1.0$ where ``weight >= VALID_THRESHOLD``.

    Raises:
        ValueError: If ``weight`` is not 2-D or its length is not $T$ -- which is what a
            loader running at the wrong ``trim_minutes`` produces.
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D (B, T), got shape {tuple(weight.shape)}")
    if weight.size(1) != geometry.t:
        raise ValueError(
            f"weight length {weight.size(1)} != geometry.t {geometry.t}; this geometry "
            "assumes the trimmed loader (trim_minutes: 1.0 -> T = 300 decimated steps), "
            "so a mismatch means the loader ran at a different trim_minutes"
        )
    return (weight >= VALID_THRESHOLD).to(weight.dtype)


def forecast_mask(
    weight: torch.Tensor,
    geometry: TrimmedRawGeometry,
    *,
    coverage_floor: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Per-anchor forecast validity on the decimated grid, plus each anchor's coverage.

    The mask gates each forecast step on three factors,

    $$
    m_{t,\tau} = \mathbb{1}[t \ge w]\; v_t\; v_{t+1+\tau},
    $$

    where $v$ is the thresholded ``weight``: the warm-up prefix, the anchor's own validity, and
    the forecast step's validity. Broadcasting over $r$ reproduces the per-raw-sample mask
    exactly, because every raw sample of decimated step $k$ shares ``weight[k]``.

    ``coverage_frac`` is the fraction of the anchor's $H$ forecast steps that are valid. An
    anchor below ``coverage_floor`` is zeroed *entirely*: a half-masked anchor's summed NLL is
    computed over half a window, and the base-minus-full gap read off it is spuriously small.
    ``coverage_floor = 0.0`` disables the floor and reproduces the per-step behaviour exactly.

    Args:
        weight: Decimated validity signal $(B, T)$.
        geometry: The trimmed-grid geometry.
        coverage_floor: Minimum valid fraction of the forecast window for an anchor to
            contribute at all.

    Returns:
        ``(mask, coverage_frac)``: the forecast mask $(B, T_{\mathrm{valid}}, H)$ and the
        per-anchor coverage $(B, T_{\mathrm{valid}})$. Coverage is the raw future-window
        fraction, reported before the warm-up, anchor-validity and floor factors are applied.

    Raises:
        ValueError: If ``weight`` does not match the geometry (see :func:`_validate_weight`).
    """
    valid = _validate_weight(weight, geometry)
    t_valid, horizon = geometry.t_valid, geometry.horizon

    # Future-step validity: window tau of anchor t reads decimated step t + 1 + tau.
    future = valid[:, 1:].unfold(dimension=1, size=horizon, step=1)  # (B, T_valid, H)
    coverage_frac = future.mean(dim=-1)  # (B, T_valid)

    warm = (
        torch.arange(t_valid, device=weight.device) >= geometry.warmup
    ).to(valid.dtype)  # (T_valid,)
    anchor = valid[:, :t_valid]  # (B, T_valid)
    keep = (coverage_frac >= float(coverage_floor)).to(valid.dtype)  # (B, T_valid)

    mask = warm[None, :, None] * (anchor * keep)[:, :, None] * future
    return mask, coverage_frac


def contributing_anchors(forecast: torch.Tensor) -> torch.Tensor:
    r"""Which anchors carry a reconstruction term at all.

    $$c_t = \mathbb{1}\!\left[\max_\tau m_{t,\tau} > 0\right].$$

    One definition, used by both the loss's per-anchor denominator and :func:`kl_mask`, so the
    reconstruction support and the KL support cannot be derived by two rules that drift apart.

    Args:
        forecast: The forecast mask $(B, T_{\mathrm{valid}}, H)$ from :func:`forecast_mask`.

    Returns:
        A $0/1$ indicator $(B, T_{\mathrm{valid}})$ in ``forecast``'s dtype.
    """
    return (forecast.amax(dim=-1) > 0).to(forecast.dtype)


def kl_mask(forecast: torch.Tensor, geometry: TrimmedRawGeometry) -> torch.Tensor:
    r"""Per-anchor KL support: exactly the anchors the reconstruction scores.

    $$m^{\mathrm{KL}}_t = c_t \;\text{for}\; t < T - H, \quad 0 \;\text{otherwise},$$

    with $c_t$ from :func:`contributing_anchors`. Taking the *forecast mask* rather than
    ``weight`` is what makes the two supports identical by construction: every factor the
    forecast mask applies -- the warm-up prefix, the anchor's own validity, each forecast step's
    validity and the coverage floor -- is inherited here, rather than being restated by a second
    expression that a later edit could change on one side only.

    Deriving the support instead of restating it as $\mathbb{1}[w \le t < T-H]\,v_t$ matters
    beyond tidiness. An anchor dropped by the coverage floor, or one whose forecast window lies
    wholly in a gap, has no reconstruction term pulling the posterior off the prior; charging
    $\beta \cdot \mathrm{KL}$ on it regularises it onto the prior for free. Those anchors cluster
    immediately before every signal-loss gap, so the artifact is a *localised* KL suppression
    that reads as coupling fading right where the signal degrades -- the same failure the tail
    $H$ exclusion exists to prevent, in the place it is hardest to recognise.

    Args:
        forecast: The forecast mask $(B, T_{\mathrm{valid}}, H)$ from :func:`forecast_mask`.
        geometry: The trimmed-grid geometry.

    Returns:
        The KL anchor mask $(B, T)$, zero outside $[w, T - H)$ and zero on any anchor the
        reconstruction does not score.

    Raises:
        ValueError: If ``forecast`` is not the $(B, T_{\mathrm{valid}}, H)$ mask this geometry
            describes -- which is what passing ``weight`` by mistake produces.
    """
    expected = (geometry.t_valid, geometry.horizon)
    if forecast.dim() != 3 or tuple(forecast.shape[1:]) != expected:
        raise ValueError(
            f"forecast mask must be (B, T_valid, H) = (B, {expected[0]}, {expected[1]}) for "
            f"this geometry, got shape {tuple(forecast.shape)}; kl_mask takes the forecast "
            "mask from forecast_mask(), not the raw weight, so that the KL support is the "
            "reconstruction's support by construction"
        )

    contributing = contributing_anchors(forecast)  # (B, T_valid), zero for t < warmup already
    tail = forecast.new_zeros((forecast.shape[0], geometry.t - geometry.t_valid))
    return torch.cat((contributing, tail), dim=1)
