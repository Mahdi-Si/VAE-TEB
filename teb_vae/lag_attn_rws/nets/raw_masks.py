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

**Which anchors are scored is an argument, not a constant here.** All three functions take an
optional $(B, A)$ anchor index with a boolean validity companion; omitting it means the dense
range $[0, T_{\mathrm{valid}})$, which is what a model decoding every anchor asks for. The
forecast mask and the contributing indicator then carry that axis, and :func:`kl_mask` -- and only
:func:`kl_mask` -- scatters it back to the $(B, T)$ grid the latent tensors live on.


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

from typing import Optional, Tuple

import torch

from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry

#: A decimated step counts as valid only when fully valid; see the module docstring.
VALID_THRESHOLD = 1.0


def _validate_anchors(
    anchors: torch.Tensor,
    anchor_valid: Optional[torch.Tensor],
    geometry: TrimmedRawGeometry,
    batch: int,
) -> None:
    r"""Check an explicit anchor set against the geometry it indexes into.

    An anchor set that a caller supplies is an index, and every way of getting an index wrong here
    is silent. Out of range means $\ge T_{\mathrm{valid}}$, **not** $\ge T$: the tail $H$ anchors
    have no fully observed forecast window, and a scatter that reached them would write into the
    region :func:`kl_mask` otherwise guarantees is zero. A repeated index among the *valid* entries
    is refused for a sharper reason -- the reconstruction would gather and score that anchor's
    target block twice while the KL support, which is a set, would count it once, so the two
    denominators would diverge and $\beta$ would quietly stop meaning what it means everywhere
    else. Repeats among invalid entries are the padding convention and are fine.

    Args:
        anchors: Anchor indices $(B, A)$, integer.
        anchor_valid: Which entries are real $(B, A)$, or ``None`` when every entry is.
        geometry: The trimmed-grid geometry.
        batch: The batch size the anchors must be positional against.

    Raises:
        ValueError: If the shape, dtype, range or uniqueness is wrong, naming the offending value.
    """
    if anchors.dim() != 2:
        raise ValueError(f"anchors must be 2-D (B, A), got shape {tuple(anchors.shape)}")
    if anchors.size(0) != batch:
        raise ValueError(
            f"anchors has batch {anchors.size(0)} but the weight has {batch}; the anchor set is "
            f"per sample, so a mismatch would score one sample's targets at another's anchors"
        )
    if torch.is_floating_point(anchors) or anchors.is_complex():
        raise ValueError(
            f"anchors must be an integer tensor (it indexes the anchor axis), got dtype "
            f"{anchors.dtype}"
        )

    valid = (
        torch.ones_like(anchors, dtype=torch.bool) if anchor_valid is None else anchor_valid
    )
    if tuple(valid.shape) != tuple(anchors.shape):
        raise ValueError(
            f"anchor_valid has shape {tuple(valid.shape)} but anchors has "
            f"{tuple(anchors.shape)}; the two are positional against each other"
        )

    outside = (anchors < 0) | (anchors >= geometry.t_valid)
    if bool(outside.any()):
        offending = int(anchors[outside][0])
        raise ValueError(
            f"anchor {offending} is outside [0, T_valid) = [0, {geometry.t_valid}); the tail "
            f"{geometry.horizon} anchors have no fully observed forecast window and the KL "
            f"support is guaranteed zero there"
        )

    # Duplicates among the valid entries only, per row. Sorting is cheap at these widths and is
    # what makes "the same anchor twice" detectable without materialising a per-row set.
    masked = torch.where(valid, anchors, torch.full_like(anchors, -1))
    ordered, _ = masked.sort(dim=1)
    repeated = (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)
    if bool(repeated.any()):
        offending = int(ordered[:, 1:][repeated][0])
        raise ValueError(
            f"anchor {offending} appears twice among the valid entries of one row; its target "
            f"block would be scored twice by the reconstruction and once by the KL, so the two "
            f"per-anchor denominators would disagree"
        )


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
    anchors: Optional[torch.Tensor] = None,
    anchor_valid: Optional[torch.Tensor] = None,
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

    **The anchor axis is an argument.** With ``anchors = None`` it is every anchor in
    $[0, T_{\mathrm{valid}})$ -- the dense range a model decodes when it decodes all of them, and
    deliberately *not* $[w, T_{\mathrm{valid}})$, because the warm-up is a factor of the mask
    rather than a restriction of the axis. A supplied index gathers instead, and the returned
    tensors carry that axis: $A$ where the dense form carries $T_{\mathrm{valid}}$. The warm-up
    factor still applies to a supplied anchor, so an anchor below $w$ is zeroed rather than
    honoured.

    ``anchor_valid`` marks padding, and it is multiplied *into* the mask rather than merely
    reported. A padded slot repeats a legal anchor index -- that is the convention that keeps
    every index in range -- so without the multiply its row would be fully live, its target block
    would be gathered and scored a second time, and the KL support, which is a set, would count it
    once.

    Args:
        weight: Decimated validity signal $(B, T)$.
        geometry: The trimmed-grid geometry.
        coverage_floor: Minimum valid fraction of the forecast window for an anchor to
            contribute at all.
        anchors: Optional anchor index $(B, A)$, integer, in $[0, T_{\mathrm{valid}})$.
        anchor_valid: Optional $(B, A)$ boolean companion; ``None`` means every entry is real.
            Ignored when ``anchors`` is ``None``, where the axis has no padding by construction.

    Returns:
        ``(mask, coverage_frac)``: the forecast mask $(B, A, H)$ and the per-anchor coverage
        $(B, A)$, with $A = T_{\mathrm{valid}}$ in the dense case. Coverage is the raw
        future-window fraction, reported before the warm-up, anchor-validity, padding and floor
        factors are applied.

    Raises:
        ValueError: If ``weight`` does not match the geometry (see :func:`_validate_weight`), or
            if ``anchors`` is out of range, duplicated among its valid entries, or shaped
            inconsistently with ``anchor_valid`` (see :func:`_validate_anchors`).
    """
    valid = _validate_weight(weight, geometry)
    t_valid, horizon = geometry.t_valid, geometry.horizon

    # Future-step validity: window tau of anchor t reads decimated step t + 1 + tau. The unfold is
    # a strided view over a (B, T) tensor, so it costs nothing even when only A of its rows are
    # then gathered.
    future = valid[:, 1:].unfold(dimension=1, size=horizon, step=1)  # (B, T_valid, H)

    if anchors is None:
        coverage_frac = future.mean(dim=-1)  # (B, T_valid)
        warm = (
            torch.arange(t_valid, device=weight.device) >= geometry.warmup
        ).to(valid.dtype)  # (T_valid,)
        anchor = valid[:, :t_valid]  # (B, T_valid)
        keep = (coverage_frac >= float(coverage_floor)).to(valid.dtype)  # (B, T_valid)

        mask = warm[None, :, None] * (anchor * keep)[:, :, None] * future
        return mask, coverage_frac

    _validate_anchors(anchors, anchor_valid, geometry, weight.size(0))
    index = anchors.to(torch.long)
    rows = torch.arange(weight.size(0), device=weight.device)[:, None]
    future = future[rows, index]  # (B, A, H)
    coverage_frac = future.mean(dim=-1)  # (B, A)

    warm = (index >= geometry.warmup).to(valid.dtype)  # (B, A)
    anchor = valid.gather(1, index)  # (B, A)
    keep = (coverage_frac >= float(coverage_floor)).to(valid.dtype)  # (B, A)
    live = (
        warm
        if anchor_valid is None
        else warm * anchor_valid.to(valid.dtype)
    )

    mask = (live * anchor * keep)[:, :, None] * future
    return mask, coverage_frac


def contributing_anchors(forecast: torch.Tensor) -> torch.Tensor:
    r"""Which anchors carry a reconstruction term at all.

    $$c_t = \mathbb{1}\!\left[\max_\tau m_{t,\tau} > 0\right].$$

    One definition, used by both the loss's per-anchor denominator and :func:`kl_mask`, so the
    reconstruction support and the KL support cannot be derived by two rules that drift apart.

    The rank check is not decoration. This function reduces the **last** axis only, so a mask that
    arrived with an extra axis would return a tensor one rank too high, every denominator built
    from it would be inflated by that axis's length, and nothing downstream would raise -- the one
    failure in this module whose symptom is a wrong number rather than an exception.

    Args:
        forecast: The forecast mask $(B, A, H)$ from :func:`forecast_mask`.

    Returns:
        A $0/1$ indicator $(B, A)$ in ``forecast``'s dtype.

    Raises:
        ValueError: If ``forecast`` is not 3-D, naming the shape received.
    """
    if forecast.dim() != 3:
        raise ValueError(
            f"the forecast mask must be 3-D (B, A, H), got shape {tuple(forecast.shape)}; this "
            f"function reduces the last axis alone, so a mask of another rank would silently "
            f"inflate every per-anchor denominator built from it"
        )
    return (forecast.amax(dim=-1) > 0).to(forecast.dtype)


def kl_mask(
    forecast: torch.Tensor,
    geometry: TrimmedRawGeometry,
    *,
    anchors: Optional[torch.Tensor] = None,
    anchor_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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

    **This is the one place the anchor axis is scattered back to dense.** The return is
    $(B, T)$ whatever the anchor axis was, because the latent tensors it gates are $(B, T, d_z)$
    and are produced at every step regardless of which anchors were decoded. With an explicit
    anchor set the contributing indicator is written into that grid at the anchors' own positions
    -- a scatter rather than a slice, because a gathered anchor set is not a contiguous prefix.
    The scatter reduces by maximum rather than overwriting: a padded slot repeats a legal index,
    and a plain write would let its zero land on top of the real anchor's one.

    Args:
        forecast: The forecast mask $(B, A, H)$ from :func:`forecast_mask`.
        geometry: The trimmed-grid geometry.
        anchors: The anchor index $(B, A)$ the mask was built at, or ``None`` for the dense range.
        anchor_valid: Its $(B, A)$ boolean companion, forwarded only so the same validation runs
            on both masks; the padding is already folded into ``forecast``.

    Returns:
        The KL anchor mask $(B, T)$, zero outside $[w, T - H)$ and zero on any anchor the
        reconstruction does not score.

    Raises:
        ValueError: If ``forecast`` is not the mask this geometry and anchor set describe --
            which is what passing ``weight`` by mistake produces.
    """
    expected_anchors = geometry.t_valid if anchors is None else int(anchors.shape[1])
    expected = (expected_anchors, geometry.horizon)
    if forecast.dim() != 3 or tuple(forecast.shape[1:]) != expected:
        raise ValueError(
            f"forecast mask must be (B, A, H) = (B, {expected[0]}, {expected[1]}) for this "
            f"geometry and anchor set, got shape {tuple(forecast.shape)}; kl_mask takes the "
            "forecast mask from forecast_mask(), not the raw weight, so that the KL support is "
            "the reconstruction's support by construction"
        )

    contributing = contributing_anchors(forecast)  # (B, A), zero for t < warmup already
    if anchors is None:
        tail = forecast.new_zeros((forecast.shape[0], geometry.t - geometry.t_valid))
        return torch.cat((contributing, tail), dim=1)

    _validate_anchors(anchors, anchor_valid, geometry, forecast.shape[0])
    support = forecast.new_zeros((forecast.shape[0], geometry.t))
    return support.scatter_reduce_(
        1, anchors.to(torch.long), contributing, reduce="amax", include_self=True
    )
