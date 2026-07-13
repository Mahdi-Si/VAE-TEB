r"""Multi-resolution validity masks for the raw-signal VAE-TEB v4 model.

The stored dataset carries a *decimated* validity signal ``weight`` of length $\tilde T = 330$
($0.25$ Hz). The raw model works at three resolutions and needs a mask at each:

- **Front-end / raw grid** ($L_{\mathrm{raw}} = 5280$): which raw samples are valid. Built by
  nearest-upsampling ``weight`` by the decimation factor $D = 16$ (:func:`frontend_mask`), optionally
  refined by a sentinel/finite check (:func:`raw_validity_mask`, Sprint 3).
- **Low-rate anchor grid** ($T = 300$): whether an anchor's *own present block* is fully valid
  (:func:`low_rate_mask`, Sprint 3).
- **Forecast / KL grids**: which future raw samples and which anchors enter the loss
  (:func:`forecast_mask` / :func:`kl_mask`, Sprint 3).

Under the v4 standardization convention (loader z-scores ``fhr``/``up``; see
``vae-teb-raw-v4-spec-and-sprints.md`` §5.5), the decimated ``weight`` is the **authoritative** gap
signal: it survives normalization, whereas a raw-bpm ``SENTINEL`` (e.g. $0$ bpm) does not. Hence
``mask_mode='weight_only'`` is the default, and :func:`frontend_mask` is the primary front-end mask.

Sprint 2 provides :func:`frontend_mask` (needed by ``SeqVaeRawV4._default_batch_to_inputs``); the full
suite lands in Sprint 3 (S3-T02).
"""
from __future__ import annotations

from typing import Optional

import torch

from model.vae_teb_prediction.model.model_raw.geometry import D, GEOMETRY, RAW_LEN, RawGeometry
from model.vae_teb_prediction.model.model_raw.raw_targets import build_future_index

_MASK_MODE_CHOICES = ("weight_only", "sentinel_refine")


def frontend_mask(
    weight: torch.Tensor,
    raw_len: int = RAW_LEN,
    decimation: int = D,
) -> torch.Tensor:
    r"""Nearest-upsample the decimated ``weight`` to a raw-resolution validity mask.

    Each decimated step $k$ gates the $D$ raw samples $[Dk, Dk + D)$, so the raw mask is a
    length-$D$ nearest-neighbour expansion of $(\texttt{weight} > 0)$:

    $$
    m^{\mathrm{raw}}[n] = \mathbb{1}\!\left[\texttt{weight}\big[\lfloor n / D \rfloor\big] > 0\right],
    \qquad n \in [0, L_{\mathrm{raw}}).
    $$

    Args:
        weight: Decimated validity $(B, \tilde T)$ with $\tilde T = L_{\mathrm{raw}} / D$. Values
            $> 0$ are valid.
        raw_len: Raw samples per segment $L_{\mathrm{raw}}$.
        decimation: Front-end total stride / raw substeps per low-rate step $D$.

    Returns:
        A raw-resolution validity mask $(B, L_{\mathrm{raw}})$, dtype matching ``weight``, values in
        $\{0, 1\}$.

    Raises:
        ValueError: If ``weight``'s length is not $L_{\mathrm{raw}} / D$ (fails loudly on a trimmed
            loader, which would silently misalign every downstream index).
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D (B, T_tilde), got shape {tuple(weight.shape)}")
    expected = raw_len // decimation
    if weight.size(1) != expected:
        raise ValueError(
            f"weight length {weight.size(1)} != raw_len/decimation = {expected} "
            f"(raw_len={raw_len}, decimation={decimation}); the loader must be untrimmed "
            "(trim_minutes: null) for the raw model."
        )
    valid = (weight > 0).to(weight.dtype)
    return valid.repeat_interleave(decimation, dim=1)


def raw_validity_mask(
    fhr_raw: torch.Tensor,
    weight: torch.Tensor,
    *,
    geometry: RawGeometry = GEOMETRY,
    mask_mode: str = "weight_only",
    sentinel: Optional[float] = None,
) -> torch.Tensor:
    r"""Full raw-resolution validity mask $m^{\mathrm{raw}} \in \{0,1\}^{B \times L_{\mathrm{raw}}}$.

    Always starts from the authoritative decimated ``weight`` (:func:`frontend_mask`). In
    ``sentinel_refine`` mode it is additionally intersected with a finite / non-sentinel check on the
    raw signal:

    $$
    m^{\mathrm{raw}}[n] = \big(\texttt{weight}[\lfloor n/D \rfloor] > 0\big)
    \wedge \operatorname{isfinite}(x^y[n]) \wedge (x^y[n] \neq \text{SENTINEL}).
    $$

    Under the v4 standardization convention the loader normalizes ``fhr``, so a raw-bpm ``SENTINEL``
    does not survive and ``weight_only`` (the default) is authoritative.

    Args:
        fhr_raw: Raw FHR signal $(B, L_{\mathrm{raw}})$ (only used in ``sentinel_refine``).
        weight: Decimated validity $(B, \tilde T)$.
        geometry: The raw/low-rate geometry.
        mask_mode: ``'weight_only'`` (default) or ``'sentinel_refine'``.
        sentinel: The raw sentinel value; required (non-``None``) for ``sentinel_refine``.

    Returns:
        A raw validity mask $(B, L_{\mathrm{raw}})$, dtype matching ``weight``.

    Raises:
        ValueError: On an unknown ``mask_mode``, or ``sentinel_refine`` with ``sentinel is None``.
    """
    if mask_mode not in _MASK_MODE_CHOICES:
        raise ValueError(f"mask_mode must be one of {_MASK_MODE_CHOICES}, got {mask_mode!r}")
    m_raw = frontend_mask(weight, geometry.raw_len, geometry.decimation)
    if mask_mode == "sentinel_refine":
        if sentinel is None:
            raise ValueError("mask_mode='sentinel_refine' requires a non-None sentinel")
        refine = torch.isfinite(fhr_raw) & (fhr_raw != sentinel)
        m_raw = m_raw * refine.to(m_raw.dtype)
    return m_raw


def low_rate_mask(m_raw: torch.Tensor, geometry: RawGeometry = GEOMETRY) -> torch.Tensor:
    r"""Low-rate anchor-validity mask $m^{\mathrm{low}}_t = \min_{r} m^{\mathrm{raw}}[D(t+\mathrm{CROP}) + r]$.

    An anchor is valid only if **every** raw sample of its own present block (raw
    $[D(t+\mathrm{CROP}),\, D(t+\mathrm{CROP}) + R)$, i.e. decimated ``weight[t+CROP]``) is valid.

    Args:
        m_raw: Raw validity mask $(B, L_{\mathrm{raw}})$.
        geometry: The raw/low-rate geometry.

    Returns:
        The low-rate anchor-validity mask $(B, T)$.
    """
    d, t_n, r_n = geometry.decimation, geometry.t, geometry.r
    t = torch.arange(t_n, dtype=torch.long, device=m_raw.device)
    r = torch.arange(r_n, dtype=torch.long, device=m_raw.device)
    idx = d * (t + geometry.crop)[:, None] + r[None, :]        # (T, R) = own_present_start(t) + r
    vals = m_raw.index_select(1, idx.reshape(-1)).reshape(m_raw.size(0), t_n, r_n)
    return vals.min(dim=-1).values                              # (B, T)


def forecast_mask(
    m_raw: torch.Tensor,
    m_low: torch.Tensor,
    geometry: RawGeometry = GEOMETRY,
    *,
    future_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Per-sample forecast mask $m_{t,\tau,r} = \mathbb{1}[t \ge w]\, m^{\mathrm{low}}_t\, m^{\mathrm{raw}}[\mathrm{future\_block\_start}(t) + D\tau + r]$.

    Gates each future raw sample on (a) the warm-up prefix, (b) the anchor's own validity, and (c) the
    future sample's own validity -- so a gap at either the anchor or any forecast step contributes zero
    to the loss.

    Args:
        m_raw: Raw validity mask $(B, L_{\mathrm{raw}})$.
        m_low: Low-rate anchor-validity mask $(B, T)$ (from :func:`low_rate_mask`).
        geometry: The raw/low-rate geometry.
        future_index: Optional precomputed $(T_{\mathrm{valid}}, H, R)$ index grid (e.g. a cached
            model buffer) to avoid rebuilding it on the training hot path; built from ``geometry``
            when ``None``. Must match the index :func:`raw_targets.build_future_target` gathers.

    Returns:
        The forecast mask $(B, T_{\mathrm{valid}}, H, R)$.
    """
    idx = build_future_index(geometry) if future_index is None else future_index
    idx = idx.to(m_raw.device)                                 # (T_valid, H, R)
    b = m_raw.size(0)
    fut = m_raw.index_select(1, idx.reshape(-1)).reshape(
        b, geometry.t_valid, geometry.horizon, geometry.r
    )
    t = torch.arange(geometry.t_valid, device=m_raw.device)
    warm = (t >= geometry.warmup).to(m_raw.dtype)              # (T_valid,)
    anchor = m_low[:, : geometry.t_valid]                       # (B, T_valid)
    return warm[None, :, None, None] * anchor[:, :, None, None] * fut


def kl_mask(m_low: torch.Tensor, geometry: RawGeometry = GEOMETRY) -> torch.Tensor:
    r"""KL anchor mask $m^{\mathrm{KL}}_t = \mathbb{1}[t \ge w]\, m^{\mathrm{low}}_t$ over all $T$ anchors.

    This is the per-anchor data-validity weight handed to the inherited ``_kld_loss`` (which applies
    the anchor **support** $[w, T-H)$ itself); the warm-up factor here is redundant-but-explicit.

    Args:
        m_low: Low-rate anchor-validity mask $(B, T)$.
        geometry: The raw/low-rate geometry.

    Returns:
        The KL anchor mask $(B, T)$.
    """
    t = torch.arange(geometry.t, device=m_low.device)
    warm = (t >= geometry.warmup).to(m_low.dtype)              # (T,)
    return warm[None, :] * m_low
