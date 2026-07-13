r"""Crop-aligned raw future-FHR target extraction for the raw-signal VAE-TEB v4 model.

For each trained low-rate anchor $t \in [0, T_{\mathrm{valid}})$ (with $T_{\mathrm{valid}} = T - H$), the
$2$-minute future raw FHR block is the $(H, R)$ grid of raw samples

$$
X^+_{t,\tau,r} = x^y\big[\mathrm{future\_block\_start}(t) + D\,\tau + r\big],
\qquad \tau \in [0, H),\; r \in [0, R),
$$

where $\mathrm{future\_block\_start}(t) = n_{\mathrm{raw}}(t) + 1 = D\,(t + \mathrm{CROP} + 1)$ -- the raw
sample **one past** the cropped anchor's causal endpoint (see ``geometry.py``). This module builds the
shared integer index grid (:func:`build_future_index`, reused by the mask module) and gathers it
(:func:`build_future_target`). The warm-up prefix $[0, w)$ is **not** trimmed here -- it is masked out
downstream by the forecast/KL masks -- so the target keeps a fixed $T_{\mathrm{valid}}$ anchor axis.
"""
from __future__ import annotations

from typing import Optional

import torch

from model.vae_teb_prediction.model.model_raw.geometry import GEOMETRY, RawGeometry


def build_future_index(geometry: RawGeometry = GEOMETRY) -> torch.Tensor:
    r"""Build the $(T_{\mathrm{valid}}, H, R)$ integer grid of raw future-target indices.

    Element $[t, \tau, r] = \mathrm{future\_block\_start}(t) + D\,\tau + r$ with
    $\mathrm{future\_block\_start}(t) = D\,(t + \mathrm{CROP} + 1)$.

    Args:
        geometry: The raw/low-rate geometry (defaults to the production geometry).

    Returns:
        A ``torch.long`` tensor of shape $(T_{\mathrm{valid}}, H, R)$ of raw sample indices, every
        entry in $[0, L_{\mathrm{raw}})$.
    """
    d = geometry.decimation
    t = torch.arange(geometry.t_valid, dtype=torch.long)      # (T_valid,)
    tau = torch.arange(geometry.horizon, dtype=torch.long)    # (H,)
    r = torch.arange(geometry.r, dtype=torch.long)            # (R,)
    start = d * (t + geometry.crop + 1)                        # (T_valid,) = future_block_start(t)
    return start[:, None, None] + d * tau[None, :, None] + r[None, None, :]  # (T_valid, H, R)


def build_future_target(
    fhr_raw: torch.Tensor,
    geometry: RawGeometry = GEOMETRY,
    *,
    future_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Gather the crop-aligned future raw FHR block $X^+ \in \mathbb{R}^{B \times T_{\mathrm{valid}} \times H \times R}$.

    Args:
        fhr_raw: Raw FHR signal $(B, L_{\mathrm{raw}})$ (loader-normalized).
        geometry: The raw/low-rate geometry (defaults to the production geometry). Must match the raw
            length of ``fhr_raw``.
        future_index: Optional precomputed $(T_{\mathrm{valid}}, H, R)$ index grid (e.g. a cached
            model buffer) to avoid rebuilding it on the training hot path; built from ``geometry``
            when ``None``.

    Returns:
        The future raw FHR target $X^+$ of shape $(B, T_{\mathrm{valid}}, H, R)$.

    Raises:
        ValueError: If ``fhr_raw``'s length does not match ``geometry.raw_len``.
    """
    if fhr_raw.dim() != 2:
        raise ValueError(f"fhr_raw must be 2-D (B, L_raw), got shape {tuple(fhr_raw.shape)}")
    if fhr_raw.size(1) != geometry.raw_len:
        raise ValueError(
            f"fhr_raw length {fhr_raw.size(1)} != geometry.raw_len {geometry.raw_len}"
        )
    idx = build_future_index(geometry) if future_index is None else future_index
    idx = idx.to(fhr_raw.device)                           # (T_valid, H, R)
    gathered = fhr_raw.index_select(1, idx.reshape(-1))     # (B, T_valid*H*R)
    return gathered.reshape(
        fhr_raw.size(0), geometry.t_valid, geometry.horizon, geometry.r
    )
