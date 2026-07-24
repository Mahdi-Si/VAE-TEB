r"""Raw future-FHR target extraction on the trimmed grid.

For each anchor $t \in [0, T_{\mathrm{valid}})$ the forecast target is the $(H, R)$ grid of raw
samples

$$
X^+_{t,\tau,r} = x\big[\mathrm{future\_block\_start}(t) + D\,\tau + r\big]
             = x\big[D\,(t + 1) + D\,\tau + r\big],
\qquad \tau \in [0, H),\; r \in [0, R),
$$

starting one sample past the anchor's causal endpoint (see ``geometry.py`` for why the trimmed
grid has no crop offset). The warm-up prefix $[0, w)$ is *not* trimmed here -- it is masked out
downstream -- so the target keeps a fixed $T_{\mathrm{valid}}$ anchor axis.

Pure torch: no config, no batch schema. Tensors go in as arguments.
"""
from __future__ import annotations

from typing import Optional

import torch

from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry


def build_future_index(geometry: TrimmedRawGeometry) -> torch.Tensor:
    r"""Build the $(T_{\mathrm{valid}}, H, R)$ integer grid of raw future-target indices.

    Element $[t, \tau, r] = D\,(t + 1) + D\,\tau + r$.

    Args:
        geometry: The trimmed-grid geometry.

    Returns:
        A ``torch.long`` tensor of shape $(T_{\mathrm{valid}}, H, R)$, every entry in
        $[D, L_{\mathrm{raw}})$.
    """
    d = geometry.decimation
    t = torch.arange(geometry.t_valid, dtype=torch.long)
    tau = torch.arange(geometry.horizon, dtype=torch.long)
    r = torch.arange(geometry.r, dtype=torch.long)
    start = d * (t + 1)  # (T_valid,) = future_block_start(t)
    return start[:, None, None] + d * tau[None, :, None] + r[None, None, :]


def build_future_target(
    fhr_raw: torch.Tensor,
    geometry: TrimmedRawGeometry,
    *,
    future_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Gather the future raw target $X^+ \in \mathbb{R}^{B \times T_{\mathrm{valid}} \times H \times R}$.

    Args:
        fhr_raw: Raw target signal $(B, L_{\mathrm{raw}})$, loader-normalized.
        geometry: The trimmed-grid geometry; must match the raw length of ``fhr_raw``.
        future_index: Optional precomputed $(T_{\mathrm{valid}}, H, R)$ index grid (e.g. a
            cached buffer) to avoid rebuilding it on the training hot path; built from
            ``geometry`` when ``None``.

    Returns:
        The future raw target of shape $(B, T_{\mathrm{valid}}, H, R)$.

    Raises:
        ValueError: If ``fhr_raw`` is not 2-D, or its length does not match
            ``geometry.raw_len`` -- which is what a loader running at the wrong
            ``trim_minutes`` produces.
    """
    if fhr_raw.dim() != 2:
        raise ValueError(f"fhr_raw must be 2-D (B, L_raw), got shape {tuple(fhr_raw.shape)}")
    if fhr_raw.size(1) != geometry.raw_len:
        raise ValueError(
            f"fhr_raw length {fhr_raw.size(1)} != geometry.raw_len {geometry.raw_len}; "
            "this geometry assumes the loader's symmetric trim has already been applied "
            "(trim_minutes: 1.0 -> 4800 raw samples), so a mismatch means the loader ran "
            "at a different trim_minutes than the geometry was built for"
        )
    idx = build_future_index(geometry) if future_index is None else future_index
    idx = idx.to(fhr_raw.device)
    gathered = fhr_raw.index_select(1, idx.reshape(-1))
    return gathered.reshape(
        fhr_raw.size(0), geometry.t_valid, geometry.horizon, geometry.r
    )
