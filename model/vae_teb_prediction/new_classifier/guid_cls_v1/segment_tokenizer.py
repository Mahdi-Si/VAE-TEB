"""Per-segment tokenizer for ``guid_cls_v1`` (PRD §6).

Consumes per-timestep VAE outputs from the precompute cache (or live VAE) and
produces a 256-d ``segment_token`` per (B, N) cell, plus a 6-d TE summary.
The token composition is the one mandated by ``classifier_description.md``
§5.5 and PRD §6:

    r_{g,n,t} = [LN(h^y) | LN(mu_post_norm) | LN(Δμ) | log(1+K)]   (177-d)
        → step_proj → (192-d)
        → masked attentive pool over T → s_core (192-d)
    u_TE      = [bar_K, K_max, K_late, m_lag, σ_lag, H_lag]         (6-d)
    c_meta    (10-d, from dataset)
    seg_token = token_proj(cat([s_core, u_TE, c_meta]))              (256-d)

The model **does not** read raw ``epoch``; cross-delivery censoring is applied
inside the dataset and surfaces here only via ``hat_w`` (the classifier-time
mask) and the precomputed ``c_meta`` vector.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-6


class _ResidualMLP(nn.Module):
    """Two-layer pre-norm residual MLP block.

    ``y = x + W_up(GELU(Dropout(W_down(LN(x)))))`` with optional bottleneck
    dimension. Used inside the per-step projection and the token projection.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden_dim)
        self.up = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Args:
            x: Tensor of shape ``(..., dim)``.

        Returns:
            Tensor of the same shape.
        """
        h = self.norm(x)
        h = self.down(h)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.up(h)
        return x + h


class _MaskedAttentivePoolOverT(nn.Module):
    """Additive attentive pool over the time axis of a per-segment tensor.

    Score: ``v^T tanh(W u_t) + log(eps + hat_w[..., t])``. The log-mass bias
    drives padded / low-quality steps to receive ~zero weight without ever
    yielding NaNs (we clamp the bias before adding).
    """

    def __init__(self, dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, u: torch.Tensor, hat_w: torch.Tensor) -> torch.Tensor:
        """Pool over the time axis.

        Args:
            u: Per-step features ``(B, N, T, dim)``.
            hat_w: Per-step classifier-time weights ``(B, N, T)``.

        Returns:
            ``(B, N, dim)`` pooled segment cores.
        """
        logits = self.score(torch.tanh(self.fc(u))).squeeze(-1)  # (B, N, T)
        log_mass = torch.log(torch.clamp(hat_w, min=_EPS))
        scores = logits + log_mass
        # Avoid all-zero weight rows: if every step is masked, fall back to a
        # uniform soft attention so the output is well-defined (the segment
        # itself is then masked out downstream by segment_mask).
        all_zero = (hat_w.sum(dim=-1, keepdim=True) <= _EPS)
        scores = torch.where(
            all_zero.expand_as(scores),
            torch.zeros_like(scores),
            scores,
        )
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, N, T, 1)
        return (weights * u).sum(dim=-2)                    # (B, N, dim)


def _compute_te_summary(
    mean_alpha: torch.Tensor,
    kld_per_t: torch.Tensor,
    hat_w: torch.Tensor,
    *,
    late_window_steps: int = 75,
    eps: float = _EPS,
) -> torch.Tensor:
    """Compute the 6-d TE summary per segment.

    Args:
        mean_alpha: ``(B, N, T, L)`` averaged attention over heads.
        kld_per_t: ``(B, N, T)`` per-step KL.
        hat_w: ``(B, N, T)`` classifier-time weight mask.
        late_window_steps: Number of trailing valid steps used for ``K_late``
            (PRD §5.7 says "last 75 valid steps").
        eps: Numerical guard.

    Returns:
        ``(B, N, 6)`` summary stacked as
        ``[bar_K, K_max, K_late, m_lag, σ_lag, H_lag]``.
    """
    _B, _N, _T, L = mean_alpha.shape
    valid = hat_w > 0.0                          # (B, N, T)
    valid_f = valid.float()

    # bar_K — mean KL over valid steps
    bar_K = (kld_per_t * hat_w).sum(dim=-1) / hat_w.sum(dim=-1).clamp_min(eps)

    # K_max — max KL over valid steps
    masked_K = torch.where(valid, kld_per_t, torch.full_like(kld_per_t, float("-inf")))
    K_max = masked_K.max(dim=-1).values
    K_max = torch.where(torch.isfinite(K_max), K_max, torch.zeros_like(K_max))

    # K_late — mean KL over the last `late_window_steps` valid steps.
    # Use a cumulative count of valid steps from the right edge.
    valid_rev = torch.flip(valid_f, dims=[-1])
    cum_rev = torch.cumsum(valid_rev, dim=-1)            # (B, N, T)
    late_mask_rev = (cum_rev <= float(late_window_steps)) & (valid_rev > 0)
    late_mask = torch.flip(late_mask_rev, dims=[-1]).float()
    late_denom = late_mask.sum(dim=-1).clamp_min(eps)
    K_late = (kld_per_t * late_mask).sum(dim=-1) / late_denom

    # Lag distribution q_l: weighted by the TE lag mass hat_w * K * alpha,
    # not by attention alone.
    weight = (
        hat_w.unsqueeze(-1)
        * kld_per_t.unsqueeze(-1)
        * mean_alpha
    )                                                   # (B, N, T, L)
    q_unnorm = weight.sum(dim=-2)                         # (B, N, L)
    q_norm = q_unnorm / q_unnorm.sum(dim=-1, keepdim=True).clamp_min(eps)

    lags = torch.arange(L, device=mean_alpha.device, dtype=mean_alpha.dtype)
    m_lag = (q_norm * lags).sum(dim=-1)
    sigma_lag = (q_norm * (lags - m_lag.unsqueeze(-1)) ** 2).sum(dim=-1).clamp_min(0.0).sqrt()
    H_lag = -(q_norm * torch.log(q_norm + eps)).sum(dim=-1)

    # ``valid_f`` participates in ``late_mask``; ``valid`` (bool) gates the max.
    del valid_f, valid
    summary = torch.stack(
        [bar_K, K_max, K_late, m_lag, sigma_lag, H_lag], dim=-1
    )                                                     # (B, N, 6)
    # Replace any residual NaN/Inf with zeros (e.g. all-masked segments).
    return torch.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0)


class VaeSegmentTokenizer(nn.Module):
    """Segment tokenizer (PRD §6).

    Consumes per-timestep VAE outputs (already cached in fold-specific
    normalisation by the dataset) and emits a 256-d token per segment.

    Args:
        d_model_vae: Width of ``h^y`` (must match the VAE's ``d_model``).
        d_z: Latent dimensionality (must match the VAE).
        d_seg: Segment-core width after pooling (default 192).
        d_model: Output token width consumed by the temporal transformer
            (default 256).
        c_meta_dim: Number of causal-metadata features supplied by the
            dataset (default 10, see PRD §4.4).
        te_summary_dim: Number of TE-summary features (default 6).
        dropout: Dropout used inside residual MLPs.
        late_window_steps: Number of trailing valid steps that count as
            "late" for ``K_late`` (PRD §5.7).
    """

    def __init__(
        self,
        *,
        d_model_vae: int = 128,
        d_z: int = 24,
        d_seg: int = 192,
        d_model: int = 256,
        c_meta_dim: int = 10,
        te_summary_dim: int = 6,
        dropout: float = 0.1,
        late_window_steps: int = 75,
    ) -> None:
        super().__init__()
        self.d_model_vae = int(d_model_vae)
        self.d_z = int(d_z)
        self.d_seg = int(d_seg)
        self.d_model = int(d_model)
        self.c_meta_dim = int(c_meta_dim)
        self.te_summary_dim = int(te_summary_dim)
        self.late_window_steps = int(late_window_steps)

        # Per-stream LayerNorms keep h^y / mu_post / Δμ on comparable scales.
        self.ln_h_y = nn.LayerNorm(d_model_vae)
        self.ln_mu = nn.LayerNorm(d_z)
        self.ln_dmu = nn.LayerNorm(d_z)

        per_step_in = d_model_vae + 2 * d_z + 1
        self.step_proj = nn.Sequential(
            nn.Linear(per_step_in, d_seg),
            nn.LayerNorm(d_seg),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.step_residual = _ResidualMLP(d_seg, dropout=dropout)
        self.pool = _MaskedAttentivePoolOverT(d_seg, hidden_dim=64)

        token_in = d_seg + te_summary_dim + c_meta_dim
        self.token_proj = nn.Sequential(
            nn.Linear(token_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.token_residual = _ResidualMLP(d_model, dropout=dropout)

    def forward(
        self,
        *,
        h_y: torch.Tensor,
        mu_prior_norm: torch.Tensor,
        mu_post_norm: torch.Tensor,
        kld_per_t: torch.Tensor,
        mean_alpha: torch.Tensor,
        hat_w: torch.Tensor,
        c_meta: torch.Tensor,
        segment_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Build per-segment tokens.

        Args:
            h_y: ``(B, N, T, d_model_vae)`` — already normalised by the VAE
                encoder (LayerNorm exit). When loaded from cache the tensor
                is fp16-cast-to-fp32 by the dataset.
            mu_prior_norm: ``(B, N, T, d_z)`` prior mean z-scored by the
                per-fold cache stats (same ``(mean, std)`` applied to
                ``mu_post_norm``).
            mu_post_norm: ``(B, N, T, d_z)`` posterior mean z-scored by the
                per-fold cache stats.
            kld_per_t: ``(B, N, T)`` per-step KL.
            mean_alpha: ``(B, N, T, L)`` heads-mean attention.
            hat_w: ``(B, N, T)`` classifier-time mask (per PRD §6.3).
            c_meta: ``(B, N, c_meta_dim)`` causal metadata vector.
            segment_mask: ``(B, N)`` bool — True for valid segments. Padded
                rows produce zero tokens.

        Returns:
            Dict with:
              * ``segment_token``: ``(B, N, d_model)``
              * ``s_core``: ``(B, N, d_seg)``
              * ``u_TE``: ``(B, N, te_summary_dim)``
        """
        # Δμ in the normalised space. Because the dataset z-scores both
        # ``mu_post`` and ``mu_prior`` with the same fold stats, this
        # difference equals ``(mu_post_raw - mu_prior_raw) / std``: the raw
        # posterior delta scaled by the latent std (description §5.2).
        delta_mu = mu_post_norm - mu_prior_norm

        r = torch.cat(
            [
                self.ln_h_y(h_y),
                self.ln_mu(mu_post_norm),
                self.ln_dmu(delta_mu),
                torch.log1p(kld_per_t).unsqueeze(-1),
            ],
            dim=-1,
        )                                                 # (B, N, T, in)

        u = self.step_proj(r)                             # (B, N, T, d_seg)
        u = self.step_residual(u)
        s_core = self.pool(u, hat_w)                       # (B, N, d_seg)

        u_te = _compute_te_summary(
            mean_alpha=mean_alpha,
            kld_per_t=kld_per_t,
            hat_w=hat_w,
            late_window_steps=self.late_window_steps,
        )                                                 # (B, N, 6)

        token_input = torch.cat([s_core, u_te, c_meta], dim=-1)
        x = self.token_proj(token_input)                  # (B, N, d_model)
        x = self.token_residual(x)
        x = x * segment_mask.to(x.dtype).unsqueeze(-1)    # zero padded rows

        return {
            "segment_token": x,
            "s_core": s_core,
            "u_TE": u_te,
        }


__all__ = ["VaeSegmentTokenizer"]
