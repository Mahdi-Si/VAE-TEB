"""Prediction heads, latent modules, and window representation export.

Contains:
    - ForecastHead: Per-horizon forecasting head used by self/fused/TE paths.
    - SelfLatentModule: Legacy intrinsic FHR latent from the old experimental v2.
      It is kept only for compatibility with older checkpoints/tools.
    - TELatentModule: TE-style coupling latent with posterior and conditional prior.
    - WindowRepresentationExport: Window embedding matching model.md §20.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import AttentionPool


class _ResidualMLP(nn.Module):
    """Small residual MLP used by latent modules."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.skip_proj = (
            nn.Linear(in_dim, hidden_dim, bias=False)
            if in_dim != hidden_dim else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        skip = self.skip_proj(h)
        h = self.drop(F.silu(self.fc1(h)))
        h = self.drop(F.silu(self.fc2(h))) + skip
        return self.fc3(h)


class ForecastHead(nn.Module):
    """Shared-backbone per-horizon forecasting head."""

    def __init__(
        self,
        in_dim: int,
        d_out: int,
        horizons: Tuple[int, ...] = (8, 15, 30),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_out = d_out
        self.horizons = horizons
        hidden = 4 * in_dim

        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.skip_proj = (
            nn.Linear(in_dim, hidden, bias=False)
            if in_dim != hidden else nn.Identity()
        )

        self.projections = nn.ModuleDict({
            str(h): nn.Linear(hidden, h * d_out) for h in horizons
        })

    def forward(self, summary: Tensor) -> Dict[int, Tensor]:
        n = summary.shape[0]
        skip = self.skip_proj(summary)
        h = self.drop(F.silu(self.fc1(summary)))
        h = self.drop(F.silu(self.fc2(h))) + skip

        preds: Dict[int, Tensor] = {}
        for horizon in self.horizons:
            raw = self.projections[str(horizon)](h)
            preds[horizon] = raw.view(n, horizon, self.d_out)
        return preds


class SelfLatentModule(nn.Module):
    """Legacy intrinsic FHR latent module kept for compatibility only."""

    def __init__(
        self,
        d_model: int,
        d_z_self: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_z_self = d_z_self
        self.posterior = _ResidualMLP(
            in_dim=d_model,
            hidden_dim=max(1, 4 * max(d_z_self, 1)),
            out_dim=max(2, 2 * max(d_z_self, 1)),
            dropout=dropout,
        )

    def forward(self, s_f: Tensor) -> Dict[str, Tensor]:
        if self.d_z_self <= 0:
            empty = s_f.new_zeros(s_f.shape[0], 0)
            return {"z_self": empty, "mu_self": empty, "logvar_self": empty}

        post_out = self.posterior(s_f)
        mu_self, logvar_self = post_out.chunk(2, dim=-1)
        z_self = self._reparameterize(mu_self, logvar_self)
        return {
            "z_self": z_self,
            "mu_self": mu_self,
            "logvar_self": logvar_self,
        }

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    @staticmethod
    def kl_divergence(
        mu: Tensor,
        logvar: Tensor,
        free_bits: float = 0.0,
    ) -> Tensor:
        if mu.numel() == 0:
            return mu.new_tensor(0.0)
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        if free_bits > 0.0:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        return kl_per_dim.sum(dim=-1).mean()


class TELatentModule(nn.Module):
    """TE-style local coupling latent with posterior and conditional prior."""

    def __init__(
        self,
        d_model: int,
        d_z: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_z = d_z
        self.posterior = _ResidualMLP(
            in_dim=2 * d_model,
            hidden_dim=4 * d_z,
            out_dim=2 * d_z,
            dropout=dropout,
        )
        self.prior = _ResidualMLP(
            in_dim=d_model,
            hidden_dim=4 * d_z,
            out_dim=2 * d_z,
            dropout=dropout,
        )

    def forward(self, s_f: Tensor, s_u: Tensor) -> Dict[str, Tensor]:
        post_out = self.posterior(torch.cat([s_f, s_u], dim=-1))
        mu_post, logvar_post = post_out.chunk(2, dim=-1)

        prior_out = self.prior(s_f)
        mu_prior, logvar_prior = prior_out.chunk(2, dim=-1)

        z_te = self._reparameterize(mu_post, logvar_post)
        return {
            "z_te": z_te,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
        }

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    @staticmethod
    def kl_divergence(
        mu_post: Tensor,
        logvar_post: Tensor,
        mu_prior: Tensor,
        logvar_prior: Tensor,
        free_bits: float = 0.0,
    ) -> Tensor:
        kl_per_dim = 0.5 * (
            logvar_prior - logvar_post
            + (logvar_post.exp() + (mu_post - mu_prior).pow(2))
            / logvar_prior.exp()
            - 1.0
        )
        if free_bits > 0.0:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        return kl_per_dim.sum(dim=-1).mean()


class WindowRepresentationExport(nn.Module):
    """Window-level embedding matching model.md §20.

    Output:
        - e_F  = attn(H_F) + max(H_F)                      -> 2*d
        - e_FU = attn(H_FU) + max(H_FU) + 4 quarter means -> 6*d
        - e_TE = mean(mu_post) + max(mu_post)             -> 2*d_z

    Total dimension: 8*d + 2*d_z.
    """

    def __init__(self, d_model: int, d_z_transfer: int = 16) -> None:
        super().__init__()
        self.attn_pool_f = AttentionPool(d_model)
        self.attn_pool_fu = AttentionPool(d_model)
        self.d_model = d_model
        self.d_z_transfer = d_z_transfer

    @property
    def output_dim(self) -> int:
        return 8 * self.d_model + 2 * self.d_z_transfer

    def forward(self, h_f: Tensor, h_fu: Tensor, te_mus: Tensor) -> Tensor:
        e_f_attn = self.attn_pool_f(h_f)
        e_f_max = h_f.max(dim=1).values

        e_fu_attn = self.attn_pool_fu(h_fu)
        e_fu_max = h_fu.max(dim=1).values

        _, t, _ = h_fu.shape
        quarter = t // 4
        quarter_means = []
        for i in range(4):
            start = i * quarter
            end = (i + 1) * quarter if i < 3 else t
            quarter_means.append(h_fu[:, start:end, :].mean(dim=1))

        e_te_mean = te_mus.mean(dim=1)
        e_te_max = te_mus.max(dim=1).values

        return torch.cat(
            [e_f_attn, e_f_max, e_fu_attn, e_fu_max]
            + quarter_means
            + [e_te_mean, e_te_max],
            dim=-1,
        )
