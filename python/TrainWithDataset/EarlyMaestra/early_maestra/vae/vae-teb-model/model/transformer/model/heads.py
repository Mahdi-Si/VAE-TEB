"""Prediction heads, TE latent module, and window representation export.

Contains:
    - ForecastHead: Per-horizon MLP forecasting head (spec §13).
    - TELatentModule: Posterior/prior networks and reparameterization (spec §14).
    - WindowRepresentationExport: Pooled window-level embedding (spec §20).
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import AttentionPool


class ForecastHead(nn.Module):
    """Per-horizon MLP forecasting head (spec §13).

    Contains a separate MLP for each prediction horizon.  Each MLP maps from a
    summary vector to a future block of shape ``(h, d_out)``.

    Shared architecture for the self-only, fused, and TE residual heads (each
    instantiated with its own parameters and ``in_dim``).

    Args:
        in_dim: Input dimension (``d_model`` for self/fused, ``d_model + d_z``
            for TE residual).
        d_out: Output feature dimension per time step (``d_f``).
        horizons: Tuple of prediction horizons.
        dropout: Dropout probability.
    """

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
        self.heads = nn.ModuleDict()
        for h in horizons:
            self.heads[str(h)] = nn.Sequential(
                nn.Linear(in_dim, 4 * in_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * in_dim, h * d_out),
            )

    def forward(self, summary: Tensor) -> Dict[int, Tensor]:
        """Forward pass.

        Args:
            summary: Summary vector of shape ``(N, in_dim)`` where
                ``N = B * K`` (batch * anchors).

        Returns:
            Dictionary mapping each horizon ``h`` to a prediction tensor of
            shape ``(N, h, d_out)``.
        """
        N = summary.shape[0]
        preds = {}
        for h in self.horizons:
            raw = self.heads[str(h)](summary)         # (N, h * d_out)
            preds[h] = raw.view(N, h, self.d_out)     # (N, h, d_out)
        return preds


class TELatentModule(nn.Module):
    """TE-style local coupling latent with posterior and prior (spec §14).

    The posterior sees both FHR and UP summaries to capture the full
    source-conditioned distribution.  The prior sees only the FHR summary
    and represents what the coupling latent would look like without UP
    information.

    Args:
        d_model: Backbone dimension (input dim for each summary).
        d_z: Latent dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        d_z: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_z = d_z

        # Posterior: input is [s_F | s_U] of dim 2*d_model
        self.posterior = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, 4 * d_z),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_z, 2 * d_z),  # -> (mu, log_var)
        )

        # Prior: input is s_F of dim d_model
        self.prior = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_z),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_z, 2 * d_z),  # -> (mu0, log_var0)
        )

    def forward(
        self, s_f: Tensor, s_u: Tensor
    ) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            s_f: Intrinsic FHR summary of shape ``(N, d_model)``.
            s_u: UP summary of shape ``(N, d_model)``.

        Returns:
            Dictionary with keys:
                - ``z_te``: Sampled latent ``(N, d_z)``.
                - ``mu_post``: Posterior mean ``(N, d_z)``.
                - ``logvar_post``: Posterior log-variance ``(N, d_z)``.
                - ``mu_prior``: Prior mean ``(N, d_z)``.
                - ``logvar_prior``: Prior log-variance ``(N, d_z)``.
        """
        # Posterior
        post_out = self.posterior(torch.cat([s_f, s_u], dim=-1))
        mu_post, logvar_post = post_out.chunk(2, dim=-1)

        # Prior
        prior_out = self.prior(s_f)
        mu_prior, logvar_prior = prior_out.chunk(2, dim=-1)

        # Reparameterize
        z_te = self._reparameterize(mu_post, logvar_post)

        return {
            "z_te": z_te,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
        }

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample z = mu + sigma * epsilon via the reparameterization trick.

        Args:
            mu: Mean of shape ``(N, d_z)``.
            logvar: Log-variance of shape ``(N, d_z)``.

        Returns:
            Sampled latent of shape ``(N, d_z)``.
        """
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
    ) -> Tensor:
        """Closed-form KL divergence between two diagonal Gaussians.

        Computes ``KL(q || p)`` where ``q = N(mu_post, diag(exp(logvar_post)))``
        and ``p = N(mu_prior, diag(exp(logvar_prior)))``.

        Args:
            mu_post: Posterior mean ``(N, d_z)``.
            logvar_post: Posterior log-variance ``(N, d_z)``.
            mu_prior: Prior mean ``(N, d_z)``.
            logvar_prior: Prior log-variance ``(N, d_z)``.

        Returns:
            Mean KL divergence (scalar).
        """
        # KL(q||p) = 0.5 * sum(log(sigma_p^2/sigma_q^2) + (sigma_q^2 + (mu_q-mu_p)^2)/sigma_p^2 - 1)
        kl = 0.5 * (
            logvar_prior - logvar_post
            + (logvar_post.exp() + (mu_post - mu_prior).pow(2))
            / logvar_prior.exp()
            - 1.0
        )
        return kl.sum(dim=-1).mean()


class WindowRepresentationExport(nn.Module):
    """Pooled window-level embedding for downstream tasks (spec §20).

    Produces a fixed-size embedding by concatenating:
        - ``e_F``: Intrinsic FHR summary (attn_pool + max_pool) → ``2*d``.
        - ``e_FU``: Fused summary (attn_pool + max_pool + 4 quarter means) → ``6*d``.
        - ``e_TE``: TE summary (mean + max of posterior means) → ``2*d_z``.
    Total: ``8*d + 2*d_z``.

    Args:
        d_model: Backbone dimension.
        d_z: TE latent dimension.
    """

    def __init__(self, d_model: int, d_z: int = 16) -> None:
        super().__init__()
        self.attn_pool_f = AttentionPool(d_model)
        self.attn_pool_fu = AttentionPool(d_model)
        self.d_model = d_model
        self.d_z = d_z

    @property
    def output_dim(self) -> int:
        """Total dimension of the exported window embedding."""
        return 8 * self.d_model + 2 * self.d_z

    def forward(
        self,
        h_f: Tensor,
        h_fu: Tensor,
        te_mus: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            h_f: FHR-only encoder states of shape ``(B, T, d)``.
            h_fu: Fused encoder states of shape ``(B, T, d)``.
            te_mus: Posterior means at anchor grid positions of shape
                ``(B, K_grid, d_z)``.

        Returns:
            Window embedding of shape ``(B, output_dim)``.
        """
        # --- Intrinsic FHR summary (e_F): 2*d ---
        e_f_attn = self.attn_pool_f(h_f)                    # (B, d)
        e_f_max = h_f.max(dim=1).values                     # (B, d)

        # --- Fused summary (e_FU): 6*d ---
        e_fu_attn = self.attn_pool_fu(h_fu)                 # (B, d)
        e_fu_max = h_fu.max(dim=1).values                   # (B, d)

        B, T, d = h_fu.shape
        quarter = T // 4
        quarters = []
        for i in range(4):
            start = i * quarter
            end = (i + 1) * quarter if i < 3 else T
            quarters.append(h_fu[:, start:end, :].mean(dim=1))  # (B, d)

        # --- TE summary (e_TE): 2*d_z ---
        e_te_mean = te_mus.mean(dim=1)                       # (B, d_z)
        e_te_max = te_mus.max(dim=1).values                  # (B, d_z)

        # --- Concatenate ---
        e_win = torch.cat(
            [e_f_attn, e_f_max,
             e_fu_attn, e_fu_max] + quarters +
            [e_te_mean, e_te_max],
            dim=-1,
        )
        return e_win
