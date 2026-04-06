"""Prediction heads, latent modules, and window representation export.

Contains:
    - ForecastHead: Shared-backbone per-horizon forecasting head (spec §13, v2).
    - SelfLatentModule: Intrinsic FHR latent with standard Gaussian prior (v2).
    - TELatentModule: TE-style coupling latent with conditional prior (spec §14, v2).
    - WindowRepresentationExport: Enriched pooled window-level embedding (spec §20, v2).
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import AttentionPool, RMSNorm


# ---------------------------------------------------------------------------
# Residual MLP helper
# ---------------------------------------------------------------------------

class _ResidualMLP(nn.Module):
    """3-layer residual MLP used by latent modules.

    Architecture::

        RMSNorm(in_dim)
        -> Linear(in_dim, hidden_dim) -> SiLU -> Dropout
        -> Linear(hidden_dim, hidden_dim) -> SiLU -> Dropout -> +skip
        -> Linear(hidden_dim, out_dim)

    Args:
        in_dim: Input dimension.
        hidden_dim: Hidden layer dimension.
        out_dim: Output dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        # Skip projection if dimensions mismatch
        self.skip_proj = (
            nn.Linear(in_dim, hidden_dim, bias=False)
            if in_dim != hidden_dim else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(N, in_dim)``.

        Returns:
            Output tensor of shape ``(N, out_dim)``.
        """
        h = self.norm(x)
        skip = self.skip_proj(h)
        h = self.drop(F.silu(self.fc1(h)))
        h = self.drop(F.silu(self.fc2(h))) + skip
        return self.fc3(h)


# ---------------------------------------------------------------------------
# Forecast Head (v2: shared backbone + per-horizon projection)
# ---------------------------------------------------------------------------

class ForecastHead(nn.Module):
    """Shared-backbone per-horizon MLP forecasting head (spec §13, v2).

    Uses a shared 2-layer residual backbone followed by per-horizon linear
    projections.  This encourages cross-horizon consistency while reducing
    total parameters compared to fully independent per-horizon MLPs.

    Args:
        in_dim: Input dimension (``d_model`` for self/fused,
            ``d_model + d_z_self + d_z_transfer`` for TE residual).
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
        hidden = 4 * in_dim

        # Shared 2-layer residual backbone
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        # Skip projection for residual
        self.skip_proj = (
            nn.Linear(in_dim, hidden, bias=False)
            if in_dim != hidden else nn.Identity()
        )

        # Per-horizon output projections
        self.projections = nn.ModuleDict()
        for h in horizons:
            self.projections[str(h)] = nn.Linear(hidden, h * d_out)

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

        # Shared backbone with residual
        skip = self.skip_proj(summary)
        h = self.drop(F.silu(self.fc1(summary)))
        h = self.drop(F.silu(self.fc2(h))) + skip  # (N, hidden)

        # Per-horizon projections
        preds = {}
        for horizon in self.horizons:
            raw = self.projections[str(horizon)](h)     # (N, horizon * d_out)
            preds[horizon] = raw.view(N, horizon, self.d_out)
        return preds


# ---------------------------------------------------------------------------
# Self Latent Module (v2: intrinsic FHR state)
# ---------------------------------------------------------------------------

class SelfLatentModule(nn.Module):
    """Intrinsic FHR latent module with standard Gaussian prior (v2).

    Encodes the intrinsic fetal state at each anchor into a compact latent
    ``z_self`` with KL regularization against N(0, I).  This captures
    baseline, variability, and deceleration severity -- information
    identifiable from FHR alone.

    Args:
        d_model: Backbone dimension (input dim for s_F summary).
        d_z_self: Latent dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        d_z_self: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_z_self = d_z_self

        # Posterior: s_F -> (mu_self, logvar_self)
        self.posterior = _ResidualMLP(
            in_dim=d_model,
            hidden_dim=4 * d_z_self,
            out_dim=2 * d_z_self,
            dropout=dropout,
        )

    def forward(self, s_f: Tensor) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            s_f: Intrinsic FHR summary of shape ``(N, d_model)``.

        Returns:
            Dictionary with keys:
                - ``z_self``: Sampled latent ``(N, d_z_self)``.
                - ``mu_self``: Posterior mean ``(N, d_z_self)``.
                - ``logvar_self``: Posterior log-variance ``(N, d_z_self)``.
        """
        post_out = self.posterior(s_f)
        mu_self, logvar_self = post_out.chunk(2, dim=-1)

        z_self = self._reparameterize(mu_self, logvar_self)

        return {
            "z_self": z_self,
            "mu_self": mu_self,
            "logvar_self": logvar_self,
        }

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample z = mu + sigma * epsilon via the reparameterization trick.

        Args:
            mu: Mean of shape ``(N, d_z_self)``.
            logvar: Log-variance of shape ``(N, d_z_self)``.

        Returns:
            Sampled latent of shape ``(N, d_z_self)``.
        """
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
        """KL divergence KL(q(z_self) || N(0, I)) with optional free bits.

        Args:
            mu: Posterior mean ``(N, d_z_self)``.
            logvar: Posterior log-variance ``(N, d_z_self)``.
            free_bits: Per-dimension KL floor in nats.

        Returns:
            Mean KL divergence (scalar).
        """
        # KL(N(mu, sigma^2) || N(0, I)) = 0.5 * (mu^2 + sigma^2 - 1 - log(sigma^2))
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        if free_bits > 0.0:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        return kl_per_dim.sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# TE Latent Module (v2: deeper networks + free bits)
# ---------------------------------------------------------------------------

class TELatentModule(nn.Module):
    """TE-style local coupling latent with posterior and prior (spec §14, v2).

    The posterior sees both FHR and UP summaries to capture the full
    source-conditioned distribution.  The prior sees only the FHR summary
    and represents what the coupling latent would look like without UP
    information.

    v2 changes: deeper 3-layer residual MLP networks and free-bits support
    in KL divergence.

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

        # Posterior: [s_F | s_U] (2*d_model) -> (mu, logvar) each (d_z,)
        self.posterior = _ResidualMLP(
            in_dim=2 * d_model,
            hidden_dim=4 * d_z,
            out_dim=2 * d_z,
            dropout=dropout,
        )

        # Prior: s_F (d_model) -> (mu0, logvar0) each (d_z,)
        self.prior = _ResidualMLP(
            in_dim=d_model,
            hidden_dim=4 * d_z,
            out_dim=2 * d_z,
            dropout=dropout,
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
        free_bits: float = 0.0,
    ) -> Tensor:
        """Closed-form KL divergence between two diagonal Gaussians with free bits.

        Computes ``KL(q || p)`` where ``q = N(mu_post, diag(exp(logvar_post)))``
        and ``p = N(mu_prior, diag(exp(logvar_prior)))``.

        Args:
            mu_post: Posterior mean ``(N, d_z)``.
            logvar_post: Posterior log-variance ``(N, d_z)``.
            mu_prior: Prior mean ``(N, d_z)``.
            logvar_prior: Prior log-variance ``(N, d_z)``.
            free_bits: Per-dimension KL floor in nats.

        Returns:
            Mean KL divergence (scalar).
        """
        kl_per_dim = 0.5 * (
            logvar_prior - logvar_post
            + (logvar_post.exp() + (mu_post - mu_prior).pow(2))
            / logvar_prior.exp()
            - 1.0
        )
        if free_bits > 0.0:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        return kl_per_dim.sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Window Representation Export (v2: enriched)
# ---------------------------------------------------------------------------

class WindowRepresentationExport(nn.Module):
    """Enriched pooled window-level embedding for downstream tasks (spec §20, v2).

    Produces a fixed-size embedding by concatenating:

    **FHR intrinsic (e_F):** ``2 * d``
        - Attention-pooled H_F + max-pooled H_F.

    **Fused summary (e_FU):** ``10 * d``
        - Attention-pooled H_FU + max-pooled H_FU.
        - 4 quarter means (temporal progression).
        - 4 quarter stds (within-quarter variability).

    **z_self summary:** ``3 * d_z_self``
        - Mean, max, std of mu_self over anchor grid.

    **z_transfer summary:** ``2 * d_z_transfer``
        - Mean, max of mu_post (TE posterior means) over anchor grid.

    **z_transfer innovation:** ``2 * d_z_transfer``
        - Mean, max of |mu_post - mu_prior| over anchor grid.

    **z_transfer KL trajectory:** ``2``
        - Mean KL, max KL over anchor grid (scalar per anchor).

    Total: ``12*d + 3*d_z_self + 4*d_z_transfer + 2``.

    Args:
        d_model: Backbone dimension.
        d_z_self: Self latent dimension.
        d_z_transfer: Transfer latent dimension.
    """

    def __init__(
        self, d_model: int, d_z_self: int = 32, d_z_transfer: int = 16,
    ) -> None:
        super().__init__()
        self.attn_pool_f = AttentionPool(d_model)
        self.attn_pool_fu = AttentionPool(d_model)
        self.d_model = d_model
        self.d_z_self = d_z_self
        self.d_z_transfer = d_z_transfer

    @property
    def output_dim(self) -> int:
        """Total dimension of the exported window embedding."""
        return (
            12 * self.d_model
            + 3 * self.d_z_self
            + 4 * self.d_z_transfer
            + 2
        )

    def forward(
        self,
        h_f: Tensor,
        h_fu: Tensor,
        self_mus: Tensor,
        te_mus: Tensor,
        te_mu_priors: Tensor,
        te_logvar_posts: Tensor,
        te_logvar_priors: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            h_f: FHR-only encoder states of shape ``(B, T, d)``.
            h_fu: Fused encoder states of shape ``(B, T, d)``.
            self_mus: Self-latent posterior means at grid positions,
                shape ``(B, K_grid, d_z_self)``.
            te_mus: TE posterior means at grid positions,
                shape ``(B, K_grid, d_z_transfer)``.
            te_mu_priors: TE prior means at grid positions,
                shape ``(B, K_grid, d_z_transfer)``.
            te_logvar_posts: TE posterior log-variances at grid positions,
                shape ``(B, K_grid, d_z_transfer)``.
            te_logvar_priors: TE prior log-variances at grid positions,
                shape ``(B, K_grid, d_z_transfer)``.

        Returns:
            Window embedding of shape ``(B, output_dim)``.
        """
        # --- Intrinsic FHR summary (e_F): 2*d ---
        e_f_attn = self.attn_pool_f(h_f)                    # (B, d)
        e_f_max = h_f.max(dim=1).values                     # (B, d)

        # --- Fused summary (e_FU): 10*d ---
        e_fu_attn = self.attn_pool_fu(h_fu)                 # (B, d)
        e_fu_max = h_fu.max(dim=1).values                   # (B, d)

        B, T, d = h_fu.shape
        quarter = T // 4
        quarter_means = []
        quarter_stds = []
        for i in range(4):
            start = i * quarter
            end = (i + 1) * quarter if i < 3 else T
            chunk = h_fu[:, start:end, :]
            quarter_means.append(chunk.mean(dim=1))          # (B, d)
            quarter_stds.append(chunk.std(dim=1))            # (B, d)

        # --- z_self summary: 3*d_z_self ---
        e_self_mean = self_mus.mean(dim=1)                   # (B, d_z_self)
        e_self_max = self_mus.max(dim=1).values              # (B, d_z_self)
        e_self_std = self_mus.std(dim=1)                     # (B, d_z_self)

        # --- z_transfer summary: 2*d_z_transfer ---
        e_te_mean = te_mus.mean(dim=1)                       # (B, d_z_transfer)
        e_te_max = te_mus.max(dim=1).values                  # (B, d_z_transfer)

        # --- z_transfer innovation: 2*d_z_transfer ---
        innovation = (te_mus - te_mu_priors).abs()           # (B, K, d_z_transfer)
        e_innov_mean = innovation.mean(dim=1)                # (B, d_z_transfer)
        e_innov_max = innovation.max(dim=1).values           # (B, d_z_transfer)

        # --- z_transfer KL trajectory: 2 ---
        # Per-anchor KL: sum over d_z_transfer dims -> (B, K)
        kl_per_anchor = 0.5 * (
            te_logvar_priors - te_logvar_posts
            + (te_logvar_posts.exp()
               + (te_mus - te_mu_priors).pow(2))
            / te_logvar_priors.exp()
            - 1.0
        ).sum(dim=-1)                                        # (B, K)
        e_kl_mean = kl_per_anchor.mean(dim=1, keepdim=True)  # (B, 1)
        e_kl_max = kl_per_anchor.max(dim=1, keepdim=True).values  # (B, 1)

        # --- Concatenate ---
        e_win = torch.cat(
            [e_f_attn, e_f_max,
             e_fu_attn, e_fu_max]
            + quarter_means + quarter_stds
            + [e_self_mean, e_self_max, e_self_std,
               e_te_mean, e_te_max,
               e_innov_mean, e_innov_max,
               e_kl_mean, e_kl_max],
            dim=-1,
        )
        return e_win
