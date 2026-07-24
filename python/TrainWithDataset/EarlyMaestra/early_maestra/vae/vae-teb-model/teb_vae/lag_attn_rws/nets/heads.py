r"""The full-latent prior head: the complete target-only latent state, and nothing else.

The prior $p(z_t \mid Y_{\le t})$ here is not a regulariser on a side channel -- it *is* the
model's forecast state. The shared decoder receives only $z$, so everything the baseline
forecast knows about the target must pass through this head's two outputs. That is also why
there is no ``decoder_state`` output: a target-only conditioning path around the latent would
turn $z$ back into a residual code, and an unused head would leave dead parameters that a
distributed run must then be told to tolerate. The head is written without it rather than
reusing the sibling's ``PriorHead`` and discarding a tensor.

The pre-bound raw log-variance is returned alongside the bounded one because the posterior is a
residual on the *raw* value: the bound is a sigmoid and not idempotent, so a residual built on
the already-bounded value could not reproduce the prior exactly at zero delta, and the exact
zero-KL initialization would silently not hold.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from teb_vae.lag_attn.nets.blocks import ResidualMLP, geometric_schedule, smooth_bound


class FullLatentPriorHead(nn.Module):
    r"""Target-only prior $p_\theta(z_t \mid Y_{\le t})$ over the full latent.

    Produces three outputs from the target state $H^y$:

    * ``mu_prior`` -- prior mean ``(B, T, d_z)``, bounded by
      $\mu\_scale \cdot \tanh(\mathrm{raw} / \mu\_scale)$ so $|\mu^p| \le \mu\_scale$.
    * ``logvar_prior`` -- prior log-variance ``(B, T, d_z)``, smoothly bounded.
    * ``raw_logvar_prior`` -- the *pre-bound* log-variance ``(B, T, d_z)``, the base the
      posterior residual is applied to.

    Each head is fed through its own ``LayerNorm`` so the raw encoder state cannot drift
    unbounded through either of them.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_z: int = 48,
        logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
        dropout: float = 0.1,
        mu_scale: float = 5.0,
    ) -> None:
        """Initialize the prior head.

        Args:
            d_model: Encoder state width.
            d_z: Latent dimensionality.
            logvar_clamp: ``(lo, hi)`` effective range of the log-variance bound.
            dropout: Dropout used inside every internal MLP.
            mu_scale: Saturation magnitude of the tanh-bounded prior mean. Set large enough to
                be non-restrictive around the $N(0, I)$ reference.

        Raises:
            ValueError: If ``mu_scale`` is not positive.
        """
        super().__init__()
        if mu_scale <= 0.0:
            raise ValueError(f"mu_scale must be > 0, got {mu_scale}")
        self.logvar_clamp = logvar_clamp
        self.mu_scale = float(mu_scale)

        # Per-head input norms decouple the two heads from shared drift in h_y.
        self.mu_input_norm = nn.LayerNorm(d_model)
        self.logvar_input_norm = nn.LayerNorm(d_model)

        self.mu_prior_head = ResidualMLP(
            input_dim=d_model,
            hidden_dims=geometric_schedule(d_model, d_z, 4),
            final_activation=False,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=dropout,
        )
        self.logvar_prior_head = ResidualMLP(
            input_dim=d_model,
            hidden_dims=geometric_schedule(d_model, d_z, 4),
            final_activation=False,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=dropout,
        )

    def forward(
        self, h_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce the complete target-only latent distribution.

        Args:
            h_y: Target history state ``(B, T, d_model)``.

        Returns:
            ``(mu_prior, logvar_prior, raw_logvar_prior)``, each ``(B, T, d_z)``, with
            ``logvar_prior == smooth_bound(raw_logvar_prior)`` exactly.
        """
        raw_mu = self.mu_prior_head(self.mu_input_norm(h_y))
        mu_prior = self.mu_scale * torch.tanh(raw_mu / self.mu_scale)

        raw_logvar_prior = self.logvar_prior_head(self.logvar_input_norm(h_y))
        logvar_prior = smooth_bound(raw_logvar_prior, *self.logvar_clamp)
        return mu_prior, logvar_prior, raw_logvar_prior
