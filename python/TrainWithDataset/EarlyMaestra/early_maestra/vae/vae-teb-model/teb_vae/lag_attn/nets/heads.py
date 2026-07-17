r"""The latent heads: the prior, the source-conditioned posterior, and the KL readout.

The model's whole claim rests on an asymmetry between two distributions at each step $t$:

* the **prior** $p(z_t \mid Y_{\le t})$ sees the target's past only;
* the **posterior** $q(z_t \mid Y_{\le t}, U_{\le t})$ additionally sees the source.

Their divergence $K_t = \mathrm{KL}(q \,\|\, p)$ is therefore *how much the source told us that
the target's own past did not* -- a transfer-entropy surrogate. That reading survives only if the
prior never sees the source, which is why the prior is target-only by construction here rather
than by convention.

Both distributions are parameterised residually: the posterior is the prior plus a bounded delta,
and the delta heads are zero-initialised. At init the posterior *is* the prior, so $K_t \equiv 0$
exactly and training starts from "the source told us nothing" rather than from noise. It is worth
knowing that this makes every KL assertion on a freshly-built model vacuous.
"""
from __future__ import annotations

from typing import Optional, Tuple, cast

import torch
from torch import nn

from teb_vae.lag_attn.nets.blocks import ResidualMLP, geometric_schedule, smooth_bound


class PriorHead(nn.Module):
    r"""Target-only prior $p(z_t \mid Y_{\le t})$ and the decoder conditioning state.

    Produces four outputs from the target state $H^y$:

    * ``mu_prior`` -- prior mean ``(B, T, d_z)``, bounded by
      $\mu\_scale \cdot \tanh(\mathrm{raw} / \mu\_scale)$ so $|\mu^p| \le \mu\_scale$.
    * ``logvar_prior`` -- prior log-variance ``(B, T, d_z)``, smoothly bounded.
    * ``decoder_state`` -- target-only conditioning for the baseline decoder ``(B, T, d_model)``.
    * ``raw_logvar_prior`` -- the *pre-bound* log-variance ``(B, T, d_z)``.

    The fourth output is not a diagnostic. The bound is a sigmoid and therefore not idempotent,
    so the posterior cannot build an exact residual from the already-bounded value -- it needs
    the raw one, or the zero-KL-at-init property is lost.

    Each head is fed through its own ``LayerNorm`` so the raw encoder state cannot drift
    unbounded through any of them.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_z: int = 24,
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

        # Per-head input norms decouple the three heads from shared drift in h_y.
        self.mu_input_norm = nn.LayerNorm(d_model)
        self.logvar_input_norm = nn.LayerNorm(d_model)
        self.dec_input_norm = nn.LayerNorm(d_model)

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
        self.decoder_state_head = ResidualMLP(
            input_dim=d_model,
            hidden_dims=geometric_schedule(d_model, d_model, 3),
            final_activation=True,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=dropout,
        )

    def forward(
        self, h_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce the prior and the decoder conditioning state.

        Args:
            h_y: Target history state ``(B, T, d_model)``.

        Returns:
            ``(mu_prior, logvar_prior, decoder_state, raw_logvar_prior)``.
        """
        raw_mu = self.mu_prior_head(self.mu_input_norm(h_y))
        mu_prior = self.mu_scale * torch.tanh(raw_mu / self.mu_scale)

        raw_logvar_prior = self.logvar_prior_head(self.logvar_input_norm(h_y))
        logvar_prior = smooth_bound(raw_logvar_prior, *self.logvar_clamp)

        decoder_state = self.decoder_state_head(self.dec_input_norm(h_y))
        return mu_prior, logvar_prior, decoder_state, raw_logvar_prior


class PosteriorHead(nn.Module):
    r"""Source-conditioned posterior $q(z_t \mid Y_{\le t}, U_{\le t})$.

    Parameterised as a bounded residual around the prior:

    $$\mu^q_t = \mu^p_t + s_\mu \tanh(\widetilde{\Delta\mu}_t / s_\mu), \qquad
    \log\sigma^{2,q}_t = \mathrm{smoothbound}\!\left(\widetilde{\log\sigma^{2,p}_t}
    + s_\ell \tanh(\widetilde{\Delta\ell}_t / s_\ell)\right).$$

    Both deltas are zero-initialised and $\tanh(0) = 0$, so at init the posterior equals the
    prior *exactly* and $K_t \equiv 0$. Note the log-variance residual is added to the prior's
    **pre-bound raw** value before bounding, not to the bounded one -- the bound is a sigmoid
    and applying it twice would not be a no-op, and the exactness of the zero is the point.

    Two latent structures are supported. Flat fusion concatenates $[LN(H^y) \,\|\, LN(A)]$ and
    produces the whole latent at once. Head-structured mode partitions the latent into
    ``num_heads`` groups and derives each group *only* from the matching attention head's summary
    $a^{(m)}$, which is what makes the per-group KL a genuine additive decomposition
    $K_t = \sum_m K_t^{(m)}$ rather than an arbitrary slice of a shared vector. The prior stays
    shared and target-only either way.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_z: int = 24,
        logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
        dropout: float = 0.1,
        delta_mu_scale: float = 3.0,
        head_structured: bool = False,
        num_heads: int = 4,
        d_head: int = 32,
        *,
        delta_logvar_scale: float = 2.0,
    ) -> None:
        r"""Initialize the posterior head.

        Args:
            d_model: Encoder state width.
            d_z: Latent dimensionality.
            logvar_clamp: ``(lo, hi)`` effective range of the log-variance bound.
            dropout: Dropout used inside every internal MLP.
            delta_mu_scale: Saturation magnitude $s_\mu$ of the tanh-bounded mean delta.
            head_structured: Partition the latent into ``num_heads`` groups, each derived from
                one attention head's summary.
            num_heads: Number of latent groups when ``head_structured``.
            d_head: Per-head summary width when ``head_structured``.
            delta_logvar_scale: Saturation magnitude $s_\ell$ of the tanh-bounded log-variance
                delta.

        Raises:
            ValueError: If either scale is not positive, or if ``head_structured`` is set and
                ``d_z`` is not divisible by ``num_heads``.
        """
        super().__init__()
        if delta_mu_scale <= 0.0:
            raise ValueError(f"delta_mu_scale must be > 0, got {delta_mu_scale}")
        if delta_logvar_scale <= 0.0:
            raise ValueError(f"delta_logvar_scale must be > 0, got {delta_logvar_scale}")

        self.logvar_clamp = logvar_clamp
        self.delta_mu_scale = float(delta_mu_scale)
        self.delta_logvar_scale = float(delta_logvar_scale)
        self.head_structured = bool(head_structured)
        self.num_heads = int(num_heads)

        # Shared across both latent structures.
        self.h_y_norm = nn.LayerNorm(d_model)

        if self.head_structured:
            if d_z % num_heads != 0:
                raise ValueError(
                    f"head_structured posterior needs d_z % num_heads == 0, "
                    f"got d_z={d_z}, num_heads={num_heads}"
                )
            self.group = d_z // num_heads
            self.d_head = int(d_head)
            self.a_head_norm = nn.LayerNorm(d_head)
            fuse_in = d_model + d_head
            fuse_out = max(2 * self.group, 16)
            # One small fusion and one delta pair per latent group.
            self.fusion = nn.ModuleList(
                [
                    ResidualMLP(
                        input_dim=fuse_in,
                        hidden_dims=geometric_schedule(fuse_in, fuse_out, 2),
                        final_activation=True,
                        use_skip_connection=True,
                        use_input_layer_norm=True,
                        activation=nn.GELU,
                        dropout=dropout,
                    )
                    for _ in range(num_heads)
                ]
            )
            self.delta_mu_head = nn.ModuleList(
                [nn.Linear(fuse_out, self.group) for _ in range(num_heads)]
            )
            self.delta_logvar_head = nn.ModuleList(
                [nn.Linear(fuse_out, self.group) for _ in range(num_heads)]
            )
        else:
            # Per-modality norms keep H^y and A on comparable scales before the concat.
            self.a_norm = nn.LayerNorm(d_model)
            fused_in = 2 * d_model
            self.fusion = ResidualMLP(
                input_dim=fused_in,
                hidden_dims=geometric_schedule(fused_in, d_model, 3),
                final_activation=True,
                use_skip_connection=True,
                use_input_layer_norm=True,
                activation=nn.GELU,
                dropout=dropout,
            )
            self.delta_mu_head = nn.Linear(d_model, d_z)
            self.delta_logvar_head = nn.Linear(d_model, d_z)

    def forward(
        self,
        h_y: torch.Tensor,
        a: torch.Tensor,
        mu_prior: torch.Tensor,
        raw_logvar_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Produce the posterior as a bounded residual around the prior.

        Args:
            h_y: Target state ``(B, T, d_model)``.
            a: Attended source summary -- ``(B, T, d_model)`` flat, or
                ``(B, T, num_heads, d_head)`` head-structured.
            mu_prior: Prior mean ``(B, T, d_z)``, the base of the mean residual.
            raw_logvar_prior: Prior pre-bound raw log-variance ``(B, T, d_z)``, the base of the
                log-variance residual.

        Returns:
            ``(mu_post, logvar_post)``, both ``(B, T, d_z)``.

        Raises:
            ValueError: If ``raw_logvar_prior`` is not supplied.
        """
        if raw_logvar_prior is None:
            raise ValueError(
                "the posterior log-variance is a residual around the prior's pre-bound raw "
                "log-variance, so raw_logvar_prior is required; call via the model's forward "
                "or encode_only, which thread it through"
            )

        if self.head_structured:
            # Per-group modules; the flat path binds the same names to single modules.
            fusion = cast(nn.ModuleList, self.fusion)
            delta_mu_head = cast(nn.ModuleList, self.delta_mu_head)
            delta_logvar_head = cast(nn.ModuleList, self.delta_logvar_head)

            h_y_normed = self.h_y_norm(h_y)
            raw_deltas, logvar_terms = [], []
            for index in range(self.num_heads):
                a_head = self.a_head_norm(a[:, :, index, :])
                fused_head = fusion[index](torch.cat([h_y_normed, a_head], dim=-1))
                raw_deltas.append(delta_mu_head[index](fused_head))
                logvar_terms.append(delta_logvar_head[index](fused_head))
            raw_delta = torch.cat(raw_deltas, dim=-1)
            logvar_term = torch.cat(logvar_terms, dim=-1)
        else:
            fused = self.fusion(torch.cat([self.h_y_norm(h_y), self.a_norm(a)], dim=-1))
            raw_delta = self.delta_mu_head(fused)
            logvar_term = self.delta_logvar_head(fused)

        # tanh(0) = 0, so with the delta heads zero-initialised both deltas are identically 0
        # at init and the posterior collapses onto the prior exactly.
        delta_mu = self.delta_mu_scale * torch.tanh(raw_delta / self.delta_mu_scale)
        mu_post = mu_prior + delta_mu

        delta_logvar = self.delta_logvar_scale * torch.tanh(logvar_term / self.delta_logvar_scale)
        # Bound the summed raw value: smooth_bound is not idempotent, so bounding the prior and
        # then adding would leave logvar_post != logvar_prior at init.
        logvar_post = smooth_bound(raw_logvar_prior + delta_logvar, *self.logvar_clamp)

        return mu_post, logvar_post


class TEAnalysisHead(nn.Module):
    r"""Turn the per-step KL and the attention weights into a lag-resolved attribution.

    A pure function with no parameters. $K_t$ says *how much* the source contributed at step $t$;
    the attention weights say *which lag* it came from. Their product is the attribution:

    $$\widetilde{TE}_{t,\ell} = \sum_m K_t^{(m)}\,\alpha^{(m)}_{t,\ell}.$$

    That identity is only rigorous when the latent groups are head-aligned, because only then is
    $K_t^{(m)}$ the KL of a group that head $m$ alone produced. Without head structure the
    per-head split is an arbitrary slice of a shared latent, and the fallback -- $K_t$ times the
    head-mean attention -- is a diagnostic rather than a decomposition. Both are reported; only
    one is a claim.
    """

    def forward(
        self,
        kld_btd: torch.Tensor,
        attn_weights: torch.Tensor,
        head_structured: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the KL total, the lag attribution, and the per-group KL.

        Args:
            kld_btd: Per-step per-dimension KL ``(B, T, d_z)``.
            attn_weights: Attention probabilities ``(B, T, num_heads, L)`` in lag order.
            head_structured: Whether the latent groups are head-aligned. Controls only the
                ``te_lag_map`` definition.

        Returns:
            ``(kld_per_t, te_lag_map, kld_per_t_per_head)`` of shapes ``(B, T)``, ``(B, T, L)``
            and ``(B, T, num_heads)``.
        """
        batch, seq_len, d_z = kld_btd.shape
        num_heads = attn_weights.shape[2]
        kld_per_t = kld_btd.sum(dim=-1)

        if d_z % num_heads == 0:
            group = d_z // num_heads
            kld_per_head = kld_btd.view(batch, seq_len, num_heads, group).sum(dim=-1)
        else:  # pragma: no cover - d_z is divisible by num_heads in practice
            kld_per_head = kld_per_t.unsqueeze(-1).expand(batch, seq_len, num_heads) / num_heads

        if head_structured and d_z % num_heads == 0:
            te_lag_map = torch.einsum("btm,btml->btl", kld_per_head, attn_weights)
        else:
            mean_alpha = attn_weights.mean(dim=-2)
            te_lag_map = kld_per_t.unsqueeze(-1) * mean_alpha

        return kld_per_t, te_lag_map, kld_per_head
