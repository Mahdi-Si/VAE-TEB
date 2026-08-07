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

from typing import List, Optional, Tuple, cast

import torch
from torch import nn

from teb_vae.lag_attn.nets.blocks import (
    ResidualMLP,
    geometric_schedule,
    smooth_bound,
    validate_choice,
)

#: How the posterior's log-variance is produced. ``'residual'`` is a bounded delta on the
#: prior's pre-bound raw value -- the parameterisation that makes $q \equiv p$ bitwise at
#: init, and which also routes the full branch's sharpening pressure onto the prior's own
#: tensor. ``'independent'`` gives the posterior its own head and severs that path.
POSTERIOR_LOGVAR_MODES = ("residual", "independent")


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
        posterior_logvar_mode: str = "residual",
        source_dropout: float = 0.0,
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
                delta. Read only under ``posterior_logvar_mode='residual'``.
            source_dropout: Dropout applied to the attended source summary before the fusion.
                An explicit rate, defaulting to $0$: this module did not exist before the
                ``source_dropout`` config key did, so *identity* is what reproduces the previous
                model here. The caller resolves the key -- at the source pathway's older sites an
                unset key means the global ``dropout``, because those sites always ran at it, and
                here it means $0$. See the model's ``source_dropout`` for both halves.

                The posterior's map from the source to $\Delta\mu$ is where the measured
                overfitting lives -- the held-out predictive gain decays while the KL and the
                displacement magnitude stay fixed, so what degrades is the *content* of that map
                rather than its size -- and this is the only seam that regularises it without also
                regularising the target pathway.
            posterior_logvar_mode: How $\log\sigma^{2,q}$ is produced. ``'residual'`` adds a
                bounded delta to the prior's **pre-bound raw** log-variance, which is what makes
                $q \equiv p$ bitwise at init. ``'independent'`` gives the posterior its own head
                and no dependence on the prior's raw value at all.

                The difference is a gradient path, not a parameterisation preference. Under
                ``'residual'`` the reconstruction's pressure to sharpen $\sigma^q$ lands on
                ``raw_logvar_prior`` -- the *prior's* tensor -- so the full branch drags the
                prior's scale down alongside its own. That is a second, indirect route to the
                prior-variance collapse, distinct from the base branch's direct one, and it is
                the reason turning off the base branch's noise alone does not stop the collapse.
                ``'independent'`` severs it.

                The cost: with two independent heads, $q \equiv p$ at init is no longer
                structural. It is recovered exactly under ``head_init_calibration``, which pins
                the prior's raw log-variance to a constant that the model's init also writes into
                this head; without that flag the KL starts small but nonzero.

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
        self.posterior_logvar_mode = validate_choice(
            posterior_logvar_mode, POSTERIOR_LOGVAR_MODES, "posterior_logvar_mode"
        )

        # Shared across both latent structures.
        self.h_y_norm = nn.LayerNorm(d_model)
        # Dropout on the ATTENDED SOURCE SUMMARY only, after its norm and before the fusion. This
        # is the one place in the fusion where the source can be regularised without touching the
        # target: past this point h_y and a are concatenated and the MLP cannot tell them apart.
        #
        # Takes an explicit rate and defaults to 0, NOT to `dropout`. This module is new with the
        # `source_dropout` key -- before it, `a` entered the fusion with no dropout at all -- so a
        # fallback to the global rate would put p=0.1 here in every config that leaves the key
        # unset, which is a silent regularisation change in every existing run and would make the
        # source_dropout sweep arms measure 0.1 -> 0.2 rather than off -> 0.2.
        self.a_dropout = nn.Dropout(float(source_dropout))

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
            # Exactly one log-variance head is built, never both: an unused one would receive no
            # gradient, which is what hangs a run under ``find_unused_parameters=False``. Same
            # shape either way, so the two modes differ in what the head's output is added to and
            # in nothing else.
            logvar_head: nn.Module = nn.ModuleList(
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
            logvar_head = nn.Linear(d_model, d_z)

        # Bound to whichever name says what it does, with the other left as None. Both attributes
        # are always present so a `hasattr` check stays honest: a residual-mode head really has no
        # `logvar_post_head`, rather than one that silently feeds nothing.
        self.delta_logvar_head = (
            logvar_head if self.posterior_logvar_mode == "residual" else None
        )
        self.logvar_post_head = (
            logvar_head if self.posterior_logvar_mode == "independent" else None
        )

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
        # Required in both modes, even though 'independent' does not read it: the argument is what
        # the caller threads the prior's raw value through, and a mode that quietly accepted None
        # would make a wrong call site fail only when the mode changed.
        if raw_logvar_prior is None:
            raise ValueError(
                "the posterior log-variance is built against the prior's pre-bound raw "
                "log-variance, so raw_logvar_prior is required; call via the model's forward "
                "or encode_only, which thread it through"
            )

        fused_heads: List[torch.Tensor] = []
        fused_flat: Optional[torch.Tensor] = None
        if self.head_structured:
            # Per-group modules; the flat path binds the same names to single modules.
            fusion = cast(nn.ModuleList, self.fusion)
            delta_mu_head = cast(nn.ModuleList, self.delta_mu_head)

            h_y_normed = self.h_y_norm(h_y)
            raw_deltas = []
            for index in range(self.num_heads):
                a_head = self.a_dropout(self.a_head_norm(a[:, :, index, :]))
                fused_head = fusion[index](torch.cat([h_y_normed, a_head], dim=-1))
                fused_heads.append(fused_head)
                raw_deltas.append(delta_mu_head[index](fused_head))
            raw_delta = torch.cat(raw_deltas, dim=-1)
        else:
            fused_flat = self.fusion(
                torch.cat([self.h_y_norm(h_y), self.a_dropout(self.a_norm(a))], dim=-1)
            )
            raw_delta = self.delta_mu_head(fused_flat)

        # tanh(0) = 0, so with the delta heads zero-initialised both deltas are identically 0
        # at init and the posterior collapses onto the prior exactly.
        delta_mu = self.delta_mu_scale * torch.tanh(raw_delta / self.delta_mu_scale)
        mu_post = mu_prior + delta_mu

        raw_logvar_post = self._run_logvar_head(fused_heads, fused_flat)
        if self.delta_logvar_head is not None:
            delta_logvar = (
                self.delta_logvar_scale * torch.tanh(raw_logvar_post / self.delta_logvar_scale)
            )
            # Bound the summed raw value: smooth_bound is not idempotent, so bounding the prior
            # and then adding would leave logvar_post != logvar_prior at init.
            logvar_post = smooth_bound(raw_logvar_prior + delta_logvar, *self.logvar_clamp)
        else:
            # `raw_logvar_prior` is not read here, which is the whole point of the mode: no
            # gradient from the full branch's reconstruction reaches the prior's log-variance
            # through this head.
            logvar_post = smooth_bound(raw_logvar_post, *self.logvar_clamp)

        return mu_post, logvar_post

    def _run_logvar_head(
        self, fused_heads: List[torch.Tensor], fused_flat: Optional[torch.Tensor]
    ) -> torch.Tensor:
        r"""Run whichever log-variance head was built over the fused features.

        One helper for both modes because the head is the same shape either way -- only what its
        output *means* differs, and that is the caller's decision. Selected by asking which
        attribute is not ``None`` rather than by re-reading the mode string, so the two attributes
        stay the single source of truth for which head exists.

        Args:
            fused_heads: Per-group fused features; non-empty under ``head_structured``.
            fused_flat: The single fused tensor; non-``None`` otherwise.

        Returns:
            The head's raw output ``(B, T, d_z)``.
        """
        head = self.delta_logvar_head if self.logvar_post_head is None else self.logvar_post_head
        if self.head_structured:
            modules = cast(nn.ModuleList, head)
            return torch.cat(
                [modules[index](fused_heads[index]) for index in range(self.num_heads)], dim=-1
            )
        return cast(nn.Linear, head)(fused_flat)


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
