r"""Local per-lag proposals, the explicit sum that fuses them, and the residual-form divergence.

The fusion boundary of this architecture, and the only place in it where two stored source times
meet. Three stages, deliberately written as three separable pieces because each is verifiable on its
own and the model that composes them is not:

**One proposal per lag.** A learned embedding $\zeta_\ell$ identifies the fixed lag *index* -- not
its content -- and one shared map takes the anchor's target state, that lag's flattened source
vector and the embedding to a pair of $d_z$ updates,

$$(r^\mu_{t,\ell},\, r^\sigma_{t,\ell})
= s_{t,\ell}\, v_{t,\ell}\;
F_\theta\bigl([h_t,\, \operatorname{vec}(E_{t,\ell}),\, \zeta_\ell]\bigr).$$

Each proposal reads exactly one stored source time, though it may combine that time's channels and
the target context. A proposal is a *suggested update*: not a sampled code, not a probability over
lags, and not an independently identified lag effect.

**An explicit sum, bounded after summation.**

$$\bar a_t = c_L \sum_\ell r^\mu_{t,\ell},
\qquad a_t = a_{\max}\tanh(\bar a_t / a_{\max}),$$

and the same for the scale channel. The limiter acts *after* the sum, so the final correction is not
a sum of separately bounded lag effects and cross-lag effects can arise through it.

**A prior-relative residual.**

$$\mu^q_t = \mu^p_t + \sigma^p_t \odot a_t,
\qquad \lambda^q_t = \lambda^p_t + 2 b_t,$$

so the mean correction is bounded in prior-standard-deviation units and the divergence collapses to

$$K_t = \tfrac12 \sum_d \bigl[a_{t,d}^2 + e^{2b_{t,d}} - 1 - 2b_{t,d}\bigr],$$

which is nonnegative because $e^x \ge 1 + x$ and exactly zero at $a = b = 0$.

**Three things this module must never grow.** A renormalisation of $c_L$ by the number of *available*
lags, which would make the same evidence weigh differently at two anchors for no reason the model
could learn. A second application of the prior's log-variance bound to $\lambda^q$: that map is a
sigmoid and is not idempotent, so applying it again would change the parameterisation and break the
exact zero-update equality. And a per-lag attribution of the divergence: the cross terms in the sum
can reinforce or cancel, so no nonnegative allocation over lags exists, and
:func:`cancellation_ratio` is the honest readout of that fact rather than a workaround for it.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

#: Standard deviation the lag embedding is drawn at. Small, because the embedding identifies an
#: index rather than carrying content: it exists so that summing a shared lag-blind function does
#: not make the representation invariant to permuting source times, and a large initial draw would
#: start the head reading lag identity as though it were evidence.
LAG_EMBED_STD = 0.02

#: Numerical floor in the cancellation ratio's denominator. A diagnostic constant, not a loss term.
CANCELLATION_EPS = 1.0e-8


def default_lag_scale(n_lags: int) -> float:
    r"""The summation convention $c_L = L^{-1/2}$.

    A numerical convention and nothing more. For uncorrelated equal-variance proposals it stabilises
    the sum's variance, but shared heads and autocorrelated source history make independence
    implausible, so in general

    $$\operatorname{Var}\Bigl(c_L \sum_\ell r_\ell\Bigr)
      = c_L^2 \sum_{\ell,k} \operatorname{Cov}(r_\ell, r_k),$$

    and $L$ identical proposals still accumulate with amplitude $\sqrt L$. Resolved once from the
    **configured** lag count and then held fixed, including where lags are masked.

    Args:
        n_lags: $L$, the configured number of candidate lags.

    Returns:
        The scale.

    Raises:
        ValueError: If ``n_lags`` is not positive.
    """
    if int(n_lags) < 1:
        raise ValueError(f"n_lags must be >= 1, got {n_lags}")
    return float(int(n_lags) ** -0.5)


class LagProposalHead(nn.Module):
    r"""One shared map from an anchor's context and one lag's source vector to a latent update.

    The only learned module on the source pathway. Its input is the concatenation, in this order,

    $$\bigl[\,h_t \;\Vert\; \operatorname{vec}(E_{t,\ell}) \;\Vert\; \zeta_\ell\,\bigr]
      \in \mathbb R^{d_h + C_U W + E},$$

    and its body is three linear layers with GELU after the first two and a linear final output, no
    dropout and no normalisation. The final projection is zeroed, so at initialisation every
    proposal is exactly zero and the full distribution *is* the prior.

    **The input projection is applied per part rather than to a materialised concatenation**, and
    the two are the same arithmetic: a linear map of a concatenation is the sum of linear maps of
    the parts. It is done that way because the concatenation is the largest tensor this
    architecture would ever build -- one entry per batch element, anchor, lag and input coordinate
    -- while the target-state part of it is constant across the lag axis and need not be repeated
    $L$ times. One ``nn.Linear`` still owns the weight, so the parameter, its name in the state
    dict and its initialisation are exactly those of the concatenated form, and a test asserts the
    two agree.

    **The selector is an argument and is never learned.** It is not an attention weight and carries
    no gradient of its own; it exists so an intervention can suppress a band of lags while holding
    everything else fixed.

    Attributes:
        d_z: Latent width each proposal updates.
        n_lags: $L$, the number of lag slots the embedding table covers.
        source_dim: Flattened width of one lag's source vector, $C_U W$.
        mean_only: Whether the scale-proposal half was built at all.
    """

    def __init__(
        self,
        *,
        d_model: int,
        d_z: int,
        n_lags: int,
        source_dim: int,
        lag_embed_dim: int = 8,
        hidden: Optional[int] = None,
        mean_only: bool = False,
    ) -> None:
        r"""Initialize the proposal head.

        Args:
            d_model: Width $d_h$ of the anchor's target state.
            d_z: Latent width $d_z$. One proposal of this width is emitted per lag, or two on the
                mean-and-scale arm.
            n_lags: $L$, the number of lag slots.
            source_dim: Flattened width of one lag's source vector, which the pointwise encoder
                reports as its own ``source_dim`` so the two cannot disagree.
            lag_embed_dim: Width $E$ of the lag embedding.
            hidden: Body width. ``None`` -- the default -- uses ``d_model``, which is the shape the
                architecture is specified at.
            mean_only: Build **no** scale-proposal parameters at all. The final projection emits
                $d_z$ outputs rather than $2 d_z$, so the arm is a different module tree and a
                different state dict rather than a flag consulted in the forward. That is the
                point: a scale head that exists but is never read is a starved parameter block
                under a distributed run and a claim in the manifest that the model updates a
                variance it does not.

        Raises:
            ValueError: If any width is not positive.
        """
        super().__init__()
        for name, value in (
            ("d_model", d_model),
            ("d_z", d_z),
            ("n_lags", n_lags),
            ("source_dim", source_dim),
            ("lag_embed_dim", lag_embed_dim),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")

        self.d_model = int(d_model)
        self.d_z = int(d_z)
        self.n_lags = int(n_lags)
        self.source_dim = int(source_dim)
        self.lag_embed_dim = int(lag_embed_dim)
        self.mean_only = bool(mean_only)

        body = self.d_model if hidden is None else int(hidden)
        if body <= 0:
            raise ValueError(f"hidden must be > 0 when given, got {hidden}")
        self.hidden = body

        # The lag identity. An embedding rather than a one-hot column of the input, so the head
        # learns how much lag identity matters instead of paying an L-wide input for it. Its
        # constructor draw survives the family's generic initialisation pass, which walks Linear,
        # Conv1d, LSTM and LayerNorm and leaves an Embedding alone -- so unlike the output
        # projection below, this needs no re-initialisation hook.
        self.lag_embedding = nn.Embedding(self.n_lags, self.lag_embed_dim)
        nn.init.normal_(self.lag_embedding.weight, mean=0.0, std=LAG_EMBED_STD)

        self.input_dim = self.d_model + self.source_dim + self.lag_embed_dim
        self.input_proj = nn.Linear(self.input_dim, self.hidden)
        self.hidden_proj = nn.Linear(self.hidden, self.hidden)
        self.output_proj = nn.Linear(
            self.hidden, self.d_z if self.mean_only else 2 * self.d_z
        )
        self.zero_output()

    def zero_output(self) -> None:
        """Zero the final projection, so every proposal starts at exactly zero.

        Called from the constructor and **again** from a composing model's post-initialisation
        block, because the family's generic initialisation pass xavier-fills every ``nn.Linear``
        after the modules are built -- so a constructor-only zero is silently refilled and the exact
        zero-update start goes with it. Idempotent.

        Initialisation only. Calling it on a trained model would discard the source pathway.
        """
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        target_state: torch.Tensor,
        source_window: torch.Tensor,
        *,
        lag_valid: Optional[torch.Tensor] = None,
        selector: Optional[torch.Tensor] = None,
        lag_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Produce one proposal pair per anchor and lag.

        Args:
            target_state: The anchor's conditioning state $h_t$, $(B, A, d_h)$. The same tensor the
                prior heads read, so a proposal is conditioned on the belief it is correcting.
            source_window: One lag window per anchor, $(B, A, L, \ldots)$ with any trailing shape
                whose product is :attr:`source_dim`. The gather emits $(B, A, L, C_U, W)$ and is
                flattened here rather than by the caller, so the flattening order is fixed in one
                place and the declared channel order the fusion reads is the encoder's own.
            lag_valid: $v_{t,\ell}$, $(B, A, L)$, true where the lag carries any available channel.
                ``None`` treats every lag as available, which is correct only for a caller that has
                already masked its window.
            selector: $s_{t,\ell}$, broadcastable to $(B, A, L)$. Externally set, normally one, and
                never learned. ``None`` is the unintervened forward.
            lag_index: Which lag slots this call covers, as a ``long`` tensor whose length is the
                window's lag axis. ``None`` -- the default -- covers every slot in order. Given
                only when the caller is chunking the lag axis, and it is the embedding's index
                rather than a slice of a full evaluation: a chunked call must identify slot
                $\ell$ as $\ell$, not as its position within the chunk, or every chunk but the
                first would read another slot's embedding.

        Returns:
            ``(mean_proposal, scale_proposal)``, each $(B, A, L, d_z)$, with the second ``None`` on
            the mean-only arm. Both are already multiplied by the selector and the lag validity, so
            a suppressed or empty lag contributes an exact zero rather than the head's response to
            an all-zero input vector -- which is a learned constant and not nothing.

        Raises:
            ValueError: If the target state is not 3-D, if the window's lag axis is not $L$, if its
                flattened width is not :attr:`source_dim`, or if the two disagree about the batch
                or anchor axes.
        """
        if target_state.dim() != 3:
            raise ValueError(
                f"target_state must be 3-D (B, A, d_model), got shape "
                f"{tuple(target_state.shape)}"
            )
        if target_state.shape[-1] != self.d_model:
            raise ValueError(
                f"target_state has width {target_state.shape[-1]} against d_model="
                f"{self.d_model}"
            )
        if source_window.dim() < 4:
            raise ValueError(
                f"source_window must be at least 4-D (B, A, L, ...), got shape "
                f"{tuple(source_window.shape)}"
            )
        batch, n_anchors = target_state.shape[0], target_state.shape[1]
        if source_window.shape[:2] != (batch, n_anchors):
            raise ValueError(
                f"source_window leading axes {tuple(source_window.shape[:2])} disagree with the "
                f"target state's {(batch, n_anchors)}; the two describe one anchor set."
            )
        lags = (
            torch.arange(self.n_lags, device=target_state.device)
            if lag_index is None
            else lag_index.to(device=target_state.device, dtype=torch.long)
        )
        if source_window.shape[2] != lags.numel():
            raise ValueError(
                f"source_window has {source_window.shape[2]} lags against "
                f"{lags.numel()} requested lag slots; the lag embedding is indexed by slot, so a "
                f"mismatch would identify the wrong slot rather than fail."
            )
        if bool(((lags < 0) | (lags >= self.n_lags)).any()):
            raise ValueError(
                f"lag_index has entries outside [0, n_lags) = [0, {self.n_lags}); the embedding "
                f"table covers the configured slots only."
            )
        # The chunk's own lag count, never the configured one: under lag chunking they differ, and
        # a reshape against the configured count either raises or silently re-bins the window.
        flat_source = source_window.reshape(batch, n_anchors, lags.numel(), -1)
        if flat_source.shape[-1] != self.source_dim:
            raise ValueError(
                f"source_window flattens to width {flat_source.shape[-1]} against source_dim="
                f"{self.source_dim}; that width is the encoder's own, so a mismatch means the "
                f"window came from a differently configured encoder."
            )

        # The input projection, applied per part. Identical arithmetic to projecting the
        # concatenation [h || vec(E) || zeta], and the slice order below IS that concatenation
        # order -- the one thing here a later edit could get wrong without a shape changing, which
        # is why a test asserts the two forms agree rather than trusting this comment.
        weight_state, weight_source, weight_embed = self.input_proj.weight.split(
            [self.d_model, self.source_dim, self.lag_embed_dim], dim=1
        )
        embedded = self.lag_embedding(lags)  # (L, E)

        projected = (
            F.linear(target_state, weight_state).unsqueeze(2)
            + F.linear(flat_source, weight_source)
            + F.linear(embedded, weight_embed)[None, None, :, :]
            + self.input_proj.bias
        )
        activated = F.gelu(projected)
        activated = F.gelu(self.hidden_proj(activated))
        raw = self.output_proj(activated)  # (B, A, L, d_z) or (B, A, L, 2 d_z)

        gate = None
        if lag_valid is not None:
            gate = lag_valid.to(raw.dtype)
        if selector is not None:
            selector_term = selector.to(raw.dtype)
            gate = selector_term if gate is None else gate * selector_term
        if gate is not None:
            raw = raw * gate.unsqueeze(-1)

        if self.mean_only:
            return raw, None
        mean_proposal, scale_proposal = raw.split(self.d_z, dim=-1)
        return mean_proposal, scale_proposal


def sum_proposals(proposals: torch.Tensor, *, c_lag: float) -> torch.Tensor:
    r"""Reduce a lag axis of proposals to one update: $c_L \sum_\ell r_\ell$.

    **The scale is an argument and is never derived from the input.** A version that divided by the
    number of currently available lags would make one anchor's evidence weigh differently from
    another's for a reason no parameter could absorb, and the difference would show up as a drifting
    coupling readout rather than as a failure.

    Args:
        proposals: Per-lag updates $(B, A, L, d_z)$, already gated by selector and validity.
        c_lag: $c_L$, resolved once from the configured lag count by :func:`default_lag_scale`.

    Returns:
        The scaled sum $(B, A, d_z)$.

    Raises:
        ValueError: If ``proposals`` is not 4-D.
    """
    if proposals.dim() != 4:
        raise ValueError(
            f"proposals must be 4-D (B, A, L, d_z), got shape {tuple(proposals.shape)}"
        )
    return float(c_lag) * proposals.sum(dim=2)


def bound_update(raw: torch.Tensor, limit: float) -> torch.Tensor:
    r"""Bound a summed update into $(-\text{limit}, \text{limit})$: $m \tanh(x / m)$.

    Applied **after** the sum, never per lag. The consequence is worth stating where the map is:
    the posterior mean is an additive correction to the prior, but its final correction is not a
    sum of separately bounded lag effects, so nonlinear cross-lag effects can arise here.

    Unlike a clamp, the gradient is nonzero everywhere, so a saturated coordinate can still recover.

    Args:
        raw: The summed update.
        limit: The saturation magnitude, strictly positive.

    Returns:
        A tensor shaped like ``raw``.

    Raises:
        ValueError: If ``limit`` is not positive. A zero limit is not an inert setting: it would
            pin the update at zero for every input while leaving a fully built head in the graph.
    """
    if float(limit) <= 0.0:
        raise ValueError(f"the update bound must be > 0, got {limit}")
    return float(limit) * torch.tanh(raw / float(limit))


def residual_parameters(
    mu_prior: torch.Tensor,
    logvar_prior: torch.Tensor,
    a: torch.Tensor,
    b: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Form the full distribution as a bounded residual on the prior's own parameters.

    $$\mu^q = \mu^p + \sigma^p \odot a,
      \qquad \lambda^q = \lambda^p + 2b,
      \qquad \sigma^q = \sigma^p \odot e^{b}.$$

    The mean correction is in **prior-standard-deviation units**, which is what makes the bound on
    $a$ a statement about the belief rather than about a raw coordinate scale. With bounds
    $a_{\max}$ and $b_{\max}$ this gives $|\mu^q_d - \mu^p_d| \le a_{\max}\sigma^p_d$ and
    $e^{-b_{\max}} \le \sigma^q_d / \sigma^p_d \le e^{b_{\max}}$.

    The returned log-variance is **final**. It must not be passed through the prior head's smooth
    bound again: that map is a sigmoid, it is not idempotent, and a second application would change
    this parameterisation and destroy the exact equality at a zero update.

    Both signs of $b$ are permitted. Extra evidence can increase or decrease conditional uncertainty
    for a particular observation, and useful information does not require pointwise variance
    reduction.

    Args:
        mu_prior: Prior mean $(B, A, d_z)$.
        logvar_prior: Prior log-variance $(B, A, d_z)$, already bounded by its own head.
        a: The bounded mean update $(B, A, d_z)$.
        b: The bounded scale update $(B, A, d_z)$, or ``None`` on the mean-only arm, where
            $\sigma^q = \sigma^p$ exactly.

    Returns:
        ``(mu_full, logvar_full)``, each $(B, A, d_z)$.
    """
    sigma_prior = torch.exp(0.5 * logvar_prior)
    mu_full = mu_prior + sigma_prior * a
    if b is None:
        # The same object would be tempting and is wrong: a caller may write into one of these.
        return mu_full, logvar_prior.clone()
    return mu_full, logvar_prior + 2.0 * b


def residual_kl(
    a: torch.Tensor, b: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""The divergence between the full and prior distributions, per latent coordinate.

    Substituting $\sigma^q/\sigma^p = e^{b}$ and $(\mu^q - \mu^p)/\sigma^p = a$ into the diagonal
    Gaussian divergence collapses it to

    $$K_{t,d} = \tfrac12\bigl[a_{t,d}^2 + e^{2b_{t,d}} - 1 - 2b_{t,d}\bigr],$$

    with no remaining dependence on the prior's own mean or scale. That is an arithmetic
    simplification, not a claim that the prior is fitted independently of the full distribution:
    the two share every parameter upstream of this point through the head's conditioning.

    The variance term is evaluated with ``expm1``, because $e^{2b} - 1$ loses its leading digits to
    cancellation for small $b$ -- exactly the regime a freshly initialised model spends its first
    epochs in, where the series is $2b^2 + \tfrac43 b^3 + \cdots$. The reduction runs in at least
    single precision for the same reason.

    Args:
        a: The bounded mean update $(B, A, d_z)$.
        b: The bounded scale update $(B, A, d_z)$, or ``None`` on the mean-only arm, where the
            divergence is $\tfrac12 \lVert a \rVert^2$.

    Returns:
        The per-coordinate divergence $(B, A, d_z)$, nonnegative, and exactly zero where both
        updates are zero.
    """
    dtype = torch.promote_types(a.dtype, torch.float32)
    term = a.to(dtype) ** 2
    if b is not None:
        scaled = 2.0 * b.to(dtype)
        term = term + torch.expm1(scaled) - scaled
    return 0.5 * term


def cancellation_from_parts(
    total: torch.Tensor, norm_sum: torch.Tensor, *, eps: float = CANCELLATION_EPS
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""The cancellation ratio from the two quantities a chunked forward can accumulate.

    $$\kappa_t = \frac{\lVert \sum_\ell r_{t,\ell} \rVert_2}
                      {\sum_\ell \lVert r_{t,\ell} \rVert_2 + \varepsilon}.$$

    Both parts are additive over lags, so a forward that never holds the whole proposal array can
    still report this exactly. It exists so that the chunked path and
    :func:`cancellation_ratio` are one formula rather than two that agree today: a second copy
    would be free to pick a different epsilon or a different norm and nothing would fail.

    Args:
        total: The **unscaled** summed proposal $(B, A, d_z)$, or its $c_L$-scaled version -- the
            scale is common to both parts and cancels out of the ratio.
        norm_sum: $\sum_\ell \lVert r_{t,\ell} \rVert_2$, $(B, A)$, under the same scaling.
        eps: Numerical floor on the denominator.

    Returns:
        ``(ratio, numerator, denominator)``, each $(B, A)$.
    """
    numerator = total.norm(dim=-1)
    return numerator / (norm_sum + float(eps)), numerator, norm_sum


def cancellation_ratio(
    proposals: torch.Tensor, *, eps: float = CANCELLATION_EPS
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""How much of the per-lag proposal mass survives the sum.

    $$\kappa_t = \frac{\bigl\lVert \sum_\ell r_{t,\ell} \bigr\rVert_2}
                      {\sum_\ell \lVert r_{t,\ell} \rVert_2 + \varepsilon}.$$

    **The numerator and denominator come back beside the ratio, and they have to.** A near-zero
    ratio means one of two entirely different things -- proposals that cancel, or proposals that are
    all near zero -- and the ratio alone cannot tell them apart. A reader who sees only $\kappa$
    cannot distinguish a source pathway that is arguing with itself from one that has switched off.

    This is a diagnostic, not an objective term and not an attribution. A divergence penalty on the
    summed update cannot penalise large proposals that cancel, which is the asymmetry this ratio
    exists to expose; it does not repair it.

    Args:
        proposals: Per-lag updates $(B, A, L, d_z)$, before the $c_L$ scaling, because the scale is
            common to numerator and denominator and cancels out of the ratio.
        eps: Numerical floor on the denominator. A diagnostic constant.

    Returns:
        ``(ratio, numerator, denominator)``, each $(B, A)$.

    Raises:
        ValueError: If ``proposals`` is not 4-D.
    """
    if proposals.dim() != 4:
        raise ValueError(
            f"proposals must be 4-D (B, A, L, d_z), got shape {tuple(proposals.shape)}"
        )
    return cancellation_from_parts(
        proposals.sum(dim=2), proposals.norm(dim=-1).sum(dim=2), eps=eps
    )


__all__ = [
    "CANCELLATION_EPS",
    "LAG_EMBED_STD",
    "LagProposalHead",
    "bound_update",
    "cancellation_from_parts",
    "cancellation_ratio",
    "default_lag_scale",
    "residual_kl",
    "residual_parameters",
    "sum_proposals",
]
