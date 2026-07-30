r"""The source-permutation control: what the model does when the source is the wrong one.

A high $K_t$ on its own proves nothing. The posterior sees the source, so it will react to *any*
source -- including one belonging to a different recording. The control asks the sharper question:
derange the batch so every target is paired with a stranger's source, and see what happens.

Two readouts come out of it, and they are not equally useful.

**In KL space**, $K_{\mathrm{shuffled}}$ is measured against the same prior. It answers "did the
source move my belief?", which a mismatched source does too -- often *more* strongly, since the
posterior only ever trained on matched pairs and a stranger's source is out of distribution. So
$K_{\mathrm{shuffled}} \gtrsim K_{\mathrm{true}}$ is routinely observed on models that plainly do
use the source. The raw KL comparison does **not** establish specificity, and reading it as though
it does is the mistake this module exists to make hard.

**In prediction space** the control discriminates. Feed the decoder a latent drawn from
$q(z \mid Y, \pi(U))$ and re-score against the true future: on a model genuinely exploiting the
source, that lands *above* the target-only baseline loss -- a wrong source is worse than no source
-- while the matched forecast lands well below it. That ordering is the negative control that
means something, and it needs no auxiliary loss term.

These are free functions taking the model as their first argument, not methods and not a mixin.
A mixin would be inheritance into the model, which is the exact construct this package exists
without: the control is something one *does to* a model, not something a model *is*.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import torch


class NoCrossGroupPartner(ValueError):
    """No permutation can pair every element with one from a *different* group.

    Raised rather than returned so a caller cannot mistake an impossible batch for a drawn one,
    and typed separately from the ordinary ``ValueError`` guards so a caller may catch exactly
    this case -- a batch holding too few distinct recordings -- without also swallowing a shape
    or size error.
    """


def groups_can_derange(groups: Sequence[Hashable]) -> bool:
    r"""Whether a cross-group derangement of ``groups`` exists at all.

    A permutation pairing every element with one of a different group is a perfect matching in
    the bipartite graph joining $i$ to every $j$ with $g_j \neq g_i$. By Hall's theorem such a
    matching exists **iff** no group holds more than half the batch,

    $$2 \max_g |g| \le B,$$

    since the largest group is the only set whose neighbourhood can be too small: any set
    spanning two or more groups may reach every element. The condition is therefore exact, not
    conservative -- a batch it rejects has no valid pairing at all, whatever algorithm is used.

    Args:
        groups: One group label per batch element.

    Returns:
        ``True`` when a cross-group derangement exists.
    """
    batch_size = len(groups)
    if batch_size < 2:
        return False
    return 2 * max(Counter(groups).values()) <= batch_size


def _grouped_derangement(
    groups: Sequence[Hashable], generator: Optional[torch.Generator]
) -> List[int]:
    r"""Draw a permutation pairing every element with one from a different group.

    Lay the batch out with each group's members contiguous, then rotate that layout by the
    largest group's size $m$. Two positions $k$ and $(k + m) \bmod B$ can only fall inside the
    same run of length $c \le m$ if $m < c$ (impossible) or, wrapping, if $B - m < c \le m$,
    i.e. $B < 2m$ -- which :func:`groups_can_derange` has already excluded. So the rotation is
    fixed-point-free *and* cross-group by construction, with no rejection loop.

    The layout order is randomised, so the draw is not a fixed function of the batch. It is
    **not** uniform over the valid cross-group derangements, which the control does not need: it
    needs each target paired with a stranger, not a uniformly chosen stranger.

    Args:
        groups: One group label per batch element; must satisfy :func:`groups_can_derange`.
        generator: Optional CPU generator for reproducibility.

    Returns:
        The permutation as a list, where element $i$ takes from index ``result[i]``.
    """
    members: Dict[Hashable, List[int]] = {}
    # Shuffling the indices first, then bucketing, randomises the order *within* each run; the
    # buckets themselves then appear in first-touch order, which the shuffle also randomises.
    for index in torch.randperm(len(groups), generator=generator).tolist():
        members.setdefault(groups[index], []).append(index)

    layout = [index for bucket in members.values() for index in bucket]
    shift = max(len(bucket) for bucket in members.values())
    perm = [0] * len(groups)
    for position, index in enumerate(layout):
        perm[index] = layout[(position + shift) % len(layout)]
    return perm


def make_derangement(
    batch_size: int,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    *,
    groups: Optional[Sequence[Hashable]] = None,
) -> torch.Tensor:
    r"""Draw a batch-index derangement $\pi$ with $\pi(i) \neq i$ for every $i$.

    Uses Sattolo's algorithm, which draws uniformly from the *cyclic* permutations of
    $\{0, \dots, B-1\}$. Every cyclic permutation of length $B \ge 2$ is fixed-point-free, so the
    guarantee is structural rather than probabilistic. The alternative -- rejection-sampling
    ``torch.randperm`` until no fixed point appears -- has no bound on its running time and, more
    importantly, one buggy predicate away from silently letting an identity mapping through. A
    control that is quietly not a control reports the model as source-specific when it is not.

    With ``groups`` the guarantee is strengthened from $\pi(i) \neq i$ to
    $g_{\pi(i)} \neq g_i$. That matters wherever batch neighbours are not independent: an
    unshuffled loader over per-recording shards puts consecutive segments of one recording in one
    batch, and $\pi(i) \neq i$ then happily pairs a segment with *its own recording's* next
    segment -- which is not a stranger's source, and weakens the control by an amount nothing
    reports.

    Args:
        batch_size: Batch size $B$; must be at least $2$.
        generator: Optional CPU generator for reproducibility.
        device: Device of the returned index tensor. Defaults to CPU.
        groups: Optional group label per batch element -- a recording identifier, typically.
            ``None`` draws the ungrouped Sattolo derangement, bit for bit.

    Returns:
        A ``(B,)`` long tensor holding $\pi$, with no fixed points and, under ``groups``, no
        within-group pairs.

    Raises:
        ValueError: If ``batch_size < 2``, where no derangement exists, or if ``groups`` has a
            different length.
        NoCrossGroupPartner: If ``groups`` admits no cross-group pairing at all -- one group
            holding more than half the batch, of which a single-group batch is the usual case.
    """
    if batch_size < 2:
        raise ValueError(
            f"a derangement requires batch_size >= 2, got {batch_size}; callers must "
            "skip the permutation control for degenerate batches"
        )

    if groups is not None:
        if len(groups) != batch_size:
            raise ValueError(
                f"groups must have one label per batch element: got {len(groups)} for "
                f"batch_size {batch_size}"
            )
        if not groups_can_derange(groups):
            largest, count = Counter(groups).most_common(1)[0]
            raise NoCrossGroupPartner(
                f"no cross-group derangement exists: group {largest!r} holds {count} of "
                f"{batch_size} elements, and a pairing needs every group to hold at most half. "
                f"Callers must exclude such a batch from the control and count the exclusion "
                f"rather than falling back to a within-group pairing."
            )
        return torch.tensor(
            _grouped_derangement(groups, generator), dtype=torch.long, device=device
        )

    perm = list(range(batch_size))
    for i in range(batch_size - 1, 0, -1):
        j = int(torch.randint(0, i, (1,), generator=generator).item())
        perm[i], perm[j] = perm[j], perm[i]
    return torch.tensor(perm, dtype=torch.long, device=device)


def resolve_perm_index(
    batch_size: int,
    perm_index: Optional[torch.Tensor],
    generator: Optional[torch.Generator],
    device: torch.device,
    *,
    groups: Optional[Sequence[Hashable]] = None,
) -> torch.Tensor:
    """Validate a supplied permutation index, or draw a fresh derangement.

    A *supplied* index is shape-checked and otherwise trusted, ``groups`` included: the caller
    that built it is the one that knows what it means, and re-deriving the grouping here would
    only be able to disagree with it.

    Args:
        batch_size: Expected batch size $B$.
        perm_index: A precomputed ``(B,)`` index, or ``None`` to draw one.
        generator: Optional CPU generator seeding the draw.
        device: Device for the result.
        groups: Optional group label per batch element; see :func:`make_derangement`.

    Returns:
        A ``(B,)`` long index tensor.

    Raises:
        ValueError: If a supplied index has the wrong shape, or the batch is too small.
        NoCrossGroupPartner: If ``groups`` admits no cross-group pairing.
    """
    if perm_index is None:
        return make_derangement(batch_size, generator=generator, device=device, groups=groups)
    perm_index = perm_index.to(device=device, dtype=torch.long)
    if perm_index.shape != (batch_size,):
        raise ValueError(
            f"perm_index must have shape ({batch_size},), got {tuple(perm_index.shape)}"
        )
    return perm_index


def _perm_posterior(
    model,
    h_y: torch.Tensor,
    h_u_perm: torch.Tensor,
    mu_prior: torch.Tensor,
    logvar_prior: torch.Tensor,
    raw_logvar_prior: Optional[torch.Tensor],
    detach_prior: bool,
    *,
    lag_band_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Re-run the attention and the posterior against a batch-permuted source state.

    Args:
        model: The model.
        h_y: Target state ``(B, T, d_model)``.
        h_u_perm: Batch-permuted source state ``(B, T, d_model)``.
        mu_prior: Prior mean ``(B, T, d_z)``.
        logvar_prior: Bounded prior log-variance ``(B, T, d_z)``.
        raw_logvar_prior: Pre-bound raw prior log-variance ``(B, T, d_z)``.
        detach_prior: Detach the prior before it enters either the posterior's residual base or
            the KL, so the control's gradient flows only through source, attention and posterior.
            Without it the control could be minimised by dragging the *prior* toward $q$ instead
            of by collapsing the source-driven deltas -- which would satisfy the objective while
            destroying the quantity being measured.
        lag_band_mask: Optional lag keep-mask; ``None`` is a bit-exact no-op.

    Returns:
        ``(mu_prior, logvar_prior, mu_post_perm, logvar_post_perm)``, the first two being the
        possibly-detached prior the caller must use as the KL reference.
    """
    if detach_prior:
        mu_prior = mu_prior.detach()
        logvar_prior = logvar_prior.detach()
        if raw_logvar_prior is not None:
            raw_logvar_prior = raw_logvar_prior.detach()

    m_lag, dead = model._combined_lag_mask(h_y.size(1), h_y.device, lag_band_mask)
    attended, alpha, attended_heads = model.lag_attn(h_y, h_u_perm, m_lag)
    attended, _, attended_heads = model._ablate_dead_anchors(attended, alpha, attended_heads, dead)

    posterior_source = attended_heads if model.head_structured_latent else attended
    mu_post_perm, logvar_post_perm = model.posterior_head(
        h_y, posterior_source, mu_prior, raw_logvar_prior
    )
    return mu_prior, logvar_prior, mu_post_perm, logvar_post_perm


def _perm_kl_result(
    model,
    mu_prior: torch.Tensor,
    logvar_prior: torch.Tensor,
    mu_post_perm: torch.Tensor,
    logvar_post_perm: torch.Tensor,
    perm_index: torch.Tensor,
    weight: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    r"""Reduce a permuted posterior to the control loss and its readouts.

    Args:
        model: The model.
        mu_prior: Prior mean ``(B, T, d_z)``.
        logvar_prior: Prior log-variance ``(B, T, d_z)``.
        mu_post_perm: Permuted posterior mean ``(B, T, d_z)``.
        logvar_post_perm: Permuted posterior log-variance ``(B, T, d_z)``.
        perm_index: The derangement used.
        weight: Optional per-step validity weight ``(B, T)``.

    Returns:
        ``perm_kl``, the differentiable support-masked mean; ``kld_shuffled``, its detached
        readout; ``kld_shuffled_per_t``, the ``(B, T)`` raw per-step KL under $\pi$ carrying the
        same semantics as the model's own ``kld_per_t`` so the two curves are comparable; and
        ``perm_index``.
    """
    # No free-bits floor here, deliberately: a positive floor zeroes the gradient in exactly the
    # low-KL regime the control exists to drive the model into.
    perm_kl = model._kld_loss(
        mu_prior=mu_prior,
        logvar_prior=logvar_prior,
        mu_post=mu_post_perm,
        logvar_post=logvar_post_perm,
        reduce_mean=True,
        weight=weight,
        free_bits=0.0,
    )
    with torch.no_grad():
        kld_shuffled_per_t = model.kld_tensor(
            mu_prior=mu_prior,
            logvar_prior=logvar_prior,
            mu_post=mu_post_perm,
            logvar_post=logvar_post_perm,
        ).sum(dim=-1)
    return {
        "perm_kl": perm_kl,
        "kld_shuffled": perm_kl.detach(),
        "kld_shuffled_per_t": kld_shuffled_per_t,
        "perm_index": perm_index,
    }


def permutation_kl(
    model,
    y_st: torch.Tensor,
    y_ph: torch.Tensor,
    u_stream: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    perm_index: Optional[torch.Tensor] = None,
    detach_prior: bool = True,
) -> Dict[str, torch.Tensor]:
    r"""The permutation control, re-encoding the deranged source from scratch.

    Deranges the batch on the source stream, re-encodes $\pi(U)$ through the adapter and encoder,
    and returns the support-masked mean

    $$L_{\mathrm{perm}} = \mathrm{KL}\!\left(q\big(z \mid Y, \pi(U)\big) \,\big\|\,
    p\big(z \mid Y\big)\right)$$

    over the same time support as the true KL, so the two are directly comparable.

    This is the diagnostic entry point. Training uses :func:`perm_kl_from_forward`, which is
    equivalent and much cheaper.

    Args:
        model: The model.
        y_st: Target scattering features ``(B, T, 43)``.
        y_ph: Target phase-harmonic features ``(B, T, 66)``.
        u_stream: Source stream ``(B, T, c_u)``.
        weight: Optional per-step validity weight ``(B, T)``.
        generator: Optional CPU generator seeding the derangement.
        perm_index: Optional precomputed derangement ``(B,)``; drawn if omitted.
        detach_prior: See :func:`_perm_posterior`.

    Returns:
        The control dict; see :func:`_perm_kl_result`.

    Raises:
        ValueError: If the batch is too small to derange.
    """
    perm_index = resolve_perm_index(y_st.size(0), perm_index, generator, y_st.device)

    target = torch.cat([y_st, y_ph], dim=-1)
    h_y = model.target_encoder(model.target_adapter(target))
    h_u_perm = model.source_encoder(model.source_adapter(u_stream[perm_index]))
    mu_prior, logvar_prior, _, raw_logvar_prior = model.prior_head(h_y)

    prior_mu, prior_logvar, post_mu, post_logvar = _perm_posterior(
        model, h_y, h_u_perm, mu_prior, logvar_prior, raw_logvar_prior, detach_prior
    )
    return _perm_kl_result(model, prior_mu, prior_logvar, post_mu, post_logvar, perm_index, weight)


def perm_kl_from_forward(
    model,
    forward_outputs: Dict[str, torch.Tensor],
    *,
    weight: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    perm_index: Optional[torch.Tensor] = None,
    detach_prior: bool = True,
) -> Dict[str, torch.Tensor]:
    r"""The permutation control, reusing the states a completed forward already computed.

    The source path contains no batch-coupled operator -- only causal convolutions, an LSTM and
    LayerNorm -- so it acts on each batch element independently and

    $$\mathrm{Encoder}\big(\mathrm{Adapter}(\pi(U))\big)_i \;=\; H^u[\pi(i)].$$

    Permuting the already-computed source state along the batch axis is therefore *exactly*
    equivalent to re-encoding a permuted source, and only the attention and the posterior need
    re-running.

    That equivalence is not merely an optimisation. It keeps the whole control inside the single
    main forward and backward, which is what lets the model keep automatic optimisation: under
    DDP a parameter used twice in one graph shares one accumulation node and is marked ready
    exactly once, whereas a second backward would either deadlock or force
    ``find_unused_parameters``. Gradient clipping, accumulation, LR scheduling and the spike
    breaker all ride on that.

    Args:
        model: The model.
        forward_outputs: The dict returned by the model's forward. Requires ``target_state``,
            ``source_state``, ``mu_prior``, ``logvar_prior`` and ``raw_logvar_prior``.
        weight: Optional per-step validity weight ``(B, T)``.
        generator: Optional CPU generator seeding the derangement.
        perm_index: Optional precomputed derangement ``(B,)``; drawn if omitted.
        detach_prior: See :func:`_perm_posterior`.

    Returns:
        The control dict; see :func:`_perm_kl_result`.

    Raises:
        ValueError: If the batch is too small to derange.
    """
    h_y = forward_outputs["target_state"]
    h_u = forward_outputs["source_state"]
    perm_index = resolve_perm_index(h_u.size(0), perm_index, generator, h_u.device)

    prior_mu, prior_logvar, post_mu, post_logvar = _perm_posterior(
        model,
        h_y,
        h_u[perm_index],
        forward_outputs["mu_prior"],
        forward_outputs["logvar_prior"],
        forward_outputs.get("raw_logvar_prior"),
        detach_prior,
    )
    return _perm_kl_result(model, prior_mu, prior_logvar, post_mu, post_logvar, perm_index, weight)


@torch.no_grad()
def perm_forward_outputs(
    model,
    forward_outputs: Dict[str, torch.Tensor],
    *,
    perm_index: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    r"""Rebuild the forecast under a deranged source: the prediction-space control.

    This is the control that discriminates. See the module docstring for why the KL-space one
    does not: a mismatched source moves the posterior at least as much as the right one, so
    $K_{\mathrm{shuffled}} \gtrsim K_{\mathrm{true}}$ says nothing. Re-scoring the *forecast*
    does -- a wrong source should be worse than no source at all.

    Only the source-dependent tensors are recomputed; the encoders, the prior and the
    target-only baseline decoder are reused untouched.

    Args:
        model: The model.
        forward_outputs: The dict returned by the model's forward.
        perm_index: Optional precomputed derangement ``(B,)``; drawn if omitted.
        generator: Optional CPU generator seeding the derangement.

    Returns:
        A shallow copy of ``forward_outputs`` with the posterior, the latent and the full
        forecast recomputed under $\pi(U)$, plus the ``perm_index`` used. Suitable to hand
        straight to the model's ``compute_loss`` with ``compute_kld_loss=False``.

    Raises:
        ValueError: If the batch is too small to derange.
    """
    h_y = forward_outputs["target_state"]
    h_u = forward_outputs["source_state"]
    perm_index = resolve_perm_index(h_u.size(0), perm_index, generator, h_u.device)

    _, _, mu_post_perm, logvar_post_perm = _perm_posterior(
        model,
        h_y,
        h_u[perm_index],
        forward_outputs["mu_prior"],
        forward_outputs["logvar_prior"],
        forward_outputs.get("raw_logvar_prior"),
        detach_prior=True,
    )
    z_perm = model.reparameterize(mu_post_perm, logvar_post_perm)
    delta_mu_perm, logvar_full_perm = model.residual_decoder(
        forward_outputs["decoder_state"], z_perm
    )

    permuted = dict(forward_outputs)
    permuted.update(
        mu_post=mu_post_perm,
        logvar_post=logvar_post_perm,
        z=z_perm,
        delta_mu_src=delta_mu_perm,
        mu_full=forward_outputs["mu_base"] + delta_mu_perm,
        logvar_full=logvar_full_perm,
        perm_index=perm_index,
    )
    return permuted
