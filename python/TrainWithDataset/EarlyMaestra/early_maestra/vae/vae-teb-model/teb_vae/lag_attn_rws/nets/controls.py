r"""The source-permutation control: what the forecast does when the source is a stranger's.

A nonzero source-conditioned KL on its own proves nothing about specificity. The posterior sees
the source, so it reacts to *any* source -- including one belonging to a different recording,
which is out of distribution for a posterior trained only on matched pairs and routinely moves it
*more*. The control that discriminates lives in prediction space: derange the batch so every
target is paired with a stranger's source, rebuild the full forecast, and re-score it against the
true raw future. A model genuinely exploiting the source has

$$D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}},$$

a wrong source being worse than no source at all -- which is also the second acceptance ordering
the model is judged by.

The derangement machinery is the sibling's, imported rather than restated; what is local is the
rebuild against *this* model's forward contract: a single shared latent, one shared decoder
invoked on $z$ and nothing else, so only the posterior, its sample and the full forecast are
source-driven. Everything else -- both encoders, the prior, the base forecast -- is reused
untouched, which is what makes "the base branch is bitwise identical under permutation" a
checkable property rather than a hope.

Free functions taking the model as their first argument, not methods: the control is something
one *does to* a model, not something a model *is*.
"""
from __future__ import annotations

from typing import Dict, Hashable, Optional, Sequence, Tuple

import torch

from teb_vae.lag_attn.nets.controls import (
    NoCrossGroupPartner,
    groups_can_derange,
    make_derangement,
    resolve_perm_index,
)

__all__ = [
    "NoCrossGroupPartner",
    "RECOMPUTED_KEYS",
    "groups_can_derange",
    "make_derangement",
    "resolve_perm_index",
    "perm_forward_outputs",
]

#: The only keys of :func:`perm_forward_outputs`'s result that describe the *permuted* pairing.
#:
#: The result is a shallow copy, so every other key is the matched forward's own tensor -- the same
#: object, not a stale copy of it. That is deliberate and cheap: the prior, both encoder states and
#: the base forecast are source-free, so a derangement cannot move them. But it means a reader that
#: reaches for ``kld_per_t`` or ``source_kl_lag_map`` on this dict gets the **matched** value with
#: nothing failing, and reports it as the shuffled control's.
#:
#: So the set is named here, beside the function that decides it, and a consumer that needs a
#: shuffled quantity not on this list recomputes it from these keys rather than reading it.
RECOMPUTED_KEYS: Tuple[str, ...] = (
    "mu_post",
    "logvar_post",
    "z_post",
    "attn_weights",
    "mu_full",
    "logvar_full",
    "perm_index",
)


@torch.no_grad()
def perm_forward_outputs(
    model,
    forward_outputs: Dict[str, torch.Tensor],
    *,
    perm_index: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    groups: Optional[Sequence[Hashable]] = None,
) -> Dict[str, torch.Tensor]:
    r"""Rebuild the posterior branch under a deranged source, reusing the computed states.

    The source path (adapter, causal convolutions, LSTM, LayerNorm) contains no batch-coupled
    operator, so permuting the already-computed ``source_state`` along the batch axis is exactly
    equivalent to re-encoding a permuted source stream; only the attention, the posterior and the
    full forecast need re-running. The attention query is rebuilt exactly as the main forward
    builds it -- $\mu^p$ alone, or $[\mu^p \Vert \log\sigma^{2,p}]$ under ``query_uses_logvar`` --
    both target-only, so a derangement of the source must not move it.

    The permuted latent is drawn with a *fresh* $\epsilon$: the common-random-numbers pairing
    exists to keep base and full comparable at $q = p$, and the shuffled branch is scored on its
    own, not differenced sample-by-sample against the matched one.

    Args:
        model: A :class:`~teb_vae.lag_attn_rws.nets.model.SeqVaeLagAttnRws`.
        forward_outputs: The dict returned by the model's forward. Requires ``target_state``,
            ``source_state``, ``mu_prior``, ``raw_logvar_prior``, and -- when the model was built
            with ``query_uses_logvar`` -- ``logvar_prior``.
        perm_index: Optional precomputed derangement ``(B,)``; drawn if omitted.
        generator: Optional CPU generator seeding the derangement draw.
        groups: Optional group label per batch element -- the recording identifier, in an
            evaluation. The draw then pairs every target with a *different recording's* source
            rather than merely a different index, which is what the control claims to measure;
            an unshuffled loader over per-recording shards otherwise puts consecutive segments
            of one recording in one batch and pairs them with each other. Ignored when
            ``perm_index`` is supplied.

    Returns:
        A shallow copy of ``forward_outputs`` with exactly :data:`RECOMPUTED_KEYS` replaced --
        ``mu_post``, ``logvar_post``, ``z_post``, ``attn_weights``, ``mu_full`` and
        ``logvar_full`` recomputed under $\pi(U)$, plus the ``perm_index`` used. Anything not on
        that list is the matched forward's own tensor and must not be read as the control's; see
        the constant. The base branch (``mu_base``, ``logvar_base``) and the prior are the
        *same tensors* as the input's -- source-free quantities are untouched by construction.
        The KL analysis keys (``kld_per_t``, ``source_kl_lag_map``, ``kld_per_t_per_head``) keep
        their matched-pair values; ``compute_loss`` recomputes the KL from the distribution
        parameters, so scoring this dict yields the *shuffled* KL, not the stale one.

    Raises:
        ValueError: If the batch is too small to derange ($B < 2$).
        NoCrossGroupPartner: If ``groups`` admits no cross-group pairing -- one group holding
            more than half the batch.
    """
    h_y = forward_outputs["target_state"]
    h_u = forward_outputs["source_state"]
    perm_index = resolve_perm_index(
        h_u.size(0), perm_index, generator, h_u.device, groups=groups
    )

    # Rebuild the attention query exactly as the main forward does: mu^p alone, or
    # [mu^p || logvar^p] under query_uses_logvar. Mirroring the flag is load-bearing -- the model
    # sizes query_proj at 2*d_z when the flag is on, so a mu^p-only query is the wrong input width
    # and the projection raises. Both parts are read off the prior head, so source purity holds.
    mu_prior = forward_outputs["mu_prior"]
    query = (
        torch.cat([mu_prior, forward_outputs["logvar_prior"]], dim=-1)
        if model.query_uses_logvar
        else mu_prior
    )
    _, alpha_perm, attended_heads_perm = model.lag_attn(
        model.query_proj(query), h_u[perm_index]
    )
    mu_post_perm, logvar_post_perm = model.posterior_head(
        h_y, attended_heads_perm, mu_prior, forward_outputs["raw_logvar_prior"]
    )
    z_post_perm = mu_post_perm + torch.randn_like(mu_post_perm) * torch.exp(
        0.5 * logvar_post_perm
    )
    mu_full_perm, logvar_full_perm = model.decoder(
        z_post_perm[:, : model.geometry.t_valid]
    )

    permuted = dict(forward_outputs)
    permuted.update(
        mu_post=mu_post_perm,
        logvar_post=logvar_post_perm,
        z_post=z_post_perm,
        attn_weights=alpha_perm,
        mu_full=mu_full_perm,
        logvar_full=logvar_full_perm,
        perm_index=perm_index,
    )
    return permuted
