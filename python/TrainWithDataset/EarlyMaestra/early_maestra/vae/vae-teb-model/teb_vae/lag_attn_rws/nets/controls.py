r"""Two source controls: a stranger's source, and no source variation at all.

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

**The second control exists because the first cannot see one specific failure.** A model whose
input adapter *announces* per-channel availability carries a pattern $m^u_{t,c}$ that is a
deterministic function of $t$ and is therefore **identical in every row of the batch**. It enters
$q(z \mid Y, U)$ and not $p(z \mid Y)$, so it can push the posterior off the prior and inflate the
coupling readout with no source information in it at all -- and no permutation of rows can remove
something every row shares. :func:`source_null_forward_outputs` replaces the source *stream* with
zeros instead, which is what floors that: see its docstring for exactly what it does and does not
establish.

The two arms share their tail -- query, attention, posterior -- and the shared half is factored
into one private function rather than written twice, because the property both arms rest on is
that everything downstream of the source state is *the model's own*, unchanged.

Free functions taking the model as their first argument, not methods: a control is something
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
from teb_vae.lag_attn_rws.nets.losses import kld_tensor, masked_source_kl
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask, kl_mask

__all__ = [
    "NoCrossGroupPartner",
    "OCCLUSION_KEYS",
    "RECOMPUTED_KEYS",
    "SOURCE_NULL_KEYS",
    "groups_can_derange",
    "make_derangement",
    "occluded_forward_outputs",
    "resolve_perm_index",
    "perm_forward_outputs",
    "source_null_forward_outputs",
    "source_null_kld",
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

#: The only keys of :func:`source_null_forward_outputs`'s result that describe the *null* pairing.
#:
#: Shorter than :data:`RECOMPUTED_KEYS` by three, and the absences are the design rather than an
#: omission. The null arm is a **KL** readout: it needs the posterior's two distribution parameters
#: and nothing else, so it draws no latent sample (``z_post``) and runs no decoder
#: (``mu_full``, ``logvar_full``). A ``torch.randn_like`` here would shift the reparameterisation
#: stream for every subsequent step of the run, which is exactly the kind of coupling that makes a
#: readout change the thing it reports on.
#:
#: The consequence is the same one the permutation control carries, and one step sharper: those
#: three keys are still present in the returned dict, holding the **matched** forward's values.
#: Scoring this dict through ``compute_loss`` would therefore return the null KL beside a
#: reconstruction term computed from the matched forecast. Read :func:`source_null_kld`, not the
#: dict, unless you want one of the three keys named here.
SOURCE_NULL_KEYS: Tuple[str, ...] = ("mu_post", "logvar_post", "attn_weights")


def _attend_and_pose(
    model, forward_outputs: Dict[str, torch.Tensor], h_u: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Run the query, the lag attention and the posterior head against a substituted source state.

    The tail both controls share, factored out so the two arms differ in **how the source state was
    produced** and in nothing else. Written twice, the arms could come to attend under different
    masks or pose different queries, and the difference between two readouts that are meant to
    bracket the same quantity would be partly an artefact of that.

    Three things it does exactly as the main forward does, each load-bearing:

    * The query is $\mu^p$, or $[\mu^p \Vert \log\sigma^{2,p}]$ under ``query_uses_logvar``.
      Mirroring the flag is not cosmetic -- the model sizes ``query_proj`` at $2 d_z$ when it is
      set, so a $\mu^p$-only query is the wrong input width and the projection raises. Both halves
      come off the prior head, so source purity holds under either arm.
    * The attention runs under **the model's own** lag mask rather than whatever
      :class:`LagCrossAttention` would build by default, so a model restricting which lags it may
      read cannot have that restriction bypassed by a control alone.
    * The posterior head is handed the matched forward's target state and prior, so the only thing
      that moved between the matched call and this one is the attended source.

    Args:
        model: The model whose modules are re-run.
        forward_outputs: The matched forward's dict; requires ``target_state``, ``mu_prior``,
            ``raw_logvar_prior`` and -- under ``query_uses_logvar`` -- ``logvar_prior``.
        h_u: The source state to attend over, ``(B, T, d_model)``.

    Returns:
        ``(alpha, mu_post, logvar_post)``: the attention weights and the posterior's two
        distribution parameters under that source state.
    """
    mu_prior = forward_outputs["mu_prior"]
    query = (
        torch.cat([mu_prior, forward_outputs["logvar_prior"]], dim=-1)
        if model.query_uses_logvar
        else mu_prior
    )
    _, alpha, attended_heads = model.lag_attn(
        model.query_proj(query),
        h_u,
        model.build_lag_mask(h_u.shape[1], h_u.device),
    )
    mu_post, logvar_post = model.posterior_head(
        forward_outputs["target_state"],
        attended_heads,
        mu_prior,
        forward_outputs["raw_logvar_prior"],
    )
    return alpha, mu_post, logvar_post


@torch.no_grad()
def perm_forward_outputs(
    model,
    forward_outputs: Dict[str, torch.Tensor],
    *,
    perm_index: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    groups: Optional[Sequence[Hashable]] = None,
    anchors: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    r"""Rebuild the posterior branch under a deranged source, reusing the computed states.

    The source path contains no batch-coupled operator on **any** ``lag_kv_source`` arm -- the
    input adapter, the causal convolutions, the LSTM, the windowed self-attention and every
    normaliser on it act per sample -- so permuting the already-computed ``source_state`` along the
    batch axis is exactly equivalent to re-encoding a permuted source stream; only the attention,
    the posterior and the full forecast need re-running. That tensor is what the model reads as
    keys and values, whichever arm built it, so this control permutes the representation the
    attention consumes rather than a deeper one it may not have.

    The attention query is rebuilt exactly as the main forward builds it -- $\mu^p$ alone, or
    $[\mu^p \Vert \log\sigma^{2,p}]$ under ``query_uses_logvar`` -- both target-only, so a
    derangement of the source must not move it.

    The permuted latent is drawn with a *fresh* $\epsilon$: the common-random-numbers pairing
    exists to keep base and full comparable at $q = p$, and the shuffled branch is scored on its
    own, not differenced sample-by-sample against the matched one.

    **The re-decode carries the matched forward's persistence input**, where the model has one. That
    tensor is the *target's* own value at the anchor, so a derangement of the source must not move
    it; decoding without it would make the shuffled forecast differ from the matched one by the
    residual as well as by the stranger's source. It is read off ``forward_outputs`` rather than
    taken as a parameter, so this function's signature does not depend on whether the model was
    built with the mechanism.

    **The decode happens at the matched forward's own anchors.** With none supplied that is the
    contiguous prefix $[0, T_{\mathrm{valid}})$, which is what a model decoding every anchor
    emits; with an anchor set it is a gather at exactly those indices, because the shuffled
    forecast is scored against the same target block and the same mask as the matched one. Two
    anchor sets would make the control's score a comparison of two different questions.

    The attention is rebuilt under **the model's own** lag mask rather than under whatever
    :class:`LagCrossAttention` would default to, so a model that restricts which lags it may read
    cannot have that restriction bypassed by the control alone -- which would show up only as a
    shuffled readout quietly computed against more source history than the matched one.

    Args:
        model: A :class:`~teb_vae.lag_attn_rws.nets.model.SeqVaeLagAttnRws`.
        forward_outputs: The dict returned by the model's forward. Requires ``target_state``,
            ``source_state``, ``mu_prior``, ``raw_logvar_prior``, and -- when the model was built
            with ``query_uses_logvar`` -- ``logvar_prior``. Its ``persistence`` entry, present only
            when the model was built with the decoder's persistence residual, is carried into the
            re-decode unchanged.
        perm_index: Optional precomputed derangement ``(B,)``; drawn if omitted.
        generator: Optional CPU generator seeding the derangement draw.
        groups: Optional group label per batch element -- the recording identifier, in an
            evaluation. The draw then pairs every target with a *different recording's* source
            rather than merely a different index, which is what the control claims to measure;
            an unshuffled loader over per-recording shards otherwise puts consecutive segments
            of one recording in one batch and pairs them with each other. Ignored when
            ``perm_index`` is supplied.
        anchors: The anchor index ``(B, A)`` the matched forward decoded at, or ``None`` for the
            contiguous prefix. Taken from ``forward_outputs['anchor_index']`` by every caller that
            has one. Its ``anchor_valid`` companion is not a parameter: padding is a property of
            the *mask*, not of the decode -- a padded slot repeats a real anchor, so decoding it
            costs one duplicated row and changes nothing -- and it reaches the objective through
            the returned dict untouched, like every other key the derangement cannot move.

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
    h_u = forward_outputs["source_state"]
    perm_index = resolve_perm_index(
        h_u.size(0), perm_index, generator, h_u.device, groups=groups
    )

    # The query, the attention under the model's own lag mask, and the posterior head: the tail
    # this arm shares with the source-null one, which is why it is one function rather than two
    # copies. What makes this arm a *permutation* is the single index below and nothing else.
    alpha_perm, mu_post_perm, logvar_post_perm = _attend_and_pose(
        model, forward_outputs, h_u[perm_index]
    )
    z_post_perm = mu_post_perm + torch.randn_like(mu_post_perm) * torch.exp(
        0.5 * logvar_post_perm
    )
    if anchors is None:
        decoded = z_post_perm[:, : model.geometry.t_valid]
    else:
        index = anchors.to(torch.long)[:, :, None].expand(-1, -1, z_post_perm.shape[-1])
        decoded = z_post_perm.gather(1, index)
    # The matched forward's own persistence tensor, resolved off the dict rather than taken as a
    # parameter, so no call site moves. It is target-only and the derangement permutes the SOURCE,
    # so the shuffled decode must carry exactly the term the matched decode carried: re-decoding
    # without it would shift the shuffle gap by the residual as well as by the stranger's source,
    # and a control whose gap moves for a non-source reason is not this control. ``None`` on a model
    # built without the residual, which is what the decoder is then also expecting.
    mu_full_perm, logvar_full_perm = model.decoder(
        decoded, persistence=forward_outputs.get("persistence")
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


@torch.no_grad()
def source_null_forward_outputs(
    model,
    forward_outputs: Dict[str, torch.Tensor],
    u_stream: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    r"""Rebuild the posterior against a source stream of zeros, re-encoding it from the boundary.

    **Why this cannot reuse the permutation control.** That function starts from
    ``forward_outputs['source_state']``, and it is *allowed* to: permuting an already-encoded
    $h_u$ along the batch axis is exactly re-encoding a permuted stream, because nothing in the
    source pathway couples batch elements. A zeroed stream is not a permutation of anything, and
    every stage of that pathway is nonlinear, so ``source_gate`` and then the model's own
    key/value path must all be re-run -- and ``u_stream`` is not in the forward dict at all, which
    is why this takes a different argument list rather than a flag.

    **The path is resolved from the model, never named here.** ``model.encode_source_kv`` is the
    same method the forward runs, so this arm re-encodes through whatever ``lag_kv_source`` built:
    the deep encoder, the local convolution stem, or the adapter alone. Naming a module instead
    would leave this control probing a tensor the attention does not read the moment the arm is
    local, and ``kld_source_null`` would describe a pathway with no consumer while every shape
    stayed correct.

    **Why the null is zeros and not a mean.** The loader z-scores each channel with constants
    accumulated *excluding* the causal warm-up region, so zero **is** the channel mean over the
    region the model actually reads. A per-batch or per-sample mean would leak the sample's own
    source statistics into the arm meant to contain none.

    **What it floors, stated exactly.** It removes all source *variation*: every batch element is
    handed the same flat trajectory, so whatever divergence survives is driven by the parts of the
    source pathway that do not depend on the data -- above all the availability announcement
    $m^u_{t,c}$, a deterministic function of $t$. It is **not** literally "the availability clock
    alone": the encoder's response to a flat trajectory is not the pattern's own response, so this
    is a slightly weaker statement and the difference from the matched readout is a slightly weaker
    attribution.

    **What this readout means depends on one model flag, and that is the point of the flag.** With
    ``prior_availability_input`` off, the clock enters $q(z \mid Y, U)$ and not $p(z \mid Y)$, so
    the surviving divergence *contains* it and this number is how large it is. With the flag on the
    prior is conditioned on **this arm's own encode** -- ``_prior_clock`` runs exactly the gate,
    the adapter and the key/value body this function runs, over exactly these zeros -- so both
    branches receive the same deterministic tensor and the divergence between them is learnable
    to zero. The same number then stops being a measurement of the clock and becomes the
    **check that the cancellation worked**, expected at $\approx 0$.

    The prior is reused from the matched forward here rather than recomputed, and under the flag
    that is what carries the clock-conditioned prior into this arm unchanged: the clock is a
    function of $t$ and of the source pathway's parameters, neither of which this intervention
    touches. Note the consequence for the readout's own honesty -- the two encodes are the same
    computation, so this control is comparing the posterior's *response* to the null source against
    the prior's, rather than comparing two different pictures of the same silence.

    **One encode, broadcast.** With $x \equiv 0$ the adapter's output is
    $W_x \mathbf 0 + W_m (m_t - \mathbf 1) + \ldots$, a function of $m_t$ alone and therefore
    identical in every batch element, so the source is encoded once at batch $1$ and expanded. The
    expansion is a view; the attention that consumes it still runs at the full batch, because the
    query is $\mu^p$ and the divergence must vary per sample as it does in the matched arm.

    **No latent sample and no decode.** The readout this exists for is a KL, which needs only the
    posterior's two distribution parameters. A ``torch.randn_like`` here would move the
    reparameterisation stream for every subsequent step of the run.

    Args:
        model: A :class:`~teb_vae.lag_attn_rws.nets.model.SeqVaeLagAttnRws` or a subclass.
        forward_outputs: The matched forward's dict. Requires ``target_state``, ``mu_prior``,
            ``raw_logvar_prior`` and -- under ``query_uses_logvar`` -- ``logvar_prior``.
        u_stream: The source stream the matched forward was given, ``(B, T, c_u)``, at the
            **declared** width, before the gate. Read for its shape, dtype and device only.

    Returns:
        A shallow copy of ``forward_outputs`` with exactly :data:`SOURCE_NULL_KEYS` replaced. Every
        other key -- the prior, both encoder states, the base forecast, and the three keys this arm
        deliberately does not rebuild -- is the matched forward's own tensor; see the constant.
    """
    zeros = u_stream.new_zeros((1, *u_stream.shape[1:]))
    gated = zeros if model.source_gate is None else model.source_gate(zeros)
    encoded = model.encode_source_kv(gated)  # (1, T, d_model)
    # expand, not repeat: the attention's projections materialise their own tensors anyway, and
    # what this arm saves is the encode -- the gate, the adapter and whichever key/value body the
    # model built -- not the attention.
    h_u_null = encoded.expand(u_stream.shape[0], -1, -1)

    alpha_null, mu_post_null, logvar_post_null = _attend_and_pose(
        model, forward_outputs, h_u_null
    )

    nulled = dict(forward_outputs)
    nulled.update(
        mu_post=mu_post_null,
        logvar_post=logvar_post_null,
        attn_weights=alpha_null,
    )
    return nulled


#: The keys :func:`occluded_forward_outputs` rebuilds, and the only ones that describe the
#: occluded arm.
#:
#: One longer than :data:`SOURCE_NULL_KEYS` and one shorter than :data:`RECOMPUTED_KEYS`, and both
#: differences are the readout this arm exists for. It **does** decode -- the quantity is a
#: forecast score resolved by horizon step, which needs the decoder's two heads -- so ``z_post``,
#: ``mu_full`` and ``logvar_full`` are rebuilt. It carries no ``perm_index``, because nothing was
#: permuted: what moved is a band of the source's own values.
#:
#: The same warning applies as to the other two arms. Every key not named here is the matched
#: forward's own tensor in the returned shallow copy, so a reader reaching for ``kld_per_t`` or
#: ``source_kl_lag_map`` on this dict gets the matched value with nothing failing.
OCCLUSION_KEYS: Tuple[str, ...] = (
    "mu_post",
    "logvar_post",
    "z_post",
    "attn_weights",
    "mu_full",
    "logvar_full",
    "source_state",
)


@torch.no_grad()
def occluded_forward_outputs(
    model,
    forward_outputs: Dict[str, torch.Tensor],
    source: torch.Tensor,
    *,
    occlusion: Optional[torch.Tensor] = None,
    anchors: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    r"""Rebuild the posterior and the forecast with part of the source's values zeroed.

    The third intervention arm, and the only **interventional** one: the permutation control asks
    what a stranger's source would have done and the null control asks what silence would have
    done, while this one removes a chosen slice of *this* source and asks what the forecast loses.
    That is the question an attention weight cannot answer -- a weight says where the model looked,
    not what looking there was worth -- and it is answered on the scale every other forecast number
    in this family is on, nats of the block score.

    **What is zeroed and why zero.** ``occlusion`` is a $(T,)$ or $(B, T)$ boolean over the source
    stream's own step axis; the stream is set to $0$ wherever it is true. Zero is the channel mean
    over the region the model reads -- the loader's constants are accumulated excluding the causal
    warm-up -- which is the same argument :func:`source_null_forward_outputs` rests on, so the
    occluded band is uninformative rather than adversarial.

    **The stream is taken after the channel gate**, which is what makes a band mean one thing. The
    gate shifts each channel onto the run's common clock, so a band of *gated* steps is a band of
    one lag range for every kept channel at once; the same band applied before the gate would land
    at $\ell + d_c$ for channel $c$ and re-smear exactly the axis the alignment exists to
    un-smear.

    **The availability announcement does not move, by construction rather than by care.** It is
    built inside the adapter from registered buffers -- a function of $t$ and the resolved warm-up
    vector -- and no value on the stream reaches it. So this intervention moves the source's
    content and leaves its clock exactly where it was, which is the confound the readout would
    otherwise measure instead of the thing it is after.

    **The path is resolved from the model.** ``model.encode_source_kv`` is the forward's own, so
    the arm re-encodes through whatever ``lag_kv_source`` built and the occluded tensor is the one
    the attention actually reads.

    **A latent is drawn, from an explicit generator.** The readout is a difference of two forecast
    scores, so the two arms must be scored under the *same* latent noise or the difference carries
    the draw as well as the band; passing one generator and reseeding it per arm is what makes the
    comparison paired. ``None`` falls back to the global stream, which is correct for a single arm
    and wrong for a comparison -- the caller that wants a difference passes one.

    Args:
        model: A :class:`~teb_vae.lag_attn_rws.nets.model.SeqVaeLagAttnRws` or a subclass.
        forward_outputs: The matched forward's dict. Requires ``target_state``, ``mu_prior``,
            ``raw_logvar_prior``, ``persistence`` where the decoder was built with the residual,
            and -- under ``query_uses_logvar`` -- ``logvar_prior``.
        source: The source stream **after** the channel gate, ``(B, T, c_kept)`` -- the same tensor
            the forward handed to ``encode_source_kv``.
        occlusion: Boolean mask over the step axis, ``(T,)`` or ``(B, T)``, true where the source
            is zeroed. ``None`` occludes nothing, which is the matched arm re-run through this
            same path and is what a difference is taken against.
        anchors: The anchor set to decode at, ``(B, A)``, in **sequence** indices. ``None`` decodes
            the contiguous prefix $[0, T_{\mathrm{valid}})$, which is almost never what a caller
            wants here. A caller narrowing this must narrow ``forward_outputs['persistence']`` to
            match: that tensor is indexed by anchor position rather than by step.
        generator: Generator for the latent draw. See above.

    Returns:
        A shallow copy of ``forward_outputs`` with exactly :data:`OCCLUSION_KEYS` replaced.
    """
    occluded = source
    if occlusion is not None:
        keep = (~occlusion.to(torch.bool)).to(source.dtype)
        occluded = source * (keep[:, :, None] if keep.dim() == 2 else keep[None, :, None])

    h_u = model.encode_source_kv(occluded)
    alpha, mu_post, logvar_post = _attend_and_pose(model, forward_outputs, h_u)

    # Drawn rather than reused from the matched forward: the posterior moved, so its own
    # reparameterisation has to be re-run against the moved parameters. What is held fixed across
    # arms is the *noise*, which is the generator's job and not this line's.
    noise = torch.randn(
        mu_post.shape, generator=generator, device=mu_post.device, dtype=mu_post.dtype
    )
    z_post = mu_post + noise * torch.exp(0.5 * logvar_post)
    if anchors is None:
        decoded = z_post[:, : model.geometry.t_valid]
    else:
        index = anchors.to(torch.long)[:, :, None].expand(-1, -1, z_post.shape[-1])
        decoded = z_post.gather(1, index)
    # The matched forward's own persistence tensor, taken as it stands, for the reason the
    # permutation control carries it: the residual is target-only and this intervention moves the
    # SOURCE, so a decode without the matched term would shift the difference for a reason that is
    # not the band. It is **not** re-gathered here, and that is the contract: ``anchors`` indexes
    # the SEQUENCE while the persistence tensor is already indexed by anchor POSITION, so the two
    # live in different spaces and a gather with one on the other reads out of bounds. A caller
    # narrowing the anchor set narrows this beside it. ``None`` on a model built without the
    # residual, which is what the decoder is then also expecting.
    mu_full, logvar_full = model.decoder(
        decoded, persistence=forward_outputs.get("persistence")
    )

    occluded_outputs = dict(forward_outputs)
    occluded_outputs.update(
        mu_post=mu_post,
        logvar_post=logvar_post,
        z_post=z_post,
        attn_weights=alpha,
        mu_full=mu_full,
        logvar_full=logvar_full,
        source_state=h_u,
    )
    return occluded_outputs


@torch.no_grad()
def source_null_kld(
    model,
    forward_outputs: Dict[str, torch.Tensor],
    u_stream: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    r"""The KL floor a source carrying no variation induces, in the coupling readout's own units.

    $$\texttt{kld\_source\_null} \;=\;
    \frac{1}{N}\sum_{b,t} m^{\mathrm{KL}}_{b,t}
    \sum_{d} \mathrm{KL}\!\left(q^{\mathrm{null}}_{b,t,d} \,\Vert\, p_{b,t,d}\right),$$

    reduced exactly as ``source_conditioned_kl_raw`` is -- summed over $d_z$, masked by the same
    $(B, T)$ anchor support, divided by the same contributing-anchor count -- because the whole
    point is to subtract one from the other. The masks are rebuilt through the objective's own two
    functions at the model's own ``coverage_floor`` and at the anchor set its forward decoded, so
    the two supports are the same set rather than two that agree by coincidence.

    $\texttt{source\_conditioned\_kl\_raw} - \texttt{kld\_source\_null}$ is the part of the
    coupling readout attributable to source *variation*. If the two are equal, the readout is
    measuring the availability clock.

    Args:
        model: The model the matched forward came from.
        forward_outputs: That forward's dict.
        u_stream: The source stream it was given, ``(B, T, c_u)``.
        weight: Decimated validity signal ``(B, T)``.

    Returns:
        A scalar tensor in nats per anchor.
    """
    nulled = source_null_forward_outputs(model, forward_outputs, u_stream)
    anchors = forward_outputs.get("anchor_index")
    anchor_valid = forward_outputs.get("anchor_valid")
    forecast, _coverage = forecast_mask(
        weight,
        model.geometry,
        coverage_floor=model.coverage_floor,
        anchors=anchors,
        anchor_valid=anchor_valid,
    )
    support = kl_mask(
        forecast, model.geometry, anchors=anchors, anchor_valid=anchor_valid
    )
    kld_btd = kld_tensor(
        mu_prior=forward_outputs["mu_prior"],
        logvar_prior=forward_outputs["logvar_prior"],
        mu_post=nulled["mu_post"],
        logvar_post=nulled["logvar_post"],
    )
    return masked_source_kl(kld_btd, support)["source_conditioned_kl_raw"]
