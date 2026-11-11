r"""Re-run the source pathway of one forward under an intervention, and score it per anchor.

The model splits into a target-only part (target encoder, prior head) and a source part (UA
adapter, lag attention, event gate, posterior head). An intervention changes only the source
part, so this module keeps the matched forward's target state and prior, re-runs the source part
with a changed input, and decodes at the latent means. Three inputs can change:

* the source values $u$ (zeroed, taken from another recording, or shifted in time);
* the validity input, which the soft mask reads in two places: the per-channel flags
  $v_{t,c} \in \{0, 1\}$ of the UA adapter and the per-key indicator $K_{t\ell}$ of the attention
  bias and of the gate's invalid-mass term;
* the event gate $g_t$, which a caller can clamp.

**Scoring.** Both branches are decoded at their latent means, $z = \mu^p$ for the base and
$z = \mu^q$ for the full branch. The per-anchor score is the model's own density on the scored
cells: the AR(1) residual likelihood of ``raw_sample_score`` with the model's cell mask and step
mask, summed over horizon and channels and not weighted by channel or horizon weights, so one
unit is one nat of the forecast block. The source gain is

$$G_a = \mathrm{NLL}^{\text{base}}_a - \mathrm{NLL}^{\text{full}}_a .$$
"""
from __future__ import annotations

from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import torch

from teb_vae.lag_attn_rws.nets.losses import raw_sample_score
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask


class Matched(NamedTuple):
    """The matched forward of one batch and everything an intervention reuses.

    Attributes:
        inputs: The forward's positional arguments ``(y_st, y_ph, u_stream, phase, stride, source_valid)``.
        out: The forward's output dictionary.
        target: The scored target ``(B, A, H, C)``.
        step_mask: The step mask ``(B, A, H)``, float.
        anchors: The anchor steps ``(B, A)``.
        anchor_valid: Which anchors are real ``(B, A)``.
        source: The UA stream after the channel gate ``(B, T, C_u)``.
        quality: The adapter's per-channel flags ``(B, T, C_u)``, float in $\{0, 1\}$.
        key_mask: The strict key mask ``(B, T, L)``, ``True`` = usable.
        lag_mask: The model's lag mask ``(T, L)``.
    """

    inputs: Tuple[Any, ...]
    out: Dict[str, torch.Tensor]
    target: torch.Tensor
    step_mask: torch.Tensor
    anchors: torch.Tensor
    anchor_valid: torch.Tensor
    source: torch.Tensor
    quality: torch.Tensor
    key_mask: torch.Tensor
    lag_mask: torch.Tensor


def matched_forward(task: Any, model: Any, batch: Any, anchor_stride: int = 5,
                    anchor_phase: int = 0) -> Matched:
    """Run the matched forward at a fixed anchor tiling and collect the reusable tensors.

    Args:
        task: The loaded task (input builder, target builder, quality weight).
        model: The network.
        batch: One batch on the model's device.
        anchor_stride: Spacing between decoded anchors, in steps.
        anchor_phase: Offset of the first anchor above the warm-up floor, in ``[0, stride)``.

    Returns:
        A :class:`Matched`.
    """
    inputs = list(task._build_forward_inputs(batch))
    inputs[3], inputs[4] = int(anchor_phase), int(anchor_stride)
    target_features, weight = task._build_raw_target(batch)
    weight = task._target_quality_weight(batch, weight)
    out = model(*inputs)
    anchors, anchor_valid = out["anchor_index"], out["anchor_valid"]
    target = model._build_forecast_target(target_features, anchors)
    step_mask, _ = forecast_mask(model.scored_weight(weight), model.geometry,
                                 coverage_floor=model.coverage_floor, anchors=anchors,
                                 anchor_valid=anchor_valid)
    u_stream, source_valid = inputs[2], inputs[5]
    source = u_stream if model.source_gate is None else model.source_gate(u_stream)
    lag_mask = model.build_lag_mask(source.shape[1], source.device)
    quality = model.source_quality_input(source_valid, source.dtype)
    key_mask = model.source_key_mask(source_valid, lag_mask, u_stream.dtype)
    return Matched(tuple(inputs), out, target, step_mask.to(target.dtype), anchors, anchor_valid,
                   source, quality, key_mask, lag_mask)


def key_mask_from_flags(model: Any, flags: torch.Tensor, lag_mask: torch.Tensor) -> torch.Tensor:
    r"""Build a key mask $K_{t\ell}$ from per-step validity on the gated channel grid.

    A step is usable when every channel flag of the step is set, and the key at lag $\ell$ of
    query $t$ reads step $t - \ell$.

    Args:
        model: The network (unused; kept so the signature matches the model's own builder).
        flags: Per-channel flags ``(B, T, C_u)`` in $\{0, 1\}$, or per-step flags ``(B, T)``.
        lag_mask: The model's lag mask ``(T, L)``.

    Returns:
        Boolean ``(B, T, L)``.
    """
    valid = flags > 0.5
    if valid.dim() == 3:
        valid = valid.all(dim=-1)
    num_lags = int(lag_mask.shape[-1])
    padded = torch.cat([valid.new_zeros(valid.shape[0], num_lags - 1), valid], dim=1)
    by_lag = padded.unfold(1, num_lags, 1).flip(-1)
    return lag_mask.to(torch.bool)[None] & by_lag


def reattend(model: Any, matched: Matched, *, source: Optional[torch.Tensor] = None,
             quality: Optional[torch.Tensor] = None, key_mask: Optional[torch.Tensor] = None,
             gate_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
             removed: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    r"""Re-run the source pathway under an intervention.

    Args:
        model: The network.
        matched: The matched forward.
        source: Replacement UA stream after the channel gate; ``None`` keeps the matched one.
        quality: Replacement adapter flags; ``None`` keeps the matched ones.
        key_mask: Replacement key mask; ``None`` keeps the matched one.
        gate_fn: Optional map applied to the learned gate $g$ ``(B, T, 1)`` before the posterior
            head uses it, for example a clamp to 0 or 1.
        removed: Optional boolean ``(B, T, L)``: keys taken out of the attention for that anchor
            (hard removal). An anchor with no key left gets its gate closed. ``None`` removes
            nothing and runs the model's own soft-mask attention.

    Returns:
        ``{'alpha', 'gate_learned', 'gate', 'mu_post', 'logvar_post', 'invalid_mass', 'h_u'}``.
    """
    out = matched.out
    source = matched.source if source is None else source
    quality = matched.quality if quality is None else quality
    key_mask = matched.key_mask if key_mask is None else key_mask
    h_u = model.encode_source_kv(source, quality)
    mu_prior = out["mu_prior"]
    query = torch.cat([mu_prior, out["logvar_prior"]], dim=-1) if model.query_uses_logvar else mu_prior
    if removed is None:
        # A model built with running-gap key removal re-attends under the matched forward's mask.
        removed = out.get("source_key_removed")
    if removed is None:
        attended = model.attend_lagged_source(model.query_proj(query), h_u, key_mask)
        alpha, heads, learned, invalid_mass = attended.alpha, attended.heads, attended.gate, attended.invalid_mass
    else:
        # The soft-mask attention of the model, with the removed keys taken out of the lag mask.
        lag_mask = matched.lag_mask.to(torch.bool)
        invalid = (lag_mask[None] & ~key_mask.to(torch.bool)).to(h_u.dtype)
        bias = invalid[:, :, None, :] * model.source_invalid_key_bias[None, None, :, None]
        kept = lag_mask[None] & ~removed
        _, alpha, heads = model.lag_attn(model.query_proj(query), h_u, kept, score_bias=bias)
        invalid_mass = (alpha * invalid[:, :, None, :]).sum(dim=-1)
        logits = model.posterior_head.event_gate_logits(heads)
        learned = torch.sigmoid(logits + model.source_invalid_gate(invalid_mass))
        learned = learned * kept.any(dim=-1, keepdim=True).to(learned.dtype)
    gate = learned if gate_fn is None else gate_fn(learned)
    mu_post, logvar_post = model.posterior_head(
        out["target_state"], heads, mu_prior, out["raw_logvar_prior"], gate=gate)
    return {"alpha": alpha, "gate_learned": learned, "gate": gate, "mu_post": mu_post,
            "logvar_post": logvar_post, "invalid_mass": invalid_mass, "h_u": h_u}


def at_anchors(values: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Gather a per-step tensor ``(B, T, ...)`` at the anchors ``(B, A)``.

    Args:
        values: A tensor whose second axis is the step axis.
        anchors: Anchor steps ``(B, A)``.

    Returns:
        ``(B, A, ...)``.
    """
    batch, count = anchors.shape
    index = anchors.reshape(batch, count, *([1] * (values.dim() - 2))).expand(batch, count, *values.shape[2:])
    return values.gather(1, index)


def block_nll(model: Any, matched: Matched, latent_mean: torch.Tensor,
              per_block: bool = False) -> torch.Tensor:
    r"""Per-anchor NLL of the forecast block, decoded at a latent mean.

    Args:
        model: The network.
        matched: The matched forward (target, masks, persistence, anchors).
        latent_mean: The latent mean per step ``(B, T, d_z)``; it is gathered at the anchors.
        per_block: Also split the sum into the scattering and the phase block.

    Returns:
        ``(B, A)`` in nats, or ``(B, A, 3)`` holding total, scattering and phase when
        ``per_block`` is set.
    """
    z = at_anchors(latent_mean, matched.anchors)
    mu, logvar = model.decoder(z, persistence=matched.out.get("persistence"))
    lk = model.forecast_likelihood_kwargs()
    score = raw_sample_score(mu, matched.target, likelihood="gaussian_nll", logvar=logvar,
                             cell_mask=lk["cell_mask"], ar_coef=lk["ar_coef"],
                             step_mask=matched.step_mask)
    score = score * matched.step_mask[..., None]
    if not per_block:
        return score.sum(dim=(-1, -2))
    n_st = 32
    return torch.stack([score.sum(dim=(-1, -2)), score[..., :n_st].sum(dim=(-1, -2)),
                        score[..., n_st:].sum(dim=(-1, -2))], dim=-1)


def anchor_kl(model: Any, matched: Matched, mu_post: torch.Tensor, logvar_post: torch.Tensor) -> torch.Tensor:
    r"""KL divergence $\mathrm{KL}(q \Vert p)$ per anchor, summed over latent dimensions.

    Args:
        model: The network.
        matched: The matched forward (prior parameters, anchors).
        mu_post: Posterior mean ``(B, T, d_z)``.
        logvar_post: Posterior log-variance ``(B, T, d_z)``.

    Returns:
        ``(B, A)`` in nats.
    """
    kld = model.kld_tensor(mu_prior=matched.out["mu_prior"], logvar_prior=matched.out["logvar_prior"],
                           mu_post=mu_post, logvar_post=logvar_post)
    return at_anchors(kld, matched.anchors).sum(dim=-1)
