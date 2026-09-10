r"""The training objective, and the one thing this package owns rather than imports: its reduction.

Every per-element and per-anchor term here is the family's, reached by import, because two
architectures are only comparable if they optimise the same thing. What is written here is how those
per-anchor numbers become one scalar, and that differs from the family's in two ways that are
requirements rather than preferences.

**The mean is over the global contributing-anchor count.** The shared reduction divides a rank's own
numerator by a rank's own denominator, and distributed gradient averaging then optimises the
*average of per-rank means*. Those coincide only when every rank scores the same number of anchors,
which is exactly what an availability-driven mask does not guarantee. The local numerator is scaled
here so that averaging over the world reproduces

$$\mathcal L = \frac{1}{N_{\mathrm{global}}} \sum_{\text{ranks}} \sum_{b,a} \ell_{b,a},$$

and the reported value is that same global mean rather than a rank's own.

**No denominator is clamped to one.** The shared reduction floors its anchor count at one, so a
batch that scored nothing reports its numerator as though one anchor had produced it. Here an empty
support returns a graph-connected zero and records a scored-anchor count of zero, so a step that
measured nothing says so and cannot be averaged into an epoch as if it had measured something.

**The divergence support is the reconstruction's, and both are anchor-indexed.** The family
scatters its support back to a dense stored grid because its latent tensors are produced at every
step. This architecture's latents are produced at the decoded anchors only, so the support stays on
the anchor axis and the scatter never happens. That is one fewer conversion between two axes that
are easy to confuse, and it is why the divergence keys this module reads are named for the anchor
axis they carry.

**What the weighted score is and is not.** With channel or horizon weights the per-anchor term is
$w_c w_\tau \cdot (-\log p)$, which is a composite training criterion and not a log density in nats.
A weighted run's reconstruction columns are therefore not comparable in nats with an unweighted
run's, and $\beta = 1$ is no longer an exact bound on anything. The weights normalise to leave the
block's magnitude alone, so what each states is a distribution over its own axis.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist

from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry
from teb_vae.lag_attn_rws.nets.losses import (
    KLD_ACTIVE_EPS,
    LOGVAR_FLOOR_MARGIN_FRAC,
    masked_raw_block_per_anchor,
)
from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors, forecast_mask
from teb_vae.lag_attn.nets.blocks import validate_choice

#: The likelihoods the reconstruction term admits, matching the family's.
LIKELIHOOD_CHOICES: Tuple[str, ...] = ("mse", "gaussian_nll")

#: Order of the quantities reduced in the single collective this objective performs. Named once so
#: the pack and the unpack cannot drift; a mismatch would silently report one metric under
#: another's name.
REDUCED_TERMS: Tuple[str, ...] = (
    "n_anchors",
    "nll_full",
    "nll_base",
    "kl_train",
    "kl_raw",
    "prior_rate",
    "coverage",
    "n_coverage",
)


def _all_reduce_sum(values: torch.Tensor) -> torch.Tensor:
    """Sum a small vector across the process group, or return it unchanged on one process.

    One collective per step, on one packed vector, rather than one per quantity: the sums are all
    needed at the same point and each separate call would be its own synchronisation.

    Args:
        values: The packed local sums.

    Returns:
        The global sums, as a new tensor.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return values
    reduced = values.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def compute_residual_objective(
    forward_outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    weight: torch.Tensor,
    geometry: TrimmedRawGeometry,
    block_width: int,
    coverage_floor: float,
    logvar_clamp: Tuple[float, float],
    channel_weight: Optional[torch.Tensor] = None,
    horizon_weight: Optional[torch.Tensor] = None,
    beta: float = 1.0,
    beta_prior: float = 0.0,
    lambda_full: float = 1.0,
    lambda_base: float = 1.0,
    likelihood: str = "gaussian_nll",
    free_bits: float = 0.0,
) -> Dict[str, Any]:
    r"""Compute the four-term objective, per anchor, over the global contributing-anchor set.

    $$\mathcal L = \lambda_q D_q + \lambda_p D_p + \beta K + \beta_p R_p,$$

    with $D_q$ and $D_p$ the masked block scores of the full and base forecasts, $K$ the
    residual-form divergence under its free-bits floor, and $R_p$ the prior's scale rate. Four
    terms and not seven: the family's three shape terms read a block's last axis as a trajectory,
    and here that axis counts channels, which have no order and no continuity with anything.

    Args:
        forward_outputs: The model's forward dict. Requires the two forecasts, the two latent
            parameter sets, the per-coordinate divergence and the anchor set.
        target: The gathered forecast target $(B, A, H, C_{\mathrm{keep}})$, at the same anchors
            the forecasts were emitted for.
        weight: Decimated validity signal $(B, T)$, already pooled onto the scored clock by the
            caller.
        geometry: The model's trimmed-grid geometry.
        block_width: What the target's last axis counts. Used only by the log-variance
            diagnostics, so a wrong value changes no gradient and rescales exactly those numbers.
        coverage_floor: Minimum valid fraction of an anchor's forecast window.
        logvar_clamp: The bound the log-variance diagnostics are read against.
        channel_weight: Per-channel weight on the block's last axis, or ``None`` for the uniform
            objective.
        horizon_weight: Per-step weight on the horizon axis, or ``None``.
        beta: Weight on the divergence.
        beta_prior: Weight on the prior scale rate.
        lambda_full: Weight on the full-forecast reconstruction.
        lambda_base: Weight on the base-forecast reconstruction.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.
        free_bits: Per-coordinate floor applied to the divergence before it enters the loss. It
            makes the trained divergence exceed the raw one by construction, which is why the two
            are reported separately and only the raw one may be read as a rate.

    Returns:
        ``{'metrics': ..., 'likelihood': ...}``. Every metric is a scalar tensor, so the mapping is
        safe to splat into a logger; the likelihood name stays outside it.

    Raises:
        ValueError: On an unknown ``likelihood``, or if the target's anchor axis disagrees with the
            anchor set the forward decoded.
    """
    validate_choice(likelihood, LIKELIHOOD_CHOICES, "likelihood")
    device, dtype = target.device, target.dtype
    anchors = forward_outputs["anchor_index"]
    anchor_valid = forward_outputs["anchor_valid"]
    if target.shape[:2] != anchors.shape:
        raise ValueError(
            f"target carries anchor axis {tuple(target.shape[:2])} against the decoded anchor set "
            f"{tuple(anchors.shape)}; the two must be one set, or the forecast is scored against "
            f"another anchor's future."
        )

    mask, coverage_frac = forecast_mask(
        weight,
        geometry,
        coverage_floor=coverage_floor,
        anchors=anchors,
        anchor_valid=anchor_valid,
    )
    # The divergence support **is** the reconstruction's contributing set, on the anchor axis. No
    # scatter to a dense grid: the latents exist at these anchors and nowhere else, so there is no
    # second axis to reconcile.
    support = contributing_anchors(mask)  # (B, A)

    full_block, _ = masked_raw_block_per_anchor(
        forward_outputs["mu_full"],
        target,
        mask,
        likelihood=likelihood,
        logvar=forward_outputs["logvar_full"],
        channel_weight=channel_weight,
        horizon_weight=horizon_weight,
    )
    base_block, _ = masked_raw_block_per_anchor(
        forward_outputs["mu_base"],
        target,
        mask,
        likelihood=likelihood,
        logvar=forward_outputs["logvar_base"],
        channel_weight=channel_weight,
        horizon_weight=horizon_weight,
    )

    kld_dim = forward_outputs["kld_per_anchor_dim"]
    kld_train_dim = kld_dim.clamp(min=float(free_bits)) if free_bits > 0.0 else kld_dim
    kl_train_anchor = kld_train_dim.sum(dim=-1) * support
    kl_raw_anchor = kld_dim.sum(dim=-1) * support

    logvar_prior = forward_outputs["logvar_prior"]
    prior_rate_anchor = (
        0.5 * (logvar_prior.exp() - 1.0 - logvar_prior)
    ).sum(dim=-1) * support

    # The single collective. Packed in the order :data:`REDUCED_TERMS` names, under no_grad,
    # because what crosses the wire is the denominator and the reported values -- never a gradient.
    with torch.no_grad():
        counted = (anchors >= geometry.warmup).to(dtype) * anchor_valid.to(dtype)
        local = torch.stack(
            [
                support.sum(),
                full_block.sum(),
                base_block.sum(),
                kl_train_anchor.sum(),
                kl_raw_anchor.sum(),
                prior_rate_anchor.sum(),
                (coverage_frac * counted).sum(),
                counted.sum(),
            ]
        ).to(device=device, dtype=torch.float64)
        totals = _all_reduce_sum(local)
    global_n = float(totals[REDUCED_TERMS.index("n_anchors")])

    if global_n <= 0.0:
        # Graph-connected through **every** term, not only the reconstruction. A rank with no
        # scored anchor must still reach each head in its backward, or a distributed run refuses
        # the step for a parameter that received no gradient -- and the batch that triggers it is
        # whichever one happened to land entirely inside a signal gap. Zero rather than a clamped
        # denominator: nothing was measured, and the metric surface says so.
        zero = (
            full_block.sum()
            + base_block.sum()
            + kl_train_anchor.sum()
            + prior_rate_anchor.sum()
        ) * 0.0
        return {
            "metrics": _empty_metrics(zero, beta, beta_prior, device, dtype),
            "likelihood": likelihood,
        }

    # The gradient scaling. Distributed gradient averaging divides by the world size, so each rank
    # multiplies its own numerator by that same factor: the averaged gradient is then the gradient
    # of the global mean rather than of the mean of per-rank means. On one process the factor is
    # one and this is the plain global mean.
    world = float(dist.get_world_size()) if dist.is_available() and dist.is_initialized() else 1.0
    scale = world / global_n

    nll_full_block = full_block.sum() * scale
    nll_base_block = base_block.sum() * scale
    kl_train = kl_train_anchor.sum() * scale
    prior_rate = prior_rate_anchor.sum() * scale

    total_loss = (
        lambda_full * nll_full_block
        + lambda_base * nll_base_block
        + beta * kl_train
        + beta_prior * prior_rate
    )

    metrics = _reported_metrics(
        forward_outputs=forward_outputs,
        totals=totals,
        global_n=global_n,
        mask=mask,
        support=support,
        block_width=block_width,
        logvar_clamp=logvar_clamp,
        kld_dim=kld_dim,
        beta=beta,
        beta_prior=beta_prior,
        device=device,
        dtype=dtype,
    )
    metrics["total_loss"] = total_loss
    return {"metrics": metrics, "likelihood": likelihood}


def _empty_metrics(
    zero: torch.Tensor,
    beta: float,
    beta_prior: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """The metric surface of a step that scored nothing, with the loss still in the graph.

    Every reported quantity is zero and ``scored_anchors`` says why, so an epoch aggregate cannot
    silently absorb a step that measured nothing as though it had measured zero nats.

    Args:
        zero: A graph-connected zero, so the backward still reaches every parameter.
        beta: The divergence weight, echoed.
        beta_prior: The prior-rate weight, echoed.
        device: Device for the echoed constants.
        dtype: Dtype for the echoed constants.

    Returns:
        The metric mapping.
    """
    flat = torch.zeros((), device=device, dtype=dtype)
    metrics = {name: flat.clone() for name in _METRIC_NAMES}
    metrics["total_loss"] = zero
    metrics["kld_beta"] = torch.tensor(float(beta), device=device, dtype=dtype)
    metrics["beta_prior"] = torch.tensor(float(beta_prior), device=device, dtype=dtype)
    return metrics


def _reported_metrics(
    *,
    forward_outputs: Dict[str, torch.Tensor],
    totals: torch.Tensor,
    global_n: float,
    mask: torch.Tensor,
    support: torch.Tensor,
    block_width: int,
    logvar_clamp: Tuple[float, float],
    kld_dim: torch.Tensor,
    beta: float,
    beta_prior: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    r"""Assemble the reported surface from the global sums and the local diagnostics.

    The four reconstruction and divergence columns are **global** means, so every rank of a
    distributed run reports one number rather than its own share. The distribution diagnostics
    below them are local: they describe this rank's own batch, which is what a per-step diagnostic
    is for, and aggregating them across ranks would cost a second collective for numbers no
    decision reads at that precision.

    Args:
        forward_outputs: The model's forward dict.
        totals: The reduced sums, in :data:`REDUCED_TERMS` order.
        global_n: The global contributing-anchor count, known positive.
        mask: The forecast mask $(B, A, H)$.
        support: The contributing-anchor indicator $(B, A)$.
        block_width: What the block's last axis counts.
        logvar_clamp: The bound the log-variance diagnostics are read against.
        kld_dim: The per-coordinate divergence $(B, A, d_z)$.
        beta: The divergence weight, echoed.
        beta_prior: The prior-rate weight, echoed.
        device: Device for the echoed constants.
        dtype: Dtype for the echoed constants.

    Returns:
        The metric mapping, without ``total_loss``, which the caller adds from the graph.
    """
    index = {name: position for position, name in enumerate(REDUCED_TERMS)}

    def global_mean(name: str) -> torch.Tensor:
        """One reduced sum divided by the global anchor count, as a scalar tensor."""
        return torch.tensor(
            float(totals[index[name]]) / global_n, device=device, dtype=dtype
        )

    with torch.no_grad():
        nll_full = global_mean("nll_full")
        nll_base = global_mean("nll_base")
        block_elements = float(mask.shape[-1] * block_width)

        lo, hi = logvar_clamp
        bound_margin = LOGVAR_FLOOR_MARGIN_FRAC * (hi - lo)
        elem_mask = mask[..., None]
        elem_denom = (elem_mask.sum() * float(block_width)).clamp_min(1.0)
        logvar_full = forward_outputs["logvar_full"]

        # The prior-variance floor watch, on the divergence's own support. The divergence carries
        # the squared mean gap divided by the prior variance, so a prior pinned at its lower bound
        # inflates the coupling readout by orders of magnitude while every decoder-side number
        # still looks healthy.
        active = support > 0
        if bool(active.any()):
            prior_masked = forward_outputs["logvar_prior"][active]
            mean_logvar_prior = prior_masked.mean()
            logvar_prior_floor_frac = (
                (prior_masked <= lo + bound_margin).to(dtype).mean()
            )
            mean_logvar_post = forward_outputs["logvar_post"][active].mean()
            delta_mu_rms = (
                (forward_outputs["mu_post"] - forward_outputs["mu_prior"])[active]
                .pow(2)
                .mean()
                .sqrt()
            )
            dim_mean = kld_dim[active].mean(dim=0)
            kld_active_frac = (dim_mean > KLD_ACTIVE_EPS).to(dtype).mean()
        else:
            flat = torch.zeros((), device=device, dtype=dtype)
            mean_logvar_prior = flat.clone()
            logvar_prior_floor_frac = flat.clone()
            mean_logvar_post = flat.clone()
            delta_mu_rms = flat.clone()
            kld_active_frac = flat.clone()

        coverage_denominator = float(totals[index["n_coverage"]])
        metrics = {
            "nll_full_block": nll_full,
            "nll_base_block": nll_base,
            # A fixed rescaling of the block score by the coefficient count, not a weighted mean:
            # it stays comparable with the block column whatever the mask density.
            "nll_full_sample": nll_full / block_elements,
            "nll_base_sample": nll_base / block_elements,
            "pred_gap": nll_base - nll_full,
            "source_conditioned_kl_train": global_mean("kl_train"),
            "source_conditioned_kl_raw": global_mean("kl_raw"),
            "prior_rate": global_mean("prior_rate"),
            "kld_active_frac": kld_active_frac,
            # The two numbers that make a nats-per-anchor column readable: how many anchors were
            # scored and how many coefficients each carried. Reported rather than assumed, because
            # partial coverage is never silently rescaled to a complete block.
            "scored_anchors": torch.tensor(global_n, device=device, dtype=dtype),
            "scored_coefficients": torch.tensor(
                block_elements, device=device, dtype=dtype
            ),
            "mask_coverage_frac": (mask.sum() / mask.numel()).to(dtype),
            "anchor_coverage_frac": torch.tensor(
                float(totals[index["coverage"]]) / max(coverage_denominator, 1.0),
                device=device,
                dtype=dtype,
            ),
            "mean_logvar_full": (logvar_full * elem_mask).sum() / elem_denom,
            "mean_logvar_base": (
                forward_outputs["logvar_base"] * elem_mask
            ).sum() / elem_denom,
            "logvar_full_floor_frac": (
                (logvar_full <= lo + bound_margin).to(dtype) * elem_mask
            ).sum() / elem_denom,
            "logvar_full_ceil_frac": (
                (logvar_full >= hi - bound_margin).to(dtype) * elem_mask
            ).sum() / elem_denom,
            "mean_logvar_prior": mean_logvar_prior,
            "mean_logvar_post": mean_logvar_post,
            "logvar_prior_floor_frac": logvar_prior_floor_frac,
            "delta_mu_rms": delta_mu_rms,
            "kld_beta": torch.tensor(float(beta), device=device, dtype=dtype),
            "beta_prior": torch.tensor(float(beta_prior), device=device, dtype=dtype),
        }
    return metrics


#: Every metric name the surface carries besides ``total_loss``, so an empty step reports the same
#: columns as a full one. Derived from a single assembled dict would be circular; written out, it is
#: checked against a real call by the objective's own test.
_METRIC_NAMES: Tuple[str, ...] = (
    "nll_full_block",
    "nll_base_block",
    "nll_full_sample",
    "nll_base_sample",
    "pred_gap",
    "source_conditioned_kl_train",
    "source_conditioned_kl_raw",
    "prior_rate",
    "kld_active_frac",
    "scored_anchors",
    "scored_coefficients",
    "mask_coverage_frac",
    "anchor_coverage_frac",
    "mean_logvar_full",
    "mean_logvar_base",
    "logvar_full_floor_frac",
    "logvar_full_ceil_frac",
    "mean_logvar_prior",
    "mean_logvar_post",
    "logvar_prior_floor_frac",
    "delta_mu_rms",
    "kld_beta",
    "beta_prior",
)


__all__ = [
    "LIKELIHOOD_CHOICES",
    "REDUCED_TERMS",
    "compute_residual_objective",
]
