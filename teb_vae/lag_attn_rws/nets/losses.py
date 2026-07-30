r"""Objective terms in explicit, consistent units: nats per anchor.

Every quantity here follows one convention. The reconstruction NLL is **summed** over the
$H \cdot R$-sample forecast block (with its full constant, so the value is a true log-density),
the KL is **summed** over $d_z$, and both are then **averaged over batch and contributing
anchors only**. Both terms are therefore in nats per anchor, which is what makes the KL weight
$\beta$ mean something at all: at $\beta = 1$, reconstruction-plus-KL is the exact ELBO of the
source-conditioned branch. The MSE variant keeps the same summed-over-the-block convention so
$\beta$ keeps its meaning across likelihoods.

Masks arrive on the decimated grid (``nets/raw_masks.py``) and are broadcast over the $R$ raw
samples of each horizon token. A masked position contributes exactly zero -- multiplicatively,
so a finite planted value at a masked position cannot move the loss at all -- and an anchor
whose whole window is masked drops out of the denominator as well as the numerator.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from teb_vae.lag_attn.nets.blocks import validate_choice
from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors

LIKELIHOOD_CHOICES = ("mse", "gaussian_nll")

_LOG_2PI = math.log(2.0 * math.pi)

# A latent dimension counts as carrying information once its mean per-step KL clears this.
# Well below any meaningful coupling, well above float noise on a collapsed dimension. Public
# because the offline latent-collapse verdict must be read against the same threshold the
# training metric reports, and two copies of a threshold are two thresholds.
KLD_ACTIVE_EPS = 1e-2


def raw_sample_score(
    mu: torch.Tensor,
    target: torch.Tensor,
    *,
    likelihood: str,
    logvar: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""The unmasked, unsummed score of every raw forecast sample.

    Under ``'gaussian_nll'`` the per-sample term carries the full constant,

    $$-\log p(x \mid \mu, \sigma^2) = \tfrac{1}{2}\left[\log 2\pi + \log\sigma^2
    + (x - \mu)^2 e^{-\log\sigma^2}\right],$$

    so summing it over an anchor's $H \cdot R$ forecast samples gives a true negative log-density
    in nats. Under ``'mse'`` the per-sample term is $(x - \mu)^2$.

    Exposed separately from :func:`masked_raw_block_per_anchor` because the *same* elementwise
    term is reduced over two different axis sets: over $(\tau, r)$ for the per-anchor block score
    the objective uses, and over $r$ alone for the horizon-resolved score the evaluation reads.
    Written twice, the two reductions could stop being decompositions of one another -- and the
    property that the horizon curve sums back to the anchor score is exactly what makes it
    readable.

    Every argument broadcasts, so a constant-mean baseline may be passed as a $(B, T, 1, 1)$
    tensor and a fixed observation variance as a scalar; the returned score is the broadcast
    shape, which ``target`` always fixes to the full grid.

    Args:
        mu: Forecast mean, broadcastable to $(B, T_{\mathrm{valid}}, H, R)$.
        target: Raw future target $(B, T_{\mathrm{valid}}, H, R)$.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.
        logvar: Forecast log-variance, broadcastable to the same shape; required under
            ``'gaussian_nll'``, ignored under ``'mse'``.

    Returns:
        The per-raw-sample score $(B, T_{\mathrm{valid}}, H, R)$.

    Raises:
        ValueError: On an unknown ``likelihood``, or ``'gaussian_nll'`` without ``logvar``.
    """
    validate_choice(likelihood, LIKELIHOOD_CHOICES, "likelihood")

    diff2 = (target - mu) ** 2
    if likelihood == "mse":
        return diff2
    if logvar is None:
        raise ValueError(
            "likelihood='gaussian_nll' requires logvar; only 'mse' works without one"
        )
    return 0.5 * (_LOG_2PI + logvar + diff2 * torch.exp(-logvar))


def masked_raw_block_per_anchor(
    mu: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    likelihood: str,
    logvar: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Each anchor's own masked block score, before any averaging.

    The per-sample term is :func:`raw_sample_score`; this sums it over the anchor's $H \cdot R$
    forecast samples. Under ``'mse'`` the sum runs over the same block, so the scale convention
    (and therefore the meaning of $\beta$) is preserved across likelihoods.

    Separated from the reduction below because a Monte Carlo predictive estimate must combine
    *per-anchor* scores across latent draws before averaging over anchors, and a second copy of
    this arithmetic in the evaluation package is exactly how the two would drift apart.

    lean-limit: the likelihood is a factorized Gaussian, so its predictive log-score
    difference estimates transfer entropy within that model family rather than the
    data-generating TE; upgrade the covariance when calibration diagnostics show the
    factorized form is the binding error.

    Args:
        mu: Forecast mean $(B, T_{\mathrm{valid}}, H, R)$.
        target: Raw future target $(B, T_{\mathrm{valid}}, H, R)$.
        mask: Decimated forecast mask $(B, T_{\mathrm{valid}}, H)$, broadcast over $r$.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.
        logvar: Forecast log-variance $(B, T_{\mathrm{valid}}, H, R)$; required under
            ``'gaussian_nll'``, ignored under ``'mse'``.

    Returns:
        ``(block_per_anchor, contributing)``, both $(B, T_{\mathrm{valid}})$: the summed block
        score of each anchor, and a $0/1$ indicator of whether the anchor contributes at all.

    Raises:
        ValueError: On an unknown ``likelihood``, or ``'gaussian_nll'`` without ``logvar``.
    """
    per_sample = raw_sample_score(mu, target, likelihood=likelihood, logvar=logvar)

    block_per_anchor = (per_sample * mask[..., None]).sum(dim=(2, 3))  # (B, T_valid)
    # The same indicator kl_mask is built from, so the two terms are averaged over one anchor
    # set rather than two that agree only by coincidence.
    # Taken from the score's dtype rather than from ``mu``'s: a constant-mean baseline may arrive
    # as a scalar of another dtype, and the indicator has to match what it accompanies.
    contributing = contributing_anchors(mask).to(block_per_anchor.dtype)  # (B, T_valid)
    return block_per_anchor, contributing


def masked_raw_likelihood(
    mu: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    likelihood: str,
    logvar: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Masked reconstruction loss, summed over the forecast block, averaged over anchors.

    See :func:`masked_raw_block_per_anchor` for the per-sample term; this is its reduction.

    Args:
        mu: Forecast mean $(B, T_{\mathrm{valid}}, H, R)$.
        target: Raw future target $(B, T_{\mathrm{valid}}, H, R)$.
        mask: Decimated forecast mask $(B, T_{\mathrm{valid}}, H)$, broadcast over $r$.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.
        logvar: Forecast log-variance $(B, T_{\mathrm{valid}}, H, R)$; required under
            ``'gaussian_nll'``, ignored under ``'mse'``.

    Returns:
        ``(d_block, d_sample)``: the per-anchor block value (summed over the $H \cdot R$
        samples, averaged over batch and contributing anchors) and the fixed rescaling
        ``d_sample = d_block / (H * R)``.

    Raises:
        ValueError: On an unknown ``likelihood``, or ``'gaussian_nll'`` without ``logvar``.
    """
    block_per_anchor, contributing = masked_raw_block_per_anchor(
        mu, target, mask, likelihood=likelihood, logvar=logvar
    )

    # Average over the anchors that contribute at all: a fully masked anchor (warm-up, gap,
    # coverage floor) leaves numerator AND denominator, so the per-anchor scale does not drift
    # with the mask density.
    n_anchors = contributing.sum().clamp_min(1.0)

    d_block = block_per_anchor.sum() / n_anchors
    d_sample = d_block / float(mu.shape[2] * mu.shape[3])
    return d_block, d_sample


def masked_source_kl(
    kld_btd: torch.Tensor,
    mask: torch.Tensor,
    *,
    free_bits: float = 0.0,
) -> Dict[str, torch.Tensor]:
    r"""The source-conditioned KL, summed over $d_z$, averaged over the masked anchors.

    Two scalars come back and they are not interchangeable. ``source_conditioned_kl_raw`` is
    the un-floored KL, computed without gradient: it is the only quantity that may be read as
    an information rate. ``source_conditioned_kl_train`` applies the free-bits floor per
    dimension per step *before* masking and is what enters the loss; with a positive floor it
    exceeds the raw value by construction, which is exactly why it must never be reported as a
    measurement.

    lean-limit: the KL is a provisional source-conditioned rate, not transfer entropy,
    because input features at time t read up to 974 s into their own future; replace the
    label with a TE claim when the reach budget is enforced and the empirical leak test
    passes at the configured budget.

    Args:
        kld_btd: Per-step per-dimension KL $(B, T, d_z)$, as ``kld_tensor`` returns it.
        mask: KL anchor mask $(B, T)$ from ``nets/raw_masks.py::kl_mask``, which derives it from
            the forecast mask -- so it is zero on exactly the anchors the reconstruction does
            not score, and both terms average over the same anchor count.
        free_bits: Per-dimension per-step floor applied to the trained KL. ``0.0`` makes the
            two returned scalars equal.

    Returns:
        ``{'source_conditioned_kl_raw', 'source_conditioned_kl_train', 'kld_active_frac'}``,
        the first two in nats per anchor, the last the fraction of latent dimensions whose
        mean masked KL clears the activity threshold.
    """
    n_anchors = mask.sum().clamp_min(1.0)

    per_t_train = kld_btd.clamp(min=float(free_bits)) if free_bits > 0.0 else kld_btd
    kl_train = (per_t_train.sum(dim=-1) * mask).sum() / n_anchors

    with torch.no_grad():
        kl_raw = (kld_btd.sum(dim=-1) * mask).sum() / n_anchors

        # Fraction of dimensions carrying information over the masked support. The boolean
        # advanced indexing below is data-dependent -- one of the reasons `compile` must stay
        # off for this model.
        support = mask > 0
        if bool(support.any()):
            kld_dim_mean = kld_btd[support].mean(dim=0)  # (n_masked_steps, d_z) -> (d_z,)
            kld_active_frac = (kld_dim_mean > KLD_ACTIVE_EPS).to(kld_btd.dtype).mean()
        else:
            kld_active_frac = torch.zeros((), device=kld_btd.device, dtype=kld_btd.dtype)

    return {
        "source_conditioned_kl_raw": kl_raw,
        "source_conditioned_kl_train": kl_train,
        "kld_active_frac": kld_active_frac,
    }
