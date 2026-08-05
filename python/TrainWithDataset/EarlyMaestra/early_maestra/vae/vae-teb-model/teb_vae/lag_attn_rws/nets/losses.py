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

The whole objective lives here as free functions, not as methods on one model. Two architectures
now forecast the same raw target under the same four-term loss, and what they optimise must never
diverge: a copy of :func:`compute_loss` in a second model class would be two definitions of the
quantity every comparison between them is read off. The model classes keep one-line methods that
delegate here, so a caller still writes ``model.compute_loss(...)`` and the arithmetic has exactly
one home.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch

from teb_vae.lag_attn.nets.blocks import validate_choice
from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry
from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask as build_forecast_mask
from teb_vae.lag_attn_rws.nets.raw_masks import kl_mask as build_kl_mask
from teb_vae.lag_attn_rws.nets.raw_targets import build_future_target

LIKELIHOOD_CHOICES = ("mse", "gaussian_nll")

_LOG_2PI = math.log(2.0 * math.pi)

# "On the floor" for a log-variance: within this fraction of the clamp range of the lower
# asymptote (and, symmetrically, of the upper one for "on the ceiling"). The smooth bound never
# reaches an asymptote exactly, so an equality test would report 0.0 forever while the variance
# sits pinned.
#
# Public because the evaluation reads the same fractions off the same tensors: a margin restated
# there would make "the prior variance is pinned" mean one thing in a training log and another in
# a summary, and the two would drift the first time either was tuned. Re-exported from
# ``nets/model.py``, which is where every existing consumer imports it from.
LOGVAR_FLOOR_MARGIN_FRAC = 0.05

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


def masked_prior_rate(
    logvar_prior: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    r"""The prior's scale rate, summed over $d_z$, averaged over the masked anchors.

    Per dimension the quantity is

    $$\tfrac{1}{2}\left(e^{\ell} - 1 - \ell\right), \qquad \ell = \log\sigma^{2,p},$$

    which is $\mathrm{KL}\!\left(\mathcal{N}(\mu_p, \sigma_p^2) \,\|\, \mathcal{N}(\mu_p, 1)\right)$
    -- the scale half of the divergence from the prior to a unit-scale Gaussian at the same
    mean. It is nonnegative, convex in $\ell$, and exactly zero at $\sigma_p = 1$; the prior
    mean does not appear, so weighting it compresses nothing the base forecast depends on.

    It exists because no other term penalises a *narrow* prior: the reconstruction strictly
    prefers a deterministic latent, and $\mathrm{KL}(q \,\|\, p)$ constrains the posterior
    against the prior without constraining the prior's own scale -- so the prior log-variance
    is otherwise free to fall until it meets its clamp.

    Reduced exactly as :func:`masked_source_kl` reduces the source divergence -- summed over
    $d_z$, masked by the same $(B, T)$ anchor support, divided by the same contributing-anchor
    count -- so the two are in the same nats-per-anchor units and addable without rescaling.
    Unlike that function's raw readout this one carries gradient: weighted, it is an objective
    term, not only a diagnostic.

    Args:
        logvar_prior: Prior log-variance ``(B, T, d_z)``.
        mask: KL anchor mask ``(B, T)`` from ``nets/raw_masks.py::kl_mask`` -- the same support
            the source divergence is averaged over.

    Returns:
        A scalar tensor in nats per anchor.
    """
    n_anchors = mask.sum().clamp_min(1.0)
    rate_btd = 0.5 * (logvar_prior.exp() - 1.0 - logvar_prior)
    return (rate_btd.sum(dim=-1) * mask).sum() / n_anchors


def kld_tensor(
    mu_prior: torch.Tensor,
    logvar_prior: torch.Tensor,
    mu_post: torch.Tensor,
    logvar_post: torch.Tensor,
) -> torch.Tensor:
    r"""Closed-form KL between two diagonal Gaussians, per step and per dimension.

    $$\mathrm{KL} = \tfrac{1}{2}\left[\log\sigma^{2,p} - \log\sigma^{2,q}
    + \frac{\sigma^{2,q} + (\mu^q - \mu^p)^2}{\sigma^{2,p}} - 1\right]$$

    Closed-form rather than sampled: this quantity is the model's output, not an intermediate,
    and a Monte-Carlo estimate would put variance straight into the number being reported.
    Returned unmasked over the full sequence; masking is the caller's job, because every caller
    wants a different window.

    A free function rather than a method, because it reads nothing off a model: the two
    architectures that report this KL must compute it identically, and one formula written twice
    is two formulas.

    Args:
        mu_prior: Prior mean ``(B, T, d_z)``.
        logvar_prior: Prior log-variance ``(B, T, d_z)``.
        mu_post: Posterior mean ``(B, T, d_z)``.
        logvar_post: Posterior log-variance ``(B, T, d_z)``.

    Returns:
        The per-step per-dimension KL ``(B, T, d_z)``.
    """
    return 0.5 * (
        logvar_prior
        - logvar_post
        + (logvar_post.exp() + (mu_post - mu_prior) ** 2) / logvar_prior.exp()
        - 1.0
    )


# lean-limit: all valid anchors are decoded every batch; add anchor subsampling when a
# measured production run exceeds device memory after the documented config levers are
# exhausted.
def compute_loss(
    forward_outputs: Dict[str, torch.Tensor],
    fhr_raw: torch.Tensor,
    *,
    weight: torch.Tensor,
    geometry: TrimmedRawGeometry,
    future_index: torch.Tensor,
    coverage_floor: float,
    logvar_clamp: Tuple[float, float],
    beta: float = 1.0,
    beta_prior: float = 0.0,
    lambda_full: float = 1.0,
    lambda_base: float = 1.0,
    likelihood: str = "gaussian_nll",
    free_bits: float = 0.0,
) -> Dict[str, Any]:
    r"""Compute the four-term objective in nats per anchor.

    $$\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
    + \beta\,\mathrm{KL}_{\mathrm{train}} + \beta_p\,R_p$$

    * $D_1$ -- masked NLL of the raw future under the **full** (posterior-latent) forecast,
      summed over the $H \cdot R$ block, averaged over contributing anchors.
    * $D_0$ -- the same under the **base** (prior-latent) forecast. At unit weights,
      $D_1 + \beta\,\mathrm{KL}$ at $\beta = 1$ is the exact ELBO of the source-conditioned
      branch; adding $D_0$ doubles the reconstruction pressure against the KL, so $\beta = 1$ is
      a principled starting point rather than a distinguished optimum.
    * $\mathrm{KL}_{\mathrm{train}}$ -- the free-bits-floored KL over the *same* anchor support
      the reconstruction uses, $[w, T - H)$: charging KL on anchors with no reconstruction term
      produces an end-of-sequence droop resembling fading coupling.
    * $R_p$ -- the prior's scale rate (:func:`masked_prior_rate`), on the KL's own support. The
      other three terms leave the prior's scale unconstrained from below, so without this one
      the prior log-variance falls onto its clamp floor and the divergence stops being a rate.
      At the shipped ``beta_prior: 0.0`` the objective is exactly the historical three-term sum;
      $R_p$ is still computed and reported, unconditionally and under every likelihood, so a
      collapsing prior is visible in any run's metrics whether or not a config opted in.

    The raw future target and every mask are built internally from ``fhr_raw`` and ``weight``,
    matching the convention that the caller passes raw signals.

    A free function taking the geometry and the two scalar bounds explicitly, rather than a method
    reading them off ``self``: this is what both raw-signal architectures optimise, and it must be
    one definition rather than one per model class. Each model keeps a thin method that supplies
    its own geometry.

    Args:
        forward_outputs: The dict returned by a model's ``forward``.
        fhr_raw: Raw target signal ``(B, L_raw)``, loader-normalized.
        weight: Decimated validity signal ``(B, T)``.
        geometry: The model's trimmed-grid geometry.
        future_index: The model's cached raw-target index grid, as
            :func:`~teb_vae.lag_attn_rws.nets.raw_targets.build_future_index` returns it.
        coverage_floor: Minimum valid fraction of an anchor's forecast window for the anchor to
            enter the loss at all.
        logvar_clamp: ``(lo, hi)`` effective range of the model's log-variances, which the two
            binding-bound diagnostics are read against.
        beta: Weight on the trained KL term.
        beta_prior: Weight on the prior scale rate. ``0.0`` -- the default, so every caller
            that predates the term is unaffected -- leaves the objective the three-term sum
            while ``prior_rate`` is still reported.
        lambda_full: Weight on the full-forecast reconstruction.
        lambda_base: Weight on the base-forecast reconstruction.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.
        free_bits: Per-dimension per-step KL floor; enters the trained KL only.

    Returns:
        ``{'metrics': ..., 'likelihood': ...}``. ``metrics`` maps names to scalar tensors -- the
        four terms, ``total_loss``, the block/sample reconstruction pairs, ``pred_gap``, both KL
        readouts, ``kld_active_frac``, ``kld_beta``, ``beta_prior``, ``prior_rate``,
        ``anchor_coverage_frac`` and the log-variance diagnostics -- and is safe to splat into a
        metric logger. The likelihood name string is deliberately outside it.

    Raises:
        ValueError: On an unknown ``likelihood``, a raw length that does not match the geometry,
            or a ``weight`` that does not match the trimmed grid.
    """
    validate_choice(likelihood, LIKELIHOOD_CHOICES, "likelihood")
    device, dtype = fhr_raw.device, fhr_raw.dtype

    target = build_future_target(fhr_raw, geometry, future_index=future_index)
    mask, coverage_frac = build_forecast_mask(
        weight, geometry, coverage_floor=coverage_floor
    )
    kl_support = build_kl_mask(mask, geometry)

    nll_full_block, nll_full_sample = masked_raw_likelihood(
        forward_outputs["mu_full"],
        target,
        mask,
        likelihood=likelihood,
        logvar=forward_outputs["logvar_full"],
    )
    nll_base_block, nll_base_sample = masked_raw_likelihood(
        forward_outputs["mu_base"],
        target,
        mask,
        likelihood=likelihood,
        logvar=forward_outputs["logvar_base"],
    )

    kld_btd = kld_tensor(
        mu_prior=forward_outputs["mu_prior"],
        logvar_prior=forward_outputs["logvar_prior"],
        mu_post=forward_outputs["mu_post"],
        logvar_post=forward_outputs["logvar_post"],
    )
    kl_terms = masked_source_kl(kld_btd, kl_support, free_bits=free_bits)

    # On the KL's own anchor support, and in the graph: weighted, this is an objective term.
    # Computed unconditionally so a run that never opted in still reports its prior's rate.
    prior_rate = masked_prior_rate(forward_outputs["logvar_prior"], kl_support)

    total_loss = (
        lambda_full * nll_full_block
        + lambda_base * nll_base_block
        + beta * kl_terms["source_conditioned_kl_train"]
        + beta_prior * prior_rate
    )

    # Diagnostics over the same masked supports the losses use, so each stays inside its own
    # bound band instead of scaling with the mask density.
    with torch.no_grad():
        pred_gap = nll_base_block - nll_full_block

        elem_mask = mask[..., None]
        elem_denom = (elem_mask.sum() * float(geometry.r)).clamp_min(1.0)
        mean_logvar_full = (forward_outputs["logvar_full"] * elem_mask).sum() / elem_denom
        mean_logvar_base = (forward_outputs["logvar_base"] * elem_mask).sum() / elem_denom

        # Whether the DECODER's log-variance bound is binding, at each end separately.
        # mean_logvar_full cannot answer this: one mean is equally consistent with a
        # well-spread distribution and with half the mass pinned on each clamp. The shipped
        # [-5, 3] was inherited from a decoder that emitted feature coefficients, and the
        # config marks it for re-derivation against a z-scored raw target from exactly these
        # two numbers. The two ends fail differently -- pinned at the floor the decoder is
        # over-confident and the NLL's squared term explodes (this is what a loss spike looks
        # like from the inside); pinned at the ceiling it has given up and is predicting
        # noise, which reads as a healthy falling NLL while pred_gap goes to zero.
        lo, hi = logvar_clamp
        bound_margin = LOGVAR_FLOOR_MARGIN_FRAC * (hi - lo)
        logvar_full = forward_outputs["logvar_full"]
        logvar_full_floor_frac = (
            (logvar_full <= lo + bound_margin).to(dtype) * elem_mask
        ).sum() / elem_denom
        logvar_full_ceil_frac = (
            (logvar_full >= hi - bound_margin).to(dtype) * elem_mask
        ).sum() / elem_denom

        # The prior-variance floor watch. The KL carries (mu_q - mu_p)^2 / sigma_p^2, so
        # a prior variance pinned on its lower clamp inflates the coupling readout by
        # orders of magnitude while the decoder-side logvar metrics above look healthy.
        support = kl_support > 0
        if bool(support.any()):
            logvar_prior_masked = forward_outputs["logvar_prior"][support]
            floor_threshold = lo + LOGVAR_FLOOR_MARGIN_FRAC * (hi - lo)
            logvar_prior_floor_frac = (
                (logvar_prior_masked <= floor_threshold).to(dtype).mean()
            )
            mean_logvar_prior = logvar_prior_masked.mean()
            mean_logvar_post = forward_outputs["logvar_post"][support].mean()
            delta_mu_rms = (
                (forward_outputs["mu_post"] - forward_outputs["mu_prior"])[support]
                .pow(2)
                .mean()
                .sqrt()
            )
        else:
            zero = torch.zeros((), device=device, dtype=dtype)
            logvar_prior_floor_frac = zero
            mean_logvar_prior = zero.clone()
            mean_logvar_post = zero.clone()
            delta_mu_rms = zero.clone()

        # Coverage over the trained anchors, pre-floor: the distribution this summarises
        # is what decides whether the shipped coverage_floor is right.
        anchor_coverage_frac = coverage_frac[:, geometry.warmup :].mean()

    metrics: Dict[str, torch.Tensor] = {
        "total_loss": total_loss,
        "nll_full_block": nll_full_block,
        "nll_full_sample": nll_full_sample,
        "nll_base_block": nll_base_block,
        "nll_base_sample": nll_base_sample,
        "pred_gap": pred_gap,
        "source_conditioned_kl_raw": kl_terms["source_conditioned_kl_raw"],
        "source_conditioned_kl_train": kl_terms["source_conditioned_kl_train"],
        "kld_active_frac": kl_terms["kld_active_frac"],
        "prior_rate": prior_rate,
        "kld_beta": torch.tensor(float(beta), device=device, dtype=dtype),
        # Echoed like kld_beta so a metrics_history.csv identifies its own arm and the
        # weighted terms can be recomposed from the file alone.
        "beta_prior": torch.tensor(float(beta_prior), device=device, dtype=dtype),
        "anchor_coverage_frac": anchor_coverage_frac,
        "mean_logvar_full": mean_logvar_full,
        "mean_logvar_base": mean_logvar_base,
        "logvar_full_floor_frac": logvar_full_floor_frac,
        "logvar_full_ceil_frac": logvar_full_ceil_frac,
        "mean_logvar_prior": mean_logvar_prior,
        "mean_logvar_post": mean_logvar_post,
        "logvar_prior_floor_frac": logvar_prior_floor_frac,
        "delta_mu_rms": delta_mu_rms,
    }
    # The one non-tensor lives outside the metric dict, so a caller cannot splat a string
    # into a numeric logger by accident.
    return {"metrics": metrics, "likelihood": likelihood}
