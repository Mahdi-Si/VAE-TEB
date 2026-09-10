r"""Matched Monte Carlo predictive scoring, its concentration diagnostic, and mixture calibration.

Every branch a run compares -- the target-only prior, the source-conditioned full branch, and each
intervened arm -- is scored **here, in one loop, under one set of draws**. That is not an
optimisation. One $\epsilon^{(k)}$ per replicate shared across branches is what makes a reported
margin a difference of predictions: two branches with identical latent parameters then produce
bitwise identical scores, and their margin is exactly zero rather than a small number that has to
be argued away.

The score itself is the marginalised predictive density,

$$D^{(K)}_{b_0,t}
  = -\operatorname{logsumexp}_{k=1}^{K}\bigl(-D^{(k)}_{b_0,t}\bigr) + \log K,$$

the negative log of the **average likelihood**. It is not the average of the per-draw negative log
likelihoods, and it is strictly smaller than it whenever the draws disagree; the difference is the
Jensen gap, and reporting the average instead would report a quantity that improves as the latent
becomes less informative. Decoding the prior mean alone is a third quantity again and is not a
predictive density at all.

**Why this package draws its own loop rather than calling the shared estimator.** The shared
``mc_predictive_block`` gathers ``latent[:, anchors]`` before decoding, which is right for an
architecture whose latent is produced at every stored step and wrong for this one, whose latent
exists at the decoded anchors and nowhere else -- there is no gather left to do. Passing it an
identity index would work and would leave a load-bearing line reading as though an anchor gather
had happened. What *is* reused is the pure arithmetic that has no architecture in it:
``marginalise_block_scores`` for the log-mean-likelihood and ``masked_raw_block_per_anchor`` for
each draw's own block score, both taken from the modules that already define them.

**The estimator's own bias, stated because it is easy to forget.** The likelihood average is
unbiased for the model likelihood; its negative logarithm is upward biased for the negative log
density at finite $K$, and the two branches' biases need not cancel. That is what
:func:`draw_concentration` is for -- it reports how many of the $K$ draws a score effectively
rests on -- and it is why a finalist is scored at more than one draw count.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from teb_vae.lag_attn_cfs.eval.metrics import marginalise_block_scores
from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor

#: Central probability levels the calibration census reports coverage at. Three rather than one:
#: a model can be well calibrated in the body and badly calibrated in the tail, and the tail is
#: where a forecast is read.
DEFAULT_COVERAGE_LEVELS: Tuple[float, ...] = (0.5, 0.9, 0.99)

#: $\sqrt 2$, for the Gaussian cumulative distribution through ``torch.erf``. Named rather than
#: written inline at three call sites.
_SQRT_TWO = math.sqrt(2.0)


@dataclass
class BranchScores:
    r"""One branch's predictive scores under the shared draws.

    Attributes:
        marginal: The marginalised per-anchor score $D^{(K)}_t$, $(B, A)$. Under ``'mse'`` this is
            the plain mean over draws, because a squared error is not a log density and its
            exponential means nothing.
        per_draw: Each draw's own per-anchor block score, $(K, B, A)$. Kept because the
            concentration diagnostic cannot be recovered from the marginal, and because a reader
            asking whether $K$ was large enough has no other evidence.
        contributing: The $0/1$ anchor indicator the draws share, $(B, A)$.
        cdf_sum: $\sum_k \Phi\bigl((V - \widehat\mu^{(k)}) / \sigma^{(k)}\bigr)$ over the scored
            block, $(B, A, H, C)$, or ``None`` for a branch the caller did not ask to calibrate.
            Accumulated inside the draw loop rather than stacked: the stacked forecasts would be
            $K$ times the size of one, which at the production geometry is gigabytes for a
            quantity that reduces to one number per coefficient.
    """

    marginal: torch.Tensor
    per_draw: torch.Tensor
    contributing: torch.Tensor
    cdf_sum: Optional[torch.Tensor] = None


def gaussian_cdf(value: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    r"""$\Phi\bigl((y - \mu)/\sigma\bigr)$ for one Gaussian component.

    Args:
        value: The observed coefficient.
        mu: The component's mean.
        logvar: The component's log-variance, so $\sigma = e^{\lambda/2}$.

    Returns:
        The component's cumulative probability at ``value``, shaped by broadcasting.
    """
    standardized = (value - mu) * torch.exp(-0.5 * logvar)
    return 0.5 * (1.0 + torch.erf(standardized / _SQRT_TWO))


@torch.no_grad()
def matched_predictive_scores(
    model: Any,
    branches: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    likelihood: str,
    num_samples: int,
    generator: Optional[torch.Generator] = None,
    persistence: Optional[torch.Tensor] = None,
    calibrate: Sequence[str] = (),
) -> Dict[str, BranchScores]:
    r"""Score every branch's forecast under common random numbers, at the decoded anchors.

    The latent parameters arrive **already anchor-indexed**, which is this architecture's contract:
    the second axis of every tensor here is a position in the decoded anchor set, not a stored
    step. Nothing is gathered, and the shape agreement against the target's own anchor axis is
    checked rather than assumed -- feeding a stored-clock tensor in would otherwise broadcast into a
    block nobody asked for.

    Args:
        model: The net, for its shared decoder. It is invoked once per branch per draw with the
            identical persistence input, so nothing source-derived reaches it except the latent.
        branches: ``{name: (mu, logvar)}``, each $(B, A, d_z)$. Every branch must share a shape;
            the first fixes the noise draw, and the iteration order fixes nothing else.
        target: The gathered forecast target $(B, A, H, C)$.
        mask: The forecast mask $(B, A, H)$.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.
        num_samples: Monte Carlo draws $K$.
        generator: Generator for $\epsilon$, on the parameters' own device. Scoring is the one
            place an evaluation adds randomness of its own, so it takes an explicit stream: two
            runs of one checkpoint must report the same numbers, and a global draw makes that a
            property of whatever else in the process drew first.
        persistence: The matched forward's own persistence input $(B, A, C)$ on a model built with
            the decoder residual, and ``None`` otherwise. The same tensor for every branch and
            every draw, because it is target-only and therefore a property of the anchor.
        calibrate: Branch names whose mixture cumulative distribution is accumulated over the
            draws. Empty by default: it costs a $(B, A, H, C)$ accumulator per named branch, and
            most branches are scored for a margin rather than for their calibration.

    Returns:
        One :class:`BranchScores` per branch, keyed as ``branches`` was.

    Raises:
        ValueError: If ``branches`` is empty, if ``num_samples`` is not positive, if a branch's
            anchor axis disagrees with the target's, or if ``calibrate`` names a branch that was
            not scored.
    """
    if not branches:
        raise ValueError("matched_predictive_scores needs at least one branch to score")
    if int(num_samples) < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    unknown = sorted(set(calibrate) - set(branches))
    if unknown:
        raise ValueError(
            f"calibrate names branches that are not being scored: {unknown}. Scored branches are "
            f"{sorted(branches)}."
        )

    reference_mu = next(iter(branches.values()))[0]
    for name, (mu, _logvar) in branches.items():
        if mu.shape[:2] != target.shape[:2]:
            raise ValueError(
                f"branch {name!r} has anchor axis {tuple(mu.shape[:2])} against the target's "
                f"{tuple(target.shape[:2])}. Every latent tensor of this architecture is indexed "
                f"by decoded anchor, not by stored step; a stored-clock tensor here would be "
                f"scored against the wrong rows."
            )

    draws: Dict[str, list] = {name: [] for name in branches}
    cdf_sums: Dict[str, torch.Tensor] = {}
    contributing: Optional[torch.Tensor] = None

    for _ in range(int(num_samples)):
        # Drawn once, outside the branch loop. This line is the common-random-numbers property and
        # is the whole reason the arms are scored together rather than one at a time.
        epsilon = (
            torch.randn_like(reference_mu)
            if generator is None
            else torch.empty_like(reference_mu).normal_(generator=generator)
        )
        for name, (mu, logvar) in branches.items():
            latent = mu + epsilon * torch.exp(0.5 * logvar)
            forecast_mu, forecast_logvar = model.decoder(latent, persistence=persistence)
            block, contributing = masked_raw_block_per_anchor(
                forecast_mu, target, mask, likelihood=likelihood, logvar=forecast_logvar
            )
            draws[name].append(block)
            if name in calibrate:
                component = gaussian_cdf(target, forecast_mu, forecast_logvar)
                cdf_sums[name] = (
                    component if name not in cdf_sums else cdf_sums[name] + component
                )

    assert contributing is not None  # the loops above ran at least once
    scored: Dict[str, BranchScores] = {}
    for name, blocks in draws.items():
        per_draw = torch.stack(blocks, dim=0)
        total = cdf_sums.get(name)
        scored[name] = BranchScores(
            marginal=marginalise_block_scores(per_draw, likelihood),
            per_draw=per_draw,
            contributing=contributing,
            cdf_sum=None if total is None else total / float(num_samples),
        )
    return scored


def draw_concentration(per_draw: torch.Tensor) -> torch.Tensor:
    r"""How many of the $K$ draws a marginalised score effectively rests on.

    With normalised likelihood weights $\alpha_k = e^{\Lambda_k} / \sum_r e^{\Lambda_r}$ and
    $\Lambda_k = -D^{(k)}$, the diagnostic is

    $$\frac{1}{\sum_k \alpha_k^2} \in [1, K],$$

    which is $K$ when every draw contributes equally and $1$ when one draw carries the whole
    average. It is computed as a softmax over the draw axis, which is the numerically stable form
    of the same ratio.

    **These are not attention weights and not a proof that the Monte Carlo error is small.** A
    score resting on one of eight draws is a warning that $K$ is too small for that anchor; a value
    near $K$ says the draws agreed, not that the estimator has converged.

    Args:
        per_draw: Each draw's per-anchor block score $(K, B, A)$, as
            :attr:`BranchScores.per_draw` carries it.

    Returns:
        The effective draw count per anchor, $(B, A)$.

    Raises:
        ValueError: If ``per_draw`` is not 3-D.
    """
    if per_draw.dim() != 3:
        raise ValueError(
            f"per_draw must be 3-D (K, B, A), got shape {tuple(per_draw.shape)}"
        )
    weights = torch.softmax(-per_draw.to(torch.float64), dim=0)
    return 1.0 / (weights**2).sum(dim=0)


def calibration_census(
    cdf: torch.Tensor,
    mask: torch.Tensor,
    *,
    levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
) -> Dict[str, Any]:
    r"""Probability-integral transform and central coverage, from the **mixture** distribution.

    The probability integral transform of the marginalised predictive law is
    $u = \widehat F(V)$ with

    $$\widehat F(y) = \mathbb E_Z\,\Phi\!\left(\frac{y - \widehat\mu(Z)}{e^{\nu(Z)/2}}\right),$$

    which is the draw average this function is handed. Central coverage follows from it with no
    quantile solve at all: an observation lies inside the central-$q$ interval of a distribution
    exactly when its own cumulative probability lies in $[(1-q)/2, (1+q)/2]$, whatever shape that
    distribution has.

    **A mean of conditional standard deviations appears nowhere in this path, and could not.** The
    predictive law is a mixture over the latent, so its variance is
    $\mathbb E_Z e^{\nu(Z)} + \operatorname{Var}_Z \widehat\mu(Z)$ -- the second term is the whole
    contribution of latent uncertainty, and a Gaussian interval built from even the correct total
    variance is still not a mixture quantile. Averaging the components' cumulative probabilities is
    exact for the mixture by linearity, which is why it is what is accumulated.

    **Sums, not means.** A pass accumulates these across batches with :func:`merge_calibration`
    and finishes them with :func:`finish_calibration`, so the reported figures are means over every
    scored coefficient of the split rather than means of per-batch means -- which would weight a
    short batch equally with a full one. The squared sum travels for the same reason: a variance
    cannot be recovered from a mean, and recomputing it from a second pass over the data would be a
    second definition of the same quantity.

    Args:
        cdf: The mixture cumulative probability at each scored coefficient, $(B, A, H, C)$.
        mask: The forecast mask $(B, A, H)$, broadcast over the channel axis.
        levels: Central probability levels to report coverage at.

    Returns:
        ``{'n_coefficients', 'pit_sum', 'pit_sq_sum', 'inside': {level: count}}``.

    Raises:
        ValueError: If a level is not strictly inside $(0, 1)$.
    """
    for level in levels:
        if not 0.0 < float(level) < 1.0:
            raise ValueError(f"a central coverage level must lie in (0, 1), got {level}")

    weights = mask.unsqueeze(-1).expand_as(cdf).to(torch.float64)
    values = cdf.to(torch.float64)
    inside: Dict[str, float] = {}
    for level in levels:
        half = 0.5 * float(level)
        within = (values >= 0.5 - half) & (values <= 0.5 + half)
        inside[f"{float(level):g}"] = float((within.to(torch.float64) * weights).sum())
    return {
        "n_coefficients": float(weights.sum()),
        "pit_sum": float((values * weights).sum()),
        "pit_sq_sum": float(((values**2) * weights).sum()),
        "inside": inside,
    }


def merge_calibration(
    left: Optional[Dict[str, Any]], right: Mapping[str, Any]
) -> Dict[str, Any]:
    """Accumulate two batches of calibration sums.

    Args:
        left: The running totals, or ``None`` on the first batch.
        right: This batch's sums.

    Returns:
        The summed totals.
    """
    if left is None:
        return {
            "n_coefficients": float(right["n_coefficients"]),
            "pit_sum": float(right["pit_sum"]),
            "pit_sq_sum": float(right["pit_sq_sum"]),
            "inside": dict(right["inside"]),
        }
    merged = {
        "n_coefficients": left["n_coefficients"] + float(right["n_coefficients"]),
        "pit_sum": left["pit_sum"] + float(right["pit_sum"]),
        "pit_sq_sum": left["pit_sq_sum"] + float(right["pit_sq_sum"]),
        "inside": dict(left["inside"]),
    }
    for level, count in right["inside"].items():
        merged["inside"][level] = merged["inside"].get(level, 0.0) + float(count)
    return merged


def finish_calibration(totals: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    r"""Reduce the accumulated sums to the figures a summary reports.

    A perfectly calibrated forecast has a uniform transform, so the mean is $1/2$ and the variance
    $1/12$, and central coverage matches its nominal level. All three are reported rather than one
    distance from uniformity: a transform shifted off $1/2$ and one that is merely over-dispersed
    are different defects with different fixes, and coverage at three levels separates a body
    problem from a tail problem.

    Args:
        totals: The accumulated sums, or ``None`` when nothing was scored.

    Returns:
        ``{'n_coefficients', 'pit_mean', 'pit_var', 'coverage': {level: fraction},
        'uniform_pit_mean', 'uniform_pit_var'}``, with every measured figure ``None`` where
        nothing was scored.
    """
    reference = {"uniform_pit_mean": 0.5, "uniform_pit_var": 1.0 / 12.0}
    if not totals or float(totals["n_coefficients"]) <= 0.0:
        return {
            "n_coefficients": 0.0,
            "pit_mean": None,
            "pit_var": None,
            "coverage": {},
            **reference,
        }
    count = float(totals["n_coefficients"])
    mean = float(totals["pit_sum"]) / count
    return {
        "n_coefficients": count,
        "pit_mean": mean,
        "pit_var": float(totals["pit_sq_sum"]) / count - mean**2,
        "coverage": {
            level: float(inside) / count for level, inside in totals["inside"].items()
        },
        **reference,
    }


__all__ = [
    "DEFAULT_COVERAGE_LEVELS",
    "BranchScores",
    "calibration_census",
    "draw_concentration",
    "finish_calibration",
    "merge_calibration",
    "gaussian_cdf",
    "matched_predictive_scores",
]
