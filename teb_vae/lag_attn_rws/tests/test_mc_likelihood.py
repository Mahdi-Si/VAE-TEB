r"""The Monte Carlo predictive estimator: its reduction, its common random numbers, and its $K=1$
degenerate case.

Three separate claims, tested separately because each fails differently and silently.

The **reduction** is $\operatorname{logsumexp}_r(-D_r) - \log K$, the log of an average
*likelihood*. An average of the $D_r$ instead -- the natural-looking mistake -- is a different and
strictly larger number whenever the draws disagree, and nothing about it looks wrong.

The **common random numbers** are what make $D_{\mathrm{base}} - D_{\mathrm{full}}$ a difference
of predictions rather than of noise. Two independent draws per branch would leave the estimator
unbiased and the difference far noisier, which shows up as an unstable readout rather than as a
failure.

At **$K = 1$** the estimator must be exactly the training-path score for the same draw, or the
evaluation and the training loop are measuring two different things.
"""
from __future__ import annotations

import math

import pytest
import torch

from teb_vae.lag_attn_rws.eval.metrics import (
    marginalise_block_scores,
    mc_predictive_block,
)
from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask
from teb_vae.lag_attn_rws.nets.raw_targets import build_future_target

from .conftest import BATCH, SEQ_LEN, TINY_KWARGS, make_stub_batch


@pytest.fixture
def scoring_setup():
    """A tiny model plus the target and mask its forecasts are scored against."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**TINY_KWARGS).eval()
    batch = make_stub_batch()
    target = build_future_target(batch.fhr, model.geometry, future_index=model.future_index)
    mask, _ = forecast_mask(batch.weight, model.geometry, coverage_floor=model.coverage_floor)
    return model, target, mask


def _latent(model, seed: int = 0):
    """A ``(mu, logvar)`` pair at the model's latent shape."""
    generator = torch.Generator().manual_seed(seed)
    shape = (BATCH, SEQ_LEN, model.d_z)
    return (
        torch.randn(shape, generator=generator),
        torch.randn(shape, generator=generator) * 0.1,
    )


# =============================================================================
# The reduction
# =============================================================================
def test_the_gaussian_reduction_is_logsumexp_minus_log_k():
    """The exact formula, against a hand-written expectation rather than a re-derivation."""
    scores = torch.tensor([[[1.0, 3.0]], [[2.0, 0.5]]])  # (K=2, B=1, T=2)

    marginalised = marginalise_block_scores(scores, "gaussian_nll")

    expected = -(torch.logsumexp(-scores, dim=0) - math.log(2.0))
    assert torch.allclose(marginalised, expected)


def test_the_gaussian_reduction_is_not_a_mean_of_log_scores():
    """The mistake this pins: averaging the per-draw NLLs. Jensen makes the marginal strictly
    smaller whenever the draws disagree, so a mean would report the model as worse than it is --
    and identically so for base and full, hiding the error in the difference."""
    scores = torch.tensor([[[1.0, 3.0]], [[2.0, 0.5]]])

    marginalised = marginalise_block_scores(scores, "gaussian_nll")

    assert bool((marginalised < scores.mean(dim=0)).all())


def test_identical_draws_make_the_marginal_equal_the_mean():
    """The boundary case: with no disagreement between draws the two reductions coincide, so the
    test above is about the spread and not about an arithmetic slip."""
    scores = torch.full((4, 2, 3), 1.75)

    assert torch.allclose(
        marginalise_block_scores(scores, "gaussian_nll"), scores.mean(dim=0)
    )


def test_the_mse_reduction_is_the_plain_expectation():
    """A squared-error block score is not a log-density, so its exponential means nothing; the
    marginal is taken in the space the quantity lives in."""
    scores = torch.tensor([[[1.0, 3.0]], [[2.0, 0.5]]])

    assert torch.allclose(marginalise_block_scores(scores, "mse"), scores.mean(dim=0))


# =============================================================================
# Common random numbers
# =============================================================================
def test_two_branches_with_the_same_latent_score_bitwise_identically(scoring_setup):
    """The common-random-numbers property, stated operationally: if the branches disagreed only
    through their noise draws, identical parameters would still give different scores."""
    model, target, mask = scoring_setup
    mu, logvar = _latent(model)

    scores, _ = mc_predictive_block(
        model,
        {"base": (mu, logvar), "full": (mu.clone(), logvar.clone())},
        target, mask, likelihood="gaussian_nll", num_samples=4,
    )

    assert torch.equal(scores["base"], scores["full"])


def test_a_third_branch_shares_the_same_draws_too(scoring_setup):
    """Every branch, not merely the first two: the shuffled control is compared against both."""
    model, target, mask = scoring_setup
    mu, logvar = _latent(model)
    branches = {name: (mu.clone(), logvar.clone()) for name in ("base", "full", "shuffled")}

    scores, _ = mc_predictive_block(
        model, branches, target, mask, likelihood="gaussian_nll", num_samples=3
    )

    assert torch.equal(scores["base"], scores["shuffled"])


def test_a_genuinely_different_latent_scores_differently(scoring_setup):
    """The other direction, so the equality above is not simply the estimator ignoring its
    inputs."""
    model, target, mask = scoring_setup
    mu, logvar = _latent(model, seed=0)
    other_mu, other_logvar = _latent(model, seed=7)

    scores, _ = mc_predictive_block(
        model,
        {"base": (mu, logvar), "full": (other_mu, other_logvar)},
        target, mask, likelihood="gaussian_nll", num_samples=2,
    )

    assert not torch.equal(scores["base"], scores["full"])


# =============================================================================
# The K = 1 degenerate case
# =============================================================================
@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_one_sample_reduces_exactly_to_the_training_path_score(scoring_setup, likelihood):
    """One draw, one decode, the training objective's own per-anchor function. If these differ,
    the evaluation is reporting a quantity the training loop never optimised."""
    model, target, mask = scoring_setup
    mu, logvar = _latent(model)

    torch.manual_seed(11)
    scores, _ = mc_predictive_block(
        model, {"base": (mu, logvar)}, target, mask,
        likelihood=likelihood, num_samples=1,
    )

    torch.manual_seed(11)
    epsilon = torch.randn_like(mu)
    latent = mu + epsilon * torch.exp(0.5 * logvar)
    with torch.no_grad():
        forecast_mu, forecast_logvar = model.decoder(latent[:, : model.geometry.t_valid])
    expected, _ = masked_raw_block_per_anchor(
        forecast_mu, target, mask, likelihood=likelihood, logvar=forecast_logvar
    )

    assert torch.allclose(scores["base"], expected, atol=1e-6)


# =============================================================================
# Guards
# =============================================================================
def test_no_branches_is_refused(scoring_setup):
    model, target, mask = scoring_setup

    with pytest.raises(ValueError, match="at least one branch"):
        mc_predictive_block(model, {}, target, mask, likelihood="mse")


def test_a_non_positive_sample_count_is_refused(scoring_setup):
    model, target, mask = scoring_setup
    mu, logvar = _latent(model)

    with pytest.raises(ValueError, match="num_samples"):
        mc_predictive_block(
            model, {"base": (mu, logvar)}, target, mask, likelihood="mse", num_samples=0
        )
