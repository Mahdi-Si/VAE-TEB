r"""``compute_loss`` and ``kld_tensor`` are the sibling's, and this is what proves it.

Both architectures optimise **one** objective, imported from the module that owns it. That is not a
convenience: this package exists to attribute a difference in results to the input representation,
and a second copy of the loss would make the comparison partly a comparison of two losses. Each
model's ``compute_loss`` is therefore a thin delegation supplying its own geometry, its own cached
index grid, its own coverage floor and its own log-variance bounds.

Three things make the check here non-vacuous, and each replaces an easier test that would have
proven nothing.

* **The four supplied arguments are compared first.** "Identical output" from two delegations that
  passed different geometries would be a statement about the inputs happening to agree on this
  batch, not about the objective being shared.
* **The forward dict is a real one**, produced by a real ``SeqVaeLagAttnTrfRws`` forward and handed
  to *both* models. A hand-written dict drifts from what ``losses.compute_loss`` actually reads --
  silently, since a missing key it does not touch on this code path costs nothing until the day it
  does.
* **``kld_tensor`` is checked against the closed form written out below**, not against the sibling.
  Comparing two one-line delegations to the same function only proves they agree with each other,
  which they would even if the function were wrong.

Every metric tensor is compared with ``torch.equal``, not just the scalar loss: a scalar can agree
while a per-anchor readout the figures and the evaluation both consume does not.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest
import torch

from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from teb_vae.lag_attn_transformer_e2e.tests.conftest import (
    BATCH,
    SEQ_LEN,
    TINY_KWARGS,
    TINY_WARMUP_PERIOD,
    make_stub_batch,
)
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
from teb_vae.lag_attn_transformer_rws.tests.conftest import TINY_KWARGS as SIBLING_TINY_KWARGS

#: Objective settings the two models are driven at. ``beta_prior`` is nonzero so the prior scale
#: rate -- the one term a config can switch on -- is exercised rather than multiplied away, and
#: ``free_bits`` is nonzero so the raw/train KL split is two different tensors rather than one.
_LOSS_KWARGS: Dict[str, Any] = dict(
    beta=0.7, beta_prior=1.0e-2, lambda_full=1.0, lambda_base=0.5, free_bits=0.05
)

#: The sibling, built at **this** model's warm-up. ``warmup`` is one of the four fields of the
#: geometry the objective is handed, and it decides which anchors the loss masks admit, so the two
#: models must share it or the comparison below would be reading two different anchor sets. They do
#: not share it by default: a four-stage stride-2 cascade needs more warm-up than the sibling's two
#: steps to keep its reach inside the budget, which is a property of the front end and not of the
#: objective. Six is legal for the sibling too -- the constructor invariant is
#: ``warmup_period < T - horizon = 12`` -- so matching it costs nothing there.
_SIBLING_MATCHED = dict(SIBLING_TINY_KWARGS, warmup_period=TINY_WARMUP_PERIOD)


@pytest.fixture(scope="module")
def pair():
    """Both models at matching geometry, plus one real forward dict and the batch it came from.

    The forward is the **sibling's**, deliberately: it is the dict shape the shared objective was
    written against, and feeding the same object to both models is what makes the comparison a
    statement about the two ``compute_loss`` methods rather than about two forwards.

    Returns:
        ``(model, sibling, forward_outputs, batch)``.
    """
    batch = make_stub_batch(BATCH, SEQ_LEN)
    torch.manual_seed(0)
    sibling = SeqVaeLagAttnTrfRws(**dict(_SIBLING_MATCHED)).eval()
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfE2E(**dict(TINY_KWARGS)).eval()

    torch.manual_seed(0)
    with torch.no_grad():
        outputs = sibling(
            batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1)
        )
    return model, sibling, outputs, batch


def _closed_form_kld(mu_p, logvar_p, mu_q, logvar_q) -> torch.Tensor:
    r"""$\mathrm{KL}(q \Vert p)$ per step per dimension, written out rather than called.

    $$\mathrm{KL} = \tfrac12\left(\ell^p - \ell^q
        + \frac{e^{\ell^q} + (\mu^q - \mu^p)^2}{e^{\ell^p}} - 1\right)$$

    Args:
        mu_p: Prior mean.
        logvar_p: Prior log-variance.
        mu_q: Posterior mean.
        logvar_q: Posterior log-variance.

    Returns:
        The per-step per-dimension KL.
    """
    return 0.5 * (
        logvar_p - logvar_q + (logvar_q.exp() + (mu_q - mu_p) ** 2) / logvar_p.exp() - 1.0
    )


# ---------------------------------------------------------------------------------------
# What the delegation supplies
# ---------------------------------------------------------------------------------------
def test_the_two_models_hand_the_objective_the_same_four_things(pair):
    """Asserted before any output is compared. The delegation's whole content is these four
    arguments, so two models that agreed on the result while disagreeing here would be agreeing by
    accident."""
    model, sibling, _outputs, _batch = pair

    assert model.geometry == sibling.geometry
    assert torch.equal(model.future_index, sibling.future_index)
    assert model.coverage_floor == sibling.coverage_floor
    assert model.logvar_clamp == sibling.logvar_clamp


# ---------------------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_every_metric_is_bitwise_equal_between_the_two_models(pair, likelihood):
    """Both observation models, because the shipped config uses one and the tiny config the other.
    ``torch.equal`` on every metric, not ``allclose`` on the total: these are two calls into the same
    function on the same tensors, so anything but bitwise equality is a difference in what was
    passed."""
    model, sibling, outputs, batch = pair

    ours = model.compute_loss(
        outputs, batch.fhr, weight=batch.weight, likelihood=likelihood, **_LOSS_KWARGS
    )
    theirs = sibling.compute_loss(
        outputs, batch.fhr, weight=batch.weight, likelihood=likelihood, **_LOSS_KWARGS
    )

    assert set(ours["metrics"]) == set(theirs["metrics"])
    differing = [
        key for key, value in ours["metrics"].items()
        if not torch.equal(torch.as_tensor(value), torch.as_tensor(theirs["metrics"][key]))
    ]
    assert not differing, differing
    assert ours["likelihood"] == theirs["likelihood"] == likelihood


def test_the_comparison_would_notice_a_different_geometry(pair):
    """The negative control for the test above: two ``compute_loss`` calls that agreed no matter
    what would prove nothing. A model built at a different coverage floor admits a different set of
    anchors, and the per-anchor readouts move."""
    _model, sibling, outputs, batch = pair
    torch.manual_seed(0)
    other = SeqVaeLagAttnTrfE2E(**dict(TINY_KWARGS, coverage_floor=0.0)).eval()

    ours = other.compute_loss(outputs, batch.fhr, weight=batch.weight, **_LOSS_KWARGS)
    theirs = sibling.compute_loss(outputs, batch.fhr, weight=batch.weight, **_LOSS_KWARGS)

    assert not torch.equal(
        torch.as_tensor(ours["metrics"]["total_loss"]),
        torch.as_tensor(theirs["metrics"]["total_loss"]),
    )


def test_the_objective_refuses_an_unknown_likelihood_through_this_model(pair):
    """The delegation passes it straight through, so the shared refusal is what an operator sees."""
    model, _sibling, outputs, batch = pair

    with pytest.raises(ValueError, match="likelihood"):
        model.compute_loss(outputs, batch.fhr, weight=batch.weight, likelihood="huber")


# ---------------------------------------------------------------------------------------
# The KL
# ---------------------------------------------------------------------------------------
def test_the_kld_tensor_is_the_closed_form(pair):
    """Against the formula, not against the sibling. Two delegations to one function agree with each
    other by construction; what has to be true is that the function is the KL."""
    model, _sibling, outputs, _batch = pair

    got = model.kld_tensor(
        mu_prior=outputs["mu_prior"],
        logvar_prior=outputs["logvar_prior"],
        mu_post=outputs["mu_post"],
        logvar_post=outputs["logvar_post"],
    )
    expected = _closed_form_kld(
        outputs["mu_prior"], outputs["logvar_prior"],
        outputs["mu_post"], outputs["logvar_post"],
    )

    assert got.shape == outputs["mu_prior"].shape
    assert torch.allclose(got, expected, atol=1e-6)


def test_the_kld_tensor_is_not_identically_zero_on_a_perturbed_posterior(
    tiny_kwargs, raw_inputs, perturb_posterior
):
    """The vacuity control. At initialisation the posterior equals the prior exactly, so the closed
    form and any wrong implementation of it both return zero, and the test above passes on either."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfE2E(**tiny_kwargs).eval()
    perturb_posterior(model)
    with torch.no_grad():
        out = model(*raw_inputs)

    got = model.kld_tensor(
        mu_prior=out["mu_prior"], logvar_prior=out["logvar_prior"],
        mu_post=out["mu_post"], logvar_post=out["logvar_post"],
    )
    expected = _closed_form_kld(
        out["mu_prior"], out["logvar_prior"], out["mu_post"], out["logvar_post"]
    )

    assert float(got.abs().max()) > 1e-6
    assert torch.allclose(got, expected, atol=1e-6)
    # Non-negative, as a KL between two Gaussians must be; -1e-7 is the last-bit cancellation floor.
    assert float(got.min()) >= -1e-7
