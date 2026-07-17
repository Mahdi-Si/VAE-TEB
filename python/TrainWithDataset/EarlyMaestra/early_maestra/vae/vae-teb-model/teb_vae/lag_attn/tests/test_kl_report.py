r"""The KL reporting keys mean different things, and conflating them would hide a collapse.

Three numbers come out of the loss and they are not interchangeable:

* ``kld_train`` -- the optimised term, free-bits floored. What gradient descent sees.
* ``kld_raw`` -- the same KL over the same support, *un*-floored and detached. What the model is
  actually telling us about the source.
* ``kld_active_frac`` -- how many latent dimensions are carrying any of it.

The distinction is load-bearing. Free-bits pays the model a floor of KL per dimension whether or
not it earned it, so ``kld_train`` stays healthy-looking through a total posterior collapse.
``kld_raw`` is the one that goes to zero. And a model can post a fine total while routing all of
it through one dimension, which is what ``kld_active_frac`` exists to catch.

**Every test here perturbs the posterior first.** At init the delta heads are zero, so $K_t
\equiv 0$ and every assertion below would pass on a model that was entirely wrong.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn.nets.model import SeqVaeLagAttn


def _model_and_out(prod_kwargs, inputs, perturb_posterior, **overrides):
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**dict(prod_kwargs, **overrides)).eval()
    perturb_posterior(model)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)
    return model, out


def test_the_perturbation_actually_makes_the_kl_nonzero(prod_kwargs, inputs, perturb_posterior):
    """The premise every other test in this file rests on."""
    _, out = _model_and_out(prod_kwargs, inputs, perturb_posterior)
    assert out["kld_per_t"].abs().max().item() > 1e-6


def test_kld_per_t_is_the_raw_full_length_sum(prod_kwargs, inputs, perturb_posterior):
    """It is the reporting curve: raw, full-T, summed over latent dims. Not the trained term."""
    model, out = _model_and_out(prod_kwargs, inputs, perturb_posterior)
    kld_btd = model.kld_tensor(
        mu_prior=out["mu_prior"],
        logvar_prior=out["logvar_prior"],
        mu_post=out["mu_post"],
        logvar_post=out["logvar_post"],
    )
    assert out["kld_per_t"].shape == inputs[0].shape[:2]
    assert torch.allclose(out["kld_per_t"], kld_btd.sum(dim=-1), atol=1e-6)


def test_per_head_kl_sums_to_the_total(prod_kwargs, inputs, perturb_posterior):
    _, out = _model_and_out(prod_kwargs, inputs, perturb_posterior)
    assert torch.allclose(
        out["kld_per_t_per_head"].sum(dim=-1), out["kld_per_t"], atol=1e-5
    )


def test_te_lag_map_sums_over_lags_to_the_raw_per_step_kl(
    prod_kwargs, inputs, perturb_posterior
):
    """The attribution redistributes $K_t$ across lags; it must not create or destroy any."""
    _, out = _model_and_out(prod_kwargs, inputs, perturb_posterior)
    # Attention rows are a distribution, so the weights sum to one and the lag axis collapses
    # back to the per-step total.
    assert torch.allclose(out["te_lag_map"].sum(dim=-1), out["kld_per_t"], atol=1e-4)


def test_free_bits_cannot_reach_the_reported_attribution(
    prod_kwargs, inputs, perturb_posterior
):
    """Free-bits is a training device. It must not inflate what the model reports.

    Structural rather than incidental: ``free_bits`` is a ``compute_loss`` argument, and
    ``te_lag_map`` is built in ``forward``, which never sees it. Asserted by running the loss at
    two very different floors and checking the forward-side attribution is untouched -- so a
    future change that *did* route a floored KL into the attribution would be caught.
    """
    model, out = _model_and_out(prod_kwargs, inputs, perturb_posterior)
    before = out["te_lag_map"].clone()

    heavily_floored = model.compute_loss(out, inputs[0], inputs[1], free_bits=5.0)
    unfloored = model.compute_loss(out, inputs[0], inputs[1], free_bits=0.0)

    # The floor really did bite, so the comparison below is not vacuous.
    assert heavily_floored["kld_train"].item() > unfloored["kld_train"].item()
    assert torch.equal(out["te_lag_map"], before)
    assert torch.allclose(out["te_lag_map"].sum(dim=-1), out["kld_per_t"], atol=1e-4)


def test_kld_train_is_at_least_kld_raw(prod_kwargs, inputs, perturb_posterior):
    r"""Free-bits clamps each per-dim KL *up* before masking, so the ordering is structural."""
    model, out = _model_and_out(prod_kwargs, inputs, perturb_posterior)
    loss = model.compute_loss(out, inputs[0], inputs[1], free_bits=0.1)
    assert loss["kld_train"].item() >= loss["kld_raw"].item() - 1e-6


def test_kld_raw_ignores_free_bits(prod_kwargs, inputs, perturb_posterior):
    """The point of the key: it is the number free-bits cannot flatter."""
    model, out = _model_and_out(prod_kwargs, inputs, perturb_posterior)
    floored = model.compute_loss(out, inputs[0], inputs[1], free_bits=0.5)
    unfloored = model.compute_loss(out, inputs[0], inputs[1], free_bits=0.0)
    assert torch.allclose(floored["kld_raw"], unfloored["kld_raw"])
    assert floored["kld_train"].item() > unfloored["kld_train"].item()


def test_kld_train_aliases_the_optimised_term(prod_kwargs, inputs, perturb_posterior):
    model, out = _model_and_out(prod_kwargs, inputs, perturb_posterior)
    loss = model.compute_loss(out, inputs[0], inputs[1], free_bits=0.1)
    assert loss["kld_train"] is loss["kld_loss"]


def test_kld_raw_is_detached_so_only_kld_train_carries_gradient(
    prod_kwargs, inputs, perturb_posterior
):
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**prod_kwargs).eval()
    perturb_posterior(model)
    torch.manual_seed(0)
    out = model(*inputs)
    loss = model.compute_loss(out, inputs[0], inputs[1], free_bits=0.1)

    assert not loss["kld_raw"].requires_grad
    assert loss["kld_train"].requires_grad


def test_the_kl_term_carries_gradient(prod_kwargs, inputs, perturb_posterior):
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**prod_kwargs)
    perturb_posterior(model)
    torch.manual_seed(0)
    out = model(*inputs)
    model.compute_loss(out, inputs[0], inputs[1], beta=1.0)["total_loss"].backward()

    grad = model.posterior_head.delta_mu_head.weight.grad
    assert grad is not None and grad.abs().max().item() > 0.0


def test_kld_active_frac_is_a_fraction(prod_kwargs, inputs, perturb_posterior):
    _, out = _model_and_out(prod_kwargs, inputs, perturb_posterior)
    value = out["kld_active_frac"]
    assert value.ndim == 0
    assert 0.0 <= value.item() <= 1.0


def test_kld_active_frac_is_zero_at_init(prod_kwargs, inputs):
    """No perturbation here on purpose: a collapsed latent must read as collapsed."""
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**prod_kwargs).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)
    assert out["kld_active_frac"].item() == 0.0


def test_disabling_the_kl_term_zeroes_both_readouts(prod_kwargs, inputs, perturb_posterior):
    model, out = _model_and_out(prod_kwargs, inputs, perturb_posterior)
    loss = model.compute_loss(out, inputs[0], inputs[1], compute_kld_loss=False)
    assert loss["kld_loss"].item() == 0.0
    assert loss["kld_raw"].item() == 0.0
