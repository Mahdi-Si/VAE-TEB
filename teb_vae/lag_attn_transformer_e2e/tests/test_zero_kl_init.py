r"""At initialisation the model must assert that the source says nothing -- exactly.

The posterior is a zero-initialised residual on the prior, one $\epsilon$ serves both samples, and
the twice-invoked shared decoder carries no dropout, so at init the KL is exactly $0$,
$z^p = z^q$ elementwise, and the base and full forecasts are **bitwise identical in train mode** --
not merely under ``eval()``. Train mode is the point: it is what the zero decoder and attention
dropout buy, and what turns the base-minus-full readout into a noise-free null.

Dropout does not break any of it, and here there is one more stochastic stage than the sibling has:
the front ends carry the configured dropout too, on top of the encoders. It survives for the same
structural reason, which is worth stating rather than inferring. The prior and the posterior are
read off **one common target forward** -- ``target_frontend`` then ``target_encoder``, computed once
and handed to ``prior_head`` and ``posterior_head`` alike -- so whatever activations dropout removes,
it removes from both branches identically. Two target passes would break the identity even with the
deltas at zero, and a front end invoked separately per branch is exactly how that would happen.

This also documents the trap that shapes the rest of the suite: because the KL is identically zero
here, **any** KL assertion on a freshly-built model passes, including on a model that is completely
wrong. The perturbation test at the bottom is what makes the others meaningful, and every other test
in this package that claims to check KL behaviour perturbs first.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E

#: Above which a perturbed KL counts as genuinely nonzero. Far below any meaningful coupling and far
#: above float noise on a collapsed one.
_TOL = 1e-6


def _closed_form_kl(out: dict) -> torch.Tensor:
    r"""$\mathrm{KL}(q \Vert p)$ per step per dimension, from the returned parameters.

    Written out here rather than called from the model, so the assertion is against the formula and
    not against the model agreeing with itself.
    """
    return 0.5 * (
        out["logvar_prior"]
        - out["logvar_post"]
        + (out["logvar_post"].exp() + (out["mu_post"] - out["mu_prior"]) ** 2)
        / out["logvar_prior"].exp()
        - 1.0
    )


def _forward(tiny_kwargs, inputs, perturb=None, *, train: bool = True):
    # Dropout deliberately ON: the bitwise identities below must hold with the front ends and the
    # encoders dropping activations, because only the decoder and both attentions are dropout-free.
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfE2E(**dict(tiny_kwargs, dropout=0.1))
    if perturb is not None:
        perturb(model)
    model.train(train)
    torch.manual_seed(0)
    return model(*inputs)


def test_the_kl_is_exactly_zero_at_init(tiny_kwargs, raw_inputs):
    out = _forward(tiny_kwargs, raw_inputs)
    assert float(_closed_form_kl(out).abs().max()) == 0.0
    # The model's own readouts agree: per-step KL and its lag attribution are exactly zero.
    assert float(out["kld_per_t"].abs().max()) == 0.0
    assert float(out["source_kl_lag_map"].abs().max()) == 0.0
    assert float(out["kld_per_t_per_head"].abs().max()) == 0.0


def test_the_posterior_equals_the_prior_at_init(tiny_kwargs, raw_inputs):
    out = _forward(tiny_kwargs, raw_inputs)
    assert torch.equal(out["mu_post"], out["mu_prior"])
    assert torch.equal(out["logvar_post"], out["logvar_prior"])


def test_the_latent_samples_are_identical_at_init(tiny_kwargs, raw_inputs):
    out = _forward(tiny_kwargs, raw_inputs)
    assert torch.equal(out["z_prior"], out["z_post"])


def test_base_and_full_forecasts_are_bitwise_identical_in_train_mode(tiny_kwargs, raw_inputs):
    """The identity that decoder dropout would break: one module, two invocations, two independent
    masks. Zero dropout in the decoder is what makes this exact."""
    out = _forward(tiny_kwargs, raw_inputs)
    assert torch.equal(out["mu_base"], out["mu_full"])
    assert torch.equal(out["logvar_base"], out["logvar_full"])


def test_the_front_end_dropout_is_actually_active_in_train_mode(tiny_kwargs, raw_inputs):
    """The control for the paragraph above, aimed at the stage this package added.

    If dropout were inert in the front ends -- unwired, or built at zero regardless of the configured
    value -- every identity in this file would hold for reasons that say nothing about one common
    target forward. Two train-mode passes over the same input, seeded differently, must differ; and
    the front end alone must be enough to make them differ, which is what the second half measures on
    the front end's own output rather than on the encoder's.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfE2E(**dict(tiny_kwargs, dropout=0.1)).train()

    torch.manual_seed(1)
    first = model(*raw_inputs)["target_state"]
    torch.manual_seed(2)
    second = model(*raw_inputs)["target_state"]
    assert not torch.equal(first, second), "dropout is not active; the train-mode claims are empty"

    y_raw, _u_raw, weight = raw_inputs
    torch.manual_seed(1)
    front_first = model.target_frontend(y_raw, weight)
    torch.manual_seed(2)
    front_second = model.target_frontend(y_raw, weight)
    assert not torch.equal(front_first, front_second), "the front end carries no active dropout"


def test_the_same_identities_hold_under_eval(tiny_kwargs, raw_inputs):
    """``eval()`` is the mode the diagnostic figure and the permutation control run in, so the
    identities are asserted there too rather than inferred from the train-mode ones."""
    out = _forward(tiny_kwargs, raw_inputs, train=False)

    assert float(out["kld_per_t"].abs().max()) == 0.0
    assert torch.equal(out["z_prior"], out["z_post"])
    assert torch.equal(out["mu_base"], out["mu_full"])
    assert torch.equal(out["logvar_base"], out["logvar_full"])


def test_everything_above_becomes_false_once_perturbed(tiny_kwargs, raw_inputs, perturb_posterior):
    """The zero must be a property of the init, not of the model being unable to produce a KL.

    Without this, a model whose KL was structurally stuck at zero -- a broken posterior, a detached
    graph, a source front end returning a constant -- would pass every test above.
    """
    out = _forward(tiny_kwargs, raw_inputs, perturb=perturb_posterior)
    assert float(_closed_form_kl(out).abs().max()) > _TOL
    assert float(out["kld_per_t"].abs().max()) > _TOL
    assert not torch.equal(out["z_prior"], out["z_post"])
    assert not torch.equal(out["mu_base"], out["mu_full"])
