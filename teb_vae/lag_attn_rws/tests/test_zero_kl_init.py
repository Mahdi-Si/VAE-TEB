r"""At initialisation the model must assert that the source says nothing -- exactly.

The posterior is a zero-initialised residual on the prior, one $\epsilon$ serves both samples,
and the twice-invoked shared decoder carries no dropout, so at init the KL is exactly $0$,
$z^p = z^q$ elementwise, and the base and full forecasts are **bitwise identical in train
mode** -- not merely under ``eval()``. Train mode is the point: it is what the zero decoder and
attention dropout buy, and what turns the base-minus-full readout into a noise-free null.

This also documents the trap that shapes the rest of this suite: because the KL is identically
zero here, **any** KL assertion on a freshly-built model passes, including on a model that is
completely wrong. The perturbation test at the bottom is what makes the others meaningful, and
every other test in the suite that claims to check KL behaviour must perturb first.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

_TOL = 1e-6


def _closed_form_kl(out: dict) -> torch.Tensor:
    r"""$\mathrm{KL}(q \Vert p)$ per step per dimension, from the returned parameters."""
    return 0.5 * (
        out["logvar_prior"]
        - out["logvar_post"]
        + (out["logvar_post"].exp() + (out["mu_post"] - out["mu_prior"]) ** 2)
        / out["logvar_prior"].exp()
        - 1.0
    )


def _train_mode_forward(tiny_kwargs, inputs, perturb=None):
    # Dropout deliberately ON: the bitwise identities below must hold in train mode with the
    # encoders dropping activations, because only the decoder and attention are dropout-free.
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**dict(tiny_kwargs, dropout=0.1))
    if perturb is not None:
        perturb(model)
    model.train()
    torch.manual_seed(0)
    return model(*inputs)


def test_the_kl_is_exactly_zero_at_init(tiny_kwargs, inputs):
    out = _train_mode_forward(tiny_kwargs, inputs)
    assert float(_closed_form_kl(out).abs().max()) == 0.0
    # The model's own readouts agree: per-step KL and its lag attribution are exactly zero.
    assert float(out["kld_per_t"].abs().max()) == 0.0
    assert float(out["source_kl_lag_map"].abs().max()) == 0.0


def test_the_posterior_equals_the_prior_at_init(tiny_kwargs, inputs):
    out = _train_mode_forward(tiny_kwargs, inputs)
    assert torch.equal(out["mu_post"], out["mu_prior"])
    assert torch.equal(out["logvar_post"], out["logvar_prior"])


def test_the_latent_samples_are_identical_at_init(tiny_kwargs, inputs):
    out = _train_mode_forward(tiny_kwargs, inputs)
    assert torch.equal(out["z_prior"], out["z_post"])


def test_base_and_full_forecasts_are_bitwise_identical_in_train_mode(tiny_kwargs, inputs):
    """The identity that decoder dropout would break: one module, two invocations, two
    independent masks. Zero dropout in the decoder is what makes this exact."""
    out = _train_mode_forward(tiny_kwargs, inputs)
    assert torch.equal(out["mu_base"], out["mu_full"])
    assert torch.equal(out["logvar_base"], out["logvar_full"])


def test_everything_above_becomes_false_once_perturbed(tiny_kwargs, inputs, perturb_posterior):
    """The zero must be a property of the init, not of the model being unable to produce a KL.

    Without this, a model whose KL was structurally stuck at zero -- a broken posterior, a
    detached graph -- would pass every test above.
    """
    out = _train_mode_forward(tiny_kwargs, inputs, perturb=perturb_posterior)
    assert float(_closed_form_kl(out).abs().max()) > _TOL
    assert not torch.equal(out["z_prior"], out["z_post"])
    assert not torch.equal(out["mu_base"], out["mu_full"])
