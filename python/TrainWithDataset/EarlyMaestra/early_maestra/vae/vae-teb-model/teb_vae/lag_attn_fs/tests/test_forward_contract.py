r"""The forward contract where this model can differ from the sibling it inherits it from.

The forward is the raw sibling's, unchanged, and its key set and latent, attention and KL readout
shapes are pinned in ``lag_attn_rws``. What is checked here is the part the target domain touches:
the decoded anchor range at this model's width, the shared reparameterisation draw off-init, and
the inherited raw index grid proved inert.

``future_index`` is a $(T_{\mathrm{valid}}, H, R)$ grid of **raw sample indices**, registered by
the base constructor and used by the base model to gather its raw target. This model inherits it
and gathers nothing with it. Rather than assert its absence -- removing it would mean overriding
``__init__``, which is exactly what the width hook exists to avoid -- the test below asserts
something stronger: zeroing it leaves every reported metric bitwise unchanged, so the raw grid
reaches no number this model produces.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.tests.conftest import BATCH, SEQ_LEN, make_patterned_batch

_FORECAST_KEYS = ("mu_base", "logvar_base", "mu_full", "logvar_full")


def _forward(kwargs, inputs, perturb=None):
    torch.manual_seed(0)
    model = SeqVaeLagAttnFs(**kwargs).eval()
    if perturb is not None:
        perturb(model)
    torch.manual_seed(0)
    with torch.no_grad():
        return model, model(*inputs)


def test_decoding_covers_the_valid_anchor_range_only(tiny_kwargs, inputs):
    """$(B, T - H, H, C)$, not $(B, T, H, C)$: the tail anchors have no fully observed future
    window and are never decoded."""
    model, out = _forward(tiny_kwargs, inputs)

    expected = (BATCH, model.geometry.t_valid, model.horizon, model.decoder_out_channels)
    assert expected[1] == SEQ_LEN - model.horizon
    for key in _FORECAST_KEYS:
        assert out[key].shape == expected, key


def test_one_epsilon_serves_both_latents_when_the_distributions_differ(
    tiny_kwargs, inputs, perturb_posterior
):
    """Two independent draws would corrupt every base-minus-full readout with sampling noise, and
    would still pass an at-init test -- so the claim is made off-init."""
    _, out = _forward(tiny_kwargs, inputs, perturb=perturb_posterior)
    assert not torch.equal(out["mu_post"], out["mu_prior"])  # genuinely off-init

    eps_prior = (out["z_prior"] - out["mu_prior"]) * torch.exp(-0.5 * out["logvar_prior"])
    eps_post = (out["z_post"] - out["mu_post"]) * torch.exp(-0.5 * out["logvar_post"])
    assert torch.allclose(eps_prior, eps_post, atol=1e-5)


def test_zeroing_the_raw_index_grid_moves_no_metric(tiny_kwargs, perturb_posterior):
    """The operational form of "this model does not gather a raw target". If any number it
    reports were built through that grid, destroying the grid would move it."""
    batch = make_patterned_batch()
    model, out = _forward(tiny_kwargs, (batch.fhr_st, batch.fhr_ph,
                                        torch.cat([batch.up_st, batch.up_ph], -1)),
                          perturb=perturb_posterior)
    features = torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1)

    reference = model.compute_loss(out, features, weight=batch.weight, beta_prior=0.11)
    with torch.no_grad():
        model.future_index.zero_()
    planted = model.compute_loss(out, features, weight=batch.weight, beta_prior=0.11)

    differing = [
        key
        for key, value in reference["metrics"].items()
        if not torch.equal(value, planted["metrics"][key])
    ]
    assert not differing, differing
    # Not vacuous: the perturbation puts every term on a non-zero value first.
    assert float(reference["metrics"]["source_conditioned_kl_raw"]) > 0.0
