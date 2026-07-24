r"""The forward return contract: the exact key set, the shapes, and the paired sampling.

The key set is asserted by equality, not by subset: an extra key is how a bypass tensor would
first appear, and a missing one is how a downstream consumer starts reading defaults.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.tests.conftest import BATCH, SEQ_LEN

_DOCUMENTED_KEYS = {
    "mu_prior",
    "logvar_prior",
    "raw_logvar_prior",
    "mu_post",
    "logvar_post",
    "z_prior",
    "z_post",
    "target_state",
    "source_state",
    "attended_source_heads",
    "attn_weights",
    "mu_base",
    "logvar_base",
    "mu_full",
    "logvar_full",
    "kld_per_t",
    "kld_per_t_per_head",
    "source_kl_lag_map",
    "mu_prior_sat_frac",
    "delta_mu_sat_frac",
}


def _forward(tiny_kwargs, inputs, perturb=None):
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**tiny_kwargs).eval()
    if perturb is not None:
        perturb(model)
    torch.manual_seed(0)
    with torch.no_grad():
        return model, model(*inputs)


def test_the_forward_returns_exactly_the_documented_key_set(tiny_kwargs, inputs):
    _, out = _forward(tiny_kwargs, inputs)
    assert set(out.keys()) == _DOCUMENTED_KEYS
    # The two pathways this architecture removed must not resurface under their old names.
    assert "decoder_state" not in out
    assert "delta_mu_src" not in out


def test_the_latent_and_state_shapes(tiny_kwargs, inputs):
    model, out = _forward(tiny_kwargs, inputs)
    for key in ("mu_prior", "logvar_prior", "raw_logvar_prior", "mu_post", "logvar_post",
                "z_prior", "z_post"):
        assert out[key].shape == (BATCH, SEQ_LEN, model.d_z), key
    for key in ("target_state", "source_state"):
        assert out[key].shape == (BATCH, SEQ_LEN, model.d_model), key


def test_the_attention_shapes(tiny_kwargs, inputs):
    model, out = _forward(tiny_kwargs, inputs)
    num_lags = model.max_lag + 1
    assert out["attn_weights"].shape == (BATCH, SEQ_LEN, model.num_heads, num_lags)
    assert out["attended_source_heads"].shape == (BATCH, SEQ_LEN, model.num_heads, 8)


def test_the_kl_readout_shapes(tiny_kwargs, inputs):
    model, out = _forward(tiny_kwargs, inputs)
    num_lags = model.max_lag + 1
    assert out["kld_per_t"].shape == (BATCH, SEQ_LEN)
    assert out["kld_per_t_per_head"].shape == (BATCH, SEQ_LEN, model.num_heads)
    assert out["source_kl_lag_map"].shape == (BATCH, SEQ_LEN, num_lags)


def test_decoding_covers_the_valid_anchor_range_only(tiny_kwargs, inputs):
    """(B, T - H, H, R), not (B, T, H, R): the tail anchors are never decoded."""
    model, out = _forward(tiny_kwargs, inputs)
    expected = (BATCH, model.geometry.t_valid, model.horizon, model.raw_per_step)
    assert expected[1] == SEQ_LEN - model.horizon
    for key in ("mu_base", "logvar_base", "mu_full", "logvar_full"):
        assert out[key].shape == expected, key


def test_one_epsilon_serves_both_latents_when_the_residual_is_zero(tiny_kwargs, inputs):
    """At init q == p, so the shared draw makes the samples bitwise equal."""
    _, out = _forward(tiny_kwargs, inputs)
    assert torch.equal(out["z_prior"], out["z_post"])


def test_one_epsilon_serves_both_latents_when_the_distributions_differ(
    tiny_kwargs, inputs, perturb_posterior
):
    """The stronger claim: even off-init, both samples recover the *same* epsilon. Two
    independent draws would pass the at-init test above and still corrupt every base-minus-full
    readout with sampling noise."""
    _, out = _forward(tiny_kwargs, inputs, perturb=perturb_posterior)
    assert not torch.equal(out["mu_post"], out["mu_prior"])  # genuinely off-init

    eps_prior = (out["z_prior"] - out["mu_prior"]) * torch.exp(-0.5 * out["logvar_prior"])
    eps_post = (out["z_post"] - out["mu_post"]) * torch.exp(-0.5 * out["logvar_post"])
    assert torch.allclose(eps_prior, eps_post, atol=1e-5)


def test_the_saturation_diagnostics_are_scalars_in_unit_range(tiny_kwargs, inputs):
    _, out = _forward(tiny_kwargs, inputs)
    for key in ("mu_prior_sat_frac", "delta_mu_sat_frac"):
        assert out[key].dim() == 0
        assert 0.0 <= float(out[key]) <= 1.0
