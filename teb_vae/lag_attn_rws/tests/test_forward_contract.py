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


def test_every_output_has_its_contract_shape(tiny_kwargs, inputs):
    """Latents and states per step, attention and KL readouts per lag, and the decode over the
    valid anchor range only: $(B, T - H, H, R)$, not $(B, T, H, R)$ -- the tail anchors are never
    decoded."""
    model, out = _forward(tiny_kwargs, inputs)
    num_lags = model.max_lag + 1
    decoded = (BATCH, model.geometry.t_valid, model.horizon, model.raw_per_step)
    assert decoded[1] == SEQ_LEN - model.horizon
    expected = {
        **{
            key: (BATCH, SEQ_LEN, model.d_z)
            for key in ("mu_prior", "logvar_prior", "raw_logvar_prior", "mu_post",
                        "logvar_post", "z_prior", "z_post")
        },
        "target_state": (BATCH, SEQ_LEN, model.d_model),
        "source_state": (BATCH, SEQ_LEN, model.d_model),
        "attn_weights": (BATCH, SEQ_LEN, model.num_heads, num_lags),
        "attended_source_heads": (BATCH, SEQ_LEN, model.num_heads, tiny_kwargs["d_head"]),
        "kld_per_t": (BATCH, SEQ_LEN),
        "kld_per_t_per_head": (BATCH, SEQ_LEN, model.num_heads),
        "source_kl_lag_map": (BATCH, SEQ_LEN, num_lags),
        **{key: decoded for key in ("mu_base", "logvar_base", "mu_full", "logvar_full")},
    }
    for key, shape in expected.items():
        assert out[key].shape == shape, key


def test_one_epsilon_serves_both_latents_when_the_distributions_differ(
    tiny_kwargs, inputs, perturb_posterior
):
    """Off-init, both samples recover the *same* epsilon (the at-init equality is pinned in
    ``test_zero_kl_init.py``). Two independent draws would corrupt every base-minus-full readout
    with sampling noise."""
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
