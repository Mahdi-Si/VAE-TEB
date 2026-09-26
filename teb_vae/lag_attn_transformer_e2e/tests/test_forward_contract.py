r"""The forward return contract, asserted against the model this one is compared with.

What this package claims is not "these keys" but "**the same** keys, at the same shapes, as the
model it replaces the input of" -- a fact about two models that can only be measured by running
both. So the structural assertions build a ``SeqVaeLagAttnTrfRws`` at the geometry that matches this
one, feed it the same stub batch's feature blocks, and compare. What is written out rather than
compared is only what the comparison could not see: that the shapes are the ones the geometry
implies, and that both latents are drawn from one shared $\epsilon$.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from teb_vae.lag_attn_transformer_e2e.tests.conftest import (
    BATCH,
    SEQ_LEN,
    TINY_WARMUP_PERIOD,
    make_stub_batch,
)
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
from teb_vae.lag_attn_transformer_rws.tests.conftest import TINY_KWARGS as SIBLING_TINY_KWARGS

#: The sibling at **this** model's warm-up, so "the same geometry" below is literal rather than
#: nearly true. The two tiny sets differ there by necessity -- a four-stage stride-2 cascade needs
#: more warm-up than the sibling's to keep its reach inside the budget.
_SIBLING_MATCHED = dict(SIBLING_TINY_KWARGS, warmup_period=TINY_WARMUP_PERIOD)


def _forward(tiny_kwargs, inputs, perturb=None):
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfE2E(**tiny_kwargs).eval()
    if perturb is not None:
        perturb(model)
    torch.manual_seed(0)
    with torch.no_grad():
        return model, model(*inputs)


@pytest.fixture(scope="module")
def sibling():
    """The comparison model's forward on the same stub batch, built once for the whole module.

    The stub batch carries both representations of the same recording -- the stored feature blocks
    and the two raw signals -- so the two models are fed the same *sample*.

    Returns:
        ``(model, outputs)``.
    """
    batch = make_stub_batch(BATCH, SEQ_LEN)
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**dict(_SIBLING_MATCHED)).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        outputs = model(
            batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1)
        )
    return model, outputs


# ---------------------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------------------
def test_the_forward_returns_the_siblings_key_set_at_the_siblings_shapes(
    tiny_kwargs, raw_inputs, sibling
):
    """Key-set equality, not a subset: an extra key is how a bypass tensor would first appear, and a
    missing one is how a downstream consumer starts reading defaults. Every shape mismatch is
    reported together, since a divergence usually lands in a family of keys."""
    _, out = _forward(tiny_kwargs, raw_inputs)
    _, reference = sibling

    assert set(out) == set(reference)
    mismatched = {
        key: (tuple(out[key].shape), tuple(reference[key].shape))
        for key in reference
        if out[key].shape != reference[key].shape
    }
    assert not mismatched, f"(this model, the model it is compared with): {mismatched}"


# ---------------------------------------------------------------------------------------
# What the comparison cannot see: the shapes the geometry implies
# ---------------------------------------------------------------------------------------
def test_the_returned_shapes_are_the_ones_the_geometry_implies(tiny_kwargs, raw_inputs):
    """Latents and states on the full $(B, T)$ grid, attention and KL readouts over the $L =
    \\mathrm{max\\_lag} + 1$ lags, forecasts over the valid anchors only -- $(B, T - H, H, r)$, not
    $(B, T, H, r)$, because the tail anchors are never decoded -- and the saturation diagnostics as
    unit-range scalars."""
    model, out = _forward(tiny_kwargs, raw_inputs)
    num_lags = model.max_lag + 1
    d_head = model.d_model // model.num_heads

    for key in ("mu_prior", "logvar_prior", "raw_logvar_prior", "mu_post", "logvar_post",
                "z_prior", "z_post"):
        assert out[key].shape == (BATCH, SEQ_LEN, model.d_z), key
    for key in ("target_state", "source_state"):
        assert out[key].shape == (BATCH, SEQ_LEN, model.d_model), key

    assert out["attn_weights"].shape == (BATCH, SEQ_LEN, model.num_heads, num_lags)
    assert out["attended_source_heads"].shape == (BATCH, SEQ_LEN, model.num_heads, d_head)
    assert out["kld_per_t"].shape == (BATCH, SEQ_LEN)
    assert out["kld_per_t_per_head"].shape == (BATCH, SEQ_LEN, model.num_heads)
    assert out["source_kl_lag_map"].shape == (BATCH, SEQ_LEN, num_lags)

    decoded = (BATCH, SEQ_LEN - model.horizon, model.horizon, model.raw_per_step)
    assert model.geometry.t_valid == SEQ_LEN - model.horizon
    for key in ("mu_base", "logvar_base", "mu_full", "logvar_full"):
        assert out[key].shape == decoded, key

    for key in ("mu_prior_sat_frac", "delta_mu_sat_frac"):
        assert out[key].dim() == 0
        assert 0.0 <= float(out[key]) <= 1.0


# ---------------------------------------------------------------------------------------
# The shared epsilon
# ---------------------------------------------------------------------------------------
def test_one_epsilon_serves_both_latents_when_the_distributions_differ(
    tiny_kwargs, raw_inputs, perturb_posterior
):
    """Off-init, both samples must recover the *same* epsilon. Two independent draws would still
    agree at init, where $q = p$, and would corrupt every base-minus-full readout with sampling
    noise."""
    _, out = _forward(tiny_kwargs, raw_inputs, perturb=perturb_posterior)
    assert not torch.equal(out["mu_post"], out["mu_prior"])  # genuinely off-init

    eps_prior = (out["z_prior"] - out["mu_prior"]) * torch.exp(-0.5 * out["logvar_prior"])
    eps_post = (out["z_post"] - out["mu_post"]) * torch.exp(-0.5 * out["logvar_post"])
    assert torch.allclose(eps_prior, eps_post, atol=1e-5)
