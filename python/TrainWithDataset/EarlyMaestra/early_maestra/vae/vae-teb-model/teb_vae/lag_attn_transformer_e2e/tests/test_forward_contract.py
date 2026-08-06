r"""The forward return contract, asserted against the model this one is compared with.

The sibling's own copy of this file pins its key set and shapes against a table written in the test.
Here that would be the wrong instrument twice over. A table is a *second* copy of the contract, so
the two could drift apart while both files stayed green; and what this package actually claims is
not "these twenty keys" but "**the same** twenty keys, at the same shapes, as the model it replaces
the input of" -- which is a fact about the two models and can only be measured by running both.

So every structural assertion here builds a ``SeqVaeLagAttnTrfRws`` at the geometry that matches
this one, feeds it the same stub batch's feature blocks, and compares. What is written out rather
than compared is only what the comparison could not see: that the shapes are the ones the geometry
implies, and that neither model grew a key named for a pathway this architecture family removed.

The two keyword sets differ in exactly the ways they must -- this one has no ``c_y``, ``c_u`` or
``use_up_st`` and carries a front-end kernel schedule, and its ``warmup_period`` is larger because a
four-stage stride-2 cascade needs the room -- and none of those touch a returned shape. The
assertion below that the two geometries agree is what keeps that true rather than assumed.
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
#: more warm-up than the sibling's two steps to keep its reach inside the budget -- and while no
#: returned *shape* depends on the warm-up, a comparison that had to carve out an exception would be
#: one exception away from carving out another.
_SIBLING_MATCHED = dict(SIBLING_TINY_KWARGS, warmup_period=TINY_WARMUP_PERIOD)

#: Keys named for pathways this architecture family does not have. Asserted absent by name as well
#: as by the set comparison, because the comparison would go quietly green if *both* models grew
#: one -- and a shared decoder-state head is exactly the kind of change that would land in the
#: shared decoder module and reach both at once.
_REMOVED_PATHWAY_KEYS = ("decoder_state", "delta_mu_src")


def _forward(tiny_kwargs, inputs, perturb=None):
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfE2E(**tiny_kwargs).eval()
    if perturb is not None:
        perturb(model)
    torch.manual_seed(0)
    with torch.no_grad():
        return model, model(*inputs)


def _sibling_forward():
    """One forward of the model this one is compared against, on the same stub batch.

    The stub batch carries both representations of the same recording -- the four stored feature
    blocks and the two raw signals -- so the two models are fed the same *sample*, which is what
    makes a key-set and shape comparison between them meaningful rather than incidental.

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


@pytest.fixture(scope="module")
def sibling():
    """The comparison model's forward, built once for the whole module."""
    return _sibling_forward()


# ---------------------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------------------
def test_the_two_models_share_the_geometry_the_shapes_are_read_from(tiny_kwargs, sibling):
    """Asserted first, because everything below compares shapes: two models at different geometries
    would agree or disagree for reasons that have nothing to do with their return contracts."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfE2E(**tiny_kwargs)
    other, _ = sibling

    assert model.geometry == other.geometry
    assert (model.sequence_length, model.d_model, model.d_z) == (
        other.sequence_length, other.d_model, other.d_z
    )
    assert (model.horizon, model.raw_per_step, model.num_heads, model.max_lag) == (
        other.horizon, other.raw_per_step, other.num_heads, other.max_lag
    )


def test_the_forward_returns_exactly_the_siblings_key_set(tiny_kwargs, raw_inputs, sibling):
    """Equality, not a subset: an extra key is how a bypass tensor would first appear, and a missing
    one is how a downstream consumer starts reading defaults. Compared against the other model
    rather than against a list, so the two contracts cannot drift apart while both files pass."""
    _, out = _forward(tiny_kwargs, raw_inputs)
    _, reference = sibling

    assert set(out) == set(reference)
    assert len(out) == 20


def test_every_returned_tensor_has_the_siblings_shape(tiny_kwargs, raw_inputs, sibling):
    """Every key at once, and the mismatches reported together: a shape divergence usually lands in
    a family of keys rather than in one, so a list says which stage moved where a first failure
    would only say that something did."""
    _, out = _forward(tiny_kwargs, raw_inputs)
    _, reference = sibling

    mismatched = {
        key: (tuple(out[key].shape), tuple(reference[key].shape))
        for key in reference
        if key in out and out[key].shape != reference[key].shape
    }

    assert not mismatched, f"(this model, the model it is compared with): {mismatched}"


@pytest.mark.parametrize("key", _REMOVED_PATHWAY_KEYS)
def test_neither_model_returns_a_key_for_a_removed_pathway(tiny_kwargs, raw_inputs, sibling, key):
    """The set comparison above cannot see a key both models grew, and the decoder is shared, so a
    reinstated bypass would arrive in both at once."""
    _, out = _forward(tiny_kwargs, raw_inputs)
    _, reference = sibling

    assert key not in out
    assert key not in reference


# ---------------------------------------------------------------------------------------
# What the comparison cannot see: the shapes the geometry implies
# ---------------------------------------------------------------------------------------
def test_the_latent_and_state_shapes(tiny_kwargs, raw_inputs):
    model, out = _forward(tiny_kwargs, raw_inputs)
    for key in ("mu_prior", "logvar_prior", "raw_logvar_prior", "mu_post", "logvar_post",
                "z_prior", "z_post"):
        assert out[key].shape == (BATCH, SEQ_LEN, model.d_z), key
    for key in ("target_state", "source_state"):
        assert out[key].shape == (BATCH, SEQ_LEN, model.d_model), key


def test_the_attention_and_kl_readout_shapes(tiny_kwargs, raw_inputs):
    model, out = _forward(tiny_kwargs, raw_inputs)
    num_lags = model.max_lag + 1
    d_head = model.d_model // model.num_heads

    assert out["attn_weights"].shape == (BATCH, SEQ_LEN, model.num_heads, num_lags)
    assert out["attended_source_heads"].shape == (BATCH, SEQ_LEN, model.num_heads, d_head)
    assert out["kld_per_t"].shape == (BATCH, SEQ_LEN)
    assert out["kld_per_t_per_head"].shape == (BATCH, SEQ_LEN, model.num_heads)
    assert out["source_kl_lag_map"].shape == (BATCH, SEQ_LEN, num_lags)


def test_decoding_covers_the_valid_anchor_range_only(tiny_kwargs, raw_inputs):
    """(B, T - H, H, R), not (B, T, H, R): the tail anchors are never decoded."""
    model, out = _forward(tiny_kwargs, raw_inputs)
    expected = (BATCH, model.geometry.t_valid, model.horizon, model.raw_per_step)

    assert expected[1] == SEQ_LEN - model.horizon
    for key in ("mu_base", "logvar_base", "mu_full", "logvar_full"):
        assert out[key].shape == expected, key


def test_the_saturation_diagnostics_are_scalars_in_unit_range(tiny_kwargs, raw_inputs):
    _, out = _forward(tiny_kwargs, raw_inputs)
    for key in ("mu_prior_sat_frac", "delta_mu_sat_frac"):
        assert out[key].dim() == 0
        assert 0.0 <= float(out[key]) <= 1.0


# ---------------------------------------------------------------------------------------
# The shared epsilon
# ---------------------------------------------------------------------------------------
def test_one_epsilon_serves_both_latents_when_the_residual_is_zero(tiny_kwargs, raw_inputs):
    """At init q == p, so the shared draw makes the samples bitwise equal."""
    _, out = _forward(tiny_kwargs, raw_inputs)
    assert torch.equal(out["z_prior"], out["z_post"])


def test_one_epsilon_serves_both_latents_when_the_distributions_differ(
    tiny_kwargs, raw_inputs, perturb_posterior
):
    """The stronger claim: even off-init, both samples recover the *same* epsilon. Two independent
    draws would pass the at-init test above and still corrupt every base-minus-full readout with
    sampling noise."""
    _, out = _forward(tiny_kwargs, raw_inputs, perturb=perturb_posterior)
    assert not torch.equal(out["mu_post"], out["mu_prior"])  # genuinely off-init

    eps_prior = (out["z_prior"] - out["mu_prior"]) * torch.exp(-0.5 * out["logvar_prior"])
    eps_post = (out["z_post"] - out["mu_post"]) * torch.exp(-0.5 * out["logvar_post"])
    assert torch.allclose(eps_prior, eps_post, atol=1e-5)
