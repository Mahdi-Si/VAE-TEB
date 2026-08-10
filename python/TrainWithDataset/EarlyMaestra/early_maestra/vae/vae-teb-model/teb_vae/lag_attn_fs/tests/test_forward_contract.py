r"""The forward return contract: the exact key set, the shapes, and the raw grid proved inert.

The key set is asserted by equality, not by subset: an extra key is how a bypass tensor would
first appear, and a missing one is how a downstream consumer starts reading defaults. It is the
sibling's set exactly -- twenty entries -- because the forward is the sibling's forward, and the
only thing this model changes about it is the last axis of the four forecast tensors.

One inherited attribute needs saying out loud. ``future_index`` is a $(T_{\mathrm{valid}}, H, R)$
grid of **raw sample indices**, registered by the base constructor and used by the base model to
gather its raw target. This model inherits it and gathers nothing with it. Rather than assert its
absence -- removing it would mean overriding ``__init__``, which is exactly what the width hook
exists to avoid -- the test below asserts something stronger and cheaper to keep true: zeroing
it leaves every reported metric bitwise unchanged, so the raw grid reaches no number this model
produces.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.tests.conftest import (
    BATCH,
    SEQ_LEN,
    SHIPPED_KWARGS,
    make_patterned_batch,
    shipped_gated_kwargs,
)

#: The sibling's documented forward keys. Written out rather than imported from its test module:
#: this is the contract *this* model promises, and importing it would make the two indistinguish-
#: able if either drifted.
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

_FORECAST_KEYS = ("mu_base", "logvar_base", "mu_full", "logvar_full")

_KEPT_CHANNELS = 78
_ALL_CHANNELS = 109


def _forward(kwargs, inputs, perturb=None):
    torch.manual_seed(0)
    model = SeqVaeLagAttnFs(**kwargs).eval()
    if perturb is not None:
        perturb(model)
    torch.manual_seed(0)
    with torch.no_grad():
        return model, model(*inputs)


def _shipped_inputs(batch_size: int = 2):
    """Seeded target and source streams at the production geometry."""
    generator = torch.Generator().manual_seed(0)
    length = SHIPPED_KWARGS["sequence_length"]
    return (
        torch.randn(batch_size, length, 43, generator=generator),
        torch.randn(batch_size, length, 66, generator=generator),
        torch.randn(batch_size, length, 58, generator=generator),
    )


# ---------------------------------------------------------------------------------------
# The key set
# ---------------------------------------------------------------------------------------
def test_the_forward_returns_exactly_the_documented_key_set(tiny_kwargs, inputs):
    _, out = _forward(tiny_kwargs, inputs)

    assert set(out.keys()) == _DOCUMENTED_KEYS
    assert len(_DOCUMENTED_KEYS) == 20


def test_the_removed_pathways_do_not_resurface(tiny_kwargs, inputs):
    """The two names this architecture removed. An extra key is how a bypass reappears, and it
    would reappear under its old name before it reappeared under a new one."""
    _, out = _forward(tiny_kwargs, inputs)

    assert "decoder_state" not in out
    assert "delta_mu_src" not in out


# ---------------------------------------------------------------------------------------
# The shapes
# ---------------------------------------------------------------------------------------
def test_the_forecast_shapes_at_the_production_geometry():
    """$(B, 270, 30, 78)$ -- the block the objective sums over, and the whole difference from the
    raw model's $(B, 270, 30, 16)$."""
    model, out = _forward(shipped_gated_kwargs(), _shipped_inputs())

    expected = (2, model.geometry.t_valid, model.horizon, _KEPT_CHANNELS)
    assert expected == (2, 270, 30, 78)
    for key in _FORECAST_KEYS:
        assert out[key].shape == expected, key


def test_the_forecast_shapes_follow_the_declared_width_when_ungated():
    """$(B, 270, 30, 109)$ at ``causal_reach_budget_s: null``."""
    model, out = _forward(dict(SHIPPED_KWARGS), _shipped_inputs())

    expected = (2, model.geometry.t_valid, model.horizon, _ALL_CHANNELS)
    for key in _FORECAST_KEYS:
        assert out[key].shape == expected, key


def test_decoding_covers_the_valid_anchor_range_only(tiny_kwargs, inputs):
    """$(B, T - H, H, C)$, not $(B, T, H, C)$: the tail anchors have no fully observed future
    window and are never decoded."""
    model, out = _forward(tiny_kwargs, inputs)

    expected = (BATCH, model.geometry.t_valid, model.horizon, model.decoder_out_channels)
    assert expected[1] == SEQ_LEN - model.horizon
    for key in _FORECAST_KEYS:
        assert out[key].shape == expected, key


def test_the_latent_and_state_shapes_are_unchanged(tiny_kwargs, inputs):
    """The latent is $d_z$-wide whatever the target is; a target-domain swap that moved it would
    mean the two models' KL readouts were not the same quantity."""
    model, out = _forward(tiny_kwargs, inputs)

    for key in ("mu_prior", "logvar_prior", "raw_logvar_prior", "mu_post", "logvar_post",
                "z_prior", "z_post"):
        assert out[key].shape == (BATCH, SEQ_LEN, model.d_z), key
    for key in ("target_state", "source_state"):
        assert out[key].shape == (BATCH, SEQ_LEN, model.d_model), key


def test_the_attention_and_kl_readout_shapes_are_unchanged(tiny_kwargs, inputs):
    model, out = _forward(tiny_kwargs, inputs)
    num_lags = model.max_lag + 1

    assert out["attn_weights"].shape == (BATCH, SEQ_LEN, model.num_heads, num_lags)
    assert out["attended_source_heads"].shape == (BATCH, SEQ_LEN, model.num_heads, 8)
    assert out["kld_per_t"].shape == (BATCH, SEQ_LEN)
    assert out["kld_per_t_per_head"].shape == (BATCH, SEQ_LEN, model.num_heads)
    assert out["source_kl_lag_map"].shape == (BATCH, SEQ_LEN, num_lags)


def test_the_saturation_diagnostics_are_scalars_in_unit_range(tiny_kwargs, inputs):
    _, out = _forward(tiny_kwargs, inputs)

    for key in ("mu_prior_sat_frac", "delta_mu_sat_frac"):
        assert out[key].dim() == 0
        assert 0.0 <= float(out[key]) <= 1.0


# ---------------------------------------------------------------------------------------
# The paired draw
# ---------------------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------------------
# The inherited raw index grid reaches nothing
# ---------------------------------------------------------------------------------------
def test_the_inherited_raw_index_grid_is_present_and_non_persistent(tiny_kwargs, inputs):
    """It is the base constructor's, and this model has no way to drop it without overriding
    ``__init__`` -- which would break the trainer's signature sweep for a megabyte. Non-persistent,
    so it reaches no checkpoint; unused, which the next test makes operational."""
    model, _ = _forward(tiny_kwargs, inputs)

    assert model.future_index.shape == (model.geometry.t_valid, model.horizon, model.geometry.r)
    assert "future_index" not in model.state_dict()


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
