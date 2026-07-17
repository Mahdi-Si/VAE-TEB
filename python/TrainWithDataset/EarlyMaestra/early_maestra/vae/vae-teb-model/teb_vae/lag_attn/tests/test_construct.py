"""The model builds, forwards, and refuses inconsistent geometry loudly.

Every validation here guards a configuration that would otherwise produce a model that is
*wrong* rather than one that fails: a channel count that disagrees with the stream it will be
fed, a head width that does not tile the model width, a latent that cannot be partitioned across
the heads it claims to be attributable to. Two of the three would otherwise surface as a shape
error somewhere deep inside a forward, on a training box, an hour in.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.nets import model as model_module
from teb_vae.lag_attn.nets.model import SeqVaeLagAttn

# The forward contract, written out rather than derived from another model. Deriving it is how
# the tree this replaces tested it, and a key set derived from the thing under test cannot fail.
_FORWARD_KEYS = {
    "mu_prior",
    "logvar_prior",
    "raw_logvar_prior",
    "mu_post",
    "logvar_post",
    "z",
    "target_state",
    "source_state",
    "decoder_state",
    "attended_source",
    "attended_source_heads",
    "attn_weights",
    "mu_base",
    "logvar_base",
    "delta_mu_src",
    "mu_full",
    "logvar_full",
    "kld_per_t",
    "kld_per_t_per_head",
    "te_lag_map",
    "warmup_mask",
    "mu_prior_sat_frac",
    "delta_mu_sat_frac",
    "kld_active_frac",
}

_ENCODE_KEYS = {
    "mu_prior",
    "logvar_prior",
    "mu_post",
    "logvar_post",
    "z",
    "target_state",
    "source_state",
    "decoder_state",
    "attended_source",
    "attended_source_heads",
    "attn_weights",
}


def test_the_model_constructs_under_the_production_config(prod_kwargs):
    torch.manual_seed(0)
    assert isinstance(SeqVaeLagAttn(**prod_kwargs), SeqVaeLagAttn)


def test_forward_returns_the_contract(prod_kwargs, inputs):
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**prod_kwargs).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)
    assert set(out) == _FORWARD_KEYS


def test_forward_carries_no_raw_future_pred(prod_kwargs, inputs):
    """It was always ``None`` -- a non-tensor in a dict of tensors, from a decoder that raised."""
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**prod_kwargs).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)
    assert "raw_future_pred" not in out
    assert all(torch.is_tensor(value) for value in out.values())


def test_forward_shapes(prod_kwargs, inputs):
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**prod_kwargs).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)

    batch, seq_len = inputs[0].shape[0], inputs[0].shape[1]
    d_z, d_model = prod_kwargs["d_z"], prod_kwargs["d_model"]
    num_lags = prod_kwargs["max_lag"] + 1
    horizon, c_y = prod_kwargs["horizon"], prod_kwargs["c_y"]

    assert out["mu_prior"].shape == (batch, seq_len, d_z)
    assert out["raw_logvar_prior"].shape == out["logvar_prior"].shape
    assert out["target_state"].shape == (batch, seq_len, d_model)
    assert out["attn_weights"].shape == (batch, seq_len, prod_kwargs["num_heads"], num_lags)
    assert out["mu_full"].shape == (batch, seq_len, horizon, c_y)
    assert out["te_lag_map"].shape == (batch, seq_len, num_lags)
    assert out["kld_per_t"].shape == (batch, seq_len)
    assert out["warmup_mask"].shape == (seq_len,)


def test_encode_only_returns_its_contract(prod_kwargs, inputs):
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**prod_kwargs).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        out = model.encode_only(*inputs)
    assert set(out) == _ENCODE_KEYS


def test_encode_only_can_return_the_posterior_mean(prod_kwargs, inputs):
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**prod_kwargs).eval()
    with torch.no_grad():
        out = model.encode_only(*inputs, sample_z=False)
    assert torch.equal(out["z"], out["mu_post"])


def test_source_channel_count_must_agree_with_the_ablation_toggle(tiny_kwargs):
    with pytest.raises(ValueError, match="c_u"):
        SeqVaeLagAttn(**dict(tiny_kwargs, c_u=58, use_up_st=True))
    with pytest.raises(ValueError, match="c_u"):
        SeqVaeLagAttn(**dict(tiny_kwargs, c_u=101, use_up_st=False))


def test_the_source_ablation_config_constructs(tiny_kwargs):
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**dict(tiny_kwargs, c_u=58, use_up_st=False))
    assert model.c_u == 58


def test_head_width_must_tile_the_model_width(tiny_kwargs):
    with pytest.raises(ValueError, match="must equal d_model"):
        SeqVaeLagAttn(**dict(tiny_kwargs, num_heads=4, d_head=9))


def test_head_structured_latent_must_partition_evenly(tiny_kwargs):
    with pytest.raises(ValueError, match="d_z % num_heads"):
        SeqVaeLagAttn(**dict(tiny_kwargs, d_z=9, head_structured_latent=True))


@pytest.mark.parametrize("max_lag", [-1, -90])
def test_a_negative_max_lag_raises(tiny_kwargs, max_lag):
    """Nothing downstream objects to an empty lag window, which is exactly the problem.

    $L = \\mathrm{max\\_lag} + 1 \\le 0$ gives a zero-width attention window. The einsums reduce
    over a zero-length axis without complaint, the attended source collapses to the output
    projection's bias, and the model trains to completion having never read the source -- then
    reports its KL as a transfer-entropy measurement of it. A config typo must not be able to
    produce that.
    """
    with pytest.raises(ValueError, match="max_lag must be >= 0"):
        SeqVaeLagAttn(**dict(tiny_kwargs, max_lag=max_lag))


def test_max_lag_zero_is_legal(tiny_kwargs):
    """The boundary: lag 0 alone is a real, if degenerate, configuration -- the current step."""
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**dict(tiny_kwargs, max_lag=0))
    assert model.lag_attn.L == 1


def test_an_unknown_kld_support_raises(tiny_kwargs):
    with pytest.raises(ValueError, match="kld_support"):
        SeqVaeLagAttn(**dict(tiny_kwargs, kld_support="everything"))


def test_retired_flags_are_not_constructor_arguments(tiny_kwargs):
    """Smooth bounding and the residual posterior are the model now, not options."""
    for retired in ("logvar_bound", "posterior_logvar", "latent_stats_momentum"):
        with pytest.raises(TypeError):
            SeqVaeLagAttn(**dict(tiny_kwargs, **{retired: "whatever"}))


def test_the_dead_lag_bank_is_gone(prod_kwargs):
    """It was constructed on every model and never called; ``unfold`` views replaced it."""
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**prod_kwargs)
    assert not hasattr(model_module, "LagMemoryBankBuilder")
    assert not hasattr(model, "lag_bank")


def test_the_latent_stats_mechanism_is_gone(prod_kwargs):
    """It was the only thing in the model that logged, reduced across ranks, or read a batch."""
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**prod_kwargs)
    for retired in (
        "fit_latent_stats",
        "normalize_latent",
        "_default_batch_to_inputs",
        "_update_latent_running_stats",
    ):
        assert not hasattr(model, retired), f"{retired} survived"
    assert not any("running" in name for name, _ in model.named_buffers())


def test_the_version_stamp_class_attribute_is_gone(prod_kwargs):
    """Nothing read it: the checkpoint stamp comes from the live class name."""
    torch.manual_seed(0)
    assert not hasattr(SeqVaeLagAttn(**prod_kwargs), "model_class")


def test_freeze_unused_attn_proj_needs_head_structure(tiny_kwargs):
    """The projection is only unused when the posterior reads the per-head summaries instead."""
    torch.manual_seed(0)
    flat = SeqVaeLagAttn(**dict(tiny_kwargs, freeze_unused_attn_proj=True))
    assert flat.frozen_attn_proj is False
    assert all(p.requires_grad for p in flat.lag_attn.W_o.parameters())

    torch.manual_seed(0)
    structured = SeqVaeLagAttn(
        **dict(tiny_kwargs, freeze_unused_attn_proj=True, head_structured_latent=True)
    )
    assert structured.frozen_attn_proj is True
    assert not any(p.requires_grad for p in structured.lag_attn.W_o.parameters())


def test_causal_norm_is_off_by_default(tiny_kwargs):
    torch.manual_seed(0)
    assert SeqVaeLagAttn(**tiny_kwargs).n_causalized_norms == 0


def test_causal_norm_replaces_exactly_the_encoder_norms(prod_kwargs):
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**prod_kwargs)
    assert model.causal_norm is True
    assert model.n_causalized_norms == 10


def test_the_horizon_core_is_deliberately_left_leaky(prod_kwargs):
    """Its norms pool over the forecast axis of one anchor, not across input time.

    Causalising them would be a change with no invariant behind it.
    """
    from torch import nn

    torch.manual_seed(0)
    model = SeqVaeLagAttn(**prod_kwargs)
    assert any(isinstance(m, nn.GroupNorm) for m in model.horizon_core.modules())


def test_both_decoders_share_the_models_horizon_core(prod_kwargs):
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**prod_kwargs)
    assert model.baseline_decoder.core is model.horizon_core
    assert model.residual_decoder.core is model.horizon_core
