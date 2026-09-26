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
from torch import nn

from teb_vae.lag_attn.nets.blocks import CausalGroupNorm
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


def test_forward_returns_the_contract(prod_kwargs, inputs):
    """The key set, every value a tensor (no ``None`` placeholder), and the boundary shapes."""
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**prod_kwargs).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)
    assert set(out) == _FORWARD_KEYS
    assert all(torch.is_tensor(value) for value in out.values())

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


@pytest.mark.parametrize(
    "c_y, c_u, use_up_st",
    [(109, 15, False), (109, 15, True), (87, 101, True)],
    ids=["source-ablation", "c_u-not-derived-from-toggle", "legacy-checkpoint-geometry"],
)
def test_the_declared_widths_are_honoured_rather_than_derived(tiny_kwargs, c_y, c_u, use_up_st):
    """The constructor trusts the caller's widths; the task checks them against the batch.

    The widths are a property of the HDF5, which this constructor cannot see. It once overwrote
    ``c_u`` with a constant chosen by ``use_up_st``, and later refused the pairing; both went stale
    the moment the dataset pipeline changed its phase-harmonic selection, taking every
    pre-migration checkpoint's rebuild with them (such a blob carries the old widths).
    Asserting ``model.c_u`` alone would pass against an implementation that recomputed the width
    actually used to build the adapter, so the adapters are read too.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**dict(tiny_kwargs, c_y=c_y, c_u=c_u, use_up_st=use_up_st))
    assert (model.c_y, model.c_u) == (c_y, c_u)
    assert model.target_adapter.linear.in_features == c_y
    assert model.source_adapter.linear.in_features == c_u


@pytest.mark.parametrize("zeroed", [{"c_u": 0}, {"c_y": 0}])
def test_a_zero_channel_width_raises(tiny_kwargs, zeroed):
    """Zero is the one width the constructor can reject without knowing the dataset.

    ``nn.Linear(0, d_model)`` is legal and returns its bias broadcast over the batch, so a zero
    width builds a model that trains to completion having never read that stream -- then reports
    its KL as a transfer entropy. Same failure mode the ``max_lag`` guard exists to prevent.
    """
    with pytest.raises(ValueError, match="must be >= 1"):
        SeqVaeLagAttn(**dict(tiny_kwargs, **zeroed))


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


def test_causal_norm_replaces_exactly_the_encoder_norms(prod_kwargs):
    """Every encoder GroupNorm is swapped; the horizon core's are deliberately left alone.

    The core's norms pool over the forecast axis of one anchor, not across input time, so
    causalising them would be a change with no invariant behind it.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**prod_kwargs)
    encoders = (model.target_encoder, model.source_encoder)

    def count(cls) -> int:
        return sum(isinstance(m, cls) for encoder in encoders for m in encoder.modules())

    assert model.causal_norm is True
    assert count(nn.GroupNorm) == 0
    assert model.n_causalized_norms == count(CausalGroupNorm) > 0
    assert any(isinstance(m, nn.GroupNorm) for m in model.horizon_core.modules())


def test_both_decoders_share_the_models_horizon_core(prod_kwargs):
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**prod_kwargs)
    assert model.baseline_decoder.core is model.horizon_core
    assert model.residual_decoder.core is model.horizon_core
