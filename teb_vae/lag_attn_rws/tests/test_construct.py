r"""Construction invariants: what exists, what is frozen, what is refused.

The constructor's guarantees are structural -- a head-structured latent, one decoder, a frozen
attention output projection, zeroed posterior deltas *after* the generic init -- and each is
asserted on the assembled model, because several of them (the zeroing order above all) hold on
the parts in isolation and silently fail in composition.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from teb_vae.lag_attn.nets.heads import PriorHead
from teb_vae.lag_attn.nets.delays import ChannelDelay, ChannelGate
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws


def _model(kwargs, **overrides) -> SeqVaeLagAttnRws:
    torch.manual_seed(0)
    return SeqVaeLagAttnRws(**dict(kwargs, **overrides))


def test_the_model_constructs_at_the_tiny_geometry(tiny_kwargs):
    model = _model(tiny_kwargs)
    assert model.geometry.raw_len == 256
    assert model.geometry.t_valid == 12


def test_the_model_constructs_at_the_production_geometry(shipped_kwargs):
    model = _model(shipped_kwargs)
    assert model.geometry.raw_len == 4800
    assert model.geometry.t_valid == 270
    assert model.n_causalized_norms > 0


def test_an_indivisible_latent_is_rejected_naming_both_values(tiny_kwargs):
    with pytest.raises(ValueError, match=r"d_z=9.*num_heads=4"):
        _model(tiny_kwargs, d_z=9)


def test_a_head_geometry_mismatch_is_rejected(tiny_kwargs):
    with pytest.raises(ValueError, match="d_model"):
        _model(tiny_kwargs, d_head=16)


def test_a_negative_max_lag_is_rejected(tiny_kwargs):
    with pytest.raises(ValueError, match="max_lag"):
        _model(tiny_kwargs, max_lag=-1)


def test_zero_channel_widths_are_rejected(tiny_kwargs):
    """nn.Linear(0, d) is legal and returns its bias, so a zero width would build a model that
    trains to completion having never read that stream."""
    with pytest.raises(ValueError, match="c_y"):
        _model(tiny_kwargs, c_y=0)
    with pytest.raises(ValueError, match="c_u"):
        _model(tiny_kwargs, c_u=0)


def test_channel_width_values_are_not_validated_here(tiny_kwargs):
    """Widths are dataset facts, checked against the first real batch at the data boundary; a
    constructor constant is exactly what went stale in the tree this replaces."""
    model = _model(tiny_kwargs, c_y=7, c_u=3)
    assert model.c_y == 7 and model.c_u == 3


def test_a_degenerate_raw_geometry_is_rejected(tiny_kwargs):
    with pytest.raises(ValueError, match="degenerate"):
        _model(tiny_kwargs, horizon=16)  # horizon == T leaves no valid anchor


def test_no_decoder_state_head_and_no_second_decoder_exist(tiny_kwargs):
    model = _model(tiny_kwargs)
    assert not hasattr(model, "residual_decoder")
    assert not hasattr(model, "baseline_decoder")
    # The sibling's PriorHead is the class that carries a decoder_state head; its absence is
    # the absence of the bypass at the module level.
    assert not any(isinstance(m, PriorHead) for m in model.modules())
    assert not hasattr(model.prior_head, "decoder_state_head")


def test_the_posterior_is_head_structured(tiny_kwargs):
    assert _model(tiny_kwargs).posterior_head.head_structured is True


def test_the_delta_heads_are_zero_on_the_assembled_model(tiny_kwargs):
    """Asserted on the assembled model, not a bare PosteriorHead: the generic initialization
    xavier-fills every linear layer, so only the zeroed-after ordering makes this true."""
    model = _model(tiny_kwargs)
    for name in ("delta_mu_head", "delta_logvar_head"):
        module = getattr(model.posterior_head, name)
        layers = list(module) if isinstance(module, nn.ModuleList) else [module]
        for layer in layers:
            assert layer.weight.abs().max().item() == 0.0, f"{name} weight not zeroed"
            if layer.bias is not None:
                assert layer.bias.abs().max().item() == 0.0, f"{name} bias not zeroed"


def test_the_zero_survives_the_generic_weight_init(tiny_kwargs):
    model = _model(tiny_kwargs, init_weights=True)
    layers = list(model.posterior_head.delta_mu_head)
    assert all(layer.weight.abs().max().item() == 0.0 for layer in layers)


def test_the_attention_output_projection_is_frozen(tiny_kwargs):
    """W_o feeds nothing under the head-structured posterior; freezing it drops it from DDP's
    expectation set."""
    attn = _model(tiny_kwargs).lag_attn
    assert attn.W_o.weight.requires_grad is False
    assert attn.W_o.bias.requires_grad is False


def test_attention_dropout_is_zero(tiny_kwargs):
    """Dropout on the attention probabilities would break the exactness of the per-lag KL
    attribution -- the returned weights must be the ones the posterior consumed."""
    model = _model(tiny_kwargs, dropout=0.1)
    assert model.lag_attn.attn_dropout.p == 0.0


def test_the_query_projection_maps_the_latent_to_the_model_width(tiny_kwargs):
    """The attention query is a projection of the prior belief -- ``d_z`` in by default (the prior
    mean alone), ``2 * d_z`` in under ``query_uses_logvar`` (mean and log-variance), ``d_model``
    out either way."""
    default_proj = _model(tiny_kwargs).query_proj
    assert default_proj.in_features == 8 and default_proj.out_features == 32

    logvar_proj = _model(tiny_kwargs, query_uses_logvar=True).query_proj
    assert logvar_proj.in_features == 16 and logvar_proj.out_features == 32


def test_query_uses_logvar_preserves_the_forward_contract_and_lag_map_identity(
    tiny_kwargs, inputs, perturb_posterior
):
    """The wider query is target-only, so the forward still returns the contract keys and the
    lag-map still sums over lags to the per-step KL exactly (checked under perturbation, since the
    identity is vacuous at the zero-KL init)."""
    model = _model(tiny_kwargs, query_uses_logvar=True).eval()
    perturb_posterior(model)
    with torch.no_grad():
        out = model(*inputs)

    assert out["mu_prior"].shape[-1] == model.d_z
    assert out["attn_weights"].shape[2] == model.num_heads
    assert float(out["kld_per_t"].abs().max()) > 0.0, "perturbation failed; test is vacuous"
    total = out["source_kl_lag_map"].sum(dim=-1)
    assert torch.allclose(total, out["kld_per_t"], atol=1e-5, rtol=1e-5)


def test_a_narrower_extra_kernel_drops_the_costed_parameters(shipped_kwargs):
    """The two appended long-dilation blocks per encoder are the widest convs in the stack and
    dominate its parameter count; narrowing their kernel from 15 to 7 removes exactly
    4 x (15 - 7) x 128 x 128 = 524,288 parameters (two appended blocks per encoder, two encoders)."""
    baseline = sum(p.numel() for p in _model(shipped_kwargs, encoder_extra_kernel=15).parameters())
    narrow = sum(p.numel() for p in _model(shipped_kwargs, encoder_extra_kernel=7).parameters())

    assert baseline - narrow == 524288


def test_a_narrowed_extra_kernel_still_forwards(tiny_kwargs, inputs):
    """With an appended dilation the narrowed kernel is actually built and used, and the forward
    still returns the contract latent width."""
    model = _model(tiny_kwargs, encoder_extra_dilations=(4,), encoder_extra_kernel=7).eval()

    with torch.no_grad():
        out = model(*inputs)

    assert out["mu_prior"].shape[-1] == model.d_z


def test_horizon_depth_4_constructs_and_forwards(tiny_kwargs, inputs):
    """The ``horizon_depth: 4`` arm's decode path -- refine dilations $(1, 2, 4, 8)$ with a fourth
    per-block FiLM generator -- is otherwise unexercised. Construct it, confirm the fourth
    generator is present and zeroed (the identity-at-init re-zero covers every depth), and run a
    forward to the contract shape."""
    model = _model(tiny_kwargs, horizon_depth=4).eval()

    refine = model.horizon_core.refine
    assert len(refine.blocks) == 4
    assert refine.film is not None and len(refine.film) == 4
    assert all(float(generator.weight.abs().max()) == 0.0 for generator in refine.film)

    with torch.no_grad():
        out = model(*inputs)
    t_valid = model.geometry.t_valid
    assert out["mu_full"].shape == (inputs[0].shape[0], t_valid, model.horizon, model.raw_per_step)


def test_the_norm_groups_arm_threads_a_single_group_to_the_pre_norms(tiny_kwargs):
    """The ``conv_norm_groups`` arm's resolved value (1) reaches both encoders' conv pre-norms;
    the default keeps each block's ``min(8, d_model)``. Covers the arm value and the threading in
    one place, without a slow full-geometry construct."""
    from pathlib import Path

    from teb_vae.lag_attn.config import load_config

    arm = load_config(
        str(Path(__file__).resolve().parents[1] / "configs" / "sweep_norm_groups_1.yaml")
    )
    assert arm["model_config"]["VAE_model"]["conv_norm_groups"] == 1

    grouped = _model(tiny_kwargs, conv_norm_groups=1)
    default = _model(tiny_kwargs)
    for encoder in (grouped.target_encoder, grouped.source_encoder):
        for block in encoder.convs:
            assert block.pre_norm.num_groups == 1
    for encoder in (default.target_encoder, default.source_encoder):
        for block in encoder.convs:
            assert block.pre_norm.num_groups == min(8, default.d_model)


def test_causal_norm_replaces_every_encoder_group_norm(tiny_kwargs):
    # Three per encoder at the base schedule: the conv pre-norms only. This net uses the plain
    # residual stack (stack_skip_connection=False), so the two inter-block skip norms per encoder
    # the sibling carries are gone.
    model = _model(tiny_kwargs, causal_norm=True)
    assert model.n_causalized_norms == 6
    assert not any(
        isinstance(m, nn.GroupNorm)
        for encoder in (model.target_encoder, model.source_encoder)
        for m in encoder.modules()
    )


# ---------------------------------------------------------------------------------------
# The causal input guard
# ---------------------------------------------------------------------------------------
def _gated(kwargs, keep_target, delays_target, keep_source, delays_source):
    return _model(
        kwargs,
        target_keep_index=keep_target,
        target_delays=delays_target,
        source_keep_index=keep_source,
        source_delays=delays_source,
    )


def test_an_unguarded_model_has_no_gather_and_no_delay(tiny_kwargs):
    """Not an identity guard -- nothing at all, so the unguarded run is structurally the model
    that existed before the guard did."""
    model = _model(tiny_kwargs)

    assert model.target_gate is None and model.source_gate is None
    assert not any(isinstance(m, (ChannelGate, ChannelDelay)) for m in model.modules())
    assert model.source_delay_steps == 0


def test_an_unguarded_forward_is_bitwise_equal_to_an_identity_guard(tiny_kwargs, inputs):
    """The other direction: the gather-and-delay path, at the identity, must change nothing.
    Without this the two configurations could differ and only the guarded one would be tested."""
    plain = _model(tiny_kwargs).eval()
    identity = _gated(
        tiny_kwargs, tuple(range(109)), (0,) * 109, tuple(range(58)), (0,) * 58
    ).eval()

    torch.manual_seed(3)
    expected = plain(*inputs)
    torch.manual_seed(3)
    got = identity(*inputs)

    assert all(torch.equal(expected[key], got[key]) for key in expected)


def test_the_adapters_are_built_for_the_surviving_widths(tiny_kwargs):
    """The model still declares the full ``c_y`` / ``c_u`` -- the data boundary checks the batch
    against those -- while the adapters see only the survivors."""
    model = _gated(tiny_kwargs, (0, 5, 9), (1, 2, 3), (2, 7), (0, 4))

    assert (model.c_y, model.c_u) == (109, 58)
    assert model.target_adapter.linear.in_features == 3
    assert model.source_adapter.linear.in_features == 2


def test_a_gated_forward_reads_only_the_surviving_channels(tiny_kwargs, inputs):
    """Perturbing a pruned channel must change nothing: a channel that fails the reach budget
    has to be genuinely gone, not merely down-weighted."""
    keep = (0, 5, 9)
    model = _gated(tiny_kwargs, keep, (0, 0, 0), (2, 7), (0, 0)).eval()
    y_st, y_ph, u_stream = inputs

    torch.manual_seed(3)
    before = model(y_st, y_ph, u_stream)["mu_prior"]
    perturbed = y_st.clone()
    perturbed[..., 1] += 100.0  # channel 1 is not in keep
    torch.manual_seed(3)
    after = model(perturbed, y_ph, u_stream)["mu_prior"]

    assert torch.equal(before, after)


def test_the_gate_buffers_stay_out_of_the_state_dict(tiny_kwargs):
    """Their length is the surviving-channel count, so a persistent copy would make a checkpoint
    trained at one reach budget fail to load at another as "keys did not align"."""
    model = _gated(tiny_kwargs, (0, 5, 9), (1, 2, 3), (2, 7), (0, 4))

    assert not [name for name in model.state_dict() if "keep_index" in name]
    assert not [name for name in model.state_dict() if "delay_steps" in name]


@pytest.mark.parametrize(
    "keep, delays, match",
    [
        ((), (), "empty"),
        ((0, 200), (0, 0), "outside"),
        ((5, 1), (0, 0), "ascending"),
        ((0, 1, 2), (0, 0), "num_channels"),
    ],
    ids=["empty", "out-of-range", "unsorted", "length-mismatch"],
)
def test_a_malformed_target_gate_is_refused(tiny_kwargs, keep, delays, match):
    """Each of these would silently gather or delay the wrong channels; an unsorted index is the
    subtlest, since the delay vector is positional against it."""
    with pytest.raises(ValueError, match=match):
        _model(tiny_kwargs, target_keep_index=keep, target_delays=delays)


def test_the_model_reports_its_own_source_delay(tiny_kwargs):
    """Every lag report adds this back. Exposed on the model rather than dug out of its internals
    by each consumer, which is how the training figure and the evaluation came to disagree."""
    model = _gated(tiny_kwargs, (0, 5, 9), (1, 2, 3), (2, 7), (4, 11))

    assert model.source_delay_steps == 11  # the max over source channels, not the target's 3
