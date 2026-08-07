r"""Construction invariants: what exists, what is refused, what is frozen, and what it all costs.

The constructor's guarantees are structural -- a head-structured latent, one decoder, a frozen
lag-attention output projection, three dropout sites pinned at zero, no recurrence and no
time-pooling normaliser on either history path -- and each is asserted on the **assembled** model,
because several of them hold on the parts in isolation and fail silently in composition.

The parameter budget is asserted as per-encoder subtotals and as *deltas*, never as one absolute
total. The encoders are what this package owns; everything downstream is imported, so a legitimate
change to a shared component must not fail a test here. The absolute number belongs in the design
record, checked against ``sum(p.numel() ...)`` rather than against a literal.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from teb_vae.lag_attn.nets.heads import PriorHead
from teb_vae.lag_attn.nets.delays import ChannelDelay, ChannelGate
from teb_vae.lag_attn_transformer_rws.nets.blocks import (
    CausalSelfAttention,
    GatedCausalConvBlock,
)
from teb_vae.lag_attn_transformer_rws.nets.encoders import CausalConvTransformerEncoder
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
from teb_vae.lag_attn_transformer_rws.tests.conftest import SHIPPED_KWARGS

#: The shipped encoder subtotals, restated as the arithmetic that produces them.
#:
#: A convolution block costs $3d^2 + 3d + dk$: $2d^2$ for the gated input projection, $d^2$ for the
#: output projection, $d$ apiece for two RMSNorms and the LayerScale, and $dk$ for the depthwise
#: filter bank. At $d = 128$ that is $50{,}176$ at $k = 5$ and $50{,}688$ at $k = 9$.
#:
#: An attention block costs $4d^2 + 3d\,d_{\mathrm{ff}} + 4d$: four bias-free projections, the
#: SwiGLU triple, and $d$ apiece for two norms and two LayerScale vectors -- $164{,}352$ at
#: $d = 128$, $d_{\mathrm{ff}} = 256$.
#:
#: Each encoder then adds $d$ for its final RMSNorm. Target: $50{,}176 + 50{,}688 +
#: 4 \cdot 164{,}352 + 128 = 758{,}400$. Source: the same stem, three attention blocks, one norm =
#: $594{,}048$.
_CONV_BLOCK_5, _CONV_BLOCK_9 = 50_176, 50_688
_ATTENTION_BLOCK = 164_352
_FINAL_NORM = 128
_TARGET_ENCODER = _CONV_BLOCK_5 + _CONV_BLOCK_9 + 4 * _ATTENTION_BLOCK + _FINAL_NORM
_SOURCE_ENCODER = _CONV_BLOCK_5 + _CONV_BLOCK_9 + 3 * _ATTENTION_BLOCK + _FINAL_NORM

#: The stem's cost across both streams, which is exactly what the stem-free architecture arm
#: removes: $2 \cdot (50{,}176 + 50{,}688)$.
_STEM_BOTH_STREAMS = 2 * (_CONV_BLOCK_5 + _CONV_BLOCK_9)

#: Recurrent and time-pooling module families that must not appear on a history path. Each would
#: make $H_t$ a function of the whole sequence, which is invisible in a loss curve and corrupts
#: exactly the quantity the model exists to measure.
_BANNED_ON_HISTORY_PATH = (
    nn.LSTM,
    nn.GRU,
    nn.RNN,
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveMaxPool1d,
    nn.AvgPool1d,
    nn.MaxPool1d,
)


def _model(kwargs, **overrides) -> SeqVaeLagAttnTrfRws:
    torch.manual_seed(0)
    return SeqVaeLagAttnTrfRws(**dict(kwargs, **overrides))


def _n_parameters(**overrides) -> int:
    """Total parameter count of a shipped-geometry model under one delta."""
    return sum(p.numel() for p in _model(SHIPPED_KWARGS, **overrides).parameters())


@pytest.fixture(scope="module")
def shipped_model() -> SeqVaeLagAttnTrfRws:
    """One production-geometry model, built once for every construction-time check here."""
    return _model(SHIPPED_KWARGS)


# ---------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------
def test_the_model_constructs_at_the_tiny_geometry(tiny_kwargs):
    model = _model(tiny_kwargs)
    assert model.geometry.raw_len == 256
    assert model.geometry.t_valid == 12


def test_the_model_constructs_at_the_production_geometry(shipped_model):
    assert shipped_model.geometry.raw_len == 4800
    assert shipped_model.geometry.t_valid == 270


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
    """``nn.Linear(0, d)`` is legal and returns its bias, so a zero width would build a model that
    trains to completion having never read that stream."""
    with pytest.raises(ValueError, match="c_y"):
        _model(tiny_kwargs, c_y=0)
    with pytest.raises(ValueError, match="c_u"):
        _model(tiny_kwargs, c_u=0)


def test_a_degenerate_raw_geometry_is_rejected(tiny_kwargs):
    with pytest.raises(ValueError, match="degenerate"):
        _model(tiny_kwargs, horizon=16)  # horizon == T leaves no valid anchor


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"encoder_conv_kernels": (5, 9, 3)}, "equal length"),
        ({"target_attention_blocks": 0}, "at least 1"),
        ({"source_attention_window": 0}, "at least 1 step"),
        ({"encoder_num_heads": 5}, "divisible"),
        ({"encoder_num_heads": 32, "d_model": 32}, "even"),
    ],
    ids=["stem-schedules", "no-attention", "zero-window", "indivisible-heads", "odd-head-width"],
)
def test_an_inconsistent_encoder_schema_is_refused(tiny_kwargs, overrides, match):
    """Each of these builds a model that is *wrong* rather than one that fails: a mismatched stem
    schedule pairs the wrong kernel with the wrong dilation, an attention-free encoder is the
    convolution stack this architecture replaces, and an odd head width silently disables half of
    every rotary rotation."""
    with pytest.raises(ValueError, match=match):
        _model(tiny_kwargs, **overrides)


# ---------------------------------------------------------------------------------------
# What must not exist
# ---------------------------------------------------------------------------------------
def test_no_second_decoder_and_no_decoder_state_head_exist(tiny_kwargs):
    model = _model(tiny_kwargs)

    assert not hasattr(model, "residual_decoder")
    assert not hasattr(model, "baseline_decoder")
    # PriorHead is the class that carries a decoder_state head; its absence is the absence of the
    # bypass at the module level.
    assert not any(isinstance(module, PriorHead) for module in model.modules())
    assert not hasattr(model.prior_head, "decoder_state_head")


def test_no_recurrence_anywhere_in_the_model(tiny_kwargs):
    """The recurrent bottleneck is the thing this architecture removed; a stray one anywhere would
    both serialise training and reintroduce the state it exists without."""
    model = _model(tiny_kwargs)
    offenders = [
        name for name, module in model.named_modules()
        if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN))
    ]

    assert not offenders, f"recurrent modules found: {offenders}"


def test_no_time_pooling_normaliser_on_either_history_path(tiny_kwargs):
    """Scoped to the history path -- both gates, both adapters, both encoders -- because that is
    where a statistic pooled over time would make $H_t$ read its own future."""
    model = _model(tiny_kwargs)
    history = {
        "target_adapter": model.target_adapter,
        "source_adapter": model.source_adapter,
        "target_encoder": model.target_encoder,
        "source_encoder": model.source_encoder,
    }
    offenders = [
        f"{stem}.{name}"
        for stem, subtree in history.items()
        for name, module in subtree.named_modules()
        if isinstance(module, _BANNED_ON_HISTORY_PATH)
    ]

    assert not offenders, f"time-pooling or recurrent modules on a history path: {offenders}"


def test_the_only_group_norms_left_are_the_horizon_cores(tiny_kwargs):
    """A deliberate exception, pinned so it stays deliberate: the horizon core's normalisers pool
    over the *forecast* axis of a single anchor, not across input time, so they cannot leak the
    target's future into a history state. Enumerating them means a new one anywhere else fails
    here rather than passing the scoped check above."""
    model = _model(tiny_kwargs)
    group_norms = [
        name for name, module in model.named_modules() if isinstance(module, nn.GroupNorm)
    ]

    assert group_norms, "no GroupNorm at all; this pin no longer describes the model"
    assert all(name.startswith("horizon_core.") for name in group_norms), group_norms


# ---------------------------------------------------------------------------------------
# The dropout contract, built at a nonzero dropout so it is not vacuous
# ---------------------------------------------------------------------------------------
def test_every_structurally_zero_dropout_is_zero_while_the_model_is_built_at_a_tenth(tiny_kwargs):
    """Three sites, three different reasons.

    The lag-attention probabilities: dropout is applied to the weights *before* they are returned,
    and the per-lag KL attribution is exact only if the returned weights are the ones the posterior
    consumed. The encoder self-attention probabilities: unnecessary at this depth and a needless
    reproducibility hazard. The decoder subtree: one module invoked twice draws two independent
    masks, so base and full would differ at initialisation even with $z^p = z^q$.
    """
    model = _model(tiny_kwargs, dropout=0.1)

    assert model.lag_attn.attn_dropout.p == 0.0
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout) and name.startswith(("decoder.", "horizon_core.")):
            assert module.p == 0.0, f"{name} has dropout {module.p}"
    # And the encoders *did* receive the configured value, so the assertions above are about zeros
    # that were chosen rather than about a model built at zero dropout throughout.
    encoder_dropouts = {
        module.p for name, module in model.target_encoder.named_modules()
        if isinstance(module, nn.Dropout)
    }
    assert encoder_dropouts == {0.1}, encoder_dropouts


def test_the_encoder_attention_probabilities_carry_no_dropout(tiny_kwargs):
    """Structural rather than configured: the attention call passes ``dropout_p=0.0`` and the
    module's own ``dropout`` is the *output* dropout of the equations.

    Measured rather than read off the source: two train-mode passes with the seed fixed *between*
    them must give identical attention outputs when the output dropout is disabled, which they can
    only do if nothing stochastic happens inside the attention itself.
    """
    model = _model(tiny_kwargs, dropout=0.1)
    attention = next(
        module for module in model.modules() if isinstance(module, CausalSelfAttention)
    )
    assert attention.dropout.p == 0.1  # the output dropout is the configured one

    attention.train()
    attention.dropout.p = 0.0
    x = torch.randn(2, int(tiny_kwargs["sequence_length"]), int(tiny_kwargs["d_model"]))
    torch.manual_seed(1)
    first = attention(x)
    torch.manual_seed(2)
    second = attention(x)

    assert torch.equal(first, second), "the attention is stochastic in train mode"


def test_the_lag_attention_output_projection_is_frozen(tiny_kwargs):
    """$W_o$ feeds nothing under the head-structured posterior; freezing it drops it from DDP's
    expectation set instead of leaving a parameter that never receives a gradient."""
    attention = _model(tiny_kwargs).lag_attn

    assert attention.W_o.weight.requires_grad is False
    assert attention.W_o.bias.requires_grad is False


def test_the_posterior_is_head_structured(tiny_kwargs):
    assert _model(tiny_kwargs).posterior_head.head_structured is True


# ---------------------------------------------------------------------------------------
# The encoder heads are not the latent groups
# ---------------------------------------------------------------------------------------
def test_the_encoder_head_count_does_not_touch_the_latent_grouping(tiny_kwargs, inputs):
    """Two independent head counts that merely coincide at the shipped configuration. A depth or
    width arm that changed one must not move the other, or the per-head KL decomposition would
    quietly stop being aligned with the lag-attention heads."""
    model = _model(tiny_kwargs, encoder_num_heads=2).eval()
    reference = _model(tiny_kwargs).eval()

    assert model.num_heads == reference.num_heads
    assert model.posterior_head.head_structured is reference.posterior_head.head_structured
    assert model.lag_attn.num_heads == reference.lag_attn.num_heads
    for encoder in (model.target_encoder, model.source_encoder):
        assert encoder.num_heads == 2

    with torch.no_grad():
        out = model(*inputs)
    assert out["kld_per_t_per_head"].shape[-1] == model.num_heads
    assert out["mu_prior"].shape[-1] % model.num_heads == 0


# ---------------------------------------------------------------------------------------
# The parameter budget
# ---------------------------------------------------------------------------------------
def test_the_shipped_encoder_subtotals(shipped_model):
    target = sum(p.numel() for p in shipped_model.target_encoder.parameters())
    source = sum(p.numel() for p in shipped_model.source_encoder.parameters())

    assert target == _TARGET_ENCODER == 758_400
    assert source == _SOURCE_ENCODER == 594_048


def test_a_block_costs_what_the_arithmetic_says(shipped_model):
    """The subtotals above are sums of these, so pinning the parts as well as the total says
    *where* a change landed rather than only that one happened."""
    d_model, d_ff = 128, 256
    conv_blocks = [
        module for module in shipped_model.target_encoder.modules()
        if isinstance(module, GatedCausalConvBlock)
    ]
    attention_blocks = list(shipped_model.target_encoder.attention_blocks)

    for block, kernel, expected in zip(conv_blocks, (5, 9), (_CONV_BLOCK_5, _CONV_BLOCK_9)):
        assert block.conv.kernel_size == kernel
        assert sum(p.numel() for p in block.parameters()) == expected
        assert expected == 3 * d_model**2 + 3 * d_model + d_model * kernel
    for block in attention_blocks:
        assert sum(p.numel() for p in block.parameters()) == _ATTENTION_BLOCK
    assert _ATTENTION_BLOCK == 4 * d_model**2 + 3 * d_model * d_ff + 4 * d_model


def test_removing_one_target_attention_block_costs_exactly_one_block():
    assert _n_parameters() - _n_parameters(target_attention_blocks=3) == _ATTENTION_BLOCK


def test_removing_the_stem_from_both_streams_costs_exactly_the_stem():
    stemless = _n_parameters(encoder_conv_kernels=(), encoder_conv_dilations=())

    assert _n_parameters() - stemless == _STEM_BOTH_STREAMS == 201_728


def test_a_stemless_encoder_is_a_working_module(tiny_kwargs, inputs):
    """Zero convolution blocks is legal, because a stem-free architecture arm needs it."""
    model = _model(tiny_kwargs, encoder_conv_kernels=(), encoder_conv_dilations=()).eval()

    assert len(model.target_encoder.conv_blocks) == 0
    assert model.n_depthwise_init == 0
    with torch.no_grad():
        out = model(*inputs)
    assert out["target_state"].shape[-1] == model.d_model


def test_the_encoders_are_the_configured_shape(shipped_model, shipped_kwargs):
    """The seven encoder keys map onto the two encoders in exactly one way: the target reads the
    full causal prefix, the source a bounded window. That asymmetry is the architecture."""
    assert isinstance(shipped_model.target_encoder, CausalConvTransformerEncoder)
    assert shipped_model.target_encoder.attention_window is None
    assert shipped_model.target_encoder.receptive_field is None
    assert (
        shipped_model.source_encoder.attention_window
        == shipped_kwargs["source_attention_window"]
    )
    assert shipped_model.source_encoder.receptive_field == 66


# ---------------------------------------------------------------------------------------
# The causal input guard
# ---------------------------------------------------------------------------------------
def test_an_unguarded_model_has_no_gather_and_no_delay(tiny_kwargs):
    """Not an identity guard -- nothing at all, so the unguarded run is structurally the model
    that existed before the guard did."""
    model = _model(tiny_kwargs)

    assert model.target_gate is None and model.source_gate is None
    assert not any(isinstance(m, (ChannelGate, ChannelDelay)) for m in model.modules())
    assert model.source_delay_steps == 0


def test_an_unguarded_forward_is_bitwise_equal_to_an_identity_guard(tiny_kwargs, inputs):
    """The other direction: the gather-and-delay path, at the identity, must change nothing.

    It also pins the availability terms: at zero delays neither is constructed, so the guarded
    model is the plain one rather than the plain one plus a constant.
    """
    plain = _model(tiny_kwargs).eval()
    identity = _model(
        tiny_kwargs,
        target_keep_index=tuple(range(109)),
        target_delays=(0,) * 109,
        source_keep_index=tuple(range(58)),
        source_delays=(0,) * 58,
    ).eval()

    assert identity.target_adapter.mask_proj is None

    torch.manual_seed(3)
    expected = plain(*inputs)
    torch.manual_seed(3)
    got = identity(*inputs)

    assert all(torch.equal(expected[key], got[key]) for key in expected)


def test_the_adapters_are_built_for_the_surviving_widths(tiny_kwargs):
    """The model still declares the full ``c_y`` / ``c_u`` -- the data boundary checks the batch
    against those -- while the adapters see only the survivors."""
    model = _model(
        tiny_kwargs,
        target_keep_index=(0, 5, 9),
        target_delays=(1, 2, 3),
        source_keep_index=(2, 7),
        source_delays=(0, 4),
    )

    assert (model.c_y, model.c_u) == (109, 58)
    assert model.target_adapter.linear.in_features == 3
    assert model.source_adapter.linear.in_features == 2


def test_a_gated_forward_reads_only_the_surviving_channels(tiny_kwargs, inputs):
    """Perturbing a pruned channel must change nothing: a channel that fails the reach budget has
    to be genuinely gone, not merely down-weighted."""
    model = _model(
        tiny_kwargs,
        target_keep_index=(0, 5, 9),
        target_delays=(0, 0, 0),
        source_keep_index=(2, 7),
        source_delays=(0, 0),
    ).eval()
    y_st, y_ph, u_stream = inputs

    torch.manual_seed(3)
    before = model(y_st, y_ph, u_stream)["mu_prior"]
    perturbed = y_st.clone()
    perturbed[..., 1] += 100.0  # channel 1 is not in keep
    torch.manual_seed(3)
    after = model(perturbed, y_ph, u_stream)["mu_prior"]

    assert torch.equal(before, after)


def test_the_gate_and_availability_buffers_stay_out_of_the_state_dict(tiny_kwargs):
    """Their length is the surviving-channel count, so a persistent copy would make a checkpoint
    trained at one reach budget fail to load at another as "keys did not align"."""
    model = _model(
        tiny_kwargs,
        target_keep_index=(0, 5, 9),
        target_delays=(1, 2, 3),
        source_keep_index=(2, 7),
        source_delays=(1, 4),
    )
    keys = list(model.state_dict())

    for fragment in ("keep_index", "delay_steps", "availability", "start_indicator"):
        assert not [name for name in keys if fragment in name], fragment
    # The learned availability parameters do belong in it -- they are weights, not geometry.
    assert any("mask_proj" in name for name in keys)
    assert any("start_embed" in name for name in keys)


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
