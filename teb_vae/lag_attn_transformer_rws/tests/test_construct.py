r"""Construction invariants: what is refused, what may not sit on a history path, and the seams.

The constructor's guarantees are structural -- no recurrence and no time-pooling normaliser on
either history path, the dropout sites pinned at zero, a latent grouping independent of the
encoder heads -- and each is asserted on the **assembled** model, because several of them hold on
the parts in isolation and fail silently in composition. Beside them: every inconsistent geometry
or encoder schema is refused at construction, the decoder-width hook a feature-domain sibling
overrides moves the head and nothing else, a stem-free encoder still runs, and the causal input
guard is bitwise inert at the identity, genuinely drops pruned channels, and keeps its
surviving-channel buffers out of the state dict.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

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


def _model(kwargs, cls=SeqVaeLagAttnTrfRws, **overrides) -> SeqVaeLagAttnTrfRws:
    torch.manual_seed(0)
    return cls(**dict(kwargs, **overrides))


# ---------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------
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
# The decoder's width hook, and the stem-free arm
# ---------------------------------------------------------------------------------------
def test_an_overridden_width_hook_moves_the_head_and_nothing_else(tiny_kwargs):
    """The seam, exercised the way a feature-domain sibling uses it.

    The hook is called at the decoder's construction site -- after both gates and before the
    generic initialisation, the depthwise repair and the two calibration passes -- so an override
    changes the emitted width and nothing about the init order. Both halves are asserted: the
    decoder is the new width, and the depthwise count and the log-variance calibration are the ones
    a $16$-wide model gets, which they could not be if the decoder had moved past either pass.
    """
    class _WiderDecoder(SeqVaeLagAttnTrfRws):
        def _default_decoder_out_channels(self) -> int:
            return 78

    # Built with the calibration on, because the calibration pass is what dates the decoder's
    # construction relative to the init block.
    calibrated = dict(tiny_kwargs)
    calibrated["head_init_calibration"] = True
    reference = _model(calibrated)
    widened = _model(calibrated, cls=_WiderDecoder)

    assert widened.decoder_out_channels == widened.decoder.out_channels == 78
    assert widened.raw_per_step == reference.raw_per_step == 16, "the geometry input is untouched"
    assert widened.n_depthwise_init == reference.n_depthwise_init
    # The calibration ran on the wide head, so it was built before the calibration pass.
    reference_bias = reference.decoder.logvar_head.bias
    assert widened.decoder.logvar_head.bias.numel() == 78
    assert torch.equal(
        widened.decoder.logvar_head.bias,
        torch.full_like(widened.decoder.logvar_head.bias, float(reference_bias[0])),
    )
    # And the width is the only structural difference: same parameter names, same shapes but the
    # decoder's two output heads.
    before = dict(reference.named_parameters())
    after = dict(widened.named_parameters())
    assert set(before) == set(after)
    assert sorted(name for name in before if before[name].shape != after[name].shape) == [
        "decoder.logvar_head.bias",
        "decoder.logvar_head.weight",
        "decoder.mean_head.bias",
        "decoder.mean_head.weight",
    ]


def test_a_stemless_encoder_is_a_working_module(tiny_kwargs, inputs):
    """Zero convolution blocks is legal, because a stem-free architecture arm needs it."""
    model = _model(tiny_kwargs, encoder_conv_kernels=(), encoder_conv_dilations=()).eval()

    assert len(model.target_encoder.conv_blocks) == 0
    assert model.n_depthwise_init == 0
    with torch.no_grad():
        out = model(*inputs)
    assert out["target_state"].shape[-1] == model.d_model


# ---------------------------------------------------------------------------------------
# The causal input guard
# ---------------------------------------------------------------------------------------
def test_an_unguarded_forward_is_bitwise_equal_to_an_identity_guard(tiny_kwargs, inputs):
    """The gather-and-delay path, at the identity, must change nothing.

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
