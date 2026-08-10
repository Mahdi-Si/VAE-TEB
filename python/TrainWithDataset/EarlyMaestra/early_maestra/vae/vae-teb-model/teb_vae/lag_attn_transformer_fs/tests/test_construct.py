r"""Construction: what the composition changes, what it inherits, and what it still refuses.

A composition is only worth having if it is provably one, so the tests split in two.

The first half pins the *difference*, and there is exactly one: each horizon token emits one value
per surviving target channel instead of $R = 16$ raw samples. It shows up in three places -- the
decoder's two output heads, the parameter total, and the last axis of the four forecast tensors --
and nowhere else. The parameter counts are written out because they *are* the claim: the heads are
$\mathrm{Linear}(128, X)$, so widening $X$ costs $2 \cdot 129 = 258$ parameters per channel and
nothing anywhere else in the model may move.

The second half pins the *sameness*: the class body is empty, both parents' members resolve where
the design says, the constructor signature is the architecture parent's down to the keyword order,
and every construction-time refusal fires with the parent's message compared as a string.

**Why there is no forward-contract module in this package.** The twenty-key return set, the absent
``decoder_state`` and ``delta_mu_src``, the head-structured posterior, the lag map's conservation
identity and every latent shape are consequences of ``vars(SeqVaeLagAttnTrfFs)`` being empty: they
are the parent's own code objects, pinned by the parent's own suite. The only thing this class
changes about the forward is the forecast tensors' last axis, and that is asserted here.
"""
from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from teb_vae.lag_attn.nets.delays import ChannelDelay, ChannelGate
from teb_vae.lag_attn.nets.heads import PriorHead
from teb_vae.lag_attn_fs.nets.feature_target import FeatureForecastTarget
from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.tests.conftest import (
    shipped_gated_kwargs as feature_shipped_gated_kwargs,
)
from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs
from teb_vae.lag_attn_transformer_fs.tests.conftest import (
    BATCH,
    SHIPPED_KWARGS,
    TINY_KEEP_INDEX,
    shipped_gated_kwargs,
)
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

#: Parameters the decoder's two output heads cost per emitted channel:
#: $\mathrm{Linear}(d_{\mathrm{hidden}} = 256, X)$ twice, weights and biases.
_PARAMS_PER_CHANNEL = 2 * (256 + 1)

#: The architecture parent's measured totals at the shipped widths, with and without the shipped
#: $120$ s reach budget. They differ: a guarded run builds availability input adapters, which the
#: unguarded one does not have.
_TRF_SHIPPED_UNGATED = 4_996_844
_TRF_SHIPPED_GATED = 5_003_116

#: This model's measured totals at the same two configurations.
_SHIPPED_GATED = 5_034_984
_SHIPPED_UNGATED = 5_044_646

#: Surviving target channels at the shipped budget, and the declared width.
_KEPT_CHANNELS = 78
_ALL_CHANNELS = 109


def _model(kwargs, cls=SeqVaeLagAttnTrfFs, **overrides):
    torch.manual_seed(0)
    return cls(**dict(kwargs, **overrides))


def _n_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


# ---------------------------------------------------------------------------------------
# The geometry, and the width that follows the budget
# ---------------------------------------------------------------------------------------
def test_the_model_constructs_at_the_tiny_geometry(tiny_kwargs):
    model = _model(tiny_kwargs)

    assert model.geometry.raw_len == 256
    assert model.geometry.t_valid == 12


def test_the_model_constructs_at_the_production_geometry(shipped_gated):
    model = _model(shipped_gated)

    assert model.geometry.raw_len == 4800
    assert model.geometry.t_valid == 270


def test_the_decoder_width_is_the_surviving_channel_count(shipped_gated):
    """78 at the shipped budget. Not a configuration key -- and unlike the conv-LSTM parent, not
    even a constructor keyword -- so the width follows the gate and a run cannot decode a width its
    target does not have."""
    model = _model(shipped_gated)

    assert model.target_gate is not None
    assert model.target_gate.out_channels == _KEPT_CHANNELS
    assert model.decoder_out_channels == _KEPT_CHANNELS
    assert model.decoder.out_channels == _KEPT_CHANNELS
    assert model.decoder.mean_head.out_features == _KEPT_CHANNELS
    assert model.decoder.logvar_head.out_features == _KEPT_CHANNELS


def test_without_a_budget_the_decoder_width_is_the_declared_width(shipped_kwargs):
    """109 at ``causal_reach_budget_s: null``. The unguarded arm is a configuration rather than an
    unhandled case, and its block cardinality is $H \\cdot c_y$."""
    model = _model(shipped_kwargs)

    assert model.target_gate is None
    assert model.decoder_out_channels == _ALL_CHANNELS == model.c_y
    assert model.decoder.out_channels == _ALL_CHANNELS


def test_the_width_follows_a_gate_of_any_size(tiny_gated):
    """Three survivors, three output channels: the rule is the gate's count, not a constant."""
    model = _model(tiny_gated)

    assert model.decoder_out_channels == len(TINY_KEEP_INDEX) == 3
    assert model.raw_per_step == 16, "the raw grid is geometry, not the decoder width"


# ---------------------------------------------------------------------------------------
# The forecast tensors: the entire delta this class introduces to the forward
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "budget, channels",
    [(None, _ALL_CHANNELS), ("shipped", _KEPT_CHANNELS)],
    ids=["ungated-109", "guarded-78"],
)
def test_the_four_forecast_tensors_carry_the_target_width(budget, channels):
    r"""$(B, 270, 30, C)$ at the production geometry, and $C$ is the only thing that moved.

    Run at the shipped $T = 300$ rather than at the tiny fixture because the shape is the claim: the
    anchor axis is $T_{\mathrm{valid}} = 270$, the horizon axis $H = 30$, and the last axis is the
    surviving-channel count the reach budget resolved. The raw parent emits $R = 16$ there.
    """
    kwargs = shipped_gated_kwargs(None if budget is None else 120.0)
    model = _model(kwargs, dropout=0.0).eval()
    generator = torch.Generator().manual_seed(0)
    length = int(kwargs["sequence_length"])
    streams = (
        torch.randn(BATCH, length, 43, generator=generator),
        torch.randn(BATCH, length, 66, generator=generator),
        torch.randn(BATCH, length, 58, generator=generator),
    )

    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*streams)

    expected = (BATCH, 270, 30, channels)
    for key in ("mu_base", "logvar_base", "mu_full", "logvar_full"):
        assert tuple(out[key].shape) == expected, key
    # The latent side is untouched, which is what makes the line above the *only* delta.
    assert tuple(out["mu_prior"].shape) == (BATCH, length, int(kwargs["d_z"]))
    assert out["target_state"].shape[-1] == model.d_model != model.decoder_out_channels


# ---------------------------------------------------------------------------------------
# What it costs
# ---------------------------------------------------------------------------------------
def test_the_decoder_delta_against_the_architecture_parent_is_the_head_cost(shipped_gated):
    """The one structural difference, priced. Widening the decoder from $R = 16$ to $78$ costs
    $514 \\times 62 = 31{,}868$ parameters and nothing else in the model may move."""
    raw = _model(shipped_gated, cls=SeqVaeLagAttnTrfRws)
    feature = _model(shipped_gated)

    raw_count = _n_parameters(raw)
    feature_count = _n_parameters(feature)

    assert raw_count == _TRF_SHIPPED_GATED
    assert feature_count - raw_count == _PARAMS_PER_CHANNEL * (_KEPT_CHANNELS - 16) == 31_868
    assert feature_count == _SHIPPED_GATED == 5_034_984


def test_the_ungated_arm_costs_the_full_declared_width(shipped_kwargs):
    """$514 \\times 93 = 47{,}802$ at $c_y = 109$, against the unguarded parent -- a guarded run
    adds the availability adapters' $6{,}272$ parameters on top of both."""
    raw = _model(shipped_kwargs, cls=SeqVaeLagAttnTrfRws)
    feature = _model(shipped_kwargs)

    assert _n_parameters(raw) == _TRF_SHIPPED_UNGATED
    assert _n_parameters(feature) - _n_parameters(raw) == _PARAMS_PER_CHANNEL * (
        _ALL_CHANNELS - 16
    ) == 47_802
    assert _n_parameters(feature) == _SHIPPED_UNGATED == 5_044_646


def test_the_encoder_swap_is_the_only_difference_against_the_feature_sibling(shipped_gated):
    """The comparison this package exists to make, priced: at a fixed target the two feature models
    differ by $91{,}342$ parameters, all of it encoders, which is a $1.8\\%$ reduction.

    That percentage is small **now** and used to be $38.4\\%$, and the change is the point rather
    than an erosion of it: the capacity revision raised the conv-Transformer encoders (six target
    blocks at $d_{\\mathrm{ff}} = 512$ against four at $256$) and left the conv-LSTM ones alone, so
    the two rows of the grid are now near parity in budget. The comparison is *better* posed for it
    -- an encoder axis read at matched capacity attributes a forecast difference to the encoder's
    structure rather than to its size -- but nothing here may quote the old headline any more.

    Each model is built from its **own** shipped keyword set, because that is the only way to build
    both -- the two constructors' schemas differ by six keywords -- and because it is what the two
    shipped configs will do. What makes it a comparison at a fixed target is asserted rather than
    assumed: the same reach budget resolves the same keep-index, so both decoders emit the same
    block, and the nats are comparable. A parameter total alone would also be satisfied by two
    models that differed in the head and compensated elsewhere.
    """
    conv_lstm = _model(feature_shipped_gated_kwargs(), cls=SeqVaeLagAttnFs)
    transformer = _model(shipped_gated)

    for name in ("mean_head", "logvar_head"):
        assert (
            getattr(conv_lstm.decoder, name).weight.shape
            == getattr(transformer.decoder, name).weight.shape
        ), name
    assert conv_lstm.decoder_out_channels == transformer.decoder_out_channels == _KEPT_CHANNELS
    assert conv_lstm.horizon == transformer.horizon == 30

    reduction = 1.0 - _n_parameters(transformer) / _n_parameters(conv_lstm)
    assert _n_parameters(conv_lstm) == 5_126_326
    assert _n_parameters(conv_lstm) - _n_parameters(transformer) == 91_342
    assert float(reduction) == pytest.approx(0.018, abs=5e-4)


#: The six keywords the conv-LSTM feature sibling's constructor takes and this one does not. Five
#: describe the encoder being replaced; the sixth is the decoder width, which is a hook here and
#: named in no YAML in the repository.
_CONV_LSTM_ONLY_KEYWORDS = (
    "lstm_layers",
    "encoder_extra_dilations",
    "encoder_extra_kernel",
    "conv_norm_groups",
    "causal_norm",
    "decoder_out_channels",
)


# ---------------------------------------------------------------------------------------
# What the class is
# ---------------------------------------------------------------------------------------
def test_the_class_itself_defines_nothing_at_all():
    """Set equality over ``vars``, and the set is **empty**.

    Not a line count, which passes a class that overrode ``forward`` in 140 lines. This is the
    cheapest guarantee the suite has: with nothing defined here, the twenty forward keys, the
    posterior's structure, the lag map and the objective's metric set cannot have moved, because
    they are the parents' own code objects.
    """
    own = {
        name
        for name, value in vars(SeqVaeLagAttnTrfFs).items()
        if callable(value) and not isinstance(value, type)
    }

    assert own == set()
    assert not [name for name in vars(SeqVaeLagAttnTrfFs) if not name.startswith("__")]
    assert "forward" not in vars(SeqVaeLagAttnTrfFs)
    assert "__init__" not in vars(SeqVaeLagAttnTrfFs)


def test_the_mro_puts_the_target_domain_ahead_of_the_architecture():
    """Asserted as a list of names, because the *order* is what makes the width hook resolve to the
    feature one. Reversed, the decoder would be built at ``raw_per_step`` and a $78$-wide feature
    block scored against a $16$-wide forecast -- and since the objective takes the block width as an
    argument, nothing would raise."""
    assert [cls.__name__ for cls in SeqVaeLagAttnTrfFs.__mro__] == [
        "SeqVaeLagAttnTrfFs",
        "FeatureForecastTarget",
        "SeqVaeLagAttnTrfRws",
        "Module",
        "object",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "_default_decoder_out_channels",
        "_build_forecast_target",
        "_resolved_forecast_gaps",
        "compute_loss",
        "TARGET_BLOCK_SPLIT",
    ],
)
def test_every_target_domain_member_is_the_mixins_own_object(name):
    """Identity, not equality: the five members must be the *same* objects the feature sibling
    reaches, or the two models would be forecasting two targets that merely agree today."""
    assert getattr(SeqVaeLagAttnTrfFs, name) is getattr(FeatureForecastTarget, name)
    assert getattr(SeqVaeLagAttnFs, name) is getattr(FeatureForecastTarget, name)


@pytest.mark.parametrize(
    "name",
    ["forward", "_reparameterize_shared", "kld_tensor", "_build_channel_gate", "_build_adapter"],
)
def test_every_architecture_member_is_the_encoder_parents_own_object(name):
    """The other direction, and the one that would catch a mixin that had quietly grown a
    ``forward``: everything the architecture owns resolves to the conv-Transformer parent."""
    assert getattr(SeqVaeLagAttnTrfFs, name) is getattr(SeqVaeLagAttnTrfRws, name)


def test_the_constructor_signature_is_the_architecture_parents():
    """The trainer builds its kwargs from an ``inspect.signature`` sweep of ``MODEL_CLS.__init__``.
    A narrowed signature would forward no configuration at all and silently build an all-defaults
    model -- no error, no shape mismatch, a run at the wrong widths."""
    signature = inspect.signature(SeqVaeLagAttnTrfFs.__init__)

    assert signature == inspect.signature(SeqVaeLagAttnTrfRws.__init__)
    # And it is *not* the conv-LSTM sibling's, which carries six keywords this one does not --
    # including ``decoder_out_channels``, which is a hook here so no second field can disagree with
    # the gate.
    assert signature != inspect.signature(SeqVaeLagAttnFs.__init__)
    conv_lstm_parameters = inspect.signature(SeqVaeLagAttnFs.__init__).parameters
    for name in _CONV_LSTM_ONLY_KEYWORDS:
        assert name not in signature.parameters, name
        assert name in conv_lstm_parameters, f"{name} is no longer the sibling's either"


# ---------------------------------------------------------------------------------------
# The inherited refusals, with the inherited messages
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(d_z=9), r"d_z=9.*num_heads=4"),
        (dict(d_head=16), "d_model"),
        (dict(max_lag=-1), "max_lag"),
        (dict(c_y=0), "c_y"),
        (dict(c_u=0), "c_u"),
        (dict(horizon=16), "degenerate"),
        (dict(encoder_conv_kernels=(5, 9, 3)), "equal length"),
        (dict(target_attention_blocks=0), "at least 1"),
        (dict(source_attention_window=0), "at least 1 step"),
        (dict(encoder_num_heads=5), "divisible"),
    ],
    ids=["indivisible-latent", "head-geometry", "negative-lag", "zero-c_y", "zero-c_u",
         "degenerate-geometry", "stem-schedules", "no-attention", "zero-window",
         "indivisible-heads"],
)
def test_the_construction_refusals_are_the_parents(tiny_kwargs, overrides, match):
    with pytest.raises(ValueError, match=match):
        _model(tiny_kwargs, **overrides)


@pytest.mark.parametrize(
    "overrides",
    [dict(d_z=9), dict(d_head=16), dict(encoder_num_heads=5), dict(target_attention_blocks=0)],
    ids=["indivisible-latent", "head-geometry", "indivisible-heads", "no-attention"],
)
def test_the_refusal_messages_are_identical_strings(tiny_kwargs, overrides):
    """Not merely "both raise": a composition that re-derived its own guards would drift in wording
    first and in behaviour later."""
    messages = []
    for cls in (SeqVaeLagAttnTrfRws, SeqVaeLagAttnTrfFs):
        with pytest.raises(ValueError) as excinfo:
            _model(tiny_kwargs, cls=cls, **overrides)
        messages.append(str(excinfo.value))

    assert messages[0] == messages[1]


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
def test_a_malformed_target_gate_is_refused_before_a_decoder_is_sized(
    tiny_kwargs, keep, delays, match
):
    """Sharper here than in the architecture parent: the gate now decides the decoder's width, so a
    gate that built anyway would put the decoder on a width the target does not have."""
    with pytest.raises(ValueError, match=match):
        _model(tiny_kwargs, target_keep_index=keep, target_delays=delays)


# ---------------------------------------------------------------------------------------
# The inherited structure
# ---------------------------------------------------------------------------------------
def test_no_decoder_state_head_and_no_second_decoder_exist(tiny_kwargs):
    model = _model(tiny_kwargs)

    assert not hasattr(model, "residual_decoder")
    assert not hasattr(model, "baseline_decoder")
    assert not any(isinstance(module, PriorHead) for module in model.modules())
    assert not hasattr(model.prior_head, "decoder_state_head")


def test_the_posterior_is_head_structured(tiny_kwargs):
    assert _model(tiny_kwargs).posterior_head.head_structured is True


def test_no_recurrence_and_no_time_pooling_normaliser_on_a_history_path(tiny_kwargs):
    """The architectural claim this cell inherits and the conv-LSTM cell cannot make: a statistic
    pooled over time on a history path would make $H_t$ read its own future, and there is none --
    which is why step-wise causality here needs no ``causal_norm`` flag to qualify it."""
    model = _model(tiny_kwargs)
    history = {
        "target_adapter": model.target_adapter,
        "source_adapter": model.source_adapter,
        "target_encoder": model.target_encoder,
        "source_encoder": model.source_encoder,
    }
    banned = (nn.LSTM, nn.GRU, nn.RNN, nn.GroupNorm, nn.BatchNorm1d, nn.AvgPool1d,
              nn.AdaptiveAvgPool1d, nn.MaxPool1d, nn.AdaptiveMaxPool1d)
    offenders = [
        f"{stem}.{name}"
        for stem, subtree in history.items()
        for name, module in subtree.named_modules()
        if isinstance(module, banned)
    ]

    assert not offenders, f"time-pooling or recurrent modules on a history path: {offenders}"
    assert not hasattr(model, "causal_norm"), (
        "a causal_norm attribute would mean there is a normaliser to causalise after all"
    )


def test_the_attention_output_projection_is_frozen(tiny_kwargs):
    attention = _model(tiny_kwargs).lag_attn

    assert attention.W_o.weight.requires_grad is False
    assert attention.W_o.bias.requires_grad is False


def test_the_gate_and_availability_buffers_stay_out_of_the_state_dict(tiny_gated):
    """Load-bearing here in a way it is not for the architecture parent: the keep-index's *length*
    is now the decoder's width, so a persistent copy would make a checkpoint trained at one budget
    fail to load at another as "keys did not align" rather than as a message about the budget."""
    model = _model(tiny_gated)
    keys = list(model.state_dict())

    for fragment in ("keep_index", "delay_steps", "availability", "start_indicator"):
        assert not [name for name in keys if fragment in name], fragment
    # The learned availability parameters do belong in it -- they are weights, not geometry.
    assert any("mask_proj" in name for name in keys)


def test_an_unguarded_model_has_no_gather_and_no_delay(tiny_kwargs):
    model = _model(tiny_kwargs)

    assert model.target_gate is None and model.source_gate is None
    assert not any(isinstance(m, (ChannelGate, ChannelDelay)) for m in model.modules())
    assert model.source_delay_steps == 0


def test_the_adapters_are_built_for_the_surviving_widths(tiny_gated):
    """The model still declares the full ``c_y`` / ``c_u`` -- the target builder's keep-index is
    positional into ``c_y`` -- while the adapters see only the survivors."""
    model = _model(tiny_gated)

    assert (model.c_y, model.c_u) == (TINY_KWARGS_C_Y, TINY_KWARGS_C_U)
    assert model.target_adapter.linear.in_features == 3
    assert model.source_adapter.linear.in_features == 2


#: The declared stream widths, which the gate narrows and the model keeps declaring.
TINY_KWARGS_C_Y = 109
TINY_KWARGS_C_U = 58


def test_the_shipped_widths_are_the_production_ones(shipped_gated):
    """A guard on the fixture rather than on the model: every count in this file is read against
    $c_y = 109$ and $H = 30$, and a changed production geometry must fail here first."""
    assert shipped_gated["c_y"] == _ALL_CHANNELS
    assert shipped_gated["horizon"] == 30
    assert len(shipped_gated["target_keep_index"]) == _KEPT_CHANNELS
    assert SHIPPED_KWARGS["horizon"] * _KEPT_CHANNELS == 2340
