r"""Construction: what the composition changes, what it inherits, and what it still refuses.

A composition is only worth having if it is provably one, so the tests split in two.

The first half pins the *difference* against the architecture parent, and there is exactly one:
each horizon token emits one value per surviving target channel instead of $R = 16$ raw samples. It
shows up in three places -- the decoder's two output heads, the parameter total, and the last axis
of the four forecast tensors -- and nowhere else. The parameter counts are written out because they
*are* the claim: the heads are $\mathrm{Linear}(256, X)$, so widening $X$ costs $2 \cdot 257 = 514$
parameters per channel and nothing anywhere else in the model may move.

The second half pins the *sameness*: the class body holds a constructor and nothing else, every
member resolves to the class the design names, the constructor signature is the architecture
parent's minus the two delay keywords plus this target domain's four, and every construction-time
refusal fires with the parent's message compared as a string.

**Why there is a constructor here and nowhere else in the composition.** The experiment driver
builds a run's kwargs by sweeping ``inspect.signature(MODEL_CLS.__init__)``, and this architecture's
keyword schema is not the conv-LSTM causal cell's -- five keys absent, seven encoder keys added --
so the schema has to be written where the class is. It holds no logic: the four causal keywords are
validated and set by the mixin, before and after ``super().__init__``, through the same two methods
the conv-LSTM cell calls.

**Why there is a forward-contract module in this package all the same.** The twenty-two-key return
set is the *mixin's*, not the architecture parent's, so unlike the two-sided conv-Transformer cell
this one cannot say "it is the parent's own code object, pinned by the parent's own suite" and stop.
``test_forward_contract.py`` pins it against the conv-LSTM causal cell instead.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from teb_vae.lag_attn.nets.heads import PriorHead
from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
from teb_vae.lag_attn_cfs.nets.causal_inputs import CausalWarmupInputs
from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
from teb_vae.lag_attn_cfs.tests.conftest import (
    shipped_warmup_kwargs as conv_lstm_shipped_warmup_kwargs,
)
from teb_vae.lag_attn_fs.nets.feature_target import FeatureForecastTarget
from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs
from teb_vae.lag_attn_transformer_cfs.tests.conftest import (
    CAUSAL_C_Y,
    CONV_LSTM_ONLY_KEYS,
    SHIPPED_KWARGS,
    TINY_STRIDE,
    TINY_TARGET_KEEP_INDEX,
    shipped_warmup_kwargs,
)
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

#: This model's measured totals at the shipped geometry, with and without the resolved warm-up
#: budget. They differ for two reasons at once: a guarded run builds availability input adapters the
#: unguarded one does not have, and the guarded decoder is $98$ channels wide against $102$.
_SHIPPED_GATED = 5_051_920
_SHIPPED_UNGATED = 5_035_416

#: The conv-LSTM causal cell's measured total at the same budget, and the difference. The encoder
#: swap costs exactly what it costs in the two-sided pair, which is the sense in which the grid's
#: two axes are independent.
_CONV_LSTM_SHIPPED_GATED = 5_143_262
_ENCODER_EDGE_PARAMETERS = 91_342

#: Surviving target channels at the shipped warm-up budget, and the declared width.
_KEPT_CHANNELS = 98


def _model(kwargs, cls=SeqVaeLagAttnTrfCfs, **overrides):
    torch.manual_seed(0)
    return cls(**dict(kwargs, **overrides))


def _n_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


# ---------------------------------------------------------------------------------------
# The geometry, and the width that follows the budget
# ---------------------------------------------------------------------------------------
def test_the_model_constructs_at_the_tiny_geometry(tiny_kwargs):
    model = _model(tiny_kwargs)

    assert model.geometry.raw_len == 384
    assert model.geometry.t_valid == 20


def test_the_model_constructs_at_the_production_geometry():
    model = _model(shipped_warmup_kwargs())

    assert model.geometry.raw_len == 4800
    assert model.geometry.t_valid == 285


def test_the_decoder_width_is_the_surviving_channel_count():
    """$98$ at the shipped warm-up budget. Not a configuration key -- and, as on the architecture
    parent, not even a constructor keyword -- so the width follows the gate and a run cannot decode
    a width its target does not have."""
    model = _model(shipped_warmup_kwargs())

    assert model.target_gate is not None
    assert model.target_gate.out_channels == _KEPT_CHANNELS
    assert model.decoder_out_channels == _KEPT_CHANNELS
    assert model.decoder.out_channels == _KEPT_CHANNELS
    assert model.decoder.mean_head.out_features == _KEPT_CHANNELS
    assert model.decoder.logvar_head.out_features == _KEPT_CHANNELS


def test_without_a_budget_the_decoder_width_is_the_declared_width(shipped_kwargs):
    """$102$ with no warm-up budget resolved. The unguarded arm is a configuration rather than an
    unhandled case, and its block cardinality is $H \\cdot c_y$ -- but it is not a run anyone should
    make on this dataset, because the leading region then enters the encoder as signal."""
    model = _model(shipped_kwargs)

    assert model.target_gate is None
    assert model.decoder_out_channels == CAUSAL_C_Y == model.c_y
    assert model.decoder.out_channels == CAUSAL_C_Y


def test_the_width_follows_a_gate_of_any_size(tiny_warmup):
    """The rule is the gate's count, not a constant."""
    model = _model(tiny_warmup)

    assert model.decoder_out_channels == len(TINY_TARGET_KEEP_INDEX)
    assert model.raw_per_step == 16, "the raw grid is geometry, not the decoder width"


def test_the_parameter_total_is_the_architectures_plus_the_wider_head():
    """Written out because it *is* the claim: the two heads are the only thing the target domain
    widens, so the gated and ungated totals must differ by the head cost of the four dropped
    channels plus whatever the availability adapters add -- and nothing else may move."""
    gated = _model(shipped_warmup_kwargs())
    ungated = _model(SHIPPED_KWARGS)

    assert _n_parameters(gated) == _SHIPPED_GATED
    assert _n_parameters(ungated) == _SHIPPED_UNGATED
    assert gated.decoder_out_channels + 4 == ungated.decoder_out_channels


def test_the_encoder_edge_costs_the_same_as_it_does_in_the_two_sided_pair():
    """The grid's premise, measured. Both cells forecast the same $98$-channel block over the same
    horizon at the same budget, so their difference is the encoder alone -- and it is the same
    $91{,}342$ parameters the two-sided pair differs by, which is what "the two axes are
    independent" has to mean numerically.

    A parameter total alone would also be satisfied by two models that differed in the head and
    compensated elsewhere, so the block itself is asserted first.
    """
    conv_lstm = _model(conv_lstm_shipped_warmup_kwargs(), cls=SeqVaeLagAttnCfs)
    transformer = _model(shipped_warmup_kwargs())

    for name in ("mean_head", "logvar_head"):
        assert (
            getattr(conv_lstm.decoder, name).weight.shape
            == getattr(transformer.decoder, name).weight.shape
        ), name
    assert conv_lstm.decoder_out_channels == transformer.decoder_out_channels == _KEPT_CHANNELS
    assert conv_lstm.horizon == transformer.horizon == 15
    assert conv_lstm.anchor_stride == transformer.anchor_stride == 15

    assert _n_parameters(conv_lstm) == _CONV_LSTM_SHIPPED_GATED
    assert _n_parameters(conv_lstm) - _n_parameters(transformer) == _ENCODER_EDGE_PARAMETERS


# ---------------------------------------------------------------------------------------
# What the class is
# ---------------------------------------------------------------------------------------
def test_the_class_defines_a_constructor_and_nothing_else():
    """Set equality over ``vars``, and the set is exactly ``{'__init__'}``.

    Not a line count, which passes a class that overrode ``forward`` in 140 lines. With nothing else
    defined here, the twenty-two forward keys, the posterior's structure, the lag map, the anchor
    tiling and the objective's metric set cannot have moved, because they are the two mixins' and
    the architecture parent's own code objects.
    """
    own = {
        name
        for name, value in vars(SeqVaeLagAttnTrfCfs).items()
        if callable(value) and not isinstance(value, type)
    }

    assert own == {"__init__"}
    assert {name for name in vars(SeqVaeLagAttnTrfCfs) if not name.startswith("__")} == set()
    assert "forward" not in vars(SeqVaeLagAttnTrfCfs)


def test_the_mro_puts_both_mixins_ahead_of_the_architecture():
    """Asserted as a list of names, because the *order* is what makes the width hook and the tiled
    forward resolve to the causal ones. Reversed, the decoder would be built at ``raw_per_step`` and
    a $98$-wide feature block scored against a $16$-wide forecast."""
    assert [cls.__name__ for cls in SeqVaeLagAttnTrfCfs.__mro__] == [
        "SeqVaeLagAttnTrfCfs",
        "CausalWarmupInputs",
        "CausalFeatureForecastTarget",
        "FeatureForecastTarget",
        "SeqVaeLagAttnTrfRws",
        "Module",
        "object",
    ]


def test_the_reversed_base_order_builds_a_raw_decoder(tiny_kwargs):
    r"""The reason the order is load-bearing, as a passing test rather than as a comment.

    Built explicitly here because the failure is loud in the wrong place. ``block_width`` would not
    catch it -- it feeds only the four log-variance diagnostics and no shape check -- so the first
    symptom is ``raw_sample_score`` computing $(\text{target} - \mu)^2$ on
    $(B, A, H, C_{\mathrm{keep}})$ against $(B, A, H, 16)$, three frames below the decision that
    caused it.

    Both consequences are asserted, because they are two different mistakes wearing one cause: the
    width hook resolves to ``raw_per_step``, and the *forward* resolves to the dense one, which
    decodes every anchor and returns no anchor set at all.
    """

    class _ReversedOrder(
        SeqVaeLagAttnTrfRws, CausalWarmupInputs, CausalFeatureForecastTarget
    ):
        """Deliberately wrong: the architecture ahead of the target domain."""

    torch.manual_seed(0)
    wrong = _ReversedOrder(**dict(tiny_kwargs))
    right = _model(tiny_kwargs)

    assert wrong.decoder.mean_head.out_features == 16 == wrong.raw_per_step
    assert right.decoder.mean_head.out_features == CAUSAL_C_Y
    assert _ReversedOrder.forward is SeqVaeLagAttnTrfRws.forward
    assert SeqVaeLagAttnTrfCfs.forward is CausalWarmupInputs.forward


@pytest.mark.parametrize(
    "name, owner",
    [
        ("forward", CausalWarmupInputs),
        ("_build_anchor_index", CausalWarmupInputs),
        ("_build_adapter", CausalWarmupInputs),
        ("build_lag_mask", CausalWarmupInputs),
        ("_set_causal_inputs", CausalWarmupInputs),
        ("_validate_causal_geometry", CausalWarmupInputs),
        ("_resolved_forecast_gaps", CausalFeatureForecastTarget),
        ("_check_anchor_floor", CausalFeatureForecastTarget),
        ("TARGET_BLOCK_SPLIT", CausalFeatureForecastTarget),
        ("SOURCE_BLOCK_SPLIT", CausalFeatureForecastTarget),
        ("_default_decoder_out_channels", FeatureForecastTarget),
        ("_build_forecast_target", FeatureForecastTarget),
        ("compute_loss", FeatureForecastTarget),
        ("_reparameterize_shared", SeqVaeLagAttnTrfRws),
        ("kld_tensor", SeqVaeLagAttnTrfRws),
        ("_build_channel_gate", SeqVaeLagAttnTrfRws),
    ],
)
def test_every_member_resolves_to_the_class_the_design_names(name, owner):
    """Identity, not equality. Two models are only comparable if the members they share are the
    *same* objects, rather than two implementations that agree today."""
    assert getattr(SeqVaeLagAttnTrfCfs, name) is getattr(owner, name)


@pytest.mark.parametrize(
    "name",
    ["forward", "_build_anchor_index", "_build_adapter", "build_lag_mask",
     "_resolved_forecast_gaps", "_default_decoder_out_channels", "TARGET_BLOCK_SPLIT"],
)
def test_the_two_causal_cells_share_every_target_domain_member(name):
    """The other direction, and the one that matters for the comparison: the conv-LSTM causal cell
    reaches the same objects. A member that had drifted onto one model would make a difference in
    results attributable to something other than the encoder."""
    assert getattr(SeqVaeLagAttnTrfCfs, name) is getattr(SeqVaeLagAttnCfs, name)


# ---------------------------------------------------------------------------------------
# The constructor signature
# ---------------------------------------------------------------------------------------
def test_the_signature_is_the_architecture_parents_with_the_delays_replaced():
    """The trainer builds its kwargs from an ``inspect.signature`` sweep of ``MODEL_CLS.__init__``.
    A narrowed signature would forward no configuration at all and silently build an all-defaults
    model -- no error, no shape mismatch, a run at the wrong widths.

    The two delay keywords are the only removals, and removing them is the point: a warm-up routed
    under a delay name would reach ``ChannelDelay``, which shifts rather than masks, and would train
    a different model with every shape intact.
    """
    parameters = inspect.signature(SeqVaeLagAttnTrfCfs.__init__).parameters
    base = inspect.signature(SeqVaeLagAttnTrfRws.__init__).parameters

    assert set(base) - set(parameters) == {"target_delays", "source_delays"}
    assert set(parameters) - set(base) == {
        "target_warmup_steps",
        "source_warmup_steps",
        "anchor_stride",
        "lag_floor",
    }
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


@pytest.mark.parametrize("key", CONV_LSTM_ONLY_KEYS)
def test_the_conv_lstm_only_keywords_are_refused(tiny_kwargs, key):
    """Each names a component this architecture does not have, so each must fail loudly rather than
    be accepted and ignored -- and each *is* a keyword of the conv-LSTM causal cell, which is what
    makes a config copied from that package fail here by name."""
    with pytest.raises(TypeError, match=key):
        _model(tiny_kwargs, **{key: 1})
    assert key in inspect.signature(SeqVaeLagAttnCfs.__init__).parameters


def test_the_geometry_defaults_are_the_causal_ones():
    """A run left at the architecture parent's defaults would be describing a dataset that does not
    exist: two-sided widths, a two-minute horizon and a floor $103$ steps below the one the kept
    channels admit."""
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(SeqVaeLagAttnTrfCfs.__init__).parameters.items()
    }

    assert defaults["c_y"] == CAUSAL_C_Y
    assert defaults["horizon"] == 15
    assert defaults["warmup_period"] == 133
    # The inert defaults: a model built with no opinion decodes densely and floors no lag, which is
    # what every sibling does. The tiling is a configuration decision and the config states it.
    assert defaults["anchor_stride"] == 1
    assert defaults["lag_floor"] == 0


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
        (dict(encoder_conv_kernels=(3, 3, 3)), "equal length"),
        (dict(target_attention_blocks=0), "at least 1"),
        (dict(source_attention_window=0), "at least 1 step"),
        (dict(encoder_num_heads=5), "divisible"),
    ],
    ids=["indivisible-latent", "head-geometry", "negative-lag", "zero-c_y", "zero-c_u",
         "stem-schedules", "no-attention", "zero-window", "indivisible-heads"],
)
def test_the_construction_refusals_are_the_architectures(tiny_kwargs, overrides, match):
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
    for cls in (SeqVaeLagAttnTrfRws, SeqVaeLagAttnTrfCfs):
        with pytest.raises(ValueError) as excinfo:
            _model(tiny_kwargs, cls=cls, **overrides)
        messages.append(str(excinfo.value))

    assert messages[0] == messages[1]


# ---------------------------------------------------------------------------------------
# The causal refusals, with the conv-LSTM cell's messages
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(anchor_stride=0), r"anchor_stride must be in"),
        (dict(anchor_stride=99), r"anchor_stride must be in"),
        (dict(lag_floor=-1), r"lag_floor must be >= 0"),
    ],
    ids=["stride-zero", "stride-above-horizon", "negative-floor"],
)
def test_the_causal_refusals_fire_here_too(tiny_warmup, overrides, match):
    with pytest.raises(ValueError, match=match):
        _model(tiny_warmup, **overrides)


def test_a_warmup_without_its_keep_index_is_refused(tiny_kwargs):
    """The one arrangement that would misroute a vector: the adapters are told apart by which gate
    they were handed, so a warm-up with no gate would leave both gates ``None`` and route the
    target's vector into both streams."""
    with pytest.raises(ValueError, match="target_warmup_steps was given without"):
        _model(tiny_kwargs, target_warmup_steps=(0, 1))


def test_a_floor_below_the_kept_channels_warmup_is_refused(tiny_warmup):
    """$F \\ge B - 1$, enforced at construction rather than assumed. Below it the objective scores
    the assumed pre-recording history of the slowest kept channel as signal, with every shape
    correct and every warm-fraction readout still reporting $1.0$."""
    with pytest.raises(ValueError, match="below the anchor floor"):
        _model(tiny_warmup, warmup_period=1)


def test_the_causal_refusal_messages_are_the_conv_lstm_cells(tiny_warmup):
    """The two cells share one target domain, so they must share its refusals *verbatim* -- a
    message that had drifted would be the first sign the two are no longer one design."""
    conv_lstm_kwargs = dict(
        tiny_warmup, lstm_layers=1
    )
    for key in ("encoder_conv_kernels", "encoder_conv_dilations", "encoder_num_heads",
                "encoder_d_ff", "target_attention_blocks", "source_attention_blocks",
                "source_attention_window"):
        conv_lstm_kwargs.pop(key)

    messages = []
    for cls, kwargs in (
        (SeqVaeLagAttnTrfCfs, tiny_warmup),
        (SeqVaeLagAttnCfs, conv_lstm_kwargs),
    ):
        with pytest.raises(ValueError) as excinfo:
            _model(kwargs, cls=cls, warmup_period=1)
        messages.append(str(excinfo.value))

    assert messages[0] == messages[1]


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
    """The architectural claim this cell inherits and the conv-LSTM causal cell cannot make: a
    statistic pooled over time on a history path would make $H_t$ read its own future, and there is
    none to disable -- ``causal_norm`` is not a keyword of this constructor at all."""
    model = _model(tiny_kwargs)

    assert not any(isinstance(module, torch.nn.LSTM) for module in model.modules())
    for name in ("target_encoder", "source_encoder"):
        for module in getattr(model, name).modules():
            assert not isinstance(
                module, (torch.nn.BatchNorm1d, torch.nn.GroupNorm, torch.nn.InstanceNorm1d)
            ), f"{name} carries a time-pooling normaliser: {type(module).__name__}"


# ---------------------------------------------------------------------------------------
# The four constants the readouts are computed against
# ---------------------------------------------------------------------------------------
def test_the_warmup_readout_constants_are_resolved_at_construction(tiny_warmup):
    """Resolved once from the budget and the geometry, and registered as **non-persistent** buffers
    so a checkpoint trained at one budget fails to load at another as a budget mismatch rather than
    as misaligned keys."""
    model = _model(tiny_warmup)

    assert model.target_warm_frac == 1.0
    assert model.warm_tertile_id.shape == (len(TINY_TARGET_KEEP_INDEX),)
    assert model.source_block_warm_st.shape == (model.sequence_length,)
    assert model.source_block_warm_ph.shape == (model.sequence_length,)

    state = model.state_dict()
    for name in ("warm_tertile_id", "source_block_warm_st", "source_block_warm_ph"):
        assert name not in state, f"{name} is persistent; a budget change would read as key drift"


def test_the_tiling_geometry_is_the_configured_one(tiny_warmup):
    """$A_{\\max} = \\lceil (T_{\\mathrm{valid}} - F)/S \\rceil$, a geometry constant no rank can
    disagree about."""
    model = _model(tiny_warmup, anchor_stride=TINY_STRIDE)
    anchors, valid = model._build_anchor_index(
        batch=2, device=torch.device("cpu"), anchor_phase=0, anchor_stride=TINY_STRIDE
    )

    span = model.geometry.t_valid - model.warmup_period
    assert anchors.shape == (2, -(-span // TINY_STRIDE))
    assert valid.all()
    assert anchors[0].tolist() == list(
        range(model.warmup_period, model.geometry.t_valid, TINY_STRIDE)
    )
