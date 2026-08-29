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
    TINY_TARGET_ALIGN_DELAYS,
    TINY_TARGET_KEEP_INDEX,
    TINY_TARGET_WARMUP_STEPS,
    build,
    shipped_warmup_kwargs,
)
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

#: This model's measured totals at the shipped geometry, with and without the resolved guard. They
#: differ for three reasons at once: a guarded run builds availability input adapters the unguarded
#: one does not have, the guarded decoder is $98$ channels wide against $102$, and the shipped
#: alignment narrows the source stream to $47$ while bringing both start-of-record embeddings into
#: existence.
_SHIPPED_GATED = 5_054_992
_SHIPPED_UNGATED = 5_039_256

#: The same model with the alignment off, which is the named comparison arm one key away at
#: ``causal_align_reference: null``. The $-768$ between the two factorises exactly: the source
#: adapter loses four channels from two $128$-wide linears, and both adapters gain a start vector.
_SHIPPED_GATED_UNALIGNED = 5_055_760

#: The conv-LSTM causal cell's measured total at the same guard, and the difference. The encoder
#: swap costs exactly what it costs in the two-sided pair, which is the sense in which the grid's
#: two axes are independent.
_CONV_LSTM_SHIPPED_GATED = 5_146_334
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
    assert model.geometry.t_valid == 270


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
    unaligned = _model(shipped_warmup_kwargs(align=False))

    assert _n_parameters(gated) == _SHIPPED_GATED
    assert _n_parameters(ungated) == _SHIPPED_UNGATED
    assert gated.decoder_out_channels + 4 == ungated.decoder_out_channels

    # The alignment arm, measured rather than asserted absent: it narrows the source stream by the
    # four channels above the reference and builds both start embeddings, which is $-768$ exactly.
    assert _n_parameters(unaligned) == _SHIPPED_GATED_UNALIGNED
    assert _n_parameters(gated) - _n_parameters(unaligned) == -4 * 128 * 2 + 2 * 128 == -768
    assert gated.source_gate.out_channels + 4 == unaligned.source_gate.out_channels


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
    assert conv_lstm.horizon == transformer.horizon == 30
    assert conv_lstm.anchor_stride == transformer.anchor_stride == 30

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
        # The alignment shifts. Present under names of their own rather than reusing the two that
        # were removed: they DO reach ``ChannelDelay``, so the distinction the removal draws is
        # between a warm-up mask and a shift, not between shifting and not shifting -- and a
        # checkpoint that recorded one under the other's name could not say which it was built at.
        "target_align_delays",
        "source_align_delays",
        # The per-block reconstruction weights. Keywords rather than a class constant because they
        # are a *run's* decision and must land in the checkpoint's ``model_kwargs``: a checkpoint
        # that did not record them would be a model whose objective could not be recovered.
        "target_weight_st",
        "target_weight_ph",
        # The novelty vector: a readout alone, changing no width, no mask and no parameter. It is
        # here rather than on the raw-target cells because a raw forecast window lies entirely
        # after the anchor, so a per-channel novelty share is undefined there rather than small.
        "target_novelty_frac",
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
    exist: two-sided widths, a two-minute horizon and a floor $104$ steps below the one the kept
    channels admit."""
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(SeqVaeLagAttnTrfCfs.__init__).parameters.items()
    }

    assert defaults["c_y"] == CAUSAL_C_Y
    assert defaults["horizon"] == 30
    assert defaults["warmup_period"] == 134
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
# The five constants the readouts are computed against
# ---------------------------------------------------------------------------------------
def test_the_warmup_readout_constants_are_resolved_at_construction(tiny_warmup):
    """Resolved once from the budget and the geometry, and registered as **non-persistent** buffers
    so a checkpoint trained at one budget fails to load at another as a budget mismatch rather than
    as misaligned keys."""
    model = _model(tiny_warmup)

    assert model.target_warm_frac == 1.0
    assert model.warm_tertile_id.shape == (len(TINY_TARGET_KEEP_INDEX),)
    # The second partition of the same kept axis, by how much of each coefficient the anchor has
    # not seen. Shaped by the survivors although the vector it comes from is declared-width, which
    # is the gather this cell inherits along with the mixin.
    assert model.novelty_tertile_id.shape == (len(TINY_TARGET_KEEP_INDEX),)
    assert model.source_block_warm_st.shape == (model.sequence_length,)
    assert model.source_block_warm_ph.shape == (model.sequence_length,)

    state = model.state_dict()
    for name in (
        "warm_tertile_id",
        "novelty_tertile_id",
        "source_block_warm_st",
        "source_block_warm_ph",
    ):
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


# =================================================================================================
# The channel alignment
# =================================================================================================
def test_omitting_the_alignment_keywords_builds_todays_model_bitwise(tiny_warmup) -> None:
    """The path an old checkpoint's saved kwargs dict actually exercises.

    A checkpoint written before these keywords existed carries neither, so construction from it
    must produce the object graph and the tensor values of the model that was trained -- not an
    equivalent one with an identity ``ChannelDelay`` in it. Asserted over the state dict *and* over
    the buffer names, because an identity shift is numerically invisible and structurally is not.
    """
    without = build(tiny_warmup)
    explicit = build(dict(tiny_warmup, target_align_delays=None, source_align_delays=None))

    assert list(without.state_dict()) == list(explicit.state_dict())
    for name, tensor in without.state_dict().items():
        assert torch.equal(tensor, explicit.state_dict()[name]), name
    assert sorted(dict(without.named_buffers())) == sorted(dict(explicit.named_buffers()))
    assert without.target_gate.max_delay == explicit.target_gate.max_delay == 0
    assert without.source_delay_steps == explicit.source_delay_steps == 0
    assert without.target_adapter.start_embed is None
    assert without.source_adapter.start_embed is None


def test_the_shift_reaches_the_gate_and_the_adapter_carries_warm_up_plus_shift(
    tiny_align,
) -> None:
    r"""The silent half of the alignment, on this cell's own composition.

    The keywords are renamed on the way to the base, so the gate is where they land; and a
    gathered-and-delayed channel is honest only once the step index has reached **both** $W'_c$ and
    $d_c$, so the vector the availability mask and the announcement are built from is the sum. Fed
    the warm-up alone, the adapter would call a channel warm $d_c$ steps early and every shape,
    every metric and every gradient would be exactly as they are now.
    """
    model = build(tiny_align)
    combined = tuple(
        wait + shift
        for wait, shift in zip(TINY_TARGET_WARMUP_STEPS, TINY_TARGET_ALIGN_DELAYS)
    )

    assert tuple(int(value) for value in model.target_gate.delay.delay_steps) == (
        TINY_TARGET_ALIGN_DELAYS
    )
    assert model.target_adapter.max_delay == max(combined)
    assert model.target_adapter.min_delay == min(combined)
    pattern = model.target_adapter.availability
    for channel, delay in enumerate(combined):
        column = pattern[:, channel]
        assert not bool(column[:delay].any()), channel
        assert bool(column[delay:].all()), channel


def test_the_availability_of_an_aligned_channel_is_the_unaligned_one_shifted_right(
    tiny_warmup, tiny_align
) -> None:
    """Stated as the relation rather than as two independent patterns: the whole claim of the
    alignment is that a channel's content moved later by $d_c$ steps and nothing else about it
    changed, so its availability must be the same staircase translated by the same $d_c$."""
    unaligned = build(tiny_warmup).target_adapter.availability
    aligned = build(tiny_align).target_adapter.availability
    steps = int(unaligned.shape[0])

    for channel, shift in enumerate(TINY_TARGET_ALIGN_DELAYS):
        expected = torch.zeros(steps, dtype=aligned.dtype)
        expected[shift:] = unaligned[: steps - shift, channel]
        assert torch.equal(aligned[:, channel], expected), channel


def test_a_non_null_reference_builds_the_start_of_record_embedding(
    tiny_warmup, tiny_align
) -> None:
    r"""A construction-time change no shipped configuration of this family has ever made.

    ``AvailabilityInputAdapter`` builds ``start_embed`` when $\min_c \delta_c > 0$. Unaligned, both
    streams have a channel at $W' = 0$, so the token is permanently inert and is not built. Under
    the shift the minimum of $W'_c + d_c$ lifts off zero on both streams and it comes into
    existence: a new learned parameter of width $d_{\mathrm{model}}$ per stream, and a live token in
    the forward pass. This is wanted, and it must be asserted rather than discovered in a parameter
    total.
    """
    off = build(tiny_warmup)
    on = build(tiny_align)

    assert off.target_adapter.start_embed is None and off.source_adapter.start_embed is None
    for adapter in (on.target_adapter, on.source_adapter):
        assert adapter.start_embed is not None
        assert adapter.start_embed.shape == (int(on.d_model),)
        assert bool(adapter.start_indicator.any()), "an inert token is not a token"

    assert sum(p.numel() for p in on.parameters()) - sum(
        p.numel() for p in off.parameters()
    ) == 2 * int(on.d_model)


def test_the_anchor_floor_rises_to_the_shifted_warmth(tiny_align, tiny_warmup) -> None:
    r"""Both requirements, and they do not move together.

    The scored target is never shifted, so its half stays where it was. The *inputs* are, and an
    aligned channel vector at step $t$ asserts one physical instant -- an assertion that is false,
    not partially true, while any entry has not arrived. So the floor must clear
    $\max_c(W'_c + d_c)$, which costs exactly one anchor here as it does at the shipped reference.
    The unaligned floor is unmoved, which is the control: a check applying the second half
    unconditionally would refuse the shipped configuration.
    """
    floor = int(tiny_align["warmup_period"])
    assert floor == max(TINY_TARGET_WARMUP_STEPS), "the flat combined vector is what this pins"

    with pytest.raises(ValueError) as error:
        build(dict(tiny_align, warmup_period=floor - 1))
    assert f"warmup_period={floor - 1}" in str(error.value)
    assert f"at least {floor}" in str(error.value)

    assert build(dict(tiny_align, warmup_period=floor)) is not None
    assert build(dict(tiny_warmup, warmup_period=floor - 1)) is not None


# =================================================================================================
# The revision's five switches, and the off-state of each
#
# Pinned per cell rather than once on the parent, because the failure this catches is *this* cell's:
# the driver builds a run's kwargs by sweeping the constructor's signature and silently drops any
# key the class does not re-list, so a switch threaded through the parent and forgotten here would
# train the baseline under the arm's name with no error and no metric saying so.
#
# The other half is the off-state. Every mechanism must reproduce, bitwise and key for key, the
# model that was trained before it existed -- that is what makes an arm comparable to a record, and
# what a checkpoint written under one setting and read under another silently violates.
# =================================================================================================
#: The five keywords the revision added to this constructor, at their off-values. Written out
#: rather than derived from the signature's defaults: comparing the defaults against themselves
#: would pass on any edit, and what has to hold is that these particular values reproduce the
#: pre-revision model.
_SWITCHES_OFF = dict(
    lag_kv_source="encoder",
    prior_availability_input=False,
    persistence_residual=False,
    horizon_weight_halflife_steps=None,
    alibi_slope_scale=1.0,
)


def test_every_switch_is_a_keyword_of_this_cell_at_its_off_default() -> None:
    """Both halves of the sweep hazard: present, and defaulting off.

    Present, because a key absent here is a key the driver cannot forward -- the arm would train as
    the baseline. Defaulting off, because the default is what an old checkpoint's saved kwargs dict
    falls back to, and a default that moved would silently rebuild an old run as a new architecture.
    """
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(SeqVaeLagAttnTrfCfs.__init__).parameters.items()
        if name in _SWITCHES_OFF
    }

    assert defaults == _SWITCHES_OFF


def test_every_switch_at_its_off_value_is_bitwise_the_model_without_the_keywords(
    tiny_warmup,
) -> None:
    """The whole off-state claim in one comparison, over the state dict and the buffer names.

    Values as well as keys, because a switch that added a zero-initialised parameter would leave
    the totals standing and change the object; and buffer names as well as parameters, because a
    non-persistent buffer is invisible to a ``state_dict`` comparison and is exactly how the
    horizon weight and the availability announcement are carried.
    """
    without = _model(tiny_warmup)
    explicit = _model(tiny_warmup, **_SWITCHES_OFF)

    assert list(without.state_dict()) == list(explicit.state_dict())
    for name, tensor in without.state_dict().items():
        assert torch.equal(tensor, explicit.state_dict()[name]), name
    assert sorted(dict(without.named_buffers())) == sorted(dict(explicit.named_buffers()))


@pytest.mark.parametrize(
    "keyword, absent",
    [
        ("prior_availability_input", "prior_head.clock_proj.weight"),
        ("persistence_residual", "decoder.persistence_weight"),
    ],
)
def test_an_off_switch_builds_no_parameter_at_all(tiny_warmup, keyword, absent) -> None:
    """Absent rather than present-and-zero, and the difference is a distributed run's: a parameter
    built and left inert has no gradient path, which is what ``find_unused_parameters=False``
    refuses. This is the encoder whose reachability suite runs, so the two halves have to agree."""
    off = _model(tiny_warmup, **{keyword: False})
    on = _model(tiny_warmup, **{keyword: True})

    assert absent not in dict(off.named_parameters())
    assert absent in dict(on.named_parameters())


def test_the_horizon_weight_is_a_non_persistent_buffer_or_nothing(tiny_warmup) -> None:
    r"""Null builds no buffer; a half-life builds one that a checkpoint does not carry.

    Non-persistent is the load-bearing half. The weight is $(H,)$, so a persistent one would put
    the horizon into the state dict and make a checkpoint unloadable at any other horizon -- for a
    tensor that is a pure function of two numbers the constructor already has.
    """
    off = _model(tiny_warmup, horizon_weight_halflife_steps=None)
    on = _model(tiny_warmup, horizon_weight_halflife_steps=5.0)

    assert "horizon_weight" not in dict(off.named_buffers())
    assert "horizon_weight" in dict(on.named_buffers())
    assert on.horizon_weight.shape == (on.horizon,)
    assert float(on.horizon_weight.sum()) == pytest.approx(float(on.horizon), rel=1e-6)
    assert not any("horizon_weight" in name for name in on.state_dict())


# =================================================================================================
# The lag attention's key/value memory
# =================================================================================================
def test_an_unknown_kv_source_is_refused_naming_the_choices(tiny_warmup) -> None:
    """By name, with the admitted set in the message. The value reaches a branch that would
    otherwise fall through to one of the arms, so an unrecognised string would silently train the
    fall-through arm under the misspelt one's name."""
    with pytest.raises(ValueError, match=r"lag_kv_source must be one of"):
        _model(tiny_warmup, lag_kv_source="conv-stem")


@pytest.mark.parametrize("arm", ["conv_stem", "adapter"])
def test_a_local_kv_arm_does_not_build_the_deep_source_encoder(tiny_warmup, arm) -> None:
    """The windowed source encoder leaves the *model*, not just the lag path.

    Nothing else consumes the source state, so under a local arm it would be a whole attention
    stack no forward reaches. On this encoder that is the larger of the two savings and the one
    the design's parameter table is read on, which is why it is asserted by state-dict prefix
    rather than by a total: a total cannot say which stack went.
    """
    deep = _model(tiny_warmup, lag_kv_source="encoder")
    local = _model(tiny_warmup, lag_kv_source=arm)

    assert [name for name in deep.state_dict() if name.startswith("source_encoder.")]
    assert [name for name in local.state_dict() if name.startswith("source_encoder.")] == []
    assert getattr(local, "source_encoder", None) is None
    assert sum(p.numel() for p in local.parameters()) < sum(p.numel() for p in deep.parameters())


def test_the_conv_stem_arm_builds_a_bounded_stem_and_the_adapter_arm_builds_nothing(
    tiny_warmup,
) -> None:
    r"""What each local arm puts in the encoder's place, and what resolves the arm.

    ``source_kv_body`` is the single place the arm becomes a module, and every consumer -- the
    forward, both source controls, the prior clock, the norm guard -- goes through it or through
    ``encode_source_kv``, so pinning it is pinning that they cannot disagree.

    **The stem's reach is asserted, not assumed.** The whole content of a local arm is that the
    value at lag $\ell$ is a function of a bounded window rather than of the prefix; a stem that
    inherited a long dilation tail would satisfy every structural assertion above and still be
    effectively whole-prefix, at which point the arm tests nothing.
    """
    stem = _model(tiny_warmup, lag_kv_source="conv_stem")
    adapter = _model(tiny_warmup, lag_kv_source="adapter")
    deep = _model(tiny_warmup, lag_kv_source="encoder")

    assert stem.source_kv_body() is stem.source_kv_stem
    assert adapter.source_kv_body() is None
    assert deep.source_kv_body() is deep.source_encoder

    assert adapter.source_kv_modules() == (adapter.source_adapter,)
    assert stem.source_kv_modules() == (stem.source_adapter, stem.source_kv_stem)
    assert 1 < stem.source_kv_stem.receptive_field < stem.sequence_length


@pytest.mark.parametrize("arm", ["encoder", "conv_stem", "adapter"])
def test_the_kv_representation_is_the_pathway_composed_in_order(tiny_warmup, arm) -> None:
    """``encode_source_kv`` is what the forward and both controls call, so what it computes has to
    be exactly the pathway's modules in order -- a helper that dropped or reordered one would move
    the keys, the values, the null control's re-encode and the prior's clock together, and every
    one of them would still be the right shape."""
    model = _model(tiny_warmup, lag_kv_source=arm).eval()
    gated = torch.randn(
        2,
        model.sequence_length,
        model.source_gate.keep_index.numel(),
        generator=torch.Generator().manual_seed(3),
    )

    with torch.no_grad():
        encoded = model.encode_source_kv(gated)
        expected = gated
        for module in model.source_kv_modules():
            expected = module(expected)

    assert torch.equal(encoded, expected)
    assert encoded.shape == (2, model.sequence_length, model.d_model)
