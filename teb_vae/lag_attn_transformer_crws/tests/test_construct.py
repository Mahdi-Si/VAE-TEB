r"""Construction: what the composition changes, what it inherits, and what it still refuses.

This cell composes one mixin over the conv-Transformer architecture, and the two ways that goes
wrong are opposite in kind.

**Too little.** A member that should be inherited is written here instead, and the conv-LSTM cell of
this row -- which composes the same mixin -- silently keeps a second copy. ``vars()`` is asserted to
be exactly ``{'__init__'}`` for that reason, and the constructor is the one member that genuinely
cannot be shared: the experiment driver builds a run's kwargs by sweeping
``inspect.signature(MODEL_CLS.__init__)``, and this architecture's keyword schema is not the
conv-LSTM cell's -- five keys absent, seven encoder keys added -- so the schema has to be written
where the class is.

**Too much.** The causal-feature *target* mixin is composed in as well, which reads perfectly
plausibly because both mixins are "the causal half". Then ``_default_decoder_out_channels`` resolves
to the feature one, the decoder emits $C_{\mathrm{keep}} = 98$ channels per horizon token, and the
target this cell gathers is $(B, A, H, 16)$. ``block_width`` does not catch it: it feeds only the
four log-variance diagnostics and no shape check. The first symptom is ``raw_sample_score`` computing
$(\text{target} - \mu)^2$ on shapes that do not broadcast, three frames below the class body that
caused it.

The parameter totals are written out because they *are* a claim. Unlike the causal-feature pair,
nothing about this target domain widens a head -- the decoder is $R = 16$ wide with a budget and
without one -- so the gated-to-ungated difference is the two availability input adapters and nothing
else, and the difference against the conv-LSTM cell of this row is the encoder and nothing else.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn.nets.heads import PriorHead
from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
from teb_vae.lag_attn_cfs.nets.causal_inputs import (
    FORWARDED_EXCLUSIONS,
    CausalWarmupInputs,
)
from teb_vae.lag_attn_crws.nets.causal_raw_inputs import CausalRawInputs
from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_crws.tests.conftest import (
    shipped_warmup_kwargs as conv_lstm_shipped_warmup_kwargs,
)
from teb_vae.lag_attn_transformer_crws.nets.model import SeqVaeLagAttnTrfCrws
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

from .conftest import (
    CAUSAL_C_U,
    CAUSAL_C_Y,
    CONV_LSTM_ONLY_KEYS,
    SHIPPED_HORIZON,
    SHIPPED_KWARGS,
    SHIPPED_WARMUP_PERIOD,
    TINY_SOURCE_WARMUP_STEPS,
    TINY_STRIDE,
    TINY_TARGET_KEEP_INDEX,
    TINY_TARGET_WARMUP_STEPS,
    build,
    shipped_warmup_kwargs,
)

#: The seven keywords that are this architecture's own. Each names a component the conv-LSTM cell
#: has no analogue of, and together they are the entire declared difference between the two cells of
#: this row.
_ENCODER_KEYS = (
    "encoder_conv_kernels",
    "encoder_conv_dilations",
    "encoder_num_heads",
    "encoder_d_ff",
    "target_attention_blocks",
    "source_attention_blocks",
    "source_attention_window",
)

#: This model's measured totals at the shipped geometry, with and without the resolved warm-up
#: budget. They differ for exactly one reason, unlike in the causal-feature pair: a guarded run
#: builds two availability input adapters an unguarded one does not have. The decoder is $R = 16$
#: wide either way, because the raw block's width is geometry rather than a gate's survivor count.
_SHIPPED_GATED = 5_009_772
_SHIPPED_UNGATED = 4_991_212

#: The conv-LSTM cell of this row at the same budget, and the difference. The encoder swap costs
#: exactly what it costs in the causal-feature pair and in the two-sided pair, which is the sense in
#: which the grid's two axes are independent.
_CONV_LSTM_SHIPPED_GATED = 5_101_114
_ENCODER_EDGE_PARAMETERS = 91_342

#: What the shipped budget keeps of the $102$ declared target-stream channels. A literal here rather
#: than a resolved count, because the wrong-composition guard below is about a width that must *not*
#: appear on this cell's decoder, and deriving it from the same resolution the model uses would let
#: both move together.
_KEPT_TARGET_CHANNELS = 98


def _model(kwargs, cls=SeqVaeLagAttnTrfCrws, **overrides):
    torch.manual_seed(0)
    return cls(**dict(kwargs, **overrides))


def _n_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


# =================================================================================================
# The class
# =================================================================================================
def test_the_mixin_comes_first_and_the_mro_is_pinned() -> None:
    r"""As a list of names, because the *order* is what makes the tiled forward, the warm-up adapter,
    the floored lag mask and the anchored objective resolve to the causal ones.

    Reversed, the architecture's dense forward wins: it decodes $[0, T_{\mathrm{valid}})$, returns no
    anchor set at all, and the objective then gathers a dense target for it -- a different model with
    every shape self-consistent.
    """
    assert SeqVaeLagAttnTrfCrws.__bases__ == (CausalRawInputs, SeqVaeLagAttnTrfRws)
    assert [cls.__name__ for cls in SeqVaeLagAttnTrfCrws.__mro__] == [
        "SeqVaeLagAttnTrfCrws",
        "CausalRawInputs",
        "CausalWarmupInputs",
        "SeqVaeLagAttnTrfRws",
        "Module",
        "object",
    ]


def test_the_model_carries_nothing_but_its_constructor() -> None:
    """Set equality over ``vars``, and the set is exactly ``{'__init__'}``.

    Not a line count, which passes a class that overrode ``forward`` in 140 lines. With nothing else
    defined here, the twenty-two forward keys, the posterior's structure, the lag map, the anchor
    tiling and the objective's metric set cannot have moved, because they are the mixin's and the
    architecture parent's own code objects.
    """
    own = {
        name
        for name, value in vars(SeqVaeLagAttnTrfCrws).items()
        if callable(value) and not isinstance(value, type)
    }

    assert own == {"__init__"}
    assert {name for name in vars(SeqVaeLagAttnTrfCrws) if not name.startswith("__")} == set()

    for shared in ("forward", "_build_anchor_index", "_build_adapter", "build_lag_mask"):
        assert shared not in vars(SeqVaeLagAttnTrfCrws), shared
        assert shared in vars(CausalWarmupInputs), shared
    for shared in ("compute_loss", "_check_anchor_floor"):
        assert shared not in vars(SeqVaeLagAttnTrfCrws), shared
        assert shared in vars(CausalRawInputs), shared


@pytest.mark.parametrize(
    "name, owner",
    [
        ("forward", CausalWarmupInputs),
        ("_build_anchor_index", CausalWarmupInputs),
        ("_build_adapter", CausalWarmupInputs),
        ("build_lag_mask", CausalWarmupInputs),
        ("_set_causal_inputs", CausalWarmupInputs),
        ("_validate_causal_geometry", CausalWarmupInputs),
        ("compute_loss", CausalRawInputs),
        ("_check_anchor_floor", CausalRawInputs),
        ("_resolve_warmup_readout_constants", CausalRawInputs),
        ("_reparameterize_shared", SeqVaeLagAttnTrfRws),
        ("kld_tensor", SeqVaeLagAttnTrfRws),
        ("_build_channel_gate", SeqVaeLagAttnTrfRws),
    ],
)
def test_every_member_resolves_to_the_class_the_design_names(name, owner) -> None:
    """Identity, not equality. Two models are only comparable if the members they share are the
    *same* objects, rather than two implementations that agree today."""
    assert getattr(SeqVaeLagAttnTrfCrws, name) is getattr(owner, name)


@pytest.mark.parametrize(
    "name",
    [
        "forward",
        "_build_anchor_index",
        "_build_adapter",
        "build_lag_mask",
        "compute_loss",
        "_check_anchor_floor",
        "_anchors_per_sample",
        "_source_lag_warmth",
    ],
)
def test_the_two_cells_of_this_row_share_every_input_domain_member(name) -> None:
    """The other direction, and the one that matters for the comparison: the conv-LSTM cell reaches
    the same objects. A member that had drifted onto one model would make a difference in results
    attributable to something other than the encoder."""
    assert getattr(SeqVaeLagAttnTrfCrws, name) is getattr(SeqVaeLagAttnCrws, name)


# =================================================================================================
# The constructor signature
# =================================================================================================
def test_the_signature_is_the_architecture_parents_with_the_delays_replaced() -> None:
    """The trainer builds its kwargs from an ``inspect.signature`` sweep of ``MODEL_CLS.__init__``.
    A narrowed signature would forward no configuration at all and silently build an all-defaults
    model -- no error, no shape mismatch, a run at the wrong widths.

    The two delay keywords are the only removals, and removing them is the point: a warm-up routed
    under a delay name would reach ``ChannelDelay``, which shifts rather than masks, and would train
    a different model with every shape intact.
    """
    parameters = inspect.signature(SeqVaeLagAttnTrfCrws.__init__).parameters
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


def test_the_forwarded_set_is_this_signature_minus_the_mixins_own_keywords() -> None:
    """Captured from ``locals()`` rather than written out a second time.

    Forty explicit ``name=name`` pairs would be the same dict with one silent failure mode: a
    keyword added to the architecture parent and forgotten here would be forwarded at its default,
    with nothing raising and no shape differing.
    """
    parameters = set(inspect.signature(SeqVaeLagAttnTrfCrws.__init__).parameters)
    base = set(inspect.signature(SeqVaeLagAttnTrfRws.__init__).parameters)

    forwarded = parameters - set(FORWARDED_EXCLUSIONS)
    assert forwarded | {"self", "target_delays", "source_delays"} == base

    source = Path(inspect.getfile(SeqVaeLagAttnTrfCrws)).read_text(encoding="utf-8")
    assert "locals().items()" in source and "FORWARDED_EXCLUSIONS" in source


@pytest.mark.parametrize("key", _ENCODER_KEYS)
def test_the_seven_encoder_keywords_are_accepted(key) -> None:
    """The positive half of the schema claim. Each is a keyword here and is **not** one of the
    conv-LSTM cell's, so a config written for that cell reaches nothing here and vice versa."""
    assert key in inspect.signature(SeqVaeLagAttnTrfCrws.__init__).parameters
    assert key not in inspect.signature(SeqVaeLagAttnCrws.__init__).parameters


def test_the_seven_encoder_keywords_reach_the_encoders(tiny_kwargs) -> None:
    """Accepted is not enough: a keyword silently swallowed into ``forwarded`` and never read would
    satisfy the signature check above. Moving the two depth keys must move the parameter total, and
    moving the window must move what the source encoder is allowed to attend to."""
    base = _model(tiny_kwargs)
    deeper = _model(tiny_kwargs, target_attention_blocks=3, source_attention_blocks=3)
    wider = _model(tiny_kwargs, encoder_d_ff=128)

    assert _n_parameters(deeper) > _n_parameters(base)
    assert _n_parameters(wider) > _n_parameters(base)
    assert base.source_encoder.attention_window == tiny_kwargs["source_attention_window"]
    assert base.target_encoder.attention_window is None, "the target reads the full causal prefix"


@pytest.mark.parametrize("key", CONV_LSTM_ONLY_KEYS)
def test_the_conv_lstm_only_keywords_are_refused(tiny_kwargs, key) -> None:
    """Each names a component this architecture does not have, so each must fail loudly rather than
    be accepted and ignored. Four of the five *are* keywords of the conv-LSTM cell of this row, which
    is what makes a config copied from that package fail here by name; ``conv_norm_groups`` is the
    fifth and is that cell's too, even though its shipped config leaves it unset."""
    with pytest.raises(TypeError, match=key):
        _model(tiny_kwargs, **{key: 1})
    assert key in inspect.signature(SeqVaeLagAttnCrws.__init__).parameters


def test_the_geometry_defaults_are_the_causal_datasets() -> None:
    """A run left at the architecture parent's defaults would be describing a dataset that does not
    exist: two-sided widths, a two-minute horizon and a floor $103$ steps below the one the kept
    channels admit.

    None of the four is forced by this target -- a raw sample is honest at every step, so no validity
    constraint ties the floor to the resolved budget here. They are held at the conv-LSTM cell's
    values so that the two differ in exactly one variable.
    """
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(
            SeqVaeLagAttnTrfCrws.__init__
        ).parameters.items()
    }

    assert defaults["c_y"] == CAUSAL_C_Y
    assert defaults["c_u"] == CAUSAL_C_U
    assert defaults["horizon"] == SHIPPED_HORIZON
    assert defaults["warmup_period"] == SHIPPED_WARMUP_PERIOD
    assert defaults["raw_per_step"] == 16
    # The inert defaults: a model built with no opinion decodes densely and floors no lag, which is
    # what every sibling does. The tiling is a configuration decision and the config states it.
    assert defaults["anchor_stride"] == 1
    assert defaults["lag_floor"] == 0


# =================================================================================================
# The decoder's width, and the mixin that must not be composed
# =================================================================================================
def test_the_model_constructs_at_both_geometries(tiny_kwargs) -> None:
    tiny = _model(tiny_kwargs)
    shipped = _model(shipped_warmup_kwargs())

    assert (tiny.geometry.raw_len, tiny.geometry.t_valid) == (384, 20)
    assert (shipped.geometry.raw_len, shipped.geometry.t_valid) == (4800, 285)


def test_the_decoder_emits_raw_samples_because_no_width_hook_is_defined(tiny_warmup) -> None:
    r"""The load-bearing **absence**: neither this class nor its mixin defines
    ``_default_decoder_out_channels``, so it resolves to the architecture's ``raw_per_step``."""
    model = _model(tiny_warmup)

    assert model.decoder.mean_head.out_features == 16 == model.raw_per_step
    assert model.decoder.logvar_head.out_features == 16
    assert model.decoder_out_channels == 16
    assert "_default_decoder_out_channels" not in vars(SeqVaeLagAttnTrfCrws)
    assert "_default_decoder_out_channels" not in vars(CausalRawInputs)
    assert (
        SeqVaeLagAttnTrfCrws._default_decoder_out_channels
        is vars(SeqVaeLagAttnTrfRws)["_default_decoder_out_channels"]
    )


def test_the_shipped_decoder_is_raw_rather_than_the_surviving_channel_count() -> None:
    """The width the causal-feature cell of this architecture builds, asserted absent here."""
    model = _model(shipped_warmup_kwargs())

    assert model.target_gate is not None
    assert model.target_gate.out_channels == _KEPT_TARGET_CHANNELS
    assert model.decoder.mean_head.out_features == 16 != _KEPT_TARGET_CHANNELS


def test_composing_the_feature_target_mixin_builds_the_wrong_decoder() -> None:
    r"""Why that absence is load-bearing, as a passing test rather than as a comment.

    Composed as ``(CausalFeatureForecastTarget, SeqVaeLagAttnTrfCrws)`` rather than as a literal
    three-base tuple, for a mechanical reason worth recording: that tuple leaves ``__init__``
    resolving to the architecture's, which does not accept this domain's four keywords, so the wrong
    model could not be constructed at all and the guard would prove nothing. Subclassing this cell
    keeps its constructor and moves only the resolution order, which is exactly the mistake.
    """
    wrong_composition = type(
        "SeqVaeLagAttnTrfCrwsWithFeatureTarget",
        (CausalFeatureForecastTarget, SeqVaeLagAttnTrfCrws),
        {},
    )
    kwargs = shipped_warmup_kwargs()

    torch.manual_seed(0)
    wrong = wrong_composition(**dict(kwargs))
    right = _model(kwargs)

    assert wrong.decoder.mean_head.out_features == _KEPT_TARGET_CHANNELS
    assert right.decoder.mean_head.out_features == 16 == right.raw_per_step
    # The cause, stated beside the symptom: which class the width hook resolved to.
    assert wrong_composition._default_decoder_out_channels is not (
        SeqVaeLagAttnTrfCrws._default_decoder_out_channels
    )


# =================================================================================================
# The parameter budget
# =================================================================================================
def test_the_budget_costs_only_the_two_availability_adapters() -> None:
    """Written out because it *is* the claim. Nothing in this target domain widens a head -- the
    decoder is $R = 16$ wide with a budget and without one -- so the gated-to-ungated difference is
    the two availability input adapters and must be nothing else."""
    gated = _model(shipped_warmup_kwargs())
    ungated = _model(SHIPPED_KWARGS)

    assert _n_parameters(gated) == _SHIPPED_GATED
    assert _n_parameters(ungated) == _SHIPPED_UNGATED
    assert gated.decoder_out_channels == ungated.decoder_out_channels == 16

    gated_names = {name for name, _ in gated.named_parameters()}
    ungated_names = {name for name, _ in ungated.named_parameters()}
    added = gated_names - ungated_names
    assert added, "the guarded model added no parameter at all"
    assert all("adapter" in name for name in added), sorted(added)
    assert ungated_names - gated_names == set()


def test_the_encoder_edge_costs_the_same_as_it_does_in_the_causal_feature_pair() -> None:
    """The grid's premise, measured. Both cells of this row forecast the same $H \\cdot R$ raw block
    at the same budget over the same tiling, so their difference is the encoder alone -- and it is
    the same $91{,}342$ parameters every other pair in the grid differs by.

    A parameter total alone would also be satisfied by two models that differed in the head and
    compensated elsewhere, so the decoder is asserted identical first.
    """
    conv_lstm = _model(conv_lstm_shipped_warmup_kwargs(), cls=SeqVaeLagAttnCrws)
    transformer = _model(shipped_warmup_kwargs())

    for name in ("mean_head", "logvar_head"):
        assert (
            getattr(conv_lstm.decoder, name).weight.shape
            == getattr(transformer.decoder, name).weight.shape
        ), name
    assert conv_lstm.decoder_out_channels == transformer.decoder_out_channels == 16
    assert conv_lstm.horizon == transformer.horizon == SHIPPED_HORIZON
    assert conv_lstm.anchor_stride == transformer.anchor_stride == SHIPPED_HORIZON

    assert _n_parameters(conv_lstm) == _CONV_LSTM_SHIPPED_GATED
    assert _n_parameters(conv_lstm) - _n_parameters(transformer) == _ENCODER_EDGE_PARAMETERS


# =================================================================================================
# The guard the constructor builds
# =================================================================================================
def test_the_adapter_is_built_at_the_warm_up_and_not_at_the_gates_delays(tiny_warmup) -> None:
    r"""The specific failure the inherited ``_build_adapter`` exists to prevent.

    ``gate.delay.delay_steps`` is all zeros under a pure gather, so the architecture's own version
    would give ``max_delay = 0`` -- no availability buffer, no mask projection, and a leading region
    of real-valued pre-recording history entering the encoder as though it were signal.
    """
    model = _model(tiny_warmup)

    assert model.target_adapter.max_delay == max(TINY_TARGET_WARMUP_STEPS)
    assert model.source_adapter.max_delay == max(TINY_SOURCE_WARMUP_STEPS)
    assert model.target_adapter.mask_proj is not None
    assert model.source_adapter.mask_proj is not None
    assert model.target_gate is not None
    assert model.target_gate.keep_index.tolist() == list(TINY_TARGET_KEEP_INDEX)
    assert model.target_adapter.in_dim == len(TINY_TARGET_KEEP_INDEX)
    assert model.source_gate is not None
    assert model.source_gate.max_delay == 0, "the gate is a gather; the warm-up is not a shift"


def test_the_two_source_warmth_buffers_are_resolved_and_non_persistent(tiny_warmup) -> None:
    """Two, not the causal-feature cell's three: ``warm_tertile_id`` partitions kept *target*
    channels and this target has none, so it must be absent rather than empty.

    Non-persistent so a checkpoint trained at one budget fails to load at another as a budget
    mismatch rather than as misaligned keys.
    """
    model = _model(tiny_warmup)

    assert model.source_block_warm_st.shape == (model.sequence_length,)
    assert model.source_block_warm_ph.shape == (model.sequence_length,)
    assert not hasattr(model, "warm_tertile_id")
    assert not hasattr(model, "target_warm_frac")

    state = model.state_dict()
    for name in ("source_block_warm_st", "source_block_warm_ph"):
        assert name not in state, f"{name} is persistent; a budget change would read as key drift"


def test_the_tiling_geometry_is_the_configured_one(tiny_warmup) -> None:
    r"""$A_{\max} = \lceil (T_{\mathrm{valid}} - F)/S \rceil$, a geometry constant no rank can
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
# The ungated arm and the inherited structure
# =================================================================================================
def test_without_a_budget_no_gate_and_no_availability_term_is_built(tiny_kwargs) -> None:
    """An unguarded run gets the model that has no guard, not an identity one."""
    model = _model(tiny_kwargs)

    assert model.target_gate is None and model.source_gate is None
    assert model.target_warmup_steps is None and model.source_warmup_steps is None
    for adapter in (model.target_adapter, model.source_adapter):
        assert adapter.mask_proj is None
        assert adapter.start_embed is None
        assert not hasattr(adapter, "availability")
    assert model.decoder_out_channels == 16


def test_the_ungated_model_is_parameter_for_parameter_the_architecture_parent(tiny_kwargs) -> None:
    """This class adds no parameter of its own: what it changes is which anchors are decoded.

    Compared against the raw-target architecture at the same keywords -- the comparison that is
    available here and is *not* available to the causal-feature cells, whose decoder is a different
    width. A parameter creeping in here (a second adapter, a learned phase, a floor embedding) fails
    rather than being absorbed into a total nobody re-derives.
    """
    causal = _model(tiny_kwargs)
    torch.manual_seed(0)
    raw = SeqVaeLagAttnTrfRws(**dict(tiny_kwargs))

    assert {name: tuple(p.shape) for name, p in causal.named_parameters()} == {
        name: tuple(p.shape) for name, p in raw.named_parameters()
    }
    assert _n_parameters(causal) == _n_parameters(raw)


def test_no_decoder_state_head_and_no_second_decoder_exist(tiny_kwargs) -> None:
    model = _model(tiny_kwargs)

    assert not hasattr(model, "residual_decoder")
    assert not hasattr(model, "baseline_decoder")
    assert not any(isinstance(module, PriorHead) for module in model.modules())
    assert not hasattr(model.prior_head, "decoder_state_head")


def test_no_recurrence_and_no_time_pooling_normaliser_on_a_history_path(tiny_kwargs) -> None:
    """The architectural claim this cell inherits and the conv-LSTM cell of this row cannot make: a
    statistic pooled over time on a history path would make $H_t$ read its own future, and there is
    none to disable -- ``causal_norm`` is not a keyword of this constructor at all."""
    model = _model(tiny_kwargs)

    assert not any(isinstance(module, torch.nn.LSTM) for module in model.modules())
    for name in ("target_encoder", "source_encoder"):
        for module in getattr(model, name).modules():
            assert not isinstance(
                module, (torch.nn.BatchNorm1d, torch.nn.GroupNorm, torch.nn.InstanceNorm1d)
            ), f"{name} carries a time-pooling normaliser: {type(module).__name__}"


# =================================================================================================
# The refusals
# =================================================================================================
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
def test_the_construction_refusals_are_the_architectures(tiny_kwargs, overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        _model(tiny_kwargs, **overrides)


@pytest.mark.parametrize(
    "overrides",
    [dict(d_z=9), dict(d_head=16), dict(encoder_num_heads=5), dict(target_attention_blocks=0)],
    ids=["indivisible-latent", "head-geometry", "indivisible-heads", "no-attention"],
)
def test_the_refusal_messages_are_identical_strings(tiny_kwargs, overrides) -> None:
    """Not merely "both raise": a composition that re-derived its own guards would drift in wording
    first and in behaviour later."""
    messages = []
    for cls in (SeqVaeLagAttnTrfRws, SeqVaeLagAttnTrfCrws):
        with pytest.raises(ValueError) as excinfo:
            _model(tiny_kwargs, cls=cls, **overrides)
        messages.append(str(excinfo.value))

    assert messages[0] == messages[1]


def test_a_warm_up_without_its_keep_index_is_refused(tiny_kwargs) -> None:
    """Unpaired, both gates stay ``None`` and the target stream's vector would route into both."""
    with pytest.raises(ValueError, match="target_keep_index"):
        _model(tiny_kwargs, target_warmup_steps=TINY_TARGET_WARMUP_STEPS)
    with pytest.raises(ValueError, match="source_keep_index"):
        _model(tiny_kwargs, source_warmup_steps=TINY_SOURCE_WARMUP_STEPS)


def test_a_stride_outside_the_horizon_is_refused_naming_it(tiny_warmup) -> None:
    """Above $H$ the decoded windows leave gaps no phase ever covers; below $1$ there is no set."""
    horizon = int(tiny_warmup["horizon"])
    for stride in (0, -1, horizon + 1):
        with pytest.raises(ValueError, match=str(stride)):
            _model(tiny_warmup, anchor_stride=stride)
    assert build(dict(tiny_warmup, anchor_stride=horizon)) is not None


def test_a_negative_lag_floor_is_refused_naming_it(tiny_warmup) -> None:
    """It would admit lags reading before the start of the sequence."""
    with pytest.raises(ValueError, match="-3"):
        _model(tiny_warmup, lag_floor=-3)


def test_a_floor_below_the_kept_channels_warm_up_is_refused(tiny_warmup) -> None:
    r"""$F \ge B - 1$, the declared input-warmth policy, enforced at construction rather than
    assumed -- and it is the conv-LSTM cell's own check, so the two cannot come to disagree."""
    with pytest.raises(ValueError, match="warm by the first forecast step"):
        _model(tiny_warmup, warmup_period=1)


def test_the_causal_refusal_messages_are_the_conv_lstm_cells(tiny_warmup) -> None:
    """The two cells share one input domain, so they must share its refusals *verbatim* -- a message
    that had drifted would be the first sign the two are no longer one design."""
    conv_lstm_kwargs = dict(tiny_warmup, lstm_layers=1)
    for key in _ENCODER_KEYS:
        conv_lstm_kwargs.pop(key)

    messages = []
    for cls, kwargs in (
        (SeqVaeLagAttnTrfCrws, tiny_warmup),
        (SeqVaeLagAttnCrws, conv_lstm_kwargs),
    ):
        with pytest.raises(ValueError) as excinfo:
            _model(kwargs, cls=cls, warmup_period=1)
        messages.append(str(excinfo.value))

    assert messages[0] == messages[1]
