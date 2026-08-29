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
    TINY_TARGET_ALIGN_DELAYS,
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

#: This model's measured totals at the shipped geometry, with and without the resolved guard. They
#: differ for exactly one KIND of reason, unlike in the causal-feature pair: everything the guard
#: adds or removes is under an input adapter. The decoder is $R = 16$ wide either way, because the
#: raw block's width is geometry rather than a gate's survivor count.
_SHIPPED_GATED = 4_989_804
_SHIPPED_UNGATED = 4_995_052

#: The same model with the alignment off, which is the named comparison arm one key away at
#: ``causal_align_reference: null``. The $-768$ between the two factorises exactly: the source
#: adapter loses four channels from two $128$-wide linears, and both adapters gain a start vector.
_SHIPPED_GATED_UNALIGNED = 5_013_612

#: The conv-LSTM cell of this row at the same guard, and the difference. The encoder swap costs
#: exactly what it costs in the causal-feature pair and in the two-sided pair, which is the sense in
#: which the grid's two axes are independent.
_CONV_LSTM_SHIPPED_GATED = 5_081_146
_ENCODER_EDGE_PARAMETERS = 91_342

#: What the shipped budget keeps of the $102$ declared target-stream channels. A literal here rather
#: than a resolved count, because the wrong-composition guard below is about a width that must *not*
#: appear on this cell's decoder, and deriving it from the same resolution the model uses would let
#: both move together.
_KEPT_TARGET_CHANNELS = 38


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

    Removing the two delay keywords is the point: a warm-up routed under a delay name would reach
    ``ChannelDelay``, which shifts rather than masks, and would train a different model with every
    shape intact.

    The third removal is a different kind of decision and is stated as one. ``persistence_residual``
    adds a term in the **target's** own stored value to the decoder mean; this row's target is the
    raw signal, so there is no stored coefficient to persist and the mechanism is declined rather
    than defaulted off. Absence from the signature is what makes it unreachable: the driver's
    signature sweep cannot forward a key that is not here, so a config carrying it is refused by
    name instead of training a model its own config does not describe.
    """
    parameters = inspect.signature(SeqVaeLagAttnTrfCrws.__init__).parameters
    base = inspect.signature(SeqVaeLagAttnTrfRws.__init__).parameters

    assert set(base) - set(parameters) == {
        "target_delays",
        "source_delays",
        "persistence_residual",
    }
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
    }
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def test_the_forwarded_set_is_this_signature_minus_the_mixins_own_keywords() -> None:
    """Captured from ``locals()`` rather than written out a second time.

    Forty explicit ``name=name`` pairs would be the same dict with one silent failure mode: a
    keyword added to the architecture parent and forgotten here would be forwarded at its default,
    with nothing raising and no shape differing.

    The right-hand union names the three the parent has and this row does not: the two delay
    keywords the warm-ups replace, and the target-only persistence residual this row declines.
    """
    parameters = set(inspect.signature(SeqVaeLagAttnTrfCrws.__init__).parameters)
    base = set(inspect.signature(SeqVaeLagAttnTrfRws.__init__).parameters)

    forwarded = parameters - set(FORWARDED_EXCLUSIONS)
    assert (
        forwarded | {"self", "target_delays", "source_delays", "persistence_residual"} == base
    )

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
    assert (shipped.geometry.raw_len, shipped.geometry.t_valid) == (4800, 270)


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
    unaligned = _model(shipped_warmup_kwargs(align=False))

    assert _n_parameters(gated) == _SHIPPED_GATED
    assert _n_parameters(ungated) == _SHIPPED_UNGATED
    assert gated.decoder_out_channels == ungated.decoder_out_channels == 16

    gated_names = {name for name, _ in gated.named_parameters()}
    ungated_names = {name for name, _ in ungated.named_parameters()}
    added = gated_names - ungated_names
    assert added, "the guarded model added no parameter at all"
    assert all("adapter" in name for name in added), sorted(added)
    assert ungated_names - gated_names == set()

    # The alignment arm, measured rather than asserted absent. At this row's 42.21 s reference the
    # cut is large on BOTH streams -- every channel slower than the reference goes, which takes the
    # whole second stored source block -- so the sign is negative: the aligned model is the smaller
    # one. Each dropped channel costs two projections at d_model, and both start embeddings are
    # built because the shifted minimum is positive. Derived from the gates rather than written
    # out, so a reference change moves the assertion with the model.
    assert _n_parameters(unaligned) == _SHIPPED_GATED_UNALIGNED

    target_narrowing = gated.target_gate.out_channels - unaligned.target_gate.out_channels
    source_narrowing = gated.source_gate.out_channels - unaligned.source_gate.out_channels
    assert (target_narrowing, source_narrowing) == (-60, -34)
    assert (
        _n_parameters(gated) - _n_parameters(unaligned)
        == (target_narrowing + source_narrowing) * 128 * 2 + 2 * 128
        == -23_808
    )
    assert gated.source_gate.out_channels - source_narrowing == unaligned.source_gate.out_channels


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

    ``gate.delay.delay_steps`` carries the alignment shifts $d_c$ and nothing about the warm-up:
    all zeros under a pure gather, which is what the unaligned ``tiny_warmup`` fixture here builds,
    and a vector unrelated to $W'_c$ under the shipped ``causal_align_reference``. Either way the
    architecture's own version would build the guard from those -- ``max_delay = 0`` on this
    fixture, so no availability buffer, no mask projection, and a leading region of real-valued
    pre-recording history entering the encoder as though it were signal.
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
    """Two, not the causal-feature cell's four: ``warm_tertile_id`` and ``novelty_tertile_id`` both
    partition kept *target* channels and this target has none, so both must be absent rather than
    empty.

    Non-persistent so a checkpoint trained at one budget fails to load at another as a budget
    mismatch rather than as misaligned keys.
    """
    model = _model(tiny_warmup)

    assert model.source_block_warm_st.shape == (model.sequence_length,)
    assert model.source_block_warm_ph.shape == (model.sequence_length,)
    assert not hasattr(model, "warm_tertile_id")
    assert not hasattr(model, "novelty_tertile_id")
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


#: This cell's own class, bound once: half of the block below asserts a property of the *signature*
#: and needs no model at all, and the other half builds through the package's own ``build`` helper
#: so two builds differ only where a keyword made them.
_CELL_CLS = SeqVaeLagAttnTrfCrws


# =================================================================================================
# The revision's switches: the four this row takes, and the one it declines
#
# Pinned per cell rather than once on the parent, because the failure this catches is *this* cell's:
# the driver builds a run's kwargs by sweeping the constructor's signature and silently drops any
# key the class does not re-list, so a switch threaded through the parent and forgotten here would
# train the baseline under the arm's name with no error and no metric saying so.
#
# The decline is the mirror image and is asserted the same way. ``persistence_residual`` adds a term
# in the TARGET's own stored coefficient to the decoder mean; this row's target is the raw signal,
# whose per-step samples are a different object, so the mechanism is refused at the constructor
# rather than shipped off. Absence from the signature is what makes it unreachable -- the sweep
# cannot forward a key that is not there.
# =================================================================================================
#: The four keywords the revision added to this constructor, at their off-values. Written out
#: rather than derived from the signature's defaults: comparing the defaults against themselves
#: would pass on any edit, and what has to hold is that these particular values reproduce the
#: pre-revision model.
_SWITCHES_OFF = dict(
    lag_kv_source="encoder",
    prior_availability_input=False,
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
        for name, parameter in inspect.signature(_CELL_CLS.__init__).parameters.items()
        if name in _SWITCHES_OFF
    }

    assert defaults == _SWITCHES_OFF


def test_the_persistence_residual_is_declined_rather_than_defaulted_off(tiny_warmup) -> None:
    """The decline, asserted where it is enforced. The keyword is not on this signature at all, so
    a config carrying it is refused by name instead of training a model its own config does not
    describe -- and the parent's own guard refuses the mechanism on a raw-target architecture even
    if the keyword were reached some other way."""
    assert "persistence_residual" not in inspect.signature(_CELL_CLS.__init__).parameters

    with pytest.raises(TypeError, match="persistence_residual"):
        build(dict(tiny_warmup, persistence_residual=True))


def test_every_switch_at_its_off_value_is_bitwise_the_model_without_the_keywords(
    tiny_warmup,
) -> None:
    """The whole off-state claim in one comparison, over the state dict and the buffer names.

    Values as well as keys, because a switch that added a zero-initialised parameter would leave
    the totals standing and change the object; and buffer names as well as parameters, because a
    non-persistent buffer is invisible to a ``state_dict`` comparison and is exactly how the
    horizon weight and the availability announcement are carried.
    """
    without = build(tiny_warmup)
    explicit = build(dict(tiny_warmup, **_SWITCHES_OFF))

    assert list(without.state_dict()) == list(explicit.state_dict())
    for name, tensor in without.state_dict().items():
        assert torch.equal(tensor, explicit.state_dict()[name]), name
    assert sorted(dict(without.named_buffers())) == sorted(dict(explicit.named_buffers()))


def test_the_prior_clock_builds_no_parameter_when_it_is_off(tiny_warmup) -> None:
    """Absent rather than present-and-zero, and the difference is a distributed run's: a parameter
    built and left inert has no gradient path, which is what ``find_unused_parameters=False``
    refuses. Both directions, so the absence is not the absence of a working mechanism."""
    off = build(dict(tiny_warmup, prior_availability_input=False))
    on = build(dict(tiny_warmup, prior_availability_input=True))

    assert "prior_head.clock_proj.weight" not in dict(off.named_parameters())
    assert "prior_head.clock_proj.weight" in dict(on.named_parameters())


def test_the_horizon_weight_is_a_non_persistent_buffer_or_nothing(tiny_warmup) -> None:
    r"""Null builds no buffer; a half-life builds one that a checkpoint does not carry.

    Non-persistent is the load-bearing half. The weight is $(H,)$, so a persistent one would put
    the horizon into the state dict and make a checkpoint unloadable at any other horizon -- for a
    tensor that is a pure function of two numbers the constructor already has.
    """
    off = build(dict(tiny_warmup, horizon_weight_halflife_steps=None))
    on = build(dict(tiny_warmup, horizon_weight_halflife_steps=5.0))

    assert "horizon_weight" not in dict(off.named_buffers())
    assert "horizon_weight" in dict(on.named_buffers())
    assert on.horizon_weight.shape == (on.horizon,)
    assert float(on.horizon_weight.sum()) == pytest.approx(float(on.horizon), rel=1e-6)
    assert not any("horizon_weight" in name for name in on.state_dict())


def test_an_unknown_kv_source_is_refused_naming_the_choices(tiny_warmup) -> None:
    """By name, with the admitted set in the message. The value reaches a branch that would
    otherwise fall through to one of the arms, so an unrecognised string would silently train the
    fall-through arm under the misspelt one's name."""
    with pytest.raises(ValueError, match=r"lag_kv_source must be one of"):
        build(dict(tiny_warmup, lag_kv_source="conv-stem"))


@pytest.mark.parametrize("arm", ["conv_stem", "adapter"])
def test_a_local_kv_arm_does_not_build_the_deep_source_encoder(tiny_warmup, arm) -> None:
    """The deep source encoder leaves the *model*, not just the lag path: nothing else consumes the
    source state, so under a local arm it would be a whole stack of parameters no forward reaches.
    Asserted by state-dict prefix rather than by a total, because a total cannot say which stack
    went."""
    deep = build(dict(tiny_warmup, lag_kv_source="encoder"))
    local = build(dict(tiny_warmup, lag_kv_source=arm))

    assert [name for name in deep.state_dict() if name.startswith("source_encoder.")]
    assert [name for name in local.state_dict() if name.startswith("source_encoder.")] == []
    assert getattr(local, "source_encoder", None) is None


@pytest.mark.parametrize("arm", ["encoder", "conv_stem", "adapter"])
def test_the_kv_representation_is_the_pathway_composed_in_order(tiny_warmup, arm) -> None:
    """``encode_source_kv`` is what the forward and both controls call, so what it computes has to
    be exactly the pathway's modules in order -- a helper that dropped or reordered one would move
    the keys, the values, the null control's re-encode and the prior's clock together, and every
    one of them would still be the right shape."""
    model = build(dict(tiny_warmup, lag_kv_source=arm)).eval()
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
