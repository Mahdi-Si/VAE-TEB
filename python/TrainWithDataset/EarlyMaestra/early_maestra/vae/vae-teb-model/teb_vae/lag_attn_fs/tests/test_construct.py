r"""Construction: what the subclass changes, what it inherits, and what it still refuses.

A subclass is only worth having if it is provably one. The tests here therefore split in two.
The first half pins the *difference* -- the decoder emits one value per surviving target channel,
at every budget and at none -- against the parameter counts it implies. The second half pins the
*sameness*: the sibling's construction-time refusals fire here with the sibling's messages, the
subclass's own callables are two named methods and nothing else, and every structural guarantee
the sibling asserts on its assembled model still holds on this one.

The parameter counts are written out because they are the whole claim. The decoder's two heads
are $\mathrm{Linear}(128, X)$, so widening $X$ costs $2 \cdot 129 = 258$ parameters per channel
and nothing anywhere else in the model may move.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from teb_vae.lag_attn.nets.delays import ChannelGate
from teb_vae.lag_attn.nets.heads import PriorHead
from teb_vae.lag_attn_fs.nets.feature_target import FeatureForecastTarget
from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.tests.conftest import (
    SHIPPED_KWARGS,
    TINY_KEEP_INDEX,
    TINY_KWARGS,
    shipped_gated_kwargs,
    tiny_gated_kwargs,
)
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

#: Parameters the decoder's two log-variance/mean heads cost per output channel:
#: $\mathrm{Linear}(d_{\mathrm{hidden}} = 256, X)$ twice, weights and biases.
_PARAMS_PER_CHANNEL = 2 * (256 + 1)

#: The raw model's measured parameter count at the shipped widths **with no reach budget**, as
#: ``lag_attn_rws/DESIGN.md`` records it, and with the shipped $120$ s budget. They differ: a
#: guarded run builds availability input adapters, which the unguarded one does not have.
_RWS_SHIPPED_UNGATED = 5_088_186
_RWS_SHIPPED_GATED = 5_094_458

#: Surviving target channels at the shipped budget, and the declared width.
_KEPT_CHANNELS = 78
_ALL_CHANNELS = 109


def _model(kwargs, cls=SeqVaeLagAttnFs, **overrides):
    torch.manual_seed(0)
    return cls(**dict(kwargs, **overrides))


# ---------------------------------------------------------------------------------------
# The width, and what it costs
# ---------------------------------------------------------------------------------------
def test_the_model_constructs_at_the_tiny_geometry(tiny_kwargs):
    model = _model(tiny_kwargs)
    assert model.geometry.raw_len == 256
    assert model.geometry.t_valid == 12


def test_the_model_constructs_at_the_production_geometry(shipped_gated):
    model = _model(shipped_gated)
    assert model.geometry.raw_len == 4800
    assert model.geometry.t_valid == 270
    assert model.n_causalized_norms > 0


def test_the_decoder_width_is_the_surviving_channel_count(shipped_gated):
    """78 at the shipped budget. Not a configuration key: the width follows the gate, so a run
    cannot decode a width its target does not have."""
    model = _model(shipped_gated)

    assert model.target_gate.out_channels == _KEPT_CHANNELS
    assert model.decoder_out_channels == _KEPT_CHANNELS
    assert model.decoder.out_channels == _KEPT_CHANNELS


def test_without_a_budget_the_decoder_width_is_the_declared_width(shipped_kwargs):
    """109 at ``causal_reach_budget_s: null``. The unguarded arm is a configuration rather than
    an unhandled case, and its block cardinality is $H \\cdot c_y$."""
    model = _model(shipped_kwargs)

    assert model.target_gate is None
    assert model.decoder_out_channels == _ALL_CHANNELS == model.c_y
    assert model.decoder.out_channels == _ALL_CHANNELS


def test_the_width_follows_a_gate_of_any_size(tiny_gated):
    """Three survivors, three output channels: the rule is the gate's count, not a constant."""
    model = _model(tiny_gated)

    assert model.decoder_out_channels == len(TINY_KEEP_INDEX) == 3
    assert model.raw_per_step == 16, "the raw grid is geometry, not the decoder width"


def test_the_decoder_delta_against_the_raw_model_is_exactly_the_head_cost(shipped_gated):
    """The one structural difference, priced. Widening the decoder from $R = 16$ to $78$ costs
    $514 \\times 62 = 31{,}868$ parameters and nothing else in the model may move."""
    raw = _model(shipped_gated, cls=SeqVaeLagAttnRws)
    feature = _model(shipped_gated)

    raw_count = sum(p.numel() for p in raw.parameters())
    feature_count = sum(p.numel() for p in feature.parameters())

    assert raw_count == _RWS_SHIPPED_GATED
    assert feature_count - raw_count == _PARAMS_PER_CHANNEL * (_KEPT_CHANNELS - 16) == 31_868
    assert feature_count == 5_126_326


def test_the_ungated_arm_costs_the_full_declared_width(shipped_kwargs):
    """$514 \\times 93 = 47{,}802$ at $c_y = 109$, against the count ``DESIGN.md`` records -- which
    is the **unguarded** model: a guarded run adds the availability adapters' parameters on top."""
    raw = _model(shipped_kwargs, cls=SeqVaeLagAttnRws)
    feature = _model(shipped_kwargs)

    raw_count = sum(p.numel() for p in raw.parameters())
    assert raw_count == _RWS_SHIPPED_UNGATED
    assert (
        sum(p.numel() for p in feature.parameters()) - raw_count
        == _PARAMS_PER_CHANNEL * (_ALL_CHANNELS - 16)
        == 47_802
    )


def test_only_the_decoder_heads_change_shape_against_the_raw_model(shipped_gated):
    """The structural claim, stated as shapes rather than values.

    Every parameter tensor has the same name in both models and the same shape in all but the
    decoder's two output heads. Values are *not* comparable and must not be asserted to be: the
    heads are drawn at a different size, so the shared RNG stream shifts from that draw onward
    and every parameter ``initialization`` reaches afterwards differs. That is the unavoidable
    consequence of changing a tensor's size, not a second difference -- and a test written the
    other way would have to be deleted the first time it ran.
    """
    raw = dict(_model(shipped_gated, cls=SeqVaeLagAttnRws).named_parameters())
    feature = dict(_model(shipped_gated).named_parameters())

    assert set(raw) == set(feature)
    reshaped = sorted(name for name in raw if raw[name].shape != feature[name].shape)
    assert reshaped == ["decoder.logvar_head.bias", "decoder.logvar_head.weight",
                        "decoder.mean_head.bias", "decoder.mean_head.weight"], reshaped
    for name in reshaped:
        assert feature[name].shape[0] == _KEPT_CHANNELS
        assert raw[name].shape[0] == SHIPPED_KWARGS["raw_per_step"]


# ---------------------------------------------------------------------------------------
# What the subclass is
# ---------------------------------------------------------------------------------------
def test_the_class_itself_defines_nothing_at_all():
    """Set equality rather than a line count, and the set is **empty**.

    The four callables and the block-split constant are the *target domain*, and they live in the
    mixin listed ahead of the base rather than in this class body -- so what this class is, is a
    composition. An empty ``vars`` is the cheapest guarantee the suite has: with nothing defined
    here, the twenty forward keys, the posterior structure, the lag map and the objective's metric
    set cannot have moved, because they are the base's own code objects. Anything appearing here --
    a ``forward``, an ``__init__``, a second decoder path -- would mean the two models had stopped
    being the same architecture over two target domains.
    """
    own = {
        name
        for name, value in vars(SeqVaeLagAttnFs).items()
        if callable(value) and not isinstance(value, type)
    }

    assert own == set()
    assert vars(SeqVaeLagAttnFs).keys() <= {"__module__", "__qualname__", "__doc__",
                                            "__firstlineno__", "__static_attributes__"}


def test_the_target_domain_arrives_from_the_mixin_ahead_of_the_base():
    """Where the five members went, and in which order they are reached.

    ``compute_loss`` is the override the design names; ``_default_decoder_out_channels`` names a
    number and builds nothing, ``_build_forecast_target`` is a new method rather than an override
    of one, and ``_resolved_forecast_gaps`` reports four partial sums of a number the objective
    already returns. The MRO is asserted as a list because the *order* is what makes the width hook
    resolve to the feature one: reversed, the decoder would be built at ``raw_per_step`` and a
    feature block scored against it, with nothing raising.
    """
    own = {
        name
        for name, value in vars(FeatureForecastTarget).items()
        if callable(value) and not isinstance(value, type)
    }

    assert own == {
        "_default_decoder_out_channels",
        "_build_forecast_target",
        "_resolved_forecast_gaps",
        "compute_loss",
    }
    assert [cls.__name__ for cls in SeqVaeLagAttnFs.__mro__] == [
        "SeqVaeLagAttnFs",
        "FeatureForecastTarget",
        "SeqVaeLagAttnRws",
        "Module",
        "object",
    ]
    for name in own:
        assert getattr(SeqVaeLagAttnFs, name) is getattr(FeatureForecastTarget, name)


def test_the_block_split_constant_moved_with_the_methods():
    """The constant the callable filter above cannot see.

    ``TARGET_BLOCK_SPLIT`` is an ``int``, so a copy left behind on this class would be invisible to
    every set-equality assertion here and would surface later as a divergence between two models
    that are supposed to share one split.
    """
    assert SeqVaeLagAttnFs.TARGET_BLOCK_SPLIT == FeatureForecastTarget.TARGET_BLOCK_SPLIT == 43
    assert "TARGET_BLOCK_SPLIT" not in vars(SeqVaeLagAttnFs)
    assert "TARGET_BLOCK_SPLIT" in vars(FeatureForecastTarget)


def test_the_mixin_is_a_plain_object_and_not_a_framework():
    """It is a move, not an abstraction: no base class, no ``__init__``, nothing to construct.

    An ``__init__`` here is the specific failure worth naming. The trainer builds a run's kwargs by
    sweeping ``MODEL_CLS.__init__`` with ``inspect.signature``; a mixin that defined one would win
    method resolution, narrow the swept signature and make the sweep forward no configuration at
    all -- an all-defaults model, silently.
    """
    assert FeatureForecastTarget.__bases__ == (object,)
    assert "__init__" not in vars(FeatureForecastTarget)
    assert FeatureForecastTarget.__init__ is object.__init__
    assert not hasattr(FeatureForecastTarget, "__init_subclass__") or (
        "__init_subclass__" not in vars(FeatureForecastTarget)
    )
    assert not getattr(FeatureForecastTarget, "__abstractmethods__", None)


def test_the_constructor_signature_is_the_siblings():
    """The trainer builds its kwargs from an ``inspect.signature`` sweep of ``MODEL_CLS.__init__``.
    A narrowed signature would forward no configuration at all and silently build an all-defaults
    model -- no error, no shape mismatch, a run at the wrong widths."""
    import inspect

    assert (
        inspect.signature(SeqVaeLagAttnFs.__init__)
        == inspect.signature(SeqVaeLagAttnRws.__init__)
    )
    assert "decoder_out_channels" in inspect.signature(SeqVaeLagAttnFs.__init__).parameters


def test_an_explicit_width_still_wins_over_the_default(tiny_gated):
    """The keyword is not removed, only defaulted differently. A sweep arm that wants a width the
    gate does not imply can still say so, and gets it."""
    model = _model(tiny_gated, decoder_out_channels=7)

    assert model.decoder_out_channels == 7
    assert model.target_gate.out_channels == 3


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
    ],
    ids=["indivisible-latent", "head-geometry", "negative-lag", "zero-c_y", "zero-c_u",
         "degenerate-geometry"],
)
def test_the_geometry_refusals_are_the_siblings(tiny_kwargs, overrides, match):
    with pytest.raises(ValueError, match=match):
        _model(tiny_kwargs, **overrides)


def test_the_refusal_messages_are_identical_strings(tiny_kwargs):
    """Not merely "both raise": a subclass that re-derived its own guards would drift in wording
    first and in behaviour later."""
    messages = []
    for cls in (SeqVaeLagAttnRws, SeqVaeLagAttnFs):
        with pytest.raises(ValueError) as excinfo:
            _model(tiny_kwargs, cls=cls, d_z=9)
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
    """Sharper here than in the sibling: the gate now decides the decoder's width, so a gate that
    built anyway would put the decoder on a width the target does not have."""
    with pytest.raises(ValueError, match=match):
        _model(tiny_kwargs, target_keep_index=keep, target_delays=delays)


# ---------------------------------------------------------------------------------------
# The inherited structure
# ---------------------------------------------------------------------------------------
def test_no_decoder_state_head_and_no_second_decoder_exist(tiny_kwargs):
    model = _model(tiny_kwargs)
    assert not hasattr(model, "residual_decoder")
    assert not hasattr(model, "baseline_decoder")
    assert not any(isinstance(m, PriorHead) for m in model.modules())
    assert not hasattr(model.prior_head, "decoder_state_head")


def test_the_posterior_is_head_structured(tiny_kwargs):
    assert _model(tiny_kwargs).posterior_head.head_structured is True


def test_the_delta_heads_are_zero_on_the_assembled_model(tiny_kwargs):
    """Asserted on the assembled model: the generic initialization xavier-fills every linear
    layer, so only the zeroed-after ordering makes this true -- and the width override must not
    have moved the decoder past it."""
    model = _model(tiny_kwargs)
    for name in ("delta_mu_head", "delta_logvar_head"):
        module = getattr(model.posterior_head, name)
        layers = list(module) if isinstance(module, nn.ModuleList) else [module]
        for layer in layers:
            assert layer.weight.abs().max().item() == 0.0, f"{name} weight not zeroed"


def test_the_output_head_calibration_reaches_every_target_channel(shipped_gated):
    """``head_init_calibration`` centres the log-variance head at $\\sigma = 1$. An uncalibrated
    tail would put the init NLL of those channels orders of magnitude above the trivial
    predictor's -- and at $78$ channels there is far more tail to get wrong than at $16$."""
    import math

    bias = _model(shipped_gated).decoder.logvar_head.bias

    assert bias.numel() == _KEPT_CHANNELS
    assert torch.allclose(bias, torch.full_like(bias, math.log(5.0 / 3.0)))


def test_the_attention_output_projection_is_frozen(tiny_kwargs):
    attn = _model(tiny_kwargs).lag_attn
    assert attn.W_o.weight.requires_grad is False
    assert attn.W_o.bias.requires_grad is False


def test_the_gate_buffers_stay_out_of_the_state_dict(tiny_gated):
    """Load-bearing here in a way it is not for the sibling: the keep-index's *length* is now the
    decoder's width, so a persistent copy would make a checkpoint trained at one budget fail to
    load at another as "keys did not align" rather than as a message about the budget."""
    model = _model(tiny_gated)

    assert not [name for name in model.state_dict() if "keep_index" in name]
    assert not [name for name in model.state_dict() if "delay_steps" in name]


def test_an_unguarded_model_has_no_gather_and_no_delay(tiny_kwargs):
    model = _model(tiny_kwargs)

    assert model.target_gate is None and model.source_gate is None
    assert not any(isinstance(m, ChannelGate) for m in model.modules())
    assert model.source_delay_steps == 0


def test_the_adapters_are_built_for_the_surviving_widths(tiny_gated):
    """The model still declares the full ``c_y`` / ``c_u`` -- the target builder's keep-index is
    positional into ``c_y`` -- while the adapters see only the survivors."""
    model = _model(tiny_gated)

    assert (model.c_y, model.c_u) == (TINY_KWARGS["c_y"], TINY_KWARGS["c_u"])
    assert model.target_adapter.linear.in_features == 3
    assert model.source_adapter.linear.in_features == 2


def test_the_shipped_widths_are_the_production_ones(shipped_gated):
    """A guard on the fixture rather than on the model: every count in this file is read against
    $c_y = 109$ and $H_d = 30$, and a changed production geometry must fail here first."""
    assert shipped_gated["c_y"] == _ALL_CHANNELS
    assert shipped_gated["horizon"] == 30
    assert len(shipped_gated["target_keep_index"]) == _KEPT_CHANNELS
    assert shipped_gated_kwargs(None).get("target_keep_index") is None
    assert tiny_gated_kwargs()["target_keep_index"] == TINY_KEEP_INDEX
    assert SHIPPED_KWARGS["horizon"] * _KEPT_CHANNELS == 2340
