r"""Construction invariants: what exists, what is refused, what is frozen, and what it all costs.

The constructor's guarantees are structural -- two independently parameterised causal front ends
against a budget the geometry dictates, a head-structured latent, one decoder, a frozen
lag-attention output projection, three dropout sites pinned at zero, no recurrence and no
time-pooling normaliser outside the horizon core -- and each is asserted on the **assembled** model,
because several of them hold on the parts in isolation and fail silently in composition.

Two families of check are this package's rather than the sibling's.

The first is the **inert keyword set**. Eight of the sibling's constructor arguments describe stored
feature blocks and the causal reach budget that prunes them, and none of them means anything to a
model that reads raw signals. They are refused by *absence*: there is no ``**kwargs``, so a
copy-pasted sibling config raises ``TypeError`` naming the key rather than quietly building a
different model. The seven **encoder** keys are the opposite case and are asserted to still be live,
because the encoder is imported unchanged and the whole comparison rests on it staying that way.

The second is the **reach budget**. The front ends refuse a stack reaching further back than
``warmup_period * raw_per_step`` raw samples, and that guard is only connected to this model's
geometry if the model is what passes the budget. Nothing else in the package asserts that link.
"""
from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from teb_vae.lag_attn.nets.decoders import BaselineFutureDecoder
from teb_vae.lag_attn.nets.heads import PriorHead
from teb_vae.lag_attn.nets.delays import ChannelDelay, ChannelGate
from teb_vae.lag_attn_transformer_e2e.nets.frontend import (
    CausalAntiAliasDecimate,
    CausalRawFrontend,
)
from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from teb_vae.lag_attn_transformer_e2e.tests.conftest import (
    SHIPPED_KWARGS,
    TINY_FRONTEND_KERNELS,
)
from teb_vae.lag_attn_transformer_rws.nets.blocks import CausalSelfAttention
from teb_vae.lag_attn_transformer_rws.nets.encoders import CausalConvTransformerEncoder

#: The eight sibling keys that describe nothing here. Five name stored feature blocks that are not
#: loaded (``c_y``, ``c_u``, ``use_up_st``) or the reach budget that prunes them
#: (``causal_reach_budget_s``); the four channel tuples are that budget's resolved output. Each
#: would reach nothing if it were quietly accepted -- and the experiment driver builds its kwargs by
#: an ``inspect.signature`` sweep that *drops* unknown keys in silence, so the loud refusal is what
#: turns "this key does nothing" into an error rather than into a run the operator misreads.
_INERT_SIBLING_KEYS = (
    "c_y",
    "c_u",
    "use_up_st",
    "causal_reach_budget_s",
    "target_keep_index",
    "target_delays",
    "source_keep_index",
    "source_delays",
)

#: The seven encoder keys, which are emphatically **not** inert: the encoder is imported unchanged,
#: so every one of them still shapes this model exactly as it shapes the sibling. Refusing one would
#: silently make an architecture arm unrunnable here while it still ran there, which is the one
#: divergence a comparison between the two may not have.
_LIVE_ENCODER_KEYS = (
    "encoder_conv_kernels",
    "encoder_conv_dilations",
    "encoder_num_heads",
    "encoder_d_ff",
    "target_attention_blocks",
    "source_attention_blocks",
    "source_attention_window",
)

#: Parameters of the two stored-feature input adapters this architecture replaces, at the shipped
#: geometry -- $128 \times (109 + 58)$ projections plus their residual MLPs. Named rather than
#: inlined because it is the baseline the front-end cost is a *delta* against, and it is a property
#: of the model being compared with rather than of this one.
ADAPTER_PARAMETERS = 156_288

#: Recurrent and time-pooling module families that must not appear on a history path. Each would
#: make $H_t$ a function of the whole sequence, which is invisible in a loss curve and corrupts
#: exactly the quantity the model exists to measure -- and here it would additionally void the
#: raw-signal causality claim the package is built on.
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


def _model(kwargs, **overrides) -> SeqVaeLagAttnTrfE2E:
    torch.manual_seed(0)
    return SeqVaeLagAttnTrfE2E(**dict(kwargs, **overrides))


@pytest.fixture(scope="module")
def shipped_model() -> SeqVaeLagAttnTrfE2E:
    """One production-geometry model, built once for every construction-time check here."""
    return _model(SHIPPED_KWARGS)


# ---------------------------------------------------------------------------------------
# Geometry, and the two keyword sets
# ---------------------------------------------------------------------------------------
def test_the_model_constructs_at_the_tiny_geometry(tiny_kwargs):
    """Where the tiny keyword set is proven, rather than merely declared. The conftest reasons out
    its warm-up and its kernel schedule against the constructor's invariants; this is the only place
    that runs the constructor against them."""
    model = _model(tiny_kwargs)

    assert model.geometry.raw_len == 256
    assert model.geometry.t_valid == 12
    assert model.warmup_period < model.sequence_length - model.horizon
    assert model.target_frontend.kernels == TINY_FRONTEND_KERNELS


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


@pytest.mark.parametrize("key", _INERT_SIBLING_KEYS)
def test_every_inert_sibling_key_is_refused_by_name(tiny_kwargs, key):
    """Refused loudly rather than absorbed by a ``**kwargs``, which is what would let a hand-copied
    sibling config silently build a model with no guard, no declared widths and no complaint."""
    with pytest.raises(TypeError, match=key):
        SeqVaeLagAttnTrfE2E(**dict(tiny_kwargs, **{key: 1}))


@pytest.mark.parametrize("key", _LIVE_ENCODER_KEYS)
def test_every_encoder_key_is_still_live(key):
    """The paired half of the refusals above. The encoder is imported unchanged and the comparison
    depends on it staying configurable in exactly the same way, so a sweep of the ban list that
    caught an encoder key would be a silent divergence between the two packages."""
    assert key in set(inspect.signature(SeqVaeLagAttnTrfE2E.__init__).parameters)


def test_init_weights_is_a_constructor_argument(tiny_kwargs):
    """It stays in the signature -- the initialisation-order tests build an uninitialised model with
    it -- while remaining outside anything a config sets: weight initialisation is not a config
    decision."""
    assert "init_weights" in set(inspect.signature(SeqVaeLagAttnTrfE2E.__init__).parameters)
    assert _model(tiny_kwargs, init_weights=False).n_depthwise_init == 0


# ---------------------------------------------------------------------------------------
# The front ends, and the budget that bounds them
# ---------------------------------------------------------------------------------------
def test_the_model_passes_the_warmup_as_the_front_ends_reach_budget(tiny_kwargs, shipped_model):
    """The one assertion that connects the front end's construction-time refusal to this model's
    geometry. The budget is not a configuration key and not a caller's choice -- it is
    ``warmup_period * raw_per_step``, the raw-sample span of the anchors that are excluded from
    every loss anyway -- and nothing else in the package checks that the model is what supplies it.
    """
    for model in (_model(tiny_kwargs), shipped_model):
        expected = model.warmup_period * model.raw_per_step
        assert model.frontend_reach_budget == expected
        for frontend in (model.target_frontend, model.source_frontend):
            assert frontend.reach_budget == expected
            assert frontend.reach_samples <= expected


def test_a_front_end_too_wide_for_the_warm_up_is_refused_at_construction(tiny_kwargs):
    """The guard doing its job through the model, naming both numbers. At the tiny geometry the
    budget is $6 \\times 16 = 96$ raw samples and the shipped kernels reach $322$."""
    with pytest.raises(ValueError, match=r"reaches \d+ raw samples but the budget is 96"):
        _model(tiny_kwargs, frontend_kernels=(65, 15, 15, 15))


def test_both_streams_get_their_own_front_end_and_share_no_parameter(tiny_kwargs):
    """Two independently parameterised front ends at identical settings. Sharing would make the
    source state a function of the target and destroy the purity the KL readout rests on."""
    model = _model(tiny_kwargs)

    assert isinstance(model.target_frontend, CausalRawFrontend)
    assert isinstance(model.source_frontend, CausalRawFrontend)
    assert model.target_frontend is not model.source_frontend
    target_ids = {id(parameter) for parameter in model.target_frontend.parameters()}
    source_ids = {id(parameter) for parameter in model.source_frontend.parameters()}
    assert target_ids and source_ids
    assert target_ids.isdisjoint(source_ids)


def test_the_front_ends_decimate_by_exactly_the_loader_grid(tiny_kwargs):
    """Token $t$'s newest raw sample is ``raw_per_step * (t + 1) - 1``, which is the anchor
    convention the geometry already uses. A front end at any other stride would emit a
    correctly-shaped tensor on a different grid, which nothing downstream could detect."""
    model = _model(tiny_kwargs)

    for frontend in (model.target_frontend, model.source_frontend):
        assert frontend.total_stride == model.raw_per_step
        assert frontend.d_model == model.d_model


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


def test_the_decoder_takes_the_latent_and_nothing_else(tiny_kwargs):
    """The no-bypass contract, stated where it can be checked: one tensor **through the latent
    path**, at $d_z$ in-features. A decoder that also took an encoder state would let gradient
    reach the forecast without passing through the latent, and $\\mu^q - \\mu^p$ would stop being
    the source's contribution.

    Asserted by arity and by width rather than by the argument's name. That name is
    ``decoder_state``, from when the decoder was conditioned on one -- the residual decoder beside
    it still takes ``(decoder_state, z)`` -- so what distinguishes the two is how many tensors enter,
    and what the first linear is wide enough to hold.

    The shared decoder's second admitted name, ``persistence``, is a **target-only** term that
    exists for the feature-target cells alone, and this cell never builds it. So the admitted set
    is pinned by name here, and the arity that matters on *this* model -- how many tensors it can
    actually be handed -- is pinned by the weight being absent, which is what makes the decoder
    refuse the second argument outright.
    """
    model = _model(tiny_kwargs)
    parameters = [
        name
        for name, parameter in inspect.signature(BaselineFutureDecoder.forward).parameters.items()
        if name != "self" and parameter.kind is not inspect.Parameter.VAR_KEYWORD
    ]

    assert parameters == ["decoder_state", "persistence"], (
        f"the decoder takes more than the latent and the target-only residual: {parameters}"
    )
    assert model.decoder.persistence_weight is None, (
        "this cell built the persistence residual, which is a feature-target mechanism"
    )
    assert model.decoder.proj.body[0].in_features == model.d_z


def test_no_recurrence_anywhere_in_the_model(tiny_kwargs):
    """The recurrent bottleneck is the thing this architecture family removed; a stray one anywhere
    would both serialise training and reintroduce the state it exists without."""
    model = _model(tiny_kwargs)
    offenders = [
        name for name, module in model.named_modules()
        if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN))
    ]

    assert not offenders, f"recurrent modules found: {offenders}"


def test_no_time_pooling_normaliser_on_either_history_path(tiny_kwargs):
    """Scoped to the history path -- both front ends and both encoders -- because that is where a
    statistic pooled over time would make $H_t$ read its own future. The front ends carry their own
    construction-time ban; this asserts the composed path from the other side."""
    model = _model(tiny_kwargs)
    history = {
        "target_frontend": model.target_frontend,
        "source_frontend": model.source_frontend,
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
    target's future into a history state. Enumerating them means a new one anywhere else fails here
    rather than passing the scoped check above."""
    model = _model(tiny_kwargs)
    group_norms = [
        name for name, module in model.named_modules() if isinstance(module, nn.GroupNorm)
    ]

    assert group_norms, "no GroupNorm at all; this pin no longer describes the model"
    assert all(name.startswith("horizon_core.") for name in group_norms), group_norms


def test_there_is_no_channel_gate_and_no_delay(tiny_kwargs):
    """The reach budget's machinery is not merely unused here, it is absent: a raw signal read
    through a strictly one-sided front end has no acausal energy to prune or delay away."""
    model = _model(tiny_kwargs)

    assert not any(isinstance(m, (ChannelGate, ChannelDelay)) for m in model.modules())
    assert model.source_delay_steps == 0


def test_the_model_reports_a_source_delay_the_figure_can_read(tiny_kwargs):
    """Read by the diagnostic figure through a silent ``getattr(model, "source_delay_steps", 0)``,
    so a model that stopped exposing it would keep drawing the same lag axis with no error at all --
    right by coincidence today, wrong the moment anything delays a stream."""
    model = _model(tiny_kwargs)

    assert hasattr(model, "source_delay_steps")
    assert model.source_delay_steps == 0


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
    # And the two stacks that *should* have received the configured value did, so the assertions
    # above are about zeros that were chosen rather than about a model built at zero throughout.
    for stem in (model.target_encoder, model.target_frontend):
        probabilities = {
            module.p for module in stem.modules() if isinstance(module, nn.Dropout)
        }
        assert probabilities == {0.1}, probabilities


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
# Buffers
# ---------------------------------------------------------------------------------------
def test_the_geometry_shaped_buffers_stay_out_of_the_state_dict(tiny_kwargs):
    """Both are constants of the architecture rather than learned state. A persistent copy would
    make a checkpoint trained at one geometry -- or at one anti-alias tap count -- fail to load at
    another, reported as keys that did not align rather than as what it is."""
    model = _model(tiny_kwargs)
    keys = list(model.state_dict())

    assert not [name for name in keys if "future_index" in name]
    assert not [name for name in keys if name.endswith("fir")]
    # ...and they do exist as buffers, so the absence above is a choice rather than an omission.
    buffers = [name for name, _ in model.named_buffers()]
    assert "future_index" in buffers
    assert len([name for name in buffers if name.endswith("fir")]) == 8


# ---------------------------------------------------------------------------------------
# The encoders, unchanged
# ---------------------------------------------------------------------------------------
def test_the_encoders_are_the_configured_shape(shipped_model, shipped_kwargs):
    """The seven encoder keys map onto the two encoders in exactly one way: the target reads the
    full causal prefix, the source a bounded window. That asymmetry is the architecture, and it is
    the sibling's asymmetry unchanged."""
    assert isinstance(shipped_model.target_encoder, CausalConvTransformerEncoder)
    assert shipped_model.target_encoder.attention_window is None
    assert shipped_model.target_encoder.receptive_field is None
    assert (
        shipped_model.source_encoder.attention_window
        == shipped_kwargs["source_attention_window"]
    )
    assert shipped_model.source_encoder.receptive_field == 66


def test_the_encoder_head_count_does_not_touch_the_latent_grouping(tiny_kwargs, raw_inputs):
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
        out = model(*raw_inputs)
    assert out["kld_per_t_per_head"].shape[-1] == model.num_heads
    assert out["mu_prior"].shape[-1] % model.num_heads == 0


def test_a_stemless_encoder_is_a_working_module(tiny_kwargs, raw_inputs):
    """Zero convolution blocks is legal, because a stem-free architecture arm needs it. The
    depthwise count then drops to the front ends' own eight, which is what proves the two
    contributions are separable."""
    model = _model(tiny_kwargs, encoder_conv_kernels=(), encoder_conv_dilations=()).eval()

    assert len(model.target_encoder.conv_blocks) == 0
    assert model.n_depthwise_init == 8
    with torch.no_grad():
        out = model(*raw_inputs)
    assert out["target_state"].shape[-1] == model.d_model


# ---------------------------------------------------------------------------------------
# The parameter budget
# ---------------------------------------------------------------------------------------
def test_the_front_ends_are_what_the_model_pays_for_its_causality(shipped_model):
    """Recorded as the *delta* against the two stored-feature adapters they replace, not as an
    absolute total: everything else in this model is imported, so a legitimate change to a shared
    component must not fail a test here. The absolute number is pinned in one place only, against
    the design record.

    $241{,}088$ for two production front ends against $156{,}288$ for the two adapters -- the entire
    architectural cost of reading the raw signal instead of a two-sided transform of it is
    $84{,}800$ parameters, about $4\\%$ of the model.
    """
    per_stream = sum(p.numel() for p in shipped_model.target_frontend.parameters())
    both = per_stream + sum(p.numel() for p in shipped_model.source_frontend.parameters())
    total = sum(p.numel() for p in shipped_model.parameters())

    # The two streams are independently parameterised at identical settings, so their costs are
    # equal by construction; a difference would mean one was built differently.
    assert both == 2 * per_stream
    # The delta against the two adapters replaced, which is the quantity that is a fact about *this
    # architecture's choice* rather than about any shared component. The absolute total lives in the
    # design record and is checked there against sum(p.numel()), in one place.
    assert both - ADAPTER_PARAMETERS == 84_800
    assert both < 0.15 * total


def test_the_anti_alias_filters_cost_nothing(shipped_model):
    """Eight fixed filters, one per stage per stream, and not one of them is a parameter. Held as an
    ``nn.Conv1d`` instead they would be both counted and Xavier-overwritten by the generic
    initialisation, and the anti-aliasing would quietly stop happening."""
    decimators = [
        module for module in shipped_model.modules()
        if isinstance(module, CausalAntiAliasDecimate)
    ]

    assert len(decimators) == 8
    assert sum(p.numel() for module in decimators for p in module.parameters()) == 0
