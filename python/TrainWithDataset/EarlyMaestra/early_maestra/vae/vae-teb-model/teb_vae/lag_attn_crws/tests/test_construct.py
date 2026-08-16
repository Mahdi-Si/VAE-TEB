r"""The model is built, its inputs begin where the warm-up says they do, and its decoder is raw.

This cell composes one mixin over the raw-signal architecture, and the two ways that goes wrong are
opposite in kind.

**Too little.** A member that should be inherited is written here instead, and the conv-Transformer
twin composing the same mixin silently does not get it. ``vars()`` is asserted to be exactly
``{'__init__'}`` for that reason, and the constructor is the one member that genuinely cannot be
shared: the experiment driver builds a run's kwargs by sweeping
``inspect.signature(MODEL_CLS.__init__)``, so a ``**kwargs`` signature would forward four keys and
build an all-defaults model at $d_{model} = 128$ on a tiny smoke config.

**Too much.** The causal-feature target mixin is composed in as well -- which is what the design's
first draft did, and it reads perfectly plausibly. Then ``_default_decoder_out_channels`` resolves to
the *feature* one, the decoder emits $C_{\mathrm{keep}} = 98$ channels per horizon token, and the
target is $(B, A, H, 16)$. ``block_width`` does not catch it: it feeds only the four log-variance
diagnostics and no shape check. The first symptom is ``raw_sample_score`` computing
$(\text{target} - \mu)^2$ on shapes that do not broadcast, three frames below the class body that
caused it. So the wrong composition is built here and its decoder width asserted, and the reason the
mixin is excluded becomes a passing test rather than a comment.

The warm-up half is the causal-feature cell's machinery reached by identity -- ``_build_adapter`` is
asserted to be that mixin's own function object in ``test_causal_raw_inputs.py`` -- so what is
checked here is that it was *reached*: the adapter is built at the warm-up rather than at the gate's
delays, which are all zero under a pure gather, and no gradient flows from inside the masked region.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
from teb_vae.lag_attn_cfs.nets.causal_inputs import (
    FORWARDED_EXCLUSIONS,
    CausalWarmupInputs,
)
from teb_vae.lag_attn_crws.nets.causal_raw_inputs import CausalRawInputs
from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

from .conftest import (
    CAUSAL_C_U,
    CAUSAL_C_Y,
    CAUSAL_ST_WIDTH,
    SHIPPED_HORIZON,
    SHIPPED_WARMUP_PERIOD,
    TINY_SOURCE_WARMUP_STEPS,
    TINY_TARGET_KEEP_INDEX,
    TINY_TARGET_WARMUP_STEPS,
    build,
    make_streams,
    shipped_warmup_kwargs,
)

#: What the shipped budget keeps of the $102$ declared target-stream channels. A literal here rather
#: than a resolved count, because the wrong-composition guard below is about a width that must *not*
#: appear on this cell's decoder, and deriving it from the same resolution the model uses would let
#: both move together.
_KEPT_TARGET_CHANNELS = 98


# =================================================================================================
# The class
# =================================================================================================
def test_the_mixin_comes_first_and_the_mro_is_pinned() -> None:
    """As a list of names, because the *order* is what makes the tiled forward, the warm-up adapter
    and the floored lag mask resolve to the causal ones.

    Reversed, the architecture's dense forward wins: it decodes $[0, T_{\\mathrm{valid}})$, returns
    no anchor set at all, and the objective then gathers a dense target for it -- a different model
    with every shape self-consistent.
    """
    assert SeqVaeLagAttnCrws.__bases__ == (CausalRawInputs, SeqVaeLagAttnRws)
    assert [cls.__name__ for cls in SeqVaeLagAttnCrws.__mro__] == [
        "SeqVaeLagAttnCrws",
        "CausalRawInputs",
        "CausalWarmupInputs",
        "SeqVaeLagAttnRws",
        "Module",
        "object",
    ]


def test_the_model_carries_nothing_but_its_constructor() -> None:
    """Everything else is encoder-agnostic and lives on the mixin the second cell composes too."""
    assert {name for name in vars(SeqVaeLagAttnCrws) if not name.startswith("__")} == set()
    assert "__init__" in vars(SeqVaeLagAttnCrws)

    for shared in ("forward", "_build_anchor_index", "_build_adapter", "build_lag_mask"):
        assert shared not in vars(SeqVaeLagAttnCrws), shared
        assert shared in vars(CausalWarmupInputs), shared
    for shared in ("compute_loss", "_check_anchor_floor"):
        assert shared not in vars(SeqVaeLagAttnCrws), shared
        assert shared in vars(CausalRawInputs), shared


def test_the_constructor_takes_warm_ups_and_refuses_delays() -> None:
    """The names are the whole guard: a warm-up routed under a delay name trains a different model.

    The full parameter list is asserted rather than only the four new names, because the failure
    that matters is the *opposite* one -- a narrowed signature. The driver builds a run's kwargs by
    sweeping this signature.
    """
    parameters = inspect.signature(SeqVaeLagAttnCrws.__init__).parameters
    base = inspect.signature(SeqVaeLagAttnRws.__init__).parameters

    assert "target_warmup_steps" in parameters and "source_warmup_steps" in parameters
    assert "anchor_stride" in parameters and "lag_floor" in parameters
    for banned in ("target_delays", "source_delays"):
        assert banned not in parameters, banned
        assert banned in base, f"{banned} is meant to be the base's, removed here"

    assert set(base) - set(parameters) == {"target_delays", "source_delays"}
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def test_the_forwarded_set_is_this_signature_minus_the_mixins_own_keywords() -> None:
    """Captured from ``locals()`` rather than written out a second time.

    Forty explicit ``name=name`` pairs would be the same dict with one silent failure mode: a
    keyword added to the base and forgotten here would be forwarded at its default, with nothing
    raising and no shape differing. The exclusion list lives on the mixin that owns those keywords,
    so a keyword removed there cannot be left behind in a filter here.
    """
    parameters = set(inspect.signature(SeqVaeLagAttnCrws.__init__).parameters)
    base = set(inspect.signature(SeqVaeLagAttnRws.__init__).parameters)

    forwarded = parameters - set(FORWARDED_EXCLUSIONS)
    assert forwarded | {"self", "target_delays", "source_delays"} == base

    source = Path(inspect.getfile(SeqVaeLagAttnCrws)).read_text(encoding="utf-8")
    assert "locals().items()" in source and "FORWARDED_EXCLUSIONS" in source


def test_a_base_keyword_no_other_test_names_still_reaches_the_base(tiny_kwargs) -> None:
    """The behavioural half of the claim above, on the two keywords a hand-written forward dict
    would be likeliest to drop: neither shapes anything, so neither would fail loudly."""
    model = build(dict(tiny_kwargs, coverage_floor=0.25, base_decode="mean"))

    assert model.coverage_floor == 0.25
    assert model.base_decode == "mean"


def test_the_geometry_defaults_are_the_causal_datasets() -> None:
    """A run left at the base's defaults would be describing a dataset that does not exist.

    None of the four is forced by this target -- a raw sample is honest at every step, so no
    validity constraint ties the floor to the resolved budget here. They are held at the
    causal-feature cells' values so that the two differ in exactly one variable.
    """
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(SeqVaeLagAttnCrws.__init__).parameters.items()
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
def test_the_decoder_emits_raw_samples_because_no_width_hook_is_defined(tiny_warmup) -> None:
    r"""The load-bearing **absence**: neither this class nor its mixin defines
    ``_default_decoder_out_channels``, so it resolves to the architecture's ``raw_per_step``."""
    model = build(tiny_warmup)

    assert model.decoder.mean_head.out_features == 16 == model.raw_per_step
    assert model.decoder_out_channels == 16
    assert "_default_decoder_out_channels" not in vars(SeqVaeLagAttnCrws)
    assert "_default_decoder_out_channels" not in vars(CausalRawInputs)
    assert (
        SeqVaeLagAttnCrws._default_decoder_out_channels
        is vars(SeqVaeLagAttnRws)["_default_decoder_out_channels"]
    )


def test_composing_the_feature_target_mixin_builds_the_wrong_decoder() -> None:
    r"""Why that absence is load-bearing, as a passing test rather than as a comment.

    The design's first draft composed three bases -- this cell's input mixin, the causal-feature
    *target* mixin, and the architecture -- and it reads plausibly: both mixins are "the causal
    half". But the second one brings a width hook, and the decoder is then built at
    $C_{\mathrm{keep}} = 98$ while the target this cell gathers is $(B, A, H, 16)$.

    Composed here as ``(CausalFeatureForecastTarget, SeqVaeLagAttnCrws)`` rather than as the draft's
    literal base tuple, for a mechanical reason worth recording: that tuple leaves ``__init__``
    resolving to the architecture's, which does not accept this domain's four keywords, so the wrong
    model could not be constructed at all and the guard would prove nothing. Subclassing this cell
    keeps its constructor and moves only the resolution order, which is exactly the mistake.
    """
    wrong_composition = type(
        "SeqVaeLagAttnCrwsWithFeatureTarget",
        (CausalFeatureForecastTarget, SeqVaeLagAttnCrws),
        {},
    )
    kwargs = shipped_warmup_kwargs()

    torch.manual_seed(0)
    wrong = wrong_composition(**dict(kwargs))
    right = build(kwargs)

    assert wrong.decoder.mean_head.out_features == _KEPT_TARGET_CHANNELS
    assert right.decoder.mean_head.out_features == 16 == right.raw_per_step
    # The cause, stated beside the symptom: which class the width hook resolved to.
    assert wrong_composition._default_decoder_out_channels is not (
        SeqVaeLagAttnCrws._default_decoder_out_channels
    )


# =================================================================================================
# The guard the constructor builds
# =================================================================================================
def test_the_source_is_gathered_whole(tiny_warmup) -> None:
    """The source keep-index is the identity, so its gate is a gather that changes nothing.

    It is still a gate rather than ``None``: the warm-up vector travels beside it and the adapter is
    built at the gate's width, so a stream with a warm-up always has one.
    """
    model = build(tiny_warmup)

    assert model.source_gate is not None
    assert model.source_gate.out_channels == CAUSAL_C_U
    assert model.source_gate.max_delay == 0, "the gate is a gather; the warm-up is not a shift"


def test_the_target_stream_gate_keeps_the_channels_the_budget_kept(tiny_warmup) -> None:
    """And the adapter is built at that width -- while the *decoder* is not, which is the whole
    difference between this cell and the causal-feature one."""
    model = build(tiny_warmup)

    assert model.target_gate is not None
    assert model.target_gate.keep_index.tolist() == list(TINY_TARGET_KEEP_INDEX)
    assert model.target_adapter.in_dim == len(TINY_TARGET_KEEP_INDEX)
    assert model.decoder_out_channels == 16 != len(TINY_TARGET_KEEP_INDEX)


def test_the_adapter_is_built_at_the_warm_up_and_not_at_the_gates_delays(tiny_warmup) -> None:
    r"""The specific failure the inherited ``_build_adapter`` exists to prevent.

    ``gate.delay.delay_steps`` is all zeros under a pure gather, so the architecture's own version
    would give ``max_delay = 0`` -- no availability buffer, no mask projection, and a leading region
    of real-valued pre-recording history entering the encoder as though it were signal.
    """
    model = build(tiny_warmup)

    assert model.target_adapter.max_delay == max(TINY_TARGET_WARMUP_STEPS)
    assert model.source_adapter.max_delay == max(TINY_SOURCE_WARMUP_STEPS)
    assert model.target_adapter.mask_proj is not None
    assert model.source_adapter.mask_proj is not None
    expected = torch.tensor(TINY_TARGET_WARMUP_STEPS)
    assert torch.equal(model.target_adapter.availability.argmax(dim=0), expected)


def test_no_gradient_flows_from_inside_the_warm_up(tiny_warmup) -> None:
    """By gradient, which is the stronger half of the masking claim.

    A value check passes on a model that happens to emit zeros in that region for some other
    reason; a zero gradient says the output is not a function of those inputs at all. The paired
    control -- the same channel past its warm-up is live -- is what makes it a statement about the
    warm-up rather than about a dead pathway.
    """
    model = build(tiny_warmup).eval()
    y_st, y_ph, u_stream = make_streams(tiny_warmup)
    y_ph = y_ph.clone().requires_grad_(True)

    torch.manual_seed(0)
    model(y_st, y_ph, u_stream)["mu_prior"].sum().backward()
    grad = y_ph.grad
    assert grad is not None

    checked = 0
    for position, declared in enumerate(TINY_TARGET_KEEP_INDEX):
        steps = TINY_TARGET_WARMUP_STEPS[position]
        if declared < CAUSAL_ST_WIDTH or steps == 0:
            continue
        channel = declared - CAUSAL_ST_WIDTH
        assert float(grad[:, :steps, channel].abs().max()) == 0.0, declared
        assert float(grad[:, steps:, channel].abs().max()) > 0.0, declared
        checked += 1
    assert checked > 0, "no kept phase-block channel had a warm-up; the probe proved nothing"


# =================================================================================================
# The ungated arm
# =================================================================================================
def test_without_a_budget_no_gate_and_no_availability_term_is_built(tiny_kwargs) -> None:
    """An unguarded run gets the model that has no guard, not an identity one."""
    model = build(tiny_kwargs)

    assert model.target_gate is None and model.source_gate is None
    assert model.target_warmup_steps is None and model.source_warmup_steps is None
    for adapter in (model.target_adapter, model.source_adapter):
        assert adapter.mask_proj is None
        assert adapter.start_embed is None
        assert not hasattr(adapter, "availability")
    assert model.decoder_out_channels == 16


def test_the_ungated_model_is_parameter_for_parameter_the_raw_signal_sibling(tiny_kwargs) -> None:
    """This class adds no parameter of its own: what it changes is which anchors are decoded.

    Compared against the raw-target architecture at the same keywords -- which is the comparison
    that is available here and is *not* available to the causal-feature cells, whose decoder is a
    different width. A parameter creeping in here (a second adapter, a learned phase, a floor
    embedding) fails rather than being absorbed into a total nobody re-derives.
    """
    causal = build(tiny_kwargs)
    torch.manual_seed(0)
    raw = SeqVaeLagAttnRws(**dict(tiny_kwargs))

    assert {name: tuple(p.shape) for name, p in causal.named_parameters()} == {
        name: tuple(p.shape) for name, p in raw.named_parameters()
    }
    assert sum(p.numel() for p in causal.parameters()) == sum(
        p.numel() for p in raw.parameters()
    )


# =================================================================================================
# The refusals the constructor inherits
# =================================================================================================
def test_a_warm_up_without_its_keep_index_is_refused(tiny_kwargs) -> None:
    """Unpaired, both gates stay ``None`` and the target stream's vector would route into both."""
    with pytest.raises(ValueError, match="target_keep_index"):
        SeqVaeLagAttnCrws(**dict(tiny_kwargs, target_warmup_steps=TINY_TARGET_WARMUP_STEPS))
    with pytest.raises(ValueError, match="source_keep_index"):
        SeqVaeLagAttnCrws(**dict(tiny_kwargs, source_warmup_steps=TINY_SOURCE_WARMUP_STEPS))


def test_a_stride_outside_the_horizon_is_refused_naming_it(tiny_warmup) -> None:
    """Above $H$ the decoded windows leave gaps no phase ever covers; below $1$ there is no set."""
    horizon = int(tiny_warmup["horizon"])
    for stride in (0, -1, horizon + 1):
        with pytest.raises(ValueError, match=str(stride)):
            SeqVaeLagAttnCrws(**dict(tiny_warmup, anchor_stride=stride))
    assert build(dict(tiny_warmup, anchor_stride=horizon)) is not None


def test_a_stride_wider_than_the_anchor_span_is_refused(tiny_warmup) -> None:
    """At the last phase the first anchor would not exist, and the sample would decode nothing."""
    model = build(tiny_warmup)
    span = model.geometry.t_valid - model.warmup_period

    with pytest.raises(ValueError, match="anchor_stride"):
        SeqVaeLagAttnCrws(
            **dict(tiny_warmup, warmup_period=model.geometry.t_valid - 1, anchor_stride=2)
        )
    assert span >= int(tiny_warmup["horizon"]), "the tiny geometry admits the shipped stride"


def test_a_negative_lag_floor_is_refused_naming_it(tiny_warmup) -> None:
    """It would admit lags reading before the start of the sequence."""
    with pytest.raises(ValueError, match="-3"):
        SeqVaeLagAttnCrws(**dict(tiny_warmup, lag_floor=-3))
