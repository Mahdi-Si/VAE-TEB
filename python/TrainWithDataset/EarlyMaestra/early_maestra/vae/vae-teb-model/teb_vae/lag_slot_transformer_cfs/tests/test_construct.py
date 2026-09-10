r"""What the model builds, what it refuses to build, and what it refuses to be told.

Three structural claims are checked here and each is invisible at review time:

* **no attention module is constructed.** The design's requirement is not that a lag cross-attention
  goes unused but that it never exists, and the only mechanical proof is a walk over the built
  module tree;
* **every parameter is reachable**, including on a rank whose source is entirely unavailable, which
  is what makes a distributed run safe without unused-parameter handling;
* **a keyword naming machinery this architecture lacks raises.** The experiment driver forwards a
  configuration key only when the constructor names it, so a key omitted from the signature is
  dropped in silence -- and a run configured for windowed source attention over a model with none
  would train to completion and report its divergence as a coupling measurement of a pathway it
  never built.
"""
from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from teb_vae.lag_attn.nets.attention import LagCrossAttention
from teb_vae.lag_attn.nets.decoders import BaselineFutureDecoder, HorizonDecoderCore
from teb_vae.lag_attn.nets.heads import PosteriorHead, TEAnalysisHead
from teb_vae.lag_attn_cfs.model_kwargs import warmup_model_kwargs
from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
from teb_vae.lag_attn_cfs.nets.causal_inputs import CausalWarmupInputs
from teb_vae.lag_attn_fs.nets.feature_target import FeatureForecastTarget
from teb_vae.lag_attn_rws.nets.heads import FullLatentPriorHead
from teb_vae.lag_attn_transformer_rws.nets.encoders import (
    CausalConvTransformerEncoder,
    GatedCausalConvStem,
)
from teb_vae.lag_slot_transformer_cfs.nets.core import REFUSED_KEYWORDS, LagResidualCore
from teb_vae.lag_slot_transformer_cfs.nets.lag_updates import LagProposalHead
from teb_vae.lag_slot_transformer_cfs.nets.model import (
    MODEL_KIND,
    SeqVaeLagResidualTrfCfs,
)
from teb_vae.lag_slot_transformer_cfs.nets.pointwise_source import PointwiseSourceEncoder
from teb_vae.lag_slot_transformer_cfs.tests.conftest import (
    DECLARED_C_U,
    TINY_D_MODEL,
    TINY_D_Z,
    TINY_SEQ_LEN,
    build_tiny_model,
    tiny_model_kwargs,
    tiny_streams,
)


def build_model(**overrides) -> SeqVaeLagResidualTrfCfs:
    """Build the tiny model with any keyword replaced.

    Args:
        **overrides: Constructor keywords to replace.

    Returns:
        The model.
    """
    return build_tiny_model(**overrides)


def every_head_loss(outputs) -> torch.Tensor:
    """A scalar that touches every output head, so a reachability check is not partial.

    Backpropagating one forecast mean alone leaves the observation log-variance head with no
    gradient and the check reports a false failure, which is a slower way of learning that the
    loss was the wrong shape.

    Args:
        outputs: The forward's dict.

    Returns:
        The summed scalar.
    """
    return sum(
        outputs[name].sum()
        for name in (
            "mu_base",
            "logvar_base",
            "mu_full",
            "logvar_full",
            "kld_per_anchor",
            "mu_prior",
            "logvar_prior",
        )
    )


# =================================================================================================
# The module tree
# =================================================================================================
def test_no_attention_module_over_the_source_is_constructed() -> None:
    """The design's central structural prohibition, measured on the built tree.

    Not "is unused" -- an unreachable parameter block is a starved entry in a distributed run's
    expectation set and a claim in the checkpoint that the model attends over a source state it
    does not have.
    """
    model = build_model()
    forbidden = (LagCrossAttention, PosteriorHead, TEAnalysisHead, GatedCausalConvStem)
    found = [
        (name, type(module).__name__)
        for name, module in model.named_modules()
        if isinstance(module, forbidden)
    ]
    assert found == []

    for absent in (
        "lag_attn",
        "query_proj",
        "posterior_head",
        "te_analysis",
        "source_encoder_deep",
        "source_kv_stem",
    ):
        assert not hasattr(model, absent), f"{absent} was constructed"

    # Exactly one history encoder, and it is the target's. A second one would be a source encoder
    # under another name.
    encoders = [
        name
        for name, module in model.named_modules()
        if isinstance(module, CausalConvTransformerEncoder)
    ]
    assert encoders == ["target_encoder"]


def test_the_model_builds_the_pieces_this_architecture_does_have() -> None:
    """The positive half, so the test above cannot pass by building nothing."""
    model = build_model()
    assert isinstance(model.source_encoder, PointwiseSourceEncoder)
    assert isinstance(model.proposal_head, LagProposalHead)
    assert isinstance(model.prior_head, FullLatentPriorHead)
    assert isinstance(model.horizon_core, HorizonDecoderCore)
    assert isinstance(model.decoder, BaselineFutureDecoder)
    assert isinstance(model.clock_proj, nn.Linear)
    assert model.MODEL_KIND == MODEL_KIND

    # The pointwise encoder holds no parameters on the recommended arm, so the whole source
    # pathway's learned weight lives in one module.
    assert not model.source_encoder.has_parameters()
    source_parameters = {
        name for name, _ in model.named_parameters() if name.startswith("source_encoder.")
    }
    assert source_parameters == set()


def test_the_prior_head_is_built_without_its_own_clock_path() -> None:
    """The clock projection belongs to the model, because the proposal head reads it too.

    A projection inside the prior head could serve the prior alone, so the conditioning state would
    have to be formed twice -- and two clocks is not one clock.
    """
    model = build_model()
    assert model.prior_head.clock_proj is None
    assert model.prior_head.clock_norm is None
    assert model.clock_proj.bias is None  # bias-free, so a zeroed weight gives an exact zero


def test_the_zeroed_projections_survive_construction() -> None:
    """Both of them, after the generic initialisation pass the constructor runs.

    Either one left xavier-filled starts the model with a source correction it never learned, and
    the exact zero-update start silently does not hold.
    """
    model = build_model()
    assert torch.all(model.proposal_head.output_proj.weight == 0.0)
    assert torch.all(model.proposal_head.output_proj.bias == 0.0)
    assert torch.all(model.clock_proj.weight == 0.0)
    assert model.n_depthwise_init > 0, "the depthwise repair did not run; the check is vacuous"


def test_the_metadata_clock_is_a_function_of_position_alone() -> None:
    """Sine and cosine pairs over stored position, one pair per frequency, and non-persistent."""
    model = build_model()
    assert model.metadata_clock.shape == (TINY_SEQ_LEN, TINY_D_MODEL)
    assert bool(torch.isfinite(model.metadata_clock).all())
    # Non-persistent, like every geometry-shaped tensor in this family.
    assert "metadata_clock" not in model.state_dict()
    # The pairs are complete: every sine coordinate has its cosine, and at step zero the sines are
    # zero and the cosines one.
    assert torch.allclose(model.metadata_clock[0, 0::2], torch.zeros(TINY_D_MODEL // 2), atol=1e-6)
    assert torch.allclose(model.metadata_clock[0, 1::2], torch.ones(TINY_D_MODEL // 2), atol=1e-6)


def test_the_mean_only_arm_is_a_different_module_tree() -> None:
    """Not a flag consulted in the forward: the scale-proposal parameters are never built."""
    full = build_model()
    lean = build_model(mean_only_residual=True)
    assert full.proposal_head.output_proj.out_features == 2 * TINY_D_Z
    assert lean.proposal_head.output_proj.out_features == TINY_D_Z
    assert sum(p.numel() for p in lean.parameters()) < sum(
        p.numel() for p in full.parameters()
    )


def test_an_odd_model_width_is_refused() -> None:
    """The clock pairs a sine with a cosine per frequency; an odd width leaves one unbuilt."""
    with pytest.raises(ValueError, match="d_model must be even"):
        build_model(d_model=TINY_D_MODEL + 1)


@pytest.mark.parametrize("bound", ["residual_mu_scale", "residual_logsigma_scale"])
def test_a_zero_residual_bound_is_refused(bound: str) -> None:
    """A zero bound pins the update at zero while leaving a fully built head in the graph."""
    with pytest.raises(ValueError, match="residual bounds must be > 0"):
        build_model(**{bound: 0.0})


# =================================================================================================
# Reachability
# =================================================================================================
def test_every_parameter_is_reachable_when_the_source_is_entirely_unavailable() -> None:
    """The condition that makes a distributed run safe without unused-parameter handling.

    A parameter multiplied by an identically-zero mask is still **in the graph** and receives a
    zeros gradient, which is what a distributed run needs. What breaks it is a parameter left out
    of the graph, which a data-dependent Python branch would do on some ranks and not others.
    """
    # Every source channel cold for the whole record, so no lag carries any available channel.
    model = build_model(
        source_warmup_steps=tuple(TINY_SEQ_LEN for _ in range(DECLARED_C_U))
    )
    y_st, y_ph, u_stream = tiny_streams()
    outputs = model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1)

    assert not bool(outputs["lag_valid"].any()), "the fixture is not the all-unavailable case"

    every_head_loss(outputs).backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert missing == []


def test_every_parameter_is_reachable_under_mixed_validity() -> None:
    """The ordinary case, so the all-unavailable check is not the only evidence.

    Mixed validity is a **per-channel** condition here, not a per-lag one: the fastest source
    channel is warm from the first stored step, so every anchor-lag pair carries something, while
    the slowest channels are still cold at the early anchors. That is the arrangement a real run
    is in, and it is why the lag-level indicator alone would not show it.
    """
    model = build_model()
    y_st, y_ph, u_stream = tiny_streams()
    outputs = model(
        y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1, return_proposals=True
    )
    channels = outputs["source_channel_mask"]
    assert bool(channels.any()) and not bool(channels.all()), "the fixture is not mixed"

    every_head_loss(outputs).backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert missing == []


def test_the_scalar_lift_parameters_are_reachable_too() -> None:
    """The one arm that puts learned weights in the source encoder."""
    model = build_model(source_scalar_lift=True)
    assert model.source_encoder.has_parameters()
    y_st, y_ph, u_stream = tiny_streams()
    outputs = model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1)
    every_head_loss(outputs).backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert missing == []


# =================================================================================================
# Refusals
# =================================================================================================
@pytest.mark.parametrize("keyword", sorted(REFUSED_KEYWORDS))
def test_every_refused_keyword_raises_by_name(keyword: str) -> None:
    """Naming the key and the reason, rather than a bare unexpected-argument error."""
    values = {
        "lag_kv_source": "conv_stem",
        "lag_bias_init": "alibi_decay",
        "posterior_logvar_mode": "independent",
        "base_decode": "mean",
    }
    with pytest.raises(ValueError, match=keyword):
        build_model(**{keyword: values.get(keyword, 1)})


def test_every_refused_keyword_is_in_the_signature() -> None:
    """Otherwise the driver drops it in silence and the refusal never fires.

    The driver forwards a configuration key only when ``inspect.signature`` names it, so a refusal
    that is not also a parameter is a refusal that can never run.
    """
    parameters = set(inspect.signature(SeqVaeLagResidualTrfCfs.__init__).parameters)
    assert set(REFUSED_KEYWORDS) <= parameters


def test_a_refused_keyword_left_unset_builds_the_model() -> None:
    """The default is ``None`` and means unset, which is what an absent configuration key gives."""
    model = build_model(**{name: None for name in REFUSED_KEYWORDS})
    assert isinstance(model, SeqVaeLagResidualTrfCfs)


def test_an_unknown_keyword_is_a_type_error() -> None:
    """Anything outside the schema, refused by Python itself and naming the key."""
    with pytest.raises(TypeError, match="conv_norm_groups"):
        build_model(conv_norm_groups=8)


# =================================================================================================
# Composition
# =================================================================================================
def test_the_resolution_order_is_the_one_the_design_names() -> None:
    """Written out, so a reorder fails here rather than training a different model.

    The mixins must come first: their construction hooks and their target gather have to win over
    the base's, and the base's ``_build_adapter`` is the fallback the warm-up mixin delegates to.
    """
    assert [cls.__name__ for cls in SeqVaeLagResidualTrfCfs.__mro__] == [
        "SeqVaeLagResidualTrfCfs",
        "CausalWarmupInputs",
        "CausalFeatureForecastTarget",
        "FeatureForecastTarget",
        "LagResidualCore",
        "Module",
        "object",
    ]


def test_each_shared_member_resolves_to_the_class_the_design_names() -> None:
    """Not merely that the order is right, but that each member comes from where it should."""
    owner = {
        "forward": SeqVaeLagResidualTrfCfs,
        "build_lag_mask": SeqVaeLagResidualTrfCfs,
        "_prior_clock": SeqVaeLagResidualTrfCfs,
        "_build_anchor_index": CausalWarmupInputs,
        "_build_adapter": CausalWarmupInputs,
        "_validate_causal_geometry": CausalWarmupInputs,
        # The model's own, and this entry is the guard: un-overridden it resolves to the mixin,
        # which reaches the shared objective and its per-rank clamped reduction.
        "compute_loss": SeqVaeLagResidualTrfCfs,
        "_anchor_target_values": CausalFeatureForecastTarget,
        "_build_forecast_target": CausalFeatureForecastTarget,
        "_default_decoder_out_channels": FeatureForecastTarget,
        "conditioning_state": LagResidualCore,
        "reparameterize_shared": LagResidualCore,
        "_build_channel_gate": LagResidualCore,
    }
    for name, expected in owner.items():
        for cls in SeqVaeLagResidualTrfCfs.__mro__:
            if name in vars(cls):
                assert cls is expected, f"{name} resolves to {cls.__name__}"
                break
        else:  # pragma: no cover - a missing member is a construction failure long before here
            pytest.fail(f"{name} is defined nowhere in the resolution order")


def test_the_warm_up_budget_resolver_accepts_this_class() -> None:
    """It sweeps the signature, so a keyword this model dropped would silently ungate the run."""
    parameters = set(inspect.signature(SeqVaeLagResidualTrfCfs.__init__).parameters)
    for required in (
        "target_keep_index",
        "target_warmup_steps",
        "source_keep_index",
        "source_warmup_steps",
        "target_novelty_frac",
    ):
        assert required in parameters
    # And the resolver itself refuses a class that lost them, which is the guard being relied on.
    assert warmup_model_kwargs(None, SeqVaeLagResidualTrfCfs) == {}


def test_the_decoder_width_follows_the_target_gate() -> None:
    """No keyword records it, so no second field can disagree with the gate."""
    model = build_model()
    assert model.decoder_out_channels == len(model.target_gate.keep_index)
    assert model.decoder.out_channels == model.decoder_out_channels


def test_the_core_is_never_meant_to_stand_alone() -> None:
    """It has no target gather and no decoder width of its own, by design.

    Recorded as a test so the omission reads as deliberate: the two hooks below are the target
    domain's, and a core that supplied its own would be a second definition of them.
    """
    assert "_default_decoder_out_channels" not in vars(LagResidualCore)
    assert "_build_forecast_target" not in vars(LagResidualCore)
