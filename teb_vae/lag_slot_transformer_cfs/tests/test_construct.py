r"""What the model builds, what it refuses to build, and what it refuses to be told.

Two structural claims are checked here and each is invisible at review time:

* **no attention module is constructed.** The design's requirement is not that a lag cross-attention
  goes unused but that it never exists, and the only mechanical proof is a walk over the built
  module tree;
* **a keyword naming machinery this architecture lacks raises.** The experiment driver forwards a
  configuration key only when the constructor names it, so a key omitted from the signature is
  dropped in silence -- and a run configured for windowed source attention over a model with none
  would train to completion and report its divergence as a coupling measurement of a pathway it
  never built.

Beside them: the zero start survives construction, the metadata clock is a function of position
alone, the arms that change the module tree change it, and each shared member resolves to the
mixin the design names. Parameter reachability lives in ``test_ddp_reachability.py``, under the
real objective.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.nets.attention import LagCrossAttention
from teb_vae.lag_attn.nets.heads import PosteriorHead, TEAnalysisHead
from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
from teb_vae.lag_attn_cfs.nets.causal_inputs import CausalWarmupInputs
from teb_vae.lag_attn_fs.nets.feature_target import FeatureForecastTarget
from teb_vae.lag_attn_transformer_rws.nets.encoders import (
    CausalConvTransformerEncoder,
    GatedCausalConvStem,
)
from teb_vae.lag_slot_transformer_cfs.nets.core import REFUSED_KEYWORDS, LagResidualCore
from teb_vae.lag_slot_transformer_cfs.nets.model import SeqVaeLagResidualTrfCfs
from teb_vae.lag_slot_transformer_cfs.tests.conftest import (
    TINY_D_MODEL,
    TINY_D_Z,
    TINY_SEQ_LEN,
    build_tiny_model,
)


def build_model(**overrides) -> SeqVaeLagResidualTrfCfs:
    """Build the tiny model with any keyword replaced.

    Args:
        **overrides: Constructor keywords to replace.

    Returns:
        The model.
    """
    return build_tiny_model(**overrides)


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


def test_the_recommended_source_encoder_holds_no_parameters() -> None:
    """The pointwise encoder holds no parameters on the recommended arm, so the whole source
    pathway's learned weight lives in the proposal head; the scalar-lift arm is the one that
    puts learned weights in the encoder."""
    model = build_model()
    assert not model.source_encoder.has_parameters()
    assert build_model(source_scalar_lift=True).source_encoder.has_parameters()


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


def test_a_refused_keyword_left_unset_builds_the_model() -> None:
    """The default is ``None`` and means unset, which is what an absent configuration key gives."""
    model = build_model(**{name: None for name in REFUSED_KEYWORDS})
    assert isinstance(model, SeqVaeLagResidualTrfCfs)


# =================================================================================================
# Composition
# =================================================================================================
def test_each_shared_member_resolves_to_the_class_the_design_names() -> None:
    """Each shared member comes from the class the design names.

    The mixins must win over the base where they supply a construction hook or the target gather,
    and the model must own ``compute_loss``.
    """
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


def test_the_decoder_width_follows_the_target_gate() -> None:
    """No keyword records it, so no second field can disagree with the gate."""
    model = build_model()
    assert model.decoder_out_channels == len(model.target_gate.keep_index)
    assert model.decoder.out_channels == model.decoder_out_channels
