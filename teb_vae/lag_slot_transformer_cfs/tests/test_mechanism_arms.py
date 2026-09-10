r"""The comparator arms, and the properties that make a comparison between them mean anything.

A mechanism-separating comparison is a set of trained models that each changed **one** declared
thing. Everything that makes it more than a set of numbers is structural and checkable here, before
a single day of training is spent:

* each arm changes what it says it changes, in the **module tree** rather than in a branch of the
  forward, so a checkpoint cannot be one arm while its configuration says another;
* every arm still starts at exactly the prior, so a divergence any of them later reports was earned
  during its own run;
* every arm still reproduces the prior when its source is silenced, so the reference identities the
  evaluation reads every margin against hold on all of them;
* the arms that must have identical trainable capacity do;
* and the one readout that is **not** comparable across the pair -- a band-suppression margin under
  two different aggregations -- is recorded as such rather than left to be differenced by a reader.

What none of this establishes is the outcome. It says the comparison is wired to answer its
question; whether the answer is interesting is what the runs are for.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest
import torch

from teb_vae.lag_attn.nets.attention import LagCrossAttention
from teb_vae.lag_attn.nets.heads import PosteriorHead, TEAnalysisHead
from teb_vae.lag_attn_transformer_rws.nets.encoders import GatedCausalConvStem
from teb_vae.lag_slot_transformer_cfs.nets import controls
from teb_vae.lag_slot_transformer_cfs.nets.conv_source import ConvSourceStem
from teb_vae.lag_slot_transformer_cfs.nets.core import pathway_parameter_counts
from teb_vae.lag_slot_transformer_cfs.nets.lag_attention import LagAttentionFusion
from teb_vae.lag_slot_transformer_cfs.nets.lag_updates import LagProposalHead
from teb_vae.lag_slot_transformer_cfs.nets.pointwise_source import PointwiseSourceEncoder

from .conftest import build_tiny_model, tiny_streams

#: The comparator arms, keyed by the name their configuration ships under, and the single leaf each
#: one moves against the recommended candidate. Written out rather than read from the configuration
#: directory: what is being asserted is that each arm is **one** change, and a table derived from
#: the files could not say that.
ARMS: Dict[str, Dict[str, Any]] = {
    "candidate": {},
    "target_only": {"source_disabled": True},
    "capacity_control": {"source_values_withheld": True},
    "mean_only": {"mean_only_residual": True},
    "pointwise_attention": {"lag_fusion": "attention"},
    "attention_reference": {"lag_fusion": "attention", "source_stem": "conv"},
}


def _forward(model, **kwargs):
    """Run one dense evaluation-mode forward on the shared seeded streams.

    Args:
        model: The net.
        **kwargs: Extra forward keywords, such as a selector.

    Returns:
        The forward's dict.
    """
    y_st, y_ph, u_stream = tiny_streams()
    model.eval()
    return model(
        y_st,
        y_ph,
        u_stream,
        anchor_phase=torch.zeros(y_st.shape[0], dtype=torch.long),
        anchor_stride=1,
        **kwargs,
    )


def _wake(model, *, seed: int = 5) -> None:
    """Move the fusion off its zero start, so an invariant is tested on a live pathway.

    Every source-off identity below is trivially true at the zero initialisation, where the update
    is zero for every input. Waking the output projection first is what turns each of them from a
    restatement of the initialisation into a statement about the arm.

    Args:
        model: The net, modified in place.
        seed: Seed applied immediately before the draw.
    """
    torch.manual_seed(seed)
    projection = model.proposal_head.output_proj
    torch.nn.init.normal_(projection.weight, std=0.4)
    torch.nn.init.normal_(projection.bias, std=0.4)


# =================================================================================================
# Each arm is a module tree
# =================================================================================================
@pytest.mark.parametrize("arm", sorted(ARMS))
def test_each_arm_declares_itself_on_the_model(arm: str) -> None:
    """The leaves reach the model as attributes, which is what every readout reports from.

    An arm whose flag were consumed at construction and then forgotten would produce correct numbers
    under a summary that could not say which arm produced them.

    Args:
        arm: The arm to build.
    """
    model = build_tiny_model(**ARMS[arm])
    for leaf, value in ARMS[arm].items():
        assert getattr(model, leaf) == value, leaf


def test_the_two_fusions_are_different_modules_and_the_stems_too() -> None:
    """Not one module consulting a flag, which could not change a checkpoint's key set."""
    local = build_tiny_model()
    attention = build_tiny_model(lag_fusion="attention")
    conv = build_tiny_model(lag_fusion="attention", source_stem="conv")

    assert isinstance(local.proposal_head, LagProposalHead)
    assert isinstance(attention.proposal_head, LagAttentionFusion)
    assert isinstance(local.source_encoder, PointwiseSourceEncoder)
    assert isinstance(conv.source_encoder, ConvSourceStem)

    # And the state dicts differ in their key sets, so a strict load across two arms refuses rather
    # than silently reshaping.
    assert set(local.state_dict()) != set(attention.state_dict())
    assert set(attention.state_dict()) != set(conv.state_dict())


def test_only_the_declared_comparator_builds_an_attention_over_lags() -> None:
    """The structural prohibition still holds everywhere it was ever claimed to.

    The recommended arm builds no attention over the source and no head-structured posterior. The
    comparator builds exactly one lag fusion, and it is this package's own -- not the sibling's
    class, whose per-head summaries feed a head-structured posterior this architecture has on no
    arm.
    """
    forbidden = (LagCrossAttention, PosteriorHead, TEAnalysisHead)
    for arm, overrides in ARMS.items():
        model = build_tiny_model(**overrides)
        found = [
            name for name, module in model.named_modules() if isinstance(module, forbidden)
        ]
        assert found == [], (arm, found)

        stems = [
            name
            for name, module in model.named_modules()
            if isinstance(module, GatedCausalConvStem)
        ]
        expected = ["source_encoder.stem"] if overrides.get("source_stem") == "conv" else []
        assert stems == expected, (arm, stems)

        fusions = [
            name
            for name, module in model.named_modules()
            if isinstance(module, LagAttentionFusion)
        ]
        assert fusions == (
            ["proposal_head"] if overrides.get("lag_fusion") == "attention" else []
        ), (arm, fusions)


def test_the_capacity_control_holds_the_candidates_budget_exactly() -> None:
    """The whole point of the arm: same head, same widths, same weights to train.

    An arm that differed in capacity as well as in what it reads could not separate "the source
    values helped" from "the bigger head helped", which is the confound it exists to remove.
    """
    candidate = build_tiny_model()
    control = build_tiny_model(source_values_withheld=True)

    assert pathway_parameter_counts(control) == pathway_parameter_counts(candidate)
    assert set(control.state_dict()) == set(candidate.state_dict())
    for name, parameter in control.named_parameters():
        assert parameter.shape == dict(candidate.named_parameters())[name].shape, name


def test_the_capacity_control_reads_no_source_value() -> None:
    """Asserted on the encoding rather than on the forward's output.

    A model whose update merely happened not to move on one batch would pass a comparison of
    forecasts. What has to be true is that the value never entered the graph at all, which is a
    property of what the encoder emitted.
    """
    control = build_tiny_model(source_values_withheld=True)
    _y_st, _y_ph, u_stream = tiny_streams()
    encoded, mask = control.source_encoder(control.source_gate(u_stream))

    assert torch.all(encoded[..., 0] == 0.0)
    # The availability bit is still there and is still informative, which is what makes the arm a
    # capacity control rather than a second target-only model.
    assert torch.equal(encoded[..., 1], mask.to(encoded.dtype))
    assert bool(mask.any()) and not bool(mask.all())


def test_the_capacity_control_is_not_the_target_only_arm() -> None:
    """It holds the pathway and starves it; the other removes it. Two questions, two budgets."""
    control = build_tiny_model(source_values_withheld=True)
    target_only = build_tiny_model(source_disabled=True)

    assert pathway_parameter_counts(control)["source"] > 0
    assert pathway_parameter_counts(target_only)["source"] == 0
    assert control.proposal_head is not None
    assert target_only.proposal_head is None


# =================================================================================================
# The invariants every arm still has to satisfy
# =================================================================================================
@pytest.mark.parametrize("arm", sorted(set(ARMS) - {"target_only"}))
def test_every_arm_starts_at_exactly_the_prior(arm: str) -> None:
    """So a divergence any arm reports was earned during its own run, not inherited.

    Args:
        arm: The arm to build.
    """
    outputs = _forward(build_tiny_model(**ARMS[arm]))

    assert torch.equal(outputs["mu_post"], outputs["mu_prior"])
    assert torch.equal(outputs["logvar_post"], outputs["logvar_prior"])
    assert float(outputs["kld_per_anchor"].abs().max()) == 0.0


@pytest.mark.parametrize("arm", sorted(set(ARMS) - {"target_only"}))
def test_every_arm_reproduces_the_prior_when_silenced(arm: str) -> None:
    """On a woken pathway, which is what makes this a statement rather than a restatement.

    This is the identity every reported margin is read against: the silence arm's margin equals the
    base-minus-full gap by construction, so a drift in it is a defect report rather than a
    measurement. It has to hold under both aggregations, and the two reach it differently -- a sum
    of zeroed terms, and a distribution with no admissible lag left in it.

    Args:
        arm: The arm to build.
    """
    model = build_tiny_model(**ARMS[arm])
    _wake(model)
    live = _forward(model)
    assert float(live["update_mean"].abs().max()) > 0.0, "the pathway did not wake"

    batch, n_anchors = live["mu_prior"].shape[0], live["mu_prior"].shape[1]
    silent = torch.zeros(batch, n_anchors, model.n_lags)
    outputs = _forward(model, selector=silent)

    assert torch.equal(outputs["mu_post"], outputs["mu_prior"])
    assert torch.equal(outputs["logvar_post"], outputs["logvar_prior"])
    assert float(outputs["kld_per_anchor"].abs().max()) == 0.0


@pytest.mark.parametrize("arm", sorted(set(ARMS) - {"target_only"}))
def test_an_empty_band_reproduces_the_matched_forward_on_every_arm(arm: str) -> None:
    """The other reference identity, and the one that differs in mechanism between the fusions.

    Under the local sum an empty band subtracts an exact zero. Under the attention it admits every
    lag the matched forward admitted. Both must land on the matched parameters exactly, or a
    reported margin carries the difference between two code paths as well as the intervention.

    Args:
        arm: The arm to build.
    """
    model = build_tiny_model(**ARMS[arm])
    _wake(model)
    matched = _forward(model)

    batch, n_anchors = matched["mu_prior"].shape[0], matched["mu_prior"].shape[1]
    keep_all = controls.band_selector(
        torch.zeros(model.n_lags, dtype=torch.bool), batch, n_anchors, dtype=torch.float32
    )
    outputs = _forward(model, selector=keep_all)

    assert torch.equal(outputs["mu_post"], matched["mu_post"])
    assert torch.equal(outputs["logvar_post"], matched["logvar_post"])


@pytest.mark.parametrize("arm", sorted(set(ARMS) - {"target_only"}))
def test_the_predictive_gradient_reaches_every_arms_final_projection(arm: str) -> None:
    """Zero initialisation is a starting point on every arm, not a fixed point on some of them.

    A comparator whose fusion could never leave zero would train as a target-only forecaster and
    report a gap of zero, which reads as a finding about the mechanism.

    Args:
        arm: The arm to build.
    """
    model = build_tiny_model(**ARMS[arm])
    model.train()
    y_st, y_ph, u_stream = tiny_streams()
    outputs = model(
        y_st,
        y_ph,
        u_stream,
        anchor_phase=torch.zeros(y_st.shape[0], dtype=torch.long),
        anchor_stride=1,
    )
    (outputs["mu_full"].sum() + outputs["logvar_full"].sum()).backward()

    gradient = model.proposal_head.output_proj.weight.grad
    assert gradient is not None and float(gradient.abs().max()) > 0.0


# =================================================================================================
# What the two fusions do differently, asserted rather than described
# =================================================================================================
def test_suppressing_a_band_redistributes_weight_under_attention_and_not_under_the_sum() -> None:
    """The reason a band margin is not comparable across the pair.

    Under the local sum, removing a band leaves every surviving proposal exactly where it was, so
    the raw update is the matched one minus the removed terms. Under the attention the survivors
    grow to fill the removed weight, so no such subtraction exists -- which is why the two arms are
    suppressed by different code and why the run records the fusion beside every margin.
    """
    band = torch.zeros(build_tiny_model().n_lags, dtype=torch.bool)
    band[0] = True

    local = build_tiny_model()
    _wake(local)
    matched = _forward(local, return_proposals=True)
    batch, n_anchors = matched["mu_prior"].shape[0], matched["mu_prior"].shape[1]
    selector = controls.band_selector(band, batch, n_anchors, dtype=torch.float32)
    suppressed = _forward(local, selector=selector)

    # The surviving proposals are untouched, so the suppressed raw update is exactly the matched one
    # less the removed band's own contribution. Nothing redistributes.
    removed = float(local.lag_scale) * matched["mean_proposals"][:, :, band].sum(dim=2)
    assert torch.allclose(
        suppressed["raw_update_mean"], matched["raw_update_mean"] - removed, atol=1e-6
    )

    # The attention arm admits no such statement, and the module that would compute one refuses
    # rather than producing a number with no interpretation.
    attention = build_tiny_model(lag_fusion="attention")
    _wake(attention)
    with pytest.raises(ValueError, match="normalised distribution"):
        controls.suppressed_parameters(attention, {}, band)

    # What it does instead: the surviving weights grow to fill the removed one, so the suppressed
    # update is not the matched one shifted by any fixed quantity. Asserted as a *difference* from
    # the local arm's behaviour rather than as a property of the numbers themselves.
    attended = _forward(attention)
    attended_band = _forward(
        attention,
        selector=controls.band_selector(band, batch, n_anchors, dtype=torch.float32),
    )
    assert not torch.allclose(
        attended_band["raw_update_mean"], attended["raw_update_mean"], atol=1e-6
    )


def test_the_attention_arm_publishes_no_distribution_over_lags() -> None:
    """A deliberate omission, not a gap.

    The weights are real on this arm. Published beside a predictive comparison they would be read
    as a lag readout by every reader and every downstream table, which is the claim the evidence
    behind this design shows cannot be supported. Both arms are interrogated by suppression
    instead.
    """
    model = build_tiny_model(lag_fusion="attention")
    _wake(model)
    outputs = _forward(model, return_proposals=True)

    assert not [key for key in outputs if "attention" in key or "attn" in key]
    assert "mean_proposals" not in outputs
    assert "scale_proposals" not in outputs
    # And no cancellation readout, because a normalised aggregation has nothing that can cancel.
    assert not [key for key in outputs if key.startswith("cancellation")]
    # The exposure input is still there: availability is a property of the channel and the step,
    # whichever representation formed the window, so two arms' exposure tables stay comparable.
    assert outputs["source_channel_mask"].shape[-1] == model.source_gate.out_channels


def test_the_convolution_stem_reaches_further_than_one_stored_step() -> None:
    """Which is the quantity the comparison against the pointwise arm exists to move.

    Read off the module rather than written down, and disclosed by the run for the same reason: it
    is the resolution floor of every band margin taken over that representation.
    """
    conv = build_tiny_model(lag_fusion="attention", source_stem="conv")
    pointwise = build_tiny_model(lag_fusion="attention")

    assert conv.source_encoder.receptive_field > 1
    assert not hasattr(pointwise.source_encoder, "receptive_field")


def test_the_conv_stem_carries_the_same_per_channel_availability_as_the_pointwise_arm() -> None:
    """So the exposure readout reports the same counts on both, and the tables are comparable.

    The stem mixes the values; it does not make the availability announcement disappear, and an
    exposure table that changed with the representation would not be an exposure table.
    """
    conv = build_tiny_model(lag_fusion="attention", source_stem="conv")
    pointwise = build_tiny_model(lag_fusion="attention")
    _y_st, _y_ph, u_stream = tiny_streams()

    _state, conv_mask = conv.source_encoder(conv.source_gate(u_stream))
    _encoded, point_mask = pointwise.source_encoder(pointwise.source_gate(u_stream))
    assert torch.equal(conv_mask, point_mask)


# =================================================================================================
# The refusals that keep an arm one declared change
# =================================================================================================
@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"source_stem": "identity"}, "source_stem"),
        ({"lag_fusion": "softmax"}, "lag_fusion"),
        (
            {"source_values_withheld": True, "source_stem": "conv"},
            "capacity control",
        ),
        ({"source_scalar_lift": True, "source_stem": "conv"}, "per-coefficient"),
        ({"lag_fusion": "attention", "lag_scale": 0.5}, "no summation to scale"),
        ({"lag_fusion": "attention", "lag_chunk": 2}, "renormalise"),
    ),
)
def test_an_undeclared_or_meaningless_arm_combination_is_refused(
    overrides: Dict[str, Any], message: str
) -> None:
    """Each of these would otherwise run and report a mechanism it did not have.

    Args:
        overrides: The constructor keywords to combine.
        message: A phrase the refusal must name, so the operator is pointed at the leaf.
    """
    with pytest.raises(ValueError, match=message):
        build_tiny_model(**overrides)


def test_the_scalar_lift_and_the_withheld_values_are_refused_together() -> None:
    """The lift of a withheld value is a learned per-channel constant on every coefficient.

    The arm would hold parameters that read nothing and would still be reported as a control on
    capacity, which is the one claim it exists to make.
    """
    with pytest.raises(ValueError, match="withhold_values"):
        build_tiny_model(source_scalar_lift=True, source_values_withheld=True)
