r"""The two input-ablation switches, and every way one of them could leak.

An input ablation leaves no trace in any tensor shape, which is why every failure here is quiet:
a persistence shortcut that still carries the original coefficient, a control that reintroduces
it through a substituted stream, a label that was zeroed alongside the input, a switch that
zeroed the phase block's first channel under the scattering coefficient's name, or a checkpoint
whose flag the evaluation silently overrode. Each is a model that trains, scores and reports
plausible numbers under a policy it does not actually implement.

Everything below is exact rather than approximate, because the ablation is a multiplication by
zero at the input boundary and nothing downstream can add the coordinate back.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_cfs.eval.preflight import EvalPreconditionUnmet, reconcile_with_checkpoint
from teb_vae.lag_slot_transformer_cfs.eval.binding import GEOMETRY_KEYS, effective_input_disclosure
from teb_vae.lag_slot_transformer_cfs.nets.model import (
    INPUT_ABLATION_KEYWORDS,
    SeqVaeLagResidualTrfCfs,
)
from teb_vae.lag_slot_transformer_cfs.nets.controls import replaced_source_stream

from .conftest import build_tiny_model, tiny_model_kwargs, tiny_streams

#: Forward outputs an input perturbation must leave bitwise unchanged when the coordinate is
#: ablated: both forecasts, both latent parameter sets, the divergence and the persistence input.
INVARIANT_KEYS = (
    "mu_prior", "logvar_prior", "mu_post", "logvar_post", "mu_base", "logvar_base",
    "mu_full", "logvar_full", "kld_per_anchor", "persistence",
)


def awake(model: SeqVaeLagResidualTrfCfs, seed: int = 1) -> SeqVaeLagResidualTrfCfs:
    """Move the proposal head off its zero start, so the source pathway can react to its input.

    Args:
        model: A freshly built model.
        seed: Seed for the perturbation.

    Returns:
        The same model, in evaluation mode.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in model.proposal_head.output_proj.parameters():
            parameter.add_(0.2 * torch.randn_like(parameter))
    return model.eval()


def run(model, y_st, y_ph, u_stream, **kwargs):
    """One dense forward under a fixed noise draw, so two calls differ only in their inputs."""
    torch.manual_seed(0)
    with torch.no_grad():
        return model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1, **kwargs)


def perturbed(stream: torch.Tensor, channel: int, amount: float = 5.0) -> torch.Tensor:
    """A copy of a stream with one channel shifted everywhere."""
    copy = stream.clone()
    copy[..., channel] += amount
    return copy


# =================================================================================================
# The target switch
# =================================================================================================
def test_the_target_switch_makes_every_output_blind_to_the_coefficient_persistence_included() -> None:
    """Perturbing the ablated coordinate changes nothing; perturbing its neighbour does.

    The persistence input is the path a task-level mask would miss: it is gathered from the
    declared stream before the gate, so an ablation applied anywhere but the input boundary
    would leave the decoder reading the anchor's own level through it.
    """
    model = awake(build_tiny_model(zero_fhr_scattering_s0=True))
    y_st, y_ph, u_stream = tiny_streams()

    reference = run(model, y_st, y_ph, u_stream)
    blind = run(model, perturbed(y_st, 0), y_ph, u_stream)
    sighted = run(model, perturbed(y_st, 1), y_ph, u_stream)

    for name in INVARIANT_KEYS:
        assert torch.equal(reference[name], blind[name]), name
    assert torch.all(reference["persistence"][..., 0] == 0.0)
    assert not torch.equal(reference["mu_full"], sighted["mu_full"])
    assert not torch.equal(reference["persistence"], sighted["persistence"])


def test_the_target_switch_leaves_the_inputs_and_the_labels_untouched() -> None:
    """The ablation is a multiplication into a new tensor, never a write into the batch.

    The task builds the forecast labels from the same field the encoder input comes from, so an
    in-place zero would silently zero the labels too and the model would be scored on a target
    it was handed for free.
    """
    model = awake(build_tiny_model(zero_fhr_scattering_s0=True))
    y_st, y_ph, u_stream = tiny_streams()
    before = (y_st.clone(), y_ph.clone(), u_stream.clone())

    outputs = run(model, y_st, y_ph, u_stream)
    labels = model._build_forecast_target(torch.cat([y_st, y_ph], dim=-1), outputs["anchor_index"])

    for original, kept in zip((y_st, y_ph, u_stream), before):
        assert torch.equal(original, kept)
    # The labels still carry the coefficient the model may not read.
    assert float(labels[..., 0].abs().max()) > 0.0


def test_the_gradient_with_respect_to_the_ablated_coordinate_is_exactly_zero() -> None:
    """An attribution through an ablated coordinate is zero by construction, not by tolerance."""
    model = awake(build_tiny_model(zero_fhr_scattering_s0=True))
    y_st, y_ph, u_stream = tiny_streams()
    y_st = y_st.clone().requires_grad_(True)

    torch.manual_seed(0)
    outputs = model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1)
    (outputs["mu_full"].sum() + outputs["kld_per_anchor"].sum()).backward()

    assert float(y_st.grad[..., 0].abs().max()) == 0.0
    assert float(y_st.grad[..., 1:].abs().max()) > 0.0


def test_the_persistence_baseline_reads_the_permitted_view_while_the_labels_keep_the_coefficient() -> None:
    """The trivial predictor the model is compared against may not read what the model may not.

    Built on the permitted view, the persistence baseline forecasts a zero level; the scored
    target, gathered from the original stream, still carries the coefficient. A baseline built on
    the original stream would beat the model on that channel with information the model was
    denied.
    """
    from teb_vae.lag_attn_cfs.eval.metrics import baseline_forecasts

    model = build_tiny_model(zero_fhr_scattering_s0=True).eval()
    y_st, y_ph, u_stream = tiny_streams()
    stream = torch.cat([y_st, y_ph], dim=-1)
    weight = torch.ones(stream.shape[0], stream.shape[1])
    outputs = run(model, y_st, y_ph, u_stream)
    anchors = outputs["anchor_index"]

    baselines = baseline_forecasts(model.permitted_target_features(stream), weight, model, anchors)
    labels = model._build_forecast_target(stream, anchors)

    assert torch.all(baselines["persistence"][..., 0] == 0.0)
    assert torch.all(baselines["segment_mean"][..., 0] == 0.0)
    assert float(baselines["persistence"][..., 1:].abs().max()) > 0.0
    assert float(labels[..., 0].abs().max()) > 0.0
    # The baseline is returned at a broadcastable shape with a horizon axis of one; squeezed, it
    # is the forward's own persistence input for that channel.
    assert torch.equal(baselines["persistence"].squeeze(2)[..., 0], outputs["persistence"][..., 0])


def test_the_permitted_view_zeroes_exactly_the_ablated_coordinate_of_the_declared_stream() -> None:
    """What every trivial baseline and probe must be built on, and nothing else changes in it."""
    model = build_tiny_model(zero_fhr_scattering_s0=True)
    y_st, y_ph, _u_stream = tiny_streams()
    stream = torch.cat([y_st, y_ph], dim=-1)

    view = model.permitted_target_features(stream)

    assert view is not stream
    assert torch.all(view[..., 0] == 0.0)
    assert torch.equal(view[..., 1:], stream[..., 1:])
    # And the identity object on a model with no target ablation.
    assert build_tiny_model().permitted_target_features(stream) is stream


# =================================================================================================
# The source switch
# =================================================================================================
def test_the_source_switch_makes_the_full_branch_blind_to_the_coefficient_and_keeps_the_mask() -> None:
    """The coefficient is a value ablation: the channel stays announced as available."""
    model = awake(build_tiny_model(zero_up_scattering_s0=True))
    y_st, y_ph, u_stream = tiny_streams()

    reference = run(model, y_st, y_ph, u_stream, return_proposals=True)
    blind = run(model, y_st, y_ph, perturbed(u_stream, 0), return_proposals=True)
    sighted = run(model, y_st, y_ph, perturbed(u_stream, 1), return_proposals=True)

    for name in INVARIANT_KEYS:
        assert torch.equal(reference[name], blind[name]), name
    assert torch.equal(reference["source_channel_mask"], blind["source_channel_mask"])
    # The mask is the ordinary one: channel 0 is warm from the first step in the fixture, so it
    # is available wherever the lag is in range.
    unablated = run(awake(build_tiny_model()), y_st, y_ph, u_stream, return_proposals=True)
    assert torch.equal(reference["source_channel_mask"], unablated["source_channel_mask"])
    assert not torch.equal(reference["mu_full"], sighted["mu_full"])


def test_the_source_switch_never_touches_the_phase_block() -> None:
    """The phase block's first channel sits at the scattering block's width, not at zero."""
    model = awake(build_tiny_model(zero_up_scattering_s0=True))
    y_st, y_ph, u_stream = tiny_streams()
    first_phase = int(model.TARGET_BLOCK_SPLIT)

    reference = run(model, y_st, y_ph, u_stream)
    moved = run(model, y_st, y_ph, perturbed(u_stream, first_phase))

    assert not torch.equal(reference["mu_full"], moved["mu_full"])


def test_the_source_switch_is_refused_without_the_scattering_block() -> None:
    """With the phase block alone the stream opens with a phase coefficient."""
    with pytest.raises(ValueError, match="zero_up_scattering_s0") as caught:
        build_tiny_model(zero_up_scattering_s0=True, use_up_st=False)
    assert "use_up_st" in str(caught.value)


def test_a_substituted_stream_cannot_reintroduce_the_coefficient() -> None:
    """The controls re-run the forward, whose first step is the ablation, so a constant stream
    that carries the recording's own level reaches the encoder with that level zeroed."""
    model = awake(build_tiny_model(zero_up_scattering_s0=True))
    y_st, y_ph, u_stream = tiny_streams()
    constant = replaced_source_stream(u_stream, "constant")
    assert float(constant[..., 0].abs().max()) > 0.0

    reference = run(model, y_st, y_ph, constant)
    shifted = run(model, y_st, y_ph, perturbed(constant, 0))

    assert torch.equal(reference["mu_full"], shifted["mu_full"])


# =================================================================================================
# Both off: the legacy forward
# =================================================================================================
def test_both_switches_off_reproduce_the_legacy_forward_bitwise() -> None:
    """The default is the model that was shipped, and the flags are absent from the base."""
    y_st, y_ph, u_stream = tiny_streams()
    legacy = awake(build_tiny_model())
    explicit = awake(build_tiny_model(zero_fhr_scattering_s0=False, zero_up_scattering_s0=False))

    left = run(legacy, y_st, y_ph, u_stream)
    right = run(explicit, y_st, y_ph, u_stream)

    for name in INVARIANT_KEYS:
        assert torch.equal(left[name], right[name]), name
    assert legacy._ablate_input_streams(y_st, u_stream) == (y_st, u_stream)


def test_the_switches_are_this_class_own_and_are_not_forwarded_to_the_base() -> None:
    """A keyword forwarded to a base that does not name it would refuse every construction."""
    from teb_vae.lag_slot_transformer_cfs.nets.core import LagResidualCore

    base_parameters = set(LagResidualCore.__init__.__code__.co_varnames)
    for keyword in INPUT_ABLATION_KEYWORDS:
        assert keyword not in base_parameters
        assert keyword in SeqVaeLagResidualTrfCfs.__init__.__code__.co_varnames


# =================================================================================================
# The checkpoint and the binding
# =================================================================================================
def test_the_flags_survive_a_checkpoint_round_trip_and_are_reconciled_by_the_binding() -> None:
    """The policy is stamped through the constructor kwargs; an override that disagrees refuses."""
    kwargs = tiny_model_kwargs(zero_fhr_scattering_s0=True)
    original = awake(SeqVaeLagResidualTrfCfs(**kwargs))
    rebuilt = SeqVaeLagResidualTrfCfs(**kwargs).eval()
    rebuilt.load_state_dict(original.state_dict())
    y_st, y_ph, u_stream = tiny_streams()

    assert rebuilt.zero_fhr_scattering_s0 and not rebuilt.zero_up_scattering_s0
    assert torch.equal(run(original, y_st, y_ph, u_stream)["mu_full"], run(rebuilt, y_st, y_ph, u_stream)["mu_full"])

    for keyword in INPUT_ABLATION_KEYWORDS:
        assert keyword in GEOMETRY_KEYS
    agreeing = {"model_config": {"VAE_model": {"zero_fhr_scattering_s0": True}}}
    disagreeing = {"model_config": {"VAE_model": {"zero_fhr_scattering_s0": False}}}
    reconcile_with_checkpoint(
        agreeing, model_kwargs=kwargs, hyper_parameters={}, geometry_keys=GEOMETRY_KEYS
    )
    with pytest.raises(EvalPreconditionUnmet, match="zero_fhr_scattering_s0"):
        reconcile_with_checkpoint(
            disagreeing, model_kwargs=kwargs, hyper_parameters={}, geometry_keys=GEOMETRY_KEYS
        )


def test_the_disclosure_names_the_ablated_coordinate_by_field_channel_and_kind() -> None:
    """What the summary says about the input policy, on every arm."""
    from teb_vae.lag_attn.eval.band_partition import KIND_ORDER0

    both = effective_input_disclosure(
        build_tiny_model(zero_fhr_scattering_s0=True, zero_up_scattering_s0=True)
    )
    none = effective_input_disclosure(build_tiny_model())
    target_only = effective_input_disclosure(
        build_tiny_model(source_disabled=True, zero_fhr_scattering_s0=True)
    )

    assert [(e["stream"], e["field"], e["channel"], e["kind"]) for e in both["ablated_inputs"]] == [
        ("target", "fhr_st", 0, KIND_ORDER0),
        ("source", "up_st", 0, KIND_ORDER0),
    ]
    assert "persistence" in both["policy"]
    assert none["ablated_inputs"] == [] and "no input ablation" in none["policy"]
    assert target_only["zero_fhr_scattering_s0"] and len(target_only["ablated_inputs"]) == 1
