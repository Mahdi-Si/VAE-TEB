r"""The warm start, and what a checkpoint of this model must carry.

A target-only forecaster and this model share their target encoder, their prior and their decoder,
so a competitively trained one is the right starting point for the base branch -- and the reason the
design wants one is that a weak baseline makes the residual branch look informative by letting it
correct a failure that had nothing to do with the source.

The transfer is the part worth testing, because every way it can go wrong is quiet:

* a **half** transfer, where some tensors arrive and others start randomly, looks exactly like a
  complete one unless the missing names are reported;
* a transfer from a checkpoint trained at **another geometry** produces a model half-initialised
  from a task it will never be scored on, and the shapes that would catch it are only some of them;
* a transfer that left the **source pathway** where the checkpoint put it would start a joint run
  with a source correction it never learned, and the exact zero start would silently not hold.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_slot_transformer_cfs.nets.model import MODEL_KIND, SeqVaeLagResidualTrfCfs
from teb_vae.lag_slot_transformer_cfs.tests.conftest import (
    build_tiny_model,
    tiny_model_kwargs,
    tiny_streams,
)
from teb_vae.lag_slot_transformer_cfs.trainer import (
    SOURCE_PREFIXES,
    TRANSFERABLE_PREFIXES,
    transfer_target_weights,
)


def source_state(seed: int = 5):
    """A stand-in target-only checkpoint: this model's own transferable tensors, perturbed.

    A real target-only forecaster is a different training run, not a different class, so its state
    is this model's minus the source pathway. Built that way here rather than by training one.

    Args:
        seed: Seed for the perturbation.

    Returns:
        The state mapping.
    """
    donor = build_tiny_model(seed=seed)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in donor.state_dict().items():
            if name.startswith(TRANSFERABLE_PREFIXES) and tensor.is_floating_point():
                tensor.add_(torch.randn(tensor.shape, generator=generator) * 0.1)
    return {
        name: tensor.clone()
        for name, tensor in donor.state_dict().items()
        if name.startswith(TRANSFERABLE_PREFIXES)
    }


def test_the_transferable_tensors_arrive_and_are_listed() -> None:
    """Every one of them, by name, so a half transfer is visible rather than plausible."""
    model = build_tiny_model()
    state = source_state()
    report = transfer_target_weights(model, state)

    assert report["missing"] == []
    assert set(report["transferred"]) == set(state)
    for name, incoming in state.items():
        assert torch.equal(model.state_dict()[name], incoming), name


def test_the_source_pathway_is_left_at_zero_after_a_transfer() -> None:
    """So a jointly trained run begins by asserting that the source says nothing.

    Re-zeroed after the copy rather than trusted from construction, which is what keeps it true if
    a future transfer prefix grows to overlap the source pathway.
    """
    model = build_tiny_model()
    with torch.no_grad():
        model.proposal_head.output_proj.weight.fill_(0.7)

    transfer_target_weights(model, source_state())

    assert torch.all(model.proposal_head.output_proj.weight == 0.0)
    assert torch.all(model.proposal_head.output_proj.bias == 0.0)


def test_the_metadata_clock_is_transferred_rather_than_re_zeroed() -> None:
    """It conditions the PRIOR, and the target-only arm of this class trains it.

    The clock is a function of stored position and of nothing else, so it is no more part of the
    source pathway than the target encoder is. Re-zeroing it -- which this transfer used to do --
    would discard a trained target-only component and start the candidate's prior from a state its
    own baseline never occupied, which is precisely the baseline weakness the warm start exists to
    avoid.
    """
    model = build_tiny_model()
    state = source_state()
    with torch.no_grad():
        model.clock_proj.weight.fill_(0.3)

    report = transfer_target_weights(model, state)

    assert any(name.startswith("clock_proj.") for name in report["transferred"])
    assert torch.equal(model.clock_proj.weight, state["clock_proj.weight"])


def test_a_donor_without_a_clock_leaves_it_at_its_constructed_zero() -> None:
    """A target-only model of another architecture carries no such tensor.

    Reported as missing rather than transferred, and left where construction put it -- which is
    zero, and is the right starting point for a conditioning term nothing has trained.
    """
    model = build_tiny_model()
    state = {
        name: tensor
        for name, tensor in source_state().items()
        if not name.startswith("clock_proj.")
    }

    report = transfer_target_weights(model, state)

    assert any(name.startswith("clock_proj.") for name in report["missing"])
    assert torch.all(model.clock_proj.weight == 0.0)


def test_a_target_only_model_can_be_warm_started_from_another() -> None:
    """It is how a second seed of the baseline starts from the first, and it must not fail on a
    proposal head that arm never builds."""
    model = build_tiny_model(source_disabled=True)
    state = {
        name: tensor
        for name, tensor in source_state().items()
        if name in set(model.state_dict())
    }

    report = transfer_target_weights(model, state)

    assert model.proposal_head is None
    assert report["transferred"]
    assert not any(name.startswith(SOURCE_PREFIXES) for name in report["reinitialised"])


def test_a_transferred_model_starts_at_the_prior() -> None:
    """The invariant that makes every later coupling number earned rather than inherited."""
    model = build_tiny_model()
    transfer_target_weights(model, source_state())
    model.eval()

    torch.manual_seed(0)
    outputs = model(*tiny_streams(), anchor_phase=0, anchor_stride=1)
    assert torch.equal(outputs["mu_post"], outputs["mu_prior"])
    assert float(outputs["kld_per_anchor"].sum()) == 0.0


def test_the_source_pathway_is_reported_as_left_as_constructed() -> None:
    """It belongs in its own list, not among the missing: absent and untransferred differ."""
    model = build_tiny_model()
    report = transfer_target_weights(model, source_state())
    left = [name for name in report["reinitialised"] if name.startswith(SOURCE_PREFIXES)]
    assert left, report["reinitialised"]
    assert not any(name.startswith(TRANSFERABLE_PREFIXES) for name in report["reinitialised"])


def test_an_absent_transferable_tensor_is_reported_rather_than_ignored() -> None:
    """A source model built without the persistence residual has no such weight.

    That is legitimate, and it is also what a silent half-transfer looks like, so the two are told
    apart by the name appearing in the report rather than by a success flag.
    """
    model = build_tiny_model()
    state = source_state()
    dropped = "decoder.mean_head.bias"
    assert dropped in state
    del state[dropped]

    report = transfer_target_weights(model, state)
    assert report["missing"] == [dropped]
    assert dropped not in report["transferred"]


def test_a_geometry_disagreement_refuses_rather_than_skipping() -> None:
    """Skipping would leave the model half-initialised from a task it will never be scored on."""
    model = build_tiny_model()
    state = source_state()
    name = "decoder.mean_head.weight"
    state[name] = torch.zeros(state[name].shape[0] + 1, state[name].shape[1])

    with pytest.raises(ValueError, match=name):
        transfer_target_weights(model, state)


def test_a_checkpoint_of_another_architecture_refuses() -> None:
    """Nothing matched, so continuing would train from random weights while the config said not."""
    model = build_tiny_model()
    with pytest.raises(ValueError, match="no tensor transferred"):
        transfer_target_weights(model, {"some.other.model.weight": torch.zeros(3)})


def test_unused_checkpoint_keys_are_reported() -> None:
    """A donor carrying a source pathway of its own would otherwise transfer it in silence."""
    model = build_tiny_model()
    state = source_state()
    state["proposal_head.output_proj.weight"] = torch.ones(1)
    report = transfer_target_weights(model, state)
    # It is neither transferred nor missing: it is a key this model does not take from a donor.
    assert "proposal_head.output_proj.weight" not in report["transferred"]
    assert torch.all(model.proposal_head.output_proj.weight == 0.0)


# =================================================================================================
# The round trip
# =================================================================================================
def test_a_checkpoint_round_trip_reproduces_the_model_and_its_invariants() -> None:
    """Same inference, same source-off equality, from the state dict alone."""
    original = build_tiny_model()
    generator = torch.Generator().manual_seed(41)
    with torch.no_grad():
        original.proposal_head.output_proj.weight.normal_(0.0, 0.3, generator=generator)
    original.eval()

    rebuilt = SeqVaeLagResidualTrfCfs(**tiny_model_kwargs()).eval()
    rebuilt.load_state_dict(original.state_dict())

    torch.manual_seed(0)
    before = original(*tiny_streams(), anchor_phase=0, anchor_stride=1)
    torch.manual_seed(0)
    after = rebuilt(*tiny_streams(), anchor_phase=0, anchor_stride=1)

    for name in ("mu_prior", "mu_post", "mu_full", "kld_per_anchor"):
        assert torch.equal(before[name], after[name]), name

    # And the source-off invariant survives the round trip, which is what every control reads.
    n_anchors = before["mu_prior"].shape[1]
    torch.manual_seed(0)
    silenced = rebuilt(
        *tiny_streams(),
        anchor_phase=0,
        anchor_stride=1,
        selector=torch.zeros(before["mu_prior"].shape[0], n_anchors, rebuilt.n_lags),
    )
    assert torch.equal(silenced["mu_post"], silenced["mu_prior"])


def test_the_mean_only_arm_and_the_full_arm_have_incompatible_state_dicts() -> None:
    """An arm is a different module tree, so a checkpoint cannot cross between them silently."""
    full = build_tiny_model()
    lean = build_tiny_model(mean_only_residual=True)
    with pytest.raises(RuntimeError):
        lean.load_state_dict(full.state_dict())


def test_the_model_kind_is_stamped_on_the_class() -> None:
    """A model kind rather than a version of the lag-attentive one.

    Matching latent and decoder widths do not make the old source fusion semantically compatible,
    and a checkpoint loadable into the wrong architecture would report one model's numbers under
    the other's name.
    """
    assert SeqVaeLagResidualTrfCfs.MODEL_KIND == MODEL_KIND
    assert MODEL_KIND.startswith("fhr_lag_residual")
