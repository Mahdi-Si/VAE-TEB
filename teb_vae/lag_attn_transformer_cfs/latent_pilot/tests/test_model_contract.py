r"""Gradient reach and freeze invariants, on the real net.

**These run on the execution machine, not in the synthetic logic subset.** Every test here builds
the actual conv-Transformer causal model, which is exactly what the fast subset is forbidden to do.
They need no clinical data and no GPU: the model is the committed tiny geometry and the batch is the
suite's synthetic stub, so this file is runnable anywhere the repository's environment is installed::

    python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests/test_model_contract.py -q

The fixtures come from the surrounding package's own suite rather than from a second definition
here. That is the point: the tiny keyword set, the warm-up staircase and the stub batch are the
geometry this family's contracts are asserted against, and a pilot that tested its freeze against
its own miniature would be testing a model nobody trains.

What these establish, and why each matters:

* an update **reaches** ``mu_post`` -- a pilot whose gradient never arrived would report the frozen
  model's numbers as the adapted model's, with no error anywhere;
* the frozen half is **bitwise** unchanged, so a before/after difference is attributable to the mean
  output alone;
* fitting the classifier alone leaves every latent coordinate exactly as pretrained, which is why
  that is not this experiment.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_transformer_cfs.latent_pilot import model as pilot_model
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError
from teb_vae.lag_attn_transformer_cfs.tests.conftest import (
    TINY_STRIDE,
    make_stub_batch,
    make_task,
    tiny_warmup_kwargs,
)


@pytest.fixture
def pilot_task():
    """The tiny gated model in its task, frozen the way a pilot fit freezes it."""
    task = make_task(model_kwargs=tiny_warmup_kwargs(anchor_stride=TINY_STRIDE))
    pilot_model.freeze_for_pilot(task.orig_model)
    return task


def _mean_head_state(model):
    """A detached snapshot of the trainable weights."""
    return {
        name: parameter.detach().clone()
        for name, parameter in pilot_model.mean_head_parameters(model)
    }


def _frozen_state(model):
    """A detached snapshot of everything the pilot does not train."""
    allowed = {name for name, _parameter in pilot_model.mean_head_parameters(model)}
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name not in allowed
    }


def _nudge(model, scale: float = 0.05) -> None:
    """Apply a deliberate update to the trainable weights only.

    Stands in for an optimizer step without running one: what these tests establish is which
    weights an update may touch and what moves when they do, not how the step was computed.
    """
    with torch.no_grad():
        for _name, parameter in pilot_model.mean_head_parameters(model):
            parameter.add_(scale * torch.ones_like(parameter))


# =============================================================================
# The freeze
# =============================================================================
def test_only_the_posterior_mean_output_is_trainable(pilot_task):
    """Every trainable tensor lives under the one module, and the count is derived from it."""
    model = pilot_task.orig_model
    record = pilot_model.describe_trainable(model)

    assert record["mean_head"]["names"], "there must be something to train"
    assert all(
        name.startswith(pilot_model.MEAN_HEAD_PATH)
        for name in record["mean_head"]["names"]
    )

    head = pilot_model.mean_head_module(model)
    expected = sum(parameter.numel() for parameter in head.parameters())
    assert record["mean_head"]["n_parameters"] == expected

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert sorted(trainable) == sorted(record["mean_head"]["names"])


def test_the_backbone_stays_in_evaluation_mode(pilot_task):
    """Dropout off, so the frozen forward is deterministic and the teacher is comparable."""
    model = pilot_task.orig_model
    assert model.training is False
    pilot_model.check_pilot_mode(model)

    model.train()
    with pytest.raises(PilotConfigError, match="training mode"):
        pilot_model.check_pilot_mode(model)


def test_a_leaked_gradient_flag_is_refused(pilot_task):
    """A parameter outside the allowlist that regained requires_grad widens the experiment."""
    model = pilot_task.orig_model
    leaked = next(
        parameter for name, parameter in model.named_parameters()
        if not name.startswith(pilot_model.MEAN_HEAD_PATH)
    )
    leaked.requires_grad_(True)
    with pytest.raises(PilotConfigError, match="require gradients"):
        pilot_model.check_pilot_mode(model)


def test_a_frozen_mean_head_is_refused(pilot_task):
    """The opposite failure: an allowlist that trains nothing is a classifier-only run."""
    model = pilot_task.orig_model
    for _name, parameter in pilot_model.mean_head_parameters(model):
        parameter.requires_grad_(False)
    with pytest.raises(PilotConfigError, match="do not require gradients"):
        pilot_model.check_pilot_mode(model)


# =============================================================================
# Gradient reach
# =============================================================================
def test_a_gradient_on_mu_post_reaches_the_mean_head_and_nothing_else(pilot_task):
    """The update this pilot performs is the one it says it performs."""
    model = pilot_task.orig_model
    batch = make_stub_batch()
    outputs = model(*pilot_model.forward_inputs(pilot_task, batch))
    outputs["mu_post"].pow(2).mean().backward()

    for name, parameter in pilot_model.mean_head_parameters(model):
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.any(parameter.grad != 0), f"{name} received an all-zero gradient"

    allowed = {name for name, _parameter in pilot_model.mean_head_parameters(model)}
    for name, parameter in model.named_parameters():
        if name not in allowed:
            assert parameter.grad is None, f"{name} is frozen but accumulated a gradient"


def test_an_update_moves_mu_post_and_leaves_every_invariant_alone(pilot_task):
    """The whole freeze, end to end, in deterministic evaluation on one batch."""
    model = pilot_task.orig_model
    batch = make_stub_batch()
    keys = pilot_model.INVARIANT_OUTPUTS + ("mu_post",)

    before = pilot_model.deterministic_outputs(pilot_task, batch, keys=keys, seed=7)
    frozen_before = _frozen_state(model)
    _nudge(model)
    after = pilot_model.deterministic_outputs(pilot_task, batch, keys=keys, seed=7)

    # mu^q = mu^p + Delta mu, and Delta mu is what moved.
    differences = pilot_model.compare_outputs(before, after)
    assert differences["mu_post"] > 0.0

    # Everything the pilot froze is identical, to the bit.
    pilot_model.assert_invariants(before, after, tolerance=0.0)
    frozen_after = _frozen_state(model)
    assert set(frozen_before) == set(frozen_after)
    for name, tensor in frozen_before.items():
        assert torch.equal(tensor, frozen_after[name]), f"{name} changed under the freeze"


def test_a_moved_invariant_is_refused(pilot_task):
    """The invariant check fails when it should, rather than only passing when it should."""
    batch = make_stub_batch()
    before = pilot_model.deterministic_outputs(pilot_task, batch, seed=7)
    after = dict(before)
    after["mu_prior"] = after["mu_prior"] + 1e-3
    with pytest.raises(PilotConfigError, match="must leave untouched"):
        pilot_model.assert_invariants(before, after)


def test_fitting_a_classifier_alone_leaves_the_latent_exactly_as_pretrained(pilot_task):
    """Why a classifier-only implementation is not this experiment."""
    model = pilot_task.orig_model
    batch = make_stub_batch()
    before = pilot_model.deterministic_outputs(
        pilot_task, batch, keys=("mu_post",), seed=11
    )

    classifier = pilot_model.LatentClassifier(int(model.d_z))
    with torch.no_grad():
        for parameter in classifier.parameters():
            parameter.add_(1.0)

    after = pilot_model.deterministic_outputs(
        pilot_task, batch, keys=("mu_post",), seed=11
    )
    assert torch.equal(before["mu_post"], after["mu_post"]), (
        "training a readout cannot change the representation it reads"
    )


def test_the_teacher_does_not_follow_the_student(pilot_task):
    """The preservation target must not move with the thing being preserved."""
    model = pilot_task.orig_model
    teacher = pilot_model.frozen_teacher(model)
    teacher_before = {
        name: parameter.detach().clone()
        for name, parameter in pilot_model.mean_head_parameters(teacher)
    }

    _nudge(model)

    for name, parameter in pilot_model.mean_head_parameters(teacher):
        assert torch.equal(parameter, teacher_before[name])
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert teacher.training is False


def test_the_teacher_contributes_no_gradient(pilot_task):
    """A teacher whose outputs carried a graph would train the target as well as the student."""
    teacher = pilot_model.frozen_teacher(pilot_task.orig_model)
    batch = make_stub_batch()
    outputs = teacher(*pilot_model.forward_inputs(pilot_task, batch))
    assert outputs["mu_post"].requires_grad is False


# =============================================================================
# The forward seam
# =============================================================================
def test_the_pilot_extracts_at_the_dense_anchor_set(pilot_task):
    """A tiled extraction would cover a fraction of the timeline and hide the gaps."""
    batch = make_stub_batch()
    inputs = pilot_model.forward_inputs(pilot_task, batch)

    assert len(inputs) == 5, "the causal forward takes five positional arguments"
    assert int(inputs[4]) == 1, "dense stride"
    assert int(inputs[3]) == 0, "dense phase"

    # Dense means consecutive: the decoded anchors step by one, with no tile gaps between them.
    outputs = pilot_task.orig_model(*inputs)
    anchors = outputs["anchor_index"][0][outputs["anchor_valid"][0]]
    assert anchors.numel() > 1
    assert torch.all(anchors[1:] - anchors[:-1] == 1)
    assert int(anchors[-1]) == pilot_task.orig_model.anchor_ceiling - 1


def test_a_tiled_geometry_is_refused(pilot_task, monkeypatch):
    """If the seam ever returned the training tiling, the extraction stops rather than shrinks."""
    original = pilot_task._build_forward_inputs

    def _tiled(batch):
        y_st, y_ph, source, _phase, _stride = original(batch)
        return y_st, y_ph, source, 0, int(TINY_STRIDE)

    monkeypatch.setattr(pilot_task, "_build_forward_inputs", _tiled)
    with pytest.raises(PilotConfigError, match="dense"):
        pilot_model.forward_inputs(pilot_task, make_stub_batch())


def test_deterministic_outputs_leave_the_random_stream_where_they_found_it(pilot_task):
    """A diagnostic that advanced the training draw would make a run irreproducible in silence."""
    batch = make_stub_batch()
    torch.manual_seed(1234)
    expected = torch.randn(4)

    torch.manual_seed(1234)
    pilot_model.deterministic_outputs(pilot_task, batch, seed=99)
    assert torch.equal(torch.randn(4), expected)


def test_two_deterministic_reads_of_one_model_agree(pilot_task):
    """Repeated evaluation-mode extraction is reproducible, which every comparison depends on."""
    batch = make_stub_batch()
    first = pilot_model.deterministic_outputs(
        pilot_task, batch, keys=pilot_model.INVARIANT_OUTPUTS + ("mu_post",), seed=3
    )
    second = pilot_model.deterministic_outputs(
        pilot_task, batch, keys=pilot_model.INVARIANT_OUTPUTS + ("mu_post",), seed=3
    )
    assert pilot_model.compare_outputs(first, second) == {key: 0.0 for key in first}


# =============================================================================
# The classifier and the optimizer allowlist
# =============================================================================
def test_the_classifier_standardizes_with_its_own_frozen_constants():
    """The scaler travels inside the module, so a vector cannot be scored under another one."""
    center = torch.tensor([1.0, -2.0])
    scale = torch.tensor([2.0, 4.0])
    classifier = pilot_model.LatentClassifier(2, center=center, scale=scale)

    standardized = classifier.standardize(torch.tensor([[3.0, 2.0]]))
    assert torch.allclose(standardized, torch.tensor([[1.0, 1.0]]))

    state = classifier.state_dict()
    assert "center" in state and "scale" in state, "the constants must survive a save"


def test_the_classifier_refuses_a_degenerate_scale():
    """A zero scale would divide the run into infinities and report them as a latent."""
    with pytest.raises(ValueError, match="positive"):
        pilot_model.LatentClassifier(2, scale=torch.tensor([1.0, 0.0]))
    with pytest.raises(ValueError, match="entries"):
        pilot_model.LatentClassifier(2, center=torch.tensor([1.0, 2.0, 3.0]))


def test_parameter_groups_are_an_allowlist_at_two_learning_rates(pilot_task):
    """Not ``model.parameters()`` filtered by a flag: a later module cannot join the fit."""
    model = pilot_task.orig_model
    classifier = pilot_model.LatentClassifier(int(model.d_z))
    groups = pilot_model.parameter_groups(
        model, classifier, mean_head_lr=1e-4, classifier_lr=1e-3, weight_decay=1e-4
    )

    assert [group["name"] for group in groups] == ["mean_head", "classifier"]
    assert groups[0]["lr"] == 1e-4 and groups[1]["lr"] == 1e-3

    listed = {id(parameter) for group in groups for parameter in group["params"]}
    allowed = {id(parameter) for _name, parameter in pilot_model.mean_head_parameters(model)}
    allowed |= {id(parameter) for parameter in classifier.parameters()}
    assert listed == allowed

    # The frozen scaler is a buffer, so it cannot be optimised into something else.
    buffers = {id(buffer) for buffer in classifier.buffers()}
    assert not (listed & buffers)


# =============================================================================
# Statistics provenance
# =============================================================================
def test_statistics_at_another_trim_are_refused(tmp_path):
    """The loader only warns on this; a warning inside a multi-hour extraction is not a guard."""
    import h5py

    path = tmp_path / "stats.hdf5"
    with h5py.File(path, "w") as handle:
        handle.attrs["trim_minutes"] = 2.0

    with pytest.raises(PilotConfigError, match="trim_minutes"):
        pilot_model.statistics_record(
            path, trim_minutes=1.0, checkpoint_stat_path=None
        )

    record = pilot_model.statistics_record(
        path, trim_minutes=2.0, checkpoint_stat_path=str(path)
    )
    assert record["same_as_checkpoint"] is True
    assert record["trim_minutes"] == 2.0


def test_repointed_statistics_are_disclosed_rather_than_refused(tmp_path):
    """A fold's statistics are legitimately not the pretraining split's."""
    import h5py

    path = tmp_path / "fold_stats.hdf5"
    with h5py.File(path, "w") as handle:
        handle.attrs["trim_minutes"] = 1.0

    record = pilot_model.statistics_record(
        path, trim_minutes=1.0, checkpoint_stat_path="/pretraining/stats.hdf5"
    )
    assert record["same_as_checkpoint"] is False
    assert "repointed" in record["note"]
    assert record["digest"]
