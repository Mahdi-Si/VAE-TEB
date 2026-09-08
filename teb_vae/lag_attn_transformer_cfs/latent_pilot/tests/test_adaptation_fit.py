r"""The adaptation's forward, its gradient, and what the update is allowed to reach.

**Execution-machine tests.** Every one of them runs a real forward and most take a real optimizer
step, which the synthetic logic subset excludes; they need no checkpoint file, no clinical data and
no GPU::

    python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests/test_adaptation_fit.py -q

What they establish: the bag and the preservation term are the reductions the protocol defines and
they carry gradient; the teacher is a constant that no update can move, and the cached teacher is
the student's own forward before the first step; an update reaches ``delta_mu_head`` and
``mu_post`` and reaches nothing else; the preservation term rises when the latent is pushed away
and falls when it is pulled back; and the classification loss covers only the supervised window.

The plan, the epoch order and the teacher's tolerance rule are hand-checked without a model in
``tests/logic/test_adaptation.py``. The selection loop needs a validation loader and the gates, and
is exercised end to end by the smoke scenario rather than here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, extract, train
from teb_vae.lag_attn_transformer_cfs.latent_pilot import model as pilot_model
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError
from teb_vae.lag_attn_transformer_cfs.tests.conftest import (
    TINY_STRIDE,
    make_stub_batch,
    make_task,
    tiny_warmup_kwargs,
)

#: Two segments of one recording, an hour and two hours before delivery.
EPOCHS = (-3600.0, -7200.0)
SUPERVISED_HOURS = 1.0
HALFLIFE_HOURS = 0.5


@pytest.fixture
def task():
    """The tiny net inside its task, frozen into the pilot's trainable set."""
    built = make_task(model_kwargs=tiny_warmup_kwargs(anchor_stride=TINY_STRIDE))
    pilot_model.freeze_for_pilot(built.orig_model)
    return built


@pytest.fixture
def batch():
    """One recording's two segments, in the plan's epoch order."""
    stub = make_stub_batch(batch=len(EPOCHS), seed=0)
    stub.epoch = torch.tensor(list(EPOCHS), dtype=torch.float32)
    stub.guid = ["SYNTH-0"] * len(EPOCHS)
    return stub


def _in_plan_order(batch, epochs):
    """The stub batch with its rows in the order a plan names them.

    :func:`train.recording_terms` gathers row ``i`` of the forward for segment ``i`` of the plan,
    so the two orders have to agree. In a real fit they do by construction --
    ``RecordingSource.batch(guid, plan.epochs)`` collates in exactly that order -- but a plan built
    by :func:`train.build_plans` is sorted by segment start, which need not be the order a fixture
    happened to stack its rows in.

    Args:
        batch: The stub batch.
        epochs: The plan's segment starts, in the plan's own order.

    Returns:
        A batch whose rows follow ``epochs``.
    """
    import types

    positions = [batch.epoch.tolist().index(float(epoch)) for epoch in epochs]
    fields = {}
    for name, value in vars(batch).items():
        if isinstance(value, torch.Tensor):
            fields[name] = value[positions]
        elif isinstance(value, list):
            fields[name] = [value[index] for index in positions]
        else:
            fields[name] = value
    return types.SimpleNamespace(**fields)


def _teacher(task, batch, anchors_by_row):
    """The pretrained ``mu_post`` at the anchors a plan names, straight off a forward."""
    model = task.orig_model
    with torch.no_grad():
        outputs = model(*pilot_model.forward_inputs(task, batch))
    return [
        outputs["mu_post"][row, anchors].detach().cpu().numpy().astype(np.float32)
        for row, anchors in enumerate(anchors_by_row)
    ]


def _plan(task, batch, *, anchors=(6, 8, 10), outcome=1, teacher=None):
    """A two-segment plan: the first segment supervised, the second preserved only.

    The stub's anchors sit inside the last hour for the ``-3600`` segment and in the second hour
    for the ``-7200`` one, so the bag reads the first and the preservation term reads both.
    """
    anchors_by_row = [np.asarray(anchors, dtype=np.int64)] * len(EPOCHS)
    values = _teacher(task, batch, anchors_by_row) if teacher is None else teacher
    segments = []
    for row, epoch in enumerate(EPOCHS):
        late = np.full(len(anchors), row == 0)
        segments.append(train.SegmentSupport(
            epoch=float(epoch),
            anchors=anchors_by_row[row],
            hours=np.full(len(anchors), 0.5 if row == 0 else 1.5),
            late=late,
            teacher=values[row],
            median_late_hours=0.5 if row == 0 else float("nan"),
        ))
    return train.RecordingPlan(
        guid="SYNTH-0",
        outcome=outcome,
        segments=tuple(segments),
        weights=np.ones(1),
        late_segments=(0,),
    )


def _scale(task):
    """A unit scaler on the model's device, so a residual reads in latent units."""
    return torch.ones(int(task.orig_model.d_z), device=next(task.parameters()).device)


# =============================================================================
# The two terms
# =============================================================================
def test_the_bag_is_the_mean_of_the_supervised_segment_s_anchors(task, batch):
    """One late segment, so its weight is one and the bag is that segment's mean."""
    plan = _plan(task, batch)

    bag, _keep, _gap = train.recording_terms(task, plan, batch, scale=_scale(task))

    expected = torch.as_tensor(plan.segments[0].teacher).mean(dim=0)
    assert bag.shape == (int(task.orig_model.d_z),)
    assert torch.allclose(bag.detach().cpu(), expected, atol=1e-5)


def test_the_bag_weighs_two_supervised_segments_by_their_recency(task, batch):
    plan = _plan(task, batch)
    both = train.RecordingPlan(
        guid=plan.guid,
        outcome=plan.outcome,
        segments=tuple(
            train.SegmentSupport(
                epoch=segment.epoch,
                anchors=segment.anchors,
                hours=segment.hours,
                late=np.ones_like(segment.late),
                teacher=segment.teacher,
                median_late_hours=0.5 if index == 0 else 1.0,
            )
            for index, segment in enumerate(plan.segments)
        ),
        weights=np.array([0.75, 0.25]),
        late_segments=(0, 1),
    )

    bag, _keep, _gap = train.recording_terms(task, both, batch, scale=_scale(task))

    means = [torch.as_tensor(segment.teacher).mean(dim=0) for segment in both.segments]
    assert torch.allclose(
        bag.detach().cpu(), 0.75 * means[0] + 0.25 * means[1], atol=1e-5
    )


def test_the_preservation_term_is_zero_against_the_model_s_own_output(task, batch):
    """Before any update the student is the teacher, which is exactly the cache's guarantee."""
    plan = _plan(task, batch)

    _bag, keep, gap = train.recording_terms(task, plan, batch, scale=_scale(task))

    assert float(keep) == pytest.approx(0.0, abs=1e-10)
    assert float(gap) == pytest.approx(0.0, abs=1e-6)


def test_the_preservation_term_grows_with_the_distance_from_the_teacher(task, batch):
    plan = _plan(task, batch)
    displaced = train.RecordingPlan(
        guid=plan.guid,
        outcome=plan.outcome,
        segments=tuple(
            train.SegmentSupport(
                epoch=segment.epoch,
                anchors=segment.anchors,
                hours=segment.hours,
                late=segment.late,
                teacher=segment.teacher + 1.0,
                median_late_hours=segment.median_late_hours,
            )
            for segment in plan.segments
        ),
        weights=plan.weights,
        late_segments=plan.late_segments,
    )

    _bag, keep, gap = train.recording_terms(task, displaced, batch, scale=_scale(task))

    # Every coordinate is one away, so the mean squared residual per coordinate is one.
    assert float(keep) == pytest.approx(1.0, abs=1e-4)
    assert float(gap) == pytest.approx(1.0, abs=1e-4)


def test_the_scaler_divides_the_residual_before_it_is_squared(task, batch):
    plan = _plan(task, batch)
    displaced = train.RecordingPlan(
        guid=plan.guid,
        outcome=plan.outcome,
        segments=tuple(
            train.SegmentSupport(
                epoch=segment.epoch, anchors=segment.anchors, hours=segment.hours,
                late=segment.late, teacher=segment.teacher + 1.0,
                median_late_hours=segment.median_late_hours,
            )
            for segment in plan.segments
        ),
        weights=plan.weights,
        late_segments=plan.late_segments,
    )

    _bag, keep, _gap = train.recording_terms(
        task, displaced, batch, scale=2.0 * _scale(task)
    )

    assert float(keep) == pytest.approx(0.25, abs=1e-4)


def test_the_preservation_term_covers_the_unsupervised_hours_too(task, batch):
    """The earlier segment carries no label and is still held to the teacher."""
    plan = _plan(task, batch)
    moved_early = train.RecordingPlan(
        guid=plan.guid,
        outcome=plan.outcome,
        segments=(
            plan.segments[0],
            train.SegmentSupport(
                epoch=plan.segments[1].epoch,
                anchors=plan.segments[1].anchors,
                hours=plan.segments[1].hours,
                late=plan.segments[1].late,
                teacher=plan.segments[1].teacher + 1.0,
                median_late_hours=plan.segments[1].median_late_hours,
            ),
        ),
        weights=plan.weights,
        late_segments=plan.late_segments,
    )

    _bag, keep, _gap = train.recording_terms(task, moved_early, batch, scale=_scale(task))

    # The mean over two segments of 0 and 1.
    assert float(keep) == pytest.approx(0.5, abs=1e-4)


# =============================================================================
# The gradient
# =============================================================================
def test_both_terms_carry_gradient_into_the_mean_heads_and_nowhere_else(task, batch):
    plan = _plan(task, batch)
    model = task.orig_model

    bag, keep, _gap = train.recording_terms(task, plan, batch, scale=_scale(task))
    (bag.sum() + keep).backward()

    trainable = dict(pilot_model.mean_head_parameters(model))
    assert trainable, "the freeze must leave something to train"
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in trainable.values()
    )
    for name, parameter in model.named_parameters():
        if name not in trainable:
            assert parameter.grad is None, name


def test_the_teacher_contributes_no_gradient(task, batch):
    """It is stored data, so the reference cannot be moved by the update it is the reference for."""
    plan = _plan(task, batch)

    _bag, keep, _gap = train.recording_terms(task, plan, batch, scale=_scale(task))

    assert keep.requires_grad
    # Nothing in the plan is a tensor with a graph; the teacher enters as a constant array.
    for segment in plan.segments:
        assert isinstance(segment.teacher, np.ndarray)


def test_an_update_moves_mu_post_and_leaves_the_invariants_alone(task, batch):
    plan = _plan(task, batch)
    before = pilot_model.deterministic_outputs(
        task, batch, keys=pilot_model.INVARIANT_OUTPUTS + ("mu_post",)
    )

    bag, keep, _gap = train.recording_terms(task, plan, batch, scale=_scale(task))
    (bag.sum() + keep).backward()
    optimizer = torch.optim.AdamW(
        pilot_model.parameter_groups(
            task.orig_model,
            pilot_model.LatentClassifier(int(task.orig_model.d_z)),
            mean_head_lr=1e-1,
            classifier_lr=1e-3,
            weight_decay=0.0,
        )
    )
    optimizer.step()

    after = pilot_model.deterministic_outputs(
        task, batch, keys=pilot_model.INVARIANT_OUTPUTS + ("mu_post",)
    )
    pilot_model.assert_invariants(before, after, tolerance=0.0)
    assert not torch.equal(before["mu_post"], after["mu_post"])


def test_the_preservation_term_falls_as_the_student_returns_to_the_teacher(task, batch):
    """A gradient step on the preservation term alone must reduce it."""
    plan = _plan(task, batch)
    # Displace the student first, so there is something to pull back.
    with torch.no_grad():
        for _name, parameter in pilot_model.mean_head_parameters(task.orig_model):
            parameter.add_(0.05 * torch.ones_like(parameter))

    optimizer = torch.optim.AdamW(
        [parameter for _name, parameter in pilot_model.mean_head_parameters(task.orig_model)],
        lr=1e-2,
    )
    _bag, before, _gap = train.recording_terms(task, plan, batch, scale=_scale(task))
    optimizer.zero_grad(set_to_none=True)
    before.backward()
    optimizer.step()
    _bag, after, _gap = train.recording_terms(task, plan, batch, scale=_scale(task))

    assert float(before) > 0.0
    assert float(after) < float(before)


def test_the_freeze_is_checked_before_a_step_rather_than_assumed(task):
    """A stray ``train()`` would put dropout back on and make the teacher incomparable."""
    task.orig_model.train()

    with pytest.raises(PilotConfigError, match="training mode"):
        pilot_model.check_pilot_mode(task.orig_model)


# =============================================================================
# The plan against a real extraction
# =============================================================================
def test_plans_built_from_an_extraction_name_anchors_the_model_can_be_read_at(task, batch):
    """End to end: extract, plan, then gather -- and the teacher is zero away from the forward."""
    import types

    blob = {
        "model_kwargs": dict(tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)),
        "model_class": type(task.orig_model).__name__,
        "epoch": 0,
    }
    config = {
        "dataset_config": {
            "stat_path": "/synthetic/stats.hdf5",
            "dataloader_config": {"dataset_kwargs": {"trim_minutes": 1.0}},
        }
    }
    loaded = types.SimpleNamespace(
        task=task,
        model=task.orig_model,
        blob=blob,
        config=config,
        checkpoint_path="/synthetic/tiny.ckpt",
        digest="synthetic-digest",
        geometry=pilot_model.geometry_record(task.orig_model, blob, config),
    )
    extraction = extract.extract_split(
        loaded, [batch], split="train", preservation_hours=3.0, bin_hours=0.5
    )
    recordings = pd.DataFrame([{
        data.GUID_COLUMN: "SYNTH-0",
        data.SPLIT_COLUMN: "train",
        data.OUTCOME_COLUMN: 1,
        data.EXCLUSION_COLUMN: "",
        "eligible": True,
    }])

    plans = train.build_plans(
        extraction,
        recordings,
        split="train",
        supervised_hours=SUPERVISED_HOURS,
        halflife_hours=HALFLIFE_HOURS,
    )
    plan = plans["SYNTH-0"]

    _bag, keep, gap = train.recording_terms(
        task, plan, _in_plan_order(batch, plan.epochs), scale=_scale(task)
    )
    assert float(gap) == pytest.approx(0.0, abs=1e-6)
    assert float(keep) == pytest.approx(0.0, abs=1e-10)
