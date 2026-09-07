r"""The plan, the balanced epoch order, and the teacher's own verification rule.

Hand-checkable arrays throughout. Nothing here builds a model, runs a forward or opens a shard: a
plan is a regrouping of an anchor frame, the epoch order is a permutation, and the teacher check is
a comparison against a tolerance. The forward, the gradient and the selection loop are checked
separately, on the execution machine.

Importing the training module pulls the model-facing half of this package with it, which is the one
weight this file carries; it constructs nothing.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, train
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError


class _Extraction:
    """The two attributes :func:`train.build_plans` reads off a real extraction."""

    def __init__(self, frame, arrays):
        self.frame, self.arrays = frame, arrays

    @property
    def retained(self):
        return data.retained(self.frame)


def _extraction(rows, *, split="train", d_z=2):
    """An anchor frame from ``(guid, epoch, anchor, hours)`` tuples, with teacher values.

    The teacher value of row ``i`` is ``[i, -i]``, so a gathered block says which rows it came
    from.
    """
    frame = pd.DataFrame([
        {
            data.SPLIT_COLUMN: split,
            data.GUID_COLUMN: guid,
            data.EPOCH_COLUMN: float(epoch),
            data.ANCHOR_COLUMN: int(anchor),
            data.HOURS_COLUMN: float(hours),
            data.EXCLUSION_COLUMN: "",
            data.ROW_COLUMN: index,
        }
        for index, (guid, epoch, anchor, hours) in enumerate(rows)
    ])
    values = np.stack(
        [np.array([index, -index], dtype=np.float32) for index in range(len(frame))]
    )
    return _Extraction(frame, {"mu_post": values[:, :d_z]})


def _recordings(outcomes, *, split="train", eligible=True):
    """A recording table carrying only what plan building reads."""
    return pd.DataFrame([
        {
            data.GUID_COLUMN: guid,
            data.SPLIT_COLUMN: split,
            data.OUTCOME_COLUMN: outcome,
            data.EXCLUSION_COLUMN: "",
            "eligible": eligible,
        }
        for guid, outcome in outcomes.items()
    ])


def _plans(rows, outcomes, **overrides):
    """Build plans at the protocol's windows unless a test moves them."""
    arguments = {"supervised_hours": 1.0, "halflife_hours": 0.5, "split": "train"}
    arguments.update(overrides)
    return train.build_plans(_extraction(rows), _recordings(outcomes), **arguments)


def _stub_plans(counts):
    """Plans with nothing but a GUID and an outcome, for the epoch-order checks."""
    plans = {}
    for outcome, total in counts.items():
        for index in range(total):
            guid = f"{'ADV' if outcome else 'HLT'}-{index:02d}"
            plans[guid] = train.RecordingPlan(
                guid=guid,
                outcome=outcome,
                segments=(),
                weights=np.ones(1),
                late_segments=(0,),
            )
    return plans


# =============================================================================
# The plan
# =============================================================================
def test_a_plan_carries_one_segment_per_epoch_with_its_anchors_ascending():
    plans = _plans(
        [
            ("ONE", -3000.0, 7, 0.5),
            ("ONE", -3000.0, 5, 0.6),
            ("ONE", -9000.0, 3, 2.5),
        ],
        {"ONE": 1},
    )
    plan = plans["ONE"]

    assert [segment.epoch for segment in plan.segments] == [-9000.0, -3000.0]
    assert plan.segments[1].anchors.tolist() == [5, 7]
    # The teacher travels with the anchors, so the reorder above moved it too.
    assert plan.segments[1].teacher[:, 0].tolist() == [1.0, 0.0]
    assert plan.n_anchors == 3


def test_only_the_supervised_window_is_marked_late_and_the_rest_is_still_preserved():
    """Earlier anchors receive preservation supervision only -- they are not relabelled."""
    plans = _plans(
        [
            ("ONE", -3000.0, 0, 0.5),
            ("ONE", -9000.0, 0, 2.5),
        ],
        {"ONE": 1},
    )
    plan = plans["ONE"]

    late = {segment.epoch: segment.has_late for segment in plan.segments}
    assert late == {-3000.0: True, -9000.0: False}
    assert plan.late_segments == (1,)
    # Both segments are in the preservation term.
    assert plan.n_anchors == 2


def test_the_recency_weights_are_normalised_and_favour_the_later_segment():
    plans = _plans(
        [
            ("ONE", -3600.0, 0, 1.0),
            ("ONE", -1800.0, 0, 0.5),
        ],
        {"ONE": 1},
    )
    weights = plans["ONE"].weights

    assert float(weights.sum()) == pytest.approx(1.0)
    # 2^-2 against 2^-1: the later segment carries twice the earlier one's share.
    assert float(weights[1] / weights[0]) == pytest.approx(2.0)


def test_the_recency_weight_uses_the_median_of_the_included_anchors():
    """Not the segment's start, and not a nominal window's midpoint."""
    plans = _plans(
        [
            ("ONE", -3600.0, 0, 0.9),
            ("ONE", -3600.0, 1, 0.5),
            ("ONE", -3600.0, 2, 0.1),
        ],
        {"ONE": 1},
    )

    assert plans["ONE"].segments[0].median_late_hours == pytest.approx(0.5)


def test_a_recording_that_never_reaches_the_supervised_window_supplies_no_plan():
    """It has no bag, so it cannot supply a classification loss."""
    with pytest.raises(PilotConfigError, match="nothing to supervise"):
        _plans([("ONE", -9000.0, 0, 2.5)], {"ONE": 1})


def test_an_ineligible_or_unlabelled_recording_never_reaches_the_adaptation():
    extraction = _extraction([
        ("KEEP", -3000.0, 0, 0.5),
        ("INELIGIBLE", -3000.0, 0, 0.5),
        ("UNLABELLED", -3000.0, 0, 0.5),
    ])
    recordings = pd.concat([
        _recordings({"KEEP": 1}),
        _recordings({"INELIGIBLE": 0}, eligible=False),
        _recordings({"UNLABELLED": None}),
    ], ignore_index=True)

    plans = train.build_plans(
        extraction, recordings, split="train", supervised_hours=1.0, halflife_hours=0.5
    )

    assert set(plans) == {"KEEP"}


def test_plans_are_refused_for_a_split_the_extraction_does_not_hold():
    with pytest.raises(PilotConfigError, match="plans were requested for 'val'"):
        train.build_plans(
            _extraction([("ONE", -3000.0, 0, 0.5)], split="train"),
            _recordings({"ONE": 1}, split="val"),
            split="val",
            supervised_hours=1.0,
            halflife_hours=0.5,
        )


def test_the_plan_names_the_segments_the_source_must_return_in_order():
    plans = _plans(
        [("ONE", -3000.0, 0, 0.5), ("ONE", -6000.0, 0, 1.5)], {"ONE": 1}
    )

    assert plans["ONE"].epochs == [-6000.0, -3000.0]


# =============================================================================
# The balanced epoch order
# =============================================================================
def test_every_batch_holds_the_same_number_of_each_class():
    plans = _stub_plans({0: 20, 1: 20})

    batches = train.epoch_batches(plans, per_class=4, seed=42, epoch=1)

    for batch in batches:
        assert len(batch) == 8
        assert sum(plans[guid].outcome for guid in batch) == 4


def test_no_recording_appears_twice_inside_one_batch():
    """Balanced sampling repeats the scarce class across batches, never within one."""
    plans = _stub_plans({0: 20, 1: 5})

    for batch in train.epoch_batches(plans, per_class=4, seed=42, epoch=1):
        assert len(set(batch)) == len(batch)


def test_an_epoch_covers_the_larger_class_once_when_the_batch_divides_it():
    """Twenty healthy recordings in blocks of four: one epoch, every one of them, exactly once.

    A pool the block does not divide leaves a remainder for the next reshuffle instead, which is
    the price of never presenting one recording twice inside a batch.
    """
    plans = _stub_plans({0: 20, 1: 5})

    batches = train.epoch_batches(plans, per_class=4, seed=42, epoch=1)
    healthy = [guid for batch in batches for guid in batch if plans[guid].outcome == 0]

    assert len(batches) == 5
    assert sorted(set(healthy)) == sorted(guid for guid in plans if plans[guid].outcome == 0)


def test_a_class_smaller_than_the_batch_shrinks_the_batch_rather_than_repeating():
    """The protocol's rule: reduce the batch and accumulate; never manufacture a distinct patient."""
    plans = _stub_plans({0: 20, 1: 2})

    batches = train.epoch_batches(plans, per_class=4, seed=42, epoch=1)

    for batch in batches:
        assert len(batch) == 4
        assert len(set(batch)) == 4


def test_the_order_is_deterministic_for_a_seed_and_differs_between_epochs():
    plans = _stub_plans({0: 12, 1: 12})

    first = train.epoch_batches(plans, per_class=4, seed=42, epoch=1)
    again = train.epoch_batches(plans, per_class=4, seed=42, epoch=1)
    later = train.epoch_batches(plans, per_class=4, seed=42, epoch=2)
    other = train.epoch_batches(plans, per_class=4, seed=43, epoch=1)

    assert first == again
    assert first != later
    assert first != other


def test_the_scarce_class_is_cycled_rather_than_dropped():
    plans = _stub_plans({0: 20, 1: 5})

    batches = train.epoch_batches(plans, per_class=4, seed=42, epoch=1)
    adverse = Counter(
        guid for batch in batches for guid in batch if plans[guid].outcome == 1
    )

    assert len(adverse) == 5
    assert sum(adverse.values()) == 20


def test_an_epoch_cannot_be_ordered_when_a_class_is_empty():
    with pytest.raises(PilotConfigError, match="needs both"):
        train.epoch_batches(_stub_plans({0: 8, 1: 0}), per_class=4, seed=42, epoch=1)


# =============================================================================
# The cached teacher's verification
# =============================================================================
def test_a_teacher_that_matches_the_first_forward_is_accepted():
    train._verify_teacher(0.0, 1e-4, "SYNTH-0")
    train._verify_teacher(1e-6, 1e-4, "SYNTH-0")


def test_a_teacher_the_first_forward_does_not_reproduce_is_refused():
    """Before the first update the two are the same model, so a difference is a wiring fault."""
    with pytest.raises(PilotConfigError, match="cached teacher does not reproduce"):
        train._verify_teacher(0.1, 1e-4, "SYNTH-0")
    with pytest.raises(PilotConfigError, match="SYNTH-7"):
        train._verify_teacher(float("nan"), 1e-4, "SYNTH-7")


# =============================================================================
# The frozen model as candidate epoch zero
# =============================================================================
def test_the_reference_gate_passes_without_comparing_the_baseline_to_itself():
    gate = train._reference_gate({"mse_full": 0.0, "support_digest": "abc"})

    assert gate.passed
    assert gate.reasons == []
    assert gate.record["rule"] == "reference"
    # A zero baseline is the case where comparing it to itself would say least.
    assert gate.record["baseline_mse_full"] == 0.0
