r"""Bag construction, the recording-level metrics, the threshold and the selection rule.

Hand-checkable arrays throughout. Nothing here constructs a classifier, runs an optimizer step or
opens a checkpoint: bag building is a reduction over a small frame, the metrics are arithmetic over
a handful of labels, and the selection rule is a comparison between two records. The fit itself --
which does construct a module and does take gradient steps -- is checked separately, on the
execution machine.

Importing the training module pulls the model-facing half of this package with it, which is the one
weight this file carries; it constructs nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, evaluate, train
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError


class _Extraction:
    """The two attributes :func:`train.build_bags` reads off a real extraction."""

    def __init__(self, frame, arrays):
        self.frame, self.arrays = frame, arrays

    @property
    def retained(self):
        return data.retained(self.frame)


def _extraction(rows, values, *, split="train"):
    """Build an anchor frame and its aligned value matrix from ``(guid, epoch, hours)`` tuples."""
    frame = pd.DataFrame([
        {
            data.SPLIT_COLUMN: split,
            data.GUID_COLUMN: guid,
            data.EPOCH_COLUMN: float(epoch),
            data.ANCHOR_COLUMN: index,
            data.HOURS_COLUMN: float(hours),
            data.EXCLUSION_COLUMN: "",
            data.ROW_COLUMN: index,
        }
        for index, (guid, epoch, hours) in enumerate(rows)
    ])
    return _Extraction(frame, {"mu_post": np.asarray(values, dtype=np.float64)})


def _recordings(outcomes, *, split="train", eligible=True):
    """A recording table carrying only what bag building reads."""
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


def _bags(**overrides):
    """Bags for two healthy and two adverse recordings, one coordinate wide."""
    frame = pd.DataFrame([
        {
            data.GUID_COLUMN: guid,
            data.SPLIT_COLUMN: overrides.get("split", "train"),
            data.OUTCOME_COLUMN: outcome,
            "n_segments": 1,
            "n_anchors": 1,
            data.HOURS_COLUMN: 0.5,
            data.ROW_COLUMN: index,
        }
        for index, (guid, outcome) in enumerate(
            [("H-0", 0), ("H-1", 0), ("A-0", 1), ("A-1", 1)]
        )
    ])
    return train.Bags(
        frame=frame,
        values=np.array([[0.0], [1.0], [2.0], [3.0]]),
        split=overrides.get("split", "train"),
        record={},
    )


# =============================================================================
# Bag construction
# =============================================================================
def test_a_bag_is_one_vector_per_recording_however_many_anchors_it_brought():
    """The reduction is anchors, then segments, then recordings -- one supervised loss each."""
    extraction = _extraction(
        [
            ("DENSE", -3000.0, 0.5), ("DENSE", -3000.0, 0.5), ("DENSE", -3000.0, 0.5),
            ("SPARSE", -3000.0, 0.5),
        ],
        [[0.0], [0.0], [6.0], [4.0]],
    )
    bags = train.build_bags(
        extraction,
        _recordings({"DENSE": 0, "SPARSE": 1}),
        split="train",
        supervised_hours=1.0,
        halflife_hours=0.5,
    )

    assert bags.record["n_recordings"] == 2
    assert bags.values.shape == (2, 1)
    # One segment each, so the recording vector is its anchors' mean: 2 and 4.
    values = dict(zip(bags.guids, bags.values[:, 0].tolist()))
    assert values["DENSE"] == pytest.approx(2.0)
    assert values["SPARSE"] == pytest.approx(4.0)


def test_the_recency_weight_favours_the_later_segment_inside_the_supervised_hour():
    """A 30-minute half-life, applied identically to both classes."""
    extraction = _extraction(
        [("ONE", -3600.0, 1.0), ("ONE", -1800.0, 0.5)],
        [[0.0], [1.0]],
    )
    bags = train.build_bags(
        extraction,
        _recordings({"ONE": 1}),
        split="train",
        supervised_hours=1.0,
        halflife_hours=0.5,
        require_both=False,
    )

    # Weights 2^-2 and 2^-1: the later segment carries twice the earlier one's share.
    assert float(bags.values[0, 0]) == pytest.approx((0.25 * 0.0 + 0.5 * 1.0) / 0.75)


def test_only_anchors_inside_the_supervised_window_enter_the_bag():
    extraction = _extraction(
        [("ONE", -3000.0, 0.5), ("ONE", -9000.0, 2.5)],
        [[1.0], [99.0]],
    )
    bags = train.build_bags(
        extraction,
        _recordings({"ONE": 1}),
        split="train",
        supervised_hours=1.0,
        halflife_hours=0.5,
        require_both=False,
    )

    assert float(bags.values[0, 0]) == pytest.approx(1.0)
    assert bags.record["n_anchors"] == 1


def test_an_ineligible_or_unlabelled_recording_never_reaches_the_fit():
    extraction = _extraction(
        [("KEEP", -3000.0, 0.5), ("INELIGIBLE", -3000.0, 0.5), ("UNLABELLED", -3000.0, 0.5)],
        [[1.0], [2.0], [3.0]],
    )
    recordings = pd.concat([
        _recordings({"KEEP": 1}),
        _recordings({"INELIGIBLE": 0}, eligible=False),
        _recordings({"UNLABELLED": None}),
    ], ignore_index=True)

    bags = train.build_bags(
        extraction,
        recordings,
        split="train",
        supervised_hours=1.0,
        halflife_hours=0.5,
        require_both=False,
    )

    assert bags.guids == ["KEEP"]


def test_bags_are_refused_for_a_split_the_extraction_does_not_hold():
    """A bag table labelled with another split's name would be fitted on the wrong population."""
    extraction = _extraction([("ONE", -3000.0, 0.5)], [[1.0]], split="train")

    with pytest.raises(PilotConfigError, match="bags were requested for 'val'"):
        train.build_bags(
            extraction,
            _recordings({"ONE": 1}, split="val"),
            split="val",
            supervised_hours=1.0,
            halflife_hours=0.5,
            require_both=False,
        )


def test_a_split_carrying_one_class_is_refused_rather_than_fitted():
    extraction = _extraction(
        [("A", -3000.0, 0.5), ("B", -3000.0, 0.5)], [[1.0], [2.0]]
    )

    with pytest.raises(PilotConfigError, match="cannot be estimated"):
        train.build_bags(
            extraction,
            _recordings({"A": 1, "B": 1}),
            split="train",
            supervised_hours=1.0,
            halflife_hours=0.5,
        )


def test_a_split_with_no_eligible_recording_is_refused():
    extraction = _extraction([("ONE", -3000.0, 0.5)], [[1.0]])

    with pytest.raises(PilotConfigError, match="no eligible"):
        train.build_bags(
            extraction,
            _recordings({"ONE": 0}, eligible=False),
            split="train",
            supervised_hours=1.0,
            halflife_hours=0.5,
            require_both=False,
        )


def test_the_prior_probe_reads_the_same_bags_through_the_same_code():
    """``mu_prior`` is pooled by the same reduction, so the probe is a comparable readout."""
    extraction = _extraction([("ONE", -3000.0, 0.5)], [[1.0]])
    extraction.arrays["mu_prior"] = np.array([[7.0]])

    bags = train.build_bags(
        extraction,
        _recordings({"ONE": 1}),
        split="train",
        supervised_hours=1.0,
        halflife_hours=0.5,
        key="mu_prior",
        require_both=False,
    )

    assert float(bags.values[0, 0]) == pytest.approx(7.0)
    assert bags.record["key"] == "mu_prior"


# =============================================================================
# The metrics the selection reads
# =============================================================================
def test_auroc_is_one_on_a_perfect_ranking_and_a_half_on_a_constant_one():
    assert evaluate.auroc([0, 0, 1, 1], [0.0, 1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert evaluate.auroc([0, 0, 1, 1], [3.0, 2.0, 1.0, 0.0]) == pytest.approx(0.0)
    assert evaluate.auroc([0, 0, 1, 1], [1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.5)


def test_auroc_is_undefined_rather_than_zero_when_a_class_is_absent():
    """A split carrying one class cannot estimate discrimination, and nan is what says so."""
    assert np.isnan(evaluate.auroc([1, 1, 1], [0.0, 1.0, 2.0]))
    assert np.isnan(evaluate.auroc([], []))


def test_the_cross_entropy_is_computed_from_logits_and_survives_a_confident_one():
    # A zero logit is an even call: log 2 nats, whichever the label.
    assert evaluate.binary_cross_entropy([1], [0.0]) == pytest.approx(np.log(2.0))
    assert evaluate.binary_cross_entropy([0], [0.0]) == pytest.approx(np.log(2.0))
    # Confidently right costs almost nothing; confidently wrong costs the logit itself.
    assert evaluate.binary_cross_entropy([1], [800.0]) == pytest.approx(0.0)
    assert evaluate.binary_cross_entropy([0], [800.0]) == pytest.approx(800.0)


def test_the_balanced_reduction_averages_the_classes_and_the_plain_one_does_not():
    # Three healthy at zero cost and one adverse at log 2: the plain mean is a quarter of it,
    # the balanced mean a half.
    labels, logits = [0, 0, 0, 1], [-800.0, -800.0, -800.0, 0.0]
    assert evaluate.binary_cross_entropy(labels, logits) == pytest.approx(np.log(2.0) / 4.0)
    assert evaluate.binary_cross_entropy(labels, logits, balanced=True) == pytest.approx(
        np.log(2.0) / 2.0
    )


def test_balanced_accuracy_is_a_half_for_a_rule_that_calls_everything_healthy():
    labels = [0, 0, 0, 1]
    assert evaluate.balanced_accuracy(labels, [-1.0] * 4, threshold=0.0) == pytest.approx(0.5)
    assert evaluate.balanced_accuracy(labels, [-1.0, -1.0, -1.0, 1.0], threshold=0.0) == 1.0
    assert np.isnan(evaluate.balanced_accuracy([1, 1], [0.0, 1.0], threshold=0.0))


# =============================================================================
# The threshold
# =============================================================================
def test_the_threshold_separates_a_separable_validation_split():
    record = evaluate.select_threshold([0, 0, 1, 1], [-2.0, -1.0, 1.0, 2.0])

    assert record["balanced_accuracy"] == pytest.approx(1.0)
    assert -1.0 < record["threshold"] <= 1.0
    assert record["population"] == "validation only"


def test_a_tie_between_thresholds_resolves_towards_sensitivity():
    """Two candidates with the same balanced accuracy: the lower one is chosen."""
    # Labels 0, 0, 1, 1 against scores 0, 1, 1, 2. Cutting at 2 catches one adverse and no
    # healthy (0.75); cutting at 1 catches both adverse and one healthy (0.75 again). The tie is
    # real, and the lower threshold -- the more sensitive rule -- takes it.
    record = evaluate.select_threshold([0, 0, 1, 1], [0.0, 1.0, 1.0, 2.0])

    assert record["balanced_accuracy"] == pytest.approx(0.75)
    assert record["threshold"] == pytest.approx(1.0)
    assert record["n_tied"] == 2
    assert "sensitivity" in record["tie_rule"]


def test_the_threshold_is_always_one_a_finite_score_can_meet():
    """``roc_curve``'s leading infinite threshold is a legal split and a useless decision rule."""
    record = evaluate.select_threshold([0, 1], [0.0, 1.0])

    assert np.isfinite(record["threshold"])


def test_a_validation_split_with_one_class_cannot_choose_a_threshold():
    with pytest.raises(evaluate.GateEvaluationError, match="one binary class"):
        evaluate.select_threshold([1, 1, 1], [0.0, 1.0, 2.0])


# =============================================================================
# The selection rule
# =============================================================================
def test_selection_takes_the_highest_validation_auroc():
    assert train.is_better({"val_auroc": 0.7, "val_bce": 9.0}, {"val_auroc": 0.6, "val_bce": 0.1})
    assert not train.is_better(
        {"val_auroc": 0.5, "val_bce": 0.1}, {"val_auroc": 0.6, "val_bce": 9.0}
    )


def test_a_tie_on_auroc_is_broken_by_the_lower_cross_entropy():
    assert train.is_better({"val_auroc": 0.7, "val_bce": 0.4}, {"val_auroc": 0.7, "val_bce": 0.5})
    assert not train.is_better(
        {"val_auroc": 0.7, "val_bce": 0.6}, {"val_auroc": 0.7, "val_bce": 0.5}
    )


def test_a_full_tie_leaves_the_earlier_step_in_place():
    """The incumbent was reached first, so an identical later step does not replace it."""
    assert not train.is_better(
        {"val_auroc": 0.7, "val_bce": 0.5}, {"val_auroc": 0.7, "val_bce": 0.5}
    )


def test_the_first_finite_result_is_always_an_improvement_and_a_nan_never_is():
    assert train.is_better({"val_auroc": 0.5, "val_bce": 1.0}, None)
    assert not train.is_better({"val_auroc": float("nan"), "val_bce": 0.0}, None)
    assert train.is_better(
        {"val_auroc": 0.1, "val_bce": 9.0}, {"val_auroc": float("nan"), "val_bce": 0.0}
    )


# =============================================================================
# Labels, including the control's permutation
# =============================================================================
def test_the_true_outcomes_are_used_when_no_permutation_is_supplied():
    bags = _bags()

    assert train._labels_for(bags, None).tolist() == [0, 0, 1, 1]
    assert train._labels_for(bags, {"val": {}}).tolist() == [0, 0, 1, 1]


def test_a_permutation_is_applied_per_recording():
    bags = _bags()
    permuted = {"train": {"H-0": 1, "H-1": 0, "A-0": 0, "A-1": 1}}

    assert train._labels_for(bags, permuted).tolist() == [1, 0, 0, 1]


def test_a_permutation_that_omits_a_recording_is_refused():
    bags = _bags()

    with pytest.raises(PilotConfigError, match="omits 2 recording"):
        train._labels_for(bags, {"train": {"H-0": 1, "H-1": 0}})
