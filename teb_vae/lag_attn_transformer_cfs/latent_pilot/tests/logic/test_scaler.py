r"""The frozen training scaler, the fingerprints, and the guards around them.

Hand-checkable arrays throughout. Nothing here builds a model, opens a checkpoint or reads a shard:
the scaler is arithmetic over a small value matrix, and the fingerprint checks are dictionary
comparisons. The extraction pass itself needs the real net and is checked separately, on the
execution machine.

Importing the extraction module pulls the model-facing half of this package with it, which is the
one weight this file carries; it constructs nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, extract
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError


def _frame(rows, split="train"):
    """Build a retained-anchor frame from ``(guid, epoch, anchor)`` tuples.

    Returns:
        The frame, with ``row`` aligned to the value matrix a caller builds alongside it.
    """
    return pd.DataFrame([
        {
            data.SPLIT_COLUMN: split,
            data.GUID_COLUMN: guid,
            data.EPOCH_COLUMN: float(epoch),
            data.ANCHOR_COLUMN: int(anchor),
            data.EXCLUSION_COLUMN: "",
            data.ROW_COLUMN: index,
        }
        for index, (guid, epoch, anchor) in enumerate(rows)
    ])


# =============================================================================
# The hierarchy
# =============================================================================
def test_every_recording_weighs_the_same_however_many_anchors_it_brought():
    """A densely covered recording must not set the scale for the cohort."""
    # One recording contributes ten anchors at 0, the other a single anchor at 10.
    rows = [("SYNTH-DENSE", -2000.0, index) for index in range(10)]
    rows.append(("SYNTH-SPARSE", -1000.0, 0))
    frame = _frame(rows)
    values = np.array([[0.0]] * 10 + [[10.0]])

    scaler = extract.fit_scaler(frame, values)

    # Recording means are 0 and 10, so the population mean is 5 -- not the pooled 10/11.
    assert float(scaler.center[0]) == pytest.approx(5.0)
    # And the spread is the spread between recordings, not within the dense one.
    assert float(scaler.scale[0]) == pytest.approx(5.0)


def test_segments_are_averaged_before_recordings():
    """The three-level order is anchors, then segments, then recordings."""
    frame = _frame([
        ("SYNTH-A", -4000.0, 0),
        ("SYNTH-A", -4000.0, 1),
        ("SYNTH-A", -4000.0, 2),
        ("SYNTH-A", -1000.0, 0),
    ])
    values = np.array([[0.0], [0.0], [0.0], [4.0]])

    scaler = extract.fit_scaler(frame, values)
    # Segment means 0 and 4; the recording is their average, and it is the only recording.
    assert float(scaler.center[0]) == pytest.approx(2.0)


def test_a_second_fit_on_the_same_inputs_gives_the_same_constants():
    """Repeatable, because the reduction is deterministic and sorted."""
    frame = _frame([("SYNTH-A", -2000.0, 0), ("SYNTH-B", -2000.0, 0), ("SYNTH-C", -1000.0, 3)])
    values = np.array([[1.0, -1.0], [3.0, 2.0], [5.0, 6.0]])

    first = extract.fit_scaler(frame, values)
    second = extract.fit_scaler(frame.iloc[::-1].copy(), values)
    assert np.array_equal(first.center, second.center)
    assert np.array_equal(first.scale, second.scale)


# =============================================================================
# The floor and collapse
# =============================================================================
def test_a_barely_varying_coordinate_is_raised_to_the_floor():
    """Otherwise its noise would dominate the standardized geometry."""
    frame = _frame([("SYNTH-A", -2000.0, 0), ("SYNTH-B", -2000.0, 0), ("SYNTH-C", -2000.0, 0)])
    # Second coordinate varies by a millionth of the first.
    values = np.array([[0.0, 0.0], [10.0, 1e-6], [20.0, 2e-6]])

    scaler = extract.fit_scaler(frame, values)

    assert float(scaler.scale[1]) == pytest.approx(scaler.record["floor"])
    assert scaler.record["n_coordinates_at_floor"] == 1
    assert float(scaler.scale[0]) > float(scaler.scale[1])
    assert scaler.record["floor"] >= 1e-3


def test_the_absolute_floor_holds_when_everything_is_tiny():
    """A cohort whose every coordinate barely moves still gets a usable scale."""
    frame = _frame([("SYNTH-A", -2000.0, 0), ("SYNTH-B", -2000.0, 0)])
    values = np.array([[0.0], [1e-9]])

    scaler = extract.fit_scaler(frame, values)
    assert scaler.record["floor"] == pytest.approx(1e-3)
    assert float(scaler.scale[0]) == pytest.approx(1e-3)


def test_a_constant_latent_is_reported_as_collapse_rather_than_scaled():
    """There is nothing to standardize and nothing to discriminate along."""
    frame = _frame([("SYNTH-A", -2000.0, 0), ("SYNTH-B", -2000.0, 0)])
    values = np.array([[3.0, 3.0], [3.0, 3.0]])

    with pytest.raises(extract.LatentCollapse, match="no mu_post coordinate varies"):
        extract.fit_scaler(frame, values)


# =============================================================================
# Split isolation and immutability
# =============================================================================
def test_the_scaler_refuses_anything_but_the_training_split():
    """Constants fitted on validation would carry that population into every comparison."""
    frame = _frame([("SYNTH-A", -2000.0, 0), ("SYNTH-B", -2000.0, 0)])
    frame.loc[1, data.SPLIT_COLUMN] = "val"
    with pytest.raises(PilotConfigError, match="training split alone"):
        extract.fit_scaler(frame, np.array([[1.0], [2.0]]))

    for split in ("val", "test"):
        other = _frame([("SYNTH-A", -2000.0, 0), ("SYNTH-B", -2000.0, 0)], split=split)
        with pytest.raises(PilotConfigError, match="training split alone"):
            extract.fit_scaler(other, np.array([[1.0], [2.0]]))


def test_the_fitted_constants_cannot_be_edited_in_place():
    """A silent rescale after fitting would move every standardized number computed later."""
    frame = _frame([("SYNTH-A", -2000.0, 0), ("SYNTH-B", -2000.0, 0)])
    scaler = extract.fit_scaler(frame, np.array([[0.0], [4.0]]))

    with pytest.raises(ValueError):
        scaler.center[0] = 99.0
    with pytest.raises(ValueError):
        scaler.scale[0] = 99.0


def test_applying_the_scaler_standardizes_and_copies():
    """The transform is (x - m)/s, and it does not touch its input."""
    frame = _frame([("SYNTH-A", -2000.0, 0), ("SYNTH-B", -2000.0, 0)])
    scaler = extract.fit_scaler(frame, np.array([[0.0], [4.0]]))

    values = np.array([[2.0], [6.0]])
    standardized = scaler.apply(values)
    assert np.allclose(standardized, (values - scaler.center) / scaler.scale)
    assert values.tolist() == [[2.0], [6.0]], "the input matrix is untouched"


def test_the_scaler_round_trips_through_a_run_directory(tmp_path):
    """Later stages read the one fitted file rather than refitting."""
    frame = _frame([("SYNTH-A", -2000.0, 0), ("SYNTH-B", -2000.0, 0)])
    scaler = extract.fit_scaler(frame, np.array([[1.0, 2.0], [5.0, 10.0]]))

    extract.save_scaler(scaler, tmp_path)
    reloaded = extract.load_scaler(tmp_path)

    assert np.allclose(reloaded.center, scaler.center)
    assert np.allclose(reloaded.scale, scaler.scale)
    assert reloaded.record["population"] == "train"
    with pytest.raises(ValueError):
        reloaded.scale[0] = 1.0


def test_a_missing_scaler_is_named_rather_than_refitted(tmp_path):
    with pytest.raises(FileNotFoundError, match="fitted once"):
        extract.load_scaler(tmp_path)


# =============================================================================
# Fingerprints and paired keys
# =============================================================================
def _fingerprint(**overrides):
    """A minimal support fingerprint."""
    base = {
        "checkpoint_digest": "abc123",
        "d_z": 64,
        "coverage_floor": 0.9,
        "anchor_stride": 1,
        "preservation_hours": 3.0,
    }
    base.update(overrides)
    return base


def test_incompatible_fingerprints_name_every_field_that_differs():
    extract.check_compatible(_fingerprint(), _fingerprint(), what="the second extraction")

    with pytest.raises(PilotConfigError, match="coverage_floor"):
        extract.check_compatible(
            _fingerprint(), _fingerprint(coverage_floor=0.5), what="the second extraction"
        )
    with pytest.raises(PilotConfigError, match="checkpoint_digest"):
        extract.check_compatible(
            _fingerprint(), _fingerprint(checkpoint_digest="deadbeef"), what="the cached extraction"
        )


def _extraction(rows, fingerprint=None):
    """A minimal extraction carrying only the identity columns the key check reads."""
    frame = _frame(rows)
    return extract.LatentExtraction(
        frame=frame,
        arrays={"mu_post": np.zeros((len(frame), 2), dtype=np.float32)},
        fingerprint=fingerprint or _fingerprint(),
        record={},
    )


def test_a_before_after_pair_must_describe_the_same_anchors_in_the_same_order():
    """A join would drop exactly the rows whose difference matters."""
    rows = [("SYNTH-A", -2000.0, 0), ("SYNTH-A", -2000.0, 1), ("SYNTH-B", -1000.0, 5)]
    extract.assert_same_keys(_extraction(rows), _extraction(rows))

    with pytest.raises(PilotConfigError, match="hold 3 and 2 anchors"):
        extract.assert_same_keys(_extraction(rows), _extraction(rows[:2]))

    shuffled = [rows[0], rows[2], rows[1]]
    with pytest.raises(PilotConfigError, match="disagree at row"):
        extract.assert_same_keys(_extraction(rows), _extraction(shuffled))


def test_a_pair_from_two_checkpoints_is_refused_before_the_keys_are_compared():
    rows = [("SYNTH-A", -2000.0, 0)]
    with pytest.raises(PilotConfigError, match="checkpoint_digest"):
        extract.assert_same_keys(
            _extraction(rows), _extraction(rows, _fingerprint(checkpoint_digest="other"))
        )


# =============================================================================
# Unusable fitting populations
# =============================================================================
def test_a_non_finite_latent_refuses_the_fit_rather_than_writing_a_nan_constant():
    """A NaN constant would reach every standardized number in the run without failing anywhere.

    It survives the collapse guard and the floor (both are ``>`` comparisons, and every
    comparison against NaN is False), lands in ``latent_scaler.json``, and turns the classifier
    logits and the validation AUROC into NaN -- which the fit reads as "not better", so it
    retains epoch zero and finishes with a report no stage of which names the cause.
    """
    frame = _frame([("SYNTH-A", -2000.0, 0), ("SYNTH-B", -2000.0, 0)])
    values = np.array([[1.0, 2.0], [float("nan"), 4.0]])

    with pytest.raises(PilotConfigError, match="not finite at coordinate"):
        extract.fit_scaler(frame, values)

    # Coordinate 1 is fine on its own; the refusal names the offending coordinate, not the run.
    finite = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert float(extract.fit_scaler(frame, finite).center[0]) == pytest.approx(2.0)


def test_an_empty_training_frame_is_named_as_empty_not_as_split_contamination():
    """The empty frame's split set is ``[]``, which the split check would otherwise reject."""
    frame = _frame([]).iloc[0:0]
    with pytest.raises(PilotConfigError, match="no retained training anchor"):
        extract.fit_scaler(frame, np.zeros((0, 2)))


# =============================================================================
# The test-split guard
# =============================================================================
def test_the_test_split_cannot_be_extracted_outside_the_evaluation_stage():
    """Enforced in code, because "we only looked at it at the end" is not demonstrable after."""
    for split in ("train", "val"):
        extract.check_split_allowed(split, allow_test=False)

    with pytest.raises(PilotConfigError, match="only be extracted from the evaluation stage"):
        extract.check_split_allowed("test", allow_test=False)

    extract.check_split_allowed("test", allow_test=True)
