r"""Held-out metrics, the paired cluster bootstrap, the strata and the label permutation.

Hand-checkable numbers throughout. Nothing here builds a model, runs a forward or opens a shard: a
metric is arithmetic over a handful of labels, a bootstrap is a seeded resample of a small index,
and the permutation is a rearrangement of a column. The fits those numbers describe are checked
separately, on the execution machine.

Importing the evaluation and training modules pulls the model-facing half of this package with them,
which is the one weight this file carries; it constructs nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn.eval import labels
from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, evaluate, train
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError


def _cohort(n_healthy=8, n_adverse=8, *, patients=False, subgroup="acidosis"):
    """A held-out table: labels, GUIDs, flags, and two models' logits.

    ``before`` ranks the adverse recordings only weakly; ``after`` ranks them perfectly. The
    difference is what the paired comparison is supposed to see.
    """
    outcomes = [0] * n_healthy + [1] * n_adverse
    rows = []
    for index, outcome in enumerate(outcomes):
        rows.append({
            data.GUID_COLUMN: f"REC-{index:02d}",
            data.SPLIT_COLUMN: "test",
            data.OUTCOME_COLUMN: outcome,
            labels.CLASS_COLUMN: "healthy" if outcome == 0 else subgroup,
            "bg_label": True if outcome == 1 else (index % 2 == 0),
            "cs_label": index % 3 == 0,
        })
    frame = pd.DataFrame(rows)
    # Perfect after, deliberately imperfect before: one healthy recording outranks one adverse.
    after = np.asarray(outcomes, dtype=np.float64) * 2.0 - 1.0
    before = after.copy()
    before[0], before[n_healthy] = 1.0, -1.0
    mapping = (
        {guid: f"P-{position // 2:02d}" for position, guid in enumerate(frame[data.GUID_COLUMN])}
        if patients else None
    )
    return frame, {"before": before, "after": after}, mapping


# =============================================================================
# The recording-level readout
# =============================================================================
def test_the_readout_reports_the_prevalence_average_precision_must_be_read_against():
    frame, columns, _patients = _cohort(n_healthy=9, n_adverse=3)

    measured = evaluate.recording_metrics(
        frame[data.OUTCOME_COLUMN], columns["after"], threshold=0.0
    )

    assert measured["auroc"] == pytest.approx(1.0)
    assert measured["average_precision"] == pytest.approx(1.0)
    assert measured["balanced_accuracy"] == pytest.approx(1.0)
    assert measured["prevalence"] == pytest.approx(0.25)
    assert measured["chance_average_precision"] == pytest.approx(0.25)
    assert measured["chance_auroc"] == 0.5
    assert measured["n_recordings"] == 12
    assert (measured["n_healthy"], measured["n_adverse"]) == (9, 3)


def test_average_precision_is_the_prevalence_for_a_useless_ranker():
    """Its chance level moves with the cohort, which is why it never travels alone."""
    outcomes = [0] * 9 + [1] * 3

    assert evaluate.average_precision(outcomes, [1.0] * 12) == pytest.approx(0.25)
    assert np.isnan(evaluate.average_precision([1, 1, 1], [0.0, 1.0, 2.0]))


def test_the_threshold_is_the_one_it_was_given_and_not_one_found_here():
    """Held-out balanced accuracy is read at validation's threshold, never at this split's best."""
    frame, columns, _patients = _cohort()
    outcomes = frame[data.OUTCOME_COLUMN]

    good = evaluate.recording_metrics(outcomes, columns["after"], threshold=0.0)
    bad = evaluate.recording_metrics(outcomes, columns["after"], threshold=99.0)

    assert good["balanced_accuracy"] == pytest.approx(1.0)
    assert bad["balanced_accuracy"] == pytest.approx(0.5)
    assert bad["auroc"] == pytest.approx(1.0), "ranking does not depend on the threshold"


# =============================================================================
# The paired cluster bootstrap
# =============================================================================
def _bootstrap(frame, columns, patients=None, **overrides):
    arguments = {
        "guids": frame[data.GUID_COLUMN].tolist(),
        "thresholds": {name: 0.0 for name in columns},
        "patients": patients,
        "resamples": 200,
        "seed": 42,
    }
    arguments.update(overrides)
    return evaluate.paired_bootstrap(frame[data.OUTCOME_COLUMN], columns, **arguments)


def test_the_point_estimate_is_the_metric_on_the_full_sample():
    """The draws estimate the spread; they do not decide the value."""
    frame, columns, _patients = _cohort()

    record = _bootstrap(frame, columns)

    assert record["models"]["after"]["auroc"]["point"] == pytest.approx(1.0)
    assert record["models"]["before"]["auroc"]["point"] == pytest.approx(
        evaluate.auroc(frame[data.OUTCOME_COLUMN], columns["before"])
    )


def test_the_paired_difference_is_reported_with_its_own_interval():
    frame, columns, _patients = _cohort()

    record = _bootstrap(frame, columns)
    paired = record["paired"]["after - before"]["auroc"]

    assert paired["point"] == pytest.approx(
        record["models"]["after"]["auroc"]["point"]
        - record["models"]["before"]["auroc"]["point"]
    )
    assert paired["lo"] <= paired["point"] <= paired["hi"]
    assert paired["lo"] >= 0.0, "the after model dominates on every draw of this fixture"


def test_the_interval_is_reproducible_from_its_seed():
    """An unseeded generator would fail this, which is the failure worth catching.

    Two *different* seeds are deliberately not asserted to differ: AUROC on sixteen recordings takes
    values a sixty-fourth apart, so two draws landing on the same percentile bounds is arithmetic
    rather than a defect.
    """
    frame, columns, _patients = _cohort()

    first = _bootstrap(frame, columns, seed=42)
    again = _bootstrap(frame, columns, seed=42)

    assert first["models"]["before"]["auroc"] == again["models"]["before"]["auroc"]
    assert first["paired"]["after - before"]["auroc"] == again["paired"]["after - before"]["auroc"]
    assert first["seed"] == 42


def test_patients_are_resampled_when_a_mapping_exists_and_guids_otherwise():
    frame, columns, patients = _cohort(patients=True)

    clustered = _bootstrap(frame, columns, patients=patients)
    flat = _bootstrap(frame, columns)

    assert clustered["grouping"] == "patient"
    assert clustered["n_units"] == 8
    assert flat["grouping"] == "guid"
    assert flat["n_units"] == 16
    # GUID-only grouping is disclosed rather than left to be inferred.
    assert "grouping_disclosure" in flat
    assert "grouping_disclosure" not in clustered


def test_a_patient_carrying_both_outcomes_forms_a_stratum_of_its_own():
    """Forcing it into one class would resample it as something it is not."""
    frame, columns, _patients = _cohort(n_healthy=4, n_adverse=4)
    # Pair each healthy recording with an adverse one under a single patient.
    guids = frame[data.GUID_COLUMN].tolist()
    mixed = {guid: f"P-{position % 4:02d}" for position, guid in enumerate(guids)}

    record = _bootstrap(frame, columns, patients=mixed)

    assert record["grouping"] == "patient"
    assert record["strata"] == {"0/1": 4}


def test_every_stratum_keeps_its_size_so_both_classes_survive_each_draw():
    frame, columns, _patients = _cohort(n_healthy=12, n_adverse=4)

    record = _bootstrap(frame, columns, resamples=100)

    assert record["strata"] == {"0": 12, "1": 4}
    assert record["n_undefined_draws"] == 0
    assert record["models"]["after"]["auroc"]["n_draws"] == 100


def test_a_cohort_too_small_to_resample_is_refused_rather_than_given_an_interval():
    frame, columns, _patients = _cohort(n_healthy=1, n_adverse=1)

    with pytest.raises(evaluate.GateEvaluationError, match="reproduces the sample"):
        _bootstrap(frame, columns)


def test_two_models_scored_on_different_populations_are_refused():
    """The comparison is paired only if both were read on the same recordings, in order."""
    frame, columns, _patients = _cohort()
    columns = dict(columns, after=columns["after"][:-1])

    with pytest.raises(evaluate.GateEvaluationError, match="same recordings"):
        _bootstrap(frame, columns)


def test_the_record_states_what_the_interval_is_conditional_on():
    frame, columns, _patients = _cohort()

    record = _bootstrap(frame, columns)

    assert "not variation across training runs" in record["note"]
    assert record["confidence"] == pytest.approx(0.95)
    assert record["resamples"] == 200


# =============================================================================
# The prespecified strata
# =============================================================================
def test_the_healthy_bg_restriction_keeps_every_adverse_recording():
    """Healthy-no-BG controls differ in ascertainment; the adverse group is not re-selected."""
    frame, _columns, _patients = _cohort(n_healthy=8, n_adverse=8)

    mask = evaluate.subgroup_rows(frame, "healthy_bg")

    outcome = np.asarray(frame[data.OUTCOME_COLUMN])
    assert mask[outcome == 1].all()
    assert mask.sum() < len(frame)


def test_a_class_contrast_keeps_the_same_healthy_controls():
    frame, _columns, _patients = _cohort(n_healthy=8, n_adverse=8, subgroup="hie")

    hie = evaluate.subgroup_rows(frame, "hie")
    acidosis = evaluate.subgroup_rows(frame, "acidosis")
    outcome = np.asarray(frame[data.OUTCOME_COLUMN])

    assert hie.sum() == len(frame)
    # No acidosis in this cohort, so the contrast is the healthy controls alone.
    assert acidosis.sum() == int((outcome == 0).sum())


def test_the_subgroup_table_reports_every_model_on_every_stratum_at_once():
    """Produced together and in a fixed order, so no subgroup can be promoted afterwards."""
    frame, columns, _patients = _cohort()

    table = evaluate.subgroup_table(
        frame, columns, thresholds={"before": 0.0, "after": 0.0}
    )

    assert list(table["subgroup"].unique()) == list(evaluate.SUBGROUPS)
    assert set(table["model"]) == {"before", "after"}
    assert len(table) == len(evaluate.SUBGROUPS) * 2


def test_a_stratum_that_cannot_estimate_discrimination_says_so_rather_than_reporting_a_number():
    frame, columns, _patients = _cohort(n_healthy=8, n_adverse=8, subgroup="acidosis")

    table = evaluate.subgroup_table(
        frame, columns, thresholds={"before": 0.0, "after": 0.0}, subgroups=("hie",)
    )

    # No HIE recording in this cohort, so the contrast is healthy-only and undefined.
    assert not table["estimable"].any()
    assert table["auroc"].isna().all()


def test_an_unknown_stratum_is_refused():
    frame, _columns, _patients = _cohort()

    with pytest.raises(evaluate.GateEvaluationError, match="prespecified"):
        evaluate.subgroup_rows(frame, "whichever-looks-best")


def test_the_coverage_contrast_describes_both_groups_without_adjusting_anything():
    frame, _columns, _patients = _cohort(n_healthy=4, n_adverse=4)
    frame["last_anchor_hours"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    frame["n_late_segments"] = [2] * 8
    frame["n_late_anchors"] = [10] * 8

    table = evaluate.coverage_contrast(frame)

    assert list(table["class"]) == ["healthy", "adverse"]
    assert table.loc[0, "last_anchor_hours_median"] == pytest.approx(0.25)
    assert table.loc[1, "last_anchor_hours_median"] == pytest.approx(0.65)
    assert list(table["n_recordings"]) == [4, 4]


# =============================================================================
# What the controls do and do not establish
# =============================================================================
def test_one_shuffled_fit_is_never_called_a_permutation_p_value():
    disclosure = evaluate.control_disclosure(n_control_fits=1, prior_probe=True)

    assert disclosure["permutation_p_value"] is False
    assert "not a permutation p-value" in disclosure["shuffled_label_note"]


def test_switching_off_the_prior_probe_withdraws_the_combined_branch_claim():
    with_probe = evaluate.control_disclosure(n_control_fits=1, prior_probe=True)
    without = evaluate.control_disclosure(n_control_fits=1, prior_probe=False)

    assert with_probe["combined_branch_claim_supported"] is True
    assert without["combined_branch_claim_supported"] is False
    assert "makes no claim" in without["prior_probe_note"]


# =============================================================================
# The label permutation
# =============================================================================
def _recordings():
    """Train and validation recordings, both carrying each class."""
    rows = []
    for split, count in (("train", 12), ("val", 8), ("test", 6)):
        for index in range(count):
            rows.append({
                data.GUID_COLUMN: f"{split.upper()}-{index:02d}",
                data.SPLIT_COLUMN: split,
                data.OUTCOME_COLUMN: index % 2,
                data.EXCLUSION_COLUMN: "",
                "eligible": True,
            })
    return pd.DataFrame(rows)


def test_the_permutation_keeps_each_split_s_class_counts():
    """It is a rearrangement, so the null has the cohort's own prevalence."""
    permuted, record = train.permute_outcomes(_recordings(), seed=42)

    for split in ("train", "val"):
        original = _recordings()
        before = original[original[data.SPLIT_COLUMN] == split][data.OUTCOME_COLUMN]
        after = permuted[permuted[data.SPLIT_COLUMN] == split][data.OUTCOME_COLUMN]
        assert sorted(before.tolist()) == sorted(int(value) for value in after.tolist())
    assert record["per_split"]["train"]["n_changed"] > 0


def test_the_test_split_is_left_alone_so_the_control_meets_true_labels_once():
    permuted, record = train.permute_outcomes(_recordings(), seed=42)
    original = _recordings()

    held_out = permuted[permuted[data.SPLIT_COLUMN] == "test"][data.OUTCOME_COLUMN].tolist()
    expected = original[original[data.SPLIT_COLUMN] == "test"][data.OUTCOME_COLUMN].tolist()

    assert [int(value) for value in held_out] == expected
    assert record["test_split_permuted"] is False


def test_permuting_the_held_out_labels_is_refused():
    with pytest.raises(PilotConfigError, match="never permuted"):
        train.permute_outcomes(_recordings(), seed=42, splits=("train", "test"))


def test_the_two_splits_are_permuted_independently():
    """Validation's draw must not be determined by the training one."""
    permuted, _record = train.permute_outcomes(_recordings(), seed=42)

    train_labels = permuted[permuted[data.SPLIT_COLUMN] == "train"][data.OUTCOME_COLUMN]
    val_labels = permuted[permuted[data.SPLIT_COLUMN] == "val"][data.OUTCOME_COLUMN]
    # Different lengths make an accidental identity between the two impossible to read as design;
    # what is asserted is that both moved, each on its own draw.
    original = _recordings()
    for split, permuted_values in (("train", train_labels), ("val", val_labels)):
        before = original[original[data.SPLIT_COLUMN] == split][data.OUTCOME_COLUMN].tolist()
        assert [int(v) for v in permuted_values.tolist()] != before


def test_the_permutation_is_reproducible_from_the_seed():
    first, _record = train.permute_outcomes(_recordings(), seed=42)
    again, _record = train.permute_outcomes(_recordings(), seed=42)
    other, _record = train.permute_outcomes(_recordings(), seed=43)

    assert first[data.OUTCOME_COLUMN].tolist() == again[data.OUTCOME_COLUMN].tolist()
    assert first[data.OUTCOME_COLUMN].tolist() != other[data.OUTCOME_COLUMN].tolist()


def test_the_permutation_leaves_the_original_table_untouched():
    original = _recordings()
    before = original[data.OUTCOME_COLUMN].tolist()

    train.permute_outcomes(original, seed=42)

    assert original[data.OUTCOME_COLUMN].tolist() == before
