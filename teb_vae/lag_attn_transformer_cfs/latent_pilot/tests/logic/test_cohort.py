r"""Cohort, label and provenance logic on hand-checkable synthetic values.

Every value here is invented and every GUID is obviously artificial. Nothing in this file opens an
HDF5 file, builds a model, loads a checkpoint or touches a GPU: the batches are plain dictionaries
of small numpy arrays, which is exactly what the one identifying pass over a split consumes.

No ``conftest.py`` is needed, and none is added on purpose. Every directory from here up to the
repository root carries an ``__init__.py``, so pytest's prepend import mode inserts the repository
root itself -- the first directory without one -- and the absolute imports below resolve from any
working directory.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn.eval import labels
from teb_vae.lag_attn_transformer_cfs.latent_pilot import data
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError

#: A segment's decimated length. Small: nothing here decodes anything, and the class code is
#: recovered from the target/weight ratio, which needs no particular length.
STEPS = 8


def _batch(rows):
    """Build one collated batch from a list of per-segment descriptions.

    Args:
        rows: Dicts carrying ``guid``, ``epoch``, ``source`` and either ``code`` (a class code with
            a validity weight) or explicit ``target``/``weight`` arrays.

    Returns:
        A batch shaped the way the data module's collation shapes one: stacked arrays for the
        numeric fields, lists of strings for the identifiers.
    """
    targets, weights = [], []
    for row in rows:
        if "target" in row:
            targets.append(np.asarray(row["target"], dtype=np.float64))
            weights.append(np.asarray(row["weight"], dtype=np.float64))
            continue
        weight = np.full(STEPS, float(row.get("weight", 1.0)))
        targets.append(weight * float(row["code"]))
        weights.append(weight)
    return {
        "guid": [row["guid"] for row in rows],
        "epoch": np.asarray([row["epoch"] for row in rows], dtype=np.float64),
        "source_file_basename": [row["source"] for row in rows],
        "target": np.stack(targets),
        "weight": np.stack(weights),
        "cs_label": np.asarray([row.get("cs", 0.0) for row in rows], dtype=np.float64),
        "bg_label": np.asarray([row.get("bg", 1.0) for row in rows], dtype=np.float64),
        "time_from_labor_onset": np.full(len(rows), np.nan),
        "second_stage_onset": np.full(len(rows), np.nan),
    }


def _segments(rows, split="train"):
    """Run the identifying pass over one synthetic batch."""
    return data.segment_frame([_batch(rows)], split=split)


# =============================================================================
# The class code is a ratio, and zero is not a class
# =============================================================================
def test_partially_valid_adverse_segment_is_not_read_as_healthy():
    """A half-valid acidosis segment stores 1.0, which is healthy's code at full validity."""
    frame = _segments([
        {"guid": "SYNTH-A", "epoch": -3600.0, "source": "acidosis_no_cs", "code": 2,
         "weight": 0.5},
    ])
    # The stored target here is literally 1.0 at every step. Read directly it would be healthy.
    assert frame.loc[0, data.CLASS_CODE_COLUMN] == 2
    assert frame.loc[0, labels.CLASS_COLUMN] == "acidosis"


def test_zero_target_and_zero_weight_carry_no_class_rather_than_healthy():
    """Neither an all-zero target nor an all-zero weight may create a class."""
    frame = _segments([
        {"guid": "SYNTH-ZERO-TARGET", "epoch": -1800.0, "source": "healthy_bg_no_cs",
         "target": np.zeros(STEPS), "weight": np.ones(STEPS)},
        {"guid": "SYNTH-ZERO-WEIGHT", "epoch": -1800.0, "source": "healthy_bg_no_cs",
         "target": np.zeros(STEPS), "weight": np.zeros(STEPS)},
    ])
    assert frame[data.CLASS_CODE_COLUMN].isna().all()
    assert frame[labels.CLASS_COLUMN].isna().all()

    recordings = data.recording_frame(frame)
    assert set(recordings[data.EXCLUSION_COLUMN]) == {data.EXCLUDED_NO_CLASS}
    # Excluded, not silently healthy: the outcome stays absent.
    assert recordings[data.OUTCOME_COLUMN].isna().all()


def test_binary_outcome_pools_adverse_and_refuses_an_unknown_code():
    """Healthy is 0, acidosis and HIE are both 1, and nothing else maps at all."""
    assert data.binary_outcome(1) == 0
    assert data.binary_outcome(2) == 1
    assert data.binary_outcome(3) == 1
    assert data.binary_outcome(4) is None
    assert data.binary_outcome(None) is None


# =============================================================================
# Recording grouping and label consistency
# =============================================================================
def test_segments_group_into_one_recording_with_its_own_time_span():
    """Three segments of one recording become one row spanning their epochs."""
    frame = _segments([
        {"guid": "SYNTH-B", "epoch": -9000.0, "source": "healthy_bg_no_cs", "code": 1},
        {"guid": "SYNTH-B", "epoch": -5400.0, "source": "healthy_bg_no_cs", "code": 1},
        {"guid": "SYNTH-B", "epoch": -1800.0, "source": "healthy_bg_no_cs", "code": 1},
    ])
    recordings = data.recording_frame(frame)

    assert len(recordings) == 1
    row = recordings.iloc[0]
    assert row[data.GUID_COLUMN] == "SYNTH-B"
    assert row["n_segments"] == 3
    assert row[data.OUTCOME_COLUMN] == 0
    assert row[data.EXCLUSION_COLUMN] == ""
    assert row["first_epoch"] == -9000.0
    # Epochs are negative, so the maximum is the segment closest to delivery.
    assert row["last_epoch"] == -1800.0


def test_conflicting_class_codes_exclude_the_recording_rather_than_vote():
    """One GUID labelled healthy in one segment and HIE in another is a cohort defect."""
    frame = _segments([
        {"guid": "SYNTH-C", "epoch": -7200.0, "source": "healthy_bg_no_cs", "code": 1},
        {"guid": "SYNTH-C", "epoch": -3600.0, "source": "healthy_bg_no_cs", "code": 3},
        {"guid": "SYNTH-C", "epoch": -1800.0, "source": "healthy_bg_no_cs", "code": 3},
    ])
    recordings = data.recording_frame(frame)
    row = recordings.iloc[0]

    # Two of three segments say HIE. A majority vote would label it HIE; the pilot excludes it.
    assert row[data.EXCLUSION_COLUMN] == data.EXCLUDED_CONFLICTING_CLASS
    assert pd.isna(row[data.CLASS_CODE_COLUMN])
    assert pd.isna(row[data.OUTCOME_COLUMN])
    assert row["class_codes_seen"] == "1,3"
    assert data.exclusion_counts(recordings) == {data.EXCLUDED_CONFLICTING_CLASS: 1}


def test_conflicting_recording_metadata_is_excluded():
    """CS status is a property of a delivery, so two segments may not disagree about it."""
    frame = _segments([
        {"guid": "SYNTH-D", "epoch": -7200.0, "source": "healthy_bg_no_cs", "code": 1, "cs": 0.0},
        {"guid": "SYNTH-D", "epoch": -3600.0, "source": "healthy_bg_no_cs", "code": 1, "cs": 1.0},
    ])
    recordings = data.recording_frame(frame)
    assert recordings.iloc[0][data.EXCLUSION_COLUMN] == data.EXCLUDED_CONFLICTING_METADATA


# =============================================================================
# Splits and grouping
# =============================================================================
def _recordings(rows):
    """Build a recording table directly, for the checks that do not need a loader pass."""
    return pd.DataFrame([
        {
            data.GUID_COLUMN: guid,
            data.SPLIT_COLUMN: split,
            data.CLASS_CODE_COLUMN: code,
            labels.CLASS_COLUMN: labels.class_name(code),
            data.OUTCOME_COLUMN: data.binary_outcome(code),
            labels.SUBGROUP_COLUMN: subgroup,
            "cs_label": False,
            "bg_label": True,
            "n_segments": 2,
            "first_epoch": -7200.0,
            "last_epoch": -1800.0,
            data.EXCLUSION_COLUMN: "",
        }
        for guid, split, code, subgroup in rows
    ])


def test_split_overlap_is_refused_by_guid():
    """A recording consolidated across two splits is named, with the splits it spans."""
    frame = _segments([
        {"guid": "SYNTH-E", "epoch": -3600.0, "source": "healthy_bg_no_cs", "code": 1},
    ], split="train")
    also = _segments([
        {"guid": "SYNTH-E", "epoch": -1800.0, "source": "healthy_bg_no_cs", "code": 1},
    ], split="test")
    recordings = data.recording_frame(pd.concat([frame, also], ignore_index=True))

    assert recordings.iloc[0]["n_splits"] == 2
    with pytest.raises(PilotConfigError, match="more than one split"):
        data.check_split_disjoint(recordings)


def test_disjoint_splits_pass_and_patient_grouping_can_still_refuse(tmp_path):
    """GUID-disjoint splits can still share a patient, and that is refused too."""
    recordings = _recordings([
        ("SYNTH-F1", "train", 1, "healthy_bg_no_cs"),
        ("SYNTH-F2", "test", 2, "acidosis_no_cs"),
    ])
    data.check_split_disjoint(recordings)

    mapping = tmp_path / "patients.json"
    mapping.write_text(json.dumps({"SYNTH-F1": "PATIENT-1", "SYNTH-F2": "PATIENT-1"}))
    grouped, record = data.attach_patient_groups(recordings, patient_map=str(mapping))

    assert record["grouping"] == "patient"
    assert record["n_distinct_groups"] == 1
    with pytest.raises(PilotConfigError, match="more than one split"):
        data.check_split_disjoint(grouped, group_column=data.PATIENT_COLUMN)


def test_absent_patient_map_falls_back_to_guid_and_says_so():
    """Every GUID is its own group, and the record discloses that rather than implying more."""
    recordings = _recordings([("SYNTH-G", "train", 1, "healthy_bg_no_cs")])
    grouped, record = data.attach_patient_groups(recordings, patient_map=None)

    assert grouped[data.PATIENT_COLUMN].tolist() == ["SYNTH-G"]
    assert record["grouping"] == "guid"
    assert record["patient_map_supplied"] is False
    assert "GUID-only" in record["note"]


# =============================================================================
# Exposure provenance
# =============================================================================
def test_unsupplied_provenance_is_unknown_and_not_disjoint(tmp_path):
    """No list means UNKNOWN exposure, which must not support a clean-holdout claim."""
    recordings = _recordings([
        ("SYNTH-H1", "train", 1, "healthy_bg_no_cs"),
        ("SYNTH-H2", "test", 2, "acidosis_no_cs"),
    ])
    record = data.exposure_record(recordings)

    assert record["pretraining"]["known"] is False
    assert record["pretraining"]["n_exposed"] is None
    assert record["clean_holdout_supported"] is False

    empty = tmp_path / "pretraining.txt"
    empty.write_text("# no recording was used\n\n")
    both_known = data.exposure_record(
        recordings,
        pretraining_guids=data.load_guid_list(str(empty)),
        selection_guids=set(),
    )
    # An empty file is a positive claim and is a different answer from no file at all.
    assert both_known["pretraining"]["known"] is True
    assert both_known["clean_holdout_supported"] is True


def test_known_exposure_lists_the_exposed_recordings():
    """A held-out GUID inside the pretraining population is named, not summarised away."""
    recordings = _recordings([
        ("SYNTH-I1", "train", 1, "healthy_bg_no_cs"),
        ("SYNTH-I2", "test", 3, "hie_no_cs"),
    ])
    record = data.exposure_record(
        recordings, pretraining_guids={"SYNTH-I2"}, selection_guids=set()
    )

    assert record["pretraining"]["exposed_guids"] == ["SYNTH-I2"]
    assert record["clean_holdout_supported"] is False


def test_a_missing_guid_list_file_raises_rather_than_reading_as_empty():
    with pytest.raises(FileNotFoundError):
        data.load_guid_list("/nonexistent/synthetic/pretraining_guids.txt")


# =============================================================================
# Class availability and coverage
# =============================================================================
def test_a_split_with_one_binary_class_is_refused():
    """Discrimination is undefined there, and the refusal names the counts."""
    recordings = _recordings([
        ("SYNTH-J1", "test", 1, "healthy_bg_no_cs"),
        ("SYNTH-J2", "test", 1, "healthy_bg_cs"),
    ])
    with pytest.raises(PilotConfigError, match="cannot estimate discrimination"):
        data.require_both_classes(recordings, splits=("test",))

    recordings = _recordings([
        ("SYNTH-J1", "test", 1, "healthy_bg_no_cs"),
        ("SYNTH-J2", "test", 2, "acidosis_no_cs"),
    ])
    data.require_both_classes(recordings, splits=("test",))


def test_excluded_recordings_do_not_count_towards_class_availability():
    """A split whose only adverse recording is excluded cannot estimate discrimination."""
    recordings = _recordings([
        ("SYNTH-K1", "test", 1, "healthy_bg_no_cs"),
        ("SYNTH-K2", "test", 2, "acidosis_no_cs"),
    ])
    recordings.loc[1, data.EXCLUSION_COLUMN] = data.EXCLUDED_CONFLICTING_CLASS
    with pytest.raises(PilotConfigError, match="cannot estimate discrimination"):
        data.require_both_classes(recordings, splits=("test",))


def test_coverage_reports_empty_strata_explicitly():
    """A subgroup this fold has none of gets a zero row rather than no row."""
    segments = _segments([
        {"guid": "SYNTH-L1", "epoch": -3600.0, "source": "healthy_bg_no_cs", "code": 1},
        {"guid": "SYNTH-L2", "epoch": -3600.0, "source": "acidosis_no_cs", "code": 2},
    ])
    recordings = data.recording_frame(segments)
    coverage = data.coverage_summary(segments, recordings)

    subgroups = coverage[coverage["stratum_kind"] == labels.SUBGROUP_COLUMN]
    assert set(subgroups["stratum"]) == set(labels.CANONICAL_SUBGROUPS)
    absent = subgroups[subgroups["stratum"] == "hie_cs"].iloc[0]
    assert absent["n_recordings"] == 0

    overall = coverage[coverage["stratum_kind"] == "all"].iloc[0]
    assert overall["n_recordings"] == 2
    assert overall["n_segments"] == 2
    # Both segments start an hour before delivery, so that is where the last observed time sits.
    assert overall["last_observed_hours_before_delivery_median"] == pytest.approx(1.0)


# =============================================================================
# Loader assembly
# =============================================================================
def _resolved_config():
    """A minimal stand-in for the configuration a training run writes beside its checkpoint."""
    return {
        "general_config": {"batch_size": {"train": 128, "test": 128}},
        "dataset_config": {
            "vae_train_datasets": ["/pretraining/train.hdf5"],
            "vae_test_datasets": ["/pretraining/test.hdf5"],
            "stat_path": "/pretraining/stats.hdf5",
            "dataloader_config": {
                "num_workers": 8,
                "normalize_fields": ["fhr_st", "fhr_ph", "up_st", "up_ph"],
                "dataset_kwargs": {
                    "load_fields": list(data.REQUIRED_LOAD_FIELDS),
                    "epoch_min": -48000,
                    "trim_minutes": 1.0,
                },
            },
        },
    }


def test_loader_config_adds_the_clinical_fields_and_repoints_both_lists():
    """The training contract survives; the identity fields are added and the split is repointed."""
    original = _resolved_config()
    built = data.pilot_loader_config(
        original,
        shards=["/fold_1/train/healthy_bg_no_cs.hdf5"],
        statistics="/fold_1/stats.hdf5",
        epoch_min=-12120.0,
        batch_size=32,
    )
    kwargs = built["dataset_config"]["dataloader_config"]["dataset_kwargs"]

    assert all(name in kwargs["load_fields"] for name in data.CLINICAL_FIELDS)
    assert all(name in kwargs["load_fields"] for name in data.REQUIRED_LOAD_FIELDS)
    assert kwargs["trim_minutes"] == 1.0, "the run's own trim must survive untouched"
    assert kwargs["label"] is None
    assert kwargs["epoch_min"] == -12120.0
    # Both lists move, so no stale pretraining path is one keystroke from being fitted on.
    assert built["dataset_config"]["vae_train_datasets"] == ["/fold_1/train/healthy_bg_no_cs.hdf5"]
    assert built["dataset_config"]["vae_test_datasets"] == ["/fold_1/train/healthy_bg_no_cs.hdf5"]
    assert built["dataset_config"]["stat_path"] == "/fold_1/stats.hdf5"
    assert built["general_config"]["batch_size"]["test"] == 32
    assert built["dataset_config"]["dataloader_config"]["num_workers"] == 0

    # Nothing shared with the input: the checkpoint's own configuration is not edited in place.
    assert original == _resolved_config()


def test_loader_config_refuses_a_run_that_did_not_load_a_required_field():
    """A missing field would surface much later as an absent tensor, so it is refused here."""
    original = _resolved_config()
    fields = original["dataset_config"]["dataloader_config"]["dataset_kwargs"]["load_fields"]
    fields.remove("weight")
    with pytest.raises(PilotConfigError, match="does not load"):
        data.pilot_loader_config(
            original, shards=["/fold_1/train/healthy_bg_no_cs.hdf5"], statistics="/s.hdf5"
        )


def test_the_clinical_fields_are_never_model_inputs():
    """Identity rides on the batch; the forward takes the coefficient streams and nothing else."""
    assert not set(data.CLINICAL_FIELDS) & set(data.MODEL_INPUT_FIELDS)


# =============================================================================
# Serialisation
# =============================================================================
def test_manifest_and_coverage_round_trip_through_the_run_directory(tmp_path):
    """Both tables are written where a later stage and a reader can find them."""
    segments = _segments([
        {"guid": "SYNTH-M1", "epoch": -3600.0, "source": "healthy_bg_no_cs", "code": 1},
        {"guid": "SYNTH-M2", "epoch": -3600.0, "source": "hie_no_cs", "code": 3},
    ])
    recordings = data.recording_frame(segments)
    manifest = data.write_manifest(recordings, tmp_path / "run")
    coverage = data.write_coverage(data.coverage_summary(segments, recordings), tmp_path / "run")

    assert manifest.name == data.MANIFEST_FILENAME
    assert coverage.name == data.COVERAGE_FILENAME
    reloaded = pd.read_csv(manifest)
    assert sorted(reloaded[data.GUID_COLUMN]) == ["SYNTH-M1", "SYNTH-M2"]
    assert sorted(reloaded[data.OUTCOME_COLUMN]) == [0, 1]
