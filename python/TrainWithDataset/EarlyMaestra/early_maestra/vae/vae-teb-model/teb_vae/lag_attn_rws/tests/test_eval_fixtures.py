r"""The generated multi-class shards are load-bearing, so their composition gets its own tests.

Every class-, subgroup- and trajectory-aware test in the evaluation suite is written against this
fixture, and each of its properties exists to stop a specific test from passing vacuously:

* **Three clinical classes over four subgroup shards.** With one class every by-class table has
  one group and every contrast self-skips; with as many shards as classes the two groupings
  coincide and a bug that swapped them would be invisible.
* **Fractional validity inside the trimmed window.** ``target`` stores the class code *scaled by*
  ``weight``, so an acidosis step at ``weight = 0.5`` stores ``1.0`` -- exactly what a fully valid
  healthy step stores. Placed at the stored edges instead, ``trim_minutes: 1.0`` would remove
  them and the class-recovery test would run on uniformly valid data.
* **Several segments per recording.** A GUID contributing one segment aggregates to itself, so
  the per-recording reduction would be an identity.
* **A NaN ``time_from_labor_onset``.** The value is NaN wherever the recording is absent from the
  labour-onset table, and it must be preserved rather than dropped.

There is a second variant, and the reason it exists is a property of the first: the plain shards
are **white noise**, so the best possible forecast of them *is* climatology and no fit against
them can beat it -- it can only memorise them. The forecastable variant puts a slowly
drifting level into ``fhr`` and encodes that level in the one feature channel the loader does not
log-transform, which is what turns "the checkpoint forecasts" from an unreachable claim into a
measurable one. Its own tests are at the end of this file, and each of them guards one way the
variant could be present but useless.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from teb_vae.lag_attn_rws.eval._reuse import labels
from teb_vae.lag_attn_rws.tests.conftest import (
    MULTI_CLASS_EDGE_WEIGHT,
    MULTI_CLASS_GUIDS_PER_SHARD,
    MULTI_CLASS_SEGMENTS_PER_GUID,
    MULTI_CLASS_SUBGROUPS,
    subgroup_labels,
)

#: What the loader yields after ``trim_minutes: 1.0`` removes 15 decimated steps from each end.
_TRIMMED_STEPS = 300
_RAW_SAMPLES = _TRIMMED_STEPS * 16


@pytest.fixture(scope="module")
def batches(multi_class_loader) -> list:
    """Every batch, materialised once. The fixture is read-only and the loader is a real one."""
    return list(multi_class_loader)


def _column(batches: list, name: str) -> list:
    values = []
    for batch in batches:
        field = batch[name]
        values.extend(field if isinstance(field, list) else field.tolist())
    return values


# ---------------------------------------------------------------------------
# It loads through the real loader, at the real geometry
# ---------------------------------------------------------------------------
def test_the_shards_load_through_the_real_data_module(batches) -> None:
    expected = len(MULTI_CLASS_SUBGROUPS) * MULTI_CLASS_GUIDS_PER_SHARD * MULTI_CLASS_SEGMENTS_PER_GUID
    assert sum(len(batch["guid"]) for batch in batches) == expected


def test_the_trimmed_geometry_is_the_one_the_raw_index_arithmetic_assumes(batches) -> None:
    r"""The forecast of anchor $t$ starts at raw sample $16(t+1)$, which is only true on the
    trimmed grid. A shard written at the trimmed length would move every anchor by a minute."""
    batch = batches[0]
    assert batch["weight"].shape[1] == _TRIMMED_STEPS
    assert batch["target"].shape[1] == _TRIMMED_STEPS
    assert batch["fhr"].shape[1] == _RAW_SAMPLES
    assert batch["fhr"].shape[1] == 16 * batch["fhr_st"].shape[1]


def test_the_channel_widths_match_the_model_s_data_contract(batches) -> None:
    batch = batches[0]
    assert batch["fhr_st"].shape[2] == 43
    assert batch["fhr_ph"].shape[2] == 66
    assert batch["up_st"].shape[2] == 43
    assert batch["up_ph"].shape[2] == 15


def test_all_five_added_fields_arrive_in_the_batch(batches) -> None:
    """The loader skips a field a shard does not carry, silently, so absence is not an error
    downstream -- it is a missing column that reads as "this cohort has no labels"."""
    for name in ("target", "epoch", "cs_label", "bg_label", "time_from_labor_onset"):
        assert name in batches[0], name


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
def test_the_recovered_classes_are_exactly_the_three_clinical_codes(batches) -> None:
    codes = set()
    for batch in batches:
        for target, weight in zip(batch["target"], batch["weight"]):
            codes.add(labels.clinical_class_code(target, weight))
    assert codes == {1, 2, 3}


def test_more_recordings_than_shards_and_more_segments_than_recordings(batches) -> None:
    guids = _column(batches, "guid")
    assert len(set(guids)) > len(MULTI_CLASS_SUBGROUPS)
    assert len(guids) > len(set(guids))
    per_guid = {guid: guids.count(guid) for guid in set(guids)}
    assert set(per_guid.values()) == {MULTI_CLASS_SEGMENTS_PER_GUID}


def test_no_recording_appears_in_two_shards(batches) -> None:
    """The holdout split is one pool; a GUID in two subgroup files is counted twice."""
    seen: dict = {}
    for batch in batches:
        for guid, source in zip(batch["guid"], batch["source_file_basename"]):
            seen.setdefault(guid, set()).add(source)
    assert all(len(shards) == 1 for shards in seen.values())


def test_both_label_axes_carry_two_values(batches) -> None:
    """The obvious substring rules label the doubly negative subgroup positive on both axes,
    which collapses each by-label table to one group."""
    assert set(_column(batches, "cs_label")) == {True, False}
    assert set(_column(batches, "bg_label")) == {True, False}
    assert subgroup_labels("healthy_no_bg_no_cs") == (0, 0)
    assert subgroup_labels("healthy_bg_cs") == (1, 1)


def test_the_epoch_column_spans_several_hours_and_stays_inside_the_shipped_filter(batches) -> None:
    epochs = np.asarray(_column(batches, "epoch"), dtype=np.float64)
    assert epochs.max() < 0.0, "epoch counts backwards from delivery"
    assert (epochs.max() - epochs.min()) / 3600.0 >= 3.0
    assert epochs.min() >= -48000.0, "outside the shipped epoch_min the loader drops the sample"
    assert np.unique(epochs).size == epochs.size, "a constant epoch bins into one trajectory bin"


def test_at_least_one_recording_has_no_labour_onset_time(batches) -> None:
    onsets = np.asarray(_column(batches, "time_from_labor_onset"), dtype=np.float64)
    assert np.isnan(onsets).any()
    assert np.isfinite(onsets).any(), "an all-NaN column would make every onset test vacuous"


# ---------------------------------------------------------------------------
# The property the class recovery exists for
# ---------------------------------------------------------------------------
def test_a_half_weighted_acidosis_step_stores_exactly_one_and_still_recovers_code_two(
    batches,
) -> None:
    """This is the case that makes reading ``target`` directly wrong, and the case the dataset's
    own ``label`` filter -- exact float equality -- silently drops."""
    found = False
    for batch in batches:
        for target, weight in zip(batch["target"], batch["weight"]):
            code = labels.clinical_class_code(target, weight)
            half = weight == MULTI_CLASS_EDGE_WEIGHT
            if code != 2 or not bool(half.any()):
                continue
            found = True
            assert torch.allclose(target[half], torch.ones(int(half.sum())))
            # Read raw, those steps are indistinguishable from a fully valid healthy step.
            assert labels.clinical_class_code(target[half], torch.ones(int(half.sum()))) == 1
    assert found, "no half-weighted acidosis segment in the fixture; the recovery test is vacuous"


def test_the_fractional_steps_survive_trimming(batches) -> None:
    """At the stored edges they would be trimmed away and every segment would read fully valid."""
    weight = batches[0]["weight"]
    assert float(weight.min()) == 0.0, "the deliberate gap"
    assert bool((weight == MULTI_CLASS_EDGE_WEIGHT).any())
    assert bool((weight == 1.0).any())


def test_no_fixture_binary_is_committed() -> None:
    """The shards are generated into ``tmp_path_factory``; the suite commits no HDF5 of its own."""
    assert not (Path(__file__).resolve().parent / "fixtures").exists()


# ---------------------------------------------------------------------------
# The forecastable variant
#
# The plain shards are white noise, so their optimal forecast *is* climatology and no fit against
# them can beat it -- it can only overfit. The forecastable variant puts a signal in and tells the
# model where it is, which is what makes "the checkpoint forecasts" a reachable statement.
# ---------------------------------------------------------------------------
def _forecastable_arrays(shards: list) -> tuple:
    """Return ``(fhr, fhr_st_channel_0)`` stacked across every forecastable shard."""
    import h5py

    signals, encodings = [], []
    for path in shards:
        with h5py.File(str(path), "r") as handle:
            signals.append(np.asarray(handle["fhr"][()], dtype=np.float64))
            encodings.append(np.asarray(handle["fhr_st"][:, 0, :], dtype=np.float64))
    return np.concatenate(signals), np.concatenate(encodings)


def test_the_forecastable_shards_carry_a_level_the_features_encode(forecastable_shards) -> None:
    r"""The level has to be *both* present in ``fhr`` and visible in the model's input, or the
    model has nothing to read: a signal the features do not carry is exactly as unpredictable as
    white noise, and the fit would again be able to do no better than climatology."""
    fhr, encoding = _forecastable_arrays(forecastable_shards)
    # The feature is on the decimated grid, the signal on the raw one; compare per decimated step.
    per_step = fhr.reshape(fhr.shape[0], encoding.shape[1], -1).mean(axis=2)

    correlation = np.corrcoef(per_step.reshape(-1), encoding.reshape(-1))[0, 1]
    assert correlation > 0.95, (
        f"the encoded channel correlates with the signal at {correlation:.3f}; below this the "
        f"fixture does not tell the model where the answer is"
    )


def test_the_forecastable_level_dominates_the_observation_noise(forecastable_shards) -> None:
    """The skill margin is roughly the ratio of the two variances, so a fixture whose noise
    dominated would make "beats climatology" a coin flip rather than a criterion."""
    fhr, _encoding = _forecastable_arrays(forecastable_shards)
    per_step = fhr.reshape(fhr.shape[0], -1, 16)

    within_step = float(per_step.var(axis=2).mean())
    across_segments = float(per_step.mean(axis=2).var())
    assert across_segments > 10.0 * within_step


def test_the_plain_shards_are_unchanged_by_the_forecastable_flag(multi_class_shards) -> None:
    """Off by default, and off means *identical*: every existing test is written against the white
    noise these shards have always carried."""
    fhr, encoding = _forecastable_arrays(multi_class_shards)

    assert float(np.abs(np.corrcoef(fhr[:, ::16].reshape(-1), encoding.reshape(-1))[0, 1])) < 0.1
    assert float(fhr.std()) > 8.0, "the plain fixture's 10 bpm of white noise"


def test_the_two_forecastable_draws_are_different_recordings(
    fit_shards, forecastable_shards
) -> None:
    """A fit and a hold-out sharing recording identifiers would make "held out" a claim about
    nothing -- and with far more parameters than segments, an in-sample skill score measures
    memorisation."""
    import h5py

    def _guids(paths) -> set:
        found = set()
        for path in paths:
            with h5py.File(str(path), "r") as handle:
                found.update(bytes(value).decode("utf-8") for value in handle["guid"][()])
        return found

    fit, holdout = _guids(fit_shards), _guids(forecastable_shards)

    assert fit and holdout
    assert not fit & holdout


# ---------------------------------------------------------------------------
# The event variant
#
# `with_events=True` injects contractions and decelerations at indices the suite can state
# exactly, so the detectors have a known answer. Three things about the variant are load-bearing
# and are guarded here: the events are actually in the stored signals, the gap is written the way
# the real pipeline writes one, and the default leaves the plain shards untouched.
# ---------------------------------------------------------------------------
def test_the_event_shards_carry_the_injected_shapes_in_the_stored_signals(event_shards) -> None:
    r"""Read straight off the HDF5, before the loader, so a failure here is the generator's rather
    than the trim's. The stored index of a trimmed index is offset by the trim."""
    import h5py

    from .conftest import (
        EVENT_CONTRACTION_AMPLITUDE,
        EVENT_DECELERATION_DEPTH_BPM,
        MULTI_CLASS_TRIM_STEPS,
        injected_event_indices,
    )

    truth = injected_event_indices()
    offset = MULTI_CLASS_TRIM_STEPS * 16
    with h5py.File(str(event_shards[0]), "r") as handle:
        fhr = np.asarray(handle["fhr"][0], dtype=np.float64)
        up = np.asarray(handle["up"][0], dtype=np.float64)

    # 0.7 of the injected depth rather than all of it: the apex sample carries the shape *plus*
    # the observation noise the variant keeps, and a threshold at the full depth is a test of
    # which way that one draw fell.
    for nadir in truth["deceleration_nadir"]:
        assert fhr[nadir + offset] < 140.0 - 0.7 * EVENT_DECELERATION_DEPTH_BPM
    for peak in truth["contraction_peak"]:
        assert up[peak + offset] > 30.0 + 0.7 * EVENT_CONTRACTION_AMPLITUDE


def test_the_event_shards_store_the_weight_gap_as_zero_bpm(event_shards) -> None:
    r"""What the real pipeline stores for a missing sample. After z-scoring that is roughly
    $-11\sigma$ -- the deepest dip in the recording -- which is exactly why a detector must mask by
    ``weight`` rather than by value, and why a fixture that left the signal alone there could not
    test the difference."""
    import h5py

    with h5py.File(str(event_shards[0]), "r") as handle:
        fhr = np.asarray(handle["fhr"][0], dtype=np.float64)
        weight = np.asarray(handle["weight"][0], dtype=np.float64)

    gapped = np.repeat(weight, 16) <= 0.0
    assert gapped.any()
    assert np.all(fhr[gapped] == 0.0)
    assert np.all(fhr[~gapped] != 0.0)


def test_the_plain_shards_are_unchanged_by_the_event_flag(multi_class_shards) -> None:
    """Off by default, and off means identical: the injected events must reach only the shards
    that asked for them, or every existing suite is measuring a different population.

    Asserted on the **mean across samples** at the injected positions rather than on one sample's
    value there. The plain variant's noise is $10\\,$bpm per sample, so a single-sample threshold
    would be a test of which way one draw fell; the mean over every segment of a shard has a
    standard error an order of magnitude smaller, and a real injected dip would move it by the
    whole $25\\,$bpm.
    """
    import h5py

    from .conftest import EVENT_DECELERATION_DEPTH_BPM, injected_event_indices

    truth = injected_event_indices()["deceleration_nadir"]
    offset = 15 * 16
    with h5py.File(str(multi_class_shards[0]), "r") as handle:
        fhr = np.asarray(handle["fhr"][()], dtype=np.float64)

    at_events = float(fhr[:, truth + offset].mean())
    everywhere = float(fhr.mean())
    assert abs(at_events - everywhere) < 0.2 * EVENT_DECELERATION_DEPTH_BPM
    assert np.all(fhr != 0.0), "the plain variant does not write the gap as a value either"
