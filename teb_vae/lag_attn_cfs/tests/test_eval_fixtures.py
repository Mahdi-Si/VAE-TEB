r"""The generated causal cohort shards are load-bearing, so their composition gets its own tests.

Every class-, subgroup- and trajectory-aware test in the evaluation suite is written against this
fixture, and each of its properties exists to stop a specific test from passing vacuously:

* **Three clinical classes over eight subgroup shards.** With one class every by-class table has
  one group and every contrast self-skips; with as many shards as classes the two groupings
  coincide and a bug that swapped them would be invisible.
* **Three recordings per shard.** The shared rank tests exclude any group with fewer than
  ``stats.MIN_GROUP_SIZE = 3`` finite values, so at two the by-cohort tables could only ever be
  exercised as skips.
* **Two segments per recording.** A GUID contributing one segment aggregates to itself, so the
  per-recording reduction would be an identity.
* **Fractional validity inside the trimmed window.** ``target`` stores the class code *scaled by*
  ``weight``, so an acidosis step at ``weight = 0.5`` stores ``1.0`` -- exactly what a fully valid
  healthy step stores. Placed at the stored edges instead, ``trim_minutes: 1.0`` would remove them
  and the class-recovery test would run on uniformly valid data.
* **A NaN ``time_from_labor_onset``.** The value is NaN wherever the recording is absent from the
  labour-onset table, and it must be preserved rather than dropped.
* **Real causal coefficients.** The blocks are the real one-sided bank's output over real raw
  segments, not ``rng.standard_normal``. This is the one thing the two-sided cells' generator does
  differently, and it is not a stylistic difference: what a causal shard claims about itself is a
  property of the *transform*, so a synthesised block would carry a fabricated
  ``causal_warmup_steps``, make ``target_warm_frac == 1.0`` vacuous, and break the source-null
  control's premise that zero is the channel mean over the region the model reads.

================================================================================================
THE RULE THIS FIXTURE MAY BE ASSERTED UNDER, WHICH BINDS EVERY TEST IN THE EVALUATION SUITE
================================================================================================

The shards are **eight real raw segments from a single production shard** (``hie_cs.hdf5``),
re-used under distinct identities across eight cohort shards, and -- where a model is involved at
all -- scored by a tiny model trained for a handful of steps.

They are therefore evidence about **schema, shape, finiteness, denominators, cohort membership,
counts, identities and refusals**, and about nothing else.

**No test may assert the sign, magnitude, direction or significance of any clinical or statistical
effect on them.** Not that a forecast gap is positive. Not that one cohort differs from another.
Not that the coupling exceeds the availability clock. Not that a lag peak lands anywhere in
particular. Every one of those is a finding about a model and a population, and this fixture is
neither: it is eight signals wearing forty-eight names.

Where a test needs a direction to be non-vacuous, it **constructs** the condition -- a batch with a
known gap, a zeroed source pathway, a perturbed posterior -- rather than hoping the fixture
supplies it.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget
from teb_vae.lag_attn_cfs.eval._reuse import labels, stats

from .conftest import (
    CAUSAL_C_U,
    CAUSAL_C_Y,
    CAUSAL_PH_WIDTH,
    CAUSAL_ST_WIDTH,
    SHIPPED_BUDGET_STEPS,
    SHIPPED_HORIZON,
    SHIPPED_SEQUENCE_LENGTH,
    SHIPPED_WARMUP_PERIOD,
    causal_config,
)

#: What the loader yields after ``trim_minutes: 1.0`` removes 15 decimated steps from each end.
_TRIMMED_STEPS = SHIPPED_SEQUENCE_LENGTH
_RAW_SAMPLES = _TRIMMED_STEPS * 16

#: What the shipped budget resolves to on causal shards, measured on the committed fixture and
#: reproduced by the generated ones.
_KEPT_TARGET_CHANNELS = 98

#: The class code each subgroup shard carries, so "the eight cover all three classes" is a
#: statement about the generator rather than about whatever it happened to write.
_EXPECTED_CLASS_CODES = {1, 2, 3}


@pytest.fixture(scope="module")
def batches(cohort_loader) -> list:
    """Every batch, materialised once. The fixture is read-only and the loader is a real one."""
    return list(cohort_loader)


def _column(batches: list, name: str) -> list:
    values = []
    for batch in batches:
        field = batch[name]
        values.extend(field if isinstance(field, list) else field.tolist())
    return values


def _resolve_over_all(cohort_shards):
    """Resolve the shipped budget with **every** generated shard configured on both splits.

    All eight rather than the two ``causal_config`` places by default: the resolver validates every
    configured shard rather than only the first, precisely so a shard built at another
    ``causal_warmup_quantile`` cannot sit beside the others and be evaluated against a geometry the
    data no longer has. Passing two of the eight would leave that path untested.
    """
    config = causal_config()
    config["dataset_config"]["vae_train_datasets"] = list(cohort_shards)
    config["dataset_config"]["vae_test_datasets"] = list(cohort_shards)
    resolved = resolve_warmup_budget(config)
    assert resolved is not None
    return resolved


# ---------------------------------------------------------------------------
# The files themselves: what a causal shard has to say about itself
# ---------------------------------------------------------------------------
def test_every_shard_declares_itself_causal_at_the_causal_widths(cohort_shards) -> None:
    """The one refusal that is load-bearing and silent otherwise: the two dataset variants share
    every field name and every dtype, and only the root ``transform`` attribute and the stored
    widths tell them apart. A two-sided shard evaluated as this one would report a causal model on
    coefficients containing their own future."""
    import h5py

    assert len(cohort_shards) == 8
    for path in cohort_shards:
        with h5py.File(path, "r") as handle:
            assert handle.attrs["transform"] == "causal", path
            assert handle["fhr_st"].shape[1] == CAUSAL_ST_WIDTH
            assert handle["fhr_ph"].shape[1] == CAUSAL_PH_WIDTH
            assert handle["up_st"].shape[1] == CAUSAL_ST_WIDTH
            assert handle["up_ph"].shape[1] == CAUSAL_C_U - CAUSAL_ST_WIDTH


#: Surviving source channels under the SINGLE-reference resolution these fixtures build -- the
#: alignment drops the four channels above the target's own clock, and the warm-up budget touches
#: this stream not at all.
_KEPT_SOURCE_CHANNELS = 47


def test_the_shipped_budget_resolves_against_the_generated_shards(cohort_shards) -> None:
    r"""The whole binding, on generated data: shard attributes -> channel tuples -> decoder width.

    The four surviving numbers are not chosen here; they are what $B = 134$ produces against a real
    causal transform, and they are the same four the committed fixture produces. A generator that
    perturbed the coefficients or the warm-up vectors would move them.
    """
    resolved = _resolve_over_all(cohort_shards)

    assert resolved.budget_steps == SHIPPED_BUDGET_STEPS
    assert resolved.target.declared_width == CAUSAL_C_Y
    assert resolved.target.kept_width == _KEPT_TARGET_CHANNELS
    # The source is never gated by the BUDGET -- every source channel is honest well inside it --
    # and it is gated by the ALIGNMENT: four channels sit above the reference, and a shift cannot
    # reach them without reading their own future. The two rules are separate and the survivors
    # here are the second's, so the keep-index is a prefix of the identity rather than the whole of
    # it: the dropped channels are the last four, which is what "the slowest" means on this stream.
    assert resolved.source.kept_width == _KEPT_SOURCE_CHANNELS
    assert resolved.source.keep_index == tuple(
        index for index in range(CAUSAL_C_U) if index not in resolved.source.dropped_index
    )


# ---------------------------------------------------------------------------
# It loads through the real loader, at the real geometry
# ---------------------------------------------------------------------------
def test_the_shards_load_through_the_real_data_module(batches, cohort_shards) -> None:
    from scripts.make_tiny_shard import COHORT_GUIDS_PER_SHARD, COHORT_SEGMENTS_PER_GUID

    expected = len(cohort_shards) * COHORT_GUIDS_PER_SHARD * COHORT_SEGMENTS_PER_GUID
    assert sum(len(batch["guid"]) for batch in batches) == expected


def test_the_trimmed_geometry_is_the_one_the_model_is_built_at(batches) -> None:
    r"""The anchor floor, the warm-up rebase and the sequence length are one geometry: the stored
    warm-up vectors are rebased by exactly this trim, so a shard written at the trimmed length
    would move every channel's validity boundary and the floor with it."""
    batch = batches[0]
    assert batch["weight"].shape[1] == _TRIMMED_STEPS
    assert batch["target"].shape[1] == _TRIMMED_STEPS
    assert batch["fhr"].shape[1] == _RAW_SAMPLES
    assert batch["fhr"].shape[1] == 16 * batch["fhr_st"].shape[1]


def test_the_dense_anchor_set_the_evaluation_decodes_at_is_not_empty(batches) -> None:
    r"""$[F, T - H)$ must hold anchors at the length the LOADER yields.

    Derived from the served length rather than from the config constant, because a fixture written
    at a shorter window would leave the evaluation with no anchors at all and the symptom would be
    an empty table rather than an error.
    """
    served = int(batches[0]["fhr_st"].shape[1])

    assert served == SHIPPED_SEQUENCE_LENGTH
    assert served - SHIPPED_HORIZON - SHIPPED_WARMUP_PERIOD > 0


def test_all_the_clinical_and_identity_fields_arrive_in_the_batch(batches) -> None:
    """The loader skips a field a shard does not carry, silently, so absence is not an error
    downstream -- it is a missing column that reads as "this cohort has no labels". ``guid`` and
    ``epoch`` are the pair the anchor tiling's phase is keyed on."""
    for name in ("target", "epoch", "guid", "cs_label", "bg_label", "time_from_labor_onset"):
        assert name in batches[0], name


def test_normalization_is_active_rather_than_silently_disabled(cohort_loader) -> None:
    """The failure mode the generated statistics file exists to rule out. The reader turns *any*
    stats-schema mismatch into a warning and carries on un-normalised, so a hand-rolled or absent
    stats file leaves every shape correct and every number wrong -- and this model's target arrives
    un-z-scored, which makes its Gaussian NLL meaningless."""
    dataset = cohort_loader.dataset

    assert dataset.normalization_enabled, (
        "the loader fell back to un-normalised data; the statistics file did not pass its schema "
        "check, and nothing but this assertion would have said so"
    )
    for field in ("fhr_st", "fhr_ph", "up_st", "up_ph"):
        assert field in dataset.normalization_stats, field


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
def test_the_recovered_classes_are_exactly_the_three_clinical_codes(batches) -> None:
    codes = set()
    for batch in batches:
        for target, weight in zip(batch["target"], batch["weight"]):
            codes.add(labels.clinical_class_code(target, weight))
    assert codes == _EXPECTED_CLASS_CODES


def test_every_shard_carries_enough_recordings_for_a_cohort_statistic(batches) -> None:
    r"""The shared rank tests exclude any group with fewer than ``MIN_GROUP_SIZE`` finite values,
    so a shard with two recordings makes every by-subgroup contrast a skip rather than a result."""
    per_shard: dict = {}
    for batch in batches:
        for guid, source in zip(batch["guid"], batch["source_file_basename"]):
            per_shard.setdefault(source, set()).add(guid)

    assert len(per_shard) == 8
    assert all(len(guids) >= stats.MIN_GROUP_SIZE for guids in per_shard.values()), per_shard


def test_more_recordings_than_shards_and_more_segments_than_recordings(batches) -> None:
    from scripts.make_tiny_shard import COHORT_SEGMENTS_PER_GUID

    guids = _column(batches, "guid")
    assert len(set(guids)) > 8
    assert len(guids) > len(set(guids))
    per_guid = {guid: guids.count(guid) for guid in set(guids)}
    assert set(per_guid.values()) == {COHORT_SEGMENTS_PER_GUID}


def test_no_recording_appears_in_two_shards(batches) -> None:
    """The holdout split is one pool; a GUID in two subgroup files is counted twice.

    Note what this does NOT claim: the underlying raw segments *are* re-used across shards, because
    the committed fixture holds eight of them and the set needs forty-eight rows. What must not
    repeat is the IDENTITY, because that is what every per-recording aggregation groups on.
    """
    seen: dict = {}
    for batch in batches:
        for guid, source in zip(batch["guid"], batch["source_file_basename"]):
            seen.setdefault(guid, set()).add(source)
    assert all(len(shards) == 1 for shards in seen.values())


def test_the_eight_subgroups_are_the_canonical_ones(batches) -> None:
    """Read through the shared labelling rather than off the filenames, so a basename the cohort
    ordering does not know would fail here rather than sorting silently to the end of every table."""
    resolved = {
        labels.subgroup_of(source)
        for batch in batches
        for source in batch["source_file_basename"]
    }

    assert resolved == set(labels.CANONICAL_SUBGROUPS)


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
    from scripts.make_tiny_shard import COHORT_EDGE_WEIGHT

    found = False
    for batch in batches:
        for target, weight in zip(batch["target"], batch["weight"]):
            code = labels.clinical_class_code(target, weight)
            half = weight == COHORT_EDGE_WEIGHT
            if code != 2 or not bool(half.any()):
                continue
            found = True
            assert torch.allclose(target[half], torch.ones(int(half.sum())))
            # Read raw, those steps are indistinguishable from a fully valid healthy step.
            assert labels.clinical_class_code(target[half], torch.ones(int(half.sum()))) == 1
    assert found, "no half-weighted acidosis segment in the fixture; the recovery test is vacuous"


def test_the_validity_profile_survives_trimming(batches) -> None:
    """At the stored edges the fractional steps would be trimmed away and every segment would read
    fully valid; without the gap every mask assertion in the suite would hold vacuously."""
    from scripts.make_tiny_shard import COHORT_EDGE_WEIGHT

    weight = batches[0]["weight"]
    assert float(weight.min()) == 0.0, "the deliberate gap"
    assert bool((weight == COHORT_EDGE_WEIGHT).any())
    assert bool((weight == 1.0).any())
