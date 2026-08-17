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

from pathlib import Path

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
#: reproduced by the generated ones: 98 of 102 target channels survive, all 51 source channels are
#: kept, and the four dropped target channels wait these many steps in the trimmed coordinates.
_KEPT_TARGET_CHANNELS = 98
_DROPPED_WARMUPS = (162, 194, 233, 278)

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


def test_every_shard_carries_the_per_block_warm_up_and_delay_attributes(cohort_shards) -> None:
    """These are what the whole channel budget is resolved from, and they exist only because the
    blocks came out of the real one-sided bank. A synthesised shard would carry a number here that
    described nothing."""
    import h5py

    with h5py.File(cohort_shards[0], "r") as handle:
        for block in ("fhr_st", "fhr_ph", "up_st", "up_ph"):
            assert "causal_warmup_steps" in handle[block].attrs, block
            assert "causal_delay_s" in handle[block].attrs, block
        # The phase blocks alone carry the channel-selection provenance the band map is read from.
        assert "sel_xi_i_hz" in handle["fhr_ph"].attrs
        assert "sel_xi_i_hz" in handle["up_ph"].attrs


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
    # The source is never gated: its keep-index is the identity by construction, not by arithmetic
    # that happens to keep everything.
    assert resolved.source.kept_width == CAUSAL_C_U
    assert resolved.source.keep_index == tuple(range(CAUSAL_C_U))


def test_the_four_dropped_target_channels_are_the_ones_the_design_names(cohort_shards) -> None:
    r"""$W' = 162, 194, 233, 278$ against a budget of $134$ -- the four slowest ``fhr_st`` filters.
    They are named rather than counted because a budget that dropped four *other* channels would
    satisfy every count in the test above."""
    target = _resolve_over_all(cohort_shards).target

    dropped = tuple(target.declared_warmup_steps[index] for index in target.dropped_index)

    assert dropped == _DROPPED_WARMUPS
    # The last four of the ``fhr_st`` block, not of the concatenated 102: the phase block follows
    # the scattering one in channel order and survives the budget whole. That distinction is what
    # makes a positional join between the 102-wide band map and the 98-wide gap vector wrong on any
    # dataset whose survivors are not a prefix.
    assert target.dropped_index == tuple(range(CAUSAL_ST_WIDTH - 4, CAUSAL_ST_WIDTH))
    assert target.block_counts() == (("fhr_st", 32, 36), ("fhr_ph", 66, 66))
    # And the slowest SURVIVOR is what the anchor floor is paired against, not the threshold.
    assert target.max_warmup <= SHIPPED_BUDGET_STEPS
    assert SHIPPED_WARMUP_PERIOD >= target.max_warmup - 1


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


def test_the_channel_widths_match_the_models_data_contract(batches) -> None:
    batch = batches[0]
    assert batch["fhr_st"].shape[2] == CAUSAL_ST_WIDTH
    assert batch["fhr_ph"].shape[2] == CAUSAL_PH_WIDTH
    assert batch["up_st"].shape[2] == CAUSAL_ST_WIDTH
    assert batch["up_ph"].shape[2] == CAUSAL_C_U - CAUSAL_ST_WIDTH


def test_the_dense_anchor_set_the_evaluation_decodes_at_is_the_full_one(batches) -> None:
    r"""$[F, T - H) = [133, 270)$ is 137 anchors, and the evaluation decodes every one of them.

    Derived from the length the LOADER yields rather than from the config constant, because a
    fixture written at a shorter window would leave the evaluation with no anchors at all and the
    symptom would be an empty table rather than an error.
    """
    served = int(batches[0]["fhr_st"].shape[1])

    assert served == SHIPPED_SEQUENCE_LENGTH
    assert served - SHIPPED_HORIZON - SHIPPED_WARMUP_PERIOD == 137


def test_all_the_clinical_fields_arrive_in_the_batch(batches) -> None:
    """The loader skips a field a shard does not carry, silently, so absence is not an error
    downstream -- it is a missing column that reads as "this cohort has no labels"."""
    for name in ("target", "epoch", "cs_label", "bg_label", "time_from_labor_onset"):
        assert name in batches[0], name


def test_guid_and_epoch_both_arrive_because_the_tile_phase_is_keyed_on_the_pair(batches) -> None:
    """``load_fields`` is honoured literally with no forced additions, and dropping either would
    put every segment on one tile grid with no shape, count or metric differing."""
    assert "guid" in batches[0]
    assert "epoch" in batches[0]


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


def test_both_label_axes_carry_two_values(batches) -> None:
    """The obvious substring rules label the doubly negative subgroup positive on both axes,
    which collapses each by-label table to one group. The generator uses an explicit table."""
    assert set(_column(batches, "cs_label")) == {True, False}
    assert set(_column(batches, "bg_label")) == {True, False}


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


def test_no_fixture_binary_is_committed() -> None:
    """The shards are generated into ``tmp_path_factory``; this suite commits no HDF5 of its own.
    Regenerating the family's committed binaries to add cohort fields would perturb every number
    the existing suites are pinned against."""
    assert not (Path(__file__).resolve().parent / "fixtures").exists()


# ---------------------------------------------------------------------------
# The trained run built on them
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_the_run_directory_has_the_layout_a_later_offline_pass_needs(cohort_run) -> None:
    """``run.py`` is handed a checkpoint and finds the run's own resolved config beside it. Both
    halves have to be there, or an evaluation would have to fall back on what a committed config
    file currently says -- which is the drift the whole arrangement exists to prevent."""
    from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME

    checkpoint_dir = Path(cohort_run) / "model_checkpoints"

    assert checkpoint_dir.is_dir()
    assert list(checkpoint_dir.glob("*.ckpt")), "the fit wrote no checkpoint"
    assert (checkpoint_dir / RESOLVED_CONFIG_FILENAME).is_file()


@pytest.mark.slow
def test_the_checkpoint_stamps_the_four_tuples_the_budget_resolved(cohort_run) -> None:
    r"""The budget is a config key that names no constructor argument: the driver resolves it
    against the *shards* into four concrete channel tuples, and those are what land in
    ``model_kwargs``. They are what makes a run's channel set recoverable from its checkpoint alone
    with no shard present -- and the decoder's width, which is the unit every nat is summed over.
    """
    path = next(iter((Path(cohort_run) / "model_checkpoints").glob("*.ckpt")))
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model_kwargs = blob["model_kwargs"]

    for key in (
        "target_keep_index", "target_warmup_steps", "source_keep_index", "source_warmup_steps"
    ):
        assert key in model_kwargs, key

    assert len(model_kwargs["target_keep_index"]) == _KEPT_TARGET_CHANNELS
    assert len(model_kwargs["target_warmup_steps"]) == _KEPT_TARGET_CHANNELS
    assert len(model_kwargs["source_keep_index"]) == CAUSAL_C_U
    assert len(model_kwargs["source_warmup_steps"]) == CAUSAL_C_U
    # The declared widths are untouched, which is what the data boundary checks against.
    assert (model_kwargs["c_y"], model_kwargs["c_u"]) == (CAUSAL_C_Y, CAUSAL_C_U)


@pytest.mark.slow
def test_the_run_records_the_budget_and_the_shards_it_was_resolved_against(cohort_run) -> None:
    """A run recording only the request would record neither what it got nor what its nats were
    summed over. The resolved config is also where a later offline pass reads the population."""
    import yaml

    from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME

    written = Path(cohort_run) / "model_checkpoints" / RESOLVED_CONFIG_FILENAME
    reloaded = yaml.safe_load(written.read_text(encoding="utf-8"))

    assert "base" not in reloaded
    vae = reloaded["model_config"]["VAE_model"]
    assert vae["causal_warmup_budget_steps"] == SHIPPED_BUDGET_STEPS
    assert vae["warmup_period"] == SHIPPED_WARMUP_PERIOD
    assert vae["causal_reach_budget_s"] is None
    shards = reloaded["dataset_config"]["vae_test_datasets"]
    assert len(shards) == 8
    assert all("REPOINT_ME" not in path for path in shards)
