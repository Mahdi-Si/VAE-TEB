r"""Conformance of the shared clinical labelling against *this* cell's data and conventions.

The labelling itself belongs to ``teb_vae/lag_attn/eval/labels.py`` and is tested there; it is
bound rather than rewritten, because a fork of it would be a fork of what a cohort *is*. What is
checked here is the part that could differ between a package and its data and would fail in
silence if it did.

**The subgroup stems.** ``CANONICAL_SUBGROUPS`` is a hardcoded eight-name table, and this cell's
holdout shards are named independently of it. If the two ever disagree, every sample carries no
subgroup, every by-subgroup table is empty, and nothing raises -- an unrecognised basename is
legitimate on the pretraining split, so it can only warn.

**The two label axes against the stem.** ``cs_label`` and ``bg_label`` arrive from the shard while
the subgroup comes from its file name, so the two can disagree without either being malformed --
and the obvious substring rules get it wrong in a specific way: ``'healthy_no_bg_no_cs'`` ends with
``'_cs'`` and contains ``'_bg_'``, so a rule built on them labels the doubly negative subgroup
positive on both axes and every by-label table collapses to one group.

**Absent is not zero.** There is no class $0$. A pad-only window and a uniformly zero ``target``
both yield *no class*; reporting one would create a phantom cohort that every by-class table would
then carry.

**The two validity predicates differ, deliberately.** ``clinical_class_code`` accepts any step with
``weight > 0``; the model's ``VALID_THRESHOLD`` counts a step as valid only at ``weight >= 1.0``.
They disagree on exactly the partially valid steps: a recording whose only valid steps are partial
still belongs to a cohort, even though the model scores none of its anchors. Reading the looser
predicate as the stricter one -- or "fixing" it to match -- would silently drop those recordings
from every by-class table.

Everything here reaches the labelling through this package's ``_reuse`` seam rather than importing
it directly, so the one import site holds.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.config_schema import load_eval_overrides
from teb_vae.lag_attn_rws.nets.raw_masks import VALID_THRESHOLD


@pytest.fixture(scope="module")
def batches(cohort_loader) -> list:
    """Every batch of the generated cohort shards, materialised once."""
    return list(cohort_loader)


# =================================================================================================
# The subgroup stems
# =================================================================================================
def test_the_eight_canonical_stems_resolve_against_the_configured_shards() -> None:
    """Both directions: every configured shard has a subgroup, and all eight are covered. The
    shipped delta points at placeholders that do not exist yet, and that is fine here -- what is
    asserted is the *naming*, which is what decides whether a run has cohorts at all."""
    shards = load_eval_overrides()["dataset_config"]["vae_test_datasets"]
    resolved = [labels.subgroup_of(path) for path in shards]

    assert None not in resolved, f"a configured shard resolved to no subgroup: {shards}"
    assert set(resolved) == set(labels.CANONICAL_SUBGROUPS)
    assert len(resolved) == len(labels.CANONICAL_SUBGROUPS)


def test_the_generated_fixture_shards_also_resolve(cohort_shards) -> None:
    """The fixture is named after real subgroups, so every by-subgroup test has real groups."""
    assert all(labels.subgroup_of(path) is not None for path in cohort_shards)
    assert {labels.subgroup_of(path) for path in cohort_shards} == set(labels.CANONICAL_SUBGROUPS)


@pytest.mark.parametrize("suffix", [".hdf5", ".h5", ""])
def test_a_subgroup_resolves_with_or_without_its_suffix_and_directory(suffix: str) -> None:
    assert labels.subgroup_of(f"/data/causal/test/acidosis_cs{suffix}") == "acidosis_cs"


def test_an_unknown_basename_warns_once_and_yields_no_subgroup(monkeypatch) -> None:
    """Once, not once per sample: twenty thousand identical lines would bury the run's log."""
    warnings: list = []
    monkeypatch.setattr(labels.logger, "warning", lambda message: warnings.append(message))
    monkeypatch.setattr(labels, "_WARNED_UNKNOWN", set())

    assert labels.subgroup_of("pretraining_split_cs.hdf5") is None
    assert labels.subgroup_of("pretraining_split_cs.hdf5") is None

    assert len(warnings) == 1
    assert "pretraining_split_cs" in warnings[0]


def test_no_source_file_yields_no_subgroup() -> None:
    assert labels.subgroup_of(None) is None
    assert labels.subgroup_of("") is None


# =================================================================================================
# The generated cohort, read the way a run reads it
# =================================================================================================
def test_every_class_code_in_the_generated_shards_resolves_to_a_name(batches) -> None:
    """A code with no name would carry a whole cohort into every table under a label a reader
    cannot interpret -- and the three the dataset actually stores are all that may appear."""
    codes = {
        labels.clinical_class_code(target, weight)
        for batch in batches
        for target, weight in zip(batch["target"], batch["weight"])
    }

    assert codes == {1, 2, 3}
    assert {labels.class_name(code) for code in codes} == {"healthy", "acidosis", "hie"}


def test_the_batch_labelling_agrees_with_the_shard_stem_on_both_axes(batches) -> None:
    """The class comes from ``target``, the subgroup from the file name, and they must agree about
    which cohort a segment belongs to. Both are resolved by the runner's own entry point rather
    than recomputed here, so what is checked is the path a run actually takes."""
    from scripts.make_tiny_shard import COHORT_SUBGROUPS

    for batch in batches:
        size = len(batch["guid"])
        resolved = labels.batch_labels(batch, size)
        for index in range(size):
            stem = str(batch["source_file_basename"][index]).replace(".hdf5", "")
            assert resolved[labels.SUBGROUP_COLUMN][index] == stem
            assert resolved[labels.CLASS_COLUMN][index] == labels.class_name(
                COHORT_SUBGROUPS[stem]["code"]
            )


def test_the_two_label_columns_agree_with_the_stem_they_were_written_for(batches) -> None:
    """``cs_label`` and ``bg_label`` are what a subgroup contrast would be cut on if it were cut
    from the labels rather than from the file name, and the doubly negative subgroup is where the
    obvious substring rules put a positive on both axes."""
    from scripts.make_tiny_shard import COHORT_SUBGROUPS

    seen = set()
    for batch in batches:
        for index in range(len(batch["guid"])):
            stem = str(batch["source_file_basename"][index]).replace(".hdf5", "")
            expected = COHORT_SUBGROUPS[stem]
            assert int(batch["cs_label"][index]) == expected["cs"], stem
            assert int(batch["bg_label"][index]) == expected["bg"], stem
            seen.add(stem)

    assert seen == set(labels.CANONICAL_SUBGROUPS)
    # The case the substring rules fail on is present rather than assumed.
    assert "healthy_no_bg_no_cs" in seen


def test_an_unrecognised_cohort_would_be_reported_rather_than_dropped(batches) -> None:
    """Non-vacuity for the ordering rule: a stem the canonical order does not know sorts after
    every one it does, and is never silently removed from a figure."""
    from teb_vae.lag_attn_cfs.eval.cohort import ordered_groups

    present = sorted({str(source).replace(".hdf5", "")
                      for batch in batches for source in batch["source_file_basename"]})

    ordered = ordered_groups([*present, "an_unnamed_shard"], labels.SUBGROUP_COLUMN)

    assert ordered[:-1] == list(labels.CANONICAL_SUBGROUPS)
    assert ordered[-1] == "an_unnamed_shard"


# =================================================================================================
# Absent is not zero
# =================================================================================================
def test_a_uniformly_zero_target_yields_no_class_rather_than_class_zero() -> None:
    """What the healthy-only pretraining split writes. A class 0 here would appear as a cohort in
    every by-class table, on every recording the model was actually trained on."""
    target = torch.zeros(300)
    weight = torch.ones(300)

    assert labels.clinical_class_code(target, weight) is None
    assert labels.class_name(None) is None


def test_a_pad_only_window_yields_no_class() -> None:
    assert labels.clinical_class_code(torch.zeros(300), torch.zeros(300)) is None


def test_an_empty_row_yields_no_class() -> None:
    assert labels.clinical_class_code(np.asarray([]), np.asarray([])) is None


def test_an_unrecognised_code_is_reported_rather_than_dropped() -> None:
    """An unknown code is a dataset question; silently discarding it would hide it."""
    assert labels.class_name(7) == "class_7"
    assert labels.class_name(1) == "healthy"
    assert labels.class_name(2) == "acidosis"
    assert labels.class_name(3) == "hie"


def test_one_anomalous_step_does_not_decide_a_recordings_cohort() -> None:
    """The code is constant over the steps it covers, so a disagreement is numerical."""
    weight = torch.ones(10)
    target = torch.full((10,), 2.0)
    target[4] = 3.0

    assert labels.clinical_class_code(target, weight) == 2


def test_a_non_finite_step_is_excluded_rather_than_read_as_a_code() -> None:
    weight = torch.ones(6)
    target = torch.tensor([2.0, 2.0, float("nan"), 2.0, 2.0, 2.0])

    assert labels.clinical_class_code(target, weight) == 2


# =================================================================================================
# The two validity predicates, and why they differ
# =================================================================================================
def test_a_wholly_partial_recording_has_a_cohort_even_though_the_model_scores_no_anchor() -> None:
    r"""``weight > 0`` against ``VALID_THRESHOLD = 1.0``: the label side is deliberately looser.

    Every step here is half-valid, so the mask builder binarises the whole segment to invalid and
    the model contributes no anchor from it -- while the recording still has a clinical class,
    which is what keeps it in the cohort tables and out of a silently narrowed denominator.
    """
    weight = torch.full((300,), 0.5)
    target = 2.0 * weight  # stores 1.0 everywhere: a fully valid healthy step's value

    assert labels.clinical_class_code(target, weight) == 2
    assert float(weight.max()) < VALID_THRESHOLD
    assert not bool((weight >= VALID_THRESHOLD).any())


def test_the_two_predicates_agree_wherever_validity_is_binary() -> None:
    """The disagreement is confined to partial steps; on binary weight there is none."""
    weight = torch.ones(20)
    weight[5:10] = 0.0
    target = 3.0 * weight

    assert labels.clinical_class_code(target, weight) == 3
    assert torch.equal(weight > 0.0, weight >= VALID_THRESHOLD)


def test_the_fixture_carries_the_partial_steps_that_make_the_difference_real(batches) -> None:
    """Not a hypothetical: the generated shards write fractional weights *inside* the trimmed
    window, so the two predicates genuinely disagree on this data rather than only in principle."""
    weight = batches[0]["weight"]

    partial = (weight > 0.0) & (weight < VALID_THRESHOLD)
    assert bool(partial.any())
