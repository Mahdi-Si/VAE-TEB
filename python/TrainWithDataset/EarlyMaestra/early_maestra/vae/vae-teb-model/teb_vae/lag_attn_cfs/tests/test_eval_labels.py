r"""Conformance of the shared clinical labelling against *this* cell's data and conventions.

The labelling itself belongs to ``teb_vae/lag_attn/eval/labels.py`` and is tested there; it is
bound rather than rewritten, because a fork of it would be a fork of what a cohort *is*. What is
checked here is the part that could differ between a package and its data and would fail in
silence if it did.

**The subgroup stems.** ``CANONICAL_SUBGROUPS`` is a hardcoded table, and this cell's holdout
shards are named independently of it. If the two ever disagree, every sample carries no subgroup,
every by-subgroup table is empty, and nothing raises -- an unrecognised basename is legitimate on
the pretraining split, so it can only warn.

**The two label axes against the stem.** ``cs_label`` and ``bg_label`` arrive from the shard while
the subgroup comes from its file name, so the two can disagree without either being malformed --
and the obvious substring rules get it wrong in a specific way: ``'healthy_no_bg_no_cs'`` ends with
``'_cs'`` and contains ``'_bg_'``, so a rule built on them labels the doubly negative subgroup
positive on both axes and every by-label table collapses to one group.

**An unrecognised cohort is ordered last rather than dropped**, so a shard the canonical order
does not know still reaches every figure instead of silently vanishing from it.
"""
from __future__ import annotations

import pytest

from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.config_schema import load_eval_overrides


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


# =================================================================================================
# The generated cohort, read the way a run reads it
# =================================================================================================
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

    assert ordered[:-1] == list(reversed(labels.CANONICAL_SUBGROUPS))
    assert ordered[-1] == "an_unnamed_shard"
