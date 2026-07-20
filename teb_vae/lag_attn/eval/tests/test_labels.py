r"""Tests for clinical-class and subgroup extraction.

The load-bearing case is **fractional weight**, and it is the reason this module exists rather
than the dataset's own ``label`` filter. That filter tests exact float equality against ``target``,
so at ``weight = 0.5`` an acidosis step storing ``target = 1.0`` is indistinguishable from a
fully-valid healthy step -- and the filter drops the segment rather than mislabelling it, which is
quieter and worse. Every other test here is a boundary around that one.

The multi-class shards are generated into ``tmp_path`` by ``conftest.write_multi_class_shards``
and loaded through the *real* ``GraphDataModule``, so the extraction is exercised against the
actual batch contract -- ``source_file_basename`` as a ``list[str]``, a trimmed $T = 300$, and a
``target`` that has been through the loader's own dtype handling.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from teb_vae.lag_attn.eval import labels
from teb_vae.lag_attn.eval.runner import get_field
from teb_vae.lag_attn.eval.tests.conftest import MULTI_CLASS_SUBGROUPS


# ---------------------------------------------------------------------------
# The class, from a scaled target
# ---------------------------------------------------------------------------
def test_a_fully_valid_recording_yields_its_class_code() -> None:
    target = torch.full((300,), 2.0)
    weight = torch.ones(300)
    assert labels.clinical_class_code(target, weight) == 2


@pytest.mark.parametrize("code", [1, 2, 3])
@pytest.mark.parametrize("fraction", [0.5, 0.25, 0.9])
def test_the_class_is_recovered_under_a_fractional_weight(code: int, fraction: float) -> None:
    """The case the dataset's exact-equality ``label`` filter would have dropped.

    At ``weight = 0.5`` an acidosis step stores ``target = 1.0``, which is exactly what a
    fully-valid healthy step stores. Reading ``target`` directly mislabels it; dividing recovers
    it.
    """
    weight = torch.full((100,), float(fraction))
    target = weight * float(code)
    assert labels.clinical_class_code(target, weight) == code


def test_a_recording_with_fractional_edges_and_a_full_middle_yields_one_class() -> None:
    """The real shape of a segment: partially-valid boundaries, fully-valid interior."""
    weight = torch.ones(50)
    weight[:5] = 0.5
    weight[-5:] = 0.25
    assert labels.clinical_class_code(weight * 3.0, weight) == 3


def test_a_pad_only_sample_yields_none_rather_than_zero() -> None:
    """Zero is not a class, and reporting one would create a phantom cohort."""
    assert labels.clinical_class_code(torch.zeros(20), torch.zeros(20)) is None


def test_a_uniformly_zero_target_yields_none() -> None:
    """What the healthy-only pretraining split writes, and what the committed fixture carries."""
    assert labels.clinical_class_code(torch.zeros(20), torch.ones(20)) is None


def test_the_zero_weight_steps_are_ignored_rather_than_dividing_by_zero() -> None:
    """The gap steps carry target 0 and weight 0; only the valid ones decide the class."""
    weight = torch.zeros(30)
    weight[10:20] = 1.0
    target = weight * 2.0
    assert labels.clinical_class_code(target, weight) == 2


def test_a_single_anomalous_step_does_not_decide_the_cohort() -> None:
    """Most common, not first: the code is constant, so a lone disagreement is numerical."""
    weight = torch.ones(21)
    target = torch.full((21,), 3.0)
    target[0] = 1.0
    assert labels.clinical_class_code(target, weight) == 3


def test_a_non_finite_step_is_skipped_rather_than_poisoning_the_vote() -> None:
    weight = torch.ones(10)
    target = torch.full((10,), 2.0)
    target[0] = float("nan")
    assert labels.clinical_class_code(target, weight) == 2


def test_an_empty_row_yields_none() -> None:
    assert labels.clinical_class_code(torch.zeros(0), torch.zeros(0)) is None


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("code, name", [(1, "healthy"), (2, "acidosis"), (3, "hie")])
def test_the_class_codes_map_to_their_clinical_names(code: int, name: str) -> None:
    assert labels.class_name(code) == name


def test_no_class_stays_no_class() -> None:
    assert labels.class_name(None) is None


def test_an_unknown_code_is_reported_rather_than_dropped() -> None:
    """An unknown code is a dataset question; discarding it would hide it."""
    assert labels.class_name(7) == "class_7"


# ---------------------------------------------------------------------------
# Subgroups
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("subgroup", labels.CANONICAL_SUBGROUPS)
def test_each_canonical_subgroup_is_recovered_from_its_basename(subgroup: str) -> None:
    assert labels.subgroup_of(f"{subgroup}.hdf5") == subgroup


def test_a_full_path_resolves_to_its_subgroup(tmp_path) -> None:
    path = tmp_path / "k_fold_cross_validation_dataset" / "test" / "acidosis_cs.hdf5"
    assert labels.subgroup_of(str(path)) == "acidosis_cs"


def test_the_suffix_is_optional() -> None:
    assert labels.subgroup_of("hie_no_cs") == "hie_no_cs"


def test_an_unknown_basename_maps_to_none_and_warns_once(caplog) -> None:
    """Once per distinct name: twenty thousand identical warnings would bury the run's log."""
    labels._WARNED_UNKNOWN.discard("not_a_subgroup")
    with caplog.at_level("WARNING"):
        assert labels.subgroup_of("not_a_subgroup.hdf5") is None
        # A second call for the same name must not warn again.
        assert labels.subgroup_of("not_a_subgroup.hdf5") is None
    assert caplog.text.count("not_a_subgroup") <= 1


def test_a_missing_basename_maps_to_none() -> None:
    assert labels.subgroup_of(None) is None
    assert labels.subgroup_of("") is None


# ---------------------------------------------------------------------------
# Distinct groups
# ---------------------------------------------------------------------------
def test_none_and_nan_are_not_groups() -> None:
    """A sample with no class belongs to no cohort; folding them would name one after absence."""
    assert labels.distinct_groups(["a", None, "b", float("nan"), "a"]) == ["a", "b"]


def test_an_entirely_unlabelled_column_has_no_groups() -> None:
    assert labels.distinct_groups([None, None]) == []


# ---------------------------------------------------------------------------
# Against the real batch contract
# ---------------------------------------------------------------------------
def test_the_generated_shards_carry_both_classes_and_distinct_subgroups(multi_class_loader) -> None:
    """Without this the class-aware paths would only ever exercise their skip branches."""
    seen_classes, seen_subgroups = set(), set()
    for batch in multi_class_loader:
        columns = labels.batch_labels(batch, len(get_field(batch, "guid")))
        seen_classes.update(value for value in columns[labels.CLASS_COLUMN] if value)
        seen_subgroups.update(value for value in columns[labels.SUBGROUP_COLUMN] if value)

    assert seen_classes == {"healthy", "acidosis"}
    assert seen_subgroups == set(MULTI_CLASS_SUBGROUPS)
    assert len(seen_subgroups) > len(seen_classes), (
        "the two groupings must not coincide, or a by-subgroup test could pass on a by-class bug"
    )


def test_the_class_survives_the_loaders_trimming_and_normalisation(multi_class_loader) -> None:
    r"""The shards' fractional edges are inside the trimmed window, so this is the real case.

    Trimming removes $15$ decimated steps from each end of a $330$-step shard; the fractional
    ``weight`` is written into the first and last four, so some of it survives into the batch and
    some does not. Either way the class must come back.
    """
    for batch in multi_class_loader:
        weight = get_field(batch, "weight")
        target = get_field(batch, "target")
        basenames = get_field(batch, "source_file_basename")
        for index in range(int(weight.shape[0])):
            expected = MULTI_CLASS_SUBGROUPS[labels.subgroup_of(basenames[index])]
            assert labels.clinical_class_code(target[index], weight[index]) == expected


def test_the_committed_single_class_shard_yields_no_class(tiny_loader) -> None:
    """Its ``target`` is all zeros, so every class-aware path must see no cohort at all."""
    for batch in tiny_loader:
        columns = labels.batch_labels(batch, len(get_field(batch, "guid")))
        assert all(value is None for value in columns[labels.CLASS_COLUMN])
        assert labels.distinct_groups(columns[labels.CLASS_COLUMN]) == []


def test_a_batch_without_a_target_yields_a_column_of_none_rather_than_raising() -> None:
    """The class axis is optional; a split without labels must produce pooled output, not a crash."""
    import types

    batch = types.SimpleNamespace(source_file_basename=["acidosis_cs.hdf5"] * 2)
    columns = labels.batch_labels(batch, 2)
    assert columns[labels.CLASS_COLUMN] == [None, None]
    assert columns[labels.SUBGROUP_COLUMN] == ["acidosis_cs", "acidosis_cs"]


def test_batch_labels_returns_one_entry_per_sample(multi_class_loader) -> None:
    """A short column would silently misalign every metric against the wrong recording."""
    for batch in multi_class_loader:
        size = int(get_field(batch, "weight").shape[0])
        columns = labels.batch_labels(batch, size)
        for name in labels.GROUP_COLUMNS:
            assert len(columns[name]) == size


def test_numpy_rows_are_accepted_as_well_as_tensors() -> None:
    """The collectors pass tensors; a CSV-driven caller passes arrays."""
    weight = np.full(10, 0.5)
    assert labels.clinical_class_code(weight * 2.0, weight) == 2
