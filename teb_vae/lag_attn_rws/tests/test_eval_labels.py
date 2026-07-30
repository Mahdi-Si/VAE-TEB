r"""Conformance of the shared clinical labelling against *this* model's data and conventions.

The labelling itself is the sibling's and is tested there; it is imported rather than rewritten.
What is checked here is the part that could differ between the two packages and would fail
silently if it did.

**The subgroup stems.** ``CANONICAL_SUBGROUPS`` is a hardcoded eight-name table, and this
package's holdout shards are named independently of it. If the two ever disagree, every sample
carries no subgroup, every by-subgroup table is empty, and nothing raises -- an unrecognised
basename is legitimate on the pretraining split, so it can only warn.

**The validity predicate.** ``clinical_class_code`` accepts any step with ``weight > 0``;
``nets/raw_masks.VALID_THRESHOLD`` counts a step as valid only at ``weight >= 1.0``. The two
disagree on exactly the partially valid steps, and the disagreement is deliberate: a recording
whose only valid steps are partial still belongs to a cohort, even though the model scores none
of its anchors. Reading the looser predicate as the stricter one -- or "fixing" it to match --
would silently drop those recordings from every by-class table.

**Absent is not zero.** There is no class $0$. A pad-only window and a uniformly zero ``target``
both yield *no class*; reporting one would create a phantom cohort that every by-class table
would then carry.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from teb_vae.lag_attn_rws.eval._reuse import labels
from teb_vae.lag_attn_rws.eval.config_schema import load_eval_overrides
from teb_vae.lag_attn_rws.nets.raw_masks import VALID_THRESHOLD


# ---------------------------------------------------------------------------
# The subgroup stems
# ---------------------------------------------------------------------------
def test_the_eight_canonical_stems_resolve_against_the_configured_shards() -> None:
    """Both directions: every configured shard has a subgroup, and all eight are covered."""
    shards = load_eval_overrides()["dataset_config"]["vae_test_datasets"]
    resolved = [labels.subgroup_of(path) for path in shards]

    assert None not in resolved, f"a configured shard resolved to no subgroup: {shards}"
    assert set(resolved) == set(labels.CANONICAL_SUBGROUPS)
    assert len(resolved) == len(labels.CANONICAL_SUBGROUPS)


def test_the_generated_fixture_shards_also_resolve(multi_class_shards) -> None:
    """The fixture is named after real subgroups, so every by-subgroup test has real groups."""
    assert all(labels.subgroup_of(path) is not None for path in multi_class_shards)


@pytest.mark.parametrize("suffix", [".hdf5", ".h5", ""])
def test_a_subgroup_resolves_with_or_without_its_suffix_and_directory(suffix: str) -> None:
    assert labels.subgroup_of(f"/data/k_fold/test/acidosis_cs{suffix}") == "acidosis_cs"


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


# ---------------------------------------------------------------------------
# Absent is not zero
# ---------------------------------------------------------------------------
def test_a_uniformly_zero_target_yields_no_class_rather_than_class_zero() -> None:
    """What the healthy-only pretraining split writes. A class 0 here would appear as a cohort
    in every by-class table, on every recording the model was actually trained on."""
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


def test_one_anomalous_step_does_not_decide_a_recording_s_cohort() -> None:
    """The code is constant over the steps it covers, so a disagreement is numerical."""
    weight = torch.ones(10)
    target = torch.full((10,), 2.0)
    target[4] = 3.0

    assert labels.clinical_class_code(target, weight) == 2


# ---------------------------------------------------------------------------
# The two validity predicates, and why they differ
# ---------------------------------------------------------------------------
def test_a_wholly_partial_recording_has_a_cohort_even_though_the_model_scores_no_anchor() -> None:
    r"""``weight > 0`` against ``VALID_THRESHOLD = 1.0``: the label side is deliberately looser.

    Every step here is half-valid, so ``raw_masks`` binarises the whole segment to invalid and
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


def test_a_non_finite_step_is_excluded_rather_than_read_as_a_code() -> None:
    weight = torch.ones(6)
    target = torch.tensor([2.0, 2.0, float("nan"), 2.0, 2.0, 2.0])

    assert labels.clinical_class_code(target, weight) == 2
