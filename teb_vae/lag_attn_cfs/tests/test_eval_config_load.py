r"""The evaluation override delta, and what merging it over a run's own config produces.

The delta is not a config and does not stand alone. Its whole contract is what it *becomes* when
deep-merged over the ``resolved_config.yaml`` a training run wrote beside its checkpoints: the
run's geometry, its resolved warm-up budget, its normalisation and its objective survive
untouched, and exactly the overridden leaves change. Both halves of that are asserted here, because
either one failing silently produces plausible numbers -- an evaluation on the wrong population,
or one that never sees the fields the clinical questions are asked in.

The committed delta itself is checked only where it is behaviour: it loads, its ``eval_config``
block validates as written, and its ``load_fields`` restates every field the run already loads
(a list replaces wholesale on merge) plus the clinical ones and the ``guid``/``epoch`` pair the
anchor tiling's phase is keyed on.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_cfs.eval.config_schema import (
    VALID_KEYS,
    load_eval_overrides,
    merge_eval_overrides,
    merge_eval_overrides_with_provenance,
    validate_eval_config,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _REPO_ROOT / "teb_vae" / "lag_attn_cfs" / "configs" / "default.yaml"

#: The fields the evaluation reads on top of the model's own data contract: the clinical ones, and
#: the ``guid``/``epoch`` pair the anchor tiling's per-segment phase is a stable hash of. Dropping
#: either of the pair decodes every segment at one tile grid with no shape, count or metric
#: differing.
_REQUIRED_LOAD_FIELDS = (
    "target", "cs_label", "bg_label", "time_from_labor_onset", "guid", "epoch",
)


@pytest.fixture(scope="module")
def overrides() -> dict:
    return load_eval_overrides()


@pytest.fixture(scope="module")
def resolved() -> dict:
    """Stands in for a checkpoint's ``resolved_config.yaml``: fully explicit, no ``base:``."""
    return load_config(str(_DEFAULT_CONFIG))


@pytest.fixture(scope="module")
def merged(resolved) -> dict:
    return merge_eval_overrides(resolved)


def _dataset_kwargs(config: dict) -> dict:
    return config["dataset_config"]["dataloader_config"]["dataset_kwargs"]


# ---------------------------------------------------------------------------
# The delta itself
# ---------------------------------------------------------------------------
def test_a_delta_carrying_a_base_key_is_refused(tmp_path) -> None:
    """A ``base:`` chain would inherit from whatever a committed config currently says rather than
    from what the run trained under, which is the drift the merge exists to avoid."""
    path = tmp_path / "bad_overrides.yaml"
    path.write_text("base: ../../configs/default.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="base"):
        load_eval_overrides(path)


def test_a_missing_delta_raises_rather_than_merging_nothing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_eval_overrides(tmp_path / "absent.yaml")


def test_the_shipped_eval_config_block_validates_as_written(overrides) -> None:
    """The shipped file is not a latent failure: every key is known and every value is in range."""
    assert set(overrides["eval_config"]) <= VALID_KEYS
    assert validate_eval_config(overrides)["seed"] == overrides["eval_config"]["seed"]


# ---------------------------------------------------------------------------
# The merge
# ---------------------------------------------------------------------------
def test_every_override_lands_in_the_merged_config(merged, overrides) -> None:
    assert merged["general_config"]["batch_size"]["test"] == (
        overrides["general_config"]["batch_size"]["test"]
    )
    assert merged["dataset_config"]["vae_test_datasets"] == (
        overrides["dataset_config"]["vae_test_datasets"]
    )
    assert merged["dataset_config"]["stat_path"] == overrides["dataset_config"]["stat_path"]
    for key, value in _dataset_kwargs(overrides).items():
        assert _dataset_kwargs(merged)[key] == value, key
    assert merged["eval_config"] == overrides["eval_config"]


def test_the_required_fields_are_added_and_the_model_s_own_fields_survive(
    merged, resolved
) -> None:
    """A list replaces wholesale on merge, so the inherited entries are restated in the delta
    rather than extended -- and a restatement that dropped one would be invisible."""
    fields = _dataset_kwargs(merged)["load_fields"]
    for name in _REQUIRED_LOAD_FIELDS:
        assert name in fields, name
    for name in _dataset_kwargs(resolved)["load_fields"]:
        assert name in fields, f"the delta's load_fields dropped the inherited {name!r}"


def test_the_run_s_own_contract_survives_the_merge(merged, resolved, overrides) -> None:
    """Geometry, the warm-up budget, normalisation and the objective come from the run, never from
    the delta."""
    assert merged["model_config"] == resolved["model_config"]
    assert merged["dataset_config"]["vae_train_datasets"] == (
        resolved["dataset_config"]["vae_train_datasets"]
    )
    loader = merged["dataset_config"]["dataloader_config"]
    assert loader["normalize_fields"] == (
        resolved["dataset_config"]["dataloader_config"]["normalize_fields"]
    )
    # Setting some dataset_kwargs entries must not drop or move the rest of the block -- the
    # inherited epoch filter, the trim and the label filter among them.
    untouched = set(_dataset_kwargs(resolved)) - set(_dataset_kwargs(overrides))
    assert untouched, "a delta overriding every dataset kwarg would make this vacuous"
    for key in untouched:
        assert _dataset_kwargs(merged)[key] == _dataset_kwargs(resolved)[key], key
    assert merged["general_config"]["batch_size"]["train"] == (
        resolved["general_config"]["batch_size"]["train"]
    )


def test_the_merge_mutates_neither_input(resolved) -> None:
    before = yaml.safe_dump(resolved, sort_keys=True)
    merge_eval_overrides(resolved)
    assert yaml.safe_dump(resolved, sort_keys=True) == before


# ---------------------------------------------------------------------------
# What the merge changed, recorded
# ---------------------------------------------------------------------------
def test_the_provenance_carries_both_values_of_every_overridden_leaf(resolved) -> None:
    """The record is what makes the divergence from the trained contract legible. Without it a
    reader of ``summary.json`` cannot tell that the shard list was repointed at the holdout
    split, that the statistics file moved with it, or that something else was overridden too.
    """
    merged, provenance = merge_eval_overrides_with_provenance(resolved)
    by_path = {record["path"]: record for record in provenance}

    fields = by_path["dataset_config.dataloader_config.dataset_kwargs.load_fields"]
    assert fields["run_value"] == _dataset_kwargs(resolved)["load_fields"]
    assert fields["eval_value"] == _dataset_kwargs(merged)["load_fields"]
    assert fields["run_value"] != fields["eval_value"]
    assert fields["in_run_config"] is True

    shards = by_path["dataset_config.vae_test_datasets"]
    assert shards["run_value"] == resolved["dataset_config"]["vae_test_datasets"]
    assert shards["eval_value"] == merged["dataset_config"]["vae_test_datasets"]

    stats = by_path["dataset_config.stat_path"]
    assert stats["run_value"] == resolved["dataset_config"]["stat_path"]
    assert stats["eval_value"] == merged["dataset_config"]["stat_path"]
    assert stats["run_value"] != stats["eval_value"]


def test_the_provenance_walks_to_the_leaves_rather_than_recording_whole_blocks(
    resolved,
) -> None:
    """A record of ``dataset_config`` changing says nothing; a record naming ``cache_size`` and
    its two values says what happened. Blocks the run does *not* carry are recorded whole,
    because there is no leaf below them to compare against."""
    _merged, provenance = merge_eval_overrides_with_provenance(resolved)
    paths = {record["path"] for record in provenance}

    assert "dataset_config.dataloader_config.dataset_kwargs.cache_size" in paths
    assert "dataset_config" not in paths
    assert "general_config.batch_size.test" in paths

    absent = [record for record in provenance if not record["in_run_config"]]
    assert [record["path"] for record in absent] == ["eval_config"], (
        "default.yaml carries no eval_config block, so the delta adds it whole"
    )


def test_the_provenance_is_ordered_and_covers_only_what_the_delta_touched(resolved) -> None:
    _merged, provenance = merge_eval_overrides_with_provenance(resolved)
    paths = [record["path"] for record in provenance]

    assert paths == sorted(paths)
    assert all(path.split(".")[0] in {"general_config", "dataset_config", "eval_config"}
               for path in paths)
