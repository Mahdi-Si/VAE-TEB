r"""This package's override delta: what it changes, and what it deliberately does not.

The delta is not a config and does not stand alone. Its contract is what it *becomes* when
deep-merged over the ``resolved_config.yaml`` a training run wrote beside its checkpoints: the
run's geometry, normalisation and objective survive untouched, and exactly five things change.

There is a second contract here the sibling's file does not have, and it is the more important
one. **This delta must equal the sibling's.** The two models exist to be compared, so a holdout
split, a Monte Carlo draw count or a bootstrap resample count that differed between them would
make every side-by-side number a comparison of two protocols rather than of two architectures --
and nothing in either run's artifacts would say so. The equality is asserted against the sibling's
committed file rather than against a copy of its values, so a change on either side fails here.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws.eval.config_schema import (
    DEFAULT_OVERRIDES_PATH as SIBLING_OVERRIDES_PATH,
    VALID_KEYS,
    load_eval_overrides,
    merge_eval_overrides,
    validate_eval_config,
)
from teb_vae.lag_attn_transformer_rws.eval.binding import TRF_BINDING

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _REPO_ROOT / "teb_vae" / "lag_attn_transformer_rws" / "configs" / "default.yaml"

#: The fields the delta adds on top of the model's own data contract.
_ADDED_LOAD_FIELDS = ("target", "epoch", "cs_label", "bg_label", "time_from_labor_onset")

#: The placeholder every shipped shard path carries, so a launch fails on a missing file with a
#: message naming the real cause rather than on a width mismatch someone might "fix" by reverting
#: the channel counts.
_REPOINT_MARKER = "REPOINT_ME"


@pytest.fixture(scope="module")
def overrides() -> dict:
    return load_eval_overrides(TRF_BINDING.overrides_path)


@pytest.fixture(scope="module")
def sibling_overrides() -> dict:
    return load_eval_overrides(SIBLING_OVERRIDES_PATH)


@pytest.fixture(scope="module")
def resolved() -> dict:
    """Stands in for a checkpoint's ``resolved_config.yaml``: fully explicit, no ``base:``."""
    return load_config(str(_DEFAULT_CONFIG))


@pytest.fixture(scope="module")
def merged(resolved) -> dict:
    return merge_eval_overrides(resolved, TRF_BINDING.overrides_path)


def _dataset_kwargs(config: dict) -> dict:
    return config["dataset_config"]["dataloader_config"]["dataset_kwargs"]


# =============================================================================
# The delta itself
# =============================================================================
def test_the_committed_delta_is_where_the_binding_says_it_is() -> None:
    assert Path(TRF_BINDING.overrides_path).is_file()
    assert Path(TRF_BINDING.overrides_path).name == "eval_overrides.yaml"


def test_the_delta_carries_no_base_key(overrides) -> None:
    """A ``base:`` chain here would inherit from whatever a committed config currently says rather
    than from what the run trained under, which is the drift the merge exists to avoid."""
    assert "base" not in overrides


def test_a_delta_carrying_a_base_key_is_refused(tmp_path) -> None:
    """Not vacuous: the loader is what enforces the rule above, and it must actually raise."""
    broken = tmp_path / "with_base.yaml"
    broken.write_text("base: default.yaml\neval_config: {seed: 1}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="base"):
        load_eval_overrides(broken)


def test_the_delta_overrides_only_what_genuinely_differs(overrides) -> None:
    assert set(overrides) == {"general_config", "dataset_config", "eval_config"}
    assert set(overrides["general_config"]) == {"batch_size"}
    assert set(overrides["dataset_config"]) == {"vae_test_datasets", "dataloader_config"}


def test_every_eval_config_key_in_the_delta_is_one_the_schema_accepts(overrides) -> None:
    unknown = set(overrides["eval_config"]) - set(VALID_KEYS)
    assert unknown == set()


def test_the_resolved_eval_config_validates(merged) -> None:
    """The validator is what a run calls before it loads a checkpoint, so a misspelled key here
    must cost a parse rather than a model load and a first pass over the shards."""
    assert validate_eval_config(merged)["seed"] == merged["eval_config"]["seed"]


def test_the_delta_points_at_the_eight_holdout_subgroup_shards(overrides) -> None:
    shards = overrides["dataset_config"]["vae_test_datasets"]

    assert len(shards) == 8
    assert len({Path(shard).name for shard in shards}) == 8


def test_no_shard_path_escaped_the_repoint_convention(overrides) -> None:
    """A path that quietly became real would make a launch on another box read a file nobody
    reviewed, instead of failing with a message naming the repoint step."""
    escaped = [
        shard
        for shard in overrides["dataset_config"]["vae_test_datasets"]
        if _REPOINT_MARKER not in shard
    ]
    assert escaped == []


def test_the_cache_is_off(overrides) -> None:
    """At ``num_workers: 0`` the loader's cache is main-process RAM, roughly 240 KB per sample,
    and it is FIFO with no reuse policy -- so against a split larger than the cache it thrashes."""
    assert _dataset_kwargs(overrides)["cache_size"] == 0


# =============================================================================
# Equality with the sibling's delta
# =============================================================================
def test_the_evaluation_settings_are_the_siblings_key_for_key(overrides, sibling_overrides) -> None:
    """The whole ``eval_config`` block, not a chosen subset: the seed, the draw count, the caps,
    the two verdict thresholds, the event window and the resample count. A difference in any of
    them makes the cross-model comparison a comparison of protocols."""
    assert overrides["eval_config"] == sibling_overrides["eval_config"]


def test_the_holdout_split_is_the_siblings(overrides, sibling_overrides) -> None:
    """Same eight files, same order. Two models evaluated on two populations are not comparable
    however carefully everything else is matched."""
    assert (
        overrides["dataset_config"]["vae_test_datasets"]
        == sibling_overrides["dataset_config"]["vae_test_datasets"]
    )


def test_the_loaded_fields_and_the_cache_are_the_siblings(overrides, sibling_overrides) -> None:
    assert _dataset_kwargs(overrides) == _dataset_kwargs(sibling_overrides)


def test_the_batch_size_matches_and_the_reason_is_written_down(overrides, sibling_overrides) -> None:
    r"""Equal, but not silently inherited: this architecture's encoder attention is
    $\mathcal O(B T^2 d)$ where the sibling's recurrent encoder was linear in $T$, so the value
    needed re-justifying rather than copying. The justification is in the file, because a number
    whose reasoning lives only in a review comment is a number the next person changes."""
    assert overrides["general_config"]["batch_size"] == sibling_overrides["general_config"][
        "batch_size"
    ]
    text = Path(TRF_BINDING.overrides_path).read_text(encoding="utf-8")
    assert "O(B T^2 d)" in text
    assert "scaled_dot_product_attention" in text


def test_the_two_deltas_differ_only_in_their_comments(overrides, sibling_overrides) -> None:
    """The strongest form of the claim, and the one that catches a key added to one file and not
    the other: parsed, the two documents are equal."""
    assert overrides == sibling_overrides


def test_the_delta_names_this_packages_launch_commands() -> None:
    """The header is what an operator reads first, and pointing at the sibling's entry points
    would have them evaluate the wrong model against this file."""
    text = Path(TRF_BINDING.overrides_path).read_text(encoding="utf-8")

    assert "teb_vae.lag_attn_transformer_rws.eval.probe" in text
    assert "teb_vae.lag_attn_transformer_rws.eval.run" in text
    assert "teb_vae.lag_attn_rws.eval." not in text


# =============================================================================
# The merge
# =============================================================================
def test_every_override_lands_in_the_merged_config(merged, overrides) -> None:
    assert merged["general_config"]["batch_size"]["test"] == (
        overrides["general_config"]["batch_size"]["test"]
    )
    assert merged["dataset_config"]["vae_test_datasets"] == (
        overrides["dataset_config"]["vae_test_datasets"]
    )
    assert merged["eval_config"] == overrides["eval_config"]


def test_the_five_clinical_fields_are_added_and_the_models_own_fields_survive(
    merged, resolved
) -> None:
    """A list replaces wholesale on merge, so the inherited fields are restated in the delta
    rather than extended -- and a miss would drop a field the model itself reads."""
    fields = _dataset_kwargs(merged)["load_fields"]

    for field in _ADDED_LOAD_FIELDS:
        assert field in fields
    for field in _dataset_kwargs(resolved)["load_fields"]:
        assert field in fields


def test_the_runs_own_contract_survives_the_merge(merged, resolved) -> None:
    """The geometry, the normalisation statistics and the objective are what the run trained
    under; the delta must touch none of them."""
    assert merged["model_config"] == resolved["model_config"]
    assert merged["dataset_config"]["stat_path"] == resolved["dataset_config"]["stat_path"]
    assert merged["dataset_config"]["vae_train_datasets"] == (
        resolved["dataset_config"]["vae_train_datasets"]
    )


def test_the_encoder_geometry_is_untouched_by_the_merge(merged, resolved) -> None:
    """Named separately from the block above because these seven are what this model *is*: a
    delta that moved one of them would evaluate a different architecture from the one trained."""
    for key in TRF_BINDING.geometry_keys:
        assert merged["model_config"]["VAE_model"].get(key) == (
            resolved["model_config"]["VAE_model"].get(key)
        )


def test_the_merge_mutates_neither_input(resolved) -> None:
    before = yaml.safe_dump(resolved, sort_keys=True)
    merge_eval_overrides(resolved, TRF_BINDING.overrides_path)

    assert yaml.safe_dump(resolved, sort_keys=True) == before
