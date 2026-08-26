r"""The evaluation override delta, and what merging it over a run's own config produces.

The delta is not a config and does not stand alone. Its whole contract is what it *becomes* when
deep-merged over the ``resolved_config.yaml`` a training run wrote beside its checkpoints: the
run's geometry, its resolved warm-up budget, its normalisation and its objective survive
untouched, and exactly a handful of things change. Both halves of that are asserted here, because
either one failing silently produces plausible numbers -- an evaluation on the wrong population,
or one that never sees the fields the clinical questions are asked in.

Two things this cell's delta does that the raw cells' does not, and each has its own test below:
it repoints ``stat_path``, because the causal statistics exclude each channel's warm-up region and
that exclusion is what makes zero the right feature-space climatology; and it restates ``guid``
and ``epoch`` in ``load_fields``, because the anchor tiling's phase is keyed on the pair and
``load_fields`` is honoured literally.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_cfs.eval.config_schema import (
    DEFAULT_OVERRIDES_PATH,
    VALID_KEYS,
    load_eval_overrides,
    merge_eval_overrides,
    merge_eval_overrides_with_provenance,
    validate_eval_config,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _REPO_ROOT / "teb_vae" / "lag_attn_cfs" / "configs" / "default.yaml"

#: The clinical fields the delta adds on top of the model's own data contract. ``epoch`` is not
#: among them here -- unlike in the raw cells, this model's training config already loads it,
#: because the anchor tiling's phase is keyed on it.
_ADDED_LOAD_FIELDS = ("target", "cs_label", "bg_label", "time_from_labor_onset")

#: The keys ``eval_config`` must carry, exactly. Written out rather than read from ``VALID_KEYS``:
#: this asserts what the committed *file* ships, and reading the schema would make the test pass
#: for any file that happened to be a subset of it.
_SHIPPED_EVAL_KEYS = {
    "seed",
    "num_mc_samples",
    "max_samples",
    "caps",
    "prior_shuffle_min_nats",
    "min_active_dims",
    "event_lag_window_s",
    "bootstrap_resamples",
    "clock_margin_min_nats",
    "figure_format",
    "max_hours_before_delivery",
}


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
def test_the_committed_delta_is_where_the_module_looks_for_it() -> None:
    assert DEFAULT_OVERRIDES_PATH.is_file()
    assert DEFAULT_OVERRIDES_PATH.name == "eval_overrides.yaml"
    # This package's own, not the pipeline it was forked from: the module resolves the path from
    # its own ``__file__``, so a copy that forgot to move would still import cleanly.
    assert DEFAULT_OVERRIDES_PATH.parents[2].name == "lag_attn_cfs"


def test_the_delta_carries_no_base_key(overrides) -> None:
    """A ``base:`` chain here would inherit from whatever a committed config currently says
    rather than from what the run trained under, which is the drift the merge exists to avoid."""
    assert "base" not in overrides


def test_a_delta_carrying_a_base_key_is_refused(tmp_path) -> None:
    path = tmp_path / "bad_overrides.yaml"
    path.write_text("base: ../../configs/default.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="base"):
        load_eval_overrides(path)


def test_a_missing_delta_raises_rather_than_merging_nothing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_eval_overrides(tmp_path / "absent.yaml")


def test_the_delta_overrides_only_what_genuinely_differs(overrides) -> None:
    """A further top-level change would be a second config in disguise."""
    assert set(overrides) == {"general_config", "dataset_config", "eval_config"}
    assert set(overrides["general_config"]) == {"batch_size"}
    assert set(overrides["general_config"]["batch_size"]) == {"test"}
    assert set(overrides["dataset_config"]) == {
        "vae_test_datasets", "stat_path", "dataloader_config"
    }
    assert set(_dataset_kwargs(overrides)) == {"load_fields", "cache_size"}


def test_the_delta_points_at_the_eight_causal_holdout_subgroup_shards(overrides) -> None:
    shards = overrides["dataset_config"]["vae_test_datasets"]
    assert len(shards) == 8
    assert all("k_fold_cross_validation_dataset/test/" in path for path in shards)
    # Deliberately non-existent, so a run fails on a missing file rather than on a `transform`
    # refusal someone might "fix" by dropping the warm-up budget.
    assert all("REPOINT_ME" in path for path in shards)
    # And the placeholder says which VARIANT has to be built: the two-sided files carry every one
    # of these field names, and only the root attribute and the widths tell them apart.
    assert all("REPOINT_ME_causal" in path for path in shards)


def test_the_delta_repoints_the_statistics_file_at_the_same_shards(overrides) -> None:
    r"""The one repoint the raw cells' delta deliberately does not make.

    The causal statistics are accumulated EXCLUDING each channel's warm-up region, which is what
    makes $0$ the channel mean over the region the model reads -- and therefore what makes the
    feature-space climatology baseline right and the source-null control a floor rather than a
    leak. A statistics file from the wrong shards breaks both with every shape still correct.
    """
    stat_path = overrides["dataset_config"]["stat_path"]

    assert "REPOINT_ME_causal" in stat_path
    assert "k_fold_cross_validation_dataset" in stat_path
    # `stat_path`, not `stats_path`: the loader parameter is the other spelling, and a typo here
    # yields None and silently disables normalization.
    assert "stats_path" not in overrides["dataset_config"]


# ---------------------------------------------------------------------------
# The eval_config block
# ---------------------------------------------------------------------------
def test_the_block_carries_exactly_the_specified_keys(overrides) -> None:
    assert set(overrides["eval_config"]) == _SHIPPED_EVAL_KEYS
    assert set(overrides["eval_config"]) <= VALID_KEYS
    # And it validates as written, so the shipped file is not a latent failure.
    assert validate_eval_config(overrides)["seed"] == overrides["eval_config"]["seed"]


def test_the_clock_margin_is_shipped_explicitly_unset(overrides) -> None:
    """Present and null rather than absent: the two resolve identically, and writing it out is
    what tells a reader the INCONCLUSIVE verdict is a decision rather than an oversight."""
    block = overrides["eval_config"]

    assert "clock_margin_min_nats" in block
    assert block["clock_margin_min_nats"] is None
    assert validate_eval_config(overrides)["clock_margin_min_nats"] is None


def test_the_waveform_cap_is_halved_and_the_oracle_cap_stays_absent(overrides) -> None:
    r"""A retained waveform set here is four $(152, 15, 98)$ fp32 tensors -- about $3.4$ MiB per
    sample against the raw cells' $2.0$ MiB -- so 64 holds roughly what their 128 does.

    ``oracle`` is the one cap whose ABSENCE means every segment, so naming a number would reduce
    what the sufficiency probe is fitted on. Absent is the complete setting, and this pins it.

    The two page caps are figure counts and retain nothing. ``pages_per_class`` is PER CLASS, so
    it is not comparable with ``pages`` and is pinned here beside it rather than derived from it.
    """
    caps = overrides["eval_config"]["caps"]

    assert caps == {
        "waveforms": 64, "attention": 64, "pages": 24, "pages_per_class": 10
    }
    assert "oracle" not in caps


# ---------------------------------------------------------------------------
# The merge
# ---------------------------------------------------------------------------
def test_every_override_lands_in_the_merged_config(merged, overrides) -> None:
    assert merged["general_config"]["batch_size"]["test"] == 32
    assert merged["dataset_config"]["vae_test_datasets"] == (
        overrides["dataset_config"]["vae_test_datasets"]
    )
    assert merged["dataset_config"]["stat_path"] == overrides["dataset_config"]["stat_path"]
    assert _dataset_kwargs(merged)["cache_size"] == 0
    assert _dataset_kwargs(merged)["load_fields"] == _dataset_kwargs(overrides)["load_fields"]
    assert merged["eval_config"] == overrides["eval_config"]


def test_the_clinical_fields_are_added_and_the_model_s_own_fields_survive(
    merged, resolved
) -> None:
    """A list replaces wholesale on merge, so the inherited entries are restated in the delta
    rather than extended -- and a restatement that dropped one would be invisible."""
    fields = _dataset_kwargs(merged)["load_fields"]
    for name in _ADDED_LOAD_FIELDS:
        assert name in fields
    for name in _dataset_kwargs(resolved)["load_fields"]:
        assert name in fields, f"the delta's load_fields dropped the inherited {name!r}"


def test_guid_and_epoch_survive_because_the_tile_phase_is_keyed_on_the_pair(merged) -> None:
    r"""``load_fields`` is honoured literally with no forced additions, and the anchor tiling's
    per-segment phase is a stable hash of the recording identifier, the segment's own start time
    (``epoch`` is ``domain_start`` in seconds), the training epoch and the seed.

    Drop either and every segment is decoded at one tile grid forever -- with $A_{\max}$ a geometry
    constant either way, so no shape, no count and no metric differs. Nothing else in the output
    would say so, which is why this is a test rather than a comment.
    """
    fields = _dataset_kwargs(merged)["load_fields"]

    assert "guid" in fields
    assert "epoch" in fields


def test_the_run_s_own_contract_survives_the_merge(merged, resolved) -> None:
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
    # Untouched by the delta: setting one dataset_kwargs entry must not drop the rest of the block.
    assert _dataset_kwargs(merged)["trim_minutes"] == 1.0
    assert merged["general_config"]["batch_size"]["train"] == (
        resolved["general_config"]["batch_size"]["train"]
    )
    # The threshold the whole channel axis follows from, and the floor paired with it.
    vae = merged["model_config"]["VAE_model"]
    assert vae["causal_warmup_budget_steps"] == 134
    assert vae["warmup_period"] == 133


def test_the_target_blocks_are_still_normalised_after_the_merge(merged) -> None:
    """A correctness requirement rather than a preference: ``fhr_st`` and ``fhr_ph`` ARE this
    model's target, and an un-z-scored target makes the Gaussian NLL meaningless with the loader
    raising nothing on its own. The delta does not restate ``normalize_fields``, so this asserts
    that not restating it was safe."""
    normalize = merged["dataset_config"]["dataloader_config"]["normalize_fields"]

    assert "fhr_st" in normalize
    assert "fhr_ph" in normalize


def test_the_inherited_epoch_filter_is_left_alone(merged) -> None:
    r"""``epoch`` is negative and the dataset floor is $-44640$ s, so ``epoch_min: -48000`` is a
    no-op; ``epoch_max: -48000`` would select nothing at all. The delta restates neither, and
    copying the sibling's ``epoch_max`` would be the bug."""
    kwargs = _dataset_kwargs(merged)
    assert kwargs["epoch_min"] == -48000
    assert kwargs["epoch_max"] is None
    assert kwargs["epoch_max"] != -48000
    assert kwargs["label"] is None


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
