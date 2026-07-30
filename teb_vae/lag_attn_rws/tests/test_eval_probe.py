r"""The loader probe reports what the split actually yielded, and refuses four ways it cannot.

Nothing else in a run reports per-file coverage, so a shard that silently contributes nothing is
invisible in every other output -- that is the predecessor's hardest bug, and this is the only
artifact that can see it. The four refusals are tested one deliberately broken config at a time,
each asserting the message rather than only the raise: a guard whose message does not name the
fix costs the same debugging session the guard was added to prevent.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from teb_vae.lag_attn_rws.eval import probe as probe_module
from teb_vae.lag_attn_rws.tests.conftest import (
    MULTI_CLASS_GUIDS_PER_SHARD,
    MULTI_CLASS_SEGMENTS_PER_GUID,
    MULTI_CLASS_SUBGROUPS,
    write_multi_class_shards,
)

_N_SHARDS = len(MULTI_CLASS_SUBGROUPS)
_N_SAMPLES = _N_SHARDS * MULTI_CLASS_GUIDS_PER_SHARD * MULTI_CLASS_SEGMENTS_PER_GUID


@pytest.fixture(scope="module")
def record(multi_class_config, multi_class_loader) -> dict:
    """One probe pass, shared by every read-only assertion below."""
    return probe_module.run_probe(
        multi_class_loader,
        configured_files=multi_class_config["dataset_config"]["vae_test_datasets"],
    )


def _loader_for(config: dict):
    from train.data_module import GraphDataModule

    return GraphDataModule(config).test_dataloader()


# ---------------------------------------------------------------------------
# What it records
# ---------------------------------------------------------------------------
def test_the_probe_counts_every_sample_and_every_shard(record) -> None:
    assert record["n_samples"] == _N_SAMPLES
    assert record["n_unique_guids"] == _N_SHARDS * MULTI_CLASS_GUIDS_PER_SHARD
    assert set(record["per_file"]) == {f"{name}.hdf5" for name in MULTI_CLASS_SUBGROUPS}
    assert sum(record["per_file"].values()) == _N_SAMPLES


def test_the_class_histogram_is_keyed_by_clinical_name_not_by_a_stored_value(record) -> None:
    """Keyed on the raw stored value it produced entries like ``'0.75'`` for a recording whose
    valid steps were partial -- enough to make a single-class split report several."""
    assert set(record["per_target_class"]) == {"healthy", "acidosis", "hie"}
    assert sum(record["per_target_class"].values()) == _N_SAMPLES


def test_both_label_axes_and_the_source_vectors_are_recorded(record) -> None:
    assert sum(record["per_cs_label"].values()) == _N_SAMPLES
    assert sum(record["per_bg_label"].values()) == _N_SAMPLES
    assert len(record["guids"]) == _N_SAMPLES
    assert len(record["source_files"]) == _N_SAMPLES


def test_the_probe_answers_whether_validity_is_ever_fractional(record) -> None:
    """The question the mask arithmetic downstream depends on: on binary weight a masked mean is
    an intersection of indicators, on fractional weight it is a weighted mean."""
    assert record["weight"]["binary"] is False
    assert 0.0 < record["weight"]["zero_frac"] < 1.0
    assert record["weight"]["min"] == 0.0
    assert record["weight"]["max"] == 1.0


def test_the_raw_target_record_spans_every_step_rather_than_one_per_recording(record) -> None:
    """The class histogram cannot answer this: a recording's whole-weight steps hide exactly the
    fractional boundaries that make reading ``target`` directly wrong."""
    values = record["target_values"]
    assert values["n_values"] > record["n_samples"]
    assert values["any_fractional"] is True
    assert values["n_non_finite"] == 0


def test_the_epoch_range_is_reported_in_seconds_and_hours(record) -> None:
    epoch = record["epoch"]
    assert epoch["max_seconds"] < 0.0, "epoch counts backwards from delivery"
    assert epoch["min_hours"] == pytest.approx(epoch["min_seconds"] / 3600.0)
    assert epoch["max_hours"] - epoch["min_hours"] >= 3.0


def test_the_absent_labour_onset_times_are_counted_rather_than_dropped(record) -> None:
    onset = record["time_from_labor_onset"]
    assert onset["n_values"] == _N_SAMPLES
    assert 0 < onset["n_nan"] < _N_SAMPLES


def test_the_probe_makes_exactly_one_pass(multi_class_loader) -> None:
    """Pure bookkeeping over the loader: it does no forward of its own."""
    passes = {"count": 0}

    class CountingLoader:
        def __iter__(self):
            passes["count"] += 1
            return iter(multi_class_loader)

    probe_module.run_probe(CountingLoader())
    assert passes["count"] == 1


# ---------------------------------------------------------------------------
# What it writes and prints
# ---------------------------------------------------------------------------
def test_the_written_json_omits_the_per_sample_vectors(multi_class_loader, tmp_path) -> None:
    """One row per sample does not belong in a file meant to be read at a glance."""
    probe_module.run_probe(multi_class_loader, output_dir=tmp_path)
    written = json.loads(
        (tmp_path / probe_module.PROBE_FILENAME).read_text(encoding="utf-8")
    )

    assert written["n_samples"] == _N_SAMPLES
    for key in probe_module.IN_MEMORY_KEYS:
        assert key not in written


def test_the_cohort_table_shows_every_count_beside_its_share(record) -> None:
    table = probe_module.format_cohort_table(record)

    for name in MULTI_CLASS_SUBGROUPS:
        assert f"{name}.hdf5" in table
    for name in ("healthy", "acidosis", "hie"):
        assert name in table
    assert f"samples          {_N_SAMPLES}" in table
    assert "%" in table, "a bare count without its share hides the coverage"


def test_the_cohort_table_survives_a_record_with_nothing_in_it() -> None:
    """It is printed after a pass that may have found very little; it must not raise there."""
    assert probe_module.format_cohort_table({"n_samples": 0}).startswith("cohort")


# ---------------------------------------------------------------------------
# The four refusals
# ---------------------------------------------------------------------------
def test_an_empty_split_raises() -> None:
    with pytest.raises(RuntimeError, match="no samples at all"):
        probe_module.run_probe([])


def test_a_configured_shard_yielding_nothing_raises(multi_class_config, multi_class_loader) -> None:
    """A departure from the predecessor, which only logged it and was ignored."""
    configured = list(multi_class_config["dataset_config"]["vae_test_datasets"])
    with pytest.raises(RuntimeError, match="yielded zero samples") as excinfo:
        probe_module.run_probe(
            multi_class_loader,
            configured_files=configured + ["/data/k_fold/test/hie_cs.hdf5"],
        )
    assert "hie_cs.hdf5" in str(excinfo.value)


def test_a_batch_capped_pass_warns_instead_of_raising(multi_class_config, multi_class_loader) -> None:
    """A batch cap reads a prefix of the concatenated index, so a missing shard is expected."""
    configured = list(multi_class_config["dataset_config"]["vae_test_datasets"])
    capped = probe_module.run_probe(
        multi_class_loader, configured_files=configured, max_batches=1
    )
    assert capped["n_batches"] == 1
    assert capped["n_samples"] < _N_SAMPLES
    assert len(capped["per_file"]) < _N_SHARDS


def test_a_missing_required_field_raises_naming_it(multi_class_config) -> None:
    """The loader skips a field a shard does not carry, silently, so a dropped ``target`` would
    present as "no classes found" rather than as a data problem."""
    config = copy.deepcopy(multi_class_config)
    kwargs = config["dataset_config"]["dataloader_config"]["dataset_kwargs"]
    kwargs["load_fields"] = [name for name in kwargs["load_fields"] if name != "target"]

    with pytest.raises(RuntimeError, match="missing required field") as excinfo:
        probe_module.run_probe(_loader_for(config))
    message = str(excinfo.value)
    assert "'target'" in message
    assert "load_fields" in message, "the message must name where the fix goes"


def test_a_guid_in_two_shards_raises(multi_class_config, tmp_path) -> None:
    """The holdout split is one pool with no fold loop, so a duplicated recording is counted
    twice and lands on both sides of every between-subgroup comparison."""
    import h5py

    shards = write_multi_class_shards(tmp_path / "duplicated")
    with h5py.File(shards[1], "r+") as handle:
        original = [value.decode("utf-8") for value in handle["guid"][()]]
        del handle["guid"]
        # One recording from the first shard, re-used under a second subgroup.
        duplicated = ["HEALTHY_NO_BG_NO_CS_000"] + original[1:]
        handle.create_dataset("guid", data=duplicated, dtype=h5py.string_dtype(encoding="utf-8"))

    config = copy.deepcopy(multi_class_config)
    config["dataset_config"]["vae_test_datasets"] = list(shards)

    with pytest.raises(RuntimeError, match="more than one shard") as excinfo:
        probe_module.run_probe(_loader_for(config))
    assert "HEALTHY_NO_BG_NO_CS_000" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The standalone command
# ---------------------------------------------------------------------------
def test_the_probe_runs_from_a_config_alone(multi_class_config, tmp_path) -> None:
    """No checkpoint, no model, no GPU -- which is what makes it the first thing to run."""
    record = probe_module.probe_config(multi_class_config, output_dir=tmp_path)

    assert record["n_samples"] == _N_SAMPLES
    assert (tmp_path / probe_module.PROBE_FILENAME).is_file()


def test_main_merges_the_committed_overrides_over_the_run_s_own_config(
    multi_class_config, multi_class_shards, tmp_path
) -> None:
    """The path the command line takes: a run config in, the delta merged over it, one pass."""
    import yaml

    # A resolved config as a training run writes one, but pointing at the training shards: the
    # delta's own vae_test_datasets are REPOINT_ME placeholders, so they are edited here exactly
    # as an operator would edit the committed file.
    config_path = tmp_path / "resolved_config.yaml"
    run_config = copy.deepcopy(multi_class_config)
    run_config.pop("eval_config")
    config_path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")

    overrides_path = tmp_path / "overrides.yaml"
    overrides = {
        "dataset_config": {"vae_test_datasets": list(multi_class_shards)},
        "eval_config": {"seed": 3},
    }
    overrides_path.write_text(yaml.safe_dump(overrides, sort_keys=False), encoding="utf-8")

    record = probe_module.main(
        config_path, overrides=overrides_path, output_dir=tmp_path / "results"
    )

    assert record["n_samples"] == _N_SAMPLES
    assert (tmp_path / "results" / probe_module.PROBE_FILENAME).is_file()


def test_the_parser_requires_a_config_and_defaults_the_rest() -> None:
    args = probe_module.build_parser().parse_args(["--config", "run/resolved_config.yaml"])

    assert args.config == "run/resolved_config.yaml"
    assert args.overrides is None and args.output_dir is None and args.max_batches is None

    with pytest.raises(SystemExit):
        probe_module.build_parser().parse_args([])


def test_the_module_is_runnable_on_its_own() -> None:
    """``python -m ...eval.probe`` is the sprint's demo; a missing entry point breaks it."""
    source = Path(probe_module.__file__).read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in source
