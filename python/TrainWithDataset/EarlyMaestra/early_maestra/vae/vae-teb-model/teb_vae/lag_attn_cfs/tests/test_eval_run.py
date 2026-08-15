r"""The evaluation entry point, driven end to end against the generated causal cohort shards.

This is the only place the evaluation package meets a real loader, and therefore the only place
three things can be checked at all: that a checkpoint's own resolved config is found and used,
that recording identifiers survive collation into the output as real strings rather than as
``'unknown'``, and that the summary is JSON a non-Python reader can parse.

Two properties are this cell's own and neither has a sibling analogue.

**The pass must build no model when it is not given one.** Every analysis reads the tables rather
than the checkpoint, so ``--output-dir <a finished run>`` with no ``--checkpoint`` must complete
with nothing constructed -- and the assertion is a binding whose model class raises if it is ever
called, because "it was fast" is not evidence.

**A prefix is not a cap.** ``--max-batches`` stops the loop and ``eval_config.max_samples`` draws
a stratified sample, and the two are routinely confused. Over eight concatenated per-subgroup
shards the first yields one subgroup and one clinical class, which is a whole-population claim
made from one cohort.
"""
from __future__ import annotations

import dataclasses
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_cfs.eval import metrics, preflight, run as run_module
from teb_vae.lag_attn_cfs.eval.binding import CFS_BINDING
from teb_vae.lag_attn_cfs.eval.collect import COLLECTION_FILENAME, load_collection
from teb_vae.lag_attn_cfs.eval.probe import PROBE_FILENAME
from teb_vae.lag_attn_cfs.eval.report_seam import (
    HEADLINE_SCALARS,
    STEPS_FILENAME,
    Report,
    step_records,
)
from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME

#: A stand-in registry for the selection tests: three names in a deliberate order, none of them
#: alphabetical, so "registry order" and "sorted" cannot be confused for one another. Written out
#: rather than taken from the real registry, so the selection assertions keep testing selection
#: as analyses land rather than turning into assertions about which ones happen to be registered.
_REGISTRY = ("forecast", "coupling", "attention")


# =================================================================================================
# The registry
# =================================================================================================
def test_every_registered_name_has_a_callable_behind_it() -> None:
    """Pinned rather than left implicit. Every readout a run reports comes from the shared
    collection pass, so an empty registry would still be a complete run with no *analysis*
    directories -- and a name registered without a module behind it fails at import rather than
    at the step it describes."""
    assert run_module.ANALYSES == tuple(run_module.ANALYSIS_FUNCTIONS)
    assert all(callable(function) for function in run_module.ANALYSIS_FUNCTIONS.values())
    assert all(
        function.__name__ == f"run_{name}_analysis"
        for name, function in run_module.ANALYSIS_FUNCTIONS.items()
    )


def test_cross_subgroup_is_registered_last() -> None:
    """Load-bearing rather than tidy: it reads the per-recording CSVs the analyses above it write,
    so on a single pass it can only test the metrics they have already produced. It does not
    *depend* on them having run -- an absent source is recorded rather than raised -- but a run of
    everything should test everything."""
    assert run_module.ANALYSES[-1] == "cross_subgroup"


def test_the_channel_map_is_the_one_unskippable_step() -> None:
    """It describes the *data* rather than the model, and it is what the band-resolved readout
    joins against -- over the kept channel axis. A run whose channel map could be skipped would be
    a run whose frequency-resolved statements have no definition of a band behind them."""
    assert set(run_module.UNSKIPPABLE_ANALYSES) == {"band_partition"}
    assert all(callable(function) for function in run_module.UNSKIPPABLE_ANALYSES.values())
    assert "band_partition" not in run_module.ANALYSIS_FUNCTIONS


def test_there_is_no_dependency_table() -> None:
    """The real dependency is on files existing on disk rather than on an analysis having run in
    this pass, which is what makes an offline ``--only`` work at all. One line adds the table the
    day a genuine correctness dependency appears."""
    assert not hasattr(run_module, "ANALYSIS_DEPENDENCIES")


def test_the_binding_the_runner_defaults_to_is_this_cells() -> None:
    """A wrong default either refuses by name or -- if two constructors happen to accept the same
    keys -- evaluates one architecture under another's name."""
    assert run_module.main.__defaults__ is not None
    assert run_module.build_parser().prog.endswith("teb_vae.lag_attn_cfs.eval.run")
    assert CFS_BINDING.tag == "lag_attn_cfs"


# =================================================================================================
# Analysis selection
# =================================================================================================
def test_only_returns_registry_order_regardless_of_the_order_typed() -> None:
    """The run order is the pipeline's: a later analysis may read what an earlier one wrote."""
    assert run_module.select_analyses(_REGISTRY, "attention,forecast", None) == [
        "forecast", "attention"
    ]


def test_only_and_skip_compose() -> None:
    assert run_module.select_analyses(_REGISTRY, "forecast,coupling", "coupling") == ["forecast"]


def test_neither_flag_selects_everything() -> None:
    assert run_module.select_analyses(_REGISTRY, None, None) == list(_REGISTRY)


@pytest.mark.parametrize("flag", ["only", "skip"])
def test_an_unknown_name_raises_naming_the_valid_set(flag: str) -> None:
    """A misspelling would otherwise silently run everything (``--only``) or nothing extra
    (``--skip``), which is indistinguishable in the output from having asked for exactly that."""
    selection: Dict[str, Any] = {"only": None, "skip": None, flag: "forcast"}
    with pytest.raises(ValueError, match="forecast"):
        run_module.select_analyses(_REGISTRY, **selection)


def test_an_unskippable_step_is_refused_by_name_rather_than_as_a_typo(monkeypatch) -> None:
    """Named rather than reported as an unknown analysis: an operator who asked for the channel
    map by name has to be told it always runs, not that it does not exist."""
    monkeypatch.setattr(run_module, "UNSKIPPABLE_ANALYSES", {"band_partition": object()})

    with pytest.raises(ValueError, match="not selectable"):
        run_module.select_analyses(_REGISTRY, "band_partition", None)


@pytest.fixture
def shared_registry(monkeypatch):
    """A stand-in shared registry, since the real one is empty until the analyses land.

    Patched rather than asserted against the empty dict: both properties below are about how an
    extra analysis composes *with* shared ones, and neither is testable where there are none.
    """
    registry = {name: object() for name in _REGISTRY}
    monkeypatch.setattr(run_module, "ANALYSIS_FUNCTIONS", registry)
    return registry


def test_a_binding_may_not_override_a_shared_analysis(shared_registry) -> None:
    """An extra analysis is an addition, never an override: silently replacing a shared
    implementation would leave two models reporting different things under one name, which is
    indistinguishable in the output from them agreeing."""
    binding = dataclasses.replace(CFS_BINDING, extra_analyses={"forecast": object()})

    with pytest.raises(ValueError, match="already in the shared registry"):
        run_module.merged_analysis_functions(binding)


def test_a_bindings_extra_analyses_are_appended_in_declaration_order(shared_registry) -> None:
    """Appended after the shared registry, so the second cfs cell reuses this runner rather than
    forking it -- and so its own analyses run last, where they can read what the shared ones
    wrote."""
    binding = dataclasses.replace(
        CFS_BINDING, extra_analyses={"warmup": object(), "source_null": object()}
    )

    assert list(run_module.merged_analysis_functions(binding)) == [
        *_REGISTRY, "warmup", "source_null"
    ]


# =================================================================================================
# A prefix is not a cap
# =================================================================================================
class _ShardedDataset:
    """A dataset shaped like the loader's, over ``n_shards`` concatenated per-subgroup files.

    Only what the cap resolution reads: ``index_map`` and ``paths``, which is how the stratum of
    every dataset index is resolved *before* anything is loaded.
    """

    def __init__(self, n_shards: int = 8, per_shard: int = 10) -> None:
        self.paths = [f"/data/subgroup_{index}.hdf5" for index in range(n_shards)]
        self.index_map = [
            (shard, row) for shard in range(n_shards) for row in range(per_shard)
        ]

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int):
        return self.index_map[index]


class _ShardedLoader:
    """A dataloader-shaped object over :class:`_ShardedDataset`."""

    def __init__(self, dataset: _ShardedDataset) -> None:
        self.dataset = dataset
        self.batch_size = 4
        self.collate_fn = None


def _shards_of(loader, indices) -> set:
    """The distinct source shards a set of dataset positions draws from."""
    keys = run_module.dataset_shard_keys(loader)
    assert keys is not None
    return {keys[int(index)] for index in indices}


def test_a_prefix_over_the_concatenated_shards_draws_one_cohort() -> None:
    """What ``--max-batches`` does, stated so the comparison below is against a real alternative
    rather than a straw one: the test loader is built unshuffled over eight per-subgroup files,
    so its first eight samples are eight segments of the first subgroup."""
    loader = _ShardedLoader(_ShardedDataset())

    assert len(_shards_of(loader, range(8))) == 1


def test_the_sample_cap_is_stratified_and_reaches_every_shard_at_the_same_count() -> None:
    """``eval_config.max_samples`` at the identical count. Stratifying by shard gives every file
    a share proportional to its size, which *guarantees* each one appears at a cap of at least
    the shard count rather than merely making it likely."""
    loader = _ShardedLoader(_ShardedDataset())

    capped, record = run_module.capped_sample_loader(loader, 8, seed=0)

    drawn = list(capped.dataset.indices)  # type: ignore[attr-defined]
    assert record["applied"] is True
    assert record["stratified_by"] == "source_file_basename"
    assert record["n_shards_drawn"] == 8
    assert len(_shards_of(loader, drawn)) == 8
    assert len(_shards_of(loader, range(8))) < len(_shards_of(loader, drawn))


def test_the_sample_cap_is_seeded_so_a_rerun_draws_the_same_samples() -> None:
    loader = _ShardedLoader(_ShardedDataset())

    first, _ = run_module.capped_sample_loader(loader, 16, seed=3)
    second, _ = run_module.capped_sample_loader(loader, 16, seed=3)

    assert list(first.dataset.indices) == list(  # type: ignore[attr-defined]
        second.dataset.indices  # type: ignore[attr-defined]
    )


def test_a_cap_that_did_nothing_says_so_rather_than_reading_as_a_whole_split() -> None:
    """A summary must never have to be read as though the whole split was seen when it was not,
    and the converse case is what makes that record trustworthy."""
    loader = _ShardedLoader(_ShardedDataset())

    returned, record = run_module.capped_sample_loader(loader, None, seed=0)

    assert returned is loader
    assert record["applied"] is False
    assert record["n_drawn"] == record["n_total"] == 80


# =================================================================================================
# The output directory
# =================================================================================================
def test_the_output_directory_is_timestamped_with_a_collision_guard(tmp_path) -> None:
    config = {"general_config": {"tag": "cfs", "folders_config": {"out_dir_base": str(tmp_path)}}}

    first = run_module.make_output_dir(config)
    second = run_module.make_output_dir(config)

    assert first != second
    assert first.name == second.name == run_module.RESULTS_DIRNAME


def test_the_binding_tag_names_the_directory_when_the_config_declares_none(tmp_path) -> None:
    """Two models' runs land in two directories rather than in one told apart only by
    timestamp."""
    config = {"general_config": {"folders_config": {"out_dir_base": str(tmp_path)}}}

    results_dir = run_module.make_output_dir(config)

    assert results_dir.parent.parent.name == f"{CFS_BINDING.tag}-eval"


def test_an_explicit_output_directory_is_used_as_given(tmp_path) -> None:
    result = run_module.make_output_dir({}, tmp_path / "here")

    assert result == tmp_path / "here" / run_module.RESULTS_DIRNAME
    assert result.is_dir()


def test_a_prior_summary_is_preserved_byte_identical_before_it_is_overwritten(tmp_path) -> None:
    """Re-running into a finished directory is the documented offline path, and it is also
    destructive: the summary is opened with mode ``'w'`` and the artifact manifest classifies
    every earlier file as stale, so a one-analysis re-run replaces a complete summary with a
    mostly-null one and exits $0$. The sanity block and the promoted verdicts exist nowhere
    else."""
    results_dir = tmp_path / "run" / run_module.RESULTS_DIRNAME
    results_dir.mkdir(parents=True)
    original = b'{"results": {"pred_gap": 1.0}}'
    (results_dir / run_module.SUMMARY_FILENAME).write_bytes(original)
    steps = b'[{"name": "probe"}]'
    (results_dir / STEPS_FILENAME).write_bytes(steps)

    run_module.make_output_dir({}, tmp_path / "run")

    backups = sorted(results_dir.glob("summary.bak.*.json"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    assert sorted(results_dir.glob("steps.bak.*.json"))[0].read_bytes() == steps
    assert not (results_dir / run_module.SUMMARY_FILENAME).exists(), (
        "a stale summary left in place would read as this pass's result if this pass then failed"
    )


def test_the_preflight_record_is_left_where_a_later_pass_can_read_it(tmp_path) -> None:
    """Deliberately *not* preserved aside: a pass with no checkpoint cannot regenerate the
    causality disclosure, and renaming the record would take it from exactly the pass that needs
    to read it back."""
    results_dir = tmp_path / "run" / run_module.RESULTS_DIRNAME
    results_dir.mkdir(parents=True)
    record = results_dir / preflight.PREFLIGHT_FILENAME
    record.write_bytes(b'{"causality": {}}')

    run_module.make_output_dir({}, tmp_path / "run")

    assert record.is_file()


def test_a_pass_with_no_preflight_record_reports_a_skip_rather_than_defaults(tmp_path) -> None:
    """A disclosure nobody produced must not read as one that passed."""
    record = run_module.read_preflight(tmp_path)

    assert record["skipped"] is True
    assert preflight.PREFLIGHT_FILENAME in record["reason"]


# =================================================================================================
# Failure isolation
# =================================================================================================
def _writing_analysis(filename: str):
    """An analysis that writes one CSV and returns the protocol's keys."""

    def _run(context, *, eval_config, output_dir, probe):
        (Path(output_dir) / filename).write_text("value\n1\n", encoding="utf-8")
        return {"n_samples": 1, "composition": {}, "plan": {"capped": False}}

    return _run


def _raising_analysis(context, *, eval_config, output_dir, probe):
    raise KeyError("mu_full")


def test_one_failing_analysis_costs_only_itself(tmp_path) -> None:
    """Five analyses, the third raising: the two before it keep their output, the two after it
    still run, the failure carries its traceback, and the run exits non-zero."""
    registry = {
        "first": _writing_analysis("first.csv"),
        "second": _writing_analysis("second.csv"),
        "third": _raising_analysis,
        "fourth": _writing_analysis("fourth.csv"),
        "fifth": _writing_analysis("fifth.csv"),
    }
    report = Report()

    run_module.run_analyses(
        report, list(registry), registry,
        context=None, eval_config={}, output_dir=tmp_path, probe=None,
    )

    records = step_records(report.steps)
    assert [record["status"] for record in records] == ["ok", "ok", "failed", "ok", "ok"]
    for name in ("first.csv", "second.csv", "fourth.csv", "fifth.csv"):
        assert (tmp_path / name).is_file()
    assert "KeyError" in records[2]["error"]
    # The frame name, which only a formatted traceback carries -- str(exc) is just "'mu_full'".
    assert "_raising_analysis" in records[2]["traceback"]
    assert report.exit_code() == 1
    assert set(report.results) == {"first", "second", "fourth", "fifth"}


def test_the_step_heartbeat_is_rewritten_as_each_analysis_finishes(tmp_path) -> None:
    """A run killed outright leaves no summary at all, and the question afterwards is which step
    it was inside."""
    registry = {"first": _writing_analysis("first.csv"), "second": _raising_analysis}
    report = Report()

    run_module.run_analyses(
        report, list(registry), registry,
        context=None, eval_config={}, output_dir=tmp_path, probe=None,
    )

    written = json.loads((tmp_path / STEPS_FILENAME).read_text(encoding="utf-8"))
    assert [record["name"] for record in written] == ["first", "second"]
    assert written[1]["status"] == "failed"


def test_the_exit_code_follows_the_steps_and_not_the_sanity_flag() -> None:
    """The asymmetry the offline acceptance gate exists for: a run whose every step succeeded can
    still be one nobody should quote a number from, and a warning that moved the exit code would
    make a failed *step* indistinguishable from a failed *check* in a shell."""
    report = Report()
    report.set("sanity", {"warning": True, "failed": ["kl_identity"], "n_failed": 1})
    report.set("config_warnings", ["eval_config.caps.oracle is inert"])
    report.set("coverage", {"per_analysis": {}, "warnings": ["two analyses disagree"]})

    assert report.exit_code() == 0


# =================================================================================================
# The command line
# =================================================================================================
def test_a_checkpoint_is_not_required_at_parse_time(tmp_path) -> None:
    """Not every readout needs the model -- one computed from a finished run's own tables does
    not -- so the parser does not refuse on behalf of a caller that would not have needed one."""
    parsed = run_module.build_parser().parse_args(["--output-dir", str(tmp_path)])

    assert parsed.checkpoint is None
    assert parsed.num_samples is None, "the draw count comes from eval_config unless overridden"


def test_the_run_still_refuses_to_start_without_one(tmp_path) -> None:
    """A checkpoint is optional only where a finished run's tables stand in for it; an empty
    directory is neither, and a run that produced nothing would be worse than one that said why."""
    with pytest.raises(SystemExit, match="--checkpoint is required"):
        run_module._cli(["--output-dir", str(tmp_path)])


def test_each_argument_records_where_its_value_came_from() -> None:
    """A run's provenance must be unambiguous after the fact rather than reconstructed from a
    shell history -- and the launch dict is resolved per key, so the two sources genuinely mix."""
    values, sources = run_module.resolve_arguments(
        ["--checkpoint", "a.ckpt"], run_args={"device": "cpu"}
    )

    assert (values["checkpoint"], sources["checkpoint"]) == ("a.ckpt", "cli")
    assert (values["device"], sources["device"]) == ("cpu", "config")
    assert (values["only"], sources["only"]) == (None, "default")


def test_a_launch_dict_key_that_is_not_an_argument_raises() -> None:
    """A typo there would otherwise silently do nothing, which is the same class of failure the
    ``eval_config`` validator guards against."""
    with pytest.raises(ValueError, match="max_sample"):
        run_module.resolve_arguments([], run_args={"max_sample": 4})


def test_the_shipped_launch_dict_resolves() -> None:
    """It ships in this file and is never exercised by a normal test run, so a key renamed on the
    parser would be found by an operator pressing Run rather than by the suite."""
    values, _ = run_module.resolve_arguments([])

    assert set(values) == set(run_module.RUN_ARGS)


# =================================================================================================
# JSON safety
# =================================================================================================
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_become_null(value: float) -> None:
    assert run_module.json_safe(value) is None


def test_numpy_and_torch_values_become_plain_python() -> None:
    converted = run_module.json_safe(
        {
            "flag": np.bool_(True),
            "count": np.int64(3),
            "value": np.float32(1.5),
            "array": np.array([1.0, 2.0]),
            "tensor": torch.tensor([3.0, 4.0]),
            "path": Path("a") / "b",
        }
    )

    # np.bool_ is checked before the int branch; otherwise True would serialise as 1.
    assert converted["flag"] is True
    assert converted["count"] == 3 and isinstance(converted["count"], int)
    assert converted["value"] == pytest.approx(1.5)
    assert converted["array"] == [1.0, 2.0]
    assert converted["tensor"] == [3.0, 4.0]
    assert isinstance(converted["path"], str)


# =================================================================================================
# End to end, against one real run
# =================================================================================================
@pytest.mark.slow
def test_the_run_writes_a_summary_and_the_config_it_used(collected_run) -> None:
    assert collected_run["summary_path"].name == run_module.SUMMARY_FILENAME
    assert (collected_run["results_dir"] / RESOLVED_CONFIG_FILENAME).is_file()
    assert collected_run["results_dir"].name == run_module.RESULTS_DIRNAME
    assert collected_run["exit_code"] == 0


@pytest.mark.slow
def test_the_summary_reports_every_section_and_the_registered_verdicts(collected_run) -> None:
    """A subset assertion rather than an exact list: the registry decides the order, a separate
    test asserts uniqueness and that order, and what is pinned here is that the schema can only
    grow."""
    results = collected_run["summary"]["results"]

    assert set(results) >= {"readouts", "latent_health", "lag", "per_recording", "verdicts"}
    names = [verdict["name"] for verdict in results["verdicts"]]
    assert set(names) >= {
        "predictive_improvement",
        "source_specificity",
        "prior_carries_target_state",
        "latent_not_collapsed",
        # The two only this cell can have.
        "coupling_exceeds_availability_clock",
        "anchor_geometry_intact",
    }
    for verdict in results["verdicts"]:
        assert verdict["status"] in {"PASS", "FAIL", "INCONCLUSIVE"}


@pytest.mark.slow
def test_the_verdicts_are_unique_and_in_registry_order(collected_run) -> None:
    """The list is read by name *and* by position -- by the acceptance gate and by the arm
    tables -- so a duplicate or a reordering is a silent change of meaning."""
    names = [verdict["name"] for verdict in collected_run["summary"]["results"]["verdicts"]]

    assert names == list(metrics.VERDICT_ORDER)
    assert len(names) == len(set(names))


@pytest.mark.slow
def test_the_unset_clock_margin_leaves_its_verdict_inconclusive_with_the_measurement(
    collected_run,
) -> None:
    """The shipped ``null`` is the setting, not an omission: a provisional threshold would decide
    a FAIL on the very run that is supposed to measure where the boundary belongs. What is *not*
    conditional on the key is the number."""
    summary = collected_run["summary"]
    verdict = next(
        entry for entry in summary["results"]["verdicts"]
        if entry["name"] == "coupling_exceeds_availability_clock"
    )

    assert summary["eval_config"]["clock_margin_min_nats"] is None
    assert verdict["status"] == "INCONCLUSIVE"
    assert math.isfinite(summary["results"]["readouts"]["coupling_minus_clock"])


@pytest.mark.slow
def test_the_summary_carries_the_checkpoint_and_config_it_evaluated(collected_run) -> None:
    summary = collected_run["summary"]

    assert Path(summary["checkpoint"]) == collected_run["checkpoint"]
    assert Path(summary["config"]).name == RESOLVED_CONFIG_FILENAME
    assert summary["device"] == "cpu"


@pytest.mark.slow
def test_the_summary_is_json_a_non_python_reader_can_parse(collected_run) -> None:
    """``json.dump`` emits the bare tokens ``NaN`` and ``Infinity`` for non-finite floats, which
    round-trip through Python and are rejected by every other parser."""
    for token in ("NaN", "Infinity", "-Infinity"):
        assert token not in collected_run["text"]


@pytest.mark.slow
def test_the_summary_holds_no_tensors_or_numpy_scalars(collected_run) -> None:
    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        else:
            assert isinstance(value, (str, int, float, bool, type(None))), type(value)

    walk(collected_run["summary"])


@pytest.mark.slow
def test_recording_identifiers_reach_the_output_as_real_guids(collected_run) -> None:
    """The one thing a stub batch cannot check: ``guid`` survives collation as a ``list[str]``,
    is never moved to a device, and must arrive as the shard's own identifiers rather than as the
    ``'unknown'`` fallback."""
    per_recording = collected_run["summary"]["results"]["per_recording"]

    assert per_recording, "no recordings were aggregated"
    assert "unknown" not in per_recording


@pytest.mark.slow
def test_the_readouts_are_finite_and_the_gap_is_the_difference(collected_run) -> None:
    readouts = collected_run["summary"]["results"]["readouts"]

    for name, value in readouts.items():
        assert value is not None and math.isfinite(value), f"{name} is {value}"
    assert readouts["pred_gap"] == pytest.approx(
        readouts["nll_base_block"] - readouts["nll_full_block"], rel=1e-5
    )
    assert readouts["coupling_minus_clock"] == pytest.approx(
        readouts["source_conditioned_kl_raw"] - readouts["kld_source_null"], rel=1e-5
    )


@pytest.mark.slow
def test_the_dumped_config_round_trips_through_the_config_reader(collected_run) -> None:
    """The merged result is the run's durable record, so it has to be readable back as one
    document -- and carry no ``base:``, which in a dumped config would point at whatever a
    committed file says today rather than at what this run used."""
    written = collected_run["results_dir"] / RESOLVED_CONFIG_FILENAME

    reloaded = load_config(str(written))

    assert "base" not in reloaded
    assert reloaded == yaml.safe_load(written.read_text(encoding="utf-8"))


@pytest.mark.slow
def test_the_summary_records_both_values_of_every_override(collected_run) -> None:
    """An evaluation runs on a configuration that is deliberately *not* the one the checkpoint
    trained under. A divergence recorded nowhere is indistinguishable from an accident, so both
    values travel into the summary.

    The shard list is deliberately **not** asserted to differ here: this fixture trains and
    evaluates on the same generated shards, so the repoint is a no-op on it. That is a property of
    the fixture rather than of the delta, and the cohort block is where it shows up honestly -- as
    ``training_cohort_disjoint`` being ``False``.
    """
    entries = {
        record["path"]: record
        for record in collected_run["summary"]["config_overrides"]["entries"]
    }

    fields = entries["dataset_config.dataloader_config.dataset_kwargs.load_fields"]
    # The substantive override, and the one every clinical question depends on: the training
    # contract carries none of these, and the loader skips a field it was not asked for, silently.
    for name in ("target", "epoch", "guid", "cs_label", "bg_label", "time_from_labor_onset"):
        assert name in fields["eval_value"], name
    assert "target" not in fields["run_value"]
    assert fields["run_value"] != fields["eval_value"]
    # Both values, for every entry, whether or not they differ.
    for path, record in entries.items():
        assert "run_value" in record and "eval_value" in record, path
    assert "dataset_config.vae_test_datasets" in entries
    assert "dataset_config.stat_path" in entries, (
        "this cell repoints the statistics too, which the raw cells' delta does not"
    )


@pytest.mark.slow
def test_the_run_carries_a_log_beside_its_artifacts(collected_run) -> None:
    assert (collected_run["results_dir"] / run_module.LOG_FILENAME).is_file()


@pytest.mark.slow
def test_the_run_used_a_single_process_loader(collected_run) -> None:
    """Spawn workers over a multi-file HDF5 dataset silently truncate every pass after the
    first, and an evaluation makes many passes."""
    written = yaml.safe_load(
        (collected_run["results_dir"] / RESOLVED_CONFIG_FILENAME).read_text(encoding="utf-8")
    )
    loader_config = written["dataset_config"]["dataloader_config"]

    assert loader_config["num_workers"] == 0
    assert loader_config["persistent_workers"] is False


@pytest.mark.slow
def test_this_cell_reports_no_input_delay(collected_run) -> None:
    """Structurally zero rather than assumed: preflight refuses a reach budget on this cell, so
    the source channels are not shifted and a lag reads as stored-coefficient time. Both places
    are asserted because different code paths write them, and only their agreement makes the lag
    axis trustworthy."""
    summary = collected_run["summary"]

    assert summary["source_delay_steps"] == 0
    assert summary["results"]["lag"]["delay_steps"] == 0


@pytest.mark.slow
def test_the_population_probe_is_written_and_read_back_by_the_sanity_block(collected_run) -> None:
    """The probe is a whole extra iteration of the split, so it runs once and leaves its record;
    the population checks read that file rather than re-walking the loader. The block is
    three-valued and deliberately does not move the exit code."""
    record = json.loads(
        (collected_run["results_dir"] / PROBE_FILENAME).read_text(encoding="utf-8")
    )
    sanity = collected_run["summary"]["results"]["sanity"]

    assert record["n_samples"] > 0
    assert record["per_file"] and all(count > 0 for count in record["per_file"].values())
    for name in ("per_file_counts", "classes_present", "target_not_truncated"):
        assert sanity["checks"][name]["verdict"] in {"pass", "fail", "inconclusive"}
    assert collected_run["exit_code"] == 0


@pytest.mark.slow
def test_every_step_carries_its_own_elapsed_time(collected_run) -> None:
    steps = collected_run["summary"]["steps"]

    assert steps, "the run recorded no steps at all"
    for record in steps:
        assert record["status"] == "ok"
        assert isinstance(record["elapsed_s"], (int, float))


@pytest.mark.slow
def test_peak_memory_is_absent_rather_than_zero_on_cpu(collected_run) -> None:
    """A $0.00$ GB peak reads as "measured, and the run used no memory", which is a claim a CPU
    box cannot make."""
    assert "max_memory_allocated_gb" not in collected_run["summary"]["results"]


@pytest.mark.slow
def test_the_summary_carries_the_exit_code_and_the_failure_list(collected_run) -> None:
    summary = collected_run["summary"]

    assert summary["exit_code"] == collected_run["exit_code"] == 0
    assert summary["failed"] == [] and summary["n_failed"] == 0
    assert summary["n_steps"] == len(summary["steps"])


# =================================================================================================
# The run context
# =================================================================================================
@pytest.mark.slow
def test_the_run_context_records_what_the_arm_tables_consume(collected_run) -> None:
    """Each checked against an independent reading rather than merely present."""
    context = collected_run["summary"]["run_context"]
    checkpoint = collected_run["checkpoint"]

    task = run_module.load_task(checkpoint, torch.device("cpu"))
    assert context["n_parameters"] == sum(p.numel() for p in task.orig_model.parameters())
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert context["train_epoch"] == blob["epoch"]
    assert context["model_class"] == blob["model_class"]
    assert context["model_class"] == CFS_BINDING.model_cls.__name__

    coverage = context["anchor_coverage_frac"]
    assert coverage["n_anchors"] > 0
    # The per-anchor table holds contributing anchors only, so the floor bounds the minimum --
    # and the note must say so, because that truncation is what a reader revising the floor
    # against this distribution has to know. To float32, the dtype the coverage travels in.
    assert coverage["min"] >= float(task.orig_model.coverage_floor) - 1e-6
    assert coverage["min"] <= coverage["median"] <= coverage["max"] <= 1.0
    assert "coverage_floor" in coverage["note"]

    scale = context["observed_loss_scale"]
    readouts = collected_run["summary"]["results"]["readouts"]
    assert scale["nll_full_block"] == pytest.approx(readouts["nll_full_block"])
    assert scale["main_loss_estimate"] == pytest.approx(
        scale["lambda_full"] * scale["nll_full_block"]
        + scale["lambda_base"] * scale["nll_base_block"]
        + scale["beta_end"] * scale["source_conditioned_kl_raw"]
        + scale["beta_prior"] * (scale["prior_rate"] or 0.0)
    )


@pytest.mark.slow
def test_the_run_context_records_both_anchor_strides_and_the_target_axis(collected_run) -> None:
    r"""This cell's own two entries, and neither is optional. A table read against the training
    CSV is unreadable without the training stride -- $A_{\max}$ differs by a factor of it -- and a
    block score is a sum over $H \cdot C_{\mathrm{keep}}$ coefficients, so two budgets' nats are
    two different denominators.
    """
    context = collected_run["summary"]["run_context"]
    task = run_module.load_task(collected_run["checkpoint"], torch.device("cpu"))
    model = task.orig_model

    geometry = context["anchor_geometry"]
    assert (geometry["anchor_phase"], geometry["anchor_stride"]) == (0, 1)
    assert geometry["training_anchor_stride"] == int(model.anchor_stride)
    assert geometry["anchor_floor"] == int(model.warmup_period)
    assert geometry["anchors_per_sample"] == metrics.expected_anchors_per_sample(model)

    axis = context["target_axis"]
    assert axis["target_declared_width"] == int(model.c_y)
    assert axis["target_kept_width"] == int(model.decoder_out_channels)
    assert axis["block_width"] == axis["horizon"] * axis["target_kept_width"]

    budget = context["warmup_budget"]
    assert budget["target_kept"] == axis["target_kept_width"]
    assert budget["target_declared"] == axis["target_declared_width"]
    assert budget["anchor_floor"] == geometry["anchor_floor"]
    assert budget["target_warm_frac"] == pytest.approx(1.0)
    assert budget["target_max_warmup_steps"] is not None


@pytest.mark.slow
def test_the_headline_block_names_every_registered_scalar(collected_run) -> None:
    """Present for every entry, resolved or not. A path that does not resolve yields ``None``
    rather than being omitted, because a *missing key* and a *null value* read differently to a
    ``pandas`` merge across two arms -- and which entries resolve on this run is the subject of
    the reproducibility suite's own check."""
    headline = collected_run["summary"]["results"]["headline"]

    missing = [name for name, _path in HEADLINE_SCALARS if name not in headline]
    assert missing == [], missing
    assert headline["pred_gap_mc_nats"] is not None
    assert headline["source_conditioned_kl_raw_nats"] is not None


# =================================================================================================
# The offline path
# =================================================================================================
class _RefusesToConstruct:
    """A model class that raises if it is ever built.

    The assertion behind "a pass with no checkpoint builds no model". Timing is not evidence --
    a run that happened to be fast could still have loaded one -- and this makes the claim
    structural.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("the model was constructed on a pass with no checkpoint")


@pytest.mark.slow
def test_a_pass_with_no_checkpoint_reruns_the_analyses_and_builds_no_model(
    collected_run, tmp_path
) -> None:
    """The whole point of splitting collection from emission, and the form a re-run takes after a
    multi-hour pass failed at its ninth step.

    Against a *copy* of the finished directory rather than the directory itself: the session
    fixture is shared, and a second pass into it would replace the summary every other test on
    this run is reading.
    """
    run_dir = tmp_path / "offline"
    shutil.copytree(collected_run["results_dir"].parent, run_dir)
    binding = dataclasses.replace(CFS_BINDING, model_cls=_RefusesToConstruct)

    exit_code = run_module.main(None, run_dir, device="cpu", binding=binding)

    results_dir = run_dir / run_module.RESULTS_DIRNAME
    summary = json.loads(
        (results_dir / run_module.SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert summary["checkpoint"] is None
    # The tables were reused rather than re-collected, so every readout is the first pass's.
    assert summary["results"]["readouts"] == collected_run["summary"]["results"]["readouts"]
    # The checkpoint facts a model-free pass cannot reproduce are absent rather than invented.
    assert summary["run_context"]["n_parameters"] is None
    assert summary["run_context"]["warmup_budget"] is None
    # But the geometry is not: it was read back off the collection record, which is what makes an
    # offline re-run able to place an anchor on a time axis at all.
    assert summary["run_context"]["anchor_geometry"]["anchor_stride"] == 1
    # And the prior summary was preserved rather than silently replaced.
    assert sorted(results_dir.glob("summary.bak.*.json"))


@pytest.mark.slow
def test_an_offline_pass_reads_the_preflight_record_the_checkpointed_one_wrote(
    collected_run, tmp_path
) -> None:
    """It cannot regenerate the causality disclosure, so it reads it -- and the summary promotes
    the same statement rather than omitting it."""
    run_dir = tmp_path / "offline_preflight"
    shutil.copytree(collected_run["results_dir"].parent, run_dir)
    binding = dataclasses.replace(CFS_BINDING, model_cls=_RefusesToConstruct)

    run_module.main(None, run_dir, device="cpu", binding=binding)

    summary = json.loads(
        (run_dir / run_module.RESULTS_DIRNAME / run_module.SUMMARY_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert summary["causality"]["statement"] == preflight.CAUSALITY_STATEMENT
    assert summary["preflight"]["reused_from"]


def test_an_empty_directory_names_what_is_missing_rather_than_producing_nothing(
    tmp_path,
) -> None:
    """A directory with a dumped config but no tables is the one case that reaches the collection
    branch with nothing to collect *with*."""
    results_dir = tmp_path / "bare" / run_module.RESULTS_DIRNAME
    results_dir.mkdir(parents=True)
    (results_dir / RESOLVED_CONFIG_FILENAME).write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=COLLECTION_FILENAME):
        run_module.main(None, tmp_path / "bare", device="cpu")


# =================================================================================================
# Checkpoint loading
#
# Rebuilt through ``eval/probe.py``, which owns that path: this model's forward takes five
# arguments and refuses a missing anchor phase above stride 1, so the probe measures its contract
# against a rebuilt model and a second reconstruction here would be a second place for the two to
# disagree. Re-exported so ``run.load_task`` is the same function.
# =================================================================================================
def test_the_loader_is_the_probes_own_rather_than_a_second_implementation() -> None:
    from teb_vae.lag_attn_cfs.eval import probe as probe_module

    assert run_module.load_task is probe_module.load_task
    assert run_module.resolved_config_for is probe_module.resolved_config_for
    assert run_module.read_checkpoint is probe_module.read_checkpoint


def _mutated(checkpoint: Path, tmp_path: Path, mutate) -> Path:
    """Save a copy of the checkpoint with one key changed."""
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mutate(blob)
    path = tmp_path / "mutated.ckpt"
    torch.save(blob, path)
    return path


@pytest.mark.slow
def test_a_checkpoint_from_another_model_is_refused(collected_run, tmp_path) -> None:
    def _rename(blob):
        blob["model_class"] = "SomeOtherModel"

    with pytest.raises(ValueError, match="model_class"):
        run_module.load_task(
            _mutated(collected_run["checkpoint"], tmp_path, _rename), torch.device("cpu")
        )


@pytest.mark.slow
def test_a_checkpoint_without_model_kwargs_is_refused(collected_run, tmp_path) -> None:
    """``SeqVaeLagAttnCfs()`` with no arguments builds the *production* geometry rather than
    raising -- and builds it **ungated**, with no keep-index and no warm-up mask -- so guessing
    would silently evaluate a different model over a different channel axis."""

    def _drop(blob):
        blob["model_kwargs"] = {}

    with pytest.raises(RuntimeError, match="no 'model_kwargs'"):
        run_module.load_task(
            _mutated(collected_run["checkpoint"], tmp_path, _drop), torch.device("cpu")
        )


@pytest.mark.slow
def test_the_loaded_weights_are_the_checkpoints_own(collected_run) -> None:
    """Every parameter, not merely a shape-compatible model."""
    checkpoint = collected_run["checkpoint"]
    task = run_module.load_task(checkpoint, torch.device("cpu"))
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = {
        key[len("_orig_model.") :]: value
        for key, value in blob["state_dict"].items()
        if key.startswith("_orig_model.")
    }

    assert saved, "the checkpoint's state dict is not wrapper-prefixed as expected"
    for name, parameter in task.orig_model.state_dict().items():
        assert torch.equal(parameter, saved[name]), f"{name} did not load"


@pytest.mark.slow
def test_the_loaded_task_is_in_evaluation_mode(collected_run) -> None:
    """Dropout live during evaluation would leave the attention rows not summing to one, so the
    lag attribution would not be a decomposition of anything."""
    assert run_module.load_task(
        collected_run["checkpoint"], torch.device("cpu")
    ).training is False


# =================================================================================================
# The cohort block
# =================================================================================================
@pytest.mark.slow
def test_the_cohort_block_counts_segments_and_recordings_on_both_axes(collected_run) -> None:
    """Both levels, because they answer different questions and routinely disagree: a subgroup
    with many segments and three recordings is one whose statistics have $n = 3$."""
    block = collected_run["summary"]["results"]["cohort"]
    results = collected_run["summary"]["results"]

    assert block["n_segments"] == results["n_samples"]
    assert block["n_recordings"] == results["n_recordings"]
    assert sum(block["by_subgroup"]["segments"].values()) == results["n_samples"]
    assert sum(block["by_subgroup"]["recordings"].values()) == results["n_recordings"]
    assert set(block["by_clinical_class"]["segments"]) <= {"healthy", "acidosis", "hie"}


@pytest.mark.slow
def test_the_fixture_evaluates_its_own_training_set_and_the_summary_says_so(collected_run) -> None:
    """Computed rather than written down, and this fixture is the case that makes the computation
    worth having: it trains and evaluates on the same generated shards, so the two cohorts overlap
    and the out-of-distribution statement must be **absent**.

    A constant asserting disjointness would outlive the configuration that made it true, and a
    summary claiming a leakage-free evaluation of a run that evaluated its own training set is
    worse than one claiming nothing.
    """
    block = collected_run["summary"]["results"]["cohort"]

    assert block["training_cohort_disjoint"] is False
    assert block["training_cohort_overlap"]
    assert "out_of_distribution" not in block


@pytest.mark.slow
def test_the_non_comparability_sentence_is_in_every_summary(collected_run) -> None:
    """An eval score and a ``test_*`` metric logged during training are computed over different
    populations, and nothing in either number says so."""
    block = collected_run["summary"]["results"]["cohort"]

    assert "not comparable" in block["non_comparability"]
    assert "different populations" in block["non_comparability"]


@pytest.mark.slow
def test_the_collection_record_reaches_the_summary_without_its_readouts(collected_run) -> None:
    """``results`` lives in the record so a directory whose forward pass is skipped still answers
    every question the pass answered; repeating it in the summary would double the largest object
    in the file."""
    block = collected_run["summary"]["collection"]
    collection = load_collection(collected_run["results_dir"])

    assert "results" not in block
    assert block["n_per_sample_rows"] == len(collection.per_sample)
    assert block["geometry"]["target_keep_index"]
    assert block["cost"]["samples_per_second"] > 0.0
