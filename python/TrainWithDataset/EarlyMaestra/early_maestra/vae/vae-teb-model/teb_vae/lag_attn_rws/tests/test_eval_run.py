r"""The evaluation entry point, driven end to end against the committed tiny shard.

This is the only place the evaluation package meets a real loader, and therefore the only place
three things can be checked at all: that a checkpoint's own resolved config is found and used,
that recording identifiers survive collation into the output as real strings rather than as
``'unknown'``, and that the summary is JSON a non-Python reader can parse.

The negative cases around checkpoint loading carry most of the remaining weight. Each of them --
a blob from another model, a blob with no architecture, a state dict that does not align -- would
otherwise produce a complete set of entirely plausible numbers from a randomly initialised model.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws.eval import metrics, run as run_module
from teb_vae.lag_attn_rws.eval.report_seam import Report, STEPS_FILENAME, step_records
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME

from .conftest import TINY_KWARGS

#: A stand-in registry for the selection tests: three names in a deliberate order, none of them
#: alphabetical, so "registry order" and "sorted" cannot be confused for one another.
_REGISTRY = ("forecast", "coupling", "attention")

# ``trained_run`` and ``evaluated`` are session fixtures in ``conftest.py``: the preflight suite
# asks about the same run, and evaluating it twice would double the suite's only expensive step.

# =============================================================================
# End to end
# =============================================================================
def test_the_run_writes_a_summary_and_the_config_it_used(evaluated):
    assert evaluated["summary_path"].name == run_module.SUMMARY_FILENAME
    assert (evaluated["results_dir"] / RESOLVED_CONFIG_FILENAME).is_file()
    assert evaluated["results_dir"].name == run_module.RESULTS_DIRNAME


def test_the_summary_reports_every_section_and_the_registered_verdicts(evaluated):
    """Relaxed from an exact list of four verdict names to a **subset** assertion, deliberately.

    The exact-equality form pinned the schema shut: every later sprint adds a criterion -- the
    prior-variance floor, the decoder-variance clamp, calibration against nominal -- and each
    would have broken this test for no reason other than that it was written before them. The
    registry now decides the order, a separate test asserts uniqueness and that order, and what
    is pinned here is that the schema can only *grow*: the four that exist still appear, in
    order, and a new one is additive.
    """
    results = evaluated["summary"]["results"]

    assert set(results) >= {"readouts", "latent_health", "lag", "per_recording", "verdicts"}
    names = [verdict["name"] for verdict in results["verdicts"]]
    assert set(names) >= {
        "predictive_improvement",
        "source_specificity",
        "prior_carries_target_state",
        "latent_not_collapsed",
    }
    for verdict in results["verdicts"]:
        assert verdict["status"] in {"PASS", "FAIL", "INCONCLUSIVE"}


def test_the_verdicts_are_unique_and_in_registry_order(evaluated):
    """The list is read by name *and* by position -- by the acceptance gate and by the arm
    tables -- so a duplicate or a reordering is a silent change of meaning."""
    names = [verdict["name"] for verdict in evaluated["summary"]["results"]["verdicts"]]

    assert names == list(metrics.VERDICT_ORDER)
    assert len(names) == len(set(names))


def test_an_additional_verdict_is_one_line_and_breaks_nothing(monkeypatch):
    """What the registry is for. Adding a criterion must not mean editing the ordering code, the
    promotion list and three tests.

    The already-registered names are read off the registry rather than listed, so this test does
    not itself become the fourth place a criterion has to be written down -- which is exactly what
    it exists to prevent.
    """
    registered = list(metrics.VERDICT_ORDER)
    monkeypatch.setattr(
        metrics, "VERDICT_REGISTRY", metrics.VERDICT_REGISTRY + (("synthetic_check", True),)
    )
    synthetic = metrics.Verdict("synthetic_check", metrics.PASS, "criterion", "detail", {})
    existing = [
        metrics.Verdict(name, metrics.INCONCLUSIVE, "c", "d", {}) for name in registered
    ]

    ordered = metrics.order_verdicts([synthetic] + existing)

    assert [verdict.name for verdict in ordered][-1] == "synthetic_check"
    assert [verdict.name for verdict in ordered][:-1] == registered


def test_an_unregistered_verdict_is_refused_rather_than_reported(evaluated):
    """A verdict absent from the registry reaches neither the reporting order nor the headline,
    so producing one silently would be producing a criterion nobody reads."""
    with pytest.raises(ValueError, match="VERDICT_REGISTRY"):
        metrics.order_verdicts(
            [metrics.Verdict("invented", metrics.PASS, "c", "d", {})]
        )


def test_a_registered_verdict_the_run_did_not_produce_is_refused():
    """The other side of the same guard: a criterion that cannot be evaluated is reported
    INCONCLUSIVE, never omitted -- a gap in the list reads as a criterion that passed."""
    with pytest.raises(ValueError, match="did not produce"):
        metrics.order_verdicts(
            [metrics.Verdict("predictive_improvement", metrics.PASS, "c", "d", {})]
        )


def test_the_summary_carries_the_checkpoint_and_config_it_evaluated(evaluated, trained_run):
    summary = evaluated["summary"]

    assert Path(summary["checkpoint"]) == trained_run
    assert Path(summary["config"]).name == RESOLVED_CONFIG_FILENAME
    assert summary["device"] == "cpu"


def test_the_summary_is_json_a_non_python_reader_can_parse(evaluated):
    """``json.dump`` emits the bare tokens ``NaN`` and ``Infinity`` for non-finite floats, which
    round-trip through Python and are rejected by every other parser."""
    for token in ("NaN", "Infinity", "-Infinity"):
        assert token not in evaluated["text"]


def test_the_summary_holds_no_tensors_or_numpy_scalars(evaluated):
    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        else:
            assert isinstance(value, (str, int, float, bool, type(None))), type(value)

    walk(evaluated["summary"])


def test_recording_identifiers_reach_the_output_as_real_guids(evaluated):
    """The one thing a stub batch cannot check: ``guid`` survives collation as a ``list[str]``,
    is never moved to a device, and must arrive as the shard's own identifiers rather than as the
    ``'unknown'`` fallback."""
    per_recording = evaluated["summary"]["results"]["per_recording"]

    assert per_recording, "no recordings were aggregated"
    assert "unknown" not in per_recording


def test_the_readouts_are_finite_and_the_gap_is_the_difference(evaluated):
    readouts = evaluated["summary"]["results"]["readouts"]

    for name, value in readouts.items():
        assert value is not None and math.isfinite(value), f"{name} is {value}"
    assert readouts["pred_gap"] == pytest.approx(
        readouts["nll_base_block"] - readouts["nll_full_block"], rel=1e-5
    )


def test_the_dumped_config_round_trips_through_the_config_reader(evaluated):
    """The merged result is the run's durable record, so it has to be readable back as one
    document -- and carry no ``base:``, which in a dumped config would point at whatever a
    committed file says today rather than at what this run used."""
    written = evaluated["results_dir"] / RESOLVED_CONFIG_FILENAME

    reloaded = load_config(str(written))

    assert "base" not in reloaded
    assert reloaded == yaml.safe_load(written.read_text(encoding="utf-8"))


def test_the_summary_records_both_values_of_every_override(evaluated):
    """An evaluation runs on a configuration that is deliberately *not* the one the checkpoint
    trained under. A divergence recorded nowhere is indistinguishable from an accident, so both
    values travel into the summary."""
    entries = {
        record["path"]: record
        for record in evaluated["summary"]["config_overrides"]["entries"]
    }

    fields = entries["dataset_config.dataloader_config.dataset_kwargs.load_fields"]
    assert "target" not in fields["run_value"], "the training contract carried no clinical fields"
    for name in ("target", "epoch", "cs_label", "bg_label", "time_from_labor_onset"):
        assert name in fields["eval_value"]
    assert entries["dataset_config.vae_test_datasets"]["run_value"] != (
        entries["dataset_config.vae_test_datasets"]["eval_value"]
    )


def test_the_run_carries_a_log_beside_its_artifacts(evaluated):
    assert (evaluated["results_dir"] / run_module.LOG_FILENAME).is_file()


def test_the_run_used_a_single_process_loader(evaluated):
    """Spawn workers over a multi-file HDF5 dataset silently truncate every pass after the
    first, and an evaluation makes many passes."""
    written = yaml.safe_load(
        (evaluated["results_dir"] / RESOLVED_CONFIG_FILENAME).read_text(encoding="utf-8")
    )
    loader_config = written["dataset_config"]["dataloader_config"]

    assert loader_config["num_workers"] == 0
    assert loader_config["persistent_workers"] is False


def test_a_guarded_run_reports_its_input_delay(evaluated):
    """The shipped config resolves a 120 s budget, whose worst source delay is 30 steps, and the
    summary must carry that number: every lag the report quotes is offset by it, so a summary
    claiming 0 would understate the physiological delay by two minutes with nothing failing.

    Both places are asserted because they are written by different code paths and only their
    agreement makes the lag axis trustworthy."""
    summary = evaluated["summary"]

    assert summary["source_delay_steps"] == 30
    assert summary["results"]["lag"]["delay_steps"] == 30


def test_the_lag_report_adds_back_the_causal_input_delay():
    r"""A checkpoint trained under a reach budget has a stale source memory, so a peak at lag
    $\ell$ refers to content $\ell + \delta$ steps back. Reporting it with $\delta = 0$
    understates the physiological delay by up to two minutes at the $120$ s budget, with nothing
    failing -- so the delay is read off the *model*, which is what was trained.
    """
    from teb_vae.lag_attn.channel_reach import resolve_stream_budgets

    budget = resolve_stream_budgets(
        {"causal_reach_budget_s": 120.0, "use_up_st": True, "warmup_period": 30,
         "c_y": 109, "c_u": 58}
    )
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(
        **dict(
            TINY_KWARGS,
            sequence_length=64,
            warmup_period=30,
            target_keep_index=budget.target_keep_index,
            target_delays=budget.target_delays,
            source_keep_index=budget.source_keep_index,
            source_delays=budget.source_delays,
        )
    )

    assert model.source_delay_steps == budget.max_delay == 30


# =============================================================================
# Finding the run's own configuration
# =============================================================================
def test_the_config_is_found_beside_the_checkpoint(trained_run):
    found = run_module.resolved_config_for(trained_run)

    assert found == trained_run.parent / RESOLVED_CONFIG_FILENAME


def test_a_checkpoint_without_its_config_names_every_path_tried(tmp_path):
    """A checkpoint copied out of its run directory has lost the record of what it trained on,
    and evaluating it against a guessed configuration is worse than not evaluating it."""
    orphan = tmp_path / "model_checkpoints" / "lonely.ckpt"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"")

    with pytest.raises(FileNotFoundError) as excinfo:
        run_module.resolved_config_for(orphan)

    message = str(excinfo.value)
    assert RESOLVED_CONFIG_FILENAME in message
    assert str(orphan.parent) in message


def test_the_output_directory_is_timestamped_with_a_collision_guard(tmp_path):
    config = {"general_config": {"tag": "rws", "folders_config": {"out_dir_base": str(tmp_path)}}}

    first = run_module.make_output_dir(config)
    second = run_module.make_output_dir(config)

    assert first != second
    assert first.name == second.name == run_module.RESULTS_DIRNAME


def test_an_explicit_output_directory_is_used_as_given(tmp_path):
    result = run_module.make_output_dir({}, tmp_path / "here")

    assert result == tmp_path / "here" / run_module.RESULTS_DIRNAME
    assert result.is_dir()


def test_a_prior_summary_is_preserved_byte_identical_before_it_is_overwritten(tmp_path):
    """Re-running into a finished directory is the documented offline path, and it is also
    destructive: the summary is opened with mode ``'w'`` and the artifact manifest classifies
    every earlier file as stale, so a one-analysis re-run replaced a complete summary with a
    mostly-null one and exited $0$. The sanity block and the promoted verdicts exist nowhere
    else."""
    results_dir = tmp_path / "run" / run_module.RESULTS_DIRNAME
    results_dir.mkdir(parents=True)
    original = b'{"results": {"pred_gap": 1.0}}'
    (results_dir / run_module.SUMMARY_FILENAME).write_bytes(original)

    run_module.make_output_dir({}, tmp_path / "run")

    backups = sorted(results_dir.glob("summary.bak.*.json"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    assert not (results_dir / run_module.SUMMARY_FILENAME).exists(), (
        "a stale summary left in place would read as this pass's result if this pass then failed"
    )


def test_the_preflight_record_is_left_where_a_later_pass_can_read_it(tmp_path):
    """Deliberately *not* preserved aside: a pass with no checkpoint cannot regenerate the
    causality disclosure, and renaming the record would take it from exactly the pass that needs
    to read it back."""
    from teb_vae.lag_attn_rws.eval import preflight

    results_dir = tmp_path / "run" / run_module.RESULTS_DIRNAME
    results_dir.mkdir(parents=True)
    record = results_dir / preflight.PREFLIGHT_FILENAME
    record.write_bytes(b'{"causality": {}}')

    run_module.make_output_dir({}, tmp_path / "run")

    assert record.is_file()


# =============================================================================
# Analysis selection
# =============================================================================
def test_only_returns_registry_order_regardless_of_the_order_typed():
    """The run order is the pipeline's: a later analysis may read what an earlier one wrote."""
    assert run_module.select_analyses(_REGISTRY, "attention,forecast", None) == [
        "forecast", "attention"
    ]


def test_only_and_skip_compose():
    assert run_module.select_analyses(_REGISTRY, "forecast,coupling", "coupling") == ["forecast"]


def test_neither_flag_selects_everything():
    assert run_module.select_analyses(_REGISTRY, None, None) == list(_REGISTRY)


@pytest.mark.parametrize("flag", ["only", "skip"])
def test_an_unknown_name_raises_naming_the_valid_set(flag):
    """A misspelling would otherwise silently run everything (``--only``) or nothing extra
    (``--skip``), which is indistinguishable in the output from having asked for exactly that."""
    selection = {"only": None, "skip": None, flag: "forcast"}
    with pytest.raises(ValueError, match="forecast"):
        run_module.select_analyses(_REGISTRY, **selection)


def test_an_unskippable_step_is_refused_by_name_rather_than_as_a_typo():
    with pytest.raises(ValueError, match="not selectable"):
        run_module.select_analyses(_REGISTRY, "band_partition", None)


def test_there_is_no_dependency_table():
    """The real dependency is on files existing on disk rather than on an analysis having run in
    this pass, which is what makes an offline ``--only`` work at all. One line adds the table the
    day a genuine correctness dependency appears."""
    assert not hasattr(run_module, "ANALYSIS_DEPENDENCIES")


# =============================================================================
# Failure isolation
# =============================================================================
def _writing_analysis(filename: str):
    """An analysis that writes one CSV and returns the protocol's keys."""

    def _run(context, *, eval_config, output_dir, probe):
        (Path(output_dir) / filename).write_text("value\n1\n", encoding="utf-8")
        return {"n_samples": 1, "composition": {}, "plan": {"capped": False}}

    return _run


def _raising_analysis(context, *, eval_config, output_dir, probe):
    raise KeyError("mu_full")


def test_one_failing_analysis_costs_only_itself(tmp_path):
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
    assert [record["status"] for record in records] == [
        "ok", "ok", "failed", "ok", "ok"
    ]
    for name in ("first.csv", "second.csv", "fourth.csv", "fifth.csv"):
        assert (tmp_path / name).is_file()
    assert "KeyError" in records[2]["error"]
    # The frame name, which only a formatted traceback carries -- str(exc) is just "'mu_full'".
    assert "_raising_analysis" in records[2]["traceback"]
    assert report.exit_code() == 1
    assert set(report.results) == {"first", "second", "fourth", "fifth"}


def test_the_step_heartbeat_is_rewritten_as_each_analysis_finishes(tmp_path):
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


# =============================================================================
# The command line
# =============================================================================
def test_a_checkpoint_is_not_required_at_parse_time(tmp_path):
    """Not every readout needs the model -- one computed from a finished run's own tables does
    not -- so the parser does not refuse on behalf of a caller that would not have needed one."""
    parsed = run_module.build_parser().parse_args(["--output-dir", str(tmp_path)])

    assert parsed.checkpoint is None
    assert parsed.num_samples is None, "the draw count comes from eval_config unless overridden"


def test_the_run_still_refuses_to_start_without_one(tmp_path):
    """A checkpoint is optional only where a finished run's tables stand in for it; an empty
    directory is neither, and a run that produced nothing would be worse than one that said why."""
    with pytest.raises(SystemExit, match="--checkpoint is required"):
        run_module._cli(["--output-dir", str(tmp_path)])


def test_each_argument_records_where_its_value_came_from():
    """A run's provenance must be unambiguous after the fact rather than reconstructed from a
    shell history -- and the launch dict is resolved per key, so the two sources genuinely mix."""
    values, sources = run_module.resolve_arguments(
        ["--checkpoint", "a.ckpt"], run_args={"device": "cpu"}
    )

    assert (values["checkpoint"], sources["checkpoint"]) == ("a.ckpt", "cli")
    assert (values["device"], sources["device"]) == ("cpu", "config")
    assert (values["only"], sources["only"]) == (None, "default")


def test_a_launch_dict_key_that_is_not_an_argument_raises():
    """A typo there would otherwise silently do nothing, which is the same class of failure the
    ``eval_config`` validator guards against."""
    with pytest.raises(ValueError, match="max_sample"):
        run_module.resolve_arguments([], run_args={"max_sample": 4})


def test_the_shipped_launch_dict_resolves(tmp_path):
    """It ships in this file and is never exercised by a normal test run, so a key renamed on the
    parser would be found by an operator pressing Run rather than by the suite."""
    values, _ = run_module.resolve_arguments([])

    assert set(values) == set(run_module.RUN_ARGS)


# =============================================================================
# Observability
# =============================================================================
def test_peak_memory_is_absent_rather_than_zero_on_cpu(evaluated):
    """A $0.00$ GB peak reads as "measured, and the run used no memory", which is a claim a CPU
    box cannot make."""
    assert "max_memory_allocated_gb" not in evaluated["summary"]["results"]


def test_every_step_carries_its_own_elapsed_time(evaluated):
    steps = evaluated["summary"]["steps"]

    assert steps, "the run recorded no steps at all"
    for record in steps:
        assert record["status"] == "ok"
        assert isinstance(record["elapsed_s"], (int, float))


def test_the_summary_carries_the_exit_code_and_the_failure_list(evaluated):
    summary = evaluated["summary"]

    assert summary["exit_code"] == evaluated["exit_code"] == 0
    assert summary["failed"] == [] and summary["n_failed"] == 0
    assert summary["n_steps"] == len(summary["steps"])


def test_the_run_context_records_what_the_arm_tables_consume(evaluated, trained_run):
    """The parameter count, the checkpoint's training epoch, the class of model that produced
    the run, the anchor-coverage distribution the ``coverage_floor`` is confirmed against, and
    the observed objective magnitude the spike breaker's ``additive_margin`` is re-derived from
    -- each checked against an independent reading rather than merely present."""
    import torch

    from teb_vae.lag_attn_rws.eval import run as run_module

    context = evaluated["summary"]["run_context"]

    task = run_module.load_task(trained_run, torch.device("cpu"))
    assert context["n_parameters"] == sum(p.numel() for p in task.orig_model.parameters())
    blob = torch.load(trained_run, map_location="cpu", weights_only=False)
    assert context["train_epoch"] == blob["epoch"]
    # Copied out of the checkpoint's own stamp, which is the only place it is written: the dumped
    # config carries every constructor keyword and not the class they build. A table that ranks
    # architectures against each other reads this key, so the summary has to carry it.
    assert context["model_class"] == blob["model_class"]
    assert context["model_class"] == run_module.RWS_BINDING.model_cls.__name__

    coverage = context["anchor_coverage_frac"]
    assert coverage["n_anchors"] > 0
    # The per-anchor table holds contributing anchors only, so the floor bounds the minimum --
    # and the note must say so, because that truncation is what a reader revising the floor
    # against this distribution has to know. To float32, the dtype the coverage travels in.
    assert coverage["min"] >= float(task.orig_model.coverage_floor) - 1e-6
    assert coverage["min"] <= coverage["median"] <= coverage["max"] <= 1.0
    assert "coverage_floor" in coverage["note"]

    scale = context["observed_loss_scale"]
    readouts = evaluated["summary"]["results"]["readouts"]
    assert scale["nll_full_block"] == pytest.approx(readouts["nll_full_block"])
    # Four terms, not the original three: the objective gained the prior scale anchor, so an
    # estimate recombined without its weighted rate would under-report an anchored run's
    # main_loss. The rate readout is absent until the collection emits it per sample, which is
    # exactly the case the recombination must survive -- contributing nothing, exact at
    # beta_prior 0.0.
    assert "beta_prior" in scale
    assert scale["main_loss_estimate"] == pytest.approx(
        scale["lambda_full"] * scale["nll_full_block"]
        + scale["lambda_base"] * scale["nll_base_block"]
        + scale["beta_end"] * scale["source_conditioned_kl_raw"]
        + scale["beta_prior"] * (scale["prior_rate"] or 0.0)
    )


# =============================================================================
# Checkpoint loading
# =============================================================================
def _mutated(trained_run: Path, tmp_path: Path, mutate) -> Path:
    """Save a copy of the checkpoint with one key changed."""
    blob = torch.load(trained_run, map_location="cpu", weights_only=False)
    mutate(blob)
    path = tmp_path / "mutated.ckpt"
    torch.save(blob, path)
    return path


def test_a_checkpoint_from_another_model_is_refused(trained_run, tmp_path):
    def _rename(blob):
        blob["model_class"] = "SomeOtherModel"

    with pytest.raises(ValueError, match="model_class"):
        run_module.load_task(_mutated(trained_run, tmp_path, _rename), torch.device("cpu"))


def test_a_checkpoint_without_model_kwargs_is_refused(trained_run, tmp_path):
    """``SeqVaeLagAttnRws()`` with no arguments builds the *production* geometry rather than
    raising, so guessing would silently evaluate a different model."""

    def _drop(blob):
        blob["model_kwargs"] = {}

    with pytest.raises(RuntimeError, match="no 'model_kwargs'"):
        run_module.load_task(_mutated(trained_run, tmp_path, _drop), torch.device("cpu"))


def test_a_checkpoint_trained_at_a_nonzero_beta_prior_is_scored_at_that_value(
    trained_run, tmp_path
):
    """Asserted rather than assumed: a reconstruction that dropped the key would silently score
    an anchored checkpoint under the unanchored objective. The distinctive value round-trips
    from the blob's ``hyper_parameters`` into the rebuilt task's own."""

    def _anchor(blob):
        blob["hyper_parameters"]["beta_prior"] = 0.037

    task = run_module.load_task(
        _mutated(trained_run, tmp_path, _anchor), torch.device("cpu")
    )

    assert float(task.hparams["beta_prior"]) == pytest.approx(0.037)


def test_a_checkpoint_without_hyperparameters_is_refused(trained_run, tmp_path):
    """The likelihood is a checkpoint fact: scoring an ``mse`` run under a Gaussian NLL would
    report a different objective's numbers with nothing raising."""

    def _drop(blob):
        del blob["hyper_parameters"]

    with pytest.raises(RuntimeError, match="no 'hyper_parameters'"):
        run_module.load_task(_mutated(trained_run, tmp_path, _drop), torch.device("cpu"))


def test_a_state_dict_that_does_not_align_is_refused(trained_run, tmp_path):
    """``load_checkpoint_strict`` returns ``None`` rather than raising, so an unchecked call
    evaluates randomly initialised weights and reports the result as a measurement."""

    def _widen(blob):
        blob["model_kwargs"] = dict(blob["model_kwargs"], d_model=64, d_head=16)

    with pytest.raises(RuntimeError, match="could not align checkpoint"):
        run_module.load_task(_mutated(trained_run, tmp_path, _widen), torch.device("cpu"))


def test_a_missing_checkpoint_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_module.load_task(tmp_path / "absent.ckpt", torch.device("cpu"))


def test_the_loaded_weights_are_the_checkpoints_own(trained_run):
    """Every parameter, not merely a shape-compatible model."""
    task = run_module.load_task(trained_run, torch.device("cpu"))
    blob = torch.load(trained_run, map_location="cpu", weights_only=False)
    saved = {
        key[len("_orig_model.") :]: value
        for key, value in blob["state_dict"].items()
        if key.startswith("_orig_model.")
    }

    assert saved, "the checkpoint's state dict is not wrapper-prefixed as expected"
    for name, parameter in task.orig_model.state_dict().items():
        assert torch.equal(parameter, saved[name]), f"{name} did not load"


def test_the_loaded_task_is_in_evaluation_mode(trained_run):
    """Dropout live during evaluation would leave the attention rows not summing to one, so the
    lag attribution would not be a decomposition of anything."""
    assert run_module.load_task(trained_run, torch.device("cpu")).training is False


# =============================================================================
# JSON safety
# =============================================================================
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_become_null(value):
    assert run_module.json_safe(value) is None


def test_numpy_and_torch_values_become_plain_python():
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


def test_an_unexpected_type_is_recorded_rather_than_dropped():
    """A stray object lands as its repr instead of killing the write at the end of a long run."""
    assert run_module.json_safe(torch.device("cpu")) == "cpu"


# =============================================================================
# The cohort block: who was evaluated, and the two statements about them
#
# Both statements are **computed** rather than written down. A constant saying the cohorts are
# disjoint outlives the configuration that made it true, and a summary asserting a leakage-free
# evaluation of a run that evaluated its own training set is worse than one asserting nothing.
# =============================================================================
def test_the_cohort_block_counts_segments_and_recordings_on_both_axes(evaluated):
    """Both levels, because they answer different questions and routinely disagree: a subgroup
    with many segments and three recordings is one whose statistics have $n = 3$."""
    from .conftest import MULTI_CLASS_SUBGROUPS

    block = evaluated["summary"]["results"]["cohort"]
    results = evaluated["summary"]["results"]

    assert block["n_segments"] == results["n_samples"]
    assert block["n_recordings"] == results["n_recordings"]
    assert set(block["by_subgroup"]["segments"]) == set(MULTI_CLASS_SUBGROUPS)
    assert sum(block["by_subgroup"]["segments"].values()) == results["n_samples"]
    assert sum(block["by_subgroup"]["recordings"].values()) == results["n_recordings"]
    assert set(block["by_clinical_class"]["segments"]) == {"healthy", "acidosis", "hie"}
    # The two levels differ on this fixture, which is what makes the pair worth reporting.
    assert block["by_subgroup"]["segments"] != block["by_subgroup"]["recordings"]


def test_a_disjoint_pair_reports_the_out_of_distribution_statement(evaluated):
    """The fixture evaluates the generated holdout shards against a config whose training list is
    the shipped one, so the two are disjoint and the statement follows."""
    block = evaluated["summary"]["results"]["cohort"]

    assert block["training_cohort_disjoint"] is True
    assert block["training_cohort_overlap"] == []
    assert "out-of-distribution" in block["out_of_distribution"]
    assert block["vae_test_datasets"]


def test_an_overlapping_pair_yields_false_and_suppresses_the_statement(tmp_path):
    """The case a string-presence test cannot catch: a constant would still be there. Evaluating
    the training set is not a leakage-free evaluation, and the summary must not say it is."""
    from teb_vae.lag_attn_rws.eval import cohort

    shard = str(tmp_path / "healthy_bg_cs.hdf5")
    config = {"dataset_config": {"vae_train_datasets": [shard], "vae_test_datasets": [shard]}}

    block = cohort.build_cohort_block(pd.DataFrame(), config, None)

    assert block["training_cohort_disjoint"] is False
    assert block["training_cohort_overlap"] == [
        os.path.normcase(os.path.abspath(shard))
    ]
    assert "out_of_distribution" not in block


def test_a_run_with_no_training_list_claims_nothing(tmp_path):
    """``None`` rather than ``False``: a run whose config named no training set cannot claim
    disjointness, and ``False`` there would read as "they overlap"."""
    from teb_vae.lag_attn_rws.eval import cohort

    block = cohort.build_cohort_block(
        pd.DataFrame(), {"dataset_config": {"vae_test_datasets": ["a.hdf5"]}}, None
    )

    assert block["training_cohort_disjoint"] is None
    assert "out_of_distribution" not in block


def test_the_unseen_subgroups_include_the_two_healthy_no_background_ones(evaluated):
    """The scope is wider than "acidosis and HIE": the pretraining split is built from the healthy
    *with-background* recordings only, so both no-background healthy subgroups are unseen too."""
    from teb_vae.lag_attn_rws.eval._reuse import labels

    block = evaluated["summary"]["results"]["cohort"]

    assert set(block["unseen_subgroups"]) >= {"healthy_no_bg_cs", "healthy_no_bg_no_cs"}
    assert set(block["pretraining_subgroups"]) == {"healthy_bg_cs", "healthy_bg_no_cs"}
    assert set(block["unseen_subgroups"]) | set(block["pretraining_subgroups"]) == set(
        labels.CANONICAL_SUBGROUPS
    )
    # And which of them this run actually scored, which is what the statement applies to here.
    assert "healthy_no_bg_no_cs" in block["unseen_subgroups_evaluated"]


def test_the_non_comparability_sentence_is_in_every_summary(evaluated):
    """An eval score and a ``test_*`` metric logged during training are computed over different
    populations, and nothing in either number says so."""
    block = evaluated["summary"]["results"]["cohort"]

    assert "not comparable" in block["non_comparability"]
    assert "different populations" in block["non_comparability"]


def test_the_labour_onset_rows_are_counted_rather_than_dropped(evaluated):
    """NaN means the recording is absent from the labour-onset table. It is the denominator of
    every labour-onset statement, and a summary that quietly dropped those rows would report a
    mean over a population it does not name."""
    block = evaluated["summary"]["results"]["cohort"]["time_from_labor_onset"]

    assert block["present"] is True
    assert block["n_rows"] == evaluated["summary"]["results"]["n_samples"]
    # The fixture writes NaN for part of one shard on purpose, so the count is non-vacuous.
    assert block["n_nan"] > 0
    assert block["n_finite"] + block["n_nan"] == block["n_rows"]
    assert 0.0 < block["nan_fraction"] < 1.0
    assert block["min_hours"] <= block["mean_hours"] <= block["max_hours"]


# =============================================================================
# Reusing a directory collected under different acceptance criteria
# =============================================================================
def _collection_record(verdict_names) -> dict:
    """The half of a collection record the currency check reads."""
    return {"results": {"verdicts": [{"name": name, "status": "PASS"} for name in verdict_names]}}


def test_a_directory_collected_under_the_current_criteria_is_reused(tmp_path):
    """The ordinary offline path: the tables were written by this pipeline, so nothing objects."""
    metrics.check_cached_verdicts(
        _collection_record([name for name, _ in metrics.VERDICT_REGISTRY])["results"]["verdicts"]
    )


def test_a_directory_collected_before_a_criterion_existed_is_refused_by_name(tmp_path):
    """The failure this check exists for, and the reason it cannot be left to ``order_verdicts``.

    That guard runs over the list a *fresh* pass builds and is never reached on the reuse path, so
    a directory collected under the earlier criteria would be re-reported verbatim: a summary
    silently missing a criterion, which reads exactly like one where the criterion passed.
    """
    stale = [name for name, _ in metrics.VERDICT_REGISTRY if name != "source_margin_positive"]

    with pytest.raises(metrics.StaleCachedVerdicts) as raised:
        metrics.check_cached_verdicts(_collection_record(stale)["results"]["verdicts"])

    message = str(raised.value)
    # What moved, in both directions, and the one way out -- an operator reading this must not
    # have to work out that re-collecting is the fix.
    assert "source_margin_positive" in message
    assert "--checkpoint" in message and "Re-collect" in message


def test_a_verdict_the_registry_no_longer_declares_is_refused_too(tmp_path):
    """The other direction. A renamed criterion leaves a record carrying a name nothing decides,
    and reporting it would put a status against a criterion this pipeline no longer has."""
    extra = [name for name, _ in metrics.VERDICT_REGISTRY] + ["a_criterion_that_was_removed"]

    with pytest.raises(metrics.StaleCachedVerdicts) as raised:
        metrics.check_cached_verdicts(_collection_record(extra)["results"]["verdicts"])

    assert "a_criterion_that_was_removed" in str(raised.value)


def test_a_record_written_before_verdicts_existed_at_all_is_not_refused(tmp_path):
    """``None`` is older than the contract this check enforces, and the analyses re-run against
    such a directory still produce their own numbers. Refusing it would be refusing a directory
    the check has nothing to say about."""
    metrics.check_cached_verdicts(None)


@pytest.mark.slow
def test_the_reuse_path_applies_the_check(trained_run, repointed_overrides, tmp_path):
    """Wired, not merely defined: the check is on the reuse branch of ``load_or_collect``, so a
    finished directory whose collection predates a criterion refuses on the *second* pass rather
    than reporting a short verdict list."""
    from teb_vae.lag_attn_rws.eval import collect as collect_module

    overrides = repointed_overrides
    output_dir = tmp_path / "run"
    assert run_module.main(
        trained_run, output_dir, overrides=overrides, device="cpu", num_samples=1,
        only="perm_control",
    ) == 0

    results_dir = Path(output_dir) / run_module.RESULTS_DIRNAME
    record_path = results_dir / collect_module.COLLECTION_FILENAME
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["results"]["verdicts"] = [
        entry for entry in record["results"]["verdicts"]
        if entry.get("name") != "source_margin_positive"
    ]
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(metrics.StaleCachedVerdicts):
        run_module.main(
            trained_run, output_dir, overrides=overrides, device="cpu", num_samples=1,
            only="perm_control",
        )
