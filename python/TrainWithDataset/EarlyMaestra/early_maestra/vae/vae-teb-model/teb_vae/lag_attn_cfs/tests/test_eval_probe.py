r"""The probe reports what the split yields and what the forward returns, and refuses five ways.

Two halves, two questions.

**The population half** is the sibling's. Nothing else in a run reports per-file coverage, so a shard
that silently contributes nothing is invisible in every other output -- that is the predecessor's
hardest bug, and this is the only artifact that can see it. The four refusals are tested one
deliberately broken input at a time, each asserting the message rather than only the raise: a guard
whose message does not name the fix costs the same debugging session the guard was added to prevent.

**The forward-contract half is this cell's own**, and it exists because the readout module has to be
written against a contract that was *measured*. Three facts about this forward have no
counterpart in the family and every one of them is a way that module could be written wrong: it takes
five positional arguments and raises without a phase above stride $1$; it returns two keys the
family's does not; and its four forecast tensors are $(B, A_{\max}, H, C_{\mathrm{keep}})$ rather
than $(B, T_{\mathrm{valid}}, H, R)$. The fifth refusal is the one that keeps the measurement
honest -- a contract measured at the training tiling would name an $A_{\max}$ no evaluation ever
produces, so anything but the dense geometry is refused rather than reported.

**What the fixtures may be asserted on.** The generated cohort shards are eight real raw segments
re-used under distinct identities, and the tiny model is untrained. Everything below is therefore
about schema, shapes, counts, identities and refusals -- never about the value of any readout.
"""
from __future__ import annotations

import copy
import json
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

from teb_vae.lag_attn_cfs.eval import launch, preflight, probe as probe_module

from .conftest import (
    TINY_STRIDE,
    TINY_WARMUP_PERIOD,
    make_stub_batch,
    make_task,
    tiny_warmup_kwargs,
)

#: The generated cohort fixture's size: eight subgroup shards, three recordings each, two segments
#: per recording. Imported from the generator rather than restated, so a fixture that grew moves
#: these with it.
from scripts.make_tiny_shard import (  # noqa: E402
    COHORT_GUIDS_PER_SHARD,
    COHORT_SEGMENTS_PER_GUID,
    COHORT_SUBGROUPS,
)

_N_SHARDS = len(COHORT_SUBGROUPS)
_N_SAMPLES = _N_SHARDS * COHORT_GUIDS_PER_SHARD * COHORT_SEGMENTS_PER_GUID


@pytest.fixture(scope="module")
def record(cohort_config, cohort_loader) -> Dict[str, Any]:
    """One population pass, shared by every read-only assertion below."""
    return probe_module.run_probe(
        cohort_loader,
        configured_files=cohort_config["dataset_config"]["vae_test_datasets"],
    )


@pytest.fixture(scope="module")
def contract() -> Dict[str, Any]:
    """One forward-contract pass over the tiny geometry, shared by the shape assertions.

    Built at the **tiling** stride, not the constructor's inert $1$: the seam under test is the one
    that resolves the dense geometry outside a training step, and a task built at stride $1$ would
    resolve the same anchor set either way and make every assertion here hold vacuously.
    """
    return probe_module.forward_contract(make_task(), make_stub_batch())


def _loader_for(config: Dict[str, Any]) -> Any:
    from train.data_module import GraphDataModule

    return GraphDataModule(config).test_dataloader()


# =================================================================================================
# What the population pass records
# =================================================================================================
def test_the_probe_counts_every_sample_and_every_shard(record) -> None:
    assert record["n_samples"] == _N_SAMPLES
    assert record["n_unique_guids"] == _N_SHARDS * COHORT_GUIDS_PER_SHARD
    assert set(record["per_file"]) == {f"{name}.hdf5" for name in COHORT_SUBGROUPS}
    assert sum(record["per_file"].values()) == _N_SAMPLES


def test_the_class_histogram_is_keyed_by_clinical_name_not_by_a_stored_value(record) -> None:
    """Keyed on the raw stored value it produced entries like ``'0.75'`` for a recording whose valid
    steps were partial -- enough to make a single-class split report several."""
    assert set(record["per_target_class"]) == {"healthy", "acidosis", "hie"}
    assert sum(record["per_target_class"].values()) == _N_SAMPLES


def test_both_label_axes_and_the_source_vectors_are_recorded(record) -> None:
    assert sum(record["per_cs_label"].values()) == _N_SAMPLES
    assert sum(record["per_bg_label"].values()) == _N_SAMPLES
    assert len(record["guids"]) == _N_SAMPLES
    assert len(record["source_files"]) == _N_SAMPLES


def test_the_probe_answers_whether_validity_is_ever_fractional(record) -> None:
    """The question the mask arithmetic downstream depends on, and it matters more here than in the
    raw cells: these coefficients carry no gap sentinel of their own, so ``weight`` is the only
    trustworthy validity signal and it gates every mask, every baseline and every event."""
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
    assert epoch["max_hours"] > epoch["min_hours"]


def test_the_absent_labour_onset_times_are_counted_rather_than_dropped(record) -> None:
    onset = record["time_from_labor_onset"]
    assert onset["n_values"] == _N_SAMPLES
    assert 0 < onset["n_nan"] < _N_SAMPLES


def test_the_probe_makes_exactly_one_pass(cohort_loader) -> None:
    """Pure bookkeeping over the loader: it does no forward of its own."""
    passes = {"count": 0}

    class CountingLoader:
        def __iter__(self):
            passes["count"] += 1
            return iter(cohort_loader)

    probe_module.run_probe(CountingLoader())
    assert passes["count"] == 1


# =================================================================================================
# What the population pass writes and prints
# =================================================================================================
def test_the_written_json_omits_the_per_sample_vectors(cohort_loader, tmp_path) -> None:
    """One row per sample does not belong in a file meant to be read at a glance."""
    probe_module.run_probe(cohort_loader, output_dir=tmp_path)
    written = json.loads((tmp_path / probe_module.PROBE_FILENAME).read_text(encoding="utf-8"))

    assert written["n_samples"] == _N_SAMPLES
    for key in probe_module.IN_MEMORY_KEYS:
        assert key not in written


def test_the_cohort_table_shows_every_count_beside_its_share(record) -> None:
    table = probe_module.format_cohort_table(record)

    for name in COHORT_SUBGROUPS:
        assert f"{name}.hdf5" in table
    for name in ("healthy", "acidosis", "hie"):
        assert name in table
    assert f"samples          {_N_SAMPLES}" in table
    assert "%" in table, "a bare count without its share hides the coverage"


def test_the_cohort_table_survives_a_record_with_nothing_in_it() -> None:
    """It is printed after a pass that may have found very little; it must not raise there."""
    assert probe_module.format_cohort_table({"n_samples": 0}).startswith("cohort")


# =================================================================================================
# The four population refusals
# =================================================================================================
def test_an_empty_split_raises() -> None:
    with pytest.raises(RuntimeError, match="no samples at all"):
        probe_module.run_probe([])


def test_a_configured_shard_yielding_nothing_raises(cohort_config, cohort_loader) -> None:
    """A departure from the predecessor, which only logged it and was ignored."""
    configured = list(cohort_config["dataset_config"]["vae_test_datasets"])

    with pytest.raises(RuntimeError, match="yielded zero samples") as excinfo:
        probe_module.run_probe(
            cohort_loader, configured_files=configured + ["/data/k_fold/test/absent_shard.hdf5"]
        )

    assert "absent_shard.hdf5" in str(excinfo.value)


def test_a_batch_capped_pass_warns_instead_of_raising(cohort_config, cohort_loader) -> None:
    """A batch cap reads a prefix of the concatenated index, so a missing shard is expected."""
    configured = list(cohort_config["dataset_config"]["vae_test_datasets"])

    capped = probe_module.run_probe(cohort_loader, configured_files=configured, max_batches=1)

    assert capped["n_batches"] == 1
    assert capped["n_samples"] < _N_SAMPLES
    assert len(capped["per_file"]) < _N_SHARDS


def test_a_missing_required_field_raises_naming_it(cohort_config) -> None:
    """The loader skips a field a shard does not carry, silently, so a dropped ``target`` would
    present as "no classes found" rather than as a data problem."""
    config = copy.deepcopy(cohort_config)
    kwargs = config["dataset_config"]["dataloader_config"]["dataset_kwargs"]
    kwargs["load_fields"] = [name for name in kwargs["load_fields"] if name != "target"]

    with pytest.raises(RuntimeError, match="missing required field") as excinfo:
        probe_module.run_probe(_loader_for(config))

    message = str(excinfo.value)
    assert "'target'" in message
    assert "load_fields" in message, "the message must name where the fix goes"


def test_a_guid_in_two_shards_raises() -> None:
    """The holdout split is one pool with no fold loop, so a duplicated recording is counted twice
    and lands on both sides of every between-subgroup comparison.

    Constructed from two stub batches rather than by rewriting a shard: the condition is a property
    of the *identifiers*, and the generator refuses to produce it, so hoping a fixture supplies it
    would leave the refusal untested.
    """
    batches = [
        types.SimpleNamespace(
            guid=["HEALTHY_NO_BG_NO_CS_000", "HEALTHY_NO_BG_NO_CS_001"],
            source_file_basename=["healthy_no_bg_no_cs.hdf5"] * 2,
        ),
        types.SimpleNamespace(
            guid=["HEALTHY_NO_BG_NO_CS_000", "HIE_CS_000"],
            source_file_basename=["hie_cs.hdf5"] * 2,
        ),
    ]

    with pytest.raises(RuntimeError, match="more than one shard") as excinfo:
        probe_module.run_probe(batches, required_fields=())

    assert "HEALTHY_NO_BG_NO_CS_000" in str(excinfo.value)


# =================================================================================================
# The forward contract
# =================================================================================================
def test_the_forward_returns_every_key_the_readouts_are_written_against(contract) -> None:
    """Twenty-two: the family's twenty, plus this cell's ``anchor_index`` and ``anchor_valid``. The
    count is asserted so a key silently dropped from the forward is a failure here rather than a
    ``KeyError`` deep inside a collection pass."""
    assert contract["n_output_keys"] == 22
    assert {"anchor_index", "anchor_valid"} <= set(contract["outputs"])
    assert "mu_base" in contract["outputs"] and "logvar_full" in contract["outputs"]


def test_the_forecast_tensors_are_on_the_anchor_axis_not_the_step_axis(contract) -> None:
    r"""$(B, A_{\max}, H, C_{\mathrm{keep}})$, which is the whole of what the readout module has to
    be rewritten for: the family's is $(B, T_{\mathrm{valid}}, H, R)$ over a contiguous prefix."""
    batch = contract["batch_size"]
    a_max = contract["anchor_index"]["a_max"]
    block = contract["block"]

    expected = [batch, a_max, block["horizon"], block["decoder_out_channels"]]
    for name in ("mu_base", "logvar_base", "mu_full", "logvar_full"):
        assert contract["outputs"][name]["shape"] == expected, name
    assert block["block_width"] == block["horizon"] * block["decoder_out_channels"]


def test_the_anchor_set_is_reported_with_its_dtype_and_its_validity(contract) -> None:
    """``anchor_index`` is ``long`` because it is a gather index; ``anchor_valid`` is the companion
    that says which entries are real, and a padded slot repeats the row's last real anchor."""
    anchors = contract["anchor_index"]
    batch, a_max = contract["batch_size"], anchors["a_max"]

    assert contract["outputs"]["anchor_index"]["shape"] == [batch, a_max]
    assert contract["outputs"]["anchor_index"]["dtype"] == "int64"
    assert contract["outputs"]["anchor_valid"]["shape"] == [batch, a_max]
    assert contract["outputs"]["anchor_valid"]["dtype"] == "bool"
    # At the dense stride every anchor in [F, T_valid) is real and distinct.
    assert anchors["first"] == TINY_WARMUP_PERIOD
    assert anchors["n_valid_min"] == anchors["n_valid_max"] == a_max
    assert anchors["n_distinct_valid_first_row"] == a_max


def test_the_contract_is_measured_at_the_dense_geometry(contract) -> None:
    r"""$(\varphi, S) = (0, 1)$, which is what ``resolve_anchor_geometry`` returns on the evaluation
    stages -- and $A_{\max} = T_{\mathrm{valid}} - F$ there, against
    $\lceil (T_{\mathrm{valid}} - F)/S \rceil$ at the training tiling."""
    geometry = contract["anchor_geometry"]

    assert (contract["anchor_phase"], contract["anchor_stride"]) == (0, 1)
    assert contract["anchor_index"]["a_max"] == geometry["t_valid"] - geometry["anchor_floor"]
    # The training stride is recorded beside it, and on this task it is genuinely different -- so
    # the two are not accidentally the same number.
    assert geometry["training_stride"] == TINY_STRIDE
    assert geometry["training_anchors_per_sample_max"] < geometry["anchors_per_sample"]


def test_a_contract_at_the_training_tiling_is_refused_rather_than_reported() -> None:
    """The refusal that keeps the measurement honest. The stage travels on the task for the length of
    one step, so a probe reached mid-step would otherwise report an $A_{\\max}$ no evaluation run
    ever produces -- and the readout module written against it would score a different population."""
    task = make_task()
    task._stage = "train"

    with pytest.raises(RuntimeError, match="not the dense"):
        probe_module.forward_contract(task, make_stub_batch())


def test_the_lag_support_is_measured_rather_than_asserted(contract) -> None:
    r"""Printed as the arithmetic rather than as a verdict: $\min \mathcal A - (L-1) - F_u$. On the
    tiny geometry it is negative, which is a legitimate truncated-support run -- so this also proves
    the probe reports the number instead of refusing the geometry."""
    support = contract["lag_support"]

    assert support["lag_support_margin_steps"] == (
        support["min_decoded_anchor"] - support["max_lag"] - support["lag_floor"]
    )
    assert support["min_decoded_anchor"] == TINY_WARMUP_PERIOD
    assert support["every_lag_valid_at_every_anchor"] is (
        support["lag_support_margin_steps"] >= 0
    )
    # One implementation, read by the probe and by preflight alike.
    assert support == preflight.lag_support(make_task().orig_model)


def test_the_warm_up_budget_is_reported_beside_the_widths_it_produced(contract) -> None:
    """The kept width is what every per-channel denominator in a run is taken over, and it is not
    recoverable from the declared width alone."""
    budget = contract["warmup_budget"]
    kwargs = tiny_warmup_kwargs()

    assert budget["target_declared_width"] == kwargs["c_y"] == 102
    assert budget["target_kept_width"] == len(kwargs["target_keep_index"])
    assert budget["target_max_warmup_steps"] == max(kwargs["target_warmup_steps"])
    assert budget["source_declared_width"] == kwargs["c_u"]
    # Identically 1.0 under the constructor's own budget-and-floor pairing refusal; any other value
    # means the checkpoint was built by code predating it.
    assert budget["target_warm_frac"] == 1.0


def test_the_printed_contract_carries_the_shapes_and_the_support_arithmetic(contract) -> None:
    """It is read off a terminal while the readout module is written, so the shapes have to be in it
    rather than only in the returned dict."""
    printed = probe_module.format_forward_contract(contract)
    a_max = contract["anchor_index"]["a_max"]
    block = contract["block"]

    assert "mu_base" in printed
    assert str([contract["batch_size"], a_max, block["horizon"], block["decoder_out_channels"]]) in (
        printed
    )
    assert "anchor_index" in printed and "int64" in printed
    assert f"A_max             {a_max}" in printed
    assert str(contract["lag_support"]["lag_support_margin_steps"]) in printed


def test_the_printed_causality_block_carries_the_statement_and_the_delays(
    cohort_config, cohort_shards
) -> None:
    """The probe prints what the run will disclose, so an operator sees the caveat before spending
    hours on a collection pass rather than after."""
    torch.manual_seed(0)
    model = make_task().orig_model
    disclosure = preflight.causality_disclosure(cohort_config, model)

    printed = probe_module.format_causality(disclosure)

    assert preflight.CAUSALITY_STATEMENT in printed
    assert "group delay" in printed
    assert "fhr_st" in printed and "up_ph" in printed
    assert str(cohort_shards[0]) in printed


# =================================================================================================
# The entry point
# =================================================================================================
def test_the_parser_takes_a_checkpoint_or_a_config_and_defaults_the_rest() -> None:
    args = probe_module.build_parser().parse_args(["--config", "run/resolved_config.yaml"])

    assert args.config == "run/resolved_config.yaml"
    assert args.checkpoint is None and args.overrides is None
    assert args.output_dir is None and args.max_batches is None and args.device is None


def test_neither_input_is_required_by_argparse_and_the_entry_point_is_what_refuses() -> None:
    """``required=True`` fires before the launch dict is ever read, so it would make an IDE
    Run-button launch impossible whatever the dict said. The parser therefore accepts an empty
    command line and ``_cli`` is what refuses it -- naming both flags, because exactly one of them is
    needed and a message demanding both would send an operator to supply a config they do not have.
    """
    assert probe_module.build_parser().parse_args([]).checkpoint is None

    with pytest.raises(SystemExit) as excinfo:
        probe_module._cli([])

    message = str(excinfo.value.code)
    assert "--checkpoint" in message and "--config" in message
    assert "RUN_ARGS" in message


def test_an_input_supplied_only_by_the_launch_dict_satisfies_the_requirement() -> None:
    """The other direction, and the point of the dict: with the value filled in there is nothing left
    to refuse, so pressing Run gets a probe rather than a usage error."""
    values, sources = launch.resolve_launch_args(
        probe_module.build_parser(), {"checkpoint": "run/model_checkpoints/last.ckpt"}, []
    )

    assert values["checkpoint"] == "run/model_checkpoints/last.ckpt"
    assert sources["checkpoint"] == launch.DICT_SOURCE
    assert values["config"] is None


def test_main_refuses_when_it_is_given_neither() -> None:
    with pytest.raises(ValueError, match="either --config"):
        probe_module.main()


def test_the_module_is_runnable_on_its_own_and_needs_no_module_that_does_not_exist_yet() -> None:
    """``python -m ...eval.probe --checkpoint <ckpt>`` is how an operator reaches this. The second half
    is why
    this module is useful at all: the readout module it is used to write does not exist yet, so a
    reach for it would make the probe unrunnable exactly when it is needed."""
    source = Path(probe_module.__file__).read_text(encoding="utf-8")

    assert 'if __name__ == "__main__":' in source
    assert "eval.metrics" not in source
    assert "eval import metrics" not in source


def test_a_checkpoint_carrying_no_model_kwargs_is_refused_naming_the_consequence(tmp_path) -> None:
    """The constructor builds the production geometry from no arguments at all -- and builds it
    *ungated*, with no keep-index and no warm-up mask -- so a blob with nothing to rebuild from would
    silently probe a different model rather than raise."""
    path = tmp_path / "empty.ckpt"
    torch.save({"state_dict": {}}, path)

    with pytest.raises(RuntimeError, match="carries no 'model_kwargs'") as excinfo:
        probe_module.load_task(path, torch.device("cpu"))

    assert "UNGATED" in str(excinfo.value)


def test_a_checkpoint_that_is_not_there_is_refused_by_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        probe_module.read_checkpoint(tmp_path / "absent.ckpt")


def test_a_checkpoint_outside_its_run_directory_names_every_place_looked(tmp_path) -> None:
    """A checkpoint moved away from its ``resolved_config.yaml`` has lost the record of what it was
    trained on, and probing it against a guessed configuration is worse than not probing it."""
    orphan = tmp_path / "model_checkpoints" / "last.ckpt"
    orphan.parent.mkdir(parents=True)
    orphan.touch()

    with pytest.raises(FileNotFoundError) as excinfo:
        probe_module.resolved_config_for(orphan)

    message = str(excinfo.value)
    assert "resolved_config.yaml" in message
    assert str(orphan.parent) in message


# =================================================================================================
# Against a real checkpoint
# =================================================================================================
@pytest.mark.slow
def test_the_command_reports_both_halves_against_a_real_checkpoint(
    cohort_run, cohort_shards, cohort_stats, tmp_path
) -> None:
    """End to end: one checkpoint in, the population, the anchor set,
    the block width, the resolved budget, the lag-support measurement and the disclosure out."""
    import yaml

    from teb_vae.lag_attn_cfs.eval.config_schema import load_eval_overrides

    # The committed delta with its two placeholder leaves repointed -- which is exactly the edit an
    # operator makes to that file -- rather than a bespoke overrides file. A delta carrying only the
    # shard paths would REPLACE the committed one, and with it the clinical `load_fields` the
    # population pass requires: the loader skips a field it was not asked for, silently.
    overrides: Dict[str, Any] = load_eval_overrides()
    overrides["dataset_config"]["vae_test_datasets"] = list(cohort_shards)
    overrides["dataset_config"]["stat_path"] = cohort_stats
    overrides_path = tmp_path / "overrides.yaml"
    overrides_path.write_text(yaml.safe_dump(overrides, sort_keys=False), encoding="utf-8")
    checkpoint = sorted((cohort_run / "model_checkpoints").glob("*.ckpt"))[0]

    result = probe_module.main(
        checkpoint=checkpoint,
        overrides=overrides_path,
        output_dir=tmp_path / "results",
        device="cpu",
    )

    population, contract = result["population"], result["forward"]
    assert population["n_samples"] == _N_SAMPLES
    assert (tmp_path / "results" / probe_module.PROBE_FILENAME).is_file()

    # The shapes the readout module is written against, from a checkpoint the driver actually
    # wrote -- so the kept width is the one the budget resolved against these shards.
    batch = contract["batch_size"]
    a_max = contract["anchor_index"]["a_max"]
    block = contract["block"]
    assert contract["outputs"]["mu_base"]["shape"] == [
        batch, a_max, block["horizon"], block["decoder_out_channels"]
    ]
    assert contract["outputs"]["anchor_index"]["shape"] == [batch, a_max]
    assert contract["outputs"]["anchor_index"]["dtype"] == "int64"
    assert result["causality"]["statement"] == preflight.CAUSALITY_STATEMENT

    printed: List[str] = [
        probe_module.format_cohort_table(population),
        probe_module.format_forward_contract(contract),
        probe_module.format_causality(result["causality"]),
    ]
    assert all(text.strip() for text in printed)
