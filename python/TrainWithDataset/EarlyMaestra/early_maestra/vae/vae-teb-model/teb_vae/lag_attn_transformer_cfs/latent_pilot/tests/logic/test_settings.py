r"""The settings schema, the launch-argument resolver, and the precedence between them.

Everything here is a dictionary, a tiny YAML file in a temporary directory, and a parser. No model,
no dataset, no checkpoint and no run: what these tests establish is that a configuration is
*accepted or refused for the stated reason*, and that the value a run ends up using is the one whose
source is supposed to win.

Three properties are worth naming, because each of them fails silently rather than loudly:

* **A parser default is not a value.** The resolver has to tell an unsupplied flag from one supplied
  at its default, or every ``RUN_ARGS`` entry the parser also declares becomes unreachable -- the
  operator edits the dictionary, nothing changes, and nothing says why.
* **Nothing mutates the caller's dictionary.** ``RUN_ARGS`` is module state; a resolver that wrote
  into it would make the second call in a process behave differently from the first.
* **Relative paths resolve against the repository root**, never the working directory, which under
  an IDE Run button is whatever the IDE chose.
"""
from __future__ import annotations

import ast
import os
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from teb_vae.lag_attn_transformer_cfs.latent_pilot import config as pilot_config
from teb_vae.lag_attn_transformer_cfs.latent_pilot import run as pilot_run
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError


def _write(directory, block) -> str:
    """Write a pilot YAML carrying ``block`` under the pilot key, and return its path."""
    path = Path(directory) / "pilot.yaml"
    path.write_text(
        yaml.safe_dump({pilot_config.PILOT_KEY: block}, sort_keys=False), encoding="utf-8"
    )
    return str(path)


# =============================================================================
# The schema
# =============================================================================
def test_an_empty_block_resolves_to_the_declared_protocol(tmp_path):
    """The defaults are the protocol, so a config that says nothing runs the declared experiment."""
    settings = pilot_config.resolve_settings(_write(tmp_path, {}))
    assert settings["fold"] == pilot_config.DEFAULTS["fold"]
    assert settings["seed"] == pilot_config.DEFAULTS["seed"]
    assert settings["windows"]["supervised_hours"] == 1.0
    assert settings["windows"]["preservation_hours"] == 3.0
    assert settings["optim"]["max_epochs"] == 10
    assert settings["gates"]["forecast_mse_max_increase"] == 0.10


def test_the_file_overrides_a_default_without_dropping_its_siblings(tmp_path):
    """A partial block edits one leaf; the rest of that block keeps the protocol's values."""
    settings = pilot_config.resolve_settings(_write(tmp_path, {"optim": {"max_epochs": 3}}))
    assert settings["optim"]["max_epochs"] == 3
    assert settings["optim"]["patience"] == pilot_config.DEFAULTS["optim"]["patience"]


def test_overrides_beat_the_file_and_device_beats_both(tmp_path):
    path = _write(tmp_path, {"seed": 7, "device": "cpu"})
    settings = pilot_config.resolve_settings(
        path, overrides={"seed": 9}, device="cuda:1"
    )
    assert settings["seed"] == 9
    assert settings["device"] == "cuda:1"


def test_a_list_setting_is_replaced_rather_than_concatenated(tmp_path):
    path = _write(tmp_path, {"paths": {"train_shards": ["a.hdf5", "b.hdf5"]}})
    settings = pilot_config.resolve_settings(
        path, overrides={"paths": {"train_shards": ["c.hdf5"]}}
    )
    assert [Path(value).name for value in settings["paths"]["train_shards"]] == ["c.hdf5"]


def test_an_unknown_key_is_refused_at_every_depth(tmp_path):
    with pytest.raises(PilotConfigError, match="unknown"):
        pilot_config.resolve_settings(_write(tmp_path, {"epochs": 3}))
    with pytest.raises(PilotConfigError, match="unknown"):
        pilot_config.resolve_settings(_write(tmp_path, {"optim": {"momentum": 0.9}}))


def test_a_key_outside_the_pilot_block_is_refused(tmp_path):
    """A model or dataset block here would look like it reconfigured the checkpoint."""
    path = Path(tmp_path) / "pilot.yaml"
    path.write_text(
        yaml.safe_dump({pilot_config.PILOT_KEY: {}, "model_config": {"d_z": 8}}), encoding="utf-8"
    )
    with pytest.raises(PilotConfigError, match="model_config"):
        pilot_config.resolve_settings(str(path))


def test_a_wrong_type_is_refused_rather_than_coerced(tmp_path):
    with pytest.raises(PilotConfigError):
        pilot_config.resolve_settings(_write(tmp_path, {"seed": "forty-two"}))
    with pytest.raises(PilotConfigError):
        pilot_config.resolve_settings(_write(tmp_path, {"paths": {"train_shards": "one.hdf5"}}))


def test_an_out_of_range_setting_is_refused(tmp_path):
    with pytest.raises(PilotConfigError):
        pilot_config.resolve_settings(_write(tmp_path, {"optim": {"mean_head_lr": 0.0}}))
    with pytest.raises(PilotConfigError):
        pilot_config.resolve_settings(_write(tmp_path, {"optim": {"recordings_per_class": 1}}))
    with pytest.raises(PilotConfigError):
        pilot_config.resolve_settings(_write(tmp_path, {"bootstrap": {"resamples": 10}}))


def test_windows_that_do_not_nest_are_refused(tmp_path):
    with pytest.raises(PilotConfigError, match="preservation_hours"):
        pilot_config.resolve_settings(
            _write(tmp_path, {"windows": {"supervised_hours": 4.0}})
        )


def test_a_bin_width_that_does_not_divide_the_window_is_refused(tmp_path):
    with pytest.raises(PilotConfigError, match="divide"):
        pilot_config.resolve_settings(_write(tmp_path, {"windows": {"bin_hours": 0.4}}))


def test_an_early_window_overlapping_the_supervised_bag_is_refused(tmp_path):
    with pytest.raises(PilotConfigError, match="overlaps"):
        pilot_config.resolve_settings(
            _write(tmp_path, {"windows": {"early_window_hours": [0.5, 2.0]}})
        )


def test_one_shard_in_two_splits_is_refused(tmp_path):
    with pytest.raises(PilotConfigError, match="both"):
        pilot_config.resolve_settings(
            _write(tmp_path, {"paths": {
                "train_shards": ["a.hdf5"], "val_shards": ["a.hdf5"],
            }})
        )


def test_a_run_root_outside_the_package_is_refused(tmp_path):
    with pytest.raises(PilotConfigError, match="outside this package"):
        pilot_config.resolve_settings(
            _write(tmp_path, {"paths": {"run_root": str(tmp_path)}})
        )


# =============================================================================
# Paths
# =============================================================================
def test_a_relative_path_resolves_against_the_repository_root_not_the_cwd(tmp_path, monkeypatch):
    path = _write(tmp_path, {"paths": {"checkpoint": "some/where/model.ckpt"}})
    monkeypatch.chdir(tmp_path)
    settings = pilot_config.resolve_settings(path)
    assert settings["paths"]["checkpoint"] == str(
        pilot_config.REPO_ROOT / "some" / "where" / "model.ckpt"
    )


def test_an_absolute_path_passes_through_untouched(tmp_path):
    absolute = str(Path(tmp_path) / "model.ckpt")
    settings = pilot_config.resolve_settings(
        _write(tmp_path, {"paths": {"checkpoint": absolute}})
    )
    assert settings["paths"]["checkpoint"] == absolute


def test_resolving_does_not_mutate_the_overrides_it_was_handed(tmp_path):
    overrides = {"optim": {"max_epochs": 2}}
    snapshot = deepcopy(overrides)
    pilot_config.resolve_settings(_write(tmp_path, {}), overrides=overrides)
    assert overrides == snapshot


def test_the_resolved_settings_record_which_file_configured_them(tmp_path):
    path = _write(tmp_path, {})
    assert pilot_config.resolve_settings(path)["config_path"] == path


def test_a_missing_config_file_names_the_root_it_resolved_against():
    with pytest.raises(FileNotFoundError, match="repository root"):
        pilot_config.resolve_settings("no/such/pilot.yaml")


# =============================================================================
# Stages and their inputs
# =============================================================================
def test_all_expands_to_every_stage_in_run_order():
    assert list(pilot_config.stage_plan(pilot_config.ALL_STAGE)) == list(pilot_config.STAGES)
    assert tuple(pilot_config.stage_plan("report")) == ("report",)


def test_an_unknown_stage_names_the_valid_ones():
    with pytest.raises(PilotConfigError, match="unknown stage"):
        pilot_config.stage_plan("finetuning")


def test_the_stage_order_locks_selection_before_the_test_split_is_read():
    stages = list(pilot_config.STAGES)
    assert stages.index("control") < stages.index("evaluate") < stages.index("report")
    assert stages.index("extract") < stages.index("baseline") < stages.index("finetune")


def test_a_fitting_stage_without_a_checkpoint_is_refused_by_name(tmp_path):
    settings = pilot_config.resolve_settings(_write(tmp_path, {}))
    with pytest.raises(PilotConfigError, match="paths.checkpoint"):
        pilot_config.require_inputs(settings, ["extract"])


def test_tests_smoke_and_report_need_no_production_paths(tmp_path):
    """Which is what lets an operator check a configuration, and re-report a finished run, on a
    machine where the shards are no longer mounted."""
    settings = pilot_config.resolve_settings(_write(tmp_path, {}))
    pilot_config.require_inputs(settings, ["tests", "smoke", "report"])


def test_the_test_shards_are_required_only_by_the_evaluating_stage():
    assert "paths.test_shards" in pilot_config.STAGE_INPUTS["evaluate"]
    for stage in ("preflight", "extract", "baseline", "finetune", "control"):
        assert "paths.test_shards" not in pilot_config.STAGE_INPUTS[stage]


# =============================================================================
# Run arguments: the dictionary, the command line, and their precedence
# =============================================================================
def test_no_arguments_at_all_gives_the_dictionary(tmp_path):
    """The IDE Run-button path: ``argv=None`` skips parsing entirely."""
    resolved = pilot_run.resolve_run_args({"stage": "preflight"}, argv=None)
    assert resolved["stage"] == "preflight"
    assert resolved["config_path"] == pilot_run.DEFAULT_CONFIG_PATH


def test_an_explicit_command_line_value_beats_the_dictionary():
    resolved = pilot_run.resolve_run_args({"stage": "preflight"}, argv=["--stage", "report"])
    assert resolved["stage"] == "report"


def test_an_unsupplied_flag_never_outranks_the_dictionary():
    """The parser default is ``None`` for every argument, including the boolean, precisely so that
    saying nothing on the command line is distinguishable from saying the default."""
    resolved = pilot_run.resolve_run_args(
        {"stage": "preflight", "device": "cpu", "run_dir": "x", "resume": True}, argv=[]
    )
    assert resolved["stage"] == "preflight"
    assert resolved["device"] == "cpu"
    assert resolved["resume"] is True


def test_a_set_override_is_parsed_as_yaml_and_merged_over_the_dictionary():
    resolved = pilot_run.resolve_run_args(
        {"overrides": {"optim": {"max_epochs": 10, "patience": 3}}},
        argv=["--set", "optim.max_epochs=2", "--set", "seed=7"],
    )
    assert resolved["overrides"]["optim"]["max_epochs"] == 2
    assert resolved["overrides"]["optim"]["patience"] == 3
    assert resolved["overrides"]["seed"] == 7


def test_a_set_override_carries_lists_and_nulls_rather_than_strings():
    resolved = pilot_run.resolve_run_args(
        {}, argv=["--set", "paths.train_shards=[a.hdf5,b.hdf5]", "--set", "device=null"]
    )
    assert resolved["overrides"]["paths"]["train_shards"] == ["a.hdf5", "b.hdf5"]
    assert resolved["overrides"]["device"] is None


def test_a_malformed_set_override_is_refused_rather_than_dropped():
    """A silently dropped override is worse than a refused one: the run would proceed under
    settings the operator believes they changed."""
    with pytest.raises(PilotConfigError, match="KEY=VALUE"):
        pilot_run.resolve_run_args({}, argv=["--set", "optim.max_epochs"])


def test_an_unknown_run_argument_is_refused_and_says_where_settings_belong():
    with pytest.raises(PilotConfigError, match="unknown run argument"):
        pilot_run.resolve_run_args({"epochs": 3}, argv=None)


def test_resume_without_a_run_directory_is_refused():
    with pytest.raises(PilotConfigError, match="nothing to resume"):
        pilot_run.resolve_run_args({"resume": True}, argv=None)


def test_an_unknown_stage_is_refused_by_the_same_check_from_either_source():
    with pytest.raises(PilotConfigError, match="unknown stage"):
        pilot_run.resolve_run_args({"stage": "finetuning"}, argv=None)
    with pytest.raises(PilotConfigError, match="unknown stage"):
        pilot_run.resolve_run_args({}, argv=["--stage", "finetuning"])


def test_resolving_never_mutates_the_module_dictionary():
    snapshot = deepcopy(pilot_run.RUN_ARGS)
    resolved = pilot_run.resolve_run_args(argv=["--set", "optim.max_epochs=2"])
    resolved["overrides"]["injected"] = True
    resolved["stage"] = "report"
    assert pilot_run.RUN_ARGS == snapshot


def test_the_resolved_overrides_share_nothing_with_the_dictionary():
    overrides = {"optim": {"max_epochs": 5}}
    resolved = pilot_run.resolve_run_args({"overrides": overrides}, argv=None)
    resolved["overrides"]["optim"]["max_epochs"] = 99
    assert overrides["optim"]["max_epochs"] == 5


# =============================================================================
# The launch convention, read off the module rather than assumed
# =============================================================================
def test_every_run_argument_is_a_key_of_the_shipped_dictionary():
    assert set(pilot_run.RUN_ARGS) == set(pilot_run.RUN_ARG_DEFAULTS)


def test_every_parser_destination_reaches_a_run_argument():
    """``set_overrides`` is the one that does not carry its own name: it folds into ``overrides``,
    which is why it is named separately rather than being allowed to look like a stray dest."""
    dests = {
        action.dest for action in pilot_run.build_parser()._actions if action.dest != "help"
    }
    assert dests - {"set_overrides"} <= set(pilot_run.RUN_ARG_DEFAULTS)
    assert "set_overrides" in dests


def test_no_argument_is_required_by_argparse():
    """``required=True`` fires before the dictionary is consulted, so it would make the Run button
    unusable whatever the dictionary said."""
    required = [
        action.dest for action in pilot_run.build_parser()._actions if action.required
    ]
    assert required == []


def test_no_argument_carries_a_non_none_argparse_default():
    defaulted = {
        action.dest: action.default
        for action in pilot_run.build_parser()._actions
        if action.dest != "help" and action.default is not None
    }
    assert defaulted == {}


def test_importing_the_runner_does_no_work():
    """Read off the source rather than inferred from behaviour: every module-level statement is an
    import, an assignment, a definition or the ``__main__`` guard. A bare call at module level is
    work done by importing, and importing happens during test collection, during ``--help``, and in
    every editor that indexes the file."""
    tree = ast.parse(Path(pilot_run.__file__).read_text(encoding="utf-8"))
    offenders = [
        type(node).__name__
        for node in tree.body
        if not isinstance(
            node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.Expr,
                   ast.FunctionDef, ast.ClassDef, ast.If)
        )
    ]
    assert offenders == []
    # An ``Expr`` at module level is a docstring here and nothing else; a call would be work.
    calls = [
        node for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert calls == []
    # Every module-level ``If`` is the launch guard or the import bootstrap, never a stage.
    guards = [node for node in tree.body if isinstance(node, ast.If)]
    assert len(guards) == 2


def test_the_smoke_configuration_is_never_the_production_one():
    assert pilot_run.SMOKE_CONFIG_PATH != pilot_run.DEFAULT_CONFIG_PATH
    assert (pilot_config.REPO_ROOT / pilot_run.SMOKE_CONFIG_PATH).is_file()
    assert (pilot_config.REPO_ROOT / pilot_run.DEFAULT_CONFIG_PATH).is_file()


def test_the_shipped_configurations_both_resolve_without_production_files():
    """An operator must be able to check a configuration before the data is mounted, and the smoke
    configuration must never point at production paths."""
    production = pilot_config.resolve_settings(
        pilot_config.REPO_ROOT / pilot_run.DEFAULT_CONFIG_PATH
    )
    smoke = pilot_config.resolve_settings(pilot_config.REPO_ROOT / pilot_run.SMOKE_CONFIG_PATH)
    assert production["paths"]["checkpoint"] is None
    assert smoke["fold"] != production["fold"]
    assert os.path.commonpath(
        [smoke["paths"]["run_root"], production["paths"]["run_root"]]
    ) == production["paths"]["run_root"]
    assert smoke["paths"]["run_root"] != production["paths"]["run_root"]
