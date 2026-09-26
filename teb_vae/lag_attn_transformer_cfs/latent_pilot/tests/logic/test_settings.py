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


def test_the_analysis_window_defaults_to_the_preservation_window(tmp_path):
    """Unset, the new key changes nothing -- which is what makes it safe to add.

    Every run written before ``windows.analysis_hours`` existed analysed exactly the window it
    preserved, and a default that did anything else would silently redescribe those runs.
    """
    settings = pilot_config.resolve_settings(_write(tmp_path, {}))
    assert settings["windows"]["analysis_hours"] is None
    assert pilot_config.analysis_hours(settings) == settings["windows"]["preservation_hours"]


def test_a_wider_analysis_window_is_accepted_and_leaves_preservation_alone(tmp_path):
    settings = pilot_config.resolve_settings(
        _write(tmp_path, {"windows": {"analysis_hours": 6.0}})
    )
    assert pilot_config.analysis_hours(settings) == 6.0
    for name in ("preservation_hours", "supervised_hours"):
        assert settings["windows"][name] == pilot_config.DEFAULTS["windows"][name]


def test_an_analysis_window_inside_the_preservation_window_is_refused(tmp_path):
    with pytest.raises(PilotConfigError, match="analysis_hours"):
        pilot_config.resolve_settings(
            _write(tmp_path, {"windows": {"analysis_hours": 2.0}})
        )


def test_the_bin_width_must_divide_the_analysis_window_not_the_preserved_one(tmp_path):
    """0.5 h divides three hours and not five, so a five-hour analysis at 0.4 h is refused."""
    pilot_config.resolve_settings(_write(tmp_path, {"windows": {"analysis_hours": 6.0}}))
    with pytest.raises(PilotConfigError, match="divide the analysis window"):
        pilot_config.resolve_settings(
            _write(tmp_path, {"windows": {"analysis_hours": 5.0, "bin_hours": 0.4}})
        )


def test_the_early_window_may_reach_into_the_wider_analysis_window(tmp_path):
    settings = pilot_config.resolve_settings(
        _write(tmp_path, {"windows": {
            "analysis_hours": 6.0, "early_window_hours": [5.0, 6.0],
        }})
    )
    assert settings["windows"]["early_window_hours"] == [5.0, 6.0]
    with pytest.raises(PilotConfigError, match="analysis window"):
        pilot_config.resolve_settings(
            _write(tmp_path, {"windows": {"early_window_hours": [2.0, 4.0]}})
        )


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


def test_a_run_root_outside_the_package_is_accepted(tmp_path):
    settings = pilot_config.resolve_settings(
        _write(tmp_path, {"paths": {"run_root": str(tmp_path / "elsewhere")}})
    )
    assert settings["paths"]["run_root"] == str((tmp_path / "elsewhere").resolve())


@pytest.mark.parametrize(
    "name,relative",
    [
        ("checkpoint", "model/best.ckpt"),
        ("train_shards", ["shards/healthy_bg_cs.hdf5"]),
    ],
)
def test_a_run_root_containing_an_input_is_refused(tmp_path, name, relative):
    """A checkpoint or a shard under the run root: either would put a run on top of its inputs."""
    value = (
        [str(tmp_path / item) for item in relative] if isinstance(relative, list)
        else str(tmp_path / relative)
    )
    with pytest.raises(PilotConfigError, match="contains the input"):
        pilot_config.resolve_settings(
            _write(tmp_path, {"paths": {"run_root": str(tmp_path), name: value}})
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
    saying nothing on the command line is distinguishable from saying the default.

    Every run argument is set away from its default, so an empty command line must resolve exactly
    as no command line at all. A parser default that were not ``None`` would overwrite its entry,
    and a ``required=True`` flag would stop the parse before the dictionary was consulted.
    """
    every_key = {
        "config_path": "some/pilot.yaml", "stage": "preflight", "device": "cpu",
        "run_dir": "x", "resume": True, "overrides": {"seed": 1},
    }
    assert set(every_key) == set(pilot_run.RUN_ARG_DEFAULTS)
    assert pilot_run.resolve_run_args(every_key, argv=[]) == pilot_run.resolve_run_args(
        every_key, argv=None
    )


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
def test_every_parser_destination_reaches_a_run_argument():
    """``set_overrides`` is the one that does not carry its own name: it folds into ``overrides``,
    which is why it is named separately rather than being allowed to look like a stray dest."""
    dests = {
        action.dest for action in pilot_run.build_parser()._actions if action.dest != "help"
    }
    assert dests - {"set_overrides"} <= set(pilot_run.RUN_ARG_DEFAULTS)
    assert "set_overrides" in dests


def test_the_shipped_configurations_both_resolve_without_production_files():
    """An operator must be able to check a configuration before the data is mounted, and the smoke
    configuration must never share the production one's inputs or its destination.

    Nothing here asserts that the production template's paths are still unset, and that is
    deliberate: filling them in is exactly what the file asks the operator to do, so a test that
    failed on a filled-in template would fail the ``tests`` stage of every real run -- the stage
    that gates all the others. Nor is the smoke root asserted to sit *under* the production root,
    which stopped being true the moment a run root was allowed to point anywhere writable.
    """
    production = pilot_config.resolve_settings(
        pilot_config.REPO_ROOT / pilot_run.DEFAULT_CONFIG_PATH
    )
    smoke = pilot_config.resolve_settings(pilot_config.REPO_ROOT / pilot_run.SMOKE_CONFIG_PATH)
    assert smoke["fold"] != production["fold"]
    assert smoke["paths"]["run_root"] != production["paths"]["run_root"]
    for name in ("checkpoint", "statistics"):
        assert smoke["paths"][name] != production["paths"][name]
    for name in ("train_shards", "val_shards", "test_shards"):
        assert not set(smoke["paths"][name]) & set(production["paths"][name])
