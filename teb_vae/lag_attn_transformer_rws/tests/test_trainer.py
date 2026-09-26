r"""The driver turns config into this model, and the entry point runs the guards before anything.

Almost all of this module's behaviour is inherited, and the inherited driver, entry point and
guards are tested where they are defined. What is checked here is what this package overrides or
supplies: the kwarg sweep's re-admission of the one key whose ``null`` is a value, the net and
task ``create_model`` builds, the step-granular learning-rate monitor, the order of the pre-flight
guards and the driver hook ahead of ``setup_config`` when launched through this entry point, the
command line and IDE launch path, and the ``compile`` key this architecture makes live.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import pytest
import yaml
from lightning.pytorch.callbacks import Callback, LearningRateMonitor

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws import trainer as shared_trainer
from teb_vae.lag_attn_transformer_rws import trainer as trainer_module
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask
from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer

from .conftest import SHIPPED_KWARGS, absolutize_dataset_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"
_MODULE_NAME = "teb_vae.lag_attn_transformer_rws.trainer"


@pytest.fixture
def driver(tmp_path):
    """A driver on the shipped config, with its output directories redirected under ``tmp_path``.

    ``setup_config`` is never called -- it would seed, open log sinks and probe MLflow -- so the
    directories are assigned directly.
    """
    instance = LagAttnTrfRwsTrainer(config_file_path=str(_CONFIG))
    instance.output_base_dir = str(tmp_path)
    instance.train_results_dir = str(tmp_path / "train_results")
    instance.model_checkpoint_dir = str(tmp_path / "model_checkpoints")
    return instance


def _tiny_config_at(tmp_path) -> str:
    """Write a resolved, path-absolutised copy of the tiny config into ``tmp_path``.

    Args:
        tmp_path: Directory to write into.

    Returns:
        The written path.
    """
    config = absolutize_dataset_paths(load_config(str(_TINY)))
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path / "runs")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------------------
# The kwarg sweep
# --------------------------------------------------------------------------------------
def test_the_shipped_config_resolves_to_the_shipped_architecture(driver):
    """Every architectural flag ``SHIPPED_KWARGS`` claims the config sets, it must set. That
    fixture is the suite's description of the production model; this keeps it honest against the
    config file itself."""
    kwargs = driver._build_model_kwargs()

    for name in (
        "lag_bias_init", "use_entmax", "use_up_st", "horizon_depth", "horizon_kernel",
        "horizon_film", "encoder_num_heads", "encoder_d_ff", "target_attention_blocks",
        "source_attention_blocks", "source_attention_window",
    ):
        assert kwargs[name] == SHIPPED_KWARGS[name], f"{name} disagrees with the shipped flag set"
    # YAML has no tuple; the constructor coerces, so the sweep hands the list through.
    assert tuple(kwargs["encoder_conv_kernels"]) == SHIPPED_KWARGS["encoder_conv_kernels"]
    assert tuple(kwargs["encoder_conv_dilations"]) == SHIPPED_KWARGS["encoder_conv_dilations"]
    assert tuple(kwargs["logvar_clamp"]) == SHIPPED_KWARGS["logvar_clamp"]


def test_the_resolved_kwargs_actually_build_a_model(driver):
    """The sweep's output is only correct if the constructor accepts it."""
    model = SeqVaeLagAttnTrfRws(**driver._build_model_kwargs())

    # The unconditional freeze the DDP strategy relies on.
    assert not any(parameter.requires_grad for parameter in model.lag_attn.W_o.parameters())


def test_a_null_source_window_reaches_the_constructor_as_the_unbounded_encoder(driver):
    """The inherited sweep drops every ``null``, reading it as "leave the constructor default".
    That is right for a key whose null means *unset* and wrong for the one key here whose null is
    a value: an unbounded source encoder **is** ``source_attention_window: null``. Dropped, the
    sweep would rebuild the shipped 16-step window while the arm still reported under the
    unbounded arm's name -- a confound with no failure anywhere to read it off."""
    driver.config["model_config"]["VAE_model"]["source_attention_window"] = None

    kwargs = driver._build_model_kwargs()

    assert "source_attention_window" in kwargs
    assert kwargs["source_attention_window"] is None
    assert SeqVaeLagAttnTrfRws(**kwargs).source_encoder.receptive_field is None


def test_the_inherited_sweep_still_drops_every_other_null(driver):
    """The other direction, and the reason the re-admission is a declared key set rather than a
    blanket change: a ``null`` anywhere else still means "use the constructor's own default", and
    forwarding one would hand the net a ``None`` where it expects a number."""
    driver.config["model_config"]["VAE_model"]["d_head"] = None

    kwargs = driver._build_model_kwargs()

    assert "d_head" not in kwargs


# --------------------------------------------------------------------------------------
# create_model
# --------------------------------------------------------------------------------------
def test_create_model_builds_this_net_and_wraps_it_in_this_task(driver):
    driver.create_model()

    assert isinstance(driver.pytorch_model, SeqVaeLagAttnTrfRws)
    assert isinstance(driver.pl_model, SeqVaeLagAttnTrfRwsTask)
    assert driver.pl_model.orig_model is driver.pytorch_model


# --------------------------------------------------------------------------------------
# The trainer kwargs: the learning-rate monitor
# --------------------------------------------------------------------------------------
def test_exactly_one_learning_rate_monitor_is_attached_and_it_is_step_granular(driver):
    """A warm-up measured in optimizer steps completes well inside the first epochs, so the
    framework's epoch-granular monitor cannot show it at all -- and a schedule nobody can observe
    is a schedule nobody can tell has silently done nothing. Replaced rather than supplemented, so
    exactly one of them logs under one name."""
    monitors = [
        callback
        for callback in driver._build_trainer_kwargs([])["callbacks"]
        if isinstance(callback, LearningRateMonitor)
    ]

    assert len(monitors) == 1
    assert monitors[0].logging_interval == "step"


def test_the_callbacks_this_model_supplies_survive_the_replacement(driver):
    """The monitor swap rebuilds the callback list, so it must not drop what ``train_model``
    handed in."""
    sentinel = Callback()

    callbacks = driver._build_trainer_kwargs([sentinel])["callbacks"]

    assert sentinel in callbacks


# --------------------------------------------------------------------------------------
# The entry point and its four pre-flight guards
# --------------------------------------------------------------------------------------
def test_the_entry_point_constructs_this_packages_driver(monkeypatch, tmp_path):
    """``main`` delegates to the shared entry point, which owns the guards and the resolved-config
    write. What this package supplies is which driver it constructs -- and that is the one thing a
    delegation can get wrong while still running."""
    seen = {}

    def _capture_init(self, config_file_path=None):
        seen["cls"] = type(self)
        raise RuntimeError("stop here")

    monkeypatch.setattr(LagAttnTrfRwsTrainer, "__init__", _capture_init)

    with pytest.raises(RuntimeError, match="stop here"):
        trainer_module.main(_tiny_config_at(tmp_path))

    assert seen["cls"] is LagAttnTrfRwsTrainer


def test_all_four_pre_flight_guards_and_the_driver_hook_run_before_setup_config(
    tmp_path, monkeypatch
):
    """Their whole value is failing before the run directory and MLflow run exist on every rank of
    a multi-rank launch. The driver's own ``preflight`` runs last of the five and still before
    ``setup_config``, so a guard a sibling architecture adds keeps the same guarantee."""
    order = []
    monkeypatch.setattr(
        LagAttnTrfRwsTrainer, "setup_config", lambda self: order.append("setup_config")
    )
    for attribute, label in (
        ("_check_stat_path", "stat_path"),
        ("_check_declared_widths_against_shard", "widths"),
        ("_check_raw_target_normalized", "fhr_normalized"),
        ("_check_causal_budget_resolves", "causal_budget"),
    ):
        monkeypatch.setattr(
            # ``**_`` because the normalisation guard is handed the driver's TARGET_FIELDS;
            # these stubs record the order and have no opinion about any guard's arguments.
            shared_trainer, attribute, lambda config, _label=label, **_: order.append(_label)
        )
    monkeypatch.setattr(
        LagAttnTrfRwsTrainer,
        "preflight",
        classmethod(lambda cls, config: order.append("preflight")),
    )
    monkeypatch.setattr(shared_trainer, "GraphDataModule", lambda config: None)
    monkeypatch.setattr(
        LagAttnTrfRwsTrainer, "create_model", lambda self: order.append("create_model")
    )

    with pytest.raises(AttributeError):
        # GraphDataModule is stubbed to None, so main dies at train_dataloader() -- after the part
        # under test. The order up to that point is the assertion.
        trainer_module.main(_tiny_config_at(tmp_path))

    assert order[:6] == [
        "stat_path", "widths", "fhr_normalized", "causal_budget", "preflight", "setup_config",
    ], order


# --------------------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------------------
def test_relative_config_paths_resolve_against_the_repository_root():
    """An IDE's working directory is arbitrary; every documented invocation is repo-root relative,
    so the resolver must anchor there and leave absolute paths alone."""
    resolved = trainer_module._resolve_cli_config_path(
        "teb_vae/lag_attn_transformer_rws/configs/tiny.yaml"
    )

    assert Path(resolved) == _TINY
    absolute = str(_TINY)
    assert trainer_module._resolve_cli_config_path(absolute) == absolute


def test_run_config_points_at_a_config_that_exists():
    """The IDE Run button resolves through ``RUN_CONFIG``; a stale path breaks it silently."""
    assert trainer_module.RUN_CONFIG is not None
    assert (_REPO_ROOT / trainer_module.RUN_CONFIG).is_file()


def _launched_config(monkeypatch, argv) -> str:
    """Execute the module's ``__main__`` block under ``argv`` and return the config it launched.

    The block is re-executed in a fresh namespace, so the ``main`` it calls is the one this
    package's module imports from the shared entry point -- patching that name is what makes the
    launch observable without running a fit.

    Args:
        monkeypatch: The pytest fixture.
        argv: The command line, including the program name.

    Returns:
        The config path the entry point was called with.
    """
    launched = {}
    monkeypatch.setattr(
        shared_trainer, "main", lambda path, trainer_cls=None: launched.setdefault("path", path)
    )
    monkeypatch.setattr(sys, "argv", list(argv))
    # The block chdirs to the repo root when it is not already there; entering it under monkeypatch
    # is what restores the working directory afterwards.
    monkeypatch.chdir(os.getcwd())

    runpy.run_module(_MODULE_NAME, run_name="__main__", alter_sys=True)

    return launched["path"]


def test_the_command_line_config_wins_over_run_config(monkeypatch, tmp_path):
    """``RUN_CONFIG`` exists and is valid, so an entry point that ignored ``--config`` would launch
    a full production run when a smoke run was asked for."""
    requested = str(tmp_path / "requested.yaml")

    assert _launched_config(monkeypatch, ["trainer", "--config", requested]) == requested


def test_a_launch_with_no_command_line_falls_back_to_run_config(monkeypatch):
    """Which is what makes an IDE Run button work at all."""
    launched = _launched_config(monkeypatch, ["trainer"])

    assert Path(launched) == _REPO_ROOT / trainer_module.RUN_CONFIG


# --------------------------------------------------------------------------------------
# torch.compile: live here, and this is the driver that decides it
# --------------------------------------------------------------------------------------
def test_the_key_is_live_here_rather_than_ignored(driver):
    """The distinction against the raw-signal base, whose LSTM encoders this architecture replaced.
    If the key stayed inert an operator could set ``compile: true``, see nothing in the log, and
    believe they had measured a compiled run."""
    driver.config["advanced_config"]["trainer"]["compile"] = True

    assert driver.compile_model_requested() is True


def test_compilation_and_attention_checkpointing_are_refused_together(driver):
    """The one genuine inductor blocker still reachable from this config surface. Silently dropping
    either would give a run that is neither the compiled one nor the checkpointed one."""
    driver.config["advanced_config"]["trainer"]["compile"] = True
    driver.config["model_config"]["VAE_model"]["attention_grad_checkpoint"] = True

    with pytest.raises(ValueError) as excinfo:
        driver.compile_model_requested()

    message = str(excinfo.value)
    assert "attention_grad_checkpoint" in message and "compile" in message
