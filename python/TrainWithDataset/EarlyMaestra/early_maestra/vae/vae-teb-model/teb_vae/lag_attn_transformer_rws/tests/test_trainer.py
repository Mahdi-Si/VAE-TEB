r"""The driver turns config into this model, and the entry point runs the guards before anything.

Almost all of this module's behaviour is inherited, which is the design and also the risk: three
class attributes are the entire difference between building this architecture and building the one
it is compared against, and a stale one produces a run that trains the wrong model, wraps it in the
wrong task, or writes its checkpoints under the wrong stem -- none of which raises.

The pre-flight guards get their own assertions for the same reason they exist: their whole value is
failing *before* the run directory, the log sinks and the MLflow run exist on every rank of a
multi-rank launch. A guard that silently stopped running would only be noticed by the run it failed
to protect.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import pytest
import torch
import yaml
from lightning.pytorch.callbacks import Callback, LearningRateMonitor

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws import trainer as shared_trainer
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer
from teb_vae.lag_attn_transformer_rws import trainer as trainer_module
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask
from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer
from train.graph_model_base import GraphModelBase

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


def _tiny_config_at(tmp_path, mutate=None) -> str:
    """Write a resolved, path-absolutised copy of the tiny config into ``tmp_path``.

    Args:
        tmp_path: Directory to write into.
        mutate: Optional callable applied to the config before it is written.

    Returns:
        The written path.
    """
    config = absolutize_dataset_paths(load_config(str(_TINY)))
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path / "runs")
    if mutate is not None:
        mutate(config)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------------------
# The three class attributes, and what they decide
# --------------------------------------------------------------------------------------
def test_the_driver_names_this_architecture_and_not_the_one_it_is_compared_against():
    """A stale attribute here trains the other model under this model's config, tag and MLflow
    experiment, and nothing anywhere raises."""
    assert LagAttnTrfRwsTrainer.MODEL_CLS is SeqVaeLagAttnTrfRws
    assert LagAttnTrfRwsTrainer.TASK_CLS is SeqVaeLagAttnTrfRwsTask
    assert LagAttnTrfRwsTrainer.CHECKPOINT_STEM == "lag-attn-trf-rws"
    assert LagAttnTrfRwsTrainer.MODEL_CLS is not LagAttnRwsTrainer.MODEL_CLS
    assert LagAttnTrfRwsTrainer.CHECKPOINT_STEM != LagAttnRwsTrainer.CHECKPOINT_STEM


def test_the_inherited_driver_still_builds_the_model_it_always_did():
    """The three attributes were introduced so this package could reuse the driver; the reuse is
    worthless if it changed what a launch of the comparison model produces."""
    assert LagAttnRwsTrainer.MODEL_CLS is SeqVaeLagAttnRws
    assert LagAttnRwsTrainer.CHECKPOINT_STEM == "lag-attn-rws"


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


def test_the_geometry_reaches_the_constructor(driver):
    kwargs = driver._build_model_kwargs()

    assert kwargs["sequence_length"] == 300
    assert kwargs["d_model"] == 128
    assert kwargs["d_z"] == 48
    assert kwargs["horizon"] == 30
    assert kwargs["raw_per_step"] == 16
    assert kwargs["warmup_period"] == 30
    assert kwargs["c_y"] == 109
    assert kwargs["c_u"] == 58
    assert kwargs["max_lag"] == 90
    assert kwargs["coverage_floor"] == 0.9


def test_loss_only_keys_do_not_reach_the_constructor(driver):
    """The net takes tensors and computes a loss on request; it owns none of these. The
    constructor is keyword-only with no ``**kwargs``, so a leaked key would be a ``TypeError`` on
    the production config -- a poor place to find out."""
    kwargs = driver._build_model_kwargs()

    for name in (
        "likelihood", "free_bits", "lambda_full", "lambda_base", "beta_schedule",
        "kld_beta", "beta_prior", "causal_reach_budget_s",
    ):
        assert name not in kwargs, f"{name} is not the net's"


def test_the_resolved_kwargs_actually_build_a_model(driver):
    """The sweep's output is only correct if the constructor accepts it."""
    model = SeqVaeLagAttnTrfRws(**driver._build_model_kwargs())

    assert isinstance(model, SeqVaeLagAttnTrfRws)
    # The unconditional freeze the DDP strategy relies on.
    assert not any(parameter.requires_grad for parameter in model.lag_attn.W_o.parameters())
    # The depthwise correction actually ran: the generic pass Xavier-fills a (C, 1, k) weight at
    # roughly an eighth of the variance-preserving scale, and a count of zero would mean the
    # repair was a no-op on a model that has a stem.
    assert model.n_depthwise_init == 4


def test_a_replaced_encoder_key_is_dropped_rather_than_forwarded(driver):
    """The sweep forwards by name against the real signature, so a copy-pasted key from the
    comparison model's config cannot crash a launch -- but it also cannot reach anything, which is
    why the config test asserts none of them is present in the first place."""
    driver.config["model_config"]["VAE_model"]["lstm_layers"] = 2

    assert "lstm_layers" not in driver._build_model_kwargs()


def test_init_weights_is_never_a_config_decision(driver):
    """Skipping initialisation would also skip the depthwise correction and the post-init
    delta-head zeroing the zero-KL start depends on; the key is refused even when a config
    supplies it."""
    driver.config["model_config"]["VAE_model"]["init_weights"] = False

    assert "init_weights" not in driver._build_model_kwargs()


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
    assert trainer_module.NULLABLE_MODEL_KEYS == frozenset({"source_attention_window"})


# --------------------------------------------------------------------------------------
# create_model
# --------------------------------------------------------------------------------------
def test_create_model_builds_this_net_and_wraps_it_in_this_task(driver):
    driver.create_model()

    assert isinstance(driver.pytorch_model, SeqVaeLagAttnTrfRws)
    assert isinstance(driver.pl_model, SeqVaeLagAttnTrfRwsTask)
    assert driver.pl_model.orig_model is driver.pytorch_model


def test_create_model_does_not_trip_on_the_absent_causal_norm_flag(driver):
    """The inherited ``create_model`` used to read ``causal_norm`` off the net unconditionally.
    There is no time-pooling normaliser left in this architecture to causalise, so the attribute
    does not exist and a bare read would be an ``AttributeError`` on the first launch."""
    driver.create_model()

    assert not hasattr(driver.pytorch_model, "causal_norm")


def test_create_model_passes_the_spike_breaker_block_to_the_task(driver):
    """The block is validated by the framework and read by the module -- but nothing forwards it
    automatically, so a driver that forgot would leave a fully configured ``enabled: true`` block
    doing nothing at all."""
    driver.create_model()

    breaker = driver.pl_model.hparams["spike_breaker"]
    assert breaker["enabled"] is True
    assert breaker["comparison_metric"] == "main_loss"
    assert breaker["ema_floor"] >= 1.0e9


def test_create_model_passes_the_loss_hyperparameters_to_the_task(driver):
    """The objective is shared with the comparison model, so these values are the comparison."""
    driver.create_model()

    hparams = driver.pl_model.hparams
    assert hparams["likelihood"] == "gaussian_nll"
    assert hparams["lambda_full"] == 1.0
    assert hparams["lambda_base"] == 1.0
    assert hparams["free_bits"] == 0.0
    assert hparams["beta_schedule"]["kind"] == "linear_warmup"
    assert hparams["beta_schedule"]["start"] == 0.0


def test_create_model_forwards_the_step_warmup(driver):
    driver.create_model()

    assert int(driver.pl_model.hparams["lr_warmup_steps"]) == 2000


def test_the_checkpoint_kwargs_are_the_ones_the_model_was_built_from(driver):
    """So the blob rebuilds into this architecture and not the constructor's defaults."""
    driver.create_model()

    assert driver.pl_model._model_kwargs == driver._build_model_kwargs()


def test_a_core_checkpoint_from_the_comparison_model_is_refused_before_it_is_loaded(
    driver, tmp_path
):
    """The two models share every downstream tensor name, so a blob from the other one would
    partially align by accident and train from a mixture of loaded and random weights."""
    foreign = tmp_path / "foreign.ckpt"
    torch.save({"state_dict": {}, "model_class": "SeqVaeLagAttnRws"}, foreign)
    driver.config["model_config"]["core_model_checkpoint"] = str(foreign)

    with pytest.raises(ValueError, match="does not match the active model class"):
        driver.create_model()


def test_an_unalignable_core_checkpoint_raises_rather_than_training_from_scratch(
    driver, tmp_path
):
    """``load_checkpoint_strict`` returns ``None`` when nothing lines up; it does not raise."""
    unrelated = tmp_path / "unrelated.ckpt"
    torch.save({"state_dict": {"nothing.like.this": torch.zeros(2)}}, unrelated)
    driver.config["model_config"]["core_model_checkpoint"] = str(unrelated)

    with pytest.raises(RuntimeError, match="could not align"):
        driver.create_model()


# --------------------------------------------------------------------------------------
# The trainer kwargs: the DDP strategy and the learning-rate monitor
# --------------------------------------------------------------------------------------
def test_the_ddp_strategy_follows_the_configured_parameter_usage(driver):
    """Plain ``'ddp'`` implies ``find_unused_parameters=False``, under which the reducer expects
    every parameter to be marked ready in every backward. Exactly one group can starve, and it is
    decided by config: the decoder log-variance heads are consumed only under ``gaussian_nll``."""
    config = driver.config

    assert driver.select_ddp_strategy(1, config) == "auto"
    assert driver.select_ddp_strategy(7, config) == "ddp"
    config["model_config"]["VAE_model"]["likelihood"] = "mse"
    assert driver.select_ddp_strategy(7, config) == "ddp_find_unused_parameters_true"


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


def test_the_frameworks_own_monitor_is_still_epoch_granular_for_the_comparison_model(tmp_path):
    """The replacement is this driver's, not a change to the framework: the comparison model must
    keep logging at the granularity its own schedule operates on."""
    sibling_config = _REPO_ROOT / "teb_vae" / "lag_attn_rws" / "configs" / "default.yaml"
    instance = LagAttnRwsTrainer(config_file_path=str(sibling_config))
    instance.train_results_dir = str(tmp_path)

    monitors = [
        callback
        for callback in instance._build_trainer_kwargs([])["callbacks"]
        if isinstance(callback, LearningRateMonitor)
    ]

    assert [monitor.logging_interval for monitor in monitors] == ["epoch"]


def test_the_callbacks_this_model_supplies_survive_the_replacement(driver):
    """The monitor swap rebuilds the callback list, so it must not drop what ``train_model``
    handed in."""
    sentinel = Callback()

    callbacks = driver._build_trainer_kwargs([sentinel])["callbacks"]

    assert sentinel in callbacks


# --------------------------------------------------------------------------------------
# The entry point and its four pre-flight guards
# --------------------------------------------------------------------------------------
class _StubDataModule:
    """A data module that hands out loaders nothing iterates."""

    def __init__(self, config=None):
        self.config = config

    def train_dataloader(self):
        return object()

    def val_dataloader(self):
        return object()


@pytest.fixture
def recording_main(monkeypatch):
    """Run ``main`` with every expensive step recorded rather than done."""
    calls = []

    def _record(name, result=None):
        def _recorder(self, *args, **kwargs):
            calls.append(name)
            return result

        return _recorder

    monkeypatch.setattr(LagAttnTrfRwsTrainer, "setup_config", _record("setup_config"))
    monkeypatch.setattr(LagAttnTrfRwsTrainer, "create_model", _record("create_model"))
    monkeypatch.setattr(LagAttnTrfRwsTrainer, "train_model", _record("train_model"))
    monkeypatch.setattr(shared_trainer, "GraphDataModule", _StubDataModule)
    return calls


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


def test_setup_config_runs_before_the_model_is_built(recording_main, tmp_path):
    """The order that decides whether a run is seeded, logged and tracked at all: building the
    model first means no seeding, no log sinks, nowhere to write, and ``mlflow_logger is None`` --
    which silently drops the MLflow callback from the fit."""
    trainer_module.main(_tiny_config_at(tmp_path))

    assert recording_main == ["setup_config", "create_model", "train_model"]


def test_all_four_pre_flight_guards_run_before_setup_config(tmp_path, monkeypatch):
    """Their whole value is failing before the run directory and MLflow run exist on every rank of
    a multi-rank launch."""
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
            shared_trainer, attribute, lambda config, _label=label: order.append(_label)
        )
    monkeypatch.setattr(shared_trainer, "GraphDataModule", lambda config: None)
    monkeypatch.setattr(
        LagAttnTrfRwsTrainer, "create_model", lambda self: order.append("create_model")
    )

    with pytest.raises(AttributeError):
        # GraphDataModule is stubbed to None, so main dies at train_dataloader() -- after the part
        # under test. The order up to that point is the assertion.
        trainer_module.main(_tiny_config_at(tmp_path))

    assert order[:5] == [
        "stat_path", "widths", "fhr_normalized", "causal_budget", "setup_config",
    ], order


def test_a_missing_stat_path_raises_before_any_training_happens(recording_main, tmp_path):
    """The loader passes ``None`` straight through and merely skips normalization, so an unset
    ``stat_path`` trains on raw-scale inputs and an unnormalized raw target, reporting nothing."""

    def _drop_stats(config):
        config["dataset_config"]["stat_path"] = None

    with pytest.raises(ValueError, match="stat_path"):
        trainer_module.main(_tiny_config_at(tmp_path, _drop_stats))

    assert "create_model" not in recording_main


def test_a_width_mismatch_against_the_shard_raises_before_any_training_happens(
    recording_main, tmp_path
):
    """$58$ is both the current ``use_up_st=true`` width and the old phase-only one, so this exact
    misconfiguration passes every config-shaped check and only the shard can catch it."""

    def _stale_width(config):
        config["model_config"]["VAE_model"]["use_up_st"] = False

    with pytest.raises(ValueError, match="channel widths disagree"):
        trainer_module.main(_tiny_config_at(tmp_path, _stale_width))

    assert "create_model" not in recording_main


@pytest.mark.parametrize("list_key", ["load_fields", "normalize_fields"])
def test_a_missing_raw_target_field_raises_naming_the_offending_list(
    recording_main, tmp_path, list_key
):
    """Without 'fhr' in ``normalize_fields`` the target arrives in bpm, the Gaussian NLL is
    meaningless, and nothing else raises -- a full run trained on nothing."""

    def _drop_fhr(config):
        dataloader = config["dataset_config"]["dataloader_config"]
        fields = (
            dataloader["dataset_kwargs"]["load_fields"]
            if list_key == "load_fields"
            else dataloader[list_key]
        )
        fields.remove("fhr")

    with pytest.raises(ValueError, match=list_key):
        trainer_module.main(_tiny_config_at(tmp_path, _drop_fhr))

    assert "create_model" not in recording_main


def test_an_unsatisfiable_reach_budget_raises_before_any_training_happens(
    recording_main, tmp_path
):
    """A budget whose delay outruns ``warmup_period`` would zero-fill trained anchors."""

    def _too_deep(config):
        config["model_config"]["VAE_model"]["causal_reach_budget_s"] = 240.0

    with pytest.raises(ValueError, match="warmup_period"):
        trainer_module.main(_tiny_config_at(tmp_path, _too_deep))

    assert "create_model" not in recording_main


def test_the_resolved_config_is_written_beside_the_checkpoints(tmp_path, monkeypatch):
    """A run's own config is otherwise recoverable only from the text of its log or from an MLflow
    artifact whose on-disk location nothing can derive, and neither is a file a later offline pass
    can open."""
    monkeypatch.setattr(LagAttnTrfRwsTrainer, "create_model", lambda self: None)
    monkeypatch.setattr(LagAttnTrfRwsTrainer, "train_model", lambda self, *args: None)
    monkeypatch.setattr(shared_trainer, "GraphDataModule", lambda config: _StubDataModule())

    captured = {}

    def _remember(self):
        # setup_config is what creates the run directories, so the write must follow it; the real
        # one runs, and only the directory it chose is recorded.
        GraphModelBase.setup_config(self)
        captured["checkpoint_dir"] = self.model_checkpoint_dir

    monkeypatch.setattr(LagAttnTrfRwsTrainer, "setup_config", _remember)

    trainer_module.main(_tiny_config_at(tmp_path))

    written = Path(captured["checkpoint_dir"]) / shared_trainer.RESOLVED_CONFIG_FILENAME
    assert written.is_file()
    reloaded = yaml.safe_load(written.read_text(encoding="utf-8"))
    # Fully resolved: the inherited keys are present and the `base:` pointer is gone.
    assert "base" not in reloaded
    assert reloaded["model_config"]["VAE_model"]["target_attention_blocks"] == 4
    # The unguarded default records the *absence* of a guard explicitly, so a reader can tell it
    # from a run written before the record existed.
    assert reloaded["model_config"][shared_trainer.RESOLVED_BUDGET_KEY] is None


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
# Hygiene
# --------------------------------------------------------------------------------------
def test_no_module_in_the_package_seeds_by_hand():
    """``general_config.seed`` through the framework's ``configure_determinism`` is the only
    seeding route; a stray global seed would silently override it while looking like diligence."""
    package_dir = Path(__file__).resolve().parents[1]
    offenders = []
    for path in package_dir.rglob("*.py"):
        if "tests" in path.parts:
            continue  # tests seed themselves for reproducibility, legitimately
        source = path.read_text(encoding="utf-8")
        for pattern in ("torch.manual_seed", "seed_everything", "np.random.seed"):
            if pattern in source:
                offenders.append(f"{path.name}: {pattern}")

    assert offenders == []
