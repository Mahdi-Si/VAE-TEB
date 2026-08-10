r"""The driver turns config into this model, and it is five class attributes and an entry point.

Almost all of this module's behaviour is inherited, which is the design and also the risk: a stale
attribute produces a run that trains the wrong model, wraps it in the wrong task, writes its
checkpoints under the wrong stem, or checks the wrong loader fields for normalisation -- and none
of those raises.

``TARGET_FIELDS`` is the one that is specific to a target-domain change and the one with no
downstream symptom at all. Without ``fhr_st`` and ``fhr_ph`` in ``normalize_fields`` the target
arrives at its stored scale, the Gaussian NLL is computed against a z-scale variance model, and a
multi-day run trains a meaningless objective to completion. The guard that catches it lives in the
shared entry point and reads the driver it was handed, so a ``trainer_cls=`` wiring mistake would
leave it checking the *raw* model's field -- which this config satisfies.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_fs import trainer as trainer_module
from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.task import SeqVaeLagAttnFsTask
from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer
from teb_vae.lag_attn_rws import trainer as shared_trainer
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.trainer import _TRACKED_METRICS, LagAttnRwsTrainer
from train.graph_model_base import GraphModelBase

from .conftest import SHIPPED_KWARGS, absolutize_dataset_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"
_MODULE_NAME = "teb_vae.lag_attn_fs.trainer"

#: Callables and class attributes this driver may define. A set rather than a count, following the
#: conv-Transformer sibling's version of this assertion: a count passes a subclass that overrode
#: ``train_model`` in 75 lines while dropping the plot callback.
_OWN_ATTRIBUTES = {
    "MODEL_CLS", "TASK_CLS", "CHECKPOINT_STEM", "TARGET_FIELDS", "TRACKED_METRICS",
}


@pytest.fixture
def driver(tmp_path):
    """A driver on the shipped config, with its output directories redirected under ``tmp_path``.

    ``setup_config`` is never called -- it would seed, open log sinks and probe MLflow -- so the
    directories are assigned directly.
    """
    instance = LagAttnFsTrainer(config_file_path=str(_CONFIG))
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


class _StubDataModule:
    """A data module that hands out loaders nothing iterates."""

    def __init__(self, config=None):
        self.config = config

    def train_dataloader(self):
        return object()

    def val_dataloader(self):
        return object()


# --------------------------------------------------------------------------------------
# The class attributes, and what they decide
# --------------------------------------------------------------------------------------
def test_the_driver_names_this_model_and_not_the_one_it_is_compared_against():
    """A stale attribute here trains the other model under this model's config, tag and MLflow
    experiment, and nothing anywhere raises."""
    assert LagAttnFsTrainer.MODEL_CLS is SeqVaeLagAttnFs
    assert LagAttnFsTrainer.TASK_CLS is SeqVaeLagAttnFsTask
    assert LagAttnFsTrainer.CHECKPOINT_STEM == "lag-attn-fs"
    assert LagAttnFsTrainer.MODEL_CLS is not LagAttnRwsTrainer.MODEL_CLS
    assert LagAttnFsTrainer.CHECKPOINT_STEM != LagAttnRwsTrainer.CHECKPOINT_STEM


def test_the_inherited_driver_still_builds_the_model_it_always_did():
    """The attributes exist so this package can reuse the driver; the reuse is worthless if it
    changed what a launch of the comparison model produces."""
    assert LagAttnRwsTrainer.MODEL_CLS is SeqVaeLagAttnRws
    assert LagAttnRwsTrainer.CHECKPOINT_STEM == "lag-attn-rws"
    assert LagAttnRwsTrainer.TARGET_FIELDS == ("fhr",)
    assert LagAttnRwsTrainer.TRACKED_METRICS == _TRACKED_METRICS


def test_the_driver_declares_five_attributes_and_overrides_no_method():
    """Every method it could override is a piece of machinery the comparison rests on: the kwarg
    sweep, ``create_model``, the callback assembly, the DDP selection. Redefining any of them here
    would be a second copy free to drift from the one the comparison model runs under."""
    own = {name for name in vars(LagAttnFsTrainer) if not name.startswith("_")}

    assert own == _OWN_ATTRIBUTES


def test_the_target_fields_are_both_stored_blocks():
    """The one guard a target-domain change moves. Both blocks, because the target is their
    concatenation: a config carrying one of them is a target with a hole in it, and the guard
    checks field by field so the refusal names which."""
    assert LagAttnFsTrainer.TARGET_FIELDS == ("fhr_st", "fhr_ph")


def test_the_tracked_metrics_are_the_inherited_list_plus_this_models_four():
    """Both directions. A name the framework never emits is a CSV column that is NaN in every row
    of every run; a metric the task emits that is not here never reaches the CSV at all."""
    added = set(LagAttnFsTrainer.TRACKED_METRICS) - set(_TRACKED_METRICS)

    assert added == {
        f"{stage}/{name}"
        for stage in ("train", "val")
        for name in ("pred_gap_tau_first", "pred_gap_tau_last", "pred_gap_st", "pred_gap_ph")
    }
    assert set(_TRACKED_METRICS) - set(LagAttnFsTrainer.TRACKED_METRICS) == set()
    # No duplicates: the collector keys on the name, and a repeat would silently write one column.
    assert len(set(LagAttnFsTrainer.TRACKED_METRICS)) == len(LagAttnFsTrainer.TRACKED_METRICS)


# --------------------------------------------------------------------------------------
# Config to constructor
# --------------------------------------------------------------------------------------
def test_the_shipped_config_resolves_to_the_shipped_architecture(driver):
    """Every architectural flag ``SHIPPED_KWARGS`` claims the config sets, it must set. That
    fixture is the suite's description of the production model; this keeps it honest against the
    config file itself."""
    kwargs = driver._build_model_kwargs()

    for name in (
        "sequence_length", "d_model", "d_z", "horizon", "raw_per_step", "warmup_period",
        "c_y", "c_u", "use_up_st", "max_lag", "num_heads", "d_head", "lstm_layers",
        "decoder_hidden", "horizon_depth", "horizon_kernel", "horizon_film",
        "horizon_embed_std", "head_init_calibration", "a_head_gain", "encoder_extra_kernel",
        "use_entmax", "attention_grad_checkpoint", "lag_bias_init", "query_uses_logvar",
        "causal_norm", "coverage_floor",
    ):
        assert kwargs[name] == SHIPPED_KWARGS[name], f"{name} disagrees with the shipped flag set"
    # YAML has no tuple; the constructor coerces, so the sweep hands the list through.
    assert tuple(kwargs["encoder_extra_dilations"]) == SHIPPED_KWARGS["encoder_extra_dilations"]
    assert tuple(kwargs["logvar_clamp"]) == SHIPPED_KWARGS["logvar_clamp"]


def test_the_reach_budget_reaches_the_constructor_as_the_four_channel_tuples(driver):
    """Translated rather than forwarded, and here the translation also fixes the decoder's width --
    so a checkpoint recording only ``causal_reach_budget_s`` could not be rebuilt without re-running
    the filter bank."""
    kwargs = driver._build_model_kwargs()

    assert "causal_reach_budget_s" not in kwargs
    assert len(kwargs["target_keep_index"]) == len(kwargs["target_delays"]) == 78
    assert len(kwargs["source_keep_index"]) == len(kwargs["source_delays"]) == 29


def test_no_decoder_width_key_reaches_the_constructor(driver):
    """``decoder_out_channels`` is deliberately absent from the config: the width follows the gate,
    and a second field naming it could disagree with the target the run is actually scored on."""
    assert "decoder_out_channels" not in driver._build_model_kwargs()


def test_init_weights_is_never_a_config_decision(driver):
    """Skipping initialisation would also skip the post-init delta-head zeroing the zero-KL start
    depends on, and the output-head calibration that keeps the init NLL of 78 channels near the
    trivial predictor's."""
    driver.config["model_config"]["VAE_model"]["init_weights"] = False

    assert "init_weights" not in driver._build_model_kwargs()


def test_the_resolved_kwargs_actually_build_the_model_the_config_describes(driver):
    """The sweep's output is only correct if the constructor accepts it -- and, here, if the model
    it produces decodes the width the reach budget kept."""
    model = SeqVaeLagAttnFs(**driver._build_model_kwargs())

    assert isinstance(model, SeqVaeLagAttnFs)
    assert model.decoder_out_channels == 78
    assert model.raw_per_step == 16
    # The unconditional freeze the DDP strategy relies on.
    assert not any(parameter.requires_grad for parameter in model.lag_attn.W_o.parameters())


# --------------------------------------------------------------------------------------
# create_model
# --------------------------------------------------------------------------------------
def test_create_model_builds_this_net_and_wraps_it_in_this_task(driver):
    driver.create_model()

    assert isinstance(driver.pytorch_model, SeqVaeLagAttnFs)
    assert isinstance(driver.pl_model, SeqVaeLagAttnFsTask)
    assert driver.pl_model.orig_model is driver.pytorch_model


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
    """The two weights the calibration sweep selects have to arrive at the objective, or the run
    trains one balance while its config states another.

    Read from the shipped config rather than restated. What this asserts is the *forwarding*, and a
    literal here would turn every retune of the pair into a failure in an unrelated driver test --
    which is a second place for the number to live and a second place for it to go stale. The
    weights' values are pinned in ``test_config_load.py``, where the reasons for them are."""
    driver.create_model()

    shipped_vae = load_config(str(_CONFIG))["model_config"]["VAE_model"]
    hparams = driver.pl_model.hparams
    assert hparams["likelihood"] == "gaussian_nll"
    assert hparams["lambda_full"] == 1.0
    assert hparams["lambda_base"] == 1.0
    assert hparams["free_bits"] == 0.0
    assert hparams["beta_schedule"] == shipped_vae["beta_schedule"]
    assert hparams["beta_prior"] == shipped_vae["beta_prior"]
    # The two are a pair: the anchor's restoring force saturates at beta_prior/2 per latent
    # dimension, so what the design fixes is their ratio and not either value.
    assert hparams["beta_prior"] / hparams["beta_schedule"]["end"] == pytest.approx(0.1)


def test_the_checkpoint_kwargs_are_the_ones_the_model_was_built_from(driver):
    """So the blob rebuilds into this architecture at this budget's width, and not into the
    constructor's defaults."""
    driver.create_model()

    assert driver.pl_model._model_kwargs == driver._build_model_kwargs()


def test_a_core_checkpoint_from_the_comparison_model_is_refused_before_it_is_loaded(
    driver, tmp_path
):
    """The two models share every tensor name but the decoder head's, so the class stamp is what
    turns a partial-alignment failure into a message naming the model that wrote the blob."""
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
# The entry point and its guards
# --------------------------------------------------------------------------------------
@pytest.fixture
def recording_main(monkeypatch):
    """Run ``main`` with every expensive step recorded rather than done.

    Patched on the **base** driver rather than on this one, so a test that drives the shared entry
    point with the comparison model's driver is stubbed too. Patching the subclass would leave that
    launch running a real fit against the committed shard, which is the difference between a
    two-second test and a wrong one.
    """
    calls = []

    def _record(name, result=None):
        def _recorder(self, *args, **kwargs):
            calls.append(name)
            return result

        return _recorder

    monkeypatch.setattr(LagAttnRwsTrainer, "setup_config", _record("setup_config"))
    monkeypatch.setattr(LagAttnRwsTrainer, "create_model", _record("create_model"))
    monkeypatch.setattr(LagAttnRwsTrainer, "train_model", _record("train_model"))
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

    monkeypatch.setattr(LagAttnFsTrainer, "__init__", _capture_init)

    with pytest.raises(RuntimeError, match="stop here"):
        trainer_module.main(_tiny_config_at(tmp_path))

    assert seen["cls"] is LagAttnFsTrainer


def test_setup_config_runs_before_the_model_is_built(recording_main, tmp_path):
    """The order that decides whether a run is seeded, logged and tracked at all: building the
    model first means no seeding, no log sinks, nowhere to write, and ``mlflow_logger is None`` --
    which silently drops the MLflow callback from the fit."""
    trainer_module.main(_tiny_config_at(tmp_path))

    assert recording_main == ["setup_config", "create_model", "train_model"]


def test_all_four_pre_flight_guards_and_the_driver_hook_run_before_setup_config(
    tmp_path, monkeypatch
):
    """Their whole value is failing before the run directory and MLflow run exist on every rank of
    a multi-rank launch."""
    order = []
    monkeypatch.setattr(
        LagAttnFsTrainer, "setup_config", lambda self: order.append("setup_config")
    )
    for attribute, label in (
        ("_check_stat_path", "stat_path"),
        ("_check_declared_widths_against_shard", "widths"),
        ("_check_raw_target_normalized", "target_normalized"),
        ("_check_causal_budget_resolves", "causal_budget"),
    ):
        monkeypatch.setattr(
            # ``**_`` because the normalisation guard is handed the driver's TARGET_FIELDS; these
            # stubs record the order and have no opinion about any guard's arguments.
            shared_trainer, attribute, lambda config, _label=label, **_: order.append(_label)
        )
    monkeypatch.setattr(shared_trainer, "GraphDataModule", lambda config: None)
    monkeypatch.setattr(
        LagAttnFsTrainer, "create_model", lambda self: order.append("create_model")
    )

    with pytest.raises(AttributeError):
        # GraphDataModule is stubbed to None, so main dies at train_dataloader() -- after the part
        # under test. The order up to that point is the assertion.
        trainer_module.main(_tiny_config_at(tmp_path))

    assert order[:5] == [
        "stat_path", "widths", "target_normalized", "causal_budget", "setup_config",
    ], order


def test_the_normalisation_guard_is_handed_this_drivers_target_fields(tmp_path, monkeypatch):
    """The plumbing a ``trainer_cls=`` wiring mistake breaks, checked by value rather than by
    behaviour: a guard wired to the wrong driver would still run and still refuse *something*."""
    seen = {}
    monkeypatch.setattr(
        shared_trainer,
        "_check_raw_target_normalized",
        lambda config, **kwargs: seen.update(kwargs),
    )
    monkeypatch.setattr(LagAttnFsTrainer, "setup_config", lambda self: None)
    monkeypatch.setattr(shared_trainer, "GraphDataModule", lambda config: None)

    with pytest.raises(AttributeError):
        trainer_module.main(_tiny_config_at(tmp_path))

    assert seen == {"fields": ("fhr_st", "fhr_ph")}


@pytest.mark.parametrize("field", ["fhr_st", "fhr_ph"])
@pytest.mark.parametrize("list_key", ["load_fields", "normalize_fields"])
def test_a_missing_target_block_raises_naming_the_field_and_the_list(
    recording_main, tmp_path, list_key, field
):
    """Without both blocks in ``normalize_fields`` the target arrives at its stored scale, the
    Gaussian NLL is meaningless, and nothing else raises -- a full run trained on nothing. Both
    fields, because "one of them is missing" is not a fix anybody can act on."""

    def _drop(config):
        dataloader = config["dataset_config"]["dataloader_config"]
        fields = (
            dataloader["dataset_kwargs"]["load_fields"]
            if list_key == "load_fields"
            else dataloader[list_key]
        )
        fields.remove(field)

    with pytest.raises(ValueError, match=rf"'{field}' in .*{list_key}"):
        trainer_module.main(_tiny_config_at(tmp_path, _drop))

    assert "create_model" not in recording_main


def test_a_config_that_only_normalizes_the_raw_signal_is_refused_here_and_not_there(
    recording_main, tmp_path
):
    """One config, normalising ``fhr`` but not ``fhr_st``: the raw-target driver passes it and this
    one must not. Both directions, because a guard wired to the wrong class would still refuse
    *something* and a one-sided test would not say which."""

    def _drop_fhr_st(config):
        config["dataset_config"]["dataloader_config"]["normalize_fields"].remove("fhr_st")

    config_path = _tiny_config_at(tmp_path, _drop_fhr_st)

    shared_trainer.main(config_path, trainer_cls=LagAttnRwsTrainer)
    assert "create_model" in recording_main

    with pytest.raises(ValueError, match=r"'fhr_st'"):
        trainer_module.main(config_path)


def test_an_unsatisfiable_reach_budget_raises_before_any_training_happens(
    recording_main, tmp_path
):
    """A budget whose delay outruns ``warmup_period`` would zero-fill trained anchors -- and here
    it would also silently re-size the decoder, since the width follows the survivors."""

    def _too_deep(config):
        config["model_config"]["VAE_model"]["causal_reach_budget_s"] = 240.0

    with pytest.raises(ValueError, match="warmup_period"):
        trainer_module.main(_tiny_config_at(tmp_path, _too_deep))

    assert "create_model" not in recording_main


def test_the_resolved_config_is_written_beside_the_checkpoints(tmp_path, monkeypatch):
    """A run's own config is otherwise recoverable only from the text of its log or from an MLflow
    artifact whose on-disk location nothing can derive. It matters more here than for the raw
    sibling: the recorded budget is what says how wide the decoder was, and therefore what the
    run's nats were summed over."""
    monkeypatch.setattr(LagAttnFsTrainer, "create_model", lambda self: None)
    monkeypatch.setattr(LagAttnFsTrainer, "train_model", lambda self, *args: None)
    monkeypatch.setattr(shared_trainer, "GraphDataModule", lambda config: _StubDataModule())

    captured = {}

    def _remember(self):
        # setup_config is what creates the run directories, so the write must follow it; the real
        # one runs, and only the directory it chose is recorded.
        GraphModelBase.setup_config(self)
        captured["checkpoint_dir"] = self.model_checkpoint_dir

    monkeypatch.setattr(LagAttnFsTrainer, "setup_config", _remember)

    trainer_module.main(_tiny_config_at(tmp_path))

    written = Path(captured["checkpoint_dir"]) / shared_trainer.RESOLVED_CONFIG_FILENAME
    assert written.is_file()
    reloaded = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert "base" not in reloaded
    # Read from the shipped config rather than restated: this assertion is about the resolution
    # *reaching* the written record, not about which weight the sweep chose, and a literal here
    # would make an unrelated retune fail a provenance test.
    shipped = load_config(str(_CONFIG))
    assert (
        reloaded["model_config"]["VAE_model"]["beta_schedule"]["end"]
        == shipped["model_config"]["VAE_model"]["beta_schedule"]["end"]
    )
    record = reloaded["model_config"][shared_trainer.RESOLVED_BUDGET_KEY]
    assert record is not None
    assert record["causal_reach_budget_s"] == 120
    assert len(record["target_keep_index"]) == 78


# --------------------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------------------
def test_relative_config_paths_resolve_against_the_repository_root():
    """An IDE's working directory is arbitrary; every documented invocation is repo-root relative,
    so the resolver must anchor there and leave absolute paths alone."""
    resolved = trainer_module._resolve_cli_config_path("teb_vae/lag_attn_fs/configs/tiny.yaml")

    assert Path(resolved) == _TINY
    absolute = str(_TINY)
    assert trainer_module._resolve_cli_config_path(absolute) == absolute


def test_run_config_points_at_a_config_that_exists():
    """The IDE Run button resolves through ``RUN_CONFIG``; a stale path breaks it silently."""
    assert trainer_module.RUN_CONFIG is not None
    assert (_REPO_ROOT / trainer_module.RUN_CONFIG).is_file()


def test_the_argument_is_optional_so_the_run_button_works_at_all():
    """``required=True`` fires before ``RUN_CONFIG`` is ever consulted, which would make the Run
    button unusable no matter what the constant said -- and the refusal, when there is nothing to
    fall back on, has to name the constant rather than only the flag."""
    source = Path(trainer_module.__file__).read_text(encoding="utf-8")

    assert "required=True" not in source
    assert "RUN_CONFIG" in source.split("parser.error(")[1]


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
