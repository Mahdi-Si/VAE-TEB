r"""The driver turns config into this model, and the entry point runs the guards before anything.

Almost all of this module's behaviour is inherited two levels deep, which is the design and also the
risk: three class attributes are the entire difference between building this architecture and
building the one it is compared against, and a stale one produces a run that trains the wrong model,
wraps it in the wrong task, or writes its checkpoints under the wrong stem -- none of which raises.

The pre-flight guards get their own assertions for the same reason they exist: their whole value is
failing *before* the run directory, the log sinks and the MLflow run exist on every rank of a
multi-rank launch. Both of the ones this package adds guard a **silent** failure -- a config key the
signature sweep drops without a word, a raw field the loader hands over unnormalized, a shard whose
geometry disagrees with the model's -- so a guard that stopped running would only be noticed by the
run it failed to protect.

The startup log gets its own assertions too, which is unusual and deliberate. The inherited sentence
says the model's input features read into their own future; that is the negation of this package's
central claim, and left in place it would appear in every production run's log beside numbers that
contradict it.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pytest
import torch
import yaml
from loguru import logger

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws import trainer as shared_trainer
from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer
from teb_vae.lag_attn_transformer_e2e import trainer as trainer_module
from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from teb_vae.lag_attn_transformer_e2e.task import SeqVaeLagAttnTrfE2ETask
from teb_vae.lag_attn_transformer_e2e.trainer import (
    INERT_MODEL_KEYS,
    LagAttnTrfE2ETrainer,
)
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer
from train.graph_model_base import GraphModelBase

from .conftest import SHIPPED_KWARGS, absolutize_dataset_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"
_MODULE_NAME = "teb_vae.lag_attn_transformer_e2e.trainer"

#: The sentence the inherited driver logs when no reach budget is configured. It is false for this
#: architecture and is the negation of its central claim, so its *absence* is asserted rather than
#: assumed. A substring rather than the whole message, so a rewording of the sibling's text does not
#: turn this into a test of that package's prose.
INHERITED_FEATURE_REACH_PHRASE = "into their own future"


@pytest.fixture
def driver(tmp_path):
    """A driver on the shipped config, with its output directories redirected under ``tmp_path``.

    ``setup_config`` is never called -- it would seed, open log sinks and probe MLflow -- so the
    directories are assigned directly.
    """
    instance = LagAttnTrfE2ETrainer(config_file_path=str(_CONFIG))
    instance.output_base_dir = str(tmp_path)
    instance.train_results_dir = str(tmp_path / "train_results")
    instance.model_checkpoint_dir = str(tmp_path / "model_checkpoints")
    return instance


@pytest.fixture
def logged():
    """Collect everything logged at INFO or above, for the duration of one test.

    Not ``caplog``: loguru does not route through the standard library's ``logging``, so a
    ``caplog.at_level`` assertion against these lines passes on a driver that logs nothing at all.
    """
    messages = []
    sink_id = logger.add(messages.append, level="INFO", format="{message}")
    yield messages
    logger.remove(sink_id)


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


def _shard_with_raw_length(tmp_path, raw_length: int, *, field: str = "fhr") -> str:
    """Write a one-sample HDF5 carrying a raw field of the requested stored length.

    A constructed shard rather than the committed one, because the case worth testing -- a stored
    length that does not trim to the model's geometry -- cannot be produced from a fixture that is
    correct by construction.

    Args:
        tmp_path: Directory to write into.
        raw_length: Stored, untrimmed length of the raw field.
        field: Name to store it under; a non-``fhr`` name produces the field-less shard case.

    Returns:
        The written path.
    """
    path = tmp_path / f"shard_{raw_length}_{field}.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(field, data=np.zeros((1, raw_length), dtype=np.float32))
    return str(path)


def _config_for_shard(
    shard_path: str, *, trim_minutes: Optional[float] = 1.0, **vae_overrides
) -> dict:
    """A minimal config carrying only what the raw-length guard reads."""
    vae = dict(sequence_length=300, raw_per_step=16)
    vae.update(vae_overrides)
    return {
        "model_config": {"VAE_model": vae},
        "dataset_config": {
            "vae_train_datasets": [shard_path],
            "dataloader_config": {"dataset_kwargs": {"trim_minutes": trim_minutes}},
        },
    }


# --------------------------------------------------------------------------------------
# The three class attributes, and what they decide
# --------------------------------------------------------------------------------------
def test_the_driver_names_this_architecture_and_not_either_sibling():
    """A stale attribute here trains another model under this model's config, tag and MLflow
    experiment, and nothing anywhere raises."""
    assert LagAttnTrfE2ETrainer.MODEL_CLS is SeqVaeLagAttnTrfE2E
    assert LagAttnTrfE2ETrainer.TASK_CLS is SeqVaeLagAttnTrfE2ETask
    assert LagAttnTrfE2ETrainer.CHECKPOINT_STEM == "lag-attn-trf-e2e"
    for sibling in (LagAttnRwsTrainer, LagAttnTrfRwsTrainer):
        assert LagAttnTrfE2ETrainer.MODEL_CLS is not sibling.MODEL_CLS
        assert LagAttnTrfE2ETrainer.CHECKPOINT_STEM != sibling.CHECKPOINT_STEM


def test_the_inherited_drivers_still_build_the_models_they_always_did():
    """The class attributes were introduced so packages like this could reuse the driver; the reuse
    is worthless if it changed what a launch of either sibling produces."""
    assert LagAttnTrfRwsTrainer.MODEL_CLS is SeqVaeLagAttnTrfRws
    assert LagAttnTrfRwsTrainer.CHECKPOINT_STEM == "lag-attn-trf-rws"


def test_the_step_warmup_machinery_is_inherited_rather_than_restated():
    """Three pieces of the conv-Transformer driver stay meaningful here and are deliberately not
    re-pointed: the nullable-key re-admission (an unbounded source encoder *is*
    ``source_attention_window: null``, and the inherited sweep would drop it), the step-granular
    learning-rate monitor, and the DDP strategy selection, which keys on ``likelihood`` alone."""
    from teb_vae.lag_attn_transformer_rws import trainer as sibling_trainer

    for method in ("_build_model_kwargs", "_build_trainer_kwargs", "create_model"):
        assert method not in vars(LagAttnTrfE2ETrainer), f"{method} is re-implemented here"
    assert "select_ddp_strategy" not in vars(LagAttnTrfE2ETrainer)
    assert sibling_trainer.NULLABLE_MODEL_KEYS == frozenset({"source_attention_window"})


def test_the_shipped_config_resolves_to_the_shipped_architecture(driver):
    """Every architectural flag ``SHIPPED_KWARGS`` claims the config sets, it must set. That
    fixture is the suite's description of the production model; this keeps it honest against the
    config file itself."""
    kwargs = driver._build_model_kwargs()

    for name in (
        "lag_bias_init", "use_entmax", "horizon_depth", "horizon_kernel", "horizon_film",
        "encoder_num_heads", "encoder_d_ff", "target_attention_blocks",
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
    assert kwargs["max_lag"] == 90
    assert kwargs["coverage_floor"] == 0.9


def test_no_front_end_key_reaches_the_constructor_from_config(driver):
    """The stage widths are derived from ``d_model`` and the kernels are a module constant, so
    there is nothing to configure -- and therefore nothing for a signature sweep to drop in
    silence, which is the failure a config key here would create."""
    kwargs = driver._build_model_kwargs()

    assert "frontend_kernels" not in kwargs
    assert not [name for name in kwargs if name.startswith("frontend")]


def test_loss_only_keys_do_not_reach_the_constructor(driver):
    """The net takes tensors and computes a loss on request; it owns none of these. The
    constructor is keyword-only with no ``**kwargs``, so a leaked key would be a ``TypeError`` on
    the production config -- a poor place to find out."""
    kwargs = driver._build_model_kwargs()

    for name in (
        "likelihood", "free_bits", "lambda_full", "lambda_base", "beta_schedule",
        "kld_beta", "beta_prior",
    ):
        assert name not in kwargs, f"{name} is not the net's"


def test_the_resolved_kwargs_actually_build_a_model(driver):
    """The sweep's output is only correct if the constructor accepts it."""
    model = SeqVaeLagAttnTrfE2E(**driver._build_model_kwargs())

    assert isinstance(model, SeqVaeLagAttnTrfE2E)
    # The unconditional freeze the DDP strategy relies on.
    assert not any(parameter.requires_grad for parameter in model.lag_attn.W_o.parameters())
    # The depthwise correction actually ran, over the stem AND the two front ends: four stem
    # convolutions plus two per stride-2 stage per stream. A count of four would mean the front
    # ends' convolutions were missed and are training an order of magnitude too quietly.
    assert model.n_depthwise_init == 12


def test_the_reach_budget_the_model_builds_its_front_ends_at_is_the_geometrys(driver):
    """Not a configuration key: ``warmup_period * raw_per_step``. An anchor outside the warm-up
    that reached further would be trained against the zero-padded convolution transient at the
    segment's start."""
    model = SeqVaeLagAttnTrfE2E(**driver._build_model_kwargs())

    assert model.frontend_reach_budget == 30 * 16
    assert model.target_frontend.reach_budget == model.frontend_reach_budget
    assert model.target_frontend.reach_samples < model.frontend_reach_budget


def test_init_weights_is_never_a_config_decision(driver):
    """Skipping initialisation would also skip the depthwise correction and the post-init
    delta-head zeroing the zero-KL start depends on; the key is refused even when a config
    supplies it."""
    driver.config["model_config"]["VAE_model"]["init_weights"] = False

    assert "init_weights" not in driver._build_model_kwargs()


# --------------------------------------------------------------------------------------
# create_model, and what the startup log says about it
# --------------------------------------------------------------------------------------
def test_create_model_builds_this_net_and_wraps_it_in_this_task(driver):
    driver.create_model()

    assert isinstance(driver.pytorch_model, SeqVaeLagAttnTrfE2E)
    assert isinstance(driver.pl_model, SeqVaeLagAttnTrfE2ETask)
    assert driver.pl_model.orig_model is driver.pytorch_model


def test_create_model_does_not_trip_on_the_absent_causal_norm_flag(driver):
    """The inherited ``create_model`` reads ``causal_norm`` off the net through a ``getattr`` with
    a safe default. There is no time-pooling normaliser in this architecture's history path to
    causalise, so the attribute does not exist and a bare read would be an ``AttributeError`` on
    the first launch."""
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
    """The objective is shared with both siblings, so these values *are* the comparison."""
    driver.create_model()

    hparams = driver.pl_model.hparams
    assert hparams["likelihood"] == "gaussian_nll"
    assert hparams["lambda_full"] == 1.0
    assert hparams["lambda_base"] == 1.0
    assert hparams["free_bits"] == 0.0
    assert hparams["beta_prior"] == 0.1
    assert hparams["beta_schedule"]["kind"] == "linear_warmup"
    assert hparams["beta_schedule"]["start"] == 0.0


def test_create_model_forwards_the_step_warmup(driver):
    """Inherited from the conv-Transformer driver; it reaches ``hparams`` and therefore every
    checkpoint, which is where the task reads it back from."""
    driver.create_model()

    assert int(driver.pl_model.hparams["lr_warmup_steps"]) == 2000


def test_the_checkpoint_kwargs_are_the_ones_the_model_was_built_from(driver):
    """So the blob rebuilds into this architecture and not the constructor's defaults."""
    driver.create_model()

    assert driver.pl_model._model_kwargs == driver._build_model_kwargs()


def test_the_startup_log_states_this_models_measured_reach(driver, logged):
    """A run's log should say what its history states are a function of, because that is the
    premise every coupling number it produces rests on."""
    driver.create_model()

    message = next(line for line in logged if "causal standing" in line)
    reach = driver.pytorch_model.target_frontend.reach_samples
    assert f"{reach} raw samples" in message
    assert "80.5 s" in message  # the reach in seconds, at 4 Hz
    assert f"budget of {driver.pytorch_model.frontend_reach_budget}" in message


def test_the_logged_reach_cannot_drift_from_the_front_end_that_was_built(driver):
    """The message is composed *before* ``pytorch_model`` is assigned -- deliberately, so a launch
    that dies in the constructor has still said what it was about to build -- which means it is
    measured on a throwaway front end. This is what stops that second construction disagreeing
    with the real one."""
    message = driver.causal_standing_message()
    driver.create_model()

    for frontend in (driver.pytorch_model.target_frontend, driver.pytorch_model.source_frontend):
        assert f"{frontend.reach_samples} raw samples" in message
        assert f"budget of {frontend.reach_budget}" in message


def test_the_startup_log_does_not_repeat_the_inherited_feature_reach_sentence(driver, logged):
    """It is the negation of this package's central claim, and it would otherwise appear in every
    production run's log beside numbers that contradict it."""
    driver.create_model()

    offenders = [line for line in logged if INHERITED_FEATURE_REACH_PHRASE in line]
    assert offenders == [], offenders


def test_the_siblings_message_is_untouched(tmp_path):
    """The replacement is this driver's, not a change to the one it inherits from: the comparison
    model's log must keep stating the standing that is true of *it*."""
    sibling_config = (
        _REPO_ROOT / "teb_vae" / "lag_attn_transformer_rws" / "configs" / "default.yaml"
    )
    instance = LagAttnTrfRwsTrainer(config_file_path=str(sibling_config))
    instance._build_model_kwargs()  # populates resolved_budget, which the message branches on

    assert INHERITED_FEATURE_REACH_PHRASE in instance.causal_standing_message()


def test_a_core_checkpoint_from_a_sibling_is_refused_before_it_is_loaded(driver, tmp_path):
    """All three models share every tensor below the encoder inputs, so a blob from another one
    would partially align by accident and train from a mixture of loaded and random weights."""
    foreign = tmp_path / "foreign.ckpt"
    torch.save({"state_dict": {}, "model_class": "SeqVaeLagAttnTrfRws"}, foreign)
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
# The entry point and the pre-flight ordering
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

    monkeypatch.setattr(LagAttnTrfE2ETrainer, "setup_config", _record("setup_config"))
    monkeypatch.setattr(LagAttnTrfE2ETrainer, "create_model", _record("create_model"))
    monkeypatch.setattr(LagAttnTrfE2ETrainer, "train_model", _record("train_model"))
    monkeypatch.setattr(shared_trainer, "GraphDataModule", _StubDataModule)
    return calls


def test_the_entry_point_constructs_this_packages_driver(monkeypatch, tmp_path):
    """``main`` delegates to the shared entry point, which owns the four guards, the temporary
    resolved-config file and the resolved-config write. What this package supplies is which driver
    it constructs -- and that is the one thing a delegation can get wrong while still running."""
    seen = {}

    def _capture_init(self, config_file_path=None):
        seen["cls"] = type(self)
        raise RuntimeError("stop here")

    monkeypatch.setattr(LagAttnTrfE2ETrainer, "__init__", _capture_init)

    with pytest.raises(RuntimeError, match="stop here"):
        trainer_module.main(_tiny_config_at(tmp_path))

    assert seen["cls"] is LagAttnTrfE2ETrainer


def test_setup_config_runs_before_the_model_is_built(recording_main, tmp_path):
    """The order that decides whether a run is seeded, logged and tracked at all: building the
    model first means no seeding, no log sinks, nowhere to write, and ``mlflow_logger is None`` --
    which silently drops the MLflow callback from the fit."""
    trainer_module.main(_tiny_config_at(tmp_path))

    assert recording_main == ["setup_config", "create_model", "train_model"]


def test_this_drivers_preflight_runs_after_the_four_guards_and_before_setup_config(
    tmp_path, monkeypatch
):
    """Their whole value is failing before the run directory and MLflow run exist on every rank of
    a multi-rank launch, and this package's own guards must keep that guarantee."""
    order = []
    monkeypatch.setattr(
        LagAttnTrfE2ETrainer, "setup_config", lambda self: order.append("setup_config")
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
    monkeypatch.setattr(
        LagAttnTrfE2ETrainer,
        "preflight",
        classmethod(lambda cls, config: order.append("preflight")),
    )
    monkeypatch.setattr(shared_trainer, "GraphDataModule", lambda config: None)
    monkeypatch.setattr(
        LagAttnTrfE2ETrainer, "create_model", lambda self: order.append("create_model")
    )

    with pytest.raises(AttributeError):
        # GraphDataModule is stubbed to None, so main dies at train_dataloader() -- after the part
        # under test. The order up to that point is the assertion.
        trainer_module.main(_tiny_config_at(tmp_path))

    assert order[:6] == [
        "stat_path", "widths", "fhr_normalized", "causal_budget", "preflight", "setup_config",
    ], order


def test_the_shipped_configs_pass_this_drivers_own_preflight():
    """Both of them, so a guard cannot be satisfied by the smoke config alone. Called directly
    rather than through ``main``: the production config's shard paths do not exist on this box,
    which the shard half treats as non-fatal and the inherited ``_check_stat_path`` does not."""
    for path in (_CONFIG, _TINY):
        LagAttnTrfE2ETrainer.preflight(load_config(str(path)))


# --------------------------------------------------------------------------------------
# Pre-flight: the inert keys
# --------------------------------------------------------------------------------------
def test_the_inert_key_set_is_exactly_the_input_representation_being_replaced():
    """Declared as a mapping from the key to what took its place, so the refusal can say both."""
    assert set(INERT_MODEL_KEYS) == {
        "c_y", "c_u", "use_up_st", "causal_reach_budget_s",
        "target_keep_index", "target_delays", "source_keep_index", "source_delays",
    }


@pytest.mark.parametrize("key", sorted(INERT_MODEL_KEYS))
def test_each_inert_key_is_refused_by_name(recording_main, tmp_path, key):
    """The signature sweep drops each of these without a word, so the run would be a different one
    from the one the operator configured -- most damagingly ``causal_reach_budget_s``, whose whole
    purpose was to bound a leak this architecture does not have."""

    def _add_key(config):
        config["model_config"]["VAE_model"][key] = 1

    with pytest.raises(ValueError, match=key):
        trainer_module.main(_tiny_config_at(tmp_path, _add_key))

    assert "create_model" not in recording_main


def test_the_refusal_names_what_replaced_the_key(recording_main, tmp_path):
    """A refusal that only said "unknown key" would leave an operator to guess whether the key was
    dropped, renamed or moved."""

    def _copy_paste_the_siblings_widths(config):
        config["model_config"]["VAE_model"].update(c_y=109, c_u=58, use_up_st=True)

    with pytest.raises(ValueError) as excinfo:
        trainer_module.main(_tiny_config_at(tmp_path, _copy_paste_the_siblings_widths))

    message = str(excinfo.value)
    assert "3 key(s)" in message
    for key in ("c_y", "c_u", "use_up_st"):
        assert key in message
    assert "front end" in message


def test_the_inherited_width_guard_stays_silent_on_this_packages_configs():
    """It returns early unless the config carries ``c_y`` *and* ``c_u``, and a config carrying
    either is refused above -- which is what makes the inert-key message the one an operator who
    copy-pasted a sibling config actually sees, rather than a channel-width message about a model
    with no channels."""
    shared_trainer._check_declared_widths_against_shard(load_config(str(_TINY)))


# --------------------------------------------------------------------------------------
# Pre-flight: the raw source
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("list_key", ["load_fields", "normalize_fields"])
def test_a_missing_raw_source_field_raises_naming_the_offending_list(
    recording_main, tmp_path, list_key
):
    """Without 'up' in ``load_fields`` the source stream does not exist; without it in
    ``normalize_fields`` nothing fails at all -- the front end owns no statistics of its own, so
    every coupling number the run reports is measured at an operating point nobody chose."""

    def _drop_up(config):
        dataloader = config["dataset_config"]["dataloader_config"]
        fields = (
            dataloader["dataset_kwargs"]["load_fields"]
            if list_key == "load_fields"
            else dataloader[list_key]
        )
        fields.remove("up")

    with pytest.raises(ValueError, match=list_key):
        trainer_module.main(_tiny_config_at(tmp_path, _drop_up))

    assert "create_model" not in recording_main


@pytest.mark.parametrize("list_key", ["load_fields", "normalize_fields"])
def test_the_inherited_guard_still_covers_the_raw_target(recording_main, tmp_path, list_key):
    """``fhr`` is both the reconstruction target and the target stream's input here, so it is
    covered twice over -- but by the *inherited* guard, which is why this package's own adds only
    ``up``."""

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


# --------------------------------------------------------------------------------------
# Pre-flight: the shard's raw length
# --------------------------------------------------------------------------------------
def test_the_committed_shards_trimmed_length_is_the_models_geometry(tmp_path):
    """The passing case, on the real fixture: $5280 - 2 \\cdot 240 = 4800 = 300 \\cdot 16$. A guard
    that compared the *stored* length would fail on every real shard."""
    config = absolutize_dataset_paths(load_config(str(_TINY)))

    trainer_module._check_raw_length_against_shard(config)


def test_a_shard_whose_trimmed_length_misses_the_geometry_is_refused(tmp_path):
    """Which is what a ``trim_minutes`` that disagrees with ``sequence_length`` produces, and it
    would otherwise surface inside the first ``training_step``."""
    config = _config_for_shard(_shard_with_raw_length(tmp_path, 5120))

    with pytest.raises(ValueError) as excinfo:
        trainer_module._check_raw_length_against_shard(config)

    message = str(excinfo.value)
    assert "5120" in message  # the stored length
    assert "trim_minutes=1.0" in message  # the trim it was read at
    assert "4640" in message  # what that trims to
    assert "4800" in message  # what the model is built for


def test_the_same_shard_passes_at_the_geometry_it_actually_carries(tmp_path):
    """So the refusal above is about the *pair* -- a stored length, a trim and a geometry -- rather
    than about the file, which is what makes its message actionable."""
    shard = _shard_with_raw_length(tmp_path, 5120)

    trainer_module._check_raw_length_against_shard(
        _config_for_shard(shard, trim_minutes=None, sequence_length=320, raw_per_step=16)
    )


@pytest.mark.parametrize(
    "case",
    ["missing_file", "missing_field", "no_shards", "no_geometry"],
    ids=["a shard that is not there", "a shard without fhr", "no shards configured",
         "no geometry configured"],
)
def test_the_shard_guard_is_non_fatal_on_anything_but_a_mismatch(tmp_path, case):
    """The data module reports a missing, unreadable or field-less shard far better than a
    pre-flight peek can, and a guard that refused on them would turn every such case into a message
    about raw lengths."""
    if case == "missing_file":
        config = _config_for_shard(str(tmp_path / "absent.hdf5"))
    elif case == "missing_field":
        config = _config_for_shard(_shard_with_raw_length(tmp_path, 5120, field="up"))
    elif case == "no_shards":
        config = _config_for_shard(_shard_with_raw_length(tmp_path, 5120))
        config["dataset_config"]["vae_train_datasets"] = []
    else:
        config = _config_for_shard(_shard_with_raw_length(tmp_path, 5120))
        del config["model_config"]["VAE_model"]["sequence_length"]

    trainer_module._check_raw_length_against_shard(config)


def test_a_geometry_mismatch_stops_a_launch_before_anything_is_built(
    recording_main, tmp_path
):
    """Through ``main``, so the guard is shown to be reached rather than merely to work when
    called."""

    def _wrong_trim(config):
        config["dataset_config"]["dataloader_config"]["dataset_kwargs"]["trim_minutes"] = 2.0

    with pytest.raises(ValueError, match="raw samples per segment"):
        trainer_module.main(_tiny_config_at(tmp_path, _wrong_trim))

    assert "create_model" not in recording_main


# --------------------------------------------------------------------------------------
# The resolved config
# --------------------------------------------------------------------------------------
def test_the_resolved_config_is_written_beside_the_checkpoints(tmp_path, monkeypatch):
    """A run's own config is otherwise recoverable only from the text of its log or from an MLflow
    artifact whose on-disk location nothing can derive, and neither is a file a later offline pass
    can open."""
    monkeypatch.setattr(LagAttnTrfE2ETrainer, "create_model", lambda self: None)
    monkeypatch.setattr(LagAttnTrfE2ETrainer, "train_model", lambda self, *args: None)
    monkeypatch.setattr(shared_trainer, "GraphDataModule", lambda config: _StubDataModule())

    captured = {}

    def _remember(self):
        # setup_config is what creates the run directories, so the write must follow it; the real
        # one runs, and only the directory it chose is recorded.
        GraphModelBase.setup_config(self)
        captured["checkpoint_dir"] = self.model_checkpoint_dir

    monkeypatch.setattr(LagAttnTrfE2ETrainer, "setup_config", _remember)

    trainer_module.main(_tiny_config_at(tmp_path))

    written = Path(captured["checkpoint_dir"]) / shared_trainer.RESOLVED_CONFIG_FILENAME
    assert written.is_file()
    reloaded = yaml.safe_load(written.read_text(encoding="utf-8"))
    # Fully resolved: the inherited keys are present and the `base:` pointer is gone.
    assert "base" not in reloaded
    assert reloaded["model_config"]["VAE_model"]["target_attention_blocks"] == 4
    # The inherited record of the reach guard. Always null here -- the key it records is refused --
    # so it says explicitly that this run had no channel guard, which is a different statement from
    # a run written before the record existed. The front end's own reach is not in this file; it is
    # in the startup log and pinned in tests/test_frontend_reach.py.
    assert reloaded["model_config"][shared_trainer.RESOLVED_BUDGET_KEY] is None
    assert "causal_reach_budget_s" not in reloaded["model_config"]["VAE_model"]


# --------------------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------------------
def test_relative_config_paths_resolve_against_the_repository_root():
    """An IDE's working directory is arbitrary; every documented invocation is repo-root relative,
    so the resolver must anchor there and leave absolute paths alone."""
    resolved = trainer_module._resolve_cli_config_path(
        "teb_vae/lag_attn_transformer_e2e/configs/tiny.yaml"
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


def test_the_config_argument_is_not_argparse_required():
    """``required=True`` fires before ``RUN_CONFIG`` is ever read, which makes the Run button
    unusable no matter what the constant says. The refusal happens after the fallback instead, and
    its message names both ways to supply the value."""
    source = Path(trainer_module.__file__).read_text(encoding="utf-8")

    assert "required=True" not in source
    assert "default=None" in source
    assert "RUN_CONFIG" in source
    assert "--config is required" in source


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
