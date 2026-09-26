r"""The driver turns config into this model, and the entry point runs this package's guards first.

Almost all of the driver's behaviour is inherited two levels deep and tested where it is defined.
What is this package's: the class attributes that make a launch build *this* net in *this* task,
the startup message that states the front end's measured reach, the entry point that hands the
shared ``main`` this driver, the three pre-flight guards specific to reading raw signals, and the
command-line / IDE Run-button path resolution.

The pre-flight guards get their own assertions for the reason they exist: their whole value is
failing *before* the run directory, the log sinks and the MLflow run exist on every rank of a
multi-rank launch, and each guards a failure that is otherwise **silent** -- a config key the
signature sweep drops without a word, a raw field the loader hands over unnormalized, a shard whose
geometry disagrees with the model's.
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
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws import trainer as shared_trainer
from teb_vae.lag_attn_transformer_e2e import trainer as trainer_module
from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from teb_vae.lag_attn_transformer_e2e.task import SeqVaeLagAttnTrfE2ETask
from teb_vae.lag_attn_transformer_e2e.trainer import (
    INERT_MODEL_KEYS,
    LagAttnTrfE2ETrainer,
)

from .conftest import SHIPPED_KWARGS, absolutize_dataset_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"
_MODULE_NAME = "teb_vae.lag_attn_transformer_e2e.trainer"


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
# Config to constructor
# --------------------------------------------------------------------------------------
def test_the_shipped_config_resolves_to_the_shipped_architecture(driver):
    """Every constructor keyword ``SHIPPED_KWARGS`` sets, the config must resolve to. That fixture
    is the suite's description of the production model; this keeps it honest against the config
    file itself, through the real signature sweep."""
    kwargs = driver._build_model_kwargs()

    for name, expected in SHIPPED_KWARGS.items():
        assert name in kwargs, f"the shipped config does not set {name}"
        # YAML has no tuple; the constructor coerces, so the sweep hands a list through.
        value = tuple(kwargs[name]) if isinstance(kwargs[name], list) else kwargs[name]
        assert value == expected, f"{name}: config {value!r} against SHIPPED_KWARGS {expected!r}"


def test_the_configured_reach_budget_builds_the_front_ends_at_the_warmup_ceiling(driver):
    """The config sets the budget in seconds; converted at the raw sampling rate it must land on
    ``warmup_period * raw_per_step`` raw samples, the span of the anchors excluded from every loss,
    and both front ends must be built against it."""
    model = SeqVaeLagAttnTrfE2E(**driver._build_model_kwargs())

    assert model.frontend_reach_budget == model.warmup_period * model.raw_per_step
    for frontend in (model.target_frontend, model.source_frontend):
        assert frontend.reach_budget == model.frontend_reach_budget


def test_create_model_builds_this_net_and_wraps_it_in_this_task(driver):
    driver.create_model()

    assert isinstance(driver.pytorch_model, SeqVaeLagAttnTrfE2E)
    assert isinstance(driver.pl_model, SeqVaeLagAttnTrfE2ETask)
    assert driver.pl_model.orig_model is driver.pytorch_model


def test_the_logged_reach_cannot_drift_from_the_front_end_that_was_built(driver):
    """The inherited startup sentence is the negation of this package's claim, so it is replaced by
    the front end's measured reach. The message is composed *before* ``pytorch_model`` is assigned,
    on a throwaway front end; this is what stops that second construction disagreeing with the real
    one."""
    message = driver.causal_standing_message()
    driver.create_model()

    for frontend in (driver.pytorch_model.target_frontend, driver.pytorch_model.source_frontend):
        assert f"{frontend.reach_samples} raw samples" in message
        assert f"budget of {frontend.reach_budget}" in message


# --------------------------------------------------------------------------------------
# The entry point
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
    """``main`` delegates to the shared entry point; which driver it constructs is the one thing a
    delegation can get wrong while still running."""
    seen = {}

    def _capture_init(self, config_file_path=None):
        seen["cls"] = type(self)
        raise RuntimeError("stop here")

    monkeypatch.setattr(LagAttnTrfE2ETrainer, "__init__", _capture_init)

    with pytest.raises(RuntimeError, match="stop here"):
        trainer_module.main(_tiny_config_at(tmp_path))

    assert seen["cls"] is LagAttnTrfE2ETrainer


def test_the_shipped_configs_pass_this_drivers_own_preflight():
    """Both of them, so a guard cannot be satisfied by the smoke config alone. Called directly
    rather than through ``main``: the production config's shard paths do not exist on this box,
    which the shard half treats as non-fatal and the inherited ``_check_stat_path`` does not."""
    for path in (_CONFIG, _TINY):
        LagAttnTrfE2ETrainer.preflight(load_config(str(path)))


# --------------------------------------------------------------------------------------
# Pre-flight: the inert keys and the raw source
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("key", sorted(INERT_MODEL_KEYS))
def test_each_inert_key_is_refused_by_name_before_setup_config(recording_main, tmp_path, key):
    """The signature sweep drops each of these without a word, so the run would be a different one
    from the one the operator configured. Refused through ``main`` before ``setup_config``, so the
    run directory and the MLflow run never exist."""

    def _add_key(config):
        config["model_config"]["VAE_model"][key] = 1

    with pytest.raises(ValueError, match=key):
        trainer_module.main(_tiny_config_at(tmp_path, _add_key))

    assert recording_main == []


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


# --------------------------------------------------------------------------------------
# Pre-flight: the shard's raw length
# --------------------------------------------------------------------------------------
def test_the_committed_shards_trimmed_length_is_the_models_geometry(tmp_path):
    """The passing case, on the real fixture. A guard that compared the *stored* length rather than
    the trimmed one would fail on every real shard."""
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
# The command line and the IDE Run button
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


def test_a_launch_with_no_command_line_falls_back_to_an_existing_run_config(monkeypatch):
    """Which is what makes an IDE Run button work at all; a stale ``RUN_CONFIG`` path would break
    it silently."""
    launched = _launched_config(monkeypatch, ["trainer"])

    assert Path(launched) == _REPO_ROOT / trainer_module.RUN_CONFIG
    assert Path(launched).is_file()
