r"""The driver turns config into this model, hands the task what only the driver knows, and launches
from the command line or the IDE's Run button.

Almost all of this module's behaviour is inherited, and the shared entry point's ordering, guards and
resolved-config write are pinned in ``lag_attn_rws``. What is this cell's own:

* ``_build_model_kwargs`` translates ``causal_warmup_budget_steps`` -- which names no constructor
  argument -- into the four channel tuples and the two shift vectors the network takes. A driver
  that failed to would build an **ungated** model that trains to completion reading the region where
  a one-sided filter's output is a function of assumed pre-recording history.
* ``create_model`` hands the task the run seed (the tile phase is derived from it and no other route
  reaches the task) and the resolved budget (the run-level warm-up figure is drawn from it).
* ``main`` and the ``__main__`` block resolve a config path against the repository root and fall back
  to ``RUN_CONFIG`` when launched with no command line.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import pytest
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_crws import trainer as trainer_module
from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_crws.task import SeqVaeLagAttnCrwsTask
from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer
from teb_vae.lag_attn_rws import trainer as shared_trainer

from .conftest import CAUSAL_C_U, CAUSAL_C_Y, absolutize_dataset_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_TINY = _CONFIG_DIR / "tiny.yaml"
_MODULE_NAME = "teb_vae.lag_attn_crws.trainer"

#: What the shipped guard resolves to against the committed causal fixture: the known answer for
#: this cell's alignment reference. The warm-up budget takes target channels off, and the alignment
#: reference takes every source channel slower than it off the source.
GUARDED_TARGET_CHANNELS = 38
GUARDED_SOURCE_CHANNELS = 17


@pytest.fixture
def driver(tmp_path):
    """A driver on the tiny config, with its output directories redirected under ``tmp_path``.

    The tiny config rather than the shipped one: the shipped config's shard paths are placeholders,
    and this driver's ``_build_model_kwargs`` **reads the shards**. The tiny variant carries the real
    geometry and points at the committed causal fixture, so every resolved number below is the
    production one. ``setup_config`` is never called, so the directories are assigned directly.
    """
    config = absolutize_dataset_paths(load_config(str(_TINY)))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    instance = LagAttnCrwsTrainer(config_file_path=str(path))
    instance.output_base_dir = str(tmp_path)
    instance.train_results_dir = str(tmp_path / "train_results")
    instance.model_checkpoint_dir = str(tmp_path / "model_checkpoints")
    return instance


# --------------------------------------------------------------------------------------
# Config to constructor
# --------------------------------------------------------------------------------------
def test_the_warmup_budget_reaches_the_constructor_and_builds_the_guarded_model(driver):
    """Translated rather than forwarded: the threshold names no constructor argument, and what the
    network takes is the concrete channel set it resolves to against the shards. The model it builds
    reads the widths the budget kept, keeps the declared widths the data boundary checks against,
    and has a decoder exactly ``raw_per_step`` wide -- no config key can move it."""
    kwargs = driver._build_model_kwargs()

    assert "causal_warmup_budget_steps" not in kwargs
    assert "decoder_out_channels" not in kwargs
    assert len(kwargs["target_keep_index"]) == len(kwargs["target_warmup_steps"])
    assert len(kwargs["source_keep_index"]) == len(kwargs["source_warmup_steps"])

    model = SeqVaeLagAttnCrws(**kwargs)

    assert model.target_adapter.linear.in_features == GUARDED_TARGET_CHANNELS
    assert model.source_adapter.linear.in_features == GUARDED_SOURCE_CHANNELS
    assert (model.c_y, model.c_u) == (CAUSAL_C_Y, CAUSAL_C_U)
    assert model.decoder_out_channels == kwargs["raw_per_step"]


def test_a_config_without_a_budget_builds_the_ungated_model(driver):
    """The arm every "the guard did something" comparison is made against, and the one whose
    silence is the hazard: no gate, no warm-up mask, and both adapters at their declared widths."""
    driver.config["model_config"]["VAE_model"]["causal_warmup_budget_steps"] = None

    kwargs = driver._build_model_kwargs()

    assert "target_keep_index" not in kwargs
    assert driver.resolved_warmup is None
    assert SeqVaeLagAttnCrws(**kwargs).target_adapter.linear.in_features == CAUSAL_C_Y


def test_create_model_builds_this_net_and_hands_the_task_the_seed_and_the_budget(driver):
    """The net and task are this cell's, and the checkpoint kwargs are the ones the net was built
    from. The run seed reaches the task's hyperparameters -- the tile phase is derived from it, so a
    resumed run that did not know it would silently re-tile every segment -- and the resolved budget
    is handed over as a plain attribute: the run-level warm-up figure is about the channels the
    budget *dropped*, which the checkpoint does not carry."""
    driver.create_model()

    assert isinstance(driver.pytorch_model, SeqVaeLagAttnCrws)
    assert isinstance(driver.pl_model, SeqVaeLagAttnCrwsTask)
    assert driver.pl_model.orig_model is driver.pytorch_model
    assert driver.pl_model.hparams["seed"] == driver.config["general_config"]["seed"]
    assert driver.pl_model.warmup_budget is driver.resolved_warmup is not None
    assert "warmup_budget" not in driver.pl_model.hparams
    # Last: rebuilding the kwargs re-resolves the budget onto the driver.
    assert driver.pl_model._model_kwargs == driver._build_model_kwargs()


# --------------------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------------------
def test_relative_config_paths_resolve_against_the_repository_root():
    """An IDE's working directory is arbitrary; every documented invocation is repo-root relative."""
    resolved = trainer_module._resolve_cli_config_path("teb_vae/lag_attn_crws/configs/tiny.yaml")

    assert Path(resolved) == _TINY
    absolute = str(_TINY)
    assert trainer_module._resolve_cli_config_path(absolute) == absolute


def _launched_config(monkeypatch, argv) -> str:
    """Execute the module's ``__main__`` block under ``argv`` and return the config it launched."""
    launched = {}
    monkeypatch.setattr(
        shared_trainer, "main", lambda path, trainer_cls=None: launched.setdefault("path", path)
    )
    monkeypatch.setattr(sys, "argv", list(argv))
    monkeypatch.chdir(os.getcwd())

    runpy.run_module(_MODULE_NAME, run_name="__main__", alter_sys=True)

    return launched["path"]


def test_the_command_line_config_wins_over_run_config(monkeypatch, tmp_path):
    requested = str(tmp_path / "requested.yaml")

    assert _launched_config(monkeypatch, ["trainer", "--config", requested]) == requested


def test_a_launch_with_no_command_line_falls_back_to_an_existing_run_config(monkeypatch):
    """The IDE Run button resolves through ``RUN_CONFIG``; a required ``--config`` or a stale path
    would break it silently."""
    launched = _launched_config(monkeypatch, ["trainer"])

    assert Path(launched) == _REPO_ROOT / trainer_module.RUN_CONFIG
    assert Path(launched).is_file()
