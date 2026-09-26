r"""The driver turns config into this model, and the entry point hands the guards this driver.

Every line of behaviour here is inherited from two parents at once, which is the design and also the
risk: **three class attributes are the entire difference** between building this architecture and
building either model it is compared against, and each failure is silent. Both parents set all three,
so resolution order alone would take the feature side -- a conv-LSTM model, built, trained and
reported under this package's config, tag and MLflow experiment, with nothing anywhere raising.

The rest of the diamond is asserted by *behaviour*: the model and task ``create_model`` builds, the
learning-rate monitor the trainer kwargs carry, the live compile key, and the target fields the
shared normalisation guard is handed. Each resolves to a different parent, and none of them raises
when it resolves to the wrong one.

The module's own code -- ``main``, the command-line path resolution and the ``RUN_CONFIG`` fallback
that makes the IDE Run button work -- is tested directly.
"""
from __future__ import annotations

import inspect
import os
import runpy
import sys
from pathlib import Path

import pytest
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer
from teb_vae.lag_attn_rws import trainer as shared_trainer
from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer
from teb_vae.lag_attn_transformer_e2e.trainer import LagAttnTrfE2ETrainer
from teb_vae.lag_attn_transformer_fs import trainer as trainer_module
from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs
from teb_vae.lag_attn_transformer_fs.task import SeqVaeLagAttnTrfFsTask
from teb_vae.lag_attn_transformer_fs.trainer import LagAttnTrfFsTrainer
from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer

from .conftest import absolutize_dataset_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"
_MODULE_NAME = "teb_vae.lag_attn_transformer_fs.trainer"

#: Class attributes this driver may declare. A set rather than a count: a count passes a subclass
#: that overrode ``train_model`` while dropping the plot callback.
_OWN_ATTRIBUTES = {"MODEL_CLS", "TASK_CLS", "CHECKPOINT_STEM"}

#: Every driver in the tree, so the checkpoint stem is checked for distinctness against all of them.
#: The stem is the checkpoint *filename*: a copy-pasted one interleaves two models' blobs in whichever
#: output tree they share.
_FAMILY_DRIVERS = (
    LagAttnRwsTrainer,
    LagAttnFsTrainer,
    LagAttnTrfRwsTrainer,
    LagAttnTrfE2ETrainer,
    LagAttnTrfFsTrainer,
)


@pytest.fixture
def driver(tmp_path):
    """A driver on the shipped config, with its output directories redirected under ``tmp_path``.

    ``setup_config`` is never called -- it would seed, open log sinks and probe MLflow -- so the
    directories are assigned directly.
    """
    instance = LagAttnTrfFsTrainer(config_file_path=str(_CONFIG))
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
# The three class attributes, and what they decide
# --------------------------------------------------------------------------------------
def test_the_driver_declares_three_attributes_and_overrides_no_method():
    """Every method it could override is a piece of machinery both comparisons rest on: the kwarg
    sweep, ``create_model``, the callback assembly, the DDP selection, the learning-rate monitor
    swap. Redefining any of them here would be a second copy free to drift from the one a comparison
    model runs under."""
    own = {name for name in vars(LagAttnTrfFsTrainer) if not name.startswith("_")}
    # ``isroutine`` rather than ``callable``: two of the three declared attributes are *classes*, and
    # a class is callable, so a plain callability filter would report the re-pointings as methods.
    methods = {
        name
        for name, value in vars(LagAttnTrfFsTrainer).items()
        if inspect.isroutine(value) or isinstance(value, (classmethod, staticmethod, property))
    }

    assert own == _OWN_ATTRIBUTES
    assert methods == set()


def test_every_drivers_checkpoint_stem_is_distinct():
    """Two models writing under one stem into a shared output tree are indistinguishable by name,
    and the blobs' ``model_class`` stamp is only discoverable after loading one."""
    stems = [cls.CHECKPOINT_STEM for cls in _FAMILY_DRIVERS]

    assert len(set(stems)) == len(stems), stems


def test_create_model_builds_this_net_and_wraps_it_in_this_task(driver):
    """Omit ``MODEL_CLS`` or ``TASK_CLS`` and the feature parent's would win resolution -- a
    conv-LSTM model under this package's config, with no error anywhere."""
    driver.create_model()

    assert type(driver.pytorch_model) is SeqVaeLagAttnTrfFs
    assert type(driver.pl_model) is SeqVaeLagAttnTrfFsTask
    assert driver.pl_model.orig_model is driver.pytorch_model


# --------------------------------------------------------------------------------------
# The conv-Transformer parent's machinery, by behaviour
# --------------------------------------------------------------------------------------
def test_exactly_one_learning_rate_monitor_is_attached_and_it_is_step_granular(driver):
    """A warm-up measured in optimizer steps completes well inside the first epochs, so the
    framework's epoch-granular monitor cannot show it at all. Replaced rather than supplemented, so
    exactly one of them logs, at one resolution."""
    monitors = [
        callback
        for callback in driver._build_trainer_kwargs([])["callbacks"]
        if isinstance(callback, LearningRateMonitor)
    ]

    assert len(monitors) == 1
    assert monitors[0].logging_interval == "step"


def test_the_compile_key_is_live_for_this_driver(driver):
    """The one behavioural difference the diamond's resolution order introduces: the feature parent
    does not read the key at all, and this driver inherits the conv-Transformer parent's reading. If
    the key stayed inert an operator could set ``compile: true``, see nothing in the log, and believe
    they had measured a compiled run."""
    driver.config["advanced_config"]["trainer"]["compile"] = True

    assert driver.compile_model_requested() is True


# --------------------------------------------------------------------------------------
# The entry point
# --------------------------------------------------------------------------------------
def test_the_entry_point_constructs_this_packages_driver(monkeypatch, tmp_path):
    """``main`` delegates to the shared entry point, which owns the guards, the temporary
    resolved-config file and the resolved-config write. What this package supplies is which driver it
    constructs -- and that is the one thing a delegation can get wrong while still running."""
    seen = {}

    def _capture_init(self, config_file_path=None):
        seen["cls"] = type(self)
        raise RuntimeError("stop here")

    monkeypatch.setattr(LagAttnTrfFsTrainer, "__init__", _capture_init)

    with pytest.raises(RuntimeError, match="stop here"):
        trainer_module.main(_tiny_config_at(tmp_path))

    assert seen["cls"] is LagAttnTrfFsTrainer


def test_the_normalisation_guard_is_handed_this_drivers_target_fields(tmp_path, monkeypatch):
    """The plumbing a ``trainer_cls=`` wiring mistake or a reordered diamond breaks. Checked by value
    rather than by behaviour, because a guard wired to the raw driver's field would still run and
    still refuse *something*, and these configs satisfy its check."""
    seen = {}
    monkeypatch.setattr(
        shared_trainer,
        "_check_raw_target_normalized",
        lambda config, **kwargs: seen.update(kwargs),
    )
    monkeypatch.setattr(LagAttnTrfFsTrainer, "setup_config", lambda self: None)
    monkeypatch.setattr(shared_trainer, "GraphDataModule", lambda config: None)

    with pytest.raises(AttributeError):
        # GraphDataModule is stubbed to None, so main dies at train_dataloader() -- after the part
        # under test.
        trainer_module.main(_tiny_config_at(tmp_path))

    assert seen == {"fields": ("fhr_st", "fhr_ph")}


# --------------------------------------------------------------------------------------
# The command line and the Run-button convention
# --------------------------------------------------------------------------------------
def test_relative_config_paths_resolve_against_the_repository_root():
    """An IDE's working directory is arbitrary; every documented invocation is repo-root relative, so
    the resolver must anchor there and leave absolute paths alone."""
    resolved = trainer_module._resolve_cli_config_path(
        "teb_vae/lag_attn_transformer_fs/configs/tiny.yaml"
    )

    assert Path(resolved) == _TINY
    absolute = str(_TINY)
    assert trainer_module._resolve_cli_config_path(absolute) == absolute


def _launched_config(monkeypatch, argv) -> str:
    """Execute the module's ``__main__`` block under ``argv`` and return the config it launched.

    The block is re-executed in a fresh namespace, so the ``main`` it calls is the one this package's
    module imports from the shared entry point -- patching that name is what makes the launch
    observable without running a fit.

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
    """``RUN_CONFIG`` exists and is valid, so an entry point that ignored ``--config`` would launch a
    full production run when a smoke run was asked for."""
    requested = str(tmp_path / "requested.yaml")

    assert _launched_config(monkeypatch, ["trainer", "--config", requested]) == requested


def test_a_launch_with_no_command_line_falls_back_to_an_existing_run_config(monkeypatch):
    """Which is what makes an IDE Run button work at all -- and a stale path would break it
    silently."""
    launched = _launched_config(monkeypatch, ["trainer"])

    assert Path(launched) == _REPO_ROOT / trainer_module.RUN_CONFIG
    assert Path(launched).is_file()
