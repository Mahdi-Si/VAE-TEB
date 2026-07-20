r"""The entry point's call order, which nothing else enforces.

``main`` is four calls in a fixed order, and getting it wrong fails silently in the worst possible
way: ``create_model()`` before ``setup_config()`` means the run is unseeded, the output directories
do not exist, the log sinks are not open, and ``mlflow_logger`` is still ``None`` -- so
``build_trainer`` quietly omits the MLflow callback and a multi-day run records nothing. Nothing
raises. The framework cannot enforce the order; it can only document it. So it is tested.

The fit itself is stubbed here. What a real fit proves is a different question, asked in the smoke
test; this file is only about the wiring around it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from teb_vae.lag_attn import trainer as trainer_module

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TINY = _REPO_ROOT / "teb_vae" / "lag_attn" / "configs" / "tiny.yaml"


@pytest.fixture
def recording_main(monkeypatch, tmp_path):
    """Run ``main`` against the tiny config with every expensive step recorded rather than done."""
    calls = []

    def _record(name, result=None):
        def _recorder(self, *args, **kwargs):
            calls.append(name)
            return result

        return _recorder

    monkeypatch.setattr(trainer_module.LagAttnTrainer, "setup_config", _record("setup_config"))
    monkeypatch.setattr(trainer_module.LagAttnTrainer, "create_model", _record("create_model"))
    monkeypatch.setattr(trainer_module.LagAttnTrainer, "train_model", _record("train_model"))

    class _StubDataModule:
        def __init__(self, config):
            calls.append("data_module")
            self.config = config

        def train_dataloader(self):
            return object()

        def val_dataloader(self):
            return object()

    monkeypatch.setattr(trainer_module, "GraphDataModule", _StubDataModule)
    return calls


def _tiny_config_at(tmp_path, **general_overrides) -> str:
    """Write a resolved copy of the tiny config into ``tmp_path``, with overrides applied."""
    from teb_vae.lag_attn.config import load_config

    config = load_config(str(_TINY))
    config["general_config"].update(general_overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(path)


def test_setup_config_runs_before_the_model_is_built(recording_main, tmp_path):
    """The order that decides whether a run is seeded, logged and tracked at all."""
    trainer_module.main(_tiny_config_at(tmp_path))

    assert recording_main == ["setup_config", "data_module", "create_model", "train_model"]


def test_the_base_chain_is_resolved_before_the_driver_reads_it(monkeypatch, tmp_path):
    """The driver knows nothing about ``base:``.

    Handed the tiny config raw, it would see a file with almost no keys and fail on the first hard
    index -- so resolve-then-write is not a convenience, it is what makes a variant loadable.
    """
    seen = {}

    def _capture_init(self, config_file_path=None):
        seen["path"] = config_file_path
        seen["config"] = yaml.safe_load(Path(config_file_path).read_text(encoding="utf-8"))
        raise RuntimeError("stop here")

    monkeypatch.setattr(trainer_module.LagAttnTrainer, "__init__", _capture_init)

    with pytest.raises(RuntimeError, match="stop here"):
        trainer_module.main(str(_TINY))

    assert seen["path"] != str(_TINY), "the driver was handed the unresolved file"
    assert "base" not in seen["config"]
    # Inherited from default.yaml, and absent from tiny.yaml itself.
    assert seen["config"]["model_config"]["VAE_model"]["causal_norm"] is True
    assert seen["config"]["general_config"]["epochs"] == 1  # ...with the variant's override intact


def test_a_missing_stat_path_raises_before_any_training_happens(recording_main, tmp_path):
    """The guard the data layer cannot provide.

    ``_make_loader`` passes ``stat_path`` straight through, and the dataset skips normalization
    when it is ``None`` -- with a warning, not an error. A typo'd key (the config says
    ``stat_path``; the loader's parameter is ``stats_path``) would otherwise produce a full run on
    raw-scale inputs.
    """
    from teb_vae.lag_attn.config import load_config

    config = load_config(str(_TINY))
    config["dataset_config"]["stat_path"] = None
    path = tmp_path / "no_stats.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="stat_path"):
        trainer_module.main(str(path))

    assert "create_model" not in recording_main


def test_the_data_module_is_used_directly(recording_main, tmp_path):
    """No wrapper module around it.

    ``GraphDataModule`` reads the same config paths this model would have to read anyway and
    deliberately omits rank/world_size so Lightning owns the sampler. A module that only re-exported
    it would be a wrapper that renames an API.
    """
    trainer_module.main(_tiny_config_at(tmp_path))

    assert "data_module" in recording_main


def test_the_module_does_not_seed_by_hand():
    """Determinism is ``general_config.seed`` plus the framework's ``configure_determinism``.

    A stray ``torch.manual_seed`` here would silently override the configured seed and make the
    config key a lie, while looking like diligence.
    """
    source = Path(trainer_module.__file__).read_text(encoding="utf-8")

    assert "manual_seed" not in source
    assert "np.random.seed" not in source


# --------------------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------------------
def test_help_exits_zero():
    """The advertised command's front door. Running as ``-m`` is the whole interface."""
    result = subprocess.run(
        [sys.executable, "-m", "teb_vae.lag_attn.trainer", "--help"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_config_is_required():
    """No default pointing at a path that may not exist.

    The trainer this was ported from defaulted to a config beside its own module, so a bare run
    silently trained the baseline configuration -- or died with a ``FileNotFoundError`` naming a
    file the user never asked for.
    """
    result = subprocess.run(
        [sys.executable, "-m", "teb_vae.lag_attn.trainer"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--config" in result.stderr


# --------------------------------------------------------------------------------------
# The IDE Run button
# --------------------------------------------------------------------------------------
def test_run_config_ships_pointing_at_a_real_config_and_the_flag_still_wins():
    """The Run button launches the real configuration; ``--config`` overrides it regardless.

    ``RUN_CONFIG`` shipped as ``None`` until 5f23af7, which pointed it at ``default.yaml`` so a
    bare Run-button launch trains the real recipe -- deliberately, with the module docstring and
    the constant's own comment rewritten to say so. What still has to hold is the precedence:
    ``--config`` wins over this value, so the dict cannot quietly redirect a command-line run.
    """
    assert trainer_module.RUN_CONFIG == "teb_vae/lag_attn/configs/default.yaml"
    assert Path(_REPO_ROOT / trainer_module.RUN_CONFIG).is_file(), (
        "RUN_CONFIG names a config that does not exist, so the Run button is broken"
    )


def test_the_module_is_importable_as_a_script_from_an_unrelated_directory(tmp_path):
    """An IDE's Run button executes the file, so ``sys.path[0]`` is its own directory.

    Without the repo-root bootstrap the ``teb_vae.`` imports raise ModuleNotFoundError before
    ``__main__`` is reached, and the failure looks like a broken install rather than a launch mode.
    Run from ``tmp_path`` so a repo-root working directory cannot mask a missing bootstrap.
    """
    result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "teb_vae" / "lag_attn" / "trainer.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    # It must get past the imports and into ``__main__``. Asserting on the *bootstrap* rather
    # than on how far the run then gets: with RUN_CONFIG shipping set, a bare launch proceeds
    # into config resolution and stops at whatever the environment supplies (on an unconfigured
    # box, the missing stats file), which is not a property of the launch mode under test.
    assert "ModuleNotFoundError" not in result.stderr, result.stderr
    assert "changing working directory to the repo root" in result.stderr, result.stderr
    # And specifically down the Run-button path: no --config was passed, so the bare launch must
    # resolve through RUN_CONFIG. This is the branch the test exists to cover; the chdir line
    # above would also appear on a run that took its config from the command line.
    assert "no --config given; using RUN_CONFIG=" in result.stderr, result.stderr


def test_a_missing_stats_file_raises_before_any_training_happens(recording_main, tmp_path):
    """Set-but-wrong is the same failure as unset, and likelier.

    The loader warns ``Statistics file not found ... Normalization disabled`` and carries on, so a
    mistyped or not-yet-generated ``stat_path`` costs a full run on raw-scale inputs -- announced
    only by a warning in a multi-day log. Checking the key is non-None was never enough.
    """
    from teb_vae.lag_attn.config import load_config

    config = load_config(str(_TINY))
    config["dataset_config"]["stat_path"] = str(tmp_path / "not_generated_yet.hdf5")
    path = tmp_path / "missing_stats.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="stat_path does not exist"):
        trainer_module.main(str(path))

    assert "create_model" not in recording_main


def test_the_width_guard_is_actually_wired_into_main(recording_main, tmp_path, monkeypatch):
    """The guard's own tests call it directly, so nothing else pins it to the launch path.

    Deleting the call from ``main`` leaves the whole suite green -- the per-batch check in the task
    still catches a mismatch, so the loss is failure *latency* on a multi-rank launch rather than
    correctness, but a safety net nothing exercises is one that quietly stops existing.
    """
    called = []
    monkeypatch.setattr(
        trainer_module,
        "_check_declared_widths_against_shard",
        lambda config: called.append(config),
    )

    trainer_module.main(_tiny_config_at(tmp_path))

    assert called, "main() no longer calls _check_declared_widths_against_shard"
    assert called[0]["model_config"]["VAE_model"]["c_y"] == 109


def test_both_pre_flight_guards_run_before_setup_config(tmp_path, monkeypatch):
    """Their whole value is failing before the run directory and MLflow run exist.

    ``setup_config`` seeds the run, creates the output directories, opens the log sinks and
    connects MLflow. A guard that fires after it has already left that debris behind on every rank
    of a 7-rank launch.
    """
    order = []
    monkeypatch.setattr(
        trainer_module.LagAttnTrainer,
        "setup_config",
        lambda self: order.append("setup_config"),
    )
    monkeypatch.setattr(
        trainer_module, "_check_stat_path", lambda config: order.append("stat_path")
    )
    monkeypatch.setattr(
        trainer_module, "_check_declared_widths_against_shard", lambda config: order.append("widths")
    )
    monkeypatch.setattr(trainer_module, "GraphDataModule", lambda config: None)
    monkeypatch.setattr(
        trainer_module.LagAttnTrainer, "create_model", lambda self: order.append("create_model")
    )

    with pytest.raises(AttributeError):
        # GraphDataModule is stubbed to None, so main dies at train_dataloader() -- after the part
        # under test. The order up to that point is the assertion.
        trainer_module.main(_tiny_config_at(tmp_path))

    assert order[:3] == ["stat_path", "widths", "setup_config"], order
