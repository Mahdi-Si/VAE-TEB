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
