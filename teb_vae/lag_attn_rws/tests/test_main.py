r"""The entry point's call order and its three pre-flight guards.

``main`` is a handful of calls in a fixed order, and getting it wrong fails silently in the
worst way: ``create_model()`` before ``setup_config()`` means the run is unseeded, the output
directories do not exist, and ``mlflow_logger`` is still ``None`` -- so the MLflow callback is
quietly omitted and a multi-day run records nothing. Nothing raises. The framework cannot
enforce the order; it can only document it. So it is tested.

The third guard is specific to this model: the raw ``fhr`` is the reconstruction *target*, and
missing normalization on it is the one misconfiguration that trains a meaningless objective to
completion with no error anywhere.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from teb_vae.lag_attn_rws import trainer as trainer_module
from teb_vae.lag_attn.config import load_config

from .conftest import absolutize_dataset_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TINY = _REPO_ROOT / "teb_vae" / "lag_attn_rws" / "configs" / "tiny.yaml"


@pytest.fixture
def recording_main(monkeypatch):
    """Run ``main`` with every expensive step recorded rather than done."""
    calls = []

    def _record(name, result=None):
        def _recorder(self, *args, **kwargs):
            calls.append(name)
            return result

        return _recorder

    monkeypatch.setattr(
        trainer_module.LagAttnRwsTrainer, "setup_config", _record("setup_config")
    )
    monkeypatch.setattr(
        trainer_module.LagAttnRwsTrainer, "create_model", _record("create_model")
    )
    monkeypatch.setattr(
        trainer_module.LagAttnRwsTrainer, "train_model", _record("train_model")
    )

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


def _tiny_config_at(tmp_path, mutate=None) -> str:
    """Write a resolved copy of the tiny config into ``tmp_path``, optionally mutated."""
    config = load_config(str(_TINY))
    if mutate is not None:
        mutate(config)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------------------
# Call order
# --------------------------------------------------------------------------------------
def test_setup_config_runs_before_the_model_is_built(recording_main, tmp_path):
    """The order that decides whether a run is seeded, logged and tracked at all."""
    trainer_module.main(_tiny_config_at(tmp_path))

    assert recording_main == ["setup_config", "data_module", "create_model", "train_model"]


def test_all_three_pre_flight_guards_run_before_setup_config(tmp_path, monkeypatch):
    """Their whole value is failing before the run directory and MLflow run exist on every
    rank of a multi-rank launch."""
    order = []
    monkeypatch.setattr(
        trainer_module.LagAttnRwsTrainer,
        "setup_config",
        lambda self: order.append("setup_config"),
    )
    monkeypatch.setattr(
        trainer_module, "_check_stat_path", lambda config: order.append("stat_path")
    )
    monkeypatch.setattr(
        trainer_module,
        "_check_declared_widths_against_shard",
        lambda config: order.append("widths"),
    )
    monkeypatch.setattr(
        trainer_module,
        "_check_raw_target_normalized",
        lambda config: order.append("fhr_normalized"),
    )
    monkeypatch.setattr(trainer_module, "GraphDataModule", lambda config: None)
    monkeypatch.setattr(
        trainer_module.LagAttnRwsTrainer,
        "create_model",
        lambda self: order.append("create_model"),
    )

    with pytest.raises(AttributeError):
        # GraphDataModule is stubbed to None, so main dies at train_dataloader() -- after the
        # part under test. The order up to that point is the assertion.
        trainer_module.main(_tiny_config_at(tmp_path))

    assert order[:4] == ["stat_path", "widths", "fhr_normalized", "setup_config"], order


def test_the_base_chain_is_resolved_before_the_driver_reads_it(monkeypatch):
    """The driver knows nothing about ``base:``; handed the raw tiny file it would fail on the
    first hard index. Resolve-then-write is what makes a variant loadable."""
    seen = {}

    def _capture_init(self, config_file_path=None):
        seen["path"] = config_file_path
        seen["config"] = yaml.safe_load(Path(config_file_path).read_text(encoding="utf-8"))
        raise RuntimeError("stop here")

    monkeypatch.setattr(trainer_module.LagAttnRwsTrainer, "__init__", _capture_init)

    with pytest.raises(RuntimeError, match="stop here"):
        trainer_module.main(str(_TINY))

    assert seen["path"] != str(_TINY), "the driver was handed the unresolved file"
    assert "base" not in seen["config"]
    # Inherited from default.yaml, and absent from tiny.yaml itself...
    assert seen["config"]["model_config"]["VAE_model"]["causal_norm"] is True
    # ...with the variant's own override intact.
    assert seen["config"]["general_config"]["epochs"] == 1


def test_the_resolved_config_is_written_beside_the_checkpoints(tmp_path, monkeypatch):
    """The evaluation entry point takes no ``--config``: it derives one from the checkpoint path.

    That only works if a run leaves its fully resolved configuration in the checkpoint
    directory. Otherwise the record of what a run trained on survives only inside the text of its
    log and in an MLflow artifact whose on-disk location nothing can derive -- neither of which
    an evaluation can open.
    """
    reloaded = _persisted_config(tmp_path, monkeypatch)

    # Fully resolved: the inherited keys are present and the `base:` pointer is gone.
    assert "base" not in reloaded
    assert reloaded["model_config"]["VAE_model"]["causal_norm"] is True
    # The unguarded default records the *absence* of a guard explicitly, so a reader can tell it
    # from a run written before the record existed.
    assert reloaded["model_config"][trainer_module.RESOLVED_BUDGET_KEY] is None


def _persisted_config(tmp_path, monkeypatch, mutate=None) -> dict:
    """Run ``main`` with everything expensive stubbed, and return the config it left on disk.

    Args:
        tmp_path: Directory the run writes into.
        monkeypatch: The pytest fixture.
        mutate: Optional extra mutation of the config before the run.

    Returns:
        The reloaded ``resolved_config.yaml``.
    """
    monkeypatch.setattr(
        trainer_module.LagAttnRwsTrainer, "create_model", lambda self: None
    )
    monkeypatch.setattr(
        trainer_module.LagAttnRwsTrainer, "train_model", lambda self, *args: None
    )
    monkeypatch.setattr(trainer_module, "GraphDataModule", lambda config: _StubDataModule())

    captured = {}

    def _remember(self):
        # setup_config is what creates the run directories, so the write must follow it.
        trainer_module.GraphModelBase.setup_config(self)
        captured["checkpoint_dir"] = self.model_checkpoint_dir

    monkeypatch.setattr(trainer_module.LagAttnRwsTrainer, "setup_config", _remember)

    def _redirect(config):
        config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path)
        absolutize_dataset_paths(config)
        if mutate is not None:
            mutate(config)

    trainer_module.main(_tiny_config_at(tmp_path, _redirect))

    written = Path(captured["checkpoint_dir"]) / trainer_module.RESOLVED_CONFIG_FILENAME
    assert written.is_file()
    return yaml.safe_load(written.read_text(encoding="utf-8"))


def test_the_resolved_config_records_the_causal_guard_the_run_actually_got(
    tmp_path, monkeypatch
):
    """The budget in seconds does not name a channel: what it resolves to depends on a filter
    bank. So a run that recorded only ``causal_reach_budget_s`` would record what it asked for
    and not what it got, and reconstructing the difference would mean rebuilding the bank.

    Written under ``model_config`` rather than inside ``VAE_model``, and under a name that is not
    a constructor argument, so re-running from the written file does not both forward the record
    and re-resolve the budget.
    """

    def _guarded(config):
        config["model_config"]["VAE_model"]["causal_reach_budget_s"] = 120.0

    record = _persisted_config(tmp_path, monkeypatch, _guarded)["model_config"][
        trainer_module.RESOLVED_BUDGET_KEY
    ]

    assert record["causal_reach_budget_s"] == 120.0
    assert record["max_delay_steps"] == 30
    assert record["channels_kept_per_block"] == {
        "fhr_st": {"kept": 27, "declared": 43},
        "fhr_ph": {"kept": 51, "declared": 66},
        "up_st": {"kept": 27, "declared": 43},
        "up_ph": {"kept": 2, "declared": 15},
    }
    assert len(record["source_delays"]) == len(record["source_keep_index"]) == 29


def test_an_unsatisfiable_reach_budget_raises_before_any_training_happens(
    recording_main, tmp_path
):
    """A budget whose delay outruns ``warmup_period`` would zero-fill trained anchors. Caught in
    the pre-flight, so it costs nothing rather than surfacing after every rank has initialised
    and the run directory and MLflow run exist."""

    def _too_deep(config):
        config["model_config"]["VAE_model"]["causal_reach_budget_s"] = 240.0

    with pytest.raises(ValueError, match="warmup_period"):
        trainer_module.main(_tiny_config_at(tmp_path, _too_deep))

    assert "create_model" not in recording_main


class _StubDataModule:
    """A data module that hands out loaders nothing iterates."""

    def train_dataloader(self):
        return object()

    def val_dataloader(self):
        return object()


# --------------------------------------------------------------------------------------
# The guards
# --------------------------------------------------------------------------------------
def test_a_missing_stat_path_raises_before_any_training_happens(recording_main, tmp_path):
    def _drop_stats(config):
        config["dataset_config"]["stat_path"] = None

    with pytest.raises(ValueError, match="stat_path"):
        trainer_module.main(_tiny_config_at(tmp_path, _drop_stats))

    assert "create_model" not in recording_main


def test_a_missing_stats_file_raises_before_any_training_happens(recording_main, tmp_path):
    """Set-but-wrong is the same failure as unset, and likelier: the loader only warns."""

    def _wrong_stats(config, _tmp=tmp_path):
        config["dataset_config"]["stat_path"] = str(_tmp / "not_generated_yet.hdf5")

    with pytest.raises(ValueError, match="stat_path does not exist"):
        trainer_module.main(_tiny_config_at(tmp_path, _wrong_stats))

    assert "create_model" not in recording_main


@pytest.mark.parametrize(
    "list_key, expected_fragment",
    [
        ("load_fields", "load_fields"),
        ("normalize_fields", "normalize_fields"),
    ],
)
def test_a_missing_fhr_raises_naming_the_offending_list(
    recording_main, tmp_path, list_key, expected_fragment
):
    """The model-specific guard: without 'fhr' in normalize_fields the target arrives in bpm,
    the Gaussian NLL is meaningless, and nothing else raises -- a full run trained on nothing."""

    def _drop_fhr(config):
        dataloader = config["dataset_config"]["dataloader_config"]
        if list_key == "load_fields":
            fields = dataloader["dataset_kwargs"]["load_fields"]
        else:
            fields = dataloader[list_key]
        fields.remove("fhr")

    with pytest.raises(ValueError, match=expected_fragment):
        trainer_module.main(_tiny_config_at(tmp_path, _drop_fhr))

    assert "create_model" not in recording_main


# Derived from __file__, not cwd-relative: the width guard swallows a failed open (a missing
# shard is the data module's to report), so a path that does not resolve would make the
# mismatch tests pass without ever reaching the width arithmetic.
_SHARD = str(
    Path(__file__).resolve().parents[2] / "lag_attn" / "tests" / "fixtures" / "tiny_shard.hdf5"
)


def _width_config(**vae):
    declared = dict(c_y=109, c_u=58, use_up_st=True)
    declared.update(vae)
    return {
        "dataset_config": {"vae_train_datasets": [_SHARD]},
        "model_config": {"VAE_model": declared},
    }


@pytest.mark.parametrize(
    "declared, expected_fragment",
    [
        # The trap: the old phase-only pairing, which no config-shaped check can catch.
        (dict(use_up_st=False, c_u=58), "up_ph=15"),
        (dict(c_y=87), "fhr_ph=66"),
        (dict(c_u=101), "up_st=43 + up_ph=15"),
    ],
)
def test_declared_widths_are_checked_against_the_shard_before_the_fit(
    declared, expected_fragment
):
    """Fails on rank 0 before the data module, not inside ``training_step``. The message must
    name the shard's own per-field widths or the reader cannot tell which number is wrong."""
    with pytest.raises(ValueError, match="channel widths disagree") as excinfo:
        trainer_module._check_declared_widths_against_shard(_width_config(**declared))

    assert expected_fragment in str(excinfo.value)


def test_widths_matching_the_shard_pass_the_pre_fit_guard():
    trainer_module._check_declared_widths_against_shard(_width_config())
    trainer_module._check_declared_widths_against_shard(
        _width_config(use_up_st=False, c_u=15)
    )


def test_the_pre_fit_width_guard_defers_rather_than_masking_a_data_module_error():
    """An unreadable shard is the data module's to report; a guard that raised here would
    replace ``FileNotFoundError: <path>`` with a width complaint about a file that does not
    exist."""
    trainer_module._check_declared_widths_against_shard(
        {
            "dataset_config": {"vae_train_datasets": ["/nonexistent/shard.hdf5"]},
            "model_config": {"VAE_model": {"c_y": 109, "c_u": 58, "use_up_st": True}},
        }
    )
    # And a config that declares nothing for it to check is a no-op, not a crash.
    trainer_module._check_declared_widths_against_shard({})


# --------------------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------------------
def test_relative_config_paths_resolve_against_the_repository_root():
    """An IDE's working directory is arbitrary; every documented invocation is repo-root
    relative, so the resolver must anchor there and leave absolute paths alone."""
    resolved = trainer_module._resolve_cli_config_path("teb_vae/lag_attn_rws/configs/tiny.yaml")

    assert Path(resolved) == _TINY
    absolute = str(_TINY)
    assert trainer_module._resolve_cli_config_path(absolute) == absolute


def test_run_config_points_at_a_config_that_exists():
    """The IDE Run button resolves through ``RUN_CONFIG``; a stale path breaks it silently."""
    assert trainer_module.RUN_CONFIG is not None
    assert (_REPO_ROOT / trainer_module.RUN_CONFIG).is_file()


def test_the_module_does_not_seed_by_hand():
    """Determinism is ``general_config.seed`` plus the framework's ``configure_determinism``; a
    stray ``torch.manual_seed`` here would silently override the configured seed."""
    source = Path(trainer_module.__file__).read_text(encoding="utf-8")

    assert "manual_seed" not in source
    assert "np.random.seed" not in source
    assert "seed_everything" not in source
