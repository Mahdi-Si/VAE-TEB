r"""The driver turns config into a model, and wires the callbacks the run depends on.

Two things are worth testing here and are easy to get silently wrong.

The config-to-constructor sweep is one. A key that fails to reach the constructor does not raise --
the constructor has a default for everything -- so the run trains a *different architecture* than
its config describes, and only a checkpoint that will not reload months later reveals it. The
assertions below therefore check the resolved kwargs against the flags the shipped config sets,
name by name.

The callback wiring is the other. Nothing fails if a callback is missing; the artefacts it would
have written simply never appear.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn.tests.conftest import SHIPPED_KWARGS
from teb_vae.lag_attn.trainer import (
    _TRACKED_METRICS,
    LagAttnTrainer,
    _check_declared_widths_against_shard,
)
from train.callbacks import (
    MetricsHistoryCsvCallback,
    MetricsLoggingCallback,
    _unreachable_metric_names,
)

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def trainer(tmp_path):
    """A driver on the shipped config, with its output directories redirected under ``tmp_path``.

    ``setup_config`` is never called -- it would seed, open log sinks and probe MLflow -- so the
    directories are assigned directly.
    """
    driver = LagAttnTrainer(config_file_path=str(_CONFIG))
    driver.output_base_dir = str(tmp_path)
    driver.train_results_dir = str(tmp_path / "train_results")
    driver.model_checkpoint_dir = str(tmp_path / "model_checkpoints")
    return driver


# --------------------------------------------------------------------------------------
# Config -> constructor
# --------------------------------------------------------------------------------------
def test_the_shipped_config_resolves_to_the_shipped_architecture(trainer):
    """Every architectural flag the suite's ``SHIPPED_KWARGS`` claims the config sets, it must set.

    That fixture is the suite's description of the production model; this is what keeps it honest
    against the config file itself. Note it is not a faithful copy in every field: it deliberately
    differs on the geometry (it is tiny) and on the permutation control, whose weight and period it
    tunes so a schedule is observable within a handful of test steps.

    The nested ``horizon_refine`` and ``encoder`` blocks are translated to flat constructor
    arguments, and the dilation list must arrive as a tuple (YAML has none): a list would compare
    unequal to the fixture's tuple.
    """
    kwargs = trainer._build_model_kwargs()

    for name in (
        "causal_norm", "kld_support", "lag_bias_init", "use_entmax", "head_structured_latent",
        "freeze_unused_attn_proj", "use_up_st",
        "horizon_depth", "horizon_film", "encoder_extra_dilations",
    ):
        assert kwargs[name] == SHIPPED_KWARGS[name], f"{name} disagrees with the shipped flag set"


def test_the_resolved_kwargs_actually_build_a_model(trainer):
    """The sweep's output is only correct if the constructor accepts it."""
    from teb_vae.lag_attn.nets.model import SeqVaeLagAttn

    model = SeqVaeLagAttn(**trainer._build_model_kwargs())

    assert model.causal_norm is True
    assert model.frozen_attn_proj is True  # the freeze the DDP strategy depends on
    # Every encoder norm was swapped, including the ones the extra dilated blocks bring. The count
    # itself is a property of the geometry, not of this sweep; the encoder tests pin what it means.
    assert model.n_causalized_norms > 0


def test_an_unknown_config_key_is_ignored_rather_than_forwarded(trainer):
    """The sweep forwards by name against the real signature, so a stale key cannot crash a run."""
    trainer.config["model_config"]["VAE_model"]["a_key_from_an_older_model"] = 42

    assert "a_key_from_an_older_model" not in trainer._build_model_kwargs()


def test_a_null_config_value_falls_through_to_the_constructor_default(trainer):
    """``null`` in YAML means "unset", and the constructor's default is the single source of it."""
    trainer.config["model_config"]["VAE_model"]["dropout"] = None

    assert "dropout" not in trainer._build_model_kwargs()


# --------------------------------------------------------------------------------------
# create_model
# --------------------------------------------------------------------------------------
def test_create_model_wraps_the_eager_net_in_its_task(trainer):
    from teb_vae.lag_attn.task import SeqVaeLagAttnTask

    trainer.create_model()

    assert isinstance(trainer.pl_model, SeqVaeLagAttnTask)
    assert trainer.pl_model.orig_model is trainer.pytorch_model
    assert trainer.pl_model.model is trainer.pl_model.orig_model  # eager, never compiled


def test_create_model_passes_the_spike_breaker_block_to_the_task(trainer):
    """The block is validated by the framework and read by the module -- but nothing forwards it.

    ``GraphModelBase`` never passes it on, so a driver that forgets leaves a fully-configured,
    fully-validated ``enabled: true`` block in the config doing nothing at all.
    """
    trainer.create_model()

    breaker = trainer.pl_model.hparams["spike_breaker"]
    assert breaker["enabled"] is True
    assert breaker["comparison_metric"] == "main_loss"
    assert breaker["ema_floor"] >= 1.0e9


def test_create_model_passes_the_loss_hyperparameters_to_the_task(trainer):
    """Each loss key reaches the task under its own name, except ``lag_smoothness_lambda``, which
    the task takes as ``lambda_lag`` -- exactly the kind of rename that silently resolves to a
    default of $0$."""
    trainer.create_model()

    hparams = trainer.pl_model.hparams
    vae = trainer.config["model_config"]["VAE_model"]
    for name in (
        "likelihood", "sigma_obs", "free_bits", "detach_baseline_in_full", "beta_schedule",
        "kld_beta", "lambda_full", "lambda_base",
    ):
        assert hparams[name] == vae[name], f"{name} did not reach the task"
    assert hparams["lambda_lag"] == vae["lag_smoothness_lambda"] > 0


def test_the_checkpoint_kwargs_are_the_ones_the_model_was_built_from(trainer):
    """So the blob rebuilds into this architecture and not the constructor's defaults."""
    trainer.create_model()

    assert trainer.pl_model._model_kwargs == trainer._build_model_kwargs()


def test_an_unalignable_core_checkpoint_raises_rather_than_training_from_scratch(trainer, tmp_path):
    """``load_checkpoint_strict`` returns ``None`` when nothing lines up; it does not raise.

    An unchecked call therefore trains a randomly-initialised model that was supposed to be warm
    started, and says nothing about it.
    """
    unrelated = tmp_path / "unrelated.ckpt"
    torch.save({"state_dict": {"nothing.like.this": torch.zeros(2)}}, unrelated)
    trainer.config["model_config"]["core_model_checkpoint"] = str(unrelated)

    with pytest.raises(RuntimeError, match="could not align"):
        trainer.create_model()


def test_a_core_checkpoint_from_another_model_is_refused_before_it_is_loaded(trainer, tmp_path):
    foreign = tmp_path / "foreign.ckpt"
    torch.save({"state_dict": {}, "model_class": "SeqVaeRawV4"}, foreign)
    trainer.config["model_config"]["core_model_checkpoint"] = str(foreign)

    with pytest.raises(ValueError, match="does not match the active model class"):
        trainer.create_model()


# --------------------------------------------------------------------------------------
# Tracked metrics
# --------------------------------------------------------------------------------------
def test_every_tracked_metric_is_a_name_the_framework_emits():
    """The rule, applied to this model's list.

    A bare name other than ``lr`` is renamed to ``{stage}/{name}`` on the way out and so matches
    nothing -- producing a column that is NaN for every epoch of every run, with no error. That is
    not hypothetical: it is what ``kld_beta`` did in the trainer this was ported from.
    """
    assert _unreachable_metric_names(_TRACKED_METRICS) == ()


def test_the_tracked_list_covers_what_the_task_emits(task, stub_batch, perturb_posterior):
    """Driven from the real metrics dict, so a new metric cannot be added without being tracked."""
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    tracked_suffixes = {name.split("/")[-1] for name in _TRACKED_METRICS}
    untracked = set(metrics) - tracked_suffixes
    assert untracked == set(), f"the task emits {untracked}, which no callback collects"


# --------------------------------------------------------------------------------------
# train_model wiring
# --------------------------------------------------------------------------------------
@pytest.fixture
def built_callbacks(trainer, monkeypatch):
    """The callback list ``train_model`` hands to the trainer builder, without running a fit."""
    captured = {}

    def _capture(callbacks, model=None):
        captured["callbacks"] = callbacks
        captured["model"] = model

        class _StubTrainer:
            def fit(self, *args, **kwargs):
                captured["fit_args"] = (args, kwargs)

        return _StubTrainer()

    monkeypatch.setattr(type(trainer), "build_trainer", staticmethod(_capture))
    trainer.create_model()
    trainer.train_model(object(), object())
    return captured


def test_the_model_passed_to_the_builder_is_the_lightning_module(built_callbacks, trainer):
    """And is therefore not something to read raw-model attributes off; see the strategy hook."""
    assert built_callbacks["model"] is trainer.pl_model


def test_the_checkpoint_callback_writes_to_the_run_checkpoint_directory(built_callbacks, trainer):
    """The framework hardcodes ``enable_checkpointing=True``, so a missing ``ModelCheckpoint``
    means Lightning adds its own -- writing into the train-results directory instead."""
    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoints = [cb for cb in built_callbacks["callbacks"] if isinstance(cb, ModelCheckpoint)]
    configured = trainer.config["advanced_config"]["callbacks"]["model_checkpoint"]

    assert len(checkpoints) == 1
    assert checkpoints[0].dirpath == trainer.model_checkpoint_dir
    # The monitor comes from config; hardcoding it would make the config key a decoration.
    assert checkpoints[0].monitor == configured["monitor"]
    assert checkpoints[0].save_top_k == configured["save_top_k"]
    # Lightning prefixes each placeholder with its own name, so `epoch={epoch}` renders
    # `epoch=epoch=00`.
    assert "epoch=" not in checkpoints[0].filename


def test_the_metrics_history_writer_is_wired_to_the_collector(built_callbacks):
    """The collector accumulates history and never writes it; the writer is what persists it."""
    callbacks = built_callbacks["callbacks"]
    collector = next(cb for cb in callbacks if isinstance(cb, MetricsLoggingCallback))
    writer = next(cb for cb in callbacks if isinstance(cb, MetricsHistoryCsvCallback))

    assert writer.source is collector
    assert collector.tracked_metrics == _TRACKED_METRICS


def test_the_enabled_diagnostic_plotter_is_wired_from_config(built_callbacks, trainer):
    """The shipped config enables the plotter; it must land once, under the run's results
    directory, with the configured schedule. (The tiny smoke run covers the disabled path.)"""
    from teb_vae.lag_attn.plotting import LagAttnPlotCallback

    configured = trainer.config["advanced_config"]["callbacks"]["lag_attn_plotting"]
    plotters = [cb for cb in built_callbacks["callbacks"] if isinstance(cb, LagAttnPlotCallback)]

    assert configured["enabled"] is True
    assert len(plotters) == 1
    assert plotters[0].output_dir.parent == Path(trainer.train_results_dir)
    assert plotters[0].plot_frequency == configured["plot_frequency"]
    assert plotters[0].num_examples == configured["num_examples"]


# --------------------------------------------------------------------------------------
# Pre-fit width guard
# --------------------------------------------------------------------------------------
# Derived from __file__, not cwd-relative, and not incidentally: the guard swallows a failed open
# (a missing shard is the data module's to report), so a path that does not resolve makes the two
# positive tests below pass without ever reaching the width arithmetic they exist to check.
_SHARD = str(Path(__file__).resolve().parent / "fixtures" / "tiny_shard.hdf5")


def _width_config(**vae):
    """A minimal config carrying just what the pre-fit width guard reads."""
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
    """Fails on rank 0 before the data module, not inside ``training_step``.

    The task's check against the real batch is the authoritative one; this only moves the failure
    earlier, before every rank has initialised and an MLflow run and run directory exist. The
    message must name the shard's own per-field widths, or the reader cannot tell which of the
    two numbers is wrong.
    """
    with pytest.raises(ValueError, match="channel widths disagree") as excinfo:
        _check_declared_widths_against_shard(_width_config(**declared))

    assert expected_fragment in str(excinfo.value)


@pytest.mark.parametrize(
    "declared", [dict(), dict(use_up_st=False, c_u=15)]
)
def test_widths_matching_the_shard_pass_the_pre_fit_guard(declared):
    _check_declared_widths_against_shard(_width_config(**declared))


def test_the_pre_fit_guard_defers_rather_than_masking_a_data_module_error():
    """An unreadable shard is the data module's to report -- it does so far better than a peek.

    A guard that raised here would replace ``FileNotFoundError: <path>`` with a width complaint
    about a file that does not exist.
    """
    _check_declared_widths_against_shard(
        {
            "dataset_config": {"vae_train_datasets": ["/nonexistent/shard.hdf5"]},
            "model_config": {"VAE_model": {"c_y": 109, "c_u": 58, "use_up_st": True}},
        }
    )
    # And a config that declares nothing for it to check is a no-op, not a crash.
    _check_declared_widths_against_shard({})
