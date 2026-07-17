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
from teb_vae.lag_attn.trainer import _TRACKED_METRICS, LagAttnTrainer
from train.callbacks import (
    MetricsHistoryCsvCallback,
    MetricsLoggingCallback,
    _unreachable_metric_names,
)

_V3 = Path(__file__).resolve().parents[1] / "configs" / "v3.yaml"


@pytest.fixture
def trainer(tmp_path):
    """A driver on the shipped config, with its output directories redirected under ``tmp_path``.

    ``setup_config`` is never called -- it would seed, open log sinks and probe MLflow -- so the
    directories are assigned directly.
    """
    driver = LagAttnTrainer(config_file_path=str(_V3))
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
    tunes so a schedule is observable within a handful of test steps. Those are asserted against
    the config's real values below rather than against the fixture.
    """
    kwargs = trainer._build_model_kwargs()

    for name in (
        "causal_norm", "kld_support", "lag_bias_init", "use_entmax", "head_structured_latent",
        "freeze_unused_attn_proj", "use_up_st",
    ):
        assert kwargs[name] == SHIPPED_KWARGS[name], f"{name} disagrees with the shipped flag set"


def test_the_permutation_control_ships_as_a_readout(trainer):
    r"""$\lambda_{\mathrm{perm}} = 0$ is the shipped value, and it is a measured decision.

    A positive weight can only suppress $K_{\mathrm{shuffled}}$ by teaching the posterior to ignore
    mismatched sources, which competes with using the source at all; it collapsed the source
    pathway outright in half the seeds it was tried on. The control still runs -- it is the
    readout -- it just does not enter the loss.
    """
    kwargs = trainer._build_model_kwargs()

    assert kwargs["lambda_perm"] == 0.0
    assert kwargs["perm_every_n_batches"] == 4


def test_the_nested_blocks_are_translated_to_flat_constructor_arguments(trainer):
    """The config groups them for readability; the constructor takes them flat."""
    kwargs = trainer._build_model_kwargs()

    assert kwargs["horizon_depth"] == 3
    assert kwargs["horizon_kernel"] == 3
    assert kwargs["horizon_film"] is True
    assert kwargs["encoder_extra_dilations"] == (8, 16)


def test_extra_dilations_arrive_as_a_tuple(trainer):
    """YAML has no tuple, and the encoder's dilation list is a fixed structural property."""
    assert isinstance(trainer._build_model_kwargs()["encoder_extra_dilations"], tuple)


def test_logvar_clamp_arrives_as_a_pair(trainer):
    kwargs = trainer._build_model_kwargs()

    assert kwargs["logvar_clamp"] == (-5.0, 3.0)
    assert isinstance(kwargs["logvar_clamp"], tuple)


def test_the_geometry_reaches_the_constructor(trainer):
    kwargs = trainer._build_model_kwargs()

    assert kwargs["sequence_length"] == 300
    assert kwargs["d_model"] == 128
    assert kwargs["d_z"] == 24
    assert kwargs["c_y"] == 87
    assert kwargs["c_u"] == 101
    assert kwargs["max_lag"] == 90


def test_loss_only_keys_do_not_reach_the_constructor(trainer):
    """The net takes tensors and computes a loss on request; it owns none of these.

    The constructor is keyword-only with no ``**kwargs``, so a leaked key is a ``TypeError`` rather
    than a silent mis-build -- but it would be a ``TypeError`` on the production config, which is a
    poor place to find out.
    """
    kwargs = trainer._build_model_kwargs()

    for name in (
        "likelihood", "sigma_obs", "free_bits", "lambda_full", "lambda_base", "beta_schedule",
        "kld_beta", "detach_baseline_in_full", "lag_smoothness_lambda", "horizon_refine", "encoder",
    ):
        assert name not in kwargs, f"{name} is the task's, not the net's"


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
def test_create_model_wraps_the_net_in_its_task(trainer):
    from teb_vae.lag_attn.task import SeqVaeLagAttnTask

    trainer.create_model()

    assert isinstance(trainer.pl_model, SeqVaeLagAttnTask)
    assert trainer.pl_model.orig_model is trainer.pytorch_model


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
    trainer.create_model()

    hparams = trainer.pl_model.hparams
    assert hparams["likelihood"] == "gaussian_nll"
    assert hparams["sigma_obs"] == "learned"
    assert hparams["free_bits"] == 0.1
    assert hparams["detach_baseline_in_full"] is True
    assert hparams["beta_schedule"]["kind"] == "linear_warmup"


def test_the_lag_smoothness_weight_is_renamed_on_the_way_in(trainer):
    """The config calls it ``lag_smoothness_lambda``; the task's argument is ``lambda_lag``.

    A rename is exactly the kind of thing that silently resolves to a default of 0.0.
    """
    trainer.create_model()

    assert trainer.pl_model.hparams["lambda_lag"] == 1.0e-3


def test_create_model_forces_eager_execution(trainer):
    trainer.create_model()

    assert trainer.pl_model.model is trainer.pl_model.orig_model


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


def test_beta_is_tracked_stage_prefixed():
    """The specific column that was silently NaN before."""
    assert "train/kld_beta" in _TRACKED_METRICS
    assert "kld_beta" not in _TRACKED_METRICS


def test_lr_is_tracked_bare():
    """The one name the framework does log unprefixed."""
    assert "lr" in _TRACKED_METRICS


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


def test_train_model_goes_through_the_framework_trainer_builder(built_callbacks):
    """No hand-rolled ``pl.Trainer``.

    The builder is what attaches the LR monitor and the MLflow run-logging callback, reconciles
    ``benchmark`` against ``deterministic``, and TTY-gates the progress bar. A hand-rolled block
    gets none of that and drifts from the config table besides.
    """
    assert "callbacks" in built_callbacks


def test_the_model_passed_to_the_builder_is_the_lightning_module(built_callbacks, trainer):
    """And is therefore not something to read raw-model attributes off; see the strategy hook."""
    assert built_callbacks["model"] is trainer.pl_model


def test_the_checkpoint_callback_writes_to_the_run_checkpoint_directory(built_callbacks, trainer):
    """The framework hardcodes ``enable_checkpointing=True``, so a missing ``ModelCheckpoint``
    means Lightning adds its own -- writing into the train-results directory instead."""
    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoints = [cb for cb in built_callbacks["callbacks"] if isinstance(cb, ModelCheckpoint)]

    assert len(checkpoints) == 1
    assert checkpoints[0].dirpath == trainer.model_checkpoint_dir


def test_the_checkpoint_monitor_comes_from_config(built_callbacks, trainer):
    """Hardcoding it would make the config key a decoration, which is what it was before."""
    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoint = next(cb for cb in built_callbacks["callbacks"] if isinstance(cb, ModelCheckpoint))
    configured = trainer.config["advanced_config"]["callbacks"]["model_checkpoint"]

    assert checkpoint.monitor == configured["monitor"]
    assert checkpoint.save_top_k == configured["save_top_k"]


def test_the_checkpoint_filename_does_not_double_prefix_the_epoch(built_callbacks):
    """Lightning prefixes each placeholder with its own name, so ``epoch={epoch}`` renders
    ``epoch=epoch=00``."""
    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoint = next(cb for cb in built_callbacks["callbacks"] if isinstance(cb, ModelCheckpoint))

    assert "epoch=" not in checkpoint.filename


def test_the_metrics_history_writer_is_wired_to_the_collector(built_callbacks):
    """The collector accumulates history and never writes it; the writer is what persists it."""
    callbacks = built_callbacks["callbacks"]
    collector = next(cb for cb in callbacks if isinstance(cb, MetricsLoggingCallback))
    writer = next(cb for cb in callbacks if isinstance(cb, MetricsHistoryCsvCallback))

    assert writer.source is collector
    assert collector.tracked_metrics == _TRACKED_METRICS
