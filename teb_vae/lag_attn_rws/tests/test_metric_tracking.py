r"""Every tracked metric is reachable, and everything the task emits is tracked.

Two silent failure modes meet here. A tracked name the framework never emits produces a CSV
column that is NaN for every epoch of every run -- a bare ``kld_beta`` did exactly that in the
tree this model's sibling was ported from. And an emitted metric no callback tracks simply never
reaches the CSV, so a new readout can be added, logged, and lost without any test noticing --
unless the tracked list is driven from the real metrics dict, which is what happens below.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask
from teb_vae.lag_attn_rws.trainer import _TRACKED_METRICS, LagAttnRwsTrainer
from train.callbacks import (
    MetricsHistoryCsvCallback,
    MetricsLoggingCallback,
    _unreachable_metric_names,
)

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


# --------------------------------------------------------------------------------------
# The tracked list
# --------------------------------------------------------------------------------------
def test_every_tracked_metric_is_a_name_the_framework_emits():
    """A bare name other than ``lr`` is renamed to ``{stage}/{name}`` on the way out and so
    matches nothing."""
    assert _unreachable_metric_names(_TRACKED_METRICS) == ()


def test_the_tracked_list_covers_what_the_task_emits(task, stub_batch, perturb_posterior):
    """Driven from the real metrics dicts of both stages, so a new metric cannot be added
    without being tracked."""
    module = task()
    perturb_posterior(module.orig_model)

    _, train_metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")
    _, val_metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    untracked = {f"train/{name}" for name in train_metrics} - set(_TRACKED_METRICS)
    untracked |= {f"val/{name}" for name in val_metrics} - set(_TRACKED_METRICS)
    assert untracked == set(), f"the task emits {untracked}, which no callback collects"


def test_the_shuffled_metrics_are_tracked_for_validation_only():
    """The permutation control never runs on a training batch, so a ``train/`` variant of its
    metrics would be a column that is NaN in every row of every run."""
    for name in ("nll_shuffled_block", "kld_shuffled", "shuffle_penalty"):
        assert f"val/{name}" in _TRACKED_METRICS
        assert f"train/{name}" not in _TRACKED_METRICS


def test_the_breaker_columns_are_tracked_train_only():
    """The breaker never runs on a validation batch; its columns are the only visibility into a
    run that silently skips every batch."""
    for name in ("spike_skipped", "spike_ema_loss"):
        assert f"train/{name}" in _TRACKED_METRICS
        assert f"val/{name}" not in _TRACKED_METRICS


def test_the_gradient_columns_are_tracked_train_only():
    """The pre-clip norm and the clip-exceedance fraction exist on the training path alone --
    they are logged from ``on_before_optimizer_step``, which validation never reaches -- so a
    ``val/`` variant would be a column that is NaN in every row of every run."""
    for name in ("grad_norm", "grad_clip_frac"):
        assert f"train/{name}" in _TRACKED_METRICS
        assert f"val/{name}" not in _TRACKED_METRICS


class _GradientHookStub:
    """The whole surface ``on_before_optimizer_step`` touches, and nothing else.

    Bound to the unbound method rather than built from a real task: the hook reads only
    ``self.parameters()``, ``self.trainer.gradient_clip_val`` and ``self.log``, so a stub tests the
    threshold predicate directly and without a fit.
    """

    def __init__(self, clip_val: Any) -> None:
        parameter = torch.nn.Parameter(torch.zeros(4))
        parameter.grad = torch.full((4,), 3.0)  # gradient norm exactly 6.0
        self._parameters = [parameter]
        self.trainer = SimpleNamespace(gradient_clip_val=clip_val)
        self.logged: dict = {}

    def parameters(self):
        return iter(self._parameters)

    def log(self, name: str, value: Any, **_: Any) -> None:
        self.logged[name] = float(value)


@pytest.mark.parametrize(
    "clip_val, expected_frac",
    [
        (None, None),      # no clipping configured
        (0.0, None),       # the "0 disables it" convention -- Lightning clips nothing here
        (0, None),         # and the integer spelling of the same
        (-1.0, None),      # any non-positive threshold, by the same predicate
        (5.0, 1.0),        # a real threshold the norm exceeds
        (100.0, 0.0),      # a real threshold it does not
    ],
)
def test_the_clip_fraction_is_logged_only_against_a_threshold_that_actually_clips(
    clip_val, expected_frac
):
    """``grad_clip_frac`` must be absent whenever Lightning is not clipping, not merely whenever
    the config omitted the key.

    ``Precision.clip_gradients`` returns early for a non-positive threshold, so a run configured
    with ``gradient_clip_val: 0`` rescales nothing -- while every finite norm exceeds zero. Gated
    on ``is not None`` alone, that run reports an exceedance fraction of $1.000$ in every row,
    which is precisely the reading that says the threshold rescaled every step. The norm itself is
    logged either way, because it is a property of the run rather than of the threshold.
    """
    stub = _GradientHookStub(clip_val)

    SeqVaeLagAttnRwsTask.on_before_optimizer_step(stub, optimizer=None)

    assert stub.logged["train/grad_norm"] == pytest.approx(6.0)
    if expected_frac is None:
        assert "train/grad_clip_frac" not in stub.logged
    else:
        assert stub.logged["train/grad_clip_frac"] == expected_frac


def test_beta_is_tracked_stage_prefixed_and_lr_bare():
    assert "train/kld_beta" in _TRACKED_METRICS
    assert "kld_beta" not in _TRACKED_METRICS
    assert "lr" in _TRACKED_METRICS  # the one name the framework logs unprefixed


# --------------------------------------------------------------------------------------
# train_model wiring
# --------------------------------------------------------------------------------------
@pytest.fixture
def built_callbacks(tmp_path, monkeypatch):
    """The callback list ``train_model`` hands to the trainer builder, without running a fit."""
    driver = LagAttnRwsTrainer(config_file_path=str(_CONFIG))
    driver.output_base_dir = str(tmp_path)
    driver.train_results_dir = str(tmp_path / "train_results")
    driver.model_checkpoint_dir = str(tmp_path / "model_checkpoints")

    captured = {}

    def _capture(callbacks, model=None):
        captured["callbacks"] = callbacks
        captured["model"] = model

        class _StubTrainer:
            def fit(self, *args, **kwargs):
                captured["fit_args"] = (args, kwargs)

        return _StubTrainer()

    monkeypatch.setattr(type(driver), "build_trainer", staticmethod(_capture))
    driver.create_model()
    driver.train_model(object(), object())
    captured["driver"] = driver
    return captured


def test_train_model_goes_through_the_framework_trainer_builder(built_callbacks):
    """No hand-rolled ``pl.Trainer``: the builder attaches the LR monitor and the MLflow
    run-logging callback, reconciles ``benchmark`` against ``deterministic``, and TTY-gates the
    progress bar. A hand-rolled block gets none of that."""
    assert "callbacks" in built_callbacks
    assert "fit_args" in built_callbacks  # the fit ran through the built trainer


def test_the_model_passed_to_the_builder_is_the_lightning_module(built_callbacks):
    assert built_callbacks["model"] is built_callbacks["driver"].pl_model


def test_the_checkpoint_callback_writes_to_the_run_checkpoint_directory(built_callbacks):
    """The framework hardcodes ``enable_checkpointing=True``, so a missing ``ModelCheckpoint``
    means Lightning adds its own -- writing into the train-results directory instead."""
    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoints = [cb for cb in built_callbacks["callbacks"] if isinstance(cb, ModelCheckpoint)]

    assert len(checkpoints) == 1
    assert checkpoints[0].dirpath == built_callbacks["driver"].model_checkpoint_dir


def test_the_checkpoint_monitor_comes_from_config_and_is_a_reachable_metric(
    built_callbacks, task, stub_batch, perturb_posterior
):
    """Two halves of one guarantee: the monitor is the config's, and the config names a metric
    that actually lands in ``callback_metrics`` -- a monitor nothing emits makes Lightning
    checkpoint on nothing."""
    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoint = next(
        cb for cb in built_callbacks["callbacks"] if isinstance(cb, ModelCheckpoint)
    )
    configured = built_callbacks["driver"].config["advanced_config"]["callbacks"][
        "model_checkpoint"
    ]
    assert checkpoint.monitor == configured["monitor"]
    assert checkpoint.save_top_k == configured["save_top_k"]

    stage, _, suffix = configured["monitor"].partition("/")
    module = task()
    perturb_posterior(module.orig_model)
    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, stage)
    assert suffix in metrics, f"the monitor {configured['monitor']} is a metric nothing emits"


def test_the_checkpoint_filename_does_not_double_prefix_the_epoch(built_callbacks):
    """Lightning prefixes each placeholder with its own name, so ``epoch={epoch}`` renders
    ``epoch=epoch=00``."""
    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoint = next(
        cb for cb in built_callbacks["callbacks"] if isinstance(cb, ModelCheckpoint)
    )

    assert "epoch=" not in checkpoint.filename


def test_the_metrics_history_writer_is_wired_to_the_collector(built_callbacks):
    """The collector accumulates history and never writes it; the writer persists it."""
    callbacks = built_callbacks["callbacks"]
    collector = next(cb for cb in callbacks if isinstance(cb, MetricsLoggingCallback))
    writer = next(cb for cb in callbacks if isinstance(cb, MetricsHistoryCsvCallback))

    assert writer.source is collector
    assert collector.tracked_metrics == _TRACKED_METRICS


def test_the_hyperparameter_callback_keys_are_explicit(built_callbacks):
    """The default list asks for names the framework does not emit (a bare ``kld_beta``,
    ``hyperparams/beta``); left to default, the beta ramp silently vanishes from
    hyperparameters.html."""
    from train.callbacks import HyperparameterLoggingCallback

    hyperparam = next(
        cb
        for cb in built_callbacks["callbacks"]
        if isinstance(cb, HyperparameterLoggingCallback)
    )

    assert hyperparam.tracked_keys == ("train/kld_beta", "lr")


def test_the_shipped_config_wires_the_diagnostic_plotter(built_callbacks):
    """``lag_attn_rws_plotting.enabled: true`` ships, so a real run emits the validation figure.
    The plotter is imported lazily inside the enabled branch, so this is also the only test that
    would catch that import breaking."""
    names = [type(cb).__name__ for cb in built_callbacks["callbacks"]]

    assert "LagAttnRwsPlotCallback" in names


def test_disabling_the_plotting_block_constructs_no_plotter(tmp_path, monkeypatch):
    """The other direction, and the reason the import sits inside the branch: a wiring that
    ignored the flag would pull matplotlib into every run that asked for no figures."""
    driver = LagAttnRwsTrainer(config_file_path=str(_CONFIG))
    driver.output_base_dir = str(tmp_path)
    driver.train_results_dir = str(tmp_path / "train_results")
    driver.model_checkpoint_dir = str(tmp_path / "model_checkpoints")
    driver.config["advanced_config"]["callbacks"]["lag_attn_rws_plotting"]["enabled"] = False

    captured = {}

    def _capture(callbacks, model=None):
        captured["callbacks"] = callbacks

        class _StubTrainer:
            def fit(self, *args, **kwargs):
                return None

        return _StubTrainer()

    monkeypatch.setattr(type(driver), "build_trainer", staticmethod(_capture))
    driver.create_model()
    driver.train_model(object(), object())

    names = [type(cb).__name__ for cb in captured["callbacks"]]
    assert "LagAttnRwsPlotCallback" not in names
