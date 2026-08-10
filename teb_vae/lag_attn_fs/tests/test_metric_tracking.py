r"""Every tracked metric is reachable, everything the task emits is tracked, and the callbacks wire.

Two silent failure modes meet here. A tracked name the framework never emits produces a CSV column
that is NaN for every epoch of every run. And an emitted metric no callback tracks simply never
reaches the CSV, so a readout can be added, logged, and lost without any test noticing -- unless
the tracked list is driven from the real metrics dict, which is what happens below.

That second mode is why this file matters more here than in either sibling. This model adds four
columns for one reason: with the evaluation pipeline deferred, every other readout is a scalar
summed over $H \cdot C_{\mathrm{keep}} = 2340$ coefficients, and one scalar cannot tell a model
that forecasts from one that reconstructs the part of the target its own history already fixes.
Those four are the only evidence a multi-day run will produce on that question, so a wiring mistake
that dropped them would not be noticed until the run was over.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

from teb_vae.lag_attn_fs.trainer import _FORECAST_GAP_SUFFIXES, LagAttnFsTrainer
from teb_vae.lag_attn_rws.trainer import _TRACKED_METRICS, LagAttnRwsTrainer
from train.callbacks import (
    HyperparameterLoggingCallback,
    MetricsHistoryCsvCallback,
    MetricsLoggingCallback,
    _unreachable_metric_names,
)

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

#: The config block the *inherited* callback assembly reads to decide whether to attach the
#: per-epoch diagnostic figure. Spelled with the comparison model's name because that is the driver
#: that reads it; renaming the block in this package's config would disable the figure in silence.
PLOTTING_BLOCK = "lag_attn_rws_plotting"

#: Names logged through a hook rather than returned in a metrics dict, and so absent from one: the
#: two spike-breaker columns the base's training step injects, the ``lr`` it logs at epoch start,
#: and the pre-clip gradient norm plus its clip-exceedance indicator the task logs in
#: ``on_before_optimizer_step`` -- which is the only place those quantities exist at all.
LOGGED_THROUGH_A_HOOK = {
    "train/spike_skipped",
    "train/spike_ema_loss",
    "lr",
    "train/grad_norm",
    "train/grad_clip_frac",
}


def _callbacks_from(config_path, tmp_path, monkeypatch, mutate=None):
    """Return the callback list ``train_model`` hands to the trainer builder, without a fit.

    Args:
        config_path: The config to drive.
        tmp_path: Directory the run's output paths are redirected under.
        monkeypatch: The pytest fixture.
        mutate: Optional callable applied to the driver's config before the model is built.

    Returns:
        ``(driver, captured)``.
    """
    driver = LagAttnFsTrainer(config_file_path=str(config_path))
    driver.output_base_dir = str(tmp_path)
    driver.train_results_dir = str(tmp_path / "train_results")
    driver.model_checkpoint_dir = str(tmp_path / "model_checkpoints")
    if mutate is not None:
        mutate(driver.config)

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
    return driver, captured


@pytest.fixture
def built(tmp_path, monkeypatch):
    """The driver and the callback list the shipped config produces."""
    driver, captured = _callbacks_from(_CONFIG, tmp_path, monkeypatch)
    return {"driver": driver, **captured}


# --------------------------------------------------------------------------------------
# The tracked list
# --------------------------------------------------------------------------------------
def test_every_tracked_metric_is_a_name_the_framework_emits():
    """A bare name other than ``lr`` is renamed to ``{stage}/{name}`` on the way out and so matches
    nothing. Checked on this driver's list, not the inherited one: the four additions are hand-
    written names and this is what says they are spelled the way the framework will emit them."""
    assert _unreachable_metric_names(LagAttnFsTrainer.TRACKED_METRICS) == ()


def test_the_tracked_list_covers_what_this_task_emits(task, stub_batch, perturb_posterior):
    """Driven from the real metrics dicts of both stages, so a metric this model emits and the list
    does not know about cannot slip through."""
    module = task()
    perturb_posterior(module.orig_model)

    _, train_metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")
    _, val_metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    tracked = set(LagAttnFsTrainer.TRACKED_METRICS)
    untracked = {f"train/{name}" for name in train_metrics} - tracked
    untracked |= {f"val/{name}" for name in val_metrics} - tracked
    assert untracked == set(), f"the task emits {untracked}, which no callback collects"


def test_every_tracked_metric_is_emitted_by_something(task, stub_batch, perturb_posterior):
    """The other direction, and the one that catches an addition to the list that nothing produces
    -- a column that is NaN in every row of every run, forever."""
    module = task()
    perturb_posterior(module.orig_model)

    _, train_metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")
    _, val_metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")
    emitted = {f"train/{name}" for name in train_metrics}
    emitted |= {f"val/{name}" for name in val_metrics}
    emitted |= LOGGED_THROUGH_A_HOOK

    assert set(LagAttnFsTrainer.TRACKED_METRICS) - emitted == set()


def test_the_inherited_list_alone_would_lose_the_four_forecast_gap_columns(
    task, stub_batch, perturb_posterior
):
    """The negative control on the ``TRACKED_METRICS`` seam. Without this driver's extension the
    four columns are emitted, logged and never collected -- which is silent, and which is the whole
    reason the seam exists rather than the list being read as a module global."""
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    lost = {f"train/{name}" for name in metrics} - set(_TRACKED_METRICS)
    assert lost == {f"train/{name}" for name in _FORECAST_GAP_SUFFIXES}
    assert lost <= set(LagAttnFsTrainer.TRACKED_METRICS)


def test_the_four_columns_are_tracked_on_both_stages():
    """They come out of ``compute_loss``, like every other term of the objective, so they exist on
    training and validation batches alike. A ``val/``-only entry would silently halve the evidence
    a run produces on the only question this model can currently answer."""
    for name in _FORECAST_GAP_SUFFIXES:
        assert f"train/{name}" in LagAttnFsTrainer.TRACKED_METRICS, name
        assert f"val/{name}" in LagAttnFsTrainer.TRACKED_METRICS, name


def test_the_gradient_norm_column_is_tracked_train_only():
    """It is what ``gradient_clip_val`` gets re-derived from, and it exists on the training path
    alone -- a ``val/`` variant would be NaN in every row of every run."""
    assert "train/grad_norm" in LagAttnFsTrainer.TRACKED_METRICS
    assert "val/grad_norm" not in LagAttnFsTrainer.TRACKED_METRICS


# --------------------------------------------------------------------------------------
# train_model wiring
# --------------------------------------------------------------------------------------
def test_train_model_goes_through_the_framework_trainer_builder(built):
    """No hand-rolled ``Trainer``: the builder attaches the learning-rate monitor and the MLflow
    run-logging callback, reconciles ``benchmark`` against ``deterministic``, and TTY-gates the
    progress bar. A hand-rolled block gets none of that."""
    assert "callbacks" in built
    assert "fit_args" in built  # the fit ran through the built trainer
    assert built["model"] is built["driver"].pl_model


def test_the_metrics_collector_is_given_this_drivers_list_and_not_the_module_global(built):
    """The seam, at the one site that consumes it. ``train_model`` is inherited whole, so if it
    still read the module global this assertion is the only thing between that and four columns
    that never appear."""
    collector = next(cb for cb in built["callbacks"] if isinstance(cb, MetricsLoggingCallback))

    assert collector.tracked_metrics == LagAttnFsTrainer.TRACKED_METRICS
    assert collector.tracked_metrics != _TRACKED_METRICS


def test_the_comparison_models_collector_is_unchanged():
    """The seam defaults to today's literal, so adding it must not have moved the comparison
    model's own column set -- every prior run's CSV is read against that list."""
    assert LagAttnRwsTrainer.TRACKED_METRICS == _TRACKED_METRICS


def test_the_metrics_history_writer_is_wired_to_the_collector(built):
    """The collector accumulates history and never writes it; the writer persists it."""
    collector = next(cb for cb in built["callbacks"] if isinstance(cb, MetricsLoggingCallback))
    writer = next(cb for cb in built["callbacks"] if isinstance(cb, MetricsHistoryCsvCallback))

    assert writer.source is collector


def test_the_checkpoint_callback_writes_to_the_run_checkpoint_directory(built):
    """The framework hardcodes ``enable_checkpointing=True``, so a missing ``ModelCheckpoint``
    means Lightning adds its own -- writing into the train-results directory instead."""
    checkpoints = [cb for cb in built["callbacks"] if isinstance(cb, ModelCheckpoint)]

    assert len(checkpoints) == 1
    assert checkpoints[0].dirpath == built["driver"].model_checkpoint_dir


def test_the_checkpoint_filename_carries_this_models_stem_and_no_double_prefix(built):
    """Lightning prefixes each placeholder with its own name, so a stem containing ``epoch=``
    renders as ``epoch=epoch=00``. And the stem must be this model's: two models writing
    ``lag-attn-rws-epoch=00.ckpt`` into a shared directory would be indistinguishable by name."""
    checkpoint = next(cb for cb in built["callbacks"] if isinstance(cb, ModelCheckpoint))
    filename = str(checkpoint.filename)

    assert "epoch=" not in filename
    assert filename == "lag-attn-fs-{epoch:02d}"


def test_the_checkpoint_monitor_comes_from_config_and_is_a_metric_the_task_emits(
    built, task, stub_batch, perturb_posterior
):
    """Two halves of one guarantee: the monitor is the config's, and the config names a metric that
    actually lands in ``callback_metrics`` -- a monitor nothing emits makes Lightning checkpoint on
    nothing."""
    checkpoint = next(cb for cb in built["callbacks"] if isinstance(cb, ModelCheckpoint))
    configured = built["driver"].config["advanced_config"]["callbacks"]["model_checkpoint"]

    assert checkpoint.monitor == configured["monitor"]
    assert checkpoint.save_top_k == configured["save_top_k"]

    stage, _, suffix = configured["monitor"].partition("/")
    module = task()
    perturb_posterior(module.orig_model)
    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, stage)
    assert suffix in metrics, f"the monitor {configured['monitor']} is a metric nothing emits"


def test_the_hyperparameter_callback_keys_are_explicit(built):
    """The default list asks for names the framework does not emit (a bare ``kld_beta``), and left
    to default every series is NaN -- so the beta ramp, the knob this target domain retuned,
    silently vanishes from the report."""
    hyperparam = next(
        cb for cb in built["callbacks"] if isinstance(cb, HyperparameterLoggingCallback)
    )

    assert hyperparam.tracked_keys == ("train/kld_beta", "lr")


# --------------------------------------------------------------------------------------
# The diagnostic figure, and the block name that switches it on
# --------------------------------------------------------------------------------------
def test_the_plotting_block_name_is_the_inherited_drivers_literal():
    """The callback assembly is inherited, so the key it reads is that module's spelling. A config
    that renamed the block to match this package would leave the figure permanently off, with the
    flag still reading ``enabled: true`` to anyone looking at the config."""
    assert (
        LagAttnFsTrainer.PLOT_CONFIG_KEY == PLOTTING_BLOCK == LagAttnRwsTrainer.PLOT_CONFIG_KEY
    )
    config = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    assert PLOTTING_BLOCK in config["advanced_config"]["callbacks"]


def test_the_shipped_config_wires_the_diagnostic_plotter(built):
    """The plotter is imported lazily inside the enabled branch, so this is also the only test that
    would catch that import breaking. The callback itself is the comparison model's -- this package
    writes no plotting module, and the page seam is reached through the task."""
    names = [type(cb).__name__ for cb in built["callbacks"]]

    assert "LagAttnRwsPlotCallback" in names


def test_disabling_the_plotting_block_constructs_no_plotter(tmp_path, monkeypatch):
    """The other direction, and the reason the import sits inside the branch: a wiring that ignored
    the flag would pull matplotlib into every run that asked for no figures."""

    def _disable(config):
        config["advanced_config"]["callbacks"][PLOTTING_BLOCK]["enabled"] = False

    _, captured = _callbacks_from(_CONFIG, tmp_path, monkeypatch, _disable)

    names = [type(cb).__name__ for cb in captured["callbacks"]]
    assert "LagAttnRwsPlotCallback" not in names
