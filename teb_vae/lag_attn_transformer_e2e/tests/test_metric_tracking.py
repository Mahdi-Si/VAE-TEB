r"""Every tracked metric is reachable, everything the task emits is tracked, and the callbacks wire.

Two silent failure modes meet here. A tracked name the framework never emits produces a CSV column
that is NaN for every epoch of every run. And an emitted metric no callback tracks simply never
reaches the CSV, so a readout can be added, logged, and lost without any test noticing -- unless the
tracked list is driven from the real metrics dict, which is what happens below.

This matters more here than it does for either sibling. There is no evaluation pipeline for this
architecture yet, so ``metrics_history.csv``, the tracked metric surface, ``train/grad_norm`` and
the per-epoch diagnostic figure are the **only** readout a run produces: a column that silently
never fills is not an inconvenience, it is a missing result.

The tracked list itself is inherited, unchanged, because the metric surface is unchanged -- the task
is the comparison model's task plus one input builder. That makes drift here a question about *this*
model's metrics dict rather than about a list, which is why the coverage assertions drive the real
task rather than restating what it ought to emit.

The checkpoint filename gets its own assertion because the stem is the one string this package
supplies to the inherited callback assembly, and it walks into a known trap: Lightning auto-prefixes
each placeholder with its own name, so a stem containing ``epoch=`` renders as ``epoch=epoch=00``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

from teb_vae.lag_attn_rws.trainer import _TRACKED_METRICS, LagAttnRwsTrainer
from teb_vae.lag_attn_transformer_e2e.trainer import LagAttnTrfE2ETrainer
from train.callbacks import (
    HyperparameterLoggingCallback,
    MetricsHistoryCsvCallback,
    MetricsLoggingCallback,
    _unreachable_metric_names,
)

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

#: The config block the *inherited* callback assembly reads to decide whether to attach the
#: per-epoch diagnostic figure. Spelled with the raw-signal model's name because that is the driver
#: that reads it; renaming the block in this package's config would disable the figure in silence.
PLOTTING_BLOCK = "lag_attn_rws_plotting"


# --------------------------------------------------------------------------------------
# The tracked list
# --------------------------------------------------------------------------------------
def test_every_tracked_metric_is_a_name_the_framework_emits():
    """A bare name other than ``lr`` is renamed to ``{stage}/{name}`` on the way out and so
    matches nothing."""
    assert _unreachable_metric_names(_TRACKED_METRICS) == ()


def test_the_tracked_list_covers_what_this_task_emits(task, stub_batch, perturb_posterior):
    """Driven from the real metrics dicts of both stages, so a metric this architecture emits and
    the inherited list does not know about cannot slip through."""
    module = task()
    perturb_posterior(module.orig_model)

    _, train_metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")
    _, val_metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    untracked = {f"train/{name}" for name in train_metrics} - set(_TRACKED_METRICS)
    untracked |= {f"val/{name}" for name in val_metrics} - set(_TRACKED_METRICS)
    assert untracked == set(), f"the task emits {untracked}, which no callback collects"


def test_every_tracked_metric_is_emitted_by_something(task, stub_batch, perturb_posterior):
    """The other direction. Five names are logged through a hook rather than returned in a metrics
    dict and so cannot appear in one: the two spike-breaker columns the base's training step
    injects, the ``lr`` it logs at epoch start, and the pre-clip gradient norm plus its
    clip-exceedance indicator the task logs in ``on_before_optimizer_step`` -- which is the only
    place those quantities exist at all."""
    logged_through_a_hook = {
        "train/spike_skipped", "train/spike_ema_loss", "lr", "train/grad_norm",
        "train/grad_clip_frac",
    }
    module = task()
    perturb_posterior(module.orig_model)

    _, train_metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")
    _, val_metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")
    emitted = {f"train/{name}" for name in train_metrics}
    emitted |= {f"val/{name}" for name in val_metrics}
    emitted |= logged_through_a_hook

    assert set(_TRACKED_METRICS) - emitted == set()


def test_the_gradient_norm_column_is_tracked_train_only():
    """It is what this architecture's inherited-and-provisional ``gradient_clip_val`` gets
    re-derived from -- the front ends are a gradient path the run it was measured on did not have
    -- and it exists on the training path alone, so a ``val/`` variant would be NaN in every row."""
    assert "train/grad_norm" in _TRACKED_METRICS
    assert "train/grad_clip_frac" in _TRACKED_METRICS
    assert "val/grad_norm" not in _TRACKED_METRICS


# --------------------------------------------------------------------------------------
# train_model wiring
# --------------------------------------------------------------------------------------
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
    driver = LagAttnTrfE2ETrainer(config_file_path=str(config_path))
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


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The driver and the callback list the shipped config produces.

    Module-scoped: it builds the full production model, and every assertion below is a different
    question about the same object. ``monkeypatch`` is function-scoped, so the one patch this needs
    is applied and undone by hand.
    """
    from _pytest.monkeypatch import MonkeyPatch

    patcher = MonkeyPatch()
    try:
        driver, captured = _callbacks_from(
            _CONFIG, tmp_path_factory.mktemp("wiring"), patcher
        )
    finally:
        patcher.undo()
    return {"driver": driver, **captured}


def test_train_model_goes_through_the_framework_trainer_builder(built):
    """No hand-rolled ``Trainer``: the builder attaches the learning-rate monitor and the MLflow
    run-logging callback, reconciles ``benchmark`` against ``deterministic``, and TTY-gates the
    progress bar. A hand-rolled block gets none of that."""
    assert "callbacks" in built
    assert "fit_args" in built  # the fit ran through the built trainer
    assert built["model"] is built["driver"].pl_model


def test_the_checkpoint_callback_writes_to_the_run_checkpoint_directory(built):
    """The framework hardcodes ``enable_checkpointing=True``, so a missing ``ModelCheckpoint``
    means Lightning adds its own -- writing into the train-results directory instead."""
    checkpoints = [cb for cb in built["callbacks"] if isinstance(cb, ModelCheckpoint)]

    assert len(checkpoints) == 1
    assert checkpoints[0].dirpath == built["driver"].model_checkpoint_dir


def test_the_checkpoint_filename_carries_this_models_stem_and_no_double_prefix(built):
    """Lightning prefixes each placeholder with its own name, so a stem containing ``epoch=``
    renders as ``epoch=epoch=00``. And the stem must be this model's: three architectures writing
    ``lag-attn-rws-epoch=00.ckpt`` into a shared directory would be indistinguishable by name."""
    checkpoint = next(cb for cb in built["callbacks"] if isinstance(cb, ModelCheckpoint))
    filename = str(checkpoint.filename)

    assert "epoch=" not in filename
    assert filename == "lag-attn-trf-e2e-{epoch:02d}"


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


def test_the_spike_breakers_comparison_metric_is_a_metric_the_task_emits(
    built, task, stub_batch, perturb_posterior
):
    """The breaker watches ``metrics[comparison_metric]`` by exact, unprefixed name and falls back
    to the returned loss *silently* when it is absent -- so a misnamed key leaves a fully configured
    breaker watching a quantity it never sees, and this repository has already lost a run to a
    breaker behaving unexpectedly."""
    configured = built["driver"].config["advanced_config"]["spike_breaker"]["comparison_metric"]
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert "/" not in configured
    assert configured in metrics


def test_the_metrics_history_writer_is_wired_to_the_collector(built):
    """The collector accumulates history and never writes it; the writer persists it."""
    collector = next(cb for cb in built["callbacks"] if isinstance(cb, MetricsLoggingCallback))
    writer = next(cb for cb in built["callbacks"] if isinstance(cb, MetricsHistoryCsvCallback))

    assert writer.source is collector
    assert collector.tracked_metrics == _TRACKED_METRICS


def test_the_hyperparameter_callback_keys_are_explicit(built):
    """The default list asks for names the framework does not emit (a bare ``kld_beta``), and left
    to default every series is NaN -- so the beta ramp silently vanishes from the report."""
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
    flag still reading ``enabled: true`` to anyone looking at the config -- and the figure is one of
    the four readouts a run of this model has.

    Asserted on the resolved attribute rather than on ``train_model``'s source: this driver could
    also silence its own figure by overriding the key, which no reading of the inherited method
    would show."""
    assert LagAttnTrfE2ETrainer.PLOT_CONFIG_KEY == PLOTTING_BLOCK == LagAttnRwsTrainer.PLOT_CONFIG_KEY
    config = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    assert PLOTTING_BLOCK in config["advanced_config"]["callbacks"]


def test_the_shipped_config_wires_the_diagnostic_plotter(built):
    """The plotter is imported lazily inside the enabled branch, so this is also the only test that
    would catch that import breaking."""
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
