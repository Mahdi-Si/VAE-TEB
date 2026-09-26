r"""Tracked and emitted metrics agree both ways, and this driver's own seams wire.

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

The callback assembly itself (trainer builder, checkpoint directory, history writer, the plotting
flag) is inherited from the raw sibling and pinned in its suite; what is checked here is this
driver's list, its checkpoint stem and monitor, and that the shipped config switches the inherited
diagnostic figure on.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint

from teb_vae.lag_attn_fs.trainer import _FORECAST_GAP_SUFFIXES, LagAttnFsTrainer
from teb_vae.lag_attn_rws.trainer import _TRACKED_METRICS
from train.callbacks import HyperparameterLoggingCallback, MetricsLoggingCallback

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

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


def _callbacks_from(config_path, tmp_path, monkeypatch):
    """Return the callback list ``train_model`` hands to the trainer builder, without a fit.

    Args:
        config_path: The config to drive.
        tmp_path: Directory the run's output paths are redirected under.
        monkeypatch: The pytest fixture.

    Returns:
        ``(driver, captured)``.
    """
    driver = LagAttnFsTrainer(config_file_path=str(config_path))
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
    return driver, captured


@pytest.fixture
def built(tmp_path, monkeypatch):
    """The driver and the callback list the shipped config produces."""
    driver, captured = _callbacks_from(_CONFIG, tmp_path, monkeypatch)
    return {"driver": driver, **captured}


# --------------------------------------------------------------------------------------
# The tracked list
# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
# train_model wiring
# --------------------------------------------------------------------------------------
def test_the_metrics_collector_is_given_this_drivers_list_and_not_the_module_global(built):
    """The seam, at the one site that consumes it. ``train_model`` is inherited whole, so if it
    still read the module global this assertion is the only thing between that and four columns
    that never appear."""
    collector = next(cb for cb in built["callbacks"] if isinstance(cb, MetricsLoggingCallback))

    assert collector.tracked_metrics == LagAttnFsTrainer.TRACKED_METRICS
    assert collector.tracked_metrics != _TRACKED_METRICS


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
# The diagnostic figure
# --------------------------------------------------------------------------------------
def test_the_shipped_config_wires_the_diagnostic_plotter(built):
    """The plotter is imported lazily inside the enabled branch, so this is also the only test that
    would catch that import breaking. The callback itself is the comparison model's -- this package
    writes no plotting module, and the page seam is reached through the task."""
    names = [type(cb).__name__ for cb in built["callbacks"]]

    assert "LagAttnRwsPlotCallback" in names
