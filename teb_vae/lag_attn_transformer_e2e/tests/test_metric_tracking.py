r"""Every tracked metric is reachable, everything this task emits is tracked, and the callbacks the
shipped config wires watch metrics this task actually emits.

Two silent failure modes meet here. A tracked name the framework never emits produces a CSV column
that is NaN for every epoch of every run. And an emitted metric no callback tracks simply never
reaches the CSV, so a readout can be added, logged, and lost without any test noticing -- unless the
tracked list is driven from the real metrics dict, which is what happens below. There is no
evaluation pipeline for this architecture, so the CSV is the only readout a run produces.

The callback assembly itself is inherited and tested in ``lag_attn_rws``. What is this package's is
the checkpoint stem it supplies, and whether the monitor and the spike breaker name metrics *this*
task emits. The stem walks into a known trap: Lightning auto-prefixes each placeholder with its own
name, so a stem containing ``epoch=`` renders as ``epoch=epoch=00``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint

from teb_vae.lag_attn_rws.trainer import _TRACKED_METRICS
from teb_vae.lag_attn_transformer_e2e.trainer import LagAttnTrfE2ETrainer

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


# --------------------------------------------------------------------------------------
# The tracked list
# --------------------------------------------------------------------------------------
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
    clip-exceedance indicator the task logs in ``on_before_optimizer_step``."""
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


# --------------------------------------------------------------------------------------
# train_model wiring
# --------------------------------------------------------------------------------------
def _callbacks_from(config_path, tmp_path, monkeypatch):
    """Return the callback list ``train_model`` hands to the trainer builder, without a fit.

    Args:
        config_path: The config to drive.
        tmp_path: Directory the run's output paths are redirected under.
        monkeypatch: The pytest fixture.

    Returns:
        ``(driver, captured)``.
    """
    driver = LagAttnTrfE2ETrainer(config_file_path=str(config_path))
    driver.output_base_dir = str(tmp_path)
    driver.train_results_dir = str(tmp_path / "train_results")
    driver.model_checkpoint_dir = str(tmp_path / "model_checkpoints")

    captured = {}

    def _capture(callbacks, model=None):
        captured["callbacks"] = callbacks

        class _StubTrainer:
            def fit(self, *args, **kwargs):
                pass

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


def test_the_checkpoint_filename_carries_this_models_stem_and_no_double_prefix(built):
    """Lightning prefixes each placeholder with its own name, so a stem containing ``epoch=``
    renders as ``epoch=epoch=00``. And the stem must name this model: three architectures writing
    the same stem into a shared directory would be indistinguishable by name."""
    checkpoint = next(cb for cb in built["callbacks"] if isinstance(cb, ModelCheckpoint))
    filename = str(checkpoint.filename)

    assert "epoch=" not in filename
    assert filename.startswith(LagAttnTrfE2ETrainer.CHECKPOINT_STEM)
    assert "e2e" in filename


def test_the_checkpoint_monitor_comes_from_config_and_is_a_metric_the_task_emits(
    built, task, stub_batch, perturb_posterior
):
    """Two halves of one guarantee: the monitor is the config's, and the config names a metric that
    actually lands in ``callback_metrics`` -- a monitor nothing emits makes Lightning checkpoint on
    nothing."""
    checkpoint = next(cb for cb in built["callbacks"] if isinstance(cb, ModelCheckpoint))
    configured = built["driver"].config["advanced_config"]["callbacks"]["model_checkpoint"]

    assert checkpoint.monitor == configured["monitor"]

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
    breaker watching a quantity it never sees."""
    configured = built["driver"].config["advanced_config"]["spike_breaker"]["comparison_metric"]
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert "/" not in configured
    assert configured in metrics
