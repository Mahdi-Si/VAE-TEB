r"""The tracked metric list is exactly what this task emits, and the driver hands it to the collector.

Two silent failure modes meet here. A tracked name the framework never emits produces a CSV column
that is NaN for every epoch of every run. And an emitted metric no callback tracks simply never
reaches the CSV, so a readout can be added, logged, and lost without any test noticing -- unless the
tracked list is driven from the real metrics dict, which is what happens below.

The seam half checks the one site that consumes the list: ``train_model`` is inherited whole, so the
collector must be built from this driver's ``TRACKED_METRICS`` rather than the module global, and the
configured checkpoint monitor must name a metric the task actually emits.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer
from teb_vae.lag_attn_rws.trainer import _TRACKED_METRICS
from train.callbacks import MetricsLoggingCallback

from .conftest import absolutize_dataset_paths

_TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"

#: Names logged through a hook rather than returned in a metrics dict, and so absent from one.
LOGGED_THROUGH_A_HOOK = {
    "train/spike_skipped",
    "train/spike_ema_loss",
    "lr",
    "train/grad_norm",
    "train/grad_clip_frac",
}


def _config_at(tmp_path) -> Path:
    """Write a path-absolutised copy of the tiny config, which is the one that reads real shards."""
    from teb_vae.lag_attn.config import load_config

    config = absolutize_dataset_paths(load_config(str(_TINY)))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _callbacks_from(config_path, tmp_path, monkeypatch):
    """Return the callback list ``train_model`` hands to the trainer builder, without a fit."""
    driver = LagAttnCrwsTrainer(config_file_path=str(config_path))
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
    """The driver and the callback list the tiny config produces."""
    driver, captured = _callbacks_from(_config_at(tmp_path), tmp_path, monkeypatch)
    return {"driver": driver, **captured}


# --------------------------------------------------------------------------------------
# The tracked list
# --------------------------------------------------------------------------------------
def test_the_tracked_list_is_exactly_what_this_task_emits(task, stub_batch, perturb_posterior):
    """Both directions, driven from the real metrics dicts of both stages: an emitted metric the list
    does not know about is never collected, and a tracked name nothing emits is a column that is NaN
    in every row of every run. A validation-only readout tracked under ``train/`` (or the reverse)
    fails the second direction."""
    module = task()
    perturb_posterior(module.orig_model)

    _, train_metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")
    _, val_metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")
    emitted = {f"train/{name}" for name in train_metrics}
    emitted |= {f"val/{name}" for name in val_metrics}
    tracked = set(LagAttnCrwsTrainer.TRACKED_METRICS)

    assert emitted - tracked == set(), f"the task emits {emitted - tracked}, which nothing collects"
    assert tracked - (emitted | LOGGED_THROUGH_A_HOOK) == set()


# --------------------------------------------------------------------------------------
# The seam, at the one site that consumes it
# --------------------------------------------------------------------------------------
def test_the_metrics_collector_is_given_this_drivers_list_and_not_the_module_global(built):
    """``train_model`` is inherited whole, so if it still read the module global this driver's added
    columns would never appear."""
    collector = next(cb for cb in built["callbacks"] if isinstance(cb, MetricsLoggingCallback))

    assert collector.tracked_metrics == LagAttnCrwsTrainer.TRACKED_METRICS
    assert collector.tracked_metrics != _TRACKED_METRICS


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
