r"""Every tracked metric is reachable, everything the task emits is tracked, and the callbacks wire.

Two silent failure modes meet here. A tracked name the framework never emits produces a CSV column
that is NaN for every epoch of every run. And an emitted metric no callback tracks simply never
reaches the CSV, so a readout can be added, logged, and lost without any test noticing -- unless the
tracked list is driven from the real metrics dict, which is what happens below.

That second mode matters more here than in either sibling, because several of this model's columns
are the only evidence a multi-day run will produce on the questions the package exists to ask:
the two geometry guards ``target_warm_frac`` and ``anchors_per_sample``, and ``kld_source_null``,
which says whether the coupling readout measures the source availability *clock* rather than
source content.

The callback half checks the wiring this cell's config decides: the collector gets this driver's
list, both checkpoint criteria write under distinct stems of this model's name into the run's
checkpoint directory on metrics the task emits, and the shipped config attaches the diagnostic
plotter.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer
from teb_vae.lag_attn_rws.trainer import _TRACKED_METRICS
from train.callbacks import MetricsLoggingCallback, _unreachable_metric_names

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
    driver = LagAttnCfsTrainer(config_file_path=str(config_path))
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
def test_every_tracked_metric_is_a_name_the_framework_emits():
    """A bare name other than ``lr`` is renamed to ``{stage}/{name}`` on the way out and so matches
    nothing. Checked on this driver's list: the additions are hand-written names and this is what
    says they are spelled the way the framework will emit them."""
    assert _unreachable_metric_names(LagAttnCfsTrainer.TRACKED_METRICS) == ()


def test_the_tracked_list_and_the_emitted_metrics_agree_in_both_directions(
    task, stub_batch, perturb_posterior
):
    """Driven from the real metrics dicts of both stages. A metric this model emits that the list
    does not know about is lost; a tracked name nothing produces is a column that is NaN in every
    row of every run. And no duplicates: the collector keys on the name, so a repeat would silently
    write one column."""
    module = task()
    perturb_posterior(module.orig_model)

    _, train_metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")
    _, val_metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")
    emitted = {f"train/{name}" for name in train_metrics}
    emitted |= {f"val/{name}" for name in val_metrics}
    tracked = set(LagAttnCfsTrainer.TRACKED_METRICS)

    assert emitted - tracked == set(), f"the task emits {emitted - tracked}, which nothing collects"
    assert tracked - emitted - LOGGED_THROUGH_A_HOOK == set()
    assert len(tracked) == len(LagAttnCfsTrainer.TRACKED_METRICS)
    # The negative control on the seam: the inherited module-global list alone would lose columns
    # this model emits, so the first assertion above can fail.
    lost = emitted - set(_TRACKED_METRICS)
    assert lost and lost <= tracked


# --------------------------------------------------------------------------------------
# train_model wiring
# --------------------------------------------------------------------------------------
def test_train_model_goes_through_the_framework_trainer_builder(built):
    """No hand-rolled ``Trainer``: the builder attaches the learning-rate monitor and the MLflow
    run-logging callback, reconciles ``benchmark`` against ``deterministic``, and TTY-gates the
    progress bar."""
    assert "callbacks" in built
    assert "fit_args" in built  # the fit ran through the built trainer
    assert built["model"] is built["driver"].pl_model


def test_the_metrics_collector_is_given_this_drivers_list_and_not_the_module_global(built):
    """The seam, at the one site that consumes it. ``train_model`` is inherited whole, so if it
    still read the module global this assertion is the only thing between that and the columns
    this model adds never appearing."""
    collector = next(cb for cb in built["callbacks"] if isinstance(cb, MetricsLoggingCallback))

    assert collector.tracked_metrics == LagAttnCfsTrainer.TRACKED_METRICS
    assert collector.tracked_metrics != _TRACKED_METRICS


def test_both_checkpoint_callbacks_write_to_the_run_checkpoint_directory(built):
    """The framework hardcodes ``enable_checkpointing=True``, so a missing ``ModelCheckpoint`` means
    Lightning adds its own -- writing into the train-results directory instead.

    **Two rather than one**, because the composite optimum and the best conditioned forecast are
    different epochs: a run of this family has been observed to minimise ``val/total_loss`` and
    ``val/nll_full_block`` fifty-odd epochs apart, and a single criterion makes the other epoch's
    weights unrecoverable afterwards. The second is built only where the config names a monitor for
    it, which is what keeps the two-sided cells at one.
    """
    checkpoints = [cb for cb in built["callbacks"] if isinstance(cb, ModelCheckpoint)]
    configured = built["driver"].config["advanced_config"]["callbacks"]["model_checkpoint"]

    assert configured.get("secondary_monitor"), "this cell's config names no second criterion"
    assert len(checkpoints) == 2
    for checkpoint in checkpoints:
        assert checkpoint.dirpath == built["driver"].model_checkpoint_dir
    assert {checkpoint.monitor for checkpoint in checkpoints} == {
        configured["monitor"],
        configured["secondary_monitor"],
    }


def test_the_two_checkpoint_criteria_write_under_distinct_stems(built):
    """Lightning prefixes each placeholder with its own name, so a stem containing ``epoch=``
    renders as ``epoch=epoch=00``. And the stem must be this model's: three models writing
    ``lag-attn-rws-epoch=00.ckpt`` into a shared directory would be indistinguishable by name.

    The two criteria's stems must additionally differ from each other, and that is not cosmetic:
    with one stem Lightning would have each criterion overwrite the other's file at the same epoch,
    leaving one criterion's best silently unsaved -- the exact failure the second criterion exists
    to prevent.
    """
    checkpoints = [cb for cb in built["callbacks"] if isinstance(cb, ModelCheckpoint)]
    configured = built["driver"].config["advanced_config"]["callbacks"]["model_checkpoint"]
    primary = next(cb for cb in checkpoints if cb.monitor == configured["monitor"])
    secondary = next(cb for cb in checkpoints if cb.monitor == configured["secondary_monitor"])

    assert str(secondary.filename) != str(primary.filename)
    for checkpoint in checkpoints:
        assert "epoch=" not in str(checkpoint.filename)
        assert str(checkpoint.filename).startswith(f"{LagAttnCfsTrainer.CHECKPOINT_STEM}-")


def test_both_checkpoint_monitors_come_from_config_and_are_metrics_the_task_emits(
    built, task, stub_batch, perturb_posterior
):
    """Two halves of one guarantee, for both criteria: each monitor is the config's, and the config
    names a metric that actually lands in ``callback_metrics`` -- a ``ModelCheckpoint`` whose
    monitor nothing emits saves nothing and says nothing. The two criteria also differ, or two
    callbacks would keep the same epochs twice."""
    checkpoints = [cb for cb in built["callbacks"] if isinstance(cb, ModelCheckpoint)]
    configured = built["driver"].config["advanced_config"]["callbacks"]["model_checkpoint"]
    primary = next(cb for cb in checkpoints if cb.monitor == configured["monitor"])

    assert primary.save_top_k == configured["save_top_k"]
    assert configured["secondary_monitor"] != configured["monitor"]

    module = task()
    perturb_posterior(module.orig_model)
    for monitor in (configured["monitor"], configured["secondary_monitor"]):
        stage, _, suffix = monitor.partition("/")
        _, metrics = module.compute_loss_and_metrics(stub_batch, 0, stage)
        assert suffix in metrics, f"the monitor {monitor} is a metric nothing emits"


# --------------------------------------------------------------------------------------
# The diagnostic figure
# --------------------------------------------------------------------------------------
def test_the_shipped_config_wires_the_diagnostic_plotter(built):
    """The plotter is imported lazily inside the enabled branch, and the block that enables it is
    read under the inherited driver's spelling -- so this is the test that would catch either the
    import breaking or the config block being renamed out of reach."""
    names = [type(cb).__name__ for cb in built["callbacks"]]

    assert "LagAttnRwsPlotCallback" in names
