r"""Every tracked metric is reachable, everything the task emits is tracked, and the seam is wired.

Two silent failure modes meet here. A tracked name the framework never emits produces a CSV column
that is NaN for every epoch of every run. And an emitted metric no callback tracks simply never
reaches the CSV, so a readout can be added, logged, and lost without any test noticing -- unless the
tracked list is driven from the real metrics dict, which is what happens below.

That second mode matters here because four of this model's columns are the only evidence a multi-day
run will produce on the questions the package exists to ask. One is a geometry *guard* --
``anchors_per_sample`` must sit at its geometry-derived value -- and one, ``kld_source_null``, is the
single most important number on the page: if it sits close to ``source_conditioned_kl_raw`` then the
coupling readout is measuring the source availability *clock* rather than source content, and
nothing else in the run can say so.

The headline column is not among the four, and that is the point of the cell. ``pred_gap`` is the
parent's own, emitted by the shared objective, tracked by the parent's own list, and read against the
comparison model's runs -- so the tests below also assert, in the other direction, that this
package's arrival moved neither of the parent's two literals.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

from teb_vae.lag_attn_crws.trainer import (
    _CAUSAL_RAW_SUFFIXES,
    _CAUSAL_RAW_VAL_ONLY_SUFFIXES,
    LagAttnCrwsTrainer,
)
from teb_vae.lag_attn_rws.trainer import (
    _METRIC_SUFFIXES,
    _TRACKED_METRICS,
    _TRAIN_ONLY_SUFFIXES,
    _VAL_ONLY_SUFFIXES,
    LagAttnRwsTrainer,
)
from train.callbacks import (
    MetricsHistoryCsvCallback,
    MetricsLoggingCallback,
    _unreachable_metric_names,
)

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
def test_every_tracked_metric_is_a_name_the_framework_emits():
    """A bare name other than ``lr`` is renamed to ``{stage}/{name}`` on the way out and so matches
    nothing. Checked on this driver's list: the additions are hand-written names and this is what
    says they are spelled the way the framework will emit them."""
    assert _unreachable_metric_names(LagAttnCrwsTrainer.TRACKED_METRICS) == ()


def test_the_tracked_list_covers_what_this_task_emits(task, stub_batch, perturb_posterior):
    """Driven from the real metrics dicts of both stages, so a metric this model emits and the list
    does not know about cannot slip through."""
    module = task()
    perturb_posterior(module.orig_model)

    _, train_metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")
    _, val_metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    tracked = set(LagAttnCrwsTrainer.TRACKED_METRICS)
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

    assert set(LagAttnCrwsTrainer.TRACKED_METRICS) - emitted == set()


def test_the_added_columns_are_exactly_the_declared_ones():
    """Both directions against the comparison model's list, so the delta is a decision rather than
    whatever the code happened to emit."""
    added = set(LagAttnCrwsTrainer.TRACKED_METRICS) - set(_TRACKED_METRICS)

    assert added == {
        f"{stage}/{name}" for stage in ("train", "val") for name in _CAUSAL_RAW_SUFFIXES
    } | {f"val/{name}" for name in _CAUSAL_RAW_VAL_ONLY_SUFFIXES}
    assert set(_TRACKED_METRICS) - set(LagAttnCrwsTrainer.TRACKED_METRICS) == set()
    # No duplicates: the collector keys on the name, and a repeat would silently write one column.
    assert len(set(LagAttnCrwsTrainer.TRACKED_METRICS)) == len(LagAttnCrwsTrainer.TRACKED_METRICS)


def test_the_three_per_stage_columns_are_tracked_on_both_stages():
    """They come out of ``compute_loss``, like every other term of the objective, so they exist on
    training and validation batches alike. A ``val/``-only entry would silently halve the evidence a
    run produces -- and for the geometry guard it would remove the training rows entirely, which are
    the ones a tiling bug would show in."""
    for name in _CAUSAL_RAW_SUFFIXES:
        assert f"train/{name}" in LagAttnCrwsTrainer.TRACKED_METRICS, name
        assert f"val/{name}" in LagAttnCrwsTrainer.TRACKED_METRICS, name


def test_the_source_null_floor_is_tracked_on_validation_only():
    """It costs a source encode per step and is a readout that never enters the objective, so it
    runs on validation batches alone; a ``train/`` variant would be NaN in every row of every run."""
    assert _CAUSAL_RAW_VAL_ONLY_SUFFIXES == ("kld_source_null",)
    assert "val/kld_source_null" in LagAttnCrwsTrainer.TRACKED_METRICS
    assert "train/kld_source_null" not in LagAttnCrwsTrainer.TRACKED_METRICS


@pytest.mark.parametrize("name", _VAL_ONLY_SUFFIXES + _CAUSAL_RAW_VAL_ONLY_SUFFIXES)
def test_no_validation_only_name_has_a_training_variant(name):
    """The general form of the rule above, over the inherited validation-only names too: each is
    absent -- never zero-filled -- on the steps that did not run it, and the framework's epoch value
    is the mean over the steps that reported a metric."""
    assert f"val/{name}" in LagAttnCrwsTrainer.TRACKED_METRICS
    assert f"train/{name}" not in LagAttnCrwsTrainer.TRACKED_METRICS


@pytest.mark.parametrize("name", _TRAIN_ONLY_SUFFIXES)
def test_no_training_only_name_has_a_validation_variant(name):
    """The mirror: the spike breaker's two diagnostic columns and the two gradient-norm ones exist on
    the training path alone, and a ``val/`` variant would be a column no run ever fills."""
    assert f"train/{name}" in LagAttnCrwsTrainer.TRACKED_METRICS
    assert f"val/{name}" not in LagAttnCrwsTrainer.TRACKED_METRICS


def test_the_comparison_models_collector_is_unchanged_by_this_packages_arrival():
    """The seam defaults to the module literal, so adding to it must not have moved the comparison
    model's own column set -- every prior run's CSV is read against that list, and ``pred_gap`` is
    read across the two."""
    assert LagAttnRwsTrainer.TRACKED_METRICS == _TRACKED_METRICS
    assert "pred_gap" in _METRIC_SUFFIXES
    assert _VAL_ONLY_SUFFIXES == ("nll_shuffled_block", "kld_shuffled", "shuffle_penalty")
    assert _TRAIN_ONLY_SUFFIXES == (
        "spike_skipped", "spike_ema_loss", "grad_norm", "grad_clip_frac"
    )


def test_the_inherited_list_alone_would_lose_this_models_columns(
    task, stub_batch, perturb_posterior
):
    """The negative control on the ``TRACKED_METRICS`` seam. Without this driver's extension the
    columns are emitted, logged and never collected -- which is silent, and which is the whole reason
    the seam exists rather than the list being read as a module global."""
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    lost = {f"train/{name}" for name in metrics} - set(_TRACKED_METRICS)
    assert lost == {f"train/{name}" for name in _CAUSAL_RAW_SUFFIXES}
    assert lost <= set(LagAttnCrwsTrainer.TRACKED_METRICS)


# --------------------------------------------------------------------------------------
# The seam, at the one site that consumes it
# --------------------------------------------------------------------------------------
def test_the_metrics_collector_is_given_this_drivers_list_and_not_the_module_global(built):
    """``train_model`` is inherited whole, so if it still read the module global this assertion is
    the only thing between that and seven columns that never appear."""
    collector = next(cb for cb in built["callbacks"] if isinstance(cb, MetricsLoggingCallback))

    assert collector.tracked_metrics == LagAttnCrwsTrainer.TRACKED_METRICS
    assert collector.tracked_metrics != _TRACKED_METRICS


def test_the_metrics_history_writer_is_wired_to_the_collector(built):
    """The collector accumulates history and never writes it; the writer persists it."""
    collector = next(cb for cb in built["callbacks"] if isinstance(cb, MetricsLoggingCallback))
    writer = next(cb for cb in built["callbacks"] if isinstance(cb, MetricsHistoryCsvCallback))

    assert writer.source is collector


def test_the_checkpoint_filename_carries_this_models_stem_and_no_double_prefix(built):
    """Lightning prefixes each placeholder with its own name, so a stem containing ``epoch=``
    renders as ``epoch=epoch=00``. And the stem must be this model's: two models writing
    ``lag-attn-rws-epoch=00.ckpt`` into a shared directory would be indistinguishable by name."""
    checkpoint = next(cb for cb in built["callbacks"] if isinstance(cb, ModelCheckpoint))
    filename = str(checkpoint.filename)

    assert "epoch=" not in filename
    assert filename == "lag-attn-crws-{epoch:02d}"
    assert checkpoint.dirpath == built["driver"].model_checkpoint_dir


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
