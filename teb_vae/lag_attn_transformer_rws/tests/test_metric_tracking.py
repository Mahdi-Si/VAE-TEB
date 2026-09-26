r"""The inherited tracked-metric list against this task, and the callbacks the shipped config wires.

Two silent failure modes meet here. A tracked name nothing emits produces a CSV column that is NaN
for every epoch of every run. And an emitted metric no callback tracks simply never reaches the
CSV, so a readout can be added, logged, and lost without any test noticing -- unless the tracked
list is compared with the real metrics dicts of this task, which is what happens below.

The callback assembly is inherited, so what is checked here is only what this package supplies to
it: the checkpoint stem, which walks into a known trap -- Lightning auto-prefixes each placeholder
with its own name, so a stem containing ``epoch=`` renders as ``epoch=epoch=00`` -- the shipped
config's checkpoint monitor, and its diagnostic-figure block.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint

from teb_vae.lag_attn_rws.trainer import _TRACKED_METRICS, LagAttnRwsTrainer
from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


# --------------------------------------------------------------------------------------
# The tracked list
# --------------------------------------------------------------------------------------
def test_the_tracked_list_and_this_tasks_metrics_agree_in_both_directions(
    task, stub_batch, perturb_posterior
):
    """Driven from the real metrics dicts of both stages. Five tracked names are logged through a
    hook rather than returned in a metrics dict and so cannot appear in one: the two spike-breaker
    columns the base's training step injects, the ``lr`` it logs at epoch start, and the pre-clip
    gradient norm plus its clip-exceedance indicator the task logs in
    ``on_before_optimizer_step``."""
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

    untracked = emitted - set(_TRACKED_METRICS)
    assert untracked == set(), f"the task emits {untracked}, which no callback collects"
    assert set(_TRACKED_METRICS) - emitted - logged_through_a_hook == set()


# --------------------------------------------------------------------------------------
# train_model wiring
# --------------------------------------------------------------------------------------
@pytest.fixture
def built(tmp_path, monkeypatch):
    """The driver on the shipped config and the callback list ``train_model`` hands to the trainer
    builder, captured without a fit."""
    driver = LagAttnTrfRwsTrainer(config_file_path=str(_CONFIG))
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
    return {"driver": driver, **captured}


def test_the_checkpoint_filename_carries_this_models_stem_and_no_double_prefix(built):
    """Lightning prefixes each placeholder with its own name, so a stem containing ``epoch=``
    renders as ``epoch=epoch=00``. And the stem must be this model's: two architectures writing
    the same checkpoint name into a shared directory would be indistinguishable by name."""
    checkpoint = next(cb for cb in built["callbacks"] if isinstance(cb, ModelCheckpoint))
    filename = str(checkpoint.filename)

    assert "epoch=" not in filename
    assert filename.startswith(LagAttnTrfRwsTrainer.CHECKPOINT_STEM)
    assert LagAttnTrfRwsTrainer.CHECKPOINT_STEM != LagAttnRwsTrainer.CHECKPOINT_STEM


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


def test_the_shipped_config_wires_the_diagnostic_plotter(built):
    """The plotter is imported lazily inside the enabled branch, and the block that enables it is
    read under the inherited driver's spelling -- so this is the test that catches either the
    import breaking or the block being renamed in this package's config."""
    names = [type(cb).__name__ for cb in built["callbacks"]]

    assert "LagAttnRwsPlotCallback" in names
