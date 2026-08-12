r"""One real fit, through the real entry point, against the committed shard.

Everything else in this suite tests a piece in isolation. This runs the whole thing: config ->
pre-flight guards -> ``setup_config`` -> data module -> model -> ``build_trainer`` -> ``fit`` ->
checkpoint, on a CPU, in seconds. It is the only place the failures that live *between* the pieces
can surface -- a config key that reaches nothing, a metric name no callback collects, a callback
that raises on the first validation epoch, a diagnostic figure that fails to draw.

Driven through ``main`` rather than by assembling the driver by hand, deliberately: the four
pre-flight guards, the temporary resolved-config file and the resolved-config write beside the
checkpoints hang off the entry point and are reached no other way. This is therefore also the only
test in which the target-field normalisation guard runs against a real config, with a real loader
behind it, and lets the run proceed.

Two epochs rather than the config's one: ``lr`` is logged at train-epoch *start* with
``on_epoch=True``, so its first CSV cell is always NaN, and the second epoch is what exercises the
scheduler stepping at all.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_fs import trainer as trainer_module
from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer
from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME
from train.graph_models_utils import check_model_class, load_checkpoint_strict

from .conftest import absolutize_dataset_paths

pytestmark = pytest.mark.slow

_TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"

#: Epochs the fit runs. See the module docstring for why it is not the config's one.
SMOKE_EPOCHS = 2

#: What the shipped reach budget resolves to, pinned so a "guarded" fit cannot silently be the
#: unguarded one -- which here would also silently change the decoder's width and therefore the
#: units of every number the run reports.
GUARDED_TARGET_CHANNELS = 78
GUARDED_SOURCE_CHANNELS = 29
GUARDED_MAX_DELAY = 30


def _run_fit(tmp_path):
    """Run one real fit through the entry point and return the driver and its fitted trainer.

    Args:
        tmp_path: Directory the run writes into.

    Returns:
        ``(driver, trainer)``.
    """
    config = absolutize_dataset_paths(load_config(str(_TINY)))
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path)
    config["general_config"]["epochs"] = SMOKE_EPOCHS
    # Off: this asserts the training path, not the profiler's output.
    config["advanced_config"]["trainer"]["profiler"] = None

    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    captured = {}
    original_train_model = LagAttnFsTrainer.train_model

    def _capture_train_model(self, train_loader, validation_loader):
        result = original_train_model(self, train_loader, validation_loader)
        captured["driver"] = self
        captured["trainer"] = result
        return result

    LagAttnFsTrainer.train_model = _capture_train_model
    try:
        trainer_module.main(str(config_path))
    finally:
        # Deleted rather than reassigned: the method is inherited, and leaving a copy on the
        # subclass would shadow a later change to the one it inherits.
        del LagAttnFsTrainer.train_model

    return captured["driver"], captured["trainer"]


@pytest.fixture(scope="module")
def fit(tmp_path_factory):
    """One real fit. Module-scoped: this is the expensive test in the suite, and every assertion
    below is a different question about the same run."""
    return _run_fit(tmp_path_factory.mktemp("smoke"))


# --------------------------------------------------------------------------------------
# The fit itself
# --------------------------------------------------------------------------------------
def test_the_fit_completes(fit):
    _, trainer = fit

    assert trainer.current_epoch == SMOKE_EPOCHS
    assert trainer.state.finished


def test_the_losses_stay_finite(fit):
    _, trainer = fit

    for name, value in trainer.callback_metrics.items():
        assert math.isfinite(float(value)), f"{name} is {float(value)}"


def test_the_gradient_norm_stays_finite(fit):
    """The quantity ``gradient_clip_val`` is re-derived from. If it were non-finite the clip
    coefficient would be zero and the run would train nothing while completing normally -- which is
    exactly the failure the guarded raw-signal arms once had."""
    _, trainer = fit

    grad_norm = float(trainer.callback_metrics["train/grad_norm"])

    assert math.isfinite(grad_norm), f"train/grad_norm is {grad_norm}"


def test_the_run_trains_at_the_budgets_width_and_not_the_raw_grids(fit):
    """The whole binding, end to end: config -> filter bank -> channel tuples -> decoder width.

    A unit test can check each link; only a fit can check they are connected, and the failure this
    catches is silent: at ``raw_per_step`` the model would decode $16$ channels of a $78$-channel
    target and the shapes would still line up nowhere -- but a decoder built from the *declared*
    $c_y$ would run to completion scoring $109$ channels against a $78$-channel block's worth of
    target, which is a different objective under the same config.
    """
    driver, _ = fit
    model = driver.pytorch_model

    assert model.decoder_out_channels == GUARDED_TARGET_CHANNELS
    assert model.target_adapter.linear.in_features == GUARDED_TARGET_CHANNELS
    assert model.source_adapter.linear.in_features == GUARDED_SOURCE_CHANNELS
    assert model.target_gate.max_delay == GUARDED_MAX_DELAY
    assert model.raw_per_step == 16
    # The declared widths are untouched, which is what the data boundary checks against.
    assert (model.c_y, model.c_u) == (109, 58)


def test_the_zero_kl_init_invariant_survives_the_whole_stack(fit):
    r"""At initialisation $q(z_t \mid Y, U) = p(z_t \mid Y)$ exactly, so $K = 0$.

    Re-derived from a freshly built model rather than read off the trained run: after an epoch the
    KL is legitimately nonzero, and the question is whether the model *this config* builds starts at
    zero -- after config resolution, the kwarg sweep and the framework's own seeding have each had a
    chance to break it.
    """
    driver, _ = fit
    model = SeqVaeLagAttnFs(**driver._build_model_kwargs()).eval()
    generator = torch.Generator().manual_seed(0)
    batch_size, seq_len = 2, model.sequence_length
    outputs = model(
        torch.randn(batch_size, seq_len, 43, generator=generator),
        torch.randn(batch_size, seq_len, 66, generator=generator),
        torch.randn(batch_size, seq_len, model.c_u, generator=generator),
    )

    assert float(outputs["kld_per_t"].abs().max()) == 0.0
    # The shipped `base_decode: mean` leaves the two forecasts differing by the posterior's own
    # noise; what must hold instead is that the base forecast IS the decode of mu^p.
    assert torch.equal(outputs["z_prior"], outputs["mu_prior"])
    expected_base, _ = model.decoder(outputs["mu_prior"][:, : model.geometry.t_valid])
    assert torch.equal(outputs["mu_base"], expected_base)
    assert not torch.equal(outputs["mu_base"], outputs["mu_full"])


# --------------------------------------------------------------------------------------
# What the run records
# --------------------------------------------------------------------------------------
def test_every_declared_metric_reaches_the_logger(fit):
    """The gap between "the task emits it" and "a callback collected it" is silent otherwise. The
    shuffled readouts are the ones only a real validation loop can prove wired."""
    _, trainer = fit

    for name in (
        "train/total_loss",
        "train/main_loss",
        "train/grad_norm",
        "train/source_conditioned_kl_raw",
        "val/total_loss",
        "val/nll_shuffled_block",
        "val/kld_shuffled",
        "val/shuffle_penalty",
    ):
        assert name in trainer.callback_metrics, f"{name} never reached callback_metrics"


def test_the_metrics_csv_carries_every_tracked_key_and_no_all_nan_column(fit):
    """Both halves of the tracked list's contract, on a real run: a name the framework never emits
    is a column that is NaN in every row of every run, and a tracked name that produced no column
    at all is a readout nothing ever recorded."""
    driver, _ = fit
    frame = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")

    missing = [
        name for name in LagAttnFsTrainer.TRACKED_METRICS if name not in frame.columns
    ]
    assert missing == [], f"tracked but never written to the CSV: {missing}"
    all_nan = [column for column in frame.columns if frame[column].isna().all()]
    assert all_nan == [], f"columns that are NaN for every epoch: {all_nan}"


def test_the_four_forecast_gap_columns_reach_the_csv_and_recompose(fit):
    """This model's whole observability addition, on a real run rather than on a stub batch. Both
    splits must recompose to the ``pred_gap`` in the same row: a column that reached the CSV
    carrying some other quantity would pass every test above."""
    driver, _ = fit
    frame = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")

    for stage in ("train", "val"):
        row = frame[[f"{stage}/pred_gap", *(
            f"{stage}/{name}" for name in
            ("pred_gap_tau_first", "pred_gap_tau_last", "pred_gap_st", "pred_gap_ph")
        )]].dropna()
        assert not row.empty, f"no epoch recorded the {stage} forecast gaps"
        blocks = row[f"{stage}/pred_gap_st"] + row[f"{stage}/pred_gap_ph"]
        assert blocks.sub(row[f"{stage}/pred_gap"]).abs().max() < 1e-3, stage


def test_the_scheduled_beta_and_the_anchor_weight_reach_the_csv(fit):
    """The resolved schedule value, which starts at exactly zero -- the posterior-collapse guard the
    config documents -- and the echoed anchor weight, whose whole job is to let a
    ``metrics_history.csv`` identify its own arm afterwards.

    The anchor is compared against **the run's own resolved config**, not against a literal. The
    task's default is $0$, so a weight silently dropped at the kwarg sweep still fails here; a
    literal would additionally fail whenever the calibration sweep moves the pair, which is a
    different thing and belongs in ``test_config_load.py``."""
    driver, _ = fit
    frame = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")
    configured = float(
        load_config(str(_TINY))["model_config"]["VAE_model"]["beta_prior"]
    )

    assert float(frame["train/kld_beta"].iloc[0]) == pytest.approx(0.0)
    assert configured > 0.0, "the anchor is off in tiny.yaml; this test would pass vacuously"
    assert float(frame["train/beta_prior"].iloc[0]) == pytest.approx(configured)


def test_the_prior_and_clip_diagnostics_reach_the_csv(fit):
    """Through the tiny config's ``mse`` likelihood -- the configuration where a term emitted only
    under ``gaussian_nll`` would silently produce an all-NaN column."""
    driver, _ = fit
    frame = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")

    for column in (
        "train/prior_rate", "val/prior_rate",
        "train/logvar_prior_floor_frac", "val/logvar_prior_floor_frac",
        "train/grad_clip_frac",
    ):
        assert column in frame.columns, f"{column} never reached the CSV"
        assert frame[column].notna().any(), f"{column} is NaN in every epoch"

    clip_frac = frame["train/grad_clip_frac"].dropna()
    assert bool(((clip_frac >= 0.0) & (clip_frac <= 1.0)).all())


def test_the_run_directory_has_the_expected_layout(fit):
    """The log sinks, the checkpoint directory and the resolved config a later offline pass needs."""
    driver, _ = fit

    assert (Path(driver.train_results_dir) / "full.log").is_file()
    assert (Path(driver.train_results_dir) / "metrics_history.csv").is_file()
    assert Path(driver.model_checkpoint_dir).is_dir()


def test_the_resolved_config_records_the_budget_the_run_actually_got(fit):
    """The budget in seconds does not name a channel: what it resolves to depends on a filter bank,
    and here it also decides the decoder's width. A run recording only the request would record
    neither what it got nor what its nats were summed over."""
    from teb_vae.lag_attn_rws.trainer import RESOLVED_BUDGET_KEY

    driver, _ = fit

    written = Path(driver.model_checkpoint_dir) / RESOLVED_CONFIG_FILENAME
    assert written.is_file()
    reloaded = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert "base" not in reloaded
    record = reloaded["model_config"][RESOLVED_BUDGET_KEY]
    assert record["causal_reach_budget_s"] == 120
    assert record["max_delay_steps"] == GUARDED_MAX_DELAY
    assert len(record["target_keep_index"]) == GUARDED_TARGET_CHANNELS
    assert len(record["source_keep_index"]) == GUARDED_SOURCE_CHANNELS


def test_the_checkpoint_is_written_under_this_models_stem(fit):
    """Two models writing ``lag-attn-rws-epoch=00.ckpt`` into a shared directory would be
    indistinguishable by name, and the stem is one of the five strings this package supplies to the
    inherited callback assembly."""
    driver, _ = fit

    checkpoints = list(Path(driver.model_checkpoint_dir).glob("*.ckpt"))

    assert checkpoints, "no checkpoint was written; Lightning's default would have gone elsewhere"
    assert all(path.name.startswith("lag-attn-fs-epoch=") for path in checkpoints), [
        path.name for path in checkpoints
    ]


def test_the_checkpoint_carries_its_contract_and_reloads_at_its_own_width(fit):
    """The end of the road: a blob that describes itself and rebuilds without a config file. The
    stamped keep-index is what carries the decoder's width, since ``decoder_out_channels`` is
    deliberately not recorded -- a second field naming the width could disagree with the gate."""
    driver, _ = fit

    path = next(iter(Path(driver.model_checkpoint_dir).glob("*.ckpt")))
    blob = torch.load(path, map_location="cpu", weights_only=False)

    assert blob["model_class"] == "SeqVaeLagAttnFs"
    assert blob["model_kwargs"] == driver._build_model_kwargs()
    assert len(blob["model_kwargs"]["target_keep_index"]) == GUARDED_TARGET_CHANNELS
    assert "decoder_out_channels" not in blob["model_kwargs"]
    check_model_class(blob, "SeqVaeLagAttnFs")
    rebuilt = SeqVaeLagAttnFs(**blob["model_kwargs"])
    assert rebuilt.decoder_out_channels == GUARDED_TARGET_CHANNELS
    assert load_checkpoint_strict(rebuilt, blob) is not None, (
        "the checkpoint's state dict did not align into a model rebuilt from its own kwargs"
    )


def test_the_plotting_callback_is_attached_and_does_not_fail_the_fit(fit):
    """The callback runs on every validation epoch and swallows its own exceptions by design, so
    the property that matters on the training path is that a figure is never worth failing a fit
    for. Both halves are asserted: it really is attached (a wiring mistake would make the rest of
    this test vacuous), and the fit that ran with it finished."""
    driver, trainer = fit

    attached = [
        callback for callback in trainer.callbacks
        if type(callback).__name__ == "LagAttnRwsPlotCallback"
    ]

    assert len(attached) == 1
    assert trainer.state.finished
    assert (Path(driver.train_results_dir) / "lag_attn_rws_diagnostics").is_dir()


def test_the_page_renders_for_every_plotted_epoch_and_sample(fit):
    """The diagnostic page, drawn by the real callback on a real validation epoch.

    Everything else about the figure is checked on a stub batch. What only a fit can show is that
    the callback's route to it survives the whole stack: the task names its rows, the callback
    resolves them, the shared builder draws five rows of its own around them, and the file reaches
    disk. The callback swallows its own exceptions by design -- a figure is never worth failing a
    multi-day fit for -- so a page that raised would leave exactly the same green run with an empty
    directory, which is why the count is asserted rather than the absence of an error.

    One file per drawn sample per plotted epoch, and the config plots every epoch.
    """
    driver, trainer = fit
    callback = next(
        item for item in trainer.callbacks if type(item).__name__ == "LagAttnRwsPlotCallback"
    )

    # The page prefix, not `*.pdf`: the callback writes the run-level causal-input-budget figure
    # into the same directory, and it is not a page of any epoch.
    figures = sorted(
        (Path(driver.train_results_dir) / "lag_attn_rws_diagnostics").glob(
            "lag_attn_rws_epoch*.pdf"
        )
    )
    # The stem is `..._epoch%04d_sample%d_%s`, so the epoch is recoverable from the name; a page
    # per epoch is the property, not a total, because the batch may hold fewer samples than the
    # config asks pages for.
    per_epoch: Dict[str, List[str]] = defaultdict(list)
    for path in figures:
        per_epoch[path.stem.split("_")[3]].append(path.name)

    assert sorted(per_epoch) == [f"epoch{epoch:04d}" for epoch in range(SMOKE_EPOCHS)], [
        path.name for path in figures
    ]
    for epoch, names in per_epoch.items():
        assert 1 <= len(names) <= callback.num_examples, (epoch, names)
        assert len(set(names)) == len(names), f"{epoch} overwrote one of its own pages"
    assert all(path.stat().st_size > 0 for path in figures)
