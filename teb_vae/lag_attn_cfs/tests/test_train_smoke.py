r"""One real fit, through the real entry point, against the committed causal shard.

Everything else in this suite tests a piece in isolation. This runs the whole thing: config ->
pre-flight guards -> ``setup_config`` -> data module -> model -> ``build_trainer`` -> ``fit`` ->
checkpoint, on a CPU, in seconds. It is the only place the failures that live *between* the pieces
can surface -- a config key that reaches nothing, a metric name no callback collects, a callback that
raises on the first validation epoch, and, specific to this cell, the two that arrive only on a
**validation** step: the permutation control decoding at the wrong anchor set, and the source-null
arm re-encoding a stream that is no longer in the forward dict.

Driven through ``main`` rather than by assembling the driver by hand, deliberately: the four shared
pre-flight guards, this driver's own five, the temporary resolved-config file and the resolved-config
write beside the checkpoints hang off the entry point and are reached no other way.

Two epochs rather than the config's one: ``lr`` is logged at train-epoch *start* with
``on_epoch=True``, so its first CSV cell is always NaN, and the second epoch is what exercises the
scheduler stepping at all -- and, here, what makes the tile phase rotate, since the epoch is one of
the four halves of its key.

**The diagnostic page is asserted here rather than only in its own file**, and the log is asserted
with it. Every one of its three builders fails *quietly* on this model -- the shared forecast rows
walk a dense $(T_{\mathrm{valid}}, H, C)$ block against this one's $(A_{\max}, H, C)$, and the input
rows and the run-level figure are built from the production two-sided filter bank, which refuses
these channel widths -- and all three failures are swallowed by design, because a figure is never
worth failing a multi-day fit for. So a page with two rows missing and a figure that was never
written looks exactly like a page nobody asked for, unless a real fit is made to produce one and the
warnings it did not emit are named.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_cfs import trainer as trainer_module
from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer
from teb_vae.lag_attn_cfs.warmup_budget import BUDGET_FIGURE_STEM
from teb_vae.lag_attn_rws import plotting as plotting_module
from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME
from train.graph_models_utils import check_model_class, load_checkpoint_strict

from .conftest import (
    CAUSAL_C_U,
    CAUSAL_C_Y,
    CAUSAL_PH_WIDTH,
    CAUSAL_ST_WIDTH,
    SHIPPED_HORIZON,
    SHIPPED_WARMUP_PERIOD,
    absolutize_dataset_paths,
)

pytestmark = pytest.mark.slow

_TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"

#: Epochs the fit runs. See the module docstring for why it is not the config's one.
SMOKE_EPOCHS = 2

#: What the shipped warm-up budget resolves to, pinned so a "guarded" fit cannot silently be the
#: unguarded one -- which here would also silently change the decoder's width and therefore the
#: units of every number the run reports.
GUARDED_TARGET_CHANNELS = 98
GUARDED_SOURCE_CHANNELS = CAUSAL_C_U

#: The anchor counts the two stages must produce, derived here the way the model derives them so a
#: geometry change re-derives them rather than failing a literal.
DENSE_ANCHORS = 300 - SHIPPED_HORIZON - SHIPPED_WARMUP_PERIOD
TILE_COUNT = -(-DENSE_ANCHORS // SHIPPED_HORIZON)


def _run_fit(tmp_path, seed=None):
    """Run one real fit through the entry point and return the driver and its fitted trainer.

    Args:
        tmp_path: Directory the run writes into.
        seed: Optional override of ``general_config.seed``.

    Returns:
        ``(driver, trainer)``.
    """
    config = absolutize_dataset_paths(load_config(str(_TINY)))
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path)
    config["general_config"]["epochs"] = SMOKE_EPOCHS
    if seed is not None:
        config["general_config"]["seed"] = seed
    # Off: this asserts the training path, not the profiler's output.
    config["advanced_config"]["trainer"]["profiler"] = None

    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    captured = {}
    original_train_model = LagAttnCfsTrainer.train_model

    def _capture_train_model(self, train_loader, validation_loader):
        result = original_train_model(self, train_loader, validation_loader)
        captured["driver"] = self
        captured["trainer"] = result
        return result

    LagAttnCfsTrainer.train_model = _capture_train_model
    try:
        trainer_module.main(str(config_path))
    finally:
        # Deleted rather than reassigned: the method is inherited, and leaving a copy on the
        # subclass would shadow a later change to the one it inherits.
        del LagAttnCfsTrainer.train_model

    return captured["driver"], captured["trainer"]


def _metrics(driver) -> pd.DataFrame:
    return pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")


def _run_fit_in_subprocess(tmp_path, hash_seed: str) -> pd.DataFrame:
    """Run one fit in a **separate process, on the CPU**, and return its metrics history.

    Three deliberate choices, each removing a source of variation that has nothing to do with what
    this package decides:

    * a separate process, because "two runs of one config" is literally two processes -- and because
      it is the only way to vary ``PYTHONHASHSEED``, whose per-process salt is what would break a
      tile phase derived from Python's own ``hash()`` with nothing raising;
    * the CPU, because the shipped config chooses cuDNN autotuning, which selects convolution
      algorithms by timing them and therefore accumulates the same sums in different orders. No seed
      controls that. Forcing ``deterministic: true`` instead does not work from inside a test
      session: it needs ``CUBLAS_WORKSPACE_CONFIG`` set before CUDA is initialised, and by then
      another test has initialised it;
    * ``num_workers: 0`` is already the tiny config's, so no loader worker seeding enters either.

    Args:
        tmp_path: Directory the run writes into.
        hash_seed: The ``PYTHONHASHSEED`` the subprocess runs under.

    Returns:
        The run's ``metrics_history.csv``.
    """
    import os
    import subprocess
    import sys

    config = absolutize_dataset_paths(load_config(str(_TINY)))
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path)
    config["general_config"]["epochs"] = SMOKE_EPOCHS
    config["advanced_config"]["trainer"]["profiler"] = None
    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]);"
            "from teb_vae.lag_attn_cfs.trainer import main; main(sys.argv[2])",
            str(repo_root),
            str(config_path),
        ],
        env={
            **dict(os.environ),
            "PYTHONHASHSEED": hash_seed,
            "CUDA_VISIBLE_DEVICES": "",
        },
        cwd=str(repo_root),
        capture_output=True,
        check=True,
        text=True,
    )
    written = list(tmp_path.rglob("metrics_history.csv"))
    assert len(written) == 1, written
    return pd.read_csv(written[0])


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


def test_the_spike_breaker_never_latched(fit):
    """The re-derived ``additive_margin`` moved **down**, so the live risk is a margin that skips
    ordinary batches -- which reads in a log exactly like a model that keeps blowing up."""
    driver, _ = fit
    frame = _metrics(driver)

    assert float(frame["train/spike_skipped"].max()) == 0.0


def test_the_gradient_norm_stays_finite(fit):
    """The quantity ``gradient_clip_val`` is re-derived from. If it were non-finite the clip
    coefficient would be zero and the run would train nothing while completing normally."""
    _, trainer = fit

    grad_norm = float(trainer.callback_metrics["train/grad_norm"])

    assert math.isfinite(grad_norm), f"train/grad_norm is {grad_norm}"


def test_the_run_trains_at_the_budgets_width_and_the_configs_tiling(fit):
    """The whole binding, end to end: config -> shard attributes -> channel tuples -> decoder width,
    and config -> stride -> decoded anchor set.

    A unit test can check each link; only a fit can check they are connected, and the failure this
    catches is silent: a decoder built from the *declared* $c_y$ would run to completion scoring
    $102$ channels against a $98$-channel block's worth of target, which is a different objective
    under the same config."""
    driver, _ = fit
    model = driver.pytorch_model

    assert model.decoder_out_channels == GUARDED_TARGET_CHANNELS
    assert model.target_adapter.linear.in_features == GUARDED_TARGET_CHANNELS
    assert model.source_adapter.linear.in_features == GUARDED_SOURCE_CHANNELS
    assert model.anchor_stride == model.horizon == SHIPPED_HORIZON
    assert model.warmup_period == SHIPPED_WARMUP_PERIOD
    assert model.raw_per_step == 16
    # The declared widths are untouched, which is what the data boundary checks against.
    assert (model.c_y, model.c_u) == (CAUSAL_C_Y, CAUSAL_C_U)


def test_the_input_adapters_carry_the_availability_terms_the_warm_up_needs(fit):
    """The mechanism the whole package exists for, on the model a real launch built. Without the
    ``_build_adapter`` override the gate's delays are all zero -- it is a pure gather -- so
    ``max_delay`` would be $0$ and **neither** availability term would exist, and the leading region
    of every channel would enter the encoder as though it were signal."""
    driver, _ = fit
    model = driver.pytorch_model

    for adapter in (model.target_adapter, model.source_adapter):
        assert adapter.availability is not None
        assert adapter.mask_proj is not None
    assert model.target_adapter.availability.shape == (
        model.sequence_length, GUARDED_TARGET_CHANNELS
    )
    # And the boundary is the resolved one rather than a guess at it.
    assert int(model.target_adapter.availability[:, -1].sum()) == model.sequence_length - max(
        model.target_warmup_steps
    )


def test_the_zero_kl_init_invariant_survives_the_whole_stack(fit):
    r"""At initialisation $q(z_t \mid Y, U) = p(z_t \mid Y)$ exactly, so $K = 0$.

    Re-derived from a freshly built model rather than read off the trained run: after an epoch the
    KL is legitimately nonzero, and the question is whether the model *this config* builds starts at
    zero -- after config resolution, the kwarg sweep and the framework's own seeding have each had a
    chance to break it."""
    driver, _ = fit
    model = SeqVaeLagAttnCfs(**driver._build_model_kwargs()).eval()
    generator = torch.Generator().manual_seed(0)
    batch_size, seq_len = 2, model.sequence_length
    outputs = model(
        torch.randn(batch_size, seq_len, CAUSAL_ST_WIDTH, generator=generator),
        torch.randn(batch_size, seq_len, CAUSAL_PH_WIDTH, generator=generator),
        torch.randn(batch_size, seq_len, model.c_u, generator=generator),
        torch.zeros(batch_size, dtype=torch.long),
    )

    assert float(outputs["kld_per_t"].abs().max()) == 0.0
    assert torch.equal(outputs["z_prior"], outputs["mu_prior"])
    assert outputs["mu_base"].shape == (batch_size, TILE_COUNT, SHIPPED_HORIZON,
                                        GUARDED_TARGET_CHANNELS)
    assert not torch.equal(outputs["mu_base"], outputs["mu_full"])


# --------------------------------------------------------------------------------------
# What the run records
# --------------------------------------------------------------------------------------
def test_every_declared_metric_reaches_the_logger(fit):
    """The gap between "the task emits it" and "a callback collected it" is silent otherwise. The
    two validation-only arms are the ones only a real validation loop can prove wired."""
    _, trainer = fit

    for name in (
        "train/total_loss",
        "train/main_loss",
        "train/grad_norm",
        "train/source_conditioned_kl_raw",
        "train/target_warm_frac",
        "train/anchors_per_sample",
        "val/total_loss",
        "val/nll_shuffled_block",
        "val/kld_shuffled",
        "val/shuffle_penalty",
        "val/kld_source_null",
    ):
        assert name in trainer.callback_metrics, f"{name} never reached callback_metrics"


def test_the_metrics_csv_carries_every_tracked_key_and_no_all_nan_column(fit):
    """Both halves of the tracked list's contract, on a real run: a name the framework never emits
    is a column that is NaN in every row of every run, and a tracked name that produced no column at
    all is a readout nothing ever recorded."""
    driver, _ = fit
    frame = _metrics(driver)

    missing = [name for name in LagAttnCfsTrainer.TRACKED_METRICS if name not in frame.columns]
    assert missing == [], f"tracked but never written to the CSV: {missing}"
    all_nan = [column for column in frame.columns if frame[column].isna().all()]
    assert all_nan == [], f"columns that are NaN for every epoch: {all_nan}"


def test_the_warm_fraction_is_exactly_one_on_every_row_of_both_stages(fit):
    """A **stamped provenance column**, not a runtime measurement: it is resolved at construction
    and the constructor already refuses a violating budget-and-floor pairing, so a value other than
    $1.0$ means the checkpoint was built by code that predates that refusal. It is here because it
    is what makes the pairing readable off a run months later."""
    driver, _ = fit
    frame = _metrics(driver)

    for stage in ("train", "val"):
        column = frame[f"{stage}/target_warm_frac"].dropna()
        assert not column.empty, stage
        assert (column == 1.0).all(), f"{stage}/target_warm_frac is not exactly 1.0: {list(column)}"


def test_the_anchor_count_sits_at_its_geometry_derived_value_on_both_stages(fit):
    """The other geometry guard. Training tiles, so the count is one of the two tile counts the
    phase can produce; validation decodes every valid anchor, so it is exactly the dense range's
    length. A value off either band means the tiling is not the one the configuration states."""
    driver, _ = fit
    frame = _metrics(driver)

    train = frame["train/anchors_per_sample"].dropna()
    val = frame["val/anchors_per_sample"].dropna()

    assert not train.empty and not val.empty
    assert bool(((train >= TILE_COUNT - 1) & (train <= TILE_COUNT)).all()), list(train)
    assert (val == float(DENSE_ANCHORS)).all(), list(val)


def test_both_gap_splits_recompose_to_the_gap_in_the_same_row(fit):
    """This model's whole observability addition, on a real run rather than on a stub batch. Both
    splits must recompose to the ``pred_gap`` in the same row: a column that reached the CSV carrying
    some other quantity would pass every test above.

    The tolerance is **absolute** rather than relative, and that is measured rather than lax:
    ``pred_gap`` is a difference of two sums over $1470$ coefficients running to $\\approx 10^{3}$
    nats, so it loses several decimal digits to float32 cancellation before either split is formed.
    """
    driver, _ = fit
    frame = _metrics(driver)

    for stage in ("train", "val"):
        rows = frame[
            [
                f"{stage}/pred_gap",
                *(f"{stage}/{name}" for name in ("pred_gap_st", "pred_gap_ph")),
                *(f"{stage}/pred_gap_warm_{part}" for part in ("lo", "mid", "hi")),
            ]
        ].dropna()
        assert not rows.empty, f"no epoch recorded the {stage} forecast gaps"

        blocks = rows[f"{stage}/pred_gap_st"] + rows[f"{stage}/pred_gap_ph"]
        tertiles = sum(
            rows[f"{stage}/pred_gap_warm_{part}"] for part in ("lo", "mid", "hi")
        )
        # Split against split is the sharp comparison: both difference the two branches elementwise
        # and neither pays the cancellation the gap itself does.
        assert blocks.sub(tertiles).abs().max() < 1e-3, stage
        assert blocks.sub(rows[f"{stage}/pred_gap"]).abs().max() < 1e-1, stage


def test_the_source_null_floor_is_reported_beside_the_coupling_readout(fit):
    """The single most important pair on the page: if the two sit close, the coupling readout is
    measuring the source availability *clock* rather than source content. Validation only, because
    it costs a source encode and never enters the objective."""
    driver, _ = fit
    frame = _metrics(driver)

    assert "train/kld_source_null" not in frame.columns
    column = frame["val/kld_source_null"].dropna()
    assert not column.empty
    assert bool((column >= 0.0).all())
    assert bool(column.notna().all())


def test_the_run_directory_has_the_expected_layout(fit):
    """The log sinks, the checkpoint directory and the resolved config a later offline pass needs."""
    driver, _ = fit

    assert (Path(driver.train_results_dir) / "full.log").is_file()
    assert (Path(driver.train_results_dir) / "metrics_history.csv").is_file()
    assert Path(driver.model_checkpoint_dir).is_dir()


def test_the_resolved_config_records_the_budget_the_run_actually_got(fit):
    """The threshold does not name a channel: what it resolves to depends on the shards' own
    attributes, and here it also decides the decoder's width. A run recording only the request would
    record neither what it got nor what its nats were summed over."""
    driver, _ = fit

    written = Path(driver.model_checkpoint_dir) / RESOLVED_CONFIG_FILENAME
    assert written.is_file()
    reloaded = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert "base" not in reloaded
    vae = reloaded["model_config"]["VAE_model"]
    assert vae["causal_warmup_budget_steps"] == 134
    assert vae["warmup_period"] == SHIPPED_WARMUP_PERIOD
    assert vae["anchor_stride"] == SHIPPED_HORIZON
    assert vae["causal_reach_budget_s"] is None


def test_the_checkpoint_is_written_under_this_models_stem(fit):
    """Three models writing ``lag-attn-rws-epoch=00.ckpt`` into a shared directory would be
    indistinguishable by name."""
    driver, _ = fit

    checkpoints = list(Path(driver.model_checkpoint_dir).glob("*.ckpt"))

    assert checkpoints, "no checkpoint was written; Lightning's default would have gone elsewhere"
    assert all(path.name.startswith("lag-attn-cfs-epoch=") for path in checkpoints), [
        path.name for path in checkpoints
    ]


def test_the_checkpoint_carries_its_contract_and_reloads_at_its_own_width(fit):
    """The end of the road: a blob that describes itself and rebuilds without a config file **or a
    shard**. The stamped keep-index carries the decoder's width and the stamped warm-up vectors
    carry the availability terms, since the budget that produced both is resolved from data the
    loader may not have."""
    driver, _ = fit

    path = next(iter(Path(driver.model_checkpoint_dir).glob("*.ckpt")))
    blob = torch.load(path, map_location="cpu", weights_only=False)

    assert blob["model_class"] == "SeqVaeLagAttnCfs"
    assert blob["model_kwargs"] == driver._build_model_kwargs()
    assert len(blob["model_kwargs"]["target_keep_index"]) == GUARDED_TARGET_CHANNELS
    assert "decoder_out_channels" not in blob["model_kwargs"]
    check_model_class(blob, "SeqVaeLagAttnCfs")
    rebuilt = SeqVaeLagAttnCfs(**blob["model_kwargs"])
    assert rebuilt.decoder_out_channels == GUARDED_TARGET_CHANNELS
    assert load_checkpoint_strict(rebuilt, blob) is not None, (
        "the checkpoint's state dict did not align into a model rebuilt from its own kwargs"
    )
    # The run seed reaches the blob too, which is what lets a resumed run reproduce its tile grid.
    assert blob["hyper_parameters"]["seed"] == 42


def test_the_plotting_callback_draws_the_whole_page_for_every_epoch(fit):
    """The callback runs on every validation epoch and swallows its own exceptions by design, so
    what it did not draw is invisible on the training path -- which is why the files are counted
    here rather than the fit merely being observed to finish.

    One page per requested sample per epoch, plus **one** run-level warm-up figure for the whole
    fit: it describes the configuration rather than an epoch, and the callback's latch is what keeps
    a second one from being written on the second validation pass."""
    driver, trainer = fit
    directory = Path(driver.train_results_dir) / "lag_attn_rws_diagnostics"

    attached = [
        callback for callback in trainer.callbacks
        if type(callback).__name__ == "LagAttnRwsPlotCallback"
    ]

    assert len(attached) == 1
    assert trainer.state.finished
    assert directory.is_dir()
    pages = sorted(directory.glob("lag_attn_rws_epoch*.pdf"))
    assert len(pages) == SMOKE_EPOCHS * attached[0].num_examples
    assert [path.name for path in directory.glob("causal_*.pdf")] == [
        f"{BUDGET_FIGURE_STEM}.pdf"
    ]


def test_the_page_carries_both_input_rows_and_the_run_warns_about_neither(tmp_path):
    r"""The three quiet failures, asserted together and against a **real** fit's own module.

    Each of this model's three page builders replaces one welded to something it does not have --
    a dense $(T_{\mathrm{valid}}, H, C)$ anchor block, or the production two-sided filter bank --
    and each of the shipped ones raises inside a handler that warns and continues. So the symptom
    of a seam not being reached is a green suite, three log lines nobody reads, and a page with two
    rows missing beside a figure that was never written.

    Driven through the callback with the fit's own task rather than a hand-built one, because the
    thing that failed before was the *route*: the seams existed and this model's five-tensor forward
    could not travel down them.
    """
    from loguru import logger

    from teb_vae.lag_attn_rws.plotting import LagAttnRwsPlotCallback

    driver, trainer = _run_fit(tmp_path)
    # Through the callback's own fetch, so this draws the batch a real validation epoch drew.
    batch = plotting_module._first_validation_batch(trainer)
    callback = LagAttnRwsPlotCallback(tmp_path / "page", num_examples=1, file_format="png")

    messages = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    figures = []
    original = plotting_module.build_diagnostic_figure

    def _capture(**kwargs):
        figure = original(**kwargs)
        figures.append(figure)
        return figure

    plotting_module.build_diagnostic_figure = _capture
    try:
        callback._generate_plots(trainer, batch, driver.pl_model, epoch=0)
    finally:
        plotting_module.build_diagnostic_figure = original
        logger.remove(sink_id)

    try:
        assert messages == []
        titles = [ax.get_title() for ax in figures[0].axes if ax.get_title()]
        assert sum(title.startswith("Model input — target") for title in titles) == 1
        assert sum(title.startswith("Model input — source") for title in titles) == 1
        assert len(titles) == 9
        assert (callback.output_dir / f"{BUDGET_FIGURE_STEM}.png").exists()
        # The budget the figure describes reached the task from the driver, not from the
        # checkpoint: the channels it dropped are exactly what ``model_kwargs`` cannot carry.
        assert driver.pl_model.warmup_budget is driver.resolved_warmup
        assert driver.resolved_warmup.target.kept_width == GUARDED_TARGET_CHANNELS
    finally:
        import matplotlib.pyplot as plt

        plt.close("all")


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------
def test_two_runs_of_one_config_produce_an_identical_metric_row_set(tmp_path_factory):
    """Determinism as the requirement states it: an identical **metric row set**, not merely
    identical anchor indices.

    Identical anchors are necessary and nowhere near sufficient -- the reparameterisation $\\epsilon$
    and the permutation generator both move too -- and this cell adds one more thing that could
    break it and would not raise: the tile phase. Derived from a stable hash of the segment, the
    epoch and the seed, it survives a re-run; drawn from the global RNG it would not, and the only
    symptom would be two runs of one config disagreeing.

    Driven as two separate CPU processes under **different** ``PYTHONHASHSEED`` values; see
    :func:`_run_fit_in_subprocess` for why each of those three choices is there rather than an
    in-process GPU pair. The differing salt is not incidental: a phase derived from Python's own
    ``hash()`` would move with it, and nothing would raise -- $A_{\\max}$ is a geometry constant
    either way.
    """
    first = _run_fit_in_subprocess(tmp_path_factory.mktemp("det_a"), hash_seed="0")
    second = _run_fit_in_subprocess(tmp_path_factory.mktemp("det_b"), hash_seed="12345")

    assert list(first.columns) == list(second.columns)
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_under_the_shipped_settings_the_geometry_columns_are_still_exact(tmp_path_factory, fit):
    """What survives ``benchmark: true``, and what does not.

    Every number the model *computes* moves in the last few float32 digits, because the autotuner
    picks different reduction orders; measured over this fit, the worst run-to-run disagreement in
    any column is about $6 \\times 10^{-3}$ absolute, and the largest *relative* one lands on a
    tertile gap whose own value is near zero at initialisation.

    Nothing the run *decides* moves at all, and those are the columns that matter for reading a run
    months later: the stamped warm fraction and the decoded anchor count are functions of the
    geometry and the derived phase, so they are exact regardless of which convolution kernel the
    autotuner chose. If either of them drifted, the tiling itself would be non-reproducible.

    The bound is set from that measurement with room rather than tight to it: the largest single
    column is the pre-clip gradient norm, which runs to $10^{3}$, so a tenth of a nat there is
    seven significant figures.
    """
    repeat_driver, _ = _run_fit(tmp_path_factory.mktemp("smoke_repeat"))
    first, second = _metrics(fit[0]), _metrics(repeat_driver)

    assert list(first.columns) == list(second.columns)
    for column in (
        "train/target_warm_frac", "val/target_warm_frac",
        "train/anchors_per_sample", "val/anchors_per_sample",
        "epoch",
    ):
        pd.testing.assert_series_equal(first[column], second[column], check_exact=True)

    numeric = first.select_dtypes("number").columns
    worst = (first[numeric] - second[numeric]).abs().max().max()
    assert worst < 1e-1, f"the shipped settings drifted by {worst}, far beyond float32 noise"


def test_the_shipped_config_trades_bitwise_determinism_for_speed():
    """Recorded rather than left implicit, because the test above is otherwise a puzzling
    weakening: the config chooses cuDNN autotuning, which is what makes a production run fast and
    what makes two of them disagree in the last float32 digits. A run that needs bitwise
    reproducibility sets these two keys and pays for it."""
    trainer_block = load_config(str(_TINY))["advanced_config"]["trainer"]

    assert trainer_block["benchmark"] is True
    assert trainer_block["deterministic"] is False


def test_a_different_seed_moves_the_run(tmp_path_factory, fit):
    """The negative control on the determinism tests: an assertion that held for *every* seed would
    be describing a run that ignores its configuration rather than one that reproduces. The seed is
    load-bearing twice over here -- it seeds the framework, and it is one of the four halves of the
    tile-phase key."""
    reseeded_driver, _ = _run_fit(tmp_path_factory.mktemp("smoke_seed"), seed=7)
    baseline, reseeded = _metrics(fit[0]), _metrics(reseeded_driver)

    assert list(baseline.columns) == list(reseeded.columns)
    assert not baseline["train/total_loss"].equals(reseeded["train/total_loss"])
