r"""One real fit, through the real entry point, against the committed causal shard.

Everything else in this suite tests a piece in isolation. This runs the whole thing: config ->
pre-flight guards -> ``setup_config`` -> data module -> model -> ``build_trainer`` -> ``fit`` ->
checkpoint, on a CPU, in seconds. It is the only place the failures that live *between* the pieces
can surface -- a config key that reaches nothing, a metric name no callback collects, a callback that
raises on the first validation epoch, and, specific to this cell, three that arrive only from a real
run: the permutation control decoding at the wrong anchor set, the source-null arm re-encoding a
stream that is no longer in the forward dict, and the step-granular learning-rate ramp this
architecture needs, which a unit test can build but only a fit can show *attached*.

Driven through ``main`` rather than by assembling the driver by hand, deliberately: the four shared
pre-flight guards, this driver's own six, the temporary resolved-config file and the resolved-config
write beside the checkpoints hang off the entry point and are reached no other way.

Two epochs rather than the config's one: ``lr`` is logged at train-epoch *start* with
``on_epoch=True``, so its first CSV cell is always NaN, and the second epoch is what exercises the
scheduler stepping at all -- and, here, what makes the tile phase rotate, since the epoch is one of
the four halves of its key.

**What this file does not assert is the diagnostic page's contents.** Both replaced row builders live
in the conv-LSTM cell of this row and are asserted where they are written; here the page is a
resolution rather than a drawing, and its failures are swallowed by design -- a figure is never worth
failing a multi-day fit for. What is asserted is what must hold whatever the page did: the callback
is attached, the fit finished, and no exception escaped the validation epoch it draws on.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME
from teb_vae.lag_attn_transformer_crws import trainer as trainer_module
from teb_vae.lag_attn_transformer_crws.nets.model import SeqVaeLagAttnTrfCrws
from teb_vae.lag_attn_transformer_crws.trainer import LagAttnTrfCrwsTrainer
from train.graph_models_utils import check_model_class, load_checkpoint_strict

from .conftest import (
    CAUSAL_C_U,
    CAUSAL_C_Y,
    CAUSAL_PH_WIDTH,
    CAUSAL_ST_WIDTH,
    SHIPPED_BUDGET_STEPS,
    SHIPPED_HORIZON,
    SHIPPED_SEQUENCE_LENGTH,
    SHIPPED_WARMUP_PERIOD,
    absolutize_dataset_paths,
)

pytestmark = pytest.mark.slow

_TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"

#: Epochs the fit runs. See the module docstring for why it is not the config's one.
SMOKE_EPOCHS = 2

#: The run seed ``configs/default.yaml`` ships and ``tiny.yaml`` inherits. One of the four halves of
#: the tile-phase key, which is why it has to reach the checkpoint.
SHIPPED_SEED = 42

#: How far two fits of the shipped configuration may differ on a column the model *computes*, as
#: ``atol + rtol * column magnitude``. Both parts are needed and each is set about an order of
#: magnitude above what was measured: the absolute part covers the columns whose noise is
#: cancellation rather than magnitude -- ``pred_gap`` is a difference of two block scores and moves
#: on a magnitude below $1$ -- and the relative part covers ``grad_norm``, simply the largest number
#: the run reports. See the test that reads them.
_DRIFT_ATOL = 5.0e-3
_DRIFT_RTOL = 2.0e-3

#: What the shipped warm-up budget resolves to on the **input** side. Pinned so a "guarded" fit
#: cannot silently be the unguarded one -- which here would change what the encoders read while
#: leaving the decoder's width, and therefore every shape on the page, exactly as it was.
GUARDED_TARGET_CHANNELS = 98
GUARDED_SOURCE_CHANNELS = CAUSAL_C_U

#: Raw samples a horizon token emits. The decoder's width, and the *only* thing that decides it:
#: unlike the causal-feature cells, no budget and no gate can move it.
RAW_PER_STEP = 16

#: The anchor counts the two stages must produce, derived here the way the model derives them so a
#: geometry change re-derives them rather than failing a literal.
DENSE_ANCHORS = SHIPPED_SEQUENCE_LENGTH - SHIPPED_HORIZON - SHIPPED_WARMUP_PERIOD
TILE_COUNT = -(-DENSE_ANCHORS // SHIPPED_HORIZON)

#: The ramp ``configs/tiny.yaml`` shortens to, so a one-epoch smoke does not saturate it instantly.
TINY_LR_WARMUP_STEPS = 4


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
    original_train_model = LagAttnTrfCrwsTrainer.train_model

    def _capture_train_model(self, train_loader, validation_loader):
        result = original_train_model(self, train_loader, validation_loader)
        captured["driver"] = self
        captured["trainer"] = result
        return result

    LagAttnTrfCrwsTrainer.train_model = _capture_train_model
    try:
        trainer_module.main(str(config_path))
    finally:
        # Deleted rather than reassigned: the method is inherited, and leaving a copy on the
        # subclass would shadow a later change to the one it inherits.
        del LagAttnTrfCrwsTrainer.train_model

    return captured["driver"], captured["trainer"]


def _metrics(driver) -> pd.DataFrame:
    return pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")


def _run_fit_in_subprocess(tmp_path, hash_seed: str) -> pd.DataFrame:
    """Run one fit in a **separate process, on the CPU**, and return its metrics history.

    Four deliberate choices, each removing a source of variation that has nothing to do with what
    this package decides:

    * a separate process, because "two runs of one config" is literally two processes -- and because
      it is the only way to vary ``PYTHONHASHSEED``, whose per-process salt is what would break a
      tile phase derived from Python's own ``hash()`` with nothing raising;
    * the CPU, because the shipped config chooses cuDNN autotuning, which selects convolution
      algorithms by timing them and therefore accumulates the same sums in different orders. No seed
      controls that. Forcing ``deterministic: true`` instead does not work from inside a test
      session: it needs ``CUBLAS_WORKSPACE_CONFIG`` set before CUDA is initialised, and by then
      another test has initialised it;
    * **one intra-op thread**, because the CPU pool is the same hazard in another spelling: a
      reduction split across $n$ workers accumulates in the order they finish, and the split itself
      moves with how loaded the machine is. Measured rather than assumed on the conv-LSTM cell of
      this row, and inherited here because it is a property of the pool rather than of an encoder --
      this architecture's attention reductions are if anything wider, not narrower;
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
            "from teb_vae.lag_attn_transformer_crws.trainer import main; main(sys.argv[2])",
            str(repo_root),
            str(config_path),
        ],
        env={
            **dict(os.environ),
            "PYTHONHASHSEED": hash_seed,
            "CUDA_VISIBLE_DEVICES": "",
            # Set in the environment rather than through ``torch.set_num_threads``: the pool is
            # sized when torch is imported, and the entry point below imports it before any line
            # of ours runs.
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
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
def test_the_fit_completes(fit) -> None:
    _, trainer = fit

    assert trainer.current_epoch == SMOKE_EPOCHS
    assert trainer.state.finished


def test_the_losses_stay_finite(fit) -> None:
    _, trainer = fit

    for name, value in trainer.callback_metrics.items():
        assert math.isfinite(float(value)), f"{name} is {float(value)}"


def test_the_spike_breaker_never_latched(fit) -> None:
    """The margin is this encoder's own measurement, and both directions of getting it wrong are
    silent: a margin that never fires leaves the breaker as its non-finite guard alone, and a margin
    that skipped ordinary batches would read in a log exactly like a model that keeps blowing up.
    Only this column separates the second from a real divergence."""
    driver, _ = fit
    frame = _metrics(driver)

    assert float(frame["train/spike_skipped"].max()) == 0.0


def test_the_gradient_norm_stays_finite(fit) -> None:
    """The quantity ``gradient_clip_val`` is re-derived from. If it were non-finite the clip
    coefficient would be zero and the run would train nothing while completing normally."""
    _, trainer = fit

    grad_norm = float(trainer.callback_metrics["train/grad_norm"])

    assert math.isfinite(grad_norm), f"train/grad_norm is {grad_norm}"


def test_the_step_granular_learning_rate_ramp_was_live_during_the_fit(fit) -> None:
    """The half of the diamond the conv-Transformer parent exists for, asserted on a fit rather than
    on a hand-built optimizer.

    Three things have to line up and each fails silently on its own: the config's key has to reach
    ``hparams`` (it travels through ``create_model``, which the *causal* parent also defines), the
    task's ``build_lr_scheduler`` has to be the conv-Transformer parent's rather than the shared
    task's, and Lightning has to have attached the result at ``interval='step'``. Attached at
    ``'epoch'`` the ramp would take ``lr_warmup_steps`` *epochs* to complete, with the same class,
    the same lambda and nothing in the log saying so.
    """
    driver, trainer = fit

    assert int(driver.pl_model.hparams["lr_warmup_steps"]) == TINY_LR_WARMUP_STEPS
    configs = list(trainer.lr_scheduler_configs)
    assert len(configs) == 1, configs
    assert configs[0].interval == "step"
    assert isinstance(configs[0].scheduler, LambdaLR)

    factor = configs[0].scheduler.lr_lambdas[0]
    assert factor(0) == pytest.approx(1.0 / TINY_LR_WARMUP_STEPS)
    assert factor(TINY_LR_WARMUP_STEPS - 1) == pytest.approx(1.0)


def test_the_run_trains_at_the_budgets_width_and_the_configs_tiling(fit) -> None:
    """The whole binding, end to end: config -> shard attributes -> channel tuples -> adapter width,
    and config -> stride -> decoded anchor set.

    A unit test can check each link; only a fit can check they are connected, and the failure this
    catches is silent: input adapters built from the *declared* $c_y$ would run to completion reading
    the four channels whose warm-up the budget dropped as though they were signal.

    The decoder is the load-bearing **non**-binding: it is $R$ raw samples per horizon token whatever
    the budget resolves to, which is the one thing that separates this row from the causal-feature
    one it shares every input with."""
    driver, _ = fit
    model = driver.pytorch_model

    assert isinstance(model, SeqVaeLagAttnTrfCrws)
    assert model.decoder_out_channels == model.raw_per_step == RAW_PER_STEP
    assert model.target_adapter.linear.in_features == GUARDED_TARGET_CHANNELS
    assert model.source_adapter.linear.in_features == GUARDED_SOURCE_CHANNELS
    assert model.anchor_stride == model.horizon == SHIPPED_HORIZON
    assert model.warmup_period == SHIPPED_WARMUP_PERIOD
    # The declared widths are untouched, which is what the data boundary checks against.
    assert (model.c_y, model.c_u) == (CAUSAL_C_Y, CAUSAL_C_U)
    # And the encoder a launch built is this package's, not the conv-LSTM cell's.
    assert not any(isinstance(module, torch.nn.LSTM) for module in model.modules())


def test_the_input_adapters_carry_the_availability_terms_the_warm_up_needs(fit) -> None:
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


def test_the_zero_kl_init_invariant_survives_the_whole_stack(fit) -> None:
    r"""At initialisation $q(z_t \mid Y, U) = p(z_t \mid Y)$ exactly, so $K = 0$.

    Re-derived from a freshly built model rather than read off the trained run: after an epoch the
    KL is legitimately nonzero, and the question is whether the model *this config* builds starts at
    zero -- after config resolution, the kwarg sweep and the framework's own seeding have each had a
    chance to break it.

    The two forecast branches are **not** bitwise equal here and that is the shipped
    ``base_decode: mean``: the base branch decodes at $\mu^p$ while the full branch still carries the
    posterior's own draw. The KL is exactly zero regardless, because it depends on the two
    distributions rather than on the samples taken from them."""
    driver, _ = fit
    model = SeqVaeLagAttnTrfCrws(**driver._build_model_kwargs()).eval()
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
    assert outputs["mu_base"].shape == (batch_size, TILE_COUNT, SHIPPED_HORIZON, RAW_PER_STEP)
    assert not torch.equal(outputs["mu_base"], outputs["mu_full"])


# --------------------------------------------------------------------------------------
# What the run records
# --------------------------------------------------------------------------------------
def test_every_declared_metric_reaches_the_logger(fit) -> None:
    """The gap between "the task emits it" and "a callback collected it" is silent otherwise. The
    two validation-only arms are the ones only a real validation loop can prove wired."""
    _, trainer = fit

    for name in (
        "train/total_loss",
        "train/main_loss",
        "train/grad_norm",
        "train/pred_gap",
        "train/source_conditioned_kl_raw",
        "train/anchors_per_sample",
        "train/source_lag_warmth_frac_st",
        "train/source_lag_warmth_frac_ph",
        "val/total_loss",
        "val/nll_shuffled_block",
        "val/kld_shuffled",
        "val/shuffle_penalty",
        "val/kld_source_null",
    ):
        assert name in trainer.callback_metrics, f"{name} never reached callback_metrics"


def test_the_metrics_csv_carries_every_tracked_key_and_no_all_nan_column(fit) -> None:
    """Both halves of the tracked list's contract, on a real run: a name the framework never emits
    is a column that is NaN in every row of every run, and a tracked name that produced no column at
    all is a readout nothing ever recorded."""
    driver, _ = fit
    frame = _metrics(driver)

    missing = [
        name for name in LagAttnTrfCrwsTrainer.TRACKED_METRICS if name not in frame.columns
    ]
    assert missing == [], f"tracked but never written to the CSV: {missing}"
    all_nan = [column for column in frame.columns if frame[column].isna().all()]
    assert all_nan == [], f"columns that are NaN for every epoch: {all_nan}"


def test_the_five_target_channel_columns_are_absent_rather_than_vacuous(fit) -> None:
    """The readouts the causal-feature cells report and this row drops. Every one partitions kept
    **target** channels, and this target has none -- its last axis counts raw samples, which have no
    warm-up, no filter and no order to rank by -- so a re-pointed column would be a measurement of
    nothing that reads exactly like a measurement."""
    driver, _ = fit
    frame = _metrics(driver)

    for stage in ("train", "val"):
        for name in (
            "target_warm_frac", "pred_gap_st", "pred_gap_ph",
            "pred_gap_warm_lo", "pred_gap_warm_mid", "pred_gap_warm_hi",
        ):
            assert f"{stage}/{name}" not in frame.columns, f"{stage}/{name}"


def test_the_anchor_count_sits_at_its_geometry_derived_value_on_both_stages(fit) -> None:
    """The geometry guard, and the one column of the three that is not a result. Training tiles, so
    the count is one of the two tile counts the phase can produce; validation decodes every valid
    anchor, so it is exactly the dense range's length. A value off either band means the tiling is
    not the one the configuration states."""
    driver, _ = fit
    frame = _metrics(driver)

    train = frame["train/anchors_per_sample"].dropna()
    val = frame["val/anchors_per_sample"].dropna()

    assert not train.empty and not val.empty
    assert bool(((train >= TILE_COUNT - 1) & (train <= TILE_COUNT)).all()), list(train)
    assert (val == float(DENSE_ANCHORS)).all(), list(val)


def test_the_headline_gap_is_recorded_on_both_stages_and_is_finite(fit) -> None:
    """The readout this cell exists to produce: the same quantity the raw-signal siblings report,
    over the same raw target, computed from inputs that do not contain the answer.

    Its **sign** is not a criterion here or anywhere. What is asserted is that the column exists, is
    finite on every recorded row, and is the difference of the two block scores in the same row --
    a column carrying some other quantity would satisfy every other test in this file."""
    driver, _ = fit
    frame = _metrics(driver)

    for stage in ("train", "val"):
        rows = frame[
            [f"{stage}/pred_gap", f"{stage}/nll_base_block", f"{stage}/nll_full_block"]
        ].dropna()
        assert not rows.empty, f"no epoch recorded the {stage} forecast gap"
        assert bool(rows[f"{stage}/pred_gap"].apply(math.isfinite).all())
        recomposed = rows[f"{stage}/nll_base_block"] - rows[f"{stage}/nll_full_block"]
        # Absolute rather than relative, and measured rather than lax: the gap is a difference of
        # two sums over 240 raw samples, so it loses several decimal digits to float32 cancellation
        # before it is ever written, and at initialisation it is near zero -- which is exactly the
        # regime in which a relative tolerance means nothing.
        assert recomposed.sub(rows[f"{stage}/pred_gap"]).abs().max() < 1e-3, stage


def test_the_two_source_warmth_columns_are_fractions_and_move_together(fit) -> None:
    """The compromise the design makes on the source, sized rather than argued about: every source
    channel is kept, including those still inside their own warm-up over most of the window, and lag
    attention searches back into exactly that region. Both columns are fractions by construction, so
    a value outside $[0, 1]$ is an inverted or mis-normalised metric rather than a surprising run."""
    driver, _ = fit
    frame = _metrics(driver)

    for stage in ("train", "val"):
        for block in ("st", "ph"):
            column = frame[f"{stage}/source_lag_warmth_frac_{block}"].dropna()
            assert not column.empty, f"{stage}/{block}"
            assert bool(((column >= 0.0) & (column <= 1.0)).all()), list(column)


def test_the_source_null_floor_is_reported_beside_the_coupling_readout(fit) -> None:
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


def test_the_two_readouts_the_beta_pair_is_answered_from_are_present_and_finite(fit) -> None:
    r"""$\beta = 1.0$ and $\beta_p = 0.1$ are inherited across two edges of the grid, so whether they
    transfer is an open question rather than a settled one -- and these are the two columns it is
    answered from. Asserted present and finite rather than inside a band: a band would be this file
    inventing the answer."""
    driver, _ = fit
    frame = _metrics(driver)

    for stage in ("train", "val"):
        for name in ("kld_active_frac", "logvar_prior_floor_frac"):
            column = frame[f"{stage}/{name}"].dropna()
            assert not column.empty, f"{stage}/{name}"
            assert bool(column.apply(math.isfinite).all()), list(column)


def test_the_run_directory_has_the_expected_layout(fit) -> None:
    """The log sinks, the checkpoint directory and the resolved config a later offline pass needs."""
    driver, _ = fit

    assert (Path(driver.train_results_dir) / "full.log").is_file()
    assert (Path(driver.train_results_dir) / "metrics_history.csv").is_file()
    assert Path(driver.model_checkpoint_dir).is_dir()


def test_the_resolved_config_records_the_budget_the_run_actually_got(fit) -> None:
    """The threshold does not name a channel: what it resolves to depends on the shards' own
    attributes. A run recording only the request would record neither what it got nor which input
    channels its encoders were fed."""
    driver, _ = fit

    written = Path(driver.model_checkpoint_dir) / RESOLVED_CONFIG_FILENAME
    assert written.is_file()
    reloaded = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert "base" not in reloaded
    vae = reloaded["model_config"]["VAE_model"]
    assert vae["causal_warmup_budget_steps"] == SHIPPED_BUDGET_STEPS
    assert vae["warmup_period"] == SHIPPED_WARMUP_PERIOD
    assert vae["anchor_stride"] == SHIPPED_HORIZON
    assert vae["causal_reach_budget_s"] is None
    # And the encoder half, which is what a later reader needs to know which cell wrote the run.
    assert vae["target_attention_blocks"] == 2
    assert "lstm_layers" not in vae


def test_the_checkpoint_is_written_under_this_models_stem(fit) -> None:
    """Eight models writing ``lag-attn-rws-epoch=00.ckpt`` into a shared directory would be
    indistinguishable by name."""
    driver, _ = fit

    checkpoints = list(Path(driver.model_checkpoint_dir).glob("*.ckpt"))

    assert checkpoints, "no checkpoint was written; Lightning's default would have gone elsewhere"
    assert all(path.name.startswith("lag-attn-trf-crws-epoch=") for path in checkpoints), [
        path.name for path in checkpoints
    ]


def test_the_checkpoint_carries_its_contract_and_reloads_with_no_shard_present(
    fit, tmp_path, monkeypatch
) -> None:
    """The end of the road: a blob that describes itself and rebuilds without a config file **or a
    shard**.

    The stamped keep-index carries which channels the encoders read and the stamped warm-up vectors
    carry the availability terms, and the budget that resolved both is a property of the *data* --
    so a blob recording only the threshold could not be rebuilt anywhere the shards are not.

    Driven from a directory containing no HDF5 at all, with the working directory moved there, so a
    rebuild that reached for a shard by a relative path fails rather than quietly finding one."""
    driver, _ = fit

    path = next(iter(Path(driver.model_checkpoint_dir).glob("*.ckpt")))
    blob = torch.load(path, map_location="cpu", weights_only=False)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)

    assert blob["model_class"] == "SeqVaeLagAttnTrfCrws"
    assert blob["model_kwargs"] == driver._build_model_kwargs()
    assert len(blob["model_kwargs"]["target_keep_index"]) == GUARDED_TARGET_CHANNELS
    assert "decoder_out_channels" not in blob["model_kwargs"]
    check_model_class(blob, "SeqVaeLagAttnTrfCrws")
    assert not list(empty.glob("*.hdf5"))
    rebuilt = SeqVaeLagAttnTrfCrws(**blob["model_kwargs"])
    assert rebuilt.decoder_out_channels == RAW_PER_STEP
    assert rebuilt.target_adapter.linear.in_features == GUARDED_TARGET_CHANNELS
    assert load_checkpoint_strict(rebuilt, blob) is not None, (
        "the checkpoint's state dict did not align into a model rebuilt from its own kwargs"
    )
    # The run seed reaches the blob too, which is what lets a resumed run reproduce its tile grid.
    assert blob["hyper_parameters"]["seed"] == SHIPPED_SEED
    # And so does the ramp, which is the conv-Transformer half of the schedule.
    assert blob["hyper_parameters"]["lr_warmup_steps"] == TINY_LR_WARMUP_STEPS


def test_a_resume_from_the_written_checkpoint_reproduces_the_same_tile_grid(fit) -> None:
    r"""The property the determinism requirement rests on and nothing else tests: the per-segment
    phase is a stable hash of the recording identifier, the segment's own start time, the training
    epoch and the **seed**, and a resumed run learns that seed only from the checkpoint's
    hyperparameters -- the framework reads ``general_config.seed`` once, into its own determinism
    setup, and hands it to no task.

    Reconstructed the way a resume would: a task rebuilt from the blob's recorded seed must resolve
    the same $\varphi$, and therefore the same ``anchor_index``, as one carrying the seed the run was
    launched with, for the same ``(guid, domain_start, epoch)``. A resume that lost it would re-tile
    every segment from the epoch it resumed at -- and $A_{\max}$ is a geometry constant either way,
    so no shape, no count and no metric would differ.

    The negative control is the second half rather than a separate test: without it every assertion
    here would also hold for a phase that ignored the seed entirely."""
    driver, _ = fit
    from .conftest import make_stub_batch

    path = next(iter(Path(driver.model_checkpoint_dir).glob("*.ckpt")))
    blob = torch.load(path, map_location="cpu", weights_only=False)

    def _task(seed):
        """A trainer-less task at one seed. ``current_epoch`` is $0$ on all three, which is what
        holds the epoch half of the key fixed while the seed half is varied."""
        return type(driver.pl_model)(
            SeqVaeLagAttnTrfCrws(**blob["model_kwargs"]),
            lr=1e-3,
            model_kwargs=blob["model_kwargs"],
            seed=seed,
        )

    batch = make_stub_batch(2, SHIPPED_SEQUENCE_LENGTH)
    resumed = _task(blob["hyper_parameters"]["seed"])
    as_launched = _task(SHIPPED_SEED)
    reseeded = _task(SHIPPED_SEED + 1)

    resumed_phase = resumed.anchor_phase(batch)
    assert torch.equal(resumed_phase, as_launched.anchor_phase(batch))
    assert not torch.equal(resumed_phase, reseeded.anchor_phase(batch))

    # And the phase is only the key: what a run is reproducing is the decoded anchor set itself.
    device = torch.device("cpu")
    resumed_index, resumed_valid = resumed.orig_model._build_anchor_index(
        batch=2, device=device, anchor_phase=resumed_phase
    )
    launched_index, launched_valid = as_launched.orig_model._build_anchor_index(
        batch=2, device=device, anchor_phase=as_launched.anchor_phase(batch)
    )
    other_index, _ = reseeded.orig_model._build_anchor_index(
        batch=2, device=device, anchor_phase=reseeded.anchor_phase(batch)
    )

    assert torch.equal(resumed_index, launched_index)
    assert torch.equal(resumed_valid, launched_valid)
    assert resumed_index.shape == other_index.shape  # A_max is a geometry constant either way
    assert not torch.equal(resumed_index, other_index)


def test_the_plotting_callback_is_attached_and_the_fit_survives_whatever_it_drew(fit) -> None:
    """What must hold regardless of the page. The callback runs on every validation epoch and
    swallows its own exceptions by design, so a page that failed is invisible on the training path
    -- which cuts both ways: it cannot abort the fit, and it cannot be asserted from here either.

    Attached exactly once, the fit finished, and the output directory exists. What is *in* it is not
    this file's claim: both replaced row builders live in the conv-LSTM cell of this row and are
    drawn and asserted there."""
    driver, trainer = fit
    directory = Path(driver.train_results_dir) / "lag_attn_rws_diagnostics"

    attached = [
        callback for callback in trainer.callbacks
        if type(callback).__name__ == "LagAttnRwsPlotCallback"
    ]

    assert len(attached) == 1
    assert trainer.state.finished
    assert directory.is_dir()


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------
def test_two_runs_of_one_config_produce_an_identical_metric_row_set(tmp_path_factory) -> None:
    r"""Determinism as the requirement states it: an identical **metric row set**, not merely
    identical anchor indices.

    Identical anchors are necessary and nowhere near sufficient -- the reparameterisation $\epsilon$
    and the permutation generator both move too -- and this cell adds one more thing that could
    break it and would not raise: the tile phase. Derived from a stable hash of the segment, the
    epoch and the seed, it survives a re-run; drawn from the global RNG it would not, and the only
    symptom would be two runs of one config disagreeing.

    Driven as two separate CPU processes under **different** ``PYTHONHASHSEED`` values; see
    :func:`_run_fit_in_subprocess` for why each of those three choices is there rather than an
    in-process GPU pair. The differing salt is not incidental: a phase derived from Python's own
    ``hash()`` would move with it, and nothing would raise -- $A_{\max}$ is a geometry constant
    either way.
    """
    first = _run_fit_in_subprocess(tmp_path_factory.mktemp("det_a"), hash_seed="0")
    second = _run_fit_in_subprocess(tmp_path_factory.mktemp("det_b"), hash_seed="12345")

    assert list(first.columns) == list(second.columns)
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_under_the_shipped_settings_the_geometry_columns_are_still_exact(
    tmp_path_factory, fit
) -> None:
    r"""What survives ``benchmark: true``, and what does not.

    Every number the model *computes* moves in the last few float32 digits, because the autotuner
    picks different reduction orders. Nothing the run *decides* moves at all, and that is the column
    that matters for reading a run months later: the decoded anchor count is a function of the
    geometry and the derived phase, so it is exact regardless of which convolution kernel the
    autotuner chose. If it drifted, the tiling itself would be non-reproducible.

    The bound on everything else is **per column, and has both an absolute and a relative part**,
    because the two columns that move most do so for two different reasons and one bound cannot
    describe both: ``train/grad_norm`` is simply the largest number the run reports, so its movement
    is relative, while ``train/pred_gap`` is the difference of two block scores on a magnitude below
    $1$, so its movement is cancellation and is absolute. A single absolute bound is vacuous for the
    first and unmeetable for the second.
    """
    repeat_driver, _ = _run_fit(tmp_path_factory.mktemp("smoke_repeat"))
    first, second = _metrics(fit[0]), _metrics(repeat_driver)

    assert list(first.columns) == list(second.columns)
    for column in ("train/anchors_per_sample", "val/anchors_per_sample", "epoch"):
        pd.testing.assert_series_equal(first[column], second[column], check_exact=True)

    numeric = first.select_dtypes("number").columns
    drift = (first[numeric] - second[numeric]).abs().max()
    allowed = _DRIFT_ATOL + _DRIFT_RTOL * first[numeric].abs().max()
    offenders = {
        str(column): float(drift[column])
        for column in numeric
        if drift[column] > allowed[column]
    }
    assert not offenders, (
        f"the shipped settings drifted beyond float32 noise on {offenders}, against "
        f"{ {name: float(allowed[name]) for name in offenders} }"
    )


def test_the_shipped_config_trades_bitwise_determinism_for_speed() -> None:
    """Recorded rather than left implicit, because the test above is otherwise a puzzling
    weakening: the config chooses cuDNN autotuning, which is what makes a production run fast and
    what makes two of them disagree in the last float32 digits. A run that needs bitwise
    reproducibility sets these two keys and pays for it."""
    trainer_block = load_config(str(_TINY))["advanced_config"]["trainer"]

    assert trainer_block["benchmark"] is True
    assert trainer_block["deterministic"] is False


def test_a_different_seed_moves_the_run(tmp_path_factory, fit) -> None:
    """The negative control on the determinism tests: an assertion that held for *every* seed would
    be describing a run that ignores its configuration rather than one that reproduces. The seed is
    load-bearing twice over here -- it seeds the framework, and it is one of the four halves of the
    tile-phase key."""
    reseeded_driver, _ = _run_fit(tmp_path_factory.mktemp("smoke_seed"), seed=7)
    baseline, reseeded = _metrics(fit[0]), _metrics(reseeded_driver)

    assert list(baseline.columns) == list(reseeded.columns)
    assert not baseline["train/total_loss"].equals(reseeded["train/total_loss"])
