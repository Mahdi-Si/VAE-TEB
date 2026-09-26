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
from teb_vae.lag_attn_transformer_crws import trainer as trainer_module
from teb_vae.lag_attn_transformer_crws.nets.model import SeqVaeLagAttnTrfCrws
from teb_vae.lag_attn_transformer_crws.trainer import LagAttnTrfCrwsTrainer
from train.graph_models_utils import check_model_class, load_checkpoint_strict

from .conftest import (
    CAUSAL_PH_WIDTH,
    CAUSAL_ST_WIDTH,
    SHIPPED_HORIZON,
    SHIPPED_SEQUENCE_LENGTH,
    SHIPPED_WARMUP_PERIOD,
    absolutize_dataset_paths,
)

pytestmark = pytest.mark.slow

_TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"

#: Epochs the fit runs. See the module docstring for why it is not the config's one.
SMOKE_EPOCHS = 2

#: The anchor counts the two stages must produce, derived here the way the model derives them so a
#: geometry change re-derives them rather than failing a literal.
DENSE_ANCHORS = SHIPPED_SEQUENCE_LENGTH - SHIPPED_HORIZON - SHIPPED_WARMUP_PERIOD
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
def test_the_fit_completes_with_every_reported_number_finite(fit) -> None:
    """Including ``train/grad_norm``, the quantity ``gradient_clip_val`` is re-derived from: were it
    non-finite the clip coefficient would be zero and the run would train nothing while completing
    normally."""
    _, trainer = fit

    assert trainer.current_epoch == SMOKE_EPOCHS
    assert trainer.state.finished
    assert "train/grad_norm" in trainer.callback_metrics
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
    warmup = int(load_config(str(_TINY))["general_config"]["lr_warmup_steps"])

    assert int(driver.pl_model.hparams["lr_warmup_steps"]) == warmup
    configs = list(trainer.lr_scheduler_configs)
    assert len(configs) == 1, configs
    assert configs[0].interval == "step"
    assert isinstance(configs[0].scheduler, LambdaLR)

    factor = configs[0].scheduler.lr_lambdas[0]
    assert factor(0) == pytest.approx(1.0 / warmup)
    assert factor(warmup - 1) == pytest.approx(1.0)


def test_the_run_trains_at_the_budgets_width_and_the_configs_tiling(fit) -> None:
    r"""The whole binding, end to end: config -> shard attributes -> channel tuples -> adapter width
    and availability terms, and config -> stride -> decoded anchor set.

    A unit test can check each link; only a fit can check they are connected, and the failure this
    catches is silent: input adapters built from the *declared* $c_y$, or without the availability
    terms, would run to completion reading channels whose warm-up the budget dropped as though they
    were signal. The decoder is the load-bearing **non**-binding: it is $R$ raw samples per horizon
    token whatever the budget resolves to."""
    driver, _ = fit
    model = driver.pytorch_model
    kwargs = driver._build_model_kwargs()
    kept_target = len(kwargs["target_keep_index"])

    assert isinstance(model, SeqVaeLagAttnTrfCrws)
    assert model.decoder_out_channels == model.raw_per_step
    assert model.target_adapter.linear.in_features == kept_target < model.c_y
    assert model.source_adapter.linear.in_features == len(kwargs["source_keep_index"])
    for adapter in (model.target_adapter, model.source_adapter):
        assert adapter.availability is not None
        assert adapter.mask_proj is not None
    assert model.target_adapter.availability.shape == (model.sequence_length, kept_target)
    assert model.anchor_stride == model.horizon == SHIPPED_HORIZON
    assert model.warmup_period == SHIPPED_WARMUP_PERIOD


def test_the_zero_kl_init_invariant_survives_the_whole_stack(fit) -> None:
    r"""At initialisation $q(z_t \mid Y, U) = p(z_t \mid Y)$ exactly, so $K = 0$.

    Re-derived from a freshly built model rather than read off the trained run: after an epoch the
    KL is legitimately nonzero, and the question is whether the model *this config* builds starts at
    zero -- after config resolution, the kwarg sweep and the framework's own seeding have each had a
    chance to break it. Exact regardless of the shipped ``base_decode``, because the KL depends on
    the two distributions rather than on the samples taken from them."""
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

    general = load_config(str(_TINY))["general_config"]

    assert blob["model_class"] == "SeqVaeLagAttnTrfCrws"
    assert blob["model_kwargs"] == driver._build_model_kwargs()
    check_model_class(blob, "SeqVaeLagAttnTrfCrws")
    assert not list(empty.glob("*.hdf5"))
    rebuilt = SeqVaeLagAttnTrfCrws(**blob["model_kwargs"])
    assert rebuilt.target_adapter.linear.in_features == len(
        blob["model_kwargs"]["target_keep_index"]
    )
    assert load_checkpoint_strict(rebuilt, blob) is not None, (
        "the checkpoint's state dict did not align into a model rebuilt from its own kwargs"
    )
    # The run seed reaches the blob too, which is what lets a resumed run reproduce its tile grid.
    assert blob["hyper_parameters"]["seed"] == general["seed"]
    # And so does the ramp, which is the conv-Transformer half of the schedule.
    assert blob["hyper_parameters"]["lr_warmup_steps"] == general["lr_warmup_steps"]


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


def test_a_different_seed_moves_the_run(tmp_path_factory, fit) -> None:
    """The negative control on the determinism test: an assertion that held for *every* seed would
    be describing a run that ignores its configuration rather than one that reproduces. The seed is
    load-bearing twice over here -- it seeds the framework, and it is one of the four halves of the
    tile-phase key."""
    reseeded_driver, _ = _run_fit(tmp_path_factory.mktemp("smoke_seed"), seed=7)
    baseline, reseeded = _metrics(fit[0]), _metrics(reseeded_driver)

    assert list(baseline.columns) == list(reseeded.columns)
    assert not baseline["train/total_loss"].equals(reseeded["train/total_loss"])
