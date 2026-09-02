r"""One real fit, through the real entry point, against the committed causal shard.

Everything else in this suite tests a piece in isolation. This runs the whole thing: config ->
pre-flight guards -> ``setup_config`` -> data module -> model -> ``build_trainer`` -> ``fit`` ->
checkpoint, on a CPU, in seconds. It is the only place the failures that live *between* the pieces
can surface -- a config key that reaches nothing, a metric name no callback collects, a callback that
raises on the first validation epoch, and, specific to this target domain, the two that arrive only
on a **validation** step: the permutation control decoding at the wrong anchor set, and the
source-null arm re-encoding a stream that is no longer in the forward dict.

It is also where the *diamond* is exercised end to end for the first time. The step-granular
learning-rate ramp reaches the run through one parent and the tiling phase through the other, and
neither is reachable by building either half alone.

Driven through ``main`` rather than by assembling the driver by hand, deliberately: the four shared
pre-flight guards, the causal parent's own five, the temporary resolved-config file and the
resolved-config write beside the checkpoints hang off the entry point and are reached no other way.

Two epochs rather than the config's one: ``lr`` is logged at train-epoch *start* with
``on_epoch=True``, so its first CSV cell is always NaN, and the second epoch is what exercises the
scheduler stepping at all -- and, here, what makes the tile phase rotate, since the epoch is one of
the four halves of its key.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME
from teb_vae.lag_attn_transformer_cfs import trainer as trainer_module
from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs
from teb_vae.lag_attn_transformer_cfs.trainer import LagAttnTrfCfsTrainer
from train.graph_models_utils import check_model_class, load_checkpoint_strict

from .conftest import (
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
#:
#: The two come from different rules and, since the two streams stopped sharing a clock, from
#: different references. The **budget** takes four ``fhr_st`` channels off the target and never
#: touches the source; the **alignment references** take every channel above their own stream's
#: reference and never touch the other stream. The source number is therefore neither ``c_u`` --
#: which it was while nothing gated that stream -- nor the target's survivor count: it is what the
#: shipped source clock, a hundred-odd seconds faster than the target's, leaves standing.
GUARDED_TARGET_CHANNELS = 98
GUARDED_SOURCE_CHANNELS = 39

#: The anchor counts the two stages must produce, derived here the way the model derives them so a
#: geometry change re-derives them rather than failing a literal.
#: The dense count is taken over the EFFECTIVE ceiling: the shipped physical forecast clock's
#: largest advance, resolved against the committed shard exactly as the run resolves it, removes
#: the trailing anchors before anything is decoded -- and the tiling divides by the shipped
#: stride of 5, which travels with that clock rather than with the horizon.
SHIPPED_ANCHOR_STRIDE = 5


def _physical_advance() -> int:
    """The shipped clock's largest advance, resolved from the committed shard's own delays."""
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget
    from teb_vae.lag_attn_cfs.tests.conftest import causal_config

    budget = resolve_warmup_budget(causal_config(causal_target_forecast_clock="physical"))
    assert budget is not None
    return budget.max_forecast_advance


DENSE_ANCHORS = 300 - SHIPPED_HORIZON - SHIPPED_WARMUP_PERIOD - _physical_advance()
TILE_COUNT = -(-DENSE_ANCHORS // SHIPPED_ANCHOR_STRIDE)


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
    original_train_model = LagAttnTrfCfsTrainer.train_model

    def _capture_train_model(self, train_loader, validation_loader):
        result = original_train_model(self, train_loader, validation_loader)
        captured["driver"] = self
        captured["trainer"] = result
        return result

    LagAttnTrfCfsTrainer.train_model = _capture_train_model
    try:
        trainer_module.main(str(config_path))
    finally:
        # Deleted rather than reassigned: the method is inherited, and leaving a copy on the
        # subclass would shadow a later change to the one it inherits.
        del LagAttnTrfCfsTrainer.train_model

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
      controls that;
    * ``num_workers: 0`` is already the tiny config's, so no loader worker seeding enters either.

    One thread, and that one is specific to this architecture rather than shared with the conv-LSTM
    cell. Torch's multi-threaded CPU reductions -- which this encoder reaches through its attention
    and its depthwise stem -- split a sum across whatever threads the scheduler gives them, so two
    processes on a loaded machine can accumulate the same values in different orders. The observed
    disagreement is float32 round-off (about $3 \times 10^{-6}$ relative on
    ``mean_logvar_full``), not a different computation, but a *bitwise* claim cannot be made over
    it. ``OMP_NUM_THREADS=1`` removes the only remaining source of variation and keeps the assertion
    exact, which is what determinism is supposed to mean.

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
            "from teb_vae.lag_attn_transformer_cfs.trainer import main; main(sys.argv[2])",
            str(repo_root),
            str(config_path),
        ],
        env={
            **dict(os.environ),
            "PYTHONHASHSEED": hash_seed,
            "CUDA_VISIBLE_DEVICES": "",
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
def test_the_fit_completes(fit):
    _driver, trainer = fit

    assert trainer.current_epoch == SMOKE_EPOCHS
    assert trainer.state.finished


def test_the_losses_stay_finite(fit):
    frame = _metrics(fit[0])

    for column in ("train/total_loss", "val/total_loss", "train/main_loss"):
        assert frame[column].notna().any(), column
        assert frame[column].dropna().apply(math.isfinite).all(), column


def test_the_spike_breaker_never_latched(fit):
    """A breaker that fired on an ordinary smoke fit would be mis-tuned for this objective, and the
    symptom on a production box is a run that trains on a fraction of its batches."""
    frame = _metrics(fit[0])

    if "train/spike_skipped" in frame.columns:
        assert float(frame["train/spike_skipped"].fillna(0.0).max()) == 0.0


def test_the_run_trains_at_the_budgets_width_and_the_configs_tiling(fit):
    """The two things a smoke run must not silently change: the decoder's width, which is the units
    of every reported nat, and the stride, which is what they were averaged over."""
    driver, _trainer = fit
    model = driver.pytorch_model

    assert isinstance(model, SeqVaeLagAttnTrfCfs)
    assert model.decoder_out_channels == GUARDED_TARGET_CHANNELS
    assert model.anchor_stride == SHIPPED_ANCHOR_STRIDE
    assert model.warmup_period == SHIPPED_WARMUP_PERIOD
    assert driver.resolved_warmup is not None


def test_the_step_granular_ramp_was_live_during_the_fit(fit):
    """The conv-Transformer half of the diamond, reached through a real fit. The tiny config states
    a short ``lr_warmup_steps`` so the ramp is observable inside two epochs rather than saturated
    from the first step; at $0$ the task would have delegated to the epoch path silently."""
    driver, _trainer = fit

    assert int(driver.pl_model.hparams["lr_warmup_steps"]) > 0
    frame = _metrics(driver)
    assert "lr" in frame.columns
    assert frame["lr"].dropna().gt(0.0).all()


def test_the_input_adapters_carry_the_availability_terms_the_warm_up_needs(fit):
    r"""The failure ``_build_adapter`` exists to prevent, checked on the model a real run built.

    The architecture parent sizes the guard from ``gate.delay.delay_steps``, which carries only the
    alignment shifts $d_c$ and nothing of the warm-up: $0$ on an unaligned config, where the gate is
    a pure gather, and $\max_c d_c$ under this cell's shipped
    ``causal_align_reference: target_max``. The override passes $W'_c + d_c$ per channel, and the
    zero-marginal-warm-up lemma makes $\max_c(W'_c + d_c) = \max_c W'_c$ -- which is why the
    equality asserted below survives the alignment unchanged."""
    driver, _trainer = fit
    model = driver.pytorch_model

    for adapter, expected in (
        (model.target_adapter, model.target_warmup_steps),
        (model.source_adapter, model.source_warmup_steps),
    ):
        assert adapter.mask_proj is not None
        assert adapter.max_delay == max(expected)


# --------------------------------------------------------------------------------------
# What the CSV says
# --------------------------------------------------------------------------------------
def test_every_declared_metric_reaches_the_logger(fit):
    """Both directions. A tracked name the framework never emits is a column that is NaN in every
    row of every run; a metric the task emits that nothing tracks never reaches the CSV at all."""
    frame = _metrics(fit[0])

    missing = [name for name in LagAttnTrfCfsTrainer.TRACKED_METRICS if name not in frame.columns]
    assert missing == [], missing


def test_no_tracked_column_is_all_nan(fit):
    frame = _metrics(fit[0])

    empty = [
        name
        for name in LagAttnTrfCfsTrainer.TRACKED_METRICS
        if name in frame.columns and frame[name].isna().all()
    ]
    assert empty == [], empty


def test_the_warm_fraction_is_exactly_one_on_every_row_of_both_stages(fit):
    """A geometry guard rather than a result: it is resolved at construction under the pairing
    refusal, so any other value means the checkpoint was built by code predating that refusal."""
    frame = _metrics(fit[0])

    for column in ("train/target_warm_frac", "val/target_warm_frac"):
        values = frame[column].dropna()
        assert not values.empty, column
        assert (values == 1.0).all(), f"{column}: {sorted(set(values))}"


def test_the_anchor_count_sits_at_its_geometry_derived_value_on_both_stages(fit):
    """The other geometry guard. Training tiles, so the count sits in the band the phase can
    produce; validation decodes densely, so it is exactly $T_{\\mathrm{valid}} - F$."""
    frame = _metrics(fit[0])

    # The last phase's tile count: the stride divides the span, not the horizon -- the two were
    # interchangeable only while the tiling partitioned at S = H.
    fewest = -(-(DENSE_ANCHORS - (SHIPPED_ANCHOR_STRIDE - 1)) // SHIPPED_ANCHOR_STRIDE)
    train = frame["train/anchors_per_sample"].dropna()
    assert not train.empty
    assert train.between(fewest, TILE_COUNT).all(), sorted(set(train))

    validation = frame["val/anchors_per_sample"].dropna()
    assert not validation.empty
    assert (validation == float(DENSE_ANCHORS)).all(), sorted(set(validation))


def test_both_channel_splits_recompose_to_the_gap_in_the_same_row(fit):
    """The tertile split and the block split cut the same channel axis two ways, so each must sum to
    the same total -- and they are compared against *each other* rather than against ``pred_gap``,
    which is a difference of two block NLLs and loses several decimal digits to cancellation before
    any split is formed."""
    frame = _metrics(fit[0]).dropna(
        subset=["val/pred_gap_warm_lo", "val/pred_gap_st"]
    )
    assert not frame.empty

    tertiles = (
        frame["val/pred_gap_warm_lo"]
        + frame["val/pred_gap_warm_mid"]
        + frame["val/pred_gap_warm_hi"]
    )
    blocks = frame["val/pred_gap_st"] + frame["val/pred_gap_ph"]

    assert ((tertiles - blocks).abs() < 1e-4 * (blocks.abs() + 1.0)).all()


def test_the_source_null_floor_is_reported_beside_the_coupling_readout(fit):
    """Validation-only, and absent rather than zero-filled on training rows: the framework's epoch
    value is the mean over the steps that reported a metric, so a zero placeholder would scale the
    aggregate toward nothing."""
    frame = _metrics(fit[0])

    assert "val/kld_source_null" in frame.columns
    assert frame["val/kld_source_null"].notna().any()
    assert "train/kld_source_null" not in frame.columns


def test_the_lag_warmth_columns_are_fractions(fit):
    """Normalised by the attention mass actually present, so the value stays in $[0, 1]$ even when
    rows have no admissible lag at all."""
    frame = _metrics(fit[0])

    for column in ("val/source_lag_warmth_frac_st", "val/source_lag_warmth_frac_ph"):
        values = frame[column].dropna()
        assert not values.empty, column
        assert values.between(0.0, 1.0).all(), column


# --------------------------------------------------------------------------------------
# What the run directory holds
# --------------------------------------------------------------------------------------
def test_the_run_directory_has_the_expected_layout(fit):
    driver, _trainer = fit

    assert (Path(driver.train_results_dir) / "metrics_history.csv").exists()
    assert Path(driver.model_checkpoint_dir).is_dir()


def test_the_resolved_config_records_the_budget_the_run_actually_got(fit):
    """The run's own provenance record: the threshold it was launched with, beside the geometry it
    resolved to."""
    driver, _trainer = fit
    resolved = yaml.safe_load(
        (Path(driver.model_checkpoint_dir) / RESOLVED_CONFIG_FILENAME).read_text(encoding="utf-8")
    )

    vae = resolved["model_config"]["VAE_model"]
    assert vae["causal_warmup_budget_steps"] == 134
    assert vae["anchor_stride"] == SHIPPED_ANCHOR_STRIDE
    assert vae["causal_reach_budget_s"] is None


def test_the_checkpoint_is_written_under_this_models_stem(fit):
    """Two models writing under one stem into a shared output tree are indistinguishable by name."""
    driver, _trainer = fit

    written = sorted(Path(driver.model_checkpoint_dir).glob("*.ckpt"))
    assert written, "no checkpoint was written"
    for path in written:
        assert path.name.startswith(LagAttnTrfCfsTrainer.CHECKPOINT_STEM), path.name


def test_both_checkpoint_criteria_wrote_a_file_under_distinct_stems(fit):
    """The second criterion, end to end, which is the only place its filename is decided.

    The composite optimum and the best conditioned forecast are different epochs, so a run keeps
    both -- and the two callbacks must not write the same name. With one stem Lightning would have
    each overwrite the other's file at the same epoch, leaving one criterion's best silently
    unsaved: two ``ModelCheckpoint``s in the callback list, one set of files on disk, and nothing
    in the log about it. That is a construction-time property nothing but a real fit exercises.
    """
    driver, _trainer = fit
    configured = driver.config["advanced_config"]["callbacks"]["model_checkpoint"]
    stem = LagAttnTrfCfsTrainer.CHECKPOINT_STEM
    names = {path.name for path in Path(driver.model_checkpoint_dir).glob("*.ckpt")}

    assert configured.get("secondary_monitor"), "this run kept only one criterion"
    primary = {name for name in names if name.startswith(f"{stem}-epoch=")}
    secondary = names - primary

    assert primary, f"the primary criterion wrote nothing: {sorted(names)}"
    assert secondary, f"the second criterion wrote nothing: {sorted(names)}"
    assert all("nll" in name for name in secondary), sorted(secondary)


def test_the_checkpoint_carries_its_contract_and_reloads_at_its_own_width(fit):
    """The end of the chain: a blob written by a real fit rebuilds this architecture at this
    budget, with no shard consulted."""
    driver, _trainer = fit
    path = sorted(Path(driver.model_checkpoint_dir).glob("*.ckpt"))[0]

    blob = torch.load(path, map_location="cpu", weights_only=False)
    check_model_class(blob, "SeqVaeLagAttnTrfCfs")
    rebuilt = SeqVaeLagAttnTrfCfs(**blob["model_kwargs"])

    assert load_checkpoint_strict(rebuilt, blob) is not None
    assert rebuilt.decoder_out_channels == GUARDED_TARGET_CHANNELS
    assert len(rebuilt.source_warmup_steps) == GUARDED_SOURCE_CHANNELS
    assert rebuilt.anchor_stride == SHIPPED_ANCHOR_STRIDE


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------
def test_two_runs_of_one_config_produce_an_identical_metric_row_set(tmp_path_factory):
    """Determinism as the requirement states it: an identical **metric row set**, not merely
    identical anchor indices.

    Identical anchors are necessary and nowhere near sufficient -- the reparameterisation $\\epsilon$
    and the permutation generator both move too -- and this target domain adds one more thing that
    could break it and would not raise: the tile phase. Derived from a stable hash of the segment,
    the epoch and the seed, it survives a re-run; drawn from the global RNG it would not, and the
    only symptom would be two runs of one config disagreeing.

    Driven as two separate CPU processes under **different** ``PYTHONHASHSEED`` values. The differing
    salt is not incidental: a phase derived from Python's own ``hash()`` would move with it, and
    nothing would raise -- $A_{\\max}$ is a geometry constant either way.
    """
    first = _run_fit_in_subprocess(tmp_path_factory.mktemp("det_a"), hash_seed="0")
    second = _run_fit_in_subprocess(tmp_path_factory.mktemp("det_b"), hash_seed="12345")

    assert list(first.columns) == list(second.columns)
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_a_different_seed_moves_the_run(tmp_path_factory, fit):
    """The negative control on the determinism test: an assertion that held for *every* seed would
    be describing a run that ignores its configuration rather than one that reproduces. The seed is
    load-bearing twice over here -- it seeds the framework, and it is one of the four halves of the
    tile-phase key."""
    reseeded_driver, _trainer = _run_fit(tmp_path_factory.mktemp("smoke_seed"), seed=7)
    baseline, reseeded = _metrics(fit[0]), _metrics(reseeded_driver)

    assert list(baseline.columns) == list(reseeded.columns)
    assert not baseline["train/total_loss"].equals(reseeded["train/total_loss"])
