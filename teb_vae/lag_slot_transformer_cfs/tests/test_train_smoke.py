r"""One real fit, through the real entry point, against the committed fixture shard.

Everything else in this suite tests a piece in isolation. This runs the whole thing: configuration,
pre-flight guards, data module, warm-up budget resolution, model, trainer, fit, checkpoint. It is
the only place the failures that live *between* the pieces can surface -- a configuration key that
reaches nothing, a metric name no callback collects, a callback that raises on the first validation
epoch, and, specific to this architecture, the two that arrive only on a validation step: the
inherited permutation control reaching modules that were never built, and the inherited source-null
readout encoding through a pathway that holds no parameters. Both are refused by the task, and this
is where that refusal is exercised rather than asserted.

It is also where the resolution-order diamond runs end to end for the first time. The step-granular
learning-rate ramp reaches the run through one parent and the tiling phase through the other, and
neither is reachable by building either half alone.

Two epochs rather than the configuration's one: the learning rate is logged at train-epoch start, so
its first row is always empty, and the second epoch is what steps the scheduler at all -- and what
makes the tile phase rotate, since the epoch is one of the four parts of its key.

**This is also where the gradient distribution is measured.** The clip and the spike margin are
inherited from a sibling whose gradient distribution was measured on a different source pathway.
They are starting guards here, not tuned thresholds, and the run records what this model's norms
actually look like so the next operator has a number rather than an inheritance.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_cfs.tests.conftest import absolutize_dataset_paths
from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME
from teb_vae.lag_slot_transformer_cfs import trainer as trainer_module
from teb_vae.lag_slot_transformer_cfs.nets.model import SeqVaeLagResidualTrfCfs
from teb_vae.lag_slot_transformer_cfs.task import TASK_METRIC_SUFFIXES
from teb_vae.lag_slot_transformer_cfs.trainer import (
    LagResidualTrfCfsTrainer,
    strip_task_prefix,
)

pytestmark = pytest.mark.slow

#: The configuration the fit runs.
TINY_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"

#: Where the fit writes its checkpoints and its resolved configuration, relative to the run root.
CHECKPOINT_DIRNAME = "model_checkpoints"

#: Epochs the fit runs. See the module docstring for why it is not the configuration's one.
SMOKE_EPOCHS = 2

#: What the shipped warm-up budget resolves to, pinned so a "guarded" fit cannot silently be the
#: unguarded one -- which here would also change the decoder's width and therefore the units of
#: every number the run reports. The two counts come from different rules: the budget takes four
#: channels off the target and never touches the source.
GUARDED_TARGET_CHANNELS = 76
GUARDED_SOURCE_CHANNELS = 46


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """Run one fit through the entry point and hand back the driver and its metric history.

    Module-scoped because the fit is the expensive part of this file and every assertion below
    reads the same run. A test that needed a *different* run would build its own.

    Args:
        tmp_path_factory: pytest's directory factory.

    Returns:
        ``(driver, metrics)``.
    """
    tmp_path = tmp_path_factory.mktemp("smoke")
    config = absolutize_dataset_paths(load_config(str(TINY_CONFIG)))
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path)
    config["general_config"]["epochs"] = SMOKE_EPOCHS
    # Off: this asserts the training path, not the profiler's output.
    config["advanced_config"]["trainer"]["profiler"] = None

    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    captured = {}
    original = LagResidualTrfCfsTrainer.train_model

    def capture(self, train_loader, validation_loader):
        """Run the inherited fit and keep the driver, which is the only handle on the run."""
        result = original(self, train_loader, validation_loader)
        captured["driver"] = self
        return result

    LagResidualTrfCfsTrainer.train_model = capture
    try:
        trainer_module.main(str(config_path))
    finally:
        # Deleted rather than reassigned: the method is inherited, and leaving a copy on the
        # subclass would shadow a later change to the one it inherits.
        del LagResidualTrfCfsTrainer.train_model

    driver = captured["driver"]
    history = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")
    return driver, history


def test_the_fit_completes_and_builds_this_architecture(fitted) -> None:
    """The whole path, from a configuration file to a fitted model."""
    driver, _ = fitted
    assert isinstance(driver.pytorch_model, SeqVaeLagResidualTrfCfs)
    # No attention module reached the run, which every other check here takes for granted.
    assert not hasattr(driver.pytorch_model, "lag_attn")
    assert not hasattr(driver.pytorch_model, "posterior_head")


def test_the_warm_up_budget_resolved_and_gated_both_streams(fitted) -> None:
    """A "guarded" fit that silently ran ungated would change the decoder's width with it."""
    driver, _ = fitted
    model = driver.pytorch_model
    assert model.target_gate is not None and model.source_gate is not None
    assert model.target_gate.out_channels == GUARDED_TARGET_CHANNELS
    assert model.source_gate.out_channels == GUARDED_SOURCE_CHANNELS
    assert model.decoder_out_channels == GUARDED_TARGET_CHANNELS


def test_the_source_encoder_holds_no_parameters_in_a_real_run(fitted) -> None:
    """The recommended arm's claim, verified on the model a fit actually built."""
    driver, _ = fitted
    assert not driver.pytorch_model.source_encoder.has_parameters()


def test_the_objective_columns_are_present_and_finite(fitted) -> None:
    """Including the two that make a nats-per-anchor column readable."""
    _, history = fitted
    for column in (
        "train/total_loss",
        "val/total_loss",
        "val/nll_full_block",
        "val/nll_base_block",
        "val/pred_gap",
        "val/source_conditioned_kl_raw",
        "val/prior_rate",
        "val/scored_anchors",
        "val/scored_coefficients",
    ):
        assert column in history.columns, column
        values = history[column].dropna()
        assert len(values) > 0, column
        assert values.notna().all()

    assert float(history["val/scored_anchors"].dropna().iloc[-1]) > 0.0
    assert float(history["val/scored_coefficients"].dropna().iloc[-1]) == (
        GUARDED_TARGET_CHANNELS * int(
            load_config(str(TINY_CONFIG))["model_config"]["VAE_model"]["horizon"]
        )
    )


def test_this_architecture_s_own_readouts_reach_the_history(fitted) -> None:
    """The four numbers only this model has, on the stage that reports them."""
    _, history = fitted
    for suffix in TASK_METRIC_SUFFIXES:
        column = f"val/{suffix}"
        assert column in history.columns, column
        assert history[column].dropna().notna().all(), column


def test_no_column_names_a_tensor_this_architecture_lacks(fitted) -> None:
    """The inherited surface names three groups this objective does not produce.

    A tracked name nothing produces is a column empty in every row, which reads as a measurement
    that came out blank rather than as an absent one.
    """
    _, history = fitted
    for absent in (
        "delta_mu_sat_frac",
        "kld_source_null",
        "nll_shuffled_block",
        "shuffle_penalty",
        "aux_multiscale",
    ):
        assert not any(column.endswith(f"/{absent}") for column in history.columns), absent


def test_the_source_pathway_starts_near_zero_and_leaves_it(fitted) -> None:
    """The exact zero start is proven on a constructed model; what a run adds is that it moves.

    The reported divergence is an epoch aggregate over steps, and by the end of the first epoch the
    model has already taken optimizer steps -- so a real run cannot show the exact zero, only that
    it began negligibly close to it and grew. A pathway that stayed inert would train as a
    target-only forecaster and report a zero gap as a finding, which is the failure worth catching
    here.
    """
    _, history = fitted
    divergence = history["val/source_conditioned_kl_raw"].dropna()
    assert len(divergence) >= 2
    first, last = float(divergence.iloc[0]), float(divergence.iloc[-1])
    assert first < 1e-2, first
    assert last > first, (first, last)


def test_the_gradient_distribution_is_recorded_for_the_next_operator(fitted) -> None:
    """The clip is inherited from a sibling with a different source pathway.

    This does not assert a threshold: it asserts that the run produced the numbers from which one
    can be chosen, and that the inherited clip is not silently rescaling every step -- which would
    leave the run optimising little but weight decay.
    """
    _, history = fitted
    norms = history["train/grad_norm"].dropna()
    assert len(norms) > 0
    assert float(norms.max()) > 0.0

    # The fraction of optimizer steps whose pre-clip norm exceeded the threshold. The epoch value
    # of the norm is an aggregate, so "how often did the clip bind" is recoverable from no other
    # recorded quantity -- and a run where it binds every step is optimising little but weight
    # decay while every other column looks healthy.
    fraction = history["train/grad_clip_frac"].dropna()
    assert len(fraction) > 0
    assert float(fraction.max()) < 1.0, "the inherited clip rescales every step"

    # The spike breaker's own surface, without which a run that trained normally and then skipped
    # every batch forever shows nothing at all.
    assert "train/spike_skipped" in history.columns


def test_the_run_writes_a_resolved_configuration_beside_its_checkpoints(fitted) -> None:
    """It is the durable record: a run's settings must be recoverable from its output alone."""
    driver, _ = fitted
    # Beside the checkpoints rather than beside the metric history: the evaluation entry point
    # derives this path from the checkpoint it is given, so the two travel together.
    resolved = (
        Path(driver.train_results_dir).parent / CHECKPOINT_DIRNAME / RESOLVED_CONFIG_FILENAME
    )
    assert resolved.exists(), resolved
    written = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    block = written["model_config"]["VAE_model"]
    # The chunk sizes especially: they change floating-point summation order, so a run that set
    # them and did not record them could not be reproduced.
    assert block["anchor_chunk"] is not None
    assert block["lag_chunk"] is not None


def test_a_written_checkpoint_rebuilds_the_model_and_its_invariants(fitted) -> None:
    """Architecture, weights and inference, from the checkpoint alone.

    Compared against a **second rebuild of the same checkpoint** rather than against the model
    left in memory: the checkpoint callback writes at validation end and the fit continues past
    it, so the in-memory weights are legitimately a later epoch's. What a round trip has to
    establish is that the file is sufficient, which is what two rebuilds agreeing shows.
    """
    driver, _ = fitted
    checkpoint_dir = Path(driver.train_results_dir).parent / CHECKPOINT_DIRNAME
    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
    assert checkpoints, f"the fit wrote no checkpoint under {checkpoint_dir}"
    # Both criteria kept a file, and the stem tells them apart. Without the stem the two would
    # interleave, and without this package's own stem they would interleave with a sibling's.
    stems = {path.name.split("-epoch=")[0] for path in checkpoints}
    assert stems == {
        "lag-residual-trf-cfs",
        "lag-residual-trf-cfs-val-nll_full_block",
    }, stems

    blob = torch.load(str(checkpoints[-1]), map_location="cpu", weights_only=False)
    assert "model_kwargs" in blob, "the checkpoint carries no constructor kwargs"
    state = strip_task_prefix(blob["state_dict"])

    first = SeqVaeLagResidualTrfCfs(**blob["model_kwargs"]).eval()
    first.load_state_dict(state)
    second = SeqVaeLagResidualTrfCfs(**blob["model_kwargs"]).eval()
    second.load_state_dict(state)

    # Every tensor the file carries arrived, so the rebuild is the checkpoint and not a default.
    for name, tensor in first.state_dict().items():
        assert torch.equal(tensor, state[name]), name

    inputs = (
        torch.randn(2, first.sequence_length, 36),
        torch.randn(2, first.sequence_length, 44),
        torch.randn(2, first.sequence_length, first.c_u),
    )
    torch.manual_seed(0)
    left = first(*inputs, anchor_phase=0, anchor_stride=1)
    torch.manual_seed(0)
    right = second(*inputs, anchor_phase=0, anchor_stride=1)
    for name in ("mu_prior", "mu_post", "mu_full", "kld_per_anchor"):
        assert torch.equal(left[name], right[name]), name

    # And the source-off invariant survives the round trip, which is what every control reads.
    n_anchors = left["mu_prior"].shape[1]
    torch.manual_seed(0)
    silenced = first(
        *inputs,
        anchor_phase=0,
        anchor_stride=1,
        selector=torch.zeros(2, n_anchors, first.n_lags),
    )
    assert torch.equal(silenced["mu_post"], silenced["mu_prior"])
