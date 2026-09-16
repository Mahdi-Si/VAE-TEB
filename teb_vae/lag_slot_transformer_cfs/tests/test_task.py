r"""The training task: its resolution order, its metric surface, and the two controls it refuses.

The step is written out rather than inherited, so what has to be checked is that it still *is* the
shared step in every respect that matters -- the hyperparameters read by the same names, the loss
reached through the same seam, the spike-breaker key intact -- while the three tensors this
architecture does not have are gone rather than substituted.

The two refusals are the interesting part. The inherited permutation control and the inherited
source-null readout both reach modules that were never built, so leaving either in place would
raise on the first validation step of the first real run. They are refused explicitly, with the
reason recorded, rather than left to fail.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Optional

import pytest
import torch

from teb_vae.lag_attn_cfs.task import SeqVaeLagAttnCfsTask
from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask
from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask
from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask
from teb_vae.lag_slot_transformer_cfs.task import (
    TASK_METRIC_SUFFIXES,
    VALIDATION_MC_DRAWS_KEY,
    VALIDATION_MONITOR_SUFFIXES,
    SeqVaeLagResidualTrfCfsTask,
    recording_grouped_totals,
)
from teb_vae.lag_slot_transformer_cfs.tests.conftest import (
    DECLARED_C_U,
    DECLARED_ST,
    DECLARED_TARGET_PH,
    TINY_BATCH,
    TINY_SEQ_LEN,
    build_tiny_model,
    tiny_streams,
)


class StubBatch:
    """The fields the task reads off a batch, with nothing else.

    A stub rather than a shard read, because what this file checks is the step's arithmetic and its
    metric surface: the loader has its own tests, and going through it would make every assertion
    here depend on a fixture file.
    """

    def __init__(self, *, gap: bool = False) -> None:
        """Build one deterministic batch.

        Args:
            gap: Zero a stretch of the validity signal, so some anchors are unscored.
        """
        y_st, y_ph, u_stream = tiny_streams()
        self.fhr_st, self.fhr_ph, self.up_st = y_st, y_ph, u_stream[:, :, :DECLARED_ST]
        self.up_ph = u_stream[:, :, DECLARED_ST:]
        self.weight = torch.ones(TINY_BATCH, TINY_SEQ_LEN)
        if gap:
            self.weight[:, 12:18] = 0.0
        # The two fields the tiling phase is keyed on. Absent, the task refuses by name.
        self.guid = [f"guid-{index}" for index in range(TINY_BATCH)]
        self.epoch = torch.zeros(TINY_BATCH)


def build_task(
    *, validation_mc_draws: Optional[int] = None, seed: int = 0, **overrides
) -> SeqVaeLagResidualTrfCfsTask:
    """Wrap a deterministically built tiny model in the task.

    Args:
        validation_mc_draws: The predictive monitor's draw count, or ``None`` for the legacy
            surface.
        seed: The run seed the tiling phase and the noise bank are keyed on.
        **overrides: Constructor keywords for the model.

    Returns:
        The task, with its source pathway moved off zero so the two branches differ.
    """
    model = build_tiny_model(**overrides)
    if model.proposal_head is not None:
        generator = torch.Generator().manual_seed(31)
        with torch.no_grad():
            model.proposal_head.output_proj.weight.normal_(0.0, 0.3, generator=generator)
    return SeqVaeLagResidualTrfCfsTask(
        model,
        lr=1e-3,
        likelihood="gaussian_nll",
        kld_beta=1.0,
        beta_prior=0.1,
        beta_schedule=None,
        seed=seed,
        validation_mc_draws=validation_mc_draws,
    )


# =================================================================================================
# Composition
# =================================================================================================
def test_the_resolution_order_is_the_one_the_design_names() -> None:
    """Written out, so a reorder fails here rather than training under another step."""
    names = [cls.__name__ for cls in SeqVaeLagResidualTrfCfsTask.__mro__]
    assert names[:5] == [
        "SeqVaeLagResidualTrfCfsTask",
        "SeqVaeLagAttnCfsTask",
        "SeqVaeLagAttnFsTask",
        "SeqVaeLagAttnTrfRwsTask",
        "SeqVaeLagAttnRwsTask",
    ]


def test_the_step_granular_learning_rate_ramp_comes_from_the_transformer_parent() -> None:
    """It is the whole reason that parent is in the bases at all.

    A pre-normalised attention stack needs the ramp in its first few hundred optimizer steps, which
    an epoch-granularity schedule cannot address.
    """
    for cls in SeqVaeLagResidualTrfCfsTask.__mro__:
        if "build_lr_scheduler" in vars(cls):
            assert cls is SeqVaeLagAttnTrfRwsTask
            break
    else:  # pragma: no cover - a missing scheduler is a construction failure long before here
        pytest.fail("build_lr_scheduler is defined nowhere in the resolution order")


def test_the_tiling_seams_come_from_the_causal_parent() -> None:
    """The phase derivation and the five-argument forward assembly are that package's."""
    for name in ("anchor_phase", "resolve_anchor_geometry", "_build_forward_inputs"):
        for cls in SeqVaeLagResidualTrfCfsTask.__mro__:
            if name in vars(cls):
                assert cls is SeqVaeLagAttnCfsTask, name
                break


def test_the_step_is_this_package_s_own() -> None:
    """Along with the latent-gap readout and the two refusals.

    The shared step reads a saturation key this architecture does not emit and pairs a dense
    support with an anchor-indexed latent, so it cannot be inherited.
    """
    for name in ("compute_loss_and_metrics", "_mu_gap_rms", "_added_metrics", "_should_run_perm"):
        for cls in SeqVaeLagResidualTrfCfsTask.__mro__:
            if name in vars(cls):
                assert cls is SeqVaeLagResidualTrfCfsTask, name
                break


# =================================================================================================
# The step
# =================================================================================================
@pytest.mark.parametrize("stage", ["train", "val", "test"])
def test_a_step_runs_and_reports_a_finite_loss(stage: str) -> None:
    """On every stage, without a trainer attached.

    Args:
        stage: The stage to run.
    """
    task = build_task()
    torch.manual_seed(0)
    loss, metrics = task.compute_loss_and_metrics(StubBatch(), 0, stage)

    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert float(metrics["scored_anchors"]) > 0.0
    assert torch.equal(metrics["main_loss"], loss.detach())


def test_the_training_stage_tiles_and_the_evaluation_stages_decode_densely() -> None:
    """The stride is a stage decision, and the dense set is what a reported number is over."""
    task = build_task()
    torch.manual_seed(0)
    _, tiled = task.compute_loss_and_metrics(StubBatch(), 0, "train")
    torch.manual_seed(0)
    _, dense = task.compute_loss_and_metrics(StubBatch(), 0, "val")
    assert float(dense["scored_anchors"]) > float(tiled["scored_anchors"])


def test_the_stage_does_not_leak_past_one_step() -> None:
    """It travels on the instance, so a caller reaching the input builder outside a step must
    get the dense, epoch-independent geometry rather than a training tile grid."""
    task = build_task()
    before = task._stage
    torch.manual_seed(0)
    task.compute_loss_and_metrics(StubBatch(), 0, "train")
    assert task._stage == before


def test_the_metric_surface_carries_this_architecture_s_readouts(
) -> None:
    """The four numbers only this model has, plus the objective's own columns."""
    task = build_task()
    torch.manual_seed(0)
    _, metrics = task.compute_loss_and_metrics(StubBatch(), 0, "val")

    for name in TASK_METRIC_SUFFIXES:
        assert name in metrics, name
    for name in ("nll_full_block", "pred_gap", "source_conditioned_kl_raw", "prior_rate"):
        assert name in metrics, name


def test_the_metric_surface_names_no_tensor_this_architecture_lacks() -> None:
    """The shared step reports a posterior-bound saturation this model does not have.

    Reported under that name it would be read against the sibling's column and mean something
    else: the bound here is in prior standard deviations, and there are two of them.
    """
    task = build_task()
    torch.manual_seed(0)
    _, metrics = task.compute_loss_and_metrics(StubBatch(), 0, "val")
    for absent in ("delta_mu_sat_frac", "kld_source_null", "nll_shuffled_block", "shuffle_penalty"):
        assert absent not in metrics


def test_the_mean_only_arm_reports_no_scale_channel() -> None:
    """Absent rather than zero-filled, matching the forward."""
    task = build_task(mean_only_residual=True)
    torch.manual_seed(0)
    _, metrics = task.compute_loss_and_metrics(StubBatch(), 0, "val")
    assert "cancellation_ratio_scale" not in metrics
    assert "cancellation_ratio_mean" in metrics


def test_the_latent_gap_is_averaged_over_the_scored_anchors() -> None:
    """The same support the divergence beside it uses, so the two are read against each other."""
    task = build_task()
    torch.manual_seed(0)
    _, full = task.compute_loss_and_metrics(StubBatch(), 0, "val")
    torch.manual_seed(0)
    _, gapped = task.compute_loss_and_metrics(StubBatch(gap=True), 0, "val")

    assert float(gapped["scored_anchors"]) < float(full["scored_anchors"])
    assert torch.isfinite(gapped["mu_post_prior_gap_rms"])
    assert float(gapped["mu_post_prior_gap_rms"]) > 0.0


def test_a_fully_masked_batch_reports_zero_rather_than_raising() -> None:
    """A step that scored nothing must still return, and must say it scored nothing."""
    task = build_task()
    batch = StubBatch()
    batch.weight = torch.zeros_like(batch.weight)
    torch.manual_seed(0)
    loss, metrics = task.compute_loss_and_metrics(batch, 0, "val")

    assert float(loss) == 0.0
    assert float(metrics["scored_anchors"]) == 0.0
    assert float(metrics["mu_post_prior_gap_rms"]) == 0.0


# =================================================================================================
# The two refusals
# =================================================================================================
def test_the_permutation_control_never_runs() -> None:
    """It rebuilds the full branch through modules this architecture does not have.

    Refused explicitly rather than left to raise, because the failure would arrive on the first
    validation step of the first real run rather than here.
    """
    task = build_task()
    assert task._should_run_perm(TINY_BATCH, "val") is False
    assert task._should_run_perm(TINY_BATCH, "test") is False


def test_the_source_null_readout_is_empty_rather_than_wrong() -> None:
    """The inherited one encodes a zeroed stream through a pathway that holds no parameters."""
    task = build_task()
    torch.manual_seed(0)
    outputs = task.model(*task._build_forward_inputs(StubBatch()))
    added = task._added_metrics(
        task._build_forward_inputs(StubBatch()),
        outputs,
        torch.ones(TINY_BATCH, TINY_SEQ_LEN),
        "val",
    )
    assert added == {}


def test_a_readout_colliding_with_an_objective_metric_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A collision would replace that column with no error, in the history and in the backend."""
    task = build_task()
    monkeypatch.setattr(
        type(task),
        "_added_metrics",
        lambda self, *args, **kwargs: {"pred_gap": torch.zeros(())},
    )
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="pred_gap"):
        task.compute_loss_and_metrics(StubBatch(), 0, "val")


def test_a_batch_missing_a_phase_key_field_is_refused_by_name() -> None:
    """Without it every segment sits on one tile grid forever and no count differs."""
    task = build_task()
    batch = StubBatch()
    batch.guid = None
    with pytest.raises(RuntimeError, match="guid"):
        task.compute_loss_and_metrics(batch, 0, "train")


# =================================================================================================
# The predictive validation monitor
# =================================================================================================
def dense_forward(task: SeqVaeLagResidualTrfCfsTask, batch: StubBatch):
    """One dense forward through the task's own input builders, with the labels beside it.

    Args:
        task: The task.
        batch: The batch.

    Returns:
        ``(outputs, target_features, weight)``.
    """
    inputs = task._build_forward_inputs(batch)
    target_features, weight = task._build_raw_target(batch)
    torch.manual_seed(0)
    return task.model(*inputs), target_features, weight


def test_the_monitor_is_absent_without_the_key_and_on_the_training_stage() -> None:
    """A legacy run logs exactly the columns it always did; a tiled batch is never monitored."""
    legacy = build_task()
    torch.manual_seed(0)
    _, metrics = legacy.compute_loss_and_metrics(StubBatch(), 0, "val")
    assert legacy.hparams.get(VALIDATION_MC_DRAWS_KEY) is None
    assert not any(name in metrics for name in VALIDATION_MONITOR_SUFFIXES)

    monitored = build_task(validation_mc_draws=2)
    torch.manual_seed(0)
    _, tiled = monitored.compute_loss_and_metrics(StubBatch(), 0, "train")
    assert not any(name in tiled for name in VALIDATION_MONITOR_SUFFIXES)
    torch.manual_seed(0)
    _, dense = monitored.compute_loss_and_metrics(StubBatch(), 0, "val")
    for name in VALIDATION_MONITOR_SUFFIXES:
        assert name in dense and torch.isfinite(dense[name]), name
    assert torch.equal(dense["pred_gap_mc"], dense["pred_nll_base_mc"] - dense["pred_nll_full_mc"])


def test_with_one_draw_the_monitor_is_the_unweighted_conditional_score_of_its_draw() -> None:
    """The mixture over one draw is that draw's block score, unweighted, averaged over anchors.

    Recomputed by hand from the same noise bank: the decoder at the single latent draw, the
    objective's own elementwise score on the objective's own mask with no channel or horizon
    weight, and the anchor mean -- which on this batch equals the recording-grouped mean because
    every sample is its own recording with the same scored-anchor count.
    """
    task = build_task(validation_mc_draws=1)
    batch = StubBatch()
    outputs, target_features, weight = dense_forward(task, batch)
    monitor = task._predictive_monitor(batch, outputs, target_features, weight, 1)

    model = task.orig_model
    anchors = outputs["anchor_index"]
    target = model._build_forecast_target(target_features, anchors)
    mask, _ = forecast_mask(
        model.scored_weight(weight),
        model.geometry,
        coverage_floor=model.coverage_floor,
        anchors=anchors,
        anchor_valid=outputs["anchor_valid"],
    )
    generator = task.validation_noise_generator(batch, anchors.device)
    epsilon = torch.empty_like(outputs["mu_post"]).normal_(generator=generator)
    expected: Dict[str, torch.Tensor] = {}
    for column, mu_key, logvar_key in (
        ("pred_nll_full_mc", "mu_post", "logvar_post"),
        ("pred_nll_base_mc", "mu_prior", "logvar_prior"),
    ):
        latent = outputs[mu_key] + epsilon * torch.exp(0.5 * outputs[logvar_key])
        forecast_mu, forecast_logvar = model.decoder(latent, persistence=outputs.get("persistence"))
        block, contributing = masked_raw_block_per_anchor(
            forecast_mu, target, mask, likelihood="gaussian_nll", logvar=forecast_logvar
        )
        expected[column] = (block * contributing).sum() / contributing.sum()
    # The same scored-anchor count in every sample, so the anchor mean and the recording mean
    # coincide and the hand computation above is the monitor's estimand exactly.
    assert bool((mask.sum(dim=(1, 2)) == mask[0].sum()).all())
    for column, value in expected.items():
        assert torch.allclose(monitor[column].double(), value.double(), rtol=1e-5, atol=1e-6), column
    # The gap is the difference of the two logged columns in their own dtype, as the objective
    # forms its own; a difference of two single-precision block scores carries their rounding.
    assert torch.allclose(
        monitor["pred_gap_mc"].double(),
        (expected["pred_nll_base_mc"] - expected["pred_nll_full_mc"]).double(),
        rtol=1e-3,
        atol=1e-3,
    )
    # And it is the paired difference of the two branches: on this batch the full branch has
    # moved off the prior, so the gap is not the exact zero of a matched pair.
    assert float(monitor["pred_gap_mc"]) != 0.0


def test_the_monitor_reads_a_fixed_noise_bank() -> None:
    """Two validation passes at the same weights report the same value, bitwise.

    The objective's own columns move with the process RNG between the two passes, which is what
    shows the monitor's determinism is the bank's rather than the process's.
    """
    task = build_task(validation_mc_draws=4)
    torch.manual_seed(1)
    _, first = task.compute_loss_and_metrics(StubBatch(), 0, "val")
    torch.manual_seed(2)
    _, second = task.compute_loss_and_metrics(StubBatch(), 0, "val")
    for name in VALIDATION_MONITOR_SUFFIXES:
        assert torch.equal(first[name], second[name]), name
    assert not torch.equal(first["nll_full_block"], second["nll_full_block"])


def test_the_bank_is_keyed_on_the_segments_and_the_run_seed() -> None:
    """Another recording in the batch, or another run seed, is another bank."""
    task = build_task(validation_mc_draws=4, seed=3)
    batch = StubBatch()
    reference = task.validation_noise_generator(batch, torch.device("cpu")).initial_seed()

    renamed = StubBatch()
    renamed.guid = list(renamed.guid)
    renamed.guid[0] = "another-recording"
    assert task.validation_noise_generator(renamed, torch.device("cpu")).initial_seed() != reference

    reseeded = build_task(validation_mc_draws=4, seed=4)
    assert reseeded.validation_noise_generator(batch, torch.device("cpu")).initial_seed() != reference

    # And the same batch under the same seed is the same bank, whatever the process RNG did.
    torch.manual_seed(99)
    assert task.validation_noise_generator(batch, torch.device("cpu")).initial_seed() == reference


def test_the_target_only_arm_reports_an_exactly_zero_predictive_gap() -> None:
    """The full distribution is the prior, the draws are shared, so the two scores are one."""
    task = build_task(validation_mc_draws=3, source_disabled=True)
    torch.manual_seed(0)
    _, metrics = task.compute_loss_and_metrics(StubBatch(), 0, "val")
    assert torch.equal(metrics["pred_nll_full_mc"], metrics["pred_nll_base_mc"])
    assert float(metrics["pred_gap_mc"]) == 0.0


def test_a_monitored_batch_that_scored_nothing_reports_zero_rather_than_raising() -> None:
    """As the objective does on the same batch."""
    task = build_task(validation_mc_draws=2)
    batch = StubBatch()
    batch.weight = torch.zeros_like(batch.weight)
    torch.manual_seed(0)
    _, metrics = task.compute_loss_and_metrics(batch, 0, "val")
    for name in VALIDATION_MONITOR_SUFFIXES:
        assert float(metrics[name]) == 0.0, name


def test_a_non_positive_draw_count_is_refused_at_construction() -> None:
    """A mixture over no draws is not a score."""
    with pytest.raises(ValueError, match=VALIDATION_MC_DRAWS_KEY):
        build_task(validation_mc_draws=0)


def test_recording_grouped_totals_weight_recordings_equally() -> None:
    """A recording contributing two segments counts once; an unscored one is not counted.

    Hand-built: recording ``a`` holds two samples with two and one scored anchors, recording
    ``b`` one sample with one, and recording ``c`` a sample with nothing scored.
    """
    values = torch.tensor(
        [[1.0, 3.0, 100.0], [5.0, 100.0, 100.0], [7.0, 100.0, 100.0], [9.0, 9.0, 9.0]]
    )
    contributing = torch.tensor(
        [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    total, count = recording_grouped_totals(values, contributing, ["a", "a", "b", "c"])
    # a: (1 + 3 + 5) / 3 = 3; b: 7 / 1 = 7; c: unscored.
    assert float(count) == 2.0
    assert float(total) == pytest.approx(3.0 + 7.0)
    with pytest.raises(ValueError, match="one per sample"):
        recording_grouped_totals(values, contributing, ["a", "b"])


# =================================================================================================
# The clip fraction
# =================================================================================================
def set_gradient_norm(task: SeqVaeLagResidualTrfCfsTask, norm: float) -> None:
    """Give the task's parameters a gradient whose total norm is exactly ``norm``.

    Args:
        task: The task.
        norm: The total gradient norm to plant.
    """
    parameters = [parameter for parameter in task.parameters()]
    for parameter in parameters:
        parameter.grad = torch.zeros_like(parameter)
    parameters[0].grad.view(-1)[0] = float(norm)


def test_the_clip_fraction_counts_every_optimizer_step_of_the_epoch() -> None:
    """Steps between the sampled ones are counted too, and the last batch logs the fraction.

    Four steps against a clip of one, two of them over it, only the last on the logging cadence:
    the logged fraction is one half, not the last step's own zero-or-one.
    """
    task = build_task()
    logged: Dict[str, torch.Tensor] = {}
    task.log = lambda name, value, **kwargs: logged.__setitem__(name, value.detach().clone())
    task._on_train_epoch_start_hook()

    for norm, last in ((2.0, False), (0.5, False), (3.0, False), (0.1, True)):
        task._trainer = SimpleNamespace(gradient_clip_val=1.0, is_last_batch=last, global_step=7)
        set_gradient_norm(task, norm)
        task.on_before_optimizer_step(optimizer=None)
    assert float(logged["train/grad_clip_frac"]) == pytest.approx(0.5)
    assert float(logged["train/grad_norm"]) == pytest.approx(0.1)

    # A new epoch starts its own count.
    task._on_train_epoch_start_hook()
    task._trainer = SimpleNamespace(gradient_clip_val=1.0, is_last_batch=True, global_step=8)
    set_gradient_norm(task, 0.2)
    task.on_before_optimizer_step(optimizer=None)
    assert float(logged["train/grad_clip_frac"]) == 0.0


def test_no_clip_fraction_is_logged_without_a_positive_clip() -> None:
    """A fraction against no threshold answers no question; the norm is still logged."""
    task = build_task()
    logged: Dict[str, torch.Tensor] = {}
    task.log = lambda name, value, **kwargs: logged.__setitem__(name, value.detach().clone())
    task._on_train_epoch_start_hook()
    for clip in (None, 0.0):
        task._trainer = SimpleNamespace(gradient_clip_val=clip, is_last_batch=True, global_step=0)
        set_gradient_norm(task, 4.0)
        task.on_before_optimizer_step(optimizer=None)
        assert "train/grad_clip_frac" not in logged
        assert float(logged["train/grad_norm"]) == pytest.approx(4.0)
