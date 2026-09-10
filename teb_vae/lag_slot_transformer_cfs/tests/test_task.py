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

import pytest
import torch

from teb_vae.lag_attn_cfs.task import SeqVaeLagAttnCfsTask
from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask
from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask
from teb_vae.lag_slot_transformer_cfs.task import (
    TASK_METRIC_SUFFIXES,
    SeqVaeLagResidualTrfCfsTask,
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


def build_task(**overrides) -> SeqVaeLagResidualTrfCfsTask:
    """Wrap a deterministically built tiny model in the task.

    Args:
        **overrides: Constructor keywords for the model.

    Returns:
        The task, with its source pathway moved off zero so the two branches differ.
    """
    model = build_tiny_model(**overrides)
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
