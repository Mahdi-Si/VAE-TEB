r"""The task is the comparison model's task plus one method, and that is the whole point.

Two architectures are only comparable if they optimise the same thing. Here that is the same code:
the loss, the $\beta$ schedule, the metric surface, the permutation control, the spike-breaker
wiring, the step-granular learning-rate ramp and the checkpoint contract are all inherited. What is
tested here is the one method this package adds -- what the net is handed -- and two consequences
of inheriting the rest from the right parent: the learning-rate ramp is attached per optimizer step,
and the metric set is the comparison model's, driven from both tasks' real metrics dicts rather
than from a list.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask

from .conftest import TASK_HPARAMS


class _FakeTrainer:
    """The two properties ``build_lr_scheduler`` reads off ``self.trainer``, and nothing else.

    ``estimated_stepping_batches`` is the optimizer-step total for the whole run, not per epoch,
    and is typed ``float`` because an unlimited run is reported as infinity.
    """

    def __init__(self, estimated_stepping_batches: float = 100.0, max_epochs: int = 10) -> None:
        self.estimated_stepping_batches = estimated_stepping_batches
        self.max_epochs = max_epochs


def _sibling_task():
    """The comparison model wrapped in its own task, at that suite's tiny geometry.

    Built here rather than imported as a fixture because the two models' keyword schemas differ;
    what the two share is the objective and the metric surface, which is what the comparison below
    is about.

    Returns:
        A ``SeqVaeLagAttnTrfRwsTask`` at the sibling suite's tiny geometry.
    """
    from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
    from teb_vae.lag_attn_transformer_rws.tests.conftest import (
        TINY_KWARGS as SIBLING_TINY_KWARGS,
    )

    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**SIBLING_TINY_KWARGS)
    task = SeqVaeLagAttnTrfRwsTask(
        model, lr=1e-3, model_kwargs=dict(SIBLING_TINY_KWARGS), **TASK_HPARAMS
    )
    task.setup("fit")
    return task


# --------------------------------------------------------------------------------------
# The one method: what the net is handed
# --------------------------------------------------------------------------------------
def test_the_hook_hands_the_net_the_scored_target_the_raw_source_and_the_weight(task, stub_batch):
    """The whole architectural difference, expressed as three tensors. The target and the weight are
    the very objects ``_build_raw_target`` -- the single source of the reconstruction target --
    returns: a model scored against a tensor other than the one it was shown produces a plausible
    loss curve and a meaningless result, and nothing anywhere raises."""
    module = task()

    inputs = module._build_forward_inputs(stub_batch)
    scored_target, weight = module._build_raw_target(stub_batch)

    assert len(inputs) == 3
    assert inputs[0] is scored_target
    assert torch.equal(inputs[0], stub_batch.fhr)
    assert torch.equal(inputs[1], stub_batch.up)
    assert inputs[2] is weight


def test_a_batch_without_the_raw_source_raises_naming_both_config_lists(task, stub_batch):
    """Missing from ``load_fields`` the source stream does not exist; missing from
    ``normalize_fields`` nothing fails at all -- the front end owns no statistics of its own, so an
    unnormalized source shifts every coupling number the run reports in silence."""
    module = task()
    del stub_batch.up

    with pytest.raises(RuntimeError, match="load_fields") as excinfo:
        module._build_forward_inputs(stub_batch)

    assert "normalize_fields" in str(excinfo.value)
    assert "`up`" in str(excinfo.value)


def test_the_feature_block_builders_are_never_reached(task, stub_batch):
    """The two stream builders this hook replaces read the stored feature blocks and check them
    against widths this net does not have. They are unreachable rather than merely unused: a batch
    carrying none of those fields still produces a finite loss that carries gradient."""
    module = task()
    for field in ("fhr_st", "fhr_ph", "up_st", "up_ph"):
        delattr(stub_batch, field)

    loss, _metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert torch.isfinite(loss)
    assert loss.requires_grad


# --------------------------------------------------------------------------------------
# The learning-rate schedule
# --------------------------------------------------------------------------------------
def test_the_step_warmup_is_attached_at_step_granularity(task):
    """The assertion that catches subclassing the wrong parent: the step ramp exists only on the
    conv-Transformer task, and a ramp measured in optimizer steps but attached at
    ``interval: "epoch"`` takes ``lr_warmup_steps`` *epochs* -- silently. The ramp's arithmetic
    itself is inherited code and is pinned in the sibling's own suite."""
    module = task()
    setattr(module.hparams, "lr_warmup_steps", 4)
    module.trainer = _FakeTrainer()

    schedule = module.build_lr_scheduler(
        torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    )

    assert isinstance(schedule, dict)
    assert schedule["interval"] == "step"
    assert schedule["scheduler"] is not None


# --------------------------------------------------------------------------------------
# The metric surface
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("stage", ["train", "val"])
def test_the_metric_set_is_identical_to_the_comparison_models(
    task, stub_batch, perturb_posterior, stage
):
    """Driven from both models' real metrics dicts rather than from a list. A metric this model
    emitted and the other did not would be a column the inherited tracked-metric list does not
    collect; one it lost would be a readout the comparison can no longer be made on. Validation
    adds the permutation-control readouts, so a control that stopped running shows there."""
    mine = task()
    theirs = _sibling_task()
    perturb_posterior(mine.orig_model)
    perturb_posterior(theirs.orig_model)

    _, my_metrics = mine.compute_loss_and_metrics(stub_batch, 0, stage)
    _, their_metrics = theirs.compute_loss_and_metrics(stub_batch, 0, stage)

    assert set(my_metrics) == set(their_metrics)
