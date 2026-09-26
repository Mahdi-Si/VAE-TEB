r"""The task is one diamond: the causal-input cell on one side, the step-granular ramp on the other.

The class body is empty, so what can go wrong is only *where each half resolves*. Both parents
descend from the shared task, so a member defined on both sides would resolve to the causal parent by
order alone, silently. The tests below read one behaviour from each side through the diamond: the
learning-rate ramp the conv-Transformer parent adds, and the five-argument forward inputs and the
metric surface the causal-input parent adds. The behaviours themselves are the parents' and are
pinned in their own suites.
"""
from __future__ import annotations

import pytest
import torch
from torch.optim.lr_scheduler import LambdaLR

from .conftest import TINY_STRIDE, make_stub_batch

#: The readouts the causal-input parent adds on both stages.
_CAUSAL_METRICS = (
    "anchors_per_sample",
    "source_lag_warmth_frac_st",
    "source_lag_warmth_frac_ph",
)

#: The readouts that cost a second encode or decode per step and never enter the objective, so they
#: run on validation alone: the source-null floor and the permutation control's three.
_VAL_ONLY_METRICS = ("kld_source_null", "nll_shuffled_block", "kld_shuffled", "shuffle_penalty")


class _FakeTrainer:
    """The two properties ``build_lr_scheduler`` reads, and nothing else.

    ``estimated_stepping_batches`` is the optimizer-step total for the **whole run**, not per epoch.
    It is typed ``float`` because an unlimited run is reported as infinity.
    """

    def __init__(self, estimated_stepping_batches: float, max_epochs: int) -> None:
        self.estimated_stepping_batches = estimated_stepping_batches
        self.max_epochs = max_epochs


def test_the_ramp_is_a_step_granular_lambda_when_configured(task) -> None:
    """One ``LambdaLR`` at ``interval: 'step'``, carrying both the ramp and the milestone decay: a
    ramp measured in steps but stepped once per epoch would take ``lr_warmup_steps`` *epochs* to
    complete, silently.

    Reached here through the diamond rather than through the conv-Transformer task directly, which
    is the whole point: the causal parent comes first in the linearisation, so a member it grew would
    shadow this one.
    """
    warmup = 100
    module = task()
    setattr(module.hparams, "lr_warmup_steps", warmup)
    setattr(module.hparams, "lr_milestones", [])
    module.trainer = _FakeTrainer(10.0 * warmup, 10)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)

    schedule = module.build_lr_scheduler(optimizer)

    assert isinstance(schedule, dict)
    assert isinstance(schedule["scheduler"], LambdaLR)
    assert schedule["interval"] == "step"
    factor = schedule["scheduler"].lr_lambdas[0]
    assert factor(0) == pytest.approx(1.0 / warmup)
    assert factor(warmup - 1) == pytest.approx(1.0)
    assert factor(5 * warmup) == pytest.approx(1.0)


def test_the_forward_takes_the_causal_parents_five_arguments(task, stub_batch) -> None:
    """Five, not the architecture parent's three: the two target-stream blocks, the source stream,
    the per-sample tile phase and the stride."""
    module = task()
    module._stage = "train"

    inputs = module._build_forward_inputs(stub_batch)

    assert len(inputs) == 5
    phase, stride = inputs[3], inputs[4]
    assert stride == TINY_STRIDE
    assert isinstance(phase, torch.Tensor) and phase.shape == (stub_batch.fhr_st.shape[0],)
    assert bool(((phase >= 0) & (phase < TINY_STRIDE)).all())


def test_the_metric_surface_carries_the_causal_readouts_and_the_validation_only_arms(
    task, perturb_posterior
) -> None:
    """Emitted through the diamond exactly as on the conv-LSTM cell of this row. The validation-only
    arms are absent from training, never zero-filled: the framework's epoch value is the mean over
    the steps that reported it, so a zero placeholder would scale the aggregate toward nothing, and
    a tracked ``train/`` variant would be a column that is NaN in every row of every run."""
    module = task()
    perturb_posterior(module.orig_model)
    batch = make_stub_batch()

    _loss, train_metrics = module.compute_loss_and_metrics(batch, 0, "train")
    _loss, val_metrics = module.compute_loss_and_metrics(batch, 0, "val")

    for name in _CAUSAL_METRICS:
        assert name in train_metrics, name
        assert name in val_metrics, name
    for name in _VAL_ONLY_METRICS:
        assert name in val_metrics, name
        assert name not in train_metrics, name
