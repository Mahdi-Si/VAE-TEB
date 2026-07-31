r"""The step-granular learning-rate warm-up, traced rather than described.

A schedule is the kind of thing that silently does nothing. A ramp measured in optimizer steps but
attached at ``interval: "epoch"`` takes ``lr_warmup_steps`` *epochs*; a milestone left in epoch
units against a step-granular scheduler fires thousands of steps early; a ``SequentialLR`` whose
second scheduler restarts its counter fires them late. None of those raises, and at epoch
granularity the framework's own monitor cannot show any of them. So the assertions below step a
real scheduler and read the learning rate back off the optimizer, against hand-computed values.

The zero case is the one that is easy to leave untested and easy to break: at
``lr_warmup_steps = 0`` this must be the framework's schedule *elementwise*, because that is what
keeps the epoch-granularity path reachable from configuration rather than merely believed to be.
"""
from __future__ import annotations

from typing import List

import pytest
import torch

from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask

#: Milestones and the run shape the traces below are computed against. Small, because the trace is
#: stepped for real: $50$ steps of a ``LambdaLR`` is instant and $800$ would not be.
MILESTONES = [3, 5]
TOTAL_STEPS = 100
MAX_EPOCHS = 10

#: Optimizer steps per epoch this run shape implies, and therefore the multiplier that turns an
#: epoch milestone into a step milestone.
STEPS_PER_EPOCH = TOTAL_STEPS // MAX_EPOCHS

#: The framework's decay factor per milestone; the task reads it from the same hparam.
GAMMA = 0.1

#: Base learning rate every trace is expressed as a fraction of.
BASE_LR = 1e-3


class _FakeTrainer:
    """The two properties ``build_lr_scheduler`` reads, and nothing else.

    ``estimated_stepping_batches`` is the optimizer-step total for the **whole run**, not per
    epoch -- reading it as per-epoch is the arithmetic error this fake exists to make visible. It
    is typed ``float`` because an unlimited run is reported as infinity, not as a large integer.
    """

    def __init__(self, estimated_stepping_batches: float, max_epochs: int) -> None:
        self.estimated_stepping_batches = estimated_stepping_batches
        self.max_epochs = max_epochs


def _attach(module, warmup_steps: int, *, total_steps: float = TOTAL_STEPS,
            max_epochs: int = MAX_EPOCHS):
    """Set the warm-up hparam and a fake trainer on ``module``, as a real run would.

    ``lr_warmup_steps`` reaches the task through ``apply_config_hyperparameters``, which sets it
    on ``hparams`` after construction; that is reproduced here rather than routed around, so the
    test drives the same attribute lookup the driver populates.

    Args:
        module: The task.
        warmup_steps: The value to configure.
        total_steps: What the trainer reports as the run's optimizer-step total.
        max_epochs: What the trainer reports as its epoch count.

    Returns:
        ``module``.
    """
    setattr(module.hparams, "lr_warmup_steps", warmup_steps)
    module.trainer = _FakeTrainer(total_steps, max_epochs)
    return module


def _trace(module, steps: int) -> List[float]:
    """Step the module's schedule ``steps`` times and return the learning rate at each step.

    Args:
        module: A task with its schedule configured.
        steps: How many scheduler steps to record.

    Returns:
        The learning rate *before* each step, so index $s$ is the rate step $s$ trains at.
    """
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)
    schedule = module.build_lr_scheduler(optimizer)
    scheduler = schedule["scheduler"] if isinstance(schedule, dict) else schedule
    values = []
    for _ in range(steps):
        values.append(float(optimizer.param_groups[0]["lr"]))
        if scheduler is not None:
            scheduler.step()
    return values


# --------------------------------------------------------------------------------------
# The zero case: the framework's schedule, elementwise
# --------------------------------------------------------------------------------------
def test_zero_warmup_steps_is_the_frameworks_schedule_elementwise(task):
    """Not "similar": the same numbers. At $0$ the epoch-granularity path -- including the
    framework's own ``lr_warmup_epochs`` -- stays reachable from configuration at no cost here, and
    a divergence would mean two schedules exist where the config describes one."""
    module = _attach(task(lr_milestones=MILESTONES), 0)
    steps = MILESTONES[0] + 2

    mine = _trace(module, steps)
    # The base implementation, called on the same module with the same hparams.
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)
    schedule = SeqVaeLagAttnRwsTask.build_lr_scheduler(module, optimizer)
    scheduler = schedule["scheduler"] if isinstance(schedule, dict) else schedule
    theirs = []
    for _ in range(steps):
        theirs.append(float(optimizer.param_groups[0]["lr"]))
        scheduler.step()

    assert mine == theirs
    # And the trace is not a constant, so "elementwise equal" is not equality between two flat
    # lines: the milestone decay has to have fired inside the window.
    assert len(set(mine)) > 1


def test_zero_warmup_steps_keeps_the_frameworks_epoch_interval(task):
    """A step-granular interval on an epoch-granular schedule would apply the milestone decay
    ``steps_per_epoch`` times too early."""
    module = _attach(task(lr_milestones=MILESTONES), 0)

    schedule = module.build_lr_scheduler(torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))]))

    assert schedule["interval"] == "epoch"


def test_no_milestones_and_no_warmup_is_no_scheduler_at_all(task):
    """The framework's own contract, preserved: a run that configures neither gets a constant
    learning rate rather than an identity scheduler that logs as if it were doing something."""
    module = _attach(task(), 0)

    assert module.build_lr_scheduler(torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])) is None


# --------------------------------------------------------------------------------------
# The step ramp
# --------------------------------------------------------------------------------------
def test_the_step_warmup_ramps_holds_and_then_decays(task):
    r"""Four hand-computed points on the trace of

    $$\mathrm{factor}(s) = \min\!\left(1, \frac{s+1}{S_{\mathrm{warm}}}\right)
      \cdot \gamma^{\left|\{m : m \le s\}\right|}.$$

    The first epoch milestone is epoch $3$ of a $100$-step, $10$-epoch run, so it lands at step
    $30$ -- and landing there rather than at step $3$ is the whole content of the epoch-to-step
    conversion.
    """
    warmup_steps = 4
    module = _attach(task(lr_milestones=MILESTONES), warmup_steps)

    trace = _trace(module, STEPS_PER_EPOCH * (MILESTONES[1] + 1))

    assert trace[0] == pytest.approx(BASE_LR / warmup_steps)  # ramps from lr_max / N
    assert trace[warmup_steps - 1] == pytest.approx(BASE_LR)  # reaches lr_max at the last ramp step
    assert trace[warmup_steps] == pytest.approx(BASE_LR)  # and holds
    first_milestone = MILESTONES[0] * STEPS_PER_EPOCH
    assert trace[first_milestone - 1] == pytest.approx(BASE_LR)  # still undecayed the step before
    assert trace[first_milestone] == pytest.approx(BASE_LR * GAMMA)  # drops by gamma exactly there
    second_milestone = MILESTONES[1] * STEPS_PER_EPOCH
    assert trace[second_milestone] == pytest.approx(BASE_LR * GAMMA**2)


def test_the_ramp_is_linear_and_starts_from_a_fraction_rather_than_a_tenth(task):
    r"""The framework's ``LinearLR`` path cannot express this at all: it rejects
    ``start_factor=0.0`` and its own value is $0.1$, a tenfold discontinuity at step zero. A ramp
    from $1/N$ approaches zero as $N$ grows, which is what a fragile pre-norm stack needs."""
    warmup_steps = 8
    module = _attach(task(lr_milestones=MILESTONES), warmup_steps)

    trace = _trace(module, warmup_steps)

    for step in range(warmup_steps):
        assert trace[step] == pytest.approx(BASE_LR * (step + 1) / warmup_steps)


def test_the_step_schedule_is_stepped_once_per_optimizer_step(task):
    """A ramp measured in steps and stepped once per epoch takes ``lr_warmup_steps`` epochs to
    complete, silently."""
    module = _attach(task(lr_milestones=MILESTONES), 4)

    schedule = module.build_lr_scheduler(torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))]))

    assert schedule["interval"] == "step"
    assert schedule["frequency"] == 1


@pytest.mark.parametrize(
    "total_steps, max_epochs",
    [
        (0, 0),
        # An unlimited run -- `epochs: -1` to train until early stopping, or a dataloader of
        # unknown length. Lightning reports the step total as *infinity* here, so this case is
        # not a smaller version of the one above: converting it to an int raises, and the raise
        # would land inside `configure_optimizers`, with the run directory and every DDP rank
        # already up.
        (float("inf"), -1),
    ],
    ids=["no-epoch-count", "unlimited-run"],
)
def test_a_run_reporting_no_epoch_count_still_produces_a_monotone_milestone_sequence(
    task, total_steps, max_epochs
):
    """``max_epochs`` can be zero or unlimited. The floor keeps every milestone distinct rather
    than collapsing them all onto step zero, which would decay the rate twice on the first step."""
    module = _attach(
        task(lr_milestones=MILESTONES), 2, total_steps=total_steps, max_epochs=max_epochs
    )

    trace = _trace(module, MILESTONES[1] + 2)

    assert trace[0] == pytest.approx(BASE_LR / 2)
    assert trace[MILESTONES[0]] == pytest.approx(BASE_LR * GAMMA)
    assert trace[MILESTONES[1]] == pytest.approx(BASE_LR * GAMMA**2)


def test_a_negative_warmup_raises_naming_the_value(task):
    """Silently clamping it to zero would train under the framework's schedule while the config
    described a ramp."""
    module = _attach(task(lr_milestones=MILESTONES), -5)

    with pytest.raises(ValueError, match="-5"):
        module.build_lr_scheduler(torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))]))


def test_the_configured_warmup_reaches_the_task_through_the_driver(tmp_path):
    """The seam that fails silently: ``lr_warmup_steps`` lives in the config and the inherited
    ``create_model`` knows nothing about it, so a driver that did not forward it would leave the
    step path configured and dead, with the run training under the epoch schedule instead."""
    from pathlib import Path

    import yaml

    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer

    from .conftest import absolutize_dataset_paths

    tiny = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"
    config = absolutize_dataset_paths(load_config(str(tiny)))
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    driver = LagAttnTrfRwsTrainer(config_file_path=str(path))
    driver.create_model()

    configured = config["general_config"]["lr_warmup_steps"]
    assert configured > 0
    assert int(driver.pl_model.hparams["lr_warmup_steps"]) == configured
