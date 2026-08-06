r"""The task is the comparison model's task plus one method, and that is the whole point.

Two architectures are only comparable if they optimise the same thing. Here that is not a claim
about two copies of an objective agreeing -- it is the same code: the loss, the $\beta$ schedule,
the metric surface, the permutation control, the spike-breaker wiring, the gradient-norm logging,
the step-granular learning-rate ramp and the checkpoint contract are all inherited unmodified. So
the assertions below are mostly about *absence*: a re-added ``training_step`` silently disables the
config-gated loss-spike breaker, a re-added constructor is a second keyword schema for the same
objective, and neither fails anything on its own.

Two of them are about which parent was subclassed, which is this package's own way to get it
wrong. The step ramp lives only on the conv-Transformer task; a subclass of the raw-signal one
would leave the config requesting ``lr_warmup_steps: 2000``, the monitor logging per step, and no
ramp existing at all, with nothing raising.

The metric-set assertion is driven from both models' real metrics dicts on the same batch, rather
than from a list written here, so it is a fact about the two tasks and not about two copies of a
table.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask
from teb_vae.lag_attn_transformer_e2e.task import SeqVaeLagAttnTrfE2ETask
from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask

from .conftest import TASK_HPARAMS, TINY_KWARGS


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

    Built here rather than imported as a fixture because the two models' keyword schemas differ:
    that one declares three stored-feature widths this one has never heard of, and this one has a
    front-end kernel schedule that one does not. What the two share is the objective and the
    metric surface, which is exactly what the comparison below is about.

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
# What the subclass is
# --------------------------------------------------------------------------------------
def test_the_subclass_adds_exactly_one_method():
    """A second override is a second thing that can diverge from the objective being shared.

    ``__module__``, ``__doc__`` and the like are class-body bookkeeping rather than behaviour, so
    only callables are counted.
    """
    added = {
        name
        for name, value in vars(SeqVaeLagAttnTrfE2ETask).items()
        if callable(value) and not name.startswith("__")
    }

    assert added == {"_build_forward_inputs"}


def test_the_parent_is_the_conv_transformer_task_not_the_raw_signal_one():
    """The one inheritance mistake this package can make and not notice.

    ``build_lr_scheduler`` -- the step-granular ramp -- exists only on the conv-Transformer task.
    Subclassing the raw-signal one instead would leave ``lr_warmup_steps: 2000`` in the config, the
    learning-rate monitor logging per step, and no ramp attached anywhere, with nothing raising and
    the CSV column looking merely flat.
    """
    assert issubclass(SeqVaeLagAttnTrfE2ETask, SeqVaeLagAttnTrfRwsTask)
    assert SeqVaeLagAttnTrfE2ETask.build_lr_scheduler is SeqVaeLagAttnTrfRwsTask.build_lr_scheduler
    assert (
        SeqVaeLagAttnTrfE2ETask.build_lr_scheduler is not SeqVaeLagAttnRwsTask.build_lr_scheduler
    )


@pytest.mark.parametrize(
    "method",
    ["training_step", "validation_step", "test_step", "forward", "configure_optimizers",
     "compute_loss_and_metrics", "on_save_checkpoint", "build_lr_scheduler", "__init__"],
)
def test_the_inherited_machinery_is_not_taken_back(method):
    """``training_step`` is the one that matters most: the framework's version runs the
    config-gated spike breaker, and a subclass that defines its own silently disables it.
    ``configure_optimizers`` is the second: it is what calls ``build_lr_scheduler`` at all."""
    assert method not in vars(SeqVaeLagAttnTrfE2ETask), (
        f"{method} is overridden; the inherited implementation is the seam this model uses"
    )


def test_the_constructor_keyword_set_is_the_siblings():
    """It is what a later evaluation entry point would reconstruct the task from, so a keyword this
    task took and the sibling did not would be a second objective schema for one shared objective.
    Compared as an ordered mapping, because a reordered signature is still a different call."""
    mine = inspect.signature(SeqVaeLagAttnTrfE2ETask.__init__).parameters
    theirs = inspect.signature(SeqVaeLagAttnTrfRwsTask.__init__).parameters

    assert list(mine) == list(theirs)
    assert "compile_model" not in mine  # a property of the net, not a caller's choice


def test_compilation_stays_off_and_the_eager_module_is_what_runs(task):
    """Inductor is defeated by the data-dependent boolean mask indexing behind
    ``kld_active_frac`` and by the masked source KL, both of which are inherited."""
    module = task()

    assert module.model is module.orig_model
    assert module.hparams.get("compile_model") is False


# --------------------------------------------------------------------------------------
# The one method: what the net is handed
# --------------------------------------------------------------------------------------
def test_the_hook_returns_the_two_raw_signals_and_the_weight(task, stub_batch):
    """The whole architectural difference, expressed as three tensors."""
    module = task()

    inputs = module._build_forward_inputs(stub_batch)

    assert len(inputs) == 3
    assert torch.equal(inputs[0], stub_batch.fhr)
    assert torch.equal(inputs[1], stub_batch.up)
    assert torch.equal(inputs[2], stub_batch.weight)


def test_the_target_the_front_end_reads_is_the_tensor_the_loss_scores(task, stub_batch):
    """Identity, not equality. A model scored against a tensor other than the one it was shown
    produces a plausible loss curve and a meaningless result, and nothing anywhere raises -- so the
    guarantee is that both come from ``_build_raw_target``, the single source of the target."""
    module = task()

    forward_input = module._build_forward_inputs(stub_batch)[0]
    scored_target, weight = module._build_raw_target(stub_batch)

    assert forward_input is scored_target
    assert module._build_forward_inputs(stub_batch)[2] is weight


def test_the_single_source_of_the_target_is_not_overridden():
    """``_build_raw_target`` stays the inherited one: one builder for the reconstruction target is
    what stops the target and the target stream's input drifting apart."""
    assert "_build_raw_target" not in vars(SeqVaeLagAttnTrfE2ETask)


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


def test_a_batch_without_the_raw_target_raises_naming_both_config_lists(task, stub_batch):
    """The inherited refusal, reached through this hook rather than through the target builder's
    own call site -- which is what makes it the *first* thing a misconfigured run hits."""
    module = task()
    del stub_batch.fhr

    with pytest.raises(RuntimeError, match="load_fields") as excinfo:
        module._build_forward_inputs(stub_batch)

    assert "normalize_fields" in str(excinfo.value)
    assert "`fhr`" in str(excinfo.value)


def test_the_feature_block_builders_are_never_reached(task, stub_batch):
    """The two stream builders this hook replaces read ``fhr_st``/``fhr_ph``/``up_st``/``up_ph``
    and check them against ``c_y``/``c_u``, neither of which this net has. They are unreachable
    rather than merely unused: a batch carrying none of those fields still produces a loss."""
    module = task()
    for field in ("fhr_st", "fhr_ph", "up_st", "up_ph"):
        delattr(stub_batch, field)

    loss, _metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert torch.isfinite(loss)


# --------------------------------------------------------------------------------------
# The learning-rate schedule
# --------------------------------------------------------------------------------------
def test_the_step_warmup_is_attached_at_step_granularity(task):
    """The single assertion that catches the wrong-parent error above. A ramp measured in optimizer
    steps but attached at ``interval: "epoch"`` takes ``lr_warmup_steps`` *epochs* -- silently. The
    ramp's arithmetic itself is inherited code and is pinned in the sibling's own suite."""
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
def test_the_train_metric_set_is_identical_to_the_comparison_models(
    task, stub_batch, perturb_posterior
):
    """Driven from both models' real metrics dicts rather than from a list, so the claim is about
    the two tasks. A metric this model emitted and the other did not would be a column the
    inherited tracked-metric list does not collect; one it lost would be a readout the comparison
    can no longer be made on."""
    mine = task()
    theirs = _sibling_task()
    perturb_posterior(mine.orig_model)
    perturb_posterior(theirs.orig_model)

    _, my_metrics = mine.compute_loss_and_metrics(stub_batch, 0, "train")
    _, their_metrics = theirs.compute_loss_and_metrics(stub_batch, 0, "train")

    assert set(my_metrics) == set(their_metrics)


def test_the_validation_metric_set_is_identical_too(task, stub_batch, perturb_posterior):
    """Validation adds the three permutation-control readouts, which only a source-conditioned
    forward can produce -- so this is also where a control that silently stopped running shows."""
    mine = task()
    theirs = _sibling_task()
    perturb_posterior(mine.orig_model)
    perturb_posterior(theirs.orig_model)

    _, my_metrics = mine.compute_loss_and_metrics(stub_batch, 0, "val")
    _, their_metrics = theirs.compute_loss_and_metrics(stub_batch, 0, "val")

    assert set(my_metrics) == set(their_metrics)
    assert {"nll_shuffled_block", "kld_shuffled", "shuffle_penalty"} <= set(my_metrics)


def test_every_metric_is_numeric_and_unprefixed(task, stub_batch, perturb_posterior):
    """A name carrying a '/' bypasses stage framing and can poison a ``ModelCheckpoint`` monitor."""
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    for name, value in metrics.items():
        assert isinstance(value, torch.Tensor), f"{name} is a {type(value).__name__}"
        assert "/" not in name


def test_the_loss_is_finite_and_carries_gradient(task, stub_batch, perturb_posterior):
    module = task()
    perturb_posterior(module.orig_model)

    loss, _ = module.compute_loss_and_metrics(stub_batch, 1, "train")

    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_the_zero_kl_start_survives_the_task(task, stub_batch):
    """At initialisation the posterior *is* the prior, so the coupling readout starts at exactly
    zero and every nat it later reports had to be earned. Seen here through the task's own
    diagnostic, after the batch assembly and the loss have both run."""
    module = task()

    _, metrics = module.compute_loss_and_metrics(stub_batch, 1, "train")

    assert float(metrics["source_conditioned_kl_raw"]) == pytest.approx(0.0, abs=1e-6)
    assert float(metrics["mu_post_prior_gap_rms"]) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# The checkpoint stamp
# --------------------------------------------------------------------------------------
def test_a_checkpoint_written_by_this_task_stamps_this_model(task):
    """The stamp is what lets a loader refuse a blob from either sibling before trying to align it
    -- all three share every tensor below the encoder inputs and would partially align."""
    module = task()
    checkpoint = {"state_dict": module.state_dict(), "epoch": 3}

    module.on_save_checkpoint(checkpoint)

    assert checkpoint["model_class"] == "SeqVaeLagAttnTrfE2E"
    assert checkpoint["model_kwargs"] == TINY_KWARGS
    assert checkpoint["epoch"] == 3  # the override must not clobber Lightning's own fields
