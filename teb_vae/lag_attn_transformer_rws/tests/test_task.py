r"""The task is the comparison model's task plus one method, and that is the whole point.

Two architectures are only comparable if they optimise the same thing. Here that is not a claim
about two copies of an objective agreeing -- it is the same code: the loss, the $\beta$ schedule,
the metric surface, the permutation control, the spike-breaker wiring, the gradient-norm logging
and the checkpoint contract are all inherited unmodified. So the assertions below are mostly about
*absence*: a re-added ``training_step`` silently disables the config-gated loss-spike breaker, a
re-added constructor is a second keyword schema for the same objective, and neither fails anything
on its own.

The metric-set assertion is driven from both models' real metrics dicts on the same batch, rather
than from a list written here, so it is a fact about the two tasks and not about two copies of a
table.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask
from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask

from .conftest import TASK_HPARAMS, TINY_KWARGS


def _sibling_task():
    """The comparison model wrapped in its own task, at its own tiny geometry.

    Built here rather than imported as a fixture because the two models' keyword schemas differ:
    this one has seven encoder keys that one has never heard of, and five of that one's keys name
    nothing here. What the two share is the batch contract and the objective, which is exactly what
    the metric-set comparison is about.

    Returns:
        A ``SeqVaeLagAttnRwsTask`` at the sibling suite's tiny geometry.
    """
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
    from teb_vae.lag_attn_rws.tests.conftest import TINY_KWARGS as SIBLING_TINY_KWARGS

    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**SIBLING_TINY_KWARGS)
    task = SeqVaeLagAttnRwsTask(model, lr=1e-3, model_kwargs=dict(SIBLING_TINY_KWARGS), **TASK_HPARAMS)
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
        for name, value in vars(SeqVaeLagAttnTrfRwsTask).items()
        if callable(value) and not name.startswith("__")
    }

    assert added == {"build_lr_scheduler"}


@pytest.mark.parametrize(
    "method",
    ["training_step", "validation_step", "test_step", "forward", "configure_optimizers",
     "compute_loss_and_metrics", "on_save_checkpoint", "__init__"],
)
def test_the_inherited_machinery_is_not_taken_back(method):
    """``training_step`` is the one that matters most: the framework's version runs the
    config-gated spike breaker, and a subclass that defines its own silently disables it.
    ``configure_optimizers`` is the second: it is what calls ``build_lr_scheduler`` at all."""
    assert method not in vars(SeqVaeLagAttnTrfRwsTask), (
        f"{method} is overridden; the inherited implementation is the seam this model uses"
    )


def test_the_task_is_the_comparison_models_task(task):
    module = task()

    assert isinstance(module, SeqVaeLagAttnRwsTask)


def test_compilation_stays_off_and_the_eager_module_is_what_runs(task):
    """The recurrence is gone, but the data-dependent boolean mask indexing behind
    ``kld_active_frac`` and the masked source KL still defeat inductor."""
    module = task()

    assert module.model is module.orig_model
    assert module.hparams.get("compile_model") is False


def test_compilation_is_not_a_constructor_argument():
    """A property of the net, not a caller's choice, so no config can re-enable it."""
    assert "compile_model" not in inspect.signature(SeqVaeLagAttnTrfRwsTask.__init__).parameters


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
    """The stamp is what lets a loader refuse a blob from the comparison model before trying to
    align it -- the two share every downstream tensor name and would partially align by accident."""
    module = task()
    checkpoint = {"state_dict": module.state_dict(), "epoch": 3}

    module.on_save_checkpoint(checkpoint)

    assert checkpoint["model_class"] == "SeqVaeLagAttnTrfRws"
    assert checkpoint["model_kwargs"] == TINY_KWARGS
    assert checkpoint["epoch"] == 3  # the override must not clobber Lightning's own fields
