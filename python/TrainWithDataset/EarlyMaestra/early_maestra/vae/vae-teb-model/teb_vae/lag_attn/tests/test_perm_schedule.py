r"""The fused source-permutation control and its DDP-safe schedule.

The control rides inside the **single main backward**. That is what lets the task keep Lightning's
automatic optimization -- and with it the gradient clip, gradient accumulation, LR scheduler and
loss-spike circuit breaker -- while the strategy selector still returns plain ``'ddp'``.

Two invariants make that safe, and both are pinned here:

* On *every* step type, perm and non-perm alike, each ``requires_grad`` parameter receives a
  gradient. Under ``find_unused_parameters=False`` the DDP reducer expects exactly that; a
  parameter left ungradiented on some steps raises or deadlocks. This -- not the strategy string --
  is what licenses the fast strategy.
* The schedule is a pure function of ``batch_idx``, which Lightning keeps identical across ranks,
  and is MIN-reduced besides, so a rank with a degenerate ($B < 2$) batch cannot branch alone.
"""
from __future__ import annotations

import pytest
import torch


def _perm_active(metrics) -> bool:
    """Whether the control ran on this step, as an outside observer would tell.

    Presence, not value: the control's metrics are omitted entirely on the steps it skips, because
    a zero placeholder would be averaged into the epoch aggregate as if it were a measurement.
    """
    return "kld_shuffled" in metrics


def test_the_schedule_fires_only_on_multiples_of_perm_every_n_batches(
    task, stub_batch, perturb_posterior
):
    module = task()
    perturb_posterior(module.orig_model)  # else every KL is 0 and the check is vacuous
    assert module.orig_model.perm_every_n_batches == 2

    fired = [
        _perm_active(module.compute_loss_and_metrics(stub_batch, batch_idx, "train")[1])
        for batch_idx in range(6)
    ]

    assert fired == [True, False, True, False, True, False]


def test_perm_loss_enters_the_total_only_on_scheduled_steps(task, stub_batch, perturb_posterior):
    module = task()
    perturb_posterior(module.orig_model)
    assert float(module.orig_model.lambda_perm) > 0.0

    total_on, metrics_on = module.compute_loss_and_metrics(stub_batch, 0, "train")
    total_off, metrics_off = module.compute_loss_and_metrics(stub_batch, 1, "train")

    # On-schedule: total = main + lambda_perm * L_perm, and L_perm is genuinely nonzero.
    assert float(metrics_on["perm_loss"]) > 0.0
    assert float(metrics_on["kld_shuffled"]) > 0.0
    assert float(total_on) == pytest.approx(
        float(metrics_on["main_loss"]) + float(metrics_on["perm_loss"]), rel=1e-6
    )
    # Off-schedule: the control contributes nothing, and reports nothing.
    assert "perm_loss" not in metrics_off
    assert float(total_off) == pytest.approx(float(metrics_off["main_loss"]), rel=1e-6)


def test_main_loss_is_perm_free_on_every_step(task, stub_batch, perturb_posterior):
    """What makes ``comparison_metric: main_loss`` worth configuring.

    The control fires periodically, so a breaker watching the returned ``total_loss`` would see a
    periodic jump, learn an EMA between the two levels, and eventually skip every perm step -- a
    breaker whose own statistic is the artefact it reacts to.
    """
    module = task()
    perturb_posterior(module.orig_model)

    # Seeded before *each* forward: the model samples z, so two calls on one batch differ by the
    # reparameterisation noise alone. Without this the comparison below measures that noise.
    torch.manual_seed(1)
    total_on, metrics_on = module.compute_loss_and_metrics(stub_batch, 0, "train")
    torch.manual_seed(1)
    _, metrics_off = module.compute_loss_and_metrics(stub_batch, 1, "train")

    # The perm step's returned loss differs from its main loss...
    assert float(total_on) != pytest.approx(float(metrics_on["main_loss"]), rel=1e-9)
    # ...but main_loss itself is the same quantity on both steps, so the series has no jump.
    assert float(metrics_on["main_loss"]) == pytest.approx(float(metrics_off["main_loss"]), rel=1e-6)


def test_main_loss_is_detached(task, stub_batch, perturb_posterior):
    """It is a report, not a term. A metric carrying a graph would keep it alive after the step."""
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert not metrics["main_loss"].requires_grad


def test_the_shuffled_kl_ratio_is_reported(task, stub_batch, perturb_posterior):
    r"""The readout $K_{\mathrm{shuffled}} / K_{\mathrm{raw}}$."""
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    expected = float(metrics["kld_shuffled"]) / float(metrics["kld_raw"])
    assert float(metrics["kld_shuffled_ratio"]) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("batch_idx", [0, 1])
def test_every_parameter_receives_a_gradient(task, stub_batch, perturb_posterior, batch_idx):
    """The evidence that plain ``'ddp'`` is safe, on both a perm and a plain step."""
    module = task()
    perturb_posterior(module.orig_model)

    module.zero_grad(set_to_none=True)
    loss, _ = module.compute_loss_and_metrics(stub_batch, batch_idx, "train")
    loss.backward()

    starved = [
        name
        for name, parameter in module.orig_model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not starved, f"parameters without gradient on batch_idx={batch_idx}: {starved}"


def test_a_single_backward_is_enough(task, stub_batch, perturb_posterior):
    """A parameter used twice in one graph must not need a second backward.

    The permutation branch re-runs the attention and the posterior head. Both usages share one
    ``AccumulateGrad`` node, so a single ``.backward()`` accumulates them and DDP marks each
    parameter ready exactly once. A second ``manual_backward`` would raise under plain ``'ddp'``.
    """
    module = task()
    perturb_posterior(module.orig_model)
    attention_parameter = module.orig_model.lag_attn.lag_embeddings

    module.zero_grad(set_to_none=True)
    loss_off, _ = module.compute_loss_and_metrics(stub_batch, 1, "train")
    loss_off.backward()
    grad_without_perm = attention_parameter.grad.detach().clone()

    module.zero_grad(set_to_none=True)
    loss_on, _ = module.compute_loss_and_metrics(stub_batch, 0, "train")
    loss_on.backward()  # exactly one backward, even though the attention ran twice
    grad_with_perm = attention_parameter.grad.detach().clone()

    assert torch.isfinite(grad_with_perm).all()
    assert not torch.allclose(grad_with_perm, grad_without_perm), (
        "the permutation branch contributed no gradient to the shared attention parameters"
    )


def test_validation_runs_the_control_every_step_without_adding_to_the_loss(
    task, stub_batch, perturb_posterior
):
    r"""``val/kld_shuffled`` against ``val/kld_raw`` is a headline readout, so it must be measured
    on every validation batch -- and it must never enter the validation loss, which a
    ``ModelCheckpoint`` monitors."""
    module = task()
    perturb_posterior(module.orig_model)

    for batch_idx in (0, 1, 3):
        loss, metrics = module.compute_loss_and_metrics(stub_batch, batch_idx, "val")
        assert float(metrics["kld_shuffled"]) > 0.0, "control skipped on a validation batch"
        assert float(metrics["perm_loss"]) == 0.0
        assert float(loss) == pytest.approx(float(metrics["main_loss"]), rel=1e-6)


def test_a_degenerate_batch_skips_the_control(task, make_stub_batch_fn, perturb_posterior):
    """A single-sample batch cannot be deranged, and the last batch of a shard often is one.

    Skipped, not crashed: the derangement helper raises for $B < 2$ by design, so the schedule --
    not a caught exception -- is what has to keep it away from that call.
    """
    module = task()
    perturb_posterior(module.orig_model)

    loss, metrics = module.compute_loss_and_metrics(make_stub_batch_fn(batch_size=1), 0, "train")

    assert "perm_loss" not in metrics
    assert "kld_shuffled" not in metrics
    assert torch.isfinite(loss)


def test_the_control_metrics_are_omitted_rather_than_zeroed_when_it_does_not_run(
    task, make_stub_batch_fn, perturb_posterior
):
    """Zeros would not be a neutral placeholder here; they would be wrong numbers.

    The framework logs every metric with ``on_epoch=True``, whose epoch value is the mean over the
    steps that reported it. A zero on each of the 3-in-4 steps the control skips would quarter the
    epoch-aggregated ``train/feat_loss_shuffled``, inverting the very ordering the control exists to
    check and making a healthy model read as a collapsed one. Omitted, the mean covers the perm
    steps alone.
    """
    module = task()
    perturb_posterior(module.orig_model)
    control_keys = {
        "perm_loss",
        "kld_shuffled",
        "kld_shuffled_ratio",
        "feat_loss_shuffled",
        "shuffle_penalty",
    }

    _, on_schedule = module.compute_loss_and_metrics(make_stub_batch_fn(), 0, "train")
    _, off_schedule = module.compute_loss_and_metrics(make_stub_batch_fn(), 1, "train")
    _, degenerate = module.compute_loss_and_metrics(make_stub_batch_fn(batch_size=1), 0, "train")

    assert control_keys <= set(on_schedule)
    assert control_keys.isdisjoint(off_schedule)
    assert control_keys.isdisjoint(degenerate)


def test_lambda_perm_zero_keeps_the_readout_but_leaves_the_loss_alone(
    task, stub_batch, prod_kwargs, perturb_posterior
):
    """The production default. The control ships as a diagnostic, not an objective."""
    module = task(model_kwargs=dict(prod_kwargs, lambda_perm=0.0))
    perturb_posterior(module.orig_model)

    total, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert float(metrics["perm_loss"]) == 0.0
    assert float(total) == pytest.approx(float(metrics["main_loss"]), rel=1e-6)
    # ...but the readout is still measured and logged.
    assert float(metrics["kld_shuffled"]) > 0.0
    assert float(metrics["feat_loss_shuffled"]) > 0.0


def test_the_prediction_space_control_is_reported(task, stub_batch, perturb_posterior):
    r"""``feat_loss_shuffled`` is the control that actually discriminates.

    $\mathrm{KL}(q\|p)$ rises under a deranged source rather than falling, so the KL-space control
    cannot establish source specificity. The forecast can: a model that exploits the source has
    ``feat_loss < base_loss < feat_loss_shuffled``.
    """
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    assert torch.isfinite(metrics["feat_loss_shuffled"])
    expected = float(metrics["feat_loss_shuffled"]) - float(metrics["feat_loss"])
    assert float(metrics["shuffle_penalty"]) == pytest.approx(expected, rel=1e-5)


def test_a_readout_only_control_builds_no_autograd_graph(task, stub_batch, prod_kwargs, perturb_posterior):
    """With ``lambda_perm=0`` the control must not carry gradient into the backward."""
    module = task(model_kwargs=dict(prod_kwargs, lambda_perm=0.0))
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert not metrics["kld_shuffled"].requires_grad
    assert not metrics["feat_loss_shuffled"].requires_grad


def test_the_permutation_generator_is_seeded_per_rank_after_attach(task, monkeypatch):
    """Seeding in ``setup`` rather than ``__init__`` is what makes this true.

    At ``__init__`` the module is unattached, so ``global_rank`` reads 0 on every rank and a
    generator seeded there would be identical everywhere -- while the docstring claimed otherwise.
    Ranks hold different data, so their shuffles should differ; only the *schedule* must not.

    ``global_rank`` is a read-only property that reads the attached trainer, so the second rank is
    faked by shadowing it on the class -- which is what attaching would achieve here.
    """
    rank_zero = task()

    rank_one = task()
    monkeypatch.setattr(type(rank_one), "global_rank", 1, raising=False)
    rank_one._perm_generator = None
    rank_one.setup("fit")

    assert rank_zero._perm_generator.initial_seed() != rank_one._perm_generator.initial_seed()


def test_the_schedule_itself_is_rank_invariant(task, stub_batch):
    """The other half: shuffles may differ across ranks, the schedule may not.

    It is a pure function of ``batch_idx``, which Lightning keeps identical across ranks. A
    schedule that consulted the rank-seeded generator would diverge, and DDP would deadlock on the
    first step where two ranks disagreed.
    """
    module = task()

    assert [module._should_run_perm(i, 4, "train") for i in range(6)] == [
        True, False, True, False, True, False
    ]
