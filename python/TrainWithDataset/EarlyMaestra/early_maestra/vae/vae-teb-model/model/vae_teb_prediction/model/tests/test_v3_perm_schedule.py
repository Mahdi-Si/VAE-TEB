r"""S4-T03a: the fused source-permutation control and its DDP-safe schedule.

The control rides inside the **single main backward**. That is what lets v3 keep Lightning's
automatic optimization -- and with it the gradient clip, gradient accumulation, LR scheduler,
and loss-spike circuit breaker -- while ``_select_ddp_strategy`` still returns plain ``'ddp'``.

Two invariants make that safe and are pinned here:

* On *every* step type, perm and non-perm alike, each ``requires_grad`` parameter receives a
  gradient. With ``find_unused_parameters=False`` the DDP reducer expects exactly that; a
  parameter left ungradiented on some steps would raise or deadlock.
* The schedule is a pure function of ``batch_idx``, which Lightning keeps identical across
  ranks, and is additionally MIN-reduced so a rank with a degenerate (``B < 2``) batch cannot
  branch alone.
"""
from __future__ import annotations

import pytest
import torch


def _perm_active(metrics) -> bool:
    return float(metrics["kld_shuffled"]) != 0.0 or float(metrics["perm_loss"]) != 0.0


def test_schedule_fires_only_on_multiples_of_perm_every_n_batches(v3_pl, stub_batch,
                                                                  perturb_posterior):
    pl_module = v3_pl()
    perturb_posterior(pl_module.orig_model)  # else every KL is 0 and the check is vacuous
    assert pl_module.orig_model.perm_every_n_batches == 2

    fired = []
    for batch_idx in range(6):
        _, metrics = pl_module.compute_loss_and_metrics(stub_batch, batch_idx, "train")
        fired.append(_perm_active(metrics))
    assert fired == [True, False, True, False, True, False]


def test_perm_loss_enters_the_total_only_on_scheduled_steps(v3_pl, stub_batch,
                                                            perturb_posterior):
    pl_module = v3_pl()
    perturb_posterior(pl_module.orig_model)
    lambda_perm = float(pl_module.orig_model.lambda_perm)
    assert lambda_perm > 0.0

    total_on, m_on = pl_module.compute_loss_and_metrics(stub_batch, 0, "train")
    total_off, m_off = pl_module.compute_loss_and_metrics(stub_batch, 1, "train")

    # On-schedule: total = main + lambda_perm * L_perm, and L_perm is genuinely nonzero.
    assert float(m_on["perm_loss"]) > 0.0
    assert float(m_on["kld_shuffled"]) > 0.0
    expected = float(m_on["main_loss"]) + float(m_on["perm_loss"])
    assert float(total_on) == pytest.approx(expected, rel=1e-6)

    # Off-schedule: the control contributes nothing.
    assert float(m_off["perm_loss"]) == 0.0
    assert float(total_off) == pytest.approx(float(m_off["main_loss"]), rel=1e-6)


def test_kld_shuffled_ratio_is_reported(v3_pl, stub_batch, perturb_posterior):
    r"""The G6 readout :math:`K_{\mathrm{shuffled}} / K_{\mathrm{raw}}`."""
    pl_module = v3_pl()
    perturb_posterior(pl_module.orig_model)
    _, metrics = pl_module.compute_loss_and_metrics(stub_batch, 0, "train")

    ratio = float(metrics["kld_shuffled"]) / float(metrics["kld_raw"])
    assert float(metrics["kld_shuffled_ratio"]) == pytest.approx(ratio, rel=1e-4)


@pytest.mark.parametrize("batch_idx", [0, 1])
def test_every_parameter_receives_a_gradient(v3_pl, stub_batch, perturb_posterior, batch_idx):
    """This -- not the strategy string -- is what licenses find_unused_parameters=False."""
    pl_module = v3_pl()
    perturb_posterior(pl_module.orig_model)

    pl_module.zero_grad(set_to_none=True)
    loss, _ = pl_module.compute_loss_and_metrics(stub_batch, batch_idx, "train")
    loss.backward()

    starved = [
        name for name, p in pl_module.orig_model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not starved, f"parameters without gradient on batch_idx={batch_idx}: {starved}"


def test_a_single_backward_is_enough(v3_pl, stub_batch, perturb_posterior):
    """A parameter used twice in one graph must not need a second backward pass.

    The permutation branch re-runs ``lag_attn`` and ``posterior_head``. Both usages share one
    ``AccumulateGrad`` node, so a single ``.backward()`` accumulates them and DDP marks each
    parameter ready exactly once.
    """
    pl_module = v3_pl()
    perturb_posterior(pl_module.orig_model)
    attn_param = pl_module.orig_model.lag_attn.lag_embeddings

    pl_module.zero_grad(set_to_none=True)
    loss_off, _ = pl_module.compute_loss_and_metrics(stub_batch, 1, "train")
    loss_off.backward()
    grad_without_perm = attn_param.grad.detach().clone()

    pl_module.zero_grad(set_to_none=True)
    loss_on, _ = pl_module.compute_loss_and_metrics(stub_batch, 0, "train")
    loss_on.backward()  # exactly one backward, even though lag_attn ran twice
    grad_with_perm = attn_param.grad.detach().clone()

    assert torch.isfinite(grad_with_perm).all()
    assert not torch.allclose(grad_with_perm, grad_without_perm), (
        "the permutation branch contributed no gradient to the shared attention parameters"
    )


def test_validation_runs_the_control_every_step_without_adding_to_the_loss(
    v3_pl, stub_batch, perturb_posterior
):
    r"""``val/kld_shuffled`` vs ``val/kld_raw`` is the headline G6 readout, so it must be
    measured on every validation batch -- but it must never enter the validation loss."""
    pl_module = v3_pl()
    perturb_posterior(pl_module.orig_model)

    for batch_idx in (0, 1, 3):
        loss, metrics = pl_module.compute_loss_and_metrics(stub_batch, batch_idx, "val")
        assert float(metrics["kld_shuffled"]) > 0.0, "control skipped on a validation batch"
        assert float(metrics["perm_loss"]) == 0.0
        assert float(loss) == pytest.approx(float(metrics["main_loss"]), rel=1e-6)


def test_degenerate_batch_skips_the_control(v3_pl, make_stub_batch_fn, perturb_posterior):
    """A single-sample batch cannot be deranged; the control must be skipped, not crash."""
    pl_module = v3_pl()
    perturb_posterior(pl_module.orig_model)
    single = make_stub_batch_fn(batch_size=1)

    loss, metrics = pl_module.compute_loss_and_metrics(single, 0, "train")
    assert float(metrics["perm_loss"]) == 0.0
    assert float(metrics["kld_shuffled"]) == 0.0
    assert torch.isfinite(loss)


def test_lambda_perm_zero_keeps_the_readout_but_leaves_the_loss_alone(
    v3_pl, stub_batch, prod_kwargs, perturb_posterior
):
    """The production default. G6 ships as a diagnostic, not an objective."""
    pl_module = v3_pl(model_kwargs=dict(prod_kwargs, lambda_perm=0.0))
    perturb_posterior(pl_module.orig_model)

    total, metrics = pl_module.compute_loss_and_metrics(stub_batch, 0, "train")
    assert float(metrics["perm_loss"]) == 0.0
    assert float(total) == pytest.approx(float(metrics["main_loss"]), rel=1e-6)
    # ...but the readout is still measured and logged.
    assert float(metrics["kld_shuffled"]) > 0.0
    assert float(metrics["feat_loss_shuffled"]) > 0.0


def test_prediction_space_control_is_reported(v3_pl, stub_batch, perturb_posterior):
    r"""``feat_loss_shuffled`` is the control that actually discriminates.

    :math:`\mathrm{KL}(q\|p)` rises under a deranged source rather than falling, so the KL-space
    control cannot establish source specificity. The forecast can: a model that exploits the
    source has ``feat_loss < base_loss < feat_loss_shuffled``.
    """
    pl_module = v3_pl()
    perturb_posterior(pl_module.orig_model)
    _, metrics = pl_module.compute_loss_and_metrics(stub_batch, 0, "val")

    assert torch.isfinite(metrics["feat_loss_shuffled"])
    expected = float(metrics["feat_loss_shuffled"]) - float(metrics["feat_loss"])
    assert float(metrics["shuffle_penalty"]) == pytest.approx(expected, rel=1e-5)


def test_readout_only_control_builds_no_autograd_graph(v3_pl, stub_batch, perturb_posterior,
                                                       prod_kwargs):
    """With ``lambda_perm=0`` the control must not carry gradient into the backward."""
    pl_module = v3_pl(model_kwargs=dict(prod_kwargs, lambda_perm=0.0))
    perturb_posterior(pl_module.orig_model)

    _, metrics = pl_module.compute_loss_and_metrics(stub_batch, 0, "train")
    assert not metrics["kld_shuffled"].requires_grad
    assert not metrics["feat_loss_shuffled"].requires_grad
