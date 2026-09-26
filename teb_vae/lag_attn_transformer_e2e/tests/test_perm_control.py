r"""The imported permutation control, run against this model.

``lag_attn_rws.nets.controls`` is duck-typed on ``query_uses_logvar``, ``query_proj``, ``lag_attn``,
``posterior_head``, ``decoder`` and ``geometry`` -- never on the model class -- so it *should* work
here unchanged. This file makes that a fact rather than a hope, because the control is the one place
a renamed attribute would silently disable a validation-time check: the task calls it on every
validation batch, and a model missing one of those names would simply stop producing the
specificity readouts.

The control's own contract -- which keys it rebuilds, which it reuses, the derangement and its
refusal of a singleton batch -- is tested where it is defined, in ``lag_attn_rws``. What is tested
here is that it runs on *this* model, including under ``query_uses_logvar``, and that this task
runs it on validation only.

Every assertion perturbs the posterior first. At init the posterior *is* the prior, so a deranged
source moves nothing and every shuffled readout is $0$ for reasons that have nothing to do with
being correct.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_rws.nets import controls
from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E

from .conftest import BATCH, SEQ_LEN, TINY_KWARGS, make_stub_batch

_BATCH = 4


def _model(perturb_posterior=None, **overrides) -> SeqVaeLagAttnTrfE2E:
    """Build the tiny model, optionally with a non-degenerate posterior.

    Args:
        perturb_posterior: The suite's perturbation fixture, or ``None`` to leave the model at its
            zero-KL initialisation.
        **overrides: Constructor keyword overrides on top of :data:`TINY_KWARGS`.

    Returns:
        The model in eval mode, so two forwards of the same input agree.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfE2E(**dict(TINY_KWARGS, **overrides)).eval()
    if perturb_posterior is not None:
        perturb_posterior(model)
    return model


def _forward(model, batch):
    """Run the model on a stub batch's raw fields, seeded so the shared epsilon is reproducible."""
    torch.manual_seed(0)
    with torch.no_grad():
        return model(batch.fhr, batch.up, batch.weight)


def _permute(model, out, batch_size: int = _BATCH):
    """Apply the control at an explicit derangement, so the pairing does not depend on a seed."""
    return controls.perm_forward_outputs(
        model, out, perm_index=controls.make_derangement(batch_size)
    )


def test_the_source_driven_tensors_are_genuinely_rebuilt(perturb_posterior):
    """The control reaches every attribute it needs on this model and actually rebuilds the
    source-conditioned half of the forward."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH, SEQ_LEN))

    permuted = _permute(model, out)

    for key in ("mu_post", "logvar_post", "z_post", "attn_weights", "mu_full", "logvar_full"):
        assert not torch.equal(permuted[key], out[key]), f"{key} was not rebuilt"


def test_the_control_rebuilds_the_query_under_query_uses_logvar(perturb_posterior):
    """``query_uses_logvar`` sizes ``query_proj`` at $2 d_z$ and the main forward feeds it
    $[\\mu^p \\Vert \\ell^p]$. The control must rebuild the query the same way; a $\\mu^p$-only
    query is $d_z$-wide and would raise a shape mismatch for that whole arm."""
    model = _model(perturb_posterior, query_uses_logvar=True)
    out = _forward(model, make_stub_batch(_BATCH, SEQ_LEN))

    permuted = _permute(model, out)

    for key in ("mu_post", "z_post", "mu_full"):
        assert not torch.equal(permuted[key], out[key]), f"{key} was not rebuilt"


def test_the_model_itself_forwards_at_batch_one(perturb_posterior):
    """The refusal above is the *control's*, not the model's: a rank can receive a batch of one
    under DDP and must still train on it. Two front ends and two encoders all have to survive a
    singleton batch for that to hold."""
    model = _model(perturb_posterior)

    out = _forward(model, make_stub_batch(1, SEQ_LEN))

    assert out["mu_prior"].shape[0] == 1
    assert torch.isfinite(out["mu_full"]).all()


def test_the_task_runs_the_control_on_validation_and_not_on_training(task, perturb_posterior):
    """Where the control actually fires. It is validation-only by design -- it is a readout and
    never enters the objective -- and the task reaches it through the same duck-typed function, so
    a name the control needs and this model lacks would surface as three metrics quietly missing
    rather than as an error."""
    module = task()
    perturb_posterior(module.orig_model)
    batch = make_stub_batch(BATCH, SEQ_LEN)

    _, train_metrics = module.compute_loss_and_metrics(batch, 0, "train")
    _, val_metrics = module.compute_loss_and_metrics(batch, 0, "val")

    assert "nll_shuffled_block" not in train_metrics
    assert {"nll_shuffled_block", "kld_shuffled", "shuffle_penalty"} <= set(val_metrics)
    assert float(val_metrics["kld_shuffled"]) > 0.0
