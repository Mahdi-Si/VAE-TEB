r"""The imported permutation control, run against this model.

``lag_attn_rws.nets.controls`` is duck-typed on ``query_uses_logvar``, ``query_proj``, ``lag_attn``,
``posterior_head``, ``decoder`` and ``geometry`` -- never on the model class -- so it *should* work
here unchanged. This file makes that a fact rather than a hope, because the control is the one place
a renamed attribute would silently disable a validation-time check: the task calls it on every
validation batch, and a model missing one of those names would simply stop producing the
specificity readouts.

The control itself is architecture-agnostic: it operates on the already-computed ``source_state``
and never touches an encoder, let alone a front end. What is *not* architecture-agnostic is the
contract it depends on -- that a derangement of the source leaves every target-only quantity
**untouched**, so that $D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}$ compares
three forecasts against one unmoved reference. That is a property of the assembled model, and here
it is asserted by object identity, which is stronger than equality and is what the control actually
promises.

Every assertion perturbs the posterior first. At init the posterior *is* the prior, so a deranged
source moves nothing and every shuffled readout is $0$ for reasons that have nothing to do with
being correct.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_rws.nets import controls
from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E

from .conftest import BATCH, SEQ_LEN, TINY_KWARGS, make_stub_batch

_BATCH = 4

#: Keys the control leaves as the matched forward's own tensors. The prior, both encoder states and
#: the base forecast are source-free, so a derangement cannot move them -- and a reader who took one
#: of these off the permuted dict would be reading the matched value and reporting it as the
#: control's.
_UNTOUCHED_KEYS = (
    "mu_prior",
    "logvar_prior",
    "raw_logvar_prior",
    "target_state",
    "source_state",
    "mu_base",
    "logvar_base",
)


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


def test_the_control_replaces_exactly_the_keys_it_declares(perturb_posterior):
    """Exactly, in both directions: an unlisted key that moved would be an undeclared control
    output, and a listed key that did not move would be a control that rebuilt nothing."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH, SEQ_LEN))

    permuted = _permute(model, out)

    added = set(permuted) - set(out)
    assert added == {"perm_index"}
    moved = {key for key in out if permuted[key] is not out[key]}
    assert moved == set(controls.RECOMPUTED_KEYS) - {"perm_index"}


def test_the_source_free_tensors_are_the_same_objects(perturb_posterior):
    """Identity, not equality. The control promises to reuse the computed states rather than to
    reproduce them, and equality would also hold for a control that recomputed them and happened
    to agree."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH, SEQ_LEN))

    permuted = _permute(model, out)

    for key in _UNTOUCHED_KEYS:
        assert permuted[key] is out[key], f"{key} was rebuilt under permutation"


def test_the_source_driven_tensors_are_genuinely_rebuilt(perturb_posterior):
    """The mirror image, so the test above cannot pass on a control that rebuilds nothing."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH, SEQ_LEN))

    permuted = _permute(model, out)

    for key in ("mu_post", "logvar_post", "z_post", "attn_weights", "mu_full", "logvar_full"):
        assert not torch.equal(permuted[key], out[key]), f"{key} was not rebuilt"


def test_re_scoring_the_permuted_dict_reproduces_the_base_score_exactly(perturb_posterior):
    """The consequence the acceptance ordering rests on: the "no source" reference has not moved,
    checked by re-scoring rather than by identity alone. Bitwise, because the base branch is a
    function of the prior alone and the shared epsilon is already fixed in the dict."""
    model = _model(perturb_posterior)
    batch = make_stub_batch(_BATCH, SEQ_LEN)
    out = _forward(model, batch)

    permuted = _permute(model, out)
    true = model.compute_loss(out, batch.fhr, weight=batch.weight)["metrics"]
    shuffled = model.compute_loss(permuted, batch.fhr, weight=batch.weight)["metrics"]

    assert torch.equal(true["nll_base_block"], shuffled["nll_base_block"])


def test_scoring_the_permuted_dict_yields_a_different_nonzero_kl(perturb_posterior):
    """``compute_loss`` recomputes the KL from the distribution parameters, so the permuted
    posterior yields a genuinely different KL against the same prior -- which is what makes the
    shuffled readout a measurement rather than a copy of the matched one."""
    model = _model(perturb_posterior)
    batch = make_stub_batch(_BATCH, SEQ_LEN)
    out = _forward(model, batch)

    permuted = _permute(model, out)
    true = model.compute_loss(out, batch.fhr, weight=batch.weight)["metrics"]
    shuffled = model.compute_loss(permuted, batch.fhr, weight=batch.weight)["metrics"]

    assert float(shuffled["source_conditioned_kl_raw"]) > 0.0
    assert float(shuffled["source_conditioned_kl_raw"]) != pytest.approx(
        float(true["source_conditioned_kl_raw"]), rel=1e-6
    )


def test_the_control_rebuilds_the_query_under_query_uses_logvar(perturb_posterior):
    """``query_uses_logvar`` sizes ``query_proj`` at $2 d_z$ and the main forward feeds it
    $[\\mu^p \\Vert \\ell^p]$. The control must rebuild the query the same way; a $\\mu^p$-only
    query is $d_z$-wide and would raise a shape mismatch for that whole arm."""
    model = _model(perturb_posterior, query_uses_logvar=True)
    out = _forward(model, make_stub_batch(_BATCH, SEQ_LEN))

    permuted = _permute(model, out)

    for key in ("mu_post", "z_post", "mu_full"):
        assert not torch.equal(permuted[key], out[key]), f"{key} was not rebuilt"


def test_the_control_uses_a_real_derangement(perturb_posterior):
    """Without the fixed-point ban some samples would be paired with their own source, and the
    control would report a mixture of the matched and shuffled scores under the shuffled name."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH, SEQ_LEN))

    permuted = controls.perm_forward_outputs(model, out)

    perm = permuted["perm_index"]
    assert not bool((perm == torch.arange(_BATCH)).any()), "the derangement has a fixed point"


def test_a_degenerate_batch_cannot_be_deranged(perturb_posterior):
    """$B < 2$ has no derangement, so the control refuses rather than pairing a sample with
    itself and reporting the result as a stranger's source."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(1, SEQ_LEN))

    with pytest.raises(ValueError, match="batch_size >= 2"):
        controls.perm_forward_outputs(model, out)


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
