r"""The validation-only permutation control: source-free branches untouched, no rank alone.

The control's scientific job is the acceptance ordering
$D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}$, and the property that makes
that ordering readable is pinned first: a derangement of the source must leave every
target-only quantity -- the prior, the base forecast, the base score -- bitwise unchanged, or
the "no source" reference has quietly moved.

**Every KL assertion perturbs the posterior first.** At init the posterior *is* the prior, so a
deranged source moves nothing and every shuffled readout is 0 for reasons that have nothing to
do with being correct.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_rws.nets import controls
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.tests.conftest import TINY_KWARGS, make_stub_batch

_BATCH = 4


def _model(perturb_posterior=None, **overrides) -> SeqVaeLagAttnRws:
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**dict(TINY_KWARGS, **overrides)).eval()
    if perturb_posterior is not None:
        perturb_posterior(model)
    return model


def _forward(model, batch):
    torch.manual_seed(0)
    with torch.no_grad():
        return model(batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], -1))


# --------------------------------------------------------------------------------------
# The rebuild itself
# --------------------------------------------------------------------------------------
def test_the_base_branch_and_prior_are_bitwise_identical_under_permutation(perturb_posterior):
    """The source-free reference must not move -- and not only by object identity: re-scoring
    the permuted dict must reproduce the base score exactly."""
    model = _model(perturb_posterior)
    batch = make_stub_batch(_BATCH)
    out = _forward(model, batch)

    permuted = controls.perm_forward_outputs(
        model, out, perm_index=controls.make_derangement(_BATCH)
    )

    for key in ("mu_prior", "logvar_prior", "mu_base", "logvar_base", "target_state"):
        assert torch.equal(permuted[key], out[key]), f"{key} moved under permutation"

    true = model.compute_loss(out, batch.fhr, weight=batch.weight)["metrics"]
    shuffled = model.compute_loss(permuted, batch.fhr, weight=batch.weight)["metrics"]
    assert torch.equal(true["nll_base_block"], shuffled["nll_base_block"])


def test_the_source_driven_tensors_are_genuinely_rebuilt(perturb_posterior):
    """The mirror image, so the test above cannot pass on a control that rebuilds nothing."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH))

    permuted = controls.perm_forward_outputs(
        model, out, perm_index=controls.make_derangement(_BATCH)
    )

    for key in ("mu_post", "z_post", "mu_full"):
        assert not torch.equal(permuted[key], out[key]), f"{key} was not rebuilt"


def test_scoring_the_permuted_dict_yields_the_shuffled_kl(perturb_posterior):
    """``compute_loss`` recomputes the KL from the distribution parameters, so the permuted
    posterior yields a genuinely different (and nonzero) KL against the same prior."""
    model = _model(perturb_posterior)
    batch = make_stub_batch(_BATCH)
    out = _forward(model, batch)

    permuted = controls.perm_forward_outputs(
        model, out, perm_index=controls.make_derangement(_BATCH)
    )
    true = model.compute_loss(out, batch.fhr, weight=batch.weight)["metrics"]
    shuffled = model.compute_loss(permuted, batch.fhr, weight=batch.weight)["metrics"]

    assert float(shuffled["source_conditioned_kl_raw"]) > 0.0
    assert float(shuffled["source_conditioned_kl_raw"]) != pytest.approx(
        float(true["source_conditioned_kl_raw"]), rel=1e-6
    )


def test_the_control_rebuilds_the_query_under_query_uses_logvar(perturb_posterior):
    """``query_uses_logvar`` sizes ``query_proj`` at ``2*d_z`` and the main forward feeds it
    ``[mu^p || logvar^p]``. The control must rebuild the query the same way; a ``mu^p``-only query
    is ``d_z``-wide and would raise a shape mismatch in the projection for that whole sweep arm."""
    model = _model(perturb_posterior, query_uses_logvar=True)
    out = _forward(model, make_stub_batch(_BATCH))

    permuted = controls.perm_forward_outputs(
        model, out, perm_index=controls.make_derangement(_BATCH)
    )

    # It ran without a width mismatch, and the source-driven tensors were genuinely rebuilt.
    for key in ("mu_post", "z_post", "mu_full"):
        assert not torch.equal(permuted[key], out[key]), f"{key} was not rebuilt"


def test_the_control_uses_a_real_derangement(perturb_posterior):
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH))

    permuted = controls.perm_forward_outputs(model, out)

    perm = permuted["perm_index"]
    assert not bool((perm == torch.arange(_BATCH)).any()), "the derangement has a fixed point"


def test_a_degenerate_batch_cannot_be_deranged(perturb_posterior):
    """B < 2 has no derangement; the *task's* schedule is what must keep it away from here."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(1))

    with pytest.raises(ValueError, match="batch_size >= 2"):
        controls.perm_forward_outputs(model, out)


# --------------------------------------------------------------------------------------
# The task wiring
# --------------------------------------------------------------------------------------
_CONTROL_KEYS = {"nll_shuffled_block", "kld_shuffled", "shuffle_penalty"}


def test_the_control_never_runs_on_a_training_batch(task, stub_batch, perturb_posterior):
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert _CONTROL_KEYS.isdisjoint(metrics)


def test_validation_runs_it_and_it_contributes_nothing_to_the_loss(
    task, stub_batch, perturb_posterior
):
    """The returned validation loss must be exactly the perm-free objective: a control that
    leaked into ``val/total_loss`` would steer the ``ModelCheckpoint`` selection."""
    module = task()
    perturb_posterior(module.orig_model)

    loss, metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    assert _CONTROL_KEYS <= set(metrics)
    assert float(metrics["kld_shuffled"]) > 0.0
    assert float(loss) == pytest.approx(float(metrics["main_loss"]), rel=1e-6)


def test_shuffle_penalty_is_the_shuffled_minus_matched_score(task, stub_batch, perturb_posterior):
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    expected = float(metrics["nll_shuffled_block"]) - float(metrics["nll_full_block"])
    assert float(metrics["shuffle_penalty"]) == pytest.approx(expected, rel=1e-5)


def test_the_shuffled_readouts_build_no_autograd_graph(task, stub_batch, perturb_posterior):
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    for key in _CONTROL_KEYS:
        assert not metrics[key].requires_grad, key


def test_a_degenerate_validation_batch_skips_with_metrics_absent_not_zero(
    task, make_stub_batch_fn, perturb_posterior
):
    """Zeros would be wrong numbers, not placeholders: the epoch aggregate is the mean over the
    steps that reported a metric, so zero-filling skipped steps scales the shuffled readouts
    toward nothing and inverts the ordering the control exists to check."""
    module = task()
    perturb_posterior(module.orig_model)

    loss, metrics = module.compute_loss_and_metrics(make_stub_batch_fn(batch_size=1), 0, "val")

    assert _CONTROL_KEYS.isdisjoint(metrics)
    assert torch.isfinite(loss)


def test_a_false_on_any_rank_wins_the_collective_decision(
    task, stub_batch, perturb_posterior, monkeypatch
):
    """The MIN reduce: a rank whose last uneven batch is degenerate cannot be outvoted, or the
    ranks that ran the control would hang the ``sync_dist`` on its metrics. Simulated by a fake
    collective whose peer always answers 'cannot', which must silence this rank too."""
    import teb_vae.lag_attn_rws.task as task_module

    module = task()
    perturb_posterior(module.orig_model)
    reduce_ops = []

    monkeypatch.setattr(task_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(task_module.dist, "is_initialized", lambda: True)

    def _peer_says_no(tensor, op=None):
        reduce_ops.append(op)
        tensor.copy_(torch.minimum(tensor, torch.zeros_like(tensor)))  # peer's flag is 0

    monkeypatch.setattr(task_module.dist, "all_reduce", _peer_says_no)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    assert _CONTROL_KEYS.isdisjoint(metrics), "a rank ran the control alone"
    assert reduce_ops == [task_module.dist.ReduceOp.MIN], (
        "the decision must be MIN-reduced exactly once: run only if every rank can"
    )


def test_a_true_from_every_rank_keeps_the_control_running(
    task, stub_batch, perturb_posterior, monkeypatch
):
    """The mirror image: the fake collective must be able to pass a decision through, or the
    test above passes on a wiring that simply never runs the control under DDP."""
    import teb_vae.lag_attn_rws.task as task_module

    module = task()
    perturb_posterior(module.orig_model)

    monkeypatch.setattr(task_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(task_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(task_module.dist, "all_reduce", lambda tensor, op=None: None)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    assert _CONTROL_KEYS <= set(metrics)
