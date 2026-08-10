r"""The imported permutation control, run against this model and through this task.

``lag_attn_rws.nets.controls`` is duck-typed on ``query_uses_logvar``, ``query_proj``,
``lag_attn``, ``posterior_head``, ``decoder`` and ``geometry`` -- never on the model class -- so it
*should* work here unchanged. Making that a fact rather than a hope matters more for a subclass
than for a rewrite: nothing about a subclass fails loudly, and the task calls the control inside
its own validation branch, so a model missing one of those names would simply stop producing the
specificity readouts and every other column would look healthy.

The property that makes the control readable is the one the target-domain change must not break:
a derangement of the source leaves every target-only quantity **untouched**, so that
$D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}$ compares three forecasts against
one unmoved reference. Here that reference is a feature block rather than a raw trace, and it is
re-scored rather than merely asserted equal.

Every assertion perturbs the posterior first. At initialisation the posterior *is* the prior, so a
deranged source moves nothing and every shuffled readout is $0$ for reasons that have nothing to do
with being correct.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_rws.nets import controls

from .conftest import SEQ_LEN, TINY_KWARGS, make_stub_batch

#: Batch size the control runs at. Four rather than two, so a derangement has more than one shape
#: and a fixed point is a real possibility rather than an arithmetic impossibility.
_BATCH = 4

#: Keys the control leaves as the matched forward's own tensors. The prior, both encoder states and
#: the base forecast are source-free, so a derangement cannot move them -- and a reader who took
#: one of these off the permuted dict would be reading the matched value and reporting it as the
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


def _model(perturb_posterior=None, **overrides) -> SeqVaeLagAttnFs:
    torch.manual_seed(0)
    model = SeqVaeLagAttnFs(**dict(TINY_KWARGS, **overrides)).eval()
    if perturb_posterior is not None:
        perturb_posterior(model)
    return model


def _forward(model, batch):
    torch.manual_seed(0)
    with torch.no_grad():
        return model(batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], -1))


def _features(batch) -> torch.Tensor:
    """The concatenated target stream, as the task hands it to the objective."""
    return torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1)


def _permute(model, out, batch_size: int = _BATCH):
    return controls.perm_forward_outputs(
        model, out, perm_index=controls.make_derangement(batch_size)
    )


# ---------------------------------------------------------------------------------------
# The control itself, against this net
# ---------------------------------------------------------------------------------------
def test_the_control_replaces_exactly_the_keys_it_declares(perturb_posterior):
    """Exactly, in both directions: an unlisted key that moved would be an undeclared control
    output, and a listed key that did not move would be a control that rebuilt nothing."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH, SEQ_LEN))

    permuted = _permute(model, out)

    assert set(permuted) - set(out) == {"perm_index"}
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
    """The mirror image, so the test above cannot pass on a control that rebuilds nothing. The two
    forecast keys are this model's widened ones, which is the only place the control touches
    anything whose shape the target domain changed."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH, SEQ_LEN))

    permuted = _permute(model, out)

    for key in ("mu_post", "logvar_post", "z_post", "attn_weights", "mu_full", "logvar_full"):
        assert not torch.equal(permuted[key], out[key]), f"{key} was not rebuilt"
    assert permuted["mu_full"].shape[-1] == model.decoder_out_channels


def test_re_scoring_the_permuted_dict_reproduces_the_base_score_exactly(perturb_posterior):
    """The consequence the acceptance ordering rests on: the "no source" reference has not moved,
    checked by re-scoring the feature block rather than by identity alone."""
    model = _model(perturb_posterior)
    batch = make_stub_batch(_BATCH, SEQ_LEN)
    out = _forward(model, batch)

    permuted = _permute(model, out)
    true = model.compute_loss(out, _features(batch), weight=batch.weight)["metrics"]
    shuffled = model.compute_loss(permuted, _features(batch), weight=batch.weight)["metrics"]

    assert torch.equal(true["nll_base_block"], shuffled["nll_base_block"])


def test_scoring_the_permuted_dict_yields_a_different_nonzero_kl(perturb_posterior):
    """``compute_loss`` recomputes the KL from the distribution parameters, so the permuted
    posterior yields a genuinely different KL against the same prior -- which is what makes the
    shuffled readout a measurement rather than a copy of the matched one."""
    model = _model(perturb_posterior)
    batch = make_stub_batch(_BATCH, SEQ_LEN)
    out = _forward(model, batch)

    permuted = _permute(model, out)
    true = model.compute_loss(out, _features(batch), weight=batch.weight)["metrics"]
    shuffled = model.compute_loss(permuted, _features(batch), weight=batch.weight)["metrics"]

    assert float(shuffled["source_conditioned_kl_raw"]) > 0.0
    assert float(shuffled["source_conditioned_kl_raw"]) != pytest.approx(
        float(true["source_conditioned_kl_raw"]), rel=1e-6
    )


def test_a_degenerate_batch_cannot_be_deranged(perturb_posterior):
    """$B < 2$ has no derangement, so the control refuses rather than pairing a sample with itself
    and reporting the result as a stranger's source."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(1, SEQ_LEN))

    with pytest.raises(ValueError, match="batch_size >= 2"):
        controls.perm_forward_outputs(model, out)


def test_the_control_uses_a_real_derangement(perturb_posterior):
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH, SEQ_LEN))

    permuted = controls.perm_forward_outputs(model, out)

    assert not bool((permuted["perm_index"] == torch.arange(_BATCH)).any())


# ---------------------------------------------------------------------------------------
# The control as the task runs it
# ---------------------------------------------------------------------------------------
def test_the_three_readouts_appear_on_validation_only(task, perturb_posterior):
    """Absent, never zero-filled, on the steps that did not run it: the framework aggregates a
    metric as the mean over the steps that reported it, so a zero placeholder would scale the
    epoch value down and invert the ordering the control exists to check."""
    module = task()
    perturb_posterior(module.orig_model)
    batch = make_stub_batch(_BATCH, SEQ_LEN)

    _, train_metrics = module.compute_loss_and_metrics(batch, 0, "train")
    _, val_metrics = module.compute_loss_and_metrics(batch, 0, "val")

    control = {"nll_shuffled_block", "kld_shuffled", "shuffle_penalty"}
    assert control & set(train_metrics) == set()
    assert control <= set(val_metrics)


def test_the_control_is_skipped_on_a_degenerate_validation_batch(task, perturb_posterior):
    """A rank's last uneven batch can be a single sample. The step must still produce a loss."""
    module = task()
    perturb_posterior(module.orig_model)

    loss, metrics = module.compute_loss_and_metrics(make_stub_batch(1, SEQ_LEN), 0, "val")

    assert torch.isfinite(loss)
    assert "nll_shuffled_block" not in metrics


def test_the_shuffled_penalty_is_the_gap_against_the_matched_full_score(
    task, perturb_posterior
):
    """The reported penalty must be a difference of two scores of the *same* target block, or the
    negative control measures the target domain rather than the source pathway."""
    module = task()
    perturb_posterior(module.orig_model)
    batch = make_stub_batch(_BATCH, SEQ_LEN)

    _, metrics = module.compute_loss_and_metrics(batch, 0, "val")

    assert float(metrics["shuffle_penalty"]) == pytest.approx(
        float(metrics["nll_shuffled_block"]) - float(metrics["nll_full_block"]), rel=1e-5
    )


def test_the_control_leaves_the_prior_and_base_branch_bitwise_unchanged(task, perturb_posterior):
    """Through the task, on the same batch under the same seed, with and without the control.

    The control runs on validation steps only, so the training step is the version of the same
    computation in which it did not run. Every target-only readout must be **bitwise** identical
    across the two: a control that leaked into the matched forward -- by re-running it, by
    consuming a draw from the shared generator, or by scoring the permuted dict in place -- would
    move exactly these numbers, and by an amount no tolerance would flag.
    """
    module = task()
    perturb_posterior(module.orig_model)
    batch = make_stub_batch(_BATCH, SEQ_LEN)

    torch.manual_seed(11)
    _, without_control = module.compute_loss_and_metrics(batch, 0, "train")
    torch.manual_seed(11)
    _, with_control = module.compute_loss_and_metrics(batch, 0, "val")

    assert "nll_shuffled_block" in with_control and "nll_shuffled_block" not in without_control
    for name in (
        "nll_base_block", "nll_base_sample", "prior_rate", "mean_logvar_prior",
        "source_conditioned_kl_raw", "mu_post_prior_gap_rms",
    ):
        assert torch.equal(with_control[name], without_control[name]), name
