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

The derangement, the source-free reuse, the shuffled KL, the degenerate-batch refusals and the
task's validation-only wiring are pinned against the raw sibling in ``lag_attn_rws``. What is
checked here is what this model could break: that the control rebuilds this model's widened
forecast, that re-scoring the permuted dict against the feature block leaves the base score
bitwise unmoved, and that the three readouts still appear when this task validates.

Every assertion perturbs the posterior first. At initialisation the posterior *is* the prior, so a
deranged source moves nothing and every shuffled readout is $0$ for reasons that have nothing to do
with being correct.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_rws.nets import controls

from .conftest import SEQ_LEN, TINY_KWARGS, make_stub_batch

#: Batch size the control runs at. Four rather than two, so a derangement has more than one shape
#: and a fixed point is a real possibility rather than an arithmetic impossibility.
_BATCH = 4


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
