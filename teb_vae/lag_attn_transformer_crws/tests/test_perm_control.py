r"""The imported permutation control, run against this architecture at a tiled anchor set.

``nll_shuffled_block``, ``kld_shuffled`` and ``shuffle_penalty`` are tracked and **validation-only**,
so without this file their first execution would be on the production box after a full training
epoch. That is not a hypothetical: the control decodes ``z_post_perm`` at the anchors the matched
forward used, and until that argument was threaded through the shared call site it decoded the
contiguous prefix $[0, T_{\mathrm{valid}})$ instead -- a shape error rather than a wrong number, and
one that arrives only on a validation step.

The control's own semantics -- what it replaces, what it leaves as the matched forward's objects,
the derangement, the degenerate-batch refusal -- are the shared control's and are pinned by the
conv-LSTM cell of this row and the architecture parent. What is re-earned here is the combination:
this encoder's source pathway, re-encoded under a derangement, decoded at a tiled anchor set, and
scored through this package's task.

Every assertion perturbs the posterior first. At initialisation the posterior *is* the prior, so a
deranged source moves nothing and every shuffled readout is $0$ for reasons that have nothing to do
with being correct.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_rws.nets import controls
from teb_vae.lag_attn_transformer_crws.nets.model import SeqVaeLagAttnTrfCrws

from .conftest import TINY_STRIDE, make_stub_batch, tiny_warmup_kwargs

#: Batch size the control runs at. Four rather than two, so a derangement has more than one shape
#: and a fixed point is a real possibility rather than an arithmetic impossibility.
_BATCH = 4


def _model(perturb_posterior) -> SeqVaeLagAttnTrfCrws:
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfCrws(**tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)).eval()
    perturb_posterior(model)
    return model


def _forward(model, batch, phase=0):
    """One matched forward at the **training** tiling, which is where the shapes differ."""
    torch.manual_seed(0)
    with torch.no_grad():
        return model(
            batch.fhr_st,
            batch.fhr_ph,
            torch.cat([batch.up_st, batch.up_ph], -1),
            phase,
            TINY_STRIDE,
        )


def test_the_permuted_decode_has_the_matched_shape_only_when_given_the_anchors(
    perturb_posterior,
) -> None:
    """The failure a real fit found on the conv-LSTM cells, re-earned here: the control must decode
    at the tile, not at the dense prefix. Asserted as a shape rather than through ``torch.equal``,
    which returns ``False`` on a mismatch and would read as an ordinary inequality.

    The negative control is the second half: without the anchor argument the control decodes the
    dense prefix, which is a *different* shape at a tiling -- so passing it is load-bearing rather
    than tidy.
    """
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH))

    permuted = controls.perm_forward_outputs(
        model,
        out,
        generator=torch.Generator().manual_seed(0),
        anchors=out["anchor_index"],
    )
    dense = controls.perm_forward_outputs(
        model, out, generator=torch.Generator().manual_seed(0)
    )

    for key in ("mu_full", "logvar_full"):
        assert permuted[key].shape == out[key].shape, key
    assert permuted["mu_full"].shape[1] == out["anchor_index"].shape[1]
    assert dense["mu_full"].shape != out["mu_full"].shape
    assert dense["mu_full"].shape[1] == model.geometry.t_valid


def test_the_control_runs_on_a_real_validation_step_and_scores_the_matched_window(
    task, perturb_posterior
) -> None:
    """The end-to-end version, and the one whose absence cost the conv-LSTM cells their first
    validation epoch: the task's own step resolves the dense set on ``val``, and the control has to
    decode at it without a shape error. The reported penalty must then be a difference of two
    scores of the *same* raw window at the *same* anchors, or the negative control measures the
    geometry rather than the source pathway."""
    module = task()
    perturb_posterior(module.orig_model)

    _loss, metrics = module.compute_loss_and_metrics(make_stub_batch(_BATCH), 0, "val")

    assert torch.isfinite(metrics["shuffle_penalty"]).all()
    assert float(metrics["kld_shuffled"]) != 0.0
    assert float(metrics["shuffle_penalty"]) == pytest.approx(
        float(metrics["nll_shuffled_block"]) - float(metrics["nll_full_block"]), rel=1e-5
    )
