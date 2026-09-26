r"""The imported permutation control, run against a model that decodes a tiled anchor set.

The control's own mechanics -- the derangement, the recomputed key set, identity of the source-free
tensors, the floored lag mask, the degenerate-batch refusal -- belong to ``lag_attn_rws`` and are
pinned there and, for an anchored model, in ``lag_attn_cfs``. What is this cell's is the raw target
gathered at the control's anchors and the task that runs it, so what is checked here is:

* re-scoring the permuted dict through this model's ``compute_loss`` reproduces the matched base
  score bit for bit -- the anchored gather reads ``anchor_index``, so a control that touched the
  anchors would move this number;
* on a validation step the three control readouts appear (and on a training step they do not),
  separate from the matched ones after perturbation, and ``shuffle_penalty`` is the gap against the
  matched full score;
* running the control leaves every target-only readout of the step bitwise unchanged.

Every assertion perturbs the posterior first. At initialisation the posterior *is* the prior, so a
deranged source moves nothing and every shuffled readout would hold vacuously.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_rws.nets import controls

from .conftest import TINY_SEQ_LEN, TINY_STRIDE, make_stub_batch, tiny_warmup_kwargs

#: Batch size the control runs at: more than two, so a derangement has more than one shape.
_BATCH = 4

#: The three validation-only readouts the control emits.
_CONTROL_KEYS = {"nll_shuffled_block", "kld_shuffled", "shuffle_penalty"}


def _model(perturb_posterior) -> SeqVaeLagAttnCrws:
    torch.manual_seed(0)
    model = SeqVaeLagAttnCrws(**tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)).eval()
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


def test_re_scoring_the_permuted_dict_reproduces_the_base_score_bit_exactly(perturb_posterior):
    """The consequence the acceptance ordering rests on: the "no source" reference has not moved,
    checked by re-scoring the raw window at the same anchors rather than by identity alone. It is
    the anchored gather that makes this a claim rather than a tautology -- the target is built from
    ``anchor_index``, so a control that touched the anchors would move this number too."""
    model = _model(perturb_posterior)
    batch = make_stub_batch(_BATCH, TINY_SEQ_LEN)
    out = _forward(model, batch)

    permuted = controls.perm_forward_outputs(
        model,
        out,
        perm_index=controls.make_derangement(_BATCH),
        anchors=out["anchor_index"],
    )
    true = model.compute_loss(out, batch.fhr, weight=batch.weight)["metrics"]
    shuffled = model.compute_loss(permuted, batch.fhr, weight=batch.weight)["metrics"]

    assert permuted["mu_full"].shape == out["mu_full"].shape
    assert torch.equal(true["nll_base_block"], shuffled["nll_base_block"])


def test_the_control_runs_on_validation_only_and_separates_from_the_matched_readouts(
    task, perturb_posterior
):
    """Absent, never zero-filled, on a training step: the framework aggregates a metric as the mean
    over the steps that reported it. On a validation step a stranger's source must give a different
    posterior *and* a different forecast -- a control that rebuilt the KL and not the decode would
    satisfy one half -- and the penalty must be a difference of two scores of the same raw window at
    the same anchors."""
    module = task()
    perturb_posterior(module.orig_model)
    batch = make_stub_batch(_BATCH, TINY_SEQ_LEN)

    _, train_metrics = module.compute_loss_and_metrics(batch, 0, "train")
    _, metrics = module.compute_loss_and_metrics(batch, 0, "val")

    assert _CONTROL_KEYS & set(train_metrics) == set()
    assert _CONTROL_KEYS <= set(metrics)
    assert float(metrics["kld_shuffled"]) != pytest.approx(
        float(metrics["source_conditioned_kl_raw"]), rel=1e-6
    )
    assert float(metrics["nll_shuffled_block"]) != pytest.approx(
        float(metrics["nll_full_block"]), rel=1e-6
    )
    assert float(metrics["shuffle_penalty"]) == pytest.approx(
        float(metrics["nll_shuffled_block"]) - float(metrics["nll_full_block"]), rel=1e-5
    )


def test_the_control_leaves_the_prior_and_base_branch_bitwise_unchanged(task, perturb_posterior):
    """On the same batch under the same seed, with and without the control.

    Compared across two **validation** steps rather than against a training step: this cell decodes
    a different anchor set on the two stages. Every target-only readout must be **bitwise**
    identical: a control that leaked into the matched forward -- by re-running it, by consuming a
    draw from the shared generator, or by scoring the permuted dict in place -- would move exactly
    these numbers, by an amount no tolerance would flag.
    """
    module = task()
    perturb_posterior(module.orig_model)
    batch = make_stub_batch(_BATCH, TINY_SEQ_LEN)

    torch.manual_seed(11)
    _, with_control = module.compute_loss_and_metrics(batch, 0, "val")
    module._should_run_perm = lambda batch_size, stage: False
    torch.manual_seed(11)
    _, without_control = module.compute_loss_and_metrics(batch, 0, "val")

    assert "nll_shuffled_block" in with_control and "nll_shuffled_block" not in without_control
    for name in (
        "nll_base_block", "nll_base_sample", "prior_rate", "mean_logvar_prior",
        "source_conditioned_kl_raw", "mu_post_prior_gap_rms", "anchors_per_sample",
        "source_lag_warmth_frac_st", "source_lag_warmth_frac_ph", "kld_source_null",
    ):
        assert torch.equal(with_control[name], without_control[name]), name
