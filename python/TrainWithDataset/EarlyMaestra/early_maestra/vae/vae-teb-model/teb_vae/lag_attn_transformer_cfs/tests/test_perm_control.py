r"""The imported permutation control, run against this architecture at a tiled anchor set.

``nll_shuffled_block``, ``kld_shuffled`` and ``shuffle_penalty`` are tracked and **validation-only**,
so without this file their first execution would be on the production box after a full training
epoch. That is not a hypothetical: the control decodes ``z_post_perm`` at the anchors the matched
forward used, and until that argument was threaded through the shared call site it decoded the
contiguous prefix $[0, T_{\mathrm{valid}})$ instead -- a shape error rather than a wrong number, and
one that arrives only on a validation step.

The first assertion below is the shape, explicitly, and that is deliberate: ``torch.equal`` returns
``False`` on a shape mismatch, so a test written the way the two-sided siblings' are would pass on a
wrong-shaped control.

Every assertion perturbs the posterior first. At initialisation the posterior *is* the prior, so a
deranged source moves nothing and every shuffled readout is $0$ for reasons that have nothing to do
with being correct.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_rws.nets import controls
from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs

from .conftest import TINY_STRIDE, make_stub_batch, tiny_warmup_kwargs

#: Batch size the control runs at. Four rather than two, so a derangement has more than one shape
#: and a fixed point is a real possibility rather than an arithmetic impossibility.
_BATCH = 4

#: Keys the control leaves as the matched forward's own tensors. The prior, both encoder states and
#: the base forecast are source-free, so a derangement cannot move them -- and the two anchor keys
#: are a property of the *geometry*, not of the pairing, so a control that rebuilt them would be
#: scoring the shuffled forecast against a different anchor set than the matched one.
_UNTOUCHED_KEYS = (
    "mu_prior",
    "logvar_prior",
    "raw_logvar_prior",
    "target_state",
    "source_state",
    "mu_base",
    "logvar_base",
    "anchor_index",
    "anchor_valid",
)


def _model(perturb_posterior=None, **overrides) -> SeqVaeLagAttnTrfCfs:
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfCfs(
        **tiny_warmup_kwargs(anchor_stride=TINY_STRIDE, **overrides)
    ).eval()
    if perturb_posterior is not None:
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


def _permuted(model, out, generator=None):
    """Run the shared control at the matched forward's own anchors."""
    return controls.perm_forward_outputs(
        model,
        out,
        generator=generator or torch.Generator().manual_seed(0),
        anchors=out["anchor_index"],
    )


# =================================================================================================
# The shape, explicitly
# =================================================================================================
def test_the_permuted_decode_has_the_same_shape_as_the_matched_one(perturb_posterior):
    """The failure a real fit found on the conv-LSTM cell, re-earned here: the control must decode
    at the tile, not at the dense prefix. Asserted as a shape rather than through ``torch.equal``,
    which returns ``False`` on a mismatch and would read as an ordinary inequality."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH))

    permuted = _permuted(model, out)

    for key in ("mu_full", "logvar_full"):
        assert permuted[key].shape == out[key].shape, key
    assert permuted["mu_full"].shape[1] == out["anchor_index"].shape[1]


def test_a_control_given_no_anchors_would_decode_a_different_shape(perturb_posterior):
    """The negative control on the test above: without the anchor argument the control decodes the
    dense prefix, which is a *different* shape here -- so passing it is load-bearing rather than
    tidy."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH))

    dense = controls.perm_forward_outputs(
        model, out, generator=torch.Generator().manual_seed(0)
    )

    assert dense["mu_full"].shape != out["mu_full"].shape
    assert dense["mu_full"].shape[1] == model.geometry.t_valid


# =================================================================================================
# What the control replaces, and what it leaves alone
# =================================================================================================
def test_the_source_free_tensors_are_the_same_objects(perturb_posterior):
    """Identity rather than equality. A derangement of the source cannot move a target-only
    quantity, and the comparison $D_{\\mathrm{full}} < D_{\\mathrm{base}} <
    D_{\\mathrm{shuffled}}$ is only readable against one unmoved reference."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH))

    permuted = _permuted(model, out)

    for key in _UNTOUCHED_KEYS:
        assert permuted[key] is out[key], key


def test_the_source_driven_tensors_are_genuinely_rebuilt(perturb_posterior):
    """The positive direction: the posterior and the full forecast must move, or the control is
    measuring nothing."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH))

    permuted = _permuted(model, out)

    for key in ("mu_post", "logvar_post", "mu_full", "logvar_full"):
        assert not torch.equal(permuted[key], out[key]), key


def test_the_attention_is_rebuilt_under_this_models_own_lag_mask(perturb_posterior):
    """The control poses the query and attends again, and it must do so under the model's own mask
    -- reached through ``build_lag_mask``, which this target domain overrides for the lag floor. A
    control that built its own would compare two attention geometries."""
    model = _model(perturb_posterior, lag_floor=2)
    out = _forward(model, make_stub_batch(_BATCH))

    permuted = _permuted(model, out)

    mask = model.build_lag_mask(model.sequence_length, out["attn_weights"].device)
    forbidden = ~mask  # (T, L)
    weights = permuted["attn_weights"]  # (B, T, heads, L)
    assert float(weights.permute(0, 2, 1, 3)[:, :, forbidden].abs().max()) == 0.0


# =================================================================================================
# Through the task, on validation only
# =================================================================================================
def test_the_three_readouts_appear_on_validation_only(task, perturb_posterior):
    """A readout that never enters the objective and costs a second decode per step. Tracking a
    ``train/`` variant would produce a column that is NaN in every row of every run."""
    module = task()
    perturb_posterior(module.orig_model)
    batch = make_stub_batch(_BATCH)

    _loss, train_metrics = module.compute_loss_and_metrics(batch, 0, "train")
    _loss, val_metrics = module.compute_loss_and_metrics(batch, 0, "val")

    for name in ("nll_shuffled_block", "kld_shuffled", "shuffle_penalty"):
        assert name in val_metrics, name
        assert name not in train_metrics, name


def test_the_control_runs_on_a_real_validation_step_without_a_shape_error(
    task, perturb_posterior
):
    """The end-to-end version, and the one whose absence cost the conv-LSTM cell its first
    validation epoch: the task's own step resolves $(0, 1)$ on ``val``, so the control has to decode
    at the *dense* set there -- and at the tiled one whenever a caller supplies a training
    geometry."""
    module = task()
    perturb_posterior(module.orig_model)

    _loss, metrics = module.compute_loss_and_metrics(make_stub_batch(_BATCH), 0, "val")

    assert torch.isfinite(metrics["shuffle_penalty"]).all()
    assert float(metrics["kld_shuffled"]) != 0.0


def test_the_shuffled_penalty_is_the_gap_against_the_matched_full_score(
    task, perturb_posterior
):
    """The reported penalty must be a difference of two scores of the *same* target block at the
    *same* anchors, or the negative control measures the geometry rather than the source pathway."""
    module = task()
    perturb_posterior(module.orig_model)

    _loss, metrics = module.compute_loss_and_metrics(make_stub_batch(_BATCH), 0, "val")

    assert float(metrics["shuffle_penalty"]) == pytest.approx(
        float(metrics["nll_shuffled_block"]) - float(metrics["nll_full_block"]), rel=1e-5
    )


def test_the_task_hands_the_control_the_matched_forwards_anchors(task, perturb_posterior):
    """The wiring the shape assertions above are worthless without: the shared call site reads
    ``forward_outputs['anchor_index']``, which is absent on every model that decodes every anchor --
    so the seam is expressed once and no subclass overrides the step to use it."""
    seen = {}
    original = controls.perm_forward_outputs

    def _record(model, forward_outputs, **kwargs):
        seen["anchors"] = kwargs.get("anchors")
        return original(model, forward_outputs, **kwargs)

    module = task()
    perturb_posterior(module.orig_model)
    controls.perm_forward_outputs = _record
    try:
        module.compute_loss_and_metrics(make_stub_batch(_BATCH), 0, "val")
    finally:
        controls.perm_forward_outputs = original

    assert seen["anchors"] is not None
    # The validation stage decodes densely, so this is the dense anchor set -- which is still an
    # explicit one, and still not the contiguous prefix the argument's absence would mean.
    model = module.orig_model
    assert seen["anchors"].shape[1] == model.geometry.t_valid - model.warmup_period


def test_a_degenerate_batch_cannot_be_deranged(perturb_posterior):
    """$B < 2$ has no derangement, so the control refuses rather than pairing a sample with itself
    and reporting the result as a stranger's source."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(1))

    with pytest.raises(ValueError, match="batch_size >= 2"):
        _permuted(model, out)


def test_the_control_is_skipped_on_a_degenerate_validation_batch(task, perturb_posterior):
    """A rank's last uneven batch can be a single sample. The step must still produce a loss."""
    module = task()
    perturb_posterior(module.orig_model)

    loss, metrics = module.compute_loss_and_metrics(make_stub_batch(1), 0, "val")

    assert torch.isfinite(loss)
    assert "nll_shuffled_block" not in metrics


def test_the_control_uses_a_real_derangement(perturb_posterior):
    """No fixed point: a row paired with itself contributes a matched score to a shuffled
    average."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH))

    permuted = _permuted(model, out)

    assert not bool((permuted["perm_index"] == torch.arange(_BATCH)).any())
