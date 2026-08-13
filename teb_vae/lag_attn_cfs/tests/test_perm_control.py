r"""The imported permutation control, run against a model that decodes a tiled anchor set.

``nll_shuffled_block``, ``kld_shuffled`` and ``shuffle_penalty`` are tracked and **validation-only**,
so without this file their first execution would be on the production box after a full training
epoch. That is not a hypothetical: the control decodes ``z_post_perm`` at the anchors the matched
forward used, and until that argument was threaded through the shared call site it decoded the
contiguous prefix $[0, T_{\mathrm{valid}})$ instead -- $(B, 285, H, 98)$ against a $(B, 11, H)$ mask,
which is a shape error rather than a wrong number, and one that arrives only on a validation step.

The first assertion below is the shape, explicitly, and that is deliberate: ``torch.equal`` returns
``False`` on a shape mismatch, so a test written the way the two-sided sibling's is would pass on a
wrong-shaped control.

The property that makes the control readable is the one the tiling must not break: a derangement of
the source leaves every target-only quantity **untouched**, so that
$D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}$ compares three forecasts against one
unmoved reference -- gathered at one anchor set, since two anchor sets would make the comparison a
comparison of two questions.

Every assertion perturbs the posterior first. At initialisation the posterior *is* the prior, so a
deranged source moves nothing and every shuffled readout is $0$ for reasons that have nothing to do
with being correct.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
from teb_vae.lag_attn_rws.nets import controls

from .conftest import TINY_SEQ_LEN, TINY_STRIDE, make_stub_batch, tiny_warmup_kwargs

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


def _model(perturb_posterior=None, **overrides) -> SeqVaeLagAttnCfs:
    torch.manual_seed(0)
    model = SeqVaeLagAttnCfs(
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


def _features(batch) -> torch.Tensor:
    """The concatenated target stream, as the task hands it to the objective."""
    return torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1)


def _permute(model, out, batch_size: int = _BATCH):
    """The control, at the matched forward's own anchors -- as the shared task call site runs it."""
    return controls.perm_forward_outputs(
        model,
        out,
        perm_index=controls.make_derangement(batch_size),
        anchors=out["anchor_index"],
    )


# ---------------------------------------------------------------------------------------
# The shapes
# ---------------------------------------------------------------------------------------
def test_the_permuted_decode_has_the_same_shape_as_the_matched_one(perturb_posterior):
    """Asserted explicitly rather than through ``torch.equal``, which returns ``False`` on a shape
    mismatch -- so the two-sided sibling's version of this test would pass on a control that decoded
    285 anchors against an 11-anchor mask."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH, TINY_SEQ_LEN))

    permuted = _permute(model, out)

    for key in ("mu_full", "logvar_full"):
        assert permuted[key].shape == out[key].shape, key
    assert permuted["mu_full"].shape[1] == out["anchor_index"].shape[1]
    assert permuted["mu_full"].shape[-1] == model.decoder_out_channels


def test_a_control_given_no_anchors_would_decode_a_different_shape(perturb_posterior):
    """The negative control on the assertion above, and the failure the shared call site's
    ``anchors=`` argument exists to prevent: with none supplied the control decodes the contiguous
    prefix $[0, T_{\\mathrm{valid}})$, which is what a model decoding every anchor emits and what
    this one does not."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH, TINY_SEQ_LEN))

    dense = controls.perm_forward_outputs(
        model, out, perm_index=controls.make_derangement(_BATCH)
    )

    assert dense["mu_full"].shape[1] == model.geometry.t_valid
    assert dense["mu_full"].shape[1] != out["mu_full"].shape[1]


def test_the_anchor_keys_are_absent_from_the_recomputed_set_and_identical_between_the_dicts():
    """They describe the geometry rather than the pairing. A control that rebuilt them would be
    scoring the shuffled forecast at a different anchor set than the matched one, and the difference
    the readout reports would be partly that."""
    assert "anchor_index" not in controls.RECOMPUTED_KEYS
    assert "anchor_valid" not in controls.RECOMPUTED_KEYS

    model = _model()
    out = _forward(model, make_stub_batch(_BATCH, TINY_SEQ_LEN))
    permuted = _permute(model, out)

    assert permuted["anchor_index"] is out["anchor_index"]
    assert torch.equal(permuted["anchor_valid"], out["anchor_valid"])


# ---------------------------------------------------------------------------------------
# The control itself, against this net
# ---------------------------------------------------------------------------------------
def test_the_control_replaces_exactly_the_keys_it_declares(perturb_posterior):
    """Exactly, in both directions: an unlisted key that moved would be an undeclared control
    output, and a listed key that did not move would be a control that rebuilt nothing."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH, TINY_SEQ_LEN))

    permuted = _permute(model, out)

    assert set(permuted) - set(out) == {"perm_index"}
    moved = {key for key in out if permuted[key] is not out[key]}
    assert moved == set(controls.RECOMPUTED_KEYS) - {"perm_index"}


def test_the_source_free_tensors_are_the_same_objects(perturb_posterior):
    """Identity, not equality. The control promises to reuse the computed states rather than to
    reproduce them, and equality would also hold for a control that recomputed them and happened to
    agree."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH, TINY_SEQ_LEN))

    permuted = _permute(model, out)

    for key in _UNTOUCHED_KEYS:
        assert permuted[key] is out[key], f"{key} was rebuilt under permutation"


def test_the_source_driven_tensors_are_genuinely_rebuilt(perturb_posterior):
    """The mirror image, so the test above cannot pass on a control that rebuilds nothing."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH, TINY_SEQ_LEN))

    permuted = _permute(model, out)

    for key in ("mu_post", "logvar_post", "z_post", "attn_weights", "mu_full", "logvar_full"):
        assert not torch.equal(permuted[key], out[key]), f"{key} was not rebuilt"


def test_the_attention_is_rebuilt_under_the_models_own_floored_lag_mask(perturb_posterior):
    """The control routes through ``build_lag_mask`` rather than letting the attention build its
    own, so a model restricting which lags it may read cannot have that restriction bypassed by the
    control alone -- which would show up only as a shuffled readout quietly computed against more
    source history than the matched one."""
    model = _model(perturb_posterior, lag_floor=3)
    out = _forward(model, make_stub_batch(_BATCH, TINY_SEQ_LEN))

    permuted = _permute(model, out)

    mask = model.build_lag_mask(TINY_SEQ_LEN, torch.device("cpu"))
    # (T, L) against the attention's (B, T, num_heads, L).
    forbidden = (~mask)[None, :, None, :]
    assert bool(forbidden.any()), "the floor forbids nothing; this test would be vacuous"
    assert float((permuted["attn_weights"] * forbidden).abs().max()) == 0.0


def test_re_scoring_the_permuted_dict_reproduces_the_base_score_exactly(perturb_posterior):
    """The consequence the acceptance ordering rests on: the "no source" reference has not moved,
    checked by re-scoring the feature block at the same anchors rather than by identity alone."""
    model = _model(perturb_posterior)
    batch = make_stub_batch(_BATCH, TINY_SEQ_LEN)
    out = _forward(model, batch)

    permuted = _permute(model, out)
    true = model.compute_loss(out, _features(batch), weight=batch.weight)["metrics"]
    shuffled = model.compute_loss(permuted, _features(batch), weight=batch.weight)["metrics"]

    assert torch.equal(true["nll_base_block"], shuffled["nll_base_block"])


def test_scoring_the_permuted_dict_yields_a_different_nonzero_kl(perturb_posterior):
    """``compute_loss`` recomputes the KL from the distribution parameters, so the permuted
    posterior yields a genuinely different KL against the same prior."""
    model = _model(perturb_posterior)
    batch = make_stub_batch(_BATCH, TINY_SEQ_LEN)
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
    out = _forward(model, make_stub_batch(1, TINY_SEQ_LEN))

    with pytest.raises(ValueError, match="batch_size >= 2"):
        controls.perm_forward_outputs(model, out, anchors=out["anchor_index"])


def test_the_control_uses_a_real_derangement(perturb_posterior):
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH, TINY_SEQ_LEN))

    permuted = controls.perm_forward_outputs(model, out, anchors=out["anchor_index"])

    assert not bool((permuted["perm_index"] == torch.arange(_BATCH)).any())


# ---------------------------------------------------------------------------------------
# The control as the task runs it
# ---------------------------------------------------------------------------------------
def test_the_three_readouts_appear_on_validation_only(task, perturb_posterior):
    """Absent, never zero-filled, on the steps that did not run it: the framework aggregates a metric
    as the mean over the steps that reported it, so a zero placeholder would scale the epoch value
    down and invert the ordering the control exists to check."""
    module = task()
    perturb_posterior(module.orig_model)
    batch = make_stub_batch(_BATCH, TINY_SEQ_LEN)

    _, train_metrics = module.compute_loss_and_metrics(batch, 0, "train")
    _, val_metrics = module.compute_loss_and_metrics(batch, 0, "val")

    control = {"nll_shuffled_block", "kld_shuffled", "shuffle_penalty"}
    assert control & set(train_metrics) == set()
    assert control <= set(val_metrics)


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
        module.compute_loss_and_metrics(make_stub_batch(_BATCH, TINY_SEQ_LEN), 0, "val")
    finally:
        controls.perm_forward_outputs = original

    assert seen["anchors"] is not None
    # The validation stage decodes densely, so this is the dense anchor set -- which is still an
    # explicit one, and still not the contiguous prefix the argument's absence would mean.
    assert seen["anchors"].shape[1] == module.orig_model.geometry.t_valid - (
        module.orig_model.warmup_period
    )


def test_the_control_is_skipped_on_a_degenerate_validation_batch(task, perturb_posterior):
    """A rank's last uneven batch can be a single sample. The step must still produce a loss."""
    module = task()
    perturb_posterior(module.orig_model)

    loss, metrics = module.compute_loss_and_metrics(make_stub_batch(1, TINY_SEQ_LEN), 0, "val")

    assert torch.isfinite(loss)
    assert "nll_shuffled_block" not in metrics


def test_the_shuffled_penalty_is_the_gap_against_the_matched_full_score(task, perturb_posterior):
    """The reported penalty must be a difference of two scores of the *same* target block at the
    *same* anchors, or the negative control measures the geometry rather than the source pathway."""
    module = task()
    perturb_posterior(module.orig_model)
    batch = make_stub_batch(_BATCH, TINY_SEQ_LEN)

    _, metrics = module.compute_loss_and_metrics(batch, 0, "val")

    assert float(metrics["shuffle_penalty"]) == pytest.approx(
        float(metrics["nll_shuffled_block"]) - float(metrics["nll_full_block"]), rel=1e-5
    )


def test_the_control_leaves_the_prior_and_base_branch_bitwise_unchanged(task, perturb_posterior):
    """On the same batch under the same seed, with and without the control.

    Compared across two **validation** steps rather than against a training step, which is how the
    two-sided sibling writes it: this cell decodes a different anchor set on the two stages, so a
    train-versus-val comparison would differ for a reason that has nothing to do with the control.
    Every target-only readout must be **bitwise** identical: a control that leaked into the matched
    forward -- by re-running it, by consuming a draw from the shared generator, or by scoring the
    permuted dict in place -- would move exactly these numbers, by an amount no tolerance would flag.
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
        "target_warm_frac", "kld_source_null",
    ):
        assert torch.equal(with_control[name], without_control[name]), name
