r"""The objective as this model wires it: the target it builds, and the width it declares.

The arithmetic is not retested here. ``lag_attn_rws/nets/losses.py`` owns every term, every
reduction and every reported metric, and its own suite pins them; a second copy of those
assertions would be a second copy of one piece of evidence. What is this model's to get wrong is
the wiring, and it has exactly two moving parts:

* **The target.** Gathered from the caller's feature stream by the target gate's keep-index and
  unfolded into each anchor's future window -- never delayed, and never rebuilt from a raw grid.
* **``block_width``.** $C_{\mathrm{keep}}$, the surviving-channel count. It feeds **only** the
  four per-element log-variance diagnostics, never a loss term, so passing ``geometry.r`` here
  would change no gradient, fail no shape check, and rescale exactly those four reported numbers
  by $78/16 = 4.875$ at the shipped budget. Those four are where ``logvar_clamp`` is re-derived
  from and where a collapsing decoder variance is first visible, so the mistake would be silent
  and expensive at once.

Both are therefore checked against **hand-written** quantities rather than against the
implementation. A ratio the objective computes from a width the objective was given is
self-consistent for any wrong width.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.figure_primitives import future_target
from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.tests.conftest import (
    SHIPPED_KWARGS,
    STUB_GAP_STEP,
    make_patterned_batch,
    make_stub_batch,
    shipped_gated_kwargs,
)
from teb_vae.lag_attn_rws.nets.losses import LOGVAR_FLOOR_MARGIN_FRAC, raw_sample_score
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors, forecast_mask

#: Coefficients the recomposition runs at. Mutually distinct and none of them a default: at equal
#: weights a term swapped for another passes, at ``beta_prior=0`` the fourth term is multiplied
#: away, and at ``free_bits=0`` the raw and trained KL are one tensor rather than two.
_COEFFICIENTS = dict(
    beta=0.7, beta_prior=0.11, lambda_full=1.0, lambda_base=0.3, free_bits=0.05
)

#: The shipped budget's surviving channels and the block the reconstruction sums over there.
#: Hand-written: $30 \times 78$. The point of the constant is that it is *not* read back from the
#: model that is being checked against it.
_KEPT_CHANNELS = 78
_SHIPPED_BLOCK = 2340

#: The four metrics this model reports and the raw-signal sibling does not: ``pred_gap`` resolved
#: by horizon step and split by stored block. Declared once so the key-set comparison names the
#: same set the resolved-gap tests below drive.
_RESOLVED_GAP_KEYS = {
    "pred_gap_tau_first",
    "pred_gap_tau_last",
    "pred_gap_st",
    "pred_gap_ph",
}


def _model(kwargs, cls=SeqVaeLagAttnFs, **overrides):
    torch.manual_seed(0)
    return cls(**dict(kwargs, **overrides)).eval()


def _forward(model, batch):
    torch.manual_seed(0)
    with torch.no_grad():
        return model(
            batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1)
        )


def _features(batch) -> torch.Tensor:
    """The concatenated target stream, in the declared block order."""
    return torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1)


def _loss_batch():
    """A tiny batch whose feature values are order one.

    The planted-pattern batch is the right fixture for questions about *which* coefficient landed
    where, and the wrong one for questions about a loss value: its values run to $15{,}000$, a
    summed squared error over them reaches $10^{10}$, and a float32 four-term recomposition at
    that magnitude fails on round-off rather than on a wiring mistake.
    """
    return make_stub_batch()


def _shipped_batch(batch_size: int = 2, gap_step: int = 150):
    """A production-geometry batch with a deliberate gap, so the mask is not uniformly one.

    The stub batch's own gap sits at step $10$, which is inside the shipped warm-up of $30$ and
    therefore invisible to every mask; a gap that no mask can see would leave the masked
    reductions below asserting nothing about masking.
    """
    length = SHIPPED_KWARGS["sequence_length"]
    batch = make_stub_batch(batch_size, length)
    batch.weight = torch.ones(batch_size, length)
    batch.weight[:, gap_step] = 0.0
    return batch


# ---------------------------------------------------------------------------------------
# The target the objective is handed
# ---------------------------------------------------------------------------------------
def test_the_target_is_the_gathered_unfold_the_suite_is_pinned_against(tiny_gated):
    """The model's own builder against the composition ``test_feature_target.py`` pins: the shared
    unfold, then ``index_select`` on the surviving channels. Two definitions of the target would
    let the figure path and the loss path score different windows."""
    model = _model(tiny_gated)
    batch = make_patterned_batch()

    built = model._build_forecast_target(_features(batch))
    expected = torch.index_select(
        future_target(batch.fhr_st, batch.fhr_ph, model.horizon), -1, model.target_gate.keep_index
    )

    assert built.shape == expected.shape
    assert torch.equal(built, expected)


def test_gathering_before_the_unfold_is_the_same_target(tiny_gated):
    """The model gathers first, which keeps the copy at $(B, T, C)$ rather than at
    $(B, T_{\\mathrm{valid}}, H, C)$ -- a factor of $H$, a third of a gigabyte at the production
    batch. The two orders commute, and this is where that is checked rather than assumed."""
    model = _model(tiny_gated)
    batch = make_patterned_batch()
    stream = _features(batch)

    gather_last = torch.index_select(
        stream[:, 1:, :].unfold(dimension=1, size=model.horizon, step=1).permute(0, 1, 3, 2),
        -1,
        model.target_gate.keep_index,
    )

    assert torch.equal(model._build_forecast_target(stream), gather_last)


def test_the_target_is_not_delayed(tiny_gated):
    """The sharpest correctness trap in the model, asserted where the target is actually built.
    The gate's delays are non-zero and distinct here, so a builder that called the gate would be
    wrong by a different number of steps in each channel -- and nothing downstream would fail."""
    model = _model(tiny_gated)
    batch = make_patterned_batch()
    stream = _features(batch)

    built = model._build_forecast_target(stream)
    delayed = (
        model.target_gate(stream)[:, 1:, :]
        .unfold(dimension=1, size=model.horizon, step=1)
        .permute(0, 1, 3, 2)
    )

    assert built.shape == delayed.shape
    assert not torch.equal(built, delayed)
    # And specifically: the delayed block at anchor t is the correct block at t - delta.
    delays = model.target_gate.delay.delay_steps
    channel = int(torch.argmax(delays))
    shift = int(delays[channel])
    assert shift > 0
    assert torch.equal(delayed[:, 5, :, channel], built[:, 5 - shift, :, channel])


def test_the_ungated_target_keeps_every_declared_channel(tiny_kwargs):
    model = _model(tiny_kwargs)
    batch = make_patterned_batch()

    built = model._build_forecast_target(_features(batch))

    assert model.target_gate is None
    assert built.shape[-1] == model.c_y == 109
    assert torch.equal(built, future_target(batch.fhr_st, batch.fhr_ph, model.horizon))


@pytest.mark.parametrize(
    "shape, match",
    [
        ((2, 16), "3-D"),
        ((2, 20, 109), "trim_minutes"),
        ((2, 16, 78), "c_y=109"),
    ],
    ids=["not-3d", "wrong-length", "already-gathered"],
)
def test_a_target_stream_that_does_not_match_the_geometry_is_refused(
    tiny_kwargs, shape, match
):
    """The third case is the one worth having: a caller that gathered the channels itself would
    hand over a correctly-ranked tensor whose keep-index positions no longer mean what the model
    thinks, and the gather would silently take the wrong 78 of them."""
    model = _model(tiny_kwargs)

    with pytest.raises(ValueError, match=match):
        model._build_forecast_target(torch.zeros(shape))


# ---------------------------------------------------------------------------------------
# The four-term total
# ---------------------------------------------------------------------------------------
def test_the_total_is_the_documented_four_term_sum(tiny_gated, perturb_posterior):
    """Distinct coefficients, perturbed model: the total must recompose from the returned parts
    under exactly the documented weights, and the three-term recomposition must fall short."""
    model = _model(tiny_gated)
    perturb_posterior(model)
    batch = _loss_batch()

    metrics = model.compute_loss(
        _forward(model, batch), _features(batch), weight=batch.weight, **_COEFFICIENTS
    )["metrics"]
    recomposed = (
        _COEFFICIENTS["lambda_full"] * metrics["nll_full_block"]
        + _COEFFICIENTS["lambda_base"] * metrics["nll_base_block"]
        + _COEFFICIENTS["beta"] * metrics["source_conditioned_kl_train"]
        + _COEFFICIENTS["beta_prior"] * metrics["prior_rate"]
    )

    assert torch.allclose(metrics["total_loss"], recomposed, rtol=1e-6, atol=1e-6)
    assert float(metrics["source_conditioned_kl_train"]) > 0.0
    assert float(metrics["prior_rate"]) > 0.0
    three_term = recomposed - _COEFFICIENTS["beta_prior"] * metrics["prior_rate"]
    assert not torch.allclose(metrics["total_loss"], three_term, rtol=1e-6)


def test_the_metric_key_set_is_the_siblings_plus_the_four_resolved_gaps(tiny_kwargs):
    """Exact in both directions, against a declared addition rather than a free one. Every
    downstream reader -- the tracked-metric list, the loss-curve page, the spike breaker -- is
    keyed by name, so a name in one model and not the other is a column that silently empties; and
    a *fifth* addition arriving unannounced would be a readout no callback collects.

    The four are :data:`_RESOLVED_GAP_KEYS`, and they are partial sums of the ``pred_gap`` beside
    them rather than new quantities -- which is why the raw-signal sibling neither has them nor
    needs them: its block is thirty horizon steps of one physical signal, so neither split says
    anything there."""
    batch = _loss_batch()
    feature_model = _model(tiny_kwargs)
    raw_model = _model(tiny_kwargs, cls=SeqVaeLagAttnRws)

    feature_result = feature_model.compute_loss(
        _forward(feature_model, batch), _features(batch), weight=batch.weight
    )
    raw_result = raw_model.compute_loss(
        _forward(raw_model, batch), batch.fhr, weight=batch.weight
    )

    assert set(feature_result) == set(raw_result) == {"metrics", "likelihood"}
    assert set(feature_result["metrics"]) - set(raw_result["metrics"]) == _RESOLVED_GAP_KEYS
    assert set(raw_result["metrics"]) - set(feature_result["metrics"]) == set()
    assert all(isinstance(value, torch.Tensor) for value in feature_result["metrics"].values())


# ---------------------------------------------------------------------------------------
# The resolved forecast gaps
#
# Four partial sums of ``pred_gap``, and the only reason they exist is that with no evaluation
# pipeline the summed number cannot separate forecasting from reconstruction of the part of the
# target the model's own history already determines. So what is checked is exactly that: they are
# partial sums (both splits recompose), they are per-anchor (the same denominator), and the block
# split follows the reach budget's keep-index rather than assuming the survivors are contiguous.
# ---------------------------------------------------------------------------------------
def _gap_by_horizon_step(model, outs, batch, likelihood: str) -> torch.Tensor:
    """The per-horizon-step forecast gap, assembled here from the objective's own primitives.

    Written out rather than read off the model so the two endpoint metrics are checked against a
    quantity this file computed: a curve derived from the same private method that produced the
    metrics would be self-consistent whatever either did.

    Args:
        model: The net.
        outs: Its forward dict.
        batch: The batch the forward was run on.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.

    Returns:
        The gap $(H,)$, in nats per anchor.
    """
    target = model._build_forecast_target(_features(batch))
    mask, _coverage = forecast_mask(
        batch.weight, model.geometry, coverage_floor=model.coverage_floor
    )
    n_anchors = contributing_anchors(mask).to(target.dtype).sum().clamp_min(1.0)
    gap = (
        raw_sample_score(outs["mu_base"], target, likelihood=likelihood, logvar=outs["logvar_base"])
        - raw_sample_score(
            outs["mu_full"], target, likelihood=likelihood, logvar=outs["logvar_full"]
        )
    ) * mask[..., None]
    return gap.sum(dim=(0, 1, 3)) / n_anchors


@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_the_horizon_split_recomposes_to_the_reported_gap(shipped_gated, likelihood):
    r"""Summed over $\tau$ the horizon curve is ``pred_gap``, and its two endpoints are the two
    reported scalars. Both halves matter: the endpoints alone would pass for a curve that was not
    a decomposition of anything, and the recomposition alone would pass for endpoints read off the
    wrong end."""
    model = _model(shipped_gated)
    batch = _shipped_batch()
    outs = _forward(model, batch)

    by_tau = _gap_by_horizon_step(model, outs, batch, likelihood)
    metrics = model.compute_loss(
        outs, _features(batch), weight=batch.weight, likelihood=likelihood
    )["metrics"]

    assert by_tau.numel() == SHIPPED_KWARGS["horizon"] == 30
    assert float(metrics["pred_gap_tau_first"]) == pytest.approx(float(by_tau[0]), rel=1e-5)
    assert float(metrics["pred_gap_tau_last"]) == pytest.approx(float(by_tau[-1]), rel=1e-5)
    assert float(by_tau.sum()) == pytest.approx(float(metrics["pred_gap"]), rel=1e-4)


def test_the_two_reported_steps_are_the_first_and_the_last_and_not_each_other(
    shipped_gated, perturb_posterior
):
    """The endpoints are the whole point of the horizon split -- half the target's support lies in
    observed history at $\\tau = 0$ and none of it does at $\\tau = 29$ -- so a model whose gap is
    the same at both would make the readout say nothing. Perturbed, so the two genuinely differ."""
    model = _model(shipped_gated)
    perturb_posterior(model)
    batch = _shipped_batch()
    outs = _forward(model, batch)

    by_tau = _gap_by_horizon_step(model, outs, batch, "gaussian_nll")
    metrics = model.compute_loss(outs, _features(batch), weight=batch.weight)["metrics"]

    assert float(by_tau[0]) != pytest.approx(float(by_tau[-1]), rel=1e-6)
    assert float(metrics["pred_gap_tau_first"]) != pytest.approx(
        float(metrics["pred_gap_tau_last"]), rel=1e-6
    )


@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_the_block_split_recomposes_to_the_reported_gap(shipped_gated, likelihood):
    """The other axis, and the one where a wrong split is invisible: the two parts add back to
    ``pred_gap`` for **any** partition of the channels, so recomposition alone would not catch a
    boundary in the wrong place. The counts are asserted beside it."""
    model = _model(shipped_gated)
    batch = _shipped_batch()

    metrics = model.compute_loss(
        _forward(model, batch), _features(batch), weight=batch.weight, likelihood=likelihood
    )["metrics"]

    assert float(metrics["pred_gap_st"]) + float(metrics["pred_gap_ph"]) == pytest.approx(
        float(metrics["pred_gap"]), rel=1e-4
    )


def test_the_block_split_is_made_against_the_declared_index_not_the_kept_axis(shipped_gated):
    r"""The boundary is $43$ in the **declared** channel order and $27$ along the **kept** axis the
    forecast block is indexed by, and those two numbers are what a wrong split confuses.

    At the shipped budget $27$ of the first block's $43$ channels survive and $51$ of the second's
    $66$. Splitting the kept axis at the declared $43$ instead -- the natural mistake, since $43$
    is the number written down -- would put $16$ of the second block's channels into the first
    block's total and leave both reported numbers wrong by a large amount, with everything else
    unchanged.
    """
    model = _model(shipped_gated)
    keep = model.target_gate.keep_index

    first_block = int((keep < SeqVaeLagAttnFs.TARGET_BLOCK_SPLIT).sum())

    assert SeqVaeLagAttnFs.TARGET_BLOCK_SPLIT == 43
    assert (first_block, int(keep.numel()) - first_block) == (27, 51)
    # The two numbers really are different, so the confusion is reachable rather than hypothetical.
    assert first_block != SeqVaeLagAttnFs.TARGET_BLOCK_SPLIT
    assert int((keep[:SeqVaeLagAttnFs.TARGET_BLOCK_SPLIT] >= 43).sum()) == 16


def test_a_narrower_budget_moves_the_split_with_it(shipped_kwargs):
    r"""The split is a function of the resolved keep-index, not of a fixed offset, so a different
    budget re-partitions it. Checked at ``null``, where every declared channel survives and the two
    parts are the two blocks' full widths -- the one configuration in which the answer is known
    without consulting the filter bank."""
    model = _model(shipped_kwargs)
    outs = _forward(model, _shipped_batch())
    batch = _shipped_batch()

    metrics = model.compute_loss(outs, _features(batch), weight=batch.weight)["metrics"]

    assert model.target_gate is None
    assert model.decoder_out_channels == SHIPPED_KWARGS["c_y"] == 109
    assert float(metrics["pred_gap_st"]) + float(metrics["pred_gap_ph"]) == pytest.approx(
        float(metrics["pred_gap"]), rel=1e-4
    )


def test_the_resolved_gaps_are_zero_at_init_like_the_gap_they_decompose(shipped_gated):
    """At initialisation the posterior is the prior, so every part of the gap is zero. A split
    that read the wrong branch, or scored against a differently-built target, would be nonzero
    here while ``pred_gap`` stayed at zero."""
    model = _model(shipped_gated)
    batch = _shipped_batch()

    metrics = model.compute_loss(
        _forward(model, batch), _features(batch), weight=batch.weight, likelihood="mse"
    )["metrics"]

    for name in sorted(_RESOLVED_GAP_KEYS):
        assert float(metrics[name]) == pytest.approx(0.0, abs=1e-6), name


def test_the_resolved_gaps_carry_no_gradient(shipped_gated, perturb_posterior):
    """They are diagnostics. A term that reached the graph would be a fifth objective term that no
    weight in the config controls."""
    model = _model(shipped_gated)
    perturb_posterior(model)
    batch = _shipped_batch()

    outs = model(batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1))
    metrics = model.compute_loss(outs, _features(batch), weight=batch.weight)["metrics"]

    for name in sorted(_RESOLVED_GAP_KEYS):
        assert not metrics[name].requires_grad, name


def test_the_resolved_gaps_are_per_anchor_like_every_other_reported_term(shipped_gated):
    r"""Same denominator as ``pred_gap``: the contributing-anchor count, not the anchor count.

    Halving the valid anchors must leave the four in the same band rather than halving them, which
    is what "nats per anchor" means and what makes them addable to the number they decompose. The
    check is the recomposition itself under a heavier mask -- a split that divided by a different
    denominator would still recompose to *something*, but not to ``pred_gap``.
    """
    model = _model(shipped_gated)
    batch = _shipped_batch()
    batch.weight[:, : batch.weight.shape[1] // 2] = 0.0

    metrics = model.compute_loss(
        _forward(model, batch), _features(batch), weight=batch.weight
    )["metrics"]

    assert float(metrics["pred_gap_st"]) + float(metrics["pred_gap_ph"]) == pytest.approx(
        float(metrics["pred_gap"]), rel=1e-4
    )
    assert float(metrics["anchor_coverage_frac"]) < 1.0  # the mask really did bite


@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_at_init_the_two_reconstruction_terms_are_bitwise_equal(tiny_gated, likelihood):
    """The zero-KL start restated on the loss path: a wiring mistake between forward and loss --
    a stale key, a wrong branch fed to the wrong term -- leaves the forward-path test green and
    this one red."""
    model = _model(tiny_gated)
    batch = _loss_batch()

    metrics = model.compute_loss(
        _forward(model, batch), _features(batch), weight=batch.weight, likelihood=likelihood
    )["metrics"]

    assert torch.equal(metrics["nll_full_block"], metrics["nll_base_block"])
    assert float(metrics["source_conditioned_kl_train"]) == 0.0
    assert float(metrics["pred_gap"]) == 0.0


def test_an_unknown_likelihood_is_rejected_listing_the_choices(tiny_kwargs):
    model = _model(tiny_kwargs)
    batch = _loss_batch()

    with pytest.raises(ValueError, match=r"mse.*gaussian_nll"):
        model.compute_loss(
            _forward(model, batch), _features(batch), weight=batch.weight, likelihood="huber"
        )


def test_the_objective_carries_gradient(tiny_gated, perturb_posterior):
    """A smoke check that the assembled total is trainable, and that the gradient reaches the
    decoder head whose width this model changed."""
    model = _model(tiny_gated).train()
    perturb_posterior(model)
    batch = _loss_batch()

    out = model(batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1))
    result = model.compute_loss(out, _features(batch), weight=batch.weight)
    result["metrics"]["total_loss"].backward()

    assert model.decoder.mean_head.weight.grad is not None
    assert float(model.decoder.mean_head.weight.grad.abs().max()) > 0.0


# ---------------------------------------------------------------------------------------
# The masked plant
# ---------------------------------------------------------------------------------------
def test_a_gapped_step_is_invisible_end_to_end(tiny_gated):
    """The only route by which the target stream enters the loss is the gather, so planting an
    absurd value at the gapped step must leave every reconstruction number bitwise unchanged.

    Multiplicative masking is what makes this exact rather than merely small: an additive mask
    would leave $10^{9}$ contributing $0 \\times 10^{9}$ in float, which is $0$, but $10^{9}$
    inside a squared error would already have overflowed the sum before the mask was applied."""
    model = _model(tiny_gated)
    batch = _loss_batch()
    out = _forward(model, batch)
    stream = _features(batch)

    reference = model.compute_loss(out, stream, weight=batch.weight)
    planted = stream.clone()
    planted[:, STUB_GAP_STEP, :] = 1.0e9
    result = model.compute_loss(out, planted, weight=batch.weight)

    differing = [
        key
        for key, value in reference["metrics"].items()
        if not torch.equal(value, result["metrics"][key])
    ]
    assert not differing, differing

    # Not vacuous: an unmasked step moves the loss by a lot. Step 5, not the gap's neighbour --
    # at H = 4 and coverage_floor = 0.9 the four anchors whose window covers the gap are dropped
    # *whole*, so steps 7 through 11 are unscored too and planting there would prove nothing.
    unmasked = stream.clone()
    unmasked[:, 5, :] = 1.0e9
    moved = model.compute_loss(out, unmasked, weight=batch.weight)
    assert not torch.equal(
        reference["metrics"]["nll_full_block"], moved["metrics"]["nll_full_block"]
    )


# ---------------------------------------------------------------------------------------
# block_width: the trap
# ---------------------------------------------------------------------------------------
def test_the_sample_score_divides_the_block_by_the_hand_written_cardinality():
    """$H_d \\cdot C_{\\mathrm{keep}} = 30 \\times 78 = 2340$, written out rather than read off the
    model. A ratio computed from a width the objective was *given* is self-consistent for any
    wrong width; only a constant from outside can catch one."""
    model = _model(shipped_gated_kwargs())
    batch = _shipped_batch()

    metrics = model.compute_loss(
        _forward(model, batch), _features(batch), weight=batch.weight
    )["metrics"]

    assert model.decoder_out_channels == _KEPT_CHANNELS
    assert float(metrics["nll_full_sample"]) == pytest.approx(
        float(metrics["nll_full_block"]) / _SHIPPED_BLOCK, rel=1e-6
    )
    assert float(metrics["nll_base_sample"]) == pytest.approx(
        float(metrics["nll_base_block"]) / _SHIPPED_BLOCK, rel=1e-6
    )
    # The raw model's block, for the contrast beta is recalibrated against.
    assert _SHIPPED_BLOCK / (30 * SHIPPED_KWARGS["raw_per_step"]) == pytest.approx(4.875)


def test_mean_logvar_full_is_the_per_coefficient_mean_not_the_per_raw_sample_one():
    """The one assertion that catches ``block_width`` passed as ``geometry.r``.

    That mistake changes no loss, fails no shape check and moves nothing else in the metric dict;
    it rescales these four numbers by $78/16 = 4.875$. They are what ``logvar_clamp`` is
    re-derived from and what a collapsing decoder variance shows up in first, so the hand
    computation is written out here and the wrong denominator is asserted *not* to match.
    """
    model = _model(shipped_gated_kwargs())
    batch = _shipped_batch()
    out = _forward(model, batch)

    metrics = model.compute_loss(out, _features(batch), weight=batch.weight)["metrics"]

    mask, _ = forecast_mask(batch.weight, model.geometry, coverage_floor=model.coverage_floor)
    elem_mask = mask[..., None]
    correct_denominator = elem_mask.sum() * float(_KEPT_CHANNELS)
    expected = (out["logvar_full"] * elem_mask).sum() / correct_denominator

    assert torch.equal(metrics["mean_logvar_full"], expected)
    assert torch.equal(
        metrics["mean_logvar_base"],
        (out["logvar_base"] * elem_mask).sum() / correct_denominator,
    )

    # The negative control: the raw grid's R would give a value 4.875x larger, and the reported
    # one must not be it.
    wrong = (out["logvar_full"] * elem_mask).sum() / (
        elem_mask.sum() * float(model.geometry.r)
    )
    assert not torch.allclose(metrics["mean_logvar_full"], wrong)
    assert float(wrong / expected) == pytest.approx(_KEPT_CHANNELS / model.geometry.r, rel=1e-4)


def test_the_binding_bound_fractions_use_the_same_coefficient_denominator():
    """The other two of the four. They are *fractions*, so a wrong denominator does not merely
    rescale them -- it lets them exceed $1$, and a floor fraction above one is how this mistake
    would eventually be noticed rather than how it would be caught."""
    model = _model(shipped_gated_kwargs())
    batch = _shipped_batch()
    out = _forward(model, batch)

    metrics = model.compute_loss(out, _features(batch), weight=batch.weight)["metrics"]

    mask, _ = forecast_mask(batch.weight, model.geometry, coverage_floor=model.coverage_floor)
    elem_mask = mask[..., None]
    denominator = elem_mask.sum() * float(_KEPT_CHANNELS)
    lo, hi = model.logvar_clamp
    margin = LOGVAR_FLOOR_MARGIN_FRAC * (hi - lo)

    expected_floor = (
        (out["logvar_full"] <= lo + margin).to(out["logvar_full"].dtype) * elem_mask
    ).sum() / denominator
    expected_ceil = (
        (out["logvar_full"] >= hi - margin).to(out["logvar_full"].dtype) * elem_mask
    ).sum() / denominator

    assert torch.equal(metrics["logvar_full_floor_frac"], expected_floor)
    assert torch.equal(metrics["logvar_full_ceil_frac"], expected_ceil)
    assert 0.0 <= float(metrics["logvar_full_floor_frac"]) <= 1.0
    assert 0.0 <= float(metrics["logvar_full_ceil_frac"]) <= 1.0


def test_the_block_width_follows_the_gate_at_every_budget(tiny_gated, tiny_kwargs):
    """Three survivors, then none: the sample score divides by $H_d$ times whatever the decoder
    emits, so the two must move together or one of them is a constant in disguise."""
    batch = _loss_batch()

    for kwargs, channels in ((tiny_gated, 3), (tiny_kwargs, 109)):
        model = _model(kwargs)
        metrics = model.compute_loss(
            _forward(model, batch), _features(batch), weight=batch.weight
        )["metrics"]

        assert model.decoder_out_channels == channels
        assert float(metrics["nll_full_sample"]) == pytest.approx(
            float(metrics["nll_full_block"]) / (model.horizon * channels), rel=1e-6
        )
