r"""The objective as this model wires it: the target it builds, and the width it declares.

The arithmetic is not retested here. ``lag_attn_rws/nets/losses.py`` owns every term, every
reduction and every reported metric, and its own suite pins them. What is this model's to get wrong
is the wiring, and it has exactly two moving parts:

* **The target.** Gathered from the caller's feature stream by the target gate's keep-index and
  unfolded into each anchor's future window -- never delayed, and never rebuilt from a raw grid.
* **``block_width``.** $C_{\mathrm{keep}}$, the surviving-channel count. It feeds **only** the four
  per-element log-variance diagnostics, never a loss term, so passing ``geometry.r`` here would
  change no gradient, fail no shape check, and rescale exactly those four reported numbers by
  $78/16 = 4.875$ at the shipped budget. Those four are where ``logvar_clamp`` is re-derived from
  and where a collapsing decoder variance is first visible, so the mistake would be silent and
  expensive at once.

Both are checked against **hand-written** quantities rather than against the implementation. A ratio
the objective computes from a width the objective was given is self-consistent for any wrong width.

**Why the index identity is re-checked here rather than inherited.** The mixin is shared with the
conv-LSTM feature model, and that model's suite already pins it -- but the *gate* the mixin reads is
built at **this** architecture's own construction site, from this constructor's own keyword handling.
That the mixin reaches an un-delayed keep-index there is a separate fact, and it is asserted against
a hand-written slice-and-stack rather than against ``figure_primitives.future_target``: comparing
against the shared helper checks the unfold and the gather but **not** the delay, because a target
that wrongly applied it and a reference that wrongly applied it agree perfectly.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_fs.nets.feature_target import FeatureForecastTarget
from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs
from teb_vae.lag_attn_transformer_fs.tests.conftest import (
    SHIPPED_KWARGS,
    STUB_GAP_STEP,
    TINY_KEEP_INDEX,
    TINY_KWARGS,
    make_patterned_batch,
    make_stub_batch,
    shipped_gated_kwargs,
    tiny_gated_kwargs,
)
from teb_vae.lag_attn_rws.nets.losses import LOGVAR_FLOOR_MARGIN_FRAC
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask
from teb_vae.lag_attn_rws.tests.test_objective import assert_objective_reassembles
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

#: Coefficients the recomposition runs at. Mutually distinct and none of them a default: at equal
#: weights a term swapped for another passes, at ``beta_prior=0`` the fourth term is multiplied
#: away, and at ``free_bits=0`` the raw and trained KL are one tensor rather than two.
_COEFFICIENTS = dict(
    beta=0.7, beta_prior=0.11, lambda_full=1.0, lambda_base=0.3, free_bits=0.05
)

#: The shipped budget's surviving channels and the block the reconstruction sums over there.
#: Hand-written: $30 \times 78$. The point of the constant is that it is *not* read back from the
#: model being checked against it.
_KEPT_CHANNELS = 78
_SHIPPED_BLOCK = 2340

#: The full metric surface: the objective's own keys plus the four resolved forecast gaps.
_RESOLVED_GAP_KEYS = {
    "pred_gap_tau_first",
    "pred_gap_tau_last",
    "pred_gap_st",
    "pred_gap_ph",
}
_OBJECTIVE_KEYS = 27
_TOTAL_KEYS = 31


def _model(kwargs, cls=SeqVaeLagAttnTrfFs, **overrides):
    torch.manual_seed(0)
    return cls(**dict(kwargs, **overrides)).eval()


def _features(batch) -> torch.Tensor:
    """The concatenated target stream, in the declared block order."""
    return torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1)


def _forward(model, batch):
    torch.manual_seed(0)
    with torch.no_grad():
        return model(
            batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1)
        )


def _loss_batch():
    """A tiny batch whose feature values are order one.

    The planted-pattern batch is the right fixture for questions about *which* coefficient landed
    where, and the wrong one for questions about a loss value: its values run to $15{,}000$, a summed
    squared error over them reaches $10^{10}$, and a float32 four-term recomposition at that
    magnitude fails on round-off rather than on a wiring mistake.
    """
    return make_stub_batch()


def _shipped_batch(batch_size: int = 2, gap_step: int = 150):
    """A production-geometry batch with a deliberate gap, so the mask is not uniformly one.

    The stub batch's own gap sits at step $10$, inside the shipped warm-up of $30$ and therefore
    invisible to every mask; a gap no mask can see would leave the masked reductions below asserting
    nothing about masking.
    """
    length = int(SHIPPED_KWARGS["sequence_length"])
    batch = make_stub_batch(batch_size, length)
    batch.weight = torch.ones(batch_size, length)
    batch.weight[:, gap_step] = 0.0
    return batch


def _stacked_block(stream: torch.Tensor, horizon: int, t_valid: int) -> torch.Tensor:
    r"""The target block built by slicing and stacking, sharing no arithmetic with ``unfold``.

    Horizon step $\tau$ of every anchor is one contiguous slice of the stream, so the whole block is
    $H$ slices stacked on a new axis.

    Args:
        stream: A feature stream $(B, T, C)$.
        horizon: The forecast horizon $H$.
        t_valid: The number of valid anchors.

    Returns:
        The block $(B, T_{\mathrm{valid}}, H, C)$.
    """
    return torch.stack(
        [stream[:, 1 + tau : 1 + tau + t_valid, :] for tau in range(horizon)], dim=2
    )


# ---------------------------------------------------------------------------------------
# The index identity, against a hand-written slice
# ---------------------------------------------------------------------------------------
def test_the_index_identity_holds_at_every_position(shipped_gated):
    r"""$Y^{+}[b, t, \tau, k] = Y[b,\, t + 1 + \tau,\, \mathrm{keep}[k]]$, whole block.

    Against a slice-and-stack written in this file rather than against the shared unfold helper: the
    helper checks the window and the gather but not the delay, and a builder that wrongly applied the
    delay would agree with a reference that wrongly applied it.
    """
    model = _model(shipped_gated)
    batch = make_patterned_batch(2, int(SHIPPED_KWARGS["sequence_length"]))
    stream = _features(batch)

    built = model._build_forecast_target(stream)
    kept = torch.index_select(stream, -1, model.target_gate.keep_index)
    expected = _stacked_block(kept, model.horizon, model.geometry.t_valid)

    assert built.shape == (2, 270, 30, _KEPT_CHANNELS)
    assert torch.equal(built, expected)


@pytest.mark.parametrize(
    "anchor, tau",
    [(0, 0), (0, 29), (137, 11), (269, 29)],
    ids=["first-anchor-first-step", "first-anchor-last-step", "interior", "last-anchor-last-step"],
)
def test_the_planted_value_names_the_step_and_channel_it_came_from(shipped_gated, anchor, tau):
    r"""The same identity read off the planted pattern at four named positions, including the first
    and last valid anchor.

    Element $(b, t, \tau, k)$ is $(t + 1 + \tau)\,S + \mathrm{keep}[k]$, so dividing by $S$ recovers
    the *stored step* and the remainder recovers the *stored channel* -- which must be
    $\mathrm{keep}[k]$, not $k$. A gather of the first $78$ channels, or of the wrong $78$, fails
    here and nowhere else; and a delayed gather fails on the recovered step.
    """
    from teb_vae.lag_attn_transformer_fs.tests.conftest import PATTERN_STEP_SCALE

    model = _model(shipped_gated)
    batch = make_patterned_batch(2, int(SHIPPED_KWARGS["sequence_length"]))

    block = model._build_forecast_target(_features(batch))[0, anchor, tau]
    recovered_step = torch.div(block, PATTERN_STEP_SCALE, rounding_mode="floor")
    recovered_channel = block - recovered_step * PATTERN_STEP_SCALE

    assert torch.equal(recovered_step, torch.full_like(recovered_step, float(anchor + 1 + tau)))
    assert torch.equal(recovered_channel, model.target_gate.keep_index.to(block.dtype))


def test_the_target_is_not_what_this_models_gate_emits(shipped_gated):
    """The negative half, at this architecture's own gate.

    ``ChannelGate.forward`` is ``self.delay(index_select(...))`` -- there is no gather-only method --
    so a builder that called the gate would be delayed, and nothing downstream would fail. And the
    mistake is not partial: at the shipped budget **all 78** survivors carry a non-zero delay, so a
    gate-built target is wrong in every channel it contains.
    """
    model = _model(shipped_gated)
    batch = make_patterned_batch(2, int(SHIPPED_KWARGS["sequence_length"]))
    stream = _features(batch)
    delays = model.target_gate.delay.delay_steps

    assert delays.numel() == _KEPT_CHANNELS
    assert int((delays > 0).sum()) == _KEPT_CHANNELS
    assert int(delays.min()) == 1 and int(delays.max()) == 30

    built = model._build_forecast_target(stream)
    delayed = _stacked_block(
        model.target_gate(stream), model.horizon, model.geometry.t_valid
    )

    assert delayed.shape == built.shape  # the mistake is invisible to every shape check
    assert not torch.equal(built, delayed)
    # Specifically: the delayed block at anchor t is the correct block at t - delta, per channel.
    for channel in (int(torch.argmax(delays)), int(torch.argmin(delays))):
        shift = int(delays[channel])
        assert torch.equal(delayed[:, 200, :, channel], built[:, 200 - shift, :, channel])
        assert not torch.equal(delayed[:, 200, :, channel], built[:, 200, :, channel])


def test_the_ungated_target_keeps_every_declared_channel(shipped_kwargs):
    model = _model(shipped_kwargs)
    batch = make_patterned_batch(2, int(SHIPPED_KWARGS["sequence_length"]))
    stream = _features(batch)

    built = model._build_forecast_target(stream)

    assert model.target_gate is None
    assert built.shape[-1] == model.c_y == 109
    assert torch.equal(built, _stacked_block(stream, model.horizon, model.geometry.t_valid))


@pytest.mark.parametrize(
    "shape, match",
    [
        ((2, 16), "3-D"),
        ((2, 20, 109), "trim_minutes"),
        ((2, 16, 78), "c_y=109"),
    ],
    ids=["not-3d", "wrong-length", "already-gathered"],
)
def test_a_target_stream_that_does_not_match_the_geometry_is_refused(tiny_kwargs, shape, match):
    """The third case is the one worth having: a caller that gathered the channels itself would hand
    over a correctly-ranked tensor whose keep-index positions no longer mean what the model thinks,
    and the gather would silently take the wrong $78$ of them."""
    model = _model(tiny_kwargs)

    with pytest.raises(ValueError, match=match):
        model._build_forecast_target(torch.zeros(shape))


# ---------------------------------------------------------------------------------------
# The four-term total
# ---------------------------------------------------------------------------------------
def test_the_total_is_the_documented_four_term_sum(tiny_gated, perturb_posterior):
    """Distinct coefficients, perturbed model: the total must recompose from the returned parts under
    exactly the documented weights to $10^{-6}$ **relative**, and the three-term recomposition must
    fall short."""
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


def test_the_metric_key_set_is_the_feature_siblings_exactly(tiny_kwargs):
    """$27$ objective keys plus the four resolved gaps, $31$, and identical to the model this one is
    compared against at a fixed target.

    Exact in both directions. Every downstream reader -- the tracked-metric list, the loss-curve
    page, the spike breaker -- is keyed by name, so a name in one model and not the other is a column
    that silently empties; and a *fifth* addition arriving unannounced would be a readout no callback
    collects. The comparison against the *raw* variant is the same statement from the other side:
    exactly the four resolved gaps separate them.
    """
    batch = _loss_batch()
    feature = _model(tiny_kwargs)
    raw = _model(tiny_kwargs, cls=SeqVaeLagAttnTrfRws)

    feature_result = feature.compute_loss(
        _forward(feature, batch), _features(batch), weight=batch.weight
    )
    raw_result = raw.compute_loss(_forward(raw, batch), batch.fhr, weight=batch.weight)

    assert set(feature_result) == set(raw_result) == {"metrics", "likelihood"}
    assert len(raw_result["metrics"]) == _OBJECTIVE_KEYS
    assert len(feature_result["metrics"]) == _TOTAL_KEYS
    assert set(feature_result["metrics"]) - set(raw_result["metrics"]) == _RESOLVED_GAP_KEYS
    assert set(raw_result["metrics"]) - set(feature_result["metrics"]) == set()
    assert all(isinstance(value, torch.Tensor) for value in feature_result["metrics"].values())


def test_the_metric_key_set_matches_the_conv_lstm_feature_model_name_for_name(tiny_kwargs):
    """The comparison the encoder axis is read along. Both feature models delegate to the same mixin,
    so the key sets must be identical -- and if they were not, the two ``RESULTS.md`` tables would
    have different columns and neither would say so.

    Each model is built from its own suite's tiny keyword set, since the two constructors' schemas
    differ; both describe the same $T = 16$, $H = 4$ geometry, so the batch serves both.
    """
    from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
    from teb_vae.lag_attn_fs.tests.conftest import TINY_KWARGS as CONV_LSTM_TINY_KWARGS

    batch = _loss_batch()
    transformer = _model(tiny_kwargs)
    conv_lstm = _model(CONV_LSTM_TINY_KWARGS, cls=SeqVaeLagAttnFs)

    assert conv_lstm.geometry.t == transformer.geometry.t
    assert conv_lstm.horizon == transformer.horizon

    transformer_keys = set(
        transformer.compute_loss(
            _forward(transformer, batch), _features(batch), weight=batch.weight
        )["metrics"]
    )
    conv_lstm_keys = set(
        conv_lstm.compute_loss(
            _forward(conv_lstm, batch), _features(batch), weight=batch.weight
        )["metrics"]
    )

    assert transformer_keys == conv_lstm_keys
    assert len(transformer_keys) == _TOTAL_KEYS


# ---------------------------------------------------------------------------------------
# The resolved gaps recompose
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_both_block_splits_recompose_to_the_reported_gap(shipped_gated, likelihood):
    """The two stored blocks' partial gaps add back to ``pred_gap``, and the counts are asserted
    beside it -- because recomposition alone holds for **any** partition of the channels and so
    would not catch a boundary in the wrong place."""
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
    $66$. Splitting the kept axis at the declared $43$ instead -- the natural mistake, since $43$ is
    the number written down -- would put $16$ of the second block's channels into the first block's
    total and leave both reported numbers wrong by a large amount, with everything else unchanged.
    """
    model = _model(shipped_gated)
    keep = model.target_gate.keep_index
    split = FeatureForecastTarget.TARGET_BLOCK_SPLIT

    first_block = int((keep < split).sum())

    assert split == 43
    assert (first_block, int(keep.numel()) - first_block) == (27, 51)
    assert first_block != split, "the confusion is reachable rather than hypothetical"
    assert int((keep[:split] >= 43).sum()) == 16


def test_the_resolved_gaps_are_zero_at_init_and_carry_no_gradient(shipped_gated):
    """At initialisation the posterior is the prior, so every part of the gap is zero; a split that
    read the wrong branch, or scored against a differently-built target, would be non-zero here while
    ``pred_gap`` stayed at zero. And they are diagnostics: a term that reached the graph would be a
    fifth objective term no weight in the config controls."""
    model = _model(shipped_gated)
    batch = _shipped_batch()

    metrics = model.compute_loss(
        _forward(model, batch), _features(batch), weight=batch.weight, likelihood="mse"
    )["metrics"]

    for name in sorted(_RESOLVED_GAP_KEYS):
        assert float(metrics[name]) == pytest.approx(0.0, abs=1e-6), name
        assert not metrics[name].requires_grad, name


@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_at_init_the_two_reconstruction_terms_are_bitwise_equal(tiny_gated, likelihood):
    """The zero-KL start restated on the loss path: a wiring mistake between forward and loss -- a
    stale key, a wrong branch fed to the wrong term -- leaves the forward-path test green and this one
    red."""
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


def test_the_objective_carries_gradient_to_the_widened_head(tiny_gated, perturb_posterior):
    """A smoke check that the assembled total is trainable, and that the gradient reaches the decoder
    head whose width this model changed."""
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
    """The only route by which the target stream enters the loss is the gather, so planting an absurd
    value at the gapped step must leave every reconstruction number bitwise unchanged.

    Multiplicative masking is what makes this exact rather than merely small: an additive mask would
    leave $10^{9}$ contributing $0 \\times 10^{9}$ in float, which is $0$, but $10^{9}$ inside a
    squared error would already have overflowed the sum before the mask was applied.
    """
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

    # Not vacuous: an unmasked step moves the loss by a lot. Step 5, not the gap's neighbour -- at
    # H = 4 and coverage_floor = 0.9 the anchors whose window covers the gap are dropped *whole*, so
    # steps 7 through 11 are unscored too and planting there would prove nothing.
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
    """$H \\cdot C_{\\mathrm{keep}} = 30 \\times 78 = 2340$, written out rather than read off the
    model. A ratio computed from a width the objective was *given* is self-consistent for any wrong
    width; only a constant from outside can catch one."""
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
    # The raw variant's block, for the contrast beta was recalibrated against.
    assert _SHIPPED_BLOCK / (30 * int(SHIPPED_KWARGS["raw_per_step"])) == pytest.approx(4.875)


def test_mean_logvar_full_is_the_per_coefficient_mean_not_the_per_raw_sample_one():
    """The one assertion that catches ``block_width`` passed as ``geometry.r``.

    That mistake changes no loss, fails no shape check and moves nothing else in the metric dict; it
    rescales these four numbers by $78/16 = 4.875$. They are what ``logvar_clamp`` is re-derived from
    and what a collapsing decoder variance shows up in first, so the hand computation is written out
    here and the wrong denominator is asserted *not* to match.
    """
    model = _model(shipped_gated_kwargs())
    batch = _shipped_batch()
    out = _forward(model, batch)

    metrics = model.compute_loss(out, _features(batch), weight=batch.weight)["metrics"]

    mask, _coverage = forecast_mask(
        batch.weight, model.geometry, coverage_floor=model.coverage_floor
    )
    elem_mask = mask[..., None]
    correct_denominator = elem_mask.sum() * float(_KEPT_CHANNELS)
    expected = (out["logvar_full"] * elem_mask).sum() / correct_denominator

    assert torch.equal(metrics["mean_logvar_full"], expected)
    assert torch.equal(
        metrics["mean_logvar_base"],
        (out["logvar_base"] * elem_mask).sum() / correct_denominator,
    )

    # The negative control: the raw grid's R would give a value 4.875x larger, and the reported one
    # must not be it.
    wrong = (out["logvar_full"] * elem_mask).sum() / (
        elem_mask.sum() * float(model.geometry.r)
    )
    assert not torch.allclose(metrics["mean_logvar_full"], wrong)
    assert float(wrong / expected) == pytest.approx(_KEPT_CHANNELS / model.geometry.r, rel=1e-4)


def test_the_binding_bound_fractions_use_the_same_coefficient_denominator():
    """The other two of the four. They are *fractions*, so a wrong denominator does not merely
    rescale them -- it lets them exceed $1$, and a floor fraction above one is how this mistake would
    eventually be noticed rather than how it would be caught."""
    model = _model(shipped_gated_kwargs())
    batch = _shipped_batch()
    out = _forward(model, batch)

    metrics = model.compute_loss(out, _features(batch), weight=batch.weight)["metrics"]

    mask, _coverage = forecast_mask(
        batch.weight, model.geometry, coverage_floor=model.coverage_floor
    )
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


@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
@pytest.mark.parametrize("guard", ["ungated", "gated"], ids=["ungated", "gated"])
def test_every_metric_reassembles_from_the_primitives(perturb_posterior, likelihood, guard):
    """This model's metrics, against the raw-signal suite's independent reassembly.

    The arithmetic is the shared objective's -- one harness for all four forecasters -- and what
    this file supplies is what this model owns: its target, built by the slice-and-stack that
    shares no arithmetic with ``unfold``, its block width, and the four resolved forecast gaps.
    """
    gated = guard == "gated"
    model = _model(tiny_gated_kwargs() if gated else dict(TINY_KWARGS))
    perturb_posterior(model)
    batch = _loss_batch()
    outs = _forward(model, batch)

    target = _stacked_block(_features(batch), model.horizon, model.geometry.t_valid)
    if gated:
        target = torch.index_select(target, -1, torch.tensor(TINY_KEEP_INDEX))

    assert_objective_reassembles(
        model,
        outs,
        target,
        batch.weight,
        model.compute_loss(
            outs, _features(batch), weight=batch.weight, likelihood=likelihood, **_COEFFICIENTS
        )["metrics"],
        likelihood=likelihood,
        coefficients=_COEFFICIENTS,
        # Hand-written: the surviving width, or the declared one when nothing was dropped.
        block_width=len(TINY_KEEP_INDEX) if gated else 109,
        package_owned=_RESOLVED_GAP_KEYS,
    )


def test_the_block_width_follows_the_gate_at_every_budget(tiny_gated, tiny_kwargs):
    """Three survivors, then none: the sample score divides by $H$ times whatever the decoder emits,
    so the two must move together or one of them is a constant in disguise."""
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
