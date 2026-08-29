r"""The objective as this model wires it: the block it sums, the anchors it sums over, and the mask.

The arithmetic is not retested here. ``lag_attn_rws/nets/losses.py`` owns every term, every
reduction and every reported metric, and its own suite pins them; a second copy of those
assertions would be a second copy of one piece of evidence. What this package can get wrong is
narrower, and is what is checked:

* **The block width.** $C_{\mathrm{keep}}$, the surviving-channel count, not the raw grid's $R$.
  It feeds only the four per-element log-variance diagnostics, never a loss term, so passing the
  wrong one would change no gradient, fail no shape check, and rescale exactly those four numbers.
* **The anchor set.** Every per-anchor denominator is a count of *decoded* anchors, and this is
  the first model in the family whose decoded set is not the dense range. A padded slot that
  reached the loss would score a real target twice while the KL support counted it once.
* **The mask.** A value planted where the mask is zero must move the loss by *exactly* zero --
  multiplicatively, not approximately.

The tiling itself is isolated by scoring one forward two ways: through this model's own
``compute_loss``, and through the **shared** objective given the same anchor index explicitly,
against a target built here rather than by the model. Comparing against ``anchors=None`` instead
is not available, and the reason is worth stating rather than discovering: ``None`` means the
dense range $[0, T_{\mathrm{valid}})$, which is $F$ entries *longer* than this family's densest
set -- so the target would not even have the forecast's shape, and where the shapes did line up
``anchor_coverage_frac`` would still be averaged over a different set.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_cfs.tests.conftest import (
    BATCH,
    CAUSAL_C_Y,
    TINY_KWARGS,
    TINY_STRIDE,
    build,
    make_streams,
    shipped_warmup_kwargs,
    tiny_warmup_kwargs,
)
from teb_vae.lag_attn_rws.nets.losses import compute_loss as compute_shared_objective
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors, forecast_mask

#: The shipped budget's surviving target channels and the block the reconstruction sums over.
#: Hand-written -- $15 \times 98$ -- and the point of the constants is that they are *not* read
#: back from the model being checked against them. A ratio computed from a width the objective was
#: **given** is self-consistent for any wrong width.
_SHIPPED_KEPT_CHANNELS = 98
_SHIPPED_BLOCK = 2940

#: What this package's ``compute_loss`` adds to the raw-signal sibling's metric dict: the four
#: resolved gaps it inherits from the two-sided feature target, plus the ten this family
#: introduces. Declared as one set so an unannounced fifteenth addition fails here rather than
#: arriving in a CSV no callback collects.
_ADDED_METRIC_KEYS = {
    "pred_gap_tau_first",
    "pred_gap_tau_last",
    "pred_gap_st",
    "pred_gap_ph",
    "pred_gap_warm_lo",
    "pred_gap_warm_mid",
    "pred_gap_warm_hi",
    "pred_gap_novel_lo",
    "pred_gap_novel_mid",
    "pred_gap_novel_hi",
    "target_warm_frac",
    "anchors_per_sample",
    "source_lag_warmth_frac_st",
    "source_lag_warmth_frac_ph",
}

#: The two phases the tiled fixtures run at. Chosen so the second row is one anchor short of the
#: first, which is the only way a padded slot exists at all -- and every padding assertion below
#: would pass vacuously without one.
_PHASES = (0, 3)


def _weight(model, batch: int = BATCH, gap_step: int = -1) -> torch.Tensor:
    """An all-valid decimated weight, optionally with one step zeroed."""
    weight = torch.ones(batch, model.geometry.t)
    if gap_step >= 0:
        weight[:, gap_step] = 0.0
    return weight


def _forward(model, streams, phase, stride=None):
    torch.manual_seed(0)
    with torch.no_grad():
        return model(*streams, phase, stride)


def _tiled(stride: int = TINY_STRIDE):
    """The tiny model at a tiling, its three input tensors, and its concatenated target stream."""
    kwargs = tiny_warmup_kwargs(anchor_stride=stride)
    model = build(kwargs).eval()
    y_st, y_ph, u_stream = make_streams(kwargs)
    return model, (y_st, y_ph, u_stream), torch.cat([y_st, y_ph], dim=-1)


# =================================================================================================
# The block width
# =================================================================================================
def test_the_sample_score_divides_the_block_by_the_hand_written_cardinality() -> None:
    r"""$H \cdot C_{\mathrm{keep}} = 30 \times 98 = 2940$, written out rather than read off the
    model -- and asserted a second time as ``horizon * decoder_out_channels``, so a horizon or
    budget change re-derives it instead of failing this literal."""
    kwargs = shipped_warmup_kwargs()
    model = build(kwargs).eval()
    streams = make_streams(kwargs)
    out = _forward(model, streams, torch.zeros(BATCH, dtype=torch.long))

    metrics = model.compute_loss(
        out, torch.cat(streams[:2], dim=-1), weight=_weight(model)
    )["metrics"]

    assert model.decoder_out_channels == _SHIPPED_KEPT_CHANNELS
    assert model.horizon * model.decoder_out_channels == _SHIPPED_BLOCK
    for branch in ("full", "base"):
        assert float(metrics[f"nll_{branch}_sample"]) == pytest.approx(
            float(metrics[f"nll_{branch}_block"]) / _SHIPPED_BLOCK, rel=1e-6
        )
    # The raw grid's R would divide by 15 x 16 = 240 instead: a factor of 6.125 out, no shape
    # wrong and no gradient changed.
    assert _SHIPPED_BLOCK / (model.horizon * model.geometry.r) == pytest.approx(6.125)


def test_the_block_width_follows_the_budget_at_both_guard_states() -> None:
    """Two guard states, one relation. The sample score divides by $H$ times whatever the decoder
    emits, so the two must move together or one of them is a constant in disguise."""
    guarded = tiny_warmup_kwargs()
    cases = ((guarded, len(guarded["target_keep_index"])), (dict(TINY_KWARGS), CAUSAL_C_Y))

    for kwargs, channels in cases:
        model = build(kwargs).eval()
        y_st, y_ph, u_stream = make_streams(kwargs)
        out = _forward(model, (y_st, y_ph, u_stream), None)

        metrics = model.compute_loss(
            out, torch.cat([y_st, y_ph], dim=-1), weight=_weight(model)
        )["metrics"]

        assert model.decoder_out_channels == channels
        assert float(metrics["nll_full_sample"]) == pytest.approx(
            float(metrics["nll_full_block"]) / (model.horizon * channels), rel=1e-6
        )


# =================================================================================================
# The anchor set the block is averaged over
# =================================================================================================
def test_the_anchor_denominator_is_the_hand_computed_tile_count() -> None:
    r"""$\sum_b |\mathcal A(\varphi_b)|$, computed here from the geometry rather than read off the
    model: $\lceil (T_{\mathrm{valid}} - F - \varphi)/S \rceil$ per sample.

    The denominator is the one thing a tiled objective can get wrong with no shape changing --
    ``contributing_anchors`` reduces the last axis alone, so a mask of another rank inflates every
    per-anchor denominator silently -- and it is what makes every reported number nats *per
    anchor* rather than nats per batch.
    """
    model, streams, _features = _tiled()
    phase = torch.tensor(_PHASES)
    out = _forward(model, streams, phase)

    span = model.geometry.t_valid - model.warmup_period
    expected = [-(-(span - int(value)) // TINY_STRIDE) for value in phase]
    assert len(set(expected)) == 2, "the fixture no longer produces a short row"

    mask, _coverage = forecast_mask(
        _weight(model),
        model.geometry,
        coverage_floor=model.coverage_floor,
        anchors=out["anchor_index"],
        anchor_valid=out["anchor_valid"],
    )

    assert int(out["anchor_valid"].sum()) == sum(expected)
    assert int(contributing_anchors(mask).sum()) == sum(expected)


def test_a_padded_anchor_slot_moves_the_loss_by_exactly_zero() -> None:
    r"""The padding convention's whole justification, asserted at the loss.

    A padded slot repeats its row's last real anchor, so its forecast row is a *duplicate* of a
    live one -- and if the mask did not multiply ``anchor_valid`` in, that target block would be
    scored twice by the reconstruction while the KL support, which is a set, counted it once. The
    two per-anchor denominators would diverge and $\beta$ would quietly stop meaning what it means
    in every other cell of the grid.

    Exactly zero, not approximately: the mask is multiplicative, so an absurd value at a padded
    slot leaves every reported number bitwise unchanged.
    """
    model, streams, features = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))
    weight = _weight(model)

    padded = ~out["anchor_valid"]
    assert bool(padded.any()), "no padded slot in this batch; the probe would be vacuous"

    reference = model.compute_loss(out, features, weight=weight)["metrics"]
    planted = dict(out)
    for key in ("mu_base", "mu_full", "logvar_base", "logvar_full"):
        tensor = out[key].clone()
        tensor[padded] = 1.0e9
        planted[key] = tensor
    moved = model.compute_loss(planted, features, weight=weight)["metrics"]

    differing = [
        name for name, value in reference.items() if not torch.equal(value, moved[name])
    ]
    assert not differing, differing

    # Not vacuous: the same plant at a *live* slot moves the loss by a lot.
    live = out["anchor_valid"].clone()
    live[:, 1:] = False
    at_live = dict(out)
    at_live["mu_full"] = out["mu_full"].clone()
    at_live["mu_full"][live] = 1.0e9
    assert not torch.equal(
        reference["nll_full_block"],
        model.compute_loss(at_live, features, weight=weight)["metrics"]["nll_full_block"],
    )


def test_a_gapped_step_moves_the_loss_by_exactly_zero() -> None:
    r"""The other half of the multiplicative-mask claim, on the target stream rather than the
    forecast.

    The only route by which the stream enters the loss is the gather, so a planted $10^{9}$ at a
    step no surviving anchor scores must leave every reported number bitwise unchanged. Exactness
    is the point: an additive mask would leave $0 \times 10^{9}$, which is $0$ in float -- but
    $10^{9}$ inside a squared error would already have overflowed the sum before the mask was
    applied.
    """
    model, streams, features = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))

    # A step the first decoded anchor's window covers. Zeroing it drops that anchor *whole* --
    # its coverage falls below the floor -- so the planted value is scored by nothing.
    gap_step = int(out["anchor_index"][0, 0]) + 2
    weight = _weight(model, gap_step=gap_step)

    reference = model.compute_loss(out, features, weight=weight)["metrics"]
    planted = features.clone()
    planted[:, gap_step, :] = 1.0e9
    moved = model.compute_loss(out, planted, weight=weight)["metrics"]

    differing = [
        name for name, value in reference.items() if not torch.equal(value, moved[name])
    ]
    assert not differing, differing

    # Not vacuous: a step the *second* decoded anchor reads, which the gap does not touch.
    live_step = int(out["anchor_index"][0, 1]) + 1
    unmasked = features.clone()
    unmasked[:, live_step, :] = 1.0e9
    assert not torch.equal(
        reference["nll_full_block"],
        model.compute_loss(out, unmasked, weight=weight)["metrics"]["nll_full_block"],
    )


# =================================================================================================
# The shape terms, off
# =================================================================================================
def test_the_three_shape_terms_ship_off_and_report_exact_zeros() -> None:
    """This target domain's last axis counts *channels*, which have no order and no continuity, so
    a pooled trajectory, a first difference and a boundary sample all mean nothing here. The
    zeros-when-off contract is what keeps those three columns honest zeros rather than raw-domain
    formulas evaluated over a channel axis."""
    model, streams, features = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))

    metrics = model.compute_loss(out, features, weight=_weight(model))["metrics"]

    for name in ("aux_multiscale", "aux_derivative", "aux_boundary"):
        assert float(metrics[name]) == 0.0, name
    for name in ("lambda_ms", "lambda_deriv", "lambda_boundary"):
        assert float(metrics[name]) == 0.0, name


def test_a_weighted_boundary_term_is_refused_because_the_anchors_are_not_neighbours() -> None:
    r"""``masked_boundary_gap`` identifies anchor $t$'s last observed sample with a slice of anchor
    $t-1$'s target block, which is a slicing identity only while the anchor axis is contiguous. On
    a tile the two rows are $S$ steps apart, so the term would compare unrelated samples."""
    model, streams, features = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))

    with pytest.raises(ValueError, match="lambda_boundary"):
        model.compute_loss(out, features, weight=_weight(model), lambda_boundary=0.1)


# =================================================================================================
# The tiling, isolated
# =================================================================================================
@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_the_dense_stride_is_the_shared_objective_given_the_same_anchors(
    perturb_posterior, likelihood
) -> None:
    r"""At ``anchor_stride: 1`` the model decodes exactly $[F, T_{\mathrm{valid}})$, and scoring
    that forward through the **shared** objective with the same index supplied explicitly -- and a
    target built here rather than by the model -- reproduces every metric bitwise.

    This is what isolates the tiling: the delegation adds the fourteen readouts of this package
    and changes nothing whatever about the objective it delegates to.
    """
    model, streams, features = _tiled(stride=1)
    perturb_posterior(model)
    out = _forward(model, streams, None)
    weight = _weight(model)

    dense = torch.arange(model.warmup_period, model.geometry.t_valid)
    explicit = dense[None, :].expand(BATCH, -1).contiguous()
    assert torch.equal(out["anchor_index"], explicit)
    assert bool(out["anchor_valid"].all())

    through_model = model.compute_loss(
        out, features, weight=weight, likelihood=likelihood
    )["metrics"]
    reference = compute_shared_objective(
        dict(
            out,
            anchor_index=explicit,
            anchor_valid=torch.ones_like(explicit, dtype=torch.bool),
        ),
        model._build_forecast_target(features, explicit),
        weight=weight,
        geometry=model.geometry,
        block_width=model.decoder_out_channels,
        coverage_floor=model.coverage_floor,
        logvar_clamp=model.logvar_clamp,
        likelihood=likelihood,
    )["metrics"]

    assert set(through_model) - set(reference) == _ADDED_METRIC_KEYS
    assert set(reference) - set(through_model) == set()
    differing = [
        name
        for name, value in reference.items()
        if not torch.equal(value, through_model[name])
    ]
    assert not differing, differing


def test_the_dense_range_is_not_the_none_anchor_set_and_the_objective_says_so() -> None:
    r"""Why the reference above supplies the index explicitly. ``anchors=None`` means
    $[0, T_{\mathrm{valid}})$ -- the range a model decoding *every* anchor emits -- which is $F$
    entries longer than this family's densest set, so the target it builds does not even have the
    forecast's shape."""
    model, streams, features = _tiled(stride=1)
    out = _forward(model, streams, None)

    assert out["anchor_index"].shape[1] == model.geometry.t_valid - model.warmup_period
    assert out["anchor_index"].shape[1] != model.geometry.t_valid

    stripped = {key: value for key, value in out.items() if not key.startswith("anchor_")}
    with pytest.raises(RuntimeError):
        model.compute_loss(stripped, features, weight=_weight(model))


# =================================================================================================
# The metric surface
# =================================================================================================
def test_the_metric_key_set_is_the_raw_siblings_plus_this_packages_fourteen() -> None:
    """Exact in both directions, against a declared addition rather than a free one. Every
    downstream reader is keyed by name, so a name in one model and not the other is a column that
    silently empties -- and a fifteenth addition arriving unannounced would be a readout no callback
    collects."""
    model, streams, features = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))
    causal = model.compute_loss(out, features, weight=_weight(model))["metrics"]

    torch.manual_seed(0)
    raw = SeqVaeLagAttnRws(**dict(TINY_KWARGS)).eval()
    raw_streams = make_streams(TINY_KWARGS)
    torch.manual_seed(0)
    with torch.no_grad():
        raw_out = raw(*raw_streams)
    raw_metrics = raw.compute_loss(
        raw_out, torch.zeros(BATCH, raw.geometry.raw_len), weight=_weight(raw)
    )["metrics"]

    assert set(causal) - set(raw_metrics) == _ADDED_METRIC_KEYS
    assert set(raw_metrics) - set(causal) == set()
    assert all(isinstance(value, torch.Tensor) for value in causal.values())


def test_the_objective_carries_gradient_to_the_widened_decoder_head() -> None:
    """A smoke check that the assembled total is trainable *through the anchor gather*, and that
    the gradient reaches the head whose width this target domain changed."""
    model, streams, features = _tiled()
    model.train()

    out = model(*streams, torch.tensor(_PHASES))
    result = model.compute_loss(out, features, weight=_weight(model))
    result["metrics"]["total_loss"].backward()

    assert model.decoder.mean_head.weight.grad is not None
    assert float(model.decoder.mean_head.weight.grad.abs().max()) > 0.0


@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_at_init_the_two_reconstruction_terms_are_bitwise_equal(likelihood) -> None:
    """The zero-KL start restated on the loss path: a wiring mistake between the forward's anchor
    gather and the loss's -- a stale index, one branch gathered at another's anchors -- leaves the
    forward-path test green and this one red."""
    model, streams, features = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))

    metrics = model.compute_loss(
        out, features, weight=_weight(model), likelihood=likelihood
    )["metrics"]

    assert torch.equal(metrics["nll_full_block"], metrics["nll_base_block"])
    assert float(metrics["source_conditioned_kl_train"]) == 0.0
    assert float(metrics["pred_gap"]) == 0.0


def test_the_shipped_block_is_comparable_to_no_sibling() -> None:
    r"""Recorded where it is checkable rather than only in prose: at the two-sided sibling's own
    $H = 30$ but $98$ kept channels against its $78$, the block is $2940$ against $2340$ -- so a nat
    from this configuration is still comparable to no other cell of the grid, and the horizon is no
    longer the reason."""
    model = build(shipped_warmup_kwargs())

    assert model.horizon == 30
    assert model.horizon * model.decoder_out_channels == _SHIPPED_BLOCK
    assert _SHIPPED_BLOCK != 30 * 78


# =================================================================================================
# The horizon weighting, where it touches the block arithmetic
#
# The reconstruction is a SUM over the block, and the two loss-scale constants a run is guarded by
# -- `gradient_clip_val` and the spike breaker's `additive_margin` -- are both stated in nats of
# that sum. So a horizon weight is a change to the objective's units unless it is renormalised, and
# it is: the weight sums to H, so what moves is the distribution across horizon steps and not the
# scale. What is checked here is that the block arithmetic above survives it.
# =================================================================================================
def test_the_weighted_block_still_divides_by_the_same_cardinality() -> None:
    r"""The sample score is the block over $H \cdot C_{\mathrm{keep}}$, and that denominator is a
    count of coefficients rather than a sum of weights.

    Which is the right choice and also the one that could silently be made wrong: dividing by
    $\sum_\tau w_\tau \cdot C_{\mathrm{keep}}$ would give the same number today, because the weight
    sums to $H$ -- and would start giving a different one the day the renormalisation moved. So the
    relation is asserted under the weight, against the same hand-written cardinality.
    """
    kwargs = shipped_warmup_kwargs(horizon_weight_halflife_steps=15.0)
    model = build(kwargs).eval()
    streams = make_streams(kwargs)
    out = _forward(model, streams, torch.zeros(BATCH, dtype=torch.long))

    metrics = model.compute_loss(
        out, torch.cat(streams[:2], dim=-1), weight=_weight(model)
    )["metrics"]

    assert model.horizon * model.decoder_out_channels == _SHIPPED_BLOCK
    for branch in ("full", "base"):
        assert float(metrics[f"nll_{branch}_sample"]) == pytest.approx(
            float(metrics[f"nll_{branch}_block"]) / _SHIPPED_BLOCK, rel=1e-6
        )


def test_the_weight_moves_the_block_by_far_less_than_it_moves_a_horizon_step() -> None:
    r"""The property both loss-scale constants rest on, measured on the real objective rather than
    on the weight vector alone.

    Two claims, and the second is what makes the first non-vacuous. The weighted block is close to
    the unweighted one, so a run's clip and margin do not go out of date. And the weight is a real
    reweighting -- the per-horizon-step partial sums move substantially -- so the first is not the
    trivial statement that the weight did nothing.

    The tolerance is loose on purpose. Exact equality would hold only on a score that is flat in
    $\tau$, and this one is not: a forecast is worse at the far steps, so redistributing mass
    towards the near ones moves the total a little. What is refused is a factor, not a percent.
    """
    kwargs = shipped_warmup_kwargs()
    streams = make_streams(kwargs)
    features = torch.cat(streams[:2], dim=-1)

    scores = {}
    for name, halflife in (("uniform", None), ("weighted", 15.0)):
        model = build(dict(kwargs, horizon_weight_halflife_steps=halflife)).eval()
        out = _forward(model, streams, torch.zeros(BATCH, dtype=torch.long))
        scores[name] = float(
            model.compute_loss(out, features, weight=_weight(model))["metrics"][
                "nll_full_block"
            ]
        )

    ratio = scores["weighted"] / scores["uniform"]
    assert 0.9 < ratio < 1.1, (
        f"the horizon weight moved the block by {ratio:.3f}x, so gradient_clip_val and the spike "
        f"breaker's additive_margin -- both stated in nats of this sum -- are out of date"
    )

    # Non-vacuity: the weight itself is far from flat, so the ratio above is a property of the
    # renormalisation rather than of a mechanism that is not running.
    from teb_vae.lag_attn_rws.nets.losses import horizon_decay_weight

    weight = horizon_decay_weight(15.0, int(kwargs["horizon"]))
    assert float(weight[0]) / float(weight[-1]) > 3.0
    assert float(weight.sum()) == pytest.approx(float(kwargs["horizon"]), rel=1e-6)
