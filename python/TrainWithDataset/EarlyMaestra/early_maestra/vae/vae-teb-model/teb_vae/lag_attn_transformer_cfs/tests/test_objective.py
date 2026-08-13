r"""The objective as this model wires it: the block it sums, the anchors it sums over, and the mask.

The arithmetic is not retested here. ``lag_attn_rws/nets/losses.py`` owns every term, every
reduction and every reported metric, and its own suite pins them; a second copy of those assertions
would be a second copy of one piece of evidence. What this *composition* can get wrong is narrower,
and is what is checked:

* **The block width.** $C_{\mathrm{keep}}$, the surviving-channel count, not the architecture
  parent's raw grid $R$. It feeds only the four per-element log-variance diagnostics, never a loss
  term, so passing the wrong one would change no gradient, fail no shape check, and rescale exactly
  those four numbers -- which is why the width is pinned rather than assumed to follow from the base
  order.
* **The target.** Gathered at the keep-index and never delayed. A delayed target would ask anchor
  $t$ to forecast the future of anchor $t - \delta_c$, per channel, with every shape unchanged.
* **The anchor set.** Every per-anchor denominator is a count of *decoded* anchors. A padded slot
  that reached the loss would score a real target twice while the KL support counted it once.
* **The metric surface.** Exact in both directions against the conv-LSTM causal cell's, because the
  two are read side by side and a name in one and not the other is a column that silently empties.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
from teb_vae.lag_attn_cfs.tests.conftest import (
    TINY_KWARGS as CONV_LSTM_TINY_KWARGS,
)
from teb_vae.lag_attn_cfs.tests.conftest import (
    tiny_warmup_kwargs as conv_lstm_tiny_warmup_kwargs,
)
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors, forecast_mask

from .conftest import (
    BATCH,
    TINY_KWARGS,
    TINY_STRIDE,
    build,
    make_streams,
    shipped_warmup_kwargs,
    tiny_warmup_kwargs,
)

#: The shipped budget's surviving target channels and the block the reconstruction sums over.
#: Hand-written -- $15 \times 98$ -- and the point of the constants is that they are *not* read back
#: from the model being checked against them. A ratio computed from a width the objective was
#: **given** is self-consistent for any wrong width.
_SHIPPED_KEPT_CHANNELS = 98
_SHIPPED_BLOCK = 1470

#: What this target domain adds to the raw-signal sibling's metric dict: the four resolved gaps it
#: inherits from the two-sided feature target, plus the seven this family introduces.
_ADDED_METRIC_KEYS = {
    "pred_gap_tau_first",
    "pred_gap_tau_last",
    "pred_gap_st",
    "pred_gap_ph",
    "pred_gap_warm_lo",
    "pred_gap_warm_mid",
    "pred_gap_warm_hi",
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
# The block width and the target
# =================================================================================================
def test_the_block_width_follows_the_budget_and_not_the_raw_grid() -> None:
    r"""$H \cdot C_{\mathrm{keep}}$, not $H \cdot R$. Reversing the base order would give the
    latter, and nothing in the objective would raise: ``block_width`` is an *argument*."""
    model, streams, features = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))
    metrics = model.compute_loss(out, features, weight=_weight(model))["metrics"]

    kept = model.target_gate.out_channels
    assert model.decoder_out_channels == kept
    assert out["mu_base"].shape[-1] == kept != model.raw_per_step
    # The four per-element log-variance diagnostics are the only readers of the width, and they are
    # bounded by the clamp -- which is a per-COEFFICIENT statement here, not a per-raw-sample one.
    for name in ("logvar_full_floor_frac", "logvar_full_ceil_frac"):
        assert 0.0 <= float(metrics[name]) <= 1.0, name


def test_the_target_is_gathered_at_the_keep_index_and_never_delayed() -> None:
    r"""$Y^{+}[b, a, \tau, k] = Y[b,\, t_a + 1 + \tau,\, \mathrm{keep}[k]]$, position by position,
    with the pattern naming its own coordinates so a transposed gather or an off-by-one anchor is a
    wrong *value* rather than a right shape."""
    model, _streams, _features = _tiled()
    batch, length = BATCH, model.geometry.t
    channels = model.c_y
    features = (
        torch.arange(batch, dtype=torch.float32).view(-1, 1, 1) * (length * channels)
        + torch.arange(length, dtype=torch.float32).view(1, -1, 1) * channels
        + torch.arange(channels, dtype=torch.float32).view(1, 1, -1)
    )
    anchors = torch.tensor([[5, 9], [6, 10]], dtype=torch.long)

    target = model._build_forecast_target(features, anchors=anchors)

    keep = model.target_gate.keep_index
    for b in range(batch):
        for a in range(anchors.shape[1]):
            for tau in range(model.horizon):
                expected = features[b, int(anchors[b, a]) + 1 + tau, keep]
                assert torch.equal(target[b, a, tau], expected)


# =================================================================================================
# The anchor denominator, and the padded slot
# =================================================================================================
def test_the_anchor_denominator_is_the_hand_computed_tile_count() -> None:
    """Counted from the geometry rather than read off the mask the objective built, because a
    denominator read from the thing under test is self-consistent with any error in it."""
    model, streams, _features = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))
    mask, _coverage = forecast_mask(
        _weight(model),
        model.geometry,
        coverage_floor=model.coverage_floor,
        anchors=out["anchor_index"],
        anchor_valid=out["anchor_valid"],
    )

    span = model.geometry.t_valid - model.warmup_period
    expected = [
        (span - phase + TINY_STRIDE - 1) // TINY_STRIDE for phase in _PHASES
    ]

    assert contributing_anchors(mask).sum(dim=-1).tolist() == expected


def test_a_padded_anchor_slot_moves_the_loss_by_exactly_zero() -> None:
    """The failure the padding convention exists to prevent, at the one phase that produces a
    padded slot. A slot holding a *distinct legal* anchor would be gathered and scored a second
    time while the KL support -- a set -- counted it once, and $\\beta$ would quietly change
    meaning."""
    model, streams, features = _tiled()
    short_phase = max(_PHASES)
    out = _forward(model, streams, torch.tensor([short_phase, short_phase]))
    assert not bool(out["anchor_valid"].all()), "no padded slot exists; the test is vacuous"

    full = model.compute_loss(out, features, weight=_weight(model))["metrics"]

    # Recompute with every padded slot pointed somewhere else entirely. The padded index is masked
    # out, so nothing about the loss may move.
    moved = dict(out)
    index = out["anchor_index"].clone()
    index[~out["anchor_valid"]] = model.warmup_period
    moved["anchor_index"] = index
    shifted = model.compute_loss(moved, features, weight=_weight(model))["metrics"]

    for name in ("nll_full_block", "nll_base_block", "total_loss", "pred_gap"):
        assert torch.equal(full[name], shifted[name]), name


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
# The metric surface
# =================================================================================================
def test_the_metric_key_set_is_the_raw_siblings_plus_this_target_domains_eleven() -> None:
    """Exact in both directions, against a declared addition rather than a free one. Every
    downstream reader is keyed by name, so a name in one model and not the other is a column that
    silently empties."""
    model, streams, features = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))
    causal = model.compute_loss(out, features, weight=_weight(model))["metrics"]

    torch.manual_seed(0)
    raw = SeqVaeLagAttnRws(**dict(CONV_LSTM_TINY_KWARGS)).eval()
    raw_streams = make_streams(CONV_LSTM_TINY_KWARGS)
    torch.manual_seed(0)
    with torch.no_grad():
        raw_out = raw(*raw_streams)
    raw_metrics = raw.compute_loss(
        raw_out, torch.zeros(BATCH, raw.geometry.raw_len), weight=_weight(raw)
    )["metrics"]

    assert set(causal) - set(raw_metrics) == _ADDED_METRIC_KEYS
    assert set(raw_metrics) - set(causal) == set()
    assert all(isinstance(value, torch.Tensor) for value in causal.values())


def test_the_metric_key_set_is_the_conv_lstm_causal_cells_exactly() -> None:
    """The encoder edge on the metric surface: the two cells are read side by side, and a readout
    present on one and absent on the other is a comparison nobody can make."""
    model, streams, features = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))
    mine = model.compute_loss(out, features, weight=_weight(model))["metrics"]

    kwargs = conv_lstm_tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
    torch.manual_seed(0)
    conv_lstm = SeqVaeLagAttnCfs(**kwargs).eval()
    y_st, y_ph, u_stream = make_streams(CONV_LSTM_TINY_KWARGS)
    torch.manual_seed(0)
    with torch.no_grad():
        theirs_out = conv_lstm(y_st, y_ph, u_stream, torch.tensor(_PHASES))
    theirs = conv_lstm.compute_loss(
        theirs_out, torch.cat([y_st, y_ph], dim=-1), weight=_weight(conv_lstm)
    )["metrics"]

    assert set(mine) == set(theirs)


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


def test_the_three_shape_terms_ship_off_and_report_exact_zeros() -> None:
    """Raw-waveform concepts against an unordered channel index, and the boundary term is
    additionally a slicing identity over adjacent anchors."""
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


def test_the_shipped_block_is_comparable_to_one_sibling_only() -> None:
    r"""Recorded where it is checkable rather than only in prose: at $H = 15$ against the two-sided
    cells' $30$, and $98$ kept channels against their $78$, the block is $1470$ against $2340$ --
    so a nat from this configuration is comparable to the conv-LSTM causal cell and to nothing
    else."""
    model = build(shipped_warmup_kwargs())

    assert model.horizon == 15
    assert model.decoder_out_channels == _SHIPPED_KEPT_CHANNELS
    assert model.horizon * model.decoder_out_channels == _SHIPPED_BLOCK
    assert _SHIPPED_BLOCK != 30 * 78
    assert _SHIPPED_BLOCK != model.horizon * model.raw_per_step
