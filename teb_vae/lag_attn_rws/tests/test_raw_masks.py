r"""The decimated forecast/KL masks, pinned against a naive raw-resolution gather.

The tiny geometry ($T = 16$, $H = 4$, $w = 2$, $T_{\mathrm{valid}} = 12$) is small enough that
the expected mask for a planted gap can be written out by hand: a gap at step $g$ zeroes anchor
$g$ itself (its own present block is invalid) and forecast step $\tau = g - 1 - t$ of every
anchor $t \in [g - H, g)$ (its future window covers the gap).
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry
from teb_vae.lag_attn_rws.nets.raw_masks import (
    VALID_THRESHOLD,
    contributing_anchors,
    forecast_mask,
    kl_mask,
)
from teb_vae.lag_attn_rws.nets.raw_targets import build_future_index

_TINY = TrimmedRawGeometry(raw_len=256, decimation=16, horizon=4, warmup=2)
_BATCH = 2


def _weight(gap_steps: tuple[int, ...] = ()) -> torch.Tensor:
    weight = torch.ones(_BATCH, _TINY.t)
    for step in gap_steps:
        weight[:, step] = 0.0
    return weight


def _naive_raw_mask(weight: torch.Tensor) -> torch.Tensor:
    """The mask built the raw-resolution way: upsample, gather per raw index, gate.

    This is the reference the decimated construction must equal exactly. It shares only the
    index grid with the code under test; the upsample-and-gather is independent.
    """
    valid = (weight >= VALID_THRESHOLD).to(weight.dtype)
    raw_valid = valid.repeat_interleave(_TINY.decimation, dim=1)          # (B, raw_len)
    idx = build_future_index(_TINY)                                        # (T_valid, H, R)
    future = raw_valid[:, idx.reshape(-1)].reshape(
        weight.size(0), _TINY.t_valid, _TINY.horizon, _TINY.r
    )
    warm = (torch.arange(_TINY.t_valid) >= _TINY.warmup).to(weight.dtype)
    anchor = valid[:, : _TINY.t_valid]
    return warm[None, :, None, None] * anchor[:, :, None, None] * future


@pytest.mark.parametrize("gaps", [(), (10,), (3, 10), (0, 7, 15)])
def test_the_decimated_mask_broadcast_over_r_equals_the_naive_raw_gather(gaps):
    """The equivalence is pinned, not assumed: validity is constant within a decimated step."""
    weight = _weight(gaps)
    mask, _ = forecast_mask(weight, _TINY)
    broadcast = mask[:, :, :, None].expand(-1, -1, -1, _TINY.r)
    assert torch.equal(broadcast, _naive_raw_mask(weight))


def test_a_planted_gap_zeroes_exactly_the_affected_anchor_samples():
    gap = 10
    mask, _ = forecast_mask(_weight((gap,)), _TINY)

    expected = torch.ones(_BATCH, _TINY.t_valid, _TINY.horizon)
    expected[:, : _TINY.warmup] = 0.0                     # warm-up prefix
    expected[:, gap] = 0.0                                # the gapped anchor itself
    for anchor in range(gap - _TINY.horizon, gap):        # anchors whose window covers the gap
        expected[:, anchor, gap - 1 - anchor] = 0.0
    assert torch.equal(mask, expected)


def test_a_gap_free_weight_masks_only_the_warmup_prefix():
    mask, coverage = forecast_mask(_weight(), _TINY)
    assert torch.equal(mask[:, _TINY.warmup :], torch.ones(_BATCH, 10, _TINY.horizon))
    assert (mask[:, : _TINY.warmup] == 0.0).all()
    assert torch.equal(coverage, torch.ones(_BATCH, _TINY.t_valid))


def test_coverage_frac_reports_the_valid_fraction_of_the_future_window():
    _, coverage = forecast_mask(_weight((10,)), _TINY)
    # Anchors 6..9 have one of their four future steps gapped; every other window is whole.
    expected = torch.ones(_BATCH, _TINY.t_valid)
    expected[:, 6:10] = 0.75
    assert torch.equal(coverage, expected)


def test_an_anchor_below_the_coverage_floor_is_zeroed_entirely():
    mask, _ = forecast_mask(_weight((10,)), _TINY, coverage_floor=0.8)
    assert (mask[:, 6:10] == 0.0).all()                   # 0.75 < 0.8: whole anchors dropped
    assert torch.equal(mask[:, 4:6], torch.ones(_BATCH, 2, _TINY.horizon))


def test_a_zero_coverage_floor_reproduces_the_per_step_behaviour_exactly():
    weight = _weight((3, 10))
    floored, _ = forecast_mask(weight, _TINY, coverage_floor=0.0)
    plain, _ = forecast_mask(weight, _TINY)
    assert torch.equal(floored, plain)


def _kl_mask_of(weight: torch.Tensor, *, coverage_floor: float = 0.0) -> torch.Tensor:
    """The KL mask for a weight, through the forecast mask that now defines its support."""
    forecast, _ = forecast_mask(weight, _TINY, coverage_floor=coverage_floor)
    return kl_mask(forecast, _TINY)


def test_kl_mask_is_zero_outside_the_decoded_anchor_support():
    """KL charged on the tail H anchors -- which have no reconstruction term -- would be
    regularised onto the prior for free, an end-of-sequence droop resembling fading coupling."""
    mask = _kl_mask_of(_weight())
    assert mask.shape == (_BATCH, _TINY.t)
    assert (mask[:, : _TINY.warmup] == 0.0).all()
    assert (mask[:, _TINY.t_valid :] == 0.0).all()
    assert (mask[:, _TINY.warmup : _TINY.t_valid] == 1.0).all()


def test_kl_mask_drops_gapped_anchors_inside_the_support():
    mask = _kl_mask_of(_weight((10,)))
    assert (mask[:, 10] == 0.0).all()
    assert (mask[:, 9] == 1.0).all()


def test_the_kl_support_is_exactly_the_anchors_the_reconstruction_scores():
    """The invariant the KL mask exists to hold: an anchor charged beta*KL but carrying no
    reconstruction term has nothing pulling the posterior off the prior, so it is regularised
    onto it for free. Those anchors cluster just before every gap, where the resulting KL
    suppression reads as coupling fading rather than as a masking artifact.
    """
    weight = _weight((8, 9, 10))
    forecast, _ = forecast_mask(weight, _TINY, coverage_floor=0.5)
    support = kl_mask(forecast, _TINY)

    scored = (forecast.amax(dim=-1) > 0).to(support.dtype)
    assert torch.equal(support[:, : _TINY.t_valid], scored)
    assert (support[:, _TINY.t_valid :] == 0.0).all()
    # Both loss terms must therefore average over one anchor count, not two.
    assert float(support.sum()) == float(scored.sum())


def test_the_coverage_floor_removes_an_anchor_from_the_kl_as_well():
    """Without this the floor is one-sided: the anchor leaves the reconstruction but keeps
    paying beta*KL, which is strictly worse than leaving it in both."""
    weight = _weight((8, 9, 10))
    lenient = _kl_mask_of(weight, coverage_floor=0.0)
    strict = _kl_mask_of(weight, coverage_floor=0.9)

    assert float(strict.sum()) < float(lenient.sum())
    assert (strict <= lenient).all()


def test_a_forecast_mask_shaped_wrongly_is_rejected():
    """Passing `weight` where the forecast mask belongs would silently restore the old,
    wider KL support, so the shape guard names what the argument actually is."""
    with pytest.raises(ValueError, match="forecast mask"):
        kl_mask(_weight(), _TINY)


def test_a_fractional_weight_is_not_valid():
    """The >= 1.0 threshold: a partially valid step still contains sentinel raw samples at
    roughly -11 sigma, and a summed 480-sample NLL would be dominated by them."""
    weight = _weight()
    weight[:, 5] = 0.5
    mask, _ = forecast_mask(weight, _TINY)
    assert (mask[:, 5] == 0.0).all()                      # as an anchor
    assert (mask[:, 4, 0] == 0.0).all()                   # as a future step (t=4, tau=0 -> 5)
    assert (_kl_mask_of(weight)[:, 5] == 0.0).all()


def test_a_wrong_weight_length_is_rejected_naming_the_trim_requirement():
    with pytest.raises(ValueError, match="trim_minutes"):
        forecast_mask(torch.ones(_BATCH, _TINY.t + 30), _TINY)


def test_a_non_2d_weight_is_rejected():
    with pytest.raises(ValueError, match="2-D"):
        forecast_mask(torch.ones(_TINY.t), _TINY)


# ---------------------------------------------------------------------------------------
# An explicit anchor set
#
# Omitting one means the dense range [0, T_valid) -- deliberately not [warmup, T_valid), because
# the warm-up is a *factor* of the mask and not a restriction of its axis. Supplying the full
# range must therefore reproduce the dense result element for element, which is the property that
# makes the argument inert for every model that does not use it.
# ---------------------------------------------------------------------------------------
def _dense_anchors(batch: int = _BATCH) -> torch.Tensor:
    """The full anchor range as an explicit index, one row per sample."""
    return torch.arange(_TINY.t_valid).expand(batch, -1).contiguous()


@pytest.mark.parametrize("gaps", [(), (10,), (3, 10)])
def test_the_full_range_supplied_explicitly_is_the_dense_result(gaps):
    weight = _weight(gaps)
    dense_mask, dense_coverage = forecast_mask(weight, _TINY)
    anchored_mask, anchored_coverage = forecast_mask(
        weight, _TINY, anchors=_dense_anchors()
    )

    assert torch.equal(anchored_mask, dense_mask)
    assert torch.equal(anchored_coverage, dense_coverage)
    assert torch.equal(
        kl_mask(anchored_mask, _TINY, anchors=_dense_anchors()),
        kl_mask(dense_mask, _TINY),
    )


def test_a_gathered_anchor_set_is_the_dense_rows_it_names():
    """Three anchors out of twelve: each row of the gathered mask is the dense mask's row at that
    anchor, which is what makes the gather a *selection* rather than a second construction."""
    weight = _weight((10,))
    dense, dense_coverage = forecast_mask(weight, _TINY)
    chosen = torch.tensor([[2, 6, 9], [4, 7, 11]])

    mask, coverage = forecast_mask(weight, _TINY, anchors=chosen)

    assert mask.shape == (_BATCH, 3, _TINY.horizon)
    for row in range(_BATCH):
        for slot, anchor in enumerate(chosen[row].tolist()):
            assert torch.equal(mask[row, slot], dense[row, anchor])
            assert torch.equal(coverage[row, slot], dense_coverage[row, anchor])


def test_the_warmup_still_zeroes_an_anchor_supplied_below_it():
    """The warm-up is a factor, not a range: naming anchor 0 explicitly does not buy it in."""
    mask, _ = forecast_mask(_weight(), _TINY, anchors=torch.tensor([[0, 1, 2], [0, 1, 2]]))

    assert (mask[:, :_TINY.warmup] == 0.0).all()
    assert (mask[:, _TINY.warmup] == 1.0).all()


def test_a_padded_slot_contributes_to_nothing():
    """The padding convention: a short row repeats its last valid anchor and marks it invalid.

    Without ``anchor_valid`` multiplied into the mask that row would be fully live, so the repeated
    anchor's target block would be gathered and scored twice while the KL support -- a set --
    counted it once, and the two per-anchor denominators would diverge.
    """
    anchors = torch.tensor([[3, 5, 5], [3, 5, 7]])
    valid = torch.tensor([[True, True, False], [True, True, True]])

    mask, _ = forecast_mask(_weight(), _TINY, anchors=anchors, anchor_valid=valid)
    support = kl_mask(mask, _TINY, anchors=anchors, anchor_valid=valid)

    assert (mask[0, 2] == 0.0).all()
    assert (contributing_anchors(mask)[0] == torch.tensor([1.0, 1.0, 0.0])).all()
    # The repeated anchor is still supported once: the scatter reduces by maximum, so the padded
    # slot's zero cannot land on top of the real slot's one.
    assert float(support[0, 5]) == 1.0
    assert float(support.sum()) == 5.0


def test_the_kl_support_is_scattered_to_the_anchors_own_positions():
    """Position by position, against a hand-written three-anchor example."""
    anchors = torch.tensor([[2, 6, 9], [2, 6, 9]])
    mask, _ = forecast_mask(_weight(), _TINY, anchors=anchors)

    support = kl_mask(mask, _TINY, anchors=anchors)

    expected = torch.zeros(_BATCH, _TINY.t)
    expected[:, [2, 6, 9]] = 1.0
    assert support.shape == (_BATCH, _TINY.t)
    assert torch.equal(support, expected)


def test_a_gapped_anchor_is_absent_from_the_scattered_support():
    """The support is derived, not restated: anchor 10 is named but its own step is invalid."""
    anchors = torch.tensor([[9, 10, 11], [9, 10, 11]])
    mask, _ = forecast_mask(_weight((10,)), _TINY, anchors=anchors)

    support = kl_mask(mask, _TINY, anchors=anchors)

    assert float(support[0, 10]) == 0.0
    assert float(support[0, 9]) == 1.0


def test_an_anchor_at_or_past_t_valid_is_refused_naming_it():
    """Out of range means past $T_{\\mathrm{valid}}$, not past $T$: an anchor in the tail has no
    fully observed window, and scattering there would write into the region the dense form
    guarantees is zero."""
    with pytest.raises(ValueError, match=r"anchor 12 .*\[0, 12\)"):
        forecast_mask(_weight(), _TINY, anchors=torch.tensor([[2, 12], [2, 3]]))


def test_a_repeated_valid_anchor_is_refused_naming_it():
    with pytest.raises(ValueError, match="anchor 5 appears twice"):
        forecast_mask(_weight(), _TINY, anchors=torch.tensor([[5, 5], [2, 3]]))


def test_a_float_anchor_tensor_is_refused():
    with pytest.raises(ValueError, match="integer tensor"):
        forecast_mask(_weight(), _TINY, anchors=torch.tensor([[2.0, 3.0], [2.0, 3.0]]))


def test_contributing_anchors_refuses_a_mask_of_the_wrong_rank():
    """The one failure in this module whose symptom is a wrong number rather than an exception:
    the reduction runs on the last axis alone, so an extra axis inflates every denominator built
    from the result by that axis's length and nothing downstream raises."""
    mask, _ = forecast_mask(_weight(), _TINY)

    with pytest.raises(ValueError, match=r"3-D \(B, A, H\)"):
        contributing_anchors(mask[..., None])


def test_a_kl_mask_shaped_for_a_different_anchor_set_is_rejected():
    """The guard that names what the argument is, kept meaningful under a gathered axis."""
    anchors = torch.tensor([[2, 6, 9], [2, 6, 9]])
    mask, _ = forecast_mask(_weight(), _TINY, anchors=anchors)

    with pytest.raises(ValueError, match="forecast mask"):
        kl_mask(mask, _TINY)  # the dense form expects T_valid rows, not 3
