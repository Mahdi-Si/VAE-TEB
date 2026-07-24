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
from teb_vae.lag_attn_rws.nets.raw_masks import VALID_THRESHOLD, forecast_mask, kl_mask
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
