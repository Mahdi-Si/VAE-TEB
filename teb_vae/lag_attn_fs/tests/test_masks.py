r"""The forecast and KL masks are reused unchanged against a feature-shaped block.

``lag_attn_rws/nets/raw_masks.py`` is named for a raw target and is not about one.
:func:`forecast_mask`, :func:`contributing_anchors` and :func:`kl_mask` read only ``geometry.t``,
``geometry.t_valid``, ``geometry.horizon`` and ``geometry.warmup`` -- never ``raw_len``,
``decimation`` or ``r`` -- and emit $(B, T_{\mathrm{valid}}, H_d)$ and $(B, T)$ masks that carry
no channel axis at all. A block of $78$ feature coefficients per horizon token is masked by
exactly the same tensor as a block of $16$ raw samples, broadcast over its last axis.

That is a claim, not a convenience, and this file is where it is checked rather than assumed: the
tests below drive the shipped masks with a geometry whose raw grid differs while its decimated
grid does not, and mask a feature block at two different channel counts with one mask.

What the masks *mean* -- the $\ge 1.0$ validity threshold, the coverage floor, the derivation of
the KL support from the forecast mask rather than from ``weight`` -- is pinned in the package that
owns them. Restating it here would be a second copy of one piece of evidence.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_fs.tests.conftest import SHIPPED_KWARGS
from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry
from teb_vae.lag_attn_rws.nets.raw_masks import (
    contributing_anchors,
    forecast_mask,
    kl_mask,
)

_BATCH = 2

#: The production geometry: $T = 300$, $H_d = 30$, $w = 30$, so $T_{\mathrm{valid}} = 270$.
_GEOMETRY = TrimmedRawGeometry(
    raw_len=SHIPPED_KWARGS["sequence_length"] * SHIPPED_KWARGS["raw_per_step"],
    decimation=SHIPPED_KWARGS["raw_per_step"],
    horizon=SHIPPED_KWARGS["horizon"],
    warmup=SHIPPED_KWARGS["warmup_period"],
)

#: The same decimated grid reached from a different raw one -- half the samples at half the
#: decimation. Every quantity the masks are allowed to read is identical; ``raw_len``,
#: ``decimation`` and ``r`` are not. A mask that differs between the two is reading the raw grid.
_HALF_RAW_GEOMETRY = TrimmedRawGeometry(
    raw_len=SHIPPED_KWARGS["sequence_length"] * SHIPPED_KWARGS["raw_per_step"] // 2,
    decimation=SHIPPED_KWARGS["raw_per_step"] // 2,
    horizon=SHIPPED_KWARGS["horizon"],
    warmup=SHIPPED_KWARGS["warmup_period"],
)

#: Surviving and declared target channel counts. Both are exercised, because "the mask broadcasts
#: over the channel axis" is only worth asserting if it is asserted at more than one width.
_KEPT_CHANNELS = 78
_ALL_CHANNELS = 109

#: A gap planted well inside the trained-anchor range $[30, 270)$, far enough from both ends that
#: the anchors it removes are removed by the gap rather than by the warm-up or the tail.
_GAP_STEP = 150


def _weight(gap_steps: tuple = ()) -> torch.Tensor:
    """A fully valid decimated weight with the given steps zeroed.

    Args:
        gap_steps: Decimated steps to mark invalid.

    Returns:
        The weight $(B, T)$.
    """
    weight = torch.ones(_BATCH, _GEOMETRY.t)
    for step in gap_steps:
        weight[:, step] = 0.0
    return weight


# ---------------------------------------------------------------------------------------
# Shapes: no channel axis anywhere
# ---------------------------------------------------------------------------------------
def test_the_masks_have_the_shapes_a_feature_block_needs():
    """$(B, 270, 30)$ and $(B, 300)$ -- the same shapes the raw model gets, because validity is a
    property of the decimated step and of nothing the two targets differ in."""
    mask, coverage = forecast_mask(_weight(), _GEOMETRY, coverage_floor=0.9)

    assert mask.shape == (_BATCH, _GEOMETRY.t_valid, _GEOMETRY.horizon)
    assert coverage.shape == (_BATCH, _GEOMETRY.t_valid)
    assert contributing_anchors(mask).shape == (_BATCH, _GEOMETRY.t_valid)
    assert kl_mask(mask, _GEOMETRY).shape == (_BATCH, _GEOMETRY.t)


def test_the_masks_never_consult_the_raw_grid():
    """The reason ``raw_masks`` is reusable unchanged rather than reusable by inspection. Two
    geometries agreeing on $(T, T_{\\mathrm{valid}}, H_d, w)$ and differing in ``raw_len``,
    ``decimation`` and ``r`` must produce bitwise identical masks."""
    assert _HALF_RAW_GEOMETRY.r != _GEOMETRY.r
    assert _HALF_RAW_GEOMETRY.t == _GEOMETRY.t
    weight = _weight((_GAP_STEP,))

    mask, coverage = forecast_mask(weight, _GEOMETRY, coverage_floor=0.9)
    other_mask, other_coverage = forecast_mask(weight, _HALF_RAW_GEOMETRY, coverage_floor=0.9)

    assert torch.equal(mask, other_mask)
    assert torch.equal(coverage, other_coverage)
    assert torch.equal(kl_mask(mask, _GEOMETRY), kl_mask(other_mask, _HALF_RAW_GEOMETRY))


# ---------------------------------------------------------------------------------------
# Broadcasting over the channel axis
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("channels", [_KEPT_CHANNELS, _ALL_CHANNELS])
def test_one_mask_gates_a_block_of_any_channel_count(channels):
    """The gated and ungated arms differ in $C$ and share the mask; so would a third budget."""
    block = torch.randn(_BATCH, _GEOMETRY.t_valid, _GEOMETRY.horizon, channels)
    mask, _ = forecast_mask(_weight((_GAP_STEP,)), _GEOMETRY, coverage_floor=0.9)

    gated = block * mask[..., None]

    assert gated.shape == block.shape
    assert torch.equal(gated[mask.bool()], block[mask.bool()])
    assert float(gated[~mask.bool()].abs().sum()) == 0.0


def test_the_mask_treats_every_channel_alike():
    """Validity is constant across channels because it is a property of the decimated step, so a
    masked position is masked in all $78$ coefficients rather than in some of them. A per-channel
    mask would leave a partially scored horizon token, whose summed NLL is not comparable to a
    whole one's."""
    block = torch.randn(_BATCH, _GEOMETRY.t_valid, _GEOMETRY.horizon, _KEPT_CHANNELS)
    mask, _ = forecast_mask(_weight((_GAP_STEP,)), _GEOMETRY, coverage_floor=0.9)

    surviving = (block * mask[..., None] != 0.0).sum(dim=-1)

    assert set(surviving.unique().tolist()) <= {0, _KEPT_CHANNELS}


def test_the_block_denominator_is_the_masked_coefficient_count():
    r"""What the objective divides by: $H_d \cdot C = 2340$ per contributing anchor, against the
    raw model's $H_d \cdot R = 480$. The factor of $4.9$ between them is why $\beta$ is
    recalibrated rather than inherited."""
    mask, _ = forecast_mask(_weight(), _GEOMETRY, coverage_floor=0.9)
    anchors = float(contributing_anchors(mask).sum())

    elements = float(mask.sum()) * _KEPT_CHANNELS

    assert elements == anchors * _GEOMETRY.horizon * _KEPT_CHANNELS
    assert _GEOMETRY.horizon * _KEPT_CHANNELS == 2340


# ---------------------------------------------------------------------------------------
# A planted gap
# ---------------------------------------------------------------------------------------
def test_a_planted_gap_removes_exactly_the_expected_anchors_and_steps():
    r"""A gap at step $g$ zeroes anchor $g$ itself -- its own present block is invalid -- and
    forecast step $\tau = g - 1 - t$ of every anchor $t \in [g - H_d, g)$, whose window covers it.
    Everything else outside the warm-up prefix survives."""
    mask, _ = forecast_mask(_weight((_GAP_STEP,)), _GEOMETRY)

    expected = torch.ones(_BATCH, _GEOMETRY.t_valid, _GEOMETRY.horizon)
    expected[:, : _GEOMETRY.warmup] = 0.0
    expected[:, _GAP_STEP] = 0.0
    for anchor in range(_GAP_STEP - _GEOMETRY.horizon, _GAP_STEP):
        expected[:, anchor, _GAP_STEP - 1 - anchor] = 0.0

    assert torch.equal(mask, expected)


def test_the_coverage_floor_drops_the_partially_covered_anchors_whole():
    r"""At $H_d = 30$ a single gapped step costs an anchor $1/30$ of its window, so the shipped
    ``coverage_floor: 0.9`` is *not* what removes it -- $29/30 = 0.967$ clears the floor. The
    thirty anchors reaching into a **three-step** gap fall to $0.9$ and are kept; a four-step gap
    is what drops them. Pinned because the floor's bite depends on $H_d$, and a reader coming from
    the tiny geometry ($H_d = 4$, where one gapped step costs $0.25$) will expect otherwise.
    """
    lenient, _ = forecast_mask(_weight((_GAP_STEP,)), _GEOMETRY, coverage_floor=0.9)
    reaching = range(_GAP_STEP - _GEOMETRY.horizon, _GAP_STEP)
    assert float(lenient[:, reaching].sum()) > 0.0

    # Three gapped steps: 27/30 = 0.9 exactly, and the floor comparison is inclusive.
    narrow, coverage = forecast_mask(
        _weight(tuple(range(_GAP_STEP, _GAP_STEP + 3))), _GEOMETRY, coverage_floor=0.9
    )
    assert float(coverage[0, _GAP_STEP - 1]) == pytest.approx(27.0 / 30.0)
    assert float(narrow[:, _GAP_STEP - 1].sum()) > 0.0

    # Four: 26/30 = 0.867, and the anchor leaves the reconstruction whole rather than partly.
    wide, coverage = forecast_mask(
        _weight(tuple(range(_GAP_STEP, _GAP_STEP + 4))), _GEOMETRY, coverage_floor=0.9
    )
    assert float(coverage[0, _GAP_STEP - 1]) == pytest.approx(26.0 / 30.0)
    assert float(wide[:, _GAP_STEP - 1].sum()) == 0.0


def test_the_kl_support_is_exactly_the_anchors_the_reconstruction_scores():
    """Inherited, not restated: the KL mask is derived from the forecast mask, so an anchor the
    feature reconstruction does not score is not charged $\\beta\\,\\mathrm{KL}$ either. Without
    it those anchors -- which cluster immediately before every gap -- are regularised onto the
    prior for free, and the artifact reads as coupling fading where the signal degrades."""
    mask, _ = forecast_mask(_weight((_GAP_STEP,)), _GEOMETRY, coverage_floor=0.9)
    support = kl_mask(mask, _GEOMETRY)
    scored = contributing_anchors(mask)

    assert torch.equal(support[:, : _GEOMETRY.t_valid], scored)
    assert (support[:, _GEOMETRY.t_valid :] == 0.0).all()
    assert (support[:, _GAP_STEP] == 0.0).all()
    # Both loss terms therefore average over one anchor count, not two.
    assert float(support.sum()) == float(scored.sum())


def test_the_tail_and_the_warmup_carry_no_kl():
    """The last $H_d$ steps have no fully observed window and the first $w$ are discarded, so
    neither is charged KL. $270 - 30 = 240$ anchors remain on a gap-free segment."""
    mask, _ = forecast_mask(_weight(), _GEOMETRY)
    support = kl_mask(mask, _GEOMETRY)

    assert (support[:, : _GEOMETRY.warmup] == 0.0).all()
    assert (support[:, _GEOMETRY.t_valid :] == 0.0).all()
    assert float(support.sum()) == _BATCH * (_GEOMETRY.t_valid - _GEOMETRY.warmup)
