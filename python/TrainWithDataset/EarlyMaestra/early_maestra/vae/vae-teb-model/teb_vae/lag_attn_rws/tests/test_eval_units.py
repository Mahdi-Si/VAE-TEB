r"""The two z-to-bpm conversions, and why the wrong one is the plausible-looking one.

The loader z-scores FHR, so every number this pipeline computes is in $z$ units until something
converts it. There are **two** conversions and they are not interchangeable:

* a **level** -- a forecast, a truth curve, a band edge -- is affine: $x\,(s + \varepsilon) + m$;
* a **spread** -- a standard deviation, an RMSE, an MAE, a mean signed error -- is a difference of
  levels, so $m$ cancels and only the scale survives: $x\,(s + \varepsilon)$.

Putting a spread through the level map is the failure this module exists to prevent, and it is
dangerous precisely because it is quiet: an RMSE of $0.1$ z-units is about $1$ bpm, and the affine
map turns it into $141$ bpm -- a physiologically reasonable fetal heart rate, and therefore a
number nobody questions. There is no shape check, no unit check and no assertion that catches it
downstream.

The third property is the honest fallback: without the loader's statistics the numbers stay in
$z$ units under the ``normalised`` label rather than being relabelled as something they are not.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from teb_vae.lag_attn_rws.eval.metrics import (
    BPM_UNIT,
    NORMALISED_UNIT,
    fhr_normalization,
    sigma_to_bpm,
    to_bpm,
)

#: Round numbers, so every expectation below is arithmetic a reader can do in their head.
HAND_STATS = {"fhr": {"mean": 140.0, "std": 10.0}}


# =============================================================================
# The level conversion
# =============================================================================
def test_a_level_round_trips_through_the_loaders_own_statistics(multi_class_loader) -> None:
    """Against the real statistics file, through the real loader, rather than a hand-built dict.

    The committed statistics are what the shards were normalised with, so a conversion that
    round-trips against them is a conversion that would put a real forecast back in bpm.

    To float32, not to float64: the loader caches its two constants as float32 tensors and
    ``denormalize_signal_data`` reads those in preference to the float64 scalars beside them, so
    the round trip carries the loader's own precision rather than the statistics file's.
    """
    stats = multi_class_loader.dataset.get_normalization_stats()
    resolved = fhr_normalization(stats)
    assert resolved is not None
    mean, scale = resolved
    original = np.array([120.0, 140.0, 165.5], dtype=np.float64)

    recovered, unit = to_bpm((original - mean) / scale, stats)

    assert unit == BPM_UNIT
    assert recovered == pytest.approx(original, rel=1e-6)


def test_the_level_conversion_is_the_repositorys_own_denormaliser(multi_class_loader) -> None:
    """Identity of arithmetic, not merely of result: a second copy of the two constants would
    drift the moment the statistics format grew a third."""
    from train.graph_models_utils import denormalize_signal_data

    stats = multi_class_loader.dataset.get_normalization_stats()
    values = np.linspace(-3.0, 3.0, 7)

    converted, _unit = to_bpm(values, stats)
    reference = denormalize_signal_data(torch.as_tensor(values), "fhr", stats)

    assert converted == pytest.approx(reference.numpy(), rel=1e-12)


# =============================================================================
# The spread conversion
# =============================================================================
def test_a_spread_scales_by_the_standard_deviation_and_is_not_offset() -> None:
    r"""Hand-computed: at $s = 10$ an RMSE of $0.25$ z-units is $2.5$ bpm, not $142.5$."""
    converted, unit = sigma_to_bpm([0.25, 1.0], HAND_STATS)

    assert unit == BPM_UNIT
    # 10.0 + 1e-8, so the comparison is against the same epsilon the level conversion applies.
    assert converted == pytest.approx([2.5, 10.0], rel=1e-8)


def test_the_two_conversions_differ_by_exactly_the_mean() -> None:
    """The non-vacuity check. Both functions are one multiplication apart, so a test that only
    asserted "it returned a number" would pass with the two swapped."""
    spread, _ = sigma_to_bpm([0.25], HAND_STATS)
    level, _ = to_bpm([0.25], HAND_STATS)

    assert float(level[0] - spread[0]) == pytest.approx(HAND_STATS["fhr"]["mean"], rel=1e-9)


def test_a_signed_bias_converts_as_a_spread_because_it_is_a_difference() -> None:
    """A mean forecast error of zero is zero bpm of bias, not 140 bpm of it."""
    converted, _unit = sigma_to_bpm([0.0], HAND_STATS)

    assert float(converted[0]) == 0.0


# =============================================================================
# The honest fallback
# =============================================================================
@pytest.mark.parametrize("stats", [None, {}, {"up": {"mean": 30.0, "std": 10.0}}, {"fhr": {}}])
def test_absent_statistics_leave_the_values_in_z_units_under_the_normalised_label(stats) -> None:
    """Every shape of "we do not know": no dict, an empty one, one for another field, and one
    whose ``fhr`` entry carries neither constant."""
    assert fhr_normalization(stats) is None

    level, level_unit = to_bpm([1.0, 2.0], stats)
    spread, spread_unit = sigma_to_bpm([1.0, 2.0], stats)

    assert level_unit == spread_unit == NORMALISED_UNIT
    assert level == pytest.approx([1.0, 2.0])
    assert spread == pytest.approx([1.0, 2.0])


def test_the_conversions_accept_arrays_of_any_shape() -> None:
    """The callers hand these a scalar, a three-element list and a length-$H$ curve."""
    curve = np.linspace(0.0, 1.0, 30).reshape(5, 6)

    converted, _unit = sigma_to_bpm(curve, HAND_STATS)

    assert converted.shape == curve.shape
