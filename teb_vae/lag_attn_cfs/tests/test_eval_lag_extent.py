"""The lag-extent analysis's index arithmetic: each test fails on an off-by-one that would shift
every reported lag without raising."""
import numpy as np

from teb_vae.lag_attn_cfs.eval.analyses.lag_extent import (
    contraction_lags,
    effect_lags,
    mass_quantile,
    support_extent,
)


def test_support_and_quantiles_name_the_right_lags():
    weights = np.zeros((2, 10))
    weights[0, [2, 3, 7]] = [0.5, 0.3, 0.2]  # support 2..7; 50 % reached at lag 2, 90 % at lag 7
    first, last = support_extent(weights)
    assert first[0] == 2 and last[0] == 7
    assert np.isnan(first[1]) and np.isnan(last[1])  # an all-zero row has no support
    assert mass_quantile(weights, 0.5)[0] == 2
    assert mass_quantile(weights, 0.8)[0] == 3
    assert mass_quantile(weights, 0.9)[0] == 7
    assert np.isnan(mass_quantile(weights, 0.5)[1])


def test_truncation_crossings_interpolate_on_the_grid():
    grid = np.array([0, 2, 4, 8])
    # no-key arm 0, full 10: keep <= k recovers 4, 8, 9.5, 10; keep >= k keeps 10, 5, 1, 0.
    lags = effect_lags(0.0, np.array([4.0, 8.0, 9.5, 10.0]), np.array([10.0, 5.0, 1.0, 0.0]), grid)
    assert np.isclose(lags["effect_mid_lag"], 0.5)  # 0.4 -> 0.8 between lags 0 and 2
    assert np.isclose(lags["effect_end_lag"], 2.0 + 2.0 * (0.9 - 0.8) / (0.95 - 0.8))
    assert np.isclose(lags["effect_start_lag"], 0.0 + 2.0 * (0.9 - 1.0) / (0.5 - 1.0))
    # No effect to locate: every readout is NaN rather than a lag.
    flat = effect_lags(0.0, np.zeros(4), np.zeros(4), grid)
    assert np.isnan(flat["effect_end_lag"]) and np.isnan(flat["effect_start_lag"])


def test_contraction_body_maps_to_lags_behind_the_anchor():
    decimation = 16
    onset, end = np.array([160]), np.array([400])  # steps 10 .. 25 on the raw grid
    anchors = np.array([5, 10, 20, 40])
    geo = contraction_lags(onset, end, anchors, decimation)
    assert geo["index"][0] == -1 and np.isnan(geo["age_s"][0])  # before the onset: none behind it
    assert (geo["lo"][1], geo["hi"][1]) == (0, 0)  # the onset step itself, at lag 0
    assert (geo["lo"][2], geo["hi"][2]) == (0, 10)  # still running: lags 0 .. anchor - onset
    assert (geo["lo"][3], geo["hi"][3]) == (15, 30)  # finished: 40 - 25 .. 40 - 10
    assert np.isclose(geo["age_s"][3], (decimation * 41 - 1 - 160) / 4.0)
