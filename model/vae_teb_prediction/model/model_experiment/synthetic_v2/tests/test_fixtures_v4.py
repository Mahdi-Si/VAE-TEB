r"""S0-T06: validate the fabricated ``synthetic_v4`` fixtures + the ``v4`` marker.

These exercise the estimator/gate math on known-answer fabrications (no model training), so the
Sprint 6/7 grading code can be unit-tested before any headline run.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.v4


def test_v4_marker_is_registered(pytestconfig) -> None:
    r"""The ``v4`` marker is registered (so ``-m v4`` selection works without a warning)."""
    markers = pytestconfig.getini("markers")
    assert any(line.startswith("v4:") for line in markers), "v4 marker not registered"


def test_signal_kbar_is_linear_with_requested_slope(signal_kbar_fixture) -> None:
    r"""$\bar K$ is linear in te with the requested slope $\gamma$ (validates the calibrator input)."""
    data = signal_kbar_fixture(gamma=1.3, noise=0.01, reps=100, seed=0)
    kbar, te = data["kbar"], data["te_true"]
    slope, intercept = np.polyfit(te, kbar, 1)
    assert slope == pytest.approx(1.3, abs=0.05)
    assert intercept == pytest.approx(0.0, abs=0.05)


def test_signal_kbar_null_level_is_pure_noise(signal_kbar_fixture) -> None:
    r"""At te=0 (the null cells) $\bar K$ has no signal component -- only the injected noise."""
    data = signal_kbar_fixture(gamma=1.0, noise=0.02, reps=200, seed=1)
    kbar, te = data["kbar"], data["te_true"]
    null_kbar = kbar[te == 0.0]
    assert abs(float(null_kbar.mean())) < 0.02


def test_source_exploiting_prediction_ordering(source_exploiting_outputs) -> None:
    r"""Prediction-space ordering holds by construction: feat < base < feat^{pi(U)}."""
    target = source_exploiting_outputs["target"]
    clean = source_exploiting_outputs["clean"]
    permuted = source_exploiting_outputs["permuted"]

    def _mse(pred):
        return float(np.mean((pred - target) ** 2))

    l_feat = _mse(clean["mu_full"])
    l_base = _mse(clean["mu_base"])
    l_feat_shuffled = _mse(permuted["mu_full"])
    assert l_feat < l_base < l_feat_shuffled


def test_planted_lag_recovers_argmax(planted_lag_te_lag_map) -> None:
    r"""``argmax_l mean_{b,t} te_lag_map`` recovers the planted lag $D$."""
    te_lag_map = planted_lag_te_lag_map["te_lag_map"]
    planted = planted_lag_te_lag_map["planted_lag"]
    lag_profile = te_lag_map.mean(axis=(0, 1))          # (L,)
    assert int(np.argmax(lag_profile)) == planted


def test_te_lag_map_sum_is_kld_per_t(planted_lag_te_lag_map) -> None:
    r"""The lag-sum identity holds: $\bar K_t = \sum_\ell \mathrm{te\_lag\_map}_{t,\ell}$."""
    te_lag_map = planted_lag_te_lag_map["te_lag_map"]
    kld_per_t = planted_lag_te_lag_map["kld_per_t"]
    np.testing.assert_allclose(te_lag_map.sum(axis=2), kld_per_t, rtol=1e-6)


def test_tiny_raw_checkpoint_builds(tiny_raw_checkpoint) -> None:
    r"""The re-used ``model_raw`` tiny checkpoint builds and carries its model_kwargs."""
    path, kwargs = tiny_raw_checkpoint
    assert path.is_file()
    assert kwargs["raw_len"] == 512 and kwargs["decimation"] == 16
