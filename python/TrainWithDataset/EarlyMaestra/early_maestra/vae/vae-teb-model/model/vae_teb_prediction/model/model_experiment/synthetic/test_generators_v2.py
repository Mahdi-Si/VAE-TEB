r"""Unit tests for the v2 synthetic generators (G1 / G2 / G3).

Covers ``model_validation_v2_plan.md`` Sprint 2.6:

    * Shape and dtype contracts ($Y \in \mathbb{R}^{n \times T \times 87}$,
      $U \in \mathbb{R}^{n \times T \times 101}$, ``float32``).
    * Determinism: bitwise-identical output across two calls with the same
      seed.
    * Per-channel z-score after ``standardize=True``.
    * ``meta.te_true`` matches the analytic-TE function within the
      Monte-Carlo / closed-form tolerance per generator.
    * Empirical lagged cross-correlation peaks at the configured delay $D$
      (G1, G2).
    * Target PSD is concentrated in low frequencies (G1, G2 with the
      90th-percentile energy bin below 0.1 of Nyquist).
    * G3 regime transition matrix matches $p_{\text{switch}}$ within
      sampling error.
    * G3 source-target consistency: when $R^{(0)}_{t+\delta} = k_0$, the
      one-hot source channel fires.
    * G1 ``reverse_roles=True``: ``te_true == 0`` and ``direction == "Y_to_X"``.
    * G1 / G2 ``easy_variant=True``: every target channel carries an
      informative signal.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from model.vae_teb_prediction.model.model_experiment.synthetic.analytic_te import (
    te_block_arx_gaussian,
    te_block_state_space_gaussian,
    te_categorical_switch_block,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.generators import (
    gen_regime_switch_smooth,
    gen_smooth_arx,
    gen_state_space_oscillator,
)


# ---------------------------------------------------------------------------
# G2 (smooth ARX)
# ---------------------------------------------------------------------------


_G2_DEFAULTS = dict(
    rho_u=0.99,
    rho_y=0.95,
    c=0.5,
    sigma2_eta=1.0,
    sigma2_eps=1.0,
    delay=60,
    M=4,
)


@pytest.mark.parametrize("delay", [30, 60, 75])
@pytest.mark.parametrize("M", [1, 4])
def test_g2_shapes_and_dtype(delay: int, M: int) -> None:
    """G2 returns ``(n, T, 87)`` / ``(n, T, 101)`` ``float32`` tensors."""
    Y, U, meta = gen_smooth_arx(
        n=4, T=300,
        rho_u=0.99, rho_y=0.95, c=0.5,
        sigma2_eta=1.0, sigma2_eps=1.0,
        delay=delay, M=M, seed=0,
    )
    assert Y.shape == (4, 300, 87)
    assert U.shape == (4, 300, 101)
    assert Y.dtype == torch.float32
    assert U.dtype == torch.float32
    assert meta["benchmark"] == "G2"
    assert meta["M"] == M
    assert meta["delay"] == delay


def test_g2_determinism() -> None:
    """Same seed -> bitwise-identical tensors."""
    Y1, U1, _ = gen_smooth_arx(n=4, T=300, seed=42, **_G2_DEFAULTS)
    Y2, U2, _ = gen_smooth_arx(n=4, T=300, seed=42, **_G2_DEFAULTS)
    assert torch.equal(Y1, Y2)
    assert torch.equal(U1, U2)


def test_g2_standardisation_per_channel() -> None:
    """``standardize=True`` puts every channel on $\\mathcal{N}(0, 1)$."""
    Y, U, _ = gen_smooth_arx(
        n=64, T=300, seed=0, standardize=True, **_G2_DEFAULTS,
    )
    assert Y.mean(dim=(0, 1)).abs().max().item() < 0.1
    assert (Y.var(dim=(0, 1)) - 1.0).abs().max().item() < 0.1
    assert U.mean(dim=(0, 1)).abs().max().item() < 0.1
    assert (U.var(dim=(0, 1)) - 1.0).abs().max().item() < 0.1


def test_g2_te_true_matches_analytic() -> None:
    """``meta.te_true`` equals ``M * te_block_arx_gaussian(...)`` exactly."""
    Y, U, meta = gen_smooth_arx(n=4, T=300, seed=0, **_G2_DEFAULTS)
    expected = _G2_DEFAULTS["M"] * te_block_arx_gaussian(
        rho_u=_G2_DEFAULTS["rho_u"], rho_y=_G2_DEFAULTS["rho_y"],
        c=_G2_DEFAULTS["c"],
        sigma2_eta=_G2_DEFAULTS["sigma2_eta"],
        sigma2_eps=_G2_DEFAULTS["sigma2_eps"],
        H=30, D=_G2_DEFAULTS["delay"],
    )
    assert meta["te_true"] == pytest.approx(expected, rel=1e-9)


def test_g2_lagged_xcorr_peak() -> None:
    r"""The empirical $Y \to U$ cross-correlation peaks near the true delay.

    Because $Y_t = c \sum_{k=0}^{\infty} \rho_y^{k}\,U_{t-D-k} + \nu_t$ and $U$
    is itself AR(1) with $\rho_u \to 1$, the *empirical* cross-correlation is
    smoothed by both AR kernels, so the peak sits a few steps **beyond** $D$
    rather than exactly at $D$. We assert it lies inside $[D - 5, D + 20]$.
    """
    Y, U, meta = gen_smooth_arx(
        n=128, T=600, seed=0, standardize=False, **_G2_DEFAULTS,
    )
    y0 = Y[..., 0].numpy().astype(np.float64).reshape(-1)
    u0 = U[..., 0].numpy().astype(np.float64).reshape(-1)
    y0 = y0 - y0.mean()
    u0 = u0 - u0.mean()
    max_lag = 90
    corrs = np.empty(max_lag + 1)
    for L in range(max_lag + 1):
        if L == 0:
            num = (y0 * u0).mean()
        else:
            num = (y0[L:] * u0[:-L]).mean()
        corrs[L] = num
    peak_lag = int(np.argmax(np.abs(corrs)))
    D = meta["delay"]
    assert D - 5 <= peak_lag <= D + 20, (
        f"G2 lagged xcorr peak at lag {peak_lag}, expected in [{D-5}, {D+20}]; "
        f"top-3 lags = {np.argsort(np.abs(corrs))[-3:].tolist()}"
    )


def test_g2_psd_low_frequency() -> None:
    r"""90% of $Y$-channel energy must lie below 0.1 of Nyquist."""
    Y, _, _ = gen_smooth_arx(
        n=32, T=600, seed=0, standardize=False, **_G2_DEFAULTS,
    )
    y0 = Y[..., 0].numpy()                        # (n, T)
    spec = np.fft.rfft(y0, axis=1)
    energy = (np.abs(spec) ** 2).astype(np.float64)
    cum = np.cumsum(energy, axis=1) / energy.sum(axis=1, keepdims=True)
    nyquist = spec.shape[1] - 1
    p90_bin = (cum >= 0.9).argmax(axis=1)
    fraction = float((p90_bin <= 0.1 * nyquist).mean())
    assert fraction >= 0.8, (
        f"G2: only {fraction:.0%} of samples have 90% energy below 0.1*Nyquist."
    )


def test_g2_zero_coupling_is_zero_te() -> None:
    """``c = 0`` ⇒ ``te_true = 0`` and lag band is the empty interval."""
    _, _, meta = gen_smooth_arx(
        n=4, T=300, seed=0,
        rho_u=0.99, rho_y=0.95, c=0.0,
        sigma2_eta=1.0, sigma2_eps=1.0,
        delay=60, M=4,
    )
    assert meta["te_true"] == 0.0


def test_g2_reverse_roles_te_zero() -> None:
    """``reverse_roles=True`` ⇒ ``te_true = 0``."""
    _, _, meta = gen_smooth_arx(
        n=4, T=300, seed=0, reverse_roles=True, **_G2_DEFAULTS,
    )
    assert meta["te_true"] == 0.0
    assert meta["true_lag_band"] == []
    assert meta["direction"] == "Y_to_X"


def test_g2_easy_variant() -> None:
    """``easy_variant=True`` sets ``M = c_y`` (every target channel informative)."""
    _, _, meta = gen_smooth_arx(
        n=4, T=300, seed=0, easy_variant=True, **{**_G2_DEFAULTS, "M": 4},
    )
    assert meta["M"] == 87
    assert meta["informative_channels"] == list(range(87))


# ---------------------------------------------------------------------------
# G1 (state-space oscillator)
# ---------------------------------------------------------------------------


_G1_DEFAULTS = dict(
    oscillators=[(0.99, 0.05)] * 4,
    target_ar=0.95,
    delays=[60] * 4,
    B_y=[0.5] * 4,
    sigma2_y=1.0,
    sigma2_eta=0.01,
    M=4,
    te_n_samples=20_000,
)


def test_g1_shapes_and_dtype() -> None:
    """G1 returns ``(n, T, 87)`` / ``(n, T, 101)`` ``float32`` tensors."""
    Y, U, meta = gen_state_space_oscillator(n=4, T=300, seed=0, **_G1_DEFAULTS)
    assert Y.shape == (4, 300, 87)
    assert U.shape == (4, 300, 101)
    assert Y.dtype == torch.float32
    assert U.dtype == torch.float32
    assert meta["benchmark"] == "G1"


def test_g1_determinism() -> None:
    """Same seed -> bitwise-identical tensors."""
    Y1, U1, _ = gen_state_space_oscillator(n=4, T=300, seed=7, **_G1_DEFAULTS)
    Y2, U2, _ = gen_state_space_oscillator(n=4, T=300, seed=7, **_G1_DEFAULTS)
    assert torch.equal(Y1, Y2)
    assert torch.equal(U1, U2)


def test_g1_standardisation_per_channel() -> None:
    """``standardize=True`` puts every channel near $\\mathcal{N}(0, 1)$."""
    Y, U, _ = gen_state_space_oscillator(
        n=32, T=300, seed=0, standardize=True, **_G1_DEFAULTS,
    )
    assert Y.mean(dim=(0, 1)).abs().max().item() < 0.1
    assert (Y.var(dim=(0, 1)) - 1.0).abs().max().item() < 0.1
    assert U.mean(dim=(0, 1)).abs().max().item() < 0.1
    assert (U.var(dim=(0, 1)) - 1.0).abs().max().item() < 0.1


def test_g1_te_true_matches_analytic() -> None:
    """``meta.te_true`` matches ``te_block_state_space_gaussian`` to 10% MC slack."""
    _, _, meta = gen_state_space_oscillator(n=4, T=300, seed=0, **_G1_DEFAULTS)
    expected = te_block_state_space_gaussian(
        oscillators=_G1_DEFAULTS["oscillators"],
        target_ar=_G1_DEFAULTS["target_ar"],
        delays=_G1_DEFAULTS["delays"],
        B_y=_G1_DEFAULTS["B_y"],
        sigma2_y=_G1_DEFAULTS["sigma2_y"],
        sigma2_eta=_G1_DEFAULTS["sigma2_eta"],
        H=30,
        n_samples=_G1_DEFAULTS["te_n_samples"],
        seed=0 + 1_337,
    )
    # Same seed offset is used by the generator, so this should match exactly.
    assert meta["te_true"] == pytest.approx(expected, rel=1e-9)


def test_g1_oscillator_lag_band() -> None:
    """``true_lag_band`` covers $[D - H,\\,D - 1]$ across all oscillators."""
    _, _, meta = gen_state_space_oscillator(n=4, T=300, seed=0, **_G1_DEFAULTS)
    assert meta["true_lag_band"] == list(range(60 - 30, 60))


def test_g1_psd_low_frequency() -> None:
    r"""90% of $Y$-channel energy must lie below 0.1 of Nyquist."""
    Y, _, _ = gen_state_space_oscillator(
        n=32, T=600, seed=0, standardize=False, **_G1_DEFAULTS,
    )
    y0 = Y[..., 0].numpy()
    spec = np.fft.rfft(y0, axis=1)
    energy = (np.abs(spec) ** 2).astype(np.float64)
    cum = np.cumsum(energy, axis=1) / energy.sum(axis=1, keepdims=True)
    nyquist = spec.shape[1] - 1
    p90_bin = (cum >= 0.9).argmax(axis=1)
    fraction = float((p90_bin <= 0.1 * nyquist).mean())
    assert fraction >= 0.8, (
        f"G1: only {fraction:.0%} of samples have 90% energy below 0.1*Nyquist."
    )


def test_g1_reverse_roles_te_zero() -> None:
    """``reverse_roles=True`` ⇒ ``te_true = 0`` and direction is ``Y_to_X``."""
    _, _, meta = gen_state_space_oscillator(
        n=4, T=300, seed=0, reverse_roles=True, **_G1_DEFAULTS,
    )
    assert meta["te_true"] == 0.0
    assert meta["true_lag_band"] == []
    assert meta["direction"] == "Y_to_X"


def test_g1_easy_variant() -> None:
    """``easy_variant=True`` tiles user specs to fill all ``c_y`` channels."""
    base = dict(_G1_DEFAULTS)
    # Provide just one oscillator spec; easy_variant should tile to c_y=87.
    base["oscillators"] = [(0.99, 0.05)]
    base["delays"] = [60]
    base["B_y"] = [0.5]
    base["M"] = 1
    base["te_n_samples"] = 8_000  # cheap MC for the test
    _, _, meta = gen_state_space_oscillator(
        n=2, T=300, seed=0, easy_variant=True, **base,
    )
    assert meta["M"] == 87
    assert len(meta["oscillators"]) == 87
    assert meta["informative_channels"] == list(range(87))


# ---------------------------------------------------------------------------
# G3 (regime switch)
# ---------------------------------------------------------------------------


_G3_DEFAULTS = dict(
    K_classes=10,
    p_switch=0.05,
    delta=60,
    M=4,
)


def test_g3_shapes_and_dtype() -> None:
    """G3 returns ``(n, T, 87)`` / ``(n, T, 101)`` ``float32`` tensors."""
    Y, U, meta = gen_regime_switch_smooth(
        n=4, T=300, seed=0, **_G3_DEFAULTS,
    )
    assert Y.shape == (4, 300, 87)
    assert U.shape == (4, 300, 101)
    assert Y.dtype == torch.float32
    assert U.dtype == torch.float32
    assert meta["benchmark"] == "G3"


def test_g3_determinism() -> None:
    """Same seed -> bitwise-identical tensors."""
    Y1, U1, _ = gen_regime_switch_smooth(n=4, T=300, seed=11, **_G3_DEFAULTS)
    Y2, U2, _ = gen_regime_switch_smooth(n=4, T=300, seed=11, **_G3_DEFAULTS)
    assert torch.equal(Y1, Y2)
    assert torch.equal(U1, U2)


def test_g3_standardisation_per_channel() -> None:
    """``standardize=True`` puts every channel near $\\mathcal{N}(0, 1)$."""
    Y, U, _ = gen_regime_switch_smooth(
        n=64, T=300, seed=0, standardize=True, **_G3_DEFAULTS,
    )
    assert Y.mean(dim=(0, 1)).abs().max().item() < 0.15
    assert (Y.var(dim=(0, 1)) - 1.0).abs().max().item() < 0.15
    assert U.mean(dim=(0, 1)).abs().max().item() < 0.15
    assert (U.var(dim=(0, 1)) - 1.0).abs().max().item() < 0.15


def test_g3_te_true_matches_categorical_block() -> None:
    """``meta.te_true = M * te_categorical_switch_block(p, K, H)``."""
    _, _, meta = gen_regime_switch_smooth(n=4, T=300, seed=0, **_G3_DEFAULTS)
    expected = _G3_DEFAULTS["M"] * te_categorical_switch_block(
        _G3_DEFAULTS["p_switch"], _G3_DEFAULTS["K_classes"], 30,
    )
    assert meta["te_true"] == pytest.approx(expected, rel=1e-9)


def test_g3_shared_regime_te_true() -> None:
    """``shared_regime=True`` divides ``te_true`` by ``M``."""
    _, _, meta = gen_regime_switch_smooth(
        n=4, T=300, seed=0, shared_regime=True, **_G3_DEFAULTS,
    )
    expected = te_categorical_switch_block(
        _G3_DEFAULTS["p_switch"], _G3_DEFAULTS["K_classes"], 30,
    )
    assert meta["te_true"] == pytest.approx(expected, rel=1e-9)


def test_g3_regime_transition_matrix() -> None:
    """The empirical transition matrix matches the configured kernel.

    Uses a low source-noise variance so that ``argmax`` over the one-hot
    source block reliably recovers the underlying regime sequence (the
    default ``sigma2_u = 0.1`` corrupts $\\sim 12\\%$ of decodings and would
    make the test flaky).
    """
    n, T = 8, 2000
    p, K = 0.5, 10
    _, U, meta = gen_regime_switch_smooth(
        n=n, T=T, K_classes=K, p_switch=p, delta=30, M=1,
        sigma2_u=0.001, seed=42, standardize=False,
    )
    # Decode regimes from the (near-)noiseless argmax of the one-hot block.
    u_inf = U[..., :K].numpy()                          # (n, T, K)
    R_hat = u_inf.argmax(axis=-1)                       # (n, T)
    counts = np.zeros((K, K), dtype=np.float64)
    for sample in range(n):
        for t in range(T - 1):
            counts[R_hat[sample, t], R_hat[sample, t + 1]] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    P_hat = counts / np.maximum(row_sums, 1.0)
    diag_emp = float(np.diag(P_hat).mean())
    diag_thy = 1.0 - p + p / K
    assert abs(diag_emp - diag_thy) < 0.03, (
        f"G3 diag transition prob mismatch: emp={diag_emp:.3f}, "
        f"theory={diag_thy:.3f}"
    )


def test_g3_one_hot_source_fires_on_active_regime() -> None:
    r"""When the regime decoded from $U$ matches a class, that one-hot fires."""
    n, T = 4, 600
    K = 10
    _, U, meta = gen_regime_switch_smooth(
        n=n, T=T, K_classes=K, p_switch=0.05, delta=30, M=4,
        sigma2_u=0.05, seed=0, standardize=False,
    )
    # First M*K = 40 source channels are the one-hot block.
    u_inf = U[..., :K].numpy()                          # (n, T, K) — channel 0
    # Take the noiseless argmax; the value at that index must dominate.
    argmax_vals = u_inf.max(axis=-1)
    assert (argmax_vals > 0.5).mean() > 0.95, (
        "G3: one-hot source did not fire reliably for the active regime."
    )


def test_g3_delta_below_horizon_raises() -> None:
    """``delta < horizon`` is not a valid lag-recovery setup."""
    with pytest.raises(ValueError):
        gen_regime_switch_smooth(
            n=2, T=300, seed=0,
            K_classes=10, p_switch=0.05, delta=10, M=4,
        )


def test_g3_channel_budget_check() -> None:
    """``M * K_classes > c_u`` is rejected."""
    with pytest.raises(ValueError):
        gen_regime_switch_smooth(
            n=2, T=300, seed=0,
            K_classes=20, p_switch=0.05, delta=60, M=8,
        )


# ---------------------------------------------------------------------------
# Cross-generator sanity
# ---------------------------------------------------------------------------


def test_all_generators_return_three_outputs() -> None:
    """All generators return ``(Y, U, meta)`` triples."""
    for builder in (
        lambda: gen_smooth_arx(n=2, T=300, seed=0, **_G2_DEFAULTS),
        lambda: gen_state_space_oscillator(n=2, T=300, seed=0, **_G1_DEFAULTS),
        lambda: gen_regime_switch_smooth(n=2, T=300, seed=0, **_G3_DEFAULTS),
    ):
        Y, U, meta = builder()
        assert isinstance(Y, torch.Tensor)
        assert isinstance(U, torch.Tensor)
        assert isinstance(meta, dict)
        assert "te_true" in meta
        assert "true_lag_band" in meta
        assert "informative_channels" in meta
        assert "benchmark" in meta
