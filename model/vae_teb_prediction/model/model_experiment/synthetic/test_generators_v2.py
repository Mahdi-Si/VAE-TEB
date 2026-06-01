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
    r"""``standardize=True`` z-scores all channels except the Y small-noise tail.

    Under the v2 structured decomposition, the last ``n_smallnoise`` target
    channels keep their pre-placement $\sigma_{\text{smallnoise}}\ll 1$ so
    their MSE contribution stays tiny; only the head $[0, c_y - n_{\text
    {smallnoise}})$ block is z-scored to $\mathcal{N}(0, 1)$. The source
    buffer is fully standardised.
    """
    Y, U, meta = gen_smooth_arx(
        n=64, T=300, seed=0, standardize=True, **_G2_DEFAULTS,
    )
    n_sn = int(meta["channel_decomp"]["n_smallnoise"])
    sigma_sn = float(meta["channel_decomp"]["sigma_smallnoise"])
    head = Y[..., : Y.shape[-1] - n_sn]
    tail = Y[..., Y.shape[-1] - n_sn :]
    assert head.mean(dim=(0, 1)).abs().max().item() < 0.1
    assert (head.var(dim=(0, 1)) - 1.0).abs().max().item() < 0.1
    # Tail keeps its low-variance contract (within 30% relative slack at
    # n=64 samples).
    assert (tail.var(dim=(0, 1)) - sigma_sn ** 2).abs().max().item() < 0.3 * sigma_sn ** 2
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
    r"""``standardize=True`` z-scores all channels except the Y small-noise tail."""
    Y, U, meta = gen_state_space_oscillator(
        n=32, T=300, seed=0, standardize=True, **_G1_DEFAULTS,
    )
    n_sn = int(meta["channel_decomp"]["n_smallnoise"])
    sigma_sn = float(meta["channel_decomp"]["sigma_smallnoise"])
    head = Y[..., : Y.shape[-1] - n_sn]
    tail = Y[..., Y.shape[-1] - n_sn :]
    assert head.mean(dim=(0, 1)).abs().max().item() < 0.1
    assert (head.var(dim=(0, 1)) - 1.0).abs().max().item() < 0.1
    assert (tail.var(dim=(0, 1)) - sigma_sn ** 2).abs().max().item() < 0.3 * sigma_sn ** 2
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
# G1 / G2 variable per-sample delay (real-data small-lag regime)
# ---------------------------------------------------------------------------


_G1_VAR_DEFAULTS = dict(
    oscillators=[(0.99, 0.05)] * 4,
    target_ar=0.95,
    B_y=[0.02] * 4,
    sigma2_y=1.0,
    sigma2_eta=0.01,
    M=4,
    delay_min=1,
    delay_max=15,
    K_history=160,
    te_n_samples=4_000,
)


def test_g1_variable_delay_range_and_band() -> None:
    r"""Variable G1: per-sample delays land in $\{1..15\}$ and the union band
    is $\{0,\dots,d_{\max}-1\}$."""
    _, _, meta = gen_state_space_oscillator(n=128, T=200, seed=0, **_G1_VAR_DEFAULTS)
    dps = meta["delays_per_sample"]
    assert meta["variable_delay"] is True
    assert len(dps) == 128
    assert min(dps) >= 1 and max(dps) <= 15
    assert meta["delay_min"] == 1 and meta["delay_max"] == 15
    assert meta["true_lag_band"] == list(range(0, max(dps)))


def test_g1_variable_delay_te_is_sample_mean() -> None:
    """Variable G1: ``te_true`` equals the mean of the per-sample exact TE."""
    _, _, meta = gen_state_space_oscillator(n=96, T=200, seed=1, **_G1_VAR_DEFAULTS)
    tps = np.asarray(meta["te_per_sample"], dtype=float)
    assert len(tps) == 96
    assert meta["te_true"] == pytest.approx(float(tps.mean()), rel=1e-12)
    # te_by_delay keys cover every distinct drawn delay.
    distinct = {str(d) for d in set(meta["delays_per_sample"])}
    assert set(meta["te_by_delay"].keys()) == distinct
    # Real-data band check: this coupling sits inside [0.1, 3] nats.
    assert 0.1 <= meta["te_true"] <= 3.0


def test_g1_variable_delay_determinism() -> None:
    """Variable G1: same seed ⇒ identical draws and data."""
    Y1, U1, m1 = gen_state_space_oscillator(n=16, T=160, seed=5, **_G1_VAR_DEFAULTS)
    Y2, U2, m2 = gen_state_space_oscillator(n=16, T=160, seed=5, **_G1_VAR_DEFAULTS)
    assert m1["delays_per_sample"] == m2["delays_per_sample"]
    assert torch.equal(Y1, Y2) and torch.equal(U1, U2)


def test_g1_variable_vs_fixed_mutual_exclusion() -> None:
    """Supplying both ``delays`` and ``delay_min`` is rejected."""
    bad = dict(_G1_VAR_DEFAULTS)
    bad["delays"] = [5] * 4
    with pytest.raises(ValueError, match="mutually exclusive"):
        gen_state_space_oscillator(n=4, T=120, seed=0, **bad)


def test_g2_variable_delay_range_and_band() -> None:
    r"""Variable G2: per-sample delays in $\{1..15\}$, band $\{0,\dots,d_{\max}-1\}$,
    ``te_true`` is the per-sample mean, and it sits in the real-data band."""
    _, _, meta = gen_smooth_arx(
        n=128, T=200, rho_u=0.99, rho_y=0.95, c=0.035,
        sigma2_eta=1.0, sigma2_eps=1.0, M=4,
        delay_min=1, delay_max=15, K_history=160, seed=0,
    )
    dps = meta["delays_per_sample"]
    assert meta["variable_delay"] is True
    assert min(dps) >= 1 and max(dps) <= 15
    assert meta["true_lag_band"] == list(range(0, max(dps)))
    tps = np.asarray(meta["te_per_sample"], dtype=float)
    assert meta["te_true"] == pytest.approx(float(tps.mean()), rel=1e-9)
    assert 0.1 <= meta["te_true"] <= 3.0


def test_g2_variable_delay_fixed_mode_still_works() -> None:
    """G2 fixed scalar ``delay`` mode is preserved (single delay, band {D-H..D-1})."""
    _, _, meta = gen_smooth_arx(
        n=8, T=300, rho_u=0.99, rho_y=0.95, c=0.5,
        sigma2_eta=1.0, sigma2_eps=1.0, delay=60, M=4, seed=0,
    )
    assert meta["variable_delay"] is False
    assert meta["delay"] == 60
    assert meta["true_lag_band"] == list(range(60 - 30, 60))


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
    r"""``standardize=True`` z-scores all channels except the Y small-noise tail."""
    Y, U, meta = gen_regime_switch_smooth(
        n=64, T=300, seed=0, standardize=True, **_G3_DEFAULTS,
    )
    n_sn = int(meta["channel_decomp"]["n_smallnoise"])
    sigma_sn = float(meta["channel_decomp"]["sigma_smallnoise"])
    head = Y[..., : Y.shape[-1] - n_sn]
    tail = Y[..., Y.shape[-1] - n_sn :]
    assert head.mean(dim=(0, 1)).abs().max().item() < 0.15
    assert (head.var(dim=(0, 1)) - 1.0).abs().max().item() < 0.15
    assert (tail.var(dim=(0, 1)) - sigma_sn ** 2).abs().max().item() < 0.3 * sigma_sn ** 2
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


# ---------------------------------------------------------------------------
# Channel decomposition v2 (structured distractors)
# ---------------------------------------------------------------------------
#
# Covers the replacement of pure-N(0,1) distractor padding with the
# [TE | self-predictable | small-noise] target layout and the
# [TE | AR(1) distractor | pure noise] source layout (see generators.py
# DEFAULT_DECOMP_PARAMS and model_validation_v2.md ``Recommended channel
# design`` section).


@pytest.mark.parametrize(
    "builder, benchmark, m_source_factor",
    [
        (lambda: gen_smooth_arx(n=4, T=300, seed=0, **_G2_DEFAULTS), "G2", 1),
        (lambda: gen_state_space_oscillator(n=4, T=300, seed=0, **_G1_DEFAULTS), "G1", 1),
        (lambda: gen_regime_switch_smooth(n=4, T=300, seed=0, **_G3_DEFAULTS), "G3", 10),
    ],
)
def test_channel_layout_order(builder, benchmark: str, m_source_factor: int) -> None:
    r"""``meta['channel_layout']`` reports contiguous ``[TE | mid | tail]`` blocks."""
    _, _, meta = builder()
    M = meta["M"]
    layout = meta["channel_layout"]
    decomp = meta["channel_decomp"]
    c_y, c_u = meta["c_y"], meta["c_u"]
    # Target side: TE -> self -> smallnoise.
    assert layout["Y"]["te"] == list(range(0, M))
    assert layout["Y"]["self"][0] == M
    assert layout["Y"]["self"][-1] == M + decomp["n_self"] - 1
    assert layout["Y"]["smallnoise"][0] == M + decomp["n_self"]
    assert layout["Y"]["smallnoise"][-1] == c_y - 1
    assert len(layout["Y"]["smallnoise"]) == decomp["n_smallnoise"]
    # Source side: TE -> dist -> noise. G3 source TE block is M * K_classes.
    m_source = M * m_source_factor
    assert decomp["m_source"] == m_source
    assert layout["U"]["te"] == list(range(0, m_source))
    assert layout["U"]["dist"][0] == m_source if decomp["n_dist"] > 0 else True
    assert layout["U"]["noise"][-1] == c_u - 1
    # Budgets close.
    assert M + decomp["n_self"] + decomp["n_smallnoise"] == c_y
    assert m_source + decomp["n_dist"] + decomp["n_noise"] == c_u


def test_self_block_has_temporal_autocorrelation() -> None:
    r"""$Y^{\text{self}}$ has clearly positive lag-1 autocorrelation per channel.

    Both AR(1) ($\rho \in [0.95, 0.995]$) and oscillator (period $\ge 60$
    steps $\Rightarrow$ $\rho_1 = \cos(2\pi/\text{period})\gtrsim 0.94$)
    halves produce strong short-term dependence. We assert
    $\rho_1(\text{per channel}) > 0.5$ at the harshest of the two.
    """
    Y, _, meta = gen_smooth_arx(
        n=32, T=300, seed=0, standardize=True, **_G2_DEFAULTS,
    )
    self_idx = meta["channel_layout"]["Y"]["self"]
    Yself = Y[..., self_idx]                                  # (n, T, n_self)
    x = Yself - Yself.mean(dim=(0, 1), keepdim=True)
    num = (x[:, 1:, :] * x[:, :-1, :]).mean(dim=(0, 1))
    den = (x ** 2).mean(dim=(0, 1)).clamp_min(1e-8)
    rho1 = num / den
    assert rho1.min().item() > 0.5, (
        f"Y^self min lag-1 autocorrelation {rho1.min().item():.3f} <= 0.5; "
        f"per-channel values = {rho1.tolist()}"
    )


def test_self_block_independent_of_source_te() -> None:
    r"""$Y^{\text{self}}$ is uncoupled from $U^{\text{TE}}$ (zero-lag xcorr)."""
    Y, U, meta = gen_smooth_arx(
        n=64, T=300, seed=0, standardize=True, **_G2_DEFAULTS,
    )
    Yself = Y[..., meta["channel_layout"]["Y"]["self"]]
    Ute = U[..., meta["channel_layout"]["U"]["te"]]
    Ys = (Yself - Yself.mean(dim=(0, 1))).reshape(-1, Yself.shape[-1])
    Us = (Ute - Ute.mean(dim=(0, 1))).reshape(-1, Ute.shape[-1])
    cc = (Ys.T @ Us) / Ys.shape[0]
    cc = cc / (Ys.std(0).unsqueeze(1) * Us.std(0).unsqueeze(0) + 1e-8)
    max_abs = cc.abs().max().item()
    assert max_abs < 0.12, (
        f"Y^self ⟂ U^TE xcorr too large: {max_abs:.3f}; cross-corr matrix "
        f"shape={tuple(cc.shape)}"
    )


def test_smallnoise_block_variance_after_standardise() -> None:
    r"""The small-noise tail keeps its $\sigma^2_{\text{smallnoise}}$ variance."""
    Y, _, meta = gen_smooth_arx(
        n=64, T=600, seed=0, standardize=True, **_G2_DEFAULTS,
    )
    sn_idx = meta["channel_layout"]["Y"]["smallnoise"]
    sigma = float(meta["channel_decomp"]["sigma_smallnoise"])
    Ysn = Y[..., sn_idx]
    # Empirical variance per channel: relative slack 30% at n=64, T=600.
    var = Ysn.var(dim=(0, 1))
    assert ((var - sigma ** 2).abs() / (sigma ** 2)).max().item() < 0.3, (
        f"smallnoise tail variance deviates from sigma^2={sigma**2:.6f}; "
        f"per-channel var = {var.tolist()}"
    )


def test_budget_validation_raises_on_oversized_smallnoise() -> None:
    """An explicit ``channel_decomp`` with mismatched budget raises ``ValueError``."""
    bad_decomp = {
        "m": 4,
        "n_self": 70,
        "n_smallnoise": 999,                # blatantly oversized
        "m_source": 4,
        "n_dist": 80,
        "n_noise": 17,
        "sigma_smallnoise": 0.05,
        "ar1_fraction": 0.5,
        "rho_range_self": (0.95, 0.995),
        "rho_range_dist": (0.90, 0.995),
        "osc_period_range": (60, 200),
        "osc_amp_range": (0.5, 1.5),
    }
    with pytest.raises(ValueError, match="budget"):
        gen_smooth_arx(
            n=2, T=100, seed=0, channel_decomp=bad_decomp, **_G2_DEFAULTS,
        )


def test_budget_validation_raises_on_m_mismatch() -> None:
    """A ``channel_decomp`` whose ``m`` disagrees with generator ``M`` raises."""
    decomp_wrong_m = {
        "m": 99,                                # disagrees with M=4
        "n_self": 70, "n_smallnoise": 13,
        "m_source": 4, "n_dist": 80, "n_noise": 17,
        "sigma_smallnoise": 0.05, "ar1_fraction": 0.5,
        "rho_range_self": (0.95, 0.995), "rho_range_dist": (0.90, 0.995),
        "osc_period_range": (60, 200), "osc_amp_range": (0.5, 1.5),
    }
    with pytest.raises(ValueError, match="m"):
        gen_smooth_arx(
            n=2, T=100, seed=0, channel_decomp=decomp_wrong_m, **_G2_DEFAULTS,
        )


def test_default_decomp_clamps_for_easy_variant() -> None:
    r"""``easy_variant=True`` (M = c_y) collapses ``n_self`` / ``n_smallnoise`` to 0."""
    _, _, meta = gen_smooth_arx(
        n=4, T=300, seed=0, easy_variant=True, **{**_G2_DEFAULTS, "M": 4},
    )
    decomp = meta["channel_decomp"]
    assert decomp["m"] == meta["c_y"]                  # full informative width
    assert decomp["n_self"] == 0
    assert decomp["n_smallnoise"] == 0
    assert decomp["m_source"] == meta["c_y"]
    # Source: c_u=101 - m_source=87 - n_noise=14 (clamped from 17) = 0.
    assert decomp["n_dist"] == 0


def test_determinism_with_structured_decomp() -> None:
    r"""Same seed -> bitwise-identical buffers under structured padding too."""
    Y1, U1, m1 = gen_smooth_arx(n=4, T=300, seed=42, **_G2_DEFAULTS)
    Y2, U2, m2 = gen_smooth_arx(n=4, T=300, seed=42, **_G2_DEFAULTS)
    assert torch.equal(Y1, Y2)
    assert torch.equal(U1, U2)
    # The resolved layout is also identical (RNG-free).
    assert m1["channel_layout"] == m2["channel_layout"]


def test_source_dist_block_is_ar1() -> None:
    r"""$U^{\text{dist}}$ channels have AR(1)-like lag-1 autocorrelation."""
    _, U, meta = gen_smooth_arx(
        n=32, T=600, seed=0, standardize=True, **_G2_DEFAULTS,
    )
    dist_idx = meta["channel_layout"]["U"]["dist"]
    Ud = U[..., dist_idx]
    x = Ud - Ud.mean(dim=(0, 1), keepdim=True)
    num = (x[:, 1:, :] * x[:, :-1, :]).mean(dim=(0, 1))
    den = (x ** 2).mean(dim=(0, 1)).clamp_min(1e-8)
    rho1 = num / den
    # rho_range_dist defaults to (0.90, 0.995); empirical lag-1 should be high.
    assert rho1.min().item() > 0.7, (
        f"U^dist min lag-1 autocorrelation {rho1.min().item():.3f} <= 0.7"
    )


def test_source_noise_block_is_white() -> None:
    r"""$U^{\text{noise}}$ channels are essentially white (lag-1 xcorr ≈ 0)."""
    _, U, meta = gen_smooth_arx(
        n=32, T=600, seed=0, standardize=True, **_G2_DEFAULTS,
    )
    noise_idx = meta["channel_layout"]["U"]["noise"]
    Un = U[..., noise_idx]
    x = Un - Un.mean(dim=(0, 1), keepdim=True)
    num = (x[:, 1:, :] * x[:, :-1, :]).mean(dim=(0, 1))
    den = (x ** 2).mean(dim=(0, 1)).clamp_min(1e-8)
    rho1 = num / den
    assert rho1.abs().max().item() < 0.1, (
        f"U^noise max |lag-1 autocorrelation| {rho1.abs().max().item():.3f} >= 0.1"
    )


def test_te_true_invariant_to_decomp() -> None:
    r"""Changing the distractor decomposition does not move ``meta['te_true']``."""
    decomp_a = {
        "m": 4, "n_self": 70, "n_smallnoise": 13,
        "m_source": 4, "n_dist": 80, "n_noise": 17,
        "sigma_smallnoise": 0.05, "ar1_fraction": 0.5,
        "rho_range_self": (0.95, 0.995), "rho_range_dist": (0.90, 0.995),
        "osc_period_range": (60, 200), "osc_amp_range": (0.5, 1.5),
    }
    decomp_b = {
        **decomp_a,
        "n_self": 50, "n_smallnoise": 33,           # different decomp
        "n_dist": 90, "n_noise": 7,
        "sigma_smallnoise": 0.10,
    }
    _, _, meta_a = gen_smooth_arx(
        n=4, T=300, seed=0, channel_decomp=decomp_a, **_G2_DEFAULTS,
    )
    _, _, meta_b = gen_smooth_arx(
        n=4, T=300, seed=0, channel_decomp=decomp_b, **_G2_DEFAULTS,
    )
    assert meta_a["te_true"] == pytest.approx(meta_b["te_true"], rel=1e-12)
