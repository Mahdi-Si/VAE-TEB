r"""Tests for the Sprint 1 raw signal generators (S1-T01…S1-T05).

Covers the coupled latent pair (stationary std, lag-1 autocorrelation, low-frequency
spectrum, null-vs-signal coupling), the DC draws and power-based dressing bands, the
band-limited upsample (length + anti-alias by construction), the strictly-positive AM
envelope and raw composition, the AM-separation analytic pre-check, and the annotated
raw preview + null-separability.

All tests are pure numpy (the scattering transform and Lightning arrive in Sprints 2+),
so the suite runs in a few seconds. Numerical expectations were verified against the
ported :mod:`analytic_te` and the vendored kymatio filter bank; tolerances are set so a
mis-ported recurrence, a wrong band-power scale, or a broken upsample is caught.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import raw_generators as rg
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.visualize_v2 import (
    plot_raw_preview,
)

# v2 single-pathway operating regime (config benchmarks.G1_raw).
_R, _W = 0.80, 0.10
_TARGET_AR = 0.40
_SIGMA2_Y = 1.0
_SIGMA2_ETA = 0.01
_D = 8
_FS = 4.0
_N_RAW = 5280
_T_TOT = 330
_FS_DEC = _FS / rg.DECIMATION      # 0.25 Hz
_F_PULSE = 0.02
_Q = 4
_AM_RATIO = 4.0

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config_synth_v2.yaml"
)


@pytest.fixture(scope="module")
def config() -> dict:
    r"""Load the parsed ``config_synth_v2.yaml``."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# ---------------------------------------------------------------------------
# S1-T01: coupled latent pair + true-lag trajectory
# ---------------------------------------------------------------------------


def test_ar2_stationary_std_and_autocorr() -> None:
    r"""The AR(2) closed-forms match the analytic values for the v2 oscillator."""
    assert rg.ar2_stationary_std(_R, _W, _SIGMA2_ETA) == pytest.approx(0.5419, abs=1e-3)
    assert rg.ar2_lag1_autocorr(_R, _W) == pytest.approx(0.9707, abs=1e-3)


def test_latent_pair_std_and_autocorr() -> None:
    r"""The simulated source matches the stationary std and lag-1 autocorrelation."""
    c, _ = rg.simulate_latent_pair(
        2000, _T_TOT, r=_R, w=_W, target_ar=_TARGET_AR, B=1.5, D=_D,
        sigma2_y=_SIGMA2_Y, sigma2_eta=_SIGMA2_ETA, seed=0,
    )
    assert c.shape == (2000, _T_TOT)
    assert c.std() == pytest.approx(rg.ar2_stationary_std(_R, _W, _SIGMA2_ETA), rel=0.05)
    cc = c - c.mean(axis=1, keepdims=True)
    rho1 = float((cc[:, :-1] * cc[:, 1:]).mean() / (cc * cc).mean())
    assert rho1 == pytest.approx(rg.ar2_lag1_autocorr(_R, _W), abs=0.02)


def test_latent_pair_is_low_frequency() -> None:
    r"""The source spectral peak lies in the low-frequency contraction band (< 0.01 Hz).

    The damped AR(2) ($r = 0.8$) is overdamped, so its PSD peaks *below* the pole-angle
    frequency ($\omega f_s^{\mathrm{dec}} / 2\pi \approx 0.004$ Hz), well within the
    slow contraction band — not up in the FHRV / carrier range.
    """
    c, _ = rg.simulate_latent_pair(
        2000, _T_TOT, r=_R, w=_W, target_ar=_TARGET_AR, B=0.0, D=_D,
        sigma2_y=_SIGMA2_Y, sigma2_eta=_SIGMA2_ETA, seed=0,
    )
    cc = c - c.mean(axis=1, keepdims=True)
    psd = (np.abs(np.fft.rfft(cc, axis=1)) ** 2).mean(0)
    freqs = np.fft.rfftfreq(_T_TOT, d=1.0) * _FS_DEC     # Hz on the decimated grid
    peak_hz = freqs[1 + int(np.argmax(psd[1:]))]         # exclude the DC bin
    assert 0.0 < peak_hz < 0.01


@pytest.mark.parametrize("B, lo, hi", [(0.0, 0.0, 0.08), (1.5, 0.2, 1.0)])
def test_latent_pair_coupling(B: float, lo: float, hi: float) -> None:
    r"""$\mathrm{corr}(d_k, c_{k-D})$ is ~0 for a null cell and clearly positive with coupling."""
    c, d = rg.simulate_latent_pair(
        2000, _T_TOT, r=_R, w=_W, target_ar=_TARGET_AR, B=B, D=_D,
        sigma2_y=_SIGMA2_Y, sigma2_eta=_SIGMA2_ETA, seed=0,
    )
    a = c[:, :-_D].ravel()
    b = d[:, _D:].ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    assert lo <= abs(corr) <= hi


def test_true_lag_trajectory_flat() -> None:
    r"""``true_lag_trajectory`` is a flat int16 line at $D$ (fixed mode)."""
    tt = rg.true_lag_trajectory(5, _T_TOT, _D)
    assert tt.shape == (5, _T_TOT)
    assert tt.dtype == np.int16
    assert np.all(tt == _D)


# ---------------------------------------------------------------------------
# S1-T02: DC baseline + independent dressing bands
# ---------------------------------------------------------------------------


def test_draw_dc_ranges() -> None:
    r"""Per-sample DC draws fall in the physiological FHR/UP ranges."""
    rng = np.random.default_rng(0)
    mu_fhr, mu_up = rg.draw_dc(5000, rng, fhr_range=(110.0, 160.0), up_range=(5.0, 25.0))
    assert mu_fhr.shape == (5000,) and mu_up.shape == (5000,)
    assert mu_fhr.min() >= 110.0 and mu_fhr.max() <= 160.0
    assert mu_up.min() >= 5.0 and mu_up.max() <= 25.0


def test_synth_band_power_exact() -> None:
    r"""On-DFT-grid band synthesis yields per-sample variance $\approx P$ (bin orthogonality)."""
    rng = np.random.default_rng(0)
    P = 11.25
    v = rg.synth_band(64, _N_RAW, _FS, 0.03, 0.15, P, rng)
    assert v.shape == (64, _N_RAW)
    assert v.var() == pytest.approx(P, rel=0.05)


def test_synth_band_edge_cases() -> None:
    r"""Zero power (or an empty band) returns zeros."""
    rng = np.random.default_rng(0)
    assert np.all(rg.synth_band(4, _N_RAW, _FS, 0.03, 0.15, 0.0, rng) == 0.0)
    # A band with no DFT bin at this N (a razor-thin interval between grid points).
    assert np.all(rg.synth_band(4, _N_RAW, _FS, 0.03001, 0.03002, 5.0, rng) == 0.0)


def test_synth_fhrv_band_powers() -> None:
    r"""FHRV power lands in the configured LF/MF/HF bands (and VLF stays empty)."""
    rng = np.random.default_rng(0)
    powers = {"LF": 11.25, "MF": 6.25, "HF": 2.5}
    v = rg.synth_fhrv(64, _N_RAW, _FS, powers, rng)
    freqs = np.fft.rfftfreq(_N_RAW, d=1.0 / _FS)
    spec = (np.abs(np.fft.rfft(v, axis=1)) ** 2).mean(0)

    def band_var(f_lo: float, f_hi: float) -> float:
        # Parseval on the (mean-removed) real signal: Var = 2 * sum|X|^2 / N^2 over
        # positive-frequency bins.
        mask = (freqs >= f_lo) & (freqs < f_hi)
        return float(2.0 * spec[mask].sum() / (_N_RAW ** 2))

    assert band_var(*rg.FHRV_BANDS["LF"]) == pytest.approx(11.25, rel=0.15)
    assert band_var(*rg.FHRV_BANDS["MF"]) == pytest.approx(6.25, rel=0.15)
    assert band_var(*rg.FHRV_BANDS["HF"]) == pytest.approx(2.5, rel=0.15)
    assert band_var(0.003, 0.03) < 0.05      # VLF (coupled band) carries no FHRV dressing
    assert v.var() == pytest.approx(20.0, rel=0.1)


def test_synth_band_notch_preserves_power_and_clears_carrier() -> None:
    r"""A notch removes the carrier neighbourhood but the surviving bins keep full power $P$."""
    rng_seed = 1
    P = 11.25
    f_pulse, Q = 0.06, 4
    notch = (f_pulse * 2.0 ** (-1.0 / Q), f_pulse * 2.0 ** (1.0 / Q))
    v_plain = rg.synth_band(128, _N_RAW, _FS, 0.03, 0.15, P, np.random.default_rng(rng_seed))
    v_notch = rg.synth_band(128, _N_RAW, _FS, 0.03, 0.15, P, np.random.default_rng(rng_seed),
                            notch=notch)
    # Total power is preserved (surviving bins are re-normalised via m_b / amp).
    assert v_notch.var() == pytest.approx(P, rel=0.05)
    assert v_plain.var() == pytest.approx(P, rel=0.05)
    # Energy in the notch neighbourhood is essentially zero.
    freqs = np.fft.rfftfreq(_N_RAW, d=1.0 / _FS)
    spec = (np.abs(np.fft.rfft(v_notch, axis=1)) ** 2).sum(0)
    in_notch = (freqs >= notch[0]) & (freqs <= notch[1])
    assert spec[in_notch].sum() / spec.sum() < 1e-9
    # A notch covering the whole band returns zeros (empty-band guard).
    z = rg.synth_band(4, _N_RAW, _FS, 0.055, 0.065, P, np.random.default_rng(0), notch=(0.0, 1.0))
    assert np.all(z == 0.0)


def test_synth_fhrv_notch_clears_carrier_keeps_total_power() -> None:
    r"""``synth_fhrv`` forwards the notch: LF loses the carrier band, total power unchanged."""
    powers = {"LF": 11.25, "MF": 6.25, "HF": 2.5}
    f_pulse, Q = 0.06, 4
    notch = (f_pulse * 2.0 ** (-1.0 / Q), f_pulse * 2.0 ** (1.0 / Q))
    v = rg.synth_fhrv(64, _N_RAW, _FS, powers, np.random.default_rng(0), notch=notch)
    freqs = np.fft.rfftfreq(_N_RAW, d=1.0 / _FS)
    spec = (np.abs(np.fft.rfft(v, axis=1)) ** 2).sum(0)
    in_notch = (freqs >= notch[0]) & (freqs <= notch[1])
    assert spec[in_notch].sum() / spec.sum() < 1e-9        # carrier neighbourhood cleared
    assert v.var() == pytest.approx(20.0, rel=0.1)         # total FHRV power preserved


def test_wander_independent_of_latent() -> None:
    r"""The (independent) baseline wander is uncorrelated with the coupled latent."""
    rng = np.random.default_rng(1)
    wander = rg.synth_baseline_wander(200, _N_RAW, _FS, 3.0, rng)
    c, _ = rg.simulate_latent_pair(
        200, _T_TOT, r=_R, w=_W, target_ar=_TARGET_AR, B=1.5, D=_D,
        sigma2_y=_SIGMA2_Y, sigma2_eta=_SIGMA2_ETA, seed=7,
    )
    c_up = rg.upsample_bandlimited(c, rg.DECIMATION, fs_dec=_FS_DEC, lowpass_hz=_F_PULSE)
    corr = float(np.corrcoef(wander.ravel(), c_up.ravel())[0, 1])
    assert abs(corr) < 0.1


# ---------------------------------------------------------------------------
# S1-T03a: band-limited upsample
# ---------------------------------------------------------------------------


def test_upsample_length_and_antialias() -> None:
    r"""FFT upsample: correct length, no energy at/above the cutoff, bounded overshoot."""
    c, _ = rg.simulate_latent_pair(
        64, _T_TOT, r=_R, w=_W, target_ar=_TARGET_AR, B=1.5, D=_D,
        sigma2_y=_SIGMA2_Y, sigma2_eta=_SIGMA2_ETA, seed=0,
    )
    c_up = rg.upsample_bandlimited(
        c, rg.DECIMATION, fs_dec=_FS_DEC, lowpass_hz=_F_PULSE, detrend=False
    )
    assert c_up.shape == (64, _N_RAW)
    assert np.isfinite(c_up).all()
    cc = c_up - c_up.mean(axis=1, keepdims=True)
    spec = (np.abs(np.fft.rfft(cc, axis=1)) ** 2).sum()
    freqs = np.fft.rfftfreq(_N_RAW, d=1.0 / _FS)
    above = (np.abs(np.fft.rfft(cc, axis=1)) ** 2)[:, freqs >= _F_PULSE].sum()
    assert above / spec < 1e-6                 # strictly band-limited below f_pulse
    assert np.abs(c_up).max() < 1.5 * np.abs(c).max()


def test_upsample_preserves_dc_and_1d() -> None:
    r"""Upsampling preserves the mean and accepts 1-D input."""
    x = np.linspace(0.0, 1.0, _T_TOT) + 3.0
    y = rg.upsample_bandlimited(x, rg.DECIMATION, fs_dec=_FS_DEC, detrend=True)
    assert y.shape == (_N_RAW,)
    assert y.mean() == pytest.approx(x.mean(), rel=1e-3)


# ---------------------------------------------------------------------------
# S1-T03b: positive AM envelope + raw composition
# ---------------------------------------------------------------------------


def test_am_envelope_positivity_and_peak() -> None:
    r"""The AM envelope is strictly positive and hits ``amp_peak`` at $\tilde x = \sigma_{\mathrm{ref}}$."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 1000))
    sigma_ref = rg.AM_MAX_SIGMA_MULT * 1.0
    A = rg.am_envelope(x, sigma_ref=sigma_ref, am_offset_ratio=_AM_RATIO, amp_peak=70.0)
    assert A.min() > 0.0
    # At x == sigma_ref the envelope equals amp_peak; at x == 0 it equals a0.
    peak = rg.am_envelope(
        np.array([sigma_ref]), sigma_ref=sigma_ref, am_offset_ratio=_AM_RATIO, amp_peak=70.0
    )
    assert float(peak[0]) == pytest.approx(70.0, rel=1e-9)


def test_am_envelope_rejects_bad_ratio() -> None:
    r"""``am_offset_ratio <= 1`` (no positivity margin) is rejected."""
    with pytest.raises(ValueError):
        rg.am_envelope(np.zeros(4), sigma_ref=1.0, am_offset_ratio=1.0, amp_peak=70.0)


def test_generate_cell_raw_shapes_and_provenance(config: dict) -> None:
    r"""``generate_cell_raw`` returns correctly-shaped raw signals + provenance."""
    out = rg.generate_cell_raw(6, B=1.5, D=_D, config=config, seed=0, te_inj=2.0)
    assert out["fhr_raw"].shape == (6, _N_RAW)
    assert out["up_raw"].shape == (6, _N_RAW)
    assert out["true_lag_tt"].shape == (6, _T_TOT)
    assert np.isfinite(out["fhr_raw"]).all() and np.isfinite(out["up_raw"]).all()
    assert set(out["meta"]) >= {"D", "B", "te_inj", "render_mode", "T_tot", "n_raw", "fs", "f_pulse"}
    assert out["meta"]["D"] == _D and out["meta"]["te_inj"] == 2.0
    # DC levels survive into the composed signals (the carrier averages to ~0 over time).
    assert 110.0 - 5.0 <= out["fhr_raw"].mean() <= 160.0 + 5.0
    assert 5.0 - 5.0 <= out["up_raw"].mean() <= 25.0 + 5.0


def test_generate_cell_raw_envelope_positive_everywhere(config: dict) -> None:
    r"""The rendered amplitude envelopes $A_u, A_y$ are strictly positive everywhere (S1-T03b)."""
    out = rg.generate_cell_raw(8, B=1.5, D=_D, config=config, seed=0)
    c_t = out["latents"]["c_tilde"]
    d_t = out["latents"]["d_tilde"]
    ref_c = rg.AM_MAX_SIGMA_MULT * rg.ar2_stationary_std(_R, _W, _SIGMA2_ETA)
    ref_d = rg.AM_MAX_SIGMA_MULT * float(d_t.std())
    A_u = rg.am_envelope(c_t, sigma_ref=ref_c, am_offset_ratio=_AM_RATIO, amp_peak=70.0)
    A_y = rg.am_envelope(d_t, sigma_ref=ref_d, am_offset_ratio=_AM_RATIO, amp_peak=40.0)
    assert A_u.min() > 0.0 and A_y.min() > 0.0


def test_generate_cell_raw_deterministic(config: dict) -> None:
    r"""A fixed seed reproduces the raw signals bit-for-bit (determinism)."""
    a = rg.generate_cell_raw(4, B=1.0, D=_D, config=config, seed=3)
    b = rg.generate_cell_raw(4, B=1.0, D=_D, config=config, seed=3)
    assert np.array_equal(a["fhr_raw"], b["fhr_raw"])
    assert np.array_equal(a["up_raw"], b["up_raw"])


def test_generate_cell_raw_pulse_train_now_implemented(config: dict) -> None:
    r"""``pulse_train`` render is implemented in Sprint 7 (S7-T04) and no longer raises."""
    out = rg.generate_cell_raw(2, B=1.0, D=_D, config=config, seed=0, render_mode="pulse_train")
    assert out["fhr_raw"].shape == (2, _N_RAW)
    assert out["meta"]["render_mode"] == "pulse_train"


# ---------------------------------------------------------------------------
# S1-T04: AM-separation analytic pre-check
# ---------------------------------------------------------------------------


def test_am_separation_margin_default_flagged() -> None:
    r"""The default ($f_{\mathrm{pulse}} = 0.02$) is flagged marginal; a raised carrier improves."""
    res = rg.am_separation_margin(
        r=_R, w=_W, f_pulse=0.02, Q=_Q, fs=_FS, am_offset_ratio=_AM_RATIO, sigma2_eta=_SIGMA2_ETA
    )
    assert res["margin_peak"] < 1.0
    assert res["adequate"] is False
    assert res["margin_peak"] == pytest.approx(0.522, abs=0.02)
    assert res["mod_depth_rms"] == pytest.approx(1.0 / (rg.AM_MAX_SIGMA_MULT * _AM_RATIO), rel=1e-9)
    assert "am_offset_ratio" in res["recommendation"]

    hi = rg.am_separation_margin(
        r=_R, w=_W, f_pulse=0.05, Q=_Q, fs=_FS, am_offset_ratio=_AM_RATIO, sigma2_eta=_SIGMA2_ETA
    )
    assert hi["margin_peak"] > 1.0


def test_am_separation_monotone_in_f_pulse() -> None:
    r"""``margin_peak`` and ``preservation`` increase monotonically with the carrier frequency."""
    margins, pres = [], []
    for fp in (0.02, 0.05, 0.08):
        res = rg.am_separation_margin(
            r=_R, w=_W, f_pulse=fp, Q=_Q, fs=_FS, am_offset_ratio=_AM_RATIO, sigma2_eta=_SIGMA2_ETA
        )
        margins.append(res["margin_peak"])
        pres.append(res["preservation"])
    assert margins[0] < margins[1] < margins[2]
    assert pres[0] < pres[1] < pres[2]
    assert 0.0 < pres[0] < 1.0


def test_am_separation_from_config(config: dict) -> None:
    r"""The config-driven pre-check runs and clears the locked cell (S4-T01: f_pulse=0.06).

    Sprint 3's de-risk raised the carrier to ``f_pulse=0.06`` Hz precisely so the AM
    envelope/carrier separation is adequate (margin > 1); the locked config therefore
    reports ``adequate=True`` where the old 0.02 default was MARGINAL.
    """
    res = rg.am_separation_from_config(config)
    assert set(res) >= {"margin_peak", "margin_edge", "preservation", "sigma_wav_hz",
                        "f_env_peak", "mod_depth_rms", "adequate", "recommendation"}
    assert res["adequate"] is True
    assert res["margin_peak"] > 1.0


# ---------------------------------------------------------------------------
# S1-T05: raw preview figure + null-separability
# ---------------------------------------------------------------------------


def _band_envelope(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> np.ndarray:
    r"""Analytic-signal envelope of ``x`` band-passed to ``[f_lo, f_hi]`` (numpy-only)."""
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / fs)
    spec = np.fft.rfft(x, axis=-1) * ((freqs >= f_lo) & (freqs <= f_hi))[None, :]
    bp = np.fft.irfft(spec, n=x.shape[-1], axis=-1)
    ana = np.fft.fft(bp, axis=-1)
    N = x.shape[-1]
    h = np.zeros(N)
    h[0] = 1.0
    if N % 2 == 0:
        h[1 : N // 2] = 2.0
        h[N // 2] = 1.0
    else:
        h[1 : (N + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(ana * h[None, :], axis=-1))


def _env_xcorr_at_lag(
    up: np.ndarray, fhr: np.ndarray, lag_raw: int, f_pulse: float
) -> float:
    r"""Lagged cross-correlation of the carrier-band envelopes (edges trimmed).

    The band tracks the config's carrier ``f_pulse`` (``[0.5 f_pulse, 1.6 f_pulse]``) so
    the probe follows the locked carrier rather than a hard-coded 0.02 Hz band.
    """
    f_lo, f_hi = f_pulse * 0.5, f_pulse * 1.6
    eu = _band_envelope(up, _FS, f_lo, f_hi)[:, 240:-240]
    ey = _band_envelope(fhr, _FS, f_lo, f_hi)[:, 240:-240]
    a = (eu[:, :-lag_raw]).ravel()
    b = (ey[:, lag_raw:]).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).mean() / (a.std() * b.std() + 1e-12))


def test_null_separability(config: dict) -> None:
    r"""The carrier-band envelope cross-correlation is ~0 for a null cell, positive with coupling.

    The shared carrier is removed by taking the envelope (the coupling lives in the
    amplitude), so a null cell has no envelope coupling at the lag while a strong cell does.
    The band follows the locked carrier ``raw.f_pulse`` (S4-T01: 0.06 Hz).
    """
    f_pulse = float(config["benchmarks"]["G1_raw"]["raw"]["f_pulse"])
    lag_raw = rg.DECIMATION * _D
    null = rg.generate_cell_raw(200, B=0.0, D=_D, config=config, seed=0)
    sig = rg.generate_cell_raw(200, B=1.5, D=_D, config=config, seed=0)
    assert abs(_env_xcorr_at_lag(null["up_raw"], null["fhr_raw"], lag_raw, f_pulse)) < 0.1
    assert _env_xcorr_at_lag(sig["up_raw"], sig["fhr_raw"], lag_raw, f_pulse) > 0.15


def test_plot_raw_preview_writes_files(config: dict, tmp_path: Path) -> None:
    r"""The raw preview writes both a PDF and a PNG (S1-T05)."""
    out = rg.generate_cell_raw(2, B=1.5, D=_D, config=config, seed=0, te_inj=2.0)
    written = plot_raw_preview(
        out["fhr_raw"], out["up_raw"], tmp_path / "raw_preview",
        meta=out["meta"], fs=_FS, sample=0,
    )
    assert len(written) == 2
    for path in written:
        assert path.exists() and path.stat().st_size > 0


def test_plot_raw_preview_preserves_dotted_stem(config: dict, tmp_path: Path) -> None:
    r"""A stem containing a dot (e.g. a TE value) is not truncated at the dot."""
    out = rg.generate_cell_raw(1, B=1.0, D=_D, config=config, seed=0)
    written = plot_raw_preview(out["fhr_raw"], out["up_raw"], tmp_path / "G1_te2.0_D8")
    names = sorted(p.name for p in written)
    assert names == ["G1_te2.0_D8.pdf", "G1_te2.0_D8.png"]
    assert all(p.exists() for p in written)


# ---------------------------------------------------------------------------
# Regression guards for the code-review findings
# ---------------------------------------------------------------------------


def test_generate_cell_raw_rejects_non_multiple_n_raw(config: dict) -> None:
    r"""``n_raw`` not divisible by the decimation factor raises a clear error, not a broadcast crash."""
    cfg = copy.deepcopy(config)
    cfg["benchmarks"]["G1_raw"]["raw"]["n_raw"] = 5000     # not a multiple of 16
    with pytest.raises(ValueError, match="multiple"):
        rg.generate_cell_raw(2, B=1.0, D=_D, config=cfg, seed=0)


def test_generate_cell_raw_rejects_unknown_render_mode(config: dict) -> None:
    r"""An unknown / falsy render_mode override is validated, not silently replaced (S7-T04)."""
    with pytest.raises(ValueError, match="unknown render_mode"):
        rg.generate_cell_raw(2, B=1.0, D=_D, config=config, seed=0, render_mode="")
    with pytest.raises(ValueError, match="unknown render_mode"):
        rg.generate_cell_raw(2, B=1.0, D=_D, config=config, seed=0, render_mode="bogus")


# ---------------------------------------------------------------------------
# S7-T04: pulse_train render variant
# ---------------------------------------------------------------------------


def test_make_pulse_train_nonneg_and_fundamental() -> None:
    r"""The raised-cosine event train is non-negative in [0,1] with its peak at ``rate_hz``."""
    g = rg.make_pulse_train(_N_RAW, _FS, 0.06, duty=0.5)
    assert g.shape == (_N_RAW,)
    assert g.min() >= 0.0 and g.max() <= 1.0 + 1e-9
    spec = np.abs(np.fft.rfft(g - g.mean()))
    freqs = np.fft.rfftfreq(_N_RAW, d=1.0 / _FS)
    assert abs(float(freqs[int(spec.argmax())]) - 0.06) < 0.01


def test_make_pulse_train_rejects_bad_params() -> None:
    r"""Out-of-range duty / non-positive rate raise clear errors."""
    with pytest.raises(ValueError, match="duty"):
        rg.make_pulse_train(_N_RAW, _FS, 0.06, duty=0.0)
    with pytest.raises(ValueError, match="rate_hz"):
        rg.make_pulse_train(_N_RAW, _FS, 0.0, duty=0.5)


def test_pulse_train_render_shapes_and_positivity(config: dict) -> None:
    r"""``render_mode='pulse_train'`` yields valid raw pairs with non-negative one-sided events."""
    out = rg.generate_cell_raw(4, B=1.5, D=_D, config=config, seed=0, te_inj=2.0,
                               render_mode="pulse_train")
    assert out["fhr_raw"].shape == (4, _N_RAW)
    assert out["up_raw"].shape == (4, _N_RAW)
    assert out["meta"]["render_mode"] == "pulse_train"
    lat = out["latents"]
    # One-sided raised-cosine carrier and rendered bands (unlike the symmetric am_carrier).
    assert lat["carrier_u"].min() >= 0.0 and lat["carrier_u"].max() <= 1.0 + 1e-9
    assert (lat["A_u"] > 0).all() and (lat["A_y"] > 0).all()
    assert (lat["u_c"] >= 0).all() and (lat["y_d"] >= 0).all()


@pytest.mark.parametrize("r, w", [(0.0, 0.10), (0.80, 0.0)])
def test_am_separation_rejects_degenerate_oscillator(r: float, w: float) -> None:
    r"""A degenerate oscillator (r=0 or w=0) raises a clear error instead of crashing."""
    with pytest.raises(ValueError):
        rg.am_separation_margin(
            r=r, w=w, f_pulse=0.02, Q=_Q, fs=_FS, am_offset_ratio=_AM_RATIO, sigma2_eta=_SIGMA2_ETA
        )


def test_synth_accelerations_zero_rate_is_empty() -> None:
    r"""``rate_per_min = 0`` disables accelerations (all-zero output)."""
    rng = np.random.default_rng(0)
    out = rg.synth_accelerations(4, _N_RAW, _FS, amp_bpm=(10.0, 25.0), rate_per_min=0.0, rng=rng)
    assert out.shape == (4, _N_RAW)
    assert np.all(out == 0.0)


def test_upsample_low_cutoff_keeps_dc() -> None:
    r"""A cutoff below the first AC bin keeps DC (constant output), not an all-zero signal."""
    x = np.full((3, _T_TOT), 7.0) + 0.01 * np.arange(_T_TOT)[None, :]
    y = rg.upsample_bandlimited(x, rg.DECIMATION, fs_dec=_FS_DEC, lowpass_hz=1e-5, detrend=False)
    assert y.shape == (3, _N_RAW)
    assert not np.allclose(y, 0.0)
    assert y.mean() == pytest.approx(x.mean(), rel=1e-3)


def test_upsample_detrend_reproduces_linear_at_samples() -> None:
    r"""With ``detrend=True`` a linear ramp is reproduced exactly at the original sample points."""
    x = np.linspace(3.0, 10.0, _T_TOT)
    y = rg.upsample_bandlimited(x, rg.DECIMATION, fs_dec=_FS_DEC, detrend=True)
    assert np.allclose(y[:: rg.DECIMATION], x, atol=1e-9)
