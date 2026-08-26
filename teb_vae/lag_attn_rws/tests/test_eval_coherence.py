r"""The coherence analysis: the $\tau$-slice construction, the estimator, and the exact identities.

The analysis rests on one structural claim -- that concatenating a fixed horizon step $\tau$ over
consecutive anchors yields a contiguous, gap-free, non-overlapping $4\,$Hz series -- and on one
arithmetic claim, that the normalised residual spectrum splits exactly into an irreducible, a
timing and an amplitude term. Both are proved here rather than asserted in a docstring, because
every number the analysis emits is wrong if either fails and neither failure is loud.

The tests are ordered as the module is built: geometry, then bands, then the estimator's known
answers, then the identities, then the reduction.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from teb_vae.lag_attn_rws.eval.figures_seam import figure_filename
from teb_vae.lag_attn_rws.eval import metrics, spectra
from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask
from teb_vae.lag_attn_rws.nets.raw_targets import build_future_target

#: The shipped geometry. Every expected number below is stated for it explicitly rather than
#: recomputed from the layout, so a change to the layout fails here instead of quietly moving the
#: expectations along with itself.
SHIPPED = TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=30, warmup=30)


@pytest.fixture(scope="module")
def layout() -> spectra.SliceGeometry:
    """The Welch layout at the shipped geometry."""
    return spectra.slice_geometry(
        t_valid=SHIPPED.t_valid,
        warmup=SHIPPED.warmup,
        horizon=SHIPPED.horizon,
        raw_per_step=SHIPPED.r,
    )


# =============================================================================
# The tau-slice construction -- the claim every number downstream depends on
# =============================================================================
def test_a_tau_slice_is_exactly_the_raw_signal_over_its_own_span(layout) -> None:
    r"""**The test the whole analysis rests on.**

    Concatenating a fixed $\tau$ over consecutive anchors must reproduce the raw trace itself over
    $[R(w + 1 + \tau),\ R(T_{\mathrm{valid}} + 1 + \tau))$, sample for sample. Checked against the
    model's *own* target builder rather than against a re-derived index, so a change to the
    geometry moves both sides together and this test still means something.

    If this fails, every coherence, gain, phase and band number the analysis emits is a spectrum of
    a scrambled signal -- and nothing else in the pipeline would notice, because a scrambled series
    still has a perfectly plausible-looking spectrum.
    """
    rng = np.random.default_rng(0)
    fhr_raw = torch.tensor(
        rng.standard_normal((2, SHIPPED.raw_len)), dtype=torch.float64
    )
    target = build_future_target(fhr_raw, SHIPPED)

    sliced = metrics.tau_slices(target, warmup=SHIPPED.warmup)

    assert sliced.shape == (2, SHIPPED.horizon, layout.n_samples)
    for tau in (0, 1, 17, SHIPPED.horizon - 1):
        # The last anchor contributes R samples *beyond* R(T_valid + tau), so the span closes at
        # R(T_valid + 1 + tau) -- which at tau = H - 1 is exactly raw_len.
        start = SHIPPED.r * (SHIPPED.warmup + 1 + tau)
        stop = SHIPPED.r * (SHIPPED.t_valid + 1 + tau)
        assert stop - start == layout.n_samples
        assert stop <= SHIPPED.raw_len
        assert torch.equal(sliced[:, tau], fhr_raw[:, start:stop])


def test_the_slice_is_not_merely_a_permutation_of_the_right_samples(layout) -> None:
    """Order is the content. A transposed reshape holds exactly the same values and would pass any
    set-equality check, so the identity above is asserted elementwise and this test states why: a
    ramp is reproduced only if the samples arrive in time order."""
    ramp = torch.arange(SHIPPED.raw_len, dtype=torch.float64)[None, :]
    target = build_future_target(ramp, SHIPPED)

    sliced = metrics.tau_slices(target, warmup=SHIPPED.warmup)[0, 0]

    # Strictly increasing by one raw sample throughout: contiguous, in order, nothing repeated.
    assert torch.equal(torch.diff(sliced), torch.ones(layout.n_samples - 1, dtype=torch.float64))


def test_every_tau_slice_stays_inside_the_record(layout) -> None:
    r"""The last slice closes at $R(T_{\mathrm{valid}} + H)$, which is exactly the end of the
    record -- the construction uses the whole trace and never reads past it."""
    assert SHIPPED.r * (SHIPPED.t_valid + SHIPPED.horizon) == SHIPPED.raw_len
    assert layout.n_samples == (SHIPPED.t_valid - SHIPPED.warmup) * SHIPPED.r


def test_window_validity_is_an_exact_all_over_the_anchors_a_window_spans(layout) -> None:
    """Whole-window dropping, hand-checked against a mask with one zeroed anchor.

    A window is 32 anchors and the hop is 16, so a single invalid anchor falls inside at most two
    windows -- and both must go. Interpolating instead would put a deterministic ramp into a
    spectral estimate, in exactly the low-frequency bands this analysis is read for.
    """
    weight = torch.ones(1, SHIPPED.t, dtype=torch.float64)
    weight[0, 100] = 0.0
    mask, _coverage = forecast_mask(weight, SHIPPED, coverage_floor=0.9)

    keep = metrics.tau_slice_window_validity(mask, layout=layout)

    assert keep.shape == (1, SHIPPED.horizon, layout.n_windows)
    # Anchor 100 is invalid, and so is every anchor whose forecast window covers step 100.
    invalid_anchors = {
        anchor
        for anchor in range(SHIPPED.warmup, SHIPPED.t_valid)
        if float(mask[0, anchor].max()) == 0.0
    }
    assert invalid_anchors, "the fixture must actually invalidate an anchor"
    for index in range(layout.n_windows):
        start, stop = layout.window_anchor_span(index)
        spans_invalid = any(start <= anchor < stop for anchor in invalid_anchors)
        assert bool(keep[0, 0, index] > 0) is not spans_invalid


def test_a_fully_scored_segment_keeps_every_window(layout) -> None:
    """The complement, so the test above cannot pass by dropping everything."""
    weight = torch.ones(1, SHIPPED.t, dtype=torch.float64)
    mask, _coverage = forecast_mask(weight, SHIPPED, coverage_floor=0.9)

    keep = metrics.tau_slice_window_validity(mask, layout=layout)

    assert int((keep > 0).sum()) == SHIPPED.horizon * layout.n_windows


# =============================================================================
# The batch accumulator
# =============================================================================
def _synthetic_batch(
    *, gain: float = 0.8, noise: float = 0.1, seed: int = 0, invalid_steps: tuple = ()
) -> dict:
    r"""A model-shaped batch: a raw trace, its $(B, T_{\mathrm{valid}}, H, R)$ target, two forecast
    branches built from it at a known gain, a raw UP trace, and the real forecast mask."""
    rng = np.random.default_rng(seed)
    fhr = torch.tensor(rng.standard_normal((2, SHIPPED.raw_len)), dtype=torch.float32)
    up = torch.tensor(rng.standard_normal((2, SHIPPED.raw_len)), dtype=torch.float32)
    target = build_future_target(fhr, SHIPPED)
    generator = torch.Generator().manual_seed(seed)
    weight = torch.ones(2, SHIPPED.t, dtype=torch.float32)
    for step in invalid_steps:
        weight[:, step] = 0.0
    mask, _coverage = forecast_mask(weight, SHIPPED, coverage_floor=0.9)
    return {
        "fhr": fhr,
        "up": up,
        "target": target,
        "mu_base": gain * 0.5 * target
        + noise * torch.randn(target.shape, generator=generator),
        "mu_full": gain * target + noise * torch.randn(target.shape, generator=generator),
        "mask": mask,
        "weight": weight,
    }


def test_the_parseval_identity_holds_on_a_model_shaped_batch(layout) -> None:
    r"""The FFT side and the time-domain side, accumulated independently in
    :func:`cross_spectral_sums`, agree to float64 precision.

    This is the identity a run reports in its sanity block. It has three independent ways to be
    wrong -- the $N U$ divisor, the one-sided doubling, and which series the mean is removed from
    -- and each is a plain multiplicative error that no other readout in this pipeline would
    notice. The inputs are float32 model output; the arithmetic is float64, so what this measures
    is the estimator rather than the precision.
    """
    batch = _synthetic_batch()

    sums = metrics.cross_spectral_sums(
        batch["target"], batch["mu_base"], batch["mu_full"], batch["up"], batch["mask"],
        layout=layout,
    )

    for branch in ("base", "full"):
        spectral = (
            sums["sxx"] + sums[f"syy_{branch}"] - 2.0 * sums[f"sxy_{branch}_re"]
        ).sum(dim=-1)
        time_domain = sums[f"ss_detrended_{branch}"]
        assert torch.allclose(spectral, time_domain, rtol=1e-10, atol=1e-12)


def test_the_accumulator_recovers_an_injected_spectral_gain(layout) -> None:
    r"""A forecast built as $g \cdot x$ plus a little noise reads back at gain $g$.

    The known-answer test for the whole chain: slice, window, transform, accumulate, ratio. A
    transposed reshape or a mis-scaled taper would still produce a plausible coherence, but not the
    gain that was put in.
    """
    batch = _synthetic_batch(gain=0.8, noise=0.02)

    sums = metrics.cross_spectral_sums(
        batch["target"], batch["mu_base"], batch["mu_full"], batch["up"], batch["mask"],
        layout=layout,
    )
    out = spectra.derive(
        sums["sxx"][0, 0].numpy(),
        sums["syy_full"][0, 0].numpy(),
        (sums["sxy_full_re"] + 1j * sums["sxy_full_im"])[0, 0].numpy(),
    )

    assert np.nanmean(out["gain"]) == pytest.approx(0.8, rel=0.05)
    assert np.nanmean(out["coherence"]) > 0.95
    assert np.nanmax(np.abs(out["decomposition_residual"])) < 1e-10


def test_no_invalid_sample_reaches_the_estimate(layout) -> None:
    r"""**The non-interpolation proof.** Poison every invalid anchor's forecast and truth with
    $\pm 10^9$ and assert every accumulator is *bit-identical* to the clean run.

    Far stronger than checking a window count: it shows the dropped windows contribute nothing at
    all, rather than contributing something small. A gap is stored as $0$ bpm -- about $-11\sigma$
    after z-scoring, a finite number and the deepest deceleration in the recording -- so an
    estimator that merely tolerated gaps would produce a perfectly finite, entirely wrong spectrum.
    """
    invalid = (100, 101, 200)
    clean = _synthetic_batch(invalid_steps=invalid)
    poisoned = _synthetic_batch(invalid_steps=invalid)

    # Wherever the mask is zero, replace the truth and both branches with an absurd value.
    dead = clean["mask"] == 0
    assert bool(dead.any()), "the fixture must actually mask something"
    for name in ("target", "mu_base", "mu_full"):
        poisoned[name] = poisoned[name].clone()
        poisoned[name][dead] = 1e9

    arguments = dict(layout=layout)
    before = metrics.cross_spectral_sums(
        clean["target"], clean["mu_base"], clean["mu_full"], clean["up"], clean["mask"], **arguments
    )
    after = metrics.cross_spectral_sums(
        poisoned["target"], poisoned["mu_base"], poisoned["mu_full"], poisoned["up"],
        poisoned["mask"], **arguments,
    )

    assert set(before) == set(after)
    for name, value in before.items():
        assert torch.equal(value, after[name]), f"{name} changed when invalid samples were poisoned"
    # And the drop was real: fewer windows survived than the geometry allows.
    assert float(before["n_windows"].min()) < float(before["n_windows_possible"].min())


def test_a_masked_anchor_costs_the_windows_that_span_it_and_no_others(layout) -> None:
    """The count itself, hand-checked. A window is 32 anchors at a hop of 16, so one invalid anchor
    in the interior falls inside exactly two windows."""
    batch = _synthetic_batch(invalid_steps=(150,))

    sums = metrics.cross_spectral_sums(
        batch["target"], batch["mu_base"], batch["mu_full"], batch["up"], batch["mask"],
        layout=layout,
    )

    invalid_anchors = {
        anchor
        for anchor in range(SHIPPED.warmup, SHIPPED.t_valid)
        if float(batch["mask"][0, anchor].max()) == 0.0
    }
    expected = sum(
        1
        for index in range(layout.n_windows)
        if not any(
            layout.window_anchor_span(index)[0] <= anchor < layout.window_anchor_span(index)[1]
            for anchor in invalid_anchors
        )
    )
    assert float(sums["n_windows"][0, 0]) == float(expected)
    assert expected < layout.n_windows


def test_the_source_slice_is_the_contemporaneous_uterine_pressure(layout) -> None:
    r"""The UP slice for lead $\tau$ covers the same raw span the forecast does -- the pressure
    *during* the window being forecast, which the model never read.

    That is what makes the source coherence a statement about anticipation rather than about
    read-off, and it is why the alignment is asserted rather than assumed.
    """
    ramp = torch.arange(SHIPPED.raw_len, dtype=torch.float64)[None, :]

    source = metrics.source_tau_slices(ramp, layout=layout)
    forecast = metrics.tau_slices(build_future_target(ramp, SHIPPED), warmup=SHIPPED.warmup)

    assert source.shape == forecast.shape
    assert torch.equal(source, forecast)


def test_a_batch_with_no_source_omits_the_source_statistics_rather_than_zeroing_them(
    layout,
) -> None:
    """Absent, not zero. A zero ``suu`` would ratio to a coherence of ``NaN`` that reads as
    "measured and found nothing" rather than as "never collected"."""
    batch = _synthetic_batch()

    sums = metrics.cross_spectral_sums(
        batch["target"], batch["mu_base"], batch["mu_full"], None, batch["mask"], layout=layout
    )

    assert "suu" not in sums
    assert not any(name.startswith("sux_") for name in sums)
    assert "sxx" in sums


def test_a_geometry_too_short_to_hold_a_window_accumulates_nothing(layout) -> None:
    """An empty dict, which is what the analysis reads to record a skip. Zeros would be a
    measurement of a geometry that measured nothing."""
    tiny = spectra.slice_geometry(t_valid=16, warmup=4, horizon=4, raw_per_step=16)
    batch = _synthetic_batch()

    sums = metrics.cross_spectral_sums(
        batch["target"], batch["mu_base"], batch["mu_full"], batch["up"], batch["mask"], layout=tiny
    )

    assert sums == {}


# =============================================================================
# Slice and window geometry
# =============================================================================
def test_the_shipped_layout_is_the_one_the_document_describes(layout) -> None:
    """The numbers ``EVAL.md`` quotes, pinned. A silent change to any of them would move every
    frequency axis and every lead time in the analysis without failing anything else."""
    assert layout.n_anchors == 240
    assert layout.n_samples == 3840
    assert layout.nperseg == 512
    assert layout.hop == 256
    assert layout.n_windows == 14
    assert layout.n_freq == 257
    assert layout.delta_f_hz == pytest.approx(0.0078125)


def test_windows_span_whole_anchors_which_is_what_makes_the_gap_rule_exact(layout) -> None:
    r"""The window and the hop are multiples of $R$, so a Welch window is a whole number of
    anchors.

    This is the property the drop rule depends on. The forecast mask is constant within a decimated
    step; if a window began or ended mid-step, its validity would be a judgement about a partially
    covered step rather than an ``all()`` over the mask, and the estimator would need a tolerance
    where it currently needs none.
    """
    assert layout.nperseg % SHIPPED.r == 0
    assert layout.hop % SHIPPED.r == 0

    # The windows tile the trained-anchor range and the last one ends exactly on its end.
    assert layout.window_anchor_span(0) == (30, 62)
    assert layout.window_anchor_span(layout.n_windows - 1) == (238, 270)
    for index in range(layout.n_windows):
        start, stop = layout.window_anchor_span(index)
        assert SHIPPED.warmup <= start < stop <= SHIPPED.t_valid


def test_the_lead_time_axis_tiles_the_horizon_without_overlap_or_gap(layout) -> None:
    r"""Slice $\tau$ holds lead times $[4\tau + 0.25,\ 4\tau + 4]$ s, and the thirty slices tile
    $0$--$120$ s. That axis is the whole reason the construction is used instead of an STFT inside
    one forecast block."""
    assert layout.lead_seconds(0) == pytest.approx((0.25, 4.0))
    assert layout.lead_seconds(29) == pytest.approx((116.25, 120.0))

    spans = [layout.lead_seconds(tau) for tau in range(SHIPPED.horizon)]
    for (_, previous_hi), (next_lo, _) in zip(spans, spans[1:]):
        # Consecutive slices meet at one raw sample's spacing: no overlap, and nothing skipped.
        assert next_lo - previous_hi == pytest.approx(1.0 / spectra.FS_RAW)


def test_a_geometry_too_short_for_one_window_reports_zero_rather_than_raising() -> None:
    """A tiny geometry is a fact about the run, not a bug in it. ``n_windows == 0`` is what the
    analysis reads to record a skip; a raise here would take the whole pass down instead."""
    tiny = spectra.slice_geometry(t_valid=16, warmup=4, horizon=4, raw_per_step=16)
    assert tiny.n_windows == 0


def test_a_hop_longer_than_the_window_is_refused() -> None:
    """Consecutive windows would leave un-analysed samples between them, so the spectrum would
    describe a subsample of the slice while its window count claimed otherwise."""
    with pytest.raises(ValueError, match="hop_steps"):
        spectra.slice_geometry(
            t_valid=270, warmup=30, horizon=30, raw_per_step=16, nperseg_steps=8, hop_steps=16
        )


# =============================================================================
# Bands
# =============================================================================
def test_the_bands_partition_every_bin_exactly_once(layout) -> None:
    """Parseval holds here as a sum over bands only if the bands partition the bins. A band table
    that dropped one would break the reconciliation by an amount that reads as a normalisation
    bug rather than as a missing band."""
    axis = spectra.frequency_axis(layout.nperseg)
    assigned = spectra.band_index(axis)

    assert assigned.shape == (layout.n_freq,)
    assert (assigned >= 0).all()
    assert set(assigned.tolist()) == set(range(len(spectra.BANDS_HZ)))


def test_the_band_bin_counts_are_the_ones_the_document_quotes(layout) -> None:
    r"""Including the DC bin in ``vlf``, which is a choice rather than an oversight: the per-window
    mean is removed but the Hann taper leaves a residue, so bin $0$ carries signal and must live in
    some band for the identity to close."""
    axis = spectra.frequency_axis(layout.nperseg)
    counts = spectra.band_bin_counts(axis)

    assert counts == {"vlf": 4, "lf": 16, "mf": 44, "hf": 64, "noise": 129}
    assert sum(counts.values()) == layout.n_freq


def test_a_band_table_that_leaves_a_bin_uncovered_is_refused(layout) -> None:
    """Named rather than silent: an uncovered bin is invisible downstream except as a broken
    identity, which a reader would attribute to the estimator."""
    axis = spectra.frequency_axis(layout.nperseg)
    with pytest.raises(ValueError, match="falls in no band"):
        spectra.band_index(axis, {"low": (0.0, 0.5)})


def test_the_seam_bins_are_exact_bin_indices(layout) -> None:
    r"""The seam period is $R$ raw samples, so harmonic $k$ lands on bin $k \cdot
    \mathrm{nperseg}/R$ -- an integer, which is a third reason ``nperseg`` is a multiple of $R$. A
    seam frequency between bins would smear across the very neighbourhood it is compared against.
    """
    bins = spectra.seam_bins(layout.nperseg, SHIPPED.r)
    axis = spectra.frequency_axis(layout.nperseg)

    assert bins.tolist() == [32, 64, 96, 128, 160, 192, 224, 256]
    assert axis[bins[0]] == pytest.approx(spectra.FS_RAW / SHIPPED.r)
    assert axis[bins[0]] == pytest.approx(0.25)


def test_the_seam_fundamental_sits_inside_a_band_rather_than_on_its_edge(layout) -> None:
    """The reason this analysis does not reuse ``CLINICAL_BANDS``: that table's 0.25 Hz edge is
    exactly the seam frequency, and a band boundary is the worst place for an artifact to sit."""
    axis = spectra.frequency_axis(layout.nperseg)
    assigned = spectra.band_index(axis)
    names = spectra.band_names()
    fundamental = int(spectra.seam_bins(layout.nperseg, SHIPPED.r)[0])

    assert names[assigned[fundamental]] == "mf"
    # Strictly interior: its neighbours on both sides are in the same band.
    assert assigned[fundamental - 1] == assigned[fundamental] == assigned[fundamental + 1]


# =============================================================================
# The estimator's known answers
# =============================================================================
def _spectra_of(x: np.ndarray, y: np.ndarray, nperseg: int) -> tuple:
    """Return one window's $(S_{xx}, S_{yy}, S_{xy})$ under the module's stated convention."""
    window = spectra.welch_window(nperseg)
    weights = spectra.one_sided_weights(nperseg)
    scale = float(window @ window)
    xw = window * (x - x.mean())
    yw = window * (y - y.mean())
    fx = np.fft.rfft(xw)
    fy = np.fft.rfft(yw)
    denominator = nperseg * scale
    return (
        weights * (np.conj(fx) * fx).real / denominator,
        weights * (np.conj(fy) * fy).real / denominator,
        weights * np.conj(fx) * fy / denominator,
    )


def test_a_perfect_forecast_scores_one_and_costs_nothing(layout) -> None:
    r"""$y = x$: $\gamma^2 = 1$, $g = 1$, $\phi = 0$, and all three error terms vanish."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(layout.nperseg)
    sxx, syy, sxy = _spectra_of(x, x.copy(), layout.nperseg)

    out = spectra.derive(sxx, syy, sxy)
    usable = np.isfinite(out["coherence"])

    assert out["coherence"][usable] == pytest.approx(1.0)
    assert out["gain"][usable] == pytest.approx(1.0)
    assert out["phase_rad"][usable] == pytest.approx(0.0, abs=1e-12)
    assert out["irreducible"][usable] == pytest.approx(0.0, abs=1e-12)
    assert out["timing"][usable] == pytest.approx(0.0, abs=1e-12)
    assert out["amplitude"][usable] == pytest.approx(0.0, abs=1e-12)


def test_pure_attenuation_reads_as_amplitude_error_and_not_as_lost_coherence(layout) -> None:
    r"""$y = 0.5x$ -- the over-smoothing signature. Coherence stays at $1$ and the whole error is
    the amplitude term, $(g - 1)^2 = 0.25$.

    This is the case that makes the decomposition worth emitting: a coherence of $1.0$ alone would
    report this forecast as perfect, while it carries a quarter of the truth's variance as error.
    """
    rng = np.random.default_rng(1)
    x = rng.standard_normal(layout.nperseg)
    sxx, syy, sxy = _spectra_of(x, 0.5 * x, layout.nperseg)

    out = spectra.derive(sxx, syy, sxy)
    usable = np.isfinite(out["coherence"])

    assert out["coherence"][usable] == pytest.approx(1.0)
    assert out["gain"][usable] == pytest.approx(0.5)
    assert out["timing"][usable] == pytest.approx(0.0, abs=1e-12)
    assert out["amplitude"][usable] == pytest.approx(0.25)
    assert out["residual_normalised"][usable] == pytest.approx(0.25)


def test_the_two_error_terms_separate_the_two_failure_modes_cleanly(layout) -> None:
    r"""``timing`` vanishes exactly at $\phi = 0$ for any gain, and ``amplitude`` exactly at
    $g = \gamma$ for any phase. Neither leaks into the other.

    That separation is the whole reason for choosing this split over the algebraically equivalent
    $\gamma^2\sin^2\phi + (g - \gamma\cos\phi)^2$, in which a pure delay reports amplitude error.
    """
    rng = np.random.default_rng(11)
    gamma = rng.uniform(0.1, 1.0, size=500)
    gain = rng.uniform(0.1, 3.0, size=500)
    sxx = np.ones_like(gain)

    # Right phase, arbitrary gain -> no timing error.
    aligned = spectra.derive(sxx, gain**2, gamma * gain + 0j)
    assert aligned["timing"] == pytest.approx(0.0, abs=1e-12)

    # Arbitrary phase, gain equal to the coherent amplitude -> no amplitude error.
    phase = rng.uniform(-np.pi, np.pi, size=500)
    matched = spectra.derive(sxx, gamma**2, gamma**2 * np.exp(1j * phase))
    assert matched["gain"] == pytest.approx(gamma)
    assert matched["amplitude"] == pytest.approx(0.0, abs=1e-12)


def test_a_pure_delay_is_entirely_timing_error_and_costs_no_amplitude(layout) -> None:
    r"""$y(t) = x(t - d)$: coherence $1$, gain $1$, amplitude term $0$, and the whole normalised
    residual is $2 - 2\cos(2\pi f d)$.

    Built as an exact cross-spectrum rather than from a shifted slice of noise. A shifted *window*
    of a random sequence is not a delayed copy of what the window saw -- the two contain different
    samples at their edges -- so it carries a real gain fluctuation of about $\pm 10\%$ and would
    test the construction rather than the algebra. The windowing path has its own test, below.

    The sign convention is pinned here because it is the one thing about a phase that is
    conventional rather than mathematical: **a positive delay means the forecast lags the truth**,
    and the cross-spectral phase is correspondingly negative.
    """
    axis = spectra.frequency_axis(layout.nperseg)[1:200]
    delay_s = 2.0
    sxx = np.ones_like(axis)
    syy = np.ones_like(axis)
    sxy = np.exp(-2j * np.pi * axis * delay_s)

    out = spectra.derive(sxx, syy, sxy)
    turn = 2.0 * np.pi * axis * delay_s

    assert out["coherence"] == pytest.approx(1.0)
    assert out["gain"] == pytest.approx(1.0)
    # Compared on the unit circle, because the reported phase is wrapped to $(-\pi, \pi]$ while
    # $2\pi f d$ is not -- at $d = 2$ s it passes a full turn by $0.5$ Hz. That wrapping is exactly
    # why `estimate_delay` searches instead of unwrapping.
    assert np.exp(1j * out["phase_rad"]) == pytest.approx(np.exp(-1j * turn))
    # The whole error is timing. The alternative split -- $\gamma^2\sin^2\phi$ with
    # $(g - \gamma\cos\phi)^2$ -- would report $(1 - \cos\phi)^2$ of amplitude error here, for a
    # forecast whose amplitude is exactly right.
    assert out["amplitude"] == pytest.approx(0.0, abs=1e-12)
    assert out["timing"] == pytest.approx(2.0 - 2.0 * np.cos(turn))
    assert out["residual_normalised"] == pytest.approx(2.0 - 2.0 * np.cos(turn))

    recovered, concentration = spectra.estimate_delay(sxy, axis)
    assert float(recovered) == pytest.approx(delay_s)
    assert float(concentration) == pytest.approx(1.0)


def test_the_sign_convention_survives_a_real_windowed_shift(layout) -> None:
    """The convention pinned above, re-checked through the actual windowing and detrending path on
    a genuinely shifted signal -- where the gain fluctuates and the coherence is merely near one."""
    rng = np.random.default_rng(2)
    lag = 8  # raw samples, i.e. 2 s at 4 Hz
    source = rng.standard_normal(layout.nperseg + 4 * lag)
    x = source[4 * lag : 4 * lag + layout.nperseg]
    y = source[3 * lag : 3 * lag + layout.nperseg]  # y is x delayed by `lag` samples
    sxy = _spectra_of(x, y, layout.nperseg)[2]
    axis = spectra.frequency_axis(layout.nperseg)
    band = slice(1, 200)

    recovered, concentration = spectra.estimate_delay(sxy[band], axis[band])

    assert float(recovered) == pytest.approx(lag / spectra.FS_RAW)
    assert float(concentration) > 0.9


def test_a_delay_is_identifiable_only_within_the_alias_period(layout) -> None:
    r"""A uniform bin grid cannot separate $d$ from $d + 1/\Delta f$: the rotation is identical at
    every bin. At the shipped $\Delta f$ that period is $128$ s, so the principal interval is
    $\pm 64$ s.

    A wider search is **refused** rather than served. Outside the interval the grid holds exact
    ties, and which one an ``argmax`` reaches first is an implementation detail; returning it would
    dress that up as a measurement. Phase unwrapping has the same limit -- this is a property of
    the grid, not a cost of searching rather than fitting.
    """
    axis = spectra.frequency_axis(layout.nperseg)[1:200]
    assert spectra.delay_alias_period(axis) == pytest.approx(128.0)

    with pytest.raises(ValueError, match="alias limit"):
        spectra.estimate_delay(np.ones_like(axis, dtype=np.complex128), axis, max_seconds=120.0)

    # A delay outside the interval is reported as its principal value, not as itself.
    aliased = np.exp(-2j * np.pi * axis * 100.0)
    recovered, _ = spectra.estimate_delay(aliased, axis)
    assert float(recovered) == pytest.approx(100.0 - 128.0)


def test_a_single_bin_supports_no_delay_at_all() -> None:
    """One bin's phase constrains a delay only modulo $1/f$, which is not a constraint. ``NaN``
    rather than an arbitrary grid point."""
    recovered, concentration = spectra.estimate_delay(
        np.array([1.0 + 0.0j]), np.array([0.1])
    )
    assert np.isnan(recovered).all()
    assert np.isnan(concentration).all()


def test_one_window_is_exactly_coherent_which_is_why_nothing_ratios_early(layout) -> None:
    r"""Two unrelated signals score $\gamma^2 = 1$ on a single window, exactly.

    This is the non-vacuity test for the whole storage design. Magnitude-squared coherence carries
    no information until cross-spectra are *averaged*, so an implementation that ratioed per
    segment -- or per window -- would report perfect coherence everywhere and look plausible doing
    it. All averaging must come from accumulation.
    """
    rng = np.random.default_rng(3)
    x = rng.standard_normal(layout.nperseg)
    y = rng.standard_normal(layout.nperseg)
    sxx, syy, sxy = _spectra_of(x, y, layout.nperseg)

    out = spectra.derive(sxx, syy, sxy)
    usable = np.isfinite(out["coherence"])
    assert out["coherence"][usable] == pytest.approx(1.0)


def test_averaging_cross_spectra_drives_independent_signals_towards_zero(layout) -> None:
    r"""The counterpart: accumulate many windows and the bias falls as $(1 - \gamma^2)/n_d$."""
    rng = np.random.default_rng(4)
    n_windows = 100
    sxx = np.zeros(layout.n_freq)
    syy = np.zeros(layout.n_freq)
    sxy = np.zeros(layout.n_freq, dtype=np.complex128)
    for _ in range(n_windows):
        one = _spectra_of(
            rng.standard_normal(layout.nperseg), rng.standard_normal(layout.nperseg), layout.nperseg
        )
        sxx += one[0]
        syy += one[1]
        sxy += one[2]

    coherence = spectra.derive(sxx, syy, sxy)["coherence"]
    assert np.nanmean(coherence) < 0.05


def test_summing_then_ratioing_equals_the_ratio_of_the_pooled_windows(layout) -> None:
    """Two segments' stored sums, added and then ratioed, give exactly what one pass over both
    window sets gives. That equality is what lets the per-recording and per-cohort reductions be
    plain sums, and it is the reason nothing stored is a ratio."""
    rng = np.random.default_rng(5)
    parts = []
    for _ in range(2):
        sxx = np.zeros(layout.n_freq)
        syy = np.zeros(layout.n_freq)
        sxy = np.zeros(layout.n_freq, dtype=np.complex128)
        for _ in range(5):
            x = rng.standard_normal(layout.nperseg)
            one = _spectra_of(x, 0.7 * x + rng.standard_normal(layout.nperseg), layout.nperseg)
            sxx += one[0]
            syy += one[1]
            sxy += one[2]
        parts.append((sxx, syy, sxy))

    combined = spectra.derive(
        parts[0][0] + parts[1][0], parts[0][1] + parts[1][1], parts[0][2] + parts[1][2]
    )
    pooled = spectra.derive(
        sum(part[0] for part in parts),
        sum(part[1] for part in parts),
        sum(part[2] for part in parts),
    )
    assert combined["coherence"] == pytest.approx(pooled["coherence"], rel=1e-15)


# =============================================================================
# The identities
# =============================================================================
def test_the_three_way_decomposition_is_exact_for_arbitrary_spectra() -> None:
    r"""On random positive-definite triples rather than on a constructed signal, so the identity is
    tested as algebra rather than as a property of one waveform."""
    rng = np.random.default_rng(6)
    sxx = rng.uniform(0.1, 10.0, size=2000)
    syy = rng.uniform(0.1, 10.0, size=2000)
    # Cauchy-Schwarz: |S_xy|^2 <= S_xx S_yy, which any real accumulation satisfies.
    magnitude = rng.uniform(0.0, 1.0, size=2000) * np.sqrt(sxx * syy)
    phase = rng.uniform(-np.pi, np.pi, size=2000)
    sxy = magnitude * np.exp(1j * phase)

    out = spectra.derive(sxx, syy, sxy)

    assert np.abs(out["decomposition_residual"]).max() < 1e-12
    assert (out["irreducible"] >= -1e-12).all()
    assert (out["timing"] >= -1e-12).all()
    assert (out["amplitude"] >= -1e-12).all()


def test_parseval_holds_exactly_under_the_modules_scaling_convention(layout) -> None:
    r"""$\sum_k P_{ee,k}$ equals the windowed, detrended residual sum of squares over $U$.

    The convention has three independent ways to be wrong -- the $N U$ divisor, the one-sided
    doubling, and which series the mean is removed from -- and each is a plain multiplicative error
    that no other test in this repository would notice. It is the identity a run reports in its
    sanity block, so it is proved here on real arithmetic first.
    """
    rng = np.random.default_rng(7)
    x = rng.standard_normal(layout.nperseg)
    y = 0.6 * x + 0.4 * rng.standard_normal(layout.nperseg)
    sxx, syy, sxy = _spectra_of(x, y, layout.nperseg)

    window = spectra.welch_window(layout.nperseg)
    scale = float(window @ window)
    residual = y - x
    reference = float(((window * (residual - residual.mean())) ** 2).sum() / scale)

    assert float((sxx + syy - 2.0 * sxy.real).sum()) == pytest.approx(reference, rel=1e-12)


def test_the_band_sums_reconcile_with_the_full_spectrum(layout) -> None:
    """Because the bands partition the bins, a band-collapsed residual sums back to the whole. The
    per-segment statistics are stored band-collapsed, so this is what keeps the stored form
    reconcilable with the identity."""
    rng = np.random.default_rng(8)
    x = rng.standard_normal(layout.nperseg)
    sxx, syy, sxy = _spectra_of(x, 0.6 * x + rng.standard_normal(layout.nperseg), layout.nperseg)

    axis = spectra.frequency_axis(layout.nperseg)
    assigned = spectra.band_index(axis)
    n_bands = len(spectra.BANDS_HZ)
    banded = (
        spectra.collapse_to_bands(sxx, assigned, n_bands),
        spectra.collapse_to_bands(syy, assigned, n_bands),
        spectra.collapse_to_bands(sxy, assigned, n_bands),
    )

    whole = float((sxx + syy - 2.0 * sxy.real).sum())
    from_bands = float((banded[0] + banded[1] - 2.0 * np.real(banded[2])).sum())
    assert from_bands == pytest.approx(whole, rel=1e-12)


def test_the_seam_ratio_finds_an_injected_seam_and_leaves_a_clean_spectrum_alone(layout) -> None:
    r"""A $16$-sample-periodic sawtooth is exactly what a per-token linear output head can produce
    at a token boundary. The control is the clean signal: without it a raised ratio could be
    whatever the FHR happens to do at $0.25$ Hz."""
    rng = np.random.default_rng(9)
    clean = rng.standard_normal(layout.nperseg)
    sawtooth = np.tile(np.linspace(-1.0, 1.0, SHIPPED.r), layout.nperseg // SHIPPED.r)
    seamed = clean + 3.0 * sawtooth

    bins = spectra.seam_bins(layout.nperseg, SHIPPED.r)
    clean_ratio = spectra.seam_ratio(_spectra_of(clean, clean, layout.nperseg)[0], bins)
    seamed_ratio = spectra.seam_ratio(_spectra_of(seamed, seamed, layout.nperseg)[0], bins)

    # The control is unremarkable at every harmonic; nothing in white noise knows about 0.25 Hz.
    assert np.nanmax(clean_ratio) < 10.0
    # The fundamental is unmistakable, and every harmonic is raised above the control. A sawtooth's
    # harmonics fall as $k^{-2}$ in amplitude, so the eighth is far weaker than the first -- the
    # elementwise comparison is the claim that survives that decay.
    assert seamed_ratio[0] > 50.0
    assert (seamed_ratio > clean_ratio).all()


# =============================================================================
# Stored-vector plumbing
# =============================================================================
def test_the_flattened_vector_round_trips_through_the_shared_reshape() -> None:
    r"""The per-segment statistics travel with one trailing axis, $\tau$-major. Every consumer
    unflattens through one function rather than with its own opinion about which index is major --
    a transposed reshape would silently relabel lead times as bands."""
    n_bands = len(spectra.BANDS_HZ)
    original = np.arange(SHIPPED.horizon * n_bands, dtype=np.float64)

    restored = spectra.reshape_band_horizon(
        original, horizon=SHIPPED.horizon, n_bands=n_bands
    )

    assert restored.shape == (SHIPPED.horizon, n_bands)
    assert restored[0].tolist() == list(range(n_bands))
    assert restored[1, 0] == float(n_bands)


def test_a_vector_from_another_geometry_is_refused_rather_than_reshaped() -> None:
    """An older run's tables carry a different horizon or band count; reshaping them anyway would
    relabel every lead time."""
    with pytest.raises(ValueError, match="different geometry or band table"):
        spectra.reshape_band_horizon(np.zeros(7), horizon=SHIPPED.horizon, n_bands=5)


def test_an_empty_accumulator_reads_as_unmeasured_rather_than_as_perfectly_coherent() -> None:
    """A bin no window contributed to is ``NaN``. Reporting ``0/0`` as coherence ``1.0`` is the
    single most flattering way this could fail, and an all-zero accumulator is exactly what a
    fully masked recording produces."""
    zeros = np.zeros(4)
    out = spectra.derive(zeros, zeros, np.zeros(4, dtype=np.complex128))

    assert np.isnan(out["coherence"]).all()
    assert np.isnan(out["gain"]).all()
    assert np.isnan(out["residual_normalised"]).all()


def test_the_layout_round_trips_through_the_record_a_run_dumps(layout) -> None:
    """An offline re-run rebuilds the layout the pass ran under rather than re-deriving one from
    the shipped constants -- so a run collected at a different window length stays readable, and
    its frequencies and lead times are its own rather than today's."""
    restored = spectra.layout_from_record(layout.describe())

    assert restored == layout


def test_a_record_from_before_this_analysis_rebuilds_no_layout() -> None:
    """``None`` rather than a guess. A layout invented for a run that did not record one would
    relabel every frequency and every lead time in it."""
    assert spectra.layout_from_record(None) is None
    assert spectra.layout_from_record({"nperseg": 512}) is None


# =============================================================================
# The analysis end to end, on a real (non-skip) path
# =============================================================================
def _collected(layout, *, n_segments: int = 12, n_guids: int = 4, gain: float = 0.8) -> dict:
    r"""Build what a collection pass would have left behind, at the **shipped** geometry.

    The smoke run cannot stand in for this. Its fixture model is $T = 16$, $H = 4$, so the trained
    anchors span $160$ raw samples against an ``nperseg`` of $512$: ``n_windows == 0``, the
    accumulator returns ``{}``, and the analysis takes the same skip path a bare context does. So
    the entire reduction -- every CSV, every populated figure builder -- would otherwise run in no
    test at all.

    The band collapse and the pooling mirror ``collect._Collector._accumulate_spectral`` exactly,
    because that is the shape the analysis reads.
    """
    import pandas as pd

    band_index = spectra.band_index(spectra.frequency_axis(layout.nperseg))
    n_bands = len(spectra.band_names())
    rows, blocks, pooled = [], {}, {}
    for index in range(n_segments):
        batch = _synthetic_batch(gain=gain, noise=0.1, seed=index)
        sums = metrics.cross_spectral_sums(
            batch["target"][:1], batch["mu_base"][:1], batch["mu_full"][:1],
            batch["up"][:1], batch["mask"][:1], layout=layout,
        )
        assert sums, "the shipped geometry must produce cross-spectral sums"
        clinical = ["healthy", "acidosis", "hie"][index % 3]
        rows.append({
            "sample_index": index,
            "guid": f"guid{index % n_guids}",
            "epoch": -1000 * index,
            "clinical_class": clinical,
            "subgroup": "healthy_bg_cs" if clinical == "healthy" else f"{clinical}_cs",
            "n_anchors": 240,
        })
        for name, tensor in sums.items():
            array = tensor.numpy()
            if array.ndim == 3:
                collapsed = spectra.collapse_to_bands(array, band_index, n_bands)
                blocks.setdefault(f"coherence_{name}", []).append(
                    collapsed.reshape(array.shape[0], -1)
                )
            else:
                blocks.setdefault(f"coherence_{name}", []).append(array)
            for cohort in ("all", clinical):
                key = f"{cohort}_{name}"
                pooled[key] = array.sum(0) if key not in pooled else pooled[key] + array.sum(0)

    return {
        "per_sample": pd.DataFrame(rows),
        "vectors": {name: np.concatenate(parts, axis=0) for name, parts in blocks.items()},
        "spectra": pooled,
        "record": {
            "geometry": {"raw_per_step": SHIPPED.r},
            "coherence": {**layout.describe(), "bands": spectra.band_edges()},
        },
    }


@pytest.fixture(scope="module")
def analysed(layout, tmp_path_factory):
    """Run the analysis once on a real collection, and let the assertions read its output."""
    import types

    from teb_vae.lag_attn_rws.eval.analyses import coherence as analysis
    from teb_vae.lag_attn_rws.eval.figures_seam import configure_figure_style

    configure_figure_style()
    collected = _collected(layout)
    collection = types.SimpleNamespace(**collected, results={})
    context = types.SimpleNamespace(collection=collection, config={}, task=None, loader=None)
    directory = tmp_path_factory.mktemp("coherence_run")

    result = analysis.run_coherence_analysis(
        context, eval_config={"bootstrap_resamples": 60, "seed": 42}, output_dir=directory
    )
    return {"result": result, "directory": directory / analysis.ANALYSIS_DIRNAME}


def test_the_analysis_reports_the_protocols_keys_on_a_real_run(analysed) -> None:
    """Segments, not recordings: the coverage block compares ``n_samples`` against every other
    analysis, and a recording count would read as a population disagreement with all of them."""
    result = analysed["result"]

    assert result.get("skipped") is None
    assert result["n_samples"] == 12
    assert result["composition"] == {"n_recordings": 4}
    assert result["plan"]["capped"] is False
    assert result["grouped_frames"], "the runner's by-cohort fan-out needs a declared frame"


def test_every_emitted_table_is_readable_and_carries_its_promised_columns(analysed) -> None:
    """Each CSV parses and holds the columns the guide and the consumers name. A table that opens
    is the minimum; ``cross_subgroup`` reads one of these off disk."""
    import pandas as pd

    from teb_vae.lag_attn_rws.eval.analyses import coherence as analysis

    directory = analysed["directory"]
    for name in (
        analysis.BANDS_FILENAME, analysis.PER_RECORDING_FILENAME, analysis.SPECTRUM_FILENAME,
        analysis.COHORT_SPECTRUM_FILENAME, analysis.SOURCE_FILENAME, analysis.SEAM_FILENAME,
    ):
        frame = pd.read_csv(directory / name)
        assert not frame.empty, f"{name} is empty on a real run"

    bands = pd.read_csv(directory / analysis.BANDS_FILENAME)
    assert {"branch", "hrv_band", "tau", "lead_seconds", "coherence", "gain", "n_bins"} <= set(
        bands.columns
    )
    # The delay never travels without the number that says whether it means anything.
    assert {"group_delay_s", "group_delay_concentration"} <= set(bands.columns)

    per_recording = pd.read_csv(directory / analysis.PER_RECORDING_FILENAME)
    # The exact columns cross_subgroup and the grouped fan-out resolve by name.
    from teb_vae.lag_attn_rws.eval.analyses import cross_subgroup

    wanted = {
        source.column
        for source in cross_subgroup.METRIC_SOURCES
        if source.analysis == analysis.ANALYSIS_DIRNAME
    } | set(analysis.GROUPED_METRICS)
    assert wanted <= set(per_recording.columns), sorted(wanted - set(per_recording.columns))


def test_the_real_run_recovers_the_injected_gain_and_reports_finite_headline_scalars(
    analysed,
) -> None:
    r"""The forecast was built as $0.8x$ plus noise, so the whole chain -- slice, window, transform,
    accumulate, sum per recording, ratio, average -- must read a gain of $0.8$ back.

    The headline scalars must also be **finite**: ``check_headline_finite`` fails a run on a
    non-finite number, so a NaN here is a sanity failure rather than a missing measurement.
    """
    import math

    headline = analysed["result"]["headline"]

    assert headline["gain_full_lf"] == pytest.approx(0.8, rel=0.05)
    assert 0.0 <= headline["coherence_full_lf"] <= 1.0
    for name, value in headline.items():
        assert value is None or math.isfinite(float(value)), f"{name} is not finite"


def test_the_real_run_reconciles_the_spectral_residual_with_the_time_domain(analysed) -> None:
    """The identity the sanity block gates on, measured through the analysis rather than through
    the accumulator alone."""
    reconciliation = analysed["result"]["reconciliation"]

    assert reconciliation["parseval_compared"] is True
    assert reconciliation["parseval_max_relative_error"] < 1e-10
    # Detrending removes only the level, which for this forecast is a small share of the residual.
    assert 0.9 < reconciliation["detrended_share_of_raw_full"] <= 1.0


def test_the_lead_time_axis_on_the_emitted_table_is_the_layouts_own(analysed, layout) -> None:
    """The lead times in the CSV must be the layout's, not a re-derived approximation of them."""
    import pandas as pd

    from teb_vae.lag_attn_rws.eval.analyses import coherence as analysis

    bands = pd.read_csv(analysed["directory"] / analysis.BANDS_FILENAME)
    observed = bands[(bands["branch"] == "full") & (bands["hrv_band"] == "lf")].sort_values("tau")

    assert observed["lead_seconds"].to_numpy() == pytest.approx(layout.lead_center_seconds())


def test_the_real_run_result_is_json_safe(analysed) -> None:
    """``summary.json`` is written with ``allow_nan=False``, so a NaN or a numpy scalar anywhere in
    the returned block raises at the very end of a multi-hour run."""
    import json

    from teb_vae.lag_attn_rws.eval.report_seam import json_safe

    json.dumps(json_safe(analysed["result"]), allow_nan=False)


def test_the_real_run_renders_every_figure(analysed) -> None:
    """All six, on populated data rather than on the empty-frame path."""
    from teb_vae.lag_attn_rws.eval.analyses import coherence as analysis

    for name in (
        figure_filename(analysis.LEAD_TIME_FIGURE), figure_filename(analysis.SPECTRUM_FIGURE), figure_filename(analysis.BANDS_FIGURE),
        figure_filename(analysis.DECOMPOSITION_FIGURE), figure_filename(analysis.SOURCE_FIGURE), figure_filename(analysis.SEAM_FIGURE),
    ):
        path = analysed["directory"] / name
        assert path.is_file() and path.stat().st_size > 1000, f"{name} did not render"


def test_a_batch_without_a_source_does_not_shift_every_later_row(layout) -> None:
    r"""The vectors sidecar is aligned to ``per_sample.csv`` **by position**.

    A statistic that exists on one batch and not the next -- which is exactly what the source
    statistics do, since they need a raw UP trace -- would otherwise end up short by that batch's
    rows while every other key grew. From the first such batch onward one segment's spectra would
    sit on another segment's row, and every per-recording and per-cohort number downstream would be
    reading the wrong recording. Nothing would raise.
    """
    import types

    from teb_vae.lag_attn_rws.eval import collect

    collector = collect._Collector(
        collect.RetentionPlan.build({}, n_total=4, seed=0), geometry=SHIPPED
    )
    batch = _synthetic_batch()
    with_source = metrics.cross_spectral_sums(
        batch["target"][:1], batch["mu_base"][:1], batch["mu_full"][:1],
        batch["up"][:1], batch["mask"][:1], layout=layout,
    )
    without_source = metrics.cross_spectral_sums(
        batch["target"][:1], batch["mu_base"][:1], batch["mu_full"][:1],
        None, batch["mask"][:1], layout=layout,
    )
    assert "suu" in with_source and "suu" not in without_source

    scored = np.array([True])
    for sums in (with_source, without_source, with_source):
        collector._accumulate_spectral(
            {}, types.SimpleNamespace(spectral_sums=sums), 1, scored
        )

    stacked = {
        name: np.concatenate(parts, axis=0) for name, parts in collector._spectral.items()
    }
    lengths = {name: array.shape[0] for name, array in stacked.items()}
    assert set(lengths.values()) == {3}, f"row counts disagree: {lengths}"
    # The middle batch carried no source, and says so as NaN rather than by being absent.
    assert np.isnan(stacked["coherence_suu"][1]).all()
    assert np.isfinite(stacked["coherence_suu"][0]).any()
    assert np.isfinite(stacked["coherence_suu"][2]).any()


def test_a_reconciliation_that_compared_nothing_reports_nan_rather_than_zero() -> None:
    r"""The one exact correctness gate on the estimator must not certify itself.

    ``reconciliation``'s worst-case is a maximum, so a seed of $0.0$ that no comparison ever raises
    reaches ``check_coherence_parseval`` as a finite value below the tolerance -- and the check
    reports PASS, "matches the time domain to 0", having compared nothing at all. That is exactly
    the state it exists to catch: a run whose stored sums carry no time-domain reference, or whose
    every window was dropped. ``NaN`` routes it to INCONCLUSIVE instead.
    """
    import math

    from teb_vae.lag_attn_rws.eval.analyses import coherence as analysis
    from teb_vae.lag_attn_rws.eval import report_seam

    # Cross-spectra present, but no ss_detrended_* reference to compare them against.
    partial = {
        "sxx": np.ones((2, 3, 5)),
        "syy_base": np.ones((2, 3, 5)),
        "sxy_base_re": np.ones((2, 3, 5)),
    }

    out = analysis.reconciliation(partial)

    assert out["parseval_compared"] is False
    assert math.isnan(out["parseval_max_relative_error"])
    verdict = report_seam.check_coherence_parseval({"coherence": {"reconciliation": out}})
    assert verdict["verdict"] == report_seam.INCONCLUSIVE, verdict


def test_a_real_reconciliation_still_passes_the_gate(analysed) -> None:
    """The complement, so the test above cannot pass by making the gate permanently inconclusive."""
    from teb_vae.lag_attn_rws.eval import report_seam

    verdict = report_seam.check_coherence_parseval(
        {"coherence": {"reconciliation": analysed["result"]["reconciliation"]}}
    )

    assert verdict["verdict"] == "pass", verdict


def test_the_stored_sums_survive_the_round_trip_to_disk_bit_for_bit(tmp_path) -> None:
    """The sums are float64 and must come back as float64 exactly.

    This is why they travel in the ``npz`` sidecars rather than as columns of ``per_sample.csv``:
    CSV round-trips a float only through its decimal form, which is the whole reason
    ``PER_SAMPLE_FLOAT_PRECISION`` exists. A last-bit difference would make the same run report
    different coherences depending on whether an analysis read the tables in memory or off disk --
    exactly the property the offline re-run path exists to have.
    """
    import pandas as pd

    from teb_vae.lag_attn_rws.eval import collect

    rng = np.random.default_rng(12)
    vectors = {"coherence_sxx": rng.standard_normal((3, 150))}
    pooled = {"all_sxx": rng.standard_normal((30, 257))}
    collection = collect.Collection(
        per_sample=pd.DataFrame({"guid": ["a", "b", "c"], "sample_index": [0, 1, 2]}),
        per_anchor=pd.DataFrame({"guid": ["a"], "epoch": [0], "anchor": [0]}),
        vectors=vectors,
        spectra=pooled,
        record={"n_per_sample_rows": 3, "n_per_anchor_rows": 1},
    )

    collect.write_collection(collection, tmp_path)
    restored = collect.load_collection(tmp_path)

    assert (tmp_path / collect.COHERENCE_FILENAME).is_file()
    assert set(restored.spectra) == set(pooled)
    for name, array in pooled.items():
        assert restored.spectra[name].dtype == np.float64
        assert np.array_equal(restored.spectra[name], array)
    assert np.array_equal(restored.vectors["coherence_sxx"], vectors["coherence_sxx"])


def test_a_directory_without_the_spectra_sidecar_loads_with_an_empty_one(tmp_path) -> None:
    """A run collected before this analysis existed still loads; the analysis reports the skip."""
    import pandas as pd

    from teb_vae.lag_attn_rws.eval import collect

    collection = collect.Collection(
        per_sample=pd.DataFrame({"guid": ["a"], "sample_index": [0]}),
        per_anchor=pd.DataFrame({"guid": ["a"], "epoch": [0], "anchor": [0]}),
        record={"n_per_sample_rows": 1, "n_per_anchor_rows": 1},
    )
    collect.write_collection(collection, tmp_path)

    assert not (tmp_path / collect.COHERENCE_FILENAME).exists()
    assert collect.load_collection(tmp_path).spectra == {}


def test_the_analysis_records_a_skip_when_the_tables_carry_no_spectra(tmp_path) -> None:
    """A run collected before this analysis existed is a recorded skip naming the re-collection --
    and it still writes its six figures, because the figure manifest binds every emitted PDF by
    name and a run that silently emitted none would fail that binding instead of reporting this.
    """
    import types

    import pandas as pd

    from teb_vae.lag_attn_rws.eval.analyses import coherence as analysis
    from teb_vae.lag_attn_rws.eval.figures_seam import configure_figure_style

    configure_figure_style()
    collection = types.SimpleNamespace(
        per_sample=pd.DataFrame({"guid": ["a"], "sample_index": [0]}),
        vectors={},
        spectra={},
        record={},
        results={},
    )
    context = types.SimpleNamespace(collection=collection, config={}, task=None, loader=None)

    result = analysis.run_coherence_analysis(
        context, eval_config={"bootstrap_resamples": 10, "seed": 0}, output_dir=tmp_path
    )

    # The protocol's keys, with n_samples None rather than 0 -- a zero would enter the coverage
    # block as a disagreement with every analysis that did score a population.
    assert result["n_samples"] is None
    assert result["composition"] == {}
    assert result["plan"]["capped"] is False
    assert result["skipped"] is True
    assert "re-collect" in result["reason"]

    directory = tmp_path / analysis.ANALYSIS_DIRNAME
    rendered = sorted(path.name for path in directory.glob("*.pdf"))
    assert len(rendered) == 6
    assert figure_filename(analysis.LEAD_TIME_FIGURE) in rendered

    # And **no** CSVs. A placeholder would be a bare newline, which `pd.read_csv` refuses -- so it
    # would take down `cross_subgroup`, which reads this analysis's per-recording table off disk
    # and is written to handle an absent source by recording it.
    assert list(directory.glob("*.csv")) == []
    assert result["files"] == [name for name in result["files"] if name.endswith(".pdf")]


def test_cross_subgroup_survives_a_skipped_coherence_run(tmp_path) -> None:
    """The failure the empty-CSV placeholder would have caused, pinned from the consumer's side.

    ``cross_subgroup`` reads finished per-recording CSVs off disk and records a missing source
    rather than raising -- which is what keeps ``--only cross_subgroup`` working against a partial
    directory. That guard tests for the file's *absence*; a zero-byte file passes it and then
    raises inside ``read_csv``.
    """
    from teb_vae.lag_attn_rws.eval.analyses import coherence as analysis
    from teb_vae.lag_attn_rws.eval.analyses import cross_subgroup

    sources = [
        source
        for source in cross_subgroup.METRIC_SOURCES
        if source.analysis == analysis.ANALYSIS_DIRNAME
    ]
    assert sources, "cross_subgroup should read at least one coherence metric"
    (tmp_path / analysis.ANALYSIS_DIRNAME).mkdir()

    frames, missing = cross_subgroup.load_metric_frames(tmp_path, sources)

    assert frames == {}
    assert len(missing) == len(sources)
    assert all("was not written" in record["reason"] for record in missing)


def test_pooling_over_lead_time_averages_ratios_rather_than_summing_spectra() -> None:
    r"""A $\tau$-pooled number is the mean over $\tau$ of the per-$\tau$ ratio. Summing $S_{xy}$
    across lead times instead would let a $\tau$-dependent phase cancel, so a forecast whose timing
    error grows with lead time would report as one with no timing error at all."""
    values = np.array([[0.2, 0.4], [0.6, 0.8], [np.nan, 1.0]])

    pooled = spectra.mean_over_horizon(values)

    assert pooled.tolist() == pytest.approx([0.4, (0.4 + 0.8 + 1.0) / 3.0])
