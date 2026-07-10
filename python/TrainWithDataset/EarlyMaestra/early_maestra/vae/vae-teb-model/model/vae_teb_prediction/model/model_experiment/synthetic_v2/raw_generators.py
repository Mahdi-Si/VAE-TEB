r"""Raw 4 Hz FHR/UP signal generators for ``synthetic_v2`` (Sprint 1).

This module turns a solved coupling $B$ (from the ported inverter in
:mod:`analytic_te`) into a pair of raw $4\,\mathrm{Hz}$ waveforms — one FHR, one
UP — that carry a single, analytically-known block transfer entropy from the UP
contraction band to the FHR deceleration band. It implements the composition
model (§5), the coupled linear-Gaussian latent pair on the decimated grid (§6),
the band-limited upsample and strictly-positive amplitude-modulation rendering
(§7), and the AM-separation analytic pre-check (S1-T04), all of
``SYNTHETIC_V2_RAW_TE_PIPELINE_EXPLAINED.md``.

Two time grids (§3) are used throughout and must not be confused:

* **decimated / latent grid** — $f_s^{\mathrm{dec}} = f_s / 16 = 0.25\,\mathrm{Hz}$,
  $T' = N_{\mathrm{raw}}/16 = 330$ steps ($4\,\mathrm s$/step). The coupled latent
  pair $(c_k, d_k)$ lives here.
* **raw grid** — $f_s = 4\,\mathrm{Hz}$, $N_{\mathrm{raw}} = 5280$ samples
  ($22\,\mathrm{min}$). The rendered waveforms live here.

The coupled latent pair reuses :func:`analytic_te._simulate_state_space_gaussian`
verbatim: using the *same* simulator that the inverter / TE-label
(:func:`analytic_te.B_y_for_mean_te_block_state_space`) was built on guarantees
the generated data matches the data-generating process the injected-TE label
$\mathrm{TE}_{\mathrm{inj}}$ was computed for.

Design/build references: ``SYNTHETIC_V2_RAW_TE_PIPELINE_EXPLAINED.md`` §3–§7 and
``SYNTHETIC_V2_SPEC_AND_SPRINTS.md`` Sprint 1.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .analytic_te import _simulate_state_space_gaussian

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Scattering decimation factor (== ``scattering.T``); maps $N_{\mathrm{raw}}$ raw
#: samples to $N_{\mathrm{raw}} / 16$ decimated steps.
DECIMATION: int = 16

#: Independent FHR-variability (FHRV) dressing bands in Hz (§4). VLF
#: ($0.003$–$0.03\,\mathrm{Hz}$) is deliberately **excluded** — it is reserved for
#: the coupled deceleration band, so FHRV dressing never pollutes the coupling.
FHRV_BANDS: Dict[str, Tuple[float, float]] = {
    "LF": (0.03, 0.15),
    "MF": (0.15, 0.5),
    "HF": (0.5, 1.0),
}

#: Independent UP slow-drift band in Hz (§4): $>333\,\mathrm s$ period.
UP_DRIFT_BAND: Tuple[float, float] = (0.0008, 0.003)

#: FHR baseline-wander band in Hz. Placed **below** the coupled deceleration band
#: ($0.003$–$0.03\,\mathrm{Hz}$) so the (independent) wander cannot leak into the
#: coupled channel (§5.1, and the technical-review AM caveat).
FHR_WANDER_BAND: Tuple[float, float] = (0.0008, 0.003)

#: Multiple of the latent standard deviation used as the stable "max" surrogate for
#: the AM envelope scale (§7; technical review). Using $k_{\mathrm{ref}}\,\sigma$
#: instead of a per-sample $\max|c|$ keeps the modulation depth constant across
#: samples. With ``am_offset_ratio = 4`` the resulting RMS modulation depth is
#: $\sigma / (k_{\mathrm{ref}}\,\sigma \cdot \mathrm{ratio}) = 1 / (k_{\mathrm{ref}}\,\mathrm{ratio})$.
AM_MAX_SIGMA_MULT: float = 4.0

#: Default hard floor for the AM envelope, as a fraction of its DC level $a_0$. The
#: positivity is already guaranteed by ``am_offset_ratio > 1``; this only guards the
#: astronomically-rare Gaussian tail so ``A > 0`` holds *everywhere* (S1-T03b).
AM_FLOOR_FRAC: float = 0.05

#: Default acceleration bump width in seconds (raised-cosine half-support).
ACCEL_WIDTH_S: float = 20.0

# Named RNG substreams (indices into the spawned SeedSequence children). Keeping
# each component on its own stream makes a B=0 cell a *true* zero-coupling null and
# keeps a change in one component from perturbing the others.
_STREAM_LATENT = 0
_STREAM_DC = 1
_STREAM_FHRV = 2
_STREAM_ACCEL = 3
_STREAM_FHR_WANDER = 4
_STREAM_FHR_NOISE = 5
_STREAM_UP_DRIFT = 6
_STREAM_UP_NOISE = 7
_STREAM_CARRIER = 8
_N_STREAMS = 9


# ---------------------------------------------------------------------------
# AR(2) analytics (S1-T01)
# ---------------------------------------------------------------------------


def ar2_coeffs(r: float, w: float) -> Tuple[float, float]:
    r"""Return the AR(2) coefficients $(\phi_1, \phi_2)$ for poles at $r e^{\pm i\omega}$.

    $$\phi_1 = 2 r \cos\omega, \qquad \phi_2 = -r^2.$$

    Args:
        r: Pole radius (spectral radius) $r \in [0, 1)$.
        w: Pole angle $\omega$ in rad/step.

    Returns:
        The pair ``(phi1, phi2)``.
    """
    return 2.0 * r * math.cos(w), -(r ** 2)


def ar2_stationary_std(r: float, w: float, sigma2_eta: float) -> float:
    r"""Closed-form stationary standard deviation of the AR(2) source $c_k$.

    For $c_k = \phi_1 c_{k-1} + \phi_2 c_{k-2} + \eta_k$ with
    $\eta_k \sim \mathcal N(0, \sigma_\eta^2)$, the stationary variance is

    $$
    \operatorname{Var}(c) =
      \frac{\sigma_\eta^2\,(1 - \phi_2)}
           {(1 + \phi_2)\,\bigl[(1 - \phi_2)^2 - \phi_1^2\bigr]} .
    $$

    Using this analytic scale (rather than a noisy per-sample $\max|c|$) keeps the
    AM modulation depth constant across samples (§7, technical review).

    Args:
        r: Pole radius.
        w: Pole angle in rad/step.
        sigma2_eta: Source innovation variance $\sigma_\eta^2 > 0$.

    Returns:
        The stationary standard deviation $\sigma_c$.
    """
    phi1, phi2 = ar2_coeffs(r, w)
    var = sigma2_eta * (1.0 - phi2) / ((1.0 + phi2) * ((1.0 - phi2) ** 2 - phi1 ** 2))
    if var <= 0.0:
        raise ValueError(
            f"ar2_stationary_std: non-positive variance ({var:.4g}); the AR(2) "
            f"(r={r}, w={w}) is non-stationary."
        )
    return math.sqrt(var)


def ar2_lag1_autocorr(r: float, w: float) -> float:
    r"""Lag-1 autocorrelation of the AR(2) source, $\rho_1 = \phi_1 / (1 - \phi_2)$.

    Args:
        r: Pole radius.
        w: Pole angle in rad/step.

    Returns:
        The theoretical lag-1 autocorrelation $\rho_1$.
    """
    phi1, phi2 = ar2_coeffs(r, w)
    return phi1 / (1.0 - phi2)


def simulate_latent_pair(
    n: int,
    T_tot: int,
    *,
    r: float,
    w: float,
    target_ar: float,
    B: float,
    D: int,
    sigma2_y: float,
    sigma2_eta: float,
    burn_in: int = 500,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Simulate the single-pathway coupled latent pair $(c_k, d_k)$ on the decimated grid.

    Thin wrapper over :func:`analytic_te._simulate_state_space_gaussian` with
    $M = 1$: the source is the AR(2) contraction-strength latent

    $$c_k = 2 r \cos(\omega)\,c_{k-1} - r^2 c_{k-2} + \eta_k, \quad
      \eta_k \sim \mathcal N(0, \sigma_\eta^2),$$

    and the target is the AR(1) deceleration-depth latent with delayed coupling

    $$d_k = A_y\,d_{k-1} + B\,c_{k-D} + \varepsilon_k, \quad
      \varepsilon_k \sim \mathcal N(0, \sigma_y^2).$$

    The $B c_{k-D}$ term is the only source of transfer entropy; $B = 0$ is a null
    cell. Reusing the analytic simulator (rather than reimplementing the recurrence)
    guarantees the generated latent matches the DGP the injected-TE label was solved
    for.

    Args:
        n: Number of independent samples.
        T_tot: Kept (post-burn-in) sequence length on the decimated grid (330 in v2).
        r: Source pole radius.
        w: Source pole angle in rad/step.
        target_ar: Target self-coefficient $A_y \in [0, 1)$.
        B: Coupling strength $B$ (the solved ``B_y_scalar``; ``0.0`` for a null cell).
        D: Source→target delay $D \ge 1$ in decimated steps.
        sigma2_y: Target innovation variance $\sigma_y^2 > 0$.
        sigma2_eta: Source innovation variance $\sigma_\eta^2 > 0$.
        burn_in: Warm-up steps discarded so the kept window is stationary.
        seed: NumPy generator seed for the latent simulation.

    Returns:
        A tuple ``(c, d)`` of ``np.ndarray`` of shape ``(n, T_tot)`` — the source
        contraction-strength latent and the coupled target deceleration-depth latent.
    """
    if D < 1:
        raise ValueError(f"simulate_latent_pair: D must be >= 1, got {D}.")
    S, Y = _simulate_state_space_gaussian(
        n=n,
        T=T_tot,
        oscillators=[(r, w)],
        target_ar=target_ar,
        delays=[int(D)],
        B_y=[float(B)],
        sigma2_y=sigma2_y,
        sigma2_eta=sigma2_eta,
        burn_in=burn_in,
        seed=seed,
    )
    return S[:, :, 0], Y[:, :, 0]


def true_lag_trajectory(n: int, T: int, D: int) -> np.ndarray:
    r"""Per-sample, per-step true source→target delay of shape $(n, T)$ (fixed mode).

    Ported from ``synthetic/generators.py::_true_lag_trajectory`` for the fixed-lag
    case: a flat line at $D$ for every sample and step. This is the ground-truth lag
    that lag attention must recover; the informative lag *band* is
    $\mathcal L^\star = \{\max(0, D - H), \dots, D - 1\}$ (§6.2).

    Args:
        n: Number of samples.
        T: Sequence length.
        D: Fixed delay $D$.

    Returns:
        An ``int16`` ``np.ndarray`` of shape ``(n, T)`` filled with $D$.
    """
    return np.full((n, T), int(D), dtype=np.int16)


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------


def _spawn_streams(seed: int, k: int = _N_STREAMS) -> Sequence[np.random.SeedSequence]:
    r"""Spawn ``k`` independent, reproducible :class:`numpy.random.SeedSequence` children.

    Args:
        seed: Umbrella seed.
        k: Number of independent substreams.

    Returns:
        A list of ``k`` child ``SeedSequence`` objects.
    """
    return np.random.SeedSequence(seed).spawn(k)


def _int_seed(ss: np.random.SeedSequence) -> int:
    r"""Derive a deterministic non-negative ``int`` seed from a ``SeedSequence``.

    Used to feed :func:`analytic_te._simulate_state_space_gaussian`, which takes an
    ``int`` seed rather than a generator.

    Args:
        ss: A ``SeedSequence`` child.

    Returns:
        A non-negative Python ``int``.
    """
    return int(ss.generate_state(1, dtype=np.uint32)[0])


# ---------------------------------------------------------------------------
# Independent dressing bands and DC (S1-T02)
# ---------------------------------------------------------------------------


def draw_dc(
    n: int,
    rng: np.random.Generator,
    *,
    fhr_range: Tuple[float, float] = (110.0, 160.0),
    up_range: Tuple[float, float] = (5.0, 25.0),
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Draw per-sample physiological DC levels $\mu_{\mathrm{FHR}}, \mu_{\mathrm{UP}}$ (§5.1).

    The DC terms populate scattering channel 0 (which is *not* log-transformed) with
    physiological values; they are drawn independently of the coupled pathway, so
    they add no source→target dependency.

    Args:
        n: Number of samples.
        rng: NumPy generator (its own substream).
        fhr_range: Inclusive FHR baseline range $[110, 160]$ bpm (neutral 140).
        up_range: Inclusive UP resting-tone range $[5, 25]$ mmHg (IUPC).

    Returns:
        A tuple ``(mu_fhr, mu_up)`` of ``(n,)`` float arrays.
    """
    mu_fhr = rng.uniform(fhr_range[0], fhr_range[1], size=n)
    mu_up = rng.uniform(up_range[0], up_range[1], size=n)
    return mu_fhr, mu_up


def synth_band(
    n: int,
    N: int,
    fs: float,
    f_lo: float,
    f_hi: float,
    power: float,
    rng: np.random.Generator,
    *,
    notch: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    r"""Random-phase cosine-sum band on the DFT grid with variance $\approx P$ (§5.3).

    Synthesises, per sample, $v[n] = \sum_{m} A_m \cos(2\pi f_m n / f_s + \phi_m)$
    over the DFT bins $f_m = m f_s / N$ falling in $[f_{\mathrm{lo}}, f_{\mathrm{hi}}]$,
    with $A_m = \sqrt{2 P / M_b}$ and $\phi_m \sim U(0, 2\pi)$ drawn independently per
    sample and per bin. Because the bins lie exactly on the DFT grid, the per-sample
    time-averaged power is exactly $\sum_m A_m^2 / 2 = P$ (bin orthogonality), and the
    band is periodic in the window (no spectral leakage). The signal is built as an
    inverse real FFT: $X[m] = (N/2) A_m e^{i\phi_m}$ on the band bins, zero elsewhere.

    An optional ``notch`` excises a sub-interval $[\texttt{notch\_lo}, \texttt{notch\_hi}]$
    of DFT bins before $M_b$ and $A_m$ are computed, so the *surviving* bins are
    re-normalised to still carry the full target power $P$ (the variance guarantee is
    preserved because $M_b = $ ``bins.size`` and $A_m = \sqrt{2P/M_b}$ are evaluated
    **after** the notch). This keeps the independent dressing band out of a protected
    carrier neighbourhood (the coupled pulse-shape channel; §7, §19) without changing its
    total power.

    Args:
        n: Number of samples.
        N: Signal length (raw grid, e.g. 5280).
        fs: Sampling rate in Hz.
        f_lo: Lower band edge in Hz (exclusive of DC).
        f_hi: Upper band edge in Hz (below Nyquist).
        power: Target band power $P$ (variance in signal units$^2$).
        rng: NumPy generator (its own substream).
        notch: Optional ``(notch_lo, notch_hi)`` Hz interval whose DFT bins are removed
            from the band before power normalisation; ``None`` disables the notch. Bins
            with centre frequency $f_m \in [\texttt{notch\_lo}, \texttt{notch\_hi}]$ are
            dropped (inclusive).

    Returns:
        A ``(n, N)`` float array with per-sample variance $\approx P$; zeros if the
        band (after any notch) contains no DFT bin (with the given ``N``) or
        ``power <= 0``.
    """
    df = fs / N
    k_lo = int(math.ceil(f_lo / df))
    k_hi = int(math.floor(f_hi / df))
    k_lo = max(k_lo, 1)                     # never include DC
    k_hi = min(k_hi, N // 2 - 1)            # never include Nyquist
    if k_hi < k_lo or power <= 0.0:
        return np.zeros((n, N), dtype=float)
    bins = np.arange(k_lo, k_hi + 1)
    if notch is not None:
        # Excise the protected carrier neighbourhood BEFORE m_b/amp are computed, so the
        # surviving bins are re-normalised to still carry the full power P (§7.3, §19).
        notch_lo, notch_hi = float(notch[0]), float(notch[1])
        keep = (bins * df < notch_lo) | (bins * df > notch_hi)
        bins = bins[keep]
        if bins.size == 0:
            return np.zeros((n, N), dtype=float)
    m_b = bins.size
    amp = math.sqrt(2.0 * power / m_b)
    phases = rng.uniform(0.0, 2.0 * math.pi, size=(n, m_b))
    spec = np.zeros((n, N // 2 + 1), dtype=complex)
    spec[:, bins] = (N / 2.0) * amp * np.exp(1j * phases)
    return np.fft.irfft(spec, n=N, axis=-1)


def synth_fhrv(
    n: int,
    N: int,
    fs: float,
    band_powers: Dict[str, float],
    rng: np.random.Generator,
    *,
    notch: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    r"""Independent FHR-variability dressing: the sum of the LF/MF/HF bands (§5.2–§5.3).

    The optional ``notch`` is forwarded to every band's :func:`synth_band` call. At the
    locked carrier it intersects only the LF band ($0.03$–$0.15\,\mathrm{Hz}$); MF/HF drop
    no bins and are unaffected. This keeps the independent FHRV dressing out of the coupled
    deceleration pulse-shape channel (§7.3, §19) without changing each band's total power.

    Args:
        n: Number of samples.
        N: Signal length (raw grid).
        fs: Sampling rate in Hz.
        band_powers: Target power per band, e.g. ``{"LF": 11.25, "MF": 6.25, "HF": 2.5}``
            in bpm$^2$. VLF is intentionally absent (reserved for the coupled band).
        rng: NumPy generator (its own substream).
        notch: Optional ``(notch_lo, notch_hi)`` Hz interval to excise from every band
            (the protected carrier neighbourhood); ``None`` disables the notch.

    Returns:
        A ``(n, N)`` float array — the summed LF+MF+HF variability.
    """
    out = np.zeros((n, N), dtype=float)
    for name, (f_lo, f_hi) in FHRV_BANDS.items():
        power = float(band_powers.get(name, 0.0))
        out += synth_band(n, N, fs, f_lo, f_hi, power, rng, notch=notch)
    return out


def synth_baseline_wander(
    n: int,
    N: int,
    fs: float,
    std: float,
    rng: np.random.Generator,
    band: Tuple[float, float] = FHR_WANDER_BAND,
) -> np.ndarray:
    r"""Independent slow baseline wander in a sub-decel band (§5.1).

    Placed below $0.003\,\mathrm{Hz}$ so the (independent) wander cannot leak into the
    coupled deceleration band. Power is $\text{std}^2$.

    Args:
        n: Number of samples.
        N: Signal length.
        fs: Sampling rate in Hz.
        std: Target wander standard deviation.
        rng: NumPy generator (its own substream).
        band: Wander band in Hz.

    Returns:
        A ``(n, N)`` float array.
    """
    return synth_band(n, N, fs, band[0], band[1], std * std, rng)


def synth_up_drift(
    n: int,
    N: int,
    fs: float,
    std: float,
    rng: np.random.Generator,
    band: Tuple[float, float] = UP_DRIFT_BAND,
) -> np.ndarray:
    r"""Independent UP slow drift / tone-change dressing (§4–§5). Power is $\text{std}^2$.

    Args:
        n: Number of samples.
        N: Signal length.
        fs: Sampling rate in Hz.
        std: Target drift standard deviation.
        rng: NumPy generator (its own substream).
        band: Drift band in Hz.

    Returns:
        A ``(n, N)`` float array.
    """
    return synth_band(n, N, fs, band[0], band[1], std * std, rng)


def synth_accelerations(
    n: int,
    N: int,
    fs: float,
    *,
    amp_bpm: Tuple[float, float],
    rate_per_min: float,
    rng: np.random.Generator,
    width_s: float = ACCEL_WIDTH_S,
) -> np.ndarray:
    r"""Independent FHR accelerations: raised-cosine upward bumps (§5.2).

    Places approximately ``rate_per_min`` events per minute at random times, each a
    raised-cosine bump of half-support ``width_s`` seconds and amplitude drawn from
    ``amp_bpm``. Independent of the coupled pathway. A ``rate_per_min`` that rounds to
    zero events yields an all-zero output (accelerations can be disabled for a control).

    Args:
        n: Number of samples.
        N: Signal length (raw grid).
        fs: Sampling rate in Hz.
        amp_bpm: Inclusive amplitude range $[10, 25]$ bpm.
        rate_per_min: Mean acceleration rate per minute.
        rng: NumPy generator (its own substream).
        width_s: Raised-cosine half-support in seconds.

    Returns:
        A ``(n, N)`` non-negative float array of accelerations (zeros if no events).
    """
    duration_min = N / fs / 60.0
    n_events = int(round(rate_per_min * duration_min))
    out = np.zeros((n, N), dtype=float)
    if n_events <= 0:
        return out
    half = max(1, int(round(width_s * fs)))
    # Raised-cosine template of length 2*half+1, unit peak, zero at the edges.
    t = np.arange(-half, half + 1)
    template = 0.5 * (1.0 + np.cos(np.pi * t / half))
    for i in range(n):
        centres = rng.integers(half, N - half, size=n_events)
        amps = rng.uniform(amp_bpm[0], amp_bpm[1], size=n_events)
        for c, a in zip(centres, amps):
            out[i, c - half : c + half + 1] += a * template
    return out


def white_noise(n: int, N: int, std: float, rng: np.random.Generator) -> np.ndarray:
    r"""Independent broadband (measurement / toco) noise, $\mathcal N(0, \text{std}^2)$.

    Args:
        n: Number of samples.
        N: Signal length.
        std: Noise standard deviation.
        rng: NumPy generator (its own substream).

    Returns:
        A ``(n, N)`` float array (zeros if ``std <= 0``).
    """
    if std <= 0.0:
        return np.zeros((n, N), dtype=float)
    return std * rng.standard_normal((n, N))


# ---------------------------------------------------------------------------
# Band-limited upsample (S1-T03a)
# ---------------------------------------------------------------------------


def upsample_bandlimited(
    x: np.ndarray,
    up: int = DECIMATION,
    *,
    fs_dec: float,
    lowpass_hz: Optional[float] = None,
    detrend: bool = True,
) -> np.ndarray:
    r"""Anti-aliased band-limited upsample of a decimated-grid signal by an integer factor.

    Uses frequency-domain zero-padding (ideal sinc interpolation, equivalent to
    ``scipy.signal.resample`` but numpy-only). Frequency-domain zero-padding creates
    **no aliasing images** by construction. Two care points are handled:

    * **Nyquist bin.** For an even input length $L$ the real Nyquist coefficient
      ``X[L//2]`` is zeroed before re-embedding (it would otherwise inject a
      $\cos(\pi n)$ artifact). Harmless here because the envelope's energy sits far
      below the decimated Nyquist.
    * **Anti-alias cutoff.** If ``lowpass_hz`` is given, every decimated-grid bin at or
      above that frequency is zeroed (DC is always kept), so "no significant spectral
      energy at/above ``lowpass_hz``" is a construction guarantee **when
      ``detrend=False``** (the production path). With ``detrend=True`` the re-added
      endpoint line has a broadband spectrum that leaks a little above the cutoff, so
      the strict band-limit guarantee holds only for ``detrend=False``.

    The amplitude scale is ``* up`` ($= N_{\mathrm{out}} / L$) so DC and sinusoid
    amplitudes are preserved. Endpoints are optionally detrended (a line joining the
    first and last sample is subtracted before the FFT and re-added on the fine grid)
    to suppress periodic-wrap ringing; the downstream 15-step/end trim discards the
    outer region regardless. The re-added line is placed on the same input-coordinate
    grid it was removed on (input sample $m$ maps to output index $m\,\mathrm{up}$).

    Args:
        x: Input ``(n, L)`` (or ``(L,)``) real array on the decimated grid.
        up: Integer upsampling factor (16 in v2).
        fs_dec: Decimated-grid sampling rate in Hz (used to convert ``lowpass_hz``).
        lowpass_hz: Optional anti-alias cutoff in Hz; bins at/above it are zeroed.
        detrend: Subtract/re-add an endpoint line to reduce wrap ringing.

    Returns:
        A ``(n, L*up)`` (or ``(L*up,)``) real array on the raw grid.
    """
    x = np.asarray(x, dtype=float)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[None, :]
    n, L = x.shape
    N_out = L * up

    ramp = None
    work = x
    if detrend and L > 1:
        x0 = x[:, :1]
        x1 = x[:, -1:]
        lin = (np.arange(L) / (L - 1))[None, :]
        work = x - (x0 + (x1 - x0) * lin)
        # Re-add on the SAME input-coordinate grid: input sample m sits at output
        # index m*up, so the line reaches its endpoint value at output index (L-1)*up
        # (not N_out-1). This makes detrend cancel the removed trend at the samples.
        lin_out = (np.arange(N_out) / ((L - 1) * up))[None, :]
        ramp = x0 + (x1 - x0) * lin_out           # (n, N_out)

    spec = np.fft.rfft(work, axis=-1)             # (n, L//2 + 1)
    if L % 2 == 0:
        spec[:, -1] = 0.0                         # kill input Nyquist
    if lowpass_hz is not None:
        # Zero bins at/above the cutoff but always keep DC (bin 0): a cutoff below the
        # first AC bin would otherwise wipe the whole spectrum to a constant.
        k_cut = max(1, int(math.floor(lowpass_hz / fs_dec * L)))
        if k_cut < spec.shape[-1]:
            spec[:, k_cut:] = 0.0
    spec_pad = np.zeros((n, N_out // 2 + 1), dtype=complex)
    spec_pad[:, : spec.shape[-1]] = spec
    y = np.fft.irfft(spec_pad, n=N_out, axis=-1) * up
    if ramp is not None:
        y = y + ramp
    return y[0] if squeeze else y


# ---------------------------------------------------------------------------
# Positive AM envelope, carrier, raw composition (S1-T03b)
# ---------------------------------------------------------------------------


def am_envelope(
    x_tilde: np.ndarray,
    *,
    sigma_ref: float,
    am_offset_ratio: float,
    amp_peak: float,
    a_min: Optional[float] = None,
) -> np.ndarray:
    r"""Strictly-positive amplitude envelope $A = a_0 + a_1 \tilde x$ (§7).

    Sets the DC level from the positivity ratio and the amplitude scale so the
    envelope peak (at $\tilde x = +k_{\mathrm{ref}}\sigma$, i.e. ``sigma_ref``) equals
    ``amp_peak``:

    $$
    a_{\mathrm{scale}} = \frac{\texttt{amp\_peak}}{\texttt{ratio} + 1}, \quad
    a_0 = \texttt{ratio}\cdot a_{\mathrm{scale}}, \quad
    a_1 = a_{\mathrm{scale}} / \texttt{sigma\_ref}.
    $$

    With ``am_offset_ratio > 1`` the nominal trough
    $a_0 - a_1\,\texttt{sigma\_ref} = a_{\mathrm{scale}}(\texttt{ratio} - 1) > 0$; a
    hard floor ``a_min`` clamps the rare Gaussian tail so $A > 0$ everywhere.

    Args:
        x_tilde: Upsampled latent envelope ``(n, N)`` (mean ~0).
        sigma_ref: Stable envelope scale $k_{\mathrm{ref}}\sigma$ (from
            :data:`AM_MAX_SIGMA_MULT` times the latent std).
        am_offset_ratio: Positivity ratio $> 1$ (``raw.am_offset_ratio``).
        amp_peak: Envelope value at $\tilde x = \texttt{sigma\_ref}$ (physiological
            peak amplitude, e.g. contraction top in mmHg / decel depth in bpm).
        a_min: Hard positive floor; defaults to :data:`AM_FLOOR_FRAC` $\times a_0$.

    Returns:
        A strictly-positive envelope array the shape of ``x_tilde``.
    """
    if am_offset_ratio <= 1.0:
        raise ValueError(
            f"am_envelope: am_offset_ratio must be > 1 for positivity, got "
            f"{am_offset_ratio}."
        )
    if sigma_ref <= 0.0:
        raise ValueError(f"am_envelope: sigma_ref must be > 0, got {sigma_ref}.")
    a_scale = amp_peak / (am_offset_ratio + 1.0)
    a0 = am_offset_ratio * a_scale
    a1 = a_scale / sigma_ref
    if a_min is None:
        a_min = AM_FLOOR_FRAC * a0
    A = a0 + a1 * x_tilde
    return np.maximum(A, a_min)


def make_carrier(
    N: int, fs: float, f_pulse: float, phase: Union[float, np.ndarray] = 0.0
) -> np.ndarray:
    r"""Fixed unit-amplitude narrowband carrier $g[n] = \cos(2\pi f_{\mathrm{pulse}} n / f_s + \varphi)$.

    A pure cosine is maximally narrowband, which best satisfies the AM approximation
    $|x \star \psi_\lambda| \approx \kappa A$ (§7.1). The carrier sign/phase is
    irrelevant to the coupling the model reads: the first-order scattering channel is
    a modulus and is phase-blind (§7.1).

    Args:
        N: Signal length (raw grid).
        fs: Sampling rate in Hz.
        f_pulse: Carrier (pulse-shape) frequency in Hz.
        phase: Carrier phase $\varphi$ in radians. A scalar gives a ``(N,)`` carrier; a
            per-sample ``(n, 1)`` phase array broadcasts to a ``(n, N)`` carrier.

    Returns:
        A ``(N,)`` float array for a scalar phase, or ``(n, N)`` for an ``(n, 1)`` phase.
    """
    t = np.arange(N) / fs
    return np.cos(2.0 * math.pi * f_pulse * t + phase)


def make_pulse_train(
    N: int,
    fs: float,
    rate_hz: float,
    *,
    duty: float = 0.5,
    phase: Union[float, np.ndarray] = 0.0,
) -> np.ndarray:
    r"""Non-negative raised-cosine event train at ``rate_hz`` (``pulse_train`` render, S7-T04).

    Each period $P = f_s / \texttt{rate\_hz}$ samples holds one raised-cosine bump of
    fractional width ``duty`` centred in the period:

    $$
    g[n] = \tfrac12\!\left(1 + \cos\!\frac{\pi\,\delta}{\texttt{duty}/2}\right)\;\text{for}\;
    |\delta| \le \tfrac{\texttt{duty}}{2}, \quad 0 \text{ otherwise},
    $$

    where $\delta$ is the signed distance (in cycles) from the bump centre. The train is a
    periodic pulse train whose fundamental sits at ``rate_hz`` (default the carrier
    ``f_pulse``), so the same fs-correct pulse-shape scattering channel carries the
    coupling as in ``am_carrier``. Unlike the signed cosine carrier this event train is
    one-sided ($g \ge 0$), so a positive envelope renders clinically-realistic upward
    contractions / downward decelerations (the waveform-realistic variant, §7.3).

    Args:
        N: Signal length (raw grid).
        fs: Sampling rate in Hz.
        rate_hz: Event rate in Hz (the pulse-train fundamental).
        duty: Fraction of each period the raised-cosine bump occupies, in $(0, 1]$.
        phase: Phase offset in **cycles** ($[0, 1)$). A scalar gives a ``(N,)`` train; a
            per-sample ``(n, 1)`` array broadcasts to a ``(n, N)`` train.

    Returns:
        A non-negative ``(N,)`` array for a scalar phase, or ``(n, N)`` for an ``(n, 1)``
        phase; peak amplitude $1$ at each bump centre.
    """
    if not (0.0 < duty <= 1.0):
        raise ValueError(f"make_pulse_train: duty must be in (0, 1], got {duty}.")
    if rate_hz <= 0.0:
        raise ValueError(f"make_pulse_train: rate_hz must be > 0, got {rate_hz}.")
    period = fs / rate_hz
    # Position within the current period, in cycles [0, 1), shifted by ``phase``.
    frac = ((np.arange(N) / period) + phase) % 1.0
    half = duty / 2.0
    dist = np.abs(frac - 0.5)  # distance (cycles) from the bump centre at frac=0.5
    bump = 0.5 * (1.0 + np.cos(np.pi * dist / half))
    return np.where(dist <= half, bump, 0.0)


def generate_cell_raw(
    n: int,
    *,
    B: float,
    D: int,
    config: Dict[str, Any],
    benchmark: str = "G1_raw",
    seed: int = 0,
    te_inj: Optional[float] = None,
    render_mode: Optional[str] = None,
) -> Dict[str, Any]:
    r"""Generate ``n`` raw FHR/UP pairs for one cell $(B, D)$ (§5–§7, S1-T03b).

    Composes each raw signal from a physiological DC, the AM-rendered coupled pulse,
    and mutually-independent dressing bands (§5):

    $$
    x_{\mathrm{UP}} = \mu_{\mathrm{UP}} + A_u\,g + u_{\mathrm{drift}} + \varepsilon_{\mathrm{UP}}, \qquad
    x_{\mathrm{FHR}} = \mu_{\mathrm{FHR}} - A_y\,g + \mathrm{FHRV} + a + u_{\mathrm{wander}} + \varepsilon_{\mathrm{FHR}},
    $$

    where $A_u = a_0 + a_1 \tilde c$ and $A_y = b_0 + b_1 \tilde d$ are the
    strictly-positive amplitude envelopes carrying the coupled latents (§7) and $g$ is
    the pulse-shape carrier. The coupled term $B c_{k-D}$ inside $d_k$ is the only
    source→target dependency; everything else is independent dressing.

    In ``render_mode='direct'`` (§7.4) there is **no carrier** ($g \equiv 1$) and **no AM
    modulation**: the coupled bands are the upsampled latents scaled straight to
    physiological peak amplitude — $u_{\mathrm c} = \gamma_u\,(\tilde c)_+$ and
    $y_{\mathrm d} = \gamma_y\,(\tilde d)_+$ under the default one-sided rectifier
    $(\cdot)_+$ (clinically upward contractions / downward decelerations), or the bipolar
    $u_{\mathrm c} = \gamma_u\,\tilde c$, $y_{\mathrm d} = \gamma_y\,\tilde d$ when
    ``raw.direct.one_sided`` is false. This yields realistic raw FHR/UP-like traces but
    forfeits the carrier's guarantee of a known coupled scattering channel.

    Args:
        n: Number of samples to generate.
        B: Coupling strength (the solved ``B_y_scalar``; ``0.0`` for a null cell).
        D: Fixed source→target lag $D \ge 1$ in decimated steps.
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        seed: Umbrella seed for this cell's generation (spawns independent substreams).
        te_inj: Optional injected-TE label to stamp into the returned ``meta``.
        render_mode: Override for ``raw.render_mode`` -- ``am_carrier`` (default, signed
            narrowband cosine carrier), ``pulse_train`` (one-sided raised-cosine event
            train; the waveform-realistic variant, §7.3 / S7-T04), or ``direct`` (no
            carrier and no AM modulation: the coupled latents are rendered *directly* as
            the low-frequency contraction / deceleration waveform, §7.4). ``direct`` trades
            the known-scattering-channel TE guarantee (the carrier's raison d'être) for
            clinically realistic raw FHR/UP-like traces; its ``one_sided`` shape knob is
            read from ``raw.direct``.

    Returns:
        A dict with keys ``fhr_raw`` ``(n, N_raw)``, ``up_raw`` ``(n, N_raw)``,
        ``true_lag_tt`` ``(n, T_tot)`` int16, ``latents`` (the coupled-pathway
        intermediates ``c``, ``d``, ``c_tilde``, ``d_tilde``, ``A_u``, ``A_y``,
        ``carrier_u``, ``carrier_y``, ``u_c``, ``y_d``; in ``direct`` mode ``A_u==u_c``,
        ``A_y==y_d`` and the carriers are flat ones), and ``meta`` (``D``, ``B``,
        ``te_inj``, ``render_mode``, ``direct_one_sided``, ``T_tot``, ``n_raw``, ``fs``,
        ``f_pulse``).
    """
    bench = config["benchmarks"][benchmark]
    data = bench["data"]
    raw = bench["raw"]
    # ``is None`` (not truthiness): a falsy override such as '' must be validated, not
    # silently replaced by the config default.
    render_mode = raw.get("render_mode", "am_carrier") if render_mode is None else render_mode
    if render_mode not in ("am_carrier", "pulse_train", "direct"):
        raise ValueError(
            f"generate_cell_raw: unknown render_mode {render_mode!r} "
            "(expected 'am_carrier', 'pulse_train', or 'direct')."
        )
    # ``direct`` (§7.4) renders the coupled latents straight onto the raw grid with no
    # carrier and no amplitude modulation, so several carrier-specific steps below are
    # bypassed (the f_pulse envelope low-pass, the FHRV carrier notch, the AM envelope,
    # and the carrier itself).
    is_direct = render_mode == "direct"
    # Set inside the ``direct`` branch below; stays ``None`` for the carrier renders so it
    # is stamped into ``meta`` only when meaningful.
    one_sided: Optional[bool] = None

    fs = float(raw["fs"])
    N = int(raw["n_raw"])
    if N % DECIMATION != 0:
        raise ValueError(
            f"generate_cell_raw: raw.n_raw ({N}) must be an integer multiple of the "
            f"decimation factor ({DECIMATION}); the coupled term is rendered on the "
            "decimated grid and upsampled, and would otherwise not align with the "
            "full-length dressing bands."
        )
    fs_dec = fs / DECIMATION
    T_tot = N // DECIMATION
    f_pulse = float(raw["f_pulse"])
    ratio = float(raw["am_offset_ratio"])
    # FHRV notch (§7.3, §19): excise a one-Q-step carrier neighbourhood
    # $[f_{\mathrm{pulse}} 2^{-1/Q},\, f_{\mathrm{pulse}} 2^{+1/Q}]$ from the independent
    # FHRV dressing so its LF band does not pollute the coupled decel pulse-shape channel
    # (the locked f_pulse=0.06 Hz sits inside FHRV LF 0.03-0.15). Enabled by default; the
    # notch width matches the coupled-channel selection tolerance in
    # ``scattering_adapter.coupled_channel_indices``. ``scattering`` is only bound here.
    # In ``direct`` mode there is no coupled carrier channel to protect, so the notch is
    # meaningless -- keep the full (more realistic) FHRV dressing regardless of the flag.
    fhrv_notch: Optional[Tuple[float, float]] = None
    if bool(raw.get("fhrv_notch_enabled", True)) and not is_direct:
        Q = int(config["benchmarks"][benchmark]["scattering"]["Q"])
        fhrv_notch = (f_pulse * 2.0 ** (-1.0 / Q), f_pulse * 2.0 ** (1.0 / Q))
    r, w = float(data["oscillators"][0][0]), float(data["oscillators"][0][1])
    target_ar = float(data["target_ar"])
    sigma2_y = float(data["sigma2_y"])
    sigma2_eta = float(data["sigma2_eta"])

    streams = _spawn_streams(seed)
    rng_dc = np.random.default_rng(streams[_STREAM_DC])
    rng_fhrv = np.random.default_rng(streams[_STREAM_FHRV])
    rng_accel = np.random.default_rng(streams[_STREAM_ACCEL])
    rng_fhr_wander = np.random.default_rng(streams[_STREAM_FHR_WANDER])
    rng_fhr_noise = np.random.default_rng(streams[_STREAM_FHR_NOISE])
    rng_up_drift = np.random.default_rng(streams[_STREAM_UP_DRIFT])
    rng_up_noise = np.random.default_rng(streams[_STREAM_UP_NOISE])
    rng_carrier = np.random.default_rng(streams[_STREAM_CARRIER])

    # --- coupled latent pair + AM rendering ---
    c, d = simulate_latent_pair(
        n,
        T_tot,
        r=r,
        w=w,
        target_ar=target_ar,
        B=B,
        D=D,
        sigma2_y=sigma2_y,
        sigma2_eta=sigma2_eta,
        seed=_int_seed(streams[_STREAM_LATENT]),
    )
    # detrend=False keeps the envelope *strictly* band-limited below the carrier: a
    # re-added endpoint ramp would leak broadband content near f_pulse and slightly
    # contaminate the coupled carrier channel. Edge ringing is discarded by the
    # downstream 15-step/end trim. In ``direct`` mode there is no carrier, so the
    # f_pulse anti-alias cutoff is dropped and the full (already-slow) latent band is
    # kept as the rendered contraction / deceleration waveform.
    envelope_lowpass = None if is_direct else f_pulse
    c_tilde = upsample_bandlimited(c, DECIMATION, fs_dec=fs_dec, lowpass_hz=envelope_lowpass, detrend=False)
    d_tilde = upsample_bandlimited(d, DECIMATION, fs_dec=fs_dec, lowpass_hz=envelope_lowpass, detrend=False)

    # Scale each envelope by the *pooled* (over the batch) std of its own upsampled
    # latent, so UP and FHR are calibrated symmetrically (pooled std is stable across
    # samples, unlike a per-sample max). Fall back to the analytic AR(2) std only if a
    # latent is degenerately flat.
    fallback = ar2_stationary_std(r, w, sigma2_eta)
    sigma_c_up = float(c_tilde.std())
    sigma_d_up = float(d_tilde.std())
    ref_c = AM_MAX_SIGMA_MULT * (sigma_c_up if sigma_c_up > 0.0 else fallback)
    ref_d = AM_MAX_SIGMA_MULT * (sigma_d_up if sigma_d_up > 0.0 else fallback)

    if is_direct:
        # --- direct rendering: no carrier, no AM modulation (§7.4) --------------------
        # The coupled latents ARE the low-frequency contraction / deceleration waveform.
        # Each is scaled so the physiological peak amplitude (``contraction_mmHg[1]`` above
        # tone; ``decel_depth_bpm[1]`` below baseline) is reached at the +k_ref sigma
        # extreme -- the full peak slope, not the AM modulation slope a1 (there is no a0
        # positivity margin to reserve here). ``one_sided`` (default) applies a positive
        # rectifier so UP renders one-sided upward contractions and FHR one-sided downward
        # decelerations (clinically realistic); ``one_sided: false`` keeps the zero-mean
        # bipolar latent, which preserves the exact *linear* c->d coupling but lets UP swing
        # below its resting tone. Either way the carrier's known-scattering-channel TE
        # guarantee is gone: the coupled information now spreads across the low-frequency
        # scattering channels rather than concentrating in the f_pulse pulse-shape channel.
        direct_cfg = raw.get("direct", {}) or {}
        one_sided = bool(direct_cfg.get("one_sided", True))
        gain_u = float(raw["contraction_mmHg"][1]) / ref_c
        gain_y = float(raw["decel_depth_bpm"][1]) / ref_d
        if one_sided:
            u_c = gain_u * np.maximum(c_tilde, 0.0)
            y_d = gain_y * np.maximum(d_tilde, 0.0)
        else:
            u_c = gain_u * c_tilde
            y_d = gain_y * d_tilde
        # Expose a flat unit "carrier" and treat the rendered deflection as its own
        # envelope so the S7 latent/AM decomposition figure stays well-defined (its
        # carrier panel simply reads flat at 1, communicating "no carrier").
        A_u, A_y = u_c, y_d
        carrier_u = np.ones(N, dtype=float)
        carrier_y = np.ones(N, dtype=float)
    else:
        # --- am_carrier / pulse_train rendering: coupled band = envelope x carrier -----
        # ``amp_peak`` is the envelope value at the +ref extreme (the physiological *peak*,
        # index [1] of the config range); the DC offset a0 is set by ``am_offset_ratio``,
        # so the [0] entry is a nominal floor and is intentionally not consumed here.
        A_u = am_envelope(
            c_tilde, sigma_ref=ref_c, am_offset_ratio=ratio, amp_peak=float(raw["contraction_mmHg"][1])
        )
        A_y = am_envelope(
            d_tilde, sigma_ref=ref_d, am_offset_ratio=ratio, amp_peak=float(raw["decel_depth_bpm"][1])
        )
        # Independent per-sample carrier phases for UP and FHR: the scattering modulus is
        # phase-blind, so this does not change the (envelope-carried) coupling in
        # up_st/fhr_st, but it removes the shared-carrier cross-correlation artifact that a
        # single deterministic carrier would create between the two raw signals. A (n, 1)
        # phase broadcasts through the carrier builder to a per-sample (n, N) carrier.
        if render_mode == "am_carrier":
            # Signed narrowband cosine carrier (default): cleanest AM model, symmetric about
            # the baseline -- best for the sign-blind scattering modulus (§7.1).
            phase_u = rng_carrier.uniform(0.0, 2.0 * math.pi, size=(n, 1))
            phase_y = rng_carrier.uniform(0.0, 2.0 * math.pi, size=(n, 1))
            carrier_u = make_carrier(N, fs, f_pulse, phase=phase_u)
            carrier_y = make_carrier(N, fs, f_pulse, phase=phase_y)
        else:  # render_mode == "pulse_train"
            # One-sided raised-cosine event train (waveform-realistic variant, §7.3): upward
            # contractions / downward decelerations. Its fundamental sits at ``rate_hz`` (the
            # carrier ``f_pulse`` by default), so the same fs-correct pulse-shape scattering
            # channel carries the coupling; its (generally lower) frac_Phi is measured in S7-T04.
            pt = raw.get("pulse_train", {}) or {}
            rate_hz = float(pt.get("rate_hz", f_pulse))
            duty = float(pt.get("duty", 0.5))
            phase_u = rng_carrier.uniform(0.0, 1.0, size=(n, 1))  # phase in cycles
            phase_y = rng_carrier.uniform(0.0, 1.0, size=(n, 1))
            carrier_u = make_pulse_train(N, fs, rate_hz, duty=duty, phase=phase_u)
            carrier_y = make_pulse_train(N, fs, rate_hz, duty=duty, phase=phase_y)
        u_c = A_u * carrier_u
        y_d = A_y * carrier_y

    # --- DC + independent dressing ---
    mu_fhr, mu_up = draw_dc(
        n, rng_dc, fhr_range=tuple(raw["mu_fhr_bpm"]), up_range=tuple(raw["mu_up"])
    )
    fhrv = synth_fhrv(n, N, fs, dict(raw["fhrv_band_power"]), rng_fhrv, notch=fhrv_notch)
    accels = synth_accelerations(
        n, N, fs,
        amp_bpm=tuple(raw["accel"]["amp_bpm"]),
        rate_per_min=float(raw["accel"]["rate_per_min"]),
        rng=rng_accel,
    )
    wander_std = raw.get("baseline_wander_std", {})
    fhr_wander = synth_baseline_wander(n, N, fs, float(wander_std.get("fhr", 0.0)), rng_fhr_wander)
    up_drift = synth_up_drift(n, N, fs, float(wander_std.get("up", 0.0)), rng_up_drift)
    noise = raw.get("noise_std", {})
    fhr_noise = white_noise(n, N, float(noise.get("fhr", 0.0)), rng_fhr_noise)
    up_noise = white_noise(n, N, float(noise.get("up", 0.0)), rng_up_noise)

    fhr_raw = mu_fhr[:, None] - y_d + fhrv + accels + fhr_wander + fhr_noise
    up_raw = mu_up[:, None] + u_c + up_drift + up_noise

    return {
        "fhr_raw": fhr_raw,
        "up_raw": up_raw,
        "true_lag_tt": true_lag_trajectory(n, T_tot, D),
        # ``latents`` carries the coupled-pathway intermediates for the S7-T03 AM
        # envelope/carrier decomposition figure: the decimated latents (c, d), their
        # band-limited upsamples (c_tilde, d_tilde), the strictly-positive envelopes
        # (A_u, A_y), the pulse-shape carriers (carrier_u, carrier_y), and the rendered
        # coupled bands (u_c, y_d). These are already computed above (no extra work);
        # the build path reads only fhr_raw/up_raw/true_lag_tt and drops the rest.
        "latents": {
            "c": c, "d": d, "c_tilde": c_tilde, "d_tilde": d_tilde,
            "A_u": A_u, "A_y": A_y,
            "carrier_u": carrier_u, "carrier_y": carrier_y,
            "u_c": u_c, "y_d": y_d,
        },
        "meta": {
            "D": int(D),
            "B": float(B),
            "te_inj": None if te_inj is None else float(te_inj),
            "render_mode": render_mode,
            "direct_one_sided": one_sided,
            "T_tot": T_tot,
            "n_raw": N,
            "fs": fs,
            "f_pulse": f_pulse,
            "fhrv_notch": None if fhrv_notch is None else [float(fhrv_notch[0]), float(fhrv_notch[1])],
        },
    }


# ---------------------------------------------------------------------------
# AM-separation analytic pre-check (S1-T04)
# ---------------------------------------------------------------------------


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    r"""Trapezoidal integral $\int y\,dx$ (version-agnostic; avoids deprecated ``np.trapz``).

    Args:
        y: Integrand samples.
        x: Monotone sample locations (same length as ``y``).

    Returns:
        The trapezoidal estimate of $\int y\,dx$.
    """
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def ar2_psd(f_hz: np.ndarray, r: float, w: float, sigma2_eta: float, fs_dec: float) -> np.ndarray:
    r"""One-sided AR(2) power spectral density on the decimated grid, in Hz.

    $$S(f) = \frac{\sigma_\eta^2}{\bigl|1 - \phi_1 e^{-i\Omega} - \phi_2 e^{-2i\Omega}\bigr|^2},
      \qquad \Omega = 2\pi f / f_s^{\mathrm{dec}}.$$

    Args:
        f_hz: Frequencies in Hz (array).
        r: Pole radius.
        w: Pole angle in rad/step.
        sigma2_eta: Source innovation variance.
        fs_dec: Decimated-grid sampling rate in Hz.

    Returns:
        The PSD evaluated at ``f_hz`` (same shape).
    """
    phi1, phi2 = ar2_coeffs(r, w)
    omega = 2.0 * math.pi * np.asarray(f_hz, dtype=float) / fs_dec
    denom = np.abs(1.0 - phi1 * np.exp(-1j * omega) - phi2 * np.exp(-2j * omega)) ** 2
    return sigma2_eta / denom


def wavelet_sigma_hz(f_pulse: float, Q: int, fs: float) -> float:
    r"""Analyzing-wavelet frequency-std $\sigma$ (Hz) at the carrier, from kymatio.

    Reads kymatio's own constant-Q width via
    :func:`kymatio.scattering1d.filter_bank.compute_sigma_psi` at the normalised
    centre $\xi = f_{\mathrm{pulse}} / f_s$, then converts to Hz ($\sigma_{\mathrm{Hz}}
    = \sigma_\xi f_s$). This is the *faithful* passband — the naive $f / Q$
    approximation over-estimates the width by $\sim 1.4\times$ because kymatio's $Q$
    counts wavelets-per-octave, not the classical quality factor (which is
    $\approx 5.78$ for $Q = 4$). The carrier $f_{\mathrm{pulse}}$ (locked
    $0.06\,\mathrm{Hz}$; originally $0.02$) sits in the constant-$Q$ regime (transition
    $\approx 0.0019\,\mathrm{Hz}$), where this formula is exact.

    Args:
        f_pulse: Carrier frequency in Hz.
        Q: Scattering wavelets-per-octave (``scattering.Q``, 4 in v2).
        fs: Raw sampling rate in Hz.

    Returns:
        The wavelet frequency standard deviation $\sigma$ in Hz.
    """
    from kymatio.scattering1d.filter_bank import compute_sigma_psi  # lazy: heavy import

    xi = f_pulse / fs
    return float(compute_sigma_psi(xi, Q)) * fs


def am_separation_margin(
    *,
    r: float,
    w: float,
    f_pulse: float,
    Q: int,
    fs: float,
    decimation: int = DECIMATION,
    am_offset_ratio: float,
    sigma2_eta: float,
    k_ref: float = AM_MAX_SIGMA_MULT,
) -> Dict[str, Any]:
    r"""AM-separation analytic pre-check for the coupled pulse-shape channel (S1-T04).

    Compares the AR(2) *envelope* spectrum to the analyzing wavelet's passband at the
    carrier, to gauge — before any transform or pilot — whether the amplitude
    fluctuations that carry the coupling will survive the sign-blind scattering
    modulus (§7). The wavelet demodulates the envelope with an effective Gaussian
    low-pass of frequency-std $\sigma_{\mathrm{wav}}$ (Hz), so an envelope component at
    frequency $f$ is attenuated by $\exp(-f^2 / \sigma_{\mathrm{wav}}^2)$ (power). The
    metrics returned:

    * ``margin_peak`` = $\sigma_{\mathrm{wav}} / f_{\mathrm{env,peak}}$, the ratio of
      the wavelet width to the envelope's dominant (pole-angle) sideband offset
      $f_{\mathrm{env,peak}} = \omega f_s^{\mathrm{dec}} / (2\pi)$. **Want $\ge 1$**
      (the dominant sideband falls within $1\sigma$).
    * ``margin_edge`` = $\sigma_{\mathrm{wav}} / f_{\mathrm{env,edge}}$ against the
      $-3\,\mathrm{dB}$ envelope upper edge $f_{\mathrm{env,peak}} + \tfrac12\Delta\omega$,
      where the *full* $-3\,\mathrm{dB}$ resonance width is $\Delta\omega \approx -2\ln r$
      rad/step (so the centre-to-edge half-width is $-\ln r$); a stricter,
      full-preservation bar.
    * ``preservation`` = $\int S_{\mathrm{env}}(f)\,e^{-f^2/\sigma_{\mathrm{wav}}^2}\,df
      \big/ \int S_{\mathrm{env}}(f)\,df$ using the *true* AR(2) PSD — a direct
      pre-estimate of $\mathrm{frac}_\Phi$.
    * ``mod_depth_rms`` = $1 / (k_{\mathrm{ref}}\cdot\texttt{am\_offset\_ratio})$, the
      RMS amplitude-modulation depth carrying the coupling. Shallow depth (the default
      ``am_offset_ratio = 4`` gives $\approx 0.0625$) is a leading $\mathrm{frac}_\Phi$
      risk — often more than the carrier frequency (technical review).

    A cell is flagged inadequate when ``margin_peak < 1``; the ``recommendation``
    string suggests the levers (deepen modulation via a smaller ``am_offset_ratio``;
    raise ``f_pulse`` — but note $0.05\,\mathrm{Hz}$ collides with the FHRV LF band on
    the FHR side — or narrow $\omega$ to lower $f_{\mathrm{env,peak}}$). The actual
    fix/proof is the Sprint 3 ``frac_Phi`` gate; this is only the pre-check.

    Args:
        r: Source pole radius.
        w: Source pole angle in rad/step.
        f_pulse: Carrier frequency in Hz.
        Q: Scattering wavelets-per-octave.
        fs: Raw sampling rate in Hz.
        decimation: Scattering decimation factor (16).
        am_offset_ratio: AM positivity ratio (``raw.am_offset_ratio``).
        sigma2_eta: Source innovation variance (for the PSD scale; cancels in the ratio).
        k_ref: Envelope-scale multiple (:data:`AM_MAX_SIGMA_MULT`).

    Returns:
        A dict with ``margin_peak``, ``margin_edge``, ``preservation``,
        ``sigma_wav_hz``, ``f_env_peak``, ``f_env_edge``, ``mod_depth_rms``,
        ``adequate`` (bool), and ``recommendation`` (str).
    """
    # This analysis assumes an under-damped oscillatory envelope: a zero pole angle
    # (w == 0) has no sideband offset and a zero radius (r == 0) has no memory, both of
    # which make the margin/bandwidth undefined. Fail with a clear message rather than a
    # cryptic ZeroDivisionError / math-domain error.
    if not 0.0 < r < 1.0:
        raise ValueError(f"am_separation_margin: r must be in (0, 1), got {r}.")
    if w <= 0.0:
        raise ValueError(f"am_separation_margin: w must be > 0 (oscillatory), got {w}.")

    fs_dec = fs / decimation
    sigma_wav = wavelet_sigma_hz(f_pulse, Q, fs)
    f_env_peak = w * fs_dec / (2.0 * math.pi)
    # AR(2) resonance -3 dB bandwidth: Delta_omega ~ -2 ln r rad/step is the FULL width
    # (valid for r -> 1); the centre-to-edge half-width added to f_env_peak is half of it.
    full_bw = (-2.0 * math.log(r)) * fs_dec / (2.0 * math.pi)
    half_bw = 0.5 * full_bw
    f_env_edge = f_env_peak + half_bw

    margin_peak = sigma_wav / f_env_peak
    margin_edge = sigma_wav / f_env_edge

    # Preservation integral against the true AR(2) PSD (power-weighted).
    f = np.linspace(0.0, fs_dec / 2.0, 4000)
    s_env = ar2_psd(f, r, w, sigma2_eta, fs_dec)
    weight = np.exp(-(f ** 2) / (sigma_wav ** 2))
    preservation = float(_trapezoid(s_env * weight, f) / _trapezoid(s_env, f))

    mod_depth_rms = 1.0 / (k_ref * am_offset_ratio)
    adequate = margin_peak >= 1.0

    if adequate:
        recommendation = (
            "AM separation adequate (margin_peak >= 1). Confirm with the Sprint 3 "
            "frac_Phi gate at build sample size."
        )
    else:
        recommendation = (
            f"AM separation marginal (margin_peak={margin_peak:.2f} < 1, "
            f"preservation~{preservation:.2f}). Levers: deepen modulation by lowering "
            f"am_offset_ratio (~1.5-2.5; current mod_depth_rms={mod_depth_rms:.3f}); "
            "raise f_pulse (note ~0.05 Hz overlaps the FHRV LF band 0.03-0.15 on the "
            "FHR side); or narrow omega to lower f_env_peak. Prove via the Sprint 3 "
            "frac_Phi gate."
        )

    return {
        "margin_peak": float(margin_peak),
        "margin_edge": float(margin_edge),
        "preservation": preservation,
        "sigma_wav_hz": float(sigma_wav),
        "f_env_peak": float(f_env_peak),
        "f_env_edge": float(f_env_edge),
        "mod_depth_rms": float(mod_depth_rms),
        "adequate": bool(adequate),
        "recommendation": recommendation,
    }


def am_separation_from_config(config: Dict[str, Any], benchmark: str = "G1_raw") -> Dict[str, Any]:
    r"""Run :func:`am_separation_margin` from a parsed config (the ``--am-check`` hook).

    Extracts $(r, \omega)$ from ``data.oscillators[0]`` and the carrier / ratio /
    sampling knobs from the ``raw`` and ``scattering`` blocks.

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.

    Returns:
        The :func:`am_separation_margin` result dict.
    """
    bench = config["benchmarks"][benchmark]
    data = bench["data"]
    raw = bench["raw"]
    scattering = bench["scattering"]
    r, w = float(data["oscillators"][0][0]), float(data["oscillators"][0][1])
    return am_separation_margin(
        r=r,
        w=w,
        f_pulse=float(raw["f_pulse"]),
        Q=int(scattering["Q"]),
        fs=float(raw["fs"]),
        am_offset_ratio=float(raw["am_offset_ratio"]),
        sigma2_eta=float(data["sigma2_eta"]),
    )
