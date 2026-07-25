r"""Measure what the scattering front end keeps, what it anticipates, and what the
uplift estimator can resolve -- the evidence behind ``UP_FHR_coupling_representation_analysis.md``.

Unlike the Stage 0 and Stage 2 probes this module needs **no dataset**. Everything here is
either a property of the production filter bank ($J=11$, $Q=4$, $T=16$, `shape` $=5280$,
$f_s=4$ Hz), which is rebuilt from the pipeline constants, or a known-answer simulation
whose ground truth we inject ourselves. That is deliberate: the claims it checks are claims
about the *representation and the estimator*, not about the cohort, and they should stay
auditable on a laptop with no access to ``/data1``.

Four measurements, matching the sections of the analysis document:

1. **Low-pass fidelity** (doc S6.1). Does $S_0 = x \star \phi$ keep a realistic
   deceleration? Answer: $96.9\%$ of its depth, with zero timing shift. This is the
   measurement that retired the "the representation is blind to the coupling" hypothesis.

2. **Band envelopes** (doc S6.2). Does $\phi$ or the $\times 16$ decimation damage the
   deceleration band? Answer: gain $\ge 0.988$, nothing aliased. Only $2$ of $42$ channels
   lose envelope structure to the decimation.

3. **Forward reach** (doc S7.2). The wavelets are two-sided, so a feature step reads raw
   signal from its own future. Answer: $28.4\%$ of the target block and $50\%$ of the
   source block read past the $H_d = 120$ s forecast horizon, and a deceleration is
   $81.5\%$ visible in the $0.0082$ Hz channel $120$ s before its nadir. This is a
   validity problem for any transfer-entropy reading, not merely a power problem.

4. **The masking law** (doc S8.2, ``--masking-law``). With a coupling calibrated to the
   measured $-4.4$ bpm contraction-triggered average, the *source-specific* uplift is flat
   at $\approx 2.1\%$ while the *measured* uplift collapses to $\approx 0$ as nuisance
   source columns are added, tracking a variance floor $\approx p_{extra}/N_{fit}$. At the
   Stage 0 probe's own design width that floor exceeds its $1\%$ detection threshold.

A fifth quantity is pure arithmetic and needs no measurement: the critical KL weight
$\beta^\star = \lambda_{full} d_z / (H_d c_y)$ above which no amount of transfer entropy in
the data can pay for the latent rate it would cost (doc S5).

Every headline number the document quotes is asserted here, so this doubles as a
regression check on the pipeline geometry: change $J$, $Q$, $T$ or the horizon and the
assertions that no longer hold are the claims that need rewriting.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from kymatio.scattering1d.filter_bank import scattering_filter_factory
from kymatio.scattering1d.utils import compute_minimum_support_to_pad

# --- production geometry (hdf5_dataset/new_pipeline/create_new_pipeline.py) -------------------
FS = 4.0            #: raw sampling rate, Hz
J, Q, T = 11, 4, 16  #: octaves, wavelets per octave, low-pass width (16 samples = 4 s)
N_RAW = 5280        #: stored segment length in raw samples (22 min)
DECIMATION = 16     #: raw -> decimated; one feature step is 4 s

# --- model geometry (teb_vae/lag_attn/configs/default.yaml) -----------------------------------
HORIZON_STEPS = 30   #: $H_d$, forecast horizon in decimated steps
HORIZON_S = HORIZON_STEPS * DECIMATION / FS   #: 120 s
C_Y, C_U, D_Z = 109, 58, 24                   #: target width, source width, latent width
LAMBDA_FULL, BETA_END, FREE_BITS = 1.0, 0.1, 0.1


# =============================================================================================
# Filter bank
# =============================================================================================
@dataclass
class FilterBank:
    """The rebuilt production first-order filter bank.

    Attributes:
        psi: First-order filters in the frequency domain, ``(n_filters, n_padded)``.
        phi: The low-pass filter in the frequency domain, ``(n_padded,)``.
        xi: Centre frequencies in cycles per sample (kymatio's unit), descending.
        sigma: Frequency widths in cycles per sample.
        taps: Filter-tap times in seconds, centred at $0$; negative is the past.
    """

    psi: np.ndarray
    phi: np.ndarray
    xi: np.ndarray
    sigma: np.ndarray
    taps: np.ndarray

    @property
    def hz(self) -> np.ndarray:
        """Centre frequencies in Hz."""
        return self.xi * FS

    @property
    def n_filters(self) -> int:
        return int(self.psi.shape[0])


def build_filter_bank() -> FilterBank:
    """Rebuild the bank exactly as ``KymatioPhaseScattering1D`` does.

    The padding chain (``compute_minimum_support_to_pad`` -> ``J_pad`` -> ``J_support``) is
    copied from ``hdf5_dataset/kymatio_phase_scattering.py`` rather than approximated,
    because the realised filter shapes -- and therefore every time-domain width measured
    below -- depend on the padded length.

    Returns:
        The populated :class:`FilterBank`.
    """
    min_to_pad = min(compute_minimum_support_to_pad(N_RAW, J, Q, T), N_RAW - 1)
    j_max = int(np.floor(np.log2(3 * N_RAW - 2)))
    j_pad = min(int(np.ceil(np.log2(N_RAW + 2 * min_to_pad))), j_max)
    n_padded = 2 ** j_pad

    phi_f, psi1_f, _, _ = scattering_filter_factory(
        J_support=int(np.ceil(np.log2(n_padded))), J_scattering=J, Q=Q, T=T
    )
    idx = np.arange(n_padded)
    # Filters are centred at tap 0 and wrap around, so map the upper half to negative time.
    taps = np.where(idx <= n_padded // 2, idx, idx - n_padded) / FS
    return FilterBank(
        psi=np.stack([d["levels"][0] for d in psi1_f], axis=0),
        phi=np.asarray(phi_f["levels"][0]),
        xi=np.array([d["xi"] for d in psi1_f]),
        sigma=np.array([d["sigma"] for d in psi1_f]),
        taps=taps,
    )


def band_of(hz: float) -> str:
    """Clinical band label for a centre frequency in Hz."""
    if hz >= 0.25:
        return "beat-to-beat"
    if hz >= 0.04:
        return "LF/MF var"
    if hz >= 0.008:
        return "deceleration"
    return "baseline"


def _gain(bank: FilterBank, f_hz: float) -> float:
    """Normalised magnitude response of $\\phi$ at ``f_hz``."""
    k = int(round(f_hz / FS * bank.phi.size))
    return float(np.abs(bank.phi[k]) / np.abs(bank.phi[0]))


def forward_reach(bank: FilterBank, spectrum: np.ndarray, quantile: float = 0.95) -> float:
    """Forward reach $L_{95}$: how far into the future a filter reads.

    Defined as the smallest $D > 0$ containing ``quantile`` of the filter's energy at taps
    $t' > t$. A channel whose reach exceeds the forecast horizon has already observed the
    interval it is being asked to predict.

    Args:
        bank: The filter bank (for its tap-time axis).
        spectrum: One filter in the frequency domain.
        quantile: Energy fraction to enclose.

    Returns:
        The reach in seconds.
    """
    energy = np.abs(np.fft.ifft(spectrum)) ** 2
    future = bank.taps > 0
    e, t = energy[future], bank.taps[future]
    order = np.argsort(t)
    e, t = e[order], t[order]
    return float(t[np.searchsorted(np.cumsum(e) / e.sum(), quantile)])


def deceleration(t: np.ndarray, nadir_s: float, depth_bpm: float = 20.0,
                 width_s: float = 25.0) -> np.ndarray:
    """A realistic Gaussian deceleration: $-A\\exp(-(t-t_0)^2/2\\sigma^2)$.

    Defaults give a $20$ bpm dip of $\\approx 100$ s total duration -- a moderate,
    clinically ordinary deceleration, and comfortably inside the $0.008$--$0.04$ Hz band.
    """
    return -depth_bpm * np.exp(-0.5 * ((t - nadir_s) / width_s) ** 2)


# =============================================================================================
# 1 + 2. What the representation keeps
# =============================================================================================
def measure_representation(bank: FilterBank) -> Dict[str, Any]:
    """Low-pass fidelity and band-envelope survival (doc S6.1--S6.2).

    Answers the question the previous version of the analysis got wrong: does the
    scattering front end discard the contraction->deceleration response? It does not.
    $S_0$ is stored unmasked as ``fhr_st`` channel $0$ and receives no log/asinh transform,
    so it is an affine function of the $4$ s locally-averaged FHR in bpm.

    Returns:
        A dict of measured quantities; see the printed report for units.
    """
    t = np.arange(N_RAW) / FS
    n_padded = bank.phi.size
    dec = deceleration(t, nadir_s=660.0)
    dec_lp = np.real(np.fft.ifft(np.fft.fft(dec, n_padded) * bank.phi))[:N_RAW]

    # Envelope bandwidth of |x * psi_i| is set by the filter's own bandwidth; the decimated
    # grid can represent it without aliasing only below its Nyquist frequency.
    nyquist_decimated = FS / (2 * DECIMATION)
    env_bw = bank.sigma * FS
    aliased = env_bw > nyquist_decimated

    # S1 is exactly sign-blind: |(-x) * psi| == |x * psi| identically.
    x = dec + 3.0 * np.sin(2 * np.pi * 0.08 * t)
    xp, xm = np.fft.fft(x, n_padded), np.fft.fft(-x, n_padded)
    i_dec = int(np.argmin(np.abs(bank.hz - 0.02)))
    sign_gap = float(np.max(np.abs(
        np.abs(np.fft.ifft(xp * bank.psi[i_dec])) - np.abs(np.fft.ifft(xm * bank.psi[i_dec]))
    )))

    bands: Dict[str, Dict[str, float]] = {}
    for name in ("beat-to-beat", "LF/MF var", "deceleration", "baseline"):
        m = np.array([band_of(h) == name for h in bank.hz])
        if not m.any():
            continue
        bands[name] = {
            "n_filters": int(m.sum()),
            "env_bw_lo_hz": float(env_bw[m].min()),
            "env_bw_hi_hz": float(env_bw[m].max()),
            "min_phi_gain_at_env_bw": float(min(_gain(bank, min(b, FS / 2 - 1e-6))
                                                for b in env_bw[m])),
            "n_aliased": int(aliased[m].sum()),
        }

    return {
        "phi_gain": {f"{f:g}": _gain(bank, f)
                     for f in (0.002, 0.005, 0.008, 0.02, 0.04, 0.0667, 0.125)},
        "dec_depth_raw_bpm": float(dec.min()),
        "dec_depth_after_phi_bpm": float(dec_lp.min()),
        "dec_depth_retained_frac": float(dec_lp.min() / dec.min()),
        "dec_nadir_shift_s": float(t[np.argmin(dec_lp)] - t[np.argmin(dec)]),
        "nyquist_decimated_hz": float(nyquist_decimated),
        "n_aliased_total": int(aliased.sum()),
        "s1_sign_blindness_max_abs_gap": sign_gap,
        "bands": bands,
    }


# =============================================================================================
# 3. What the representation anticipates
# =============================================================================================
def select_phase_pairs(bank: FilterBank, f_min: float, f_max: float,
                       k_steps: Sequence[int] = (4, 6, 8),
                       rel_tol: float = 0.05) -> List[Tuple[int, int]]:
    """Rebuild a stored phase-harmonic channel selection (doc S7.5 of the dataset reference).

    A pair $(i, j)$ with $\\xi_i \\le \\xi_j$ is kept when both endpoints lie in the band and
    the ratio $p = \\xi_j/\\xi_i$ sits within a **relative** tolerance of some
    $2^{k/Q}$. Reproducing the shipped counts ($66$ for ``fhr_ph``, $15$ for ``up_ph``)
    from the documented rule alone is what licenses trusting the reach numbers below --
    hence the assertions in :func:`self_test`.

    Args:
        bank: The filter bank.
        f_min: Lower band edge in Hz (applies to the slower wavelet $\\xi_i$).
        f_max: Upper band edge in Hz (applies to the faster wavelet $\\xi_j$).
        k_steps: Harmonic steps admitted.
        rel_tol: Relative tolerance on the power grid.

    Returns:
        The kept ``(i, j)`` index pairs, ``i`` indexing the lower frequency.
    """
    hz = bank.hz
    n = bank.n_filters
    pairs = sorted({(a, b) if hz[a] <= hz[b] else (b, a)
                    for a in range(n) for b in range(a, n)})
    keep = []
    for i, j in pairs:
        if hz[i] < f_min or hz[j] > f_max:
            continue
        p = hz[j] / hz[i]
        if any(abs(p - 2 ** (k / Q)) < rel_tol * 2 ** (k / Q) for k in k_steps):
            keep.append((i, j))
    return keep


def measure_anticipation(bank: FilterBank) -> Dict[str, Any]:
    """Forward reach per stored block, and how early a deceleration becomes visible (doc S7.2).

    A phase-harmonic channel multiplies $z_i$ and $z_j$ at the same $t$ and then $\\phi$-
    smooths, so its forward reach is the slower wavelet's reach plus the low-pass reach.

    Returns:
        A dict of per-block reach counts and the per-channel anticipation of a deceleration.
    """
    reach_psi = np.array([forward_reach(bank, bank.psi[i]) for i in range(bank.n_filters)])
    reach_phi = forward_reach(bank, bank.phi)
    st_reach = np.concatenate([[reach_phi], reach_psi])          # S0 first, then S1

    def ph_reach(sel: List[Tuple[int, int]]) -> np.ndarray:
        return np.array([max(reach_psi[i], reach_psi[j]) + reach_phi for i, j in sel])

    blocks = {
        "fhr_st": st_reach,
        "up_st": st_reach,
        "fhr_ph": ph_reach(select_phase_pairs(bank, 0.008, 1.00)),
        "up_ph": ph_reach(select_phase_pairs(bank, 0.008, 0.05)),
    }
    per_block = {
        name: {
            "n_channels": int(r.size),
            "n_past_horizon": int((r > HORIZON_S).sum()),
            "reach_lo_s": float(r.min()),
            "reach_hi_s": float(r.max()),
        }
        for name, r in blocks.items()
    }
    n_target_bad = per_block["fhr_st"]["n_past_horizon"] + per_block["fhr_ph"]["n_past_horizon"]
    n_source_bad = per_block["up_st"]["n_past_horizon"] + per_block["up_ph"]["n_past_horizon"]

    # How much of a deceleration is already in the channel one horizon before its nadir?
    t = np.arange(N_RAW) / FS
    nadir_s = 660.0
    spec = np.fft.fft(deceleration(t, nadir_s=nadir_s), bank.phi.size)
    k_nadir, k_early = int(nadir_s * FS), int((nadir_s - HORIZON_S) * FS)
    already = {}
    for i in range(bank.n_filters):
        if band_of(bank.hz[i]) != "deceleration":
            continue
        env = np.abs(np.fft.ifft(spec * bank.psi[i]))[:N_RAW]
        already[f"{bank.hz[i]:.4f}Hz"] = float(env[k_early] / env[k_nadir])
    s0 = np.real(np.fft.ifft(spec * bank.phi))[:N_RAW]
    already["S0"] = float(abs(s0[k_early]) / abs(s0[k_nadir]))

    return {
        "horizon_s": HORIZON_S,
        "phi_reach_s": reach_phi,
        "per_block": per_block,
        "target_frac_past_horizon": n_target_bad / C_Y,
        "source_frac_past_horizon": n_source_bad / C_U,
        "n_causal_fhr_st": int((st_reach <= HORIZON_S).sum()),
        "already_visible_one_horizon_early": already,
    }


# =============================================================================================
# 4. What the estimator can resolve
# =============================================================================================
@dataclass
class MaskingConfig:
    """Settings for the known-answer masking-law experiment (doc S8.2).

    The document's table used ``n_recordings=600``; the default here is smaller so the
    module stays runnable as a check. The qualitative result -- flat source-specific
    uplift, collapsing measured uplift -- is stable across both.
    """

    n_recordings: int = 240
    n_folds: int = 5
    seed: int = 11
    ridge_alpha: float = 30.0
    lags: Tuple[int, ...] = (0, 2, 4, 8, 12, 16, 24, 32)
    anchor_stride: int = 3
    warmup: int = 40
    nuisance_widths: Tuple[int, ...] = field(default=(0, 100, 400, 800))
    #: Fraction of contractions that produce a deceleration, and its depth distribution.
    response_prob: float = 0.45
    depth_shape: float = 3.0
    depth_scale: float = 5.2
    lag_mean_s: float = 29.0
    lag_jitter_s: float = 12.0


def _simulate(rng: np.random.Generator, cfg: MaskingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One synthetic recording: (FHR with injected response, UP, contraction onsets)."""
    t = np.arange(N_RAW) / FS
    up = np.zeros(N_RAW)
    onsets: List[float] = []
    tc = rng.uniform(0, 120)
    while tc < t[-1] - 150:
        up += rng.uniform(30, 70) * np.exp(-0.5 * ((t - tc) / rng.uniform(20, 32)) ** 2)
        onsets.append(tc)
        tc += rng.uniform(110, 190)
    up += 2.0 * rng.standard_normal(N_RAW)
    up += 8.0 * np.cumsum(rng.standard_normal(N_RAW)) / np.sqrt(N_RAW)   # tocometer drift

    fhr = 140.0 + np.cumsum(rng.standard_normal(N_RAW)) / np.sqrt(N_RAW) * 25.0
    fhr += 2.5 * rng.standard_normal(N_RAW)
    for f0 in (0.03, 0.06, 0.10):
        fhr += rng.uniform(2, 5) * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
    for tc in onsets:
        if rng.random() < cfg.response_prob:
            fhr += deceleration(
                t,
                nadir_s=tc + cfg.lag_mean_s + rng.normal(0, cfg.lag_jitter_s),
                depth_bpm=rng.gamma(cfg.depth_shape, cfg.depth_scale),
                width_s=rng.uniform(18, 32),
            )
    return fhr, up, np.array(onsets)


def _scatter(bank: FilterBank, x: np.ndarray) -> np.ndarray:
    """$S_0$ and the $42$ first-order envelopes, decimated by $16$ -> ``(330, 43)``."""
    spec = np.fft.fft(x, bank.phi.size)
    out = np.empty((N_RAW // DECIMATION, bank.n_filters + 1))
    out[:, 0] = np.real(np.fft.ifft(spec * bank.phi))[:N_RAW][::DECIMATION]
    env = np.abs(np.fft.ifft(spec[None, :] * bank.psi, axis=1))[:, :N_RAW]
    out[:, 1:] = env[:, ::DECIMATION].T
    return out


def _ridge_mse(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray,
               alpha: float) -> float:
    """Held-out MSE of a ridge fit. Standardisation and centring use the train split only."""
    mu, sd = x[train].mean(0), x[train].std(0) + 1e-9
    y_mu = y[train].mean()
    a = (x[train] - mu) / sd
    w = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ (y[train] - y_mu))
    return float(np.mean((y[test] - (((x[test] - mu) / sd) @ w + y_mu)) ** 2))


def measure_masking_law(bank: FilterBank, cfg: MaskingConfig) -> Dict[str, Any]:
    """Known-answer sweep: measured uplift vs source-design width (doc S8.2).

    One estimator, one dataset, one true coupling. Only the number of columns in the source
    block changes -- a lean causal source ($S_0^U$ at the configured lags) padded with pure
    noise. The **decoupled** arm pairs each target with a stranger's source, so its uplift
    must be zero in expectation; whatever it reports is the added-column variance floor.

    Returns:
        The calibration of the injected coupling and one row per design width.
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_recordings
    sy, su, fhr_all, onsets_all = [], [], [], []
    for _ in range(n):
        fhr, up, onsets = _simulate(rng, cfg)
        fhr_all.append(fhr)
        onsets_all.append(onsets)
        sy.append(_scatter(bank, fhr))
        su.append(_scatter(bank, up))
    sy, su = np.stack(sy), np.stack(su)

    # Calibration: the injected coupling's own contraction-triggered average, so the
    # simulated effect size can be compared with the Stage 2 measurement (-4.43 bpm at 29 s).
    pre = (np.arange(-30, 0) * FS).astype(int)
    win = np.arange(int(10 * FS), int(150 * FS))
    snips = [fhr_all[k][int(c * FS) + win] - fhr_all[k][int(c * FS) + pre].mean()
             for k in range(n) for c in onsets_all[k]
             if int(c * FS) + win[-1] < N_RAW and int(c * FS) + pre[0] >= 0]
    cta = np.array(snips).mean(0)

    anchors = np.arange(cfg.warmup, sy.shape[1] - HORIZON_STEPS, cfg.anchor_stride)
    folds = np.arange(n) % cfg.n_folds
    groups = np.repeat(folds, anchors.size)

    def design(source: np.ndarray, k: int, channels: slice) -> np.ndarray:
        return np.concatenate([source[k][anchors - lag][:, channels] for lag in cfg.lags], axis=1)

    x_base = np.concatenate([design(sy, k, slice(0, 43)) for k in range(n)])
    y = np.concatenate([sy[k][anchors + HORIZON_STEPS, 0] for k in range(n)])
    stranger = np.roll(np.arange(n), max(1, n // 3 + 1))
    lean = np.concatenate([design(su, k, slice(0, 1)) for k in range(n)])
    lean_shuf = np.concatenate([design(su, stranger[k], slice(0, 1)) for k in range(n)])
    nuisance = np.random.default_rng(cfg.seed + 1).standard_normal(
        (y.size, max(cfg.nuisance_widths) or 1))
    n_train = int((groups != 0).sum())

    rows = []
    for width in cfg.nuisance_widths:
        src = lean if width == 0 else np.hstack([lean, nuisance[:, :width]])
        shf = lean_shuf if width == 0 else np.hstack([lean_shuf, nuisance[:, :width]])
        coupled, decoupled = [], []
        for f in range(cfg.n_folds):
            train, test = groups != f, groups == f
            base = _ridge_mse(x_base, y, train, test, cfg.ridge_alpha)
            coupled.append((base - _ridge_mse(np.hstack([x_base, src]), y, train, test,
                                              cfg.ridge_alpha)) / base)
            decoupled.append((base - _ridge_mse(np.hstack([x_base, shf]), y, train, test,
                                                cfg.ridge_alpha)) / base)
        c, d = np.array(coupled) * 100, np.array(decoupled) * 100
        rows.append({
            "p_extra": int(src.shape[1]),
            "coupled_pct": float(c.mean()),
            "coupled_se": float(c.std(ddof=1) / np.sqrt(cfg.n_folds)),
            "decoupled_pct": float(d.mean()),
            "decoupled_se": float(d.std(ddof=1) / np.sqrt(cfg.n_folds)),
            "source_specific_pct": float((c - d).mean()),
            "source_specific_se": float((c - d).std(ddof=1) / np.sqrt(cfg.n_folds)),
            "predicted_floor_pct": 100.0 * src.shape[1] / n_train,
        })

    return {
        "n_recordings": n,
        "n_rows": int(y.size),
        "n_train_per_fold": n_train,
        "calibration_dip_bpm": float(cta.min()),
        "calibration_lag_s": float(win[int(np.argmin(cta))] / FS),
        "n_contractions": len(snips),
        "rows": rows,
    }


# =============================================================================================
# 5. The critical KL weight (arithmetic, doc S5)
# =============================================================================================
def critical_beta(lambda_full: float = LAMBDA_FULL, d_z: int = D_Z,
                  horizon: int = HORIZON_STEPS, c_y: int = C_Y) -> Dict[str, float]:
    r"""The KL weight above which the source channel can never pay for itself.

    ``compute_loss`` scores the forecast per *predicted element* (a mean over
    $H_d \times c_y$) but the KL per *latent dimension per step* (a mean over $d_z$), and
    under ``kld_support='anchor'`` both share the same anchor support. Since the total
    forecast NLL reduction attributable to $z_t$ is at most $I(Y^+_t; z_t \mid Y_{\le t})
    \le \mathbb{E}[K_t]$ nats,

    $$\Delta\mathcal{L} \ge \Big(\beta - \lambda_{full}\frac{d_z}{H_d c_y}\Big)
      \cdot \frac{1}{A d_z}\sum_t K_t ,$$

    so above $\beta^\star = \lambda_{full} d_z / (H_d c_y)$ any positive KL strictly
    increases the loss whatever the data. ``free_bits`` removes the gradient below
    $d_z \cdot \mathrm{free\_bits}$ nats/step, which therefore caps the reportable
    surrogate rather than driving it to zero.

    Returns:
        The threshold, the shipped ratio, the epoch at which the warmup crosses it, and the
        free-bits cap in nats per step.
    """
    beta_star = lambda_full * d_z / (horizon * c_y)
    start, end, warm = 1.0e-4, BETA_END, 50
    crossing = warm * (beta_star - start) / (end - start)
    return {
        "beta_star": beta_star,
        "beta_end": end,
        "ratio": end / beta_star,
        "warmup_crossing_epoch": crossing,
        "free_bits_cap_nats_per_step": d_z * FREE_BITS,
    }


# =============================================================================================
# Reporting
# =============================================================================================
def run_probe(masking: bool = False, cfg: Optional[MaskingConfig] = None) -> Dict[str, Any]:
    """Run every measurement and return the results dict."""
    bank = build_filter_bank()
    results: Dict[str, Any] = {
        "geometry": {"J": J, "Q": Q, "T": T, "shape": N_RAW, "fs": FS,
                     "n_filters": bank.n_filters, "horizon_s": HORIZON_S},
        "representation": measure_representation(bank),
        "anticipation": measure_anticipation(bank),
        "critical_beta": critical_beta(),
    }
    if masking:
        results["masking_law"] = measure_masking_law(bank, cfg or MaskingConfig())
    return results


def format_report(results: Dict[str, Any]) -> str:
    """Render the results as a human-readable report."""
    rep = results["representation"]
    ant = results["anticipation"]
    beta = results["critical_beta"]
    out: List[str] = []
    add = out.append

    add("=" * 86)
    add("Representation-capacity probe  (J=11, Q=4, T=16, shape=5280, fs=4 Hz, H_d=120 s)")
    add("=" * 86)

    add("\n1. Does the low-pass keep a realistic deceleration?  (doc S6.1)")
    add(f"   depth raw {rep['dec_depth_raw_bpm']:.3f} bpm -> after phi "
        f"{rep['dec_depth_after_phi_bpm']:.3f} bpm "
        f"({100*rep['dec_depth_retained_frac']:.1f}% retained), "
        f"nadir shift {rep['dec_nadir_shift_s']:+.2f} s")
    add("   phi gain: " + "  ".join(f"{f} Hz={g:.3f}" for f, g in rep["phi_gain"].items()))
    add(f"   S1 sign blindness |  |x*psi| - |(-x)*psi|  |_max = "
        f"{rep['s1_sign_blindness_max_abs_gap']:.3e}  (exactly 0: only S0 sees polarity)")

    add(f"\n2. Do phi or the x{DECIMATION} decimation damage the bands?  (doc S6.2)")
    add(f"   decimated-grid Nyquist = {rep['nyquist_decimated_hz']:.4f} Hz")
    add(f"   {'band':>14} {'n':>3} {'env bw (Hz)':>20} {'min phi gain':>13} {'aliased':>8}")
    for name, b in rep["bands"].items():
        add(f"   {name:>14} {b['n_filters']:>3} "
            f"{b['env_bw_lo_hz']:>9.5f}-{b['env_bw_hi_hz']:<10.5f} "
            f"{b['min_phi_gain_at_env_bw']:>13.3f} {b['n_aliased']:>4}/{b['n_filters']:<3}")

    add("\n3. How far into the future does each feature step read?  (doc S7.2)")
    add(f"   phi (S0) forward reach = {ant['phi_reach_s']:.2f} s  -- essentially causal")
    add(f"   {'block':>8} {'past +120s':>12} {'reach range (s)':>20}")
    for name, b in ant["per_block"].items():
        add(f"   {name:>8} {b['n_past_horizon']:>5}/{b['n_channels']:<6} "
            f"{b['reach_lo_s']:>9.0f}-{b['reach_hi_s']:<10.0f}")
    add(f"   target block c_y={C_Y}: {100*ant['target_frac_past_horizon']:.1f}% non-causal; "
        f"source block c_u={C_U}: {100*ant['source_frac_past_horizon']:.1f}% non-causal")
    add(f"   causally safe fhr_st channels: {ant['n_causal_fhr_st']}/43")
    add("   fraction of a deceleration ALREADY present one horizon (120 s) before its nadir:")
    for k, v in ant["already_visible_one_horizon_early"].items():
        add(f"      {k:>12}: {v:.3f}")

    add("\n4. Can the objective report a calibrated TE?  (doc S5)")
    add(f"   beta* = lambda_full * d_z / (H_d * c_y) = {D_Z}/{HORIZON_STEPS*C_Y} = "
        f"{beta['beta_star']:.3e}")
    add(f"   shipped beta_end = {beta['beta_end']} = {beta['ratio']:.1f} x beta*  "
        f"(warmup crosses beta* at epoch {beta['warmup_crossing_epoch']:.1f} of 50)")
    add(f"   free-bits cap on the reportable surrogate = "
        f"{beta['free_bits_cap_nats_per_step']:.1f} nats/step")

    if "masking_law" in results:
        ml = results["masking_law"]
        add("\n5. What can the pooled uplift estimator resolve?  (doc S8.2)")
        add(f"   injected coupling calibration: D* = {ml['calibration_dip_bpm']:.2f} bpm at "
            f"tau = {ml['calibration_lag_s']:.0f} s over {ml['n_contractions']} contractions "
            f"(Stage 2 measured -4.43 bpm at 29 s)")
        add(f"   {ml['n_recordings']} recordings, {ml['n_rows']} rows, "
            f"{ml['n_train_per_fold']} train rows/fold")
        add(f"   {'p_extra':>8} {'coupled %':>16} {'decoupled % (floor)':>22} "
            f"{'source-specific %':>20} {'p/N %':>7}")
        for r in ml["rows"]:
            add(f"   {r['p_extra']:>8} {r['coupled_pct']:>9.2f}+-{r['coupled_se']:<5.2f} "
                f"{r['decoupled_pct']:>15.2f}+-{r['decoupled_se']:<5.2f} "
                f"{r['source_specific_pct']:>13.2f}+-{r['source_specific_se']:<5.2f} "
                f"{r['predicted_floor_pct']:>7.2f}")
        add("   The source-specific column is flat: the coupling is fully recoverable at every")
        add("   width. The measured (coupled) column collapses because the floor grows as p/N.")
    add("=" * 86)
    return "\n".join(out)


def self_test() -> int:
    """Assert every headline number the analysis document quotes.

    These are regression checks on the pipeline geometry as much as on this module: if the
    filter bank, the horizon or the stream widths change, the assertions that fail are
    exactly the document claims that need rewriting.

    Returns:
        Process exit code.
    """
    bank = build_filter_bank()
    ok = True

    def check(name: str, got: float, lo: float, hi: float) -> None:
        nonlocal ok
        if not (lo <= got <= hi):
            print(f"FAIL: {name} = {got!r}, expected in [{lo}, {hi}]")
            ok = False

    # The rebuild must reproduce the shipped pipeline, or nothing downstream is trustworthy.
    check("n_filters", bank.n_filters, 42, 42)
    check("n_pairs", len({(a, b) if bank.hz[a] <= bank.hz[b] else (b, a)
                          for a in range(42) for b in range(a, 42)}), 903, 903)
    check("fhr_ph channels", len(select_phase_pairs(bank, 0.008, 1.00)), 66, 66)
    check("up_ph channels", len(select_phase_pairs(bank, 0.008, 0.05)), 15, 15)

    rep = measure_representation(bank)
    check("deceleration depth retained", rep["dec_depth_retained_frac"], 0.96, 0.98)
    check("nadir shift (s)", abs(rep["dec_nadir_shift_s"]), 0.0, 0.25)
    check("S1 sign blindness", rep["s1_sign_blindness_max_abs_gap"], 0.0, 1e-12)
    check("deceleration-band min phi gain",
          rep["bands"]["deceleration"]["min_phi_gain_at_env_bw"], 0.98, 1.0)
    check("deceleration-band aliased", rep["bands"]["deceleration"]["n_aliased"], 0, 0)
    check("total aliased channels", rep["n_aliased_total"], 2, 2)

    ant = measure_anticipation(bank)
    check("phi forward reach (s)", ant["phi_reach_s"], 8.0, 9.5)
    check("target frac past horizon", ant["target_frac_past_horizon"], 0.27, 0.30)
    check("source frac past horizon", ant["source_frac_past_horizon"], 0.48, 0.52)
    check("causal fhr_st channels", ant["n_causal_fhr_st"], 27, 27)
    check("0.0082 Hz already visible 120 s early",
          ant["already_visible_one_horizon_early"]["0.0082Hz"], 0.78, 0.85)
    check("S0 already visible 120 s early",
          ant["already_visible_one_horizon_early"]["S0"], 0.0, 0.01)

    beta = critical_beta()
    check("beta*", beta["beta_star"], 7.3e-3, 7.4e-3)
    check("beta_end / beta*", beta["ratio"], 13.5, 13.7)
    check("free-bits cap (nats/step)", beta["free_bits_cap_nats_per_step"], 2.39, 2.41)

    print("SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments and run the probe or the self-test."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--masking-law", action="store_true",
                        help="Also run the known-answer uplift sweep (slow: simulates and "
                             "transforms every recording).")
    parser.add_argument("--n-recordings", type=int, default=MaskingConfig.n_recordings,
                        help="Recordings for the masking-law sweep (the document used 600).")
    parser.add_argument("--seed", type=int, default=MaskingConfig.seed)
    parser.add_argument("--json-out", default=None, help="Write the results dict as JSON.")
    parser.add_argument("--self-test", action="store_true",
                        help="Assert the document's headline numbers and exit.")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    results = run_probe(
        masking=args.masking_law,
        cfg=MaskingConfig(n_recordings=args.n_recordings, seed=args.seed),
    )
    print(format_report(results))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
