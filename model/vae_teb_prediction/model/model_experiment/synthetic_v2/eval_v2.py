r"""The three-TE de-risk probes and the pre-flight harness for ``synthetic_v2`` (Sprint 3).

This module measures — **model-free, before any GPU training** — how much of the
injected block transfer entropy survives raw rendering and the real scattering
transform, so a downstream calibration failure can be attributed to the *transform*
rather than the *model* (§10, §14.3). It provides:

* :func:`slice_coupled_channels` (S3-T02) — cut the fs-correct coupled decel /
  contraction pulse-shape scattering channels into the single-channel arrays the
  ported R0 probe :func:`analytic_te.realizable_te_block_from_arrays` expects.
* :func:`measure_te_scat` (S3-T03) — the feature-space realizability probe
  $\mathrm{TE}_{\mathrm{scat}}$ and the preservation fraction
  $\mathrm{frac}_\Phi = \mathrm{TE}_{\mathrm{scat}} / \mathrm{TE}_{\mathrm{inj}}$.
* :func:`measure_te_raw` (S3-T04) — the raw-waveform TE
  $\mathrm{TE}_{\mathrm{raw}} = I(x_{\mathrm{FHR}}^{+}; x_{\mathrm{UP}}^{-} \mid
  x_{\mathrm{FHR}}^{-})$ on the band-limited-decimated raw signals.
* :func:`run_realizability_preflight` (S3-T05) — the pilot-grid harness that runs all
  three probes per cell, writes ``realizability.json``, and honours the fatal gate.
* :func:`sweep_render_knobs` (S3-T06) — the recovery sweep over the render knobs
  (``f_pulse`` / ``am_offset_ratio`` / $\omega$) when $\mathrm{frac}_\Phi$ is low.

Evaluation *gates* on a trained checkpoint (K-bar calibration, lag recovery, null
controls) land in Sprint 6. See ``SYNTHETIC_V2_RAW_TE_PIPELINE_EXPLAINED.md`` §10,
§14 and ``SYNTHETIC_V2_SPEC_AND_SPRINTS.md`` Sprint 3.
"""

from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .analytic_te import realizable_te_block_from_arrays, snr_per_step_for_te_block
from .build_dataset_v2 import (
    CellV2,
    enumerate_cells_v2,
    generate_pilot_samples,
    resolve_cache_dir,
    solve_cell_coupling,
)
from .raw_generators import DECIMATION, am_separation_margin, ar2_lag1_autocorr

logger = logging.getLogger(__name__)

# Edge trim per end on the decimated grid (mirrors ``scattering_adapter.TRIM_STEPS``;
# duplicated as a plain int so the pure-NumPy probes and their tests do not import
# the torch/kymatio-heavy adapter). ``run_realizability_preflight`` passes the live
# adapter's ``trim`` / ``T`` so any change there overrides this default.
_TRIM_STEPS: int = 15

# Carrier band half-width factor for the optional bandpassed TE_raw variant. The AM
# rendering up-converts BOTH coupled terms (u_c, y_d) onto the pulse-shape carrier at
# ``f_pulse`` (§7), so the coupled energy lives near ``f_pulse`` -- NOT in the low-
# frequency envelope/rhythm band. The diagnostic band is therefore centred on the
# carrier: ``[f_pulse / factor, f_pulse * factor]`` (factor 1.6 ~ +-2/3 octave, wide
# enough for the +-envelope-bandwidth sidebands, narrow enough to reject VLF dressing
# and HF noise).
_CARRIER_BAND_FACTOR: float = 1.6

# Z-score epsilon (matches the production per-channel z-score).
_ZSCORE_EPS: float = 1e-8


def _zscore_channel(x: np.ndarray) -> np.ndarray:
    r"""Pooled per-channel z-score of a single-channel signal block $(n, T)$.

    The R0 probe (:func:`analytic_te.realizable_te_block_from_arrays`) scales its ridge by
    the Gram diagonal, so an un-removed physiological DC (~140 bpm / ~15 mmHg) would
    inflate the penalty by $\mathrm{DC}^2$ and shrink the coupling coefficients to ~0. The
    scattering ``*_st`` features are already per-channel z-scored, so the raw-domain probe
    must match. Uses a single pooled mean/std over all $(n, T)$ values.

    Args:
        x: A single-channel block $(n, T)$.

    Returns:
        The z-scored block $(n, T)$.
    """
    mu = float(x.mean())
    sd = float(x.std())
    return (x - mu) / (sd + _ZSCORE_EPS)


# ---------------------------------------------------------------------------
# S3-T02: coupled-channel slicing wrapper
# ---------------------------------------------------------------------------


def slice_coupled_channels(
    fields: Dict[str, np.ndarray], coupled: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Slice the coupled pulse-shape scattering channels into R0-probe arrays (S3-T02).

    The R0 probe :func:`analytic_te.realizable_te_block_from_arrays` conditions on the
    informative source columns ``U[:, :, :M]``; in v2 the single informative channel is
    **not** column 0 but the fs-correct pulse-shape channel identified by
    :meth:`scattering_adapter.ScatteringAdapter.coupled_channel_indices`. This wrapper
    cuts that one channel out of ``fhr_st`` (target $Y$) and ``up_st`` (source $U$) into
    the $(n, T, 1)$ single-channel arrays the probe expects (``M = 1``).

    The feature grid is already the trimmed ``latent[15:315]`` window (the adapter trims
    15 steps per end), so **no additional time alignment is applied here** — the returned
    arrays are on the same 300-step grid as the model's inputs.

    .. note::
        This conditions the target self-history $Y^{-}$ on **one** FHR channel, not all
        87 the model sees. The resulting $\mathrm{frac}_\Phi$ is therefore a
        *coupled-sub-process* realizability estimate — a diagnostic of how much coupling
        reaches the pulse-shape channel — not a tight bound on the full model's
        information (§14.3, S3-T02 acceptance).

    Args:
        fields: The model-facing feature dict with ``fhr_st`` $(n, T, 43)$ and ``up_st``
            $(n, T, 43)$ (as returned by ``transform_and_normalise``).
        coupled: The dict from ``coupled_channel_indices`` with integer ``fhr_st`` and
            ``up_st`` channel indices.

    Returns:
        ``(Y, U)`` — contiguous float64 arrays of shape $(n, T, 1)$: the coupled decel
        (``fhr_st``) target and contraction (``up_st``) source channels.
    """
    fi = int(coupled["fhr_st"])
    ui = int(coupled["up_st"])
    Y = np.ascontiguousarray(fields["fhr_st"][:, :, fi : fi + 1], dtype=float)
    U = np.ascontiguousarray(fields["up_st"][:, :, ui : ui + 1], dtype=float)
    return Y, U


# ---------------------------------------------------------------------------
# Shared R0 evaluation (anchors x seeds averaging)
# ---------------------------------------------------------------------------


def _r0_gain_over_anchors(
    Y: np.ndarray,
    U: np.ndarray,
    *,
    K: int,
    H: int,
    delay_max: int,
    ridge: float,
    n_anchors: int,
    n_seeds: int,
) -> Dict[str, Any]:
    r"""Average the R0 realizable block-TE gain over anchors and train/test splits.

    Runs :func:`analytic_te.realizable_te_block_from_arrays` at ``n_anchors`` anchor
    positions spread over the valid window $[K, T - H]$ and ``n_seeds`` train/test row
    shuffles, dropping ``ill_conditioned`` runs, and returns the mean gain (nats). The
    averaging tames the finite-sample variance of a single held-out determinant ratio.

    Args:
        Y: Target array $(n, T, 1)$.
        U: Source array $(n, T, 1)$.
        K: History depth $Y^{-}$/$U^{-}$ (the cell's ``K_history``).
        H: Forecast horizon.
        delay_max: Upper source lag scope (the cell's fixed lag $D$).
        ridge: Ridge penalty for the probe regressions.
        n_anchors: Number of anchor positions averaged.
        n_seeds: Number of train/test shuffles averaged.

    Returns:
        Dict with ``gain`` (mean block TE in nats, ``nan`` if every run is
        ill-conditioned), ``n_used``, ``n_total``, and ``ill_fraction``.
    """
    T = int(Y.shape[1])
    lo, hi = int(K), int(T - H)
    if hi <= lo:
        anchors = [lo]
    else:
        anchors = sorted(
            {int(round(a)) for a in np.linspace(lo, hi - 1, max(1, int(n_anchors)))}
        )
    gains: List[float] = []
    n_total = 0
    for anchor in anchors:
        for seed in range(max(1, int(n_seeds))):
            n_total += 1
            res = realizable_te_block_from_arrays(
                Y, U, M=1, K=int(K), H=int(H), delay_max=int(delay_max),
                anchor=int(anchor), ridge=float(ridge), seed=int(seed),
            )
            if res.get("ill_conditioned"):
                continue
            gains.append(float(res["realizable_gain"]))
    if not gains:
        return {"gain": float("nan"), "n_used": 0, "n_total": n_total,
                "ill_fraction": 1.0}
    return {
        "gain": float(np.mean(gains)),
        "n_used": len(gains),
        "n_total": n_total,
        "ill_fraction": float((n_total - len(gains)) / max(n_total, 1)),
    }


def _probe_knobs(config: Dict[str, Any], benchmark: str) -> Dict[str, Any]:
    r"""Extract the shared R0-probe knobs $(K, H, \text{ridge}, n_{\text{anchors}},
    n_{\text{seeds}}, \text{frac\_threshold})$ from config.

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.

    Returns:
        A dict with ``K``, ``H``, ``ridge``, ``n_anchors``, ``n_seeds``,
        ``frac_threshold`` (the lower bound) and ``frac_upper`` (the upper bound of the
        two-sided pass band; defaults to $\infty$ when unset, i.e. lower-bound-only).
    """
    bench = config["benchmarks"][benchmark]
    data = bench["data"]
    ev = bench["eval"]["realizability"]
    return {
        "K": int(data["K_history"]),
        "H": int(data["horizon"]),
        "ridge": float(ev["ridge"]),
        "n_anchors": int(ev.get("n_anchors", 3)),
        "n_seeds": int(ev.get("n_seeds", 2)),
        "frac_threshold": float(ev["frac_threshold"]),
        "frac_upper": float(ev.get("frac_upper", float("inf"))),
    }


# ---------------------------------------------------------------------------
# S3-T03: frac_Phi / TE_scat probe
# ---------------------------------------------------------------------------


def measure_te_scat(
    fields: Dict[str, np.ndarray],
    cell: CellV2,
    coupled: Dict[str, Any],
    *,
    config: Dict[str, Any],
    benchmark: str = "G1_raw",
) -> Dict[str, Any]:
    r"""Measure the scattering-realizable TE and preservation fraction (S3-T03).

    Slices the coupled pulse-shape channels (:func:`slice_coupled_channels`) and runs
    the ported R0 probe (ridge + 70/30 held-out, averaged over anchors and seeds) to
    estimate

    $$
    \mathrm{TE}_{\mathrm{scat}} =
        \widehat{\mathrm{TE}}^{(H),\,\mathrm{real}}_{U_{\mathrm{st}} \to Y_{\mathrm{st}}},
    \qquad
    \mathrm{frac}_\Phi = \frac{\mathrm{TE}_{\mathrm{scat}}}{\mathrm{TE}_{\mathrm{inj}}},
    $$

    where $\mathrm{TE}_{\mathrm{inj}} = $ ``cell.te_block_realised``. $\mathrm{frac}_\Phi
    \to 1$ means the coupling survives into the model-facing features; $\ll 1$ means the
    rendering/transform lost it (retune via :func:`sweep_render_knobs`).

    Args:
        fields: The model-facing feature dict (``fhr_st`` / ``up_st`` used).
        cell: The solved cell (supplies ``D`` and the $\mathrm{TE}_{\mathrm{inj}}$ label).
        coupled: The ``coupled_channel_indices`` dict.
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.

    Returns:
        Dict with ``te_scat`` (nats), ``frac_phi`` (``None`` for a null cell where
        $\mathrm{TE}_{\mathrm{inj}} = 0$), ``snr_per_step``, ``n_used``, ``ill_fraction``,
        ``frac_threshold`` (lower bound), ``frac_upper`` (upper bound), ``frac_over_one``
        (``frac_phi > frac_upper`` -- flags probe overestimation, not information gain), and
        ``passes`` (the **two-sided** ``frac_threshold <= frac_phi <= frac_upper``).

    .. note::
        $\mathrm{frac}_\Phi$ is an ordinal, coupled-sub-process realizability estimate. The
        R0 reduced model conditions the target future on a **single** coarse ``fhr_st``
        channel (:func:`slice_coupled_channels`), a weaker conditioning set than the true
        latent history, so it can over-attribute predictability to the source and return
        $\mathrm{frac}_\Phi > 1$. Such values are flagged (``frac_over_one``) and fail the
        two-sided gate rather than silently passing; cross-check ``te_raw`` and the null
        cells (§10, §14.3).
    """
    knobs = _probe_knobs(config, benchmark)
    Y, U = slice_coupled_channels(fields, coupled)
    r0 = _r0_gain_over_anchors(
        Y, U, K=knobs["K"], H=knobs["H"], delay_max=int(cell.D),
        ridge=knobs["ridge"], n_anchors=knobs["n_anchors"], n_seeds=knobs["n_seeds"],
    )
    te_scat = r0["gain"]
    te_inj = float(cell.te_block_realised)
    frac_phi: Optional[float]
    if te_inj > 0.0 and np.isfinite(te_scat):
        frac_phi = float(te_scat / te_inj)
    else:
        frac_phi = None
    snr = (float(snr_per_step_for_te_block(te_scat, knobs["H"], 1))
           if np.isfinite(te_scat) and te_scat > 0.0 else 0.0)
    frac_lo = knobs["frac_threshold"]
    frac_upper = knobs["frac_upper"]
    frac_over_one = bool(frac_phi is not None and frac_phi > frac_upper)
    return {
        "te_scat": float(te_scat) if np.isfinite(te_scat) else float("nan"),
        "frac_phi": frac_phi,
        "snr_per_step": snr,
        "n_used": int(r0["n_used"]),
        "ill_fraction": float(r0["ill_fraction"]),
        "frac_threshold": float(frac_lo),
        "frac_upper": float(frac_upper),
        "frac_over_one": frac_over_one,
        "passes": bool(frac_phi is not None and frac_lo <= frac_phi <= frac_upper),
    }


# ---------------------------------------------------------------------------
# S3-T04: TE_raw probe (raw-domain determinant ratio)
# ---------------------------------------------------------------------------


def _fourier_decimate(x: np.ndarray, n_out: int) -> np.ndarray:
    r"""Anti-aliased Fourier decimation of ``x`` along its last axis to ``n_out`` samples.

    Truncates the real FFT to the retained low-frequency bins and inverts at the shorter
    length (a ``scipy.signal.resample``-style band-limited downsample), so content above
    the new Nyquist is removed rather than aliased.

    Args:
        x: Array $(n, N)$.
        n_out: Output length ($< N$).

    Returns:
        The decimated array $(n, n_{\mathrm{out}})$.
    """
    x = np.asarray(x, dtype=float)
    N = int(x.shape[1])
    xf = np.fft.rfft(x, axis=1)
    keep = n_out // 2 + 1
    yf = xf[:, :keep] * (float(n_out) / float(N))
    return np.fft.irfft(yf, n=int(n_out), axis=1)


def _bandpass(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> np.ndarray:
    r"""Zero-phase FFT band-pass of ``x`` along its last axis to $[f_{lo}, f_{hi}]$ Hz.

    Args:
        x: Array $(n, N)$.
        fs: Sampling rate in Hz.
        f_lo: Lower band edge in Hz (inclusive).
        f_hi: Upper band edge in Hz (inclusive).

    Returns:
        The band-passed array $(n, N)$.
    """
    x = np.asarray(x, dtype=float)
    N = int(x.shape[1])
    freqs = np.fft.rfftfreq(N, d=1.0 / float(fs))
    xf = np.fft.rfft(x, axis=1)
    mask = (freqs >= float(f_lo)) & (freqs <= float(f_hi))
    xf[:, ~mask] = 0.0
    return np.fft.irfft(xf, n=N, axis=1)


def _hilbert_envelope(x: np.ndarray) -> np.ndarray:
    r"""Analytic-signal amplitude envelope $\lvert x + i\,\mathcal H\{x\}\rvert$ along axis 1.

    Demodulates a narrowband carrier: for $x(t) = A(t)\cos(2\pi f t + \varphi)$ the analytic
    envelope recovers the non-negative amplitude $A(t)$ (up to a constant), which is where
    the AM-rendered coupling lives (§7). This is the raw-domain, learned-filter-free analog
    of the scattering modulus $\lvert x \star \psi_\lambda\rvert$ — without it a *linear*
    block-TE probe sees only the sign/phase-carrying carrier and cannot extract the
    amplitude coupling. Uses the standard FFT construction (no SciPy dependency).

    Args:
        x: Real array $(n, N)$ (band-limited to the carrier first).

    Returns:
        The non-negative envelope $(n, N)$.
    """
    x = np.asarray(x, dtype=float)
    N = int(x.shape[1])
    xf = np.fft.fft(x, axis=1)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1.0
        h[1 : N // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (N + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(xf * h[None, :], axis=1))


def measure_te_raw(
    fhr_raw: np.ndarray,
    up_raw: np.ndarray,
    *,
    D: int,
    config: Dict[str, Any],
    benchmark: str = "G1_raw",
    demodulate: bool = True,
    decimation: Optional[int] = None,
    trim: Optional[int] = None,
) -> Dict[str, Any]:
    r"""Measure the raw-waveform block TE $\mathrm{TE}_{\mathrm{raw}}$ (S3-T04).

    The coupling is rendered as **amplitude modulation** on the pulse-shape carrier at
    $f_{\mathrm{pulse}}$ (§7), so the source→target information lives in the carrier's
    *amplitude envelope*, not in its sign/phase. This probe therefore band-passes both raw
    signals to the carrier band $[f_{\mathrm{pulse}}/1.6,\, f_{\mathrm{pulse}}\cdot 1.6]$,
    **demodulates** them to their analytic-signal envelopes (:func:`_hilbert_envelope` —
    the learned-filter-free analog of the scattering modulus), decimates to the analysis
    grid (Fourier decimation $5280 \to 330$, symmetric ``trim``/end $\to 300$), per-channel
    z-scores, and runs the same R0 determinant-ratio probe as $\mathrm{TE}_{\mathrm{scat}}$
    on the single FHR (target) and UP (source) envelope channels ($M = 1$, lags $0..D$):

    $$
    \mathrm{TE}_{\mathrm{raw}} =
        I\!\bigl(A_{\mathrm{FHR}}^{+}; A_{\mathrm{UP}}^{-} \mid A_{\mathrm{FHR}}^{-}\bigr).
    $$

    This is the physical raw-domain envelope TE — model-free (no scattering filter bank, no
    log/asinh, no learned masks), expected to *track* but not equal $\mathrm{TE}_{\mathrm{inj}}$
    (§8.3, §10) and useful for the $\mathrm{frac}_\Phi > 1$ anomaly check (§19). It is pure
    NumPy (no GPU). ``demodulate=False`` skips the Hilbert step and runs the probe on the
    band-passed carrier directly — a control that reads ~0 because a *linear* probe cannot
    extract amplitude coupling from a phase-carrying carrier (this is precisely why the
    scattering modulus is needed).

    Args:
        fhr_raw: FHR waveform(s) $(n, N)$.
        up_raw: UP waveform(s) $(n, N)$.
        D: The cell's fixed lag (source-lag scope for the probe).
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        demodulate: Extract the carrier-band amplitude envelope before the probe (default
            ``True``); ``False`` runs on the band-passed carrier directly (the ~0 control).
        decimation: Decimation factor; defaults to :data:`raw_generators.DECIMATION`.
            ``run_realizability_preflight`` passes the live adapter's ``T`` so the raw
            grid matches the scattering grid exactly.
        trim: Edge trim per end; defaults to :data:`_TRIM_STEPS`.

    Returns:
        Dict with ``te_raw`` (nats), ``snr_per_step``, ``n_used``, ``ill_fraction``.
    """
    fhr_raw = np.asarray(fhr_raw, dtype=float)
    up_raw = np.asarray(up_raw, dtype=float)
    if fhr_raw.ndim != 2 or up_raw.ndim != 2:
        raise ValueError("measure_te_raw: fhr_raw and up_raw must be (n, N).")
    knobs = _probe_knobs(config, benchmark)
    raw_cfg = config["benchmarks"][benchmark]["raw"]
    fs = float(raw_cfg["fs"])
    q = int(DECIMATION if decimation is None else decimation)
    trim = int(_TRIM_STEPS if trim is None else trim)
    n_dec = int(fhr_raw.shape[1] // q)

    # Isolate the carrier band where the AM-rendered coupling lives, then demodulate to the
    # amplitude envelope (the coupled quantity) before the linear block-TE probe.
    f_pulse = float(raw_cfg["f_pulse"])
    band = (f_pulse / _CARRIER_BAND_FACTOR, f_pulse * _CARRIER_BAND_FACTOR)
    fhr_b = _bandpass(fhr_raw, fs, *band)
    up_b = _bandpass(up_raw, fs, *band)
    if demodulate:
        fhr_b = _hilbert_envelope(fhr_b)
        up_b = _hilbert_envelope(up_b)

    fhr_dec = _fourier_decimate(fhr_b, n_dec)
    up_dec = _fourier_decimate(up_b, n_dec)
    if trim > 0:
        fhr_dec = fhr_dec[:, trim : n_dec - trim]
        up_dec = up_dec[:, trim : n_dec - trim]

    # Match the scattering *_st features: per-channel z-score before the R0 probe so its
    # Gram-scaled ridge is not inflated by the physiological DC (would shrink TE to ~0).
    Y = np.ascontiguousarray(_zscore_channel(fhr_dec)[:, :, None])
    U = np.ascontiguousarray(_zscore_channel(up_dec)[:, :, None])
    r0 = _r0_gain_over_anchors(
        Y, U, K=knobs["K"], H=knobs["H"], delay_max=int(D), ridge=knobs["ridge"],
        n_anchors=knobs["n_anchors"], n_seeds=knobs["n_seeds"],
    )
    te_raw = r0["gain"]
    snr = (float(snr_per_step_for_te_block(te_raw, knobs["H"], 1))
           if np.isfinite(te_raw) and te_raw > 0.0 else 0.0)
    return {
        "te_raw": float(te_raw) if np.isfinite(te_raw) else float("nan"),
        "snr_per_step": snr,
        "n_used": int(r0["n_used"]),
        "ill_fraction": float(r0["ill_fraction"]),
    }


# ---------------------------------------------------------------------------
# S3-T05: pre-flight harness over the pilot grid (fatal gate)
# ---------------------------------------------------------------------------


def _summarise_preflight(
    per_cell: Dict[int, Dict[str, Any]],
    *,
    frac_threshold: float,
    frac_upper: float = float("inf"),
) -> Dict[str, Any]:
    r"""Summarise the per-cell pre-flight table into pass/fail gate statistics.

    The gate is **two-sided**: a signal cell passes iff
    $\texttt{frac\_threshold} \le \mathrm{frac}_\Phi \le \texttt{frac\_upper}$. A cell with
    $\mathrm{frac}_\Phi > \texttt{frac\_upper}$ fails as ``over_one`` (probe overestimation
    from single-channel conditioning, §10/§14.3), not as under-preservation.

    Args:
        per_cell: The per-cell result dict keyed by ``cell_id``.
        frac_threshold: The $\mathrm{frac}_\Phi$ lower bound.
        frac_upper: The $\mathrm{frac}_\Phi$ upper bound ($\infty$ = lower-bound-only).

    Returns:
        A summary dict (``frac_threshold``, ``frac_upper``, ``n_signal_cells``,
        ``n_null_cells``, ``mean_frac_signal``, ``min_frac_signal``, ``max_frac_signal``,
        ``n_frac_over_one``, ``headline_pass``, ``failing_cell_ids``,
        ``over_one_cell_ids``).
    """
    signal = [c for c in per_cell.values() if float(c["te_inj"]) > 0.0]
    fracs = [float(c["frac_phi"]) for c in signal
             if c["frac_phi"] is not None and np.isfinite(c["frac_phi"])]

    def _valid(c: Dict[str, Any]) -> bool:
        return c["frac_phi"] is not None and np.isfinite(float(c["frac_phi"]))

    failing = sorted(
        int(c["cell_id"]) for c in signal
        if not (_valid(c) and frac_threshold <= float(c["frac_phi"]) <= frac_upper)
    )
    over_one = sorted(
        int(c["cell_id"]) for c in signal
        if _valid(c) and float(c["frac_phi"]) > frac_upper
    )
    return {
        "frac_threshold": float(frac_threshold),
        "frac_upper": float(frac_upper),
        "n_signal_cells": len(signal),
        "n_null_cells": len(per_cell) - len(signal),
        "mean_frac_signal": float(np.mean(fracs)) if fracs else None,
        "min_frac_signal": float(np.min(fracs)) if fracs else None,
        "max_frac_signal": float(np.max(fracs)) if fracs else None,
        "n_frac_over_one": len(over_one),
        "over_one_cell_ids": over_one,
        "headline_pass": len(failing) == 0 and len(signal) > 0,
        "failing_cell_ids": failing,
    }


def _print_preflight_table(result: Dict[str, Any]) -> None:
    r"""Print the pre-flight three-TE table to stdout.

    Args:
        result: The dict returned by :func:`run_realizability_preflight`.
    """
    grid = result["grid"]
    print(
        "[r0-realizability] pilot grid "
        f"target_te={grid['target_te_grid']} lag={grid['lag_grid']} "
        f"n_per_cell={grid['n_per_cell']}"
    )
    coupled = result["coupled_channel"]
    print(f"  coupled st channel = {coupled['fhr_st']}  "
          f"({coupled['hz']:.5f} Hz, xi={coupled['xi']:.5f})")
    s = result["summary"]
    frac_lo = s["frac_threshold"]
    frac_hi = s.get("frac_upper", float("inf"))
    print(f"  {'cell':>4} {'te_inj':>7} {'D':>3} {'te_raw':>8} {'te_scat':>8} "
          f"{'frac_phi':>9} {'snr/st':>7} {'note':>6}")
    for cid in sorted(result["per_cell"]):
        c = result["per_cell"][cid]
        frac = "   null" if c["frac_phi"] is None else f"{c['frac_phi']:9.3f}"
        if c["frac_phi"] is None:
            note = ""
        elif float(c["frac_phi"]) > frac_hi:
            note = "HIGH"            # probe overestimation (frac_Phi > upper bound)
        elif float(c["frac_phi"]) >= frac_lo:
            note = "PASS"
        else:
            note = "LOW"
        print(f"  {cid:>4} {c['te_inj']:7.3f} {c['D']:>3} {c['te_raw']:8.3f} "
              f"{c['te_scat']:8.3f} {frac} {c['snr_per_step']:7.3f} {note:>6}")
    mean_frac = "n/a" if s["mean_frac_signal"] is None else f"{s['mean_frac_signal']:.3f}"
    band = f"[{frac_lo:.2f}, {frac_hi:.2f}]" if np.isfinite(frac_hi) else f">= {frac_lo:.2f}"
    print(f"  summary: mean frac_phi(signal)={mean_frac}  pass-band={band}  "
          f"n_frac_over_one={s.get('n_frac_over_one', 0)}  "
          f"headline_pass={s['headline_pass']}  failing={s['failing_cell_ids']}")


def run_realizability_preflight(
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw",
    pilot: bool = True,
    out_dir: Optional[Path] = None,
    n_per_cell: Optional[int] = None,
    adapter: Optional[Any] = None,
    print_table: bool = True,
) -> Dict[str, Any]:
    r"""Run the three-TE realizability pre-flight over the pilot (or full) grid (S3-T05).

    For each enumerated cell this generates raw pairs in memory
    (:func:`generate_pilot_samples`), runs the real scattering transform + normalisation
    (:class:`scattering_adapter.ScatteringAdapter`), and measures $\mathrm{TE}_{\mathrm{inj}}$
    (the cell label), $\mathrm{TE}_{\mathrm{raw}}$, and $\mathrm{TE}_{\mathrm{scat}}$ with
    $\mathrm{frac}_\Phi$. It writes ``realizability.json`` and honours
    ``eval.realizability.fatal``: when fatal and any signal cell falls below
    ``frac_threshold`` it raises, else it only warns. This is the gate that decides
    whether v2 is worth training (§14.3).

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        pilot: Use the ``eval.realizability.pilot`` grid when ``True``, else the full
            ``mix`` grid.
        out_dir: Directory to write ``realizability.json`` into (skipped when ``None``).
        n_per_cell: Override the grid's per-cell sample count.
        adapter: A pre-built :class:`ScatteringAdapter` to reuse (built here when
            ``None``).
        print_table: Print the stdout table when ``True``.

    Returns:
        A dict with ``summary``, ``per_cell`` (keyed by ``cell_id``), ``dropped``,
        ``coupled_channel``, and ``grid``.

    Raises:
        RuntimeError: If ``eval.realizability.fatal`` is set and the headline gate fails.
    """
    bench = config["benchmarks"][benchmark]
    ev = bench["eval"]["realizability"]
    if pilot:
        pcfg = ev["pilot"]
        te_grid = list(pcfg["target_te_grid"])
        lag_grid = list(pcfg["lag_grid"])
        n = int(n_per_cell if n_per_cell is not None else pcfg["n_per_cell"])
    else:
        mix = bench["mix"]
        te_grid = list(mix["target_te_grid"])
        lag_grid = list(mix["lag_grid"])
        n = int(n_per_cell if n_per_cell is not None else mix["n_per_cell_train"])

    cells, dropped = enumerate_cells_v2(
        config, benchmark=benchmark, target_te_grid=te_grid, lag_grid=lag_grid
    )
    if adapter is None:
        from .scattering_adapter import ScatteringAdapter
        adapter = ScatteringAdapter(config, benchmark=benchmark)
    coupled = adapter.coupled_channel_indices()

    per_cell: Dict[int, Dict[str, Any]] = {}
    for cell in cells:
        raw = generate_pilot_samples(cell, n, "train", config, benchmark=benchmark)
        fields, _ = adapter.transform_and_normalise(raw["fhr_raw"], raw["up_raw"])
        scat = measure_te_scat(fields, cell, coupled, config=config, benchmark=benchmark)
        traw = measure_te_raw(
            raw["fhr_raw"], raw["up_raw"], D=cell.D, config=config, benchmark=benchmark,
            decimation=adapter.T, trim=adapter.trim,
        )
        per_cell[cell.cell_id] = {
            "cell_id": int(cell.cell_id),
            "target_te": float(cell.target_te),
            "D": int(cell.D),
            "B_y_scalar": float(cell.B_y_scalar),
            "te_inj": float(cell.te_block_realised),
            "te_raw": float(traw["te_raw"]),
            "te_scat": float(scat["te_scat"]),
            "frac_phi": scat["frac_phi"],
            "frac_over_one": bool(scat["frac_over_one"]),
            "snr_per_step": float(scat["snr_per_step"]),
            "n": int(n),
        }

    summary = _summarise_preflight(
        per_cell,
        frac_threshold=float(ev["frac_threshold"]),
        frac_upper=float(ev.get("frac_upper", float("inf"))),
    )
    result = {
        "summary": summary,
        "per_cell": per_cell,
        "dropped": dropped,
        "coupled_channel": dict(coupled),
        "grid": {"target_te_grid": te_grid, "lag_grid": lag_grid, "n_per_cell": n},
    }
    if print_table:
        _print_preflight_table(result)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "realizability.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(result), handle, indent=2)
        logger.info("run_realizability_preflight: wrote %s", path)

    # Gate on actual failing cells, not on ``headline_pass`` (which is also False for a
    # signal-free grid, e.g. a null-only smoke run) so the fatal gate never raises with an
    # empty failing list.
    if bool(ev.get("fatal", False)) and summary["failing_cell_ids"]:
        band = (f"[{summary['frac_threshold']}, {summary['frac_upper']}]"
                if np.isfinite(summary["frac_upper"]) else f">= {summary['frac_threshold']}")
        raise RuntimeError(
            "run_realizability_preflight: frac_Phi gate FAILED (fatal=true); "
            f"failing cells {summary['failing_cell_ids']} outside pass-band {band} "
            f"(over-one cells {summary['over_one_cell_ids']} indicate probe bias, not "
            "under-preservation). Retune via sweep_render_knobs (S3-T06) or set "
            "eval.realizability.fatal=false."
        )
    return result


# ---------------------------------------------------------------------------
# S3-T06: frac_Phi recovery / tuning sweep
# ---------------------------------------------------------------------------


def _override_render_config(
    config: Dict[str, Any],
    benchmark: str,
    *,
    f_pulse: float,
    am_offset_ratio: float,
    omega: float,
) -> Dict[str, Any]:
    r"""Deep-copy ``config`` and override the three render knobs for a sweep point.

    Args:
        config: The parsed config tree.
        benchmark: Active benchmark key.
        f_pulse: New ``raw.f_pulse`` (Hz).
        am_offset_ratio: New ``raw.am_offset_ratio``.
        omega: New source pole angle $\omega$ (``data.oscillators[0][1]``).

    Returns:
        A modified deep copy of ``config``.
    """
    cfg = copy.deepcopy(config)
    bench = cfg["benchmarks"][benchmark]
    r = float(bench["data"]["oscillators"][0][0])
    bench["data"]["oscillators"] = [[r, float(omega)]]
    bench["raw"]["f_pulse"] = float(f_pulse)
    bench["raw"]["am_offset_ratio"] = float(am_offset_ratio)
    return cfg


def sweep_render_knobs(
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw",
    out_dir: Optional[Path] = None,
    adapter: Optional[Any] = None,
    print_table: bool = True,
) -> Dict[str, Any]:
    r"""Sweep the render knobs to recover a high-$\mathrm{frac}_\Phi$ setting (S3-T06).

    Crosses ``eval.realizability.recovery.{f_pulse_grid, am_offset_ratio_grid,
    omega_grid}`` and, for each combination, computes the analytic AM-separation margin
    (:func:`raw_generators.am_separation_margin`, cheap), a real
    :func:`measure_te_scat` on a strong cell (generate $\to$ transform $\to$ slice $\to$
    probe), and the lag-identifiability re-check
    :func:`raw_generators.ar2_lag1_autocorr`. The scattering transform is
    ``f_pulse``/``am_offset_ratio``/$\omega$-**independent**, so a single adapter is
    reused; only the coupled-channel index (an argmin over ``center_freqs``) and the raw
    generation depend on the knobs. The chosen setting maximises $\mathrm{frac}_\Phi$
    subject to ``margin_peak >= 1`` and ``lag1_autocorr >= lag1_autocorr_floor``. The
    result is written to ``recovery.json`` — the manifest field that records the chosen
    render knobs (config locking is deferred to S4-T01).

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        out_dir: Directory to write ``recovery.json`` into (skipped when ``None``).
        adapter: A pre-built adapter to reuse (built here when ``None``).
        print_table: Print the stdout table when ``True``.

    Returns:
        A dict with ``table`` (per-setting rows) and ``chosen`` (the selected setting,
        or ``None`` if the sweep is empty).
    """
    bench = config["benchmarks"][benchmark]
    rec = bench["eval"]["realizability"]["recovery"]
    f_grid = list(rec["f_pulse_grid"])
    ratio_grid = list(rec["am_offset_ratio_grid"])
    omega_grid = list(rec["omega_grid"])
    n = int(rec["n_per_cell"])
    target_te = float(rec["target_te"])
    D = int(rec["D"])
    floor = float(rec["lag1_autocorr_floor"])

    r = float(bench["data"]["oscillators"][0][0])
    Q = int(bench["scattering"]["Q"])
    fs = float(bench["raw"]["fs"])
    sigma2_eta = float(bench["data"]["sigma2_eta"])

    if adapter is None:
        from .scattering_adapter import ScatteringAdapter
        adapter = ScatteringAdapter(config, benchmark=benchmark)

    # The injected-TE label depends on omega (the DGP), not on f_pulse / ratio, so the
    # inverter Monte-Carlo is memoised per omega (2 solves, not 3x3x2).
    solved_by_omega: Dict[float, CellV2] = {}
    table: List[Dict[str, Any]] = []
    for f_pulse in f_grid:
        for ratio in ratio_grid:
            for omega in omega_grid:
                cfg = _override_render_config(
                    config, benchmark, f_pulse=f_pulse, am_offset_ratio=ratio, omega=omega
                )
                cell = solved_by_omega.get(omega)
                if cell is None:
                    sol = solve_cell_coupling(cfg, target_te, D, benchmark=benchmark)
                    cell = CellV2(cell_id=0, target_te=target_te, D=D,
                                  B_y_scalar=float(sol["B_y_scalar"]),
                                  te_block_realised=float(sol["te_block"]))
                    solved_by_omega[omega] = cell

                margin = am_separation_margin(
                    r=r, w=float(omega), f_pulse=float(f_pulse), Q=Q, fs=fs,
                    am_offset_ratio=float(ratio), sigma2_eta=sigma2_eta,
                )
                lag1 = float(ar2_lag1_autocorr(r, float(omega)))
                row: Dict[str, Any] = {
                    "f_pulse": float(f_pulse),
                    "am_offset_ratio": float(ratio),
                    "omega": float(omega),
                    "te_inj": float(cell.te_block_realised),
                    "margin_peak": float(margin["margin_peak"]),
                    "preservation": float(margin["preservation"]),
                    "lag1_autocorr": lag1,
                }
                try:
                    # coupled_channel_indices raises if no scattering channel lands
                    # within one Q-step of this f_pulse; skip + flag that setting rather
                    # than aborting the whole sweep and discarding earlier rows.
                    coupled = adapter.coupled_channel_indices(f_pulse=float(f_pulse))
                    raw = generate_pilot_samples(cell, n, "train", cfg, benchmark=benchmark)
                    fields, _ = adapter.transform_and_normalise(
                        raw["fhr_raw"], raw["up_raw"]
                    )
                    scat = measure_te_scat(
                        fields, cell, coupled, config=cfg, benchmark=benchmark
                    )
                    row.update(
                        te_scat=float(scat["te_scat"]),
                        frac_phi=scat["frac_phi"],
                        coupled_hz=float(coupled["hz"]),
                        valid=bool(margin["margin_peak"] >= 1.0 and lag1 >= floor),
                        error=None,
                    )
                except Exception as exc:  # noqa: BLE001 -- record + skip any bad setting
                    logger.warning(
                        "sweep_render_knobs: skipping f_pulse=%g ratio=%g omega=%g: %s",
                        f_pulse, ratio, omega, exc,
                    )
                    row.update(te_scat=float("nan"), frac_phi=None,
                               coupled_hz=float("nan"), valid=False, error=str(exc))
                table.append(row)

    chosen = _pick_best_setting(table)
    result = {"table": table, "chosen": chosen,
              "constraints": {"margin_peak_min": 1.0, "lag1_autocorr_floor": floor},
              "strong_cell": {"target_te": target_te, "D": D, "n_per_cell": n}}
    if print_table:
        _print_recovery_table(result)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "recovery.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(result), handle, indent=2)
        logger.info("sweep_render_knobs: wrote %s", path)
    return result


def _pick_best_setting(table: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    r"""Pick the sweep row whose $\mathrm{frac}_\Phi$ is **closest to 1**, preferring valid rows.

    The scientific objective is *faithful* preservation, $\mathrm{frac}_\Phi \to 1$, so
    that a clean end-to-end calibration ($\gamma_{\mathrm{scat}} \to 1$ *and*
    $\mathrm{frac}_\Phi \approx 1 \Rightarrow \gamma_{\mathrm{inj}} \to 1$) is possible
    (§10, §14.1). A value $\gg 1$ is treated as estimator / $\phi$-smoothing inflation,
    **not** created information (§10), so maximising $\mathrm{frac}_\Phi$ would recommend a
    dubious setting; we minimise $\lvert \mathrm{frac}_\Phi - 1 \rvert$ instead.

    A row is *valid* when it satisfies both the AM-separation (``margin_peak >= 1``) and
    lag-identifiability (``lag1_autocorr >= floor``) constraints. Valid rows win; if none
    is valid the closest-to-1 overall row is returned (with ``valid=False``) so the caller
    can flag the compromise.

    Args:
        table: The per-setting sweep rows.

    Returns:
        The chosen row (a shallow copy) or ``None`` when ``table`` is empty.
    """
    rows = [row for row in table
            if row["frac_phi"] is not None and np.isfinite(row["frac_phi"])]
    if not rows:
        return None
    valid = [row for row in rows if row["valid"]]
    pool = valid if valid else rows
    best = min(pool, key=lambda row: abs(float(row["frac_phi"]) - 1.0))
    return dict(best)


def _print_recovery_table(result: Dict[str, Any]) -> None:
    r"""Print the recovery sweep table + chosen setting to stdout.

    Args:
        result: The dict returned by :func:`sweep_render_knobs`.
    """
    print("[recover] render-knob sweep (frac_Phi vs f_pulse / am_offset_ratio / omega)")
    print(f"  {'f_pulse':>8} {'ratio':>6} {'omega':>6} {'te_scat':>8} {'frac_phi':>9} "
          f"{'margin':>7} {'lag1':>6} {'valid':>6}")
    for row in result["table"]:
        frac = "     nan" if row["frac_phi"] is None else f"{row['frac_phi']:9.3f}"
        print(f"  {row['f_pulse']:8.3f} {row['am_offset_ratio']:6.2f} {row['omega']:6.3f} "
              f"{row['te_scat']:8.3f} {frac} {row['margin_peak']:7.2f} "
              f"{row['lag1_autocorr']:6.3f} {str(row['valid']):>6}")
    chosen = result["chosen"]
    if chosen is None:
        print("  chosen: NONE (no finite frac_Phi in the sweep)")
    else:
        print(f"  chosen: f_pulse={chosen['f_pulse']:g} "
              f"am_offset_ratio={chosen['am_offset_ratio']:g} omega={chosen['omega']:g} "
              f"-> frac_phi={chosen['frac_phi']:.3f} "
              f"(margin_peak={chosen['margin_peak']:.2f}, "
              f"lag1={chosen['lag1_autocorr']:.3f}, valid={chosen['valid']})")


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    r"""Recursively convert NumPy scalars / arrays to JSON-native types.

    Args:
        obj: Any nested dict / list / scalar structure.

    Returns:
        The structure with NumPy scalars cast to ``float``/``int`` and arrays to lists.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ===========================================================================
# Sprint 6: evaluation gates on a trained checkpoint.
#
# These grade the *trained model* against the known ground truth: the per-step
# KL surrogate $\bar K$ (calibration vs $\mathrm{TE}_{\mathrm{inj}}$ and
# $\mathrm{TE}_{\mathrm{scat}}$), the lag attention (recovery of the true lag
# band $\mathcal L^\star$), and the null-control collapse. The model is used
# only through its ``forward`` contract (``kld_per_t`` / ``te_lag_map`` /
# ``attn_weights``); nothing in the model, loss, or trainer changes. torch and
# the loader/model stack are imported lazily so the pure-NumPy probes above stay
# importable without them. See EXPLAINED §13-§14 and SPRINTS Sprint 6.
# ===========================================================================


def _run_results_dir(config: Dict[str, Any], benchmark: str,
                     arm: Optional[str] = None) -> Path:
    r"""Resolve the ``results/<tag>/<arm>/`` directory (local copy of the driver helper).

    Duplicated from ``run_pipeline_v2._run_dir`` so :func:`run_eval` does not import the
    driver (which imports this module) and create a cycle. ``arm`` scopes the run root to
    ``results/<tag>/<arm>/`` for the ``synthetic_v3`` arms; ``arm=None`` reproduces the
    arm-less ``results/<tag>/`` exactly.

    Args:
        config: The parsed config tree.
        benchmark: Active benchmark key (fallback tag).
        arm: The resolved arm name, or ``None`` for the arm-less run root.

    Returns:
        The run root directory as an absolute :class:`Path` (not created).
    """
    tag = str(config.get("experiment", {}).get("tag", benchmark))
    results_dir = Path(config.get("paths", {}).get("results_dir", "./results"))
    if not results_dir.is_absolute():
        results_dir = Path(__file__).resolve().parent / results_dir
    base = results_dir / tag
    return base / str(arm) if arm else base


# ---------------------------------------------------------------------------
# S6-T01: per-sample K-bar collection over the clean window
# ---------------------------------------------------------------------------


def _clean_window_mean(kld_bt, delay, *, warmup: int, horizon: int):
    r"""Average ``kld_per_t`` over each sample's clean window (EXPLAINED §13).

    The clean window excludes the encoder warm-up and the last horizon, and additionally
    the lead-in a sample's own delayed coupling needs to become identifiable:

    $$
    \mathcal W_i = \bigl[\max(w,\ D_i-1),\ T-H\bigr),
    $$

    with $w$ the warm-up, $H$ the horizon and $D_i$ the sample's fixed source$\to$target
    lag. The per-sample surrogate is $\bar K_i = \operatorname{mean}_{t\in\mathcal W_i}
    K_{i,t}$ where $K_{i,t} = $ ``kld_per_t`` (the nat-scale total KL per step).

    Args:
        kld_bt: The model's ``kld_per_t`` of shape $(B, T)$.
        delay: The per-sample lag $D_i$ of shape $(B,)$.
        warmup: The model warm-up period $w$.
        horizon: The forecast horizon $H$.

    Returns:
        A tuple ``(kbar, valid)``: the $(B,)$ per-sample clean-window mean and the
        $(B, T)$ boolean clean-window mask (reused to aggregate ``te_lag_map``).
    """
    import torch

    T = int(kld_bt.shape[1])
    device = kld_bt.device
    t_idx = torch.arange(T, device=device).unsqueeze(0)                 # (1, T)
    lo = torch.clamp(delay.to(device).long() - 1, min=int(warmup)).unsqueeze(1)  # (B, 1)
    hi = int(T - horizon)
    valid = (t_idx >= lo) & (t_idx < hi)                                # (B, T) bool
    valid_f = valid.to(kld_bt.dtype)
    denom = valid_f.sum(dim=1).clamp(min=1.0)
    kbar = (kld_bt * valid_f).sum(dim=1) / denom
    return kbar, valid


def _masked_forecast_loss(mu, y_plus, valid):
    r"""Per-sample clean-window MSE of a forecast mean against the unfolded future target.

    $$\mathcal{L}_b \;=\; \frac{1}{|\mathcal{T}_b|\, H_d\, C_y}
      \sum_{t \in \mathcal{T}_b} \sum_{h=1}^{H_d} \sum_{c=1}^{C_y}
      \bigl(\mu_{b,t,h,c} - Y^+_{b,t,h,c}\bigr)^2,$$

    where $\mathcal{T}_b$ is sample $b$'s clean window from :func:`_clean_window_mean`, namely
    $[\max(\text{warmup},\, D_b - 1),\; T - H)$. The normalisation matches the model's own
    ``compute_loss`` MSE, so the returned value is on the same scale as ``feat_loss`` /
    ``base_loss`` in ``per_sample_eval.npz``.

    This is the **single source of truth** for the forecast loss. ``collect_per_sample_kbar``
    scores the clean and the corrupted forwards with it, and ``lag_intervention_v3`` scores
    the lag-band-masked forwards with it, so
    $\Delta L_G = \mathcal{L}^{\text{masked }G} - \mathcal{L}_{\mathrm{feat}}$ is a difference
    of two identically-normalised quantities rather than of two independent re-derivations.

    Args:
        mu: Forecast mean $(B, T, H_d, C_y)$; only its first $T - H_d$ anchors are scored.
        y_plus: Unfolded future target $(B, T - H_d, H_d, C_y)$, i.e. the one-step-shifted
            ``Y[:, 1:].unfold(1, H_d, 1)``. Both ``eval_v2`` and
            ``TestRunner.build_future_target`` produce exactly this.
        valid: $(B, T)$ boolean clean-window mask from :func:`_clean_window_mean`.

    Returns:
        $(B,)$ tensor of per-sample losses.
    """
    t_valid = int(y_plus.shape[1])
    hd, c_y = int(y_plus.shape[2]), int(y_plus.shape[3])
    m = valid[:, :t_valid].to(mu.dtype)                      # (B, T_valid)
    w = m[:, :, None, None]
    cnt = m.sum(dim=1).clamp(min=1.0) * float(hd * c_y)      # (B,)
    return (((mu[:, :t_valid] - y_plus) ** 2) * w).sum(dim=(1, 2, 3)) / cnt


def _corrupt_source(u_stream, mode: str, *, generator=None):
    r"""Corrupt the source stream $U$ for a null control (EXPLAINED §14.4).

    Two label-free destructions of the true $U\to Y$ coupling that leave $U$'s marginal
    statistics intact:

    * ``shuffle`` -- a cross-batch permutation of the source rows, so each target is paired
      with **another** sample's source (destroys the per-sample coupling). A singleton
      batch cannot be shuffled, so it is returned unchanged.
    * ``reverse`` -- a time-reversal of the source stream (destroys the causal lag).

    Warning:
        It is tempting to expect $\bar K$ to collapse to the intercept $\alpha$ under either
        corruption, giving ``null_ratio`` $\to 0$. It does not, and cannot:
        $\mathrm{KL}(q\,\|\,p)$ measures that the source moved the posterior, not that it
        moved it *correctly*, and a corrupted source is out of distribution for a posterior
        trained only on matched pairs -- so it typically moves the belief **more** (v3 Finding
        F2). What *does* collapse is the corrupted forecast's accuracy, which is why these
        forwards are also scored in prediction space (:func:`prediction_controls`).

    Args:
        u_stream: The source tensor $(B, T, C_u)$.
        mode: ``"shuffle"`` or ``"reverse"``.
        generator: A CPU :class:`torch.Generator` for the reproducible shuffle permutation.

    Returns:
        The corrupted source tensor of the same shape.

    Raises:
        ValueError: On an unknown ``mode``.
    """
    import torch

    if mode == "reverse":
        return torch.flip(u_stream, dims=[1])
    if mode == "shuffle":
        B = u_stream.shape[0]
        if B < 2:
            return u_stream
        perm = torch.randperm(B, generator=generator).to(u_stream.device)
        if torch.equal(perm, torch.arange(B, device=u_stream.device)):
            perm = torch.roll(perm, 1)          # avoid the identity (self-pairing)
        return u_stream[perm]
    raise ValueError(f"unknown null control mode: {mode!r}")


# The per-sample KLD *summary* family (§14.5). Every entry is a model-free scalar
# reduction of the ``forward`` KLD tensors over each sample's clean window
# $\mathcal W_i$, giving several "flavours" of the surrogate $\bar K$ to correlate
# against the transfer entropy. They fall in four groups:
#
#   * time-summaries of the total per-step KL $K_t$ = ``kld_per_t`` over $\mathcal W_i$:
#     ``kbar`` (mean, the canonical surrogate), ``kbar_sum`` (integrated nats),
#     ``kbar_max`` (peak coupling), ``kbar_median`` (robust centre), ``kbar_p90``
#     (upper tail);
#   * window variants (means of $K_t$ over a different support): ``kbar_full`` (all
#     $T$ steps) and ``kbar_postwarm`` ($[w, T-H)$, warm-up excluded but WITHOUT the
#     per-sample delay floor) -- isolate the effect of the averaging window;
#   * directed-KL split of ``te_lag_map`` over the true lag band
#     $\mathcal L^\star = \{\max(0, D-H), \dots, D-1\}$: ``kbar_inband`` (the KL the
#     attention routes to the *correct* source lags) and ``kbar_outband`` (the rest;
#     ``kbar_inband + kbar_outband == kbar`` because $\sum_\ell$ ``te_lag_map`` $= K_t$).
#
# The per-head split ``kld_per_t_per_head`` is stored separately as ``kbar_head``
# $(N, \text{num\_heads})$ and expanded to ``kbar_head0 .. kbar_head{M-1}`` (which of
# the contiguous latent groups carries the coupling). Keep this list, the
# ``_write_per_sample_eval`` payload, :func:`fit_calibration`'s ``kld_variants`` block,
# and ``visualize_v2`` in sync.
KLD_SCALAR_VARIANTS: Tuple[str, ...] = (
    "kbar", "kbar_sum", "kbar_max", "kbar_median", "kbar_p90",
    "kbar_full", "kbar_postwarm", "kbar_inband", "kbar_outband",
)

#: Summaries that average $K_t$ over steps the model was never trained to shape, and are
#: therefore **not** admissible as a TE surrogate when ``kld_support == "anchor"`` (S4-T01).
#: ``kbar_full`` spans the warm-up prefix and the untrained final-$H$ tail. It is still
#: collected and written to ``per_sample_eval.npz`` -- it is evidence about what the untrained
#: region does -- but it is flagged ``out_of_support`` and excluded from best-variant
#: selection. ``kbar_postwarm`` is the exact anchor support and is the right comparator.
_OUT_OF_SUPPORT_UNDER_ANCHOR: Tuple[str, ...] = ("kbar_full",)


def _row_window_reductions(x_bt: np.ndarray, valid_bt: np.ndarray) -> Dict[str, np.ndarray]:
    r"""Per-row reductions of ``x_bt`` over each sample's boolean clean window (§14.5).

    Reduces the $(B, T)$ per-step KL to several per-sample scalars, each over that row's
    valid clean window $\mathcal W_i$ (the same mask :func:`_clean_window_mean` averages):
    the integrated ``sum`` ($\sum_{t\in\mathcal W} K_t$, total nats), the ``max`` (peak
    coupling), the robust ``median``, and the upper-tail ``p90``. A row with an empty
    window gets ``nan``.

    Args:
        x_bt: Per-step values $(B, T)$ (e.g. ``kld_per_t``).
        valid_bt: Boolean clean-window mask $(B, T)$.

    Returns:
        A dict of $(B,)$ ``float64`` arrays keyed ``sum`` / ``max`` / ``median`` / ``p90``.
    """
    B = int(x_bt.shape[0])
    out = {k: np.full(B, np.nan, dtype=np.float64)
           for k in ("sum", "max", "median", "p90")}
    for i in range(B):
        w = x_bt[i][valid_bt[i]]
        if w.size == 0:
            continue
        out["sum"][i] = float(w.sum())
        out["max"][i] = float(w.max())
        out["median"][i] = float(np.median(w))
        out["p90"][i] = float(np.percentile(w, 90.0))
    return out


def collect_per_sample_kbar(
    model,
    loader,
    device,
    *,
    warmup: int,
    horizon: int,
    controls: Sequence[str] = (),
    control_seed: int = 0,
) -> Dict[str, Any]:
    r"""Collect the per-sample surrogate $\bar K$ over the clean window (S6-T01).

    One evaluation pass over ``loader``. Per batch it runs the model ``forward``, averages
    ``kld_per_t`` over each sample's clean window (:func:`_clean_window_mean`), and stamps
    the per-sample provenance (``te_inj``/``te_scat``/``frac_phi``/``cell_id``/``delay``/
    ``held_out``). For each requested null control it re-runs ``forward`` with a corrupted
    source (:func:`_corrupt_source`) and records $\bar K_{\mathrm{null}}$ **and** the
    corrupted forecast loss. In the **same pass** it accumulates, per ``cell_id``, the
    clean-window mean of the model's ``te_lag_map`` (the KLD-weighted per-lag attribution)
    into a lag profile, so the S6-T03 lag recovery needs no second forward.

    v2 drops the v1 ``M`` / ``band_id`` grouping and the ``per_dim_kl_by_M`` /
    ``kld_time_by_band`` structures (single pathway, fixed lag); grouping is by
    ``cell_id`` with the window floor from the per-sample ``delay``.

    Warning:
        ``feat_loss_<control>`` scores an **input-stream** corruption: the source stream $U$
        is time-reversed or row-permuted *before* the encoder, so the whole source pathway
        sees garbage. It is a different object from the training-time ``feat_loss_perm``
        logged by ``pl_module_v2``, which deranges the already-encoded ``source_state`` along
        the batch axis and leaves the encoders untouched. The two corruptions have different
        magnitudes and must never be tabulated as comparable.

    Args:
        model: A loaded :class:`SeqVaeLagAttnV1` in ``eval`` mode.
        loader: A ``DataLoader`` yielding batched :class:`dataset_v2.AttributeDict` items.
        device: The device to run the forward on.
        warmup: The model warm-up period $w$.
        horizon: The forecast horizon $H$.
        controls: Null controls to also evaluate (subset of ``{"shuffle", "reverse"}``).
        control_seed: Seed for the reproducible shuffle permutation.

    Returns:
        A dict of length-$N$ arrays (``kbar``, ``te_inj``, ``te_scat``, ``frac_phi``,
        ``cell_id``, ``delay``, ``held_out``, ``feat_loss``, ``base_loss``, ``pred_gain``,
        ``uplift_rel``, and per control ``kbar_<control>`` / ``feat_loss_<control>`` /
        ``shuffle_penalty_<control>``), the KLD summary family
        (:data:`KLD_SCALAR_VARIANTS`: ``kbar_sum`` / ``kbar_max`` / ``kbar_median`` /
        ``kbar_p90`` / ``kbar_full`` / ``kbar_postwarm`` / ``kbar_inband`` /
        ``kbar_outband``) plus per-head ``kbar_head`` $(N, \text{num\_heads})$ and its
        expanded ``kbar_head0 .. kbar_head{M-1}`` columns, plus ``lag_profiles``
        (``{cell_id: (L,) float64}``), ``kbar_over_time``, ``lag_counts``, ``n`` and ``T``.

        ``feat_loss`` and ``base_loss`` are the clean-window MSEs
        $\mathcal L_{\mathrm{feat}}$ / $\mathcal L_{\mathrm{base}}$ whose difference is
        ``pred_gain``; ``shuffle_penalty_<control>`` is
        $\mathcal L_{\mathrm{feat}}^{\text{corrupted}} - \mathcal L_{\mathrm{feat}}$, which
        must be **positive** on a model that genuinely exploits the source.
    """
    import torch

    from .dataset_v2 import build_u_stream

    model.eval()
    gen = torch.Generator()
    gen.manual_seed(int(control_seed))

    cols: Dict[str, List[np.ndarray]] = {
        "kbar": [], "te_inj": [], "te_scat": [], "frac_phi": [],
        "cell_id": [], "delay": [], "held_out": [],
        "feat_loss": [], "base_loss": [], "pred_gain": [], "uplift_rel": [],
        # KLD summary family (§14.5): time-summaries, window variants, directed-KL split.
        "kbar_sum": [], "kbar_max": [], "kbar_median": [], "kbar_p90": [],
        "kbar_full": [], "kbar_postwarm": [], "kbar_inband": [], "kbar_outband": [],
    }
    head_chunks: List[np.ndarray] = []   # per-batch (B, num_heads) clean-window per-head KL
    control_cols: Dict[str, List[np.ndarray]] = {c: [] for c in controls}
    # Prediction-space control columns (S3-T02b), keyed by their FULL name rather than
    # sharing the ``kbar_`` prefix, so ``_write_per_sample_eval``'s explicit key list and
    # the ``kbar_head*`` glob never collide with them.
    pred_loss_cols: Dict[str, List[np.ndarray]] = {}
    for _c in controls:
        pred_loss_cols[f"feat_loss_{_c}"] = []
        pred_loss_cols[f"shuffle_penalty_{_c}"] = []
    lag_sum: Dict[int, np.ndarray] = {}
    lag_cnt: Dict[int, int] = {}
    kbar_t_sum: Dict[int, np.ndarray] = {}   # cell_id -> (T,) sum of per-step kld
    T_seen = 0

    def _prov_or(batch, name, default, dtype, size):
        r"""Read a per-sample provenance array (``(B,)``), or a filled default if absent."""
        val = batch.get(name, None) if hasattr(batch, "get") else None
        if val is None:
            return np.full(int(size), default, dtype=dtype)
        arr = val.detach().cpu().numpy() if torch.is_tensor(val) else np.asarray(val)
        return np.asarray(arr).astype(dtype)

    with torch.no_grad():
        for batch in loader:
            y_st = batch.fhr_st.to(device)
            y_ph = batch.fhr_ph.to(device)
            u_stream = build_u_stream(batch).to(device)

            out = model(y_st, y_ph, u_stream)
            kld_bt = out["kld_per_t"]                                   # (B, T)
            bsz, T_seen = int(kld_bt.shape[0]), int(kld_bt.shape[1])

            # Per-sample provenance (already CPU scalars after collate).
            delay_np = _prov_or(batch, "delay", 0, np.int64, bsz)
            delay_t = torch.as_tensor(delay_np).to(device)
            kbar, valid = _clean_window_mean(
                kld_bt, delay_t, warmup=warmup, horizon=horizon
            )
            cols["kbar"].append(kbar.detach().cpu().numpy())

            # --- KLD summary family over the clean window (§14.5) --------------------
            # One pass of reductions on the per-step KL; reused below for the per-cell
            # kbar-over-time profile so ``kld_per_t`` is moved to host memory once.
            vmask = valid.to(kld_bt.dtype)                                # (B, T)
            vdenom = vmask.sum(dim=1).clamp(min=1.0)                      # (B,)
            kld_np = kld_bt.detach().cpu().numpy().astype(np.float64)     # (B, T)
            valid_np = valid.detach().cpu().numpy()                       # (B, T) bool
            red = _row_window_reductions(kld_np, valid_np)
            cols["kbar_sum"].append(red["sum"])
            cols["kbar_max"].append(red["max"])
            cols["kbar_median"].append(red["median"])
            cols["kbar_p90"].append(red["p90"])
            # Window variants: full-sequence mean and post-warm-up mean (no delay floor).
            cols["kbar_full"].append(kld_np.mean(axis=1))
            hi_pw = int(kld_np.shape[1] - horizon)
            if hi_pw > int(warmup):
                cols["kbar_postwarm"].append(kld_np[:, int(warmup):hi_pw].mean(axis=1))
            else:
                cols["kbar_postwarm"].append(np.full(bsz, np.nan, dtype=np.float64))

            # Directed-KL split over the true lag band L* = {max(0,D-H)..D-1}: the KL the
            # attention routes to the correct source lags (in-band) vs the rest (out-band).
            # te_lag_map sums to kld_per_t over lags, so in+out == kbar exactly.
            te_map = out.get("te_lag_map", None)
            if te_map is not None:
                Lc = int(te_map.shape[-1])
                l_idx = torch.arange(Lc, device=device).unsqueeze(0)       # (1, L)
                lo_b = torch.clamp(delay_t.long() - int(horizon), min=0).unsqueeze(1)
                hi_b = delay_t.long().unsqueeze(1)                         # (B, 1)
                band = ((l_idx >= lo_b) & (l_idx < hi_b)).to(te_map.dtype)  # (B, L)
                inband_t = (te_map * band.unsqueeze(1)).sum(dim=-1)         # (B, T)
                kbar_inb = (inband_t * vmask).sum(dim=1) / vdenom          # (B,)
                cols["kbar_inband"].append(kbar_inb.detach().cpu().numpy())
                cols["kbar_outband"].append((kbar - kbar_inb).detach().cpu().numpy())
            else:
                cols["kbar_inband"].append(np.full(bsz, np.nan, dtype=np.float64))
                cols["kbar_outband"].append(np.full(bsz, np.nan, dtype=np.float64))

            # Per-head KL split: clean-window mean of each contiguous latent group's KL.
            kph = out.get("kld_per_t_per_head", None)
            if kph is not None:
                head_kbar = (kph * vmask.unsqueeze(-1)).sum(dim=1) / vdenom.unsqueeze(-1)
                head_chunks.append(head_kbar.detach().cpu().numpy())

            cid = _prov_or(batch, "cell_id", 0, np.int64, bsz)
            cols["cell_id"].append(cid)
            cols["delay"].append(delay_np)
            cols["te_inj"].append(_prov_or(batch, "te_true", np.nan, np.float64, bsz))
            cols["te_scat"].append(_prov_or(batch, "te_scat", np.nan, np.float64, bsz))
            cols["frac_phi"].append(_prov_or(batch, "frac_phi", np.nan, np.float64, bsz))
            cols["held_out"].append(_prov_or(batch, "held_out", 0, np.int64, bsz))

            # Prediction gain (forecast "uplift"): how much the full model's forecast
            # beats the FHR-only baseline forecast, averaged over the SAME clean window
            # as $\bar K$. This is the model-free-of-KL second axis of evidence -- a large
            # $\Delta L$ where $\bar K$ is also large confirms the source is genuinely
            # used, not just encoded. Reuses the forecast heads already in ``out``
            # (mu_full = mu_base + delta_mu_src); the future target is the one-step-shifted
            # unfold of $Y = [y_{st}\,\|\,y_{ph}]$, matching ``compute_loss`` (MSE).
            #
            # The future target and its clean-window weights are built ONCE per batch and
            # reused by the corrupted forwards below (S3-T02b), so scoring a control costs
            # no extra unfold and no extra forward.
            mu_full = out.get("mu_full")
            mu_base = out.get("mu_base")
            nan_b = torch.full((bsz,), float("nan"), device=device)
            have_target = False
            Y_plus = None
            T_valid = 0
            if mu_full is not None and mu_base is not None:
                Hd = int(mu_full.shape[2])
                T_valid = int(y_st.shape[1]) - Hd
                if T_valid > 0:
                    have_target = True
                    Y = torch.cat([y_st, y_ph], dim=-1)              # (B, T, C_y)
                    Y_plus = Y[:, 1:, :].unfold(1, Hd, 1).permute(0, 1, 3, 2)  # (B,Tv,Hd,C)

            def _forecast_loss(mu):
                r"""Clean-window MSE of a forecast mean against the shared future target."""
                return _masked_forecast_loss(mu, Y_plus, valid)

            if have_target:
                l_full = _forecast_loss(mu_full)
                l_base = _forecast_loss(mu_base)
                pred_gain = l_base - l_full
                uplift_rel = pred_gain / l_base.clamp(min=1e-12)
            else:
                l_full = l_base = nan_b
                pred_gain = nan_b
                uplift_rel = nan_b
            # S3-T02a: the absolute per-sample forecast losses, not just their difference.
            # ``pred_gain == base_loss - feat_loss`` remains exact by construction.
            cols["feat_loss"].append(l_full.detach().cpu().numpy())
            cols["base_loss"].append(l_base.detach().cpu().numpy())
            cols["pred_gain"].append(pred_gain.detach().cpu().numpy())
            cols["uplift_rel"].append(uplift_rel.detach().cpu().numpy())

            # Per-cell lag profile: clean-window mean of te_lag_map, summed per cell.
            lag_map = out.get("te_lag_map", None)
            if lag_map is None:
                lag_map = out["attn_weights"].mean(dim=2)              # (B, T, L) fallback
            valid_f = valid.to(lag_map.dtype).unsqueeze(-1)            # (B, T, 1)
            denom = valid_f.sum(dim=1).clamp(min=1.0)                  # (B, 1)
            prof = (lag_map * valid_f).sum(dim=1) / denom             # (B, L)
            prof_np = prof.detach().cpu().numpy().astype(np.float64)
            L = prof_np.shape[1]
            # Per-cell per-step KL trajectory (the K-bar-over-time profile, before the
            # clean-window collapse), summed per cell alongside the lag profile. ``kld_np``
            # was already moved to host memory above for the KLD summary family.
            for i in range(prof_np.shape[0]):
                c = int(cid[i])
                if c not in lag_sum:
                    lag_sum[c] = np.zeros(L, dtype=np.float64)
                    lag_cnt[c] = 0
                    kbar_t_sum[c] = np.zeros(int(kld_np.shape[1]), dtype=np.float64)
                lag_sum[c] += prof_np[i]
                kbar_t_sum[c] += kld_np[i]
                lag_cnt[c] += 1

            # Null controls: exactly ONE corrupted forward per control per batch. Both the
            # KL-space readout (``kbar_<ctrl>``) and the prediction-space gate
            # (``feat_loss_<ctrl>``, S3-T02b) are scored from that single forward, against
            # the same ``Y_plus`` and the same clean-window weights built above.
            for ctrl in controls:
                u_bad = _corrupt_source(u_stream, ctrl, generator=gen)
                out_c = model(y_st, y_ph, u_bad)
                kbar_c, _ = _clean_window_mean(
                    out_c["kld_per_t"], delay_t, warmup=warmup, horizon=horizon
                )
                control_cols[ctrl].append(kbar_c.detach().cpu().numpy())

                mu_full_c = out_c.get("mu_full")
                if have_target and mu_full_c is not None:
                    l_full_c = _forecast_loss(mu_full_c)
                else:
                    l_full_c = nan_b
                pred_loss_cols[f"feat_loss_{ctrl}"].append(
                    l_full_c.detach().cpu().numpy())
                pred_loss_cols[f"shuffle_penalty_{ctrl}"].append(
                    (l_full_c - l_full).detach().cpu().numpy())

    result: Dict[str, Any] = {k: np.concatenate(v) for k, v in cols.items()}
    if head_chunks:
        head_arr = np.concatenate(head_chunks, axis=0)      # (N, num_heads)
        result["kbar_head"] = head_arr
        for m in range(int(head_arr.shape[1])):
            result[f"kbar_head{m}"] = head_arr[:, m]
    for ctrl, chunks in control_cols.items():
        result[f"kbar_{ctrl}"] = np.concatenate(chunks) if chunks else np.zeros(0)
    for name, chunks in pred_loss_cols.items():
        result[name] = np.concatenate(chunks) if chunks else np.zeros(0)
    result["lag_profiles"] = {c: lag_sum[c] / max(1, lag_cnt[c]) for c in lag_sum}
    result["kbar_over_time"] = {c: kbar_t_sum[c] / max(1, lag_cnt[c]) for c in kbar_t_sum}
    result["lag_counts"] = {c: int(lag_cnt[c]) for c in lag_cnt}
    result["n"] = int(result["kbar"].shape[0])
    result["T"] = T_seen
    return result


# ---------------------------------------------------------------------------
# S6-T02: gamma-calibration of K-bar against TE_inj and TE_scat
# ---------------------------------------------------------------------------


def fit_calibration_slope(points: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    r"""Ordinary-least-squares fit of $\bar K = \alpha + \gamma\,\mathrm{TE}$ (S6-T02).

    $\gamma = \operatorname{Cov}(x,y)/\operatorname{Var}(x)$,
    $\alpha = \bar y - \gamma\bar x$, and $R^2 = 1 - \mathrm{SS_{res}}/\mathrm{SS_{tot}}$
    (v1 ``calibration.fit_calibration_slope``). A calibrated surrogate has
    $\gamma \to 1$.

    Args:
        points: An iterable of $(\mathrm{TE}, \bar K)$ pairs (typically per-cell means).

    Returns:
        A dict with ``alpha``, ``gamma``, ``r2`` and ``n``.

    Raises:
        ValueError: With fewer than two finite points, or when all $\mathrm{TE}$ coincide.
    """
    pts = [(float(t), float(k)) for t, k in points
           if np.isfinite(t) and np.isfinite(k)]
    if len(pts) < 2:
        raise ValueError("need >= 2 finite points for a calibration slope")
    x = np.array([p[0] for p in pts], dtype=np.float64)
    y = np.array([p[1] for p in pts], dtype=np.float64)
    if np.allclose(x, x[0]):
        raise ValueError("all TE values coincide; slope undefined")
    gamma = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    alpha = float(y.mean() - gamma * x.mean())
    yhat = alpha + gamma * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    return {"alpha": alpha, "gamma": gamma, "r2": r2, "n": len(pts)}


def _nanmean_safe(x: np.ndarray) -> float:
    r"""``nanmean`` that returns ``nan`` for an empty / all-NaN slice without warning."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).any():
        return float("nan")
    return float(np.nanmean(x[np.isfinite(x)]))


def _group_per_cell(arrs: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    r"""Reduce the per-sample arrays to per-cell means keyed by ``cell_id``.

    Args:
        arrs: The dict returned by :func:`collect_per_sample_kbar`.

    Returns:
        ``{cell_id: {cell_id, te_inj, te_scat, kbar, delay, frac_phi, held_out, n}}``.
    """
    cid = np.asarray(arrs["cell_id"])
    out: Dict[int, Dict[str, Any]] = {}
    for c in np.unique(cid):
        sel = cid == c
        out[int(c)] = {
            "cell_id": int(c),
            "te_inj": _nanmean_safe(np.asarray(arrs["te_inj"])[sel]),
            "te_scat": _nanmean_safe(np.asarray(arrs["te_scat"])[sel]),
            "kbar": _nanmean_safe(np.asarray(arrs["kbar"])[sel]),
            "delay": int(np.round(np.mean(np.asarray(arrs["delay"])[sel]))),
            "frac_phi": _nanmean_safe(np.asarray(arrs["frac_phi"])[sel])
            if "frac_phi" in arrs else float("nan"),
            "pred_gain": _nanmean_safe(np.asarray(arrs["pred_gain"])[sel])
            if "pred_gain" in arrs else float("nan"),
            "uplift_rel": _nanmean_safe(np.asarray(arrs["uplift_rel"])[sel])
            if "uplift_rel" in arrs else float("nan"),
            "held_out": int(np.round(np.mean(np.asarray(arrs["held_out"])[sel])))
            if "held_out" in arrs else 0,
            "n": int(np.sum(sel)),
        }
    return out


def _spearman_sign(points: Sequence[Tuple[float, float]]) -> Optional[float]:
    r"""Rank (Spearman) correlation between $\mathrm{TE}$ and $\bar K$ across cells.

    Args:
        points: An iterable of $(\mathrm{TE}, \bar K)$ pairs.

    Returns:
        The Spearman $\rho$ (positive $\Rightarrow \bar K$ increases with TE), or ``None``
        when it is undefined (fewer than two points, or a constant rank).
    """
    pts = [(float(t), float(k)) for t, k in points
           if np.isfinite(t) and np.isfinite(k)]
    if len(pts) < 2:
        return None
    t = np.array([p[0] for p in pts]); k = np.array([p[1] for p in pts])
    rt = np.argsort(np.argsort(t)).astype(np.float64)
    rk = np.argsort(np.argsort(k)).astype(np.float64)
    if np.std(rt) == 0 or np.std(rk) == 0:
        return None
    return float(np.corrcoef(rt, rk)[0, 1])


def _rank_average(a: np.ndarray) -> np.ndarray:
    r"""Tie-aware (midrank) ranks: tied values share the mean of their ordinal ranks.

    Matches ``scipy.stats.rankdata(method='average')`` without the dependency. Needed
    because the per-sample TE axis has few distinct levels with many ties, which biases
    a plain ``argsort(argsort(.))`` rank.

    Args:
        a: A 1-D array.

    Returns:
        The average ranks of ``a``.
    """
    a = np.asarray(a, dtype=np.float64)
    order = a.argsort(kind="mergesort")
    ordinal = np.empty(a.size, dtype=np.float64)
    ordinal[order] = np.arange(1, a.size + 1, dtype=np.float64)
    uniq, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(uniq.size, dtype=np.float64)
    np.add.at(sums, inv, ordinal)
    return (sums / cnt)[inv]


def _pearson_finite(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    r"""Pearson correlation over the finite $(x, y)$ pairs (``None`` when undefined)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return None
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _spearman_finite(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    r"""Tie-aware Spearman rank correlation over the finite $(x, y)$ pairs (``None`` if undefined)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2:
        return None
    rx, ry = _rank_average(x[m]), _rank_average(y[m])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


#: Bootstrap resamples for the null-cell confidence interval (S4-T02).
_N_BOOT: int = 2000

#: Default gate on $\bar K$ over the $\mathrm{TE}_{\mathrm{inj}} = 0$ cells, in nats.
_KBAR_NULL_THRESHOLD: float = 0.05


def _bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = _N_BOOT,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    r"""Percentile bootstrap CI of the mean of ``values``.

    Resamples ``values`` with replacement ``n_boot`` times and takes the empirical quantiles
    of the resampled means. Non-finite entries are dropped first. There is no closed-form
    alternative here: the null cells number ~3, so a normal-approximation interval on a
    3-point mean would be a fiction.

    Args:
        values: 1-D array of per-cell means.
        n_boot: Number of bootstrap resamples.
        ci: Central coverage of the returned interval (``0.95`` -> 2.5%/97.5% quantiles).
        rng: Seeded generator; a fresh default one is used when omitted (so callers that
            need reproducibility must pass their own).

    Returns:
        ``(mean, ci_lo, ci_hi)``; all three are ``nan`` when nothing finite remains.
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float(x[0]), float(x[0])
    gen = rng if rng is not None else np.random.default_rng()
    idx = gen.integers(0, x.size, size=(int(n_boot), x.size))
    means = x[idx].mean(axis=1)
    lo_q = (1.0 - float(ci)) / 2.0
    lo, hi = np.quantile(means, [lo_q, 1.0 - lo_q])
    return float(x.mean()), float(lo), float(hi)


def fit_calibration(
    arrs: Dict[str, Any],
    *,
    kld_support: str = "full",
    kbar_null_threshold: float = _KBAR_NULL_THRESHOLD,
    boot_seed: int = 0,
) -> Dict[str, Any]:
    r"""Fit $\bar K = \alpha + \gamma\,\mathrm{TE}$ vs both $\mathrm{TE}_{\mathrm{inj}}$
    and $\mathrm{TE}_{\mathrm{scat}}$ (S6-T02).

    Reports both slopes/intercepts/$R^2$ and a Spearman-sign monotonicity flag. Null cells
    ($\mathrm{TE}_{\mathrm{inj}} = 0$) are kept as the near-origin anchor. A clean result is
    $\gamma_{\mathrm{scat}} \to 1$ (§14.1); reporting against both TEs makes any
    $\mathrm{frac}_\Phi$ inflation visible.

    Two v3-specific readings are added on top (Sprint 4).

    **Out-of-support summaries (S4-T01).** Under ``kld_support == "anchor"`` the model is only
    trained to shape $K_t$ on the anchor-aligned support, so ``kbar_full`` -- which averages
    ``kld_per_t`` over **all** $T$ -- is partly measuring a warm-up prefix and an untrained
    final-$H$ tail. It stays in the table (it is evidence) but is stamped
    ``out_of_support: True`` so no consumer can crown it best-variant. ``kbar_postwarm`` is
    the correct anchor-support comparator.

    **Null-cell intercept (S4-T02).** v3 initialises with $K \equiv 0$, so the intercept
    $\alpha$ no longer absorbs a random log-variance-head floor. That makes
    $\bar K \big|_{\mathrm{TE}_{\mathrm{inj}} = 0} \to 0$ a claim the model can actually be
    held to -- one v1 structurally could not satisfy -- and a calibrated surrogate additionally
    has $\alpha \approx \bar K \big|_{\mathrm{TE} = 0}$.

    Args:
        arrs: The dict returned by :func:`collect_per_sample_kbar`.
        kld_support: The evaluated model's ``kld_support`` (``"full"`` / ``"anchor"``). Read
            with ``getattr(model, "kld_support", "full")`` so a v1 alias never raises.
        kbar_null_threshold: Gate on the mean $\bar K$ over the null cells, in nats.
        boot_seed: Seed for the null-cell bootstrap, so the CI is reproducible.

    Returns:
        A dict with the per-cell ``gamma_inj``/``gamma_scat``/``alpha_*``/``r2_*``,
        ``spearman_*``, ``monotonic_*``, ``n_cells`` and a ``per_cell`` table, plus the
        full-$N$ pooled per-sample fit (``gamma_inj_sample``/``gamma_scat_sample``/
        ``alpha_*_sample``/``r2_*_sample``, ``n_samples``), a ``by_lag`` table of
        per-lag per-sample slopes (``{D: {gamma_inj, alpha_inj, r2_inj, gamma_scat, ...,
        n}}``), a ``kld_variants`` table of per-KLD-summary calibrations
        (``{variant: {gamma_inj, r2_inj, pearson_inj, spearman_inj, gamma_scat, ...,
        out_of_support, n}}``; §14.5) covering every :data:`KLD_SCALAR_VARIANTS` entry and
        per-head column present, and ``kbar_at_null_cells``
        (``{mean, std, ci_lo, ci_hi, n_cells, threshold, pass}``, or ``None`` when the grid
        has no ``te_inj == 0`` cell).
    """
    per_cell = _group_per_cell(arrs)
    cells = list(per_cell.values())
    inj_points = [(c["te_inj"], c["kbar"]) for c in cells]
    scat_points = [(c["te_scat"], c["kbar"]) for c in cells if np.isfinite(c["te_scat"])]

    def _safe_fit(points):
        try:
            return fit_calibration_slope(points)
        except ValueError:
            return None

    fit_inj = _safe_fit(inj_points)
    fit_scat = _safe_fit(scat_points)
    rho_inj = _spearman_sign(inj_points)
    rho_scat = _spearman_sign(scat_points)

    # --- pooled per-SAMPLE calibration (the full-N "overall average behaviour") ----------
    # The per-cell fit above uses only ~15 points; the pooled per-sample fit uses every
    # evaluated sample, so the reported slope reflects the whole distribution (and the
    # per-sample scatter figure regresses on the same points). te_inj / te_scat are
    # per-cell-constant, so x still spans the discrete TE levels -> the slope is defined.
    kbar_a = np.asarray(arrs["kbar"], dtype=np.float64)
    te_inj_a = np.asarray(arrs["te_inj"], dtype=np.float64)
    te_scat_a = np.asarray(arrs["te_scat"], dtype=np.float64)
    fit_inj_s = _safe_fit(list(zip(te_inj_a, kbar_a)))
    fit_scat_s = _safe_fit(list(zip(te_scat_a, kbar_a)))

    # --- per-lag calibration (one fit per lag D over its per-sample points) ---------------
    delay_a = np.asarray(arrs.get("delay", np.zeros_like(kbar_a)))
    by_lag: Dict[int, Dict[str, Any]] = {}
    for d in np.unique(delay_a[np.isfinite(delay_a.astype(np.float64))]) \
            if delay_a.size else []:
        sel = delay_a == d
        f_i = _safe_fit(list(zip(te_inj_a[sel], kbar_a[sel])))
        f_s = _safe_fit(list(zip(te_scat_a[sel], kbar_a[sel])))
        by_lag[int(d)] = {
            "gamma_inj": f_i["gamma"] if f_i else None,
            "alpha_inj": f_i["alpha"] if f_i else None,
            "r2_inj": f_i["r2"] if f_i else None,
            "gamma_scat": f_s["gamma"] if f_s else None,
            "alpha_scat": f_s["alpha"] if f_s else None,
            "r2_scat": f_s["r2"] if f_s else None,
            "n": int(np.sum(sel)),
        }

    # --- per-variant calibration (§14.5): every KLD summary vs TE_inj / TE_scat ----------
    # Each variant is a per-sample scalar (an alternative flavour of $\bar K$); we report
    # its pooled per-sample slope $\gamma$, $R^2$, Pearson $r$ and (tie-aware) Spearman
    # $\rho$ against both TEs, so the report/figures can rank which KLD summary tracks TE
    # best. Includes ``kbar`` itself and the per-head columns present in ``arrs``.
    variant_names = [v for v in KLD_SCALAR_VARIANTS if v in arrs]
    variant_names += sorted(
        k for k in arrs
        if str(k).startswith("kbar_head") and np.asarray(arrs.get(k)).ndim == 1
    )
    kld_variants: Dict[str, Any] = {}
    for v in variant_names:
        y = np.asarray(arrs[v], dtype=np.float64)
        if y.shape != kbar_a.shape:
            continue
        entry: Dict[str, Any] = {
            "n": int(np.isfinite(y).sum()),
            # S4-T01: ``kbar_full`` averages over the whole sequence, so under anchor-aligned
            # KL support it partly measures a region the model was never trained to shape.
            # Kept, flagged, and excluded from best-variant selection.
            "out_of_support": bool(v in _OUT_OF_SUPPORT_UNDER_ANCHOR
                                   and str(kld_support) == "anchor"),
        }
        for pref, te in (("inj", te_inj_a), ("scat", te_scat_a)):
            f = _safe_fit(list(zip(te, y)))
            entry[f"gamma_{pref}"] = f["gamma"] if f else None
            entry[f"alpha_{pref}"] = f["alpha"] if f else None
            entry[f"r2_{pref}"] = f["r2"] if f else None
            entry[f"pearson_{pref}"] = _pearson_finite(te, y)
            entry[f"spearman_{pref}"] = _spearman_finite(te, y)
        kld_variants[v] = entry

    # --- null-cell intercept gate (S4-T02) ------------------------------------------------
    null_kbars = np.asarray([c["kbar"] for c in cells if float(c["te_inj"]) == 0.0],
                            dtype=np.float64)
    if null_kbars.size:
        mean_null, ci_lo, ci_hi = _bootstrap_ci(
            null_kbars, rng=np.random.default_rng(int(boot_seed)))
        kbar_at_null_cells: Optional[Dict[str, Any]] = {
            "mean": mean_null,
            "std": float(np.std(null_kbars)) if null_kbars.size > 1 else 0.0,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_cells": int(null_kbars.size),
            "n_boot": int(_N_BOOT),
            "threshold": float(kbar_null_threshold),
            "pass": bool(np.isfinite(mean_null) and mean_null < float(kbar_null_threshold)),
        }
    else:
        kbar_at_null_cells = None

    return {
        "kld_support": str(kld_support),
        "kbar_at_null_cells": kbar_at_null_cells,
        "gamma_inj": fit_inj["gamma"] if fit_inj else None,
        "alpha_inj": fit_inj["alpha"] if fit_inj else None,
        "r2_inj": fit_inj["r2"] if fit_inj else None,
        "gamma_scat": fit_scat["gamma"] if fit_scat else None,
        "alpha_scat": fit_scat["alpha"] if fit_scat else None,
        "r2_scat": fit_scat["r2"] if fit_scat else None,
        "spearman_inj": rho_inj,
        "spearman_scat": rho_scat,
        "monotonic_inj": bool(rho_inj is not None and rho_inj > 0),
        "monotonic_scat": bool(rho_scat is not None and rho_scat > 0),
        "n_cells": len(cells),
        # Pooled per-sample fit + per-lag table (Enhancement A/D): full-N calibration.
        "n_samples": int(kbar_a.size),
        "gamma_inj_sample": fit_inj_s["gamma"] if fit_inj_s else None,
        "alpha_inj_sample": fit_inj_s["alpha"] if fit_inj_s else None,
        "r2_inj_sample": fit_inj_s["r2"] if fit_inj_s else None,
        "gamma_scat_sample": fit_scat_s["gamma"] if fit_scat_s else None,
        "alpha_scat_sample": fit_scat_s["alpha"] if fit_scat_s else None,
        "r2_scat_sample": fit_scat_s["r2"] if fit_scat_s else None,
        "by_lag": by_lag,
        # Per-variant KLD-summary calibration (§14.5): {variant: {gamma_*, r2_*,
        # pearson_*, spearman_*, n}} vs both TE_inj and TE_scat.
        "kld_variants": kld_variants,
        "per_cell": [
            {k: c[k] for k in ("cell_id", "te_inj", "te_scat", "kbar", "delay", "n", "frac_phi")}
            for c in cells
        ],
    }


# ---------------------------------------------------------------------------
# S6-T03: lag recovery from the attention lag profile
# ---------------------------------------------------------------------------


def _true_lag_band(D: int, horizon: int) -> List[int]:
    r"""The true past-source lag band $\mathcal L^\star = \{\max(0, D-H),\dots,D-1\}$.

    Future step $d_{k+\tau}$ ($\tau=1,\dots,H$) is driven by $c_{k+\tau-D}$, i.e. the source
    at lag $D-\tau$; sweeping $\tau$ gives this band (EXPLAINED §6.2/§14.2).

    Args:
        D: The source$\to$target lag (decimated steps).
        horizon: The forecast horizon $H$.

    Returns:
        The sorted list of in-band lag indices.
    """
    return list(range(max(0, int(D) - int(horizon)), int(D)))


def score_lag_profile(
    profile: "Sequence[float] | np.ndarray",
    true_band: Sequence[int],
    *,
    tolerance: int = 1,
) -> Dict[str, Any]:
    r"""Score one lag profile against the true band $\mathcal L^\star$ (S6-T03).

    Normalises the $(L,)$ profile to a distribution, sums the mass inside $\mathcal
    L^\star$ (**LagMass**), locates the argmax lag, and measures its distance to the
    nearest in-band lag with a $\pm$ ``tolerance`` step allowance (the $4\,\mathrm s$
    $\phi$-averaging blur, §14.2).

    Args:
        profile: The per-lag attribution of shape $(L,)$ (KLD-weighted ``te_lag_map``).
        true_band: The in-band lag indices $\mathcal L^\star$.
        tolerance: The $\pm$ step tolerance for the argmax match.

    Returns:
        A dict with ``lag_mass``, ``peak_lag``, ``peak_lag_err`` and ``within_tol``.
    """
    a = np.asarray(profile, dtype=np.float64).ravel()
    a = np.where(np.isfinite(a), a, 0.0)
    total = a.sum()
    norm = a / total if total > 0 else np.zeros_like(a)
    band = [int(b) for b in true_band if 0 <= int(b) < a.size]
    lag_mass = float(norm[band].sum()) if band else 0.0
    peak = int(np.argmax(a)) if a.size and a.max() > 0 else -1
    peak_err = int(min(abs(peak - b) for b in band)) if (band and peak >= 0) else -1
    within = bool(0 <= peak_err <= int(tolerance))
    return {
        "lag_mass": lag_mass,
        "peak_lag": peak,
        "peak_lag_err": peak_err,
        "within_tol": within,
    }


def recover_lags(
    lag_profiles: Dict[int, np.ndarray],
    cells_by_id: Dict[int, Dict[str, Any]],
    *,
    horizon: int,
    tolerance: int = 1,
    threshold: float = 0.8,
) -> Dict[str, Any]:
    r"""Score attention lag recovery per cell (S6-T03).

    For each cell builds $\mathcal L^\star$ from its lag $D$ and horizon $H$, scores the
    accumulated ``te_lag_map`` profile (:func:`score_lag_profile`), and aggregates over the
    **signal** cells ($\mathrm{TE}_{\mathrm{inj}} > 0$).

    Args:
        lag_profiles: ``{cell_id: (L,)}`` from :func:`collect_per_sample_kbar`.
        cells_by_id: ``{cell_id: {delay, te_inj, ...}}`` (from :func:`_group_per_cell`).
        horizon: The forecast horizon $H$.
        tolerance: The $\pm$ step tolerance for the argmax match.
        threshold: The LagMass pass threshold (``eval.lag_mass_threshold``).

    Returns:
        A dict with a per-cell table plus ``mean_lag_mass``, ``frac_within_tol`` and the
        ``mean_lag_mass_pass`` gate.
    """
    per_cell: Dict[int, Dict[str, Any]] = {}
    masses: List[float] = []
    within: List[bool] = []
    for cid, cell in cells_by_id.items():
        D = int(cell["delay"])
        band = _true_lag_band(D, horizon)
        is_null = bool(cell.get("te_inj", 0.0) <= 0.0)
        prof = lag_profiles.get(cid)
        if prof is None:
            per_cell[cid] = {
                "D": D, "true_band": band, "is_null": is_null,
                "lag_mass": None, "peak_lag": None, "peak_lag_err": None,
                "within_tol": None,
            }
            continue
        s = score_lag_profile(prof, band, tolerance=tolerance)
        per_cell[cid] = {"D": D, "true_band": band, "is_null": is_null, **s}
        if not is_null:
            masses.append(s["lag_mass"])
            within.append(s["within_tol"])
    mean_lag_mass = float(np.mean(masses)) if masses else None
    frac_within = float(np.mean(within)) if within else None
    return {
        "per_cell": per_cell,
        "mean_lag_mass": mean_lag_mass,
        "frac_within_tol": frac_within,
        "lag_mass_threshold": float(threshold),
        "tolerance": int(tolerance),
        "mean_lag_mass_pass": bool(
            mean_lag_mass is not None and mean_lag_mass >= threshold
        ),
    }


# ---------------------------------------------------------------------------
# S6-T04: null controls, metrics.json, and the minimal report
# ---------------------------------------------------------------------------


#: Caption for the demoted KL-space null ratio, carried in ``metrics.json`` so no downstream
#: consumer can read it as a "must reach zero" gate. See :func:`null_ratios`.
NULL_RATIO_NOTE: str = (
    "Readout only, NOT a gate: KL(q||p) measures 'the source moved my belief', not "
    "'...correctly'. A deranged source is still a source, and a posterior trained on matched "
    "pairs reacts to it out of distribution -- typically more strongly. Expect a finite value "
    "near 1.0 even on a model that demonstrably exploits the source (v3 Finding F2). The gate "
    "that does discriminate is metrics['prediction_controls']."
)


def null_ratios(arrs: Dict[str, Any], controls: Sequence[str]) -> Dict[str, Any]:
    r"""Per-cell and overall null ratios $\bar K_{\mathrm{null}} / \bar K_{\mathrm{signal}}$.

    For each control the ratio is computed per cell and averaged over the **signal** cells
    ($\mathrm{TE}_{\mathrm{inj}} > 0$).

    Warning:
        This is a **readout, not a gate** (S3-T03). Earlier revisions of this pipeline gated
        on $\bar K_{\mathrm{null}} / \bar K_{\mathrm{signal}} \to 0$, which no honest model can
        satisfy: $\mathrm{KL}(q \,\|\, p)$ measures that the source moved the posterior, not
        that it moved it *correctly*. v3 measured the ratio in $[1.02, 1.10]$ on a model that
        was demonstrably exploiting the source (Finding F2). Each returned control therefore
        carries ``expected_to_vanish: False`` and a ``note``. The discriminating gate is
        :func:`prediction_controls`.

    Args:
        arrs: The dict from :func:`collect_per_sample_kbar` (needs ``kbar_<control>``).
        controls: The evaluated controls.

    Returns:
        ``{control: {mean_ratio, per_cell, expected_to_vanish, note}}`` for every control
        present in ``arrs``.
    """
    cid = np.asarray(arrs["cell_id"])
    te_inj = np.asarray(arrs["te_inj"])
    kbar = np.asarray(arrs["kbar"])
    out: Dict[str, Any] = {}
    for ctrl in controls:
        key = f"kbar_{ctrl}"
        if key not in arrs or np.asarray(arrs[key]).size == 0:
            continue
        kc = np.asarray(arrs[key])
        per_cell: Dict[int, Dict[str, float]] = {}
        ratios: List[float] = []
        for c in np.unique(cid):
            sel = cid == c
            sig = float(np.nanmean(kbar[sel]))
            nul = float(np.nanmean(kc[sel]))
            ratio = float(nul / sig) if abs(sig) > 1e-12 else float("nan")
            per_cell[int(c)] = {
                "kbar_signal": sig, "kbar_null": nul, "null_ratio": ratio,
                "te_inj": float(np.nanmean(te_inj[sel])),
            }
            if np.nanmean(te_inj[sel]) > 0.0 and np.isfinite(ratio):
                ratios.append(ratio)
        out[ctrl] = {
            "mean_ratio": float(np.mean(ratios)) if ratios else None,
            "per_cell": per_cell,
            # Sibling keys, so the existing ``nul[ctrl]["mean_ratio"]`` readers are unaffected.
            "expected_to_vanish": False,
            "note": NULL_RATIO_NOTE,
        }
    return out


#: Relative tolerance for the null-cell coincidence check in :func:`prediction_controls`.
#: Relative, not absolute, because the forecast losses are MSEs in normalised scattering units
#: whose scale is set by the build's ``norm_stats``, not by anything the model controls. The
#: value is a knob: on the 400-step pilot checkpoint the residual head is still adding a
#: source-independent perturbation worth ~5% of ``base_loss``, so 0.05 sits right at the edge.
#: Re-tune it against a trained checkpoint (S8-T04) rather than reading it as a physical bound.
_NULL_CELL_RTOL: float = 0.05


def prediction_controls(
    arrs: Dict[str, Any],
    controls: Sequence[str],
    *,
    null_tol: float = _NULL_CELL_RTOL,
) -> Dict[str, Any]:
    r"""The prediction-space control that replaces the KL null ratio as the headline gate.

    A wrong source must be **worse than no source**. Per cell, over the same clean window as
    $\bar K$:

    .. math::

        \mathcal L_{\mathrm{feat}} \;<\; \mathcal L_{\mathrm{base}}
        \;<\; \mathcal L_{\mathrm{feat}}^{\pi(U)},
        \qquad
        \text{shuffle\_penalty} = \mathcal L_{\mathrm{feat}}^{\pi(U)}
        - \mathcal L_{\mathrm{feat}} \;>\; 0 .

    The left inequality says the source helps; the right says the *true* source helps, rather
    than the source pathway merely adding capacity. Unlike the KL ratio this cannot be passed
    by a posterior that simply reacts strongly to out-of-distribution input, because the
    forecast is scored against the real future.

    The ordering is asserted only on **signal** cells ($\mathrm{TE}_{\mathrm{inj}} > 0$),
    mirroring :func:`null_ratios`' convention. On the $\mathrm{TE}_{\mathrm{inj}} = 0$ null
    cells there is nothing for the source to carry, so the three losses must instead
    *coincide*: that is checked as ``null_cell_consistent``, within ``null_tol`` **relative to
    that cell's** $\mathcal L_{\mathrm{base}}$.

    Note:
        ``feat_loss_<control>`` scores an **input-stream** corruption, not the training-time
        ``feat_loss_perm`` (a ``source_state`` batch derangement). The two are different
        objects; see :func:`collect_per_sample_kbar`.

    Note:
        The left inequality is a real constraint on an *undertrained* model, not a formality.
        The residual head perturbs $\mu_{\mathrm{full}}$ away from $\mu_{\mathrm{base}}$ from
        the first step, and until the source pathway carries information that perturbation is
        source-independent noise that only *hurts* the forecast. So a fresh run reads
        $\mathcal L_{\mathrm{feat}} > \mathcal L_{\mathrm{base}}$ with
        $\text{shuffle\_penalty} \approx 0$: the gate is something training earns.

    Args:
        arrs: The dict from :func:`collect_per_sample_kbar`; needs ``feat_loss``,
            ``base_loss`` and ``feat_loss_<control>``.
        controls: The evaluated controls.
        null_tol: Tolerance for the null-cell coincidence check, **relative** to that cell's
            ``base_loss`` (the losses are MSEs in normalised units of arbitrary scale).

    Returns:
        ``{"controls", "n_signal_cells", "overall", "per_cell"}``. ``overall`` carries the mean
        losses over signal cells plus ``ordering_pass_<ctrl>``, ``ordering_pass`` (all controls,
        all signal cells) and ``ordering_pass_frac``. ``per_cell`` maps ``cell_id`` to its
        losses, ``te_inj``, per-control ``ordering_pass_<ctrl>`` / ``shuffle_penalty_<ctrl>``,
        and ``null_cell_consistent`` (``None`` on signal cells).
    """
    present = [c for c in controls
               if f"feat_loss_{c}" in arrs and np.asarray(arrs[f"feat_loss_{c}"]).size]
    cid = np.asarray(arrs["cell_id"])
    te_inj = np.asarray(arrs["te_inj"])
    feat = np.asarray(arrs.get("feat_loss", np.full(cid.shape, np.nan)))
    base = np.asarray(arrs.get("base_loss", np.full(cid.shape, np.nan)))

    per_cell: Dict[int, Dict[str, Any]] = {}
    signal_cells: List[int] = []
    for c in np.unique(cid):
        sel = cid == c
        te_c = float(_nanmean_safe(te_inj[sel]))
        f_c = float(_nanmean_safe(feat[sel]))
        b_c = float(_nanmean_safe(base[sel]))
        is_signal = te_c > 0.0
        if is_signal:
            signal_cells.append(int(c))
        entry: Dict[str, Any] = {
            "te_inj": te_c, "feat_loss": f_c, "base_loss": b_c,
            "pred_gain": b_c - f_c, "null_cell_consistent": None,
        }
        spreads: List[float] = [abs(f_c - b_c)]
        for ctrl in present:
            fc_c = float(_nanmean_safe(np.asarray(arrs[f"feat_loss_{ctrl}"])[sel]))
            entry[f"feat_loss_{ctrl}"] = fc_c
            entry[f"shuffle_penalty_{ctrl}"] = fc_c - f_c
            # The gate is meaningless on a null cell (no true source to be wrong about).
            entry[f"ordering_pass_{ctrl}"] = (
                bool(np.isfinite([f_c, b_c, fc_c]).all() and f_c < b_c < fc_c)
                if is_signal else None
            )
            spreads.append(abs(fc_c - f_c))
        if not is_signal:
            scale = abs(b_c) if np.isfinite(b_c) and abs(b_c) > 1e-12 else 1.0
            entry["null_cell_consistent"] = bool(
                np.isfinite(spreads).all() and max(spreads) < float(null_tol) * scale)
        per_cell[int(c)] = entry

    overall: Dict[str, Any] = {
        "feat_loss": float(_nanmean_safe(np.asarray(
            [per_cell[c]["feat_loss"] for c in signal_cells]))) if signal_cells else None,
        "base_loss": float(_nanmean_safe(np.asarray(
            [per_cell[c]["base_loss"] for c in signal_cells]))) if signal_cells else None,
    }
    passes: List[bool] = []
    for ctrl in present:
        if signal_cells:
            overall[f"feat_loss_{ctrl}"] = float(_nanmean_safe(np.asarray(
                [per_cell[c][f"feat_loss_{ctrl}"] for c in signal_cells])))
            overall[f"shuffle_penalty_{ctrl}"] = float(_nanmean_safe(np.asarray(
                [per_cell[c][f"shuffle_penalty_{ctrl}"] for c in signal_cells])))
            ctrl_pass = [bool(per_cell[c][f"ordering_pass_{ctrl}"]) for c in signal_cells]
            overall[f"ordering_pass_{ctrl}"] = bool(all(ctrl_pass))
            passes.extend(ctrl_pass)
        else:
            overall[f"feat_loss_{ctrl}"] = None
            overall[f"shuffle_penalty_{ctrl}"] = None
            overall[f"ordering_pass_{ctrl}"] = None
    overall["ordering_pass"] = bool(passes and all(passes))
    overall["ordering_pass_frac"] = (
        float(np.mean(passes)) if passes else None)

    return {
        "controls": list(present),
        "n_signal_cells": len(signal_cells),
        "overall": overall,
        "per_cell": per_cell,
    }


def _null_probe(arrs: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    r"""Model-free null realizability: $\mathrm{TE}_{\mathrm{scat}}$ on the null cells.

    Confirms the independent dressing alone yields $\mathrm{TE}_{\mathrm{scat}}\approx 0$
    (§14.4), read from the build-stamped ``sample_te_scat``. If a ``realizability.json`` is
    present in ``out_dir`` its null-cell $\mathrm{TE}_{\mathrm{raw}}$ / frac_Phi are folded
    in (best-effort).

    Args:
        arrs: The dict from :func:`collect_per_sample_kbar`.
        out_dir: The run directory (searched for ``realizability.json``).

    Returns:
        A dict with ``null_cell_ids``, ``null_te_scat_mean`` and optional
        ``realizability_null``.
    """
    te_inj = np.asarray(arrs["te_inj"])
    te_scat = np.asarray(arrs["te_scat"])
    cid = np.asarray(arrs["cell_id"])
    null_mask = te_inj <= 0.0
    result: Dict[str, Any] = {
        "null_cell_ids": sorted(int(c) for c in np.unique(cid[null_mask]))
        if null_mask.any() else [],
        "null_te_scat_mean": _nanmean_safe(te_scat[null_mask])
        if null_mask.any() else None,
    }
    rj = Path(out_dir) / "realizability.json"
    if rj.is_file():
        try:
            data = json.loads(rj.read_text(encoding="utf-8"))
            rn: Dict[str, Any] = {}
            for k, v in data.get("per_cell", {}).items():
                tgt = v.get("target_te", v.get("te_inj", None))
                if tgt is not None and float(tgt) == 0.0:
                    rn[str(k)] = {
                        "te_scat": v.get("te_scat"),
                        "te_raw": v.get("te_raw"),
                        "frac_phi": v.get("frac_phi"),
                    }
            if rn:
                result["realizability_null"] = rn
        except Exception:  # pragma: no cover - best-effort fold
            pass
    return result


def _cells_by_id_from_arrs(arrs: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    r"""Build the per-cell manifest (delay / te_inj) needed by :func:`recover_lags`."""
    return _group_per_cell(arrs)


def _resolve_eval_checkpoint(out_dir: Path, ckpt: Optional[Any]) -> Path:
    r"""Resolve the checkpoint to evaluate (explicit ``ckpt`` else best/final in ``out_dir``).

    Args:
        out_dir: The run directory holding ``best.ckpt`` / ``final.ckpt``.
        ckpt: An explicit checkpoint path, or ``None`` to auto-discover.

    Returns:
        The resolved checkpoint :class:`Path`.

    Raises:
        FileNotFoundError: When no checkpoint is found.
    """
    if ckpt is not None:
        p = Path(ckpt)
        if not p.is_file():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        return p
    for name in ("best.ckpt", "final.ckpt"):
        cand = Path(out_dir) / name
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"no best.ckpt / final.ckpt under {out_dir}; run --stage train first."
    )


def _write_per_sample_eval(
    arrs: Dict[str, Any],
    out_dir: Path,
    split: str,
    controls: Sequence[str],
    *,
    arm: Optional[str] = None,
    model_class: Optional[str] = None,
) -> Path:
    r"""Write the length-$N$ per-sample eval arrays to ``per_sample_eval.npz`` (Enhancement A).

    ``metrics.json`` collapses everything to ~15 per-cell means; this side-car keeps the full
    per-sample vectors so the per-sample TE-vs-$\bar K$ scatter and the per-lag calibration can
    plot every evaluated sample (and re-fit on the same points the pooled calibration used).

    Args:
        arrs: The dict returned by :func:`collect_per_sample_kbar`.
        out_dir: The run directory to write into.
        split: The evaluated split name (stored as a 0-d string array).
        controls: The null controls whose per-sample $\bar K$ and forecast-loss arrays are
            also stored (``kbar_<ctrl>``, ``feat_loss_<ctrl>``, ``shuffle_penalty_<ctrl>``).
        arm: Resolved arm name, stored as a 0-d string array (S4-T03).
        model_class: The graded checkpoint's model class, stored as a 0-d string array.

    Returns:
        The written ``per_sample_eval.npz`` :class:`Path`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = ("kbar", "te_inj", "te_scat", "frac_phi",
            "cell_id", "delay", "held_out",
            "feat_loss", "base_loss", "pred_gain", "uplift_rel")
    # The KLD summary family (§14.5) + per-head columns back the KLD-vs-TE figures; the
    # 2-D ``kbar_head`` matrix is stored too so a caller can recover the head structure.
    variants = tuple(v for v in KLD_SCALAR_VARIANTS if v != "kbar")
    head = tuple(k for k in arrs if str(k).startswith("kbar_head"))
    keys = base + variants + head
    payload: Dict[str, np.ndarray] = {
        k: np.asarray(arrs[k]) for k in keys if k in arrs
    }
    for ctrl in controls:
        # ``kbar_<ctrl>`` is the KL-space readout; the two ``*_loss`` columns are the
        # prediction-space control that actually discriminates (S3-T02b / S3-T03).
        for key in (f"kbar_{ctrl}", f"feat_loss_{ctrl}", f"shuffle_penalty_{ctrl}"):
            if key in arrs and np.asarray(arrs[key]).size:
                payload[key] = np.asarray(arrs[key])
    payload["split"] = np.asarray(str(split))
    # Provenance as 0-d string arrays (S4-T03), so a stray .npz can always be traced back to
    # the arm and architecture that produced it.
    payload["arm"] = np.asarray(str(arm) if arm is not None else "")
    payload["model_class"] = np.asarray(str(model_class) if model_class is not None else "")
    path = out_dir / "per_sample_eval.npz"
    np.savez(str(path), **payload)
    logger.info("run_eval: wrote %s (n=%d)", path, int(np.asarray(arrs["kbar"]).shape[0]))
    return path


def _pick_eval_loader(dm, split: str):
    r"""Pick an ordered eval loader for ``split`` with a test -> val -> train fallback."""
    order = [split] + [s for s in ("test", "val", "train") if s != split]
    for s in order:
        if s == "test":
            ld = dm.test_dataloader()
        elif s == "val":
            ld = dm.val_dataloader()
        else:
            ld = dm.make_plain_train_loader()
        if ld is not None:
            return ld, s
    raise RuntimeError("no split available to evaluate")


def run_eval(
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw",
    ckpt: Optional[Any] = None,
    split: str = "test",
    out_dir: Optional[Path] = None,
    batch_size: Optional[int] = None,
    device: Optional[Any] = None,
    arm: Optional[str] = None,
) -> Dict[str, Any]:
    r"""Run the Sprint 6 evaluation gates on a trained checkpoint (S6-T04).

    Loads the checkpoint, runs one clean-window pass to collect $\bar K$ (with null
    controls), fits the $\gamma$-calibration vs both $\mathrm{TE}_{\mathrm{inj}}$ and
    $\mathrm{TE}_{\mathrm{scat}}$, scores lag recovery, computes null ratios and the
    model-free null probe, and writes ``metrics.json`` under the run directory.

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        ckpt: Explicit checkpoint path (else ``best.ckpt``/``final.ckpt`` in ``out_dir``).
        split: The split to evaluate (``test`` by default; falls back to val/train).
        out_dir: The run directory (defaults to ``results/<tag>/<arm>/``).
        batch_size: Eval batch size (defaults to ``optim.batch_size``).
        device: Torch device (defaults to CUDA when available).
        arm: The resolved arm name, used only to arm-scope the fallback ``out_dir``
            when it is not passed explicitly (the driver always passes an arm+split
            ``out_dir``).

    Returns:
        The assembled ``metrics`` dict (also written to ``out_dir/metrics.json``).
    """
    import torch

    from .datamodule_v2 import SyntheticTEDataModuleV2

    # Ensure the repo root (six levels up: synthetic_v2 -> model_experiment -> model ->
    # vae_teb_prediction -> model -> <repo root>) leads sys.path so ``train.*`` resolves
    # even when run_eval is the first entry point. Mirrors run_pipeline_v2's bootstrap.
    _REPO_ROOT = str(Path(__file__).resolve().parents[5])
    if _REPO_ROOT in sys.path:
        sys.path.remove(_REPO_ROOT)
    sys.path.insert(0, _REPO_ROOT)
    from train.graph_models_utils import load_checkpoint_strict

    model_cfg = config["model"]
    bench = config["benchmarks"][benchmark]
    ev = bench.get("eval", {})
    warmup = int(model_cfg["warmup_period"])
    horizon = int(model_cfg["horizon"])
    controls = list(ev.get("null_controls", []))
    tolerance = int(ev.get("lag_tolerance_steps", 1))
    lag_threshold = float(ev.get("lag_mass_threshold", 0.8))

    out_dir = (_run_results_dir(config, benchmark, arm) if out_dir is None
               else Path(out_dir))
    dev = (torch.device(device) if device is not None
           else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = _resolve_eval_checkpoint(out_dir, ckpt)

    # Rebuild from the CHECKPOINT's own class + kwargs (not the config), so an arm is always
    # graded with the architecture it was trained under; the config supplies only the eval
    # knobs (warmup / horizon / null controls) read above. Deserialise the blob ONCE and
    # cross-check its ``model_class`` against the class the config expects (a v3 checkpoint
    # graded under a v1 config -- or vice versa -- raises rather than silently mismatching).
    from .pl_module_v2 import (
        rebuild_model_from_checkpoint,
        resolved_model_class_name,
    )

    blob = torch.load(str(ckpt_path), map_location=str(dev), weights_only=False)
    expected_class = resolved_model_class_name(model_cfg)
    model, _ = rebuild_model_from_checkpoint(blob, dev, expected_class=expected_class)
    loaded = load_checkpoint_strict(model, blob)
    if loaded is None:
        raise RuntimeError(f"checkpoint load failed (state-dict mismatch): {ckpt_path}")
    model = loaded.to(dev)
    model.eval()

    bs = int(batch_size if batch_size is not None
             else config.get("optim", {}).get("batch_size", 128))
    dm = SyntheticTEDataModuleV2(config, batch_size=bs, benchmark=benchmark)
    dm.setup("fit")
    loader, used_split = _pick_eval_loader(dm, split)

    seed = int(config.get("seeds", {}).get("base_seed", 0))
    arrs = collect_per_sample_kbar(
        model, loader, dev, warmup=warmup, horizon=horizon,
        controls=controls, control_seed=seed,
    )

    # Persist the length-N per-sample arrays (kept out of metrics.json, which stays per-cell).
    # These back the per-sample TE-vs-KLD scatter and the per-lag calibration figures, so the
    # analysis is no longer collapsed to ~15 cell means before plotting. Guarded + non-fatal:
    # a diagnostic side-car must never abort the eval or lose the primary metrics.json.
    # S4-T03: provenance read off the graded CHECKPOINT (not the config), so the recorded
    # kwargs are provably those of the weights that produced these numbers. Three
    # structurally-identical arms otherwise leave three indistinguishable artifact trees.
    resolved_arm = arm if arm is not None else blob.get("arm")
    model_class = str(blob.get("model_class", type(model).__name__))
    model_kwargs = dict(blob.get("model_kwargs", {}) or {})

    try:
        _write_per_sample_eval(arrs, out_dir, used_split, controls,
                               arm=resolved_arm, model_class=model_class)
    except Exception as exc:  # noqa: BLE001
        logger.warning("run_eval: per_sample_eval.npz not written (%s)", exc)

    # ``kld_support`` lives only on v3 instances; the getattr guard keeps the v1 alias working.
    calibration = fit_calibration(
        arrs,
        kld_support=str(getattr(model, "kld_support", "full")),
        kbar_null_threshold=float(ev.get("kbar_null_threshold", _KBAR_NULL_THRESHOLD)),
        boot_seed=seed,
    )
    cells_by_id = _cells_by_id_from_arrs(arrs)
    lag = recover_lags(
        arrs["lag_profiles"], cells_by_id,
        horizon=horizon, tolerance=tolerance, threshold=lag_threshold,
    )
    nulls = null_ratios(arrs, controls)
    # The headline control (S3-T03). ``null_ratios`` above is retained as a readout only.
    pred_controls = prediction_controls(arrs, controls)
    probe = _null_probe(arrs, out_dir)

    # frac_Phi summary over the signal cells (from the build-stamped per-sample value).
    signal = np.asarray(arrs["te_inj"]) > 0.0
    frac_all = np.asarray(arrs["frac_phi"])
    frac_signal = frac_all[signal & np.isfinite(frac_all)]
    frac_summary = {
        "mean": float(np.mean(frac_signal)) if frac_signal.size else None,
        "min": float(np.min(frac_signal)) if frac_signal.size else None,
        "max": float(np.max(frac_signal)) if frac_signal.size else None,
    }

    # Merged per-cell table for the report / CSV.
    null_by_cell = {ctrl: nulls[ctrl]["per_cell"] for ctrl in nulls}
    pred_by_cell = pred_controls["per_cell"]
    per_cell_table: List[Dict[str, Any]] = []
    for cid, cell in sorted(cells_by_id.items()):
        pcell = pred_by_cell.get(cid, {})
        row = {
            "cell_id": cid,
            "te_inj": cell["te_inj"],
            "te_scat": cell["te_scat"],
            "D": cell["delay"],
            "kbar_mean": cell["kbar"],
            "n": cell["n"],
            "frac_phi": cell["frac_phi"],
            "feat_loss": pcell.get("feat_loss"),
            "base_loss": pcell.get("base_loss"),
            "pred_gain": cell["pred_gain"],
            "uplift_rel": cell["uplift_rel"],
            "lag_mass": lag["per_cell"].get(cid, {}).get("lag_mass"),
            "peak_lag_err": lag["per_cell"].get(cid, {}).get("peak_lag_err"),
        }
        for ctrl in nulls:
            # KL-space readout (near 1.0 on an honest model; see NULL_RATIO_NOTE).
            row[f"null_{ctrl}_ratio"] = null_by_cell[ctrl].get(cid, {}).get("null_ratio")
        for ctrl in pred_controls["controls"]:
            # Prediction-space gate: the columns the headline figure and report render.
            row[f"feat_loss_{ctrl}"] = pcell.get(f"feat_loss_{ctrl}")
            row[f"shuffle_penalty_{ctrl}"] = pcell.get(f"shuffle_penalty_{ctrl}")
            row[f"ordering_pass_{ctrl}"] = pcell.get(f"ordering_pass_{ctrl}")
        per_cell_table.append(row)

    # Per-cell variable-length diagnostics (kept out of the flat table): the attention
    # lag profile A_ell and the per-step $\bar K_t$ trajectory. ``_jsonable`` tolist()s
    # the numpy arrays and stringifies the int cell keys on write.
    lag_profiles = arrs.get("lag_profiles", {}) or {}
    kbar_over_time = arrs.get("kbar_over_time", {}) or {}
    lag_counts = arrs.get("lag_counts", {}) or {}
    per_cell_profiles = {
        int(cid): {
            "lag_profile": lag_profiles.get(cid),
            "kbar_over_time": kbar_over_time.get(cid),
            "lag_count": int(lag_counts.get(cid, 0)),
        }
        for cid in sorted(set(lag_profiles) | set(kbar_over_time))
    }

    metrics: Dict[str, Any] = {
        "run_tag": str(config.get("experiment", {}).get("tag", benchmark)),
        "benchmark": benchmark,
        # --- provenance (S4-T03): which arm, which architecture, which data, which weights --
        "arm": resolved_arm,
        "model_class": model_class,
        "model_kwargs": model_kwargs,
        "loss_settings": dict(config.get("loss", {}) or {}),
        "cache_dir": str(resolve_cache_dir(config, benchmark=benchmark)),
        "ckpt": str(ckpt_path),
        "split": used_split,
        "device": str(dev),
        "warmup": warmup,
        "horizon": horizon,
        "T": int(arrs["T"]),
        "n_samples": int(arrs["n"]),
        "n_cells": calibration["n_cells"],
        "calibration": calibration,
        "lag_recovery": lag,
        "prediction_controls": pred_controls,
        "null_controls": nulls,
        "null_probe": probe,
        "frac_phi": frac_summary,
        "per_cell": per_cell_table,
        "per_cell_profiles": per_cell_profiles,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(_jsonable(metrics), handle, indent=2)
    logger.info("run_eval: wrote %s", out_dir / "metrics.json")
    return metrics


def _fmt(x: Any, spec: str = ".4g") -> str:
    r"""Format a scalar for the markdown report (``n/a`` for ``None``/non-finite)."""
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return "n/a"
    return format(xf, spec)


def write_report(metrics: Dict[str, Any], out_dir: Path) -> Path:
    r"""Write the minimal Sprint-6 markdown report from ``metrics`` (S6-T04).

    A compact summary of the calibration, lag-recovery, prediction-control and preservation
    gates, plus the KL-space null ratio as a **readout**. This is intentionally minimal:
    Sprint 7 (S7-T05 ``final_report_v2``) supersedes it with the full journal report and
    figure gallery.

    Args:
        metrics: The dict returned by :func:`run_eval`.
        out_dir: The run directory to write ``report.md`` into.

    Returns:
        The written ``report.md`` :class:`Path`.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cal = metrics.get("calibration", {})
    lag = metrics.get("lag_recovery", {})
    nul = metrics.get("null_controls", {})
    pred = metrics.get("prediction_controls", {}) or {}
    probe = metrics.get("null_probe", {})
    frac = metrics.get("frac_phi", {})

    lines: List[str] = [
        "# synthetic_v2 - Sprint 6 evaluation report",
        "",
        f"- run tag: `{metrics.get('run_tag')}`",
        f"- checkpoint: `{metrics.get('ckpt')}`",
        f"- split: `{metrics.get('split')}` "
        f"(n_samples={metrics.get('n_samples')}, n_cells={metrics.get('n_cells')})",
        f"- clean window: `[max(w={metrics.get('warmup')}, D-1), "
        f"T-H={int(metrics.get('T', 0)) - int(metrics.get('horizon', 0))})`",
        "",
        "## Calibration  (K-bar = alpha + gamma * TE)",
        "",
        "| target | gamma | alpha | R^2 | monotonic |",
        "|---|---|---|---|---|",
        f"| TE_inj  | {_fmt(cal.get('gamma_inj'))} | {_fmt(cal.get('alpha_inj'))} "
        f"| {_fmt(cal.get('r2_inj'))} | {cal.get('monotonic_inj')} |",
        f"| TE_scat | {_fmt(cal.get('gamma_scat'))} | {_fmt(cal.get('alpha_scat'))} "
        f"| {_fmt(cal.get('r2_scat'))} | {cal.get('monotonic_scat')} |",
        "",
        "## Lag recovery",
        "",
        f"- mean LagMass (signal cells): {_fmt(lag.get('mean_lag_mass'))} "
        f"(threshold {_fmt(lag.get('lag_mass_threshold'))}, "
        f"pass={lag.get('mean_lag_mass_pass')})",
        f"- fraction within +-{lag.get('tolerance')} step: "
        f"{_fmt(lag.get('frac_within_tol'))}",
        "",
        "## Prediction controls  (gate: L_feat < L_base < L_feat_corrupted)",
        "",
    ]
    p_over = pred.get("overall") or {}
    p_ctrls = pred.get("controls") or []
    if not p_ctrls:
        lines.append("- n/a (no prediction controls in metrics)")
    else:
        lines.append(
            f"- L_feat = {_fmt(p_over.get('feat_loss'))}, "
            f"L_base = {_fmt(p_over.get('base_loss'))} (signal cells)")
        for ctrl in p_ctrls:
            verdict = p_over.get(f"ordering_pass_{ctrl}")
            mark = "n/a" if verdict is None else ("pass" if verdict else "FAIL")
            lines.append(
                f"- {ctrl}: L_feat_corrupted = {_fmt(p_over.get(f'feat_loss_{ctrl}'))}, "
                f"penalty = {_fmt(p_over.get(f'shuffle_penalty_{ctrl}'))}, "
                f"ordering = {mark}")
        lines.append(
            f"- signal cells passing: {_fmt(p_over.get('ordering_pass_frac'), '.0%')} "
            f"of {pred.get('n_signal_cells', '?')}")
    lines += [
        "",
        # The heading is load-bearing (asserted by test_null_run_eval_writes_metrics), but the
        # caption is not: the ratio is a readout, not a gate. See eval_v2.NULL_RATIO_NOTE.
        "## Null controls  (readout: K-bar_null / K-bar_signal, expected ~1, NOT ->0)",
        "",
    ]
    for ctrl, res in nul.items():
        lines.append(f"- {ctrl}: mean null_ratio = {_fmt(res.get('mean_ratio'))}")
    lines += [
        f"- null-cell TE_scat (dressing only): {_fmt(probe.get('null_te_scat_mean'))}",
        "",
        "> KL(q||p) measures that the source moved the belief, not that it moved it "
        "correctly; a deranged source is still a source, and is out of distribution for a "
        "posterior trained on matched pairs. This ratio therefore sits near 1.0 even on a "
        "model that demonstrably exploits the source (v3 Finding F2). The discriminating "
        "gate is the prediction control above.",
        "",
        "## Preservation",
        "",
        f"- mean frac_Phi (signal cells): {_fmt(frac.get('mean'))} "
        f"[{_fmt(frac.get('min'))}, {_fmt(frac.get('max'))}]",
        "",
        "> Minimal Sprint-6 report; Sprint 7 (S7-T05 `final_report_v2`) supersedes it "
        "with the full figure gallery and per-sample TE-annotated diagnostics.",
        "",
    ]
    path = out / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("write_report: wrote %s", path)
    return path
