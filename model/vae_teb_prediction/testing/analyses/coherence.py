"""
Time-frequency coherence analysis for VAE-TEB models.

This module analyzes reconstruction coherence between the original FHR
signal and the model's predicted FHR signal. Optional UP-FHR coherence
can be computed to assess physiological coupling preservation.

Example:
    >>> from testing.analyses.coherence import run_coherence_analysis
    >>> results = run_coherence_analysis(runner, test_loader)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from scipy import signal
import matplotlib.pyplot as plt

from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.collectors import _extract_epoch, _extract_guid, _extract_label
from model.vae_teb_prediction.testing.metrics import aggregate_predictions
from model.vae_teb_prediction.testing.visualizers import (
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_ORANGE,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TITLE,
    _style_axes,
    _tighten_xaxis,
    plot_coherence_analysis,
    plot_coherence_signals,
    plot_coherence_spectrum,
    plot_cross_correlation,
    plot_horizon_spectra,
    plot_psd_comparison,
    plot_reconstruction_coherence,
    plot_spectrum_delta,
    plot_time_frequency_coherence,
    plot_time_frequency_map,
)

COHERENCE_BANDS = [
    ("VLF", 0.0, 0.04),
    ("LF", 0.04, 0.15),
    ("HF", 0.15, 0.5),
    ("Total", 0.0, 0.5),
]


def _resolve_time_frequency_methods(method: Optional[str]) -> List[str]:
    if not method:
        return ["stft"]
    key = str(method).strip().lower()
    if key in {"both", "all", "stft+wavelet", "wavelet+stft"}:
        return ["stft", "wavelet"]
    if key in {"stft", "wavelet"}:
        return [key]
    logger.warning("Unknown time-frequency method '%s', using STFT only.", method)
    return ["stft"]


def _welch_segment_count(n_samples: int, nperseg: int, noverlap: int) -> int:
    if n_samples < nperseg or nperseg <= 0:
        return 0
    step = max(1, nperseg - noverlap)
    return 1 + max(0, (n_samples - nperseg) // step)


def _coherence_significance_threshold(k_segments: int, alpha: float) -> float:
    if k_segments <= 1 or alpha <= 0.0 or alpha >= 1.0:
        return np.nan
    return float(1.0 - alpha ** (1.0 / (k_segments - 1)))


def _compute_horizon_spectra(
    frequencies: np.ndarray,
    times: np.ndarray,
    coherence: np.ndarray,
    *,
    early_seconds: float,
    late_seconds: float,
) -> Dict[str, np.ndarray]:
    if frequencies.size == 0 or times.size == 0 or coherence.size == 0:
        return {"early": np.array([]), "late": np.array([]), "delta": np.array([])}

    total_duration = float(times[-1]) if times.size > 0 else 0.0
    early_seconds = max(0.0, float(early_seconds))
    late_seconds = max(0.0, float(late_seconds))

    early_mask = times <= early_seconds if early_seconds > 0 else np.zeros_like(times, dtype=bool)
    late_mask = times >= (total_duration - late_seconds) if late_seconds > 0 else np.zeros_like(times, dtype=bool)

    if not early_mask.any():
        early_mask = np.zeros_like(times, dtype=bool)
        early_mask[0] = True
    if not late_mask.any():
        late_mask = np.zeros_like(times, dtype=bool)
        late_mask[-1] = True

    early = np.nanmean(coherence[:, early_mask], axis=1)
    late = np.nanmean(coherence[:, late_mask], axis=1)
    delta = late - early
    return {"early": early, "late": late, "delta": delta}


def _collect_prediction_windows(
    reference_full: np.ndarray,
    pred_windows: np.ndarray,
    *,
    stride: int,
    warmup: int,
    max_windows: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    windows: List[Tuple[np.ndarray, np.ndarray]] = []
    min_len = len(reference_full)
    if min_len < 2:
        return windows
    T, H = pred_windows.shape
    for t in range(warmup, T):
        if max_windows is not None and len(windows) >= max_windows:
            break
        start = t * stride
        end = start + H
        if end > min_len:
            break
        windows.append((reference_full[start:end], pred_windows[t]))
    return windows


def _compute_permutation_ci(
    windows: List[Tuple[np.ndarray, np.ndarray]],
    *,
    map_fn,
    num_permutations: int,
    alpha: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    if num_permutations <= 0 or len(windows) < 2:
        return {"lower": np.array([]), "upper": np.array([]), "frequencies": np.array([]), "times": np.array([])}

    refs = [w[0] for w in windows]
    preds = [w[1] for w in windows]

    perm_maps: List[np.ndarray] = []
    freqs = None
    times = None
    n_windows = len(preds)
    for _ in range(num_permutations):
        perm = rng.permutation(n_windows)
        acc = None
        count = 0
        for i, j in enumerate(perm):
            tf_map = map_fn(refs[i], preds[j])
            if tf_map["coherence"].size == 0:
                continue
            if acc is None:
                acc = tf_map["coherence"].copy()
                freqs = tf_map["frequencies"]
                times = tf_map["times"]
            else:
                if tf_map["coherence"].shape != acc.shape:
                    continue
                acc += tf_map["coherence"]
            count += 1
        if acc is None or count == 0:
            continue
        perm_maps.append(acc / count)

    if not perm_maps:
        return {"lower": np.array([]), "upper": np.array([]), "frequencies": np.array([]), "times": np.array([])}

    perm_stack = np.stack(perm_maps, axis=0)
    lower = np.nanpercentile(perm_stack, alpha * 100.0, axis=0)
    upper = np.nanpercentile(perm_stack, (1.0 - alpha) * 100.0, axis=0)

    return {
        "lower": lower,
        "upper": upper,
        "frequencies": freqs if freqs is not None else np.array([]),
        "times": times if times is not None else np.array([]),
    }


def _stack_consistent(maps: List[np.ndarray]) -> np.ndarray:
    if not maps:
        return np.empty((0, 0))
    shape = maps[0].shape
    filtered = [m for m in maps if m.shape == shape]
    if not filtered:
        return np.empty((0, 0))
    return np.stack(filtered, axis=0)


def _plot_band_trends(
    df: pd.DataFrame,
    output_path: Path,
    *,
    x_col: str,
    x_label: str,
    title: str,
    y_label: str = "Coherence",
    label_pair: Tuple[str, str] = ("Reference", "Reconstruction"),
    single_label: str = "Coherence",
    xlim: Optional[Tuple[float, float]] = None,
    invert_x: bool = False,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bands = [b[0] for b in COHERENCE_BANDS]
    n_bands = len(bands)
    cols = 1
    rows = n_bands
    fig, axes = plt.subplots(rows, cols, figsize=(7.6, 1.9 * rows), sharex=False)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array(axes)[np.newaxis, :]
    elif cols == 1:
        axes = np.array(axes)[:, np.newaxis]

    has_pair = "orig_mean" in df.columns and "recon_mean" in df.columns
    has_single = "coherence_mean" in df.columns

    if not has_pair and not has_single:
        return

    for idx, band in enumerate(bands):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        subset = df[df["band"] == band].sort_values(x_col)
        if subset.empty:
            ax.axis("off")
            continue

        x_vals = subset[x_col].values
        if has_pair:
            ax.plot(
                x_vals,
                subset["orig_mean"],
                color=COLOR_BLUE,
                linewidth=1.2,
                label=label_pair[0],
            )
            ax.plot(
                x_vals,
                subset["recon_mean"],
                color=COLOR_ORANGE,
                linewidth=1.2,
                linestyle="--",
                label=label_pair[1],
            )

            if "orig_std" in subset.columns:
                ax.fill_between(
                    x_vals,
                    subset["orig_mean"] - subset["orig_std"],
                    subset["orig_mean"] + subset["orig_std"],
                    color=COLOR_BLUE,
                    alpha=0.2,
                    label="Reference +/- 1 SD",
                )
            if "recon_std" in subset.columns:
                ax.fill_between(
                    x_vals,
                    subset["recon_mean"] - subset["recon_std"],
                    subset["recon_mean"] + subset["recon_std"],
                    color=COLOR_ORANGE,
                    alpha=0.2,
                    label="Reconstruction +/- 1 SD",
                )
        else:
            ax.plot(
                x_vals,
                subset["coherence_mean"],
                color=COLOR_BLUE,
                linewidth=1.2,
                label=single_label,
            )
            if "coherence_std" in subset.columns:
                ax.fill_between(
                    x_vals,
                    subset["coherence_mean"] - subset["coherence_std"],
                    subset["coherence_mean"] + subset["coherence_std"],
                    color=COLOR_BLUE,
                    alpha=0.2,
                    label=f"{single_label} +/- 1 SD",
                )

        ax.set_title(f"{band} Band", fontsize=FONT_TITLE, fontweight="normal")
        ax.set_xlabel(x_label, fontsize=FONT_LABEL)
        ax.set_ylabel(y_label, fontsize=FONT_LABEL)
        if y_label.lower().startswith("coherence"):
            y_vals = []
            if has_pair:
                y_vals.extend(subset["orig_mean"].values.tolist())
                y_vals.extend(subset["recon_mean"].values.tolist())
            else:
                y_vals.extend(subset["coherence_mean"].values.tolist())
            y_arr = np.asarray(y_vals, dtype=float)
            y_arr = y_arr[np.isfinite(y_arr)]
            if y_arr.size > 0:
                y_min = float(np.min(y_arr))
                y_max = float(np.max(y_arr))
                pad = max(0.02, 0.05 * (y_max - y_min))
                y_min -= pad
                y_max += pad
                if y_min < 0.0:
                    y_min = 0.0
                if y_max > 1.0:
                    y_max = 1.0
                if y_max <= y_min:
                    y_max = min(1.0, y_min + 0.1)
                ax.set_ylim(y_min, y_max)
        ax.legend(loc="best", fontsize=FONT_LEGEND, framealpha=0.95)
        _style_axes(ax, grid="both", minor_ticks=True)
        if xlim is None:
            _tighten_xaxis(ax, x_vals)
        else:
            ax.set_xlim(xlim[0], xlim[1])
        if invert_x:
            ax.invert_xaxis()

    # Hide unused subplots (defensive; should not trigger with single column)
    for idx in range(n_bands, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=FONT_TITLE, y=1.02, fontweight="normal")
    fig.tight_layout(pad=0.6)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def compute_stft_coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 4.0,
    nperseg: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute magnitude-squared coherence using Welch's method.

    The coherence Cxy measures linear correlation between signals at each
    frequency, with values in [0, 1].

    Args:
        x: First signal array.
        y: Second signal array (must match length of x).
        fs: Sampling frequency in Hz (default 4.0).
        nperseg: Segment length for Welch's method (default 64 = 16 seconds).

    Returns:
        Tuple of:
            - frequencies: Frequency array in Hz
            - coherence: Coherence values in [0, 1]

    Example:
        >>> freqs, coh = compute_stft_coherence(fhr_signal, fhr_pred)
        >>> plt.plot(freqs, coh)
    """

    # Ensure same length
    min_len = min(len(x), len(y))
    x, y = x[:min_len], y[:min_len]

    # Compute coherence
    frequencies, coherence = signal.coherence(
        x, y, fs=fs, nperseg=nperseg, noverlap=nperseg // 2
    )

    return frequencies, coherence


def compute_welch_psd(
    x: np.ndarray,
    *,
    fs: float = 4.0,
    nperseg: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch PSD for a signal.

    Args:
        x: Signal array.
        fs: Sampling frequency in Hz.
        nperseg: Segment length for Welch's method.

    Returns:
        Tuple of (frequencies, PSD).
    """
    if x.size < 2:
        return np.array([]), np.array([])

    nperseg_used = min(nperseg, x.size)
    if nperseg_used < 2:
        return np.array([]), np.array([])

    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg_used, noverlap=nperseg_used // 2)
    return freqs, psd


def compute_bandpower(
    x: np.ndarray,
    *,
    fs: float,
    bands: List[Tuple[str, float, float]],
    nperseg: int = 128,
) -> Dict[str, float]:
    """
    Compute band power for defined frequency bands using Welch PSD.

    Args:
        x: Signal array.
        fs: Sampling frequency in Hz.
        bands: List of (name, f_low, f_high).
        nperseg: Welch segment length.

    Returns:
        Dict mapping band name to band power.
    """
    freqs, psd = compute_welch_psd(x, fs=fs, nperseg=nperseg)
    if freqs.size == 0 or psd.size == 0:
        return {name: np.nan for name, _, _ in bands}

    band_powers: Dict[str, float] = {}
    for name, f_low, f_high in bands:
        mask = (freqs >= f_low) & (freqs < f_high)
        if not mask.any():
            band_powers[name] = np.nan
            continue
        band_powers[name] = float(np.trapz(psd[mask], freqs[mask]))

    return band_powers


def compute_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fs: float = 4.0,
    max_lag_sec: float = 120.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized cross-correlation between two signals.

    Args:
        x: First signal array.
        y: Second signal array.
        fs: Sampling frequency in Hz.
        max_lag_sec: Max lag to keep (seconds).

    Returns:
        Tuple of (lags_sec, correlation).
    """
    min_len = min(len(x), len(y))
    if min_len < 2:
        return np.array([]), np.array([])

    x = x[:min_len] - np.mean(x[:min_len])
    y = y[:min_len] - np.mean(y[:min_len])

    corr = signal.correlate(y, x, mode="full")
    lags = signal.correlation_lags(len(y), len(x), mode="full")

    denom = np.std(x) * np.std(y) * max(min_len, 1)
    if denom > 0:
        corr = corr / denom
    else:
        corr = corr * 0.0

    max_lag = int(max_lag_sec * fs)
    if max_lag <= 0:
        return np.array([]), np.array([])

    center = len(corr) // 2
    start = max(center - max_lag, 0)
    end = min(center + max_lag + 1, len(corr))

    return lags[start:end] / fs, corr[start:end]


def _band_means(
    frequencies: np.ndarray,
    coherence: np.ndarray,
) -> Dict[str, np.ndarray]:
    if frequencies is None or coherence is None:
        return {}

    results: Dict[str, np.ndarray] = {}
    for band_name, f_low, f_high in COHERENCE_BANDS:
        mask = (frequencies >= f_low) & (frequencies < f_high)
        if not mask.any():
            results[band_name] = np.full(coherence.shape[-1], np.nan) if coherence.ndim == 2 else np.nan
            continue

        if coherence.ndim == 1:
            results[band_name] = float(np.nanmean(coherence[mask]))
        else:
            results[band_name] = np.nanmean(coherence[mask, :], axis=0)

    return results


def compute_windowed_coherence_map(
    reference_full: np.ndarray,
    pred_windows: np.ndarray,
    *,
    fs: float,
    stride: int,
    warmup: int,
    nperseg: int,
) -> Dict[str, np.ndarray]:
    """
    Compute per-window coherence maps aligned to the model's 2-minute predictions.

    Args:
        reference_full: Reference signal array (raw FHR).
        pred_windows: Predicted windows, shape (T, H).
        fs: Sampling frequency in Hz.
        stride: Step size in raw samples between windows (decimation factor).
        warmup: Number of initial timesteps to skip.
        nperseg: Welch segment length for coherence.

    Returns:
        Dict with:
            - 'frequencies': Frequency array
            - 'times': Time array (window centers, seconds)
            - 'coherence': Coherence map (freq x time) between reference and predicted window
    """
    min_len = len(reference_full)
    if min_len < 2:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence": np.empty((0, 0)),
        }

    T, H = pred_windows.shape
    if H < 2:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence": np.empty((0, 0)),
        }

    nperseg_used = min(nperseg, H)
    if nperseg_used < 2:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence": np.empty((0, 0)),
        }

    coh_list: List[np.ndarray] = []
    times: List[float] = []
    window_indices: List[int] = []
    frequencies = None

    for t in range(warmup, T):
        start = t * stride
        end = start + H
        if end > min_len:
            break

        ref_win = reference_full[start:end]
        pred_win = pred_windows[t]

        freq, coh = compute_stft_coherence(ref_win, pred_win, fs=fs, nperseg=nperseg_used)

        if frequencies is None:
            frequencies = freq

        coh_list.append(coh)
        times.append((start + H / 2) / fs)
        window_indices.append(t)

    if not coh_list:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence": np.empty((0, 0)),
        }

    coherence = np.stack(coh_list, axis=1)

    return {
        "frequencies": frequencies,
        "times": np.array(times),
        "window_indices": np.array(window_indices),
        "coherence": coherence,
    }


def compute_wavelet_coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 4.0,
    num_scales: int = 50,
    *,
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
    pad_mode: Optional[str] = "reflect",
    pad_max_fraction: float = 0.25,
    coi_scale: float = 1.65,
    apply_coi_mask: bool = True,
    smooth_kernel: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Compute wavelet coherence between two signals.

    Uses continuous wavelet transform for time-frequency coherence analysis.

    Args:
        x: First signal array.
        y: Second signal array.
        fs: Sampling frequency in Hz.
        num_scales: Number of wavelet scales to compute.

    Returns:
        Dict with:
            - 'frequencies': Frequency array
            - 'times': Time array
            - 'coherence': 2D coherence matrix (freq x time)
            - 'phase': Phase difference matrix

    Example:
        >>> result = compute_wavelet_coherence(fhr_orig, fhr_pred)
        >>> plt.pcolormesh(result['times'], result['frequencies'], result['coherence'])
    """
    try:
        import pywt
    except ImportError:
        raise ImportError("PyWavelets is required for wavelet coherence. Install with: pip install pywavelets")

    min_len = min(len(x), len(y))
    if min_len < 2:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence": np.empty((0, 0)),
            "phase": np.empty((0, 0)),
            "scales": np.array([]),
        }

    x = x[:min_len]
    y = y[:min_len]

    wavelet = "morl"
    dt = 1.0 / fs
    center_freq = pywt.central_frequency(wavelet)

    min_resolvable = 1.0 / max(min_len / fs, 1e-6)
    if min_freq is None:
        min_freq = 0.003
    min_freq = max(min_freq, min_resolvable)

    if max_freq is None:
        max_freq = fs / 2.0
    max_freq = min(max_freq, fs / 2.0)

    if min_freq >= max_freq or min_freq <= 0.0:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence": np.empty((0, 0)),
            "phase": np.empty((0, 0)),
            "scales": np.array([]),
        }

    scale_min = center_freq / (max_freq * dt)
    scale_max = center_freq / (min_freq * dt)
    scale_min = max(scale_min, 1.0)
    scale_max = min(scale_max, max(min_len / 2.0, scale_min + 1e-6))
    if scale_max <= scale_min:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence": np.empty((0, 0)),
            "phase": np.empty((0, 0)),
            "scales": np.array([]),
        }

    scales = np.logspace(np.log10(scale_min), np.log10(scale_max), num_scales)

    pad_samples = 0
    if pad_mode:
        max_scale = float(np.max(scales))
        coi_samples = int(np.ceil(coi_scale * max_scale))
        max_pad = int(np.floor(min_len * pad_max_fraction))
        pad_samples = max(0, min(coi_samples, max_pad))

    if pad_samples > 0:
        x = np.pad(x, (pad_samples, pad_samples), mode=pad_mode)
        y = np.pad(y, (pad_samples, pad_samples), mode=pad_mode)

    coefs_x, frequencies = pywt.cwt(x, scales, wavelet, sampling_period=dt)
    coefs_y, _ = pywt.cwt(y, scales, wavelet, sampling_period=dt)

    cross_spectrum = coefs_x * np.conj(coefs_y)
    power_x = np.abs(coefs_x) ** 2
    power_y = np.abs(coefs_y) ** 2

    kernel_size = max(1, int(smooth_kernel))
    kernel = np.ones(kernel_size) / kernel_size

    coherence = np.zeros_like(power_x)
    for i in range(len(scales)):
        smoothed_cross = np.convolve(cross_spectrum[i], kernel, mode="same")
        smoothed_x = np.convolve(power_x[i], kernel, mode="same")
        smoothed_y = np.convolve(power_y[i], kernel, mode="same")
        coherence[i] = np.abs(smoothed_cross) ** 2 / (smoothed_x * smoothed_y + 1e-12)

    phase = np.angle(cross_spectrum)

    if pad_samples > 0:
        coherence = coherence[:, pad_samples:pad_samples + min_len]
        phase = phase[:, pad_samples:pad_samples + min_len]

    if apply_coi_mask:
        for i, scale in enumerate(scales):
            edge = int(np.ceil(coi_scale * scale))
            if edge <= 0:
                continue
            if edge * 2 >= min_len:
                coherence[i, :] = np.nan
                phase[i, :] = np.nan
                continue
            coherence[i, :edge] = np.nan
            coherence[i, -edge:] = np.nan
            phase[i, :edge] = np.nan
            phase[i, -edge:] = np.nan

    times = np.arange(min_len) / fs

    if frequencies.ndim == 1 and frequencies.size > 1:
        if np.any(np.diff(frequencies) < 0):
            order = np.argsort(frequencies)
            frequencies = frequencies[order]
            coherence = coherence[order, :]
            phase = phase[order, :]
            scales = scales[order]

    return {
        "frequencies": frequencies,
        "times": times,
        "coherence": coherence,
        "phase": phase,
        "scales": scales,
    }


def compute_stft_coherence_map(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 4.0,
    nperseg: int = 128,
    noverlap: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute time-frequency coherence using STFT.

    Args:
        x: First signal array.
        y: Second signal array.
        fs: Sampling frequency in Hz.
        nperseg: STFT segment length.
        noverlap: STFT overlap (defaults to 50%).

    Returns:
        Dict with:
            - 'frequencies': Frequency array
            - 'times': Time array
            - 'coherence': Coherence matrix (freq x time)
    """
    min_len = min(len(x), len(y))
    x, y = x[:min_len], y[:min_len]

    if min_len < 2:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence": np.empty((0, 0)),
        }

    if nperseg > min_len:
        nperseg = min_len

    if noverlap is None:
        noverlap = nperseg // 2
    elif noverlap >= nperseg:
        noverlap = max(0, nperseg // 2)

    freqs, times, stft_x = signal.stft(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
        window="hann",
    )
    _, _, stft_y = signal.stft(
        y,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
        window="hann",
    )

    sxy = stft_x * np.conj(stft_y)
    sxx = np.abs(stft_x) ** 2
    syy = np.abs(stft_y) ** 2

    coherence = np.abs(sxy) ** 2 / (sxx * syy + 1e-12)

    return {"frequencies": freqs, "times": times, "coherence": coherence}


def compute_window_relative_time_frequency(
    reference_full: np.ndarray,
    pred_windows: np.ndarray,
    *,
    fs: float,
    stride: int,
    warmup: int,
    nperseg: int,
    max_windows: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute average time-frequency coherence within 2-minute windows.

    Aligns STFT coherence maps to the start of each prediction window and
    averages across windows to show coherence vs time-from-window-start.

    Returns:
        Dict with:
            - 'frequencies': Frequency array
            - 'times': Time array
            - 'coherence': Mean coherence (freq x time)
            - 'coherence_std': Std across windows (freq x time)
            - 'coherence_cv': Coefficient of variation across windows
            - 'n_windows': Number of windows aggregated
    """
    min_len = len(reference_full)
    T, H = pred_windows.shape
    if min_len < 2 or H < 2:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence": np.empty((0, 0)),
            "n_windows": 0,
        }

    nperseg_used = min(nperseg, H)
    if nperseg_used < 2:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence": np.empty((0, 0)),
            "n_windows": 0,
        }

    acc = None
    acc_sq = None
    freqs = None
    times = None
    count = 0

    for t in range(warmup, T):
        if max_windows is not None and count >= max_windows:
            break

        start = t * stride
        end = start + H
        if end > min_len:
            break

        ref_win = reference_full[start:end]
        pred_win = pred_windows[t]

        tf_map = compute_stft_coherence_map(ref_win, pred_win, fs=fs, nperseg=nperseg_used)

        if tf_map["coherence"].size == 0:
            continue

        if acc is None:
            acc = tf_map["coherence"].copy()
            acc_sq = np.square(tf_map["coherence"].astype(float))
            freqs = tf_map["frequencies"]
            times = tf_map["times"]
        else:
            if tf_map["coherence"].shape != acc.shape:
                continue
            acc += tf_map["coherence"]
            acc_sq += np.square(tf_map["coherence"].astype(float))

        count += 1

    if acc is None:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence": np.empty((0, 0)),
            "n_windows": 0,
        }

    mean = acc / max(count, 1)
    variance = acc_sq / max(count, 1) - np.square(mean)
    variance = np.clip(variance, 0.0, None)
    std = np.sqrt(variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = std / mean
        cv[~np.isfinite(cv)] = np.nan

    return {
        "frequencies": freqs,
        "times": times,
        "coherence": mean,
        "coherence_std": std,
        "coherence_cv": cv,
        "n_windows": count,
    }


def compute_window_relative_time_frequency_wavelet(
    reference_full: np.ndarray,
    pred_windows: np.ndarray,
    *,
    fs: float,
    stride: int,
    warmup: int,
    num_scales: int,
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
    pad_mode: Optional[str] = "reflect",
    pad_max_fraction: float = 0.25,
    coi_scale: float = 1.65,
    apply_coi_mask: bool = True,
    max_windows: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute average wavelet coherence within 2-minute windows.

    Aligns wavelet coherence maps to the start of each prediction window and
    averages across windows to show coherence vs time-from-window-start.

    Returns:
        Dict with:
            - 'frequencies': Frequency array
            - 'times': Time array
            - 'coherence': Mean coherence (freq x time)
            - 'coherence_std': Std across windows (freq x time)
            - 'coherence_cv': Coefficient of variation across windows
            - 'plv': Phase-locking value map
            - 'phase_mean': Circular mean phase map
            - 'n_windows': Number of windows aggregated
    """
    min_len = len(reference_full)
    T, H = pred_windows.shape
    if min_len < 2 or H < 2:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence": np.empty((0, 0)),
            "n_windows": 0,
        }

    acc = None
    acc_sq = None
    acc_count = None
    phase_sum = None
    phase_count = None
    freqs = None
    times = None
    count = 0

    for t in range(warmup, T):
        if max_windows is not None and count >= max_windows:
            break

        start = t * stride
        end = start + H
        if end > min_len:
            break

        ref_win = reference_full[start:end]
        pred_win = pred_windows[t]

        tf_map = compute_wavelet_coherence(
            ref_win,
            pred_win,
            fs=fs,
            num_scales=num_scales,
            min_freq=min_freq,
            max_freq=max_freq,
            pad_mode=pad_mode,
            pad_max_fraction=pad_max_fraction,
            coi_scale=coi_scale,
            apply_coi_mask=apply_coi_mask,
        )

        if tf_map["coherence"].size == 0:
            continue

        coh = tf_map["coherence"]
        phase = tf_map["phase"]
        if acc is None:
            acc = np.zeros_like(coh, dtype=float)
            acc_sq = np.zeros_like(coh, dtype=float)
            acc_count = np.zeros_like(coh, dtype=float)
            phase_sum = np.zeros_like(coh, dtype=complex)
            phase_count = np.zeros_like(coh, dtype=float)
            freqs = tf_map["frequencies"]
            times = tf_map["times"]
        else:
            if coh.shape != acc.shape:
                continue

        mask = np.isfinite(coh)
        phase_mask = np.isfinite(phase)
        acc += np.where(mask, coh, 0.0)
        acc_sq += np.where(mask, coh * coh, 0.0)
        acc_count += mask.astype(float)
        if phase_sum is not None:
            phase_sum += np.where(phase_mask, np.exp(1j * phase), 0.0)
            phase_count += phase_mask.astype(float)
        count += 1

    if acc is None:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence": np.empty((0, 0)),
            "n_windows": 0,
        }

    coherence = acc / np.maximum(acc_count, 1.0)
    coherence[acc_count == 0] = np.nan
    variance = acc_sq / np.maximum(acc_count, 1.0) - np.square(coherence)
    variance = np.clip(variance, 0.0, None)
    coherence_std = np.sqrt(variance)
    coherence_std[acc_count == 0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        coherence_cv = coherence_std / coherence
        coherence_cv[~np.isfinite(coherence_cv)] = np.nan

    plv = None
    phase_mean = None
    if phase_sum is not None and phase_count is not None:
        plv = np.abs(phase_sum / np.maximum(phase_count, 1.0))
        plv[phase_count == 0] = np.nan
        phase_mean = np.angle(phase_sum)
        phase_mean[phase_count == 0] = np.nan

    return {
        "frequencies": freqs,
        "times": times,
        "coherence": coherence,
        "coherence_std": coherence_std,
        "coherence_cv": coherence_cv,
        "plv": plv if plv is not None else np.empty((0, 0)),
        "phase_mean": phase_mean if phase_mean is not None else np.empty((0, 0)),
        "n_windows": count,
    }


def run_coherence_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 50,
    nperseg: int = 64,
    *,
    fs: float = 4.0,
    max_detailed_samples: int = 5,
    include_up_coherence: bool = False,
    time_frequency_method: str = "both",
    time_frequency_nperseg: int = 128,
    time_frequency_num_scales: int = 50,
    time_frequency_min_freq: Optional[float] = None,
    time_frequency_max_freq: float = 0.5,
    time_frequency_early_seconds: float = 30.0,
    time_frequency_late_seconds: float = 30.0,
    time_frequency_permutations: int = 0,
    time_frequency_permutation_alpha: float = 0.05,
    time_frequency_permutation_seed: Optional[int] = None,
    time_frequency_permutation_max_samples: Optional[int] = None,
    wavelet_pad_mode: Optional[str] = "reflect",
    wavelet_pad_max_fraction: float = 0.25,
    wavelet_coi_scale: float = 1.65,
    wavelet_apply_coi_mask: bool = True,
    window_nperseg: int = 128,
    window_relative_nperseg: int = 128,
    max_window_timefreq_windows: Optional[int] = None,
    bandpower_nperseg: int = 128,
    psd_nperseg: int = 256,
    max_corr_lag_sec: float = 120.0,
) -> Dict[str, Any]:
    """
    Run complete FHR reconstruction coherence analysis.

    Computes coherence between original FHR and reconstructed FHR, along with
    per-window and within-window time-frequency diagnostics. Optionally computes
    UP-FHR coherence to assess physiological coupling preservation. Also records
    PSD and cross-correlation summaries for reconstruction quality.

    Args:
        runner: TestRunner with model and device configured.
        loader: PyTorch DataLoader for test data.
        max_samples: Maximum samples to process (default 50).
        nperseg: Segment length for STFT coherence (default 64).
        fs: Sampling frequency in Hz (default 4.0).
        max_detailed_samples: Number of per-sample plots to generate.
        include_up_coherence: If True and UP is available, compute UP-FHR coherence.
        time_frequency_method: "stft", "wavelet", or "both" for time-frequency plots.
        time_frequency_nperseg: STFT segment length for time-frequency plots.
        time_frequency_num_scales: Number of scales for wavelet coherence.
        time_frequency_min_freq: Minimum frequency for wavelet scale selection.
        time_frequency_max_freq: Max frequency to display in time-frequency plots.
        time_frequency_early_seconds: Early-horizon window length for spectra comparisons.
        time_frequency_late_seconds: Late-horizon window length for spectra comparisons.
        time_frequency_permutations: Number of permutation samples for TF CIs (0 disables).
        time_frequency_permutation_alpha: Significance level for permutation CIs.
        time_frequency_permutation_seed: RNG seed for permutation sampling.
        time_frequency_permutation_max_samples: Max samples to run permutations for.
        wavelet_pad_mode: Padding mode for wavelet coherence (e.g. "reflect").
        wavelet_pad_max_fraction: Max padding as fraction of window length.
        wavelet_coi_scale: COI multiplier for edge masking in wavelet maps.
        wavelet_apply_coi_mask: Whether to mask COI regions with NaN.
        window_nperseg: Welch segment length for per-window (2-minute) coherence.
        window_relative_nperseg: STFT segment length for within-window coherence.
        max_window_timefreq_windows: Optional limit on windows for relative maps.
        bandpower_nperseg: Welch segment length for bandpower per window.
        psd_nperseg: Welch segment length for PSD comparison.
        max_corr_lag_sec: Max lag for cross-correlation plot (seconds).

    Returns:
        Dict with:
            - 'frequencies': Frequency array
            - 'recon_coherence_mean': Mean FHR reconstruction coherence
            - 'recon_coherence_std': Std FHR reconstruction coherence
            - 'n_samples': Number of samples analyzed
            - optional UP-FHR coherence arrays if include_up_coherence is True

    Example:
        >>> results = run_coherence_analysis(runner, test_loader)
        >>> print(f"Mean coherence preserved: {results['recon_coherence_mean'].mean():.3f}")
    """

    logger.info(f"Running FHR reconstruction coherence analysis (max {max_samples} samples)...")

    recon_coherence_list: List[np.ndarray] = []
    up_coherence_original_list: List[np.ndarray] = []
    up_coherence_recon_list: List[np.ndarray] = []
    psd_orig_list: List[np.ndarray] = []
    psd_recon_list: List[np.ndarray] = []
    psd_resid_list: List[np.ndarray] = []
    corr_list: List[np.ndarray] = []
    coherence_thresholds: List[float] = []
    frequencies = None
    up_frequencies = None
    psd_frequencies = None
    corr_lags_sec = None
    processed = 0
    detailed_saved = 0
    epoch_records: List[Dict[str, Any]] = []
    window_records: List[Dict[str, Any]] = []
    relative_records: List[Dict[str, Any]] = []
    relative_records_wavelet: List[Dict[str, Any]] = []
    relative_tf_mean_list: List[np.ndarray] = []
    relative_tf_std_list: List[np.ndarray] = []
    relative_tf_cv_list: List[np.ndarray] = []
    relative_tf_wavelet_mean_list: List[np.ndarray] = []
    relative_tf_wavelet_std_list: List[np.ndarray] = []
    relative_tf_wavelet_cv_list: List[np.ndarray] = []
    relative_tf_wavelet_plv_list: List[np.ndarray] = []
    relative_tf_wavelet_phase_list: List[np.ndarray] = []
    horizon_early_list: List[np.ndarray] = []
    horizon_late_list: List[np.ndarray] = []
    horizon_delta_list: List[np.ndarray] = []
    horizon_early_wavelet_list: List[np.ndarray] = []
    horizon_late_wavelet_list: List[np.ndarray] = []
    horizon_delta_wavelet_list: List[np.ndarray] = []
    permutation_lower_list: List[np.ndarray] = []
    permutation_upper_list: List[np.ndarray] = []
    permutation_lower_wavelet_list: List[np.ndarray] = []
    permutation_upper_wavelet_list: List[np.ndarray] = []
    bandpower_records: List[Dict[str, Any]] = []
    tf_methods = _resolve_time_frequency_methods(time_frequency_method)
    rng = np.random.default_rng(time_frequency_permutation_seed)
    relative_tf_template = None
    relative_tf_wavelet_template = None
    perm_sample_limit = (
        max_detailed_samples
        if time_frequency_permutation_max_samples is None
        else int(time_frequency_permutation_max_samples)
    )

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = batch.fhr_st.size(0)

            has_up = hasattr(batch, "up") and batch.up is not None
            if include_up_coherence and not has_up:
                logger.warning("UP signal not available in batch - skipping UP-FHR coherence.")

            # Forward pass
            outputs = runner.forward(batch)
            mu_pr = outputs.get("mu_pr")

            if mu_pr is None:
                continue

            # Aggregate predictions
            avg_pred, valid_mask = aggregate_predictions(
                runner.model, mu_pr, raw_len=batch.fhr.size(1)
            )

            if avg_pred is None:
                continue

            # Process each sample
            for idx in range(batch_size):
                if max_samples and processed >= max_samples:
                    break

                guid = _extract_guid(batch, idx)
                epoch = _extract_epoch(batch, idx)
                label = _extract_label(batch, idx)

                fhr_orig_full = batch.fhr[idx].cpu().numpy()
                fhr_recon_full = avg_pred[idx].cpu().numpy()
                up_full = batch.up[idx].cpu().numpy() if has_up else None

                fhr_orig = fhr_orig_full
                fhr_recon = fhr_recon_full
                up = up_full if include_up_coherence and up_full is not None else None
                mask = valid_mask[idx].cpu().numpy() if valid_mask is not None else None

                if mask is not None and mask.any():
                    start = int(np.argmax(mask))
                    end = int(len(mask) - np.argmax(mask[::-1]))
                    fhr_orig = fhr_orig[start:end]
                    fhr_recon = fhr_recon[start:end]
                    if up is not None:
                        up = up[start:end]

                min_len = min(len(fhr_orig), len(fhr_recon))
                if up is not None:
                    min_len = min(min_len, len(up))
                if min_len < nperseg:
                    logger.warning(
                        f"Sample {idx} too short for coherence (len={min_len}, nperseg={nperseg}); skipping."
                    )
                    continue

                k_segments = _welch_segment_count(min_len, nperseg, nperseg // 2)
                threshold = _coherence_significance_threshold(k_segments, time_frequency_permutation_alpha)
                if np.isfinite(threshold):
                    coherence_thresholds.append(threshold)

                fhr_orig = fhr_orig[:min_len]
                fhr_recon = fhr_recon[:min_len]
                if up is not None:
                    up = up[:min_len]

                window_tf = {
                    "frequencies": np.array([]),
                    "times": np.array([]),
                    "window_indices": np.array([]),
                    "coherence": np.empty((0, 0)),
                }
                relative_tf = {
                    "frequencies": np.array([]),
                    "times": np.array([]),
                    "coherence": np.empty((0, 0)),
                    "n_windows": 0,
                }
                relative_tf_wavelet = {
                    "frequencies": np.array([]),
                    "times": np.array([]),
                    "coherence": np.empty((0, 0)),
                    "n_windows": 0,
                }
                sample_window_band = None
                sample_relative_band = None
                sample_relative_band_wavelet = None
                sample_relative_perm_stft = None
                sample_relative_perm_wavelet = None
                sample_bandpower = None

                # Compute coherence
                try:
                    freq, coh_recon = compute_stft_coherence(fhr_orig, fhr_recon, fs=fs, nperseg=nperseg)
                    recon_coherence_list.append(coh_recon)

                    if frequencies is None:
                        frequencies = freq

                    band_recon = _band_means(freq, coh_recon)
                    epoch_record = {
                        "guid": guid,
                        "epoch": epoch,
                        "label": label,
                        "sample_idx": processed,
                    }
                    for band in COHERENCE_BANDS:
                        name = band[0]
                        epoch_record[f"{name}_coherence"] = float(band_recon.get(name, np.nan))

                    if include_up_coherence and up is not None:
                        up_freq, coh_up_orig = compute_stft_coherence(up, fhr_orig, fs=fs, nperseg=nperseg)
                        _, coh_up_recon = compute_stft_coherence(up, fhr_recon, fs=fs, nperseg=nperseg)

                        if up_frequencies is None:
                            up_frequencies = up_freq
                        elif up_freq.shape != up_frequencies.shape or not np.allclose(up_freq, up_frequencies):
                            logger.warning("UP coherence frequency mismatch; skipping UP-FHR coherence for this sample.")
                        else:
                            up_coherence_original_list.append(coh_up_orig)
                            up_coherence_recon_list.append(coh_up_recon)
                            up_band_orig = _band_means(up_freq, coh_up_orig)
                            up_band_recon = _band_means(up_freq, coh_up_recon)
                            for band in COHERENCE_BANDS:
                                name = band[0]
                                epoch_record[f"{name}_up_orig"] = float(up_band_orig.get(name, np.nan))
                                epoch_record[f"{name}_up_recon"] = float(up_band_recon.get(name, np.nan))
                                epoch_record[f"{name}_up_delta"] = float(
                                    up_band_recon.get(name, np.nan) - up_band_orig.get(name, np.nan)
                                )
                    epoch_records.append(epoch_record)

                    mu_pr_window = mu_pr[idx].detach().cpu().numpy()
                    window_tf = compute_windowed_coherence_map(
                        fhr_orig_full,
                        mu_pr_window,
                        fs=fs,
                        stride=runner.decimation_factor,
                        warmup=runner.warmup_steps,
                        nperseg=window_nperseg,
                    )
                    if window_tf["coherence"].size > 0:
                        window_band = _band_means(window_tf["frequencies"], window_tf["coherence"])
                        sample_window_band = window_band
                        window_indices = window_tf.get("window_indices", np.arange(window_tf["coherence"].shape[1]))
                        for band_name, vals in window_band.items():
                            baseline = float(band_recon.get(band_name, np.nan))
                            for w_idx, window_index in enumerate(window_indices):
                                window_start_sample = int(window_index) * int(runner.decimation_factor)
                                coherence_val = float(vals[w_idx]) if w_idx < len(vals) else np.nan
                                delta = (
                                    coherence_val - baseline
                                    if np.isfinite(coherence_val) and np.isfinite(baseline)
                                    else np.nan
                                )
                                window_records.append({
                                    "guid": guid,
                                    "epoch": epoch,
                                    "label": label,
                                    "sample_idx": processed,
                                    "band": band_name,
                                    "window_index": int(window_index),
                                    "window_start_sample": window_start_sample,
                                    "window_start_sec": window_start_sample / fs,
                                    "window_center_sec": float(window_tf["times"][w_idx]) if w_idx < len(window_tf["times"]) else np.nan,
                                    "coherence": coherence_val,
                                    "coherence_baseline": baseline,
                                    "coherence_delta": delta,
                                })

                        relative_tf = compute_window_relative_time_frequency(
                            fhr_orig_full,
                            mu_pr_window,
                            fs=fs,
                            stride=runner.decimation_factor,
                            warmup=runner.warmup_steps,
                            nperseg=window_relative_nperseg,
                            max_windows=max_window_timefreq_windows,
                        )
                        if relative_tf["coherence"].size > 0:
                            relative_band = _band_means(relative_tf["frequencies"], relative_tf["coherence"])
                            sample_relative_band = relative_band
                            if relative_tf_template is None:
                                relative_tf_template = {
                                    "frequencies": relative_tf["frequencies"],
                                    "times": relative_tf["times"],
                                }
                            relative_tf_mean_list.append(relative_tf["coherence"])
                            if "coherence_std" in relative_tf:
                                relative_tf_std_list.append(relative_tf["coherence_std"])
                            if "coherence_cv" in relative_tf:
                                relative_tf_cv_list.append(relative_tf["coherence_cv"])
                            horizon = _compute_horizon_spectra(
                                relative_tf["frequencies"],
                                relative_tf["times"],
                                relative_tf["coherence"],
                                early_seconds=time_frequency_early_seconds,
                                late_seconds=time_frequency_late_seconds,
                            )
                            if horizon["early"].size:
                                horizon_early_list.append(horizon["early"])
                                horizon_late_list.append(horizon["late"])
                                horizon_delta_list.append(horizon["delta"])
                            for band_name, vals in relative_band.items():
                                baseline = float(band_recon.get(band_name, np.nan))
                                for t_idx, rel_time in enumerate(relative_tf["times"]):
                                    coherence_val = float(vals[t_idx]) if t_idx < len(vals) else np.nan
                                    delta = (
                                        coherence_val - baseline
                                        if np.isfinite(coherence_val) and np.isfinite(baseline)
                                        else np.nan
                                    )
                                    relative_records.append({
                                        "guid": guid,
                                        "epoch": epoch,
                                        "label": label,
                                        "sample_idx": processed,
                                        "band": band_name,
                                        "relative_frame_index": int(t_idx),
                                        "relative_time_sec": float(rel_time),
                                        "coherence": coherence_val,
                                        "coherence_baseline": baseline,
                                        "coherence_delta": delta,
                                    })
                        if "wavelet" in tf_methods:
                            try:
                                relative_tf_wavelet = compute_window_relative_time_frequency_wavelet(
                                    fhr_orig_full,
                                    mu_pr_window,
                                    fs=fs,
                                    stride=runner.decimation_factor,
                                    warmup=runner.warmup_steps,
                                    num_scales=time_frequency_num_scales,
                                    min_freq=time_frequency_min_freq,
                                    max_freq=time_frequency_max_freq,
                                    pad_mode=wavelet_pad_mode,
                                    pad_max_fraction=wavelet_pad_max_fraction,
                                    coi_scale=wavelet_coi_scale,
                                    apply_coi_mask=wavelet_apply_coi_mask,
                                    max_windows=max_window_timefreq_windows,
                                )
                            except ImportError as e:
                                logger.warning(f"Wavelet coherence skipped: {e}")
                            except Exception as e:
                                logger.warning(f"Wavelet coherence failed for {guid}: {e}")
                            else:
                                if relative_tf_wavelet["coherence"].size > 0:
                                    relative_band_wavelet = _band_means(
                                        relative_tf_wavelet["frequencies"],
                                        relative_tf_wavelet["coherence"],
                                    )
                                    sample_relative_band_wavelet = relative_band_wavelet
                                    if relative_tf_wavelet_template is None:
                                        relative_tf_wavelet_template = {
                                            "frequencies": relative_tf_wavelet["frequencies"],
                                            "times": relative_tf_wavelet["times"],
                                        }
                                    relative_tf_wavelet_mean_list.append(relative_tf_wavelet["coherence"])
                                    if "coherence_std" in relative_tf_wavelet:
                                        relative_tf_wavelet_std_list.append(relative_tf_wavelet["coherence_std"])
                                    if "coherence_cv" in relative_tf_wavelet:
                                        relative_tf_wavelet_cv_list.append(relative_tf_wavelet["coherence_cv"])
                                    plv_map = relative_tf_wavelet.get("plv")
                                    if plv_map is not None and plv_map.size:
                                        relative_tf_wavelet_plv_list.append(plv_map)
                                    phase_mean_map = relative_tf_wavelet.get("phase_mean")
                                    if phase_mean_map is not None and phase_mean_map.size:
                                        relative_tf_wavelet_phase_list.append(phase_mean_map)
                                    horizon_wavelet = _compute_horizon_spectra(
                                        relative_tf_wavelet["frequencies"],
                                        relative_tf_wavelet["times"],
                                        relative_tf_wavelet["coherence"],
                                        early_seconds=time_frequency_early_seconds,
                                        late_seconds=time_frequency_late_seconds,
                                    )
                                    if horizon_wavelet["early"].size:
                                        horizon_early_wavelet_list.append(horizon_wavelet["early"])
                                        horizon_late_wavelet_list.append(horizon_wavelet["late"])
                                        horizon_delta_wavelet_list.append(horizon_wavelet["delta"])
                                    for band_name, vals in relative_band_wavelet.items():
                                        baseline = float(band_recon.get(band_name, np.nan))
                                        for t_idx, rel_time in enumerate(relative_tf_wavelet["times"]):
                                            coherence_val = float(vals[t_idx]) if t_idx < len(vals) else np.nan
                                            delta = (
                                                coherence_val - baseline
                                                if np.isfinite(coherence_val) and np.isfinite(baseline)
                                                else np.nan
                                            )
                                            relative_records_wavelet.append({
                                                "guid": guid,
                                                "epoch": epoch,
                                                "label": label,
                                                "sample_idx": processed,
                                                "band": band_name,
                                                "relative_frame_index": int(t_idx),
                                                "relative_time_sec": float(rel_time),
                                                "coherence": coherence_val,
                                                "coherence_baseline": baseline,
                                                "coherence_delta": delta,
                                            })

                        if time_frequency_permutations > 0 and processed < perm_sample_limit:
                            windows = _collect_prediction_windows(
                                fhr_orig_full,
                                mu_pr_window,
                                stride=runner.decimation_factor,
                                warmup=runner.warmup_steps,
                                max_windows=max_window_timefreq_windows,
                            )
                            if windows:
                                if "stft" in tf_methods:
                                    perm_ci = _compute_permutation_ci(
                                        windows,
                                        map_fn=lambda a, b: compute_stft_coherence_map(
                                            a, b, fs=fs, nperseg=window_relative_nperseg
                                        ),
                                        num_permutations=time_frequency_permutations,
                                        alpha=time_frequency_permutation_alpha,
                                        rng=rng,
                                    )
                                    if perm_ci["upper"].size:
                                        permutation_lower_list.append(perm_ci["lower"])
                                        permutation_upper_list.append(perm_ci["upper"])
                                        sample_relative_perm_stft = perm_ci
                                if "wavelet" in tf_methods:
                                    try:
                                        perm_ci_wavelet = _compute_permutation_ci(
                                            windows,
                                            map_fn=lambda a, b: compute_wavelet_coherence(
                                                a,
                                                b,
                                                fs=fs,
                                                num_scales=time_frequency_num_scales,
                                                min_freq=time_frequency_min_freq,
                                                max_freq=time_frequency_max_freq,
                                                pad_mode=wavelet_pad_mode,
                                                pad_max_fraction=wavelet_pad_max_fraction,
                                                coi_scale=wavelet_coi_scale,
                                                apply_coi_mask=wavelet_apply_coi_mask,
                                            ),
                                            num_permutations=time_frequency_permutations,
                                            alpha=time_frequency_permutation_alpha,
                                            rng=rng,
                                        )
                                    except ImportError as e:
                                        logger.warning(f"Wavelet permutation CIs skipped: {e}")
                                    else:
                                        if perm_ci_wavelet["upper"].size:
                                            permutation_lower_wavelet_list.append(perm_ci_wavelet["lower"])
                                            permutation_upper_wavelet_list.append(perm_ci_wavelet["upper"])
                                            sample_relative_perm_wavelet = perm_ci_wavelet

                    # Per-window bandpower analysis
                    sample_bandpower_rows: List[Dict[str, Any]] = []
                    T_win, H_win = mu_pr_window.shape
                    for t in range(runner.warmup_steps, T_win):
                        start = t * int(runner.decimation_factor)
                        end = start + H_win
                        if end > len(fhr_orig_full):
                            break

                        ref_win = fhr_orig_full[start:end]
                        pred_win = mu_pr_window[t]
                        bandpower_orig = compute_bandpower(
                            ref_win, fs=fs, bands=COHERENCE_BANDS, nperseg=bandpower_nperseg
                        )
                        bandpower_recon = compute_bandpower(
                            pred_win, fs=fs, bands=COHERENCE_BANDS, nperseg=bandpower_nperseg
                        )

                        for band_name in bandpower_orig.keys():
                            bp_orig = bandpower_orig.get(band_name, np.nan)
                            bp_recon = bandpower_recon.get(band_name, np.nan)
                            record = {
                                "guid": guid,
                                "epoch": epoch,
                                "label": label,
                                "sample_idx": processed,
                                "band": band_name,
                                "window_index": int(t),
                                "window_start_sample": int(start),
                                "window_start_sec": start / fs,
                                "bandpower_original": bp_orig,
                                "bandpower_reconstructed": bp_recon,
                                "bandpower_delta": bp_recon - bp_orig if np.isfinite(bp_orig) and np.isfinite(bp_recon) else np.nan,
                            }
                            bandpower_records.append(record)
                            sample_bandpower_rows.append(record)

                    if sample_bandpower_rows:
                        sample_bandpower = pd.DataFrame(sample_bandpower_rows)

                    psd_freq, psd_orig = compute_welch_psd(fhr_orig, fs=fs, nperseg=psd_nperseg)
                    _, psd_recon = compute_welch_psd(fhr_recon, fs=fs, nperseg=psd_nperseg)
                    _, psd_resid = compute_welch_psd(fhr_orig - fhr_recon, fs=fs, nperseg=psd_nperseg)
                    if psd_freq.size > 0:
                        if psd_frequencies is None:
                            psd_frequencies = psd_freq
                            psd_orig_list.append(psd_orig)
                            psd_recon_list.append(psd_recon)
                            psd_resid_list.append(psd_resid)
                        elif psd_freq.shape == psd_frequencies.shape and np.allclose(psd_freq, psd_frequencies):
                            psd_orig_list.append(psd_orig)
                            psd_recon_list.append(psd_recon)
                            psd_resid_list.append(psd_resid)

                    lags_sec, corr = compute_cross_correlation(
                        fhr_orig, fhr_recon, fs=fs, max_lag_sec=max_corr_lag_sec
                    )
                    if lags_sec.size > 0:
                        if corr_lags_sec is None:
                            corr_lags_sec = lags_sec
                            corr_list.append(corr)
                        elif lags_sec.shape == corr_lags_sec.shape and np.allclose(lags_sec, corr_lags_sec):
                            corr_list.append(corr)

                    processed += 1
                except Exception as e:
                    logger.warning(f"Coherence computation failed for sample {idx}: {e}")
                    continue

                if detailed_saved < max_detailed_samples:
                    output_dir = runner.ensure_dir("coherence")
                    sample_dir = output_dir / "samples"
                    sample_dir.mkdir(parents=True, exist_ok=True)

                    guid = getattr(batch, "guid", None)
                    guid_val = None
                    if guid is not None:
                        try:
                            guid_val = guid[idx]
                        except Exception:
                            guid_val = None
                    guid_text = str(guid_val) if guid_val is not None else "unknown"
                    sample_name = f"sample_{processed:03d}"

                    signal_title = f"{sample_name} | GUID={guid_text}"
                    plot_coherence_signals(
                        up,
                        fhr_orig,
                        fhr_recon,
                        sample_dir / f"{sample_name}_signals.svg",
                        fs=fs,
                        title=signal_title,
                    )

                    for method in tf_methods:
                        try:
                            if method == "wavelet":
                                tf_map = compute_wavelet_coherence(
                                    fhr_orig,
                                    fhr_recon,
                                    fs=fs,
                                    num_scales=time_frequency_num_scales,
                                    min_freq=time_frequency_min_freq,
                                    max_freq=time_frequency_max_freq,
                                    pad_mode=wavelet_pad_mode,
                                    pad_max_fraction=wavelet_pad_max_fraction,
                                    coi_scale=wavelet_coi_scale,
                                    apply_coi_mask=wavelet_apply_coi_mask,
                                )
                            else:
                                tf_map = compute_stft_coherence_map(
                                    fhr_orig,
                                    fhr_recon,
                                    fs=fs,
                                    nperseg=time_frequency_nperseg,
                                )

                            if tf_map["coherence"].size == 0:
                                logger.warning(
                                    "Time-frequency coherence empty for %s (%s); skipping plot.",
                                    sample_name,
                                    method,
                                )
                                continue

                            suffix = "wavelet" if method == "wavelet" else "stft"
                            plot_time_frequency_coherence(
                                tf_map["frequencies"],
                                tf_map["times"],
                                tf_map["coherence"],
                                output_path=sample_dir / f"{sample_name}_time_frequency_{suffix}.svg",
                                max_freq=time_frequency_max_freq,
                                title=f"{signal_title} ({suffix.upper()})",
                            )
                        except ImportError as e:
                            logger.warning(f"Time-frequency coherence skipped ({method}): {e}")
                        except Exception as e:
                            logger.warning(f"Time-frequency coherence failed for {sample_name} ({method}): {e}")

                    try:
                        if window_tf["coherence"].size == 0:
                            logger.warning(f"Windowed coherence empty for {sample_name}; skipping plot.")
                        else:
                            plot_time_frequency_coherence(
                                window_tf["frequencies"],
                                window_tf["times"],
                                window_tf["coherence"],
                                output_path=sample_dir / f"{sample_name}_windowed_coherence.svg",
                                max_freq=time_frequency_max_freq,
                                title=f"{signal_title} (2-min window coherence)",
                            )
                            if sample_window_band is not None:
                                window_band_df = pd.DataFrame([
                                    {
                                        "band": band_name,
                                        "window_index": int(window_idx),
                                        "coherence_mean": float(val) if w_idx < len(vals) else np.nan,
                                    }
                                    for band_name, vals in sample_window_band.items()
                                    for w_idx, window_idx in enumerate(
                                        window_tf.get(
                                            "window_indices",
                                            np.arange(window_tf["coherence"].shape[1]),
                                        )
                                    )
                                    for val in [vals[w_idx]] if w_idx < len(vals)
                                ])
                                if not window_band_df.empty:
                                    _plot_band_trends(
                                        window_band_df,
                                        sample_dir / f"{sample_name}_window_band_coherence.svg",
                                        x_col="window_index",
                                        x_label="Window index",
                                        title=f"{signal_title} (band coherence vs window index)",
                                        y_label="Coherence",
                                    )
                                    if band_recon:
                                        delta_rows = []
                                        for band_name, vals in sample_window_band.items():
                                            baseline = float(band_recon.get(band_name, np.nan))
                                            for w_idx, window_idx in enumerate(
                                                window_tf.get(
                                                    "window_indices",
                                                    np.arange(window_tf["coherence"].shape[1]),
                                                )
                                            ):
                                                if w_idx >= len(vals):
                                                    continue
                                                val = float(vals[w_idx])
                                                if not np.isfinite(val) or not np.isfinite(baseline):
                                                    continue
                                                delta_rows.append({
                                                    "band": band_name,
                                                    "window_index": int(window_idx),
                                                    "coherence_mean": val - baseline,
                                                })
                                        if delta_rows:
                                            window_delta_df = pd.DataFrame(delta_rows)
                                            _plot_band_trends(
                                                window_delta_df,
                                                sample_dir / f"{sample_name}_window_band_coherence_delta.svg",
                                                x_col="window_index",
                                                x_label="Window index",
                                                title=f"{signal_title} (delta coherence vs window index)",
                                                y_label="Delta coherence",
                                                single_label="Delta coherence",
                                            )
                    except Exception as e:
                        logger.warning(f"Windowed coherence failed for {sample_name}: {e}")

                    try:
                        if relative_tf["coherence"].size == 0:
                            logger.warning(f"Relative window coherence empty for {sample_name}; skipping STFT plot.")
                        else:
                            plot_time_frequency_coherence(
                                relative_tf["frequencies"],
                                relative_tf["times"],
                                relative_tf["coherence"],
                                output_path=sample_dir / f"{sample_name}_relative_window_time_frequency_stft.svg",
                                max_freq=time_frequency_max_freq,
                                title=f"{signal_title} (window-relative coherence, STFT)",
                            )
                            if relative_tf.get("coherence_std") is not None:
                                plot_time_frequency_map(
                                    relative_tf["frequencies"],
                                    relative_tf["times"],
                                    relative_tf["coherence_std"],
                                    output_path=sample_dir / f"{sample_name}_relative_window_time_frequency_stft_std.svg",
                                    max_freq=time_frequency_max_freq,
                                    title=f"{signal_title} (window-relative coherence std, STFT)",
                                    cmap="viridis",
                                    colorbar_label="Std",
                                )
                            if relative_tf.get("coherence_cv") is not None:
                                plot_time_frequency_map(
                                    relative_tf["frequencies"],
                                    relative_tf["times"],
                                    relative_tf["coherence_cv"],
                                    output_path=sample_dir / f"{sample_name}_relative_window_time_frequency_stft_cv.svg",
                                    max_freq=time_frequency_max_freq,
                                    title=f"{signal_title} (window-relative coherence CV, STFT)",
                                    cmap="magma",
                                    colorbar_label="CV",
                                )
                            if sample_relative_band is not None:
                                relative_band_df = pd.DataFrame([
                                    {
                                        "band": band_name,
                                        "relative_frame_index": int(t_idx),
                                        "coherence_mean": float(val) if t_idx < len(vals) else np.nan,
                                    }
                                    for band_name, vals in sample_relative_band.items()
                                    for t_idx, val in enumerate(vals)
                                ])
                                if not relative_band_df.empty:
                                    _plot_band_trends(
                                        relative_band_df,
                                        sample_dir / f"{sample_name}_relative_band_coherence.svg",
                                        x_col="relative_frame_index",
                                        x_label="STFT frame index",
                                        title=f"{signal_title} (band coherence vs frame index)",
                                        y_label="Coherence",
                                    )
                                    if band_recon:
                                        delta_rows = []
                                        for band_name, vals in sample_relative_band.items():
                                            baseline = float(band_recon.get(band_name, np.nan))
                                            for t_idx, val in enumerate(vals):
                                                val_f = float(val)
                                                if not np.isfinite(val_f) or not np.isfinite(baseline):
                                                    continue
                                                delta_rows.append({
                                                    "band": band_name,
                                                    "relative_frame_index": int(t_idx),
                                                    "coherence_mean": val_f - baseline,
                                                })
                                        if delta_rows:
                                            relative_delta_df = pd.DataFrame(delta_rows)
                                            _plot_band_trends(
                                                relative_delta_df,
                                                sample_dir / f"{sample_name}_relative_band_coherence_delta.svg",
                                                x_col="relative_frame_index",
                                                x_label="STFT frame index",
                                                title=f"{signal_title} (delta coherence vs frame index)",
                                                y_label="Delta coherence",
                                                single_label="Delta coherence",
                                            )
                    except Exception as e:
                        logger.warning(f"Relative window coherence failed for {sample_name}: {e}")

                    try:
                        if "wavelet" in tf_methods:
                            if relative_tf_wavelet["coherence"].size == 0:
                                logger.warning(
                                    f"Relative window coherence empty for {sample_name}; skipping wavelet plot."
                                )
                            else:
                                plot_time_frequency_coherence(
                                    relative_tf_wavelet["frequencies"],
                                    relative_tf_wavelet["times"],
                                    relative_tf_wavelet["coherence"],
                                    output_path=sample_dir / f"{sample_name}_relative_window_time_frequency_wavelet.svg",
                                    max_freq=time_frequency_max_freq,
                                    title=f"{signal_title} (window-relative coherence, WAVELET)",
                                )
                                if relative_tf_wavelet.get("coherence_std") is not None:
                                    plot_time_frequency_map(
                                        relative_tf_wavelet["frequencies"],
                                        relative_tf_wavelet["times"],
                                        relative_tf_wavelet["coherence_std"],
                                        output_path=sample_dir / f"{sample_name}_relative_window_time_frequency_wavelet_std.svg",
                                        max_freq=time_frequency_max_freq,
                                        title=f"{signal_title} (window-relative coherence std, WAVELET)",
                                        cmap="viridis",
                                        colorbar_label="Std",
                                    )
                                if relative_tf_wavelet.get("coherence_cv") is not None:
                                    plot_time_frequency_map(
                                        relative_tf_wavelet["frequencies"],
                                        relative_tf_wavelet["times"],
                                        relative_tf_wavelet["coherence_cv"],
                                        output_path=sample_dir / f"{sample_name}_relative_window_time_frequency_wavelet_cv.svg",
                                        max_freq=time_frequency_max_freq,
                                        title=f"{signal_title} (window-relative coherence CV, WAVELET)",
                                        cmap="magma",
                                        colorbar_label="CV",
                                    )
                                if relative_tf_wavelet.get("plv") is not None:
                                    plot_time_frequency_map(
                                        relative_tf_wavelet["frequencies"],
                                        relative_tf_wavelet["times"],
                                        relative_tf_wavelet["plv"],
                                        output_path=sample_dir / f"{sample_name}_relative_window_wavelet_plv.svg",
                                        max_freq=time_frequency_max_freq,
                                        title=f"{signal_title} (wavelet PLV)",
                                        cmap="viridis",
                                        vmin=0.0,
                                        vmax=1.0,
                                        colorbar_label="PLV",
                                    )
                                if relative_tf_wavelet.get("phase_mean") is not None:
                                    plot_time_frequency_map(
                                        relative_tf_wavelet["frequencies"],
                                        relative_tf_wavelet["times"],
                                        relative_tf_wavelet["phase_mean"],
                                        output_path=sample_dir / f"{sample_name}_relative_window_wavelet_phase.svg",
                                        max_freq=time_frequency_max_freq,
                                        title=f"{signal_title} (wavelet phase mean)",
                                        cmap="twilight",
                                        vmin=-np.pi,
                                        vmax=np.pi,
                                        colorbar_label="Phase (rad)",
                                    )
                                if sample_relative_band_wavelet is not None:
                                    relative_band_df = pd.DataFrame([
                                        {
                                            "band": band_name,
                                            "relative_time_sec": float(rel_time),
                                            "coherence_mean": float(val) if t_idx < len(vals) else np.nan,
                                        }
                                        for band_name, vals in sample_relative_band_wavelet.items()
                                        for t_idx, (rel_time, val) in enumerate(
                                            zip(relative_tf_wavelet["times"], vals)
                                        )
                                    ])
                                    if not relative_band_df.empty:
                                        _plot_band_trends(
                                            relative_band_df,
                                            sample_dir / f"{sample_name}_relative_band_coherence_wavelet.svg",
                                            x_col="relative_time_sec",
                                            x_label="Time From Window Start (seconds)",
                                            title=f"{signal_title} (band coherence vs time, WAVELET)",
                                            y_label="Coherence",
                                        )
                                        if band_recon:
                                            delta_rows = []
                                            for band_name, vals in sample_relative_band_wavelet.items():
                                                baseline = float(band_recon.get(band_name, np.nan))
                                                for rel_time, val in zip(relative_tf_wavelet["times"], vals):
                                                    val_f = float(val)
                                                    if not np.isfinite(val_f) or not np.isfinite(baseline):
                                                        continue
                                                    delta_rows.append({
                                                        "band": band_name,
                                                        "relative_time_sec": float(rel_time),
                                                        "coherence_mean": val_f - baseline,
                                                    })
                                            if delta_rows:
                                                relative_delta_df = pd.DataFrame(delta_rows)
                                            _plot_band_trends(
                                                relative_delta_df,
                                                sample_dir / f"{sample_name}_relative_band_coherence_delta_wavelet.svg",
                                                x_col="relative_time_sec",
                                                x_label="Time From Window Start (seconds)",
                                                title=f"{signal_title} (delta coherence vs time, WAVELET)",
                                                y_label="Delta coherence",
                                                single_label="Delta coherence",
                                            )
                    except Exception as e:
                        logger.warning(f"Relative window wavelet coherence failed for {sample_name}: {e}")

                    try:
                        if sample_relative_perm_stft is not None:
                            plot_time_frequency_map(
                                sample_relative_perm_stft["frequencies"],
                                sample_relative_perm_stft["times"],
                                sample_relative_perm_stft["upper"],
                                output_path=sample_dir / f"{sample_name}_relative_window_perm_upper_stft.svg",
                                max_freq=time_frequency_max_freq,
                                title=f"{signal_title} (perm upper CI, STFT)",
                                cmap="viridis",
                                vmin=0.0,
                                vmax=1.0,
                                colorbar_label="Coherence",
                            )
                    except Exception as e:
                        logger.warning(f"Permutation CI plot failed for {sample_name} (STFT): {e}")

                    try:
                        if sample_relative_perm_wavelet is not None:
                            plot_time_frequency_map(
                                sample_relative_perm_wavelet["frequencies"],
                                sample_relative_perm_wavelet["times"],
                                sample_relative_perm_wavelet["upper"],
                                output_path=sample_dir / f"{sample_name}_relative_window_perm_upper_wavelet.svg",
                                max_freq=time_frequency_max_freq,
                                title=f"{signal_title} (perm upper CI, WAVELET)",
                                cmap="viridis",
                                vmin=0.0,
                                vmax=1.0,
                                colorbar_label="Coherence",
                            )
                    except Exception as e:
                        logger.warning(f"Permutation CI plot failed for {sample_name} (WAVELET): {e}")

                    try:
                        if mu_pr_window is not None and mu_pr_window.size > 0:
                            T_win, H_win = mu_pr_window.shape
                            single_index = runner.warmup_steps if runner.warmup_steps < T_win else 0
                            start = int(single_index) * int(runner.decimation_factor)
                            end = start + H_win
                            if end <= len(fhr_orig_full):
                                ref_win = fhr_orig_full[start:end]
                                pred_win = mu_pr_window[single_index][: len(ref_win)]

                                window_prefix = f"{sample_name}_single_window_{single_index}"
                                window_title = f"{signal_title} (window {single_index}, {H_win} samples)"

                                plot_coherence_signals(
                                    None,
                                    ref_win,
                                    pred_win,
                                    sample_dir / f"{window_prefix}_signals.svg",
                                    fs=fs,
                                    title=window_title,
                                )

                                nperseg_used = min(window_nperseg, len(ref_win))
                                if nperseg_used >= 2:
                                    freq_win, coh_win = compute_stft_coherence(
                                        ref_win,
                                        pred_win,
                                        fs=fs,
                                        nperseg=nperseg_used,
                                    )
                                    if freq_win.size > 0:
                                        plot_coherence_spectrum(
                                            freq_win,
                                            coh_win,
                                            sample_dir / f"{window_prefix}_coherence_spectrum.svg",
                                            title=f"{window_title} (coherence spectrum)",
                                            max_freq=time_frequency_max_freq,
                                        )

                                for method in tf_methods:
                                    try:
                                        if method == "wavelet":
                                            tf_map = compute_wavelet_coherence(
                                                ref_win,
                                                pred_win,
                                                fs=fs,
                                                num_scales=time_frequency_num_scales,
                                                min_freq=time_frequency_min_freq,
                                                max_freq=time_frequency_max_freq,
                                                pad_mode=wavelet_pad_mode,
                                                pad_max_fraction=wavelet_pad_max_fraction,
                                                coi_scale=wavelet_coi_scale,
                                                apply_coi_mask=wavelet_apply_coi_mask,
                                            )
                                        else:
                                            tf_map = compute_stft_coherence_map(
                                                ref_win,
                                                pred_win,
                                                fs=fs,
                                                nperseg=time_frequency_nperseg,
                                            )
                                    except ImportError as e:
                                        logger.warning(f"Single-window wavelet coherence skipped: {e}")
                                        continue
                                    except Exception as e:
                                        logger.warning(
                                            f"Single-window time-frequency coherence failed for {sample_name}: {e}"
                                        )
                                        continue

                                    if tf_map["coherence"].size == 0:
                                        continue
                                    suffix = "wavelet" if method == "wavelet" else "stft"
                                    plot_time_frequency_coherence(
                                        tf_map["frequencies"],
                                        tf_map["times"],
                                        tf_map["coherence"],
                                        output_path=sample_dir / f"{window_prefix}_time_frequency_{suffix}.svg",
                                        max_freq=time_frequency_max_freq,
                                        title=f"{window_title} (time-frequency coherence, {suffix.upper()})",
                                    )

                                psd_freq, psd_orig = compute_welch_psd(ref_win, fs=fs, nperseg=psd_nperseg)
                                _, psd_recon = compute_welch_psd(pred_win, fs=fs, nperseg=psd_nperseg)
                                if psd_freq.size > 0:
                                    plot_psd_comparison(
                                        psd_freq,
                                        psd_orig,
                                        np.zeros_like(psd_orig),
                                        psd_recon,
                                        np.zeros_like(psd_recon),
                                        sample_dir,
                                        filename=f"{window_prefix}_psd.svg",
                                    )

                                lags_sec, corr = compute_cross_correlation(
                                    ref_win, pred_win, fs=fs, max_lag_sec=max_corr_lag_sec
                                )
                                if lags_sec.size > 0:
                                    plot_cross_correlation(
                                        lags_sec,
                                        corr,
                                        np.zeros_like(corr),
                                        sample_dir,
                                        filename=f"{window_prefix}_cross_correlation.svg",
                                    )
                    except Exception as e:
                        logger.warning(f"Single-window coherence analysis failed for {sample_name}: {e}")

                    try:
                        if sample_bandpower is not None and not sample_bandpower.empty:
                            sample_bandpower = sample_bandpower.copy()
                            sample_bandpower.rename(columns={
                                "bandpower_original": "orig_mean",
                                "bandpower_reconstructed": "recon_mean",
                            }, inplace=True)
                            _plot_band_trends(
                                sample_bandpower,
                                sample_dir / f"{sample_name}_window_bandpower.svg",
                                x_col="window_index",
                                x_label="Window index",
                                title=f"{signal_title} (band power vs window index)",
                                y_label="Band power (a.u.)",
                                label_pair=("Original", "Reconstruction"),
                            )
                    except Exception as e:
                        logger.warning(f"Bandpower plot failed for {sample_name}: {e}")

                    try:
                        psd_freq, psd_orig = compute_welch_psd(fhr_orig, fs=fs, nperseg=psd_nperseg)
                        _, psd_recon = compute_welch_psd(fhr_recon, fs=fs, nperseg=psd_nperseg)
                        _, psd_resid = compute_welch_psd(fhr_orig - fhr_recon, fs=fs, nperseg=psd_nperseg)
                        if psd_freq.size > 0:
                            plot_psd_comparison(
                                psd_freq,
                                psd_orig,
                                np.zeros_like(psd_orig),
                                psd_recon,
                                np.zeros_like(psd_recon),
                                sample_dir,
                                psd_residual_mean=psd_resid,
                                psd_residual_std=np.zeros_like(psd_resid),
                                filename=f"{sample_name}_psd.svg",
                            )
                    except Exception as e:
                        logger.warning(f"PSD comparison failed for {sample_name}: {e}")

                    try:
                        lags_sec, corr = compute_cross_correlation(
                            fhr_orig, fhr_recon, fs=fs, max_lag_sec=max_corr_lag_sec
                        )
                        if lags_sec.size > 0:
                            plot_cross_correlation(
                                lags_sec,
                                corr,
                                np.zeros_like(corr),
                                sample_dir,
                                filename=f"{sample_name}_cross_correlation.svg",
                            )
                    except Exception as e:
                        logger.warning(f"Cross-correlation failed for {sample_name}: {e}")

                    detailed_saved += 1

            if max_samples and processed >= max_samples:
                break

    if not recon_coherence_list:
        logger.warning("No coherence data collected.")
        return {}

    # Stack and compute statistics
    coh_recon_arr = np.array(recon_coherence_list)

    results = {
        "frequencies": frequencies,
        "recon_coherence_mean": np.mean(coh_recon_arr, axis=0),
        "recon_coherence_std": np.std(coh_recon_arr, axis=0),
        "n_samples": processed,
    }
    results["coherence_reconstructed"] = results["recon_coherence_mean"]
    results["coherence_std_reconstructed"] = results["recon_coherence_std"]
    if coherence_thresholds:
        results["coherence_significance_threshold_mean"] = float(np.nanmean(coherence_thresholds))

    # Create visualization
    output_dir = runner.ensure_dir("coherence")
    plot_reconstruction_coherence(
        results["frequencies"],
        results["recon_coherence_mean"],
        results["recon_coherence_std"],
        output_dir,
        significance_threshold=results.get("coherence_significance_threshold_mean"),
    )

    if include_up_coherence and up_coherence_original_list and up_frequencies is not None:
        up_orig_arr = np.array(up_coherence_original_list)
        up_recon_arr = np.array(up_coherence_recon_list)
        results.update({
            "up_frequencies": up_frequencies,
            "up_coherence_original_mean": np.mean(up_orig_arr, axis=0),
            "up_coherence_reconstructed_mean": np.mean(up_recon_arr, axis=0),
            "up_coherence_original_std": np.std(up_orig_arr, axis=0),
            "up_coherence_reconstructed_std": np.std(up_recon_arr, axis=0),
        })
        plot_coherence_analysis(
            results["up_frequencies"],
            results["up_coherence_original_mean"],
            results["up_coherence_reconstructed_mean"],
            output_dir,
            filename="up_fhr_coherence.svg",
        )

    if psd_orig_list and psd_frequencies is not None:
        psd_orig_arr = np.stack(psd_orig_list, axis=0)
        psd_recon_arr = np.stack(psd_recon_list, axis=0)
        psd_resid_arr = np.stack(psd_resid_list, axis=0)
        results.update({
            "psd_frequencies": psd_frequencies,
            "psd_orig_mean": np.mean(psd_orig_arr, axis=0),
            "psd_orig_std": np.std(psd_orig_arr, axis=0),
            "psd_recon_mean": np.mean(psd_recon_arr, axis=0),
            "psd_recon_std": np.std(psd_recon_arr, axis=0),
            "psd_resid_mean": np.mean(psd_resid_arr, axis=0),
            "psd_resid_std": np.std(psd_resid_arr, axis=0),
        })
        plot_psd_comparison(
            results["psd_frequencies"],
            results["psd_orig_mean"],
            results["psd_orig_std"],
            results["psd_recon_mean"],
            results["psd_recon_std"],
            output_dir,
            psd_residual_mean=results["psd_resid_mean"],
            psd_residual_std=results["psd_resid_std"],
        )

    if corr_list and corr_lags_sec is not None:
        corr_arr = np.stack(corr_list, axis=0)
        results.update({
            "corr_lags_sec": corr_lags_sec,
            "corr_mean": np.mean(corr_arr, axis=0),
            "corr_std": np.std(corr_arr, axis=0),
        })
        plot_cross_correlation(
            results["corr_lags_sec"],
            results["corr_mean"],
            results["corr_std"],
            output_dir,
        )

    # Save detailed data
    if epoch_records:
        df_epochs = pd.DataFrame(epoch_records)
        df_epochs.to_csv(output_dir / "epoch_coherence_summary.csv", index=False)
        if "epoch" in df_epochs.columns:
            epoch_band_rows: List[Dict[str, Any]] = []
            df_epochs = df_epochs.copy()
            df_epochs["hours_before"] = df_epochs["epoch"].apply(
                lambda val: (-val / 3600.0) if pd.notna(val) else np.nan
            )
            for band_name, _, _ in COHERENCE_BANDS:
                col = f"{band_name}_coherence"
                if col not in df_epochs.columns:
                    continue
                subset = df_epochs[["hours_before", col]].dropna()
                for _, row in subset.iterrows():
                    epoch_band_rows.append({
                        "band": band_name,
                        "hours_before": float(row["hours_before"]),
                        "coherence": float(row[col]),
                    })

            if epoch_band_rows:
                df_epoch_band = pd.DataFrame(epoch_band_rows)
                df_epoch_band = df_epoch_band[
                    (df_epoch_band["hours_before"] >= 0.0) & (df_epoch_band["hours_before"] <= 6.0)
                ]
                df_epoch_band["hour_bin"] = (df_epoch_band["hours_before"] * 2).round() / 2
                epoch_agg = df_epoch_band.groupby(["band", "hour_bin"]).agg(
                    coherence_mean=("coherence", "mean"),
                    coherence_std=("coherence", "std"),
                ).reset_index()
                epoch_agg.to_csv(output_dir / "epoch_coherence_aggregate.csv", index=False)
                _plot_band_trends(
                    epoch_agg,
                    output_dir / "epoch_coherence_trends.svg",
                    x_col="hour_bin",
                    x_label="Hours Before Birth",
                    title="Coherence vs Hours Before Birth",
                    y_label="Coherence",
                    xlim=(0.0, 6.0),
                    invert_x=True,
                )

    if window_records:
        df_windows = pd.DataFrame(window_records)
        df_windows.to_csv(output_dir / "window_coherence_summary.csv", index=False)

        window_agg = df_windows.groupby(["band", "window_index"]).agg(
            window_start_sec=("window_start_sec", "mean"),
            coherence_mean=("coherence", "mean"),
            coherence_std=("coherence", "std"),
        ).reset_index()
        window_agg.to_csv(output_dir / "window_coherence_aggregate.csv", index=False)

        _plot_band_trends(
            window_agg,
            output_dir / "window_coherence_trends.svg",
            x_col="window_start_sec",
            x_label="Window Start (seconds)",
            title="Coherence vs Window Start",
            y_label="Coherence",
        )

        _plot_band_trends(
            window_agg,
            output_dir / "window_coherence_trends_index.svg",
            x_col="window_index",
            x_label="Window index",
            title="Coherence vs Window Index",
            y_label="Coherence",
        )

        if "coherence_delta" in df_windows.columns:
            window_delta = df_windows.groupby(["band", "window_index"]).agg(
                window_start_sec=("window_start_sec", "mean"),
                coherence_mean=("coherence_delta", "mean"),
                coherence_std=("coherence_delta", "std"),
            ).reset_index()
            window_delta.to_csv(output_dir / "window_coherence_delta_aggregate.csv", index=False)

            _plot_band_trends(
                window_delta,
                output_dir / "window_coherence_delta_trends.svg",
                x_col="window_start_sec",
                x_label="Window Start (seconds)",
                title="Delta Coherence vs Window Start",
                y_label="Delta coherence",
                single_label="Delta coherence",
            )

            _plot_band_trends(
                window_delta,
                output_dir / "window_coherence_delta_trends_index.svg",
                x_col="window_index",
                x_label="Window index",
                title="Delta Coherence vs Window Index",
                y_label="Delta coherence",
                single_label="Delta coherence",
            )

    if relative_records:
        df_relative = pd.DataFrame(relative_records)
        df_relative.to_csv(output_dir / "relative_window_coherence_summary.csv", index=False)

        relative_agg = df_relative.groupby(["band", "relative_time_sec"]).agg(
            coherence_mean=("coherence", "mean"),
            coherence_std=("coherence", "std"),
        ).reset_index()
        relative_agg.to_csv(output_dir / "relative_window_coherence_aggregate.csv", index=False)

        _plot_band_trends(
            relative_agg,
            output_dir / "relative_window_coherence_trends.svg",
            x_col="relative_time_sec",
            x_label="Time From Window Start (seconds)",
            title="Coherence vs Time From Window Start",
            y_label="Coherence",
        )

        relative_index_agg = df_relative.groupby(["band", "relative_frame_index"]).agg(
            coherence_mean=("coherence", "mean"),
            coherence_std=("coherence", "std"),
        ).reset_index()
        relative_index_agg.to_csv(output_dir / "relative_window_coherence_index_aggregate.csv", index=False)

        _plot_band_trends(
            relative_index_agg,
            output_dir / "relative_window_coherence_trends_index.svg",
            x_col="relative_frame_index",
            x_label="STFT frame index",
            title="Coherence vs Frame Index",
            y_label="Coherence",
        )

        if "coherence_delta" in df_relative.columns:
            relative_delta = df_relative.groupby(["band", "relative_time_sec"]).agg(
                coherence_mean=("coherence_delta", "mean"),
                coherence_std=("coherence_delta", "std"),
            ).reset_index()
            relative_delta.to_csv(output_dir / "relative_window_coherence_delta_aggregate.csv", index=False)

            _plot_band_trends(
                relative_delta,
                output_dir / "relative_window_coherence_delta_trends.svg",
                x_col="relative_time_sec",
                x_label="Time From Window Start (seconds)",
                title="Delta Coherence vs Time From Window Start",
                y_label="Delta coherence",
                single_label="Delta coherence",
            )

            relative_delta_index = df_relative.groupby(["band", "relative_frame_index"]).agg(
                coherence_mean=("coherence_delta", "mean"),
                coherence_std=("coherence_delta", "std"),
            ).reset_index()
            relative_delta_index.to_csv(
                output_dir / "relative_window_coherence_delta_index_aggregate.csv",
                index=False,
            )

            _plot_band_trends(
                relative_delta_index,
                output_dir / "relative_window_coherence_delta_trends_index.svg",
                x_col="relative_frame_index",
                x_label="STFT frame index",
                title="Delta Coherence vs Frame Index",
                y_label="Delta coherence",
                single_label="Delta coherence",
            )

    if relative_tf_mean_list and relative_tf_template is not None:
        tf_stack = _stack_consistent(relative_tf_mean_list)
        tf_std_stack = _stack_consistent(relative_tf_std_list)
        tf_cv_stack = _stack_consistent(relative_tf_cv_list)
        if tf_stack.size:
            tf_mean = np.nanmean(tf_stack, axis=0)
            tf_std = np.nanstd(tf_stack, axis=0)
            plot_time_frequency_coherence(
                relative_tf_template["frequencies"],
                relative_tf_template["times"],
                tf_mean,
                output_path=output_dir / "relative_window_time_frequency_mean_stft.svg",
                max_freq=time_frequency_max_freq,
                title="Window-Relative Coherence Mean (STFT)",
            )
            plot_time_frequency_map(
                relative_tf_template["frequencies"],
                relative_tf_template["times"],
                tf_std,
                output_path=output_dir / "relative_window_time_frequency_std_stft.svg",
                max_freq=time_frequency_max_freq,
                title="Window-Relative Coherence Std Across Samples (STFT)",
                cmap="viridis",
                colorbar_label="Std",
            )
        if tf_std_stack.size:
            tf_std_mean = np.nanmean(tf_std_stack, axis=0)
            plot_time_frequency_map(
                relative_tf_template["frequencies"],
                relative_tf_template["times"],
                tf_std_mean,
                output_path=output_dir / "relative_window_time_frequency_window_std_stft.svg",
                max_freq=time_frequency_max_freq,
                title="Window-to-Window Std (STFT)",
                cmap="viridis",
                colorbar_label="Std",
            )
        if tf_cv_stack.size:
            tf_cv_mean = np.nanmean(tf_cv_stack, axis=0)
            plot_time_frequency_map(
                relative_tf_template["frequencies"],
                relative_tf_template["times"],
                tf_cv_mean,
                output_path=output_dir / "relative_window_time_frequency_window_cv_stft.svg",
                max_freq=time_frequency_max_freq,
                title="Window-to-Window CV (STFT)",
                cmap="magma",
                colorbar_label="CV",
            )

    if relative_tf_wavelet_mean_list and relative_tf_wavelet_template is not None:
        tfw_stack = _stack_consistent(relative_tf_wavelet_mean_list)
        tfw_std_stack = _stack_consistent(relative_tf_wavelet_std_list)
        tfw_cv_stack = _stack_consistent(relative_tf_wavelet_cv_list)
        if tfw_stack.size:
            tfw_mean = np.nanmean(tfw_stack, axis=0)
            tfw_std = np.nanstd(tfw_stack, axis=0)
            plot_time_frequency_coherence(
                relative_tf_wavelet_template["frequencies"],
                relative_tf_wavelet_template["times"],
                tfw_mean,
                output_path=output_dir / "relative_window_time_frequency_mean_wavelet.svg",
                max_freq=time_frequency_max_freq,
                title="Window-Relative Coherence Mean (WAVELET)",
            )
            plot_time_frequency_map(
                relative_tf_wavelet_template["frequencies"],
                relative_tf_wavelet_template["times"],
                tfw_std,
                output_path=output_dir / "relative_window_time_frequency_std_wavelet.svg",
                max_freq=time_frequency_max_freq,
                title="Window-Relative Coherence Std Across Samples (WAVELET)",
                cmap="viridis",
                colorbar_label="Std",
            )
        if tfw_std_stack.size:
            tfw_std_mean = np.nanmean(tfw_std_stack, axis=0)
            plot_time_frequency_map(
                relative_tf_wavelet_template["frequencies"],
                relative_tf_wavelet_template["times"],
                tfw_std_mean,
                output_path=output_dir / "relative_window_time_frequency_window_std_wavelet.svg",
                max_freq=time_frequency_max_freq,
                title="Window-to-Window Std (WAVELET)",
                cmap="viridis",
                colorbar_label="Std",
            )
        if tfw_cv_stack.size:
            tfw_cv_mean = np.nanmean(tfw_cv_stack, axis=0)
            plot_time_frequency_map(
                relative_tf_wavelet_template["frequencies"],
                relative_tf_wavelet_template["times"],
                tfw_cv_mean,
                output_path=output_dir / "relative_window_time_frequency_window_cv_wavelet.svg",
                max_freq=time_frequency_max_freq,
                title="Window-to-Window CV (WAVELET)",
                cmap="magma",
                colorbar_label="CV",
            )

        if relative_tf_wavelet_plv_list:
            plv_stack = _stack_consistent(relative_tf_wavelet_plv_list)
            if plv_stack.size:
                plv_mean = np.nanmean(plv_stack, axis=0)
                plot_time_frequency_map(
                    relative_tf_wavelet_template["frequencies"],
                    relative_tf_wavelet_template["times"],
                    plv_mean,
                    output_path=output_dir / "relative_window_wavelet_plv.svg",
                    max_freq=time_frequency_max_freq,
                    title="Wavelet PLV (Mean Across Samples)",
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                    colorbar_label="PLV",
                )

        if relative_tf_wavelet_phase_list:
            phase_stack = _stack_consistent(relative_tf_wavelet_phase_list)
            if phase_stack.size:
                exp_phase = np.exp(1j * phase_stack)
                exp_phase[~np.isfinite(phase_stack)] = np.nan + 1j * np.nan
                phase_sum = np.nansum(exp_phase, axis=0)
                phase_count = np.sum(np.isfinite(phase_stack), axis=0)
                phase_mean = np.angle(phase_sum)
                phase_mean[phase_count == 0] = np.nan
                plot_time_frequency_map(
                    relative_tf_wavelet_template["frequencies"],
                    relative_tf_wavelet_template["times"],
                    phase_mean,
                    output_path=output_dir / "relative_window_wavelet_phase_mean.svg",
                    max_freq=time_frequency_max_freq,
                    title="Wavelet Phase Mean (Across Samples)",
                    cmap="twilight",
                    vmin=-np.pi,
                    vmax=np.pi,
                    colorbar_label="Phase (rad)",
                )

    if horizon_early_list and relative_tf_template is not None:
        early_stack = _stack_consistent(horizon_early_list)
        late_stack = _stack_consistent(horizon_late_list)
        delta_stack = _stack_consistent(horizon_delta_list)
        if early_stack.size and late_stack.size:
            early_mean = np.nanmean(early_stack, axis=0)
            late_mean = np.nanmean(late_stack, axis=0)
            early_std = np.nanstd(early_stack, axis=0)
            late_std = np.nanstd(late_stack, axis=0)
            horizon_df = pd.DataFrame({
                "frequency": relative_tf_template["frequencies"],
                "early_mean": early_mean,
                "early_std": early_std,
                "late_mean": late_mean,
                "late_std": late_std,
            })
            horizon_df["delta_mean"] = horizon_df["late_mean"] - horizon_df["early_mean"]
            horizon_df["delta_std"] = np.nanstd(delta_stack, axis=0) if delta_stack.size else np.nan
            horizon_df.to_csv(output_dir / "relative_window_horizon_spectra_stft.csv", index=False)
            plot_horizon_spectra(
                relative_tf_template["frequencies"],
                early_mean,
                early_std,
                late_mean,
                late_std,
                output_path=output_dir / "relative_window_horizon_spectra_stft.svg",
                max_freq=time_frequency_max_freq,
                title="Early vs Late Horizon Coherence (STFT)",
            )
        if delta_stack.size:
            delta_mean = np.nanmean(delta_stack, axis=0)
            delta_std = np.nanstd(delta_stack, axis=0)
            plot_spectrum_delta(
                relative_tf_template["frequencies"],
                delta_mean,
                delta_std,
                output_path=output_dir / "relative_window_horizon_delta_stft.svg",
                max_freq=time_frequency_max_freq,
                title="Horizon Delta Coherence (STFT)",
            )

    if horizon_early_wavelet_list and relative_tf_wavelet_template is not None:
        early_stack = _stack_consistent(horizon_early_wavelet_list)
        late_stack = _stack_consistent(horizon_late_wavelet_list)
        delta_stack = _stack_consistent(horizon_delta_wavelet_list)
        if early_stack.size and late_stack.size:
            early_mean = np.nanmean(early_stack, axis=0)
            late_mean = np.nanmean(late_stack, axis=0)
            early_std = np.nanstd(early_stack, axis=0)
            late_std = np.nanstd(late_stack, axis=0)
            horizon_df = pd.DataFrame({
                "frequency": relative_tf_wavelet_template["frequencies"],
                "early_mean": early_mean,
                "early_std": early_std,
                "late_mean": late_mean,
                "late_std": late_std,
            })
            horizon_df["delta_mean"] = horizon_df["late_mean"] - horizon_df["early_mean"]
            horizon_df["delta_std"] = np.nanstd(delta_stack, axis=0) if delta_stack.size else np.nan
            horizon_df.to_csv(output_dir / "relative_window_horizon_spectra_wavelet.csv", index=False)
            plot_horizon_spectra(
                relative_tf_wavelet_template["frequencies"],
                early_mean,
                early_std,
                late_mean,
                late_std,
                output_path=output_dir / "relative_window_horizon_spectra_wavelet.svg",
                max_freq=time_frequency_max_freq,
                title="Early vs Late Horizon Coherence (WAVELET)",
            )
        if delta_stack.size:
            delta_mean = np.nanmean(delta_stack, axis=0)
            delta_std = np.nanstd(delta_stack, axis=0)
            plot_spectrum_delta(
                relative_tf_wavelet_template["frequencies"],
                delta_mean,
                delta_std,
                output_path=output_dir / "relative_window_horizon_delta_wavelet.svg",
                max_freq=time_frequency_max_freq,
                title="Horizon Delta Coherence (WAVELET)",
            )

    if permutation_upper_list and relative_tf_template is not None:
        perm_upper_stack = _stack_consistent(permutation_upper_list)
        perm_lower_stack = _stack_consistent(permutation_lower_list)
        if perm_upper_stack.size:
            perm_upper_mean = np.nanmean(perm_upper_stack, axis=0)
            plot_time_frequency_map(
                relative_tf_template["frequencies"],
                relative_tf_template["times"],
                perm_upper_mean,
                output_path=output_dir / "relative_window_perm_upper_stft.svg",
                max_freq=time_frequency_max_freq,
                title="Permutation Upper CI (STFT)",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                colorbar_label="Coherence",
            )
        if perm_lower_stack.size:
            perm_lower_mean = np.nanmean(perm_lower_stack, axis=0)
            plot_time_frequency_map(
                relative_tf_template["frequencies"],
                relative_tf_template["times"],
                perm_lower_mean,
                output_path=output_dir / "relative_window_perm_lower_stft.svg",
                max_freq=time_frequency_max_freq,
                title="Permutation Lower CI (STFT)",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                colorbar_label="Coherence",
            )

    if permutation_upper_wavelet_list and relative_tf_wavelet_template is not None:
        perm_upper_stack = _stack_consistent(permutation_upper_wavelet_list)
        perm_lower_stack = _stack_consistent(permutation_lower_wavelet_list)
        if perm_upper_stack.size:
            perm_upper_mean = np.nanmean(perm_upper_stack, axis=0)
            plot_time_frequency_map(
                relative_tf_wavelet_template["frequencies"],
                relative_tf_wavelet_template["times"],
                perm_upper_mean,
                output_path=output_dir / "relative_window_perm_upper_wavelet.svg",
                max_freq=time_frequency_max_freq,
                title="Permutation Upper CI (WAVELET)",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                colorbar_label="Coherence",
            )
        if perm_lower_stack.size:
            perm_lower_mean = np.nanmean(perm_lower_stack, axis=0)
            plot_time_frequency_map(
                relative_tf_wavelet_template["frequencies"],
                relative_tf_wavelet_template["times"],
                perm_lower_mean,
                output_path=output_dir / "relative_window_perm_lower_wavelet.svg",
                max_freq=time_frequency_max_freq,
                title="Permutation Lower CI (WAVELET)",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                colorbar_label="Coherence",
            )

    if relative_records_wavelet:
        df_relative_wavelet = pd.DataFrame(relative_records_wavelet)
        df_relative_wavelet.to_csv(
            output_dir / "relative_window_coherence_summary_wavelet.csv",
            index=False,
        )

        relative_wavelet_agg = df_relative_wavelet.groupby(["band", "relative_time_sec"]).agg(
            coherence_mean=("coherence", "mean"),
            coherence_std=("coherence", "std"),
        ).reset_index()
        relative_wavelet_agg.to_csv(
            output_dir / "relative_window_coherence_aggregate_wavelet.csv",
            index=False,
        )

        _plot_band_trends(
            relative_wavelet_agg,
            output_dir / "relative_window_coherence_trends_wavelet.svg",
            x_col="relative_time_sec",
            x_label="Time From Window Start (seconds)",
            title="Wavelet Coherence vs Time From Window Start",
            y_label="Coherence",
        )

        relative_wavelet_index = df_relative_wavelet.groupby(["band", "relative_frame_index"]).agg(
            coherence_mean=("coherence", "mean"),
            coherence_std=("coherence", "std"),
        ).reset_index()
        relative_wavelet_index.to_csv(
            output_dir / "relative_window_coherence_index_aggregate_wavelet.csv",
            index=False,
        )

        _plot_band_trends(
            relative_wavelet_index,
            output_dir / "relative_window_coherence_trends_index_wavelet.svg",
            x_col="relative_frame_index",
            x_label="Wavelet time index",
            title="Wavelet Coherence vs Time Index",
            y_label="Coherence",
        )

        if "coherence_delta" in df_relative_wavelet.columns:
            relative_wavelet_delta = df_relative_wavelet.groupby(["band", "relative_time_sec"]).agg(
                coherence_mean=("coherence_delta", "mean"),
                coherence_std=("coherence_delta", "std"),
            ).reset_index()
            relative_wavelet_delta.to_csv(
                output_dir / "relative_window_coherence_delta_aggregate_wavelet.csv",
                index=False,
            )

            _plot_band_trends(
                relative_wavelet_delta,
                output_dir / "relative_window_coherence_delta_trends_wavelet.svg",
                x_col="relative_time_sec",
                x_label="Time From Window Start (seconds)",
                title="Wavelet Delta Coherence vs Time From Window Start",
                y_label="Delta coherence",
                single_label="Delta coherence",
            )

            relative_wavelet_delta_index = df_relative_wavelet.groupby(
                ["band", "relative_frame_index"]
            ).agg(
                coherence_mean=("coherence_delta", "mean"),
                coherence_std=("coherence_delta", "std"),
            ).reset_index()
            relative_wavelet_delta_index.to_csv(
                output_dir / "relative_window_coherence_delta_index_aggregate_wavelet.csv",
                index=False,
            )

            _plot_band_trends(
                relative_wavelet_delta_index,
                output_dir / "relative_window_coherence_delta_trends_index_wavelet.svg",
                x_col="relative_frame_index",
                x_label="Wavelet time index",
                title="Wavelet Delta Coherence vs Time Index",
                y_label="Delta coherence",
                single_label="Delta coherence",
            )

    if bandpower_records:
        df_bandpower = pd.DataFrame(bandpower_records)
        df_bandpower.to_csv(output_dir / "window_bandpower_summary.csv", index=False)

        bandpower_agg = df_bandpower.groupby(["band", "window_index"]).agg(
            window_start_sec=("window_start_sec", "mean"),
            orig_mean=("bandpower_original", "mean"),
            orig_std=("bandpower_original", "std"),
            recon_mean=("bandpower_reconstructed", "mean"),
            recon_std=("bandpower_reconstructed", "std"),
        ).reset_index()
        bandpower_agg.to_csv(output_dir / "window_bandpower_aggregate.csv", index=False)

        _plot_band_trends(
            bandpower_agg,
            output_dir / "window_bandpower_trends.svg",
            x_col="window_start_sec",
            x_label="Window Start (seconds)",
            title="Band Power vs Window Start",
            y_label="Band power (a.u.)",
            label_pair=("Original", "Reconstruction"),
        )

        _plot_band_trends(
            bandpower_agg,
            output_dir / "window_bandpower_trends_index.svg",
            x_col="window_index",
            x_label="Window index",
            title="Band Power vs Window Index",
            y_label="Band power (a.u.)",
            label_pair=("Original", "Reconstruction"),
        )

        if "bandpower_delta" in df_bandpower.columns:
            bandpower_delta = df_bandpower.groupby(["band", "window_index"]).agg(
                window_start_sec=("window_start_sec", "mean"),
                coherence_mean=("bandpower_delta", "mean"),
                coherence_std=("bandpower_delta", "std"),
            ).reset_index()
            bandpower_delta.to_csv(output_dir / "window_bandpower_delta_aggregate.csv", index=False)

            _plot_band_trends(
                bandpower_delta,
                output_dir / "window_bandpower_delta_trends.svg",
                x_col="window_start_sec",
                x_label="Window Start (seconds)",
                title="Band Power Delta vs Window Start",
                y_label="Band power delta (a.u.)",
                single_label="Delta",
            )

            _plot_band_trends(
                bandpower_delta,
                output_dir / "window_bandpower_delta_trends_index.svg",
                x_col="window_index",
                x_label="Window index",
                title="Band Power Delta vs Window Index",
                y_label="Band power delta (a.u.)",
                single_label="Delta",
            )

    # Save summary statistics
    summary_path = output_dir / "coherence_summary.txt"
    with open(summary_path, "w") as f:
        f.write("FHR Reconstruction Coherence Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Samples analyzed: {processed}\n")
        f.write(f"Segment length: {nperseg} samples ({nperseg / fs:.1f} seconds)\n\n")
        if coherence_thresholds:
            threshold_mean = float(np.nanmean(coherence_thresholds))
            f.write(f"Mean significance threshold (alpha={time_frequency_permutation_alpha:.3f}): {threshold_mean:.4f}\n\n")
        f.write("Mean coherence by frequency band (FHR vs reconstructed):\n")

        # Compute band averages
        bands = [
            ("VLF (0-0.04 Hz)", 0, 0.04),
            ("LF (0.04-0.15 Hz)", 0.04, 0.15),
            ("HF (0.15-0.5 Hz)", 0.15, 0.5),
            ("Total (0-0.5 Hz)", 0, 0.5),
        ]

        for band_name, f_low, f_high in bands:
            mask = (frequencies >= f_low) & (frequencies < f_high)
            if mask.any():
                recon_mean = results["recon_coherence_mean"][mask].mean()
                f.write(f"  {band_name}: {recon_mean:.4f}\n")

        if include_up_coherence and "up_coherence_original_mean" in results:
            f.write("\nUP-FHR coherence by frequency band (optional):\n")
            for band_name, f_low, f_high in bands:
                mask = (results["up_frequencies"] >= f_low) & (results["up_frequencies"] < f_high)
                if mask.any():
                    orig_mean = results["up_coherence_original_mean"][mask].mean()
                    recon_mean = results["up_coherence_reconstructed_mean"][mask].mean()
                    f.write(f"  {band_name}:\n")
                    f.write(f"    UP vs FHR:     {orig_mean:.4f}\n")
                    f.write(f"    UP vs recon:   {recon_mean:.4f}\n")

        f.write("\nAdditional outputs:\n")
        f.write("  reconstruction_coherence.svg\n")
        f.write("  psd_comparison.svg\n")
        f.write("  cross_correlation.svg\n")
        if include_up_coherence and "up_coherence_original_mean" in results:
            f.write("  up_fhr_coherence.svg\n")
        f.write("  epoch_coherence_summary.csv\n")
        f.write("  epoch_coherence_aggregate.csv\n")
        f.write("  epoch_coherence_trends.svg\n")
        f.write("  window_coherence_summary.csv\n")
        f.write("  window_coherence_aggregate.csv\n")
        f.write("  window_coherence_trends.svg\n")
        f.write("  window_coherence_trends_index.svg\n")
        f.write("  window_coherence_delta_aggregate.csv\n")
        f.write("  window_coherence_delta_trends.svg\n")
        f.write("  window_coherence_delta_trends_index.svg\n")
        f.write("  relative_window_coherence_summary.csv\n")
        f.write("  relative_window_coherence_aggregate.csv\n")
        f.write("  relative_window_coherence_trends.svg\n")
        f.write("  relative_window_coherence_index_aggregate.csv\n")
        f.write("  relative_window_coherence_trends_index.svg\n")
        f.write("  relative_window_coherence_delta_aggregate.csv\n")
        f.write("  relative_window_coherence_delta_trends.svg\n")
        f.write("  relative_window_coherence_delta_index_aggregate.csv\n")
        f.write("  relative_window_coherence_delta_trends_index.svg\n")
        f.write("  relative_window_time_frequency_mean_stft.svg\n")
        f.write("  relative_window_time_frequency_std_stft.svg\n")
        f.write("  relative_window_time_frequency_window_std_stft.svg\n")
        f.write("  relative_window_time_frequency_window_cv_stft.svg\n")
        f.write("  relative_window_horizon_spectra_stft.svg\n")
        f.write("  relative_window_horizon_delta_stft.svg\n")
        f.write("  relative_window_horizon_spectra_stft.csv\n")
        f.write("  relative_window_perm_upper_stft.svg\n")
        f.write("  relative_window_perm_lower_stft.svg\n")
        if relative_records_wavelet:
            f.write("  relative_window_coherence_summary_wavelet.csv\n")
            f.write("  relative_window_coherence_aggregate_wavelet.csv\n")
            f.write("  relative_window_coherence_trends_wavelet.svg\n")
            f.write("  relative_window_coherence_index_aggregate_wavelet.csv\n")
            f.write("  relative_window_coherence_trends_index_wavelet.svg\n")
            f.write("  relative_window_coherence_delta_aggregate_wavelet.csv\n")
            f.write("  relative_window_coherence_delta_trends_wavelet.svg\n")
            f.write("  relative_window_coherence_delta_index_aggregate_wavelet.csv\n")
            f.write("  relative_window_coherence_delta_trends_index_wavelet.svg\n")
            f.write("  relative_window_time_frequency_mean_wavelet.svg\n")
            f.write("  relative_window_time_frequency_std_wavelet.svg\n")
            f.write("  relative_window_time_frequency_window_std_wavelet.svg\n")
            f.write("  relative_window_time_frequency_window_cv_wavelet.svg\n")
            f.write("  relative_window_wavelet_plv.svg\n")
            f.write("  relative_window_wavelet_phase_mean.svg\n")
            f.write("  relative_window_horizon_spectra_wavelet.svg\n")
            f.write("  relative_window_horizon_delta_wavelet.svg\n")
            f.write("  relative_window_horizon_spectra_wavelet.csv\n")
            f.write("  relative_window_perm_upper_wavelet.svg\n")
            f.write("  relative_window_perm_lower_wavelet.svg\n")
        f.write("  window_bandpower_summary.csv\n")
        f.write("  window_bandpower_aggregate.csv\n")
        f.write("  window_bandpower_trends.svg\n")
        f.write("  window_bandpower_trends_index.svg\n")
        f.write("  window_bandpower_delta_aggregate.csv\n")
        f.write("  window_bandpower_delta_trends.svg\n")
        f.write("  window_bandpower_delta_trends_index.svg\n")
        f.write("  samples/<sample>_signals.svg\n")
        f.write("  samples/<sample>_time_frequency_stft.svg\n")
        f.write("  samples/<sample>_time_frequency_wavelet.svg\n")
        f.write("  samples/<sample>_windowed_coherence.svg\n")
        f.write("  samples/<sample>_relative_window_time_frequency_stft.svg\n")
        f.write("  samples/<sample>_relative_window_time_frequency_stft_std.svg\n")
        f.write("  samples/<sample>_relative_window_time_frequency_stft_cv.svg\n")
        f.write("  samples/<sample>_relative_window_time_frequency_wavelet.svg\n")
        f.write("  samples/<sample>_relative_window_time_frequency_wavelet_std.svg\n")
        f.write("  samples/<sample>_relative_window_time_frequency_wavelet_cv.svg\n")
        f.write("  samples/<sample>_window_band_coherence.svg\n")
        f.write("  samples/<sample>_window_band_coherence_delta.svg\n")
        f.write("  samples/<sample>_relative_band_coherence.svg\n")
        f.write("  samples/<sample>_relative_band_coherence_delta.svg\n")
        f.write("  samples/<sample>_relative_band_coherence_wavelet.svg\n")
        f.write("  samples/<sample>_relative_band_coherence_delta_wavelet.svg\n")
        f.write("  samples/<sample>_relative_window_wavelet_plv.svg\n")
        f.write("  samples/<sample>_relative_window_wavelet_phase.svg\n")
        f.write("  samples/<sample>_relative_window_perm_upper_stft.svg\n")
        f.write("  samples/<sample>_relative_window_perm_upper_wavelet.svg\n")
        f.write("  samples/<sample>_single_window_*_signals.svg\n")
        f.write("  samples/<sample>_single_window_*_coherence_spectrum.svg\n")
        f.write("  samples/<sample>_single_window_*_time_frequency_stft.svg\n")
        f.write("  samples/<sample>_single_window_*_time_frequency_wavelet.svg\n")
        f.write("  samples/<sample>_single_window_*_psd.svg\n")
        f.write("  samples/<sample>_single_window_*_cross_correlation.svg\n")
        f.write("  samples/<sample>_window_bandpower.svg\n")
        f.write("  samples/<sample>_psd.svg\n")
        f.write("  samples/<sample>_cross_correlation.svg\n")

    logger.info(f"Coherence analysis complete. Results saved to {output_dir}")

    return results
