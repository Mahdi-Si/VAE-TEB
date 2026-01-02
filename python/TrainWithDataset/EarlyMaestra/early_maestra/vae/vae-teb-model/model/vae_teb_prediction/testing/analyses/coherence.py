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
    _style_axes,
    plot_coherence_analysis,
    plot_coherence_signals,
    plot_cross_correlation,
    plot_psd_comparison,
    plot_reconstruction_coherence,
    plot_time_frequency_coherence,
)

COHERENCE_BANDS = [
    ("VLF", 0.0, 0.04),
    ("LF", 0.04, 0.15),
    ("HF", 0.15, 0.5),
    ("Total", 0.0, 0.5),
]


def _plot_band_trends(
    df: pd.DataFrame,
    output_path: Path,
    *,
    x_col: str,
    x_label: str,
    title: str,
    y_label: str = "Coherence",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bands = [b[0] for b in COHERENCE_BANDS]
    n_bands = len(bands)
    cols = 2
    rows = int(np.ceil(n_bands / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.0, 3.0 * rows), sharex=False)
    axes = np.atleast_2d(axes)

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
            ax.plot(x_vals, subset["orig_mean"], color=COLOR_BLUE, linewidth=1.4, label="Reference")
            ax.plot(x_vals, subset["recon_mean"], color=COLOR_ORANGE, linewidth=1.4, linestyle="--", label="Reconstruction")

            if "orig_std" in subset.columns:
                ax.fill_between(
                    x_vals,
                    subset["orig_mean"] - subset["orig_std"],
                    subset["orig_mean"] + subset["orig_std"],
                    color=COLOR_BLUE,
                    alpha=0.2,
                )
            if "recon_std" in subset.columns:
                ax.fill_between(
                    x_vals,
                    subset["recon_mean"] - subset["recon_std"],
                    subset["recon_mean"] + subset["recon_std"],
                    color=COLOR_ORANGE,
                    alpha=0.2,
                )
        else:
            ax.plot(x_vals, subset["coherence_mean"], color=COLOR_BLUE, linewidth=1.4, label="Coherence")
            if "coherence_std" in subset.columns:
                ax.fill_between(
                    x_vals,
                    subset["coherence_mean"] - subset["coherence_std"],
                    subset["coherence_mean"] + subset["coherence_std"],
                    color=COLOR_BLUE,
                    alpha=0.2,
                )

        ax.set_title(f"{band} Band")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if y_label.lower().startswith("coherence"):
            ax.set_ylim(0, 1)
        ax.legend(loc="best", fontsize=7, framealpha=0.95)
        _style_axes(ax, grid="both", minor_ticks=True)

    # Hide unused subplots
    for idx in range(n_bands, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=9, y=1.02, fontweight="bold")
    fig.tight_layout()
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

    # Ensure same length
    min_len = min(len(x), len(y))
    x, y = x[:min_len], y[:min_len]

    # Define scales (logarithmically spaced)
    scales = np.logspace(np.log10(2), np.log10(min_len // 4), num_scales)

    # Compute CWT for both signals
    wavelet = "morl"  # Morlet wavelet
    coefs_x, frequencies = pywt.cwt(x, scales, wavelet, sampling_period=1.0 / fs)
    coefs_y, _ = pywt.cwt(y, scales, wavelet, sampling_period=1.0 / fs)

    # Compute cross-spectrum and power spectra
    cross_spectrum = coefs_x * np.conj(coefs_y)
    power_x = np.abs(coefs_x) ** 2
    power_y = np.abs(coefs_y) ** 2

    # Coherence = |Sxy|^2 / (Sxx * Syy)
    # Apply smoothing for stable coherence estimate
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size

    coherence = np.zeros_like(power_x)
    for i in range(len(scales)):
        smoothed_cross = np.convolve(np.abs(cross_spectrum[i]), kernel, mode="same")
        smoothed_x = np.convolve(power_x[i], kernel, mode="same")
        smoothed_y = np.convolve(power_y[i], kernel, mode="same")
        coherence[i] = smoothed_cross ** 2 / (smoothed_x * smoothed_y + 1e-12)

    # Time array
    times = np.arange(min_len) / fs

    # Phase difference
    phase = np.angle(cross_spectrum)

    return {
        "frequencies": frequencies,
        "times": times,
        "coherence": coherence,
        "phase": phase,
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

    freqs, times, stft_x = signal.stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
    _, _, stft_y = signal.stft(y, fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None)

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
            freqs = tf_map["frequencies"]
            times = tf_map["times"]
        else:
            if tf_map["coherence"].shape != acc.shape:
                continue
            acc += tf_map["coherence"]

        count += 1

    if acc is None:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence": np.empty((0, 0)),
            "n_windows": 0,
        }

    return {
        "frequencies": freqs,
        "times": times,
        "coherence": acc / max(count, 1),
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
    time_frequency_method: str = "stft",
    time_frequency_nperseg: int = 128,
    time_frequency_num_scales: int = 50,
    time_frequency_max_freq: float = 0.5,
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
        time_frequency_method: "stft" or "wavelet" for time-frequency plots.
        time_frequency_nperseg: STFT segment length for time-frequency plots.
        time_frequency_num_scales: Number of scales for wavelet coherence.
        time_frequency_max_freq: Max frequency to display in time-frequency plots.
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
    frequencies = None
    up_frequencies = None
    psd_frequencies = None
    corr_lags_sec = None
    processed = 0
    detailed_saved = 0
    epoch_records: List[Dict[str, Any]] = []
    window_records: List[Dict[str, Any]] = []
    relative_records: List[Dict[str, Any]] = []
    bandpower_records: List[Dict[str, Any]] = []

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
                sample_window_band = None
                sample_relative_band = None
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
                        sample_dir / f"{sample_name}_signals.png",
                        fs=fs,
                        title=signal_title,
                    )

                    method = str(time_frequency_method).lower() if time_frequency_method else "stft"
                    if method not in ("stft", "wavelet"):
                        logger.warning(f"Unknown time-frequency method '{time_frequency_method}', using STFT.")
                        method = "stft"

                    try:
                        if method == "wavelet":
                            tf_map = compute_wavelet_coherence(
                                fhr_orig, fhr_recon, fs=fs, num_scales=time_frequency_num_scales
                            )
                        else:
                            tf_map = compute_stft_coherence_map(
                                fhr_orig, fhr_recon, fs=fs, nperseg=time_frequency_nperseg
                            )

                        if tf_map["coherence"].size == 0:
                            logger.warning(f"Time-frequency coherence empty for {sample_name}; skipping plot.")
                        else:
                            plot_time_frequency_coherence(
                                tf_map["frequencies"],
                                tf_map["times"],
                                tf_map["coherence"],
                                output_path=sample_dir / f"{sample_name}_time_frequency.png",
                                max_freq=time_frequency_max_freq,
                                title=signal_title,
                            )
                    except ImportError as e:
                        logger.warning(f"Time-frequency coherence skipped: {e}")
                    except Exception as e:
                        logger.warning(f"Time-frequency coherence failed for {sample_name}: {e}")

                    try:
                        if window_tf["coherence"].size == 0:
                            logger.warning(f"Windowed coherence empty for {sample_name}; skipping plot.")
                        else:
                            plot_time_frequency_coherence(
                                window_tf["frequencies"],
                                window_tf["times"],
                                window_tf["coherence"],
                                output_path=sample_dir / f"{sample_name}_windowed_coherence.png",
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
                                        sample_dir / f"{sample_name}_window_band_coherence.png",
                                        x_col="window_index",
                                        x_label="Window index",
                                        title=f"{signal_title} (band coherence vs window index)",
                                        y_label="Coherence",
                                    )
                    except Exception as e:
                        logger.warning(f"Windowed coherence failed for {sample_name}: {e}")

                    try:
                        if relative_tf["coherence"].size == 0:
                            logger.warning(f"Relative window coherence empty for {sample_name}; skipping plot.")
                        else:
                            plot_time_frequency_coherence(
                                relative_tf["frequencies"],
                                relative_tf["times"],
                                relative_tf["coherence"],
                                output_path=sample_dir / f"{sample_name}_relative_window_time_frequency.png",
                                max_freq=time_frequency_max_freq,
                                title=f"{signal_title} (window-relative coherence)",
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
                                        sample_dir / f"{sample_name}_relative_band_coherence.png",
                                        x_col="relative_frame_index",
                                        x_label="STFT frame index",
                                        title=f"{signal_title} (band coherence vs frame index)",
                                        y_label="Coherence",
                                    )
                    except Exception as e:
                        logger.warning(f"Relative window coherence failed for {sample_name}: {e}")

                    try:
                        if sample_bandpower is not None and not sample_bandpower.empty:
                            sample_bandpower = sample_bandpower.copy()
                            sample_bandpower.rename(columns={
                                "bandpower_original": "orig_mean",
                                "bandpower_reconstructed": "recon_mean",
                            }, inplace=True)
                            _plot_band_trends(
                                sample_bandpower,
                                sample_dir / f"{sample_name}_window_bandpower.png",
                                x_col="window_index",
                                x_label="Window index",
                                title=f"{signal_title} (band power vs window index)",
                                y_label="Band power (a.u.)",
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
                                filename=f"{sample_name}_psd.png",
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
                                filename=f"{sample_name}_cross_correlation.png",
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

    # Create visualization
    output_dir = runner.ensure_dir("coherence")
    plot_reconstruction_coherence(
        results["frequencies"],
        results["recon_coherence_mean"],
        results["recon_coherence_std"],
        output_dir,
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
            filename="up_fhr_coherence.png",
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
            output_dir / "window_coherence_trends.png",
            x_col="window_start_sec",
            x_label="Window Start (seconds)",
            title="Coherence vs Window Start",
            y_label="Coherence",
        )

        _plot_band_trends(
            window_agg,
            output_dir / "window_coherence_trends_index.png",
            x_col="window_index",
            x_label="Window index",
            title="Coherence vs Window Index",
            y_label="Coherence",
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
            output_dir / "relative_window_coherence_trends.png",
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
            output_dir / "relative_window_coherence_trends_index.png",
            x_col="relative_frame_index",
            x_label="STFT frame index",
            title="Coherence vs Frame Index",
            y_label="Coherence",
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
            output_dir / "window_bandpower_trends.png",
            x_col="window_start_sec",
            x_label="Window Start (seconds)",
            title="Band Power vs Window Start",
            y_label="Band power (a.u.)",
        )

        _plot_band_trends(
            bandpower_agg,
            output_dir / "window_bandpower_trends_index.png",
            x_col="window_index",
            x_label="Window index",
            title="Band Power vs Window Index",
            y_label="Band power (a.u.)",
        )

    # Save summary statistics
    summary_path = output_dir / "coherence_summary.txt"
    with open(summary_path, "w") as f:
        f.write("FHR Reconstruction Coherence Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Samples analyzed: {processed}\n")
        f.write(f"Segment length: {nperseg} samples ({nperseg / fs:.1f} seconds)\n\n")
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
        f.write("  reconstruction_coherence.png\n")
        f.write("  psd_comparison.png\n")
        f.write("  cross_correlation.png\n")
        if include_up_coherence and "up_coherence_original_mean" in results:
            f.write("  up_fhr_coherence.png\n")
        f.write("  epoch_coherence_summary.csv\n")
        f.write("  window_coherence_summary.csv\n")
        f.write("  window_coherence_aggregate.csv\n")
        f.write("  window_coherence_trends.png\n")
        f.write("  window_coherence_trends_index.png\n")
        f.write("  relative_window_coherence_summary.csv\n")
        f.write("  relative_window_coherence_aggregate.csv\n")
        f.write("  relative_window_coherence_trends.png\n")
        f.write("  relative_window_coherence_index_aggregate.csv\n")
        f.write("  relative_window_coherence_trends_index.png\n")
        f.write("  window_bandpower_summary.csv\n")
        f.write("  window_bandpower_aggregate.csv\n")
        f.write("  window_bandpower_trends.png\n")
        f.write("  window_bandpower_trends_index.png\n")
        f.write("  samples/<sample>_signals.png\n")
        f.write("  samples/<sample>_time_frequency.png\n")
        f.write("  samples/<sample>_windowed_coherence.png\n")
        f.write("  samples/<sample>_relative_window_time_frequency.png\n")
        f.write("  samples/<sample>_window_band_coherence.png\n")
        f.write("  samples/<sample>_relative_band_coherence.png\n")
        f.write("  samples/<sample>_window_bandpower.png\n")
        f.write("  samples/<sample>_psd.png\n")
        f.write("  samples/<sample>_cross_correlation.png\n")

    logger.info(f"Coherence analysis complete. Results saved to {output_dir}")

    return results
