"""
Time-frequency coherence analysis for VAE-TEB models.

This module analyzes the spectral coherence between UP (uterine pressure)
and FHR (fetal heart rate) signals, comparing original vs reconstructed.

Coherence preservation indicates that the model maintains the physiological
coupling between contractions and heart rate variability.

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
    plot_coherence_analysis,
    plot_coherence_signals,
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
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bands = [b[0] for b in COHERENCE_BANDS]
    n_bands = len(bands)
    cols = 2
    rows = int(np.ceil(n_bands / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), sharex=False)
    axes = np.atleast_2d(axes)

    for idx, band in enumerate(bands):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        subset = df[df["band"] == band].sort_values(x_col)
        if subset.empty:
            ax.axis("off")
            continue

        x_vals = subset[x_col].values
        ax.plot(x_vals, subset["orig_mean"], color="#4C72B0", linewidth=1.5, label="Original")
        ax.plot(x_vals, subset["recon_mean"], color="#C44E52", linewidth=1.5, linestyle="--", label="Reconstructed")

        if "orig_std" in subset.columns:
            ax.fill_between(
                x_vals,
                subset["orig_mean"] - subset["orig_std"],
                subset["orig_mean"] + subset["orig_std"],
                color="#4C72B0",
                alpha=0.2,
            )
        if "recon_std" in subset.columns:
            ax.fill_between(
                x_vals,
                subset["recon_mean"] - subset["recon_std"],
                subset["recon_mean"] + subset["recon_std"],
                color="#C44E52",
                alpha=0.2,
            )

        ax.set_title(f"{band} Band")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Coherence")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    # Hide unused subplots
    for idx in range(n_bands, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
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
        >>> freqs, coh = compute_stft_coherence(up_signal, fhr_signal)
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
    up: np.ndarray,
    fhr_original: np.ndarray,
    fhr_recon_windows: np.ndarray,
    *,
    fs: float,
    stride: int,
    warmup: int,
    nperseg: int,
) -> Dict[str, np.ndarray]:
    """
    Compute per-window coherence maps aligned to the model's 2-minute predictions.

    Args:
        up: UP signal array (raw).
        fhr_original: Ground-truth FHR signal array (raw).
        fhr_recon_windows: Predicted windows, shape (T, H).
        fs: Sampling frequency in Hz.
        stride: Step size in raw samples between windows (decimation factor).
        warmup: Number of initial timesteps to skip.
        nperseg: Welch segment length for coherence.

    Returns:
        Dict with:
            - 'frequencies': Frequency array
            - 'times': Time array (window centers, seconds)
            - 'coherence_original': Coherence map (freq x time)
            - 'coherence_reconstructed': Coherence map (freq x time)
    """
    min_len = min(len(up), len(fhr_original))
    if min_len < 2:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence_original": np.empty((0, 0)),
            "coherence_reconstructed": np.empty((0, 0)),
        }

    T, H = fhr_recon_windows.shape
    if H < 2:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence_original": np.empty((0, 0)),
            "coherence_reconstructed": np.empty((0, 0)),
        }

    nperseg_used = min(nperseg, H)
    if nperseg_used < 2:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence_original": np.empty((0, 0)),
            "coherence_reconstructed": np.empty((0, 0)),
        }

    coh_orig_list: List[np.ndarray] = []
    coh_recon_list: List[np.ndarray] = []
    times: List[float] = []
    window_indices: List[int] = []
    frequencies = None

    for t in range(warmup, T):
        start = t * stride
        end = start + H
        if end > min_len:
            break

        up_win = up[start:end]
        fhr_win = fhr_original[start:end]
        recon_win = fhr_recon_windows[t]

        freq, coh_orig = compute_stft_coherence(up_win, fhr_win, fs=fs, nperseg=nperseg_used)
        _, coh_recon = compute_stft_coherence(up_win, recon_win, fs=fs, nperseg=nperseg_used)

        if frequencies is None:
            frequencies = freq

        coh_orig_list.append(coh_orig)
        coh_recon_list.append(coh_recon)
        times.append((start + H / 2) / fs)
        window_indices.append(t)

    if not coh_orig_list:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence_original": np.empty((0, 0)),
            "coherence_reconstructed": np.empty((0, 0)),
        }

    coherence_original = np.stack(coh_orig_list, axis=1)
    coherence_reconstructed = np.stack(coh_recon_list, axis=1)

    return {
        "frequencies": frequencies,
        "times": np.array(times),
        "window_indices": np.array(window_indices),
        "coherence_original": coherence_original,
        "coherence_reconstructed": coherence_reconstructed,
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
        >>> result = compute_wavelet_coherence(up, fhr)
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
    up: np.ndarray,
    fhr_original: np.ndarray,
    fhr_recon_windows: np.ndarray,
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
    min_len = min(len(up), len(fhr_original))
    T, H = fhr_recon_windows.shape
    if min_len < 2 or H < 2:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence_original": np.empty((0, 0)),
            "coherence_reconstructed": np.empty((0, 0)),
            "n_windows": 0,
        }

    nperseg_used = min(nperseg, H)
    if nperseg_used < 2:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence_original": np.empty((0, 0)),
            "coherence_reconstructed": np.empty((0, 0)),
            "n_windows": 0,
        }

    acc_orig = None
    acc_recon = None
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

        up_win = up[start:end]
        fhr_win = fhr_original[start:end]
        recon_win = fhr_recon_windows[t]

        orig_tf = compute_stft_coherence_map(up_win, fhr_win, fs=fs, nperseg=nperseg_used)
        recon_tf = compute_stft_coherence_map(up_win, recon_win, fs=fs, nperseg=nperseg_used)

        if orig_tf["coherence"].size == 0 or recon_tf["coherence"].size == 0:
            continue

        if acc_orig is None:
            acc_orig = orig_tf["coherence"].copy()
            acc_recon = recon_tf["coherence"].copy()
            freqs = orig_tf["frequencies"]
            times = orig_tf["times"]
        else:
            if orig_tf["coherence"].shape != acc_orig.shape:
                continue
            acc_orig += orig_tf["coherence"]
            acc_recon += recon_tf["coherence"]

        count += 1

    if acc_orig is None:
        return {
            "frequencies": np.array([]),
            "times": np.array([]),
            "coherence_original": np.empty((0, 0)),
            "coherence_reconstructed": np.empty((0, 0)),
            "n_windows": 0,
        }

    return {
        "frequencies": freqs,
        "times": times,
        "coherence_original": acc_orig / max(count, 1),
        "coherence_reconstructed": acc_recon / max(count, 1),
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
    time_frequency_method: str = "stft",
    time_frequency_nperseg: int = 128,
    time_frequency_num_scales: int = 50,
    time_frequency_max_freq: float = 0.5,
    window_nperseg: int = 128,
    window_relative_nperseg: int = 128,
    max_window_timefreq_windows: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run complete UP-FHR coherence analysis.

    Compares spectral coherence between:
    1. UP and original FHR (ground truth coupling)
    2. UP and reconstructed FHR (preserved coupling)

    This helps assess whether the model maintains physiologically
    meaningful relationships between contractions and heart rate.

    Args:
        runner: TestRunner with model and device configured.
        loader: PyTorch DataLoader for test data.
        max_samples: Maximum samples to process (default 50).
        nperseg: Segment length for STFT coherence (default 64).
        fs: Sampling frequency in Hz (default 4.0).
        max_detailed_samples: Number of per-sample plots to generate.
        time_frequency_method: "stft" or "wavelet" for time-frequency plots.
        time_frequency_nperseg: STFT segment length for time-frequency plots.
        time_frequency_num_scales: Number of scales for wavelet coherence.
        time_frequency_max_freq: Max frequency to display in time-frequency plots.
        window_nperseg: Welch segment length for per-window (2-minute) coherence.
        window_relative_nperseg: STFT segment length for within-window coherence.
        max_window_timefreq_windows: Optional limit on windows for relative maps.

    Returns:
        Dict with:
            - 'frequencies': Frequency array
            - 'coherence_original': Mean coherence UP-FHR original
            - 'coherence_reconstructed': Mean coherence UP-FHR reconstructed
            - 'coherence_std_original': Std across samples
            - 'coherence_std_reconstructed': Std across samples
            - 'n_samples': Number of samples analyzed

    Example:
        >>> results = run_coherence_analysis(runner, test_loader)
        >>> print(f"Mean coherence preserved: {results['coherence_reconstructed'].mean():.3f}")
    """

    logger.info(f"Running UP-FHR coherence analysis (max {max_samples} samples)...")

    coherence_original_list: List[np.ndarray] = []
    coherence_recon_list: List[np.ndarray] = []
    frequencies = None
    processed = 0
    detailed_saved = 0
    epoch_records: List[Dict[str, Any]] = []
    window_records: List[Dict[str, Any]] = []
    relative_records: List[Dict[str, Any]] = []

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = batch.fhr_st.size(0)

            # Check if UP signal is available
            if not hasattr(batch, "up") or batch.up is None:
                logger.warning("UP signal not available in batch - skipping coherence analysis")
                continue

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

                up_full = batch.up[idx].cpu().numpy()
                fhr_orig_full = batch.fhr[idx].cpu().numpy()
                fhr_recon_full = avg_pred[idx].cpu().numpy()
                up = up_full
                fhr_orig = fhr_orig_full
                fhr_recon = fhr_recon_full
                mask = valid_mask[idx].cpu().numpy() if valid_mask is not None else None

                if mask is not None and mask.any():
                    start = int(np.argmax(mask))
                    end = int(len(mask) - np.argmax(mask[::-1]))
                    up = up[start:end]
                    fhr_orig = fhr_orig[start:end]
                    fhr_recon = fhr_recon[start:end]

                min_len = min(len(up), len(fhr_orig), len(fhr_recon))
                if min_len < nperseg:
                    logger.warning(
                        f"Sample {idx} too short for coherence (len={min_len}, nperseg={nperseg}); skipping."
                    )
                    continue

                window_tf = {
                    "frequencies": np.array([]),
                    "times": np.array([]),
                    "window_indices": np.array([]),
                    "coherence_original": np.empty((0, 0)),
                    "coherence_reconstructed": np.empty((0, 0)),
                }
                relative_tf = {
                    "frequencies": np.array([]),
                    "times": np.array([]),
                    "coherence_original": np.empty((0, 0)),
                    "coherence_reconstructed": np.empty((0, 0)),
                    "n_windows": 0,
                }

                # Compute coherence
                try:
                    freq, coh_orig = compute_stft_coherence(up, fhr_orig, fs=fs, nperseg=nperseg)
                    _, coh_recon = compute_stft_coherence(up, fhr_recon, fs=fs, nperseg=nperseg)

                    coherence_original_list.append(coh_orig)
                    coherence_recon_list.append(coh_recon)

                    if frequencies is None:
                        frequencies = freq

                    band_orig = _band_means(freq, coh_orig)
                    band_recon = _band_means(freq, coh_recon)
                    epoch_record = {
                        "guid": guid,
                        "epoch": epoch,
                        "label": label,
                        "sample_idx": processed,
                    }
                    for band in COHERENCE_BANDS:
                        name = band[0]
                        epoch_record[f"{name}_orig"] = float(band_orig.get(name, np.nan))
                        epoch_record[f"{name}_recon"] = float(band_recon.get(name, np.nan))
                        epoch_record[f"{name}_delta"] = float(band_recon.get(name, np.nan) - band_orig.get(name, np.nan))
                    epoch_records.append(epoch_record)

                    mu_pr_window = mu_pr[idx].detach().cpu().numpy()
                    window_tf = compute_windowed_coherence_map(
                        up_full,
                        fhr_orig_full,
                        mu_pr_window,
                        fs=fs,
                        stride=runner.decimation_factor,
                        warmup=runner.warmup_steps,
                        nperseg=window_nperseg,
                    )
                    if window_tf["coherence_original"].size > 0:
                        window_band_orig = _band_means(window_tf["frequencies"], window_tf["coherence_original"])
                        window_band_recon = _band_means(window_tf["frequencies"], window_tf["coherence_reconstructed"])
                        window_indices = window_tf.get("window_indices", np.arange(window_tf["coherence_original"].shape[1]))
                        for band_name in window_band_orig.keys():
                            orig_vals = window_band_orig[band_name]
                            recon_vals = window_band_recon.get(band_name)
                            if recon_vals is None:
                                continue
                            for w_idx, window_index in enumerate(window_indices):
                                window_start_sample = int(window_index) * int(runner.decimation_factor)
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
                                    "coherence_original": float(orig_vals[w_idx]) if w_idx < len(orig_vals) else np.nan,
                                    "coherence_reconstructed": float(recon_vals[w_idx]) if w_idx < len(recon_vals) else np.nan,
                                    "coherence_delta": float(recon_vals[w_idx] - orig_vals[w_idx]) if w_idx < len(orig_vals) else np.nan,
                                })

                        relative_tf = compute_window_relative_time_frequency(
                            up_full,
                            fhr_orig_full,
                            mu_pr_window,
                            fs=fs,
                            stride=runner.decimation_factor,
                            warmup=runner.warmup_steps,
                            nperseg=window_relative_nperseg,
                            max_windows=max_window_timefreq_windows,
                        )
                        if relative_tf["coherence_original"].size > 0:
                            relative_band_orig = _band_means(relative_tf["frequencies"], relative_tf["coherence_original"])
                            relative_band_recon = _band_means(relative_tf["frequencies"], relative_tf["coherence_reconstructed"])
                            for band_name in relative_band_orig.keys():
                                orig_vals = relative_band_orig[band_name]
                                recon_vals = relative_band_recon.get(band_name)
                                if recon_vals is None:
                                    continue
                                for t_idx, rel_time in enumerate(relative_tf["times"]):
                                    relative_records.append({
                                        "guid": guid,
                                        "epoch": epoch,
                                        "label": label,
                                        "sample_idx": processed,
                                        "band": band_name,
                                        "relative_time_sec": float(rel_time),
                                        "coherence_original": float(orig_vals[t_idx]) if t_idx < len(orig_vals) else np.nan,
                                        "coherence_reconstructed": float(recon_vals[t_idx]) if t_idx < len(recon_vals) else np.nan,
                                        "coherence_delta": float(recon_vals[t_idx] - orig_vals[t_idx]) if t_idx < len(orig_vals) else np.nan,
                                    })

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
                            orig_tf = compute_wavelet_coherence(
                                up, fhr_orig, fs=fs, num_scales=time_frequency_num_scales
                            )
                            recon_tf = compute_wavelet_coherence(
                                up, fhr_recon, fs=fs, num_scales=time_frequency_num_scales
                            )
                        else:
                            orig_tf = compute_stft_coherence_map(
                                up, fhr_orig, fs=fs, nperseg=time_frequency_nperseg
                            )
                            recon_tf = compute_stft_coherence_map(
                                up, fhr_recon, fs=fs, nperseg=time_frequency_nperseg
                            )

                        if orig_tf["coherence"].size == 0 or recon_tf["coherence"].size == 0:
                            logger.warning(f"Time-frequency coherence empty for {sample_name}; skipping plot.")
                        else:
                            plot_time_frequency_coherence(
                                orig_tf["frequencies"],
                                orig_tf["times"],
                                orig_tf["coherence"],
                                recon_tf["coherence"],
                                sample_dir / f"{sample_name}_time_frequency.png",
                                max_freq=time_frequency_max_freq,
                                title=signal_title,
                            )
                    except ImportError as e:
                        logger.warning(f"Time-frequency coherence skipped: {e}")
                    except Exception as e:
                        logger.warning(f"Time-frequency coherence failed for {sample_name}: {e}")

                    try:
                        if window_tf["coherence_original"].size == 0:
                            logger.warning(f"Windowed coherence empty for {sample_name}; skipping plot.")
                        else:
                            plot_time_frequency_coherence(
                                window_tf["frequencies"],
                                window_tf["times"],
                                window_tf["coherence_original"],
                                window_tf["coherence_reconstructed"],
                                sample_dir / f"{sample_name}_windowed_coherence.png",
                                max_freq=time_frequency_max_freq,
                                title=f"{signal_title} (2-min window coherence)",
                            )
                    except Exception as e:
                        logger.warning(f"Windowed coherence failed for {sample_name}: {e}")

                    try:
                        if relative_tf["coherence_original"].size == 0:
                            logger.warning(f"Relative window coherence empty for {sample_name}; skipping plot.")
                        else:
                            plot_time_frequency_coherence(
                                relative_tf["frequencies"],
                                relative_tf["times"],
                                relative_tf["coherence_original"],
                                relative_tf["coherence_reconstructed"],
                                sample_dir / f"{sample_name}_relative_window_time_frequency.png",
                                max_freq=time_frequency_max_freq,
                                title=f"{signal_title} (window-relative coherence)",
                            )
                    except Exception as e:
                        logger.warning(f"Relative window coherence failed for {sample_name}: {e}")

                    detailed_saved += 1

            if max_samples and processed >= max_samples:
                break

    if not coherence_original_list:
        logger.warning("No coherence data collected.")
        return {}

    # Stack and compute statistics
    coh_orig_arr = np.array(coherence_original_list)
    coh_recon_arr = np.array(coherence_recon_list)

    results = {
        "frequencies": frequencies,
        "coherence_original": np.mean(coh_orig_arr, axis=0),
        "coherence_reconstructed": np.mean(coh_recon_arr, axis=0),
        "coherence_std_original": np.std(coh_orig_arr, axis=0),
        "coherence_std_reconstructed": np.std(coh_recon_arr, axis=0),
        "n_samples": processed,
    }

    # Create visualization
    output_dir = runner.ensure_dir("coherence")
    plot_coherence_analysis(
        results["frequencies"],
        results["coherence_original"],
        results["coherence_reconstructed"],
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
            orig_mean=("coherence_original", "mean"),
            orig_std=("coherence_original", "std"),
            recon_mean=("coherence_reconstructed", "mean"),
            recon_std=("coherence_reconstructed", "std"),
        ).reset_index()
        window_agg.to_csv(output_dir / "window_coherence_aggregate.csv", index=False)

        _plot_band_trends(
            window_agg,
            output_dir / "window_coherence_trends.png",
            x_col="window_start_sec",
            x_label="Window Start (seconds)",
            title="Coherence vs Window Start",
        )

    if relative_records:
        df_relative = pd.DataFrame(relative_records)
        df_relative.to_csv(output_dir / "relative_window_coherence_summary.csv", index=False)

        relative_agg = df_relative.groupby(["band", "relative_time_sec"]).agg(
            orig_mean=("coherence_original", "mean"),
            orig_std=("coherence_original", "std"),
            recon_mean=("coherence_reconstructed", "mean"),
            recon_std=("coherence_reconstructed", "std"),
        ).reset_index()
        relative_agg.to_csv(output_dir / "relative_window_coherence_aggregate.csv", index=False)

        _plot_band_trends(
            relative_agg,
            output_dir / "relative_window_coherence_trends.png",
            x_col="relative_time_sec",
            x_label="Time From Window Start (seconds)",
            title="Coherence vs Time From Window Start",
        )

    # Save summary statistics
    summary_path = output_dir / "coherence_summary.txt"
    with open(summary_path, "w") as f:
        f.write("UP-FHR Coherence Analysis Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Samples analyzed: {processed}\n")
        f.write(f"Segment length: {nperseg} samples ({nperseg / 4.0:.1f} seconds)\n\n")
        f.write("Mean coherence by frequency band:\n")

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
                orig_mean = results["coherence_original"][mask].mean()
                recon_mean = results["coherence_reconstructed"][mask].mean()
                f.write(f"  {band_name}:\n")
                f.write(f"    Original:     {orig_mean:.4f}\n")
                f.write(f"    Reconstructed: {recon_mean:.4f}\n")
                f.write(f"    Preservation:  {recon_mean / max(orig_mean, 1e-6) * 100:.1f}%\n")

        f.write("\nAdditional outputs:\n")
        f.write("  epoch_coherence_summary.csv\n")
        f.write("  window_coherence_summary.csv\n")
        f.write("  window_coherence_aggregate.csv\n")
        f.write("  window_coherence_trends.png\n")
        f.write("  relative_window_coherence_summary.csv\n")
        f.write("  relative_window_coherence_aggregate.csv\n")
        f.write("  relative_window_coherence_trends.png\n")
        f.write("  samples/<sample>_windowed_coherence.png\n")
        f.write("  samples/<sample>_relative_window_time_frequency.png\n")

    logger.info(f"Coherence analysis complete. Results saved to {output_dir}")

    return results
