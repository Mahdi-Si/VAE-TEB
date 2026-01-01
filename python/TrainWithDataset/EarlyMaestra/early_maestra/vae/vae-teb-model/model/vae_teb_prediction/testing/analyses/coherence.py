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
from loguru import logger

from scipy import signal

from ..base import TestRunner
from ..metrics import aggregate_predictions
from ..visualizers import plot_coherence_analysis


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


def run_coherence_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 50,
    nperseg: int = 64,
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

    Returns:
        Dict with:
            - 'frequencies': Frequency array
            - 'coherence_original': Mean coherence UP-FHR original
            - 'coherence_reconstructed': Mean coherence UP-FHR reconstructed
            - 'coherence_std_original': Std across samples
            - 'coherence_std_reconstructed': Std across samples

    Example:
        >>> results = run_coherence_analysis(runner, test_loader)
        >>> print(f"Mean coherence preserved: {results['coherence_reconstructed'].mean():.3f}")
    """

    logger.info(f"Running UP-FHR coherence analysis (max {max_samples} samples)...")

    coherence_original_list: List[np.ndarray] = []
    coherence_recon_list: List[np.ndarray] = []
    frequencies = None
    processed = 0

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

                up = batch.up[idx].cpu().numpy()
                fhr_orig = batch.fhr[idx].cpu().numpy()
                fhr_recon = avg_pred[idx].cpu().numpy()

                # Compute coherence
                try:
                    freq, coh_orig = compute_stft_coherence(up, fhr_orig, nperseg=nperseg)
                    _, coh_recon = compute_stft_coherence(up, fhr_recon, nperseg=nperseg)

                    coherence_original_list.append(coh_orig)
                    coherence_recon_list.append(coh_recon)

                    if frequencies is None:
                        frequencies = freq

                    processed += 1
                except Exception as e:
                    logger.warning(f"Coherence computation failed for sample {idx}: {e}")
                    continue

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

    logger.info(f"Coherence analysis complete. Results saved to {output_dir}")

    return results
