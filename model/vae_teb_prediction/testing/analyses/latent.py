"""
Latent space analysis for VAE-TEB models.

This module provides analyses for understanding the latent representation:
- Distribution of each latent dimension
- Latent interpolation between samples
- 3D visualization of latent space

Example:
    >>> from testing.analyses.latent import run_latent_distribution_analysis
    >>> latents = run_latent_distribution_analysis(runner, test_loader)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger

from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.collectors import collect_latents, collect_predictions
from model.vae_teb_prediction.testing.visualizers import plot_latent_distributions
from model.vae_teb_prediction.testing.visualizers_interactive import plot_latent_space_3d, plot_latent_interpolation_interactive


def run_latent_distribution_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 500,
) -> np.ndarray:
    """
    Analyze and visualize the distribution of latent dimensions.

    Creates a grid of histograms showing the marginal distribution of
    each latent dimension across all samples and timesteps.

    Args:
        runner: TestRunner with model and device configured.
        loader: PyTorch DataLoader for test data.
        max_samples: Maximum samples to process (default 500).

    Returns:
        Numpy array of shape (N*T, D) with all latent values.

    Example:
        >>> latents = run_latent_distribution_analysis(runner, test_loader)
        >>> print(f"Latent shape: {latents.shape}")
    """
    if max_samples <= 0:
        logger.info("Latent distribution analysis skipped (max_samples <= 0)")
        return np.array([])

    logger.info(f"Collecting latent representations (max {max_samples} samples)...")

    # Collect latent representations
    latents = collect_latents(runner, loader, max_samples)

    if latents.size == 0:
        logger.warning("No latent samples collected.")
        return latents

    logger.info(f"Collected latents: shape={latents.shape}")

    # Create distribution plot
    output_dir = runner.ensure_dir("latent_distribution")
    plot_latent_distributions(latents, output_dir)

    logger.info(f"Latent distribution plot saved to {output_dir}")

    return latents


def run_latent_space_visualization(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 500,
) -> np.ndarray:
    """
    Create an interactive 3D PCA visualization of the latent space.

    Projects high-dimensional latent codes to 3D using PCA and creates
    an interactive Plotly scatter plot colored by class label.

    Args:
        runner: TestRunner with model and device configured.
        loader: PyTorch DataLoader for test data.
        max_samples: Maximum samples to process.

    Returns:
        Numpy array of latent values.

    Example:
        >>> latents = run_latent_space_visualization(runner, test_loader)
    """
    logger.info("Creating 3D latent space visualization...")

    # Collect predictions to get latents with labels
    samples = collect_predictions(runner, loader, max_samples)

    if not samples:
        logger.warning("No samples collected for latent space visualization.")
        return np.array([])

    # Extract latents and labels (use mean over time for each sample)
    latents_list = []
    labels_list = []

    for s in samples:
        if s["latent"] is not None:
            # Mean over time dimension: (T, D) -> (D,)
            mean_latent = np.mean(s["latent"], axis=0)
            latents_list.append(mean_latent)
            labels_list.append(s.get("label", 0) or 0)

    if not latents_list:
        return np.array([])

    latents = np.array(latents_list)
    labels = np.array(labels_list)

    # Create 3D visualization
    output_dir = runner.ensure_dir("latent_space")
    output_path = output_dir / "latent_space_3d.html"
    plot_latent_space_3d(latents, labels, output_path)

    logger.info(f"Latent space 3D plot saved to {output_path}")

    return latents


def run_latent_interpolation(
    runner: TestRunner,
    loader: Any,
    num_pairs: int = 5,
    num_steps: int = 10,
) -> List[Dict[str, Any]]:
    """
    Perform latent space interpolation between sample pairs.

    Collects pairs of samples and interpolates their latent codes,
    decoding each interpolated point to visualize the transition.

    Args:
        runner: TestRunner with model and device configured.
        loader: PyTorch DataLoader for test data.
        num_pairs: Number of sample pairs to interpolate.
        num_steps: Number of interpolation steps (default 10).

    Returns:
        List of dicts with interpolation results.

    Example:
        >>> results = run_latent_interpolation(runner, test_loader, num_pairs=3)
    """
    logger.info(f"Running latent interpolation ({num_pairs} pairs, {num_steps} steps)...")

    # Collect samples for interpolation (need 2 * num_pairs)
    samples = collect_predictions(runner, loader, max_samples=2 * num_pairs)

    if len(samples) < 2:
        logger.warning("Not enough samples for interpolation.")
        return []

    # Create pairs from consecutive samples
    results = []

    with runner.inference_mode():
        for i in range(0, min(len(samples) - 1, num_pairs * 2), 2):
            s1, s2 = samples[i], samples[i + 1]

            if s1["latent"] is None or s2["latent"] is None:
                continue

            # Get mean latent codes
            z1 = torch.tensor(s1["latent"]).mean(dim=0).to(runner.device)  # (D,)
            z2 = torch.tensor(s2["latent"]).mean(dim=0).to(runner.device)  # (D,)

            # Interpolate
            interpolated_signals = []
            alphas = np.linspace(0, 1, num_steps)

            for alpha in alphas:
                # Linear interpolation
                z_interp = (1 - alpha) * z1 + alpha * z2

                # Expand to sequence: (D,) -> (1, T, D)
                T = 300  # Default sequence length
                z_seq = z_interp.unsqueeze(0).unsqueeze(0).expand(1, T, -1)

                # Decode
                _, mu_pr, _ = runner.model.decoder(z_seq)

                # Average to get raw signal
                from ..metrics import aggregate_predictions
                avg_pred, _ = aggregate_predictions(runner.model, mu_pr)

                if avg_pred is not None:
                    interpolated_signals.append(avg_pred[0].cpu().numpy())

            if interpolated_signals:
                results.append({
                    "z_start": z1.cpu().numpy(),
                    "z_end": z2.cpu().numpy(),
                    "interpolated_signals": interpolated_signals,
                    "guid_start": s1.get("guid"),
                    "guid_end": s2.get("guid"),
                })

    # Create interactive visualization if we have results
    if results:
        output_dir = runner.ensure_dir("latent_interpolation")
        output_path = output_dir / "interpolation.html"
        plot_latent_interpolation_interactive(results, output_path)
        logger.info(f"Latent interpolation saved to {output_path}")

    return results
