"""Latent-space analyses for the lag-attn v1 testing pipeline.

Covers:

- :func:`run_latent_distribution_analysis` — per-dimension marginal
  histograms over ``z``.
- :func:`run_latent_space_visualization` — 3D PCA scatter of time-averaged
  latents coloured by outcome class.
- :func:`run_latent_interpolation` — stub that warns and returns an
  empty list. Raw-signal interpolation is not supported by the lag-attn
  v1 model because it forecasts feature trajectories, not raw FHR.

Example:
    >>> from testing.analyses.latent import run_latent_distribution_analysis
    >>> latents = run_latent_distribution_analysis(runner, test_loader)
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from loguru import logger

from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.collectors import (
    collect_latents,
    collect_predictions,
)
from model.vae_teb_prediction.testing.visualizers import plot_latent_distributions
from model.vae_teb_prediction.testing.visualizers_interactive import plot_latent_space_3d


def run_latent_distribution_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 500,
) -> np.ndarray:
    """Plot per-dimension marginal distributions of the latent ``z``.

    Args:
        runner: Loaded :class:`TestRunner`.
        loader: PyTorch DataLoader.
        max_samples: Maximum samples to process.

    Returns:
        Flattened latent matrix of shape ``(N * T, d_z)``.
    """
    if max_samples <= 0:
        logger.info("Latent distribution analysis skipped (max_samples <= 0)")
        return np.array([])

    logger.info(f"Collecting latent representations (max {max_samples} samples)...")

    latents = collect_latents(runner, loader, max_samples)
    if latents.size == 0:
        logger.warning("No latent samples collected.")
        return latents

    logger.info(f"Collected latents: shape={latents.shape}")

    output_dir = runner.ensure_dir("latent_distribution")
    plot_latent_distributions(latents, output_dir)
    logger.info(f"Latent distribution plot saved to {output_dir}")

    return latents


def run_latent_space_visualization(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 500,
) -> np.ndarray:
    """Render a 3D PCA visualisation of per-sample time-averaged latents.

    Collects per-sample records via :func:`collect_predictions`, averages
    each sample's latent trajectory ``z (T, d_z)`` over the time axis, and
    plots the resulting ``(N, d_z)`` matrix in 3D PCA space coloured by
    outcome class.

    Args:
        runner: Loaded :class:`TestRunner`.
        loader: PyTorch DataLoader.
        max_samples: Maximum samples to process.

    Returns:
        ``(N, d_z)`` matrix of per-sample latent means.
    """
    logger.info("Creating 3D latent space visualization...")

    samples = collect_predictions(runner, loader, max_samples)
    if not samples:
        logger.warning("No samples collected for latent space visualization.")
        return np.array([])

    latents_list = []
    labels_list = []
    for s in samples:
        z_arr = s.get("z")
        if z_arr is None:
            continue
        latents_list.append(np.asarray(z_arr).mean(axis=0))
        labels_list.append(int(s.get("label") or 0))

    if not latents_list:
        return np.array([])

    latents = np.asarray(latents_list)
    labels = np.asarray(labels_list)

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
    """Stub: latent interpolation is not supported under lag-attn v1.

    The legacy pipeline decoded interpolated latent codes back to a raw
    FHR signal. The v1 model does not reconstruct raw FHR — its residual
    decoder produces 87-channel future feature trajectories at anchor
    points — so a morphing visualisation is no longer well defined.

    Args:
        runner: Unused (kept for API compatibility).
        loader: Unused.
        num_pairs: Unused.
        num_steps: Unused.

    Returns:
        Always ``[]``.
    """
    del runner, loader, num_pairs, num_steps
    logger.warning(
        "run_latent_interpolation is a no-op for lag-attn v1 (no raw-signal "
        "decoder). Returning an empty list."
    )
    return []
