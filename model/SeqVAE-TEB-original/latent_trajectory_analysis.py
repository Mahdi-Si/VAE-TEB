import contextlib
import copy
import importlib.util
import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import yaml
from vae_teb_model import SeqVaeTeb
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import joblib
from umap import UMAP
from umap.umap_ import UMAP
from hmmlearn.hmm import GaussianHMM
from scipy.linalg import sqrtm
from hdf5_dataset.hdf5_dataset import create_optimized_dataloader, build_guid_filtered_dataloader
from model.graph_models_utils import load_checkpoint_torch

from graph_model_test import SeqVAEGraphModelTest
plt.switch_backend("Agg")


# ------------------------------------------------------
# Utility functions
# ------------------------------------------------------

def preprocess_latent(latent, window_length=9, polyorder=2, denoise=True):
    """
    Preprocess latent trajectories with robust z-score normalization and
    optional denoising.

    Args:
        latent: torch.Tensor of shape (Batch_size, time_steps, latent_dim)
        window_length: Savitzky-Golay filter window length (default: 9)
        polyorder: Savitzky-Golay polynomial order (default: 2)
        denoise: Whether to apply Savitzky-Golay filter (default: True)

    Returns:
        Preprocessed latent tensor of same shape
    """
    if not denoise:
        median = torch.median(latent, dim=1, keepdim=True).values
        mad = torch.median(torch.abs(latent - median), dim=1, keepdim=True).values
        mad = torch.where(mad == 0, torch.ones_like(mad), mad)
        return (latent - median) / mad

    from scipy.signal import savgol_filter

    latent_np = latent.cpu().numpy()
    median = np.median(latent_np, axis=1, keepdims=True)
    mad = np.median(np.abs(latent_np - median), axis=1, keepdims=True)
    mad = np.where(mad == 0, 1.0, mad)
    latent_normalized = (latent_np - median) / mad

    batch_size, time_steps, latent_dim = latent_normalized.shape
    for b in range(batch_size):
        for d in range(latent_dim):
            if time_steps >= window_length:
                latent_normalized[b, :, d] = savgol_filter(
                    latent_normalized[b, :, d],
                    window_length=window_length,
                    polyorder=polyorder
                )

    return torch.from_numpy(latent_normalized).to(latent.device)


def reduce_latent_dimensionality(
    latent_data,
    method='pca',
    n_components=3,
    n_neighbors=15,
    min_dist=0.1,
    return_reducer=False
):
    """
    Reduce latent trajectory dimensionality for visualization and analysis.

    Args:
        latent_data: torch.Tensor of shape (Batch_size, time_steps, latent_dim)
        method: str, one of ['pca', 'isomap', 'umap', 'tsne', 'diffusion']
        n_components: int, target dimensions (2 or 3)
        n_neighbors: int, neighbors for manifold methods (5-30)
        min_dist: float, UMAP minimum distance (0.0-1.0)
        return_reducer: bool, whether to return fitted reducer object

    Returns:
        reduced_data: np.ndarray of shape (Batch_size, time_steps, n_components)
        reducer: (optional) fitted dimensionality reduction object
    """
    from sklearn.manifold import Isomap
    from scipy.spatial.distance import pdist, squareform

    batch_size, time_steps, latent_dim = latent_data.shape

    latent_flat = latent_data.cpu().numpy().reshape(-1, latent_dim)

    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
        reduced_flat = reducer.fit_transform(latent_flat)

    elif method == 'isomap':
        reducer = Isomap(n_neighbors=n_neighbors, n_components=n_components, n_jobs=-1)
        reduced_flat = reducer.fit_transform(latent_flat)

    elif method == 'umap':
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='euclidean',
            random_state=42,
            n_jobs=-1
        )
        reduced_flat = reducer.fit_transform(latent_flat)

    elif method == 'tsne':
        reducer = TSNE(
            n_components=n_components,
            perplexity=min(30, latent_flat.shape[0] // 4),
            random_state=42,
            n_jobs=-1
        )
        reduced_flat = reducer.fit_transform(latent_flat)

    elif method == 'diffusion':
        from sklearn.metrics.pairwise import rbf_kernel

        gamma = 1.0 / latent_dim
        K = rbf_kernel(latent_flat, gamma=gamma)

        D = np.diag(K.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        L = D_inv_sqrt @ K @ D_inv_sqrt

        eigenvalues, eigenvectors = np.linalg.eigh(L)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        reduced_flat = eigenvectors[:, 1:n_components+1] * eigenvalues[1:n_components+1]
        reducer = None

    else:
        raise ValueError(f"Unknown method: {method}. Choose from ['pca', 'isomap', 'umap', 'tsne', 'diffusion']")

    reduced_data = reduced_flat.reshape(batch_size, time_steps, n_components)

    if return_reducer:
        return reduced_data, reducer
    return reduced_data


def summarize_trajectory(
    latent_trajectory,
    k=5,
    method='changepoint',
    epsilon=None,
    return_indices=False
):
    """
    Summarize latent trajectory in time by selecting k representative points.

    Args:
        latent_trajectory: np.ndarray of shape (time_steps, n_dims) or (batch_size, time_steps, n_dims)
        k: int, number of keyframes to extract (e.g., 1 for single summary, 5 for segments)
        method: str, summarization strategy
            - 'changepoint': Change-point segmentation using ruptures library (preserves regimes)
            - 'rdp': Ramer-Douglas-Peucker polyline simplification (preserves shape)
            - 'quantile': Quantile samples along arclength (even coverage)
            - 'medoid': Single medoid (most representative point) - only for k=1
            - 'frechet': Fréchet mean (Euclidean mean) - only for k=1
        epsilon: float, RDP tolerance parameter (only for method='rdp')
            If None, automatically tuned to get approximately k vertices
        return_indices: bool, if True return time indices of selected points

    Returns:
        summarized_trajectory: np.ndarray of shape (k, n_dims) or (batch_size, k, n_dims)
        indices (optional): np.ndarray of shape (k,) or (batch_size, k) - time indices

    Notes:
        - For nonlinear embeddings (UMAP/Isomap), use 'medoid', 'changepoint', 'rdp', or 'quantile'
        - Avoid 'frechet' (average) for nonlinear embeddings as it may produce unrealizable points
        - 'changepoint' requires ruptures package: pip install ruptures
    """
    # Handle batch dimension
    if latent_trajectory.ndim == 3:
        batch_size = latent_trajectory.shape[0]
        results = []
        indices_list = []
        for b in range(batch_size):
            result = summarize_trajectory(
                latent_trajectory[b],
                k=k,
                method=method,
                epsilon=epsilon,
                return_indices=return_indices
            )
            if return_indices:
                results.append(result[0])
                indices_list.append(result[1])
            else:
                results.append(result)

        if return_indices:
            return np.array(results), np.array(indices_list)
        return np.array(results)

    time_steps, n_dims = latent_trajectory.shape

    # Auto-set k=1 for methods that only support single point
    if method in ['medoid', 'frechet']:
        if k != 1:
            logger.warning(f"Method '{method}' only supports k=1. Automatically setting k=1 (provided k={k} ignored).")
        k = 1

    # K=1 methods: single summary point
    if k == 1:
        if method == 'medoid':
            # Find most representative actual time point
            centroid = np.mean(latent_trajectory, axis=0)
            distances = np.linalg.norm(latent_trajectory - centroid, axis=1)
            medoid_idx = np.argmin(distances)
            if return_indices:
                return latent_trajectory[medoid_idx:medoid_idx+1], np.array([medoid_idx])
            return latent_trajectory[medoid_idx:medoid_idx+1]

        elif method == 'frechet':
            # Euclidean mean (may not be an actual observed point)
            mean_point = np.mean(latent_trajectory, axis=0, keepdims=True)
            if return_indices:
                # Return closest actual point index
                distances = np.linalg.norm(latent_trajectory - mean_point, axis=1)
                closest_idx = np.argmin(distances)
                return mean_point, np.array([closest_idx])
            return mean_point

        # For other methods with k=1, proceed to K>1 section below
        # (changepoint, rdp, quantile can handle k=1)

    # K>1 methods: multiple keyframes
    if method == 'changepoint':
        try:
            import ruptures as rpt
        except ImportError:
            raise ImportError(
                "ruptures library required for changepoint method. "
                "Install with: pip install ruptures"
            )

        # Use Dynp (dynamic programming) algorithm which supports n_bkps parameter
        # Pelt requires penalty parameter, so we use Dynp for exact k segments
        algo = rpt.Dynp(model="rbf", min_size=2, jump=1).fit(latent_trajectory)
        # Find k segments (k-1 breakpoints)
        result = algo.predict(n_bkps=k-1)

        # Extract segment boundaries (add start point)
        segment_bounds = [0] + result

        # For each segment, find the medoid
        keyframe_indices = []
        for i in range(len(segment_bounds) - 1):
            start_idx = segment_bounds[i]
            end_idx = segment_bounds[i + 1]
            segment = latent_trajectory[start_idx:end_idx]

            # Find medoid within segment
            segment_centroid = np.mean(segment, axis=0)
            distances = np.linalg.norm(segment - segment_centroid, axis=1)
            medoid_offset = np.argmin(distances)
            keyframe_indices.append(start_idx + medoid_offset)

        keyframe_indices = np.array(keyframe_indices)
        summarized = latent_trajectory[keyframe_indices]

        if return_indices:
            return summarized, keyframe_indices
        return summarized

    elif method == 'rdp':
        # Ramer-Douglas-Peucker polyline simplification
        def rdp_recursive(points, indices, epsilon):
            if len(points) <= 2:
                return list(indices)

            # Find point with max distance from line
            start, end = points[0], points[-1]
            line_vec = end - start
            line_len = np.linalg.norm(line_vec)

            if line_len < 1e-10:
                return [indices[0], indices[-1]]

            line_unitvec = line_vec / line_len

            # Calculate perpendicular distances
            point_vecs = points - start
            projections = np.dot(point_vecs, line_unitvec)
            projected_points = start + np.outer(projections, line_unitvec)
            distances = np.linalg.norm(points - projected_points, axis=1)

            max_dist_idx = np.argmax(distances)
            max_dist = distances[max_dist_idx]

            if max_dist < epsilon:
                return [indices[0], indices[-1]]

            # Recursively simplify left and right segments
            left_indices = rdp_recursive(
                points[:max_dist_idx+1],
                indices[:max_dist_idx+1],
                epsilon
            )
            right_indices = rdp_recursive(
                points[max_dist_idx:],
                indices[max_dist_idx:],
                epsilon
            )

            # Concatenate lists (not arrays) to avoid broadcasting error
            return left_indices[:-1] + right_indices

        # Auto-tune epsilon if not provided
        if epsilon is None:
            # Binary search for epsilon that gives approximately k points
            all_indices = np.arange(time_steps)
            epsilon_low, epsilon_high = 0.0, np.max(np.linalg.norm(
                latent_trajectory[1:] - latent_trajectory[:-1], axis=1
            )) * 10

            for _ in range(20):  # Max 20 iterations
                epsilon_mid = (epsilon_low + epsilon_high) / 2
                result_indices = rdp_recursive(latent_trajectory, all_indices, epsilon_mid)
                n_points = len(result_indices)

                if n_points == k:
                    break
                elif n_points > k:
                    epsilon_low = epsilon_mid
                else:
                    epsilon_high = epsilon_mid

            epsilon = epsilon_mid

        # Apply RDP
        all_indices = np.arange(time_steps)
        keyframe_indices = rdp_recursive(latent_trajectory, all_indices, epsilon)
        keyframe_indices = np.array(keyframe_indices)

        # If we have more or fewer than k points, sample/interpolate
        if len(keyframe_indices) > k:
            # Sample k evenly spaced indices
            step = len(keyframe_indices) / k
            selected = [keyframe_indices[int(i * step)] for i in range(k)]
            keyframe_indices = np.array(selected)
        elif len(keyframe_indices) < k:
            # Add more points by finding largest gaps
            while len(keyframe_indices) < k:
                gaps = keyframe_indices[1:] - keyframe_indices[:-1]
                max_gap_idx = np.argmax(gaps)
                new_idx = (keyframe_indices[max_gap_idx] + keyframe_indices[max_gap_idx + 1]) // 2
                keyframe_indices = np.insert(keyframe_indices, max_gap_idx + 1, new_idx)

        summarized = latent_trajectory[keyframe_indices]

        if return_indices:
            return summarized, keyframe_indices
        return summarized

    elif method == 'quantile':
        # Quantile samples along arclength
        # Compute cumulative arclength
        segment_lengths = np.linalg.norm(
            latent_trajectory[1:] - latent_trajectory[:-1], axis=1
        )
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_length[-1]

        # Define k quantiles (including 0% and 100%)
        quantiles = np.linspace(0, 1, k)
        target_lengths = quantiles * total_length

        # Find indices closest to each target length
        keyframe_indices = []
        for target_len in target_lengths:
            idx = np.argmin(np.abs(cumulative_length - target_len))
            keyframe_indices.append(idx)

        keyframe_indices = np.array(keyframe_indices)
        summarized = latent_trajectory[keyframe_indices]

        if return_indices:
            return summarized, keyframe_indices
        return summarized

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: "
            "'changepoint', 'rdp', 'quantile', 'medoid', 'frechet'"
        )


# ------------------------------------------------------
# plotting methods
# ------------------------------------------------------

def plot_latent_changepoints_with_raw(
    latent_mean,
    fhr,
    up,
    save_path,
    sample_ids=None,
    n_changepoints=5,
    decimation_factor=16,
    cmap='viridis',
    figsize=(14, 8)
):
    """
    Visualize latent trajectories alongside raw signals with shared changepoints.

    Args:
        latent_mean: torch.Tensor or np.ndarray of shape (batch, time_steps, latent_dim)
            Latent representation mean per batch.
        fhr: torch.Tensor or np.ndarray of shape (batch, time_steps * decimation_factor)
            Fetal heart rate signals corresponding to each latent trajectory.
        up: torch.Tensor or np.ndarray of shape (batch, time_steps * decimation_factor)
            Uterine pressure signals corresponding to each latent trajectory.
        save_path: str, directory path where figures will be written.
        sample_ids: Optional sequence of identifiers (len == batch). Defaults to sequential indices.
        n_changepoints: int, number of changepoints to detect in the latent space (>=0).
        decimation_factor: int, ratio between raw signal length and latent length (default: 16).
        cmap: str, matplotlib colormap name for the latent heatmap.
        figsize: tuple, figure size passed to matplotlib.

    Returns:
        List[Dict[str, Any]] containing per-sample changepoint metadata with keys:
            - 'sample_id'
            - 'latent_changepoints' (np.ndarray of latent indices)
            - 'raw_changepoints' (np.ndarray of raw indices)

    Notes:
        - Changepoints are detected in latent space using ruptures Dynp segmentation.
        - Raw changepoints are derived by multiplying latent indices by decimation_factor.
    """
    try:
        import ruptures as rpt
    except ImportError as exc:
        raise ImportError(
            "plot_latent_changepoints_with_raw requires the 'ruptures' package. "
            "Install with: pip install ruptures"
        ) from exc

    def _to_numpy(data):
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    latent_np = _to_numpy(latent_mean)
    fhr_np = _to_numpy(fhr)
    up_np = _to_numpy(up)

    if latent_np.ndim != 3:
        raise ValueError(
            f"latent_mean must have shape (batch, time_steps, latent_dim); got {latent_np.shape}"
        )
    if fhr_np.ndim != 2 or up_np.ndim != 2:
        raise ValueError(
            f"fhr and up must have shape (batch, raw_time_steps); got {fhr_np.shape}, {up_np.shape}"
        )
    if latent_np.shape[0] != fhr_np.shape[0] or latent_np.shape[0] != up_np.shape[0]:
        raise ValueError(
            "latent_mean, fhr, and up must share the same batch dimension."
        )
    if fhr_np.shape != up_np.shape:
        raise ValueError("fhr and up must have identical shapes.")

    batch_size, latent_time_steps, _ = latent_np.shape
    if sample_ids is None:
        sample_ids = [f"sample_{idx}" for idx in range(batch_size)]
    if len(sample_ids) != batch_size:
        raise ValueError("sample_ids length must match batch size.")

    os.makedirs(save_path, exist_ok=True)

    results: List[Dict[str, Any]] = []

    max_bkps = max(0, int(n_changepoints))
    for batch_idx in range(batch_size):
        sample_id = str(sample_ids[batch_idx])
        latent_sample = latent_np[batch_idx]
        fhr_sample = fhr_np[batch_idx]
        up_sample = up_np[batch_idx]

        per_sample_max_bkps = min(max_bkps, max(0, latent_time_steps - 1))
        if per_sample_max_bkps > 0:
            algo = rpt.Dynp(model="rbf", min_size=2, jump=1).fit(latent_sample)
            bkps = algo.predict(n_bkps=per_sample_max_bkps)
            latent_cps = [int(idx) for idx in bkps if idx < latent_time_steps]
        else:
            latent_cps = []

        latent_cps = np.asarray(sorted({cp for cp in latent_cps if cp > 0}), dtype=int)
        raw_expected_len = latent_time_steps * decimation_factor
        raw_len = fhr_sample.shape[0]
        if raw_len != raw_expected_len:
            logger.warning(
                f"[{sample_id}] Raw signal length ({raw_len}) does not match "
                f"latent_time_steps * decimation_factor ({raw_expected_len})."
            )

        raw_cps = np.asarray(
            [
                min(raw_len - 1, cp_idx * decimation_factor)
                for cp_idx in latent_cps
                if raw_len > 0
            ],
            dtype=int
        )

        fig, axes = plt.subplots(
            2, 1, figsize=figsize, sharex=False, gridspec_kw={'height_ratios': [2, 1]}
        )

        im = axes[0].imshow(
            latent_sample,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            interpolation='nearest'
        )
        axes[0].set_ylabel('Latent Time Index')
        axes[0].set_title('Latent Representation (mean)')
        cbar = plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Activation', rotation=270, labelpad=15)

        for cp_idx in latent_cps:
            axes[0].axhline(
                cp_idx - 0.5,
                color='white',
                linestyle='--',
                linewidth=1.5,
                alpha=0.8
            )

        time_axis = np.arange(raw_len)
        axes[1].plot(time_axis, fhr_sample, label='FHR', color='#E74C3C', linewidth=1.2)
        axes[1].plot(time_axis, up_sample, label='UP', color='#2E86C1', linewidth=1.0)
        axes[1].set_xlabel('Raw Time Index')
        axes[1].set_ylabel('Normalized Amplitude')
        axes[1].set_title('Raw Signals (FHR & UP)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right')

        for raw_cp in raw_cps:
            axes[1].axvline(
                raw_cp,
                color='black',
                linestyle='--',
                linewidth=1.0,
                alpha=0.6
            )

        fig.suptitle(f'Latent Trajectory Changepoints - {sample_id}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        figure_path = os.path.join(
            save_path,
            f'latent_raw_changepoints_{sample_id}.png'
        )
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)

        results.append(
            {
                'sample_id': sample_id,
                'latent_changepoints': latent_cps,
                'raw_changepoints': raw_cps
            }
        )

    return results

def plot_complete_fhr_up_timeline(
    fhr,
    up,
    epoch,
    save_path,
    sample_id='sample_0',
    title='Complete FHR and UP Timeline',
    segment_duration_minutes=20,
    sampling_rate_hz=4,
    detect_changepoints=False,
    n_changepoints=5
):
    """
    Plot complete FHR and UP signals along timeline with missing segments as gaps.

    Args:
        fhr: torch.Tensor or np.ndarray of shape (Batch_size, time_steps)
             Fetal heart rate signal
        up: torch.Tensor or np.ndarray of shape (Batch_size, time_steps)
            Uterine pressure signal
        epoch: torch.Tensor or np.ndarray of shape (Batch_size,)
               Seconds before birth for each segment
        save_path: str, directory path to save plot
        sample_id: str or int, identifier for the sample
        title: str, plot title
        segment_duration_minutes: int, duration of each segment in minutes (default: 20)
        sampling_rate_hz: float, sampling rate in Hz (default: 4 Hz)
        detect_changepoints: bool, whether to detect and visualize changepoints in FHR (default: False)
        n_changepoints: int, number of changepoints to detect per segment (default: 5)

    Saves:
        - Interactive HTML plot: {save_path}/timeline_{sample_id}.html

    Notes:
        - Missing segments will appear as gaps in the timeline
        - The plot is wide and zoomable for detailed inspection
        - X-axis shows time in minutes before birth (negative values)
        - Changepoint detection uses ruptures library (install with: pip install ruptures)
        - Changepoints are detected independently for each 20-minute segment
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.error("plotly not installed. Install with: pip install plotly")
        return

    # Convert to numpy if needed
    if torch.is_tensor(fhr):
        fhr = fhr.cpu().numpy()
    if torch.is_tensor(up):
        up = up.cpu().numpy()
    if torch.is_tensor(epoch):
        epoch = epoch.cpu().numpy()

    sample_id = str(sample_id)
    os.makedirs(save_path, exist_ok=True)

    batch_size, time_steps = fhr.shape

    # Calculate time points for each sample
    time_per_step = 1.0 / sampling_rate_hz  # seconds per time step

    # Create subplot figure with 2 rows (FHR and UP)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Fetal Heart Rate (FHR)', 'Uterine Pressure (UP)'),
        row_heights=[0.5, 0.5]
    )

    # Plot each segment individually so they don't connect
    for i in range(batch_size):
        # Epoch is already negative (e.g., -3600 = 3600 seconds before birth)
        # Convert to minutes: -3600 seconds = -60 minutes
        segment_start_minutes = epoch[i] / 60.0  # Already negative

        # Create time array for this segment in minutes
        segment_times_minutes = segment_start_minutes + np.arange(time_steps) * time_per_step / 60.0

        # Add FHR trace for this segment
        fig.add_trace(
            go.Scatter(
                x=segment_times_minutes,
                y=fhr[i],
                mode='lines',
                name='FHR' if i == 0 else None,  # Only show legend for first segment
                legendgroup='FHR',
                showlegend=(i == 0),
                line=dict(color='#E74C3C', width=1.5),
                hovertemplate=f'Epoch {i}<br>Time: %{{x:.2f}} min before birth<br>FHR: %{{y:.1f}} bpm<extra></extra>'
            ),
            row=1, col=1
        )

        # Add UP trace for this segment
        fig.add_trace(
            go.Scatter(
                x=segment_times_minutes,
                y=up[i],
                mode='lines',
                name='UP' if i == 0 else None,  # Only show legend for first segment
                legendgroup='UP',
                showlegend=(i == 0),
                line=dict(color='#3498DB', width=1.5),
                hovertemplate=f'Epoch {i}<br>Time: %{{x:.2f}} min before birth<br>UP: %{{y:.1f}}<extra></extra>'
            ),
            row=2, col=1
        )

        # Detect and visualize changepoints if requested
        if detect_changepoints:
            try:
                import ruptures as rpt
            except ImportError:
                if i == 0:  # Only warn once
                    logger.warning(
                        "ruptures library required for changepoint detection. "
                        "Install with: pip install ruptures"
                    )
                continue

            # Apply changepoint detection to FHR signal for this segment
            fhr_signal = fhr[i].reshape(-1, 1)  # Shape (time_steps, 1)

            # Use Dynp (dynamic programming) algorithm
            algo = rpt.Dynp(model="rbf", min_size=2, jump=1).fit(fhr_signal)

            # Find n_changepoints segments (n_changepoints-1 breakpoints)
            # Handle case where segment is too short for requested changepoints
            max_possible_changepoints = max(1, time_steps // 10)  # At least 10 points per segment
            actual_n_changepoints = min(n_changepoints, max_possible_changepoints)

            try:
                changepoint_indices = algo.predict(n_bkps=actual_n_changepoints - 1)
            except Exception as e:
                logger.warning(f"Changepoint detection failed for epoch {i}: {e}")
                continue

            # Convert changepoint indices to time in minutes
            changepoint_times = segment_start_minutes + np.array(changepoint_indices) * time_per_step / 60.0

            # Add vertical lines at changepoints on FHR plot
            for cp_idx, cp_time in enumerate(changepoint_times[:-1]):  # Exclude last point (end of segment)
                fig.add_vline(
                    x=cp_time,
                    line=dict(color='green', width=2, dash='dash'),
                    opacity=0.6,
                    row=1, col=1,
                    annotation=dict(
                        text=f'CP',
                        showarrow=False,
                        font=dict(size=10, color='green'),
                        yshift=10
                    ) if cp_idx == 0 and i == 0 else None  # Only annotate first changepoint of first segment
                )

            # Add legend entry for changepoints (only once)
            if i == 0:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode='lines',
                        name='Changepoints',
                        line=dict(color='green', width=2, dash='dash'),
                        showlegend=True
                    ),
                    row=1, col=1
                )

    # Update layout for wide, zoomable plot
    fig.update_layout(
        title=dict(
            text=f'{title} - {sample_id}',
            font=dict(size=20)
        ),
        width=2000,  # Wide plot for detailed inspection
        height=600,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Update x-axes
    fig.update_xaxes(
        title_text='Time (minutes before birth)',
        gridcolor='lightgray',
        showgrid=True,
        showline=True,
        linewidth=2,
        linecolor='black',
        mirror=True,
        row=2, col=1
    )

    fig.update_xaxes(
        showline=True,
        linewidth=2,
        linecolor='black',
        mirror=True,
        showgrid=True,
        gridcolor='lightgray',
        row=1, col=1
    )

    # Update y-axis for FHR
    fig.update_yaxes(
        title_text='FHR (bpm)',
        gridcolor='lightgray',
        showgrid=True,
        showline=True,
        linewidth=2,
        linecolor='black',
        mirror=True,
        row=1, col=1
    )

    # Update y-axis for UP
    fig.update_yaxes(
        title_text='UP (mmHg)',
        gridcolor='lightgray',
        showgrid=True,
        showline=True,
        linewidth=2,
        linecolor='black',
        mirror=True,
        row=2, col=1
    )

    # Save the plot
    output_path = f'{save_path}/timeline_{sample_id}.html'
    fig.write_html(output_path)
    logger.info(f"Timeline plot saved to: {output_path}")


def plot_latent_trajectory(
    latent_trajectory,
    save_path,
    sample_id='sample_0',
    title='Latent Trajectory',
    color_by_time=True,
    point_size=20,
    arrow_scale=0.3,
    plot_animation=True,
    save_data=False
):
    """
    Plot and save latent trajectory with arrows showing temporal evolution.

    Args:
        latent_trajectory: np.ndarray of shape (1, time_steps, n_dims) or (time_steps, n_dims)
                          where n_dims is 2 or 3
        save_path: str, directory path to save plots
        sample_id: str or int, identifier for the sample (will be converted to string)
        title: str, plot title
        color_by_time: bool, color points by time progression
        point_size: int, size of trajectory points
        arrow_scale: float, scale factor for arrow size
        plot_animation: bool, whether to generate animated GIF (default: True)
        save_data: bool, whether to save trajectory data as .npy file (default: False)

    Saves:
        - Static plot: {save_path}/trajectory_{sample_id}.png
        - Animated GIF (if plot_animation=True): {save_path}/trajectory_{sample_id}_animated.gif
        - 3D HTML (if 3D): {save_path}/trajectory_{sample_id}_3d.html
        - Data file (if save_data=True): {save_path}/trajectory_{sample_id}_data.npy
    """
    # Convert sample_id to string
    sample_id = str(sample_id)
    import matplotlib.animation as animation
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d

    # Custom 3D arrow class
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def do_3d_projection(self, renderer=None):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            return np.min(zs)

    if latent_trajectory.ndim == 3:
        latent_trajectory = latent_trajectory.squeeze(0)

    time_steps, n_dims = latent_trajectory.shape
    os.makedirs(save_path, exist_ok=True)

    # Save trajectory data if requested
    if save_data:
        data_file_path = f'{save_path}/trajectory_{sample_id}_data.npy'
        np.save(data_file_path, latent_trajectory)

    if color_by_time:
        colors = plt.cm.viridis(np.linspace(0, 1, time_steps))
    else:
        colors = ['blue'] * time_steps

    if n_dims == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(latent_trajectory[:, 0], latent_trajectory[:, 1],
                            c=np.arange(time_steps), cmap='viridis',
                            s=point_size, zorder=3)

        for i in range(time_steps - 1):
            ax.annotate(
                '',
                xy=latent_trajectory[i+1],
                xytext=latent_trajectory[i],
                arrowprops=dict(
                    arrowstyle='->',
                    lw=1.5,
                    color=colors[i],
                    alpha=0.7,
                    edgecolor='black',
                    linewidth=0.3
                )
            )

        ax.scatter(
            latent_trajectory[0, 0],
            latent_trajectory[0, 1],
            c='green', s=100, marker='o', label='Start', zorder=4
        )
        ax.scatter(
            latent_trajectory[-1, 0],
            latent_trajectory[-1, 1],
            c='red', s=100, marker='X', label='End', zorder=4
        )

        plt.colorbar(scatter, ax=ax, label='Time Step')
        ax.set_xlabel('Latent Dim 1')
        ax.set_ylabel('Latent Dim 2')
        ax.set_title(f'{title} - {sample_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/trajectory_{sample_id}.png', dpi=150)
        plt.close()

        # 2D Animation
        if plot_animation:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_xlim(latent_trajectory[:, 0].min() - 0.5, latent_trajectory[:, 0].max() + 0.5)
            ax.set_ylim(latent_trajectory[:, 1].min() - 0.5, latent_trajectory[:, 1].max() + 0.5)
            ax.set_xlabel('Latent Dim 1')
            ax.set_ylabel('Latent Dim 2')
            ax.set_title(f'{title} - {sample_id} (Animated)')
            ax.grid(True, alpha=0.3)

            line, = ax.plot([], [], 'b-', alpha=0.6, lw=2)
            point, = ax.plot([], [], 'ro', markersize=8)

            def init():
                line.set_data([], [])
                point.set_data([], [])
                return line, point

            def animate(i):
                line.set_data(latent_trajectory[:i+1, 0], latent_trajectory[:i+1, 1])
                point.set_data([latent_trajectory[i, 0]], [latent_trajectory[i, 1]])
                return line, point

            anim = animation.FuncAnimation(
                fig, animate, init_func=init,
                frames=time_steps, interval=50, blit=True
            )
            anim.save(
                f'{save_path}/trajectory_{sample_id}_animated.gif',
                writer='pillow', fps=20
            )
            plt.close()

    elif n_dims == 3:
        # 3D Static Plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            latent_trajectory[:, 0], latent_trajectory[:, 1],
            latent_trajectory[:, 2], c=np.arange(time_steps),
            cmap='viridis', s=point_size
        )

        # Draw arrows in 3D
        for i in range(time_steps - 1):
            arrow = Arrow3D(
                [latent_trajectory[i, 0], latent_trajectory[i+1, 0]],
                [latent_trajectory[i, 1], latent_trajectory[i+1, 1]],
                [latent_trajectory[i, 2], latent_trajectory[i+1, 2]],
                mutation_scale=15,
                arrowstyle='-|>',
                color=colors[i],
                alpha=0.7,
                shrinkA=0,
                shrinkB=0
            )
            # Set edge properties after initialization to avoid warning
            arrow.set_edgecolor('black')
            arrow.set_linewidth(1.5)
            ax.add_artist(arrow)

        ax.scatter(
            latent_trajectory[0, 0], latent_trajectory[0, 1],
            latent_trajectory[0, 2], c='green', s=150, marker='o', label='Start'
        )
        ax.scatter(latent_trajectory[-1, 0], latent_trajectory[-1, 1],
                  latent_trajectory[-1, 2], c='red', s=150, marker='X', label='End')

        plt.colorbar(scatter, ax=ax, label='Time Step', shrink=0.6)
        ax.set_xlabel('Latent Dim 1')
        ax.set_ylabel('Latent Dim 2')
        ax.set_zlabel('Latent Dim 3')
        ax.set_title(f'{title} - {sample_id}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path}/trajectory_{sample_id}.png', dpi=150)
        plt.close()

        # 3D Animation
        if plot_animation:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(latent_trajectory[:, 0].min() - 0.5, latent_trajectory[:, 0].max() + 0.5)
            ax.set_ylim(latent_trajectory[:, 1].min() - 0.5, latent_trajectory[:, 1].max() + 0.5)
            ax.set_zlim(latent_trajectory[:, 2].min() - 0.5, latent_trajectory[:, 2].max() + 0.5)
            ax.set_xlabel('Latent Dim 1')
            ax.set_ylabel('Latent Dim 2')
            ax.set_zlabel('Latent Dim 3')
            ax.set_title(f'{title} - {sample_id} (Animated)')

            line, = ax.plot([], [], [], 'b-', alpha=0.6, lw=2)
            point, = ax.plot([], [], [], 'ro', markersize=8)

            def init():
                line.set_data([], [])
                line.set_3d_properties([])
                point.set_data([], [])
                point.set_3d_properties([])
                return line, point

            def animate(i):
                line.set_data(latent_trajectory[:i+1, 0], latent_trajectory[:i+1, 1])
                line.set_3d_properties(latent_trajectory[:i+1, 2])
                point.set_data([latent_trajectory[i, 0]], [latent_trajectory[i, 1]])
                point.set_3d_properties([latent_trajectory[i, 2]])
                return line, point

            anim = animation.FuncAnimation(fig, animate, init_func=init,
                                          frames=time_steps, interval=50, blit=True)
            anim.save(f'{save_path}/trajectory_{sample_id}_animated.gif',
                     writer='pillow', fps=20)
            plt.close()

        # 3D Interactive HTML using plotly
        try:
            import plotly.graph_objects as go

            fig = go.Figure(data=[go.Scatter3d(
                x=latent_trajectory[:, 0],
                y=latent_trajectory[:, 1],
                z=latent_trajectory[:, 2],
                mode='markers+lines',
                marker=dict(
                    size=5,
                    color=np.arange(time_steps),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Time Step",
                        x=1.15,
                        xanchor='left',
                        thickness=15,
                        len=0.7
                    )
                ),
                line=dict(color='blue', width=2),
                text=[f'Time: {i}' for i in range(time_steps)],
                hoverinfo='text',
                name='Trajectory',
                showlegend=True
            )])

            fig.add_trace(go.Scatter3d(
                x=[latent_trajectory[0, 0]], y=[latent_trajectory[0, 1]], z=[latent_trajectory[0, 2]],
                mode='markers', marker=dict(size=10, color='green'), name='Start'
            ))

            fig.add_trace(go.Scatter3d(
                x=[latent_trajectory[-1, 0]], y=[latent_trajectory[-1, 1]], z=[latent_trajectory[-1, 2]],
                mode='markers', marker=dict(size=10, color='red', symbol='x'), name='End'
            ))

            fig.update_layout(
                title=f'{title} - {sample_id} (Interactive 3D)',
                scene=dict(
                    xaxis_title='Latent Dim 1',
                    yaxis_title='Latent Dim 2',
                    zaxis_title='Latent Dim 3'
                ),
                width=1200, height=800,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    xanchor='left',
                    yanchor='top',
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='black',
                    borderwidth=1
                )
            )

            fig.write_html(f'{save_path}/trajectory_{sample_id}_3d.html')
        except ImportError:
            logger.warning("plotly not installed, skipping interactive 3D HTML plot")


def compare_trajectory_classes(
    trajectory_file_lists,
    save_path,
    class_labels=None,
    title='Trajectory Comparison',
    colormaps=None,
    plot_animation=True,
    point_size=20,
    alpha=0.6
):
    """
    Compare latent trajectories from different classes by plotting them together.

    Args:
        trajectory_file_lists: List[List[str]], 2-3 lists of file paths to .npy trajectory data
                              Each list represents a different class
        save_path: str, directory path to save comparison plots
        class_labels: Optional[List[str]], labels for each class (default: Class 0, Class 1, ...)
        title: str, plot title
        colormaps: Optional[List[str]], matplotlib colormaps for each class
                   (default: ['Reds', 'Blues', 'Greens'])
        plot_animation: bool, whether to generate animated GIF (default: True)
        point_size: int, size of trajectory points
        alpha: float, transparency of trajectories (0-1)

    Saves:
        - Static plot: {save_path}/trajectory_comparison.png
        - Animated GIF (if plot_animation=True): {save_path}/trajectory_comparison_animated.gif
        - 3D HTML (if 3D): {save_path}/trajectory_comparison_3d.html
    """
    import matplotlib.animation as animation
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d

    # Custom 3D arrow class
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def do_3d_projection(self, renderer=None):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            return np.min(zs)

    # Validate input
    n_classes = len(trajectory_file_lists)
    if n_classes < 2 or n_classes > 3:
        raise ValueError(f"Expected 2-3 trajectory lists, got {n_classes}")

    # Set default labels
    if class_labels is None:
        class_labels = [f'Class {i}' for i in range(n_classes)]
    elif len(class_labels) != n_classes:
        raise ValueError(f"Number of labels ({len(class_labels)}) must match number of classes ({n_classes})")

    # Set default colormaps
    if colormaps is None:
        colormaps = ['Reds', 'Blues', 'Greens'][:n_classes]
    elif len(colormaps) != n_classes:
        raise ValueError(f"Number of colormaps ({len(colormaps)}) must match number of classes ({n_classes})")

    os.makedirs(save_path, exist_ok=True)

    # Load all trajectories
    all_trajectories = []
    all_class_ids = []

    for class_idx, file_list in enumerate(trajectory_file_lists):
        for file_path in file_list:
            trajectory = np.load(file_path)
            if trajectory.ndim == 3:
                trajectory = trajectory.squeeze(0)
            all_trajectories.append(trajectory)
            all_class_ids.append(class_idx)

    if len(all_trajectories) == 0:
        logger.error("No trajectories loaded")
        return

    # Determine dimensionality (assume all have same dimensionality)
    n_dims = all_trajectories[0].shape[1]
    if n_dims not in [2, 3]:
        raise ValueError(f"Trajectories must be 2D or 3D, got {n_dims}D")

    # Verify all trajectories have same dimensionality
    for traj in all_trajectories:
        if traj.shape[1] != n_dims:
            raise ValueError(f"All trajectories must have same dimensionality. Expected {n_dims}, got {traj.shape[1]}")

    # Plot trajectories
    if n_dims == 2:
        # 2D Static Plot
        fig, ax = plt.subplots(figsize=(12, 10))

        for traj_idx, (trajectory, class_idx) in enumerate(zip(all_trajectories, all_class_ids)):
            time_steps = trajectory.shape[0]
            cmap = plt.get_cmap(colormaps[class_idx])
            colors = cmap(np.linspace(0.3, 0.9, time_steps))

            # Plot trajectory points
            scatter = ax.scatter(
                trajectory[:, 0], trajectory[:, 1],
                c=np.arange(time_steps), cmap=colormaps[class_idx],
                s=point_size, alpha=alpha,
                label=class_labels[class_idx] if traj_idx == 0 or all_class_ids[traj_idx-1] != class_idx else None
            )

            # Draw arrows
            for i in range(time_steps - 1):
                ax.annotate(
                    '',
                    xy=trajectory[i+1],
                    xytext=trajectory[i],
                    arrowprops=dict(
                        arrowstyle='->',
                        lw=1.0,
                        color=colors[i],
                        alpha=alpha * 0.8
                    )
                )

            # Mark start and end points
            ax.scatter(
                trajectory[0, 0], trajectory[0, 1],
                c=[cmap(0.3)], s=80, marker='o',
                edgecolors='black', linewidths=1.5, alpha=alpha, zorder=4
            )
            ax.scatter(
                trajectory[-1, 0], trajectory[-1, 1],
                c=[cmap(0.9)], s=80, marker='X',
                edgecolors='black', linewidths=1.5, alpha=alpha, zorder=4
            )

        ax.set_xlabel('Latent Dim 1', fontsize=12)
        ax.set_ylabel('Latent Dim 2', fontsize=12)
        ax.set_title(f'{title}', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/trajectory_comparison.png', dpi=150)
        plt.close()

        # 2D Animation
        if plot_animation:
            fig, ax = plt.subplots(figsize=(12, 10))

            # Get overall bounds
            all_x = np.concatenate([traj[:, 0] for traj in all_trajectories])
            all_y = np.concatenate([traj[:, 1] for traj in all_trajectories])
            ax.set_xlim(all_x.min() - 0.5, all_x.max() + 0.5)
            ax.set_ylim(all_y.min() - 0.5, all_y.max() + 0.5)
            ax.set_xlabel('Latent Dim 1', fontsize=12)
            ax.set_ylabel('Latent Dim 2', fontsize=12)
            ax.set_title(f'{title} (Animated)', fontsize=14)
            ax.grid(True, alpha=0.3)

            # Create legend
            for class_idx, label in enumerate(class_labels):
                ax.plot([], [], color=plt.get_cmap(colormaps[class_idx])(0.6),
                       label=label, linewidth=2)
            ax.legend(loc='best', fontsize=10)

            max_time_steps = max(traj.shape[0] for traj in all_trajectories)

            lines = []
            points = []
            for class_idx in range(n_classes):
                cmap = plt.get_cmap(colormaps[class_idx])
                line, = ax.plot([], [], alpha=alpha, lw=2, color=cmap(0.6))
                point, = ax.plot([], [], 'o', markersize=8, color=cmap(0.8))
                lines.append(line)
                points.append(point)

            def init():
                for line, point in zip(lines, points):
                    line.set_data([], [])
                    point.set_data([], [])
                return lines + points

            def animate(i):
                for traj_idx, (trajectory, class_idx) in enumerate(zip(all_trajectories, all_class_ids)):
                    time_steps = trajectory.shape[0]
                    frame_idx = min(i, time_steps - 1)

                    lines[class_idx].set_data(
                        trajectory[:frame_idx+1, 0],
                        trajectory[:frame_idx+1, 1]
                    )
                    points[class_idx].set_data(
                        [trajectory[frame_idx, 0]],
                        [trajectory[frame_idx, 1]]
                    )
                return lines + points

            anim = animation.FuncAnimation(
                fig, animate, init_func=init,
                frames=max_time_steps, interval=50, blit=True
            )
            anim.save(
                f'{save_path}/trajectory_comparison_animated.gif',
                writer='pillow', fps=20
            )
            plt.close()

    elif n_dims == 3:
        # 3D Static Plot
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')

        for traj_idx, (trajectory, class_idx) in enumerate(zip(all_trajectories, all_class_ids)):
            time_steps = trajectory.shape[0]
            cmap = plt.get_cmap(colormaps[class_idx])
            colors = cmap(np.linspace(0.3, 0.9, time_steps))

            # Plot trajectory points
            scatter = ax.scatter(
                trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                c=np.arange(time_steps), cmap=colormaps[class_idx],
                s=point_size, alpha=alpha,
                label=class_labels[class_idx] if traj_idx == 0 or all_class_ids[traj_idx-1] != class_idx else None
            )

            # Draw arrows in 3D
            for i in range(time_steps - 1):
                arrow = Arrow3D(
                    [trajectory[i, 0], trajectory[i+1, 0]],
                    [trajectory[i, 1], trajectory[i+1, 1]],
                    [trajectory[i, 2], trajectory[i+1, 2]],
                    mutation_scale=15,
                    arrowstyle='-|>',
                    color=colors[i],
                    alpha=alpha * 0.8,
                    shrinkA=0,
                    shrinkB=0
                )
                arrow.set_edgecolor(colors[i])
                arrow.set_linewidth(1.0)
                ax.add_artist(arrow)

            # Mark start and end points
            ax.scatter(
                trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                c=[cmap(0.3)], s=100, marker='o',
                edgecolors='black', linewidths=1.5, alpha=alpha
            )
            ax.scatter(
                trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                c=[cmap(0.9)], s=100, marker='X',
                edgecolors='black', linewidths=1.5, alpha=alpha
            )

        ax.set_xlabel('Latent Dim 1', fontsize=12)
        ax.set_ylabel('Latent Dim 2', fontsize=12)
        ax.set_zlabel('Latent Dim 3', fontsize=12)
        ax.set_title(f'{title}', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{save_path}/trajectory_comparison.png', dpi=150)
        plt.close()

        # 3D Animation
        if plot_animation:
            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')

            # Get overall bounds
            all_x = np.concatenate([traj[:, 0] for traj in all_trajectories])
            all_y = np.concatenate([traj[:, 1] for traj in all_trajectories])
            all_z = np.concatenate([traj[:, 2] for traj in all_trajectories])
            ax.set_xlim(all_x.min() - 0.5, all_x.max() + 0.5)
            ax.set_ylim(all_y.min() - 0.5, all_y.max() + 0.5)
            ax.set_zlim(all_z.min() - 0.5, all_z.max() + 0.5)
            ax.set_xlabel('Latent Dim 1', fontsize=12)
            ax.set_ylabel('Latent Dim 2', fontsize=12)
            ax.set_zlabel('Latent Dim 3', fontsize=12)
            ax.set_title(f'{title} (Animated)', fontsize=14)

            # Create legend
            for class_idx, label in enumerate(class_labels):
                ax.plot([], [], [], color=plt.get_cmap(colormaps[class_idx])(0.6),
                       label=label, linewidth=2)
            ax.legend(loc='best', fontsize=10)

            max_time_steps = max(traj.shape[0] for traj in all_trajectories)

            lines = []
            points = []
            for class_idx in range(n_classes):
                cmap = plt.get_cmap(colormaps[class_idx])
                line, = ax.plot([], [], [], alpha=alpha, lw=2, color=cmap(0.6))
                point, = ax.plot([], [], [], 'o', markersize=8, color=cmap(0.8))
                lines.append(line)
                points.append(point)

            def init():
                for line, point in zip(lines, points):
                    line.set_data([], [])
                    line.set_3d_properties([])
                    point.set_data([], [])
                    point.set_3d_properties([])
                return lines + points

            def animate(i):
                for traj_idx, (trajectory, class_idx) in enumerate(zip(all_trajectories, all_class_ids)):
                    time_steps = trajectory.shape[0]
                    frame_idx = min(i, time_steps - 1)

                    lines[class_idx].set_data(
                        trajectory[:frame_idx+1, 0],
                        trajectory[:frame_idx+1, 1]
                    )
                    lines[class_idx].set_3d_properties(trajectory[:frame_idx+1, 2])

                    points[class_idx].set_data(
                        [trajectory[frame_idx, 0]],
                        [trajectory[frame_idx, 1]]
                    )
                    points[class_idx].set_3d_properties([trajectory[frame_idx, 2]])
                return lines + points

            anim = animation.FuncAnimation(
                fig, animate, init_func=init,
                frames=max_time_steps, interval=50, blit=True
            )
            anim.save(
                f'{save_path}/trajectory_comparison_animated.gif',
                writer='pillow', fps=20
            )
            plt.close()

        # 3D Interactive HTML using plotly
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            for traj_idx, (trajectory, class_idx) in enumerate(zip(all_trajectories, all_class_ids)):
                time_steps = trajectory.shape[0]
                cmap = plt.get_cmap(colormaps[class_idx])

                # Convert matplotlib colormap to RGB values
                color_values = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
                               for r, g, b, _ in cmap(np.linspace(0.3, 0.9, time_steps))]

                fig.add_trace(go.Scatter3d(
                    x=trajectory[:, 0],
                    y=trajectory[:, 1],
                    z=trajectory[:, 2],
                    mode='markers+lines',
                    marker=dict(
                        size=5,
                        color=np.arange(time_steps),
                        colorscale=[[i/(time_steps-1), color_values[i]] for i in range(time_steps)],
                        showscale=False
                    ),
                    line=dict(color=color_values[time_steps//2], width=2),
                    text=[f'{class_labels[class_idx]} - Time: {i}' for i in range(time_steps)],
                    hoverinfo='text',
                    name=class_labels[class_idx] if traj_idx == 0 or all_class_ids[traj_idx-1] != class_idx else None,
                    showlegend=traj_idx == 0 or all_class_ids[traj_idx-1] != class_idx,
                    legendgroup=f'class_{class_idx}',
                    opacity=alpha
                ))

            fig.update_layout(
                title=f'{title} (Interactive 3D)',
                scene=dict(
                    xaxis_title='Latent Dim 1',
                    yaxis_title='Latent Dim 2',
                    zaxis_title='Latent Dim 3'
                ),
                width=1400, height=1000,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    xanchor='left',
                    yanchor='top',
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='black',
                    borderwidth=1
                )
            )

            fig.write_html(f'{save_path}/trajectory_comparison_3d.html')
        except ImportError:
            logger.warning("plotly not installed, skipping interactive 3D HTML plot")


# ------------------------------------------------------
# Graph Model
# ------------------------------------------------------

class LatentTrajectoryGraph(SeqVAEGraphModelTest):
    def __init__(self, config_file_path):
        super().__init__(config_file_path)
        if self.cuda_devices:
            cuda_device = f"cuda:{self.cuda_devices[0]}"
        else:
            cuda_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def create_pytorch_testing_model(self):
        self.setup_config()
        
        checkpoint_path = self.config.get('seqvae_testing').get('test_checkpoint_path')
        self.pytorch_model = SeqVaeTeb()
        self.pytorch_model = load_checkpoint_torch(model=self.pytorch_model, checkpoint_path=checkpoint_path)
        self.pytorch_model.eval()
        return self.pytorch_model
    
    def latent_trajectory_tests(self, test_dataloader, save_path=None, laten_dim_reduction_type='pca', time_dim_reduction_type='changepoint'):
        config_latent = self.config.get('seqvae_testing').get('latent_representation')
        if save_path is not None:
            save_dir = os.path.join(self.test_results_dir, save_path)
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = self.test_results_dir

        all_batches = list(test_dataloader)
        selected_batches = random.sample(all_batches, min(10, len(all_batches)))

        with torch.inference_mode():
            for batch in selected_batches:
                fhr_st = batch.fhr_st.to(self.device)
                fhr_ph = batch.fhr_ph.to(self.device)
                fhr_up_ph = batch.fhr_up_ph.to(self.device)
                fhr = batch.fhr.to(self.device)
                up = batch.up.to(self.device)
                epoch = batch.epoch  # Shape (Batch_size,)
                guid_in_batch = batch['guid'][0]

                # Sort all batch data by epoch values
                sort_indices = torch.argsort(epoch)
                fhr_st = fhr_st[sort_indices]
                fhr_ph = fhr_ph[sort_indices]
                fhr_up_ph = fhr_up_ph[sort_indices]
                fhr = fhr[sort_indices]  # shape: (Batch, time_step*16)
                up = up[sort_indices]  # shape: (Batch, time_step*16)
                epoch = epoch[sort_indices]

                fhr_up_plot_path = os.path.join(save_dir, "fhr-up-signals")
                plot_complete_fhr_up_timeline(
                    fhr=fhr,
                    up=up,
                    epoch=epoch,
                    save_path=fhr_up_plot_path,
                    sample_id=f"{guid_in_batch}",
                    title='Complete FHR and UP Timeline',
                    sampling_rate_hz=4,
                    detect_changepoints=True
                )
                self.pytorch_model.to(self.device)
                model_output = self.pytorch_model(
                    y_st=fhr_st,
                    y_ph=fhr_ph,
                    x_ph=fhr_up_ph,
                )
                latent_mean = model_output['mu_post']  # Shape (Batch_size, time_steps, latent_dim)
                latent_var = torch.exp(model_output['logvar_post'])  # (Batch_size, time_steps, latent_dim)
                proccessed_latent_mean = latent_mean
                # proccessed_latent_mean = preprocess_latent(latent=latent_mean, denoise=False)  # Shape (Batch_size, time_steps, latent_dim)
                # proccessed_latent_var = preprocess_latent(latent=latent_var, denoise=False)  # (Batch_size, time_steps, latent_dim)
                
                
                plot_latent_changepoints_with_raw(
                    fhr=fhr,
                    up=up,
                    epoch=epoch,
                    latent_trajectory=latent_mean,
                    save_path=fhr_up_plot_path,
                    sample_id=f"{guid_in_batch}",
                    title='FHR, UP and Latent Changepoints',
                    sampling_rate_hz=4,
                    point_size=50
                )
                
                # reduced_latent_mean = reduce_latent_dimensionality(
                #     proccessed_latent_mean,
                #     method=laten_dim_reduction_type,
                #     n_components=3,
                #     n_neighbors=15,
                #     min_dist=0.1,
                #     return_reducer=False
                # )

                # epoch_laten_trajectory_path = os.path.join(save_dir, "per_epoch_trajectory")
                # plot_latent_trajectory(
                #     reduced_latent_mean[0],
                #     epoch_laten_trajectory_path,
                #     sample_id=f"_{guid_in_batch}-{epoch[0]}-start",
                #     title='Latent Trajectory',
                #     color_by_time=True,
                #     point_size=100,
                #     arrow_scale=0.3
                # )
                
                # plot_latent_trajectory(
                #     reduced_latent_mean[-1],
                #     epoch_laten_trajectory_path,
                #     sample_id=f"_{guid_in_batch}-{epoch[1]}-end",
                #     title='Latent Trajectory',
                #     color_by_time=True,
                #     point_size=100,
                #     arrow_scale=0.3
                # )

                
                
                # batch_size, reduced_time_steps, reduced_latent_dim = reduced_latent_mean.shape
                # all_reduced_latent_mean = reduced_latent_mean.reshape(
                #     batch_size * reduced_time_steps, reduced_latent_dim
                # )
                # all_laten_trajectory_path = os.path.join(save_dir, "all_epochs_trajectory_complete")
                # plot_latent_trajectory(
                #     all_reduced_latent_mean,
                #     all_laten_trajectory_path,
                #     sample_id=f"_{guid_in_batch}",
                #     title='Latent Trajectory',
                #     color_by_time=True,
                #     point_size=100,
                #     arrow_scale=0.3,
                #     plot_animation=False,
                #     save_data=True
                # )                
                
                # reduced_latent_mean_summary = summarize_trajectory(
                #     reduced_latent_mean,
                #     k=10,
                #     method=time_dim_reduction_type,
                #     epsilon=None,
                #     return_indices=False
                # )   # shape (Bathc_size, reduces_time_steps, reduced_latent_dim)


                # epoch_laten_trajectory_path = os.path.join(save_dir, "per_epoch_trajectory_summary")
                # plot_latent_trajectory(
                #     reduced_latent_mean_summary[0],
                #     epoch_laten_trajectory_path,
                #     sample_id=f"_{guid_in_batch}-{epoch[0]}-start",
                #     title='Latent Trajectory',
                #     color_by_time=True,
                #     point_size=100,
                #     arrow_scale=0.3
                # )
                
                # plot_latent_trajectory(
                #     reduced_latent_mean_summary[-1],
                #     epoch_laten_trajectory_path,
                #     sample_id=f"_{guid_in_batch}-{epoch[-1]}-end",
                #     title='Latent Trajectory',
                #     color_by_time=True,
                #     point_size=100,
                #     arrow_scale=0.3
                # )

                # # Reshape to (Batch_size * reduced_time_steps, reduced_latent_dim)
                # # keep epoch-sorted order from the earlier sorting
                # batch_size, reduced_time_steps, reduced_latent_dim = reduced_latent_mean_summary.shape
                # all_reduced_latent_mean_summary = reduced_latent_mean_summary.reshape(
                #     batch_size * reduced_time_steps, reduced_latent_dim
                # )
                # all_laten_trajectory_path = os.path.join(save_dir, "all_epochs_trajectory_summary")
                # plot_latent_trajectory(
                #     all_reduced_latent_mean_summary,
                #     all_laten_trajectory_path,
                #     sample_id=f"_{guid_in_batch}",
                #     title='Latent Trajectory',
                #     color_by_time=True,
                #     point_size=100,
                #     arrow_scale=0.3,
                #     save_data=True
                # )
                



def main():
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    config_file_path = 'SeqVAE-TEB-original/config_v.yaml'
    project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if not os.path.isabs(config_file_path):
        config_file_path = os.path.join(project_root, config_file_path)

    config_file_path = os.path.normpath(config_file_path)
    if not os.path.exists(config_file_path):
        logger.error(f"Configuration file not found at the resolved path: {config_file_path}")
    with open(config_file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    graph_model = LatentTrajectoryGraph(config_file_path=config_file_path)
    graph_model.setup_config()
    graph_model.create_pytorch_testing_model()

    dataset_config = config.get('dataset_config')
    test_dataset_path_list = dataset_config.get('vae_test_datasets')
    dataloader_config = dataset_config.get('dataloader_config')

    for test_dataset_path in test_dataset_path_list:
        base_name = os.path.basename(test_dataset_path)
        file_name = os.path.splitext(base_name)[0]

        eligible_guids, test_dataloader = build_guid_filtered_dataloader(
            [test_dataset_path],
            max_guids=11,
            min_samples=22,
            stats_path=dataset_config.get('stat_path'),
            normalize_fields=dataloader_config.get('normalize_fields'),
            **dataloader_config.get('dataset_kwargs', {})
        )
        latent_dim_reduction_methods = ['pca', 'isomap', 'umap', 'tsne', 'diffusion']
        latent_time_reduction_methods = ['changepoint', 'rdp', 'quantile', 'medoid', 'frechet']
        graph_model.latent_trajectory_tests(test_dataloader=test_dataloader,
                                            save_path=file_name,
                                            laten_dim_reduction_type='isomap', time_dim_reduction_type='changepoint')

    
    
if __name__ == '__main__':
    main()
    # compare_trajectory_classes(
    #     [
    #         [
    #             r"/data/deid/isilon/MS_model/seq_vae_teb_results/pre_training/2025-10-09--[18-38]--latent_test_trajectory_no_normalization_isomap_changepoint/test_results/hie_cs/all_epochs_trajectory_complete/trajectory__EDB32D23E6D148908E7B84588F0E04CA_data.npy",
    #         ],
    #         [
    #             r"/data/deid/isilon/MS_model/seq_vae_teb_results/pre_training/2025-10-09--[18-38]--latent_test_trajectory_no_normalization_isomap_changepoint/test_results/healthy_no_bg_no_cs/all_epochs_trajectory_complete/trajectory__2203847F514E487884D410B8605BFA2F_data.npy",
    #         ]
    #     ],
    #     save_path=r"/data/deid/isilon/MS_model/seq_vae_teb_results/pre_training/2025-10-09--[18-38]--latent_test_trajectory_no_normalization_isomap_changepoint/train_results",
    #     class_labels=["HIE", "Healthy"],
    #     title='Trajectory Comparison',
    #     colormaps=None,
    #     plot_animation=False,
    #     point_size=20,
    #     alpha=0.6
    # )
