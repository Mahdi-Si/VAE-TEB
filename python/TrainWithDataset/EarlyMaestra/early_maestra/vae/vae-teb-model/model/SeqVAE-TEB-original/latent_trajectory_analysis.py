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
from matplotlib.gridspec import GridSpec
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
    return_indices=False,
    precomputed_indices=None
):
    """
    Summarize latent trajectory in time by selecting k representative points.

    Args:
        latent_trajectory: np.ndarray of shape (time_steps, n_dims) or (batch_size, time_steps, n_dims)
        k: int, number of keyframes to extract (e.g., 1 for single summary, 5 for segments)
        method: str, summarization strategy
            - 'changepoint': Low-complexity PELT segmentation via ruptures (pass changepoint_algo='gradient'
              to reuse the built-in fast detector)
            - 'rdp': Ramer-Douglas-Peucker polyline simplification (preserves shape)
            - 'quantile': Quantile samples along arclength (even coverage)
            - 'medoid': Single medoid (most representative point) - only for k=1
            - 'frechet': Fréchet mean (Euclidean mean) - only for k=1
        epsilon: float, RDP tolerance parameter (only for method='rdp')
            If None, automatically tuned to get approximately k vertices
        return_indices: bool, if True return time indices of selected points
        precomputed_indices: Optional iterable of changepoint indices to reuse when method='changepoint'.

    Returns:
        summarized_trajectory: np.ndarray of shape (k, n_dims) or (batch_size, k, n_dims)
        indices (optional): np.ndarray of shape (k,) or (batch_size, k) - time indices

    Notes:
        - For nonlinear embeddings (UMAP/Isomap), use 'medoid', 'changepoint', 'rdp', or 'quantile'
        - Avoid 'frechet' (average) for nonlinear embeddings as it may produce unrealizable points
        - 'changepoint' defaults to a low-complexity ruptures implementation (requires `pip install ruptures`);
          set changepoint_algo='gradient' to activate the built-in detector when ruptures is unavailable
    """
    # Handle batch dimension
    if latent_trajectory.ndim == 3:
        batch_size = latent_trajectory.shape[0]
        results = []
        indices_list = []
        precomputed_seq = None
        precomputed_map = None
        if precomputed_indices is not None:
            if isinstance(precomputed_indices, dict):
                precomputed_map = precomputed_indices
            elif isinstance(precomputed_indices, (list, tuple)):
                precomputed_seq = list(precomputed_indices)
            else:
                raise TypeError(
                    "precomputed_indices must be a dict or list/tuple when latent_trajectory has a batch dimension."
                )
        for b in range(batch_size):
            sample_precomputed = None
            if precomputed_map is not None:
                sample_precomputed = precomputed_map.get(b)
            elif precomputed_seq is not None and b < len(precomputed_seq):
                sample_precomputed = precomputed_seq[b]
            result = summarize_trajectory(
                latent_trajectory[b],
                k=k,
                method=method,
                epsilon=epsilon,
                return_indices=return_indices,
                precomputed_indices=sample_precomputed
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
        changepoint_indices = None

        if precomputed_indices is not None:
            changepoint_indices = np.asarray(precomputed_indices, dtype=int)
        else:
            detector_fn = _create_changepoint_detector()
            max_breakpoints = max(0, int(k) - 1)
            changepoint_indices = detector_fn(latent_trajectory, max_breakpoints)

        if changepoint_indices is None:
            changepoint_indices = np.asarray([], dtype=int)

        changepoint_indices = np.asarray(
            sorted(
                {
                    int(idx)
                    for idx in np.asarray(changepoint_indices).tolist()
                    if 0 < int(idx) < time_steps
                }
            ),
            dtype=int
        )

        segment_bounds = [0] + changepoint_indices.tolist() + [time_steps]

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

def _create_changepoint_detector(
    changepoint_algo: str = 'pelt',
    changepoint_model: Optional[str] = 'rbf',
    changepoint_kwargs: Optional[Dict[str, Any]] = None
):
    """
    Build a simple callable that detects changepoints using either ruptures (default PELT)
    or a lightweight gradient-based scorer.
    """
    algo_name = (changepoint_algo or 'pelt').lower()
    base_kwargs = dict(changepoint_kwargs or {})

    if algo_name in {'gradient', 'fast'}:
        smoothing_window = int(max(1, base_kwargs.pop('smoothing_window', 5)))
        min_distance = base_kwargs.pop('min_distance', None)
        min_distance_ratio = float(base_kwargs.pop('min_distance_ratio', 0.05))
        candidate_pool_multiplier = max(1, int(base_kwargs.pop('candidate_pool_multiplier', 5)))
        normalize_differences = bool(base_kwargs.pop('normalize_differences', True))
        robust_scale = bool(base_kwargs.pop('robust_scale', False))

        def _detect(sample_array: np.ndarray, max_bkps: int) -> np.ndarray:
            if max_bkps <= 0:
                return np.asarray([], dtype=int)

            data = np.asarray(sample_array, dtype=np.float64)
            if data.ndim == 1:
                data = data[:, None]

            diffs = np.diff(data, axis=0)
            if diffs.size == 0:
                return np.asarray([], dtype=int)

            if normalize_differences:
                if robust_scale:
                    median = np.median(diffs, axis=0)
                    mad = np.median(np.abs(diffs - median), axis=0)
                    mad[mad < 1e-12] = 1.0
                    diffs = (diffs - median) / mad
                else:
                    std = diffs.std(axis=0, ddof=1)
                    std[std < 1e-12] = 1.0
                    diffs = diffs / std

            scores = np.linalg.norm(diffs, axis=1)

            if smoothing_window > 1 and scores.size >= smoothing_window:
                kernel = np.ones(smoothing_window, dtype=np.float64) / float(smoothing_window)
                scores = np.convolve(scores, kernel, mode='same')

            candidate_pool = max(max_bkps, candidate_pool_multiplier * max_bkps)
            candidate_pool = min(candidate_pool, scores.size)
            if candidate_pool <= 0:
                return np.asarray([], dtype=int)

            candidate_indices = np.argpartition(-scores, candidate_pool - 1)[:candidate_pool]
            ranked_indices = sorted(candidate_indices, key=lambda idx: scores[idx], reverse=True)

            if min_distance is not None:
                min_sep = max(1, int(min_distance))
            else:
                min_sep = max(1, int(round(min_distance_ratio * data.shape[0])))

            selected: List[int] = []
            for idx in ranked_indices:
                cp = idx + 1
                if cp <= 0 or cp >= data.shape[0]:
                    continue
                if all(abs(cp - prev) >= min_sep for prev in selected):
                    selected.append(cp)
                if len(selected) >= max_bkps:
                    break

            if len(selected) < max_bkps:
                for idx in ranked_indices:
                    cp = idx + 1
                    if cp <= 0 or cp >= data.shape[0] or cp in selected:
                        continue
                    selected.append(cp)
                    if len(selected) >= max_bkps:
                        break

            return np.asarray(sorted(set(selected))[:max_bkps], dtype=int)

        return _detect

    import ruptures as rpt

    algo_map = {
        'pelt': rpt.Pelt,
        'binseg': rpt.Binseg,
        'bottomup': rpt.BottomUp,
        'window': rpt.Window,
        'dynp': rpt.Dynp,
    }

    if algo_name not in algo_map:
        return _create_changepoint_detector('gradient', changepoint_model, changepoint_kwargs)

    if changepoint_model is not None and 'model' not in base_kwargs:
        base_kwargs['model'] = changepoint_model
    if algo_name == 'dynp':
        base_kwargs.setdefault('min_size', 2)
        base_kwargs.setdefault('jump', 1)

    AlgoCls = algo_map[algo_name]

    def _detect(sample_array: np.ndarray, max_bkps: int) -> np.ndarray:
        if max_bkps <= 0:
            return np.asarray([], dtype=int)

        data = np.asarray(sample_array, dtype=np.float64)
        if data.ndim == 1:
            data = data[:, None]

        kwargs = dict(base_kwargs)
        if algo_name == 'window':
            kwargs.setdefault('width', max(2, data.shape[0] // 20))

        algo = AlgoCls(**kwargs).fit(data)
        try:
            bkps = algo.predict(n_bkps=max_bkps)
        except TypeError:
            bkps = algo.predict(max_bkps)

        filtered = [int(idx) for idx in bkps if 0 < int(idx) < data.shape[0]]
        return np.asarray(sorted(set(filtered)), dtype=int)

    return _detect


def detect_changepoints(
    latent_sample,
    n_changepoints: int = 5,
    decimation_factor: int = 16,
    raw_signal=None,
    detect_raw: bool = True,
    detector=None,
    changepoint_algo: str = 'pelt',
    changepoint_model: Optional[str] = 'rbf',
    changepoint_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, np.ndarray]:
    """
    Detect changepoints for a single latent trajectory and optional raw signal.

    Args:
        latent_sample: (time_steps, latent_dim) array-like.
        n_changepoints: Maximum number of changepoints to detect (>=0).
        decimation_factor: Ratio between raw length and latent length for mapping indices.
        raw_signal: Optional 1D array-like of the raw FHR signal corresponding to latent_sample.
        detect_raw: Whether to detect changepoints directly on raw_signal (requires raw_signal).
        detector: Optional callable returned by `_create_changepoint_detector`. If None, one will be created.
        changepoint_algo / changepoint_model / changepoint_kwargs: Parameters for detector construction
            when `detector` is None. Defaults use the PELT algorithm via ruptures; pass
            changepoint_algo='gradient' to activate the built-in fast detector.

    Returns:
        Dict with keys:
            - 'latent_changepoints': np.ndarray of changepoint indices in latent steps.
            - 'raw_changepoints': np.ndarray of mapped indices in raw space (if raw_signal provided).
            - 'raw_detected_changepoints': np.ndarray of indices detected directly in raw_signal
              (empty if raw detection disabled or raw_signal not provided).
    """

    def _to_numpy(data):
        if data is None:
            return None
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    latent_np = _to_numpy(latent_sample)
    if latent_np is None:
        latent_np = np.zeros((0, 1), dtype=np.float64)
    latent_np = np.asarray(latent_np, dtype=np.float64)
    if latent_np.ndim == 1:
        latent_np = latent_np[:, None]
    elif latent_np.ndim > 2:
        latent_np = latent_np.reshape(latent_np.shape[0], -1)

    latent_time_steps = latent_np.shape[0]
    max_bkps = min(max(0, int(n_changepoints)), max(0, latent_time_steps - 1))

    detector_fn = detector
    if detector_fn is None:
        detector_fn = _create_changepoint_detector(
            changepoint_algo=changepoint_algo,
            changepoint_model=changepoint_model,
            changepoint_kwargs=changepoint_kwargs
        )

    latent_cps = detector_fn(latent_np, max_bkps)

    raw_cps_from_latent = np.asarray([], dtype=int)
    raw_detected_cps = np.asarray([], dtype=int)

    raw_np = _to_numpy(raw_signal)
    if raw_np is not None:
        raw_np = raw_np.reshape(-1)
        raw_len = raw_np.shape[0]
        if raw_len > 0:
            raw_cps_from_latent = np.asarray(
                [
                    min(raw_len - 1, int(cp_idx * decimation_factor))
                    for cp_idx in latent_cps
                ],
                dtype=int
            )

            if detect_raw:
                per_sample_raw_bkps = min(max_bkps, max(0, raw_len - 1))
                if per_sample_raw_bkps > 0:
                    raw_detected_cps = detector_fn(raw_np.reshape(-1, 1), per_sample_raw_bkps)

    return {
        'latent_changepoints': np.asarray(latent_cps, dtype=int),
        'raw_changepoints': raw_cps_from_latent,
        'raw_detected_changepoints': raw_detected_cps,
    }


def plot_latent_changepoints_with_raw(
    latent_mean,
    fhr,
    up,
    save_path,
    sample_ids=None,
    n_changepoints=5,
    decimation_factor=16,
    cmap='viridis',
    figsize=(14, 4),
    max_samples=5,
    random_state=None,
    show_colorbar=True,
    changepoint_algo='pelt',
    changepoint_model='rbf',
    changepoint_kwargs=None,
    precomputed_changepoints=None
):
    """
    Visualize latent trajectories alongside raw signals with shared changepoints.

    Args:
        latent_mean: torch.Tensor or np.ndarray of shape (batch, time_steps, latent_dim)
            Latent representation mean per batch.
        fhr: torch.Tensor or np.ndarray of shape (batch, time_steps * decimation_factor)
            Fetal heart rate signals corresponding to each latent trajectory.
        up: Ignored (kept for backward compatibility). Raw changepoint comparisons use FHR only.
        save_path: str, directory path where figures will be written.
        sample_ids: Optional sequence of identifiers (len == batch). Defaults to sequential indices.
        n_changepoints: int, number of changepoints to detect in the latent space (>=0).
        decimation_factor: int, ratio between raw signal length and latent length (default: 16).
        cmap: str, matplotlib colormap name for the latent heatmap.
        figsize: tuple, base figure size per sample pair (latent + raw axes).
        max_samples: int, maximum number of samples to visualize (randomly selected, default: 5).
        random_state: Optional int for reproducible sample selection.
        show_colorbar: bool, whether to attach a colorbar next to each latent plot.
        changepoint_algo: str, detector alias applied to latent & raw (default 'pelt' from ruptures;
            pass 'gradient' to reuse the built-in detector).
        changepoint_model: str or None, model parameter forwarded to ruptures algorithms.
        changepoint_kwargs: Optional dict, extra keyword arguments for the selected detector.
        precomputed_changepoints: Optional dict or list/tuple containing per-sample changepoint
            results (as returned by `detect_changepoints`). If provided, detections are reused.

    Returns:
        List[Dict[str, Any]] containing per-sample changepoint metadata for plotted samples with keys:
            - 'sample_id'
            - 'latent_changepoints' (np.ndarray of latent indices)
            - 'raw_changepoints' (np.ndarray of raw indices mapped from latent)
            - 'raw_detected_changepoints' (np.ndarray of raw indices detected from raw signals)

    Notes:
        - Changepoints are detected in latent space and FHR space using the same detector settings.
        - Raw changepoints mapped from latent are derived by multiplying latent indices by decimation_factor.
    """
    def _to_numpy(data):
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    latent_np = _to_numpy(latent_mean)
    fhr_np = _to_numpy(fhr)

    detector_fn = _create_changepoint_detector(
        changepoint_algo=changepoint_algo,
        changepoint_model=changepoint_model,
        changepoint_kwargs=changepoint_kwargs
    )

    if latent_np.ndim == 2:
        latent_np = latent_np[None, ...]
    elif latent_np.ndim > 3:
        latent_np = latent_np.reshape(latent_np.shape[0], latent_np.shape[1], -1)

    if fhr_np.ndim == 1:
        fhr_np = fhr_np[None, ...]

    batch_size = min(latent_np.shape[0], fhr_np.shape[0])
    latent_np = latent_np[:batch_size]
    fhr_np = fhr_np[:batch_size]

    batch_size, latent_time_steps, _ = latent_np.shape

    precomputed_list: Optional[List[Dict[str, Any]]] = None
    precomputed_dict: Optional[Dict[Any, Dict[str, Any]]] = None
    if precomputed_changepoints is not None:
        if isinstance(precomputed_changepoints, dict):
            precomputed_dict = precomputed_changepoints
        elif isinstance(precomputed_changepoints, (list, tuple)):
            precomputed_list = list(precomputed_changepoints)
        else:
            raise TypeError(
                "precomputed_changepoints must be a dict or list/tuple of per-sample results."
            )
    if sample_ids is None:
        sample_ids = [f"sample_{idx}" for idx in range(batch_size)]
    if len(sample_ids) != batch_size:
        raise ValueError("sample_ids length must match batch size.")

    if max_samples is not None:
        if max_samples <= 0:
            logger.warning("max_samples <= 0; no plots generated.")
            return []
        max_samples = min(batch_size, int(max_samples))
    else:
        max_samples = batch_size

    rng = np.random.default_rng(random_state)
    if max_samples < batch_size:
        selected_indices = rng.choice(batch_size, size=max_samples, replace=False)
    else:
        selected_indices = np.arange(batch_size)
    selected_indices = np.sort(selected_indices)

    if selected_indices.size == 0:
        logger.warning("No samples selected for plotting.")
        return []

    os.makedirs(save_path, exist_ok=True)

    results: List[Dict[str, Any]] = []

    max_bkps = max(0, int(n_changepoints))

    def _fetch_precomputed(idx: int, sample_label: str) -> Optional[Dict[str, Any]]:
        entry = None
        if precomputed_dict is not None:
            entry = precomputed_dict.get(sample_label)
            if entry is None:
                entry = precomputed_dict.get(idx)
        if entry is None and precomputed_list is not None and idx < len(precomputed_list):
            entry = precomputed_list[idx]
        return entry

    total_rows = selected_indices.size * 3
    fig_height = max(figsize[1] * selected_indices.size * 1.5, 3.5)
    n_cols = 2 if show_colorbar else 1
    width_ratios = [25, 1] if show_colorbar else None
    fig = plt.figure(figsize=(figsize[0], fig_height))
    gs = GridSpec(
        total_rows,
        n_cols,
        figure=fig,
        height_ratios=[1.0] * total_rows,
        width_ratios=width_ratios,
    )

    for idx_counter, batch_idx in enumerate(selected_indices):
        sample_id = str(sample_ids[batch_idx])
        latent_sample = latent_np[batch_idx]
        fhr_sample = fhr_np[batch_idx]

        cp_entry = _fetch_precomputed(batch_idx, sample_id)
        if cp_entry is None:
            cp_entry = detect_changepoints(
                latent_sample=latent_sample,
                n_changepoints=max_bkps,
                decimation_factor=decimation_factor,
                raw_signal=fhr_sample,
                detect_raw=True,
                detector=detector_fn
            )

        latent_cps = np.asarray(cp_entry.get('latent_changepoints', []), dtype=int)

        raw_expected_len = latent_time_steps * decimation_factor
        raw_len = fhr_sample.shape[0]
        if raw_len != raw_expected_len:
            logger.warning(
                f"[{sample_id}] Raw signal length ({raw_len}) does not match "
                f"latent_time_steps * decimation_factor ({raw_expected_len})."
            )

        raw_cps_from_latent = np.asarray(
            cp_entry.get('raw_changepoints', []),
            dtype=int
        )
        raw_detected_cps = np.asarray(
            cp_entry.get('raw_detected_changepoints', []),
            dtype=int
        )

        if raw_cps_from_latent.size == 0 and raw_len > 0 and latent_cps.size > 0:
            raw_cps_from_latent = np.asarray(
                [
                    min(raw_len - 1, int(cp_idx * decimation_factor))
                    for cp_idx in latent_cps
                ],
                dtype=int
            )

        row_latent = 3 * idx_counter
        row_raw_latent = row_latent + 1
        row_raw_detected = row_latent + 2

        if show_colorbar:
            latent_ax = fig.add_subplot(gs[row_latent, 0])
            raw_latent_ax = fig.add_subplot(gs[row_raw_latent, 0], sharex=latent_ax)
            raw_detected_ax = fig.add_subplot(gs[row_raw_detected, 0], sharex=latent_ax)
            cax = fig.add_subplot(gs[row_latent:row_raw_detected+1, 1])
        else:
            latent_ax = fig.add_subplot(gs[row_latent, 0])
            cax = None
            raw_latent_ax = fig.add_subplot(gs[row_raw_latent, 0], sharex=latent_ax)
            raw_detected_ax = fig.add_subplot(gs[row_raw_detected, 0], sharex=latent_ax)

        latent_dims = latent_sample.shape[1]
        x_max = max(raw_len - 1, 0)
        extent = (0, x_max, -0.5, latent_dims - 0.5)
        im = latent_ax.imshow(
            latent_sample.T,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            interpolation='nearest',
            extent=extent
        )
        latent_ax.set_xlim(extent[0], extent[1])
        latent_ax.set_ylim(extent[2], extent[3])
        latent_ax.set_ylabel('Latent Dimension')
        latent_ax.set_title(f'Latent Representation (mean) - {sample_id}')
        latent_ax.grid(False)
        if idx_counter != selected_indices.size - 1:
            latent_ax.tick_params(labelbottom=False)
        else:
            latent_ax.set_xlabel('Raw Time Index')

        if show_colorbar and cax is not None:
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.set_ylabel('Activation', rotation=270, labelpad=12)

        for raw_cp in raw_cps_from_latent:
            latent_ax.axvline(
                raw_cp,
                color='white',
                linestyle='--',
                linewidth=1.5,
                alpha=0.8
            )

        time_axis = np.arange(raw_len)
        raw_latent_ax.plot(time_axis, fhr_sample, label='FHR', color='#E74C3C', linewidth=1.2)
        raw_latent_ax.set_ylabel('Normalized Amplitude')
        raw_latent_ax.set_title(f'FHR w/ Latent CPs - {sample_id}')
        raw_latent_ax.grid(True, alpha=0.3)
        raw_latent_ax.legend(loc='upper right')
        raw_latent_ax.set_xlim(extent[0], extent[1])
        raw_latent_ax.tick_params(labelbottom=False)

        for raw_cp in raw_cps_from_latent:
            raw_latent_ax.axvline(
                raw_cp,
                color='black',
                linestyle='--',
                linewidth=1.0,
                alpha=0.6
            )

        raw_detected_ax.plot(time_axis, fhr_sample, label='FHR', color='#E74C3C', linewidth=1.2)
        raw_detected_ax.set_ylabel('Normalized Amplitude')
        raw_detected_ax.set_title(f'FHR w/ Raw CPs - {sample_id}')
        raw_detected_ax.grid(True, alpha=0.3)
        raw_detected_ax.set_xlim(extent[0], extent[1])
        if idx_counter != selected_indices.size - 1:
            raw_detected_ax.tick_params(labelbottom=False)
        else:
            raw_detected_ax.set_xlabel('Raw Time Index')

        latent_line_added = False
        raw_line_added = False
        for raw_cp in raw_cps_from_latent:
            raw_detected_ax.axvline(
                raw_cp,
                color='#7F8C8D',
                linestyle='--',
                linewidth=1.0,
                alpha=0.6,
                label='Latent CPs' if not latent_line_added else None
            )
            latent_line_added = True
        for raw_cp in raw_detected_cps:
            raw_detected_ax.axvline(
                raw_cp,
                color='#27AE60',
                linestyle='-',
                linewidth=1.2,
                alpha=0.75,
                label='Raw CPs' if not raw_line_added else None
            )
            raw_line_added = True

        if latent_line_added or raw_line_added:
            raw_detected_ax.legend(loc='upper right')

        results.append(
            {
                'sample_id': sample_id,
                'latent_changepoints': latent_cps,
                'raw_changepoints': raw_cps_from_latent,
                'raw_detected_changepoints': raw_detected_cps
            }
        )

    fig.suptitle('Latent vs Raw Changepoints (sampled)', fontsize=14, y=0.995)
    fig.subplots_adjust(
        top=0.92,
        bottom=0.07,
        left=0.08,
        right=0.98,
        hspace=0.3,
        wspace=0.12 if show_colorbar else 0.05
    )

    selected_labels = "_".join([str(sample_ids[idx]) for idx in selected_indices])
    selected_labels = selected_labels.replace(os.sep, "-")
    if len(selected_labels) > 80:
        selected_labels = selected_labels[:77] + "..."

    figure_path = os.path.join(
        save_path,
        f'latent_raw_changepoints_samples_{selected_labels}.png'
    )
    fig.savefig(figure_path, dpi=180, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    return results


def summarize_latent_segments(
    latent_mean,
    epoch=None,
    sample_ids: Optional[Sequence[Any]] = None,
    n_changepoints: int = 5,
    decimation_factor: int = 16,
    raw_sample_rate_hz: float = 4.0,
    changepoint_algo: str = 'pelt',
    changepoint_model: Optional[str] = 'rbf',
    changepoint_kwargs: Optional[Dict[str, Any]] = None,
    precomputed_changepoints: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Compute per-segment latent statistics after changepoint detection.

    Args:
        latent_mean: torch.Tensor or np.ndarray of shape (batch, time_steps, latent_dim).
        epoch: Optional torch.Tensor or np.ndarray of shape (batch,) giving raw-sample start
            indices relative to delivery (see dataset docs). Used to express segment timing.
        sample_ids: Optional identifiers per sample (len == batch). Defaults to sequential indices.
        n_changepoints: Maximum number of changepoints to detect per sample (>=0).
        decimation_factor: Ratio between raw signal length and latent length (default: 16).
        raw_sample_rate_hz: Raw sampling rate in Hz (default: 4.0).
        changepoint_algo: Detector alias (default 'pelt' via ruptures; pass 'gradient' for the fast builtin).
        changepoint_model: Optional model argument for ruptures detectors (ignored for 'gradient').
        changepoint_kwargs: Extra kwargs forwarded to the selected detector.
        precomputed_changepoints: Optional dict/list containing per-sample changepoint dictionaries
            (as returned by `detect_changepoints`). When supplied, latent changepoints are reused.

    Returns:
        List of dictionaries, one per sample, each containing:
            - 'sample_id'
            - 'epoch_raw_index'
            - 'latent_changepoints'
            - 'segments': list of segment-level statistics dictionaries with keys:
                * 'segment_index'
                * 'start_step', 'end_step', 'length_steps'
                * 'duration_seconds'
                * 'start_minutes_rel_delivery', 'end_minutes_rel_delivery' (if epoch provided)
                * 'mean_vector', 'variance_vector'
                * 'dominant_latent_dim'
                * 'mean_velocity', 'mean_speed', 'direction_unit_vector'
                * 'mean_activation_norm'
    """

    def _to_numpy(data):
        if data is None:
            return None
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    latent_np = _to_numpy(latent_mean)
    epoch_np = _to_numpy(epoch)

    if latent_np.ndim != 3:
        raise ValueError(
            f"latent_mean must have shape (batch, time_steps, latent_dim); got {latent_np.shape}"
        )

    batch_size, latent_time_steps, latent_dim = latent_np.shape

    if sample_ids is None:
        sample_ids = [f"sample_{idx}" for idx in range(batch_size)]
    if len(sample_ids) != batch_size:
        raise ValueError("sample_ids length must match batch size.")

    if epoch_np is not None:
        epoch_np = epoch_np.astype(np.float64)
        if epoch_np.ndim != 1 or epoch_np.shape[0] != batch_size:
            raise ValueError("epoch must be 1D with length equal to batch size.")

    precomputed_list: Optional[List[Dict[str, Any]]] = None
    precomputed_dict: Optional[Dict[Any, Dict[str, Any]]] = None
    if precomputed_changepoints is not None:
        if isinstance(precomputed_changepoints, dict):
            precomputed_dict = precomputed_changepoints
        elif isinstance(precomputed_changepoints, (list, tuple)):
            precomputed_list = list(precomputed_changepoints)
        else:
            raise TypeError(
                "precomputed_changepoints must be a dict or list/tuple of per-sample results."
            )

    detector_fn = None

    seconds_per_step = None
    minutes_scale = None
    if raw_sample_rate_hz and decimation_factor:
        seconds_per_step = decimation_factor / float(raw_sample_rate_hz)
        minutes_scale = raw_sample_rate_hz * 60.0

    max_bkps = max(0, int(n_changepoints))
    results: List[Dict[str, Any]] = []

    def _fetch_precomputed(idx: int, sample_label: str) -> Optional[Dict[str, Any]]:
        entry = None
        if precomputed_dict is not None:
            entry = precomputed_dict.get(sample_label)
            if entry is None:
                entry = precomputed_dict.get(idx)
        if entry is None and precomputed_list is not None and idx < len(precomputed_list):
            entry = precomputed_list[idx]
        return entry

    for batch_idx in range(batch_size):
        sample_id = str(sample_ids[batch_idx])
        latent_sample = latent_np[batch_idx]
        cp_entry = _fetch_precomputed(batch_idx, sample_id)
        if cp_entry is None:
            if detector_fn is None:
                detector_fn = _create_changepoint_detector(
                    changepoint_algo=changepoint_algo,
                    changepoint_model=changepoint_model,
                    changepoint_kwargs=changepoint_kwargs
                )
            cp_entry = detect_changepoints(
                latent_sample=latent_sample,
                n_changepoints=max_bkps,
                decimation_factor=decimation_factor,
                raw_signal=None,
                detect_raw=False,
                detector=detector_fn
            )

        latent_cps = np.asarray(cp_entry.get('latent_changepoints', []), dtype=int)

        segment_boundaries = np.concatenate(([0], latent_cps, [latent_time_steps]))
        epoch_value = None if epoch_np is None else float(epoch_np[batch_idx])
        segments: List[Dict[str, Any]] = []

        for seg_idx in range(len(segment_boundaries) - 1):
            start_step = int(segment_boundaries[seg_idx])
            end_step = int(segment_boundaries[seg_idx + 1])
            length_steps = end_step - start_step
            if length_steps <= 0:
                continue

            segment_values = latent_sample[start_step:end_step]
            segment_mean = segment_values.mean(axis=0)
            if length_steps > 1:
                segment_var = segment_values.var(axis=0, ddof=1)
                velocity = np.diff(segment_values, axis=0)
                mean_velocity = velocity.mean(axis=0)
            else:
                segment_var = np.zeros(latent_dim, dtype=np.float64)
                mean_velocity = np.zeros(latent_dim, dtype=np.float64)

            mean_speed = float(np.linalg.norm(mean_velocity))
            if mean_speed > 0:
                direction_unit_vector = mean_velocity / mean_speed
            else:
                direction_unit_vector = np.zeros_like(mean_velocity)

            dominant_dim = int(np.argmax(segment_var)) if segment_var.size else None
            mean_activation_norm = float(np.mean(np.linalg.norm(segment_values, axis=1)))

            duration_seconds = None
            if seconds_per_step is not None:
                duration_seconds = float(length_steps * seconds_per_step)

            start_minutes = end_minutes = None
            if epoch_value is not None and minutes_scale:
                raw_start = epoch_value + start_step * decimation_factor
                raw_end = epoch_value + end_step * decimation_factor
                start_minutes = float(raw_start / minutes_scale)
                end_minutes = float(raw_end / minutes_scale)

            segments.append(
                {
                    'segment_index': seg_idx,
                    'start_step': start_step,
                    'end_step': end_step,
                    'length_steps': length_steps,
                    'duration_seconds': duration_seconds,
                    'start_minutes_rel_delivery': start_minutes,
                    'end_minutes_rel_delivery': end_minutes,
                    'mean_vector': segment_mean,
                    'variance_vector': segment_var,
                    'dominant_latent_dim': dominant_dim,
                    'mean_velocity': mean_velocity,
                    'mean_speed': mean_speed,
                    'direction_unit_vector': direction_unit_vector,
                    'mean_activation_norm': mean_activation_norm,
                }
            )

        results.append(
            {
                'sample_id': sample_id,
                'epoch_raw_index': epoch_value,
                'latent_changepoints': latent_cps,
                'segments': segments,
            }
        )

    return results


def plot_segment_statistics(
    segment_stats: Sequence[Dict[str, Any]],
    save_path: str,
    csv_filename: str = "segment_stats.csv"
) -> Optional[pd.DataFrame]:
    """
    Visualize aggregated latent segment statistics.

    Args:
        segment_stats: Iterable of per-sample dictionaries produced by summarize_latent_segments.
        save_path: Directory where figures (and optional CSV) will be written.
        csv_filename: Name of the CSV file to export flattened segment statistics.

    Returns:
        pandas.DataFrame containing flattened segment statistics (or None if no data).
    """
    if not segment_stats:
        return None

    rows: List[Dict[str, Any]] = []
    for entry in segment_stats:
        sample_id = entry.get('sample_id')
        epoch_raw_index = entry.get('epoch_raw_index')
        for seg in entry.get('segments', []):
            row = {
                'sample_id': sample_id,
                'epoch_raw_index': epoch_raw_index,
                'segment_index': seg.get('segment_index'),
                'start_step': seg.get('start_step'),
                'end_step': seg.get('end_step'),
                'length_steps': seg.get('length_steps'),
                'duration_seconds': seg.get('duration_seconds'),
                'start_minutes_rel_delivery': seg.get('start_minutes_rel_delivery'),
                'end_minutes_rel_delivery': seg.get('end_minutes_rel_delivery'),
                'dominant_latent_dim': seg.get('dominant_latent_dim'),
                'mean_speed': seg.get('mean_speed'),
                'mean_activation_norm': seg.get('mean_activation_norm'),
            }
            rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)

    os.makedirs(save_path, exist_ok=True)
    csv_path = os.path.join(save_path, csv_filename)
    df.to_csv(csv_path, index=False)

    # Histogram of segment durations (minutes)
    durations = df['duration_seconds'].dropna()
    if not durations.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        durations_minutes = durations / 60.0
        ax.hist(durations_minutes, bins=20, color='#2E86C1', alpha=0.75, edgecolor='black')
        ax.set_title('Segment Duration Distribution')
        ax.set_xlabel('Duration (minutes)')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, 'segment_duration_hist.png'), dpi=160)
        plt.close(fig)

    # Scatter: start time relative to delivery vs mean speed
    start_speed = df[['start_minutes_rel_delivery', 'mean_speed']].dropna()
    if not start_speed.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            start_speed['start_minutes_rel_delivery'],
            start_speed['mean_speed'],
            s=25,
            c='#E74C3C',
            alpha=0.7,
            edgecolors='black',
            linewidths=0.4
        )
        ax.set_title('Mean Latent Speed vs Start Time')
        ax.set_xlabel('Start Minutes Relative to Delivery')
        ax.set_ylabel('Mean Latent Speed (L2 norm)')
        ax.axvline(0.0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, 'mean_speed_vs_start.png'), dpi=160)
        plt.close(fig)

    # Bar chart: dominant latent dimension counts
    dominant_dims = df['dominant_latent_dim'].dropna()
    if not dominant_dims.empty:
        dominant_counts = (
            dominant_dims.astype(int)
            .value_counts()
            .sort_index()
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(dominant_counts.index.astype(str), dominant_counts.values, color='#27AE60', alpha=0.8)
        ax.set_title('Dominant Latent Dimension Frequency')
        ax.set_xlabel('Latent Dimension Index')
        ax.set_ylabel('Segment Count')
        ax.grid(True, axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, 'dominant_latent_dim_counts.png'), dpi=160)
        plt.close(fig)

    # Bar chart: segments per sample
    segments_per_sample = df.groupby('sample_id')['segment_index'].count()
    if not segments_per_sample.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        segments_per_sample.sort_values(ascending=False).plot(
            kind='bar',
            ax=ax,
            color='#9B59B6',
            alpha=0.8
        )
        ax.set_title('Segments per Sample')
        ax.set_xlabel('Sample ID')
        ax.set_ylabel('Number of Segments')
        ax.tick_params(axis='x', rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, 'segments_per_sample.png'), dpi=160)
        plt.close(fig)

    return df


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
        - Changepoint detection uses the configured detector (default PELT via ruptures)
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

    detector_fn = _create_changepoint_detector() if detect_changepoints else None

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
        if detect_changepoints and detector_fn is not None:
            fhr_signal = fhr[i].reshape(-1, 1)  # Shape (time_steps, 1)

            # Determine a reasonable number of breakpoints while guarding short segments.
            max_possible_segments = max(1, time_steps // 10)  # At least 10 points per segment
            requested_segments = max(1, int(n_changepoints))
            actual_segments = min(requested_segments, max_possible_segments)
            per_segment_bkps = max(0, actual_segments - 1)

            if per_segment_bkps <= 0:
                continue

            try:
                changepoint_indices = detector_fn(fhr_signal, per_segment_bkps)
            except Exception as exc:
                if i == 0:
                    logger.warning(f"Changepoint detection failed for epoch {i}: {exc}")
                continue

            changepoint_indices = np.asarray(changepoint_indices, dtype=int)
            if changepoint_indices.size == 0:
                continue

            # Convert changepoint indices to time in minutes
            changepoint_times = segment_start_minutes + changepoint_indices * time_per_step / 60.0

            # Add vertical lines at changepoints on FHR plot
            for cp_idx, cp_time in enumerate(changepoint_times):
                fig.add_vline(
                    x=cp_time,
                    line=dict(color='green', width=2, dash='dash'),
                    opacity=0.6,
                    row=1, col=1,
                    annotation=dict(
                        text='CP',
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

        changepoint_cfg = config_latent or {}
        n_changepoints = int(changepoint_cfg.get('n_changepoints', 5))
        default_decimation = int(changepoint_cfg.get('decimation_factor', 16))
        raw_sample_rate_hz = float(changepoint_cfg.get('raw_sample_rate_hz', 4.0))
        changepoint_algo = changepoint_cfg.get('changepoint_algo', 'dynp')
        changepoint_model = changepoint_cfg.get('changepoint_model', 'rbf')
        changepoint_kwargs = changepoint_cfg.get('changepoint_kwargs', {}) or {}
        if not isinstance(changepoint_kwargs, dict):
            changepoint_kwargs = {}

        changepoint_detector = _create_changepoint_detector(
            changepoint_algo=changepoint_algo,
            changepoint_model=changepoint_model,
            changepoint_kwargs=copy.deepcopy(changepoint_kwargs)
        )

        all_changepoint_results: List[Dict[str, Any]] = []
        all_segment_stats: List[Dict[str, Any]] = []

        with torch.inference_mode():
            for batch in selected_batches:
                fhr_st = batch.fhr_st.to(self.device)
                fhr_ph = batch.fhr_ph.to(self.device)
                fhr_up_ph = batch.fhr_up_ph.to(self.device)
                fhr = batch.fhr.to(self.device)
                up = batch.up.to(self.device)
                epoch = batch.epoch  # Shape (Batch_size,)
                guid_in_batch = str(batch['guid'][0])

                # Sort all batch data by epoch values
                sort_indices = torch.argsort(epoch)
                fhr_st = fhr_st[sort_indices]
                fhr_ph = fhr_ph[sort_indices]
                fhr_up_ph = fhr_up_ph[sort_indices]
                fhr = fhr[sort_indices]  # shape: (Batch, time_step*16)
                up = up[sort_indices]  # shape: (Batch, time_step*16)
                epoch = epoch[sort_indices]
                epoch_cpu = epoch.detach().cpu()

                sample_ids = [f"{guid_in_batch}_{idx:02d}" for idx in range(len(epoch))]

                fhr_up_plot_path = os.path.join(save_dir, "fhr-up-signals")
                # plot_complete_fhr_up_timeline(
                #     fhr=fhr,
                #     up=up,
                #     epoch=epoch,
                #     save_path=fhr_up_plot_path,
                #     sample_id=f"{guid_in_batch}",
                #     title='Complete FHR and UP Timeline',
                #     sampling_rate_hz=4,
                #     detect_changepoints=True
                # )
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
                
                reduced_latent_mean = reduce_latent_dimensionality(
                    proccessed_latent_mean,
                    method=laten_dim_reduction_type,
                    n_components=3,
                    n_neighbors=15,
                    min_dist=0.1,
                    return_reducer=False
                )

                effective_decimation = max(
                    1,
                    int(fhr.shape[1] // max(1, reduced_latent_mean.shape[1]))
                )
                if effective_decimation <= 0:
                    effective_decimation = default_decimation

                changepoint_results: List[Dict[str, Any]] = []
                for sample_idx in range(reduced_latent_mean.shape[0]):
                    cp_info = detect_changepoints(
                        latent_sample=reduced_latent_mean[sample_idx],
                        n_changepoints=n_changepoints,
                        decimation_factor=effective_decimation,
                        raw_signal=fhr[sample_idx],
                        detect_raw=True,
                        detector=changepoint_detector
                    )
                    cp_info['sample_id'] = sample_ids[sample_idx]
                    changepoint_results.append(cp_info)
                all_changepoint_results.extend(changepoint_results)

                plot_latent_changepoints_with_raw(
                    fhr=fhr,
                    up=up,
                    latent_mean=reduced_latent_mean,
                    save_path=fhr_up_plot_path,
                    sample_ids=sample_ids,
                    n_changepoints=n_changepoints,
                    decimation_factor=effective_decimation,
                    changepoint_algo=changepoint_algo,
                    changepoint_model=changepoint_model,
                    changepoint_kwargs=changepoint_kwargs,
                    precomputed_changepoints=changepoint_results
                )

                segment_stats = summarize_latent_segments(
                    latent_mean=reduced_latent_mean,
                    epoch=epoch_cpu,
                    sample_ids=sample_ids,
                    n_changepoints=n_changepoints,
                    decimation_factor=effective_decimation,
                    raw_sample_rate_hz=raw_sample_rate_hz,
                    changepoint_algo=changepoint_algo,
                    changepoint_model=changepoint_model,
                    changepoint_kwargs=changepoint_kwargs,
                    precomputed_changepoints=changepoint_results
                )
                all_segment_stats.extend(segment_stats)
                
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

        segment_stats_dir = os.path.join(save_dir, "segment_statistics")
        segment_stats_table = plot_segment_statistics(all_segment_stats, segment_stats_dir)

        return {
            'changepoints': all_changepoint_results,
            'segment_stats': all_segment_stats,
            'segment_stats_table': segment_stats_table,
            'segment_stats_path': segment_stats_dir,
        }

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


"""
For vs_ProPlan and vs_ProPlan_% we should compare with respect to Calculated ProPlan so 
it should be the in period minus proplan and (in period  / proplan - 1)*100
Revise 
"""