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

def plot_latent_trajectory(
    latent_trajectory,
    save_path,
    sample_id='sample_0',
    title='Latent Trajectory',
    color_by_time=True,
    point_size=20,
    arrow_scale=0.3,
    plot_animation=True
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

    Saves:
        - Static plot: {save_path}/trajectory_{sample_id}.png
        - Animated GIF (if plot_animation=True): {save_path}/trajectory_{sample_id}_animated.gif
        - 3D HTML (if 3D): {save_path}/trajectory_{sample_id}_3d.html
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
                epoch = batch.epoch  # Shape (Batch_size,)
                guid_in_batch = batch['guid'][0]

                # Sort all batch data by epoch values
                sort_indices = torch.argsort(epoch)
                fhr_st = fhr_st[sort_indices]
                fhr_ph = fhr_ph[sort_indices]
                fhr_up_ph = fhr_up_ph[sort_indices]
                fhr = fhr[sort_indices]
                epoch = epoch[sort_indices]

                self.pytorch_model.to(self.device)
                model_output = self.pytorch_model(
                    y_st=fhr_st,
                    y_ph=fhr_ph,
                    x_ph=fhr_up_ph,
                )
                latent_mean = model_output['mu_post']  # Shape (Batch_size, time_steps, latent_dim)
                latent_var = torch.exp(model_output['logvar_post'])  # (Batch_size, time_steps, latent_dim)
                proccessed_latent_mean = preprocess_latent(latent=latent_mean, denoise=False)  # Shape (Batch_size, time_steps, latent_dim)
                proccessed_latent_var = preprocess_latent(latent=latent_var, denoise=False)  # (Batch_size, time_steps, latent_dim)
                
                reduced_latent_mean = reduce_latent_dimensionality(
                    proccessed_latent_mean,
                    method=laten_dim_reduction_type,
                    n_components=3,
                    n_neighbors=15,
                    min_dist=0.1,
                    return_reducer=False
                )

                epoch_laten_trajectory_path = os.path.join(save_dir, "per_epoch_trajectory")
                plot_latent_trajectory(
                    reduced_latent_mean[0],
                    epoch_laten_trajectory_path,
                    sample_id=f"_{guid_in_batch}-{epoch[0]}-start",
                    title='Latent Trajectory',
                    color_by_time=True,
                    point_size=100,
                    arrow_scale=0.3
                )
                
                plot_latent_trajectory(
                    reduced_latent_mean[-1],
                    epoch_laten_trajectory_path,
                    sample_id=f"_{guid_in_batch}-{epoch[1]}-end",
                    title='Latent Trajectory',
                    color_by_time=True,
                    point_size=100,
                    arrow_scale=0.3
                )

                
                
                batch_size, reduced_time_steps, reduced_latent_dim = reduced_latent_mean.shape
                all_reduced_latent_mean = reduced_latent_mean.reshape(
                    batch_size * reduced_time_steps, reduced_latent_dim
                )
                all_laten_trajectory_path = os.path.join(save_dir, "all_epochs_trajectory_complete")
                plot_latent_trajectory(
                    all_reduced_latent_mean,
                    all_laten_trajectory_path,
                    sample_id=f"_{guid_in_batch}",
                    title='Latent Trajectory',
                    color_by_time=True,
                    point_size=100,
                    arrow_scale=0.3,
                    plot_animation=False
                )                
                
                reduced_latent_mean_summary = summarize_trajectory(
                    reduced_latent_mean,
                    k=10,
                    method=time_dim_reduction_type,
                    epsilon=None,
                    return_indices=False
                )   # shape (Bathc_size, reduces_time_steps, reduced_latent_dim)


                epoch_laten_trajectory_path = os.path.join(save_dir, "per_epoch_trajectory_summary")
                plot_latent_trajectory(
                    reduced_latent_mean_summary[0],
                    epoch_laten_trajectory_path,
                    sample_id=f"_{guid_in_batch}-{epoch[0]}-start",
                    title='Latent Trajectory',
                    color_by_time=True,
                    point_size=100,
                    arrow_scale=0.3
                )
                
                plot_latent_trajectory(
                    reduced_latent_mean_summary[-1],
                    epoch_laten_trajectory_path,
                    sample_id=f"_{guid_in_batch}-{epoch[-1]}-end",
                    title='Latent Trajectory',
                    color_by_time=True,
                    point_size=100,
                    arrow_scale=0.3
                )

                # Reshape to (Batch_size * reduced_time_steps, reduced_latent_dim)
                # keep epoch-sorted order from the earlier sorting
                batch_size, reduced_time_steps, reduced_latent_dim = reduced_latent_mean_summary.shape
                all_reduced_latent_mean_summary = reduced_latent_mean_summary.reshape(
                    batch_size * reduced_time_steps, reduced_latent_dim
                )
                all_laten_trajectory_path = os.path.join(save_dir, "all_epochs_trajectory_summary")
                plot_latent_trajectory(
                    all_reduced_latent_mean_summary,
                    all_laten_trajectory_path,
                    sample_id=f"_{guid_in_batch}",
                    title='Latent Trajectory',
                    color_by_time=True,
                    point_size=100,
                    arrow_scale=0.3
                )

                print('done')


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
                                            laten_dim_reduction_type='pca', time_dim_reduction_type='changepoint')

    
    
if __name__ == '__main__':
    main()

