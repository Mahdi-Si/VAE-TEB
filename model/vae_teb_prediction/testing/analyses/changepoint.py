"""
Changepoint detection and trajectory summarization for VAE-TEB testing.

This module provides functions for detecting changepoints in latent trajectories,
summarizing trajectories into keyframes, and computing per-segment statistics.

Changepoint detection is useful for identifying regime changes in the fetal
monitoring data, which may correspond to clinically significant events.

Example:
    >>> from testing.analyses.changepoint import detect_changepoints, summarize_trajectory
    >>> cp_results = detect_changepoints(latent_sample, n_changepoints=5)
    >>> keyframes = summarize_trajectory(latent_trajectory, k=5, method='changepoint')
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from loguru import logger


def create_changepoint_detector(
    algo: str = "pelt",
    model: Optional[str] = "rbf",
    **kwargs,
) -> Callable[[np.ndarray, int], np.ndarray]:
    """
    Create a changepoint detector function using ruptures or a fast gradient-based fallback.

    Args:
        algo: Detection algorithm. Options:
            - 'pelt': Pruned Exact Linear Time (fast, exact) - requires ruptures
            - 'binseg': Binary Segmentation
            - 'bottomup': Bottom-Up Segmentation
            - 'window': Window-Based Segmentation
            - 'dynp': Dynamic Programming
            - 'gradient' or 'fast': Fast gradient-based detector (no dependencies)
        model: Cost model for ruptures algorithms (e.g., 'rbf', 'l2', 'linear').
            Ignored for 'gradient' method.
        **kwargs: Additional arguments passed to the detector.
            For gradient method:
                - smoothing_window (int): Window for smoothing scores (default 5)
                - min_distance_ratio (float): Minimum distance between changepoints
                    as fraction of signal length (default 0.05)
                - candidate_pool_multiplier (int): Candidates to consider (default 5)

    Returns:
        Callable that takes (signal_array, max_breakpoints) and returns changepoint indices.

    Example:
        >>> detector = create_changepoint_detector(algo='pelt', model='rbf')
        >>> changepoints = detector(latent_sample, max_breakpoints=5)
    """
    algo_name = (algo or "pelt").lower()

    # Fast gradient-based detector (no external dependencies)
    if algo_name in {"gradient", "fast"}:
        smoothing_window = int(max(1, kwargs.pop("smoothing_window", 5)))
        min_distance_ratio = float(kwargs.pop("min_distance_ratio", 0.05))
        candidate_pool_multiplier = max(1, int(kwargs.pop("candidate_pool_multiplier", 5)))
        normalize_differences = bool(kwargs.pop("normalize_differences", True))
        robust_scale = bool(kwargs.pop("robust_scale", False))

        def _detect_gradient(sample_array: np.ndarray, max_bkps: int) -> np.ndarray:
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

            # Compute change magnitude scores
            scores = np.linalg.norm(diffs, axis=1)

            # Optional smoothing
            if smoothing_window > 1 and scores.size >= smoothing_window:
                kernel = np.ones(smoothing_window, dtype=np.float64) / float(smoothing_window)
                scores = np.convolve(scores, kernel, mode="same")

            # Get candidate pool
            candidate_pool = max(max_bkps, candidate_pool_multiplier * max_bkps)
            candidate_pool = min(candidate_pool, scores.size)
            if candidate_pool <= 0:
                return np.asarray([], dtype=int)

            candidate_indices = np.argpartition(-scores, candidate_pool - 1)[:candidate_pool]
            ranked_indices = sorted(candidate_indices, key=lambda idx: scores[idx], reverse=True)

            # Minimum separation between changepoints
            min_sep = max(1, int(round(min_distance_ratio * data.shape[0])))

            # Greedily select changepoints with minimum separation
            selected: List[int] = []
            for idx in ranked_indices:
                cp = idx + 1  # Changepoint is after the diff
                if cp <= 0 or cp >= data.shape[0]:
                    continue
                if all(abs(cp - prev) >= min_sep for prev in selected):
                    selected.append(cp)
                if len(selected) >= max_bkps:
                    break

            return np.asarray(sorted(set(selected))[:max_bkps], dtype=int)

        return _detect_gradient

    # Try ruptures-based detection
    try:
        import ruptures as rpt
    except ImportError:
        logger.warning(
            f"ruptures not installed, falling back to gradient-based detector. "
            f"Install with: pip install ruptures"
        )
        return create_changepoint_detector("gradient", model, **kwargs)

    algo_map = {
        "pelt": rpt.Pelt,
        "binseg": rpt.Binseg,
        "bottomup": rpt.BottomUp,
        "window": rpt.Window,
        "dynp": rpt.Dynp,
    }

    if algo_name not in algo_map:
        logger.warning(f"Unknown algorithm '{algo_name}', falling back to gradient detector")
        return create_changepoint_detector("gradient", model, **kwargs)

    base_kwargs = dict(kwargs)
    if model is not None and "model" not in base_kwargs:
        base_kwargs["model"] = model
    if algo_name == "dynp":
        base_kwargs.setdefault("min_size", 2)
        base_kwargs.setdefault("jump", 1)

    AlgoCls = algo_map[algo_name]

    def _detect_ruptures(sample_array: np.ndarray, max_bkps: int) -> np.ndarray:
        if max_bkps <= 0:
            return np.asarray([], dtype=int)

        data = np.asarray(sample_array, dtype=np.float64)
        if data.ndim == 1:
            data = data[:, None]

        local_kwargs = dict(base_kwargs)
        if algo_name == "window":
            local_kwargs.setdefault("width", max(2, data.shape[0] // 20))

        algo = AlgoCls(**local_kwargs).fit(data)
        try:
            bkps = algo.predict(n_bkps=max_bkps)
        except TypeError:
            bkps = algo.predict(max_bkps)

        # Filter to valid indices (exclude endpoint)
        filtered = [int(idx) for idx in bkps if 0 < int(idx) < data.shape[0]]
        return np.asarray(sorted(set(filtered)), dtype=int)

    return _detect_ruptures


def detect_changepoints(
    latent_sample: np.ndarray,
    n_changepoints: int = 5,
    decimation_factor: int = 16,
    raw_signal: Optional[np.ndarray] = None,
    detect_raw: bool = True,
    detector: Optional[Callable] = None,
    algo: str = "pelt",
    model: Optional[str] = "rbf",
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Detect changepoints in a latent trajectory and optionally in the raw signal.

    Args:
        latent_sample: Latent trajectory array of shape (T, D) where T is time steps
            and D is latent dimension.
        n_changepoints: Maximum number of changepoints to detect (default 5).
        decimation_factor: Ratio between raw signal length and latent length (default 16).
            Used to map latent changepoints to raw signal indices.
        raw_signal: Optional 1D raw signal (e.g., FHR). If provided, changepoints
            will also be mapped to raw signal indices.
        detect_raw: If True and raw_signal is provided, also detect changepoints
            directly in the raw signal (default True).
        detector: Pre-built detector function. If None, creates one using algo/model.
        algo: Detection algorithm (see create_changepoint_detector).
        model: Cost model for ruptures algorithms.
        **kwargs: Additional arguments for detector.

    Returns:
        Dict with keys:
            - 'latent_changepoints': Changepoint indices in latent space
            - 'raw_changepoints': Changepoint indices mapped to raw signal space
            - 'raw_detected_changepoints': Changepoints detected directly in raw signal
                (empty if detect_raw=False or raw_signal not provided)

    Example:
        >>> cp_results = detect_changepoints(latent_sample, n_changepoints=5)
        >>> print(f"Found {len(cp_results['latent_changepoints'])} changepoints")
    """
    # Convert to numpy if needed
    if torch.is_tensor(latent_sample):
        latent_sample = latent_sample.detach().cpu().numpy()
    latent_np = np.asarray(latent_sample, dtype=np.float64)

    if latent_np.ndim == 1:
        latent_np = latent_np[:, None]
    elif latent_np.ndim > 2:
        latent_np = latent_np.reshape(latent_np.shape[0], -1)

    latent_time_steps = latent_np.shape[0]
    max_bkps = min(max(0, int(n_changepoints)), max(0, latent_time_steps - 1))

    # Create detector if not provided
    detector_fn = detector
    if detector_fn is None:
        detector_fn = create_changepoint_detector(algo=algo, model=model, **kwargs)

    # Detect changepoints in latent space
    latent_cps = detector_fn(latent_np, max_bkps)

    # Initialize results
    raw_cps_from_latent = np.asarray([], dtype=int)
    raw_detected_cps = np.asarray([], dtype=int)

    # Process raw signal if provided
    if raw_signal is not None:
        if torch.is_tensor(raw_signal):
            raw_signal = raw_signal.detach().cpu().numpy()
        raw_np = np.asarray(raw_signal).reshape(-1)
        raw_len = raw_np.shape[0]

        if raw_len > 0:
            # Map latent changepoints to raw signal indices
            raw_cps_from_latent = np.asarray(
                [min(raw_len - 1, int(cp_idx * decimation_factor)) for cp_idx in latent_cps],
                dtype=int,
            )

            # Detect changepoints directly in raw signal
            if detect_raw:
                per_sample_raw_bkps = min(max_bkps, max(0, raw_len - 1))
                if per_sample_raw_bkps > 0:
                    raw_detected_cps = detector_fn(raw_np.reshape(-1, 1), per_sample_raw_bkps)

    return {
        "latent_changepoints": np.asarray(latent_cps, dtype=int),
        "raw_changepoints": raw_cps_from_latent,
        "raw_detected_changepoints": raw_detected_cps,
    }


def summarize_trajectory(
    latent_trajectory: np.ndarray,
    k: int = 5,
    method: str = "changepoint",
    epsilon: Optional[float] = None,
    return_indices: bool = False,
    precomputed_indices: Optional[np.ndarray] = None,
    detector: Optional[Callable] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Summarize latent trajectory by selecting k representative points.

    Args:
        latent_trajectory: Trajectory array of shape (T, D) or (B, T, D).
            If 3D, processes each batch element separately.
        k: Number of keyframes to extract (default 5).
        method: Summarization strategy:
            - 'changepoint': Segment at changepoints, return medoid of each segment
            - 'rdp': Ramer-Douglas-Peucker polyline simplification
            - 'quantile': Quantile samples along arclength
            - 'medoid': Single most representative point (forces k=1)
            - 'frechet': Euclidean mean point (forces k=1)
        epsilon: RDP tolerance parameter (auto-tuned if None, only for 'rdp').
        return_indices: If True, also return time indices of selected points.
        precomputed_indices: Pre-computed changepoint indices (only for 'changepoint').
        detector: Pre-built changepoint detector (only for 'changepoint').

    Returns:
        If return_indices is False:
            keyframes: Array of shape (k, D) or (B, k, D)
        If return_indices is True:
            Tuple of (keyframes, indices) where indices are the time positions.

    Example:
        >>> keyframes = summarize_trajectory(latent_trajectory, k=5, method='changepoint')
        >>> print(keyframes.shape)  # (5, 16) for D=16
    """
    # Handle batch dimension
    if latent_trajectory.ndim == 3:
        batch_size = latent_trajectory.shape[0]
        results = []
        indices_list = []

        for b in range(batch_size):
            # Get precomputed indices for this sample if available
            sample_precomputed = None
            if precomputed_indices is not None:
                if isinstance(precomputed_indices, dict):
                    sample_precomputed = precomputed_indices.get(b)
                elif isinstance(precomputed_indices, (list, tuple)) and b < len(precomputed_indices):
                    sample_precomputed = precomputed_indices[b]

            result = summarize_trajectory(
                latent_trajectory[b],
                k=k,
                method=method,
                epsilon=epsilon,
                return_indices=return_indices,
                precomputed_indices=sample_precomputed,
                detector=detector,
            )
            if return_indices:
                results.append(result[0])
                indices_list.append(result[1])
            else:
                results.append(result)

        if return_indices:
            return np.array(results), np.array(indices_list)
        return np.array(results)

    # Single trajectory processing
    time_steps, n_dims = latent_trajectory.shape

    # Methods that only support k=1
    if method in ["medoid", "frechet"]:
        if k != 1:
            logger.debug(f"Method '{method}' only supports k=1, setting k=1")
        k = 1

    # Single point methods
    if k == 1:
        if method == "medoid":
            centroid = np.mean(latent_trajectory, axis=0)
            distances = np.linalg.norm(latent_trajectory - centroid, axis=1)
            medoid_idx = np.argmin(distances)
            if return_indices:
                return latent_trajectory[medoid_idx : medoid_idx + 1], np.array([medoid_idx])
            return latent_trajectory[medoid_idx : medoid_idx + 1]

        elif method == "frechet":
            mean_point = np.mean(latent_trajectory, axis=0, keepdims=True)
            if return_indices:
                distances = np.linalg.norm(latent_trajectory - mean_point, axis=1)
                closest_idx = np.argmin(distances)
                return mean_point, np.array([closest_idx])
            return mean_point

    # Multi-point methods
    if method == "changepoint":
        # Get or compute changepoints
        if precomputed_indices is not None:
            changepoint_indices = np.asarray(precomputed_indices, dtype=int)
        else:
            if detector is None:
                detector = create_changepoint_detector()
            max_breakpoints = max(0, int(k) - 1)
            changepoint_indices = detector(latent_trajectory, max_breakpoints)

        # Clean up indices
        changepoint_indices = np.asarray(
            sorted({int(idx) for idx in changepoint_indices if 0 < int(idx) < time_steps}),
            dtype=int,
        )

        # Build segment boundaries
        segment_bounds = [0] + changepoint_indices.tolist() + [time_steps]

        # Get medoid of each segment
        keyframe_indices = []
        for i in range(len(segment_bounds) - 1):
            start_idx = segment_bounds[i]
            end_idx = segment_bounds[i + 1]
            segment = latent_trajectory[start_idx:end_idx]

            if len(segment) > 0:
                segment_centroid = np.mean(segment, axis=0)
                distances = np.linalg.norm(segment - segment_centroid, axis=1)
                medoid_offset = np.argmin(distances)
                keyframe_indices.append(start_idx + medoid_offset)

        keyframe_indices = np.array(keyframe_indices)
        summarized = latent_trajectory[keyframe_indices]

        if return_indices:
            return summarized, keyframe_indices
        return summarized

    elif method == "rdp":
        # Ramer-Douglas-Peucker polyline simplification
        def rdp_recursive(points, indices, eps):
            if len(points) <= 2:
                return list(indices)

            start, end = points[0], points[-1]
            line_vec = end - start
            line_len = np.linalg.norm(line_vec)

            if line_len < 1e-10:
                return [indices[0], indices[-1]]

            line_unitvec = line_vec / line_len
            point_vecs = points - start
            projections = np.dot(point_vecs, line_unitvec)
            projected_points = start + np.outer(projections, line_unitvec)
            distances = np.linalg.norm(points - projected_points, axis=1)

            max_dist_idx = np.argmax(distances)
            max_dist = distances[max_dist_idx]

            if max_dist < eps:
                return [indices[0], indices[-1]]

            left_indices = rdp_recursive(
                points[: max_dist_idx + 1], indices[: max_dist_idx + 1], eps
            )
            right_indices = rdp_recursive(points[max_dist_idx:], indices[max_dist_idx:], eps)

            return left_indices[:-1] + right_indices

        # Auto-tune epsilon if not provided
        if epsilon is None:
            all_indices = np.arange(time_steps)
            segment_lengths = np.linalg.norm(
                latent_trajectory[1:] - latent_trajectory[:-1], axis=1
            )
            epsilon_low, epsilon_high = 0.0, np.max(segment_lengths) * 10 if len(segment_lengths) > 0 else 1.0

            for _ in range(20):
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

        # Adjust to exactly k points
        if len(keyframe_indices) > k:
            step = len(keyframe_indices) / k
            selected = [keyframe_indices[int(i * step)] for i in range(k)]
            keyframe_indices = np.array(selected)
        elif len(keyframe_indices) < k:
            while len(keyframe_indices) < k:
                gaps = keyframe_indices[1:] - keyframe_indices[:-1]
                max_gap_idx = np.argmax(gaps)
                new_idx = (keyframe_indices[max_gap_idx] + keyframe_indices[max_gap_idx + 1]) // 2
                keyframe_indices = np.insert(keyframe_indices, max_gap_idx + 1, new_idx)

        summarized = latent_trajectory[keyframe_indices]

        if return_indices:
            return summarized, keyframe_indices
        return summarized

    elif method == "quantile":
        # Quantile samples along arclength
        segment_lengths = np.linalg.norm(latent_trajectory[1:] - latent_trajectory[:-1], axis=1)
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_length[-1]

        quantiles = np.linspace(0, 1, k)
        target_lengths = quantiles * total_length

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
            f"Unknown method '{method}'. Choose from: 'changepoint', 'rdp', 'quantile', 'medoid', 'frechet'"
        )


def summarize_latent_segments(
    latent_mean: np.ndarray,
    epoch: Optional[np.ndarray] = None,
    sample_ids: Optional[Sequence[Any]] = None,
    n_changepoints: int = 5,
    decimation_factor: int = 16,
    raw_sample_rate_hz: float = 4.0,
    detector: Optional[Callable] = None,
    precomputed_changepoints: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Compute per-segment latent statistics after changepoint detection.

    For each segment (region between changepoints), computes mean, variance,
    velocity, and other statistics useful for trajectory analysis.

    Args:
        latent_mean: Latent trajectory array of shape (B, T, D).
        epoch: Optional array of shape (B,) with raw-sample start indices
            relative to delivery. Used for timing information.
        sample_ids: Optional identifiers for each sample.
        n_changepoints: Number of changepoints to detect per sample.
        decimation_factor: Ratio between raw and latent sampling rates.
        raw_sample_rate_hz: Raw signal sampling rate (default 4.0 Hz).
        detector: Pre-built changepoint detector.
        precomputed_changepoints: Dict or list of pre-computed changepoint results.

    Returns:
        List of per-sample dicts, each containing:
            - 'sample_id': Sample identifier
            - 'epoch_raw_index': Epoch timing
            - 'latent_changepoints': Detected changepoint indices
            - 'segments': List of segment statistics dicts with keys:
                * 'segment_index', 'start_step', 'end_step', 'length_steps'
                * 'duration_seconds'
                * 'start_minutes_rel_delivery', 'end_minutes_rel_delivery'
                * 'mean_vector', 'variance_vector'
                * 'dominant_latent_dim'
                * 'mean_velocity', 'mean_speed', 'direction_unit_vector'
                * 'mean_activation_norm'

    Example:
        >>> segment_stats = summarize_latent_segments(latent_mean, n_changepoints=5)
        >>> for sample in segment_stats:
        ...     print(f"Sample {sample['sample_id']}: {len(sample['segments'])} segments")
    """
    # Convert to numpy if needed
    if torch.is_tensor(latent_mean):
        latent_mean = latent_mean.detach().cpu().numpy()
    latent_np = np.asarray(latent_mean, dtype=np.float64)

    if latent_np.ndim != 3:
        raise ValueError(f"latent_mean must have shape (B, T, D); got {latent_np.shape}")

    batch_size, latent_time_steps, latent_dim = latent_np.shape

    # Process epoch
    epoch_np = None
    if epoch is not None:
        if torch.is_tensor(epoch):
            epoch = epoch.detach().cpu().numpy()
        epoch_np = np.asarray(epoch, dtype=np.float64)
        if epoch_np.ndim != 1 or epoch_np.shape[0] != batch_size:
            raise ValueError("epoch must be 1D with length equal to batch size")

    # Set default sample IDs
    if sample_ids is None:
        sample_ids = [f"sample_{idx}" for idx in range(batch_size)]
    if len(sample_ids) != batch_size:
        raise ValueError("sample_ids length must match batch size")

    # Parse precomputed changepoints
    precomputed_list = None
    precomputed_dict = None
    if precomputed_changepoints is not None:
        if isinstance(precomputed_changepoints, dict):
            precomputed_dict = precomputed_changepoints
        elif isinstance(precomputed_changepoints, (list, tuple)):
            precomputed_list = list(precomputed_changepoints)

    # Create detector if needed
    detector_fn = detector
    if detector_fn is None:
        detector_fn = create_changepoint_detector()

    # Timing calculations
    seconds_per_step = decimation_factor / float(raw_sample_rate_hz) if raw_sample_rate_hz else None
    minutes_scale = raw_sample_rate_hz * 60.0 if raw_sample_rate_hz else None

    max_bkps = max(0, int(n_changepoints))
    results: List[Dict[str, Any]] = []

    def _fetch_precomputed(idx: int, sample_label: str):
        entry = None
        if precomputed_dict is not None:
            entry = precomputed_dict.get(sample_label) or precomputed_dict.get(idx)
        if entry is None and precomputed_list is not None and idx < len(precomputed_list):
            entry = precomputed_list[idx]
        return entry

    for batch_idx in range(batch_size):
        sample_id = str(sample_ids[batch_idx])
        latent_sample = latent_np[batch_idx]

        # Get or compute changepoints
        cp_entry = _fetch_precomputed(batch_idx, sample_id)
        if cp_entry is None:
            cp_entry = detect_changepoints(
                latent_sample=latent_sample,
                n_changepoints=max_bkps,
                decimation_factor=decimation_factor,
                raw_signal=None,
                detect_raw=False,
                detector=detector_fn,
            )

        latent_cps = np.asarray(cp_entry.get("latent_changepoints", []), dtype=int)

        # Build segment boundaries
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
            direction_unit_vector = mean_velocity / mean_speed if mean_speed > 0 else np.zeros_like(mean_velocity)

            dominant_dim = int(np.argmax(segment_var)) if segment_var.size else None
            mean_activation_norm = float(np.mean(np.linalg.norm(segment_values, axis=1)))

            # Timing
            duration_seconds = float(length_steps * seconds_per_step) if seconds_per_step else None
            start_minutes = end_minutes = None
            if epoch_value is not None and minutes_scale:
                raw_start = epoch_value + start_step * decimation_factor
                raw_end = epoch_value + end_step * decimation_factor
                start_minutes = float(raw_start / minutes_scale)
                end_minutes = float(raw_end / minutes_scale)

            segments.append({
                "segment_index": seg_idx,
                "start_step": start_step,
                "end_step": end_step,
                "length_steps": length_steps,
                "duration_seconds": duration_seconds,
                "start_minutes_rel_delivery": start_minutes,
                "end_minutes_rel_delivery": end_minutes,
                "mean_vector": segment_mean,
                "variance_vector": segment_var,
                "dominant_latent_dim": dominant_dim,
                "mean_velocity": mean_velocity,
                "mean_speed": mean_speed,
                "direction_unit_vector": direction_unit_vector,
                "mean_activation_norm": mean_activation_norm,
            })

        results.append({
            "sample_id": sample_id,
            "epoch_raw_index": epoch_value,
            "latent_changepoints": latent_cps,
            "segments": segments,
        })

    return results
