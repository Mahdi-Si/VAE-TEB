from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import torch.utils.data
import matplotlib
import os
import pickle
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d
import numpy as np
import matplotlib.pyplot as plt
import random

from early_maestra.adaptor.mimo_adaptor import EarlyMaestraMimoAdaptor
import json
import shutil
import logging
import traceback
import math
import random
import h5py
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import KFold

from hdf5_dataset import create_initial_hdf5, append_sample, append_samples_batch
from guid_analysis import GuidTrackingEntry, GuidScreeningResult


from Variational_AutoEncoder.seqvae_teb.hdf5_dataset.kymatio_phase_scattering import KymatioPhaseScattering1D

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_handler)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

matplotlib.use('Agg')
torch.backends.cudnn.enabled = False

random.seed(42)
np.random.seed(42)
torch.random.manual_seed(42)
torch.cuda.manual_seed(42)


# ----------------------------------------------------------------------------------------------------------------------
# Utility functions
#-----------------------------------------------------------------------------------------------------------------------

def _normalize_guid(guid_str):
    """Normalize a GUID string for matching: uppercase, remove hyphens.

    The labor onset CSV may have GUIDs formatted as
    ``8CFD48E7-EC58-42F1-...`` while .mat filenames use
    ``8CFD48E7EC5842F1...`` (or vice versa).  Normalizing both sides
    ensures reliable matching.
    """
    return guid_str.strip().upper().replace('-', '')


def load_labor_onset_data(csv_path):
    """Load labor onset times from CSV and return a GUID -> seconds mapping.

    The CSV must have columns ``trace_guid`` and ``labor_onset_hours``.
    ``labor_onset_hours`` is in hours relative to delivery (negative = before).
    GUIDs with missing ``labor_onset_hours`` are omitted from the map.

    Keys are normalized (uppercase, hyphens removed) so that
    ``8CFD48E7-EC58-42F1-9DE3-4CAB18C96D07`` and
    ``8CFD48E7EC5842F19DE34CAB18C96D07`` both match.

    Returns:
        dict mapping normalized GUID string to labor onset time in **seconds**
        relative to delivery (same sign convention as ``epoch`` / ``domain_start``).
    """
    df = pd.read_csv(csv_path)
    labor_onset_map = {}
    n_missing = 0
    for _, row in df.iterrows():
        guid = _normalize_guid(str(row['trace_guid']))
        hours = row.get('labor_onset_hours')
        if pd.notna(hours) and str(hours).strip() != '':
            labor_onset_map[guid] = float(hours) * 3600.0  # hours -> seconds
        else:
            n_missing += 1
    print(f"[TLO] Loaded labor onset data for {len(labor_onset_map)} GUIDs "
          f"({n_missing} with missing labor_onset_hours) from {csv_path}")
    if labor_onset_map:
        sample_key = next(iter(labor_onset_map))
        sample_val = labor_onset_map[sample_key]
        print(f"[TLO] Sample CSV GUID (normalized): {sample_key} -> {sample_val/3600:.2f}h")
    return labor_onset_map


def interpolate_bad_values(signal_2d):
    """Replace NaN/Inf values with linear interpolation, per row.

    For each row (segment), valid samples are used as knots and bad samples
    are filled by linear interpolation.  Edge NaNs are extrapolated flat
    (np.interp default).  Rows that are entirely bad are filled with 0.
    Operates in-place and returns the array.
    """
    bad = ~np.isfinite(signal_2d)
    if not bad.any():
        return signal_2d
    indices = np.arange(signal_2d.shape[1])
    for row_idx in range(signal_2d.shape[0]):
        row_bad = bad[row_idx]
        if not row_bad.any():
            continue
        row_good = ~row_bad
        if not row_good.any():
            signal_2d[row_idx] = 0.0
            continue
        signal_2d[row_idx, row_bad] = np.interp(
            indices[row_bad], indices[row_good], signal_2d[row_idx, row_good])
    return signal_2d


def find_flat_regions(signal, tolerance=1e-3, min_length=20):
    """
    Finds flat regions in the signal that are at least `min_length` samples long.

    Parameters:
      - signal: array-like, the input signal
      - tolerance: float, the threshold for considering consecutive points as "flat"
      - min_length: int, minimum number of consecutive "flat" samples to qualify

    Returns:
      - flat_regions: list of tuples, each tuple contains the start and end indices
                      of a flat region of length >= min_length
    """
    flat_regions = []
    start_idx = None

    for i in range(1, len(signal)):
        if abs(signal[i] - signal[i-1]) <= tolerance:
            # we're in a flat segment
            if start_idx is None:
                start_idx = i-1
        else:
            # break in flatness
            if start_idx is not None:
                end_idx = i-1
                if (end_idx - start_idx + 1) >= min_length:
                    flat_regions.append((start_idx, end_idx))
                start_idx = None

    # handle case where signal ends in a flat region
    if start_idx is not None:
        end_idx = len(signal) - 1
        if (end_idx - start_idx + 1) >= min_length:
            flat_regions.append((start_idx, end_idx))

    return flat_regions


def detect_flat_region(signal, threshold=0.5, window=5):
    """
    Helper method
    Detects flat regions in the given signal.
    :param signal: List or numpy array containing the signal values.
    :param threshold: The threshold for the derivative to consider the signal as flat.
    :param window: The window size for smoothing the derivative.
    :return: A list of tuples indicating the start and end indices of flat regions.
    """
    # Calculate the derivative (rate of change) of the signal
    derivative = np.diff(signal, n=1)
    # Smooth the derivative using a uniform filter to reduce noise
    smooth_derivative = uniform_filter1d(np.abs(derivative), size=window)
    # Find indices where the absolute value of the smoothed derivative is below the threshold
    flat_indices = np.where(smooth_derivative < threshold)[0]
    # Group the flat indices into contiguous flat regions
    flat_regions = []
    if flat_indices.size > 0:
        start_idx = flat_indices[0]
        for i in range(1, len(flat_indices)):
            # If there is a gap between indices, it means the end of the current flat region
            if flat_indices[i] > flat_indices[i - 1] + 1:
                end_idx = flat_indices[i - 1]
                flat_regions.append((start_idx, end_idx))
                start_idx = flat_indices[i]
        # Add the last flat region
        flat_regions.append((start_idx, flat_indices[-1]))

    return flat_regions


def deduplicate_segments(domain_starts, sample_weights):
    """Remove duplicate segments that share the same domain_start value.

    When multiple short raw segments align to the same grid point in
    ``split_long(do_equalize=True)``, they produce identical ``domain_start``
    values.  This function keeps only the segment with the highest mean
    sample weight for each unique ``domain_start``.

    Args:
        domain_starts: Array-like of domain_start values (one per segment).
        sample_weights: 2-D array of shape ``(n_segments, n_samples)``.

    Returns:
        ``(keep_indices, removed_indices, duplicate_groups)`` where
        *duplicate_groups* maps each duplicated ``domain_start`` value to
        the list of original indices that shared it.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for idx, ds in enumerate(domain_starts):
        groups[ds].append(idx)

    keep_indices = []
    removed_indices = []
    duplicate_groups = {}

    for ds, indices in groups.items():
        if len(indices) == 1:
            keep_indices.append(indices[0])
        else:
            duplicate_groups[ds] = indices
            # Keep the segment with the highest mean sample weight
            best_idx = max(indices, key=lambda i: float(np.mean(sample_weights[i, :])))
            keep_indices.append(best_idx)
            for i in indices:
                if i != best_idx:
                    removed_indices.append(i)

    keep_indices.sort()
    removed_indices.sort()
    return keep_indices, removed_indices, duplicate_groups


def prescreen_guid(record_path, base_block_size=3520, overlap_percentage=1/11,
                   labor_onset_map=None):
    """Run MIMO + quality filtering on a single .mat file WITHOUT scattering.

    Extracts the same segment pipeline as
    ``create_hdf5_dataset_from_records_list`` (MIMO load, prepare_data,
    sanitization, deduplication, weight + flat-region filtering) but skips
    the expensive scattering transform, making it ~100x faster.

    Args:
        record_path: Full path to the .mat file.
        base_block_size: Base block size for ``prepare_data``
            (default 3520 -> 5280 sample segments = 22 min at 4 Hz).
        overlap_percentage: Overlap fraction for ``split_long``
            (default 1/11 -> 20 min step, continuous after 1-min trim).
        labor_onset_map: Optional dict mapping normalized GUID ->
            labor onset time in seconds (from ``load_labor_onset_data``).

    Returns:
        GuidScreeningResult: Screening outcome for this GUID.
    """
    signal_length = int(base_block_size * 1.5)
    step_size = int(signal_length * (1 - overlap_percentage))
    segment_minutes = signal_length / (4 * 60)  # at 4 Hz
    step_minutes = step_size / (4 * 60)

    guid_key = os.path.splitext(os.path.basename(record_path))[0]

    # Default error result
    def _error_result(msg):
        return GuidScreeningResult(
            guid=guid_key, record_path=record_path,
            n_total_segments=0, n_after_dedup=0, n_valid_segments=0,
            n_low_weight=0, n_flat_region=0, n_duplicate=0,
            estimated_valid_hours=0.0, has_labor_onset=False,
            labor_onset_hours=float('nan'),
            domain_start_range=(float('nan'), float('nan')),
            n_post_delivery=0, rejection_reason=None,
            error=True, error_msg=msg)

    try:
        # --- MIMO load & prepare (identical to create_hdf5_dataset_from_records_list) ---
        mimo_adaptor = EarlyMaestraMimoAdaptor(
            do_transpose=True, process_targets=True,
            n_aux_labels=None, signal_indices=range(0, 2),
            n_input_chan=2, labels=["HIE", "ACIDOSIS", "HEALTHY"],
            up_shift_secs=-20, default_target_index=0)
        mimo_adaptor.read_single_input(
            record_path, out_dec_factor=16, out_dec_factor_offset=0,
            target_is_onehot=True, dtype=np.float32)
        mimo_prepared, _ = mimo_adaptor.mimo.prepare_data(
            batch_size=1, do_evaluate=True, align_left=True,
            do_split=True, do_pad=True, do_reflect=True,
            base_length=base_block_size, do_equalize=True, do_merge=True,
            min_domain_start=[-44640, -44640],
            max_domain_start=[np.inf, np.inf],
            overlap_percentage=overlap_percentage)

        fhr = mimo_prepared.block_input[:, :, 1].copy()
        up = mimo_prepared.block_input[:, :, 0].copy()
        domain_starts = mimo_prepared.domain_start
        sample_weights = mimo_prepared.sample_weights
        n_total = fhr.shape[0]

        # --- Sanitize signals ---
        interpolate_bad_values(fhr)
        interpolate_bad_values(up)
        fhr = np.clip(fhr, 0, 500).astype(np.float32)
        up = np.clip(up, -50, 500).astype(np.float32)
        tiny = np.finfo(np.float32).tiny
        fhr[(fhr != 0) & (np.abs(fhr) < tiny)] = 0.0
        up[(up != 0) & (np.abs(up) < tiny)] = 0.0

        # --- Deduplicate ---
        keep_idx, removed_idx, _ = deduplicate_segments(domain_starts, sample_weights)
        n_duplicate = len(removed_idx)
        if removed_idx:
            fhr = fhr[keep_idx]
            up = up[keep_idx]
            sample_weights = sample_weights[keep_idx]
            domain_starts = [domain_starts[i] for i in keep_idx]
        n_after_dedup = fhr.shape[0]

        # --- Quality filtering (weight threshold + flat region) ---
        n_low_weight = 0
        n_flat_region = 0
        valid_domain_starts = []
        for i in range(fhr.shape[0]):
            if np.mean(sample_weights[i, :]) < 0.90:
                n_low_weight += 1
                continue
            fhr_flat = find_flat_regions(fhr[i, :], tolerance=1e-9)
            up_flat = find_flat_regions(up[i, :], tolerance=1e-9)
            fhr_lens = [end - start + 1 for start, end in fhr_flat]
            up_lens = [end - start + 1 for start, end in up_flat]
            max_flat_fhr = max(fhr_lens, default=0)
            max_flat_up = max(up_lens, default=0)
            total_flat_fhr = sum(l for l in fhr_lens if l >= 240)
            if (max_flat_fhr > 480 or max_flat_up > 1200
                    or total_flat_fhr > 1200):
                n_flat_region += 1
            else:
                valid_domain_starts.append(domain_starts[i])

        n_valid = len(valid_domain_starts)

        # --- Coverage estimate ---
        if n_valid > 0:
            estimated_hours = ((n_valid - 1) * step_minutes + segment_minutes) / 60
        else:
            estimated_hours = 0.0

        # --- Labor onset lookup ---
        has_labor_onset = False
        labor_onset_hours = float('nan')
        if labor_onset_map:
            normalized_key = _normalize_guid(guid_key)
            lo_sec = labor_onset_map.get(normalized_key, float('nan'))
            if not math.isnan(lo_sec):
                has_labor_onset = True
                labor_onset_hours = lo_sec / 3600.0

        # --- Domain start range & post-delivery count ---
        if valid_domain_starts:
            ds_range = (min(valid_domain_starts), max(valid_domain_starts))
            n_post_delivery = sum(1 for ds in valid_domain_starts if ds >= 0)
        else:
            ds_range = (float('nan'), float('nan'))
            n_post_delivery = 0

        return GuidScreeningResult(
            guid=guid_key, record_path=record_path,
            n_total_segments=n_total, n_after_dedup=n_after_dedup,
            n_valid_segments=n_valid, n_low_weight=n_low_weight,
            n_flat_region=n_flat_region, n_duplicate=n_duplicate,
            estimated_valid_hours=estimated_hours,
            has_labor_onset=has_labor_onset,
            labor_onset_hours=labor_onset_hours,
            domain_start_range=ds_range,
            n_post_delivery=n_post_delivery,
            rejection_reason=None)

    except Exception as e:
        return _error_result(str(e))


def prescreen_guids_for_classification(
    candidate_files, labor_onset_map=None,
    base_block_size=3520, overlap_percentage=1/11,
    min_segments_unhealthy=6, min_segments_healthy=9,
    max_post_delivery_ratio=0.30, output_dir=None,
    num_workers=None):
    """Pre-screen all classification candidate GUIDs and return filtered file lists.

    Runs ``prescreen_guid`` on every candidate, applies per-class acceptance
    criteria, logs a comprehensive summary table, and optionally saves
    screening results as JSON for reproducibility.

    Args:
        candidate_files: Dict mapping subgroup name -> list of .mat file paths.
            Subgroup names must start with ``acidosis_``, ``hie_``, or
            ``healthy_`` to determine class-specific thresholds.
        labor_onset_map: Optional dict from ``load_labor_onset_data``.
        base_block_size: Base block size for MIMO (default 3520).
        overlap_percentage: Overlap fraction (default 1/11).
        min_segments_unhealthy: Minimum valid segments for acidosis/HIE GUIDs.
        min_segments_healthy: Minimum valid segments for healthy GUIDs.
        max_post_delivery_ratio: Reject GUIDs where more than this fraction
            of valid segments are post-delivery (``domain_start >= 0``).
        output_dir: If provided, save ``guid_screening_results.json`` here.
        num_workers: Number of parallel workers for pre-screening. Defaults to
            ``min(os.cpu_count(), 8)``. Set to 1 to disable parallelism.

    Returns:
        Dict mapping subgroup name -> list of .mat file paths that passed
        screening.

    Raises:
        ValueError: If any subgroup is entirely emptied by screening.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)

    all_results: Dict[str, List[GuidScreeningResult]] = {}
    filtered_files: Dict[str, List[str]] = {}

    total_candidates = sum(len(v) for v in candidate_files.values())
    logger.info(f"Pre-screening {total_candidates} GUIDs across "
                f"{len(candidate_files)} subgroups "
                f"(workers={num_workers})")

    prescreen_fn = partial(
        prescreen_guid,
        base_block_size=base_block_size,
        overlap_percentage=overlap_percentage,
        labor_onset_map=labor_onset_map)

    for subgroup, file_list in candidate_files.items():
        is_unhealthy = subgroup.startswith(('acidosis_', 'hie_'))
        min_segs = min_segments_unhealthy if is_unhealthy else min_segments_healthy
        results = []

        if num_workers <= 1:
            # Sequential fallback
            for record_path in tqdm(file_list,
                                    desc=f"Pre-screening {subgroup}",
                                    leave=False):
                result = prescreen_fn(record_path)
                results.append(result)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_path = {
                    executor.submit(prescreen_fn, path): path
                    for path in file_list}
                for future in tqdm(as_completed(future_to_path),
                                   total=len(future_to_path),
                                   desc=f"Pre-screening {subgroup}",
                                   leave=False):
                    results.append(future.result())

        # Restore original file order (as_completed returns in arbitrary order,
        # but downstream KFold splits depend on deterministic ordering).
        path_order = {p: i for i, p in enumerate(file_list)}
        results.sort(key=lambda r: path_order.get(r.record_path, 0))

        # --- Apply rejection criteria ---
        for result in results:
            if result.error:
                result.rejection_reason = f"processing error: {result.error_msg}"
            elif result.n_valid_segments < min_segs:
                result.rejection_reason = (
                    f"insufficient segments: {result.n_valid_segments} < {min_segs} "
                    f"(~{result.estimated_valid_hours:.1f}h)")
            elif not is_unhealthy and not result.has_labor_onset:
                result.rejection_reason = "healthy GUID missing labor onset data"
            elif (result.n_valid_segments > 0
                  and result.n_post_delivery / result.n_valid_segments
                      > max_post_delivery_ratio):
                result.rejection_reason = (
                    f"post-delivery ratio too high: "
                    f"{result.n_post_delivery}/{result.n_valid_segments} "
                    f"= {result.n_post_delivery/result.n_valid_segments:.0%} "
                    f"> {max_post_delivery_ratio:.0%}")

        all_results[subgroup] = results
        accepted = [r for r in results if r.rejection_reason is None]
        filtered_files[subgroup] = [r.record_path for r in accepted]

    # --- Log summary table ---
    logger.info("=" * 80)
    logger.info("GUID PRE-SCREENING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Subgroup':<28} {'Before':>7} {'After':>7} {'Rejected':>8} "
                f"{'Error':>6} {'LowSeg':>7} {'NoTLO':>6} {'PostDel':>8}")
    logger.info("-" * 80)

    total_before = 0
    total_after = 0
    for subgroup in candidate_files:
        results = all_results[subgroup]
        n_before = len(results)
        accepted = [r for r in results if r.rejection_reason is None]
        n_after = len(accepted)
        n_error = sum(1 for r in results if r.error)
        n_low_seg = sum(1 for r in results
                        if r.rejection_reason and 'insufficient segments' in r.rejection_reason)
        n_no_tlo = sum(1 for r in results
                       if r.rejection_reason and 'missing labor onset' in r.rejection_reason)
        n_post_del = sum(1 for r in results
                         if r.rejection_reason and 'post-delivery ratio' in r.rejection_reason)
        total_before += n_before
        total_after += n_after
        logger.info(f"{subgroup:<28} {n_before:>7} {n_after:>7} "
                    f"{n_before - n_after:>8} {n_error:>6} "
                    f"{n_low_seg:>7} {n_no_tlo:>6} {n_post_del:>8}")

    logger.info("-" * 80)
    logger.info(f"{'TOTAL':<28} {total_before:>7} {total_after:>7} "
                f"{total_before - total_after:>8}")
    logger.info("=" * 80)

    # --- Warnings ---
    for subgroup in candidate_files:
        n_before = len(all_results[subgroup])
        n_after = len(filtered_files[subgroup])
        if n_after == 0:
            raise ValueError(
                f"Pre-screening emptied subgroup '{subgroup}' "
                f"({n_before} -> 0 GUIDs). Lower min_segments thresholds or "
                f"check data quality.")
        reject_pct = (n_before - n_after) / n_before * 100 if n_before else 0
        if reject_pct > 50:
            logger.warning(
                f"Subgroup '{subgroup}': {reject_pct:.0f}% rejected "
                f"({n_before} -> {n_after})")

    # --- Save screening results as JSON ---
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "guid_screening_results.json")
        serializable = {}
        for subgroup, results in all_results.items():
            serializable[subgroup] = []
            for r in results:
                entry = {
                    'guid': r.guid,
                    'record_path': r.record_path,
                    'n_total_segments': r.n_total_segments,
                    'n_after_dedup': r.n_after_dedup,
                    'n_valid_segments': r.n_valid_segments,
                    'n_low_weight': r.n_low_weight,
                    'n_flat_region': r.n_flat_region,
                    'n_duplicate': r.n_duplicate,
                    'estimated_valid_hours': r.estimated_valid_hours,
                    'has_labor_onset': r.has_labor_onset,
                    'labor_onset_hours': (None if math.isnan(r.labor_onset_hours)
                                          else r.labor_onset_hours),
                    'domain_start_range': [
                        None if math.isnan(r.domain_start_range[0])
                        else r.domain_start_range[0],
                        None if math.isnan(r.domain_start_range[1])
                        else r.domain_start_range[1]],
                    'n_post_delivery': r.n_post_delivery,
                    'rejection_reason': r.rejection_reason,
                    'error': r.error,
                    'error_msg': r.error_msg,
                }
                serializable[subgroup].append(entry)
        with open(json_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Screening results saved to {json_path}")

    return filtered_files


def plot_fhr_signals(fhr_data, domain_starts, start_idx=0, sampling_rate=4, save_path=None):
    """
    Plot 4 consecutive FHR signals in vertically stacked subplots.

    Parameters:
    -----------
    fhr_data : np.ndarray
        FHR data with shape (n_segments, n_samples)
    domain_starts : list
        List of domain start values for each segment
    start_idx : int, default=0
        Starting index for plotting (will plot signals at start_idx, start_idx+1, start_idx+2, start_idx+3)
    sampling_rate : float, default=4
        Sampling rate in Hz for time axis conversion
    save_path : str, optional
        Path to save the plot. If None, plot will be displayed.

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """

    # Ensure we have enough signals to plot
    n_signals = fhr_data.shape[0]
    if start_idx + 3 >= n_signals:
        raise ValueError(f"Not enough signals to plot. Have {n_signals}, need at least {start_idx + 4}")

    # Create time axis in minutes
    n_samples = fhr_data.shape[1]
    time_minutes = np.arange(n_samples) / (sampling_rate * 60)  # Convert to minutes

    # Create figure with 4 vertically stacked subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 6), sharex=True)
    fig.suptitle(f'FHR Signals - Starting from Index {start_idx}', fontsize=12)

    for i in range(4):
        signal_idx = start_idx + i
        ax = axes[i]

        # Plot the FHR signal
        ax.plot(time_minutes, fhr_data[signal_idx, :], linewidth=1.0)

        # Add labels and formatting
        ax.set_ylabel('FHR (bpm)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(50, 200)  # Typical FHR range

        # Add domain start information
        domain_start_minutes = domain_starts[signal_idx] / 60  # Convert to minutes
        ax.set_title(f'Signal {signal_idx} - Domain Start: {domain_start_minutes:.1f} min',
                     fontsize=12, pad=10)

        # Add some statistics
        mean_fhr = np.mean(fhr_data[signal_idx, :])
        std_fhr = np.std(fhr_data[signal_idx, :])
        ax.text(0.02, 0.95, f'Mean: {mean_fhr:.1f} ± {std_fhr:.1f} bpm',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='top')

    # Set x-axis label only for bottom subplot
    axes[-1].set_xlabel('Time (minutes)', fontsize=12)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_channel_range(tensor: torch.Tensor,
                       start_chan: int,
                       end_chan: int,
                       save_path: str=None,
                       figsize=(10, 6)):
    """
    Plots channels [start_chan…end_chan] of a (m, n) tensor as an image.

    Args:
        tensor (torch.Tensor): shape (m, n), where m=channels, n=time steps
        start_chan (int): 1-based index of first channel to plot
        end_chan (int):   1-based index of last  channel to plot
        save_path (str):  path (including filename) to save the PNG
        figsize (tuple):  matplotlib figure size
    """
    # convert to NumPy and select (convert 1-based to 0-based indices)
    # ensure tensor is on CPU
    arr = tensor.detach().cpu().numpy()
    sel = arr[start_chan - 1 : end_chan, :]  # shape = (end_chan-start_chan+1, n)

    plt.figure(figsize=figsize)
    plt.imshow(sel,
               aspect='auto',
               interpolation='none',
               origin='lower')      # channels on the y-axis
    plt.colorbar(label='Amplitude')
    plt.xlabel('Time step')
    plt.ylabel('Channel index')
    plt.title(f'Channels {start_chan}–{end_chan}')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


def plot_stacked_channels(tensor: torch.Tensor,
                          start_chan: int = 1,
                          end_chan: int = None,
                          save_path: str = None,
                          figsize=(12, 2),
                          dpi=150):
    """
    Plots each channel in [start_chan…end_chan] of a (m, n) tensor as individual
    time-series subplots stacked vertically.

    Args:
        tensor (torch.Tensor): shape (m, n), where m=channels, n=time steps
        start_chan (int): 1-based index of first channel to plot (default: 1)
        end_chan (int):   1-based index of last channel to plot (default: m)
        save_path (str):  if provided, path (including filename) to save the figure
        figsize (tuple):  width & height of each subplot row, total height = figsize[1] * num_chans
        dpi (int):        resolution of saved figure
    """
    # Move to CPU/NumPy and handle channel bounds
    arr = tensor.detach().cpu().numpy()
    m, n = arr.shape
    if end_chan is None or end_chan > m:
        end_chan = m
    if start_chan < 1 or start_chan > end_chan:
        raise ValueError("start_chan must be ≥1 and ≤ end_chan")

    # Select channels (convert 1-based to 0-based)
    sel = arr[start_chan - 1:end_chan, :]  # shape = (num_chans, n)
    num_chans = sel.shape[0]

    # Create figure & axes
    fig, axes = plt.subplots(num_chans,
                             1,
                             figsize=(figsize[0], figsize[1] * num_chans),
                             sharex=True)

    # If only one channel, axes is not an array
    if num_chans == 1:
        axes = [axes]

    # Plot each channel
    for idx, ax in enumerate(axes, start=start_chan):
        ax.plot(sel[idx - start_chan, :])
        ax.set_ylabel(f"Ch {idx}")
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[-1].set_xlabel("Time step")
    fig.suptitle(f"Channels {start_chan}–{end_chan} (stacked)", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.close()

# ------------------------------------------------------------------------------------------
# folds creation method
# ------------------------------------------------------------------------------------------

def create_cv_splits(
    data: dict[str, list[str]],
    n_splits: int = 10,
    val_ratio: float = 0.1,
    random_state: int = 42
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """
    Perform stratified-by-subgroup 10-fold CV, with an inner train/validation split.

    Args:
        data: Mapping subgroup name → list of file paths.
        n_splits: Number of outer folds (here: 10).
        val_ratio: Fraction of the remaining after test to use as validation.
        random_state: Seed for reproducibility.

    Returns:
        folds: {
            'fold_1': {
                'train': { subgroup: [paths], … },
                'val':   { subgroup: [paths], … },
                'test':  { subgroup: [paths], … },
            },
            … up to 'fold_10'
        }
    """
    # prepare outer KFold per subgroup
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits_per_group = {
        group: list(kf.split(file_list))
        for group, file_list in data.items()
    }

    folds: dict[str, dict] = {}
    for fold_idx in range(n_splits):
        fold_name = f"fold_{fold_idx+1}"
        fold_data = {'train': {}, 'val': {}, 'test': {}}

        for group, splits in splits_per_group.items():
            train_val_idx, test_idx = splits[fold_idx]

            # build test set for this group
            test_files = [data[group][i] for i in test_idx]

            # split train/val from the remaining indices
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_ratio,
                shuffle=True,
                random_state=random_state
            )
            train_files = [data[group][i] for i in train_idx]
            val_files   = [data[group][i] for i in val_idx]

            # store
            fold_data['train'][group] = train_files
            fold_data['val'][group]   = val_files
            fold_data['test'][group]  = test_files

        folds[fold_name] = fold_data

    return folds

def compute_scattering_masks(signal_length, scattering_T=16, device=None):
    """Compute all coefficient selection masks once.

    v3 changes vs v2:
    - Cross-phase: UP cap raised from 0.02 to 0.05 Hz in both bands
    - UP self-phase: new, using select_fhr_phase_coefficients(min_freq=0.002)

    Returns dict with masks and channel counts.
    """
    tmp_model = KymatioPhaseScattering1D(
        J=11, Q=4, T=scattering_T, shape=signal_length, device=device,
        tukey_alpha=None, max_order=1)

    # FHR self-phase (unchanged from v2)
    phase_sel = tmp_model.select_fhr_phase_coefficients(min_freq=0.006)
    phase_mask = phase_sel['optimal_mask']

    # FHR-UP cross-phase (v3: raised UP cap to 0.05 Hz)
    cross_sel = tmp_model.select_fhr_up_cross_coefficients_v2(
        band_a_up_max_hz=0.05,
        band_b_up_max_hz=0.05)
    cross_mask = cross_sel['cross_mask']

    # UP self-phase (v3: new — lower min_freq captures contraction frequencies)
    up_phase_sel = tmp_model.select_fhr_phase_coefficients(min_freq=0.002)
    up_phase_mask = up_phase_sel['optimal_mask']

    n_phase = int(phase_mask.sum().item())
    n_cross = int(cross_mask.sum().item())
    n_up_phase = int(up_phase_mask.sum().item())
    n_combined_cross = n_cross + n_up_phase

    cross_metadata = cross_sel.get('metadata', {})

    return {
        'phase_mask': phase_mask,
        'cross_mask': cross_mask,
        'up_phase_mask': up_phase_mask,
        'n_phase': n_phase,
        'n_cross': n_cross,
        'n_up_phase': n_up_phase,
        'n_combined_cross': n_combined_cross,
        'cross_metadata': cross_metadata,
    }


# ----------------------------------------------------------------------------------------------------------------------
# Dataset creation method
#-----------------------------------------------------------------------------------------------------------------------
def create_hdf5_dataset_from_records_list(
    hdf5_path=None, records_list=None, file_limit=-1,
    base_block_size=3520, save_name=None, min_domain_start=None,
    cs_label=None, bg_label=None, pre_defined_target=None, device=None, overlap_percentage=1/11,
    run_guid_analysis=False, precomputed_masks=None, labor_onset_map=None,
    scatter_batch_size=16):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scattering_T = 16
    signal_length = int(base_block_size * 1.5)
    sequence_length = signal_length // scattering_T

    # Initialize scattering transform with optimal configuration for FHR analysis
    st_model = KymatioPhaseScattering1D(J=11, Q=4, T=scattering_T, shape=signal_length, device=device, tukey_alpha=None, max_order=1)

    # Get coefficient selection masks (v3: raised UP cap + UP self-phase)
    if precomputed_masks is not None:
        phase_mask = precomputed_masks['phase_mask'].to(device)
        cross_mask = precomputed_masks['cross_mask'].to(device)
        up_phase_mask = precomputed_masks['up_phase_mask'].to(device)
        n_phase = precomputed_masks['n_phase']
        n_cross = precomputed_masks['n_cross']
        n_up_phase = precomputed_masks['n_up_phase']
        n_combined_cross = precomputed_masks['n_combined_cross']
    else:
        masks = compute_scattering_masks(signal_length, scattering_T, device)
        phase_mask = masks['phase_mask']
        cross_mask = masks['cross_mask']
        up_phase_mask = masks['up_phase_mask']
        n_phase = masks['n_phase']
        n_cross = masks['n_cross']
        n_up_phase = masks['n_up_phase']
        n_combined_cross = masks['n_combined_cross']

    logger.info(f"Using coefficient selection (v3: raised UP cap + UP self-phase):")
    logger.info(f"  - FHR phase: {n_phase} coefficients")
    logger.info(f"  - FHR-UP cross-phase: {n_cross} coefficients")
    logger.info(f"  - UP self-phase: {n_up_phase} coefficients")
    logger.info(f"  - Combined fhr_up_ph: {n_combined_cross} (cross + UP self-phase)")
    errors_list = []
    guid_tracking = {} if run_guid_analysis else None
    counter_rec = 0
    if file_limit > 0:
        records_list = records_list[:file_limit]
    for record in tqdm(records_list):
        counter_rec += 1
        logger.info(f'The count is  --------->  {counter_rec}')
        try:
            # target_map: Healthy=0, Acidosis=1, HIE=2
            # pre_defined_target: HEALTHY=1, ACIDOSIS=2, HIE=3
            # So default_target_index = pre_defined_target - 1
            default_ti = (pre_defined_target - 1) if pre_defined_target is not None else None
            mimo_adaptor = EarlyMaestraMimoAdaptor(
                do_transpose=True,
                process_targets=True,
                n_aux_labels=None,
                signal_indices=range(0, 2),
                n_input_chan=2,
                labels=["HIE", "ACIDOSIS", "HEALTHY"],
                up_shift_secs=-20,
                default_target_index=default_ti,
                )
            mimo_adaptor.read_single_input(
                record, out_dec_factor=16, out_dec_factor_offset=0, target_is_onehot=True,
                dtype=np.float32)
            mimo_prepared, n_padded = mimo_adaptor.mimo.prepare_data(
                batch_size=1, do_evaluate=True, align_left=True,
                do_split=True,
                do_pad=True,
                do_reflect=True,
                base_length=base_block_size,
                do_equalize=True,
                do_merge=True,
                min_domain_start=[-44640, -44640],
                max_domain_start=[np.inf, np.inf],
                overlap_percentage=overlap_percentage,
                )

            epoch_samples = mimo_prepared.block_input.shape[1]
            fhr = mimo_prepared.block_input[:, :, 1].copy()
            up = mimo_prepared.block_input[:, :, 0].copy()

            # Sanitize raw signals: linear-interpolate NaN/Inf, clamp range,
            # flush denormalized floats.
            n_bad_fhr = int((~np.isfinite(fhr)).sum())
            n_bad_up = int((~np.isfinite(up)).sum())
            if n_bad_fhr or n_bad_up:
                logger.warning(f"{os.path.splitext(os.path.split(record)[1])[0]}: "
                               f"interpolating {n_bad_fhr + n_bad_up} NaN/Inf values "
                               f"(FHR={n_bad_fhr}, UP={n_bad_up})")
                interpolate_bad_values(fhr)
                interpolate_bad_values(up)
            # Clamp to generous physiological range (avoids MKL FFT overflow)
            fhr = np.clip(fhr, 0, 500).astype(np.float32)
            up = np.clip(up, -50, 500).astype(np.float32)
            # Flush denormalized floats to zero (MKL can choke on subnormals)
            tiny = np.finfo(np.float32).tiny  # ~1.18e-38
            fhr[(fhr != 0) & (np.abs(fhr) < tiny)] = 0.0
            up[(up != 0) & (np.abs(up) < tiny)] = 0.0
            domain_starts = mimo_prepared.domain_start
            guid_key = os.path.splitext(os.path.split(record)[1])[0]
            # Lookup labor onset time for this GUID (seconds relative to delivery)
            if labor_onset_map:
                normalized_key = _normalize_guid(guid_key)
                labor_onset_sec = labor_onset_map.get(normalized_key, float('nan'))
                if math.isnan(labor_onset_sec):
                    print(f"  [TLO] GUID not in CSV: {guid_key} (normalized: {normalized_key})")
                else:
                    print(f"  [TLO] GUID matched: {guid_key} -> labor_onset={labor_onset_sec/3600:.2f}h ({labor_onset_sec:.0f}s)")
            else:
                labor_onset_sec = float('nan')
            # Surface post-delivery segments from prepare_data
            n_post_delivery = sum(1 for ds in domain_starts if ds >= 0)
            if n_post_delivery > 0:
                logger.warning(
                    f"{guid_key}: {n_post_delivery}/{len(domain_starts)} segments have "
                    f"domain_start >= 0 (post-delivery). Max domain_start="
                    f"{max(domain_starts):.1f}s. Source: split_long() equalization "
                    f"padding or recording extends past delivery.")
            if guid_tracking is not None:
                guid_tracking[guid_key] = GuidTrackingEntry(
                    all_domain_starts=[float(ds) for ds in domain_starts],
                    included_domain_starts=[], skipped_low_weight=[],
                    skipped_flat_region=[], skipped_scatter_failed=[])
            block_targets = mimo_prepared.block_target
            if isinstance(block_targets, np.ndarray) and block_targets.ndim < 3:
                block_targets = []
            elif isinstance(block_targets, list):
                block_targets = []
            # sample_weights = np.repeat(mimo_prepared.sample_weights, repeats=16, axis=1)
            sample_weights = mimo_prepared.sample_weights

            # ---- Deduplicate segments with identical domain_start ----
            keep_idx, removed_idx, dup_groups = deduplicate_segments(domain_starts, sample_weights)
            if removed_idx:
                logger.info(f"{guid_key}: deduplicating {len(removed_idx)} segments "
                            f"across {len(dup_groups)} duplicate domain_start groups")
                for ds_val, idx_list in dup_groups.items():
                    logger.debug(f"  domain_start={ds_val:.1f}: {len(idx_list)} segments, "
                                 f"keeping best weight")
                if guid_tracking is not None:
                    guid_tracking[guid_key].skipped_duplicate.extend(
                        float(domain_starts[i]) for i in removed_idx)
                fhr = fhr[keep_idx]
                up = up[keep_idx]
                sample_weights = sample_weights[keep_idx]
                domain_starts = [domain_starts[i] for i in keep_idx]

            # Initialize per-segment status tracking for signal strip plots
            plot_segment_status = None
            if guid_tracking is not None:
                plot_segment_status = ['pending'] * fhr.shape[0]

            record_file = os.path.split(record)
            record_name = os.path.splitext(record_file[1])

            # ---- Step 1: Filter segments BEFORE scattering ----
            valid_indices = []
            for i in range(fhr.shape[0]):
                if np.mean(sample_weights[i, :]) < 0.90:
                    if guid_tracking is not None:
                        guid_tracking[guid_key].skipped_low_weight.append(float(domain_starts[i]))
                        if plot_segment_status is not None:
                            plot_segment_status[i] = 'low_weight'
                    continue

                # Flat region detection
                fhr_flat_regions = find_flat_regions(fhr[i, :], tolerance=1e-9)
                up_flat_regions = find_flat_regions(up[i, :], tolerance=1e-9)
                fhr_flat_lengths = [end - start + 1 for start, end in fhr_flat_regions]
                up_flat_lengths = [end - start + 1 for start, end in up_flat_regions]
                max_flat_fhr_len = max(fhr_flat_lengths, default=0)
                max_flat_up_len = max(up_flat_lengths, default=0)
                # For FHR cumulative: only count flat regions >= 40 samples (10 sec at 4 Hz)
                total_flat_fhr_len = sum(l for l in fhr_flat_lengths if l >= 240)
                # total_flat_up_len = sum(up_flat_lengths)  # UP cumulative removed

                if (max_flat_fhr_len > 480 or
                        max_flat_up_len > 1200 or
                        total_flat_fhr_len > 1200):
                        # total_flat_up_len > 2400  # UP cumulative condition removed
                    logger.info(f'Flat region detected for {record_name} in {domain_starts[i]}')
                    if guid_tracking is not None:
                        guid_tracking[guid_key].skipped_flat_region.append(float(domain_starts[i]))
                        if plot_segment_status is not None:
                            plot_segment_status[i] = 'flat_region'
                else:
                    valid_indices.append(i)

            if not valid_indices:
                logger.info(f'{guid_key}: all {fhr.shape[0]} segments filtered out, skipping record')
                # Generate strip plot for fully-rejected GUIDs
                if plot_segment_status is not None and hdf5_path:
                    from guid_analysis import plot_guid_signal_strip
                    hdf5_stem = os.path.splitext(os.path.basename(hdf5_path))[0]
                    strip_dir = os.path.join(
                        os.path.dirname(os.path.abspath(hdf5_path)),
                        f'{hdf5_stem}_guid_analysis', 'signal_strips')
                    plot_guid_signal_strip(
                        guid=guid_key, fhr=fhr, up=up,
                        domain_starts=domain_starts,
                        segment_status=plot_segment_status,
                        output_dir=strip_dir)
                continue

            # ---- Step 2: Batched scattering on valid segments ----
            valid_fhr = fhr[valid_indices]
            valid_up = up[valid_indices]
            st_input = torch.from_numpy(
                np.stack([valid_fhr, valid_up], axis=1)).float().to(device)

            n_valid = st_input.shape[0]
            st_phase_list = [None] * n_valid
            st_cross_list = [None] * n_valid
            st_up_phase_list = [None] * n_valid
            st_up_scatter_list = [None] * n_valid
            scatter_failed = set()

            for batch_start in range(0, n_valid, scatter_batch_size):
                batch_end = min(batch_start + scatter_batch_size, n_valid)
                batch = st_input[batch_start:batch_end]
                try:
                    batch_phase = st_model(
                        x=batch, compute_phase=True,
                        compute_cross_phase=False,
                        scattering_channel=0, phase_channels=[0])
                    batch_cross = st_model(
                        x=batch, compute_phase=False,
                        compute_cross_phase=True,
                        scattering_channel=0, phase_channels=[0, 1])
                    batch_up_phase = st_model(
                        x=batch, compute_phase=True,
                        compute_cross_phase=False,
                        scattering_channel=0, phase_channels=[1])
                    batch_up_scatter = st_model(
                        x=batch, compute_phase=False,
                        compute_cross_phase=False,
                        scattering_channel=1)
                    # Slice batch results into per-segment dicts.
                    # Only slice tensors whose first dim matches batch size
                    # (autoc_idx is a 1-D model buffer and must be kept as-is).
                    bs = batch.shape[0]
                    for local_j in range(bs):
                        gj = batch_start + local_j
                        st_phase_list[gj] = {
                            k: (v[local_j:local_j+1]
                                if isinstance(v, torch.Tensor) and v.shape[0] == bs
                                else v)
                            for k, v in batch_phase.items()}
                        st_cross_list[gj] = {
                            k: (v[local_j:local_j+1]
                                if isinstance(v, torch.Tensor) and v.shape[0] == bs
                                else v)
                            for k, v in batch_cross.items()}
                        st_up_phase_list[gj] = {
                            k: (v[local_j:local_j+1]
                                if isinstance(v, torch.Tensor) and v.shape[0] == bs
                                else v)
                            for k, v in batch_up_phase.items()}
                        st_up_scatter_list[gj] = {
                            k: (v[local_j:local_j+1]
                                if isinstance(v, torch.Tensor) and v.shape[0] == bs
                                else v)
                            for k, v in batch_up_scatter.items()}
                except RuntimeError:
                    # Batch failed — retry each segment individually
                    for local_j in range(batch.shape[0]):
                        gj = batch_start + local_j
                        seg = st_input[gj:gj+1]
                        try:
                            seg_phase = st_model(
                                x=seg, compute_phase=True,
                                compute_cross_phase=False,
                                scattering_channel=0, phase_channels=[0])
                            seg_cross = st_model(
                                x=seg, compute_phase=False,
                                compute_cross_phase=True,
                                scattering_channel=0, phase_channels=[0, 1])
                            seg_up_phase = st_model(
                                x=seg, compute_phase=True,
                                compute_cross_phase=False,
                                scattering_channel=0, phase_channels=[1])
                            seg_up_scatter = st_model(
                                x=seg, compute_phase=False,
                                compute_cross_phase=False,
                                scattering_channel=1)
                            st_phase_list[gj] = seg_phase
                            st_cross_list[gj] = seg_cross
                            st_up_phase_list[gj] = seg_up_phase
                            st_up_scatter_list[gj] = seg_up_scatter
                        except RuntimeError as seg_err:
                            orig_idx = valid_indices[gj]
                            logger.error(
                                f"{guid_key} segment {orig_idx} "
                                f"(epoch={domain_starts[orig_idx]}): "
                                f"scattering failed: {seg_err}")
                            scatter_failed.add(gj)
                            if guid_tracking is not None:
                                guid_tracking[guid_key].skipped_scatter_failed.append(
                                    float(domain_starts[orig_idx]))

            # ---- Step 3: Collect and batch-write valid scattered segments ----
            batch_fhr_l, batch_up_l = [], []
            batch_fhr_st_l, batch_fhr_ph_l, batch_fhr_up_ph_l, batch_up_st_l = [], [], [], []
            batch_target_l, batch_weight_l = [], []
            batch_guid_l, batch_epoch_l = [], []
            batch_cs_l, batch_bg_l, batch_tlo_l = [], [], []

            for seg_j in range(n_valid):
                if seg_j in scatter_failed:
                    continue
                orig_idx = valid_indices[seg_j]

                fhr_st_coeff = st_phase_list[seg_j]['scattering'][0]
                up_st_coeff = st_up_scatter_list[seg_j]['scattering'][0]
                fhr_st_phase_full = st_phase_list[seg_j]['phase_corr'][0]
                fhr_up_cc_phase_full = st_cross_list[seg_j]['cross_phase_corr'][0]

                fhr_ph_coeff = fhr_st_phase_full[phase_mask, :]
                cross_ph = fhr_up_cc_phase_full[cross_mask, :]

                # v3: UP self-phase
                up_self_phase_full = st_up_phase_list[seg_j]['phase_corr'][0]
                up_self_ph = up_self_phase_full[up_phase_mask, :]

                # Concatenate cross-phase + UP self-phase into fhr_up_ph
                fhr_up_ph_coeff = torch.cat([cross_ph, up_self_ph], dim=0)

                ds_val = float(domain_starts[orig_idx])
                if ds_val >= 0:
                    logger.warning(
                        f"{guid_key} segment {orig_idx}: post-delivery domain_start="
                        f"{ds_val:.1f}s (equalization padding or recording extends past delivery)")
                if guid_tracking is not None:
                    guid_tracking[guid_key].included_domain_starts.append(ds_val)
                # time_from_labor_onset = epoch - labor_onset (seconds since labor onset)
                tflo = float(domain_starts[orig_idx]) - labor_onset_sec
                print(f"    [TLO] seg {orig_idx}: epoch={domain_starts[orig_idx]:.0f}s - labor_onset={labor_onset_sec:.0f}s = tflo={tflo:.0f}s ({tflo/3600:.2f}h)")

                batch_fhr_l.append(fhr[orig_idx, :])
                batch_up_l.append(up[orig_idx, :])
                batch_fhr_st_l.append(fhr_st_coeff.detach().cpu().numpy())
                batch_up_st_l.append(up_st_coeff.detach().cpu().numpy())
                batch_fhr_ph_l.append(fhr_ph_coeff.detach().cpu().numpy())
                batch_fhr_up_ph_l.append(fhr_up_ph_coeff.detach().cpu().numpy())
                batch_target_l.append(pre_defined_target * sample_weights[orig_idx, :])
                batch_weight_l.append(sample_weights[orig_idx, :])
                batch_guid_l.append(record_name[0])
                batch_epoch_l.append(domain_starts[orig_idx])
                batch_cs_l.append(cs_label)
                batch_bg_l.append(bg_label)
                batch_tlo_l.append(tflo)

            if batch_fhr_l:
                append_samples_batch(
                    path=hdf5_path,
                    fhr_batch=np.stack(batch_fhr_l),
                    up_batch=np.stack(batch_up_l),
                    fhr_st_batch=np.stack(batch_fhr_st_l),
                    fhr_ph_batch=np.stack(batch_fhr_ph_l),
                    fhr_up_ph_batch=np.stack(batch_fhr_up_ph_l),
                    target_batch=np.stack(batch_target_l),
                    weight_batch=np.stack(batch_weight_l),
                    guid_batch=batch_guid_l,
                    epoch_batch=np.array(batch_epoch_l, dtype=np.float32),
                    cs_label_batch=np.array(batch_cs_l, dtype=np.uint8),
                    bg_label_batch=np.array(batch_bg_l, dtype=np.uint8),
                    tlo_batch=np.array(batch_tlo_l, dtype=np.float32),
                    up_st_batch=np.stack(batch_up_st_l))

            # ---- Generate signal strip plot ----
            if plot_segment_status is not None and hdf5_path:
                # Mark scatter_failed segments
                for seg_j in scatter_failed:
                    plot_segment_status[valid_indices[seg_j]] = 'scatter_failed'
                # Mark remaining valid (successfully saved) segments as included
                for seg_j in range(len(valid_indices)):
                    if seg_j not in scatter_failed:
                        plot_segment_status[valid_indices[seg_j]] = 'included'
                from guid_analysis import plot_guid_signal_strip
                hdf5_stem = os.path.splitext(os.path.basename(hdf5_path))[0]
                strip_dir = os.path.join(
                    os.path.dirname(os.path.abspath(hdf5_path)),
                    f'{hdf5_stem}_guid_analysis', 'signal_strips')
                plot_guid_signal_strip(
                    guid=guid_key, fhr=fhr, up=up,
                    domain_starts=domain_starts,
                    segment_status=plot_segment_status,
                    output_dir=strip_dir)

        except Exception as e:
            errors_list.append(record)
            logger.error(f"Failed processing {record}:\n{traceback.format_exc()}")
            if guid_tracking is not None:
                err_guid = os.path.splitext(os.path.split(record)[1])[0]
                guid_tracking[err_guid] = GuidTrackingEntry(
                    error=True, error_msg=str(e))

    if run_guid_analysis and hdf5_path and guid_tracking is not None:
        from guid_analysis import run_guid_analysis as _run_analysis
        segment_dur = signal_length / 4
        try:
            _run_analysis(hdf5_path, guid_tracking, segment_duration_sec=segment_dur)
        except Exception as e:
            logger.error(f"GUID analysis failed: {e}")

    # Log labor onset match summary
    if labor_onset_map and hdf5_path:
        try:
            import h5py as _h5
            with _h5.File(hdf5_path, 'r') as _f:
                if 'time_from_labor_onset' in _f:
                    tflo_arr = _f['time_from_labor_onset'][:]
                    n_total = len(tflo_arr)
                    n_nan = int(np.isnan(tflo_arr).sum())
                    n_valid = n_total - n_nan
                    logger.info(f"Labor onset summary for {os.path.basename(hdf5_path)}: "
                                f"{n_valid}/{n_total} segments matched "
                                f"({n_nan} NaN = GUID not in CSV)")
        except Exception as e:
            logger.warning(f"Could not read labor onset summary: {e}")

    return errors_list


def create_records(records_base_path_=None, output_base_path_=None, run_guid_analysis=False,
                   base_block_size=3520, overlap_percentage=1/11, labor_onset_csv_path=None,
                   classification_pickle_path=None, dataset_mode="both"):
    """Create HDF5 datasets for VAE pre-training and/or classification.

    Orchestrates the full pipeline: file discovery, GUID pre-screening,
    class balancing, fold creation, and HDF5 dataset generation.

    Modes of operation (controlled by ``dataset_mode``):
        - ``"pretrain"``: Only creates VAE pre-training HDF5 files
          (train/test, CS/NoCS) from the healthy_bg subgroups using a
          90/10 split. No GUID pre-screening is performed. Requires
          ``records_base_path_``.
        - ``"classification"``: Only creates classification HDF5 files.
          If ``classification_pickle_path`` is provided, loads pre-computed
          fold assignments; otherwise runs the full pipeline from scratch.
        - ``"both"`` (default): Creates the pre-training datasets first,
          then continues with the classification pipeline.

    Args:
        records_base_path_: Root directory containing the 16 StudyGroup
            subfolders. Required when ``dataset_mode`` includes pre-training
            or when ``classification_pickle_path`` is None.
        output_base_path_: Directory where HDF5 datasets and the pickle
            file are written.
        run_guid_analysis: If True, run per-GUID coverage analysis on the
            first fold/partition.
        base_block_size: Base signal block size in samples (default 3520).
        overlap_percentage: Overlap fraction between consecutive segments
            (default 1/11).
        labor_onset_csv_path: Path to the labor onset CSV. If None, no
            time-from-labor-onset is computed.
        classification_pickle_path: Path to an existing
            ``classification_dataset_records.pickle``. When provided,
            the fold structure is loaded from this file and file discovery,
            pre-screening, class balancing, and fold creation are skipped.
        dataset_mode: Controls which datasets to create. One of
            ``"pretrain"``, ``"classification"``, or ``"both"``
            (default ``"both"``).

    Raises:
        ValueError: If ``dataset_mode`` is not a valid mode, if
            ``records_base_path_`` is missing when required, or if the
            pickle has an unexpected structure.
        FileNotFoundError: If ``classification_pickle_path`` is provided
            but does not exist.
    """
    # Load labor onset data if CSV path provided
    labor_onset_map = None
    if labor_onset_csv_path is not None:
        labor_onset_map = load_labor_onset_data(labor_onset_csv_path)

    signal_length = int(base_block_size * 1.5)
    sequence_length = signal_length // 16

    # Compute scattering masks once for all datasets (v3)
    masks = compute_scattering_masks(signal_length, scattering_T=16)
    n_combined_cross = masks['n_combined_cross']
    total_channels = masks['n_phase'] + n_combined_cross
    logger.info(f"v3 channel layout: {masks['n_phase']} phase + "
                f"{masks['n_cross']} cross + {masks['n_up_phase']} UP self-phase = "
                f"{total_channels} total")

    # Validate dataset_mode
    valid_modes = ("pretrain", "classification", "both")
    if dataset_mode not in valid_modes:
        raise ValueError(
            f"dataset_mode must be one of {valid_modes}, got '{dataset_mode}'")

    # ---------------------------
    # Pre-training mode: create VAE datasets
    # ---------------------------
    if dataset_mode in ("pretrain", "both"):
        if records_base_path_ is None:
            raise ValueError(
                "records_base_path_ is required when dataset_mode "
                "is 'pretrain' or 'both'")

        healthy_bg_cs_path = os.path.join(
            records_base_path_, 'HEALTHY_NO_ACIDOSIS_CS', 'EFMOut')
        healthy_bg_no_cs_path = os.path.join(
            records_base_path_, 'HEALTHY_NO_ACIDOSIS_NoCS', 'EFMOut')

        healthy_bg_cs_files = [
            os.path.join(healthy_bg_cs_path, f)
            for f in os.listdir(healthy_bg_cs_path) if f.endswith('.mat')]
        healthy_bg_no_cs_files = [
            os.path.join(healthy_bg_no_cs_path, f)
            for f in os.listdir(healthy_bg_no_cs_path) if f.endswith('.mat')]

        # VAE 90/10 split
        n_healthy_bg_cs = len(healthy_bg_cs_files)
        n_healthy_bg_no_cs = len(healthy_bg_no_cs_files)

        random.shuffle(healthy_bg_cs_files)
        healthy_bg_cs_files_vae_train = healthy_bg_cs_files[:int(n_healthy_bg_cs * 0.9)]
        healthy_bg_cs_files_vae_test = healthy_bg_cs_files[int(n_healthy_bg_cs * 0.9):]

        random.shuffle(healthy_bg_no_cs_files)
        healthy_bg_no_cs_files_vae_train = healthy_bg_no_cs_files[:int(n_healthy_bg_no_cs * 0.9)]
        healthy_bg_no_cs_files_vae_test = healthy_bg_no_cs_files[int(n_healthy_bg_no_cs * 0.9):]

        logger.info(f"Pre-training only mode: "
                    f"healthy_bg_cs={n_healthy_bg_cs} (train={len(healthy_bg_cs_files_vae_train)}, "
                    f"test={len(healthy_bg_cs_files_vae_test)}), "
                    f"healthy_bg_no_cs={n_healthy_bg_no_cs} (train={len(healthy_bg_no_cs_files_vae_train)}, "
                    f"test={len(healthy_bg_no_cs_files_vae_test)})")

        # Create 4 VAE pre-training HDF5 files
        pre_train_path = os.path.join(output_base_path_, "pre_training_dataset")
        os.makedirs(pre_train_path, exist_ok=True)

        pre_training_dataset = os.path.join(pre_train_path, "train_dataset_cs.hdf5")
        create_initial_hdf5(path=pre_training_dataset, len_signal=signal_length, n_channels=total_channels, len_sequence=sequence_length, n_cross_phase_channels=n_combined_cross, n_up_st_channels=43)
        create_hdf5_dataset_from_records_list(
            records_list=healthy_bg_cs_files_vae_train,
            hdf5_path=pre_training_dataset,
            base_block_size=base_block_size,
            overlap_percentage=overlap_percentage,
            cs_label=True,
            bg_label=True,
            pre_defined_target=1,
            run_guid_analysis=run_guid_analysis,
            precomputed_masks=masks,
            labor_onset_map=labor_onset_map)

        pre_training_dataset = os.path.join(pre_train_path, "train_dataset_no_cs.hdf5")
        create_initial_hdf5(path=pre_training_dataset, len_signal=signal_length, n_channels=total_channels, len_sequence=sequence_length, n_cross_phase_channels=n_combined_cross, n_up_st_channels=43)
        create_hdf5_dataset_from_records_list(
            records_list=healthy_bg_no_cs_files_vae_train,
            hdf5_path=pre_training_dataset,
            base_block_size=base_block_size,
            overlap_percentage=overlap_percentage,
            cs_label=False,
            bg_label=True,
            pre_defined_target=1,
            run_guid_analysis=run_guid_analysis,
            precomputed_masks=masks,
            labor_onset_map=labor_onset_map)

        pre_training_dataset = os.path.join(pre_train_path, "test_dataset_cs.hdf5")
        create_initial_hdf5(path=pre_training_dataset, len_signal=signal_length, n_channels=total_channels, len_sequence=sequence_length, n_cross_phase_channels=n_combined_cross, n_up_st_channels=43)
        create_hdf5_dataset_from_records_list(
            records_list=healthy_bg_cs_files_vae_test,
            hdf5_path=pre_training_dataset,
            base_block_size=base_block_size,
            overlap_percentage=overlap_percentage,
            cs_label=True,
            bg_label=True,
            pre_defined_target=1,
            run_guid_analysis=run_guid_analysis,
            precomputed_masks=masks,
            labor_onset_map=labor_onset_map)

        pre_training_dataset = os.path.join(pre_train_path, "test_dataset_no_cs.hdf5")
        create_initial_hdf5(path=pre_training_dataset, len_signal=signal_length, n_channels=total_channels, len_sequence=sequence_length, n_cross_phase_channels=n_combined_cross, n_up_st_channels=43)
        create_hdf5_dataset_from_records_list(
            records_list=healthy_bg_no_cs_files_vae_test,
            hdf5_path=pre_training_dataset,
            base_block_size=base_block_size,
            overlap_percentage=overlap_percentage,
            cs_label=False,
            bg_label=True,
            pre_defined_target=1,
            run_guid_analysis=run_guid_analysis,
            precomputed_masks=masks,
            labor_onset_map=labor_onset_map)

        logger.info(f"Pre-training datasets created in {pre_train_path}")
        if dataset_mode == "pretrain":
            return []

    if dataset_mode not in ("classification", "both"):
        return []

    if classification_pickle_path is not None:
        # ---------------------------
        # REUSE MODE: Load pre-computed classification folds
        # ---------------------------
        if not os.path.isfile(classification_pickle_path):
            raise FileNotFoundError(
                f"Classification pickle not found: {classification_pickle_path}")
        logger.info(f"Loading pre-computed classification folds from: "
                    f"{classification_pickle_path}")
        with open(classification_pickle_path, 'rb') as f:
            classification_folds = pickle.load(f)
        logger.info(f"Loaded {len(classification_folds)} folds: "
                    f"{list(classification_folds.keys())}")

        # Validate pickle structure
        expected_subgroups = {
            'healthy_no_bg_no_cs', 'healthy_no_bg_cs', 'healthy_bg_cs',
            'healthy_bg_no_cs', 'acidosis_cs', 'acidosis_no_cs',
            'hie_cs', 'hie_no_cs',
        }
        first_fold = next(iter(classification_folds.values()))
        first_partition = next(iter(first_fold.values()))
        actual_subgroups = set(first_partition.keys())
        missing = expected_subgroups - actual_subgroups
        if missing:
            raise ValueError(
                f"Pickle is missing expected subgroups: {missing}. "
                f"Found: {actual_subgroups}")

        # Log per-fold statistics for traceability
        for fold_name, partitions in classification_folds.items():
            for part_name, subgroups in partitions.items():
                total = sum(len(v) for v in subgroups.values())
                logger.info(f"  {fold_name}/{part_name}: {total} files "
                            f"across {len(subgroups)} subgroups")
    else:
        # ---------------------------
        # FULL PIPELINE: file discovery through fold creation
        # ---------------------------
        if records_base_path_ is None:
            raise ValueError(
                "records_base_path_ is required when "
                "classification_pickle_path is not provided")

        list_of_folders_dict = {
            1: "ACIDOSIS_NO_HIE_CS",
            2: "ACIDOSIS_NO_HIE_NoCS",
            3: "DEATH_lt_6_CS",
            4: "DEATH_lt_6_NoCS",
            5: "DISTANT_HIE_CS",
            6: "DISTANT_HIE_NoCS",
            7: "HEALTHY_NO_ACIDOSIS_CS",
            8: "HEALTHY_NO_ACIDOSIS_NoCS",
            9: "HEALTHY_NO_BG_CS",
            10: "HEALTHY_NO_BG_NoCS",
            11: "HIE_CS",
            12: "HIE_NoCS",
            13: "INTERVENTION_NO_ACIDOSIS_CS",
            14: "INTERVENTION_NO_ACIDOSIS_NoCS",
            15: "INTERVENTION_NO_BG_CS",
            16: "INTERVENTION_NO_BG_NoCS",
        }

        healthy_no_bg_no_cs_path = os.path.join(records_base_path_, list_of_folders_dict[10], 'EFMOut')
        healthy_no_bg_cs_path = os.path.join(records_base_path_, list_of_folders_dict[9], 'EFMOut')
        healthy_bg_cs_path = os.path.join(records_base_path_, list_of_folders_dict[7], 'EFMOut')
        healthy_bg_no_cs_path = os.path.join(records_base_path_, list_of_folders_dict[8], 'EFMOut')

        acidosis_cs_path = os.path.join(records_base_path_, list_of_folders_dict[1], 'EFMOut')
        acidosis_no_cs_path = os.path.join(records_base_path_, list_of_folders_dict[2], 'EFMOut')

        hie_cs_path =  os.path.join(records_base_path_, list_of_folders_dict[11], 'EFMOut')
        hie_no_cs_path = os.path.join(records_base_path_, list_of_folders_dict[12], 'EFMOut')

        healthy_no_bg_no_cs_files = [os.path.join(healthy_no_bg_no_cs_path, f) for f in os.listdir(healthy_no_bg_no_cs_path) if f.endswith('.mat')]
        healthy_no_bg_cs_files = [os.path.join(healthy_no_bg_cs_path, f) for f in os.listdir(healthy_no_bg_cs_path) if f.endswith('.mat')]
        healthy_bg_cs_files = [os.path.join(healthy_bg_cs_path, f) for f in os.listdir(healthy_bg_cs_path) if f.endswith('.mat')]
        healthy_bg_no_cs_files = [os.path.join(healthy_bg_no_cs_path, f) for f in os.listdir(healthy_bg_no_cs_path) if f.endswith('.mat')]

        acidosis_cs_files = [os.path.join(acidosis_cs_path, f) for f in os.listdir(acidosis_cs_path) if f.endswith('.mat')]
        acidosis_no_cs_files = [os.path.join(acidosis_no_cs_path, f) for f in os.listdir(acidosis_no_cs_path) if f.endswith('.mat')]

        hie_cs_files = [os.path.join(hie_cs_path, f) for f in os.listdir(hie_cs_path) if f.endswith('.mat')]
        hie_no_cs_files = [os.path.join(hie_no_cs_path, f) for f in os.listdir(hie_no_cs_path) if f.endswith('.mat')]

        # ---------------------------
        # VAE 90/10 split (BEFORE pre-screening — VAE data doesn't need TLO/min hours)
        # ---------------------------
        n_healthy_bg_cs = len(healthy_bg_cs_files)
        n_healthy_bg_no_cs = len(healthy_bg_no_cs_files)

        random.shuffle(healthy_bg_cs_files)
        healthy_bg_cs_files_vae_train = healthy_bg_cs_files[:int(n_healthy_bg_cs * 0.9)]
        healthy_bg_cs_files_vae_test = healthy_bg_cs_files[int(n_healthy_bg_cs * 0.9):]

        random.shuffle(healthy_bg_no_cs_files)
        healthy_bg_no_cs_files_vae_train = healthy_bg_no_cs_files[:int(n_healthy_bg_no_cs * 0.9)]
        healthy_bg_no_cs_files_vae_test = healthy_bg_no_cs_files[int(n_healthy_bg_no_cs * 0.9):]

        # ---------------------------
        # GUID pre-screening for classification candidates
        # ---------------------------
        # Build candidate file lists:
        #   - unhealthy: all acidosis + hie files
        #   - healthy: no_bg files (full) + bg files (ALL, including VAE train)
        prescreening_candidates = {
            'acidosis_cs': acidosis_cs_files,
            'acidosis_no_cs': acidosis_no_cs_files,
            'hie_cs': hie_cs_files,
            'hie_no_cs': hie_no_cs_files,
            'healthy_no_bg_no_cs': healthy_no_bg_no_cs_files,
            'healthy_no_bg_cs': healthy_no_bg_cs_files,
            'healthy_bg_cs': healthy_bg_cs_files,
            'healthy_bg_no_cs': healthy_bg_no_cs_files,
        }

        filtered = prescreen_guids_for_classification(
            prescreening_candidates,
            labor_onset_map=labor_onset_map,
            base_block_size=base_block_size,
            overlap_percentage=overlap_percentage,
            output_dir=output_base_path_)

        # Update file lists from pre-screening results
        acidosis_cs_files = filtered['acidosis_cs']
        acidosis_no_cs_files = filtered['acidosis_no_cs']
        hie_cs_files = filtered['hie_cs']
        hie_no_cs_files = filtered['hie_no_cs']
        healthy_no_bg_no_cs_files = filtered['healthy_no_bg_no_cs']
        healthy_no_bg_cs_files = filtered['healthy_no_bg_cs']
        healthy_bg_cs_files = filtered['healthy_bg_cs']
        healthy_bg_no_cs_files = filtered['healthy_bg_no_cs']

        # ---------------------------
        # Recount after pre-screening and class balance
        # ---------------------------
        n_acidosis_cs = len(acidosis_cs_files)
        n_acidosis_no_cs = len(acidosis_no_cs_files)
        n_acidosis_total = n_acidosis_cs + n_acidosis_no_cs

        n_hie_cs = len(hie_cs_files)
        n_hie_no_cs = len(hie_no_cs_files)
        n_hie_total = n_hie_cs + n_hie_no_cs

        n_unhealthy_total = n_acidosis_total + n_hie_total

        # Available pools after pre-screening (BG uses ALL files)
        file_pools = {
            "NoBG_NoCS": healthy_no_bg_no_cs_files,
            "NoBG_CS": healthy_no_bg_cs_files,
            "BG_CS": healthy_bg_cs_files,
            "BG_NoCS": healthy_bg_no_cs_files,
        }
        counts_available = {k: len(v) for k, v in file_pools.items()}
        total_healthy_available = sum(counts_available.values())

        # Use original (pre-split) population sizes to compute proportions
        # so that BG subgroups get their fair share, not the deflated 10%
        population_counts = {
            "NoBG_NoCS": len(healthy_no_bg_no_cs_files),
            "NoBG_CS": len(healthy_no_bg_cs_files),
            "BG_CS": n_healthy_bg_cs,
            "BG_NoCS": n_healthy_bg_no_cs,
        }
        total_population = sum(population_counts.values())

        healthy_to_unhealthy_ratio = 5
        n_target_desired = n_unhealthy_total * healthy_to_unhealthy_ratio
        if total_healthy_available < n_target_desired:
            logger.warning(
                f"Filtered healthy GUIDs ({total_healthy_available}) < "
                f"{healthy_to_unhealthy_ratio}x unhealthy "
                f"({n_target_desired}). Capping healthy target to available count.")
            n_target = total_healthy_available
        else:
            n_target = n_target_desired

        # Allocate proportionally based on original population sizes
        target_healthy = {
            k: int(round((population_counts[k] / total_population) * n_target))
            for k in population_counts
        }

        # Fix rounding residuals
        diff = n_target - sum(target_healthy.values())
        if diff:
            largest = max(population_counts, key=population_counts.get)
            target_healthy[largest] += diff

        # Cap targets to available counts per subgroup, redistribute overflow
        overflow = 0
        for k in target_healthy:
            available = counts_available[k]
            if target_healthy[k] > available:
                overflow += target_healthy[k] - available
                target_healthy[k] = available
        # Redistribute overflow to subgroups with remaining capacity
        if overflow > 0:
            for k in sorted(target_healthy, key=lambda x: counts_available[x] - target_healthy[x], reverse=True):
                room = counts_available[k] - target_healthy[k]
                add = min(overflow, room)
                target_healthy[k] += add
                overflow -= add
                if overflow == 0:
                    break
            if overflow > 0:
                logger.warning(f"Could not redistribute {overflow} healthy GUIDs — "
                               f"all subgroups at capacity")

        logger.info(f"Class balancing (post pre-screening): "
                    f"unhealthy={n_unhealthy_total}, "
                    f"healthy target={sum(target_healthy.values())} "
                    f"({target_healthy}), "
                    f"population proportions={population_counts}")

        healthy_no_bg_no_cs_files_subsampled = random.sample(healthy_no_bg_no_cs_files, target_healthy['NoBG_NoCS'])
        healthy_no_bg_cs_files_subsampled = random.sample(healthy_no_bg_cs_files, target_healthy['NoBG_CS'])
        healthy_bg_cs_files_subsampled = random.sample(healthy_bg_cs_files, target_healthy['BG_CS'])
        healthy_bg_no_cs_files_subsampled = random.sample(healthy_bg_no_cs_files, target_healthy['BG_NoCS'])

        cross_validation_records = {
            'healthy_no_bg_no_cs': healthy_no_bg_no_cs_files_subsampled,
            'healthy_no_bg_cs':    healthy_no_bg_cs_files_subsampled,
            'healthy_bg_cs':       healthy_bg_cs_files_subsampled,
            'healthy_bg_no_cs':    healthy_bg_no_cs_files_subsampled,
            'acidosis_cs':         acidosis_cs_files,
            'acidosis_no_cs':      acidosis_no_cs_files,
            'hie_cs':              hie_cs_files,
            'hie_no_cs':           hie_no_cs_files,
        }
        classification_folds = create_cv_splits(cross_validation_records, n_splits=10, val_ratio=0.1, random_state=42)
        classification_dataset_records_path = os.path.join(output_base_path_, "classification_dataset_records.pickle")
        with open(classification_dataset_records_path, 'wb') as f:
            pickle.dump(classification_folds, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ---------------------------
    # Vae Train and Test
    # ---------------------------
    # pre_train_path = os.path.join(output_base_path_, "pre_training_dataset")
    # os.makedirs(pre_train_path, exist_ok=True)
    # pre_training_dataset = os.path.join(pre_train_path, "train_dataset_cs.hdf5")
    # create_initial_hdf5(path=pre_training_dataset, len_signal=signal_length, n_channels=total_channels, len_sequence=sequence_length, n_cross_phase_channels=n_combined_cross, n_up_st_channels=43)
    # create_hdf5_dataset_from_records_list(
    #     records_list=healthy_bg_cs_files_vae_train,
    #     hdf5_path=pre_training_dataset,
    #     base_block_size=base_block_size,
    #     overlap_percentage=overlap_percentage,
    #     cs_label=True,
    #     bg_label=True,
    #     pre_defined_target=1,
    #     run_guid_analysis=run_guid_analysis,
    #     precomputed_masks=masks,
    #     labor_onset_map=labor_onset_map)

    # pre_training_dataset = os.path.join(pre_train_path, "train_dataset_no_cs.hdf5")
    # create_initial_hdf5(path=pre_training_dataset, len_signal=signal_length, n_channels=total_channels, len_sequence=sequence_length, n_cross_phase_channels=n_combined_cross, n_up_st_channels=43)
    # create_hdf5_dataset_from_records_list(records_list=healthy_bg_no_cs_files_vae_train,
    #                                       hdf5_path=pre_training_dataset,
    #                                       base_block_size=base_block_size,
    #                                       overlap_percentage=overlap_percentage,
    #                                       cs_label=False,
    #                                       bg_label=True,
    #                                       pre_defined_target=1,
    #                                       run_guid_analysis=run_guid_analysis,
    #     precomputed_masks=masks,
    #     labor_onset_map=labor_onset_map)


    # pre_training_dataset = os.path.join(pre_train_path, "test_dataset_cs.hdf5")
    # create_initial_hdf5(path=pre_training_dataset, len_signal=signal_length, n_channels=total_channels, len_sequence=sequence_length, n_cross_phase_channels=n_combined_cross, n_up_st_channels=43)
    # create_hdf5_dataset_from_records_list(records_list=healthy_bg_cs_files_vae_test,
    #                                       hdf5_path=pre_training_dataset,
    #                                       base_block_size=base_block_size,
    #                                       overlap_percentage=overlap_percentage,
    #                                       cs_label=True,
    #                                       bg_label=True,
    #                                       pre_defined_target=1,
    #                                       run_guid_analysis=run_guid_analysis,
    #     precomputed_masks=masks,
    #     labor_onset_map=labor_onset_map)

    # pre_training_dataset = os.path.join(pre_train_path, "test_dataset_no_cs.hdf5")
    # create_initial_hdf5(path=pre_training_dataset, len_signal=signal_length, n_channels=total_channels, len_sequence=sequence_length, n_cross_phase_channels=n_combined_cross, n_up_st_channels=43)
    # create_hdf5_dataset_from_records_list(records_list=healthy_bg_no_cs_files_vae_test,
    #                                       hdf5_path=pre_training_dataset,
    #                                       base_block_size=base_block_size,
    #                                       overlap_percentage=overlap_percentage,
    #                                       cs_label=False,
    #                                       bg_label=True,
    #                                       pre_defined_target=1,
    #                                       run_guid_analysis=run_guid_analysis,
    #     precomputed_masks=masks,
    #     labor_onset_map=labor_onset_map)
    # ---------------------------
    # Classifications
    # ---------------------------
    k_fold_cross_validation_path = os.path.join(output_base_path_, "k_fold_cross_validation_dataset")
    os.makedirs(k_fold_cross_validation_path, exist_ok=True)
    run_guid_analysis = False
    for fold in classification_folds:
        print('done')
        fold_path = os.path.join(k_fold_cross_validation_path, str(fold))
        os.makedirs(fold_path, exist_ok=True)
        fold_datasets = classification_folds.get(fold)
        for dataset_partition in fold_datasets:
            dataset_partition_path = os.path.join(fold_path, str(dataset_partition))
            os.makedirs(dataset_partition_path, exist_ok=True)
            sub_groups_list = fold_datasets.get(dataset_partition)

            selected_sub_group = "healthy_no_bg_no_cs"
            sub_group_path = os.path.join(dataset_partition_path, f"{selected_sub_group}.hdf5")
            sub_group_records_list = sub_groups_list.get(selected_sub_group)
            create_initial_hdf5(path=sub_group_path, len_signal=signal_length, n_channels=total_channels, len_sequence=sequence_length, n_cross_phase_channels=n_combined_cross, n_up_st_channels=43)
            create_hdf5_dataset_from_records_list(records_list=sub_group_records_list,
                                                  hdf5_path=sub_group_path,
                                                  base_block_size=base_block_size,
                                                  overlap_percentage=overlap_percentage,
                                                  cs_label=False,
                                                  bg_label=False,
                                                  pre_defined_target=1,
                                                  run_guid_analysis=run_guid_analysis,
        precomputed_masks=masks,
        labor_onset_map=labor_onset_map)

            selected_sub_group = "healthy_no_bg_cs"
            sub_group_path = os.path.join(dataset_partition_path, f"{selected_sub_group}.hdf5")
            sub_group_records_list = sub_groups_list.get(selected_sub_group)
            create_initial_hdf5(path=sub_group_path, len_signal=signal_length, n_channels=total_channels, len_sequence=sequence_length, n_cross_phase_channels=n_combined_cross, n_up_st_channels=43)
            create_hdf5_dataset_from_records_list(records_list=sub_group_records_list,
                                                  hdf5_path=sub_group_path,
                                                  base_block_size=base_block_size,
                                                  overlap_percentage=overlap_percentage,
                                                  cs_label=True,
                                                  bg_label=False,
                                                  pre_defined_target=1,
                                                  run_guid_analysis=run_guid_analysis,
        precomputed_masks=masks,
        labor_onset_map=labor_onset_map)

            selected_sub_group = "healthy_bg_cs"
            sub_group_path = os.path.join(dataset_partition_path, f"{selected_sub_group}.hdf5")
            sub_group_records_list = sub_groups_list.get(selected_sub_group)
            create_initial_hdf5(path=sub_group_path, len_signal=signal_length, n_channels=total_channels, len_sequence=sequence_length, n_cross_phase_channels=n_combined_cross, n_up_st_channels=43)
            create_hdf5_dataset_from_records_list(records_list=sub_group_records_list,
                                                  hdf5_path=sub_group_path,
                                                  base_block_size=base_block_size,
                                                  overlap_percentage=overlap_percentage,
                                                  cs_label=True,
                                                  bg_label=True,
                                                  pre_defined_target=1,
                                                  run_guid_analysis=run_guid_analysis,
        precomputed_masks=masks,
        labor_onset_map=labor_onset_map)

            selected_sub_group = "healthy_bg_no_cs"
            sub_group_path = os.path.join(dataset_partition_path, f"{selected_sub_group}.hdf5")
            sub_group_records_list = sub_groups_list.get(selected_sub_group)
            create_initial_hdf5(path=sub_group_path, len_signal=signal_length, n_channels=total_channels, len_sequence=sequence_length, n_cross_phase_channels=n_combined_cross, n_up_st_channels=43)
            create_hdf5_dataset_from_records_list(records_list=sub_group_records_list,
                                                  hdf5_path=sub_group_path,
                                                  base_block_size=base_block_size,
                                                  overlap_percentage=overlap_percentage,
                                                  cs_label=False,
                                                  bg_label=True,
                                                  pre_defined_target=1,
                                                  run_guid_analysis=run_guid_analysis,
        precomputed_masks=masks,
        labor_onset_map=labor_onset_map)

            selected_sub_group = "acidosis_cs"
            sub_group_path = os.path.join(dataset_partition_path, f"{selected_sub_group}.hdf5")
            sub_group_records_list = sub_groups_list.get(selected_sub_group)
            create_initial_hdf5(path=sub_group_path, len_signal=signal_length, n_channels=total_channels, len_sequence=sequence_length, n_cross_phase_channels=n_combined_cross, n_up_st_channels=43)
            create_hdf5_dataset_from_records_list(records_list=sub_group_records_list,
                                                  hdf5_path=sub_group_path,
                                                  base_block_size=base_block_size,
                                                  overlap_percentage=overlap_percentage,
                                                  cs_label=True,
                                                  bg_label=True,
                                                  pre_defined_target=2,
                                                  run_guid_analysis=run_guid_analysis,
        precomputed_masks=masks,
        labor_onset_map=labor_onset_map)

            selected_sub_group = "acidosis_no_cs"
            sub_group_path = os.path.join(dataset_partition_path, f"{selected_sub_group}.hdf5")
            sub_group_records_list = sub_groups_list.get(selected_sub_group)
            create_initial_hdf5(path=sub_group_path, len_signal=signal_length, n_channels=total_channels, len_sequence=sequence_length, n_cross_phase_channels=n_combined_cross, n_up_st_channels=43)
            create_hdf5_dataset_from_records_list(records_list=sub_group_records_list,
                                                  hdf5_path=sub_group_path,
                                                  base_block_size=base_block_size,
                                                  overlap_percentage=overlap_percentage,
                                                  cs_label=False,
                                                  bg_label=True,
                                                  pre_defined_target=2,
                                                  run_guid_analysis=run_guid_analysis,
        precomputed_masks=masks,
        labor_onset_map=labor_onset_map)

            selected_sub_group = "hie_cs"
            sub_group_path = os.path.join(dataset_partition_path, f"{selected_sub_group}.hdf5")
            sub_group_records_list = sub_groups_list.get(selected_sub_group)
            create_initial_hdf5(path=sub_group_path, len_signal=signal_length, n_channels=total_channels, len_sequence=sequence_length, n_cross_phase_channels=n_combined_cross, n_up_st_channels=43)
            create_hdf5_dataset_from_records_list(records_list=sub_group_records_list,
                                                  hdf5_path=sub_group_path,
                                                  base_block_size=base_block_size,
                                                  overlap_percentage=overlap_percentage,
                                                  cs_label=True,
                                                  bg_label=True,
                                                  pre_defined_target=3,
                                                  run_guid_analysis=run_guid_analysis,
        precomputed_masks=masks,
        labor_onset_map=labor_onset_map)

            selected_sub_group = "hie_no_cs"
            sub_group_path = os.path.join(dataset_partition_path, f"{selected_sub_group}.hdf5")
            sub_group_records_list = sub_groups_list.get(selected_sub_group)
            create_initial_hdf5(path=sub_group_path, len_signal=signal_length, n_channels=total_channels, len_sequence=sequence_length, n_cross_phase_channels=n_combined_cross, n_up_st_channels=43)
            create_hdf5_dataset_from_records_list(records_list=sub_group_records_list,
                                                  hdf5_path=sub_group_path,
                                                  base_block_size=base_block_size,
                                                  overlap_percentage=overlap_percentage,
                                                  cs_label=False,
                                                  bg_label=True,
                                                  pre_defined_target=3,
                                                  run_guid_analysis=run_guid_analysis,
        precomputed_masks=masks,
        labor_onset_map=labor_onset_map)
        run_guid_analysis = False


if __name__ == "__main__":
    base_folder = r'/data/deid/datafabric/fetal-heart-tracing/StudyGroup2022_v4/'
    base_output_folder = r'/data1/fetal-heart-tracing/HDF5_Datasets/last_12_hours'

    # Set to an existing pickle path to skip file discovery/pre-screening/fold creation.
    # Set to None to run the full pipeline.
    classification_pickle = None  # e.g. r'/path/to/classification_dataset_records.pickle'

    # Dataset creation mode: "pretrain", "classification", or "both".
    dataset_mode = "both"

    base_block_size = 3520
    overlap_percentage = 1 / 11
    labor_onset_csv = None

    create_records(
        records_base_path_=base_folder,
        output_base_path_=base_output_folder,
        base_block_size=base_block_size,
        overlap_percentage=overlap_percentage,
        labor_onset_csv_path=labor_onset_csv,
        classification_pickle_path=classification_pickle,
        dataset_mode=dataset_mode,
    )

    # hdf_file = "test_dataset_no_cs.hdf5"
    
    # try:
    #     print("Opening HDF5 dataset...")
    #     with h5py.File(hdf_file, "r") as dataset:
            
    #         # Print dataset structure
    #         print("\n" + "="*60)
    #         print("DATASET STRUCTURE")
    #         print("="*60)
    #         print("Available fields in dataset:")
    #         for key in dataset.keys():
    #             shape = dataset[key].shape
    #             dtype = dataset[key].dtype
    #             print(f"  {key}: shape={shape}, dtype={dtype}")
            
    #         # Check if dataset has samples
    #         if len(dataset.keys()) == 0:
    #             print("Dataset is empty!")
    #         else:
    #             # Get the number of samples (assuming all fields have same first dimension)
    #             first_key = list(dataset.keys())[0]
    #             n_samples = dataset[first_key].shape[0]
    #             print(f"\nTotal number of samples: {n_samples}")
                
    #             if n_samples > 0:
    #                 # Example: Get the first sample (index 0)
    #                 sample_idx = 100
    #                 print(f"\n" + "="*60)
    #                 print(f"SAMPLE {sample_idx} DETAILS")
    #                 print("="*60)
                    
    #                 sample_data = {}
    #                 for field in dataset.keys():
    #                     sample_data[field] = dataset[field][sample_idx]
                    
    #                 # Display sample information
    #                 for field, data in sample_data.items():
    #                     if isinstance(data, np.ndarray):
    #                         print(f"{field}:")
    #                         print(f"  Shape: {data.shape}")
    #                         print(f"  Dtype: {data.dtype}")
    #                         if data.size > 0:
    #                             if data.ndim == 1:
    #                                 # 1D array - show basic stats
    #                                 print(f"  Min: {np.min(data):.4f}, Max: {np.max(data):.4f}")
    #                                 print(f"  Mean: {np.mean(data):.4f}, Std: {np.std(data):.4f}")
    #                                 print(f"  First 5 values: {data[:5]}")
    #                             elif data.ndim == 2:
    #                                 # 2D array - show shape and channel stats
    #                                 print(f"  Channels: {data.shape[0]}, Sequence length: {data.shape[1]}")
    #                                 print(f"  Min: {np.min(data):.4f}, Max: {np.max(data):.4f}")
    #                                 print(f"  Mean: {np.mean(data):.4f}, Std: {np.std(data):.4f}")
    #                                 # Show stats for first few channels
    #                                 for i in range(min(3, data.shape[0])):
    #                                     ch_data = data[i, :]
    #                                     print(f"    Ch {i}: mean={np.mean(ch_data):.4f}, std={np.std(ch_data):.4f}")
    #                                 if data.shape[0] > 3:
    #                                     print(f"    ... ({data.shape[0] - 3} more channels)")
    #                     else:
    #                         # Scalar values
    #                         print(f"{field}: {data}")
    #                     print()
                    
    #                 # Example: How to use the sample data
    #                 print("="*60)
    #                 print("EXAMPLE USAGE")  
    #                 print("="*60)
    #                 print("\n# Example code for using the loaded sample:")
    #                 print("# Access specific fields:")
    #                 if 'fhr' in sample_data:
    #                     print(f"# fhr_signal = sample_data['fhr']  # Shape: {sample_data['fhr'].shape}")
    #                 if 'up' in sample_data:
    #                     print(f"# up_signal = sample_data['up']    # Shape: {sample_data['up'].shape}")
    #                 if 'fhr_st' in sample_data:
    #                     print(f"# fhr_st = sample_data['fhr_st']   # Shape: {sample_data['fhr_st'].shape}")
    #                 if 'fhr_ph' in sample_data:
    #                     print(f"# fhr_ph = sample_data['fhr_ph']   # Shape: {sample_data['fhr_ph'].shape}")
    #                 if 'fhr_up_ph' in sample_data:
    #                     print(f"# fhr_up_ph = sample_data['fhr_up_ph'] # Shape: {sample_data['fhr_up_ph'].shape}")
                    
    #                 print("\n# Convert to torch tensors for deep learning:")
    #                 print("# import torch")
    #                 if 'fhr' in sample_data and 'up' in sample_data:
    #                     print("# fhr_tensor = torch.from_numpy(sample_data['fhr']).float()")
    #                     print("# up_tensor = torch.from_numpy(sample_data['up']).float()")
    #                 if 'fhr_st' in sample_data:
    #                     print("# fhr_st_tensor = torch.from_numpy(sample_data['fhr_st']).float()")
                    
    #                 print("\n# Example batch processing (multiple samples):")
    #                 batch_size = min(4, n_samples)
    #                 print(f"# batch_fhr = dataset['fhr'][:{batch_size}]  # Shape: {dataset['fhr'][:batch_size].shape}")
    #                 if 'fhr_st' in dataset:
    #                     print(f"# batch_fhr_st = dataset['fhr_st'][:{batch_size}]  # Shape: {dataset['fhr_st'][:batch_size].shape}")
                    
    #                 # Show another sample if available
    #                 if n_samples > 1:
    #                     sample_idx = min(1, n_samples - 1)
    #                     print(f"\n" + "="*60)
    #                     print(f"QUICK VIEW: SAMPLE {sample_idx}")
    #                     print("="*60)
    #                     for field in ['guid', 'epoch', 'target', 'cs_label', 'bg_label']:
    #                         if field in dataset:
    #                             value = dataset[field][sample_idx]
    #                             print(f"{field}: {value}")
    #             else:
    #                 print("\nDataset contains no samples!")
                    
    #     print(f"\nSuccessfully examined HDF5 dataset: {hdf_file}")
        
    # except FileNotFoundError:
    #     print(f"HDF5 file not found: {hdf_file}")
    #     print("Please make sure the file exists or create it first using the dataset creation functions.")
    # except Exception as e:
    #     print(f"Error reading HDF5 dataset: {e}")
    
    # print('\nDone!')