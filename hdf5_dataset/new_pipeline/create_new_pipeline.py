"""New HDF5 dataset creation pipeline for CTG classification and VAE pretraining.

Self-contained script that:
  1. Prescreens all GUIDs for valid signal in the last 6 hours.
  2. Selects GUIDs for classification with class balancing + TLO constraints.
  3. Creates 10-fold stratified CV splits (80/10/10).
  4. Builds classification HDF5 datasets (12.4h range, v3 scattering).
  5. Builds pretraining HDF5 datasets from BG subgroup leftovers.
"""

import os
import sys
import math
import json
import pickle
import random
import logging
import traceback
from collections import defaultdict
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from early_maestra.adaptor.mimo_adaptor import EarlyMaestraMimoAdaptor
from Variational_AutoEncoder.seqvae_teb.hdf5_dataset.kymatio_phase_scattering import (
    KymatioPhaseScattering1D,
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
matplotlib.use("Agg")
torch.backends.cudnn.enabled = False

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Mappings
# ---------------------------------------------------------------------------
FOLDER_TO_SUBGROUP = {
    "ACIDOSIS_NO_HIE_CS": "acidosis_cs",
    "ACIDOSIS_NO_HIE_NoCS": "acidosis_no_cs",
    "HEALTHY_NO_ACIDOSIS_CS": "healthy_bg_cs",
    "HEALTHY_NO_ACIDOSIS_NoCS": "healthy_bg_no_cs",
    "HEALTHY_NO_BG_CS": "healthy_no_bg_cs",
    "HEALTHY_NO_BG_NoCS": "healthy_no_bg_no_cs",
    "HIE_CS": "hie_cs",
    "HIE_NoCS": "hie_no_cs",
}

CSV_NAME_TO_SUBGROUP = {
    "HIE_NoCS": "hie_no_cs",
    "HIE_CS": "hie_cs",
    "ACIDOSIS_NO_HIE_CS": "acidosis_cs",
    "ACIDOSIS_NO_HIE_NoCS": "acidosis_no_cs",
    "HEALTHY_NO_ACIDOSIS_CS": "healthy_bg_cs",
    "HEALTHY_NO_ACIDOSIS_NoCS": "healthy_bg_no_cs",
    "HEALTHY_NO_BG_CS": "healthy_no_bg_cs",
    "HEALTHY_NO_BG_NoCS": "healthy_no_bg_no_cs",
}

SUBGROUP_TO_FOLDER = {v: k for k, v in FOLDER_TO_SUBGROUP.items()}

UNHEALTHY_SUBGROUPS = {"acidosis_cs", "acidosis_no_cs", "hie_cs", "hie_no_cs"}
HEALTHY_SUBGROUPS = {
    "healthy_bg_cs",
    "healthy_bg_no_cs",
    "healthy_no_bg_cs",
    "healthy_no_bg_no_cs",
}
BG_SUBGROUPS = {"healthy_bg_cs", "healthy_bg_no_cs"}

# (pre_defined_target, cs_label, bg_label)
SUBGROUP_META = {
    "healthy_no_bg_no_cs": (1, False, False),
    "healthy_no_bg_cs": (1, True, False),
    "healthy_bg_cs": (1, True, True),
    "healthy_bg_no_cs": (1, False, True),
    "acidosis_cs": (2, True, True),
    "acidosis_no_cs": (2, False, True),
    "hie_cs": (3, True, True),
    "hie_no_cs": (3, False, True),
}

BASE_BLOCK_SIZE = 3520
OVERLAP_PERCENTAGE = 1 / 11
SIGNAL_LENGTH = int(BASE_BLOCK_SIZE * 1.5)  # 5280
SEQUENCE_LENGTH = SIGNAL_LENGTH // 16  # 330
SEGMENT_DURATION_SEC = SIGNAL_LENGTH / 4  # 1320 s = 22 min
STEP_SIZE = int(SIGNAL_LENGTH * (1 - OVERLAP_PERCENTAGE))  # 4800
STEP_DURATION_SEC = STEP_SIZE / 4  # 1200 s = 20 min

SIX_HOURS_SEC = 21600
MIN_DOMAIN_START_SCREENING = -(SIX_HOURS_SEC + SEGMENT_DURATION_SEC)  # -22920
MIN_DOMAIN_START_DATASET = -44640  # ~12.4 hours

MIN_VALID_HOURS_6H = 2.0
WEIGHT_THRESHOLD = 0.90
FLAT_TOLERANCE = 1e-9

N_FOLDS = 10
VAL_RATIO = 1 / 9  # 80 / 10 / 10
RANDOM_STATE = 42
TLO_WITH_RATIO = 0.75


# ============================================================================
# Verbosity helpers
# ============================================================================
@contextmanager
def suppress_stdout_stderr():
    """Redirect stdout and stderr to devnull temporarily."""
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def setup_verbosity(verbose: bool):
    """Configure logging level.  Progress bars are always shown.

    Args:
        verbose: If True show INFO logs, else ERROR only.
            Progress bars (tqdm) are always enabled regardless of this flag.
    """
    level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(
        level=level,
        format="%(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger.setLevel(level)


# ============================================================================
# Dataclasses
# ============================================================================
@dataclass
class GuidTrackingEntry:
    """Per-GUID tracking data collected during HDF5 creation."""

    all_domain_starts: List[float] = field(default_factory=list)
    included_domain_starts: List[float] = field(default_factory=list)
    skipped_low_weight: List[float] = field(default_factory=list)
    skipped_flat_region: List[float] = field(default_factory=list)
    skipped_scatter_failed: List[float] = field(default_factory=list)
    skipped_duplicate: List[float] = field(default_factory=list)
    skipped_post_delivery: List[float] = field(default_factory=list)
    error: bool = False
    error_msg: Optional[str] = None


# ============================================================================
# Signal utility functions
# ============================================================================
def _normalize_guid(guid_str: str) -> str:
    """Normalize a GUID string for matching: uppercase, remove hyphens.

    Args:
        guid_str: Raw GUID string from filename or CSV.

    Returns:
        Uppercased GUID with hyphens stripped.
    """
    return guid_str.strip().upper().replace("-", "")


def interpolate_bad_values(signal_2d: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf values with linear interpolation, per row.

    For each row, valid samples serve as interpolation knots. Edge NaNs are
    extrapolated flat.  Rows that are entirely bad are filled with 0.
    Operates in-place.

    Args:
        signal_2d: 2-D array of shape ``(n_segments, n_samples)``.

    Returns:
        The same array, modified in-place.
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
            indices[row_bad], indices[row_good], signal_2d[row_idx, row_good]
        )
    return signal_2d


def find_flat_regions(
    signal: np.ndarray, tolerance: float = 1e-3, min_length: int = 20
) -> List[Tuple[int, int]]:
    """Find flat regions in a 1-D signal.

    Args:
        signal: 1-D array.
        tolerance: Max abs difference between consecutive samples to be
            considered flat.
        min_length: Minimum number of consecutive flat samples to qualify.

    Returns:
        List of ``(start_idx, end_idx)`` tuples for qualifying flat regions.
    """
    flat_regions: List[Tuple[int, int]] = []
    start_idx = None
    for i in range(1, len(signal)):
        if abs(signal[i] - signal[i - 1]) <= tolerance:
            if start_idx is None:
                start_idx = i - 1
        else:
            if start_idx is not None:
                end_idx = i - 1
                if (end_idx - start_idx + 1) >= min_length:
                    flat_regions.append((start_idx, end_idx))
                start_idx = None
    if start_idx is not None:
        end_idx = len(signal) - 1
        if (end_idx - start_idx + 1) >= min_length:
            flat_regions.append((start_idx, end_idx))
    return flat_regions


def deduplicate_segments(
    domain_starts: list, sample_weights: np.ndarray
) -> Tuple[List[int], List[int], Dict]:
    """Remove duplicate segments sharing the same domain_start value.

    Keeps only the segment with the highest mean sample weight for each
    unique domain_start.

    Args:
        domain_starts: Per-segment domain_start values.
        sample_weights: 2-D array ``(n_segments, n_samples)``.

    Returns:
        ``(keep_indices, removed_indices, duplicate_groups)``
    """
    groups: Dict[float, List[int]] = defaultdict(list)
    for idx, ds in enumerate(domain_starts):
        groups[ds].append(idx)

    keep_indices: List[int] = []
    removed_indices: List[int] = []
    duplicate_groups: Dict[float, List[int]] = {}

    for ds, indices in groups.items():
        if len(indices) == 1:
            keep_indices.append(indices[0])
        else:
            duplicate_groups[ds] = indices
            best_idx = max(
                indices, key=lambda i: float(np.mean(sample_weights[i, :]))
            )
            keep_indices.append(best_idx)
            for i in indices:
                if i != best_idx:
                    removed_indices.append(i)

    keep_indices.sort()
    removed_indices.sort()
    return keep_indices, removed_indices, duplicate_groups


# ============================================================================
# HDF5 I/O
# ============================================================================
def create_initial_hdf5(
    path: str,
    len_signal: int,
    n_channels: int,
    len_sequence: int = 300,
    n_cross_phase_channels: int = 62,
) -> None:
    """Create a new empty HDF5 file with the full dataset schema.

    Includes v3 scattering layout and the ``second_stage_onset`` field.

    Args:
        path: Output HDF5 file path (overwrites if exists).
        len_signal: Raw signal length (e.g. 5760).
        n_channels: Total phase + cross-phase channels.
        len_sequence: Sequence dimension length.
        n_cross_phase_channels: Channels for fhr_up_ph.
    """
    try:
        os.remove(path)
    except OSError:
        pass

    chunk_n = 32
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w", libver="latest") as h5f:
        h5f.create_dataset(
            "fhr",
            shape=(0, len_signal),
            maxshape=(None, len_signal),
            dtype="f4",
            chunks=(chunk_n, len_signal),
            compression="lzf",
        )
        h5f.create_dataset(
            "up",
            shape=(0, len_signal),
            maxshape=(None, len_signal),
            dtype="f4",
            chunks=(chunk_n, len_signal),
            compression="lzf",
        )
        h5f.create_dataset(
            "fhr_st",
            shape=(0, 43, len_sequence),
            maxshape=(None, 43, len_sequence),
            dtype="f4",
            chunks=(chunk_n, 43, len_sequence),
            compression="lzf",
        )
        h5f.create_dataset(
            "fhr_ph",
            shape=(0, 44, len_sequence),
            maxshape=(None, 44, len_sequence),
            dtype="f4",
            chunks=(chunk_n, 44, len_sequence),
            compression="lzf",
        )
        h5f.create_dataset(
            "fhr_up_ph",
            shape=(0, n_cross_phase_channels, len_sequence),
            maxshape=(None, n_cross_phase_channels, len_sequence),
            dtype="f4",
            chunks=(chunk_n, n_cross_phase_channels, len_sequence),
            compression="lzf",
        )
        h5f.create_dataset(
            "target",
            shape=(0, len_sequence),
            maxshape=(None, len_sequence),
            dtype="f4",
            chunks=(chunk_n, len_sequence),
            compression="lzf",
        )
        h5f.create_dataset(
            "weight",
            shape=(0, len_sequence),
            maxshape=(None, len_sequence),
            dtype="f4",
            chunks=(chunk_n, len_sequence),
            compression="lzf",
        )
        h5f.create_dataset(
            "epoch",
            shape=(0,),
            maxshape=(None,),
            dtype="f4",
            chunks=(chunk_n,),
            compression="lzf",
        )
        h5f.create_dataset(
            "cs_label",
            shape=(0,),
            maxshape=(None,),
            dtype="u1",
            chunks=(chunk_n,),
            compression="lzf",
        )
        h5f.create_dataset(
            "bg_label",
            shape=(0,),
            maxshape=(None,),
            dtype="u1",
            chunks=(chunk_n,),
            compression="lzf",
        )
        h5f.create_dataset(
            "time_from_labor_onset",
            shape=(0,),
            maxshape=(None,),
            dtype="f4",
            chunks=(chunk_n,),
            compression="lzf",
        )
        h5f.create_dataset(
            "second_stage_onset",
            shape=(0,),
            maxshape=(None,),
            dtype="f4",
            chunks=(chunk_n,),
            compression="lzf",
        )
        h5f.create_dataset(
            "guid",
            shape=(0,),
            maxshape=(None,),
            dtype=str_dt,
            chunks=(chunk_n,),
        )


def append_samples_batch(
    path: str,
    fhr_batch: np.ndarray,
    up_batch: np.ndarray,
    fhr_st_batch: np.ndarray,
    fhr_ph_batch: np.ndarray,
    fhr_up_ph_batch: np.ndarray,
    target_batch: np.ndarray,
    weight_batch: np.ndarray,
    guid_batch: list,
    epoch_batch: np.ndarray,
    cs_label_batch: np.ndarray,
    bg_label_batch: np.ndarray,
    tlo_batch: np.ndarray,
    second_stage_batch: np.ndarray,
) -> None:
    """Append K samples to an existing HDF5 file in a single open/close.

    Args:
        path: Path to existing HDF5 file.
        fhr_batch: Shape ``(K, len_signal)``.
        up_batch: Shape ``(K, len_signal)``.
        fhr_st_batch: Shape ``(K, 43, len_seq)``.
        fhr_ph_batch: Shape ``(K, n_ph, len_seq)``.
        fhr_up_ph_batch: Shape ``(K, n_cross, len_seq)``.
        target_batch: Shape ``(K, len_seq)``.
        weight_batch: Shape ``(K, len_seq)``.
        guid_batch: List of GUID strings, length K.
        epoch_batch: Shape ``(K,)``, float32.
        cs_label_batch: Shape ``(K,)``, uint8.
        bg_label_batch: Shape ``(K,)``, uint8.
        tlo_batch: Shape ``(K,)``, float32.
        second_stage_batch: Shape ``(K,)``, float32.
    """
    k = fhr_batch.shape[0]
    if k == 0:
        return
    with h5py.File(path, "a", libver="latest") as h5f:
        idx = h5f["fhr"].shape[0]
        new_size = idx + k
        for _name, ds in h5f.items():
            ds.resize((new_size,) + ds.shape[1:])
        h5f["fhr"][idx:new_size] = fhr_batch
        h5f["up"][idx:new_size] = up_batch
        h5f["fhr_st"][idx:new_size] = fhr_st_batch
        h5f["fhr_ph"][idx:new_size] = fhr_ph_batch
        h5f["fhr_up_ph"][idx:new_size] = fhr_up_ph_batch
        h5f["target"][idx:new_size] = target_batch
        h5f["weight"][idx:new_size] = weight_batch
        h5f["epoch"][idx:new_size] = epoch_batch
        h5f["cs_label"][idx:new_size] = cs_label_batch.astype(np.uint8)
        h5f["bg_label"][idx:new_size] = bg_label_batch.astype(np.uint8)
        if "time_from_labor_onset" in h5f:
            h5f["time_from_labor_onset"][idx:new_size] = tlo_batch
        if "second_stage_onset" in h5f:
            h5f["second_stage_onset"][idx:new_size] = second_stage_batch
        for i, g in enumerate(guid_batch):
            h5f["guid"][idx + i] = g


# ============================================================================
# Scattering masks (v3)
# ============================================================================
def compute_scattering_masks(
    signal_length: int, scattering_T: int = 16, device=None
) -> Dict[str, Any]:
    """Compute v3 coefficient selection masks once.

    v3: UP cap raised to 0.05 Hz in both bands, UP self-phase added.

    Args:
        signal_length: Raw signal length (e.g. 5280).
        scattering_T: Decimation factor.
        device: Torch device.

    Returns:
        Dict with masks, channel counts, and metadata.
    """
    tmp_model = KymatioPhaseScattering1D(
        J=11,
        Q=4,
        T=scattering_T,
        shape=signal_length,
        device=device,
        tukey_alpha=None,
        max_order=1,
    )
    phase_sel = tmp_model.select_fhr_phase_coefficients(min_freq=0.006)
    phase_mask = phase_sel["optimal_mask"]

    cross_sel = tmp_model.select_fhr_up_cross_coefficients_v2(
        band_a_up_max_hz=0.05, band_b_up_max_hz=0.05
    )
    cross_mask = cross_sel["cross_mask"]

    up_phase_sel = tmp_model.select_fhr_phase_coefficients(min_freq=0.002)
    up_phase_mask = up_phase_sel["optimal_mask"]

    n_phase = int(phase_mask.sum().item())
    n_cross = int(cross_mask.sum().item())
    n_up_phase = int(up_phase_mask.sum().item())
    n_combined_cross = n_cross + n_up_phase

    return {
        "phase_mask": phase_mask,
        "cross_mask": cross_mask,
        "up_phase_mask": up_phase_mask,
        "n_phase": n_phase,
        "n_cross": n_cross,
        "n_up_phase": n_up_phase,
        "n_combined_cross": n_combined_cross,
        "cross_metadata": cross_sel.get("metadata", {}),
    }


# ============================================================================
# CSV metadata loading
# ============================================================================
def load_csv_metadata(
    csv_path: str, verbose: bool = True
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Load labor onset and second stage data from the complete CSV.

    Args:
        csv_path: Path to CSV with columns ``trace_guid``,
            ``labor_onset_hours``, ``second_stage_onset_hours``.
        verbose: Whether to print summary info.

    Returns:
        Tuple of ``(labor_onset_map, second_stage_map)`` where each maps
        normalized GUID to seconds relative to delivery.
    """
    df = pd.read_csv(csv_path)
    labor_onset_map: Dict[str, float] = {}
    second_stage_map: Dict[str, float] = {}
    n_tlo_missing = 0
    n_ss_missing = 0

    for _, row in df.iterrows():
        guid = _normalize_guid(str(row["trace_guid"]))

        hours = row.get("labor_onset_hours")
        if pd.notna(hours) and str(hours).strip() != "":
            labor_onset_map[guid] = float(hours) * 3600.0
        else:
            n_tlo_missing += 1

        ss_hours = row.get("second_stage_onset_hours")
        if pd.notna(ss_hours) and str(ss_hours).strip() != "":
            second_stage_map[guid] = float(ss_hours) * 3600.0
        else:
            n_ss_missing += 1

    if verbose:
        logger.info(
            f"Loaded TLO for {len(labor_onset_map)} GUIDs "
            f"({n_tlo_missing} missing) from {csv_path}"
        )
        logger.info(
            f"Loaded second-stage for {len(second_stage_map)} GUIDs "
            f"({n_ss_missing} missing)"
        )
    return labor_onset_map, second_stage_map


# ============================================================================
# Step 1: GUID prescreening
# ============================================================================
def _run_mimo_pipeline(
    record_path: str,
    min_domain_start: float,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """Load a .mat file through MIMO and return signals, domain_starts, weights.

    Args:
        record_path: Path to .mat file.
        min_domain_start: min_domain_start for prepare_data (both channels).
        verbose: If False, suppress MIMO stdout.

    Returns:
        ``(fhr, up, domain_starts, sample_weights)``
    """
    mimo_adaptor = EarlyMaestraMimoAdaptor(
        do_transpose=True,
        process_targets=True,
        n_aux_labels=None,
        signal_indices=range(0, 2),
        n_input_chan=2,
        labels=["HIE", "ACIDOSIS", "HEALTHY"],
        up_shift_secs=-20,
        default_target_index=0,
    )
    if verbose:
        mimo_adaptor.read_single_input(
            record_path,
            out_dec_factor=16,
            out_dec_factor_offset=0,
            target_is_onehot=True,
            dtype=np.float32,
        )
        mimo_prepared, _ = mimo_adaptor.mimo.prepare_data(
            batch_size=1,
            do_evaluate=True,
            align_left=True,
            do_split=True,
            do_pad=True,
            do_reflect=True,
            base_length=BASE_BLOCK_SIZE,
            do_equalize=True,
            do_merge=True,
            min_domain_start=[min_domain_start, min_domain_start],
            max_domain_start=[np.inf, np.inf],
            overlap_percentage=OVERLAP_PERCENTAGE,
        )
    else:
        with suppress_stdout_stderr():
            mimo_adaptor.read_single_input(
                record_path,
                out_dec_factor=16,
                out_dec_factor_offset=0,
                target_is_onehot=True,
                dtype=np.float32,
            )
            mimo_prepared, _ = mimo_adaptor.mimo.prepare_data(
                batch_size=1,
                do_evaluate=True,
                align_left=True,
                do_split=True,
                do_pad=True,
                do_reflect=True,
                base_length=BASE_BLOCK_SIZE,
                do_equalize=True,
                do_merge=True,
                min_domain_start=[min_domain_start, min_domain_start],
                max_domain_start=[np.inf, np.inf],
                overlap_percentage=OVERLAP_PERCENTAGE,
            )

    fhr = mimo_prepared.block_input[:, :, 1].copy()
    up = mimo_prepared.block_input[:, :, 0].copy()
    domain_starts = list(mimo_prepared.domain_start)
    sample_weights = mimo_prepared.sample_weights
    return fhr, up, domain_starts, sample_weights


def _sanitize_signals(
    fhr: np.ndarray, up: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate bad values, clamp range, flush denormals. In-place.

    Args:
        fhr: FHR array ``(N, samples)``.
        up: UP array ``(N, samples)``.

    Returns:
        ``(fhr, up)`` sanitized in-place.
    """
    interpolate_bad_values(fhr)
    interpolate_bad_values(up)
    fhr = np.clip(fhr, 0, 500).astype(np.float32)
    up = np.clip(up, -50, 500).astype(np.float32)
    tiny = np.finfo(np.float32).tiny
    fhr[(fhr != 0) & (np.abs(fhr) < tiny)] = 0.0
    up[(up != 0) & (np.abs(up) < tiny)] = 0.0
    return fhr, up


def _quality_filter_segments(
    fhr: np.ndarray,
    up: np.ndarray,
    sample_weights: np.ndarray,
    domain_starts: list,
) -> Tuple[List[int], int, int]:
    """Apply weight threshold and flat region detection.

    Args:
        fhr: ``(N, samples)``
        up: ``(N, samples)``
        sample_weights: ``(N, n_dec)``
        domain_starts: Per-segment domain_start values.

    Returns:
        ``(valid_indices, n_low_weight, n_flat_region)``
    """
    n_low_weight = 0
    n_flat_region = 0
    valid_indices: List[int] = []

    for i in range(fhr.shape[0]):
        if np.mean(sample_weights[i, :]) < WEIGHT_THRESHOLD:
            n_low_weight += 1
            continue

        fhr_flat = find_flat_regions(fhr[i, :], tolerance=FLAT_TOLERANCE)
        up_flat = find_flat_regions(up[i, :], tolerance=FLAT_TOLERANCE)
        fhr_lens = [end - start + 1 for start, end in fhr_flat]
        up_lens = [end - start + 1 for start, end in up_flat]
        max_flat_fhr = max(fhr_lens, default=0)
        max_flat_up = max(up_lens, default=0)
        total_flat_fhr = sum(l for l in fhr_lens if l >= 240)

        if max_flat_fhr > 480 or max_flat_up > 1200 or total_flat_fhr > 1200:
            n_flat_region += 1
        else:
            valid_indices.append(i)

    return valid_indices, n_low_weight, n_flat_region


def prescreen_guid_6h(
    record_path: str,
    subgroup: str,
    tlo_hours: float,
    second_stage_hours: float,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Prescreen a single GUID for valid signal in the last 6 hours.

    Runs full MIMO + quality pipeline with a restricted domain range, then
    filters to the 6 h window and removes post-delivery segments.

    Args:
        record_path: Full path to .mat file.
        subgroup: Internal subgroup name.
        tlo_hours: Labor onset in hours (NaN if missing).
        second_stage_hours: Second stage onset in hours (NaN if missing).
        verbose: Suppress MIMO output when False.

    Returns:
        Dict with screening results (see output CSV schema in the plan).
    """
    guid_key = os.path.splitext(os.path.basename(record_path))[0]

    def _error(msg):
        return {
            "guid": guid_key,
            "subgroup": subgroup,
            "record_path": record_path,
            "n_total_segments": 0,
            "n_after_dedup": 0,
            "n_valid_segments_6h": 0,
            "n_low_weight": 0,
            "n_flat_region": 0,
            "n_duplicate": 0,
            "estimated_valid_hours_6h": 0.0,
            "has_tlo": not math.isnan(tlo_hours),
            "tlo_hours": tlo_hours,
            "has_second_stage": not math.isnan(second_stage_hours),
            "second_stage_hours": second_stage_hours,
            "domain_start_min": float("nan"),
            "domain_start_max": float("nan"),
            "n_post_delivery": 0,
            "eligible": False,
            "error": True,
            "error_msg": msg,
        }

    try:
        fhr, up, domain_starts, sample_weights = _run_mimo_pipeline(
            record_path, MIN_DOMAIN_START_SCREENING, verbose
        )
        n_total = fhr.shape[0]

        fhr, up = _sanitize_signals(fhr, up)

        keep_idx, removed_idx, _ = deduplicate_segments(
            domain_starts, sample_weights
        )
        n_duplicate = len(removed_idx)
        if removed_idx:
            fhr = fhr[keep_idx]
            up = up[keep_idx]
            sample_weights = sample_weights[keep_idx]
            domain_starts = [domain_starts[i] for i in keep_idx]
        n_after_dedup = fhr.shape[0]

        valid_idx, n_low_weight, n_flat_region = _quality_filter_segments(
            fhr, up, sample_weights, domain_starts
        )

        # Filter to 6 h window and exclude post-delivery
        valid_6h_ds: List[float] = []
        n_post_delivery = 0
        for i in valid_idx:
            ds = domain_starts[i]
            if ds >= 0:
                n_post_delivery += 1
                continue
            if ds > MIN_DOMAIN_START_SCREENING:
                valid_6h_ds.append(ds)

        n_valid_6h = len(valid_6h_ds)
        if n_valid_6h > 0:
            est_hours = (
                (n_valid_6h - 1) * STEP_DURATION_SEC + SEGMENT_DURATION_SEC
            ) / 3600.0
            ds_min = min(valid_6h_ds)
            ds_max = max(valid_6h_ds)
        else:
            est_hours = 0.0
            ds_min = float("nan")
            ds_max = float("nan")

        return {
            "guid": guid_key,
            "subgroup": subgroup,
            "record_path": record_path,
            "n_total_segments": n_total,
            "n_after_dedup": n_after_dedup,
            "n_valid_segments_6h": n_valid_6h,
            "n_low_weight": n_low_weight,
            "n_flat_region": n_flat_region,
            "n_duplicate": n_duplicate,
            "estimated_valid_hours_6h": est_hours,
            "has_tlo": not math.isnan(tlo_hours),
            "tlo_hours": tlo_hours,
            "has_second_stage": not math.isnan(second_stage_hours),
            "second_stage_hours": second_stage_hours,
            "domain_start_min": ds_min,
            "domain_start_max": ds_max,
            "n_post_delivery": n_post_delivery,
            "eligible": est_hours >= MIN_VALID_HOURS_6H,
            "error": False,
            "error_msg": "",
        }
    except Exception as e:
        return _error(str(e))


def prescreen_all_guids(
    records_base_path: str,
    tlo_csv_path: str,
    output_csv_path: str,
    verbose: bool = True,
    num_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Discover all GUIDs and prescreen each for 6 h valid signal.

    Args:
        records_base_path: Root directory with StudyGroup subfolders.
        tlo_csv_path: Complete CSV with TLO + second stage data.
        output_csv_path: Where to save the screening results CSV.
        verbose: Verbosity flag.
        num_workers: Parallel workers (default ``min(cpu_count, 8)``).

    Returns:
        DataFrame with one row per GUID, full screening details.
    """
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)
    # Load TLO/second-stage lookup
    labor_map, ss_map = load_csv_metadata(tlo_csv_path, verbose)

    # Discover .mat files
    all_jobs: List[Tuple[str, str, float, float]] = []
    for folder_name, subgroup in FOLDER_TO_SUBGROUP.items():
        efm_dir = os.path.join(records_base_path, folder_name, "EFMOut")
        if not os.path.isdir(efm_dir):
            logger.warning(f"Folder not found, skipping: {efm_dir}")
            continue
        for fname in sorted(os.listdir(efm_dir)):
            if not fname.endswith(".mat"):
                continue
            fpath = os.path.join(efm_dir, fname)
            guid_key = _normalize_guid(os.path.splitext(fname)[0])
            tlo_h = labor_map.get(guid_key, float("nan"))
            ss_h = ss_map.get(guid_key, float("nan"))
            if not math.isnan(tlo_h):
                tlo_h = tlo_h / 3600.0  # back to hours for storage
            if not math.isnan(ss_h):
                ss_h = ss_h / 3600.0
            all_jobs.append((fpath, subgroup, tlo_h, ss_h))

    logger.info(
        f"Prescreening {len(all_jobs)} GUIDs across "
        f"{len(FOLDER_TO_SUBGROUP)} subgroups (workers={num_workers})"
    )

    results: List[Dict[str, Any]] = []
    fn = partial(
        _prescreen_worker,
        verbose=verbose,
    )

    if num_workers <= 1:
        for job in tqdm(all_jobs, desc="Prescreening"):
            results.append(fn(job))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(fn, job): job for job in all_jobs}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Prescreening",
                disable=False,
            ):
                results.append(future.result())

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Screening results saved to {output_csv_path}")

    # Summary
    n_eligible = df["eligible"].sum()
    n_error = df["error"].sum()
    logger.info(
        f"Prescreening done: {len(df)} total, {n_eligible} eligible, "
        f"{n_error} errors"
    )
    for sg in sorted(df["subgroup"].unique()):
        sg_df = df[df["subgroup"] == sg]
        sg_el = sg_df["eligible"].sum()
        sg_err = sg_df["error"].sum()
        logger.info(f"  {sg:<28} {len(sg_df):>5} total, {sg_el:>5} eligible, {sg_err:>3} errors")

    return df


def _prescreen_worker(
    job: Tuple[str, str, float, float], verbose: bool = True
) -> Dict[str, Any]:
    """Worker wrapper for prescreen_guid_6h (unpacks tuple).

    Args:
        job: ``(record_path, subgroup, tlo_hours, second_stage_hours)``
        verbose: Verbosity flag.

    Returns:
        Screening result dict.
    """
    record_path, subgroup, tlo_h, ss_h = job
    return prescreen_guid_6h(record_path, subgroup, tlo_h, ss_h, verbose)


# ============================================================================
# Step 2: GUID selection for classification
# ============================================================================
def select_classification_guids(
    screening_df: pd.DataFrame,
    tlo_with_ratio: float = TLO_WITH_RATIO,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """Select GUIDs for classification with class balancing.

    Unhealthy GUIDs: all eligible included.
    Healthy GUIDs: subsampled so total healthy = total unhealthy, with
    proportional subgroup allocation, minority floor for ``healthy_bg_cs``,
    proportional boost for ``healthy_bg_no_cs``, deficit absorbed only from
    ``healthy_no_bg_no_cs``, and 75/25 TLO ratio per subgroup.

    Args:
        screening_df: DataFrame from prescreening (all GUIDs).
        tlo_with_ratio: Target TLO-present fraction per healthy subgroup.
        random_state: Random seed.
        verbose: Verbosity flag.

    Returns:
        Dict mapping subgroup name to list of .mat file paths.
    """
    rng = random.Random(random_state)
    eligible = screening_df[
        (screening_df["eligible"] == True) & (screening_df["error"] == False)
    ].copy()

    # --- Unhealthy: all eligible ---
    unhealthy = eligible[eligible["subgroup"].isin(UNHEALTHY_SUBGROUPS)]
    unhealthy_by_sg: Dict[str, List[str]] = {}
    for sg, grp in unhealthy.groupby("subgroup"):
        unhealthy_by_sg[sg] = grp["record_path"].tolist()
    n_unhealthy = len(unhealthy)

    unhealthy_counts = {sg: len(p) for sg, p in unhealthy_by_sg.items()}
    minority_unhealthy_sg = min(unhealthy_counts, key=unhealthy_counts.get)
    minority_unhealthy_n = unhealthy_counts[minority_unhealthy_sg]

    # --- Healthy eligible pools ---
    healthy = eligible[eligible["subgroup"].isin(HEALTHY_SUBGROUPS)]
    healthy_pools: Dict[str, pd.DataFrame] = {
        sg: grp for sg, grp in healthy.groupby("subgroup")
    }
    eligible_counts = {sg: len(grp) for sg, grp in healthy_pools.items()}
    total_healthy_eligible = sum(eligible_counts.values())

    # Initial proportional targets
    targets: Dict[str, int] = {}
    for sg in HEALTHY_SUBGROUPS:
        cnt = eligible_counts.get(sg, 0)
        targets[sg] = round((cnt / max(total_healthy_eligible, 1)) * n_unhealthy)

    # Minority floor + BG pair boost
    original_bg_cs = targets.get("healthy_bg_cs", 0)
    if original_bg_cs < minority_unhealthy_n:
        bump_bg_cs = minority_unhealthy_n - original_bg_cs
        targets["healthy_bg_cs"] = minority_unhealthy_n

        original_bg_no_cs = targets.get("healthy_bg_no_cs", 0)
        if original_bg_cs > 0:
            boost_ratio = bump_bg_cs / original_bg_cs
        else:
            boost_ratio = 1.0
        bump_bg_no_cs = round(original_bg_no_cs * boost_ratio)
        bump_bg_no_cs = min(
            bump_bg_no_cs,
            eligible_counts.get("healthy_bg_no_cs", 0) - original_bg_no_cs,
        )
        bump_bg_no_cs = max(bump_bg_no_cs, 0)
        targets["healthy_bg_no_cs"] = original_bg_no_cs + bump_bg_no_cs

        total_deficit = bump_bg_cs + bump_bg_no_cs
        targets["healthy_no_bg_no_cs"] -= total_deficit

        if targets["healthy_no_bg_no_cs"] < 0:
            overflow = abs(targets["healthy_no_bg_no_cs"])
            targets["healthy_no_bg_no_cs"] = 0
            targets["healthy_no_bg_cs"] = max(
                targets.get("healthy_no_bg_cs", 0) - overflow, 0
            )

    # Rounding fix
    diff = n_unhealthy - sum(targets.values())
    if diff != 0:
        targets["healthy_no_bg_no_cs"] += diff

    # Cap to available
    for sg in list(targets.keys()):
        avail = eligible_counts.get(sg, 0)
        if targets[sg] > avail:
            overflow = targets[sg] - avail
            targets[sg] = avail
            # redistribute to largest with room
            for fallback in ["healthy_no_bg_no_cs", "healthy_no_bg_cs",
                             "healthy_bg_no_cs", "healthy_bg_cs"]:
                if fallback == sg:
                    continue
                room = eligible_counts.get(fallback, 0) - targets[fallback]
                add = min(overflow, room)
                targets[fallback] += add
                overflow -= add
                if overflow == 0:
                    break

    # Sample per subgroup with TLO constraint
    selected: Dict[str, List[str]] = {}
    for sg in HEALTHY_SUBGROUPS:
        target_n = targets[sg]
        if target_n <= 0:
            selected[sg] = []
            continue
        pool = healthy_pools.get(sg, pd.DataFrame())
        pool_with = pool[pool["has_tlo"] == True]["record_path"].tolist()
        pool_without = pool[pool["has_tlo"] == False]["record_path"].tolist()

        n_with = round(target_n * tlo_with_ratio)
        n_without = target_n - n_with

        if len(pool_with) < n_with:
            n_with = len(pool_with)
            n_without = min(target_n - n_with, len(pool_without))
        if len(pool_without) < n_without:
            n_without = len(pool_without)
            n_with = min(target_n - n_without, len(pool_with))

        rng.shuffle(pool_with)
        rng.shuffle(pool_without)
        selected[sg] = pool_with[:n_with] + pool_without[:n_without]

    result = {**unhealthy_by_sg, **selected}

    # Summary
    if verbose:
        total_h = sum(len(v) for sg, v in result.items() if sg in HEALTHY_SUBGROUPS)
        total_u = sum(len(v) for sg, v in result.items() if sg in UNHEALTHY_SUBGROUPS)
        logger.info("=" * 70)
        logger.info("GUID SELECTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"{'Subgroup':<28} {'Eligible':>8} {'Selected':>8} {'TLO':>5}")
        logger.info("-" * 70)
        for sg in sorted(result.keys()):
            el = eligible_counts.get(sg, unhealthy_counts.get(sg, 0))
            sel = len(result[sg])
            n_tlo = sum(
                1
                for p in result[sg]
                if not eligible[eligible["record_path"] == p]["has_tlo"].empty
                and eligible[eligible["record_path"] == p]["has_tlo"].iloc[0]
            ) if sg in HEALTHY_SUBGROUPS else "n/a"
            logger.info(f"  {sg:<28} {el:>8} {sel:>8} {str(n_tlo):>5}")
        logger.info("-" * 70)
        logger.info(f"  Total unhealthy: {total_u}")
        logger.info(f"  Total healthy:   {total_h}")
        logger.info(
            f"  Minority unhealthy ({minority_unhealthy_sg}): "
            f"{minority_unhealthy_n}"
        )
        logger.info(
            f"  Minority healthy (healthy_bg_cs): "
            f"{len(result.get('healthy_bg_cs', []))}"
        )
        logger.info("=" * 70)

    return result


# ============================================================================
# Step 3: K-Fold CV splits
# ============================================================================
def create_cv_splits(
    data: Dict[str, List[str]],
    n_splits: int = N_FOLDS,
    val_ratio: float = VAL_RATIO,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """Create stratified-by-subgroup K-fold CV with inner train/val split.

    Args:
        data: Mapping subgroup name to list of file paths.
        n_splits: Number of outer folds.
        val_ratio: Fraction of non-test to use as validation.
        random_state: Seed for reproducibility.

    Returns:
        ``{"fold_1": {"train": {sg: [paths]}, "val": {...}, "test": {...}}, ...}``
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits_per_group = {
        group: list(kf.split(file_list))
        for group, file_list in data.items()
    }

    folds: Dict[str, Dict] = {}
    for fold_idx in range(n_splits):
        fold_name = f"fold_{fold_idx + 1}"
        fold_data = {"train": {}, "val": {}, "test": {}}

        for group, splits in splits_per_group.items():
            train_val_idx, test_idx = splits[fold_idx]
            test_files = [data[group][i] for i in test_idx]
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_ratio,
                shuffle=True,
                random_state=random_state,
            )
            fold_data["train"][group] = [data[group][i] for i in train_idx]
            fold_data["val"][group] = [data[group][i] for i in val_idx]
            fold_data["test"][group] = test_files

        folds[fold_name] = fold_data

    return folds


# ============================================================================
# Step 4: HDF5 dataset creation from records list
# ============================================================================
def create_hdf5_dataset_from_records_list(
    hdf5_path: str,
    records_list: List[str],
    cs_label: bool,
    bg_label: bool,
    pre_defined_target: int,
    precomputed_masks: Dict[str, Any],
    labor_onset_map: Dict[str, float],
    second_stage_map: Dict[str, float],
    base_block_size: int = BASE_BLOCK_SIZE,
    overlap_percentage: float = OVERLAP_PERCENTAGE,
    device: Optional[torch.device] = None,
    run_guid_analysis: bool = False,
    scatter_batch_size: int = 16,
    verbose: bool = True,
) -> List[str]:
    """Process a list of .mat files and write segments to an HDF5 file.

    Runs the full MIMO + sanitize + dedup + quality + post-delivery-skip +
    scattering v3 pipeline for each record, then batch-writes valid segments.

    Args:
        hdf5_path: Output HDF5 file (must already be created via
            ``create_initial_hdf5``).
        records_list: List of .mat file paths.
        cs_label: Caesarean section flag for all records.
        bg_label: Blood gas flag for all records.
        pre_defined_target: Class target (1=HEALTHY, 2=ACIDOSIS, 3=HIE).
        precomputed_masks: Dict from ``compute_scattering_masks``.
        labor_onset_map: Normalized GUID -> TLO in seconds.
        second_stage_map: Normalized GUID -> second stage in seconds.
        base_block_size: Base block size for MIMO.
        overlap_percentage: Overlap fraction.
        device: Torch device for scattering.
        run_guid_analysis: Collect per-GUID tracking data.
        scatter_batch_size: Scattering batch size.
        verbose: Verbosity flag.

    Returns:
        List of record paths that errored.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scattering_T = 16
    signal_length = int(base_block_size * 1.5)

    st_model = KymatioPhaseScattering1D(
        J=11, Q=4, T=scattering_T, shape=signal_length,
        device=device, tukey_alpha=None, max_order=1,
    )

    phase_mask = precomputed_masks["phase_mask"].to(device)
    cross_mask = precomputed_masks["cross_mask"].to(device)
    up_phase_mask = precomputed_masks["up_phase_mask"].to(device)

    errors_list: List[str] = []
    guid_tracking: Optional[Dict[str, GuidTrackingEntry]] = (
        {} if run_guid_analysis else None
    )

    for record in tqdm(records_list, desc=os.path.basename(hdf5_path)):
        try:
            default_ti = (
                (pre_defined_target - 1)
                if pre_defined_target is not None
                else None
            )
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
            if verbose:
                mimo_adaptor.read_single_input(
                    record, out_dec_factor=16, out_dec_factor_offset=0,
                    target_is_onehot=True, dtype=np.float32,
                )
                mimo_prepared, _ = mimo_adaptor.mimo.prepare_data(
                    batch_size=1, do_evaluate=True, align_left=True,
                    do_split=True, do_pad=True, do_reflect=True,
                    base_length=base_block_size, do_equalize=True,
                    do_merge=True,
                    min_domain_start=[MIN_DOMAIN_START_DATASET, MIN_DOMAIN_START_DATASET],
                    max_domain_start=[np.inf, np.inf],
                    overlap_percentage=overlap_percentage,
                )
            else:
                with suppress_stdout_stderr():
                    mimo_adaptor.read_single_input(
                        record, out_dec_factor=16, out_dec_factor_offset=0,
                        target_is_onehot=True, dtype=np.float32,
                    )
                    mimo_prepared, _ = mimo_adaptor.mimo.prepare_data(
                        batch_size=1, do_evaluate=True, align_left=True,
                        do_split=True, do_pad=True, do_reflect=True,
                        base_length=base_block_size, do_equalize=True,
                        do_merge=True,
                        min_domain_start=[MIN_DOMAIN_START_DATASET, MIN_DOMAIN_START_DATASET],
                        max_domain_start=[np.inf, np.inf],
                        overlap_percentage=overlap_percentage,
                    )

            fhr = mimo_prepared.block_input[:, :, 1].copy()
            up = mimo_prepared.block_input[:, :, 0].copy()
            domain_starts = list(mimo_prepared.domain_start)
            sample_weights = mimo_prepared.sample_weights

            fhr, up = _sanitize_signals(fhr, up)

            guid_key = os.path.splitext(os.path.basename(record))[0]
            normalized_key = _normalize_guid(guid_key)
            labor_onset_sec = labor_onset_map.get(normalized_key, float("nan"))
            ss_sec = second_stage_map.get(normalized_key, float("nan"))

            if guid_tracking is not None:
                guid_tracking[guid_key] = GuidTrackingEntry(
                    all_domain_starts=[float(ds) for ds in domain_starts],
                )

            # Dedup
            keep_idx, removed_idx, _ = deduplicate_segments(
                domain_starts, sample_weights
            )
            if removed_idx:
                if guid_tracking is not None:
                    guid_tracking[guid_key].skipped_duplicate.extend(
                        float(domain_starts[i]) for i in removed_idx
                    )
                fhr = fhr[keep_idx]
                up = up[keep_idx]
                sample_weights = sample_weights[keep_idx]
                domain_starts = [domain_starts[i] for i in keep_idx]

            # Quality filter
            valid_indices: List[int] = []
            for i in range(fhr.shape[0]):
                if np.mean(sample_weights[i, :]) < WEIGHT_THRESHOLD:
                    if guid_tracking is not None:
                        guid_tracking[guid_key].skipped_low_weight.append(
                            float(domain_starts[i])
                        )
                    continue
                fhr_flat = find_flat_regions(fhr[i, :], tolerance=FLAT_TOLERANCE)
                up_flat = find_flat_regions(up[i, :], tolerance=FLAT_TOLERANCE)
                fhr_lens = [end - start + 1 for start, end in fhr_flat]
                up_lens = [end - start + 1 for start, end in up_flat]
                max_flat_fhr = max(fhr_lens, default=0)
                max_flat_up = max(up_lens, default=0)
                total_flat_fhr = sum(l for l in fhr_lens if l >= 240)
                if (
                    max_flat_fhr > 480
                    or max_flat_up > 1200
                    or total_flat_fhr > 1200
                ):
                    if guid_tracking is not None:
                        guid_tracking[guid_key].skipped_flat_region.append(
                            float(domain_starts[i])
                        )
                    continue
                # Skip post-delivery segments
                if domain_starts[i] >= 0:
                    if guid_tracking is not None:
                        guid_tracking[guid_key].skipped_post_delivery.append(
                            float(domain_starts[i])
                        )
                    continue
                valid_indices.append(i)

            if not valid_indices:
                continue

            # Batched scattering
            valid_fhr = fhr[valid_indices]
            valid_up = up[valid_indices]
            st_input = torch.from_numpy(
                np.stack([valid_fhr, valid_up], axis=1)
            ).float().to(device)

            n_valid = st_input.shape[0]
            st_phase_list = [None] * n_valid
            st_cross_list = [None] * n_valid
            st_up_phase_list = [None] * n_valid
            scatter_failed: set = set()

            for batch_start in range(0, n_valid, scatter_batch_size):
                batch_end = min(batch_start + scatter_batch_size, n_valid)
                batch = st_input[batch_start:batch_end]
                try:
                    bp = st_model(
                        x=batch, compute_phase=True,
                        compute_cross_phase=False,
                        scattering_channel=0, phase_channels=[0],
                    )
                    bc = st_model(
                        x=batch, compute_phase=False,
                        compute_cross_phase=True,
                        scattering_channel=0, phase_channels=[0, 1],
                    )
                    bup = st_model(
                        x=batch, compute_phase=True,
                        compute_cross_phase=False,
                        scattering_channel=0, phase_channels=[1],
                    )
                    bs = batch.shape[0]
                    for lj in range(bs):
                        gj = batch_start + lj
                        st_phase_list[gj] = {
                            k: (v[lj:lj+1] if isinstance(v, torch.Tensor) and v.shape[0] == bs else v)
                            for k, v in bp.items()
                        }
                        st_cross_list[gj] = {
                            k: (v[lj:lj+1] if isinstance(v, torch.Tensor) and v.shape[0] == bs else v)
                            for k, v in bc.items()
                        }
                        st_up_phase_list[gj] = {
                            k: (v[lj:lj+1] if isinstance(v, torch.Tensor) and v.shape[0] == bs else v)
                            for k, v in bup.items()
                        }
                except RuntimeError:
                    for lj in range(batch.shape[0]):
                        gj = batch_start + lj
                        seg = st_input[gj:gj+1]
                        try:
                            sp = st_model(x=seg, compute_phase=True, compute_cross_phase=False, scattering_channel=0, phase_channels=[0])
                            sc = st_model(x=seg, compute_phase=False, compute_cross_phase=True, scattering_channel=0, phase_channels=[0, 1])
                            su = st_model(x=seg, compute_phase=True, compute_cross_phase=False, scattering_channel=0, phase_channels=[1])
                            st_phase_list[gj] = sp
                            st_cross_list[gj] = sc
                            st_up_phase_list[gj] = su
                        except RuntimeError as seg_err:
                            orig_idx = valid_indices[gj]
                            logger.error(
                                f"{guid_key} seg {orig_idx} "
                                f"(epoch={domain_starts[orig_idx]}): "
                                f"scattering failed: {seg_err}"
                            )
                            scatter_failed.add(gj)
                            if guid_tracking is not None:
                                guid_tracking[guid_key].skipped_scatter_failed.append(
                                    float(domain_starts[orig_idx])
                                )

            # Collect valid scattered segments
            b_fhr, b_up = [], []
            b_fhr_st, b_fhr_ph, b_fhr_up_ph = [], [], []
            b_target, b_weight = [], []
            b_guid, b_epoch = [], []
            b_cs, b_bg, b_tlo, b_ss = [], [], [], []

            record_name = os.path.splitext(os.path.basename(record))[0]

            for seg_j in range(n_valid):
                if seg_j in scatter_failed:
                    continue
                orig_idx = valid_indices[seg_j]

                fhr_st_coeff = st_phase_list[seg_j]["scattering"][0]
                fhr_ph_full = st_phase_list[seg_j]["phase_corr"][0]
                cross_full = st_cross_list[seg_j]["cross_phase_corr"][0]
                up_ph_full = st_up_phase_list[seg_j]["phase_corr"][0]

                fhr_ph_coeff = fhr_ph_full[phase_mask, :]
                cross_ph = cross_full[cross_mask, :]
                up_self_ph = up_ph_full[up_phase_mask, :]
                fhr_up_ph_coeff = torch.cat([cross_ph, up_self_ph], dim=0)

                if guid_tracking is not None:
                    guid_tracking[guid_key].included_domain_starts.append(
                        float(domain_starts[orig_idx])
                    )

                tflo = float(domain_starts[orig_idx]) - labor_onset_sec
                tss = float(domain_starts[orig_idx]) - ss_sec

                b_fhr.append(fhr[orig_idx, :])
                b_up.append(up[orig_idx, :])
                b_fhr_st.append(fhr_st_coeff.detach().cpu().numpy())
                b_fhr_ph.append(fhr_ph_coeff.detach().cpu().numpy())
                b_fhr_up_ph.append(fhr_up_ph_coeff.detach().cpu().numpy())
                b_target.append(pre_defined_target * sample_weights[orig_idx, :])
                b_weight.append(sample_weights[orig_idx, :])
                b_guid.append(record_name)
                b_epoch.append(domain_starts[orig_idx])
                b_cs.append(cs_label)
                b_bg.append(bg_label)
                b_tlo.append(tflo)
                b_ss.append(tss)

            if b_fhr:
                append_samples_batch(
                    path=hdf5_path,
                    fhr_batch=np.stack(b_fhr),
                    up_batch=np.stack(b_up),
                    fhr_st_batch=np.stack(b_fhr_st),
                    fhr_ph_batch=np.stack(b_fhr_ph),
                    fhr_up_ph_batch=np.stack(b_fhr_up_ph),
                    target_batch=np.stack(b_target),
                    weight_batch=np.stack(b_weight),
                    guid_batch=b_guid,
                    epoch_batch=np.array(b_epoch, dtype=np.float32),
                    cs_label_batch=np.array(b_cs, dtype=np.uint8),
                    bg_label_batch=np.array(b_bg, dtype=np.uint8),
                    tlo_batch=np.array(b_tlo, dtype=np.float32),
                    second_stage_batch=np.array(b_ss, dtype=np.float32),
                )

        except Exception as e:
            errors_list.append(record)
            logger.error(f"Failed processing {record}:\n{traceback.format_exc()}")
            if guid_tracking is not None:
                err_guid = os.path.splitext(os.path.basename(record))[0]
                guid_tracking[err_guid] = GuidTrackingEntry(
                    error=True, error_msg=str(e)
                )

    # GUID analysis
    if run_guid_analysis and hdf5_path and guid_tracking:
        try:
            from guid_analysis import run_guid_analysis as _run_analysis
            segment_dur = signal_length / 4
            _run_analysis(hdf5_path, guid_tracking, segment_duration_sec=segment_dur)
        except Exception as e:
            logger.error(f"GUID analysis failed: {e}")

    return errors_list


# ============================================================================
# Step 5: Main orchestrator
# ============================================================================
def create_new_pipeline(
    records_base_path: str,
    output_base_path: str,
    tlo_csv_path: str,
    verbose: bool = True,
    scatter_batch_size: int = 16,
    num_workers: Optional[int] = None,
    screening_csv_path: Optional[str] = None,
    classification_pickle_path: Optional[str] = None,
):
    """Run the complete new dataset creation pipeline.

    Steps:
        1. Prescreen all GUIDs for valid signal in last 6 hours.
        2. Select GUIDs for classification (balanced).
        3. Create 10-fold stratified CV splits (80/10/10).
        4. Build classification HDF5 datasets.
        5. Build pretraining HDF5 datasets from BG subgroup leftovers.

    Args:
        records_base_path: Root dir with StudyGroup subfolders.
        output_base_path: Output directory for all generated files.
        tlo_csv_path: Path to complete CSV with TLO + second stage data.
        verbose: If False, suppress all output except errors.
        scatter_batch_size: Scattering batch size.
        num_workers: Parallel prescreening workers.
        screening_csv_path: Skip Step 1, load this pre-computed CSV.
        classification_pickle_path: Skip Steps 1-3, load this pickle.
    """
    setup_verbosity(verbose)
    os.makedirs(output_base_path, exist_ok=True)

    # Load CSV metadata (needed for HDF5 creation in all paths)
    labor_onset_map, second_stage_map = load_csv_metadata(tlo_csv_path, verbose)

    # Compute scattering masks once
    logger.info("Computing scattering masks (v3)...")
    masks = compute_scattering_masks(SIGNAL_LENGTH, scattering_T=16)
    n_combined_cross = masks["n_combined_cross"]
    total_channels = masks["n_phase"] + n_combined_cross
    logger.info(
        f"v3 layout: {masks['n_phase']} phase + {masks['n_cross']} cross + "
        f"{masks['n_up_phase']} UP self-phase = {total_channels} total"
    )

    sequence_length = SIGNAL_LENGTH // 16

    # ------------------------------------------------------------------
    # Resolve starting point based on skip flags
    # ------------------------------------------------------------------
    if classification_pickle_path is not None:
        # Skip Steps 1-3
        logger.info(
            f"Loading pre-computed folds from: {classification_pickle_path}"
        )
        with open(classification_pickle_path, "rb") as f:
            classification_folds = pickle.load(f)
        logger.info(f"Loaded {len(classification_folds)} folds")

        # Still need all BG files for pretraining leftovers
        all_bg_cs_files = _discover_mat_files(
            records_base_path, "HEALTHY_NO_ACIDOSIS_CS"
        )
        all_bg_no_cs_files = _discover_mat_files(
            records_base_path, "HEALTHY_NO_ACIDOSIS_NoCS"
        )
        # Collect classification BG files from pickle
        cls_bg_cs = set()
        cls_bg_no_cs = set()
        for fold_data in classification_folds.values():
            for part in fold_data.values():
                cls_bg_cs.update(part.get("healthy_bg_cs", []))
                cls_bg_no_cs.update(part.get("healthy_bg_no_cs", []))

    else:
        # --- Step 1: Prescreening ---
        if screening_csv_path is not None:
            logger.info(f"Loading screening CSV: {screening_csv_path}")
            screening_df = pd.read_csv(screening_csv_path)
        else:
            screening_csv_out = os.path.join(
                output_base_path, "guid_screening_results.csv"
            )
            screening_df = prescreen_all_guids(
                records_base_path, tlo_csv_path, screening_csv_out,
                verbose=verbose, num_workers=num_workers,
            )

        # --- Step 2: GUID selection ---
        classification_guids = select_classification_guids(
            screening_df, verbose=verbose
        )

        # Save selection summary
        summary_path = os.path.join(
            output_base_path, "classification_guid_selection_summary.json"
        )
        summary = {
            sg: {
                "count": len(paths),
                "paths_sample": paths[:3] if paths else [],
            }
            for sg, paths in classification_guids.items()
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # --- Step 3: Fold creation ---
        classification_folds = create_cv_splits(
            classification_guids, n_splits=N_FOLDS,
            val_ratio=VAL_RATIO, random_state=RANDOM_STATE,
        )
        pickle_path = os.path.join(
            output_base_path, "classification_dataset_records.pickle"
        )
        with open(pickle_path, "wb") as f:
            pickle.dump(classification_folds, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Fold assignments saved to {pickle_path}")

        # Identify BG files for pretraining
        all_bg_cs_files = _discover_mat_files(
            records_base_path, "HEALTHY_NO_ACIDOSIS_CS"
        )
        all_bg_no_cs_files = _discover_mat_files(
            records_base_path, "HEALTHY_NO_ACIDOSIS_NoCS"
        )
        cls_bg_cs = set(classification_guids.get("healthy_bg_cs", []))
        cls_bg_no_cs = set(classification_guids.get("healthy_bg_no_cs", []))

    # ------------------------------------------------------------------
    # Step 4: Classification HDF5 creation
    # ------------------------------------------------------------------
    kfold_path = os.path.join(output_base_path, "k_fold_cross_validation_dataset")
    os.makedirs(kfold_path, exist_ok=True)

    run_ga = True  # GUID analysis on first fold only
    for fold_name, fold_data in classification_folds.items():
        logger.info(f"Processing {fold_name}...")
        fold_dir = os.path.join(kfold_path, fold_name)
        os.makedirs(fold_dir, exist_ok=True)

        for partition_name, subgroups in fold_data.items():
            part_dir = os.path.join(fold_dir, partition_name)
            os.makedirs(part_dir, exist_ok=True)

            for sg, records in subgroups.items():
                target, cs, bg = SUBGROUP_META[sg]
                hdf5_file = os.path.join(part_dir, f"{sg}.hdf5")
                create_initial_hdf5(
                    path=hdf5_file,
                    len_signal=SIGNAL_LENGTH,
                    n_channels=total_channels,
                    len_sequence=sequence_length,
                    n_cross_phase_channels=n_combined_cross,
                )
                create_hdf5_dataset_from_records_list(
                    hdf5_path=hdf5_file,
                    records_list=records,
                    cs_label=cs,
                    bg_label=bg,
                    pre_defined_target=target,
                    precomputed_masks=masks,
                    labor_onset_map=labor_onset_map,
                    second_stage_map=second_stage_map,
                    base_block_size=BASE_BLOCK_SIZE,
                    overlap_percentage=OVERLAP_PERCENTAGE,
                    run_guid_analysis=run_ga,
                    scatter_batch_size=scatter_batch_size,
                    verbose=verbose,
                )

        run_ga = False  # only first fold

    logger.info("Classification datasets complete.")

    # ------------------------------------------------------------------
    # Step 5: Pretraining HDF5 creation
    # ------------------------------------------------------------------
    pretrain_path = os.path.join(output_base_path, "pre_training_dataset")
    os.makedirs(pretrain_path, exist_ok=True)

    leftover_bg_cs = [f for f in all_bg_cs_files if f not in cls_bg_cs]
    leftover_bg_no_cs = [f for f in all_bg_no_cs_files if f not in cls_bg_no_cs]

    logger.info(
        f"Pretraining leftovers: BG_CS={len(leftover_bg_cs)}, "
        f"BG_NoCS={len(leftover_bg_no_cs)}"
    )

    random.shuffle(leftover_bg_cs)
    split_cs = int(len(leftover_bg_cs) * 0.9)
    train_cs = leftover_bg_cs[:split_cs]
    test_cs = leftover_bg_cs[split_cs:]

    random.shuffle(leftover_bg_no_cs)
    split_no_cs = int(len(leftover_bg_no_cs) * 0.9)
    train_no_cs = leftover_bg_no_cs[:split_no_cs]
    test_no_cs = leftover_bg_no_cs[split_no_cs:]

    pretrain_sets = [
        ("train_dataset_cs.hdf5", train_cs, True, True),
        ("train_dataset_no_cs.hdf5", train_no_cs, False, True),
        ("test_dataset_cs.hdf5", test_cs, True, True),
        ("test_dataset_no_cs.hdf5", test_no_cs, False, True),
    ]

    for fname, records, cs, bg in pretrain_sets:
        hdf5_file = os.path.join(pretrain_path, fname)
        logger.info(f"Creating {fname} ({len(records)} GUIDs)...")
        create_initial_hdf5(
            path=hdf5_file,
            len_signal=SIGNAL_LENGTH,
            n_channels=total_channels,
            len_sequence=sequence_length,
            n_cross_phase_channels=n_combined_cross,
        )
        create_hdf5_dataset_from_records_list(
            hdf5_path=hdf5_file,
            records_list=records,
            cs_label=cs,
            bg_label=bg,
            pre_defined_target=1,  # all healthy
            precomputed_masks=masks,
            labor_onset_map=labor_onset_map,
            second_stage_map=second_stage_map,
            base_block_size=BASE_BLOCK_SIZE,
            overlap_percentage=OVERLAP_PERCENTAGE,
            run_guid_analysis=False,
            scatter_batch_size=scatter_batch_size,
            verbose=verbose,
        )

    logger.info("Pretraining datasets complete.")
    logger.info("Pipeline finished.")


def _discover_mat_files(records_base_path: str, folder_name: str) -> List[str]:
    """List all .mat files in a StudyGroup subfolder.

    Args:
        records_base_path: Root dir with StudyGroup subfolders.
        folder_name: e.g. ``"HEALTHY_NO_ACIDOSIS_CS"``.

    Returns:
        Sorted list of full .mat file paths.
    """
    efm_dir = os.path.join(records_base_path, folder_name, "EFMOut")
    if not os.path.isdir(efm_dir):
        return []
    return sorted(
        os.path.join(efm_dir, f) for f in os.listdir(efm_dir) if f.endswith(".mat")
    )


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    # ---- Configure paths here ----
    records_base_path = r"/data/deid/datafabric/fetal-heart-tracing/StudyGroup2022_v4/"
    output_base_path = r"/data1/fetal-heart-tracing/HDF5_Datasets/new_pipeline_6h"
    tlo_csv_path = r"/path/to/complete_labor_onset.csv"

    # ---- Options ----
    verbose = False
    scatter_batch_size = 128
    num_workers = None  # defaults to min(cpu_count, 8)

    # ---- Resume / skip flags (set to None for full pipeline) ----
    screening_csv_path = None  # e.g. r"/path/to/guid_screening_results.csv"
    classification_pickle_path = None  # e.g. r"/path/to/classification_dataset_records.pickle"

    create_new_pipeline(
        records_base_path=records_base_path,
        output_base_path=output_base_path,
        tlo_csv_path=tlo_csv_path,
        verbose=verbose,
        scatter_batch_size=scatter_batch_size,
        num_workers=num_workers,
        screening_csv_path=screening_csv_path,
        classification_pickle_path=classification_pickle_path,
    )
