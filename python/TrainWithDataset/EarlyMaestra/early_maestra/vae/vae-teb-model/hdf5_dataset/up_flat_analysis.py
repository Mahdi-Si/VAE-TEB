"""UP flat region analysis tool for EFM (CTG) recordings.

Processes complete UP (uterine pressure) signals from .mat EFM recordings to
identify flat regions — periods where the UP signal is constant or near-constant,
indicating either a genuine absence of contractions or a transducer/signal issue.

The output CSV has two types of rows, distinguished by the ``row_type`` column:
  - ``segment_summary``: One row per segment. Provides segment-level statistics.
    All ``flat_*`` columns (except ``flat_excluded_by_fhr_filter``) are NaN because
    this row describes the segment as a whole, not a specific flat region.
  - ``flat_region``: One row per detected flat region within a segment. All columns
    are populated.

Output CSV columns
------------------
guid : str
    Recording identifier — the .mat filename without its extension. Uniquely
    identifies one CTG recording.

subgroup : str
    Clinical outcome subgroup key derived from the folder name, e.g.
    ``'hie_cs'``, ``'acidosis_no_cs'``, ``'healthy_no_bg_no_cs'``.

clinical_class : str
    High-level clinical class: ``'HEALTHY'``, ``'ACIDOSIS'``, or ``'HIE'``.

cs_label : bool
    True if the recording involved a caesarean section delivery.

bg_label : bool
    True if a blood gas measurement was available for this recording.

segment_idx : int
    Zero-based index of the segment within this recording (after deduplication).
    Segments are produced by the MIMO ``split_long`` pipeline with configurable
    ``base_block_size`` and ``overlap_percentage``.

domain_start_sec : float
    Start time of the segment in **seconds relative to delivery**. Negative values
    mean before delivery, 0 means delivery time, positive means after delivery.
    Computed by the MIMO equalization/split pipeline.

domain_start_min : float
    Same as ``domain_start_sec`` but converted to minutes (``domain_start_sec / 60``).

segment_fhr_quality : float, range [0, 1]
    Mean of the full-resolution (5760-sample) FHR validity weight for this segment.
    Computed as ``mean(fhr != 0)`` over all 5760 samples. A value of 1.0 means the
    FHR signal is present everywhere; lower values indicate gaps or signal loss.
    This is NOT the decimated 360-point weight — it uses every sample.

segment_mean_weight_decimated : float, range [0, 1]
    Mean of the decimated (360-point) sample_weight vector produced by the MIMO
    pipeline's ``calc_sample_weights()``. Each of the 360 points corresponds to a
    16-sample block of the input. Included for direct comparison with the weight
    threshold used in the HDF5 dataset creation pipeline.

segment_passes_weight_filter : bool
    True if ``segment_mean_weight_decimated >= weight_threshold`` (default 0.90).
    This mirrors the quality gate in ``create_hdf5_dataset.py`` that rejects
    segments with more than ~10% signal gaps. Segments that fail are still included
    in the CSV but flagged here.

segment_n_flat_regions : int
    Total number of UP flat regions detected in this segment. Zero if no flat
    regions were found.

segment_total_flat_duration_samples : int
    Sum of all flat region durations (in samples) within this segment.

segment_total_flat_duration_sec : float
    Sum of all flat region durations (in seconds) within this segment.

row_type : str
    Either ``'segment_summary'`` or ``'flat_region'``. Determines which other
    columns are populated vs NaN.

flat_region_idx : int
    Zero-based index of this flat region within the segment. Set to -1 for
    ``segment_summary`` rows.

flat_start_sample : int or NaN
    Start sample index (0-based, inclusive) of the flat region within the segment's
    5760-sample window. **NaN for segment_summary rows** because there is no
    specific flat region to reference.

flat_end_sample : int or NaN
    End sample index (inclusive) of the flat region. **NaN for segment_summary rows.**

flat_duration_samples : int or NaN
    Length of the flat region in samples (``flat_end_sample - flat_start_sample + 1``).
    **NaN for segment_summary rows.**

flat_duration_sec : float or NaN
    Duration of the flat region in seconds (``flat_duration_samples / sampling_rate``).
    **NaN for segment_summary rows.**

flat_abs_start_sec : float or NaN
    Absolute start time of the flat region in seconds relative to delivery.
    Computed as ``domain_start_sec + flat_start_sample / sampling_rate``.
    **NaN for segment_summary rows.**

flat_abs_end_sec : float or NaN
    Absolute end time of the flat region in seconds relative to delivery.
    Computed as ``domain_start_sec + flat_end_sample / sampling_rate``.
    **NaN for segment_summary rows.**

flat_fhr_valid_pct : float or NaN, range [0, 100]
    Percentage of samples within the flat region where FHR is valid (non-zero).
    A value near 0% indicates the flat region overlaps with a signal gap or padding
    (both UP and FHR are zero) — this is likely not a genuine UP flat region but
    rather missing data. A value near 100% indicates the FHR monitor was active
    during the flat UP period, meaning the UP flatness is genuine (no contractions
    or transducer issue). **NaN for segment_summary rows.**

flat_mean_fhr : float or NaN
    Mean FHR value during the flat region, computed only over samples where FHR is
    valid (non-zero). If no FHR samples are valid within the flat region (i.e.
    ``flat_fhr_valid_pct == 0``), this is NaN because there are no valid values to
    average. **Also NaN for segment_summary rows.**

flat_mean_up : float or NaN
    Mean UP value during the flat region (all samples, including zeros).
    **NaN for segment_summary rows.**

flat_excluded_by_fhr_filter : bool
    True if this flat region was flagged as likely gap/padding rather than genuine
    flat UP. The flag is set when ``exclude_invalid_fhr_flat_regions=True`` (default)
    AND ``flat_fhr_valid_pct < fhr_valid_threshold`` (default 50%). The row is still
    present in the CSV but downstream consumers can filter on this flag.
    For ``segment_summary`` rows this is always False.
"""

import os
import argparse
import logging
import traceback

import numpy as np
import pandas as pd
from tqdm import tqdm

from early_maestra.adaptor.mimo_adaptor import EarlyMaestraMimoAdaptor
from create_hdf5_dataset import (
    find_flat_regions,
    deduplicate_segments,
    interpolate_bad_values,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Column descriptions written as comment header lines in the CSV when
# include_column_descriptions=True. Each line is prefixed with "# ".
# ---------------------------------------------------------------------------
CSV_COLUMN_DESCRIPTIONS = {
    "guid": "Recording identifier — .mat filename without extension. Uniquely identifies one CTG recording.",
    "subgroup": "Clinical outcome subgroup key, e.g. 'hie_cs', 'acidosis_no_cs', 'healthy_no_bg_no_cs'.",
    "clinical_class": "High-level clinical class: 'HEALTHY', 'ACIDOSIS', or 'HIE'.",
    "cs_label": "True if the recording involved a caesarean section delivery.",
    "bg_label": "True if a blood gas measurement was available for this recording.",
    "segment_idx": "Zero-based segment index within the recording (after deduplication).",
    "domain_start_sec": "Segment start time in seconds relative to delivery (negative = before, 0 = delivery).",
    "domain_start_min": "Segment start time in minutes relative to delivery (domain_start_sec / 60).",
    "segment_fhr_quality": "Mean full-resolution FHR validity weight [0-1]. Computed as mean(fhr != 0) over all 5760 samples.",
    "segment_mean_weight_decimated": "Mean decimated (360-point) sample_weight from MIMO calc_sample_weights(). Each point covers 16 input samples.",
    "segment_passes_weight_filter": "True if segment_mean_weight_decimated >= weight_threshold (default 0.90).",
    "segment_n_flat_regions": "Number of UP flat regions detected in this segment.",
    "segment_total_flat_duration_samples": "Sum of all flat region durations (samples) in this segment.",
    "segment_total_flat_duration_sec": "Sum of all flat region durations (seconds) in this segment.",
    "row_type": "Row type: 'segment_summary' (one per segment) or 'flat_region' (one per detected flat region).",
    "flat_region_idx": "Zero-based flat region index within the segment. -1 for segment_summary rows.",
    "flat_start_sample": "Start sample index (0-based, inclusive) of the flat region. NaN for segment_summary rows.",
    "flat_end_sample": "End sample index (inclusive) of the flat region. NaN for segment_summary rows.",
    "flat_duration_samples": "Flat region length in samples (end - start + 1). NaN for segment_summary rows.",
    "flat_duration_sec": "Flat region duration in seconds (flat_duration_samples / sampling_rate). NaN for segment_summary rows.",
    "flat_abs_start_sec": "Absolute flat region start in seconds relative to delivery. NaN for segment_summary rows.",
    "flat_abs_end_sec": "Absolute flat region end in seconds relative to delivery. NaN for segment_summary rows.",
    "flat_fhr_valid_pct": "Percent of flat region where FHR is valid (non-zero) [0-100]. Near 0 = likely gap/padding, near 100 = genuine flat UP. NaN for segment_summary rows.",
    "flat_mean_fhr": "Mean FHR during the flat region (valid samples only). NaN if no valid FHR samples or for segment_summary rows.",
    "flat_mean_up": "Mean UP value during the flat region (all samples). NaN for segment_summary rows.",
    "flat_excluded_by_fhr_filter": "True if flat region was flagged as likely gap/padding (flat_fhr_valid_pct < fhr_valid_threshold). Always False for segment_summary rows.",
}


SUBGROUP_METADATA = {
    "HEALTHY_NO_BG_NoCS": {
        "subgroup": "healthy_no_bg_no_cs",
        "clinical_class": "HEALTHY",
        "cs_label": False,
        "bg_label": False,
        "pre_defined_target": 1,
    },
    "HEALTHY_NO_BG_CS": {
        "subgroup": "healthy_no_bg_cs",
        "clinical_class": "HEALTHY",
        "cs_label": True,
        "bg_label": False,
        "pre_defined_target": 1,
    },
    "HEALTHY_NO_ACIDOSIS_CS": {
        "subgroup": "healthy_bg_cs",
        "clinical_class": "HEALTHY",
        "cs_label": True,
        "bg_label": True,
        "pre_defined_target": 1,
    },
    "HEALTHY_NO_ACIDOSIS_NoCS": {
        "subgroup": "healthy_bg_no_cs",
        "clinical_class": "HEALTHY",
        "cs_label": False,
        "bg_label": True,
        "pre_defined_target": 1,
    },
    "ACIDOSIS_NO_HIE_CS": {
        "subgroup": "acidosis_cs",
        "clinical_class": "ACIDOSIS",
        "cs_label": True,
        "bg_label": True,
        "pre_defined_target": 2,
    },
    "ACIDOSIS_NO_HIE_NoCS": {
        "subgroup": "acidosis_no_cs",
        "clinical_class": "ACIDOSIS",
        "cs_label": False,
        "bg_label": True,
        "pre_defined_target": 2,
    },
    "HIE_CS": {
        "subgroup": "hie_cs",
        "clinical_class": "HIE",
        "cs_label": True,
        "bg_label": True,
        "pre_defined_target": 3,
    },
    "HIE_NoCS": {
        "subgroup": "hie_no_cs",
        "clinical_class": "HIE",
        "cs_label": False,
        "bg_label": True,
        "pre_defined_target": 3,
    },
}


def discover_records_from_base_path(records_base_path, folder_filter=None):
    """Discover .mat EFM recording files from the standard outcome folder structure.

    Scans the 8 standard outcome subfolders (e.g. HIE_CS/EFMOut/) under the
    base path and builds a list of record dicts with clinical metadata attached.

    Args:
        records_base_path: Root directory containing the outcome subfolders.
            Expected structure: ``<base>/<FOLDER_NAME>/EFMOut/*.mat``
            where FOLDER_NAME is one of the 8 keys in SUBGROUP_METADATA
            (e.g. ``HIE_CS``, ``ACIDOSIS_NO_HIE_NoCS``, ``HEALTHY_NO_BG_CS``, etc.).
        folder_filter: Optional list of folder names to restrict processing to.
            Only folders whose names appear in this list will be scanned.
            If None, all 8 standard folders are scanned.

    Returns:
        A list of dicts, each containing:
            - ``record_path`` (str): Full path to the .mat file.
            - ``folder_name`` (str): Outcome folder name (e.g. ``'HIE_CS'``).
            - ``subgroup`` (str): Short subgroup key (e.g. ``'hie_cs'``).
            - ``clinical_class`` (str): One of ``'HEALTHY'``, ``'ACIDOSIS'``, ``'HIE'``.
            - ``cs_label`` (bool): Whether the recording involved caesarean section.
            - ``bg_label`` (bool): Whether a blood gas measurement was available.
            - ``pre_defined_target`` (int): Class index (1=HEALTHY, 2=ACIDOSIS, 3=HIE).
    """
    records = []
    folders_to_process = SUBGROUP_METADATA.keys()
    if folder_filter is not None:
        folders_to_process = [f for f in folders_to_process if f in folder_filter]

    for folder_name in sorted(folders_to_process):
        meta = SUBGROUP_METADATA[folder_name]
        efm_dir = os.path.join(records_base_path, folder_name, "EFMOut")
        if not os.path.isdir(efm_dir):
            logger.warning(f"Folder not found, skipping: {efm_dir}")
            continue
        mat_files = sorted(
            os.path.join(efm_dir, f)
            for f in os.listdir(efm_dir)
            if f.endswith(".mat")
        )
        for fpath in mat_files:
            records.append({
                "record_path": fpath,
                "folder_name": folder_name,
                "subgroup": meta["subgroup"],
                "clinical_class": meta["clinical_class"],
                "cs_label": meta["cs_label"],
                "bg_label": meta["bg_label"],
                "pre_defined_target": meta["pre_defined_target"],
            })

    logger.info(f"Discovered {len(records)} .mat files across "
                f"{len(set(r['folder_name'] for r in records))} folders")
    return records



def _process_single_record(
    record_info,
    base_block_size=3840,
    overlap_percentage=0.0,
    up_shift_secs=0.0,
    min_domain_start=-np.inf,
    max_domain_start=np.inf,
    flat_tolerance=1e-9,
    flat_min_length=20,
    weight_threshold=0.90,
    fhr_valid_threshold=50.0,
    exclude_invalid_fhr_flat_regions=True,
    sampling_rate=4.0,
):
    """Process a single .mat recording and extract all UP flat region information.

    Loads the recording via EarlyMaestraMimoAdaptor, segments it using the MIMO
    ``prepare_data`` pipeline (split_long, equalization, overlap, deduplication),
    then detects UP flat regions in each segment. For every segment a summary row
    is emitted, and for every detected flat region an additional detail row is
    emitted with timing, duration, and FHR quality context.

    The processing mirrors ``create_hdf5_dataset_from_records_list`` but skips the
    scattering transform, allowing full-resolution (5760-sample) FHR weight
    vectors instead of the decimated 360-point version.

    Args:
        record_info: Dict with keys from ``discover_records_from_base_path``:
            ``record_path``, ``folder_name``, ``subgroup``, ``clinical_class``,
            ``cs_label``, ``bg_label``, ``pre_defined_target``.
        base_block_size: Base segment length in samples before the 1.5x multiplier.
            Final segment length = floor(base_block_size * 1.5). With the default
            3840, segments are 5760 samples = 1440 sec = 24 min at 4 Hz.
        overlap_percentage: Fraction of overlap between consecutive segments
            (0.0 = no overlap, 0.5 = 50% overlap with step size = half a segment).
        up_shift_secs: Time shift applied to the UP signal in seconds. Negative
            values advance UP relative to FHR (e.g. -20 compensates for the
            physiological delay between uterine contraction and FHR response).
        min_domain_start: Earliest segment domain_start (seconds relative to
            delivery) to include. Segments starting before this are discarded.
        max_domain_start: Latest segment domain_start (seconds relative to
            delivery) to include. Use ``np.inf`` to keep all segments including
            post-delivery ones.
        flat_tolerance: Maximum absolute difference between consecutive UP samples
            for the pair to be considered "flat". Tighter values (e.g. 1e-9) detect
            only truly constant regions; looser values (e.g. 1e-3) detect
            near-constant regions.
        flat_min_length: Minimum number of consecutive flat samples required for a
            region to be reported. At 4 Hz, 20 samples = 5 sec, 960 samples = 4 min.
        weight_threshold: Minimum mean decimated sample_weight (0-1) for a segment
            to pass the quality filter. This is recorded in
            ``segment_passes_weight_filter`` but does not remove segments from output.
        fhr_valid_threshold: Minimum percentage (0-100) of FHR-valid samples within
            a flat region for it to NOT be flagged by the FHR filter. Flat regions
            with ``flat_fhr_valid_pct < fhr_valid_threshold`` are marked as
            ``flat_excluded_by_fhr_filter=True`` (likely gap/padding, not real flat UP).
        exclude_invalid_fhr_flat_regions: If True, flat regions where FHR validity
            is below ``fhr_valid_threshold`` are flagged via
            ``flat_excluded_by_fhr_filter``. If False, no flagging is applied.
        sampling_rate: Signal sampling rate in Hz. Used to convert between sample
            counts and seconds/minutes. Default 4.0 Hz.

    Returns:
        A list of row dicts. Each dict has the same set of keys (the CSV columns).
        Rows with ``row_type='segment_summary'`` are emitted once per segment.
        Rows with ``row_type='flat_region'`` are emitted once per detected flat
        region. Segment-summary rows have NaN for flat-region-specific fields.
    """
    record_path = record_info["record_path"]
    guid = os.path.splitext(os.path.basename(record_path))[0]
    subgroup = record_info["subgroup"]
    clinical_class = record_info["clinical_class"]
    cs_label = record_info["cs_label"]
    bg_label = record_info["bg_label"]
    pre_defined_target = record_info["pre_defined_target"]

    default_ti = (pre_defined_target - 1) if pre_defined_target is not None else None
    mimo_adaptor = EarlyMaestraMimoAdaptor(
        do_transpose=True,
        process_targets=True,
        n_aux_labels=None,
        signal_indices=range(0, 2),
        n_input_chan=2,
        labels=["HIE", "ACIDOSIS", "HEALTHY"],
        up_shift_secs=up_shift_secs,
        default_target_index=default_ti,
    )
    mimo_adaptor.read_single_input(
        record_path, out_dec_factor=16, out_dec_factor_offset=0,
        target_is_onehot=True, dtype=np.float32,
    )
    mimo_prepared, _ = mimo_adaptor.mimo.prepare_data(
        batch_size=1, do_evaluate=True, align_left=True,
        do_split=True, do_pad=True, do_reflect=True,
        base_length=base_block_size,
        do_equalize=True, do_merge=True,
        min_domain_start=[min_domain_start, min_domain_start],
        max_domain_start=[max_domain_start, max_domain_start],
        overlap_percentage=overlap_percentage,
    )

    up = mimo_prepared.block_input[:, :, 0].copy()   # (N, 5760)
    fhr = mimo_prepared.block_input[:, :, 1].copy()   # (N, 5760)
    domain_starts = mimo_prepared.domain_start
    sample_weights = mimo_prepared.sample_weights      # (N, 360) decimated

    if up.shape[0] == 0:
        logger.info(f"{guid}: no segments after prepare_data, skipping")
        return []

    n_bad_fhr = int((~np.isfinite(fhr)).sum())
    n_bad_up = int((~np.isfinite(up)).sum())
    if n_bad_fhr or n_bad_up:
        logger.warning(f"{guid}: interpolating {n_bad_fhr + n_bad_up} NaN/Inf "
                       f"(FHR={n_bad_fhr}, UP={n_bad_up})")
        interpolate_bad_values(fhr)
        interpolate_bad_values(up)
    fhr = np.clip(fhr, 0, 500).astype(np.float32)
    up = np.clip(up, -50, 500).astype(np.float32)
    tiny = np.finfo(np.float32).tiny
    fhr[(fhr != 0) & (np.abs(fhr) < tiny)] = 0.0
    up[(up != 0) & (np.abs(up) < tiny)] = 0.0

    fhr_weight_full = (fhr != 0).astype(np.float32)   # (N, 5760)

    keep_idx, removed_idx, _ = deduplicate_segments(domain_starts, sample_weights)
    if removed_idx:
        logger.info(f"{guid}: deduplicating {len(removed_idx)} segments")
        fhr = fhr[keep_idx]
        up = up[keep_idx]
        fhr_weight_full = fhr_weight_full[keep_idx]
        sample_weights = sample_weights[keep_idx]
        domain_starts = [domain_starts[i] for i in keep_idx]

    rows = []
    n_segments = up.shape[0]

    for seg_i in range(n_segments):
        ds_sec = float(domain_starts[seg_i])
        ds_min = ds_sec / 60.0

        seg_fhr_quality = float(np.mean(fhr_weight_full[seg_i, :]))
        seg_mean_weight_dec = float(np.mean(sample_weights[seg_i, :]))
        seg_passes_weight = seg_mean_weight_dec >= weight_threshold

        up_flat = find_flat_regions(
            up[seg_i, :], tolerance=flat_tolerance, min_length=flat_min_length,
        )

        n_flat = len(up_flat)
        total_flat_samples = sum(end - start + 1 for start, end in up_flat)
        total_flat_sec = total_flat_samples / sampling_rate

        common = {
            "guid": guid,
            "subgroup": subgroup,
            "clinical_class": clinical_class,
            "cs_label": cs_label,
            "bg_label": bg_label,
            "segment_idx": seg_i,
            "domain_start_sec": ds_sec,
            "domain_start_min": ds_min,
            "segment_fhr_quality": seg_fhr_quality,
            "segment_mean_weight_decimated": seg_mean_weight_dec,
            "segment_passes_weight_filter": seg_passes_weight,
            "segment_n_flat_regions": n_flat,
            "segment_total_flat_duration_samples": total_flat_samples,
            "segment_total_flat_duration_sec": total_flat_sec,
        }

        summary = {
            **common,
            "row_type": "segment_summary",
            "flat_region_idx": -1,
            "flat_start_sample": np.nan,
            "flat_end_sample": np.nan,
            "flat_duration_samples": np.nan,
            "flat_duration_sec": np.nan,
            "flat_abs_start_sec": np.nan,
            "flat_abs_end_sec": np.nan,
            "flat_fhr_valid_pct": np.nan,
            "flat_mean_fhr": np.nan,
            "flat_mean_up": np.nan,
            "flat_excluded_by_fhr_filter": False,
        }
        rows.append(summary)

        for fr_i, (fr_start, fr_end) in enumerate(up_flat):
            fr_len = fr_end - fr_start + 1
            fr_dur_sec = fr_len / sampling_rate

            fr_abs_start = ds_sec + fr_start / sampling_rate
            fr_abs_end = ds_sec + fr_end / sampling_rate

            fhr_slice = fhr[seg_i, fr_start:fr_end + 1]
            fhr_valid_mask = fhr_slice != 0
            n_valid = int(fhr_valid_mask.sum())
            fhr_valid_pct = 100.0 * n_valid / fr_len

            if n_valid > 0:
                flat_mean_fhr = float(np.mean(fhr_slice[fhr_valid_mask]))
            else:
                flat_mean_fhr = np.nan

            flat_mean_up = float(np.mean(up[seg_i, fr_start:fr_end + 1]))

            excluded = (
                exclude_invalid_fhr_flat_regions
                and fhr_valid_pct < fhr_valid_threshold
            )

            row = {
                **common,
                "row_type": "flat_region",
                "flat_region_idx": fr_i,
                "flat_start_sample": fr_start,
                "flat_end_sample": fr_end,
                "flat_duration_samples": fr_len,
                "flat_duration_sec": fr_dur_sec,
                "flat_abs_start_sec": fr_abs_start,
                "flat_abs_end_sec": fr_abs_end,
                "flat_fhr_valid_pct": fhr_valid_pct,
                "flat_mean_fhr": flat_mean_fhr,
                "flat_mean_up": flat_mean_up,
                "flat_excluded_by_fhr_filter": excluded,
            }
            rows.append(row)

    return rows


def analyze_up_flat_regions(
    records=None,
    records_base_path=None,
    output_csv_path="up_flat_analysis.csv",
    base_block_size=3840,
    overlap_percentage=0.5,
    up_shift_secs=-20.0,
    min_domain_start=-44640.0,
    max_domain_start=np.inf,
    flat_tolerance=1e-9,
    flat_min_length=20,
    weight_threshold=0.90,
    fhr_valid_threshold=50.0,
    exclude_invalid_fhr_flat_regions=True,
    sampling_rate=4.0,
    folder_filter=None,
    file_limit=-1,
    include_column_descriptions=True,
):
    """Analyze UP flat regions across all EFM recordings and save results to CSV.

    Orchestrates the full analysis pipeline: discovers .mat files, processes each
    recording to detect UP flat regions (with FHR quality context), collects all
    results into a single DataFrame, and saves to CSV.

    Either ``records`` (a pre-built list) or ``records_base_path`` (to auto-discover
    files) must be provided. If both are given, ``records`` takes precedence.

    Args:
        records: Pre-built list of record dicts as returned by
            ``discover_records_from_base_path``. If provided, ``records_base_path``
            and ``folder_filter`` are ignored.
        records_base_path: Root directory containing outcome subfolders. Used to
            auto-discover .mat files when ``records`` is None.
        output_csv_path: Path where the output CSV will be written.
        base_block_size: Base segment length in samples before the 1.5x multiplier.
            Final segment length = floor(base_block_size * 1.5). With the default
            3840, segments are 5760 samples = 1440 sec = 24 min at 4 Hz.
        overlap_percentage: Fraction of overlap between consecutive segments
            (0.0 = no overlap, 0.5 = 50% overlap).
        up_shift_secs: Time shift applied to the UP signal in seconds. Negative
            values advance UP relative to FHR (e.g. -20 compensates for the
            physiological delay between contraction and FHR response).
        min_domain_start: Earliest segment domain_start (seconds relative to
            delivery) to include. Default -44640 sec (~12.4 hours before delivery).
        max_domain_start: Latest segment domain_start (seconds relative to
            delivery) to include. Use ``np.inf`` to keep all including post-delivery.
        flat_tolerance: Maximum absolute difference between consecutive UP samples
            for the pair to be considered "flat". Default 1e-9 detects only truly
            constant regions.
        flat_min_length: Minimum number of consecutive flat samples to qualify as a
            flat region. At 4 Hz: 20 samples = 5 sec, 960 = 4 min, 1200 = 5 min.
        weight_threshold: Minimum mean decimated sample_weight (0-1) for a segment
            to be flagged as passing the quality filter in the output.
        fhr_valid_threshold: Minimum percentage (0-100) of FHR-valid samples within
            a flat region to NOT be flagged by the FHR filter.
        exclude_invalid_fhr_flat_regions: If True, flat regions where FHR validity
            is below ``fhr_valid_threshold`` are flagged as likely gap/padding.
        sampling_rate: Signal sampling rate in Hz. Default 4.0 Hz.
        folder_filter: List of folder names to restrict file discovery to
            (e.g. ``['HIE_CS', 'ACIDOSIS_NO_HIE_CS']``). Only used when
            ``records`` is None. If None, all 8 standard folders are scanned.
        file_limit: Maximum number of .mat files to process per subgroup. Use -1
            for no limit. Useful for quick testing on a subset.
        include_column_descriptions: If True (default), write comment lines at the
            top of the CSV describing each column before the header row and data.
            Each description line is prefixed with ``# ``. If False, the CSV
            contains only the standard header row followed by data rows.

    Returns:
        A pandas DataFrame with one row per flat region plus one segment-summary
        row per segment. Key columns include:

        - ``guid``, ``subgroup``, ``clinical_class``, ``cs_label``, ``bg_label``
        - ``segment_idx``, ``domain_start_sec``, ``domain_start_min``
        - ``segment_fhr_quality``, ``segment_mean_weight_decimated``
        - ``row_type`` (``'segment_summary'`` or ``'flat_region'``)
        - ``flat_region_idx``, ``flat_start_sample``, ``flat_end_sample``
        - ``flat_duration_samples``, ``flat_duration_sec``
        - ``flat_abs_start_sec``, ``flat_abs_end_sec`` (relative to delivery)
        - ``flat_fhr_valid_pct``, ``flat_mean_fhr``, ``flat_mean_up``
        - ``flat_excluded_by_fhr_filter``

        The DataFrame is also saved to ``output_csv_path``.
    """
    if records is None:
        if records_base_path is None:
            raise ValueError("Provide either records or records_base_path")
        records = discover_records_from_base_path(records_base_path, folder_filter)

    if not records:
        logger.warning("No records to process")
        return pd.DataFrame()

    if file_limit > 0:
        from collections import defaultdict
        by_subgroup = defaultdict(list)
        for r in records:
            by_subgroup[r["subgroup"]].append(r)
        limited = []
        for sg_records in by_subgroup.values():
            limited.extend(sg_records[:file_limit])
        records = limited
        logger.info(f"After file_limit={file_limit}: {len(records)} records")

    all_rows = []
    errors = []

    for rec in tqdm(records, desc="Analyzing UP flat regions"):
        try:
            rows = _process_single_record(
                rec,
                base_block_size=base_block_size,
                overlap_percentage=overlap_percentage,
                up_shift_secs=up_shift_secs,
                min_domain_start=min_domain_start,
                max_domain_start=max_domain_start,
                flat_tolerance=flat_tolerance,
                flat_min_length=flat_min_length,
                weight_threshold=weight_threshold,
                fhr_valid_threshold=fhr_valid_threshold,
                exclude_invalid_fhr_flat_regions=exclude_invalid_fhr_flat_regions,
                sampling_rate=sampling_rate,
            )
            all_rows.extend(rows)
        except Exception:
            guid = os.path.splitext(os.path.basename(rec["record_path"]))[0]
            logger.error(f"Failed processing {guid}:\n{traceback.format_exc()}")
            errors.append(rec["record_path"])

    if errors:
        logger.warning(f"{len(errors)} files failed processing")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        if include_column_descriptions:
            with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
                # Write column description header as comment lines
                f.write("# UP Flat Region Analysis — Column Descriptions\n")
                f.write("#\n")
                for col in df.columns:
                    desc = CSV_COLUMN_DESCRIPTIONS.get(col, "No description available.")
                    f.write(f"# {col}: {desc}\n")
                f.write("#\n")
                # Write the actual CSV data (header + rows) after the comments
                df.to_csv(f, index=False)
        else:
            df.to_csv(output_csv_path, index=False)
        logger.info(f"Saved {len(df)} rows to {output_csv_path}")

        n_summaries = (df["row_type"] == "segment_summary").sum()
        n_flat = (df["row_type"] == "flat_region").sum()
        n_guids = df["guid"].nunique()
        logger.info(f"  {n_guids} GUIDs, {n_summaries} segments, {n_flat} flat regions")
    else:
        logger.warning("No rows produced — empty DataFrame")

    return df


def main():
    """CLI entry point for the UP flat region analysis tool.

    Parses command-line arguments and delegates to ``analyze_up_flat_regions``.
    All parameters of the analysis function are exposed as CLI flags.
    """
    parser = argparse.ArgumentParser(
        description="Analyze UP flat regions across EFM recordings.",
    )
    parser.add_argument("--records_base_path", type=str, required=True,
                        help="Base path containing outcome subfolders")
    parser.add_argument("--output_csv", type=str, default="up_flat_analysis.csv",
                        help="Output CSV path (default: up_flat_analysis.csv)")
    parser.add_argument("--base_block_size", type=int, default=3840)
    parser.add_argument("--overlap_percentage", type=float, default=0.0)
    parser.add_argument("--up_shift_secs", type=float, default=0.0)
    parser.add_argument("--min_domain_start", type=float, default=-44640.0)
    parser.add_argument("--max_domain_start", type=float, default=float("inf"))
    parser.add_argument("--flat_tolerance", type=float, default=1e-9)
    parser.add_argument("--flat_min_length", type=int, default=960)
    parser.add_argument("--weight_threshold", type=float, default=0.90)
    parser.add_argument("--fhr_valid_threshold", type=float, default=50.0)
    parser.add_argument("--no_exclude_invalid_fhr", action="store_true",
                        help="Disable FHR-validity flagging of flat regions")
    parser.add_argument("--sampling_rate", type=float, default=4.0)
    parser.add_argument("--folder_filter", nargs="+", default=None,
                        help="Folder names to process (default: all 8)")
    parser.add_argument("--file_limit", type=int, default=-1,
                        help="Max files per subgroup (-1 = all)")
    parser.add_argument("--no_column_descriptions", action="store_true",
                        help="Omit column description comments from CSV header")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    analyze_up_flat_regions(
        records_base_path=args.records_base_path,
        output_csv_path=args.output_csv,
        base_block_size=args.base_block_size,
        overlap_percentage=args.overlap_percentage,
        up_shift_secs=args.up_shift_secs,
        min_domain_start=args.min_domain_start,
        max_domain_start=args.max_domain_start,
        flat_tolerance=args.flat_tolerance,
        flat_min_length=args.flat_min_length,
        weight_threshold=args.weight_threshold,
        fhr_valid_threshold=args.fhr_valid_threshold,
        exclude_invalid_fhr_flat_regions=not args.no_exclude_invalid_fhr,
        sampling_rate=args.sampling_rate,
        folder_filter=args.folder_filter,
        file_limit=args.file_limit,
        include_column_descriptions=not args.no_column_descriptions,
    )


if __name__ == "__main__":
    main()
