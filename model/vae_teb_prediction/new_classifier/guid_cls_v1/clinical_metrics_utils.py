"""
Clinical decision rule, time-binned metric, threshold-search and plotting
utilities for the GUID-level classifier (``guid_cls_v1``) evaluation pipeline.

This module is **self-contained**: it has zero coupling to the legacy
``vae_teb_prediction.classifier`` / ``vae_teb_prediction.new_classifier``
model code. It was forked from the previous ``new_classifier/evaluate_classifier.py``
with all legacy-model orchestration helpers (``create_model_from_config``,
``run_inference``, ``_evaluate_single_fold``, ``_run_inference_for_fold``,
``main``, ``aggregate_existing_results``, ``load_best_checkpoint``,
``find_latest_checkpoint_in_fold``, the ``__main__`` driver) stripped out.

The retained surface is purely DataFrame / numpy / matplotlib utilities:

- Clinical decision rule (``apply_clinical_decision_rule``,
  ``fill_missing_epochs``, ``ensure_committed_epochs_filled``,
  ``verify_clinical_decision_rule``).
- Threshold finders at FPR=target @ decision-time
  (``find_threshold_for_{instantaneous,committed_cumulative,
  committed_overall}_fpr_at_1h``).
- Time-binned metric computers
  (``compute_{instantaneous,committed_cumulative,committed_overall}_metrics``,
  ``compute_subgroup_metrics_by_time``).
- ROC helpers (``compute_guid_level_roc``, ``compute_committed_cumulative_roc``,
  ``plot_roc_curve``, ``plot_aggregated_roc_curves``).
- Per-fold dataset stats (``generate_fold_dataset_stats`` and its private
  plotters).
- Subgroup analysis (``create_enhanced_subgroup_filters``,
  ``compute_subgroup_statistics``, ``plot_subgroup_analysis``).
- Cross-fold aggregators
  (``_aggregate_*``, ``plot_aggregated_metric_type``,
  ``plot_aggregated_subgroup_comparison``, ``generate_aggregated_plots``).
- Validation helpers inlined from the old ``validation_utils`` module
  (``ensure_epoch_hours``, ``validate_predictions_df``,
  ``validate_guid_consistency``, ``log_dataframe_stats``,
  ``verify_clinical_decision_rule``).

Saves CSV schema (used by callers): ``guid, epoch, target, binary_target,
predicted_class, prob_class_0, prob_class_1`` (plus pipeline-specific
columns added by the caller).
"""

import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from loguru import logger
import yaml
import json
from datetime import datetime
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ModuleNotFoundError:  # seaborn is optional; plots fall back to matplotlib
    sns = None
from sklearn.metrics import roc_curve, auc, confusion_matrix


# ---------------------------------------------------------------------------
# Validation helpers (inlined from the old ``validation_utils`` module so this
# module has no legacy coupling). These are leaf functions: pure pandas /
# numpy / loguru.
# ---------------------------------------------------------------------------


def ensure_epoch_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has ``epoch_hours`` column.

    Epochs are stored as negative seconds before delivery; ``epoch_hours``
    is positive hours before delivery (so ``epoch=-43200s -> 12.0h``).
    Post-delivery epochs (``epoch > 0``) become negative ``epoch_hours``,
    which keeps them distinguishable from pre-delivery rows.

    Args:
        df: DataFrame with an ``epoch`` column (negative seconds before
            delivery).

    Returns:
        DataFrame with ``epoch_hours`` column (positive hours before
        delivery). Returns input unchanged if the column already exists.

    Raises:
        ValueError: If ``epoch`` column is missing.
    """
    if 'epoch_hours' in df.columns:
        return df
    if 'epoch' not in df.columns:
        raise ValueError("DataFrame missing required 'epoch' column for time conversion")
    df = df.copy()
    df['epoch_hours'] = -df['epoch'].astype(float) / 3600
    logger.debug(f"Added epoch_hours column (converted from epoch): {len(df)} rows")
    return df


def validate_predictions_df(df: pd.DataFrame, data_type: str = "predictions") -> None:
    """Validate predictions DataFrame structure and value ranges.

    The schema check is head-aware: the binary-head block
    (``binary_target`` / ``prob_class_1``) is only enforced when those
    columns are present. Under a 3-class-only run the binary columns
    are intentionally omitted by ``run_inference_per_position`` and
    there is nothing to validate on the binary side. The always-
    present columns (``guid``, ``epoch``) are required unconditionally.

    Args:
        df: Predictions DataFrame to validate.
        data_type: Label for error messages (e.g. ``"Validation"``).

    Raises:
        ValueError: On missing always-required columns or out-of-range
            values in the binary block (when that block is present).
    """
    always_required = ['guid', 'epoch']
    missing_always = set(always_required) - set(df.columns)
    if missing_always:
        raise ValueError(
            f"{data_type} DataFrame missing required columns: {missing_always}. "
            f"Available columns: {list(df.columns)}"
        )

    has_binary = {'binary_target', 'prob_class_1'}.issubset(df.columns)
    if has_binary:
        if not df['binary_target'].isin([0, 1]).all():
            invalid_vals = df['binary_target'].unique()
            invalid_count = (~df['binary_target'].isin([0, 1])).sum()
            raise ValueError(
                f"{data_type} has {invalid_count} samples with invalid binary_target values: "
                f"{invalid_vals} (expected 0 or 1 only)"
            )
        if not ((df['prob_class_1'] >= 0) & (df['prob_class_1'] <= 1)).all():
            invalid_count = ((df['prob_class_1'] < 0) | (df['prob_class_1'] > 1)).sum()
            min_val = df['prob_class_1'].min()
            max_val = df['prob_class_1'].max()
            raise ValueError(
                f"{data_type} has {invalid_count} samples with prob_class_1 values outside [0, 1] range. "
                f"Range found: [{min_val:.4f}, {max_val:.4f}]"
            )

    duplicates = df.duplicated(subset=['guid', 'epoch'])
    if duplicates.any():
        n_dup = duplicates.sum()
        logger.warning(
            f"{data_type} has {n_dup} duplicate (guid, epoch) pairs. "
            f"This may indicate data quality issues."
        )


def validate_guid_consistency(df: pd.DataFrame) -> None:
    """Warn if any GUID has inconsistent ``binary_target`` values across epochs.

    A GUID's outcome should be constant across all of its time epochs.
    Inconsistencies indicate upstream data-quality issues.
    """
    if 'guid' not in df.columns or 'binary_target' not in df.columns:
        logger.debug("Skipping GUID consistency check: required columns not present")
        return
    guid_targets = df.groupby('guid')['binary_target'].nunique()
    inconsistent_guids = guid_targets[guid_targets > 1]
    if len(inconsistent_guids) > 0:
        logger.warning(
            f"Found {len(inconsistent_guids)} GUIDs with inconsistent binary_target values across epochs. "
            f"Examples: {list(inconsistent_guids.index[:5])}. "
            f"This may indicate data quality issues - a GUID should have consistent outcome."
        )
        example_guid = inconsistent_guids.index[0]
        example_data = df[df['guid'] == example_guid][['epoch', 'binary_target', 'target']].sort_values('epoch')
        logger.debug(f"Example inconsistent GUID {example_guid}:\n{example_data.head(10)}")


def log_dataframe_stats(df: pd.DataFrame, label: str) -> None:
    """Log sample / GUID / class-distribution / epoch-range stats for a DataFrame."""
    logger.info(f"{label} Statistics:")
    logger.info(f"  Total samples: {len(df)}")
    if 'guid' in df.columns:
        n_guids = df['guid'].nunique()
        logger.info(f"  Unique GUIDs: {n_guids}")
        epochs_per_guid = df.groupby('guid').size()
        logger.info(
            f"  Epochs per GUID: mean={epochs_per_guid.mean():.1f}, "
            f"median={epochs_per_guid.median():.0f}, "
            f"min={epochs_per_guid.min()}, max={epochs_per_guid.max()}"
        )
    if 'binary_target' in df.columns:
        class_dist = df['binary_target'].value_counts().to_dict()
        total = len(df)
        class_0_pct = class_dist.get(0, 0) / total * 100 if total > 0 else 0
        class_1_pct = class_dist.get(1, 0) / total * 100 if total > 0 else 0
        logger.info(
            f"  Class distribution: "
            f"class_0={class_dist.get(0, 0)} ({class_0_pct:.1f}%), "
            f"class_1={class_dist.get(1, 0)} ({class_1_pct:.1f}%)"
        )
    if 'epoch' in df.columns:
        furthest_hours = -df['epoch'].min() / 3600
        nearest_hours = -df['epoch'].max() / 3600
        logger.info(
            f"  Epoch range: [{df['epoch'].min():.1f}, {df['epoch'].max():.1f}] seconds "
            f"({furthest_hours:.1f}h to {nearest_hours:.1f}h before birth)"
        )
    if 'is_filled' in df.columns:
        n_filled = df['is_filled'].sum()
        pct = n_filled / len(df) * 100 if len(df) > 0 else 0
        logger.info(f"  Filled epochs: {n_filled} ({pct:.1f}%)")
    if 'clinical_pred' in df.columns:
        n_positive = (df['clinical_pred'] == 1).sum()
        pct_positive = n_positive / len(df) * 100 if len(df) > 0 else 0
        logger.info(f"  Positive predictions: {n_positive} ({pct_positive:.1f}%)")


def verify_clinical_decision_rule(df: pd.DataFrame, label: str = "Clinical") -> None:
    """Verify that GUID-level forward-filling of ``clinical_pred`` is correct.

    Once a GUID is detected as unhealthy, all subsequent epochs (closer to
    delivery) must remain marked as unhealthy. All earlier epochs must be
    marked as healthy. ``first_detection_epoch`` must be consistent across
    all rows of a GUID.

    Args:
        df: DataFrame with ``clinical_pred`` and ``first_detection_epoch``
            columns.
        label: Label for error messages.

    Raises:
        ValueError: If forward-filling is incorrect.
    """
    required_cols = ['guid', 'epoch', 'clinical_pred', 'first_detection_epoch']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        logger.warning(
            f"{label}: Cannot verify clinical decision rule - missing columns: {missing_cols}"
        )
        return
    if len(df) == 0:
        logger.warning(f"{label}: Cannot verify clinical decision rule - empty DataFrame")
        return

    total_guids = df['guid'].nunique()
    guids_checked = 0
    violations_found = 0
    violation_details = []

    for guid in df['guid'].unique():
        guid_mask = df['guid'] == guid
        guid_data = df.loc[guid_mask].copy()
        guid_data = guid_data.sort_values('epoch', ascending=False)
        if len(guid_data) <= 1:
            continue
        guids_checked += 1
        first_detections = guid_data['first_detection_epoch'].dropna().unique()
        if len(first_detections) == 0 or guid_data['first_detection_epoch'].isna().all():
            if not (guid_data['clinical_pred'] == 0).all():
                violations_found += 1
                n_positive = (guid_data['clinical_pred'] == 1).sum()
                violation_details.append({
                    'guid': guid,
                    'issue': 'no_detection_but_positive_preds',
                    'details': f"{n_positive}/{len(guid_data)} epochs marked positive despite no detection"
                })
            continue
        if len(first_detections) > 1:
            violations_found += 1
            violation_details.append({
                'guid': guid,
                'issue': 'inconsistent_first_detection',
                'details': f"Multiple first_detection_epoch values: {first_detections}"
            })
            continue
        first_detection_epoch = first_detections[0]
        epochs_after = guid_data[guid_data['epoch'] >= first_detection_epoch]
        if not (epochs_after['clinical_pred'] == 1).all():
            violations_found += 1
            n_wrong = (epochs_after['clinical_pred'] == 0).sum()
            violation_details.append({
                'guid': guid,
                'issue': 'missing_forward_fill_after_detection',
                'details': (
                    f"{n_wrong}/{len(epochs_after)} epochs after first detection "
                    f"(epoch>={first_detection_epoch:.1f}) marked as healthy (should be unhealthy)"
                )
            })
        epochs_before = guid_data[guid_data['epoch'] < first_detection_epoch]
        if len(epochs_before) > 0 and not (epochs_before['clinical_pred'] == 0).all():
            violations_found += 1
            n_wrong = (epochs_before['clinical_pred'] == 1).sum()
            violation_details.append({
                'guid': guid,
                'issue': 'incorrect_positive_before_detection',
                'details': (
                    f"{n_wrong}/{len(epochs_before)} epochs before first detection "
                    f"(epoch<{first_detection_epoch:.1f}) marked as unhealthy (should be healthy)"
                )
            })

    if violations_found == 0:
        logger.info(
            f"{label}: Clinical decision rule verification PASSED "
            f"({guids_checked}/{total_guids} GUIDs checked, 0 violations)"
        )
    else:
        logger.error(
            f"{label}: Clinical decision rule verification FAILED "
            f"({violations_found}/{guids_checked} GUIDs with violations)"
        )
        for i, violation in enumerate(violation_details[:5]):
            logger.error(
                f"  Violation {i+1}: GUID={violation['guid']}, "
                f"Issue={violation['issue']}, Details={violation['details']}"
            )
        if len(violation_details) > 5:
            logger.error(f"  ... and {len(violation_details) - 5} more violations")
        raise ValueError(
            f"{label}: Clinical decision rule verification failed with {violations_found} violations. "
            f"Forward-filling may not be applied correctly."
        )


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Any object (dict, list, numpy type, or primitive)

    Returns:
        Object with all numpy types converted to Python natives
    """
    # Handle None first
    if obj is None:
        return None
    # Handle dicts and lists (recurse into nested structures)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    # Handle pandas DataFrame and Series (convert to serializable form before pd.isna check)
    elif isinstance(obj, pd.DataFrame):
        return convert_numpy_types(obj.to_dict('records'))
    elif isinstance(obj, pd.Series):
        return convert_numpy_types(obj.to_list())
    # Handle numpy arrays (convert to list, which will recurse)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(item) for item in obj.tolist()]
    # Handle scalar numpy types
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        val = float(obj)
        # Handle NaN and inf/-inf
        if np.isnan(val) or np.isinf(val):
            return None  # Convert to null for JSON compatibility
        return val
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Handle Python float NaN/inf
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    else:
        # Handle pandas NaT and other pd.NA types (scalar only)
        try:
            if pd.isna(obj):
                return None
        except (ValueError, TypeError):
            pass
        return obj


def compute_epoch_intervals(guid_epochs: np.ndarray) -> float:
    """
    Compute typical epoch interval for a GUID.

    Since epoch intervals are variable (not fixed at 20 minutes), we compute
    the median interval for robustness to outliers.

    Args:
        guid_epochs: Sorted array of epoch values (in descending order: far → near birth)

    Returns:
        Median interval between consecutive epochs in seconds
    """
    if guid_epochs is None or len(guid_epochs) < 2:
        return 0.0  # Cannot compute interval with single epoch

    # Epochs are seconds before birth (typically negative). Sort far->near birth.
    epochs_sorted = np.sort(np.unique(np.asarray(guid_epochs, dtype=float)))
    intervals = np.diff(epochs_sorted)
    intervals = intervals[intervals > 0]

    # Return median interval (robust to outliers)
    median_interval = float(np.median(intervals)) if len(intervals) else 0.0

    return median_interval


def infer_epoch_interval_seconds(df: pd.DataFrame) -> float:
    """
    Infer a typical epoch interval from the data (in seconds).

    Uses the median of per-GUID median intervals to be robust to outliers and
    intermittent missing epochs.
    """
    if df is None or len(df) < 2 or 'epoch' not in df.columns:
        return 0.0

    # Use only original (non-filled) epochs to infer the dataset grid.
    if 'is_filled' in df.columns:
        df = df[df['is_filled'] == False]  # noqa: E712
        if len(df) < 2:
            return 0.0

    if 'guid' not in df.columns:
        epochs = np.sort(np.unique(df['epoch'].values.astype(float)))
        return float(np.median(np.diff(epochs))) if len(epochs) >= 2 else 0.0

    # Robust grid step: take the MODE of per-GUID consecutive deltas (rounded to 1s).
    deltas: List[float] = []
    for _, g in df.groupby('guid', sort=False):
        epochs_sorted = np.sort(np.unique(g['epoch'].values.astype(float)))
        if len(epochs_sorted) < 2:
            continue
        d = np.diff(epochs_sorted)
        d = d[d > 0]
        if len(d):
            deltas.extend(np.round(d, 0).tolist())

    if len(deltas) >= 10:
        vc = pd.Series(deltas).value_counts()
        return float(vc.index[0])

    # Fallback: median of per-GUID medians.
    medians: List[float] = []
    for _, g in df.groupby('guid', sort=False):
        interval = compute_epoch_intervals(g['epoch'].values)
        if interval > 0:
            medians.append(interval)

    return float(np.median(medians)) if medians else 0.0


def guid_snapshot_at_or_before_time(
    df: pd.DataFrame,
    decision_time_hours: float,
    pred_col: str = 'clinical_pred'
) -> pd.DataFrame:
    """
    Return one row per GUID with the prediction state available at decision_time_hours.

    At time T hours before delivery, only epochs with start >= T are available; the
    state is the latest available epoch at/just before T (closest to birth but not after).

    Returns columns: guid, binary_target, pred_at_time
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=['guid', 'binary_target', 'pred_at_time'])

    df_in = ensure_epoch_hours(df.copy())
    if 'guid' not in df_in.columns or 'binary_target' not in df_in.columns or pred_col not in df_in.columns:
        return pd.DataFrame(columns=['guid', 'binary_target', 'pred_at_time'])

    available = df_in[df_in['epoch_hours'] >= decision_time_hours].copy()
    if len(available) == 0:
        return pd.DataFrame(columns=['guid', 'binary_target', 'pred_at_time'])

    idx = available.groupby('guid')['epoch_hours'].idxmin()
    snap = available.loc[idx, ['guid', pred_col]].rename(columns={pred_col: 'pred_at_time'})
    truth = df_in.groupby('guid', sort=False)['binary_target'].max().rename('binary_target')
    return snap.set_index('guid').join(truth, how='inner').reset_index()


def apply_clinical_decision_rule(
    df: pd.DataFrame,
    threshold: float,
    verify: bool = True
) -> pd.DataFrame:
    """
    Apply clinical decision rule: once a baby is detected as unhealthy,
    all subsequent epochs (closer to birth) are labeled as unhealthy.

    Args:
        df: Predictions dataframe with columns: guid, epoch, binary_target, prob_class_1, etc.
        threshold: Classification threshold

    Returns:
        Enhanced dataframe with additional columns:
        - model_pred: Original model prediction (0/1) based on threshold
        - clinical_pred: Prediction after clinical decision rule (0/1)
        - first_detection_epoch: Epoch where first positive detected (NaN if never detected)
    """
    df = df.copy()

    # Add model prediction based on threshold
    df['model_pred'] = (df['prob_class_1'] >= threshold).astype(int)

    # Initialize clinical prediction and first detection columns
    df['clinical_pred'] = df['model_pred'].copy()
    df['first_detection_epoch'] = np.nan

    # Process each GUID separately
    for guid in df['guid'].unique():
        guid_mask = df['guid'] == guid
        guid_data = df.loc[guid_mask].copy()

        # Epochs are seconds before birth (typically negative):
        # more negative = further from birth. Sort far -> near birth.
        guid_data = guid_data.sort_values('epoch', ascending=True)

        # Apply forward-filling logic
        detected = False
        first_detection = np.nan
        clinical_preds = []

        for idx, row in guid_data.iterrows():
            if row['model_pred'] == 1 and not detected:
                # First detection
                detected = True
                first_detection = row['epoch']

            # Once detected, all subsequent epochs are positive
            clinical_preds.append(1 if detected else 0)

        # Update dataframe with clinical predictions
        guid_data['clinical_pred'] = clinical_preds
        guid_data['first_detection_epoch'] = first_detection if detected else np.nan

        # Update original dataframe
        # CRITICAL FIX: Use index-aware assignment (without .values) to preserve alignment
        # guid_data is sorted, but pandas will match rows by index
        df.loc[guid_mask, 'clinical_pred'] = guid_data['clinical_pred']
        df.loc[guid_mask, 'first_detection_epoch'] = guid_data['first_detection_epoch']

    if verify:
        logger.info(
            f"Clinical decision rule applied. "
            f"Model positives: {df['model_pred'].sum()}, "
            f"Clinical positives: {df['clinical_pred'].sum()}"
        )

        # Verify clinical decision rule applied correctly. ``verify_clinical_decision_rule``
        # is defined at module top of this file (inlined from the legacy ``validation_utils``).
        verify_clinical_decision_rule(df, "Clinical Decision Rule")

    return df


def fill_missing_epochs(
    df: pd.DataFrame,
    max_gap_multiplier: Optional[float] = None,
    log_summary: bool = True,
    fill_until_birth: bool = False,
    birth_epoch_seconds: float = 0.0
) -> pd.DataFrame:
    """
    Fill missing epochs for each GUID using forward-filling strategy.

    Args:
        df: Predictions dataframe with clinical decision rule applied
        max_gap_multiplier: If provided, only fill gaps <= multiplier * typical_interval.
                           If None, fill all missing epochs within each GUID range.
        fill_until_birth: If True, extend each GUID to birth_epoch_seconds using forward-fill.
        birth_epoch_seconds: Epoch value (seconds before birth) treated as birth (default 0.0).

    Returns:
        Complete dataframe with all epochs (filled + original)
        New column 'is_filled': True if epoch was missing and filled
    """
    df = df.copy()
    df['is_filled'] = False

    all_rows = []
    total_skipped = 0  # FIX #6: Track skipped epochs across all GUIDs

    # Process each GUID separately
    for guid in df['guid'].unique():
        guid_mask = df['guid'] == guid
        guid_data = df.loc[guid_mask].copy()

        # Sort far -> near birth (more negative -> less negative)
        guid_data = guid_data.sort_values('epoch', ascending=True)
        epochs = guid_data['epoch'].values.astype(float)

        # Compute typical interval for this GUID
        typical_interval = compute_epoch_intervals(epochs)

        if typical_interval == 0 or len(epochs) < 2:
            # Skip filling if only one epoch or interval cannot be computed
            all_rows.append(guid_data)
            continue

        max_gap = typical_interval * max_gap_multiplier if max_gap_multiplier is not None else float('inf')

        # Identify missing epochs
        min_epoch = epochs.min()
        max_epoch = epochs.max()
        fill_end_epoch = max_epoch
        if fill_until_birth:
            fill_end_epoch = max(fill_end_epoch, birth_epoch_seconds)

        # FIX #7: Use np.arange with explicit step instead of linspace
        # This ensures spacing equals typical_interval exactly
        expected_epochs = np.arange(min_epoch, fill_end_epoch + typical_interval/2, typical_interval)

        # Round to avoid floating point issues
        expected_epochs = np.round(expected_epochs, 1)
        if fill_until_birth:
            birth_epoch = float(np.round(birth_epoch_seconds, 1))
            expected_epochs = np.unique(np.append(expected_epochs, birth_epoch))
        existing_epochs_set = set(np.round(epochs, 1))

        # Find missing epochs (only fill gaps <= max_gap)
        filled_rows = []
        guid_data_dict = guid_data.set_index('epoch').to_dict('index')
        skipped_epochs = []  # FIX #6: Track per-GUID skipped epochs

        for i, exp_epoch in enumerate(expected_epochs):
            if exp_epoch in existing_epochs_set:
                # Epoch exists, use original data
                closest_epoch = epochs[np.argmin(np.abs(epochs - exp_epoch))]
                row_data = dict(guid_data_dict[closest_epoch])
                row_data['epoch'] = float(closest_epoch)
                row_data['is_filled'] = False
                filled_rows.append(row_data)
            else:
                # Missing epoch - check if gap is within acceptable range
                if i == 0:
                    # First epoch missing - default to healthy
                    filled_row = {
                        'guid': guid,
                        'epoch': float(exp_epoch),
                        'cs_label': guid_data.iloc[0]['cs_label'] if 'cs_label' in guid_data.columns else None,
                        'bg_label': guid_data.iloc[0]['bg_label'] if 'bg_label' in guid_data.columns else None,
                        'binary_target': 0,  # Default to healthy
                        'target': 1,  # Healthy label
                        'prob_class_0': np.nan,
                        'prob_class_1': np.nan,
                        'model_pred': 0,
                        'clinical_pred': 0,
                        'first_detection_epoch': np.nan,
                        'is_filled': True
                    }
                    filled_rows.append(filled_row)
                else:
                    # Find previous epoch (next smaller epoch value, further from birth)
                    # For negative epochs: more negative = earlier in time
                    previous_epochs = epochs[epochs < exp_epoch]
                    if len(previous_epochs) > 0:
                        previous_epoch = previous_epochs.max()
                        gap = exp_epoch - previous_epoch

                        if gap <= max_gap:
                            # Fill from previous epoch (further from birth)
                            previous_data = dict(guid_data_dict[previous_epoch])
                            filled_row = {
                                'guid': guid,
                                'epoch': float(exp_epoch),
                                'cs_label': previous_data.get('cs_label'),
                                'bg_label': previous_data.get('bg_label'),
                                'binary_target': previous_data['binary_target'],
                                'target': previous_data['target'],
                                'prob_class_0': np.nan,
                                'prob_class_1': np.nan,
                                'model_pred': previous_data['model_pred'],
                                'clinical_pred': previous_data['clinical_pred'],
                                'first_detection_epoch': previous_data.get('first_detection_epoch', np.nan),
                                'is_filled': True
                            }
                            filled_rows.append(filled_row)
                        else:
                            # FIX #6: Log skipped epoch (gap too large)
                            skipped_epochs.append({
                                'epoch': exp_epoch,
                                'gap_seconds': gap,
                                'max_gap_seconds': max_gap
                            })

        # FIX #6: Log skipped epochs for this GUID
        if skipped_epochs and log_summary:
            total_skipped += len(skipped_epochs)
            logger.debug(
                f"GUID {guid}: Skipped {len(skipped_epochs)} epochs due to large gaps. "
                f"Example: epoch {skipped_epochs[0]['epoch']:.1f}s, "
                f"gap {skipped_epochs[0]['gap_seconds']:.1f}s > max {max_gap:.1f}s "
                f"(no suitable previous epoch within acceptable range)"
            )

        # Convert filled rows to dataframe
        if filled_rows:
            guid_complete = pd.DataFrame(filled_rows)
            all_rows.append(guid_complete)
        else:
            all_rows.append(guid_data)

    # Combine all GUIDs
    result_df = pd.concat(all_rows, ignore_index=True)

    # Sort by GUID and epoch
    result_df = result_df.sort_values(['guid', 'epoch'], ascending=[True, False])

    # FIX #6: Enhanced logging with skipped epoch count
    n_filled = result_df['is_filled'].sum()
    n_total = len(result_df)
    if log_summary:
        logger.info(f"Missing epoch filling complete:")
        logger.info(f"  Original epochs: {n_total - n_filled}")
        logger.info(f"  Filled epochs: {n_filled}")
        logger.info(f"  Skipped epochs (gap too large): {total_skipped}")
        logger.info(f"  Total epochs: {n_total}")

    return result_df


def ensure_committed_epochs_filled(
    df: pd.DataFrame,
    birth_epoch_seconds: float = 0.0
) -> pd.DataFrame:
    """
    Ensure committed-metric inputs include forward-filled epochs up to birth.
    """
    if df is None or len(df) == 0:
        return df

    if 'guid' not in df.columns or 'epoch' not in df.columns:
        return df

    needs_fill = 'is_filled' not in df.columns
    max_by_guid = df.groupby('guid')['epoch'].max()
    if len(max_by_guid) > 0 and max_by_guid.min() < birth_epoch_seconds - 1e-6:
        needs_fill = True

    if not needs_fill:
        return df

    return fill_missing_epochs(
        df,
        max_gap_multiplier=None,
        log_summary=False,
        fill_until_birth=True,
        birth_epoch_seconds=birth_epoch_seconds
    )


def _is_better_threshold(
    fpr: float,
    fpr_diff: float,
    best_fpr: Optional[float],
    best_fpr_diff: float,
    target_fpr: float,
) -> bool:
    """Determine if a candidate threshold is better than the current best.

    Conservative strategy: prefer FPR <= target_fpr over FPR > target_fpr.
    Among conservative candidates, pick the one closest to target (highest
    FPR still <= target).  Among aggressive candidates (only when no
    conservative exists), pick the one closest to target.

    Args:
        fpr: FPR achieved by the candidate threshold.
        fpr_diff: ``abs(fpr - target_fpr)`` for the candidate.
        best_fpr: FPR of the current best threshold (``None`` if no best yet).
        best_fpr_diff: ``abs(best_fpr - target_fpr)`` (``inf`` initially).
        target_fpr: Target false-positive rate.

    Returns:
        ``True`` if the candidate should replace the current best.
    """
    is_conservative = fpr <= target_fpr
    was_conservative = best_fpr is not None and best_fpr <= target_fpr

    if is_conservative:
        if not was_conservative:
            # First conservative candidate always wins over aggressive
            return True
        # Both conservative — prefer closer to target (higher FPR ≤ target)
        return fpr_diff < best_fpr_diff
    else:
        if was_conservative:
            # Never replace a conservative best with an aggressive candidate
            return False
        # Both aggressive — prefer closer to target (lower FPR > target)
        return fpr_diff < best_fpr_diff


def find_threshold_for_instantaneous_fpr_at_1h(
    val_df: pd.DataFrame,
    target_fpr: float = 0.05,
    time_window_hours: float = 1.0,
    max_gap_multiplier: Optional[float] = None,
    fallback_tolerance_hours: float = 0.5
) -> Tuple[float, Dict]:
    """
    Find threshold for instantaneous decision metric at 1h before birth.

    Optimizes threshold to achieve target FPR for the INSTANTANEOUS metric:
    - FPR(1h) = FP(t=1h) / N(t=1h)
    - Only counts epochs within the 1h time bin

    Args:
        val_df: Validation DataFrame with columns: guid, epoch, binary_target, prob_class_1
        target_fpr: Target false positive rate (default 0.05)
        time_window_hours: Decision time in hours before delivery (default 1.0)
        max_gap_multiplier: For epoch filling. If None, fill all missing epochs
        fallback_tolerance_hours: If no data at exact time_window_hours, use nearest within tolerance

    Returns:
        threshold: Optimal threshold value
        metrics_dict: Performance metrics including actual time used
    """
    logger.info("=" * 80)
    logger.info(f"Finding threshold for INSTANTANEOUS FPR={target_fpr} at {time_window_hours}h before delivery")
    logger.info("=" * 80)

    val_df = ensure_epoch_hours(val_df.copy())

    # Create time bins aligned to the actual epoch grid
    time_bins = compute_time_bins(val_df, exclude_last_minutes=0.0)

    probs = val_df['prob_class_1'].dropna().values.astype(float)
    if len(probs) == 0:
        logger.error("No probabilities found - cannot determine threshold")
        return 0.5, {}

    threshold_candidates = np.unique(np.clip(probs, 0.0, 1.0))
    threshold_candidates = np.unique(np.concatenate(([0.0], threshold_candidates, [1.0])))
    threshold_candidates.sort()

    best_threshold = 0.5
    best_fpr_diff = float('inf')
    best_metrics = None

    # Binary search over threshold candidates
    lo, hi = 0, len(threshold_candidates) - 1
    max_iter = min(25, len(threshold_candidates))
    it = 0

    while lo <= hi and it < max_iter:
        mid = (lo + hi) // 2
        thresh = float(threshold_candidates[mid])

        # Apply threshold and clinical decision rule
        df_clinical = apply_clinical_decision_rule(val_df, thresh, verify=False)
        df_clinical = fill_missing_epochs(df_clinical, max_gap_multiplier, log_summary=False)

        # Compute instantaneous metrics at 1h bin
        instantaneous_df = compute_instantaneous_metrics(df_clinical, time_bins, subgroup_filter=None)

        # Find bin closest to target time
        if len(instantaneous_df) == 0:
            it += 1
            break

        target_bin = instantaneous_df.iloc[(instantaneous_df['bin_center'] - time_window_hours).abs().argmin()]

        # Check if bin is within tolerance
        actual_time = target_bin['bin_center']
        if abs(actual_time - time_window_hours) > fallback_tolerance_hours:
            logger.warning(f"No data within {fallback_tolerance_hours}h of target time {time_window_hours}h")
            it += 1
            break

        fpr = target_bin['fpr']
        sensitivity = target_bin['sensitivity']

        if pd.isna(fpr) or pd.isna(sensitivity):
            it += 1
            break

        fpr_diff = abs(fpr - target_fpr)
        best_fpr = best_metrics['fpr'] if best_metrics is not None else None
        if _is_better_threshold(fpr, fpr_diff, best_fpr, best_fpr_diff, target_fpr):
            best_fpr_diff = fpr_diff
            best_threshold = thresh
            best_metrics = {
                'threshold': thresh,
                'fpr': float(fpr),
                'sensitivity': float(sensitivity),
                'specificity': float(1 - fpr),
                'actual_time_hours': float(actual_time),
                'target_time_hours': time_window_hours,
                'n_positive': int(target_bin['n_positive']),
                'n_negative': int(target_bin['n_negative']),
                'metric_type': 'instantaneous'
            }

        if fpr > target_fpr:
            lo = mid + 1
        else:
            hi = mid - 1

        it += 1

    if best_metrics is None:
        logger.error("No valid thresholds found for instantaneous metric")
        return 0.5, {}

    logger.info(f"Selected threshold: {best_threshold:.3f}")
    logger.info(f"Achieved instantaneous FPR at {best_metrics['actual_time_hours']:.1f}h: {best_metrics['fpr']:.4f} (target: {target_fpr})")
    logger.info(f"Instantaneous sensitivity: {best_metrics['sensitivity']:.4f}")
    if best_metrics['fpr'] > target_fpr:
        logger.warning(f"Could not achieve target FPR={target_fpr} — closest conservative FPR not available")
    logger.info("=" * 80)

    return best_threshold, best_metrics


def find_threshold_for_committed_cumulative_fpr_at_1h(
    val_df: pd.DataFrame,
    target_fpr: float = 0.05,
    time_window_hours: float = 1.0,
    max_gap_multiplier: Optional[float] = None,
    fallback_tolerance_hours: float = 0.5
) -> Tuple[float, Dict]:
    """
    Find threshold for committed_cumulative decision metric at 1h before birth.

    Optimizes threshold to achieve target FPR for the COMMITTED CUMULATIVE metric:
    - FPR(1h) = FP(t≤1h) / N(t≤1h)
    - Cumulative detections from 1h to birth
    - Denominator = GUIDs available at 1h (CHANGING)

    Args:
        val_df: Validation DataFrame with columns: guid, epoch, binary_target, prob_class_1
        target_fpr: Target false positive rate (default 0.05)
        time_window_hours: Decision time in hours before delivery (default 1.0)
        max_gap_multiplier: For epoch filling. If None, fill all missing epochs
        fallback_tolerance_hours: If no data at exact time_window_hours, use nearest within tolerance

    Returns:
        threshold: Optimal threshold value
        metrics_dict: Performance metrics including actual time used
    """
    logger.info("=" * 80)
    logger.info(f"Finding threshold for COMMITTED CUMULATIVE FPR={target_fpr} at {time_window_hours}h before delivery")
    logger.info("=" * 80)

    val_df = ensure_epoch_hours(val_df.copy())

    # Create time bins aligned to the actual epoch grid
    time_bins = compute_time_bins(val_df, exclude_last_minutes=0.0)

    probs = val_df['prob_class_1'].dropna().values.astype(float)
    if len(probs) == 0:
        logger.error("No probabilities found - cannot determine threshold")
        return 0.5, {}

    threshold_candidates = np.unique(np.clip(probs, 0.0, 1.0))
    threshold_candidates = np.unique(np.concatenate(([0.0], threshold_candidates, [1.0])))
    threshold_candidates.sort()

    best_threshold = 0.5
    best_fpr_diff = float('inf')
    best_metrics = None

    # Binary search over threshold candidates
    lo, hi = 0, len(threshold_candidates) - 1
    max_iter = min(25, len(threshold_candidates))
    it = 0

    while lo <= hi and it < max_iter:
        mid = (lo + hi) // 2
        thresh = float(threshold_candidates[mid])

        # Apply threshold and clinical decision rule
        df_clinical = apply_clinical_decision_rule(val_df, thresh, verify=False)
        df_clinical = fill_missing_epochs(df_clinical, max_gap_multiplier=None, log_summary=False, fill_until_birth=True)

        # Compute committed cumulative metrics
        cumulative_df = compute_committed_cumulative_metrics(df_clinical, time_bins, subgroup_filter=None)

        # Find bin closest to target time
        if len(cumulative_df) == 0:
            it += 1
            break

        target_bin = cumulative_df.iloc[(cumulative_df['bin_center'] - time_window_hours).abs().argmin()]

        # Check if bin is within tolerance
        actual_time = target_bin['bin_center']
        if abs(actual_time - time_window_hours) > fallback_tolerance_hours:
            logger.warning(f"No data within {fallback_tolerance_hours}h of target time {time_window_hours}h")
            it += 1
            break

        fpr = target_bin['fpr']
        sensitivity = target_bin['sensitivity']

        if pd.isna(fpr) or pd.isna(sensitivity):
            it += 1
            break

        fpr_diff = abs(fpr - target_fpr)
        best_fpr = best_metrics['fpr'] if best_metrics is not None else None
        if _is_better_threshold(fpr, fpr_diff, best_fpr, best_fpr_diff, target_fpr):
            best_fpr_diff = fpr_diff
            best_threshold = thresh
            best_metrics = {
                'threshold': thresh,
                'fpr': float(fpr),
                'sensitivity': float(sensitivity),
                'specificity': float(1 - fpr),
                'actual_time_hours': float(actual_time),
                'target_time_hours': time_window_hours,
                'n_positive_available': int(target_bin['n_positive_available']),
                'n_negative_available': int(target_bin['n_negative_available']),
                'metric_type': 'committed_cumulative'
            }

        if fpr > target_fpr:
            lo = mid + 1
        else:
            hi = mid - 1

        it += 1

    if best_metrics is None:
        logger.error("No valid thresholds found for committed_cumulative metric")
        return 0.5, {}

    logger.info(f"Selected threshold: {best_threshold:.3f}")
    logger.info(f"Achieved committed_cumulative FPR at {best_metrics['actual_time_hours']:.1f}h: {best_metrics['fpr']:.4f} (target: {target_fpr})")
    logger.info(f"Committed_cumulative sensitivity: {best_metrics['sensitivity']:.4f}")
    if best_metrics['fpr'] > target_fpr:
        logger.warning(f"Could not achieve target FPR={target_fpr} — closest conservative FPR not available")
    logger.info("=" * 80)

    return best_threshold, best_metrics


def find_threshold_for_committed_overall_fpr_at_1h(
    val_df: pd.DataFrame,
    target_fpr: float = 0.05,
    time_window_hours: float = 1.0,
    max_gap_multiplier: Optional[float] = None,
    fallback_tolerance_hours: float = 0.5
) -> Tuple[float, Dict]:
    """
    Find threshold for committed_overall decision metric at 1h before birth (PRIMARY).

    Optimizes threshold to achieve target FPR for the COMMITTED OVERALL metric:
    - FPR(1h) = FP(t≤1h) / N(t≤0)
    - Cumulative detections from 1h to birth
    - Denominator = ALL GUIDs in dataset (FIXED)

    This is the PRIMARY METRIC for clinical reporting.

    Args:
        val_df: Validation DataFrame with columns: guid, epoch, binary_target, prob_class_1
        target_fpr: Target false positive rate (default 0.05)
        time_window_hours: Decision time in hours before delivery (default 1.0)
        max_gap_multiplier: For epoch filling. If None, fill all missing epochs
        fallback_tolerance_hours: If no data at exact time_window_hours, use nearest within tolerance

    Returns:
        threshold: Optimal threshold value
        metrics_dict: Performance metrics including actual time used
    """
    logger.info("=" * 80)
    logger.info(f"Finding threshold for COMMITTED OVERALL (PRIMARY) FPR={target_fpr} at {time_window_hours}h before delivery")
    logger.info("=" * 80)

    val_df = ensure_epoch_hours(val_df.copy())

    # Create time bins aligned to the actual epoch grid
    time_bins = compute_time_bins(val_df, exclude_last_minutes=0.0)

    probs = val_df['prob_class_1'].dropna().values.astype(float)
    if len(probs) == 0:
        logger.error("No probabilities found - cannot determine threshold")
        return 0.5, {}

    threshold_candidates = np.unique(np.clip(probs, 0.0, 1.0))
    threshold_candidates = np.unique(np.concatenate(([0.0], threshold_candidates, [1.0])))
    threshold_candidates.sort()

    best_threshold = 0.5
    best_fpr_diff = float('inf')
    best_metrics = None

    # Binary search over threshold candidates
    lo, hi = 0, len(threshold_candidates) - 1
    max_iter = min(25, len(threshold_candidates))
    it = 0

    while lo <= hi and it < max_iter:
        mid = (lo + hi) // 2
        thresh = float(threshold_candidates[mid])

        # Apply threshold and clinical decision rule
        df_clinical = apply_clinical_decision_rule(val_df, thresh, verify=False)
        df_clinical = fill_missing_epochs(df_clinical, max_gap_multiplier=None, log_summary=False, fill_until_birth=True)

        # Compute committed overall metrics (PRIMARY)
        overall_df = compute_committed_overall_metrics(df_clinical, time_bins, subgroup_filter=None)

        # Find bin closest to target time
        if len(overall_df) == 0:
            it += 1
            break

        target_bin = overall_df.iloc[(overall_df['bin_center'] - time_window_hours).abs().argmin()]

        # Check if bin is within tolerance
        actual_time = target_bin['bin_center']
        if abs(actual_time - time_window_hours) > fallback_tolerance_hours:
            logger.warning(f"No data within {fallback_tolerance_hours}h of target time {time_window_hours}h")
            it += 1
            break

        fpr = target_bin['fpr']
        sensitivity = target_bin['sensitivity']

        if pd.isna(fpr) or pd.isna(sensitivity):
            it += 1
            break

        fpr_diff = abs(fpr - target_fpr)
        best_fpr = best_metrics['fpr'] if best_metrics is not None else None
        if _is_better_threshold(fpr, fpr_diff, best_fpr, best_fpr_diff, target_fpr):
            best_fpr_diff = fpr_diff
            best_threshold = thresh
            best_metrics = {
                'threshold': thresh,
                'fpr': float(fpr),
                'sensitivity': float(sensitivity),
                'specificity': float(1 - fpr),
                'actual_time_hours': float(actual_time),
                'target_time_hours': time_window_hours,
                'n_positive_total': int(target_bin['n_positive_total']),
                'n_negative_total': int(target_bin['n_negative_total']),
                'n_available_positive': int(target_bin['n_available_positive']),
                'n_available_negative': int(target_bin['n_available_negative']),
                'metric_type': 'committed_overall',
                'is_primary_metric': True
            }

        if fpr > target_fpr:
            lo = mid + 1
        else:
            hi = mid - 1

        it += 1

    if best_metrics is None:
        logger.error("No valid thresholds found for committed_overall (PRIMARY) metric")
        return 0.5, {}

    logger.info(f"Selected threshold (PRIMARY): {best_threshold:.3f}")
    logger.info(f"Achieved committed_overall FPR at {best_metrics['actual_time_hours']:.1f}h: {best_metrics['fpr']:.4f} (target: {target_fpr})")
    logger.info(f"Committed_overall sensitivity: {best_metrics['sensitivity']:.4f}")
    logger.info(f"FIXED denominators - P(t≤0)={best_metrics['n_positive_total']}, N(t≤0)={best_metrics['n_negative_total']}")
    if best_metrics['fpr'] > target_fpr:
        logger.warning(f"Could not achieve target FPR={target_fpr} — closest conservative FPR not available")
    logger.info("=" * 80)

    return best_threshold, best_metrics


def compute_time_bins(df: pd.DataFrame, exclude_last_minutes: float = 30.0) -> np.ndarray:
    """
    Compute dynamic time bins from actual epoch data with a uniform bin size
    inferred from the dataset's epoch spacing.

    Args:
        df: DataFrame with 'epoch' column (negative seconds before birth)
        exclude_last_minutes: Exclude last N minutes before birth from bins (default: 30 min = 0.5h)

    Returns:
        Array of time bin edges in hours (positive values: 0.5 = 30min before birth, 6 = 6h before birth)
    """
    df_bins = df.copy()
    df_interval = df_bins
    if 'is_filled' in df_bins.columns:
        df_interval = df_bins[df_bins['is_filled'] == False]  # noqa: E712

    # Ensure epoch_hours exists (positive hours before birth)
    df_interval = ensure_epoch_hours(df_interval)
    df_bins = ensure_epoch_hours(df_bins)

    # Infer bin size from typical epoch interval (fallback to 20 minutes if inference fails)
    inferred_seconds = infer_epoch_interval_seconds(df_interval)
    bin_size_hours = (inferred_seconds / 3600.0) if inferred_seconds > 0 else (1.0 / 3.0)
    bin_size_hours = round(bin_size_hours * 60) / 60.0

    # Convert exclusion from minutes to hours
    exclude_hours = exclude_last_minutes / 60.0  # 30min = 0.5h

    if len(df_bins) == 0:
        return np.array([exclude_hours, exclude_hours + bin_size_hours])

    # Filter out post-delivery epochs (epoch_hours <= 0 after negation fix)
    # and apply the exclusion window
    df_included = df_bins[(df_bins['epoch_hours'] > 0) & (df_bins['epoch_hours'] >= exclude_hours)]
    if len(df_included) == 0:
        return np.array([exclude_hours, exclude_hours + bin_size_hours])

    # Anchor bins to the actual epoch grid (bin centers at epoch start times).
    first_center = float(df_included['epoch_hours'].min())
    last_center = float(df_included['epoch_hours'].max())

    start_edge = first_center - bin_size_hours / 2.0
    end_edge = last_center + bin_size_hours / 2.0 + 1e-9

    # Clamp to exclude window so we don't show bins that overlap excluded region.
    start_edge = max(start_edge, exclude_hours)
    if start_edge >= end_edge:
        end_edge = start_edge + bin_size_hours

    bins = np.arange(start_edge, end_edge, bin_size_hours)

    return bins


def compute_time_windows(df: pd.DataFrame, exclude_last_minutes: float = 30.0) -> List[Tuple[float, float]]:
    """
    Compute time windows for ROC/confusion matrix analysis.

    Args:
        df: DataFrame with 'epoch' column (negative seconds before birth)
        exclude_last_minutes: Exclude last N minutes before birth from windows (default: 30 min = 0.5h)

    Returns:
        List of (start_hour, end_hour) tuples (positive values: 0.5 = 30min before birth)
    """
    # Epochs are negative seconds before birth: negate min() to get furthest positive hours
    max_epoch_hours = -df['epoch'].min() / 3600

    # Convert exclusion from minutes to hours
    exclude_hours = exclude_last_minutes / 60.0  # 30min = 0.5h

    # Fixed critical windows (adjusted to start at exclude_hours)
    windows = [(exclude_hours, 1), (1, 2), (2, 4), (4, 6), (exclude_hours, 6)]

    # Add additional windows if data extends beyond 6 hours
    if max_epoch_hours > 6:
        windows.extend([(6, 12), (exclude_hours, 12)])

    return windows


# ==================================================================================
# SUBGROUP ANALYSIS FUNCTIONS
# ==================================================================================

def create_enhanced_subgroup_filters() -> Dict[str, callable]:
    """
    Create enhanced filter functions for all subgroups.

    Includes comprehensive subgroups for both unhealthy and healthy cases,
    stratified by CS status, BG status, and their combinations.

    UNHEALTHY SUBGROUPS:
    - Basic: acidosis, hie, unhealthy (combined)
    - CS stratified: unhealthy_cs_pos/neg, hie_cs_pos/neg, acidosis_cs_pos/neg
    - BG stratified (Acidosis only): acidosis_bg_pos/neg
      Note: All HIE cases are bg_positive, so no HIE BG stratification

    HEALTHY SUBGROUPS:
    - Basic: healthy
    - CS stratified: healthy_cs_pos/neg
    - BG stratified: healthy_bg_pos/neg
    - BG×CS combinations (4-way): healthy_bg_pos_cs_pos/neg, healthy_bg_neg_cs_pos/neg

    Returns:
        Dictionary mapping subgroup_name -> filter_function
    """
    return {
        # =====================================================================
        # UNHEALTHY SUBGROUPS
        # =====================================================================

        # Basic diagnosis subgroups
        'acidosis': lambda df: df['target'] == 2,
        'hie': lambda df: df['target'] == 3,
        'unhealthy': lambda df: (df['target'] == 2) | (df['target'] == 3),

        # Unhealthy stratified by CS status
        'unhealthy_cs_pos': lambda df: ((df['target'] == 2) | (df['target'] == 3)) & (df['cs_label'] == True),
        'unhealthy_cs_neg': lambda df: ((df['target'] == 2) | (df['target'] == 3)) & (df['cs_label'] == False),

        # HIE stratified by CS status
        'hie_cs_pos': lambda df: (df['target'] == 3) & (df['cs_label'] == True),
        'hie_cs_neg': lambda df: (df['target'] == 3) & (df['cs_label'] == False),

        # Acidosis stratified by CS status
        'acidosis_cs_pos': lambda df: (df['target'] == 2) & (df['cs_label'] == True),
        'acidosis_cs_neg': lambda df: (df['target'] == 2) & (df['cs_label'] == False),

        # Acidosis stratified by BG status (ONLY acidosis; all HIE are bg_positive)
        'acidosis_bg_pos': lambda df: (df['target'] == 2) & (df['bg_label'] == True),
        'acidosis_bg_neg': lambda df: (df['target'] == 2) & (df['bg_label'] == False),

        # =====================================================================
        # HEALTHY SUBGROUPS
        # =====================================================================

        # Basic healthy
        'healthy': lambda df: df['target'] == 1,

        # Healthy stratified by CS status
        'healthy_cs_pos': lambda df: (df['target'] == 1) & (df['cs_label'] == True),
        'healthy_cs_neg': lambda df: (df['target'] == 1) & (df['cs_label'] == False),

        # Healthy stratified by BG status
        'healthy_bg_pos': lambda df: (df['target'] == 1) & (df['bg_label'] == True),
        'healthy_bg_neg': lambda df: (df['target'] == 1) & (df['bg_label'] == False),

        # Healthy BG×CS combinations (4-way)
        'healthy_bg_pos_cs_pos': lambda df: (df['target'] == 1) & (df['bg_label'] == True) & (df['cs_label'] == True),
        'healthy_bg_pos_cs_neg': lambda df: (df['target'] == 1) & (df['bg_label'] == True) & (df['cs_label'] == False),
        'healthy_bg_neg_cs_pos': lambda df: (df['target'] == 1) & (df['bg_label'] == False) & (df['cs_label'] == True),
        'healthy_bg_neg_cs_neg': lambda df: (df['target'] == 1) & (df['bg_label'] == False) & (df['cs_label'] == False),

        # =====================================================================
        # LEGACY SUBGROUPS (for backward compatibility)
        # =====================================================================

        # CS status (global)
        'cs_positive': lambda df: df['cs_label'] == True,
        'cs_negative': lambda df: df['cs_label'] == False,

        # BG label (global)
        'bg_positive': lambda df: df['bg_label'] == True,
        'bg_negative': lambda df: df['bg_label'] == False,
    }


def compute_subgroup_statistics(
    df: pd.DataFrame,
    subgroup_filters: Dict[str, callable]
) -> Dict[str, Dict[str, int]]:
    """
    Compute dataset statistics for all subgroups.

    For each subgroup, computes:
    - n_guids: Number of unique patients (GUIDs) in subgroup
    - n_epochs: Total number of epochs in subgroup
    - n_positive: Number of unhealthy GUIDs (binary_target==1)
    - n_negative: Number of healthy GUIDs (binary_target==0)

    Args:
        df: Full predictions dataframe with columns: guid, target, binary_target,
            cs_label, bg_label, epoch
        subgroup_filters: Dictionary mapping subgroup_name -> filter_function

    Returns:
        Dictionary mapping subgroup_name -> statistics dict with:
            - n_guids: Unique patient count
            - n_epochs: Total epoch count
            - n_positive: Unhealthy patient count
            - n_negative: Healthy patient count
            - diagnosis_breakdown: Count by target (1=Healthy, 2=Acidosis, 3=HIE)
    """
    statistics = {}

    for subgroup_name, filter_func in subgroup_filters.items():
        try:
            # Apply subgroup filter
            subgroup_df = df[filter_func(df)].copy()

            if len(subgroup_df) == 0:
                statistics[subgroup_name] = {
                    'n_guids': 0,
                    'n_epochs': 0,
                    'n_positive': 0,
                    'n_negative': 0,
                    'diagnosis_breakdown': {
                        'healthy': 0,
                        'acidosis': 0,
                        'hie': 0
                    }
                }
                continue

            # Basic counts
            n_guids = subgroup_df['guid'].nunique()
            n_epochs = len(subgroup_df)

            # Get unique GUIDs and their targets
            guid_targets = subgroup_df.groupby('guid').agg({
                'binary_target': 'first',
                'target': 'first'
            })

            n_positive = (guid_targets['binary_target'] == 1).sum()
            n_negative = (guid_targets['binary_target'] == 0).sum()

            # Diagnosis breakdown (by unique GUIDs)
            diagnosis_breakdown = {
                'healthy': (guid_targets['target'] == 1).sum(),
                'acidosis': (guid_targets['target'] == 2).sum(),
                'hie': (guid_targets['target'] == 3).sum()
            }

            statistics[subgroup_name] = {
                'n_guids': int(n_guids),
                'n_epochs': int(n_epochs),
                'n_positive': int(n_positive),
                'n_negative': int(n_negative),
                'diagnosis_breakdown': {
                    'healthy': int(diagnosis_breakdown['healthy']),
                    'acidosis': int(diagnosis_breakdown['acidosis']),
                    'hie': int(diagnosis_breakdown['hie'])
                }
            }

        except Exception as e:
            logger.warning(f"Failed to compute statistics for {subgroup_name}: {e}")
            statistics[subgroup_name] = {
                'n_guids': 0,
                'n_epochs': 0,
                'n_positive': 0,
                'n_negative': 0,
                'diagnosis_breakdown': {
                    'healthy': 0,
                    'acidosis': 0,
                    'hie': 0
                },
                'error': str(e)
            }

    return statistics


# ---------------------------------------------------------------------------
# Per-fold dataset statistics with visualizations
# ---------------------------------------------------------------------------

def _seconds_to_hhmm(seconds: float) -> str:
    """Convert seconds before birth to HH:MM format (absolute value)."""
    abs_seconds = abs(seconds)
    hours = int(abs_seconds // 3600)
    minutes = int((abs_seconds % 3600) // 60)
    return f"{hours:02d}:{minutes:02d}"


def _get_clinical_subgroups():
    """
    Return the clinical subgroup definitions used across all dataset-stats plots.

    Each entry: (key, label, filter_function, colour).

    Subgroups are **mutually exclusive within each diagnosis** when stratified
    by CS, and within healthy when stratified by BG.
    """
    return {
        # -- Healthy, stratified by BG -----------------------------------
        'healthy_bg_pos': ('Healthy BG+',
                           lambda d: (d['target'] == 1) & (d['bg_label'] == True),   # noqa: E712
                           '#2ecc71'),
        'healthy_bg_neg': ('Healthy BG\u2212',
                           lambda d: (d['target'] == 1) & (d['bg_label'] == False),  # noqa: E712
                           '#27ae60'),
        # -- Healthy, stratified by CS -----------------------------------
        'healthy_cs_pos': ('Healthy CS+',
                           lambda d: (d['target'] == 1) & (d['cs_label'] == True),   # noqa: E712
                           '#a8e6cf'),
        'healthy_cs_neg': ('Healthy CS\u2212',
                           lambda d: (d['target'] == 1) & (d['cs_label'] == False),  # noqa: E712
                           '#6ab04c'),
        # -- Healthy, stratified by BG\u00d7CS 4-way ---------------------------
        'healthy_bg_pos_cs_pos': ('Healthy BG+ CS+',
                          lambda d: (d['target'] == 1) & (d['bg_label'] == True) & (d['cs_label'] == True),   # noqa: E712
                          '#e74c3c'),
        'healthy_bg_pos_cs_neg': ('Healthy BG+ CS\u2212',
                          lambda d: (d['target'] == 1) & (d['bg_label'] == True) & (d['cs_label'] == False),  # noqa: E712
                          '#3498db'),
        'healthy_bg_neg_cs_pos': ('Healthy BG\u2212 CS+',
                          lambda d: (d['target'] == 1) & (d['bg_label'] == False) & (d['cs_label'] == True),  # noqa: E712
                          '#f39c12'),
        'healthy_bg_neg_cs_neg': ('Healthy BG\u2212 CS\u2212',
                          lambda d: (d['target'] == 1) & (d['bg_label'] == False) & (d['cs_label'] == False), # noqa: E712
                          '#9b59b6'),
        # -- Acidosis, stratified by CS ----------------------------------
        'acidosis_cs_pos': ('Acidosis CS+',
                            lambda d: (d['target'] == 2) & (d['cs_label'] == True),  # noqa: E712
                            '#e74c3c'),
        'acidosis_cs_neg': ('Acidosis CS\u2212',
                            lambda d: (d['target'] == 2) & (d['cs_label'] == False), # noqa: E712
                            '#c0392b'),
        # -- HIE, stratified by CS ---------------------------------------
        'hie_cs_pos': ('HIE CS+',
                       lambda d: (d['target'] == 3) & (d['cs_label'] == True),       # noqa: E712
                       '#e67e22'),
        'hie_cs_neg': ('HIE CS\u2212',
                       lambda d: (d['target'] == 3) & (d['cs_label'] == False),      # noqa: E712
                       '#d35400'),
    }


def _compute_subgroup_stats(df: pd.DataFrame, subgroups: dict) -> dict:
    """
    Compute per-subgroup statistics (n_guids, n_epochs, epochs_per_guid).

    Args:
        df: Predictions DataFrame (must have epoch_hours already).
        subgroups: Dict from ``_get_clinical_subgroups()``.

    Returns:
        Dict mapping subgroup key -> stats dict.
    """
    results = {}
    for key, (label, filt, _color) in subgroups.items():
        sub = df[filt(df)]
        n_guids = int(sub['guid'].nunique())
        n_epochs = len(sub)
        if n_guids > 0:
            epg = sub.groupby('guid').size()
            epg_stats = {
                'min': int(epg.min()),
                'max': int(epg.max()),
                'mean': float(epg.mean()),
                'median': float(epg.median()),
                'std': float(epg.std()) if len(epg) > 1 else 0.0,
            }
        else:
            epg_stats = {'min': 0, 'max': 0, 'mean': 0.0, 'median': 0.0, 'std': 0.0}
        results[key] = {
            'label': label,
            'n_guids': n_guids,
            'n_epochs': n_epochs,
            'epochs_per_guid': epg_stats,
        }
    return results


def generate_fold_dataset_stats(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    output_dir: Path,
    title_suffix: str = "Test Set",
) -> None:
    """
    Generate per-fold dataset statistics folder with JSON summary, plots, and CSV.

    All outputs are stratified by clinical subgroup:
      Healthy  — BG+/BG\u2212  and CS+/CS\u2212
      Acidosis — CS+/CS\u2212
      HIE      — CS+/CS\u2212

    Creates a ``dataset_stats/`` directory containing:
    - dataset_summary.json       — overall + per-subgroup statistics
    - dataset_overview.pdf       — 2\u00d72 overview figure with subgroup bars
    - subgroup_overview.pdf      — grouped bar chart (GUIDs & epochs per subgroup)
    - epochs_per_time_bin.pdf    — stacked by Healthy / Acidosis / HIE
    - epochs_per_time_bin_subgroups.pdf — per-diagnosis panels with CS/BG stratification
    - epochs_per_guid_ranked.pdf — ranked bar chart coloured by diagnosis
    - label_cross_table.csv      — target \u00d7 cs_label \u00d7 bg_label (GUID + epoch counts)

    Args:
        df: Predictions DataFrame with columns: guid, epoch, target,
            binary_target, cs_label, bg_label.
        time_bins: Array of time-bin edges in hours (from ``compute_time_bins``).
        output_dir: Directory to create (will be created if absent).
        title_suffix: Text appended to plot titles (e.g. "Fold 1 Test Set").
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure epoch_hours column exists
    df_stats = ensure_epoch_hours(df.copy())

    subgroups = _get_clinical_subgroups()
    subgroup_stats = _compute_subgroup_stats(df_stats, subgroups)

    # --- A. Compute overall summary statistics --------------------------
    n_guids = int(df_stats['guid'].nunique())
    n_epochs = len(df_stats)

    epochs_per_guid = df_stats.groupby('guid').size()
    epg_stats = {
        'min': int(epochs_per_guid.min()),
        'max': int(epochs_per_guid.max()),
        'mean': float(epochs_per_guid.mean()),
        'median': float(epochs_per_guid.median()),
        'std': float(epochs_per_guid.std()) if len(epochs_per_guid) > 1 else 0.0,
    }

    min_sec = float(df_stats['epoch'].min())
    max_sec = float(df_stats['epoch'].max())
    n_unique_time_points = int(df_stats['epoch'].nunique())

    # GUID-level label info
    guid_info = df_stats.groupby('guid').agg({
        'target': 'first',
        'binary_target': 'first',
        'cs_label': 'first',
        'bg_label': 'first',
    })

    target_map = {1: 'healthy', 2: 'acidosis', 3: 'hie'}
    guid_target_counts = {v: int((guid_info['target'] == k).sum()) for k, v in target_map.items()}
    guid_cs_counts = {
        'cs_positive': int(guid_info['cs_label'].eq(True).sum()),
        'cs_negative': int(guid_info['cs_label'].eq(False).sum()),
    }
    guid_bg_counts = {
        'bg_positive': int(guid_info['bg_label'].eq(True).sum()),
        'bg_negative': int(guid_info['bg_label'].eq(False).sum()),
    }

    # Epoch-level label counts
    epoch_target_counts = {v: int((df_stats['target'] == k).sum()) for k, v in target_map.items()}
    epoch_cs_counts = {
        'cs_positive': int(df_stats['cs_label'].eq(True).sum()),
        'cs_negative': int(df_stats['cs_label'].eq(False).sum()),
    }
    epoch_bg_counts = {
        'bg_positive': int(df_stats['bg_label'].eq(True).sum()),
        'bg_negative': int(df_stats['bg_label'].eq(False).sum()),
    }

    # Epochs per time bin (overall)
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2.0
    bin_epoch_counts = []
    for i in range(len(time_bins) - 1):
        lo, hi = time_bins[i], time_bins[i + 1]
        mask = (df_stats['epoch_hours'] >= lo) & (df_stats['epoch_hours'] < hi)
        bin_epoch_counts.append({
            'bin_center_hours': float(bin_centers[i]),
            'n_epochs': int(mask.sum()),
        })

    summary = {
        'total_guids': n_guids,
        'total_epochs': n_epochs,
        'epochs_per_guid': epg_stats,
        'time_range': {
            'min_seconds': min_sec,
            'max_seconds': max_sec,
            'min_hours': float(min_sec / 3600.0),
            'max_hours': float(max_sec / 3600.0),
            'min_hhmm': _seconds_to_hhmm(min_sec),
            'max_hhmm': _seconds_to_hhmm(max_sec),
        },
        'n_unique_time_points': n_unique_time_points,
        'label_distributions': {
            'guid_level': {
                'target': guid_target_counts,
                'cs_label': guid_cs_counts,
                'bg_label': guid_bg_counts,
            },
            'epoch_level': {
                'target': epoch_target_counts,
                'cs_label': epoch_cs_counts,
                'bg_label': epoch_bg_counts,
            },
        },
        'epochs_per_time_bin': bin_epoch_counts,
        'subgroups': subgroup_stats,
    }

    with open(output_dir / "dataset_summary.json", 'w') as f:
        json.dump(convert_numpy_types(summary), f, indent=2)
    logger.info(f"Dataset summary saved to {output_dir / 'dataset_summary.json'}")

    # --- B. Dataset overview (2x2) --------------------------------------
    _plot_dataset_overview(df_stats, summary, subgroup_stats, output_dir, title_suffix)

    # --- C. Subgroup overview bar chart ---------------------------------
    _plot_subgroup_overview(subgroup_stats, subgroups, output_dir, title_suffix)

    # --- D. Epochs per time bin (diagnosis-level stacked) ---------------
    _plot_epochs_per_time_bin(df_stats, time_bins, output_dir, title_suffix)

    # --- E. Epochs per time bin (per-diagnosis CS/BG panels) ------------
    _plot_epochs_per_time_bin_subgroups(df_stats, time_bins, subgroups, output_dir, title_suffix)

    # --- F. Epochs per GUID ranked --------------------------------------
    _plot_epochs_per_guid_ranked(df_stats, output_dir, title_suffix)

    # --- G. Label cross-table -------------------------------------------
    _save_label_cross_table(df_stats, output_dir)

    logger.info(f"Fold dataset stats saved to {output_dir}")


def _plot_dataset_overview(
    df: pd.DataFrame,
    summary: dict,
    subgroup_stats: dict,
    output_dir: Path,
    title_suffix: str,
) -> None:
    """Create 2x2 overview figure: histogram, time dist, subgroup bars, summary text."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Dataset Overview — {title_suffix}", fontsize=12, fontweight='bold', y=0.98)

    subgroups = _get_clinical_subgroups()

    # Colours
    c_green = '#2ecc71'
    c_blue = '#3498db'
    c_red = '#e74c3c'

    # [0,0] Epochs per GUID histogram
    ax = axes[0, 0]
    epochs_per_guid = df.groupby('guid').size()
    ax.hist(epochs_per_guid, bins=min(30, len(epochs_per_guid.unique())),
            color=c_blue, edgecolor='white', alpha=0.8)
    mean_val = float(epochs_per_guid.mean())
    median_val = float(epochs_per_guid.median())
    ax.axvline(mean_val, color=c_red, linestyle='--', linewidth=1.5,
               label=f"Mean: {mean_val:.1f}")
    ax.axvline(median_val, color=c_green, linestyle=':', linewidth=1.5,
               label=f"Median: {median_val:.1f}")
    ax.set_xlabel("Epochs per GUID")
    ax.set_ylabel("Number of GUIDs")
    ax.set_title(f"Epochs per Baby (n={summary['total_guids']})")
    ax.legend(fontsize=7)

    # [0,1] Time distribution (epoch count vs hours before birth)
    ax = axes[0, 1]
    epoch_hours = df['epoch_hours']
    ax.hist(epoch_hours, bins=min(50, len(epoch_hours.unique())),
            color=c_blue, edgecolor='white', alpha=0.8)
    ax.set_xlabel("Hours before birth")
    ax.set_ylabel("Number of epochs")
    ax.set_title(f"Time Distribution ({summary['n_unique_time_points']} unique pts)")
    ax.invert_xaxis()

    # [1,0] Subgroup GUID counts (grouped bar chart)
    ax = axes[1, 0]
    # Show the 8 clinical subgroups
    sg_keys = list(subgroups.keys())
    sg_labels = [subgroups[k][0] for k in sg_keys]
    sg_colors = [subgroups[k][2] for k in sg_keys]
    sg_guids = [subgroup_stats.get(k, {}).get('n_guids', 0) for k in sg_keys]

    x_pos = np.arange(len(sg_keys))
    bars = ax.bar(x_pos, sg_guids, color=sg_colors, edgecolor='white', alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sg_labels, rotation=40, ha='right', fontsize=7)
    ax.set_ylabel("Number of GUIDs")
    ax.set_title("Subgroup GUID Counts")
    for bar, cnt in zip(bars, sg_guids):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{cnt}", ha='center', va='bottom', fontsize=7)

    # [1,1] Text summary box with subgroup breakdown
    ax = axes[1, 1]
    ax.axis('off')
    epg = summary['epochs_per_guid']
    tr = summary['time_range']
    tgt = summary['label_distributions']['guid_level']['target']

    # Build subgroup lines
    sg_lines = []
    for k in sg_keys:
        s = subgroup_stats.get(k, {})
        sg_lines.append(f"  {subgroups[k][0]:16s}  {s.get('n_guids',0):3d} GUIDs  {s.get('n_epochs',0):5d} epochs")

    text = (
        f"Dataset Summary\n"
        f"{'─' * 42}\n"
        f"Total Epochs:  {summary['total_epochs']:,}\n"
        f"Unique GUIDs:  {summary['total_guids']:,}\n"
        f"  Healthy: {tgt.get('healthy', 0)}  "
        f"Acidosis: {tgt.get('acidosis', 0)}  "
        f"HIE: {tgt.get('hie', 0)}\n\n"
        f"Epochs/GUID: {epg['mean']:.1f} mean, {epg['median']:.1f} med\n"
        f"Time: {tr['max_hhmm']} \u2192 {tr['min_hhmm']} (HH:MM)\n\n"
        f"Subgroups:\n" + "\n".join(sg_lines)
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=7.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ecf0f1', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_dir / "dataset_overview.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_subgroup_overview(
    subgroup_stats: dict,
    subgroups: dict,
    output_dir: Path,
    title_suffix: str,
) -> None:
    """Grouped bar chart showing n_guids and n_epochs for every clinical subgroup."""
    sg_keys = list(subgroups.keys())
    sg_labels = [subgroups[k][0] for k in sg_keys]
    sg_colors = [subgroups[k][2] for k in sg_keys]
    sg_guids = [subgroup_stats.get(k, {}).get('n_guids', 0) for k in sg_keys]
    sg_epochs = [subgroup_stats.get(k, {}).get('n_epochs', 0) for k in sg_keys]

    x = np.arange(len(sg_keys))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(18, 5))

    # GUIDs bars (left)
    bars_g = ax1.bar(x - width / 2, sg_guids, width, color=sg_colors,
                     edgecolor='white', alpha=0.85, label='GUIDs')
    ax1.set_ylabel("Number of GUIDs")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sg_labels, rotation=35, ha='right', fontsize=8)

    # Epochs bars (right, secondary axis)
    ax2 = ax1.twinx()
    bars_e = ax2.bar(x + width / 2, sg_epochs, width, color=sg_colors,
                     edgecolor='white', alpha=0.45, hatch='//', label='Epochs')
    ax2.set_ylabel("Number of Epochs")

    # Count labels
    for bar, cnt in zip(bars_g, sg_guids):
        if cnt > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{cnt}", ha='center', va='bottom', fontsize=7)
    for bar, cnt in zip(bars_e, sg_epochs):
        if cnt > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{cnt}", ha='center', va='bottom', fontsize=7)

    # Combined legend
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(facecolor='gray', alpha=0.85, label='GUIDs (solid)'),
        Patch(facecolor='gray', alpha=0.45, hatch='//', label='Epochs (hatched)'),
    ], fontsize=8, loc='upper right')

    ax1.set_title(f"Subgroup Overview — {title_suffix}", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "subgroup_overview.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_epochs_per_time_bin(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    output_dir: Path,
    title_suffix: str,
) -> None:
    """Stacked bar chart of epochs per metric time bin (Healthy / Acidosis / HIE)."""
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2.0
    n_bins = len(bin_centers)

    counts_healthy = np.zeros(n_bins)
    counts_acidosis = np.zeros(n_bins)
    counts_hie = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = time_bins[i], time_bins[i + 1]
        mask = (df['epoch_hours'] >= lo) & (df['epoch_hours'] < hi)
        bin_df = df[mask]
        counts_healthy[i] = (bin_df['target'] == 1).sum()
        counts_acidosis[i] = (bin_df['target'] == 2).sum()
        counts_hie[i] = (bin_df['target'] == 3).sum()

    bar_width = float(np.median(np.diff(time_bins)) * 0.8) if n_bins > 1 else 0.3

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(bin_centers, counts_healthy, width=bar_width,
           label='Healthy', color='#2ecc71', edgecolor='white', alpha=0.85)
    ax.bar(bin_centers, counts_acidosis, width=bar_width, bottom=counts_healthy,
           label='Acidosis', color='#e74c3c', edgecolor='white', alpha=0.85)
    ax.bar(bin_centers, counts_hie, width=bar_width,
           bottom=counts_healthy + counts_acidosis,
           label='HIE', color='#e67e22', edgecolor='white', alpha=0.85)

    total = int(counts_healthy.sum() + counts_acidosis.sum() + counts_hie.sum())
    ax.set_xlabel("Hours before birth")
    ax.set_ylabel("Number of epochs")
    ax.set_title(f"Epochs per Time Bin — {title_suffix}\n"
                 f"({n_bins} bins, {total} total epochs)")
    ax.legend(fontsize=9)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_dir / "epochs_per_time_bin.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_epochs_per_time_bin_subgroups(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    subgroups: dict,
    output_dir: Path,
    title_suffix: str,
) -> None:
    """
    Multi-panel figure: per-diagnosis time-bin breakdown with CS/BG stratification.

    Layout (4 rows):
      Row 0 — Healthy: BG+ vs BG\u2212  |  Healthy: CS+ vs CS\u2212
      Row 1 — Acidosis: CS+ vs CS\u2212  |  (empty / summary text)
      Row 2 — HIE: CS+ vs CS\u2212       |  (empty / summary text)
      Row 3 — Healthy: BG\u00d7CS 4-way  |  (summary text)
    """
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2.0
    n_bins = len(bin_centers)
    bar_width = float(np.median(np.diff(time_bins)) * 0.8) if n_bins > 1 else 0.3

    def _bin_counts(filt):
        counts = np.zeros(n_bins)
        for i in range(n_bins):
            lo, hi = time_bins[i], time_bins[i + 1]
            mask_time = (df['epoch_hours'] >= lo) & (df['epoch_hours'] < hi)
            counts[i] = filt(df[mask_time]).sum()
        return counts

    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(f"Epochs per Time Bin by Subgroup — {title_suffix}",
                 fontsize=13, fontweight='bold', y=0.98)

    # --- Row 0, Col 0: Healthy by BG ---
    ax = axes[0, 0]
    c_bg_pos = _bin_counts(lambda d: (d['target'] == 1) & (d['bg_label'] == True))   # noqa: E712
    c_bg_neg = _bin_counts(lambda d: (d['target'] == 1) & (d['bg_label'] == False))  # noqa: E712
    ax.bar(bin_centers, c_bg_neg, width=bar_width,
           label=f"Healthy BG\u2212 ({int(c_bg_neg.sum())})",
           color='#27ae60', edgecolor='white', alpha=0.85)
    ax.bar(bin_centers, c_bg_pos, width=bar_width, bottom=c_bg_neg,
           label=f"Healthy BG+ ({int(c_bg_pos.sum())})",
           color='#2ecc71', edgecolor='white', alpha=0.85)
    ax.set_title("Healthy — BG stratification")
    ax.set_ylabel("Epochs")
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    # --- Row 0, Col 1: Healthy by CS ---
    ax = axes[0, 1]
    c_cs_pos = _bin_counts(lambda d: (d['target'] == 1) & (d['cs_label'] == True))   # noqa: E712
    c_cs_neg = _bin_counts(lambda d: (d['target'] == 1) & (d['cs_label'] == False))  # noqa: E712
    ax.bar(bin_centers, c_cs_neg, width=bar_width,
           label=f"Healthy CS\u2212 ({int(c_cs_neg.sum())})",
           color='#6ab04c', edgecolor='white', alpha=0.85)
    ax.bar(bin_centers, c_cs_pos, width=bar_width, bottom=c_cs_neg,
           label=f"Healthy CS+ ({int(c_cs_pos.sum())})",
           color='#a8e6cf', edgecolor='white', alpha=0.85)
    ax.set_title("Healthy — CS stratification")
    ax.set_ylabel("Epochs")
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    # --- Row 1, Col 0: Acidosis by CS ---
    ax = axes[1, 0]
    c_acid_cs_pos = _bin_counts(lambda d: (d['target'] == 2) & (d['cs_label'] == True))   # noqa: E712
    c_acid_cs_neg = _bin_counts(lambda d: (d['target'] == 2) & (d['cs_label'] == False))  # noqa: E712
    ax.bar(bin_centers, c_acid_cs_neg, width=bar_width,
           label=f"Acidosis CS\u2212 ({int(c_acid_cs_neg.sum())})",
           color='#c0392b', edgecolor='white', alpha=0.85)
    ax.bar(bin_centers, c_acid_cs_pos, width=bar_width, bottom=c_acid_cs_neg,
           label=f"Acidosis CS+ ({int(c_acid_cs_pos.sum())})",
           color='#e74c3c', edgecolor='white', alpha=0.85)
    ax.set_title("Acidosis — CS stratification")
    ax.set_ylabel("Epochs")
    ax.set_xlabel("Hours before birth")
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    # --- Row 1, Col 1: Acidosis summary text ---
    ax = axes[1, 1]
    ax.axis('off')
    n_acid = int((df['target'] == 2).sum())
    n_acid_guids = int(df[df['target'] == 2]['guid'].nunique())
    ax.text(0.1, 0.7,
            f"Acidosis: {n_acid_guids} GUIDs, {n_acid} epochs\n"
            f"  CS+: {int(c_acid_cs_pos.sum())} epochs\n"
            f"  CS\u2212: {int(c_acid_cs_neg.sum())} epochs",
            transform=ax.transAxes, fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fadbd8', alpha=0.4))

    # --- Row 2, Col 0: HIE by CS ---
    ax = axes[2, 0]
    c_hie_cs_pos = _bin_counts(lambda d: (d['target'] == 3) & (d['cs_label'] == True))   # noqa: E712
    c_hie_cs_neg = _bin_counts(lambda d: (d['target'] == 3) & (d['cs_label'] == False))  # noqa: E712
    ax.bar(bin_centers, c_hie_cs_neg, width=bar_width,
           label=f"HIE CS\u2212 ({int(c_hie_cs_neg.sum())})",
           color='#d35400', edgecolor='white', alpha=0.85)
    ax.bar(bin_centers, c_hie_cs_pos, width=bar_width, bottom=c_hie_cs_neg,
           label=f"HIE CS+ ({int(c_hie_cs_pos.sum())})",
           color='#e67e22', edgecolor='white', alpha=0.85)
    ax.set_title("HIE — CS stratification")
    ax.set_ylabel("Epochs")
    ax.set_xlabel("Hours before birth")
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    # --- Row 2, Col 1: HIE summary text ---
    ax = axes[2, 1]
    ax.axis('off')
    n_hie = int((df['target'] == 3).sum())
    n_hie_guids = int(df[df['target'] == 3]['guid'].nunique())
    ax.text(0.1, 0.7,
            f"HIE: {n_hie_guids} GUIDs, {n_hie} epochs\n"
            f"  CS+: {int(c_hie_cs_pos.sum())} epochs\n"
            f"  CS\u2212: {int(c_hie_cs_neg.sum())} epochs",
            transform=ax.transAxes, fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fdebd0', alpha=0.4))

    # --- Row 3, Col 0: Healthy by BG×CS 4-way ---
    ax = axes[3, 0]
    c_bg_pos_cs_pos = _bin_counts(lambda d: (d['target'] == 1) & (d['bg_label'] == True) & (d['cs_label'] == True))    # noqa: E712
    c_bg_pos_cs_neg = _bin_counts(lambda d: (d['target'] == 1) & (d['bg_label'] == True) & (d['cs_label'] == False))   # noqa: E712
    c_bg_neg_cs_pos = _bin_counts(lambda d: (d['target'] == 1) & (d['bg_label'] == False) & (d['cs_label'] == True))   # noqa: E712
    c_bg_neg_cs_neg = _bin_counts(lambda d: (d['target'] == 1) & (d['bg_label'] == False) & (d['cs_label'] == False))  # noqa: E712

    bottom = np.zeros(n_bins)
    for counts, label, color in [
        (c_bg_neg_cs_neg, f"BG− CS− ({int(c_bg_neg_cs_neg.sum())})", '#9b59b6'),
        (c_bg_neg_cs_pos, f"BG− CS+ ({int(c_bg_neg_cs_pos.sum())})", '#f39c12'),
        (c_bg_pos_cs_neg, f"BG+ CS− ({int(c_bg_pos_cs_neg.sum())})", '#3498db'),
        (c_bg_pos_cs_pos, f"BG+ CS+ ({int(c_bg_pos_cs_pos.sum())})", '#e74c3c'),
    ]:
        ax.bar(bin_centers, counts, width=bar_width, bottom=bottom,
               label=label, color=color, edgecolor='white', alpha=0.85)
        bottom += counts
    ax.set_title("Healthy — BG×CS 4-way")
    ax.set_ylabel("Epochs")
    ax.set_xlabel("Hours before birth")
    ax.legend(fontsize=7)
    ax.invert_xaxis()

    # --- Row 3, Col 1: Healthy BG×CS summary text ---
    ax = axes[3, 1]
    ax.axis('off')
    n_healthy = int((df['target'] == 1).sum())
    n_healthy_guids = int(df[df['target'] == 1]['guid'].nunique())
    ax.text(0.1, 0.7,
            f"Healthy: {n_healthy_guids} GUIDs, {n_healthy} epochs\n"
            f"  BG+ CS+: {int(c_bg_pos_cs_pos.sum())} epochs\n"
            f"  BG+ CS−: {int(c_bg_pos_cs_neg.sum())} epochs\n"
            f"  BG− CS+: {int(c_bg_neg_cs_pos.sum())} epochs\n"
            f"  BG− CS−: {int(c_bg_neg_cs_neg.sum())} epochs",
            transform=ax.transAxes, fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#d5f5e3', alpha=0.4))

    plt.tight_layout()
    plt.savefig(output_dir / "epochs_per_time_bin_subgroups.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_epochs_per_guid_ranked(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str,
) -> None:
    """Ranked bar chart of epochs per GUID, coloured by diagnosis (Healthy/Acidosis/HIE)."""
    guid_info = df.groupby('guid').agg(
        n_epochs=('epoch', 'size'),
        target=('target', 'first'),
    ).sort_values('n_epochs', ascending=False).reset_index()

    diag_colors = {1: '#2ecc71', 2: '#e74c3c', 3: '#e67e22'}
    colors = [diag_colors.get(t, '#95a5a6') for t in guid_info['target']]

    x = np.arange(len(guid_info))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, guid_info['n_epochs'].values, color=colors, alpha=0.85, width=1.0)
    ax.set_xlabel("GUID rank (sorted by epoch count)")
    ax.set_ylabel("Number of epochs")
    ax.set_title(f"Epochs per GUID (ranked) — {title_suffix}")

    mean_val = float(guid_info['n_epochs'].mean())
    ax.axhline(mean_val, color='#3498db', linestyle='--', linewidth=1.5,
               label=f"Mean: {mean_val:.1f}")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.85, label='Healthy'),
        Patch(facecolor='#e74c3c', alpha=0.85, label='Acidosis'),
        Patch(facecolor='#e67e22', alpha=0.85, label='HIE'),
        ax.get_lines()[0],
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "epochs_per_guid_ranked.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_label_cross_table(df: pd.DataFrame, output_dir: Path) -> None:
    """Save target x cs_label x bg_label cross-tabulation (GUID + epoch counts)."""
    guid_info = df.groupby('guid').agg({
        'target': 'first',
        'cs_label': 'first',
        'bg_label': 'first',
    }).reset_index()

    target_map = {1: 'Healthy', 2: 'Acidosis', 3: 'HIE'}

    rows = []
    for tgt_val, tgt_name in target_map.items():
        for cs_val in [False, True]:
            for bg_val in [False, True]:
                guid_mask = (
                    (guid_info['target'] == tgt_val) &
                    (guid_info['cs_label'] == cs_val) &
                    (guid_info['bg_label'] == bg_val)
                )
                epoch_mask = (
                    (df['target'] == tgt_val) &
                    (df['cs_label'] == cs_val) &
                    (df['bg_label'] == bg_val)
                )
                rows.append({
                    'target': tgt_name,
                    'cs_label': int(cs_val),
                    'bg_label': int(bg_val),
                    'n_guids': int(guid_mask.sum()),
                    'n_epochs': int(epoch_mask.sum()),
                })

    cross_df = pd.DataFrame(rows)
    cross_df.to_csv(output_dir / "label_cross_table.csv", index=False)


def compute_subgroup_metrics_by_time(
    df: pd.DataFrame,
    subgroup_name: str,
    subgroup_filter: callable,
    time_bins: np.ndarray,
    guid_level: bool = False
) -> pd.DataFrame:
    """
    Compute sensitivity/specificity for a specific subgroup across time bins.

    Args:
        df: Full predictions dataframe
        subgroup_name: Name of subgroup (for labeling)
        subgroup_filter: Function returning boolean mask for subgroup
        time_bins: Time bin edges in hours
        guid_level: If True, compute GUID-level cumulative sensitivity (monotonically increasing)
                   If False, compute epoch-level sensitivity within each bin

    Returns:
        DataFrame with: bin_center, sensitivity, specificity, n_samples,
                       n_detected, prevalence, guid_sensitivity (if guid_level=True)
    """
    # Apply subgroup filter
    subgroup_df = df[subgroup_filter(df)].copy()

    if len(subgroup_df) == 0:
        logger.warning(f"No samples in subgroup: {subgroup_name}")
        return pd.DataFrame()

    # Ensure epoch_hours exists (positive hours before birth)
    subgroup_df = ensure_epoch_hours(subgroup_df)

    results = []

    # For GUID-level cumulative metrics, pre-compute total unhealthy GUIDs
    if guid_level:
        unhealthy_guids = subgroup_df[subgroup_df['binary_target'] == 1]['guid'].unique()
        total_unhealthy_guids = len(unhealthy_guids)

    for i in range(len(time_bins) - 1):
        bin_start, bin_end = time_bins[i], time_bins[i + 1]
        bin_center = (bin_start + bin_end) / 2

        # Filter to time bin
        bin_mask = (subgroup_df['epoch_hours'] >= bin_start) & (subgroup_df['epoch_hours'] < bin_end)
        bin_data = subgroup_df[bin_mask]

        if len(bin_data) == 0:
            result = {
                'bin_center': bin_center,
                'n_samples': 0,
                'sensitivity': np.nan,
                'specificity': np.nan,
                'n_detected': 0,
                'prevalence': 0.0
            }
            if guid_level:
                result['guid_sensitivity'] = np.nan
                result['guid_specificity'] = np.nan
            results.append(result)
            continue

        # Compute epoch-level metrics within this bin
        class_0 = bin_data[bin_data['binary_target'] == 0]
        class_1 = bin_data[bin_data['binary_target'] == 1]

        epoch_sensitivity = (class_1['clinical_pred'] == 1).mean() if len(class_1) > 0 else np.nan
        epoch_specificity = 1 - (class_0['clinical_pred'] == 1).mean() if len(class_0) > 0 else np.nan
        n_detected = (bin_data['clinical_pred'] == 1).sum()
        prevalence = len(class_1) / len(bin_data) if len(bin_data) > 0 else 0.0

        result = {
            'bin_center': bin_center,
            'n_samples': len(bin_data),
            'sensitivity': epoch_sensitivity,
            'specificity': epoch_specificity,
            'n_detected': n_detected,
            'prevalence': prevalence
        }

        # Compute GUID-level metrics at this decision time (no look-ahead) if requested
        if guid_level:
            # At time T hours before birth, only epochs with start >= T are available.
            available = subgroup_df[subgroup_df['epoch_hours'] >= bin_center].copy()

            if len(available) == 0:
                guid_sensitivity = np.nan
                guid_specificity = np.nan
                n_unhealthy_guids = 0
                n_detected_guids = 0
            else:
                idx = available.groupby('guid')['epoch_hours'].idxmin()
                snap = available.loc[idx, ['guid', 'clinical_pred']].rename(columns={'clinical_pred': 'pred_at_time'})
                truth = subgroup_df.groupby('guid', sort=False)['binary_target'].max().rename('binary_target')
                snap = snap.set_index('guid').join(truth, how='inner').reset_index()

                unhealthy_guid_preds = snap[snap['binary_target'] == 1]
                healthy_guid_preds = snap[snap['binary_target'] == 0]

                guid_sensitivity = (unhealthy_guid_preds['pred_at_time'] == 1).mean() if len(unhealthy_guid_preds) > 0 else np.nan
                guid_specificity = 1 - (healthy_guid_preds['pred_at_time'] == 1).mean() if len(healthy_guid_preds) > 0 else np.nan
                n_unhealthy_guids = len(unhealthy_guid_preds)
                n_detected_guids = (unhealthy_guid_preds['pred_at_time'] == 1).sum() if len(unhealthy_guid_preds) > 0 else 0

            result['guid_sensitivity'] = guid_sensitivity
            result['guid_specificity'] = guid_specificity
            result['n_unhealthy_guids'] = n_unhealthy_guids
            result['n_detected_guids'] = n_detected_guids

        results.append(result)

    df_result = pd.DataFrame(results)
    df_result['subgroup'] = subgroup_name

    # MONOTONICITY VALIDATION: For GUID-level metrics, sensitivity should be non-decreasing
    # as we approach delivery (time before birth decreases).
    if guid_level and 'guid_sensitivity' in df_result.columns:
        # Remove rows with NaN sensitivity for validation
        valid_rows = df_result[df_result['guid_sensitivity'].notna()].sort_values('bin_center', ascending=False)

        if len(valid_rows) > 1:
            sensitivities = valid_rows['guid_sensitivity'].values
            violations = []

            for i in range(1, len(sensitivities)):
                # As we move to later bins (closer to birth), sensitivity should not decrease
                if sensitivities[i] < sensitivities[i-1] - 1e-6:  # Allow small numerical error
                    violations.append({
                        'bin_index': i,
                        'bin_center': valid_rows.iloc[i]['bin_center'],
                        'prev_sensitivity': sensitivities[i-1],
                        'curr_sensitivity': sensitivities[i],
                        'decrease': sensitivities[i-1] - sensitivities[i]
                    })

            if violations:
                logger.warning(f"GUID-level monotonicity violations detected in {subgroup_name}:")
                for v in violations:
                    logger.warning(
                        f"  Bin {v['bin_index']} ({v['bin_center']:.1f}h): "
                        f"Sensitivity decreased from {v['prev_sensitivity']:.4f} to {v['curr_sensitivity']:.4f} "
                        f"(Δ = {v['decrease']:.4f})"
                    )
            else:
                logger.debug(f"  ✓ GUID-level sensitivity is monotonically non-decreasing for {subgroup_name}")

    return df_result


# =============================================================================
# New Metric Computation Functions (Instantaneous + Committed Decisions)
# =============================================================================

def compute_instantaneous_metrics(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    subgroup_filter: Optional[callable] = None
) -> pd.DataFrame:
    """
    Compute instantaneous decision metrics at specific time bins.

    For each time bin τ:
    - Filter epochs in bin: epoch_hours in [bin_start, bin_end)
    - Sensitivity(τ) = TP(t=τ) / P(t=τ)
    - FPR(τ) = FP(t=τ) / N(t=τ)
    - Specificity(τ) = TN(t=τ) / N(t=τ) = 1 - FPR(τ)

    This metric shows performance at a specific time window only.
    NOT monotonic - can fluctuate based on which epochs fall in each bin.

    Args:
        df: DataFrame with columns ['guid', 'epoch_hours', 'binary_target', 'clinical_pred']
        time_bins: Array of time bin edges (hours before birth)
        subgroup_filter: Optional filter function to apply to df before computation

    Returns:
        DataFrame with columns:
            - bin_center: Center of time bin (hours before birth)
            - sensitivity: TP / P in this bin
            - specificity: TN / N in this bin
            - fpr: FP / N in this bin
            - n_positive: Number of positive cases in bin
            - n_negative: Number of negative cases in bin
            - n_tp: True positives in bin
            - n_fp: False positives in bin
            - n_tn: True negatives in bin
            - n_fn: False negatives in bin
    """
    # Apply subgroup filter if provided
    if subgroup_filter is not None:
        df = df[subgroup_filter(df)].copy()

    if len(df) == 0:
        logger.warning("compute_instantaneous_metrics: Empty dataframe after subgroup filter")
        return pd.DataFrame()

    # Ensure required columns
    df = ensure_epoch_hours(df.copy())

    results = []

    for i in range(len(time_bins) - 1):
        bin_start, bin_end = time_bins[i], time_bins[i + 1]
        bin_center = (bin_start + bin_end) / 2

        # Filter epochs in this time bin only
        bin_mask = (df['epoch_hours'] >= bin_start) & (df['epoch_hours'] < bin_end)
        bin_data = df[bin_mask]

        if len(bin_data) == 0:
            # No data in this bin - record NaN
            results.append({
                'bin_center': bin_center,
                'sensitivity': np.nan,
                'specificity': np.nan,
                'fpr': np.nan,
                'n_positive': 0,
                'n_negative': 0,
                'n_tp': 0,
                'n_fp': 0,
                'n_tn': 0,
                'n_fn': 0
            })
            continue

        # Count positives and negatives in this bin
        positives = bin_data[bin_data['binary_target'] == 1]
        negatives = bin_data[bin_data['binary_target'] == 0]

        n_positive = len(positives)
        n_negative = len(negatives)

        # Count TP, FP, TN, FN in this bin
        n_tp = ((bin_data['binary_target'] == 1) & (bin_data['clinical_pred'] == 1)).sum()
        n_fp = ((bin_data['binary_target'] == 0) & (bin_data['clinical_pred'] == 1)).sum()
        n_tn = ((bin_data['binary_target'] == 0) & (bin_data['clinical_pred'] == 0)).sum()
        n_fn = ((bin_data['binary_target'] == 1) & (bin_data['clinical_pred'] == 0)).sum()

        # Compute metrics
        sensitivity = n_tp / n_positive if n_positive > 0 else np.nan
        fpr = n_fp / n_negative if n_negative > 0 else np.nan
        specificity = n_tn / n_negative if n_negative > 0 else np.nan

        results.append({
            'bin_center': bin_center,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'fpr': fpr,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'n_tp': n_tp,
            'n_fp': n_fp,
            'n_tn': n_tn,
            'n_fn': n_fn
        })

    result_df = pd.DataFrame(results)

    logger.info(
        f"compute_instantaneous_metrics: Computed metrics for {len(result_df)} time bins "
        f"({result_df['sensitivity'].notna().sum()} non-NaN bins)"
    )

    return result_df


def compute_committed_cumulative_metrics(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    subgroup_filter: Optional[callable] = None
) -> pd.DataFrame:
    """
    Compute committed decision metrics with CHANGING denominator.

    For each time bin center τ:
    - Get GUIDs with data available at time τ: epoch_hours >= τ
    - For each GUID, check if detected using epochs at/earlier than τ (epoch_hours >= τ)
    - Sensitivity(τ) = TP(t≤τ) / P(t≤τ)
    - FPR(τ) = FP(t≤τ) / N(t≤τ)

    where:
    - TP(t≤τ) = detected unhealthy GUIDs (with data at τ)
    - P(t≤τ) = total unhealthy GUIDs available at time τ
    - FP(t≤τ) = detected healthy GUIDs (with data at τ)
    - N(t≤τ) = total healthy GUIDs available at time τ

    The denominator CHANGES as we approach birth (more GUIDs have data).
    Sensitivity can move up or down depending on who becomes available.

    Args:
        df: DataFrame with columns ['guid', 'epoch_hours', 'binary_target', 'clinical_pred']
        time_bins: Array of time bin edges (hours before birth)
        subgroup_filter: Optional filter function to apply to df before computation

    Returns:
        DataFrame with columns:
            - bin_center: Center of time bin (hours before birth)
            - sensitivity: Cumulative detection rate for positives available at τ
            - specificity: Cumulative true negative rate for negatives available at τ
            - fpr: Cumulative false positive rate for negatives available at τ
            - n_positive_available: Number of positive GUIDs available at τ
            - n_negative_available: Number of negative GUIDs available at τ
            - n_detected_positive: Number of detected positive GUIDs
            - n_detected_negative: Number of detected negative GUIDs
    """
    # Apply subgroup filter if provided
    if subgroup_filter is not None:
        df = df[subgroup_filter(df)].copy()

    if len(df) == 0:
        logger.warning("compute_committed_cumulative_metrics: Empty dataframe after subgroup filter")
        return pd.DataFrame()

    df = ensure_committed_epochs_filled(df)

    # Ensure required columns
    df = ensure_epoch_hours(df.copy())

    # Get GUID-level target labels (binary_target should be consistent per GUID)
    guid_targets = df.groupby('guid')['binary_target'].first()

    results = []

    for i in range(len(time_bins) - 1):
        bin_start, bin_end = time_bins[i], time_bins[i + 1]
        bin_center = (bin_start + bin_end) / 2

        # COMMITTED CUMULATIVE: Denominator is "cases available/monitored at time tau"
        # This means: babies that have data extending to tau or beyond (further from birth).
        # Implementation: epoch_hours >= bin_center
        #
        # As tau decreases (closer to birth):
        #   - More babies have data -> denominator increases
        #   - Detection window [tau, max] expands -> numerator can increase
        #   - Sensitivity can move in either direction as new GUIDs appear
        #
        # This is fundamentally different from committed_overall which has a FIXED denominator.
        available_mask = df['epoch_hours'] >= bin_center
        available_guids = df[available_mask]['guid'].unique()

        if len(available_guids) == 0:
            results.append({
                'bin_center': bin_center,
                'sensitivity': np.nan,
                'specificity': np.nan,
                'fpr': np.nan,
                'n_positive_available': 0,
                'n_negative_available': 0,
                'n_detected_positive': 0,
                'n_detected_negative': 0
            })
            continue

        # Split by target
        available_positive_guids = [g for g in available_guids if guid_targets.get(g, -1) == 1]
        available_negative_guids = [g for g in available_guids if guid_targets.get(g, -1) == 0]

        n_positive_available = len(available_positive_guids)
        n_negative_available = len(available_negative_guids)

        # For each GUID being monitored at tau, check if detected using epochs at/earlier than tau.
        detected_positive = 0
        for guid in available_positive_guids:
            guid_data = df[(df['guid'] == guid) & (df['epoch_hours'] >= bin_center)]
            if len(guid_data) > 0 and (guid_data['clinical_pred'] == 1).any():
                detected_positive += 1

        detected_negative = 0
        for guid in available_negative_guids:
            guid_data = df[(df['guid'] == guid) & (df['epoch_hours'] >= bin_center)]
            if len(guid_data) > 0 and (guid_data['clinical_pred'] == 1).any():
                detected_negative += 1

        # Compute metrics
        sensitivity = detected_positive / n_positive_available if n_positive_available > 0 else np.nan
        fpr = detected_negative / n_negative_available if n_negative_available > 0 else np.nan
        specificity = (n_negative_available - detected_negative) / n_negative_available if n_negative_available > 0 else np.nan

        results.append({
            'bin_center': bin_center,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'fpr': fpr,
            'n_positive_available': n_positive_available,
            'n_negative_available': n_negative_available,
            'n_detected_positive': detected_positive,
            'n_detected_negative': detected_negative
        })

    result_df = pd.DataFrame(results)

    # Sort by bin_center in DESCENDING order (far from birth -> near birth)
    result_df = result_df.sort_values('bin_center', ascending=False).reset_index(drop=True)

    logger.info(
        f"compute_committed_cumulative_metrics: Computed metrics for {len(result_df)} time bins "
        f"({result_df['sensitivity'].notna().sum()} non-NaN bins)"
    )

    # Sensitivity is not guaranteed to be monotonic with a changing denominator.
    valid_sens = result_df['sensitivity'].dropna()
    if len(valid_sens) > 1:
        violations = (valid_sens.diff() < -1e-6).sum()
        if violations > 0:
            logger.debug(f"Committed cumulative sensitivity decreases in {violations} bins (expected with changing denominator)")
        else:
            logger.debug("Committed cumulative sensitivity is non-decreasing across bins")

    return result_df


def compute_committed_overall_metrics(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    subgroup_filter: Optional[callable] = None
) -> pd.DataFrame:
    """
    Compute committed decision metrics with FIXED denominator (PRIMARY METRIC).

    For each time bin center τ:
    - Get ALL GUIDs in dataset: P(t≤0) and N(t≤0) are FIXED
    - For GUIDs with data at time tau, check if detected using epochs at/earlier than tau (epoch_hours >= tau)
    - Sensitivity(τ) = TP(t≤τ) / P(t≤0)
    - FPR(τ) = FP(t≤τ) / N(t≤0)

    where:
    - TP(t≤τ) = detected unhealthy GUIDs (with data at τ)
    - P(t≤0) = ALL unhealthy GUIDs in dataset (FIXED)
    - FP(t≤τ) = detected healthy GUIDs (with data at τ)
    - N(t≤0) = ALL healthy GUIDs in dataset (FIXED)

    GUIDs not yet available at time τ contribute 0 to numerator (not detected yet).

    The denominator is FIXED across all time bins.
    This is the PRIMARY METRIC for clinical reporting.
    MUST be monotonically non-decreasing.

    Args:
        df: DataFrame with columns ['guid', 'epoch_hours', 'binary_target', 'clinical_pred']
        time_bins: Array of time bin edges (hours before birth)
        subgroup_filter: Optional filter function to apply to df before computation

    Returns:
        DataFrame with columns:
            - bin_center: Center of time bin (hours before birth)
            - sensitivity: Cumulative detection rate (detected / ALL positives)
            - specificity: Cumulative true negative rate (not detected / ALL negatives)
            - fpr: Cumulative false positive rate (detected / ALL negatives)
            - n_positive_total: Total positive GUIDs in dataset (FIXED)
            - n_negative_total: Total negative GUIDs in dataset (FIXED)
            - n_detected_positive: Number of detected positive GUIDs at τ
            - n_detected_negative: Number of detected negative GUIDs at τ
            - n_available_positive: Number of positive GUIDs with data at τ
            - n_available_negative: Number of negative GUIDs with data at τ
    """
    # Apply subgroup filter if provided
    if subgroup_filter is not None:
        df = df[subgroup_filter(df)].copy()

    if len(df) == 0:
        logger.warning("compute_committed_overall_metrics: Empty dataframe after subgroup filter")
        return pd.DataFrame()

    df = ensure_committed_epochs_filled(df)

    # Ensure required columns
    df = ensure_epoch_hours(df.copy())

    # Get ALL GUIDs and their targets (FIXED denominator)
    guid_targets = df.groupby('guid')['binary_target'].first()
    all_positive_guids = guid_targets[guid_targets == 1].index.tolist()
    all_negative_guids = guid_targets[guid_targets == 0].index.tolist()

    n_positive_total = len(all_positive_guids)  # FIXED
    n_negative_total = len(all_negative_guids)  # FIXED

    logger.info(
        f"compute_committed_overall_metrics: FIXED denominators - "
        f"P(t≤0)={n_positive_total}, N(t≤0)={n_negative_total}"
    )

    results = []

    for i in range(len(time_bins) - 1):
        bin_start, bin_end = time_bins[i], time_bins[i + 1]
        bin_center = (bin_start + bin_end) / 2

        logger.debug(f"  [BIN {i}] bin_center={bin_center:.2f}h, range=[{bin_start:.2f}, {bin_end:.2f})")

        # For COMMITTED OVERALL with FIXED denominator:
        # We check ALL positive/negative GUIDs to see if they were detected using epochs at/earlier than tau.
        # We DON'T restrict to "available" GUIDs because once detected, they should stay detected
        # at all smaller tau values (closer to birth) (monotonicity requirement)

        # Count detections for ALL positive GUIDs (not just those with data at τ)
        detected_positive = 0
        for guid in all_positive_guids:
            # Check if this GUID has ANY epoch with clinical_pred=1 with epoch_hours >= tau
            guid_data = df[(df['guid'] == guid) & (df['epoch_hours'] >= bin_center)]
            if len(guid_data) > 0 and (guid_data['clinical_pred'] == 1).any():
                detected_positive += 1

        detected_negative = 0
        for guid in all_negative_guids:
            # Check if this GUID has ANY epoch with clinical_pred=1 with epoch_hours >= tau
            guid_data = df[(df['guid'] == guid) & (df['epoch_hours'] >= bin_center)]
            if len(guid_data) > 0 and (guid_data['clinical_pred'] == 1).any():
                detected_negative += 1

        # For reporting: also track how many GUIDs have data extending to τ
        available_mask = df['epoch_hours'] >= bin_center
        available_guids = df[available_mask]['guid'].unique()
        n_available_positive = len([g for g in available_guids if g in all_positive_guids])
        n_available_negative = len([g for g in available_guids if g in all_negative_guids])

        logger.debug(f"  [BIN {i}] GUIDs with data at τ (epoch_hours >= {bin_center:.2f}): {n_available_positive} positive, {n_available_negative} negative")
        logger.debug(f"  [BIN {i}] Detected (clinical_pred=1 where epoch_hours >= {bin_center:.2f}): {detected_positive}/{n_positive_total} positive")
        logger.debug(f"  [BIN {i}] Sensitivity = {detected_positive} / {n_positive_total} = {detected_positive/n_positive_total if n_positive_total > 0 else 0:.4f}")

        # Compute metrics with FIXED denominator
        sensitivity = detected_positive / n_positive_total if n_positive_total > 0 else np.nan
        fpr = detected_negative / n_negative_total if n_negative_total > 0 else np.nan
        specificity = (n_negative_total - detected_negative) / n_negative_total if n_negative_total > 0 else np.nan

        results.append({
            'bin_center': bin_center,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'fpr': fpr,
            'n_positive_total': n_positive_total,
            'n_negative_total': n_negative_total,
            'n_detected_positive': detected_positive,
            'n_detected_negative': detected_negative,
            'n_available_positive': n_available_positive,
            'n_available_negative': n_available_negative
        })

    result_df = pd.DataFrame(results)

    # Sort by bin_center in DESCENDING order (far from birth -> near birth)
    # This aligns monotonicity checks with approaching birth (tau decreasing)
    result_df = result_df.sort_values('bin_center', ascending=False).reset_index(drop=True)

    logger.info(
        f"compute_committed_overall_metrics: Computed PRIMARY METRIC for {len(result_df)} time bins "
        f"({result_df['sensitivity'].notna().sum()} non-NaN bins)"
    )

    # Verify monotonicity (CRITICAL for primary metric)
    # After sorting, bins go from large tau to small tau (far->near from birth)
    # Sensitivity should be non-decreasing as we approach birth
    valid_sens = result_df['sensitivity'].dropna()
    if len(valid_sens) > 1:
        violations = (valid_sens.diff() < -1e-6).sum()
        if violations > 0:
            logger.error(f"❌ PRIMARY METRIC violation: Committed overall sensitivity has {violations} monotonicity violations!")
            # Log details
            for idx in range(1, len(result_df)):
                prev_sens = result_df.iloc[idx - 1]['sensitivity']
                curr_sens = result_df.iloc[idx]['sensitivity']
                prev_time = result_df.iloc[idx - 1]['bin_center']
                curr_time = result_df.iloc[idx]['bin_center']
                if pd.notna(prev_sens) and pd.notna(curr_sens) and curr_sens < prev_sens - 1e-6:
                    logger.error(
                        f"  Bin {idx} (τ={curr_time:.1f}h): "
                        f"Sensitivity decreased from {prev_sens:.4f} (at τ={prev_time:.1f}h) to {curr_sens:.4f}"
                    )
        else:
            logger.info("✓ PRIMARY METRIC: Committed overall sensitivity is monotonically non-decreasing")

    return result_df


def compute_all_metric_types(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    subgroup_filter: Optional[callable] = None
) -> Dict[str, pd.DataFrame]:
    """
    Wrapper to compute all three metric types efficiently.

    Args:
        df: DataFrame with columns ['guid', 'epoch_hours', 'binary_target', 'clinical_pred']
        time_bins: Array of time bin edges (hours before birth)
        subgroup_filter: Optional filter function to apply before computing metrics

    Returns:
        Dictionary with keys:
            - 'instantaneous': DataFrame from compute_instantaneous_metrics()
            - 'committed_cumulative': DataFrame from compute_committed_cumulative_metrics()
            - 'committed_overall': DataFrame from compute_committed_overall_metrics() [PRIMARY]
    """
    logger.info("=" * 80)
    logger.info("Computing all three metric types...")
    logger.info("=" * 80)

    # Compute each metric type
    instantaneous_df = compute_instantaneous_metrics(df, time_bins, subgroup_filter)
    committed_cumulative_df = compute_committed_cumulative_metrics(df, time_bins, subgroup_filter)
    committed_overall_df = compute_committed_overall_metrics(df, time_bins, subgroup_filter)

    logger.info("All metric types computed successfully")

    return {
        'instantaneous': instantaneous_df,
        'committed_cumulative': committed_cumulative_df,
        'committed_overall': committed_overall_df  # PRIMARY METRIC
    }


# =============================================================================
# New Plotting Functions (Three Metric Types)
# =============================================================================

def annotate_decision_time(
    ax,
    *,
    decision_time_hours: Optional[float],
    label: str = "decision time",
) -> None:
    """Draw a dashed vertical reference line at ``decision_time_hours``.

    No-op when ``decision_time_hours`` is None / NaN, or when the value
    falls outside the current x-axis range. Intended to be called on
    every "vs hours-before-birth" plot so the threshold-search operating
    point is visually obvious. Does not change axis limits.

    Args:
        ax: matplotlib axes (already drawn so xlim is set).
        decision_time_hours: Time in hours before birth at which the
            classifier's operating point was selected (typically
            ``evaluation.decision_time_hours``).
        label: Annotation text drawn above the line.
    """
    if decision_time_hours is None:
        return
    try:
        x = float(decision_time_hours)
    except (TypeError, ValueError):
        return
    if not np.isfinite(x):
        return
    xlim = ax.get_xlim()
    lo, hi = (min(xlim), max(xlim))
    if x < lo or x > hi:
        return
    ax.axvline(
        x=x,
        color="0.4",
        linestyle="--",
        linewidth=1.0,
        zorder=0,
    )
    ylim = ax.get_ylim()
    ax.text(
        x=x,
        y=ylim[1],
        s=f" {label} ({x:g}h)",
        fontsize=7,
        color="0.4",
        va="top",
        ha="left",
    )


def plot_single_metric_type(
    metrics_df: pd.DataFrame,
    metric_type: str,
    output_dir: Path,
    title_suffix: str = "",
    *,
    decision_time_hours: Optional[float] = None,
) -> None:
    """
    Create comprehensive plots for a single metric type.

    Generates multiple plot variations:
    1. sensitivity_vs_time.png - Sensitivity only
    2. sensitivity_specificity_vs_time.png - Sensitivity + Specificity
    3. sensitivity_fpr_vs_time.png - Sensitivity + FPR
    4. all_metrics_vs_time.png - All three metrics together
    5. fpr_vs_time.png - FPR only (emphasized metric)

    Args:
        metrics_df: DataFrame with columns ['bin_center', 'sensitivity', 'specificity', 'fpr']
        metric_type: One of 'instantaneous', 'committed_cumulative', 'committed_overall'
        output_dir: Directory to save plots
        title_suffix: Additional suffix for plot titles (e.g., "Test Set", "Fold 1")
    """
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    if metrics_df is None or len(metrics_df) == 0:
        logger.warning(f"No data to plot for {metric_type}")
        return

    # Filter out NaN values
    valid_df = metrics_df[metrics_df['sensitivity'].notna()].copy()
    if len(valid_df) == 0:
        logger.warning(f"No valid (non-NaN) data to plot for {metric_type}")
        return

    valid_df = valid_df.sort_values('bin_center', ascending=False)
    x = valid_df['bin_center'].values

    # Metric type labels
    metric_labels = {
        'instantaneous': 'Instantaneous Decisions',
        'committed_cumulative': 'Committed Decisions (Cumulative)',
        'committed_overall': 'Committed Decisions (Overall) - PRIMARY'
    }
    metric_label = metric_labels.get(metric_type, metric_type)

    # Color scheme
    colors = {
        'sensitivity': '#2ecc71',  # Green
        'specificity': '#3498db',  # Blue
        'fpr': '#e74c3c'  # Red
    }

    # --- Plot 1: Sensitivity Only ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, valid_df['sensitivity'], marker='o', label='Sensitivity',
            color=colors['sensitivity'], linewidth=2.5, markersize=6)
    ax.set_xlabel('Hours Before Birth', fontsize=13)
    ax.set_ylabel('Sensitivity', fontsize=13)
    title = f'Sensitivity vs Time - {metric_label}'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.invert_xaxis()
    annotate_decision_time(ax, decision_time_hours=decision_time_hours)
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_vs_time.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_dir.name}/sensitivity_vs_time.png")

    # --- Plot 2: Sensitivity + Specificity ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, valid_df['sensitivity'], marker='o', label='Sensitivity',
            color=colors['sensitivity'], linewidth=2.5, markersize=6)
    ax.plot(x, valid_df['specificity'], marker='s', label='Specificity',
            color=colors['specificity'], linewidth=2.5, markersize=6)
    ax.set_xlabel('Hours Before Birth', fontsize=13)
    ax.set_ylabel('Metric Value', fontsize=13)
    title = f'Sensitivity & Specificity vs Time - {metric_label}'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.invert_xaxis()
    annotate_decision_time(ax, decision_time_hours=decision_time_hours)
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_specificity_vs_time.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_dir.name}/sensitivity_specificity_vs_time.png")

    # --- Plot 3: Sensitivity + FPR ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, valid_df['sensitivity'], marker='o', label='Sensitivity',
            color=colors['sensitivity'], linewidth=2.5, markersize=6)
    ax.plot(x, valid_df['fpr'], marker='^', label='FPR',
            color=colors['fpr'], linewidth=2.5, markersize=6)
    ax.set_xlabel('Hours Before Birth', fontsize=13)
    ax.set_ylabel('Metric Value', fontsize=13)
    title = f'Sensitivity & FPR vs Time - {metric_label}'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.invert_xaxis()
    annotate_decision_time(ax, decision_time_hours=decision_time_hours)
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_fpr_vs_time.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_dir.name}/sensitivity_fpr_vs_time.png")

    # --- Plot 4: All Three Metrics ---
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(x, valid_df['sensitivity'], marker='o', label='Sensitivity',
            color=colors['sensitivity'], linewidth=2.5, markersize=7, linestyle='-')
    ax.plot(x, valid_df['specificity'], marker='s', label='Specificity',
            color=colors['specificity'], linewidth=2.5, markersize=7, linestyle='--')
    ax.plot(x, valid_df['fpr'], marker='^', label='FPR',
            color=colors['fpr'], linewidth=2.5, markersize=7, linestyle=':')
    ax.set_xlabel('Hours Before Birth', fontsize=14)
    ax.set_ylabel('Metric Value', fontsize=14)
    title = f'All Metrics vs Time - {metric_label}'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.invert_xaxis()
    annotate_decision_time(ax, decision_time_hours=decision_time_hours)
    plt.tight_layout()
    plt.savefig(output_dir / "all_metrics_vs_time.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_dir.name}/all_metrics_vs_time.png")

    # --- Plot 5: FPR Only (emphasized metric) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, valid_df['fpr'], marker='^', label='FPR',
            color=colors['fpr'], linewidth=2.5, markersize=6)
    ax.set_xlabel('Hours Before Birth', fontsize=13)
    ax.set_ylabel('FPR', fontsize=13)
    title = f'FPR vs Time - {metric_label}'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.invert_xaxis()
    annotate_decision_time(ax, decision_time_hours=decision_time_hours)
    plt.tight_layout()
    plt.savefig(output_dir / "fpr_vs_time.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_dir.name}/fpr_vs_time.png")


def plot_all_metric_types_for_fold(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    output_base_dir: Path,
    title_suffix: str = ""
) -> Dict[str, pd.DataFrame]:
    """
    Generate complete plot sets for all three metric types.

    Creates directory structure:
    output_base_dir/
        instantaneous/
            sensitivity_vs_time.png
            sensitivity_specificity_vs_time.png
            sensitivity_fpr_vs_time.png
            all_metrics_vs_time.png
        committed_cumulative/
            [same plots]
        committed_overall/  # PRIMARY
            [same plots]

    Args:
        df: DataFrame with clinical predictions
        time_bins: Time bin edges
        output_base_dir: Base directory for all plots
        title_suffix: Suffix for plot titles

    Returns:
        Dictionary with all three metric DataFrames
    """
    logger.info("=" * 80)
    logger.info("Generating plots for all three metric types...")
    logger.info("=" * 80)

    # Compute all three metric types
    metrics_dict = compute_all_metric_types(df, time_bins, subgroup_filter=None)

    # Plot each metric type
    for metric_type in ['instantaneous', 'committed_cumulative', 'committed_overall']:
        metric_output_dir = output_base_dir / metric_type
        logger.info(f"Plotting {metric_type}...")

        plot_single_metric_type(
            metrics_dict[metric_type],
            metric_type,
            metric_output_dir,
            title_suffix
        )

    logger.info("All metric type plots generated successfully")
    return metrics_dict


def plot_metric_type_comparison(
    metrics_dict: Dict[str, pd.DataFrame],
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """
    Side-by-side comparison of all three metric types.

    Creates a 3-column layout showing:
    - Instantaneous | Committed Cumulative | Committed Overall (PRIMARY)

    Helps visualize differences between metric types.

    Args:
        metrics_dict: Dictionary with 'instantaneous', 'committed_cumulative', 'committed_overall' DataFrames
        output_dir: Directory to save comparison plot
        title_suffix: Additional suffix for title
    """
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    metric_types = ['instantaneous', 'committed_cumulative', 'committed_overall']
    titles = ['Instantaneous', 'Committed Cumulative', 'Committed Overall\n(PRIMARY)']
    colors = {'sensitivity': '#2ecc71', 'fpr': '#e74c3c'}

    for ax, metric_type, title in zip(axes, metric_types, titles):
        df = metrics_dict.get(metric_type)
        if df is None or len(df) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=13, fontweight='bold')
            continue

        valid_df = df[df['sensitivity'].notna()].copy()
        if len(valid_df) == 0:
            ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=13, fontweight='bold')
            continue

        valid_df = valid_df.sort_values('bin_center', ascending=False)
        x = valid_df['bin_center'].values

        ax.plot(x, valid_df['sensitivity'], marker='o', label='Sensitivity',
                color=colors['sensitivity'], linewidth=2, markersize=5)
        ax.plot(x, valid_df['fpr'], marker='^', label='FPR',
                color=colors['fpr'], linewidth=2, markersize=5)

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Hours Before Birth', fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel('Metric Value', fontsize=11)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.invert_xaxis()

    main_title = 'Metric Type Comparison: Sensitivity & FPR'
    if title_suffix:
        main_title += f' ({title_suffix})'
    fig.suptitle(main_title, fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_dir / "metric_type_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: comparison/metric_type_comparison.png")


def plot_subgroup_analysis(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    metric_type: str,
    subgroup_filters: Dict[str, callable],
    output_dir: Path,
    title_suffix: str = "",
    decision_time_hours: Optional[float] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate plots comparing metrics across subgroups for a specific metric type.

    Args:
        df: DataFrame with clinical predictions
        time_bins: Time bin edges
        metric_type: 'instantaneous', 'committed_cumulative', or 'committed_overall'
        subgroup_filters: Dictionary of subgroup_name -> filter_function
        output_dir: Directory to save subgroup plots
        title_suffix: Additional suffix for titles
        decision_time_hours: When provided, every subgroup time-axis plot
            gets a dashed vertical reference line at that x value (so the
            operating point used for threshold search is visually obvious
            on each panel). Mirrors the per-class plotters' behaviour.

    Returns:
        Dictionary mapping subgroup_name -> metrics DataFrame
    """
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Computing {metric_type} metrics for {len(subgroup_filters)} subgroups...")

    # Compute metrics for each subgroup
    subgroup_metrics = {}
    compute_func = {
        'instantaneous': compute_instantaneous_metrics,
        'committed_cumulative': compute_committed_cumulative_metrics,
        'committed_overall': compute_committed_overall_metrics
    }[metric_type]

    for subgroup_name, subgroup_filter in subgroup_filters.items():
        try:
            metrics_df = compute_func(df, time_bins, subgroup_filter)
            if metrics_df is not None and len(metrics_df) > 0:
                subgroup_metrics[subgroup_name] = metrics_df
        except Exception as e:
            logger.warning(f"Failed to compute {metric_type} for subgroup {subgroup_name}: {e}")

    # Compute GUID counts for legend labels
    subgroup_guid_counts = {}
    for name, filt in subgroup_filters.items():
        try:
            subgroup_guid_counts[name] = int(df[filt(df)]['guid'].nunique())
        except Exception:
            subgroup_guid_counts[name] = 0

    if len(subgroup_metrics) == 0:
        logger.warning(f"No valid subgroup metrics computed for {metric_type}")
        return {}

    # Create comparison plots by category
    _plot_diagnosis_comparison(subgroup_metrics, metric_type, output_dir, title_suffix, subgroup_guid_counts=subgroup_guid_counts, decision_time_hours=decision_time_hours)
    _plot_cs_stratification(subgroup_metrics, metric_type, output_dir, title_suffix, subgroup_guid_counts=subgroup_guid_counts, decision_time_hours=decision_time_hours)
    _plot_bg_stratification(subgroup_metrics, metric_type, output_dir, title_suffix, subgroup_guid_counts=subgroup_guid_counts, decision_time_hours=decision_time_hours)
    _plot_healthy_subgroups(subgroup_metrics, metric_type, output_dir, title_suffix, subgroup_guid_counts=subgroup_guid_counts, decision_time_hours=decision_time_hours)

    logger.info(f"Subgroup analysis plots generated for {metric_type}")
    return subgroup_metrics


def _plot_diagnosis_comparison(
    subgroup_metrics: Dict[str, pd.DataFrame],
    metric_type: str,
    output_dir: Path,
    title_suffix: str,
    subgroup_guid_counts: Optional[Dict[str, int]] = None,
    decision_time_hours: Optional[float] = None,
) -> None:
    """Plot comparison of basic diagnosis subgroups (healthy, acidosis, hie, unhealthy)."""
    import matplotlib.pyplot as plt

    diagnosis_groups = ['healthy', 'acidosis', 'hie', 'unhealthy']
    available_groups = [g for g in diagnosis_groups if g in subgroup_metrics]

    if len(available_groups) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'healthy': '#2ecc71', 'acidosis': '#e74c3c', 'hie': '#e67e22', 'unhealthy': '#95a5a6'}

    for group in available_groups:
        df = subgroup_metrics[group]
        valid_df = df[df['sensitivity'].notna()].sort_values('bin_center', ascending=False)
        if len(valid_df) > 0:
            n = subgroup_guid_counts.get(group, 0) if subgroup_guid_counts else 0
            label = f"{group.capitalize()} (N={n})" if n > 0 else group.capitalize()
            ax.plot(valid_df['bin_center'], valid_df['sensitivity'],
                   marker='o', label=label, linewidth=2.5,
                   color=colors.get(group, None), markersize=6)

    ax.set_xlabel('Hours Before Birth', fontsize=13)
    ax.set_ylabel('Sensitivity', fontsize=13)
    title = f'Diagnosis Comparison - {metric_type.replace("_", " ").title()}'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.invert_xaxis()
    annotate_decision_time(ax, decision_time_hours=decision_time_hours)
    plt.tight_layout()
    plt.savefig(output_dir / "diagnosis_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_dir.name}/diagnosis_comparison.png")


def _plot_cs_stratification(
    subgroup_metrics: Dict[str, pd.DataFrame],
    metric_type: str,
    output_dir: Path,
    title_suffix: str,
    subgroup_guid_counts: Optional[Dict[str, int]] = None,
    decision_time_hours: Optional[float] = None,
) -> None:
    """Plot CS stratification for unhealthy, HIE, and Acidosis separately."""
    import matplotlib.pyplot as plt

    stratifications = [
        (['unhealthy_cs_pos', 'unhealthy_cs_neg'], 'Unhealthy by CS Status'),
        (['hie_cs_pos', 'hie_cs_neg'], 'HIE by CS Status'),
        (['acidosis_cs_pos', 'acidosis_cs_neg'], 'Acidosis by CS Status')
    ]

    for groups, title_text in stratifications:
        available_groups = [g for g in groups if g in subgroup_metrics]
        if len(available_groups) == 0:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = {'pos': '#3498db', 'neg': '#9b59b6'}

        for group in available_groups:
            df = subgroup_metrics[group]
            valid_df = df[df['sensitivity'].notna()].sort_values('bin_center', ascending=False)
            if len(valid_df) > 0:
                base_label = 'CS Positive' if 'pos' in group else 'CS Negative'
                n = subgroup_guid_counts.get(group, 0) if subgroup_guid_counts else 0
                label = f"{base_label} (N={n})" if n > 0 else base_label
                color = colors['pos'] if 'pos' in group else colors['neg']
                ax.plot(valid_df['bin_center'], valid_df['sensitivity'],
                       marker='o', label=label, linewidth=2.5,
                       color=color, markersize=6)

        ax.set_xlabel('Hours Before Birth', fontsize=13)
        ax.set_ylabel('Sensitivity', fontsize=13)
        title = f'{title_text} - {metric_type.replace("_", " ").title()}'
        if title_suffix:
            title += f' ({title_suffix})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.invert_xaxis()
        annotate_decision_time(ax, decision_time_hours=decision_time_hours)
        plt.tight_layout()
        filename = f"{groups[0].rsplit('_', 2)[0]}_cs_stratification.png"
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: {output_dir.name}/{filename}")


def _plot_bg_stratification(
    subgroup_metrics: Dict[str, pd.DataFrame],
    metric_type: str,
    output_dir: Path,
    title_suffix: str,
    subgroup_guid_counts: Optional[Dict[str, int]] = None,
    decision_time_hours: Optional[float] = None,
) -> None:
    """Plot BG stratification for Acidosis only (all HIE are bg_positive)."""
    import matplotlib.pyplot as plt

    bg_groups = ['acidosis_bg_pos', 'acidosis_bg_neg']
    available_groups = [g for g in bg_groups if g in subgroup_metrics]

    if len(available_groups) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'pos': '#f39c12', 'neg': '#16a085'}

    for group in available_groups:
        df = subgroup_metrics[group]
        valid_df = df[df['sensitivity'].notna()].sort_values('bin_center', ascending=False)
        if len(valid_df) > 0:
            base_label = 'BG Positive' if 'pos' in group else 'BG Negative'
            n = subgroup_guid_counts.get(group, 0) if subgroup_guid_counts else 0
            label = f"{base_label} (N={n})" if n > 0 else base_label
            color = colors['pos'] if 'pos' in group else colors['neg']
            ax.plot(valid_df['bin_center'], valid_df['sensitivity'],
                   marker='o', label=label, linewidth=2.5,
                   color=color, markersize=6)

    ax.set_xlabel('Hours Before Birth', fontsize=13)
    ax.set_ylabel('Sensitivity', fontsize=13)
    title = f'Acidosis by BG Status - {metric_type.replace("_", " ").title()}'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.invert_xaxis()
    annotate_decision_time(ax, decision_time_hours=decision_time_hours)
    plt.tight_layout()
    plt.savefig(output_dir / "acidosis_bg_stratification.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_dir.name}/acidosis_bg_stratification.png")


def _plot_healthy_subgroups(
    subgroup_metrics: Dict[str, pd.DataFrame],
    metric_type: str,
    output_dir: Path,
    title_suffix: str,
    subgroup_guid_counts: Optional[Dict[str, int]] = None,
    decision_time_hours: Optional[float] = None,
) -> None:
    """
    Plot healthy subgroup stratifications (CS, BG, and combinations).

    NOTE: For healthy subgroups (all negative cases), we plot SPECIFICITY instead of
    sensitivity, since sensitivity is undefined (no positive cases to detect).
    Specificity shows how well we correctly identify healthy patients as healthy.
    """
    import matplotlib.pyplot as plt

    # Healthy by CS
    cs_groups = ['healthy_cs_pos', 'healthy_cs_neg']
    available_cs = [g for g in cs_groups if g in subgroup_metrics]

    if len(available_cs) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = {'pos': '#3498db', 'neg': '#9b59b6'}

        for group in available_cs:
            df = subgroup_metrics[group]
            # For healthy subgroups, use SPECIFICITY (not sensitivity which is NaN)
            valid_df = df[df['specificity'].notna()].sort_values('bin_center', ascending=False)
            if len(valid_df) > 0:
                base_label = 'CS Positive' if 'pos' in group else 'CS Negative'
                n = subgroup_guid_counts.get(group, 0) if subgroup_guid_counts else 0
                label = f"{base_label} (N={n})" if n > 0 else base_label
                color = colors['pos'] if 'pos' in group else colors['neg']
                ax.plot(valid_df['bin_center'], valid_df['specificity'],
                       marker='o', label=label, linewidth=2.5,
                       color=color, markersize=6)

        ax.set_xlabel('Hours Before Birth', fontsize=13)
        ax.set_ylabel('Specificity (Correctly Identified as Healthy)', fontsize=13)
        title = f'Healthy by CS Status - {metric_type.replace("_", " ").title()}'
        if title_suffix:
            title += f' ({title_suffix})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.invert_xaxis()
        annotate_decision_time(ax, decision_time_hours=decision_time_hours)
        plt.tight_layout()
        plt.savefig(output_dir / "healthy_cs_stratification.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: {output_dir.name}/healthy_cs_stratification.png")

    # Healthy by BG
    bg_groups = ['healthy_bg_pos', 'healthy_bg_neg']
    available_bg = [g for g in bg_groups if g in subgroup_metrics]

    if len(available_bg) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = {'pos': '#f39c12', 'neg': '#16a085'}

        for group in available_bg:
            df = subgroup_metrics[group]
            # For healthy subgroups, use SPECIFICITY (not sensitivity which is NaN)
            valid_df = df[df['specificity'].notna()].sort_values('bin_center', ascending=False)
            if len(valid_df) > 0:
                base_label = 'BG Positive' if 'pos' in group else 'BG Negative'
                n = subgroup_guid_counts.get(group, 0) if subgroup_guid_counts else 0
                label = f"{base_label} (N={n})" if n > 0 else base_label
                color = colors['pos'] if 'pos' in group else colors['neg']
                ax.plot(valid_df['bin_center'], valid_df['specificity'],
                       marker='o', label=label, linewidth=2.5,
                       color=color, markersize=6)

        ax.set_xlabel('Hours Before Birth', fontsize=13)
        ax.set_ylabel('Specificity (Correctly Identified as Healthy)', fontsize=13)
        title = f'Healthy by BG Status - {metric_type.replace("_", " ").title()}'
        if title_suffix:
            title += f' ({title_suffix})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.invert_xaxis()
        annotate_decision_time(ax, decision_time_hours=decision_time_hours)
        plt.tight_layout()
        plt.savefig(output_dir / "healthy_bg_stratification.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: {output_dir.name}/healthy_bg_stratification.png")

    # Healthy BG×CS 4-way combination
    combo_groups = ['healthy_bg_pos_cs_pos', 'healthy_bg_pos_cs_neg',
                    'healthy_bg_neg_cs_pos', 'healthy_bg_neg_cs_neg']
    available_combo = [g for g in combo_groups if g in subgroup_metrics]

    if len(available_combo) > 0:
        fig, ax = plt.subplots(figsize=(14, 7))
        colors = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6']

        for i, group in enumerate(available_combo):
            df = subgroup_metrics[group]
            # For healthy subgroups, use SPECIFICITY (not sensitivity which is NaN)
            valid_df = df[df['specificity'].notna()].sort_values('bin_center', ascending=False)
            if len(valid_df) > 0:
                base_label = group.replace('healthy_', '').replace('_', ' ').upper()
                n = subgroup_guid_counts.get(group, 0) if subgroup_guid_counts else 0
                label = f"{base_label} (N={n})" if n > 0 else base_label
                ax.plot(valid_df['bin_center'], valid_df['specificity'],
                       marker='o', label=label, linewidth=2.5,
                       color=colors[i % len(colors)], markersize=6)

        ax.set_xlabel('Hours Before Birth', fontsize=13)
        ax.set_ylabel('Specificity (Correctly Identified as Healthy)', fontsize=13)
        title = f'Healthy BG×CS Combinations - {metric_type.replace("_", " ").title()}'
        if title_suffix:
            title += f' ({title_suffix})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.invert_xaxis()
        annotate_decision_time(ax, decision_time_hours=decision_time_hours)
        plt.tight_layout()
        plt.savefig(output_dir / "healthy_bg_cs_combinations.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: {output_dir.name}/healthy_bg_cs_combinations.png")


# =============================================================================
# Complete Evaluation Pipeline Integration
# =============================================================================

def generate_three_metric_type_analysis(
    df_raw: pd.DataFrame,
    thresholds: Dict[str, float],
    output_base_dir: Path,
    exclude_last_minutes: float = 30.0,
    title_suffix: str = "Test Set",
    max_gap_multiplier: Optional[float] = None,
    decision_time_hours: float = 1.0,
) -> Dict:
    """Complete evaluation pipeline using three metric types with separate thresholds.

    Each metric type gets its own threshold and CDR application, so that
    each one can independently achieve the target FPR at the decision time.

    Directory structure:
        output_base_dir/
            three_metric_types/
                instantaneous/
                    sensitivity_vs_time.png  ...
                    subgroups/ ...
                committed_cumulative/
                    sensitivity_vs_time.png  ...
                    subgroups/ ...
                committed_overall/  # PRIMARY
                    sensitivity_vs_time.png  ...
                    subgroups/ ...
                comparison/
                    metric_type_comparison.png
                thresholds.json
                metrics_summary.json

    Args:
        df_raw: Raw predictions DataFrame (pre-CDR) with columns
            ``guid``, ``epoch``, ``binary_target``, ``prob_class_1``, etc.
        thresholds: Per-metric-type thresholds, e.g.
            ``{'instantaneous': 0.4, 'committed_cumulative': 0.35,
              'committed_overall': 0.3}``.
        output_base_dir: Base directory for all outputs.
        exclude_last_minutes: Exclude last N minutes from analysis.
        title_suffix: Suffix for plot titles.
        max_gap_multiplier: For ``fill_missing_epochs``.
        decision_time_hours: Decision time in hours before delivery for
            extracting decision-point metrics.

    Returns:
        Dictionary with thresholds, metrics, decision-point metrics, and
        subgroup analysis for all three types.
    """
    logger.info("=" * 80)
    logger.info("THREE METRIC TYPE ANALYSIS - SEPARATE THRESHOLDS PER METRIC TYPE")
    logger.info("=" * 80)

    analysis_dir = output_base_dir / "three_metric_types"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    metric_type_names = ['instantaneous', 'committed_cumulative', 'committed_overall']
    compute_funcs = {
        'instantaneous': compute_instantaneous_metrics,
        'committed_cumulative': compute_committed_cumulative_metrics,
        'committed_overall': compute_committed_overall_metrics,
    }

    # --- Step 1: For each metric type, apply its own CDR + compute metrics ---
    metrics_dict = {}
    cdr_dfs = {}  # CDR'd DataFrames per metric type (for subgroup analysis)
    decision_point_metrics = {}

    for mt in metric_type_names:
        thresh = thresholds.get(mt)
        if thresh is None:
            logger.warning(f"No threshold for {mt} — skipping")
            continue

        logger.info(f"  [{mt}] Applying CDR with threshold={thresh:.4f}")
        df_clinical = apply_clinical_decision_rule(df_raw.copy(), thresh, verify=False)

        fill_until_birth = mt in ('committed_cumulative', 'committed_overall')
        df_clinical = fill_missing_epochs(
            df_clinical,
            max_gap_multiplier=max_gap_multiplier if not fill_until_birth else None,
            log_summary=False,
            fill_until_birth=fill_until_birth,
        )
        cdr_dfs[mt] = df_clinical

        # Compute time bins from the filled DataFrame
        time_bins = compute_time_bins(df_clinical, exclude_last_minutes=exclude_last_minutes)
        logger.info(f"  [{mt}] Time bins: {len(time_bins)-1} bins")

        # Compute this metric type
        mt_df = compute_funcs[mt](df_clinical, time_bins, subgroup_filter=None)
        metrics_dict[mt] = mt_df

        # Plot
        metric_output_dir = analysis_dir / mt
        plot_single_metric_type(mt_df, mt, metric_output_dir, title_suffix)

        # Extract decision-point metrics
        dp = _extract_decision_point_metrics(mt_df, decision_time_hours)
        if dp:
            dp['threshold'] = float(thresh)
        decision_point_metrics[mt] = dp
        logger.info(
            f"  [{mt}] FPR@{decision_time_hours}h = "
            f"{dp.get('fpr_at_decision', 'N/A')}"
        )

    # --- Step 2: Comparison plot across metric types --------------------------
    comparison_dir = analysis_dir / "comparison"
    plot_metric_type_comparison(metrics_dict, comparison_dir, title_suffix)

    # --- Step 2b: Dataset stats (use committed_overall CDR for stats) ---------
    # Pick any CDR'd df for dataset stats — they all have the same GUIDs
    stats_df = cdr_dfs.get('committed_overall', df_raw)
    # Recompute time bins for the stats df
    stats_time_bins = compute_time_bins(
        ensure_committed_epochs_filled(stats_df),
        exclude_last_minutes=exclude_last_minutes,
    )
    dataset_stats_dir = analysis_dir / "dataset_stats"
    generate_fold_dataset_stats(stats_df, stats_time_bins, dataset_stats_dir, title_suffix)

    # --- Step 3: Subgroup analysis per metric type (each uses its own CDR) ----
    logger.info("Generating subgroup analysis for all three metric types...")
    subgroup_filters = create_enhanced_subgroup_filters()
    all_subgroup_metrics = {}

    for mt in metric_type_names:
        if mt not in cdr_dfs or mt not in metrics_dict:
            continue
        logger.info(f"  - {mt} subgroups...")
        mt_subgroup_dir = analysis_dir / mt / "subgroups"
        mt_time_bins = compute_time_bins(
            cdr_dfs[mt], exclude_last_minutes=exclude_last_minutes
        )
        mt_subgroups = plot_subgroup_analysis(
            cdr_dfs[mt], mt_time_bins, mt,
            subgroup_filters, mt_subgroup_dir, title_suffix,
        )
        all_subgroup_metrics[mt] = mt_subgroups

    # --- Step 4: Dataset statistics -------------------------------------------
    logger.info("Computing dataset statistics for all subgroups...")
    subgroup_statistics = compute_subgroup_statistics(stats_df, subgroup_filters)

    overall_statistics = {
        'total_guids': int(stats_df['guid'].nunique()),
        'total_epochs': int(len(stats_df)),
        'diagnosis_counts': {
            'healthy': int(stats_df.groupby('guid')['target'].first().eq(1).sum()),
            'acidosis': int(stats_df.groupby('guid')['target'].first().eq(2).sum()),
            'hie': int(stats_df.groupby('guid')['target'].first().eq(3).sum()),
        },
        'cs_counts': {
            'cs_positive': int(stats_df.groupby('guid')['cs_label'].first().eq(True).sum()),
            'cs_negative': int(stats_df.groupby('guid')['cs_label'].first().eq(False).sum()),
        },
        'bg_counts': {
            'bg_positive': int(stats_df.groupby('guid')['bg_label'].first().eq(True).sum()),
            'bg_negative': int(stats_df.groupby('guid')['bg_label'].first().eq(False).sum()),
        },
    }

    # --- Step 5: Save summaries -----------------------------------------------
    summary = {
        'thresholds': {mt: float(thresholds[mt]) for mt in metric_type_names if mt in thresholds},
        'metric_types': {
            mt: _summarize_metrics_df(metrics_dict.get(mt))
            for mt in metric_type_names
        },
        'decision_point_metrics': decision_point_metrics,
        'subgroups': {
            mt: {
                name: _summarize_metrics_df(sg_df)
                for name, sg_df in all_subgroup_metrics.get(mt, {}).items()
            }
            for mt in metric_type_names
        },
        'dataset_statistics': {
            'overall': overall_statistics,
            'subgroups': subgroup_statistics,
        },
    }

    with open(analysis_dir / "metrics_summary.json", 'w') as f:
        json.dump(convert_numpy_types(summary), f, indent=2)

    # Save thresholds separately for quick reference
    with open(analysis_dir / "thresholds.json", 'w') as f:
        json.dump(convert_numpy_types({
            'thresholds': {mt: float(thresholds[mt]) for mt in metric_type_names if mt in thresholds},
            'decision_point_metrics': decision_point_metrics,
        }), f, indent=2)

    with open(analysis_dir / "dataset_statistics.json", 'w') as f:
        json.dump(convert_numpy_types({
            'overall': overall_statistics,
            'subgroups': subgroup_statistics,
        }), f, indent=2)

    logger.info(f"Dataset statistics computed for {len(subgroup_statistics)} subgroups")
    logger.info("Three metric type analysis complete")
    logger.info(f"Results saved to: {analysis_dir}")

    return {
        'metrics_dict': metrics_dict,
        'decision_point_metrics': decision_point_metrics,
        'subgroup_metrics': all_subgroup_metrics,
        'summary': summary,
        'dataset_statistics': {
            'overall': overall_statistics,
            'subgroups': subgroup_statistics,
        },
    }


def _summarize_metrics_df(df: pd.DataFrame) -> Dict:
    """Helper to summarize a metrics DataFrame for JSON export."""
    if df is None or len(df) == 0:
        return {}

    valid_df = df[df['sensitivity'].notna()]
    if len(valid_df) == 0:
        return {}

    return {
        'n_bins': int(len(df)),
        'n_valid_bins': int(len(valid_df)),
        'sensitivity_mean': float(valid_df['sensitivity'].mean()),
        'sensitivity_std': float(valid_df['sensitivity'].std()),
        'sensitivity_min': float(valid_df['sensitivity'].min()),
        'sensitivity_max': float(valid_df['sensitivity'].max()),
        'fpr_mean': float(valid_df['fpr'].mean()) if 'fpr' in valid_df else None,
        'fpr_std': float(valid_df['fpr'].std()) if 'fpr' in valid_df else None,
    }


def _extract_decision_point_metrics(
    metrics_df: pd.DataFrame,
    decision_time_hours: float = 1.0,
    tolerance: float = 0.5
) -> Dict:
    """Extract metrics at the bin closest to the decision time point.

    Unlike ``_summarize_metrics_df`` which averages across ALL time bins,
    this function returns metrics at the specific clinical decision point
    (e.g. 1 hour before delivery).

    Args:
        metrics_df: Per-bin metrics DataFrame with columns
            ``bin_center``, ``fpr``, ``sensitivity``, ``specificity``.
        decision_time_hours: Target decision time in hours before delivery.
        tolerance: Maximum allowed distance from target bin centre (hours).

    Returns:
        Dictionary with decision-point metrics, or empty dict if no valid
        bin is found within tolerance.
    """
    if metrics_df is None or len(metrics_df) == 0:
        return {}

    valid_df = metrics_df[metrics_df['sensitivity'].notna()]
    if len(valid_df) == 0:
        return {}

    idx = (valid_df['bin_center'] - decision_time_hours).abs().idxmin()
    row = valid_df.loc[idx]
    actual_time = float(row['bin_center'])

    if abs(actual_time - decision_time_hours) > tolerance:
        logger.warning(
            f"No bin within {tolerance}h of decision time {decision_time_hours}h "
            f"(closest: {actual_time:.2f}h)"
        )
        return {}

    result = {
        'fpr_at_decision': float(row['fpr']) if pd.notna(row.get('fpr')) else None,
        'sensitivity_at_decision': float(row['sensitivity']),
        'specificity_at_decision': float(row['specificity']) if pd.notna(row.get('specificity')) else None,
        'actual_decision_time_hours': actual_time,
    }
    return result


# ============================================================================
# ROC CURVE UTILITIES
# ============================================================================

def compute_guid_level_roc(
    df_raw: pd.DataFrame,
    decision_time_hours: float = 1.0
) -> Dict:
    """Compute GUID-level ROC curve at the clinical decision time point.

    For each GUID, aggregates the maximum predicted probability across all
    epochs at or before the decision time.  This yields one score per GUID,
    enabling a meaningful clinical ROC analysis.

    Args:
        df_raw: Raw predictions DataFrame (pre-CDR) with columns
            ``guid``, ``epoch`` (or ``epoch_hours``), ``prob_class_1``,
            ``binary_target``.
        decision_time_hours: Only consider epochs at or before this time
            (hours before delivery).

    Returns:
        Dictionary with keys ``fpr``, ``tpr``, ``thresholds``, ``auc``,
        ``n_positive``, ``n_negative``.  Arrays are Python lists for JSON
        serialisation.  Returns empty dict on failure.
    """
    df = ensure_epoch_hours(df_raw.copy())

    # Keep epochs at or before the decision time (epoch_hours >= decision_time)
    df_filtered = df[df['epoch_hours'] >= decision_time_hours].copy()
    if len(df_filtered) == 0:
        logger.warning(
            f"No epochs at or before {decision_time_hours}h — cannot compute ROC"
        )
        return {}

    # GUID-level aggregation: max probability, true label
    guid_scores = df_filtered.groupby('guid').agg(
        score=('prob_class_1', 'max'),
        label=('binary_target', 'max'),
    ).reset_index()

    labels = guid_scores['label'].values.astype(int)
    scores = guid_scores['score'].values.astype(float)

    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)

    if n_pos == 0 or n_neg == 0:
        logger.warning("Only one class present — ROC undefined")
        return {}

    fpr_arr, tpr_arr, thresh_arr = roc_curve(labels, scores)
    roc_auc = float(auc(fpr_arr, tpr_arr))

    return {
        'fpr': fpr_arr.tolist(),
        'tpr': tpr_arr.tolist(),
        'thresholds': thresh_arr.tolist(),
        'auc': roc_auc,
        'n_positive': n_pos,
        'n_negative': n_neg,
    }


def compute_committed_cumulative_roc(
    df_raw: pd.DataFrame,
    decision_time_hours: float = 1.0,
    max_gap_multiplier: Optional[float] = None,
    n_thresholds: int = 200,
) -> Dict:
    """Compute ROC curve using committed-cumulative FPR/TPR at decision time.

    For each threshold candidate:
      1. Apply CDR + fill_missing_epochs(fill_until_birth=True).
      2. Among GUIDs with data at the decision time, count detections.
      3. Record FPR and sensitivity (TPR) at the decision time bin.

    This produces a clinically meaningful ROC that accounts for the CDR
    forward-fill logic and the committed-cumulative denominator (GUIDs
    available at the decision time).

    Args:
        df_raw: Raw predictions DataFrame (pre-CDR) with columns
            ``guid``, ``epoch`` (or ``epoch_hours``), ``prob_class_1``,
            ``binary_target``.
        decision_time_hours: Decision time in hours before delivery.
        max_gap_multiplier: Passed to ``fill_missing_epochs``.
        n_thresholds: Maximum number of threshold candidates to evaluate.

    Returns:
        Dictionary with keys ``fpr``, ``tpr``, ``thresholds``, ``auc``,
        ``n_positive``, ``n_negative``.  Returns empty dict on failure.
    """
    probs = df_raw['prob_class_1'].values
    candidates = np.sort(np.unique(probs))
    if len(candidates) > n_thresholds:
        indices = np.linspace(0, len(candidates) - 1, n_thresholds, dtype=int)
        candidates = candidates[indices]

    fprs, tprs = [], []

    for thresh in candidates:
        df_cdr = apply_clinical_decision_rule(df_raw.copy(), thresh, verify=False)
        df_filled = fill_missing_epochs(
            df_cdr, fill_until_birth=True, max_gap_multiplier=max_gap_multiplier,
        )
        df_filled = ensure_epoch_hours(df_filled)

        guid_targets = df_filled.groupby('guid')['binary_target'].max()
        positive_guids = set(guid_targets[guid_targets == 1].index)
        negative_guids = set(guid_targets[guid_targets == 0].index)

        at_decision = df_filled[df_filled['epoch_hours'] >= decision_time_hours]
        available_guids = set(at_decision['guid'].unique())

        n_pos_avail = len(available_guids & positive_guids)
        n_neg_avail = len(available_guids & negative_guids)

        if n_pos_avail == 0 or n_neg_avail == 0:
            continue

        tp, fp = 0, 0
        detected_guids = set(
            at_decision.loc[at_decision['clinical_pred'] == 1, 'guid'].unique()
        )
        tp = len(detected_guids & positive_guids & available_guids)
        fp = len(detected_guids & negative_guids & available_guids)

        tprs.append(tp / n_pos_avail)
        fprs.append(fp / n_neg_avail)

    if len(fprs) < 2:
        logger.warning("Not enough valid threshold points for CC-ROC")
        return {}

    fprs = np.array(fprs)
    tprs = np.array(tprs)

    sorted_idx = np.argsort(fprs)
    fprs = fprs[sorted_idx]
    tprs = tprs[sorted_idx]

    # Add endpoints
    fprs = np.concatenate([[0.0], fprs, [1.0]])
    tprs = np.concatenate([[0.0], tprs, [1.0]])

    # Remove duplicate FPR values (keep max TPR)
    unique_fprs, unique_idx = np.unique(fprs, return_index=True)
    unique_tprs = np.array([tprs[fprs == f].max() for f in unique_fprs])

    roc_auc = float(np.trapz(unique_tprs, unique_fprs))

    n_pos = len(positive_guids)
    n_neg = len(negative_guids)

    return {
        'fpr': unique_fprs.tolist(),
        'tpr': unique_tprs.tolist(),
        'thresholds': candidates.tolist(),
        'auc': roc_auc,
        'n_positive': n_pos,
        'n_negative': n_neg,
    }


def plot_roc_curve(
    roc_data: Dict,
    output_path: Path,
    title_suffix: str = "",
    threshold: Optional[float] = None
) -> None:
    """Plot a single ROC curve and save to disk.

    Args:
        roc_data: Dictionary returned by ``compute_guid_level_roc``.
        output_path: File path for the saved figure.
        title_suffix: Extra text appended to the plot title.
        threshold: If provided, marks the operating point on the curve.
    """
    if not roc_data:
        return

    fpr = np.array(roc_data['fpr'])
    tpr = np.array(roc_data['tpr'])
    roc_auc = roc_data['auc']

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color='#2980b9', linewidth=2,
            label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1)

    if threshold is not None and 'thresholds' in roc_data:
        thresholds = np.array(roc_data['thresholds'])
        idx = np.argmin(np.abs(thresholds - threshold))
        ax.plot(fpr[idx], tpr[idx], 'ro', markersize=10,
                label=f'Operating point (t={threshold:.3f})')

    title = 'GUID-Level ROC Curve'
    if title_suffix:
        title += f' — {title_suffix}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curve saved: {output_path}")


def plot_aggregated_roc_curves(
    all_roc_data: List[Dict],
    output_path: Path,
    n_folds: int
) -> None:
    """Plot aggregated ROC curves across k-folds.

    Overlays per-fold curves (thin lines) and adds a mean +/- std band.

    Args:
        all_roc_data: List of dicts returned by ``compute_guid_level_roc``
            (one per fold).
        output_path: File path for the saved figure.
        n_folds: Total number of folds (for title).
    """
    valid_roc = [r for r in all_roc_data if r and 'fpr' in r]
    if not valid_roc:
        logger.warning("No valid ROC data for aggregated plot")
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    # Common FPR grid for interpolation
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []

    for i, rd in enumerate(valid_roc):
        fpr = np.array(rd['fpr'])
        tpr = np.array(rd['tpr'])
        roc_auc = rd['auc']
        aucs.append(roc_auc)

        ax.plot(fpr, tpr, alpha=0.25, linewidth=1,
                label=f'Fold {i} (AUC={roc_auc:.3f})')

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))

    ax.plot(mean_fpr, mean_tpr, color='#2980b9', linewidth=2.5,
            label=f'Mean ROC (AUC={mean_auc:.3f} ± {std_auc:.3f})')
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                     color='#2980b9', alpha=0.15)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1)

    ax.set_title(f'Aggregated GUID-Level ROC ({n_folds}-Fold)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Aggregated ROC plot saved: {output_path}")


def _aggregate_fold_results(all_fold_results: List[Dict]) -> Dict:
    """
    Aggregate results across all folds.

    Args:
        all_fold_results: List of fold result dictionaries

    Returns:
        Aggregated results dictionary
    """
    n = len(all_fold_results)

    if n == 0:
        return {}

    # Aggregate PRIMARY threshold
    thresholds = [r['primary_threshold'] for r in all_fold_results]
    threshold_mean = float(np.mean(thresholds))
    threshold_std = float(np.std(thresholds))

    # Aggregate validation metrics
    val_sens = [r['validation_sensitivity'] for r in all_fold_results]
    val_spec = [r['validation_specificity'] for r in all_fold_results]
    val_fpr = [r['validation_fpr'] for r in all_fold_results]
    val_acc = [r['validation_accuracy'] for r in all_fold_results]

    # Aggregate test metrics (PRIMARY - committed_overall)
    test_sens = [r['test_sensitivity_mean'] for r in all_fold_results]
    test_spec = [r['test_specificity_mean'] for r in all_fold_results]
    test_fpr = [r['test_fpr_mean'] for r in all_fold_results]

    # Aggregate three metric type analysis across folds
    three_metric_aggregated = _aggregate_three_metric_analysis(all_fold_results)

    # Aggregate dataset statistics across folds
    dataset_stats_aggregated = _aggregate_dataset_statistics(all_fold_results)

    aggregated = {
        'timestamp': datetime.now().isoformat(),
        'n_folds': n,

        # PRIMARY threshold statistics
        'primary_threshold_mean': threshold_mean,
        'primary_threshold_std': threshold_std,
        'primary_threshold_min': float(np.min(thresholds)),
        'primary_threshold_max': float(np.max(thresholds)),

        # Validation metrics (aggregated across folds)
        'validation_sensitivity_mean': float(np.mean(val_sens)),
        'validation_sensitivity_std': float(np.std(val_sens)),
        'validation_specificity_mean': float(np.mean(val_spec)),
        'validation_specificity_std': float(np.std(val_spec)),
        'validation_fpr_mean': float(np.mean(val_fpr)),
        'validation_fpr_std': float(np.std(val_fpr)),
        'validation_accuracy_mean': float(np.mean(val_acc)),
        'validation_accuracy_std': float(np.std(val_acc)),

        # Test metrics (PRIMARY - committed_overall aggregated across folds)
        'test_sensitivity_mean': float(np.mean(test_sens)),
        'test_sensitivity_std': float(np.std(test_sens)),
        'test_sensitivity_min': float(np.min(test_sens)),
        'test_sensitivity_max': float(np.max(test_sens)),
        'test_specificity_mean': float(np.mean(test_spec)),
        'test_specificity_std': float(np.std(test_spec)),
        'test_specificity_min': float(np.min(test_spec)),
        'test_specificity_max': float(np.max(test_spec)),
        'test_fpr_mean': float(np.mean(test_fpr)),
        'test_fpr_std': float(np.std(test_fpr)),
        'test_fpr_min': float(np.min(test_fpr)),
        'test_fpr_max': float(np.max(test_fpr)),

        # Three metric type analysis (all metrics and subgroups aggregated)
        'three_metric_analysis_aggregated': three_metric_aggregated,

        # Dataset statistics (aggregated across folds)
        'dataset_statistics_aggregated': dataset_stats_aggregated,

        # Individual fold results (includes per-fold three metric analysis)
        'fold_results': all_fold_results,
    }

    return aggregated


def _aggregate_three_metric_analysis(all_fold_results: List[Dict]) -> Dict:
    """
    Aggregate three metric type analysis (including subgroups) across folds.

    Args:
        all_fold_results: List of fold result dictionaries

    Returns:
        Aggregated three metric analysis with mean/std for all metric types and subgroups
    """
    aggregated = {
        'metric_types': {},
        'subgroups': {}
    }

    # Extract three_metric_analysis from each fold
    fold_analyses = [
        r.get('three_metric_analysis', {})
        for r in all_fold_results
        if r.get('three_metric_analysis')
    ]

    if not fold_analyses:
        return aggregated

    # Aggregate main metric types (instantaneous, committed_cumulative, committed_overall)
    for metric_type in ['instantaneous', 'committed_cumulative', 'committed_overall']:
        metric_values = {}
        for fold in fold_analyses:
            fold_metrics = fold.get('metric_types', {}).get(metric_type, {})
            for key, value in fold_metrics.items():
                if isinstance(value, (int, float)):
                    if key not in metric_values:
                        metric_values[key] = []
                    metric_values[key].append(value)

        # Compute mean/std for each metric
        aggregated['metric_types'][metric_type] = {
            f'{key}_mean': float(np.mean(values))
            for key, values in metric_values.items()
        }
        aggregated['metric_types'][metric_type].update({
            f'{key}_std': float(np.std(values))
            for key, values in metric_values.items()
        })

    # Aggregate subgroups for each metric type
    for metric_type in ['instantaneous', 'committed_cumulative', 'committed_overall']:
        aggregated['subgroups'][metric_type] = {}

        # Get all subgroup names across folds
        all_subgroup_names = set()
        for fold in fold_analyses:
            subgroups = fold.get('subgroups', {}).get(metric_type, {})
            all_subgroup_names.update(subgroups.keys())

        # Aggregate each subgroup
        for subgroup_name in all_subgroup_names:
            subgroup_values = {}
            for fold in fold_analyses:
                fold_subgroup = fold.get('subgroups', {}).get(metric_type, {}).get(subgroup_name, {})
                for key, value in fold_subgroup.items():
                    if isinstance(value, (int, float)):
                        if key not in subgroup_values:
                            subgroup_values[key] = []
                        subgroup_values[key].append(value)

            # Compute mean/std
            if subgroup_values:
                aggregated['subgroups'][metric_type][subgroup_name] = {
                    f'{key}_mean': float(np.mean(values))
                    for key, values in subgroup_values.items()
                }
                aggregated['subgroups'][metric_type][subgroup_name].update({
                    f'{key}_std': float(np.std(values))
                    for key, values in subgroup_values.items()
                })

    return aggregated


def _aggregate_dataset_statistics(all_fold_results: List[Dict]) -> Dict:
    """
    Aggregate dataset statistics across folds.

    Args:
        all_fold_results: List of fold result dictionaries

    Returns:
        Dictionary with aggregated statistics (mean, std, min, max) for:
        - Overall statistics (total GUIDs, diagnosis counts, CS counts, BG counts)
        - Subgroup statistics (counts for each subgroup)
    """
    # Extract dataset statistics from each fold
    fold_stats = []
    for r in all_fold_results:
        three_metric_full = r.get('three_metric_results_full', {})
        if three_metric_full and 'dataset_statistics' in three_metric_full:
            fold_stats.append(three_metric_full['dataset_statistics'])

    if not fold_stats:
        return {}

    # Aggregate overall statistics
    overall_stats = {
        'total_guids': _aggregate_stat([s['overall']['total_guids'] for s in fold_stats]),
        'total_epochs': _aggregate_stat([s['overall']['total_epochs'] for s in fold_stats]),
        'diagnosis_counts': {
            'healthy': _aggregate_stat([s['overall']['diagnosis_counts']['healthy'] for s in fold_stats]),
            'acidosis': _aggregate_stat([s['overall']['diagnosis_counts']['acidosis'] for s in fold_stats]),
            'hie': _aggregate_stat([s['overall']['diagnosis_counts']['hie'] for s in fold_stats])
        },
        'cs_counts': {
            'cs_positive': _aggregate_stat([s['overall']['cs_counts']['cs_positive'] for s in fold_stats]),
            'cs_negative': _aggregate_stat([s['overall']['cs_counts']['cs_negative'] for s in fold_stats])
        },
        'bg_counts': {
            'bg_positive': _aggregate_stat([s['overall']['bg_counts']['bg_positive'] for s in fold_stats]),
            'bg_negative': _aggregate_stat([s['overall']['bg_counts']['bg_negative'] for s in fold_stats])
        }
    }

    # Aggregate subgroup statistics
    all_subgroup_names = set()
    for s in fold_stats:
        all_subgroup_names.update(s['subgroups'].keys())

    subgroup_stats = {}
    for subgroup_name in all_subgroup_names:
        try:
            n_guids_list = [s['subgroups'][subgroup_name]['n_guids'] for s in fold_stats if subgroup_name in s['subgroups']]
            n_epochs_list = [s['subgroups'][subgroup_name]['n_epochs'] for s in fold_stats if subgroup_name in s['subgroups']]
            n_positive_list = [s['subgroups'][subgroup_name]['n_positive'] for s in fold_stats if subgroup_name in s['subgroups']]
            n_negative_list = [s['subgroups'][subgroup_name]['n_negative'] for s in fold_stats if subgroup_name in s['subgroups']]

            healthy_list = [s['subgroups'][subgroup_name]['diagnosis_breakdown']['healthy'] for s in fold_stats if subgroup_name in s['subgroups']]
            acidosis_list = [s['subgroups'][subgroup_name]['diagnosis_breakdown']['acidosis'] for s in fold_stats if subgroup_name in s['subgroups']]
            hie_list = [s['subgroups'][subgroup_name]['diagnosis_breakdown']['hie'] for s in fold_stats if subgroup_name in s['subgroups']]

            subgroup_stats[subgroup_name] = {
                'n_guids': _aggregate_stat(n_guids_list),
                'n_epochs': _aggregate_stat(n_epochs_list),
                'n_positive': _aggregate_stat(n_positive_list),
                'n_negative': _aggregate_stat(n_negative_list),
                'diagnosis_breakdown': {
                    'healthy': _aggregate_stat(healthy_list),
                    'acidosis': _aggregate_stat(acidosis_list),
                    'hie': _aggregate_stat(hie_list)
                }
            }
        except Exception as e:
            logger.warning(f"Failed to aggregate statistics for {subgroup_name}: {e}")
            continue

    return {
        'overall': overall_stats,
        'subgroups': subgroup_stats
    }


def _aggregate_stat(values: List[float]) -> Dict[str, float]:
    """Helper to compute mean, std, min, max for a list of values."""
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values))
    }


def _aggregate_dataframes_across_folds(
    fold_dfs: List[pd.DataFrame],
    metrics: List[str] = ['sensitivity', 'specificity', 'fpr']
) -> pd.DataFrame:
    """
    Aggregate DataFrames across folds to compute mean, min, max for each time bin.

    Args:
        fold_dfs: List of DataFrames from different folds (same structure)
        metrics: List of metric column names to aggregate

    Returns:
        Aggregated DataFrame with columns:
            - bin_center
            - {metric}_mean, {metric}_min, {metric}_max for each metric
            - n_folds (number of folds contributing to each bin)
    """
    if not fold_dfs:
        return pd.DataFrame()

    # Filter out None and empty DataFrames
    valid_dfs = [df for df in fold_dfs if df is not None and len(df) > 0]
    if not valid_dfs:
        return pd.DataFrame()

    # Get all unique bin_centers across folds
    all_bins = set()
    for df in valid_dfs:
        if 'bin_center' in df.columns:
            all_bins.update(df['bin_center'].unique())

    if not all_bins:
        return pd.DataFrame()

    all_bins = sorted(all_bins)

    # Aggregate metrics for each bin
    aggregated_data = []
    for bin_center in all_bins:
        bin_data = {'bin_center': bin_center}

        for metric in metrics:
            values = []
            for df in valid_dfs:
                if metric in df.columns:
                    bin_rows = df[df['bin_center'] == bin_center]
                    if len(bin_rows) > 0:
                        value = bin_rows[metric].iloc[0]
                        if pd.notna(value):
                            values.append(value)

            if values:
                bin_data[f'{metric}_mean'] = np.mean(values)
                bin_data[f'{metric}_min'] = np.min(values)
                bin_data[f'{metric}_max'] = np.max(values)
                bin_data[f'{metric}_std'] = np.std(values)
                bin_data[f'{metric}_n_folds'] = len(values)
            else:
                bin_data[f'{metric}_mean'] = np.nan
                bin_data[f'{metric}_min'] = np.nan
                bin_data[f'{metric}_max'] = np.nan
                bin_data[f'{metric}_std'] = np.nan
                bin_data[f'{metric}_n_folds'] = 0

        aggregated_data.append(bin_data)

    return pd.DataFrame(aggregated_data)


def plot_aggregated_metric_type(
    aggregated_df: pd.DataFrame,
    metric_type: str,
    output_dir: Path,
    n_folds: int
):
    """
    Plot aggregated metrics for a single metric type with min/max error bands.

    Generates 5 plots:
    1. Sensitivity vs time with min/max band
    2. Sensitivity + Specificity vs time with bands
    3. Sensitivity + FPR vs time with bands
    4. All metrics vs time with bands
    5. FPR only vs time with band (emphasized metric)

    Args:
        aggregated_df: Aggregated DataFrame with mean, min, max columns
        metric_type: 'instantaneous', 'committed_cumulative', or 'committed_overall'
        output_dir: Directory to save plots
        n_folds: Number of folds used in aggregation
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(aggregated_df) == 0:
        logger.warning(f"No data to plot for aggregated {metric_type}")
        return

    # Filter to non-NaN bins
    valid_df = aggregated_df[aggregated_df['sensitivity_mean'].notna()].copy()
    if len(valid_df) == 0:
        logger.warning(f"No valid data to plot for aggregated {metric_type}")
        return

    # Sort by bin_center descending and extract x as NumPy array (matches per-fold plots)
    valid_df = valid_df.sort_values('bin_center', ascending=False).reset_index(drop=True)
    x = valid_df['bin_center'].values

    metric_type_title = {
        'instantaneous': 'Instantaneous',
        'committed_cumulative': 'Committed (Cumulative)',
        'committed_overall': 'Committed (Overall - PRIMARY)'
    }.get(metric_type, metric_type)

    # Plot 1: Sensitivity vs Time
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, valid_df['sensitivity_mean'].values,
            'b-o', linewidth=2.5, markersize=6, label='Mean Sensitivity', markerfacecolor='blue', markeredgecolor='darkblue')
    ax.fill_between(x,
                     valid_df['sensitivity_min'].values,
                     valid_df['sensitivity_max'].values,
                     alpha=0.3, color='blue', label='Min-Max Range')
    ax.set_xlabel('Hours Before Birth', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sensitivity', fontsize=14, fontweight='bold')
    ax.set_title(f'{metric_type_title} - Sensitivity vs Time\n(Aggregated across {n_folds} folds)',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'sensitivity_vs_time_aggregated.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Sensitivity + Specificity
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, valid_df['sensitivity_mean'].values,
            'b-o', linewidth=2.5, markersize=6, label='Mean Sensitivity', markerfacecolor='blue', markeredgecolor='darkblue')
    ax.fill_between(x,
                     valid_df['sensitivity_min'].values,
                     valid_df['sensitivity_max'].values,
                     alpha=0.3, color='blue')

    if 'specificity_mean' in valid_df.columns:
        ax.plot(x, valid_df['specificity_mean'].values,
                'g-s', linewidth=2.5, markersize=6, label='Mean Specificity', markerfacecolor='green', markeredgecolor='darkgreen')
        ax.fill_between(x,
                         valid_df['specificity_min'].values,
                         valid_df['specificity_max'].values,
                         alpha=0.3, color='green')

    ax.set_xlabel('Hours Before Birth', fontsize=14, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=14, fontweight='bold')
    ax.set_title(f'{metric_type_title} - Sensitivity & Specificity\n(Aggregated across {n_folds} folds)',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'sensitivity_specificity_vs_time_aggregated.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Sensitivity + FPR
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, valid_df['sensitivity_mean'].values,
            'b-o', linewidth=2.5, markersize=6, label='Mean Sensitivity', markerfacecolor='blue', markeredgecolor='darkblue')
    ax.fill_between(x,
                     valid_df['sensitivity_min'].values,
                     valid_df['sensitivity_max'].values,
                     alpha=0.3, color='blue')

    if 'fpr_mean' in valid_df.columns:
        ax.plot(x, valid_df['fpr_mean'].values,
                'r-^', linewidth=2.5, markersize=6, label='Mean FPR', markerfacecolor='red', markeredgecolor='darkred')
        ax.fill_between(x,
                         valid_df['fpr_min'].values,
                         valid_df['fpr_max'].values,
                         alpha=0.3, color='red')

    ax.set_xlabel('Hours Before Birth', fontsize=14, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=14, fontweight='bold')
    ax.set_title(f'{metric_type_title} - Sensitivity & FPR\n(Aggregated across {n_folds} folds)',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'sensitivity_fpr_vs_time_aggregated.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: All metrics
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, valid_df['sensitivity_mean'].values,
            'b-o', linewidth=2.5, markersize=6, label='Mean Sensitivity', markerfacecolor='blue', markeredgecolor='darkblue')
    ax.fill_between(x,
                     valid_df['sensitivity_min'].values,
                     valid_df['sensitivity_max'].values,
                     alpha=0.2, color='blue')

    if 'specificity_mean' in valid_df.columns:
        ax.plot(x, valid_df['specificity_mean'].values,
                'g-s', linewidth=2.5, markersize=6, label='Mean Specificity', markerfacecolor='green', markeredgecolor='darkgreen')
        ax.fill_between(x,
                         valid_df['specificity_min'].values,
                         valid_df['specificity_max'].values,
                         alpha=0.2, color='green')

    if 'fpr_mean' in valid_df.columns:
        ax.plot(x, valid_df['fpr_mean'].values,
                'r-^', linewidth=2.5, markersize=6, label='Mean FPR', markerfacecolor='red', markeredgecolor='darkred')
        ax.fill_between(x,
                         valid_df['fpr_min'].values,
                         valid_df['fpr_max'].values,
                         alpha=0.2, color='red')

    ax.set_xlabel('Hours Before Birth', fontsize=14, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=14, fontweight='bold')
    ax.set_title(f'{metric_type_title} - All Metrics\n(Aggregated across {n_folds} folds)',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'all_metrics_vs_time_aggregated.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 5: FPR Only (emphasized metric)
    if 'fpr_mean' in valid_df.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(x, valid_df['fpr_mean'].values,
                'r-^', linewidth=2.5, markersize=6, label='Mean FPR',
                markerfacecolor='red', markeredgecolor='darkred')
        ax.fill_between(x,
                        valid_df['fpr_min'].values,
                        valid_df['fpr_max'].values,
                        alpha=0.3, color='red', label='Min-Max Range')
        ax.set_xlabel('Hours Before Birth', fontsize=14, fontweight='bold')
        ax.set_ylabel('FPR', fontsize=14, fontweight='bold')
        ax.set_title(f'{metric_type_title} - FPR vs Time\n(Aggregated across {n_folds} folds)',
                     fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.invert_xaxis()
        plt.tight_layout()
        plt.savefig(output_dir / 'fpr_vs_time_aggregated.png', dpi=300, bbox_inches='tight')
        plt.close()

    logger.info(f"  Saved 5 aggregated plots for {metric_type} to {output_dir}")


def plot_aggregated_subgroup_comparison(
    subgroup_dfs: Dict[str, pd.DataFrame],
    subgroup_names: List[str],
    output_path: Path,
    title: str,
    n_folds: int,
    metric: str = 'sensitivity'
):
    """
    Plot aggregated subgroup comparison with min/max bands.

    Args:
        subgroup_dfs: Dict mapping subgroup name to aggregated DataFrame
        subgroup_names: List of subgroup names to plot
        output_path: Path to save plot
        title: Plot title
        n_folds: Number of folds
        metric: Metric to plot ('sensitivity' or 'specificity')
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(subgroup_names)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']  # Different markers for variety

    for i, name in enumerate(subgroup_names):
        df = subgroup_dfs.get(name)
        if df is None or len(df) == 0:
            continue

        mean_col = f'{metric}_mean'
        min_col = f'{metric}_min'
        max_col = f'{metric}_max'

        if mean_col not in df.columns:
            continue

        valid_df = df[df[mean_col].notna()].sort_values('bin_center', ascending=False).reset_index(drop=True)
        if len(valid_df) == 0:
            continue

        sg_x = valid_df['bin_center'].values
        marker = markers[i % len(markers)]
        ax.plot(sg_x, valid_df[mean_col].values,
                marker=marker, linewidth=2.5, markersize=6, label=name, color=colors[i],
                markerfacecolor=colors[i], markeredgecolor='black', markeredgewidth=0.5)
        ax.fill_between(sg_x,
                         valid_df[min_col].values,
                         valid_df[max_col].values,
                         alpha=0.2, color=colors[i])

    ax.set_xlabel('Hours Before Birth', fontsize=14, fontweight='bold')
    metric_label = metric.capitalize()
    ax.set_ylabel(metric_label, fontsize=14, fontweight='bold')
    ax.set_title(f'{title}\n(Aggregated across {n_folds} folds)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved aggregated subgroup plot: {output_path.name}")


def generate_aggregated_plots(
    all_fold_results: List[Dict],
    output_base_dir: Path,
    n_folds: int,
    data_source: str = "test",
):
    """
    Generate aggregated plots across all folds for all three metric types and subgroups.

    Args:
        all_fold_results: List of fold result dictionaries
        output_base_dir: Base output directory
        n_folds: Total number of folds
        data_source: Source of data to aggregate ("test" or "validation").
            Controls which key is read from fold results and the output
            directory name.
    """
    logger.info("="*80)
    logger.info("GENERATING AGGREGATED PLOTS ACROSS FOLDS ({})", data_source)
    logger.info("="*80)

    dir_name = "validation_aggregated_plots" if data_source == "validation" else "aggregated_plots"
    aggregated_dir = output_base_dir / dir_name
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    # Extract three_metric_results_full from each fold and reconstruct DataFrames
    data_key = "val_three_metric_results_full" if data_source == "validation" else "three_metric_results_full"
    fold_analyses = []
    for r in all_fold_results:
        three_metric_full = r.get(data_key, {})
        if not three_metric_full:
            continue

        # Reconstruct DataFrames from dicts
        reconstructed = {}

        # Reconstruct metrics_dict
        if 'metrics_dict' in three_metric_full:
            reconstructed['metrics_dict'] = {
                k: pd.DataFrame(v) if v is not None else None
                for k, v in three_metric_full['metrics_dict'].items()
            }

        # Reconstruct subgroup_metrics
        if 'subgroup_metrics' in three_metric_full:
            reconstructed['subgroup_metrics'] = {
                metric_type: {
                    subgroup: pd.DataFrame(df_dict) if df_dict is not None else None
                    for subgroup, df_dict in subgroups.items()
                }
                for metric_type, subgroups in three_metric_full['subgroup_metrics'].items()
            }

        if reconstructed:
            fold_analyses.append(reconstructed)

    if not fold_analyses:
        logger.warning("No {} data found in fold results. "
                      "Please re-run evaluation to generate plots.", data_key)
        return

    # Aggregate and plot each metric type
    for metric_type in ['instantaneous', 'committed_cumulative', 'committed_overall']:
        logger.info(f"Aggregating and plotting {metric_type}...")

        # Extract DataFrames for this metric type across folds
        fold_dfs = []
        for fold_analysis in fold_analyses:
            metrics_dict = fold_analysis.get('metrics_dict', {})
            df = metrics_dict.get(metric_type)
            if df is not None:
                fold_dfs.append(df)

        if fold_dfs:
            # Aggregate DataFrames
            aggregated_df = _aggregate_dataframes_across_folds(fold_dfs)

            # Plot aggregated metrics
            metric_output_dir = aggregated_dir / metric_type
            plot_aggregated_metric_type(aggregated_df, metric_type, metric_output_dir, n_folds)
        else:
            logger.warning(f"  No data found for {metric_type}")

    # Aggregate and plot subgroups for each metric type
    logger.info("Aggregating and plotting subgroups...")

    for metric_type in ['instantaneous', 'committed_cumulative', 'committed_overall']:
        logger.info(f"  Processing {metric_type} subgroups...")

        # Get all subgroup names across folds
        all_subgroup_names = set()
        for fold_analysis in fold_analyses:
            subgroup_metrics = fold_analysis.get('subgroup_metrics', {})
            metric_subgroups = subgroup_metrics.get(metric_type, {})
            all_subgroup_names.update(metric_subgroups.keys())

        if not all_subgroup_names:
            logger.warning(f"    No subgroups found for {metric_type}")
            continue

        # Aggregate each subgroup
        aggregated_subgroups = {}
        for subgroup_name in all_subgroup_names:
            fold_subgroup_dfs = []
            for fold_analysis in fold_analyses:
                subgroup_metrics = fold_analysis.get('subgroup_metrics', {})
                metric_subgroups = subgroup_metrics.get(metric_type, {})
                subgroup_df = metric_subgroups.get(subgroup_name)
                if subgroup_df is not None:
                    fold_subgroup_dfs.append(subgroup_df)

            if fold_subgroup_dfs:
                aggregated_df = _aggregate_dataframes_across_folds(fold_subgroup_dfs)
                aggregated_subgroups[subgroup_name] = aggregated_df

        # Create subgroup comparison plots
        subgroup_output_dir = aggregated_dir / metric_type / "subgroups"
        subgroup_output_dir.mkdir(parents=True, exist_ok=True)

        # Diagnosis comparison
        diagnosis_subgroups = ['healthy', 'acidosis', 'hie', 'unhealthy']
        available_diagnosis = [s for s in diagnosis_subgroups if s in aggregated_subgroups]
        if available_diagnosis:
            plot_aggregated_subgroup_comparison(
                aggregated_subgroups, available_diagnosis,
                subgroup_output_dir / 'diagnosis_comparison_aggregated.png',
                f'{metric_type.replace("_", " ").title()} - Diagnosis Comparison',
                n_folds
            )
            plot_aggregated_subgroup_comparison(
                aggregated_subgroups, available_diagnosis,
                subgroup_output_dir / 'diagnosis_comparison_fpr_aggregated.png',
                f'{metric_type.replace("_", " ").title()} - Diagnosis Comparison (FPR)',
                n_folds,
                metric='fpr'
            )

        # Unhealthy CS stratification
        unhealthy_cs = ['unhealthy_cs_pos', 'unhealthy_cs_neg']
        available_unhealthy_cs = [s for s in unhealthy_cs if s in aggregated_subgroups]
        if available_unhealthy_cs:
            plot_aggregated_subgroup_comparison(
                aggregated_subgroups, available_unhealthy_cs,
                subgroup_output_dir / 'unhealthy_cs_stratification_aggregated.png',
                f'{metric_type.replace("_", " ").title()} - Unhealthy: CS Stratification',
                n_folds
            )
            plot_aggregated_subgroup_comparison(
                aggregated_subgroups, available_unhealthy_cs,
                subgroup_output_dir / 'unhealthy_cs_stratification_fpr_aggregated.png',
                f'{metric_type.replace("_", " ").title()} - Unhealthy: CS Stratification (FPR)',
                n_folds,
                metric='fpr'
            )

        # HIE CS stratification
        hie_cs = ['hie_cs_pos', 'hie_cs_neg']
        available_hie_cs = [s for s in hie_cs if s in aggregated_subgroups]
        if available_hie_cs:
            plot_aggregated_subgroup_comparison(
                aggregated_subgroups, available_hie_cs,
                subgroup_output_dir / 'hie_cs_stratification_aggregated.png',
                f'{metric_type.replace("_", " ").title()} - HIE: CS Stratification',
                n_folds
            )
            plot_aggregated_subgroup_comparison(
                aggregated_subgroups, available_hie_cs,
                subgroup_output_dir / 'hie_cs_stratification_fpr_aggregated.png',
                f'{metric_type.replace("_", " ").title()} - HIE: CS Stratification (FPR)',
                n_folds,
                metric='fpr'
            )

        # Acidosis CS stratification
        acidosis_cs = ['acidosis_cs_pos', 'acidosis_cs_neg']
        available_acidosis_cs = [s for s in acidosis_cs if s in aggregated_subgroups]
        if available_acidosis_cs:
            plot_aggregated_subgroup_comparison(
                aggregated_subgroups, available_acidosis_cs,
                subgroup_output_dir / 'acidosis_cs_stratification_aggregated.png',
                f'{metric_type.replace("_", " ").title()} - Acidosis: CS Stratification',
                n_folds
            )
            plot_aggregated_subgroup_comparison(
                aggregated_subgroups, available_acidosis_cs,
                subgroup_output_dir / 'acidosis_cs_stratification_fpr_aggregated.png',
                f'{metric_type.replace("_", " ").title()} - Acidosis: CS Stratification (FPR)',
                n_folds,
                metric='fpr'
            )

        # Acidosis BG stratification
        acidosis_bg = ['acidosis_bg_pos', 'acidosis_bg_neg']
        available_acidosis_bg = [s for s in acidosis_bg if s in aggregated_subgroups]
        if available_acidosis_bg:
            plot_aggregated_subgroup_comparison(
                aggregated_subgroups, available_acidosis_bg,
                subgroup_output_dir / 'acidosis_bg_stratification_aggregated.png',
                f'{metric_type.replace("_", " ").title()} - Acidosis: BG Stratification',
                n_folds
            )
            plot_aggregated_subgroup_comparison(
                aggregated_subgroups, available_acidosis_bg,
                subgroup_output_dir / 'acidosis_bg_stratification_fpr_aggregated.png',
                f'{metric_type.replace("_", " ").title()} - Acidosis: BG Stratification (FPR)',
                n_folds,
                metric='fpr'
            )

        # Healthy CS stratification (use specificity instead of sensitivity)
        healthy_cs = ['healthy_cs_pos', 'healthy_cs_neg']
        available_healthy_cs = [s for s in healthy_cs if s in aggregated_subgroups]
        if available_healthy_cs:
            plot_aggregated_subgroup_comparison(
                aggregated_subgroups, available_healthy_cs,
                subgroup_output_dir / 'healthy_cs_stratification_aggregated.png',
                f'{metric_type.replace("_", " ").title()} - Healthy: CS Stratification',
                n_folds,
                metric='specificity'
            )
            plot_aggregated_subgroup_comparison(
                aggregated_subgroups, available_healthy_cs,
                subgroup_output_dir / 'healthy_cs_stratification_fpr_aggregated.png',
                f'{metric_type.replace("_", " ").title()} - Healthy: CS Stratification (FPR)',
                n_folds,
                metric='fpr'
            )

        # Healthy BG stratification (use specificity)
        healthy_bg = ['healthy_bg_pos', 'healthy_bg_neg']
        available_healthy_bg = [s for s in healthy_bg if s in aggregated_subgroups]
        if available_healthy_bg:
            plot_aggregated_subgroup_comparison(
                aggregated_subgroups, available_healthy_bg,
                subgroup_output_dir / 'healthy_bg_stratification_aggregated.png',
                f'{metric_type.replace("_", " ").title()} - Healthy: BG Stratification',
                n_folds,
                metric='specificity'
            )
            plot_aggregated_subgroup_comparison(
                aggregated_subgroups, available_healthy_bg,
                subgroup_output_dir / 'healthy_bg_stratification_fpr_aggregated.png',
                f'{metric_type.replace("_", " ").title()} - Healthy: BG Stratification (FPR)',
                n_folds,
                metric='fpr'
            )

        # Healthy BG×CS combinations (use specificity)
        healthy_bg_cs = ['healthy_bg_pos_cs_pos', 'healthy_bg_pos_cs_neg',
                         'healthy_bg_neg_cs_pos', 'healthy_bg_neg_cs_neg']
        available_healthy_bg_cs = [s for s in healthy_bg_cs if s in aggregated_subgroups]
        if available_healthy_bg_cs:
            plot_aggregated_subgroup_comparison(
                aggregated_subgroups, available_healthy_bg_cs,
                subgroup_output_dir / 'healthy_bg_cs_combinations_aggregated.png',
                f'{metric_type.replace("_", " ").title()} - Healthy: BG×CS Combinations',
                n_folds,
                metric='specificity'
            )
            plot_aggregated_subgroup_comparison(
                aggregated_subgroups, available_healthy_bg_cs,
                subgroup_output_dir / 'healthy_bg_cs_combinations_fpr_aggregated.png',
                f'{metric_type.replace("_", " ").title()} - Healthy: BG×CS Combinations (FPR)',
                n_folds,
                metric='fpr'
            )

    logger.info(f"Aggregated plots saved to: {aggregated_dir}")
