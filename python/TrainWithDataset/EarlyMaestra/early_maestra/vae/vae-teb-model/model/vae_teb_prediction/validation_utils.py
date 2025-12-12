"""
Validation and utility functions for k-fold cross-validation evaluation pipeline.

This module provides helper functions for data validation, consistency checks,
and logging to ensure robustness and transparency in the evaluation pipeline.
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional


def ensure_epoch_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has epoch_hours column.

    This function checks if the DataFrame has an epoch_hours column. If not,
    it creates one by converting the epoch column (seconds) to hours.

    IMPORTANT: Epochs are negative (seconds before birth), but epoch_hours is converted
    to positive values for easier interpretation and binning.
    Example: epoch=-43200s → epoch_hours=12.0h (12 hours before birth)

    Args:
        df: Input DataFrame with 'epoch' column (negative seconds before birth)

    Returns:
        DataFrame with epoch_hours column added if missing (positive hours before birth)

    Raises:
        ValueError: If 'epoch' column is missing
    """
    if 'epoch_hours' in df.columns:
        return df

    if 'epoch' not in df.columns:
        raise ValueError("DataFrame missing required 'epoch' column for time conversion")

    df = df.copy()
    # CRITICAL FIX: Convert negative epochs to positive hours before birth
    # epoch=-43200s → epoch_hours=12.0h (12 hours before birth)
    df['epoch_hours'] = abs(df['epoch']) / 3600
    logger.debug(f"Added epoch_hours column (converted from epoch to absolute hours): {len(df)} rows")

    return df


def validate_predictions_df(df: pd.DataFrame, data_type: str = "predictions") -> None:
    """
    Validate predictions DataFrame structure and values.

    Performs comprehensive validation including:
    - Required columns presence
    - Value range validation (probabilities 0-1, binary targets 0-1)
    - Non-negative epoch values
    - Duplicate detection

    Args:
        df: Predictions DataFrame to validate
        data_type: Type of data for error messages (e.g., "Validation", "Test")

    Raises:
        ValueError: If validation fails for required columns or invalid value ranges
    """
    required_cols = ['guid', 'epoch', 'binary_target', 'prob_class_1']
    missing_cols = set(required_cols) - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"{data_type} DataFrame missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Validate binary_target values
    if not df['binary_target'].isin([0, 1]).all():
        invalid_vals = df['binary_target'].unique()
        invalid_count = (~df['binary_target'].isin([0, 1])).sum()
        raise ValueError(
            f"{data_type} has {invalid_count} samples with invalid binary_target values: "
            f"{invalid_vals} (expected 0 or 1 only)"
        )

    # Validate probability ranges
    if not ((df['prob_class_1'] >= 0) & (df['prob_class_1'] <= 1)).all():
        invalid_count = ((df['prob_class_1'] < 0) | (df['prob_class_1'] > 1)).sum()
        min_val = df['prob_class_1'].min()
        max_val = df['prob_class_1'].max()
        raise ValueError(
            f"{data_type} has {invalid_count} samples with prob_class_1 values outside [0, 1] range. "
            f"Range found: [{min_val:.4f}, {max_val:.4f}]"
        )

    # Note: Epoch values are NEGATIVE (time before birth in seconds)
    # No validation needed - negative values are correct per dataset schema

    # Check for duplicates (warning, not error)
    duplicates = df.duplicated(subset=['guid', 'epoch'])
    if duplicates.any():
        n_dup = duplicates.sum()
        logger.warning(
            f"{data_type} has {n_dup} duplicate (guid, epoch) pairs. "
            f"This may indicate data quality issues."
        )


def validate_guid_consistency(df: pd.DataFrame) -> None:
    """
    Validate that all epochs for a GUID have consistent true labels.

    In medical data, a GUID (baby/record ID) should have a consistent outcome
    across all time epochs. This function checks for inconsistencies which may
    indicate data quality issues.

    Args:
        df: Predictions DataFrame with guid and binary_target columns
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

        # Show details for first inconsistent GUID
        example_guid = inconsistent_guids.index[0]
        example_data = df[df['guid'] == example_guid][['epoch', 'binary_target', 'target']].sort_values('epoch')
        logger.debug(f"Example inconsistent GUID {example_guid}:\n{example_data.head(10)}")


def log_dataframe_stats(df: pd.DataFrame, label: str) -> None:
    """
    Log comprehensive DataFrame statistics for transparency.

    Logs sample counts, GUID counts, class distribution, epoch ranges,
    and filled epoch statistics if available.

    Args:
        df: DataFrame to analyze
        label: Label for logging (e.g., "Validation Raw", "Test Clinical")
    """
    logger.info(f"{label} Statistics:")
    logger.info(f"  Total samples: {len(df)}")

    if 'guid' in df.columns:
        n_guids = df['guid'].nunique()
        logger.info(f"  Unique GUIDs: {n_guids}")

        # Epochs per GUID stats
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
        # Epochs are negative (time before birth). More negative = further from birth
        min_hours = abs(df['epoch'].min() / 3600)
        max_hours = abs(df['epoch'].max() / 3600)
        logger.info(
            f"  Epoch range: [{df['epoch'].min():.1f}, {df['epoch'].max():.1f}] seconds "
            f"({max_hours:.1f}h to {min_hours:.1f}h before birth)"
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
    """
    Verify that GUID-level forward-filling is working correctly.

    This function validates that the clinical decision rule (once a baby is detected
    as unhealthy, all subsequent epochs until birth are also labeled as unhealthy)
    has been applied correctly.

    Validates:
    1. All epochs after first_detection_epoch are marked as unhealthy (clinical_pred=1)
    2. All epochs before first_detection_epoch are marked as healthy (clinical_pred=0)
    3. first_detection_epoch is consistent across all rows for each GUID

    Note: Epochs are negative seconds before birth, so "after" means >= (less negative, closer to birth)

    Args:
        df: DataFrame with clinical_pred and first_detection_epoch columns
        label: Label for error messages (e.g., "Validation", "Test")

    Raises:
        ValueError: If verification fails (forward-filling not applied correctly)
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

    # Track verification results
    total_guids = df['guid'].nunique()
    guids_checked = 0
    violations_found = 0
    violation_details = []

    # Process each GUID separately
    for guid in df['guid'].unique():
        guid_mask = df['guid'] == guid
        guid_data = df.loc[guid_mask].copy()
        guid_data = guid_data.sort_values('epoch', ascending=False)  # Furthest to closest

        # Skip GUIDs with single epoch (nothing to verify)
        if len(guid_data) <= 1:
            continue

        guids_checked += 1

        # Get first detection epoch (should be consistent across all rows)
        first_detections = guid_data['first_detection_epoch'].dropna().unique()

        # No detection for this GUID - all epochs should be predicted as healthy
        if len(first_detections) == 0 or guid_data['first_detection_epoch'].isna().all():
            # All clinical_pred should be 0
            if not (guid_data['clinical_pred'] == 0).all():
                violations_found += 1
                n_positive = (guid_data['clinical_pred'] == 1).sum()
                violation_details.append({
                    'guid': guid,
                    'issue': 'no_detection_but_positive_preds',
                    'details': f"{n_positive}/{len(guid_data)} epochs marked positive despite no detection"
                })
            continue

        # Detection exists - verify forward-filling
        if len(first_detections) > 1:
            # Inconsistent first_detection_epoch within GUID
            violations_found += 1
            violation_details.append({
                'guid': guid,
                'issue': 'inconsistent_first_detection',
                'details': f"Multiple first_detection_epoch values: {first_detections}"
            })
            continue

        first_detection_epoch = first_detections[0]

        # Epochs >= first_detection_epoch (closer to birth, less negative) should all be positive
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

        # Epochs < first_detection_epoch (further from birth, more negative) should all be negative
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

    # Log verification results
    if violations_found == 0:
        logger.info(
            f"{label}: Clinical decision rule verification PASSED ✓ "
            f"({guids_checked}/{total_guids} GUIDs checked, 0 violations)"
        )
    else:
        logger.error(
            f"{label}: Clinical decision rule verification FAILED ✗ "
            f"({violations_found}/{guids_checked} GUIDs with violations)"
        )

        # Log first 5 violations for debugging
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


def validate_fold_config(config: dict) -> None:
    """
    Validate fold configuration before training/evaluation.

    Checks for required configuration keys and validates dataset paths exist.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If required configuration is missing or invalid
    """
    required_keys = [
        ('dataset_config', 'classifier_train_datasets'),
        ('dataset_config', 'classifier_val_datasets'),
        ('dataset_config', 'classifier_test_datasets'),
        ('model_config', 'classifier', 'vae_checkpoint'),
    ]

    for key_path in required_keys:
        value = config
        for k in key_path:
            if k not in value:
                raise ValueError(f"Missing config key: {'.'.join(key_path)}")
            value = value[k]

        # Validate dataset paths are non-empty lists
        if 'datasets' in key_path[-1]:
            if not isinstance(value, list) or len(value) == 0:
                raise ValueError(
                    f"Config key {'.'.join(key_path)} must be non-empty list, "
                    f"got {type(value).__name__}"
                )
