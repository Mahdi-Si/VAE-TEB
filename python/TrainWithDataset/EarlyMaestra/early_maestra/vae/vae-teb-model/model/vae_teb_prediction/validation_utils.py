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

    Args:
        df: Input DataFrame with 'epoch' column (seconds before birth)

    Returns:
        DataFrame with epoch_hours column added if missing

    Raises:
        ValueError: If 'epoch' column is missing
    """
    if 'epoch_hours' in df.columns:
        return df

    if 'epoch' not in df.columns:
        raise ValueError("DataFrame missing required 'epoch' column for time conversion")

    df = df.copy()
    df['epoch_hours'] = df['epoch'] / 3600
    logger.debug(f"Added epoch_hours column (converted from epoch): {len(df)} rows")

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

    # Validate epoch values
    if (df['epoch'] < 0).any():
        negative_count = (df['epoch'] < 0).sum()
        raise ValueError(
            f"{data_type} has {negative_count} samples with negative epoch values. "
            f"Epoch should represent time before birth (non-negative)."
        )

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
        logger.info(
            f"  Epoch range: [{df['epoch'].min():.1f}, {df['epoch'].max():.1f}] seconds "
            f"({df['epoch'].min()/3600:.1f}h - {df['epoch'].max()/3600:.1f}h before birth)"
        )

    if 'is_filled' in df.columns:
        n_filled = df['is_filled'].sum()
        pct = n_filled / len(df) * 100 if len(df) > 0 else 0
        logger.info(f"  Filled epochs: {n_filled} ({pct:.1f}%)")

    if 'clinical_pred' in df.columns:
        n_positive = (df['clinical_pred'] == 1).sum()
        pct_positive = n_positive / len(df) * 100 if len(df) > 0 else 0
        logger.info(f"  Positive predictions: {n_positive} ({pct_positive:.1f}%)")


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
