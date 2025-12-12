"""
Post-training evaluation for classifier model.

After training completes:
1. Load best checkpoint (based on validation accuracy)
2. Run inference on validation set → save predictions, determine threshold
3. Run inference on test set with threshold → save predictions

Saves: guid, epoch, target, binary_target, predicted_class, prob_class_0, prob_class_1
"""

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

from model.vae_teb_prediction.vae_teb_model_prediction import SeqVae
from model.vae_teb_prediction.prediction_classification_model import (
    VaeTebTimeSeriesClassifier,
    BiLSTMAttentionClassifier,
    LSTMClassifier,
    CNN1DClassifier,
    TransformerClassifier,
)
from hdf5_dataset.hdf5_dataset import create_optimized_dataloader
from train.graph_models_utils import load_checkpoint_strict
from model.vae_teb_prediction.validation_utils import (
    ensure_epoch_hours,
    validate_predictions_df,
    validate_guid_consistency,
    log_dataframe_stats
)


def load_best_checkpoint(checkpoint_dir: str, device: str = 'cuda:0') -> torch.nn.Module:
    """
    Load the best model checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Find best checkpoint (highest accuracy or lowest loss)
    ckpt_files = list(checkpoint_dir.glob("*.ckpt"))

    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    # Sort by modification time (most recent first) or by filename
    ckpt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    best_ckpt = ckpt_files[0]

    logger.info(f"Loading checkpoint: {best_ckpt}")

    # Load checkpoint
    checkpoint = torch.load(best_ckpt, map_location=device)

    # The checkpoint contains the LightningModule state
    # We need to extract the underlying model state
    state_dict = checkpoint['state_dict']

    # Remove 'model.' prefix from keys (LightningModule wraps the model)
    model_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # Remove 'model.' prefix
            model_state_dict[new_key] = value

    return model_state_dict, best_ckpt


def find_latest_checkpoint_in_fold(fold_dir: Path) -> Path:
    """
    Locate the most recent checkpoint within a fold directory.
    Searches recursively to accommodate timestamped run folders.
    """
    ckpt_files = sorted(
        fold_dir.rglob("*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found under {fold_dir}")
    return ckpt_files[0]


def create_model_from_config(config: Dict, device: str = 'cuda:0') -> VaeTebTimeSeriesClassifier:
    """
    Create model from configuration.

    Args:
        config: Configuration dictionary
        device: Device to create model on

    Returns:
        VaeTebTimeSeriesClassifier model
    """
    classifier_config = config['model_config']['classifier']

    # Create VAE model
    vae_model = SeqVae()

    # Load VAE checkpoint
    vae_checkpoint = classifier_config['vae_checkpoint']
    logger.info(f"Loading VAE from: {vae_checkpoint}")
    vae_state = torch.load(vae_checkpoint, map_location=device)
    if 'state_dict' in vae_state:
        vae_state = vae_state['state_dict']
    vae_model.load_state_dict(vae_state, strict=False)

    # Create classifier
    classifier_type = classifier_config.get('type', 'lstm')
    latent_dim = classifier_config.get('latent_dim', 16)
    num_classes = classifier_config.get('num_classes', 2)
    classifier = LSTMClassifier(
        input_dim=latent_dim,
        num_classes=num_classes,
        hidden_dim=classifier_config.get('hidden_dim', 128),
        num_layers=classifier_config.get('num_layers', 2),
        bidirectional=classifier_config.get('bidirectional', False),
        dropout=classifier_config.get('dropout', 0.1),
    )

    model = VaeTebTimeSeriesClassifier(
        vae_model=vae_model,
        classifier=classifier,
        freeze_vae=classifier_config.get('freeze_vae', True),
        use_posterior=classifier_config.get('use_posterior', True),
        sample_latent=classifier_config.get('sample_latent', False),
        class_weights=classifier_config.get('class_weights'),
    )

    return model


def run_inference(
    model: torch.nn.Module,
    dataloader,
    device: str = 'cuda:0'
) -> pd.DataFrame:
    """
    Run inference on a dataset and collect predictions.

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on

    Returns:
        DataFrame with predictions
    """
    model.eval()
    model.to(device)

    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            y_st = batch.fhr_st.to(device)
            y_ph = batch.fhr_ph.to(device)
            x_ph = batch.fhr_up_ph.to(device)
            target_seq = batch.target.to(device)

            # Get metadata
            guid = batch.guid if hasattr(batch, 'guid') else None
            epoch_val = batch.epoch if hasattr(batch, 'epoch') else None
            cs_label = batch.cs_label if hasattr(batch, 'cs_label') else None
            bg_label = batch.bg_label if hasattr(batch, 'bg_label') else None

            # Aggregate target to single label per sample
            target_labels = target_seq.max(dim=1)[0]  # (B,) - values 1, 2, or 3
            binary_labels = (target_labels > 1).long()  # (B,) - class 0 or 1

            # Get predictions
            outputs = model(y_st=y_st, y_ph=y_ph, x_ph=x_ph)
            probs = outputs["probs"]  # (B, 2)
            preds = outputs["preds"]  # (B,)

            # Move to CPU
            target_labels = target_labels.cpu().numpy()
            binary_labels = binary_labels.cpu().numpy()
            preds = preds.cpu().numpy()
            probs = probs.cpu().numpy()

            # Store predictions
            batch_size = y_st.size(0)
            for i in range(batch_size):
                pred_dict = {
                    'guid': guid[i] if guid is not None else f"sample_{len(predictions)}",
                    'epoch': float(epoch_val[i]) if epoch_val is not None else -1.0,
                    'cs_label': bool(cs_label[i]) if cs_label is not None else None,
                    'bg_label': bool(bg_label[i]) if bg_label is not None else None,
                    'target': int(target_labels[i]),  # Original (1, 2, or 3)
                    'binary_target': int(binary_labels[i]),  # Binary (0 or 1)
                    'predicted_class': int(preds[i]),  # Predicted (0 or 1)
                    'prob_class_0': float(probs[i, 0]),
                    'prob_class_1': float(probs[i, 1]),
                }
                predictions.append(pred_dict)

    return pd.DataFrame(predictions)


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

    if 'guid' not in df.columns:
        epochs = np.sort(np.unique(df['epoch'].values.astype(float)))
        return float(np.median(np.diff(epochs))) if len(epochs) >= 2 else 0.0

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

        # Verify clinical decision rule applied correctly
        from model.vae_teb_prediction.validation_utils import verify_clinical_decision_rule
        verify_clinical_decision_rule(df, "Clinical Decision Rule")

    return df


def fill_missing_epochs(
    df: pd.DataFrame,
    max_gap_multiplier: Optional[float] = None,
    log_summary: bool = True
) -> pd.DataFrame:
    """
    Fill missing epochs for each GUID using forward-filling strategy.

    Args:
        df: Predictions dataframe with clinical decision rule applied
        max_gap_multiplier: If provided, only fill gaps <= multiplier * typical_interval.
                           If None, fill all missing epochs within each GUID range.

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

        # FIX #7: Use np.arange with explicit step instead of linspace
        # This ensures spacing equals typical_interval exactly
        expected_epochs = np.arange(min_epoch, max_epoch + typical_interval/2, typical_interval)

        # Round to avoid floating point issues
        expected_epochs = np.round(expected_epochs, 1)
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


def find_threshold_for_fpr_epoch(
    val_df: pd.DataFrame,
    target_fpr: float = 0.05
) -> Tuple[float, Dict]:
    """
    Find classification threshold that achieves target false positive rate at epoch level.

    This is the original per-epoch FPR approach.

    Args:
        val_df: Validation predictions DataFrame
        target_fpr: Target false positive rate (e.g., 0.05 for 5%)

    Returns:
        Tuple of (threshold, metrics_dict)
    """
    # Get true negatives (class 0 samples)
    class_0_mask = val_df['binary_target'] == 0
    class_0_probs = val_df.loc[class_0_mask, 'prob_class_1'].values

    if len(class_0_probs) == 0:
        logger.warning("No class 0 samples in validation set!")
        return 0.5, {}

    # Sort probabilities
    sorted_probs = np.sort(class_0_probs)

    # Find threshold at target FPR percentile
    threshold_idx = int(len(sorted_probs) * (1 - target_fpr))
    threshold = sorted_probs[threshold_idx] if threshold_idx < len(sorted_probs) else sorted_probs[-1]

    # Apply threshold to get predictions
    val_df['threshold_pred'] = (val_df['prob_class_1'] >= threshold).astype(int)

    # Compute metrics
    accuracy = (val_df['threshold_pred'] == val_df['binary_target']).mean()

    # Class-specific metrics
    class_0_samples = val_df[val_df['binary_target'] == 0]
    class_1_samples = val_df[val_df['binary_target'] == 1]

    if len(class_0_samples) > 0:
        fpr = (class_0_samples['threshold_pred'] == 1).mean()  # False positives / negatives
        specificity = 1 - fpr
    else:
        fpr = 0.0
        specificity = 1.0

    if len(class_1_samples) > 0:
        sensitivity = (class_1_samples['threshold_pred'] == 1).mean()  # True positives / positives
    else:
        sensitivity = 1.0

    metrics = {
        'threshold': float(threshold),
        'target_fpr': target_fpr,
        'actual_fpr': float(fpr),
        'specificity': float(specificity),
        'sensitivity': float(sensitivity),
        'accuracy': float(accuracy),
        'n_class_0': len(class_0_samples),
        'n_class_1': len(class_1_samples),
    }

    logger.info("=" * 80)
    logger.info("THRESHOLD DETERMINATION (Validation Set)")
    logger.info("=" * 80)
    logger.info(f"Target FPR: {target_fpr:.4f}")
    logger.info(f"Selected threshold: {threshold:.4f}")
    logger.info(f"Actual FPR: {fpr:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")
    logger.info(f"Sensitivity: {sensitivity:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info("=" * 80)

    return threshold, metrics


def find_threshold_for_fpr_guid(
    val_df: pd.DataFrame,
    target_fpr: float = 0.05,
    threshold_candidates: np.ndarray = None
) -> Tuple[float, Dict]:
    """
    Find classification threshold that achieves target false positive rate at GUID level.

    This approach is more clinically relevant as it considers per-patient (GUID) outcomes
    after applying the clinical decision rule.

    Args:
        val_df: Validation predictions DataFrame
        target_fpr: Target false positive rate at GUID level (e.g., 0.05 for 5%)
        threshold_candidates: Array of thresholds to test (default: 0.0 to 1.0 in steps of 0.01)

    Returns:
        Tuple of (threshold, metrics_dict)
    """
    if threshold_candidates is None:
        probs = val_df['prob_class_1'].dropna().values.astype(float)
        if len(probs) == 0:
            logger.warning("No probabilities found in validation set!")
            return 0.5, {}
        threshold_candidates = np.unique(np.clip(probs, 0.0, 1.0))
        threshold_candidates = np.unique(np.concatenate(([0.0], threshold_candidates, [1.0])))

    threshold_candidates = np.asarray(threshold_candidates, dtype=float)
    threshold_candidates.sort()

    best_threshold = float(threshold_candidates[len(threshold_candidates) // 2]) if len(threshold_candidates) else 0.5
    best_fpr_diff = float('inf')
    best_metrics = {}

    # FPR is monotonic non-increasing with threshold; use binary search over candidates.
    lo, hi = 0, len(threshold_candidates) - 1
    max_iter = min(25, len(threshold_candidates))  # guard for pathological candidate lists
    it = 0

    while lo <= hi and it < max_iter:
        mid = (lo + hi) // 2
        thresh = float(threshold_candidates[mid])

        # Apply clinical decision rule at this threshold
        df_clinical = apply_clinical_decision_rule(val_df, thresh, verify=False)

        # Determine GUID-level labels
        guid_stats = df_clinical.groupby('guid').agg({
            'binary_target': 'max',  # GUID is unhealthy if any epoch is unhealthy
            'clinical_pred': 'max'   # GUID is predicted unhealthy if any epoch predicted unhealthy
        }).reset_index()

        # Compute GUID-level metrics
        healthy_guids = guid_stats[guid_stats['binary_target'] == 0]
        unhealthy_guids = guid_stats[guid_stats['binary_target'] == 1]

        if len(healthy_guids) == 0:
            continue

        # GUID-level FPR
        guid_fpr = (healthy_guids['clinical_pred'] == 1).sum() / len(healthy_guids)

        # Check if this is closest to target FPR
        fpr_diff = abs(guid_fpr - target_fpr)
        if fpr_diff < best_fpr_diff:
            best_fpr_diff = fpr_diff
            best_threshold = thresh

            # Compute additional metrics at this threshold
            guid_specificity = 1 - guid_fpr

            if len(unhealthy_guids) > 0:
                guid_sensitivity = (unhealthy_guids['clinical_pred'] == 1).sum() / len(unhealthy_guids)
            else:
                guid_sensitivity = 1.0

            guid_accuracy = (guid_stats['binary_target'] == guid_stats['clinical_pred']).mean()

            best_metrics = {
                'threshold': float(best_threshold),
                'target_fpr': target_fpr,
                'actual_fpr': float(guid_fpr),
                'specificity': float(guid_specificity),
                'sensitivity': float(guid_sensitivity),
                'accuracy': float(guid_accuracy),
                'n_healthy_guids': len(healthy_guids),
                'n_unhealthy_guids': len(unhealthy_guids),
            }

        # Adjust search bounds
        if guid_fpr > target_fpr:
            lo = mid + 1  # increase threshold to reduce FPR
        else:
            hi = mid - 1  # decrease threshold to increase FPR

        it += 1

    logger.info("=" * 80)
    logger.info("GUID-LEVEL THRESHOLD DETERMINATION (Validation Set)")
    logger.info("=" * 80)
    logger.info(f"Target GUID-level FPR: {target_fpr:.4f}")
    logger.info(f"Selected threshold: {best_threshold:.4f}")
    logger.info(f"Actual GUID-level FPR: {best_metrics.get('actual_fpr', 0):.4f}")
    logger.info(f"GUID-level Specificity: {best_metrics.get('specificity', 0):.4f}")
    logger.info(f"GUID-level Sensitivity: {best_metrics.get('sensitivity', 0):.4f}")
    logger.info(f"GUID-level Accuracy: {best_metrics.get('accuracy', 0):.4f}")
    logger.info(f"Healthy GUIDs: {best_metrics.get('n_healthy_guids', 0)}")
    logger.info(f"Unhealthy GUIDs: {best_metrics.get('n_unhealthy_guids', 0)}")
    logger.info("=" * 80)

    return best_threshold, best_metrics


def find_threshold_for_fpr_at_time_window(
    val_df: pd.DataFrame,
    target_fpr: float = 0.05,
    time_window_hours: float = 1.0,
    max_gap_multiplier: Optional[float] = None
) -> Tuple[float, Dict]:
    """
    Find threshold to achieve target FPR specifically at a time window before delivery.

    This function optimizes the threshold to achieve the desired FPR at a specific
    time window (e.g., 1 hour before delivery), which is more clinically relevant
    than global optimization across all time points.

    Args:
        val_df: Validation dataframe with columns: guid, epoch, binary_target, prob_class_1
        target_fpr: Desired false positive rate (default 0.05)
        time_window_hours: Decision time in hours before delivery (default 1.0 = 1h before delivery)
        max_gap_multiplier: For epoch filling. If None, fill all missing epochs.

    Returns:
        threshold: Optimal threshold value
        metrics_dict: Performance metrics at this threshold
    """
    logger.info("=" * 80)
    logger.info(f"Finding threshold for target FPR={target_fpr} at decision time {time_window_hours}h before delivery")
    logger.info("=" * 80)

    val_df = ensure_epoch_hours(val_df.copy())

    probs = val_df['prob_class_1'].dropna().values.astype(float)
    if len(probs) == 0:
        logger.error("No probabilities found - cannot determine threshold")
        return 0.5, {}

    threshold_candidates = np.unique(np.clip(probs, 0.0, 1.0))
    threshold_candidates = np.unique(np.concatenate(([0.0], threshold_candidates, [1.0])))
    threshold_candidates.sort()

    best_threshold = float(threshold_candidates[len(threshold_candidates) // 2]) if len(threshold_candidates) else 0.5
    best_fpr_diff = float('inf')
    best_row = None

    # FPR is monotonic non-increasing with threshold; use binary search over candidates.
    lo, hi = 0, len(threshold_candidates) - 1
    max_iter = min(25, len(threshold_candidates))
    it = 0

    while lo <= hi and it < max_iter:
        mid = (lo + hi) // 2
        thresh = float(threshold_candidates[mid])

        df_clinical = apply_clinical_decision_rule(val_df, thresh, verify=False)
        df_clinical = fill_missing_epochs(df_clinical, max_gap_multiplier, log_summary=False)

        guid_stats = guid_snapshot_at_or_before_time(df_clinical, time_window_hours)
        if len(guid_stats) == 0:
            it += 1
            break

        healthy_guids = guid_stats[guid_stats['binary_target'] == 0]
        unhealthy_guids = guid_stats[guid_stats['binary_target'] == 1]

        if len(healthy_guids) == 0:
            it += 1
            break

        guid_fpr = (healthy_guids['pred_at_time'] == 1).mean()
        guid_sensitivity = (unhealthy_guids['pred_at_time'] == 1).mean() if len(unhealthy_guids) > 0 else 0.0

        fpr_diff = abs(guid_fpr - target_fpr)
        if fpr_diff < best_fpr_diff:
            best_fpr_diff = fpr_diff
            best_threshold = thresh
            best_row = {
                'threshold': thresh,
                'guid_fpr': float(guid_fpr),
                'guid_sensitivity': float(guid_sensitivity),
                'n_healthy_guids': int(len(healthy_guids)),
                'n_unhealthy_guids': int(len(unhealthy_guids)),
            }

        if guid_fpr > target_fpr:
            lo = mid + 1  # increase threshold to reduce FPR
        else:
            hi = mid - 1  # decrease threshold to increase FPR

        it += 1

    if best_row is None:
        logger.error("No valid thresholds found")
        return 0.5, {}

    logger.info(f"Selected threshold: {best_threshold:.3f}")
    logger.info(f"Achieved GUID-level FPR at {time_window_hours}h: {best_row['guid_fpr']:.4f} (target: {target_fpr})")
    logger.info(f"GUID-level sensitivity at {time_window_hours}h: {best_row['guid_sensitivity']:.4f}")
    logger.info(f"Healthy GUIDs in window: {int(best_row['n_healthy_guids'])}")
    logger.info(f"Unhealthy GUIDs in window: {int(best_row['n_unhealthy_guids'])}")
    logger.info("=" * 80)

    metrics_dict = {
        'threshold': best_threshold,
        'guid_fpr_at_window': best_row['guid_fpr'],
        'guid_sensitivity_at_window': best_row['guid_sensitivity'],
        'time_window_hours': time_window_hours,
        'n_healthy_guids': int(best_row['n_healthy_guids']),
        'n_unhealthy_guids': int(best_row['n_unhealthy_guids']),
        'fpr': best_row['guid_fpr'],
        'sensitivity': best_row['guid_sensitivity']
    }

    return best_threshold, metrics_dict


def find_optimal_thresholds(
    val_df: pd.DataFrame,
    target_fpr: float = 0.05,
    time_window_hours: float = 1.0,
    max_gap_multiplier: Optional[float] = None
) -> Dict:
    """
    Compute epoch-level, GUID-level, and time-specific thresholds for comparison.

    Args:
        val_df: Validation predictions DataFrame
        target_fpr: Target false positive rate for threshold determination
        time_window_hours: Decision time for time-specific optimization (default 1.0h before delivery)
        max_gap_multiplier: For missing epoch filling in time-specific optimization (None fills all)

    Returns:
        Dictionary with epoch-level, GUID-level, and time-specific threshold information
    """
    logger.info("Computing optimal thresholds using three approaches...")

    # Epoch-level threshold (global optimization at epoch level)
    epoch_threshold, epoch_metrics = find_threshold_for_fpr_epoch(val_df, target_fpr)

    # GUID-level threshold (global optimization at GUID level)
    guid_threshold, guid_metrics = find_threshold_for_fpr_guid(val_df, target_fpr)

    # Time-specific threshold (optimized for specific time window before delivery)
    time_threshold, time_metrics = find_threshold_for_fpr_at_time_window(
        val_df,
        target_fpr=target_fpr,
        time_window_hours=time_window_hours,
        max_gap_multiplier=max_gap_multiplier
    )

    results = {
        'epoch_level': {
            'threshold': epoch_threshold,
            **epoch_metrics
        },
        'guid_level': {
            'threshold': guid_threshold,
            **guid_metrics
        },
        'time_specific': {
            'threshold': time_threshold,
            **time_metrics
        },
        'target_fpr': target_fpr,
        'time_window_hours': time_window_hours
    }

    logger.info("=" * 80)
    logger.info("THRESHOLD COMPARISON")
    logger.info("=" * 80)
    logger.info(f"Epoch-level threshold:     {epoch_threshold:.4f} (FPR: {epoch_metrics['actual_fpr']:.4f})")
    logger.info(f"GUID-level threshold:      {guid_threshold:.4f} (FPR: {guid_metrics.get('actual_fpr', guid_metrics.get('fpr', 0)):.4f})")
    logger.info(f"Time-specific threshold:   {time_threshold:.4f} (FPR at {time_window_hours}h: {time_metrics['guid_fpr_at_window']:.4f})")
    logger.info(f"")
    logger.info(f"RECOMMENDATION: Use time-specific threshold ({time_threshold:.4f}) for clinically relevant evaluation")
    logger.info("=" * 80)

    return results


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
    # CRITICAL FIX: Epochs are negative! Use abs(min()) to get furthest time from birth
    # Example: epochs from -43200 to -3600 → min=-43200 → abs=-43200 → 12 hours before birth
    max_epoch_hours = abs(df['epoch'].min()) / 3600

    # Infer bin size from typical epoch interval (fallback to 20 minutes if inference fails)
    inferred_seconds = infer_epoch_interval_seconds(df)
    bin_size_hours = (inferred_seconds / 3600.0) if inferred_seconds > 0 else (1.0 / 3.0)

    # Convert exclusion from minutes to hours
    exclude_hours = exclude_last_minutes / 60.0  # 30min = 0.5h

    # Create uniform bins starting from exclude_hours (e.g., 0.5h) to max_epoch_hours
    # This excludes the last N minutes before birth from analysis
    bins = np.arange(exclude_hours, max_epoch_hours + bin_size_hours, bin_size_hours)

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
    # CRITICAL FIX: Epochs are negative! Use abs(min()) to get furthest time from birth
    max_epoch_hours = abs(df['epoch'].min()) / 3600

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

def create_subgroup_filters() -> Dict[str, callable]:
    """
    Create filter functions for all subgroups.

    Returns:
        Dictionary mapping subgroup_name -> filter_function
    """
    return {
        # Diagnosis-based subgroups
        'acidosis': lambda df: df['target'] == 2,
        'hie': lambda df: df['target'] == 3,
        'healthy': lambda df: df['target'] == 1,

        # CS status subgroups
        'cs_positive': lambda df: df['cs_label'] == True,
        'cs_negative': lambda df: df['cs_label'] == False,

        # BG label subgroups
        'bg_positive': lambda df: df['bg_label'] == True,
        'bg_negative': lambda df: df['bg_label'] == False,

        # Combined subgroups
        'acidosis_cs_pos': lambda df: (df['target'] == 2) & (df['cs_label'] == True),
        'acidosis_cs_neg': lambda df: (df['target'] == 2) & (df['cs_label'] == False),
        'hie_cs_pos': lambda df: (df['target'] == 3) & (df['cs_label'] == True),
        'hie_cs_neg': lambda df: (df['target'] == 3) & (df['cs_label'] == False),
    }


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

    # Ensure epoch_hours exists
    if 'epoch_hours' not in subgroup_df.columns:
        subgroup_df['epoch_hours'] = abs(subgroup_df['epoch']) / 3600

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


def plot_metrics_vs_time(
    df: pd.DataFrame,
    output_dir: Path,
    prefix: str = "",
    exclude_last_minutes: float = 30.0
) -> None:
    """
    Plot sensitivity and specificity as function of time before birth.

    Args:
        df: Predictions dataframe with clinical_pred
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames (e.g., "epoch_threshold_" or "guid_threshold_")
        exclude_last_minutes: Exclude last N minutes before birth from plots (default: 30 min)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute dynamic time bins
    time_bins = compute_time_bins(df, exclude_last_minutes=exclude_last_minutes)

    # Ensure epoch_hours column exists
    df = ensure_epoch_hours(df.copy())

    if len(time_bins) < 2:
        logger.warning("Not enough time bins to plot sensitivity/specificity vs time")
        return

    # ------------------------------
    # Epoch-level metrics per bin
    # ------------------------------
    epoch_rows = []
    for i in range(len(time_bins) - 1):
        bin_start, bin_end = time_bins[i], time_bins[i + 1]
        bin_center = (bin_start + bin_end) / 2
        bin_mask = (df['epoch_hours'] >= bin_start) & (df['epoch_hours'] < bin_end)
        bin_data = df[bin_mask]

        if len(bin_data) == 0:
            continue

        class_0 = bin_data[bin_data['binary_target'] == 0]
        class_1 = bin_data[bin_data['binary_target'] == 1]

        epoch_rows.append({
            'bin_center': float(bin_center),
            'sensitivity': (class_1['clinical_pred'] == 1).mean() if len(class_1) > 0 else np.nan,
            'specificity': 1 - (class_0['clinical_pred'] == 1).mean() if len(class_0) > 0 else np.nan,
            'n_class_0': int(len(class_0)),
            'n_class_1': int(len(class_1)),
        })

    epoch_df = pd.DataFrame(epoch_rows).sort_values('bin_center', ascending=False)

    if len(epoch_df) > 0:
        x = epoch_df['bin_center'].values
        bin_width = float(np.median(np.diff(np.sort(x)))) if len(x) > 1 else 0.25
        bar_width = 0.85 * bin_width

        fig, (ax, ax_counts) = plt.subplots(
            2, 1,
            figsize=(12, 7),
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True
        )

        ax.plot(x, epoch_df['sensitivity'], marker='o', label='Sensitivity (epoch-level)', linewidth=2, markersize=5)
        ax.plot(x, epoch_df['specificity'], marker='s', label='Specificity (epoch-level)', linewidth=2, markersize=5)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Sensitivity/Specificity vs Time Before Birth (Epoch-Level)', fontsize=13)
        ax.legend(fontsize=10, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        ax_counts.bar(x, epoch_df['n_class_0'], width=bar_width, label='Healthy epochs', alpha=0.7)
        ax_counts.bar(x, epoch_df['n_class_1'], bottom=epoch_df['n_class_0'], width=bar_width,
                      label='Unhealthy epochs', alpha=0.7)
        ax_counts.set_ylabel('Epochs', fontsize=10)
        ax_counts.set_xlabel('Hours Before Birth (segment start)', fontsize=12)
        ax_counts.grid(True, alpha=0.2, axis='y')
        ax_counts.legend(fontsize=9, loc='upper left')

        # Earlier times on the left, closer to birth on the right
        ax.invert_xaxis()

        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}sensitivity_specificity_vs_time.png", dpi=150)
        plt.close()
        logger.info(f"Saved: {prefix}sensitivity_specificity_vs_time.png")
    else:
        logger.warning("No epoch-level bins contained data; skipping epoch-level time plot")

    # ------------------------------
    # GUID-level decision-time curve
    # ------------------------------
    times = epoch_df['bin_center'].values if len(epoch_df) > 0 else np.array([(time_bins[i] + time_bins[i + 1]) / 2 for i in range(len(time_bins) - 1)])
    guid_rows = []
    for t in times:
        snap = guid_snapshot_at_or_before_time(df, float(t), pred_col='clinical_pred')
        if len(snap) == 0:
            continue

        healthy = snap[snap['binary_target'] == 0]
        unhealthy = snap[snap['binary_target'] == 1]

        guid_rows.append({
            'bin_center': float(t),
            'guid_sensitivity': (unhealthy['pred_at_time'] == 1).mean() if len(unhealthy) > 0 else np.nan,
            'guid_specificity': 1 - (healthy['pred_at_time'] == 1).mean() if len(healthy) > 0 else np.nan,
            'n_healthy_guids': int(len(healthy)),
            'n_unhealthy_guids': int(len(unhealthy)),
        })

    guid_df = pd.DataFrame(guid_rows).sort_values('bin_center', ascending=False)
    if len(guid_df) == 0:
        logger.warning("No GUID snapshots available for decision-time plot; skipping")
        return

    x = guid_df['bin_center'].values
    bin_width = float(np.median(np.diff(np.sort(x)))) if len(x) > 1 else 0.25
    bar_width = 0.85 * bin_width

    fig, (ax, ax_counts) = plt.subplots(
        2, 1,
        figsize=(12, 7),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )

    ax.plot(x, guid_df['guid_sensitivity'], marker='o', label='Sensitivity (GUID @ time)', linewidth=2, markersize=5)
    ax.plot(x, guid_df['guid_specificity'], marker='s', label='Specificity (GUID @ time)', linewidth=2, markersize=5)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Sensitivity/Specificity vs Time Before Birth (GUID State at Time)', fontsize=13)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    ax_counts.bar(x, guid_df['n_healthy_guids'], width=bar_width, label='Healthy GUIDs', alpha=0.7)
    ax_counts.bar(x, guid_df['n_unhealthy_guids'], bottom=guid_df['n_healthy_guids'], width=bar_width,
                  label='Unhealthy GUIDs', alpha=0.7)
    ax_counts.set_ylabel('GUIDs', fontsize=10)
    ax_counts.set_xlabel('Hours Before Birth (decision time)', fontsize=12)
    ax_counts.grid(True, alpha=0.2, axis='y')
    ax_counts.legend(fontsize=9, loc='upper left')

    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}guid_sensitivity_specificity_vs_time.png", dpi=150)
    plt.close()

    logger.info(f"Saved: {prefix}guid_sensitivity_specificity_vs_time.png")


def plot_roc_curves_by_time(
    df: pd.DataFrame,
    output_dir: Path,
    prefix: str = "",
    exclude_last_minutes: float = 30.0
) -> None:
    """
    Plot ROC curves for different time windows before birth.

    Args:
        df: Predictions dataframe
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
        exclude_last_minutes: Exclude last N minutes before birth from plots (default: 30 min)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute dynamic time windows
    time_windows = compute_time_windows(df, exclude_last_minutes=exclude_last_minutes)

    # FIX #5: Ensure epoch_hours column exists
    df = ensure_epoch_hours(df.copy())

    # Plot all ROC curves on one figure
    fig, ax = plt.subplots(figsize=(10, 10))

    for start_h, end_h in time_windows:
        window_mask = (df['epoch_hours'] >= start_h) & (df['epoch_hours'] < end_h)
        window_data = df[window_mask]

        if len(window_data) < 10 or window_data['binary_target'].nunique() < 2:
            continue

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(window_data['binary_target'], window_data['prob_class_1'])
        roc_auc = auc(fpr, tpr)

        # Plot (show negative time range: -end to -start hours before birth)
        label = f"{-end_h:.1f} to {-start_h:.1f}h (AUC={roc_auc:.3f})"
        ax.plot(fpr, tpr, label=label, linewidth=2)

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.500)')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves by Time Window Before Birth', fontsize=14)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}roc_curves_by_time.png", dpi=150)
    plt.close()

    logger.info(f"Saved: {prefix}roc_curves_by_time.png")


def plot_detection_timing(
    df: pd.DataFrame,
    output_dir: Path,
    prefix: str = ""
) -> None:
    """
    Analyze when unhealthy babies are first detected.

    Args:
        df: Predictions dataframe with first_detection_epoch column
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get unique GUIDs with their first detection epochs
    guid_data = df.groupby('guid').agg({
        'first_detection_epoch': 'first',
        'binary_target': 'max'  # GUID is unhealthy if any epoch unhealthy
    }).reset_index()

    # Filter to unhealthy GUIDs only
    unhealthy_guids = guid_data[guid_data['binary_target'] == 1]
    detected = unhealthy_guids[unhealthy_guids['first_detection_epoch'].notna()]

    if len(detected) == 0:
        logger.warning("No detections found for timing analysis")
        return

    # Convert to hours (negate to show negative hours before birth)
    detection_hours = -(detected['first_detection_epoch'].values / 3600)

    # Plot 1: Histogram of detection times
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].hist(detection_hours, bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Time Before Birth (hours, 0 = delivery)', fontsize=12)
    axes[0].set_ylabel('Number of Detections', fontsize=12)
    axes[0].set_title('Distribution of First Detection Times', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Cumulative detection curve
    sorted_hours = np.sort(detection_hours)
    cumulative_pct = np.arange(1, len(sorted_hours) + 1) / len(sorted_hours) * 100

    axes[1].plot(sorted_hours, cumulative_pct, linewidth=2)
    axes[1].set_xlabel('Time Before Birth (hours, 0 = delivery)', fontsize=12)
    axes[1].set_ylabel('Cumulative % of Unhealthy GUIDs Detected', fontsize=12)
    axes[1].set_title('Cumulative Detection Curve', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}detection_timing.png", dpi=150)
    plt.close()

    logger.info(f"Saved: {prefix}detection_timing.png")


def analyze_detection_timing_statistics(
    df: pd.DataFrame,
    subgroups: Dict[str, callable] = None,
    output_dir: Path = None,
    prefix: str = ""
) -> Dict:
    """
    Comprehensive detection timing analysis with lead time metrics.

    Computes:
    - First detection time statistics (mean, median, std, percentiles)
    - Lead time before delivery (advance warning)
    - Detection rates at time windows (0.5-1h, 1-2h, 2-4h, 4-6h, 6+h)
    - Subgroup-specific timing comparison

    Args:
        df: Predictions dataframe with first_detection_epoch column
        subgroups: Optional dictionary of subgroup filters for comparison
        output_dir: Directory to save results (JSON and CSV)
        prefix: Prefix for output filenames

    Returns:
        Dictionary with detection timing statistics
    """
    # Get unique GUIDs with their first detection epochs
    guid_data = df.groupby('guid').agg({
        'first_detection_epoch': 'first',
        'binary_target': 'max'  # GUID is unhealthy if any epoch unhealthy
    }).reset_index()

    # Filter to unhealthy GUIDs
    unhealthy_guids = guid_data[guid_data['binary_target'] == 1]
    detected = unhealthy_guids[unhealthy_guids['first_detection_epoch'].notna()]

    # Overall statistics
    n_unhealthy = len(unhealthy_guids)
    n_detected = len(detected)
    detection_rate = (n_detected / n_unhealthy * 100) if n_unhealthy > 0 else 0.0

    overall_stats = {
        'n_total_guids': len(guid_data),
        'n_unhealthy_guids': n_unhealthy,
        'n_detected': n_detected,
        'n_missed': n_unhealthy - n_detected,
        'detection_rate': detection_rate
    }

    if n_detected > 0:
        # Convert to positive hours before birth
        detection_hours = abs(detected['first_detection_epoch'].values) / 3600

        overall_stats.update({
            'mean_detection_time_hours': float(np.mean(detection_hours)),
            'median_detection_time_hours': float(np.median(detection_hours)),
            'std_detection_time_hours': float(np.std(detection_hours)),
            'min_detection_time_hours': float(np.min(detection_hours)),
            'max_detection_time_hours': float(np.max(detection_hours)),
            'percentiles': {
                '25th': float(np.percentile(detection_hours, 25)),
                '50th': float(np.percentile(detection_hours, 50)),
                '75th': float(np.percentile(detection_hours, 75)),
                '90th': float(np.percentile(detection_hours, 90)),
                '95th': float(np.percentile(detection_hours, 95))
            }
        })

        # Detection counts by time window
        windows = {
            '0.5-1h': (0.5, 1),
            '1-2h': (1, 2),
            '2-4h': (2, 4),
            '4-6h': (4, 6),
            '6h+': (6, float('inf'))
        }

        detection_windows = {}
        for window_name, (start, end) in windows.items():
            count = ((detection_hours >= start) & (detection_hours < end)).sum()
            pct = (count / n_detected * 100) if n_detected > 0 else 0.0
            detection_windows[window_name] = {
                'count': int(count),
                'percentage': float(pct)
            }

        overall_stats['detection_windows'] = detection_windows
    else:
        overall_stats.update({
            'mean_detection_time_hours': None,
            'median_detection_time_hours': None,
            'std_detection_time_hours': None,
            'min_detection_time_hours': None,
            'max_detection_time_hours': None,
            'percentiles': {},
            'detection_windows': {}
        })

    results = {'overall': overall_stats}

    # Subgroup-specific statistics
    if subgroups:
        subgroup_stats = {}
        for subgroup_name, subgroup_filter in subgroups.items():
            try:
                subgroup_df = df[subgroup_filter(df)]
                subgroup_guid_data = subgroup_df.groupby('guid').agg({
                    'first_detection_epoch': 'first',
                    'binary_target': 'max'
                }).reset_index()

                subgroup_unhealthy = subgroup_guid_data[subgroup_guid_data['binary_target'] == 1]
                subgroup_detected = subgroup_unhealthy[subgroup_unhealthy['first_detection_epoch'].notna()]

                n_subgroup_unhealthy = len(subgroup_unhealthy)
                n_subgroup_detected = len(subgroup_detected)
                subgroup_detection_rate = (n_subgroup_detected / n_subgroup_unhealthy * 100) if n_subgroup_unhealthy > 0 else 0.0

                subgroup_stat = {
                    'n_unhealthy_guids': n_subgroup_unhealthy,
                    'n_detected': n_subgroup_detected,
                    'n_missed': n_subgroup_unhealthy - n_subgroup_detected,
                    'detection_rate': subgroup_detection_rate
                }

                if n_subgroup_detected > 0:
                    subgroup_hours = abs(subgroup_detected['first_detection_epoch'].values) / 3600
                    subgroup_stat.update({
                        'mean_detection_time_hours': float(np.mean(subgroup_hours)),
                        'median_detection_time_hours': float(np.median(subgroup_hours)),
                        'std_detection_time_hours': float(np.std(subgroup_hours)),
                        'min_detection_time_hours': float(np.min(subgroup_hours)),
                        'max_detection_time_hours': float(np.max(subgroup_hours))
                    })
                else:
                    subgroup_stat.update({
                        'mean_detection_time_hours': None,
                        'median_detection_time_hours': None,
                        'std_detection_time_hours': None,
                        'min_detection_time_hours': None,
                        'max_detection_time_hours': None
                    })

                subgroup_stats[subgroup_name] = subgroup_stat

            except Exception as e:
                logger.warning(f"Error computing detection stats for subgroup {subgroup_name}: {e}")
                subgroup_stats[subgroup_name] = {'error': str(e)}

        results['subgroups'] = subgroup_stats

    # Log summary
    logger.info(f"Detection Timing Statistics:")
    logger.info(f"  Overall: {n_detected}/{n_unhealthy} detected ({detection_rate:.1f}%)")
    if n_detected > 0:
        logger.info(f"  Mean detection time: {overall_stats['mean_detection_time_hours']:.2f}h before birth")
        logger.info(f"  Median detection time: {overall_stats['median_detection_time_hours']:.2f}h before birth")

    return results


def plot_detection_timing_enhanced(
    df: pd.DataFrame,
    subgroups: Dict[str, callable],
    output_dir: Path,
    prefix: str = ""
) -> None:
    """
    Enhanced detection timing plots with subgroup comparison.

    Creates 2x2 grid:
    - Top-left: Overlaid histograms by subgroup
    - Top-right: Cumulative detection curves
    - Bottom-left: Box plots (detection time distribution)
    - Bottom-right: Violin plots (lead time distribution)

    Args:
        df: Predictions dataframe
        subgroups: Dictionary of subgroup filters
        output_dir: Directory to save plots
        prefix: Prefix for filenames
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green

    # Collect detection times for each subgroup
    subgroup_detections = {}
    for idx, (subgroup_name, subgroup_filter) in enumerate(subgroups.items()):
        subgroup_df = df[subgroup_filter(df)]
        guid_data = subgroup_df.groupby('guid').agg({
            'first_detection_epoch': 'first',
            'binary_target': 'max'
        }).reset_index()

        unhealthy = guid_data[guid_data['binary_target'] == 1]
        detected = unhealthy[unhealthy['first_detection_epoch'].notna()]

        if len(detected) > 0:
            # Convert to positive hours before birth
            detection_hours = abs(detected['first_detection_epoch'].values) / 3600
            subgroup_detections[subgroup_name] = detection_hours

    if len(subgroup_detections) == 0:
        logger.warning("No detections found for enhanced timing analysis")
        return

    # Plot 1: Overlaid histograms
    for idx, (name, hours) in enumerate(subgroup_detections.items()):
        axes[0, 0].hist(hours, bins=20, alpha=0.6, label=name.title(),
                       edgecolor='black', color=colors[idx % len(colors)])
    axes[0, 0].set_xlabel('Time Before Birth (hours)', fontsize=11)
    axes[0, 0].set_ylabel('Number of Detections', fontsize=11)
    axes[0, 0].set_title('Detection Time Distribution by Subgroup', fontsize=13)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Cumulative detection curves
    for idx, (name, hours) in enumerate(subgroup_detections.items()):
        sorted_hours = np.sort(hours)
        cumulative_pct = np.arange(1, len(sorted_hours) + 1) / len(sorted_hours) * 100
        axes[0, 1].plot(sorted_hours, cumulative_pct, linewidth=2.5,
                       label=name.title(), color=colors[idx % len(colors)])
    axes[0, 1].set_xlabel('Time Before Birth (hours)', fontsize=11)
    axes[0, 1].set_ylabel('Cumulative % Detected', fontsize=11)
    axes[0, 1].set_title('Cumulative Detection Curves', fontsize=13)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 105])

    # Plot 3: Box plots
    box_data = [hours for hours in subgroup_detections.values()]
    box_labels = [name.title() for name in subgroup_detections.keys()]
    bp = axes[1, 0].boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1, 0].set_ylabel('Time Before Birth (hours)', fontsize=11)
    axes[1, 0].set_title('Detection Time Distribution (Box Plots)', fontsize=13)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: Violin plots
    positions = np.arange(1, len(subgroup_detections) + 1)
    for idx, (name, hours) in enumerate(subgroup_detections.items()):
        parts = axes[1, 1].violinplot([hours], positions=[positions[idx]],
                                     showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[idx % len(colors)])
            pc.set_alpha(0.6)
    axes[1, 1].set_xticks(positions)
    axes[1, 1].set_xticklabels([name.title() for name in subgroup_detections.keys()])
    axes[1, 1].set_ylabel('Time Before Birth (hours)', fontsize=11)
    axes[1, 1].set_title('Lead Time Distribution (Violin Plots)', fontsize=13)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}enhanced_detection_timing.png", dpi=150)
    plt.close()

    logger.info(f"Saved: {prefix}enhanced_detection_timing.png")


def create_lead_time_table(
    df: pd.DataFrame,
    subgroups: Dict[str, callable]
) -> pd.DataFrame:
    """
    Create summary table of lead time metrics by subgroup.

    Returns DataFrame with:
    - subgroup, n_unhealthy_guids, n_detected, detection_rate (%)
    - mean/median/std lead_time_hours
    - pct_detected_gt_1h, pct_detected_gt_2h, pct_detected_gt_6h
    - min/max lead_time

    Args:
        df: Predictions dataframe
        subgroups: Dictionary of subgroup filters

    Returns:
        DataFrame with lead time metrics
    """
    results = []

    for subgroup_name, subgroup_filter in subgroups.items():
        subgroup_df = df[subgroup_filter(df)]
        guid_data = subgroup_df.groupby('guid').agg({
            'first_detection_epoch': 'first',
            'binary_target': 'max'
        }).reset_index()

        unhealthy = guid_data[guid_data['binary_target'] == 1]
        detected = unhealthy[unhealthy['first_detection_epoch'].notna()]

        n_unhealthy = len(unhealthy)
        n_detected = len(detected)
        detection_rate = (n_detected / n_unhealthy * 100) if n_unhealthy > 0 else 0.0

        row = {
            'subgroup': subgroup_name,
            'n_unhealthy_guids': n_unhealthy,
            'n_detected': n_detected,
            'detection_rate_pct': detection_rate
        }

        if n_detected > 0:
            # Lead time = time before birth when first detected
            lead_time_hours = abs(detected['first_detection_epoch'].values) / 3600

            row.update({
                'mean_lead_time_hours': np.mean(lead_time_hours),
                'median_lead_time_hours': np.median(lead_time_hours),
                'std_lead_time_hours': np.std(lead_time_hours),
                'min_lead_time_hours': np.min(lead_time_hours),
                'max_lead_time_hours': np.max(lead_time_hours),
                'pct_detected_gt_1h': (lead_time_hours > 1).sum() / n_detected * 100,
                'pct_detected_gt_2h': (lead_time_hours > 2).sum() / n_detected * 100,
                'pct_detected_gt_6h': (lead_time_hours > 6).sum() / n_detected * 100
            })
        else:
            row.update({
                'mean_lead_time_hours': np.nan,
                'median_lead_time_hours': np.nan,
                'std_lead_time_hours': np.nan,
                'min_lead_time_hours': np.nan,
                'max_lead_time_hours': np.nan,
                'pct_detected_gt_1h': 0.0,
                'pct_detected_gt_2h': 0.0,
                'pct_detected_gt_6h': 0.0
            })

        results.append(row)

    return pd.DataFrame(results)


def plot_confusion_matrix_heatmap(
    ax: plt.Axes,
    cm: np.ndarray,
    cmap: str,
    xticklabels: List[str],
    yticklabels: List[str],
) -> None:
    """
    Plot a 2x2 confusion matrix, using seaborn if available.
    """
    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap=cmap,
            ax=ax,
            cbar=False,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
        )
        return

    im = ax.imshow(cm, cmap=plt.get_cmap(cmap))
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha='center', va='center', color='black', fontsize=10)
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(range(len(yticklabels)))
    ax.set_yticklabels(yticklabels)
    # Keep layout similar to seaborn
    ax.set_xlim(-0.5, len(xticklabels) - 0.5)
    ax.set_ylim(len(yticklabels) - 0.5, -0.5)


def plot_confusion_matrices_by_time(
    df: pd.DataFrame,
    output_dir: Path,
    prefix: str = "",
    exclude_last_minutes: float = 30.0
) -> None:
    """
    Plot confusion matrices for different time windows.

    Args:
        df: Predictions dataframe
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
        exclude_last_minutes: Exclude last N minutes before birth from plots (default: 30 min)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute dynamic time windows
    time_windows = compute_time_windows(df, exclude_last_minutes=exclude_last_minutes)

    # FIX #5: Ensure epoch_hours column exists
    df = ensure_epoch_hours(df.copy())

    # Create grid of confusion matrices
    n_windows = len(time_windows)
    n_cols = 3
    n_rows = (n_windows + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    # FIX #8: Simplified axes flattening - handles single plot, 1D array, and 2D array
    axes = np.atleast_1d(axes).flatten()

    for idx, (start_h, end_h) in enumerate(time_windows):
        window_mask = (df['epoch_hours'] >= start_h) & (df['epoch_hours'] < end_h)
        window_data = df[window_mask]

        if len(window_data) < 10:
            axes[idx].axis('off')
            continue

        # Compute confusion matrix
        # Force 2x2 matrix even if only one class present in window
        cm = confusion_matrix(window_data['binary_target'], window_data['clinical_pred'], labels=[0, 1])

        plot_confusion_matrix_heatmap(
            axes[idx],
            cm,
            cmap='Blues',
            xticklabels=['Healthy', 'Unhealthy'],
            yticklabels=['Healthy', 'Unhealthy'],
        )
        # Show negative time range: -end to -start hours before birth
        axes[idx].set_title(f'{-end_h:.1f} to {-start_h:.1f}h before birth')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    # Hide unused subplots
    for idx in range(len(time_windows), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}confusion_matrices_by_time.png", dpi=150)
    plt.close()

    logger.info(f"Saved: {prefix}confusion_matrices_by_time.png")


def plot_guid_level_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    prefix: str = ""
) -> None:
    """
    GUID-level performance analysis.

    Args:
        df: Predictions dataframe
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate to GUID level
    guid_data = df.groupby('guid').agg({
        'binary_target': 'max',      # GUID unhealthy if any epoch unhealthy
        'clinical_pred': 'max',       # GUID predicted unhealthy if any epoch predicted unhealthy
        'first_detection_epoch': 'first',
        'is_filled': 'sum'            # Count filled epochs
    }).reset_index()

    guid_data['n_epochs'] = df.groupby('guid').size().values

    # Plot 1: GUID-level confusion matrix
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Force 2x2 matrix even if only one class present
    cm = confusion_matrix(guid_data['binary_target'], guid_data['clinical_pred'], labels=[0, 1])
    plot_confusion_matrix_heatmap(
        axes[0, 0],
        cm,
        cmap='Blues',
        xticklabels=['Healthy', 'Unhealthy'],
        yticklabels=['Healthy', 'Unhealthy'],
    )
    axes[0, 0].set_title('GUID-Level Classification Matrix')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')

    # Plot 2: Time to detection for unhealthy GUIDs
    unhealthy = guid_data[guid_data['binary_target'] == 1].copy()
    detected = unhealthy[unhealthy['first_detection_epoch'].notna()]
    missed = unhealthy[unhealthy['first_detection_epoch'].isna()]

    # Negate to show negative hours before birth
    detection_hours = -(detected['first_detection_epoch'].values / 3600) if len(detected) > 0 else []

    if len(detection_hours) > 0:
        axes[0, 1].hist(detection_hours, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Time Before Birth (hours, 0 = delivery)', fontsize=10)
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title(f'Detection Time Distribution (n={len(detected)})')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No detections', ha='center', va='center')
        axes[0, 1].axis('off')

    # Plot 3: Epoch coverage distribution
    axes[1, 0].hist(guid_data['n_epochs'], bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Number of Epochs per GUID')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Epoch Coverage Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Filled vs. original epochs
    if 'is_filled' in guid_data.columns:
        guid_data['pct_filled'] = guid_data['is_filled'] / guid_data['n_epochs'] * 100
        axes[1, 1].hist(guid_data['pct_filled'], bins=20, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('% Filled Epochs')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Proportion of Filled Epochs per GUID')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}guid_level_analysis.png", dpi=150)
    plt.close()

    logger.info(f"Saved: {prefix}guid_level_analysis.png")


def plot_threshold_comparison(
    df_epoch: pd.DataFrame,
    df_guid: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Compare performance of epoch-level vs. GUID-level thresholds.

    Args:
        df_epoch: Predictions using epoch-level threshold
        df_guid: Predictions using GUID-level threshold
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Metrics comparison
    metrics_names = ['Sensitivity', 'Specificity', 'Accuracy']

    # Compute metrics for both
    epoch_metrics = []
    guid_metrics = []

    for df_data in [df_epoch, df_guid]:
        class_0 = df_data[df_data['binary_target'] == 0]
        class_1 = df_data[df_data['binary_target'] == 1]

        sens = (class_1['clinical_pred'] == 1).mean() if len(class_1) > 0 else 0
        spec = 1 - (class_0['clinical_pred'] == 1).mean() if len(class_0) > 0 else 0
        acc = (df_data['binary_target'] == df_data['clinical_pred']).mean()

        if df_data is df_epoch:
            epoch_metrics = [sens, spec, acc]
        else:
            guid_metrics = [sens, spec, acc]

    x = np.arange(len(metrics_names))
    width = 0.35

    axes[0, 0].bar(x - width/2, epoch_metrics, width, label='Epoch Threshold', alpha=0.8)
    axes[0, 0].bar(x + width/2, guid_metrics, width, label='GUID Threshold', alpha=0.8)
    axes[0, 0].set_ylabel('Metric Value')
    axes[0, 0].set_title('Metrics Comparison: Epoch vs. GUID Threshold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics_names)
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1.05])
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Plot 2: FPR comparison over time
    time_bins = compute_time_bins(df_epoch)
    # FIX #5: Ensure epoch_hours column exists
    df_epoch = ensure_epoch_hours(df_epoch)
    df_guid = ensure_epoch_hours(df_guid)

    epoch_fprs, guid_fprs, bin_centers = [], [], []

    for i in range(len(time_bins) - 1):
        bin_start, bin_end = time_bins[i], time_bins[i + 1]

        for df_data, fpr_list in [(df_epoch, epoch_fprs), (df_guid, guid_fprs)]:
            bin_mask = (df_data['epoch_hours'] >= bin_start) & (df_data['epoch_hours'] < bin_end)
            class_0 = df_data[bin_mask & (df_data['binary_target'] == 0)]

            if len(class_0) > 0:
                fpr = (class_0['clinical_pred'] == 1).mean()
                fpr_list.append(fpr)
            else:
                fpr_list.append(np.nan)

        bin_centers.append((bin_start + bin_end) / 2)

    axes[0, 1].plot(bin_centers, epoch_fprs, marker='o', label='Epoch Threshold', linewidth=2)
    axes[0, 1].plot(bin_centers, guid_fprs, marker='s', label='GUID Threshold', linewidth=2)
    axes[0, 1].set_xlabel('Time Before Birth (hours)')
    axes[0, 1].set_ylabel('False Positive Rate')
    axes[0, 1].set_title('FPR Comparison Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Detection count comparison
    epoch_counts, guid_counts = [], []

    for i in range(len(time_bins) - 1):
        bin_start, bin_end = time_bins[i], time_bins[i + 1]

        for df_data, count_list in [(df_epoch, epoch_counts), (df_guid, guid_counts)]:
            bin_mask = (df_data['epoch_hours'] >= bin_start) & (df_data['epoch_hours'] < bin_end)
            count = (df_data[bin_mask]['clinical_pred'] == 1).sum()
            count_list.append(count)

    axes[1, 0].plot(bin_centers, epoch_counts, marker='o', label='Epoch Threshold', linewidth=2)
    axes[1, 0].plot(bin_centers, guid_counts, marker='s', label='GUID Threshold', linewidth=2)
    axes[1, 0].set_xlabel('Time Before Birth (hours)')
    axes[1, 0].set_ylabel('Number of Positive Predictions')
    axes[1, 0].set_title('Detection Count Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Agreement analysis
    # CRITICAL FIX: Use explicit merge to ensure alignment (don't assume row order matches)
    agreement_df = pd.merge(
        df_epoch[['guid', 'epoch', 'clinical_pred']].rename(columns={'clinical_pred': 'epoch_pred'}),
        df_guid[['guid', 'epoch', 'clinical_pred']].rename(columns={'clinical_pred': 'guid_pred'}),
        on=['guid', 'epoch'],
        how='inner',
        validate='1:1'
    )

    logger.debug(f"Threshold comparison: {len(agreement_df)} matched samples")

    # Force 2x2 matrix even if only one class present
    cm = confusion_matrix(agreement_df['epoch_pred'], agreement_df['guid_pred'], labels=[0, 1])
    plot_confusion_matrix_heatmap(
        axes[1, 1],
        cm,
        cmap='Greens',
        xticklabels=['GUID: Neg', 'GUID: Pos'],
        yticklabels=['Epoch: Neg', 'Epoch: Pos'],
    )
    axes[1, 1].set_title('Agreement Between Thresholds')
    axes[1, 1].set_ylabel('Epoch Threshold')
    axes[1, 1].set_xlabel('GUID Threshold')

    plt.tight_layout()
    plt.savefig(output_dir / "threshold_comparison.png", dpi=150)
    plt.close()

    logger.info("Saved: threshold_comparison.png")


def plot_subgroup_metrics_vs_time(
    df: pd.DataFrame,
    subgroups: Dict[str, callable],
    time_bins: np.ndarray,
    output_dir: Path,
    prefix: str = "",
    guid_level: bool = False
) -> None:
    """
    Plot sensitivity/specificity for multiple subgroups on same figure.

    Creates two separate plots:
    - Sensitivity comparison across subgroups
    - Specificity comparison across subgroups

    Args:
        df: Predictions dataframe
        subgroups: Dictionary mapping subgroup_name -> filter_function
        time_bins: Time bin edges in hours
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
        guid_level: If True, plot GUID-level cumulative sensitivity (monotonically increasing)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # FIX #5: Ensure epoch_hours column exists
    df = ensure_epoch_hours(df.copy())

    # Compute metrics for each subgroup
    subgroup_results = {}
    empty_subgroups = []
    for subgroup_name, subgroup_filter in subgroups.items():
        metrics_df = compute_subgroup_metrics_by_time(df, subgroup_name, subgroup_filter, time_bins, guid_level=guid_level)
        if len(metrics_df) > 0:
            subgroup_results[subgroup_name] = metrics_df
        else:
            empty_subgroups.append(subgroup_name)

    if len(subgroup_results) == 0:
        logger.warning(f"No subgroup data available for plotting {prefix}. All subgroups empty: {', '.join(empty_subgroups)}")
        return

    if empty_subgroups:
        logger.info(f"Plotting {len(subgroup_results)} subgroups (skipped {len(empty_subgroups)}: {', '.join(empty_subgroups)})")

    # Create figure with 2 subplots: sensitivity and specificity
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(subgroup_results)))
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X']

    for idx, (subgroup_name, metrics_df) in enumerate(subgroup_results.items()):
        color = colors[idx]
        marker = markers[idx % len(markers)]

        # Negate bin_center to show negative hours (before delivery) on left → 0 (delivery) on right
        x = -metrics_df['bin_center'].values

        # Use GUID-level or epoch-level metrics depending on mode
        if guid_level and 'guid_sensitivity' in metrics_df.columns:
            sens_values = metrics_df['guid_sensitivity'].values
            spec_values = metrics_df['guid_specificity'].values
        else:
            sens_values = metrics_df['sensitivity'].values
            spec_values = metrics_df['specificity'].values

        # Sensitivity subplot (skip if all NaN)
        if not np.all(np.isnan(sens_values)):
            axes[0].plot(x, sens_values,
                        marker=marker, label=subgroup_name.replace('_', ' ').title(),
                        linewidth=2, markersize=6, color=color)

        # Specificity subplot (skip if all NaN)
        if not np.all(np.isnan(spec_values)):
            axes[1].plot(x, spec_values,
                        marker=marker, label=subgroup_name.replace('_', ' ').title(),
                        linewidth=2, markersize=6, color=color)

    # Format sensitivity subplot
    metric_type = "GUID-Level Cumulative" if guid_level else "Epoch-Level"
    axes[0].set_xlabel('Time Before Birth (hours, 0 = delivery)', fontsize=12)
    axes[0].set_ylabel(f'Sensitivity (TPR) - {metric_type}', fontsize=12)
    axes[0].set_title(f'Sensitivity by Subgroup ({metric_type})', fontsize=14)
    axes[0].legend(fontsize=10, loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # Format specificity subplot
    axes[1].set_xlabel('Time Before Birth (hours, 0 = delivery)', fontsize=12)
    axes[1].set_ylabel(f'Specificity (TNR) - {metric_type}', fontsize=12)
    axes[1].set_title(f'Specificity by Subgroup ({metric_type})', fontsize=14)
    axes[1].legend(fontsize=10, loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    plt.tight_layout()
    filename_suffix = "guid_level" if guid_level else "epoch_level"
    plt.savefig(output_dir / f"{prefix}subgroup_metrics_vs_time_{filename_suffix}.png", dpi=150)
    plt.close()

    logger.info(f"Saved: {prefix}subgroup_metrics_vs_time_{filename_suffix}.png")


def plot_subgroup_roc_curves(
    df: pd.DataFrame,
    subgroups: Dict[str, callable],
    output_dir: Path,
    prefix: str = ""
) -> None:
    """
    Plot ROC curves comparing different subgroups.

    Args:
        df: Predictions dataframe
        subgroups: Dictionary mapping subgroup_name -> filter_function
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(subgroups)))

    for idx, (subgroup_name, subgroup_filter) in enumerate(subgroups.items()):
        subgroup_df = df[subgroup_filter(df)]

        if len(subgroup_df) < 10 or subgroup_df['binary_target'].nunique() < 2:
            logger.warning(f"Insufficient data for ROC curve in subgroup: {subgroup_name}")
            continue

        # Compute ROC
        fpr, tpr, _ = roc_curve(subgroup_df['binary_target'], subgroup_df['prob_class_1'])
        roc_auc = auc(fpr, tpr)

        label = f"{subgroup_name.replace('_', ' ').title()} (AUC={roc_auc:.3f})"
        ax.plot(fpr, tpr, label=label, linewidth=2, color=colors[idx])

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.500)')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves by Subgroup', fontsize=14)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}subgroup_roc_curves.png", dpi=150)
    plt.close()

    logger.info(f"Saved: {prefix}subgroup_roc_curves.png")


def plot_enhanced_subgroup_comparison(
    df: pd.DataFrame,
    subgroups: Dict[str, callable],
    time_bins: np.ndarray,
    output_dir: Path,
    prefix: str = "",
    metric: str = "both"
) -> None:
    """
    Enhanced subgroup comparison with side-by-side epoch and GUID-level views.

    Creates 2x2 grid comparing epoch-level vs GUID-level metrics for subgroups.

    Args:
        df: Predictions dataframe
        subgroups: Dictionary mapping subgroup_name -> filter_function
        time_bins: Time bin edges in hours
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
        metric: "sensitivity", "specificity", or "both"
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # FIX #5: Ensure epoch_hours column exists
    df = ensure_epoch_hours(df.copy())

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green
    markers = ['o', 's', '^']

    # Compute metrics for each subgroup (both epoch and GUID level)
    for idx, (subgroup_name, subgroup_filter) in enumerate(subgroups.items()):
        # Epoch-level metrics
        epoch_metrics = compute_subgroup_metrics_by_time(
            df, subgroup_name, subgroup_filter, time_bins, guid_level=False
        )

        # GUID-level metrics
        guid_metrics = compute_subgroup_metrics_by_time(
            df, subgroup_name, subgroup_filter, time_bins, guid_level=True
        )

        if len(epoch_metrics) == 0:
            continue

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        label = subgroup_name.replace('_', ' ').title()

        # Negate bin_center to show negative hours before birth
        x_epoch = -epoch_metrics['bin_center'].values
        x_guid = -guid_metrics['bin_center'].values if len(guid_metrics) > 0 else []

        # Top-left: Epoch-level Sensitivity
        if 'sensitivity' in epoch_metrics.columns and not np.all(np.isnan(epoch_metrics['sensitivity'])):
            axes[0, 0].plot(x_epoch, epoch_metrics['sensitivity'].values,
                           marker=marker, label=label, linewidth=2, markersize=6, color=color)

        # Top-right: GUID-level Cumulative Sensitivity
        if len(guid_metrics) > 0 and 'guid_sensitivity' in guid_metrics.columns:
            axes[0, 1].plot(x_guid, guid_metrics['guid_sensitivity'].values,
                           marker=marker, label=label, linewidth=2, markersize=6, color=color)

        # Bottom-left: Epoch-level Specificity
        if 'specificity' in epoch_metrics.columns and not np.all(np.isnan(epoch_metrics['specificity'])):
            axes[1, 0].plot(x_epoch, epoch_metrics['specificity'].values,
                           marker=marker, label=label, linewidth=2, markersize=6, color=color)

        # Bottom-right: GUID-level Cumulative Specificity
        if len(guid_metrics) > 0 and 'guid_specificity' in guid_metrics.columns:
            axes[1, 1].plot(x_guid, guid_metrics['guid_specificity'].values,
                           marker=marker, label=label, linewidth=2, markersize=6, color=color)

    # Format subplots
    for ax, title in zip(axes.flat, [
        'Epoch-Level Sensitivity',
        'GUID-Level Cumulative Sensitivity (Monotonic)',
        'Epoch-Level Specificity',
        'GUID-Level Cumulative Specificity'
    ]):
        ax.set_xlabel('Time Before Birth (hours, 0 = delivery)', fontsize=11)
        ax.set_ylabel('Metric Value', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}enhanced_sensitivity_specificity.png", dpi=150)
    plt.close()

    logger.info(f"Saved: {prefix}enhanced_sensitivity_specificity.png")


def compute_subgroup_statistics_summary(
    df: pd.DataFrame,
    subgroups: Dict[str, callable],
    time_bins: np.ndarray
) -> pd.DataFrame:
    """
    Compute comprehensive statistics for each subgroup.

    Returns DataFrame with overall sensitivity/specificity and detection rates
    at different time windows.

    Args:
        df: Predictions dataframe
        subgroups: Dictionary mapping subgroup_name -> filter_function
        time_bins: Time bin edges in hours

    Returns:
        DataFrame with subgroup statistics
    """
    # FIX #5: Ensure epoch_hours column exists
    df = ensure_epoch_hours(df.copy())

    results = []

    for subgroup_name, subgroup_filter in subgroups.items():
        subgroup_df = df[subgroup_filter(df)]

        if len(subgroup_df) == 0:
            continue

        # GUID-level statistics
        guid_data = subgroup_df.groupby('guid').agg({
            'binary_target': 'max',
            'clinical_pred': 'max',
            'first_detection_epoch': 'first'
        }).reset_index()

        n_total_guids = len(guid_data)
        unhealthy_guids = guid_data[guid_data['binary_target'] == 1]
        healthy_guids = guid_data[guid_data['binary_target'] == 0]

        n_unhealthy = len(unhealthy_guids)
        n_healthy = len(healthy_guids)

        # Overall GUID-level metrics
        if n_unhealthy > 0:
            overall_sensitivity = (unhealthy_guids['clinical_pred'] == 1).sum() / n_unhealthy
        else:
            overall_sensitivity = np.nan

        if n_healthy > 0:
            overall_specificity = 1 - (healthy_guids['clinical_pred'] == 1).sum() / n_healthy
        else:
            overall_specificity = np.nan

        # Detection rates at key time windows
        detected = unhealthy_guids[unhealthy_guids['first_detection_epoch'].notna()]
        if len(detected) > 0:
            detection_hours = abs(detected['first_detection_epoch'].values) / 3600
            detection_rate_1h = (detection_hours <= 1).sum() / len(unhealthy_guids) * 100
            detection_rate_2h = (detection_hours <= 2).sum() / len(unhealthy_guids) * 100
            detection_rate_6h = (detection_hours <= 6).sum() / len(unhealthy_guids) * 100
            mean_detection_time = np.mean(detection_hours)
            median_detection_time = np.median(detection_hours)
        else:
            detection_rate_1h = 0.0
            detection_rate_2h = 0.0
            detection_rate_6h = 0.0
            mean_detection_time = np.nan
            median_detection_time = np.nan

        row = {
            'subgroup_name': subgroup_name,
            'n_total_guids': n_total_guids,
            'n_unhealthy_guids': n_unhealthy,
            'n_healthy_guids': n_healthy,
            'overall_sensitivity': overall_sensitivity,
            'overall_specificity': overall_specificity,
            'detection_rate_1h': detection_rate_1h,
            'detection_rate_2h': detection_rate_2h,
            'detection_rate_6h': detection_rate_6h,
            'mean_detection_time': mean_detection_time,
            'median_detection_time': median_detection_time
        }

        results.append(row)

    return pd.DataFrame(results)


def plot_subgroup_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    prefix: str = ""
) -> None:
    """
    Visualize sample distribution across subgroups.

    Shows:
    - Target distribution (Healthy, Acidosis, HIE)
    - CS status distribution
    - Combined subgroup counts
    - Prevalence over time

    Args:
        df: Predictions dataframe
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Count unique GUIDs per subgroup
    guid_stats = df.groupby('guid').agg({
        'target': 'max',
        'cs_label': 'first',
        'binary_target': 'max'
    }).reset_index()

    # Plot 1: Target distribution
    target_counts = guid_stats['target'].value_counts().sort_index()
    target_labels = {1: 'Healthy', 2: 'Acidosis', 3: 'HIE'}
    axes[0, 0].bar([target_labels.get(k, str(k)) for k in target_counts.index],
                   target_counts.values, alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('Number of GUIDs', fontsize=12)
    axes[0, 0].set_title('Distribution by Diagnosis', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Plot 2: CS status distribution
    cs_counts = guid_stats['cs_label'].value_counts()
    cs_labels = {True: 'CS+', False: 'CS-'}
    axes[0, 1].bar([cs_labels.get(k, 'Unknown') for k in cs_counts.index],
                   cs_counts.values, alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('Number of GUIDs', fontsize=12)
    axes[0, 1].set_title('Distribution by CS Status', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Combined subgroup distribution (Diagnosis × CS)
    combined = guid_stats.groupby(['target', 'cs_label']).size().reset_index(name='count')
    combined['label'] = combined.apply(
        lambda row: f"{target_labels.get(row['target'], str(row['target']))} - {cs_labels.get(row['cs_label'], 'Unknown')}",
        axis=1
    )
    axes[1, 0].barh(combined['label'], combined['count'], alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Number of GUIDs', fontsize=12)
    axes[1, 0].set_title('Combined Subgroup Distribution', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # Plot 4: Prevalence by time (using epoch_hours bins)
    # FIX #5: Ensure epoch_hours column exists
    df_copy = ensure_epoch_hours(df.copy())

    time_bins = compute_time_bins(df_copy)
    bin_centers = []
    acidosis_prev = []
    hie_prev = []

    for i in range(len(time_bins) - 1):
        bin_start, bin_end = time_bins[i], time_bins[i + 1]
        bin_mask = (df_copy['epoch_hours'] >= bin_start) & (df_copy['epoch_hours'] < bin_end)
        bin_data = df_copy[bin_mask]

        if len(bin_data) > 0:
            bin_centers.append((bin_start + bin_end) / 2)
            acidosis_prev.append((bin_data['target'] == 2).mean() * 100)
            hie_prev.append((bin_data['target'] == 3).mean() * 100)

    if len(bin_centers) > 0:
        # Negate bin_centers to show negative hours (before delivery) on left → 0 (delivery) on right
        x = [-bc for bc in bin_centers]
        axes[1, 1].plot(x, acidosis_prev, marker='o', label='Acidosis', linewidth=2)
        axes[1, 1].plot(x, hie_prev, marker='s', label='HIE', linewidth=2)
        axes[1, 1].set_xlabel('Time Before Birth (hours, 0 = delivery)', fontsize=12)
        axes[1, 1].set_ylabel('Prevalence (%)', fontsize=12)
        axes[1, 1].set_title('Diagnosis Prevalence Over Time', fontsize=14)
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No time data available', ha='center', va='center')
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}subgroup_distribution.png", dpi=150)
    plt.close()

    logger.info(f"Saved: {prefix}subgroup_distribution.png")


def save_subgroup_metrics(
    df: pd.DataFrame,
    output_dir: Path,
    prefix: str = ""
) -> None:
    """
    Compute and save subgroup metrics to CSV files.

    Args:
        df: Predictions dataframe
        output_dir: Directory to save metrics
        prefix: Prefix for output filenames
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # FIX #5: Ensure epoch_hours column exists
    df = ensure_epoch_hours(df.copy())

    # Get all subgroup filters including overall comparison
    subgroup_filters = create_subgroup_filters()
    subgroup_filters['unhealthy'] = lambda df: df['binary_target'] == 1

    time_bins = compute_time_bins(df)

    all_subgroup_metrics = []
    saved_subgroups = []
    skipped_subgroups = []

    for subgroup_name, subgroup_filter in subgroup_filters.items():
        metrics_df = compute_subgroup_metrics_by_time(df, subgroup_name, subgroup_filter, time_bins)
        if len(metrics_df) > 0:
            all_subgroup_metrics.append(metrics_df)
            saved_subgroups.append(subgroup_name)
        else:
            skipped_subgroups.append(subgroup_name)

    if all_subgroup_metrics:
        combined_df = pd.concat(all_subgroup_metrics, ignore_index=True)
        csv_path = output_dir / f"{prefix}subgroup_metrics.csv"
        combined_df.to_csv(csv_path, index=False)
        logger.info(f"Saved subgroup metrics: {csv_path}")
        logger.info(f"  Saved subgroups ({len(saved_subgroups)}): {', '.join(saved_subgroups)}")
        if skipped_subgroups:
            logger.info(f"  Skipped subgroups ({len(skipped_subgroups)}): {', '.join(skipped_subgroups)}")
    else:
        logger.warning(f"No subgroup metrics to save. All subgroups were empty: {', '.join(skipped_subgroups)}")


def evaluate_fold(
    model: torch.nn.Module,
    fold_dir: str,
    config_path: str,
    checkpoint_path: str,
    target_fpr: float = 0.05,
    device: str = 'cuda:0'
) -> Dict:
    """
    Evaluate a single fold after training.

    Args:
        model: PyTorch model to evaluate
        fold_dir: Fold output directory
        config_path: Path to fold config
        checkpoint_path: Path to best model checkpoint
        target_fpr: Target false positive rate for threshold selection
        device: Device to run on

    Returns:
        Dictionary with evaluation results
    """
    fold_dir = Path(fold_dir)

    logger.info(f"Evaluating fold: {fold_dir}")
    logger.info(f"Using checkpoint: {checkpoint_path}")

    # Load config for dataset settings
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    eval_cfg = (
        config.get('model_config', {})
        .get('classifier', {})
        .get('evaluation', {})
    ) or {}
    decision_time_hours = float(eval_cfg.get('decision_time_hours', 1.0))
    exclude_last_minutes = float(eval_cfg.get('exclude_last_minutes', 30.0))
    max_gap_multiplier = eval_cfg.get('max_gap_multiplier', None)

    # Load checkpoint using utility function
    model = load_checkpoint_strict(model, checkpoint_path, map_location=device)
    if model is None:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}")

    model.eval()
    logger.info(f"Model loaded successfully from: {checkpoint_path}")

    # Get dataset config
    dataset_config = config['dataset_config']
    dataloader_config = dataset_config['dataloader_config']
    dataset_kwargs = dataloader_config.get('dataset_kwargs', {})
    normalized_fields = dataloader_config.get('normalize_fields')
    stat_path = dataset_config.get('stat_path')
    batch_size = config['general_config']['batch_size']['test']

    # Create validation dataloader
    logger.info("Running validation inference...")
    val_dataloader = create_optimized_dataloader(
        hdf5_files=dataset_config['classifier_val_datasets'],
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        stats_path=stat_path,
        normalize_fields=normalized_fields,
        pin_memory=False,
        rank=0,
        world_size=1,
        **dataset_kwargs
    )

    # Run validation inference
    val_df_raw = run_inference(model, val_dataloader, device)

    # FIX #9: Validate predictions DataFrame
    validate_predictions_df(val_df_raw, "Validation")
    validate_guid_consistency(val_df_raw)
    log_dataframe_stats(val_df_raw, "Validation Raw")

    # Save raw validation predictions
    val_output_dir = fold_dir / "evaluation"
    val_output_dir.mkdir(parents=True, exist_ok=True)
    val_df_raw.to_csv(val_output_dir / "validation_predictions_raw.csv", index=False)
    logger.info("Validation raw predictions saved")

    # Find optimal thresholds using three approaches (epoch-level, GUID-level, time-specific)
    logger.info("=" * 80)
    logger.info("THRESHOLD DETERMINATION")
    logger.info("=" * 80)
    threshold_results = find_optimal_thresholds(
        val_df_raw,
        target_fpr=target_fpr,
        time_window_hours=decision_time_hours,
        max_gap_multiplier=max_gap_multiplier
    )

    # Save threshold information
    with open(val_output_dir / "validation_threshold_info.json", 'w') as f:
        json.dump(threshold_results, f, indent=2)

    # Apply clinical decision rule and fill missing epochs for validation
    # Using TIME-SPECIFIC threshold (optimized for 1h before delivery)
    time_threshold = threshold_results['time_specific']['threshold']
    logger.info(f"Using time-specific threshold: {time_threshold:.4f} (optimized for {decision_time_hours}h before delivery)")
    val_df_clinical = apply_clinical_decision_rule(val_df_raw.copy(), time_threshold)
    val_df_clinical = fill_missing_epochs(val_df_clinical, max_gap_multiplier=max_gap_multiplier)

    # Verify forward-filling after epoch filling
    from model.vae_teb_prediction.validation_utils import verify_clinical_decision_rule
    verify_clinical_decision_rule(val_df_clinical, "Validation (post-filling)")

    # Save clinical validation predictions
    val_df_clinical.to_csv(val_output_dir / "validation_predictions_clinical.csv", index=False)
    logger.info("Validation clinical predictions saved")

    # Create test dataloader
    logger.info("=" * 80)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 80)
    test_dataloader = create_optimized_dataloader(
        hdf5_files=dataset_config['classifier_test_datasets'],
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        stats_path=stat_path,
        normalize_fields=normalized_fields,
        pin_memory=False,
        rank=0,
        world_size=1,
        **dataset_kwargs
    )

    # Run test inference
    test_df_raw = run_inference(model, test_dataloader, device)

    # FIX #9: Validate predictions DataFrame
    validate_predictions_df(test_df_raw, "Test")
    validate_guid_consistency(test_df_raw)
    log_dataframe_stats(test_df_raw, "Test Raw")

    # Save raw test predictions
    test_df_raw.to_csv(val_output_dir / "test_predictions_raw.csv", index=False)
    logger.info("Test raw predictions saved")

    # Process test set with all three thresholds (time_specific is primary/recommended)
    test_results = {}

    for threshold_type in ['time_specific', 'guid_level', 'epoch_level']:
        is_primary = (threshold_type == 'time_specific')
        marker = " [PRIMARY - Clinically Optimized]" if is_primary else ""
        logger.info(f"\nProcessing test set with {threshold_type} threshold...{marker}")

        threshold_value = threshold_results[threshold_type]['threshold']

        # Apply clinical decision rule
        test_df = apply_clinical_decision_rule(test_df_raw.copy(), threshold_value)
        # FIX #9: Log statistics after clinical decision rule
        log_dataframe_stats(test_df, f"Test Clinical ({threshold_type})")

        # Fill missing epochs
        test_df = fill_missing_epochs(test_df, max_gap_multiplier=max_gap_multiplier)
        # FIX #9: Log statistics after filling
        log_dataframe_stats(test_df, f"Test Filled ({threshold_type})")

        # Verify forward-filling after epoch filling
        verify_clinical_decision_rule(test_df, f"Test {threshold_type} (post-filling)")

        # Save predictions
        test_df.to_csv(val_output_dir / f"test_predictions_clinical_{threshold_type}.csv", index=False)

        # Compute test metrics
        test_metrics = {}

        # Epoch-level metrics
        class_0 = test_df[test_df['binary_target'] == 0]
        class_1 = test_df[test_df['binary_target'] == 1]

        test_metrics['epoch_metrics'] = {
            'accuracy': (test_df['binary_target'] == test_df['clinical_pred']).mean(),
            'sensitivity': (class_1['clinical_pred'] == 1).mean() if len(class_1) > 0 else 1.0,
            'specificity': 1 - (class_0['clinical_pred'] == 1).mean() if len(class_0) > 0 else 1.0,
            'fpr': (class_0['clinical_pred'] == 1).mean() if len(class_0) > 0 else 0.0,
            'n_class_0': len(class_0),
            'n_class_1': len(class_1),
        }

        # GUID-level metrics
        guid_stats = test_df.groupby('guid').agg({
            'binary_target': 'max',
            'clinical_pred': 'max'
        }).reset_index()

        healthy_guids = guid_stats[guid_stats['binary_target'] == 0]
        unhealthy_guids = guid_stats[guid_stats['binary_target'] == 1]

        test_metrics['guid_metrics'] = {
            'accuracy': (guid_stats['binary_target'] == guid_stats['clinical_pred']).mean(),
            'sensitivity': (unhealthy_guids['clinical_pred'] == 1).sum() / len(unhealthy_guids) if len(unhealthy_guids) > 0 else 1.0,
            'specificity': 1 - (healthy_guids['clinical_pred'] == 1).sum() / len(healthy_guids) if len(healthy_guids) > 0 else 1.0,
            'fpr': (healthy_guids['clinical_pred'] == 1).sum() / len(healthy_guids) if len(healthy_guids) > 0 else 0.0,
            'n_healthy_guids': len(healthy_guids),
            'n_unhealthy_guids': len(unhealthy_guids),
        }

        test_metrics['threshold'] = threshold_value
        test_results[threshold_type] = test_metrics

        # Save metrics
        with open(val_output_dir / f"test_metrics_{threshold_type}.json", 'w') as f:
            json.dump(test_metrics, f, indent=2)

        logger.info(f"{threshold_type.upper()} THRESHOLD RESULTS:")
        logger.info(f"  Threshold: {threshold_value:.4f}")
        logger.info(f"  Epoch-level - Acc: {test_metrics['epoch_metrics']['accuracy']:.4f}, "
                   f"Sens: {test_metrics['epoch_metrics']['sensitivity']:.4f}, "
                   f"Spec: {test_metrics['epoch_metrics']['specificity']:.4f}")
        logger.info(f"  GUID-level  - Acc: {test_metrics['guid_metrics']['accuracy']:.4f}, "
                   f"Sens: {test_metrics['guid_metrics']['sensitivity']:.4f}, "
                   f"Spec: {test_metrics['guid_metrics']['specificity']:.4f}")

    # Save comparison metrics
    with open(val_output_dir / "test_metrics_comparison.json", 'w') as f:
        json.dump(test_results, f, indent=2)

    # Generate visualizations for all three thresholds
    logger.info("=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)

    for threshold_type in ['time_specific', 'guid_level', 'epoch_level']:
        threshold_value = threshold_results[threshold_type]['threshold']

        # Apply clinical rule for visualization
        test_df = apply_clinical_decision_rule(test_df_raw.copy(), threshold_value)
        test_df = fill_missing_epochs(test_df, max_gap_multiplier=max_gap_multiplier)

        # Diagnostic: Log target distribution
        if 'target' in test_df.columns:
            target_counts = test_df['target'].value_counts().sort_index()
            logger.info(f"{threshold_type} - Target distribution in test set:")
            for target_val, count in target_counts.items():
                target_name = {1: 'Healthy', 2: 'Acidosis', 3: 'HIE'}.get(target_val, f'Unknown({target_val})')
                logger.info(f"  {target_name} (target={target_val}): {count} epochs")

            # Log GUID-level distribution
            guid_targets = test_df.groupby('guid')['target'].first().value_counts().sort_index()
            logger.info(f"{threshold_type} - Target distribution (GUID-level):")
            for target_val, count in guid_targets.items():
                target_name = {1: 'Healthy', 2: 'Acidosis', 3: 'HIE'}.get(target_val, f'Unknown({target_val})')
                logger.info(f"  {target_name} (target={target_val}): {count} GUIDs")

        # Create output directory for this threshold
        plots_dir = val_output_dir / "plots" / threshold_type
        plots_dir.mkdir(parents=True, exist_ok=True)

        prefix = ""

        # Generate all plots
        try:
            plot_metrics_vs_time(test_df, plots_dir, prefix, exclude_last_minutes=exclude_last_minutes)
            plot_roc_curves_by_time(test_df, plots_dir, prefix, exclude_last_minutes=exclude_last_minutes)
            plot_confusion_matrices_by_time(test_df, plots_dir, prefix, exclude_last_minutes=exclude_last_minutes)
            plot_guid_level_analysis(test_df, plots_dir, prefix)

            # Original detection timing plot (keep for compatibility)
            plot_detection_timing(test_df, plots_dir, prefix)
        except Exception as e:
            logger.warning(f"Error generating plots for {threshold_type}: {e}")

        # Generate subgroup analysis
        logger.info(f"Generating subgroup analysis for {threshold_type}...")

        # Create subgroup filters
        subgroup_filters = create_subgroup_filters()

        # Enhanced detection timing analysis
        try:
            logger.info("Generating enhanced detection timing analysis...")

            # Comprehensive detection timing statistics
            detection_stats = analyze_detection_timing_statistics(
                test_df,
                subgroups=subgroup_filters,
                output_dir=plots_dir,
                prefix=prefix
            )

            # Save detection statistics as JSON
            with open(plots_dir / f"{prefix}detection_timing_statistics.json", 'w') as f:
                json.dump(detection_stats, f, indent=2)

            # Enhanced plots with subgroup comparison (Acidosis vs HIE)
            timing_subgroups = {
                'acidosis': subgroup_filters['acidosis'],
                'hie': subgroup_filters['hie']
            }
            plot_detection_timing_enhanced(test_df, timing_subgroups, plots_dir, prefix)

            # Lead time metrics table
            lead_time_subgroups = {
                'acidosis': subgroup_filters['acidosis'],
                'hie': subgroup_filters['hie'],
                'healthy': subgroup_filters['healthy']
            }
            lead_time_table = create_lead_time_table(test_df, lead_time_subgroups)
            lead_time_table.to_csv(plots_dir / f"{prefix}lead_time_metrics.csv", index=False)

            logger.info(f"Detection timing analysis complete")
        except Exception as e:
            logger.warning(f"Error in enhanced detection timing analysis: {e}")

        # Create subgroup output directory
        subgroup_dir = plots_dir / "subgroup_analysis"
        subgroup_dir.mkdir(parents=True, exist_ok=True)

        # Compute time bins for subgroup analysis (excluding last 30 minutes before birth)
        time_bins = compute_time_bins(test_df, exclude_last_minutes=exclude_last_minutes)
        logger.info(f"Time bins computed (excluding last {exclude_last_minutes:.0f}min): {len(time_bins)-1} bins from {time_bins[0]:.2f}h to {time_bins[-1]:.2f}h")

        # Plot diagnosis comparison (Acidosis vs HIE) - Both epoch and GUID level
        try:
            diagnosis_subgroups = {
                'acidosis': subgroup_filters['acidosis'],
                'hie': subgroup_filters['hie']
            }
            # Epoch-level metrics
            plot_subgroup_metrics_vs_time(
                test_df, diagnosis_subgroups,
                time_bins,
                subgroup_dir, prefix="diagnosis_", guid_level=False
            )
            # GUID-level cumulative metrics (monotonically increasing)
            plot_subgroup_metrics_vs_time(
                test_df, diagnosis_subgroups,
                time_bins,
                subgroup_dir, prefix="diagnosis_", guid_level=True
            )
            plot_subgroup_roc_curves(
                test_df, diagnosis_subgroups,
                subgroup_dir, prefix="diagnosis_"
            )

            # Enhanced subgroup comparison (Epoch vs GUID level side-by-side)
            plot_enhanced_subgroup_comparison(
                test_df, diagnosis_subgroups, time_bins,
                subgroup_dir, prefix="diagnosis_", metric="both"
            )

            # Compute and save subgroup statistics summary
            stats_df = compute_subgroup_statistics_summary(
                test_df, diagnosis_subgroups, time_bins
            )
            stats_df.to_csv(subgroup_dir / "diagnosis_statistics.csv", index=False)
            logger.info(f"Saved diagnosis subgroup statistics: {len(stats_df)} subgroups")
        except Exception as e:
            logger.warning(f"Error generating diagnosis subgroup plots: {e}")

        # Plot CS status comparison - Both epoch and GUID level
        try:
            cs_subgroups = {
                'cs_positive': subgroup_filters['cs_positive'],
                'cs_negative': subgroup_filters['cs_negative']
            }
            # Epoch-level metrics
            plot_subgroup_metrics_vs_time(
                test_df, cs_subgroups,
                time_bins,
                subgroup_dir, prefix="cs_status_", guid_level=False
            )
            # GUID-level cumulative metrics
            plot_subgroup_metrics_vs_time(
                test_df, cs_subgroups,
                time_bins,
                subgroup_dir, prefix="cs_status_", guid_level=True
            )
            plot_subgroup_roc_curves(
                test_df, cs_subgroups,
                subgroup_dir, prefix="cs_status_"
            )
        except Exception as e:
            logger.warning(f"Error generating CS status subgroup plots: {e}")

        # Plot BG label comparison - Both epoch and GUID level
        try:
            bg_subgroups = {
                'bg_positive': subgroup_filters['bg_positive'],
                'bg_negative': subgroup_filters['bg_negative']
            }
            plot_subgroup_metrics_vs_time(
                test_df, bg_subgroups,
                time_bins,
                subgroup_dir, prefix="bg_label_", guid_level=False
            )
            plot_subgroup_metrics_vs_time(
                test_df, bg_subgroups,
                time_bins,
                subgroup_dir, prefix="bg_label_", guid_level=True
            )
            plot_subgroup_roc_curves(
                test_df, bg_subgroups,
                subgroup_dir, prefix="bg_label_"
            )
        except Exception as e:
            logger.warning(f"Error generating BG label subgroup plots: {e}")

        # Plot combined subgroups - Both epoch and GUID level
        try:
            combined_subgroups = {
                'acidosis_cs_pos': subgroup_filters['acidosis_cs_pos'],
                'acidosis_cs_neg': subgroup_filters['acidosis_cs_neg'],
                'hie_cs_pos': subgroup_filters['hie_cs_pos'],
                'hie_cs_neg': subgroup_filters['hie_cs_neg']
            }
            plot_subgroup_metrics_vs_time(
                test_df, combined_subgroups,
                time_bins,
                subgroup_dir, prefix="combined_", guid_level=False
            )
            plot_subgroup_metrics_vs_time(
                test_df, combined_subgroups,
                time_bins,
                subgroup_dir, prefix="combined_", guid_level=True
            )
        except Exception as e:
            logger.warning(f"Error generating combined subgroup plots: {e}")

        # Plot overall comparison (Healthy vs Unhealthy) - Both epoch and GUID level
        try:
            overall_subgroups = {
                'healthy': subgroup_filters['healthy'],
                'unhealthy': lambda df: df['binary_target'] == 1
            }
            # Epoch-level
            plot_subgroup_metrics_vs_time(
                test_df, overall_subgroups,
                time_bins,
                subgroup_dir, prefix="overall_", guid_level=False
            )
            # GUID-level (THIS IS THE KEY ONE - monotonically increasing)
            plot_subgroup_metrics_vs_time(
                test_df, overall_subgroups,
                time_bins,
                subgroup_dir, prefix="overall_", guid_level=True
            )
        except Exception as e:
            logger.warning(f"Error generating overall subgroup plots: {e}")

        # Plot sample distribution
        try:
            plot_subgroup_distribution(test_df, subgroup_dir, prefix="")
        except Exception as e:
            logger.warning(f"Error generating subgroup distribution plot: {e}")

        # Save subgroup metrics to CSV
        try:
            save_subgroup_metrics(test_df, subgroup_dir, prefix="")
        except Exception as e:
            logger.warning(f"Error saving subgroup metrics CSV: {e}")

        logger.info(f"Subgroup analysis complete for {threshold_type}")

    # Generate comparison plots
    try:
        # Load both processed test sets
        df_epoch = apply_clinical_decision_rule(test_df_raw.copy(), threshold_results['epoch_level']['threshold'])
        df_epoch = fill_missing_epochs(df_epoch)

        df_guid = apply_clinical_decision_rule(test_df_raw.copy(), threshold_results['guid_level']['threshold'])
        df_guid = fill_missing_epochs(df_guid)

        comparison_dir = val_output_dir / "plots" / "comparison"
        plot_threshold_comparison(df_epoch, df_guid, comparison_dir)
    except Exception as e:
        logger.warning(f"Error generating comparison plots: {e}")

    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)

    # Save processing metadata for reproducibility and debugging
    processing_metadata = {
        'fold_dir': str(fold_dir),
        'checkpoint_path': str(checkpoint_path),
        'device': device,
        'target_fpr': target_fpr,
        'validation_samples': len(val_df_raw),
        'test_samples': len(test_df_raw),
        'thresholds_used': {
            'epoch_level': threshold_results['epoch_level']['threshold'],
            'guid_level': threshold_results['guid_level']['threshold']
        },
        'evaluation_timestamp': datetime.now().isoformat()
    }

    metadata_path = val_output_dir / "processing_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(processing_metadata, f, indent=2)
    logger.info(f"Saved processing metadata: {metadata_path}")

    # ------------------------------------------------------------------
    # Compatibility helpers for older callers that expect a flat schema
    # ------------------------------------------------------------------
    preferred_threshold_type = 'guid_level'
    compat_threshold_type = preferred_threshold_type if threshold_results.get(preferred_threshold_type) else None
    if compat_threshold_type is None and threshold_results.get('epoch_level'):
        compat_threshold_type = 'epoch_level'

    validation_metrics_compat = {}
    if compat_threshold_type is not None:
        validation_metrics_compat = dict(threshold_results.get(compat_threshold_type, {}))
        if validation_metrics_compat:
            validation_metrics_compat.setdefault('threshold_type', compat_threshold_type)
            target_fpr_value = threshold_results.get('target_fpr')
            if target_fpr_value is not None and 'target_fpr' not in validation_metrics_compat:
                validation_metrics_compat['target_fpr'] = target_fpr_value

    test_metrics_compat = {}
    if compat_threshold_type is not None:
        test_result = test_results.get(compat_threshold_type, {})
        metrics_key = 'guid_metrics' if compat_threshold_type == 'guid_level' else 'epoch_metrics'
        test_metrics_compat = dict(test_result.get(metrics_key, {}))
        if test_metrics_compat:
            test_metrics_compat.setdefault('threshold_type', compat_threshold_type)
            threshold_value = test_result.get('threshold')
            if threshold_value is not None and 'threshold' not in test_metrics_compat:
                test_metrics_compat['threshold'] = threshold_value

    return {
        'threshold_info': threshold_results,
        'test_results': test_results,
        'validation_metrics': validation_metrics_compat,
        'test_metrics': test_metrics_compat,
    }


# ==================================================================================
# K-FOLD CROSS-VALIDATION RESULT AGGREGATION
# ==================================================================================

def load_fold_predictions(fold_dir: Path, threshold_type: str) -> pd.DataFrame:
    """
    Load predictions CSV for a single fold.

    Args:
        fold_dir: Path to fold directory (e.g., fold_1)
        threshold_type: 'epoch_level' or 'guid_level'

    Returns:
        DataFrame with predictions, or None if file not found
    """
    csv_path = fold_dir / "evaluation" / f"test_predictions_clinical_{threshold_type}.csv"

    if not csv_path.exists():
        logger.warning(f"CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    df['epoch_hours'] = df['epoch'] / 3600  # Convert seconds to hours
    return df


def load_all_folds(base_dir: Path, threshold_type: str, num_folds: int) -> List[pd.DataFrame]:
    """
    Load predictions from all folds.

    Args:
        base_dir: Base directory containing fold_1, fold_2, ..., fold_N
        threshold_type: 'epoch_level' or 'guid_level'
        num_folds: Number of folds to load

    Returns:
        List of DataFrames, one per successfully loaded fold
    """
    fold_data = []

    for fold_id in range(1, num_folds + 1):
        fold_dir = base_dir / f"fold_{fold_id}"
        df = load_fold_predictions(fold_dir, threshold_type)

        if df is not None:
            df['fold_id'] = fold_id  # Track which fold this came from
            fold_data.append(df)
            logger.info(f"Loaded fold {fold_id}: {len(df)} samples")

    logger.info(f"Successfully loaded {len(fold_data)}/{num_folds} folds")
    return fold_data


def compute_fold_metrics_by_time(df: pd.DataFrame, time_bins: np.ndarray) -> pd.DataFrame:
    """
    Compute sensitivity and specificity for each time bin in a single fold.

    Args:
        df: Predictions dataframe for one fold
        time_bins: Time bin edges in hours (from compute_time_bins)

    Returns:
        DataFrame with columns: bin_start, bin_end, bin_center, sensitivity,
                                specificity, n_class_0, n_class_1
    """
    # FIX #5: Ensure epoch_hours column exists
    df = ensure_epoch_hours(df)

    results = []

    for i in range(len(time_bins) - 1):
        bin_start, bin_end = time_bins[i], time_bins[i + 1]
        bin_center = (bin_start + bin_end) / 2

        # Filter to this time bin
        bin_mask = (df['epoch_hours'] >= bin_start) & (df['epoch_hours'] < bin_end)
        bin_data = df[bin_mask]

        if len(bin_data) == 0:
            # No data in this bin - use NaN
            results.append({
                'bin_start': bin_start,
                'bin_end': bin_end,
                'bin_center': bin_center,
                'sensitivity': np.nan,
                'specificity': np.nan,
                'n_class_0': 0,
                'n_class_1': 0
            })
            continue

        # Compute metrics (same logic as plot_metrics_vs_time)
        class_0 = bin_data[bin_data['binary_target'] == 0]
        class_1 = bin_data[bin_data['binary_target'] == 1]

        specificity = 1 - (class_0['clinical_pred'] == 1).mean() if len(class_0) > 0 else np.nan
        sensitivity = (class_1['clinical_pred'] == 1).mean() if len(class_1) > 0 else np.nan

        results.append({
            'bin_start': bin_start,
            'bin_end': bin_end,
            'bin_center': bin_center,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'n_class_0': len(class_0),
            'n_class_1': len(class_1)
        })

    return pd.DataFrame(results)


def aggregate_metrics_across_folds(fold_metrics_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate metrics from all folds.

    Args:
        fold_metrics_list: List of DataFrames, one per fold (from compute_fold_metrics_by_time)

    Returns:
        DataFrame with columns: bin_center, sensitivity_mean, sensitivity_min, sensitivity_max,
                                specificity_mean, specificity_min, specificity_max, n_folds
    """
    # Concatenate all folds
    all_metrics = pd.concat(fold_metrics_list, ignore_index=True)

    # Group by time bin and aggregate
    aggregated = all_metrics.groupby('bin_center').agg({
        'sensitivity': ['mean', 'min', 'max', 'count'],
        'specificity': ['mean', 'min', 'max', 'count'],
        'n_class_0': 'sum',
        'n_class_1': 'sum'
    }).reset_index()

    # Flatten column names
    aggregated.columns = [
        'bin_center',
        'sensitivity_mean', 'sensitivity_min', 'sensitivity_max', 'n_folds_sens',
        'specificity_mean', 'specificity_min', 'specificity_max', 'n_folds_spec',
        'total_class_0', 'total_class_1'
    ]

    return aggregated


def plot_aggregated_metrics(
    aggregated_df: pd.DataFrame,
    output_dir: Path,
    threshold_type: str
) -> None:
    """
    Plot sensitivity and specificity with confidence bounds across all folds.

    Args:
        aggregated_df: Aggregated metrics from aggregate_metrics_across_folds
        output_dir: Directory to save plot
        threshold_type: 'epoch_level' or 'guid_level' (for title)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Negate bin_center to show negative hours (before delivery) on left → 0 (delivery) on right
    x = -aggregated_df['bin_center'].values

    # Plot sensitivity
    ax.plot(x, aggregated_df['sensitivity_mean'],
            marker='o', label='Sensitivity (mean)',
            linewidth=2, markersize=6, color='#2E86AB')
    ax.fill_between(x,
                     aggregated_df['sensitivity_min'],
                     aggregated_df['sensitivity_max'],
                     alpha=0.2, color='#2E86AB', label='Sensitivity (min-max)')

    # Plot specificity
    ax.plot(x, aggregated_df['specificity_mean'],
            marker='s', label='Specificity (mean)',
            linewidth=2, markersize=6, color='#A23B72')
    ax.fill_between(x,
                     aggregated_df['specificity_min'],
                     aggregated_df['specificity_max'],
                     alpha=0.2, color='#A23B72', label='Specificity (min-max)')

    # Formatting (follows existing conventions)
    ax.set_xlabel('Time Before Birth (hours, 0 = delivery)', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title(
        f'10-Fold CV: Sensitivity and Specificity vs. Time ({threshold_type.replace("_", " ").title()})',
        fontsize=14
    )
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    output_path = output_dir / f"kfold_aggregated_metrics_{threshold_type}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved aggregated plot: {output_path}")


def aggregate_kfold_metrics(
    kfold_results_dir: str,
    threshold_type: str = 'both',
    num_folds: int = 10,
    output_dir: str = None
) -> Dict:
    """
    Aggregate sensitivity/specificity metrics across all K-fold cross-validation folds.

    This function loads test predictions from all folds, computes metrics per time bin
    for each fold, and aggregates across folds with mean and confidence bounds (min/max).

    Args:
        kfold_results_dir: Base directory containing fold_1, fold_2, ..., fold_N
        threshold_type: 'epoch_level', 'guid_level', or 'both'
        num_folds: Number of folds (default: 10)
        output_dir: Where to save aggregated results (default: kfold_results_dir/aggregated)

    Returns:
        Dictionary with aggregated metrics and metadata
    """
    base_dir = Path(kfold_results_dir)

    if output_dir is None:
        output_dir = base_dir / "aggregated"
    else:
        output_dir = Path(output_dir)

    threshold_types = ['epoch_level', 'guid_level'] if threshold_type == 'both' else [threshold_type]

    all_results = {}

    for thr_type in threshold_types:
        logger.info("=" * 80)
        logger.info(f"AGGREGATING RESULTS FOR: {thr_type}")
        logger.info("=" * 80)

        # 1. Load all folds
        fold_data = load_all_folds(base_dir, thr_type, num_folds)

        if len(fold_data) == 0:
            logger.error(f"No folds loaded for {thr_type}. Skipping.")
            continue

        # 2. Compute unified time bins from first fold (all use same bins)
        logger.info("Computing time bins from first fold...")
        time_bins = compute_time_bins(fold_data[0])
        logger.info(f"Time bins: {time_bins}")

        # 3. Compute metrics for each fold
        logger.info("Computing metrics per fold...")
        fold_metrics_list = []
        for i, df in enumerate(fold_data, 1):
            fold_metrics = compute_fold_metrics_by_time(df, time_bins)
            fold_metrics['fold_id'] = df['fold_id'].iloc[0]  # Add fold ID
            fold_metrics_list.append(fold_metrics)
            logger.info(f"  Fold {i}: computed metrics for {len(fold_metrics)} time bins")

        # 4. Aggregate across folds
        logger.info("Aggregating across folds...")
        aggregated_df = aggregate_metrics_across_folds(fold_metrics_list)

        # 5. Save outputs
        thr_output_dir = output_dir / thr_type
        thr_output_dir.mkdir(parents=True, exist_ok=True)

        # Save aggregated metrics
        agg_csv_path = thr_output_dir / "aggregated_metrics.csv"
        aggregated_df.to_csv(agg_csv_path, index=False)
        logger.info(f"Saved: {agg_csv_path}")

        # Save raw fold metrics (for debugging)
        raw_csv_path = thr_output_dir / "fold_metrics_raw.csv"
        all_fold_metrics = pd.concat(fold_metrics_list, ignore_index=True)
        all_fold_metrics.to_csv(raw_csv_path, index=False)
        logger.info(f"Saved: {raw_csv_path}")

        # Save summary metadata
        summary = {
            'threshold_type': thr_type,
            'num_folds_requested': num_folds,
            'num_folds_loaded': len(fold_data),
            'missing_folds': [i for i in range(1, num_folds + 1) if i not in [df['fold_id'].iloc[0] for df in fold_data]],
            'time_bins_used': time_bins.tolist(),
            'aggregation_date': datetime.now().isoformat()
        }
        summary_path = thr_output_dir / "aggregation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved: {summary_path}")

        # 6. Generate plot
        logger.info("Generating aggregated plot...")
        plot_aggregated_metrics(aggregated_df, thr_output_dir, thr_type)

        # Store results
        all_results[thr_type] = {
            'aggregated_metrics': aggregated_df,
            'fold_metrics': all_fold_metrics,
            'summary': summary
        }

        logger.info(f"Aggregation complete for {thr_type}")
        logger.info("=" * 80)

    logger.info("ALL AGGREGATIONS COMPLETE!")
    return all_results


# ==================================================================================
# SUBGROUP AGGREGATION FUNCTIONS
# ==================================================================================

def aggregate_subgroup_metrics(
    kfold_results_dir: str,
    threshold_type: str = 'epoch_level',
    num_folds: int = 10,
    completed_fold_ids: List[int] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Aggregate subgroup metrics across all folds.

    Args:
        kfold_results_dir: Base directory containing fold_1, fold_2, ..., fold_N
        threshold_type: 'epoch_level' or 'guid_level'
        num_folds: Number of folds configured (for reference)
        completed_fold_ids: List of fold IDs to aggregate. If None, tries to load 1..num_folds.

    Returns:
        Tuple of (aggregated_df, metadata_dict)
    """
    # Determine which folds to load
    if completed_fold_ids is None:
        fold_ids_to_load = list(range(1, num_folds + 1))
    else:
        fold_ids_to_load = sorted(completed_fold_ids)

    all_fold_data = []
    loaded_fold_ids = []

    # Load subgroup metrics from each fold
    for fold_id in fold_ids_to_load:
        csv_path = Path(kfold_results_dir) / f"fold_{fold_id}" / "evaluation" / "plots" / threshold_type / "subgroup_analysis" / "subgroup_metrics.csv"

        if csv_path.exists():
            fold_df = pd.read_csv(csv_path)
            fold_df['fold_id'] = fold_id
            all_fold_data.append(fold_df)
            loaded_fold_ids.append(fold_id)
        else:
            logger.warning(f"Missing subgroup metrics for fold {fold_id}: {csv_path}")

    # Build metadata
    missing_fold_ids = sorted(set(fold_ids_to_load) - set(loaded_fold_ids))
    metadata = {
        'requested_folds': fold_ids_to_load,
        'loaded_folds': loaded_fold_ids,
        'missing_folds': missing_fold_ids,
        'n_requested': len(fold_ids_to_load),
        'n_loaded': len(loaded_fold_ids),
        'n_missing': len(missing_fold_ids),
        'aggregation_timestamp': datetime.now().isoformat()
    }

    if not all_fold_data:
        logger.error(f"No subgroup metrics found for {threshold_type}. Requested: {fold_ids_to_load}")
        return pd.DataFrame(), metadata

    # Concatenate all folds
    combined_df = pd.concat(all_fold_data, ignore_index=True)

    # Group by subgroup and bin_center, compute mean ± std
    agg_metrics = combined_df.groupby(['subgroup', 'bin_center']).agg({
        'sensitivity': ['mean', 'std', 'min', 'max'],
        'specificity': ['mean', 'std', 'min', 'max'],
        'n_samples': 'sum',
        'prevalence': 'mean'
    }).reset_index()

    # Flatten column names
    agg_metrics.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in agg_metrics.columns.values]

    return agg_metrics, metadata


def plot_aggregated_subgroup_metrics(
    agg_df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """
    Plot aggregated subgroup metrics with mean ± std error bands.

    Creates plots for key subgroup comparisons:
    - Diagnosis: Acidosis vs HIE
    - Delivery: CS+ vs CS-
    - Background: BG+ vs BG-
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define subgroup sets to plot
    plot_groups = {
        'diagnosis': ['acidosis', 'hie'],
        'cs_status': ['cs_positive', 'cs_negative'],
        'bg_label': ['bg_positive', 'bg_negative']
    }

    for group_name, subgroup_list in plot_groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        colors = plt.cm.Set2(np.linspace(0, 1, len(subgroup_list)))

        for idx, subgroup_name in enumerate(subgroup_list):
            subgroup_data = agg_df[agg_df['subgroup'] == subgroup_name]

            if len(subgroup_data) == 0:
                continue

            # Negate bin_center to show negative hours (before delivery) on left → 0 (delivery) on right
            x = -subgroup_data['bin_center'].values

            # Sensitivity plot with error bands
            sens_mean = subgroup_data['sensitivity_mean'].values
            sens_std = subgroup_data['sensitivity_std'].values

            # Filter out NaN values
            valid_sens = ~np.isnan(sens_mean)
            if valid_sens.any():
                axes[0].plot(x[valid_sens], sens_mean[valid_sens], marker='o',
                           label=subgroup_name.replace('_', ' ').title(),
                           linewidth=2, color=colors[idx])
                # Only fill_between where std is also valid
                valid_std = valid_sens & ~np.isnan(sens_std)
                if valid_std.any():
                    axes[0].fill_between(x[valid_std],
                                        sens_mean[valid_std] - sens_std[valid_std],
                                        sens_mean[valid_std] + sens_std[valid_std],
                                        alpha=0.2, color=colors[idx])

            # Specificity plot with error bands
            spec_mean = subgroup_data['specificity_mean'].values
            spec_std = subgroup_data['specificity_std'].values

            # Filter out NaN values
            valid_spec = ~np.isnan(spec_mean)
            if valid_spec.any():
                axes[1].plot(x[valid_spec], spec_mean[valid_spec], marker='o',
                           label=subgroup_name.replace('_', ' ').title(),
                           linewidth=2, color=colors[idx])
                # Only fill_between where std is also valid
                valid_std = valid_spec & ~np.isnan(spec_std)
                if valid_std.any():
                    axes[1].fill_between(x[valid_std],
                                        spec_mean[valid_std] - spec_std[valid_std],
                                        spec_mean[valid_std] + spec_std[valid_std],
                                        alpha=0.2, color=colors[idx])

        # Format plots
        axes[0].set_xlabel('Time Before Birth (hours, 0 = delivery)', fontsize=12)
        axes[0].set_ylabel('Sensitivity (mean ± std)', fontsize=12)
        axes[0].set_title(f'Sensitivity by {group_name.replace("_", " ").title()}', fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.05])

        axes[1].set_xlabel('Time Before Birth (hours, 0 = delivery)', fontsize=12)
        axes[1].set_ylabel('Specificity (mean ± std)', fontsize=12)
        axes[1].set_title(f'Specificity by {group_name.replace("_", " ").title()}', fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])

        plt.suptitle(f'Cross-Fold Aggregated Metrics: {group_name.replace("_", " ").title()} {title_suffix}',
                     fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f"aggregated_{group_name}_metrics.png", dpi=150)
        plt.close()

        logger.info(f"Saved: aggregated_{group_name}_metrics.png")


def run_subgroup_aggregation(
    kfold_results_dir: str,
    num_folds: int = 10,
    completed_fold_ids: List[int] = None
):
    """
    Run complete subgroup aggregation across folds.

    Args:
        kfold_results_dir: Base directory containing fold_1, fold_2, ..., fold_N
        num_folds: Total number of folds configured
        completed_fold_ids: List of fold IDs to aggregate. If None, tries all folds 1..num_folds.

    Usage:
        run_subgroup_aggregation("/path/to/kfold_results", num_folds=10, completed_fold_ids=[1,2,3])
    """
    output_dir = Path(kfold_results_dir) / "aggregated_analysis" / "subgroups"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metadata = {}

    for threshold_type in ['epoch_level', 'guid_level']:
        logger.info(f"Aggregating subgroup metrics for {threshold_type}...")

        # Aggregate metrics (now returns tuple)
        agg_df, metadata = aggregate_subgroup_metrics(
            kfold_results_dir, threshold_type, num_folds, completed_fold_ids
        )

        all_metadata[threshold_type] = metadata

        if agg_df.empty:
            logger.warning(f"No data for {threshold_type}, skipping...")
            continue

        # Save aggregated CSV
        csv_path = output_dir / f"aggregated_subgroup_metrics_{threshold_type}.csv"
        agg_df.to_csv(csv_path, index=False)
        logger.info(f"Saved aggregated metrics: {csv_path}")

        # Plot aggregated metrics
        plot_dir = output_dir / threshold_type
        plot_aggregated_subgroup_metrics(agg_df, plot_dir, title_suffix=f"({threshold_type})")

    # Save aggregation metadata
    metadata_path = output_dir / "aggregation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    logger.info(f"Saved aggregation metadata: {metadata_path}")

    # Log summary
    logger.info("=" * 80)
    logger.info("AGGREGATION SUMMARY")
    logger.info("=" * 80)
    for thr_type, meta in all_metadata.items():
        logger.info(f"{thr_type}:")
        logger.info(f"  Loaded folds: {meta['loaded_folds']}")
        logger.info(f"  Missing folds: {meta['missing_folds']}")
        logger.info(f"  Coverage: {meta['n_loaded']}/{meta['n_requested']} folds")
    logger.info("=" * 80)

    logger.info("Subgroup aggregation complete!")


def main():
    """
    Run evaluation for every fold (plus aggregated metrics) from a k-fold results root.
    Set ROOT_RESULTS_DIR to the directory containing fold_1, fold_2, ..., fold_N.
    """
    ROOT_RESULTS_DIR = "/data/deid/isilon/MS_model/classifier_kfold_results"
    TARGET_FPR = 0.05
    DEVICE = "cuda:0"
    FOLDS_TO_EVALUATE = None  # e.g., [1, 3, 5] to evaluate specific folds
    RUN_AGGREGATIONS = True

    root_dir = Path(ROOT_RESULTS_DIR)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root results directory not found: {root_dir}")

    fold_dirs: List[Tuple[int, Path]] = []
    for path in sorted(root_dir.iterdir()):
        if not path.is_dir():
            continue
        if not path.name.startswith("fold_"):
            continue
        try:
            fold_id = int(path.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        fold_dirs.append((fold_id, path))

    if not fold_dirs:
        logger.error(f"No fold directories found in {root_dir}")
        return

    if FOLDS_TO_EVALUATE is not None:
        requested = set(FOLDS_TO_EVALUATE)
        fold_dirs = [(fid, fdir) for fid, fdir in fold_dirs if fid in requested]
        if not fold_dirs:
            logger.error(f"No matching folds found for requested IDs: {FOLDS_TO_EVALUATE}")
            return

    logger.info(f"Evaluating folds: {[fid for fid, _ in fold_dirs]}")

    for fold_id, fold_dir in fold_dirs:
        config_path = fold_dir / "config.yaml"
        if not config_path.exists():
            logger.warning(f"Config not found for fold {fold_id}: {config_path}")
            continue
        try:
            checkpoint_path = find_latest_checkpoint_in_fold(fold_dir)
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue

        logger.info("=" * 80)
        logger.info(f"Evaluating Fold {fold_id}")
        logger.info("=" * 80)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        model = create_model_from_config(config, DEVICE)
        evaluate_fold(
            model=model,
            fold_dir=str(fold_dir),
            config_path=str(config_path),
            checkpoint_path=str(checkpoint_path),
            target_fpr=TARGET_FPR,
            device=DEVICE
        )

    if RUN_AGGREGATIONS:
        max_fold_id = max(fid for fid, _ in fold_dirs)
        logger.info("Running cross-fold aggregation...")
        aggregate_kfold_metrics(
            kfold_results_dir=ROOT_RESULTS_DIR,
            threshold_type='both',
            num_folds=max_fold_id
        )
        run_subgroup_aggregation(
            kfold_results_dir=ROOT_RESULTS_DIR,
            num_folds=max_fold_id
        )

    logger.info("Complete evaluation pipeline finished.")


if __name__ == '__main__':
    main()
