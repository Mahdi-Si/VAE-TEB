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
from typing import Dict, Tuple, List
from loguru import logger
import yaml
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
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
    if len(guid_epochs) < 2:
        return 0.0  # Cannot compute interval with single epoch

    # Compute differences between consecutive epochs
    # epochs are in descending order, so diff = epochs[i] - epochs[i+1]
    intervals = np.diff(guid_epochs[::-1])  # Reverse to get ascending order for diff

    # Return median interval (robust to outliers)
    median_interval = np.median(intervals)

    return median_interval


def apply_clinical_decision_rule(
    df: pd.DataFrame,
    threshold: float
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

        # Sort by epoch in descending order (furthest from birth to closest)
        guid_data = guid_data.sort_values('epoch', ascending=False)

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
        df.loc[guid_mask, 'clinical_pred'] = guid_data['clinical_pred'].values
        df.loc[guid_mask, 'first_detection_epoch'] = guid_data['first_detection_epoch'].values

    logger.info(f"Clinical decision rule applied. "
                f"Model positives: {df['model_pred'].sum()}, "
                f"Clinical positives: {df['clinical_pred'].sum()}")

    return df


def fill_missing_epochs(
    df: pd.DataFrame,
    max_gap_multiplier: float = 2.0
) -> pd.DataFrame:
    """
    Fill missing epochs for each GUID using forward-filling strategy.

    Args:
        df: Predictions dataframe with clinical decision rule applied
        max_gap_multiplier: Only fill gaps <= multiplier * typical_interval

    Returns:
        Complete dataframe with all epochs (filled + original)
        New column 'is_filled': True if epoch was missing and filled
    """
    df = df.copy()
    df['is_filled'] = False

    all_rows = []

    # Process each GUID separately
    for guid in df['guid'].unique():
        guid_mask = df['guid'] == guid
        guid_data = df.loc[guid_mask].copy()

        # Sort by epoch in descending order
        guid_data = guid_data.sort_values('epoch', ascending=False)
        epochs = guid_data['epoch'].values

        # Compute typical interval for this GUID
        typical_interval = compute_epoch_intervals(epochs)

        if typical_interval == 0 or len(epochs) < 2:
            # Skip filling if only one epoch or interval cannot be computed
            all_rows.append(guid_data)
            continue

        max_gap = typical_interval * max_gap_multiplier

        # Identify missing epochs
        min_epoch = epochs.min()
        max_epoch = epochs.max()

        # Generate expected epochs
        num_expected = int((max_epoch - min_epoch) / typical_interval) + 1
        expected_epochs = np.linspace(min_epoch, max_epoch, num_expected)

        # Round to avoid floating point issues
        expected_epochs = np.round(expected_epochs, 1)
        existing_epochs_set = set(np.round(epochs, 1))

        # Find missing epochs (only fill gaps <= max_gap)
        filled_rows = []
        guid_data_dict = guid_data.set_index('epoch').to_dict('index')

        for i, exp_epoch in enumerate(expected_epochs):
            if exp_epoch in existing_epochs_set:
                # Epoch exists, use original data
                closest_epoch = epochs[np.argmin(np.abs(epochs - exp_epoch))]
                row_data = guid_data_dict[closest_epoch]
                row_data['epoch'] = closest_epoch
                row_data['is_filled'] = False
                filled_rows.append(row_data)
            else:
                # Missing epoch - check if gap is within acceptable range
                if i == 0:
                    # First epoch missing - default to healthy
                    filled_row = {
                        'guid': guid,
                        'epoch': exp_epoch,
                        'cs_label': guid_data.iloc[0]['cs_label'] if 'cs_label' in guid_data.columns else None,
                        'bg_label': guid_data.iloc[0]['bg_label'] if 'bg_label' in guid_data.columns else None,
                        'binary_target': 0,  # Default to healthy
                        'target': 1,  # Healthy label
                        'prob_class_0': np.nan,
                        'prob_class_1': np.nan,
                        'model_pred': 0,
                        'clinical_pred': 0,
                        'first_detection_epoch': np.nan,
                        'is_fil2led': True
                    }
                    filled_rows.append(filled_row)
                else:
                    # Find preceding epoch (next larger epoch value)
                    preceding_epochs = epochs[epochs > exp_epoch]
                    if len(preceding_epochs) > 0:
                        preceding_epoch = preceding_epochs.min()
                        gap = preceding_epoch - exp_epoch

                        if gap <= max_gap:
                            # Fill from preceding epoch
                            preceding_data = guid_data_dict[preceding_epoch]
                            filled_row = {
                                'guid': guid,
                                'epoch': exp_epoch,
                                'cs_label': preceding_data.get('cs_label'),
                                'bg_label': preceding_data.get('bg_label'),
                                'binary_target': preceding_data['binary_target'],
                                'target': preceding_data['target'],
                                'prob_class_0': np.nan,
                                'prob_class_1': np.nan,
                                'model_pred': preceding_data['model_pred'],
                                'clinical_pred': preceding_data['clinical_pred'],
                                'first_detection_epoch': preceding_data.get('first_detection_epoch', np.nan),
                                'is_filled': True
                            }
                            filled_rows.append(filled_row)

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

    n_filled = result_df['is_filled'].sum()
    n_total = len(result_df)
    logger.info(f"Missing epoch filling complete. "
                f"Original epochs: {n_total - n_filled}, "
                f"Filled epochs: {n_filled}, "
                f"Total: {n_total}")

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
        threshold_candidates = np.arange(0.0, 1.01, 0.01)

    best_threshold = 0.5
    best_fpr_diff = float('inf')
    best_metrics = {}

    for thresh in threshold_candidates:
        # Apply clinical decision rule at this threshold
        df_clinical = apply_clinical_decision_rule(val_df.copy(), thresh)

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


def find_optimal_thresholds(
    val_df: pd.DataFrame,
    target_fpr: float = 0.05
) -> Dict:
    """
    Compute both epoch-level and GUID-level thresholds for comparison.

    Args:
        val_df: Validation predictions DataFrame
        target_fpr: Target false positive rate for threshold determination

    Returns:
        Dictionary with both epoch-level and GUID-level threshold information
    """
    logger.info("Computing optimal thresholds using both approaches...")

    # Epoch-level threshold
    epoch_threshold, epoch_metrics = find_threshold_for_fpr_epoch(val_df, target_fpr)

    # GUID-level threshold
    guid_threshold, guid_metrics = find_threshold_for_fpr_guid(val_df, target_fpr)

    results = {
        'epoch_level': {
            'threshold': epoch_threshold,
            **epoch_metrics
        },
        'guid_level': {
            'threshold': guid_threshold,
            **guid_metrics
        },
        'target_fpr': target_fpr
    }

    logger.info("=" * 80)
    logger.info("THRESHOLD COMPARISON")
    logger.info("=" * 80)
    logger.info(f"Epoch-level threshold: {epoch_threshold:.4f} (FPR: {epoch_metrics['actual_fpr']:.4f})")
    logger.info(f"GUID-level threshold: {guid_threshold:.4f} (FPR: {guid_metrics['actual_fpr']:.4f})")
    logger.info("=" * 80)

    return results


def compute_time_bins(df: pd.DataFrame) -> np.ndarray:
    """
    Compute dynamic time bins from actual epoch data.

    Focuses on last 6 hours with finer granularity (30-min bins),
    then coarser bins beyond that.

    Args:
        df: DataFrame with 'epoch' column (seconds before birth)

    Returns:
        Array of time bin edges in hours
    """
    max_epoch_hours = df['epoch'].max() / 3600

    if max_epoch_hours <= 6:
        # All data within 6 hours - use 30-min bins
        bins = np.arange(0, max_epoch_hours + 0.5, 0.5)
    else:
        # Fine bins for 0-6h, coarser beyond
        bins = np.concatenate([
            np.arange(0, 6, 0.5),                          # 0-6h: 30-min bins
            np.arange(6, min(12, max_epoch_hours), 1),    # 6-12h: 1-hour bins
            np.arange(12, max_epoch_hours + 2, 2)          # >12h: 2-hour bins
        ])

    return bins


def compute_time_windows(df: pd.DataFrame) -> List[Tuple[float, float]]:
    """
    Compute time windows for ROC/confusion matrix analysis.

    Args:
        df: DataFrame with 'epoch' column (seconds before birth)

    Returns:
        List of (start_hour, end_hour) tuples
    """
    max_epoch_hours = df['epoch'].max() / 3600

    # Fixed critical windows
    windows = [(0, 1), (1, 2), (2, 4), (4, 6), (0, 6)]

    # Add additional windows if data extends beyond 6 hours
    if max_epoch_hours > 6:
        windows.extend([(6, 12), (0, 12)])

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
    time_bins: np.ndarray
) -> pd.DataFrame:
    """
    Compute sensitivity/specificity for a specific subgroup across time bins.

    Args:
        df: Full predictions dataframe
        subgroup_name: Name of subgroup (for labeling)
        subgroup_filter: Function returning boolean mask for subgroup
        time_bins: Time bin edges in hours

    Returns:
        DataFrame with: bin_center, sensitivity, specificity, n_samples,
                       n_detected, prevalence
    """
    # Apply subgroup filter
    subgroup_df = df[subgroup_filter(df)].copy()

    if len(subgroup_df) == 0:
        logger.warning(f"No samples in subgroup: {subgroup_name}")
        return pd.DataFrame()

    # Ensure epoch_hours exists
    if 'epoch_hours' not in subgroup_df.columns:
        subgroup_df['epoch_hours'] = subgroup_df['epoch'] / 3600

    results = []

    for i in range(len(time_bins) - 1):
        bin_start, bin_end = time_bins[i], time_bins[i + 1]
        bin_center = (bin_start + bin_end) / 2

        # Filter to time bin
        bin_mask = (subgroup_df['epoch_hours'] >= bin_start) & (subgroup_df['epoch_hours'] < bin_end)
        bin_data = subgroup_df[bin_mask]

        if len(bin_data) == 0:
            results.append({
                'bin_center': bin_center,
                'n_samples': 0,
                'sensitivity': np.nan,
                'specificity': np.nan,
                'n_detected': 0,
                'prevalence': 0.0
            })
            continue

        # Compute metrics (same pattern as existing code)
        class_0 = bin_data[bin_data['binary_target'] == 0]
        class_1 = bin_data[bin_data['binary_target'] == 1]

        sensitivity = (class_1['clinical_pred'] == 1).mean() if len(class_1) > 0 else np.nan
        specificity = 1 - (class_0['clinical_pred'] == 1).mean() if len(class_0) > 0 else np.nan
        n_detected = (bin_data['clinical_pred'] == 1).sum()
        prevalence = len(class_1) / len(bin_data) if len(bin_data) > 0 else 0.0

        results.append({
            'bin_center': bin_center,
            'n_samples': len(bin_data),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'n_detected': n_detected,
            'prevalence': prevalence
        })

    df_result = pd.DataFrame(results)
    df_result['subgroup'] = subgroup_name
    return df_result


def plot_metrics_vs_time(
    df: pd.DataFrame,
    output_dir: Path,
    prefix: str = ""
) -> None:
    """
    Plot sensitivity and specificity as function of time before birth.

    Args:
        df: Predictions dataframe with clinical_pred
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames (e.g., "epoch_threshold_" or "guid_threshold_")
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute dynamic time bins
    time_bins = compute_time_bins(df)

    # Convert epoch to hours
    df = df.copy()
    df['epoch_hours'] = df['epoch'] / 3600

    # Compute metrics for each time bin
    sensitivities = []
    specificities = []
    bin_centers = []

    for i in range(len(time_bins) - 1):
        bin_start, bin_end = time_bins[i], time_bins[i + 1]
        bin_mask = (df['epoch_hours'] >= bin_start) & (df['epoch_hours'] < bin_end)
        bin_data = df[bin_mask]

        if len(bin_data) == 0:
            continue

        # Compute metrics
        class_0 = bin_data[bin_data['binary_target'] == 0]
        class_1 = bin_data[bin_data['binary_target'] == 1]

        if len(class_0) > 0:
            specificity = 1 - (class_0['clinical_pred'] == 1).mean()
            specificities.append(specificity)
        else:
            specificities.append(np.nan)

        if len(class_1) > 0:
            sensitivity = (class_1['clinical_pred'] == 1).mean()
            sensitivities.append(sensitivity)
        else:
            sensitivities.append(np.nan)

        bin_centers.append((bin_start + bin_end) / 2)

    # Plot combined figure
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(bin_centers, sensitivities, marker='o', label='Sensitivity (TPR)', linewidth=2, markersize=6)
    ax.plot(bin_centers, specificities, marker='s', label='Specificity (TNR)', linewidth=2, markersize=6)

    ax.set_xlabel('Time Before Birth (hours)', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Sensitivity and Specificity vs. Time Before Birth', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}sensitivity_specificity_vs_time.png", dpi=150)
    plt.close()

    logger.info(f"Saved: {prefix}sensitivity_specificity_vs_time.png")


def plot_roc_curves_by_time(
    df: pd.DataFrame,
    output_dir: Path,
    prefix: str = ""
) -> None:
    """
    Plot ROC curves for different time windows before birth.

    Args:
        df: Predictions dataframe
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute dynamic time windows
    time_windows = compute_time_windows(df)

    # Convert epoch to hours
    df = df.copy()
    df['epoch_hours'] = df['epoch'] / 3600

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

        # Plot
        label = f"{start_h}-{end_h}h (AUC={roc_auc:.3f})"
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

    # Convert to hours
    detection_hours = detected['first_detection_epoch'].values / 3600

    # Plot 1: Histogram of detection times
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].hist(detection_hours, bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Time Before Birth (hours)', fontsize=12)
    axes[0].set_ylabel('Number of Detections', fontsize=12)
    axes[0].set_title('Distribution of First Detection Times', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Cumulative detection curve
    sorted_hours = np.sort(detection_hours)
    cumulative_pct = np.arange(1, len(sorted_hours) + 1) / len(sorted_hours) * 100

    axes[1].plot(sorted_hours, cumulative_pct, linewidth=2)
    axes[1].set_xlabel('Time Before Birth (hours)', fontsize=12)
    axes[1].set_ylabel('Cumulative % of Unhealthy GUIDs Detected', fontsize=12)
    axes[1].set_title('Cumulative Detection Curve', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}detection_timing.png", dpi=150)
    plt.close()

    logger.info(f"Saved: {prefix}detection_timing.png")


def plot_confusion_matrices_by_time(
    df: pd.DataFrame,
    output_dir: Path,
    prefix: str = ""
) -> None:
    """
    Plot confusion matrices for different time windows.

    Args:
        df: Predictions dataframe
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute dynamic time windows
    time_windows = compute_time_windows(df)

    # Convert epoch to hours
    df = df.copy()
    df['epoch_hours'] = df['epoch'] / 3600

    # Create grid of confusion matrices
    n_windows = len(time_windows)
    n_cols = 3
    n_rows = (n_windows + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for idx, (start_h, end_h) in enumerate(time_windows):
        window_mask = (df['epoch_hours'] >= start_h) & (df['epoch_hours'] < end_h)
        window_data = df[window_mask]

        if len(window_data) < 10:
            axes[idx].axis('off')
            continue

        # Compute confusion matrix
        cm = confusion_matrix(window_data['binary_target'], window_data['clinical_pred'])

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Healthy', 'Unhealthy'],
                    yticklabels=['Healthy', 'Unhealthy'])
        axes[idx].set_title(f'{start_h}-{end_h}h before birth')
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

    cm = confusion_matrix(guid_data['binary_target'], guid_data['clinical_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Healthy', 'Unhealthy'],
                yticklabels=['Healthy', 'Unhealthy'])
    axes[0, 0].set_title('GUID-Level Classification Matrix')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')

    # Plot 2: Time to detection for unhealthy GUIDs
    unhealthy = guid_data[guid_data['binary_target'] == 1].copy()
    detected = unhealthy[unhealthy['first_detection_epoch'].notna()]
    missed = unhealthy[unhealthy['first_detection_epoch'].isna()]

    detection_hours = detected['first_detection_epoch'].values / 3600 if len(detected) > 0 else []

    if len(detection_hours) > 0:
        axes[0, 1].hist(detection_hours, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Time Before Birth (hours)')
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
    df_epoch['epoch_hours'] = df_epoch['epoch'] / 3600
    df_guid['epoch_hours'] = df_guid['epoch'] / 3600

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
    agreement_df = pd.DataFrame({
        'epoch_pred': df_epoch['clinical_pred'],
        'guid_pred': df_guid['clinical_pred']
    })

    cm = confusion_matrix(agreement_df['epoch_pred'], agreement_df['guid_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[1, 1],
                xticklabels=['GUID: Neg', 'GUID: Pos'],
                yticklabels=['Epoch: Neg', 'Epoch: Pos'])
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
    prefix: str = ""
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
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure epoch_hours exists
    df = df.copy()
    if 'epoch_hours' not in df.columns:
        df['epoch_hours'] = df['epoch'] / 3600

    # Compute metrics for each subgroup
    subgroup_results = {}
    for subgroup_name, subgroup_filter in subgroups.items():
        metrics_df = compute_subgroup_metrics_by_time(df, subgroup_name, subgroup_filter, time_bins)
        if len(metrics_df) > 0:
            subgroup_results[subgroup_name] = metrics_df

    if len(subgroup_results) == 0:
        logger.warning("No subgroup data available for plotting")
        return

    # Create figure with 2 subplots: sensitivity and specificity
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(subgroup_results)))
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X']

    for idx, (subgroup_name, metrics_df) in enumerate(subgroup_results.items()):
        color = colors[idx]
        marker = markers[idx % len(markers)]

        x = metrics_df['bin_center'].values

        # Sensitivity subplot
        axes[0].plot(x, metrics_df['sensitivity'],
                    marker=marker, label=subgroup_name.replace('_', ' ').title(),
                    linewidth=2, markersize=6, color=color)

        # Specificity subplot
        axes[1].plot(x, metrics_df['specificity'],
                    marker=marker, label=subgroup_name.replace('_', ' ').title(),
                    linewidth=2, markersize=6, color=color)

    # Format sensitivity subplot
    axes[0].set_xlabel('Time Before Birth (hours)', fontsize=12)
    axes[0].set_ylabel('Sensitivity (TPR)', fontsize=12)
    axes[0].set_title('Sensitivity by Subgroup', fontsize=14)
    axes[0].legend(fontsize=10, loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # Format specificity subplot
    axes[1].set_xlabel('Time Before Birth (hours)', fontsize=12)
    axes[1].set_ylabel('Specificity (TNR)', fontsize=12)
    axes[1].set_title('Specificity by Subgroup', fontsize=14)
    axes[1].legend(fontsize=10, loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}subgroup_metrics_vs_time.png", dpi=150)
    plt.close()

    logger.info(f"Saved: {prefix}subgroup_metrics_vs_time.png")


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
    df_copy = df.copy()
    if 'epoch_hours' not in df_copy.columns:
        df_copy['epoch_hours'] = df_copy['epoch'] / 3600

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
        axes[1, 1].plot(bin_centers, acidosis_prev, marker='o', label='Acidosis', linewidth=2)
        axes[1, 1].plot(bin_centers, hie_prev, marker='s', label='HIE', linewidth=2)
        axes[1, 1].set_xlabel('Time Before Birth (hours)', fontsize=12)
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

    # Ensure epoch_hours exists
    df = df.copy()
    if 'epoch_hours' not in df.columns:
        df['epoch_hours'] = df['epoch'] / 3600

    subgroup_filters = create_subgroup_filters()
    time_bins = compute_time_bins(df)

    all_subgroup_metrics = []

    for subgroup_name, subgroup_filter in subgroup_filters.items():
        metrics_df = compute_subgroup_metrics_by_time(df, subgroup_name, subgroup_filter, time_bins)
        if len(metrics_df) > 0:
            all_subgroup_metrics.append(metrics_df)

    if all_subgroup_metrics:
        combined_df = pd.concat(all_subgroup_metrics, ignore_index=True)
        csv_path = output_dir / f"{prefix}subgroup_metrics.csv"
        combined_df.to_csv(csv_path, index=False)
        logger.info(f"Saved: {csv_path}")
    else:
        logger.warning("No subgroup metrics to save")


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

    # Save raw validation predictions
    val_output_dir = fold_dir / "evaluation"
    val_output_dir.mkdir(parents=True, exist_ok=True)
    val_df_raw.to_csv(val_output_dir / "validation_predictions_raw.csv", index=False)
    logger.info("Validation raw predictions saved")

    # Find optimal thresholds using both approaches
    logger.info("=" * 80)
    logger.info("THRESHOLD DETERMINATION")
    logger.info("=" * 80)
    threshold_results = find_optimal_thresholds(val_df_raw, target_fpr)

    # Save threshold information
    with open(val_output_dir / "validation_threshold_info.json", 'w') as f:
        json.dump(threshold_results, f, indent=2)

    # Apply clinical decision rule and fill missing epochs for validation
    # (using epoch-level threshold for validation data processing)
    epoch_threshold = threshold_results['epoch_level']['threshold']
    val_df_clinical = apply_clinical_decision_rule(val_df_raw.copy(), epoch_threshold)
    val_df_clinical = fill_missing_epochs(val_df_clinical, max_gap_multiplier=2.0)

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

    # Save raw test predictions
    test_df_raw.to_csv(val_output_dir / "test_predictions_raw.csv", index=False)
    logger.info("Test raw predictions saved")

    # Process test set with both thresholds
    test_results = {}

    for threshold_type in ['epoch_level', 'guid_level']:
        logger.info(f"\nProcessing test set with {threshold_type} threshold...")

        threshold_value = threshold_results[threshold_type]['threshold']

        # Apply clinical decision rule
        test_df = apply_clinical_decision_rule(test_df_raw.copy(), threshold_value)

        # Fill missing epochs
        test_df = fill_missing_epochs(test_df, max_gap_multiplier=2.0)

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

    # Generate visualizations for both thresholds
    logger.info("=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)

    for threshold_type in ['epoch_level', 'guid_level']:
        threshold_value = threshold_results[threshold_type]['threshold']

        # Apply clinical rule for visualization
        test_df = apply_clinical_decision_rule(test_df_raw.copy(), threshold_value)
        test_df = fill_missing_epochs(test_df, max_gap_multiplier=2.0)

        # Create output directory for this threshold
        plots_dir = val_output_dir / "plots" / threshold_type
        plots_dir.mkdir(parents=True, exist_ok=True)

        prefix = ""

        # Generate all plots
        try:
            plot_metrics_vs_time(test_df, plots_dir, prefix)
            plot_roc_curves_by_time(test_df, plots_dir, prefix)
            plot_detection_timing(test_df, plots_dir, prefix)
            plot_confusion_matrices_by_time(test_df, plots_dir, prefix)
            plot_guid_level_analysis(test_df, plots_dir, prefix)
        except Exception as e:
            logger.warning(f"Error generating plots for {threshold_type}: {e}")

        # Generate subgroup analysis
        try:
            logger.info(f"Generating subgroup analysis for {threshold_type}...")

            # Create subgroup filters
            subgroup_filters = create_subgroup_filters()

            # Diagnosis-based subgroups (Acidosis vs HIE)
            diagnosis_subgroups = {
                'acidosis': subgroup_filters['acidosis'],
                'hie': subgroup_filters['hie']
            }

            # CS status subgroups
            cs_subgroups = {
                'cs_positive': subgroup_filters['cs_positive'],
                'cs_negative': subgroup_filters['cs_negative']
            }

            # Combined subgroups
            combined_subgroups = {
                'acidosis_cs_pos': subgroup_filters['acidosis_cs_pos'],
                'acidosis_cs_neg': subgroup_filters['acidosis_cs_neg'],
                'hie_cs_pos': subgroup_filters['hie_cs_pos'],
                'hie_cs_neg': subgroup_filters['hie_cs_neg']
            }

            # Create subgroup output directory
            subgroup_dir = plots_dir / "subgroup_analysis"

            # Compute time bins for subgroup analysis
            time_bins = compute_time_bins(test_df)

            # Plot diagnosis comparison (Acidosis vs HIE)
            plot_subgroup_metrics_vs_time(
                test_df, diagnosis_subgroups,
                time_bins,
                subgroup_dir, prefix="diagnosis_"
            )
            plot_subgroup_roc_curves(
                test_df, diagnosis_subgroups,
                subgroup_dir, prefix="diagnosis_"
            )

            # Plot CS status comparison
            plot_subgroup_metrics_vs_time(
                test_df, cs_subgroups,
                time_bins,
                subgroup_dir, prefix="cs_status_"
            )
            plot_subgroup_roc_curves(
                test_df, cs_subgroups,
                subgroup_dir, prefix="cs_status_"
            )

            # Plot BG label comparison
            bg_subgroups = {
                'bg_positive': subgroup_filters['bg_positive'],
                'bg_negative': subgroup_filters['bg_negative']
            }

            plot_subgroup_metrics_vs_time(
                test_df, bg_subgroups,
                time_bins,
                subgroup_dir, prefix="bg_label_"
            )
            plot_subgroup_roc_curves(
                test_df, bg_subgroups,
                subgroup_dir, prefix="bg_label_"
            )

            # Plot combined subgroups
            plot_subgroup_metrics_vs_time(
                test_df, combined_subgroups,
                time_bins,
                subgroup_dir, prefix="combined_"
            )

            # Plot sample distribution
            plot_subgroup_distribution(test_df, subgroup_dir, prefix="")

            # Save subgroup metrics to CSV
            save_subgroup_metrics(test_df, subgroup_dir, prefix="")

            logger.info(f"Subgroup analysis complete for {threshold_type}")

        except Exception as e:
            logger.warning(f"Error generating subgroup analysis for {threshold_type}: {e}")

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

    return {
        'threshold_info': threshold_results,
        'test_results': test_results,
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

    x = aggregated_df['bin_center'].values

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
    ax.set_xlabel('Time Before Birth (hours)', fontsize=12)
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
    num_folds: int = 10
) -> pd.DataFrame:
    """
    Aggregate subgroup metrics across all folds.

    Args:
        kfold_results_dir: Base directory containing fold_1, fold_2, ..., fold_N
        threshold_type: 'epoch_level' or 'guid_level'
        num_folds: Number of folds to aggregate

    Returns:
        DataFrame with aggregated metrics (mean ± std) per subgroup and time bin
    """
    all_fold_data = []

    # Load subgroup metrics from each fold
    for fold_id in range(1, num_folds + 1):
        csv_path = Path(kfold_results_dir) / f"fold_{fold_id}" / "evaluation" / "plots" / threshold_type / "subgroup_analysis" / "subgroup_metrics.csv"

        if csv_path.exists():
            fold_df = pd.read_csv(csv_path)
            fold_df['fold_id'] = fold_id
            all_fold_data.append(fold_df)
        else:
            logger.warning(f"Missing subgroup metrics for fold {fold_id}: {csv_path}")

    if not all_fold_data:
        logger.error("No subgroup metrics found across folds")
        return pd.DataFrame()

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

    return agg_metrics


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

            x = subgroup_data['bin_center'].values

            # Sensitivity plot with error bands
            sens_mean = subgroup_data['sensitivity_mean'].values
            sens_std = subgroup_data['sensitivity_std'].values

            axes[0].plot(x, sens_mean, marker='o', label=subgroup_name.replace('_', ' ').title(),
                        linewidth=2, color=colors[idx])
            axes[0].fill_between(x, sens_mean - sens_std, sens_mean + sens_std,
                                 alpha=0.2, color=colors[idx])

            # Specificity plot with error bands
            spec_mean = subgroup_data['specificity_mean'].values
            spec_std = subgroup_data['specificity_std'].values

            axes[1].plot(x, spec_mean, marker='o', label=subgroup_name.replace('_', ' ').title(),
                        linewidth=2, color=colors[idx])
            axes[1].fill_between(x, spec_mean - spec_std, spec_mean + spec_std,
                                 alpha=0.2, color=colors[idx])

        # Format plots
        axes[0].set_xlabel('Time Before Birth (hours)', fontsize=12)
        axes[0].set_ylabel('Sensitivity (mean ± std)', fontsize=12)
        axes[0].set_title(f'Sensitivity by {group_name.replace("_", " ").title()}', fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.05])

        axes[1].set_xlabel('Time Before Birth (hours)', fontsize=12)
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
    num_folds: int = 10
):
    """
    Run complete subgroup aggregation across folds.

    Usage:
        run_subgroup_aggregation("/path/to/kfold_results", num_folds=10)
    """
    output_dir = Path(kfold_results_dir) / "aggregated_analysis" / "subgroups"
    output_dir.mkdir(parents=True, exist_ok=True)

    for threshold_type in ['epoch_level', 'guid_level']:
        logger.info(f"Aggregating subgroup metrics for {threshold_type}...")

        # Aggregate metrics
        agg_df = aggregate_subgroup_metrics(kfold_results_dir, threshold_type, num_folds)

        if agg_df.empty:
            logger.warning(f"No data for {threshold_type}, skipping...")
            continue

        # Save aggregated CSV
        csv_path = output_dir / f"aggregated_subgroup_metrics_{threshold_type}.csv"
        agg_df.to_csv(csv_path, index=False)
        logger.info(f"Saved: {csv_path}")

        # Plot aggregated metrics
        plot_dir = output_dir / threshold_type
        plot_aggregated_subgroup_metrics(agg_df, plot_dir, title_suffix=f"({threshold_type})")

    logger.info("Subgroup aggregation complete!")


def main():
    """
    Example usage: Evaluate a trained fold standalone.

    Note: In the k-fold pipeline, the model is passed directly from the trainer.
    For standalone evaluation, you need to create the model from config first.
    """
    FOLD_DIR = "/data/deid/isilon/MS_model/classifier_kfold_results/fold_1"
    CONFIG_PATH = f"{FOLD_DIR}/config.yaml"
    CHECKPOINT_PATH = f"{FOLD_DIR}/checkpoints/classifier-model-epoch=XX-acc=0.XXXX.ckpt"  # Update with actual path
    TARGET_FPR = 0.05  # 5% false positive rate
    DEVICE = "cuda:0"

    # For standalone evaluation, create model from config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    model = create_model_from_config(config, DEVICE)

    results = evaluate_fold(
        model=model,
        fold_dir=FOLD_DIR,
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        target_fpr=TARGET_FPR,
        device=DEVICE
    )

    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
