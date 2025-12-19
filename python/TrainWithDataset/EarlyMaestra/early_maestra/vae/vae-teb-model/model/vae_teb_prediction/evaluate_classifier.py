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

    # Create time bins around target time
    bin_width = 0.5  # 30 minutes
    time_bins = np.array([
        time_window_hours + bin_width,
        time_window_hours,
        max(0, time_window_hours - bin_width)
    ])

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
        if fpr_diff < best_fpr_diff:
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

    # Create time bins around target time
    bin_width = 0.5
    time_bins = np.array([
        time_window_hours + bin_width,
        time_window_hours,
        max(0, time_window_hours - bin_width)
    ])

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
        if fpr_diff < best_fpr_diff:
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

    # Create time bins around target time
    bin_width = 0.5
    time_bins = np.array([
        time_window_hours + bin_width,
        time_window_hours,
        max(0, time_window_hours - bin_width)
    ])

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
        if fpr_diff < best_fpr_diff:
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
    if 'is_filled' in df_bins.columns:
        df_bins = df_bins[df_bins['is_filled'] == False]  # noqa: E712

    # Ensure epoch_hours exists (positive hours before birth)
    df_bins = ensure_epoch_hours(df_bins)

    # Infer bin size from typical epoch interval (fallback to 20 minutes if inference fails)
    inferred_seconds = infer_epoch_interval_seconds(df_bins)
    bin_size_hours = (inferred_seconds / 3600.0) if inferred_seconds > 0 else (1.0 / 3.0)

    # Convert exclusion from minutes to hours
    exclude_hours = exclude_last_minutes / 60.0  # 30min = 0.5h

    if len(df_bins) == 0:
        return np.array([exclude_hours, exclude_hours + bin_size_hours])

    df_included = df_bins[df_bins['epoch_hours'] >= exclude_hours]
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
    - For each GUID, check if detected (any epoch with clinical_pred=1 from τ to birth)
    - Sensitivity(τ) = TP(t≤τ) / P(t≤τ)
    - FPR(τ) = FP(t≤τ) / N(t≤τ)

    where:
    - TP(t≤τ) = detected unhealthy GUIDs (with data at τ)
    - P(t≤τ) = total unhealthy GUIDs available at time τ
    - FP(t≤τ) = detected healthy GUIDs (with data at τ)
    - N(t≤τ) = total healthy GUIDs available at time τ

    The denominator CHANGES as we approach birth (more GUIDs have data).
    This metric is monotonically non-decreasing.

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

    # Ensure required columns
    df = ensure_epoch_hours(df.copy())

    # Get GUID-level target labels (binary_target should be consistent per GUID)
    guid_targets = df.groupby('guid')['binary_target'].first()

    results = []

    for i in range(len(time_bins) - 1):
        bin_start, bin_end = time_bins[i], time_bins[i + 1]
        bin_center = (bin_start + bin_end) / 2

        # Get GUIDs with data available at time τ (bin_center)
        # Available means they have at least one epoch with epoch_hours >= bin_center
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

        # For each available GUID, check if detected (any clinical_pred=1 from τ to birth)
        # "From τ to birth" means epoch_hours >= bin_center
        detected_positive = 0
        for guid in available_positive_guids:
            guid_data = df[(df['guid'] == guid) & (df['epoch_hours'] >= bin_center)]
            if (guid_data['clinical_pred'] == 1).any():
                detected_positive += 1

        detected_negative = 0
        for guid in available_negative_guids:
            guid_data = df[(df['guid'] == guid) & (df['epoch_hours'] >= bin_center)]
            if (guid_data['clinical_pred'] == 1).any():
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

    logger.info(
        f"compute_committed_cumulative_metrics: Computed metrics for {len(result_df)} time bins "
        f"({result_df['sensitivity'].notna().sum()} non-NaN bins)"
    )

    # Verify monotonicity
    valid_sens = result_df['sensitivity'].dropna()
    if len(valid_sens) > 1:
        violations = (valid_sens.diff() < -1e-6).sum()
        if violations > 0:
            logger.warning(f"⚠️  Committed cumulative sensitivity has {violations} monotonicity violations!")
        else:
            logger.info("✓ Committed cumulative sensitivity is monotonically non-decreasing")

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
    - For GUIDs with data at time τ, check if detected (any clinical_pred=1 from τ to birth)
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

        # Get GUIDs with data available at time τ (bin_center)
        available_mask = df['epoch_hours'] >= bin_center
        available_guids = df[available_mask]['guid'].unique()

        # Split available GUIDs by target
        available_positive_guids = [g for g in available_guids if g in all_positive_guids]
        available_negative_guids = [g for g in available_guids if g in all_negative_guids]

        n_available_positive = len(available_positive_guids)
        n_available_negative = len(available_negative_guids)

        # For each available GUID, check if detected (any clinical_pred=1 from τ to birth)
        detected_positive = 0
        for guid in available_positive_guids:
            guid_data = df[(df['guid'] == guid) & (df['epoch_hours'] >= bin_center)]
            if (guid_data['clinical_pred'] == 1).any():
                detected_positive += 1

        detected_negative = 0
        for guid in available_negative_guids:
            guid_data = df[(df['guid'] == guid) & (df['epoch_hours'] >= bin_center)]
            if (guid_data['clinical_pred'] == 1).any():
                detected_negative += 1

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

    logger.info(
        f"compute_committed_overall_metrics: Computed PRIMARY METRIC for {len(result_df)} time bins "
        f"({result_df['sensitivity'].notna().sum()} non-NaN bins)"
    )

    # Verify monotonicity (CRITICAL for primary metric)
    valid_sens = result_df['sensitivity'].dropna()
    if len(valid_sens) > 1:
        violations = (valid_sens.diff() < -1e-6).sum()
        if violations > 0:
            logger.error(f"❌ PRIMARY METRIC violation: Committed overall sensitivity has {violations} monotonicity violations!")
            # Log details
            for idx in range(1, len(result_df)):
                prev_sens = result_df.iloc[idx - 1]['sensitivity']
                curr_sens = result_df.iloc[idx]['sensitivity']
                if pd.notna(prev_sens) and pd.notna(curr_sens) and curr_sens < prev_sens - 1e-6:
                    logger.error(
                        f"  Bin {idx} ({result_df.iloc[idx]['bin_center']:.1f}h): "
                        f"Sensitivity decreased from {prev_sens:.4f} to {curr_sens:.4f}"
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

def plot_single_metric_type(
    metrics_df: pd.DataFrame,
    metric_type: str,
    output_dir: Path,
    title_suffix: str = ""
) -> None:
    """
    Create comprehensive plots for a single metric type.

    Generates multiple plot variations:
    1. sensitivity_vs_time.png - Sensitivity only
    2. sensitivity_specificity_vs_time.png - Sensitivity + Specificity
    3. sensitivity_fpr_vs_time.png - Sensitivity + FPR
    4. all_metrics_vs_time.png - All three metrics together

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
    plt.tight_layout()
    plt.savefig(output_dir / "all_metrics_vs_time.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_dir.name}/all_metrics_vs_time.png")


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
    title_suffix: str = ""
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

    if len(subgroup_metrics) == 0:
        logger.warning(f"No valid subgroup metrics computed for {metric_type}")
        return {}

    # Create comparison plots by category
    _plot_diagnosis_comparison(subgroup_metrics, metric_type, output_dir, title_suffix)
    _plot_cs_stratification(subgroup_metrics, metric_type, output_dir, title_suffix)
    _plot_bg_stratification(subgroup_metrics, metric_type, output_dir, title_suffix)
    _plot_healthy_subgroups(subgroup_metrics, metric_type, output_dir, title_suffix)

    logger.info(f"Subgroup analysis plots generated for {metric_type}")
    return subgroup_metrics


def _plot_diagnosis_comparison(
    subgroup_metrics: Dict[str, pd.DataFrame],
    metric_type: str,
    output_dir: Path,
    title_suffix: str
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
            ax.plot(valid_df['bin_center'], valid_df['sensitivity'],
                   marker='o', label=group.capitalize(), linewidth=2.5,
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
    plt.tight_layout()
    plt.savefig(output_dir / "diagnosis_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_dir.name}/diagnosis_comparison.png")


def _plot_cs_stratification(
    subgroup_metrics: Dict[str, pd.DataFrame],
    metric_type: str,
    output_dir: Path,
    title_suffix: str
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
                label = 'CS Positive' if 'pos' in group else 'CS Negative'
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
        plt.tight_layout()
        filename = f"{groups[0].rsplit('_', 2)[0]}_cs_stratification.png"
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: {output_dir.name}/{filename}")


def _plot_bg_stratification(
    subgroup_metrics: Dict[str, pd.DataFrame],
    metric_type: str,
    output_dir: Path,
    title_suffix: str
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
            label = 'BG Positive' if 'pos' in group else 'BG Negative'
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
    plt.tight_layout()
    plt.savefig(output_dir / "acidosis_bg_stratification.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_dir.name}/acidosis_bg_stratification.png")


def _plot_healthy_subgroups(
    subgroup_metrics: Dict[str, pd.DataFrame],
    metric_type: str,
    output_dir: Path,
    title_suffix: str
) -> None:
    """Plot healthy subgroup stratifications (CS, BG, and combinations)."""
    import matplotlib.pyplot as plt

    # Healthy by CS
    cs_groups = ['healthy_cs_pos', 'healthy_cs_neg']
    available_cs = [g for g in cs_groups if g in subgroup_metrics]

    if len(available_cs) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = {'pos': '#3498db', 'neg': '#9b59b6'}

        for group in available_cs:
            df = subgroup_metrics[group]
            valid_df = df[df['sensitivity'].notna()].sort_values('bin_center', ascending=False)
            if len(valid_df) > 0:
                label = 'CS Positive' if 'pos' in group else 'CS Negative'
                color = colors['pos'] if 'pos' in group else colors['neg']
                ax.plot(valid_df['bin_center'], valid_df['sensitivity'],
                       marker='o', label=label, linewidth=2.5,
                       color=color, markersize=6)

        ax.set_xlabel('Hours Before Birth', fontsize=13)
        ax.set_ylabel('Sensitivity', fontsize=13)
        title = f'Healthy by CS Status - {metric_type.replace("_", " ").title()}'
        if title_suffix:
            title += f' ({title_suffix})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.invert_xaxis()
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
            valid_df = df[df['sensitivity'].notna()].sort_values('bin_center', ascending=False)
            if len(valid_df) > 0:
                label = 'BG Positive' if 'pos' in group else 'BG Negative'
                color = colors['pos'] if 'pos' in group else colors['neg']
                ax.plot(valid_df['bin_center'], valid_df['sensitivity'],
                       marker='o', label=label, linewidth=2.5,
                       color=color, markersize=6)

        ax.set_xlabel('Hours Before Birth', fontsize=13)
        ax.set_ylabel('Sensitivity', fontsize=13)
        title = f'Healthy by BG Status - {metric_type.replace("_", " ").title()}'
        if title_suffix:
            title += f' ({title_suffix})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.invert_xaxis()
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
            valid_df = df[df['sensitivity'].notna()].sort_values('bin_center', ascending=False)
            if len(valid_df) > 0:
                label = group.replace('healthy_', '').replace('_', ' ').upper()
                ax.plot(valid_df['bin_center'], valid_df['sensitivity'],
                       marker='o', label=label, linewidth=2.5,
                       color=colors[i % len(colors)], markersize=6)

        ax.set_xlabel('Hours Before Birth', fontsize=13)
        ax.set_ylabel('Sensitivity', fontsize=13)
        title = f'Healthy BG×CS Combinations - {metric_type.replace("_", " ").title()}'
        if title_suffix:
            title += f' ({title_suffix})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.invert_xaxis()
        plt.tight_layout()
        plt.savefig(output_dir / "healthy_bg_cs_combinations.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: {output_dir.name}/healthy_bg_cs_combinations.png")


# =============================================================================
# Complete Evaluation Pipeline Integration
# =============================================================================

def generate_three_metric_type_analysis(
    df: pd.DataFrame,
    output_base_dir: Path,
    exclude_last_minutes: float = 30.0,
    title_suffix: str = "Test Set"
) -> Dict:
    """
    Complete evaluation pipeline using three metric types.

    Generates:
    1. Three separate thresholds (instantaneous, committed_cumulative, committed_overall)
    2. Plots for each metric type
    3. Metric type comparison plots
    4. Subgroup analysis for committed_overall (PRIMARY)
    5. Saves all metrics as JSON

    Directory structure:
    output_base_dir/
        three_metric_types/
            instantaneous/
                sensitivity_vs_time.png
                ...
            committed_cumulative/
                ...
            committed_overall/  # PRIMARY
                ...
                subgroups/
                    diagnosis_comparison.png
                    ...
            comparison/
                metric_type_comparison.png
            thresholds.json
            metrics_summary.json

    Args:
        df: DataFrame with predictions and clinical_pred column
        output_base_dir: Base directory for all outputs
        exclude_last_minutes: Exclude last N minutes from analysis
        title_suffix: Suffix for plot titles

    Returns:
        Dictionary with thresholds and metrics for all three types
    """
    logger.info("=" * 80)
    logger.info("THREE METRIC TYPE ANALYSIS - NEW PIPELINE")
    logger.info("=" * 80)

    analysis_dir = output_base_dir / "three_metric_types"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Compute time bins
    time_bins = compute_time_bins(df, exclude_last_minutes=exclude_last_minutes)
    logger.info(f"Time bins: {len(time_bins)-1} bins from {time_bins[0]:.1f}h to {time_bins[-1]:.1f}h")

    # Step 1: Generate plots for all three metric types
    metrics_dict = plot_all_metric_types_for_fold(
        df, time_bins, analysis_dir, title_suffix
    )

    # Step 2: Generate metric type comparison
    comparison_dir = analysis_dir / "comparison"
    plot_metric_type_comparison(metrics_dict, comparison_dir, title_suffix)

    # Step 3: Generate subgroup analysis for PRIMARY metric (committed_overall)
    logger.info("Generating subgroup analysis for committed_overall (PRIMARY)...")
    subgroup_filters = create_enhanced_subgroup_filters()
    subgroup_dir = analysis_dir / "committed_overall" / "subgroups"

    subgroup_metrics = plot_subgroup_analysis(
        df, time_bins, 'committed_overall',
        subgroup_filters, subgroup_dir, title_suffix
    )

    # Step 4: Save metrics summary
    summary = {
        'metric_types': {
            'instantaneous': _summarize_metrics_df(metrics_dict.get('instantaneous')),
            'committed_cumulative': _summarize_metrics_df(metrics_dict.get('committed_cumulative')),
            'committed_overall': _summarize_metrics_df(metrics_dict.get('committed_overall'))
        },
        'subgroups': {
            name: _summarize_metrics_df(df)
            for name, df in subgroup_metrics.items()
        }
    }

    with open(analysis_dir / "metrics_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("Three metric type analysis complete")
    logger.info(f"Results saved to: {analysis_dir}")

    return {
        'metrics_dict': metrics_dict,
        'subgroup_metrics': subgroup_metrics,
        'summary': summary
    }


def _summarize_metrics_df(df: pd.DataFrame) -> Dict:
    """Helper to summarize a metrics DataFrame for JSON export."""
    if df is None or len(df) == 0:
        return {}

    valid_df = df[df['sensitivity'].notna()]
    if len(valid_df) == 0:
        return {}

    return {
        'n_bins': len(df),
        'n_valid_bins': len(valid_df),
        'sensitivity_mean': float(valid_df['sensitivity'].mean()),
        'sensitivity_std': float(valid_df['sensitivity'].std()),
        'sensitivity_min': float(valid_df['sensitivity'].min()),
        'sensitivity_max': float(valid_df['sensitivity'].max()),
        'fpr_mean': float(valid_df['fpr'].mean()) if 'fpr' in valid_df else None,
        'fpr_std': float(valid_df['fpr'].std()) if 'fpr' in valid_df else None,
    }


# ============================================================================
# MAIN FUNCTION - Post-Training Evaluation Pipeline
# ============================================================================

def main(
    output_base_dir: str,
    target_fpr: float = 0.15,
    device: str = 'cuda:0',
    exclude_last_minutes: float = 30.0,
    max_gap_multiplier: Optional[float] = None,
    regenerate_predictions: bool = False
):
    """
    Run evaluation pipeline on all completed folds.

    This function is designed for post-training evaluation where training
    has already completed and you want to (re)run the evaluation pipeline
    across all folds.

    Args:
        output_base_dir: Base directory containing fold_1, fold_2, ..., fold_N
                        subdirectories with trained models and configs
        target_fpr: Target false positive rate for threshold optimization
                   (default: 0.15 = 15% FPR)
        device: Device to run inference on (default: 'cuda:0')
        exclude_last_minutes: Exclude last N minutes before birth from time-based
                            analysis (default: 30.0 minutes)
        max_gap_multiplier: Maximum gap multiplier for epoch filling
                          (default: None = use config or auto-detect)
        regenerate_predictions: If True, regenerate predictions even if cached
                               predictions exist (default: False)

    Returns:
        Dictionary with aggregated results across all folds

    Directory Structure Expected:
        output_base_dir/
            fold_1/
                config.yaml
                checkpoints/
                    epoch=X-val_accuracy=Y.ckpt
                evaluation/  (will be created/updated)
            fold_2/
                ...
            ...
            aggregated_results.json  (will be created)

    Outputs Generated Per Fold:
        evaluation/
            validation_predictions_raw.csv
            validation_predictions_clinical.csv
            test_predictions_raw.csv
            test_predictions_clinical.csv
            threshold_info.json
            three_metric_types/
                instantaneous/
                    [4 plots per metric type]
                committed_cumulative/
                    [4 plots per metric type]
                committed_overall/  (PRIMARY METRIC)
                    [4 plots per metric type]
                    subgroups/
                        [8 subgroup comparison plots]
                comparison/
                    metric_type_comparison.png
                metrics_summary.json
            fold_results.json

    Outputs Generated Across Folds:
        aggregated_results.json - Summary statistics across all folds
    """
    output_base_dir = Path(output_base_dir)

    if not output_base_dir.exists():
        raise FileNotFoundError(f"Output base directory not found: {output_base_dir}")

    logger.info("="*80)
    logger.info("POST-TRAINING EVALUATION PIPELINE")
    logger.info("="*80)
    logger.info(f"Output base directory: {output_base_dir}")
    logger.info(f"Target FPR: {target_fpr}")
    logger.info(f"Device: {device}")
    logger.info(f"Exclude last minutes: {exclude_last_minutes}")
    logger.info(f"Regenerate predictions: {regenerate_predictions}")
    logger.info("="*80)

    # Find all fold directories (fold_1, fold_2, ..., fold_N)
    fold_dirs = sorted(
        [d for d in output_base_dir.iterdir() if d.is_dir() and d.name.startswith('fold_')],
        key=lambda x: int(x.name.split('_')[1])
    )

    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found in {output_base_dir}")

    logger.info(f"Found {len(fold_dirs)} fold directories: {[d.name for d in fold_dirs]}")

    # Process each fold
    all_fold_results = []
    successful_folds = []
    failed_folds = []

    for fold_dir in fold_dirs:
        fold_id = int(fold_dir.name.split('_')[1])
        logger.info("")
        logger.info("="*80)
        logger.info(f"Processing Fold {fold_id}")
        logger.info("="*80)

        try:
            # Evaluate single fold
            fold_results = _evaluate_single_fold(
                fold_dir=fold_dir,
                fold_id=fold_id,
                target_fpr=target_fpr,
                device=device,
                exclude_last_minutes=exclude_last_minutes,
                max_gap_multiplier=max_gap_multiplier,
                regenerate_predictions=regenerate_predictions
            )

            all_fold_results.append(fold_results)
            successful_folds.append(fold_id)
            logger.info(f"Fold {fold_id}: COMPLETED SUCCESSFULLY")

        except Exception as e:
            logger.error(f"Fold {fold_id}: FAILED with error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            failed_folds.append(fold_id)

    # Aggregate results across all successful folds
    logger.info("")
    logger.info("="*80)
    logger.info("AGGREGATING RESULTS ACROSS FOLDS")
    logger.info("="*80)

    if not all_fold_results:
        logger.error("No folds completed successfully. Cannot aggregate results.")
        return {
            'status': 'failed',
            'successful_folds': successful_folds,
            'failed_folds': failed_folds,
            'n_successful': 0,
            'n_failed': len(failed_folds)
        }

    aggregated = _aggregate_fold_results(all_fold_results)
    aggregated['successful_folds'] = successful_folds
    aggregated['failed_folds'] = failed_folds
    aggregated['n_successful'] = len(successful_folds)
    aggregated['n_failed'] = len(failed_folds)

    # Save aggregated results
    aggregated_path = output_base_dir / "aggregated_results.json"
    with open(aggregated_path, 'w') as f:
        json.dump(aggregated, f, indent=2)

    logger.info(f"Aggregated results saved to: {aggregated_path}")

    # Print summary
    logger.info("")
    logger.info("="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"Total folds: {len(fold_dirs)}")
    logger.info(f"Successful: {len(successful_folds)} {successful_folds}")
    logger.info(f"Failed: {len(failed_folds)} {failed_folds}")
    logger.info("")
    logger.info("PRIMARY METRICS (committed_overall) - Aggregated Across Folds:")
    logger.info(f"  Validation Sensitivity: {aggregated['validation_sensitivity_mean']:.3f} +/- {aggregated['validation_sensitivity_std']:.3f}")
    logger.info(f"  Validation Specificity: {aggregated['validation_specificity_mean']:.3f} +/- {aggregated['validation_specificity_std']:.3f}")
    logger.info(f"  Validation FPR: {aggregated['validation_fpr_mean']:.3f} +/- {aggregated['validation_fpr_std']:.3f}")
    logger.info(f"  Test Sensitivity: {aggregated['test_sensitivity_mean']:.3f} +/- {aggregated['test_sensitivity_std']:.3f}")
    logger.info(f"  Test Specificity: {aggregated['test_specificity_mean']:.3f} +/- {aggregated['test_specificity_std']:.3f}")
    logger.info(f"  Test FPR: {aggregated['test_fpr_mean']:.3f} +/- {aggregated['test_fpr_std']:.3f}")
    logger.info("="*80)

    return aggregated


def _evaluate_single_fold(
    fold_dir: Path,
    fold_id: int,
    target_fpr: float,
    device: str,
    exclude_last_minutes: float,
    max_gap_multiplier: Optional[float],
    regenerate_predictions: bool
) -> Dict:
    """
    Evaluate a single fold.

    Args:
        fold_dir: Path to fold directory
        fold_id: Fold ID number
        target_fpr: Target false positive rate
        device: Device to run inference on
        exclude_last_minutes: Minutes to exclude from analysis
        max_gap_multiplier: Maximum gap multiplier for epoch filling
        regenerate_predictions: Whether to regenerate predictions

    Returns:
        Dictionary with fold results
    """
    # Load config
    config_path = fold_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Fold {fold_id}: Loaded config from {config_path}")

    # Create evaluation directory
    evaluation_dir = fold_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Check for cached predictions
    val_raw_path = evaluation_dir / "validation_predictions_raw.csv"
    test_raw_path = evaluation_dir / "test_predictions_raw.csv"

    use_cached_val = val_raw_path.exists() and not regenerate_predictions
    use_cached_test = test_raw_path.exists() and not regenerate_predictions

    # Load or generate validation predictions
    if use_cached_val:
        logger.info(f"Fold {fold_id}: Loading cached validation predictions from {val_raw_path}")
        val_df_raw = pd.read_csv(val_raw_path)
    else:
        logger.info(f"Fold {fold_id}: Generating validation predictions...")
        val_df_raw = _run_inference_for_fold(
            fold_dir=fold_dir,
            config=config,
            device=device,
            split='val'
        )
        val_df_raw.to_csv(val_raw_path, index=False)
        logger.info(f"Fold {fold_id}: Validation predictions saved ({len(val_df_raw)} rows)")

    # Find PRIMARY threshold on validation set
    logger.info(f"Fold {fold_id}: Finding PRIMARY threshold (target_fpr={target_fpr})...")
    primary_threshold, threshold_metrics = find_threshold_for_committed_overall_fpr_at_1h(
        val_df_raw,
        target_fpr=target_fpr,
        time_window_hours=1.0,
        fallback_tolerance_hours=0.5
    )

    logger.info(f"Fold {fold_id}: PRIMARY threshold = {primary_threshold:.4f}")
    logger.info(f"  Validation sensitivity: {threshold_metrics.get('sensitivity', 0):.3f}")
    logger.info(f"  Validation specificity: {threshold_metrics.get('specificity', 0):.3f}")
    logger.info(f"  Validation FPR: {threshold_metrics.get('fpr', 0):.3f}")

    # Apply clinical decision rule to validation set
    val_df_clinical = apply_clinical_decision_rule(val_df_raw.copy(), primary_threshold)
    val_df_clinical = fill_missing_epochs(val_df_clinical, max_gap_multiplier=max_gap_multiplier)
    val_df_clinical.to_csv(evaluation_dir / "validation_predictions_clinical.csv", index=False)

    # Load or generate test predictions
    if use_cached_test:
        logger.info(f"Fold {fold_id}: Loading cached test predictions from {test_raw_path}")
        test_df_raw = pd.read_csv(test_raw_path)
    else:
        logger.info(f"Fold {fold_id}: Generating test predictions...")
        test_df_raw = _run_inference_for_fold(
            fold_dir=fold_dir,
            config=config,
            device=device,
            split='test'
        )
        test_df_raw.to_csv(test_raw_path, index=False)
        logger.info(f"Fold {fold_id}: Test predictions saved ({len(test_df_raw)} rows)")

    # Apply clinical decision rule to test set
    test_df_clinical = apply_clinical_decision_rule(test_df_raw.copy(), primary_threshold)
    test_df_clinical = fill_missing_epochs(test_df_clinical, max_gap_multiplier=max_gap_multiplier)
    test_df_clinical.to_csv(evaluation_dir / "test_predictions_clinical.csv", index=False)

    # Generate three metric type analysis
    logger.info(f"Fold {fold_id}: Generating three metric type analysis...")
    try:
        three_metric_results = generate_three_metric_type_analysis(
            test_df_clinical,
            output_base_dir=evaluation_dir,
            exclude_last_minutes=exclude_last_minutes,
            title_suffix=f"Fold {fold_id}"
        )
        logger.info(f"Fold {fold_id}: Three metric type analysis complete")
    except Exception as e:
        logger.warning(f"Fold {fold_id}: Three metric type analysis failed: {e}")
        three_metric_results = {}

    # Extract PRIMARY metrics from three metric type results
    primary_metrics = {}
    if three_metric_results and 'summary' in three_metric_results:
        primary_summary = three_metric_results['summary'].get('metric_types', {}).get('committed_overall', {})
        primary_metrics = {
            'test_sensitivity_mean': primary_summary.get('sensitivity_mean', 0.0),
            'test_sensitivity_std': primary_summary.get('sensitivity_std', 0.0),
            'test_specificity_mean': primary_summary.get('specificity_mean', 0.0),
            'test_specificity_std': primary_summary.get('specificity_std', 0.0),
            'test_fpr_mean': primary_summary.get('fpr_mean', 0.0),
            'test_fpr_std': primary_summary.get('fpr_std', 0.0),
        }

    # Save threshold and metrics info
    threshold_info = {
        'primary_threshold': float(primary_threshold),
        'target_fpr': float(target_fpr),
        'validation_metrics': {
            'sensitivity': float(threshold_metrics.get('sensitivity', 0)),
            'specificity': float(threshold_metrics.get('specificity', 0)),
            'fpr': float(threshold_metrics.get('fpr', 0)),
            'accuracy': float(threshold_metrics.get('accuracy', 0)),
            'time_window_hours': float(threshold_metrics.get('time_window_hours', 1.0)),
        },
        'test_metrics_primary': primary_metrics,
    }

    with open(evaluation_dir / "threshold_info.json", 'w') as f:
        json.dump(threshold_info, f, indent=2)

    # Create fold results
    fold_results = {
        'fold_id': fold_id,
        'primary_threshold': float(primary_threshold),
        'validation_sensitivity': float(threshold_metrics.get('sensitivity', 0)),
        'validation_specificity': float(threshold_metrics.get('specificity', 0)),
        'validation_fpr': float(threshold_metrics.get('fpr', 0)),
        'validation_accuracy': float(threshold_metrics.get('accuracy', 0)),
        'test_sensitivity_mean': primary_metrics.get('test_sensitivity_mean', 0.0),
        'test_sensitivity_std': primary_metrics.get('test_sensitivity_std', 0.0),
        'test_specificity_mean': primary_metrics.get('test_specificity_mean', 0.0),
        'test_specificity_std': primary_metrics.get('test_specificity_std', 0.0),
        'test_fpr_mean': primary_metrics.get('test_fpr_mean', 0.0),
        'test_fpr_std': primary_metrics.get('test_fpr_std', 0.0),
        'status': 'success'
    }

    # Save fold results
    results_path = fold_dir / "fold_results.json"
    with open(results_path, 'w') as f:
        json.dump(fold_results, f, indent=2)

    logger.info(f"Fold {fold_id}: Results saved to {results_path}")

    return fold_results


def _run_inference_for_fold(
    fold_dir: Path,
    config: Dict,
    device: str,
    split: str
) -> pd.DataFrame:
    """
    Run inference for a specific split (val/test) of a fold.

    Args:
        fold_dir: Path to fold directory
        config: Configuration dictionary
        device: Device to run inference on
        split: Data split ('val' or 'test')

    Returns:
        DataFrame with predictions
    """
    # Find checkpoint
    checkpoint_path = find_latest_checkpoint_in_fold(fold_dir)
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    # Create model
    model = create_model_from_config(config, device=device)

    # Load checkpoint state
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']

    # Remove 'model.' prefix from keys (LightningModule wraps the model)
    model_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]
            model_state_dict[new_key] = value

    model.load_state_dict(model_state_dict, strict=True)
    model.eval()

    # Get dataset configuration
    fold_datasets = config.get('fold_datasets', {})
    if split not in fold_datasets:
        raise ValueError(f"Split '{split}' not found in fold_datasets config")

    hdf5_files = fold_datasets[split]

    # Get dataloader configuration
    dataloader_config = config.get('general_config', {}).get('dataloader', {})
    batch_size = config.get('general_config', {}).get('batch_size', {}).get(split, 32)

    # Get normalization configuration
    stats_path = config.get('general_config', {}).get('stats_path')
    normalized_fields = config.get('general_config', {}).get('normalized_fields', [])

    # Additional dataset kwargs
    dataset_config = config.get('model_config', {}).get('dataset', {})
    dataset_kwargs = {
        'target_label_name': dataset_config.get('target_label_name', 'target'),
        'epoch_hour_field': dataset_config.get('epoch_hour_field', 'epoch_hours'),
        'guid_field': dataset_config.get('guid_field', 'guid'),
        'max_epochs': dataset_config.get('max_epochs', None),
        'sequence_fields': dataset_config.get('sequence_fields', []),
        'pad_sequences': dataset_config.get('pad_sequences', True),
    }

    # Create dataloader
    dataloader = create_optimized_dataloader(
        hdf5_files=hdf5_files,
        batch_size=batch_size,
        num_workers=dataloader_config.get('num_workers', 4),
        shuffle=False,
        stats_path=stats_path,
        normalize_fields=normalized_fields,
        pin_memory=True,
        rank=0,
        world_size=1,
        **dataset_kwargs
    )

    # Run inference
    predictions_df = run_inference(model, dataloader, device=device)

    return predictions_df


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

        # Individual fold results
        'fold_results': all_fold_results,
    }

    return aggregated


if __name__ == '__main__':
    # Example usage for post-training evaluation
    # Modify these parameters as needed for your setup

    OUTPUT_BASE_DIR = r"C:\Users\mahdi\Desktop\teb_vae_model\outputs\kfold_results"
    TARGET_FPR = 0.15  # 15% FPR target
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    EXCLUDE_LAST_MINUTES = 30.0  # Exclude last 30 minutes before birth
    REGENERATE_PREDICTIONS = False  # Set to True to regenerate all predictions

    # Run evaluation pipeline
    results = main(
        output_base_dir=OUTPUT_BASE_DIR,
        target_fpr=TARGET_FPR,
        device=DEVICE,
        exclude_last_minutes=EXCLUDE_LAST_MINUTES,
        regenerate_predictions=REGENERATE_PREDICTIONS
    )

    print("\nEvaluation pipeline completed!")
    print(f"Results saved to: {OUTPUT_BASE_DIR}/aggregated_results.json")


