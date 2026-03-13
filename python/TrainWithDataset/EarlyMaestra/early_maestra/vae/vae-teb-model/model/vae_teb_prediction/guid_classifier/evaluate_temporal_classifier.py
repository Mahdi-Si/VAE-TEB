"""Evaluation pipeline for the temporal VAE classifier.

Provides inference, model loading, and single-fold evaluation functions for the
:class:`TemporalVaeClassifier`.  Reuses the existing evaluation utilities
(threshold optimisation, clinical decision rule, metrics, plots) from
``evaluate_classifier.py`` — the temporal model produces an identical CSV
format so all downstream code works unchanged.

Typical single-fold evaluation::

    results = evaluate_single_fold_temporal(
        fold_dir="/path/to/fold_1",
        config=config_dict,
        device="cuda:0",
    )
"""

from __future__ import annotations

import gc
import json
import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from loguru import logger

from train.graph_models_utils import (
    load_checkpoint_strict,
    _prepare_checkpoint_state_dict,
)

# NOTE: Imports from evaluate_classifier are LAZY (inside functions) because
# that module imports SeqVae at the top level which triggers
# ``from model.model_utils import *`` — a module that doesn't exist in all
# environments.  Lazy importing avoids this broken chain.


# ---------------------------------------------------------------------------
#  Model creation from config
# ---------------------------------------------------------------------------


def create_temporal_model_from_config(
    config: Dict,
    device: str = "cuda:0",
) -> nn.Module:
    """Create a :class:`TemporalVaeClassifier` from a config dict.

    Loads the frozen VAE from its checkpoint, builds the temporal model with
    the same architecture parameters used during training, and moves the
    model to the specified device in eval mode.

    Args:
        config: Full config dict (as loaded from ``config_temporal.yaml``).
        device: Target device string (e.g. ``'cuda:0'`` or ``'cpu'``).

    Returns:
        :class:`TemporalVaeClassifier` in eval mode on *device*.
    """
    # Lazy imports to avoid broken dependency chain (model_utils)
    from model.vae_teb_prediction.vae_teb_model_prediction import SeqVae
    from model.vae_teb_prediction.guid_classifier.temporal_classification_model import (
        TemporalVaeClassifier,
    )

    model_cfg = config.get("model_config", {})

    # 1. Load frozen VAE
    vae_checkpoint = model_cfg.get("vae_checkpoint")
    if vae_checkpoint is None:
        raise ValueError("vae_checkpoint must be provided in model_config")

    vae_model = SeqVae()
    loaded_vae = load_checkpoint_strict(vae_model, checkpoint=vae_checkpoint)
    if loaded_vae is None:
        raise RuntimeError(
            "Strict VAE checkpoint loading failed during temporal evaluation. "
            f"Checkpoint: {vae_checkpoint}"
        )
    logger.info("Evaluation: VAE loaded from {}", vae_checkpoint)

    # 2. Build TemporalVaeClassifier with matching architecture
    seg_cfg = model_cfg.get("segment_encoder", {})
    lstm_cfg = model_cfg.get("temporal_lstm", {})
    feat_cfg = model_cfg.get("temporal_features", {})
    head_cfg = model_cfg.get("classifier_head", {})
    seg_idx_cfg = feat_cfg.get("segment_index", {})
    tlo_cfg = feat_cfg.get("time_from_labor_onset", {})
    dt_cfg = feat_cfg.get("delta_t", {})

    model = TemporalVaeClassifier(
        vae_model=vae_model,
        segment_encoder_type=seg_cfg.get("type", "mean_pool"),
        d_seg=seg_cfg.get("d_seg", 64),
        temporal_lstm_hidden=lstm_cfg.get("hidden_dim", 128),
        temporal_lstm_layers=lstm_cfg.get("num_layers", 2),
        temporal_lstm_dropout=lstm_cfg.get("dropout", 0.1),
        gap_encoding=model_cfg.get("gap_encoding", "concat"),
        position_embed_dim=(
            seg_idx_cfg.get("embed_dim", 8)
            if seg_idx_cfg.get("enabled", False)
            else 0
        ),
        max_position_index=seg_idx_cfg.get("max_index", 40),
        tlo_enabled=tlo_cfg.get("enabled", False),
        tlo_embed_dim=tlo_cfg.get("embed_dim", 0),
        tlo_dropout=tlo_cfg.get("dropout", 0.1),
        delta_t_embed_dim=dt_cfg.get("embed_dim", 0),
        delta_t_dropout=dt_cfg.get("dropout", 0.1),
        persist_segment_state=seg_cfg.get("persist_state", False),
        segment_state_decay=seg_cfg.get("state_decay", True),
        num_classes=head_cfg.get("num_classes", 2),
        classifier_dropout=head_cfg.get("dropout", 0.1),
        mlp_multiplier=head_cfg.get("mlp_multiplier", 2.0),
        vae_chunk_size=model_cfg.get("vae_chunk_size", 32),
        use_posterior=model_cfg.get("use_posterior", True),
        freeze_vae=model_cfg.get("freeze_vae", True),
        cnn_kernel=seg_cfg.get("cnn_kernel", 7),
    )

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
#  Temporal inference
# ---------------------------------------------------------------------------


def run_temporal_inference(
    model: nn.Module,
    dataloader,
    device: str = "cuda:0",
) -> pd.DataFrame:
    """Run inference with the temporal model, producing per-segment predictions.

    Iterates over GUID-sequence batches from a bucketed sequence DataLoader,
    unpacks ``(B, S_max, 2)`` probabilities into individual per-segment rows
    using ``mask`` / ``lengths``.

    The output format is **identical** to
    :func:`evaluate_classifier.run_inference` for full backward compatibility
    with threshold optimisation, clinical decision rule, metrics, and
    plotting.

    Args:
        model: :class:`TemporalVaeClassifier` in eval mode.
        dataloader: Bucketed sequence DataLoader yielding
            ``sequence_collate_fn`` batches.
        device: CUDA device string.

    Returns:
        DataFrame with columns: ``guid``, ``epoch``, ``target``,
        ``binary_target``, ``predicted_class``, ``prob_class_0``,
        ``prob_class_1``, ``cs_label``, ``bg_label``, ``tlo_hours``.
    """
    model.eval()
    model.to(device)

    rows: List[Dict] = []

    with torch.no_grad():
        for batch in dataloader:
            # Move tensors to device
            batch_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(device)
                else:
                    batch_device[k] = v

            outputs = model(batch_device)
            probs = outputs["probs"]       # (B, S_max, 2)
            mask = batch_device["mask"]    # (B, S_max) bool
            lengths = batch_device["lengths"]  # (B,)
            target = batch_device["target"]    # (B, S_max, 300)
            epoch_val = batch_device["epoch"]  # (B, S_max)

            # TLO — may contain NaN
            tlo = batch_device.get("time_from_labor_onset")  # (B, S_max) or None

            B = len(batch_device["guid"])
            for i in range(B):
                L = lengths[i].item()
                for j in range(L):
                    seg_target = target[i, j].max().item()
                    tlo_val = float("nan")
                    if tlo is not None:
                        tlo_val = tlo[i, j].item() / 3600.0  # NaN preserved

                    rows.append({
                        "guid": batch_device["guid"][i],
                        "epoch": float(epoch_val[i, j].item()),
                        "cs_label": bool(batch_device["cs_label"][i].item()),
                        "bg_label": bool(batch_device["bg_label"][i].item()),
                        "target": int(seg_target),
                        "binary_target": int(seg_target > 1),
                        "predicted_class": int(probs[i, j].argmax().item()),
                        "prob_class_0": float(probs[i, j, 0].item()),
                        "prob_class_1": float(probs[i, j, 1].item()),
                        "tlo_hours": tlo_val,
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Single-fold evaluation
# ---------------------------------------------------------------------------


def _load_temporal_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: str = "cpu",
    allow_backward_compat: bool = False,
) -> nn.Module:
    """Load a Lightning-saved temporal checkpoint into the model.

    The checkpoint is produced by Lightning's ``ModelCheckpoint`` callback
    and contains the full ``PlTemporalClassifier`` state dict.  We strip
    the ``model.`` prefix that Lightning adds.  VAE keys from the temporal
    checkpoint are ignored because evaluation always uses the explicit VAE
    checkpoint declared in config.

    By default, temporal keys must match the current model exactly.
    Missing, unexpected, or shape-mismatched non-VAE keys raise immediately.
    Set ``allow_backward_compat=True`` to permit partial loading for older
    checkpoints, in which case incompatible non-VAE keys are skipped and left
    randomly initialised.

    Args:
        model: :class:`TemporalVaeClassifier` to load weights into.
        checkpoint_path: Path to ``.ckpt`` file.
        device: Map location for ``torch.load``.
        allow_backward_compat: If ``True``, allow partial loading of temporal
            keys for older checkpoints.  Default ``False``.

    Returns:
        The model with loaded weights.
    """
    cleaned = _prepare_checkpoint_state_dict(checkpoint_path, map_location=device)
    if cleaned is None:
        raise RuntimeError(
            "Unable to extract a normalized state_dict from the temporal "
            f"checkpoint: {checkpoint_path}"
        )

    # Filter out VAE keys (evaluation uses explicit VAE checkpoint from config)
    ignored_vae_keys = sorted(k for k in cleaned if k.startswith("vae_model."))
    # Filter out training-only buffers not needed for inference
    _INFERENCE_IGNORED_KEYS = {"class_weights"}
    ignored_training_keys = sorted(k for k in cleaned if k in _INFERENCE_IGNORED_KEYS)
    temporal_state = {
        k: v for k, v in cleaned.items()
        if not k.startswith("vae_model.") and k not in _INFERENCE_IGNORED_KEYS
    }

    model_state = model.state_dict()
    expected_temporal = {
        k: v for k, v in model_state.items()
        if not k.startswith("vae_model.")
    }

    checkpoint_keys = set(temporal_state.keys())
    expected_keys = set(expected_temporal.keys())

    missing_temporal = sorted(expected_keys - checkpoint_keys)
    unexpected_temporal = sorted(checkpoint_keys - expected_keys)

    shape_mismatches = []
    for key in sorted(expected_keys & checkpoint_keys):
        if expected_temporal[key].shape != temporal_state[key].shape:
            shape_mismatches.append(
                (
                    key,
                    tuple(temporal_state[key].shape),
                    tuple(expected_temporal[key].shape),
                )
            )

    if (missing_temporal or unexpected_temporal or shape_mismatches) and not allow_backward_compat:
        mismatch_parts = []
        if missing_temporal:
            mismatch_parts.append(f"missing={missing_temporal}")
        if unexpected_temporal:
            mismatch_parts.append(f"unexpected={unexpected_temporal}")
        if shape_mismatches:
            mismatch_parts.append(
                "shape_mismatches="
                + str(
                    [
                        {
                            "key": key,
                            "checkpoint_shape": ckpt_shape,
                            "model_shape": model_shape,
                        }
                        for key, ckpt_shape, model_shape in shape_mismatches
                    ]
                )
            )
        raise RuntimeError(
            "Temporal checkpoint is incompatible with the current model. "
            "Non-VAE parameter mismatches are not allowed by default. "
            "Re-run evaluation with allow_backward_compat=True only if you "
            "intentionally want partial loading of an older checkpoint. "
            + " ".join(mismatch_parts)
        )

    if allow_backward_compat:
        for key in unexpected_temporal:
            temporal_state.pop(key, None)
        for key, _, _ in shape_mismatches:
            temporal_state.pop(key, None)

    missing, unexpected = model.load_state_dict(temporal_state, strict=False)
    non_vae_missing = sorted(k for k in missing if not k.startswith("vae_model."))
    non_vae_unexpected = sorted(k for k in unexpected if not k.startswith("vae_model."))

    if (non_vae_missing or non_vae_unexpected) and not allow_backward_compat:
        raise RuntimeError(
            "Temporal checkpoint load left non-VAE parameters unresolved. "
            f"missing={non_vae_missing} unexpected={non_vae_unexpected}"
        )

    if allow_backward_compat:
        if missing_temporal:
            logger.warning(
                "Backward-compatible checkpoint load: missing temporal keys "
                "left at model init values: {}",
                missing_temporal,
            )
        if unexpected_temporal:
            logger.warning(
                "Backward-compatible checkpoint load: unexpected temporal "
                "keys ignored: {}",
                unexpected_temporal,
            )
        if shape_mismatches:
            logger.warning(
                "Backward-compatible checkpoint load: shape-mismatched "
                "temporal keys ignored: {}",
                [
                    {
                        "key": key,
                        "checkpoint_shape": ckpt_shape,
                        "model_shape": model_shape,
                    }
                    for key, ckpt_shape, model_shape in shape_mismatches
                ],
            )
        if non_vae_missing:
            logger.warning(
                "Backward-compatible checkpoint load: unresolved temporal "
                "keys after load: {}",
                non_vae_missing,
            )
        if non_vae_unexpected:
            logger.warning(
                "Backward-compatible checkpoint load: unexpected temporal "
                "keys reported by load_state_dict: {}",
                non_vae_unexpected,
            )

    logger.info(
        "Temporal checkpoint loaded from {} "
        "(ignored VAE keys: {}, ignored training-only keys: {}, "
        "compat_mode: {}, temporal_missing: {}, "
        "temporal_unexpected: {}, temporal_shape_mismatches: {})",
        checkpoint_path,
        len(ignored_vae_keys),
        len(ignored_training_keys),
        allow_backward_compat,
        len(missing_temporal),
        len(unexpected_temporal),
        len(shape_mismatches),
    )
    return model


def evaluate_single_fold_temporal(
    fold_dir: str,
    config: Dict,
    device: str = "cuda:0",
    target_fpr: float = 0.2,
    exclude_last_minutes: float = 30.0,
    decision_time_hours: float = 1.0,
    max_gap_multiplier: Optional[float] = None,
    regenerate_predictions: bool = False,
    allow_backward_compat: bool = False,
) -> Dict:
    """Full evaluation of one trained temporal fold.

    Loads the best checkpoint, runs inference on validation and test sets,
    finds an optimal threshold on validation, applies the clinical decision
    rule, and generates three-metric-type plots.

    Args:
        fold_dir: Path to ``fold_N/`` directory containing ``checkpoints/``
            and ``config.yaml``.
        config: Full config dict from ``config_temporal.yaml``.
        device: CUDA device string.
        target_fpr: Target false-positive rate for threshold optimisation.
        exclude_last_minutes: Minutes to exclude from time-based analysis.
        decision_time_hours: Decision time in hours before delivery.
        max_gap_multiplier: Gap multiplier for epoch filling (``None`` = fill
            all).
        regenerate_predictions: If ``True``, regenerate predictions even when
            cached CSVs exist.
        allow_backward_compat: If ``True``, allow partial loading of older
            temporal checkpoints with missing or mismatched non-VAE keys.

    Returns:
        Dict with fold results including ``fold_id``, ``primary_threshold``,
        ``validation_sensitivity``, ``test_sensitivity_mean``, etc.
    """
    # Lazy import — evaluate_classifier imports SeqVae which triggers model_utils
    from model.vae_teb_prediction.evaluate_classifier import (
        apply_clinical_decision_rule,
        compute_guid_level_roc,
        convert_numpy_types,
        fill_missing_epochs,
        find_latest_checkpoint_in_fold,
        find_threshold_for_committed_cumulative_fpr_at_1h,
        find_threshold_for_committed_overall_fpr_at_1h,
        find_threshold_for_instantaneous_fpr_at_1h,
        generate_three_metric_type_analysis,
        plot_roc_curve,
    )

    fold_dir = Path(fold_dir)
    fold_id = int(fold_dir.name.split("_")[1])

    logger.info("=" * 80)
    logger.info("Evaluating temporal fold {} at {}", fold_id, fold_dir)
    logger.info("=" * 80)

    # --- Evaluation directory ------------------------------------------------
    evaluation_dir = fold_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # --- Find best checkpoint -----------------------------------------------
    checkpoint_path = find_latest_checkpoint_in_fold(fold_dir)
    logger.info("Fold {}: using checkpoint {}", fold_id, checkpoint_path)

    # --- Create model and load weights --------------------------------------
    model = create_temporal_model_from_config(config, device=device)
    model = _load_temporal_checkpoint(
        model,
        checkpoint_path,
        device=device,
        allow_backward_compat=allow_backward_compat,
    )
    model.to(device)
    model.eval()

    # --- Data loaders -------------------------------------------------------
    from model.vae_teb_prediction.kfold_classifier_trainer import get_fold_datasets
    from model.vae_teb_prediction.guid_classifier.length_bucket_sampler import (
        create_bucketed_sequence_dataloader,
    )

    dataset_cfg = config.get("dataset_config", {})
    kfold_base_path = dataset_cfg["kfold_base_path"]
    fold_datasets = get_fold_datasets(kfold_base_path, fold_id)

    dataloader_cfg = dataset_cfg.get("dataloader_config", {})
    dataset_kwargs = dataloader_cfg.get("dataset_kwargs", {})
    stat_path = dataset_cfg.get("stat_path")
    normalize_fields = dataloader_cfg.get("normalize_fields")
    bucket_cfg = dataset_cfg.get("bucket_sampler", {})
    batch_size_test = config["general_config"]["batch_size"]["test"]

    common_dl_kwargs = dict(
        num_workers=dataloader_cfg.get("num_workers", 0),
        segment_duration=dataloader_cfg.get("segment_duration", 1200.0),
        guid_cache_size=dataloader_cfg.get("guid_cache_size", 128),
        stats_path=stat_path,
        normalize_fields=normalize_fields,
        prefetch_factor=dataloader_cfg.get("prefetch_factor", 2),
        seed=42,
        **dataset_kwargs,
    )

    # --- Validation predictions ----------------------------------------------
    val_raw_path = evaluation_dir / "validation_predictions_raw.csv"
    if val_raw_path.exists() and not regenerate_predictions:
        logger.info("Fold {}: Loading cached validation predictions", fold_id)
        val_df_raw = pd.read_csv(val_raw_path)
    else:
        logger.info("Fold {}: Running validation inference...", fold_id)
        val_loader, _ = create_bucketed_sequence_dataloader(
            hdf5_files=fold_datasets["val"],
            batch_size=batch_size_test,
            bucket_ranges=bucket_cfg.get("bucket_ranges"),
            shuffle=False,
            **common_dl_kwargs,
        )
        val_df_raw = run_temporal_inference(model, val_loader, device=device)
        val_df_raw.to_csv(val_raw_path, index=False)
        logger.info("Fold {}: Validation predictions saved ({} rows)", fold_id, len(val_df_raw))

    # --- Find thresholds on validation set (one per metric type) -------------
    logger.info(
        "Fold {}: Finding thresholds (target_fpr={}, time={}h)...",
        fold_id, target_fpr, decision_time_hours,
    )

    # PRIMARY — committed_overall
    primary_threshold, threshold_metrics = find_threshold_for_committed_overall_fpr_at_1h(
        val_df_raw,
        target_fpr=target_fpr,
        time_window_hours=decision_time_hours,
        max_gap_multiplier=max_gap_multiplier,
        fallback_tolerance_hours=0.5,
    )

    # Committed cumulative
    threshold_cumulative, metrics_cumulative = find_threshold_for_committed_cumulative_fpr_at_1h(
        val_df_raw,
        target_fpr=target_fpr,
        time_window_hours=decision_time_hours,
        max_gap_multiplier=max_gap_multiplier,
        fallback_tolerance_hours=0.5,
    )

    # Instantaneous
    threshold_instantaneous, metrics_instantaneous = find_threshold_for_instantaneous_fpr_at_1h(
        val_df_raw,
        target_fpr=target_fpr,
        time_window_hours=decision_time_hours,
        max_gap_multiplier=max_gap_multiplier,
        fallback_tolerance_hours=0.5,
    )

    all_thresholds = {
        'instantaneous': threshold_instantaneous,
        'committed_cumulative': threshold_cumulative,
        'committed_overall': primary_threshold,
    }

    # Compute accuracy from sensitivity/specificity (primary)
    n_pos = threshold_metrics.get("n_positive_total", threshold_metrics.get("n_available_positive", 1))
    n_neg = threshold_metrics.get("n_negative_total", threshold_metrics.get("n_available_negative", 1))
    sens = threshold_metrics.get("sensitivity", 0)
    spec = threshold_metrics.get("specificity", 0)
    threshold_metrics["accuracy"] = (
        float((sens * n_pos + spec * n_neg) / (n_pos + n_neg))
        if (n_pos + n_neg) > 0
        else 0.0
    )

    logger.info(
        "Fold {}: Thresholds — overall={:.4f}, cumulative={:.4f}, instantaneous={:.4f}",
        fold_id, primary_threshold, threshold_cumulative, threshold_instantaneous,
    )

    # Apply CDR to validation set (primary threshold for backward compat)
    val_df_clinical = apply_clinical_decision_rule(val_df_raw.copy(), primary_threshold)
    val_df_clinical = fill_missing_epochs(val_df_clinical, max_gap_multiplier=max_gap_multiplier)
    val_df_clinical.to_csv(evaluation_dir / "validation_predictions_clinical.csv", index=False)

    # --- Validation three-metric-type analysis (verify thresholds) ----------
    logger.info("Fold {}: Generating validation three metric type analysis...", fold_id)
    val_three_metric_results = {}
    try:
        val_three_metric_results = generate_three_metric_type_analysis(
            val_df_raw,
            thresholds=all_thresholds,
            output_base_dir=evaluation_dir / "validation_evaluation",
            exclude_last_minutes=exclude_last_minutes,
            title_suffix=f"Temporal Fold {fold_id} — Validation",
            max_gap_multiplier=max_gap_multiplier,
            decision_time_hours=decision_time_hours,
        )
        # Log validation FPR at decision point for each metric type
        val_dp = val_three_metric_results.get("decision_point_metrics", {})
        for mt in ("committed_overall", "committed_cumulative", "instantaneous"):
            mt_dp = val_dp.get(mt, {})
            fpr_val = mt_dp.get("fpr_at_decision", "N/A")
            sens_val = mt_dp.get("sensitivity_at_decision", "N/A")
            logger.info(
                "Fold {}: VALIDATION {} — FPR@{}h={}, Sens@{}h={} (target_fpr={})",
                fold_id, mt, decision_time_hours, fpr_val,
                decision_time_hours, sens_val, target_fpr,
            )
        logger.info("Fold {}: Validation three metric type analysis complete", fold_id)
    except Exception as e:
        logger.warning("Fold {}: Validation three metric type analysis failed: {}", fold_id, e)

    val_decision_point = (
        val_three_metric_results.get("decision_point_metrics", {})
        if val_three_metric_results else {}
    )

    # --- Test predictions ---------------------------------------------------
    test_raw_path = evaluation_dir / "test_predictions_raw.csv"
    if test_raw_path.exists() and not regenerate_predictions:
        logger.info("Fold {}: Loading cached test predictions", fold_id)
        test_df_raw = pd.read_csv(test_raw_path)
    else:
        logger.info("Fold {}: Running test inference...", fold_id)
        test_loader, _ = create_bucketed_sequence_dataloader(
            hdf5_files=fold_datasets["test"],
            batch_size=batch_size_test,
            bucket_ranges=bucket_cfg.get("bucket_ranges"),
            shuffle=False,
            **common_dl_kwargs,
        )
        test_df_raw = run_temporal_inference(model, test_loader, device=device)
        test_df_raw.to_csv(test_raw_path, index=False)
        logger.info("Fold {}: Test predictions saved ({} rows)", fold_id, len(test_df_raw))

    # Save CDR'd test predictions using primary (committed_overall) threshold
    test_df_clinical = apply_clinical_decision_rule(test_df_raw.copy(), primary_threshold)
    test_df_clinical = fill_missing_epochs(test_df_clinical, max_gap_multiplier=max_gap_multiplier)
    test_df_clinical.to_csv(evaluation_dir / "test_predictions_clinical.csv", index=False)

    # --- Three-metric-type analysis (separate thresholds) -------------------
    logger.info("Fold {}: Generating three metric type analysis...", fold_id)
    three_metric_results = {}
    try:
        three_metric_results = generate_three_metric_type_analysis(
            test_df_raw,
            thresholds=all_thresholds,
            output_base_dir=evaluation_dir,
            exclude_last_minutes=exclude_last_minutes,
            title_suffix=f"Temporal Fold {fold_id}",
            max_gap_multiplier=max_gap_multiplier,
            decision_time_hours=decision_time_hours,
        )
        logger.info("Fold {}: Three metric type analysis complete", fold_id)
    except Exception as e:
        logger.warning("Fold {}: Three metric type analysis failed: {}", fold_id, e)

    # Extract PRIMARY metrics (committed_overall, mean across bins)
    primary_metrics = {}
    if three_metric_results and "summary" in three_metric_results:
        primary_summary = (
            three_metric_results["summary"]
            .get("metric_types", {})
            .get("committed_overall", {})
        )
        primary_metrics = {
            "test_sensitivity_mean": primary_summary.get("sensitivity_mean", 0.0),
            "test_sensitivity_std": primary_summary.get("sensitivity_std", 0.0),
            "test_specificity_mean": primary_summary.get("specificity_mean", 0.0),
            "test_specificity_std": primary_summary.get("specificity_std", 0.0),
            "test_fpr_mean": primary_summary.get("fpr_mean", 0.0),
            "test_fpr_std": primary_summary.get("fpr_std", 0.0),
        }

    # Extract decision-point metrics for all three metric types
    decision_point = three_metric_results.get("decision_point_metrics", {}) if three_metric_results else {}

    # --- ROC curve ----------------------------------------------------------
    roc_data = {}
    try:
        roc_data = compute_guid_level_roc(test_df_raw, decision_time_hours=decision_time_hours)
        if roc_data:
            plot_roc_curve(
                roc_data,
                evaluation_dir / "roc_curve.png",
                title_suffix=f"Temporal Fold {fold_id}",
                threshold=primary_threshold,
            )
            # Save ROC data as CSV
            roc_csv = pd.DataFrame({'fpr': roc_data['fpr'], 'tpr': roc_data['tpr']})
            roc_csv.to_csv(evaluation_dir / "roc_data.csv", index=False)
            logger.info("Fold {}: ROC AUC = {:.4f}", fold_id, roc_data['auc'])
    except Exception as e:
        logger.warning("Fold {}: ROC computation failed: {}", fold_id, e)

    # --- Save threshold info ------------------------------------------------
    threshold_info = {
        "primary_threshold": float(primary_threshold),
        "all_thresholds": {k: float(v) for k, v in all_thresholds.items()},
        "target_fpr": float(target_fpr),
        "validation_metrics_primary": {
            "sensitivity": float(threshold_metrics.get("sensitivity", 0)),
            "specificity": float(threshold_metrics.get("specificity", 0)),
            "fpr": float(threshold_metrics.get("fpr", 0)),
            "accuracy": float(threshold_metrics.get("accuracy", 0)),
            "time_window_hours": float(decision_time_hours),
        },
        "validation_decision_point_metrics": val_decision_point,
        "test_metrics_primary": primary_metrics,
        "decision_point_metrics": decision_point,
        "roc_auc": roc_data.get("auc"),
    }
    with open(evaluation_dir / "threshold_info.json", "w") as f:
        json.dump(convert_numpy_types(threshold_info), f, indent=2)

    # --- Assemble fold results dict -----------------------------------------
    fold_results = {
        "fold_id": fold_id,
        "primary_threshold": float(primary_threshold),
        "all_thresholds": {k: float(v) for k, v in all_thresholds.items()},
        "validation_sensitivity": float(threshold_metrics.get("sensitivity", 0)),
        "validation_specificity": float(threshold_metrics.get("specificity", 0)),
        "validation_fpr": float(threshold_metrics.get("fpr", 0)),
        "validation_accuracy": float(threshold_metrics.get("accuracy", 0)),
        "validation_decision_point_metrics": val_decision_point,
        # Mean across ALL time bins (backward compatible)
        "test_sensitivity_mean": primary_metrics.get("test_sensitivity_mean", 0.0),
        "test_sensitivity_std": primary_metrics.get("test_sensitivity_std", 0.0),
        "test_specificity_mean": primary_metrics.get("test_specificity_mean", 0.0),
        "test_specificity_std": primary_metrics.get("test_specificity_std", 0.0),
        "test_fpr_mean": primary_metrics.get("test_fpr_mean", 0.0),
        "test_fpr_std": primary_metrics.get("test_fpr_std", 0.0),
        # Decision-point metrics (FPR at 1h for each metric type)
        "decision_point_metrics": decision_point,
        # ROC
        "roc_auc": roc_data.get("auc"),
        "roc_data": {
            "fpr": roc_data.get("fpr", []),
            "tpr": roc_data.get("tpr", []),
        } if roc_data else {},
        "status": "success",
        "three_metric_analysis": (
            three_metric_results.get("summary", {}) if three_metric_results else {}
        ),
        "three_metric_results_full": three_metric_results if three_metric_results else {},
    }

    # Serialise fold results (convert DataFrames to dicts for JSON)
    fold_results_json = fold_results.copy()
    tmr_full = fold_results_json.get("three_metric_results_full", {})
    if tmr_full:
        tmr_copy = tmr_full.copy()
        if "metrics_dict" in tmr_copy:
            tmr_copy["metrics_dict"] = {
                k: v.to_dict("records") if v is not None else None
                for k, v in tmr_copy["metrics_dict"].items()
            }
        if "subgroup_metrics" in tmr_copy:
            tmr_copy["subgroup_metrics"] = {
                mt: {
                    sg: df.to_dict("records") if df is not None else None
                    for sg, df in sgs.items()
                }
                for mt, sgs in tmr_copy["subgroup_metrics"].items()
            }
        fold_results_json["three_metric_results_full"] = tmr_copy

    results_path = fold_dir / "fold_results.json"
    with open(results_path, "w") as f:
        json.dump(convert_numpy_types(fold_results_json), f, indent=2)

    logger.info("Fold {}: results saved to {}", fold_id, results_path)
    return fold_results


# ---------------------------------------------------------------------------
#  Subprocess entry point for parallel evaluation
# ---------------------------------------------------------------------------


def _evaluate_single_fold_subprocess(
    fold_dir: str,
    gpu_id: int,
    config_path: str,
    target_fpr: float,
    exclude_last_minutes: float,
    decision_time_hours: float,
    max_gap_multiplier: Optional[float],
    regenerate_predictions: bool,
    allow_backward_compat: bool,
) -> Dict:
    """Evaluate a single fold in a subprocess with GPU isolation.

    This function is designed to be called via
    :class:`concurrent.futures.ProcessPoolExecutor` with a ``'spawn'``
    context.  It sets ``CUDA_VISIBLE_DEVICES`` **before** any CUDA
    initialisation so the subprocess only sees the assigned GPU.

    Args:
        fold_dir: Path to the fold directory (e.g. ``/path/to/fold_1``).
        gpu_id: Physical GPU ID to use for this fold.
        config_path: Path to ``config_temporal.yaml`` (or fold-local
            ``config.yaml``).
        target_fpr: Target FPR for threshold optimisation.
        exclude_last_minutes: Minutes to exclude from time-based analysis.
        decision_time_hours: Decision time (hours before birth).
        max_gap_multiplier: Gap multiplier for epoch filling.
        regenerate_predictions: Force-regenerate predictions even if cached.
        allow_backward_compat: Allow partial loading of older checkpoints.

    Returns:
        Dict with fold evaluation results.  On failure, contains
        ``"status": "failed"`` with error details.
    """
    # Set CUDA_VISIBLE_DEVICES FIRST, before any CUDA init
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    fold_path = Path(fold_dir)
    fold_id = int(fold_path.name.split("_")[1])
    pid = os.getpid()
    logger.info(
        "Fold {} starting on GPU {} (pid={})...",
        fold_id, gpu_id, pid,
    )

    try:
        # Load config — prefer fold-local config, fall back to global
        fold_config_path = fold_path / "config.yaml"
        cfg_to_load = str(fold_config_path) if fold_config_path.exists() else config_path
        with open(cfg_to_load) as f:
            fold_config = yaml.safe_load(f)

        # Inside subprocess, device is always cuda:0 (the only visible GPU)
        fold_results = evaluate_single_fold_temporal(
            fold_dir=fold_dir,
            config=fold_config,
            device="cuda:0",
            target_fpr=target_fpr,
            exclude_last_minutes=exclude_last_minutes,
            decision_time_hours=decision_time_hours,
            max_gap_multiplier=max_gap_multiplier,
            regenerate_predictions=regenerate_predictions,
            allow_backward_compat=allow_backward_compat,
        )
        fold_results["gpu_id"] = gpu_id
        logger.info(
            "Fold {} COMPLETED on GPU {} (pid={})",
            fold_id, gpu_id, pid,
        )
        return fold_results

    except Exception as exc:
        logger.exception("Fold {} FAILED on GPU {} (pid={}):", fold_id, gpu_id, pid)
        return {
            "fold_id": fold_id,
            "gpu_id": gpu_id,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------


def main(
    output_base_dir: str,
    config_path: Optional[str] = None,
    target_fpr: Optional[float] = None,
    device: str = "cuda:0",
    exclude_last_minutes: Optional[float] = None,
    max_gap_multiplier: Optional[float] = None,
    decision_time_hours: Optional[float] = None,
    fold_ids: Optional[List[int]] = None,
    regenerate_predictions: bool = False,
    allow_backward_compat: Optional[bool] = None,
    aggregate_only: bool = False,
    gpu_ids: Optional[List[int]] = None,
    max_parallel: Optional[int] = None,
    sequential: bool = False,
    fold_timeout_hours: float = 2.0,
) -> Dict:
    """Run the temporal evaluation pipeline on all completed folds.

    Mirrors ``evaluate_classifier.main`` but uses temporal inference and
    temporal model loading.  Supports parallel multi-GPU evaluation when
    ``gpu_ids`` is provided.

    Args:
        output_base_dir: Base directory containing ``fold_1/``, ``fold_2/``,
            etc. with trained temporal models.
        config_path: Optional path to ``config_temporal.yaml``.  If provided,
            evaluation settings are read from config.  Explicit parameters
            override config values.
        target_fpr: Target FPR for threshold optimisation.
        device: CUDA device (used only in legacy sequential mode when
            ``gpu_ids`` is ``None``).
        exclude_last_minutes: Minutes to exclude from time-based analysis.
        max_gap_multiplier: Gap multiplier for epoch filling.
        decision_time_hours: Decision time (hours before birth).
        fold_ids: Specific fold IDs to evaluate.  ``None`` evaluates all.
        regenerate_predictions: Force-regenerate predictions even if cached.
        allow_backward_compat: Allow partial loading of older temporal
            checkpoints.  ``None`` defers to config.
        aggregate_only: If ``True``, skip per-fold inference and load existing
            ``fold_results.json`` from each fold directory.  Only runs
            cross-fold aggregation and plot generation.  Default ``False``.
        gpu_ids: List of GPU device IDs for parallel dispatch (e.g.
            ``[0, 1, 2, 3]``).  ``None`` falls back to legacy sequential
            mode using ``device``.
        max_parallel: Maximum number of concurrent fold evaluation workers.
            Defaults to ``min(len(gpu_ids), len(fold_dirs))``.
        sequential: If ``True`` with ``gpu_ids`` set, run folds sequentially
            through the subprocess path (useful for debugging GPU isolation).
        fold_timeout_hours: Timeout in hours for parallel ``as_completed``.
            Defaults to 2.0.

    Returns:
        Dict with aggregated results across all folds.
    """
    # Lazy import — evaluate_classifier imports SeqVae which triggers model_utils
    from model.vae_teb_prediction.evaluate_classifier import convert_numpy_types

    output_base_dir = Path(output_base_dir)
    if not output_base_dir.exists():
        raise FileNotFoundError(f"Output base directory not found: {output_base_dir}")

    # Load evaluation settings from config if provided
    eval_cfg: Dict = {}
    dataset_cfg: Dict = {}
    if config_path and Path(config_path).exists():
        logger.info("Loading temporal evaluation config from: {}", config_path)
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        eval_cfg = config_data.get("model_config", {}).get("evaluation", {}) or {}
        dataset_cfg = config_data.get("dataset_config", {}) or {}

    # Apply config with parameter overrides
    target_fpr = target_fpr if target_fpr is not None else float(eval_cfg.get("target_fpr", 0.2))
    exclude_last_minutes = (
        exclude_last_minutes
        if exclude_last_minutes is not None
        else float(eval_cfg.get("exclude_last_minutes", 30.0))
    )
    max_gap_multiplier = (
        max_gap_multiplier
        if max_gap_multiplier is not None
        else eval_cfg.get("max_gap_multiplier")
    )
    decision_time_hours = (
        decision_time_hours
        if decision_time_hours is not None
        else float(eval_cfg.get("decision_time_hours", 1.0))
    )
    allow_backward_compat = (
        allow_backward_compat
        if allow_backward_compat is not None
        else bool(eval_cfg.get("allow_backward_compat_checkpoint_loading", False))
    )
    fold_ids = fold_ids if fold_ids is not None else dataset_cfg.get("fold_ids")

    # --- Aggregate-only mode: skip inference, load from disk ------------------
    if aggregate_only:
        logger.info("=" * 80)
        logger.info("TEMPORAL EVALUATION — AGGREGATE ONLY MODE")
        logger.info("=" * 80)
        logger.info("Output base dir: {}", output_base_dir)
        logger.info("Exclude last minutes: {}", exclude_last_minutes)

        from model.vae_teb_prediction.guid_classifier.kfold_temporal_trainer import (
            aggregate_temporal_results,
        )

        aggregated = aggregate_temporal_results(
            output_base_dir=str(output_base_dir),
            fold_ids=fold_ids,
            fold_results=None,  # triggers _load_fold_results_from_disk
            exclude_last_minutes=exclude_last_minutes,
        )
        logger.info("Aggregate-only mode completed.")
        return aggregated

    logger.info("=" * 80)
    logger.info("TEMPORAL EVALUATION PIPELINE")
    logger.info("=" * 80)
    logger.info("Output base dir: {}", output_base_dir)
    logger.info("Target FPR: {}", target_fpr)
    logger.info("Decision time: {}h", decision_time_hours)
    logger.info("Exclude last minutes: {}", exclude_last_minutes)
    logger.info("Backward-compatible checkpoint loading: {}", allow_backward_compat)

    # Discover fold directories
    all_fold_dirs = sorted(
        [d for d in output_base_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")],
        key=lambda x: int(x.name.split("_")[1]),
    )
    if not all_fold_dirs:
        raise FileNotFoundError(f"No fold directories found in {output_base_dir}")

    if fold_ids:
        fold_dirs = [d for d in all_fold_dirs if int(d.name.split("_")[1]) in fold_ids]
    else:
        fold_dirs = all_fold_dirs

    logger.info("Evaluating {} folds: {}", len(fold_dirs), [d.name for d in fold_dirs])

    # Load config for model creation
    config: Dict = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Try to load from first fold directory
        first_cfg = fold_dirs[0] / "config.yaml"
        if first_cfg.exists():
            with open(first_cfg) as f:
                config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(
                "No config_path provided and no config.yaml found in first fold dir"
            )

    # Process each fold
    all_fold_results: List[Dict] = []
    successful_folds: List[int] = []
    failed_folds: List[int] = []

    # Resolve config_path to absolute string for subprocess serialisation
    resolved_config_path: Optional[str] = None
    if config_path:
        resolved_config_path = str(Path(config_path).resolve())

    # Common kwargs for subprocess entry point
    subprocess_kwargs = dict(
        config_path=resolved_config_path or "",
        target_fpr=target_fpr,
        exclude_last_minutes=exclude_last_minutes,
        decision_time_hours=decision_time_hours,
        max_gap_multiplier=max_gap_multiplier,
        regenerate_predictions=regenerate_predictions,
        allow_backward_compat=allow_backward_compat,
    )

    use_parallel = (
        gpu_ids is not None
        and not sequential
        and (max_parallel is None or max_parallel > 1)
    )

    if gpu_ids is not None and not use_parallel:
        # ----- Branch B: Sequential with GPU isolation (debug mode) --------
        logger.info("Running folds sequentially with GPU isolation...")
        n_total = len(fold_dirs)
        for job_idx, fold_dir in enumerate(fold_dirs):
            fold_id = int(fold_dir.name.split("_")[1])
            gpu_id = gpu_ids[job_idx % len(gpu_ids)]
            result = _evaluate_single_fold_subprocess(
                fold_dir=str(fold_dir),
                gpu_id=gpu_id,
                **subprocess_kwargs,
            )
            if result.get("status") == "failed":
                logger.error(
                    "Fold {}: FAILED — {}",
                    fold_id, result.get("error", "unknown"),
                )
                failed_folds.append(fold_id)
            else:
                all_fold_results.append(result)
                successful_folds.append(fold_id)
                logger.info(
                    "Fold {}: COMPLETED ({}/{} folds done)",
                    fold_id, job_idx + 1, n_total,
                )

    elif use_parallel:
        # ----- Branch A: Parallel multi-GPU dispatch -----------------------
        effective_max_parallel = (
            min(len(gpu_ids), len(fold_dirs))
            if max_parallel is None
            else max(1, min(max_parallel, len(fold_dirs)))
        )
        logger.info(
            "Running folds in parallel (GPUs={}, max_workers={})...",
            gpu_ids, effective_max_parallel,
        )

        spawn_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=effective_max_parallel, mp_context=spawn_ctx,
        ) as executor:
            futures = {}
            for job_idx, fold_dir in enumerate(fold_dirs):
                gpu_id = gpu_ids[job_idx % len(gpu_ids)]
                future = executor.submit(
                    _evaluate_single_fold_subprocess,
                    fold_dir=str(fold_dir),
                    gpu_id=gpu_id,
                    **subprocess_kwargs,
                )
                futures[future] = int(fold_dir.name.split("_")[1])

            n_done = 0
            n_total = len(futures)
            try:
                for future in as_completed(
                    futures, timeout=fold_timeout_hours * 3600,
                ):
                    fold_id = futures[future]
                    try:
                        result = future.result()
                        if result.get("status") == "failed":
                            logger.error(
                                "Fold {}: FAILED — {}",
                                fold_id, result.get("error", "unknown"),
                            )
                            failed_folds.append(fold_id)
                        else:
                            all_fold_results.append(result)
                            successful_folds.append(fold_id)
                        n_done += 1
                        logger.info(
                            "Fold {}: done ({}/{} folds complete)",
                            fold_id, n_done, n_total,
                        )
                    except Exception as exc:
                        logger.exception("Fold {} raised exception:", fold_id)
                        failed_folds.append(fold_id)
                        n_done += 1
            except TimeoutError:
                timed_out_folds = [
                    fid for fut, fid in futures.items() if not fut.done()
                ]
                logger.error(
                    "Parallel evaluation timed out after {:.1f}h — "
                    "{} fold(s) did not complete: {}",
                    fold_timeout_hours,
                    len(timed_out_folds),
                    timed_out_folds,
                )
                for future, fid in futures.items():
                    if not future.done():
                        future.cancel()
                        failed_folds.append(fid)

    else:
        # ----- Branch C: Legacy sequential (no GPU isolation) --------------
        for fold_dir in fold_dirs:
            fold_id = int(fold_dir.name.split("_")[1])
            try:
                # Load fold-specific config if available
                fold_config_path = fold_dir / "config.yaml"
                if fold_config_path.exists():
                    with open(fold_config_path) as f:
                        fold_config = yaml.safe_load(f)
                else:
                    fold_config = config

                fold_results = evaluate_single_fold_temporal(
                    fold_dir=str(fold_dir),
                    config=fold_config,
                    device=device,
                    target_fpr=target_fpr,
                    exclude_last_minutes=exclude_last_minutes,
                    decision_time_hours=decision_time_hours,
                    max_gap_multiplier=max_gap_multiplier,
                    regenerate_predictions=regenerate_predictions,
                    allow_backward_compat=allow_backward_compat,
                )
                all_fold_results.append(fold_results)
                successful_folds.append(fold_id)
                logger.info("Fold {}: COMPLETED SUCCESSFULLY", fold_id)

            except Exception as e:
                logger.error("Fold {}: FAILED with error: {}", fold_id, e)
                logger.error(traceback.format_exc())
                failed_folds.append(fold_id)

    # Aggregate results and generate cross-fold plots through the same helper
    # used by the k-fold trainer so standalone evaluation and training share
    # the exact same aggregation behaviour.
    if all_fold_results:
        try:
            from model.vae_teb_prediction.guid_classifier.kfold_temporal_trainer import (
                aggregate_temporal_results,
            )

            aggregated = aggregate_temporal_results(
                output_base_dir=str(output_base_dir),
                fold_results=all_fold_results,
                exclude_last_minutes=exclude_last_minutes,
            )
        except Exception as exc:
            logger.error("Cross-fold evaluation aggregation failed: {}", exc)
            aggregated = _aggregate_temporal_results(all_fold_results)
    else:
        aggregated = _aggregate_temporal_results(all_fold_results)

    aggregated["successful_folds"] = successful_folds
    aggregated["failed_folds"] = failed_folds

    agg_path = output_base_dir / "aggregated_results.json"
    with open(agg_path, "w") as f:
        json.dump(convert_numpy_types(aggregated), f, indent=2)

    logger.info("Aggregated results saved to {}", agg_path)
    return aggregated


def _aggregate_temporal_results(all_fold_results: List[Dict]) -> Dict:
    """Aggregate evaluation results across folds.

    Args:
        all_fold_results: List of per-fold result dicts.

    Returns:
        Dict with mean/std statistics.
    """
    if not all_fold_results:
        return {"status": "failed", "n_successful": 0}

    def _mean_std(vals: List[float]) -> Tuple[float, float]:
        arr = np.array(vals, dtype=float)
        return float(arr.mean()), float(arr.std())

    def _safe_collect(key: str) -> List[float]:
        return [r[key] for r in all_fold_results if r.get(key) is not None]

    thresholds = _safe_collect("primary_threshold")
    val_sens = _safe_collect("validation_sensitivity")
    val_spec = _safe_collect("validation_specificity")
    val_fpr = _safe_collect("validation_fpr")
    test_sens = _safe_collect("test_sensitivity_mean")
    test_spec = _safe_collect("test_specificity_mean")
    test_fpr = _safe_collect("test_fpr_mean")

    t_m, t_s = _mean_std(thresholds) if thresholds else (0.0, 0.0)
    vs_m, vs_s = _mean_std(val_sens) if val_sens else (0.0, 0.0)
    vsp_m, vsp_s = _mean_std(val_spec) if val_spec else (0.0, 0.0)
    vf_m, vf_s = _mean_std(val_fpr) if val_fpr else (0.0, 0.0)
    ts_m, ts_s = _mean_std(test_sens) if test_sens else (0.0, 0.0)
    tsp_m, tsp_s = _mean_std(test_spec) if test_spec else (0.0, 0.0)
    tf_m, tf_s = _mean_std(test_fpr) if test_fpr else (0.0, 0.0)

    result = {
        "status": "success",
        "n_successful": len(all_fold_results),
        "threshold_mean": t_m,
        "threshold_std": t_s,
        "validation_sensitivity_mean": vs_m,
        "validation_sensitivity_std": vs_s,
        "validation_specificity_mean": vsp_m,
        "validation_specificity_std": vsp_s,
        "validation_fpr_mean": vf_m,
        "validation_fpr_std": vf_s,
        "test_sensitivity_mean": ts_m,
        "test_sensitivity_std": ts_s,
        "test_specificity_mean": tsp_m,
        "test_specificity_std": tsp_s,
        "test_fpr_mean": tf_m,
        "test_fpr_std": tf_s,
    }

    # Aggregate decision-point metrics per metric type (test AND validation)
    metric_types = ['instantaneous', 'committed_cumulative', 'committed_overall']
    for mt in metric_types:
        # --- Test decision-point metrics ---
        fpr_vals = []
        sens_vals = []
        thresh_vals = []
        for r in all_fold_results:
            dp = r.get("decision_point_metrics", {}).get(mt, {})
            if dp.get("fpr_at_decision") is not None:
                fpr_vals.append(dp["fpr_at_decision"])
            if dp.get("sensitivity_at_decision") is not None:
                sens_vals.append(dp["sensitivity_at_decision"])
            at = r.get("all_thresholds", {}).get(mt)
            if at is not None:
                thresh_vals.append(at)

        if fpr_vals:
            m, s = _mean_std(fpr_vals)
            result[f"test_fpr_at_decision_{mt}_mean"] = m
            result[f"test_fpr_at_decision_{mt}_std"] = s
        if sens_vals:
            m, s = _mean_std(sens_vals)
            result[f"test_sensitivity_at_decision_{mt}_mean"] = m
            result[f"test_sensitivity_at_decision_{mt}_std"] = s
        if thresh_vals:
            m, s = _mean_std(thresh_vals)
            result[f"threshold_{mt}_mean"] = m
            result[f"threshold_{mt}_std"] = s

        # --- Validation decision-point metrics ---
        val_fpr_vals = []
        val_sens_vals = []
        for r in all_fold_results:
            vdp = r.get("validation_decision_point_metrics", {}).get(mt, {})
            if vdp.get("fpr_at_decision") is not None:
                val_fpr_vals.append(vdp["fpr_at_decision"])
            if vdp.get("sensitivity_at_decision") is not None:
                val_sens_vals.append(vdp["sensitivity_at_decision"])

        if val_fpr_vals:
            m, s = _mean_std(val_fpr_vals)
            result[f"val_fpr_at_decision_{mt}_mean"] = m
            result[f"val_fpr_at_decision_{mt}_std"] = s
        if val_sens_vals:
            m, s = _mean_std(val_sens_vals)
            result[f"val_sensitivity_at_decision_{mt}_mean"] = m
            result[f"val_sensitivity_at_decision_{mt}_std"] = s

    # Aggregate ROC AUC
    roc_aucs = _safe_collect("roc_auc")
    if roc_aucs:
        m, s = _mean_std(roc_aucs)
        result["roc_auc_mean"] = m
        result["roc_auc_std"] = s

    return result


if __name__ == "__main__":
    # ==========================================================================
    # Post-Training Temporal Evaluation Pipeline
    # ==========================================================================
    # All settings (target_fpr, exclude_last_minutes, decision_time_hours,
    # max_gap_multiplier, fold_ids) are read from config_temporal.yaml.
    # You can override them by passing explicit values to main().

    # Path to config file (same directory as this script)
    CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config_temporal.yaml"
    )

    # Read output_base_dir from config
    with open(CONFIG_PATH, "r") as _f:
        _cfg = yaml.safe_load(_f)
    OUTPUT_BASE_DIR = (
        _cfg.get("general_config", {})
        .get("folders_config", {})
        .get("out_dir_base", os.getcwd())
    )

    # Runtime options (not in config)
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    REGENERATE_PREDICTIONS = False  # Set to True to regenerate all predictions
    AGGREGATE_ONLY = False  # Set to True to only generate aggregated plots

    # Parallel evaluation settings (read from config, overridable here)
    _general_cfg = _cfg.get("general_config", {})
    GPU_IDS = _general_cfg.get("cuda_devices", None)      # e.g. [0,1,2,3,4]
    MAX_PARALLEL = _general_cfg.get("max_parallel_folds", None)  # e.g. 5
    SEQUENTIAL = False      # Set True to debug subprocess path without parallelism
    FOLD_TIMEOUT_HOURS = 2.0

    # ======================================================================
    # MODE 1: Full Evaluation Pipeline (evaluate all folds + aggregate)
    # ======================================================================
    if not AGGREGATE_ONLY:
        results = main(
            output_base_dir=OUTPUT_BASE_DIR,
            config_path=CONFIG_PATH,
            device=DEVICE,
            regenerate_predictions=REGENERATE_PREDICTIONS,
            aggregate_only=False,
            gpu_ids=GPU_IDS,
            max_parallel=MAX_PARALLEL,
            sequential=SEQUENTIAL,
            fold_timeout_hours=FOLD_TIMEOUT_HOURS,
            # Optional overrides (uncomment to override config values):
            # target_fpr=0.15,
            # exclude_last_minutes=30.0,
            # decision_time_hours=1.0,
            # max_gap_multiplier=None,
            # fold_ids=[1, 3, 9],  # Only evaluate specific folds
        )
        print("\nEvaluation pipeline completed!")
        print(f"Results saved to: {OUTPUT_BASE_DIR}/aggregated_results.json")
        print(f"Aggregated plots saved to: {OUTPUT_BASE_DIR}/aggregated_plots/")

    # ======================================================================
    # MODE 2: Aggregate Only (generate aggregated plots from existing results)
    # ======================================================================
    else:
        results = main(
            output_base_dir=OUTPUT_BASE_DIR,
            config_path=CONFIG_PATH,
            aggregate_only=True,
            # Optional overrides (uncomment to override config values):
            # fold_ids=[1, 3, 9],
            # exclude_last_minutes=30.0,
        )
        print("\nAggregation completed!")
        print(f"Aggregated plots saved to: {OUTPUT_BASE_DIR}/aggregated_plots/")
