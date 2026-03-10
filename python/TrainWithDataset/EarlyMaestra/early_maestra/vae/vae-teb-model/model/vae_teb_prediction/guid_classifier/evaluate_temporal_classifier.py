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

import json
import os
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

    model = TemporalVaeClassifier(
        vae_model=vae_model,
        segment_encoder_type=seg_cfg.get("type", "mean_pool"),
        d_seg=seg_cfg.get("d_seg", 64),
        temporal_lstm_hidden=lstm_cfg.get("hidden_dim", 128),
        temporal_lstm_layers=lstm_cfg.get("num_layers", 2),
        temporal_lstm_dropout=lstm_cfg.get("dropout", 0.1),
        gap_encoding=model_cfg.get("gap_encoding", "concat"),
        position_embed_dim=(
            seg_idx_cfg.get("embed_dim", 16)
            if seg_idx_cfg.get("enabled", False)
            else 0
        ),
        max_position_index=seg_idx_cfg.get("max_index", 40),
        tlo_enabled=tlo_cfg.get("enabled", False),
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

    ignored_vae_keys = sorted(k for k in cleaned if k.startswith("vae_model."))
    temporal_state = {
        k: v for k, v in cleaned.items()
        if not k.startswith("vae_model.")
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
        "(ignored VAE keys: {}, compat_mode: {}, temporal_missing: {}, "
        "temporal_unexpected: {}, temporal_shape_mismatches: {})",
        checkpoint_path,
        len(ignored_vae_keys),
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
        convert_numpy_types,
        fill_missing_epochs,
        find_latest_checkpoint_in_fold,
        find_threshold_for_committed_overall_fpr_at_1h,
        generate_three_metric_type_analysis,
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

    # --- Find threshold on validation set -----------------------------------
    logger.info(
        "Fold {}: Finding PRIMARY threshold (target_fpr={}, time={}h)...",
        fold_id, target_fpr, decision_time_hours,
    )
    primary_threshold, threshold_metrics = find_threshold_for_committed_overall_fpr_at_1h(
        val_df_raw,
        target_fpr=target_fpr,
        time_window_hours=decision_time_hours,
        max_gap_multiplier=max_gap_multiplier,
        fallback_tolerance_hours=0.5,
    )

    # Compute accuracy from sensitivity/specificity
    n_pos = threshold_metrics.get("n_positive_total", threshold_metrics.get("n_available_positive", 1))
    n_neg = threshold_metrics.get("n_negative_total", threshold_metrics.get("n_available_negative", 1))
    sens = threshold_metrics.get("sensitivity", 0)
    spec = threshold_metrics.get("specificity", 0)
    threshold_metrics["accuracy"] = (
        float((sens * n_pos + spec * n_neg) / (n_pos + n_neg))
        if (n_pos + n_neg) > 0
        else 0.0
    )

    logger.info("Fold {}: PRIMARY threshold = {:.4f}", fold_id, primary_threshold)

    # Apply CDR to validation set
    val_df_clinical = apply_clinical_decision_rule(val_df_raw.copy(), primary_threshold)
    val_df_clinical = fill_missing_epochs(val_df_clinical, max_gap_multiplier=max_gap_multiplier)
    val_df_clinical.to_csv(evaluation_dir / "validation_predictions_clinical.csv", index=False)

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

    # Apply CDR to test set
    test_df_clinical = apply_clinical_decision_rule(test_df_raw.copy(), primary_threshold)
    test_df_clinical = fill_missing_epochs(test_df_clinical, max_gap_multiplier=max_gap_multiplier)
    test_df_clinical.to_csv(evaluation_dir / "test_predictions_clinical.csv", index=False)

    # --- Three-metric-type analysis -----------------------------------------
    logger.info("Fold {}: Generating three metric type analysis...", fold_id)
    three_metric_results = {}
    try:
        three_metric_results = generate_three_metric_type_analysis(
            test_df_clinical,
            output_base_dir=evaluation_dir,
            exclude_last_minutes=exclude_last_minutes,
            title_suffix=f"Temporal Fold {fold_id}",
        )
        logger.info("Fold {}: Three metric type analysis complete", fold_id)
    except Exception as e:
        logger.warning("Fold {}: Three metric type analysis failed: {}", fold_id, e)

    # Extract PRIMARY metrics
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

    # --- Save threshold info ------------------------------------------------
    threshold_info = {
        "primary_threshold": float(primary_threshold),
        "target_fpr": float(target_fpr),
        "validation_metrics": {
            "sensitivity": float(threshold_metrics.get("sensitivity", 0)),
            "specificity": float(threshold_metrics.get("specificity", 0)),
            "fpr": float(threshold_metrics.get("fpr", 0)),
            "accuracy": float(threshold_metrics.get("accuracy", 0)),
            "time_window_hours": float(decision_time_hours),
        },
        "test_metrics_primary": primary_metrics,
    }
    with open(evaluation_dir / "threshold_info.json", "w") as f:
        json.dump(convert_numpy_types(threshold_info), f, indent=2)

    # --- Assemble fold results dict -----------------------------------------
    fold_results = {
        "fold_id": fold_id,
        "primary_threshold": float(primary_threshold),
        "validation_sensitivity": float(threshold_metrics.get("sensitivity", 0)),
        "validation_specificity": float(threshold_metrics.get("specificity", 0)),
        "validation_fpr": float(threshold_metrics.get("fpr", 0)),
        "validation_accuracy": float(threshold_metrics.get("accuracy", 0)),
        "test_sensitivity_mean": primary_metrics.get("test_sensitivity_mean", 0.0),
        "test_sensitivity_std": primary_metrics.get("test_sensitivity_std", 0.0),
        "test_specificity_mean": primary_metrics.get("test_specificity_mean", 0.0),
        "test_specificity_std": primary_metrics.get("test_specificity_std", 0.0),
        "test_fpr_mean": primary_metrics.get("test_fpr_mean", 0.0),
        "test_fpr_std": primary_metrics.get("test_fpr_std", 0.0),
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
) -> Dict:
    """Run the temporal evaluation pipeline on all completed folds.

    Mirrors ``evaluate_classifier.main`` but uses temporal inference and
    temporal model loading.

    Args:
        output_base_dir: Base directory containing ``fold_1/``, ``fold_2/``,
            etc. with trained temporal models.
        config_path: Optional path to ``config_temporal.yaml``.  If provided,
            evaluation settings are read from config.  Explicit parameters
            override config values.
        target_fpr: Target FPR for threshold optimisation.
        device: CUDA device.
        exclude_last_minutes: Minutes to exclude from time-based analysis.
        max_gap_multiplier: Gap multiplier for epoch filling.
        decision_time_hours: Decision time (hours before birth).
        fold_ids: Specific fold IDs to evaluate.  ``None`` evaluates all.
        regenerate_predictions: Force-regenerate predictions even if cached.
        allow_backward_compat: Allow partial loading of older temporal
            checkpoints.  ``None`` defers to config.

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
            import traceback
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

    thresholds = [r["primary_threshold"] for r in all_fold_results]
    val_sens = [r["validation_sensitivity"] for r in all_fold_results]
    val_spec = [r["validation_specificity"] for r in all_fold_results]
    val_fpr = [r["validation_fpr"] for r in all_fold_results]
    test_sens = [r["test_sensitivity_mean"] for r in all_fold_results]
    test_spec = [r["test_specificity_mean"] for r in all_fold_results]
    test_fpr = [r["test_fpr_mean"] for r in all_fold_results]

    t_m, t_s = _mean_std(thresholds)
    vs_m, vs_s = _mean_std(val_sens)
    vsp_m, vsp_s = _mean_std(val_spec)
    vf_m, vf_s = _mean_std(val_fpr)
    ts_m, ts_s = _mean_std(test_sens)
    tsp_m, tsp_s = _mean_std(test_spec)
    tf_m, tf_s = _mean_std(test_fpr)

    return {
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
