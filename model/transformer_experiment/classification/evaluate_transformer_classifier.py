"""Evaluation pipeline for the transformer-based GRU classifier.

Provides inference, threshold optimisation, three-metric-type analysis,
ROC curve computation, subgroup analysis, and cross-fold aggregation.

All metric computation is delegated to
``model.vae_teb_prediction.evaluate_classifier`` (lazy-imported to avoid
triggering unrelated imports).  All plotting uses
``evaluation_plots.py`` with publication-quality style.

Three operational modes:

1. **Full evaluation** (default): Runs inference on val/test from the
   best checkpoint, finds thresholds, generates plots.
2. **Cached predictions** (``regenerate_predictions=False``): Skips
   inference if CSV files exist from a prior run.
3. **Aggregate only** (``aggregate_only=True``): Loads existing
   ``fold_results.json`` and generates cross-fold aggregated plots.

Typical CLI usage::

    # Full evaluation on all trained folds:
    python -m model.transformer.classification.evaluate_transformer_classifier \\
        --output_dir ./classification_results --device cuda:0

    # Re-run inference from best checkpoints:
    python -m model.transformer.classification.evaluate_transformer_classifier \\
        --output_dir ./classification_results --regenerate_predictions

    # Aggregate existing fold results only:
    python -m model.transformer.classification.evaluate_transformer_classifier \\
        --output_dir ./classification_results --aggregate_only
"""

from __future__ import annotations

import gc
import json
import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from loguru import logger

from train.graph_models_utils import load_checkpoint_strict


# ====================================================================== #
#  Model Reconstruction                                                    #
# ====================================================================== #


def create_transformer_model_from_config(
    config: Dict,
    device: str = "cuda:0",
) -> nn.Module:
    """Reconstruct a :class:`TimeAwareGRUClassifier` from a config dict.

    Loads the pretrained transformer checkpoint (if not in precomputed
    mode), builds the classifier with all config-driven parameters,
    and returns it in eval mode on the specified device.

    Args:
        config: Full config dict from ``config_classification.yaml``.
        device: Target device.

    Returns:
        ``TimeAwareGRUClassifier`` in eval mode.
    """
    from model.transformer.classification.classification_model import (
        TimeAwareGRUClassifier,
    )

    model_cfg = config.get("model_config", {})
    precompute_mode = model_cfg.get("precompute_embeddings", False)

    # Load transformer if not in precomputed mode.
    transformer_model = None
    if not precompute_mode:
        transformer_checkpoint = model_cfg.get("transformer_checkpoint")
        if transformer_checkpoint is None:
            raise ValueError(
                "transformer_checkpoint required when "
                "precompute_embeddings is False"
            )

        from model.transformer.model.model import (
            CausalMultimodalTransformer,
        )
        from model.transformer.tr_testing.base import TransformerTestRunner

        logger.info("Loading transformer from: {}", transformer_checkpoint)
        ckpt = torch.load(
            transformer_checkpoint, map_location="cpu", weights_only=False
        )
        tr_config = TransformerTestRunner._extract_config(ckpt)
        transformer_model = CausalMultimodalTransformer(tr_config)
        load_checkpoint_strict(transformer_model, ckpt)
        logger.info("Transformer loaded successfully.")

    # Build classifier from config.
    emb_cfg = model_cfg.get("segment_embedding", {})
    time_cfg = model_cfg.get("time_features", {})
    cls_cfg = model_cfg.get("classifier", {})
    loss_cfg = model_cfg.get("loss", {})
    freeze_cfg = model_cfg.get("freeze_strategy", {})

    model = TimeAwareGRUClassifier(
        transformer_model=transformer_model,
        d_embedding=emb_cfg.get("d_embedding", 416),
        time_embed_dim=time_cfg.get("embed_dim", 32),
        input_proj_dim=cls_cfg.get("input_proj_dim", 256),
        gru_hidden_dim=cls_cfg.get("gru_hidden_dim", 256),
        dropout=cls_cfg.get("dropout", 0.1),
        loss_type=loss_cfg.get("type", "bce"),
        label_smoothing=loss_cfg.get("label_smoothing", 0.0),
        transformer_chunk_size=model_cfg.get("transformer_chunk_size", 16),
        freeze_strategy="frozen",
        pooling=emb_cfg.get("pooling", "mean"),
        anchor_step=emb_cfg.get("anchor_step", 5),
        nominal_gap_minutes=time_cfg.get("nominal_gap_minutes", 20.0),
        gap_threshold_minutes=time_cfg.get("gap_threshold_minutes", 22.0),
    )

    model.to(device)
    model.eval()
    return model


# ====================================================================== #
#  Checkpoint Loading                                                      #
# ====================================================================== #


def _load_transformer_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: str = "cpu",
) -> nn.Module:
    """Load a Lightning ``.ckpt`` into the classifier.

    Lightning saves the full ``PlClassifier`` state dict with a
    ``model.`` prefix.  This function strips that prefix and loads
    only the classifier-specific keys (time_encoder, input_proj,
    gru_cell, head, decay_gate).  Transformer keys are skipped
    because the transformer is loaded separately from its own
    checkpoint.

    Args:
        model: ``TimeAwareGRUClassifier`` instance.
        checkpoint_path: Path to the ``.ckpt`` file.
        device: Map location for tensor loading.

    Returns:
        The model with loaded weights.
    """
    logger.info("Loading classifier checkpoint: {}", checkpoint_path)

    ckpt = torch.load(str(checkpoint_path), map_location=device,
                        weights_only=False)

    # Extract state dict from Lightning format.
    state_dict = ckpt.get("state_dict", ckpt)

    # Strip 'model.' prefix added by Lightning.
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]

        # Skip transformer keys (loaded separately).
        if new_key.startswith("transformer."):
            continue

        # Skip loss buffers (not needed for inference).
        if new_key in ("pos_weight", "class_weights"):
            continue

        cleaned[new_key] = value

    # Load with strict=False to tolerate missing transformer keys.
    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    # Filter expected missing keys (transformer.*).
    real_missing = [
        k for k in missing
        if not k.startswith("transformer.")
        and k not in ("pos_weight", "class_weights")
    ]
    if real_missing:
        logger.warning(
            "Missing classifier keys ({}): {}", len(real_missing),
            real_missing[:10],
        )
    if unexpected:
        logger.warning(
            "Unexpected keys ({}): {}", len(unexpected), unexpected[:10],
        )

    logger.info(
        "Classifier checkpoint loaded — {} keys, {} skipped transformer",
        len(cleaned),
        sum(1 for k in state_dict if "transformer." in k),
    )
    return model


# ====================================================================== #
#  Inference                                                               #
# ====================================================================== #


def run_transformer_inference(
    model: nn.Module,
    dataloader,
    device: str = "cuda:0",
) -> pd.DataFrame:
    """Run inference and produce per-segment prediction rows.

    Iterates the dataloader, runs the model forward pass, and unpacks
    the temporal batch ``(B, S_max, ...)`` into individual segment
    rows.  The output DataFrame is compatible with all functions in
    ``evaluate_classifier.py``.

    Args:
        model: ``TimeAwareGRUClassifier`` in eval mode.
        dataloader: Bucketed or standard sequence DataLoader.
        device: CUDA device.

    Returns:
        DataFrame with columns: ``guid``, ``epoch``, ``target``,
        ``binary_target``, ``predicted_class``, ``prob_class_0``,
        ``prob_class_1``, ``cs_label``, ``bg_label``, ``tlo_hours``.
    """
    model.eval()
    rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move tensors to device.
            batch_device: Dict[str, Any] = {}
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch_device[key] = val.to(device, non_blocking=True)
                else:
                    batch_device[key] = val

            outputs = model(batch_device)
            probs = outputs["probs"]     # (B, S_max)
            mask = outputs["mask"]       # (B, S_max)
            lengths = batch_device.get("lengths")

            target = batch_device["target"]   # (B, S_max, 300)
            epoch = batch_device["epoch"]     # (B, S_max)
            cs_label = batch_device.get("cs_label")
            bg_label = batch_device.get("bg_label")
            tlo = batch_device.get("time_from_labor_onset")
            guids = batch_device.get("guids", batch_device.get("guid"))

            B = len(guids) if isinstance(guids, list) else mask.shape[0]

            for i in range(B):
                L = int(lengths[i].item()) if lengths is not None else int(
                    mask[i].sum().item()
                )
                guid_str = guids[i] if isinstance(guids, list) else str(
                    guids[i]
                )

                cs_val = bool(cs_label[i].item()) if cs_label is not None else False
                bg_val = bool(bg_label[i].item()) if bg_label is not None else False

                for j in range(L):
                    prob_1 = float(probs[i, j].item())
                    prob_0 = 1.0 - prob_1

                    seg_target = int(target[i, j].max().item())
                    binary = int(seg_target > 1)
                    predicted = int(prob_1 > 0.5)

                    seg_epoch = float(epoch[i, j].item())

                    tlo_val = float("nan")
                    if tlo is not None:
                        tlo_raw = tlo[i, j].item()
                        if not np.isnan(tlo_raw):
                            tlo_val = tlo_raw / 3600.0

                    rows.append({
                        "guid": guid_str,
                        "epoch": seg_epoch,
                        "target": seg_target,
                        "binary_target": binary,
                        "predicted_class": predicted,
                        "prob_class_0": prob_0,
                        "prob_class_1": prob_1,
                        "cs_label": cs_val,
                        "bg_label": bg_val,
                        "tlo_hours": tlo_val,
                    })

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    "  Inference batch {}/{}", batch_idx + 1, len(dataloader)
                )

    df = pd.DataFrame(rows)
    logger.info(
        "Inference complete — {} segments, {} GUIDs",
        len(df), df["guid"].nunique() if len(df) > 0 else 0,
    )
    return df


# ====================================================================== #
#  Three-Metric Analysis (our own version with publication plots)          #
# ====================================================================== #


def _run_three_metric_analysis(
    df_raw: pd.DataFrame,
    thresholds: Dict[str, float],
    output_base_dir: Path,
    exclude_last_minutes: float = 30.0,
    title_suffix: str = "",
    max_gap_multiplier: Optional[float] = None,
    decision_time_hours: float = 1.0,
) -> Dict:
    """Run three-metric-type analysis with publication-quality plots.

    For each metric type (instantaneous, committed_cumulative,
    committed_overall), applies CDR with the type's own threshold,
    fills epochs, computes metrics, generates plots, and extracts
    decision-point metrics.

    This is our own version of
    ``evaluate_classifier.generate_three_metric_type_analysis`` that
    uses ``evaluation_plots.py`` for all visuals.

    Args:
        df_raw: Raw predictions DataFrame (pre-CDR).
        thresholds: Dict mapping metric type name to threshold value.
        output_base_dir: Root output directory for plots/JSON.
        exclude_last_minutes: Exclude bins near birth.
        title_suffix: Title annotation (e.g. "Test Set").
        max_gap_multiplier: Max gap for epoch filling.
        decision_time_hours: Decision time in hours.

    Returns:
        Dict with ``metrics_dict``, ``decision_point_metrics``,
        ``subgroup_metrics``, ``dataset_statistics``.
    """
    from model.vae_teb_prediction.classifier.evaluate_classifier import (
        apply_clinical_decision_rule,
        compute_committed_cumulative_metrics,
        compute_committed_overall_metrics,
        compute_instantaneous_metrics,
        compute_subgroup_statistics,
        compute_time_bins,
        convert_numpy_types,
        create_enhanced_subgroup_filters,
        ensure_committed_epochs_filled,
        fill_missing_epochs,
        _extract_decision_point_metrics,
        _summarize_metrics_df,
    )
    from model.transformer.classification.evaluation_plots import (
        plot_dataset_statistics,
        plot_metric_comparison,
        plot_metric_curves,
        plot_subgroup_analysis,
    )

    metric_type_names = [
        "instantaneous", "committed_cumulative", "committed_overall",
    ]
    compute_funcs = {
        "instantaneous": compute_instantaneous_metrics,
        "committed_cumulative": compute_committed_cumulative_metrics,
        "committed_overall": compute_committed_overall_metrics,
    }

    os.makedirs(output_base_dir, exist_ok=True)

    metrics_dict: Dict[str, pd.DataFrame] = {}
    decision_point_metrics: Dict[str, Dict] = {}
    subgroup_metrics: Dict[str, Dict[str, pd.DataFrame]] = {}
    all_summaries: Dict[str, Dict] = {}

    for mt in metric_type_names:
        threshold = thresholds.get(mt, thresholds.get("committed_overall", 0.5))
        mt_dir = output_base_dir / "three_metric_types" / mt

        logger.info(
            "  {} — threshold={:.4f}", mt, threshold,
        )

        # Apply CDR with this metric type's threshold.
        df_clinical = apply_clinical_decision_rule(
            df_raw.copy(), threshold, verify=True,
        )

        # Fill missing epochs.
        fill_birth = mt != "instantaneous"
        df_filled = fill_missing_epochs(
            df_clinical,
            max_gap_multiplier=max_gap_multiplier,
            fill_until_birth=fill_birth,
            log_summary=False,
        )

        # Compute time bins and metrics.
        time_bins = compute_time_bins(df_filled, exclude_last_minutes)
        mt_df = compute_funcs[mt](df_filled, time_bins)
        metrics_dict[mt] = mt_df

        # Plot metric curves.
        plot_metric_curves(mt_df, mt, mt_dir, title_suffix)

        # Extract decision-point metrics.
        dp = _extract_decision_point_metrics(
            mt_df, decision_time_hours, mt, threshold,
        )
        decision_point_metrics[mt] = dp

        # Summarize.
        all_summaries[mt] = _summarize_metrics_df(mt_df)

        # Subgroup analysis.
        subgroup_filters = create_enhanced_subgroup_filters()
        sg_metrics: Dict[str, pd.DataFrame] = {}
        sg_guid_counts: Dict[str, int] = {}

        for sg_name, sg_filter in subgroup_filters.items():
            try:
                sg_df = compute_funcs[mt](df_filled, time_bins, sg_filter)
                if not sg_df.empty:
                    sg_metrics[sg_name] = sg_df
                    sg_mask = sg_filter(df_filled)
                    sg_guid_counts[sg_name] = int(
                        df_filled.loc[sg_mask, "guid"].nunique()
                    )
            except Exception:
                pass

        subgroup_metrics[mt] = sg_metrics

        # Plot subgroup analysis.
        sg_dir = mt_dir / "subgroups"
        plot_subgroup_analysis(
            sg_metrics, mt, sg_dir, title_suffix, sg_guid_counts,
        )

    # Comparison plot.
    comp_dir = output_base_dir / "three_metric_types" / "comparison"
    plot_metric_comparison(metrics_dict, comp_dir, title_suffix)

    # Dataset statistics.
    stats_dir = output_base_dir / "three_metric_types" / "dataset_stats"
    first_filled_df = None
    for mt in metric_type_names:
        threshold = thresholds.get(mt, 0.5)
        df_clinical = apply_clinical_decision_rule(
            df_raw.copy(), threshold, verify=False,
        )
        first_filled_df = fill_missing_epochs(
            df_clinical, max_gap_multiplier=max_gap_multiplier,
            fill_until_birth=True, log_summary=False,
        )
        break

    if first_filled_df is not None:
        time_bins = compute_time_bins(first_filled_df, exclude_last_minutes)
        plot_dataset_statistics(first_filled_df, time_bins, stats_dir,
                                title_suffix)

        # Save dataset statistics JSON.
        subgroup_filters = create_enhanced_subgroup_filters()
        ds_stats = compute_subgroup_statistics(first_filled_df,
                                                subgroup_filters)
    else:
        ds_stats = {}

    # Save JSON summaries.
    _save_json(
        output_base_dir / "three_metric_types" / "metrics_summary.json",
        convert_numpy_types(all_summaries),
    )
    _save_json(
        output_base_dir / "three_metric_types" / "thresholds.json",
        convert_numpy_types(thresholds),
    )
    _save_json(
        output_base_dir / "three_metric_types" / "dataset_statistics.json",
        convert_numpy_types(ds_stats),
    )

    return {
        "metrics_dict": {
            mt: df.to_dict("records") for mt, df in metrics_dict.items()
        },
        "decision_point_metrics": decision_point_metrics,
        "subgroup_metrics": {
            mt: {sg: df.to_dict("records") for sg, df in sgs.items()}
            for mt, sgs in subgroup_metrics.items()
        },
        "dataset_statistics": ds_stats,
        "summaries": all_summaries,
    }


def _save_json(path: Path, data: Any) -> None:
    """Save data as JSON with directory creation."""
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ====================================================================== #
#  Single-Fold Evaluation                                                  #
# ====================================================================== #


def evaluate_single_fold_transformer(
    fold_dir: str,
    config: Dict,
    device: str = "cuda:0",
    target_fpr: float = 0.2,
    exclude_last_minutes: float = 30.0,
    decision_time_hours: float = 1.0,
    max_gap_multiplier: Optional[float] = None,
    regenerate_predictions: bool = False,
) -> Dict:
    """Evaluate one trained fold of the transformer classifier.

    Workflow:
        1. Resolve best checkpoint (from fold_results.json or scan).
        2. Reconstruct model + load classifier weights.
        3. Run inference on validation (if not cached).
        4. Find 3 thresholds on validation.
        5. Run three-metric analysis on validation.
        6. Run inference on test (if not cached).
        7. Run three-metric analysis on test.
        8. Compute ROC curves.
        9. Save threshold_info.json, fold_results.json.

    Args:
        fold_dir: Path to the fold directory (e.g. ``fold_1/``).
        config: Full config dict.
        device: CUDA device.
        target_fpr: Target false-positive rate.
        exclude_last_minutes: Exclude bins near birth.
        decision_time_hours: Decision time in hours before birth.
        max_gap_multiplier: Max gap multiplier for epoch filling.
        regenerate_predictions: If True, re-run inference even if
            cached CSVs exist.

    Returns:
        Dict with all fold evaluation results.
    """
    from model.vae_teb_prediction.classifier.evaluate_classifier import (
        apply_clinical_decision_rule,
        compute_guid_level_roc,
        compute_committed_cumulative_roc,
        convert_numpy_types,
        find_latest_checkpoint_in_fold,
        find_threshold_for_committed_cumulative_fpr_at_1h,
        find_threshold_for_committed_overall_fpr_at_1h,
        find_threshold_for_instantaneous_fpr_at_1h,
    )
    from model.transformer.classification.evaluation_plots import (
        plot_roc_curve,
    )

    fold_path = Path(fold_dir)
    fold_id = int(fold_path.name.split("_")[-1]) if "_" in fold_path.name else 0
    eval_dir = fold_path / "evaluation"
    os.makedirs(eval_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Evaluating fold {} — target_fpr={}", fold_id, target_fpr)
    logger.info("=" * 60)

    # ---- 1. Resolve best checkpoint ---------------------------------- #
    best_ckpt_path = None
    fold_results_path = fold_path / "fold_results.json"
    if fold_results_path.exists():
        try:
            with open(fold_results_path) as f:
                prev_results = json.load(f)
            best_ckpt_path = prev_results.get("best_checkpoint_path")
        except Exception:
            pass

    if not best_ckpt_path or not Path(best_ckpt_path).exists():
        best_ckpt_path = str(find_latest_checkpoint_in_fold(fold_path))

    logger.info("Best checkpoint: {}", best_ckpt_path)

    # ---- 2. Model reconstruction ------------------------------------- #
    model = create_transformer_model_from_config(config, device)
    model = _load_transformer_checkpoint(model, Path(best_ckpt_path), device)
    model.to(device).eval()

    # ---- 3. Dataloaders ---------------------------------------------- #
    val_loader, test_loader = _create_eval_dataloaders(config, fold_id)

    # ---- 4. Validation inference ------------------------------------- #
    val_csv = eval_dir / "validation_predictions_raw.csv"
    if val_csv.exists() and not regenerate_predictions:
        logger.info("Loading cached validation predictions: {}", val_csv)
        val_df = pd.read_csv(val_csv)
    else:
        logger.info("Running validation inference...")
        val_df = run_transformer_inference(model, val_loader, device)
        val_df.to_csv(val_csv, index=False)
        logger.info("Saved validation predictions: {}", val_csv)

    # ---- 5. Find thresholds on validation ---------------------------- #
    logger.info("Finding thresholds (target_fpr={})...", target_fpr)

    thresh_overall, metrics_overall = find_threshold_for_committed_overall_fpr_at_1h(
        val_df, target_fpr=target_fpr, time_window_hours=decision_time_hours,
        max_gap_multiplier=max_gap_multiplier,
    )
    thresh_cumulative, metrics_cumulative = find_threshold_for_committed_cumulative_fpr_at_1h(
        val_df, target_fpr=target_fpr, time_window_hours=decision_time_hours,
        max_gap_multiplier=max_gap_multiplier,
    )
    thresh_instantaneous, metrics_instantaneous = find_threshold_for_instantaneous_fpr_at_1h(
        val_df, target_fpr=target_fpr, time_window_hours=decision_time_hours,
        max_gap_multiplier=max_gap_multiplier,
    )

    primary_threshold = thresh_overall
    all_thresholds = {
        "instantaneous": thresh_instantaneous,
        "committed_cumulative": thresh_cumulative,
        "committed_overall": thresh_overall,
    }

    logger.info(
        "Thresholds — overall={:.4f}, cumulative={:.4f}, "
        "instantaneous={:.4f}",
        thresh_overall, thresh_cumulative, thresh_instantaneous,
    )

    # Validation summary metrics from primary threshold.
    val_sens = metrics_overall.get("sensitivity", 0.0)
    val_spec = metrics_overall.get("specificity", 0.0)
    val_fpr = metrics_overall.get("fpr", 0.0)
    n_pos = metrics_overall.get("n_positive", 0)
    n_neg = metrics_overall.get("n_negative", 0)
    val_accuracy = (
        (val_sens * n_pos + val_spec * n_neg) / max(n_pos + n_neg, 1)
    )

    # Apply CDR to validation and save clinical predictions.
    val_clinical = apply_clinical_decision_rule(
        val_df.copy(), primary_threshold,
    )
    val_clinical.to_csv(
        eval_dir / "validation_predictions_clinical.csv", index=False,
    )

    # ---- 6. Validation three-metric analysis ------------------------- #
    logger.info("Running validation three-metric analysis...")
    val_analysis = _run_three_metric_analysis(
        val_df, all_thresholds,
        eval_dir / "validation_evaluation",
        exclude_last_minutes=exclude_last_minutes,
        title_suffix=f"Validation — Fold {fold_id}",
        max_gap_multiplier=max_gap_multiplier,
        decision_time_hours=decision_time_hours,
    )

    # ---- 7. Test inference ------------------------------------------- #
    test_csv = eval_dir / "test_predictions_raw.csv"
    if test_csv.exists() and not regenerate_predictions:
        logger.info("Loading cached test predictions: {}", test_csv)
        test_df = pd.read_csv(test_csv)
    else:
        logger.info("Running test inference...")
        test_df = run_transformer_inference(model, test_loader, device)
        test_df.to_csv(test_csv, index=False)
        logger.info("Saved test predictions: {}", test_csv)

    # Apply CDR to test and save.
    test_clinical = apply_clinical_decision_rule(
        test_df.copy(), primary_threshold,
    )
    test_clinical.to_csv(
        eval_dir / "test_predictions_clinical.csv", index=False,
    )

    # ---- 8. Test three-metric analysis ------------------------------- #
    logger.info("Running test three-metric analysis...")
    test_analysis = _run_three_metric_analysis(
        test_df, all_thresholds,
        eval_dir,
        exclude_last_minutes=exclude_last_minutes,
        title_suffix=f"Test — Fold {fold_id}",
        max_gap_multiplier=max_gap_multiplier,
        decision_time_hours=decision_time_hours,
    )

    # Extract test primary metrics (mean across bins).
    test_primary = test_analysis.get("summaries", {}).get(
        "committed_overall", {}
    )

    # ---- 9. ROC curves ----------------------------------------------- #
    logger.info("Computing ROC curves...")
    roc_data = compute_guid_level_roc(test_df, decision_time_hours)
    plot_roc_curve(
        roc_data, eval_dir / "roc_curve.png",
        title_suffix=f"Fold {fold_id}",
        threshold=primary_threshold,
    )

    # Save ROC data.
    roc_csv = eval_dir / "roc_data.csv"
    pd.DataFrame({
        "fpr": roc_data.get("fpr", []),
        "tpr": roc_data.get("tpr", []),
    }).to_csv(roc_csv, index=False)

    # CC-ROC.
    cc_roc_data = compute_committed_cumulative_roc(
        test_df, decision_time_hours, max_gap_multiplier,
    )
    plot_roc_curve(
        cc_roc_data,
        eval_dir / "roc_curve_committed_cumulative.png",
        title_suffix=f"CC-ROC — Fold {fold_id}",
    )

    # ---- 10. Save results -------------------------------------------- #
    threshold_info = {
        "primary_threshold": primary_threshold,
        "all_thresholds": all_thresholds,
        "target_fpr": target_fpr,
        "decision_time_hours": decision_time_hours,
        "validation_metrics_overall": convert_numpy_types(metrics_overall),
        "validation_metrics_cumulative": convert_numpy_types(metrics_cumulative),
        "validation_metrics_instantaneous": convert_numpy_types(metrics_instantaneous),
        "roc_auc": roc_data.get("auc"),
        "cc_roc_auc": cc_roc_data.get("auc"),
    }
    _save_json(eval_dir / "threshold_info.json", threshold_info)

    fold_results = {
        "fold_id": fold_id,
        "status": "success",

        "primary_threshold": primary_threshold,
        "all_thresholds": all_thresholds,

        "validation_sensitivity": val_sens,
        "validation_specificity": val_spec,
        "validation_fpr": val_fpr,
        "validation_accuracy": val_accuracy,

        "test_sensitivity_mean": test_primary.get("sensitivity_mean"),
        "test_sensitivity_std": test_primary.get("sensitivity_std"),
        "test_specificity_mean": test_primary.get("specificity_mean"),
        "test_specificity_std": test_primary.get("specificity_std"),
        "test_fpr_mean": test_primary.get("fpr_mean"),
        "test_fpr_std": test_primary.get("fpr_std"),

        "decision_point_metrics": test_analysis.get(
            "decision_point_metrics", {}
        ),

        "roc_auc": roc_data.get("auc"),
        "roc_data": {
            "fpr": [float(x) for x in roc_data.get("fpr", [])],
            "tpr": [float(x) for x in roc_data.get("tpr", [])],
        },
        "cc_roc_auc": cc_roc_data.get("auc"),

        "three_metric_analysis": test_analysis.get("summaries", {}),
    }

    fold_results_serialized = convert_numpy_types(fold_results)
    _save_json(fold_path / "fold_results.json", fold_results_serialized)

    logger.info(
        "Fold {} evaluation complete — threshold={:.4f}, "
        "val_sens={:.3f}, ROC AUC={:.3f}",
        fold_id, primary_threshold, val_sens,
        roc_data.get("auc", 0.0),
    )

    # Cleanup model to free GPU memory.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return fold_results


# ====================================================================== #
#  Eval Dataloader Helper                                                  #
# ====================================================================== #


def _create_eval_dataloaders(
    config: Dict,
    fold_id: int,
) -> Tuple:
    """Create validation and test dataloaders for evaluation.

    Args:
        config: Full config dict.
        fold_id: Fold number.

    Returns:
        Tuple of ``(val_loader, test_loader)``.
    """
    from model.vae_teb_prediction.classifier.kfold_classifier_trainer import (
        get_fold_datasets,
    )
    from model.vae_teb_prediction.classifier.guid_classifier.length_bucket_sampler import (
        create_bucketed_sequence_dataloader,
    )

    dataset_cfg = config.get("dataset_config", {})
    kfold_base_path = dataset_cfg["kfold_base_path"]
    test_mode = dataset_cfg.get("test_mode", None)
    fold_datasets = get_fold_datasets(kfold_base_path, fold_id,
                                       test_mode=test_mode)

    dl_cfg = dataset_cfg.get("dataloader_config", {})
    ds_kwargs = dict(dl_cfg.get("dataset_kwargs", {}))
    stat_path = dataset_cfg.get("stat_path")
    normalize_fields = dl_cfg.get("normalize_fields")
    seg_duration = dl_cfg.get("segment_duration", 1200.0)
    guid_cache = dl_cfg.get("guid_cache_size", 128)
    num_workers = dl_cfg.get("num_workers", 0)
    prefetch = dl_cfg.get("prefetch_factor", 2)
    pin_mem = dl_cfg.get("pin_memory", False)

    batch_size = config["general_config"]["batch_size"]["test"]
    bucket_cfg = dataset_cfg.get("bucket_sampler", {})
    bucket_ranges = bucket_cfg.get("bucket_ranges")

    model_cfg = config.get("model_config", {})
    use_precomputed = model_cfg.get("precompute_embeddings", False)

    common_kwargs = dict(
        num_workers=num_workers,
        segment_duration=seg_duration,
        guid_cache_size=guid_cache,
        stats_path=stat_path,
        normalize_fields=normalize_fields,
        prefetch_factor=prefetch,
        pin_memory=pin_mem,
        seed=42,
        **ds_kwargs,
    )

    if use_precomputed:
        from model.transformer.classification.precompute_embeddings import (
            create_precomputed_embedding_dataloader,
        )
        precomputed_dir = model_cfg.get("precomputed_dir", "")
        transformer_ckpt = model_cfg.get("transformer_checkpoint")

        val_loader, _ = create_precomputed_embedding_dataloader(
            precomputed_path=os.path.join(
                precomputed_dir, f"precomputed_fold_{fold_id}_val.hdf5"
            ),
            hdf5_files=fold_datasets["val"],
            batch_size=batch_size,
            bucket_ranges=bucket_ranges,
            shuffle=False,
            transformer_checkpoint=transformer_ckpt,
            **common_kwargs,
        )
        test_loader, _ = create_precomputed_embedding_dataloader(
            precomputed_path=os.path.join(
                precomputed_dir, f"precomputed_fold_{fold_id}_test.hdf5"
            ),
            hdf5_files=fold_datasets["test"],
            batch_size=batch_size,
            bucket_ranges=bucket_ranges,
            shuffle=False,
            transformer_checkpoint=transformer_ckpt,
            **common_kwargs,
        )
    else:
        val_loader, _ = create_bucketed_sequence_dataloader(
            hdf5_files=fold_datasets["val"],
            batch_size=batch_size,
            bucket_ranges=bucket_ranges,
            shuffle=False,
            **common_kwargs,
        )
        test_loader, _ = create_bucketed_sequence_dataloader(
            hdf5_files=fold_datasets["test"],
            batch_size=batch_size,
            bucket_ranges=bucket_ranges,
            shuffle=False,
            **common_kwargs,
        )

    return val_loader, test_loader


# ====================================================================== #
#  Cross-Fold Aggregation                                                  #
# ====================================================================== #


def _aggregate_transformer_results(
    all_fold_results: List[Dict],
) -> Dict:
    """Aggregate evaluation metrics across folds.

    Computes mean +/- std of thresholds, validation metrics, test
    metrics, decision-point metrics per metric type, and ROC AUC.

    Args:
        all_fold_results: List of per-fold result dicts from
            ``evaluate_single_fold_transformer``.

    Returns:
        Aggregated metrics dict.
    """
    def _mean_std(vals: List[float]) -> Tuple[float, float]:
        arr = np.array([v for v in vals if v is not None], dtype=float)
        if len(arr) == 0:
            return 0.0, 0.0
        return float(arr.mean()), float(arr.std())

    def _collect(key: str) -> List[float]:
        return [
            r[key] for r in all_fold_results
            if r.get(key) is not None
        ]

    agg: Dict[str, Any] = {"n_folds": len(all_fold_results)}

    # Thresholds.
    thresholds = _collect("primary_threshold")
    agg["threshold_mean"], agg["threshold_std"] = _mean_std(thresholds)

    # Validation metrics.
    for key in ("validation_sensitivity", "validation_specificity",
                "validation_fpr", "validation_accuracy"):
        vals = _collect(key)
        mean, std = _mean_std(vals)
        agg[f"{key}_mean"] = mean
        agg[f"{key}_std"] = std

    # Test metrics (mean-across-bins).
    for key in ("test_sensitivity_mean", "test_specificity_mean",
                "test_fpr_mean"):
        vals = _collect(key)
        mean, std = _mean_std(vals)
        base = key.replace("_mean", "")
        agg[f"{base}_cross_fold_mean"] = mean
        agg[f"{base}_cross_fold_std"] = std

    # Decision-point metrics per metric type.
    for mt in ("instantaneous", "committed_cumulative", "committed_overall"):
        for metric in ("fpr_at_decision", "sensitivity_at_decision"):
            vals = []
            for r in all_fold_results:
                dp = r.get("decision_point_metrics", {}).get(mt, {})
                val = dp.get(metric)
                if val is not None:
                    vals.append(val)
            mean, std = _mean_std(vals)
            agg[f"test_{metric}_{mt}_mean"] = mean
            agg[f"test_{metric}_{mt}_std"] = std

        # Per-type threshold.
        type_thresholds = []
        for r in all_fold_results:
            t = r.get("all_thresholds", {}).get(mt)
            if t is not None:
                type_thresholds.append(t)
        mean, std = _mean_std(type_thresholds)
        agg[f"threshold_{mt}_mean"] = mean
        agg[f"threshold_{mt}_std"] = std

    # ROC AUC.
    auc_vals = _collect("roc_auc")
    agg["roc_auc_mean"], agg["roc_auc_std"] = _mean_std(auc_vals)

    cc_auc_vals = _collect("cc_roc_auc")
    agg["cc_roc_auc_mean"], agg["cc_roc_auc_std"] = _mean_std(cc_auc_vals)

    return agg


def aggregate_transformer_results(
    output_base_dir: str,
    fold_ids: Optional[List[int]] = None,
    fold_results: Optional[List[Dict]] = None,
    exclude_last_minutes: float = 30.0,
) -> Dict:
    """Top-level cross-fold aggregation with plot generation.

    Args:
        output_base_dir: Root output directory with fold_* subdirs.
        fold_ids: Specific fold IDs to aggregate (None = all found).
        fold_results: Pre-loaded fold results (None = load from disk).
        exclude_last_minutes: Exclude bins near birth.

    Returns:
        Aggregated results dict.
    """
    from model.vae_teb_prediction.classifier.evaluate_classifier import (
        convert_numpy_types,
    )
    from model.transformer.classification.evaluation_plots import (
        plot_aggregated_metrics,
        plot_aggregated_roc,
    )

    base_path = Path(output_base_dir)

    # Load fold results from disk if not provided.
    if fold_results is None:
        fold_results = []
        fold_dirs = sorted(base_path.glob("fold_*"))
        for fd in fold_dirs:
            fid = int(fd.name.split("_")[-1])
            if fold_ids is not None and fid not in fold_ids:
                continue
            fr_path = fd / "fold_results.json"
            if fr_path.exists():
                try:
                    with open(fr_path) as f:
                        fold_results.append(json.load(f))
                except Exception as exc:
                    logger.warning("Failed to load {}: {}", fr_path, exc)

    if not fold_results:
        logger.warning("No fold results found for aggregation.")
        return {"status": "no_results"}

    successful = [
        r for r in fold_results if r.get("status") == "success"
    ]
    logger.info(
        "Aggregating {} successful folds (of {} total)",
        len(successful), len(fold_results),
    )

    # Scalar aggregation.
    aggregated = _aggregate_transformer_results(successful)
    aggregated["status"] = "success"

    # Log summary.
    logger.info(
        "Cross-fold: threshold={:.4f}±{:.4f}, "
        "val_sens={:.3f}±{:.3f}, ROC AUC={:.3f}±{:.3f}",
        aggregated.get("threshold_mean", 0),
        aggregated.get("threshold_std", 0),
        aggregated.get("validation_sensitivity_mean", 0),
        aggregated.get("validation_sensitivity_std", 0),
        aggregated.get("roc_auc_mean", 0),
        aggregated.get("roc_auc_std", 0),
    )

    # Generate aggregated plots.
    agg_plot_dir = base_path / "aggregated_plots"

    # Aggregated ROC.
    all_roc = [
        r["roc_data"] for r in successful
        if "roc_data" in r and r["roc_data"].get("fpr")
    ]
    for r, roc in zip(successful, all_roc):
        roc["auc"] = r.get("roc_auc", 0.0)
    if all_roc:
        plot_aggregated_roc(all_roc, agg_plot_dir, len(successful))

    # Aggregated metric curves per metric type.
    for mt in ("instantaneous", "committed_cumulative", "committed_overall"):
        fold_dfs = []
        for r in successful:
            analysis = r.get("three_metric_analysis", {})
            # The analysis stores summaries, but we need per-bin DataFrames.
            # These are stored in fold_results under three_metric_analysis
            # as summary dicts, not full DataFrames. For cross-fold
            # aggregation, we need to load from the per-fold CSVs or
            # serialized records. Since we store metrics_dict records in
            # _run_three_metric_analysis, check if they're available:
            pass

        # If per-bin DataFrames aren't in fold_results, we skip aggregated
        # metric plots (they require re-running the analysis per fold).
        # The user can re-run evaluation to generate these.

    # Save aggregated results.
    agg_path = base_path / "aggregated_evaluation_results.json"
    _save_json(agg_path, convert_numpy_types(aggregated))
    logger.info("Aggregated results saved to: {}", agg_path)

    return aggregated


# ====================================================================== #
#  Subprocess Entry Point                                                  #
# ====================================================================== #


def _evaluate_single_fold_subprocess(
    fold_dir: str,
    gpu_id: int,
    config_path: str,
    target_fpr: float,
    exclude_last_minutes: float,
    decision_time_hours: float,
    max_gap_multiplier: Optional[float],
    regenerate_predictions: bool,
) -> Dict:
    """Evaluate one fold in a subprocess.

    Sets ``CUDA_VISIBLE_DEVICES`` before CUDA init for GPU isolation.

    Args:
        fold_dir: Path to the fold directory.
        gpu_id: GPU device ID.
        config_path: Path to config YAML.
        target_fpr: Target FPR.
        exclude_last_minutes: Exclude near-birth bins.
        decision_time_hours: Decision time.
        max_gap_multiplier: Gap multiplier.
        regenerate_predictions: Force re-run inference.

    Returns:
        Fold results dict.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        # Load config from fold-local or global path.
        fold_config_path = Path(fold_dir) / "config.yaml"
        if fold_config_path.exists():
            with open(fold_config_path) as f:
                config = yaml.safe_load(f)
        else:
            with open(config_path) as f:
                config = yaml.safe_load(f)

        return evaluate_single_fold_transformer(
            fold_dir=fold_dir,
            config=config,
            device="cuda:0",
            target_fpr=target_fpr,
            exclude_last_minutes=exclude_last_minutes,
            decision_time_hours=decision_time_hours,
            max_gap_multiplier=max_gap_multiplier,
            regenerate_predictions=regenerate_predictions,
        )

    except Exception as exc:
        logger.exception("Fold evaluation failed: {}", fold_dir)
        fold_id = int(Path(fold_dir).name.split("_")[-1]) if "_" in Path(fold_dir).name else 0
        return {
            "fold_id": fold_id,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ====================================================================== #
#  Main Entry Point                                                        #
# ====================================================================== #


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
    aggregate_only: bool = False,
    gpu_ids: Optional[List[int]] = None,
    max_parallel: Optional[int] = None,
    sequential: bool = False,
    fold_timeout_hours: float = 2.0,
) -> Dict:
    """CLI entry point for evaluation.

    Supports three modes:
        - Full evaluation: inference + analysis per fold.
        - Regenerate: re-run inference from best checkpoints.
        - Aggregate only: load existing results, generate summaries.

    Args:
        output_base_dir: Root directory with fold_* subdirectories.
        config_path: Path to config YAML (auto-found if None).
        target_fpr: Target FPR (default from config or 0.2).
        device: CUDA device for sequential mode.
        exclude_last_minutes: Exclude near-birth bins.
        max_gap_multiplier: Gap multiplier for epoch filling.
        decision_time_hours: Decision time in hours.
        fold_ids: Specific fold IDs (None = all).
        regenerate_predictions: Force re-run inference.
        aggregate_only: Skip per-fold eval, aggregate only.
        gpu_ids: GPU IDs for parallel mode.
        max_parallel: Max concurrent processes.
        sequential: Force sequential evaluation.
        fold_timeout_hours: Timeout per fold.

    Returns:
        Aggregated results dict.
    """
    base_path = Path(output_base_dir)

    # Locate config.
    if config_path is None:
        # Try fold-local config first, then module default.
        first_fold = next(base_path.glob("fold_*/config.yaml"), None)
        if first_fold:
            config_path = str(first_fold)
        else:
            config_path = str(
                Path(__file__).parent / "config_classification.yaml"
            )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve defaults from config.
    eval_cfg = config.get("model_config", {}).get("evaluation", {})
    if target_fpr is None:
        target_fpr = float(eval_cfg.get("target_fpr", 0.2))
    if exclude_last_minutes is None:
        exclude_last_minutes = float(
            eval_cfg.get("exclude_last_minutes", 30.0)
        )
    if decision_time_hours is None:
        decision_time_hours = float(
            eval_cfg.get("decision_time_hours", 1.0)
        )
    if max_gap_multiplier is None:
        max_gap_multiplier = eval_cfg.get("max_gap_multiplier")

    # Discover folds.
    fold_dirs = sorted(base_path.glob("fold_*"))
    if fold_ids is not None:
        fold_dirs = [
            d for d in fold_dirs
            if int(d.name.split("_")[-1]) in fold_ids
        ]

    if not fold_dirs:
        logger.error("No fold directories found in {}", base_path)
        return {"status": "no_folds"}

    logger.info(
        "Found {} fold directories in {}", len(fold_dirs), base_path,
    )

    # --- Aggregate-only mode ------------------------------------------ #
    if aggregate_only:
        return aggregate_transformer_results(
            output_base_dir=output_base_dir,
            fold_ids=fold_ids,
            exclude_last_minutes=exclude_last_minutes,
        )

    # --- Per-fold evaluation ------------------------------------------ #
    all_results: List[Dict] = []

    if sequential or gpu_ids is None or len(gpu_ids) <= 1:
        # Sequential evaluation.
        for fold_dir in fold_dirs:
            fold_config_path = fold_dir / "config.yaml"
            cfg = config
            if fold_config_path.exists():
                with open(fold_config_path) as f:
                    cfg = yaml.safe_load(f)

            result = evaluate_single_fold_transformer(
                fold_dir=str(fold_dir),
                config=cfg,
                device=device,
                target_fpr=target_fpr,
                exclude_last_minutes=exclude_last_minutes,
                decision_time_hours=decision_time_hours,
                max_gap_multiplier=max_gap_multiplier,
                regenerate_predictions=regenerate_predictions,
            )
            all_results.append(result)
    else:
        # Parallel multi-GPU evaluation.
        if max_parallel is None:
            max_parallel = min(len(gpu_ids), len(fold_dirs))

        with ProcessPoolExecutor(
            max_workers=max_parallel,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = {}
            for idx, fold_dir in enumerate(fold_dirs):
                gpu_id = gpu_ids[idx % len(gpu_ids)]
                future = executor.submit(
                    _evaluate_single_fold_subprocess,
                    fold_dir=str(fold_dir),
                    gpu_id=gpu_id,
                    config_path=config_path,
                    target_fpr=target_fpr,
                    exclude_last_minutes=exclude_last_minutes,
                    decision_time_hours=decision_time_hours,
                    max_gap_multiplier=max_gap_multiplier,
                    regenerate_predictions=regenerate_predictions,
                )
                futures[future] = fold_dir.name

            try:
                for future in as_completed(
                    futures, timeout=fold_timeout_hours * 3600
                ):
                    name = futures[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                        logger.info("  {} completed", name)
                    except Exception as exc:
                        logger.exception("{} failed:", name)
                        all_results.append({
                            "fold_id": name,
                            "status": "failed",
                            "error": str(exc),
                        })
            except TimeoutError:
                timed_out = [
                    n for fut, n in futures.items() if not fut.done()
                ]
                logger.error("Timed out. Folds: {}", timed_out)
                for future in futures:
                    if not future.done():
                        future.cancel()

    # --- Cross-fold aggregation --------------------------------------- #
    aggregated = aggregate_transformer_results(
        output_base_dir=output_base_dir,
        fold_results=all_results,
        exclude_last_minutes=exclude_last_minutes,
    )

    return aggregated


# ====================================================================== #
#  CLI __main__                                                            #
# ====================================================================== #


def _parse_args():
    """Parse CLI arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the transformer classifier across k-folds."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Root output directory with fold_* subdirectories.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config_classification.yaml.",
    )
    parser.add_argument(
        "--target_fpr", type=float, default=None,
        help="Target FPR for threshold optimisation.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="CUDA device (sequential mode).",
    )
    parser.add_argument(
        "--exclude_last_minutes", type=float, default=None,
        help="Exclude bins within N minutes of birth.",
    )
    parser.add_argument(
        "--decision_time_hours", type=float, default=None,
        help="Decision time in hours before delivery.",
    )
    parser.add_argument(
        "--fold_ids", type=int, nargs="+", default=None,
        help="Specific fold IDs to evaluate.",
    )
    parser.add_argument(
        "--regenerate_predictions", action="store_true",
        help="Force re-run inference from best checkpoints.",
    )
    parser.add_argument(
        "--aggregate_only", action="store_true",
        help="Skip per-fold eval; aggregate existing results.",
    )
    parser.add_argument(
        "--gpu_ids", type=int, nargs="+", default=None,
        help="GPU IDs for parallel evaluation.",
    )
    parser.add_argument(
        "--max_parallel", type=int, default=None,
        help="Max concurrent fold processes.",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Force sequential (single-GPU) evaluation.",
    )
    parser.add_argument(
        "--fold_timeout_hours", type=float, default=2.0,
        help="Timeout per fold in hours.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        output_base_dir=args.output_dir,
        config_path=args.config,
        target_fpr=args.target_fpr,
        device=args.device,
        exclude_last_minutes=args.exclude_last_minutes,
        decision_time_hours=args.decision_time_hours,
        fold_ids=args.fold_ids,
        regenerate_predictions=args.regenerate_predictions,
        aggregate_only=args.aggregate_only,
        gpu_ids=args.gpu_ids,
        max_parallel=args.max_parallel,
        sequential=args.sequential,
        fold_timeout_hours=args.fold_timeout_hours,
    )
