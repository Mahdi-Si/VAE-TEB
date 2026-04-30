"""Per-fold evaluation pipeline for ``guid_cls_v1`` (PRD §11).

End-to-end flow:

1. Locate the best checkpoint inside ``fold_{k}/checkpoints/`` (uses
   ``best.ckpt`` written by Lightning's ``ModelCheckpoint``; falls back to
   the most recently modified ``*.ckpt``).
2. Build the val/test :class:`GuidSequenceDataset`.
3. Run the **prefix sweep**: for every GUID, sweep ``m ∈ {1..N_g}`` and emit
   one row per observed segment with the schema in PRD §11.3.
4. Validate the prediction CSV schema, then call the existing
   ``apply_clinical_decision_rule`` / ``fill_missing_epochs`` /
   ``compute_*_metrics`` / threshold-search utilities from
   :mod:`model.vae_teb_prediction.new_classifier.evaluate_classifier`.
5. Compute the binary GUID-level ROC and a 3-class one-vs-rest ROC + the
   3×3 confusion matrix.
6. Persist all artefacts under ``fold_{k}/evaluation/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import confusion_matrix, roc_curve
from torch.utils.data import DataLoader

from model.vae_teb_prediction.new_classifier.guid_cls_v1.collate import (
    guid_sequence_collate_fn,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.evaluate_3class_metrics import (
    run_3class_evaluation_for_metric_type,
    write_perclass_thresholds_json,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_classifier import (
    GuidClassifierConfig,
    GuidOutcomeClassifier,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_dataset import (
    GuidSequenceDataset,
)


# ---------------------------------------------------------------------------
# Lazy imports of the legacy utilities (so unit tests that don't need them
# can run without dragging in matplotlib / sklearn at import time).
# ---------------------------------------------------------------------------


def _import_legacy_utils():
    """Import + return the dict of CDR / metric / plotting helpers we reuse."""
    from model.vae_teb_prediction.new_classifier.evaluate_classifier import (  # noqa: WPS433
        apply_clinical_decision_rule,
        compute_committed_cumulative_metrics,
        compute_committed_overall_metrics,
        compute_guid_level_roc,
        compute_instantaneous_metrics,
        compute_time_bins,
        convert_numpy_types,
        create_enhanced_subgroup_filters,
        ensure_committed_epochs_filled,
        fill_missing_epochs,
        find_threshold_for_committed_cumulative_fpr_at_1h,
        find_threshold_for_committed_overall_fpr_at_1h,
        find_threshold_for_instantaneous_fpr_at_1h,
        generate_fold_dataset_stats,
        plot_aggregated_roc_curves,
        plot_all_metric_types_for_fold,
        plot_roc_curve,
        plot_single_metric_type,
        plot_subgroup_analysis,
    )
    from model.vae_teb_prediction.classifier.validation_utils import (  # noqa: WPS433
        ensure_epoch_hours,
        log_dataframe_stats,
        validate_predictions_df,
    )

    return dict(
        apply_clinical_decision_rule=apply_clinical_decision_rule,
        compute_committed_cumulative_metrics=compute_committed_cumulative_metrics,
        compute_committed_overall_metrics=compute_committed_overall_metrics,
        compute_guid_level_roc=compute_guid_level_roc,
        compute_instantaneous_metrics=compute_instantaneous_metrics,
        compute_time_bins=compute_time_bins,
        convert_numpy_types=convert_numpy_types,
        create_enhanced_subgroup_filters=create_enhanced_subgroup_filters,
        ensure_committed_epochs_filled=ensure_committed_epochs_filled,
        ensure_epoch_hours=ensure_epoch_hours,
        fill_missing_epochs=fill_missing_epochs,
        find_threshold_for_committed_cumulative_fpr_at_1h=find_threshold_for_committed_cumulative_fpr_at_1h,
        find_threshold_for_committed_overall_fpr_at_1h=find_threshold_for_committed_overall_fpr_at_1h,
        find_threshold_for_instantaneous_fpr_at_1h=find_threshold_for_instantaneous_fpr_at_1h,
        generate_fold_dataset_stats=generate_fold_dataset_stats,
        log_dataframe_stats=log_dataframe_stats,
        plot_aggregated_roc_curves=plot_aggregated_roc_curves,
        plot_all_metric_types_for_fold=plot_all_metric_types_for_fold,
        plot_roc_curve=plot_roc_curve,
        plot_single_metric_type=plot_single_metric_type,
        plot_subgroup_analysis=plot_subgroup_analysis,
        validate_predictions_df=validate_predictions_df,
    )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    """Locate the best checkpoint inside ``checkpoint_dir``.

    Preferred source of truth is the path returned by Lightning's
    ``ModelCheckpoint.best_model_path`` — callers should pass that through
    ``evaluate_single_fold(best_checkpoint_path=...)`` instead of relying on
    this discovery pass.

    Fallback logic (used when the caller did not provide an authoritative
    path): the filename template in :func:`_build_callbacks` is
    ``guid-cls-{epoch:03d}-{val/total_loss:.4f}``, so Lightning writes files
    like ``guid-cls-epoch=017-val_total_loss=0.3214.ckpt``. We parse every
    ``key=value`` pair in the stem, and if any key contains the substring
    ``loss`` we treat the associated value as the checkpoint score and sort
    **ascending** (since all training monitors are minimisation metrics).
    This is strictly better than the previous behaviour of always parsing
    ``stem.split("=")[-1]``, which silently extracted the epoch when the
    monitored metric was absent from the filename.

    Args:
        checkpoint_dir: ``fold_{k}/checkpoints/`` directory.

    Returns:
        Path to the chosen checkpoint.

    Raises:
        FileNotFoundError: When no checkpoint is found.
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir missing: {checkpoint_dir}")

    candidates = sorted(checkpoint_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")

    def _parse_score(stem: str) -> Optional[float]:
        """Extract the loss-like ``key=value`` pair from the stem, if any."""
        # The stem may contain several ``key=value`` pairs separated by
        # ``-``; the rightmost "loss"-named key wins (it is the one the
        # template closest to the end of the filename).
        score: Optional[float] = None
        for token in stem.split("-"):
            if "=" not in token:
                continue
            key, _, value = token.partition("=")
            if "loss" not in key.lower():
                continue
            try:
                score = float(value)
            except ValueError:
                continue
        return score

    scored: List[Tuple[float, Path]] = []
    unparseable: List[Path] = []
    for path in candidates:
        if path.name == "last.ckpt":
            continue
        score = _parse_score(path.stem)
        if score is None:
            unparseable.append(path)
        else:
            scored.append((score, path))

    if scored:
        scored.sort(key=lambda x: x[0])
        chosen = scored[0][1]
        logger.info(
            f"find_best_checkpoint: picked {chosen.name} "
            f"(parsed loss={scored[0][0]:.4f} from {len(scored)} scored candidates)"
        )
        return chosen
    if unparseable:
        unparseable.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        logger.warning(
            f"find_best_checkpoint: no loss-named score in filenames under "
            f"{checkpoint_dir}; falling back to most recent mtime "
            f"({unparseable[0].name}). Pass ``best_checkpoint_path`` to "
            f"evaluate_single_fold to guarantee the best ckpt is used."
        )
        return unparseable[0]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_classifier_from_checkpoint(
    checkpoint_path: Path,
    *,
    classifier_cfg: GuidClassifierConfig,
    device: torch.device,
) -> GuidOutcomeClassifier:
    """Instantiate :class:`GuidOutcomeClassifier` and load its weights.

    Strips Lightning's ``model.`` prefix from state-dict keys and skips the
    ``loss.class_weights_*`` buffers (they are restored from training data).
    Any remaining key mismatch is treated as fatal so evaluation cannot
    silently proceed with a partially-loaded model.

    Args:
        checkpoint_path: Path to a Lightning ``.ckpt``.
        classifier_cfg: Resolved hyperparameter bundle (must match training).
        device: Target device.

    Returns:
        Eval-mode :class:`GuidOutcomeClassifier` on ``device``.
    """
    classifier = GuidOutcomeClassifier(classifier_cfg)
    raw = torch.load(str(checkpoint_path), map_location="cpu")
    state = raw.get("state_dict", raw)
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("loss."):
            continue
        if k.startswith("model.model."):
            cleaned[k[len("model.model."):]] = v
        elif k.startswith("model._orig_mod."):
            cleaned[k[len("model._orig_mod."):]] = v
        elif k.startswith("model."):
            cleaned[k[len("model."):]] = v
        elif k.startswith("_orig_model."):
            cleaned[k[len("_orig_model."):]] = v
        else:
            cleaned[k] = v
    incompatible = classifier.load_state_dict(cleaned, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        missing = ", ".join(incompatible.missing_keys[:8]) or "none"
        unexpected = ", ".join(incompatible.unexpected_keys[:8]) or "none"
        raise RuntimeError(
            "Checkpoint is incompatible with GuidOutcomeClassifier after prefix "
            f"cleanup. missing_keys=[{missing}] "
            f"unexpected_keys=[{unexpected}] "
            f"checkpoint={checkpoint_path}"
        )
    classifier.to(device)
    classifier.eval()
    return classifier


# ---------------------------------------------------------------------------
# Per-position inference (causal autoregressive)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_inference_per_position(
    model: GuidOutcomeClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
) -> pd.DataFrame:
    """One-forward-per-batch inference; emit one row per observed segment.

    Under the causal autoregressive design every position's output already
    represents the model's GUID-level prediction *given history up to that
    position*. We therefore replace the old prefix sweep (``N_g`` forwards
    per GUID) with a single forward per batch and read predictions at every
    valid position.

    Args:
        model: Loaded classifier in eval mode.
        loader: Sequential DataLoader over a :class:`GuidSequenceDataset`.
        device: Compute device.

    Returns:
        DataFrame with the same schema as the legacy prefix sweep (modulo
        dropped ``aux_prob_*`` columns; ``position`` column added; the
        ``prefix_length`` column is retained as an alias of ``position``
        so the legacy aggregator still works).
    """
    rows: List[Dict[str, Any]] = []
    for batch in loader:
        # Move tensor fields to device.
        moved: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(device, non_blocking=True)
            else:
                moved[k] = v
        seg_mask = moved["segment_mask"]                          # (B, N) bool
        epochs = moved["epoch"].cpu().numpy()                     # (B, N)
        cs = moved["cs_label"].cpu().numpy()                      # (B, N)
        bg = moved["bg_label"].cpu().numpy()                      # (B, N)
        tlo = moved["time_from_labor_onset"].cpu().numpy()        # (B, N)
        sso = moved["second_stage_onset"].cpu().numpy()           # (B, N)
        labels_3 = moved["label_3"].cpu().numpy()                 # (B,)
        labels_bin = moved["label_bin"].cpu().numpy()             # (B,)
        target_per_t = moved["target_per_t"].cpu().numpy()        # (B, N, T)
        guids = list(moved["guid"])
        n_per = moved["num_segments"].cpu().numpy()

        out = model(moved)
        prob_3 = out["prob_3"].detach().cpu().numpy()             # (B, N, 3)
        prob_bin = out["prob_bin"].detach().cpu().numpy()         # (B, N)

        B = seg_mask.shape[0]
        for b in range(B):
            n_b = int(n_per[b])
            for n in range(n_b):
                seg_target = int(round(float(target_per_t[b, n].max())))
                if seg_target == 0:
                    # Fully-padded segment shouldn't appear here, but guard.
                    seg_target = int(labels_3[b]) + 1
                position = n + 1
                rows.append(
                    {
                        "guid": str(guids[b]),
                        "epoch": float(epochs[b, n]),
                        "target": int(seg_target),
                        "binary_target": int(seg_target > 1),
                        "predicted_class": int(prob_bin[b, n] >= 0.5),
                        "prob_class_0": float(1.0 - prob_bin[b, n]),
                        "prob_class_1": float(prob_bin[b, n]),
                        "prob_healthy": float(prob_3[b, n, 0]),
                        "prob_acidosis": float(prob_3[b, n, 1]),
                        "prob_hie": float(prob_3[b, n, 2]),
                        "predicted_class_3": int(prob_3[b, n].argmax()),
                        "position": int(position),
                        # Alias: legacy aggregator code reads ``prefix_length``.
                        "prefix_length": int(position),
                        "cs_label": bool(cs[b, n]),
                        "bg_label": bool(bg[b, n]),
                        "tlo_hours": float(tlo[b, n]) / 3600.0
                        if not np.isnan(tlo[b, n])
                        else float("nan"),
                        "sso_hours": float(sso[b, n]) / 3600.0
                        if not np.isnan(sso[b, n])
                        else float("nan"),
                        "guid_binary_target": int(labels_bin[b]),
                        "guid_class_3_target": int(labels_3[b]),
                    }
                )
    return pd.DataFrame(rows)


# Back-compat alias for any external consumers of the old function name.
run_inference_prefix_sweep = run_inference_per_position


# ---------------------------------------------------------------------------
# 3-class diagnostics
# ---------------------------------------------------------------------------


def compute_3class_roc_ovr(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """One-vs-rest 3-class ROC at the last observed prefix per GUID.

    Args:
        df: Predictions DataFrame (must contain ``guid``,
            ``prefix_length``, ``prob_healthy/acidosis/hie``,
            ``guid_class_3_target``).

    Returns:
        Dict ``{class_name: {"fpr": [...], "tpr": [...], "thresholds": [...], "auc": float}}``.
    """
    last = (
        df.sort_values(["guid", "prefix_length"])
        .groupby("guid", as_index=False)
        .tail(1)
    )
    targets = last["guid_class_3_target"].astype(int).values
    out: Dict[str, Dict[str, Any]] = {}
    for class_id, name, prob_col in (
        (0, "healthy", "prob_healthy"),
        (1, "acidosis", "prob_acidosis"),
        (2, "hie", "prob_hie"),
    ):
        y_true = (targets == class_id).astype(int)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            out[name] = {"fpr": [], "tpr": [], "thresholds": [], "auc": float("nan")}
            continue
        scores = last[prob_col].astype(float).values
        fpr, tpr, thr = roc_curve(y_true, scores)
        out[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thr.tolist(),
            "auc": float(sklearn_auc(fpr, tpr)),
        }
    return out


def compute_confusion_matrix_3class(df: pd.DataFrame) -> np.ndarray:
    """3×3 confusion matrix at each GUID's last observed prefix."""
    last = (
        df.sort_values(["guid", "prefix_length"])
        .groupby("guid", as_index=False)
        .tail(1)
    )
    y_true = last["guid_class_3_target"].astype(int).values
    y_pred = last["predicted_class_3"].astype(int).values
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).astype(float)


def plot_three_class_diagnostics(df: pd.DataFrame, output_dir: Path) -> None:
    """Histograms + 3×3 confusion matrix at each GUID's last observed prefix."""
    import matplotlib.pyplot as plt  # noqa: WPS433

    output_dir.mkdir(parents=True, exist_ok=True)
    last = (
        df.sort_values(["guid", "prefix_length"])
        .groupby("guid", as_index=False)
        .tail(1)
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    class_names = ["healthy", "acidosis", "hie"]
    prob_cols = ["prob_healthy", "prob_acidosis", "prob_hie"]
    for ax, name, col in zip(axes, class_names, prob_cols):
        for cls_id, cls_label in enumerate(class_names):
            mask = last["guid_class_3_target"] == cls_id
            if mask.sum() == 0:
                continue
            ax.hist(
                last.loc[mask, col].values,
                bins=30,
                alpha=0.45,
                label=f"target={cls_label} (n={int(mask.sum())})",
            )
        ax.set_title(f"P({name}) at last prefix")
        ax.set_xlabel("probability")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "three_class_probability_hist.png", dpi=150)
    plt.close(fig)

    cm = compute_confusion_matrix_3class(df)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1.0)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(3))
    ax.set_xticklabels(class_names)
    ax.set_yticks(range(3))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("target")
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}\n({int(cm[i, j])})",
                ha="center",
                va="center",
                color="black" if cm_norm[i, j] < 0.5 else "white",
                fontsize=9,
            )
    ax.set_title("3-class confusion (row-normalised)")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_3class.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level evaluation entry point
# ---------------------------------------------------------------------------


def _build_eval_loader(
    cache_path: Path,
    *,
    config: Dict[str, Any],
    batch_size: int,
) -> Tuple[GuidSequenceDataset, DataLoader]:
    """Sequential eval DataLoader.

    ``num_workers`` is forced to 0 because (a) inference happens in the
    parent process which has CUDA initialised — forking workers from such
    a parent is unsafe on Linux — and (b) prefix-sweep inference is
    one-shot so worker parallelism gives no measurable speedup.

    Honours ``model_config.classifier.{n_rel_buckets, rel_bucket_d_max}``
    by binding them into the collate function so the relative-time bucket
    index produced by the dataset matches the model's bias-table size.
    """
    from functools import partial as _partial  # noqa: WPS433

    ds_cfg = config.get("dataset_config", {})
    dataset = GuidSequenceDataset(
        cache_path,
        warmup_left=int(config["model_config"]["classifier"]["warmup_left"]),
        warmup_right=int(config["model_config"]["classifier"]["warmup_right"]),
        min_samples_per_guid=int(ds_cfg.get("min_samples_per_guid", 3)),
        min_valid_weight_fraction=float(ds_cfg.get("min_valid_weight_fraction", 0.1)),
        cross_delivery_censoring=bool(
            config["model_config"]["classifier"]["cross_delivery_censoring"]
        ),
    )
    cls_cfg = config["model_config"]["classifier"]
    collate = _partial(
        guid_sequence_collate_fn,
        rel_time_num_buckets=int(cls_cfg.get("n_rel_buckets", 32)),
        rel_time_d_max=float(cls_cfg.get("rel_bucket_d_max", 40.0)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
        pin_memory=False,
    )
    return dataset, loader


def evaluate_single_fold(
    *,
    fold_dir: Path,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    regenerate_predictions: bool = False,
    best_checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run inference + threshold search + three metric types + plots.

    Args:
        fold_dir: ``run_dir/fold_{k}`` directory.
        config: Parsed YAML config dict.
        device: Compute device. Defaults to ``cuda:0`` when available.
        regenerate_predictions: When True, rerun inference even if cached
            CSVs exist; otherwise reuse them.
        best_checkpoint_path: Optional authoritative best-checkpoint path
            (usually provided by ``train_fold``'s result). When supplied we
            skip the directory scan entirely — this avoids the filename
            parsing fragility in :func:`find_best_checkpoint` and guarantees
            the evaluator scores the exact checkpoint Lightning's
            ``ModelCheckpoint.best_model_path`` promoted during training.

    Returns:
        Result dict with threshold info + headline test metrics. Also writes
        all per-fold artefacts under ``fold_dir/evaluation/``.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    eval_dir = fold_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Cache root is derived from ``fold_dir.parent`` (i.e. the actual run
    # directory) rather than reconstructed from ``out_dir_base/tag`` so that
    # ``--output-dir`` overrides on the orchestrator are honoured. The
    # subdirectory name comes from ``precompute.out_subdir`` to stay aligned
    # with what :func:`precompute_fold_latents` writes.
    out_subdir = config.get("precompute", {}).get(
        "out_subdir", "precomputed_latents"
    )
    cache_root = fold_dir.parent / out_subdir / fold_dir.name
    val_cache = cache_root / "val.hdf5"
    test_cache = cache_root / "test.hdf5"
    assert val_cache.exists(), f"missing val cache {val_cache}"
    assert test_cache.exists(), f"missing test cache {test_cache}"

    # Build classifier config off the cache dimensions.
    cls_cfg = config["model_config"]["classifier"]
    # Peek d_z / d_model_vae from the cache attrs.
    import h5py  # noqa: WPS433

    with h5py.File(val_cache, "r", libver="latest") as fh:
        d_model_vae = int(fh.attrs["d_model"])
        d_z = int(fh.attrs["d_z"])
    head_hidden_raw = cls_cfg.get("head_hidden_dim")
    classifier_cfg = GuidClassifierConfig(
        d_model_vae=d_model_vae,
        d_z=d_z,
        d_seg=int(cls_cfg.get("d_seg", 192)),
        d_model=int(cls_cfg.get("d_model", 256)),
        n_layers=int(cls_cfg.get("n_layers", 3)),
        n_heads=int(cls_cfg.get("n_heads", 4)),
        d_head=int(cls_cfg.get("d_head", 64)),
        d_ff=int(cls_cfg.get("d_ff", 512)),
        n_rel_buckets=int(cls_cfg.get("n_rel_buckets", 32)),
        num_classes_multi=int(cls_cfg.get("num_classes_multi", 3)),
        head_hidden_dim=int(head_hidden_raw) if head_hidden_raw is not None else None,
        causal=bool(cls_cfg.get("causal", True)),
        c_meta_dim=5,
        te_summary_dim=6,
        late_window_steps=75,
        dropout=float(cls_cfg.get("dropout", 0.1)),
    )

    # Load model. Prefer the authoritative best-checkpoint path supplied by
    # the caller (typically ``train_fold``'s result); fall back to the
    # directory scan only when no explicit path is provided.
    if best_checkpoint_path:
        ckpt_path = Path(best_checkpoint_path)
        if not ckpt_path.exists():
            logger.warning(
                f"[{fold_dir.name}] best_checkpoint_path={ckpt_path} does not "
                "exist; falling back to directory scan"
            )
            ckpt_path = find_best_checkpoint(fold_dir / "checkpoints")
    else:
        ckpt_path = find_best_checkpoint(fold_dir / "checkpoints")
    classifier = load_classifier_from_checkpoint(
        ckpt_path, classifier_cfg=classifier_cfg, device=device
    )

    # Build loaders.
    eval_batch = max(
        bs for _, bs in _bucket_batch_sizes(config["general_config"])
    )
    val_ds, val_loader = _build_eval_loader(
        val_cache, config=config, batch_size=eval_batch
    )
    test_ds, test_loader = _build_eval_loader(
        test_cache, config=config, batch_size=eval_batch
    )

    # Inference.
    val_csv = eval_dir / "validation_predictions_raw.csv"
    test_csv = eval_dir / "test_predictions_raw.csv"
    if regenerate_predictions or not val_csv.exists():
        logger.info(f"[{fold_dir.name}] running val inference (prefix sweep)")
        val_df = run_inference_prefix_sweep(classifier, val_loader, device=device)
        val_df.to_csv(val_csv, index=False)
    else:
        val_df = pd.read_csv(val_csv)
    if regenerate_predictions or not test_csv.exists():
        logger.info(f"[{fold_dir.name}] running test inference (prefix sweep)")
        test_df = run_inference_prefix_sweep(classifier, test_loader, device=device)
        test_df.to_csv(test_csv, index=False)
    else:
        test_df = pd.read_csv(test_csv)

    utils = _import_legacy_utils()
    utils["validate_predictions_df"](val_df, "Validation")
    utils["validate_predictions_df"](test_df, "Test")

    val_df = utils["ensure_epoch_hours"](val_df)
    test_df = utils["ensure_epoch_hours"](test_df)

    eval_cfg = config.get("evaluation", {}) or {}
    target_fpr = float(eval_cfg.get("target_fpr", 0.2))
    decision_time_hours = float(eval_cfg.get("decision_time_hours", 1.0))
    exclude_last_minutes = float(eval_cfg.get("exclude_last_minutes", 30.0))
    max_gap_multiplier = eval_cfg.get("max_gap_multiplier")

    # Threshold search on validation.
    logger.info(f"[{fold_dir.name}] running threshold searches on validation")
    thr_inst, info_inst = utils["find_threshold_for_instantaneous_fpr_at_1h"](
        val_df,
        target_fpr=target_fpr,
        time_window_hours=decision_time_hours,
        max_gap_multiplier=max_gap_multiplier,
    )
    thr_cum, info_cum = utils["find_threshold_for_committed_cumulative_fpr_at_1h"](
        val_df,
        target_fpr=target_fpr,
        time_window_hours=decision_time_hours,
        max_gap_multiplier=max_gap_multiplier,
    )
    thr_overall, info_overall = utils[
        "find_threshold_for_committed_overall_fpr_at_1h"
    ](
        val_df,
        target_fpr=target_fpr,
        time_window_hours=decision_time_hours,
        max_gap_multiplier=max_gap_multiplier,
    )

    # Apply CDR + persist clinical CSVs (use overall threshold as primary).
    val_clinical = utils["apply_clinical_decision_rule"](val_df, thr_overall, verify=True)
    val_clinical.to_csv(eval_dir / "validation_predictions_clinical.csv", index=False)
    test_clinical = utils["apply_clinical_decision_rule"](test_df, thr_overall, verify=True)
    test_clinical.to_csv(eval_dir / "test_predictions_clinical.csv", index=False)

    # Three-metric-type analysis on the test set, each with its own threshold.
    metrics_dir = eval_dir / "three_metric_types"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metric_summaries: Dict[str, Any] = {}
    for metric_type, threshold in (
        ("instantaneous", thr_inst),
        ("committed_cumulative", thr_cum),
        ("committed_overall", thr_overall),
    ):
        sub_dir = metrics_dir / metric_type
        sub_dir.mkdir(parents=True, exist_ok=True)
        df_clinical = utils["apply_clinical_decision_rule"](
            test_df, threshold, verify=False
        )
        if metric_type != "instantaneous":
            df_clinical = utils["fill_missing_epochs"](
                df_clinical,
                max_gap_multiplier=max_gap_multiplier,
                fill_until_birth=True,
                birth_epoch_seconds=0.0,
            )
        df_clinical = utils["ensure_epoch_hours"](df_clinical)
        bins = utils["compute_time_bins"](
            df_clinical, exclude_last_minutes=exclude_last_minutes
        )
        if metric_type == "instantaneous":
            metrics_df = utils["compute_instantaneous_metrics"](df_clinical, bins, None)
        elif metric_type == "committed_cumulative":
            metrics_df = utils["compute_committed_cumulative_metrics"](
                df_clinical, bins, None
            )
        else:
            metrics_df = utils["compute_committed_overall_metrics"](df_clinical, bins, None)
        utils["plot_single_metric_type"](metrics_df, metric_type, sub_dir)
        utils["plot_subgroup_analysis"](
            df_clinical,
            bins,
            metric_type,
            utils["create_enhanced_subgroup_filters"](),
            sub_dir / "subgroups",
            title_suffix=f" - {fold_dir.name}",
        )
        # Per-class metrics + per-class subgroup stratification +
        # binary-by-underlying-class views (Phase 2 of the eval expansion).
        # All artefacts land under ``sub_dir/per_class/`` and
        # ``sub_dir/binary_by_underlying_class/`` so the legacy directory
        # tree above is unchanged.
        try:
            run_3class_evaluation_for_metric_type(
                df_clinical,
                time_bins=bins,
                metric_type=metric_type,
                output_dir=sub_dir,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                f"[{fold_dir.name}] 3-class metrics ({metric_type}) failed: {exc}"
            )
        metric_summaries[metric_type] = {
            "threshold": float(threshold),
            "n_bins": int(len(metrics_df)),
        }

    # Binary GUID-level ROC.
    roc_data = utils["compute_guid_level_roc"](
        test_df, decision_time_hours=decision_time_hours
    )
    pd.DataFrame(
        {
            "fpr": roc_data["fpr"],
            "tpr": roc_data["tpr"],
            "thresholds": roc_data["thresholds"],
        }
    ).to_csv(eval_dir / "roc_binary_data.csv", index=False)
    utils["plot_roc_curve"](
        roc_data,
        eval_dir / "roc_binary.png",
        title_suffix=f" — {fold_dir.name}",
        threshold=thr_overall,
    )

    # 3-class one-vs-rest ROC + confusion matrix + diagnostic plots.
    three_class_roc = compute_3class_roc_ovr(test_df)
    pd.DataFrame(
        [
            {"class": k, "auc": v["auc"]}
            for k, v in three_class_roc.items()
        ]
    ).to_csv(eval_dir / "roc_3class_data.csv", index=False)
    plot_three_class_diagnostics(
        test_df, eval_dir / "three_metric_types" / "three_class_diagnostics"
    )

    # Dataset statistics (PRD §11.8 / §14.4). Best-effort: the legacy helper
    # produces dataset_overview.pdf, subgroup_overview.pdf, etc. under the
    # ``dataset_stats/`` subtree. Failures are logged but non-fatal.
    try:
        ds_stats_dir = eval_dir / "three_metric_types" / "dataset_stats"
        ds_stats_dir.mkdir(parents=True, exist_ok=True)
        ds_time_bins = utils["compute_time_bins"](
            utils["ensure_epoch_hours"](test_df),
            exclude_last_minutes=exclude_last_minutes,
        )
        utils["generate_fold_dataset_stats"](
            test_df,
            ds_time_bins,
            ds_stats_dir,
            title_suffix=f"Test Set — {fold_dir.name}",
        )
    except Exception as exc:  # pragma: no cover - external code, optional artefact
        logger.warning(f"[{fold_dir.name}] generate_fold_dataset_stats failed: {exc}")

    threshold_info: Dict[str, Any] = {
        "threshold_instantaneous": float(thr_inst),
        "threshold_cumulative": float(thr_cum),
        "threshold_overall": float(thr_overall),
        "validation_instantaneous": utils["convert_numpy_types"](info_inst),
        "validation_cumulative": utils["convert_numpy_types"](info_cum),
        "validation_overall": utils["convert_numpy_types"](info_overall),
        "roc_auc_binary": float(roc_data.get("auc", float("nan"))),
        "roc_auc_3class_ovr": {
            k: float(v["auc"]) if v["auc"] == v["auc"] else None  # NaN guard
            for k, v in three_class_roc.items()
        },
    }
    (eval_dir / "threshold_info.json").write_text(
        json.dumps(threshold_info, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Per-class OvR thresholds (secondary diagnostic — primary 3-class
    # reporting stays argmax). Written to a separate file so the existing
    # ``threshold_info.json`` schema is unchanged.
    try:
        write_perclass_thresholds_json(
            val_df,
            output_path=eval_dir / "perclass_thresholds.json",
            target_fpr=target_fpr,
            decision_time_hours=decision_time_hours,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"[{fold_dir.name}] perclass threshold search failed: {exc}")

    # 3×3 confusion matrix at the overall threshold.
    cm = compute_confusion_matrix_3class(test_df)
    pd.DataFrame(cm, index=["healthy", "acidosis", "hie"], columns=["healthy", "acidosis", "hie"]).to_csv(
        eval_dir / "three_metric_types" / "three_class_diagnostics" / "confusion_matrix_3class.csv"
    )

    # Final fold-level results.
    fold_results = {
        "fold_id": int(fold_dir.name.split("_")[-1]),
        "checkpoint_path": str(ckpt_path),
        "n_val_guids": len(val_ds),
        "n_test_guids": len(test_ds),
        "metric_summaries": metric_summaries,
        "threshold_info": threshold_info,
    }
    (fold_dir / "evaluation_results.json").write_text(
        json.dumps(utils["convert_numpy_types"](fold_results), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info(
        f"[{fold_dir.name}] evaluation done: thr_overall={thr_overall:.4f} "
        f"AUC={threshold_info['roc_auc_binary']:.4f}"
    )
    return fold_results


def _bucket_batch_sizes(general_cfg: Dict[str, Any]) -> List[Tuple[Tuple[int, int], int]]:
    """Mirror of ``single_fold_trainer._bucket_batch_sizes`` for eval-only use."""
    raw = general_cfg.get("batch_size_by_bucket", {}) or {}
    buckets: List[Tuple[Tuple[int, int], int]] = []
    if raw:
        for key, bs in raw.items():
            lo_s, hi_s = str(key).split("_")
            buckets.append(((int(lo_s), int(hi_s)), int(bs)))
        buckets.sort(key=lambda x: x[0][0])
    if not buckets:
        buckets = [
            ((1, 5), 16),
            ((6, 12), 12),
            ((13, 20), 8),
            ((21, 40), 4),
        ]
    return buckets


__all__ = [
    "evaluate_single_fold",
    "find_best_checkpoint",
    "load_classifier_from_checkpoint",
    "run_inference_prefix_sweep",
    "compute_3class_roc_ovr",
    "compute_confusion_matrix_3class",
    "plot_three_class_diagnostics",
]
