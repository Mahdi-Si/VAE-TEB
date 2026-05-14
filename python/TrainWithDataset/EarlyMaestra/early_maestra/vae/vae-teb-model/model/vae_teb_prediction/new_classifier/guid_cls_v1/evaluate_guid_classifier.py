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
    run_3class_global_diagnostics,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_classifier import (
    GuidClassifierConfig,
    GuidOutcomeClassifier,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_dataset import (
    GuidSequenceDataset,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.logging_utils import (
    append_jsonl,
)


def _eval_summary_path(fold_dir: Path) -> Path:
    """Resolve the per-fold ``evaluation_summary.jsonl`` path.

    Args:
        fold_dir: The per-fold output directory.

    Returns:
        Path to ``fold_dir/logs/evaluation_summary.jsonl``. The parent
        directory is created if missing.
    """
    logs_dir = fold_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / "evaluation_summary.jsonl"


def _record_eval_event(fold_dir: Path, event: str, **payload: Any) -> None:
    """Append one structured event line to ``evaluation_summary.jsonl``.

    Best-effort: failures inside the logging path must never break
    evaluation (they would mask the real result). Each record carries
    an ISO timestamp + ``event`` discriminator so a downstream parser
    can group by phase (``"val_inference"``, ``"test_inference"``,
    ``"threshold_search"``, ``"evaluation_done"``, ``"evaluation_failed"``).
    """
    try:
        from datetime import datetime, timezone

        record = {
            "event": str(event),
            "iso_timestamp": datetime.now(timezone.utc).isoformat(),
            "fold": fold_dir.name,
            **payload,
        }
        append_jsonl(_eval_summary_path(fold_dir), record)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            f"[{fold_dir.name}] _record_eval_event({event}) failed: {exc}"
        )


# ---------------------------------------------------------------------------
# Lazy imports of the in-pipeline clinical-metric utilities (so unit tests
# that don't need them can run without dragging in matplotlib / sklearn at
# import time). All helpers live in ``clinical_metrics_utils`` — a sibling
# module forked from the legacy ``new_classifier/evaluate_classifier.py``
# with all legacy-model orchestration removed. There is no remaining
# coupling to ``vae_teb_prediction.classifier`` or
# ``vae_teb_prediction.new_classifier.prediction_classification_model``.
# ---------------------------------------------------------------------------


def _import_legacy_utils():
    """Import + return the dict of CDR / metric / plotting helpers we reuse."""
    from model.vae_teb_prediction.new_classifier.guid_cls_v1.clinical_metrics_utils import (  # noqa: WPS433
        apply_clinical_decision_rule,
        compute_committed_cumulative_metrics,
        compute_committed_overall_metrics,
        compute_guid_level_roc,
        compute_instantaneous_metrics,
        compute_time_bins,
        convert_numpy_types,
        create_enhanced_subgroup_filters,
        ensure_committed_epochs_filled,
        ensure_epoch_hours,
        fill_missing_epochs,
        find_threshold_for_committed_cumulative_fpr_at_1h,
        find_threshold_for_committed_overall_fpr_at_1h,
        find_threshold_for_instantaneous_fpr_at_1h,
        generate_fold_dataset_stats,
        log_dataframe_stats,
        plot_aggregated_roc_curves,
        plot_all_metric_types_for_fold,
        plot_roc_curve,
        plot_single_metric_type,
        plot_subgroup_analysis,
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

    Discovery order:

    1. ``checkpoint_dir / "best.ckpt"`` — current convention. ``_build_callbacks``
       configures :class:`ModelCheckpoint` with ``filename="best"``,
       ``save_top_k=1``, and ``auto_insert_metric_name=False``, so Lightning
       overwrites a single ``best.ckpt`` in place at the top of
       ``checkpoints/``. This is the fast path and always wins when present.
    2. Recursive ``rglob("*.ckpt")`` fallback for legacy layouts (e.g. older
       runs whose filename template embedded ``epoch=`` / ``val_total_loss=``
       tokens, or runs where Lightning created a ``val/`` subdirectory because
       it didn't replace ``/`` in the metric name). When multiple legacy files
       are found, the most recent by mtime is returned.

    Args:
        checkpoint_dir: ``fold_{k}/checkpoints/`` directory.

    Returns:
        Path to the chosen checkpoint.

    Raises:
        FileNotFoundError: When no checkpoint is found.
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir missing: {checkpoint_dir}")

    primary = checkpoint_dir / "best.ckpt"
    if primary.is_file():
        logger.info(f"find_best_checkpoint: using {primary}")
        return primary

    candidates = sorted(
        p for p in checkpoint_dir.rglob("*.ckpt") if p.name != "last.ckpt"
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints in {checkpoint_dir} (expected 'best.ckpt')"
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = candidates[0]
    logger.warning(
        f"find_best_checkpoint: 'best.ckpt' not found in {checkpoint_dir}; "
        f"falling back to most recent legacy checkpoint ({chosen.name}). "
        f"Pass best_checkpoint_path to evaluate_single_fold to guarantee "
        f"the intended ckpt is used."
    )
    return chosen


def load_classifier_from_checkpoint(
    checkpoint_path: Path,
    *,
    classifier_cfg: GuidClassifierConfig,
    device: torch.device,
    attach_vae: Optional[torch.nn.Module] = None,
    vae_chunk_size: int = 32,
    discard_vae_keys: bool = False,
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
        attach_vae: Optional :class:`SeqVaeLagAttnV1` (or compatible) module
            to attach as ``classifier.vae`` *before* the state-dict load.
            Pass this for live-VAE checkpoints — the saved checkpoint
            carries ``vae.*`` keys (the stage-2 fine-tuned VAE), and
            without an attachment those keys would surface as
            ``unexpected_keys`` and be rejected by the strict-load check
            below. The attached module is overwritten by the checkpoint's
            ``vae.*`` weights, so its initial parameters need only match
            in shape (e.g. a fresh ``build_vae_from_config`` is fine).
        vae_chunk_size: Forwarded to ``classifier.vae_chunk_size`` when
            ``attach_vae`` is provided. Mirrors the live-train default.
        discard_vae_keys: When ``True`` and ``attach_vae`` is None, drop
            every ``vae.*`` key from the checkpoint state-dict before the
            strict-load check. Use this when evaluating a live-VAE
            checkpoint via cached latents — the fine-tuned VAE weights
            were already consumed upstream (by
            :func:`_ensure_live_vae_eval_caches` to encode the cache)
            and the classifier doesn't need a VAE submodule for cache
            reads. Mutually exclusive with ``attach_vae``: passing both
            raises ``ValueError``.

    Returns:
        Eval-mode :class:`GuidOutcomeClassifier` on ``device``.
    """
    if attach_vae is not None and discard_vae_keys:
        raise ValueError(
            "load_classifier_from_checkpoint: attach_vae and "
            "discard_vae_keys are mutually exclusive — pick one."
        )
    classifier = GuidOutcomeClassifier(classifier_cfg)
    if attach_vae is not None:
        classifier.vae = attach_vae
        classifier.vae_chunk_size = int(vae_chunk_size)
    # ``weights_only=False`` is required because Lightning pickles the
    # full hparam bundle (including the ``LossWeights`` dataclass) into the
    # checkpoint. PyTorch 2.6 changed the default to ``weights_only=True``
    # which refuses to unpickle anything outside its allowlist. Matches the
    # convention in ``train/graph_models_utils.load_checkpoint_strict``;
    # the source of these checkpoints is our own training pipeline so the
    # safety relaxation is intentional.
    raw = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    state = raw.get("state_dict", raw)
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("loss."):
            continue
        if k.startswith("model.model."):
            stripped = k[len("model.model."):]
        elif k.startswith("model._orig_mod."):
            stripped = k[len("model._orig_mod."):]
        elif k.startswith("model."):
            stripped = k[len("model."):]
        elif k.startswith("_orig_model."):
            stripped = k[len("_orig_model."):]
        else:
            stripped = k
        if discard_vae_keys and stripped.startswith("vae."):
            continue
        cleaned[stripped] = v
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

    Column emission honours the per-head gate on ``model``:

      * Binary-head columns (``binary_target``, ``predicted_class``,
        ``prob_class_0``, ``prob_class_1``) are only emitted when the
        model carries an enabled binary head.
      * 3-class-head columns (``prob_healthy``, ``prob_acidosis``,
        ``prob_hie``, ``predicted_class_3``) are only emitted when the
        model carries an enabled 3-class head.

    The always-present columns (``guid``, ``epoch``, ``target``,
    ``position``, ``prefix_length``, ``cs_label``, ``bg_label``,
    ``tlo_hours``, ``sso_hours``, ``t_rel_sso_hours``,
    ``guid_binary_target``, ``guid_class_3_target``) are unconditional —
    downstream code can therefore always assume the GUID-level
    bookkeeping columns are there and only need to check for the
    head-specific probability columns.

    Args:
        model: Loaded classifier in eval mode.
        loader: Sequential DataLoader over a :class:`GuidSequenceDataset`.
        device: Compute device.

    Returns:
        DataFrame with the schema described above.
    """
    enable_three_class = bool(getattr(model, "enable_three_class_head", True))
    enable_binary = bool(getattr(model, "enable_binary_head", True))
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
        prob_3 = (
            out["prob_3"].detach().cpu().numpy() if enable_three_class else None
        )                                                         # (B, N, 3) or None
        prob_bin = (
            out["prob_bin"].detach().cpu().numpy() if enable_binary else None
        )                                                         # (B, N) or None

        B = seg_mask.shape[0]
        for b in range(B):
            n_b = int(n_per[b])
            for n in range(n_b):
                seg_target = int(round(float(target_per_t[b, n].max())))
                if seg_target == 0:
                    # Fully-padded segment shouldn't appear here, but guard.
                    seg_target = int(labels_3[b]) + 1
                position = n + 1
                row: Dict[str, Any] = {
                    "guid": str(guids[b]),
                    "epoch": float(epochs[b, n]),
                    "target": int(seg_target),
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
                    # ``t_rel_sso_hours`` is the explicit signed
                    # "time relative to second-stage onset" axis
                    # consumed by the SSO-anchored evaluation tree
                    # (``sso_metrics_utils``). Numerically equal to
                    # ``sso_hours`` today; kept as a separate column
                    # so the SSO axis is self-documenting in the CSV
                    # schema and resilient to any future rename of
                    # the legacy ``sso_hours`` alias.
                    "t_rel_sso_hours": float(sso[b, n]) / 3600.0
                    if not np.isnan(sso[b, n])
                    else float("nan"),
                    "guid_binary_target": int(labels_bin[b]),
                    "guid_class_3_target": int(labels_3[b]),
                }
                if prob_bin is not None:
                    row["binary_target"] = int(seg_target > 1)
                    row["predicted_class"] = int(prob_bin[b, n] >= 0.5)
                    row["prob_class_0"] = float(1.0 - prob_bin[b, n])
                    row["prob_class_1"] = float(prob_bin[b, n])
                if prob_3 is not None:
                    row["prob_healthy"] = float(prob_3[b, n, 0])
                    row["prob_acidosis"] = float(prob_3[b, n, 1])
                    row["prob_hie"] = float(prob_3[b, n, 2])
                    row["predicted_class_3"] = int(prob_3[b, n].argmax())
                rows.append(row)
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


def _ensure_live_vae_eval_caches(
    *,
    fold_dir: Path,
    config: Dict[str, Any],
    device: torch.device,
    cache_root: Path,
    best_checkpoint_path: Path,
    epoch_min_overrides: Optional[Dict[str, int]] = None,
) -> None:
    """Build val/test latent caches at eval-time for a live-VAE run.

    The live-VAE training path (``vae.freeze_vae=False``) fine-tunes the
    VAE inside the classifier (``LiveGuidSequenceDataset`` consumes raw
    segments and the classifier's ``live_forward`` encodes them on the
    fly), so it never populates ``precomputed_latents/fold_{k}/``. The
    rest of the eval pipeline is built around the cached HDF5 schema, so
    this helper reproduces what :func:`precompute_fold_latents` would
    have written — but driven by the post-stage-2 VAE that lives inside
    the classifier checkpoint, not the original (pre-training) VAE
    checkpoint referenced by ``config['vae']['checkpoint']``.

    The function is idempotent: it runs ``precompute_partition`` only
    for partitions whose cache files are missing, and skips quietly when
    both ``val.hdf5`` and ``test.hdf5`` already exist.

    Args:
        fold_dir: ``run_dir/fold_{k}``.
        config: Parsed YAML config.
        device: Compute device.
        cache_root: Output directory (``run_dir/precomputed_latents/fold_{k}``).
        best_checkpoint_path: Path to the trained classifier checkpoint
            (its ``vae.*`` entries are loaded into a fresh
            :class:`SeqVaeLagAttnV1` and used for the encoding pass).
        epoch_min_overrides: Optional ``{partition: epoch_min}`` dict.
            Honoured per partition; when set for ``"val"``/``"test"`` the
            corresponding cache is built with a wider (or narrower)
            pre-delivery window than ``dataset_config.epoch_min``.

    Notes:
        ``train_stats`` for val/test are taken from the loaded VAE's
        ``mu_post_running_*`` buffers, which are populated during training
        (``vae.fit_latent_stats`` is called early in ``train_fold`` for the
        live path) and refreshed by stage 2's encoder updates.
    """
    from hashlib import sha256 as _sha256  # noqa: WPS433

    from model.vae_teb_prediction.new_classifier.guid_cls_v1.precompute_latents import (  # noqa: WPS433
        build_vae_from_config,
        get_fold_partition_files,
        precompute_partition,
    )

    val_cache = cache_root / "val.hdf5"
    test_cache = cache_root / "test.hdf5"
    needed: List[str] = []
    if not val_cache.exists():
        needed.append("val")
    if not test_cache.exists():
        needed.append("test")
    if not needed:
        return

    cache_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"[{fold_dir.name}] live-VAE eval: missing caches "
        f"{needed}; building from classifier checkpoint "
        f"{best_checkpoint_path.name}"
    )

    # 1) Build a fresh VAE (architecture only) from config and attach to
    # a classifier shell so the full state_dict (classifier + vae.*) loads
    # in one pass. The classifier head bits are discarded — we only need
    # the VAE submodule with stage-2 weights for encoding.
    classifier_cfg = _build_classifier_cfg_from_config(config)
    vae = build_vae_from_config(config, device)
    classifier = load_classifier_from_checkpoint(
        best_checkpoint_path,
        classifier_cfg=classifier_cfg,
        device=device,
        attach_vae=vae,
        vae_chunk_size=int(config.get("vae", {}).get("vae_chunk_size", 32)),
    )
    trained_vae = classifier.vae
    if trained_vae is None:  # pragma: no cover - defensive
        raise RuntimeError(
            f"[{fold_dir.name}] live-VAE eval: classifier.vae is None after "
            "checkpoint load — was this a live-VAE training run?"
        )
    trained_vae.eval()

    # 2) Pull ``train_stats`` from the live VAE's running buffers. These
    # were fit during training (``train_fold`` calls
    # ``vae.fit_latent_stats`` for the live path) and updated by stage-2
    # encoder finetuning.
    mean_t = trained_vae.mu_post_running_mean.detach().cpu().clone()
    var_t = trained_vae.mu_post_running_var.detach().cpu().clone()
    count_t = int(trained_vae.mu_post_running_count.item())
    if count_t <= 0:
        logger.warning(
            f"[{fold_dir.name}] live-VAE eval: VAE running stats are "
            f"empty (count={count_t}). Caches will record zero stats; "
            "consider re-fitting before evaluation."
        )
    train_stats = (mean_t, var_t, count_t)

    # 3) Provenance: record the classifier checkpoint identity in the
    # cache attrs (in place of the original VAE checkpoint SHA) so the
    # cache signature reflects the actual weights used to encode it.
    ckpt_bytes = best_checkpoint_path.read_bytes()
    ckpt_sha = _sha256(ckpt_bytes).hexdigest()

    ds_cfg = config["dataset_config"]
    test_mode = ds_cfg.get("test_mode")
    kfold_base_path = ds_cfg["kfold_base_path"]
    fold_id = int(fold_dir.name.split("_")[-1])
    bs_precompute = int(config.get("precompute", {}).get("batch_size", 32))
    nw_precompute = int(config.get("precompute", {}).get("num_workers", 2))

    overrides = epoch_min_overrides or {}
    for partition in needed:
        files = get_fold_partition_files(
            kfold_base_path, fold_id, partition, test_mode=test_mode
        )
        cache_path = cache_root / f"{partition}.hdf5"
        precompute_partition(
            vae=trained_vae,
            files=files,
            config=config,
            cache_path=cache_path,
            fold_id=fold_id,
            partition=partition,
            device=device,
            batch_size=bs_precompute,
            num_workers=nw_precompute,
            train_stats=train_stats,
            vae_checkpoint_sha256_override=ckpt_sha,
            vae_checkpoint_path_override=str(best_checkpoint_path),
            epoch_min_override=overrides.get(partition),
        )
        logger.info(
            f"[{fold_dir.name}] live-VAE eval: wrote {partition} cache "
            f"-> {cache_path}"
        )

    # Free the encoding model before the eval forward path reloads the
    # classifier (without the VAE attached, since the cached path no
    # longer needs it).
    del classifier, trained_vae, vae
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _build_classifier_cfg_from_config(config: Dict[str, Any]) -> GuidClassifierConfig:
    """Construct the classifier config from YAML.

    Pulled out of :func:`evaluate_single_fold` so the live-VAE precompute
    helper can build an identical config without needing the cache file
    to peek at ``d_model`` / ``d_z``: in the live-VAE branch we read
    those from ``config['vae']['model_kwargs']`` instead.
    """
    cls_cfg = config["model_config"]["classifier"]
    vae_kwargs = config.get("vae", {}).get("model_kwargs", {})
    head_hidden_raw = cls_cfg.get("head_hidden_dim")
    return GuidClassifierConfig(
        d_model_vae=int(vae_kwargs.get("d_model")),
        d_z=int(vae_kwargs.get("d_z")),
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
    # New, head-explicit per-fold layout (replaces the old
    # ``three_metric_types/`` parent that intermingled binary and 3-class
    # artefacts at every level).
    predictions_dir = eval_dir / "predictions"
    binary_head_dir = eval_dir / "binary_head"
    multiclass_head_dir = eval_dir / "multiclass_head"
    diagnostics_dir = multiclass_head_dir / "diagnostics"
    dataset_stats_dir = eval_dir / "dataset_stats"
    for _d in (
        predictions_dir,
        binary_head_dir,
        multiclass_head_dir,
        diagnostics_dir,
        dataset_stats_dir,
    ):
        _d.mkdir(parents=True, exist_ok=True)

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

    # Per-partition window split (§ evaluation.epoch_min_test): when set,
    # ONLY the test cache is built with this wider pre-delivery window;
    # train and val both inherit ``dataset_config.epoch_min`` so the
    # threshold-search operating point is calibrated on the training
    # distribution. Threaded into the live-VAE eval-time precompute
    # below; the frozen-VAE path's cache regeneration relies on the
    # input-signature mismatch path in :func:`precompute_fold_latents`
    # and so does not need to be invoked here.
    # See ``possible_improvements.md`` §3.5 Option C.
    eval_cfg_early = config.get("evaluation", {}) or {}
    epoch_min_test_cfg = eval_cfg_early.get("epoch_min_test")
    epoch_min_overrides_eval: Dict[str, int] = {}
    if epoch_min_test_cfg is not None:
        epoch_min_overrides_eval = {"test": int(epoch_min_test_cfg)}

    # Live-VAE training (vae.freeze_vae=False) doesn't precompute latents
    # — the classifier's live_forward encodes raw segments. The downstream
    # eval path is built around the cached HDF5 schema, so when the caches
    # are missing we run a one-off precompute pass driven by the *trained*
    # VAE (the one inside the classifier checkpoint, not the original VAE
    # checkpoint). This reproduces what precompute_fold_latents would have
    # written and lets the rest of the pipeline run unchanged.
    freeze_vae = bool(config.get("vae", {}).get("freeze_vae", True))
    if not freeze_vae and (not val_cache.exists() or not test_cache.exists()):
        if best_checkpoint_path:
            ckpt_for_cache = Path(best_checkpoint_path)
            if not ckpt_for_cache.exists():
                ckpt_for_cache = find_best_checkpoint(fold_dir / "checkpoints")
        else:
            ckpt_for_cache = find_best_checkpoint(fold_dir / "checkpoints")
        _ensure_live_vae_eval_caches(
            fold_dir=fold_dir,
            config=config,
            device=device,
            cache_root=cache_root,
            best_checkpoint_path=ckpt_for_cache,
            epoch_min_overrides=epoch_min_overrides_eval or None,
        )

    assert val_cache.exists(), f"missing val cache {val_cache}"
    assert test_cache.exists(), f"missing test cache {test_cache}"

    # Window-mismatch guard for the *frozen-VAE* re-eval path. Caches
    # built during training are keyed on a partition-specific window
    # (train + val always use ``dataset_config.epoch_min``; test widens
    # to ``evaluation.epoch_min_test`` when set — §3.5 Option C). If a
    # user re-runs eval after changing either window, the stored cache
    # may not match what the YAML now expects. The signature mismatch
    # is only acted on by ``precompute_fold_latents`` (which we don't
    # call from the eval path for frozen-VAE), so we compare the stored
    # summary here and refuse to proceed on mismatch.
    if freeze_vae:
        import h5py  # noqa: WPS433
        ds_cfg_window = (config.get("dataset_config", {}) or {}).get("epoch_min")
        # Expected per-partition windows. ``None`` means "no check"
        # (e.g. ``dataset_config.epoch_min`` not configured, or
        # ``epoch_min_test`` not set so test inherits the train window).
        expected_windows: Dict[str, Optional[int]] = {
            "val": int(ds_cfg_window) if ds_cfg_window is not None else None,
        }
        if epoch_min_overrides_eval:
            expected_windows.update(
                {k: int(v) for k, v in epoch_min_overrides_eval.items()}
            )
        elif ds_cfg_window is not None:
            # When no test override is configured, test should match train.
            expected_windows["test"] = int(ds_cfg_window)
        for partition, expected_window in expected_windows.items():
            if expected_window is None:
                continue
            cache_file = cache_root / f"{partition}.hdf5"
            try:
                with h5py.File(cache_file, "r", libver="latest") as fh:
                    summary_raw = fh.attrs.get("cache_input_summary_json", "")
            except Exception as exc:
                raise RuntimeError(
                    f"could not read cache attrs from {cache_file} to "
                    f"verify epoch_min: {exc}"
                ) from exc
            try:
                summary = json.loads(str(summary_raw)) if summary_raw else {}
            except json.JSONDecodeError:
                summary = {}
            cached_window = (summary.get("dataset", {}) or {}).get("epoch_min")
            if cached_window is None:
                logger.warning(
                    f"[{fold_dir.name}] {partition}.hdf5 has no "
                    "cache_input_summary_json; cannot verify epoch_min "
                    "— proceeding but window may be stale"
                )
                continue
            if int(cached_window) != int(expected_window):
                expected_source = (
                    "evaluation.epoch_min_test"
                    if partition == "test"
                    else "dataset_config.epoch_min"
                )
                raise RuntimeError(
                    f"[{fold_dir.name}] {partition} cache was built with "
                    f"epoch_min={cached_window} but {expected_source}"
                    f"={expected_window}. Frozen-VAE re-eval cannot adjust "
                    "the window without rebuilding the cache. Either:\n"
                    f"  (a) re-run precompute_fold_latents with the "
                    f"matching epoch_min for partition {partition!r},\n"
                    "  (b) clear the stale cache and re-run training, or\n"
                    f"  (c) adjust {expected_source} to match the cached "
                    "window."
                )

    # Build classifier config off the cache dimensions.
    cls_cfg = config["model_config"]["classifier"]
    # Per-head enable flags (YAML: ``model_config.classifier.heads``).
    # Legacy configs without the ``heads`` block default both flags to
    # True so existing artefacts are unaffected.
    heads_cfg = cls_cfg.get("heads", {}) or {}
    enable_three_class_head = bool(
        (heads_cfg.get("three_class") or {}).get("enabled", True)
    )
    enable_binary_head = bool(
        (heads_cfg.get("binary") or {}).get("enabled", True)
    )
    if not (enable_three_class_head or enable_binary_head):
        raise ValueError(
            f"[{fold_dir.name}] both classifier heads are disabled in "
            "model_config.classifier.heads — eval has nothing to score."
        )
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
        enable_three_class_head=enable_three_class_head,
        enable_binary_head=enable_binary_head,
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
    # For live-VAE runs the checkpoint carries ``vae.*`` keys (stage-2
    # fine-tuned weights + running latent stats). Those weights have
    # already been consumed by ``_ensure_live_vae_eval_caches`` to encode
    # the val/test caches above; the classifier itself reads from cache
    # and doesn't need the VAE submodule, so we drop the ``vae.*`` keys
    # before the strict-load check rather than instantiating a VAE we'd
    # immediately throw away.
    classifier = load_classifier_from_checkpoint(
        ckpt_path,
        classifier_cfg=classifier_cfg,
        device=device,
        discard_vae_keys=not freeze_vae,
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

    # Inference. Predictions CSVs live under ``predictions/`` (new
    # layout); legacy ``validation_predictions_raw.csv`` /
    # ``test_predictions_raw.csv`` files at ``eval_dir`` root from older
    # runs are still readable when present.
    legacy_val_csv = eval_dir / "validation_predictions_raw.csv"
    legacy_test_csv = eval_dir / "test_predictions_raw.csv"
    val_csv = predictions_dir / "validation_raw.csv"
    test_csv = predictions_dir / "test_raw.csv"
    if not val_csv.exists() and legacy_val_csv.exists():
        val_csv = legacy_val_csv
    if not test_csv.exists() and legacy_test_csv.exists():
        test_csv = legacy_test_csv
    if regenerate_predictions or not val_csv.exists():
        logger.info(f"[{fold_dir.name}] running val inference (prefix sweep)")
        import time as _time_inf

        _t0 = _time_inf.perf_counter()
        val_df = run_inference_prefix_sweep(classifier, val_loader, device=device)
        val_df.to_csv(val_csv, index=False)
        _record_eval_event(
            fold_dir,
            "val_inference",
            seconds=float(_time_inf.perf_counter() - _t0),
            n_rows=int(len(val_df)),
            n_guids=int(val_df["guid"].nunique()) if "guid" in val_df.columns else -1,
            output_csv=str(val_csv),
        )
    else:
        val_df = pd.read_csv(val_csv)
        _record_eval_event(
            fold_dir,
            "val_inference_cached",
            n_rows=int(len(val_df)),
            n_guids=int(val_df["guid"].nunique()) if "guid" in val_df.columns else -1,
            input_csv=str(val_csv),
        )
    if regenerate_predictions or not test_csv.exists():
        logger.info(f"[{fold_dir.name}] running test inference (prefix sweep)")
        import time as _time_inf2

        _t0t = _time_inf2.perf_counter()
        test_df = run_inference_prefix_sweep(classifier, test_loader, device=device)
        test_df.to_csv(test_csv, index=False)
        _record_eval_event(
            fold_dir,
            "test_inference",
            seconds=float(_time_inf2.perf_counter() - _t0t),
            n_rows=int(len(test_df)),
            n_guids=int(test_df["guid"].nunique() if "guid" in test_df.columns else -1),
            output_csv=str(test_csv),
        )
    else:
        test_df = pd.read_csv(test_csv)
        _record_eval_event(
            fold_dir,
            "test_inference_cached",
            n_rows=int(len(test_df)),
            n_guids=int(test_df["guid"].nunique() if "guid" in test_df.columns else -1),
            input_csv=str(test_csv),
        )

    utils = _import_legacy_utils()
    utils["validate_predictions_df"](val_df, "Validation")
    utils["validate_predictions_df"](test_df, "Test")

    # Cached CSVs from older runs (pre-3-class extension) lack the
    # ``prob_healthy / prob_acidosis / prob_hie / predicted_class_3``
    # columns the 3-class evaluator needs. Detect the gap up front and
    # force a re-inference so the downstream pipeline sees a complete
    # row schema. Only enforced when the 3-class head is enabled in
    # this run — otherwise those columns are intentionally absent.
    if enable_three_class_head:
        _required_3class_cols = (
            "prob_healthy",
            "prob_acidosis",
            "prob_hie",
            "predicted_class_3",
        )
        for _label, _df, _csv in (
            ("Validation", val_df, val_csv),
            ("Test", test_df, test_csv),
        ):
            _missing = [c for c in _required_3class_cols if c not in _df.columns]
            if _missing:
                raise RuntimeError(
                    f"[{fold_dir.name}] {_label} predictions CSV {_csv} is missing "
                    f"3-class columns {_missing}. This usually means the file was "
                    f"produced by an older (binary-only) run. Re-run with "
                    f"`evaluation.regenerate_predictions: true` (or delete the "
                    f"stale CSV) to regenerate it under the current schema."
                )

    val_df = utils["ensure_epoch_hours"](val_df)
    test_df = utils["ensure_epoch_hours"](test_df)

    eval_cfg = config.get("evaluation", {}) or {}
    target_fpr = float(eval_cfg.get("target_fpr", 0.2))
    decision_time_hours = float(eval_cfg.get("decision_time_hours", 1.0))
    exclude_last_minutes = float(eval_cfg.get("exclude_last_minutes", 30.0))
    max_gap_multiplier = eval_cfg.get("max_gap_multiplier")
    fallback_tolerance_hours = float(eval_cfg.get("fallback_tolerance_hours", 0.5))
    three_class_cfg = (eval_cfg.get("three_class") or {})
    perclass_threshold_search_default = bool(
        three_class_cfg.get("threshold_search_default", True)
    )

    # ------------------------------------------------------------------
    # SSO-eligibility pre-screen (applied once for both metric trees)
    # ------------------------------------------------------------------
    # Drop GUIDs whose ``second_stage_onset`` is absent (NaN) or stored
    # as the ``0.0`` sentinel ("SSO == delivery / not recorded"). The
    # latter would otherwise produce the $-12\,\mathrm{h}$-before-SSO
    # plot artefact on the SSO axis when the per-segment epochs are
    # interpreted as offsets from SSO.
    #
    # The filter is applied *after* the raw prediction CSVs are
    # persisted (so ``test_raw.csv`` / ``validation_raw.csv`` retain
    # the full cohort for forensic inspection) and *before* the
    # threshold search and every downstream metric tree (delivery
    # axis, SSO axis, aggregator) consumes the data, so every
    # downstream consumer sees the same eligible cohort. Gated by
    # ``evaluation.sso_eligibility`` in the YAML.
    sso_elig_cfg = eval_cfg.get("sso_eligibility") or {}
    drop_nan_sso = bool(sso_elig_cfg.get("drop_nan", True))
    drop_zero_sentinel_sso = bool(sso_elig_cfg.get("drop_zero_sentinel", True))
    # Track filter stats for downstream consumers (figure footers, aggregator).
    # ``None`` means the upstream filter was not invoked (legacy behaviour).
    _val_sso_stats: Optional[Dict[str, Any]] = None
    _test_sso_stats: Optional[Dict[str, Any]] = None
    if drop_nan_sso or drop_zero_sentinel_sso:
        from model.vae_teb_prediction.new_classifier.guid_cls_v1 import (  # noqa: WPS433
            sso_metrics_utils as _sso_filter_utils,
        )
        val_df = _sso_filter_utils.ensure_t_rel_sso_hours(val_df)
        test_df = _sso_filter_utils.ensure_t_rel_sso_hours(test_df)
        val_df, _val_sso_stats = _sso_filter_utils.filter_to_sso_eligible_strict(
            val_df,
            drop_nan=drop_nan_sso,
            drop_zero_sentinel=drop_zero_sentinel_sso,
        )
        test_df, _test_sso_stats = _sso_filter_utils.filter_to_sso_eligible_strict(
            test_df,
            drop_nan=drop_nan_sso,
            drop_zero_sentinel=drop_zero_sentinel_sso,
        )
        _sso_filter_utils.write_sso_filter_summary(
            eval_dir / "sso_eligibility_filter_summary.json",
            {
                "val": _val_sso_stats,
                "test": _test_sso_stats,
                "policy": {
                    "drop_nan": drop_nan_sso,
                    "drop_zero_sentinel": drop_zero_sentinel_sso,
                },
            },
        )
        logger.info(
            f"[{fold_dir.name}] SSO-eligibility filter applied: "
            f"val kept={_val_sso_stats['n_kept_guids']}/"
            f"{_val_sso_stats['n_total_guids']} "
            f"(nan={_val_sso_stats['n_dropped_nan']}, "
            f"zero-sentinel={_val_sso_stats['n_dropped_zero_sentinel']}); "
            f"test kept={_test_sso_stats['n_kept_guids']}/"
            f"{_test_sso_stats['n_total_guids']} "
            f"(nan={_test_sso_stats['n_dropped_nan']}, "
            f"zero-sentinel={_test_sso_stats['n_dropped_zero_sentinel']})."
        )
        _record_eval_event(
            fold_dir,
            "sso_eligibility_filter",
            policy_drop_nan=drop_nan_sso,
            policy_drop_zero_sentinel=drop_zero_sentinel_sso,
            val_kept=int(_val_sso_stats["n_kept_guids"]),
            val_dropped_nan=int(_val_sso_stats["n_dropped_nan"]),
            val_dropped_zero=int(_val_sso_stats["n_dropped_zero_sentinel"]),
            test_kept=int(_test_sso_stats["n_kept_guids"]),
            test_dropped_nan=int(_test_sso_stats["n_dropped_nan"]),
            test_dropped_zero=int(_test_sso_stats["n_dropped_zero_sentinel"]),
        )

    # Threshold search on validation (binary-head only — derives the
    # operating point that the clinical decision rule applies to every
    # downstream metric). Under a 3-class-only run there is no binary
    # signal to threshold, so the entire delivery-axis loop is skipped:
    # the CDR-dependent 3-class per-class artefacts also live inside
    # that loop and follow it down. 3-class diagnostics (OvR ROC,
    # confusion matrix, global diagnostics) are CDR-independent and
    # run unconditionally further below.
    metric_summaries: Dict[str, Any] = {}
    perclass_thresholds_by_mode: Dict[str, Any] = {}
    thr_inst: float = float("nan")
    thr_cum: float = float("nan")
    thr_overall: float = float("nan")
    info_inst: Any = None
    info_cum: Any = None
    info_overall: Any = None
    if enable_binary_head:
        logger.info(f"[{fold_dir.name}] running threshold searches on validation")
        import time as _time_thr

        _t_thr = _time_thr.perf_counter()
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
        _record_eval_event(
            fold_dir,
            "threshold_search",
            seconds=float(_time_thr.perf_counter() - _t_thr),
            target_fpr=float(target_fpr),
            decision_time_hours=float(decision_time_hours),
            thr_instantaneous=float(thr_inst) if thr_inst == thr_inst else None,
            thr_cumulative=float(thr_cum) if thr_cum == thr_cum else None,
            thr_overall=float(thr_overall) if thr_overall == thr_overall else None,
        )

        # Apply CDR + persist clinical CSVs (use overall threshold as primary).
        val_clinical = utils["apply_clinical_decision_rule"](val_df, thr_overall, verify=True)
        val_clinical.to_csv(predictions_dir / "validation_clinical.csv", index=False)
        test_clinical = utils["apply_clinical_decision_rule"](test_df, thr_overall, verify=True)
        test_clinical.to_csv(predictions_dir / "test_clinical.csv", index=False)

        # Three-metric-type analysis on the test set, each with its own
        # threshold. Outputs land under ``binary_head/`` (binary-side
        # curves and subgroups) and ``multiclass_head/`` (3-class
        # per-class / subgroup / AUROC / aggregate /
        # confusion-evolution panels). The old ``three_metric_types/``
        # parent that intermingled the two heads is gone.
        binary_metrics_vs_time_dir = binary_head_dir / "metrics_vs_time"
        binary_subgroups_dir = binary_head_dir / "subgroups_vs_time"
        binary_metrics_vs_time_dir.mkdir(parents=True, exist_ok=True)
        binary_subgroups_dir.mkdir(parents=True, exist_ok=True)
        for metric_type, threshold in (
            ("instantaneous", thr_inst),
            ("committed_cumulative", thr_cum),
            ("committed_overall", thr_overall),
        ):
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

            # Persist the binary metrics-vs-time CSV (one file per mode);
            # plotters still write multi-figure PNGs into the same dir.
            metrics_df.to_csv(
                binary_metrics_vs_time_dir / f"{metric_type}.csv", index=False
            )
            per_mode_plot_dir = binary_metrics_vs_time_dir / metric_type
            per_mode_plot_dir.mkdir(parents=True, exist_ok=True)
            utils["plot_single_metric_type"](
                metrics_df,
                metric_type,
                per_mode_plot_dir,
                decision_time_hours=decision_time_hours,
            )

            # Binary subgroup analysis: persist a long-format CSV summarising
            # every subgroup's metric curve, then keep the existing rich
            # multi-PNG renderer in a per-mode sub-directory for human review.
            subgroup_filters = utils["create_enhanced_subgroup_filters"]()
            subgroup_metrics_dict = utils["plot_subgroup_analysis"](
                df_clinical,
                bins,
                metric_type,
                subgroup_filters,
                binary_subgroups_dir / metric_type,
                title_suffix=f" - {fold_dir.name}",
                decision_time_hours=decision_time_hours,
            )
            try:
                sub_long_rows: List[Dict[str, Any]] = []
                for sg_name, sg_df in (subgroup_metrics_dict or {}).items():
                    if sg_df is None or len(sg_df) == 0:
                        continue
                    for _, r in sg_df.iterrows():
                        row = {"subgroup": sg_name}
                        for col in (
                            "bin_center",
                            "sensitivity",
                            "specificity",
                            "fpr",
                            "n_pos",
                            "n_neg",
                            "n",
                        ):
                            if col in r.index:
                                row[col] = r[col]
                        sub_long_rows.append(row)
                if sub_long_rows:
                    pd.DataFrame(sub_long_rows).to_csv(
                        binary_subgroups_dir / f"{metric_type}.csv", index=False
                    )
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    f"[{fold_dir.name}] binary subgroup long-format CSV "
                    f"({metric_type}) failed"
                )

            # Per-class + per-class subgroup + by-underlying-class +
            # AUROC-vs-time + aggregate-vs-time + confusion-evolution
            # artefacts. Now writes to ``multiclass_head/`` and
            # ``binary_head/by_underlying_class_vs_time/`` directly.
            # Only run when the 3-class head is enabled — otherwise the
            # ``prob_healthy/acidosis/hie`` columns are absent and the
            # call would fail downstream.
            if enable_three_class_head:
                try:
                    res_3c = run_3class_evaluation_for_metric_type(
                        df_clinical,
                        time_bins=bins,
                        metric_type=metric_type,
                        eval_root=eval_dir,
                        df_val=val_df if perclass_threshold_search_default else None,
                        target_fpr=target_fpr,
                        decision_time_hours=decision_time_hours,
                        threshold_search_kwargs={
                            "max_gap_multiplier": max_gap_multiplier,
                            "fallback_tolerance_hours": fallback_tolerance_hours,
                        },
                    )
                    perclass_thresholds_by_mode[metric_type] = (
                        res_3c.get("perclass_threshold_info") or {}
                    )
                except Exception:  # pragma: no cover - defensive
                    logger.exception(
                        f"[{fold_dir.name}] 3-class metrics ({metric_type}) failed"
                    )
            metric_summaries[metric_type] = {
                "threshold": float(threshold),
                "n_bins": int(len(metrics_df)),
            }

    # =====================================================================
    # SSO-anchored three-metric-type analysis (parallel tree).
    # =====================================================================
    #
    # Re-runs the three-metric-type analysis with the signed
    # second-stage-onset axis. GUIDs that lack a second-stage timestamp are
    # dropped up front; the figure footers report the dropped count so
    # the omission is auditable. Output lives under
    # ``evaluation/three_metric_types_sso/`` so the original delivery-axis
    # tree is byte-untouched.
    #
    # The whole SSO block is binary-threshold-driven (CDR is applied
    # with ``thr_*`` from the binary threshold search), so it only runs
    # when the binary head is enabled. The inner per-class 3-class call
    # is independently gated on the 3-class head as well.
    if enable_binary_head:
        try:
            from model.vae_teb_prediction.new_classifier.guid_cls_v1 import (  # noqa: WPS433
                sso_metrics_utils as sso_utils,
            )

            sso_root = eval_dir / "three_metric_types_sso"
            sso_root.mkdir(parents=True, exist_ok=True)
            sso_binary_dir = sso_root / "binary_head" / "metrics_vs_time"
            sso_subgroups_dir = sso_root / "binary_head" / "subgroups_vs_time"
            sso_binary_dir.mkdir(parents=True, exist_ok=True)
            sso_subgroups_dir.mkdir(parents=True, exist_ok=True)

            # Pre-screen: drop GUIDs without an SSO timestamp.
            #
            # With the default ``evaluation.sso_eligibility`` policy this
            # filter has already run upstream (catching NaN + zero-sentinel
            # SSO before either tree consumed the data) and is a no-op
            # here; we keep this defence-in-depth call so the SSO-axis
            # tree stays NaN-safe even when the upstream policy is
            # disabled. ``n_sso_dropped`` aggregates the upstream and
            # local drop counts so figure footers report the true total
            # drop for this cohort. The canonical per-fold summary is
            # ``evaluation/sso_eligibility_filter_summary.json``; the
            # legacy ``sso_filter_summary.json`` is left in place for
            # back-compat with older aggregator footers.
            test_sso_df = sso_utils.ensure_t_rel_sso_hours(test_df)
            test_sso_df, sso_drop_stats = sso_utils.filter_to_sso_eligible(test_sso_df)
            _n_sso_dropped_here = int(sso_drop_stats["n_dropped_guids"])
            _n_sso_dropped_upstream = (
                int(_test_sso_stats["n_dropped_guids"])
                if _test_sso_stats is not None
                else 0
            )
            n_sso_dropped = _n_sso_dropped_here + _n_sso_dropped_upstream
            sso_utils.write_sso_filter_summary(
                sso_root / "sso_filter_summary.json", sso_drop_stats
            )

            if test_sso_df.empty:
                logger.warning(
                    f"[{fold_dir.name}] SSO eval skipped: no eligible GUIDs "
                    "(every test GUID is missing second_stage_onset)."
                )
            else:
                sso_metric_summaries: Dict[str, Any] = {}
                for metric_type, threshold in (
                    ("instantaneous", thr_inst),
                    ("committed_cumulative", thr_cum),
                    ("committed_overall", thr_overall),
                ):
                    df_sso_clin = utils["apply_clinical_decision_rule"](
                        test_sso_df, threshold, verify=False
                    )
                    if metric_type != "instantaneous":
                        df_sso_clin = utils["fill_missing_epochs"](
                            df_sso_clin,
                            max_gap_multiplier=max_gap_multiplier,
                            fill_until_birth=True,
                            birth_epoch_seconds=0.0,
                        )
                        df_sso_clin = sso_utils.recompute_t_rel_sso_after_fill(df_sso_clin)
                    else:
                        df_sso_clin = sso_utils.ensure_t_rel_sso_hours(df_sso_clin)

                    sso_bins = sso_utils.compute_sso_time_bins(df_sso_clin)

                    if metric_type == "instantaneous":
                        sso_metrics_df = sso_utils.compute_instantaneous_metrics_sso(
                            df_sso_clin, sso_bins, None
                        )
                    elif metric_type == "committed_cumulative":
                        sso_metrics_df = sso_utils.compute_committed_cumulative_metrics_sso(
                            df_sso_clin, sso_bins, None
                        )
                    else:
                        sso_metrics_df = sso_utils.compute_committed_overall_metrics_sso(
                            df_sso_clin, sso_bins, None
                        )

                    sso_metrics_df.to_csv(
                        sso_binary_dir / f"{metric_type}.csv", index=False
                    )
                    sso_plot_dir = sso_binary_dir / metric_type
                    sso_plot_dir.mkdir(parents=True, exist_ok=True)
                    sso_utils.plot_single_metric_type_sso(
                        sso_metrics_df,
                        metric_type,
                        sso_plot_dir,
                        title_suffix=f" - {fold_dir.name}",
                        n_dropped_guids=n_sso_dropped,
                    )

                    # SSO subgroup analysis (same filters as delivery-axis).
                    try:
                        sso_subgroup_filters = utils["create_enhanced_subgroup_filters"]()
                        sso_sg_dict = sso_utils.plot_subgroup_analysis_sso(
                            df_sso_clin,
                            sso_bins,
                            metric_type,
                            sso_subgroup_filters,
                            sso_subgroups_dir / metric_type,
                            title_suffix=f" - {fold_dir.name}",
                            n_dropped_guids=n_sso_dropped,
                        )
                        sso_utils.persist_subgroup_long_csv(
                            sso_sg_dict, sso_subgroups_dir / f"{metric_type}.csv"
                        )
                    except Exception:  # pragma: no cover - defensive
                        logger.exception(
                            f"[{fold_dir.name}] SSO subgroup ({metric_type}) failed"
                        )

                    # SSO per-class 3-class artefacts (only when 3-class head enabled).
                    if enable_three_class_head:
                        try:
                            run_3class_evaluation_for_metric_type(
                                df_sso_clin,
                                time_bins=sso_bins,
                                metric_type=metric_type,
                                eval_root=sso_root,
                                df_val=val_df if perclass_threshold_search_default else None,
                                target_fpr=target_fpr,
                                decision_time_hours=decision_time_hours,
                                threshold_search_kwargs={
                                    "max_gap_multiplier": max_gap_multiplier,
                                    "fallback_tolerance_hours": fallback_tolerance_hours,
                                },
                                axis_mode="sso",
                                n_dropped_guids=n_sso_dropped,
                            )
                        except Exception:  # pragma: no cover - defensive
                            logger.exception(
                                f"[{fold_dir.name}] SSO 3-class metrics ({metric_type}) failed"
                            )

                    sso_metric_summaries[metric_type] = {
                        "threshold": float(threshold),
                        "n_bins": int(len(sso_metrics_df)),
                        "n_dropped_guids": n_sso_dropped,
                    }
                (sso_root / "metric_summaries.json").write_text(
                    json.dumps(
                        utils["convert_numpy_types"](sso_metric_summaries),
                        indent=2, sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                metric_summaries["sso"] = sso_metric_summaries
        except Exception:  # pragma: no cover - defensive
            logger.exception(f"[{fold_dir.name}] SSO-anchored evaluation failed")

    # Binary GUID-level ROC -> ``binary_head/roc.{csv,png}``.
    roc_data: Dict[str, Any] = {}
    if enable_binary_head:
        roc_data = utils["compute_guid_level_roc"](
            test_df, decision_time_hours=decision_time_hours
        )
        pd.DataFrame(
            {
                "fpr": roc_data["fpr"],
                "tpr": roc_data["tpr"],
                "thresholds": roc_data["thresholds"],
            }
        ).to_csv(binary_head_dir / "roc.csv", index=False)
        utils["plot_roc_curve"](
            roc_data,
            binary_head_dir / "roc.png",
            title_suffix=f" — {fold_dir.name}",
            threshold=thr_overall,
        )

    # 3-class one-vs-rest ROC + confusion matrix + diagnostic plots ->
    # ``multiclass_head/diagnostics/``. Skipped under binary-only runs;
    # in that case the ``multiclass_head/`` directory exists but stays
    # empty (the aggregator auto-detects head presence per fold).
    three_class_roc: Dict[str, Dict[str, Any]] = {}
    if enable_three_class_head:
        three_class_roc = compute_3class_roc_ovr(test_df)
        pd.DataFrame(
            [
                {"class": k, "auc": v["auc"]}
                for k, v in three_class_roc.items()
            ]
        ).to_csv(diagnostics_dir / "roc_ovr.csv", index=False)
        plot_three_class_diagnostics(test_df, diagnostics_dir)
        # Extended per-fold 3-class diagnostics (calibration, PR curves,
        # probability box plots). These are CDR-independent so they only need
        # to be computed once per fold.
        try:
            run_3class_global_diagnostics(test_df, diagnostics_dir)
        except Exception:  # pragma: no cover - defensive
            logger.exception(f"[{fold_dir.name}] 3-class global diagnostics failed")

    # Dataset statistics (PRD §11.8 / §14.4). Best-effort: the legacy
    # helper produces dataset_overview.pdf, subgroup_overview.pdf, etc.
    # under the top-level ``dataset_stats/`` subtree (promoted out of
    # ``three_metric_types/`` — it is time-independent). Failures are
    # logged but non-fatal.
    try:
        ds_time_bins = utils["compute_time_bins"](
            utils["ensure_epoch_hours"](test_df),
            exclude_last_minutes=exclude_last_minutes,
        )
        utils["generate_fold_dataset_stats"](
            test_df,
            ds_time_bins,
            dataset_stats_dir,
            title_suffix=f"Test Set — {fold_dir.name}",
        )
    except Exception:  # pragma: no cover - external code, optional artefact
        logger.exception(f"[{fold_dir.name}] generate_fold_dataset_stats failed")

    # ``threshold_info`` is assembled head-aware so disabled-head keys
    # are simply absent (rather than NaN-filled placeholders the
    # aggregator would have to special-case). The aggregator inspects
    # per-fold artefacts directly, so a missing key here is harmless.
    threshold_info: Dict[str, Any] = {
        "epoch_min_train": (
            int(config.get("dataset_config", {}).get("epoch_min"))
            if config.get("dataset_config", {}).get("epoch_min") is not None
            else None
        ),
        "epoch_min_test": (
            int(epoch_min_test_cfg) if epoch_min_test_cfg is not None else None
        ),
        # Audit trail so downstream consumers can see which heads
        # produced this fold's artefacts without inferring from
        # directory contents.
        "heads_enabled": {
            "binary": bool(enable_binary_head),
            "three_class": bool(enable_three_class_head),
        },
    }
    if enable_binary_head:
        threshold_info.update(
            {
                "threshold_instantaneous": float(thr_inst),
                "threshold_cumulative": float(thr_cum),
                "threshold_overall": float(thr_overall),
                "validation_instantaneous": utils["convert_numpy_types"](info_inst),
                "validation_cumulative": utils["convert_numpy_types"](info_cum),
                "validation_overall": utils["convert_numpy_types"](info_overall),
                "roc_auc_binary": float(roc_data.get("auc", float("nan"))),
            }
        )
    if enable_three_class_head:
        threshold_info["roc_auc_3class_ovr"] = {
            k: float(v["auc"]) if v["auc"] == v["auc"] else None  # NaN guard
            for k, v in three_class_roc.items()
        }
    (eval_dir / "thresholds.json").write_text(
        json.dumps(threshold_info, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Consolidated per-class OvR thresholds across all 3 modes (binary
    # parity for the 3-class head). Written as one file under
    # ``multiclass_head/`` instead of three per-mode JSONs scattered
    # across the old ``three_metric_types/<mode>/`` tree.
    if perclass_thresholds_by_mode:
        (multiclass_head_dir / "perclass_thresholds.json").write_text(
            json.dumps(
                utils["convert_numpy_types"](perclass_thresholds_by_mode),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    # 3×3 confusion matrix at the overall threshold ->
    # ``multiclass_head/diagnostics/confusion_matrix.csv``. Skipped
    # under binary-only runs (the 3-class probability columns are
    # absent and ``compute_confusion_matrix_3class`` would raise).
    if enable_three_class_head:
        cm = compute_confusion_matrix_3class(test_df)
        pd.DataFrame(
            cm,
            index=["healthy", "acidosis", "hie"],
            columns=["healthy", "acidosis", "hie"],
        ).to_csv(diagnostics_dir / "confusion_matrix.csv")

    # Final fold-level results.
    fold_results = {
        "fold_id": int(fold_dir.name.split("_")[-1]),
        "checkpoint_path": str(ckpt_path),
        "n_val_guids": len(val_ds),
        "n_test_guids": len(test_ds),
        "metric_summaries": metric_summaries,
        "threshold_info": threshold_info,
        "heads_enabled": {
            "binary": bool(enable_binary_head),
            "three_class": bool(enable_three_class_head),
        },
    }
    (fold_dir / "evaluation_results.json").write_text(
        json.dumps(utils["convert_numpy_types"](fold_results), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    # Headline log: include whichever metrics we actually computed.
    log_parts = [f"[{fold_dir.name}] evaluation done"]
    if enable_binary_head:
        log_parts.append(f"thr_overall={thr_overall:.4f}")
        log_parts.append(
            f"AUC={threshold_info.get('roc_auc_binary', float('nan')):.4f}"
        )
    if enable_three_class_head:
        log_parts.append("3-class diagnostics emitted")
    logger.info(" ".join(log_parts))
    _record_eval_event(
        fold_dir,
        "evaluation_done",
        enable_binary_head=bool(enable_binary_head),
        enable_three_class_head=bool(enable_three_class_head),
        thr_overall=float(thr_overall) if thr_overall == thr_overall else None,
        roc_auc_binary=(
            float(threshold_info.get("roc_auc_binary", float("nan")))
            if enable_binary_head
            else None
        ),
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
