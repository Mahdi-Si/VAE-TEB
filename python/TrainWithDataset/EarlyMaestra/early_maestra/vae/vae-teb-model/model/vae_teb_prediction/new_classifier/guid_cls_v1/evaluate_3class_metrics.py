"""Per-class + binary-by-underlying-class evaluation pipeline.

The legacy time-binned utilities in
:mod:`model.vae_teb_prediction.new_classifier.evaluate_classifier`
(``apply_clinical_decision_rule``, ``compute_instantaneous_metrics``,
``compute_committed_cumulative_metrics``,
``compute_committed_overall_metrics``, etc.) are **column-generic**: they
read ``binary_target`` / ``clinical_pred`` / ``prob_class_1`` from the
prediction DataFrame.

This module extends evaluation to the 3-class case **without modifying the
legacy module**: we add per-class clinical columns to the DataFrame, then
either call the legacy time-binned metrics with renamed columns, or call
the legacy CDR with a swapped probability column. Three new families of
output are produced per fold:

1. **Per-class time-binned metrics** — for each class
   ``c ∈ {healthy, acidosis, hie}``, compute sensitivity_c / FPR_c /
   specificity_c vs time before delivery using the same three metric
   types (instantaneous / committed_cumulative / committed_overall).
2. **Per-class subgroup stratification** — same metrics × CS/BG subgroups.
3. **Binary by underlying class** — restrict the dataset to
   HEALTHY ∪ {ACIDOSIS} or HEALTHY ∪ {HIE} and recompute the binary
   metrics. Tells us whether the binary classifier detects both unhealthy
   types equally.

Class-id convention (matches the dataset's `target` field):
- ``target == 1`` → HEALTHY
- ``target == 2`` → ACIDOSIS
- ``target == 3`` → HIE

The corresponding probability columns are ``prob_healthy``,
``prob_acidosis``, ``prob_hie``. The 0-based class id (used by
``predicted_class_3``) maps as ``0 → healthy``, ``1 → acidosis``,
``2 → hie``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


# Class names + their related (1-indexed) target id and the prob column.
CLASS_INFO: Tuple[Tuple[str, int, str], ...] = (
    ("healthy", 1, "prob_healthy"),
    ("acidosis", 2, "prob_acidosis"),
    ("hie", 3, "prob_hie"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _legacy_utils() -> Dict[str, Any]:
    """Late-import the legacy utilities (avoids matplotlib at import time)."""
    from model.vae_teb_prediction.new_classifier.evaluate_classifier import (  # noqa: WPS433
        apply_clinical_decision_rule,
        compute_committed_cumulative_metrics,
        compute_committed_overall_metrics,
        compute_instantaneous_metrics,
        create_enhanced_subgroup_filters,
        ensure_committed_epochs_filled,
        find_threshold_for_committed_overall_fpr_at_1h,
        plot_single_metric_type,
    )
    from model.vae_teb_prediction.classifier.validation_utils import (  # noqa: WPS433
        ensure_epoch_hours,
    )

    return dict(
        apply_clinical_decision_rule=apply_clinical_decision_rule,
        compute_committed_cumulative_metrics=compute_committed_cumulative_metrics,
        compute_committed_overall_metrics=compute_committed_overall_metrics,
        compute_instantaneous_metrics=compute_instantaneous_metrics,
        create_enhanced_subgroup_filters=create_enhanced_subgroup_filters,
        ensure_committed_epochs_filled=ensure_committed_epochs_filled,
        ensure_epoch_hours=ensure_epoch_hours,
        find_threshold_for_committed_overall_fpr_at_1h=find_threshold_for_committed_overall_fpr_at_1h,
        plot_single_metric_type=plot_single_metric_type,
    )


METRIC_TYPES: Tuple[str, ...] = ("instantaneous", "committed_cumulative", "committed_overall")


def _legacy_metric_fn(name: str, utils: Mapping[str, Any]) -> Callable[..., pd.DataFrame]:
    return {
        "instantaneous": utils["compute_instantaneous_metrics"],
        "committed_cumulative": utils["compute_committed_cumulative_metrics"],
        "committed_overall": utils["compute_committed_overall_metrics"],
    }[name]


# ---------------------------------------------------------------------------
# Per-class clinical columns
# ---------------------------------------------------------------------------


def add_perclass_clinical_columns(
    df: pd.DataFrame,
    *,
    thresholds: Optional[Mapping[str, float]] = None,
    apply_cdr: bool = True,
) -> pd.DataFrame:
    """Add per-class ``clinical_pred_<class>`` and ``binary_target_<class>``.

    Two prediction modes:
      * ``thresholds is None`` (default, primary): argmax over ``prob_3``.
        ``clinical_pred_<c> = (predicted_class_3 == c).astype(int)`` —
        each row predicts at most one class.
      * Per-class OvR thresholds (secondary diagnostic): for each class
        ``c``, ``clinical_pred_<c> = (prob_<c> >= thresholds[c]).astype(int)``.

    Args:
        df: Predictions DataFrame from :func:`run_inference_per_position`.
            Must contain ``prob_healthy``, ``prob_acidosis``, ``prob_hie``,
            ``predicted_class_3``, ``target`` (1-indexed).
        thresholds: Optional mapping ``{class_name: threshold}``. When set,
            uses OvR-threshold mode and applies the legacy clinical decision
            rule (forward-fill within GUID after first detection) per class.
        apply_cdr: When ``True`` (default) and thresholds are provided,
            apply :func:`apply_clinical_decision_rule` per class so the
            forward-fill semantics match the binary-side pipeline. Has no
            effect in argmax mode (argmax doesn't have a CDR analogue
            because it's not threshold-based).

    Returns:
        Copy of ``df`` with added columns:
            ``clinical_pred_healthy/acidosis/hie``,
            ``binary_target_healthy/acidosis/hie``.
    """
    df = df.copy()
    target = df["target"].astype(int).to_numpy()

    if thresholds is None:
        # Argmax mode (primary 3-class reporting).
        pred_3 = df["predicted_class_3"].astype(int).to_numpy()
        for class_name, target_id, _ in CLASS_INFO:
            class_zero_indexed = target_id - 1
            df[f"clinical_pred_{class_name}"] = (pred_3 == class_zero_indexed).astype(int)
            df[f"binary_target_{class_name}"] = (target == target_id).astype(int)
        return df

    # OvR-threshold mode (secondary). Forward-fill via the legacy CDR.
    utils = _legacy_utils() if apply_cdr else None
    for class_name, target_id, prob_col in CLASS_INFO:
        df[f"binary_target_{class_name}"] = (target == target_id).astype(int)
        thr = float(thresholds[class_name])
        df[f"_model_pred_{class_name}"] = (df[prob_col] >= thr).astype(int)
        if utils is not None:
            # Reuse the legacy CDR by swapping the probability column it reads.
            tmp = df[["guid", "epoch", "binary_target", "prob_class_1"]].copy() if False else None
            del tmp  # noqa: F841 - kept above for documentation
            swap = df.copy()
            swap["prob_class_1"] = swap[prob_col]
            swap["binary_target"] = swap[f"binary_target_{class_name}"]
            swap = utils["apply_clinical_decision_rule"](swap, threshold=thr, verify=False)
            df[f"clinical_pred_{class_name}"] = swap["clinical_pred"].to_numpy()
        else:
            df[f"clinical_pred_{class_name}"] = df[f"_model_pred_{class_name}"]
        df = df.drop(columns=[f"_model_pred_{class_name}"])
    return df


# ---------------------------------------------------------------------------
# Per-class time-binned metrics
# ---------------------------------------------------------------------------


def compute_perclass_time_binned_metrics(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    metric_type: str,
    class_name: str,
    *,
    subgroup_filter: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
) -> pd.DataFrame:
    """Per-class wrapper around the legacy time-binned binary metrics.

    Renames ``clinical_pred_<class>`` → ``clinical_pred`` and
    ``binary_target_<class>`` → ``binary_target`` in a temporary view of
    ``df`` and calls the legacy binary metric fn. Returns the same
    columns the legacy fn returns (``bin_center``, ``sensitivity``,
    ``fpr``, ``specificity``, ``n_*``).

    Args:
        df: DataFrame from :func:`add_perclass_clinical_columns`.
        time_bins: Bin edges in hours-before-delivery (``compute_time_bins``).
        metric_type: One of ``METRIC_TYPES``.
        class_name: One of ``healthy``, ``acidosis``, ``hie``.
        subgroup_filter: Optional row-level boolean mask producer.

    Returns:
        Per-class metrics DataFrame; the columns match the legacy
        binary-side schema so :func:`plot_single_metric_type` can render
        the result without modification.
    """
    if metric_type not in METRIC_TYPES:
        raise ValueError(f"Unknown metric_type {metric_type!r}; expected one of {METRIC_TYPES}")
    utils = _legacy_utils()
    pred_col = f"clinical_pred_{class_name}"
    bt_col = f"binary_target_{class_name}"
    if pred_col not in df.columns or bt_col not in df.columns:
        raise KeyError(
            f"Missing per-class columns ({pred_col!r}, {bt_col!r}) — "
            "call add_perclass_clinical_columns first."
        )

    swap = df.copy()
    swap["clinical_pred"] = swap[pred_col]
    swap["binary_target"] = swap[bt_col]
    fn = _legacy_metric_fn(metric_type, utils)
    return fn(swap, time_bins, subgroup_filter)


# ---------------------------------------------------------------------------
# Binary metrics restricted to one underlying unhealthy class
# ---------------------------------------------------------------------------


def compute_binary_by_underlying_class(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    metric_type: str,
    restrict_class: str,
    *,
    subgroup_filter: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
) -> pd.DataFrame:
    """Binary metric on the subset HEALTHY ∪ {restrict_class}.

    Tells us whether the binary classifier detects (e.g.) ACIDOSIS as
    well as it detects HIE — a clinically critical question that the
    pooled binary metric obscures.

    Args:
        df: Full predictions DataFrame (must have ``target``,
            ``binary_target``, ``clinical_pred`` set by the legacy CDR).
        time_bins: Bin edges in hours-before-delivery.
        metric_type: One of ``METRIC_TYPES``.
        restrict_class: ``"acidosis"`` or ``"hie"`` — the unhealthy class
            to keep alongside HEALTHY.
        subgroup_filter: Optional row-level boolean mask producer.

    Returns:
        DataFrame with the same shape as the legacy binary metrics.
    """
    if restrict_class not in {"acidosis", "hie"}:
        raise ValueError(
            f"restrict_class must be 'acidosis' or 'hie', got {restrict_class!r}"
        )
    target_id = {name: tid for name, tid, _ in CLASS_INFO}[restrict_class]
    keep = df[(df["target"] == 1) | (df["target"] == target_id)].copy()
    if keep.empty:
        return pd.DataFrame()
    utils = _legacy_utils()
    fn = _legacy_metric_fn(metric_type, utils)
    return fn(keep, time_bins, subgroup_filter)


# ---------------------------------------------------------------------------
# Per-class threshold search (secondary diagnostic)
# ---------------------------------------------------------------------------


def find_perclass_threshold_at_target_fpr(
    df_val: pd.DataFrame,
    class_name: str,
    *,
    target_fpr: float = 0.2,
    decision_time_hours: float = 1.0,
    max_iters: int = 25,
    fallback_tolerance_hours: float = 0.5,
) -> Tuple[float, Dict[str, Any]]:
    """OvR-threshold search analogous to ``find_threshold_for_committed_overall_fpr_at_1h``.

    Used as a **secondary** diagnostic. Primary 3-class reporting stays
    argmax-based. The legacy fn reads ``prob_class_1``, so we temporarily
    swap in the per-class probability column and per-class binary target
    before calling it.

    Args:
        df_val: Validation prediction rows (output of
            :func:`run_inference_per_position`).
        class_name: One of ``"healthy"``, ``"acidosis"``, ``"hie"``.
        target_fpr: Target false-positive rate at the decision time.
        decision_time_hours: Hours before delivery at which to evaluate.
        max_iters: Bisection iteration cap (legacy default).
        fallback_tolerance_hours: Decision-time fallback window
            (legacy default).

    Returns:
        Tuple ``(threshold, info_dict)``. ``info_dict`` carries the
        per-class FPR / sensitivity / specificity at the decision time.
    """
    utils = _legacy_utils()
    target_id = {name: tid for name, tid, _ in CLASS_INFO}[class_name]
    prob_col = {name: pc for name, _, pc in CLASS_INFO}[class_name]

    swap = df_val.copy()
    swap["prob_class_1"] = swap[prob_col]
    swap["binary_target"] = (swap["target"] == target_id).astype(int)
    return utils["find_threshold_for_committed_overall_fpr_at_1h"](
        swap,
        target_fpr=target_fpr,
        decision_time_hours=decision_time_hours,
        max_iters=max_iters,
        fallback_tolerance_hours=fallback_tolerance_hours,
    )


# ---------------------------------------------------------------------------
# Plot helpers — reuse legacy plotters where possible
# ---------------------------------------------------------------------------


def plot_perclass_panel(
    metrics_per_class: Mapping[str, pd.DataFrame],
    *,
    output_dir: Path,
    metric_type: str,
    title_suffix: str = "",
) -> None:
    """Render per-class metric panels using the legacy binary plotter.

    Each class gets its own subdirectory under ``output_dir`` (so the
    legacy file names ``sensitivity_vs_time.png`` etc. don't collide).

    Args:
        metrics_per_class: Mapping ``{class_name: metrics_df}``.
        output_dir: Output root for the panel set.
        metric_type: One of ``METRIC_TYPES`` — passed through to the
            legacy plotter for axis labelling consistency.
        title_suffix: Optional suffix appended to the legacy title.
    """
    utils = _legacy_utils()
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name, metrics_df in metrics_per_class.items():
        if metrics_df is None or metrics_df.empty:
            continue
        sub = output_dir / class_name
        sub.mkdir(parents=True, exist_ok=True)
        utils["plot_single_metric_type"](
            metrics_df=metrics_df,
            metric_type=metric_type,
            output_dir=sub,
            title_suffix=f" [{class_name}{title_suffix}]",
        )


def plot_subgroup_perclass(
    df: pd.DataFrame,
    *,
    time_bins: np.ndarray,
    metric_type: str,
    output_dir: Path,
) -> None:
    """For each (class × subgroup) compute metrics and reuse the legacy plotter.

    Subgroups come from
    :func:`evaluate_classifier.create_enhanced_subgroup_filters`.

    Args:
        df: DataFrame from :func:`add_perclass_clinical_columns`.
        time_bins: Bin edges in hours-before-delivery.
        metric_type: One of ``METRIC_TYPES``.
        output_dir: Output root. Created if missing.
    """
    utils = _legacy_utils()
    filters = utils["create_enhanced_subgroup_filters"]()
    output_dir.mkdir(parents=True, exist_ok=True)
    for subgroup_name, filter_fn in filters.items():
        for class_name, _, _ in CLASS_INFO:
            try:
                metrics_df = compute_perclass_time_binned_metrics(
                    df,
                    time_bins=time_bins,
                    metric_type=metric_type,
                    class_name=class_name,
                    subgroup_filter=filter_fn,
                )
            except Exception:  # noqa: BLE001 — defensive: bad subgroup yields empty
                continue
            if metrics_df is None or metrics_df.empty:
                continue
            sub = output_dir / class_name / subgroup_name
            sub.mkdir(parents=True, exist_ok=True)
            metrics_df.to_csv(sub / "metrics.csv", index=False)
            utils["plot_single_metric_type"](
                metrics_df=metrics_df,
                metric_type=metric_type,
                output_dir=sub,
                title_suffix=f" [{class_name} | {subgroup_name}]",
            )


def plot_binary_by_underlying_class(
    df: pd.DataFrame,
    *,
    time_bins: np.ndarray,
    metric_type: str,
    output_dir: Path,
) -> None:
    """Plot binary metrics restricted to HEALTHY ∪ {ACIDOSIS} and HEALTHY ∪ {HIE}."""
    utils = _legacy_utils()
    output_dir.mkdir(parents=True, exist_ok=True)
    for restrict_class in ("acidosis", "hie"):
        metrics_df = compute_binary_by_underlying_class(
            df, time_bins=time_bins, metric_type=metric_type, restrict_class=restrict_class
        )
        if metrics_df is None or metrics_df.empty:
            continue
        sub = output_dir / f"binary_only_{restrict_class}"
        sub.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(sub / "metrics.csv", index=False)
        utils["plot_single_metric_type"](
            metrics_df=metrics_df,
            metric_type=metric_type,
            output_dir=sub,
            title_suffix=f" [binary | only {restrict_class}]",
        )


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_3class_evaluation_for_metric_type(
    df: pd.DataFrame,
    *,
    time_bins: np.ndarray,
    metric_type: str,
    output_dir: Path,
    thresholds: Optional[Mapping[str, float]] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute and persist all per-class artefacts for one metric type.

    Args:
        df: Predictions DataFrame after the legacy CDR has been applied
            (so ``binary_target`` / ``clinical_pred`` exist in their
            binary-pooled form). Per-class columns are added internally.
        time_bins: Bin edges (legacy ``compute_time_bins``).
        metric_type: One of ``METRIC_TYPES``.
        output_dir: Per-fold ``three_metric_types/<metric_type>/`` root.
            ``per_class/`` and ``binary_by_underlying_class/`` are created
            beneath it.
        thresholds: Optional per-class OvR thresholds. ``None`` (default)
            uses argmax for clinical predictions.

    Returns:
        ``{class_name: metrics_df}`` plus the binary-by-underlying-class
        DataFrames keyed as ``binary_only_acidosis`` / ``binary_only_hie``.
    """
    out: Dict[str, pd.DataFrame] = {}
    df_pc = add_perclass_clinical_columns(df, thresholds=thresholds)

    pc_root = output_dir / "per_class"
    pc_root.mkdir(parents=True, exist_ok=True)
    metrics_per_class: Dict[str, pd.DataFrame] = {}
    for class_name, _, _ in CLASS_INFO:
        m = compute_perclass_time_binned_metrics(
            df_pc, time_bins=time_bins, metric_type=metric_type, class_name=class_name
        )
        m.to_csv(pc_root / f"{class_name}_metrics.csv", index=False)
        metrics_per_class[class_name] = m
        out[class_name] = m
    plot_perclass_panel(metrics_per_class, output_dir=pc_root, metric_type=metric_type)
    plot_subgroup_perclass(
        df_pc, time_bins=time_bins, metric_type=metric_type, output_dir=pc_root / "subgroups"
    )

    bin_root = output_dir / "binary_by_underlying_class"
    bin_root.mkdir(parents=True, exist_ok=True)
    plot_binary_by_underlying_class(
        df, time_bins=time_bins, metric_type=metric_type, output_dir=bin_root
    )
    for restrict_class in ("acidosis", "hie"):
        m_bin = compute_binary_by_underlying_class(
            df, time_bins=time_bins, metric_type=metric_type, restrict_class=restrict_class
        )
        out[f"binary_only_{restrict_class}"] = m_bin

    return out


def write_perclass_thresholds_json(
    df_val: pd.DataFrame,
    *,
    output_path: Path,
    target_fpr: float = 0.2,
    decision_time_hours: float = 1.0,
) -> Dict[str, Any]:
    """Run :func:`find_perclass_threshold_at_target_fpr` per class + save JSON.

    Args:
        df_val: Validation predictions DataFrame.
        output_path: Where to write ``perclass_thresholds.json``.
        target_fpr: Target FPR at the decision time.
        decision_time_hours: Decision time in hours before delivery.

    Returns:
        Dict carrying per-class thresholds and validation metrics.
    """
    perclass_info: Dict[str, Any] = {}
    for class_name, _, _ in CLASS_INFO:
        thr, info = find_perclass_threshold_at_target_fpr(
            df_val,
            class_name=class_name,
            target_fpr=target_fpr,
            decision_time_hours=decision_time_hours,
        )
        perclass_info[class_name] = {"threshold": float(thr), **info}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(perclass_info, indent=2, default=float))
    return perclass_info


__all__ = [
    "CLASS_INFO",
    "METRIC_TYPES",
    "add_perclass_clinical_columns",
    "compute_binary_by_underlying_class",
    "compute_perclass_time_binned_metrics",
    "find_perclass_threshold_at_target_fpr",
    "plot_binary_by_underlying_class",
    "plot_perclass_panel",
    "plot_subgroup_perclass",
    "run_3class_evaluation_for_metric_type",
    "write_perclass_thresholds_json",
]
