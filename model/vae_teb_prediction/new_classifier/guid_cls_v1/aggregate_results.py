"""Cross-fold aggregation for ``guid_cls_v1`` (PRD §12).

Reads each ``fold_{k}/evaluation/`` and produces:

* ``aggregated_plots/aggregated_roc_binary.png`` — mean ± std ROC across folds.
* ``aggregated_plots/aggregated_roc_3class_ovr/{healthy,acidosis,hie}.png``
  — per-class one-vs-rest aggregated ROCs.
* ``aggregated_plots/three_class_diagnostics/`` — mean confusion matrix +
  per-class probability histograms across folds.
* ``aggregated_results.json`` with mean/std for each headline metric.

The legacy ``generate_aggregated_plots`` from
``new_classifier.evaluate_classifier`` is invoked separately when the
metric-type DataFrames are available; here we cover only the parts the new
3-class pipeline adds on top of that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import auc as sklearn_auc

from model.vae_teb_prediction.new_classifier.guid_cls_v1.evaluate_guid_classifier import (
    compute_3class_roc_ovr,
    compute_confusion_matrix_3class,
)


def _list_fold_dirs(run_dir: Path) -> List[Path]:
    """Return the sorted ``fold_*`` subdirectories inside ``run_dir``."""
    folds = []
    for child in sorted(run_dir.glob("fold_*")):
        if not child.is_dir():
            continue
        try:
            int(child.name.split("_")[-1])
        except ValueError:
            continue
        folds.append(child)
    return folds


def _resolve_test_predictions_csv(fold_dir: Path) -> Optional[Path]:
    """Return the per-fold raw test-predictions CSV path under either layout.

    Prefers the new layout ``evaluation/predictions/test_raw.csv``; falls
    back to the legacy ``evaluation/test_predictions_raw.csv`` so old runs
    can still be aggregated for binary/3-class ROC bookkeeping.
    """
    new_path = fold_dir / "evaluation" / "predictions" / "test_raw.csv"
    if new_path.exists():
        return new_path
    legacy = fold_dir / "evaluation" / "test_predictions_raw.csv"
    if legacy.exists():
        return legacy
    return None


def _resolve_binary_roc_csv(fold_dir: Path) -> Optional[Path]:
    """Return the binary ROC CSV path under either layout."""
    new_path = fold_dir / "evaluation" / "binary_head" / "roc.csv"
    if new_path.exists():
        return new_path
    legacy = fold_dir / "evaluation" / "roc_binary_data.csv"
    if legacy.exists():
        return legacy
    return None


def _resolve_thresholds_json(fold_dir: Path) -> Optional[Path]:
    """Return the thresholds JSON path under either layout."""
    new_path = fold_dir / "evaluation" / "thresholds.json"
    if new_path.exists():
        return new_path
    legacy = fold_dir / "evaluation" / "threshold_info.json"
    if legacy.exists():
        return legacy
    return None


def _has_new_layout(fold_dir: Path) -> bool:
    """True iff this fold was evaluated with the new head-explicit layout."""
    return (fold_dir / "evaluation" / "binary_head").is_dir()


def _load_fold_test_csv(fold_dir: Path) -> Optional[pd.DataFrame]:
    csv = _resolve_test_predictions_csv(fold_dir)
    if csv is None:
        return None
    try:
        return pd.read_csv(csv)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"failed to read {csv}: {exc}")
        return None


def _interpolate_roc(
    fpr: Sequence[float], tpr: Sequence[float], grid: np.ndarray
) -> np.ndarray:
    """Interpolate a per-fold ROC onto a common FPR grid for averaging."""
    fpr_arr = np.asarray(fpr, dtype=float)
    tpr_arr = np.asarray(tpr, dtype=float)
    if fpr_arr.size == 0:
        return np.full_like(grid, np.nan, dtype=float)
    # Ensure monotonically non-decreasing fpr.
    order = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[order]
    tpr_sorted = tpr_arr[order]
    return np.interp(grid, fpr_sorted, tpr_sorted)


def _plot_aggregated_binary_roc(
    per_fold_rocs: List[Tuple[int, Dict[str, Any]]], output_path: Path
) -> Optional[float]:
    """Mean ± std ROC across folds (binary head)."""
    import matplotlib.pyplot as plt  # noqa: WPS433

    if not per_fold_rocs:
        return None
    grid = np.linspace(0.0, 1.0, 101)
    tprs = []
    aucs = []
    for fid, roc in per_fold_rocs:
        if not roc:
            continue
        tprs.append(_interpolate_roc(roc["fpr"], roc["tpr"], grid))
        aucs.append(float(roc.get("auc", float("nan"))))
    if not tprs:
        return None
    arr = np.vstack(tprs)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    fig, ax = plt.subplots(figsize=(5, 5))
    for fid, roc in per_fold_rocs:
        if not roc:
            continue
        ax.plot(roc["fpr"], roc["tpr"], alpha=0.25, label=f"fold {fid}")
    ax.plot(grid, mean, color="C0", lw=2.0, label="mean")
    ax.fill_between(grid, mean - std, mean + std, color="C0", alpha=0.15, label="±1 std")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    mean_auc = float(np.nanmean(aucs)) if aucs else float("nan")
    std_auc = float(np.nanstd(aucs)) if aucs else float("nan")
    ax.set_title(f"Binary GUID ROC — mean AUC={mean_auc:.3f} ± {std_auc:.3f}")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return mean_auc


def _plot_aggregated_3class_roc(
    per_fold_3class: List[Tuple[int, Dict[str, Dict[str, Any]]]],
    output_dir: Path,
) -> Dict[str, Optional[float]]:
    """Per-class one-vs-rest aggregated ROCs."""
    import matplotlib.pyplot as plt  # noqa: WPS433

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Optional[float]] = {}
    grid = np.linspace(0.0, 1.0, 101)
    for class_name in ("healthy", "acidosis", "hie"):
        tprs = []
        aucs = []
        for fid, ovr in per_fold_3class:
            if class_name not in ovr:
                continue
            data = ovr[class_name]
            if not data["fpr"]:
                continue
            tprs.append(_interpolate_roc(data["fpr"], data["tpr"], grid))
            auc_val = data.get("auc")
            if auc_val == auc_val:
                aucs.append(float(auc_val))
        if not tprs:
            summary[class_name] = None
            continue
        arr = np.vstack(tprs)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        fig, ax = plt.subplots(figsize=(5, 5))
        for fid, ovr in per_fold_3class:
            data = ovr.get(class_name) if ovr else None
            if not data or not data["fpr"]:
                continue
            ax.plot(data["fpr"], data["tpr"], alpha=0.25, label=f"fold {fid}")
        ax.plot(grid, mean, color="C0", lw=2.0, label="mean")
        ax.fill_between(grid, mean - std, mean + std, color="C0", alpha=0.15, label="±1 std")
        ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        mean_auc = float(np.nanmean(aucs)) if aucs else float("nan")
        std_auc = float(np.nanstd(aucs)) if aucs else float("nan")
        ax.set_title(f"{class_name.upper()} OvR — mean AUC={mean_auc:.3f} ± {std_auc:.3f}")
        ax.legend(fontsize=7, loc="lower right", ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{class_name}.png", dpi=150)
        plt.close(fig)
        summary[class_name] = mean_auc
    return summary


def _plot_mean_confusion_matrix(
    per_fold_cms: List[np.ndarray], output_path: Path
) -> np.ndarray:
    """Plot the row-normalised mean 3×3 confusion across folds."""
    import matplotlib.pyplot as plt  # noqa: WPS433

    arr = np.stack([cm / cm.sum(axis=1, keepdims=True).clip(min=1.0) for cm in per_fold_cms])
    mean_cm = arr.mean(axis=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mean_cm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    class_names = ["healthy", "acidosis", "hie"]
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
                f"{mean_cm[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if mean_cm[i, j] < 0.5 else "white",
                fontsize=10,
            )
    ax.set_title("Mean 3-class confusion across folds")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return mean_cm


PER_CLASS_NAMES: Tuple[str, ...] = ("healthy", "acidosis", "hie")
METRIC_TYPES: Tuple[str, ...] = ("instantaneous", "committed_cumulative", "committed_overall")


def _interp_metric_curve(df: pd.DataFrame, grid: np.ndarray, key: str) -> np.ndarray:
    """Interpolate ``df[key]`` onto ``grid`` indexed by ``bin_center``."""
    if df is None or df.empty or "bin_center" not in df.columns or key not in df.columns:
        return np.full_like(grid, np.nan, dtype=float)
    centers = df["bin_center"].astype(float).to_numpy()
    values = df[key].astype(float).to_numpy()
    valid = np.isfinite(centers) & np.isfinite(values)
    if valid.sum() < 2:
        return np.full_like(grid, np.nan, dtype=float)
    order = np.argsort(centers[valid])
    return np.interp(grid, centers[valid][order], values[valid][order])


def _plot_mean_curve_with_band(
    ax,
    grid: np.ndarray,
    per_fold: List[np.ndarray],
    *,
    label: str,
    color: str,
    decision_time_hours: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """Plot mean ± std (and per-fold thin lines) on ``ax``. Returns (mean_at_x0, std_at_x0).

    When ``decision_time_hours`` is provided, draws a dashed vertical
    reference line at that time-vs-delivery so the operating point is
    visible on every cross-fold curve.
    """
    if not per_fold:
        return None
    arr = np.vstack(per_fold)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    for row in per_fold:
        ax.plot(grid, row, alpha=0.20, color=color, lw=0.7)
    ax.plot(grid, mean, color=color, lw=2.0, label=label)
    ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.15)
    if decision_time_hours is not None:
        try:
            from model.vae_teb_prediction.new_classifier.guid_cls_v1.clinical_metrics_utils import (  # noqa: WPS433
                annotate_decision_time,
            )

            annotate_decision_time(ax, decision_time_hours=float(decision_time_hours))
        except Exception:  # pragma: no cover - defensive
            pass
    return float(np.nanmean(mean)), float(np.nanmean(std))


def _aggregate_perclass_metric_curves(
    fold_dirs: List[Path],
    agg_root: Path,
    *,
    decision_time_hours: Optional[float] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Mean ± std per-class time-binned metric curves across folds.

    Reads each fold's long-format
    ``multiclass_head/per_class_vs_time/<metric_type>.csv`` and emits
    cross-fold curves under
    ``aggregated_plots/multiclass_head/per_class_vs_time/<metric_type>_<class>.{csv,png}``.
    """
    import matplotlib.pyplot as plt  # noqa: WPS433

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for metric_type in METRIC_TYPES:
        per_class_summary: Dict[str, Dict[str, float]] = {}
        out_dir = agg_root / "multiclass_head" / "per_class_vs_time"
        out_dir.mkdir(parents=True, exist_ok=True)

        # First pass: read each fold's long-format CSV and split by class.
        per_fold_by_class: Dict[str, List[pd.DataFrame]] = {
            c: [] for c in PER_CLASS_NAMES
        }
        for fd in fold_dirs:
            csv = (
                fd
                / "evaluation"
                / "multiclass_head"
                / "per_class_vs_time"
                / f"{metric_type}.csv"
            )
            if not csv.exists():
                continue
            try:
                df = pd.read_csv(csv)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"could not read {csv}: {exc}")
                continue
            if df.empty or "bin_center" not in df.columns or "class" not in df.columns:
                continue
            for class_name in PER_CLASS_NAMES:
                sub = df[df["class"].astype(str) == class_name]
                if sub.empty:
                    continue
                per_fold_by_class[class_name].append(sub.reset_index(drop=True))

        for class_name in PER_CLASS_NAMES:
            collected = per_fold_by_class[class_name]
            if not collected:
                continue
            grid_min = min(
                float(np.nanmin(d["bin_center"].astype(float).to_numpy()))
                for d in collected
                if d["bin_center"].astype(float).to_numpy().size
            )
            grid_max = max(
                float(np.nanmax(d["bin_center"].astype(float).to_numpy()))
                for d in collected
                if d["bin_center"].astype(float).to_numpy().size
            )
            if not (np.isfinite(grid_min) and np.isfinite(grid_max)) or grid_min >= grid_max:
                continue
            grid = np.linspace(grid_min, grid_max, 64)
            per_fold_sens = [
                _interp_metric_curve(d, grid, "sensitivity") for d in collected
            ]
            per_fold_fpr = [
                _interp_metric_curve(d, grid, "fpr") for d in collected
            ]

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            sens_stat = _plot_mean_curve_with_band(
                axes[0], grid, per_fold_sens, label="mean", color="C0",
                decision_time_hours=decision_time_hours,
            )
            axes[0].set_title(f"{class_name} sensitivity vs time ({metric_type})")
            axes[0].set_xlabel("hours before delivery")
            axes[0].set_ylabel("sensitivity")
            axes[0].invert_xaxis()
            axes[0].set_ylim(-0.05, 1.05)

            fpr_stat = _plot_mean_curve_with_band(
                axes[1], grid, per_fold_fpr, label="mean", color="C3",
                decision_time_hours=decision_time_hours,
            )
            axes[1].set_title(f"{class_name} FPR vs time ({metric_type})")
            axes[1].set_xlabel("hours before delivery")
            axes[1].set_ylabel("FPR")
            axes[1].invert_xaxis()
            axes[1].set_ylim(-0.05, 1.05)

            fig.tight_layout()
            fig.savefig(out_dir / f"{metric_type}_{class_name}.png", dpi=150)
            plt.close(fig)

            sens_arr = np.vstack(per_fold_sens)
            fpr_arr = np.vstack(per_fold_fpr)
            pd.DataFrame(
                {
                    "bin_center": grid,
                    "sensitivity_mean": np.nanmean(sens_arr, axis=0),
                    "sensitivity_std": np.nanstd(sens_arr, axis=0),
                    "fpr_mean": np.nanmean(fpr_arr, axis=0),
                    "fpr_std": np.nanstd(fpr_arr, axis=0),
                    "n_folds": np.sum(np.isfinite(sens_arr), axis=0),
                }
            ).to_csv(out_dir / f"{metric_type}_{class_name}.csv", index=False)

            per_class_summary[class_name] = {
                "n_folds": int(len(per_fold_sens)),
                "sensitivity_mean_avg_over_bins": (sens_stat or (float("nan"), float("nan")))[0],
                "sensitivity_std_avg_over_bins": (sens_stat or (float("nan"), float("nan")))[1],
                "fpr_mean_avg_over_bins": (fpr_stat or (float("nan"), float("nan")))[0],
                "fpr_std_avg_over_bins": (fpr_stat or (float("nan"), float("nan")))[1],
            }
        if per_class_summary:
            summary[metric_type] = per_class_summary
    return summary


def _aggregate_binary_by_underlying_class_curves(
    fold_dirs: List[Path],
    agg_root: Path,
    *,
    decision_time_hours: Optional[float] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate the binary-by-underlying-class long-format CSVs across folds.

    Reads each fold's
    ``binary_head/by_underlying_class_vs_time/<metric_type>.csv`` and
    emits cross-fold curves under
    ``aggregated_plots/binary_head/by_underlying_class_vs_time/<metric_type>_<restrict>.{csv,png}``.
    """
    import matplotlib.pyplot as plt  # noqa: WPS433

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for metric_type in METRIC_TYPES:
        sub_summary: Dict[str, Dict[str, float]] = {}
        out_dir = agg_root / "binary_head" / "by_underlying_class_vs_time"
        out_dir.mkdir(parents=True, exist_ok=True)

        per_fold_by_restrict: Dict[str, List[pd.DataFrame]] = {
            "acidosis": [],
            "hie": [],
        }
        for fd in fold_dirs:
            csv = (
                fd
                / "evaluation"
                / "binary_head"
                / "by_underlying_class_vs_time"
                / f"{metric_type}.csv"
            )
            if not csv.exists():
                continue
            try:
                df = pd.read_csv(csv)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"could not read {csv}: {exc}")
                continue
            if df.empty or "bin_center" not in df.columns or "restrict_class" not in df.columns:
                continue
            for restrict in per_fold_by_restrict:
                sub = df[df["restrict_class"].astype(str) == restrict]
                if sub.empty:
                    continue
                per_fold_by_restrict[restrict].append(sub.reset_index(drop=True))

        for restrict, collected in per_fold_by_restrict.items():
            if not collected:
                continue
            grid_min = min(
                float(np.nanmin(d["bin_center"].astype(float).to_numpy()))
                for d in collected
                if d["bin_center"].astype(float).to_numpy().size
            )
            grid_max = max(
                float(np.nanmax(d["bin_center"].astype(float).to_numpy()))
                for d in collected
                if d["bin_center"].astype(float).to_numpy().size
            )
            if not (np.isfinite(grid_min) and np.isfinite(grid_max)) or grid_min >= grid_max:
                continue
            grid = np.linspace(grid_min, grid_max, 64)
            sens = [_interp_metric_curve(d, grid, "sensitivity") for d in collected]
            fpr = [_interp_metric_curve(d, grid, "fpr") for d in collected]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            sens_stat = _plot_mean_curve_with_band(
                axes[0], grid, sens, label="mean", color="C0",
                decision_time_hours=decision_time_hours,
            )
            axes[0].set_title(f"binary | only {restrict} — sensitivity ({metric_type})")
            axes[0].set_xlabel("hours before delivery")
            axes[0].set_ylabel("sensitivity")
            axes[0].invert_xaxis()
            axes[0].set_ylim(-0.05, 1.05)
            fpr_stat = _plot_mean_curve_with_band(
                axes[1], grid, fpr, label="mean", color="C3",
                decision_time_hours=decision_time_hours,
            )
            axes[1].set_title(f"binary | only {restrict} — FPR ({metric_type})")
            axes[1].set_xlabel("hours before delivery")
            axes[1].set_ylabel("FPR")
            axes[1].invert_xaxis()
            axes[1].set_ylim(-0.05, 1.05)
            fig.tight_layout()
            fig.savefig(out_dir / f"{metric_type}_{restrict}.png", dpi=150)
            plt.close(fig)

            sens_arr = np.vstack(sens)
            fpr_arr = np.vstack(fpr)
            pd.DataFrame(
                {
                    "bin_center": grid,
                    "sensitivity_mean": np.nanmean(sens_arr, axis=0),
                    "sensitivity_std": np.nanstd(sens_arr, axis=0),
                    "fpr_mean": np.nanmean(fpr_arr, axis=0),
                    "fpr_std": np.nanstd(fpr_arr, axis=0),
                    "n_folds": np.sum(np.isfinite(sens_arr), axis=0),
                }
            ).to_csv(out_dir / f"{metric_type}_{restrict}.csv", index=False)

            sub_summary[restrict] = {
                "n_folds": int(len(sens)),
                "sensitivity_mean_avg_over_bins": (sens_stat or (float("nan"), float("nan")))[0],
                "fpr_mean_avg_over_bins": (fpr_stat or (float("nan"), float("nan")))[0],
            }
        if sub_summary:
            summary[metric_type] = sub_summary
    return summary


def _aggregate_perclass_subgroup_metric_curves(
    fold_dirs: List[Path],
    agg_root: Path,
    *,
    decision_time_hours: Optional[float] = None,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Mean ± std per-class × subgroup time-binned curves across folds.

    Reads each fold's long-format
    ``multiclass_head/per_class_subgroups_vs_time/<metric_type>.csv``
    (cols: ``bin_center, class, subgroup, sensitivity, specificity, fpr,
    n_pos, n_neg, n``) and emits cross-fold curves under
    ``aggregated_plots/multiclass_head/per_class_subgroups_vs_time/<metric_type>/<class>_<subgroup>.{csv,png}``.

    Returns:
        Nested dict keyed by ``[metric_type][class][subgroup]`` carrying
        ``n_folds``, ``sensitivity_mean_avg_over_bins``,
        ``fpr_mean_avg_over_bins``.
    """
    import matplotlib.pyplot as plt  # noqa: WPS433

    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for metric_type in METRIC_TYPES:
        per_class_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        out_root = (
            agg_root
            / "multiclass_head"
            / "per_class_subgroups_vs_time"
            / metric_type
        )

        # First pass: read each fold's long-format CSV and group by
        # (class, subgroup).
        per_fold_by_combo: Dict[Tuple[str, str], List[pd.DataFrame]] = {}
        for fd in fold_dirs:
            csv = (
                fd
                / "evaluation"
                / "multiclass_head"
                / "per_class_subgroups_vs_time"
                / f"{metric_type}.csv"
            )
            if not csv.exists():
                continue
            try:
                df = pd.read_csv(csv)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"could not read {csv}: {exc}")
                continue
            required = {"bin_center", "class", "subgroup"}
            if df.empty or not required.issubset(set(df.columns)):
                continue
            for (cls, sg), grp in df.groupby(["class", "subgroup"], dropna=False):
                if grp.empty:
                    continue
                key = (str(cls), str(sg))
                per_fold_by_combo.setdefault(key, []).append(
                    grp.reset_index(drop=True)
                )

        if not per_fold_by_combo:
            continue
        out_root.mkdir(parents=True, exist_ok=True)
        for (class_name, subgroup_name), collected in per_fold_by_combo.items():
            if class_name not in PER_CLASS_NAMES:
                continue
            grid_min = min(
                float(np.nanmin(d["bin_center"].astype(float).to_numpy()))
                for d in collected
                if d["bin_center"].astype(float).to_numpy().size
            )
            grid_max = max(
                float(np.nanmax(d["bin_center"].astype(float).to_numpy()))
                for d in collected
                if d["bin_center"].astype(float).to_numpy().size
            )
            if not (np.isfinite(grid_min) and np.isfinite(grid_max)) or grid_min >= grid_max:
                continue
            grid = np.linspace(grid_min, grid_max, 64)
            per_fold_sens = [
                _interp_metric_curve(d, grid, "sensitivity") for d in collected
            ]
            per_fold_fpr = [
                _interp_metric_curve(d, grid, "fpr") for d in collected
            ]

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            sens_stat = _plot_mean_curve_with_band(
                axes[0], grid, per_fold_sens, label="mean", color="C0",
                decision_time_hours=decision_time_hours,
            )
            axes[0].set_title(
                f"{class_name} | {subgroup_name} sensitivity ({metric_type})"
            )
            axes[0].set_xlabel("hours before delivery")
            axes[0].set_ylabel("sensitivity")
            axes[0].invert_xaxis()
            axes[0].set_ylim(-0.05, 1.05)
            fpr_stat = _plot_mean_curve_with_band(
                axes[1], grid, per_fold_fpr, label="mean", color="C3",
                decision_time_hours=decision_time_hours,
            )
            axes[1].set_title(
                f"{class_name} | {subgroup_name} FPR ({metric_type})"
            )
            axes[1].set_xlabel("hours before delivery")
            axes[1].set_ylabel("FPR")
            axes[1].invert_xaxis()
            axes[1].set_ylim(-0.05, 1.05)
            fig.tight_layout()
            stem = f"{class_name}_{subgroup_name}"
            fig.savefig(out_root / f"{stem}.png", dpi=150)
            plt.close(fig)

            sens_arr = np.vstack(per_fold_sens)
            fpr_arr = np.vstack(per_fold_fpr)
            pd.DataFrame(
                {
                    "bin_center": grid,
                    "sensitivity_mean": np.nanmean(sens_arr, axis=0),
                    "sensitivity_std": np.nanstd(sens_arr, axis=0),
                    "fpr_mean": np.nanmean(fpr_arr, axis=0),
                    "fpr_std": np.nanstd(fpr_arr, axis=0),
                    "n_folds": np.sum(np.isfinite(sens_arr), axis=0),
                }
            ).to_csv(out_root / f"{stem}.csv", index=False)

            entry = {
                "n_folds": int(len(per_fold_sens)),
                "sensitivity_mean_avg_over_bins": (sens_stat or (float("nan"), float("nan")))[0],
                "sensitivity_std_avg_over_bins": (sens_stat or (float("nan"), float("nan")))[1],
                "fpr_mean_avg_over_bins": (fpr_stat or (float("nan"), float("nan")))[0],
                "fpr_std_avg_over_bins": (fpr_stat or (float("nan"), float("nan")))[1],
            }
            per_class_summary.setdefault(class_name, {})[subgroup_name] = entry
        if per_class_summary:
            summary[metric_type] = per_class_summary
    return summary


def _aggregate_perclass_auroc_curves(
    fold_dirs: List[Path],
    agg_root: Path,
    *,
    decision_time_hours: Optional[float] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Mean ± std per-class one-vs-rest AUROC vs time across folds.

    Reads each fold's long-format
    ``multiclass_head/auroc_vs_time/<metric_type>.csv`` (cols:
    ``bin_center, class, auroc, n_pos, n_neg, n``) and emits cross-fold
    curves under
    ``aggregated_plots/multiclass_head/auroc_vs_time/<metric_type>.{csv,png}``.

    The per-class panel uses the standard 3-class palette. A horizontal
    chance line at AUROC=0.5 is drawn for orientation.
    """
    import matplotlib.pyplot as plt  # noqa: WPS433

    palette = {"healthy": "#27ae60", "acidosis": "#f39c12", "hie": "#c0392b"}
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for metric_type in METRIC_TYPES:
        per_class_summary: Dict[str, Dict[str, float]] = {}
        out_dir = agg_root / "multiclass_head" / "auroc_vs_time"

        per_fold_by_class: Dict[str, List[pd.DataFrame]] = {
            c: [] for c in PER_CLASS_NAMES
        }
        for fd in fold_dirs:
            csv = (
                fd
                / "evaluation"
                / "multiclass_head"
                / "auroc_vs_time"
                / f"{metric_type}.csv"
            )
            if not csv.exists():
                continue
            try:
                df = pd.read_csv(csv)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"could not read {csv}: {exc}")
                continue
            if df.empty or "bin_center" not in df.columns or "class" not in df.columns:
                continue
            for class_name in PER_CLASS_NAMES:
                sub = df[df["class"].astype(str) == class_name]
                if sub.empty:
                    continue
                per_fold_by_class[class_name].append(sub.reset_index(drop=True))

        if not any(per_fold_by_class.values()):
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        # Combined 3-class panel.
        fig, ax = plt.subplots(figsize=(8, 5))
        # Determine global grid from union of bins across classes.
        all_centers: List[float] = []
        for collected in per_fold_by_class.values():
            for d in collected:
                centers = d["bin_center"].astype(float).to_numpy()
                if centers.size:
                    all_centers.extend(centers.tolist())
        if not all_centers:
            plt.close(fig)
            continue
        grid = np.linspace(
            float(np.nanmin(all_centers)), float(np.nanmax(all_centers)), 64
        )
        long_rows: List[Dict[str, Any]] = []
        for class_name in PER_CLASS_NAMES:
            collected = per_fold_by_class[class_name]
            if not collected:
                continue
            per_fold = [
                _interp_metric_curve(d, grid, "auroc") for d in collected
            ]
            stat = _plot_mean_curve_with_band(
                ax, grid, per_fold,
                label=class_name, color=palette.get(class_name, "C0"),
                decision_time_hours=decision_time_hours,
            )
            arr = np.vstack(per_fold)
            for i, c in enumerate(grid):
                long_rows.append(
                    {
                        "bin_center": float(c),
                        "class": class_name,
                        "auroc_mean": float(np.nanmean(arr[:, i])),
                        "auroc_std": float(np.nanstd(arr[:, i])),
                        "n_folds": int(np.sum(np.isfinite(arr[:, i]))),
                    }
                )
            per_class_summary[class_name] = {
                "n_folds": int(len(per_fold)),
                "auroc_mean_avg_over_bins": (stat or (float("nan"), float("nan")))[0],
                "auroc_std_avg_over_bins": (stat or (float("nan"), float("nan")))[1],
            }
        ax.axhline(0.5, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("hours before delivery")
        ax.set_ylabel("AUROC (one-vs-rest)")
        ax.set_title(f"per-class AUROC vs time ({metric_type})")
        ax.invert_xaxis()
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=9, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric_type}.png", dpi=150)
        plt.close(fig)

        if long_rows:
            pd.DataFrame(long_rows).to_csv(
                out_dir / f"{metric_type}.csv", index=False
            )
        if per_class_summary:
            summary[metric_type] = per_class_summary
    return summary


def _aggregate_legacy_metric_plots(run_dir: Path, fold_dirs: List[Path]) -> None:
    """Bridge into ``clinical_metrics_utils.generate_aggregated_plots``.

    The legacy aggregator expects each fold's
    ``fold_results.json`` to carry a ``three_metric_results_full``
    block which the current per-fold trainer **does not** write
    (the new aggregator covers the same ground via
    :func:`_aggregate_perclass_metric_curves` and
    :func:`_aggregate_perclass_subgroup_metric_curves`). Calling this
    is therefore a no-op on every current run — we keep the helper as
    a forward-compatibility hook so a future producer can opt in by
    populating the expected key. The "no data found" warning that the
    legacy aggregator emits internally has been demoted here to a
    single info-level log so it stops looking like an error in the
    cross-fold output.
    """
    fold_results: List[Dict[str, Any]] = []
    for fd in fold_dirs:
        rj = fd / "fold_results.json"
        if not rj.exists():
            continue
        try:
            payload = json.loads(rj.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"could not parse {rj}: {exc}")
            continue
        if "three_metric_results_full" not in payload:
            continue  # legacy aggregator has nothing to do for this fold
        fold_results.append(payload)
    if not fold_results:
        logger.info(
            "_aggregate_legacy_metric_plots: no fold has the legacy "
            "``three_metric_results_full`` block — skipping. The new "
            "per-class / subgroup aggregators have already produced the "
            "current-pipeline equivalents."
        )
        return

    try:
        from model.vae_teb_prediction.new_classifier.guid_cls_v1.clinical_metrics_utils import (  # noqa: WPS433
            generate_aggregated_plots,
        )
    except Exception as exc:
        logger.warning(f"generate_aggregated_plots unavailable: {exc}")
        return
    try:
        generate_aggregated_plots(
            fold_results,
            run_dir,
            n_folds=len(fold_dirs),
            data_source="test",
        )
    except Exception as exc:  # pragma: no cover - external code
        logger.warning(f"generate_aggregated_plots failed: {exc}")


def aggregate_results(
    *,
    run_dir: Path,
    fold_ids: Optional[Sequence[int]] = None,
    decision_time_hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Aggregate fold artefacts and write summary plots / JSON.

    Args:
        run_dir: Run directory holding ``fold_*/`` subdirectories.
        fold_ids: Optional subset; defaults to every ``fold_*`` present.
        decision_time_hours: Hours before delivery at which the per-fold
            operating point was selected. When provided, every cross-fold
            time-axis plot draws a dashed vertical reference line via
            :func:`clinical_metrics_utils.annotate_decision_time`. When
            ``None``, the value is auto-detected from the first fold's
            ``thresholds.json`` (``decision_time_hours`` field, if
            present) or by reading ``evaluation.decision_time_hours``
            from the run-level ``config_guid_cls_v1.yaml``.

    Returns:
        Aggregated summary dict (also written to
        ``aggregated_results.json``).
    """
    fold_dirs = _list_fold_dirs(run_dir)
    if fold_ids is not None:
        wanted = {int(f) for f in fold_ids}
        fold_dirs = [fd for fd in fold_dirs if int(fd.name.split("_")[-1]) in wanted]
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* dirs found under {run_dir}")

    # Layout detection: warn loudly when a fold still uses the old
    # ``three_metric_types/`` tree so the operator knows that fold needs
    # to be re-evaluated against the new layout before its long-format
    # CSVs become available. Continue regardless; the legacy ROC / 3-class
    # pieces still aggregate from the (preserved) raw CSVs.
    new_layout = [fd for fd in fold_dirs if _has_new_layout(fd)]
    if not new_layout:
        logger.warning(
            f"aggregate_results: none of the {len(fold_dirs)} folds under "
            f"{run_dir} expose the new ``binary_head/`` + ``multiclass_head/`` "
            "layout. Per-class / subgroup / AUROC cross-fold aggregations "
            "will be empty. Re-run evaluate_kfold to refresh the per-fold "
            "tree under the current layout."
        )
    elif len(new_layout) < len(fold_dirs):
        legacy = [
            fd.name for fd in fold_dirs if not _has_new_layout(fd)
        ]
        logger.warning(
            f"aggregate_results: {len(legacy)}/{len(fold_dirs)} folds use "
            f"the legacy ``three_metric_types/`` layout: {legacy}. Their "
            "per-class / subgroup / AUROC contributions will be skipped. "
            "Re-run evaluate_single_fold on those folds to refresh."
        )

    # Resolve ``decision_time_hours`` for the vertical-reference annotation
    # on every vs-time plot (best-effort; falls back to None when neither
    # source is available).
    if decision_time_hours is None:
        for fd in fold_dirs:
            thr_path = _resolve_thresholds_json(fd)
            if thr_path is None:
                continue
            try:
                thr_payload = json.loads(thr_path.read_text(encoding="utf-8"))
            except Exception:  # pragma: no cover
                continue
            cand = thr_payload.get("decision_time_hours")
            if cand is not None:
                try:
                    decision_time_hours = float(cand)
                    break
                except (TypeError, ValueError):
                    continue
    if decision_time_hours is None:
        cfg_path = run_dir / "config_guid_cls_v1.yaml"
        if cfg_path.exists():
            try:
                import yaml  # noqa: WPS433

                cfg_payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                cand = (
                    (cfg_payload or {}).get("evaluation", {}) or {}
                ).get("decision_time_hours")
                if cand is not None:
                    decision_time_hours = float(cand)
            except Exception:  # pragma: no cover
                pass

    agg_root = run_dir / "aggregated_plots"
    agg_root.mkdir(parents=True, exist_ok=True)
    # New head-explicit cross-fold layout (parallel to per-fold tree).
    binary_head_agg = agg_root / "binary_head"
    multiclass_head_agg = agg_root / "multiclass_head"
    diag_root = multiclass_head_agg / "diagnostics"
    binary_head_agg.mkdir(parents=True, exist_ok=True)
    multiclass_head_agg.mkdir(parents=True, exist_ok=True)
    diag_root.mkdir(parents=True, exist_ok=True)

    per_fold_binary_roc: List[Tuple[int, Dict[str, Any]]] = []
    per_fold_3class_roc: List[Tuple[int, Dict[str, Dict[str, Any]]]] = []
    per_fold_cms: List[np.ndarray] = []
    per_fold_thresholds: List[Tuple[int, Dict[str, Any]]] = []

    for fd in fold_dirs:
        fid = int(fd.name.split("_")[-1])

        # Binary ROC: stored as ``binary_head/roc.csv`` (new layout) or
        # ``roc_binary_data.csv`` (legacy). Skip empty / malformed files
        # so a partially-failed fold doesn't poison the cross-fold summary.
        roc_csv = _resolve_binary_roc_csv(fd)
        if roc_csv is not None:
            try:
                roc_df = pd.read_csv(roc_csv)
                if roc_df.empty or "fpr" not in roc_df.columns or "tpr" not in roc_df.columns:
                    logger.warning(
                        f"fold {fid}: {roc_csv.name} is empty or missing "
                        "fpr/tpr columns; skipping."
                    )
                else:
                    fpr = roc_df["fpr"].astype(float).tolist()
                    tpr = roc_df["tpr"].astype(float).tolist()
                    if len(fpr) < 2:
                        logger.warning(
                            f"fold {fid}: ROC has fewer than 2 points; skipping."
                        )
                    else:
                        roc_dict = {
                            "fpr": fpr,
                            "tpr": tpr,
                            "auc": float(sklearn_auc(fpr, tpr)),
                        }
                        per_fold_binary_roc.append((fid, roc_dict))
            except Exception as exc:  # pragma: no cover
                logger.warning(f"fold {fid}: failed to read binary ROC CSV: {exc}")

        # 3-class OvR ROC: recomputed from the test predictions CSV.
        test_df = _load_fold_test_csv(fd)
        if test_df is not None:
            try:
                ovr = compute_3class_roc_ovr(test_df)
                per_fold_3class_roc.append((fid, ovr))
                cm = compute_confusion_matrix_3class(test_df)
                per_fold_cms.append(cm)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"fold {fid}: failed 3-class aggregation: {exc}")

        # Threshold info (new ``thresholds.json`` or legacy
        # ``threshold_info.json``).
        thr_json = _resolve_thresholds_json(fd)
        if thr_json is not None:
            try:
                per_fold_thresholds.append(
                    (fid, json.loads(thr_json.read_text(encoding="utf-8")))
                )
            except Exception as exc:
                logger.warning(f"fold {fid}: failed to read {thr_json.name}: {exc}")

    binary_auc_mean = _plot_aggregated_binary_roc(
        per_fold_binary_roc, binary_head_agg / "roc.png"
    )
    three_class_auc_means = _plot_aggregated_3class_roc(
        per_fold_3class_roc, multiclass_head_agg / "roc_ovr"
    )
    mean_cm: Optional[np.ndarray] = None
    if per_fold_cms:
        mean_cm = _plot_mean_confusion_matrix(
            per_fold_cms, diag_root / "mean_confusion_matrix.png"
        )

    # Legacy three-metric-type aggregator (no-op on current pipeline).
    _aggregate_legacy_metric_plots(run_dir, fold_dirs)

    # New per-class + binary-by-underlying-class + AUROC aggregations
    # (driven by long-format CSVs; emit nothing when no fold uses the new
    # layout).
    perclass_summary = _aggregate_perclass_metric_curves(
        fold_dirs, agg_root, decision_time_hours=decision_time_hours
    )
    binary_by_class_summary = _aggregate_binary_by_underlying_class_curves(
        fold_dirs, agg_root, decision_time_hours=decision_time_hours
    )
    perclass_subgroup_summary = _aggregate_perclass_subgroup_metric_curves(
        fold_dirs, agg_root, decision_time_hours=decision_time_hours
    )
    perclass_auroc_summary = _aggregate_perclass_auroc_curves(
        fold_dirs, agg_root, decision_time_hours=decision_time_hours
    )

    # Threshold + AUC scalar summary.
    def _stat(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
        clean = [v for v in values if v is not None and v == v]
        if not clean:
            return {"mean": None, "std": None, "n": 0}
        if len(clean) == 1:
            return {"mean": float(clean[0]), "std": 0.0, "n": 1}
        import statistics

        return {
            "mean": float(statistics.mean(clean)),
            "std": float(statistics.stdev(clean)),
            "n": int(len(clean)),
        }

    summary: Dict[str, Any] = {
        "n_folds": len(fold_dirs),
        "fold_ids": [int(fd.name.split("_")[-1]) for fd in fold_dirs],
        "binary_auc": _stat([roc.get("auc", float("nan")) for _, roc in per_fold_binary_roc]),
        "three_class_auc_ovr": {
            cls: _stat([ovr.get(cls, {}).get("auc") for _, ovr in per_fold_3class_roc])
            for cls in ("healthy", "acidosis", "hie")
        },
        "thresholds": {
            "instantaneous": _stat(
                [t.get("threshold_instantaneous") for _, t in per_fold_thresholds]
            ),
            "cumulative": _stat(
                [t.get("threshold_cumulative") for _, t in per_fold_thresholds]
            ),
            "overall": _stat(
                [t.get("threshold_overall") for _, t in per_fold_thresholds]
            ),
        },
    }
    if mean_cm is not None:
        summary["confusion_matrix_3class_mean"] = mean_cm.tolist()
    summary["per_class_metric_curves"] = perclass_summary
    summary["binary_by_underlying_class_curves"] = binary_by_class_summary
    summary["per_class_subgroup_metric_curves"] = perclass_subgroup_summary
    summary["per_class_auroc_vs_time"] = perclass_auroc_summary
    if decision_time_hours is not None:
        summary["decision_time_hours"] = float(decision_time_hours)

    (run_dir / "aggregated_results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(
        f"aggregate_results: binary_auc_mean={binary_auc_mean} "
        f"three_class_auc_ovr={three_class_auc_means}"
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector for testing.

    Returns:
        Exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Aggregate guid_cls_v1 per-fold results"
    )
    parser.add_argument("--run-dir", required=True, help="Run directory containing fold_*/")
    parser.add_argument("--fold-ids", type=int, nargs="*", default=None)
    parser.add_argument(
        "--decision-time-hours",
        type=float,
        default=None,
        help=(
            "Override the decision-time vertical-line annotation drawn on "
            "every cross-fold vs-time plot. Defaults to the value persisted "
            "in per-fold thresholds.json or the run-level config."
        ),
    )
    args = parser.parse_args(argv)
    aggregate_results(
        run_dir=Path(args.run_dir).resolve(),
        fold_ids=args.fold_ids,
        decision_time_hours=args.decision_time_hours,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
