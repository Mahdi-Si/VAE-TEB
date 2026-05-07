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

# Palettes copied verbatim from the per-fold plotters in
# ``clinical_metrics_utils.py`` so cross-fold plots are visually identical
# to the per-fold reference. Edit those palettes in lock-step with the
# per-fold helpers if they ever change.
_DIAGNOSIS_PALETTE: Dict[str, str] = {
    "healthy": "#2ecc71",
    "acidosis": "#e74c3c",
    "hie": "#e67e22",
    "unhealthy": "#95a5a6",
}
_CS_POS_NEG: Dict[str, str] = {"pos": "#3498db", "neg": "#9b59b6"}
_BG_POS_NEG: Dict[str, str] = {"pos": "#f39c12", "neg": "#16a085"}
_HEALTHY_COMBO_COLORS: Tuple[str, ...] = (
    "#e74c3c",
    "#3498db",
    "#f39c12",
    "#9b59b6",
)
# 3-class panel palette used by per-class / AUROC plotters (slightly
# different from diagnosis palette — matches the per-fold helpers).
_PER_CLASS_PALETTE: Dict[str, str] = {
    "healthy": "#27ae60",
    "acidosis": "#f39c12",
    "hie": "#c0392b",
}
# Binary metric line colors (sensitivity / specificity / fpr).
_BINARY_METRIC_PALETTE: Dict[str, str] = {
    "sensitivity": "#2ecc71",
    "specificity": "#3498db",
    "fpr": "#e74c3c",
}


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


def _shared_grid(frames: Sequence[pd.DataFrame], n_points: int = 64) -> Optional[np.ndarray]:
    """Build a common ``bin_center`` interpolation grid spanning all frames.

    Returns ``None`` when no finite range is available (e.g. every frame
    is empty or has fewer than two distinct bin centres).
    """
    centres: List[float] = []
    for d in frames:
        if d is None or d.empty or "bin_center" not in d.columns:
            continue
        arr = d["bin_center"].astype(float).to_numpy()
        if arr.size:
            centres.append(float(np.nanmin(arr)))
            centres.append(float(np.nanmax(arr)))
    if not centres:
        return None
    grid_min = float(np.nanmin(centres))
    grid_max = float(np.nanmax(centres))
    if not (np.isfinite(grid_min) and np.isfinite(grid_max)) or grid_min >= grid_max:
        return None
    return np.linspace(grid_min, grid_max, n_points)


def _plot_mean_minmax_curve(
    ax,
    grid: np.ndarray,
    per_fold: List[np.ndarray],
    *,
    label: str,
    color: str,
    marker: str = "o",
    linewidth: float = 2.5,
    markersize: float = 6.0,
    band_alpha: float = 0.2,
) -> Optional[Tuple[float, float]]:
    """Plot the per-fold mean line + min/max fill_between band.

    No per-fold thin lines and no decision-time annotation — the band
    captures cross-fold spread, and the band's meaning is communicated
    once per figure via :func:`_add_minmax_band_note`.

    Args:
        ax: Matplotlib axes to draw on.
        grid: Common x-axis grid (``bin_center`` values, hours before delivery).
        per_fold: List of length ``n_folds``, each entry a 1-D array of
            metric values aligned to ``grid`` (NaN where the fold has no
            data at that bin).
        label: Legend label for the mean line.
        color: Line + band colour.
        marker: Marker style for the mean line (default ``"o"``).
        linewidth: Mean-line width (default 2.5 — matches per-fold).
        markersize: Marker size (default 6 — matches per-fold).
        band_alpha: Min/max band opacity (default 0.2).

    Returns:
        ``(mean_avg_over_bins, range_avg_over_bins)`` summary scalars,
        or ``None`` when ``per_fold`` is empty.
    """
    if not per_fold:
        return None
    arr = np.vstack(per_fold)
    mean = np.nanmean(arr, axis=0)
    lo = np.nanmin(arr, axis=0)
    hi = np.nanmax(arr, axis=0)
    ax.plot(
        grid, mean, color=color, lw=linewidth, marker=marker,
        markersize=markersize, label=label,
    )
    ax.fill_between(grid, lo, hi, color=color, alpha=band_alpha, linewidth=0)
    return float(np.nanmean(mean)), float(np.nanmean(hi - lo))


def _add_minmax_band_note(fig, n_folds: int) -> None:
    """Add a one-line caption stating the band represents min/max across folds."""
    fig.text(
        0.99,
        0.01,
        f"Shaded region: min/max across {int(n_folds)} folds",
        fontsize=9,
        color="0.4",
        ha="right",
        va="bottom",
    )


def _aggregate_perclass_metric_curves(
    fold_dirs: List[Path],
    agg_root: Path,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Cross-fold per-class metric curves (mirrors ``plot_perclass_panel_combined``).

    Reads each fold's long-format
    ``multiclass_head/per_class_vs_time/<metric_type>.csv`` and emits
    one combined 3-panel PNG per metric_type
    (``aggregated_plots/multiclass_head/per_class_vs_time/<metric_type>_panel.png``)
    plus a per-class CSV
    (``<metric_type>_<class>.csv`` with ``_mean / _min / _max / n_folds``
    columns).

    Each panel shows three lines (sensitivity, specificity, FPR) for one
    class, with min/max bands across folds. No decision-time line.
    """
    import matplotlib.pyplot as plt  # noqa: WPS433

    metric_label = {
        "instantaneous": "Instantaneous Decisions",
        "committed_cumulative": "Committed (Cumulative)",
        "committed_overall": "Committed (Overall)",
    }

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for metric_type in METRIC_TYPES:
        out_dir = agg_root / "multiclass_head" / "per_class_vs_time"

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

        if not any(per_fold_by_class.values()):
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        n_folds_used = max(len(v) for v in per_fold_by_class.values())
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        per_class_summary: Dict[str, Dict[str, float]] = {}

        for ax, class_name in zip(axes, PER_CLASS_NAMES):
            collected = per_fold_by_class[class_name]
            if not collected:
                ax.text(0.5, 0.5, f"{class_name}: no data", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            grid = _shared_grid(collected)
            if grid is None:
                ax.text(0.5, 0.5, f"{class_name}: insufficient bins", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            per_metric: Dict[str, List[np.ndarray]] = {
                "sensitivity": [_interp_metric_curve(d, grid, "sensitivity") for d in collected],
                "specificity": [_interp_metric_curve(d, grid, "specificity") for d in collected],
                "fpr": [_interp_metric_curve(d, grid, "fpr") for d in collected],
            }
            sens_stat = _plot_mean_minmax_curve(
                ax, grid, per_metric["sensitivity"],
                label="Sensitivity", color=_BINARY_METRIC_PALETTE["sensitivity"],
                marker="o", linewidth=2.2,
            )
            _plot_mean_minmax_curve(
                ax, grid, per_metric["specificity"],
                label="Specificity", color=_BINARY_METRIC_PALETTE["specificity"],
                marker="s", linewidth=2.2,
            )
            fpr_stat = _plot_mean_minmax_curve(
                ax, grid, per_metric["fpr"],
                label="FPR", color=_BINARY_METRIC_PALETTE["fpr"],
                marker="^", linewidth=2.2,
            )
            ax.set_title(
                class_name,
                color=_PER_CLASS_PALETTE.get(class_name, "black"),
                fontweight="bold",
            )
            ax.set_xlabel("Hours Before Birth")
            ax.set_ylim([0, 1.05])
            ax.invert_xaxis()
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc="best")

            # Per-class CSV: min/max per metric.
            sens_arr = np.vstack(per_metric["sensitivity"])
            spec_arr = np.vstack(per_metric["specificity"])
            fpr_arr = np.vstack(per_metric["fpr"])
            pd.DataFrame(
                {
                    "bin_center": grid,
                    "sensitivity_mean": np.nanmean(sens_arr, axis=0),
                    "sensitivity_min": np.nanmin(sens_arr, axis=0),
                    "sensitivity_max": np.nanmax(sens_arr, axis=0),
                    "specificity_mean": np.nanmean(spec_arr, axis=0),
                    "specificity_min": np.nanmin(spec_arr, axis=0),
                    "specificity_max": np.nanmax(spec_arr, axis=0),
                    "fpr_mean": np.nanmean(fpr_arr, axis=0),
                    "fpr_min": np.nanmin(fpr_arr, axis=0),
                    "fpr_max": np.nanmax(fpr_arr, axis=0),
                    "n_folds": np.sum(np.isfinite(sens_arr), axis=0),
                }
            ).to_csv(out_dir / f"{metric_type}_{class_name}.csv", index=False)

            per_class_summary[class_name] = {
                "n_folds": int(len(collected)),
                "sensitivity_mean_avg_over_bins": (sens_stat or (float("nan"), float("nan")))[0],
                "sensitivity_minmax_avg_over_bins": (sens_stat or (float("nan"), float("nan")))[1],
                "fpr_mean_avg_over_bins": (fpr_stat or (float("nan"), float("nan")))[0],
                "fpr_minmax_avg_over_bins": (fpr_stat or (float("nan"), float("nan")))[1],
            }

        axes[0].set_ylabel("Metric value")
        suptitle = f"Per-class metrics vs time — {metric_label.get(metric_type, metric_type)}"
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
        fig.tight_layout()
        _add_minmax_band_note(fig, n_folds_used)
        fig.savefig(out_dir / f"{metric_type}_panel.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        if per_class_summary:
            summary[metric_type] = per_class_summary
    return summary


def _aggregate_binary_by_underlying_class_curves(
    fold_dirs: List[Path],
    agg_root: Path,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Cross-fold binary curves restricted to each underlying 3-class label.

    Reads ``binary_head/by_underlying_class_vs_time/<metric_type>.csv``
    per fold and emits one combined plot per metric_type under
    ``aggregated_plots/binary_head/by_underlying_class_vs_time/<metric_type>.png``
    plus per-(metric_type, restrict_class) CSVs with ``_mean / _min / _max``.

    Layout: single axes overlaying acidosis (orange) and HIE (red)
    sensitivity curves, with min/max bands. No decision-time line.
    """
    import matplotlib.pyplot as plt  # noqa: WPS433

    palette = {"acidosis": _PER_CLASS_PALETTE["acidosis"], "hie": _PER_CLASS_PALETTE["hie"]}
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for metric_type in METRIC_TYPES:
        sub_summary: Dict[str, Dict[str, float]] = {}
        out_dir = agg_root / "binary_head" / "by_underlying_class_vs_time"

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

        if not any(per_fold_by_restrict.values()):
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        n_folds_used = max(len(v) for v in per_fold_by_restrict.values())

        fig, ax = plt.subplots(figsize=(12, 6))
        for restrict, collected in per_fold_by_restrict.items():
            if not collected:
                continue
            grid = _shared_grid(collected)
            if grid is None:
                continue
            sens = [_interp_metric_curve(d, grid, "sensitivity") for d in collected]
            fpr = [_interp_metric_curve(d, grid, "fpr") for d in collected]
            spec = [_interp_metric_curve(d, grid, "specificity") for d in collected]

            sens_stat = _plot_mean_minmax_curve(
                ax, grid, sens,
                label=f"Only {restrict.capitalize()} (N={len(collected)} folds)",
                color=palette[restrict],
                marker="o",
            )
            # FPR is preserved in the CSV below but not overlaid on the
            # combined sensitivity plot to avoid 4-line clutter.

            # Per-(metric_type, restrict_class) CSV with all three metrics.
            sens_arr = np.vstack(sens)
            fpr_arr = np.vstack(fpr)
            spec_arr = np.vstack(spec)
            pd.DataFrame(
                {
                    "bin_center": grid,
                    "sensitivity_mean": np.nanmean(sens_arr, axis=0),
                    "sensitivity_min": np.nanmin(sens_arr, axis=0),
                    "sensitivity_max": np.nanmax(sens_arr, axis=0),
                    "specificity_mean": np.nanmean(spec_arr, axis=0),
                    "specificity_min": np.nanmin(spec_arr, axis=0),
                    "specificity_max": np.nanmax(spec_arr, axis=0),
                    "fpr_mean": np.nanmean(fpr_arr, axis=0),
                    "fpr_min": np.nanmin(fpr_arr, axis=0),
                    "fpr_max": np.nanmax(fpr_arr, axis=0),
                    "n_folds": np.sum(np.isfinite(sens_arr), axis=0),
                }
            ).to_csv(out_dir / f"{metric_type}_{restrict}.csv", index=False)

            sub_summary[restrict] = {
                "n_folds": int(len(collected)),
                "sensitivity_mean_avg_over_bins": (sens_stat or (float("nan"), float("nan")))[0],
                "sensitivity_minmax_avg_over_bins": (sens_stat or (float("nan"), float("nan")))[1],
                "fpr_mean_avg_over_bins": (
                    float(np.nanmean(np.nanmean(fpr_arr, axis=0))) if fpr else float("nan")
                ),
            }

        ax.set_xlabel("Hours Before Birth", fontsize=13)
        ax.set_ylabel("Sensitivity", fontsize=13)
        ax.set_title(
            f"Binary head — by underlying class — {metric_type.replace('_', ' ').title()}",
            fontsize=14, fontweight="bold",
        )
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.invert_xaxis()
        fig.tight_layout()
        _add_minmax_band_note(fig, n_folds_used)
        fig.savefig(out_dir / f"{metric_type}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        if sub_summary:
            summary[metric_type] = sub_summary
    return summary


def _aggregate_perclass_subgroup_metric_curves(
    fold_dirs: List[Path],
    agg_root: Path,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Cross-fold per-class × subgroup curves with min/max bands.

    Reads each fold's long-format
    ``multiclass_head/per_class_subgroups_vs_time/<metric_type>.csv``
    (cols: ``bin_center, class, subgroup, sensitivity, specificity,
    fpr, n_pos, n_neg, n``) and emits cross-fold curves under
    ``aggregated_plots/multiclass_head/per_class_subgroups_vs_time/<metric_type>/<class>_<subgroup>.{csv,png}``.

    Each (class, subgroup) cell renders a 2-panel figure (sensitivity
    + FPR), each panel showing the cross-fold mean line + a min/max
    fill_between band. No decision-time line.

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
            for grp_key, grp in df.groupby(["class", "subgroup"], dropna=False):
                cls, sg = grp_key  # type: ignore[misc]
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
            grid = _shared_grid(collected)
            if grid is None:
                continue
            per_fold_sens = [
                _interp_metric_curve(d, grid, "sensitivity") for d in collected
            ]
            per_fold_fpr = [
                _interp_metric_curve(d, grid, "fpr") for d in collected
            ]
            color = _PER_CLASS_PALETTE.get(class_name, "C0")

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sens_stat = _plot_mean_minmax_curve(
                axes[0], grid, per_fold_sens,
                label=f"{subgroup_name} (N={len(collected)} folds)",
                color=color, marker="o",
            )
            axes[0].set_title(
                f"{class_name} | {subgroup_name} — sensitivity",
                fontsize=14, fontweight="bold",
                color=color,
            )
            axes[0].set_xlabel("Hours Before Birth", fontsize=13)
            axes[0].set_ylabel("Sensitivity", fontsize=13)
            axes[0].invert_xaxis()
            axes[0].set_ylim([0, 1.05])
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(fontsize=10, loc="best")

            fpr_stat = _plot_mean_minmax_curve(
                axes[1], grid, per_fold_fpr,
                label=f"{subgroup_name} (N={len(collected)} folds)",
                color=_BINARY_METRIC_PALETTE["fpr"], marker="s",
            )
            axes[1].set_title(
                f"{class_name} | {subgroup_name} — FPR",
                fontsize=14, fontweight="bold",
            )
            axes[1].set_xlabel("Hours Before Birth", fontsize=13)
            axes[1].set_ylabel("FPR", fontsize=13)
            axes[1].invert_xaxis()
            axes[1].set_ylim([0, 1.05])
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(fontsize=10, loc="best")

            fig.tight_layout()
            _add_minmax_band_note(fig, len(collected))
            stem = f"{class_name}_{subgroup_name}"
            fig.savefig(out_root / f"{stem}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            sens_arr = np.vstack(per_fold_sens)
            fpr_arr = np.vstack(per_fold_fpr)
            pd.DataFrame(
                {
                    "bin_center": grid,
                    "sensitivity_mean": np.nanmean(sens_arr, axis=0),
                    "sensitivity_min": np.nanmin(sens_arr, axis=0),
                    "sensitivity_max": np.nanmax(sens_arr, axis=0),
                    "fpr_mean": np.nanmean(fpr_arr, axis=0),
                    "fpr_min": np.nanmin(fpr_arr, axis=0),
                    "fpr_max": np.nanmax(fpr_arr, axis=0),
                    "n_folds": np.sum(np.isfinite(sens_arr), axis=0),
                }
            ).to_csv(out_root / f"{stem}.csv", index=False)

            entry = {
                "n_folds": int(len(per_fold_sens)),
                "sensitivity_mean_avg_over_bins": (sens_stat or (float("nan"), float("nan")))[0],
                "sensitivity_minmax_avg_over_bins": (sens_stat or (float("nan"), float("nan")))[1],
                "fpr_mean_avg_over_bins": (fpr_stat or (float("nan"), float("nan")))[0],
                "fpr_minmax_avg_over_bins": (fpr_stat or (float("nan"), float("nan")))[1],
            }
            per_class_summary.setdefault(class_name, {})[subgroup_name] = entry
        if per_class_summary:
            summary[metric_type] = per_class_summary
    return summary


def _aggregate_perclass_auroc_curves(
    fold_dirs: List[Path],
    agg_root: Path,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Cross-fold per-class one-vs-rest AUROC vs time (min/max bands).

    Reads each fold's long-format
    ``multiclass_head/auroc_vs_time/<metric_type>.csv`` (cols:
    ``bin_center, class, auroc, n_pos, n_neg, n``) and emits cross-fold
    curves under
    ``aggregated_plots/multiclass_head/auroc_vs_time/<metric_type>.{csv,png}``.

    The per-class panel uses the standard 3-class palette. A horizontal
    chance line at AUROC=0.5 is drawn for orientation. No decision-time
    line.
    """
    import matplotlib.pyplot as plt  # noqa: WPS433

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
        n_folds_used = max(len(v) for v in per_fold_by_class.values())

        all_frames: List[pd.DataFrame] = [
            d for col in per_fold_by_class.values() for d in col
        ]
        grid = _shared_grid(all_frames)
        if grid is None:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        long_rows: List[Dict[str, Any]] = []
        for class_name in PER_CLASS_NAMES:
            collected = per_fold_by_class[class_name]
            if not collected:
                continue
            per_fold = [
                _interp_metric_curve(d, grid, "auroc") for d in collected
            ]
            stat = _plot_mean_minmax_curve(
                ax, grid, per_fold,
                label=f"{class_name} (N={len(collected)} folds)",
                color=_PER_CLASS_PALETTE.get(class_name, "C0"),
                marker="o",
            )
            arr = np.vstack(per_fold)
            for i, c in enumerate(grid):
                long_rows.append(
                    {
                        "bin_center": float(c),
                        "class": class_name,
                        "auroc_mean": float(np.nanmean(arr[:, i])),
                        "auroc_min": float(np.nanmin(arr[:, i])),
                        "auroc_max": float(np.nanmax(arr[:, i])),
                        "n_folds": int(np.sum(np.isfinite(arr[:, i]))),
                    }
                )
            per_class_summary[class_name] = {
                "n_folds": int(len(per_fold)),
                "auroc_mean_avg_over_bins": (stat or (float("nan"), float("nan")))[0],
                "auroc_minmax_avg_over_bins": (stat or (float("nan"), float("nan")))[1],
            }
        ax.axhline(0.5, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("Hours Before Birth", fontsize=13)
        ax.set_ylabel("AUROC (one-vs-rest)", fontsize=13)
        ax.set_title(
            f"Per-class AUROC vs time — {metric_type.replace('_', ' ').title()}",
            fontsize=14, fontweight="bold",
        )
        ax.invert_xaxis()
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc="best")
        fig.tight_layout()
        _add_minmax_band_note(fig, n_folds_used)
        fig.savefig(out_dir / f"{metric_type}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        if long_rows:
            pd.DataFrame(long_rows).to_csv(
                out_dir / f"{metric_type}.csv", index=False
            )
        if per_class_summary:
            summary[metric_type] = per_class_summary
    return summary


# ---------------------------------------------------------------------------
# Binary metrics-vs-time aggregator (overall sensitivity / specificity / FPR)
# ---------------------------------------------------------------------------


def _aggregate_binary_metrics_vs_time_curves(
    fold_dirs: List[Path],
    agg_root: Path,
) -> Dict[str, Dict[str, float]]:
    """Cross-fold binary head sensitivity / specificity / FPR vs time.

    Reads each fold's ``binary_head/metrics_vs_time/<metric_type>.csv``
    (cols: ``bin_center, sensitivity, specificity, fpr, n_pos, n_neg, n``)
    and emits one combined PNG per metric_type at
    ``aggregated_plots/binary_head/metrics_vs_time/<metric_type>.png``
    plus a CSV with ``_mean / _min / _max / n_folds`` for each metric.

    Layout mirrors the per-fold ``plot_single_metric_type`` style: three
    lines on a single 12×6 axes, palette
    ``{sensitivity: green, specificity: blue, fpr: red}``, no decision
    line. Min/max bands across folds.
    """
    import matplotlib.pyplot as plt  # noqa: WPS433

    summary: Dict[str, Dict[str, float]] = {}
    for metric_type in METRIC_TYPES:
        out_dir = agg_root / "binary_head" / "metrics_vs_time"

        per_fold_frames: List[pd.DataFrame] = []
        for fd in fold_dirs:
            csv = (
                fd
                / "evaluation"
                / "binary_head"
                / "metrics_vs_time"
                / f"{metric_type}.csv"
            )
            if not csv.exists():
                continue
            try:
                df = pd.read_csv(csv)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"could not read {csv}: {exc}")
                continue
            if df.empty or "bin_center" not in df.columns:
                continue
            per_fold_frames.append(df)

        if not per_fold_frames:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        grid = _shared_grid(per_fold_frames)
        if grid is None:
            continue

        per_metric: Dict[str, List[np.ndarray]] = {
            "sensitivity": [_interp_metric_curve(d, grid, "sensitivity") for d in per_fold_frames],
            "specificity": [_interp_metric_curve(d, grid, "specificity") for d in per_fold_frames],
            "fpr": [_interp_metric_curve(d, grid, "fpr") for d in per_fold_frames],
        }

        fig, ax = plt.subplots(figsize=(12, 6))
        per_mt_summary: Dict[str, float] = {}
        for metric_name, color in _BINARY_METRIC_PALETTE.items():
            stat = _plot_mean_minmax_curve(
                ax, grid, per_metric[metric_name],
                label=metric_name.capitalize(),
                color=color,
                marker={"sensitivity": "o", "specificity": "s", "fpr": "^"}[metric_name],
            )
            if stat is not None:
                per_mt_summary[f"{metric_name}_mean_avg_over_bins"] = stat[0]
                per_mt_summary[f"{metric_name}_minmax_avg_over_bins"] = stat[1]

        ax.set_xlabel("Hours Before Birth", fontsize=13)
        ax.set_ylabel("Metric value", fontsize=13)
        ax.set_title(
            f"Binary head — metrics vs time — {metric_type.replace('_', ' ').title()}",
            fontsize=14, fontweight="bold",
        )
        ax.set_ylim([0, 1.05])
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc="best")
        fig.tight_layout()
        _add_minmax_band_note(fig, len(per_fold_frames))
        fig.savefig(out_dir / f"{metric_type}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        sens_arr = np.vstack(per_metric["sensitivity"])
        spec_arr = np.vstack(per_metric["specificity"])
        fpr_arr = np.vstack(per_metric["fpr"])
        pd.DataFrame(
            {
                "bin_center": grid,
                "sensitivity_mean": np.nanmean(sens_arr, axis=0),
                "sensitivity_min": np.nanmin(sens_arr, axis=0),
                "sensitivity_max": np.nanmax(sens_arr, axis=0),
                "specificity_mean": np.nanmean(spec_arr, axis=0),
                "specificity_min": np.nanmin(spec_arr, axis=0),
                "specificity_max": np.nanmax(spec_arr, axis=0),
                "fpr_mean": np.nanmean(fpr_arr, axis=0),
                "fpr_min": np.nanmin(fpr_arr, axis=0),
                "fpr_max": np.nanmax(fpr_arr, axis=0),
                "n_folds": np.sum(np.isfinite(sens_arr), axis=0),
            }
        ).to_csv(out_dir / f"{metric_type}.csv", index=False)

        per_mt_summary["n_folds"] = float(len(per_fold_frames))
        summary[metric_type] = per_mt_summary
    return summary


# ---------------------------------------------------------------------------
# Binary subgroup aggregator (diagnosis / CS / BG / healthy stratification)
# ---------------------------------------------------------------------------


def _read_binary_subgroup_long(
    fold_dirs: List[Path], metric_type: str,
) -> Dict[str, List[pd.DataFrame]]:
    """Read each fold's ``binary_head/subgroups_vs_time/<metric_type>.csv``.

    Groups the long-format rows by ``subgroup`` and returns a dict mapping
    ``subgroup_name -> [per-fold DataFrame slice]``. Folds with a missing
    or unreadable CSV contribute nothing for that metric_type.
    """
    out: Dict[str, List[pd.DataFrame]] = {}
    for fd in fold_dirs:
        csv = (
            fd
            / "evaluation"
            / "binary_head"
            / "subgroups_vs_time"
            / f"{metric_type}.csv"
        )
        if not csv.exists():
            continue
        try:
            df = pd.read_csv(csv)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"could not read {csv}: {exc}")
            continue
        if df.empty or "subgroup" not in df.columns or "bin_center" not in df.columns:
            continue
        for sg_name, grp in df.groupby("subgroup", dropna=False):
            if grp.empty:
                continue
            out.setdefault(str(sg_name), []).append(grp.reset_index(drop=True))
    return out


def _write_subgroup_csv(
    output_path: Path,
    grid: np.ndarray,
    per_fold_metric: List[np.ndarray],
    *,
    metric_name: str,
) -> None:
    """Persist the cross-fold mean/min/max for one metric column."""
    arr = np.vstack(per_fold_metric)
    pd.DataFrame(
        {
            "bin_center": grid,
            f"{metric_name}_mean": np.nanmean(arr, axis=0),
            f"{metric_name}_min": np.nanmin(arr, axis=0),
            f"{metric_name}_max": np.nanmax(arr, axis=0),
            "n_folds": np.sum(np.isfinite(arr), axis=0),
        }
    ).to_csv(output_path, index=False)


def _plot_agg_diagnosis_comparison(
    per_fold_by_subgroup: Dict[str, List[pd.DataFrame]],
    output_dir: Path,
    metric_type: str,
) -> None:
    """Cross-fold mirror of :func:`clinical_metrics_utils._plot_diagnosis_comparison`.

    Renders an overlay of healthy/acidosis/hie/unhealthy sensitivity
    (specificity for healthy) curves with min/max bands. Output:
    ``output_dir/diagnosis_comparison.png`` + per-group CSVs.
    """
    import matplotlib.pyplot as plt  # noqa: WPS433

    diagnosis_groups = ["healthy", "acidosis", "hie", "unhealthy"]
    available = [g for g in diagnosis_groups if g in per_fold_by_subgroup]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    n_folds_max = 0
    for group in available:
        collected = per_fold_by_subgroup[group]
        if not collected:
            continue
        grid = _shared_grid(collected)
        if grid is None:
            continue
        # Healthy uses specificity (no positives → sensitivity = NaN);
        # all other groups use sensitivity.
        metric_col = "specificity" if group == "healthy" else "sensitivity"
        per_fold = [_interp_metric_curve(d, grid, metric_col) for d in collected]
        _plot_mean_minmax_curve(
            ax, grid, per_fold,
            label=f"{group.capitalize()} (N={len(collected)} folds)",
            color=_DIAGNOSIS_PALETTE.get(group, "0.4"),
            marker="o",
        )
        _write_subgroup_csv(
            output_dir / f"diagnosis_{group}.csv", grid, per_fold,
            metric_name=metric_col,
        )
        n_folds_max = max(n_folds_max, len(collected))

    ax.set_xlabel("Hours Before Birth", fontsize=13)
    ax.set_ylabel("Sensitivity (Specificity for Healthy)", fontsize=13)
    ax.set_title(
        f"Diagnosis Comparison - {metric_type.replace('_', ' ').title()}",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.invert_xaxis()
    fig.tight_layout()
    _add_minmax_band_note(fig, n_folds_max)
    fig.savefig(output_dir / "diagnosis_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_agg_cs_stratification(
    per_fold_by_subgroup: Dict[str, List[pd.DataFrame]],
    output_dir: Path,
    metric_type: str,
) -> None:
    """Cross-fold mirror of :func:`clinical_metrics_utils._plot_cs_stratification`.

    Three plots: unhealthy / hie / acidosis × CS pos/neg. Sensitivity y.
    """
    import matplotlib.pyplot as plt  # noqa: WPS433

    stratifications = [
        (["unhealthy_cs_pos", "unhealthy_cs_neg"], "Unhealthy by CS Status",
         "unhealthy_cs_stratification"),
        (["hie_cs_pos", "hie_cs_neg"], "HIE by CS Status",
         "hie_cs_stratification"),
        (["acidosis_cs_pos", "acidosis_cs_neg"], "Acidosis by CS Status",
         "acidosis_cs_stratification"),
    ]
    for groups, title_text, file_stem in stratifications:
        available = [g for g in groups if g in per_fold_by_subgroup]
        if not available:
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        n_folds_max = 0
        for group in available:
            collected = per_fold_by_subgroup[group]
            if not collected:
                continue
            grid = _shared_grid(collected)
            if grid is None:
                continue
            per_fold = [_interp_metric_curve(d, grid, "sensitivity") for d in collected]
            label_base = "CS Positive" if "pos" in group else "CS Negative"
            color = _CS_POS_NEG["pos"] if "pos" in group else _CS_POS_NEG["neg"]
            _plot_mean_minmax_curve(
                ax, grid, per_fold,
                label=f"{label_base} (N={len(collected)} folds)",
                color=color, marker="o",
            )
            _write_subgroup_csv(
                output_dir / f"{group}.csv", grid, per_fold,
                metric_name="sensitivity",
            )
            n_folds_max = max(n_folds_max, len(collected))

        ax.set_xlabel("Hours Before Birth", fontsize=13)
        ax.set_ylabel("Sensitivity", fontsize=13)
        ax.set_title(
            f"{title_text} - {metric_type.replace('_', ' ').title()}",
            fontsize=14, fontweight="bold",
        )
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.invert_xaxis()
        fig.tight_layout()
        _add_minmax_band_note(fig, n_folds_max)
        fig.savefig(output_dir / f"{file_stem}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_agg_bg_stratification(
    per_fold_by_subgroup: Dict[str, List[pd.DataFrame]],
    output_dir: Path,
    metric_type: str,
) -> None:
    """Cross-fold mirror of :func:`clinical_metrics_utils._plot_bg_stratification`.

    One plot: acidosis × BG pos/neg. Sensitivity y.
    """
    import matplotlib.pyplot as plt  # noqa: WPS433

    groups = ["acidosis_bg_pos", "acidosis_bg_neg"]
    available = [g for g in groups if g in per_fold_by_subgroup]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    n_folds_max = 0
    for group in available:
        collected = per_fold_by_subgroup[group]
        if not collected:
            continue
        grid = _shared_grid(collected)
        if grid is None:
            continue
        per_fold = [_interp_metric_curve(d, grid, "sensitivity") for d in collected]
        label_base = "BG Positive" if "pos" in group else "BG Negative"
        color = _BG_POS_NEG["pos"] if "pos" in group else _BG_POS_NEG["neg"]
        _plot_mean_minmax_curve(
            ax, grid, per_fold,
            label=f"{label_base} (N={len(collected)} folds)",
            color=color, marker="o",
        )
        _write_subgroup_csv(
            output_dir / f"{group}.csv", grid, per_fold,
            metric_name="sensitivity",
        )
        n_folds_max = max(n_folds_max, len(collected))

    ax.set_xlabel("Hours Before Birth", fontsize=13)
    ax.set_ylabel("Sensitivity", fontsize=13)
    ax.set_title(
        f"Acidosis by BG Status - {metric_type.replace('_', ' ').title()}",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.invert_xaxis()
    fig.tight_layout()
    _add_minmax_band_note(fig, n_folds_max)
    fig.savefig(output_dir / "acidosis_bg_stratification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_agg_healthy_subgroups(
    per_fold_by_subgroup: Dict[str, List[pd.DataFrame]],
    output_dir: Path,
    metric_type: str,
) -> None:
    """Cross-fold mirror of :func:`clinical_metrics_utils._plot_healthy_subgroups`.

    Three plots (CS, BG, BG×CS combos). Healthy subgroups are pure
    negatives, so y is specificity (not sensitivity).
    """
    import matplotlib.pyplot as plt  # noqa: WPS433

    # Healthy by CS.
    cs_groups = ["healthy_cs_pos", "healthy_cs_neg"]
    available_cs = [g for g in cs_groups if g in per_fold_by_subgroup]
    if available_cs:
        fig, ax = plt.subplots(figsize=(12, 6))
        n_folds_max = 0
        for group in available_cs:
            collected = per_fold_by_subgroup[group]
            if not collected:
                continue
            grid = _shared_grid(collected)
            if grid is None:
                continue
            per_fold = [_interp_metric_curve(d, grid, "specificity") for d in collected]
            label_base = "CS Positive" if "pos" in group else "CS Negative"
            color = _CS_POS_NEG["pos"] if "pos" in group else _CS_POS_NEG["neg"]
            _plot_mean_minmax_curve(
                ax, grid, per_fold,
                label=f"{label_base} (N={len(collected)} folds)",
                color=color, marker="o",
            )
            _write_subgroup_csv(
                output_dir / f"{group}.csv", grid, per_fold,
                metric_name="specificity",
            )
            n_folds_max = max(n_folds_max, len(collected))
        ax.set_xlabel("Hours Before Birth", fontsize=13)
        ax.set_ylabel("Specificity (Correctly Identified as Healthy)", fontsize=13)
        ax.set_title(
            f"Healthy by CS Status - {metric_type.replace('_', ' ').title()}",
            fontsize=14, fontweight="bold",
        )
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.invert_xaxis()
        fig.tight_layout()
        _add_minmax_band_note(fig, n_folds_max)
        fig.savefig(output_dir / "healthy_cs_stratification.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Healthy by BG.
    bg_groups = ["healthy_bg_pos", "healthy_bg_neg"]
    available_bg = [g for g in bg_groups if g in per_fold_by_subgroup]
    if available_bg:
        fig, ax = plt.subplots(figsize=(12, 6))
        n_folds_max = 0
        for group in available_bg:
            collected = per_fold_by_subgroup[group]
            if not collected:
                continue
            grid = _shared_grid(collected)
            if grid is None:
                continue
            per_fold = [_interp_metric_curve(d, grid, "specificity") for d in collected]
            label_base = "BG Positive" if "pos" in group else "BG Negative"
            color = _BG_POS_NEG["pos"] if "pos" in group else _BG_POS_NEG["neg"]
            _plot_mean_minmax_curve(
                ax, grid, per_fold,
                label=f"{label_base} (N={len(collected)} folds)",
                color=color, marker="o",
            )
            _write_subgroup_csv(
                output_dir / f"{group}.csv", grid, per_fold,
                metric_name="specificity",
            )
            n_folds_max = max(n_folds_max, len(collected))
        ax.set_xlabel("Hours Before Birth", fontsize=13)
        ax.set_ylabel("Specificity (Correctly Identified as Healthy)", fontsize=13)
        ax.set_title(
            f"Healthy by BG Status - {metric_type.replace('_', ' ').title()}",
            fontsize=14, fontweight="bold",
        )
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.invert_xaxis()
        fig.tight_layout()
        _add_minmax_band_note(fig, n_folds_max)
        fig.savefig(output_dir / "healthy_bg_stratification.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Healthy BG×CS 4-way combinations.
    combo_groups = [
        "healthy_bg_pos_cs_pos", "healthy_bg_pos_cs_neg",
        "healthy_bg_neg_cs_pos", "healthy_bg_neg_cs_neg",
    ]
    available_combo = [g for g in combo_groups if g in per_fold_by_subgroup]
    if available_combo:
        fig, ax = plt.subplots(figsize=(14, 7))
        n_folds_max = 0
        for i, group in enumerate(available_combo):
            collected = per_fold_by_subgroup[group]
            if not collected:
                continue
            grid = _shared_grid(collected)
            if grid is None:
                continue
            per_fold = [_interp_metric_curve(d, grid, "specificity") for d in collected]
            label_base = group.replace("healthy_", "").replace("_", " ").upper()
            color = _HEALTHY_COMBO_COLORS[i % len(_HEALTHY_COMBO_COLORS)]
            _plot_mean_minmax_curve(
                ax, grid, per_fold,
                label=f"{label_base} (N={len(collected)} folds)",
                color=color, marker="o",
            )
            _write_subgroup_csv(
                output_dir / f"{group}.csv", grid, per_fold,
                metric_name="specificity",
            )
            n_folds_max = max(n_folds_max, len(collected))
        ax.set_xlabel("Hours Before Birth", fontsize=13)
        ax.set_ylabel("Specificity (Correctly Identified as Healthy)", fontsize=13)
        ax.set_title(
            f"Healthy BG×CS Combinations - {metric_type.replace('_', ' ').title()}",
            fontsize=14, fontweight="bold",
        )
        ax.legend(fontsize=10, loc="best", ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.invert_xaxis()
        fig.tight_layout()
        _add_minmax_band_note(fig, n_folds_max)
        fig.savefig(output_dir / "healthy_bg_cs_combinations.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _aggregate_binary_subgroup_curves(
    fold_dirs: List[Path],
    agg_root: Path,
) -> Dict[str, Dict[str, int]]:
    """Cross-fold binary subgroup analysis.

    Reads each fold's
    ``binary_head/subgroups_vs_time/<metric_type>.csv`` and dispatches to
    four sub-helpers (diagnosis comparison, CS stratification, BG
    stratification, healthy subgroups) — each mirroring its per-fold
    counterpart in ``clinical_metrics_utils.py``.

    Outputs per metric_type land under
    ``aggregated_plots/binary_head/subgroups_vs_time/<metric_type>/``.

    Returns:
        ``{metric_type: {subgroup_name: n_folds}}`` — useful as a
        bookkeeping summary in the run-level JSON.
    """
    summary: Dict[str, Dict[str, int]] = {}
    for metric_type in METRIC_TYPES:
        per_fold_by_subgroup = _read_binary_subgroup_long(fold_dirs, metric_type)
        if not per_fold_by_subgroup:
            continue
        out_dir = agg_root / "binary_head" / "subgroups_vs_time" / metric_type
        out_dir.mkdir(parents=True, exist_ok=True)
        _plot_agg_diagnosis_comparison(per_fold_by_subgroup, out_dir, metric_type)
        _plot_agg_cs_stratification(per_fold_by_subgroup, out_dir, metric_type)
        _plot_agg_bg_stratification(per_fold_by_subgroup, out_dir, metric_type)
        _plot_agg_healthy_subgroups(per_fold_by_subgroup, out_dir, metric_type)
        summary[metric_type] = {
            sg: len(frames) for sg, frames in per_fold_by_subgroup.items()
        }
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

    Cross-fold plots are styled to match their per-fold counterparts
    in :mod:`clinical_metrics_utils` and :mod:`evaluate_3class_metrics`,
    with one addition: a min/max fill_between band across folds plus a
    figure-level note ``"Shaded region: min/max across N folds"``.
    Decision-time vertical reference lines are intentionally **not**
    drawn on aggregated plots — the per-fold figures remain the
    operating-point reference.

    Args:
        run_dir: Run directory holding ``fold_*/`` subdirectories.
        fold_ids: Optional subset; defaults to every ``fold_*`` present.
        decision_time_hours: Auto-resolved from per-fold ``thresholds.json``
            or the run-level config, kept here only so the value can be
            recorded in ``aggregated_results.json`` under
            ``decision_time_hours_input`` for traceability. **Not** drawn
            on any aggregated plot in v2; pass-through is preserved for
            backward-compat with callers that still supply it.

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

    # Cross-fold aggregations driven by per-fold long-format CSVs. Each
    # leaf helper produces plots styled to match its per-fold counterpart
    # plus min/max bands across folds. None of them draws the
    # decision-time vertical line — the orchestrator-level
    # ``decision_time_hours`` argument is preserved for backward compat
    # with callers (kfold_trainer.py / evaluate_kfold.py) but is no
    # longer threaded into individual plotters; the aggregated plots
    # intentionally exclude that annotation per the v2 design.
    perclass_summary = _aggregate_perclass_metric_curves(fold_dirs, agg_root)
    binary_by_class_summary = _aggregate_binary_by_underlying_class_curves(
        fold_dirs, agg_root,
    )
    perclass_subgroup_summary = _aggregate_perclass_subgroup_metric_curves(
        fold_dirs, agg_root,
    )
    perclass_auroc_summary = _aggregate_perclass_auroc_curves(fold_dirs, agg_root)
    binary_metrics_summary = _aggregate_binary_metrics_vs_time_curves(
        fold_dirs, agg_root,
    )
    binary_subgroup_summary = _aggregate_binary_subgroup_curves(fold_dirs, agg_root)

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
    summary["binary_metrics_vs_time"] = binary_metrics_summary
    summary["binary_subgroup_metrics"] = binary_subgroup_summary
    if decision_time_hours is not None:
        # Recorded for traceability only — the value is *not* drawn on
        # any aggregated plot.
        summary["decision_time_hours_input"] = float(decision_time_hours)

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
            "Recorded in aggregated_results.json as "
            "``decision_time_hours_input`` for traceability only — "
            "aggregated plots no longer draw the dashed vertical line "
            "(per-fold figures remain the operating-point reference). "
            "Defaults to the value persisted in per-fold thresholds.json "
            "or the run-level config."
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
