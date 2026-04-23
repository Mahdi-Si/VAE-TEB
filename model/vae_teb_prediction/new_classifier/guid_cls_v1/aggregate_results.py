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


def _load_fold_test_csv(fold_dir: Path) -> Optional[pd.DataFrame]:
    csv = fold_dir / "evaluation" / "test_predictions_raw.csv"
    if not csv.exists():
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


def _aggregate_legacy_metric_plots(run_dir: Path, fold_dirs: List[Path]) -> None:
    """Best-effort call into ``new_classifier.evaluate_classifier``'s
    aggregator. Failures are logged but not fatal (the aggregator depends on
    the legacy module that may have other broken pieces).
    """
    try:
        from model.vae_teb_prediction.new_classifier.evaluate_classifier import (  # noqa: WPS433
            generate_aggregated_plots,
        )
    except Exception as exc:
        logger.warning(f"legacy generate_aggregated_plots unavailable: {exc}")
        return

    fold_results: List[Dict[str, Any]] = []
    for fd in fold_dirs:
        rj = fd / "fold_results.json"
        if not rj.exists():
            continue
        try:
            fold_results.append(json.loads(rj.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning(f"could not parse {rj}: {exc}")
            continue
    if not fold_results:
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
) -> Dict[str, Any]:
    """Aggregate fold artefacts and write summary plots / JSON.

    Args:
        run_dir: Run directory holding ``fold_*/`` subdirectories.
        fold_ids: Optional subset; defaults to every ``fold_*`` present.

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

    agg_root = run_dir / "aggregated_plots"
    agg_root.mkdir(parents=True, exist_ok=True)
    diag_root = agg_root / "three_class_diagnostics"
    diag_root.mkdir(parents=True, exist_ok=True)

    per_fold_binary_roc: List[Tuple[int, Dict[str, Any]]] = []
    per_fold_3class_roc: List[Tuple[int, Dict[str, Dict[str, Any]]]] = []
    per_fold_cms: List[np.ndarray] = []
    per_fold_thresholds: List[Tuple[int, Dict[str, Any]]] = []

    for fd in fold_dirs:
        fid = int(fd.name.split("_")[-1])
        eval_dir = fd / "evaluation"

        # Binary ROC: stored as roc_binary_data.csv (fpr, tpr, thresholds).
        # Skip empty / malformed files so a partially-failed fold doesn't
        # poison the cross-fold summary.
        roc_csv = eval_dir / "roc_binary_data.csv"
        if roc_csv.exists():
            try:
                roc_df = pd.read_csv(roc_csv)
                if roc_df.empty or "fpr" not in roc_df.columns or "tpr" not in roc_df.columns:
                    logger.warning(
                        f"fold {fid}: roc_binary_data.csv is empty or missing "
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

        # Threshold info.
        thr_json = eval_dir / "threshold_info.json"
        if thr_json.exists():
            try:
                per_fold_thresholds.append(
                    (fid, json.loads(thr_json.read_text(encoding="utf-8")))
                )
            except Exception as exc:
                logger.warning(f"fold {fid}: failed to read threshold_info.json: {exc}")

    binary_auc_mean = _plot_aggregated_binary_roc(
        per_fold_binary_roc, agg_root / "aggregated_roc_binary.png"
    )
    three_class_auc_means = _plot_aggregated_3class_roc(
        per_fold_3class_roc, agg_root / "aggregated_roc_3class_ovr"
    )
    mean_cm: Optional[np.ndarray] = None
    if per_fold_cms:
        mean_cm = _plot_mean_confusion_matrix(
            per_fold_cms, diag_root / "mean_confusion_matrix_3class.png"
        )

    # Legacy three-metric-type aggregator.
    _aggregate_legacy_metric_plots(run_dir, fold_dirs)

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
    args = parser.parse_args(argv)
    aggregate_results(
        run_dir=Path(args.run_dir).resolve(),
        fold_ids=args.fold_ids,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
