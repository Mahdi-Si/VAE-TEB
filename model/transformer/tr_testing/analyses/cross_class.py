"""Cross-class statistical comparison (Category 6).

Statistical tests, classification experiments, and cross-class histogram
overlays comparing outcome classes. Gracefully handles single-class input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from model.transformer.tr_testing.base import TransformerTestRunner
from model.transformer.tr_testing.collectors import (
    collect_embeddings,
    collect_forecast_metrics,
    collect_loss_components,
    collect_te_latent_data,
)


def _run_statistical_tests(
    metrics_df: pd.DataFrame,
    metric_col: str = "mae",
) -> pd.DataFrame:
    """Run Kruskal-Wallis and pairwise Mann-Whitney U tests.

    Args:
        metrics_df: DataFrame with ``class_label`` and metric columns.
        metric_col: Column to test.

    Returns:
        DataFrame with test results.
    """
    try:
        from scipy.stats import kruskal, mannwhitneyu
    except ImportError:
        logger.warning("scipy not available; skipping statistical tests")
        return pd.DataFrame()

    classes = sorted(metrics_df["class_label"].unique())
    if len(classes) < 2:
        return pd.DataFrame()

    rows = []

    # Kruskal-Wallis across all classes
    groups = [
        metrics_df[metrics_df["class_label"] == c][metric_col].dropna()
        for c in classes
    ]
    groups = [g.values for g in groups if len(g) > 0]
    if len(groups) >= 2:
        stat, p = kruskal(*groups)
        rows.append({
            "test": "kruskal_wallis",
            "metric": metric_col,
            "class_a": "all",
            "class_b": "all",
            "statistic": stat,
            "p_value": p,
        })

    # Pairwise Mann-Whitney
    for i, ca in enumerate(classes):
        for cb in classes[i + 1:]:
            ga = metrics_df[metrics_df["class_label"] == ca][metric_col].dropna()
            gb = metrics_df[metrics_df["class_label"] == cb][metric_col].dropna()
            if len(ga) < 2 or len(gb) < 2:
                continue
            stat, p = mannwhitneyu(ga, gb, alternative="two-sided")
            rows.append({
                "test": "mann_whitney",
                "metric": metric_col,
                "class_a": ca,
                "class_b": cb,
                "statistic": stat,
                "p_value": p,
            })

    return pd.DataFrame(rows)


def _compute_effect_sizes(
    metrics_df: pd.DataFrame,
    metric_col: str = "mae",
) -> pd.DataFrame:
    """Compute Cohen's d for all pairwise class comparisons.

    Args:
        metrics_df: DataFrame with ``class_label`` and metric columns.
        metric_col: Column to compute effect size for.

    Returns:
        DataFrame with effect sizes.
    """
    classes = sorted(metrics_df["class_label"].unique())
    rows = []
    for i, ca in enumerate(classes):
        for cb in classes[i + 1:]:
            ga = metrics_df[metrics_df["class_label"] == ca][metric_col].dropna()
            gb = metrics_df[metrics_df["class_label"] == cb][metric_col].dropna()
            if len(ga) < 2 or len(gb) < 2:
                continue
            ma, mb = ga.mean(), gb.mean()
            sa, sb = ga.std(), gb.std()
            pooled_std = np.sqrt(
                ((len(ga) - 1) * sa**2 + (len(gb) - 1) * sb**2)
                / (len(ga) + len(gb) - 2)
            )
            d = (ma - mb) / max(pooled_std, 1e-10)
            rows.append({
                "metric": metric_col,
                "class_a": ca,
                "class_b": cb,
                "cohens_d": d,
                "mean_a": ma,
                "mean_b": mb,
            })
    return pd.DataFrame(rows)


def _run_embedding_classification(
    embeddings_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
) -> Dict[str, Any]:
    """Run logistic regression classification on embeddings.

    Args:
        embeddings_dict: Dict mapping component names to arrays.
        labels: Class label array.

    Returns:
        Dict with ROC curves and confusion matrices per component.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            confusion_matrix,
            roc_auc_score,
            roc_curve,
        )
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import LabelEncoder, StandardScaler
    except ImportError:
        logger.warning("sklearn not available; skipping classification")
        return {}

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {}

    le = LabelEncoder()
    y = le.fit_transform(labels)
    results = {}

    for name, X in embeddings_dict.items():
        if X.shape[0] < 10:
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Cross-validated predictions
        n_splits = min(5, min(np.bincount(y)))
        if n_splits < 2:
            continue

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=42)
        y_true_all, y_pred_all, y_prob_all = [], [], []

        for train_idx, test_idx in skf.split(X_scaled, y):
            clf = LogisticRegression(max_iter=1000, solver="lbfgs",
                                     multi_class="auto")
            clf.fit(X_scaled[train_idx], y[train_idx])
            y_pred_all.extend(clf.predict(X_scaled[test_idx]))
            y_true_all.extend(y[test_idx])

            if hasattr(clf, "predict_proba"):
                y_prob_all.extend(
                    clf.predict_proba(X_scaled[test_idx]).tolist()
                )

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        cm = confusion_matrix(y_true_all, y_pred_all)
        acc = (y_true_all == y_pred_all).mean()

        comp_result = {
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
            "class_names": le.classes_.tolist(),
        }

        # ROC curves (binary or one-vs-rest)
        if y_prob_all and len(unique_labels) == 2:
            y_prob = np.array(y_prob_all)
            fpr, tpr, _ = roc_curve(y_true_all, y_prob[:, 1])
            auc = roc_auc_score(y_true_all, y_prob[:, 1])
            comp_result["roc"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(auc),
            }

        results[name] = comp_result

    return results


def run_cross_class_analysis(
    runner: TransformerTestRunner,
    class_loaders: Dict[str, Any],
    output_dir: Path,
    max_samples: Optional[int] = None,
    forecast_results: Optional[Dict] = None,
    te_results: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run cross-class statistical comparison.

    Gracefully handles single-class input by skipping statistical tests.

    Args:
        runner: TransformerTestRunner instance.
        class_loaders: Dict mapping class names to DataLoaders.
        output_dir: Output directory.
        max_samples: Maximum samples per class.
        forecast_results: Pre-computed forecast results (reuse CSVs).
        te_results: Pre-computed TE results (reuse CSVs).

    Returns:
        Summary dict.
    """
    from model.transformer.tr_testing.visualizers import (
        plot_class_mae_comparison,
        plot_confusion_matrices,
        plot_cross_class_kl_histograms,
        plot_cross_class_loss_histograms,
        plot_cross_class_mae_histograms,
        plot_cross_class_mse_histograms,
        plot_cross_class_snr_histograms,
        plot_cross_class_vaf_histograms,
        plot_effect_size_heatmap,
        plot_metric_summary_table,
        plot_roc_curves,
        plot_significance_heatmap,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = list(class_loaders.keys())
    n_classes = len(class_names)

    # Load or collect metrics
    forecast_csv = runner.output_dir / "forecasting" / "forecast_metrics.csv"
    loss_csv = runner.output_dir / "forecasting" / "loss_components.csv"
    te_seg_csv = runner.output_dir / "te_coupling" / "te_segment_data.csv"

    if forecast_csv.exists():
        metrics_df = pd.read_csv(forecast_csv)
    else:
        logger.info("  Collecting forecast metrics for cross-class...")
        all_m = []
        for cn, loader in class_loaders.items():
            all_m.append(collect_forecast_metrics(
                runner, loader, cn, max_samples=max_samples
            ))
        metrics_df = pd.concat(all_m, ignore_index=True)

    if loss_csv.exists():
        loss_df = pd.read_csv(loss_csv)
    else:
        logger.info("  Collecting loss components for cross-class...")
        all_l = []
        for cn, loader in class_loaders.items():
            all_l.append(collect_loss_components(
                runner, loader, cn, max_samples=max_samples
            ))
        loss_df = pd.concat(all_l, ignore_index=True)

    if te_seg_csv.exists():
        te_seg_df = pd.read_csv(te_seg_csv)
    else:
        logger.info("  Collecting TE data for cross-class...")
        all_t_seg = []
        for cn, loader in class_loaders.items():
            _, seg = collect_te_latent_data(
                runner, loader, cn, max_samples=max_samples
            )
            all_t_seg.append(seg)
        te_seg_df = pd.concat(all_t_seg, ignore_index=True)

    plots = {}

    def _try_plot(name, fn, *args, **kwargs):
        try:
            path = fn(*args, **kwargs)
            plots[name] = str(path)
        except Exception as e:
            logger.warning(f"Plot {name} failed: {e}")

    # --- Always generated (even with 1 class) ---
    _try_plot("cross_class_mae_histograms",
              plot_cross_class_mae_histograms, metrics_df, output_dir)
    _try_plot("cross_class_vaf_histograms",
              plot_cross_class_vaf_histograms, metrics_df, output_dir)
    _try_plot("cross_class_snr_histograms",
              plot_cross_class_snr_histograms, metrics_df, output_dir)
    _try_plot("cross_class_mse_histograms",
              plot_cross_class_mse_histograms, metrics_df, output_dir)
    _try_plot("cross_class_loss_histograms",
              plot_cross_class_loss_histograms, loss_df, output_dir)
    _try_plot("cross_class_kl_histograms",
              plot_cross_class_kl_histograms, te_seg_df, output_dir)
    # Build summary table dict: {class: {metric: (mean, std)}}
    summary_table = {}
    for cls in metrics_df["class_label"].unique():
        cls_df = metrics_df[metrics_df["class_label"] == cls]
        cls_summary = {}
        for metric in ("mae", "mse", "vaf", "snr"):
            vals = cls_df[metric].dropna()
            if len(vals) > 0:
                cls_summary[metric] = (float(vals.mean()), float(vals.std()))
        summary_table[cls] = cls_summary
    _try_plot("metric_summary_table",
              plot_metric_summary_table, summary_table, output_dir)
    _try_plot("class_mae_comparison",
              plot_class_mae_comparison, metrics_df, output_dir)

    # --- Requires >= 2 classes ---
    summary = {"plots": plots, "n_classes": n_classes}

    if n_classes >= 2:
        # Statistical tests
        all_tests = []
        for metric in ("mae", "mse", "vaf", "snr"):
            tests = _run_statistical_tests(metrics_df, metric)
            all_tests.append(tests)

        test_df = pd.concat(all_tests, ignore_index=True)
        test_df.to_csv(output_dir / "statistical_tests.csv", index=False)

        _try_plot("significance_heatmap",
                  plot_significance_heatmap, test_df, output_dir)

        # Effect sizes
        all_effects = []
        for metric in ("mae", "mse", "vaf", "snr"):
            effects = _compute_effect_sizes(metrics_df, metric)
            all_effects.append(effects)
        effect_df = pd.concat(all_effects, ignore_index=True)

        _try_plot("effect_size_heatmap",
                  plot_effect_size_heatmap, effect_df, output_dir)

        # Classification on embeddings
        emb_data = {}
        emb_file = runner.output_dir / "representation" / "embeddings.npz"
        meta_file = runner.output_dir / "representation" / "embedding_metadata.csv"
        if emb_file.exists() and meta_file.exists():
            data = np.load(emb_file)
            emb_data = {k: data[k] for k in data.files}
            labels = pd.read_csv(meta_file)["class_label"].values
        else:
            # Collect fresh
            all_emb = {"e_win": [], "e_F": [], "e_FU": [], "e_TE": []}
            all_labels = []
            for cn, loader in class_loaders.items():
                emb = collect_embeddings(
                    runner, loader, cn, max_samples=max_samples
                )
                for k in all_emb:
                    all_emb[k].append(emb[k])
                all_labels.extend(
                    emb["metadata"]["class_label"].tolist()
                )
            emb_data = {k: np.concatenate(v) for k, v in all_emb.items()}
            labels = np.array(all_labels)

        if emb_data:
            clf_results = _run_embedding_classification(emb_data, labels)
            if clf_results:
                import json
                with open(output_dir / "classification_results.json", "w") as f:
                    json.dump(clf_results, f, indent=2, default=str)

                _try_plot("roc_curves",
                          plot_roc_curves, clf_results, output_dir)
                _try_plot("confusion_matrices",
                          plot_confusion_matrices, clf_results, output_dir)
                summary["classification"] = {
                    k: v.get("accuracy") for k, v in clf_results.items()
                }
    else:
        logger.warning(
            "Only 1 class provided — skipping statistical tests, "
            "effect sizes, and classification"
        )

    logger.info(
        f"Cross-class analysis: {n_classes} classes, {len(plots)} plots"
    )
    return summary
