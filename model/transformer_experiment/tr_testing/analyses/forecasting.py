"""Forecasting performance analysis (Category 2).

Aggregate forecasting metrics across the full test set, comparing all
3 heads, 3 horizons, and all classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

from model.transformer.tr_testing.base import TransformerTestRunner
from model.transformer.tr_testing.collectors import collect_forecast_metrics, collect_loss_components


def run_forecasting_analysis(
    runner: TransformerTestRunner,
    class_loaders: Dict[str, Any],
    output_dir: Path,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Run forecasting performance analysis for all classes.

    Collects MAE/MSE/VAF/SNR/Huber metrics and loss components, generates
    10 figure types, and saves CSVs.

    Args:
        runner: TransformerTestRunner instance.
        class_loaders: Dict mapping class names to DataLoaders.
        output_dir: Output directory for plots and data.
        max_samples: Maximum samples per class.

    Returns:
        Summary dict with metrics DataFrames and figure paths.
    """
    from model.transformer.tr_testing.visualizers import (
        plot_mae_histograms,
        plot_mae_boxplots_by_class,
        plot_head_comparison_scatter,
        plot_improvement_distribution,
        plot_error_vs_time,
        plot_loss_decomposition,
        plot_head_radar,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics from all classes
    all_metrics = []
    all_losses = []
    for class_name, loader in class_loaders.items():
        logger.info(f"  Collecting forecast metrics for {class_name}...")
        metrics_df = collect_forecast_metrics(
            runner, loader, class_name, max_samples=max_samples
        )
        all_metrics.append(metrics_df)

        logger.info(f"  Collecting loss components for {class_name}...")
        loss_df = collect_loss_components(
            runner, loader, class_name, max_samples=max_samples
        )
        all_losses.append(loss_df)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    loss_df = pd.concat(all_losses, ignore_index=True)

    # Save CSVs
    metrics_df.to_csv(output_dir / "forecast_metrics.csv", index=False)
    loss_df.to_csv(output_dir / "loss_components.csv", index=False)

    # Generate plots with error isolation
    plots = {}

    def _try_plot(name, fn, *args, **kwargs):
        try:
            path = fn(*args, **kwargs)
            plots[name] = str(path)
        except Exception as e:
            logger.warning(f"Plot {name} failed: {e}")

    _try_plot("mae_histograms",
              plot_mae_histograms, metrics_df, output_dir)
    _try_plot("mae_boxplots",
              plot_mae_boxplots_by_class, metrics_df, output_dir)
    _try_plot("fused_vs_self",
              plot_head_comparison_scatter, metrics_df, output_dir,
              "self", "fused")
    _try_plot("te_vs_self",
              plot_head_comparison_scatter, metrics_df, output_dir,
              "self", "te")
    _try_plot("improvement",
              plot_improvement_distribution, metrics_df, output_dir)
    _try_plot("error_vs_time",
              plot_error_vs_time, metrics_df, output_dir)
    _try_plot("loss_decomposition",
              plot_loss_decomposition, loss_df, output_dir)
    _try_plot("head_radar",
              plot_head_radar, metrics_df, output_dir)

    # Compute summary statistics
    summary = {"plots": plots}
    for head in ("self", "fused", "te"):
        head_df = metrics_df[metrics_df["head"] == head]
        for h in runner.config.horizons:
            h_df = head_df[head_df["horizon"] == h]
            key = f"{head}_h{h}"
            summary[f"{key}_mae_mean"] = float(h_df["mae"].mean())
            summary[f"{key}_mae_std"] = float(h_df["mae"].std())

    import json
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(
        f"Forecasting analysis: {len(metrics_df)} metric rows, "
        f"{len(plots)} plots"
    )
    return summary
