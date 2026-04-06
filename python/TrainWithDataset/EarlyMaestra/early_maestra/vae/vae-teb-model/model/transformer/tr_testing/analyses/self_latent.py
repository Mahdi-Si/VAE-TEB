"""Self-latent (z_self) quality analysis (Category 3b, v2).

Analyses specific to the intrinsic FHR latent z_self: utilization,
class separability, dimensional structure, and posterior collapse detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from model.transformer.tr_testing.base import TransformerTestRunner
from model.transformer.tr_testing.collectors import collect_self_latent_data


def run_self_latent_analysis(
    runner: TransformerTestRunner,
    class_loaders: Dict[str, Any],
    output_dir: Path,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Run z_self latent quality analysis for all classes.

    Collects anchor-level and segment-level self-latent data, generates
    diagnostic plots, and reports utilization metrics.  Reuses TE coupling
    visualizers by renaming columns to match expected format.

    Args:
        runner: TransformerTestRunner instance.
        class_loaders: Dict mapping class names to DataLoaders.
        output_dir: Output directory for plots and data.
        max_samples: Maximum samples per class.

    Returns:
        Summary dict with DataFrames, figure paths, and utilization stats.
    """
    from model.transformer.tr_testing.visualizers import (
        plot_kl_distributions,
        plot_kl_per_dimension,
        plot_te_latent_projection,
        plot_posterior_variance,
        plot_te_correlation_matrix,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_anchor = []
    all_segment = []
    for class_name, loader in class_loaders.items():
        logger.info(f"  Collecting z_self latent data for {class_name}...")
        anchor_df, segment_df = collect_self_latent_data(
            runner, loader, class_name, max_samples=max_samples
        )
        all_anchor.append(anchor_df)
        all_segment.append(segment_df)

    anchor_df = pd.concat(all_anchor, ignore_index=True)
    segment_df = pd.concat(all_segment, ignore_index=True)

    # Save CSVs
    anchor_df.to_csv(output_dir / "self_latent_anchor_data.csv", index=False)
    segment_df.to_csv(output_dir / "self_latent_segment_data.csv", index=False)

    # --- Prepare adapted DataFrames for TE visualizers ---
    # The TE visualizers expect mu_post_* / logvar_post_* / mu_prior_* columns.
    # Rename mu_self_* -> mu_post_* so visualizers work out of the box.
    anchor_adapted = anchor_df.copy()
    segment_adapted = segment_df.copy()
    d_z_self = runner.config.d_z_self

    rename_anchor = {}
    rename_segment = {}
    for d in range(d_z_self):
        rename_anchor[f"mu_self_{d}"] = f"mu_post_{d}"
        rename_anchor[f"logvar_self_{d}"] = f"logvar_post_{d}"
        rename_segment[f"mu_self_mean_{d}"] = f"mu_post_mean_{d}"
        rename_segment[f"mu_self_max_{d}"] = f"mu_post_max_{d}"
        rename_segment[f"mu_self_min_{d}"] = f"mu_post_min_{d}"
        rename_segment[f"logvar_self_mean_{d}"] = f"logvar_post_mean_{d}"
    anchor_adapted = anchor_adapted.rename(columns=rename_anchor)
    segment_adapted = segment_adapted.rename(columns=rename_segment)

    # Generate plots
    plots = {}

    def _try_plot(name, fn, *args, **kwargs):
        try:
            path = fn(*args, **kwargs)
            plots[name] = str(path)
        except Exception as e:
            logger.warning(f"Plot {name} failed: {e}")

    _try_plot("self_kl_distributions",
              plot_kl_distributions, segment_adapted, output_dir)
    _try_plot("self_kl_per_dimension",
              plot_kl_per_dimension, segment_adapted, output_dir)
    _try_plot("self_posterior_variance",
              plot_posterior_variance, segment_adapted, output_dir)

    # Correlation matrix of z_self dimensions
    _try_plot("self_correlation",
              plot_te_correlation_matrix, anchor_adapted, output_dir)

    # Latent projection (PCA/UMAP of mu_self means)
    mu_cols = [f"mu_post_mean_{d}" for d in range(d_z_self)]
    if all(c in segment_adapted.columns for c in mu_cols):
        self_data = segment_adapted[mu_cols].values
        labels = segment_adapted["class_label"].values
        _try_plot("self_latent_pca",
                  plot_te_latent_projection, self_data, labels,
                  output_dir, method="pca")
        if len(labels) > 30:
            _try_plot("self_latent_umap",
                      plot_te_latent_projection, self_data, labels,
                      output_dir, method="umap")

    # Summary statistics
    summary = {"plots": plots}

    if "utilization" in segment_df.columns:
        mean_util = segment_df["utilization"].mean()
        summary["mean_utilization"] = float(mean_util)
        summary["d_z_self"] = d_z_self
        summary["free_bits"] = runner.config.free_bits

        for cls in segment_df["class_label"].unique():
            cls_df = segment_df[segment_df["class_label"] == cls]
            summary[f"{cls}_utilization_mean"] = float(
                cls_df["utilization"].mean()
            )
            summary[f"{cls}_active_dims_mean"] = float(
                cls_df["active_dims"].mean()
            )
            summary[f"{cls}_kl_mean"] = float(cls_df["kl_mean"].mean())

    logger.info(
        f"Self-latent analysis: {len(anchor_df)} anchor rows, "
        f"{len(segment_df)} segment rows, {len(plots)} plots"
    )
    if "mean_utilization" in summary:
        logger.info(
            f"  Utilization: {summary['mean_utilization']:.1%} "
            f"({summary.get('mean_utilization', 0) * d_z_self:.0f}/{d_z_self} dims active)"
        )
    return summary
