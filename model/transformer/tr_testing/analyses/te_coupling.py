"""TE / Coupling analysis (Category 3).

Analyses specific to the transfer entropy latent and the UP->FHR coupling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..base import TransformerTestRunner
from ..collectors import collect_te_latent_data


def run_te_coupling_analysis(
    runner: TransformerTestRunner,
    class_loaders: Dict[str, Any],
    output_dir: Path,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Run TE coupling analysis for all classes.

    Collects anchor-level and segment-level TE data, generates 9 figure
    types, and saves CSVs.

    Args:
        runner: TransformerTestRunner instance.
        class_loaders: Dict mapping class names to DataLoaders.
        output_dir: Output directory for plots and data.
        max_samples: Maximum samples per class.

    Returns:
        Summary dict with TE DataFrames and figure paths.
    """
    from ..visualizers import (
        plot_kl_distributions,
        plot_kl_per_dimension,
        plot_kl_vs_anchor,
        plot_posterior_vs_prior,
        plot_te_residual_analysis,
        plot_te_latent_projection,
        plot_posterior_variance,
        plot_te_correlation_matrix,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_anchor = []
    all_segment = []
    for class_name, loader in class_loaders.items():
        logger.info(f"  Collecting TE latent data for {class_name}...")
        anchor_df, segment_df = collect_te_latent_data(
            runner, loader, class_name, max_samples=max_samples
        )
        all_anchor.append(anchor_df)
        all_segment.append(segment_df)

    anchor_df = pd.concat(all_anchor, ignore_index=True)
    segment_df = pd.concat(all_segment, ignore_index=True)

    # Save CSVs
    anchor_df.to_csv(output_dir / "te_anchor_data.csv", index=False)
    segment_df.to_csv(output_dir / "te_segment_data.csv", index=False)

    # Generate plots
    plots = {}

    def _try_plot(name, fn, *args, **kwargs):
        try:
            path = fn(*args, **kwargs)
            plots[name] = str(path)
        except Exception as e:
            logger.warning(f"Plot {name} failed: {e}")

    _try_plot("kl_distributions",
              plot_kl_distributions, segment_df, output_dir)
    _try_plot("kl_per_dimension",
              plot_kl_per_dimension, segment_df, output_dir)
    _try_plot("kl_vs_anchor",
              plot_kl_vs_anchor, anchor_df, output_dir)
    _try_plot("posterior_vs_prior",
              plot_posterior_vs_prior, anchor_df, output_dir)
    _try_plot("te_residual",
              plot_te_residual_analysis, segment_df, output_dir)
    _try_plot("posterior_variance",
              plot_posterior_variance, segment_df, output_dir)
    _try_plot("te_correlation",
              plot_te_correlation_matrix, anchor_df, output_dir)

    # TE latent projection (PCA of mu_post means)
    d_z = runner.config.d_z
    mu_cols = [f"mu_post_mean_{d}" for d in range(d_z)]
    if all(c in segment_df.columns for c in mu_cols):
        te_data = segment_df[mu_cols].values
        labels = segment_df["class_label"].values
        _try_plot("te_latent_pca",
                  plot_te_latent_projection, te_data, labels,
                  output_dir, method="pca")
        _try_plot("te_latent_umap",
                  plot_te_latent_projection, te_data, labels,
                  output_dir, method="umap")

    # Summary statistics
    summary = {"plots": plots}
    for cls in segment_df["class_label"].unique():
        cls_df = segment_df[segment_df["class_label"] == cls]
        summary[f"{cls}_kl_mean"] = float(cls_df["kl_mean"].mean())
        summary[f"{cls}_kl_std"] = float(cls_df["kl_mean"].std())
        for h in runner.config.horizons:
            col = f"residual_norm_mean_h{h}"
            if col in cls_df.columns:
                summary[f"{cls}_residual_h{h}_mean"] = float(
                    cls_df[col].mean()
                )

    logger.info(
        f"TE coupling analysis: {len(anchor_df)} anchor rows, "
        f"{len(segment_df)} segment rows, {len(plots)} plots"
    )
    return summary
