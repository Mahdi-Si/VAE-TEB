"""PCA-based analysis of the per-dim KL trajectory.

Fits a PCA on the closed-form per-time per-dim KL contributions
(``0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p)^2)/var_p - 1)``)
and exposes the top components both as per-sample summaries and as
per-time trajectories. The resulting CSVs and plots make it easy to:

- visualise how much of the latent KL signal the top three components
  capture (scree plot),
- inspect class separation in the (PC1, PC2) plane,
- compare per-class mean PC trajectories in a single overlay.

The same PCA fit is reused inside the TE_Calculated comparison pipeline
(via :func:`pca_trajectory` in ``te_kld_analysis``) so the empirical-vs-
model comparison can score not only the original ``kld`` scalar but also
the projected components.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.collectors import (
    collect_kld_trajectory,
    collect_metrics,
)
from model.vae_teb_prediction.testing.visualizers import (
    COLOR_BLUE,
    COLOR_ORANGE,
    COLOR_VERMILLION,
    _style_axes,
)

CLASS_NAMES = {1: "HEALTHY", 2: "ACIDOSIS", 3: "HIE"}
CLASS_COLORS = {1: COLOR_BLUE, 2: COLOR_ORANGE, 3: COLOR_VERMILLION}


def _load_pca_artifacts(pca_dir: Path) -> Optional[Dict[str, Any]]:
    """Load the persisted PCA artifacts written by :func:`collect_metrics`.

    Args:
        pca_dir: Directory containing ``ev_ratio.json`` and ``components.npy``.

    Returns:
        Dict with the parsed artifacts, or None if missing.
    """
    ev_path = pca_dir / "ev_ratio.json"
    comp_path = pca_dir / "components.npy"
    mean_path = pca_dir / "mean.npy"
    if not ev_path.exists() or not comp_path.exists():
        return None
    with open(ev_path) as fh:
        ev = json.load(fh)
    components = np.load(comp_path)
    mean = np.load(mean_path) if mean_path.exists() else None
    return {"ev": ev, "components": components, "mean": mean}


def _build_pca_model(artifacts: Dict[str, Any]):
    """Reconstruct a sklearn PCA-like object from persisted artifacts.

    Avoids re-fitting; reuses the components / mean already computed in
    :func:`collect_metrics`.
    """
    from sklearn.decomposition import PCA

    components = np.asarray(artifacts["components"], dtype=np.float32)
    n_components, d_z = components.shape
    pca = PCA(n_components=n_components)
    pca.components_ = components
    pca.mean_ = (
        np.asarray(artifacts["mean"], dtype=np.float32)
        if artifacts.get("mean") is not None
        else np.zeros(d_z, dtype=np.float32)
    )
    pca.n_components_ = n_components
    pca.n_features_in_ = d_z
    pca.explained_variance_ratio_ = np.asarray(
        artifacts["ev"]["explained_variance_ratio"], dtype=np.float32
    )
    pca.explained_variance_ = pca.explained_variance_ratio_.copy()
    return pca


def _plot_scree(ev_ratio: np.ndarray, out_path: Path) -> None:
    """Bar + cumulative line plot of the explained-variance ratio."""
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    n = ev_ratio.size
    xs = np.arange(1, n + 1)
    ax.bar(xs, ev_ratio, color=COLOR_BLUE, alpha=0.85, label="per-component")
    ax2 = ax.twinx()
    ax2.plot(xs, np.cumsum(ev_ratio), "-o", color=COLOR_VERMILLION, label="cumulative")
    ax.set_xlabel("PC index")
    ax.set_ylabel("explained variance ratio")
    ax2.set_ylabel("cumulative")
    ax.set_xticks(xs)
    _style_axes(ax)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_pc12_scatter_by_class(df: pd.DataFrame, out_path: Path) -> None:
    """Per-sample PC1 vs PC2 scatter coloured by outcome class."""
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    for label_id, name in CLASS_NAMES.items():
        sub = df[df["label"] == label_id]
        if sub.empty:
            continue
        ax.scatter(
            sub["kld_pc1"],
            sub["kld_pc2"],
            s=14,
            alpha=0.6,
            color=CLASS_COLORS[label_id],
            label=f"{name} (n={len(sub)})",
        )
    ax.axhline(0, color="#888888", lw=0.4, zorder=0)
    ax.axvline(0, color="#888888", lw=0.4, zorder=0)
    ax.set_xlabel("PC1 (per-sample mean)")
    ax.set_ylabel("PC2 (per-sample mean)")
    ax.set_title("Per-sample KL PCA, top 2 components")
    ax.legend(loc="best", frameon=True)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_pc_trajectories_overlay(
    traj_df: pd.DataFrame,
    n_pcs: int,
    out_path: Path,
) -> None:
    """Per-class mean trajectories of each top component vs timestep."""
    pc_cols = [f"kld_pc{k + 1}_t" for k in range(n_pcs) if f"kld_pc{k + 1}_t" in traj_df.columns]
    if not pc_cols:
        return
    fig, axes = plt.subplots(len(pc_cols), 1, figsize=(7.2, 2.4 * len(pc_cols)), sharex=True)
    if len(pc_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, pc_cols):
        for label_id, name in CLASS_NAMES.items():
            sub = traj_df[traj_df["label"] == label_id]
            if sub.empty:
                continue
            grouped = sub.groupby("timestep")[col]
            mean = grouped.mean()
            sem = grouped.sem()
            ax.plot(mean.index, mean.values, color=CLASS_COLORS[label_id], label=name)
            ax.fill_between(
                mean.index,
                (mean - sem).values,
                (mean + sem).values,
                color=CLASS_COLORS[label_id],
                alpha=0.18,
                lw=0,
            )
        ax.set_ylabel(col)
        _style_axes(ax)
    axes[-1].set_xlabel("timestep")
    axes[0].legend(loc="best", frameon=True)
    fig.suptitle("PCA components of per-time per-dim KL by class")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_kld_pca_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 500,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fit / load PCA on per-dim KL and emit summary plots.

    Reuses the PCA artifacts written by :func:`collect_metrics` when
    available; otherwise re-runs collection to populate them. Always
    re-collects the per-time trajectory through :func:`collect_kld_trajectory`
    with the fitted PCA so the per-class overlay can be drawn.

    Args:
        runner: Loaded :class:`TestRunner`.
        loader: Standard PyTorch DataLoader (segment-level).
        max_samples: Maximum samples to process when collection is needed.
        output_dir: Optional override directory. Defaults to
            ``runner.ensure_dir("kld_pca")``.

    Returns:
        Dict with ``n_samples``, ``n_components``,
        ``explained_variance_ratio``, and the path of the PCA artifacts.
    """
    if max_samples <= 0:
        logger.info("kld_pca: skipped (max_samples <= 0)")
        return {}

    if output_dir is None:
        output_dir = runner.ensure_dir("kld_pca")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pca_artifact_dir = Path(runner.output_dir) / "pca_kld"
    artifacts = _load_pca_artifacts(pca_artifact_dir)

    metrics_df: Optional[pd.DataFrame] = None
    if artifacts is None:
        # Force a fresh collect_metrics pass to populate the artifacts.
        logger.info("kld_pca: PCA artifacts not found, running collect_metrics ...")
        metrics_df = collect_metrics(
            runner,
            loader,
            max_samples=max_samples,
            pca_components=3,
            pca_output_dir=pca_artifact_dir,
        )
        artifacts = _load_pca_artifacts(pca_artifact_dir)

    if artifacts is None:
        logger.warning("kld_pca: PCA fit failed, no artifacts produced.")
        return {"n_samples": 0}

    ev_ratio = np.asarray(
        artifacts["ev"]["explained_variance_ratio"], dtype=np.float32
    )
    n_components = int(artifacts["ev"]["n_components"])

    # Scree plot.
    _plot_scree(ev_ratio, output_dir / "scree.pdf")

    # Per-sample PC1 vs PC2 scatter -- prefer the histogram CSV if present.
    if metrics_df is None:
        hist_csv = Path(runner.output_dir) / "histograms" / "histogram_metrics.csv"
        if hist_csv.exists():
            metrics_df = pd.read_csv(hist_csv)
    if metrics_df is not None and {"kld_pc1", "kld_pc2"}.issubset(metrics_df.columns):
        _plot_pc12_scatter_by_class(metrics_df, output_dir / "pc12_scatter_by_class.pdf")

    # Per-time per-class trajectories.
    pca_model = _build_pca_model(artifacts)
    traj_df = collect_kld_trajectory(
        runner, loader, max_samples=max_samples, pca_model=pca_model
    )
    if not traj_df.empty:
        traj_csv = output_dir / "kld_pc_trajectory.csv"
        traj_df.to_csv(traj_csv, index=False)
        _plot_pc_trajectories_overlay(
            traj_df, n_components, output_dir / "pc_trajectories_overlay.pdf"
        )

    summary = {
        "n_samples": int(artifacts["ev"]["n_samples_fitted"]),
        "n_components": n_components,
        "explained_variance_ratio": [float(x) for x in ev_ratio],
        "pca_dir": str(pca_artifact_dir),
    }
    logger.info(
        f"kld_pca: n={summary['n_samples']}, ev_ratio={summary['explained_variance_ratio']}"
    )
    return summary
