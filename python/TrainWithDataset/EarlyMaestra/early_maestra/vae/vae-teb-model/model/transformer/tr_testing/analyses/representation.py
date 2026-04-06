"""Representation and embedding analysis (Category 4).

Analysis of learned representations: encoder states, gate activations,
fusion contributions, and window embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..base import TransformerTestRunner
from ..collectors import collect_embeddings, collect_gate_and_fusion


def _compute_linear_separability(
    embeddings_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute linear separability scores for each embedding component.

    Uses logistic regression with stratified 5-fold cross-validation.

    Args:
        embeddings_dict: Dict mapping component names to arrays.
        labels: Class label array.

    Returns:
        Dict mapping component names to mean accuracy scores.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.warning("sklearn not available; skipping separability")
        return {}

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {k: float("nan") for k in embeddings_dict}

    scores = {}
    for name, X in embeddings_dict.items():
        if X.shape[0] < 10:
            scores[name] = float("nan")
            continue
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=1000, solver="lbfgs",
                                 multi_class="auto")
        cv = min(5, min(np.bincount(
            np.searchsorted(unique_labels, labels)
        )))
        if cv < 2:
            scores[name] = float("nan")
            continue
        acc = cross_val_score(clf, X_scaled, labels, cv=cv,
                              scoring="accuracy")
        scores[name] = float(acc.mean())
    return scores


def _compute_clustering_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute clustering quality metrics.

    Args:
        embeddings: ``(N, D)`` array.
        labels: Class label array.

    Returns:
        Dict with silhouette, davies_bouldin, calinski_harabasz scores.
    """
    try:
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )
    except ImportError:
        return {}

    unique = np.unique(labels)
    if len(unique) < 2 or embeddings.shape[0] < 10:
        return {}

    label_idx = np.searchsorted(unique, labels)
    n_samples = min(5000, embeddings.shape[0])
    if embeddings.shape[0] > n_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(embeddings.shape[0], n_samples, replace=False)
        embeddings = embeddings[idx]
        label_idx = label_idx[idx]

    return {
        "silhouette": float(silhouette_score(embeddings, label_idx)),
        "davies_bouldin": float(davies_bouldin_score(embeddings, label_idx)),
        "calinski_harabasz": float(
            calinski_harabasz_score(embeddings, label_idx)
        ),
    }


def run_representation_analysis(
    runner: TransformerTestRunner,
    class_loaders: Dict[str, Any],
    output_dir: Path,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Run representation and embedding analysis for all classes.

    Args:
        runner: TransformerTestRunner instance.
        class_loaders: Dict mapping class names to DataLoaders.
        output_dir: Output directory.
        max_samples: Maximum samples per class.

    Returns:
        Summary dict.
    """
    from ..visualizers import (
        plot_embedding_projection,
        plot_embedding_norms,
        plot_fusion_distribution,
        plot_gate_distribution,
        plot_gate_temporal_profile,
        plot_linear_separability,
        plot_clustering_quality,
        plot_variance_spectrum,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect embeddings from all classes
    all_embeddings = {"e_win": [], "e_F": [], "e_FU": [], "e_TE": []}
    all_meta = []
    for class_name, loader in class_loaders.items():
        logger.info(f"  Collecting embeddings for {class_name}...")
        emb = collect_embeddings(
            runner, loader, class_name, max_samples=max_samples
        )
        for key in all_embeddings:
            all_embeddings[key].append(emb[key])
        all_meta.append(emb["metadata"])

    # Merge
    embeddings = {
        k: np.concatenate(v, axis=0) for k, v in all_embeddings.items()
    }
    meta_df = pd.concat(all_meta, ignore_index=True)
    labels = meta_df["class_label"].values

    # Save
    np.savez_compressed(output_dir / "embeddings.npz", **embeddings)
    meta_df.to_csv(output_dir / "embedding_metadata.csv", index=False)

    # Collect gate and fusion data
    all_gate = []
    for class_name, loader in class_loaders.items():
        logger.info(f"  Collecting gate/fusion for {class_name}...")
        gf_df = collect_gate_and_fusion(
            runner, loader, class_name, max_samples=max_samples
        )
        all_gate.append(gf_df)
    gate_df = pd.concat(all_gate, ignore_index=True)
    gate_df.to_csv(output_dir / "gate_fusion_data.csv", index=False)

    # Generate plots
    plots = {}

    def _try_plot(name, fn, *args, **kwargs):
        try:
            path = fn(*args, **kwargs)
            plots[name] = str(path)
        except Exception as e:
            logger.warning(f"Plot {name} failed: {e}")

    # Embedding projections
    for component in ("e_win", "e_F", "e_FU", "e_TE"):
        _try_plot(f"{component}_pca",
                  plot_embedding_projection, embeddings[component],
                  labels, output_dir, method="pca", component=component)

    _try_plot("e_win_umap",
              plot_embedding_projection, embeddings["e_win"],
              labels, output_dir, method="umap", component="e_win")
    _try_plot("e_win_tsne",
              plot_embedding_projection, embeddings["e_win"],
              labels, output_dir, method="tsne", component="e_win")

    _try_plot("embedding_norms",
              plot_embedding_norms, embeddings, labels, output_dir)
    _try_plot("fusion_distribution",
              plot_fusion_distribution, gate_df, output_dir)
    _try_plot("gate_distribution",
              plot_gate_distribution, gate_df, output_dir)
    _try_plot("gate_temporal_profile",
              plot_gate_temporal_profile, gate_df, output_dir)
    _try_plot("variance_spectrum",
              plot_variance_spectrum, embeddings, output_dir)

    # Linear separability
    sep_scores = _compute_linear_separability(embeddings, labels)
    if sep_scores:
        _try_plot("linear_separability",
                  plot_linear_separability, sep_scores, output_dir)

    # Clustering quality
    cluster_scores = _compute_clustering_quality(embeddings["e_win"], labels)
    if cluster_scores:
        _try_plot("clustering_quality",
                  plot_clustering_quality, cluster_scores, output_dir)

    summary = {
        "plots": plots,
        "n_samples": len(meta_df),
        "embedding_dim": embeddings["e_win"].shape[1] if len(embeddings["e_win"]) > 0 else 0,
        "separability": sep_scores,
        "clustering": cluster_scores,
    }

    logger.info(
        f"Representation analysis: {len(meta_df)} embeddings, "
        f"{len(plots)} plots"
    )
    return summary
