"""Class separation analysis for discriminative fine-tuning evaluation.

Quantifies and visualizes whether the VAE latent space has become
class-separable after discriminative fine-tuning (center loss + auxiliary
classifier).  Four tiers of analysis are provided:

    1. **Clustering quality** — Silhouette, Davies-Bouldin, Calinski-Harabasz,
       Fisher discriminant ratio, cohesion/separation metrics.
    2. **Linear separability** — Stratified K-fold with Logistic Regression & LDA.
    3. **Temporal evolution** — How class separation changes as delivery
       approaches (metrics binned by hours-before-birth).
    4. **Center loss effectiveness** — Distance to learned EMA centroids vs
       nearest foreign centroid (requires discriminative checkpoint).

Example:
    >>> from testing.analyses.class_separation import run_class_separation_analysis
    >>> report = run_class_separation_analysis(
    ...     latent_df,
    ...     output_dir="results/class_separation",
    ...     checkpoint_path="disc_model.ckpt",
    ... )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger


# ============================================================================
# Tier 1 — Clustering Quality Metrics
# ============================================================================

def compute_cluster_quality_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    max_samples: int = 50_000,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute standard clustering quality metrics.

    Args:
        X: Feature matrix of shape ``(N, D)``.
        labels: Integer class labels of shape ``(N,)``.
        max_samples: If ``N`` exceeds this, stratified subsampling is applied
            to keep compute tractable (Silhouette is O(N²)).
        random_state: Seed for reproducible subsampling.

    Returns:
        Dictionary with keys ``silhouette``, ``davies_bouldin``, and
        ``calinski_harabasz``.  Values are ``float('nan')`` on failure.
    """
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        logger.warning("Need >= 2 classes for clustering metrics.")
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
        }

    # Stratified subsample if too large
    if len(X) > max_samples:
        rng = np.random.RandomState(random_state)
        idx = _stratified_subsample(labels, max_samples, rng)
        X, labels = X[idx], labels[idx]

    result: Dict[str, float] = {}
    try:
        result["silhouette"] = float(silhouette_score(X, labels, sample_size=min(len(X), 10_000), random_state=random_state))
    except Exception as exc:
        logger.warning(f"Silhouette score failed: {exc}")
        result["silhouette"] = float("nan")

    try:
        result["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    except Exception as exc:
        logger.warning(f"Davies-Bouldin score failed: {exc}")
        result["davies_bouldin"] = float("nan")

    try:
        result["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    except Exception as exc:
        logger.warning(f"Calinski-Harabasz score failed: {exc}")
        result["calinski_harabasz"] = float("nan")

    return result


def compute_class_cohesion_separation(
    X: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """Compute per-class cohesion, pairwise separation, and Fisher ratio.

    Cohesion is the mean intra-class L2 distance to the class centroid.
    Separation is the pairwise L2 distance between centroids.  The Fisher
    discriminant ratio is ``trace(S_B) / trace(S_W)`` where ``S_B`` and
    ``S_W`` are the between-class and within-class scatter matrices.

    Args:
        X: Feature matrix of shape ``(N, D)``.
        labels: Integer class labels of shape ``(N,)``.

    Returns:
        Dictionary containing:
            - ``centroids``: dict mapping class → centroid ndarray.
            - ``within_class_mean``: dict mapping class → mean L2 to centroid.
            - ``within_class_std``: dict mapping class → std of L2 distances.
            - ``between_class_distances``: dict mapping ``"i_vs_j"`` → L2 dist.
            - ``separation_ratios``: dict mapping ``"i_vs_j"`` → ratio.
            - ``fisher_ratio``: scalar trace(S_B) / trace(S_W).
    """
    unique = np.unique(labels)
    global_centroid = X.mean(axis=0)

    centroids: Dict[int, np.ndarray] = {}
    within_mean: Dict[int, float] = {}
    within_std: Dict[int, float] = {}
    S_W = np.zeros((X.shape[1], X.shape[1]))
    S_B = np.zeros((X.shape[1], X.shape[1]))

    for c in unique:
        mask = labels == c
        Xc = X[mask]
        mu_c = Xc.mean(axis=0)
        centroids[int(c)] = mu_c

        diffs = Xc - mu_c
        dists = np.linalg.norm(diffs, axis=1)
        within_mean[int(c)] = float(dists.mean())
        within_std[int(c)] = float(dists.std())

        # Scatter matrices
        S_W += diffs.T @ diffs
        diff_bc = (mu_c - global_centroid).reshape(-1, 1)
        S_B += mask.sum() * (diff_bc @ diff_bc.T)

    # Pairwise centroid distances + separation ratios
    between: Dict[str, float] = {}
    sep_ratios: Dict[str, float] = {}
    for i, ci in enumerate(unique):
        for j, cj in enumerate(unique):
            if j <= i:
                continue
            key = f"{int(ci)}_vs_{int(cj)}"
            dist = float(np.linalg.norm(centroids[int(ci)] - centroids[int(cj)]))
            between[key] = dist
            avg_within = 0.5 * (within_mean[int(ci)] + within_mean[int(cj)])
            sep_ratios[key] = dist / max(avg_within, 1e-12)

    trace_sw = max(float(np.trace(S_W)), 1e-12)
    fisher = float(np.trace(S_B)) / trace_sw

    return {
        "centroids": {int(k): v.tolist() for k, v in centroids.items()},
        "within_class_mean": within_mean,
        "within_class_std": within_std,
        "between_class_distances": between,
        "separation_ratios": sep_ratios,
        "fisher_ratio": fisher,
    }


def compute_linear_separability(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    n_splits: int = 5,
    max_samples: int = 100_000,
    random_state: int = 42,
) -> Dict[str, float]:
    """Evaluate linear separability via stratified K-fold CV.

    Fits a Logistic Regression and LDA classifier independently and
    reports mean ± std accuracy for each.

    Args:
        X: Feature matrix of shape ``(N, D)``.
        labels: Integer class labels of shape ``(N,)``.
        n_splits: Number of stratified folds.
        max_samples: Subsample if dataset is very large.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with ``logreg_mean``, ``logreg_std``, ``lda_mean``,
        ``lda_std`` accuracy values.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    unique = np.unique(labels)
    if len(unique) < 2:
        return {"logreg_mean": float("nan"), "logreg_std": float("nan"),
                "lda_mean": float("nan"), "lda_std": float("nan")}

    # Ensure minimum samples per class for stratified CV
    min_per_class = min(np.sum(labels == c) for c in unique)
    if min_per_class < n_splits:
        logger.warning(f"Min class count ({min_per_class}) < n_splits ({n_splits}). Reducing n_splits.")
        n_splits = max(2, min_per_class)

    if len(X) > max_samples:
        rng = np.random.RandomState(random_state)
        idx = _stratified_subsample(labels, max_samples, rng)
        X, labels = X[idx], labels[idx]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    result: Dict[str, float] = {}

    # Logistic Regression
    try:
        pipe_lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, random_state=random_state, solver="lbfgs"),
        )
        scores_lr = cross_val_score(pipe_lr, X, labels, cv=cv, scoring="accuracy")
        result["logreg_mean"] = float(scores_lr.mean())
        result["logreg_std"] = float(scores_lr.std())
    except Exception as exc:
        logger.warning(f"LogReg CV failed: {exc}")
        result["logreg_mean"] = float("nan")
        result["logreg_std"] = float("nan")

    # LDA
    try:
        pipe_lda = make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(),
        )
        scores_lda = cross_val_score(pipe_lda, X, labels, cv=cv, scoring="accuracy")
        result["lda_mean"] = float(scores_lda.mean())
        result["lda_std"] = float(scores_lda.std())
    except Exception as exc:
        logger.warning(f"LDA CV failed: {exc}")
        result["lda_mean"] = float("nan")
        result["lda_std"] = float("nan")

    return result


# ============================================================================
# Tier 3 — Temporal Evolution of Separation
# ============================================================================

def compute_temporal_separation(
    latent_df: pd.DataFrame,
    *,
    z_cols: Optional[List[str]] = None,
    label_col: str = "label",
    time_col: str = "hours_before",
    n_bins: int = 24,
    min_samples_per_bin: int = 30,
) -> pd.DataFrame:
    """Compute class-separation metrics in temporal bins.

    Bins the data by ``hours_before`` (time-to-birth) and computes
    silhouette, Davies-Bouldin, Calinski-Harabasz, and Fisher ratio
    at each bin.  This shows whether separation increases as delivery
    approaches.

    Args:
        latent_df: DataFrame with z-columns, label, and hours_before.
        z_cols: Explicit z-column names.  Auto-detected if ``None``.
        label_col: Column with integer or string labels.
        time_col: Column with hours before delivery.
        n_bins: Number of temporal bins.
        min_samples_per_bin: Minimum samples required per bin for valid
            metric computation.

    Returns:
        DataFrame with one row per bin containing ``bin_center``,
        ``n_samples``, ``n_classes``, ``silhouette``, ``davies_bouldin``,
        ``calinski_harabasz``, ``fisher_ratio``, and per-pair centroid
        distances.
    """
    if z_cols is None:
        z_cols = _detect_z_cols(latent_df)
    if not z_cols or time_col not in latent_df.columns:
        return pd.DataFrame()

    df = latent_df.dropna(subset=z_cols + [time_col, label_col]).copy()
    df = df[df[label_col].astype(str) != "unknown"].reset_index(drop=True)

    # Encode labels as integers
    labels_enc, _ = _encode_labels(df[label_col].values)
    df["_label_int"] = labels_enc

    # Bin by time
    max_h = df[time_col].max()
    bins = np.linspace(0, max_h, n_bins + 1)
    df["_time_bin"] = pd.cut(df[time_col], bins=bins, labels=False, include_lowest=True)

    rows = []
    for bin_idx in range(n_bins):
        subset = df[df["_time_bin"] == bin_idx]
        if len(subset) < min_samples_per_bin:
            continue

        X = subset[z_cols].values
        y = subset["_label_int"].values
        n_classes = len(np.unique(y))

        bin_center = float(0.5 * (bins[bin_idx] + bins[bin_idx + 1]))
        row: Dict[str, Any] = {
            "bin_center": bin_center,
            "n_samples": len(subset),
            "n_classes": n_classes,
        }

        if n_classes >= 2:
            cq = compute_cluster_quality_metrics(X, y, max_samples=10_000)
            row.update(cq)

            cs = compute_class_cohesion_separation(X, y)
            row["fisher_ratio"] = cs["fisher_ratio"]
            row.update({f"centroid_dist_{k}": v for k, v in cs["between_class_distances"].items()})
            row.update({f"within_mean_{k}": v for k, v in cs["within_class_mean"].items()})
        else:
            row.update({
                "silhouette": float("nan"),
                "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan"),
                "fisher_ratio": float("nan"),
            })

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# Tier 4 — Center Loss Effectiveness
# ============================================================================

def load_discriminative_centers(
    checkpoint_path: Union[str, Path],
) -> Optional[np.ndarray]:
    """Load learned EMA centroids from a discriminative checkpoint.

    Args:
        checkpoint_path: Path to a discriminative fine-tuning checkpoint
            containing ``center_loss.centers`` buffer.

    Returns:
        Centroid array of shape ``(num_classes, latent_dim)`` or ``None``
        if no center-loss buffer is found.
    """
    import torch

    path = Path(checkpoint_path)
    if not path.exists():
        logger.warning(f"Checkpoint not found: {path}")
        return None

    raw = torch.load(str(path), map_location="cpu")

    sd: Any = raw
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in raw:
                sd = raw[key]
                break

    if not isinstance(sd, dict):
        return None

    for key, val in sd.items():
        if key.endswith("center_loss.centers"):
            centers = val.cpu().numpy()
            logger.info(f"Loaded EMA centers from checkpoint: shape {centers.shape}")
            return centers

    logger.info("No center_loss.centers found in checkpoint.")
    return None


def compute_center_loss_effectiveness(
    X: np.ndarray,
    labels: np.ndarray,
    learned_centers: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate how well samples cluster around their learned EMA centroids.

    For each sample, computes the L2 distance to its own class centroid
    and to the nearest foreign centroid.  A well-trained center loss yields
    small own-distances and large foreign-distances.

    Args:
        X: Feature matrix of shape ``(N, D)``.
        labels: Integer 0-indexed class labels of shape ``(N,)``.
        learned_centers: Centroid array of shape ``(C, D)`` from the
            discriminative checkpoint.

    Returns:
        Dictionary with:
            - ``per_class_own_dist``: dict of class → list of own distances.
            - ``per_class_foreign_dist``: dict of class → list of nearest
              foreign distances.
            - ``misassignment_rate``: fraction closer to wrong centroid.
            - ``mean_own_dist``: global mean own-centroid distance.
            - ``mean_foreign_dist``: global mean nearest-foreign distance.
            - ``separation_ratio``: mean_foreign / mean_own.
    """
    num_classes = learned_centers.shape[0]

    # Distances to all centroids: (N, C)
    dists = np.linalg.norm(
        X[:, np.newaxis, :] - learned_centers[np.newaxis, :, :], axis=2
    )

    own_dists = np.array([dists[i, labels[i]] for i in range(len(X))])

    # Nearest foreign centroid distance
    foreign_dists = np.full(len(X), np.inf)
    for i in range(len(X)):
        for c in range(num_classes):
            if c != labels[i]:
                foreign_dists[i] = min(foreign_dists[i], dists[i, c])

    misassigned = (own_dists > foreign_dists).sum()
    misassignment_rate = float(misassigned) / max(len(X), 1)

    per_class_own: Dict[int, List[float]] = {}
    per_class_foreign: Dict[int, List[float]] = {}
    for c in range(num_classes):
        mask = labels == c
        per_class_own[c] = own_dists[mask].tolist()
        per_class_foreign[c] = foreign_dists[mask].tolist()

    mean_own = float(own_dists.mean())
    mean_foreign = float(foreign_dists.mean())

    return {
        "per_class_own_dist": per_class_own,
        "per_class_foreign_dist": per_class_foreign,
        "misassignment_rate": misassignment_rate,
        "mean_own_dist": mean_own,
        "mean_foreign_dist": mean_foreign,
        "separation_ratio": mean_foreign / max(mean_own, 1e-12),
    }


# ============================================================================
# Orchestrator
# ============================================================================

def run_class_separation_analysis(
    latent_df: pd.DataFrame,
    output_dir: Union[str, Path],
    *,
    checkpoint_path: Optional[Union[str, Path]] = None,
    z_cols: Optional[List[str]] = None,
    label_col: str = "label",
    time_col: str = "hours_before",
    n_temporal_bins: int = 24,
    max_samples_clustering: int = 50_000,
    max_samples_linear: int = 100_000,
) -> Dict[str, Any]:
    """Run the full class-separation analysis pipeline.

    Orchestrates all four tiers, saving a JSON report, temporal CSV,
    and all plots.  Each tier runs in a try/except so one failure does
    not block others.

    Args:
        latent_df: DataFrame with z0..z15, label, hours_before columns.
            Typically loaded from ``latent_trajectories.parquet``.
        output_dir: Directory for all outputs.  A ``class_separation/``
            subdirectory is created automatically.
        checkpoint_path: Optional path to discriminative checkpoint for
            Tier 4 (center-loss effectiveness).
        z_cols: Explicit z-column names.  Auto-detected if ``None``.
        label_col: Column name for class labels.
        time_col: Column name for hours-before-birth.
        n_temporal_bins: Number of bins for temporal separation analysis.
        max_samples_clustering: Max samples for clustering metrics.
        max_samples_linear: Max samples for linear separability.

    Returns:
        Dictionary of all scalar metrics from all tiers plus paths to
        saved artifacts.
    """
    from model.vae_teb_prediction.testing.visualizers import (
        plot_centroid_distance_heatmap,
        plot_class_separation_scatter_2d,
        plot_distance_to_centroid_violins,
        plot_per_dimension_boxplots,
        plot_temporal_class_separation,
    )

    out = Path(output_dir) / "class_separation"
    out.mkdir(parents=True, exist_ok=True)

    if z_cols is None:
        z_cols = _detect_z_cols(latent_df)
    if not z_cols:
        logger.error("No z-columns found in DataFrame.")
        return {"error": "no z-columns"}

    # Filter to valid data (drop NaNs and unknown labels)
    df = latent_df.dropna(subset=z_cols + [label_col]).copy()
    df = df[df[label_col].astype(str) != "unknown"].reset_index(drop=True)
    labels_enc, label_map = _encode_labels(df[label_col].values)
    X = df[z_cols].values

    unique = np.unique(labels_enc)
    if len(unique) < 2:
        logger.error(f"Only {len(unique)} class(es) found. Need >= 2.")
        return {"error": f"only {len(unique)} class(es)"}

    logger.info(f"Class separation analysis: {len(X)} samples, {len(unique)} classes, {len(z_cols)} dims")
    logger.info(f"Label mapping: {label_map}")

    report: Dict[str, Any] = {"label_map": label_map, "n_samples": len(X), "n_dims": len(z_cols)}

    # ------------------------------------------------------------------
    # Tier 1: Clustering quality metrics
    # ------------------------------------------------------------------
    try:
        logger.info("Tier 1: Computing clustering quality metrics...")
        cq = compute_cluster_quality_metrics(X, labels_enc, max_samples=max_samples_clustering)
        report["clustering"] = cq
        logger.info(f"  Silhouette={cq['silhouette']:.4f}  DB={cq['davies_bouldin']:.4f}  CH={cq['calinski_harabasz']:.1f}")
    except Exception as exc:
        logger.error(f"Tier 1 clustering metrics failed: {exc}")
        report["clustering"] = {"error": str(exc)}

    try:
        logger.info("Tier 1: Computing cohesion/separation metrics...")
        cs = compute_class_cohesion_separation(X, labels_enc)
        report["cohesion_separation"] = {
            k: v for k, v in cs.items() if k != "centroids"
        }
        report["cohesion_separation"]["fisher_ratio"] = cs["fisher_ratio"]
        logger.info(f"  Fisher ratio={cs['fisher_ratio']:.4f}")
        for k, v in cs["between_class_distances"].items():
            logger.info(f"  Centroid dist {k}: {v:.4f}")
    except Exception as exc:
        logger.error(f"Tier 1 cohesion/separation failed: {exc}")
        report["cohesion_separation"] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # Tier 1b: Linear separability
    # ------------------------------------------------------------------
    try:
        logger.info("Tier 1b: Computing linear separability...")
        ls = compute_linear_separability(X, labels_enc, max_samples=max_samples_linear)
        report["linear_separability"] = ls
        logger.info(f"  LogReg CV: {ls['logreg_mean']:.4f} ± {ls['logreg_std']:.4f}")
        logger.info(f"  LDA CV:    {ls['lda_mean']:.4f} ± {ls['lda_std']:.4f}")
    except Exception as exc:
        logger.error(f"Tier 1b linear separability failed: {exc}")
        report["linear_separability"] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # Tier 2: Visualizations
    # ------------------------------------------------------------------
    try:
        logger.info("Tier 2: Generating 2D scatter plots...")
        _generate_scatter_plots(X, labels_enc, label_map, out, plot_class_separation_scatter_2d)
    except Exception as exc:
        logger.error(f"Tier 2 scatter plots failed: {exc}")

    try:
        logger.info("Tier 2: Generating per-dimension boxplots...")
        plot_per_dimension_boxplots(X, labels_enc, out / "per_dimension_boxplots.png", label_map=label_map, dim_names=z_cols)
    except Exception as exc:
        logger.error(f"Tier 2 boxplots failed: {exc}")

    try:
        if "cohesion_separation" in report and "error" not in report.get("cohesion_separation", {}):
            logger.info("Tier 2: Generating centroid distance heatmap...")
            cs_data = compute_class_cohesion_separation(X, labels_enc)
            plot_centroid_distance_heatmap(cs_data["between_class_distances"], cs_data["centroids"], out / "centroid_distance_heatmap.png", label_map=label_map)
    except Exception as exc:
        logger.error(f"Tier 2 heatmap failed: {exc}")

    try:
        if "cohesion_separation" in report and "error" not in report.get("cohesion_separation", {}):
            logger.info("Tier 2: Generating distance-to-centroid violins...")
            cs_data = compute_class_cohesion_separation(X, labels_enc)
            _plot_intra_class_violins(X, labels_enc, cs_data["centroids"], label_map, out, plot_distance_to_centroid_violins)
    except Exception as exc:
        logger.error(f"Tier 2 violin plots failed: {exc}")

    # ------------------------------------------------------------------
    # Tier 3: Temporal separation evolution
    # ------------------------------------------------------------------
    try:
        if time_col in latent_df.columns:
            logger.info("Tier 3: Computing temporal separation evolution...")
            temporal_df = compute_temporal_separation(
                df, z_cols=z_cols, label_col=label_col,
                time_col=time_col, n_bins=n_temporal_bins,
            )
            if not temporal_df.empty:
                temporal_df.to_csv(out / "temporal_separation.csv", index=False)
                report["temporal_n_bins"] = len(temporal_df)
                plot_temporal_class_separation(temporal_df, out / "temporal_separation_metrics.png")
                logger.info(f"  {len(temporal_df)} temporal bins saved.")
            else:
                logger.warning("Temporal separation: no bins with sufficient data.")
        else:
            logger.info(f"Skipping temporal analysis: '{time_col}' column not found.")
    except Exception as exc:
        logger.error(f"Tier 3 temporal separation failed: {exc}")

    # ------------------------------------------------------------------
    # Tier 4: Center loss effectiveness (optional)
    # ------------------------------------------------------------------
    if checkpoint_path is not None:
        try:
            logger.info("Tier 4: Evaluating center loss effectiveness...")
            centers = load_discriminative_centers(checkpoint_path)
            if centers is not None:
                eff = compute_center_loss_effectiveness(X, labels_enc, centers)
                report["center_loss"] = {
                    k: v for k, v in eff.items()
                    if k not in ("per_class_own_dist", "per_class_foreign_dist")
                }
                logger.info(f"  Misassignment rate: {eff['misassignment_rate']:.4f}")
                logger.info(f"  Mean own dist: {eff['mean_own_dist']:.4f}")
                logger.info(f"  Mean foreign dist: {eff['mean_foreign_dist']:.4f}")
                logger.info(f"  Separation ratio: {eff['separation_ratio']:.4f}")

                # Violin plot of distances
                plot_distance_to_centroid_violins(
                    eff["per_class_own_dist"],
                    out / "center_loss_distances.png",
                    label_map=label_map,
                    title="Distance to Learned EMA Centroid",
                    foreign_dists=eff["per_class_foreign_dist"],
                )
            else:
                logger.info("No center_loss.centers in checkpoint — skipping Tier 4.")
        except Exception as exc:
            logger.error(f"Tier 4 center loss analysis failed: {exc}")

    # ------------------------------------------------------------------
    # Save JSON report
    # ------------------------------------------------------------------
    report_path = out / "class_separation_report.json"
    _save_json(report, report_path)
    logger.info(f"Class separation report saved to {report_path}")

    return report


# ============================================================================
# Private helpers
# ============================================================================

def _detect_z_cols(df: pd.DataFrame) -> List[str]:
    """Auto-detect z-dimension columns (z0, z1, ..., z15)."""
    return [c for c in df.columns if c.startswith("z") and c[1:].isdigit()]


def _encode_labels(labels: np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
    """Encode string/mixed labels into integer codes.

    Assumes unknowns have already been filtered out of the input.

    Returns:
        Tuple of (integer labels, {int: original_label_string} map).
    """
    unique_raw = sorted(set(str(l) for l in labels))
    label_to_int = {l: i for i, l in enumerate(unique_raw)}
    encoded = np.array([label_to_int[str(l)] for l in labels])
    label_map = {v: k for k, v in label_to_int.items()}
    return encoded, label_map


def _stratified_subsample(
    labels: np.ndarray,
    max_samples: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Stratified subsample preserving class proportions."""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    indices = []
    for c, cnt in zip(unique, counts):
        c_idx = np.where(labels == c)[0]
        n_take = max(1, int(round(cnt / total * max_samples)))
        n_take = min(n_take, len(c_idx))
        indices.append(rng.choice(c_idx, n_take, replace=False))
    return np.concatenate(indices)


def _generate_scatter_plots(
    X: np.ndarray,
    labels: np.ndarray,
    label_map: Dict[int, str],
    output_dir: Path,
    plot_fn: Any,
) -> None:
    """Generate PCA and t-SNE 2D scatter plots."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    centroids_pca = {}
    for c in np.unique(labels):
        centroids_pca[int(c)] = X_pca[labels == c].mean(axis=0)

    plot_fn(
        X_pca, labels, output_dir / "class_separation_scatter_pca.png",
        centroids=centroids_pca, method_name="PCA",
        label_map=label_map,
        explained_var=pca.explained_variance_ratio_[:2],
    )

    # t-SNE (subsample for speed)
    max_tsne = 10_000
    if len(X) > max_tsne:
        rng = np.random.RandomState(42)
        idx = _stratified_subsample(labels, max_tsne, rng)
        X_tsne_in, labels_tsne = X[idx], labels[idx]
    else:
        X_tsne_in, labels_tsne = X, labels

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_tsne_in) - 1))
    X_tsne = tsne.fit_transform(X_tsne_in)
    centroids_tsne = {}
    for c in np.unique(labels_tsne):
        centroids_tsne[int(c)] = X_tsne[labels_tsne == c].mean(axis=0)

    plot_fn(
        X_tsne, labels_tsne, output_dir / "class_separation_scatter_tsne.png",
        centroids=centroids_tsne, method_name="t-SNE",
        label_map=label_map,
    )


def _plot_intra_class_violins(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: Dict[int, Any],
    label_map: Dict[int, str],
    output_dir: Path,
    plot_fn: Any,
) -> None:
    """Compute and plot per-class distance-to-centroid distributions."""
    # Convert centroids to numpy arrays
    centroid_arrays = {
        k: np.array(v) if not isinstance(v, np.ndarray) else v
        for k, v in centroids.items()
    }

    per_class_dists: Dict[int, List[float]] = {}
    for c in np.unique(labels):
        mask = labels == c
        Xc = X[mask]
        mu = centroid_arrays[int(c)]
        dists = np.linalg.norm(Xc - mu, axis=1)
        per_class_dists[int(c)] = dists.tolist()

    plot_fn(
        per_class_dists,
        output_dir / "distance_to_centroid_violins.png",
        label_map=label_map,
        title="Distance to Empirical Class Centroid",
    )


def _save_json(data: Dict[str, Any], path: Path) -> None:
    """Save a dictionary as JSON with numpy-safe serialization."""
    def _default(obj):
        if isinstance(obj, (np.floating, float)):
            if np.isnan(obj) or np.isinf(obj):
                return str(obj)
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_default)
