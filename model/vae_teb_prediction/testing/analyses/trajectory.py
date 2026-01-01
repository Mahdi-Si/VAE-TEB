"""
Trajectory analysis for VAE-TEB models.

Analyzes how latent representations and KL divergence (transfer entropy)
evolve over time before birth. This module is migrated from the standalone
trajectory_analysis.py with integration into the modular testing framework.

IMPORTANT: For proper trajectory analysis, use a GUID-based DataLoader where
each batch contains all epochs from a single patient. This ensures correct
temporal ordering and enables per-patient trajectory visualization.

Use `build_guid_filtered_dataloader` from hdf5_dataset.hdf5_dataset:

    >>> from hdf5_dataset.hdf5_dataset import build_guid_filtered_dataloader
    >>> guids, guid_loader = build_guid_filtered_dataloader(
    ...     dataset_paths=["test.h5"],
    ...     min_samples=3,  # At least 3 epochs per patient
    ... )
    >>> results = run_trajectory_analysis(runner, guid_loader)

With standard batching, the analysis will still work but batches will contain
mixed patients, which is less efficient for per-patient trajectory plots.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from loguru import logger

from ..base import TestRunner
from ..collectors import _extract_epoch, _extract_guid, _extract_label
from ..metrics import compute_kld
from ..visualizers import plot_kld_trajectory

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Constants
TIMESTEP_SECONDS = 4.0  # Each latent step = 4 seconds


class TrajectoryAnalyzer:
    """
    Analyzes latent trajectories and KLD evolution before birth.

    Collects per-timestep latent codes and KLD from the model, organizes
    by patient (GUID) and time-to-birth, and generates visualizations.

    Attributes:
        runner: TestRunner with model and device.
        loader: DataLoader for test data.
        output_dir: Directory for saving results.
        time_range_hours: Hours before birth to analyze.
        min_epochs_per_guid: Minimum epochs required per patient.

    Example:
        >>> analyzer = TrajectoryAnalyzer(runner, loader, output_dir, time_range_hours=12)
        >>> results = analyzer.run()
    """

    def __init__(
        self,
        runner: TestRunner,
        loader: Any,
        output_dir: Path,
        time_range_hours: Optional[float] = 12.0,
        min_epochs_per_guid: int = 3,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize the trajectory analyzer.

        Args:
            runner: TestRunner with model and device configured.
            loader: DataLoader providing batches with metadata.
            output_dir: Where to save outputs.
            time_range_hours: Analyze this many hours before birth (None = all).
            min_epochs_per_guid: Minimum epochs per patient to include.
            class_names: Names for outcome classes.
        """
        self.runner = runner
        self.loader = loader
        self.output_dir = Path(output_dir)
        self.time_range_hours = time_range_hours
        self.min_epochs_per_guid = min_epochs_per_guid

        # Get latent dimension from model
        self.latent_dim = int(getattr(runner.model, "latent_dim_z", 16))
        self.warmup_steps = runner.warmup_steps

        # Class configuration
        self.class_names = class_names or ["healthy", "acidosis", "HIE"]
        self.colors = {
            "healthy": "#2ecc71",
            "acidosis": "#e74c3c",
            "HIE": "#9b59b6",
            "unknown": "#95a5a6",
        }

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "dashboards").mkdir(exist_ok=True)

        # Results storage
        self.latent_df: Optional[pd.DataFrame] = None
        self.epoch_df: Optional[pd.DataFrame] = None

    def run(
        self,
        skip_dashboards: bool = False,
        n_dashboards: int = 12,
    ) -> Dict[str, Any]:
        """
        Run complete trajectory analysis pipeline.

        Steps:
            1. Collect latent codes and KLD from model
            2. Add dynamics (velocity, acceleration)
            3. Fit PCA for visualization
            4. Save data to parquet
            5. Generate plots
            6. Generate per-patient dashboards
            7. Compute summary metrics

        Args:
            skip_dashboards: If True, skip per-patient dashboard generation.
            n_dashboards: Maximum number of patient dashboards to generate.

        Returns:
            Dict with summary statistics and output paths.
        """
        logger.info("Starting trajectory analysis...")

        # Step 1: Collect data
        self.latent_df, self.epoch_df = self._collect_data()

        if self.latent_df.empty:
            logger.warning("No trajectory data collected!")
            return {"status": "empty", "n_samples": 0}

        # Step 2: Add dynamics
        self.latent_df = self._add_dynamics(self.latent_df)

        # Step 3: Fit PCA
        self.latent_df = self._fit_pca(self.latent_df)

        # Step 4: Save data
        self.latent_df.to_parquet(self.output_dir / "latent_trajectories.parquet")
        self.epoch_df.to_parquet(self.output_dir / "epoch_summary.parquet")

        # Step 5: Generate plots
        self._plot_kld_vs_time()
        self._plot_kld_by_class()
        self._plot_latent_space()

        # Step 6: Dashboards
        if not skip_dashboards:
            self._generate_dashboards(n_dashboards)

        # Step 7: Compute metrics
        metrics = self._compute_metrics()

        # Save summary
        summary = {
            "n_samples": len(self.latent_df),
            "n_guids": self.epoch_df["guid"].nunique() if not self.epoch_df.empty else 0,
            "n_epochs": len(self.epoch_df),
            "time_range_hours": self.time_range_hours,
            "metrics": metrics,
        }

        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Trajectory analysis complete: {summary['n_guids']} patients, {summary['n_epochs']} epochs")
        return summary

    def _collect_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Collect latent trajectories and KLD from dataloader."""
        latent_rows: List[Dict[str, Any]] = []
        epoch_rows: List[Dict[str, Any]] = []
        guid_epoch_count: Dict[str, int] = defaultdict(int)

        with self.runner.inference_mode():
            for batch in self.runner.iter_batches(self.loader):
                batch_size = batch.fhr_st.size(0)

                # Forward pass
                outputs = self.runner.forward(batch)

                # Extract latents (use mu_post for deterministic representation)
                latent = outputs.get("mu_post", outputs.get("z"))
                if latent is None:
                    continue
                latent_np = latent.cpu().numpy()  # (B, T, D)

                # Compute per-timestep KLD
                kld_tensor = compute_kld(outputs, self.warmup_steps)
                kld = kld_tensor.sum(dim=-1).cpu().numpy() if kld_tensor is not None else None  # (B, T)

                # Get uncertainty if available
                uncertainty = None
                if "logvar_post" in outputs:
                    uncertainty = outputs["logvar_post"].exp().sum(dim=-1).cpu().numpy()

                # Process each sample
                for idx in range(batch_size):
                    guid = _extract_guid(batch, idx)
                    epoch_sec = _extract_epoch(batch, idx) or 0.0
                    label = self._get_label_name(batch, idx)

                    hours_before = abs(epoch_sec) / 3600.0

                    # Filter by time range
                    if self.time_range_hours and hours_before > self.time_range_hours:
                        continue

                    guid_epoch_count[guid] += 1
                    T = latent_np.shape[1]

                    # Per-timestep data
                    for t in range(T):
                        row = {
                            "guid": guid,
                            "epoch_sec": epoch_sec,
                            "hours_before": hours_before,
                            "label": label,
                            "t": t,
                            "t_sec": t * TIMESTEP_SECONDS,
                            "kld": float(kld[idx, t]) if kld is not None and np.isfinite(kld[idx, t]) else np.nan,
                        }

                        # Add latent dimensions
                        for d in range(min(self.latent_dim, latent_np.shape[2])):
                            row[f"z{d}"] = float(latent_np[idx, t, d])

                        if uncertainty is not None:
                            row["uncertainty"] = float(uncertainty[idx, t])

                        latent_rows.append(row)

                    # Per-epoch summary (excluding warmup)
                    valid_kld = kld[idx, self.warmup_steps:] if kld is not None else None
                    epoch_rows.append({
                        "guid": guid,
                        "epoch_sec": epoch_sec,
                        "hours_before": hours_before,
                        "label": label,
                        "kld_mean": float(np.nanmean(valid_kld)) if valid_kld is not None else np.nan,
                        "kld_std": float(np.nanstd(valid_kld)) if valid_kld is not None else np.nan,
                    })

        # Create DataFrames
        latent_df = pd.DataFrame(latent_rows)
        epoch_df = pd.DataFrame(epoch_rows)

        # Filter by minimum epochs per GUID
        valid_guids = {g for g, c in guid_epoch_count.items() if c >= self.min_epochs_per_guid}

        if not latent_df.empty:
            latent_df = latent_df[latent_df["guid"].isin(valid_guids)]
        if not epoch_df.empty:
            epoch_df = epoch_df[epoch_df["guid"].isin(valid_guids)]

        logger.info(f"Collected {len(epoch_df)} epochs from {len(valid_guids)} patients")
        return latent_df, epoch_df

    def _get_label_name(self, batch: Any, idx: int) -> str:
        """Convert numeric label to class name."""
        label = _extract_label(batch, idx)
        if label is None or label == 0:
            return "unknown"
        if 1 <= label <= len(self.class_names):
            return self.class_names[label - 1]
        return "unknown"

    def _add_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add velocity and acceleration in latent space."""
        if df.empty:
            return df

        z_cols = [f"z{d}" for d in range(self.latent_dim) if f"z{d}" in df.columns]
        if not z_cols:
            return df

        def compute_dynamics(group: pd.DataFrame) -> pd.DataFrame:
            group = group.sort_values("t")
            Z = group[z_cols].values

            # Velocity (first derivative)
            dZ = np.diff(Z, axis=0, prepend=Z[[0]])
            group["speed"] = np.linalg.norm(dZ, axis=1)

            # Acceleration (second derivative)
            d2Z = np.diff(dZ, axis=0, prepend=dZ[[0]])
            group["accel"] = np.linalg.norm(d2Z, axis=1)

            return group

        return df.groupby(["guid", "epoch_sec"], group_keys=False).apply(compute_dynamics)

    def _fit_pca(self, df: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
        """Fit PCA and add principal component columns."""
        if df.empty or not HAS_SKLEARN:
            return df

        z_cols = [f"z{d}" for d in range(self.latent_dim) if f"z{d}" in df.columns]
        if not z_cols:
            return df

        Z = df[z_cols].values
        # Handle NaNs
        valid_mask = np.all(np.isfinite(Z), axis=1)
        if valid_mask.sum() < 10:
            return df

        pca = PCA(n_components=n_components)
        Z_valid = Z[valid_mask]
        X = np.full((len(Z), n_components), np.nan)
        X[valid_mask] = pca.fit_transform(Z_valid)

        for i in range(n_components):
            df[f"pc{i + 1}"] = X[:, i]

        logger.info(f"PCA variance explained: {pca.explained_variance_ratio_.round(3)}")
        return df

    def _plot_kld_vs_time(self) -> None:
        """Plot mean KLD vs hours before birth."""
        if self.epoch_df.empty:
            return

        # Use the standard visualizer
        plot_kld_trajectory(self.epoch_df, self.output_dir / "plots")

    def _plot_kld_by_class(self) -> None:
        """Plot KLD trajectories separated by class."""
        if self.epoch_df.empty or "label" not in self.epoch_df.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        for label in self.epoch_df["label"].unique():
            if label == "unknown":
                continue

            subset = self.epoch_df[self.epoch_df["label"] == label]
            subset = subset.copy()
            subset["hour_bin"] = (subset["hours_before"] * 2).round() / 2  # 30-min bins

            agg = subset.groupby("hour_bin")["kld_mean"].agg(["mean", "std"]).reset_index()
            agg = agg[agg["hour_bin"].notna()]

            color = self.colors.get(label, "#666666")
            ax.plot(agg["hour_bin"], agg["mean"], color=color, linewidth=2, label=label, marker="o", markersize=4)
            ax.fill_between(agg["hour_bin"], agg["mean"] - agg["std"], agg["mean"] + agg["std"], alpha=0.2, color=color)

        ax.set_xlabel("Hours Before Birth")
        ax.set_ylabel("KLD (Transfer Entropy)")
        ax.set_title("Transfer Entropy by Outcome Class")
        ax.invert_xaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.output_dir / "plots" / "kld_by_class.png", dpi=200)
        plt.close(fig)

    def _plot_latent_space(self) -> None:
        """Plot 2D PCA of latent space colored by class."""
        if self.latent_df.empty or "pc1" not in self.latent_df.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        for label in self.latent_df["label"].unique():
            subset = self.latent_df[self.latent_df["label"] == label]
            color = self.colors.get(label, "#666666")
            ax.scatter(subset["pc1"], subset["pc2"], c=color, alpha=0.3, s=1, label=label)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Latent Space (PCA Projection)")
        ax.legend(markerscale=5)

        fig.tight_layout()
        fig.savefig(self.output_dir / "plots" / "latent_space.png", dpi=200)
        plt.close(fig)

    def _generate_dashboards(self, n_dashboards: int = 12) -> None:
        """Generate per-patient multi-panel dashboards."""
        if self.epoch_df.empty:
            return

        # Select GUIDs with most epochs
        guid_counts = self.epoch_df["guid"].value_counts()
        top_guids = guid_counts.head(n_dashboards).index.tolist()

        for guid in top_guids:
            self._create_dashboard(guid)

    def _create_dashboard(self, guid: str) -> None:
        """Create a multi-panel dashboard for one patient."""
        latent_data = self.latent_df[self.latent_df["guid"] == guid] if not self.latent_df.empty else pd.DataFrame()
        epoch_data = self.epoch_df[self.epoch_df["guid"] == guid] if not self.epoch_df.empty else pd.DataFrame()

        if epoch_data.empty:
            return

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig)

        # Panel 1: KLD over epochs
        ax1 = fig.add_subplot(gs[0, :2])
        if not epoch_data.empty:
            epoch_data = epoch_data.sort_values("hours_before", ascending=False)
            ax1.plot(epoch_data["hours_before"], epoch_data["kld_mean"], "b-o", linewidth=2)
            ax1.fill_between(
                epoch_data["hours_before"],
                epoch_data["kld_mean"] - epoch_data["kld_std"],
                epoch_data["kld_mean"] + epoch_data["kld_std"],
                alpha=0.3,
            )
        ax1.set_xlabel("Hours Before Birth")
        ax1.set_ylabel("KLD")
        ax1.set_title(f"GUID: {guid} - KLD Trajectory")
        ax1.invert_xaxis()

        # Panel 2: Latent PC trajectory
        ax2 = fig.add_subplot(gs[0, 2])
        if not latent_data.empty and "pc1" in latent_data.columns:
            ax2.scatter(latent_data["pc1"], latent_data["pc2"], c=latent_data["hours_before"], cmap="viridis", s=1, alpha=0.5)
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
            ax2.set_title("Latent Trajectory")

        # Panel 3: Speed/dynamics
        ax3 = fig.add_subplot(gs[1, :])
        if not latent_data.empty and "speed" in latent_data.columns:
            for epoch in latent_data["epoch_sec"].unique():
                subset = latent_data[latent_data["epoch_sec"] == epoch]
                ax3.plot(subset["t"], subset["speed"], alpha=0.5, linewidth=0.5)
        ax3.set_xlabel("Timestep")
        ax3.set_ylabel("Latent Speed")
        ax3.set_title("Latent Dynamics")

        # Panel 4: Per-dimension traces
        ax4 = fig.add_subplot(gs[2, :])
        z_cols = [c for c in latent_data.columns if c.startswith("z") and c[1:].isdigit()][:4]
        if not latent_data.empty and z_cols:
            sample_epoch = latent_data["epoch_sec"].iloc[0]
            subset = latent_data[latent_data["epoch_sec"] == sample_epoch]
            for col in z_cols:
                ax4.plot(subset["t"], subset[col], label=col, alpha=0.8)
            ax4.legend()
        ax4.set_xlabel("Timestep")
        ax4.set_ylabel("Latent Value")
        ax4.set_title("Latent Dimensions (First Epoch)")

        fig.tight_layout()
        fig.savefig(self.output_dir / "dashboards" / f"{guid}.png", dpi=150)
        plt.close(fig)

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute summary metrics."""
        metrics = {}

        if not self.epoch_df.empty:
            metrics["kld_mean"] = float(self.epoch_df["kld_mean"].mean())
            metrics["kld_std"] = float(self.epoch_df["kld_mean"].std())

        # Silhouette score for class separation (if sklearn available)
        if HAS_SKLEARN and not self.latent_df.empty and "label" in self.latent_df.columns:
            try:
                z_cols = [c for c in self.latent_df.columns if c.startswith("z") and c[1:].isdigit()]
                if z_cols:
                    X = self.latent_df[z_cols].values
                    labels = self.latent_df["label"].values

                    # Filter to valid labels
                    valid = (labels != "unknown") & np.all(np.isfinite(X), axis=1)
                    if valid.sum() > 100:
                        score = silhouette_score(X[valid], labels[valid])
                        metrics["silhouette_score"] = float(score)
            except Exception as e:
                logger.warning(f"Could not compute silhouette score: {e}")

        return metrics


def run_trajectory_analysis(
    runner: TestRunner,
    loader: Any,
    time_range_hours: float = 12.0,
    min_epochs_per_guid: int = 3,
    skip_dashboards: bool = False,
    n_dashboards: int = 12,
) -> Dict[str, Any]:
    """
    Run complete trajectory analysis.

    Convenience function that creates a TrajectoryAnalyzer and runs it.

    Args:
        runner: TestRunner with model and device.
        loader: DataLoader for test data.
        time_range_hours: Hours before birth to analyze.
        min_epochs_per_guid: Minimum epochs per patient.
        skip_dashboards: If True, skip dashboard generation.
        n_dashboards: Maximum dashboards to generate.

    Returns:
        Summary dict with statistics and output paths.

    Example:
        >>> results = run_trajectory_analysis(runner, test_loader)
        >>> print(f"Analyzed {results['n_guids']} patients")
    """
    output_dir = runner.ensure_dir("trajectory")

    analyzer = TrajectoryAnalyzer(
        runner=runner,
        loader=loader,
        output_dir=output_dir,
        time_range_hours=time_range_hours,
        min_epochs_per_guid=min_epochs_per_guid,
    )

    return analyzer.run(skip_dashboards=skip_dashboards, n_dashboards=n_dashboards)
