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

from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.collectors import _extract_epoch, _extract_guid, _extract_label
from model.vae_teb_prediction.testing.metrics import (
    compute_kld_per_timestep,
    compute_trajectory_features,
    preprocess_latent,
    reduce_latent_dimensionality,
)
from model.vae_teb_prediction.testing.visualizers import (
    plot_kld_trajectory,
    plot_kld_guid_trajectory,
    plot_kld_trajectory_3d,
    plot_guid_absolute_trajectory,
    plot_guid_trajectory_3d,
    plot_latent_trajectory_2d,
    plot_latent_trajectory_3d,
    plot_latent_changepoints_with_raw,
    plot_recurrence,
    plot_segment_statistics,
    plot_trajectory_comparison,
)

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Optional imports for advanced features
try:
    from model.vae_teb_prediction.testing.analyses.changepoint import (
        detect_changepoints,
        summarize_latent_segments,
        create_changepoint_detector,
    )
    HAS_CHANGEPOINT = True
except ImportError:
    HAS_CHANGEPOINT = False

try:
    from model.vae_teb_prediction.testing.visualizers_interactive import (
        plot_latent_trajectory_3d_interactive,
        plot_kld_trajectory_3d_interactive,
        plot_fhr_timeline,
        plot_fhr_up_timeline,
        plot_guid_trajectory_3d_interactive,
        plot_trajectory_animation,
        plot_trajectory_comparison_interactive,
    )
    HAS_INTERACTIVE = True
except ImportError:
    HAS_INTERACTIVE = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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
        dim_reduction_method: str = "pca",
        n_changepoints: int = 5,
        plot_3d: bool = True,
        plot_animations: bool = False,
        decimation_factor: int = 16,
        changepoint_algo: str = "pelt",
        preprocess_latent_trajectories: bool = False,
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
            dim_reduction_method: Dimensionality reduction method
                ('pca', 'umap', 'tsne', 'isomap', 'diffusion').
            n_changepoints: Number of changepoints to detect per sample.
            plot_3d: Whether to generate 3D trajectory plots.
            plot_animations: Whether to generate trajectory animations (slower).
            decimation_factor: Ratio between raw signal and latent lengths.
            changepoint_algo: Changepoint detection algorithm
                ('pelt', 'binseg', 'bottomup', 'window', 'dynp', 'gradient').
            preprocess_latent_trajectories: If True, apply robust normalization
                and optional denoising to latent z-columns before analysis.
        """
        self.runner = runner
        self.loader = loader
        self.output_dir = Path(output_dir)
        self.time_range_hours = time_range_hours
        self.min_epochs_per_guid = min_epochs_per_guid

        # Get latent dimension from model
        self.latent_dim = int(getattr(runner.model, "latent_dim_z", 16))
        self.warmup_steps = runner.warmup_steps

        # Advanced analysis settings
        self.dim_reduction_method = dim_reduction_method
        self.n_changepoints = n_changepoints
        self.plot_3d = plot_3d
        self.plot_animations = plot_animations
        self.decimation_factor = decimation_factor
        self.changepoint_algo = changepoint_algo
        self.preprocess_latent_trajectories = preprocess_latent_trajectories

        # Class configuration
        self.class_names = class_names or ["healthy", "acidosis", "HIE"]
        self.colors = {
            "healthy": "#609966",    # Muted green (matches COLOR_GREEN)
            "acidosis": "#EB5B00",   # Vivid vermillion (matches COLOR_VERMILLION)
            "HIE": "#112D4E",        # Deep navy-purple (matches COLOR_PURPLE)
            "unknown": "#393E46",    # Dark gray (matches COLOR_GRAY)
        }

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "dashboards").mkdir(exist_ok=True)
        (self.output_dir / "changepoint_analysis").mkdir(exist_ok=True)
        (self.output_dir / "npy").mkdir(exist_ok=True)
        if plot_animations:
            (self.output_dir / "animations").mkdir(exist_ok=True)

        # Results storage
        self.latent_df: Optional[pd.DataFrame] = None
        self.epoch_df: Optional[pd.DataFrame] = None
        self.raw_data: Dict[str, Dict[str, Any]] = {}  # Store raw signals per GUID
        self.changepoint_results: Dict[str, Any] = {}
        self.segment_stats: List[Dict[str, Any]] = []
        self.guid_trajectories: Dict[str, Dict[str, Any]] = {}
        self.guid_features_df: Optional[pd.DataFrame] = None

    def run(
        self,
        skip_dashboards: bool = False,
        n_dashboards: int = 12,
        class_analysis: bool = False,
        n_trajectory_samples: int = 5,
        n_kld_guid_plots: int = 12,
        kld_guid_list: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run complete trajectory analysis pipeline.

        Steps:
            1. Collect latent codes and KLD from model
            2. Add dynamics (velocity, acceleration)
            3. Apply dimensionality reduction (PCA, UMAP, t-SNE, etc.)
            4. Detect changepoints and compute segment statistics
            5. Save data to parquet
            6. Generate plots (including 3D trajectories)
            7. Generate per-patient dashboards
            8. Generate animations (if enabled)
            9. Compare class trajectories (if class_analysis enabled)
            10. Compute summary metrics

        Args:
            skip_dashboards: If True, skip per-patient dashboard generation.
            n_dashboards: Maximum number of patient dashboards to generate.
            class_analysis: If True, include class-based plots/analysis.
            n_trajectory_samples: Number of samples for 3D trajectory plots.
            n_kld_guid_plots: Number of GUIDs to plot for per-epoch KLD trends.
            kld_guid_list: Optional list of GUIDs to plot (overrides top-N selection).

        Returns:
            Dict with summary statistics and output paths.
        """
        logger.info("Starting trajectory analysis...")
        self.class_analysis = class_analysis

        # Step 1: Collect data
        self.latent_df, self.epoch_df = self._collect_data()

        if self.latent_df.empty:
            logger.warning("No trajectory data collected!")
            return {"status": "empty", "n_samples": 0}

        # Step 1b: Optional latent preprocessing
        if self.preprocess_latent_trajectories:
            self._preprocess_latents()

        # Step 2: Add dynamics
        self.latent_df = self._add_dynamics(self.latent_df)

        # Step 3: Apply dimensionality reduction (global fit)
        self._reduce_dimensionality()

        # Step 3b: Build GUID-level stitched trajectories
        self.guid_trajectories = self._build_guid_trajectories()

        # Step 3c: Extract per-GUID trajectory features
        self.guid_features_df = self._extract_guid_features()

        # Step 4: Detect changepoints and compute segment statistics
        if HAS_CHANGEPOINT and self.n_changepoints > 0:
            self._detect_changepoints_all()
            self._compute_segment_statistics()

        # Step 5: Save data + export CSVs (including NPY)
        self.latent_df.to_parquet(self.output_dir / "latent_trajectories.parquet")
        self.epoch_df.to_parquet(self.output_dir / "epoch_summary.parquet")
        self._export_csvs()

        # Step 6: Generate plots (including recurrence)
        self._plot_kld_vs_time()
        if class_analysis:
            self._plot_kld_by_class()
        self._plot_kld_guid_trajectories(n_samples=n_kld_guid_plots, guid_list=kld_guid_list)
        self._plot_latent_space(color_by_label=class_analysis)
        self._plot_guid_absolute_trajectories(n_samples=n_kld_guid_plots, guid_list=kld_guid_list)
        self._plot_recurrence_samples(n_samples=min(5, n_trajectory_samples))

        # Step 7: 3D trajectory plots (per-epoch + GUID-level)
        if self.plot_3d:
            self._plot_3d_trajectories(n_samples=n_trajectory_samples)
            self._plot_guid_3d_trajectories(n_samples=n_trajectory_samples)
            self._plot_kld_3d_trajectories(n_samples=n_trajectory_samples)

        # Step 8: Changepoint analysis plots
        if HAS_CHANGEPOINT and self.changepoint_results:
            self._plot_changepoints_with_raw(n_samples=n_trajectory_samples)
            if self.segment_stats:
                self._plot_segment_stats()

        # Step 9: Dashboards
        if not skip_dashboards:
            self._generate_dashboards(n_dashboards)

        # Step 10: Generate animations (if enabled)
        if self.plot_animations and HAS_INTERACTIVE:
            self._generate_trajectory_animations(n_samples=min(3, n_trajectory_samples))

        # Step 10b: FHR+UP timelines
        if HAS_INTERACTIVE:
            self._plot_fhr_up_timelines(n_samples=min(5, n_trajectory_samples))

        # Step 11: Compare class trajectories
        if class_analysis and HAS_INTERACTIVE:
            self._compare_class_trajectories()

        # Step 10: Compute metrics
        metrics = self._compute_metrics()

        # Save summary
        summary = {
            "n_samples": len(self.latent_df),
            "n_guids": self.epoch_df["guid"].nunique() if not self.epoch_df.empty else 0,
            "n_epochs": len(self.epoch_df),
            "n_guid_trajectories": len(self.guid_trajectories),
            "n_guid_features": len(self.guid_features_df) if self.guid_features_df is not None else 0,
            "time_range_hours": self.time_range_hours,
            "dim_reduction_method": self.dim_reduction_method,
            "n_changepoints": self.n_changepoints,
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

                # Compute per-timestep KLD mean over latent dims
                kld_t = compute_kld_per_timestep(outputs, self.warmup_steps)
                kld = kld_t.cpu().numpy() if kld_t is not None else None  # (B, T)

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
                            "t_abs_sec": float(epoch_sec) + t * TIMESTEP_SECONDS,
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

                    # Store raw data for ALL epochs per GUID
                    fhr_signal = None
                    if hasattr(batch, "fhr") and batch.fhr is not None:
                        fhr_signal = batch.fhr[idx].cpu().numpy() if hasattr(batch.fhr, 'cpu') else np.asarray(batch.fhr[idx])
                    elif hasattr(batch, "fhr_st") and batch.fhr_st is not None:
                        fhr_signal = batch.fhr_st[idx].cpu().numpy().flatten() if hasattr(batch.fhr_st, 'cpu') else np.asarray(batch.fhr_st[idx]).flatten()

                    up_signal = None
                    if hasattr(batch, "up") and batch.up is not None:
                        up_signal = batch.up[idx].cpu().numpy().flatten() if hasattr(batch.up, 'cpu') else np.asarray(batch.up[idx]).flatten()

                    if guid not in self.raw_data:
                        self.raw_data[guid] = {"epochs": [], "label": label}

                    self.raw_data[guid]["epochs"].append({
                        "latent_mean": latent_np[idx],       # (T, D)
                        "fhr": fhr_signal,
                        "up": up_signal,
                        "epoch_sec": epoch_sec,
                        "kld_timestep": kld[idx] if kld is not None else None,
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

    def _preprocess_latents(self) -> None:
        """Apply robust normalization + optional denoising to latent z-columns.

        Groups latent_df by (guid, epoch_sec), normalizes each group's z-columns
        using preprocess_latent() from metrics.py, and updates both latent_df
        and raw_data in place.
        """
        if self.latent_df is None or self.latent_df.empty:
            return

        z_cols = [f"z{d}" for d in range(self.latent_dim) if f"z{d}" in self.latent_df.columns]
        if not z_cols:
            return

        if not HAS_TORCH:
            logger.warning("torch not available, skipping latent preprocessing")
            return

        processed_count = 0

        for (guid, epoch_sec), group in self.latent_df.groupby(["guid", "epoch_sec"]):
            Z = group[z_cols].values  # (T, D)
            Z_tensor = torch.from_numpy(Z[np.newaxis, :, :]).float()  # (1, T, D)

            try:
                Z_processed = preprocess_latent(Z_tensor)  # (1, T, D)
                Z_out = Z_processed[0].numpy()  # (T, D)

                # Update latent_df in place
                for d, col in enumerate(z_cols):
                    self.latent_df.loc[group.index, col] = Z_out[:, d]

                # Update raw_data
                if guid in self.raw_data:
                    for ep in self.raw_data[guid]["epochs"]:
                        if ep.get("epoch_sec") == epoch_sec:
                            ep["latent_mean"] = Z_out
                            break

                processed_count += 1
            except Exception as e:
                logger.debug(f"Preprocessing failed for {guid} epoch {epoch_sec}: {e}")

        logger.info(f"Preprocessed latents for {processed_count} (guid, epoch) groups")

    def _get_label_name(self, batch: Any, idx: int) -> str:
        """Convert numeric label to class name."""
        label = _extract_label(batch, idx)
        if label is None or label == 0:
            return "unknown"
        if 1 <= label <= len(self.class_names):
            return self.class_names[label - 1]
        return "unknown"

    def _add_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add velocity/acceleration in latent space and KLD dynamics."""
        if df.empty:
            return df

        z_cols = [f"z{d}" for d in range(self.latent_dim) if f"z{d}" in df.columns]
        has_kld = "kld" in df.columns
        if not z_cols and not has_kld:
            return df

        def compute_dynamics(group: pd.DataFrame) -> pd.DataFrame:
            group = group.sort_values("t")
            if z_cols:
                Z = group[z_cols].values

                # Velocity (first derivative)
                dZ = np.diff(Z, axis=0, prepend=Z[[0]])
                group["speed"] = np.linalg.norm(dZ, axis=1)

                # Acceleration (second derivative)
                d2Z = np.diff(dZ, axis=0, prepend=dZ[[0]])
                group["accel"] = np.linalg.norm(d2Z, axis=1)

            if has_kld:
                kld_series = pd.Series(group["kld"].astype(float).values)
                kld_velocity = kld_series.diff().to_numpy()
                kld_accel = pd.Series(kld_velocity).diff().to_numpy()
                group["kld_velocity"] = kld_velocity
                group["kld_accel"] = kld_accel

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
        """Plot mean KLD vs hours before birth and save data to CSV."""
        if self.epoch_df.empty:
            return

        # Use the standard visualizer without class split
        epoch_df = self.epoch_df.drop(columns=["label"], errors="ignore")
        plot_kld_trajectory(epoch_df, self.output_dir / "plots")

        # Save raw epoch data as CSV for reproducibility
        csv_path = self.output_dir / "kld_trajectory_raw.csv"
        self.epoch_df.to_csv(csv_path, index=False)
        logger.info(f"Saved raw KLD trajectory data to {csv_path}")

        # Compute and save the aggregated data that's actually plotted (30-min bins)
        df_for_agg = self.epoch_df.copy()
        df_for_agg["hour_bin"] = (df_for_agg["hours_before"] * 2).round() / 2
        agg_df = df_for_agg.groupby("hour_bin")["kld_mean"].agg(["mean", "std", "count"]).reset_index()
        agg_df.columns = ["hour_bin", "kld_mean", "kld_std", "n_samples"]
        agg_csv_path = self.output_dir / "kld_trajectory_aggregated.csv"
        agg_df.to_csv(agg_csv_path, index=False)
        logger.info(f"Saved aggregated KLD trajectory data to {agg_csv_path}")

    def _plot_kld_by_class(self) -> None:
        """Plot KLD trajectories separated by class and save data to CSV."""
        if self.epoch_df.empty or "label" not in self.epoch_df.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Collect aggregated data for all classes to save to CSV
        all_class_agg: List[pd.DataFrame] = []

        for label in self.epoch_df["label"].unique():
            if label == "unknown":
                continue

            subset = self.epoch_df[self.epoch_df["label"] == label]
            subset = subset.copy()
            subset["hour_bin"] = (subset["hours_before"] * 2).round() / 2  # 30-min bins

            agg = subset.groupby("hour_bin")["kld_mean"].agg(["mean", "std", "count"]).reset_index()
            agg = agg[agg["hour_bin"].notna()]
            agg["label"] = label
            agg.columns = ["hour_bin", "kld_mean", "kld_std", "n_samples", "label"]
            all_class_agg.append(agg)

            color = self.colors.get(label, "#666666")
            ax.plot(agg["hour_bin"], agg["kld_mean"], color=color, linewidth=2, label=label, marker="o", markersize=4)
            ax.fill_between(agg["hour_bin"], agg["kld_mean"] - agg["kld_std"], agg["kld_mean"] + agg["kld_std"], alpha=0.2, color=color)

        ax.set_xlabel("Hours Before Birth")
        ax.set_ylabel("KLD (Transfer Entropy)")
        ax.set_title("Transfer Entropy by Outcome Class")
        ax.invert_xaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.output_dir / "plots" / "kld_by_class.pdf", dpi=200)
        plt.close(fig)

        # Save class-aggregated data to CSV
        if all_class_agg:
            class_agg_df = pd.concat(all_class_agg, ignore_index=True)
            csv_path = self.output_dir / "kld_trajectory_by_class.csv"
            class_agg_df.to_csv(csv_path, index=False)
            logger.info(f"Saved class-aggregated KLD trajectory data to {csv_path}")

    def _plot_kld_guid_trajectories(
        self,
        *,
        n_samples: int = 12,
        guid_list: Optional[List[str]] = None,
    ) -> None:
        """Plot per-epoch KLD mean trajectories for selected GUIDs."""
        if self.epoch_df.empty:
            return

        if guid_list:
            selected_guids = [guid for guid in guid_list if guid in self.epoch_df["guid"].unique()]
        else:
            guid_counts = self.epoch_df["guid"].value_counts()
            selected_guids = guid_counts.head(n_samples).index.tolist()

        for guid in selected_guids:
            subset = self.epoch_df[self.epoch_df["guid"] == guid]
            if subset.empty:
                continue
            plot_kld_guid_trajectory(
                subset,
                self.output_dir / "plots",
                guid=guid,
            )

    def _plot_guid_absolute_trajectories(
        self,
        *,
        n_samples: int = 12,
        guid_list: Optional[List[str]] = None,
    ) -> None:
        """Plot concatenated trajectories per GUID using absolute time ordering."""
        if self.latent_df.empty:
            return

        if guid_list:
            selected_guids = [guid for guid in guid_list if guid in self.latent_df["guid"].unique()]
        else:
            guid_counts = self.latent_df["guid"].value_counts()
            selected_guids = guid_counts.head(n_samples).index.tolist()

        if "gpc1" in self.latent_df.columns and "gpc2" in self.latent_df.columns:
            x_col, y_col = "gpc1", "gpc2"
        elif "pc1" in self.latent_df.columns and "pc2" in self.latent_df.columns:
            x_col, y_col = "pc1", "pc2"
        elif "rd1" in self.latent_df.columns and "rd2" in self.latent_df.columns:
            x_col, y_col = "rd1", "rd2"
        else:
            z_cols = [f"z{d}" for d in range(self.latent_dim) if f"z{d}" in self.latent_df.columns]
            if len(z_cols) < 2:
                return
            x_col, y_col = z_cols[:2]

        plots_dir = self.output_dir / "plots"

        for guid in selected_guids:
            subset = self.latent_df[self.latent_df["guid"] == guid]
            if subset.empty:
                continue
            plot_guid_absolute_trajectory(
                subset,
                plots_dir / f"guid_absolute_trajectory_{guid}.pdf",
                guid=guid,
                x_col=x_col,
                y_col=y_col,
                color_by="t_abs_sec",
                show_epoch_boundaries=True,
            )

    def _plot_latent_space(self, *, color_by_label: bool = True) -> None:
        """Plot 2D PCA of latent space (optionally colored by class)."""
        if self.latent_df.empty or "pc1" not in self.latent_df.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        if color_by_label and "label" in self.latent_df.columns:
            for label in self.latent_df["label"].unique():
                subset = self.latent_df[self.latent_df["label"] == label]
                color = self.colors.get(label, "#666666")
                ax.scatter(subset["pc1"], subset["pc2"], c=color, alpha=0.3, s=1, label=label)
        else:
            ax.scatter(self.latent_df["pc1"], self.latent_df["pc2"], c="#4C72B0", alpha=0.25, s=1)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Latent Space (PCA Projection)")
        if color_by_label and "label" in self.latent_df.columns:
            ax.legend(markerscale=5)

        fig.tight_layout()
        fig.savefig(self.output_dir / "plots" / "latent_space.pdf", dpi=200)
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
            ax2.scatter(latent_data["pc1"], latent_data["pc2"], c=latent_data["hours_before"], cmap="bwr", s=1, alpha=0.5)
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
        fig.savefig(self.output_dir / "dashboards" / f"{guid}.pdf", dpi=150)
        plt.close(fig)

    def _compute_metrics(self) -> Dict[str, float]:
        """Compute summary metrics."""
        metrics = {}

        if not self.epoch_df.empty:
            metrics["kld_mean"] = float(self.epoch_df["kld_mean"].mean())
            metrics["kld_std"] = float(self.epoch_df["kld_mean"].std())

        # Silhouette score for class separation (if sklearn available)
        if self.class_analysis and HAS_SKLEARN and not self.latent_df.empty and "label" in self.latent_df.columns:
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

        # Trajectory feature summary stats
        if self.guid_features_df is not None and not self.guid_features_df.empty:
            feat_cols = [c for c in self.guid_features_df.columns
                         if c not in ("guid", "label", "n_epochs", "duration_hours")]
            for col in feat_cols:
                vals = self.guid_features_df[col].replace([np.inf, -np.inf], np.nan).dropna()
                if vals.size > 0:
                    metrics[f"traj_{col}_mean"] = float(vals.mean())
                    metrics[f"traj_{col}_std"] = float(vals.std())
            metrics["n_guid_trajectories"] = len(self.guid_features_df)

        return metrics

    # -------------------------------------------------------------------------
    # New methods for advanced trajectory analysis
    # -------------------------------------------------------------------------

    def _reduce_dimensionality(self) -> None:
        """Apply dimensionality reduction globally across ALL latent points.

        Fits one reducer on all valid z-columns from self.latent_df, ensuring
        coordinates are comparable across epochs and GUIDs (required for
        trajectory stitching). For non-linear methods on large datasets,
        subsamples to fit then transforms the full set.
        """
        if self.latent_df.empty:
            return

        z_cols = [f"z{d}" for d in range(self.latent_dim) if f"z{d}" in self.latent_df.columns]
        if not z_cols:
            return

        Z = self.latent_df[z_cols].values
        valid_mask = np.all(np.isfinite(Z), axis=1)
        n_valid = int(valid_mask.sum())

        if n_valid < 10:
            logger.warning("Too few valid latent points for dimensionality reduction")
            return

        n_components = min(3, len(z_cols))
        method = self.dim_reduction_method.lower()

        try:
            if method == "pca":
                if not HAS_SKLEARN:
                    logger.warning("sklearn not available for PCA")
                    return
                reducer = PCA(n_components=n_components)
                X = np.full((len(Z), n_components), np.nan)
                X[valid_mask] = reducer.fit_transform(Z[valid_mask])
                var_explained = reducer.explained_variance_ratio_
                logger.info(f"Global PCA variance explained: {var_explained.round(3)}")
            else:
                # For non-linear methods, use reduce_latent_dimensionality
                # Subsample if too large (>50k points) for manifold methods
                Z_valid = Z[valid_mask]
                max_fit = 50000
                if Z_valid.shape[0] > max_fit:
                    rng = np.random.RandomState(42)
                    fit_idx = rng.choice(Z_valid.shape[0], max_fit, replace=False)
                    Z_fit = Z_valid[fit_idx]
                else:
                    Z_fit = Z_valid

                # reshape to (1, N, D) as expected by reduce_latent_dimensionality
                Z_3d = Z_fit[np.newaxis, :, :]
                if HAS_TORCH:
                    Z_tensor = torch.from_numpy(Z_3d).float()
                    reduced, reducer = reduce_latent_dimensionality(
                        Z_tensor,
                        method=method,
                        n_components=n_components,
                        return_reducer=True,
                    )
                    reduced_flat = reduced[0]  # (N_fit, n_components)

                    # Transform remaining points if subsampled and reducer supports it
                    if Z_valid.shape[0] > max_fit and reducer is not None and hasattr(reducer, "transform"):
                        all_reduced = reducer.transform(Z_valid)
                    elif Z_valid.shape[0] <= max_fit:
                        all_reduced = reduced_flat
                    else:
                        # Fallback: re-run on all data with PCA
                        logger.info(f"{method} does not support transform; falling back to global PCA for full set")
                        pca_fallback = PCA(n_components=n_components)
                        all_reduced = pca_fallback.fit_transform(Z_valid)
                else:
                    # Fallback to PCA without torch
                    if not HAS_SKLEARN:
                        return
                    pca_fallback = PCA(n_components=n_components)
                    all_reduced = pca_fallback.fit_transform(Z_valid)

                X = np.full((len(Z), n_components), np.nan)
                X[valid_mask] = all_reduced

            # Write results as pc1, pc2, pc3 (and gpc1, gpc2, gpc3 alias)
            for i in range(n_components):
                self.latent_df[f"pc{i + 1}"] = X[:, i]
                self.latent_df[f"gpc{i + 1}"] = X[:, i]

            logger.info(f"Applied global {method.upper()} dimensionality reduction ({n_valid} points)")

        except Exception as e:
            logger.warning(f"Global dimensionality reduction failed: {e}")
            # Fallback to per-column PCA
            self.latent_df = self._fit_pca(self.latent_df)

    def _detect_changepoints_all(self) -> None:
        """Detect changepoints for all samples with stored raw data."""
        if not HAS_CHANGEPOINT:
            logger.warning("Changepoint detection not available (ruptures not installed)")
            return

        if not self.raw_data:
            logger.debug("No raw data stored for changepoint detection")
            return

        detector = create_changepoint_detector(algo=self.changepoint_algo, model="rbf")

        for guid, data in self.raw_data.items():
            epochs = data.get("epochs", [])
            if not epochs:
                continue

            guid_cp_results = []
            for epoch_data in epochs:
                latent_mean = epoch_data.get("latent_mean")
                fhr = epoch_data.get("fhr")

                if latent_mean is None:
                    continue

                try:
                    cp_result = detect_changepoints(
                        latent_sample=latent_mean,
                        n_changepoints=self.n_changepoints,
                        decimation_factor=self.decimation_factor,
                        raw_signal=fhr,
                        detect_raw=(fhr is not None),
                        detector=detector,
                    )
                    cp_result["epoch_sec"] = epoch_data.get("epoch_sec")
                    guid_cp_results.append(cp_result)
                except Exception as e:
                    logger.debug(f"Changepoint detection failed for {guid}: {e}")

            if guid_cp_results:
                self.changepoint_results[guid] = guid_cp_results

        if self.changepoint_results:
            logger.info(f"Detected changepoints for {len(self.changepoint_results)} GUIDs")

    def _compute_segment_statistics(self) -> None:
        """Compute per-segment statistics after changepoint detection (all epochs)."""
        if not HAS_CHANGEPOINT or not self.raw_data:
            return

        all_segment_stats = []

        for guid, data in self.raw_data.items():
            epochs = data.get("epochs", [])
            cp_results = self.changepoint_results.get(guid, [])

            for epoch_idx, epoch_data in enumerate(epochs):
                latent_mean = epoch_data.get("latent_mean")
                epoch_sec = epoch_data.get("epoch_sec")

                if latent_mean is None:
                    continue

                precomputed = cp_results[epoch_idx] if epoch_idx < len(cp_results) else None

                try:
                    stats = summarize_latent_segments(
                        latent_mean=latent_mean[np.newaxis, :, :] if latent_mean.ndim == 2 else latent_mean,
                        epoch=np.array([epoch_sec]) if epoch_sec is not None else None,
                        sample_ids=[f"{guid}_ep{epoch_idx}"],
                        n_changepoints=self.n_changepoints,
                        decimation_factor=self.decimation_factor,
                        precomputed_changepoints=precomputed,
                    )
                    all_segment_stats.extend(stats)
                except Exception as e:
                    logger.debug(f"Segment statistics failed for {guid} ep{epoch_idx}: {e}")
                    continue

        self.segment_stats = all_segment_stats
        if self.segment_stats:
            logger.info(f"Computed segment statistics for {len(self.segment_stats)} epoch(s)")

    # -------------------------------------------------------------------------
    # GUID-level trajectory stitching & feature extraction
    # -------------------------------------------------------------------------

    def _build_guid_trajectories(self) -> Dict[str, Dict[str, Any]]:
        """
        Stitch all epochs from each GUID into continuous time-ordered trajectories.

        For each GUID:
        1. Extract all rows from latent_df, sort by t_abs_sec
        2. For overlapping timesteps (same t_abs_sec from 2 epochs), average latent vectors
        3. Build unified time axis, latent array, and KLD series

        Returns:
            Dict[guid] -> {
                "latent": (T_total, D), "kld": (T_total,), "time_axis": (T_total,),
                "label": str, "n_epochs": int, "epoch_boundaries": List[int]
            }
        """
        if self.latent_df is None or self.latent_df.empty:
            return {}

        z_cols = [f"z{d}" for d in range(self.latent_dim) if f"z{d}" in self.latent_df.columns]
        if not z_cols:
            return {}

        trajectories: Dict[str, Dict[str, Any]] = {}
        has_kld = "kld" in self.latent_df.columns

        for guid in self.latent_df["guid"].unique():
            subset = self.latent_df[self.latent_df["guid"] == guid].copy()
            if subset.empty:
                continue

            label = subset["label"].iloc[0]
            n_epochs = int(subset["epoch_sec"].nunique())

            # Average overlapping timesteps (same t_abs_sec from different epochs)
            agg_cols = {c: "mean" for c in z_cols}
            if has_kld:
                agg_cols["kld"] = "mean"
            # Also average PC columns if present
            for pc_col in ["pc1", "pc2", "pc3"]:
                if pc_col in subset.columns:
                    agg_cols[pc_col] = "mean"

            stitched = subset.groupby("t_abs_sec", as_index=False).agg(agg_cols)
            stitched = stitched.sort_values("t_abs_sec").reset_index(drop=True)

            latent_arr = stitched[z_cols].values  # (T_total, D)
            time_axis = stitched["t_abs_sec"].values  # (T_total,)
            kld_arr = stitched["kld"].values if has_kld else np.full(len(stitched), np.nan)

            # Compute epoch boundaries (indices where new epochs start)
            epoch_starts = sorted(subset["epoch_sec"].unique())
            epoch_boundaries = []
            for es in epoch_starts:
                first_t = subset.loc[subset["epoch_sec"] == es, "t_abs_sec"].min()
                idx_arr = np.searchsorted(time_axis, first_t)
                epoch_boundaries.append(int(idx_arr))

            trajectories[guid] = {
                "latent": latent_arr,
                "kld": kld_arr,
                "time_axis": time_axis,
                "label": label,
                "n_epochs": n_epochs,
                "epoch_boundaries": epoch_boundaries,
            }

        logger.info(f"Built stitched trajectories for {len(trajectories)} GUIDs")
        return trajectories

    def _extract_guid_features(self) -> pd.DataFrame:
        """
        For each GUID's stitched trajectory, compute shape metrics.

        Returns:
            DataFrame with columns: guid, label, n_epochs, duration_hours,
            path_length, displacement, tortuosity, mean_speed, std_speed,
            max_speed, mean_accel, mean_curvature, max_curvature, spread,
            kld_mean, kld_std, kld_slope, kld_peak, kld_range
        """
        if not self.guid_trajectories:
            return pd.DataFrame()

        rows = []
        for guid, traj_data in self.guid_trajectories.items():
            latent = traj_data["latent"]
            kld = traj_data["kld"]
            time_axis = traj_data["time_axis"]

            if latent.shape[0] < 2:
                continue

            # Compute shape metrics
            features = compute_trajectory_features(
                latent,
                kld_series=kld,
                dt=TIMESTEP_SECONDS,
            )

            duration_sec = float(time_axis[-1] - time_axis[0]) if len(time_axis) > 1 else 0.0

            row = {
                "guid": guid,
                "label": traj_data["label"],
                "n_epochs": traj_data["n_epochs"],
                "duration_hours": duration_sec / 3600.0,
            }
            row.update(features)
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            logger.info(f"Extracted trajectory features for {len(df)} GUIDs")
        return df

    def _export_csvs(self) -> None:
        """Export trajectory data as CSV files for external analysis.

        Produces:
        1. guid_trajectory_features.csv — One row per GUID with shape metrics + label
        2. latent_trajectories_stitched.csv — Per-timestep stitched data
        """
        # 1. Per-GUID features
        if self.guid_features_df is not None and not self.guid_features_df.empty:
            feat_path = self.output_dir / "guid_trajectory_features.csv"
            self.guid_features_df.to_csv(feat_path, index=False)
            logger.info(f"Exported GUID features to {feat_path}")

        # 2. Stitched per-timestep data
        if self.guid_trajectories:
            z_cols = [f"z{d}" for d in range(self.latent_dim)]
            rows = []
            for guid, traj_data in self.guid_trajectories.items():
                latent = traj_data["latent"]
                kld = traj_data["kld"]
                time_axis = traj_data["time_axis"]
                label = traj_data["label"]

                for i in range(latent.shape[0]):
                    row = {
                        "guid": guid,
                        "label": label,
                        "t_abs_sec": float(time_axis[i]),
                        "hours_before_birth": abs(float(time_axis[i])) / 3600.0,
                        "kld": float(kld[i]) if np.isfinite(kld[i]) else np.nan,
                    }
                    for d in range(min(self.latent_dim, latent.shape[1])):
                        row[f"z{d}"] = float(latent[i, d])

                    # Add PC coordinates if available in latent_df
                    rows.append(row)

            if rows:
                stitched_df = pd.DataFrame(rows)

                # Merge PC coordinates from latent_df if available
                if self.latent_df is not None and "pc1" in self.latent_df.columns:
                    pc_cols = [c for c in ["pc1", "pc2", "pc3"] if c in self.latent_df.columns]
                    # Get PC values by matching guid + t_abs_sec
                    pc_data = self.latent_df.groupby(["guid", "t_abs_sec"], as_index=False)[pc_cols].mean()
                    stitched_df = stitched_df.merge(pc_data, on=["guid", "t_abs_sec"], how="left")

                stitched_path = self.output_dir / "latent_trajectories_stitched.csv"
                stitched_df.to_csv(stitched_path, index=False)
                logger.info(f"Exported stitched trajectories to {stitched_path} ({len(stitched_df)} rows)")

        # 3. Export NPY arrays for each GUID trajectory
        if self.guid_trajectories:
            npy_dir = self.output_dir / "npy"
            npy_dir.mkdir(exist_ok=True)
            for guid, traj_data in self.guid_trajectories.items():
                np.save(npy_dir / f"{guid}_latent.npy", traj_data["latent"])
                np.save(npy_dir / f"{guid}_kld.npy", traj_data["kld"])
                np.save(npy_dir / f"{guid}_time.npy", traj_data["time_axis"])
            logger.info(f"Exported NPY arrays for {len(self.guid_trajectories)} GUIDs to {npy_dir}")

    def _plot_recurrence_samples(self, n_samples: int = 5) -> None:
        """Generate recurrence plots for representative GUID trajectories."""
        if not self.guid_trajectories:
            return

        plots_dir = self.output_dir / "plots"

        # Select GUIDs with most timesteps
        guid_sizes = {g: d["latent"].shape[0] for g, d in self.guid_trajectories.items()}
        top_guids = sorted(guid_sizes, key=guid_sizes.get, reverse=True)[:n_samples]

        for guid in top_guids:
            traj_data = self.guid_trajectories[guid]
            latent = traj_data["latent"]
            if latent.shape[0] < 10:
                continue

            try:
                plot_recurrence(
                    latent,
                    plots_dir / f"recurrence_{guid}.pdf",
                    sample_id=guid,
                )
            except Exception as e:
                logger.debug(f"Recurrence plot failed for {guid}: {e}")

        logger.info(f"Generated recurrence plots for {min(n_samples, len(top_guids))} GUIDs")

    def _plot_3d_trajectories(self, n_samples: int = 5) -> None:
        """Generate 3D trajectory plots for selected samples."""
        if self.latent_df.empty:
            return

        # Get samples with most time points
        sample_counts = self.latent_df.groupby(["guid", "epoch_sec"]).size()
        top_samples = sample_counts.nlargest(n_samples).index.tolist()

        plots_dir = self.output_dir / "plots"

        for guid, epoch in top_samples:
            mask = (self.latent_df["guid"] == guid) & (self.latent_df["epoch_sec"] == epoch)
            subset = self.latent_df[mask].sort_values("t")

            if len(subset) < 5:
                continue

            # Try reduced dimensions first, fallback to z0, z1, z2
            if "rd1" in subset.columns:
                traj = subset[["rd1", "rd2", "rd3"]].values
            elif "pc1" in subset.columns:
                traj = subset[["pc1", "pc2", "pc3"]].values if "pc3" in subset.columns else subset[["pc1", "pc2"]].values
            else:
                z_cols = [f"z{d}" for d in range(3) if f"z{d}" in subset.columns]
                if len(z_cols) < 2:
                    continue
                traj = subset[z_cols].values

            sample_id = f"{guid}_{int(epoch)}"

            # Static 2D plot
            if traj.shape[1] >= 2:
                try:
                    plot_latent_trajectory_2d(
                        traj[:, :2],
                        plots_dir / f"trajectory_2d_{sample_id}.pdf",
                        sample_id=sample_id,
                    )
                except Exception as e:
                    logger.debug(f"2D trajectory plot failed: {e}")

            # Static 3D plot
            if traj.shape[1] >= 3:
                try:
                    plot_latent_trajectory_3d(
                        traj[:, :3],
                        plots_dir / f"trajectory_3d_{sample_id}.pdf",
                        sample_id=sample_id,
                    )
                except Exception as e:
                    logger.debug(f"3D trajectory plot failed: {e}")

            # Interactive 3D plot
            if HAS_INTERACTIVE and traj.shape[1] >= 3:
                try:
                    plot_latent_trajectory_3d_interactive(
                        traj[:, :3],
                        plots_dir / f"trajectory_3d_{sample_id}.html",
                        sample_id=sample_id,
                    )
                except Exception as e:
                    logger.debug(f"Interactive 3D trajectory plot failed: {e}")

        logger.info(f"Generated 3D trajectory plots for {min(n_samples, len(top_samples))} samples")

    def _plot_kld_3d_trajectories(self, n_samples: int = 5) -> None:
        """Generate 3D KLD trajectory plots for selected samples."""
        if self.latent_df.empty or "kld_velocity" not in self.latent_df.columns:
            return

        sample_counts = self.latent_df.groupby(["guid", "epoch_sec"]).size()
        top_samples = sample_counts.nlargest(n_samples).index.tolist()

        plots_dir = self.output_dir / "plots"

        for guid, epoch in top_samples:
            mask = (self.latent_df["guid"] == guid) & (self.latent_df["epoch_sec"] == epoch)
            subset = self.latent_df[mask].sort_values("t")

            if len(subset) < 5:
                continue

            traj_df = subset[["kld", "kld_velocity", "kld_accel"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(traj_df) < 5:
                continue

            traj = traj_df.values
            sample_id = f"{guid}_{int(epoch)}"

            try:
                plot_kld_trajectory_3d(
                    traj,
                    plots_dir / f"kld_trajectory_3d_{sample_id}.pdf",
                    sample_id=sample_id,
                )
            except Exception as e:
                logger.debug(f"KLD 3D trajectory plot failed: {e}")

            if HAS_INTERACTIVE:
                try:
                    plot_kld_trajectory_3d_interactive(
                        traj,
                        plots_dir / f"kld_trajectory_3d_{sample_id}.html",
                        sample_id=sample_id,
                    )
                except Exception as e:
                    logger.debug(f"Interactive KLD 3D trajectory plot failed: {e}")

        logger.info(f"Generated KLD 3D trajectory plots for {min(n_samples, len(top_samples))} samples")

    def _plot_guid_3d_trajectories(
        self, n_samples: int = 12, guid_list: Optional[List[str]] = None,
    ) -> None:
        """Generate 3D trajectory plots for stitched GUID-level trajectories."""
        if not self.guid_trajectories:
            return

        plots_dir = self.output_dir / "plots"

        # Select GUIDs
        if guid_list:
            selected = [g for g in guid_list if g in self.guid_trajectories]
        else:
            guid_sizes = {g: d["latent"].shape[0] for g, d in self.guid_trajectories.items()}
            selected = sorted(guid_sizes, key=guid_sizes.get, reverse=True)[:n_samples]

        for guid in selected:
            traj_data = self.guid_trajectories[guid]
            time_axis = traj_data["time_axis"]
            epoch_boundaries = traj_data.get("epoch_boundaries", [])

            # Build 3D coordinates from globally fitted PCs
            subset = self.latent_df[self.latent_df["guid"] == guid].copy()
            if subset.empty or "pc1" not in subset.columns or "pc3" not in subset.columns:
                continue

            # Average overlapping timesteps, sort by t_abs_sec
            stitched = (
                subset.groupby("t_abs_sec", as_index=False)[["pc1", "pc2", "pc3"]].mean()
                .sort_values("t_abs_sec")
                .reset_index(drop=True)
            )

            if len(stitched) < 5:
                continue

            traj_3d = stitched[["pc1", "pc2", "pc3"]].values
            t_axis = stitched["t_abs_sec"].values

            # Static matplotlib plot
            try:
                from model.vae_teb_prediction.testing.visualizers import plot_guid_trajectory_3d
                plot_guid_trajectory_3d(
                    traj_3d,
                    plots_dir / f"guid_trajectory_3d_{guid}.pdf",
                    sample_id=guid,
                    time_axis=t_axis,
                    epoch_boundaries=epoch_boundaries,
                )
            except Exception as e:
                logger.debug(f"GUID 3D static plot failed for {guid}: {e}")

            # Interactive Plotly plot
            if HAS_INTERACTIVE:
                try:
                    from model.vae_teb_prediction.testing.visualizers_interactive import (
                        plot_guid_trajectory_3d_interactive,
                    )
                    plot_guid_trajectory_3d_interactive(
                        traj_3d,
                        plots_dir / f"guid_trajectory_3d_{guid}.html",
                        sample_id=guid,
                        time_axis=t_axis,
                        epoch_boundaries=epoch_boundaries,
                    )
                except Exception as e:
                    logger.debug(f"GUID 3D interactive plot failed for {guid}: {e}")

            # Animation (optional)
            if self.plot_animations and HAS_INTERACTIVE:
                try:
                    animations_dir = self.output_dir / "animations"
                    animations_dir.mkdir(exist_ok=True)
                    plot_trajectory_animation(
                        traj_3d,
                        animations_dir / f"guid_trajectory_{guid}.gif",
                        sample_id=guid,
                        fps=15,
                        duration_seconds=5.0,
                    )
                except Exception as e:
                    logger.debug(f"GUID animation failed for {guid}: {e}")

        logger.info(f"Generated GUID-level 3D trajectory plots for {len(selected)} GUIDs")

    def _plot_fhr_up_timelines(
        self, n_samples: int = 5, guid_list: Optional[List[str]] = None,
    ) -> None:
        """Generate interactive FHR+UP timelines for selected GUIDs."""
        if not HAS_INTERACTIVE or not self.raw_data:
            return

        plots_dir = self.output_dir / "plots"

        # Select GUIDs with most epochs
        if guid_list:
            selected = [g for g in guid_list if g in self.raw_data]
        else:
            guid_epoch_counts = {g: len(d["epochs"]) for g, d in self.raw_data.items()}
            selected = sorted(guid_epoch_counts, key=guid_epoch_counts.get, reverse=True)[:n_samples]

        for guid in selected:
            data = self.raw_data[guid]
            epochs = data.get("epochs", [])
            if not epochs:
                continue

            # Sort epochs by epoch_sec
            sorted_epochs = sorted(epochs, key=lambda e: e.get("epoch_sec", 0))

            # Stack FHR and UP arrays
            fhr_list = []
            up_list = []
            epoch_secs = []
            has_up = False

            for ep in sorted_epochs:
                fhr = ep.get("fhr")
                if fhr is None:
                    continue
                fhr_flat = fhr.flatten()
                fhr_list.append(fhr_flat)
                epoch_secs.append(ep.get("epoch_sec", 0))

                up = ep.get("up")
                if up is not None:
                    up_list.append(up.flatten())
                    has_up = True
                else:
                    up_list.append(np.zeros_like(fhr_flat))

            if not fhr_list:
                continue

            # Ensure consistent shape by padding/truncating to max length
            max_len = max(f.shape[0] for f in fhr_list)
            fhr_arr = np.zeros((len(fhr_list), max_len))
            up_arr = np.zeros((len(up_list), max_len)) if has_up else None

            for i, f in enumerate(fhr_list):
                fhr_arr[i, :f.shape[0]] = f
            if has_up:
                for i, u in enumerate(up_list):
                    up_arr[i, :u.shape[0]] = u

            epoch_arr = np.array(epoch_secs)

            # Gather changepoint times if available
            cp_times = None
            cp_results = self.changepoint_results.get(guid, [])
            if cp_results:
                cp_time_list = []
                for cp_r in cp_results:
                    ep_sec = cp_r.get("epoch_sec", 0) or 0
                    raw_cps = cp_r.get("raw_changepoints", np.array([]))
                    if len(raw_cps) > 0:
                        # Convert raw indices to minutes
                        for rc in raw_cps:
                            cp_min = (ep_sec + rc / 4.0) / 60.0
                            cp_time_list.append(cp_min)
                if cp_time_list:
                    cp_times = np.array(cp_time_list)

            try:
                from model.vae_teb_prediction.testing.visualizers_interactive import (
                    plot_fhr_up_timeline,
                )
                plot_fhr_up_timeline(
                    fhr=fhr_arr,
                    up=up_arr,
                    epoch=epoch_arr,
                    output_path=plots_dir / f"fhr_up_timeline_{guid}.html",
                    sample_id=guid,
                    changepoint_times=cp_times,
                )
            except Exception as e:
                logger.debug(f"FHR+UP timeline failed for {guid}: {e}")

        logger.info(f"Generated FHR+UP timelines for {len(selected)} GUIDs")

    def _plot_changepoints_with_raw(self, n_samples: int = 5) -> None:
        """Plot changepoints overlaid on raw FHR signals (all epochs)."""
        if not self.changepoint_results or not self.raw_data:
            return

        cp_dir = self.output_dir / "changepoint_analysis"
        samples_plotted = 0

        for guid, cp_results_list in self.changepoint_results.items():
            if samples_plotted >= n_samples:
                break

            data = self.raw_data.get(guid)
            if data is None:
                continue

            epochs = data.get("epochs", [])
            if not epochs:
                continue

            for epoch_idx, epoch_data in enumerate(epochs):
                if samples_plotted >= n_samples:
                    break

                latent_mean = epoch_data.get("latent_mean")
                fhr = epoch_data.get("fhr")

                if latent_mean is None or fhr is None:
                    continue

                cp_result = cp_results_list[epoch_idx] if epoch_idx < len(cp_results_list) else None
                if cp_result is None:
                    continue

                try:
                    plot_latent_changepoints_with_raw(
                        latent_mean=latent_mean,
                        fhr=fhr,
                        changepoint_results=cp_result,
                        output_path=cp_dir / f"changepoints_{guid}_ep{epoch_idx}.pdf",
                        sample_id=f"{guid}_ep{epoch_idx}",
                        decimation_factor=self.decimation_factor,
                    )
                    samples_plotted += 1
                except Exception as e:
                    logger.debug(f"Changepoint plot failed for {guid} ep{epoch_idx}: {e}")

        if samples_plotted > 0:
            logger.info(f"Generated changepoint plots for {samples_plotted} epoch(s)")

    def _plot_segment_stats(self) -> None:
        """Plot aggregated segment statistics."""
        if not self.segment_stats:
            return

        try:
            df = plot_segment_statistics(
                self.segment_stats,
                self.output_dir / "changepoint_analysis",
                filename_prefix="segment",
            )
            if df is not None:
                df.to_csv(self.output_dir / "segment_stats.csv", index=False)
                logger.info("Generated segment statistics plots")
        except Exception as e:
            logger.warning(f"Segment statistics plotting failed: {e}")

    def _generate_trajectory_animations(self, n_samples: int = 3) -> None:
        """Generate animated trajectory GIFs."""
        if not HAS_INTERACTIVE:
            return

        if self.latent_df.empty:
            return

        animations_dir = self.output_dir / "animations"

        sample_counts = self.latent_df.groupby(["guid", "epoch_sec"]).size()
        top_samples = sample_counts.nlargest(n_samples).index.tolist()

        for guid, epoch in top_samples:
            mask = (self.latent_df["guid"] == guid) & (self.latent_df["epoch_sec"] == epoch)
            subset = self.latent_df[mask].sort_values("t")

            if len(subset) < 10:
                continue

            # Get trajectory coordinates
            if "rd1" in subset.columns:
                traj = subset[["rd1", "rd2", "rd3"]].values
            elif "pc1" in subset.columns and "pc3" in subset.columns:
                traj = subset[["pc1", "pc2", "pc3"]].values
            elif "pc1" in subset.columns:
                traj = subset[["pc1", "pc2"]].values
            else:
                z_cols = [f"z{d}" for d in range(3) if f"z{d}" in subset.columns]
                if len(z_cols) < 2:
                    continue
                traj = subset[z_cols].values

            sample_id = f"{guid}_{int(epoch)}"

            try:
                success = plot_trajectory_animation(
                    traj,
                    animations_dir / f"trajectory_{sample_id}.gif",
                    sample_id=sample_id,
                    fps=15,
                    duration_seconds=5.0,
                )
                if success:
                    logger.debug(f"Generated animation for {sample_id}")
            except Exception as e:
                logger.debug(f"Animation generation failed for {sample_id}: {e}")

        logger.info(f"Generated trajectory animations")

    def _compare_class_trajectories(self) -> None:
        """Compare trajectories across outcome classes."""
        if not HAS_INTERACTIVE or self.latent_df.empty or "label" not in self.latent_df.columns:
            return

        # Group trajectories by class
        trajectories_by_class: Dict[str, List[np.ndarray]] = defaultdict(list)

        for label in self.latent_df["label"].unique():
            if label == "unknown":
                continue

            subset = self.latent_df[self.latent_df["label"] == label]
            unique_samples = subset[["guid", "epoch_sec"]].drop_duplicates()

            for _, row in unique_samples.head(5).iterrows():  # Max 5 per class
                guid, epoch = row["guid"], row["epoch_sec"]
                mask = (subset["guid"] == guid) & (subset["epoch_sec"] == epoch)
                sample = subset[mask].sort_values("t")

                if len(sample) < 5:
                    continue

                # Get coordinates
                if "rd1" in sample.columns:
                    traj = sample[["rd1", "rd2", "rd3"]].values
                elif "pc1" in sample.columns and "pc3" in sample.columns:
                    traj = sample[["pc1", "pc2", "pc3"]].values
                else:
                    continue

                trajectories_by_class[label].append(traj)

        if not trajectories_by_class:
            return

        # Convert lists to arrays for plotting
        trajectories_dict = {}
        for label, traj_list in trajectories_by_class.items():
            if traj_list:
                # Stack first trajectory for each class (for simple comparison)
                trajectories_dict[label] = traj_list[0]

        try:
            # Static comparison
            plot_trajectory_comparison(
                trajectories_dict,
                self.output_dir / "plots",
                n_components=3,
                filename="trajectory_class_comparison.pdf",
            )

            # Interactive comparison
            plot_trajectory_comparison_interactive(
                trajectories_dict,
                self.output_dir / "plots" / "trajectory_class_comparison.html",
                title="Trajectory Comparison by Outcome Class",
                n_components=3,
            )
            logger.info("Generated class trajectory comparison plots")
        except Exception as e:
            logger.warning(f"Class trajectory comparison failed: {e}")


def run_trajectory_analysis(
    runner: TestRunner,
    loader: Any,
    time_range_hours: float = 12.0,
    min_epochs_per_guid: int = 3,
    skip_dashboards: bool = False,
    n_dashboards: int = 12,
    class_analysis: bool = False,
    dim_reduction_method: str = "pca",
    n_changepoints: int = 5,
    plot_3d: bool = True,
    plot_animations: bool = False,
    n_trajectory_samples: int = 5,
    n_kld_guid_plots: int = 12,
    kld_guid_list: Optional[List[str]] = None,
    decimation_factor: int = 16,
    changepoint_algo: str = "pelt",
    preprocess_latent_trajectories: bool = False,
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
        class_analysis: If True, include class-based plots/analysis.
        dim_reduction_method: Dimensionality reduction method
            ('pca', 'umap', 'tsne', 'isomap', 'diffusion').
        n_changepoints: Number of changepoints to detect per sample.
        plot_3d: Whether to generate 3D trajectory plots.
        plot_animations: Whether to generate animated trajectory GIFs.
        n_trajectory_samples: Number of samples for trajectory plots.
        n_kld_guid_plots: Number of GUIDs to plot for per-epoch KLD trends.
        kld_guid_list: Optional list of GUIDs to plot (overrides top-N selection).
        decimation_factor: Ratio between raw signal and latent lengths.
        changepoint_algo: Changepoint detection algorithm
            ('pelt', 'binseg', 'bottomup', 'window', 'dynp', 'gradient').
        preprocess_latent_trajectories: If True, apply robust normalization
            and optional denoising to latent z-columns before analysis.

    Returns:
        Summary dict with statistics and output paths.

    Example:
        >>> results = run_trajectory_analysis(runner, test_loader)
        >>> print(f"Analyzed {results['n_guids']} patients")

        >>> # With advanced options
        >>> results = run_trajectory_analysis(
        ...     runner, test_loader,
        ...     dim_reduction_method='umap',
        ...     n_changepoints=5,
        ...     plot_animations=True,
        ...     changepoint_algo='binseg',
        ... )
    """
    output_dir = runner.ensure_dir("trajectory")

    analyzer = TrajectoryAnalyzer(
        runner=runner,
        loader=loader,
        output_dir=output_dir,
        time_range_hours=time_range_hours,
        min_epochs_per_guid=min_epochs_per_guid,
        dim_reduction_method=dim_reduction_method,
        n_changepoints=n_changepoints,
        plot_3d=plot_3d,
        plot_animations=plot_animations,
        decimation_factor=decimation_factor,
        changepoint_algo=changepoint_algo,
        preprocess_latent_trajectories=preprocess_latent_trajectories,
    )

    return analyzer.run(
        skip_dashboards=skip_dashboards,
        n_dashboards=n_dashboards,
        class_analysis=class_analysis,
        n_trajectory_samples=n_trajectory_samples,
        n_kld_guid_plots=n_kld_guid_plots,
        kld_guid_list=kld_guid_list,
    )
