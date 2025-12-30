"""
Trajectory and KLD Analysis for VAE-TEB Model.

Analyzes how latent representations and KL divergence evolve over time before birth.
Designed for research use - simple, readable, easy to modify.

Usage:
    analyzer = TrajectoryAnalyzer(model, dataloader, output_dir)
    results = analyzer.run()
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from loguru import logger

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

plt.switch_backend("Agg")

# Constants
TIMESTEP_SECONDS = 4.0  # Each latent step = 4 seconds (4 Hz sampling, T=16 decimation)
WARMUP_STEPS = 30  # First 30 timesteps are warmup (masked during training)


class TrajectoryAnalyzer:
    """
    Analyzes latent trajectories and KLD over time before birth.

    Collects per-timestep latent codes and KLD from model, organizes by GUID
    and time-to-birth, generates visualizations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader,
        output_dir: str | Path,
        time_range_hours: Optional[float] = 12.0,
        min_epochs_per_guid: int = 3,
        latent_dim: int = 16,
        class_names: List[str] = None,
    ):
        """
        Args:
            model: VAE-TEB model with kld_tensor() method
            dataloader: Provides batches with fhr_st, fhr_ph, fhr_up_ph, guid, epoch
            output_dir: Where to save outputs
            time_range_hours: Analyze this many hours before birth (None = all)
            min_epochs_per_guid: Minimum epochs per GUID to include
            latent_dim: Latent space dimensionality
            class_names: Names for outcome classes
        """
        self.model = model
        self.dataloader = dataloader
        self.output_dir = Path(output_dir)
        self.time_range_hours = time_range_hours
        self.min_epochs_per_guid = min_epochs_per_guid
        self.latent_dim = int(getattr(model, "latent_dim_z", latent_dim))
        self.warmup_steps = int(getattr(model, "warmup_period", WARMUP_STEPS))
        self.class_names = class_names or ["healthy", "acidosis", "HIE"]

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "dashboards").mkdir(exist_ok=True)

        # Results storage
        self.latent_df: Optional[pd.DataFrame] = None
        self.epoch_df: Optional[pd.DataFrame] = None

        # Class colors for plots
        self.colors = {"healthy": "#2ecc71", "acidosis": "#e74c3c", "HIE": "#9b59b6", "unknown": "#95a5a6"}

    def run(self, skip_dashboards: bool = False, n_dashboards: int = 12) -> Dict[str, Any]:
        """
        Run complete analysis pipeline.

        Returns dict with summary statistics and paths to outputs.
        """
        logger.info("Starting trajectory analysis...")

        # Step 1: Collect latents and KLD from model
        self.latent_df, self.epoch_df = self._collect_data()

        if self.latent_df.empty:
            logger.warning("No data collected!")
            return {"status": "empty"}

        # Step 2: Add dynamics (velocity, acceleration in latent space)
        self.latent_df = self._add_dynamics(self.latent_df)

        # Step 3: Fit PCA for visualization
        self.latent_df = self._fit_pca(self.latent_df)

        # Step 4: Save data
        self.latent_df.to_parquet(self.output_dir / "latent_trajectories.parquet")
        self.epoch_df.to_parquet(self.output_dir / "epoch_summary.parquet")

        # Step 5: Generate plots
        self._plot_kld_vs_time()
        self._plot_kld_by_class()
        self._plot_kld_distributions()
        self._plot_latent_space()
        self._plot_dynamics()

        # Step 6: Generate per-GUID dashboards
        if not skip_dashboards:
            self._generate_dashboards(n_dashboards)

        # Step 7: Compute metrics
        metrics = self._compute_metrics()

        # Save summary
        summary = {
            "n_samples": len(self.latent_df),
            "n_guids": self.epoch_df["guid"].nunique(),
            "n_epochs": len(self.epoch_df),
            "time_range_hours": self.time_range_hours,
            "metrics": metrics,
        }

        import json
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Analysis complete: {summary['n_guids']} GUIDs, {summary['n_epochs']} epochs")
        return summary

    def _collect_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Collect latent trajectories and KLD from dataloader."""

        latent_rows = []
        epoch_rows = []
        guid_epoch_count = defaultdict(int)

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Collecting data"):
                # Get inputs
                y_st = getattr(batch, "fhr_st", None)
                y_ph = getattr(batch, "fhr_ph", None)
                x_ph = getattr(batch, "fhr_up_ph", None)

                if y_st is None or y_ph is None or x_ph is None:
                    continue

                B = y_st.shape[0]

                # Forward pass
                outputs = self.model(
                    y_st=y_st.to(self.device),
                    y_ph=y_ph.to(self.device),
                    x_ph=x_ph.to(self.device),
                )

                # Extract latents (use mu_post for deterministic representation)
                latent = outputs.get("mu_post", outputs.get("z")).cpu().numpy()  # (B, T, D)

                # Compute KLD per timestep
                kld = self._compute_kld(outputs)  # (B, T)

                # Compute uncertainty (posterior variance)
                uncertainty = None
                if "logvar_post" in outputs:
                    uncertainty = outputs["logvar_post"].exp().sum(dim=-1).cpu().numpy()  # (B, T)

                # Process each sample in batch
                for i in range(B):
                    guid = self._get_guid(batch, i)
                    epoch_sec = self._get_epoch(batch, i)
                    label = self._get_label(batch, i)

                    hours_before = abs(epoch_sec) / 3600.0

                    # Filter by time range
                    if self.time_range_hours and hours_before > self.time_range_hours:
                        continue

                    guid_epoch_count[guid] += 1
                    T = latent.shape[1]

                    # Per-timestep data
                    for t in range(T):
                        row = {
                            "guid": guid,
                            "epoch_sec": epoch_sec,
                            "hours_before": hours_before,
                            "label": label,
                            "t": t,
                            "t_sec": t * TIMESTEP_SECONDS,
                            "kld": kld[i, t] if kld is not None else np.nan,
                        }

                        # Add latent dimensions
                        for d in range(self.latent_dim):
                            row[f"z{d}"] = latent[i, t, d]

                        if uncertainty is not None:
                            row["uncertainty"] = uncertainty[i, t]

                        latent_rows.append(row)

                    # Per-epoch summary
                    valid_kld = kld[i, self.warmup_steps:] if kld is not None else None
                    epoch_rows.append({
                        "guid": guid,
                        "epoch_sec": epoch_sec,
                        "hours_before": hours_before,
                        "label": label,
                        "kld_mean": np.nanmean(valid_kld) if valid_kld is not None else np.nan,
                        "kld_std": np.nanstd(valid_kld) if valid_kld is not None else np.nan,
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

        logger.info(f"Collected {len(epoch_df)} epochs from {len(valid_guids)} GUIDs")
        return latent_df, epoch_df

    def _compute_kld(self, outputs: Dict[str, torch.Tensor]) -> Optional[np.ndarray]:
        """Compute per-timestep KLD from model outputs."""
        mu_prior = outputs.get("mu_prior")
        logvar_prior = outputs.get("logvar_prior")
        mu_post = outputs.get("mu_post")
        logvar_post = outputs.get("logvar_post")

        if any(x is None for x in [mu_prior, logvar_prior, mu_post, logvar_post]):
            return None

        # KL(q || p) = 0.5 * sum[log(var_p/var_q) + (var_q + (mu_q - mu_p)^2)/var_p - 1]
        kld = 0.5 * (
            logvar_prior - logvar_post +
            (logvar_post.exp() + (mu_post - mu_prior)**2) / logvar_prior.exp() - 1.0
        )

        # Sum over latent dimensions
        kld = kld.sum(dim=-1).cpu().numpy()  # (B, T)

        # Mask warmup period
        kld[:, :self.warmup_steps] = np.nan

        return kld

    def _get_guid(self, batch, idx: int) -> str:
        """Extract GUID from batch."""
        guid = getattr(batch, "guid", None)
        if guid is None:
            return f"sample_{idx}"
        val = guid[idx]
        if isinstance(val, bytes):
            return val.decode("utf-8")
        if isinstance(val, torch.Tensor):
            return str(val.item())
        return str(val)

    def _get_epoch(self, batch, idx: int) -> float:
        """Extract epoch (seconds before birth) from batch."""
        epoch = getattr(batch, "epoch", None)
        if epoch is None:
            return 0.0
        val = epoch[idx]
        if isinstance(val, torch.Tensor):
            return float(val.item())
        return float(val)

    def _get_label(self, batch, idx: int) -> str:
        """Extract class label from batch."""
        target = getattr(batch, "target", None)
        if target is None:
            return "unknown"

        val = target[idx]
        if isinstance(val, torch.Tensor):
            val = val.cpu().numpy()
        weight = getattr(batch, "weight", None)
        if weight is not None:
            weight = weight[idx]
            if isinstance(weight, torch.Tensor):
                weight = weight.cpu().numpy()

        # Handle different label formats
        if isinstance(val, np.ndarray):
            if val.ndim == 0:
                label_idx = int(np.round(float(val)))
            elif val.size <= len(self.class_names):
                label_idx = int(np.argmax(val))
            else:
                if isinstance(weight, np.ndarray) and weight.shape == val.shape:
                    masked = val[weight > 0]
                else:
                    masked = val[val != 0]
                if masked.size == 0:
                    label_idx = -1
                else:
                    label_idx = int(np.round(float(np.nanmax(masked))))
        else:
            label_idx = int(np.round(float(val)))

        if 1 <= label_idx <= len(self.class_names):
            return self.class_names[label_idx - 1]
        if 0 <= label_idx < len(self.class_names):
            return self.class_names[label_idx]
        return "unknown"

    def _add_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add velocity and acceleration in latent space."""
        if df.empty:
            return df

        z_cols = [f"z{d}" for d in range(self.latent_dim)]

        def compute_dynamics(group):
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
        """Fit PCA and add PC columns."""
        if df.empty or not HAS_SKLEARN:
            return df

        z_cols = [f"z{d}" for d in range(self.latent_dim)]
        Z = df[z_cols].values

        pca = PCA(n_components=n_components)
        X = pca.fit_transform(Z)

        for i in range(n_components):
            df[f"pc{i+1}"] = X[:, i]

        logger.info(f"PCA variance explained: {pca.explained_variance_ratio_}")
        return df

    # ==================== PLOTTING METHODS ====================

    def _plot_kld_vs_time(self):
        """Plot mean KLD vs hours before birth."""
        if self.epoch_df.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Aggregate by time bins
        df = self.epoch_df.copy()
        df["time_bin"] = (df["hours_before"] // 1).astype(int)  # 1-hour bins

        agg = df.groupby("time_bin")["kld_mean"].agg(["mean", "std", "count"])
        agg = agg[agg["count"] >= 5]  # Require at least 5 samples

        x = agg.index.values
        ax.plot(x, agg["mean"], "b-", linewidth=2, label="Mean KLD")
        ax.fill_between(x, agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                       alpha=0.3, color="blue")

        ax.set_xlabel("Hours Before Birth", fontsize=12)
        ax.set_ylabel("KLD (Transfer Entropy)", fontsize=12)
        ax.set_title("KLD vs Time Before Birth", fontsize=14)
        ax.invert_xaxis()  # Birth at right
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(self.output_dir / "plots" / "kld_vs_time.png", dpi=150)
        plt.close(fig)

    def _plot_kld_by_class(self):
        """Plot KLD over time, separated by class."""
        if self.epoch_df.empty or "label" not in self.epoch_df:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        for label in self.epoch_df["label"].unique():
            if label == "unknown":
                continue

            df = self.epoch_df[self.epoch_df["label"] == label]
            df = df.copy()
            df["time_bin"] = (df["hours_before"] // 1).astype(int)

            agg = df.groupby("time_bin")["kld_mean"].agg(["mean", "std"])

            color = self.colors.get(label, "#333333")
            ax.plot(agg.index, agg["mean"], color=color, linewidth=2, label=label)
            ax.fill_between(agg.index, agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                          alpha=0.2, color=color)

        ax.set_xlabel("Hours Before Birth", fontsize=12)
        ax.set_ylabel("KLD", fontsize=12)
        ax.set_title("KLD by Outcome Class", fontsize=14)
        ax.invert_xaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.output_dir / "plots" / "kld_by_class.png", dpi=150)
        plt.close(fig)

    def _plot_kld_distributions(self):
        """Plot KLD distributions for different time windows."""
        if self.epoch_df.empty:
            return

        # Create time bins dynamically based on data range
        df = self.epoch_df.copy()
        max_hours = df["hours_before"].max()
        bin_size = 2  # 2-hour bins
        n_bins = int(np.ceil(max_hours / bin_size))
        n_bins = max(1, min(n_bins, 12))  # Limit to 12 bins max

        bins = [i * bin_size for i in range(n_bins + 1)]
        labels = [f"{bins[i]}-{bins[i+1]}h" for i in range(n_bins)]

        df["time_window"] = pd.cut(df["hours_before"], bins=bins, labels=labels)
        df = df.dropna(subset=["time_window", "kld_mean"])

        if df.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        if HAS_SEABORN:
            sns.violinplot(data=df, x="time_window", y="kld_mean", ax=ax, palette="viridis")
        else:
            df.boxplot(column="kld_mean", by="time_window", ax=ax)
            plt.suptitle("")

        ax.set_xlabel("Time Before Birth", fontsize=12)
        ax.set_ylabel("KLD", fontsize=12)
        ax.set_title("KLD Distribution by Time Window", fontsize=14)

        fig.tight_layout()
        fig.savefig(self.output_dir / "plots" / "kld_distributions.png", dpi=150)
        plt.close(fig)

    def _plot_latent_space(self):
        """Plot latent space (PCA) colored by time and class."""
        if self.latent_df.empty or "pc1" not in self.latent_df:
            return

        # Sample for speed
        df = self.latent_df.sample(min(20000, len(self.latent_df)), random_state=42)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Color by time
        scatter1 = axes[0].scatter(df["pc1"], df["pc2"], c=df["hours_before"],
                                   s=3, alpha=0.5, cmap="viridis")
        plt.colorbar(scatter1, ax=axes[0], label="Hours Before Birth")
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        axes[0].set_title("Latent Space (colored by time)")

        # Color by class
        for label in df["label"].unique():
            if label == "unknown":
                continue
            mask = df["label"] == label
            axes[1].scatter(df.loc[mask, "pc1"], df.loc[mask, "pc2"],
                          s=3, alpha=0.3, label=label, color=self.colors.get(label, "#333"))
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        axes[1].set_title("Latent Space (colored by class)")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(self.output_dir / "plots" / "latent_space.png", dpi=150)
        plt.close(fig)

    def _plot_dynamics(self):
        """Plot latent dynamics (speed, acceleration) vs time."""
        if self.latent_df.empty or "speed" not in self.latent_df:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Aggregate by time bins
        df = self.latent_df.copy()
        df["time_bin"] = (df["hours_before"] // 0.5).astype(int) * 0.5  # 30-min bins

        for ax, metric, title in [(axes[0], "speed", "Latent Velocity"),
                                   (axes[1], "accel", "Latent Acceleration")]:
            agg = df.groupby("time_bin")[metric].agg(["mean", "std"])

            ax.plot(agg.index, agg["mean"], "b-", linewidth=2)
            ax.fill_between(agg.index, agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                          alpha=0.3)
            ax.set_xlabel("Hours Before Birth")
            ax.set_ylabel(title)
            ax.set_title(f"{title} vs Time")
            ax.invert_xaxis()
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.output_dir / "plots" / "dynamics.png", dpi=150)
        plt.close(fig)

    def _generate_dashboards(self, n: int = 12):
        """Generate detailed dashboards for representative GUIDs."""
        if self.epoch_df.empty:
            return

        # Select GUIDs with most epochs, stratified by class
        selected = []
        for label in self.epoch_df["label"].unique():
            if label == "unknown":
                continue
            label_guids = self.epoch_df[self.epoch_df["label"] == label]
            top_guids = label_guids.groupby("guid").size().nlargest(n // 3).index.tolist()
            selected.extend(top_guids)

        # Fill remaining with any GUIDs
        if len(selected) < n:
            remaining = self.epoch_df[~self.epoch_df["guid"].isin(selected)]
            more = remaining.groupby("guid").size().nlargest(n - len(selected)).index.tolist()
            selected.extend(more)

        for guid in selected[:n]:
            self._plot_guid_dashboard(guid)

    def _plot_guid_dashboard(self, guid: str):
        """Generate multi-panel dashboard for a single GUID."""
        latent = self.latent_df[self.latent_df["guid"] == guid]
        epochs = self.epoch_df[self.epoch_df["guid"] == guid]

        if latent.empty or epochs.empty:
            return

        label = epochs["label"].iloc[0]

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        # 1. KLD over epochs (hours before birth)
        ax1 = fig.add_subplot(gs[0, 0])
        epochs_sorted = epochs.sort_values("hours_before", ascending=False)
        ax1.plot(epochs_sorted["hours_before"], epochs_sorted["kld_mean"],
                "o-", color=self.colors.get(label, "blue"))
        ax1.set_xlabel("Hours Before Birth")
        ax1.set_ylabel("KLD")
        ax1.set_title("KLD Trajectory")
        ax1.invert_xaxis()
        ax1.grid(True, alpha=0.3)

        # 2. Latent trajectory (PC1 vs PC2)
        ax2 = fig.add_subplot(gs[0, 1])
        if "pc1" in latent.columns:
            scatter = ax2.scatter(latent["pc1"], latent["pc2"], c=latent["t"],
                                 s=5, alpha=0.5, cmap="plasma")
            plt.colorbar(scatter, ax=ax2, label="Timestep")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_title("Latent Trajectory")

        # 3. Speed across epochs
        ax3 = fig.add_subplot(gs[0, 2])
        if "speed" in latent.columns:
            for epoch in latent["epoch_sec"].unique()[:5]:
                ep_data = latent[latent["epoch_sec"] == epoch].sort_values("t")
                ax3.plot(ep_data["t_sec"], ep_data["speed"], alpha=0.5)
        ax3.set_xlabel("Time in Epoch (s)")
        ax3.set_ylabel("Speed")
        ax3.set_title("Latent Velocity")
        ax3.grid(True, alpha=0.3)

        # 4. KLD within epochs
        ax4 = fig.add_subplot(gs[1, 0])
        profile = latent.groupby("t")["kld"].mean()
        ax4.plot(profile.index * TIMESTEP_SECONDS, profile.values, "b-", linewidth=2)
        ax4.axvspan(0, self.warmup_steps * TIMESTEP_SECONDS, alpha=0.2, color="red", label="Warmup")
        ax4.set_xlabel("Time in Epoch (s)")
        ax4.set_ylabel("KLD")
        ax4.set_title("Within-Epoch KLD Profile")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Uncertainty over time
        ax5 = fig.add_subplot(gs[1, 1])
        if "uncertainty" in latent.columns:
            unc_by_epoch = latent.groupby("epoch_sec")["uncertainty"].mean()
            hours = pd.Series(epochs.set_index("epoch_sec")["hours_before"])
            unc_df = pd.DataFrame({"uncertainty": unc_by_epoch, "hours": hours.loc[unc_by_epoch.index]})
            unc_df = unc_df.sort_values("hours", ascending=False)
            ax5.plot(unc_df["hours"], unc_df["uncertainty"], "o-", color=self.colors.get(label, "blue"))
            ax5.invert_xaxis()
        ax5.set_xlabel("Hours Before Birth")
        ax5.set_ylabel("Uncertainty")
        ax5.set_title("Posterior Uncertainty")
        ax5.grid(True, alpha=0.3)

        # 6. z0, z1, z2 over time (first 3 latent dims)
        for i, d in enumerate(range(3)):
            ax = fig.add_subplot(gs[2, i])
            col = f"z{d}"
            if col in latent.columns:
                for epoch in latent["epoch_sec"].unique()[:3]:
                    ep_data = latent[latent["epoch_sec"] == epoch].sort_values("t")
                    ax.plot(ep_data["t_sec"], ep_data[col], alpha=0.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(f"z{d}")
            ax.set_title(f"Latent Dim {d}")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"GUID: {guid} | Class: {label}", fontsize=14, fontweight="bold")

        fig.savefig(self.output_dir / "dashboards" / f"{guid}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute summary metrics."""
        metrics = {}

        if not self.epoch_df.empty and "kld_mean" in self.epoch_df:
            kld = self.epoch_df["kld_mean"].dropna()
            metrics["kld"] = {
                "mean": float(kld.mean()),
                "std": float(kld.std()),
                "median": float(kld.median()),
            }

        # Class separation (silhouette score)
        if HAS_SKLEARN and not self.latent_df.empty:
            z_cols = [f"z{d}" for d in range(self.latent_dim)]
            labeled = self.latent_df[self.latent_df["label"] != "unknown"]

            if len(labeled) > 1000 and labeled["label"].nunique() >= 2:
                sample = labeled.sample(min(10000, len(labeled)), random_state=42)
                Z = sample[z_cols].values
                y = sample["label"].astype("category").cat.codes.values

                try:
                    metrics["silhouette"] = float(silhouette_score(Z, y))
                except Exception:
                    pass

        return metrics


# Convenience function for simple usage
def run_trajectory_analysis(
    model: torch.nn.Module,
    dataloader,
    output_dir: str | Path,
    time_range_hours: Optional[float] = 12.0,
    min_epochs_per_guid: int = 3,
    n_dashboards: int = 12,
) -> Dict[str, Any]:
    """
    Run trajectory and KLD analysis.

    Args:
        model: VAE-TEB model
        dataloader: DataLoader with fhr_st, fhr_ph, fhr_up_ph, guid, epoch
        output_dir: Where to save outputs
        time_range_hours: Analyze this many hours before birth (None = all)
        min_epochs_per_guid: Minimum epochs per GUID
        n_dashboards: Number of per-GUID dashboards to generate

    Returns:
        Summary dict with statistics and metrics
    """
    analyzer = TrajectoryAnalyzer(
        model=model,
        dataloader=dataloader,
        output_dir=output_dir,
        time_range_hours=time_range_hours,
        min_epochs_per_guid=min_epochs_per_guid,
    )
    return analyzer.run(n_dashboards=n_dashboards)
