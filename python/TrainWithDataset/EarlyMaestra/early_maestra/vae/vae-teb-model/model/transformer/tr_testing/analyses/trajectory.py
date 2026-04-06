"""Temporal / trajectory analysis (Category 5).

Per-patient (GUID) trajectory analysis over time-to-delivery. Requires
GUID-based DataLoaders where each batch = one patient's epochs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from model.transformer.tr_testing.base import TransformerTestRunner
from model.transformer.tr_testing.metrics import (
    compute_gate_statistics,
    compute_kl_per_anchor,
    compute_per_anchor_mae,
    compute_te_residual_norm,
)


def _collect_guid_trajectories(
    runner: TransformerTestRunner,
    loader: Any,
    class_label: str,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Collect per-epoch metrics for trajectory analysis.

    Each row is one 20-minute segment with segment-level aggregated
    metrics. Requires a GUID-based loader.

    Args:
        runner: TransformerTestRunner instance.
        loader: GUID-based DataLoader.
        class_label: Class name string.
        max_samples: Maximum number of segments.

    Returns:
        DataFrame with one row per segment, sorted by (guid, epoch).
    """
    from model.transformer.tr_testing.collectors import _build_metadata_row

    rows = []
    count = 0

    with runner.inference_mode():
        for batch in loader:
            Y = batch.fhr_st.to(runner.device)
            U = batch.up_st.to(runner.device)
            B = Y.shape[0]

            if max_samples and count >= max_samples:
                break

            # Anchor-based forward
            outputs = runner.forward_with_anchors(Y, U)
            anchors = outputs["anchor_indices"]
            K = anchors.shape[1]

            # Embedding
            e_win = runner.forward_for_embedding(Y, U).cpu().numpy()

            # Intermediates (gate)
            intermediates = runner.extract_intermediates(Y, U)
            gate_stats = compute_gate_statistics(intermediates["gate"])

            # KL
            kl = compute_kl_per_anchor(
                outputs["mu_post"], outputs["logvar_post"],
                outputs["mu_prior"], outputs["logvar_prior"],
            ).reshape(B, K).mean(dim=1).cpu().numpy()

            # MAE (fused, max horizon)
            h_max = max(runner.config.horizons)
            fused_mae = compute_per_anchor_mae(
                outputs["Y_hat_fus"], Y, anchors, runner.config.guard_gap
            )[h_max].reshape(B, K).mean(dim=1).cpu().numpy()

            # TE residual norm
            r_norms = compute_te_residual_norm(outputs["R_hat"])
            r_mean = r_norms[h_max].reshape(B, K).mean(dim=1).cpu().numpy()

            for i in range(B):
                if max_samples and count >= max_samples:
                    break
                row = _build_metadata_row(batch, i, class_label)
                row["kl_mean"] = float(kl[i])
                row["mean_gate"] = float(gate_stats["mean"][i].item())
                row[f"mae_fused_h{h_max}"] = float(fused_mae[i])
                row[f"residual_norm_h{h_max}"] = float(r_mean[i])
                row["e_win"] = e_win[i].tolist()
                rows.append(row)
                count += 1

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["guid", "epoch"]).reset_index(drop=True)
    return df


def _compute_embedding_drift(traj_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rate of embedding change between consecutive epochs.

    Args:
        traj_df: Trajectory DataFrame with 'e_win' column (list).

    Returns:
        DataFrame with drift rates per pair of consecutive epochs.
    """
    drift_rows = []
    for guid, group in traj_df.groupby("guid"):
        if len(group) < 2:
            continue
        group = group.sort_values("epoch")
        embeddings = np.array(group["e_win"].tolist())
        epochs = group["epoch"].values
        cls = group["class_label"].iloc[0]

        for j in range(1, len(embeddings)):
            dt = abs(epochs[j] - epochs[j - 1])
            if dt < 1e-6:
                continue
            dist = np.linalg.norm(embeddings[j] - embeddings[j - 1])
            drift_rows.append({
                "guid": guid,
                "class_label": cls,
                "epoch": epochs[j],
                "drift": dist,
                "drift_rate": dist / dt,
            })
    return pd.DataFrame(drift_rows)


def run_trajectory_analysis(
    runner: TransformerTestRunner,
    guid_loaders: Dict[str, Any],
    output_dir: Path,
    max_samples: Optional[int] = None,
    min_epochs_per_guid: int = 5,
) -> Dict[str, Any]:
    """Run temporal trajectory analysis for all classes.

    Args:
        runner: TransformerTestRunner instance.
        guid_loaders: Dict mapping class names to GUID-based DataLoaders.
        output_dir: Output directory.
        max_samples: Maximum samples per class.
        min_epochs_per_guid: Minimum epochs for a GUID to be included
            in trajectory plots.

    Returns:
        Summary dict.
    """
    from model.transformer.tr_testing.visualizers import (
        plot_class_mean_trajectory,
        plot_embedding_drift,
        plot_guid_trajectory,
        plot_trajectory_comparison,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_guid_dir = output_dir / "per_guid"
    per_guid_dir.mkdir(parents=True, exist_ok=True)

    # Collect trajectory data from all classes
    all_traj = []
    for class_name, loader in guid_loaders.items():
        logger.info(f"  Collecting trajectories for {class_name}...")
        traj_df = _collect_guid_trajectories(
            runner, loader, class_name, max_samples=max_samples
        )
        all_traj.append(traj_df)

    traj_df = pd.concat(all_traj, ignore_index=True)

    # Save trajectory data (without embedding lists for CSV)
    save_df = traj_df.drop(columns=["e_win"], errors="ignore")
    save_df.to_csv(output_dir / "trajectory_data.csv", index=False)

    plots = {}

    def _try_plot(name, fn, *args, **kwargs):
        try:
            path = fn(*args, **kwargs)
            plots[name] = str(path)
        except Exception as e:
            logger.warning(f"Plot {name} failed: {e}")

    # Build class-mean trajectory dicts for visualizers
    h_max = max(runner.config.horizons)

    def _build_class_mean_dict(df, metric_col):
        """Transform DataFrame to {class: {time, metric, sem}} dict."""
        result = {}
        for cls, grp in df.groupby("class_label"):
            if metric_col not in grp.columns:
                continue
            # Bin by epoch_hours (1-hour bins)
            grp = grp.dropna(subset=[metric_col])
            if grp.empty:
                continue
            grp = grp.sort_values("epoch")
            hours = grp["epoch"].values / 3600.0
            vals = grp[metric_col].values
            # Simple binned average
            bin_edges = np.arange(np.floor(hours.min()), np.ceil(hours.max()) + 1, 1.0)
            if len(bin_edges) < 2:
                result[cls] = {"time": hours, "metric": vals, "sem": np.zeros_like(vals)}
                continue
            bin_idx = np.digitize(hours, bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)
            time_out, metric_out, sem_out = [], [], []
            for b in range(len(bin_edges) - 1):
                mask = bin_idx == b
                if mask.sum() > 0:
                    time_out.append((bin_edges[b] + bin_edges[b + 1]) / 2)
                    metric_out.append(vals[mask].mean())
                    sem_out.append(vals[mask].std() / max(np.sqrt(mask.sum()), 1))
            result[cls] = {
                "time": np.array(time_out),
                "metric": np.array(metric_out),
                "sem": np.array(sem_out),
            }
        return result

    for metric_col in ["kl_mean", "mean_gate", f"mae_fused_h{h_max}",
                        f"residual_norm_h{h_max}"]:
        mean_dict = _build_class_mean_dict(traj_df, metric_col)
        _try_plot(f"class_mean_{metric_col}",
                  plot_class_mean_trajectory, mean_dict, output_dir,
                  metric=metric_col)

    # Per-GUID trajectories (for GUIDs with enough epochs)
    guid_counts = traj_df.groupby("guid").size()
    eligible_guids = guid_counts[
        guid_counts >= min_epochs_per_guid
    ].index.tolist()
    max_guid_plots = min(20, len(eligible_guids))

    for guid in eligible_guids[:max_guid_plots]:
        gdf = traj_df[traj_df["guid"] == guid].sort_values("epoch")
        # Transform DataFrame row to dict expected by plot_guid_trajectory
        guid_dict = {
            "guid": guid,
            "class_label": gdf["class_label"].iloc[0],
            "epochs": gdf["epoch"].values,
            "e_win": np.array(gdf["e_win"].tolist()),
        }
        _try_plot(
            f"guid_{guid[:8]}",
            plot_guid_trajectory, guid_dict, per_guid_dir,
        )

    # Embedding drift — transform DataFrame to {class: array} dict
    drift_df = _compute_embedding_drift(traj_df)
    if not drift_df.empty:
        drift_dict = {
            cls: grp["drift_rate"].values
            for cls, grp in drift_df.groupby("class_label")
        }
        _try_plot("embedding_drift",
                  plot_embedding_drift, drift_dict, output_dir)

    # Cross-class trajectory comparison — build {class: {mean_proj, time}} dict
    def _build_trajectory_comparison(df):
        result = {}
        for cls, grp in df.groupby("class_label"):
            ewin_list = grp["e_win"].tolist()
            if not ewin_list:
                continue
            ewin_arr = np.array(ewin_list)
            # PCA to 2D
            centered = ewin_arr - ewin_arr.mean(axis=0, keepdims=True)
            try:
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                proj = centered @ Vt[:2].T
            except np.linalg.LinAlgError:
                proj = centered[:, :2]
            result[cls] = {
                "mean_proj": proj,
                "time": grp["epoch"].values / 3600.0,
            }
        return result

    traj_comp = _build_trajectory_comparison(traj_df)
    _try_plot("trajectory_comparison",
              plot_trajectory_comparison, traj_comp, output_dir)

    summary = {
        "plots": plots,
        "n_segments": len(traj_df),
        "n_guids": traj_df["guid"].nunique() if not traj_df.empty else 0,
        "n_eligible_guids": len(eligible_guids),
    }

    logger.info(
        f"Trajectory analysis: {summary['n_segments']} segments, "
        f"{summary['n_guids']} GUIDs, {len(plots)} plots"
    )
    return summary
