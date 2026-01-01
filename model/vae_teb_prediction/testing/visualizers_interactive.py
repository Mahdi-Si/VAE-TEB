"""
Interactive Plotly visualization functions for VAE-TEB testing.

This module provides interactive HTML visualizations using Plotly for
exploratory data analysis. Plots include hover information, zoom/pan,
and interactive legends.

Example:
    >>> from testing.visualizers_interactive import plot_kld_trajectory_interactive
    >>> plot_kld_trajectory_interactive(trajectory_df, Path("results/kld.html"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def _check_plotly():
    """Raise ImportError if Plotly is not available."""
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for interactive visualizations. Install with: pip install plotly")


def plot_reconstruction_interactive(
    sample: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Create an interactive 3-panel reconstruction analysis figure.

    Panels:
        1. Signal overlay with hover showing exact values
        2. Residual with zoom/pan
        3. Latent heatmap with hover

    Args:
        sample: Dict with 'y_true', 'y_pred', 'y_pred_std', 'latent', 'metrics'.
        output_path: Path to save the HTML file.

    Example:
        >>> plot_reconstruction_interactive(sample, Path("results/sample.html"))
    """
    _check_plotly()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = sample["y_true"]
    y_pred = sample["y_pred"]
    y_pred_std = sample.get("y_pred_std")
    latent = sample.get("latent")
    metrics = sample.get("metrics", {})

    # Create time axis in minutes
    time = np.arange(len(y_true)) / 4.0 / 60.0

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.2, 0.3],
        subplot_titles=["Signal Reconstruction", "Residual", "Latent Representation"],
        vertical_spacing=0.08,
    )

    # ----- Panel 1: Signal reconstruction -----
    fig.add_trace(
        go.Scatter(
            x=time, y=y_true,
            name="Ground Truth",
            line=dict(color="#4C72B0", width=1),
            hovertemplate="Time: %{x:.2f} min<br>Value: %{y:.3f}<extra>Ground Truth</extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=y_pred,
            name="Prediction",
            line=dict(color="#C44E52", width=1),
            hovertemplate="Time: %{x:.2f} min<br>Value: %{y:.3f}<extra>Prediction</extra>",
        ),
        row=1, col=1,
    )

    # Add uncertainty band if available
    if y_pred_std is not None:
        upper = y_pred + 2 * y_pred_std
        lower = y_pred - 2 * y_pred_std
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([time, time[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor="rgba(196, 78, 82, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="±2σ",
                showlegend=True,
            ),
            row=1, col=1,
        )

    # ----- Panel 2: Residual -----
    residual = y_true - y_pred
    fig.add_trace(
        go.Scatter(
            x=time, y=residual,
            name="Residual",
            fill="tozeroy",
            line=dict(color="#55A868", width=1),
            hovertemplate="Time: %{x:.2f} min<br>Error: %{y:.3f}<extra>Residual</extra>",
        ),
        row=2, col=1,
    )

    # ----- Panel 3: Latent heatmap -----
    if latent is not None and latent.size > 0:
        fig.add_trace(
            go.Heatmap(
                z=latent.T,
                colorscale="RdBu_r",
                showscale=True,
                colorbar=dict(title="Value", x=1.02),
                hovertemplate="Timestep: %{x}<br>Dim: %{y}<br>Value: %{z:.3f}<extra></extra>",
            ),
            row=3, col=1,
        )

    # Update layout
    metrics_text = f"VAF: {metrics.get('vaf', 0):.3f} | SNR: {metrics.get('snr', 0):.1f} dB | KLD: {metrics.get('kld', 0):.4f}"

    fig.update_layout(
        title=dict(
            text=f"Sample Analysis - GUID: {sample.get('guid', 'N/A')}<br><sub>{metrics_text}</sub>",
            x=0.5,
        ),
        height=800,
        showlegend=True,
        legend=dict(x=1.05, y=0.95),
    )

    fig.update_xaxes(title_text="Time (minutes)", row=2, col=1)
    fig.update_yaxes(title_text="FHR", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=2, col=1)
    fig.update_xaxes(title_text="Timestep", row=3, col=1)
    fig.update_yaxes(title_text="Latent Dim", row=3, col=1)

    fig.write_html(str(output_path), include_plotlyjs="cdn")


def plot_kld_trajectory_interactive(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Create an interactive KLD trajectory plot with hover and filtering.

    Features:
        - Hover shows GUID, exact values
        - Click legend to filter by class
        - Pan/zoom support

    Args:
        df: DataFrame with 'hours_before', 'kld_mean', 'guid', and optionally 'label'.
        output_path: Path to save the HTML file.

    Example:
        >>> plot_kld_trajectory_interactive(trajectory_df, Path("results/kld.html"))
    """
    _check_plotly()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty or "hours_before" not in df.columns:
        return

    # Prepare data
    has_labels = "label" in df.columns and df["label"].notna().any()

    if has_labels:
        # Map labels to names
        label_map = {0: "Unknown", 1: "Healthy", 2: "Acidosis", 3: "HIE"}
        df = df.copy()
        df["class_name"] = df["label"].map(label_map).fillna("Unknown")

        fig = px.scatter(
            df,
            x="hours_before",
            y="kld_mean",
            color="class_name",
            hover_data=["guid", "epoch"],
            color_discrete_map={
                "Healthy": "#55A868",
                "Acidosis": "#C44E52",
                "HIE": "#8172B2",
                "Unknown": "#666666",
            },
            labels={
                "hours_before": "Hours Before Birth",
                "kld_mean": "KLD (Transfer Entropy)",
                "class_name": "Class",
            },
        )
    else:
        fig = px.scatter(
            df,
            x="hours_before",
            y="kld_mean",
            hover_data=["guid", "epoch"] if "guid" in df.columns else None,
            labels={
                "hours_before": "Hours Before Birth",
                "kld_mean": "KLD (Transfer Entropy)",
            },
        )

    # Add trend line (LOWESS smoothing per class)
    if has_labels:
        for class_name in df["class_name"].unique():
            subset = df[df["class_name"] == class_name].sort_values("hours_before")
            if len(subset) > 10:
                # Simple moving average for trend
                window = min(20, len(subset) // 5)
                if window > 1:
                    trend = subset["kld_mean"].rolling(window, center=True).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=subset["hours_before"],
                            y=trend,
                            mode="lines",
                            name=f"{class_name} (trend)",
                            line=dict(width=2),
                            showlegend=False,
                        )
                    )

    fig.update_layout(
        title="Transfer Entropy Evolution Before Delivery",
        xaxis=dict(autorange="reversed"),  # Time flows toward birth
        height=600,
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn")


def plot_latent_space_3d(
    latents: np.ndarray,
    labels: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
) -> go.Figure:
    """
    Create an interactive 3D scatter plot of PCA-reduced latent space.

    Args:
        latents: Array of shape (N, D) with latent representations.
        labels: Optional array of shape (N,) with class labels.
        output_path: Optional path to save HTML. If None, returns figure.

    Returns:
        Plotly Figure object.

    Example:
        >>> fig = plot_latent_space_3d(latents, labels)
        >>> fig.show()
    """
    _check_plotly()

    from sklearn.decomposition import PCA

    if latents.size == 0:
        return go.Figure()

    # Reduce to 3D using PCA
    pca = PCA(n_components=3)
    latents_3d = pca.fit_transform(latents)

    # Prepare dataframe for plotting
    df = pd.DataFrame({
        "PC1": latents_3d[:, 0],
        "PC2": latents_3d[:, 1],
        "PC3": latents_3d[:, 2],
    })

    if labels is not None:
        label_map = {0: "Unknown", 1: "Healthy", 2: "Acidosis", 3: "HIE"}
        df["class"] = [label_map.get(int(l), f"Class {l}") for l in labels]

        fig = px.scatter_3d(
            df, x="PC1", y="PC2", z="PC3",
            color="class",
            color_discrete_map={
                "Healthy": "#55A868",
                "Acidosis": "#C44E52",
                "HIE": "#8172B2",
                "Unknown": "#666666",
            },
        )
    else:
        fig = px.scatter_3d(df, x="PC1", y="PC2", z="PC3")

    # Update layout
    variance_explained = pca.explained_variance_ratio_ * 100
    fig.update_layout(
        title=f"Latent Space (PCA: {variance_explained.sum():.1f}% variance explained)",
        scene=dict(
            xaxis_title=f"PC1 ({variance_explained[0]:.1f}%)",
            yaxis_title=f"PC2 ({variance_explained[1]:.1f}%)",
            zaxis_title=f"PC3 ({variance_explained[2]:.1f}%)",
        ),
        height=700,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path), include_plotlyjs="cdn")

    return fig


def plot_latent_interpolation_interactive(
    samples: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Create an interactive latent interpolation visualization with slider.

    Shows how the reconstructed signal morphs as we interpolate between
    two latent codes.

    Args:
        samples: List of dicts with 'interpolated_signals', 'z_start', 'z_end'.
            Each dict represents one interpolation pair.
        output_path: Path to save the HTML file.

    Example:
        >>> plot_latent_interpolation_interactive(interp_samples, Path("results/interp.html"))
    """
    _check_plotly()

    if not samples:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use first sample for demonstration
    sample = samples[0]
    signals = sample.get("interpolated_signals", [])

    if not signals:
        return

    n_steps = len(signals)
    time = np.arange(len(signals[0])) / 4.0 / 60.0  # Minutes

    # Create figure with frames for animation
    fig = go.Figure()

    # Add all frames
    frames = []
    for i, signal in enumerate(signals):
        frames.append(
            go.Frame(
                data=[go.Scatter(x=time, y=signal, mode="lines", line=dict(color="#4C72B0"))],
                name=str(i),
            )
        )

    # Initial data (first frame)
    fig.add_trace(
        go.Scatter(
            x=time, y=signals[0],
            mode="lines",
            line=dict(color="#4C72B0", width=2),
            name="Interpolated Signal",
        )
    )

    fig.frames = frames

    # Create slider
    sliders = [
        dict(
            active=0,
            steps=[
                dict(
                    method="animate",
                    args=[[str(i)], dict(frame=dict(duration=100, redraw=True), mode="immediate")],
                    label=f"{i/(n_steps-1):.2f}" if n_steps > 1 else "0",
                )
                for i in range(n_steps)
            ],
            x=0.1,
            len=0.8,
            xanchor="left",
            y=0,
            yanchor="top",
            currentvalue=dict(
                prefix="Interpolation α = ",
                visible=True,
                xanchor="center",
            ),
            transition=dict(duration=100),
        )
    ]

    fig.update_layout(
        title="Latent Space Interpolation",
        xaxis_title="Time (minutes)",
        yaxis_title="FHR (normalized)",
        sliders=sliders,
        height=500,
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn")


def plot_metrics_comparison_interactive(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Create interactive scatter matrix of metrics colored by class.

    Args:
        df: DataFrame with 'vaf', 'mse', 'snr', 'kld', and optionally 'label'.
        output_path: Path to save HTML file.

    Example:
        >>> plot_metrics_comparison_interactive(metrics_df, Path("results/metrics.html"))
    """
    _check_plotly()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data
    metrics_cols = ["vaf", "mse", "snr", "kld"]
    available_cols = [c for c in metrics_cols if c in df.columns]

    if len(available_cols) < 2:
        return

    df_plot = df[available_cols].copy()

    # Add labels if available
    if "label" in df.columns:
        label_map = {0: "Unknown", 1: "Healthy", 2: "Acidosis", 3: "HIE"}
        df_plot["class"] = df["label"].map(label_map).fillna("Unknown")

        fig = px.scatter_matrix(
            df_plot,
            dimensions=available_cols,
            color="class",
            color_discrete_map={
                "Healthy": "#55A868",
                "Acidosis": "#C44E52",
                "HIE": "#8172B2",
                "Unknown": "#666666",
            },
        )
    else:
        fig = px.scatter_matrix(df_plot, dimensions=available_cols)

    fig.update_layout(
        title="Metrics Comparison",
        height=800,
        width=900,
    )

    fig.update_traces(diagonal_visible=False, showupperhalf=False)

    fig.write_html(str(output_path), include_plotlyjs="cdn")
