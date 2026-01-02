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
            line=dict(color="#3F72AF", width=1),
            hovertemplate="Time: %{x:.2f} min<br>Value: %{y:.3f}<extra>Ground Truth</extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=y_pred,
            name="Prediction",
            line=dict(color="#EB5B00", width=1),
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
                fillcolor="rgba(235, 91, 0, 0.2)",
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
            line=dict(color="#609966", width=1),
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
    *,
    by_class: bool = False,
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
        by_class: Whether to color/group by class labels (default False).

    Example:
        >>> plot_kld_trajectory_interactive(trajectory_df, Path("results/kld.html"))
    """
    _check_plotly()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty or "hours_before" not in df.columns:
        return

    # Prepare data
    has_labels = by_class and "label" in df.columns and df["label"].notna().any()

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
                "Healthy": "#609966",
                "Acidosis": "#EB5B00",
                "HIE": "#112D4E",
                "Unknown": "#393E46",
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
                "Healthy": "#609966",
                "Acidosis": "#EB5B00",
                "HIE": "#112D4E",
                "Unknown": "#393E46",
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
                data=[go.Scatter(x=time, y=signal, mode="lines", line=dict(color="#3F72AF"))],
                name=str(i),
            )
        )

    # Initial data (first frame)
    fig.add_trace(
        go.Scatter(
            x=time, y=signals[0],
            mode="lines",
            line=dict(color="#3F72AF", width=2),
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
                "Healthy": "#609966",
                "Acidosis": "#EB5B00",
                "HIE": "#112D4E",
                "Unknown": "#393E46",
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


# -----------------------------------------------------------------------------
# Trajectory Visualization Functions
# -----------------------------------------------------------------------------


def plot_latent_trajectory_3d_interactive(
    trajectory: np.ndarray,
    output_path: Path,
    *,
    sample_id: str = "sample",
    color_by_time: bool = True,
    point_size: int = 5,
    line_width: int = 2,
) -> Optional[go.Figure]:
    """
    Create an interactive 3D latent trajectory plot using Plotly.

    Features:
        - Rotate, zoom, pan the 3D view
        - Hover shows time step information
        - Start (green) and end (red) markers
        - Optional color gradient by time progression

    Args:
        trajectory: Array of shape (T, 3) or (1, T, 3) with reduced trajectory.
        output_path: Path to save HTML file.
        sample_id: Sample identifier for the title.
        color_by_time: Whether to color points by time progression.
        point_size: Size of trajectory points.
        line_width: Width of trajectory line.

    Returns:
        Plotly Figure object, or None if plotting fails.

    Example:
        >>> plot_latent_trajectory_3d_interactive(traj_3d, Path("results/traj.html"))
    """
    _check_plotly()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if trajectory.ndim == 3:
        trajectory = trajectory.squeeze(0)

    if trajectory.shape[1] < 3:
        return None

    time_steps = trajectory.shape[0]

    # Main trajectory trace
    if color_by_time:
        main_trace = go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=trajectory[:, 2],
            mode="markers+lines",
            marker=dict(
                size=point_size,
                color=np.arange(time_steps),
                colorscale="Cividis",
                showscale=True,
                colorbar=dict(
                    title="Time Step",
                    x=1.05,
                    xanchor="left",
                    thickness=15,
                    len=0.7,
                ),
            ),
            line=dict(color="rgba(100, 100, 100, 0.5)", width=line_width),
            text=[f"Time: {i}" for i in range(time_steps)],
            hovertemplate="Time: %{text}<br>Dim1: %{x:.3f}<br>Dim2: %{y:.3f}<br>Dim3: %{z:.3f}<extra></extra>",
            name="Trajectory",
            showlegend=True,
        )
    else:
        main_trace = go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=trajectory[:, 2],
            mode="markers+lines",
            marker=dict(size=point_size, color="#3F72AF"),
            line=dict(color="#3F72AF", width=line_width),
            text=[f"Time: {i}" for i in range(time_steps)],
            hovertemplate="Time: %{text}<br>Dim1: %{x:.3f}<br>Dim2: %{y:.3f}<br>Dim3: %{z:.3f}<extra></extra>",
            name="Trajectory",
            showlegend=True,
        )

    # Start marker
    start_trace = go.Scatter3d(
        x=[trajectory[0, 0]],
        y=[trajectory[0, 1]],
        z=[trajectory[0, 2]],
        mode="markers",
        marker=dict(size=10, color="#609966", symbol="circle"),
        name="Start",
        hovertemplate="START<br>Dim1: %{x:.3f}<br>Dim2: %{y:.3f}<br>Dim3: %{z:.3f}<extra></extra>",
    )

    # End marker
    end_trace = go.Scatter3d(
        x=[trajectory[-1, 0]],
        y=[trajectory[-1, 1]],
        z=[trajectory[-1, 2]],
        mode="markers",
        marker=dict(size=10, color="#EB5B00", symbol="x"),
        name="End",
        hovertemplate="END<br>Dim1: %{x:.3f}<br>Dim2: %{y:.3f}<br>Dim3: %{z:.3f}<extra></extra>",
    )

    fig = go.Figure(data=[main_trace, start_trace, end_trace])

    fig.update_layout(
        title=dict(
            text=f"Latent Trajectory 3D - {sample_id}",
            x=0.5,
        ),
        scene=dict(
            xaxis_title="Latent Dim 1",
            yaxis_title="Latent Dim 2",
            zaxis_title="Latent Dim 3",
            aspectmode="data",
        ),
        width=1000,
        height=800,
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
        ),
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return fig


def plot_fhr_timeline(
    fhr: np.ndarray,
    epoch: np.ndarray,
    output_path: Path,
    *,
    sample_id: str = "sample",
    sampling_rate_hz: float = 4.0,
    segment_duration_minutes: float = 20.0,
    detect_changepoints: bool = False,
    n_changepoints: int = 5,
) -> None:
    """
    Plot complete FHR timeline with missing segments as gaps.

    Creates an interactive Plotly figure with zoom/pan support.
    Multiple 20-minute segments are shown along a continuous timeline.

    Args:
        fhr: FHR signals of shape (N, T) where N is number of segments.
        epoch: Epoch values of shape (N,) - seconds before birth for each segment.
        output_path: Path to save HTML file.
        sample_id: Sample identifier for the title.
        sampling_rate_hz: Sampling rate in Hz (default 4.0).
        segment_duration_minutes: Duration of each segment in minutes (default 20.0).
        detect_changepoints: Whether to detect and visualize changepoints (default False).
        n_changepoints: Number of changepoints per segment (default 5).

    Example:
        >>> plot_fhr_timeline(fhr_segments, epoch_values, Path("results/timeline.html"))
    """
    _check_plotly()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_id = str(sample_id)

    # Handle tensor conversion
    if hasattr(fhr, "cpu"):
        fhr = fhr.cpu().numpy()
    if hasattr(epoch, "cpu"):
        epoch = epoch.cpu().numpy()

    fhr = np.asarray(fhr)
    epoch = np.asarray(epoch)

    if fhr.ndim == 1:
        fhr = fhr[None, :]
        epoch = np.array([epoch]) if epoch.ndim == 0 else epoch[:1]

    batch_size, time_steps = fhr.shape
    time_per_step = 1.0 / sampling_rate_hz

    fig = go.Figure()

    # Import changepoint detector if needed
    detector_fn = None
    if detect_changepoints:
        try:
            from .analyses.changepoint import create_changepoint_detector
            detector_fn = create_changepoint_detector(algo="gradient")
        except ImportError:
            detect_changepoints = False

    # Plot each segment
    for i in range(batch_size):
        # Epoch is seconds before birth (negative)
        segment_start_minutes = float(epoch[i]) / 60.0

        # Create time array for this segment in minutes
        segment_times_minutes = segment_start_minutes + np.arange(time_steps) * time_per_step / 60.0

        # Add FHR trace for this segment
        fig.add_trace(
            go.Scatter(
                x=segment_times_minutes,
                y=fhr[i],
                mode="lines",
                name="FHR" if i == 0 else None,
                legendgroup="FHR",
                showlegend=(i == 0),
                line=dict(color="#EB5B00", width=1.5),
                hovertemplate=f"Segment {i}<br>Time: %{{x:.2f}} min<br>FHR: %{{y:.1f}}<extra></extra>",
            )
        )

        # Detect and visualize changepoints if requested
        if detect_changepoints and detector_fn is not None:
            try:
                fhr_signal = fhr[i].reshape(-1, 1)
                max_possible = max(1, time_steps // 10)
                actual_bkps = min(n_changepoints, max_possible)
                per_segment_bkps = max(0, actual_bkps - 1)

                if per_segment_bkps > 0:
                    cp_indices = detector_fn(fhr_signal, per_segment_bkps)
                    cp_indices = np.asarray(cp_indices, dtype=int)

                    if cp_indices.size > 0:
                        cp_times = segment_start_minutes + cp_indices * time_per_step / 60.0
                        for cp_time in cp_times:
                            fig.add_vline(
                                x=cp_time,
                                line=dict(color="#609966", width=2, dash="dash"),
                                opacity=0.6,
                            )
            except Exception:
                pass

    # Add changepoint legend entry (if used)
    if detect_changepoints:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="Changepoints",
                line=dict(color="#609966", width=2, dash="dash"),
                showlegend=True,
            )
        )

    fig.update_layout(
        title=dict(
            text=f"FHR Timeline - {sample_id}",
            font=dict(size=18),
        ),
        width=1800,
        height=500,
        showlegend=True,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.update_xaxes(
        title_text="Time (minutes before birth)",
        gridcolor="#EEEEEE",
        showgrid=True,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
    )

    fig.update_yaxes(
        title_text="FHR (normalized)",
        gridcolor="lightgray",
        showgrid=True,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn")


def plot_trajectory_animation(
    trajectory: np.ndarray,
    output_path: Path,
    *,
    sample_id: str = "sample",
    fps: int = 20,
    duration_seconds: float = 5.0,
) -> bool:
    """
    Create an animated GIF of the latent trajectory evolution.

    Generates 2D or 3D animation depending on trajectory dimensions.

    Args:
        trajectory: Array of shape (T, D) where D is 2 or 3.
        output_path: Path to save GIF file (should end in .gif).
        sample_id: Sample identifier for the title.
        fps: Frames per second (default 20).
        duration_seconds: Target animation duration in seconds (default 5.0).

    Returns:
        True if animation was successfully created, False otherwise.

    Example:
        >>> plot_trajectory_animation(traj_3d, Path("results/traj.gif"))
    """
    try:
        import matplotlib.animation as animation
        from matplotlib import pyplot as plt
    except ImportError:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if trajectory.ndim == 3:
        trajectory = trajectory.squeeze(0)

    time_steps, n_dims = trajectory.shape
    if n_dims not in [2, 3]:
        return False

    # Calculate frame interval based on target duration
    n_frames = min(time_steps, int(fps * duration_seconds))
    frame_indices = np.linspace(0, time_steps - 1, n_frames, dtype=int)

    if n_dims == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(trajectory[:, 0].min() - 0.5, trajectory[:, 0].max() + 0.5)
        ax.set_ylim(trajectory[:, 1].min() - 0.5, trajectory[:, 1].max() + 0.5)
        ax.set_xlabel("Latent Dim 1")
        ax.set_ylabel("Latent Dim 2")
        ax.set_title(f"Latent Trajectory - {sample_id}")
        ax.grid(True, alpha=0.3)

        (line,) = ax.plot([], [], "b-", alpha=0.6, lw=2)
        (point,) = ax.plot([], [], "ro", markersize=8)

        def init():
            line.set_data([], [])
            point.set_data([], [])
            return line, point

        def animate(frame_idx):
            idx = frame_indices[frame_idx]
            line.set_data(trajectory[: idx + 1, 0], trajectory[: idx + 1, 1])
            point.set_data([trajectory[idx, 0]], [trajectory[idx, 1]])
            return line, point

        anim = animation.FuncAnimation(
            fig, animate, init_func=init, frames=n_frames, interval=1000 // fps, blit=True
        )
        anim.save(str(output_path), writer="pillow", fps=fps)
        plt.close(fig)
        return True

    elif n_dims == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(trajectory[:, 0].min() - 0.5, trajectory[:, 0].max() + 0.5)
        ax.set_ylim(trajectory[:, 1].min() - 0.5, trajectory[:, 1].max() + 0.5)
        ax.set_zlim(trajectory[:, 2].min() - 0.5, trajectory[:, 2].max() + 0.5)
        ax.set_xlabel("Latent Dim 1")
        ax.set_ylabel("Latent Dim 2")
        ax.set_zlabel("Latent Dim 3")
        ax.set_title(f"Latent Trajectory 3D - {sample_id}")

        (line,) = ax.plot([], [], [], "b-", alpha=0.6, lw=2)
        (point,) = ax.plot([], [], [], "ro", markersize=8)

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return line, point

        def animate(frame_idx):
            idx = frame_indices[frame_idx]
            line.set_data(trajectory[: idx + 1, 0], trajectory[: idx + 1, 1])
            line.set_3d_properties(trajectory[: idx + 1, 2])
            point.set_data([trajectory[idx, 0]], [trajectory[idx, 1]])
            point.set_3d_properties([trajectory[idx, 2]])
            return line, point

        anim = animation.FuncAnimation(
            fig, animate, init_func=init, frames=n_frames, interval=1000 // fps, blit=True
        )
        anim.save(str(output_path), writer="pillow", fps=fps)
        plt.close(fig)
        return True

    return False


def plot_trajectory_comparison_interactive(
    trajectories: Dict[str, np.ndarray],
    output_path: Path,
    *,
    title: str = "Trajectory Comparison",
    n_components: int = 3,
) -> Optional[go.Figure]:
    """
    Create interactive comparison of trajectories across different classes.

    Plots multiple trajectories in the same 3D space with different colors
    to compare latent dynamics between outcome classes.

    Args:
        trajectories: Dict mapping class names to trajectory arrays.
            Each array should be shape (N, T, D) or (T, D).
        output_path: Path to save HTML file.
        title: Plot title.
        n_components: Number of dimensions to plot (2 or 3, default 3).

    Returns:
        Plotly Figure object, or None if plotting fails.

    Example:
        >>> trajectories = {"Healthy": healthy_traj, "Acidosis": acidosis_traj}
        >>> plot_trajectory_comparison_interactive(trajectories, Path("results/compare.html"))
    """
    _check_plotly()

    if not trajectories:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Color palette for classes
    class_colors = {
        "Healthy": "#609966",
        "Acidosis": "#EB5B00",
        "HIE": "#112D4E",
        "Unknown": "#393E46",
    }
    default_colors = ["#3F72AF", "#FFB200", "#609966", "#00ADB5", "#EB5B00"]

    fig = go.Figure()

    for idx, (class_name, traj_data) in enumerate(trajectories.items()):
        if traj_data is None or traj_data.size == 0:
            continue

        # Get color for this class
        color = class_colors.get(class_name, default_colors[idx % len(default_colors)])

        # Handle different input shapes
        if traj_data.ndim == 3:
            # Multiple trajectories per class - plot each
            n_samples = min(traj_data.shape[0], 5)  # Limit to 5 per class
            for i in range(n_samples):
                traj = traj_data[i]
                time_steps = traj.shape[0]

                if n_components == 3 and traj.shape[1] >= 3:
                    fig.add_trace(
                        go.Scatter3d(
                            x=traj[:, 0],
                            y=traj[:, 1],
                            z=traj[:, 2],
                            mode="lines+markers",
                            marker=dict(size=3, color=color, opacity=0.7),
                            line=dict(color=color, width=2),
                            name=class_name if i == 0 else None,
                            legendgroup=class_name,
                            showlegend=(i == 0),
                            hovertemplate=f"{class_name} #{i+1}<br>Time: %{{text}}<extra></extra>",
                            text=[str(t) for t in range(time_steps)],
                        )
                    )
                elif traj.shape[1] >= 2:
                    fig.add_trace(
                        go.Scatter(
                            x=traj[:, 0],
                            y=traj[:, 1],
                            mode="lines+markers",
                            marker=dict(size=4, color=color, opacity=0.7),
                            line=dict(color=color, width=2),
                            name=class_name if i == 0 else None,
                            legendgroup=class_name,
                            showlegend=(i == 0),
                            hovertemplate=f"{class_name} #{i+1}<br>Time: %{{text}}<extra></extra>",
                            text=[str(t) for t in range(time_steps)],
                        )
                    )
        else:
            # Single trajectory
            traj = traj_data
            time_steps = traj.shape[0]

            if n_components == 3 and traj.shape[1] >= 3:
                fig.add_trace(
                    go.Scatter3d(
                        x=traj[:, 0],
                        y=traj[:, 1],
                        z=traj[:, 2],
                        mode="lines+markers",
                        marker=dict(size=3, color=color, opacity=0.7),
                        line=dict(color=color, width=2),
                        name=class_name,
                        hovertemplate=f"{class_name}<br>Time: %{{text}}<extra></extra>",
                        text=[str(t) for t in range(time_steps)],
                    )
                )
            elif traj.shape[1] >= 2:
                fig.add_trace(
                    go.Scatter(
                        x=traj[:, 0],
                        y=traj[:, 1],
                        mode="lines+markers",
                        marker=dict(size=4, color=color, opacity=0.7),
                        line=dict(color=color, width=2),
                        name=class_name,
                        hovertemplate=f"{class_name}<br>Time: %{{text}}<extra></extra>",
                        text=[str(t) for t in range(time_steps)],
                    )
                )

    # Determine if we're in 3D or 2D mode
    has_3d = any(
        isinstance(trace, go.Scatter3d) for trace in fig.data
    )

    if has_3d:
        fig.update_layout(
            title=dict(text=title, x=0.5),
            scene=dict(
                xaxis_title="Latent Dim 1",
                yaxis_title="Latent Dim 2",
                zaxis_title="Latent Dim 3",
                aspectmode="data",
            ),
            width=1000,
            height=800,
            legend=dict(
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.8)",
            ),
        )
    else:
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Latent Dim 1",
            yaxis_title="Latent Dim 2",
            width=900,
            height=700,
            legend=dict(
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.8)",
            ),
        )

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return fig
