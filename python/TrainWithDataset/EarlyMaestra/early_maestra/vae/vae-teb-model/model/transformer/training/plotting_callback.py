"""Diagnostic plotting callback for the Causal Multimodal Forecasting Transformer.

Generates a consolidated 10-row figure during validation showing model inputs,
internal representations, forecast quality, TE latent structure, and the
exported window embedding.  Style matches the publication-quality format in
``model/vae_teb_prediction/testing/plot_single_samples.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from loguru import logger

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from model.transformer.model import TransformerConfig, sample_anchors

# ---------------------------------------------------------------------------
# Colour palette (matches visualizers.py)
# ---------------------------------------------------------------------------
COLOR_BLUE = "#3F72AF"
COLOR_ORANGE = "#FFB200"
COLOR_GREEN = "#609966"
COLOR_SKY = "#00ADB5"
COLOR_PURPLE = "#112D4E"
COLOR_VERMILLION = "#EB5B00"
COLOR_GRAY = "#393E46"
COLOR_BLACK = "#222831"
COLOR_LIGHT_GRAY = "#EEEEEE"
COLOR_SAGE = "#9DC08B"
SAVE_DPI = 300  # lower than 600 for training diagnostics to save disk


# ---------------------------------------------------------------------------
# Style helpers (adapted from plot_single_samples.py)
# ---------------------------------------------------------------------------

def _apply_publication_style() -> None:
    """Apply publication-quality matplotlib style."""
    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": SAVE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.family": "serif",
        "font.serif": [
            "Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif",
        ],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": COLOR_BLACK,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.axisbelow": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.3,
        "grid.color": COLOR_LIGHT_GRAY,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.fancybox": False,
        "legend.edgecolor": COLOR_GRAY,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _style_axes(ax, *, grid: str = "major") -> None:
    """Apply clean styling with all four spines visible."""
    ax.set_axisbelow(True)
    if grid in ("both", "major"):
        ax.grid(True, linestyle="-", alpha=0.4, linewidth=0.4, color=COLOR_LIGHT_GRAY)
    if grid == "both":
        ax.grid(True, which="minor", linestyle=":", alpha=0.25, linewidth=0.3,
                color=COLOR_LIGHT_GRAY)
        ax.minorticks_on()
    for spine in ("top", "bottom", "left", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(COLOR_BLACK)
        ax.spines[spine].set_linewidth(0.6)


def _add_colorbar(fig, mappable, ax, *, label: Optional[str] = None):
    """Attach an aligned colorbar to *ax*."""
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.8, pad=0.02)
    if label:
        cbar.set_label(label, fontsize=8, color=COLOR_BLACK)
    cbar.ax.tick_params(labelsize=7, colors=COLOR_BLACK)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor(COLOR_LIGHT_GRAY)
    return cbar


def _heatmap(ax, data, *, cmap="bwr", origin="upper", title="", ylabel="Channel",
             xlabel="Time Steps", label="Value", fig=None):
    """Draw a coefficient heatmap on *ax*."""
    vabs = np.nanmax(np.abs(data)) or 1.0
    im = ax.imshow(data, aspect="auto", cmap=cmap, origin=origin,
                   vmin=-vabs, vmax=vabs)
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(False)
    for spine in ("top", "bottom", "left", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(COLOR_BLACK)
        ax.spines[spine].set_linewidth(0.6)
    if fig is not None:
        _add_colorbar(fig, im, ax, label=label)
    return im


# ---------------------------------------------------------------------------
# Validation-batch extraction (avoids importing from train.callbacks)
# ---------------------------------------------------------------------------

def _first_validation_batch(trainer) -> Optional[Any]:
    """Fetch the first validation batch from the trainer."""
    val_dls = trainer.val_dataloaders
    if val_dls is None:
        return None
    dl = val_dls[0] if isinstance(val_dls, (list, tuple)) else val_dls
    try:
        return next(iter(dl))
    except StopIteration:
        return None


# ---------------------------------------------------------------------------
# Core figure builder
# ---------------------------------------------------------------------------

def _build_diagnostic_figure(
    Y: torch.Tensor,
    U: torch.Tensor,
    outputs: Dict[str, Any],
    gate: torch.Tensor,
    e_win: torch.Tensor,
    config: TransformerConfig,
    sample_idx: int,
    epoch: int,
    beta: float,
    fhr_raw: Optional[torch.Tensor] = None,
    up_raw: Optional[torch.Tensor] = None,
) -> plt.Figure:
    """Build the consolidated diagnostic figure for one sample.

    Args:
        Y: FHR-ST input ``(B, T, d_f)``.
        U: UP-ST input ``(B, T, d_u)``.
        outputs: Training-mode forward output dict.
        gate: Gate activations ``(B, T, d_model)`` from the fusion module.
        e_win: Window embedding ``(B, embed_dim)`` from inference forward.
        config: Transformer configuration.
        sample_idx: Index of the sample within the batch.
        epoch: Current training epoch.
        beta: Current KL beta weight.
        fhr_raw: Optional raw FHR signal ``(B, L_raw)`` for the top plot.
        up_raw: Optional raw UP signal ``(B, L_raw)`` for the top plot.

    Returns:
        The matplotlib Figure object.
    """
    i = sample_idx
    B = Y.shape[0]
    K = outputs["anchor_indices"].shape[1]
    d_f = config.d_f
    d_z = config.d_z
    d = config.d_model
    g = config.guard_gap
    horizons = config.horizons
    h_max = config.max_horizon

    # Detach everything to numpy
    y_np = Y[i].detach().cpu().numpy()                    # (T, d_f)
    u_np = U[i].detach().cpu().numpy()                    # (T, d_u)
    hf_np = outputs["H_F"][i].detach().cpu().numpy()      # (T, d)
    hfu_np = outputs["H_FU"][i].detach().cpu().numpy()    # (T, d)
    gate_np = gate[i].detach().cpu().numpy()              # (T, d)
    anchors_np = outputs["anchor_indices"][i].detach().cpu().numpy()  # (K,)

    # TE latent params — reshape from (B*K, d_z) to (K, d_z) for this sample
    mu_post_all = outputs["mu_post"].detach().cpu().numpy()      # (B*K, d_z)
    mu_prior_all = outputs["mu_prior"].detach().cpu().numpy()
    logvar_post_all = outputs["logvar_post"].detach().cpu().numpy()
    logvar_prior_all = outputs["logvar_prior"].detach().cpu().numpy()

    # Extract this sample's anchors: indices [i*K : (i+1)*K]
    mu_post = mu_post_all[i * K:(i + 1) * K]                    # (K, d_z)
    mu_prior = mu_prior_all[i * K:(i + 1) * K]
    logvar_post = logvar_post_all[i * K:(i + 1) * K]
    logvar_prior = logvar_prior_all[i * K:(i + 1) * K]

    ewin_np = e_win[i].detach().cpu().numpy()                    # (embed_dim,)

    # Check if raw signals are available
    has_raw = (
        fhr_raw is not None
        and up_raw is not None
        and fhr_raw.shape[0] > i
    )

    n_rows = 13 if has_raw else 12
    _apply_publication_style()
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 3.2 * n_rows),
                             constrained_layout=True)
    row = 0

    # ---- Row 0 (optional): Raw FHR and UP ----
    if has_raw:
        ax = axes[row]
        fhr_np = fhr_raw[i].detach().cpu().numpy().ravel()
        up_np = up_raw[i].detach().cpu().numpy().ravel()
        fs = 4.0  # sampling rate
        t_raw = np.arange(len(fhr_np)) / fs

        ax.plot(t_raw, fhr_np, color=COLOR_BLUE, linewidth=0.8, label="FHR (bpm)")
        ax2 = ax.twinx()
        ax2.plot(t_raw, up_np, color=COLOR_GREEN, linewidth=0.8, label="UP (mmHg)")
        ax2.set_ylabel("UP (mmHg)", fontsize=8, color=COLOR_GREEN)
        ax2.tick_params(axis="y", labelcolor=COLOR_GREEN)

        ax.set_title("Raw FHR and UP Signals", fontsize=9, pad=6)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("FHR (bpm)", fontsize=8, color=COLOR_BLUE)
        ax.tick_params(axis="y", labelcolor=COLOR_BLUE)
        ax.set_xlim(t_raw[0], t_raw[-1])
        ax.margins(x=0.0)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc="upper right", fontsize=7, framealpha=0.95)
        _style_axes(ax, grid="both")
        row += 1

    # ---- FHR-ST Input ----
    _heatmap(axes[row], y_np.T, title="FHR Scattering Transform (Input)",
             ylabel="ST Channel", label="Coeff", fig=fig)
    row += 1

    # ---- UP-ST Input ----
    _heatmap(axes[row], u_np.T, title="UP Scattering Transform (Input)",
             ylabel="ST Channel", label="Coeff", fig=fig)
    row += 1

    # ---- FHR Encoder States ----
    _heatmap(axes[row], hf_np.T, cmap="bwr", origin="lower",
             title="FHR Encoder States (H_F)", ylabel="Hidden Dim",
             label="Activation", fig=fig)
    row += 1

    # ---- Fused Encoder States ----
    _heatmap(axes[row], hfu_np.T, cmap="bwr", origin="lower",
             title="Fused Encoder States (H_FU)", ylabel="Hidden Dim",
             label="Activation", fig=fig)
    row += 1

    # ---- Latent Dynamics (PCA of H_FU) ----
    ax = axes[row]
    # PCA: project 192-dim H_FU to top 3 principal components
    hfu_centered = hfu_np - hfu_np.mean(axis=0, keepdims=True)       # (T, d)
    try:
        U, S, Vt = np.linalg.svd(hfu_centered, full_matrices=False)
        n_components = min(3, Vt.shape[0])
        pca_proj = hfu_centered @ Vt[:n_components].T                # (T, 3)
        variance_explained = (S[:n_components] ** 2) / (S ** 2).sum()
        t_steps = np.arange(pca_proj.shape[0])
        pca_colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN]
        for c in range(n_components):
            ve = variance_explained[c] * 100
            ax.plot(t_steps, pca_proj[:, c], color=pca_colors[c],
                    linewidth=0.9, label=f"PC{c + 1} ({ve:.1f}%)")
        ax.set_title("Latent Dynamics \u2014 PCA of Fused Representation (H_FU)",
                      fontsize=9, pad=6)
    except np.linalg.LinAlgError:
        ax.text(0.5, 0.5, "SVD failed", ha="center", va="center", fontsize=9)
        ax.set_title("Latent Dynamics (PCA failed)", fontsize=9, pad=6)
    ax.set_xlabel("Time Steps", fontsize=8)
    ax.set_ylabel("PC Value", fontsize=8)
    ax.set_xlim(0, hfu_np.shape[0] - 1)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.95)
    _style_axes(ax, grid="both")
    row += 1

    # ---- Fusion Contribution: ||H_FU - H_F|| over time ----
    ax = axes[row]
    diff_norm = np.linalg.norm(hfu_np - hf_np, axis=-1)             # (T,)
    hf_norm = np.linalg.norm(hf_np, axis=-1)                         # (T,)
    relative_change = diff_norm / (hf_norm + 1e-8)                   # (T,)
    t_steps = np.arange(len(diff_norm))

    ax.plot(t_steps, diff_norm, color=COLOR_ORANGE, linewidth=0.9,
            label="||H_FU \u2212 H_F||")
    ax2 = ax.twinx()
    ax2.plot(t_steps, relative_change * 100, color=COLOR_PURPLE,
             linewidth=0.7, alpha=0.7, label="Relative change (%)")
    ax2.set_ylabel("Relative Change (%)", fontsize=8, color=COLOR_PURPLE)
    ax2.tick_params(axis="y", labelcolor=COLOR_PURPLE)

    # Mark anchor positions
    for a in anchors_np:
        ax.axvline(a, color=COLOR_VERMILLION, linestyle=":", linewidth=0.5,
                   alpha=0.5)

    ax.set_title(
        f"Fusion Contribution \u2014 ||H_FU \u2212 H_F|| over Time "
        f"(mean={diff_norm.mean():.3f})",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time Steps", fontsize=8)
    ax.set_ylabel("L2 Distance", fontsize=8, color=COLOR_ORANGE)
    ax.tick_params(axis="y", labelcolor=COLOR_ORANGE)
    ax.set_xlim(0, len(diff_norm) - 1)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=7, loc="upper right", framealpha=0.95)
    _style_axes(ax, grid="both")
    row += 1

    # ---- Gate Activations ----
    ax = axes[row]
    t = np.arange(gate_np.shape[0])
    gate_mean = gate_np.mean(axis=-1)
    gate_std = gate_np.std(axis=-1)
    ax.plot(t, gate_mean, color=COLOR_BLUE, linewidth=1.0, label="Mean gate")
    ax.fill_between(t, gate_mean - gate_std, gate_mean + gate_std,
                    color=COLOR_SKY, alpha=0.3, label="\u00b11 SD")
    ax.axhline(0.5, color=COLOR_GRAY, linestyle="--", linewidth=0.6, alpha=0.5)
    # Mark anchor positions
    for a in anchors_np:
        ax.axvline(a, color=COLOR_VERMILLION, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_title("Mean Gate Activation (UP \u2192 FHR Influence)", fontsize=9, pad=6)
    ax.set_xlabel("Time Steps", fontsize=8)
    ax.set_ylabel("Gate Value", fontsize=8)
    ax.set_xlim(0, len(t) - 1)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.95)
    _style_axes(ax, grid="both")
    row += 1

    # ---- Anchor Forecast — Self vs Fused vs TE vs Target ----
    ax = axes[row]
    # Pick the first anchor for visualization, use the largest horizon
    a0 = int(anchors_np[0])
    target_start = a0 + g + 1
    target_end = target_start + h_max
    t_forecast = np.arange(target_start, min(target_end, y_np.shape[0]))
    ch = 0  # visualize channel 0

    if target_end <= y_np.shape[0]:
        gt_segment = y_np[target_start:target_end, ch]
        ax.plot(t_forecast, gt_segment, color=COLOR_BLACK, linewidth=1.2,
                label="Ground Truth", zorder=4)

        # Predictions for h_max — take first anchor's prediction
        for head_name, head_key, color in [
            ("Self-only", "Y_hat_self", COLOR_BLUE),
            ("Fused", "Y_hat_fus", COLOR_ORANGE),
            ("TE-augmented", "Y_hat_te", COLOR_GREEN),
        ]:
            pred = outputs[head_key][h_max][i * K].detach().cpu().numpy()[:, ch]
            ax.plot(t_forecast[:len(pred)], pred, color=color, linewidth=1.0,
                    label=head_name, alpha=0.85)

    # Shade the context window
    ctx_start = max(0, a0 - config.ctx_len + 1)
    ax.axvspan(ctx_start, a0, alpha=0.08, color=COLOR_PURPLE, label="Context")
    ax.axvline(a0, color=COLOR_VERMILLION, linewidth=1.0, linestyle="--", label="Anchor")
    ax.axvspan(a0, target_start, alpha=0.06, color=COLOR_GRAY, label="Guard gap")

    ax.set_title(
        f"Forecast at Anchor a={a0} (channel {ch}, horizon={h_max})",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time Steps", fontsize=8)
    ax.set_ylabel("Value", fontsize=8)
    ax.legend(fontsize=6, loc="upper right", ncol=3, framealpha=0.95)
    _style_axes(ax, grid="both")
    row += 1

    # ---- Forecast Error by Horizon ----
    ax = axes[row]
    head_names = ["Self-only", "Fused", "TE-aug"]
    head_keys = ["Y_hat_self", "Y_hat_fus", "Y_hat_te"]
    head_colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN]
    bar_width = 0.25
    x_positions = np.arange(len(horizons))

    for j, (hname, hkey, hcolor) in enumerate(zip(head_names, head_keys, head_colors)):
        maes = []
        for h in horizons:
            pred_h = outputs[hkey][h][i * K:(i + 1) * K].detach().cpu().numpy()  # (K, h, d_f)
            # Extract target
            target_blocks = []
            for ki in range(K):
                a_ki = int(anchors_np[ki])
                ts = a_ki + g + 1
                te = ts + h
                if te <= y_np.shape[0]:
                    target_blocks.append(y_np[ts:te])
            if target_blocks:
                target_arr = np.stack(target_blocks)  # (K, h, d_f)
                mae = np.mean(np.abs(pred_h[:len(target_blocks)] - target_arr))
            else:
                mae = 0.0
            maes.append(mae)
        ax.bar(x_positions + j * bar_width, maes, bar_width, color=hcolor,
               alpha=0.85, edgecolor=COLOR_BLACK, linewidth=0.3, label=hname)

    ax.set_xticks(x_positions + bar_width)
    ax.set_xticklabels([f"h={h}" for h in horizons])
    ax.set_title("Mean Absolute Error by Horizon and Head", fontsize=9, pad=6)
    ax.set_xlabel("Prediction Horizon", fontsize=8)
    ax.set_ylabel("MAE", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.95)
    _style_axes(ax, grid="major")
    row += 1

    # ---- TE Latent — Posterior & Prior Means ----
    ax = axes[row]
    combined = np.concatenate([mu_post.T, mu_prior.T], axis=0)  # (2*d_z, K)
    vabs = np.nanmax(np.abs(combined)) or 1.0
    im = ax.imshow(combined, aspect="auto", cmap="bwr", origin="lower",
                   vmin=-vabs, vmax=vabs)
    ax.axhline(d_z - 0.5, color="white", linewidth=1.5, linestyle="--")
    ax.set_yticks([d_z // 2, d_z + d_z // 2])
    ax.set_yticklabels(["Posterior \u03bc", "Prior \u03bc\u2070"])
    ax.set_xlabel("Anchor Index", fontsize=8)
    ax.set_title("TE Latent: Posterior vs Prior Means at Anchors", fontsize=9, pad=6)
    ax.grid(False)
    for spine in ("top", "bottom", "left", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(COLOR_BLACK)
        ax.spines[spine].set_linewidth(0.6)
    _add_colorbar(fig, im, ax, label="Value")
    row += 1

    # ---- KL Divergence per Anchor ----
    ax = axes[row]
    # Per-anchor, per-dimension KL
    kl_per_dim = 0.5 * (
        logvar_prior - logvar_post
        + (np.exp(logvar_post) + (mu_post - mu_prior) ** 2) / np.exp(logvar_prior)
        - 1.0
    )  # (K, d_z)
    kl_per_anchor = kl_per_dim.sum(axis=-1)  # (K,)

    anchor_labels = [f"a={int(a)}" for a in anchors_np]
    bars = ax.bar(anchor_labels, kl_per_anchor, color=COLOR_PURPLE, alpha=0.85,
                  edgecolor=COLOR_BLACK, linewidth=0.3)
    # Value labels
    for bar, val in zip(bars, kl_per_anchor):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=6)
    ax.set_title(
        f"KL(posterior || prior) per Anchor (\u03b2_transfer={beta:.6f})",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Anchor", fontsize=8)
    ax.set_ylabel("KL Divergence (nats)", fontsize=8)
    _style_axes(ax, grid="major")
    row += 1

    # ---- Window Embedding ----
    ax = axes[row]
    n_total = len(ewin_np)
    # v2 component boundaries
    d_z_s = config.d_z_self
    d_z_t = config.d_z_transfer
    boundary_f = 2 * d
    boundary_fu = boundary_f + 10 * d
    boundary_self = boundary_fu + 3 * d_z_s

    x_emb = np.arange(n_total)
    ax.bar(x_emb[:boundary_f], np.abs(ewin_np[:boundary_f]),
           color=COLOR_BLUE, alpha=0.7, width=1.0, label=f"e_F ({boundary_f}d)")
    ax.bar(x_emb[boundary_f:boundary_fu], np.abs(ewin_np[boundary_f:boundary_fu]),
           color=COLOR_ORANGE, alpha=0.7, width=1.0, label=f"e_FU ({boundary_fu - boundary_f}d)")
    ax.bar(x_emb[boundary_fu:boundary_self], np.abs(ewin_np[boundary_fu:boundary_self]),
           color=COLOR_VERMILLION, alpha=0.7, width=1.0, label=f"e_self ({boundary_self - boundary_fu}d)")
    ax.bar(x_emb[boundary_self:], np.abs(ewin_np[boundary_self:]),
           color=COLOR_GREEN, alpha=0.7, width=1.0, label=f"e_TE ({n_total - boundary_self}d)")

    # Section dividers
    ax.axvline(boundary_f - 0.5, color=COLOR_BLACK, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(boundary_fu - 0.5, color=COLOR_BLACK, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(boundary_self - 0.5, color=COLOR_BLACK, linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_title("Window Embedding (e_win) \u2014 Component Magnitudes", fontsize=9, pad=6)
    ax.set_xlabel("Embedding Dimension", fontsize=8)
    ax.set_ylabel("|Activation|", fontsize=8)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.95)
    _style_axes(ax, grid="major")

    fig.suptitle(
        f"Transformer Diagnostics \u2014 Epoch {epoch}, Sample {sample_idx}",
        fontsize=12, fontweight="normal", y=1.005, color=COLOR_PURPLE,
    )
    return fig


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class TransformerPlotCallback(Callback):
    """Periodic diagnostic plots during causal transformer training.

    Generates a consolidated 10-row figure per validation sample showing
    model inputs, encoder states, gate activations, forecasts from all three
    heads, TE latent structure, KL divergence, and the window embedding.

    Args:
        output_dir: Directory for saving plot files.
        plot_frequency: Generate plots every N epochs.
        num_examples: Number of validation samples to plot each time.
        file_format: Image format (``"pdf"`` or ``"png"``).
        mlflow_logger: Optional MLflow logger for artifact logging.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        plot_frequency: int = 5,
        num_examples: int = 2,
        *,
        file_format: str = "pdf",
        mlflow_logger=None,
    ) -> None:
        super().__init__()
        if not _HAS_MPL:
            logger.warning("matplotlib not available; TransformerPlotCallback disabled.")
        self.output_dir = Path(output_dir) / "transformer_diagnostics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_frequency = max(1, int(plot_frequency))
        self.num_examples = max(1, int(num_examples))
        self.file_format = file_format.lower()
        self._mlflow_logger = mlflow_logger

    def _log_artifact(self, path: Path) -> None:
        """Log a file as an MLflow artifact if a logger is available."""
        if self._mlflow_logger is None:
            return
        try:
            self._mlflow_logger.experiment.log_artifact(
                self._mlflow_logger.run_id, str(path),
            )
        except Exception as exc:
            logger.debug(f"MLflow artifact logging failed: {exc}")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Generate diagnostic plots at the configured frequency.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``PlCausalTransformer`` Lightning module.
        """
        if not _HAS_MPL:
            return
        if not trainer.is_global_zero:
            return
        if trainer.sanity_checking:
            return
        if trainer.current_epoch % self.plot_frequency != 0:
            return

        batch = _first_validation_batch(trainer)
        if batch is None:
            logger.debug("No validation batch available for plotting.")
            return

        try:
            self._generate_plots(batch, pl_module, trainer.current_epoch)
        except Exception as exc:
            logger.warning(f"TransformerPlotCallback failed: {exc}")

    @torch.no_grad()
    def _generate_plots(self, batch, pl_module, epoch: int) -> None:
        """Run inference and build diagnostic figures.

        Args:
            batch: A validation batch (AttributeDict).
            pl_module: The ``PlCausalTransformer`` module.
            epoch: Current epoch number.
        """
        model = pl_module.orig_model
        config = pl_module._transformer_config
        device = pl_module.device

        Y = batch["fhr_st"].to(device)
        U = batch["up_st"].to(device)
        num_samples = min(self.num_examples, Y.shape[0])

        # Optional raw signals (available if load_fields includes fhr/up)
        fhr_raw = batch.get("fhr")
        up_raw = batch.get("up")
        if fhr_raw is not None:
            fhr_raw = fhr_raw.to(device)
        if up_raw is not None:
            up_raw = up_raw.to(device)

        # --- Training-mode forward (with anchors) ---
        anchors = sample_anchors(Y, U, config, training=False)
        model.eval()
        outputs = model(Y, U, anchor_indices=anchors)

        # --- Gate extraction ---
        F_out = model.fhr_stem(Y)
        S_out = model.up_stem(U)
        H_F = model.fhr_encoder(F_out)
        H_U = model.up_encoder(S_out)
        context = model.fusion.cross_attns[0](target=H_F, source=H_U)
        gate = torch.sigmoid(
            model.fusion.gate_projs[0](torch.cat([H_F, context], dim=-1))
        )

        # --- Inference-mode forward (window embedding) ---
        inf_out = model(Y, U)
        e_win = inf_out["e_win"]

        # Current betas (v2: dual KL scheduling)
        beta_transfer = getattr(pl_module, "_current_beta_transfer", 0.0)
        beta_self = getattr(pl_module, "_current_beta_self", 0.0)
        # Pass transfer beta for display (backward compat with figure builder)
        beta = beta_transfer

        for s in range(num_samples):
            fig = _build_diagnostic_figure(
                Y=Y, U=U, outputs=outputs, gate=gate, e_win=e_win,
                config=config, sample_idx=s, epoch=epoch, beta=beta,
                fhr_raw=fhr_raw, up_raw=up_raw,
            )
            fname = (
                f"transformer_diagnostics_epoch{epoch:04d}"
                f"_sample{s}.{self.file_format}"
            )
            save_path = self.output_dir / fname
            fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
            plt.close(fig)
            self._log_artifact(save_path)

        logger.info(
            f"Saved {num_samples} diagnostic plots for epoch {epoch} "
            f"to {self.output_dir}"
        )
