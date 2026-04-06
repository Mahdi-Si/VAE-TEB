"""Comprehensive matplotlib plotting module for the transformer testing pipeline.

Provides ~45 plotting functions organized by category:
    1. Per-sample diagnostic
    2. Forecasting
    3. TE coupling
    4. Representation
    5. Trajectory
    6. Cross-class comparisons
    7. Dataset statistics

All functions take numpy/DataFrame data + output path, save the figure, and
return the output path as a ``pathlib.Path``.  Every function uses the shared
style module for publication-quality consistency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from .style import (
    apply_publication_style, style_axes, add_colorbar, heatmap, save_figure,
    get_class_colors, COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_SKY,
    COLOR_PURPLE, COLOR_VERMILLION, COLOR_GRAY, COLOR_BLACK, COLOR_LIGHT_GRAY,
    COLOR_SAGE, SAVE_DPI, HEAD_COLORS, HEAD_LABELS,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HEAD_NAMES = ("self", "fused", "te")
_HEAD_KEYS = ("Y_hat_self", "Y_hat_fus", "Y_hat_te")
_PCA_COLORS = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN]


def _ensure_path(path: Union[str, Path]) -> Path:
    """Convert *path* to ``Path`` and create parent directories.

    Args:
        path: File path string or ``Path`` object.

    Returns:
        Resolved ``Path`` with existing parent directory.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _pca_svd(data: np.ndarray, n_components: int = 3):
    """Compute PCA via SVD (centered).

    Args:
        data: 2-D array ``(N, D)``.
        n_components: Number of principal components to retain.

    Returns:
        Tuple of ``(projections, variance_explained)`` where projections is
        ``(N, n_components)`` and variance_explained is ``(n_components,)``.
        Returns ``(None, None)`` on failure.
    """
    centered = data - data.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        nc = min(n_components, Vt.shape[0])
        proj = centered @ Vt[:nc].T
        ve = (S[:nc] ** 2) / max((S ** 2).sum(), 1e-12)
        return proj, ve
    except np.linalg.LinAlgError:
        return None, None


def _try_projection(data: np.ndarray, method: str = "pca",
                    n_components: int = 2):
    """2-D (or 3-D) projection with fallback to PCA.

    Args:
        data: ``(N, D)`` array.
        method: One of ``"pca"``, ``"umap"``, ``"tsne"``.
        n_components: Target dimensionality.

    Returns:
        Tuple of ``(embedding, method_used)`` where embedding is
        ``(N, n_components)`` and *method_used* is the string name of the
        method that actually succeeded.
    """
    if data.shape[0] < 2:
        return np.zeros((data.shape[0], n_components)), "pca"

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            return reducer.fit_transform(data), "umap"
        except Exception:
            pass
    elif method == "tsne":
        try:
            from sklearn.manifold import TSNE
            perp = min(30, data.shape[0] - 1)
            tsne = TSNE(n_components=n_components, perplexity=max(perp, 2),
                        random_state=42)
            return tsne.fit_transform(data), "tsne"
        except Exception:
            pass

    proj, ve = _pca_svd(data, n_components)
    if proj is not None:
        return proj, "pca"
    return np.zeros((data.shape[0], n_components)), "pca"


def _overlaid_histograms(ax, data_dict: Dict[str, np.ndarray],
                         class_colors: Dict[str, str], *,
                         xlabel: str = "", bins: int = 40) -> None:
    """Draw overlaid histograms for multiple classes on *ax*.

    Args:
        ax: Matplotlib Axes.
        data_dict: ``{class_name: 1-D array}``.
        class_colors: ``{class_name: color_string}``.
        xlabel: X-axis label.
        bins: Number of histogram bins.
    """
    for cls, vals in data_dict.items():
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, alpha=0.6, color=class_colors.get(cls, COLOR_GRAY),
                edgecolor=COLOR_BLACK, linewidth=0.3, label=cls, density=True)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.95)


# ===================================================================
# Category 1: Per-sample diagnostic
# ===================================================================

def plot_sample_diagnostic(sample_data: dict, output_path, config) -> Path:
    """Create a 17-row diagnostic figure for a single sample.

    Mirrors the training callback diagnostic but with additional rows for
    per-channel error, TE residual analysis, and embedding decomposition.

    Rows:
        0: Raw FHR+UP twin axis (if available)
        1: FHR scattering heatmap (bwr)
        2: UP scattering heatmap (bwr)
        3: H_F encoder states heatmap
        4: H_U encoder states heatmap
        5: H_FU fused encoder states heatmap
        6: PCA of H_F (3 PCs with variance explained)
        7: PCA of H_FU (3 PCs with variance explained)
        8: Fusion contribution (L2 dist + relative change, twin axis)
        9: Gate activation (mean +/- SD, anchors marked)
        10: Forecast at all horizons (3 subplots: h=8,15,30, each with
            GT vs 3 heads)
        11: MAE by horizon (grouped bar: 3 heads x 3 horizons)
        12: Per-channel MAE heatmap (43 ch x 3 horizons, fused head)
        13: TE latent posterior vs prior (split heatmap)
        14: KL per anchor (bar chart)
        15: TE residual magnitude (bar per horizon per anchor)
        16: Window embedding components (stacked bars: e_F, e_FU, e_TE)

    Args:
        sample_data: Dictionary with keys ``Y``, ``U``, ``fhr_raw``
            (optional), ``up_raw`` (optional), ``H_F``, ``H_U``, ``H_FU``,
            ``gate``, ``anchor_indices``, ``Y_hat_self``, ``Y_hat_fus``,
            ``Y_hat_te``, ``R_hat``, ``mu_post``, ``mu_prior``,
            ``logvar_post``, ``logvar_prior``, ``e_win``, ``guid``,
            ``epoch``, ``class_label``.
        output_path: File path for the saved figure.
        config: ``TransformerConfig`` instance (has ``horizons``,
            ``guard_gap``, ``d_model``, ``d_z``, ``ctx_len``, ``d_f``).

    Returns:
        ``Path`` to the saved figure.
    """
    output_path = _ensure_path(output_path)
    apply_publication_style()

    sd = sample_data
    has_raw = "fhr_raw" in sd and "up_raw" in sd

    # Total row count: 17 if raw signals present, 16 otherwise
    # Row 10 uses 3 subplots, so we use gridspec
    n_base_rows = 17 if has_raw else 16

    fig = plt.figure(figsize=(16, 3.2 * n_base_rows), constrained_layout=True)
    gs = fig.add_gridspec(n_base_rows, 3)
    row = 0

    y_np = sd["Y"]          # (T, d_f)
    u_np = sd["U"]          # (T, d_u)
    hf_np = sd["H_F"]       # (T, d)
    hu_np = sd["H_U"]       # (T, d)
    hfu_np = sd["H_FU"]     # (T, d)
    gate_np = sd["gate"]    # (T, d)
    anchors_np = sd["anchor_indices"]  # (K,)
    mu_post = sd["mu_post"]          # (K, d_z)
    mu_prior = sd["mu_prior"]        # (K, d_z)
    logvar_post = sd["logvar_post"]  # (K, d_z)
    logvar_prior = sd["logvar_prior"]  # (K, d_z)
    e_win_np = sd["e_win"]             # (embed_dim,)
    horizons = config.horizons
    g = config.guard_gap
    d_z = config.d_z
    d = config.d_model
    K = anchors_np.shape[0]

    guid = sd.get("guid", "?")
    epoch = sd.get("epoch", "?")
    cls_label = sd.get("class_label", "?")

    # ---- Row 0 (optional): Raw FHR and UP ----
    if has_raw:
        ax = fig.add_subplot(gs[row, :])
        fhr_np = sd["fhr_raw"].ravel()
        up_np_raw = sd["up_raw"].ravel()
        fs = 4.0
        t_raw = np.arange(len(fhr_np)) / fs
        ax.plot(t_raw, fhr_np, color=COLOR_BLUE, linewidth=0.8, label="FHR (bpm)")
        ax2 = ax.twinx()
        ax2.plot(t_raw, up_np_raw, color=COLOR_GREEN, linewidth=0.8,
                 label="UP (mmHg)")
        ax2.set_ylabel("UP (mmHg)", fontsize=8, color=COLOR_GREEN)
        ax2.tick_params(axis="y", labelcolor=COLOR_GREEN)
        ax.set_title("Raw FHR and UP Signals", fontsize=9, pad=6)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("FHR (bpm)", fontsize=8, color=COLOR_BLUE)
        ax.tick_params(axis="y", labelcolor=COLOR_BLUE)
        ax.set_xlim(t_raw[0], t_raw[-1])
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
                  fontsize=7, framealpha=0.95)
        style_axes(ax, grid="both")
        row += 1

    # ---- Row 1: FHR-ST heatmap ----
    ax = fig.add_subplot(gs[row, :])
    heatmap(ax, y_np.T, title="FHR Scattering Transform (Input)",
            ylabel="ST Channel", label="Coeff", fig=fig)
    row += 1

    # ---- Row 2: UP-ST heatmap ----
    ax = fig.add_subplot(gs[row, :])
    heatmap(ax, u_np.T, title="UP Scattering Transform (Input)",
            ylabel="ST Channel", label="Coeff", fig=fig)
    row += 1

    # ---- Row 3: H_F ----
    ax = fig.add_subplot(gs[row, :])
    heatmap(ax, hf_np.T, cmap="bwr", origin="lower",
            title="FHR Encoder States (H_F)", ylabel="Hidden Dim",
            label="Activation", fig=fig)
    row += 1

    # ---- Row 4: H_U ----
    ax = fig.add_subplot(gs[row, :])
    heatmap(ax, hu_np.T, cmap="bwr", origin="lower",
            title="UP Encoder States (H_U)", ylabel="Hidden Dim",
            label="Activation", fig=fig)
    row += 1

    # ---- Row 5: H_FU ----
    ax = fig.add_subplot(gs[row, :])
    heatmap(ax, hfu_np.T, cmap="bwr", origin="lower",
            title="Fused Encoder States (H_FU)", ylabel="Hidden Dim",
            label="Activation", fig=fig)
    row += 1

    # ---- Row 6: PCA of H_F ----
    ax = fig.add_subplot(gs[row, :])
    proj_hf, ve_hf = _pca_svd(hf_np, 3)
    if proj_hf is not None:
        t_steps = np.arange(proj_hf.shape[0])
        for c in range(proj_hf.shape[1]):
            ax.plot(t_steps, proj_hf[:, c], color=_PCA_COLORS[c],
                    linewidth=0.9, label=f"PC{c+1} ({ve_hf[c]*100:.1f}%)")
        ax.set_title("PCA of FHR Encoder States (H_F)", fontsize=9, pad=6)
    else:
        ax.text(0.5, 0.5, "SVD failed", ha="center", va="center", fontsize=9)
        ax.set_title("PCA of H_F (SVD failed)", fontsize=9, pad=6)
    ax.set_xlabel("Time Steps", fontsize=8)
    ax.set_ylabel("PC Value", fontsize=8)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.95)
    style_axes(ax, grid="both")
    row += 1

    # ---- Row 7: PCA of H_FU ----
    ax = fig.add_subplot(gs[row, :])
    proj_hfu, ve_hfu = _pca_svd(hfu_np, 3)
    if proj_hfu is not None:
        t_steps = np.arange(proj_hfu.shape[0])
        for c in range(proj_hfu.shape[1]):
            ax.plot(t_steps, proj_hfu[:, c], color=_PCA_COLORS[c],
                    linewidth=0.9, label=f"PC{c+1} ({ve_hfu[c]*100:.1f}%)")
        ax.set_title("PCA of Fused Encoder States (H_FU)", fontsize=9, pad=6)
    else:
        ax.text(0.5, 0.5, "SVD failed", ha="center", va="center", fontsize=9)
        ax.set_title("PCA of H_FU (SVD failed)", fontsize=9, pad=6)
    ax.set_xlabel("Time Steps", fontsize=8)
    ax.set_ylabel("PC Value", fontsize=8)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.95)
    style_axes(ax, grid="both")
    row += 1

    # ---- Row 8: Fusion contribution ----
    ax = fig.add_subplot(gs[row, :])
    diff_norm = np.linalg.norm(hfu_np - hf_np, axis=-1)
    hf_norm = np.linalg.norm(hf_np, axis=-1)
    relative_change = diff_norm / (hf_norm + 1e-8)
    t_steps = np.arange(len(diff_norm))
    ax.plot(t_steps, diff_norm, color=COLOR_ORANGE, linewidth=0.9,
            label="||H_FU - H_F||")
    ax2 = ax.twinx()
    ax2.plot(t_steps, relative_change * 100, color=COLOR_PURPLE,
             linewidth=0.7, alpha=0.7, label="Relative change (%)")
    ax2.set_ylabel("Relative Change (%)", fontsize=8, color=COLOR_PURPLE)
    ax2.tick_params(axis="y", labelcolor=COLOR_PURPLE)
    for a in anchors_np:
        ax.axvline(a, color=COLOR_VERMILLION, linestyle=":", linewidth=0.5,
                   alpha=0.5)
    ax.set_title(f"Fusion Contribution (mean={diff_norm.mean():.3f})",
                 fontsize=9, pad=6)
    ax.set_xlabel("Time Steps", fontsize=8)
    ax.set_ylabel("L2 Distance", fontsize=8, color=COLOR_ORANGE)
    ax.tick_params(axis="y", labelcolor=COLOR_ORANGE)
    ax.set_xlim(0, len(diff_norm) - 1)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
              loc="upper right", framealpha=0.95)
    style_axes(ax, grid="both")
    row += 1

    # ---- Row 9: Gate activation ----
    ax = fig.add_subplot(gs[row, :])
    t = np.arange(gate_np.shape[0])
    gate_mean = gate_np.mean(axis=-1)
    gate_std = gate_np.std(axis=-1)
    ax.plot(t, gate_mean, color=COLOR_BLUE, linewidth=1.0, label="Mean gate")
    ax.fill_between(t, gate_mean - gate_std, gate_mean + gate_std,
                    color=COLOR_SKY, alpha=0.3, label="\u00b11 SD")
    ax.axhline(0.5, color=COLOR_GRAY, linestyle="--", linewidth=0.6, alpha=0.5)
    for a in anchors_np:
        ax.axvline(a, color=COLOR_VERMILLION, linestyle=":", linewidth=0.5,
                   alpha=0.6)
    ax.set_title("Mean Gate Activation (UP \u2192 FHR Influence)", fontsize=9,
                 pad=6)
    ax.set_xlabel("Time Steps", fontsize=8)
    ax.set_ylabel("Gate Value", fontsize=8)
    ax.set_xlim(0, len(t) - 1)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.95)
    style_axes(ax, grid="both")
    row += 1

    # ---- Row 10: Forecast at all horizons (3 subplots) ----
    for hi, h in enumerate(horizons):
        ax = fig.add_subplot(gs[row, hi])
        a0 = int(anchors_np[0])
        ts = a0 + g + 1
        te = ts + h
        t_fc = np.arange(ts, min(te, y_np.shape[0]))
        ch = 0
        if te <= y_np.shape[0]:
            gt = y_np[ts:te, ch]
            ax.plot(t_fc, gt, color=COLOR_BLACK, linewidth=1.2,
                    label="GT", zorder=4)
            for head_name, head_key in zip(_HEAD_NAMES, _HEAD_KEYS):
                pred = sd[head_key][h][0, :, ch]  # first anchor
                ax.plot(t_fc[:len(pred)], pred[:len(t_fc)],
                        color=HEAD_COLORS[head_name], linewidth=1.0,
                        label=HEAD_LABELS[head_name], alpha=0.85)
        ax.set_title(f"h={h}", fontsize=9, pad=4)
        ax.set_xlabel("Time Steps", fontsize=7)
        if hi == 0:
            ax.set_ylabel("Value", fontsize=8)
        ax.legend(fontsize=5, loc="upper right", framealpha=0.9)
        style_axes(ax, grid="both")
    row += 1

    # ---- Row 11: MAE by horizon ----
    ax = fig.add_subplot(gs[row, :])
    bar_width = 0.25
    x_pos = np.arange(len(horizons))
    for j, head_name in enumerate(_HEAD_NAMES):
        head_key = _HEAD_KEYS[j]
        maes = []
        for h in horizons:
            pred_h = sd[head_key][h]  # (K, h, d_f)
            targets = []
            for ki in range(K):
                a_ki = int(anchors_np[ki])
                ts = a_ki + g + 1
                te_h = ts + h
                if te_h <= y_np.shape[0]:
                    targets.append(y_np[ts:te_h])
            if targets:
                tar = np.stack(targets)
                mae_val = np.mean(np.abs(pred_h[:len(targets)] - tar))
            else:
                mae_val = 0.0
            maes.append(mae_val)
        ax.bar(x_pos + j * bar_width, maes, bar_width,
               color=HEAD_COLORS[head_name], alpha=0.85,
               edgecolor=COLOR_BLACK, linewidth=0.3,
               label=HEAD_LABELS[head_name])
    ax.set_xticks(x_pos + bar_width)
    ax.set_xticklabels([f"h={h}" for h in horizons])
    ax.set_title("MAE by Horizon and Head", fontsize=9, pad=6)
    ax.set_xlabel("Prediction Horizon", fontsize=8)
    ax.set_ylabel("MAE", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.95)
    style_axes(ax, grid="major")
    row += 1

    # ---- Row 12: Per-channel MAE heatmap (fused head) ----
    ax = fig.add_subplot(gs[row, :])
    ch_mae = np.zeros((len(horizons), y_np.shape[1]))
    for hi, h in enumerate(horizons):
        pred_h = sd["Y_hat_fus"][h]  # (K, h, d_f)
        targets = []
        for ki in range(K):
            a_ki = int(anchors_np[ki])
            ts = a_ki + g + 1
            te_h = ts + h
            if te_h <= y_np.shape[0]:
                targets.append(y_np[ts:te_h])
        if targets:
            tar = np.stack(targets)
            ch_mae[hi] = np.mean(np.abs(pred_h[:len(targets)] - tar),
                                 axis=(0, 1))
    heatmap(ax, ch_mae, cmap="YlOrRd", origin="upper",
            title="Per-channel MAE (Fused Head)", ylabel="Horizon",
            xlabel="Channel", label="MAE", fig=fig)
    ax.set_yticks(np.arange(len(horizons)))
    ax.set_yticklabels([f"h={h}" for h in horizons])
    row += 1

    # ---- Row 13: TE latent posterior vs prior (split heatmap) ----
    ax = fig.add_subplot(gs[row, :])
    combined = np.concatenate([mu_post.T, mu_prior.T], axis=0)
    vabs = np.nanmax(np.abs(combined)) or 1.0
    im = ax.imshow(combined, aspect="auto", cmap="bwr", origin="lower",
                   vmin=-vabs, vmax=vabs)
    ax.axhline(d_z - 0.5, color="white", linewidth=1.5, linestyle="--")
    ax.set_yticks([d_z // 2, d_z + d_z // 2])
    ax.set_yticklabels(["Posterior \u03bc", "Prior \u03bc\u2070"])
    ax.set_xlabel("Anchor Index", fontsize=8)
    ax.set_title("TE Latent: Posterior vs Prior Means", fontsize=9, pad=6)
    ax.grid(False)
    add_colorbar(fig, im, ax, label="Value")
    row += 1

    # ---- Row 14: KL per anchor ----
    ax = fig.add_subplot(gs[row, :])
    kl_per_dim = 0.5 * (
        logvar_prior - logvar_post
        + (np.exp(logvar_post) + (mu_post - mu_prior) ** 2)
        / np.exp(logvar_prior) - 1.0
    )
    kl_per_anchor = kl_per_dim.sum(axis=-1)
    anchor_labels = [f"a={int(a)}" for a in anchors_np]
    bars = ax.bar(anchor_labels, kl_per_anchor, color=COLOR_PURPLE,
                  alpha=0.85, edgecolor=COLOR_BLACK, linewidth=0.3)
    for bar, val in zip(bars, kl_per_anchor):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002, f"{val:.4f}",
                ha="center", va="bottom", fontsize=6)
    ax.set_title("KL(posterior || prior) per Anchor", fontsize=9, pad=6)
    ax.set_xlabel("Anchor", fontsize=8)
    ax.set_ylabel("KL (nats)", fontsize=8)
    style_axes(ax, grid="major")
    row += 1

    # ---- Row 15: TE residual magnitude ----
    ax = fig.add_subplot(gs[row, :])
    bar_width_r = 0.8 / len(horizons)
    for hi, h in enumerate(horizons):
        r_h = sd["R_hat"][h]  # (K, h, d_f)
        r_norm = np.linalg.norm(r_h.reshape(K, -1), axis=-1)
        x_r = np.arange(K) + hi * bar_width_r
        ax.bar(x_r, r_norm, bar_width_r,
               color=_PCA_COLORS[hi % len(_PCA_COLORS)], alpha=0.8,
               edgecolor=COLOR_BLACK, linewidth=0.3, label=f"h={h}")
    ax.set_xticks(np.arange(K) + bar_width_r * (len(horizons) - 1) / 2)
    ax.set_xticklabels(anchor_labels)
    ax.set_title("TE Residual Magnitude per Horizon per Anchor", fontsize=9,
                 pad=6)
    ax.set_xlabel("Anchor", fontsize=8)
    ax.set_ylabel("L2 Norm", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.95)
    style_axes(ax, grid="major")
    row += 1

    # ---- Row 16: Window embedding components ----
    ax = fig.add_subplot(gs[row, :])
    n_total = len(e_win_np)
    boundary_f = 2 * d
    boundary_fu = boundary_f + 6 * d
    x_emb = np.arange(n_total)
    ax.bar(x_emb[:boundary_f], np.abs(e_win_np[:boundary_f]),
           color=COLOR_BLUE, alpha=0.7, width=1.0,
           label=f"e_F ({boundary_f}d)")
    ax.bar(x_emb[boundary_f:boundary_fu],
           np.abs(e_win_np[boundary_f:boundary_fu]),
           color=COLOR_ORANGE, alpha=0.7, width=1.0,
           label=f"e_FU ({boundary_fu - boundary_f}d)")
    ax.bar(x_emb[boundary_fu:], np.abs(e_win_np[boundary_fu:]),
           color=COLOR_GREEN, alpha=0.7, width=1.0,
           label=f"e_TE ({n_total - boundary_fu}d)")
    ax.axvline(boundary_f - 0.5, color=COLOR_BLACK, linewidth=0.8,
               linestyle="--", alpha=0.5)
    ax.axvline(boundary_fu - 0.5, color=COLOR_BLACK, linewidth=0.8,
               linestyle="--", alpha=0.5)
    ax.set_title("Window Embedding (e_win) -- Component Magnitudes",
                 fontsize=9, pad=6)
    ax.set_xlabel("Embedding Dimension", fontsize=8)
    ax.set_ylabel("|Activation|", fontsize=8)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.95)
    style_axes(ax, grid="major")

    fig.suptitle(
        f"Sample Diagnostic -- GUID={guid}, Epoch={epoch}, Class={cls_label}",
        fontsize=12, fontweight="normal", y=1.002, color=COLOR_PURPLE,
    )

    save_figure(fig, output_path)
    return output_path


# ===================================================================
# Category 2: Forecasting
# ===================================================================

def plot_mae_histograms(metrics_df: pd.DataFrame, output_dir) -> Path:
    """3x3 grid of MAE histograms: rows=heads, cols=horizons.

    Args:
        metrics_df: DataFrame with columns ``head``, ``horizon``, ``mae``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "mae_histograms.png")
    if metrics_df.empty:
        return output_path

    apply_publication_style()
    heads = [h for h in _HEAD_NAMES if h in metrics_df["head"].unique()]
    horizons = sorted(metrics_df["horizon"].unique())
    n_heads = len(heads)
    n_hor = len(horizons)

    fig, axes = plt.subplots(n_heads, n_hor, figsize=(4 * n_hor, 3.5 * n_heads),
                             constrained_layout=True)
    if n_heads == 1 and n_hor == 1:
        axes = np.array([[axes]])
    elif n_heads == 1:
        axes = axes[np.newaxis, :]
    elif n_hor == 1:
        axes = axes[:, np.newaxis]

    for ri, head in enumerate(heads):
        for ci, h in enumerate(horizons):
            ax = axes[ri, ci]
            subset = metrics_df[(metrics_df["head"] == head) &
                                (metrics_df["horizon"] == h)]
            if not subset.empty:
                ax.hist(subset["mae"].values, bins=40, alpha=0.8,
                        color=HEAD_COLORS.get(head, COLOR_GRAY),
                        edgecolor=COLOR_BLACK, linewidth=0.3)
            ax.set_title(f"{HEAD_LABELS.get(head, head)}, h={h}", fontsize=9)
            ax.set_xlabel("MAE", fontsize=8)
            if ci == 0:
                ax.set_ylabel("Count", fontsize=8)
            style_axes(ax)

    fig.suptitle("MAE Distributions by Head and Horizon", fontsize=11,
                 color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path


def plot_mae_boxplots_by_class(metrics_df: pd.DataFrame, output_dir) -> Path:
    """Grouped box plots: x=horizon, color=head, panels=class.

    Args:
        metrics_df: DataFrame with columns ``head``, ``horizon``, ``mae``,
            ``class_label``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "mae_boxplots_by_class.png")
    if metrics_df.empty:
        return output_path

    apply_publication_style()
    classes = sorted(metrics_df["class_label"].unique())
    n_cls = max(len(classes), 1)
    horizons = sorted(metrics_df["horizon"].unique())
    heads = [h for h in _HEAD_NAMES if h in metrics_df["head"].unique()]

    fig, axes = plt.subplots(1, n_cls, figsize=(5 * n_cls, 5),
                             constrained_layout=True, squeeze=False)

    for ci, cls in enumerate(classes):
        ax = axes[0, ci]
        cls_df = metrics_df[metrics_df["class_label"] == cls]
        positions = []
        bp_data = []
        bp_colors = []
        tick_positions = []
        tick_labels = []
        group_width = len(heads) + 1
        for hi, h in enumerate(horizons):
            tick_positions.append(hi * group_width + len(heads) / 2)
            tick_labels.append(f"h={h}")
            for ji, head in enumerate(heads):
                pos = hi * group_width + ji
                positions.append(pos)
                vals = cls_df[(cls_df["horizon"] == h) &
                              (cls_df["head"] == head)]["mae"].values
                bp_data.append(vals if len(vals) > 0 else [0])
                bp_colors.append(HEAD_COLORS.get(head, COLOR_GRAY))

        bp = ax.boxplot(bp_data, positions=positions, widths=0.7,
                        patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], bp_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor(COLOR_BLACK)
            patch.set_linewidth(0.5)
        for element in ("whiskers", "caps", "medians"):
            plt.setp(bp[element], color=COLOR_BLACK, linewidth=0.5)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_title(cls, fontsize=9)
        ax.set_ylabel("MAE", fontsize=8)
        style_axes(ax)

    # Legend
    legend_handles = [plt.Line2D([0], [0], color=HEAD_COLORS[h], lw=6,
                                 alpha=0.7, label=HEAD_LABELS[h])
                      for h in heads]
    axes[0, -1].legend(handles=legend_handles, fontsize=7, loc="upper right")
    fig.suptitle("MAE by Horizon and Head per Class", fontsize=11,
                 color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path


def plot_head_comparison_scatter(metrics_df: pd.DataFrame, output_dir,
                                head_x: str = "self",
                                head_y: str = "fused") -> Path:
    """Scatter plot: x=head_x MAE, y=head_y MAE, colored by class.

    Points below the diagonal indicate head_y outperforms head_x.

    Args:
        metrics_df: DataFrame with columns ``head``, ``horizon``, ``mae``,
            ``class_label``, ``guid``, ``epoch``.
        output_dir: Directory to save the figure.
        head_x: Head name for x-axis.
        head_y: Head name for y-axis.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / f"head_scatter_{head_x}_vs_{head_y}.png")
    if metrics_df.empty:
        return output_path

    apply_publication_style()
    horizons = sorted(metrics_df["horizon"].unique())
    classes = sorted(metrics_df["class_label"].unique())
    class_colors = get_class_colors(classes)

    fig, axes = plt.subplots(1, len(horizons),
                             figsize=(5 * len(horizons), 5),
                             constrained_layout=True, squeeze=False)

    for hi, h in enumerate(horizons):
        ax = axes[0, hi]
        h_df = metrics_df[metrics_df["horizon"] == h]
        # Pivot to get one column per head
        df_x = h_df[h_df["head"] == head_x][["guid", "epoch", "mae",
                                               "class_label"]].copy()
        df_x = df_x.rename(columns={"mae": "mae_x"})
        df_y = h_df[h_df["head"] == head_y][["guid", "epoch", "mae"]].copy()
        df_y = df_y.rename(columns={"mae": "mae_y"})

        if df_x.empty or df_y.empty:
            ax.set_title(f"h={h} (no data)", fontsize=9)
            style_axes(ax)
            continue

        merged = pd.merge(df_x, df_y, on=["guid", "epoch"], how="inner")

        for cls in classes:
            m = merged[merged["class_label"] == cls]
            if m.empty:
                continue
            ax.scatter(m["mae_x"], m["mae_y"], s=8, alpha=0.5,
                       color=class_colors[cls], label=cls, edgecolors="none")

        lims = [0, max(merged["mae_x"].max(), merged["mae_y"].max()) * 1.05]
        ax.plot(lims, lims, "--", color=COLOR_GRAY, linewidth=0.8, alpha=0.6)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"{HEAD_LABELS.get(head_x, head_x)} MAE", fontsize=8)
        ax.set_ylabel(f"{HEAD_LABELS.get(head_y, head_y)} MAE", fontsize=8)
        ax.set_title(f"h={h}", fontsize=9)
        ax.legend(fontsize=6, framealpha=0.9)
        style_axes(ax)

    fig.suptitle(
        f"{HEAD_LABELS.get(head_x, head_x)} vs "
        f"{HEAD_LABELS.get(head_y, head_y)} MAE",
        fontsize=11, color=COLOR_PURPLE,
    )
    save_figure(fig, output_path)
    return output_path


def plot_improvement_distribution(metrics_df: pd.DataFrame,
                                  output_dir) -> Path:
    """Histograms of relative improvement (MAE_self - MAE_fused)/MAE_self.

    One panel per class.

    Args:
        metrics_df: DataFrame with columns ``head``, ``horizon``, ``mae``,
            ``class_label``, ``guid``, ``epoch``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / "improvement_distribution.png")
    if metrics_df.empty:
        return output_path

    apply_publication_style()
    classes = sorted(metrics_df["class_label"].unique())
    class_colors = get_class_colors(classes)
    horizons = sorted(metrics_df["horizon"].unique())

    fig, axes = plt.subplots(1, len(classes),
                             figsize=(5 * len(classes), 4),
                             constrained_layout=True, squeeze=False)

    for ci, cls in enumerate(classes):
        ax = axes[0, ci]
        cls_df = metrics_df[metrics_df["class_label"] == cls]
        for h in horizons:
            h_df = cls_df[cls_df["horizon"] == h]
            self_df = h_df[h_df["head"] == "self"][["guid", "epoch", "mae"]].rename(
                columns={"mae": "mae_self"})
            fused_df = h_df[h_df["head"] == "fused"][["guid", "epoch", "mae"]].rename(
                columns={"mae": "mae_fused"})
            merged = pd.merge(self_df, fused_df, on=["guid", "epoch"],
                              how="inner")
            if merged.empty:
                continue
            improvement = ((merged["mae_self"] - merged["mae_fused"])
                           / merged["mae_self"].clip(lower=1e-10))
            ax.hist(improvement.values, bins=40, alpha=0.6,
                    edgecolor=COLOR_BLACK, linewidth=0.3,
                    label=f"h={h}", density=True)
        ax.axvline(0, color=COLOR_GRAY, linestyle="--", linewidth=0.6)
        ax.set_title(cls, fontsize=9)
        ax.set_xlabel("Relative Improvement", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=7, framealpha=0.95)
        style_axes(ax)

    fig.suptitle("Fused vs Self Improvement Distribution", fontsize=11,
                 color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path


def plot_channel_error_heatmap(metrics_df: pd.DataFrame,
                               Y_hat_fus_channels: np.ndarray,
                               output_dir) -> Path:
    """Heatmap of per-channel MAE: x=channel(0..42), y=horizon.

    Args:
        metrics_df: DataFrame (unused beyond shape validation).
        Y_hat_fus_channels: 2-D array ``(n_horizons, n_channels)`` of mean
            per-channel MAE.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / "channel_error_heatmap.png")
    if Y_hat_fus_channels.size == 0:
        return output_path

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
    heatmap(ax, Y_hat_fus_channels, cmap="YlOrRd", origin="upper",
            title="Per-channel Mean MAE (Fused Head)", ylabel="Horizon",
            xlabel="Channel", label="MAE", fig=fig)
    save_figure(fig, output_path)
    return output_path


def plot_error_vs_anchor(metrics_df: pd.DataFrame, output_dir) -> Path:
    """Line plot of MAE vs anchor index, one line per head.

    Args:
        metrics_df: DataFrame with columns ``head``, ``mae``,
            ``anchor_idx`` (if available) or computed from ordering.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "error_vs_anchor.png")
    if metrics_df.empty:
        return output_path

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    heads = [h for h in _HEAD_NAMES if h in metrics_df["head"].unique()]
    horizons = sorted(metrics_df["horizon"].unique())

    # Use the largest horizon for this visualization
    h_max = horizons[-1] if horizons else 30
    h_df = metrics_df[metrics_df["horizon"] == h_max]

    if "anchor_idx" in h_df.columns:
        anchor_col = "anchor_idx"
    elif "anchor_timestep" in h_df.columns:
        anchor_col = "anchor_timestep"
    else:
        # Approximate: group by sample order
        ax.text(0.5, 0.5, "No anchor index column", ha="center",
                va="center", fontsize=9, transform=ax.transAxes)
        style_axes(ax)
        save_figure(fig, output_path)
        return output_path

    for head in heads:
        sub = h_df[h_df["head"] == head]
        if sub.empty:
            continue
        grouped = sub.groupby(anchor_col)["mae"].mean().sort_index()
        ax.plot(grouped.index, grouped.values, color=HEAD_COLORS[head],
                linewidth=1.0, marker="o", markersize=3,
                label=HEAD_LABELS[head])

    ax.set_title(f"MAE vs Anchor Index (h={h_max})", fontsize=9, pad=6)
    ax.set_xlabel("Anchor Index", fontsize=8)
    ax.set_ylabel("Mean MAE", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.95)
    style_axes(ax)
    save_figure(fig, output_path)
    return output_path


def plot_error_vs_time(metrics_df: pd.DataFrame, output_dir) -> Path:
    """Line plot of MAE vs hours-before-birth, one line per class.

    Args:
        metrics_df: DataFrame with columns ``mae``, ``epoch_hours``,
            ``class_label``.  Uses the fused head only.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "error_vs_time.png")
    if metrics_df.empty:
        return output_path

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    classes = sorted(metrics_df["class_label"].unique())
    class_colors = get_class_colors(classes)

    fused_df = metrics_df[metrics_df["head"] == "fused"] if "head" in metrics_df.columns else metrics_df

    if "epoch_hours" not in fused_df.columns:
        ax.text(0.5, 0.5, "No epoch_hours column", ha="center", va="center",
                fontsize=9, transform=ax.transAxes)
        style_axes(ax)
        save_figure(fig, output_path)
        return output_path

    for cls in classes:
        sub = fused_df[fused_df["class_label"] == cls].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("epoch_hours")
        # Bin into 20 time bins
        n_bins = min(20, len(sub))
        sub["time_bin"] = pd.cut(sub["epoch_hours"], bins=n_bins)
        grouped = sub.groupby("time_bin", observed=True)["mae"].mean()
        bin_centers = [interval.mid for interval in grouped.index]
        ax.plot(bin_centers, grouped.values, color=class_colors[cls],
                linewidth=1.0, marker="o", markersize=3, label=cls)

    ax.set_title("MAE vs Hours Before Birth (Fused Head)", fontsize=9, pad=6)
    ax.set_xlabel("Hours Before Birth", fontsize=8)
    ax.set_ylabel("Mean MAE", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.95)
    style_axes(ax)
    save_figure(fig, output_path)
    return output_path


def plot_loss_decomposition(loss_df: pd.DataFrame, output_dir) -> Path:
    """Stacked bars: L_fus, L_delta, L_self, L_te, L_kl per class.

    Args:
        loss_df: DataFrame with columns ``class_label``, ``L_fus``,
            ``L_delta``, ``L_self``, ``L_te``, ``L_kl``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "loss_decomposition.png")
    if loss_df.empty:
        return output_path

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    classes = sorted(loss_df["class_label"].unique())
    loss_components = ["L_fus", "L_delta", "L_self", "L_te", "L_kl"]
    component_colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_SKY, COLOR_GREEN,
                        COLOR_PURPLE]
    available = [c for c in loss_components if c in loss_df.columns]

    x = np.arange(len(classes))
    bottoms = np.zeros(len(classes))
    for comp, color in zip(available, component_colors):
        means = [loss_df[loss_df["class_label"] == cls][comp].mean()
                 for cls in classes]
        means = np.array(means)
        ax.bar(x, means, bottom=bottoms, color=color, alpha=0.85,
               edgecolor=COLOR_BLACK, linewidth=0.3, label=comp)
        bottoms += means

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_title("Loss Decomposition by Class", fontsize=9, pad=6)
    ax.set_xlabel("Class", fontsize=8)
    ax.set_ylabel("Loss", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.95)
    style_axes(ax)
    save_figure(fig, output_path)
    return output_path


def plot_head_radar(metrics_df: pd.DataFrame, output_dir) -> Path:
    """Radar chart comparing 3 heads across metrics, one panel per class.

    Metrics shown: MAE, MSE, VAF, SNR, Huber (from metrics_df columns).

    Args:
        metrics_df: DataFrame with columns ``head``, ``class_label``,
            ``mae``, ``mse``, ``vaf``, ``snr``, ``huber_loss``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "head_radar.png")
    if metrics_df.empty:
        return output_path

    apply_publication_style()
    classes = sorted(metrics_df["class_label"].unique())
    heads = [h for h in _HEAD_NAMES if h in metrics_df["head"].unique()]
    metric_cols = [c for c in ["mae", "mse", "vaf", "snr", "huber_loss"]
                   if c in metrics_df.columns]
    n_metrics = len(metric_cols)
    if n_metrics < 3:
        return output_path

    fig, axes = plt.subplots(1, len(classes),
                             figsize=(5 * len(classes), 5),
                             subplot_kw=dict(polar=True),
                             constrained_layout=True, squeeze=False)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    for ci, cls in enumerate(classes):
        ax = axes[0, ci]
        cls_df = metrics_df[metrics_df["class_label"] == cls]
        for head in heads:
            h_df = cls_df[cls_df["head"] == head]
            if h_df.empty:
                continue
            vals = [h_df[m].mean() for m in metric_cols]
            # Normalize to [0, 1] for radar
            val_max = max(np.abs(vals)) if max(np.abs(vals)) > 0 else 1.0
            vals_norm = [v / val_max for v in vals]
            vals_norm += vals_norm[:1]
            ax.plot(angles, vals_norm, color=HEAD_COLORS[head], linewidth=1.0,
                    label=HEAD_LABELS[head])
            ax.fill(angles, vals_norm, color=HEAD_COLORS[head], alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_cols, fontsize=7)
        ax.set_title(cls, fontsize=9, pad=12)
        ax.legend(fontsize=6, loc="upper right")

    fig.suptitle("Head Comparison Radar", fontsize=11, color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path


# ===================================================================
# Category 3: TE coupling
# ===================================================================

def plot_kl_distributions(te_seg_df: pd.DataFrame, output_dir) -> Path:
    """Histograms of kl_mean per class (separate panels).

    Args:
        te_seg_df: Segment-level TE DataFrame with ``kl_mean`` and
            ``class_label`` columns.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "kl_distributions.png")
    if te_seg_df.empty:
        return output_path

    apply_publication_style()
    classes = sorted(te_seg_df["class_label"].unique())
    class_colors = get_class_colors(classes)
    n_cls = max(len(classes), 1)

    fig, axes = plt.subplots(1, n_cls, figsize=(5 * n_cls, 4),
                             constrained_layout=True, squeeze=False)

    for ci, cls in enumerate(classes):
        ax = axes[0, ci]
        vals = te_seg_df[te_seg_df["class_label"] == cls]["kl_mean"].dropna().values
        if len(vals) > 0:
            ax.hist(vals, bins=40, alpha=0.8, color=class_colors[cls],
                    edgecolor=COLOR_BLACK, linewidth=0.3)
            ax.axvline(vals.mean(), color=COLOR_BLACK, linestyle="--",
                       linewidth=0.8, label=f"mean={vals.mean():.3f}")
        ax.set_title(cls, fontsize=9)
        ax.set_xlabel("KL Mean", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.legend(fontsize=7, framealpha=0.95)
        style_axes(ax)

    fig.suptitle("KL Divergence Distributions", fontsize=11,
                 color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path


def plot_kl_per_dimension(te_seg_df: pd.DataFrame, output_dir) -> Path:
    """Bar chart of mean KL per latent dimension, grouped by class.

    Args:
        te_seg_df: Segment-level TE DataFrame with columns
            ``kl_dim_mean_0`` ... ``kl_dim_mean_15`` and ``class_label``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "kl_per_dimension.png")
    if te_seg_df.empty:
        return output_path

    apply_publication_style()
    classes = sorted(te_seg_df["class_label"].unique())
    class_colors = get_class_colors(classes)

    # Find KL dimension columns
    kl_cols = [c for c in te_seg_df.columns if c.startswith("kl_dim_mean_")]
    d_z = len(kl_cols)
    if d_z == 0:
        return output_path

    fig, ax = plt.subplots(figsize=(max(10, d_z * 0.8), 5),
                           constrained_layout=True)
    x = np.arange(d_z)
    bar_width = 0.8 / max(len(classes), 1)

    for ci, cls in enumerate(classes):
        cls_df = te_seg_df[te_seg_df["class_label"] == cls]
        means = [cls_df[c].mean() for c in kl_cols]
        ax.bar(x + ci * bar_width, means, bar_width,
               color=class_colors[cls], alpha=0.8,
               edgecolor=COLOR_BLACK, linewidth=0.3, label=cls)

    ax.set_xticks(x + bar_width * (len(classes) - 1) / 2)
    ax.set_xticklabels([f"z{i}" for i in range(d_z)], fontsize=7)
    ax.set_title("Mean KL per Latent Dimension", fontsize=9, pad=6)
    ax.set_xlabel("Latent Dimension", fontsize=8)
    ax.set_ylabel("Mean KL", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.95)
    style_axes(ax)
    save_figure(fig, output_path)
    return output_path


def plot_kl_vs_anchor(te_anchor_df: pd.DataFrame, output_dir) -> Path:
    """Line plot of mean KL at each anchor position, per class.

    Args:
        te_anchor_df: Anchor-level TE DataFrame with ``anchor_idx``,
            ``kl_total``, ``class_label``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "kl_vs_anchor.png")
    if te_anchor_df.empty:
        return output_path

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    classes = sorted(te_anchor_df["class_label"].unique())
    class_colors = get_class_colors(classes)

    for cls in classes:
        sub = te_anchor_df[te_anchor_df["class_label"] == cls]
        grouped = sub.groupby("anchor_idx")["kl_total"].mean().sort_index()
        ax.plot(grouped.index, grouped.values, color=class_colors[cls],
                linewidth=1.0, marker="o", markersize=3, label=cls)

    ax.set_title("Mean KL vs Anchor Position", fontsize=9, pad=6)
    ax.set_xlabel("Anchor Index", fontsize=8)
    ax.set_ylabel("Mean KL", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.95)
    style_axes(ax)
    save_figure(fig, output_path)
    return output_path


def plot_posterior_vs_prior(te_anchor_df: pd.DataFrame, output_dir) -> Path:
    """4x4 grid of scatter plots: mu_post vs mu_prior per latent dim.

    Each subplot is one latent dimension, points colored by class.

    Args:
        te_anchor_df: Anchor-level TE DataFrame with ``mu_post_d``,
            ``mu_prior_d`` columns (d=0..15) and ``class_label``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "posterior_vs_prior.png")
    if te_anchor_df.empty:
        return output_path

    apply_publication_style()
    # Detect number of latent dims
    post_cols = [c for c in te_anchor_df.columns if c.startswith("mu_post_")]
    d_z = len(post_cols)
    if d_z == 0:
        return output_path

    n_side = int(np.ceil(np.sqrt(d_z)))
    classes = sorted(te_anchor_df["class_label"].unique())
    class_colors = get_class_colors(classes)

    fig, axes = plt.subplots(n_side, n_side,
                             figsize=(3 * n_side, 3 * n_side),
                             constrained_layout=True)
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for d in range(d_z):
        ax = axes_flat[d]
        post_col = f"mu_post_{d}"
        prior_col = f"mu_prior_{d}"
        if post_col not in te_anchor_df.columns or prior_col not in te_anchor_df.columns:
            ax.set_visible(False)
            continue
        for cls in classes:
            sub = te_anchor_df[te_anchor_df["class_label"] == cls]
            ax.scatter(sub[prior_col], sub[post_col], s=4, alpha=0.4,
                       color=class_colors[cls], edgecolors="none", label=cls)
        lims_all = [
            min(te_anchor_df[prior_col].min(), te_anchor_df[post_col].min()),
            max(te_anchor_df[prior_col].max(), te_anchor_df[post_col].max()),
        ]
        ax.plot(lims_all, lims_all, "--", color=COLOR_GRAY, linewidth=0.6)
        ax.set_title(f"z_{d}", fontsize=8)
        ax.set_xlabel("Prior", fontsize=6)
        ax.set_ylabel("Posterior", fontsize=6)
        ax.tick_params(labelsize=6)
        style_axes(ax)

    for d in range(d_z, len(axes_flat)):
        axes_flat[d].set_visible(False)

    # Single legend at top
    if d_z > 0:
        handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                              color=class_colors[c], markersize=4, label=c)
                   for c in classes]
        fig.legend(handles=handles, fontsize=7, loc="upper right",
                   framealpha=0.95)

    fig.suptitle("Posterior vs Prior Means per Latent Dimension", fontsize=11,
                 color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path


def plot_te_residual_analysis(te_seg_df: pd.DataFrame, output_dir) -> Path:
    """Distribution of residual_norm_mean per horizon per class.

    Args:
        te_seg_df: Segment-level TE DataFrame with columns
            ``residual_norm_mean_h{h}`` and ``class_label``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "te_residual_analysis.png")
    if te_seg_df.empty:
        return output_path

    apply_publication_style()
    classes = sorted(te_seg_df["class_label"].unique())
    class_colors = get_class_colors(classes)

    # Find residual columns
    res_cols = [c for c in te_seg_df.columns
                if c.startswith("residual_norm_mean_h")]
    if not res_cols:
        return output_path

    fig, axes = plt.subplots(1, len(res_cols),
                             figsize=(5 * len(res_cols), 4),
                             constrained_layout=True, squeeze=False)

    for ri, col in enumerate(res_cols):
        ax = axes[0, ri]
        horizon_label = col.replace("residual_norm_mean_", "")
        data_dict = {cls: te_seg_df[te_seg_df["class_label"] == cls][col].dropna().values
                     for cls in classes}
        _overlaid_histograms(ax, data_dict, class_colors,
                             xlabel="Residual Norm")
        ax.set_title(f"Residual Norm ({horizon_label})", fontsize=9)
        style_axes(ax)

    fig.suptitle("TE Residual Magnitude Distributions", fontsize=11,
                 color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path


def plot_te_latent_projection(te_data: np.ndarray, labels: np.ndarray,
                              output_dir, method: str = "pca") -> Path:
    """2-D projection of mu_post across samples, colored by class.

    Args:
        te_data: ``(N, d_z)`` posterior means.
        labels: ``(N,)`` string class labels.
        output_dir: Directory to save the figure.
        method: Projection method: ``"pca"``, ``"umap"``, or ``"tsne"``.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / f"te_latent_projection_{method}.png")
    if te_data.size == 0:
        return output_path

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    emb, method_used = _try_projection(te_data, method, n_components=2)
    classes = sorted(set(labels))
    class_colors = get_class_colors(classes)

    for cls in classes:
        mask = labels == cls
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.5,
                   color=class_colors[cls], edgecolors="none", label=cls)

    ax.set_title(f"TE Latent Projection ({method_used.upper()})", fontsize=9,
                 pad=6)
    ax.set_xlabel("Component 1", fontsize=8)
    ax.set_ylabel("Component 2", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.95)
    style_axes(ax)
    save_figure(fig, output_path)
    return output_path


def plot_posterior_variance(te_seg_df: pd.DataFrame, output_dir) -> Path:
    """Distribution of exp(logvar_post_mean) per class.

    Args:
        te_seg_df: Segment-level TE DataFrame with ``logvar_post_mean_d``
            columns and ``class_label``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "posterior_variance.png")
    if te_seg_df.empty:
        return output_path

    apply_publication_style()
    classes = sorted(te_seg_df["class_label"].unique())
    class_colors = get_class_colors(classes)

    lv_cols = [c for c in te_seg_df.columns if c.startswith("logvar_post_mean_")]
    if not lv_cols:
        return output_path

    # Compute mean posterior variance across all dims per sample
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    data_dict = {}
    for cls in classes:
        sub = te_seg_df[te_seg_df["class_label"] == cls]
        mean_var = np.exp(sub[lv_cols].values).mean(axis=1)
        data_dict[cls] = mean_var

    _overlaid_histograms(ax, data_dict, class_colors,
                         xlabel="Mean Posterior Variance")
    ax.set_title("Posterior Variance Distribution", fontsize=9, pad=6)
    style_axes(ax)
    save_figure(fig, output_path)
    return output_path


def plot_te_correlation_matrix(te_anchor_df: pd.DataFrame,
                               output_dir) -> Path:
    """Heatmap of correlations between 16 latent dims, per class.

    Args:
        te_anchor_df: Anchor-level TE DataFrame with ``mu_post_d`` columns
            and ``class_label``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / "te_correlation_matrix.png")
    if te_anchor_df.empty:
        return output_path

    apply_publication_style()
    post_cols = sorted([c for c in te_anchor_df.columns
                        if c.startswith("mu_post_")],
                       key=lambda x: int(x.split("_")[-1]))
    d_z = len(post_cols)
    if d_z == 0:
        return output_path

    classes = sorted(te_anchor_df["class_label"].unique())
    n_cls = max(len(classes), 1)

    fig, axes = plt.subplots(1, n_cls, figsize=(5 * n_cls, 4.5),
                             constrained_layout=True, squeeze=False)

    for ci, cls in enumerate(classes):
        ax = axes[0, ci]
        sub = te_anchor_df[te_anchor_df["class_label"] == cls][post_cols]
        if sub.shape[0] < 3:
            ax.text(0.5, 0.5, "Too few samples", ha="center", va="center",
                    fontsize=9, transform=ax.transAxes)
            ax.set_title(cls, fontsize=9)
            continue
        corr = sub.corr().values
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(np.arange(d_z))
        ax.set_yticks(np.arange(d_z))
        ax.set_xticklabels([f"z{i}" for i in range(d_z)], fontsize=6,
                           rotation=45)
        ax.set_yticklabels([f"z{i}" for i in range(d_z)], fontsize=6)
        ax.set_title(cls, fontsize=9)
        ax.grid(False)
        add_colorbar(fig, im, ax, label="Correlation")

    fig.suptitle("TE Latent Correlation Matrices", fontsize=11,
                 color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path


# ===================================================================
# Category 4: Representation
# ===================================================================

def plot_embedding_projection(embeddings: np.ndarray, labels: np.ndarray,
                              output_dir, method: str = "pca",
                              component: str = "e_win") -> Path:
    """2-D/3-D projection of embeddings colored by class.

    Args:
        embeddings: ``(N, D)`` embedding array.
        labels: ``(N,)`` string class labels.
        output_dir: Directory to save the figure.
        method: ``"pca"``, ``"umap"``, or ``"tsne"``.
        component: Embedding component name (for title), e.g.
            ``"e_win"``, ``"e_F"``, ``"e_FU"``, ``"e_TE"``.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / f"embedding_{component}_{method}.png")
    if embeddings.size == 0:
        return output_path

    apply_publication_style()
    emb, method_used = _try_projection(embeddings, method, n_components=2)
    classes = sorted(set(labels))
    class_colors = get_class_colors(classes)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for cls in classes:
        mask = np.array(labels) == cls
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.5,
                   color=class_colors[cls], edgecolors="none", label=cls)

    ax.set_title(f"{component} Projection ({method_used.upper()})",
                 fontsize=9, pad=6)
    ax.set_xlabel("Component 1", fontsize=8)
    ax.set_ylabel("Component 2", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.95)
    style_axes(ax)
    save_figure(fig, output_path)
    return output_path


def plot_embedding_norms(embeddings_dict: Dict[str, np.ndarray],
                         labels: np.ndarray, output_dir) -> Path:
    """Violin plots of |e_F|, |e_FU|, |e_TE| norms per class.

    Args:
        embeddings_dict: ``{"e_F": (N, D1), "e_FU": (N, D2),
            "e_TE": (N, D3)}`` arrays.
        labels: ``(N,)`` string class labels.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "embedding_norms.png")
    if not embeddings_dict:
        return output_path

    apply_publication_style()
    classes = sorted(set(labels))
    class_colors = get_class_colors(classes)
    components = [k for k in ("e_F", "e_FU", "e_TE")
                  if k in embeddings_dict and embeddings_dict[k].size > 0]
    if not components:
        return output_path

    fig, axes = plt.subplots(1, len(components),
                             figsize=(4 * len(components), 5),
                             constrained_layout=True, squeeze=False)

    for ci_comp, comp in enumerate(components):
        ax = axes[0, ci_comp]
        data_lists = []
        positions = []
        colors_list = []
        for ci_cls, cls in enumerate(classes):
            mask = np.array(labels) == cls
            norms = np.linalg.norm(embeddings_dict[comp][mask], axis=1)
            if len(norms) == 0:
                continue
            data_lists.append(norms)
            positions.append(ci_cls)
            colors_list.append(class_colors[cls])

        if data_lists:
            parts = ax.violinplot(data_lists, positions=positions,
                                  showmeans=True, showmedians=True)
            for pc, color in zip(parts["bodies"], colors_list):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)

        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=7)
        ax.set_title(f"|{comp}|", fontsize=9)
        ax.set_ylabel("L2 Norm", fontsize=8)
        style_axes(ax)

    fig.suptitle("Embedding Component Norms", fontsize=11,
                 color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path


def plot_fusion_distribution(gate_fusion_df: pd.DataFrame,
                             output_dir) -> Path:
    """Histogram of mean_fusion_dist per class.

    Args:
        gate_fusion_df: DataFrame with ``mean_fusion_dist`` and
            ``class_label`` columns.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / "fusion_distribution.png")
    if gate_fusion_df.empty:
        return output_path

    apply_publication_style()
    classes = sorted(gate_fusion_df["class_label"].unique())
    class_colors = get_class_colors(classes)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    data_dict = {
        cls: gate_fusion_df[gate_fusion_df["class_label"] == cls][
            "mean_fusion_dist"].dropna().values
        for cls in classes
    }
    _overlaid_histograms(ax, data_dict, class_colors,
                         xlabel="Mean Fusion Distance")
    ax.set_title("Fusion Distance Distribution", fontsize=9, pad=6)
    style_axes(ax)
    save_figure(fig, output_path)
    return output_path


def plot_gate_distribution(gate_fusion_df: pd.DataFrame,
                           output_dir) -> Path:
    """Histogram of mean_gate per class.

    Args:
        gate_fusion_df: DataFrame with ``mean_gate`` and ``class_label``
            columns.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "gate_distribution.png")
    if gate_fusion_df.empty:
        return output_path

    apply_publication_style()
    classes = sorted(gate_fusion_df["class_label"].unique())
    class_colors = get_class_colors(classes)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    data_dict = {
        cls: gate_fusion_df[gate_fusion_df["class_label"] == cls][
            "mean_gate"].dropna().values
        for cls in classes
    }
    _overlaid_histograms(ax, data_dict, class_colors, xlabel="Mean Gate")
    ax.set_title("Gate Activation Distribution", fontsize=9, pad=6)
    style_axes(ax)
    save_figure(fig, output_path)
    return output_path


def plot_gate_temporal_profile(gate_fusion_df: pd.DataFrame,
                               output_dir) -> Path:
    """Mean gate over time (x=step), lines per class, +/- SEM.

    Args:
        gate_fusion_df: DataFrame with ``gate_t000``, ``gate_t001``, ...
            columns and ``class_label``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / "gate_temporal_profile.png")
    if gate_fusion_df.empty:
        return output_path

    apply_publication_style()
    gate_cols = sorted([c for c in gate_fusion_df.columns
                        if c.startswith("gate_t")],
                       key=lambda x: int(x.replace("gate_t", "")))
    if not gate_cols:
        return output_path

    classes = sorted(gate_fusion_df["class_label"].unique())
    class_colors = get_class_colors(classes)
    t_steps = np.arange(len(gate_cols))

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    for cls in classes:
        sub = gate_fusion_df[gate_fusion_df["class_label"] == cls][gate_cols]
        if sub.empty:
            continue
        vals = sub.values  # (N, T)
        mean = vals.mean(axis=0)
        sem = vals.std(axis=0) / max(np.sqrt(vals.shape[0]), 1)
        ax.plot(t_steps, mean, color=class_colors[cls], linewidth=1.0,
                label=cls)
        ax.fill_between(t_steps, mean - sem, mean + sem,
                        color=class_colors[cls], alpha=0.2)

    ax.set_title("Gate Temporal Profile (+/- SEM)", fontsize=9, pad=6)
    ax.set_xlabel("Time Step", fontsize=8)
    ax.set_ylabel("Mean Gate", fontsize=8)
    ax.set_xlim(0, len(gate_cols) - 1)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, framealpha=0.95)
    style_axes(ax, grid="both")
    save_figure(fig, output_path)
    return output_path


def plot_linear_separability(scores: Dict[str, float], output_dir) -> Path:
    """Bar chart of LDA/LogReg accuracy on e_win, e_F, e_FU, e_TE.

    Args:
        scores: ``{"e_win": acc, "e_F": acc, "e_FU": acc, "e_TE": acc}``
            classification accuracy values in [0, 1].
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / "linear_separability.png")
    if not scores:
        return output_path

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    components = list(scores.keys())
    vals = [scores[k] for k in components]
    comp_colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_PURPLE]

    bars = ax.bar(components, vals,
                  color=comp_colors[:len(components)], alpha=0.85,
                  edgecolor=COLOR_BLACK, linewidth=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_title("Linear Separability (Classification Accuracy)", fontsize=9,
                 pad=6)
    ax.set_ylabel("Accuracy", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0 / max(len(components), 1), color=COLOR_GRAY,
               linestyle="--", linewidth=0.6, alpha=0.5, label="Chance")
    ax.legend(fontsize=7, framealpha=0.95)
    style_axes(ax)
    save_figure(fig, output_path)
    return output_path


def plot_clustering_quality(scores: Dict[str, float], output_dir) -> Path:
    """Bar chart of Silhouette, Davies-Bouldin, Calinski-Harabasz scores.

    Args:
        scores: ``{"silhouette": val, "davies_bouldin": val,
            "calinski_harabasz": val}`` clustering quality scores.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "clustering_quality.png")
    if not scores:
        return output_path

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    metric_names = list(scores.keys())
    vals = [scores[k] for k in metric_names]
    colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_PURPLE,
              COLOR_VERMILLION]

    bars = ax.bar(range(len(metric_names)), vals,
                  color=colors[:len(metric_names)], alpha=0.85,
                  edgecolor=COLOR_BLACK, linewidth=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=7, rotation=15)
    ax.set_title("Clustering Quality Metrics", fontsize=9, pad=6)
    ax.set_ylabel("Score", fontsize=8)
    style_axes(ax)
    save_figure(fig, output_path)
    return output_path


def plot_variance_spectrum(embeddings_dict: Dict[str, np.ndarray],
                           output_dir) -> Path:
    """Cumulative PCA variance for e_win, e_F, e_FU, e_TE.

    Args:
        embeddings_dict: ``{"e_win": (N, D), "e_F": ..., "e_FU": ...,
            "e_TE": ...}`` arrays.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "variance_spectrum.png")
    if not embeddings_dict:
        return output_path

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    comp_colors = {"e_win": COLOR_BLACK, "e_F": COLOR_BLUE,
                   "e_FU": COLOR_ORANGE, "e_TE": COLOR_GREEN}

    for comp_name, data in embeddings_dict.items():
        if data.size == 0 or data.shape[0] < 2:
            continue
        centered = data - data.mean(axis=0, keepdims=True)
        try:
            _, S, _ = np.linalg.svd(centered, full_matrices=False)
            var_explained = (S ** 2) / max((S ** 2).sum(), 1e-12)
            cum_var = np.cumsum(var_explained)
            n_show = min(50, len(cum_var))
            ax.plot(np.arange(1, n_show + 1), cum_var[:n_show],
                    color=comp_colors.get(comp_name, COLOR_GRAY),
                    linewidth=1.0, marker=".", markersize=3,
                    label=comp_name)
        except np.linalg.LinAlgError:
            continue

    ax.axhline(0.95, color=COLOR_GRAY, linestyle="--", linewidth=0.6,
               alpha=0.5, label="95%")
    ax.set_title("Cumulative PCA Variance Spectrum", fontsize=9, pad=6)
    ax.set_xlabel("Number of Components", fontsize=8)
    ax.set_ylabel("Cumulative Variance Explained", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, framealpha=0.95)
    style_axes(ax, grid="both")
    save_figure(fig, output_path)
    return output_path


# ===================================================================
# Category 5: Trajectory
# ===================================================================

def plot_guid_trajectory(guid_data: Dict[str, Any], output_dir) -> Path:
    """PCA trajectory for one GUID colored by time.

    Args:
        guid_data: Dictionary with keys ``e_win`` (``(N_epochs, D)``),
            ``epochs`` (``(N_epochs,)``), ``guid`` (str),
            ``class_label`` (str).
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    guid = guid_data.get("guid", "unknown")
    output_path = _ensure_path(
        Path(output_dir) / f"trajectory_{guid}.png")
    e_win = guid_data.get("e_win", np.empty((0, 0)))
    if e_win.size == 0 or e_win.shape[0] < 2:
        return output_path

    apply_publication_style()
    proj, ve = _pca_svd(e_win, 2)
    if proj is None:
        return output_path

    epochs = guid_data.get("epochs", np.arange(e_win.shape[0]))

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=epochs, cmap="viridis",
                    s=15, edgecolors=COLOR_BLACK, linewidths=0.3, zorder=3)
    ax.plot(proj[:, 0], proj[:, 1], color=COLOR_GRAY, linewidth=0.5,
            alpha=0.5, zorder=2)
    add_colorbar(fig, sc, ax, label="Epoch")

    ax.set_title(f"GUID {guid} -- PCA Trajectory", fontsize=9, pad=6)
    ax.set_xlabel(f"PC1 ({ve[0]*100:.1f}%)", fontsize=8)
    ax.set_ylabel(f"PC2 ({ve[1]*100:.1f}%)", fontsize=8)
    style_axes(ax)
    save_figure(fig, output_path)
    return output_path


def plot_class_mean_trajectory(trajectory_data: Dict[str, Any], output_dir,
                               metric: str = "kl_mean") -> Path:
    """Averaged metric vs time-to-delivery with confidence bands, per class.

    Args:
        trajectory_data: ``{"class_name": {"time": (N,), "metric": (N,),
            "sem": (N,)}, ...}`` per-class averaged trajectory data.
        output_dir: Directory to save the figure.
        metric: Name of the metric (for title/axis label).

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / f"class_mean_trajectory_{metric}.png")
    if not trajectory_data:
        return output_path

    apply_publication_style()
    classes = sorted(trajectory_data.keys())
    class_colors = get_class_colors(classes)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    for cls in classes:
        td = trajectory_data[cls]
        t = td.get("time", np.array([]))
        m = td.get("metric", np.array([]))
        sem = td.get("sem", np.zeros_like(m))
        if len(t) == 0:
            continue
        ax.plot(t, m, color=class_colors[cls], linewidth=1.0, label=cls)
        ax.fill_between(t, m - sem, m + sem, color=class_colors[cls],
                        alpha=0.2)

    ax.set_title(f"Mean {metric} vs Time-to-Delivery", fontsize=9, pad=6)
    ax.set_xlabel("Time to Delivery (hours)", fontsize=8)
    ax.set_ylabel(metric, fontsize=8)
    ax.legend(fontsize=7, framealpha=0.95)
    style_axes(ax, grid="both")
    save_figure(fig, output_path)
    return output_path


def plot_embedding_drift(drift_data: Dict[str, np.ndarray],
                         output_dir) -> Path:
    """Box plot of e_win change rate between consecutive epochs, by class.

    Args:
        drift_data: ``{"class_name": 1-D array of drift values, ...}``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "embedding_drift.png")
    if not drift_data:
        return output_path

    apply_publication_style()
    classes = sorted(drift_data.keys())
    class_colors = get_class_colors(classes)

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(classes)), 5),
                           constrained_layout=True)

    bp_data = [drift_data[cls] for cls in classes
               if len(drift_data[cls]) > 0]
    bp_labels = [cls for cls in classes if len(drift_data[cls]) > 0]
    bp_colors = [class_colors[cls] for cls in bp_labels]

    if bp_data:
        bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True,
                        showfliers=False)
        for patch, color in zip(bp["boxes"], bp_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor(COLOR_BLACK)
            patch.set_linewidth(0.5)
        for element in ("whiskers", "caps", "medians"):
            plt.setp(bp[element], color=COLOR_BLACK, linewidth=0.5)

    ax.set_title("Embedding Drift Between Consecutive Epochs", fontsize=9,
                 pad=6)
    ax.set_ylabel("||e_win(t+1) - e_win(t)||", fontsize=8)
    style_axes(ax)
    save_figure(fig, output_path)
    return output_path


def plot_changepoints(trajectory_data: Dict[str, Any], output_dir) -> Path:
    """KL trajectory with changepoint markers.

    Args:
        trajectory_data: Dictionary with ``time`` (``(N,)``), ``kl``
            (``(N,)``), ``changepoints`` (list of indices or times),
            and ``class_label`` (str).
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "changepoints.png")
    if not trajectory_data:
        return output_path

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    t = trajectory_data.get("time", np.array([]))
    kl = trajectory_data.get("kl", np.array([]))
    cps = trajectory_data.get("changepoints", [])
    cls_label = trajectory_data.get("class_label", "unknown")

    if len(t) == 0:
        save_figure(fig, output_path)
        return output_path

    ax.plot(t, kl, color=COLOR_BLUE, linewidth=1.0, label="KL")
    for cp in cps:
        ax.axvline(cp, color=COLOR_VERMILLION, linestyle="--", linewidth=0.8,
                   alpha=0.7)
    if cps:
        ax.axvline(cps[0], color=COLOR_VERMILLION, linestyle="--",
                   linewidth=0.8, alpha=0.7, label="Changepoint")

    ax.set_title(f"KL Trajectory with Changepoints ({cls_label})",
                 fontsize=9, pad=6)
    ax.set_xlabel("Time", fontsize=8)
    ax.set_ylabel("KL Divergence", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.95)
    style_axes(ax, grid="both")
    save_figure(fig, output_path)
    return output_path


def plot_3d_trajectory(trajectory_data: Dict[str, Any], output_dir) -> Path:
    """3-D PCA trajectory colored by time.

    Args:
        trajectory_data: Dictionary with ``e_win`` (``(N, D)``),
            ``epochs`` (``(N,)``), ``guid`` (str).
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    guid = trajectory_data.get("guid", "unknown")
    output_path = _ensure_path(
        Path(output_dir) / f"3d_trajectory_{guid}.png")
    e_win = trajectory_data.get("e_win", np.empty((0, 0)))
    if e_win.size == 0 or e_win.shape[0] < 3:
        return output_path

    apply_publication_style()
    proj, ve = _pca_svd(e_win, 3)
    if proj is None or proj.shape[1] < 3:
        return output_path

    epochs = trajectory_data.get("epochs", np.arange(e_win.shape[0]))

    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=epochs,
                    cmap="viridis", s=15, edgecolors=COLOR_BLACK,
                    linewidths=0.2)
    ax.plot(proj[:, 0], proj[:, 1], proj[:, 2], color=COLOR_GRAY,
            linewidth=0.4, alpha=0.5)
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label="Epoch")

    ax.set_xlabel(f"PC1 ({ve[0]*100:.1f}%)", fontsize=7)
    ax.set_ylabel(f"PC2 ({ve[1]*100:.1f}%)", fontsize=7)
    ax.set_zlabel(f"PC3 ({ve[2]*100:.1f}%)", fontsize=7)
    ax.set_title(f"3D PCA Trajectory -- GUID {guid}", fontsize=9, pad=6)
    save_figure(fig, output_path)
    return output_path


def plot_trajectory_comparison(class_trajectories: Dict[str, Any],
                               output_dir) -> Path:
    """Overlay mean PCA trajectories from different classes.

    Args:
        class_trajectories: ``{"class_name": {"mean_proj": (N, 2),
            "time": (N,)}, ...}`` mean trajectories per class.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / "trajectory_comparison.png")
    if not class_trajectories:
        return output_path

    apply_publication_style()
    classes = sorted(class_trajectories.keys())
    class_colors = get_class_colors(classes)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

    for cls in classes:
        td = class_trajectories[cls]
        proj = td.get("mean_proj", np.empty((0, 2)))
        if proj.shape[0] < 2:
            continue
        ax.plot(proj[:, 0], proj[:, 1], color=class_colors[cls],
                linewidth=1.2, label=cls)
        ax.scatter(proj[0, 0], proj[0, 1], color=class_colors[cls],
                   marker="o", s=30, zorder=3, edgecolors=COLOR_BLACK,
                   linewidths=0.5)
        ax.scatter(proj[-1, 0], proj[-1, 1], color=class_colors[cls],
                   marker="X", s=40, zorder=3, edgecolors=COLOR_BLACK,
                   linewidths=0.5)

    ax.set_title("Mean Trajectory Comparison (PCA)", fontsize=9, pad=6)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.95)
    style_axes(ax)
    save_figure(fig, output_path)
    return output_path


# ===================================================================
# Category 6: Cross-class
# ===================================================================

def plot_metric_summary_table(results: Dict[str, Dict[str, Any]],
                              output_dir) -> Path:
    """Rendered table figure showing mean +/- std of all metrics per class.

    Args:
        results: ``{"class_name": {"metric_name": (mean, std), ...}, ...}``
            nested dictionary of metric statistics.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "metric_summary_table.png")
    if not results:
        return output_path

    apply_publication_style()
    classes = sorted(results.keys())
    if not classes:
        return output_path

    # Collect all metric names
    metric_names = sorted(set().union(*(results[c].keys() for c in classes)))
    n_metrics = len(metric_names)
    n_cls = len(classes)

    fig, ax = plt.subplots(
        figsize=(2 + 2.5 * n_cls, 0.4 * n_metrics + 1.5),
        constrained_layout=True,
    )
    ax.axis("off")

    col_labels = ["Metric"] + classes
    cell_text = []
    for metric in metric_names:
        row = [metric]
        for cls in classes:
            val = results[cls].get(metric, (float("nan"), float("nan")))
            if isinstance(val, tuple) and len(val) == 2:
                row.append(f"{val[0]:.4f} +/- {val[1]:.4f}")
            else:
                row.append(f"{val:.4f}" if not np.isnan(val) else "N/A")
        cell_text.append(row)

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.3)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(COLOR_PURPLE)
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(len(cell_text)):
        bg = COLOR_LIGHT_GRAY if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(bg)

    ax.set_title("Metric Summary", fontsize=11, pad=10, color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path


def plot_class_mae_comparison(metrics_df: pd.DataFrame, output_dir) -> Path:
    """Multi-panel bar chart with error bars, one panel per horizon.

    Args:
        metrics_df: DataFrame with ``class_label``, ``head``, ``horizon``,
            ``mae``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / "class_mae_comparison.png")
    if metrics_df.empty:
        return output_path

    apply_publication_style()
    horizons = sorted(metrics_df["horizon"].unique())
    classes = sorted(metrics_df["class_label"].unique())
    heads = [h for h in _HEAD_NAMES if h in metrics_df["head"].unique()]

    fig, axes = plt.subplots(1, len(horizons),
                             figsize=(5 * len(horizons), 5),
                             constrained_layout=True, squeeze=False)

    for hi, h in enumerate(horizons):
        ax = axes[0, hi]
        h_df = metrics_df[metrics_df["horizon"] == h]
        x = np.arange(len(classes))
        bar_w = 0.8 / max(len(heads), 1)

        for ji, head in enumerate(heads):
            means = []
            stds = []
            for cls in classes:
                sub = h_df[(h_df["head"] == head) &
                           (h_df["class_label"] == cls)]["mae"]
                means.append(sub.mean() if not sub.empty else 0)
                stds.append(sub.std() if not sub.empty else 0)
            ax.bar(x + ji * bar_w, means, bar_w, yerr=stds,
                   color=HEAD_COLORS[head], alpha=0.85,
                   edgecolor=COLOR_BLACK, linewidth=0.3,
                   error_kw=dict(elinewidth=0.5, capsize=2),
                   label=HEAD_LABELS[head])

        ax.set_xticks(x + bar_w * (len(heads) - 1) / 2)
        ax.set_xticklabels(classes, fontsize=7)
        ax.set_title(f"h={h}", fontsize=9)
        ax.set_ylabel("MAE", fontsize=8)
        ax.legend(fontsize=6, framealpha=0.95)
        style_axes(ax)

    fig.suptitle("MAE Comparison Across Classes", fontsize=11,
                 color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path


def plot_significance_heatmap(test_results: pd.DataFrame,
                              output_dir) -> Path:
    """Heatmap of p-values from pairwise statistical tests.

    Args:
        test_results: DataFrame or 2-D array of p-values with class names
            as both index and columns.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / "significance_heatmap.png")

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    if isinstance(test_results, pd.DataFrame):
        data = test_results.values
        labels = list(test_results.columns)
    elif isinstance(test_results, np.ndarray):
        data = test_results
        labels = [str(i) for i in range(data.shape[0])]
    else:
        save_figure(fig, output_path)
        return output_path

    if data.size == 0:
        save_figure(fig, output_path)
        return output_path

    im = ax.imshow(data, cmap="RdYlGn_r", vmin=0, vmax=0.1, aspect="equal")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=7)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            marker = "***" if val < 0.001 else ("**" if val < 0.01 else
                     ("*" if val < 0.05 else "ns"))
            ax.text(j, i, f"{val:.3f}\n{marker}", ha="center", va="center",
                    fontsize=6, color="white" if val < 0.05 else COLOR_BLACK)

    ax.set_title("Pairwise Significance (p-values)", fontsize=9, pad=6)
    ax.grid(False)
    add_colorbar(fig, im, ax, label="p-value")
    save_figure(fig, output_path)
    return output_path


def plot_effect_size_heatmap(effect_sizes: pd.DataFrame,
                             output_dir) -> Path:
    """Heatmap of Cohen's d effect sizes.

    Args:
        effect_sizes: DataFrame or 2-D array of Cohen's d values with
            class names as both index and columns.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / "effect_size_heatmap.png")

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    if isinstance(effect_sizes, pd.DataFrame):
        data = effect_sizes.values
        labels = list(effect_sizes.columns)
    elif isinstance(effect_sizes, np.ndarray):
        data = effect_sizes
        labels = [str(i) for i in range(data.shape[0])]
    else:
        save_figure(fig, output_path)
        return output_path

    if data.size == 0:
        save_figure(fig, output_path)
        return output_path

    vabs = max(np.nanmax(np.abs(data)), 0.1)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                   aspect="equal")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=7)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=6)

    ax.set_title("Effect Sizes (Cohen's d)", fontsize=9, pad=6)
    ax.grid(False)
    add_colorbar(fig, im, ax, label="Cohen's d")
    save_figure(fig, output_path)
    return output_path


def plot_roc_curves(classification_results: Dict[str, Any],
                    output_dir) -> Path:
    """ROC curves for embedding-based classification.

    Args:
        classification_results: ``{"class_name": {"fpr": array, "tpr":
            array, "auc": float}, ...}`` per-class ROC data.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "roc_curves.png")
    if not classification_results:
        return output_path

    apply_publication_style()
    classes = sorted(classification_results.keys())
    class_colors = get_class_colors(classes)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    for cls in classes:
        rd = classification_results[cls]
        fpr = rd.get("fpr", np.array([]))
        tpr = rd.get("tpr", np.array([]))
        auc_val = rd.get("auc", 0.0)
        if len(fpr) == 0:
            continue
        ax.plot(fpr, tpr, color=class_colors[cls], linewidth=1.0,
                label=f"{cls} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "--", color=COLOR_GRAY, linewidth=0.6)
    ax.set_title("ROC Curves", fontsize=9, pad=6)
    ax.set_xlabel("False Positive Rate", fontsize=8)
    ax.set_ylabel("True Positive Rate", fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, framealpha=0.95, loc="lower right")
    style_axes(ax)
    save_figure(fig, output_path)
    return output_path


def plot_confusion_matrices(classification_results: Dict[str, Any],
                            output_dir) -> Path:
    """Confusion matrix subplots for each embedding component.

    Args:
        classification_results: ``{"component_name": {"cm": 2-D array,
            "labels": list}, ...}`` confusion matrix data.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / "confusion_matrices.png")
    if not classification_results:
        return output_path

    apply_publication_style()
    components = sorted(classification_results.keys())
    n_comp = max(len(components), 1)

    fig, axes = plt.subplots(1, n_comp, figsize=(5 * n_comp, 4.5),
                             constrained_layout=True, squeeze=False)

    for ci, comp in enumerate(components):
        ax = axes[0, ci]
        rd = classification_results[comp]
        cm = rd.get("cm", np.array([]))
        labels = rd.get("labels", [])
        if cm.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=9, transform=ax.transAxes)
            ax.set_title(comp, fontsize=9)
            continue

        n = cm.shape[0]
        im = ax.imshow(cm, cmap="Blues", aspect="equal")
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        if labels:
            ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
            ax.set_yticklabels(labels, fontsize=6)

        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                        fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else COLOR_BLACK)

        ax.set_title(comp, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_ylabel("True", fontsize=7)
        ax.grid(False)
        add_colorbar(fig, im, ax, label="Count")

    fig.suptitle("Confusion Matrices", fontsize=11, color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path


def plot_cross_class_mae_histograms(metrics_df: pd.DataFrame,
                                    output_dir) -> Path:
    """3x3 grid (head x horizon) of overlaid MAE histograms by class.

    Args:
        metrics_df: DataFrame with ``head``, ``horizon``, ``mae``,
            ``class_label``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / "cross_class_mae_histograms.png")
    if metrics_df.empty:
        return output_path

    apply_publication_style()
    heads = [h for h in _HEAD_NAMES if h in metrics_df["head"].unique()]
    horizons = sorted(metrics_df["horizon"].unique())
    classes = sorted(metrics_df["class_label"].unique())
    class_colors = get_class_colors(classes)
    n_heads = len(heads)
    n_hor = len(horizons)

    fig, axes = plt.subplots(n_heads, n_hor,
                             figsize=(4 * n_hor, 3.5 * n_heads),
                             constrained_layout=True)
    if n_heads == 1 and n_hor == 1:
        axes = np.array([[axes]])
    elif n_heads == 1:
        axes = axes[np.newaxis, :]
    elif n_hor == 1:
        axes = axes[:, np.newaxis]

    for ri, head in enumerate(heads):
        for ci, h in enumerate(horizons):
            ax = axes[ri, ci]
            h_df = metrics_df[(metrics_df["head"] == head) &
                              (metrics_df["horizon"] == h)]
            data_dict = {
                cls: h_df[h_df["class_label"] == cls]["mae"].values
                for cls in classes
            }
            _overlaid_histograms(ax, data_dict, class_colors, xlabel="MAE")
            ax.set_title(f"{HEAD_LABELS.get(head, head)}, h={h}", fontsize=9)
            style_axes(ax)

    fig.suptitle("MAE Distributions by Class", fontsize=11,
                 color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path


def plot_cross_class_vaf_histograms(metrics_df: pd.DataFrame,
                                    output_dir) -> Path:
    """3 subplots (one per head) of overlaid VAF histograms by class.

    Args:
        metrics_df: DataFrame with ``head``, ``vaf``, ``class_label``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / "cross_class_vaf_histograms.png")
    if metrics_df.empty or "vaf" not in metrics_df.columns:
        return output_path

    apply_publication_style()
    heads = [h for h in _HEAD_NAMES if h in metrics_df["head"].unique()]
    classes = sorted(metrics_df["class_label"].unique())
    class_colors = get_class_colors(classes)

    fig, axes = plt.subplots(1, len(heads), figsize=(5 * len(heads), 4),
                             constrained_layout=True, squeeze=False)

    for hi, head in enumerate(heads):
        ax = axes[0, hi]
        h_df = metrics_df[metrics_df["head"] == head]
        data_dict = {
            cls: h_df[h_df["class_label"] == cls]["vaf"].dropna().values
            for cls in classes
        }
        _overlaid_histograms(ax, data_dict, class_colors, xlabel="VAF")
        ax.set_title(HEAD_LABELS.get(head, head), fontsize=9)
        style_axes(ax)

    fig.suptitle("VAF Distributions by Class", fontsize=11,
                 color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path


def plot_cross_class_snr_histograms(metrics_df: pd.DataFrame,
                                    output_dir) -> Path:
    """3 subplots (one per head) of overlaid SNR histograms by class.

    Args:
        metrics_df: DataFrame with ``head``, ``snr``, ``class_label``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / "cross_class_snr_histograms.png")
    if metrics_df.empty or "snr" not in metrics_df.columns:
        return output_path

    apply_publication_style()
    heads = [h for h in _HEAD_NAMES if h in metrics_df["head"].unique()]
    classes = sorted(metrics_df["class_label"].unique())
    class_colors = get_class_colors(classes)

    fig, axes = plt.subplots(1, len(heads), figsize=(5 * len(heads), 4),
                             constrained_layout=True, squeeze=False)

    for hi, head in enumerate(heads):
        ax = axes[0, hi]
        h_df = metrics_df[metrics_df["head"] == head]
        data_dict = {
            cls: h_df[h_df["class_label"] == cls]["snr"].dropna().values
            for cls in classes
        }
        _overlaid_histograms(ax, data_dict, class_colors, xlabel="SNR (dB)")
        ax.set_title(HEAD_LABELS.get(head, head), fontsize=9)
        style_axes(ax)

    fig.suptitle("SNR Distributions by Class", fontsize=11,
                 color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path


def plot_cross_class_mse_histograms(metrics_df: pd.DataFrame,
                                    output_dir) -> Path:
    """3x3 grid (head x horizon) of overlaid MSE histograms by class.

    Args:
        metrics_df: DataFrame with ``head``, ``horizon``, ``mse``,
            ``class_label``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / "cross_class_mse_histograms.png")
    if metrics_df.empty or "mse" not in metrics_df.columns:
        return output_path

    apply_publication_style()
    heads = [h for h in _HEAD_NAMES if h in metrics_df["head"].unique()]
    horizons = sorted(metrics_df["horizon"].unique())
    classes = sorted(metrics_df["class_label"].unique())
    class_colors = get_class_colors(classes)
    n_heads = len(heads)
    n_hor = len(horizons)

    fig, axes = plt.subplots(n_heads, n_hor,
                             figsize=(4 * n_hor, 3.5 * n_heads),
                             constrained_layout=True)
    if n_heads == 1 and n_hor == 1:
        axes = np.array([[axes]])
    elif n_heads == 1:
        axes = axes[np.newaxis, :]
    elif n_hor == 1:
        axes = axes[:, np.newaxis]

    for ri, head in enumerate(heads):
        for ci, h in enumerate(horizons):
            ax = axes[ri, ci]
            h_df = metrics_df[(metrics_df["head"] == head) &
                              (metrics_df["horizon"] == h)]
            data_dict = {
                cls: h_df[h_df["class_label"] == cls]["mse"].dropna().values
                for cls in classes
            }
            _overlaid_histograms(ax, data_dict, class_colors, xlabel="MSE")
            ax.set_title(f"{HEAD_LABELS.get(head, head)}, h={h}", fontsize=9)
            style_axes(ax)

    fig.suptitle("MSE Distributions by Class", fontsize=11,
                 color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path


def plot_cross_class_loss_histograms(loss_df: pd.DataFrame,
                                     output_dir) -> Path:
    """5 subplots (one per loss component) of overlaid histograms by class.

    Args:
        loss_df: DataFrame with ``class_label`` and loss columns
            ``L_fus``, ``L_delta``, ``L_self``, ``L_te``, ``L_kl``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / "cross_class_loss_histograms.png")
    if loss_df.empty:
        return output_path

    apply_publication_style()
    loss_cols = [c for c in ["L_fus", "L_delta", "L_self", "L_te", "L_kl"]
                 if c in loss_df.columns]
    if not loss_cols:
        return output_path

    classes = sorted(loss_df["class_label"].unique())
    class_colors = get_class_colors(classes)
    n_loss = len(loss_cols)

    fig, axes = plt.subplots(1, n_loss, figsize=(4 * n_loss, 4),
                             constrained_layout=True, squeeze=False)

    for li, lc in enumerate(loss_cols):
        ax = axes[0, li]
        data_dict = {
            cls: loss_df[loss_df["class_label"] == cls][lc].dropna().values
            for cls in classes
        }
        _overlaid_histograms(ax, data_dict, class_colors, xlabel=lc)
        ax.set_title(lc, fontsize=9)
        style_axes(ax)

    fig.suptitle("Loss Component Distributions by Class", fontsize=11,
                 color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path


def plot_cross_class_kl_histograms(te_seg_df: pd.DataFrame,
                                   output_dir) -> Path:
    """Single overlaid histogram of KL divergence by class.

    Args:
        te_seg_df: Segment-level TE DataFrame with ``kl_mean`` and
            ``class_label``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / "cross_class_kl_histograms.png")
    if te_seg_df.empty:
        return output_path

    apply_publication_style()
    classes = sorted(te_seg_df["class_label"].unique())
    class_colors = get_class_colors(classes)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    data_dict = {
        cls: te_seg_df[te_seg_df["class_label"] == cls][
            "kl_mean"].dropna().values
        for cls in classes
    }
    _overlaid_histograms(ax, data_dict, class_colors, xlabel="KL Mean")
    ax.set_title("KL Divergence Distribution by Class", fontsize=9, pad=6)
    style_axes(ax)
    save_figure(fig, output_path)
    return output_path


# ===================================================================
# Category 7: Dataset statistics
# ===================================================================

def plot_dataset_overview(stats: Dict[str, Any], output_dir) -> Path:
    """2x2 grid: sample counts, GUIDs per class, epochs per GUID hist, time dist.

    Args:
        stats: Dictionary with keys:
            ``"samples_per_class"``: ``{class: count}``
            ``"guids_per_class"``: ``{class: count}``
            ``"epochs_per_guid"``: ``{guid: n_epochs}`` or list of counts
            ``"time_distribution"``: ``{class: 1-D array of epoch_hours}``
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "dataset_overview.png")
    if not stats:
        return output_path

    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    # --- Panel 0,0: Sample counts ---
    ax = axes[0, 0]
    spc = stats.get("samples_per_class", {})
    if spc:
        classes = sorted(spc.keys())
        class_colors = get_class_colors(classes)
        bars = ax.bar(classes, [spc[c] for c in classes],
                      color=[class_colors[c] for c in classes], alpha=0.85,
                      edgecolor=COLOR_BLACK, linewidth=0.3)
        for bar, cls in zip(bars, classes):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5, str(spc[cls]),
                    ha="center", va="bottom", fontsize=7)
    ax.set_title("Samples per Class", fontsize=9)
    ax.set_ylabel("Count", fontsize=8)
    style_axes(ax)

    # --- Panel 0,1: GUIDs per class ---
    ax = axes[0, 1]
    gpc = stats.get("guids_per_class", {})
    if gpc:
        classes = sorted(gpc.keys())
        class_colors = get_class_colors(classes)
        bars = ax.bar(classes, [gpc[c] for c in classes],
                      color=[class_colors[c] for c in classes], alpha=0.85,
                      edgecolor=COLOR_BLACK, linewidth=0.3)
        for bar, cls in zip(bars, classes):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5, str(gpc[cls]),
                    ha="center", va="bottom", fontsize=7)
    ax.set_title("GUIDs per Class", fontsize=9)
    ax.set_ylabel("Count", fontsize=8)
    style_axes(ax)

    # --- Panel 1,0: Epochs per GUID histogram ---
    ax = axes[1, 0]
    epg = stats.get("epochs_per_guid", {})
    if epg:
        if isinstance(epg, dict):
            counts = list(epg.values())
        else:
            counts = list(epg)
        if counts:
            ax.hist(counts, bins=min(30, max(len(counts), 5)), alpha=0.8,
                    color=COLOR_BLUE, edgecolor=COLOR_BLACK, linewidth=0.3)
    ax.set_title("Epochs per GUID", fontsize=9)
    ax.set_xlabel("Number of Epochs", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    style_axes(ax)

    # --- Panel 1,1: Time-to-delivery distribution ---
    ax = axes[1, 1]
    td = stats.get("time_distribution", {})
    if td:
        classes = sorted(td.keys())
        class_colors = get_class_colors(classes)
        data_dict = {cls: np.asarray(td[cls]) for cls in classes}
        _overlaid_histograms(ax, data_dict, class_colors,
                             xlabel="Hours Before Delivery")
    ax.set_title("Time-to-Delivery Distribution", fontsize=9)
    style_axes(ax)

    fig.suptitle("Dataset Overview", fontsize=11, color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path


def plot_time_distribution(stats: Dict[str, Any], output_dir) -> Path:
    """Detailed time-to-delivery histograms per class (separate panels).

    Args:
        stats: Dictionary with ``"time_distribution"``: ``{class: 1-D array
            of epoch_hours}``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(Path(output_dir) / "time_distribution.png")
    td = stats.get("time_distribution", {})
    if not td:
        return output_path

    apply_publication_style()
    classes = sorted(td.keys())
    class_colors = get_class_colors(classes)
    n_cls = max(len(classes), 1)

    fig, axes = plt.subplots(1, n_cls, figsize=(5 * n_cls, 4),
                             constrained_layout=True, squeeze=False)

    for ci, cls in enumerate(classes):
        ax = axes[0, ci]
        vals = np.asarray(td[cls])
        if len(vals) > 0:
            ax.hist(vals, bins=40, alpha=0.8, color=class_colors[cls],
                    edgecolor=COLOR_BLACK, linewidth=0.3)
            ax.axvline(np.median(vals), color=COLOR_BLACK, linestyle="--",
                       linewidth=0.8,
                       label=f"median={np.median(vals):.1f}h")
        ax.set_title(cls, fontsize=9)
        ax.set_xlabel("Hours Before Delivery", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.legend(fontsize=7, framealpha=0.95)
        style_axes(ax)

    fig.suptitle("Time-to-Delivery Distributions", fontsize=11,
                 color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path


def plot_st_coefficient_stats(stats: Dict[str, Any], output_dir) -> Path:
    """Per-channel mean/std of FHR-ST and UP-ST per class.

    Args:
        stats: Dictionary with:
            ``"fhr_st_stats"``: ``{class: {"mean": (d_f,), "std": (d_f,)}}``
            ``"up_st_stats"``: ``{class: {"mean": (d_u,), "std": (d_u,)}}``
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_path = _ensure_path(
        Path(output_dir) / "st_coefficient_stats.png")
    if not stats:
        return output_path

    fhr_stats = stats.get("fhr_st_stats", {})
    up_stats = stats.get("up_st_stats", {})
    if not fhr_stats and not up_stats:
        return output_path

    apply_publication_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    # --- FHR-ST ---
    ax = axes[0]
    if fhr_stats:
        classes = sorted(fhr_stats.keys())
        class_colors = get_class_colors(classes)
        for cls in classes:
            mean_vals = np.asarray(fhr_stats[cls]["mean"])
            std_vals = np.asarray(fhr_stats[cls]["std"])
            x = np.arange(len(mean_vals))
            ax.plot(x, mean_vals, color=class_colors[cls], linewidth=1.0,
                    label=cls)
            ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals,
                            color=class_colors[cls], alpha=0.15)
    ax.set_title("FHR-ST Channel Statistics", fontsize=9, pad=6)
    ax.set_xlabel("Channel", fontsize=8)
    ax.set_ylabel("Value", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.95)
    style_axes(ax, grid="both")

    # --- UP-ST ---
    ax = axes[1]
    if up_stats:
        classes = sorted(up_stats.keys())
        class_colors = get_class_colors(classes)
        for cls in classes:
            mean_vals = np.asarray(up_stats[cls]["mean"])
            std_vals = np.asarray(up_stats[cls]["std"])
            x = np.arange(len(mean_vals))
            ax.plot(x, mean_vals, color=class_colors[cls], linewidth=1.0,
                    label=cls)
            ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals,
                            color=class_colors[cls], alpha=0.15)
    ax.set_title("UP-ST Channel Statistics", fontsize=9, pad=6)
    ax.set_xlabel("Channel", fontsize=8)
    ax.set_ylabel("Value", fontsize=8)
    ax.legend(fontsize=7, framealpha=0.95)
    style_axes(ax, grid="both")

    fig.suptitle("Scattering Transform Coefficient Statistics", fontsize=11,
                 color=COLOR_PURPLE)
    save_figure(fig, output_path)
    return output_path
