"""Diagnostic plotting callback for :class:`SeqVaeLagAttnV1` training.

Generates a consolidated multi-row publication-quality figure on every
validation epoch (gated by ``plot_frequency``). Style helpers are imported
from :mod:`model.transformer.tr_testing.style` so training diagnostics and
test-time figures share the same visual language.

Row layout
----------
0.  Raw FHR + UP — twin y-axes (optional, only if ``fhr``/``up`` are loaded).
1.  FHR features — stacked ``[fhr_st | fhr_ph]`` heatmap (87, T) with a
    horizontal separator at the st/ph boundary.
2.  UP features — stacked ``[up_st | up_ph]`` heatmap (101, T) with a
    separator at the st/ph boundary. Collapses to just ``up_ph`` (58, T)
    when ``use_up_st=False``.
3.  Latent ``z`` heatmap (d_z, T).
4.  Posterior μ vs prior μ stacked heatmap (2·d_z, T) with separator.
5.  KLD per-dim — ``d_z`` tiny panels in a 4×6 grid, each on its own y-axis
    so low-variance dimensions are still readable.
6.  Total KLD per time step + attention entropy (twin-axis trace).
7.  Lag attention matrix (L, T) with the argmax-lag-per-step overlay.
8.  TE lag attribution map (``kld_per_t × mean_alpha``) (L, T).
9.  Feature forecast at an anchor — three subplots for three selected
    channels showing GT vs baseline vs full prediction across the horizon.
10. Per-anchor feature MAE for baseline and full (shows the "UP gain").
11. Residual magnitude ``||delta_mu_src||`` vs KL (twin-axis trace).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from loguru import logger

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402

from model.transformer.tr_testing.style import (
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_LIGHT_GRAY,
    COLOR_ORANGE,
    COLOR_PURPLE,
    COLOR_SKY,
    COLOR_VERMILLION,
    SAVE_DPI,
    add_colorbar,
    apply_publication_style,
    heatmap,
    save_figure,
    style_axes,
)


# =============================================================================
# Small helpers
# =============================================================================


def _first_validation_batch(trainer) -> Optional[Any]:
    """Fetch the first validation batch from the trainer (rank-0 safe)."""
    val_dls = trainer.val_dataloaders
    if val_dls is None:
        return None
    dl = val_dls[0] if isinstance(val_dls, (list, tuple)) else val_dls
    try:
        return next(iter(dl))
    except StopIteration:
        return None


def _get_field(batch: Any, name: str) -> Optional[Any]:
    """Safely pull ``name`` from an ``AttributeDict`` or dict batch."""
    if isinstance(batch, dict):
        return batch.get(name)
    return getattr(batch, name, None)


def _guid_of(batch: Any, index: int = 0) -> str:
    """Extract a human-readable GUID string for sample ``index``."""
    field = _get_field(batch, "guid")
    if field is None:
        return "unknown"
    if isinstance(field, (list, tuple)):
        if not field:
            return "unknown"
        return str(field[index % len(field)])
    if isinstance(field, torch.Tensor):
        try:
            return str(field[index].item())
        except Exception:  # noqa: BLE001
            return "unknown"
    return str(field)


def _np(t: torch.Tensor) -> np.ndarray:
    """Detach + move to CPU + float32 numpy view (strict, non-optional)."""
    return t.detach().cpu().float().numpy()




def _kld_per_dim_np(
    mu_prior: np.ndarray,
    logvar_prior: np.ndarray,
    mu_post: np.ndarray,
    logvar_post: np.ndarray,
) -> np.ndarray:
    """Closed-form diagonal-Gaussian KL, per timestep per latent dim.

    Args:
        mu_prior: ``(T, d_z)`` prior mean.
        logvar_prior: ``(T, d_z)`` prior log-variance.
        mu_post: ``(T, d_z)`` posterior mean.
        logvar_post: ``(T, d_z)`` posterior log-variance.

    Returns:
        ``(T, d_z)`` per-step per-dim KL in nats.
    """
    return 0.5 * (
        logvar_prior
        - logvar_post
        + (np.exp(logvar_post) + (mu_post - mu_prior) ** 2) / np.exp(logvar_prior)
        - 1.0
    )


def _shade_warmup(ax: Any, warmup: int, *, color: str = COLOR_LIGHT_GRAY) -> None:
    """Shade the first ``warmup`` decimated steps on an axes."""
    if warmup and warmup > 0:
        ax.axvspan(-0.5, warmup - 0.5, color=color, alpha=0.35, zorder=0)


def _stack_feature_blocks(
    top: np.ndarray, bottom: Optional[np.ndarray]
) -> Tuple[np.ndarray, Optional[int]]:
    """Vertically stack two feature blocks and return the separator row.

    Args:
        top: ``(C_top, T)`` upper block.
        bottom: ``(C_bot, T)`` lower block, or ``None``.

    Returns:
        ``(stacked, separator_row)`` where ``separator_row`` is the y-position
        of the boundary between the two blocks (``C_top - 0.5``), or ``None``
        when only the top block is present.
    """
    if bottom is None:
        return top, None
    stacked = np.concatenate([top, bottom], axis=0)
    return stacked, top.shape[0] - 1


# =============================================================================
# Figure builder
# =============================================================================


def _build_diagnostic_figure(
    *,
    outs: Dict[str, Any],
    y_st: torch.Tensor,
    y_ph: torch.Tensor,
    up_st: Optional[torch.Tensor],
    up_self_phase: torch.Tensor,
    fhr_raw: Optional[torch.Tensor],
    up_raw: Optional[torch.Tensor],
    sample_idx: int,
    epoch: int,
    guid: str,
    warmup: int,
    horizon: int,
    forecast_channels: Tuple[int, ...],
    forecast_anchor_frac: float,
    beta: float,
    feat_loss: float,
    base_loss: float,
    kld_loss: float,
) -> "plt.Figure":
    """Build the full diagnostic figure for one validation sample.

    Args:
        outs: Forward-output dict from :meth:`SeqVaeLagAttnV1.forward`.
        y_st: FHR scattering features ``(B, T, 43)``.
        y_ph: FHR phase features ``(B, T, 44)``.
        up_st: UP scattering features ``(B, T, 43)``, or ``None`` if absent.
        up_self_phase: UP self-phase slice ``(B, T, 58)``.
        fhr_raw: Raw FHR trace ``(B, R)`` or ``None``.
        up_raw: Raw UP trace ``(B, R)`` or ``None``.
        sample_idx: Index into the batch.
        epoch: Current training epoch.
        guid: GUID string for the figure title.
        warmup: Warmup period ``T_w`` (for shading invalid regions).
        horizon: Decimated forecast horizon ``H_d``.
        forecast_channels: Three channel indices to show in the anchor-forecast row.
        forecast_anchor_frac: Anchor position as a fraction of ``T`` (0..1).
        beta: Current KL weight.
        feat_loss: Current ``L_feat`` (full-forecast MSE).
        base_loss: Current ``L_base`` (baseline-forecast MSE).
        kld_loss: Current ``L_KL`` (mean KL).

    Returns:
        The constructed :class:`matplotlib.figure.Figure`. The caller is
        responsible for saving and closing it.
    """
    i = sample_idx

    # ---- Numpy views of the relevant tensors for this sample ---------------
    y_st_np = _np(y_st[i])                                          # (T, 43)
    y_ph_np = _np(y_ph[i])                                          # (T, 44)
    up_self_np = _np(up_self_phase[i])                              # (T, 58)
    up_st_np = _np(up_st[i]) if up_st is not None else None         # (T, 43) or None

    mu_prior_np = _np(outs["mu_prior"][i])                          # (T, d_z)
    logvar_prior_np = _np(outs["logvar_prior"][i])
    mu_post_np = _np(outs["mu_post"][i])
    logvar_post_np = _np(outs["logvar_post"][i])
    z_np = _np(outs["z"][i])                                        # (T, d_z)

    attn_np = _np(outs["attn_weights"][i])                          # (T, M, L)
    te_lag_np = _np(outs["te_lag_map"][i])                          # (T, L)
    kld_per_t_np = _np(outs["kld_per_t"][i])                        # (T,)

    mu_base_np = _np(outs["mu_base"][i])                            # (T, H_d, 87)
    mu_full_np = _np(outs["mu_full"][i])                            # (T, H_d, 87)
    delta_mu_src_np = _np(outs["delta_mu_src"][i])                  # (T, H_d, 87)

    T = y_st_np.shape[0]
    d_z = mu_prior_np.shape[-1]
    L = attn_np.shape[-1]
    H_d = horizon

    Y_full_np = np.concatenate([y_st_np, y_ph_np], axis=-1)         # (T, 87)

    # Per-dim per-step KL, computed once and reused.
    kld_per_dim = _kld_per_dim_np(
        mu_prior_np, logvar_prior_np, mu_post_np, logvar_post_np
    )  # (T, d_z)

    # Mean attention over heads and argmax lag per step (for lag overlay).
    mean_alpha = attn_np.mean(axis=1)                               # (T, L)
    argmax_lag = mean_alpha.argmax(axis=-1)                         # (T,)

    # Attention entropy per step (sanity check for sharpness evolution).
    eps = 1e-12
    attn_entropy_per_step = -(mean_alpha * np.log(mean_alpha + eps)).sum(axis=-1)

    # Future target via unfold (on numpy).
    if T - H_d > 0:
        # Y_plus_np[t] == Y[t+1 : t+1+H_d, :] for t in [0, T-H_d)
        Y_shift_np = Y_full_np[1:]                                  # (T-1, 87)
        Y_plus_np = np.lib.stride_tricks.sliding_window_view(
            Y_shift_np, window_shape=H_d, axis=0
        )  # (T-H_d, 87, H_d)
        Y_plus_np = Y_plus_np.transpose(0, 2, 1)                    # (T-H_d, H_d, 87)
    else:
        Y_plus_np = None

    # Per-anchor MAE traces.
    valid_anchors = np.arange(warmup, max(warmup, T - H_d))
    mae_full = None
    mae_base = None
    if Y_plus_np is not None and valid_anchors.size > 0:
        diff_full = np.abs(mu_full_np[valid_anchors] - Y_plus_np[valid_anchors])
        diff_base = np.abs(mu_base_np[valid_anchors] - Y_plus_np[valid_anchors])
        mae_full = diff_full.mean(axis=(1, 2))                      # (T_valid,)
        mae_base = diff_base.mean(axis=(1, 2))

    # Residual magnitude per anchor.
    delta_src_l2 = np.linalg.norm(
        delta_mu_src_np.reshape(T, -1), axis=-1
    )                                                               # (T,)

    # ------------------------------------------------------------------
    # Figure and gridspec
    # ------------------------------------------------------------------
    apply_publication_style()

    has_raw = fhr_raw is not None and up_raw is not None

    # Row heights: raw is optional, KLD-per-dim grid takes 3x a normal row.
    row_specs = []  # (name, height_ratio)
    if has_raw:
        row_specs.append(("raw", 1.0))
    row_specs += [
        ("fhr_feats", 1.3),
        ("up_feats", 1.4),
        ("z_latent", 1.0),
        ("post_prior", 1.1),
        ("kld_dims", 4.0),
        ("kld_total", 0.9),
        ("lag_attn", 1.2),
        ("te_lag", 1.2),
        ("forecast", 1.0),
        ("mae", 0.9),
        ("residual", 0.9),
    ]
    n_rows = len(row_specs)
    height_ratios = [h for _, h in row_specs]
    total_height = sum(height_ratios) * 3.2
    # Use manual layout — constrained_layout conflicts with bbox_inches="tight"
    # at save time for large nested-gridspec figures.
    fig = plt.figure(figsize=(16, total_height))
    gs = GridSpec(
        n_rows, 3, figure=fig, height_ratios=height_ratios,
        left=0.05, right=0.97, top=0.985, bottom=0.02,
        hspace=0.55, wspace=0.22,
    )

    def row_axes(row: int) -> Any:
        return fig.add_subplot(gs[row, :])

    row_idx_of = {name: i for i, (name, _) in enumerate(row_specs)}

    # ---- Row: Raw FHR + UP -------------------------------------------------
    if has_raw:
        ax = row_axes(row_idx_of["raw"])
        assert fhr_raw is not None and up_raw is not None  # has_raw guard
        fhr_np = _np(fhr_raw[i]).ravel()
        up_np = _np(up_raw[i]).ravel()
        fs = 4.0  # sampling rate in Hz
        t_raw = np.arange(len(fhr_np)) / fs
        ax.plot(t_raw, fhr_np, color=COLOR_BLUE, linewidth=0.8, label="FHR")
        ax2 = ax.twinx()
        ax2.plot(t_raw, up_np, color=COLOR_GREEN, linewidth=0.8, label="UP")
        ax.set_title("Raw FHR / UP signals", fontsize=9, pad=6)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("FHR (bpm, normalised)", fontsize=8, color=COLOR_BLUE)
        ax.tick_params(axis="y", labelcolor=COLOR_BLUE)
        ax2.set_ylabel("UP (mmHg, normalised)", fontsize=8, color=COLOR_GREEN)
        ax2.tick_params(axis="y", labelcolor=COLOR_GREEN)
        ax.set_xlim(t_raw[0], t_raw[-1])
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2, labels1 + labels2,
            loc="upper right", fontsize=7, framealpha=0.95,
        )
        style_axes(ax, grid="both")

    # ---- Row: FHR features stacked heatmap --------------------------------
    ax = row_axes(row_idx_of["fhr_feats"])
    fhr_stack, fhr_sep = _stack_feature_blocks(y_st_np.T, y_ph_np.T)     # (87, T)
    heatmap(
        ax, fhr_stack, cmap="bwr", origin="upper",
        title="FHR features \u2014 scattering (rows 0-42)  |  phase (rows 43-86)",
        ylabel="Channel",
        label="Value",
        fig=fig,
    )
    if fhr_sep is not None:
        ax.axhline(fhr_sep + 0.5, color="white", linewidth=1.2, linestyle="--")
    _shade_warmup(ax, warmup)

    # ---- Row: UP features stacked heatmap ---------------------------------
    ax = row_axes(row_idx_of["up_feats"])
    if up_st_np is not None:
        up_stack, up_sep = _stack_feature_blocks(up_st_np.T, up_self_np.T)  # (101, T)
        title_up = (
            "UP features \u2014 scattering (rows 0-42)  |  self-phase (rows 43-100)"
        )
    else:
        up_stack, up_sep = up_self_np.T, None                               # (58, T)
        title_up = "UP features \u2014 self-phase only (up_st absent)"
    heatmap(
        ax, up_stack, cmap="bwr", origin="upper",
        title=title_up, ylabel="Channel", label="Value", fig=fig,
    )
    if up_sep is not None:
        ax.axhline(up_sep + 0.5, color="white", linewidth=1.2, linestyle="--")
    _shade_warmup(ax, warmup)

    # ---- Row: Latent z ----------------------------------------------------
    ax = row_axes(row_idx_of["z_latent"])
    heatmap(
        ax, z_np.T, cmap="bwr", origin="lower",
        title=f"Latent z (d_z={d_z})", ylabel="Latent dim",
        label="z", fig=fig,
    )
    _shade_warmup(ax, warmup)

    # ---- Row: Posterior μ vs prior μ split heatmap ------------------------
    ax = row_axes(row_idx_of["post_prior"])
    post_prior = np.concatenate([mu_post_np.T, mu_prior_np.T], axis=0)   # (2*d_z, T)
    vabs = np.nanmax(np.abs(post_prior)) or 1.0
    im = ax.imshow(
        post_prior, aspect="auto", cmap="bwr", origin="upper",
        vmin=-vabs, vmax=vabs,
    )
    ax.axhline(d_z - 0.5, color="white", linewidth=1.2, linestyle="--")
    ax.set_yticks([d_z // 2, d_z + d_z // 2])
    ax.set_yticklabels(["Posterior \u03bc", "Prior \u03bc\u2070"])
    ax.set_xlabel("Time Steps", fontsize=8)
    ax.set_title(
        "Posterior vs Prior means (TEB residual = posterior - prior)",
        fontsize=9, pad=6,
    )
    ax.grid(False)
    for spine in ("top", "bottom", "left", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(COLOR_BLACK)
        ax.spines[spine].set_linewidth(0.6)
    add_colorbar(fig, im, ax, label="Value")
    _shade_warmup(ax, warmup)

    # ---- Row: KLD per-dim small multiples (4 x 6 nested grid) --------------
    sub_gs = GridSpecFromSubplotSpec(
        4, 6, subplot_spec=gs[row_idx_of["kld_dims"], :],
        wspace=0.35, hspace=0.55,
    )
    for d in range(min(24, d_z)):
        rr, cc = divmod(d, 6)
        ax_d = fig.add_subplot(sub_gs[rr, cc])
        trace = kld_per_dim[:, d]
        ax_d.plot(np.arange(T), trace, color=COLOR_PURPLE, linewidth=0.8)
        _shade_warmup(ax_d, warmup)
        ax_d.set_title(
            f"z{d}  max={trace.max():.2f}",
            fontsize=6, pad=2,
        )
        ax_d.tick_params(axis="both", which="both", labelsize=5, length=2)
        # Minimal tick labelling: left column keeps y-ticks, bottom row keeps x-ticks.
        if cc != 0:
            ax_d.set_yticklabels([])
        if rr != 3:
            ax_d.set_xticklabels([])
        for spine in ("top", "bottom", "left", "right"):
            ax_d.spines[spine].set_linewidth(0.4)
            ax_d.spines[spine].set_color(COLOR_GRAY)
    # Shared supertitle for the mini-grid (uses the first cell's position).
    # Drop a synthetic annotation via fig.text so constrained_layout can handle it.
    try:
        bbox = sub_gs.get_topmost_subplotspec().get_position(fig)  # type: ignore[attr-defined]
    except Exception:
        bbox = None
    if bbox is not None:
        fig.text(
            0.5, bbox.y1 + 0.005,
            "KLD per latent dim (independent y-axis per dim)",
            ha="center", va="bottom", fontsize=9, color=COLOR_BLACK,
        )

    # ---- Row: Total KL per step + attention entropy (twin-axis) -----------
    ax = row_axes(row_idx_of["kld_total"])
    t_steps = np.arange(T)
    ax.plot(
        t_steps, kld_per_t_np, color=COLOR_PURPLE,
        linewidth=1.0, label="KL per step",
    )
    ax.set_ylabel("KL (nats)", fontsize=8, color=COLOR_PURPLE)
    ax.tick_params(axis="y", labelcolor=COLOR_PURPLE)
    ax2 = ax.twinx()
    ax2.plot(
        t_steps, attn_entropy_per_step, color=COLOR_ORANGE,
        linewidth=0.9, alpha=0.85, label="Attention entropy",
    )
    ax2.set_ylabel("Attention entropy (nats)", fontsize=8, color=COLOR_ORANGE)
    ax2.tick_params(axis="y", labelcolor=COLOR_ORANGE)
    _shade_warmup(ax, warmup)
    ax.set_title(
        "Total KL per timestep vs mean attention entropy", fontsize=9, pad=6,
    )
    ax.set_xlabel("Time Steps", fontsize=8)
    ax.set_xlim(0, T - 1)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper right", fontsize=7, framealpha=0.95,
    )
    style_axes(ax, grid="both")

    # ---- Row: Lag attention matrix (mean over heads) ----------------------
    ax = row_axes(row_idx_of["lag_attn"])
    im = ax.imshow(
        mean_alpha.T, aspect="auto", cmap="viridis", origin="lower",
        extent=[0, T - 1, 0, L - 1],
    )
    ax.plot(
        t_steps, argmax_lag, color=COLOR_VERMILLION,
        linewidth=0.9, alpha=0.9, label="argmax lag",
    )
    ax.set_title(
        f"Lag attention \u2014 mean over {attn_np.shape[1]} heads "
        f"(L={L} = 0..{L - 1})",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time Steps (anchor t)", fontsize=8)
    ax.set_ylabel("Lag \u2113 (0 = current)", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.95)
    for spine in ("top", "bottom", "left", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(COLOR_BLACK)
        ax.spines[spine].set_linewidth(0.6)
    add_colorbar(fig, im, ax, label="Attn prob")
    _shade_warmup(ax, warmup)

    # ---- Row: TE lag map (KL-weighted attention) --------------------------
    ax = row_axes(row_idx_of["te_lag"])
    te_map = te_lag_np.T  # (L, T)
    vabs_te = np.nanmax(np.abs(te_map)) or 1.0
    im = ax.imshow(
        te_map, aspect="auto", cmap="magma", origin="lower",
        extent=[0, T - 1, 0, L - 1], vmin=0.0, vmax=vabs_te,
    )
    ax.set_title(
        "TE lag attribution (KL \u00d7 mean-\u03b1) \u2014 which lag drives KL at each step",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time Steps (anchor t)", fontsize=8)
    ax.set_ylabel("Lag \u2113", fontsize=8)
    for spine in ("top", "bottom", "left", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(COLOR_BLACK)
        ax.spines[spine].set_linewidth(0.6)
    add_colorbar(fig, im, ax, label="KL weight")
    _shade_warmup(ax, warmup)

    # ---- Row: Feature forecast at one anchor (3 subplots) -----------------
    a0 = int(round(forecast_anchor_frac * (T - H_d - 1)))
    a0 = max(warmup, min(a0, max(warmup, T - H_d - 1)))
    chosen_channels = list(forecast_channels)[:3]
    # Always pad to 3 with 0s if fewer specified
    while len(chosen_channels) < 3:
        chosen_channels.append(0)

    for j, ch in enumerate(chosen_channels):
        ax_f = fig.add_subplot(gs[row_idx_of["forecast"], j])
        if Y_plus_np is not None and a0 < Y_plus_np.shape[0]:
            gt = Y_plus_np[a0, :, ch]
            pred_base = mu_base_np[a0, :, ch]
            pred_full = mu_full_np[a0, :, ch]
            horizon_axis = np.arange(1, H_d + 1)
            ax_f.plot(
                horizon_axis, gt, color=COLOR_BLACK,
                linewidth=1.2, label="GT", zorder=4,
            )
            ax_f.plot(
                horizon_axis, pred_base, color=COLOR_BLUE,
                linewidth=1.0, alpha=0.85, label="baseline",
            )
            ax_f.plot(
                horizon_axis, pred_full, color=COLOR_ORANGE,
                linewidth=1.0, alpha=0.9, label="full",
            )
        block = "fhr_st" if ch < y_st_np.shape[-1] else "fhr_ph"
        local_ch = ch if ch < y_st_np.shape[-1] else ch - y_st_np.shape[-1]
        ax_f.set_title(
            f"Forecast @ t={a0}, ch={ch} ({block}[{local_ch}])",
            fontsize=9, pad=4,
        )
        ax_f.set_xlabel("Horizon step (+\u03c4)", fontsize=7)
        if j == 0:
            ax_f.set_ylabel("Feature value", fontsize=8)
        ax_f.legend(loc="upper right", fontsize=6, framealpha=0.9)
        style_axes(ax_f, grid="both")

    # ---- Row: Per-anchor feature MAE --------------------------------------
    ax = row_axes(row_idx_of["mae"])
    if mae_full is not None and mae_base is not None:
        ax.plot(
            valid_anchors, mae_base, color=COLOR_BLUE,
            linewidth=1.0, alpha=0.9, label="baseline MAE",
        )
        ax.plot(
            valid_anchors, mae_full, color=COLOR_ORANGE,
            linewidth=1.0, alpha=0.9, label="full MAE",
        )
        gain = mae_base - mae_full
        ax2 = ax.twinx()
        ax2.plot(
            valid_anchors, gain, color=COLOR_GREEN,
            linewidth=0.9, alpha=0.85, label="gain (base-full)",
        )
        ax2.axhline(0.0, color=COLOR_GRAY, linewidth=0.5, linestyle="--")
        ax2.set_ylabel("Gain", fontsize=8, color=COLOR_GREEN)
        ax2.tick_params(axis="y", labelcolor=COLOR_GREEN)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2, labels1 + labels2,
            loc="upper right", fontsize=7, framealpha=0.95,
        )
    ax.set_title(
        "Per-anchor feature MAE (averaged over horizon and channels)",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Anchor t", fontsize=8)
    ax.set_ylabel("MAE", fontsize=8)
    ax.set_xlim(0, T - 1)
    _shade_warmup(ax, warmup)
    style_axes(ax, grid="both")

    # ---- Row: Residual magnitude vs KL -----------------------------------
    ax = row_axes(row_idx_of["residual"])
    ax.plot(
        t_steps, delta_src_l2, color=COLOR_SKY,
        linewidth=1.0, label="||\u0394\u03bc_src||",
    )
    ax.set_ylabel("L2 norm", fontsize=8, color=COLOR_SKY)
    ax.tick_params(axis="y", labelcolor=COLOR_SKY)
    ax2 = ax.twinx()
    ax2.plot(
        t_steps, kld_per_t_np, color=COLOR_PURPLE,
        linewidth=0.9, alpha=0.85, label="KL per step",
    )
    ax2.set_ylabel("KL (nats)", fontsize=8, color=COLOR_PURPLE)
    ax2.tick_params(axis="y", labelcolor=COLOR_PURPLE)
    _shade_warmup(ax, warmup)
    ax.set_title(
        "Residual decoder usage: ||\u0394\u03bc_src|| vs KL per step",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time Steps", fontsize=8)
    ax.set_xlim(0, T - 1)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper right", fontsize=7, framealpha=0.95,
    )
    style_axes(ax, grid="both")

    # ---- Super title -------------------------------------------------------
    fig.suptitle(
        f"LagAttn v1 diagnostics  \u2014  epoch {epoch}  sample {sample_idx}  "
        f"guid {guid}  \u2022  \u03b2={beta:.4g}  "
        f"L_feat={feat_loss:.4f}  L_base={base_loss:.4f}  KL={kld_loss:.4e}",
        fontsize=11, color=COLOR_PURPLE, y=1.002,
    )
    return fig


# =============================================================================
# Callback
# =============================================================================


class LagAttnV1PlotCallback(Callback):
    """Periodic diagnostic-plot callback for :class:`SeqVaeLagAttnV1` training.

    Runs on ``on_validation_epoch_end``, every ``plot_frequency`` epochs,
    generating ``num_examples`` figures per trigger (rank-0 only).

    Args:
        output_dir: Directory under which to write ``lag_attn_v1_diagnostics``.
        plot_frequency: Plot every N validation epochs.
        num_examples: Number of samples from the first validation batch.
        file_format: Output image format (``"pdf"`` or ``"png"``).
        mlflow_logger: Optional MLflow logger — each saved file is registered
            as a run artifact when set.
        forecast_channels: Feature-channel indices to draw in the forecast row.
        forecast_anchor_frac: Anchor position as a fraction of ``T`` (clipped
            to a valid range at run time).
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        plot_frequency: int = 5,
        num_examples: int = 2,
        *,
        file_format: str = "pdf",
        mlflow_logger: Any = None,
        forecast_channels: Tuple[int, ...] = (0, 43, 80),
        forecast_anchor_frac: float = 0.6,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir) / "lag_attn_v1_diagnostics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_frequency = max(1, int(plot_frequency))
        self.num_examples = max(1, int(num_examples))
        self.file_format = file_format.lower().lstrip(".")
        self._mlflow_logger = mlflow_logger
        self.forecast_channels = tuple(int(c) for c in forecast_channels)
        self.forecast_anchor_frac = float(forecast_anchor_frac)

    # ------------------------------------------------------------------
    # Lightning hook
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        """Generate diagnostic plots at the configured frequency."""
        if not getattr(trainer, "is_global_zero", True):
            return
        if getattr(trainer, "sanity_checking", False):
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_frequency != 0:
            return

        batch = _first_validation_batch(trainer)
        if batch is None:
            logger.debug("LagAttnV1PlotCallback: no validation batch available.")
            return

        try:
            self._generate_plots(batch, pl_module, epoch)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"LagAttnV1PlotCallback failed: {exc}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log_artifact(self, path: Path) -> None:
        """Register ``path`` as an MLflow artifact if a logger is attached."""
        if self._mlflow_logger is None:
            return
        try:
            self._mlflow_logger.experiment.log_artifact(
                self._mlflow_logger.run_id, str(path),
            )
        except Exception as exc:  # noqa: BLE001 — plotting must never crash training
            logger.debug(f"MLflow artifact logging failed: {exc}")

    @torch.no_grad()
    def _generate_plots(self, batch: Any, pl_module: Any, epoch: int) -> None:
        """Run the model forward + loss on the batch and build figures."""
        # Move the batch to the PL module's device using Lightning's own helper.
        batch = pl_module.transfer_batch_to_device(
            batch, pl_module.device, dataloader_idx=0,
        )
        model = pl_module.orig_model
        use_up_st = bool(getattr(model, "use_up_st", False))

        y_st = batch.fhr_st
        y_ph = batch.fhr_ph
        fhr_up_ph = batch.fhr_up_ph

        up_self = fhr_up_ph[..., 79:137]
        up_st: Optional[torch.Tensor] = None
        if use_up_st:
            up_st_field = _get_field(batch, "up_st")
            if up_st_field is None:
                logger.warning(
                    "LagAttnV1PlotCallback: use_up_st=True but batch has no "
                    "`up_st` field; skipping this epoch's plots."
                )
                return
            up_st = up_st_field
            u_stream = torch.cat([up_st, up_self], dim=-1)
        else:
            u_stream = up_self

        fhr_raw = _get_field(batch, "fhr")
        up_raw = _get_field(batch, "up")

        was_training = pl_module.training
        pl_module.eval()
        try:
            outs = model(y_st, y_ph, u_stream)
            # Current-beta lookup: prefer hparam, fall back to 0.0
            beta = float(pl_module.hparams.get("kld_beta", 0.0))
            lambda_full = float(pl_module.hparams.get("lambda_full", 1.0))
            lambda_base = float(pl_module.hparams.get("lambda_base", 0.5))
            loss_dict = model.compute_loss(
                forward_outputs=outs,
                y_st=y_st,
                y_ph=y_ph,
                beta=beta,
                lambda_full=lambda_full,
                lambda_base=lambda_base,
            )
            feat_loss = float(loss_dict["feat_loss"].detach().cpu())
            base_loss = float(loss_dict["base_loss"].detach().cpu())
            kld_loss = float(loss_dict["kld_loss"].detach().cpu())

            warmup = int(getattr(model, "warmup_period", 0))
            horizon = int(getattr(model, "horizon", 30))

            num_samples = min(self.num_examples, y_st.shape[0])
            for s in range(num_samples):
                guid = _guid_of(batch, s)
                fig = _build_diagnostic_figure(
                    outs=outs,
                    y_st=y_st,
                    y_ph=y_ph,
                    up_st=up_st,
                    up_self_phase=up_self,
                    fhr_raw=fhr_raw,
                    up_raw=up_raw,
                    sample_idx=s,
                    epoch=epoch,
                    guid=guid,
                    warmup=warmup,
                    horizon=horizon,
                    forecast_channels=self.forecast_channels,
                    forecast_anchor_frac=self.forecast_anchor_frac,
                    beta=beta,
                    feat_loss=feat_loss,
                    base_loss=base_loss,
                    kld_loss=kld_loss,
                )
                fname = (
                    f"lag_attn_v1_epoch{epoch:04d}_sample{s}_"
                    f"{guid[:16]}.{self.file_format}"
                )
                save_path = self.output_dir / fname
                save_figure(fig, save_path, dpi=SAVE_DPI, close=True)
                self._log_artifact(save_path)
            logger.info(
                f"LagAttnV1PlotCallback: saved {num_samples} figure(s) "
                f"for epoch {epoch} to {self.output_dir}"
            )
        finally:
            if was_training:
                pl_module.train()


__all__ = ["LagAttnV1PlotCallback"]
