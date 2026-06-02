"""Per-sample diagnostic plots for the lag-attn v1 testing pipeline.

This module provides a single entry point,
:func:`plot_sample_lag_attn_diagnostic`, which builds a multi-row
publication-quality figure summarising one sample's model behaviour:

- raw FHR / UP traces (when available)
- stacked ``mu_full_avg`` / ``y_plus_avg`` / residual heatmaps over the
  87 feature channels
- latent ``z`` heatmap
- KLD-per-dim heatmap (derived from the posterior/prior moments)
- lag attention heatmap with argmax-lag overlay
- TE lag attribution heatmap

The overlap-averaging and block-stacking helpers are imported directly
from :mod:`model.vae_teb_prediction.model.plotting_callback_lag_attn_v1`
so the test plots and training callback share the exact same semantics.
Any sample field that is missing is rendered as an empty panel with a
short "N/A" note so the function degrades gracefully with sparse dicts.

Example:
    >>> samples = collect_predictions(runner, loader, max_samples=3)
    >>> plot_sample_lag_attn_diagnostic(
    ...     samples[0], Path("out/sample_0.pdf"),
    ...     warmup=runner.warmup_steps, horizon=runner.horizon,
    ... )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

from model.vae_teb_prediction.model.plotting_callback_lag_attn_v1 import (
    _average_forecast_per_channel,
    _shade_warmup,
    _time_axes,
)
from model.vae_teb_prediction.testing.visualizers import (
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_LIGHT_GRAY,
    COLOR_ORANGE,
    COLOR_PURPLE,
    COLOR_VERMILLION,
    FONT_LABEL,
    FONT_TITLE,
    SAVE_DPI,
    _add_colorbar,
    _style_axes,
)

# Default raw window length (20 min @ 4 Hz after the 1-min trim at each
# end: 300 × 16 = 4800 raw samples). The sample dict's "fhr" field carries
# the raw trace only when batch.fhr was loaded by the dataset. When the
# shape disagrees we fall back to whatever length is present.
_DEFAULT_R = 4800
_DEFAULT_FS_RAW = 4.0
_DEFAULT_DECIM = 16
_FHR_ST_END = 43  # channel index splitting scattering from phase


# -----------------------------------------------------------------------------
# Internal helpers for the minute-axis diagnostic figures
# -----------------------------------------------------------------------------


def _auto_suptitle(fig: Any, title: str, *, base_fontsize: float, y: float = 0.995) -> None:
    """Set a figure suptitle, shrinking the font if the text is long.

    GUIDs in this project are 32-character hex strings. Combined with
    epoch, class, and metric annotations the title line can exceed the
    figure width at the default font size. This helper scales the font
    down so the full title always fits without truncation or wrapping.

    Args:
        fig: Matplotlib Figure.
        title: Full title string.
        base_fontsize: Starting font size (used for short titles).
        y: Vertical position passed to ``fig.suptitle``.
    """
    # Empirically, at base_fontsize a ~80-char title fits a 14-inch
    # figure. Scale down linearly beyond that so the user always sees
    # the complete GUID.
    MAX_CHARS_AT_BASE = 80
    n = len(title)
    if n > MAX_CHARS_AT_BASE:
        fontsize = max(5.0, base_fontsize * MAX_CHARS_AT_BASE / n)
    else:
        fontsize = base_fontsize
    fig.suptitle(title, fontsize=fontsize, fontweight="normal", y=y)


def _shade_warmup_min(ax: plt.Axes, warmup_min: float) -> None:
    """Shade the warmup region in minutes on a given axes."""
    if warmup_min and warmup_min > 0:
        ax.axvspan(0.0, warmup_min, color=COLOR_LIGHT_GRAY, alpha=0.35, zorder=0)


def _mark_warmup_line_min(ax: plt.Axes, warmup_min: float) -> None:
    """Draw a vertical line at the end of the warmup window (minutes axis).

    Used on panels that show **real, unmasked input signals** (raw FHR/UP
    and the normalised scattering / phase-harmonic features). The line
    indicates where the model's warmup region ends without hiding any of
    the underlying signal.

    Args:
        ax: Target axes whose x-axis is in minutes.
        warmup_min: Warmup duration in minutes. No-op when ``<= 0``.
    """
    if warmup_min and warmup_min > 0:
        ax.axvline(
            x=float(warmup_min),
            color=COLOR_GRAY,
            linewidth=0.9,
            linestyle="--",
            alpha=0.85,
            zorder=3,
        )


def _mark_warmup_line_sec(
    ax: plt.Axes, warmup: int, t_max: float, T: int
) -> None:
    """Draw a vertical line at the end of the warmup window (seconds axis).

    Same purpose as :func:`_mark_warmup_line_min` but for axes expressed
    in seconds (``t_max = R / fs_raw``).

    Args:
        ax: Target axes whose x-axis is in seconds.
        warmup: Warmup length in decimated steps.
        t_max: Full x-axis extent in seconds.
        T: Total number of decimated steps.
    """
    if warmup and warmup > 0 and T > 0:
        warmup_sec = float(warmup) * (t_max / float(T))
        ax.axvline(
            x=warmup_sec,
            color=COLOR_GRAY,
            linewidth=0.9,
            linestyle="--",
            alpha=0.85,
            zorder=3,
        )


def _mask_warmup_time_axis(
    data: np.ndarray, warmup: int, axis: int = -1
) -> np.ndarray:
    """Return a float copy of ``data`` with the warmup prefix NaN-masked.

    Matplotlib's ``imshow`` renders NaN values as transparent and they are
    ignored by auto ``vmin``/``vmax`` scaling, so NaN-masking the warmup
    region hides any start-of-sample anomalies and prevents them from
    compressing the colour/value range of the rest of the plot.

    Args:
        data: Array whose ``axis`` dimension is the time axis.
        warmup: Number of leading steps on ``axis`` to mask. Values
            ``<= 0`` leave ``data`` unchanged.
        axis: Time-axis dimension (default last axis).

    Returns:
        A NaN-masked float copy of ``data``. Returns ``data`` unchanged
        when ``warmup <= 0``.
    """
    if data is None or warmup is None or warmup <= 0:
        return data
    if not np.issubdtype(data.dtype, np.floating):
        out = data.astype(np.float32, copy=True)
    else:
        out = data.copy()
    n = out.shape[axis]
    k = min(int(warmup), n)
    if k <= 0:
        return out
    slicer: list = [slice(None)] * out.ndim
    slicer[axis] = slice(0, k)
    out[tuple(slicer)] = np.nan
    return out


def _draw_raw_panel(
    ax: plt.Axes,
    *,
    fhr: Optional[np.ndarray],
    up: Optional[np.ndarray],
    time_raw_min: np.ndarray,
    t_max_min: float,
    warmup_min: float,
    title: str = "Raw FHR / UP",
) -> None:
    """Draw the raw FHR / UP trace panel in minutes.

    Args:
        ax: Target axes. FHR is drawn on the primary y-axis; UP on a
            twin axis if present.
        fhr: Raw FHR trace, shape ``(R,)`` or ``None``.
        up: Raw UP trace, shape ``(R,)`` or ``None``.
        time_raw_min: Raw time axis in minutes, length ``R``.
        t_max_min: Total window length in minutes (for ``set_xlim``).
        warmup_min: Warmup region length in minutes.
        title: Panel title.
    """
    # Raw FHR / UP are real model inputs — draw them unmasked across the
    # whole window and mark the end of the warmup region with a single
    # vertical line rather than blanking / shading the warmup samples.
    drawn = False
    if fhr is not None and fhr.ndim == 1:
        n = min(len(fhr), len(time_raw_min))
        ax.plot(
            time_raw_min[:n], fhr[:n],
            color=COLOR_BLUE, linewidth=0.7, label="FHR",
        )
        ax.set_ylabel("FHR (bpm)", fontsize=FONT_LABEL, color=COLOR_BLUE)
        ax.tick_params(axis="y", colors=COLOR_BLUE, labelsize=6)
        drawn = True
    if up is not None and up.ndim == 1:
        ax_up = ax.twinx()
        n = min(len(up), len(time_raw_min))
        ax_up.plot(
            time_raw_min[:n], up[:n],
            color=COLOR_VERMILLION, linewidth=0.7, alpha=0.85, label="UP",
        )
        ax_up.set_ylabel("UP (mmHg)", fontsize=FONT_LABEL, color=COLOR_VERMILLION)
        ax_up.tick_params(axis="y", colors=COLOR_VERMILLION, labelsize=6)
        drawn = True
    if not drawn:
        ax.text(
            0.5, 0.5, "raw traces not available",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_ylabel("raw", fontsize=FONT_LABEL)

    ax.set_xlim(0.0, t_max_min)
    ax.set_title(title, fontsize=FONT_LABEL, fontweight="normal")
    _mark_warmup_line_min(ax, warmup_min)
    _style_axes(ax, grid="major", minor_ticks=False)


def _draw_heatmap_min(
    ax: plt.Axes,
    data: np.ndarray,
    *,
    t_max_min: float,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ylabel: str = "",
    title: str = "",
    colorbar_label: str = "",
    y_extent: Optional[float] = None,
) -> Any:
    """Draw a ``(rows, T)`` heatmap whose x-axis is time in minutes.

    Args:
        ax: Target axes.
        data: Array of shape ``(rows, T)``.
        t_max_min: Full x-axis extent in minutes.
        cmap: Colormap name.
        vmin: Minimum colour value (auto if None).
        vmax: Maximum colour value (auto if None).
        ylabel: Y-axis label.
        title: Panel title.
        colorbar_label: Colorbar label.
        y_extent: Optional y-axis extent (rows space). Defaults to
            ``(−0.5, rows − 0.5)`` which keeps pixel-per-row semantics.

    Returns:
        The ``AxesImage`` handle, or ``None`` if the panel was empty.
    """
    if data.size == 0 or not np.isfinite(data).any():
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
        if title:
            ax.set_title(title, fontsize=FONT_LABEL)
        return None

    rows = int(data.shape[0])
    y_top = float(y_extent) if y_extent is not None else (rows - 0.5)
    y_bot = 0.0 if y_extent is not None else -0.5

    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        extent=(0.0, t_max_min, y_bot, y_top),
    )
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    if title:
        ax.set_title(title, fontsize=FONT_LABEL, fontweight="normal")
    _add_colorbar(ax.figure, im, ax, label=colorbar_label)  # type: ignore[arg-type]
    return im


def _imshow_panel(
    ax: plt.Axes,
    data: np.ndarray,
    *,
    t_max: float,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    symmetric: bool = False,
    ylabel: str = "",
    separator_row: Optional[int] = None,
    title: str = "",
    invert_y: bool = False,
) -> Any:
    """Draw a ``(C, T)`` heatmap that shares the pipeline-wide time axis.

    Args:
        ax: Target matplotlib axes.
        data: Array of shape ``(C, T)``.
        t_max: X-axis extent in seconds.
        cmap: Colormap name.
        vmin: Minimum colour value (auto if None and not ``symmetric``).
        vmax: Maximum colour value (auto if None and not ``symmetric``).
        symmetric: If True, force ``vmin = -vmax`` from the data's maximum
            absolute value (useful for residual / signed panels).
        ylabel: Y-axis label.
        separator_row: Optional row index of a horizontal separator line
            (used to mark the scattering/phase split on feature rows).
        title: Panel title placed above the axes.

    Returns:
        The ``AxesImage`` handle, or ``None`` if the panel was empty/NaN.
    """
    if data.size == 0 or not np.isfinite(data).any():
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
        if title:
            ax.set_title(title, fontsize=FONT_LABEL)
        return None

    if symmetric:
        vlim = float(np.nanmax(np.abs(data))) + 1e-12
        vmin = -vlim
        vmax = vlim

    C = data.shape[0]
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        extent=(0.0, t_max, -0.5, C - 0.5),
    )
    if separator_row is not None and 0 <= separator_row < C - 1:
        ax.axhline(
            y=separator_row + 0.5,
            color=COLOR_BLACK,
            linewidth=0.5,
            linestyle="--",
        )
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    if title:
        ax.set_title(title, fontsize=FONT_LABEL)
    if invert_y:
        # Place row 0 at the top so the scattering order-0 coefficient
        # sits at the top of the panel; higher st rows go downward, and
        # the ph block sits below that after the separator line.
        ax.invert_yaxis()
    _add_colorbar(ax.figure, im, ax, label="")  # type: ignore[arg-type]
    return im


def plot_sample_lag_attn_diagnostic(
    sample: Dict[str, Any],
    out_path: Path,
    warmup: int,
    horizon: int,
    *,
    fhr_st_end: int = _FHR_ST_END,
    fs_raw: float = _DEFAULT_FS_RAW,
) -> None:
    """Render a multi-row diagnostic figure for one sample.

    Args:
        sample: Record from :func:`collect_predictions` with the keys
            ``mu_full, mu_base, delta_src, y_plus, z, attn, te_lag, kld_t,
            kld_per_dim, fhr, up, guid, epoch, label, metrics``.
        out_path: Destination PDF/PNG.
        warmup: Warmup anchors (used to shade invalid regions).
        horizon: Forecast horizon ``H_d``.
        fhr_st_end: Channel index separating scattering from phase
            (default 43 for the standard v1 config).
        fs_raw: Raw sampling rate in Hz (default 4.0).
    """
    mu_full = np.asarray(sample.get("mu_full"))            # (T, H_d, C)
    y_plus = np.asarray(sample.get("y_plus"))              # (T_valid, H_d, C)
    z = np.asarray(sample.get("z"))                        # (T, d_z)
    attn = np.asarray(sample.get("attn"))                  # (T, M, L)
    te_lag = np.asarray(sample.get("te_lag"))              # (T, L)
    kld_per_dim = np.asarray(sample.get("kld_per_dim"))    # (T, d_z)
    fhr_raw = sample.get("fhr")
    up_raw = sample.get("up")

    if mu_full.ndim != 3:
        raise ValueError(
            f"sample['mu_full'] must be (T, H_d, C), got {mu_full.shape}"
        )

    T, H_d, C = mu_full.shape
    if int(horizon) != int(H_d):
        # Non-fatal — prefer the tensor shape but warn so callers notice.
        H_d = int(H_d)
    fhr_arr = np.asarray(fhr_raw) if fhr_raw is not None else None
    up_arr = np.asarray(up_raw) if up_raw is not None else None
    R = int(fhr_arr.shape[0]) if fhr_arr is not None and fhr_arr.ndim == 1 else _DEFAULT_R
    time_raw, _, t_max = _time_axes(T=T, R=R, fs_raw=fs_raw)

    # -------- Build per-row data --------
    mu_full_avg = _average_forecast_per_channel(
        mu_full, T=T, H_d=H_d, warmup=warmup
    )  # (T, C)

    # y_plus is shape (T_valid, H_d, C); extend to (T, H_d, C) with NaN
    # padding so we can reuse the same averaging helper for a fair visual
    # comparison.
    y_full = np.full((T, H_d, C), np.nan, dtype=np.float32)
    T_valid = y_plus.shape[0] if y_plus.ndim == 3 else 0
    if T_valid > 0:
        y_full[:T_valid] = y_plus.astype(np.float32)
    y_plus_avg = _average_forecast_per_channel(
        y_full, T=T, H_d=H_d, warmup=warmup
    )

    mu_full_img = mu_full_avg.T
    y_plus_img = y_plus_avg.T
    residual_img = mu_full_img - y_plus_img

    kld_per_dim_img = (
        kld_per_dim.T if kld_per_dim.ndim == 2 else np.zeros((1, T), dtype=np.float32)
    )
    z_img = z.T if z.ndim == 2 else np.zeros((1, T), dtype=np.float32)
    attn_mean = (
        attn.mean(axis=1).T if attn.ndim == 3 else np.zeros((1, T), dtype=np.float32)
    )  # (L, T)
    te_lag_img = (
        te_lag.T if te_lag.ndim == 2 else np.zeros((1, T), dtype=np.float32)
    )  # (L, T)

    # -------- Layout --------
    # Taller per-row height (1.55" vs. the previous 1.15") removes the
    # empty space at the top of the saved PDF and gives each imshow panel
    # enough room to breathe.
    n_rows = 8
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(8.6, 1.55 * n_rows + 1.2),
        gridspec_kw={"hspace": 0.5},
        sharex=True,
    )

    # Warmup mask — hides start-of-sample anomalies from imshow auto
    # scaling on the model-output rows. Raw inputs (row 0) are left
    # unmasked on purpose and only get a vertical warmup-boundary line.
    warm = max(0, int(warmup))

    # Row 0: Raw FHR / UP traces (if present).
    # Raw signals are real model inputs — draw the full trace without
    # masking the warmup region and mark the warmup boundary with a
    # vertical line instead.
    ax = axes[0]
    drawn = False
    if fhr_arr is not None and fhr_arr.ndim == 1:
        n = min(len(fhr_arr), len(time_raw))
        ax.plot(
            time_raw[:n], fhr_arr[:n],
            color=COLOR_BLUE, linewidth=0.7, label="FHR",
        )
        drawn = True
    if up_arr is not None and up_arr.ndim == 1:
        ax2 = ax.twinx()
        n = min(len(up_arr), len(time_raw))
        ax2.plot(
            time_raw[:n], up_arr[:n],
            color=COLOR_VERMILLION, linewidth=0.7, alpha=0.8, label="UP",
        )
        ax2.tick_params(axis="y", colors=COLOR_VERMILLION, labelsize=6)
        drawn = True
    if not drawn:
        ax.text(
            0.5, 0.5, "raw traces not available",
            ha="center", va="center", transform=ax.transAxes,
        )
    ax.set_ylabel("raw FHR / UP", fontsize=FONT_LABEL)
    ax.set_xlim(0.0, t_max)
    _style_axes(ax, grid="major", minor_ticks=False)

    # NaN-mask the warmup region on every (rows, T) imshow so start-of-
    # sample anomalies do not compress the colour scale of the rest of
    # the plot.
    mu_full_img_m = _mask_warmup_time_axis(mu_full_img, warm, axis=-1)
    y_plus_img_m = _mask_warmup_time_axis(y_plus_img, warm, axis=-1)
    residual_img_m = _mask_warmup_time_axis(residual_img, warm, axis=-1)
    z_img_m = _mask_warmup_time_axis(z_img, warm, axis=-1)
    kld_per_dim_img_m = _mask_warmup_time_axis(kld_per_dim_img, warm, axis=-1)
    attn_mean_m = _mask_warmup_time_axis(attn_mean, warm, axis=-1)
    te_lag_img_m = _mask_warmup_time_axis(te_lag_img, warm, axis=-1)

    # Row 1: mu_full_avg (C, T). Channels are laid out as
    # ``fhr_st[0..fhr_st_end-1] ‖ fhr_ph[fhr_st_end..C-1]``; ``invert_y``
    # puts st order-0 on top, higher st orders below it, then ph under
    # the dashed separator.
    _imshow_panel(
        axes[1], mu_full_img_m, t_max=t_max, cmap="RdBu_r", symmetric=True,
        ylabel="mu_full_avg", separator_row=fhr_st_end - 1,
        title="Average feature forecast (mu_full)",
        invert_y=True,
    )

    # Row 2: y_plus_avg (C, T).
    _imshow_panel(
        axes[2], y_plus_img_m, t_max=t_max, cmap="RdBu_r", symmetric=True,
        ylabel="y_plus_avg", separator_row=fhr_st_end - 1,
        title="Ground truth (y_plus)",
        invert_y=True,
    )

    # Row 3: residual (C, T).
    _imshow_panel(
        axes[3], residual_img_m, t_max=t_max, cmap="RdBu_r", symmetric=True,
        ylabel="residual", separator_row=fhr_st_end - 1,
        title="mu_full - y_plus",
        invert_y=True,
    )

    # Row 4: latent z.
    _imshow_panel(
        axes[4], z_img_m, t_max=t_max, cmap="RdBu_r", symmetric=True,
        ylabel="z", title="Latent z",
    )

    # Row 5: KLD per dim.
    _imshow_panel(
        axes[5], kld_per_dim_img_m, t_max=t_max, cmap="magma", vmin=0.0,
        ylabel="KL per dim", title="KL(q||p) per latent dim",
    )

    # Row 6: Lag attention (L, T).
    _imshow_panel(
        axes[6], attn_mean_m, t_max=t_max, cmap="viridis",
        ylabel="lag k", title="Head-averaged lag attention",
    )

    # Row 7: TE lag attribution.
    _imshow_panel(
        axes[7], te_lag_img_m, t_max=t_max, cmap="inferno", vmin=0.0,
        ylabel="lag k", title="TE lag attribution",
    )

    # Row 0 shows real, unmasked input signals — mark the warmup boundary
    # with a vertical line rather than shading the warmup region. Rows
    # 1..7 show model-generated outputs whose warmup region is NaN-masked,
    # so they keep the full shaded band for context.
    _mark_warmup_line_sec(axes[0], warmup, t_max, T)
    for ax in axes[1:]:
        _shade_warmup(ax, warmup, t_max, T, color=COLOR_LIGHT_GRAY)
    axes[-1].set_xlabel("Time (s)", fontsize=FONT_LABEL)

    # Title bar with sample metadata.
    guid = sample.get("guid", "unknown")
    epoch = sample.get("epoch")
    label = sample.get("label")
    metrics = sample.get("metrics", {}) or {}
    feat_mse = metrics.get("feat_mse_total")
    uplift_rel = metrics.get("uplift_rel")
    res_ratio = metrics.get("residual_ratio")

    title_bits = [f"guid={guid}"]
    if epoch is not None:
        title_bits.append(f"epoch={float(epoch):.0f}s")
    if label is not None:
        title_bits.append(f"class={label}")
    if feat_mse is not None:
        title_bits.append(f"feat_mse={float(feat_mse):.4f}")
    if uplift_rel is not None:
        title_bits.append(f"uplift_rel={float(uplift_rel):.3f}")
    if res_ratio is not None:
        title_bits.append(f"resid_ratio={float(res_ratio):.3f}")

    _auto_suptitle(fig, "  |  ".join(title_bits), base_fontsize=FONT_TITLE)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# KLD + lag attention diagnostic figures (minute-axis)
# -----------------------------------------------------------------------------


def _draw_heatmap_direct(
    ax: plt.Axes,
    data: np.ndarray,
    *,
    t_max_min: float,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    norm: Optional[Any] = None,
    ylabel: str = "",
    title: str = "",
    y_extent: Optional[float] = None,
    cbar_ax: Optional[plt.Axes] = None,
    cbar_label: str = "",
    separator_row: Optional[float] = None,
    invert_y: bool = False,
) -> Any:
    """Draw a ``(rows, T)`` heatmap and route the colorbar to ``cbar_ax``.

    Unlike :func:`_draw_heatmap_min`, this does NOT steal horizontal width
    from ``ax`` via ``make_axes_locatable`` — the colorbar lives in a
    pre-allocated axes so the main plot keeps its full width and stays
    aligned with adjacent line panels.

    Args:
        ax: Target axes (the main plot area).
        data: Array of shape ``(rows, T)``.
        t_max_min: Full x-axis extent in minutes.
        cmap: Colormap name.
        vmin, vmax: Colour-scale limits (auto when ``None``).
        ylabel: Y-axis label.
        title: Panel title.
        y_extent: Optional y-axis top (rows space). Defaults to
            ``rows - 0.5`` which gives a pixel-per-row layout.
        cbar_ax: Dedicated axes for the colorbar. When ``None`` the
            colorbar is skipped entirely.
        cbar_label: Label placed alongside the colorbar.
        separator_row: Optional horizontal line position (data y-units)
            drawn on top of the heatmap in black. Used to mark the
            scattering / phase-harmonic boundary on combined panels.

    Returns:
        The ``AxesImage`` handle, or ``None`` if the panel was empty.
    """
    if data.size == 0 or not np.isfinite(data).any():
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
        if title:
            ax.set_title(title, fontsize=FONT_LABEL)
        return None

    rows = int(data.shape[0])
    y_top = float(y_extent) if y_extent is not None else (rows - 0.5)
    y_bot = 0.0 if y_extent is not None else -0.5

    # When an explicit ``norm`` (e.g. LogNorm) is supplied it overrides
    # ``vmin`` / ``vmax``; matplotlib raises if both are passed.
    imshow_kwargs: Dict[str, Any] = dict(
        aspect="auto",
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        extent=(0.0, t_max_min, y_bot, y_top),
    )
    if norm is not None:
        imshow_kwargs["norm"] = norm
    else:
        imshow_kwargs["vmin"] = vmin
        imshow_kwargs["vmax"] = vmax
    im = ax.imshow(data, **imshow_kwargs)
    if separator_row is not None:
        ax.axhline(
            y=float(separator_row),
            color=COLOR_BLACK,
            linewidth=1.0,
            linestyle="-",
            alpha=1.0,
            zorder=5,
        )
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    if title:
        ax.set_title(title, fontsize=FONT_LABEL, fontweight="normal")
    if invert_y:
        # Place row 0 at the top — used by st/ph feature panels so the
        # scattering order-0 coefficient is on top, higher st orders go
        # downward, and the ph block sits below the separator line.
        ax.invert_yaxis()
    if cbar_ax is not None:
        cb = ax.figure.colorbar(im, cax=cbar_ax)  # type: ignore[union-attr]
        cb.ax.tick_params(labelsize=6)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=FONT_LABEL)
    return im


def _combine_st_ph(
    st: Optional[np.ndarray], ph: Optional[np.ndarray]
) -> tuple[np.ndarray, Optional[float]]:
    """Concatenate scattering and phase-harmonic channels along the channel axis.

    Input arrays are ``(T, C_st)`` and ``(T, C_ph)``; the returned image is
    transposed to ``(C_st + C_ph, T)`` for imshow with ``origin="lower"``.
    The separator row sits between the two blocks at ``C_st - 0.5``.

    Args:
        st: Scattering features ``(T, C_st)`` or ``None``.
        ph: Phase-harmonic features ``(T, C_ph)`` or ``None``.

    Returns:
        Tuple of ``(combined_image, separator_row)``. When ``ph`` is
        ``None`` the scattering array is returned unchanged and the
        separator row is ``None``. Callers must gate on ``st`` not being
        ``None`` before invoking this helper.
    """
    if st is None:
        raise ValueError("_combine_st_ph requires a non-None scattering array.")
    if ph is None or ph.ndim != 2:
        return st.T, None
    combined = np.concatenate([st, ph], axis=1)  # (T, C_st + C_ph)
    separator_row = float(st.shape[1]) - 0.5
    return combined.T, separator_row


def plot_sample_signals_kld(
    *,
    fhr: Optional[np.ndarray],
    up: Optional[np.ndarray],
    fhr_st: np.ndarray,
    up_st: Optional[np.ndarray],
    kld_per_dim: np.ndarray,
    warmup: int,
    out_path: Path,
    fhr_ph: Optional[np.ndarray] = None,
    up_ph: Optional[np.ndarray] = None,
    guid: Optional[str] = None,
    epoch: Optional[float] = None,
    label: Optional[int] = None,
    fs_raw: float = _DEFAULT_FS_RAW,
    decim: int = _DEFAULT_DECIM,
) -> None:
    """Draw the multi-panel raw/scattering/KLD diagnostic for one sample.

    The figure uses a two-column GridSpec so that every main panel — line
    plots and heatmaps alike — occupies exactly the same width. Heatmap
    colorbars live in a dedicated narrow right-hand column, so they never
    steal horizontal width from the main plot and the column-one axes
    stay aligned top-to-bottom.

    The figure stacks, top to bottom:

    1. Raw FHR and UP traces on a twin y-axis (if present).
    2. FHR features — ``fhr_st`` concatenated with ``fhr_ph`` (if
       provided), separated by a horizontal black line at the boundary.
    3. UP features — ``up_st`` concatenated with ``up_ph`` (if provided),
       with the same black separator convention.
    4. Per-latent-dimension KLD traces — one full-width subplot per
       latent dimension.
    5. Mean ± std of KLD across latent dimensions at every timestep.

    Every panel shares a single physical-time x-axis expressed in
    **minutes**, using ``sec_per_dec = decim / fs_raw`` to convert
    decimated step index to seconds and then to minutes. Warmup is
    shaded consistently across all rows, and the pre-warmup region is
    NaN-masked on every imshow / line so start-of-sample anomalies do
    not compress the shared scales.

    Args:
        fhr: Raw FHR trace ``(R,)`` or ``None``.
        up: Raw UP trace ``(R,)`` or ``None``.
        fhr_st: Normalised FHR scattering features ``(T, C_st)``.
        up_st: Normalised UP scattering features ``(T, C_st)`` or
            ``None``.
        kld_per_dim: Per-timestep per-dim KL, shape ``(T, d_z)``.
        warmup: Number of warmup decimated steps.
        out_path: Destination PDF/PNG.
        fhr_ph: Normalised FHR phase-harmonic features ``(T, C_ph)`` or
            ``None``. When provided, concatenated with ``fhr_st`` and a
            black horizontal line separates the two blocks.
        up_ph: Normalised UP phase-harmonic features ``(T, C_ph)`` or
            ``None``. Same semantics as ``fhr_ph``.
        guid: Sample GUID for the title bar.
        epoch: Sample epoch (seconds relative to delivery) for the title.
        label: Outcome class label for the title.
        fs_raw: Raw sampling rate in Hz (default 4.0).
        decim: Decimation factor mapping raw → decimated (default 16).
    """
    if fhr_st.ndim != 2:
        raise ValueError(
            f"fhr_st must be (T, C_st), got {fhr_st.shape}"
        )
    if kld_per_dim.ndim != 2:
        raise ValueError(
            f"kld_per_dim must be (T, d_z), got {kld_per_dim.shape}"
        )

    T = int(fhr_st.shape[0])
    d_z = int(kld_per_dim.shape[1])

    sec_per_dec = float(decim) / float(fs_raw)
    t_max_min = T * sec_per_dec / 60.0
    time_dec_min = (np.arange(T) + 0.5) * sec_per_dec / 60.0
    warmup_min = max(0, int(warmup)) * sec_per_dec / 60.0

    R = int(fhr.shape[0]) if (fhr is not None and fhr.ndim == 1) else int(T * decim)
    time_raw_min = np.arange(R) / float(fs_raw) / 60.0

    # --- Layout ---
    # Two-column GridSpec keeps every main panel the same width: column 0
    # holds the plot; column 1 is a narrow gutter used only by heatmap
    # colorbars. Line panels leave column 1 empty, so the whole figure
    # aligns vertically on the left edge and the right edge of the plot
    # area.
    use_up_st = up_st is not None and up_st.ndim == 2
    n_kld_rows = int(d_z)
    n_top_rows = 2 + (1 if use_up_st else 0)  # raw, fhr_*, optional up_*
    n_bot_rows = 1                              # mean ± std
    n_total = n_top_rows + n_kld_rows + n_bot_rows

    height_ratios = (
        [1.4] * n_top_rows
        + [0.45] * n_kld_rows
        + [1.25] * n_bot_rows
    )
    fig_h = 1.9 * n_top_rows + 0.7 * n_kld_rows + 2.0
    fig = plt.figure(figsize=(14, fig_h))
    gs = GridSpec(
        n_total, 2, figure=fig,
        hspace=0.7, wspace=0.015,
        height_ratios=height_ratios,
        width_ratios=[1.0, 0.018],
        left=0.07, right=0.96, top=0.965, bottom=0.035,
    )

    # Warmup mask in decimated steps: values before ``warmup`` are NaN so
    # start-of-sample anomalies do not compress the colour/value scales.
    warm_dec = max(0, int(warmup))

    # --- Row 0: Raw signals (line plot, column 0 only) ---
    row = 0
    ax_raw = fig.add_subplot(gs[row, 0])
    _draw_raw_panel(
        ax_raw,
        fhr=fhr, up=up,
        time_raw_min=time_raw_min,
        t_max_min=t_max_min,
        warmup_min=warmup_min,
        title="Raw FHR / UP",
    )
    row += 1

    # --- Row 1: FHR features (fhr_st ‖ fhr_ph) with black separator ---
    # Scattering / phase-harmonic features are real model inputs (derived
    # from raw FHR) — draw them unmasked and mark the end of the warmup
    # window with a vertical line. Diverging RdBu_r with symmetric limits
    # matches the z-scored feature semantics (blue < 0 < red) and is kept
    # consistent with the other feature and attention panels.
    fhr_img, fhr_sep = _combine_st_ph(fhr_st, fhr_ph)
    fhr_vmax = (
        float(np.nanmax(np.abs(fhr_img)))
        if np.isfinite(fhr_img).any() else 1.0
    )
    fhr_vmax = fhr_vmax if fhr_vmax > 0 else 1.0
    ax_fhr = fig.add_subplot(gs[row, 0])
    cax_fhr = fig.add_subplot(gs[row, 1])
    fhr_title = (
        "FHR features (fhr_st │ fhr_ph)" if fhr_ph is not None
        else "FHR scattering transform"
    )
    _draw_heatmap_direct(
        ax_fhr, fhr_img,
        t_max_min=t_max_min, cmap="bwr",
        vmin=-fhr_vmax, vmax=fhr_vmax,
        ylabel="channel",
        title=fhr_title,
        cbar_ax=cax_fhr,
        separator_row=fhr_sep,
        invert_y=True,
    )
    _mark_warmup_line_min(ax_fhr, warmup_min)
    row += 1

    # --- Row 2 (optional): UP features (up_st ‖ up_ph) ---
    # Real model inputs — same convention as the FHR features row.
    if use_up_st:
        up_img, up_sep = _combine_st_ph(up_st, up_ph)
        up_vmax = (
            float(np.nanmax(np.abs(up_img)))
            if np.isfinite(up_img).any() else 1.0
        )
        up_vmax = up_vmax if up_vmax > 0 else 1.0
        ax_up = fig.add_subplot(gs[row, 0])
        cax_up = fig.add_subplot(gs[row, 1])
        up_title = (
            "UP features (up_st │ up_ph)" if up_ph is not None
            else "UP scattering transform"
        )
        _draw_heatmap_direct(
            ax_up, up_img,
            t_max_min=t_max_min, cmap="bwr",
            vmin=-up_vmax, vmax=up_vmax,
            ylabel="channel",
            title=up_title,
            cbar_ax=cax_up,
            separator_row=up_sep,
            invert_y=True,
        )
        _mark_warmup_line_min(ax_up, warmup_min)
        row += 1

    # --- Per-dim KLD traces (one full-width subplot per dim) ---
    # Line panels live in column 0 only; column 1 remains empty so every
    # main axes keeps the same horizontal extent as the heatmap rows.
    kld_plot = _mask_warmup_time_axis(kld_per_dim, warm_dec, axis=0)
    kld_grid_start = row
    for d in range(d_z):
        ax_d = fig.add_subplot(gs[kld_grid_start + d, 0])
        vals = kld_plot[:, d]
        finite_vals = vals[np.isfinite(vals)]
        y_peak = float(np.nanmax(finite_vals)) if finite_vals.size else 0.0

        ax_d.plot(
            time_dec_min, vals,
            color=COLOR_PURPLE, linewidth=0.9, alpha=0.95,
        )
        ax_d.axhline(0.0, color=COLOR_GRAY, linewidth=0.3, linestyle=":")
        ax_d.set_xlim(0.0, t_max_min)
        if y_peak > 0.0:
            ax_d.set_ylim(
                min(0.0, float(np.nanmin(finite_vals)) * 1.05),
                y_peak * 1.1,
            )
        ax_d.set_ylabel(f"dim {d}", fontsize=FONT_LABEL)
        ax_d.tick_params(axis="both", labelsize=6)
        ax_d.grid(True, which="major", alpha=0.2, linewidth=0.3)
        _shade_warmup_min(ax_d, warmup_min)

        # Only the last per-dim row carries x-tick labels.
        if d != d_z - 1:
            ax_d.tick_params(labelbottom=False)
    row += n_kld_rows

    # --- Bottom: Mean ± std KLD over dimensions (column 0 only) ---
    # Warmup rows in ``kld_plot`` are intentionally NaN-masked, so
    # ``nanmean`` / ``nanstd`` would emit a ``RuntimeWarning: Mean of
    # empty slice`` for every fully-NaN timestep. Suppress that noise --
    # the resulting NaN values are exactly what we want plotted (the
    # warmup band is shaded separately by ``_shade_warmup_min`` below).
    ax_mean = fig.add_subplot(gs[row, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_kld = np.nanmean(kld_plot, axis=1)
        std_kld = np.nanstd(kld_plot, axis=1)
    ax_mean.fill_between(
        time_dec_min,
        mean_kld - std_kld,
        mean_kld + std_kld,
        color=COLOR_BLUE,
        alpha=0.22,
        label="±1 std",
    )
    ax_mean.plot(
        time_dec_min, mean_kld,
        color=COLOR_BLUE, linewidth=1.2, label="mean",
    )
    ax_mean.axhline(0.0, color=COLOR_GRAY, linewidth=0.4, linestyle=":")
    ax_mean.set_xlim(0.0, t_max_min)
    ax_mean.set_xlabel("Time (min)", fontsize=FONT_LABEL)
    ax_mean.set_ylabel("KLD (nats)", fontsize=FONT_LABEL)
    ax_mean.set_title(
        "Mean ± std KLD across latent dimensions",
        fontsize=FONT_LABEL, fontweight="normal",
    )
    ax_mean.legend(loc="upper right", fontsize=6, frameon=True)
    _shade_warmup_min(ax_mean, warmup_min)
    _style_axes(ax_mean, grid="major", minor_ticks=True)

    # --- Title bar ---
    title_bits = []
    if guid is not None:
        title_bits.append(f"guid={guid}")
    if epoch is not None:
        title_bits.append(f"epoch={float(epoch):.0f}s")
    if label is not None:
        title_bits.append(f"class={label}")
    if title_bits:
        _auto_suptitle(fig, "  |  ".join(title_bits), base_fontsize=FONT_TITLE)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_sample_signals_kld_pca(
    *,
    fhr: Optional[np.ndarray],
    up: Optional[np.ndarray],
    fhr_st: np.ndarray,
    up_st: Optional[np.ndarray],
    kld_pcs: np.ndarray,
    warmup: int,
    out_path: Path,
    fhr_ph: Optional[np.ndarray] = None,
    up_ph: Optional[np.ndarray] = None,
    explained_variance_ratio: Optional[np.ndarray] = None,
    guid: Optional[str] = None,
    epoch: Optional[float] = None,
    label: Optional[int] = None,
    fs_raw: float = _DEFAULT_FS_RAW,
    decim: int = _DEFAULT_DECIM,
) -> None:
    """PCA variant of :func:`plot_sample_signals_kld`.

    Identical layout to the per-dim KL diagnostic (raw signals row, FHR
    feature row, optional UP feature row, per-component KLD traces, and
    a final mean ± std panel aggregating across the PCA components).
    The only difference is the middle block: instead of one row per
    latent dimension (``d_z=24``), this figure draws **one row per
    retained PCA component** (typically the top 3) projected from the
    per-time per-dim KL by the PCA fit that ``collect_metrics`` wrote
    to ``<output>/pca_kld/``.

    Args:
        fhr: Raw FHR trace ``(R,)`` or ``None``.
        up: Raw UP trace ``(R,)`` or ``None``.
        fhr_st: Normalised FHR scattering features ``(T, C_st)``.
        up_st: Normalised UP scattering features ``(T, C_st)`` or ``None``.
        kld_pcs: Per-timestep top-k PCA scores, shape ``(T, k)`` where
            ``k`` is typically 3. Must come from projecting the sample's
            ``kld_per_dim_t`` through a PCA fitted across the full test
            set (via :func:`metrics.project_kld_per_dim`).
        warmup: Number of warmup decimated steps.
        out_path: Destination PDF / PNG.
        fhr_ph: Normalised FHR phase-harmonic features ``(T, C_ph)`` or
            ``None``. Same black-separator convention as
            :func:`plot_sample_signals_kld`.
        up_ph: Normalised UP phase-harmonic features ``(T, C_ph)`` or
            ``None``.
        explained_variance_ratio: Optional ``(k,)`` array — when
            supplied, each per-component y-label is annotated with the
            share of explained variance ("PC1 (54.3%)").
        guid: Sample GUID for the title bar.
        epoch: Sample epoch (seconds relative to delivery) for the title.
        label: Outcome class label for the title.
        fs_raw: Raw sampling rate in Hz (default 4.0).
        decim: Decimation factor (default 16).
    """
    if fhr_st.ndim != 2:
        raise ValueError(
            f"fhr_st must be (T, C_st), got {fhr_st.shape}"
        )
    if kld_pcs.ndim != 2:
        raise ValueError(
            f"kld_pcs must be (T, k), got {kld_pcs.shape}"
        )

    T = int(fhr_st.shape[0])
    n_pcs = int(kld_pcs.shape[1])
    if kld_pcs.shape[0] != T:
        raise ValueError(
            f"kld_pcs time axis ({kld_pcs.shape[0]}) must match "
            f"fhr_st time axis ({T})"
        )

    ev = (
        np.asarray(explained_variance_ratio, dtype=float)
        if explained_variance_ratio is not None
        else None
    )

    sec_per_dec = float(decim) / float(fs_raw)
    t_max_min = T * sec_per_dec / 60.0
    time_dec_min = (np.arange(T) + 0.5) * sec_per_dec / 60.0
    warmup_min = max(0, int(warmup)) * sec_per_dec / 60.0

    R = int(fhr.shape[0]) if (fhr is not None and fhr.ndim == 1) else int(T * decim)
    time_raw_min = np.arange(R) / float(fs_raw) / 60.0

    use_up_st = up_st is not None and up_st.ndim == 2
    n_pc_rows = int(n_pcs)
    n_top_rows = 2 + (1 if use_up_st else 0)
    n_bot_rows = 1
    n_total = n_top_rows + n_pc_rows + n_bot_rows

    # Give the PC trace rows noticeably more vertical space than the
    # per-dim variant — with only 3 rows we can afford taller panels.
    height_ratios = (
        [1.4] * n_top_rows
        + [0.95] * n_pc_rows
        + [1.25] * n_bot_rows
    )
    fig_h = 1.9 * n_top_rows + 1.3 * n_pc_rows + 2.0
    fig = plt.figure(figsize=(14, fig_h))
    gs = GridSpec(
        n_total, 2, figure=fig,
        hspace=0.7, wspace=0.015,
        height_ratios=height_ratios,
        width_ratios=[1.0, 0.018],
        left=0.07, right=0.96, top=0.965, bottom=0.035,
    )

    warm_dec = max(0, int(warmup))

    # --- Row 0: Raw signals ---
    row = 0
    ax_raw = fig.add_subplot(gs[row, 0])
    _draw_raw_panel(
        ax_raw,
        fhr=fhr, up=up,
        time_raw_min=time_raw_min,
        t_max_min=t_max_min,
        warmup_min=warmup_min,
        title="Raw FHR / UP",
    )
    row += 1

    # --- Row 1: FHR features ---
    fhr_img, fhr_sep = _combine_st_ph(fhr_st, fhr_ph)
    fhr_vmax = (
        float(np.nanmax(np.abs(fhr_img)))
        if np.isfinite(fhr_img).any() else 1.0
    )
    fhr_vmax = fhr_vmax if fhr_vmax > 0 else 1.0
    ax_fhr = fig.add_subplot(gs[row, 0])
    cax_fhr = fig.add_subplot(gs[row, 1])
    fhr_title = (
        "FHR features (fhr_st │ fhr_ph)" if fhr_ph is not None
        else "FHR scattering transform"
    )
    _draw_heatmap_direct(
        ax_fhr, fhr_img,
        t_max_min=t_max_min, cmap="bwr",
        vmin=-fhr_vmax, vmax=fhr_vmax,
        ylabel="channel",
        title=fhr_title,
        cbar_ax=cax_fhr,
        separator_row=fhr_sep,
        invert_y=True,
    )
    _mark_warmup_line_min(ax_fhr, warmup_min)
    row += 1

    # --- Row 2 (optional): UP features ---
    if use_up_st:
        up_img, up_sep = _combine_st_ph(up_st, up_ph)
        up_vmax = (
            float(np.nanmax(np.abs(up_img)))
            if np.isfinite(up_img).any() else 1.0
        )
        up_vmax = up_vmax if up_vmax > 0 else 1.0
        ax_up = fig.add_subplot(gs[row, 0])
        cax_up = fig.add_subplot(gs[row, 1])
        up_title = (
            "UP features (up_st │ up_ph)" if up_ph is not None
            else "UP scattering transform"
        )
        _draw_heatmap_direct(
            ax_up, up_img,
            t_max_min=t_max_min, cmap="bwr",
            vmin=-up_vmax, vmax=up_vmax,
            ylabel="channel",
            title=up_title,
            cbar_ax=cax_up,
            separator_row=up_sep,
            invert_y=True,
        )
        _mark_warmup_line_min(ax_up, warmup_min)
        row += 1

    # --- Per-PC KLD traces (one row per retained component) ---
    pc_plot = _mask_warmup_time_axis(kld_pcs, warm_dec, axis=0)
    pc_grid_start = row
    for k in range(n_pcs):
        ax_k = fig.add_subplot(gs[pc_grid_start + k, 0])
        vals = pc_plot[:, k]
        finite_vals = vals[np.isfinite(vals)]
        y_min = float(np.nanmin(finite_vals)) if finite_vals.size else -1.0
        y_max = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0

        ax_k.plot(
            time_dec_min, vals,
            color=COLOR_PURPLE, linewidth=1.0, alpha=0.95,
        )
        ax_k.axhline(0.0, color=COLOR_GRAY, linewidth=0.3, linestyle=":")
        ax_k.set_xlim(0.0, t_max_min)
        if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
            span = max(y_max - y_min, 1e-12)
            ax_k.set_ylim(y_min - 0.08 * span, y_max + 0.08 * span)

        if ev is not None and k < ev.size and np.isfinite(ev[k]):
            ylabel = f"PC{k + 1}  ({100.0 * float(ev[k]):.1f}%)"
        else:
            ylabel = f"PC{k + 1}"
        ax_k.set_ylabel(ylabel, fontsize=FONT_LABEL)
        ax_k.tick_params(axis="both", labelsize=7)
        ax_k.grid(True, which="major", alpha=0.2, linewidth=0.3)
        _shade_warmup_min(ax_k, warmup_min)
        if k != n_pcs - 1:
            ax_k.tick_params(labelbottom=False)
    row += n_pc_rows

    # --- Bottom: Mean ± std across retained components ---
    # Same warmup-NaN consideration as the per-dim KLD plot above.
    ax_mean = fig.add_subplot(gs[row, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_pc = np.nanmean(pc_plot, axis=1)
        std_pc = np.nanstd(pc_plot, axis=1)
    ax_mean.fill_between(
        time_dec_min,
        mean_pc - std_pc,
        mean_pc + std_pc,
        color=COLOR_BLUE,
        alpha=0.22,
        label="±1 std",
    )
    ax_mean.plot(
        time_dec_min, mean_pc,
        color=COLOR_BLUE, linewidth=1.2, label="mean",
    )
    ax_mean.axhline(0.0, color=COLOR_GRAY, linewidth=0.4, linestyle=":")
    ax_mean.set_xlim(0.0, t_max_min)
    ax_mean.set_xlabel("Time (min)", fontsize=FONT_LABEL)
    ax_mean.set_ylabel("KLD PCA score", fontsize=FONT_LABEL)
    pc_list = ", ".join(f"PC{i + 1}" for i in range(n_pcs))
    ax_mean.set_title(
        f"Mean ± std KLD across top {n_pcs} PCA components ({pc_list})",
        fontsize=FONT_LABEL, fontweight="normal",
    )
    ax_mean.legend(loc="upper right", fontsize=6, frameon=True)
    _shade_warmup_min(ax_mean, warmup_min)
    _style_axes(ax_mean, grid="major", minor_ticks=True)

    # --- Title bar ---
    title_bits = []
    if guid is not None:
        title_bits.append(f"guid={guid}")
    if epoch is not None:
        title_bits.append(f"epoch={float(epoch):.0f}s")
    if label is not None:
        title_bits.append(f"class={label}")
    title_bits.append(f"KLD PCA top-{n_pcs}")
    _auto_suptitle(fig, "  |  ".join(title_bits), base_fontsize=FONT_TITLE)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_sample_lag_attention(
    *,
    fhr: Optional[np.ndarray],
    up: Optional[np.ndarray],
    attn_weights: np.ndarray,
    te_lag_map: np.ndarray,
    warmup: int,
    out_path: Path,
    fhr_st: Optional[np.ndarray] = None,
    fhr_ph: Optional[np.ndarray] = None,
    up_st: Optional[np.ndarray] = None,
    up_ph: Optional[np.ndarray] = None,
    true_lag_tt: Optional[np.ndarray] = None,
    true_lag_band: Optional[np.ndarray] = None,
    horizon: int = 30,
    guid: Optional[str] = None,
    epoch: Optional[float] = None,
    label: Optional[int] = None,
    fs_raw: float = _DEFAULT_FS_RAW,
    decim: int = _DEFAULT_DECIM,
) -> None:
    r"""Draw the multi-panel lag-attention diagnostic for one sample.

    The figure stacks, top to bottom:

    1. Raw FHR and UP traces (context strip).
    2. FHR features — ``fhr_st`` concatenated with ``fhr_ph`` (if
       provided), separated by a solid black horizontal line. Skipped
       entirely when no FHR-feature arrays are passed.
    3. UP features — ``up_st`` concatenated with ``up_ph`` (if
       provided), with the same black-separator convention.
    4. Head-averaged lag attention ``α̅(t, k)`` as a ``(L, T)`` heatmap
       with the **y-axis expressed in minutes** — ``lag_minutes[k] =
       k × decim / fs_raw / 60`` (4-second steps by default).
    5. TE lag attribution ``kld_per_t × mean_heads(α)`` on the same
       time-in-minutes × lag-in-minutes grid (``inferno``, linear).
    6. Same TE lag attribution drawn with a ``seismic`` diverging
       colormap on a linear scale — extra contrast for mid-range values.
    7. Same TE lag attribution drawn with ``seismic`` on a log scale
       (``LogNorm``) — pops small but non-zero regions that the linear
       views compress into near-black/near-white.
    8. Attention analysis: argmax lag per anchor (blue, left y-axis,
       in minutes) and head-averaged Shannon entropy (orange, right
       y-axis, in nats). Both curves are warmup-masked.
    9. Lag analysis: time-averaged head-averaged attention distribution
       as a bar chart with the lag axis in minutes.

    Column-one panels and heatmap colorbars live in separate GridSpec
    columns, so every main panel keeps the same horizontal extent and
    the stack aligns vertically edge-to-edge.

    Args:
        fhr: Raw FHR trace ``(R,)`` or ``None``.
        up: Raw UP trace ``(R,)`` or ``None``.
        attn_weights: Attention probabilities ``(T, M, L)``.
        te_lag_map: TE lag attribution ``(T, L)``.
        warmup: Warmup decimated steps.
        out_path: Destination PDF/PNG.
        fhr_st: Normalised FHR scattering features ``(T, C_st)`` or
            ``None``. If ``None`` the FHR feature panel is skipped.
        fhr_ph: Normalised FHR phase-harmonic features ``(T, C_ph)`` or
            ``None``. When provided together with ``fhr_st`` the two
            blocks are concatenated on the channel axis and separated
            by a solid black horizontal line at ``C_st - 0.5``.
        up_st: Normalised UP scattering features ``(T, C_st)`` or
            ``None``. Same semantics as ``fhr_st``.
        up_ph: Normalised UP phase-harmonic features ``(T, C_ph)`` or
            ``None``. Same semantics as ``fhr_ph``.
        guid: Sample GUID for the title bar.
        epoch: Sample epoch (seconds relative to delivery) for the title.
        label: Outcome class label for the title.
        fs_raw: Raw sampling rate in Hz (default 4.0).
        decim: Decimation factor (default 16).
        true_lag_tt: Synthetic-data ground-truth source→target lag $d_t$ per
            time step ``(T,)`` in decimated steps, or ``None`` (real data /
            legacy caches). When given, the informative lag band
            $\{\max(0, d_t - H)\dots d_t - 1\}$ is shaded green on the
            attention / TE-lag heatmaps and its upper edge $d_t - 1$ is drawn
            as a dashed "true lag" curve, so the model's argmax lag can be
            checked against the truth at every $t$.
        true_lag_band: Dataset-level union lag band (sequence of lag indices)
            or ``None``; shaded on the time-averaged lag-mass bar chart.
        horizon: Forecast horizon $H$ (decimated steps), used to clamp the
            informative-band floor to $\max(0, d_t - H)$. Default 30.
    """
    if attn_weights.ndim != 3:
        raise ValueError(
            f"attn_weights must be (T, M, L), got {attn_weights.shape}"
        )
    if te_lag_map.ndim != 2:
        raise ValueError(
            f"te_lag_map must be (T, L), got {te_lag_map.shape}"
        )

    T = int(attn_weights.shape[0])
    L = int(attn_weights.shape[2])
    if te_lag_map.shape != (T, L):
        raise ValueError(
            f"te_lag_map shape {te_lag_map.shape} must match "
            f"(T={T}, L={L}) from attn_weights"
        )

    sec_per_dec = float(decim) / float(fs_raw)
    t_max_min = T * sec_per_dec / 60.0
    time_dec_min = (np.arange(T) + 0.5) * sec_per_dec / 60.0
    warmup_min = max(0, int(warmup)) * sec_per_dec / 60.0

    R = int(fhr.shape[0]) if (fhr is not None and fhr.ndim == 1) else int(T * decim)
    time_raw_min = np.arange(R) / float(fs_raw) / 60.0

    # Lag axis in minutes. Lag k represents an offset of
    # k × sec_per_dec seconds into the past; we plot k at its bin
    # centre in minutes.
    lag_edges_min = np.arange(L + 1) * sec_per_dec / 60.0  # (L+1,)
    lag_centers_min = (lag_edges_min[:-1] + lag_edges_min[1:]) / 2.0
    lag_span_min = float(lag_edges_min[-1])

    # Head-averaged attention.
    alpha_bar = attn_weights.mean(axis=1)   # (T, L)

    # Warmup mask over anchors.
    valid_mask = np.ones(T, dtype=bool)
    warm = min(max(0, int(warmup)), T)
    valid_mask[:warm] = False

    # Argmax lag per anchor (head-averaged).
    argmax_idx = alpha_bar.argmax(axis=1)                   # (T,)
    argmax_lag_min = argmax_idx.astype(float) * sec_per_dec / 60.0
    argmax_lag_min_masked = argmax_lag_min.copy()
    argmax_lag_min_masked[~valid_mask] = np.nan

    # Per-head Shannon entropy (nats).
    eps = 1e-12
    safe_attn = np.clip(attn_weights, eps, None)
    entropy_per_head = -(safe_attn * np.log(safe_attn)).sum(axis=-1)  # (T, M)
    mean_entropy = entropy_per_head.mean(axis=1)                      # (T,)
    mean_entropy_masked = mean_entropy.copy()
    mean_entropy_masked[~valid_mask] = np.nan

    # Time-averaged attention mass per lag (over valid anchors).
    if valid_mask.any():
        alpha_mass_by_lag = alpha_bar[valid_mask].mean(axis=0)        # (L,)
    else:
        alpha_mass_by_lag = np.zeros(L)

    # --- Ground-truth lag overlay (synthetic data only) ---
    # The informative band is {max(0, d_t - H) .. d_t - 1} in lag-index space
    # (matches ``_union_lag_band`` / lag_recovery); converted to minutes it is
    # shaded under the dashed band-edge curve d_t - 1.
    true_lag_edge_min: Optional[np.ndarray] = None
    band_lo_min: Optional[np.ndarray] = None
    band_hi_min: Optional[np.ndarray] = None
    if true_lag_tt is not None:
        d = np.asarray(true_lag_tt, dtype=float).reshape(-1)
        if d.shape[0] == T:
            band_lo_min = np.maximum(0.0, d - float(horizon)) * sec_per_dec / 60.0
            band_hi_min = np.maximum(0.0, d - 1.0) * sec_per_dec / 60.0
            true_lag_edge_min = band_hi_min

    def _overlay_true_lag(ax: Any, *, fill: bool) -> None:
        """Draw the true informative lag band / edge on a time×lag panel."""
        if true_lag_edge_min is None:
            return
        if fill:
            ax.fill_between(
                time_dec_min, band_lo_min, band_hi_min,
                color=COLOR_GREEN, alpha=0.13, linewidth=0.0, zorder=1,
            )
        ax.plot(
            time_dec_min, true_lag_edge_min,
            color=COLOR_GREEN, linewidth=1.0, linestyle="--",
            alpha=0.95, zorder=4, label="true lag",
        )

    # --- Figure layout ---
    # Two-column GridSpec: column 0 holds every main panel, column 1 is a
    # narrow gutter that only heatmap rows use for their colorbars. Line
    # rows leave column 1 empty so every main axes keeps the same
    # horizontal extent and the stack aligns vertically.
    has_fhr_feats = fhr_st is not None and fhr_st.ndim == 2
    has_up_feats = up_st is not None and up_st.ndim == 2

    # Row order: raw, [fhr_feats], [up_feats], attn, te_lag (inferno),
    # te_lag (seismic), te_lag (seismic+log), argmax+ent, lag mass.
    # The two extra TE panels give diverging- and log-scaled views of the
    # same ``te_lag_map`` so small but non-zero regions become visible.
    row_heights: List[float] = [1.0]
    if has_fhr_feats:
        row_heights.append(1.6)
    if has_up_feats:
        row_heights.append(1.6)
    row_heights.extend([1.9, 1.9, 1.9, 1.9, 1.1, 1.2])
    n_rows = len(row_heights)
    # Figure height scales with the total of the row ratios so adding the
    # FHR/UP feature rows doesn't squash the rest of the stack. The base
    # constants reproduce the original 14" height when only the 5 original
    # rows are present (sum = 7.1 → fig_h ≈ 14).
    fig_h = 1.7 * sum(row_heights) + 2.0
    fig = plt.figure(figsize=(12, fig_h))
    gs = GridSpec(
        n_rows, 2, figure=fig,
        hspace=0.7, wspace=0.015,
        height_ratios=row_heights,
        width_ratios=[1.0, 0.02],
        left=0.10, right=0.95, top=0.955, bottom=0.045,
    )

    # --- Row 0: Raw signals (column 0 only) ---
    row = 0
    ax_raw = fig.add_subplot(gs[row, 0])
    _draw_raw_panel(
        ax_raw,
        fhr=fhr, up=up,
        time_raw_min=time_raw_min,
        t_max_min=t_max_min,
        warmup_min=warmup_min,
        title="Raw FHR / UP",
    )
    row += 1

    # --- Row ?: FHR features (fhr_st ‖ fhr_ph) with black separator ---
    # Real model inputs — draw unmasked and mark warmup with a line.
    # Normalised features are zero-centred (z-score) so a diverging
    # RdBu_r colormap with symmetric limits reads cleanly: blue = below
    # baseline, red = above.
    if has_fhr_feats:
        fhr_img, fhr_sep = _combine_st_ph(fhr_st, fhr_ph)  # type: ignore[arg-type]
        fhr_vmax = (
            float(np.nanmax(np.abs(fhr_img)))
            if np.isfinite(fhr_img).any() else 1.0
        )
        fhr_vmax = fhr_vmax if fhr_vmax > 0 else 1.0
        ax_fhr = fig.add_subplot(gs[row, 0])
        cax_fhr = fig.add_subplot(gs[row, 1])
        fhr_title = (
            "FHR features (fhr_st │ fhr_ph)" if fhr_ph is not None
            else "FHR scattering transform"
        )
        _draw_heatmap_direct(
            ax_fhr, fhr_img,
            t_max_min=t_max_min, cmap="bwr",
            vmin=-fhr_vmax, vmax=fhr_vmax,
            ylabel="channel",
            title=fhr_title,
            cbar_ax=cax_fhr,
            separator_row=fhr_sep,
            invert_y=True,
        )
        _mark_warmup_line_min(ax_fhr, warmup_min)
        row += 1

    # --- Row ?: UP features (up_st ‖ up_ph) with black separator ---
    # Real model inputs — draw unmasked and mark warmup with a line.
    if has_up_feats:
        up_img, up_sep = _combine_st_ph(up_st, up_ph)  # type: ignore[arg-type]
        up_vmax = (
            float(np.nanmax(np.abs(up_img)))
            if np.isfinite(up_img).any() else 1.0
        )
        up_vmax = up_vmax if up_vmax > 0 else 1.0
        ax_up = fig.add_subplot(gs[row, 0])
        cax_up = fig.add_subplot(gs[row, 1])
        up_title = (
            "UP features (up_st │ up_ph)" if up_ph is not None
            else "UP scattering transform"
        )
        _draw_heatmap_direct(
            ax_up, up_img,
            t_max_min=t_max_min, cmap="bwr",
            vmin=-up_vmax, vmax=up_vmax,
            ylabel="channel",
            title=up_title,
            cbar_ax=cax_up,
            separator_row=up_sep,
            invert_y=True,
        )
        _mark_warmup_line_min(ax_up, warmup_min)
        row += 1

    # Warmup-masked copies: the (L, T) heatmaps have time on axis=-1, so
    # masking leading columns hides the warmup region from both imshow
    # and auto vmin/vmax scaling.
    alpha_bar_img = _mask_warmup_time_axis(alpha_bar.T, warm, axis=-1)
    te_lag_img = _mask_warmup_time_axis(te_lag_map.T, warm, axis=-1)

    # --- Row ?: Attention matrix head-averaged ---
    # RdBu_r on [0, max]: the mean attention level (~1/L) ends up mid-
    # grey, below-average lags tint blue, above-average lags tint red —
    # makes "where the model is looking" pop visually.
    attn_vmax = (
        float(np.nanmax(alpha_bar_img)) if np.isfinite(alpha_bar_img).any()
        else 1.0
    )
    attn_vmax = attn_vmax if attn_vmax > 0 else 1.0
    ax_attn = fig.add_subplot(gs[row, 0])
    cax_attn = fig.add_subplot(gs[row, 1])
    _draw_heatmap_direct(
        ax_attn, alpha_bar_img,  # (L, T)
        t_max_min=t_max_min,
        cmap="RdBu_r",
        vmin=0.0, vmax=attn_vmax,
        ylabel="Lag (min)",
        title=r"Head-averaged lag attention  $\bar{\alpha}(t, k)$",
        cbar_ax=cax_attn,
        cbar_label="attention",
        y_extent=lag_span_min,
    )
    ax_attn.set_ylim(0.0, lag_span_min)
    ax_attn.set_xlabel("Time (min)", fontsize=FONT_LABEL)
    _shade_warmup_min(ax_attn, warmup_min)
    # Overlay the true informative-lag band (synthetic only) under the model's
    # argmax-lag curve (both in minutes), so "did attention pick the right lag
    # at each t?" is answered by eye.
    _overlay_true_lag(ax_attn, fill=True)
    if valid_mask.any():
        ax_attn.plot(
            time_dec_min[valid_mask],
            argmax_lag_min_masked[valid_mask],
            color=COLOR_VERMILLION,
            linewidth=0.9,
            alpha=0.9,
            label="argmax lag",
        )
    if valid_mask.any() or true_lag_edge_min is not None:
        ax_attn.legend(loc="upper right", fontsize=6, frameon=True)
    row += 1

    # --- Row ?: TE lag attribution (inferno, linear) ---
    # Peak used by the three TE panels below so linear/seismic/log views
    # share a common upper scale and are comparable side-by-side.
    te_vmax = (
        float(np.nanmax(te_lag_img)) if np.isfinite(te_lag_img).any() else 1.0
    )
    te_vmax = te_vmax if te_vmax > 0 else 1.0

    ax_te = fig.add_subplot(gs[row, 0])
    cax_te = fig.add_subplot(gs[row, 1])
    _draw_heatmap_direct(
        ax_te, te_lag_img,
        t_max_min=t_max_min,
        cmap="inferno",
        vmin=0.0, vmax=te_vmax,
        ylabel="Lag (min)",
        title=r"TE lag attribution  $\mathrm{KL}_t \cdot \bar{\alpha}(t, k)$",
        cbar_ax=cax_te,
        cbar_label="TE mass",
        y_extent=lag_span_min,
    )
    ax_te.set_ylim(0.0, lag_span_min)
    ax_te.set_xlabel("Time (min)", fontsize=FONT_LABEL)
    _shade_warmup_min(ax_te, warmup_min)
    _overlay_true_lag(ax_te, fill=False)
    if true_lag_edge_min is not None:
        ax_te.legend(loc="upper right", fontsize=6, frameon=True)
    row += 1

    # --- Row ?: TE lag attribution (seismic, linear) ---
    # Diverging colormap on [0, te_vmax]: low values map to the cool
    # (blue) half, mid values to near-white, and the peaks to saturated
    # red. Useful when the inferno view is dominated by the top 10% and
    # smaller regions need extra contrast.
    ax_te_seis = fig.add_subplot(gs[row, 0])
    cax_te_seis = fig.add_subplot(gs[row, 1])
    _draw_heatmap_direct(
        ax_te_seis, te_lag_img,
        t_max_min=t_max_min,
        cmap="seismic",
        vmin=0.0, vmax=te_vmax,
        ylabel="Lag (min)",
        title=r"TE lag attribution (seismic, linear)",
        cbar_ax=cax_te_seis,
        cbar_label="TE mass",
        y_extent=lag_span_min,
    )
    ax_te_seis.set_ylim(0.0, lag_span_min)
    ax_te_seis.set_xlabel("Time (min)", fontsize=FONT_LABEL)
    _shade_warmup_min(ax_te_seis, warmup_min)
    _overlay_true_lag(ax_te_seis, fill=False)
    row += 1

    # --- Row ?: TE lag attribution (seismic, log) ---
    # ``LogNorm`` requires a strictly-positive ``vmin``; we clamp to either
    # the smallest positive observed value or ``te_vmax * 1e-3`` so two
    # orders of magnitude of dynamic range always fit on the colorbar.
    # Non-positive / NaN pixels stay transparent in imshow (LogNorm maps
    # them under vmin → bottom-of-colormap, which is blue for seismic).
    finite_positive = te_lag_img[np.isfinite(te_lag_img) & (te_lag_img > 0)]
    if finite_positive.size > 0:
        log_vmin = max(float(finite_positive.min()), te_vmax * 1e-3)
    else:
        log_vmin = te_vmax * 1e-3
    ax_te_log = fig.add_subplot(gs[row, 0])
    cax_te_log = fig.add_subplot(gs[row, 1])
    _draw_heatmap_direct(
        ax_te_log, te_lag_img,
        t_max_min=t_max_min,
        cmap="seismic",
        norm=LogNorm(vmin=log_vmin, vmax=te_vmax),
        ylabel="Lag (min)",
        title=r"TE lag attribution (seismic, log)",
        cbar_ax=cax_te_log,
        cbar_label="TE mass (log)",
        y_extent=lag_span_min,
    )
    ax_te_log.set_ylim(0.0, lag_span_min)
    ax_te_log.set_xlabel("Time (min)", fontsize=FONT_LABEL)
    _shade_warmup_min(ax_te_log, warmup_min)
    _overlay_true_lag(ax_te_log, fill=False)
    row += 1

    # --- Row ?: Attention analysis — argmax lag + entropy over time ---
    ax_ana = fig.add_subplot(gs[row, 0])
    ax_ana.plot(
        time_dec_min, argmax_lag_min_masked,
        color=COLOR_BLUE, linewidth=1.0, label="argmax lag",
    )
    # Ground-truth lag (synthetic only): the direct per-t comparison against
    # the model's argmax.
    if true_lag_edge_min is not None:
        ax_ana.plot(
            time_dec_min, true_lag_edge_min,
            color=COLOR_GREEN, linewidth=1.0, linestyle="--",
            alpha=0.95, label="true lag",
        )
        ax_ana.legend(loc="upper right", fontsize=6, frameon=True)
    ax_ana.set_xlim(0.0, t_max_min)
    ax_ana.set_ylim(0.0, lag_span_min * 1.05)
    ax_ana.set_xlabel("Time (min)", fontsize=FONT_LABEL)
    ax_ana.set_ylabel("Argmax lag (min)", fontsize=FONT_LABEL, color=COLOR_BLUE)
    ax_ana.tick_params(axis="y", colors=COLOR_BLUE, labelsize=6)
    ax_ana.set_title(
        "Attention analysis: argmax lag & head-averaged entropy",
        fontsize=FONT_LABEL, fontweight="normal",
    )

    ax_ent = ax_ana.twinx()
    ax_ent.plot(
        time_dec_min, mean_entropy_masked,
        color=COLOR_VERMILLION, linewidth=0.9, alpha=0.85, label="entropy",
    )
    ax_ent.set_ylabel("Entropy (nats)", fontsize=FONT_LABEL, color=COLOR_VERMILLION)
    ax_ent.tick_params(axis="y", colors=COLOR_VERMILLION, labelsize=6)

    _shade_warmup_min(ax_ana, warmup_min)
    _style_axes(ax_ana, grid="major", minor_ticks=True)
    row += 1

    # --- Row ?: Lag analysis — time-averaged attention mass per lag ---
    ax_lag = fig.add_subplot(gs[row, 0])
    bin_width = float(sec_per_dec / 60.0)
    ax_lag.bar(
        lag_centers_min,
        alpha_mass_by_lag,
        width=bin_width * 0.95,
        color=COLOR_BLUE,
        alpha=0.85,
        edgecolor=COLOR_BLACK,
        linewidth=0.3,
    )
    # Shade the dataset-level true informative band (synthetic only) so the
    # attention mass can be read against where the transfer actually lives.
    if true_lag_band is not None and len(true_lag_band) > 0:
        tb = np.asarray(true_lag_band, dtype=float)
        ax_lag.axvspan(
            float(tb.min()) * sec_per_dec / 60.0,
            (float(tb.max()) + 1.0) * sec_per_dec / 60.0,
            color=COLOR_GREEN, alpha=0.13, linewidth=0.0, label="true band",
        )
    if alpha_mass_by_lag.size and np.isfinite(alpha_mass_by_lag).any():
        peak_idx = int(np.nanargmax(alpha_mass_by_lag))
        peak_lag = float(lag_centers_min[peak_idx])
        ax_lag.axvline(
            peak_lag,
            color=COLOR_VERMILLION,
            linewidth=0.8,
            linestyle="--",
            label=f"peak={peak_lag:.2f} min",
        )
    if (true_lag_band is not None and len(true_lag_band) > 0) or (
        alpha_mass_by_lag.size and np.isfinite(alpha_mass_by_lag).any()
    ):
        ax_lag.legend(loc="upper right", fontsize=6, frameon=True)
    ax_lag.set_xlim(0.0, lag_span_min)
    ax_lag.set_xlabel("Lag (min)", fontsize=FONT_LABEL)
    ax_lag.set_ylabel(r"Time-averaged $\bar{\alpha}(k)$", fontsize=FONT_LABEL)
    ax_lag.set_title(
        "Lag analysis: time-averaged attention mass per lag",
        fontsize=FONT_LABEL, fontweight="normal",
    )
    _style_axes(ax_lag, grid="major", minor_ticks=True)

    # --- Title bar ---
    title_bits = []
    if guid is not None:
        title_bits.append(f"guid={guid}")
    if epoch is not None:
        title_bits.append(f"epoch={float(epoch):.0f}s")
    if label is not None:
        title_bits.append(f"class={label}")
    if title_bits:
        _auto_suptitle(fig, "  |  ".join(title_bits), base_fontsize=FONT_TITLE)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "plot_sample_lag_attn_diagnostic",
    "plot_sample_signals_kld",
    "plot_sample_lag_attention",
]
