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

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
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


def _mask_warmup_signal(
    signal: Optional[np.ndarray],
    time_min: np.ndarray,
    warmup_min: float,
) -> Optional[np.ndarray]:
    """NaN-mask a 1-D signal where ``time_min < warmup_min``.

    Args:
        signal: 1-D signal (e.g., raw FHR / UP trace) or ``None``.
        time_min: Matching time axis in minutes.
        warmup_min: Warmup duration in minutes.

    Returns:
        A float copy of ``signal`` with samples in the warmup region set
        to NaN. Returns ``signal`` unchanged if it is ``None`` or the
        warmup window is empty.
    """
    if signal is None or warmup_min is None or warmup_min <= 0:
        return signal
    n = min(len(signal), len(time_min))
    out = np.asarray(signal, dtype=np.float32).copy()
    out = out[:n]
    mask = time_min[:n] < float(warmup_min)
    out[mask] = np.nan
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
    drawn = False
    if fhr is not None and fhr.ndim == 1:
        n = min(len(fhr), len(time_raw_min))
        fhr_plot = _mask_warmup_signal(fhr[:n], time_raw_min[:n], warmup_min)
        ax.plot(
            time_raw_min[:n], fhr_plot,
            color=COLOR_BLUE, linewidth=0.7, label="FHR",
        )
        ax.set_ylabel("FHR (bpm)", fontsize=FONT_LABEL, color=COLOR_BLUE)
        ax.tick_params(axis="y", colors=COLOR_BLUE, labelsize=6)
        drawn = True
    if up is not None and up.ndim == 1:
        ax_up = ax.twinx()
        n = min(len(up), len(time_raw_min))
        up_plot = _mask_warmup_signal(up[:n], time_raw_min[:n], warmup_min)
        ax_up.plot(
            time_raw_min[:n], up_plot,
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
    _shade_warmup_min(ax, warmup_min)
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
    # scaling and from the raw-signal y-limits. ``time_raw`` is seconds
    # here (not minutes), so the warmup threshold is also in seconds.
    warm = max(0, int(warmup))
    sec_per_dec = float(_DEFAULT_DECIM) / float(fs_raw)
    warmup_s = warm * sec_per_dec

    # Row 0: Raw FHR / UP traces (if present).
    ax = axes[0]
    drawn = False
    if fhr_arr is not None and fhr_arr.ndim == 1:
        n = min(len(fhr_arr), len(time_raw))
        fhr_plot = np.asarray(fhr_arr[:n], dtype=np.float32).copy()
        if warmup_s > 0:
            fhr_plot[time_raw[:n] < warmup_s] = np.nan
        ax.plot(
            time_raw[:n], fhr_plot,
            color=COLOR_BLUE, linewidth=0.7, label="FHR",
        )
        drawn = True
    if up_arr is not None and up_arr.ndim == 1:
        ax2 = ax.twinx()
        n = min(len(up_arr), len(time_raw))
        up_plot = np.asarray(up_arr[:n], dtype=np.float32).copy()
        if warmup_s > 0:
            up_plot[time_raw[:n] < warmup_s] = np.nan
        ax2.plot(
            time_raw[:n], up_plot,
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

    # Row 1: mu_full_avg (C, T).
    _imshow_panel(
        axes[1], mu_full_img_m, t_max=t_max, cmap="RdBu_r", symmetric=True,
        ylabel="mu_full_avg", separator_row=fhr_st_end - 1,
        title="Average feature forecast (mu_full)",
    )

    # Row 2: y_plus_avg (C, T).
    _imshow_panel(
        axes[2], y_plus_img_m, t_max=t_max, cmap="RdBu_r", symmetric=True,
        ylabel="y_plus_avg", separator_row=fhr_st_end - 1,
        title="Ground truth (y_plus)",
    )

    # Row 3: residual (C, T).
    _imshow_panel(
        axes[3], residual_img_m, t_max=t_max, cmap="RdBu_r", symmetric=True,
        ylabel="residual", separator_row=fhr_st_end - 1,
        title="mu_full - y_plus",
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

    for ax in axes:
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


def plot_sample_signals_kld(
    *,
    fhr: Optional[np.ndarray],
    up: Optional[np.ndarray],
    fhr_st: np.ndarray,
    up_st: Optional[np.ndarray],
    kld_per_dim: np.ndarray,
    warmup: int,
    out_path: Path,
    guid: Optional[str] = None,
    epoch: Optional[float] = None,
    label: Optional[int] = None,
    fs_raw: float = _DEFAULT_FS_RAW,
    decim: int = _DEFAULT_DECIM,
) -> None:
    """Draw the multi-panel raw/scattering/KLD diagnostic for one sample.

    The figure stacks, top to bottom:

    1. Raw FHR and UP traces on a twin y-axis (if present).
    2. FHR scattering transform ``(43, T)`` heatmap.
    3. UP scattering transform ``(43, T)`` heatmap (skipped if absent).
    4. Per-latent-dimension KLD traces — one compact subplot per latent
       dimension, arranged in a grid (up to 6 columns wide).
    5. Mean ± std of KLD across latent dimensions at every timestep.

    Every panel shares a single physical-time x-axis expressed in
    **minutes**, using ``sec_per_dec = decim / fs_raw`` to convert
    decimated step index to seconds and then to minutes. Warmup is
    shaded consistently across all rows.

    Args:
        fhr: Raw FHR trace ``(R,)`` or ``None``.
        up: Raw UP trace ``(R,)`` or ``None``.
        fhr_st: Normalised FHR scattering features ``(T, 43)``.
        up_st: Normalised UP scattering features ``(T, 43)`` or ``None``.
        kld_per_dim: Per-timestep per-dim KL, shape ``(T, d_z)``.
        warmup: Number of warmup decimated steps.
        out_path: Destination PDF/PNG.
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
    # Per-dim KLD traces are stacked one-per-row at the same width as the
    # raw FHR / UP panel, so each latent dimension gets the full time
    # axis instead of being crammed into a narrow grid cell.
    use_up_st = up_st is not None and up_st.ndim == 2
    n_cols = 1
    n_kld_rows = int(d_z)
    n_top_rows = 2 + (1 if use_up_st else 0)  # raw, fhr_st, optional up_st
    n_bot_rows = 1                              # mean ± std
    n_total = n_top_rows + n_kld_rows + n_bot_rows

    height_ratios = (
        [1.1] * n_top_rows
        + [0.45] * n_kld_rows
        + [1.25] * n_bot_rows
    )
    fig_h = 1.6 * n_top_rows + 0.7 * n_kld_rows + 1.8
    fig = plt.figure(figsize=(14, fig_h))
    gs = GridSpec(
        n_total, n_cols, figure=fig,
        hspace=0.55, wspace=0.25,
        height_ratios=height_ratios,
        left=0.07, right=0.96, top=0.96, bottom=0.04,
    )

    # --- Row 0: Raw signals ---
    row = 0
    ax_raw = fig.add_subplot(gs[row, :])
    _draw_raw_panel(
        ax_raw,
        fhr=fhr, up=up,
        time_raw_min=time_raw_min,
        t_max_min=t_max_min,
        warmup_min=warmup_min,
        title="Raw FHR / UP",
    )
    row += 1

    # Warmup mask in decimated steps: values before ``warmup`` are NaN so
    # start-of-sample anomalies do not compress the colour/value scales.
    warm_dec = max(0, int(warmup))

    # --- Row 1: FHR scattering transform (channels on y, time on x) ---
    fhr_st_img = _mask_warmup_time_axis(fhr_st.T, warm_dec, axis=-1)
    ax_fhr = fig.add_subplot(gs[row, :])
    _draw_heatmap_min(
        ax_fhr, fhr_st_img,
        t_max_min=t_max_min, cmap="viridis",
        ylabel="fhr_st ch",
        title="FHR scattering transform",
        colorbar_label="",
    )
    _shade_warmup_min(ax_fhr, warmup_min)
    row += 1

    # --- Row 2 (optional): UP scattering transform ---
    if use_up_st:
        up_st_img = _mask_warmup_time_axis(up_st.T, warm_dec, axis=-1)  # type: ignore[union-attr]
        ax_up_st = fig.add_subplot(gs[row, :])
        _draw_heatmap_min(
            ax_up_st, up_st_img,
            t_max_min=t_max_min, cmap="viridis",
            ylabel="up_st ch",
            title="UP scattering transform",
            colorbar_label="",
        )
        _shade_warmup_min(ax_up_st, warmup_min)
        row += 1

    # --- Per-dim KLD traces (one full-width subplot per dim) ---
    # Warmup samples are NaN-masked so they are not plotted and do not
    # enter the per-dim auto y-limits.
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

    # --- Bottom: Mean ± std KLD over dimensions ---
    ax_mean = fig.add_subplot(gs[row, :])
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


def plot_sample_lag_attention(
    *,
    fhr: Optional[np.ndarray],
    up: Optional[np.ndarray],
    attn_weights: np.ndarray,
    te_lag_map: np.ndarray,
    warmup: int,
    out_path: Path,
    guid: Optional[str] = None,
    epoch: Optional[float] = None,
    label: Optional[int] = None,
    fs_raw: float = _DEFAULT_FS_RAW,
    decim: int = _DEFAULT_DECIM,
) -> None:
    """Draw the multi-panel lag-attention diagnostic for one sample.

    The figure stacks, top to bottom:

    1. Raw FHR and UP traces (context strip).
    2. Head-averaged lag attention ``α̅(t, k)`` as a ``(L, T)`` heatmap
       with the **y-axis expressed in minutes** — ``lag_minutes[k] =
       k × decim / fs_raw / 60`` (4-second steps by default).
    3. TE lag attribution ``kld_per_t × mean_heads(α)`` on the same
       time-in-minutes × lag-in-minutes grid.
    4. Attention analysis: argmax lag per anchor (blue, left y-axis,
       in minutes) and head-averaged Shannon entropy (orange, right
       y-axis, in nats). Both curves are warmup-masked.
    5. Lag analysis: time-averaged head-averaged attention distribution
       as a bar chart with the lag axis in minutes.

    Args:
        fhr: Raw FHR trace ``(R,)`` or ``None``.
        up: Raw UP trace ``(R,)`` or ``None``.
        attn_weights: Attention probabilities ``(T, M, L)``.
        te_lag_map: TE lag attribution ``(T, L)``.
        warmup: Warmup decimated steps.
        out_path: Destination PDF/PNG.
        guid: Sample GUID for the title bar.
        epoch: Sample epoch (seconds relative to delivery) for the title.
        label: Outcome class label for the title.
        fs_raw: Raw sampling rate in Hz (default 4.0).
        decim: Decimation factor (default 16).
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

    # --- Figure layout ---
    fig = plt.figure(figsize=(12, 14))
    gs = GridSpec(
        5, 1, figure=fig,
        hspace=0.55,
        height_ratios=[1.0, 1.9, 1.9, 1.1, 1.2],
        left=0.10, right=0.95, top=0.94, bottom=0.06,
    )

    # --- Row 0: Raw signals ---
    ax_raw = fig.add_subplot(gs[0, 0])
    _draw_raw_panel(
        ax_raw,
        fhr=fhr, up=up,
        time_raw_min=time_raw_min,
        t_max_min=t_max_min,
        warmup_min=warmup_min,
        title="Raw FHR / UP",
    )

    # Warmup-masked copies: the (L, T) heatmaps have time on axis=-1, so
    # masking leading columns hides the warmup region from both imshow
    # and auto vmin/vmax scaling.
    alpha_bar_img = _mask_warmup_time_axis(alpha_bar.T, warm, axis=-1)
    te_lag_img = _mask_warmup_time_axis(te_lag_map.T, warm, axis=-1)

    # --- Row 1: Attention matrix head-averaged ---
    ax_attn = fig.add_subplot(gs[1, 0])
    _draw_heatmap_min(
        ax_attn, alpha_bar_img,  # (L, T)
        t_max_min=t_max_min,
        cmap="viridis",
        ylabel="Lag (min)",
        title=r"Head-averaged lag attention  $\bar{\alpha}(t, k)$",
        colorbar_label="attention",
        y_extent=lag_span_min,
    )
    ax_attn.set_ylim(0.0, lag_span_min)
    ax_attn.set_xlabel("Time (min)", fontsize=FONT_LABEL)
    _shade_warmup_min(ax_attn, warmup_min)
    # Overlay argmax-lag curve (in minutes).
    if valid_mask.any():
        ax_attn.plot(
            time_dec_min[valid_mask],
            argmax_lag_min_masked[valid_mask],
            color=COLOR_VERMILLION,
            linewidth=0.9,
            alpha=0.9,
            label="argmax lag",
        )
        ax_attn.legend(loc="upper right", fontsize=6, frameon=True)

    # --- Row 2: TE lag attribution ---
    ax_te = fig.add_subplot(gs[2, 0])
    _draw_heatmap_min(
        ax_te, te_lag_img,
        t_max_min=t_max_min,
        cmap="inferno",
        vmin=0.0,
        ylabel="Lag (min)",
        title=r"TE lag attribution  $\mathrm{KL}_t \cdot \bar{\alpha}(t, k)$",
        colorbar_label="TE mass",
        y_extent=lag_span_min,
    )
    ax_te.set_ylim(0.0, lag_span_min)
    ax_te.set_xlabel("Time (min)", fontsize=FONT_LABEL)
    _shade_warmup_min(ax_te, warmup_min)

    # --- Row 3: Attention analysis — argmax lag + entropy over time ---
    ax_ana = fig.add_subplot(gs[3, 0])
    ax_ana.plot(
        time_dec_min, argmax_lag_min_masked,
        color=COLOR_BLUE, linewidth=1.0, label="argmax lag",
    )
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

    # --- Row 4: Lag analysis — time-averaged attention mass per lag ---
    ax_lag = fig.add_subplot(gs[4, 0])
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
