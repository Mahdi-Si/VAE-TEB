"""Diagnostic plotting callback for :class:`SeqVaeLagAttnV1` training.

Generates a consolidated multi-row publication-quality figure on every
validation epoch (gated by ``plot_frequency``). Style helpers are imported
from :mod:`utils.style` so training diagnostics and
test-time figures share the same visual language.

The layout follows :mod:`model.vae_teb_prediction.testing.plot_single_samples`:
every panel is stacked in a single column, and **every main axes has the
exact same width** irrespective of whether the row shows a colorbar. This is
achieved with a 2-column :class:`matplotlib.gridspec.GridSpec` whose second
column is a narrow fixed-width slot reserved for the colorbar. Rows that are
line plots (raw signals, KL/entropy traces) simply hide their reserved cax,
so the main axes width is still driven by the same gridspec column and stays
perfectly aligned with every heatmap row above and below it.

Every row uses the same x-axis: **time in seconds, from 0 to R / fs_raw**
(typically ``R = 4800``, ``fs_raw = 4 Hz`` → 1200 s). The raw FHR/UP trace,
the decimated feature heatmaps, the latent heatmap, the KLD maps, the lag
attention, and both forecast rows are all aligned column-for-column — so a
vertical line through any time point cuts every row at the same physical
instant.

Row layout (top-to-bottom, every row is a full-width single axes):

0.  Raw FHR + UP — twin y-axes (optional, only if ``fhr``/``up`` are loaded).
1.  FHR features — stacked ``[fhr_st | fhr_ph]`` heatmap (87, T) with a
    horizontal separator at the st/ph boundary.
2.  UP features — stacked ``[up_st | up_ph]`` heatmap (101, T) with a
    separator at the st/ph boundary. Collapses to just ``up_ph`` (58, T)
    when ``use_up_st=False``.
3.  Latent ``z`` heatmap (d_z, T).
4.  Posterior μ vs prior μ stacked heatmap (2·d_z, T) with separator.
5.  KLD per latent dim — single ``(d_z, T)`` imshow (``magma``, vmin=0),
    aligned with every other row.
6.  Total KLD per time step + attention entropy (twin-axis trace).
7.  Lag attention matrix (L, T) with the argmax-lag-per-step overlay.
8.  TE lag attribution — raw ``kld_per_t × mean_alpha`` with the colour
    range clipped to the 99th percentile so rare attention spikes don't
    black-out the rest.
9.  TE lag attribution — column-normalised (each time step divided by its
    own max) so the lag-selection pattern is visible even when the per-step
    KL is tiny. Columns with effectively zero KL are masked to NaN.
10. Average forecast ``μ_full`` as an ``(C_y, T)`` imshow — overlap-averaged
    per-anchor horizons across **all** feature channels (87 rows), with a
    white separator at the fhr_st ↔ fhr_ph boundary.
11. Single-horizon forecast ``μ_full`` as an ``(C_y, T)`` imshow —
    non-overlapping stride-``H_d`` concatenation across all feature
    channels, with the same channel separator.
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
from matplotlib.gridspec import GridSpec  # noqa: E402

from utils.style import (
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_LIGHT_GRAY,
    COLOR_ORANGE,
    COLOR_PURPLE,
    COLOR_VERMILLION,
    SAVE_DPI,
    apply_publication_style,
    save_figure,
    style_axes,
)

COLOR_BLUE = "#3F72AF"
COLOR_ORANGE = "#FFB200"
COLOR_GREEN = "#46D855"
COLOR_SKY = "#00ADB5"
COLOR_PURPLE = "#5642EB"
COLOR_VERMILLION = "#F23F04"
COLOR_GRAY = "#393E46"
COLOR_BLACK = "#000000"
COLOR_LIGHT_GRAY = "#EEEEEE"
COLOR_SAGE = "#A3E782"
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


def _time_axes(
    T: int, R: int, fs_raw: float = 4.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return ``(time_raw_sec, time_dec_sec, t_max_sec)`` for unified alignment.

    All diagnostic rows share a single physical-time axis. Raw FHR/UP trace
    lives on ``time_raw`` (step ``1/fs_raw``); decimated features / latents
    live on ``time_dec`` (step ``t_max / T``). The imshow ``extent`` and the
    ``ax.set_xlim`` call for every axes both use ``(0.0, t_max_sec)``.

    Args:
        T: Number of decimated steps (e.g. 300).
        R: Number of raw samples (e.g. 4800).
        fs_raw: Raw sampling rate in Hz (default 4.0).

    Returns:
        Tuple ``(time_raw, time_dec, t_max_sec)``:
        - ``time_raw``: ``(R,)`` seconds axis for the raw signal.
        - ``time_dec``: ``(T,)`` seconds axis for decimated tensors; step
          centres at ``0, Δ, 2Δ, …`` with ``Δ = t_max / T``.
        - ``t_max_sec``: total window length in seconds (``R / fs_raw``).
    """
    t_max = float(R) / float(fs_raw)
    time_raw = np.arange(R, dtype=np.float64) / float(fs_raw)
    time_dec = np.arange(T, dtype=np.float64) * (t_max / float(T))
    return time_raw, time_dec, t_max


def _attach_lag_seconds_axis(
    ax: Any, step_seconds: float, delta_up_seconds: float
) -> Any:
    r"""Add a right-hand secondary y-axis in physical seconds (arch spec section 27).

    Maps a decimated lag index $\ell$ to $\mathrm{lag}_{\mathrm{phys}}(\ell) =
    s\,\ell + \Delta_{UP}$ (``step_seconds`` $s$, ``delta_up_seconds``
    $\Delta_{UP}$), so the lag panels read in both model-lag and physical-second
    coordinates. Non-fatal: any Matplotlib error is swallowed so plotting never
    crashes training.

    Args:
        ax: The lag-panel axes (primary y is the decimated lag $\ell$).
        step_seconds: Decimated step duration $s$ in seconds.
        delta_up_seconds: Fixed preprocessing UP shift $\Delta_{UP}$ in seconds.

    Returns:
        The created secondary axis, or ``None`` if it could not be attached.
    """
    s = float(step_seconds)
    d = float(delta_up_seconds)
    if s <= 0.0:
        return None
    try:
        sec = ax.secondary_yaxis(
            "right",
            functions=(lambda l: s * l + d, lambda v: (v - d) / s),
        )
        sec.set_ylabel("Lag (s)", fontsize=8)
        return sec
    except Exception:  # noqa: BLE001 — plotting must never crash training
        return None


def _shade_warmup(
    ax: Any,
    warmup: int,
    t_max: float,
    T: int,
    *,
    color: str = COLOR_LIGHT_GRAY,
) -> None:
    """Shade the first ``warmup`` decimated steps, in seconds, on an axes.

    Args:
        ax: Target axes.
        warmup: Warmup length in decimated steps.
        t_max: Full x-axis extent in seconds (``R / fs_raw``).
        T: Total number of decimated steps (for step-to-seconds conversion).
        color: Shading colour.
    """
    if warmup and warmup > 0 and T > 0:
        warmup_sec = float(warmup) * (t_max / float(T))
        ax.axvspan(0.0, warmup_sec, color=color, alpha=0.35, zorder=0)


def _average_forecast_per_channel(
    mu_pred: np.ndarray,
    T: int,
    H_d: int,
    warmup: int,
) -> np.ndarray:
    """Average overlapping per-anchor horizon forecasts onto the decimated axis.

    Anchor ``t ∈ [warmup, T - H_d)`` contributes its per-horizon prediction
    ``mu_pred[t, h, :]`` to the target decimated index ``τ = t + 1 + h``.
    The returned array averages every anchor's contribution to each ``τ``.
    Positions with no contributing anchor are set to ``NaN`` (so they render
    as gaps in a matplotlib imshow, or masked cells in a heatmap).

    Args:
        mu_pred: ``(T, H_d, C)`` per-anchor horizon prediction.
        T: Number of decimated steps.
        H_d: Forecast horizon in decimated steps.
        warmup: Warmup length in decimated steps.

    Returns:
        ``(T, C)`` averaged forecast (float32). Uncovered positions are NaN.
    """
    C = mu_pred.shape[-1]
    acc = np.zeros((T, C), dtype=np.float64)
    cnt = np.zeros((T,), dtype=np.float64)
    t_start = max(int(warmup), 0)
    t_end = max(t_start, T - H_d)
    for t in range(t_start, t_end):
        tau_end = min(t + 1 + H_d, T)
        tau = np.arange(t + 1, tau_end)
        h = tau - (t + 1)
        acc[tau] += mu_pred[t, h, :]
        cnt[tau] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = acc / np.where(cnt > 0.0, cnt, 1.0)[:, None]
    avg[cnt == 0.0] = np.nan
    return avg.astype(np.float32)


def _concat_single_forecasts(
    mu_pred: np.ndarray,
    T: int,
    H_d: int,
    warmup: int,
) -> np.ndarray:
    """Non-overlapping, stride-``H_d`` concatenation of per-anchor horizons.

    Starting at ``t = warmup``, walk forward in strides of ``H_d`` anchors;
    each anchor contributes its full horizon slice ``[t+1, t+1+H_d)`` to the
    output. Any positions not covered are ``NaN`` so imshow masks them.

    Args:
        mu_pred: ``(T, H_d, C)`` per-anchor horizon prediction.
        T: Number of decimated steps.
        H_d: Forecast horizon in decimated steps.
        warmup: Warmup length in decimated steps.

    Returns:
        ``(T, C)`` concatenated forecast (float32). Uncovered positions NaN.
    """
    C = mu_pred.shape[-1]
    out = np.full((T, C), np.nan, dtype=np.float32)
    t = max(int(warmup), 0)
    while t + 1 + H_d <= T and t < T:
        out[t + 1 : t + 1 + H_d, :] = mu_pred[t, :, :].astype(np.float32)
        t += H_d
    return out


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


def _safe_vabs(arr: np.ndarray) -> float:
    """Return a strictly-positive symmetric colour-limit for a ``bwr`` imshow.

    Ignores NaN/Inf entries. Falls back to ``1.0`` if the array has no finite
    values or the finite max is zero.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    vabs = float(np.abs(finite).max())
    return vabs if vabs > 0.0 else 1.0


# =============================================================================
# Figure builder
# =============================================================================


def _build_diagnostic_figure(
    *,
    outs: Dict[str, Any],
    y_st: torch.Tensor,
    y_ph: torch.Tensor,
    up_st: Optional[torch.Tensor],
    up_ph: torch.Tensor,
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
    step_seconds: float = 4.0,
    delta_up_seconds: float = 0.0,
) -> Any:
    """Build the full diagnostic figure for one validation sample.

    The figure is laid out as a **single column** of full-width axes using a
    2-column :class:`GridSpec` — column 0 hosts the main axes, column 1 is a
    narrow fixed-width slot for the colorbar axes. Line-plot rows hide their
    reserved cax so the main-axes widths stay perfectly aligned row-to-row
    regardless of whether a colorbar is visible.

    Args:
        outs: Forward-output dict from :meth:`SeqVaeLagAttnV1.forward`.
        y_st: FHR scattering features ``(B, T, 43)``.
        y_ph: FHR phase features ``(B, T, 44)``.
        up_st: UP scattering features ``(B, T, 43)``, or ``None`` if absent.
        up_ph: UP self-phase harmonics ``(B, T, 58)`` — first-class HDF5 field
            with its own per-channel asinh stats.
        fhr_raw: Raw FHR trace ``(B, R)`` or ``None``.
        up_raw: Raw UP trace ``(B, R)`` or ``None``.
        sample_idx: Index into the batch.
        epoch: Current training epoch.
        guid: GUID string for the figure title.
        warmup: Warmup period ``T_w`` (for shading invalid regions).
        horizon: Decimated forecast horizon ``H_d``.
        forecast_channels: Kept for backward compatibility with the callback
            config; no longer used — the new forecast rows draw every feature
            channel as a full imshow instead of per-channel line plots.
        forecast_anchor_frac: Kept for backward compatibility with the
            callback config; no longer used by the new forecast rows.
        beta: Current KL weight.
        feat_loss: Current ``L_feat`` (full-forecast MSE).
        base_loss: Current ``L_base`` (baseline-forecast MSE).
        kld_loss: Current ``L_KL`` (mean KL).

    Returns:
        The constructed :class:`matplotlib.figure.Figure`. The caller is
        responsible for saving and closing it.
    """
    # Legacy kwargs kept in the signature for back-compat — no longer used.
    del forecast_channels, forecast_anchor_frac

    i = sample_idx

    # ---- Numpy views of the relevant tensors for this sample ---------------
    y_st_np = _np(y_st[i])                                          # (T, 43)
    y_ph_np = _np(y_ph[i])                                          # (T, 44)
    up_ph_np = _np(up_ph[i])                                        # (T, 58)
    up_st_np = _np(up_st[i]) if up_st is not None else None         # (T, 43) or None

    mu_prior_np = _np(outs["mu_prior"][i])                          # (T, d_z)
    logvar_prior_np = _np(outs["logvar_prior"][i])
    mu_post_np = _np(outs["mu_post"][i])
    logvar_post_np = _np(outs["logvar_post"][i])
    z_np = _np(outs["z"][i])                                        # (T, d_z)

    attn_np = _np(outs["attn_weights"][i])                          # (T, M, L)
    te_lag_np = _np(outs["te_lag_map"][i])                          # (T, L)
    kld_per_t_np = _np(outs["kld_per_t"][i])                        # (T,)

    mu_full_np = _np(outs["mu_full"][i])                            # (T, H_d, 87)

    T = int(y_st_np.shape[0])
    d_z = int(mu_prior_np.shape[-1])
    L = int(attn_np.shape[-1])
    H_d = int(horizon)
    C_y = int(y_st_np.shape[-1] + y_ph_np.shape[-1])
    st_ch = int(y_st_np.shape[-1])

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

    # Full-sequence forecast reductions on mu_full only (the residual-
    # corrected prediction). The baseline mu_base is tracked in the loss via
    # lambda_base but is not drawn here to keep the layout compact.
    avg_full = _average_forecast_per_channel(mu_full_np, T, H_d, warmup)
    concat_full = _concat_single_forecasts(mu_full_np, T, H_d, warmup)

    # ------------------------------------------------------------------
    # Shared physical-time axis (seconds) — every row uses this.
    # ------------------------------------------------------------------
    has_raw = fhr_raw is not None and up_raw is not None
    fs_raw = 4.0
    if has_raw:
        assert fhr_raw is not None and up_raw is not None
        R = int(_np(fhr_raw[i]).ravel().shape[0])
    else:
        # Fall back to the nominal 16× decimation so downstream math is
        # still well-defined when the raw signal is missing from the batch.
        R = T * 16
    time_raw, time_dec, t_max = _time_axes(T, R, fs_raw=fs_raw)

    # ------------------------------------------------------------------
    # Figure and gridspec
    # ------------------------------------------------------------------
    apply_publication_style()

    # One full-width axes per row. The two TE-lag panels and the two
    # forecast panels are each their own row — no nested gridspecs — so that
    # every main axes lives in column 0 of the top-level gridspec and ends
    # up with exactly the same width.
    row_specs = []  # (name, height_ratio)
    if has_raw:
        row_specs.append(("raw", 0.9))
    row_specs += [
        ("fhr_feats", 1.35),
        ("up_feats", 1.45),
        ("z_latent", 1.0),
        ("post_prior", 1.1),
        ("kld_dims", 1.15),
        ("kld_total", 0.9),
        ("lag_attn", 1.15),
        ("te_lag_raw", 1.15),
        ("te_lag_norm", 1.15),
        ("forecast_avg", 1.35),
        ("forecast_single", 1.35),
    ]
    n_rows = len(row_specs)
    height_ratios = [h for _, h in row_specs]
    total_height = sum(height_ratios) * 2.6
    fig = plt.figure(figsize=(14, total_height))

    # 2-column gridspec: col 0 = main axes, col 1 = narrow cax. All rows
    # share col 0, so the main axes widths are identical.
    gs = GridSpec(
        n_rows, 2, figure=fig,
        height_ratios=height_ratios,
        width_ratios=[1.0, 0.022],
        left=0.065, right=0.93, top=0.985, bottom=0.025,
        hspace=0.55, wspace=0.03,
    )
    row_idx_of = {name: idx for idx, (name, _) in enumerate(row_specs)}

    def row_axes(name: str) -> Tuple[Any, Any]:
        """Create the (main, cax) pair for a named row."""
        r = row_idx_of[name]
        return fig.add_subplot(gs[r, 0]), fig.add_subplot(gs[r, 1])

    def _attach_cbar(cax: Any, im: Any, label: str) -> Any:
        """Attach a colorbar onto the reserved cax for a row."""
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(label, fontsize=8, color=COLOR_BLACK)
        cbar.ax.tick_params(labelsize=7, colors=COLOR_BLACK)
        # Cast to Any so matplotlib's ``Spine | None`` typing quirk (the
        # ``Spine.set_*`` methods confuse pyright) doesn't produce false
        # positives; the runtime behaviour is unchanged.
        outline: Any = cbar.outline
        if outline is not None:
            outline.set_linewidth(0.6)
            outline.set_edgecolor(COLOR_LIGHT_GRAY)
        return cbar

    def _hide_cax(cax: Any) -> None:
        """Hide an unused cax (keeps the main axes width consistent)."""
        cax.set_visible(False)

    def _style_heatmap_spines(ax: Any) -> None:
        """Draw all four spines on a heatmap axes."""
        ax.grid(False)
        for spine in ("top", "bottom", "left", "right"):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color(COLOR_BLACK)
            ax.spines[spine].set_linewidth(0.6)

    def _finalise_time_axis(ax: Any) -> None:
        """Every row ends with this so all panels line up column-for-column."""
        ax.set_xlim(0.0, t_max)
        _shade_warmup(ax, warmup, t_max, T)

    # ---- Row: Raw FHR + UP -------------------------------------------------
    if has_raw:
        ax, cax = row_axes("raw")
        assert fhr_raw is not None and up_raw is not None  # has_raw guard
        fhr_np = _np(fhr_raw[i]).ravel()
        up_np = _np(up_raw[i]).ravel()
        ax.plot(time_raw, fhr_np, color=COLOR_BLUE, linewidth=0.8, label="FHR")
        ax2 = ax.twinx()
        ax2.plot(time_raw, up_np, color=COLOR_GREEN, linewidth=0.8, label="UP")
        ax.set_title("Raw FHR / UP signals", fontsize=9, pad=6)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("FHR (normalised)", fontsize=8, color=COLOR_BLUE)
        ax.tick_params(axis="y", labelcolor=COLOR_BLUE)
        ax2.set_ylabel("UP (normalised)", fontsize=8, color=COLOR_GREEN)
        ax2.tick_params(axis="y", labelcolor=COLOR_GREEN)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2, labels1 + labels2,
            loc="upper right", fontsize=7, framealpha=0.95,
        )
        style_axes(ax, grid="both")
        _finalise_time_axis(ax)
        ax2.set_xlim(0.0, t_max)
        _hide_cax(cax)

    # ---- Row: FHR features stacked heatmap --------------------------------
    ax, cax = row_axes("fhr_feats")
    fhr_stack, fhr_sep = _stack_feature_blocks(y_st_np.T, y_ph_np.T)     # (87, T)
    vabs_fhr = _safe_vabs(fhr_stack)
    im = ax.imshow(
        fhr_stack, aspect="auto", cmap="bwr", origin="upper",
        vmin=-vabs_fhr, vmax=vabs_fhr,
        extent=[0.0, t_max, fhr_stack.shape[0] - 0.5, -0.5],
    )
    ax.set_title(
        "FHR features \u2014 scattering (rows 0-42)  |  phase (rows 43-86)",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Channel", fontsize=8)
    if fhr_sep is not None:
        ax.axhline(fhr_sep + 0.5, color="white", linewidth=1.2, linestyle="--")
    _style_heatmap_spines(ax)
    _attach_cbar(cax, im, "Value")
    _finalise_time_axis(ax)

    # ---- Row: UP features stacked heatmap ---------------------------------
    ax, cax = row_axes("up_feats")
    if up_st_np is not None:
        up_stack, up_sep = _stack_feature_blocks(up_st_np.T, up_ph_np.T)  # (101, T)
        title_up = (
            "UP features \u2014 scattering (rows 0-42)  |  self-phase (rows 43-100)"
        )
    else:
        up_stack, up_sep = up_ph_np.T, None                               # (58, T)
        title_up = "UP features \u2014 self-phase only (up_st absent)"
    vabs_up = _safe_vabs(up_stack)
    im = ax.imshow(
        up_stack, aspect="auto", cmap="bwr", origin="upper",
        vmin=-vabs_up, vmax=vabs_up,
        extent=[0.0, t_max, up_stack.shape[0] - 0.5, -0.5],
    )
    ax.set_title(title_up, fontsize=9, pad=6)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Channel", fontsize=8)
    if up_sep is not None:
        ax.axhline(up_sep + 0.5, color="white", linewidth=1.2, linestyle="--")
    _style_heatmap_spines(ax)
    _attach_cbar(cax, im, "Value")
    _finalise_time_axis(ax)

    # ---- Row: Latent z ----------------------------------------------------
    ax, cax = row_axes("z_latent")
    z_img = z_np.T                                                  # (d_z, T)
    vabs_z = _safe_vabs(z_img)
    im = ax.imshow(
        z_img, aspect="auto", cmap="bwr", origin="lower",
        vmin=-vabs_z, vmax=vabs_z,
        extent=[0.0, t_max, -0.5, d_z - 0.5],
    )
    ax.set_title(f"Latent z (d_z={d_z})", fontsize=9, pad=6)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Latent dim", fontsize=8)
    _style_heatmap_spines(ax)
    _attach_cbar(cax, im, "z")
    _finalise_time_axis(ax)

    # ---- Row: Posterior μ vs prior μ split heatmap ------------------------
    ax, cax = row_axes("post_prior")
    post_prior = np.concatenate([mu_post_np.T, mu_prior_np.T], axis=0)   # (2*d_z, T)
    vabs_pp = _safe_vabs(post_prior)
    im = ax.imshow(
        post_prior, aspect="auto", cmap="bwr", origin="upper",
        vmin=-vabs_pp, vmax=vabs_pp,
        extent=[0.0, t_max, 2 * d_z - 0.5, -0.5],
    )
    ax.axhline(d_z - 0.5, color="white", linewidth=1.2, linestyle="--")
    ax.set_yticks([d_z // 2, d_z + d_z // 2])
    ax.set_yticklabels(["Posterior \u03bc", "Prior \u03bc\u2070"])
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_title(
        "Posterior vs Prior means (TEB residual = posterior \u2212 prior)",
        fontsize=9, pad=6,
    )
    _style_heatmap_spines(ax)
    _attach_cbar(cax, im, "Value")
    _finalise_time_axis(ax)

    # ---- Row: KLD per-latent-dim — single imshow --------------------------
    ax, cax = row_axes("kld_dims")
    kld_img = kld_per_dim.T                                         # (d_z, T)
    # Clip tiny negative rounding artefacts from the closed-form formula.
    kld_img = np.where(np.isfinite(kld_img), kld_img, 0.0)
    kld_max = float(np.nanmax(kld_img)) if kld_img.size else 0.0
    vmax_kld = kld_max if kld_max > 0.0 else 1.0
    im = ax.imshow(
        kld_img, aspect="auto", cmap="magma", origin="lower",
        vmin=0.0, vmax=vmax_kld,
        extent=[0.0, t_max, -0.5, d_z - 0.5],
    )
    # Under v2 the flat mu_post / logvar_post are law-of-total-variance mixture
    # moments, so this closed-form per-dim KL is a moment-matched *proxy*: it no
    # longer row-sums to ``kld_per_t`` (the exact decomposed $K_t = K^R + K^Z$
    # shown in the Total-KL panel below). Annotate it so the two KL panels read
    # as mutually consistent (S6-T01). v2 is detected by its decomposed KL key.
    _kld_proxy = "kld_content" in outs
    ax.set_title(
        f"KLD per latent dim (d_z={d_z} rows) \u2014 max={kld_max:.3f} nats"
        + (" \u2014 moment-matched proxy (exact $K_t$ in Total-KL panel)"
           if _kld_proxy else ""),
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Latent dim", fontsize=8)
    _style_heatmap_spines(ax)
    _attach_cbar(cax, im, "KL (nats)")
    _finalise_time_axis(ax)

    # ---- Row: Total KL per step + attention entropy (twin-axis) -----------
    ax, cax = row_axes("kld_total")
    ax.plot(
        time_dec, kld_per_t_np, color=COLOR_PURPLE,
        linewidth=1.0, label="KL per step",
    )
    ax.set_ylabel("KL (nats)", fontsize=8, color=COLOR_PURPLE)
    ax.tick_params(axis="y", labelcolor=COLOR_PURPLE)
    ax2 = ax.twinx()
    ax2.plot(
        time_dec, attn_entropy_per_step, color=COLOR_ORANGE,
        linewidth=0.9, alpha=0.85, label="Attention entropy",
    )
    ax2.set_ylabel("Attention entropy (nats)", fontsize=8, color=COLOR_ORANGE)
    ax2.tick_params(axis="y", labelcolor=COLOR_ORANGE)
    ax.set_title(
        "Total KL per timestep vs mean attention entropy", fontsize=9, pad=6,
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper right", fontsize=7, framealpha=0.95,
    )
    style_axes(ax, grid="both")
    _finalise_time_axis(ax)
    ax2.set_xlim(0.0, t_max)
    _hide_cax(cax)

    # ---- Row: Lag attention matrix (mean over heads) ----------------------
    ax, cax = row_axes("lag_attn")
    im = ax.imshow(
        mean_alpha.T, aspect="auto", cmap="viridis", origin="lower",
        extent=[0.0, t_max, -0.5, L - 0.5],
    )
    ax.plot(
        time_dec, argmax_lag, color=COLOR_VERMILLION,
        linewidth=0.9, alpha=0.9, label="argmax lag",
    )
    ax.set_title(
        f"Lag attention \u2014 mean over {attn_np.shape[1]} heads "
        f"(L={L} = 0..{L - 1})",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Lag \u2113 (0 = current)", fontsize=8)
    _attach_lag_seconds_axis(ax, step_seconds, delta_up_seconds)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.95)
    _style_heatmap_spines(ax)
    _attach_cbar(cax, im, "Attn prob")
    _finalise_time_axis(ax)

    # ---- Rows: TE lag attribution — raw and column-normalised ------------
    # Raw: colour range clipped to the 99th percentile so rare attention
    # spikes don't black-out the rest. Column-normalised: each time step
    # divided by its own max so the lag-selection pattern is visible even
    # when the per-step KL is tiny. Columns with effectively zero KL are
    # masked to NaN so imshow leaves them blank.
    te_map = te_lag_np.T                                             # (L, T)
    te_map = np.where(np.isfinite(te_map) & (te_map > 0.0), te_map, 0.0)
    te_global_max = float(te_map.max()) if te_map.size else 0.0

    ax, cax = row_axes("te_lag_raw")
    te_positive = te_map[te_map > 0.0]
    if te_positive.size > 0:
        vmax_te_p99 = float(np.nanpercentile(te_positive, 99.0))
    else:
        vmax_te_p99 = 0.0
    if vmax_te_p99 <= 0.0:
        vmax_te_p99 = te_global_max if te_global_max > 0.0 else 1.0
    im = ax.imshow(
        te_map, aspect="auto", cmap="magma", origin="lower",
        extent=[0.0, t_max, -0.5, L - 0.5],
        vmin=0.0, vmax=vmax_te_p99,
    )
    ax.set_title(
        "TE lag attribution (KL \u00d7 mean-\u03b1) \u2014 p99-clipped "
        f"(vmax={vmax_te_p99:.3e}, max={te_global_max:.3e})",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Lag \u2113 (0 = current)", fontsize=8)
    _attach_lag_seconds_axis(ax, step_seconds, delta_up_seconds)
    _style_heatmap_spines(ax)
    _attach_cbar(cax, im, "KL weight")
    _finalise_time_axis(ax)

    ax, cax = row_axes("te_lag_norm")
    col_max = te_map.max(axis=0, keepdims=True)                      # (1, T)
    col_threshold = max(1e-12, te_global_max * 1e-6)
    valid_cols = col_max > col_threshold                             # (1, T)
    denom = np.where(valid_cols, col_max, 1.0)
    te_norm = np.where(valid_cols, te_map / denom, np.nan)           # (L, T)
    im = ax.imshow(
        te_norm, aspect="auto", cmap="viridis", origin="lower",
        extent=[0.0, t_max, -0.5, L - 0.5],
        vmin=0.0, vmax=1.0,
    )
    ax.set_title(
        "TE lag attribution \u2014 column-normalised "
        "(dominant lag per time step, independent of KL magnitude)",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Lag \u2113 (0 = current)", fontsize=8)
    _attach_lag_seconds_axis(ax, step_seconds, delta_up_seconds)
    _style_heatmap_spines(ax)
    _attach_cbar(cax, im, "Column-norm")
    _finalise_time_axis(ax)

    # ---- Rows: Forecast — avg and single, imshow over all channels ------
    # Both rows draw μ_full across all 87 feature channels (fhr_st on rows
    # 0..42, fhr_ph on rows 43..86, with a white separator line between
    # them). The first row is the overlap-averaged forecast, the second is
    # the stride-H_d single-window concatenation. A shared symmetric colour
    # range is used across both rows so the two imshows are directly
    # comparable, driven by the larger of the two finite ranges.
    forecast_stack = np.concatenate(
        [avg_full[np.isfinite(avg_full)], concat_full[np.isfinite(concat_full)]]
    )
    if forecast_stack.size > 0:
        vabs_fc = float(np.abs(forecast_stack).max())
        if vabs_fc <= 0.0:
            vabs_fc = 1.0
    else:
        vabs_fc = 1.0

    ax, cax = row_axes("forecast_avg")
    avg_img = avg_full.T                                             # (C_y, T)
    im = ax.imshow(
        avg_img, aspect="auto", cmap="bwr", origin="upper",
        vmin=-vabs_fc, vmax=vabs_fc,
        extent=[0.0, t_max, C_y - 0.5, -0.5],
    )
    ax.axhline(st_ch - 0.5, color="white", linewidth=1.2, linestyle="--")
    ax.set_title(
        "Average forecast \u03bc_full \u2014 overlap-averaged per-anchor "
        f"horizons (all {C_y} channels, H_d={H_d})",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Feature ch", fontsize=8)
    _style_heatmap_spines(ax)
    _attach_cbar(cax, im, "Value")
    _finalise_time_axis(ax)

    ax, cax = row_axes("forecast_single")
    single_img = concat_full.T                                       # (C_y, T)
    im = ax.imshow(
        single_img, aspect="auto", cmap="bwr", origin="upper",
        vmin=-vabs_fc, vmax=vabs_fc,
        extent=[0.0, t_max, C_y - 0.5, -0.5],
    )
    ax.axhline(st_ch - 0.5, color="white", linewidth=1.2, linestyle="--")
    ax.set_title(
        "Single-horizon forecast \u03bc_full \u2014 non-overlapping "
        f"concat (stride H_d={H_d}, all {C_y} channels)",
        fontsize=9, pad=6,
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Feature ch", fontsize=8)
    _style_heatmap_spines(ax)
    _attach_cbar(cax, im, "Value")
    _finalise_time_axis(ax)

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
        forecast_channels: Kept for backward compatibility with existing
            config files. The new forecast rows are full imshows over every
            feature channel, so this value is no longer used.
        forecast_anchor_frac: Kept for backward compatibility with existing
            config files. No longer used — the new forecast rows show the
            full-sequence averaged/concatenated maps instead of an anchor
            single-shot.
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
        up_ph = _get_field(batch, "up_ph")
        if up_ph is None:
            logger.warning(
                "LagAttnV1PlotCallback: batch has no `up_ph` field; skipping "
                "this epoch's plots. Make sure 'up_ph' is in "
                "dataset_kwargs.load_fields."
            )
            return

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
            # Narrow both Optional[Tensor] locals for the type checker before
            # torch.cat — pyright loses the prior None-checks across the
            # ``use_up_st`` branch and otherwise flags the list literal.
            assert up_st is not None and up_ph is not None
            u_stream = torch.cat([up_st, up_ph], dim=-1)
        else:
            u_stream = up_ph

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
            # Physical-time lag axis (arch spec section 27). Read guarded so a v1
            # model (which lacks these attrs) still renders the lag panels.
            step_seconds = float(getattr(model, "step_seconds", 4.0))
            delta_up_seconds = float(getattr(model, "delta_up_seconds", 0.0))

            num_samples = min(self.num_examples, y_st.shape[0])
            for s in range(num_samples):
                guid = _guid_of(batch, s)
                fig = _build_diagnostic_figure(
                    outs=outs,
                    y_st=y_st,
                    y_ph=y_ph,
                    up_st=up_st,
                    up_ph=up_ph,
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
                    step_seconds=step_seconds,
                    delta_up_seconds=delta_up_seconds,
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
