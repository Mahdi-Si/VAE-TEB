r"""Raw-domain visualizers for the ``SeqVaeRawV4`` evaluation pipeline (S7-T01).

The scattering pipeline's forecast plots operate on the $87$-channel feature future; the raw model
forecasts the future raw-FHR **waveform** $X^+ \in \mathbb{R}^{B \times T_{\mathrm{valid}} \times H
\times R}$. This module provides the raw analogues:

- :func:`plot_raw_forecast_overlay` -- predicted vs true raw-FHR waveform for one anchor (in
  denormalized bpm when stats are supplied), the $H\!\cdot\!R$ contiguous future samples.
- :func:`plot_raw_forecast_overlay_averaged` -- the same overlay averaged over samples and valid
  anchors, plus the $\pm 2\sigma$ predictive band when a log-variance array is given.
- :func:`plot_raw_per_horizon_error` -- MSE as a function of the horizon step $\tau$.
- :func:`plot_raw_lowpass_error` -- multi-scale block-average (low-pass) MSE bars.
- :func:`plot_raw_forecast_heatmap` -- the $(H \times T)$ mean-forecast heatmap.

Every function takes plain arrays (``numpy`` or ``torch``), writes a figure to ``out_path``, and
returns the ``Path`` written (and, for the heatmap, the $(H, T)$ array it drew) so tests can assert
on the artefact.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # headless: never require a display in eval / CI
import matplotlib.pyplot as plt
import numpy as np

from model.vae_teb_prediction.model.model_raw.testing.collectors import denormalize_signal

ArrayLike = Union[np.ndarray, "object"]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert a torch tensor or array-like to a detached float ``numpy`` array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _flatten_future(block: np.ndarray) -> np.ndarray:
    r"""Flatten a per-anchor $(H, R)$ future block to the contiguous $(H \cdot R,)$ waveform."""
    return np.asarray(block).reshape(-1)


def plot_raw_forecast_overlay(
    mu_anchor: ArrayLike,
    x_anchor: ArrayLike,
    out_path: Union[str, Path],
    *,
    fhr_stats: Optional[dict] = None,
    logvar_anchor: Optional[ArrayLike] = None,
    fs: int = 4,
    title: Optional[str] = None,
) -> Path:
    r"""Overlay the predicted and true raw-FHR waveform for a single anchor.

    Args:
        mu_anchor: Forecast mean for one anchor, shape $(H, R)$.
        x_anchor: True future block for the same anchor, shape $(H, R)$.
        out_path: Destination figure path.
        fhr_stats: Optional ``{"mean","std"}`` to render the overlay in bpm (else normalized units).
        logvar_anchor: Optional log-variance $(H, R)$ for a $\pm 2\sigma$ band.
        fs: Raw sampling rate (Hz) for the time axis.
        title: Optional plot title.

    Returns:
        The path written.
    """
    mu = _flatten_future(_to_numpy(mu_anchor))
    x = _flatten_future(_to_numpy(x_anchor))
    mu_bpm = denormalize_signal(mu, fhr_stats)
    x_bpm = denormalize_signal(x, fhr_stats)
    t = np.arange(mu.shape[0]) / float(fs)
    unit = "bpm" if fhr_stats is not None else "normalized"

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t, x_bpm, color="#1f77b4", lw=1.4, label="true FHR")
    ax.plot(t, mu_bpm, color="#d62728", lw=1.4, ls="--", label="forecast")
    if logvar_anchor is not None:
        sigma = np.exp(0.5 * _flatten_future(_to_numpy(logvar_anchor)))
        sigma_bpm = sigma * float(fhr_stats["std"]) if fhr_stats is not None else sigma
        ax.fill_between(
            t, mu_bpm - 2 * sigma_bpm, mu_bpm + 2 * sigma_bpm,
            color="#d62728", alpha=0.15, label=r"$\mu \pm 2\sigma$",
        )
    ax.set_xlabel("future time (s)")
    ax.set_ylabel(f"FHR ({unit})")
    ax.set_title(title or "Raw FHR forecast vs true")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_raw_forecast_overlay_averaged(
    mu: ArrayLike,
    x_plus: ArrayLike,
    out_path: Union[str, Path],
    *,
    warmup: int,
    horizon: int,
    fhr_stats: Optional[dict] = None,
    logvar: Optional[ArrayLike] = None,
    fs: int = 4,
) -> Path:
    r"""Overlay the sample/anchor-averaged predicted vs true future raw-FHR waveform.

    Args:
        mu: Forecast mean $(B, T, H, R)$ (the model emits the full $T$ anchor axis).
        x_plus: True future block $(B, T_{\mathrm{valid}}, H, R)$.
        out_path: Destination figure path.
        warmup: Warm-up anchors to skip.
        horizon: Forecast horizon $H$.
        fhr_stats: Optional ``{"mean","std"}`` for bpm rendering.
        logvar: Optional log-variance $(B, T, H, R)$ for a $\pm 2\sigma$ band.
        fs: Raw sampling rate (Hz).

    Returns:
        The path written.
    """
    mu = _to_numpy(mu)
    x_plus = _to_numpy(x_plus)
    T = mu.shape[1]
    end = max(T - int(horizon), 0)
    start = max(0, min(int(warmup), end))
    mu_v = mu[:, start:end]                    # (B, n, H, R)
    x_v = x_plus[:, start:end]                 # (B, n, H, R)
    b, n, h, r = mu_v.shape
    mu_mean = mu_v.reshape(b * max(n, 1), h * r).mean(axis=0)
    x_mean = x_v.reshape(b * max(n, 1), h * r).mean(axis=0)
    mu_bpm = denormalize_signal(mu_mean, fhr_stats)
    x_bpm = denormalize_signal(x_mean, fhr_stats)
    t = np.arange(h * r) / float(fs)
    unit = "bpm" if fhr_stats is not None else "normalized"

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t, x_bpm, color="#1f77b4", lw=1.6, label="true (mean)")
    ax.plot(t, mu_bpm, color="#d62728", lw=1.6, ls="--", label="forecast (mean)")
    if logvar is not None:
        lv = _to_numpy(logvar)[:, start:end].reshape(b * max(n, 1), h * r)
        sigma = np.exp(0.5 * lv).mean(axis=0)
        sigma_bpm = sigma * float(fhr_stats["std"]) if fhr_stats is not None else sigma
        ax.fill_between(
            t, mu_bpm - 2 * sigma_bpm, mu_bpm + 2 * sigma_bpm,
            color="#d62728", alpha=0.15, label=r"$\mu \pm 2\sigma$",
        )
    ax.set_xlabel("future time (s)")
    ax.set_ylabel(f"FHR ({unit})")
    ax.set_title("Mean raw FHR forecast vs true (over samples/anchors)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_raw_per_horizon_error(
    mse_per_horizon: ArrayLike,
    out_path: Union[str, Path],
    *,
    fs: int = 4,
    r_substeps: int = 16,
) -> Path:
    r"""Plot forecast MSE as a function of the horizon step $\tau$.

    Args:
        mse_per_horizon: Either a per-sample $(B, H)$ array (averaged internally) or a $(H,)$ vector.
        out_path: Destination figure path.
        fs: Raw sampling rate (Hz).
        r_substeps: Raw substeps per low-rate step $R$ (to label the horizon axis in seconds).

    Returns:
        The path written.
    """
    arr = _to_numpy(mse_per_horizon)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    horizon = arr.shape[0]
    lead_s = (np.arange(1, horizon + 1) * r_substeps) / float(fs)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(lead_s, arr, marker="o", color="#2ca02c", lw=1.5)
    ax.set_xlabel("forecast lead time (s)")
    ax.set_ylabel("MSE")
    ax.set_title("Raw forecast error by horizon step")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_raw_lowpass_error(
    mu: ArrayLike,
    x_plus: ArrayLike,
    out_path: Union[str, Path],
    *,
    warmup: int,
    horizon: int,
    scales_sec: Sequence[int] = (4, 16, 32, 60),
    fs: int = 4,
) -> Path:
    r"""Plot the multi-scale block-average (low-pass) MSE.

    At each scale the contiguous $H\!\cdot\!R$ future is block-averaged to the given width and the
    squared error of the block means is reduced -- the trend-vs-jitter decomposition of §10.

    Args:
        mu: Forecast mean $(B, T, H, R)$.
        x_plus: True future block $(B, T_{\mathrm{valid}}, H, R)$.
        out_path: Destination figure path.
        warmup: Warm-up anchors to skip.
        horizon: Forecast horizon $H$.
        scales_sec: Block-average scales in seconds.
        fs: Raw sampling rate (Hz).

    Returns:
        The path written.
    """
    mu = _to_numpy(mu)
    x_plus = _to_numpy(x_plus)
    T = mu.shape[1]
    end = max(T - int(horizon), 0)
    start = max(0, min(int(warmup), end))
    mu_v = mu[:, start:end]
    x_v = x_plus[:, start:end]
    b, n, h, r = mu_v.shape
    length = h * r
    fut_mu = mu_v.reshape(b, n, length)
    fut_x = x_v.reshape(b, n, length)

    values = []
    labels = []
    for scale in scales_sec:
        width = max(1, min(int(round(float(scale) * fs)), length))
        n_blocks = length // width
        if n_blocks == 0:
            continue
        mu_b = fut_mu[..., : n_blocks * width].reshape(b, n, n_blocks, width).mean(axis=-1)
        x_b = fut_x[..., : n_blocks * width].reshape(b, n, n_blocks, width).mean(axis=-1)
        values.append(float(((mu_b - x_b) ** 2).mean()))
        labels.append(f"{scale}s")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(range(len(values)), values, color="#9467bd")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("low-pass scale")
    ax.set_ylabel("block-average MSE")
    ax.set_title("Multi-scale low-pass forecast error")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_raw_forecast_heatmap(
    mu: ArrayLike,
    out_path: Union[str, Path],
    *,
    warmup: int,
    horizon: int,
) -> Tuple[Path, np.ndarray]:
    r"""Draw the $(H \times T)$ mean-forecast heatmap and return the array it drew.

    The forecast is averaged over samples and the $R$ raw substeps, giving a mean forecast per
    (anchor $t$, horizon step $\tau$) cell, transposed to $(H, T_v)$ so horizon runs down the rows.

    Args:
        mu: Forecast mean $(B, T, H, R)$.
        out_path: Destination figure path.
        warmup: Warm-up anchors to skip.
        horizon: Forecast horizon $H$.

    Returns:
        ``(path_written, heat)`` where ``heat`` has shape $(H, T_{\mathrm{valid}})$
        ($T_{\mathrm{valid}} = T - H$; the whole valid-anchor axis is shown, ``warmup`` only
        draws a marker line).
    """
    mu = _to_numpy(mu)
    T = mu.shape[1]
    end = max(T - int(horizon), 0)
    mu_v = mu[:, :end]                          # (B, T_valid, H, R)
    heat = mu_v.mean(axis=(0, 3)).T             # (H, T_valid)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.imshow(heat, aspect="auto", origin="lower", cmap="viridis")
    if 0 < int(warmup) < heat.shape[1]:
        ax.axvline(int(warmup) - 0.5, color="white", ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("anchor t")
    ax.set_ylabel("horizon step tau")
    ax.set_title("Mean raw forecast heatmap (H x T)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path, heat
