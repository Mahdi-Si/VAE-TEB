r"""Figure generation for ``synthetic_v2``.

Sprint 1 provides the annotated raw-signal preview (:func:`plot_raw_preview`,
S1-T05): a two-panel figure of one rendered FHR/UP pair with the physiological
baseline / resting tone and the coupled deceleration / contraction events marked.
Sprint 2 adds the scattering-coefficient heatmap (:func:`plot_scattering_heatmap`,
S2-T04): stacked FHR / UP first-order magnitude heatmaps with the fs-correct coupled
pulse-shape channel highlighted.

Sprint 7 makes :mod:`plot_style_v2` the publication source of truth (serif house
style, thin black spines, dedicated colorbar-gutter :func:`plot_style_v2.stacked_figure`)
and adds the journal figure set: the raw+scattering paired preview
(:func:`plot_raw_scatter_paired`), the latent / AM envelope-carrier decomposition
(:func:`plot_latent_am_decomposition`), the calibration / lag / frac_Phi diagnostics
panel (:func:`plot_diagnostics_panel`), and the TE-aware aggregate figures
(:func:`plot_calibration_by_lag`, :func:`plot_frac_phi_distribution`,
:func:`plot_lag_mass_summary`).

See ``SYNTHETIC_V2_SPEC_AND_SPRINTS.md`` Sprints 1, 2, 7.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    plot_style_v2 as ps,
)

# Apply the publication house style once at import so every figure below inherits
# the serif fonts, thin spines, and high-DPI raster defaults (S7-T01).
ps.apply_style()

# Semantic colour aliases (names kept for the Sprint 1/2 figures) drawn from the
# shared palette so the whole gallery stays colour-consistent.
_FHR_COLOR = ps.COLOR_BLUE
_UP_COLOR = ps.COLOR_VERMILLION
_BASELINE_COLOR = ps.COLOR_GRAY
_HIGHLIGHT_COLOR = ps.COLOR_SKY  # coupled-channel marker: contrasts against magma
_TRAIN_COLOR = ps.COLOR_BLUE
_VAL_COLOR = ps.COLOR_VERMILLION
_INJ_COLOR = ps.COLOR_BLUE       # TE_inj series
_SCAT_COLOR = ps.COLOR_VERMILLION  # TE_scat series
_BAND_COLOR = ps.COLOR_GREEN     # true lag band / reference

# Default raster resolution for PNG output across the gallery (PDF is vector).
_DPI = ps.SAVE_DPI


def _resolve_output_paths(out_path: Union[str, Path], formats: tuple) -> List[Path]:
    r"""Resolve a stem/path into concrete output paths for each requested format.

    Strips a trailing extension only when it is one of ``formats`` (so stems that
    legitimately contain dots, e.g. ``G1_te2.0_D8``, are preserved), then appends each
    format extension. Creates the parent directory.

    Args:
        out_path: Output path stem or a full path whose known suffix is stripped.
        formats: Output formats (e.g. ``("pdf", "png")``).

    Returns:
        One path per format.
    """
    out = Path(out_path)
    known = {f".{fmt.lower()}" for fmt in formats}
    stem = out.with_suffix("") if out.suffix.lower() in known else out
    stem.parent.mkdir(parents=True, exist_ok=True)
    return [stem.parent / f"{stem.name}.{fmt}" for fmt in formats]


def plot_raw_preview(
    fhr_raw: np.ndarray,
    up_raw: np.ndarray,
    out_path: Union[str, Path],
    *,
    meta: Optional[Dict[str, Any]] = None,
    fs: float = 4.0,
    sample: int = 0,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write an annotated raw FHR/UP preview figure (S1-T05).

    Renders one sample's raw waveforms as two stacked panels: FHR (bpm) on top with
    its baseline $\mu_{\mathrm{FHR}}$ marked, and UP (mmHg) below with its resting
    tone $\mu_{\mathrm{UP}}$ marked. The title carries the cell provenance
    ($\mathrm{TE}_{\mathrm{inj}}$, lag $D$, coupling $B$) when supplied in ``meta``.

    Note: with the default ``am_carrier`` render the FHR coupled term is a modulated
    sinusoid symmetric about the baseline (best for the sign-blind scattering modulus),
    not a one-sided clinical deceleration dip; the waveform-realistic ``pulse_train``
    variant lands in Sprint 7.

    Args:
        fhr_raw: FHR waveform(s), shape ``(n, N)`` or ``(N,)`` (bpm).
        up_raw: UP waveform(s), shape ``(n, N)`` or ``(N,)`` (mmHg).
        out_path: Output path stem (extensions in ``formats`` are appended) or a full
            path (its suffix is stripped and ``formats`` used).
        meta: Optional provenance dict (keys ``te_inj``, ``D``, ``B``, ``f_pulse``).
        fs: Raw sampling rate in Hz (for the time axis).
        sample: Row index to plot when the inputs are 2-D.
        formats: Output formats to write (e.g. ``("pdf", "png")``).
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    fhr = np.asarray(fhr_raw, dtype=float)
    up = np.asarray(up_raw, dtype=float)
    if fhr.ndim == 2:
        fhr = fhr[sample]
    if up.ndim == 2:
        up = up[sample]
    n = fhr.shape[-1]
    t_min = np.arange(n) / fs / 60.0

    meta = meta or {}
    te_inj = meta.get("te_inj")
    delay = meta.get("D")
    coupling = meta.get("B")
    bits = []
    if te_inj is not None:
        bits.append(rf"$\mathrm{{TE}}_{{\mathrm{{inj}}}}$ = {te_inj:g} nats")
    if delay is not None:
        bits.append(f"D = {delay} steps")
    if coupling is not None:
        bits.append(f"B = {coupling:.3g}")
    title = "synthetic_v2 raw preview" + (("   (" + ", ".join(bits) + ")") if bits else "")

    fig, (ax_fhr, ax_up) = plt.subplots(2, 1, figsize=(10.0, 5.0), sharex=True)

    ax_fhr.plot(t_min, fhr, color=_FHR_COLOR, lw=0.6)
    ax_fhr.axhline(float(fhr.mean()), color=_BASELINE_COLOR, lw=0.9, ls="--",
                   label=rf"baseline $\mu_{{\mathrm{{FHR}}}} \approx$ {fhr.mean():.0f} bpm")
    ax_fhr.set_ylabel("FHR (bpm)")
    ax_fhr.set_title(title)
    # Annotate the coupled deceleration band (the FHR target of the UP->FHR pathway).
    ax_fhr.text(0.01, 0.04, "coupled decel band (VLF, target)", transform=ax_fhr.transAxes,
                color=_HIGHLIGHT_COLOR, fontsize=6.5, va="bottom", ha="left")
    ax_fhr.legend(loc="upper right", frameon=False)

    ax_up.plot(t_min, up, color=_UP_COLOR, lw=0.6)
    ax_up.axhline(float(up.mean()), color=_BASELINE_COLOR, lw=0.9, ls="--",
                  label=rf"resting tone $\mu_{{\mathrm{{UP}}}} \approx$ {up.mean():.0f} mmHg")
    ax_up.set_ylabel("UP (mmHg)")
    ax_up.set_xlabel("time (min)")
    # Annotate the coupled contraction band (the UP source of the pathway).
    ax_up.text(0.01, 0.04, "coupled contraction band (source)", transform=ax_up.transAxes,
               color=_HIGHLIGHT_COLOR, fontsize=6.5, va="bottom", ha="left")
    ax_up.legend(loc="upper right", frameon=False)

    for ax in (ax_fhr, ax_up):
        ps.tighten_xaxis(ax, t_min)
        ps.style_axes(ax)

    fig.tight_layout()

    # Only strip a trailing extension if it is one of the requested output formats;
    # otherwise treat the whole path as the stem. This preserves stems that legitimately
    # contain dots (e.g. "G1_te2.0_D8"), which `Path.with_suffix` would corrupt. Build
    # each output path by appending the extension to the name rather than replacing a
    # (possibly spurious) suffix.
    written: List[Path] = []
    for path in _resolve_output_paths(out_path, formats):
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def _to_channels_time(field: np.ndarray, sample: int) -> np.ndarray:
    r"""Coerce a feature field to a single sample's channels-first $(C, T)$ heatmap array.

    Accepts model-facing $(n, T, C)$ or $(T, C)$ layouts and returns $(C, T)$.

    Args:
        field: Normalised feature array $(n, T, C)$ or $(T, C)$.
        sample: Row index to plot when the input is 3-D.

    Returns:
        A $(C, T)$ array (channels on the first axis).
    """
    arr = np.asarray(field, dtype=float)
    if arr.ndim == 3:
        arr = arr[sample]
    if arr.ndim != 2:
        raise ValueError(f"expected (n, T, C) or (T, C), got shape {arr.shape}")
    return arr.T  # (T, C) -> (C, T)


def plot_scattering_heatmap(
    fhr_st: np.ndarray,
    up_st: np.ndarray,
    out_path: Union[str, Path],
    *,
    coupled_idx: Optional[int] = None,
    center_freqs: Optional[np.ndarray] = None,
    fs: float = 4.0,
    sample: int = 0,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write stacked FHR / UP scattering-coefficient heatmaps (S2-T04).

    Renders the $43$-channel first-order scattering fields as two stacked
    channel$\times$time heatmaps (FHR on top, UP below) sharing one colorbar in a
    dedicated gutter (so the panel rows are not shrunk by per-axes colorbars). The
    fs-correct coupled pulse-shape channel (``coupled_idx``) is marked on both panels.
    When ``center_freqs`` (normalised $\xi$) is supplied, y-tick labels are shown in Hz
    (physical Hz $= \xi\,f_s$); channel $0$ is the order-0 low-pass baseline (``S0``).

    Args:
        fhr_st: FHR scattering field, $(n, T, C)$ or $(T, C)$ (normalised).
        up_st: UP scattering field, $(n, T, C)$ or $(T, C)$ (normalised).
        out_path: Output path stem (formats appended) or a full path.
        coupled_idx: Scattering channel index carrying the coupled carrier (highlighted).
        center_freqs: Normalised $\xi$ centre frequencies of the $C-1$ first-order
            channels (for Hz y-labels); channel $0$ is order-0.
        fs: Raw sampling rate in Hz (converts $\xi$ to Hz and steps to minutes).
        sample: Row index to plot when inputs are 3-D.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    fhr = _to_channels_time(fhr_st, sample)  # (C, T)
    up = _to_channels_time(up_st, sample)
    n_ch, n_t = fhr.shape
    # Decimated step = 16 raw samples; time axis in minutes.
    step_s = 16.0 / fs
    t_max_min = n_t * step_s / 60.0

    # Shared, robust colour scale across both panels (features are z-scored).
    both = np.concatenate([fhr.ravel(), up.ravel()])
    vmin, vmax = np.percentile(both, [1.0, 99.0])
    if vmin == vmax:
        vmin, vmax = float(both.min()), float(both.max() + 1e-6)

    fig = plt.figure(figsize=(10.0, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.03], hspace=0.18, wspace=0.03)
    ax_fhr = fig.add_subplot(gs[0, 0])
    ax_up = fig.add_subplot(gs[1, 0], sharex=ax_fhr)
    cax = fig.add_subplot(gs[:, 1])

    im = None
    for ax, data, name in ((ax_fhr, fhr, "FHR"), (ax_up, up, "UP")):
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=(0.0, t_max_min, -0.5, n_ch - 0.5),
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
            interpolation="nearest",
        )
        ax.set_ylabel(f"{name} scat. channel")
        if coupled_idx is not None:
            ax.axhline(coupled_idx, color=_HIGHLIGHT_COLOR, lw=1.2, ls="--")
            ax.text(
                0.01 * t_max_min,
                coupled_idx + 0.6,
                f"coupled ch {coupled_idx}",
                color=_HIGHLIGHT_COLOR,
                va="bottom",
                ha="left",
                fontsize=7.5,
            )
        # Keep the full thin-black box (house style); no grid on heatmaps.

    # Hz y-tick labels when the centre frequencies are supplied.
    if center_freqs is not None:
        cf = np.asarray(center_freqs, dtype=float)
        tick_ch = [0] + list(range(5, n_ch, 8))
        labels = []
        for ch in tick_ch:
            if ch == 0:
                labels.append("S0")
            else:
                labels.append(f"{cf[ch - 1] * fs:.3f}")
        for ax in (ax_fhr, ax_up):
            ax.set_yticks(tick_ch)
            ax.set_yticklabels(labels)
        ax_fhr.set_ylabel("FHR channel (Hz)")
        ax_up.set_ylabel("UP channel (Hz)")

    ax_fhr.set_title("synthetic_v2 scattering coefficients (normalised)", fontsize=10.0)
    ax_up.set_xlabel("time (min)")
    plt.setp(ax_fhr.get_xticklabels(), visible=False)
    fig.colorbar(im, cax=cax, label="z-scored magnitude")

    written: List[Path] = []
    for path in _resolve_output_paths(out_path, formats):
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def _read_metric_series(rows: List[Dict[str, str]], metric: str) -> tuple:
    r"""Extract a per-epoch ``(epochs, values)`` series for one metric from CSV rows.

    Resolves the actual Lightning column name for ``metric`` (which logs both a
    ``<metric>_step`` and a ``<metric>_epoch`` column under ``on_step=on_epoch=True``)
    by trying ``metric``, then ``metric_epoch``, then ``metric_step``. The last
    non-empty value seen per epoch wins (epoch-aggregated rows overwrite step rows).

    Args:
        rows: The parsed ``metrics.csv`` rows (from :class:`csv.DictReader`).
        metric: The logical metric name (e.g. ``train/total_loss``).

    Returns:
        ``(epochs, values)`` as sorted parallel lists; empty when the metric is
        absent from every row.
    """
    if not rows:
        return [], []
    columns = rows[0].keys()
    col = None
    for candidate in (metric, f"{metric}_epoch", f"{metric}_step"):
        if candidate in columns:
            col = candidate
            break
    if col is None:
        return [], []

    per_epoch: Dict[int, float] = {}
    for i, row in enumerate(rows):
        raw = row.get(col, "")
        if raw is None or raw == "":
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        epoch_raw = row.get("epoch", "")
        try:
            epoch = int(float(epoch_raw)) if epoch_raw not in (None, "") else i
        except (TypeError, ValueError):
            epoch = i
        per_epoch[epoch] = value  # last write per epoch wins

    epochs = sorted(per_epoch)
    return epochs, [per_epoch[e] for e in epochs]


def plot_loss_curves(
    metrics_csv: Union[str, Path],
    out_path: Union[str, Path],
    *,
    title: Optional[str] = None,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write the training loss / KL curves from a Lightning ``metrics.csv`` (S5-T04).

    Two panels: (left) the total loss $\mathcal L$ with the residual
    ($\mathcal L_{\mathrm{feat}}$) and baseline ($\mathcal L_{\mathrm{base}}$)
    components for train (and val when present); (right) the dim-summed KL
    ``kld_nats`` $= \mathrm{KL}\cdot d_z$ -- the $\bar K$ TE-surrogate scale -- so
    posterior collapse (``kld_nats`` $\to 0$) is visible at a glance.

    Args:
        metrics_csv: Path to the Lightning ``CSVLogger`` ``metrics.csv``.
        out_path: Output path stem (formats appended) or a full path.
        title: Optional figure suptitle.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.

    Raises:
        FileNotFoundError: If ``metrics_csv`` does not exist.
    """
    metrics_csv = Path(metrics_csv)
    if not metrics_csv.is_file():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_csv}")
    with open(metrics_csv, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    fig, (ax_loss, ax_kl) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # --- panel 1: total / feature / baseline losses -------------------------
    loss_specs = (
        ("train/total_loss", _TRAIN_COLOR, "-", "train total"),
        ("val/total_loss", _VAL_COLOR, "-", "val total"),
        ("train/feat_loss", _TRAIN_COLOR, "--", "train feat"),
        ("train/base_loss", _BASELINE_COLOR, ":", "train base"),
    )
    plotted_loss = False
    for metric, color, ls, label in loss_specs:
        epochs, values = _read_metric_series(rows, metric)
        if epochs:
            ax_loss.plot(epochs, values, color=color, ls=ls, lw=1.2, label=label)
            plotted_loss = True
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("loss", fontsize=10.0)
    if plotted_loss:
        ax_loss.legend(loc="upper right", fontsize=7.5, frameon=False)

    # --- panel 2: kld_nats (the K-bar surrogate scale) ----------------------
    plotted_kl = False
    for metric, color, label in (
        ("train/kld_nats", _TRAIN_COLOR, "train"),
        ("val/kld_nats", _VAL_COLOR, "val"),
    ):
        epochs, values = _read_metric_series(rows, metric)
        if epochs:
            ax_kl.plot(epochs, values, color=color, lw=1.2, label=label)
            plotted_kl = True
    ax_kl.set_xlabel("epoch")
    ax_kl.set_ylabel(r"$\bar K$  (kld_nats, nats/step)")
    ax_kl.set_title("KL divergence (TE surrogate)", fontsize=10.0)
    if plotted_kl:
        ax_kl.legend(loc="upper right", fontsize=7.5, frameon=False)

    for ax in (ax_loss, ax_kl):
        ax.margins(x=0.02)
        ps.style_axes(ax)

    fig.suptitle(title or "synthetic_v2 training curves", fontsize=ps.FONT_SUPTITLE)
    fig.tight_layout()

    written: List[Path] = []
    for path in _resolve_output_paths(out_path, formats):
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def plot_loss_curves_html(
    metrics_csv: Union[str, Path],
    out_path: Union[str, Path],
    *,
    title: Optional[str] = None,
    include_plotlyjs: bool = True,
) -> Optional[List[Path]]:
    r"""Write an interactive Plotly HTML loss / KL curve from a Lightning ``metrics.csv``.

    The interactive twin of :func:`plot_loss_curves`: the same two panels -- (left) the
    total loss $\mathcal L$ with its residual ($\mathcal L_{\mathrm{feat}}$) and baseline
    ($\mathcal L_{\mathrm{base}}$) components, and (right) the dim-summed KL ``kld_nats``
    $= \mathrm{KL}\cdot d_z$ (the $\bar K$ TE-surrogate) -- and the same palette, rendered
    to a single ``.html`` that opens offline in a browser. It is consumed live during
    training by
    :class:`~model.vae_teb_prediction.model.model_experiment.synthetic_v2.callbacks_v2.LossPlotHtmlCallback`,
    which rewrites it every few epochs so a long run can be watched mid-flight.

    Plotly is an optional dependency: if it is not importable this warns and returns
    ``None`` rather than raising, so a missing install never breaks training. Series are
    read via :func:`_read_metric_series`, which resolves the Lightning ``_epoch`` column
    suffix and de-duplicates per epoch.

    Args:
        metrics_csv: Path to the Lightning ``CSVLogger`` ``metrics.csv``.
        out_path: Output path stem (``.html`` appended) or a full ``.html`` path.
        title: Optional figure title.
        include_plotlyjs: When ``True`` (default) embed ``plotly.js`` in the file so it
            is fully self-contained and opens without a network connection; when
            ``False`` reference the CDN (smaller file, needs internet).

    Returns:
        The single-element list with the written HTML path, or ``None`` when plotly is
        unavailable or ``metrics_csv`` does not exist.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - optional dependency
        logger.warning(
            "[visualize_v2] plotly unavailable ({}); skipping HTML loss curve.", exc
        )
        return None

    metrics_csv = Path(metrics_csv)
    if not metrics_csv.is_file():
        logger.warning(
            "[visualize_v2] metrics.csv not found ({}); skipping HTML loss curve.",
            metrics_csv,
        )
        return None
    with open(metrics_csv, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("loss", "KL divergence (TE surrogate)"),
        horizontal_spacing=0.10,
    )

    # --- panel 1: total / feature / baseline losses -------------------------
    loss_specs = (
        ("train/total_loss", _TRAIN_COLOR, "solid", "train total"),
        ("val/total_loss", _VAL_COLOR, "solid", "val total"),
        ("train/feat_loss", _TRAIN_COLOR, "dash", "train feat"),
        ("train/base_loss", _BASELINE_COLOR, "dot", "train base"),
    )
    for metric, color, dash, label in loss_specs:
        epochs, values = _read_metric_series(rows, metric)
        if epochs:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=values,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, dash=dash, width=1.6),
                    marker=dict(size=4),
                    legendgroup="loss",
                ),
                row=1,
                col=1,
            )

    # --- panel 2: kld_nats (the K-bar surrogate scale) ----------------------
    for metric, color, label in (
        ("train/kld_nats", _TRAIN_COLOR, "train"),
        ("val/kld_nats", _VAL_COLOR, "val"),
    ):
        epochs, values = _read_metric_series(rows, metric)
        if epochs:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=values,
                    mode="lines+markers",
                    name=f"kld {label}",
                    line=dict(color=color, width=1.6),
                    marker=dict(size=4),
                    legendgroup="kl",
                ),
                row=1,
                col=2,
            )

    fig.update_xaxes(title_text="epoch", row=1, col=1)
    fig.update_xaxes(title_text="epoch", row=1, col=2)
    fig.update_yaxes(title_text="loss", row=1, col=1)
    fig.update_yaxes(title_text="kld_nats (nats/step)", row=1, col=2)
    fig.update_layout(
        title=title or "synthetic_v2 training curves",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.10, xanchor="left", x=0.0
        ),
        margin=dict(l=60, r=30, t=90, b=50),
    )

    out_html = _resolve_output_paths(out_path, ("html",))[0]
    fig.write_html(str(out_html), include_plotlyjs=include_plotlyjs)
    return [out_html]


def _save_fig(
    fig, out_path: Union[str, Path], formats: tuple, dpi: int, *, close: bool = True
) -> List[Path]:
    r"""Write ``fig`` to every requested format via :func:`_resolve_output_paths`.

    Uses the dotted-stem-safe path resolver (so stems like ``G1_te2.0_D8`` are not
    corrupted by ``Path.with_suffix``), unlike :func:`plot_style_v2.save_figure`.

    Args:
        fig: The figure to save.
        out_path: Output stem or full path (a known suffix is stripped).
        formats: Output formats to write (e.g. ``("pdf", "png")``).
        dpi: Raster DPI for PNG output.
        close: Whether to close the figure afterwards.

    Returns:
        The list of written file paths.
    """
    written: List[Path] = []
    for path in _resolve_output_paths(out_path, formats):
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    if close:
        plt.close(fig)
    return written


def _overlay_latent_on_channel(
    ax, t_min: np.ndarray, latent: np.ndarray, coupled_idx: int, *,
    color: str, span: float = 2.5, label: Optional[str] = None,
) -> None:
    r"""Overlay a z-scored latent as a thin line riding the coupled channel row.

    The latent is z-scored and rescaled to $\pm$``span`` channel-rows around
    ``coupled_idx`` so it sits on the highlighted scattering channel and its tracking
    is visible against the heatmap. Purely cosmetic (no axis rescale of the heatmap).

    Args:
        ax: The heatmap axes (channels on the y-axis, minutes on x).
        t_min: The heatmap time axis in minutes, length matching ``latent``.
        latent: The decimated latent slice aligned to the feature grid (``[15:315]``).
        coupled_idx: The scattering channel index the latent is overlaid on.
        color: Line colour.
        span: Half-height of the overlay in channel-rows.
        label: Optional legend label.
    """
    lat = np.asarray(latent, dtype=float).reshape(-1)
    n = min(lat.shape[0], t_min.shape[0])
    lat = lat[:n]
    std = float(lat.std())
    if std <= 0.0:
        return
    y = coupled_idx + span * (lat - lat.mean()) / (std + 1e-12)
    ax.plot(t_min[:n], y, color=color, lw=0.8, alpha=0.9, label=label)


def plot_raw_scatter_paired(
    fhr_raw: np.ndarray,
    up_raw: np.ndarray,
    fhr_st: np.ndarray,
    up_st: np.ndarray,
    out_path: Union[str, Path],
    *,
    coupled_idx: int,
    latent_c: Optional[np.ndarray] = None,
    latent_d: Optional[np.ndarray] = None,
    center_freqs: Optional[np.ndarray] = None,
    fs: float = 4.0,
    trim: int = 15,
    meta: Optional[Dict[str, Any]] = None,
    sample: int = 0,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write the headline raw + scattering paired preview (S7-T02).

    Four stacked panels sharing aligned colorbar gutters: raw FHR trace, FHR scattering
    heatmap, raw UP trace, UP scattering heatmap. The fs-correct coupled pulse-shape
    channel (``coupled_idx``) is highlighted on both heatmaps, and the decimated latent
    (sliced to the feature grid ``[trim:trim+T]``) is overlaid on that channel so its
    tracking is visible on a strong cell.

    The caller supplies the already-transformed, normalised ``*_st`` fields (so this
    module stays free of the torch / kymatio transform).

    Args:
        fhr_raw: FHR waveform(s), $(n, N)$ or $(N,)$ (bpm).
        up_raw: UP waveform(s), $(n, N)$ or $(N,)$ (mmHg).
        fhr_st: Normalised FHR scattering field, $(n, T, C)$ or $(T, C)$.
        up_st: Normalised UP scattering field, $(n, T, C)$ or $(T, C)$.
        out_path: Output path stem or full path.
        coupled_idx: The scattering channel index carrying the coupled carrier.
        latent_c: Source latent $c$ on the decimated grid, $(n, T_{\mathrm{tot}})$ or
            $(T_{\mathrm{tot}},)$; overlaid on the UP heatmap when supplied.
        latent_d: Target latent $d$, same layout; overlaid on the FHR heatmap.
        center_freqs: Normalised $\xi$ centre frequencies for Hz y-labels.
        fs: Raw sampling rate in Hz.
        trim: Symmetric decimated trim per end (feature grid = ``latent[trim:trim+T]``).
        meta: Optional provenance dict (keys ``te_inj``, ``D``, ``B``) for the title.
        sample: Row index to plot when inputs are batched.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    fhr = np.asarray(fhr_raw, dtype=float)
    up = np.asarray(up_raw, dtype=float)
    if fhr.ndim == 2:
        fhr = fhr[sample]
    if up.ndim == 2:
        up = up[sample]
    fhr_cf = _to_channels_time(fhr_st, sample)  # (C, T)
    up_cf = _to_channels_time(up_st, sample)
    n_ch, n_t = fhr_cf.shape

    n_raw = fhr.shape[-1]
    t_raw_min = np.arange(n_raw) / fs / 60.0
    step_s = 16.0 / fs
    t_dec_min = (np.arange(n_t) * step_s) / 60.0

    both = np.concatenate([fhr_cf.ravel(), up_cf.ravel()])
    vmin, vmax = np.percentile(both, [1.0, 99.0])
    if vmin == vmax:
        vmin, vmax = float(both.min()), float(both.max() + 1e-6)

    fig, axes, caxes = ps.stacked_figure(
        [1.1, 1.6, 1.1, 1.6],
        width=11.0,
        colorbar=[False, True, False, True],
        hspace=0.55,
    )
    ax_fhr_raw, ax_fhr_st, ax_up_raw, ax_up_st = axes

    # --- raw traces ---
    ax_fhr_raw.plot(t_raw_min, fhr, color=_FHR_COLOR, lw=0.5)
    ax_fhr_raw.axhline(float(fhr.mean()), color=_BASELINE_COLOR, lw=0.8, ls="--")
    ax_fhr_raw.set_ylabel("FHR (bpm)")
    ax_up_raw.plot(t_raw_min, up, color=_UP_COLOR, lw=0.5)
    ax_up_raw.axhline(float(up.mean()), color=_BASELINE_COLOR, lw=0.8, ls="--")
    ax_up_raw.set_ylabel("UP (mmHg)")
    for ax in (ax_fhr_raw, ax_up_raw):
        ps.tighten_xaxis(ax, t_raw_min)
        ps.style_axes(ax)

    # --- scattering heatmaps with the coupled channel highlighted + latent overlay ---
    for ax, cax, data, name, latent in (
        (ax_fhr_st, caxes[1], fhr_cf, "FHR", latent_d),
        (ax_up_st, caxes[3], up_cf, "UP", latent_c),
    ):
        im = ax.imshow(
            data, aspect="auto", origin="lower",
            extent=(0.0, t_dec_min[-1] if n_t else 0.0, -0.5, n_ch - 0.5),
            vmin=vmin, vmax=vmax, cmap="magma", interpolation="nearest",
        )
        ax.axhline(coupled_idx, color=_HIGHLIGHT_COLOR, lw=1.0, ls="--")
        ax.set_ylabel(f"{name} scat. channel")
        if latent is not None:
            lat = np.asarray(latent, dtype=float)
            if lat.ndim == 2:
                lat = lat[sample]
            sliced = lat[trim:trim + n_t]
            _overlay_latent_on_channel(
                ax, t_dec_min, sliced, coupled_idx, color=_HIGHLIGHT_COLOR,
                label="latent (z, on coupled ch)",
            )
            ax.legend(loc="upper right", frameon=False, fontsize=6.5)
        if cax is not None:
            ps.attach_colorbar(fig, im, cax, label="z-scored magnitude")

    if center_freqs is not None:
        cf = np.asarray(center_freqs, dtype=float)
        tick_ch = [0, coupled_idx] + list(range(8, n_ch, 12))
        tick_ch = sorted(set(t for t in tick_ch if 0 <= t < n_ch))
        labels = ["S0" if ch == 0 else f"{cf[ch - 1] * fs:.3f}" for ch in tick_ch]
        for ax in (ax_fhr_st, ax_up_st):
            ax.set_yticks(tick_ch)
            ax.set_yticklabels(labels)

    ax_up_st.set_xlabel("time (min)")

    meta = meta or {}
    bits = []
    if meta.get("te_inj") is not None:
        bits.append(rf"$\mathrm{{TE}}_{{\mathrm{{inj}}}}$={meta['te_inj']:g} nats")
    if meta.get("D") is not None:
        bits.append(f"D={meta['D']}")
    title = f"synthetic_v2 raw + scattering (coupled channel {coupled_idx})"
    fig.suptitle("   ".join([title] + bits) if bits else title, fontsize=ps.FONT_SUPTITLE)
    return _save_fig(fig, out_path, formats, dpi)


def _latent_row(latents: Dict[str, np.ndarray], key: str, sample: int) -> np.ndarray:
    r"""Extract one sample's 1-D array for ``key`` from a ``generate_cell_raw`` latents dict."""
    arr = np.asarray(latents[key], dtype=float)
    return arr[sample] if arr.ndim == 2 else arr


def plot_latent_am_decomposition(
    latents: Dict[str, np.ndarray],
    out_path: Union[str, Path],
    *,
    fs: float = 4.0,
    f_pulse: float = 0.06,
    sample: int = 0,
    meta: Optional[Dict[str, Any]] = None,
    zoom_min: float = 2.0,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write the latent-pair + AM envelope/carrier decomposition figure (S7-T03).

    Four stacked panels that make the §7 amplitude-modulation rendering legible: the
    decimated coupled latents $c$ (source) and $d$ (target); their strictly-positive
    amplitude envelopes $A_u, A_y$ on the raw grid; a zoomed view of the pulse-shape
    carrier near $f_{\mathrm{pulse}}$; and the rendered coupled bands $u_{\mathrm c} = A_u\,g$,
    $y_{\mathrm d} = A_y\,g$. Reads the ``latents`` dict returned by
    :func:`raw_generators.generate_cell_raw` (keys ``c, d, A_u, A_y, carrier_u,
    carrier_y, u_c, y_d``).

    Args:
        latents: The ``latents`` dict from :func:`raw_generators.generate_cell_raw`.
        out_path: Output path stem or full path.
        fs: Raw sampling rate in Hz.
        f_pulse: Carrier frequency in Hz (annotated on the carrier panel).
        sample: Row index to plot when the arrays are batched.
        meta: Optional provenance dict (keys ``te_inj``, ``D``, ``render_mode``) for the title.
        zoom_min: Width in minutes of the carrier zoom window.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    c = _latent_row(latents, "c", sample)
    d = _latent_row(latents, "d", sample)
    a_u = _latent_row(latents, "A_u", sample)
    a_y = _latent_row(latents, "A_y", sample)
    carrier_u = _latent_row(latents, "carrier_u", sample)
    u_c = _latent_row(latents, "u_c", sample)
    y_d = _latent_row(latents, "y_d", sample)

    t_tot = c.shape[-1]
    n_raw = a_u.shape[-1]
    step_s = 16.0 / fs
    t_dec_min = np.arange(t_tot) * step_s / 60.0
    t_raw_min = np.arange(n_raw) / fs / 60.0
    zoom = t_raw_min <= zoom_min

    # Lower the top margin so the figure suptitle clears the first panel's own title
    # (the default 0.955 top leaves them overlapping in a stacked layout).
    fig, axes, _ = ps.stacked_figure([1.0, 1.0, 1.0, 1.0], width=10.0, hspace=0.6,
                                     colorbar=False, margins=(0.10, 0.95, 0.90, 0.045))
    ax_lat, ax_env, ax_car, ax_ren = axes

    ax_lat.plot(t_dec_min, c, color=_INJ_COLOR, lw=0.9, label=r"$c$ (source, UP)")
    ax_lat.plot(t_dec_min, d, color=_SCAT_COLOR, lw=0.9, label=r"$d$ (target, FHR)")
    ax_lat.set_ylabel("latent")
    ax_lat.set_title("coupled latent pair (decimated grid)")
    ax_lat.legend(loc="upper right", frameon=False, ncol=2)

    ax_env.plot(t_raw_min, a_u, color=_INJ_COLOR, lw=0.7, label=r"$A_u$ (UP envelope)")
    ax_env.plot(t_raw_min, a_y, color=_SCAT_COLOR, lw=0.7, label=r"$A_y$ (FHR envelope)")
    ax_env.axhline(0.0, color=_BASELINE_COLOR, lw=0.6, ls=":")
    ax_env.set_ylabel("envelope")
    ax_env.set_title(r"strictly-positive AM envelopes $A = a_0 + a_1\,\tilde x$")
    ax_env.legend(loc="upper right", frameon=False, ncol=2)

    ax_car.plot(t_raw_min[zoom], carrier_u[zoom], color=_BASELINE_COLOR, lw=0.7)
    ax_car.set_ylabel("carrier")
    ax_car.set_title(rf"pulse-shape carrier $g$ near $f_{{\mathrm{{pulse}}}}$={f_pulse:g} Hz "
                     f"(first {zoom_min:g} min)")

    ax_ren.plot(t_raw_min, u_c, color=_INJ_COLOR, lw=0.5,
                label=r"$u_{\mathrm{c}}=A_u\,g$ (UP band)")
    ax_ren.plot(t_raw_min, y_d, color=_SCAT_COLOR, lw=0.5,
                label=r"$y_{\mathrm{d}}=A_y\,g$ (FHR band)")
    ax_ren.set_ylabel("rendered band")
    ax_ren.set_xlabel("time (min)")
    ax_ren.set_title("rendered coupled bands (added to / subtracted from the raw signal)")
    ax_ren.legend(loc="upper right", frameon=False, ncol=2)

    for ax, x in ((ax_lat, t_dec_min), (ax_env, t_raw_min),
                  (ax_car, t_raw_min[zoom]), (ax_ren, t_raw_min)):
        ps.tighten_xaxis(ax, x)
        ps.style_axes(ax)

    meta = meta or {}
    bits = []
    if meta.get("render_mode"):
        bits.append(f"render={meta['render_mode']}")
    if meta.get("te_inj") is not None:
        bits.append(rf"$\mathrm{{TE}}_{{\mathrm{{inj}}}}$={meta['te_inj']:g}")
    if meta.get("D") is not None:
        bits.append(f"D={meta['D']}")
    suptitle = "synthetic_v2 latent / AM decomposition"
    if bits:
        suptitle += "   (" + ", ".join(bits) + ")"
    fig.suptitle(suptitle, fontsize=ps.FONT_SUPTITLE, y=0.985)
    return _save_fig(fig, out_path, formats, dpi)


def _per_cell_arrays(metrics: Dict[str, Any]) -> Dict[str, np.ndarray]:
    r"""Collect the per-cell diagnostics columns from a ``metrics.json`` dict.

    Args:
        metrics: The dict written by :func:`eval_v2.run_eval` (its ``per_cell`` list).

    Returns:
        A dict of parallel float arrays keyed by column name (``te_inj``, ``te_scat``,
        ``kbar_mean``, ``frac_phi``, ``lag_mass``, ``D``, ``n``) plus any per-control
        null-ratio columns (``null_<ctrl>_ratio``). Missing entries are ``nan``.
    """
    rows = metrics.get("per_cell", []) or []
    keys = ["cell_id", "te_inj", "te_scat", "D", "kbar_mean", "n", "frac_phi",
            "pred_gain", "uplift_rel", "lag_mass", "peak_lag_err"]
    null_keys = sorted({k for r in rows for k in r if str(k).startswith("null_")
                        and str(k).endswith("_ratio")})
    out: Dict[str, np.ndarray] = {}
    for key in keys + null_keys:
        vals = []
        for r in rows:
            v = r.get(key)
            try:
                vals.append(float(v) if v is not None else np.nan)
            except (TypeError, ValueError):
                vals.append(np.nan)
        out[key] = np.asarray(vals, dtype=float)
    return out


def plot_diagnostics_panel(
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    realizability: Optional[Dict[str, Any]] = None,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write the calibration / preservation / lag / null diagnostics panel (S7-T03).

    A $2\times2$ evidence panel from the ``metrics.json`` produced by
    :func:`eval_v2.run_eval`:

    1. **Calibration**: per-cell $\bar K$ vs $\mathrm{TE}_{\mathrm{inj}}$ and
       $\mathrm{TE}_{\mathrm{scat}}$ with the fitted lines $\bar K = \alpha + \gamma\,\mathrm{TE}$
       (slope/intercept/$R^2$ and cell count $n$ annotated).
    2. **Preservation**: per-cell $\mathrm{frac}_\Phi$ bars with the ideal-1 reference.
    3. **Lag recovery**: per-cell LagMass bars with the pass threshold.
    4. **Null control**: per-cell null-ratio bars ($\to 0$ for a signal-using model).

    Args:
        metrics: The dict written by :func:`eval_v2.run_eval`.
        out_path: Output path stem or full path.
        realizability: Optional ``realizability.json`` dict (currently unused; reserved
            for overlaying the build-time TE_raw trend).
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    cal = metrics.get("calibration", {}) or {}
    lag = metrics.get("lag_recovery", {}) or {}
    pc = _per_cell_arrays(metrics)

    fig, ax = plt.subplots(2, 2, figsize=(10.0, 7.5))
    (ax_cal, ax_frac), (ax_lag, ax_null) = ax

    # --- (1) calibration scatter + fitted lines -----------------------------
    te_inj, te_scat, kbar = pc["te_inj"], pc["te_scat"], pc["kbar_mean"]
    ax_cal.scatter(te_inj, kbar, s=16, color=_INJ_COLOR, label=r"$\mathrm{TE}_{\mathrm{inj}}$",
                   zorder=3)
    ax_cal.scatter(te_scat, kbar, s=16, color=_SCAT_COLOR, marker="s",
                   label=r"$\mathrm{TE}_{\mathrm{scat}}$", zorder=3)
    finite_te = np.concatenate([te_inj[np.isfinite(te_inj)], te_scat[np.isfinite(te_scat)]])
    if finite_te.size:
        xs = np.linspace(0.0, float(np.nanmax(finite_te)) * 1.05 + 1e-6, 50)
        for pref, color in (("inj", _INJ_COLOR), ("scat", _SCAT_COLOR)):
            g, a = cal.get(f"gamma_{pref}"), cal.get(f"alpha_{pref}")
            if g is not None and a is not None and np.isfinite(g) and np.isfinite(a):
                ax_cal.plot(xs, a + g * xs, color=color, lw=1.0,
                            label=rf"$\gamma_{{\mathrm{{{pref}}}}}$={g:.3f}, $R^2$="
                                  f"{cal.get(f'r2_{pref}', float('nan')):.2f}")
    ax_cal.set_xlabel("TE (nats)")
    ax_cal.set_ylabel(r"$\bar K$ (nats/step)")
    ax_cal.set_title(rf"$\gamma$-calibration ($n$={cal.get('n_cells', len(kbar))} cells)")
    ax_cal.legend(loc="upper left", frameon=False, fontsize=6.5)

    # --- (2) frac_Phi per cell ----------------------------------------------
    _cell_bar(ax_frac, pc["frac_phi"], pc["cell_id"], color=_INJ_COLOR,
              ylabel=r"$\mathrm{frac}_\Phi$", title=r"preservation $\mathrm{frac}_\Phi$",
              ref=1.0, ref_label="ideal 1")

    # --- (3) LagMass per cell -----------------------------------------------
    thr = lag.get("lag_mass_threshold")
    _cell_bar(ax_lag, pc["lag_mass"], pc["cell_id"], color=_BAND_COLOR,
              ylabel="LagMass", title="lag recovery (attention in $\\mathcal{L}^\\star$)",
              ref=float(thr) if thr is not None else None, ref_label="threshold")

    # --- (4) null ratio per cell --------------------------------------------
    null_cols = [k for k in pc if k.startswith("null_") and k.endswith("_ratio")]
    if null_cols:
        col = null_cols[0]
        _cell_bar(ax_null, pc[col], pc["cell_id"], color=_SCAT_COLOR,
                  ylabel="null ratio", title=f"null control ({col})  $\\to 0$", ref=0.0,
                  ref_label=None)
    else:
        ax_null.text(0.5, 0.5, "no null controls in metrics", ha="center", va="center",
                     transform=ax_null.transAxes, color=_BASELINE_COLOR)
        ax_null.set_title("null control")

    for a in (ax_cal, ax_frac, ax_lag, ax_null):
        ps.style_axes(a)
    fig.suptitle(f"synthetic_v2 diagnostics  (run {metrics.get('run_tag', '?')}, "
                 f"split {metrics.get('split', '?')})", fontsize=ps.FONT_SUPTITLE)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_fig(fig, out_path, formats, dpi)


def _cell_bar(
    ax, values: np.ndarray, cell_ids: np.ndarray, *, color: str, ylabel: str,
    title: str, ref: Optional[float] = None, ref_label: Optional[str] = None,
) -> None:
    r"""Draw a per-cell bar chart with an optional reference line (diagnostics helper).

    Args:
        ax: The axes to draw on.
        values: Per-cell values (``nan`` entries are dropped).
        cell_ids: Per-cell ids for the x tick labels.
        color: Bar colour.
        ylabel: Y-axis label.
        title: Panel title.
        ref: Optional horizontal reference line value.
        ref_label: Optional legend label for the reference line.
    """
    vals = np.asarray(values, dtype=float)
    ids = np.asarray(cell_ids, dtype=float)
    keep = np.isfinite(vals)
    x = np.arange(int(keep.sum()))
    ax.bar(x, vals[keep], color=color, width=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(i)}" if np.isfinite(i) else "?" for i in ids[keep]],
                       fontsize=6.0, rotation=0)
    ax.set_xlabel("cell id")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ref is not None:
        ax.axhline(ref, color=_BASELINE_COLOR, lw=0.8, ls="--",
                   label=ref_label)
        if ref_label:
            ax.legend(loc="upper right", frameon=False, fontsize=6.5)


def _group_stats(x: np.ndarray, y: np.ndarray) -> Dict[float, Dict[str, float]]:
    r"""Per-group mean / std-error / count of ``y`` keyed by the unique values of ``x``.

    Args:
        x: Group key per point (e.g. the lag $D$); non-finite pairs are dropped.
        y: Value per point (e.g. $\bar K$).

    Returns:
        ``{group: {'mean', 'sem', 'n'}}`` sorted implicitly by the caller.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    out: Dict[float, Dict[str, float]] = {}
    for g in np.unique(x[keep]):
        vals = y[keep][x[keep] == g]
        n = int(vals.size)
        sem = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        out[float(g)] = {"mean": float(vals.mean()), "sem": sem, "n": n}
    return out


def plot_calibration_by_lag(
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write grouped $\bar K$-vs-TE calibration scatter coloured by lag cell (S7-T08).

    Population-level evidence (not an example): per-cell $\bar K$ against
    $\mathrm{TE}_{\mathrm{inj}}$ (left) and $\mathrm{TE}_{\mathrm{scat}}$ (right), each point
    coloured by its lag $D$, with the fitted calibration line, the $y=x$ reference, and the
    per-cell count annotated. Degrades gracefully to a single group when ``D`` is absent.

    Args:
        metrics: The ``metrics.json`` dict from :func:`eval_v2.run_eval`.
        out_path: Output path stem or full path.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    cal = metrics.get("calibration", {}) or {}
    pc = _per_cell_arrays(metrics)
    D = pc["D"]
    lags = np.unique(D[np.isfinite(D)])
    palette = ps.PALETTE_EXTENDED

    fig, (ax_inj, ax_scat) = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, te_key, pref, label in (
        (ax_inj, "te_inj", "inj", r"$\mathrm{TE}_{\mathrm{inj}}$"),
        (ax_scat, "te_scat", "scat", r"$\mathrm{TE}_{\mathrm{scat}}$"),
    ):
        te, kbar = pc[te_key], pc["kbar_mean"]
        if lags.size:
            for j, lag in enumerate(lags):
                m = np.isfinite(D) & (D == lag)
                ax.scatter(te[m], kbar[m], s=20, color=palette[j % len(palette)],
                           label=f"D={int(lag)} (n={int(m.sum())})", zorder=3)
        else:
            # No lag column -> degrade gracefully to a single pooled group (per the docstring).
            m = np.isfinite(te)
            ax.scatter(te[m], kbar[m], s=20, color=palette[0],
                       label=f"all cells (n={int(m.sum())})", zorder=3)
        finite = te[np.isfinite(te)]
        if finite.size:
            xs = np.linspace(0.0, float(np.nanmax(finite)) * 1.05 + 1e-6, 50)
            g, a = cal.get(f"gamma_{pref}"), cal.get(f"alpha_{pref}")
            if g is not None and a is not None and np.isfinite(g) and np.isfinite(a):
                ax.plot(xs, a + g * xs, color=_BASELINE_COLOR, lw=1.0,
                        label=rf"fit $\gamma$={g:.3f}, $R^2$={cal.get(f'r2_{pref}', float('nan')):.2f}")
            ax.plot(xs, xs, color=_BASELINE_COLOR, lw=0.6, ls=":", label="y=x")
        ax.set_xlabel(f"{label} (nats)")
        ax.set_title(f"{label} calibration")
        ax.legend(loc="upper left", frameon=False, fontsize=6.0)
        ps.style_axes(ax)
    ax_inj.set_ylabel(r"$\bar K$ (nats/step)")
    fig.suptitle(f"synthetic_v2 calibration by lag  (n={len(pc['cell_id'])} cells, "
                 f"run {metrics.get('run_tag', '?')})", fontsize=ps.FONT_SUPTITLE)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_fig(fig, out_path, formats, dpi)


def plot_frac_phi_distribution(
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    frac_threshold: Optional[float] = None,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write per-lag $\mathrm{frac}_\Phi$ (and $\mathrm{TE}_{\mathrm{scat}}$) distributions (S7-T08).

    Population evidence: grouped by lag $D$, the per-cell $\mathrm{frac}_\Phi$ mean $\pm$ s.e.m.
    (left, with the ideal-1 line and the pass threshold) and $\mathrm{TE}_{\mathrm{scat}}$
    mean $\pm$ s.e.m. (right), each bar annotated with its cell count.

    Args:
        metrics: The ``metrics.json`` dict from :func:`eval_v2.run_eval`.
        out_path: Output path stem or full path.
        frac_threshold: Optional pass threshold to draw on the frac_Phi panel.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    pc = _per_cell_arrays(metrics)
    fig, (ax_frac, ax_scat) = plt.subplots(1, 2, figsize=(10.0, 4.4))
    for ax, key, ylabel, title, refs in (
        (ax_frac, "frac_phi", r"$\mathrm{frac}_\Phi$", r"preservation by lag",
         [(1.0, "ideal 1"), (frac_threshold, "threshold")]),
        (ax_scat, "te_scat", r"$\mathrm{TE}_{\mathrm{scat}}$ (nats)",
         r"realizable TE by lag", [(0.0, None)]),
    ):
        stats = _group_stats(pc["D"], pc[key])
        groups = sorted(stats)
        xs = np.arange(len(groups))
        means = [stats[g]["mean"] for g in groups]
        sems = [stats[g]["sem"] for g in groups]
        ax.bar(xs, means, yerr=sems, color=_INJ_COLOR, width=0.7, capsize=3)
        for xi, g in zip(xs, groups):
            ax.text(xi, ax.get_ylim()[1], f"n={stats[g]['n']}", ha="center", va="top",
                    fontsize=6.0, color=_BASELINE_COLOR)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"D={int(g)}" for g in groups])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for ref, lbl in refs:
            if ref is not None:
                ax.axhline(ref, color=_BASELINE_COLOR, lw=0.8, ls="--", label=lbl)
        handles = [h for h in ax.get_legend_handles_labels()[1] if h]
        if handles:
            ax.legend(loc="best", frameon=False, fontsize=6.5)
        ps.style_axes(ax)
    fig.suptitle(f"synthetic_v2 preservation distributions  (run {metrics.get('run_tag', '?')})",
                 fontsize=ps.FONT_SUPTITLE)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_fig(fig, out_path, formats, dpi)


def plot_lag_mass_summary(
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write the LagMass-vs-true-lag summary grouped by lag cell (S7-T08).

    Population evidence: per-cell LagMass grouped by lag $D$ (mean $\pm$ s.e.m.), with the
    pass threshold and cell counts, so lag recovery is legible as evidence across the grid.

    Args:
        metrics: The ``metrics.json`` dict from :func:`eval_v2.run_eval`.
        out_path: Output path stem or full path.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    lag = metrics.get("lag_recovery", {}) or {}
    pc = _per_cell_arrays(metrics)
    stats = _group_stats(pc["D"], pc["lag_mass"])
    groups = sorted(stats)
    xs = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(xs, [stats[g]["mean"] for g in groups], yerr=[stats[g]["sem"] for g in groups],
           color=_BAND_COLOR, width=0.7, capsize=3)
    for xi, g in zip(xs, groups):
        ax.text(xi, stats[g]["mean"], f"n={stats[g]['n']}", ha="center", va="bottom",
                fontsize=6.0, color=_BASELINE_COLOR)
    thr = lag.get("lag_mass_threshold")
    if thr is not None:
        ax.axhline(float(thr), color=_BASELINE_COLOR, lw=0.9, ls="--",
                   label=f"threshold {float(thr):g}")
        ax.legend(loc="best", frameon=False, fontsize=6.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"D={int(g)}" for g in groups])
    ax.set_ylabel(r"LagMass (attention in $\mathcal{L}^\star$)")
    ax.set_xlabel("lag cell")
    ax.set_title(f"synthetic_v2 lag recovery by lag  "
                 f"(mean {_g(lag.get('mean_lag_mass'))}, run {metrics.get('run_tag', '?')})")
    ps.style_axes(ax)
    fig.tight_layout()
    return _save_fig(fig, out_path, formats, dpi)


def _g(x: Any) -> str:
    r"""Format a scalar for a title (``n/a`` for None/non-finite)."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"{xf:.3g}" if np.isfinite(xf) else "n/a"


# =============================================================================
# S7 extension: prediction-gap ("uplift") + previously-computed-but-unplotted
# diagnostics (per-cell lag profiles, per-step KLD, both null controls, three-TE).
# =============================================================================

_TE_CMAP = "viridis"


def _per_cell_profiles(metrics: Dict[str, Any]) -> Dict[int, Dict[str, np.ndarray]]:
    r"""Read the per-cell variable-length profiles from a ``metrics.json`` dict.

    Parses ``metrics["per_cell_profiles"]`` (written by :func:`eval_v2.run_eval`),
    coercing the JSON string cell keys back to ``int`` and the stored lists back to
    float arrays.

    Args:
        metrics: The dict written by :func:`eval_v2.run_eval`.

    Returns:
        ``{cell_id: {'lag_profile': (L,), 'kbar_over_time': (T,), 'lag_count': int}}``;
        empty arrays where a profile is absent.
    """
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for cid, prof in (metrics.get("per_cell_profiles") or {}).items():
        try:
            key = int(cid)
        except (TypeError, ValueError):
            continue
        if not isinstance(prof, dict):
            continue
        lp = prof.get("lag_profile")
        kt = prof.get("kbar_over_time")
        out[key] = {
            "lag_profile": np.asarray(lp, dtype=float) if lp is not None else np.empty(0),
            "kbar_over_time": (np.asarray(kt, dtype=float) if kt is not None
                               else np.empty(0)),
            "lag_count": int(prof.get("lag_count", 0) or 0),
        }
    return out


def _te_raw_by_cell(realizability: Optional[Dict[str, Any]]) -> Dict[int, float]:
    r"""Map ``cell_id -> te_raw`` from a ``realizability.json`` dict.

    Args:
        realizability: The dict written by
            :func:`eval_v2.run_realizability_preflight` (or ``None``).

    Returns:
        ``{cell_id: te_raw}`` (empty when ``realizability`` is ``None`` / malformed).
    """
    out: Dict[int, float] = {}
    if not realizability:
        return out
    per = realizability.get("per_cell")
    items = per.values() if isinstance(per, dict) else (per if isinstance(per, list) else [])
    for row in items:
        if not isinstance(row, dict) or row.get("cell_id") is None:
            continue
        try:
            cid = int(row["cell_id"])
            raw = row.get("te_raw")
            out[cid] = float(raw) if raw is not None else float("nan")
        except (TypeError, ValueError):
            continue
    return out


def _te_norm(te_values: np.ndarray) -> Normalize:
    r"""Build a :class:`~matplotlib.colors.Normalize` spanning the finite TE range."""
    finite = np.asarray(te_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 1.0
    if hi <= lo:
        hi = lo + 1.0
    return Normalize(vmin=lo, vmax=hi)


def _no_data(ax, msg: str) -> None:
    r"""Render a centered 'no data' placeholder on an axes."""
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            color=_BASELINE_COLOR, fontsize=8.0)


def _ols_line(x: np.ndarray, y: np.ndarray) -> Optional[tuple]:
    r"""Return ``(xs, ys, slope)`` of an OLS fit over the finite ``(x, y)`` pairs, or None."""
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2 or np.ptp(x[m]) <= 0:
        return None
    slope, intercept = np.polyfit(x[m], y[m], 1)
    xs = np.linspace(float(x[m].min()), float(x[m].max()), 50)
    return xs, intercept + slope * xs, float(slope)


def plot_pred_gain_vs_te(
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    realizability: Optional[Dict[str, Any]] = None,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write the headline prediction-gap figure: forecast uplift $\Delta L$ vs TE (pred-gap).

    The **second, independent axis of evidence** beside $\bar K$: the per-cell forecast
    "uplift" $\Delta L = L_{\mathrm{base}} - L_{\mathrm{full}}$ (baseline FHR-only forecast
    MSE minus the full-model forecast MSE, over the clean window) plotted against both
    $\mathrm{TE}_{\mathrm{inj}}$ (circles) and $\mathrm{TE}_{\mathrm{scat}}$ (squares), each
    with its OLS trend line. A model that genuinely *uses* the injected coupling should show
    $\Delta L$ rising monotonically with TE and $\Delta L > 0$ for signal cells; a null cell
    sits at $\mathrm{TE}=0,\ \Delta L \approx 0$. $\Delta L$ is an MSE-difference (matching
    the testing pipeline's ``uplift_abs``), so the informative read is the **sign and the
    monotone trend**, not an absolute $\Delta L = \mathrm{TE}$ identity (different units).

    Args:
        metrics: The dict written by :func:`eval_v2.run_eval` (needs per-cell ``pred_gain``).
        out_path: Output path stem or full path.
        realizability: Unused here (kept for a uniform call signature with the gallery).
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    pc = _per_cell_arrays(metrics)
    te_inj, te_scat, pg = pc["te_inj"], pc["te_scat"], pc["pred_gain"]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    finite = np.isfinite(te_inj) & np.isfinite(pg)
    if not finite.any():
        _no_data(ax, "no pred_gain data (run --stage eval)")
        ps.style_axes(ax)
        return _save_fig(fig, out_path, formats, dpi)

    ax.scatter(te_inj[finite], pg[finite], s=24, color=_INJ_COLOR, zorder=3,
               label=r"vs $\mathrm{TE}_{\mathrm{inj}}$")
    fit_inj = _ols_line(te_inj, pg)
    if fit_inj is not None:
        xs, ys, slope = fit_inj
        ax.plot(xs, ys, color=_INJ_COLOR, lw=1.0, label=rf"inj trend (slope={slope:.3g})")
    ms = np.isfinite(te_scat) & np.isfinite(pg)
    if ms.any():
        ax.scatter(te_scat[ms], pg[ms], s=24, marker="s", facecolors="none",
                   edgecolors=_SCAT_COLOR, zorder=3, label=r"vs $\mathrm{TE}_{\mathrm{scat}}$")
        fit_scat = _ols_line(te_scat, pg)
        if fit_scat is not None:
            xs, ys, slope = fit_scat
            ax.plot(xs, ys, color=_SCAT_COLOR, lw=1.0, ls="--",
                    label=rf"scat trend (slope={slope:.3g})")
    ax.axhline(0.0, color=_BASELINE_COLOR, lw=0.7, ls=":")
    ax.set_xlabel("TE (nats)")
    ax.set_ylabel(r"prediction gain $\Delta L = L_{\mathrm{base}} - L_{\mathrm{full}}$ (MSE)")
    ax.set_title(rf"prediction gain vs TE ($n$={int(finite.sum())} cells)")
    ax.legend(loc="upper left", frameon=False, fontsize=6.5)
    ps.style_axes(ax)
    fig.tight_layout()
    return _save_fig(fig, out_path, formats, dpi)


def plot_pred_gain_vs_kbar(
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    realizability: Optional[Dict[str, Any]] = None,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write the $(\bar K,\ \Delta L)$ plane coloured by $\mathrm{TE}_{\mathrm{inj}}$.

    Answers "does the latent KL the model spends actually buy forecast accuracy?": each
    point is a cell at its mean $\bar K$ (x) and prediction gain $\Delta L$ (y), coloured by
    the injected TE. A healthy model has both rising together along the TE gradient.

    Args:
        metrics: The dict written by :func:`eval_v2.run_eval`.
        out_path: Output path stem or full path.
        realizability: Unused (uniform gallery signature).
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    pc = _per_cell_arrays(metrics)
    kbar, pg, te = pc["kbar_mean"], pc["pred_gain"], pc["te_inj"]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    finite = np.isfinite(kbar) & np.isfinite(pg)
    if not finite.any():
        _no_data(ax, "no pred_gain data (run --stage eval)")
        ps.style_axes(ax)
        return _save_fig(fig, out_path, formats, dpi)
    norm = _te_norm(te)
    sc = ax.scatter(kbar[finite], pg[finite], c=te[finite], cmap=_TE_CMAP, norm=norm,
                    s=28, zorder=3)
    ax.axhline(0.0, color=_BASELINE_COLOR, lw=0.7, ls=":")
    ps.add_colorbar(fig, sc, ax, label=r"$\mathrm{TE}_{\mathrm{inj}}$ (nats)")
    ax.set_xlabel(r"$\bar K$ (nats/step)")
    ax.set_ylabel(r"prediction gain $\Delta L$ (MSE)")
    ax.set_title(rf"prediction gain vs $\bar K$ ($n$={int(finite.sum())} cells)")
    ps.style_axes(ax)
    fig.tight_layout()
    return _save_fig(fig, out_path, formats, dpi)


def plot_three_te(
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    realizability: Optional[Dict[str, Any]] = None,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write the per-cell three-TE comparison: $\mathrm{TE}_{\mathrm{inj}}$ / raw / scat.

    v2's core "does the coupling survive the encoder?" story in one figure: grouped bars of
    the injected label $\mathrm{TE}_{\mathrm{inj}}$, the raw-waveform TE $\mathrm{TE}_{\mathrm{raw}}$
    (joined from ``realizability.json``), and the scattering-realizable TE
    $\mathrm{TE}_{\mathrm{scat}}$ per cell, with $\mathrm{frac}_\Phi = \mathrm{TE}_{\mathrm{scat}}/
    \mathrm{TE}_{\mathrm{inj}}$ annotated above each signal cell. Null cells (TE$=0$) are
    tick-labelled ``null``. Degrades to two series (inj/scat) when ``realizability`` is absent.

    Args:
        metrics: The dict written by :func:`eval_v2.run_eval`.
        out_path: Output path stem or full path.
        realizability: The ``realizability.json`` dict (for $\mathrm{TE}_{\mathrm{raw}}$).
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    pc = _per_cell_arrays(metrics)
    cell_ids, te_inj, te_scat, frac = (pc["cell_id"], pc["te_inj"], pc["te_scat"],
                                       pc["frac_phi"])
    keep = np.isfinite(cell_ids)
    order = np.argsort(cell_ids[keep])
    cids = cell_ids[keep][order]
    inj = te_inj[keep][order]
    scat = te_scat[keep][order]
    fr = frac[keep][order]
    raw_map = _te_raw_by_cell(realizability)
    raw = np.array([raw_map.get(int(c), np.nan) for c in cids], dtype=float)
    have_raw = np.isfinite(raw).any()

    fig, ax = plt.subplots(figsize=(max(7.0, 0.7 * len(cids) + 2.0), 4.6))
    if len(cids) == 0:
        _no_data(ax, "no per-cell TE data")
        ps.style_axes(ax)
        return _save_fig(fig, out_path, formats, dpi)
    x = np.arange(len(cids))
    if have_raw:
        w = 0.27
        ax.bar(x - w, inj, width=w, color=_INJ_COLOR, label=r"$\mathrm{TE}_{\mathrm{inj}}$")
        ax.bar(x, raw, width=w, color=_BAND_COLOR, label=r"$\mathrm{TE}_{\mathrm{raw}}$")
        ax.bar(x + w, scat, width=w, color=_SCAT_COLOR, label=r"$\mathrm{TE}_{\mathrm{scat}}$")
    else:
        w = 0.4
        ax.bar(x - w / 2, inj, width=w, color=_INJ_COLOR, label=r"$\mathrm{TE}_{\mathrm{inj}}$")
        ax.bar(x + w / 2, scat, width=w, color=_SCAT_COLOR,
               label=r"$\mathrm{TE}_{\mathrm{scat}}$")
    # frac_Phi annotation above the scat bar for signal cells.
    top = ax.get_ylim()[1]
    for xi, f, tei in zip(x, fr, inj):
        if np.isfinite(f) and np.isfinite(tei) and tei > 1e-9:
            ax.text(xi + (w if have_raw else w / 2), top * 0.02, rf"$\Phi${f:.2f}",
                    ha="center", va="bottom", fontsize=5.5, color=_BASELINE_COLOR,
                    rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(c)}" + ("\nnull" if abs(t) < 1e-9 else "")
                        for c, t in zip(cids, inj)], fontsize=6.0)
    ax.set_xlabel("cell id")
    ax.set_ylabel("TE (nats)")
    ax.set_title("three transfer entropies per cell (injected / raw / scattering-realizable)")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    ps.style_axes(ax)
    fig.tight_layout()
    return _save_fig(fig, out_path, formats, dpi)


def plot_lag_profiles(
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    realizability: Optional[Dict[str, Any]] = None,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write per-cell attention/lag profiles $A_\ell$ faceted by lag $D$ (S6 lag recovery).

    The full lag artifact behind the scalar LagMass: for each distinct lag $D$ (a facet),
    every signal cell's normalised lag profile $A_\ell$ (clean-window mean of the model's
    ``te_lag_map``) is overlaid, coloured by $\mathrm{TE}_{\mathrm{inj}}$, with the true
    informative band $\mathcal L^\star = \{\max(0, D-H), \dots, D-1\}$ shaded. Recovery looks
    like mass concentrating inside the shaded band. Null cells are drawn thin gray dashed as
    the diffuse baseline. Faceting by $D$ keeps each shaded band physically meaningful.

    Args:
        metrics: The dict written by :func:`eval_v2.run_eval` (needs ``per_cell_profiles``).
        out_path: Output path stem or full path.
        realizability: Unused (uniform gallery signature).
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    prof = _per_cell_profiles(metrics)
    pc = _per_cell_arrays(metrics)
    H = int(metrics.get("horizon") or 0)
    info = {int(c): (pc["D"][i], pc["te_inj"][i])
            for i, c in enumerate(pc["cell_id"]) if np.isfinite(c)}
    lag_ds = sorted({int(info[c][0]) for c in prof
                     if c in info and np.isfinite(info[c][0])})
    if not prof or not lag_ds:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        _no_data(ax, "no per-cell lag profiles (run --stage eval)")
        ps.style_axes(ax)
        return _save_fig(fig, out_path, formats, dpi)

    norm = _te_norm(pc["te_inj"])
    cmap = plt.get_cmap(_TE_CMAP)
    n_d = len(lag_ds)
    fig, axes = plt.subplots(1, n_d, figsize=(4.2 * n_d, 4.0), squeeze=False)
    for ax, D in zip(axes[0], lag_ds):
        lo = max(0, D - H)
        ax.axvspan(lo - 0.5, D - 0.5, alpha=0.15, color=_BAND_COLOR,
                   label=r"$\mathcal{L}^\star$")
        for c, (cd, te) in info.items():
            if c not in prof or int(cd) != D:
                continue
            a = prof[c]["lag_profile"]
            if a.size == 0:
                continue
            s = a.sum()
            a_norm = a / s if s > 0 else a
            lags = np.arange(a_norm.shape[0])
            if np.isfinite(te) and te > 1e-9:
                ax.plot(lags, a_norm, lw=0.9, color=cmap(norm(te)))
            else:
                ax.plot(lags, a_norm, lw=0.7, ls="--", color=_BASELINE_COLOR, alpha=0.7)
        ax.set_title(rf"$D={D}$   ($\mathcal{{L}}^\star=\{{{lo},\dots,{D - 1}\}}$)")
        ax.set_xlabel(r"lag $\ell$ (steps)")
        ps.style_axes(ax)
    axes[0][0].set_ylabel(r"normalised attention $A_\ell$")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=list(axes[0]), label=r"$\mathrm{TE}_{\mathrm{inj}}$ (nats)",
                 fraction=0.046, pad=0.02)
    fig.suptitle(f"synthetic_v2 lag profiles by lag  (run {metrics.get('run_tag', '?')})",
                 fontsize=ps.FONT_SUPTITLE)
    return _save_fig(fig, out_path, formats, dpi)


def plot_kld_vs_time(
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    realizability: Optional[Dict[str, Any]] = None,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write per-cell per-step KL trajectories $\bar K_t$ over time (S6).

    The within-sample KL time course behind the scalar $\bar K$: one line per cell (the
    per-cell mean of ``kld_per_t``), coloured by $\mathrm{TE}_{\mathrm{inj}}$, so transient
    vs sustained coupling and warm-up behaviour are visible. Dashed markers show the warm-up
    boundary and the clean-window end $T-H$ (the region averaged into $\bar K$).

    Args:
        metrics: The dict written by :func:`eval_v2.run_eval` (needs ``per_cell_profiles``).
        out_path: Output path stem or full path.
        realizability: Unused (uniform gallery signature).
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    prof = _per_cell_profiles(metrics)
    pc = _per_cell_arrays(metrics)
    te_by_cell = {int(c): pc["te_inj"][i]
                  for i, c in enumerate(pc["cell_id"]) if np.isfinite(c)}
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    traces = {c: prof[c]["kbar_over_time"] for c in prof
              if prof[c]["kbar_over_time"].size > 0}
    if not traces:
        _no_data(ax, "no per-cell KLD trajectories (run --stage eval)")
        ps.style_axes(ax)
        return _save_fig(fig, out_path, formats, dpi)
    norm = _te_norm(np.array(list(te_by_cell.values())) if te_by_cell else np.array([0.0]))
    cmap = plt.get_cmap(_TE_CMAP)
    T = 0
    for c, tr in sorted(traces.items()):
        T = max(T, tr.shape[0])
        te = te_by_cell.get(c, np.nan)
        color = cmap(norm(te)) if np.isfinite(te) else _BASELINE_COLOR
        ax.plot(np.arange(tr.shape[0]), tr, lw=0.9, color=color, alpha=0.9)
    warmup = metrics.get("warmup")
    horizon = metrics.get("horizon")
    if warmup is not None:
        ax.axvline(float(warmup), color=_BASELINE_COLOR, lw=0.8, ls="--", label="warm-up")
    if horizon is not None and T > 0:
        ax.axvline(float(T - int(horizon)), color=_BASELINE_COLOR, lw=0.8, ls=":",
                   label=r"clean-window end $T-H$")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    ps.add_colorbar(fig, sm, ax, label=r"$\mathrm{TE}_{\mathrm{inj}}$ (nats)")
    ax.set_xlabel("time step (decimated)")
    ax.set_ylabel(r"$\bar K_t$ (nats/step)")
    ax.set_title(rf"per-cell KL over time  ($n$={len(traces)} cells, "
                 f"run {metrics.get('run_tag', '?')})")
    if warmup is not None or horizon is not None:
        ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    ps.style_axes(ax)
    fig.tight_layout()
    return _save_fig(fig, out_path, formats, dpi)


def plot_null_controls(
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    realizability: Optional[Dict[str, Any]] = None,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write per-cell null-control ratios for **all** controls (shuffle + reverse).

    Complements the 2$\times$2 diagnostics panel (which shows only the first control): the
    per-cell null ratio $\bar K_{\mathrm{null}} / \bar K_{\mathrm{signal}}$ for every
    configured control (source shuffle and time-reverse), grouped per cell. A model that
    genuinely uses the source has ratios $\to 0$ on signal cells and $\approx 1$ on nulls.

    Args:
        metrics: The dict written by :func:`eval_v2.run_eval`.
        out_path: Output path stem or full path.
        realizability: Unused (uniform gallery signature).
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    pc = _per_cell_arrays(metrics)
    null_cols = sorted(k for k in pc if k.startswith("null_") and k.endswith("_ratio"))
    cell_ids = pc["cell_id"]
    keep = np.isfinite(cell_ids)
    order = np.argsort(cell_ids[keep])
    cids = cell_ids[keep][order]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.6 * len(cids) + 2.0), 4.4))
    if not null_cols or len(cids) == 0:
        _no_data(ax, "no null controls in metrics")
        ps.style_axes(ax)
        return _save_fig(fig, out_path, formats, dpi)
    x = np.arange(len(cids))
    nc = len(null_cols)
    width = 0.8 / nc
    palette = ps.PALETTE_EXTENDED
    for j, col in enumerate(null_cols):
        vals = pc[col][keep][order]
        label = col.replace("null_", "").replace("_ratio", "")
        ax.bar(x + (j - (nc - 1) / 2.0) * width, vals, width=width,
               color=palette[j % len(palette)], label=label)
    ax.axhline(0.0, color=_BASELINE_COLOR, lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(c)}" for c in cids], fontsize=6.0)
    ax.set_xlabel("cell id")
    ax.set_ylabel(r"null ratio $\bar K_{\mathrm{null}} / \bar K_{\mathrm{signal}}$")
    ax.set_title(r"null controls per cell  ($\to 0$ signal, $\approx 1$ null)")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5, title="control")
    ps.style_axes(ax)
    fig.tight_layout()
    return _save_fig(fig, out_path, formats, dpi)
