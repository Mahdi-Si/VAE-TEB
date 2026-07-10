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

S7-T10 adds the data-generation *story* figures — the **controls** behind the previews
above rather than their outputs, reusing the pipeline's own math (:mod:`raw_generators`,
:mod:`analytic_te`) via lazy imports so this module stays free of the torch / kymatio
transform: the frequency recipe (:func:`plot_band_spectra`, Welch PSD with the
physiological bands + coupled carrier + LF notch, §4-§5); the TE control law
(:func:`plot_te_authoring`, the $\mathrm{TE}^{(H)}(B)$ sweep with the inverter-solved
$B$ per target and the SNR extractability law, §9); the coupling pathway / lag
(:func:`plot_latent_coupling`, source $\to$ target with the delay $D$ and the true lag
band $\mathcal L^\star$, §6); and the carrier de-risk (:func:`plot_am_separation`,
envelope spectrum vs the analyzing-wavelet passband at $0.06$ vs $0.02\,\mathrm{Hz}$, §7).

Sprint 8 (S8) adds the KLD-summary-family analysis (§14.5): the model exposes several
KLD tensors (``kld_per_t`` total, ``kld_per_t_per_head``, ``te_lag_map``) and each can be
summarised over a sample's clean window in several ways, so these figures relate *every*
such summary to the transfer entropy rather than only the canonical $\bar K$ —
:func:`plot_kld_variants_vs_te` (one panel per KLD flavour vs TE),
:func:`plot_kld_te_correlation` (the ranked "which summary tracks TE best"),
:func:`plot_kld_te_density` (per-sample density + fit residuals),
:func:`plot_kld_distribution_by_te` (per-TE-level violins), and
:func:`plot_per_head_kld_vs_te` (which latent head carries the coupling). They are backed
by the per-sample arrays in ``per_sample_eval.npz`` and the ``calibration.kld_variants``
block of ``metrics.json``.

See ``SYNTHETIC_V2_SPEC_AND_SPRINTS.md`` Sprints 1, 2, 7, 8.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
_HIGHLIGHT_COLOR = ps.COLOR_SKY  # band annotations / latent overlay (heatmap marker is _HEATMAP_MARK_COLOR)
_INJ_COLOR = ps.COLOR_BLUE       # TE_inj series
_SCAT_COLOR = ps.COLOR_VERMILLION  # TE_scat series
_BAND_COLOR = ps.COLOR_GREEN     # true lag band / reference

# Default raster resolution for PNG output across the gallery (PDF is vector).
_DPI = ps.SAVE_DPI

# Diverging blue-white-red colormap for every coefficient / phase-harmonic heatmap.
# The model-facing features are per-channel z-scored (mean $\approx 0$), so a diverging
# map reads their sign directly -- blue $< 0$, white $\approx 0$, red $> 0$ -- provided
# the colour scale is centred on zero (see :func:`_symmetric_limits`).
_HEATMAP_CMAP = "bwr"
# Marker colour for the coupled-channel line / annotations drawn over the ``bwr`` maps.
# Black reads cleanly against blue, white, and red alike (the old teal blended into the
# blue tail of the diverging scale).
_HEATMAP_MARK_COLOR = ps.COLOR_BLACK


def _symmetric_limits(arrays: List[np.ndarray], pct: float = 99.0) -> Tuple[float, float]:
    r"""Return a zero-centred ``(vmin, vmax)`` for a diverging colour scale.

    Pools every supplied array, takes the ``pct`` percentile of the *absolute* values
    as a robust half-range $L$ (so a few outliers do not wash the map out), and returns
    $(-L, +L)$. This places the midpoint of :data:`_HEATMAP_CMAP` (white) exactly on
    $0$, which is what makes a ``bwr`` heatmap of z-scored coefficients read its sign
    honestly.

    Args:
        arrays: The 2-D heatmap arrays that must share one colour scale.
        pct: Percentile of $|x|$ used for the symmetric limit (default $99$, i.e. the
            top $1\%$ of magnitudes are clipped).

    Returns:
        A ``(vmin, vmax) = (-L, +L)`` tuple; falls back to a tiny non-zero range when
        the pooled data is flat.
    """
    pooled = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    lim = float(np.percentile(np.abs(pooled), pct)) if pooled.size else 0.0
    if not np.isfinite(lim) or lim <= 0.0:
        lim = float(np.max(np.abs(pooled))) if pooled.size else 1.0
        if not np.isfinite(lim) or lim <= 0.0:
            lim = 1e-6
    return -lim, lim


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
    fhr_ph: Optional[np.ndarray] = None,
    up_ph: Optional[np.ndarray] = None,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write an annotated raw FHR/UP preview figure (S1-T05).

    Renders one sample's raw waveforms as two stacked panels: FHR (bpm) on top with
    its baseline $\mu_{\mathrm{FHR}}$ marked, and UP (mmHg) below with its resting
    tone $\mu_{\mathrm{UP}}$ marked. When the phase-harmonic correlation fields
    ``fhr_ph`` / ``up_ph`` are supplied, two further channel$\times$time heatmaps of
    those correlations are stacked below the traces (diverging :data:`_HEATMAP_CMAP`
    ``bwr`` on a zero-centred scale), so the raw sample and the phase structure the
    model reads are shown together. The title carries the cell provenance
    ($\mathrm{TE}_{\mathrm{inj}}$, lag $D$, coupling $B$) when supplied in ``meta``.

    Note: with the default ``am_carrier`` render the FHR coupled term is a modulated
    sinusoid symmetric about the baseline (best for the sign-blind scattering modulus),
    not a one-sided clinical deceleration dip; the waveform-realistic ``pulse_train`` and
    the carrier-free ``direct`` (§7.4) renders instead show one-sided contraction /
    deceleration deflections in the raw traces.

    Args:
        fhr_raw: FHR waveform(s), shape ``(n, N)`` or ``(N,)`` (bpm).
        up_raw: UP waveform(s), shape ``(n, N)`` or ``(N,)`` (mmHg).
        out_path: Output path stem (extensions in ``formats`` are appended) or a full
            path (its suffix is stripped and ``formats`` used).
        meta: Optional provenance dict (keys ``te_inj``, ``D``, ``B``, ``f_pulse``).
        fs: Raw sampling rate in Hz (for the time axis).
        sample: Row index to plot when the inputs are 2-D.
        fhr_ph: Optional FHR phase-harmonic correlation field, $(n, T, C)$ or $(T, C)$;
            when both phase fields are given a phase-harmonic heatmap pair is added.
        up_ph: Optional UP phase-harmonic correlation field, same layout as ``fhr_ph``.
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

    # With phase fields, use the house colorbar-gutter stack (two traces + two heatmaps);
    # without them keep the lean two-panel trace figure.
    have_phase = fhr_ph is not None and up_ph is not None
    caxes: List[Optional[Any]] = [None, None]
    if have_phase:
        fig, axes, caxes = ps.stacked_figure(
            [1.3, 1.3, 1.5, 1.5], width=10.0,
            colorbar=[False, False, True, True], hspace=0.55,
        )
        ax_fhr, ax_up, ax_fhr_ph, ax_up_ph = axes
    else:
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
    # Annotate the coupled contraction band (the UP source of the pathway).
    ax_up.text(0.01, 0.04, "coupled contraction band (source)", transform=ax_up.transAxes,
               color=_HIGHLIGHT_COLOR, fontsize=6.5, va="bottom", ha="left")
    ax_up.legend(loc="upper right", frameon=False)

    for ax in (ax_fhr, ax_up):
        ps.tighten_xaxis(ax, t_min)
        ps.style_axes(ax)

    if have_phase:
        fhr_p = _to_channels_time(fhr_ph, sample)  # (C, T)
        up_p = _to_channels_time(up_ph, sample)
        ph_vmin, ph_vmax = _symmetric_limits([fhr_p, up_p])
        step_s = 16.0 / fs
        for ax, cax, data, name in (
            (ax_fhr_ph, caxes[2], fhr_p, "FHR"),
            (ax_up_ph, caxes[3], up_p, "UP"),
        ):
            d_ch, d_t = data.shape
            panel_t_max = d_t * step_s / 60.0
            im = ax.imshow(
                data, aspect="auto", origin="lower",
                extent=(0.0, panel_t_max, -0.5, d_ch - 0.5),
                vmin=ph_vmin, vmax=ph_vmax, cmap=_HEATMAP_CMAP, interpolation="nearest",
            )
            ax.set_ylabel(f"{name} phase-harm. channel")
            ps.attach_colorbar(fig, im, cax, label="z-scored value")
        ax_fhr_ph.set_title("phase-harmonic correlation", fontsize=9.0)
        ax_up_ph.set_xlabel("time (min)")
        for ax in (ax_fhr, ax_up):  # bottom heatmap carries the shared time axis
            plt.setp(ax.get_xticklabels(), visible=False)
        return _save_fig(fig, out_path, formats, dpi)

    ax_up.set_xlabel("time (min)")
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
    fhr_ph: Optional[np.ndarray] = None,
    up_ph: Optional[np.ndarray] = None,
    coupled_idx: Optional[int] = None,
    center_freqs: Optional[np.ndarray] = None,
    fs: float = 4.0,
    sample: int = 0,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write stacked FHR / UP scattering + phase-harmonic heatmaps (S2-T04).

    Renders the $43$-channel first-order scattering fields as two stacked
    channel$\times$time heatmaps (FHR then UP), and -- when the phase-harmonic
    correlation fields ``fhr_ph`` / ``up_ph`` are supplied -- two further heatmaps of
    those correlations below them. Every panel uses the diverging :data:`_HEATMAP_CMAP`
    (``bwr``) on a zero-centred scale (:func:`_symmetric_limits`), so the sign of the
    z-scored coefficients reads directly. Each panel gets its own colorbar in a dedicated
    gutter (via :func:`plot_style_v2.stacked_figure`); the scattering pair shares one
    colour scale and the phase pair shares another. The fs-correct coupled pulse-shape
    channel (``coupled_idx``) is marked on the scattering panels. When ``center_freqs``
    (normalised $\xi$) is supplied, the scattering panels' y-tick labels are shown in Hz
    (physical Hz $= \xi\,f_s$); channel $0$ is the order-0 low-pass baseline (``S0``).

    Args:
        fhr_st: FHR scattering field, $(n, T, C)$ or $(T, C)$ (normalised).
        up_st: UP scattering field, $(n, T, C)$ or $(T, C)$ (normalised).
        out_path: Output path stem (formats appended) or a full path.
        fhr_ph: Optional FHR phase-harmonic correlation field, $(n, T, C)$ or $(T, C)$;
            when both phase fields are given a phase-harmonic heatmap pair is added.
        up_ph: Optional UP phase-harmonic correlation field, same layout as ``fhr_ph``.
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

    # Zero-centred, robust colour scale shared across the scattering pair (features are
    # per-channel z-scored, so blue/red = below/above the channel mean).
    scat_vmin, scat_vmax = _symmetric_limits([fhr, up])

    # (name, data, kind) per panel, top to bottom. The phase-harmonic pair is appended
    # only when both fields are supplied (kept optional for back-compat / lean callers).
    panels: List[Tuple[str, np.ndarray, str]] = [
        ("FHR", fhr, "scat"), ("UP", up, "scat"),
    ]
    have_phase = fhr_ph is not None and up_ph is not None
    ph_vmin = ph_vmax = 0.0
    if have_phase:
        fhr_p = _to_channels_time(fhr_ph, sample)
        up_p = _to_channels_time(up_ph, sample)
        ph_vmin, ph_vmax = _symmetric_limits([fhr_p, up_p])
        panels += [("FHR", fhr_p, "phase"), ("UP", up_p, "phase")]

    fig, axes, caxes = ps.stacked_figure(
        [1.6] * len(panels), width=10.0, colorbar=[True] * len(panels), hspace=0.5,
    )

    for ax, cax, (name, data, kind) in zip(axes, caxes, panels):
        vmin, vmax = (scat_vmin, scat_vmax) if kind == "scat" else (ph_vmin, ph_vmax)
        d_ch, d_t = data.shape
        panel_t_max = d_t * step_s / 60.0
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=(0.0, panel_t_max, -0.5, d_ch - 0.5),
            vmin=vmin,
            vmax=vmax,
            cmap=_HEATMAP_CMAP,
            interpolation="nearest",
        )
        if kind == "scat":
            ax.set_ylabel(f"{name} scat. channel")
            if coupled_idx is not None:
                ax.axhline(coupled_idx, color=_HEATMAP_MARK_COLOR, lw=1.2, ls="--")
                ax.text(
                    0.01 * panel_t_max,
                    coupled_idx + 0.6,
                    f"coupled ch {coupled_idx}",
                    color=_HEATMAP_MARK_COLOR,
                    va="bottom",
                    ha="left",
                    fontsize=7.5,
                )
        else:
            ax.set_ylabel(f"{name} phase-harm. channel")
        ps.attach_colorbar(fig, im, cax, label="z-scored value")

    # Hz y-tick labels for the scattering panels when centre frequencies are supplied.
    if center_freqs is not None:
        cf = np.asarray(center_freqs, dtype=float)
        tick_ch = [0] + list(range(5, n_ch, 8))
        labels = ["S0" if ch == 0 else f"{cf[ch - 1] * fs:.3f}" for ch in tick_ch]
        for ax, (name, _data, kind) in zip(axes, panels):
            if kind == "scat":
                ax.set_yticks(tick_ch)
                ax.set_yticklabels(labels)
                ax.set_ylabel(f"{name} channel (Hz)")

    axes[0].set_title(
        "synthetic_v2 scattering coefficients"
        + (" & phase-harmonic correlation" if have_phase else "")
        + " (normalised)",
        fontsize=10.0,
    )
    if have_phase:
        axes[2].set_title("phase-harmonic correlation", fontsize=9.0)
    # Time label only on the bottom panel; hide the inner panels' x-tick labels.
    axes[-1].set_xlabel("time (min)")
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    return _save_fig(fig, out_path, formats, dpi)


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


# Metrics excluded from the interactive training-curve HTML: the CSV bookkeeping
# columns (``epoch`` is the x-axis, ``step`` the global step) and the redundant
# ``LearningRateMonitor`` duplicate of ``lr`` (the bare ``lr`` trace is kept).
_HTML_EXCLUDE_METRICS = frozenset({"epoch", "step", "lr-AdamW"})

# Canonical metric order so the legend reads loss -> KL -> saturation/health ->
# lr, with each metric's train series immediately followed by its val twin.
# Metrics not listed here are appended in first-seen order so nothing the model
# logs is ever silently dropped from the curve.
_HTML_METRIC_ORDER = (
    "total_loss",
    "feat_loss",
    "base_loss",
    "kld_loss",
    "kld_nats",
    "kld_raw",
    "kld_train",
    "kld_active_frac",
    "pred_gap",
    # Source-permutation control (S3-T01), a no-grad readout. ``feat_loss_perm`` is a
    # ``source_state`` batch derangement -- NOT eval's input-stream ``feat_loss_shuffle``.
    "kld_shuffled",
    "kld_shuffled_ratio",
    "feat_loss_perm",
    "shuffle_penalty",
    "mu_prior_sat_frac",
    "delta_mu_sat_frac",
    "kld_beta",
    "mean_logvar_full",
    "mean_logvar_base",
    "spike_ema_loss",
    "spike_skipped",
    "spike_skips_total",
    "lr",
)


def _enumerate_html_metrics(rows: List[Dict[str, str]]) -> List[tuple]:
    r"""Order the logical metric series to draw in the interactive training curve.

    Lightning's ``CSVLogger`` forks every ``on_step=on_epoch=True`` metric into a
    ``<key>_step`` and a ``<key>_epoch`` column; this collapses that fork to the
    logical key, drops the bookkeeping / duplicate columns in
    :data:`_HTML_EXCLUDE_METRICS`, and orders the survivors by
    :data:`_HTML_METRIC_ORDER` with each metric's ``train`` series placed immediately
    before its ``val`` series in the legend. Unrecognised metrics are appended in
    first-seen order so a newly-logged quantity appears automatically.

    Args:
        rows: The parsed ``metrics.csv`` rows (from :class:`csv.DictReader`).

    Returns:
        A list of ``(label, csv_key, is_val)`` triples in draw order, where
        ``csv_key`` is the logical key handed to :func:`_read_metric_series` (e.g.
        ``train/total_loss``) and ``is_val`` marks the validation twin (used only for
        legend ordering; every trace is drawn solid).
    """
    if not rows:
        return []

    present: List[str] = []
    seen: set = set()
    for col in rows[0].keys():
        if col.endswith("_epoch"):
            base = col[:-6]
        elif col.endswith("_step"):
            base = col[:-5]
        else:
            base = col
        if base in _HTML_EXCLUDE_METRICS or base in seen:
            continue
        seen.add(base)
        present.append(base)

    def _split(base: str) -> tuple:
        for stage in ("train", "val"):
            if base.startswith(f"{stage}/"):
                return stage, base[len(stage) + 1:]
        return None, base

    by_metric: Dict[str, Dict[Optional[str], str]] = {}
    first_seen: List[str] = []
    for base in present:
        stage, metric = _split(base)
        if metric not in by_metric:
            by_metric[metric] = {}
            first_seen.append(metric)
        by_metric[metric][stage] = base

    def _emit(metric: str) -> List[tuple]:
        stages = by_metric.get(metric, {})
        out: List[tuple] = []
        if "train" in stages:
            out.append((f"train {metric}", stages["train"], False))
        if None in stages:  # stage-less (e.g. ``lr``): a single solid trace
            out.append((metric, stages[None], False))
        if "val" in stages:
            out.append((f"val {metric}", stages["val"], True))
        return out

    result: List[tuple] = []
    done: set = set()
    for metric in _HTML_METRIC_ORDER:
        if metric in by_metric:
            result.extend(_emit(metric))
            done.add(metric)
    for metric in first_seen:
        if metric not in done:
            result.extend(_emit(metric))
            done.add(metric)
    return result


def _html_trace_colors(n: int) -> List[str]:
    r"""Return ``n`` distinct hex colours (house palette first, Plotly ``Dark24`` overflow).

    The interactive training curve overlays every logged metric (~20 traces) and each
    line must have a unique colour. The house :data:`plot_style_v2.PALETTE_EXTENDED`
    supplies 8 brand-consistent hues; any beyond that are drawn -- de-duplicated --
    from Plotly's 24-colour ``Dark24`` qualitative set (dark, legible on the
    ``plotly_white`` template), for ~30 distinct colours in total. The list only
    repeats if more than ~30 traces are requested.

    Args:
        n: Number of distinct colours required.

    Returns:
        A list of ``n`` hex colour strings.
    """
    palette: List[str] = list(ps.PALETTE_EXTENDED)
    try:
        from plotly.colors import qualitative

        have = {c.upper() for c in palette}
        for hexcol in qualitative.Dark24:
            if hexcol.upper() not in have:
                palette.append(hexcol)
                have.add(hexcol.upper())
    except ImportError:  # pragma: no cover - caller already imported plotly
        pass
    if not palette:  # defensive: never modulo by zero
        palette = ["#1f77b4"]
    return [palette[i % len(palette)] for i in range(n)]


def plot_loss_curves_html(
    metrics_csv: Union[str, Path],
    out_path: Union[str, Path],
    *,
    title: Optional[str] = None,
    include_plotlyjs: bool = True,
) -> Optional[List[Path]]:
    r"""Write one interactive Plotly HTML overlaying every training metric as its own trace.

    Reads the Lightning ``CSVLogger`` ``metrics.csv`` and renders a single-panel
    ``go.Figure`` in which **every** logged per-epoch metric -- the losses
    ($\mathcal L$, $\mathcal L_{\mathrm{feat}}$, $\mathcal L_{\mathrm{base}}$), the
    per-dim KL and its dim-summed twin ``kld_nats`` ($\bar K = \mathrm{KL}\cdot d_z$,
    the TE surrogate), the predictive gap, the $\mu$ / $\Delta\mu$ saturation
    fractions, $\beta$, the spike-breaker diagnostics, and the learning rate -- is
    drawn as its own distinctly-coloured **solid** line (no dashed styling). Each
    metric's ``train`` series and its ``val`` twin are placed adjacently in the legend
    and told apart by colour; colours come from :func:`_html_trace_colors` so no two
    lines share a hue. This mirrors the v1
    ``synthetic/plot_training_curves.py`` interactive curve (one figure, one trace per
    metric) rather than a curated subset. It is consumed live during training by
    :class:`~model.vae_teb_prediction.model.model_experiment.synthetic_v2.callbacks_v2.LossPlotHtmlCallback`,
    which rewrites it every few epochs so a long run can be watched mid-flight.

    The metric set is discovered dynamically from the CSV header via
    :func:`_enumerate_html_metrics` (resolving Lightning's ``_step`` / ``_epoch`` fork
    through :func:`_read_metric_series`), so a newly-logged quantity appears
    automatically without editing this function. The very wide dynamic range across
    metrics (e.g. $\mathcal L \gg$ ``kld_beta`` $= 10^{-3}$; ``*_sat_frac`` $\in [0,1]$)
    is handled by Plotly's interactivity -- double-click a legend entry to isolate a
    trace and autoscale.

    Plotly is an optional dependency: if it is not importable this warns and returns
    ``None`` rather than raising, so a missing install never breaks training. The output
    embeds ``plotly.js`` by default so the file opens offline.

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

    # Collect every non-empty per-epoch series in canonical draw order, then colour
    # them distinctly (house palette first). Two passes so the house colours land on
    # the metrics that actually have data.
    series: List[tuple] = []
    for label, key, is_val in _enumerate_html_metrics(rows):
        epochs, values = _read_metric_series(rows, key)
        if epochs:
            series.append((label, is_val, epochs, values))

    colors = _html_trace_colors(len(series))
    fig = go.Figure()
    for (label, is_val, epochs, values), color in zip(series, colors):
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=values,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=1.6),
                marker=dict(size=4),
            )
        )

    fig.update_layout(
        title=title or "synthetic_v2 training curves",
        xaxis_title="epoch",
        yaxis_title="value",
        legend_title="metric",
        template="plotly_white",
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
    fhr_ph: Optional[np.ndarray] = None,
    up_ph: Optional[np.ndarray] = None,
    center_freqs: Optional[np.ndarray] = None,
    fs: float = 4.0,
    trim: int = 15,
    meta: Optional[Dict[str, Any]] = None,
    sample: int = 0,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write the headline raw + scattering (+ phase-harmonic) paired preview (S7-T02).

    Stacked panels sharing aligned colorbar gutters: raw FHR trace, FHR scattering
    heatmap, raw UP trace, UP scattering heatmap. When the phase-harmonic correlation
    fields ``fhr_ph`` / ``up_ph`` are supplied, a phase-harmonic heatmap is inserted below
    each scattering panel (six panels total), so the **full model input** -- both the
    scattering *magnitude* channels and the *phase-harmonic* channels -- is shown next to
    the raw signal. The fs-correct coupled pulse-shape channel (``coupled_idx``) is
    highlighted on both scattering heatmaps, and the decimated latent (sliced to the
    feature grid ``[trim:trim+T]``) is overlaid on that channel so its tracking is visible
    on a strong cell.

    The caller supplies the already-transformed, normalised ``*_st`` / ``*_ph`` fields (so
    this module stays free of the torch / kymatio transform).

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
        fhr_ph: Optional FHR phase-harmonic correlation field, $(n, T, C)$ or $(T, C)$;
            when both phase fields are given a phase-harmonic heatmap pair is added.
        up_ph: Optional UP phase-harmonic correlation field, same layout as ``fhr_ph``.
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

    # Zero-centred, robust scale for the diverging ``bwr`` map shared by both scattering
    # heatmaps (features are per-channel z-scored, so blue/red = below/above the mean).
    vmin, vmax = _symmetric_limits([fhr_cf, up_cf])

    # Optional phase-harmonic panels: the other half of the model input. When supplied a
    # phase heatmap is inserted below each scattering panel (FHR block, then UP block), so
    # the layout grows from 4 to 6 panels; the phase pair shares its own colour scale.
    have_phase = fhr_ph is not None and up_ph is not None
    if have_phase:
        fhr_pf = _to_channels_time(fhr_ph, sample)
        up_pf = _to_channels_time(up_ph, sample)
        ph_vmin, ph_vmax = _symmetric_limits([fhr_pf, up_pf])
        heights = [1.0, 1.5, 1.5, 1.0, 1.5, 1.5]
        cbar = [False, True, True, False, True, True]
    else:
        heights = [1.1, 1.6, 1.1, 1.6]
        cbar = [False, True, False, True]

    fig, axes, caxes = ps.stacked_figure(heights, width=11.0, colorbar=cbar, hspace=0.55)
    if have_phase:
        ax_fhr_raw, ax_fhr_st, ax_fhr_ph, ax_up_raw, ax_up_st, ax_up_ph = axes
    else:
        ax_fhr_raw, ax_fhr_st, ax_up_raw, ax_up_st = axes
        ax_fhr_ph = ax_up_ph = None
    cax_of = dict(zip(axes, caxes))

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
    for ax, data, name, latent in (
        (ax_fhr_st, fhr_cf, "FHR", latent_d),
        (ax_up_st, up_cf, "UP", latent_c),
    ):
        im = ax.imshow(
            data, aspect="auto", origin="lower",
            extent=(0.0, t_dec_min[-1] if n_t else 0.0, -0.5, n_ch - 0.5),
            vmin=vmin, vmax=vmax, cmap=_HEATMAP_CMAP, interpolation="nearest",
        )
        ax.axhline(coupled_idx, color=_HEATMAP_MARK_COLOR, lw=1.0, ls="--")
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
        ps.attach_colorbar(fig, im, cax_of[ax], label="z-scored value")

    # --- phase-harmonic heatmaps (the other half of the model input) ---
    if have_phase:
        for ax, data, name in ((ax_fhr_ph, fhr_pf, "FHR"), (ax_up_ph, up_pf, "UP")):
            ph_ch, ph_t = data.shape
            im = ax.imshow(
                data, aspect="auto", origin="lower",
                extent=(0.0, ph_t * step_s / 60.0, -0.5, ph_ch - 0.5),
                vmin=ph_vmin, vmax=ph_vmax, cmap=_HEATMAP_CMAP, interpolation="nearest",
            )
            ax.set_ylabel(f"{name} phase-h. channel")
            ps.attach_colorbar(fig, im, cax_of[ax], label="z-scored value")

    if center_freqs is not None:
        cf = np.asarray(center_freqs, dtype=float)
        tick_ch = [0, coupled_idx] + list(range(8, n_ch, 12))
        tick_ch = sorted(set(t for t in tick_ch if 0 <= t < n_ch))
        labels = ["S0" if ch == 0 else f"{cf[ch - 1] * fs:.3f}" for ch in tick_ch]
        for ax in (ax_fhr_st, ax_up_st):
            ax.set_yticks(tick_ch)
            ax.set_yticklabels(labels)

    (ax_up_ph if have_phase else ax_up_st).set_xlabel("time (min)")

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


# ---------------------------------------------------------------------------
# Data-generation story figures: the CONTROLS behind the previews above.
# These illustrate *how* the data is authored -- the frequency recipe (§4-§5),
# the TE control law (§9), the coupling pathway / lag (§6), and the carrier
# de-risk (§7) -- reusing the pipeline's own math (``raw_generators`` /
# ``analytic_te``), imported lazily so this module's top level stays free of the
# torch / kymatio transform stack.
# ---------------------------------------------------------------------------

#: Coupled FHR deceleration (VLF) band in Hz (EXPLAINED §4). Deliberately *excluded*
#: from :data:`raw_generators.FHRV_BANDS` -- it is reserved for the coupled pathway --
#: so it is named here only for shading.
_VLF_DECEL_BAND: Tuple[float, float] = (0.003, 0.03)
#: Coupled UP contraction-rhythm band in Hz (EXPLAINED §4): the AR(2) source
#: envelope's spectral support ($\sim 0.004\,\mathrm{Hz}$ peak).
_UP_CONTRACTION_BAND: Tuple[float, float] = (0.003, 0.0083)


def plot_band_spectra(
    fhr_raw: np.ndarray,
    up_raw: np.ndarray,
    out_path: Union[str, Path],
    *,
    fs: float = 4.0,
    meta: Optional[Dict[str, Any]] = None,
    nperseg: Optional[int] = None,
    sample: int = 0,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write the frequency-recipe figure: Welch PSDs with the physiological bands marked (§4-§5).

    Two stacked panels show the Welch power spectral density (log-log) of one raw FHR
    and one raw UP waveform, with the physiologically-placed bands of the composition
    model shaded on top. This makes the §5 additive model legible -- each raw signal is a
    sum of independent, physiologically-placed bands plus **one** coupled carrier:

    $$x_{\mathrm{UP}}(t) = \mu_{\mathrm{UP}} + u_{\mathrm c}(t) + \mathrm{drift}(t) + \varepsilon(t),
    \qquad
    x_{\mathrm{FHR}}(t) = \mu_{\mathrm{FHR}} - y_{\mathrm d}(t)
    + \textstyle\sum_{\mathrm{band}} \mathrm{FHRV} + \mathrm{accel} + \varepsilon(t).$$

    The FHR panel marks the independent baseline-wander band
    (:data:`raw_generators.FHR_WANDER_BAND`), the deceleration band
    ($\mathcal V\!\mathrm{LF}$, :data:`_VLF_DECEL_BAND`), and the independent FHRV
    LF/MF/HF dressing (:data:`raw_generators.FHRV_BANDS`). The UP panel marks the slow
    drift (:data:`raw_generators.UP_DRIFT_BAND`) and the contraction-rhythm band
    (:data:`_UP_CONTRACTION_BAND`). Both panels draw the locked pulse carrier
    $f_{\mathrm{pulse}}$ (from ``meta``) and shade the excised LF notch neighbourhood
    $[f_{\mathrm{pulse}} 2^{-1/Q}, f_{\mathrm{pulse}} 2^{+1/Q}]$ (``meta['fhrv_notch']``).
    These deceleration / contraction bands are the *envelope* rhythms of the coupled
    pathway (where $c, d$ live on the decimated grid); by the §7 AM rendering the coupled
    term itself is carried at $f_{\mathrm{pulse}}$ (sidebands around the carrier), not as
    standalone power in those bands.

    Args:
        fhr_raw: FHR waveform(s), shape $(n, N)$ or $(N,)$ (bpm).
        up_raw: UP waveform(s), shape $(n, N)$ or $(N,)$ (mmHg).
        out_path: Output path stem or full path.
        fs: Raw sampling rate in Hz.
        meta: Provenance dict; reads ``f_pulse`` (carrier Hz) and ``fhrv_notch`` (the
            $(\mathrm{lo}, \mathrm{hi})$ notch neighbourhood, or ``None`` if disabled).
        nperseg: Welch segment length; defaults to $\min(N, 4096)$ (long enough to
            resolve the sub-$0.01\,\mathrm{Hz}$ physiological bands).
        sample: Row index to plot when the inputs are batched.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    from scipy.signal import welch

    from .raw_generators import FHRV_BANDS, FHR_WANDER_BAND, UP_DRIFT_BAND

    fhr = np.asarray(fhr_raw, dtype=float)
    up = np.asarray(up_raw, dtype=float)
    if fhr.ndim == 2:
        fhr = fhr[sample]
    if up.ndim == 2:
        up = up[sample]

    n_min = int(min(fhr.shape[-1], up.shape[-1]))
    seg = int(nperseg) if nperseg is not None else min(n_min, 4096)
    seg = max(16, min(seg, n_min))  # size from the shorter signal so both grids match
    f_fhr, p_fhr = welch(fhr, fs=fs, nperseg=seg, detrend="constant")
    f_up, p_up = welch(up, fs=fs, nperseg=seg, detrend="constant")
    # Drop the DC bin (f = 0) so the log-frequency axis is well defined.
    f_fhr, p_fhr = f_fhr[1:], p_fhr[1:]
    f_up, p_up = f_up[1:], p_up[1:]

    meta = meta or {}
    f_pulse = float(meta.get("f_pulse", 0.06))
    if "fhrv_notch" in meta:
        notch = meta["fhrv_notch"]  # tuple, or None when the notch is disabled
    else:  # meta lacks it -- reconstruct the default Q=4 neighbourhood for the preview
        notch = (f_pulse * 2.0 ** (-0.25), f_pulse * 2.0 ** (0.25))

    x_lo = float(min(f_fhr[0], f_up[0]))
    x_hi = fs / 2.0

    def _shade(ax, lo: float, hi: float, color, alpha: float, label: str) -> None:
        ax.axvspan(lo, hi, color=color, alpha=alpha, lw=0.0)
        xc = float(np.clip(np.sqrt(max(lo, x_lo) * hi), x_lo, x_hi))
        ax.text(xc, 0.965, label, transform=ax.get_xaxis_transform(), ha="center",
                va="top", rotation=90, fontsize=5.5, color=_BASELINE_COLOR)

    fig, axes, _ = ps.stacked_figure([1.5, 1.5], width=10.0, hspace=0.5,
                                     colorbar=False, margins=(0.10, 0.95, 0.92, 0.06))
    ax_fhr, ax_up = axes

    ax_fhr.semilogy(f_fhr, p_fhr, color=_FHR_COLOR, lw=0.8)
    _shade(ax_fhr, *FHR_WANDER_BAND, _BASELINE_COLOR, 0.10, "wander")
    _shade(ax_fhr, *_VLF_DECEL_BAND, _HIGHLIGHT_COLOR, 0.16, "decel band (VLF)")
    for (name, band), col in zip(FHRV_BANDS.items(), (ps.COLOR_ORANGE, ps.COLOR_GREEN,
                                                      ps.COLOR_PURPLE)):
        _shade(ax_fhr, *band, col, 0.10, f"FHRV {name}")
    ax_fhr.set_ylabel(r"FHR PSD (bpm$^2$/Hz)")
    ax_fhr.set_title("FHR: independent dressing + deceleration bands "
                     "(coupled term AM-rendered onto the carrier)")

    ax_up.semilogy(f_up, p_up, color=_UP_COLOR, lw=0.8)
    _shade(ax_up, *UP_DRIFT_BAND, _BASELINE_COLOR, 0.10, "slow drift")
    _shade(ax_up, *_UP_CONTRACTION_BAND, _HIGHLIGHT_COLOR, 0.16, "contraction rhythm")
    ax_up.set_ylabel(r"UP PSD (mmHg$^2$/Hz)")
    ax_up.set_xlabel("frequency (Hz)")
    ax_up.set_title("UP: slow drift + contraction rhythm "
                    "(coupled term AM-rendered onto the same carrier)")

    for ax in (ax_fhr, ax_up):
        ax.axvline(f_pulse, color=_HEATMAP_MARK_COLOR, lw=1.1, ls="--",
                   label=rf"$f_{{\mathrm{{pulse}}}}$={f_pulse:g} Hz")
        if notch is not None:
            ax.axvspan(float(notch[0]), float(notch[1]), color=_HEATMAP_MARK_COLOR,
                       alpha=0.14, lw=0.0, label="LF notch")
        ax.set_xscale("log")
        ax.set_xlim(x_lo, x_hi)
        ax.legend(loc="lower left", frameon=False, fontsize=6.0)
        ps.style_axes(ax)

    fig.suptitle("synthetic_v2 band recipe: additive physiological bands + one coupled carrier",
                 fontsize=ps.FONT_SUPTITLE, y=0.985)
    return _save_fig(fig, out_path, formats, dpi)


def plot_te_authoring(
    config: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    benchmark: str = "G1_raw",
    delay: int = 8,
    target_te_grid: Optional[Sequence[float]] = None,
    n_b: int = 9,
    n_samples: int = 6000,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write the TE-control figure: author by TE, solve for coupling $B$, gauge extractability (§9).

    Panel (a) is the **authoring map**: the Monte-Carlo block transfer entropy
    $\mathrm{TE}^{(H)}(B)$ swept over the coupling magnitude $B$ at a fixed lag $D$
    (:func:`analytic_te.te_block_state_space_gaussian`), with the solved coupling for
    each authored ``target_te`` overlaid as marked points -- each solved by the exact
    inverter :func:`analytic_te.B_y_for_mean_te_block_state_space` the build uses, so the
    figure reproduces "author a TE, get a $B$". Panel (b) is the **extractability law**
    $$\mathrm{SNR} \approx e^{2\,\mathrm{TE}^{(H)}/(H M)} - 1$$
    (:func:`analytic_te.snr_per_step_for_te_block`, $M = \mathrm{len(oscillators)} = 1$ for
    the single-pathway benchmark), with the grid TEs marked and
    the $\sim\!1\%$ per-step innovation SNR below which the coupling is effectively
    unextractable shaded. Together: we author by TE, solve for $B$, and the SNR law sets
    what a finite-sample predictor can recover.

    Self-contained from ``config`` (reads ``benchmarks.<b>.data`` / ``.mix`` and
    ``model.horizon``); no realizability preflight required. The Monte-Carlo is the only
    cost -- keep ``n_samples`` / ``n_b`` modest for a fast preview.

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        out_path: Output path stem or full path.
        benchmark: Active benchmark key under ``benchmarks``.
        delay: Fixed source$\to$target lag $D$ (decimated steps) for the sweep and solves.
        target_te_grid: Authored block TEs (nats) to solve/mark; defaults to
            ``benchmarks.<b>.mix.target_te_grid``.
        n_b: Number of $B$ points in the $\mathrm{TE}^{(H)}(B)$ sweep.
        n_samples: Monte-Carlo sample count for each TE evaluation and solve.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    from .analytic_te import (
        B_y_for_mean_te_block_state_space,
        snr_per_step_for_te_block,
        te_block_state_space_gaussian,
    )

    bench = config["benchmarks"][benchmark]
    data = bench["data"]
    inv = bench["mix"]["inverter"]
    oscillators = [tuple(spec) for spec in data["oscillators"]]
    target_ar = float(data["target_ar"])
    sigma2_y = float(data["sigma2_y"])
    sigma2_eta = float(data["sigma2_eta"])
    h_data = int(data["horizon"])          # defines TE_inj (matches the inverter)
    k_history = int(data["K_history"])
    h_model = int(config["model"]["horizon"])  # the model forecast horizon (SNR law)
    seed = int(config.get("seeds", {}).get("inverter_mc", 0))
    m_channels = len(oscillators)

    grid = list(target_te_grid) if target_te_grid is not None else list(bench["mix"]["target_te_grid"])
    positive = [float(t) for t in grid if float(t) > 0.0]

    # Solve B for each authored TE -- the overlay points (exact inverter, fixed lag).
    solved: List[Tuple[float, float, float]] = []  # (target_te, B*, te_block_realised)
    for te in positive:
        try:
            sol = B_y_for_mean_te_block_state_space(
                target_te_block=te, delay_min=int(delay), delay_max=int(delay),
                oscillators=oscillators, target_ar=target_ar, sigma2_y=sigma2_y,
                sigma2_eta=sigma2_eta, H=h_data, K_history=k_history,
                n_samples=int(n_samples), lo=float(inv["lo"]), hi=float(inv["hi"]),
                tol=float(inv["tol"]), max_iter=int(inv["max_iter"]), seed=seed,
            )
            solved.append((te, float(sol["B_y_scalar"]), float(sol["te_block"])))
        except ValueError:
            continue  # bracket missed this target -- skip the unsolvable cell

    fig, axes, _ = ps.stacked_figure([1.5, 1.5], width=9.0, hspace=0.6,
                                     colorbar=False, margins=(0.11, 0.95, 0.92, 0.07))
    ax_a, ax_b = axes

    if solved:
        b_max = max(b for _, b, _ in solved)
        b_grid = np.linspace(0.0, 1.15 * b_max, int(n_b))
        curve = np.array([
            te_block_state_space_gaussian(
                oscillators, target_ar, [int(delay)], [float(b)], sigma2_y, sigma2_eta,
                h_data, K_history=k_history, n_samples=int(n_samples), seed=seed)
            for b in b_grid
        ])
        ax_a.plot(b_grid, curve, color=_BASELINE_COLOR, lw=1.3, marker="o", ms=2.5,
                  label=r"$\mathrm{TE}^{(H)}(B)$ (Monte-Carlo)")
        for i, (te, b_star, te_real) in enumerate(solved):
            col = ps.PALETTE_PRIMARY[i % len(ps.PALETTE_PRIMARY)]
            ax_a.plot([b_star], [te_real], marker="D", ms=7, color=col, ls="none",
                      label=rf"target {te:g} $\to B={b_star:.3g}$")
            ax_a.plot([0.0, b_star, b_star], [te_real, te_real, 0.0], color=col, lw=0.7,
                      ls=":", alpha=0.7)
        ax_a.set_ylabel(r"$\mathrm{TE}^{(H)}$ (nats)")
        ax_a.legend(loc="lower right", frameon=False, fontsize=6.0)
    else:
        _no_data(ax_a, "no solvable target TEs in the bracket")
    ax_a.set_xlabel(r"coupling magnitude $B$")
    ax_a.set_title(rf"author by TE $\to$ solve for $B$  (fixed lag $D={int(delay)}$, $H={h_data}$)")
    ps.style_axes(ax_a)

    te_hi = max([t for _, _, t in solved] + [max(grid) if grid else 0.0] + [1e-3])
    te_axis = np.linspace(0.0, 1.1 * te_hi, 200)
    snr_pct = 100.0 * np.array([snr_per_step_for_te_block(float(t), h_model, m_channels)
                                for t in te_axis])
    ax_b.axhspan(0.0, 1.0, color=_BASELINE_COLOR, alpha=0.10)
    ax_b.plot(te_axis, snr_pct, color=_INJ_COLOR, lw=1.5,
              label=r"$\mathrm{SNR}\approx e^{2\,\mathrm{TE}^{(H)}/(HM)}-1$")
    for te in positive:
        s = 100.0 * snr_per_step_for_te_block(float(te), h_model, m_channels)
        ax_b.plot([te], [s], marker="s", ms=6, color=_SCAT_COLOR, ls="none")
    ax_b.axhline(1.0, color=_BASELINE_COLOR, lw=0.9, ls="--")
    ax_b.text(1.1 * te_hi, 1.0, "  ~1% unextractable floor", ha="right", va="bottom",
              fontsize=6.5, color=_BASELINE_COLOR)
    ax_b.set_xlabel(r"injected block TE $\mathrm{TE}^{(H)}$ (nats)")
    ax_b.set_ylabel("per-step innovation SNR (%)")
    ax_b.set_title(rf"extractability law: SNR sets what is recoverable  ($H={h_model}$, $M={m_channels}$)")
    ax_b.set_xlim(0.0, 1.1 * te_hi)
    ax_b.legend(loc="upper left", frameon=False, fontsize=6.5)
    ps.style_axes(ax_b)

    fig.suptitle(f"synthetic_v2 TE authoring & extractability (benchmark {benchmark})",
                 fontsize=ps.FONT_SUPTITLE, y=0.985)
    return _save_fig(fig, out_path, formats, dpi)


def plot_latent_coupling(
    latents: Dict[str, np.ndarray],
    out_path: Union[str, Path],
    *,
    D: int,
    horizon: int,
    fs: float = 4.0,
    decimation: int = 16,
    max_lag: Optional[int] = None,
    sample: int = 0,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write the coupling-pathway figure: source $\to$ target with the delay $D$ made visible (§6).

    Panel 1 overlays the decimated coupled latents -- target $d$ (FHR deceleration depth),
    source $c$ (UP contraction strength), and $c$ shifted forward by $D$ steps ($c_{k-D}$)
    -- so the alignment of the shifted source with the target's response is legible: the
    §6.2 target obeys $d_k = A_y d_{k-1} + B\,c_{k-D} + \varepsilon_k$, i.e. contraction
    strength drives deceleration depth $D$ steps later. Panel 2 is the standardized
    cross-correlation $\mathrm{corr}(d_k, c_{k-\ell})$ vs lag $\ell$ (pooled over all rows),
    which peaks near $\ell = D$; the true past-source lag band
    $\mathcal L^\star = \{\max(0, D-H), \dots, D-1\}$ -- the lags a horizon-$H$ forecaster
    must attend to -- is shaded.

    Args:
        latents: The ``latents`` dict from :func:`raw_generators.generate_cell_raw`
            (uses ``c`` and ``d``, each $(n, T')$ or $(T',)$ on the decimated grid).
        out_path: Output path stem or full path.
        D: The source$\to$target coupling lag (decimated steps).
        horizon: The forecast horizon $H$ (sets the true lag band width).
        fs: Raw sampling rate in Hz (the latents live on ``fs / decimation``).
        decimation: Decimation factor mapping the raw grid to the latent grid.
        max_lag: Largest lag $\ell$ shown; defaults to $D + H + 5$.
        sample: Row index to plot in panel 1 when the arrays are batched.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    c1 = _latent_row(latents, "c", sample)
    d1 = _latent_row(latents, "d", sample)
    c_all = np.asarray(latents["c"], dtype=float)
    d_all = np.asarray(latents["d"], dtype=float)
    if c_all.ndim == 1:
        c_all = c_all[None, :]
    if d_all.ndim == 1:
        d_all = d_all[None, :]

    D = int(D)
    horizon = int(horizon)
    t_tot = int(c1.shape[-1])
    fs_dec = fs / float(decimation)
    t_min = np.arange(t_tot) / fs_dec / 60.0
    delay_s = D * decimation / fs
    lag_hi = int(max_lag) if max_lag else D + horizon + 5

    fig, axes, _ = ps.stacked_figure([1.4, 1.4], width=10.0, hspace=0.55,
                                     colorbar=False, margins=(0.10, 0.95, 0.92, 0.08))
    ax_t, ax_x = axes

    # Panel 1: the latent pair with the source shifted by D onto the target's response.
    ax_t.plot(t_min, d1, color=_FHR_COLOR, lw=1.1, label=r"$d$ (target: FHR decel depth)")
    ax_t.plot(t_min, c1, color=_UP_COLOR, lw=0.9, alpha=0.6,
              label=r"$c$ (source: UP contraction)")
    c_shift = np.full(t_tot, np.nan)
    if D < t_tot:
        c_shift[D:] = c1[: t_tot - D]
    ax_t.plot(t_min, c_shift, color=_HIGHLIGHT_COLOR, lw=1.2, ls="--",
              label=rf"$c_{{k-D}}$ (source shifted by $D={D}$)")
    ax_t.axhline(0.0, color=_BASELINE_COLOR, lw=0.6, ls=":")
    ax_t.set_ylabel("latent value")
    ax_t.set_xlabel("time (min)")
    ax_t.set_title(rf"coupling pathway: UP contraction $\to$ FHR deceleration, "
                   rf"delay $D={D}$ steps (${delay_s:g}$ s)")
    ax_t.legend(loc="upper right", frameon=False, fontsize=6.0, ncol=3)
    ps.tighten_xaxis(ax_t, t_min)
    ps.style_axes(ax_t)

    # Panel 2: standardized cross-correlation pooled over rows, peaking at lag D.
    cz = (c_all - c_all.mean(axis=1, keepdims=True)) / (c_all.std(axis=1, keepdims=True) + 1e-12)
    dz = (d_all - d_all.mean(axis=1, keepdims=True)) / (d_all.std(axis=1, keepdims=True) + 1e-12)
    t_c = cz.shape[1]
    lags = np.arange(0, min(lag_hi, t_c - 1) + 1)
    xcorr = np.array([float(np.mean(dz[:, lag:] * cz[:, : t_c - lag])) if lag < t_c else np.nan
                      for lag in lags])
    lo = max(0, D - horizon)
    ax_x.axvspan(lo - 0.5, D - 0.5, color=_BAND_COLOR, alpha=0.15,
                 label=rf"true lag band $\mathcal{{L}}^\star=\{{{lo},\dots,{D - 1}\}}$")
    ax_x.plot(lags, xcorr, color=_SCAT_COLOR, lw=1.2, marker="o", ms=2.5)
    ax_x.axhline(0.0, color=_BASELINE_COLOR, lw=0.6, ls=":")
    ax_x.axvline(D, color=_HEATMAP_MARK_COLOR, lw=1.1, ls="--", label=rf"$D={D}$")
    ax_x.set_xlabel(r"lag $\ell$ (decimated steps)")
    ax_x.set_ylabel(r"corr$(d_k,\,c_{k-\ell})$")
    ax_x.set_title(r"cross-correlation peaks at the coupling delay $D$")
    ax_x.set_xlim(0, int(lags[-1]) if lags.size else lag_hi)
    ax_x.legend(loc="upper right", frameon=False, fontsize=6.5)
    ps.style_axes(ax_x)

    fig.suptitle(f"synthetic_v2 coupling pathway & lag  (D={D}, H={horizon})",
                 fontsize=ps.FONT_SUPTITLE, y=0.985)
    return _save_fig(fig, out_path, formats, dpi)


def plot_am_separation(
    config: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    benchmark: str = "G1_raw",
    f_pulse_compare: float = 0.02,
    decimation: int = 16,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write the carrier de-risk figure: envelope spectrum vs analyzing-wavelet passband (§7).

    On one frequency axis (the decimated grid) it overlays the AR(2) coupling-envelope
    power spectrum $S_{\mathrm{env}}(f)$ (:func:`raw_generators.ar2_psd`, peaked near
    $0.004\,\mathrm{Hz}$) and the analyzing-wavelet passband -- a Gaussian centred at the
    carrier $f_{\mathrm{pulse}}$ of $e^{-1}$ half-width $\sigma_{\mathrm{wav}}$, i.e.
    $e^{-((f - f_{\mathrm{pulse}})/\sigma_{\mathrm{wav}})^2}$, matching the demodulation
    weight of :func:`raw_generators.am_separation_margin` (the constant-$Q$ wavelet width)
    -- for **two** carriers:
    the locked $f_{\mathrm{pulse}}$ and the rejected ``f_pulse_compare`` ($0.02\,\mathrm{Hz}$).
    The separation margin
    $$\mathrm{margin} = \sigma_{\mathrm{wav}} / f_{\mathrm{env,peak}}$$
    is annotated for each. It shows *why* the carrier is locked at $0.06\,\mathrm{Hz}$: at
    $0.02\,\mathrm{Hz}$ the (narrower, lower) constant-$Q$ passband sits close to the envelope
    band, so amplitude demodulation cannot cleanly separate carrier from envelope and the
    injected TE does not survive (EXPLAINED §7.2).

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        out_path: Output path stem or full path.
        benchmark: Active benchmark key under ``benchmarks``.
        f_pulse_compare: The rejected carrier (Hz) to contrast against the locked one.
        decimation: Decimation factor (sets the decimated Nyquist and $S_{\mathrm{env}}$ grid).
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    from .raw_generators import am_separation_margin, ar2_psd

    bench = config["benchmarks"][benchmark]
    data = bench["data"]
    raw = bench["raw"]
    scattering = bench["scattering"]
    r, w = float(data["oscillators"][0][0]), float(data["oscillators"][0][1])
    sigma2_eta = float(data["sigma2_eta"])
    f_pulse = float(raw["f_pulse"])
    q_factor = int(scattering["Q"])
    fs = float(raw["fs"])
    am_offset_ratio = float(raw["am_offset_ratio"])
    fs_dec = fs / float(decimation)

    margin_main = am_separation_margin(
        r=r, w=w, f_pulse=f_pulse, Q=q_factor, fs=fs, decimation=int(decimation),
        am_offset_ratio=am_offset_ratio, sigma2_eta=sigma2_eta,
    )
    margin_alt = am_separation_margin(
        r=r, w=w, f_pulse=float(f_pulse_compare), Q=q_factor, fs=fs, decimation=int(decimation),
        am_offset_ratio=am_offset_ratio, sigma2_eta=sigma2_eta,
    )

    f = np.linspace(1e-4, fs_dec / 2.0, 2000)
    s_env = ar2_psd(f, r, w, sigma2_eta, fs_dec)
    s_env = s_env / float(s_env.max())

    def _passband(fc: float, sigma: float) -> np.ndarray:
        return np.exp(-((f - fc) / sigma) ** 2)

    sig_main = float(margin_main["sigma_wav_hz"])
    sig_alt = float(margin_alt["sigma_wav_hz"])
    f_env_peak = float(margin_main["f_env_peak"])

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.fill_between(f, 0.0, s_env, color=_UP_COLOR, alpha=0.12)
    ax.plot(f, s_env, color=_UP_COLOR, lw=1.5,
            label=r"AR(2) envelope PSD $S_{\mathrm{env}}(f)$ (normalised)")
    ax.plot(f, _passband(f_pulse, sig_main), color=_HIGHLIGHT_COLOR, lw=1.8,
            label=rf"wavelet @ {f_pulse:g} Hz (locked; margin={margin_main['margin_peak']:.1f})")
    ax.plot(f, _passband(float(f_pulse_compare), sig_alt), color=_BASELINE_COLOR, lw=1.5,
            ls="--",
            label=rf"wavelet @ {f_pulse_compare:g} Hz (rejected; margin={margin_alt['margin_peak']:.1f})")
    ax.axvline(f_pulse, color=_HIGHLIGHT_COLOR, lw=0.8, ls=":")
    ax.axvline(float(f_pulse_compare), color=_BASELINE_COLOR, lw=0.8, ls=":")
    ax.axvline(f_env_peak, color=_UP_COLOR, lw=0.8, ls=":")
    ax.annotate(rf"$f_{{\mathrm{{env,peak}}}}\approx{f_env_peak:.4g}$ Hz",
                xy=(f_env_peak, 0.5), xytext=(f_env_peak + 0.01, 0.7),
                fontsize=6.5, color=_UP_COLOR,
                arrowprops=dict(arrowstyle="->", color=_UP_COLOR, lw=0.7))
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("normalised power / wavelet response")
    ax.set_xlim(0.0, fs_dec / 2.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    ax.set_title(r"AM separation: envelope band vs constant-$Q$ carrier passband")
    ps.style_axes(ax)
    fig.suptitle("synthetic_v2 AM separation: why the carrier is locked at 0.06 Hz",
                 fontsize=ps.FONT_SUPTITLE)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return _save_fig(fig, out_path, formats, dpi)


def _per_cell_arrays(metrics: Dict[str, Any]) -> Dict[str, np.ndarray]:
    r"""Collect the per-cell diagnostics columns from a ``metrics.json`` dict.

    Args:
        metrics: The dict written by :func:`eval_v2.run_eval` (its ``per_cell`` list).

    Returns:
        A dict of parallel float arrays keyed by column name (``te_inj``, ``te_scat``,
        ``kbar_mean``, ``frac_phi``, ``lag_mass``, ``D``, ``n``, ``feat_loss``,
        ``base_loss``) plus any per-control null-ratio columns (``null_<ctrl>_ratio``) and
        prediction-space control columns (``feat_loss_<ctrl>``, ``shuffle_penalty_<ctrl>``,
        ``ordering_pass_<ctrl>``, the latter as ``1.0`` / ``0.0`` / ``nan``). Missing entries
        are ``nan``.
    """
    rows = metrics.get("per_cell", []) or []
    keys = ["cell_id", "te_inj", "te_scat", "D", "kbar_mean", "n", "frac_phi",
            "feat_loss", "base_loss", "pred_gain", "uplift_rel", "lag_mass", "peak_lag_err"]
    null_keys = sorted({k for r in rows for k in r if str(k).startswith("null_")
                        and str(k).endswith("_ratio")})
    # Prediction-space control columns (S3-T03), discovered rather than enumerated so a new
    # control name reaches the figures without touching this helper.
    pred_keys = sorted({k for r in rows for k in r
                        if str(k).startswith(("feat_loss_", "shuffle_penalty_",
                                              "ordering_pass_"))})
    out: Dict[str, np.ndarray] = {}
    for key in keys + null_keys + pred_keys:
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
    4. **Prediction control**: per-cell shuffle-penalty bars
       $\mathcal L_{\mathrm{feat}}^{\pi(U)} - \mathcal L_{\mathrm{feat}}$, which a
       source-exploiting model drives **above** the zero reference. (This panel used to plot
       the KL null ratio against a "$\to 0$" caption; that ratio is a readout, not a gate --
       see :func:`plot_kl_shuffle_readout`.)

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

    # --- (4) prediction-space control per cell -------------------------------
    # The headline control is the SHUFFLE PENALTY, not the KL null ratio. A model that
    # exploits the source forecasts worse under a corrupted one, so the penalty
    # $\mathcal L_{\mathrm{feat}}^{\pi(U)} - \mathcal L_{\mathrm{feat}}$ must be positive on
    # every signal cell; the reference line therefore sits at 0 as a *floor to clear*, not as
    # a target to approach. (The old panel plotted the KL null ratio against a "$\to 0$"
    # caption, which no honest model can satisfy -- v3 Finding F2. That ratio now lives in
    # its own figure, ``plot_kl_shuffle_readout``.)
    pen_cols = sorted(k for k in pc if k.startswith("shuffle_penalty_"))
    if pen_cols:
        col = pen_cols[0]
        ctrl = col[len("shuffle_penalty_"):]
        _cell_bar(ax_null, pc[col], pc["cell_id"], color=_SCAT_COLOR,
                  ylabel="shuffle penalty",
                  title=f"prediction control ({ctrl}): penalty $> 0$", ref=0.0,
                  ref_label=None)
    else:
        ax_null.text(0.5, 0.5, "no prediction controls in metrics", ha="center",
                     va="center", transform=ax_null.transAxes, color=_BASELINE_COLOR)
        ax_null.set_title("prediction control")

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


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    r"""Tie-aware (midrank) ranks: tied values share the mean of their ordinal ranks.

    Plain ``argsort(argsort(a))`` assigns arbitrary *distinct* ranks to ties, which biases the
    rank correlation when a variable has few distinct levels with many ties (e.g. the per-sample
    TE axis). This returns proper average ranks, matching ``scipy.stats.rankdata(method='average')``
    without the scipy dependency.
    """
    a = np.asarray(a, dtype=np.float64)
    order = a.argsort(kind="mergesort")
    ordinal = np.empty(a.size, dtype=np.float64)
    ordinal[order] = np.arange(1, a.size + 1, dtype=np.float64)
    uniq, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(uniq.size, dtype=np.float64)
    np.add.at(sums, inv, ordinal)
    return (sums / cnt)[inv]


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    r"""Tie-aware Spearman rank correlation of two 1-D arrays (``nan`` when undefined)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    rx = _rankdata_average(x)
    ry = _rankdata_average(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _kbar_vs_te_panel(
    ax,
    te: np.ndarray,
    kbar: np.ndarray,
    cell_id: np.ndarray,
    *,
    cal: Dict[str, Any],
    pref: str,
    color: str,
    xlabel: str,
    fit: Optional[Tuple[Optional[float], Optional[float], Optional[float]]] = None,
    show_box: bool = True,
    show_identity: bool = True,
) -> int:
    r"""Draw one per-sample $\bar K$-vs-TE panel: cloud + per-level box + means + fit.

    Since the injected / realizable TE is constant within a cell, the x-axis is a small set of
    discrete levels; the per-sample $\bar K$ spread at each level is shown as a jittered point
    cloud plus a box, with the per-cell means and the calibration line overlaid.

    Args:
        ax: Target axes.
        te: Per-sample TE (x), length $N$.
        kbar: Per-sample $\bar K$ (y), length $N$.
        cell_id: Per-sample cell id (for the per-cell-mean markers).
        cal: The ``metrics['calibration']`` dict (for the pooled-sample fit fallback).
        pref: ``'inj'`` or ``'scat'`` — selects the ``gamma_<pref>_sample`` fit keys.
        color: Series colour.
        xlabel: X-axis label.
        fit: Optional explicit ``(gamma, alpha, r2)`` to draw instead of the pooled-sample fit
            (used for the per-lag panels).
        show_box: Whether to draw the per-level box overlay.
        show_identity: Whether to draw the $y=x$ reference (meaningful only for nats/step
            variants; suppressed for scale-different summaries such as the integrated sum).

    Returns:
        The number of finite points plotted.
    """
    te = np.asarray(te, dtype=float).reshape(-1)
    kbar = np.asarray(kbar, dtype=float).reshape(-1)
    cell_id = np.asarray(cell_id, dtype=float).reshape(-1)
    # Guard against a missing / partially-written per_sample array (e.g. te_scat absent): a
    # length-mismatched te would otherwise raise a broadcast error instead of a clean placeholder.
    if te.shape != kbar.shape or kbar.size == 0:
        _no_data(ax, "no per-sample data")
        ax.set_xlabel(xlabel)
        return 0
    if cell_id.shape != kbar.shape:
        cell_id = np.zeros_like(kbar)
    m = np.isfinite(te) & np.isfinite(kbar)
    if int(m.sum()) < 2:
        _no_data(ax, "no per-sample data")
        ax.set_xlabel(xlabel)
        return int(m.sum())
    te_m, kb_m, cid_m = te[m], kbar[m], cell_id[m]
    # A single TE level (e.g. a lag mapped to one cell) has no x-spread: still show the full
    # per-sample K-bar distribution (cloud + box), just without a slope / correlation.
    has_spread = bool(np.ptp(te_m) > 0)
    levels = np.unique(te_m)
    span = float(levels.max() - levels.min()) or 1.0
    jw = 0.02 * span + 1e-3                       # jitter half-width
    bw = 0.05 * span + 1e-3                       # box half-width

    rng = np.random.default_rng(0)               # fixed jitter for reproducible figures
    ax.scatter(te_m + rng.uniform(-jw, jw, size=te_m.shape[0]), kb_m,
               s=6, color=color, alpha=0.12, linewidths=0, zorder=2)

    if show_box:
        data = [kb_m[te_m == lv] for lv in levels]
        bp = ax.boxplot(data, positions=levels, widths=2 * bw, showfliers=False,
                        patch_artist=True, manage_ticks=False, zorder=3)
        for patch in bp["boxes"]:
            patch.set(facecolor="white", edgecolor=_BASELINE_COLOR, alpha=0.9, linewidth=0.8)
        for med in bp["medians"]:
            med.set(color=color, linewidth=1.3)
        for part in bp["whiskers"] + bp["caps"]:
            part.set(color=_BASELINE_COLOR, linewidth=0.7)

    cmx, cmy = [], []
    for c in np.unique(cid_m):
        s = cid_m == c
        cmx.append(float(np.mean(te_m[s])))
        cmy.append(float(np.mean(kb_m[s])))
    ax.scatter(cmx, cmy, s=30, marker="D", facecolor=_BASELINE_COLOR,
               edgecolor="white", linewidths=0.6, zorder=5, label="per-cell mean")

    xs = np.linspace(0.0, float(levels.max()) * 1.05 + 1e-6, 50)
    if fit is not None:
        g, a, r2 = fit
    else:
        g = cal.get(f"gamma_{pref}_sample")
        a = cal.get(f"alpha_{pref}_sample")
        r2 = cal.get(f"r2_{pref}_sample")
    # A fit / correlation is only defined when TE varies across the panel's samples.
    if has_spread and g is not None and a is not None and np.isfinite(g) and np.isfinite(a):
        r2f = float(r2) if r2 is not None and np.isfinite(r2) else float("nan")
        ax.plot(xs, a + g * xs, color=color, lw=1.4,
                label=rf"fit $\gamma$={g:.3f}, $R^2$={r2f:.2f}", zorder=6)
    if show_identity:
        ax.plot(xs, xs, color=_BASELINE_COLOR, lw=0.6, ls=":", label="y=x", zorder=4)

    ax.set_xlabel(xlabel)
    if has_spread:
        pear = float(np.corrcoef(te_m, kb_m)[0, 1])
        rho = _spearman_rho(te_m, kb_m)
        ax.set_title(rf"$r$={pear:.2f}, $\rho$={rho:.2f}, $n$={int(te_m.size)}")
    else:
        ax.set_title(rf"single TE level, $n$={int(te_m.size)}")
    ax.legend(loc="upper left", frameon=False, fontsize=6.0)
    return int(te_m.size)


def plot_te_kld_scatter(
    per_sample: Optional[Dict[str, Any]],
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write the per-sample average-TE vs average-$\bar K$ scatter (Enhancement B).

    Every evaluated test/val sample is one point: its time-averaged latent KL $\bar K$ (y)
    against its cell's transfer entropy (x) — the injected label $\mathrm{TE}_{\mathrm{inj}}$
    (left) and the realizable $\mathrm{TE}_{\mathrm{scat}}$ (right). Because TE is constant
    within a cell, the x-axis is a few discrete levels and the panel shows the full per-sample
    $\bar K$ distribution (jittered cloud + box), the per-cell means, and the pooled per-sample
    calibration line $\bar K = \alpha + \gamma\,\mathrm{TE}$. This is the sample-level view the
    per-cell calibration (~15 points) cannot show.

    Args:
        per_sample: The dict of length-$N$ arrays loaded from ``per_sample_eval.npz``
            (``kbar``, ``te_inj``, ``te_scat``, ``cell_id``, ...); a ``None`` / empty dict
            renders a placeholder.
        metrics: The ``metrics.json`` dict (for the calibration fit and run/split labels).
        out_path: Output path stem or full path.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    cal = metrics.get("calibration", {}) or {}
    ps_arr = per_sample or {}
    kbar = np.asarray(ps_arr.get("kbar", []), dtype=float)
    fig, (ax_inj, ax_scat) = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    if kbar.size == 0:
        for ax in (ax_inj, ax_scat):
            _no_data(ax, "no per_sample_eval.npz (run --stage eval)")
            ps.style_axes(ax)
        return _save_fig(fig, out_path, formats, dpi)

    cell_id = np.asarray(ps_arr.get("cell_id", np.zeros_like(kbar)), dtype=float)
    _kbar_vs_te_panel(ax_inj, ps_arr.get("te_inj", []), kbar, cell_id,
                      cal=cal, pref="inj", color=_INJ_COLOR,
                      xlabel=r"$\mathrm{TE}_{\mathrm{inj}}$ (nats)")
    _kbar_vs_te_panel(ax_scat, ps_arr.get("te_scat", []), kbar, cell_id,
                      cal=cal, pref="scat", color=_SCAT_COLOR,
                      xlabel=r"$\mathrm{TE}_{\mathrm{scat}}$ (nats)")
    ax_inj.set_ylabel(r"$\bar K$ (nats/step)")
    for ax in (ax_inj, ax_scat):
        ps.style_axes(ax)
    split_val = ps_arr.get("split", metrics.get("split", "?"))
    split = str(split_val.item()) if hasattr(split_val, "item") else str(split_val)
    fig.suptitle(rf"synthetic_v2 per-sample $\bar K$ vs TE  ($n$={int(kbar.size)} samples, "
                 rf"split {split}, run {metrics.get('run_tag', '?')})",
                 fontsize=ps.FONT_SUPTITLE)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_fig(fig, out_path, formats, dpi)


def plot_calibration_by_lag(
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    per_sample: Optional[Dict[str, Any]] = None,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write the $\bar K$-vs-TE calibration broken out by lag $D$ (S7-T08, Enhancement C).

    When ``per_sample`` is supplied this is **per-sample small multiples**: one column per lag
    $D$ (top row vs $\mathrm{TE}_{\mathrm{inj}}$, bottom vs $\mathrm{TE}_{\mathrm{scat}}$), each
    panel showing that lag's full per-sample $\bar K$ distribution (cloud + box), its per-cell
    means, and its own fitted $\gamma$ — so a lag group has hundreds/thousands of points rather
    than the ~5 per-cell markers the legacy view drew. Without ``per_sample`` it falls back to
    the legacy per-cell scatter coloured by $D$.

    Args:
        metrics: The ``metrics.json`` dict from :func:`eval_v2.run_eval`.
        out_path: Output path stem or full path.
        per_sample: The per-sample arrays from ``per_sample_eval.npz`` (``kbar``, ``te_inj``,
            ``te_scat``, ``cell_id``, ``delay``); ``None`` selects the legacy per-cell fallback.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    cal = metrics.get("calibration", {}) or {}

    # --- Per-sample small-multiples branch (Enhancement C) --------------------------------
    # One panel per lag D over its per-SAMPLE points (hundreds/thousands each), so a lag group
    # is a real distribution rather than the ~5 per-cell markers the legacy view showed. Top
    # row vs TE_inj, bottom row vs TE_scat; each panel carries that lag's own fitted gamma.
    if per_sample is not None and np.asarray(per_sample.get("kbar", [])).size:
        kbar = np.asarray(per_sample["kbar"], dtype=float)
        te_inj = np.asarray(per_sample.get("te_inj", []), dtype=float)
        te_scat = np.asarray(per_sample.get("te_scat", []), dtype=float)
        cell_id = np.asarray(per_sample.get("cell_id", np.zeros_like(kbar)), dtype=float)
        delay = np.asarray(per_sample.get("delay", np.zeros_like(kbar)), dtype=float)
        lags = np.unique(delay[np.isfinite(delay)])
        by_lag = cal.get("by_lag", {}) or {}

        def _lag_fit(d: float, pref: str):
            entry = by_lag.get(str(int(d)), by_lag.get(int(d)))
            if not isinstance(entry, dict):
                return None
            return (entry.get(f"gamma_{pref}"), entry.get(f"alpha_{pref}"),
                    entry.get(f"r2_{pref}"))

        ncol = max(1, int(lags.size))
        fig, axes = plt.subplots(2, ncol, figsize=(3.7 * ncol + 0.6, 8.4),
                                 sharey=True, squeeze=False)
        for j, d in enumerate(lags):
            sel = delay == d
            for row, (te, pref, color, xlab) in enumerate((
                (te_inj, "inj", _INJ_COLOR, r"$\mathrm{TE}_{\mathrm{inj}}$ (nats)"),
                (te_scat, "scat", _SCAT_COLOR, r"$\mathrm{TE}_{\mathrm{scat}}$ (nats)"),
            )):
                axc = axes[row][j]
                _kbar_vs_te_panel(axc, te[sel], kbar[sel], cell_id[sel],
                                  cal=cal, pref=pref, color=color, xlabel=xlab,
                                  fit=_lag_fit(d, pref))
                axc.set_title(rf"$D$={int(d)}  |  " + axc.get_title(), fontsize=8.0)
                ps.style_axes(axc)
        axes[0][0].set_ylabel(r"$\bar K$ (nats/step)")
        axes[1][0].set_ylabel(r"$\bar K$ (nats/step)")
        fig.suptitle(rf"synthetic_v2 calibration by lag (per-sample; $n$={int(kbar.size)} "
                     rf"samples, run {metrics.get('run_tag', '?')})",
                     fontsize=ps.FONT_SUPTITLE)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return _save_fig(fig, out_path, formats, dpi)

    # --- Legacy per-cell fallback (no per_sample arrays available) ------------------------
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
    r"""Write the per-cell **prediction-space** control: feat vs base vs corrupted-source feat.

    Three bars per cell -- $\mathcal L_{\mathrm{feat}}$, $\mathcal L_{\mathrm{base}}$ and
    $\mathcal L_{\mathrm{feat}}^{\pi(U)}$ for each configured input-stream corruption. The
    gate a model must clear on every signal cell is

    .. math::

        \mathcal L_{\mathrm{feat}} < \mathcal L_{\mathrm{base}}
        < \mathcal L_{\mathrm{feat}}^{\pi(U)},

    i.e. the source helps, and the *true* source is what helps. On the
    $\mathrm{TE}_{\mathrm{inj}} = 0$ null cells the three bars coincide.

    This figure replaced a per-cell plot of $\bar K_{\mathrm{null}} / \bar K_{\mathrm{signal}}$
    with a reference line at $0$ (S3-T04a). That ratio is a readout, not a gate -- it cannot
    vanish on an honest model -- and now has its own figure, :func:`plot_kl_shuffle_readout`.

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
    ctrls = sorted(k[len("feat_loss_"):] for k in pc if k.startswith("feat_loss_"))
    cell_ids = pc.get("cell_id", np.zeros(0))
    keep = np.isfinite(cell_ids)
    order = np.argsort(cell_ids[keep])
    cids = cell_ids[keep][order]
    fig, ax = plt.subplots(figsize=(max(7.5, 0.75 * len(cids) + 2.0), 4.4))
    if not ctrls or len(cids) == 0 or "feat_loss" not in pc:
        _no_data(ax, "no prediction controls in metrics")
        ps.style_axes(ax)
        return _save_fig(fig, out_path, formats, dpi)

    series = [(r"$\mathcal{L}_{\mathrm{feat}}$", pc["feat_loss"]),
              (r"$\mathcal{L}_{\mathrm{base}}$", pc["base_loss"])]
    series += [(rf"$\mathcal{{L}}_{{\mathrm{{feat}}}}^{{{c}}}$", pc[f"feat_loss_{c}"])
               for c in ctrls]
    x = np.arange(len(cids))
    nb = len(series)
    width = 0.86 / nb
    palette = ps.PALETTE_EXTENDED
    for j, (label, vals) in enumerate(series):
        ax.bar(x + (j - (nb - 1) / 2.0) * width, vals[keep][order], width=width,
               color=palette[j % len(palette)], label=label)

    # Mark the cells where the ordering fails, so a reader sees the verdict, not just bars.
    for c in ctrls:
        col = pc.get(f"ordering_pass_{c}")
        if col is None:
            continue
        vals = col[keep][order]
        for i, v in enumerate(vals):
            if np.isfinite(v) and v < 0.5:
                ax.annotate("FAIL", (x[i], 0.0), xytext=(0, -14),
                            textcoords="offset points", ha="center",
                            fontsize=5.5, color=_SCAT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(c)}" for c in cids], fontsize=6.0)
    ax.set_xlabel("cell id")
    ax.set_ylabel("clean-window forecast MSE")
    ax.set_title("prediction-space control per cell "
                 r"(gate: $\mathcal{L}_{\mathrm{feat}} < \mathcal{L}_{\mathrm{base}} "
                 r"< \mathcal{L}_{\mathrm{feat}}^{\pi(U)}$)")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5, ncol=max(1, nb // 2))
    ps.style_axes(ax)
    fig.tight_layout()
    return _save_fig(fig, out_path, formats, dpi)


def plot_kl_shuffle_readout(
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    realizability: Optional[Dict[str, Any]] = None,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Render the demoted KL-space ratio $\bar K_{\mathrm{null}} / \bar K_{\mathrm{signal}}$.

    This figure exists to make a negative result **legible**, not to hide it (S3-T04b). The
    ratio was once the headline gate, under the expectation that it vanishes on a model that
    uses the source. It cannot: $\mathrm{KL}(q \,\|\, p)$ measures that the source moved the
    posterior, not that it moved it *correctly*, and a deranged source driven through a
    posterior trained only on matched pairs is out of distribution -- so it typically moves
    the belief *more*. v3 measured $\bar K_{\mathrm{shuffled}} / \bar K_{\mathrm{true}} \in
    [1.02, 1.10]$ on a model that was demonstrably exploiting the source (Finding F2).

    The reference line therefore sits at $1$, not $0$. The discriminating gate lives in
    :func:`plot_null_controls`, which plots the prediction-space losses.

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
    cell_ids = pc.get("cell_id", np.zeros(0))
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
    ax.axhline(1.0, color=_BASELINE_COLOR, lw=0.8, ls="--")
    ax.annotate(r"$\approx 1$ is expected, even when the source IS used",
                xy=(0.5, 1.0), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=6.0, color=_BASELINE_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(c)}" for c in cids], fontsize=6.0)
    ax.set_xlabel("cell id")
    ax.set_ylabel(r"$\bar K_{\mathrm{shuffled}} / \bar K_{\mathrm{signal}}$")
    ax.set_title("KL-space null ratio per cell  (readout, NOT a gate)")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5, title="control")
    fig.text(0.01, 0.005,
             "Readout only: KL(q||p) measures that the source moved the posterior, not that "
             "it moved it correctly (v3 Finding F2).\nThe discriminating gate is the "
             "prediction-space control.",
             fontsize=5.5, color=_BASELINE_COLOR, va="bottom")
    ps.style_axes(ax)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save_fig(fig, out_path, formats, dpi)


# =============================================================================
# S8: the KLD-summary family vs TE (§14.5). The model exposes several KLD tensors
# (kld_per_t total, kld_per_t_per_head, te_lag_map) and each can be summarised over
# a sample's clean window in several ways; these figures relate every such summary
# to the transfer entropy so the reader can see WHICH KLD flavour tracks TE best,
# not just the single canonical K-bar. Backed by the per-sample arrays in
# ``per_sample_eval.npz`` and the ``calibration.kld_variants`` block of ``metrics.json``.
# =============================================================================

# Human-readable labels + display order for the KLD summary family (keys match the
# arrays ``eval_v2.collect_per_sample_kbar`` emits). Variants not listed here (e.g.
# extra per-head columns) are appended with a derived label.
KLD_VARIANT_LABELS = {
    "kbar": r"mean $K_t$ (clean window)",
    "kbar_sum": r"sum $K_t$ (integrated)",
    "kbar_max": r"max $K_t$ (peak)",
    "kbar_median": r"median $K_t$",
    "kbar_p90": r"p90 $K_t$",
    "kbar_full": r"mean $K_t$ (full seq.)",
    "kbar_postwarm": r"mean $K_t$ (post-warmup)",
    "kbar_inband": r"in-band KL ($\mathcal{L}^\star$)",
    "kbar_outband": r"out-band KL",
}

# The one variant whose units are integrated nats (not nats/step); its $y=x$ reference
# is meaningless, so the per-panel identity line is suppressed for it.
_NON_RATE_VARIANTS = frozenset({"kbar_sum"})


def _kld_variant_label(key: str) -> str:
    r"""Human-readable label for a KLD summary key (falls back for per-head columns)."""
    if key in KLD_VARIANT_LABELS:
        return KLD_VARIANT_LABELS[key]
    if str(key).startswith("kbar_head"):
        return rf"head {str(key).replace('kbar_head', '')} KL"
    return str(key)


def _kld_variant_keys(source: Dict[str, Any]) -> List[str]:
    r"""Ordered KLD summary keys present in a per-sample dict or a ``kld_variants`` block.

    Returns the canonical :data:`KLD_VARIANT_LABELS` order first, then any per-head
    ``kbar_head{m}`` columns, keeping only entries actually present (and, for a per-sample
    array dict, only non-empty 1-D arrays).

    Args:
        source: Either the ``per_sample_eval.npz`` array dict or a ``kld_variants`` dict.

    Returns:
        The ordered list of variant keys to plot.
    """
    def _present(k: str) -> bool:
        v = source.get(k)
        if v is None:
            return False
        arr = np.asarray(v)
        # A per-sample array must be a non-empty 1-D vector; a kld_variants entry is a dict.
        return isinstance(v, dict) or (arr.ndim == 1 and arr.size > 0) or arr.ndim == 0
    ordered = [k for k in KLD_VARIANT_LABELS if _present(k)]
    heads = sorted(k for k in source
                   if str(k).startswith("kbar_head") and k not in KLD_VARIANT_LABELS
                   and str(k)[len("kbar_head"):].isdigit() and _present(k))
    return ordered + heads


def _variant_fit(cal: Dict[str, Any], variant: str, pref: str) -> Optional[tuple]:
    r"""Return the ``(gamma, alpha, r2)`` per-sample fit for one variant from ``kld_variants``."""
    entry = (cal.get("kld_variants") or {}).get(variant)
    if not isinstance(entry, dict):
        return None
    return (entry.get(f"gamma_{pref}"), entry.get(f"alpha_{pref}"), entry.get(f"r2_{pref}"))


#: Colour for a KLD summary that averages over steps outside the model's KL support.
_OOS_COLOR = "#9e9e9e"


def _is_out_of_support(kld_variants: Dict[str, Any], variant: str) -> bool:
    r"""Whether ``variant`` is stamped ``out_of_support`` by :func:`eval_v2.fit_calibration`.

    ``False`` for a legacy ``metrics.json`` written before S4-T01, and for every variant under
    ``kld_support: full`` (the v1 / ``parity`` configuration), so nothing is greyed there.

    Args:
        kld_variants: The ``calibration.kld_variants`` block.
        variant: The summary key.

    Returns:
        Whether the variant must be excluded from best-variant selection.
    """
    entry = (kld_variants or {}).get(variant)
    return bool(isinstance(entry, dict) and entry.get("out_of_support", False))


def plot_kld_variants_vs_te(
    per_sample: Optional[Dict[str, Any]],
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    te_axis: str = "inj",
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write the KLD-summary-family vs TE grid (S8): one panel per KLD flavour.

    Each panel plots one per-sample KLD summary (y) against the transfer entropy (x,
    ``te_axis`` selects $\mathrm{TE}_{\mathrm{inj}}$ or $\mathrm{TE}_{\mathrm{scat}}$): the
    jittered per-sample cloud, the per-level box, the per-cell means, and that variant's own
    per-sample fit $\gamma$ with its Pearson $r$ / Spearman $\rho$. This is the direct answer
    to "different versions of KLD and different ways of summarising it, related to TE" — the
    time-summaries (mean / sum / max / median / p90), the window variants (full / post-warmup),
    the directed-KL split (in-band / out-band over $\mathcal L^\star$) and the per-head KL are
    all shown side by side so the reader can see which flavour is most calibrated. The default
    x-axis is $\mathrm{TE}_{\mathrm{inj}}$ (the exact, trustworthy label; §10); the per-variant
    correlations against BOTH TEs are ranked compactly by :func:`plot_kld_te_correlation`.

    Args:
        per_sample: The length-$N$ per-sample arrays from ``per_sample_eval.npz`` (needs the
            ``kbar*`` summary columns, ``te_inj`` / ``te_scat`` and ``cell_id``). ``None`` /
            empty renders a placeholder.
        metrics: The ``metrics.json`` dict (for ``calibration.kld_variants`` fits + labels).
        out_path: Output path stem or full path.
        te_axis: ``"inj"`` (default) or ``"scat"`` — which TE to place on the x-axis.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    cal = metrics.get("calibration", {}) or {}
    ps_arr = per_sample or {}
    pref = "scat" if str(te_axis) == "scat" else "inj"
    te_key = f"te_{pref}"
    te_label = (r"$\mathrm{TE}_{\mathrm{scat}}$" if pref == "scat"
                else r"$\mathrm{TE}_{\mathrm{inj}}$")
    variants = _kld_variant_keys(ps_arr)
    kbar = np.asarray(ps_arr.get("kbar", []), dtype=float)
    if kbar.size == 0 or not variants:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        _no_data(ax, "no per_sample_eval.npz KLD variants (run --stage eval)")
        ps.style_axes(ax)
        return _save_fig(fig, out_path, formats, dpi)

    te = np.asarray(ps_arr.get(te_key, []), dtype=float)
    cell_id = np.asarray(ps_arr.get("cell_id", np.zeros_like(kbar)), dtype=float)
    palette = ps.PALETTE_EXTENDED
    kv = cal.get("kld_variants") or {}

    n = len(variants)
    ncol = 3 if n > 4 else max(1, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol + 0.5, 3.3 * nrow + 0.5),
                             squeeze=False)
    flat = [ax for row in axes for ax in row]
    for j, v in enumerate(variants):
        ax = flat[j]
        y = np.asarray(ps_arr.get(v, []), dtype=float)
        # S4-T01: grey + annotate a summary that averages over the model's untrained region.
        oos = _is_out_of_support(kv, v)
        _kbar_vs_te_panel(
            ax, te, y, cell_id, cal=cal, pref=pref,
            color=_OOS_COLOR if oos else palette[j % len(palette)],
            xlabel=f"{te_label} (nats)",
            fit=_variant_fit(cal, v, pref), show_identity=v not in _NON_RATE_VARIANTS,
        )
        ax.set_ylabel(_kld_variant_label(v))
        title = _kld_variant_label(v) + ("  [OUT OF SUPPORT]" if oos else "")
        ax.set_title(f"{title}  |  " + ax.get_title(), fontsize=7.5)
        if oos:
            ax.text(0.02, 0.96, "averaged outside the KL support;\nnot a TE surrogate",
                    transform=ax.transAxes, ha="left", va="top", fontsize=5.5,
                    color=_OOS_COLOR)
        ps.style_axes(ax)
    for ax in flat[n:]:
        ax.set_visible(False)

    split_val = ps_arr.get("split", metrics.get("split", "?"))
    split = str(split_val.item()) if hasattr(split_val, "item") else str(split_val)
    fig.suptitle(rf"synthetic_v2 KLD summaries vs {te_label}  ($n$={int(kbar.size)} samples, "
                 rf"split {split}, run {metrics.get('run_tag', '?')})",
                 fontsize=ps.FONT_SUPTITLE)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return _save_fig(fig, out_path, formats, dpi)


def plot_kld_te_correlation(
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write the "which KLD summary tracks TE best" ranking (S8).

    A one-glance comparison across the whole KLD summary family: for every variant, the
    Pearson $r$ and (tie-aware) Spearman $\rho$ of its per-sample values against
    $\mathrm{TE}_{\mathrm{inj}}$ (left) and $\mathrm{TE}_{\mathrm{scat}}$ (right), read from
    the ``calibration.kld_variants`` block. Variants are ordered by $|\rho_{\mathrm{inj}}|$
    (best tracker on top) and each bar group is annotated with the per-sample slope $\gamma$.
    This turns "different versions of KLD vs TE" into a ranked, quantitative summary.

    A variant stamped ``out_of_support`` (S4-T01: ``kbar_full`` under anchor-aligned KL
    support) is rendered **greyed and annotated, and sorted to the bottom**, so the ranking can
    never crown a summary that is partly measuring an untrained region of the sequence.

    Args:
        metrics: The ``metrics.json`` dict from :func:`eval_v2.run_eval`.
        out_path: Output path stem or full path.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    cal = metrics.get("calibration", {}) or {}
    kv = cal.get("kld_variants") or {}
    variants = _kld_variant_keys(kv)
    fig, (ax_inj, ax_scat) = plt.subplots(1, 2, figsize=(11.0, max(3.5, 0.5 * len(variants) + 1.8)),
                                          sharey=True)
    if not variants:
        for ax in (ax_inj, ax_scat):
            _no_data(ax, "no calibration.kld_variants (run --stage eval)")
            ps.style_axes(ax)
        return _save_fig(fig, out_path, formats, dpi)

    def _val(v: str, key: str) -> float:
        x = (kv.get(v) or {}).get(key)
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    # Order by |Spearman rho vs TE_inj| so the best rank-tracker sits on top. Out-of-support
    # variants sort BELOW every in-support one regardless of their correlation: a high rho on
    # a window the model was never trained to shape is not evidence of a better surrogate.
    variants = sorted(
        variants,
        key=lambda v: (
            0 if _is_out_of_support(kv, v) else 1,
            abs(_val(v, "spearman_inj")) if np.isfinite(_val(v, "spearman_inj")) else -1.0,
        ),
    )
    labels = [_kld_variant_label(v) + (" (out of support)" if _is_out_of_support(kv, v) else "")
              for v in variants]
    yy = np.arange(len(variants))
    h = 0.38
    for ax, pref, title in ((ax_inj, "inj", r"vs $\mathrm{TE}_{\mathrm{inj}}$"),
                            (ax_scat, "scat", r"vs $\mathrm{TE}_{\mathrm{scat}}$")):
        pear = np.array([_val(v, f"pearson_{pref}") for v in variants])
        spear = np.array([_val(v, f"spearman_{pref}") for v in variants])
        pear_c = [_OOS_COLOR if _is_out_of_support(kv, v) else _INJ_COLOR for v in variants]
        spear_c = [_OOS_COLOR if _is_out_of_support(kv, v) else _SCAT_COLOR for v in variants]
        ax.barh(yy + h / 2, pear, height=h, color=pear_c, label="Pearson $r$")
        ax.barh(yy - h / 2, spear, height=h, color=spear_c, label=r"Spearman $\rho$")
        for y, v in zip(yy, variants):
            g = _val(v, f"gamma_{pref}")
            if np.isfinite(g):
                ax.text(0.02, y, rf"$\gamma$={g:.2g}", transform=ax.get_yaxis_transform(),
                        va="center", ha="left", fontsize=5.5, color=_BASELINE_COLOR)
        ax.axvline(0.0, color=_BASELINE_COLOR, lw=0.7)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel("correlation with TE")
        ax.set_title(title)
        ax.legend(loc="lower right", frameon=False, fontsize=6.5)
        ps.style_axes(ax)
    ax_inj.set_yticks(yy)
    ax_inj.set_yticklabels(labels, fontsize=7.0)
    fig.suptitle(f"synthetic_v2 KLD-summary vs TE correlation ranking  "
                 f"(run {metrics.get('run_tag', '?')})", fontsize=ps.FONT_SUPTITLE)
    if any(_is_out_of_support(kv, v) for v in variants):
        fig.text(0.01, 0.005,
                 "Greyed: averaged over steps outside the model's KL support (anchor-aligned "
                 "training); excluded from best-variant selection. Use kbar_postwarm.",
                 fontsize=5.5, color=_BASELINE_COLOR, va="bottom")
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_fig(fig, out_path, formats, dpi)


def plot_kld_te_density(
    per_sample: Optional[Dict[str, Any]],
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    variant: str = "kbar",
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write the per-sample $\bar K$-vs-$\mathrm{TE}_{\mathrm{inj}}$ density + residual view (S8).

    A finer look at the primary relationship than the discrete scatter: a hexbin density of
    every sample's KLD summary ``variant`` (default $\bar K$) against $\mathrm{TE}_{\mathrm{inj}}$
    (x lightly jittered so the within-level density is legible) with the pooled per-sample fit
    overlaid (left), and the fit residuals $\bar K_i - (\alpha + \gamma\,\mathrm{TE}_i)$ boxed per
    TE level (right) — the residual panel exposes nonlinearity and heteroscedasticity that a bare
    slope hides. The colour is sample count per hex.

    Args:
        per_sample: The length-$N$ per-sample arrays from ``per_sample_eval.npz``. ``None`` /
            empty renders a placeholder.
        metrics: The ``metrics.json`` dict (for the fit; ``variant``'s fit is used when present).
        out_path: Output path stem or full path.
        variant: Which KLD summary column to plot (default ``"kbar"``).
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    cal = metrics.get("calibration", {}) or {}
    ps_arr = per_sample or {}
    y = np.asarray(ps_arr.get(variant, []), dtype=float)
    te = np.asarray(ps_arr.get("te_inj", []), dtype=float)
    fig, (ax_den, ax_res) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    m = np.isfinite(y) & np.isfinite(te)
    if int(m.sum()) < 2 or np.ptp(te[m]) <= 0:
        for ax in (ax_den, ax_res):
            _no_data(ax, "no per-sample KLD/TE data (run --stage eval)")
            ps.style_axes(ax)
        return _save_fig(fig, out_path, formats, dpi)
    y, te = y[m], te[m]

    # Per-variant fit if available, else the pooled kbar fit, else an on-the-fly OLS.
    fit = _variant_fit(cal, variant, "inj")
    if fit is not None and fit[0] is not None and fit[1] is not None:
        g, a = float(fit[0]), float(fit[1])
    elif cal.get("gamma_inj_sample") is not None:
        g, a = float(cal["gamma_inj_sample"]), float(cal.get("alpha_inj_sample", 0.0))
    else:
        g, a = np.polyfit(te, y, 1)

    span = float(np.ptp(te)) or 1.0
    rng = np.random.default_rng(0)
    te_j = te + rng.uniform(-0.02 * span, 0.02 * span, size=te.shape[0])
    hb = ax_den.hexbin(te_j, y, gridsize=32, cmap="viridis", mincnt=1, linewidths=0.0)
    ps.add_colorbar(fig, hb, ax_den, label="samples per hex")
    xs = np.linspace(float(te.min()), float(te.max()), 50)
    ax_den.plot(xs, a + g * xs, color=_SCAT_COLOR, lw=1.6,
                label=rf"fit $\gamma$={g:.3f}")
    ax_den.set_xlabel(r"$\mathrm{TE}_{\mathrm{inj}}$ (nats)")
    ax_den.set_ylabel(_kld_variant_label(variant))
    ax_den.set_title(rf"density  ($n$={int(y.size)}, $r$={float(np.corrcoef(te, y)[0, 1]):.2f})")
    ax_den.legend(loc="upper left", frameon=False, fontsize=7.0)
    ps.style_axes(ax_den)

    # Residuals about the fit, boxed per TE level: reveals curvature / spread structure.
    resid = y - (a + g * te)
    levels = np.unique(te)
    data = [resid[te == lv] for lv in levels]
    bw = 0.05 * span + 1e-3
    bp = ax_res.boxplot(data, positions=levels, widths=2 * bw, showfliers=False,
                        patch_artist=True, manage_ticks=False)
    for patch in bp["boxes"]:
        patch.set(facecolor="white", edgecolor=_INJ_COLOR, alpha=0.9, linewidth=0.9)
    for med in bp["medians"]:
        med.set(color=_SCAT_COLOR, linewidth=1.3)
    for part in bp["whiskers"] + bp["caps"]:
        part.set(color=_BASELINE_COLOR, linewidth=0.7)
    ax_res.axhline(0.0, color=_BASELINE_COLOR, lw=0.8, ls="--")
    ax_res.set_xlabel(r"$\mathrm{TE}_{\mathrm{inj}}$ (nats)")
    ax_res.set_ylabel("residual (obs $-$ fit)")
    ax_res.set_title("fit residuals by TE level")
    ps.style_axes(ax_res)

    fig.suptitle(rf"synthetic_v2 {_kld_variant_label(variant)} vs TE density  "
                 rf"(run {metrics.get('run_tag', '?')})", fontsize=ps.FONT_SUPTITLE)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_fig(fig, out_path, formats, dpi)


def _violin_by_level(ax, te: np.ndarray, y: np.ndarray, *, color: str) -> int:
    r"""Draw per-TE-level violins of ``y`` with the per-level mean overlaid; return level count."""
    m = np.isfinite(te) & np.isfinite(y)
    if int(m.sum()) < 2:
        _no_data(ax, "no per-sample data")
        return 0
    te, y = te[m], y[m]
    levels = np.unique(te)
    data = [y[te == lv] for lv in levels if np.isfinite(y[te == lv]).any()]
    pos = [float(lv) for lv in levels if np.isfinite(y[te == lv]).any()]
    if not data:
        _no_data(ax, "no per-sample data")
        return 0
    span = float(np.ptp(levels)) or 1.0
    parts = ax.violinplot(data, positions=pos, widths=0.12 * span + 1e-3,
                          showmeans=True, showextrema=False)
    for body in parts["bodies"]:
        body.set(facecolor=color, edgecolor=_BASELINE_COLOR, alpha=0.35, linewidth=0.7)
    if "cmeans" in parts:
        parts["cmeans"].set(color=color, linewidth=1.4)
    return len(pos)


def plot_kld_distribution_by_te(
    per_sample: Optional[Dict[str, Any]],
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    variant: str = "kbar",
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write per-TE-level violin distributions of a KLD summary (S8).

    The full per-sample $\bar K$ distribution at each discrete TE level as violins (with the
    per-level mean), side by side for $\mathrm{TE}_{\mathrm{inj}}$ (left) and
    $\mathrm{TE}_{\mathrm{scat}}$ (right). Where the discrete scatter shows individual points,
    this shows the *shape* and *separation* of the $\bar K$ distribution across TE levels — how
    cleanly the model's KLD separates neighbouring transfer-entropy levels.

    Args:
        per_sample: The length-$N$ per-sample arrays from ``per_sample_eval.npz``. ``None`` /
            empty renders a placeholder.
        metrics: The ``metrics.json`` dict (for run/split labels).
        out_path: Output path stem or full path.
        variant: Which KLD summary column to plot (default ``"kbar"``).
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    ps_arr = per_sample or {}
    y = np.asarray(ps_arr.get(variant, []), dtype=float)
    fig, (ax_inj, ax_scat) = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    if y.size == 0:
        for ax in (ax_inj, ax_scat):
            _no_data(ax, "no per_sample_eval.npz (run --stage eval)")
            ps.style_axes(ax)
        return _save_fig(fig, out_path, formats, dpi)
    for ax, te_key, color, lab in (
        (ax_inj, "te_inj", _INJ_COLOR, r"$\mathrm{TE}_{\mathrm{inj}}$"),
        (ax_scat, "te_scat", _SCAT_COLOR, r"$\mathrm{TE}_{\mathrm{scat}}$"),
    ):
        te = np.asarray(ps_arr.get(te_key, []), dtype=float)
        _violin_by_level(ax, te, y, color=color)
        ax.set_xlabel(f"{lab} (nats)")
        ax.set_title(f"{lab}")
        ps.style_axes(ax)
    ax_inj.set_ylabel(_kld_variant_label(variant))
    fig.suptitle(rf"synthetic_v2 {_kld_variant_label(variant)} distribution by TE level  "
                 rf"(run {metrics.get('run_tag', '?')})", fontsize=ps.FONT_SUPTITLE)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_fig(fig, out_path, formats, dpi)


def plot_per_head_kld_vs_te(
    per_sample: Optional[Dict[str, Any]],
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Write the per-head KL vs $\mathrm{TE}_{\mathrm{inj}}$ figure (S8).

    The latent KL splits additively into ``num_heads`` contiguous-group KLs
    (``kld_per_t_per_head``); this plots each head's per-sample clean-window mean KL against
    $\mathrm{TE}_{\mathrm{inj}}$ as per-level means with a per-head OLS trend, so it is visible
    *which* latent head carries the injected coupling (its KL rises with TE) versus which stay
    flat. The total $\bar K$ per-level mean is drawn as a gray reference.

    Args:
        per_sample: The length-$N$ per-sample arrays from ``per_sample_eval.npz`` (needs the
            ``kbar_head{m}`` columns). ``None`` / empty (or no head columns) renders a placeholder.
        metrics: The ``metrics.json`` dict (for run/split labels).
        out_path: Output path stem or full path.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    ps_arr = per_sample or {}
    head_keys = sorted((k for k in ps_arr
                        if str(k).startswith("kbar_head")
                        and str(k)[len("kbar_head"):].isdigit()),
                       key=lambda k: int(str(k)[len("kbar_head"):]))
    te = np.asarray(ps_arr.get("te_inj", []), dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    if not head_keys or te.size == 0:
        _no_data(ax, "no per-head KLD columns (run --stage eval)")
        ps.style_axes(ax)
        return _save_fig(fig, out_path, formats, dpi)

    palette = ps.PALETTE_EXTENDED
    for j, hk in enumerate(head_keys):
        y = np.asarray(ps_arr.get(hk, []), dtype=float)
        stats = _group_stats(te, y)
        groups = sorted(stats)
        if not groups:
            continue
        gx = np.array(groups)
        gy = np.array([stats[g]["mean"] for g in groups])
        color = palette[j % len(palette)]
        rho = _spearman_rho(te, y)
        ax.plot(gx, gy, marker="o", ms=4, lw=1.0, color=color,
                label=rf"head {int(str(hk)[len('kbar_head'):])} ($\rho$={rho:.2f})")
        fit = _ols_line(te, y)
        if fit is not None:
            xs, ys, _ = fit
            ax.plot(xs, ys, color=color, lw=0.8, ls="--", alpha=0.7)
    # Total K-bar per-level mean as a gray reference.
    kbar = np.asarray(ps_arr.get("kbar", []), dtype=float)
    if kbar.size:
        st = _group_stats(te, kbar)
        gs = sorted(st)
        if gs:
            ax.plot(np.array(gs), np.array([st[g]["mean"] for g in gs]),
                    marker="s", ms=4, lw=1.2, color=_BASELINE_COLOR, label=r"total $\bar K$")
    ax.set_xlabel(r"$\mathrm{TE}_{\mathrm{inj}}$ (nats)")
    ax.set_ylabel(r"per-head mean KL (nats/step)")
    ax.set_title(rf"per-head KL vs TE  (which latent head carries the coupling; "
                 rf"run {metrics.get('run_tag', '?')})")
    ax.legend(loc="upper left", frameon=False, fontsize=6.5)
    ps.style_axes(ax)
    fig.tight_layout()
    return _save_fig(fig, out_path, formats, dpi)


# ---------------------------------------------------------------------------
# Sprint 5 (G-E) / Sprint 6 (G-F): the two analysis figures
# ---------------------------------------------------------------------------
def plot_calibration_by_te(
    by_te: Any,
    out_path: Union[str, Path],
    *,
    level: float = 0.9,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Coverage error and sharpness, resolved by injected TE level and horizon (S5-T03).

    Two heatmaps over the $(\mathrm{TE}_{\mathrm{inj}}, h)$ grid. The left panel shows the
    signed coverage error $c - p$ at a single nominal level $p$ on a diverging scale centred at
    zero: blue is over-coverage (intervals too wide), red is over-confidence. The right panel
    shows the predictive spread $\bar\sigma$.

    The diagnostic question is whether the learned variance is honest *uniformly*. A row of red
    at high $\mathrm{TE}_{\mathrm{inj}}$ says the model is over-confident exactly where the
    coupling is strong -- a failure $\bar K$ alone could never surface.

    Args:
        by_te: The long-format ``calibration_by_te`` table (a ``pandas.DataFrame`` or a list of
            row dicts) keyed ``(te_level, horizon, level)``.
        out_path: Output stem or full path.
        level: The nominal central-interval level to render.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    rows = by_te.to_dict("records") if hasattr(by_te, "to_dict") else list(by_te or [])
    rows = [r for r in rows if abs(float(r["level"]) - float(level)) < 1e-9]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    if not rows:
        for ax in axes:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=9)
            ax.set_axis_off()
        return _save_fig(fig, out_path, formats, dpi)

    te_levels = sorted({float(r["te_level"]) for r in rows})
    horizons = sorted({int(r["horizon"]) for r in rows})
    cov_err = np.full((len(te_levels), len(horizons)), np.nan)
    sharp = np.full((len(te_levels), len(horizons)), np.nan)
    for r in rows:
        i, j = te_levels.index(float(r["te_level"])), horizons.index(int(r["horizon"]))
        cov_err[i, j] = float(r["coverage_error"])
        sharp[i, j] = float(r["sharpness"])

    lim = float(np.nanmax(np.abs(cov_err))) if np.isfinite(cov_err).any() else 1.0
    lim = max(lim, 1e-6)
    panels = (
        (axes[0], cov_err, "coolwarm_r", Normalize(vmin=-lim, vmax=lim),
         rf"coverage error $c - {level:g}$"),
        (axes[1], sharp, "viridis", None, r"sharpness $\bar\sigma$"),
    )
    for ax, data, cmap, norm, title in panels:
        im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, norm=norm,
                       extent=(horizons[0] - 0.5, horizons[-1] + 0.5, -0.5,
                               len(te_levels) - 0.5))
        ax.set_yticks(range(len(te_levels)))
        ax.set_yticklabels([f"{t:g}" for t in te_levels])
        ax.set_xlabel("horizon step $h$")
        ax.set_ylabel(r"$\mathrm{TE}_{\mathrm{inj}}$ (nats)")
        ax.set_title(title, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ps.style_axes(ax)
    fig.suptitle("Predictive calibration stratified by injected TE", fontsize=9)
    fig.tight_layout()
    return _save_fig(fig, out_path, formats, dpi)


def plot_lag_intervention(
    summary: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    formats: tuple = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Interventional importance versus attention mass, per cell (S6-T04).

    One small multiple per cell. Bars are the relative interventional importance
    $\Delta L_{\mathrm{rel}}(G) = \Delta L_G / \mathcal{L}_{\mathrm{feat}}$ of each fixed
    physiologic band; the overlaid line is the normalised ``te_lag_map`` mass the attention puts
    on the same band. If the attention is a faithful attribution, the two agree.

    The true lag $D$ is drawn as a vertical marker on the band that contains it, and the panel
    title carries the per-cell in-band gate
    $\Delta L_{\mathrm{rel}}(\mathcal{L}^\star) > \Delta L_{\mathrm{rel}}(\{\ell \ge D\})$.

    Args:
        summary: The ``lag_intervention.json`` payload.
        out_path: Output stem or full path.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    bands: Dict[str, Any] = summary.get("bands") or {}
    per_cell: Dict[str, Any] = summary.get("per_cell") or {}
    cells = sorted(per_cell.values(), key=lambda c: (c.get("te_inj") or 0.0, c.get("delay") or 0))
    if not cells or not bands:
        fig, ax = plt.subplots(figsize=(5.0, 2.5))
        ax.text(0.5, 0.5, "no lag-intervention data", ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        return _save_fig(fig, out_path, formats, dpi)

    names = list(bands)
    x = np.arange(len(names))
    ncol = min(3, len(cells))
    nrow = int(np.ceil(len(cells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.5 * nrow), squeeze=False)

    for idx, cell in enumerate(cells):
        ax = axes[idx // ncol][idx % ncol]
        delta = np.array([cell.get(f"delta_L_rel_{n}", np.nan) for n in names], dtype=float)
        mass = np.array([cell.get(f"mass_{n}", np.nan) for n in names], dtype=float)

        ax.bar(x, delta, color=_INJ_COLOR, alpha=0.85, label=r"$\Delta L_{\mathrm{rel}}$")
        ax.axhline(0.0, color=_BASELINE_COLOR, lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=6, rotation=45, ha="right")
        ax.set_ylabel(r"$\Delta L_{\mathrm{rel}}$", fontsize=7)

        twin = ax.twinx()
        twin.plot(x, mass, color=_SCAT_COLOR, marker="o", ms=3, lw=1.0,
                  label="attention mass")
        twin.set_ylabel("te_lag_map mass", fontsize=7, color=_SCAT_COLOR)
        twin.tick_params(axis="y", labelcolor=_SCAT_COLOR, labelsize=6)

        # Mark the band containing the true lag D.
        delay = cell.get("delay")
        if delay is not None:
            for j, name in enumerate(names):
                lo, hi = bands[name]
                if int(lo) <= int(delay) <= int(hi):
                    ax.axvspan(j - 0.5, j + 0.5, color=_BAND_COLOR, alpha=0.15, zorder=0)
                    ax.annotate(rf"$D={delay}$", xy=(j, 0), xytext=(j, 0),
                                fontsize=6, color=_BAND_COLOR, ha="center", va="bottom")
                    break

        te = cell.get("te_inj")
        gate = cell.get("inband_gate_pass")
        verdict = "—" if gate is None else ("pass" if gate else "FAIL")
        ax.set_title(
            rf"cell {cell.get('cell_id')}  $\mathrm{{TE}}={te:.1f}$  "
            rf"$\mathcal{{L}}^\star$ gate: {verdict}" if te is not None else "",
            fontsize=7,
        )
        ps.style_axes(ax)

    for idx in range(len(cells), nrow * ncol):
        axes[idx // ncol][idx % ncol].set_axis_off()

    overall = summary.get("overall") or {}
    frac = overall.get("inband_gate_pass_frac")
    frac_txt = "n/a" if frac is None else f"{frac:.0%}"
    fig.suptitle(
        f"Interventional lag attribution (arm {summary.get('arm', '?')}, "
        f"split {summary.get('split', '?')}): in-band gate {frac_txt} of "
        f"{overall.get('n_signal_cells', '?')} signal cells",
        fontsize=9,
    )
    fig.tight_layout()
    return _save_fig(fig, out_path, formats, dpi)


#: Series drawn by :func:`plot_cmi_comparison`, in legend order.
_CMI_SERIES: Tuple[Tuple[str, str, str], ...] = (
    ("cmi_latent", "CMI (ground-truth latents)", "o"),
    ("cmi_feature_gt", "CMI (features, GT conditioning)", "s"),
    ("cmi_feature_model", r"CMI (features, $\mathtt{target\_state}$)", "^"),
)


def plot_cmi_comparison(
    summary: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = _DPI,
) -> List[Path]:
    r"""Per-cell $\mathrm{TE}_{\mathrm{inj}}$ against $\bar K$ and each CMI configuration.

    Left panel: every estimate versus the injected TE, with $95\%$ intervals, the identity line,
    and the **InfoNCE ceiling** $\log K$ drawn and labelled. Cells stamped ``near_ceiling`` are
    marked, because no absolute-nats claim is made for them. Right panel: the per-cell
    model-coupling bias $\mathrm{CMI}(\texttt{target\_state}) - \mathrm{CMI}(\text{GT})$.

    Args:
        summary: The ``cmi.json`` payload from :func:`cmi_v3.run_cmi_comparison`.
        out_path: Stem (no suffix) for the rendered files.
        formats: Output formats.
        dpi: Raster resolution.

    Returns:
        The written paths.
    """
    per_cell = summary.get("per_cell") or {}
    cells = [per_cell[k] for k in sorted(per_cell, key=lambda x: int(x))]
    if not cells:
        fig, ax = plt.subplots(figsize=(5.0, 2.5))
        ax.text(0.5, 0.5, "no CMI data", ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        return _save_fig(fig, out_path, formats, dpi)

    configs = summary.get("configs") or []
    ceiling = summary.get("ceiling_nats")
    claim_frac = summary.get("ceiling_claim_frac") or 0.0
    te = np.array([c.get("te_inj", np.nan) for c in cells], dtype=float)
    kbar = np.array([c.get("kbar", np.nan) for c in cells], dtype=float)

    has_bias = any("bias" in c for c in cells)
    fig, axes = plt.subplots(1, 2 if has_bias else 1, figsize=(9.0 if has_bias else 5.0, 3.4),
                             squeeze=False)
    ax = axes[0][0]

    lim = float(np.nanmax(te)) * 1.08 if np.isfinite(te).any() else 1.0
    ax.plot([0, lim], [0, lim], color=_BASELINE_COLOR, lw=0.8, ls=":", label="identity", zorder=1)

    colors = (_INJ_COLOR, _SCAT_COLOR, _BAND_COLOR)
    for (key, label, marker), color in zip(_CMI_SERIES, colors):
        name = key.replace("cmi_", "")
        if name not in configs:
            continue
        est = np.array([(c.get(key) or {}).get("estimate", np.nan) for c in cells], dtype=float)
        lo = np.array([(c.get(key) or {}).get("ci_lo", np.nan) for c in cells], dtype=float)
        hi = np.array([(c.get(key) or {}).get("ci_hi", np.nan) for c in cells], dtype=float)
        err = np.vstack([np.clip(est - lo, 0, None), np.clip(hi - est, 0, None)])
        ax.errorbar(te, est, yerr=err, fmt=marker, ms=4, lw=0.9, capsize=2,
                    color=color, label=label, zorder=3)
        near = np.array([bool((c.get(key) or {}).get("near_ceiling")) for c in cells])
        if near.any():
            ax.scatter(te[near], est[near], s=70, facecolors="none", edgecolors=color,
                       lw=0.8, zorder=4)

    ax.plot(te, kbar, marker="x", ms=5, mew=1.2, lw=0.0, color="#444444",
            label=r"$\bar K$ (model KL surrogate)", zorder=5)

    if ceiling:
        ax.axhline(ceiling, color=_SCAT_COLOR, lw=0.9, ls="--", zorder=1)
        ax.annotate(rf"InfoNCE ceiling $\log K = {ceiling:.2f}$ nats",
                    xy=(0.98, ceiling), xycoords=("axes fraction", "data"),
                    va="bottom", ha="right", fontsize=6, color=_SCAT_COLOR)
        if claim_frac:
            ax.axhline(claim_frac * ceiling, color=_SCAT_COLOR, lw=0.6, ls=":", alpha=0.7)
            ax.annotate(rf"no absolute claim above {claim_frac * ceiling:.2f}",
                        xy=(0.98, claim_frac * ceiling), xycoords=("axes fraction", "data"),
                        va="bottom", ha="right", fontsize=5.5, color=_SCAT_COLOR, alpha=0.8)
        ax.set_ylim(top=ceiling * 1.12)

    ax.set_xlabel(r"$\mathrm{TE}_{\mathrm{inj}}$ (nats)", fontsize=8)
    ax.set_ylabel("nats", fontsize=8)
    ax.legend(fontsize=5.5, loc="lower right", framealpha=0.9)
    ps.style_axes(ax)

    if has_bias:
        ax2 = axes[0][1]
        bias = np.array([(c.get("bias") or {}).get("estimate", np.nan) for c in cells], dtype=float)
        blo = np.array([(c.get("bias") or {}).get("ci_lo", np.nan) for c in cells], dtype=float)
        bhi = np.array([(c.get("bias") or {}).get("ci_hi", np.nan) for c in cells], dtype=float)
        err = np.vstack([np.clip(bias - blo, 0, None), np.clip(bhi - bias, 0, None)])
        ax2.errorbar(te, bias, yerr=err, fmt="o", ms=4, lw=0.9, capsize=2, color=_INJ_COLOR)
        ax2.axhline(0.0, color=_BASELINE_COLOR, lw=0.8)
        ax2.set_xlabel(r"$\mathrm{TE}_{\mathrm{inj}}$ (nats)", fontsize=8)
        ax2.set_ylabel(r"$\mathrm{CMI}(\mathtt{target\_state}) - \mathrm{CMI}(\mathrm{GT})$",
                       fontsize=7)
        ax2.set_title("model-coupling bias (rank-level claim only)", fontsize=8)
        ps.style_axes(ax2)

    rec = summary.get("recovery") or {}
    rho = rec.get("spearman_cmi_te_inj")
    rho_txt = "n/a" if rho is None else f"{rho:.3f}"
    fig.suptitle(
        f"Neural CMI vs injected TE (arm {summary.get('arm', '?')}, "
        f"split {summary.get('split', '?')}): "
        rf"$\rho(\mathrm{{CMI}}_{{\mathrm{{latent}}}}, \mathrm{{TE}}_{{\mathrm{{inj}}}})$ = "
        f"{rho_txt} over {rec.get('n_cells', '?')} cells",
        fontsize=9,
    )
    fig.tight_layout()
    return _save_fig(fig, out_path, formats, dpi)
