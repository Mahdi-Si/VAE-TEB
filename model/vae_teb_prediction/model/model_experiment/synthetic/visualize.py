r"""Visualisation utilities for cached synthetic TE datasets (Decision D7).

Renders a human-readable preview of a generated benchmark so the data can be
inspected before training and reused with confidence. The preview is a single
multi-panel ``preview.pdf`` (plus a ``preview.png``) written next to the cached
``.npz`` splits.

Public API:
    make_preview: build the one-page ``preview.{pdf,png}`` for a cached dataset.
    make_dataset_gallery: render a multi-figure ``figures/`` gallery (input /
        source heatmaps, native model fields, the forecast-target window, the
        lag structure, the analytic-TE breakdown, value distributions) so the
        inputs, outputs and transfer structure of a dataset can be inspected in
        full detail rather than from the single condensed preview.

The preview is **benchmark-aware** (Phase 7): panels 1, 2 and 5 are common to
every benchmark, while panels 3, 4 and 6 are dispatched on
``meta["benchmark"]``:

    * A / B / G (linear-Gaussian): delay-alignment scatter, per-channel lagged
      cross-correlation, and a Gaussian parameter summary.
    * C (delayed XOR): a bit-agreement scatter and per-channel bit-agreement
      rate $P(Y_j(t) = X_j(t-D))$.
    * E (two-lag Gaussian): a two-group delay-alignment scatter and per-channel
      lagged correlation evaluated at each group's own delay.

Note:
    Generators run with ``standardize=True``, which z-scores every channel to
    unit variance, so a raw variance-ratio bar carries no contrast. The
    standardisation-invariant **lagged cross-correlation** (and, for XOR, the
    sign-based bit-agreement rate) is plotted instead.

All figures use the shared publication style in :mod:`plot_style`.

Used by :mod:`build_dataset` and runnable standalone against any cached dataset.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from model.vae_teb_prediction.model.model_experiment.synthetic import (
    plot_style as ps,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.analytic_te import (
    te_block_gaussian,
)

ps.apply_style()

_MAX_LAG = 90  # the model's attention window is lags 0..90 (max_lag + 1 = 91).

# Stable colour roles across every preview panel.
_C_INFORMATIVE = ps.COLOR_BLUE      # informative channels / lag-group 1
_C_DISTRACTOR = ps.COLOR_ORANGE     # non-informative distractor channels
_C_SOURCE = ps.COLOR_GREEN          # source-stream traces / lag-group 2
_C_BAND = ps.COLOR_VERMILLION       # true source-lag band
_C_DELAY = ps.COLOR_GRAY            # delay reference line
_C_ANALYTIC = ps.COLOR_BLACK        # analytic reference lines


def _load_split(npz_path: Path, n_sub: int) -> Dict[str, np.ndarray]:
    """Load the first ``n_sub`` samples of a cached split into RAM.

    Args:
        npz_path: Path to a ``{train,val,test}.npz``.
        n_sub: Number of leading samples to load (keeps the preview cheap).

    Returns:
        Dict with ``Y`` $(n, T, 87)$, ``U`` $(n, T, 101)$ and ``weight``
        $(n, T)$ as in-memory ``float32`` arrays.
    """
    with np.load(npz_path) as npz:
        n = min(int(n_sub), int(npz["fhr_st"].shape[0]))
        fhr_st = np.asarray(npz["fhr_st"][:n], dtype=np.float32)
        fhr_ph = np.asarray(npz["fhr_ph"][:n], dtype=np.float32)
        up_st = np.asarray(npz["up_st"][:n], dtype=np.float32)
        up_ph = np.asarray(npz["up_ph"][:n], dtype=np.float32)
        weight = np.asarray(npz["weight"][:n], dtype=np.float32)
    return {
        "Y": np.concatenate([fhr_st, fhr_ph], axis=-1),
        "U": np.concatenate([up_st, up_ph], axis=-1),
        "weight": weight,
    }


def _primary_delay(meta: Dict[str, Any]) -> int:
    """Return a representative delay for the trace panels.

    Args:
        meta: The dataset metadata dict.

    Returns:
        ``meta["delay"]`` for single-delay benchmarks, else ``meta["delay1"]``.
    """
    return int(meta.get("delay", meta.get("delay1", 0)))


def _lagged_corr_per_channel(
    U: np.ndarray, Y: np.ndarray, delay: int
) -> np.ndarray:
    r"""Pooled Pearson correlation $\mathrm{corr}(U_j(t-D),\,Y_j(t))$ per channel.

    Args:
        U: Source tensor $(n, T, C_u)$.
        Y: Target tensor $(n, T, C_y)$.
        delay: Source-to-target delay $D$.

    Returns:
        Correlation per target channel, shape $(C_y,)$ (each $Y$ channel
        paired with the source channel of the same index).
    """
    c_y = Y.shape[-1]
    src = U[:, : U.shape[1] - delay, :c_y].reshape(-1, c_y)
    tgt = Y[:, delay:, :].reshape(-1, c_y)
    src = src - src.mean(axis=0, keepdims=True)
    tgt = tgt - tgt.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(src, axis=0) * np.linalg.norm(tgt, axis=0)
    return np.where(denom > 0.0, (src * tgt).sum(axis=0) / (denom + 1e-12), 0.0)


def _panels_traces(axes: np.ndarray, Y: np.ndarray, U: np.ndarray,
                   meta: Dict[str, Any]) -> None:
    """Fill panels 1-2 (target / source traces) -- common to every benchmark.

    Args:
        axes: The $3 \\times 2$ array of subplot axes.
        Y: Target tensor $(n, T, C_y)$.
        U: Source tensor $(n, T, C_u)$.
        meta: The dataset metadata dict.
    """
    T = Y.shape[1]
    t_axis = np.arange(T)
    delay = _primary_delay(meta)
    M = int(meta["M"])
    c_y = int(meta["c_y"])
    distractor = M if M < c_y else 0  # falls back to ch 0 in the easy variant.

    ax = axes[0, 0]
    ax.plot(t_axis, Y[0, :, 0], lw=0.9, color=_C_INFORMATIVE,
            label="informative ch 0")
    if M < c_y:
        ax.plot(
            t_axis, Y[0, :, distractor], lw=0.9, color=_C_DISTRACTOR,
            alpha=0.8, label=f"distractor ch {distractor}",
        )
    ax.axvline(delay, color=_C_DELAY, ls="--", lw=0.9, label=f"delay D={delay}")
    ax.set_title("Target $Y$ -- sample 0")
    ax.set_xlabel("time step $t$")
    ax.set_ylabel("standardised value")
    ps.tighten_xaxis(ax, t_axis)
    ax.legend(loc="upper right")
    ps.style_axes(ax)

    ax = axes[0, 1]
    ax.plot(t_axis, U[0, :, 0], lw=0.9, color=_C_SOURCE, label="source ch 0")
    ax.axvline(delay, color=_C_DELAY, ls="--", lw=0.9, label=f"delay D={delay}")
    ax.set_title("Source $U$ -- sample 0")
    ax.set_xlabel("time step $t$")
    ax.set_ylabel("standardised value")
    ps.tighten_xaxis(ax, t_axis)
    ax.legend(loc="upper right")
    ps.style_axes(ax)


def _panel_lag_band(ax: plt.Axes, meta: Dict[str, Any]) -> None:
    """Fill panel 5 (true source-lag band over the attention axis).

    Handles the single band (A / B / C), the two bands (E), and the empty band
    of the reverse-roles directionality benchmark (G).

    Args:
        ax: The target subplot axis.
        meta: The dataset metadata dict.
    """
    lag_axis = np.arange(_MAX_LAG + 1)
    benchmark = meta.get("benchmark", "?")
    if benchmark == "E":
        b1 = set(meta.get("lag_band_1", []))
        b2 = set(meta.get("lag_band_2", []))
        ax.bar(lag_axis, [1.0 if l in b1 else 0.0 for l in lag_axis],
               width=1.0, color=_C_INFORMATIVE, label="band 1 ($D_1$)")
        ax.bar(lag_axis, [1.0 if l in b2 else 0.0 for l in lag_axis],
               width=1.0, color=_C_BAND, alpha=0.8, label="band 2 ($D_2$)")
        ax.legend(loc="upper right")
        ax.set_title(f"True source-lag bands (two-lag)\n"
                     f"(model attention window 0..{_MAX_LAG})")
    else:
        band = sorted(set(meta.get("true_lag_band", [])))
        ax.bar(lag_axis, [1.0 if l in set(band) else 0.0 for l in lag_axis],
               width=1.0, color=_C_BAND)
        if band:
            title = (f"True source-lag band {band[0]}..{band[-1]}\n"
                     f"(model attention window 0..{_MAX_LAG})")
        else:
            title = ("No causal lag band\n"
                     "(reverse-roles directionality benchmark -- te_true=0)")
        ax.set_title(title)
    ax.set_xlabel("source lag $\\ell$")
    ax.set_ylabel("carries transfer")
    ax.set_xlim(-0.5, _MAX_LAG + 0.5)
    ax.set_ylim(0, 1.2)
    ps.style_axes(ax)


def _scatter_alignment(ax: plt.Axes, src: np.ndarray, tgt: np.ndarray,
                       color: str = _C_INFORMATIVE) -> float:
    """Scatter ``src`` against ``tgt`` (downsampled) and return their correlation.

    Args:
        ax: The target subplot axis.
        src: Flattened source values.
        tgt: Flattened target values.
        color: Marker colour.

    Returns:
        The empirical Pearson correlation of ``src`` and ``tgt``.
    """
    if src.size > 6000:
        sel = np.random.default_rng(0).choice(src.size, 6000, replace=False)
        src, tgt = src[sel], tgt[sel]
    ax.scatter(src, tgt, s=4, alpha=0.25, color=color, edgecolors="none")
    return float(np.corrcoef(src, tgt)[0, 1]) if src.size > 1 else 0.0


def _panels_gaussian(axes: np.ndarray, Y: np.ndarray, U: np.ndarray,
                    meta: Dict[str, Any]) -> None:
    """Fill panels 3, 4, 6 for the linear-Gaussian benchmarks (A / B / G).

    Args:
        axes: The $3 \\times 2$ array of subplot axes.
        Y: Target tensor $(n, T, C_y)$.
        U: Source tensor $(n, T, C_u)$.
        meta: The dataset metadata dict.
    """
    T = Y.shape[1]
    delay = int(meta["delay"])
    M = int(meta["M"])
    c_y = int(meta["c_y"])
    a = float(meta["a"])
    sigma2 = float(meta["sigma2"])
    reverse = bool(meta.get("reverse_roles", False))
    analytic_corr = (
        0.0 if reverse
        else (a / math.sqrt(a * a + sigma2) if (a * a + sigma2) > 0 else 0.0)
    )

    # --- Panel 3: delay-alignment scatter ---------------------------------
    ax = axes[1, 0]
    src = U[:, : T - delay, 0].reshape(-1)
    tgt = Y[:, delay:, 0].reshape(-1)
    emp = _scatter_alignment(ax, src, tgt, color=_C_INFORMATIVE)
    note = "  (reverse roles: anti-causal, expect ~0)" if reverse else ""
    ax.set_title(
        f"Delay alignment ch 0:  $U(t-D)$ vs $Y(t)$\n"
        f"empirical corr={emp:.3f}   analytic={analytic_corr:.3f}{note}"
    )
    ax.set_xlabel(f"$U(t-{delay})$, channel 0")
    ax.set_ylabel("$Y(t)$, channel 0")
    ps.style_axes(ax)

    # --- Panel 4: per-channel lagged correlation --------------------------
    ax = axes[1, 1]
    corr = _lagged_corr_per_channel(U, Y, delay)
    colors = [_C_INFORMATIVE if j < M else _C_DISTRACTOR for j in range(c_y)]
    ax.bar(np.arange(c_y), corr, color=colors, width=1.0)
    ax.axhline(analytic_corr, color=_C_ANALYTIC, ls="--", lw=1.0,
               label=f"analytic informative={analytic_corr:.3f}")
    ax.set_title(
        f"Per-channel lagged corr $\\mathrm{{corr}}(U_j(t-D), Y_j(t))$\n"
        f"informative=blue ($j<M={M}$), distractor=orange"
    )
    ax.set_xlabel("target channel $j$")
    ax.set_ylabel("lagged correlation")
    ax.set_xlim(-0.5, c_y - 0.5)
    ax.legend(loc="upper right")
    ps.style_axes(ax)

    # --- Panel 6: text summary --------------------------------------------
    extra = []
    if meta.get("benchmark") == "B":
        extra = [f"  AR coeff rho     : {meta.get('rho')}",
                 f"  burn_in          : {meta.get('burn_in')}"]
    if reverse:
        extra = [f"  direction        : {meta.get('direction')}",
                 f"  reverse_roles    : {meta.get('reverse_roles')}"]
    _panel_text(axes[2, 1], meta, [
        f"  delay D          : {delay}",
        f"  transfer coeff a : {a}",
        f"  noise var sigma2 : {sigma2}",
        f"  informative M    : {M} / c_y={c_y}",
        f"  easy_variant     : {meta.get('easy_variant')}",
        *extra,
    ])


def _panels_xor(axes: np.ndarray, Y: np.ndarray, U: np.ndarray,
               meta: Dict[str, Any]) -> None:
    """Fill panels 3, 4, 6 for the delayed-XOR benchmark (C).

    Args:
        axes: The $3 \\times 2$ array of subplot axes.
        Y: Target tensor $(n, T, C_y)$.
        U: Source tensor $(n, T, C_u)$.
        meta: The dataset metadata dict.
    """
    T = Y.shape[1]
    delay = int(meta["delay"])
    M = int(meta["M"])
    c_y = int(meta["c_y"])
    q = float(meta["q"])

    # --- Panel 3: delay-alignment scatter ---------------------------------
    ax = axes[1, 0]
    src = U[:, : T - delay, 0].reshape(-1)
    tgt = Y[:, delay:, 0].reshape(-1)
    emp = _scatter_alignment(ax, src, tgt, color=_C_INFORMATIVE)
    ax.set_title(
        f"XOR delay alignment ch 0:  $U(t-D)$ vs $Y(t)$\n"
        f"empirical corr={emp:.3f}   analytic $1-2q$={1.0 - 2.0 * q:.3f}"
    )
    ax.set_xlabel(f"$U(t-{delay})$, channel 0 ($\\pm 1$ + noise)")
    ax.set_ylabel("$Y(t)$, channel 0")
    ps.style_axes(ax)

    # --- Panel 4: per-channel bit-agreement rate --------------------------
    ax = axes[1, 1]
    y_bit = Y[:, delay:, :] > 0.0
    x_bit = U[:, : T - delay, :c_y] > 0.0
    agree = (y_bit == x_bit).mean(axis=(0, 1))
    colors = [_C_INFORMATIVE if j < M else _C_DISTRACTOR for j in range(c_y)]
    ax.bar(np.arange(c_y), agree, color=colors, width=1.0)
    ax.axhline(1.0 - q, color=_C_ANALYTIC, ls="--", lw=1.0,
               label=f"analytic informative $1-q$={1.0 - q:.3f}")
    ax.axhline(0.5, color=_C_DELAY, ls=":", lw=1.0, label="chance 0.5")
    ax.set_title(
        f"Per-channel bit-agreement $P(\\mathrm{{sign}}\\,Y_j(t)="
        f"\\mathrm{{sign}}\\,X_j(t-D))$\n"
        f"informative=blue ($j<M={M}$), distractor=orange"
    )
    ax.set_xlabel("target channel $j$")
    ax.set_ylabel("bit-agreement rate")
    ax.set_xlim(-0.5, c_y - 0.5)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right")
    ps.style_axes(ax)

    # --- Panel 6: text summary --------------------------------------------
    _panel_text(axes[2, 1], meta, [
        f"  delay D          : {delay}",
        f"  bit-flip prob q  : {q}",
        f"  obs_noise        : {meta.get('obs_noise')}",
        f"  informative M    : {M} / c_y={c_y}",
        f"  easy_variant     : {meta.get('easy_variant')}",
    ])


def _panels_two_lag(axes: np.ndarray, Y: np.ndarray, U: np.ndarray,
                   meta: Dict[str, Any]) -> None:
    """Fill panels 3, 4, 6 for the two-lag Gaussian benchmark (E).

    Args:
        axes: The $3 \\times 2$ array of subplot axes.
        Y: Target tensor $(n, T, C_y)$.
        U: Source tensor $(n, T, C_u)$.
        meta: The dataset metadata dict.
    """
    T = Y.shape[1]
    delay1, delay2 = int(meta["delay1"]), int(meta["delay2"])
    a1, a2 = float(meta["a1"]), float(meta["a2"])
    sigma2 = float(meta["sigma2"])
    M1, M2 = int(meta["M1"]), int(meta["M2"])
    m_total = M1 + M2
    c_y = int(meta["c_y"])

    # --- Panel 3: two-group delay-alignment scatter -----------------------
    ax = axes[1, 0]
    s1 = U[:, : T - delay1, 0].reshape(-1)
    t1 = Y[:, delay1:, 0].reshape(-1)
    e1 = _scatter_alignment(ax, s1, t1, color=_C_INFORMATIVE)
    s2 = U[:, : T - delay2, M1].reshape(-1)
    t2 = Y[:, delay2:, M1].reshape(-1)
    e2 = _scatter_alignment(ax, s2, t2, color=_C_SOURCE)
    ax.set_title(
        f"Two-lag delay alignment\n"
        f"group 1 ch 0 @ $D_1$={delay1} corr={e1:.3f}   "
        f"group 2 ch {M1} @ $D_2$={delay2} corr={e2:.3f}"
    )
    ax.set_xlabel("$U(t-D)$")
    ax.set_ylabel("$Y(t)$")
    ps.style_axes(ax)

    # --- Panel 4: per-channel lagged corr at each group's own delay -------
    ax = axes[1, 1]
    corr1 = _lagged_corr_per_channel(U, Y, delay1)
    corr2 = _lagged_corr_per_channel(U, Y, delay2)
    combined = corr1.copy()
    combined[M1:m_total] = corr2[M1:m_total]
    colors = []
    for j in range(c_y):
        if j < M1:
            colors.append(_C_INFORMATIVE)   # group 1
        elif j < m_total:
            colors.append(_C_SOURCE)        # group 2
        else:
            colors.append(_C_DISTRACTOR)    # distractor
    ax.bar(np.arange(c_y), combined, color=colors, width=1.0)
    ac1 = a1 / math.sqrt(a1 * a1 + sigma2)
    ac2 = a2 / math.sqrt(a2 * a2 + sigma2)
    ax.axhline(ac1, color=_C_INFORMATIVE, ls="--", lw=1.0,
               label=f"group 1 ~{ac1:.3f}")
    ax.axhline(ac2, color=_C_SOURCE, ls="--", lw=1.0,
               label=f"group 2 ~{ac2:.3f}")
    ax.set_title(
        "Per-channel lagged corr at the channel's own delay\n"
        f"group 1 ($j<{M1}$) blue, group 2 ($<{m_total}$) green, "
        "distractor orange"
    )
    ax.set_xlabel("target channel $j$")
    ax.set_ylabel("lagged correlation")
    ax.set_xlim(-0.5, c_y - 0.5)
    ax.legend(loc="upper right")
    ps.style_axes(ax)

    # --- Panel 6: text summary --------------------------------------------
    _panel_text(axes[2, 1], meta, [
        f"  delay D1 / D2    : {delay1} / {delay2}",
        f"  coeff a1 / a2    : {a1} / {a2}",
        f"  noise var sigma2 : {sigma2}",
        f"  M1 / M2          : {M1} / {M2}  (c_y={c_y})",
        f"  te_true_1        : {float(meta.get('te_true_1', 0.0)):.4f} nats",
        f"  te_true_2        : {float(meta.get('te_true_2', 0.0)):.4f} nats",
    ])


def _panel_text(ax: plt.Axes, meta: Dict[str, Any], param_lines: list) -> None:
    """Fill panel 6 with the ground-truth + process-parameter text block.

    Args:
        ax: The target subplot axis.
        meta: The dataset metadata dict.
        param_lines: Benchmark-specific process-parameter lines.
    """
    ax.axis("off")
    sizes = meta.get("split_sizes", {})
    band = sorted(set(meta.get("true_lag_band", [])))
    band_str = f"{band[0]}..{band[-1]}" if band else "(none -- reverse/null)"
    lines = [
        "Ground truth",
        f"  benchmark        : {meta.get('benchmark')}",
        f"  te_true (block)  : {float(meta['te_true']):.4f} nats",
        f"  te_per_step      : {float(meta['te_per_step']):.4f} nats",
        f"  true_lag_band    : {band_str}  (H={meta.get('horizon')})",
        "",
        "Process parameters",
        *param_lines,
        f"  standardized     : {meta.get('standardized')}",
        "",
        "Splits",
        f"  train / val / test : "
        f"{sizes.get('train', '?')} / {sizes.get('val', '?')} / "
        f"{sizes.get('test', '?')}",
        f"  sequence length T  : {meta.get('sequence_length')}",
    ]
    ax.text(
        0.02, 0.98, "\n".join(lines), va="top", ha="left",
        family="monospace", fontsize=9.0, color=ps.COLOR_BLACK,
        transform=ax.transAxes,
    )


def make_preview(
    out_dir: Union[str, Path],
    meta: Dict[str, Any],
    *,
    split: str = "train",
    out_pdf: Optional[Union[str, Path]] = None,
    n_sub: int = 1000,
) -> Path:
    r"""Render a one-page ``preview.{pdf,png}`` for a cached synthetic dataset.

    The panel layout adapts to ``meta["benchmark"]`` (Phase 7): panels 3, 4 and
    6 are dispatched to a Gaussian (A / B / G), XOR (C) or two-lag (E) variant;
    panels 1, 2 and 5 are shared.

    Args:
        out_dir: Directory holding ``{split}.npz`` and ``meta.json``.
        meta: The dataset metadata dict (analytic ground truth + config).
        split: Which cached split to visualise (default ``"train"``).
        out_pdf: Output PDF path. Defaults to ``out_dir/preview.pdf``; a
            sibling ``.png`` is always written alongside it.
        n_sub: Leading samples used for the scatter / correlation panels.

    Returns:
        The path of the written PDF.
    """
    out_dir = Path(out_dir)
    npz_path = out_dir / f"{split}.npz"
    out_pdf = Path(out_pdf) if out_pdf is not None else out_dir / "preview.pdf"

    data = _load_split(npz_path, n_sub)
    Y, U = data["Y"], data["U"]
    benchmark = meta.get("benchmark", "?")
    tag = meta.get("tag", "?")

    fig, axes = plt.subplots(3, 2, figsize=(12, 13.5))
    fig.suptitle(
        f"Synthetic TE dataset preview  --  Benchmark {benchmark}  (tag: {tag})",
        fontsize=ps.FONT_TITLE,
        fontweight="bold",
    )

    _panels_traces(axes, Y, U, meta)
    if benchmark == "C":
        _panels_xor(axes, Y, U, meta)
    elif benchmark == "E":
        _panels_two_lag(axes, Y, U, meta)
    else:  # A / B / G and any unknown benchmark fall back to the Gaussian view.
        _panels_gaussian(axes, Y, U, meta)
    _panel_lag_band(axes[2, 0], meta)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    ps.save_figure(fig, out_pdf.with_suffix(""))
    return out_pdf


# =============================================================================
# Multi-figure dataset gallery -- imshow-rich per-dataset inspection
# =============================================================================
#
# ``make_preview`` renders a single condensed page. The gallery below renders a
# *set* of standalone figures into ``<cache>/figures/`` so every facet of a
# dataset -- its raw inputs, the native model fields, the forecast-target
# window, the source-lag structure, the analytic-TE breakdown and the value
# distributions -- can be inspected at full size. Each figure is written as a
# ``.pdf`` + ``.png`` pair via :func:`plot_style.save_figure`.

_WARMUP_DEFAULT = 30    # model ``warmup_period`` -- steps excluded from the loss
_GALLERY_SAMPLES = 2    # per-sample heatmaps rendered for this many samples
_CORR_SAMPLE_CAP = 256  # samples used for the (slow) lagged-correlation panels
_HEATMAP_CMAP = "RdBu_r"   # diverging map for zero-centred standardised values

# Figure margins for the stacked-panel gallery figures: a lower ``top`` than
# the plot_style default leaves clear room for the bold suptitle above the
# first panel's own title.
_GALLERY_MARGINS = (0.10, 0.95, 0.92, 0.05)


def _channel_spans(meta: Dict[str, Any]) -> Dict[str, tuple]:
    """Return the ``{field: (lo, hi)}`` channel ranges of the native fields.

    Args:
        meta: The dataset metadata dict (its ``channel_map`` is used when
            present).

    Returns:
        A dict mapping each native field name (``fhr_st`` / ``fhr_ph`` /
        ``up_st`` / ``up_ph``) to its ``(lo, hi)`` half-open channel range.
    """
    cmap = meta.get("channel_map")
    if cmap:
        return {k: (int(v[0]), int(v[1])) for k, v in cmap.items()}
    return {
        "fhr_st": (0, 43), "fhr_ph": (43, 87),
        "up_st": (0, 43), "up_ph": (43, 101),
    }


def _symmetric_vlim(arr: np.ndarray, pct: float = 99.0, floor: float = 0.5) -> float:
    """Symmetric colour limit for a zero-centred (standardised) array.

    Args:
        arr: The array to scale.
        pct: Percentile of $|arr|$ used as the limit (robust to outliers).
        floor: Smallest limit returned, so near-constant panels stay readable.

    Returns:
        A positive ``float`` ``v`` for use as ``vmin=-v, vmax=+v``.
    """
    v = float(np.percentile(np.abs(np.asarray(arr, dtype=float)), pct))
    return max(floor, v)


def _heatmap(ax: plt.Axes, arr: np.ndarray, *, vlim: float,
             cmap: str = _HEATMAP_CMAP, vmin: Optional[float] = None):
    """imshow ``arr`` $(T, C)$ as a channel-(y) by time-(x) heatmap.

    Args:
        ax: The target axis.
        arr: A $(T, C)$ array (time-major); shown transposed.
        vlim: Symmetric colour limit (``vmin=-vlim`` unless ``vmin`` is given).
        cmap: Matplotlib colormap name.
        vmin: Optional explicit lower colour bound (for sequential maps).

    Returns:
        The ``AxesImage`` handle (pass to :func:`plot_style.attach_colorbar`).
    """
    T, C = arr.shape
    return ax.imshow(
        np.asarray(arr).T, aspect="auto", origin="lower", cmap=cmap,
        vmin=(-vlim if vmin is None else vmin), vmax=vlim,
        interpolation="nearest",
        extent=(0.0, float(T), -0.5, float(C) - 0.5),
    )


def _mark_informative(ax: plt.Axes, m: int, c: int, *, axis: str = "y") -> None:
    """Draw the informative / distractor channel split line on a heatmap.

    Args:
        ax: The target axis.
        m: Number of informative channels (``0 < m < c``).
        c: Total channel count.
        axis: ``"y"`` for a horizontal split (channel-on-y heatmaps).
    """
    if not 0 < m < c:
        return
    if axis == "y":
        ax.axhline(m - 0.5, color=ps.COLOR_BLACK, lw=1.1, ls="--")
        # Anchor the label *above* the split line (va="bottom"); for the small
        # informative band this keeps it clear of the x-axis ticks below.
        ax.text(
            0.012, (m - 0.5) / c + 0.02, "informative $j<M$",
            color=ps.COLOR_BLACK, fontsize=ps.FONT_TICK, ha="left", va="bottom",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
        )


def _lagged_corr_grid(
    U: np.ndarray, Y: np.ndarray, max_lag: int, n_cap: int = _CORR_SAMPLE_CAP
) -> np.ndarray:
    r"""Pooled Pearson correlation grid $\mathrm{corr}(U_j(t-\ell),\,Y_j(t))$.

    Args:
        U: Source tensor $(n, T, C_u)$.
        Y: Target tensor $(n, T, C_y)$.
        max_lag: Largest lag $\ell$ evaluated (the grid spans $0..\texttt{max\_lag}$).
        n_cap: Cap on the number of samples pooled (keeps the panel cheap).

    Returns:
        A $(C_y, \texttt{max\_lag}+1)$ array of per-channel, per-lag correlations
        (each $Y$ channel paired with the source channel of the same index).
    """
    n = min(int(n_cap), Y.shape[0])
    U, Y = U[:n], Y[:n]
    T, c_y = Y.shape[1], Y.shape[-1]
    grid = np.zeros((c_y, max_lag + 1), dtype=np.float64)
    for lag in range(max_lag + 1):
        if T - lag < 2:
            continue
        src = U[:, : T - lag, :c_y].reshape(-1, c_y).astype(np.float64)
        tgt = Y[:, lag:, :].reshape(-1, c_y).astype(np.float64)
        src = src - src.mean(axis=0, keepdims=True)
        tgt = tgt - tgt.mean(axis=0, keepdims=True)
        denom = np.linalg.norm(src, axis=0) * np.linalg.norm(tgt, axis=0)
        grid[:, lag] = np.where(denom > 0.0, (src * tgt).sum(axis=0) / (denom + 1e-12), 0.0)
    return grid


def _fig_input_heatmaps(
    data: Dict[str, np.ndarray], meta: Dict[str, Any], fig_dir: Path
) -> list:
    """Figure 1 -- raw target $Y$ and source $U$ as channel-by-time heatmaps.

    Args:
        data: Loaded split dict (``Y`` / ``U`` / ``weight``).
        meta: The dataset metadata dict.
        fig_dir: Output directory for the figure pair.

    Returns:
        The list of written file paths.
    """
    Y, U = data["Y"], data["U"]
    n_show = min(_GALLERY_SAMPLES, Y.shape[0])
    delay = _primary_delay(meta)
    M, c_y, c_u = int(meta.get("M", 0)), int(meta["c_y"]), int(meta["c_u"])
    tag = meta.get("tag", "?")

    rows = [(stream, s) for s in range(n_show) for stream in ("Y", "U")]
    fig, axes, caxes = ps.stacked_figure(
        [1.5] * len(rows), colorbar=True, width=11.0, hspace=0.62,
        margins=_GALLERY_MARGINS,
    )
    fig.suptitle(
        f"Dataset inputs -- raw heatmaps  (Benchmark {meta.get('benchmark','?')}, "
        f"tag: {tag})", fontsize=ps.FONT_TITLE, fontweight="bold",
    )
    for ax, cax, (stream, s) in zip(axes, caxes, rows):
        arr = Y[s] if stream == "Y" else U[s]
        c_tot = c_y if stream == "Y" else c_u
        vlim = _symmetric_vlim(arr)
        im = _heatmap(ax, arr, vlim=vlim)
        name = "target $Y=[fhr\\_st,fhr\\_ph]$" if stream == "Y" \
            else "source $U=[up\\_st,up\\_ph]$"
        ax.set_title(f"{name} -- sample {s}   ({c_tot} channels x {arr.shape[0]} steps)")
        ax.set_xlabel("time step $t$")
        ax.set_ylabel("channel $j$")
        ax.axvline(delay, color=ps.COLOR_GRAY, ls=":", lw=1.0)
        _mark_informative(ax, M, c_tot)
        ps.style_axes(ax, grid="none")
        ps.attach_colorbar(fig, im, cax, label="standardised value")
    return ps.save_figure(fig, fig_dir / "01_input_heatmaps")


def _fig_model_fields(
    data: Dict[str, np.ndarray], meta: Dict[str, Any], fig_dir: Path
) -> list:
    """Figure 2 -- the four native model input fields for one sample.

    Shows ``fhr_st`` / ``fhr_ph`` / ``up_st`` / ``up_ph`` as separate heatmaps,
    i.e. exactly the tensors the dataloader hands the model.

    Args:
        data: Loaded split dict.
        meta: The dataset metadata dict.
        fig_dir: Output directory for the figure pair.

    Returns:
        The list of written file paths.
    """
    Y, U = data["Y"][0], data["U"][0]
    spans = _channel_spans(meta)
    fields = [
        ("fhr_st", Y, spans["fhr_st"], "target scattering  $fhr\\_st$"),
        ("fhr_ph", Y, spans["fhr_ph"], "target phase  $fhr\\_ph$"),
        ("up_st", U, spans["up_st"], "source scattering  $up\\_st$"),
        ("up_ph", U, spans["up_ph"], "source phase  $up\\_ph$"),
    ]
    fig, axes, caxes = ps.stacked_figure(
        [1.4] * 4, colorbar=True, width=11.0, hspace=0.6,
        margins=_GALLERY_MARGINS,
    )
    fig.suptitle(
        f"Native model input fields -- sample 0  (tag: {meta.get('tag','?')})",
        fontsize=ps.FONT_TITLE, fontweight="bold",
    )
    delay = _primary_delay(meta)
    M = int(meta.get("M", 0))
    for ax, cax, (name, arr, (lo, hi), title) in zip(axes, caxes, fields):
        sub = arr[:, lo:hi]
        vlim = _symmetric_vlim(sub)
        im = _heatmap(ax, sub, vlim=vlim)
        ax.set_title(f"{title}   (channels {lo}:{hi}  ->  {hi - lo} wide)")
        ax.set_xlabel("time step $t$")
        ax.set_ylabel("field channel")
        ax.axvline(delay, color=ps.COLOR_GRAY, ls=":", lw=1.0)
        # The informative channels live at the front of the *stream*; only the
        # first scattering field of each stream can carry them.
        if name in ("fhr_st", "up_st"):
            _mark_informative(ax, min(M, hi - lo), hi - lo)
        ps.style_axes(ax, grid="none")
        ps.attach_colorbar(fig, im, cax, label="standardised value")
    return ps.save_figure(fig, fig_dir / "02_model_input_fields")


def _fig_io_forecast(
    data: Dict[str, np.ndarray], meta: Dict[str, Any], fig_dir: Path
) -> list:
    r"""Figure 3 -- the forecast contract: history, source window and target block.

    Panel 1 overlays an informative target channel with its delayed source
    drive. Panel 2 is the $(L, C_u)$ source-history window the lag attention
    consumes at one anchor. Panel 3 is the $(H_d, C_y)$ future block the
    decoder must predict at that anchor.

    Args:
        data: Loaded split dict.
        meta: The dataset metadata dict.
        fig_dir: Output directory for the figure pair.

    Returns:
        The list of written file paths.
    """
    Y, U = data["Y"][0], data["U"][0]
    T = Y.shape[0]
    delay = _primary_delay(meta)
    horizon = int(meta.get("horizon", 30))
    max_lag = _MAX_LAG
    a = meta.get("a")
    anchor_range = meta.get("clean_anchor_range", [delay - 1, T - horizon])
    t0 = int((anchor_range[0] + anchor_range[1]) // 2)
    anchors = sorted({anchor_range[0] + 5, t0, max(anchor_range[0] + 5, anchor_range[1] - 5)})

    fig, axes, caxes = ps.stacked_figure(
        [1.25, 1.5, 1.5], width=11.0, colorbar=[False, True, True], hspace=0.62,
        margins=_GALLERY_MARGINS,
    )
    fig.suptitle(
        f"Forecast contract -- inputs, source window and prediction target  "
        f"(tag: {meta.get('tag','?')})",
        fontsize=ps.FONT_TITLE, fontweight="bold",
    )

    # --- Panel 1: informative channel 0 history + delayed source drive ----
    ax = axes[0]
    t_axis = np.arange(T)
    ax.plot(t_axis, Y[:, 0], lw=0.9, color=ps.COLOR_BLUE, label="target $Y_0(t)$")
    if a is not None:
        drive = np.full(T, np.nan, dtype=float)
        drive[delay:] = float(a) * U[: T - delay, 0]
        ax.plot(t_axis, drive, lw=0.9, color=ps.COLOR_GREEN,
                label=f"delayed drive $a\\,U_0(t-{delay})$")
    ax.axvspan(0, _WARMUP_DEFAULT, color=ps.COLOR_GRAY, alpha=0.16,
               label=f"warm-up [0,{_WARMUP_DEFAULT})")
    for k, anc in enumerate(anchors):
        ax.axvspan(anc + 1, anc + horizon, color=ps.COLOR_VERMILLION, alpha=0.16,
                   label="forecast horizon" if k == 0 else None)
        ax.axvline(anc, color=ps.COLOR_VERMILLION, lw=0.8, ls="--")
    ax.set_title("informative channel 0: history, warm-up and forecast windows")
    ax.set_xlabel("time step $t$")
    ax.set_ylabel("standardised value")
    ps.tighten_xaxis(ax, t_axis)
    ax.legend(loc="upper right", ncol=2)
    ps.style_axes(ax)

    # --- Panel 2: source-history window the attention sees at anchor t0 ---
    ax = axes[1]
    lo = max(0, t0 - max_lag)
    src_win = U[lo : t0 + 1, :]                       # (L', C_u)
    vlim = _symmetric_vlim(src_win)
    im = _heatmap(ax, src_win, vlim=vlim)
    ax.set_title(
        f"source-history window at anchor $t={t0}$  "
        f"(lags 0..{t0 - lo}; model attention window 0..{max_lag})"
    )
    ax.set_xlabel(f"steps before anchor  (0 = $t-{t0 - lo}$ ... {src_win.shape[0]-1} = $t$)")
    ax.set_ylabel("source channel")
    ps.style_axes(ax, grid="none")
    ps.attach_colorbar(fig, im, caxes[1], label="standardised value")

    # --- Panel 3: the (H_d, C_y) future block the decoder predicts --------
    ax = axes[2]
    y_plus = Y[t0 + 1 : t0 + 1 + horizon, :]          # (H_d, C_y)
    vlim = _symmetric_vlim(y_plus)
    im = _heatmap(ax, y_plus, vlim=vlim)
    ax.set_title(
        f"forecast target $Y^+$ at anchor $t={t0}$   "
        f"(horizon $H_d$={horizon} x $C_y$={Y.shape[1]} -- the decoder output)"
    )
    ax.set_xlabel(r"forecast step $\tau = 1 \dots H_d$")
    ax.set_ylabel("target channel")
    ps.style_axes(ax, grid="none")
    ps.attach_colorbar(fig, im, caxes[2], label="standardised value")
    return ps.save_figure(fig, fig_dir / "03_forecast_contract")


def _fig_sample_traces(
    data: Dict[str, np.ndarray], meta: Dict[str, Any], fig_dir: Path
) -> list:
    """Figure 4 -- line traces of informative / distractor / source channels.

    Args:
        data: Loaded split dict.
        meta: The dataset metadata dict.
        fig_dir: Output directory for the figure pair.

    Returns:
        The list of written file paths.
    """
    Y, U = data["Y"], data["U"]
    T = Y.shape[1]
    t_axis = np.arange(T)
    delay = _primary_delay(meta)
    M, c_y = int(meta.get("M", 0)), int(meta["c_y"])
    n_tr = min(4, Y.shape[0])
    dist_ch = M if M < c_y else 1   # easy variant has no distractor -> ch 1

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.4), sharex=True)
    fig.suptitle(
        f"Sample traces -- {n_tr} sequences overlaid  (tag: {meta.get('tag','?')})",
        fontsize=ps.FONT_TITLE, fontweight="bold",
    )
    panels = [
        (axes[0], Y, 0, "target $Y$ -- informative channel 0"),
        (axes[1], Y, dist_ch,
         f"target $Y$ -- {'distractor' if M < c_y else 'informative'} channel {dist_ch}"),
        (axes[2], U, 0, "source $U$ -- channel 0"),
    ]
    for ax, arr, ch, title in panels:
        for s in range(n_tr):
            ax.plot(t_axis, arr[s, :, ch], lw=0.8, alpha=0.85,
                    color=ps.PALETTE_EXTENDED[s % len(ps.PALETTE_EXTENDED)])
        ax.axvline(delay, color=ps.COLOR_GRAY, ls="--", lw=0.9,
                   label=f"delay $D$={delay}")
        ax.set_title(title)
        ax.set_ylabel("std. value")
        ps.tighten_xaxis(ax, t_axis)
        ax.legend(loc="upper right")
        ps.style_axes(ax)
    axes[-1].set_xlabel("time step $t$")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return ps.save_figure(fig, fig_dir / "04_sample_traces")


def _fig_lag_structure(
    data: Dict[str, np.ndarray], meta: Dict[str, Any], fig_dir: Path
) -> list:
    r"""Figure 5 -- the source-lag structure the model's attention must recover.

    Panel 1 is the lagged cross-correlation $\mathrm{corr}(U_j(t-\ell), Y_j(t))$
    averaged over informative vs distractor channels. Panel 2 is the full
    $(C_y, L)$ correlation grid. Both shade the true lag band
    $\{D-H,\dots,D-1\}$.

    Args:
        data: Loaded split dict.
        meta: The dataset metadata dict.
        fig_dir: Output directory for the figure pair.

    Returns:
        The list of written file paths.
    """
    Y, U = data["Y"], data["U"]
    M, c_y = int(meta.get("M", 0)), int(meta["c_y"])
    delay = _primary_delay(meta)
    band = sorted(set(meta.get("true_lag_band", [])))
    grid = _lagged_corr_grid(U, Y, _MAX_LAG)          # (C_y, L)
    lag_axis = np.arange(_MAX_LAG + 1)

    fig, axes, caxes = ps.stacked_figure(
        [1.15, 1.6], width=11.0, colorbar=[False, True], hspace=0.55,
        margins=_GALLERY_MARGINS,
    )
    fig.suptitle(
        f"Source-lag structure -- where the transfer lives  (tag: {meta.get('tag','?')})",
        fontsize=ps.FONT_TITLE, fontweight="bold",
    )

    # --- Panel 1: informative- vs distractor-mean lagged correlation ------
    ax = axes[0]
    inf_curve = grid[:M].mean(axis=0) if M > 0 else np.zeros(_MAX_LAG + 1)
    ax.plot(lag_axis, inf_curve, color=ps.COLOR_BLUE, lw=1.3,
            label=f"informative channels ($j<M={M}$)")
    if M < c_y:
        dist_curve = grid[M:].mean(axis=0)
        ax.plot(lag_axis, dist_curve, color=ps.COLOR_ORANGE, lw=1.1,
                label="distractor channels")
    if band:
        ax.axvspan(band[0], band[-1], color=ps.COLOR_VERMILLION, alpha=0.16,
                   label=f"true lag band {band[0]}..{band[-1]}")
    ax.axvline(delay, color=ps.COLOR_GRAY, ls="--", lw=0.9, label=f"delay $D$={delay}")
    ax.axhline(0.0, color=ps.COLOR_BLACK, lw=0.5)
    ax.set_title(r"lagged cross-correlation $\mathrm{corr}(U_j(t-\ell),\,Y_j(t))$")
    ax.set_xlabel(r"source lag $\ell$")
    ax.set_ylabel("correlation")
    ax.set_xlim(0, _MAX_LAG)
    ax.legend(loc="upper left", ncol=2)
    ps.style_axes(ax)

    # --- Panel 2: full (channel, lag) correlation grid --------------------
    ax = axes[1]
    vlim = max(0.05, float(np.percentile(np.abs(grid), 99.5)))
    im = ax.imshow(
        grid, aspect="auto", origin="lower", cmap=_HEATMAP_CMAP,
        vmin=-vlim, vmax=vlim, interpolation="nearest",
        extent=(-0.5, _MAX_LAG + 0.5, -0.5, c_y - 0.5),
    )
    if band:
        for edge in (band[0], band[-1]):
            ax.axvline(edge, color=ps.COLOR_BLACK, lw=0.8, ls="--")
    ax.axvline(delay, color=ps.COLOR_GRAY, ls=":", lw=1.0)
    _mark_informative(ax, M, c_y)
    ax.set_title(r"per-channel lagged-correlation grid (bright square = true transfer)")
    ax.set_xlabel(r"source lag $\ell$")
    ax.set_ylabel("target channel $j$")
    ps.style_axes(ax, grid="none")
    ps.attach_colorbar(fig, im, caxes[1], label="correlation")
    return ps.save_figure(fig, fig_dir / "05_lag_structure")


def _fig_te_structure(
    _data: Dict[str, np.ndarray], meta: Dict[str, Any], fig_dir: Path
) -> list:
    r"""Figure 6 -- the analytic transfer-entropy budget (Gaussian benchmarks).

    Panel 1 is the per-channel block-TE contribution; panel 2 is the
    $\mathrm{TE}^{(H)}$-vs-$a$ curve with the dataset's operating point marked;
    panel 3 is a text summary.

    Args:
        data: Loaded split dict (unused; kept for a uniform figure signature).
        meta: The dataset metadata dict.
        fig_dir: Output directory for the figure pair.

    Returns:
        The list of written file paths.
    """
    a = float(meta["a"])
    sigma2 = float(meta["sigma2"])
    M, c_y = int(meta["M"]), int(meta["c_y"])
    H = int(meta.get("horizon", 30))
    te_true = float(meta["te_true"])
    te_per_channel = te_block_gaussian(a, sigma2, H, 1) if a > 0 else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3))
    fig.suptitle(
        f"Analytic transfer-entropy budget  (Benchmark {meta.get('benchmark','?')}, "
        f"tag: {meta.get('tag','?')})",
        fontsize=ps.FONT_TITLE, fontweight="bold",
    )

    # --- Panel 1: per-channel block-TE contribution -----------------------
    ax = axes[0]
    contrib = np.array([te_per_channel if j < M else 0.0 for j in range(c_y)])
    colors = [ps.COLOR_BLUE if j < M else ps.COLOR_ORANGE for j in range(c_y)]
    ax.bar(np.arange(c_y), contrib, color=colors, width=1.0)
    ax.set_title(f"per-channel block TE   (sum = te_true = {te_true:.3f} nats)")
    ax.set_xlabel("target channel $j$")
    ax.set_ylabel("block TE contribution (nats)")
    ax.set_xlim(-0.5, c_y - 0.5)
    ps.style_axes(ax)

    # --- Panel 2: TE-vs-a curve with the operating point ------------------
    ax = axes[1]
    a_curve = np.linspace(0.0, max(0.6, 2.0 * a), 200)
    te_curve = np.array([te_block_gaussian(av, sigma2, H, M) for av in a_curve])
    ax.plot(a_curve, te_curve, color=ps.COLOR_BLUE, lw=1.4)
    ax.scatter([a], [te_true], s=60, color=ps.COLOR_VERMILLION, zorder=3,
               edgecolors=ps.COLOR_BLACK, linewidths=0.5,
               label=f"operating point\n$a$={a:g}, TE={te_true:.3f}")
    ax.axvline(a, color=ps.COLOR_GRAY, ls=":", lw=1.0)
    ax.set_title(fr"$\mathrm{{TE}}^{{(H)}} = \frac{{H}}{{2}} M \ln(1+a^2/\sigma^2)$  (M={M})")
    ax.set_xlabel("transfer coefficient $a$")
    ax.set_ylabel("block TE (nats)")
    ax.legend(loc="upper left")
    ps.style_axes(ax)

    # --- Panel 3: text summary --------------------------------------------
    ax = axes[2]
    ax.axis("off")
    snr = a * a / sigma2
    band = sorted(set(meta.get("true_lag_band", [])))
    band_str = f"{band[0]}..{band[-1]}" if band else "(none)"
    lines = [
        "Transfer-entropy ground truth",
        f"  block TE          : {te_true:.4f} nats",
        f"  per-step TE       : {float(meta.get('te_per_step', te_true / H)):.4f} nats",
        f"  per-channel TE    : {te_per_channel:.4f} nats",
        f"  horizon H         : {H}",
        "",
        "Signal / noise",
        f"  transfer coeff a  : {a:g}",
        f"  noise var sigma2  : {sigma2:g}",
        f"  SNR a^2/sigma^2   : {snr:.4f}",
        f"  corr a/sqrt(a^2+s): {a / math.sqrt(a * a + sigma2):.4f}" if a > 0 else
        "  corr a/sqrt(a^2+s): 0.0000",
        "",
        "Channels / lags",
        f"  informative M     : {M} / c_y={c_y}",
        f"  true lag band     : {band_str}",
        f"  delay D           : {meta.get('delay')}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=9.0, color=ps.COLOR_BLACK,
            transform=ax.transAxes)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return ps.save_figure(fig, fig_dir / "06_te_structure")


def _fig_distributions(
    data: Dict[str, np.ndarray], meta: Dict[str, Any], fig_dir: Path
) -> list:
    """Figure 7 -- value distributions and the per-channel standardisation check.

    Args:
        data: Loaded split dict.
        meta: The dataset metadata dict.
        fig_dir: Output directory for the figure pair.

    Returns:
        The list of written file paths.
    """
    Y, U = data["Y"], data["U"]
    M, c_y = int(meta.get("M", 0)), int(meta["c_y"])

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2))
    fig.suptitle(
        f"Value distributions and standardisation check  (tag: {meta.get('tag','?')})",
        fontsize=ps.FONT_TITLE, fontweight="bold",
    )

    # --- Panel 1: value histograms ----------------------------------------
    ax = axes[0]
    bins = np.linspace(-5.0, 5.0, 80)
    ax.hist(Y[:, :, :M].reshape(-1), bins=bins, density=True, histtype="step",
            color=ps.COLOR_BLUE, lw=1.3, label="target informative")
    if M < c_y:
        ax.hist(Y[:, :, M:].reshape(-1), bins=bins, density=True, histtype="step",
                color=ps.COLOR_ORANGE, lw=1.1, label="target distractor")
    ax.hist(U.reshape(-1), bins=bins, density=True, histtype="step",
            color=ps.COLOR_GREEN, lw=1.1, label="source")
    ax.set_title("standardised value histogram")
    ax.set_xlabel("standardised value")
    ax.set_ylabel("density")
    ax.legend(loc="upper right")
    ps.style_axes(ax)

    # --- Panels 2-3: per-channel std (standardisation should give ~1) -----
    for ax, arr, name, m_mark, n_ch in (
        (axes[1], Y, "target $Y$", M, c_y),
        (axes[2], U, "source $U$", M, int(meta["c_u"])),
    ):
        std = arr.reshape(-1, arr.shape[-1]).std(axis=0)
        colors = [ps.COLOR_BLUE if j < m_mark else ps.COLOR_ORANGE
                  for j in range(n_ch)]
        ax.bar(np.arange(n_ch), std, color=colors, width=1.0)
        ax.axhline(1.0, color=ps.COLOR_BLACK, ls="--", lw=1.0,
                   label="unit std (target)")
        ax.set_title(f"{name} -- per-channel std")
        ax.set_xlabel("channel $j$")
        ax.set_ylabel("standard deviation")
        ax.set_xlim(-0.5, n_ch - 0.5)
        ax.set_ylim(0.0, max(1.3, float(std.max()) * 1.1))
        ax.legend(loc="lower right")
        ps.style_axes(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return ps.save_figure(fig, fig_dir / "07_distributions")


def make_dataset_gallery(
    out_dir: Union[str, Path],
    meta: Dict[str, Any],
    *,
    split: str = "train",
    n_sub: int = 1000,
) -> list:
    r"""Render the full multi-figure inspection gallery for a cached dataset.

    Writes a set of standalone ``.pdf`` + ``.png`` figures into
    ``<out_dir>/figures/``:

        * ``01_input_heatmaps``   -- raw $Y$ / $U$ channel-by-time heatmaps.
        * ``02_model_input_fields`` -- the four native fields (``fhr_st`` ...).
        * ``03_forecast_contract`` -- history, source window and $Y^+$ target.
        * ``04_sample_traces``    -- informative / distractor / source traces.
        * ``05_lag_structure``    -- lagged-correlation curve + channel-lag grid.
        * ``06_te_structure``     -- analytic-TE budget (Gaussian benchmarks A/B/G).
        * ``07_distributions``    -- value histograms + per-channel std check.

    Args:
        out_dir: Directory holding ``{split}.npz`` and ``meta.json``.
        meta: The dataset metadata dict (analytic ground truth + config).
        split: Which cached split to visualise (default ``"train"``).
        n_sub: Leading samples loaded for the panels.

    Returns:
        A flat list of every written file path (``.pdf`` and ``.png``).
    """
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = _load_split(out_dir / f"{split}.npz", n_sub)
    benchmark = meta.get("benchmark", "?")

    written: list = []
    written += _fig_input_heatmaps(data, meta, fig_dir)
    written += _fig_model_fields(data, meta, fig_dir)
    written += _fig_io_forecast(data, meta, fig_dir)
    written += _fig_sample_traces(data, meta, fig_dir)
    written += _fig_lag_structure(data, meta, fig_dir)
    # The analytic-TE budget panel needs the Gaussian (a, sigma2) parameters
    # and a causal TE budget -- skipped for the reverse-roles benchmark G
    # (anti-causal, te_true = 0 by construction).
    if benchmark in ("A", "B") and "a" in meta and "sigma2" in meta:
        written += _fig_te_structure(data, meta, fig_dir)
    written += _fig_distributions(data, meta, fig_dir)
    return written


if __name__ == "__main__":
    # Self-check: build tiny caches in temp dirs and render their previews,
    # exercising the Gaussian (A) and XOR (C) panel paths.
    import json
    import tempfile

    from model.vae_teb_prediction.model.model_experiment.synthetic.generators import (
        gen_delayed_gaussian,
        gen_delayed_xor,
        gen_two_lag_gaussian,
    )

    def _dump_and_preview(Y, U, meta, tag):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            np.savez(
                d / "train.npz",
                fhr_st=Y[..., :43].numpy(),
                fhr_ph=Y[..., 43:87].numpy(),
                up_st=U[..., :43].numpy(),
                up_ph=U[..., 43:101].numpy(),
                weight=np.ones((Y.shape[0], Y.shape[1]), dtype=np.float32),
            )
            m = dict(meta)
            m["tag"] = tag
            m["split_sizes"] = {"train": Y.shape[0], "val": 0, "test": 0}
            with open(d / "meta.json", "w", encoding="utf-8") as fh:
                json.dump(m, fh, indent=2)
            pdf = make_preview(d, m, n_sub=Y.shape[0])
            assert pdf.is_file() and pdf.stat().st_size > 0, pdf
            png = pdf.with_suffix(".png")
            assert png.is_file() and png.stat().st_size > 0, png
            gallery = make_dataset_gallery(d, m, n_sub=Y.shape[0])
            assert gallery and all(p.is_file() and p.stat().st_size > 0
                                   for p in gallery), gallery
            print(f"[preview {tag}] wrote {pdf.name} + {png.name}; "
                  f"gallery: {len(gallery)} files "
                  f"({len({p.stem for p in gallery})} figures x pdf+png)")

    _Ya, _Ua, _ma = gen_delayed_gaussian(
        n=64, T=300, delay=60, a=1.0, sigma2=1.0, M=4, seed=0
    )
    _dump_and_preview(_Ya, _Ua, _ma, "smoke_A")

    _Yc, _Uc, _mc = gen_delayed_xor(n=64, T=300, delay=60, q=0.10, M=4, seed=0)
    _dump_and_preview(_Yc, _Uc, _mc, "smoke_C")

    _Ye, _Ue, _me = gen_two_lag_gaussian(
        n=64, T=300, delay1=50, delay2=80, a1=0.4, a2=0.25,
        sigma2=1.0, M1=4, M2=4, seed=0,
    )
    _dump_and_preview(_Ye, _Ue, _me, "smoke_E")

    _Yg, _Ug, _mg = gen_delayed_gaussian(
        n=64, T=300, delay=60, a=1.0, sigma2=1.0, M=4, reverse_roles=True, seed=0
    )
    _dump_and_preview(_Yg, _Ug, _mg, "smoke_G")

    print("All visualize checks passed.")
