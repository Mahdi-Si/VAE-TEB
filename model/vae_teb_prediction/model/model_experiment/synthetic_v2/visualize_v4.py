r"""S7-T04: ground-truth-grading figures for ``synthetic_v4`` (raw-domain).

Five publication figures render the four core gates of the raw pipeline from the machine artifacts
:func:`eval_v4.run_eval_v4` writes (``metrics.json`` + ``per_sample_eval.npz``):

#. :func:`plot_kbar_vs_te_v4` -- the headline $\bar K$-versus-$\mathrm{TE}_{\mathrm{inj}}$ scatter
   (per-sample cloud + per-cell means + the pooled calibration line $\bar K = \alpha + \gamma\,
   \mathrm{TE}_{\mathrm{inj}}$).
#. :func:`plot_calibration_by_lag_v4` -- the calibration slope $\gamma$ broken out by planted lag
   $D$ (``calibration.by_lag``).
#. :func:`plot_lag_recovery_v4` -- recovered $\operatorname{argmax}_\ell \bar\alpha_{t,\ell}$ vs the
   planted $D$ and the per-cell attention lag mass (``lag_recovery``).
#. :func:`plot_pred_control_v4` -- the prediction-space source control ordering
   $\mathcal L_{\mathrm{feat}} < \mathcal L_{\mathrm{base}} < \mathcal L_{\mathrm{feat}}^{\pi(U)}$
   (``prediction_controls.overall``).
#. :func:`plot_kbar_null_bar_v4` -- per-null-cell $\bar K$ against the null-gate ceiling
   (``null_cell_gate`` + the ``calibration.per_cell`` rows at $\mathrm{TE}_{\mathrm{inj}}=0$).

Unlike the scattering-domain :mod:`visualize_v2`, the raw pipeline has a **single ground-truth
axis** ($\mathrm{TE}_{\mathrm{inj}}$; no ``te_scat``) and un-suffixed calibration keys
(``calibration.gamma``, ``by_lag[D].gamma``), so these are dedicated v4 functions rather than reuse
of the v2 dual-axis figures. The publication house style, the multi-format save, the ``no data``
placeholder, and the per-cell bar / OLS / group-stats helpers are reused from :mod:`plot_style_v2`
and :mod:`visualize_v2` unchanged. Every function degrades a missing/partial artifact to a
placeholder panel rather than raising, so a report never loses a figure to a missing key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    plot_style_v2 as ps,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.visualize_v2 import (  # noqa: E402
    _cell_bar,
    _no_data,
    _save_fig,
)

ps.apply_style()

#: Semantic colours (kept consistent with the v2 gallery: blue = K-bar/inj, gold = ground-truth TE).
_KBAR_COLOR = ps.COLOR_BLUE
_INJ_COLOR = ps.COLOR_ORANGE
_BASE_COLOR = ps.COLOR_GRAY
_SHUFFLE_COLOR = ps.COLOR_VERMILLION
_REF_COLOR = ps.COLOR_GREEN
_DPI = ps.SAVE_DPI

#: The output formats every figure writes (PDF vector + PNG raster).
_FORMATS = ("pdf", "png")


def _cal(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    r"""The ``calibration`` sub-dict of ``metrics.json`` (empty when absent)."""
    return ((metrics or {}).get("calibration") or {})


def _per_cell_rows(metrics: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    r"""The ``calibration.per_cell`` rows (``{cell_id, te_inj, kbar, delay, n}``)."""
    rows = _cal(metrics).get("per_cell") or []
    return [r for r in rows if isinstance(r, dict)]


def _f(x: Any) -> float:
    r"""Coerce to ``float`` (``nan`` on failure), so JSON ``None`` never raises."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ===========================================================================
# 1. K-bar vs TE_inj scatter (headline calibration).
# ===========================================================================
def plot_kbar_vs_te_v4(
    per_sample: Optional[Dict[str, Any]],
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    formats: tuple = _FORMATS,
    dpi: int = _DPI,
) -> List[Path]:
    r"""The per-sample $\bar K$-vs-$\mathrm{TE}_{\mathrm{inj}}$ scatter with the calibration line.

    Every evaluated sample is one point: its time-averaged latent KL $\bar K$ (y) against its cell's
    injected transfer entropy $\mathrm{TE}_{\mathrm{inj}}$ (x). Because TE is constant within a cell
    the x-axis is a few discrete levels; the panel overlays the jittered per-sample cloud, the
    per-cell means (from ``calibration.per_cell``), and the pooled per-sample calibration line
    $\bar K = \alpha_{\mathrm{sample}} + \gamma_{\mathrm{sample}}\,\mathrm{TE}_{\mathrm{inj}}$.

    Args:
        per_sample: The ``per_sample_eval.npz`` arrays (``kbar`` / ``te_inj`` / ``cell_id``); a
            ``None`` / empty dict renders a placeholder.
        metrics: The ``metrics.json`` dict (for the pooled calibration fit + labels).
        out_path: Output path stem or full path.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    cal = _cal(metrics)
    ps_arr = per_sample or {}
    kbar = np.asarray(ps_arr.get("kbar", []), dtype=float).reshape(-1)
    te = np.asarray(ps_arr.get("te_inj", []), dtype=float).reshape(-1)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    if kbar.size == 0 or te.size != kbar.size:
        _no_data(ax, "no per_sample_eval.npz (run --stage eval)")
        ps.style_axes(ax)
        return _save_fig(fig, out_path, formats, dpi)

    # Jittered per-sample cloud (levels are discrete, so jitter x to separate the columns).
    rng = np.random.default_rng(0)
    levels = np.unique(te[np.isfinite(te)])
    span = float(np.ptp(levels)) if levels.size > 1 else 1.0
    jitter = (rng.random(te.size) - 0.5) * 0.015 * span
    ax.scatter(te + jitter, kbar, s=8, alpha=0.35, color=_KBAR_COLOR, edgecolors="none",
               label=r"per-sample $\bar K$")

    # Per-cell means (larger markers) from the calibration per-cell rows.
    rows = _per_cell_rows(metrics)
    if rows:
        cx = np.array([_f(r.get("te_inj")) for r in rows])
        cy = np.array([_f(r.get("kbar")) for r in rows])
        ax.scatter(cx, cy, s=55, color=_INJ_COLOR, edgecolors=ps.COLOR_BLACK, linewidths=0.6,
                   zorder=4, label="per-cell mean")

    # Pooled per-sample calibration line.
    gamma = _f(cal.get("gamma_sample", cal.get("gamma")))
    alpha = _f(cal.get("alpha_sample", cal.get("alpha")))
    if np.isfinite(gamma) and np.isfinite(alpha) and levels.size:
        xs = np.linspace(float(levels.min()), float(levels.max()), 50)
        ax.plot(xs, alpha + gamma * xs, color=ps.COLOR_BLACK, lw=1.4,
                label=rf"$\bar K = {alpha:.3g} + {gamma:.3g}\,\mathrm{{TE}}$")

    r2 = _f(cal.get("r2_sample", cal.get("r2")))
    rho = _f(cal.get("spearman"))
    ax.set_xlabel(r"$\mathrm{TE}_{\mathrm{inj}}$ (block-nats)")
    ax.set_ylabel(r"$\bar K$ (nats/step)")
    ax.set_title(rf"raw $\bar K$ vs $\mathrm{{TE}}_{{\mathrm{{inj}}}}$  "
                 rf"($R^2={r2:.3g}$, $\rho={rho:.3g}$, $n={int(kbar.size)}$)")
    ax.legend(loc="upper left", frameon=False, fontsize=ps.FONT_LEGEND)
    ps.style_axes(ax)
    fig.tight_layout()
    return _save_fig(fig, out_path, formats, dpi)


# ===========================================================================
# 2. Calibration slope by lag.
# ===========================================================================
def plot_calibration_by_lag_v4(
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    formats: tuple = _FORMATS,
    dpi: int = _DPI,
) -> List[Path]:
    r"""The calibration slope $\gamma$ per planted lag $D$ (``calibration.by_lag``).

    One bar per lag $D$, its slope $\gamma$ annotated with the per-lag $R^2$ and sample count. The
    concentrated grid plants a single $D$, so this is typically one bar; the panel generalises to a
    lag ladder unchanged.

    Args:
        metrics: The ``metrics.json`` dict.
        out_path: Output path stem or full path.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    by_lag = _cal(metrics).get("by_lag") or {}
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    items = sorted(((int(float(d)), v) for d, v in by_lag.items()), key=lambda kv: kv[0])
    gammas = [_f((v or {}).get("gamma")) for _, v in items]
    if not items or not np.any(np.isfinite(gammas)):
        _no_data(ax, "no calibration.by_lag (run --stage eval)")
        ps.style_axes(ax)
        return _save_fig(fig, out_path, formats, dpi)

    x = np.arange(len(items))
    ax.bar(x, gammas, color=_KBAR_COLOR, width=0.6)
    ax.axhline(0.0, color=_BASE_COLOR, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}" for d, _ in items])
    for xi, (_, v) in zip(x, items):
        r2 = _f((v or {}).get("r2"))
        n = (v or {}).get("n")
        ax.annotate(rf"$R^2$={r2:.2g}" + (f"\nn={int(n)}" if n is not None else ""),
                    (xi, gammas[int(xi)]), ha="center",
                    va="bottom" if gammas[int(xi)] >= 0 else "top", fontsize=6.5)
    ax.set_xlabel("planted lag $D$ (decimated steps)")
    ax.set_ylabel(r"calibration slope $\gamma$ (nats/step per block-nat)")
    ax.set_title(r"$\bar K$-vs-$\mathrm{TE}_{\mathrm{inj}}$ slope by lag")
    ps.style_axes(ax)
    fig.tight_layout()
    return _save_fig(fig, out_path, formats, dpi)


# ===========================================================================
# 3. Lag recovery vs the planted D.
# ===========================================================================
def plot_lag_recovery_v4(
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    formats: tuple = _FORMATS,
    dpi: int = _DPI,
) -> List[Path]:
    r"""Recovered attention lag vs the planted $D$ + the per-cell lag mass (``lag_recovery``).

    Left: the recovered $\operatorname{argmax}_\ell \bar\alpha_{t,\ell}$ (``peak_lag``) against the
    planted $D$ for every signal cell, with the $y=x$ identity. Right: the per-cell attention lag
    mass with the pass threshold; null cells (no planted source) are drawn hollow.

    Args:
        metrics: The ``metrics.json`` dict.
        out_path: Output path stem or full path.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    lag = (metrics or {}).get("lag_recovery") or {}
    per_cell = lag.get("per_cell") or {}
    rows = [v for v in per_cell.values() if isinstance(v, dict)]
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    if not rows:
        for ax in (ax_l, ax_r):
            _no_data(ax, "no lag_recovery (run --stage eval)")
            ps.style_axes(ax)
        return _save_fig(fig, out_path, formats, dpi)

    # Left: recovered peak lag vs planted D (signal cells only).
    sig = [r for r in rows if not bool(r.get("is_null"))]
    planted = np.array([_f(r.get("D")) for r in sig])
    peak = np.array([_f(r.get("peak_lag")) for r in sig])
    ok = np.isfinite(planted) & np.isfinite(peak)
    if np.any(ok):
        ax_l.scatter(planted[ok], peak[ok], s=55, color=_KBAR_COLOR,
                     edgecolors=ps.COLOR_BLACK, linewidths=0.6, zorder=4)
        lo = float(np.nanmin([planted[ok].min(), peak[ok].min()]))
        hi = float(np.nanmax([planted[ok].max(), peak[ok].max()]))
        pad = 1.0
        ax_l.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=_REF_COLOR, lw=1.0, ls="--",
                  label=r"$y=x$ (perfect recovery)")
        ax_l.legend(loc="upper left", frameon=False, fontsize=ps.FONT_LEGEND)
    else:
        _no_data(ax_l, "no signal-cell lag peaks")
    ax_l.set_xlabel("planted lag $D$")
    ax_l.set_ylabel(r"recovered lag $\hat{D}$ (argmax over $\ell$)")
    ax_l.set_title("lag recovery")
    ps.style_axes(ax_l)

    # Right: per-cell lag mass with the pass threshold.
    masses = np.array([_f(r.get("lag_mass")) for r in rows])
    is_null = np.array([bool(r.get("is_null")) for r in rows])
    x = np.arange(len(rows))
    colours = [_BASE_COLOR if n else _KBAR_COLOR for n in is_null]
    ax_r.bar(x, np.nan_to_num(masses, nan=0.0), color=colours, width=0.7)
    thr = _f(lag.get("lag_mass_threshold"))
    if np.isfinite(thr):
        ax_r.axhline(thr, color=_REF_COLOR, lw=1.0, ls="--", label=f"threshold={thr:.2g}")
        ax_r.legend(loc="upper right", frameon=False, fontsize=ps.FONT_LEGEND)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([str(k) for k in per_cell.keys()], fontsize=6.0)
    ax_r.set_xlabel("cell id (grey = null)")
    ax_r.set_ylabel(r"attention lag mass in $L^{*}$")
    mean_mass = _f(lag.get("mean_lag_mass"))
    ax_r.set_title(rf"per-cell lag mass (signal mean = {mean_mass:.3g})")
    ps.style_axes(ax_r)
    fig.tight_layout()
    return _save_fig(fig, out_path, formats, dpi)


# ===========================================================================
# 4. Prediction-space source control ordering.
# ===========================================================================
def plot_pred_control_v4(
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    control: str = "shuffled",
    formats: tuple = _FORMATS,
    dpi: int = _DPI,
) -> List[Path]:
    r"""The raw prediction-space source control $\mathcal L_{\mathrm{feat}} < \mathcal L_{\mathrm{base}}
    < \mathcal L_{\mathrm{feat}}^{\pi(U)}$ (``prediction_controls.overall``).

    Three bars -- the source-using forecast loss $\mathcal L_{\mathrm{feat}}$, the source-free
    baseline $\mathcal L_{\mathrm{base}}$, and the permuted-source loss
    $\mathcal L_{\mathrm{feat}}^{\pi(U)}$ -- annotated with the pass verdict and the shuffle penalty.
    The gate holds when the middle bar sits between the other two.

    Args:
        metrics: The ``metrics.json`` dict.
        out_path: Output path stem or full path.
        control: The control column name (default ``"shuffled"``).
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    overall = ((metrics or {}).get("prediction_controls") or {}).get("overall") or {}
    feat = _f(overall.get("feat_loss"))
    base = _f(overall.get("base_loss"))
    shuf = _f(overall.get(f"feat_loss_{control}"))
    labels = [r"$L_{\mathrm{feat}}$", r"$L_{\mathrm{base}}$",
              r"$L_{\mathrm{feat}}^{\pi(U)}$"]
    vals = np.array([feat, base, shuf])
    colours = [_KBAR_COLOR, _BASE_COLOR, _SHUFFLE_COLOR]

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    if not np.any(np.isfinite(vals)):
        _no_data(ax, "no prediction_controls (run --stage eval)")
        ps.style_axes(ax)
        return _save_fig(fig, out_path, formats, dpi)

    x = np.arange(3)
    ax.bar(x, np.nan_to_num(vals, nan=0.0), color=colours, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("raw forecast MSE (normalised)")
    ordering = overall.get(f"ordering_pass_{control}", overall.get("ordering_pass"))
    penalty = _f(overall.get(f"shuffle_penalty_{control}"))
    verdict = "PASS" if ordering else "FAIL" if ordering is not None else "n/a"
    ax.set_title(f"prediction-space source control: {verdict}"
                 + (rf"  (shuffle penalty = {penalty:.3g})" if np.isfinite(penalty) else ""))
    ps.style_axes(ax)
    fig.tight_layout()
    return _save_fig(fig, out_path, formats, dpi)


# ===========================================================================
# 5. Per-null-cell K-bar bar vs the gate ceiling.
# ===========================================================================
def plot_kbar_null_bar_v4(
    metrics: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    formats: tuple = _FORMATS,
    dpi: int = _DPI,
) -> List[Path]:
    r"""Per-null-cell $\bar K$ against the null-gate ceiling (``null_cell_gate``).

    One bar per null cell ($\mathrm{TE}_{\mathrm{inj}}=0$, from ``calibration.per_cell``) with the
    decidable ceiling drawn as a reference line; the gate passes when the mean sits below it.

    Args:
        metrics: The ``metrics.json`` dict.
        out_path: Output path stem or full path.
        formats: Output formats to write.
        dpi: Raster DPI for PNG output.

    Returns:
        The list of written file paths.
    """
    gate = (metrics or {}).get("null_cell_gate") or {}
    rows = [r for r in _per_cell_rows(metrics) if _f(r.get("te_inj")) == 0.0]
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    if not rows:
        _no_data(ax, "no null (te_inj==0) cells in the grid")
        ps.style_axes(ax)
        return _save_fig(fig, out_path, formats, dpi)

    values = np.array([_f(r.get("kbar")) for r in rows])
    cell_ids = np.array([_f(r.get("cell_id")) for r in rows])
    ceiling = _f(gate.get("ceiling"))
    passed = gate.get("pass")
    verdict = "PASS" if passed else "FAIL" if passed is not None else "n/a"
    _cell_bar(ax, values, cell_ids, color=_KBAR_COLOR,
              ylabel=r"$\bar K$ at null cells (nats/step)",
              title=f"null-cell gate: {verdict}",
              ref=ceiling if np.isfinite(ceiling) else None,
              ref_label=rf"ceiling = {ceiling:.3g}" if np.isfinite(ceiling) else None)
    ps.style_axes(ax)
    fig.tight_layout()
    return _save_fig(fig, out_path, formats, dpi)


#: Every figure this module renders, as ``(stem, callable(per_sample, metrics) -> paths)`` -- the
#: report driver (S7-T05) iterates this so a new figure is registered in one place.
def figure_specs() -> List[tuple]:
    r"""Return ``[(stem, render(per_sample, metrics, out_path))]`` for the report driver.

    Each ``render`` takes ``(per_sample, metrics, out_path)`` so the driver need not know which
    figures consume the per-sample arrays; the ones that ignore ``per_sample`` simply drop it.
    """
    return [
        ("kbar_vs_te", lambda psd, m, p: plot_kbar_vs_te_v4(psd, m, p)),
        ("calibration_by_lag", lambda psd, m, p: plot_calibration_by_lag_v4(m, p)),
        ("lag_recovery", lambda psd, m, p: plot_lag_recovery_v4(m, p)),
        ("pred_control", lambda psd, m, p: plot_pred_control_v4(m, p)),
        ("kbar_null_bar", lambda psd, m, p: plot_kbar_null_bar_v4(m, p)),
    ]
