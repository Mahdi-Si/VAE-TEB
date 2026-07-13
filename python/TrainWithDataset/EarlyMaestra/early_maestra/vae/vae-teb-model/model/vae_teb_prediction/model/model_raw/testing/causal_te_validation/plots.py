"""Plotters for the causal-TE validation suite.

Per-test figures plus a single 2x2 manuscript headline figure
(:func:`plot_causal_te_summary`).

All plots reuse the project-wide style helpers from
``model.vae_teb_prediction.model.model_raw.testing.visualizers`` so they stay visually
consistent with the rest of the testing pipeline (Times New Roman,
600 dpi PNG, neutral grid, Okabe-Ito-derived class palette).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover - matplotlib is a hard dep elsewhere
    plt = None  # type: ignore[assignment]
    Figure = Any  # type: ignore[assignment, misc]

from model.vae_teb_prediction.model.model_raw.testing.visualizers import (
    CLASS_COLORS,
    CLASS_NAMES,
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
    _style_axes,
    class_color_for,
    class_label_for,
    unique_labels_in,
)


# ---------------------------------------------------------------------------
# Test 1 — UP ablation forest plot of (median delta + CI)
# ---------------------------------------------------------------------------


_CONDITION_COLORS: Dict[str, str] = {
    "zero":          COLOR_BLUE,
    "batch_permute": COLOR_ORANGE,
    "time_shuffle":  COLOR_GREEN,
}


def plot_up_ablation_forest(
    wilcoxon_csv: Path,
    out_path: Path,
) -> Optional[Path]:
    """Forest plot: median delta + 95% CI per (condition x metric).

    Args:
        wilcoxon_csv: ``up_ablation_stats/wilcoxon_results.csv``.
        out_path: Output PDF path.

    Returns:
        ``out_path`` on success, ``None`` when the input is missing /
        empty / matplotlib is unavailable.
    """
    if plt is None:
        return None
    csv = Path(wilcoxon_csv)
    if not csv.is_file():
        return None
    df = pd.read_csv(csv)
    if df.empty:
        return None
    metrics = list(df["metric"].unique())
    conditions = [c for c in _CONDITION_COLORS if c in set(df["condition"])]
    if not metrics or not conditions:
        return None

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(2.6 * len(metrics) + 0.4, 3.2),
        sharey=True,
    )
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sub = df[df["metric"] == metric]
        y = np.arange(len(conditions))
        for j, cond in enumerate(conditions):
            row = sub[sub["condition"] == cond]
            if row.empty:
                continue
            est = float(row["median_delta"].iloc[0])
            lo = float(row["ci_low"].iloc[0])
            hi = float(row["ci_high"].iloc[0])
            color = _CONDITION_COLORS[cond]
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(
                    [est], [y[j]],
                    xerr=[[est - lo], [hi - est]],
                    fmt="o", color=color, ecolor=color, capsize=2,
                    markersize=4, lw=1.0,
                )
            else:
                ax.plot([est], [y[j]], "o", color=color, markersize=4)
            p_holm = float(row["p_holm"].iloc[0]) if "p_holm" in row else float("nan")
            stars = ""
            if np.isfinite(p_holm):
                if p_holm < 0.001:
                    stars = "***"
                elif p_holm < 0.01:
                    stars = "**"
                elif p_holm < 0.05:
                    stars = "*"
            if stars:
                ax.text(
                    est, y[j] + 0.18, stars,
                    ha="center", va="bottom",
                    fontsize=FONT_LABEL * 0.8, color=COLOR_BLACK,
                )
        ax.axvline(0.0, color=COLOR_BLACK, lw=0.7, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(conditions)
        ax.set_xlabel("median $\\Delta$", fontsize=FONT_LABEL)
        ax.set_title(metric, fontsize=FONT_TITLE)
        _style_axes(ax)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Test 2 — KLD vs uplift_rel scatter (color by class) + regression line
# ---------------------------------------------------------------------------


def plot_kld_vs_uplift_scatter(
    df: pd.DataFrame,
    out_path: Path,
    *,
    x_col: str = "kld_mean",
    y_col: str = "uplift_rel",
    title: str = "KLD vs relative uplift",
) -> Optional[Path]:
    """Per-sample scatter of $K_i$ vs $\\mathrm{uplift}^{\\mathrm{rel}}_i$.

    Args:
        df: DataFrame with at least ``x_col``, ``y_col``, optional ``label``.
        out_path: Output PDF.
        x_col: Predictor column.
        y_col: Response column.
        title: Figure title.

    Returns:
        ``out_path`` on success.
    """
    if plt is None or df is None or df.empty:
        return None
    if x_col not in df.columns or y_col not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    if "label" in df.columns and unique_labels_in(df["label"]):
        for cls in unique_labels_in(df["label"]):
            sub = df[df["label"] == cls]
            ax.scatter(
                sub[x_col].to_numpy(dtype=np.float64),
                sub[y_col].to_numpy(dtype=np.float64),
                s=6, alpha=0.55,
                color=class_color_for(int(cls)),
                edgecolors="none",
                label=class_label_for(int(cls)),
            )
        ax.legend(loc="best", fontsize=FONT_LABEL * 0.8, frameon=True)
    else:
        ax.scatter(
            df[x_col].to_numpy(dtype=np.float64),
            df[y_col].to_numpy(dtype=np.float64),
            s=6, alpha=0.5, color=COLOR_BLUE, edgecolors="none",
        )

    # OLS reference line + linear-fit shading.
    x = df[x_col].to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() >= 5:
        coeffs = np.polyfit(x[finite], y[finite], 1)
        xr = np.linspace(np.nanmin(x[finite]), np.nanmax(x[finite]), 50)
        ax.plot(xr, np.polyval(coeffs, xr), color=COLOR_BLACK, lw=1.0, ls="-")
    ax.axhline(0.0, color=COLOR_GRAY, lw=0.5, ls=":")
    ax.set_xlabel(x_col, fontsize=FONT_LABEL)
    ax.set_ylabel(y_col, fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE)
    _style_axes(ax)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Test 3 — band coefficients forest plot
# ---------------------------------------------------------------------------


def plot_band_alpha_forest(
    coefs_csv: Path,
    out_path: Path,
    *,
    partition: str = "clinical_7band",
) -> Optional[Path]:
    """Per-band $\\alpha_1$ point estimate + 95% CI on a single axes.

    Args:
        coefs_csv: ``band_uplift_regression/per_band_coefficients.csv``.
        out_path: Output PDF.
        partition: Partition name to plot.

    Returns:
        ``out_path`` on success.
    """
    if plt is None:
        return None
    csv = Path(coefs_csv)
    if not csv.is_file():
        return None
    df = pd.read_csv(csv)
    sub = df[df["partition"] == partition]
    if sub.empty:
        return None
    sub = sub.dropna(subset=["alpha1_estimate"])
    if sub.empty:
        return None

    bands = sub["band"].to_list()
    y = np.arange(len(bands))
    fig, ax = plt.subplots(figsize=(4.4, max(3.0, 0.42 * len(bands) + 1.0)))
    for j, band in enumerate(bands):
        row = sub[sub["band"] == band].iloc[0]
        est = float(row["alpha1_estimate"])
        lo = float(row["alpha1_ci_low"])
        hi = float(row["alpha1_ci_high"])
        color = (
            COLOR_VERMILLION if band in ("deceleration", "early_decel", "late_decel")
            else COLOR_BLUE if band in ("variability", "lf_var", "mf_var")
            else COLOR_GRAY
        )
        if np.isfinite(lo) and np.isfinite(hi):
            ax.errorbar(
                [est], [y[j]],
                xerr=[[est - lo], [hi - est]],
                fmt="o", color=color, ecolor=color,
                capsize=2, markersize=4, lw=1.0,
            )
        else:
            ax.plot([est], [y[j]], "o", color=color, markersize=4)
    ax.axvline(0.0, color=COLOR_BLACK, lw=0.7, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(bands)
    ax.set_xlabel("$\\alpha_{1,b}$ (KLD coef on band uplift)", fontsize=FONT_LABEL)
    ax.set_title(f"Band-specific KLD coefficient ({partition})", fontsize=FONT_TITLE)
    _style_axes(ax)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Test 4 — alignment-error histogram (per class)
# ---------------------------------------------------------------------------


def plot_alignment_error_hist(
    event_pairs_csv: Path,
    out_path: Path,
) -> Optional[Path]:
    """Histogram of $|d^{\\mathrm{model}} - d^{\\mathrm{event}}|$ in seconds."""
    if plt is None:
        return None
    csv = Path(event_pairs_csv)
    if not csv.is_file():
        return None
    df = pd.read_csv(csv)
    if df.empty or "abs_error_s" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    classes = unique_labels_in(df.get("label")) if "label" in df.columns else []
    bins = np.linspace(0.0, 360.0, 25)
    if classes:
        for cls in classes:
            sub = df[df["label"] == cls]
            ax.hist(
                sub["abs_error_s"].to_numpy(dtype=np.float64),
                bins=bins, alpha=0.55, color=class_color_for(int(cls)),
                edgecolor=COLOR_BLACK, lw=0.4, label=class_label_for(int(cls)),
            )
        ax.legend(fontsize=FONT_LABEL * 0.8)
    else:
        ax.hist(
            df["abs_error_s"].to_numpy(dtype=np.float64),
            bins=bins, color=COLOR_BLUE, edgecolor=COLOR_BLACK, lw=0.4,
        )
    ax.axvline(30.0, color=COLOR_BLACK, lw=0.7, ls="--", label="30 s threshold")
    ax.set_xlabel("$|d^{\\mathrm{model}} - d^{\\mathrm{event}}|$ (s)", fontsize=FONT_LABEL)
    ax.set_ylabel("count", fontsize=FONT_LABEL)
    ax.set_title("Lag-event alignment error", fontsize=FONT_TITLE)
    _style_axes(ax)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Test 9 — per-dim contrast forest
# ---------------------------------------------------------------------------


def plot_dim_contrast_forest(
    contrast_csv: Path,
    out_path: Path,
) -> Optional[Path]:
    """Two-column forest plot of $\\Delta_j^{A/H}$ and $\\Delta_j^{HIE/H}$."""
    if plt is None:
        return None
    csv = Path(contrast_csv)
    if not csv.is_file():
        return None
    df = pd.read_csv(csv).sort_values("dim")
    if df.empty:
        return None
    fig, axes = plt.subplots(
        1, 2, figsize=(7.6, max(3.6, 0.16 * len(df) + 1.5)), sharey=True,
    )
    y = np.arange(len(df))
    for ax, prefix, color, title in (
        (axes[0], "A_H", CLASS_COLORS[2], "ACIDOSIS - HEALTHY"),
        (axes[1], "HIE_H", CLASS_COLORS[3], "HIE - HEALTHY"),
    ):
        delta_col = f"delta_{prefix}"
        lo_col = f"ci_{prefix}_low"
        hi_col = f"ci_{prefix}_high"
        if delta_col not in df.columns:
            continue
        for j, (_, row) in enumerate(df.iterrows()):
            est = float(row[delta_col])
            lo = float(row[lo_col]) if lo_col in df.columns else float("nan")
            hi = float(row[hi_col]) if hi_col in df.columns else float("nan")
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(
                    [est], [y[j]],
                    xerr=[[max(0.0, est - lo)], [max(0.0, hi - est)]],
                    fmt="o", color=color, ecolor=color,
                    capsize=1.5, markersize=3, lw=0.8,
                )
            else:
                ax.plot([est], [y[j]], "o", color=color, markersize=3)
        ax.axvline(0.0, color=COLOR_BLACK, lw=0.7, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels([f"dim {int(d)}" for d in df["dim"].to_list()],
                           fontsize=FONT_LABEL * 0.7)
        ax.set_title(title, fontsize=FONT_TITLE)
        ax.set_xlabel("$\\Delta_j$ (mean KLD per dim)", fontsize=FONT_LABEL)
        _style_axes(ax)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Test 10 — event vs quiet violins
# ---------------------------------------------------------------------------


def plot_event_vs_quiet_violins(
    per_sample_csv: Path,
    out_path: Path,
) -> Optional[Path]:
    """Three side-by-side violins of (K, C, te_max) split by event/quiet."""
    if plt is None:
        return None
    csv = Path(per_sample_csv)
    if not csv.is_file():
        return None
    df = pd.read_csv(csv)
    if df.empty:
        return None
    pairs = [
        ("K_event", "K_quiet", "$K_t$ (KLD)"),
        ("C_event", "C_quiet", "$C_t$ (concentration)"),
        ("te_max_event", "te_max_quiet", "max-lag $\\widetilde{TE}_t$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2))
    for ax, (e_col, q_col, title) in zip(axes, pairs):
        if e_col not in df.columns or q_col not in df.columns:
            continue
        ev = df[e_col].dropna().to_numpy(dtype=np.float64)
        qu = df[q_col].dropna().to_numpy(dtype=np.float64)
        if ev.size == 0 and qu.size == 0:
            continue
        parts = ax.violinplot(
            [ev, qu], positions=[1, 2], showmeans=True, showmedians=False,
        )
        body_colors = [COLOR_VERMILLION, COLOR_BLUE]
        for body, c in zip(parts["bodies"], body_colors):
            body.set_facecolor(c)
            body.set_alpha(0.45)
            body.set_edgecolor(COLOR_BLACK)
            body.set_linewidth(0.5)
        for key in ("cbars", "cmins", "cmaxes", "cmeans"):
            if key in parts:
                parts[key].set_color(COLOR_BLACK)
                parts[key].set_lw(0.6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["event", "quiet"])
        ax.set_title(title, fontsize=FONT_TITLE)
        _style_axes(ax)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Headline 4-panel figure
# ---------------------------------------------------------------------------


def plot_causal_te_summary(
    *,
    results_dir: Path,
    out_path: Path,
    representative_sample: Optional[Dict[str, Any]] = None,
    histogram_csv: Optional[Path] = None,
) -> Optional[Path]:
    """Manuscript-ready 2x2 panel summarising the causal-TE validation suite.

    Args:
        results_dir: ``<output>/causal_te_validation`` (root).
        out_path: Output PDF / PNG.
        representative_sample: Optional sample dict (with ``fhr``,
            ``up``, ``te_lag``, ``kld_t``) used in the bottom-left panel.
            When ``None`` the panel shows a placeholder note.
        histogram_csv: Optional ``histograms/histogram_metrics.csv``
            for the top-right scatter; falls back to the per-sample
            data inside the runner's regression DataFrame when absent.

    Returns:
        ``out_path`` on success, ``None`` on failure.
    """
    if plt is None:
        return None
    results_dir = Path(results_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4))

    # (top-left) UP ablation forest of forecast_degradation only.
    ax = axes[0, 0]
    wilcoxon_csv = results_dir / "up_ablation_stats" / "wilcoxon_results.csv"
    if wilcoxon_csv.is_file():
        sub = pd.read_csv(wilcoxon_csv)
        sub = sub[sub["metric"] == "forecast_degradation"]
        conditions = list(sub["condition"]) if not sub.empty else []
        for j, cond in enumerate(conditions):
            row = sub.iloc[j]
            color = _CONDITION_COLORS.get(cond, COLOR_GRAY)
            est = float(row["median_delta"])
            lo = float(row["ci_low"])
            hi = float(row["ci_high"])
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar([est], [j], xerr=[[est - lo], [hi - est]],
                            fmt="o", color=color, ecolor=color, capsize=2,
                            markersize=5, lw=1.0)
            else:
                ax.plot([est], [j], "o", color=color, markersize=5)
        ax.axvline(0.0, color=COLOR_BLACK, lw=0.7, ls="--")
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(conditions)
        ax.set_xlabel("median $\\Delta E$ (vs normal UP)", fontsize=FONT_LABEL)
    else:
        ax.text(0.5, 0.5, "up_ablation_stats not found", ha="center",
                va="center", color=COLOR_GRAY)
    ax.set_title("Test 1 — UP ablation", fontsize=FONT_TITLE)
    _style_axes(ax)

    # (top-right) KLD-uplift scatter.
    ax = axes[0, 1]
    if histogram_csv is not None and Path(histogram_csv).is_file():
        df = pd.read_csv(histogram_csv)
        if "kld_mean" in df.columns and "uplift_rel" in df.columns:
            classes = unique_labels_in(df.get("label"))
            if classes:
                for cls in classes:
                    s = df[df["label"] == cls]
                    ax.scatter(
                        s["kld_mean"].to_numpy(dtype=np.float64),
                        s["uplift_rel"].to_numpy(dtype=np.float64),
                        s=4, alpha=0.45,
                        color=class_color_for(int(cls)),
                        edgecolors="none",
                        label=class_label_for(int(cls)),
                    )
                ax.legend(fontsize=FONT_LABEL * 0.7, loc="best")
            else:
                ax.scatter(
                    df["kld_mean"].to_numpy(dtype=np.float64),
                    df["uplift_rel"].to_numpy(dtype=np.float64),
                    s=4, alpha=0.45, color=COLOR_BLUE, edgecolors="none",
                )
            x = df["kld_mean"].to_numpy(dtype=np.float64)
            y = df["uplift_rel"].to_numpy(dtype=np.float64)
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() >= 5:
                c = np.polyfit(x[finite], y[finite], 1)
                xr = np.linspace(np.nanmin(x[finite]), np.nanmax(x[finite]), 50)
                ax.plot(xr, np.polyval(c, xr), color=COLOR_BLACK, lw=1.0)
            ax.axhline(0.0, color=COLOR_GRAY, lw=0.5, ls=":")
        ax.set_xlabel("$K_i$", fontsize=FONT_LABEL)
        ax.set_ylabel("$\\mathrm{uplift}^{\\mathrm{rel}}_i$", fontsize=FONT_LABEL)
    ax.set_title("Test 2 — KLD vs uplift", fontsize=FONT_TITLE)
    _style_axes(ax)

    # (bottom-left) representative-sample TE-lag overlay.
    ax = axes[1, 0]
    if representative_sample is not None:
        te_lag = np.asarray(representative_sample.get("te_lag"))
        if te_lag.ndim == 2:
            ax.imshow(
                te_lag.T, aspect="auto", origin="lower",
                cmap="magma",
                extent=(0.0, te_lag.shape[0], 0.0, te_lag.shape[1]),
            )
            ax.set_xlabel("anchor (decim step)", fontsize=FONT_LABEL)
            ax.set_ylabel("lag $\\ell$", fontsize=FONT_LABEL)
        else:
            ax.text(0.5, 0.5, "te_lag missing", ha="center", va="center",
                    color=COLOR_GRAY)
    else:
        ax.text(0.5, 0.5, "no representative sample provided",
                ha="center", va="center", color=COLOR_GRAY)
    ax.set_title(r"Test 4 — $\widetilde{TE}_{t,\ell}$ (sample)", fontsize=FONT_TITLE)
    _style_axes(ax)

    # (bottom-right) band $\alpha_1$ forest.
    ax = axes[1, 1]
    coefs_csv = results_dir / "band_uplift_regression" / "per_band_coefficients.csv"
    if coefs_csv.is_file():
        df_coef = pd.read_csv(coefs_csv)
        df_coef = df_coef[df_coef["partition"] == "clinical_7band"]
        df_coef = df_coef.dropna(subset=["alpha1_estimate"])
        if not df_coef.empty:
            bands = df_coef["band"].to_list()
            for j, band in enumerate(bands):
                row = df_coef[df_coef["band"] == band].iloc[0]
                est = float(row["alpha1_estimate"])
                lo = float(row["alpha1_ci_low"])
                hi = float(row["alpha1_ci_high"])
                color = (
                    COLOR_VERMILLION if band in ("deceleration", "early_decel", "late_decel")
                    else COLOR_BLUE if band in ("variability", "lf_var", "mf_var")
                    else COLOR_GRAY
                )
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.errorbar([est], [j], xerr=[[est - lo], [hi - est]],
                                fmt="o", color=color, ecolor=color,
                                capsize=2, markersize=4, lw=1.0)
                else:
                    ax.plot([est], [j], "o", color=color, markersize=4)
            ax.axvline(0.0, color=COLOR_BLACK, lw=0.7, ls="--")
            ax.set_yticks(range(len(bands)))
            ax.set_yticklabels(bands, fontsize=FONT_LABEL * 0.85)
            ax.set_xlabel("$\\alpha_{1,b}$", fontsize=FONT_LABEL)
    ax.set_title("Test 3 — band-specific KLD coef", fontsize=FONT_TITLE)
    _style_axes(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


__all__ = [
    "plot_up_ablation_forest",
    "plot_kld_vs_uplift_scatter",
    "plot_band_alpha_forest",
    "plot_alignment_error_hist",
    "plot_dim_contrast_forest",
    "plot_event_vs_quiet_violins",
    "plot_causal_te_summary",
]
