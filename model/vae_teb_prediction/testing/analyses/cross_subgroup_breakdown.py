"""Phase 2 cross-subgroup post-processor for the lag-attn v1 pipeline.

This module mirrors :mod:`per_class_breakdown` but stratifies by
*subgroup name* (one of the 8 canonical CTG bg/cs cells) instead of by
outcome class id. It is a pure CSV/JSON post-processor: no model load,
no DataLoader iteration, no GPU. It reads each ``phase1/<subgroup>/``
folder produced by Phase 1 of :func:`run_full_test_pipeline_by_subgroup`
and emits cross-subgroup overlays plus formal statistical comparisons
(Kruskal-Wallis across $N$ subgroups, pairwise Mann-Whitney U with
Holm correction, Cliff's $\\delta$ effect size for the headline metrics).

Phase 2 is skipped automatically when only one subgroup is present
(no comparison to make).
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from model.vae_teb_prediction.testing.visualizers import (
    _style_axes,
    plot_aggregate_kld_violins,
    plot_kld_pc_trajectory_grid,
    plot_kld_per_dim_heatmap,
    plot_kld_segment_summary_vs_time,
    plot_kld_trajectory_by_group,
    plot_pc12_mean_trajectory_overlay,
)
from model.vae_teb_prediction.testing.analyses.per_class_breakdown import (
    _emit_horizon_overlay,
    _emit_overlay_for_metric,
)
from model.vae_teb_prediction.testing.analyses.subgroup_utils import (
    CANONICAL_ORDER,
    SUBGROUP_COLORS,
    SUBGROUP_FALLBACK_COLOR,
    SUBGROUP_TO_LABEL,
)

try:
    from scipy import stats as _scipy_stats  # type: ignore[import-untyped]
    _HAS_SCIPY = True
except Exception:  # noqa: BLE001
    _HAS_SCIPY = False
    _scipy_stats = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Headline metric lists — kept in sync with per_class_breakdown.py
# ---------------------------------------------------------------------------

#: Metrics emitted by ``collect_metrics`` and present in
#: ``histograms/histogram_metrics.csv``. Each is a per-sample scalar.
_HISTOGRAM_METRICS: Tuple[str, ...] = (
    "feat_mse_total",
    "feat_r2_total",
    "feat_mse_st",
    "feat_mse_ph",
    "uplift_rel",
    "uplift_abs",
    "residual_ratio",
    "delta_src_norm",
    "kld_mean",
    "kld_sum",
    "kld_l2",
    "kld_pca_l2_selected",
    "kld_pca_abs_sum_selected",
    "kld_pca_signed_sum_selected",
    "posterior_drift_norm",
    "attention_entropy_mean",
    "attention_concentration_mean",
    "te_lag_total_mass",
    "te_lag_peak",
)

#: Subset that drives the grand-summary plot, the Cliff's-delta
#: effect-size table, and the json digest's ``headline_means``.
_HEADLINE_METRICS: Tuple[str, ...] = (
    "feat_mse_total",
    "feat_r2_total",
    "uplift_rel",
    "residual_ratio",
    "kld_mean",
    "te_lag_peak",
)

#: Cliff's $\\delta$ magnitude bins (Romano 2006).
def _cliffs_delta_magnitude(d: float) -> str:
    a = abs(d)
    if a < 0.147:
        return "negligible"
    if a < 0.330:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Discovery & loading helpers
# ---------------------------------------------------------------------------


def _discover_subgroups(phase1_root: Path) -> List[str]:
    """Return subgroup names that have a Phase 1 ``histograms/histogram_metrics.csv``.

    Subdirectories are returned in canonical order first
    (:data:`subgroup_utils.CANONICAL_ORDER`), then any non-canonical names
    in alphabetical order.
    """
    if not phase1_root.is_dir():
        return []
    sentinel = "histograms/histogram_metrics.csv"
    found: List[str] = []
    for child in phase1_root.iterdir():
        if not child.is_dir():
            continue
        if (child / sentinel).exists():
            found.append(child.name)
    canonical = [n for n in CANONICAL_ORDER if n in found]
    others = sorted(n for n in found if n not in CANONICAL_ORDER)
    return canonical + others


def _load_subgroup_csv(
    phase1_root: Path,
    subgroups: Sequence[str],
    rel_path: str,
) -> Dict[str, pd.DataFrame]:
    """Load the same relative CSV for every subgroup.

    Subgroups that don't have the file are silently dropped from the
    returned dict — every analysis emits its own subset and Phase 2 must
    handle missing inputs gracefully.
    """
    out: Dict[str, pd.DataFrame] = {}
    for sg in subgroups:
        path = phase1_root / sg / rel_path
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"cross_subgroup_breakdown: failed to read {path}: {exc}"
            )
            continue
        if df is None or df.empty:
            continue
        out[sg] = df
    return out


def _load_subgroup_json(
    phase1_root: Path,
    subgroups: Sequence[str],
    rel_path: str,
) -> Dict[str, Any]:
    """Load a JSON file from every subgroup folder."""
    out: Dict[str, Any] = {}
    for sg in subgroups:
        path = phase1_root / sg / rel_path
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                out[sg] = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"cross_subgroup_breakdown: failed to read {path}: {exc}"
            )
    return out


def _color_map(subgroups: Iterable[str]) -> Dict[str, str]:
    return {sg: SUBGROUP_COLORS.get(sg, SUBGROUP_FALLBACK_COLOR) for sg in subgroups}


def _name_map(subgroups: Iterable[str]) -> Dict[str, str]:
    """Display label = subgroup name (uppercase replacing underscores)."""
    return {sg: sg for sg in subgroups}


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _holm_bonferroni(pvalues: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment.

    Returns adjusted $p$-values in the original input order. NaN inputs
    are passed through untouched.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = pvalues.size
    if n == 0:
        return pvalues.copy()
    adjusted = np.full(n, np.nan, dtype=float)
    finite_mask = np.isfinite(pvalues)
    finite_idx = np.where(finite_mask)[0]
    if finite_idx.size == 0:
        return adjusted
    # Sort ascending by raw p.
    order = finite_idx[np.argsort(pvalues[finite_idx])]
    m = order.size
    running_max = 0.0
    for k, idx in enumerate(order):
        scaled = pvalues[idx] * (m - k)
        # Step-down: adjusted p is non-decreasing along the sorted order.
        running_max = max(running_max, scaled)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Vectorised Cliff's $\\delta = \\frac{\\#\\{x_i > y_j\\} - \\#\\{x_i < y_j\\}}{|X|\\,|Y|}$."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    # Compare via broadcasting; for very large arrays we chunk.
    chunk = 4096
    gt = lt = 0
    for start in range(0, x.size, chunk):
        block = x[start:start + chunk][:, None]
        gt += int(np.sum(block > y[None, :]))
        lt += int(np.sum(block < y[None, :]))
    return (gt - lt) / float(x.size * y.size)


def _kruskal_wallis(values_per_group: Sequence[np.ndarray]) -> Tuple[float, float, int]:
    """Return ``(H, p, df)`` for the Kruskal-Wallis test.

    Uses ``scipy.stats.kruskal`` when available; falls back to NaN
    otherwise (the analysis still emits the per-group means/medians, just
    without a $p$-value column).
    """
    finite = [v[np.isfinite(v)] for v in values_per_group]
    finite = [v for v in finite if v.size >= 2]
    if len(finite) < 2:
        return (float("nan"), float("nan"), 0)
    if not _HAS_SCIPY or _scipy_stats is None:
        return (float("nan"), float("nan"), len(finite) - 1)
    try:
        h, p = _scipy_stats.kruskal(*finite)
        return (float(h), float(p), len(finite) - 1)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"kruskal_wallis failed: {exc}")
        return (float("nan"), float("nan"), len(finite) - 1)


def _mann_whitney(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Two-sided Mann-Whitney U. Returns ``(U, p)``; ``(NaN, NaN)`` if scipy missing."""
    if not _HAS_SCIPY or _scipy_stats is None:
        return (float("nan"), float("nan"))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return (float("nan"), float("nan"))
    try:
        u, p = _scipy_stats.mannwhitneyu(x, y, alternative="two-sided")
        return (float(u), float(p))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"mann_whitney failed: {exc}")
        return (float("nan"), float("nan"))


# ---------------------------------------------------------------------------
# Per-section processors
# ---------------------------------------------------------------------------


def _process_histogram(
    phase1_root: Path,
    overlay_dir: Path,
    subgroups: Sequence[str],
) -> Dict[str, Any]:
    """Cross-subgroup overlays of ``histograms/histogram_metrics.csv``."""
    per_sg = _load_subgroup_csv(
        phase1_root, subgroups, "histograms/histogram_metrics.csv"
    )
    if not per_sg:
        return {"status": "missing"}
    metrics_present = [
        m for m in _HISTOGRAM_METRICS
        if any(m in df.columns for df in per_sg.values())
    ]
    cmap = _color_map(per_sg.keys())
    nmap = _name_map(per_sg.keys())
    for col in metrics_present:
        _emit_overlay_for_metric(
            per_sg, col,
            overlay_dir / f"histogram_{col}_overlay.pdf",
            colors=cmap, names=nmap, title_suffix="by subgroup",
        )
    return {
        "status": "ok",
        "n_subgroups": len(per_sg),
        "n_metrics": len(metrics_present),
        "metrics": metrics_present,
    }


def _process_forecast(
    phase1_root: Path,
    overlay_dir: Path,
    subgroups: Sequence[str],
) -> Dict[str, Any]:
    """Cross-subgroup overlays of forecast horizon error."""
    per_sg = _load_subgroup_csv(
        phase1_root, subgroups, "forecast_quality/forecast_per_horizon.csv"
    )
    if not per_sg:
        return {"status": "missing"}
    cmap = _color_map(per_sg.keys())
    nmap = _name_map(per_sg.keys())
    for col in ("mse_step", "mse_st", "mse_ph"):
        _emit_horizon_overlay(
            per_sg, col,
            overlay_dir / f"forecast_{col}_horizon_overlay.pdf",
            colors=cmap, names=nmap, title_suffix="by subgroup",
        )
    return {"status": "ok", "n_subgroups": len(per_sg)}


def _process_residual(
    phase1_root: Path,
    overlay_dir: Path,
    subgroups: Sequence[str],
) -> Dict[str, Any]:
    """Cross-subgroup overlays of residual_usage per-sample stats."""
    per_sg = _load_subgroup_csv(
        phase1_root, subgroups, "residual_usage/per_sample.csv"
    )
    if not per_sg:
        return {"status": "missing"}
    cmap = _color_map(per_sg.keys())
    nmap = _name_map(per_sg.keys())
    # Columns produced by ``compute_residual_usage`` and persisted in
    # ``residual_usage/per_sample.csv``. ``delta_src_norm`` lives in the
    # pooled ``histograms/histogram_metrics.csv`` instead and is covered
    # by ``_process_histogram``.
    for col in ("residual_ratio", "delta_norm", "full_norm"):
        _emit_overlay_for_metric(
            per_sg, col,
            overlay_dir / f"residual_{col}_overlay.pdf",
            colors=cmap, names=nmap, title_suffix="by subgroup",
        )
    return {"status": "ok", "n_subgroups": len(per_sg)}


def _process_attention(
    phase1_root: Path,
    overlay_dir: Path,
    subgroups: Sequence[str],
) -> Dict[str, Any]:
    """Cross-subgroup overlays of lag-attention diagnostics."""
    argmax_per_sg = _load_subgroup_csv(
        phase1_root, subgroups, "attention/argmax_lag_per_sample.csv"
    )
    cmap = _color_map(argmax_per_sg.keys())
    nmap = _name_map(argmax_per_sg.keys())
    if argmax_per_sg:
        _emit_overlay_for_metric(
            argmax_per_sg, "argmax_lag",
            overlay_dir / "attention_argmax_lag_overlay.pdf",
            colors=cmap, names=nmap, title_suffix="by subgroup",
        )

    # Mean-attention-mass-by-lag line plot (per-lag mean across samples).
    mass_per_sg = _load_subgroup_csv(
        phase1_root, subgroups, "attention/alpha_mass_by_lag.csv"
    )
    n_mass = 0
    if mass_per_sg:
        try:
            fig, ax = plt.subplots(figsize=(5.6, 3.4))
            for sg, df in mass_per_sg.items():
                if "lag" not in df.columns or "alpha_mass" not in df.columns:
                    continue
                grouped = df.groupby("lag")["alpha_mass"].mean()
                if grouped.empty:
                    continue
                color = SUBGROUP_COLORS.get(sg, SUBGROUP_FALLBACK_COLOR)
                ax.plot(
                    np.asarray(grouped.index, dtype=float),
                    grouped.values,
                    color=color, label=sg,
                )
                n_mass += 1
            if n_mass:
                ax.set_xlabel("lag (samples)")
                ax.set_ylabel("mean attention mass")
                ax.set_title("Lag-attention mass by subgroup")
                ax.legend(loc="best", frameon=True)
                _style_axes(ax)
                fig.tight_layout()
                fig.savefig(overlay_dir / "attention_mass_by_lag_overlay.pdf")
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"attention_mass_by_lag overlay failed: {exc}")

    return {
        "status": "ok" if (argmax_per_sg or mass_per_sg) else "missing",
        "n_subgroups_argmax": len(argmax_per_sg),
        "n_subgroups_mass": n_mass,
    }


def _process_uplift(
    phase1_root: Path,
    overlay_dir: Path,
    subgroups: Sequence[str],
) -> Dict[str, Any]:
    per_sg = _load_subgroup_csv(
        phase1_root, subgroups, "uplift/per_sample.csv"
    )
    if not per_sg:
        return {"status": "missing"}
    cmap = _color_map(per_sg.keys())
    nmap = _name_map(per_sg.keys())
    for col in ("uplift_abs", "uplift_rel", "l_full", "l_base"):
        _emit_overlay_for_metric(
            per_sg, col,
            overlay_dir / f"uplift_{col}_overlay.pdf",
            colors=cmap, names=nmap, title_suffix="by subgroup",
        )
    return {"status": "ok", "n_subgroups": len(per_sg)}


def _process_te_lag(
    phase1_root: Path,
    overlay_dir: Path,
    subgroups: Sequence[str],
) -> Dict[str, Any]:
    """Mean lag profile (te_lag_mean_*) overlay across subgroups."""
    per_sg = _load_subgroup_csv(
        phase1_root, subgroups, "te_lag/te_lag_mean_per_sample.csv"
    )
    if not per_sg:
        return {"status": "missing"}
    try:
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        any_data = False
        for sg, df in per_sg.items():
            lag_cols = sorted(
                (c for c in df.columns if c.startswith("te_lag_mean_")),
                key=lambda c: int(c.rsplit("_", 1)[-1])
                if c.rsplit("_", 1)[-1].isdigit() else 10**9,
            )
            if not lag_cols:
                continue
            arr = df[lag_cols].to_numpy(dtype=float)
            if arr.size == 0:
                continue
            mean = np.nanmean(arr, axis=0)
            sem = np.nanstd(arr, axis=0) / max(np.sqrt(arr.shape[0]), 1.0)
            xs = np.arange(mean.size, dtype=float)
            color = SUBGROUP_COLORS.get(sg, SUBGROUP_FALLBACK_COLOR)
            ax.plot(xs, mean, color=color, label=f"{sg} (n={arr.shape[0]})")
            ax.fill_between(xs, mean - sem, mean + sem, color=color, alpha=0.18, lw=0)
            any_data = True
        if any_data:
            ax.set_xlabel("lag")
            ax.set_ylabel("mean TE")
            ax.set_title("Lag-resolved TE by subgroup")
            ax.legend(loc="best", frameon=True)
            _style_axes(ax)
            fig.tight_layout()
            fig.savefig(overlay_dir / "te_lag_mean_overlay.pdf")
        plt.close(fig)
        return {
            "status": "ok" if any_data else "empty",
            "n_subgroups": len(per_sg),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"te_lag overlay failed: {exc}")
        return {"status": "error", "error": str(exc)}


def _process_kld_pca(
    phase1_root: Path,
    overlay_dir: Path,
    subgroups: Sequence[str],
) -> Dict[str, Any]:
    """PC1 vs PC2 scatter overlay coloured by subgroup."""
    per_sg = _load_subgroup_csv(
        phase1_root, subgroups, "kld_pca/kld_pc_trajectory.csv"
    )
    if not per_sg:
        return {"status": "missing"}

    pc_x_candidates = ("pc1", "kld_pc1_t", "kld_pc1")
    pc_y_candidates = ("pc2", "kld_pc2_t", "kld_pc2")

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    any_data = False
    for sg, df in per_sg.items():
        x_col = next((c for c in pc_x_candidates if c in df.columns), None)
        y_col = next((c for c in pc_y_candidates if c in df.columns), None)
        if x_col is None or y_col is None:
            continue
        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            continue
        color = SUBGROUP_COLORS.get(sg, SUBGROUP_FALLBACK_COLOR)
        ax.scatter(
            x[mask], y[mask], s=4, alpha=0.35, color=color,
            label=f"{sg} (n={int(mask.sum())})", linewidths=0,
        )
        any_data = True
    if not any_data:
        plt.close(fig)
        return {"status": "empty"}
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("KLD per-dim PCA by subgroup")
    ax.legend(loc="best", frameon=True, markerscale=2.0)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(overlay_dir / "kld_pc12_scatter_overlay.pdf")
    plt.close(fig)
    return {"status": "ok", "n_subgroups": len(per_sg)}


def _concat_with_subgroup(
    per_sg: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate per-subgroup DataFrames into one, tagging each row with ``subgroup``.

    Helper for the cross-subgroup KLD overlays below — every plot
    primitive expects a single long DataFrame plus a ``group_col`` to
    pivot/aggregate on, which matches the shape we want once each
    subgroup's own ``subgroup`` column is added.
    """
    frames: List[pd.DataFrame] = []
    for sg, df in per_sg.items():
        if df is None or df.empty:
            continue
        out = df.copy()
        out["subgroup"] = sg
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _process_kld_trajectory_overlay(
    phase1_root: Path,
    overlay_dir: Path,
    subgroups: Sequence[str],
) -> Dict[str, Any]:
    """Mean ± SE of ``kld_mean`` over timestep, one ribbon per subgroup."""
    per_sg = _load_subgroup_csv(
        phase1_root, subgroups, "kld_pca/kld_pc_trajectory.csv"
    )
    combined = _concat_with_subgroup(per_sg)
    if combined.empty:
        return {"status": "missing"}
    if "timestep" not in combined.columns or "kld_mean" not in combined.columns:
        return {"status": "missing_columns"}
    plot_kld_trajectory_by_group(
        combined,
        overlay_dir / "kld_trajectory_overlay.pdf",
        group_col="subgroup",
        metric_col="kld_mean",
        time_col="timestep",
        title="KLD trajectory by subgroup (mean ± SE)",
    )
    return {"status": "ok", "n_subgroups": len(per_sg)}


def _process_kld_per_dim_subgroup(
    phase1_root: Path,
    overlay_dir: Path,
    subgroups: Sequence[str],
) -> Dict[str, Any]:
    """Per-latent-dim mean KLD heatmap, columns = subgroup."""
    per_sg = _load_subgroup_csv(
        phase1_root, subgroups, "histograms/histogram_metrics.csv"
    )
    combined = _concat_with_subgroup(per_sg)
    if combined.empty:
        return {"status": "missing"}
    # Infer d_z from the present ``kld_dim_*`` columns rather than
    # hardcoding it: the model's latent width is recoverable directly
    # from what was written.
    dim_idx: List[int] = []
    for c in combined.columns:
        if isinstance(c, str) and c.startswith("kld_dim_"):
            tail = c.split("_", 2)[-1]
            try:
                dim_idx.append(int(tail))
            except ValueError:
                continue
    if not dim_idx:
        return {"status": "missing_per_dim"}
    d_z = max(dim_idx) + 1
    n_per_subgroup = {sg: int(len(df)) for sg, df in per_sg.items()}
    plot_kld_per_dim_heatmap(
        combined, d_z,
        overlay_dir / "kld_per_dim_by_subgroup_heatmap.pdf",
        group_col="subgroup",
        n_samples_per_group=n_per_subgroup,
    )
    return {"status": "ok", "n_subgroups": len(per_sg), "d_z": d_z}


def _process_kld_pc_mean_trajectory(
    phase1_root: Path,
    overlay_dir: Path,
    subgroups: Sequence[str],
) -> Dict[str, Any]:
    """Per-subgroup mean trajectory in the (PC1, PC2) plane with arrows."""
    per_sg = _load_subgroup_csv(
        phase1_root, subgroups, "kld_pca/kld_pc_trajectory.csv"
    )
    combined = _concat_with_subgroup(per_sg)
    if combined.empty:
        return {"status": "missing"}
    needed = {"timestep", "kld_pc1_t", "kld_pc2_t"}
    if not needed.issubset(combined.columns):
        return {"status": "missing_columns"}
    plot_pc12_mean_trajectory_overlay(
        combined,
        overlay_dir / "kld_pc12_mean_trajectory_overlay.pdf",
        group_col="subgroup",
        time_col="timestep",
        pc1_col="kld_pc1_t",
        pc2_col="kld_pc2_t",
    )
    return {"status": "ok", "n_subgroups": len(per_sg)}


def _process_kld_aggregate_violins(
    phase1_root: Path,
    overlay_dir: Path,
    subgroups: Sequence[str],
) -> Dict[str, Any]:
    """Side-by-side violins of ``kld_mean / kld_sum / kld_l2`` by subgroup."""
    per_sg = _load_subgroup_csv(
        phase1_root, subgroups, "histograms/histogram_metrics.csv"
    )
    combined = _concat_with_subgroup(per_sg)
    if combined.empty:
        return {"status": "missing"}
    plot_aggregate_kld_violins(
        combined,
        overlay_dir / "kld_aggregate_violins.pdf",
        group_col="subgroup",
        metric_cols=("kld_mean", "kld_sum", "kld_l2"),
    )
    return {"status": "ok", "n_subgroups": len(per_sg)}


# ---------------------------------------------------------------------------
# KLD-vs-time-to-delivery overlays (per-segment summaries + first-6 PCs)
# ---------------------------------------------------------------------------
#
# These four overlays read the raw per-(guid, epoch) KLD CSV produced by
# the trajectory analysis (``trajectory/kld_trajectory_raw.csv``) and the
# per-(sample, timestep) KLD-PC CSV produced by ``kld_pca``
# (``kld_pca/kld_pc_trajectory.csv``). The plotting primitives are
# group-aware (``group_col="subgroup"``) so the same visualizers used in
# Phase 1 produce these Phase 2 overlays without duplication.


def _subgroup_palette(subgroups: Sequence[str]) -> Dict[str, str]:
    """Return a ``{subgroup_name: hex}`` palette for the requested subgroups.

    Falls back to ``SUBGROUP_FALLBACK_COLOR`` for any non-canonical
    subgroup name.
    """
    return {
        sg: SUBGROUP_COLORS.get(sg, SUBGROUP_FALLBACK_COLOR)
        for sg in subgroups
    }


def _process_kld_trajectory_hours_overlay(
    phase1_root: Path,
    overlay_dir: Path,
    subgroups: Sequence[str],
) -> Dict[str, Any]:
    """Per-subgroup mean ± SE of ``kld_mean`` vs hours-before-birth.

    Mirrors the Phase 1 ``trajectory/plots/kld_trajectory.pdf`` plot but
    overlays one line per subgroup on the same axes. Reads
    ``trajectory/kld_trajectory_raw.csv`` per subgroup.
    """
    per_sg = _load_subgroup_csv(
        phase1_root, subgroups, "trajectory/kld_trajectory_raw.csv"
    )
    combined = _concat_with_subgroup(per_sg)
    if combined.empty:
        return {"status": "missing"}
    if "hours_before" not in combined.columns or "kld_mean" not in combined.columns:
        return {"status": "missing_columns"}
    plot_kld_segment_summary_vs_time(
        combined,
        overlay_dir / "kld_trajectory_hours_overlay.pdf",
        metric_col="kld_mean",
        group_col="subgroup",
        n_samples=int(len(combined)),
        palette=_subgroup_palette(subgroups),
    )
    return {"status": "ok", "n_subgroups": len(per_sg)}


def _process_kld_segment_l2sq_overlay(
    phase1_root: Path,
    overlay_dir: Path,
    subgroups: Sequence[str],
) -> Dict[str, Any]:
    """Per-subgroup mean ± SE of ``kld_l2sq`` vs hours-before-birth.

    Reveals subgroup differences in *burstiness* of latent information —
    a heavy ``\\|\\mathrm{KLD}\\|_2^2`` tail with similar mean indicates
    spiky transfer rather than uniformly elevated transfer.
    """
    per_sg = _load_subgroup_csv(
        phase1_root, subgroups, "trajectory/kld_trajectory_raw.csv"
    )
    combined = _concat_with_subgroup(per_sg)
    if combined.empty:
        return {"status": "missing"}
    if "hours_before" not in combined.columns or "kld_l2sq" not in combined.columns:
        return {"status": "missing_columns"}
    plot_kld_segment_summary_vs_time(
        combined,
        overlay_dir / "kld_segment_l2sq_overlay.pdf",
        metric_col="kld_l2sq",
        group_col="subgroup",
        n_samples=int(len(combined)),
        palette=_subgroup_palette(subgroups),
    )
    return {"status": "ok", "n_subgroups": len(per_sg)}


def _process_kld_segment_max_overlay(
    phase1_root: Path,
    overlay_dir: Path,
    subgroups: Sequence[str],
) -> Dict[str, Any]:
    """Per-subgroup mean ± SE of ``kld_max`` vs hours-before-birth.

    Captures the largest single-timestep KLD per segment per subgroup —
    a higher ``max`` with similar mean indicates a few extreme spikes
    drive the difference.
    """
    per_sg = _load_subgroup_csv(
        phase1_root, subgroups, "trajectory/kld_trajectory_raw.csv"
    )
    combined = _concat_with_subgroup(per_sg)
    if combined.empty:
        return {"status": "missing"}
    if "hours_before" not in combined.columns or "kld_max" not in combined.columns:
        return {"status": "missing_columns"}
    plot_kld_segment_summary_vs_time(
        combined,
        overlay_dir / "kld_segment_max_overlay.pdf",
        metric_col="kld_max",
        group_col="subgroup",
        n_samples=int(len(combined)),
        palette=_subgroup_palette(subgroups),
    )
    return {"status": "ok", "n_subgroups": len(per_sg)}


def _process_kld_pc_trajectory_grid_overlay(
    phase1_root: Path,
    overlay_dir: Path,
    subgroups: Sequence[str],
) -> Dict[str, Any]:
    """6-panel grid (one panel per PC) of per-subgroup mean ± SE vs time-to-delivery.

    Reads each subgroup's ``kld_pca/kld_pc_trajectory.csv`` and uses the
    ``kld_pc_top{i}_t`` columns (first-6-by-eigenvalue) so the overlay
    reflects the leading latent-information directions independent of
    the contrast-selected ``kld_pc{i}_t`` family.
    """
    per_sg = _load_subgroup_csv(
        phase1_root, subgroups, "kld_pca/kld_pc_trajectory.csv"
    )
    combined = _concat_with_subgroup(per_sg)
    if combined.empty:
        return {"status": "missing"}
    has_top = any(
        isinstance(c, str) and c.startswith("kld_pc_top") and c.endswith("_t")
        for c in combined.columns
    )
    if not has_top:
        return {"status": "missing_top_pc_columns"}
    if "hours_before" not in combined.columns:
        return {"status": "missing_hours_before"}
    plot_kld_pc_trajectory_grid(
        combined,
        overlay_dir / "kld_pc_trajectory_grid_overlay.pdf",
        n_components=6,
        group_col="subgroup",
        n_samples=int(combined.groupby(["guid", "epoch"]).ngroups),
        palette=_subgroup_palette(subgroups),
        pc_prefix="kld_pc_top",
    )
    return {"status": "ok", "n_subgroups": len(per_sg)}


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------


def _collect_metric_values(
    per_sg: Dict[str, pd.DataFrame],
    metric: str,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for sg, df in per_sg.items():
        if metric not in df.columns:
            continue
        vals = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[sg] = vals
    return out


def _process_stats(
    phase1_root: Path,
    output_dir: Path,
    subgroups: Sequence[str],
) -> Dict[str, Any]:
    """Kruskal-Wallis / pairwise Mann-Whitney / Cliff's $\\delta$ tables."""
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    per_sg = _load_subgroup_csv(
        phase1_root, subgroups, "histograms/histogram_metrics.csv"
    )
    if not per_sg:
        return {"status": "missing"}

    metrics_present = [
        m for m in _HISTOGRAM_METRICS
        if any(m in df.columns for df in per_sg.values())
    ]

    # ---- Kruskal-Wallis ----
    kw_rows: List[Dict[str, Any]] = []
    for metric in metrics_present:
        groups = _collect_metric_values(per_sg, metric)
        if len(groups) < 2:
            continue
        h, p, df_ = _kruskal_wallis(list(groups.values()))
        kw_rows.append({
            "metric": metric,
            "H": h,
            "p": p,
            "df": df_,
            "n_subgroups": len(groups),
            "n_total": int(sum(v.size for v in groups.values())),
        })
    if kw_rows:
        kw_df = pd.DataFrame(kw_rows)
        kw_df["p_holm"] = _holm_bonferroni(kw_df["p"].to_numpy())
        kw_df["significant"] = (kw_df["p_holm"] < 0.05) & np.isfinite(kw_df["p_holm"])
        kw_df.to_csv(stats_dir / "kruskal_wallis.csv", index=False)
    else:
        kw_df = pd.DataFrame()

    # ---- Pairwise Mann-Whitney (only for metrics significant after Holm) ----
    mw_rows: List[Dict[str, Any]] = []
    significant_metrics: List[str] = []
    if not kw_df.empty:
        sig_mask = kw_df["significant"].to_numpy(dtype=bool)
        significant_metrics = kw_df.loc[sig_mask, "metric"].tolist()
    for metric in significant_metrics:
        groups = _collect_metric_values(per_sg, metric)
        names = list(groups.keys())
        for a, b in combinations(names, 2):
            u, p = _mann_whitney(groups[a], groups[b])
            mw_rows.append({
                "metric": metric,
                "subgroup_a": a,
                "subgroup_b": b,
                "U": u,
                "p": p,
                "n_a": int(groups[a].size),
                "n_b": int(groups[b].size),
            })
    if mw_rows:
        mw_df = pd.DataFrame(mw_rows)
        # Holm within each metric (family = pairs for that metric).
        mw_df["p_holm"] = np.nan
        for metric in mw_df["metric"].unique():
            mask = mw_df["metric"] == metric
            mw_df.loc[mask, "p_holm"] = _holm_bonferroni(
                mw_df.loc[mask, "p"].to_numpy()
            )
        mw_df["significant"] = (mw_df["p_holm"] < 0.05) & np.isfinite(mw_df["p_holm"])
        mw_df.to_csv(stats_dir / "pairwise_mann_whitney.csv", index=False)

    # ---- Cliff's delta on headline metrics ----
    cd_rows: List[Dict[str, Any]] = []
    for metric in (m for m in _HEADLINE_METRICS if m in metrics_present):
        groups = _collect_metric_values(per_sg, metric)
        names = list(groups.keys())
        for a, b in combinations(names, 2):
            d = _cliffs_delta(groups[a], groups[b])
            cd_rows.append({
                "metric": metric,
                "subgroup_a": a,
                "subgroup_b": b,
                "cliffs_delta": d,
                "magnitude": (
                    "nan" if not math.isfinite(d) else _cliffs_delta_magnitude(d)
                ),
                "n_a": int(groups[a].size),
                "n_b": int(groups[b].size),
            })
    if cd_rows:
        cd_df = pd.DataFrame(cd_rows)
        cd_df.to_csv(stats_dir / "cliffs_delta.csv", index=False)

    summary = {
        "n_metrics_tested": len(kw_rows),
        "n_significant_after_holm": int(
            kw_df["significant"].sum()) if not kw_df.empty else 0,
        "significant_metrics": significant_metrics,
        "n_pairwise_tests": len(mw_rows),
        "n_cliffs_delta_rows": len(cd_rows),
        "scipy_available": _HAS_SCIPY,
    }
    with (stats_dir / "stats_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return {"status": "ok", **summary}


# ---------------------------------------------------------------------------
# Grand summary plot + JSON digest
# ---------------------------------------------------------------------------


def _process_grand_summary(
    phase1_root: Path,
    output_dir: Path,
    subgroups: Sequence[str],
) -> Dict[str, Any]:
    """Per-subgroup means ± SE for the headline metrics, plus tidy CSV."""
    per_sg = _load_subgroup_csv(
        phase1_root, subgroups, "histograms/histogram_metrics.csv"
    )
    if not per_sg:
        return {"status": "missing"}

    metrics = [m for m in _HEADLINE_METRICS if any(m in df.columns for df in per_sg.values())]
    if not metrics:
        return {"status": "no_metrics"}

    rows: List[Dict[str, Any]] = []
    for metric in metrics:
        for sg, df in per_sg.items():
            if metric not in df.columns:
                continue
            vals = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            rows.append({
                "metric": metric,
                "subgroup": sg,
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "sem": float(np.std(vals) / max(np.sqrt(vals.size), 1.0)),
                "n": int(vals.size),
            })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "grand_summary.csv", index=False)

    # Multi-panel plot: one subplot per headline metric, bars per subgroup.
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = int(math.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.0 * n_cols, 3.0 * n_rows), squeeze=False,
    )
    for idx, metric in enumerate(metrics):
        ax = axes[idx // n_cols][idx % n_cols]
        m_df = summary_df[summary_df["metric"] == metric]
        if m_df.empty:
            ax.set_visible(False)
            continue
        sgs = list(m_df["subgroup"])
        means = m_df["mean"].to_numpy(dtype=float)
        sems = m_df["sem"].to_numpy(dtype=float)
        colors = [SUBGROUP_COLORS.get(s, SUBGROUP_FALLBACK_COLOR) for s in sgs]
        xs = np.arange(len(sgs))
        ax.bar(
            xs, means, yerr=sems, color=colors,
            edgecolor="#222831", linewidth=0.4, capsize=2,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(sgs, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("mean ± SE")
        ax.set_title(metric)
        _style_axes(ax)
    # Hide any extra axes.
    for j in range(n_metrics, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)
    fig.suptitle("Headline metrics by subgroup", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "grand_summary.pdf")
    plt.close(fig)

    return {
        "status": "ok",
        "n_subgroups": len(per_sg),
        "n_metrics": n_metrics,
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_cross_subgroup_breakdown(
    phase1_root: Path,
    output_dir: Path,
    *,
    subgroups: Optional[Sequence[str]] = None,
    parts: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Phase 2 of the per-subgroup test pipeline.

    Reads the per-subgroup CSV/JSON tree under ``phase1_root`` (one
    subdirectory per subgroup, each laid out exactly like a single
    ``run_full_test_pipeline`` output) and emits cross-subgroup overlays
    plus statistical comparisons under ``output_dir``.

    Args:
        phase1_root: Directory whose immediate children are subgroup
            folders (e.g. ``<base>/phase1/``).
        output_dir: Where the Phase 2 artefacts should land
            (e.g. ``<base>/phase2/cross_subgroup/``). Created if missing.
        subgroups: Optional explicit list of subgroup names to include.
            When ``None``, every subdirectory under ``phase1_root`` that
            contains ``histograms/histogram_metrics.csv`` is used.
        parts: Optional subset of analyses. Defaults to every section
            implemented here.

    Returns:
        Dict mapping each section name to its summary status, plus a
        top-level ``cross_subgroup_summary.json`` digest.
    """
    phase1_root = Path(phase1_root)
    output_dir = Path(output_dir)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    if subgroups is None:
        subgroups = _discover_subgroups(phase1_root)
    else:
        subgroups = list(subgroups)
    subgroups = [s for s in subgroups if (phase1_root / s).is_dir()]

    if len(subgroups) < 2:
        msg = (
            f"cross_subgroup_breakdown: only {len(subgroups)} subgroup(s) "
            f"with Phase 1 output under {phase1_root} — nothing to compare."
        )
        logger.warning(msg)
        return {"status": "skipped_single_subgroup", "n_subgroups": len(subgroups)}

    logger.info(
        f"cross_subgroup_breakdown: comparing {len(subgroups)} subgroups: {subgroups}"
    )

    available_parts = {
        "histogram":     lambda: _process_histogram(phase1_root, overlay_dir, subgroups),
        "forecast":      lambda: _process_forecast(phase1_root, overlay_dir, subgroups),
        "residual":      lambda: _process_residual(phase1_root, overlay_dir, subgroups),
        "attention":     lambda: _process_attention(phase1_root, overlay_dir, subgroups),
        "uplift":        lambda: _process_uplift(phase1_root, overlay_dir, subgroups),
        "te_lag":        lambda: _process_te_lag(phase1_root, overlay_dir, subgroups),
        "kld_pca":       lambda: _process_kld_pca(phase1_root, overlay_dir, subgroups),
        "kld_trajectory":
            lambda: _process_kld_trajectory_overlay(
                phase1_root, overlay_dir, subgroups,
            ),
        "kld_per_dim_subgroup":
            lambda: _process_kld_per_dim_subgroup(
                phase1_root, overlay_dir, subgroups,
            ),
        "kld_pc_mean_trajectory":
            lambda: _process_kld_pc_mean_trajectory(
                phase1_root, overlay_dir, subgroups,
            ),
        "kld_aggregate_violins":
            lambda: _process_kld_aggregate_violins(
                phase1_root, overlay_dir, subgroups,
            ),
        "kld_trajectory_hours":
            lambda: _process_kld_trajectory_hours_overlay(
                phase1_root, overlay_dir, subgroups,
            ),
        "kld_segment_l2sq":
            lambda: _process_kld_segment_l2sq_overlay(
                phase1_root, overlay_dir, subgroups,
            ),
        "kld_segment_max":
            lambda: _process_kld_segment_max_overlay(
                phase1_root, overlay_dir, subgroups,
            ),
        "kld_pc_trajectory_grid":
            lambda: _process_kld_pc_trajectory_grid_overlay(
                phase1_root, overlay_dir, subgroups,
            ),
        "stats":         lambda: _process_stats(phase1_root, output_dir, subgroups),
        "summary":       lambda: _process_grand_summary(
            phase1_root, output_dir, subgroups,
        ),
    }
    if parts is None:
        parts = list(available_parts.keys())

    results: Dict[str, Any] = {"subgroups": list(subgroups)}
    for name in parts:
        fn = available_parts.get(name)
        if fn is None:
            continue
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"cross_subgroup_breakdown[{name}] failed: {exc}")
            results[name] = {"status": "error", "error": str(exc)}

    # Top-level JSON digest.
    digest: Dict[str, Any] = {
        "subgroups": list(subgroups),
        "n_subgroups": len(subgroups),
        "headline_means": {},
        "n_samples_per_subgroup": {},
    }
    # Pull n_samples + headline means from the same histogram CSVs.
    per_sg = _load_subgroup_csv(
        phase1_root, subgroups, "histograms/histogram_metrics.csv",
    )
    for sg, df in per_sg.items():
        digest["n_samples_per_subgroup"][sg] = int(len(df))
    for metric in _HEADLINE_METRICS:
        digest["headline_means"][metric] = {}
        for sg, df in per_sg.items():
            if metric not in df.columns:
                continue
            vals = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                digest["headline_means"][metric][sg] = float(np.mean(vals))
    digest["expected_class_per_subgroup"] = {
        sg: SUBGROUP_TO_LABEL.get(sg) for sg in subgroups
    }
    digest["sections"] = {k: v for k, v in results.items() if k != "subgroups"}

    with (output_dir / "cross_subgroup_summary.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(digest, fh, indent=2)

    logger.info(
        f"cross_subgroup_breakdown: complete. "
        f"Outputs under {output_dir}"
    )
    return results


__all__ = [
    "run_cross_subgroup_breakdown",
]
