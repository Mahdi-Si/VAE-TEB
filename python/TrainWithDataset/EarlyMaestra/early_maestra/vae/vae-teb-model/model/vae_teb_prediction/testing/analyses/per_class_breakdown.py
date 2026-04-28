"""Per-class breakdown of pooled CSVs produced by the lag-attn v1 pipeline.

This module is a *pure post-processor*: it does not run the model. Once
``run_all_analyses`` has emitted its pooled CSVs (``histograms/``,
``forecast_quality/``, ``residual_usage/``, ``attention/``,
``horizon_error/``, ``anchor_error/``), this module reads them, splits
each by the ``label`` column (1=HEALTHY, 2=ACIDOSIS, 3=HIE), and writes
per-class subfolders plus a sibling ``class_overlay`` folder that
overlays the same metric across the three classes on shared axes.

Why a separate post-processor: each of the existing analyses already
runs its own DataLoader pass, and re-running inference four times (once
for each class subset) would be wasteful when the per-sample CSVs
already carry the class label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from model.vae_teb_prediction.testing.visualizers import (
    COLOR_BLUE,
    COLOR_ORANGE,
    COLOR_VERMILLION,
    _style_axes,
    plot_forecast_error_by_horizon,
    plot_metric_histograms,
    plot_residual_usage_trace,
    plot_uplift_histogram,
)

CLASS_NAMES = {1: "HEALTHY", 2: "ACIDOSIS", 3: "HIE"}
CLASS_FOLDER = {1: "class_healthy", 2: "class_acidosis", 3: "class_hie"}
CLASS_COLORS = {1: COLOR_BLUE, 2: COLOR_ORANGE, 3: COLOR_VERMILLION}


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV file into a DataFrame, returning None on failure."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"per_class_breakdown: failed to read {path}: {exc}")
        return None


def _filter_by_label(df: pd.DataFrame, label: int) -> pd.DataFrame:
    """Return rows with ``label == label``, preserving the schema."""
    if "label" not in df.columns:
        return df.iloc[0:0]
    mask = df["label"] == label
    return pd.DataFrame(df.loc[mask].copy())


def _emit_overlay_for_metric(
    per_class_dfs: Dict[int, pd.DataFrame],
    metric_col: str,
    out_path: Path,
    *,
    bins: int = 30,
    log: bool = False,
) -> None:
    """Overlay histograms of one metric column across the three classes."""
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    any_data = False
    for label_id, df in per_class_dfs.items():
        if metric_col not in df.columns:
            continue
        vals = df[metric_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        ax.hist(
            vals,
            bins=bins,
            histtype="stepfilled",
            color=CLASS_COLORS[label_id],
            alpha=0.4,
            label=f"{CLASS_NAMES[label_id]} (n={vals.size})",
            density=True,
        )
        any_data = True
    if not any_data:
        plt.close(fig)
        return
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(metric_col)
    ax.set_ylabel("density")
    ax.set_title(f"{metric_col} by class")
    ax.legend(loc="best", frameon=True)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _emit_horizon_overlay(
    per_class_dfs: Dict[int, pd.DataFrame],
    metric_col: str,
    out_path: Path,
    *,
    x_col: str = "h",
) -> None:
    """Line plot of ``metric_col`` aggregated over an x-axis grouping."""
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    any_data = False
    for label_id, df in per_class_dfs.items():
        if x_col not in df.columns or metric_col not in df.columns:
            continue
        grouped = df.groupby(x_col)[metric_col]
        median = grouped.median()
        q1 = grouped.quantile(0.25)
        q3 = grouped.quantile(0.75)
        if median.empty:
            continue
        x_vals = np.asarray(median.index.to_list(), dtype=float)
        ax.plot(
            x_vals,
            median.values,
            color=CLASS_COLORS[label_id],
            label=f"{CLASS_NAMES[label_id]} (n={len(df)})",
        )
        ax.fill_between(
            x_vals,
            q1.values,
            q3.values,
            color=CLASS_COLORS[label_id],
            alpha=0.18,
            lw=0,
        )
        any_data = True
    if not any_data:
        plt.close(fig)
        return
    ax.set_xlabel(x_col)
    ax.set_ylabel(f"{metric_col} (median, IQR)")
    ax.set_title(f"{metric_col} vs {x_col} by class")
    ax.legend(loc="best", frameon=True)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _process_histogram(output_root: Path, overlay_dir: Path) -> Dict[str, Any]:
    """Per-class subfolders + overlay for the histogram CSV."""
    src = output_root / "histograms" / "histogram_metrics.csv"
    df = _load_csv(src)
    if df is None or "label" not in df.columns:
        return {"status": "missing"}

    metric_cols = [
        "feat_mse_total",
        "feat_r2_total",
        "uplift_rel",
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
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    # Surface missing classes once before the loop. Otherwise an upstream
    # loader that drops a whole outcome (e.g. only HIE survives the filter)
    # produces a one-class breakdown silently.
    if "label" in df.columns:
        present_labels = {
            int(v) for v in df["label"].dropna().unique()
            if int(v) in CLASS_NAMES
        }
        missing = [
            CLASS_NAMES[lid] for lid in CLASS_NAMES if lid not in present_labels
        ]
        if missing:
            logger.warning(
                f"per_class_breakdown[histogram]: classes "
                f"{missing} have no rows in the input CSV; "
                f"per-class subfolders / overlays for them will be skipped."
            )

    per_class: Dict[int, pd.DataFrame] = {}
    for label_id in CLASS_NAMES:
        sub = _filter_by_label(df, label_id)
        if sub.empty:
            continue
        per_class[label_id] = sub
        out_dir = output_root / "per_class_breakdown" / CLASS_FOLDER[label_id]
        out_dir.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_dir / "histogram_metrics.csv", index=False)
        try:
            plot_metric_histograms(sub, out_dir, filename="metrics_histograms.pdf")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"per_class_breakdown[histogram]: plot failed for "
                f"class {CLASS_NAMES[label_id]}: {exc}"
            )

    for col in metric_cols:
        _emit_overlay_for_metric(
            per_class, col, overlay_dir / f"{col}_overlay.pdf"
        )

    return {"status": "ok", "n_classes": len(per_class), "n_metrics": len(metric_cols)}


def _process_forecast_quality(output_root: Path, overlay_dir: Path) -> Dict[str, Any]:
    """Per-class horizon-error overlay for forecast_quality CSVs."""
    per_horizon = _load_csv(output_root / "forecast_quality" / "forecast_per_horizon.csv")
    if per_horizon is None or "label" not in per_horizon.columns:
        return {"status": "missing"}

    per_class: Dict[int, pd.DataFrame] = {}
    for label_id in CLASS_NAMES:
        sub = _filter_by_label(per_horizon, label_id)
        if sub.empty:
            continue
        per_class[label_id] = sub
        out_dir = output_root / "per_class_breakdown" / CLASS_FOLDER[label_id]
        out_dir.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_dir / "forecast_per_horizon.csv", index=False)
        try:
            plot_forecast_error_by_horizon(sub, out_dir / "forecast_error_by_horizon.pdf")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"per_class_breakdown[forecast]: plot failed for "
                f"class {CLASS_NAMES[label_id]}: {exc}"
            )

    for col in ("mse_step", "mse_st", "mse_ph"):
        _emit_horizon_overlay(per_class, col, overlay_dir / f"forecast_{col}_overlay.pdf")

    return {"status": "ok", "n_classes": len(per_class)}


def _process_residual_usage(output_root: Path, overlay_dir: Path) -> Dict[str, Any]:
    """Per-class breakdown of residual_usage outputs."""
    per_sample = _load_csv(output_root / "residual_usage" / "per_sample.csv")
    per_trace = _load_csv(output_root / "residual_usage" / "per_sample_trace.csv")

    per_class_sample: Dict[int, pd.DataFrame] = {}
    if per_sample is not None and "label" in per_sample.columns:
        for label_id in CLASS_NAMES:
            sub = _filter_by_label(per_sample, label_id)
            if sub.empty:
                continue
            per_class_sample[label_id] = sub
            out_dir = output_root / "per_class_breakdown" / CLASS_FOLDER[label_id]
            out_dir.mkdir(parents=True, exist_ok=True)
            sub.to_csv(out_dir / "residual_per_sample.csv", index=False)

    if per_trace is not None and "label" in per_trace.columns:
        per_class_trace: Dict[int, pd.DataFrame] = {}
        for label_id in CLASS_NAMES:
            sub = _filter_by_label(per_trace, label_id)
            if sub.empty:
                continue
            per_class_trace[label_id] = sub
            out_dir = output_root / "per_class_breakdown" / CLASS_FOLDER[label_id]
            out_dir.mkdir(parents=True, exist_ok=True)
            sub.to_csv(out_dir / "residual_per_sample_trace.csv", index=False)
            try:
                plot_residual_usage_trace(sub, out_dir / "delta_norm_trace.pdf")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"per_class_breakdown[residual]: trace plot failed for "
                    f"class {CLASS_NAMES[label_id]}: {exc}"
                )

    for col in ("residual_ratio", "delta_norm"):
        _emit_overlay_for_metric(
            per_class_sample, col, overlay_dir / f"residual_{col}_overlay.pdf"
        )

    return {"status": "ok", "n_classes": len(per_class_sample)}


def _process_attention(output_root: Path, overlay_dir: Path) -> Dict[str, Any]:
    """Per-class breakdown for attention diagnostics."""
    argmax_csv = _load_csv(output_root / "attention" / "argmax_lag_per_sample.csv")
    head_entropy_csv = _load_csv(output_root / "attention" / "head_entropy_summary.csv")

    per_class_argmax: Dict[int, pd.DataFrame] = {}
    if argmax_csv is not None and "label" in argmax_csv.columns:
        for label_id in CLASS_NAMES:
            sub = _filter_by_label(argmax_csv, label_id)
            if sub.empty:
                continue
            per_class_argmax[label_id] = sub
            out_dir = output_root / "per_class_breakdown" / CLASS_FOLDER[label_id]
            out_dir.mkdir(parents=True, exist_ok=True)
            sub.to_csv(out_dir / "argmax_lag_per_sample.csv", index=False)

    if head_entropy_csv is not None and "label" in head_entropy_csv.columns:
        for label_id in CLASS_NAMES:
            sub = _filter_by_label(head_entropy_csv, label_id)
            if sub.empty:
                continue
            out_dir = output_root / "per_class_breakdown" / CLASS_FOLDER[label_id]
            out_dir.mkdir(parents=True, exist_ok=True)
            sub.to_csv(out_dir / "head_entropy_summary.csv", index=False)

    for col in ("argmax_lag",):
        _emit_overlay_for_metric(
            per_class_argmax, col, overlay_dir / f"attention_{col}_overlay.pdf"
        )

    return {"status": "ok", "n_classes": len(per_class_argmax)}


# Order in which we walk partition subdirectories. Each entry is the
# folder name under ``frequency_band_forecast/``; the legacy top-level
# CSV is duplicated from ``clinical_4band`` and is no longer read here.
_FBF_PARTITION_DIRS: Tuple[str, ...] = (
    "clinical_4band",
    "clinical_7band",
    "by_kind",
    "by_octave",
)

# Canonical display order per partition. When a label isn't in the
# canonical list we keep it in alphabetical order at the end (octave_*
# is the obvious case once `octave_dc` is mixed in).
_FBF_CANONICAL_ORDER: Dict[str, Tuple[str, ...]] = {
    "clinical_4band": (
        "slow_baseline", "deceleration", "variability", "beat_to_beat",
    ),
    "clinical_7band": (
        "baseline", "early_decel", "late_decel",
        "lf_var", "mf_var", "beat_to_beat", "nyquist_edge",
    ),
    "by_kind": (
        "st_S0", "st_S1", "ph_diag", "ph_h2", "ph_h3", "ph_other",
    ),
    "by_octave": tuple(),  # numeric order computed below
}


def _ordered_partition_labels(
    df: pd.DataFrame, partition: str,
) -> List[str]:
    """Return the labels of ``df['band']`` in display order for a partition."""
    seen = sorted({str(v) for v in df["band"].dropna().unique()})
    canonical = _FBF_CANONICAL_ORDER.get(partition, tuple())
    if canonical:
        ordered = [b for b in canonical if b in seen]
        leftover = [b for b in seen if b not in canonical]
        return ordered + leftover
    if partition == "by_octave":
        # octave_<int> first (numeric), then octave_dc / anything else.
        def _key(name: str) -> Tuple[int, str]:
            if name.startswith("octave_"):
                tail = name.split("_", 1)[1]
                try:
                    return (int(tail), name)
                except ValueError:
                    return (10**9, name)
            return (10**9, name)
        return sorted(seen, key=_key)
    return seen


def _process_frequency_band_partition(
    output_root: Path,
    overlay_dir: Path,
    partition: str,
) -> Dict[str, Any]:
    """Run the per-class breakdown for one partition subdirectory.

    Walks ``frequency_band_forecast/<partition>/per_sample.csv`` and
    emits per-class subset CSVs into
    ``per_class_breakdown/class_<...>/frequency_band_<partition>/`` plus
    cross-class overlay PDFs prefixed by ``<partition>_``.
    """
    src_dir = output_root / "frequency_band_forecast" / partition
    src = _load_csv(src_dir / "per_sample.csv")
    if src is None or "label" not in src.columns or "band" not in src.columns:
        return {"status": "missing"}

    per_class: Dict[int, pd.DataFrame] = {}
    for label_id in CLASS_NAMES:
        sub = _filter_by_label(src, label_id)
        if sub.empty:
            continue
        per_class[label_id] = sub
        out_dir = (
            output_root / "per_class_breakdown" / CLASS_FOLDER[label_id]
            / f"frequency_band_{partition}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_dir / "per_sample.csv", index=False)

    if not per_class:
        return {"status": "empty"}

    bands_present = _ordered_partition_labels(src, partition)
    if not bands_present:
        return {"status": "no_bands"}

    # Class-x-band overlay: one bar per (band, class) showing mean MSE.
    try:
        fig, ax = plt.subplots(
            figsize=(max(4.4, 1.4 * len(bands_present) + 1.4), 3.4)
        )
        n_classes = len(per_class)
        bar_width = 0.8 / max(n_classes, 1)
        for c_idx, (label_id, df) in enumerate(per_class.items()):
            means: List[float] = []
            stds: List[float] = []
            for band in bands_present:
                vals = pd.to_numeric(
                    df.loc[df["band"] == band, "mse_total"], errors="coerce"
                ).to_numpy()
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    means.append(float("nan"))
                    stds.append(0.0)
                else:
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals) / max(np.sqrt(vals.size), 1)))
            xs = np.arange(len(bands_present))
            offsets = (c_idx - (n_classes - 1) / 2.0) * bar_width
            ax.bar(
                xs + offsets, means, bar_width,
                yerr=stds, color=CLASS_COLORS[label_id],
                label=f"{CLASS_NAMES[label_id]} (n={len(df)})",
                edgecolor="#222831", linewidth=0.4, capsize=2,
            )
        ax.set_xticks(np.arange(len(bands_present)))
        ax.set_xticklabels(bands_present, rotation=20, ha="right")
        ax.set_ylabel("mean mse_total ± SE")
        ax.set_title(
            f"Forecast MSE per label ({partition}) — by class"
        )
        ax.legend(loc="best", frameon=True)
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(
            overlay_dir
            / f"frequency_band_{partition}_mse_by_band_overlay.pdf"
        )
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"per_class_breakdown[frequency_band/{partition}]: "
            f"bar plot failed: {exc}"
        )

    # Per-band scalar overlays so the user can compare class distributions
    # for one band at a time (filtered, not pooled across bands).
    for col in ("mse_total", "r2_total"):
        for band in bands_present:
            band_subsets = {
                lab: df.loc[df["band"] == band].copy()
                for lab, df in per_class.items()
            }
            band_subsets = {k: v for k, v in band_subsets.items() if not v.empty}
            if not band_subsets:
                continue
            _emit_overlay_for_metric(
                band_subsets, col,
                overlay_dir
                / f"frequency_band_{partition}_{band}_{col}_overlay.pdf",
            )

    return {
        "status": "ok",
        "n_classes": len(per_class),
        "n_bands": len(bands_present),
    }


def _process_frequency_band(
    output_root: Path, overlay_dir: Path,
) -> Dict[str, Any]:
    """Per-class breakdown of the frequency-band forecast outputs.

    Walks each of the four partition subdirectories
    (``clinical_4band``, ``clinical_7band``, ``by_kind``, ``by_octave``)
    produced by :func:`run_frequency_band_forecast_analysis` and emits
    per-class subset CSVs plus class-x-label overlays for every partition.
    """
    results: Dict[str, Any] = {"status": "missing"}
    per_partition: Dict[str, Dict[str, Any]] = {}
    found_any = False
    for partition in _FBF_PARTITION_DIRS:
        info = _process_frequency_band_partition(
            output_root, overlay_dir, partition,
        )
        per_partition[partition] = info
        if info.get("status") == "ok":
            found_any = True
    if found_any:
        results = {
            "status": "ok",
            "partitions": per_partition,
            "n_partitions": int(
                sum(
                    1 for v in per_partition.values()
                    if v.get("status") == "ok"
                )
            ),
        }
    else:
        results = {"status": "missing", "partitions": per_partition}
    return results


def _process_uplift(output_root: Path, overlay_dir: Path) -> Dict[str, Any]:
    """Per-class breakdown of uplift CSVs (already class-aware in plotting)."""
    src = _load_csv(output_root / "uplift" / "per_sample.csv")
    if src is None or "label" not in src.columns:
        return {"status": "missing"}

    per_class: Dict[int, pd.DataFrame] = {}
    for label_id in CLASS_NAMES:
        sub = _filter_by_label(src, label_id)
        if sub.empty:
            continue
        per_class[label_id] = sub
        out_dir = output_root / "per_class_breakdown" / CLASS_FOLDER[label_id]
        out_dir.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_dir / "uplift_per_sample.csv", index=False)
        try:
            plot_uplift_histogram(sub, out_dir / "uplift_histogram.pdf")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"per_class_breakdown[uplift]: plot failed for "
                f"class {CLASS_NAMES[label_id]}: {exc}"
            )

    for col in ("uplift_abs", "uplift_rel", "l_full", "l_base"):
        _emit_overlay_for_metric(per_class, col, overlay_dir / f"uplift_{col}_overlay.pdf")

    return {"status": "ok", "n_classes": len(per_class)}


def run_per_class_breakdown(
    output_root: Path,
    *,
    parts: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Post-process the existing pooled CSVs into per-class subfolders.

    The wrapper writes:

    - ``per_class_breakdown/class_healthy/...``
    - ``per_class_breakdown/class_acidosis/...``
    - ``per_class_breakdown/class_hie/...``
    - ``per_class_breakdown/class_overlay/...`` (cross-class overlay plots)

    Args:
        output_root: Root directory of the testing run (the same path as
            ``runner.output_dir``). Must contain at least one of the
            existing ``histograms/``, ``forecast_quality/``,
            ``residual_usage/``, ``attention/``, ``uplift/`` folders.
        parts: Optional subset of analyses to process. Defaults to all
            five (``histogram``, ``forecast``, ``residual``, ``attention``,
            ``uplift``).

    Returns:
        Dict mapping each processed analysis name to its summary status.
    """
    output_root = Path(output_root)
    overlay_dir = output_root / "per_class_breakdown" / "class_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    available_parts = {
        "histogram": _process_histogram,
        "forecast": _process_forecast_quality,
        "residual": _process_residual_usage,
        "attention": _process_attention,
        "uplift": _process_uplift,
        "frequency_band": _process_frequency_band,
    }
    if parts is None:
        parts = list(available_parts.keys())

    results: Dict[str, Any] = {}
    for name in parts:
        fn = available_parts.get(name)
        if fn is None:
            continue
        try:
            results[name] = fn(output_root, overlay_dir)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"per_class_breakdown[{name}] failed: {exc}")
            results[name] = {"status": "error", "error": str(exc)}

    logger.info(f"per_class_breakdown: {results}")
    return results
