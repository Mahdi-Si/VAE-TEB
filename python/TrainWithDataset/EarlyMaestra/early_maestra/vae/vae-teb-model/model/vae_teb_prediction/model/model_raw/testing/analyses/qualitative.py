r"""Per-sample qualitative diagnostics for the raw ``SeqVaeRawV4`` pipeline.

For a handful of samples this runs :func:`collect_predictions` and emits, per sample, a
**raw-FHR forecast overlay** (predicted vs true future waveform with the $\pm 2\sigma$ predictive
band, in denormalized bpm) via :func:`visualizers_raw.plot_raw_forecast_overlay`, plus a CSV of
per-sample metrics. This replaces the scattering-domain feature-heatmap diagnostic
(``plot_sample_lag_attn_diagnostic``), which has no meaning for a raw-waveform target (S7-T01,
"adapt to raw FHR").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from model.vae_teb_prediction.model.model_raw.testing.base import TestRunner
from model.vae_teb_prediction.model.model_raw.testing.collectors import (
    collect_predictions,
    resolve_fhr_up_denorm_stats,
)
from model.vae_teb_prediction.model.model_raw.testing.visualizers_raw import (
    plot_raw_forecast_overlay,
)


def run_sample_diagnostics(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 10,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Emit per-sample diagnostic PDFs and a metric summary CSV.

    Args:
        runner: Loaded :class:`TestRunner`.
        loader: PyTorch DataLoader.
        max_samples: Number of samples to diagnose.
        output_dir: Optional override for the output directory (defaults
            to ``runner.ensure_dir("samples_diag")``).

    Returns:
        Dict with keys ``n_plotted``, ``summary_csv`` (path), and
        ``metrics`` (list of per-sample metric dicts).
    """
    if max_samples <= 0:
        logger.info("sample_diagnostics: skipped (max_samples <= 0)")
        return {"n_plotted": 0, "summary_csv": None, "metrics": []}

    logger.info(f"Collecting {max_samples} samples for per-sample diagnostics...")
    samples = collect_predictions(runner, loader, max_samples=max_samples)
    if not samples:
        logger.warning("sample_diagnostics: no samples collected.")
        return {"n_plotted": 0, "summary_csv": None, "metrics": []}

    if output_dir is None:
        output_dir = runner.ensure_dir("samples_diag")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Denormalisation stats so the overlay reads in bpm; ``None`` -> normalized units.
    try:
        fhr_stats = resolve_fhr_up_denorm_stats(loader).get("fhr")
    except Exception:  # noqa: BLE001 - denorm is best-effort; fall back to normalized units
        fhr_stats = None
    warmup = int(runner.warmup_steps)

    # Synthetic-TE provenance columns are added only when at least one sample carries a
    # true TE (S7-T06): this keeps real-data ``sample_metrics.csv`` byte-for-byte the same
    # while synthetic v2 runs gain the TE / lag columns.
    has_te = any(s.get("te_true") is not None for s in samples)

    metrics_rows: List[Dict[str, Any]] = []
    plotted = 0
    for i, sample in enumerate(samples):
        guid = sample.get("guid", f"sample_{i}")
        epoch = sample.get("epoch")
        label = sample.get("label")
        metrics = dict(sample.get("metrics", {}) or {})

        safe_guid = str(guid).replace("/", "_") if guid is not None else f"sample_{i}"
        epoch_str = f"ep{int(epoch):+d}" if epoch is not None else f"idx{i}"
        out_path = output_dir / f"{safe_guid}_{epoch_str}.pdf"

        try:
            # ``collect_predictions`` stores raw-shaped decoder tensors: mu_full/logvar_full over the
            # full T anchor axis, y_plus over the T_valid anchors. Pick a mid valid anchor and
            # overlay the predicted vs true future raw-FHR waveform (denormalized bpm).
            mu_full = np.asarray(sample["mu_full"])          # (T, H, R)
            y_plus = np.asarray(sample["y_plus"])            # (T_valid, H, R)
            logvar_full = sample.get("logvar_full")
            t_valid = y_plus.shape[0]
            anchor = max(warmup, min(warmup + (t_valid - warmup) // 2, t_valid - 1))
            lv_anchor = (
                np.asarray(logvar_full)[anchor] if logvar_full is not None else None
            )
            title = f"{safe_guid} {epoch_str} (anchor {anchor})"
            plot_raw_forecast_overlay(
                mu_full[anchor],
                y_plus[anchor],
                out_path,
                fhr_stats=fhr_stats,
                logvar_anchor=lv_anchor,
                title=title,
            )
            plotted += 1
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to plot sample {guid}: {exc}")
            continue

        row: Dict[str, Any] = {
            "guid": guid,
            "epoch": epoch,
            "label": label,
            "out_path": str(out_path.name),
            **metrics,
        }
        if has_te:
            row.update({
                "te_true": sample.get("te_true"),
                "te_scat": sample.get("te_scat"),
                "te_raw": sample.get("te_raw"),
                "frac_phi": sample.get("frac_phi"),
                "sample_delay": sample.get("sample_delay"),
                "cell_id": sample.get("cell_id"),
            })
        metrics_rows.append(row)

    summary_df = pd.DataFrame(metrics_rows)
    summary_csv = output_dir / "sample_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)

    logger.info(f"sample_diagnostics: plotted {plotted}/{len(samples)} samples → {output_dir}")
    return {
        "n_plotted": plotted,
        "summary_csv": str(summary_csv),
        "metrics": metrics_rows,
    }
