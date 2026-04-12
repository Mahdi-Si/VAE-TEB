"""Per-sample qualitative diagnostics for the lag-attn v1 testing pipeline.

Replaces the legacy ``run_reconstruction_analysis`` and
``run_single_prediction_windows`` raw-FHR diagnostics with a single
consolidated entry point, :func:`run_sample_diagnostics`. For a handful
of samples it runs :func:`collect_predictions` and emits one
multi-row diagnostic PDF per sample via
:func:`plot_sample_lag_attn_diagnostic` plus a CSV of per-sample
metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.collectors import collect_predictions
from model.vae_teb_prediction.testing.plot_single_samples import (
    plot_sample_lag_attn_diagnostic,
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
            plot_sample_lag_attn_diagnostic(
                sample,
                out_path,
                warmup=runner.warmup_steps,
                horizon=runner.horizon,
            )
            plotted += 1
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to plot sample {guid}: {exc}")
            continue

        metrics_rows.append({
            "guid": guid,
            "epoch": epoch,
            "label": label,
            "out_path": str(out_path.name),
            **metrics,
        })

    summary_df = pd.DataFrame(metrics_rows)
    summary_csv = output_dir / "sample_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)

    logger.info(f"sample_diagnostics: plotted {plotted}/{len(samples)} samples → {output_dir}")
    return {
        "n_plotted": plotted,
        "summary_csv": str(summary_csv),
        "metrics": metrics_rows,
    }
