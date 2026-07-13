r"""Raw-waveform forecast analysis (S7-T02) -- the raw analogue of ``forecast_quality``.

The scattering pipeline scored the $87$-channel feature future per channel/band. The raw model
forecasts the future raw-FHR **waveform**, so this analysis instead reports waveform-level metrics
(VAF / MSE / SNR / $R^2$ / multi-scale low-pass, no channel split) and writes the raw plots from
:mod:`visualizers_raw`: the sample/anchor-averaged predicted-vs-true overlay, the per-horizon error
curve, the multi-scale low-pass bars, and the $(H \times T)$ forecast heatmap.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from model.vae_teb_prediction.model.model_raw.testing.base import TestRunner
from model.vae_teb_prediction.model.model_raw.testing.collectors import (
    collect_forecast_errors_per_horizon,
    resolve_fhr_up_denorm_stats,
)
from model.vae_teb_prediction.model.model_raw.testing.metrics import (
    compute_raw_forecast_metrics,
)
from model.vae_teb_prediction.model.model_raw.testing import visualizers_raw as vr


def run_raw_forecast_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = 500,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    r"""Score and plot the raw-FHR forecast; write ``raw_forecast/`` artefacts.

    Args:
        runner: Loaded raw :class:`TestRunner`.
        loader: Evaluation DataLoader.
        max_samples: Cap on samples consumed.
        output_dir: Destination; defaults to ``runner.ensure_dir("raw_forecast")``.

    Returns:
        A summary dict (mean raw metrics + written artefact paths). Never raises: a failed step is
        logged and reflected in the returned dict so the pipeline harness continues.
    """
    if output_dir is None:
        output_dir = runner.ensure_dir("raw_forecast")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        fhr_stats = resolve_fhr_up_denorm_stats(loader).get("fhr")
    except Exception:  # noqa: BLE001
        fhr_stats = None
    warmup, horizon = int(runner.warmup_steps), int(runner.horizon)

    # -- Accumulate per-sample raw metrics + stack the forecast tensors for the plots.
    mse_totals, vafs, r2s, lp_mses = [], [], [], []
    mu_chunks, x_chunks, lv_chunks = [], [], []
    n = 0
    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            outputs = runner.forward(batch)
            x_plus = runner.build_future_target(batch)
            m = compute_raw_forecast_metrics(outputs["mu_full"], x_plus, warmup, horizon)
            mse_totals.append(m["raw_mse"].detach().cpu().numpy())
            vafs.append(m["raw_vaf"].detach().cpu().numpy())
            r2s.append(m["raw_r2"].detach().cpu().numpy())
            lp_mses.append(m["raw_lowpass_mse"].detach().cpu().numpy())
            mu_chunks.append(outputs["mu_full"].detach().cpu().numpy())
            x_chunks.append(x_plus.detach().cpu().numpy())
            lv_chunks.append(outputs["logvar_full"].detach().cpu().numpy())
            n += int(outputs["mu_full"].shape[0])
            if max_samples is not None and n >= max_samples:
                break

    if not mu_chunks:
        logger.warning("raw_forecast: no samples collected.")
        return {"n_samples": 0}

    mu = np.concatenate(mu_chunks, axis=0)     # (N, T, H, R)
    x = np.concatenate(x_chunks, axis=0)       # (N, T_valid, H, R)
    lv = np.concatenate(lv_chunks, axis=0)     # (N, T, H, R)

    summary: Dict[str, Any] = {
        "n_samples": int(mu.shape[0]),
        "raw_mse_mean": float(np.concatenate(mse_totals).mean()),
        "raw_vaf_mean": float(np.concatenate(vafs).mean()),
        "raw_r2_mean": float(np.concatenate(r2s).mean()),
        "raw_lowpass_mse_mean": float(np.concatenate(lp_mses).mean()),
    }

    # -- Per-horizon error table + curve (no st/ph split).
    try:
        per_h = collect_forecast_errors_per_horizon(runner, loader, max_samples=max_samples)
        per_h.to_csv(output_dir / "per_horizon_error.csv", index=False)
        mse_by_h = per_h.groupby("h")["mse_step"].mean().sort_index().to_numpy()
        vr.plot_raw_per_horizon_error(mse_by_h, output_dir / "per_horizon_error.pdf",
                                      r_substeps=int(runner.model.geometry.r))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"raw_forecast: per-horizon step failed ({exc})")

    # -- Averaged overlay + low-pass bars + (H x T) heatmap.
    for label, fn in (
        ("overlay", lambda: vr.plot_raw_forecast_overlay_averaged(
            mu, x, output_dir / "forecast_overlay_avg.pdf",
            warmup=warmup, horizon=horizon, fhr_stats=fhr_stats, logvar=lv)),
        ("lowpass", lambda: vr.plot_raw_lowpass_error(
            mu, x, output_dir / "lowpass_error.pdf", warmup=warmup, horizon=horizon)),
        ("heatmap", lambda: vr.plot_raw_forecast_heatmap(
            mu, output_dir / "forecast_heatmap.pdf", warmup=warmup, horizon=horizon)),
    ):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"raw_forecast: {label} plot failed ({exc})")

    pd.DataFrame([summary]).to_csv(output_dir / "raw_metrics.csv", index=False)
    logger.info(
        "raw_forecast: {} samples, raw_mse={:.4f}, VAF={:.4f}, R2={:.4f}",
        summary["n_samples"], summary["raw_mse_mean"], summary["raw_vaf_mean"],
        summary["raw_r2_mean"],
    )
    return summary
