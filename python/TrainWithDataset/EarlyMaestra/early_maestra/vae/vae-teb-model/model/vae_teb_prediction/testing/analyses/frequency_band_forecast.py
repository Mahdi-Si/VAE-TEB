"""Frequency-band-stratified forecast quality analysis for lag-attn v1.

Re-evaluates the model's full feature forecast against the 87-channel
future target, but slices the squared-error tensor along the channel
axis according to the clinical frequency-band partition built by
``model.vae_teb_prediction.testing.band_partition.build_band_partition``.

For each band the analysis reports per-sample MSE, R², per-horizon-step
MSE, and per-anchor-position MSE. Class-stratified variants are emitted
when at least two clinical classes are present in the test set.

Outputs (under ``frequency_band_forecast/`` next to ``forecast_quality/``):

- ``band_partition.json`` + ``band_channel_map.csv`` — channel mapping.
- ``per_sample.csv`` — long-format ``(sample, band)`` rows.
- ``per_horizon.csv`` — long-format ``(sample, band, h)`` rows.
- ``per_anchor.csv`` — long-format ``(sample, band, t)`` rows.
- ``summary.json`` — pooled / per-class means.
- ``band_mse_violin.pdf``, ``band_r2_violin.pdf`` — pooled violins.
- ``band_horizon_error.pdf``, ``band_anchor_error.pdf`` — pooled ribbons.
- ``band_horizon_error_by_class.pdf``, ``band_anchor_error_by_class.pdf``,
  ``band_mse_violin_by_class.pdf``, ``band_r2_violin_by_class.pdf`` —
  class-aware variants when ``len(classes) >= 2``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from model.vae_teb_prediction.testing.band_partition import (
    BandPartition,
    build_band_partition,
)
from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.collectors import (
    _extract_epoch,
    _extract_guid,
    _extract_label,
)
from model.vae_teb_prediction.testing.metrics import (
    compute_band_forecast_metrics,
)
from model.vae_teb_prediction.testing.visualizers import (
    plot_band_anchor_error,
    plot_band_anchor_error_by_class,
    plot_band_horizon_error,
    plot_band_horizon_error_by_class,
    plot_band_violin,
    plot_band_violin_by_class,
    unique_labels_in,
)


def run_frequency_band_forecast_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 500,
    output_dir: Optional[Path] = None,
    *,
    fhr_phase_min_freq: float = 0.006,
    fs: float = 4.0,
    decim_step_seconds: float = 4.0,
) -> Dict[str, Any]:
    """Run frequency-band-stratified feature forecast evaluation.

    Args:
        runner: Loaded :class:`TestRunner`.
        loader: PyTorch DataLoader.
        max_samples: Maximum samples to process. ``<= 0`` skips this
            analysis entirely.
        output_dir: Optional override (defaults to
            ``runner.ensure_dir("frequency_band_forecast")``).
        fhr_phase_min_freq: Frequency floor passed to
            :func:`build_band_partition` — must match the value the
            HDF5 dataset was built with (0.006 Hz for v1).
        fs: Sampling frequency used to convert wavelet xi to Hz
            (4 Hz for the CTG dataset).
        decim_step_seconds: Physical duration of one decimated sequence
            step, used to label the anchor x-axis in minutes (4 s for
            the v1 16x decimation at 4 Hz).

    Returns:
        Dict with the per-band summary statistics, output file paths,
        and the band partition counts. Empty dict when ``max_samples
        <= 0`` or when no samples were processable.
    """
    if max_samples <= 0:
        logger.info("frequency_band_forecast: skipped (max_samples <= 0)")
        return {}

    if output_dir is None:
        output_dir = runner.ensure_dir("frequency_band_forecast")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the channel-to-band partition once. Channel counts default
    # to the v1 dataset schema (43 + 44 = 87).
    partition: BandPartition = build_band_partition(
        fhr_phase_min_freq=fhr_phase_min_freq, fs=fs,
    )
    json_path, csv_path = partition.write(output_dir)
    logger.info(
        f"frequency_band_forecast: band partition written to "
        f"{json_path.name} / {csv_path.name}"
    )

    nonempty_bands = partition.nonempty_bands()
    if not nonempty_bands:
        logger.warning("frequency_band_forecast: every band is empty — aborting")
        return {"error": "empty band partition"}

    # Mapping fed to compute_band_forecast_metrics.
    band_combined_idx = {
        b: partition.combined_idx[b] for b in nonempty_bands
    }
    n_channels_by_band = {
        b: int(partition.combined_idx[b].size) for b in nonempty_bands
    }

    # ----- Per-sample / per-horizon / per-anchor accumulation ----------
    per_sample_rows: List[Dict[str, Any]] = []
    per_horizon_rows: List[Dict[str, Any]] = []
    per_anchor_rows: List[Dict[str, Any]] = []
    processed = 0

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            outputs = runner.forward(batch)
            y_plus = runner.build_future_target(batch)
            band_metrics = compute_band_forecast_metrics(
                outputs["mu_full"], y_plus,
                runner.warmup_steps, runner.horizon,
                band_combined_idx, return_per_anchor=True,
            )

            batch_size = int(outputs["mu_full"].size(0))
            warmup = int(runner.warmup_steps)

            # Cache per-sample metadata once per batch.
            sample_meta = []
            for idx in range(batch_size):
                if max_samples and (processed + idx) >= max_samples:
                    break
                sample_meta.append({
                    "guid": _extract_guid(batch, idx),
                    "epoch": _extract_epoch(batch, idx),
                    "label": _extract_label(batch, idx),
                })
            n_kept = len(sample_meta)
            if n_kept == 0:
                break

            # Convert per-band tensors to numpy in one go (cheap on
            # small B and a handful of bands).
            for band in nonempty_bands:
                m = band_metrics[band]
                mse_total = m["mse_total"].detach().cpu().numpy()
                r2_total = m["r2_total"].detach().cpu().numpy()
                mse_per_horizon = m["mse_per_horizon"].detach().cpu().numpy()
                mse_per_anchor = m["mse_per_anchor"].detach().cpu().numpy() \
                    if "mse_per_anchor" in m else None
                n_ch = n_channels_by_band[band]

                for idx_local, meta in enumerate(sample_meta):
                    if idx_local >= mse_total.shape[0]:
                        break
                    per_sample_rows.append({
                        **meta,
                        "band": band,
                        "n_channels": n_ch,
                        "mse_total": float(mse_total[idx_local]),
                        "r2_total": float(r2_total[idx_local]),
                    })
                    for h in range(mse_per_horizon.shape[1]):
                        per_horizon_rows.append({
                            **meta,
                            "band": band,
                            "h": int(h),
                            "mse": float(mse_per_horizon[idx_local, h]),
                        })
                    if mse_per_anchor is not None:
                        # Anchor index t starts at warmup (the global
                        # absolute anchor position) and runs to T-H_d.
                        # Storing the absolute index lets the plot label
                        # the x-axis directly in minutes.
                        for t_local in range(mse_per_anchor.shape[1]):
                            per_anchor_rows.append({
                                **meta,
                                "band": band,
                                "t": int(warmup + t_local),
                                "mse": float(mse_per_anchor[idx_local, t_local]),
                            })

            processed += n_kept
            if max_samples and processed >= max_samples:
                break

    if not per_sample_rows:
        logger.warning(
            "frequency_band_forecast: no samples produced metrics "
            "(empty loader or all samples exhausted on warmup/horizon)"
        )
        return {
            "error": "no samples",
            "band_partition_json": str(json_path),
            "band_channel_map_csv": str(csv_path),
        }

    per_sample_df = pd.DataFrame(per_sample_rows)
    per_horizon_df = pd.DataFrame(per_horizon_rows)
    per_anchor_df = pd.DataFrame(per_anchor_rows)

    per_sample_csv = output_dir / "per_sample.csv"
    per_horizon_csv = output_dir / "per_horizon.csv"
    per_anchor_csv = output_dir / "per_anchor.csv"
    per_sample_df.to_csv(per_sample_csv, index=False)
    per_horizon_df.to_csv(per_horizon_csv, index=False)
    per_anchor_df.to_csv(per_anchor_csv, index=False)

    # ----- Pooled plots ------------------------------------------------
    try:
        plot_band_violin(
            per_sample_df, "mse_total",
            output_dir / "band_mse_violin.pdf",
            title="Per-sample forecast MSE per frequency band",
            n_channels_by_band=n_channels_by_band,
        )
        plot_band_violin(
            per_sample_df, "r2_total",
            output_dir / "band_r2_violin.pdf",
            title="Per-sample forecast R² per frequency band",
            n_channels_by_band=n_channels_by_band,
        )
        plot_band_horizon_error(
            per_horizon_df, output_dir / "band_horizon_error.pdf",
            value_col="mse",
        )
        plot_band_anchor_error(
            per_anchor_df, output_dir / "band_anchor_error.pdf",
            value_col="mse", decim_step_seconds=decim_step_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"frequency_band_forecast: pooled plot failed: {exc}")

    # ----- Per-class plots --------------------------------------------
    classes = unique_labels_in(per_sample_df.get("label"))
    if len(classes) >= 2:
        try:
            plot_band_violin_by_class(
                per_sample_df, "mse_total",
                output_dir / "band_mse_violin_by_class.pdf",
                title="Per-sample forecast MSE per band — by class",
            )
            plot_band_violin_by_class(
                per_sample_df, "r2_total",
                output_dir / "band_r2_violin_by_class.pdf",
                title="Per-sample forecast R² per band — by class",
            )
            plot_band_horizon_error_by_class(
                per_horizon_df,
                output_dir / "band_horizon_error_by_class.pdf",
                value_col="mse",
            )
            plot_band_anchor_error_by_class(
                per_anchor_df,
                output_dir / "band_anchor_error_by_class.pdf",
                value_col="mse",
                decim_step_seconds=decim_step_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"frequency_band_forecast: by-class plot failed: {exc}")

    # ----- Summary -----------------------------------------------------
    summary_per_band: Dict[str, Dict[str, Any]] = {}
    for band in nonempty_bands:
        sub = per_sample_df[per_sample_df["band"] == band]
        if sub.empty:
            continue
        summary_per_band[band] = {
            "n_samples": int(len(sub)),
            "n_channels": int(n_channels_by_band[band]),
            "mean_mse": float(np.nanmean(sub["mse_total"].to_numpy())),
            "median_mse": float(np.nanmedian(sub["mse_total"].to_numpy())),
            "mean_r2": float(np.nanmean(sub["r2_total"].to_numpy())),
            "median_r2": float(np.nanmedian(sub["r2_total"].to_numpy())),
        }

    summary_per_band_class: Dict[str, Dict[str, Any]] = {}
    for lab in classes:
        sub_lab = per_sample_df[per_sample_df["label"] == lab]
        if sub_lab.empty:
            continue
        per_band_lab = {}
        for band in nonempty_bands:
            sub = sub_lab[sub_lab["band"] == band]
            if sub.empty:
                continue
            per_band_lab[band] = {
                "n_samples": int(len(sub)),
                "mean_mse": float(np.nanmean(sub["mse_total"].to_numpy())),
                "mean_r2": float(np.nanmean(sub["r2_total"].to_numpy())),
            }
        summary_per_band_class[str(int(lab))] = per_band_lab

    summary = {
        "n_unique_samples": int(per_sample_df[["guid", "epoch"]]
                                .drop_duplicates().shape[0]),
        "n_records_per_sample": int(per_sample_df.shape[0]),
        "bands": list(nonempty_bands),
        "n_channels_per_band": n_channels_by_band,
        "by_band": summary_per_band,
        "by_band_and_class": summary_per_band_class,
        "fhr_phase_min_freq": float(fhr_phase_min_freq),
        "fs": float(fs),
        "decim_step_seconds": float(decim_step_seconds),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(
        "frequency_band_forecast: "
        + ", ".join(
            f"{b}={summary_per_band[b]['mean_mse']:.4f}"
            for b in nonempty_bands if b in summary_per_band
        )
    )

    summary["per_sample_csv"] = str(per_sample_csv)
    summary["per_horizon_csv"] = str(per_horizon_csv)
    summary["per_anchor_csv"] = str(per_anchor_csv)
    summary["band_partition_json"] = str(json_path)
    summary["band_channel_map_csv"] = str(csv_path)
    summary["summary_json"] = str(summary_path)
    return summary


__all__ = ["run_frequency_band_forecast_analysis"]
