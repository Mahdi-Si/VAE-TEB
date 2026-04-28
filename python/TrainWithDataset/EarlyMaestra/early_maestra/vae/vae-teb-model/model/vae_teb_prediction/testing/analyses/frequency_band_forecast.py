"""Frequency-band-stratified forecast quality analysis for lag-attn v1.

Re-evaluates the model's full feature forecast against the 87-channel
future target, slicing the squared-error tensor along the channel axis
according to **four parallel partitions** built by
``model.vae_teb_prediction.testing.band_partition.build_band_partition``:

* ``clinical_4band``  — historic clinical bands (slow_baseline, decel,
  variability, beat_to_beat).
* ``clinical_7band``  — finer 7-band split derived from the actual
  scattering frequencies (baseline, early/late decel, lf/mf var,
  beat_to_beat, nyquist_edge).
* ``by_kind``         — coefficient kind (st_S0, st_S1, ph_diag, ph_h2,
  ph_h3, ph_other) — answers "is the model better at envelope (S1)
  than rhythm phase stability (ph_diag)?".
* ``by_octave``       — per-octave bins from the kymatio J-octave bank.

In addition the analysis emits a per-channel CSV with mean forecast MSE
and R² per (sample, channel), letting users plot MSE vs centre frequency
and MSE vs phase-harmonic ratio without re-running inference.

Outputs (under ``frequency_band_forecast/``):

* Top level (back-compat duplicates of ``clinical_4band/``):
  ``per_sample.csv``, ``per_horizon.csv``, ``per_anchor.csv``,
  ``band_partition.json``, ``band_channel_map.csv``, ``summary.json``.
* ``clinical_4band/``, ``clinical_7band/``, ``by_kind/``,
  ``by_octave/`` — each carries the same set of long-format CSVs and
  pooled / by-class plots as the legacy top-level directory.
* ``per_channel/`` — ``per_channel_forecast.csv`` plus
  ``mse_vs_freq.pdf`` and ``mse_vs_harmonic_ratio.pdf``.

Class-stratified variants (``*_by_class.pdf``) are emitted inside each
partition subdirectory when at least two clinical classes are present.
The downstream ``per_class_breakdown`` post-processor walks every
partition subdirectory to emit cross-class overlays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    compute_per_channel_forecast_metrics,
)
from model.vae_teb_prediction.testing.visualizers import (
    plot_band_anchor_error,
    plot_band_anchor_error_by_class,
    plot_band_horizon_error,
    plot_band_horizon_error_by_class,
    plot_band_violin,
    plot_band_violin_by_class,
    plot_per_channel_mse_vs_freq,
    plot_phase_harmonic_mse,
    unique_labels_in,
)

# Partition names processed in canonical order. The first one is also
# duplicated at the top level for backwards compatibility with consumers
# that still read ``frequency_band_forecast/per_sample.csv`` directly.
_PARTITION_NAMES: Tuple[str, ...] = (
    "clinical_4band",
    "clinical_7band",
    "by_kind",
    "by_octave",
)
_LEGACY_DUPLICATED: str = "clinical_4band"


def _safe_plot(name: str, fn, *args, **kwargs) -> None:
    """Invoke ``fn`` defensively so one bad plot doesn't abort the run."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"frequency_band_forecast plot {name!r} failed: {exc}")


def _emit_partition_plots(
    per_sample_df: pd.DataFrame,
    per_horizon_df: pd.DataFrame,
    per_anchor_df: pd.DataFrame,
    output_dir: Path,
    *,
    n_channels_by_label: Dict[str, int],
    decim_step_seconds: float,
    title_suffix: str,
) -> None:
    """Emit the standard violin / horizon / anchor plots for one partition."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _safe_plot(
        f"violin_mse_{title_suffix}", plot_band_violin,
        per_sample_df, "mse_total",
        output_dir / "band_mse_violin.pdf",
        title=f"Per-sample forecast MSE — {title_suffix}",
        n_channels_by_band=n_channels_by_label,
    )
    _safe_plot(
        f"violin_r2_{title_suffix}", plot_band_violin,
        per_sample_df, "r2_total",
        output_dir / "band_r2_violin.pdf",
        title=f"Per-sample forecast R² — {title_suffix}",
        n_channels_by_band=n_channels_by_label,
    )
    _safe_plot(
        f"horizon_{title_suffix}", plot_band_horizon_error,
        per_horizon_df, output_dir / "band_horizon_error.pdf",
        value_col="mse",
    )
    _safe_plot(
        f"anchor_{title_suffix}", plot_band_anchor_error,
        per_anchor_df, output_dir / "band_anchor_error.pdf",
        value_col="mse", decim_step_seconds=decim_step_seconds,
    )

    classes = unique_labels_in(per_sample_df.get("label"))
    if len(classes) >= 2:
        _safe_plot(
            f"violin_mse_by_class_{title_suffix}",
            plot_band_violin_by_class,
            per_sample_df, "mse_total",
            output_dir / "band_mse_violin_by_class.pdf",
            title=f"Per-sample forecast MSE by class — {title_suffix}",
        )
        _safe_plot(
            f"violin_r2_by_class_{title_suffix}",
            plot_band_violin_by_class,
            per_sample_df, "r2_total",
            output_dir / "band_r2_violin_by_class.pdf",
            title=f"Per-sample forecast R² by class — {title_suffix}",
        )
        _safe_plot(
            f"horizon_by_class_{title_suffix}",
            plot_band_horizon_error_by_class,
            per_horizon_df,
            output_dir / "band_horizon_error_by_class.pdf",
            value_col="mse",
        )
        _safe_plot(
            f"anchor_by_class_{title_suffix}",
            plot_band_anchor_error_by_class,
            per_anchor_df,
            output_dir / "band_anchor_error_by_class.pdf",
            value_col="mse",
            decim_step_seconds=decim_step_seconds,
        )


def _summarise_partition(
    per_sample_df: pd.DataFrame,
    nonempty_labels: Tuple[str, ...],
    n_channels_by_label: Dict[str, int],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Return ``(by_label, by_label_and_class)`` summary dicts."""
    by_label: Dict[str, Dict[str, Any]] = {}
    for label in nonempty_labels:
        sub = per_sample_df[per_sample_df["band"] == label]
        if sub.empty:
            continue
        by_label[label] = {
            "n_samples": int(len(sub)),
            "n_channels": int(n_channels_by_label.get(label, 0)),
            "mean_mse": float(np.nanmean(sub["mse_total"].to_numpy())),
            "median_mse": float(np.nanmedian(sub["mse_total"].to_numpy())),
            "mean_r2": float(np.nanmean(sub["r2_total"].to_numpy())),
            "median_r2": float(np.nanmedian(sub["r2_total"].to_numpy())),
        }

    by_label_and_class: Dict[str, Dict[str, Any]] = {}
    classes = unique_labels_in(per_sample_df.get("label"))
    for cls in classes:
        sub_cls = per_sample_df[per_sample_df["label"] == cls]
        if sub_cls.empty:
            continue
        per_label_cls: Dict[str, Dict[str, Any]] = {}
        for label in nonempty_labels:
            sub = sub_cls[sub_cls["band"] == label]
            if sub.empty:
                continue
            per_label_cls[label] = {
                "n_samples": int(len(sub)),
                "mean_mse": float(np.nanmean(sub["mse_total"].to_numpy())),
                "mean_r2": float(np.nanmean(sub["r2_total"].to_numpy())),
            }
        by_label_and_class[str(int(cls))] = per_label_cls

    return by_label, by_label_and_class


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

    Drives four parallel partitions in a single inference pass and emits
    one CSV+plot triplet per partition under
    ``frequency_band_forecast/<partition>/``. Adds a per-channel CSV +
    diagnostic scatter plots under ``frequency_band_forecast/per_channel/``.
    For backwards compatibility, the ``clinical_4band`` outputs are also
    duplicated at the top of ``frequency_band_forecast/``.

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
        Dict with per-partition summary statistics, the per-channel
        summary, and output file paths. Empty dict when
        ``max_samples <= 0``.
    """
    if max_samples <= 0:
        logger.info("frequency_band_forecast: skipped (max_samples <= 0)")
        return {}

    if output_dir is None:
        output_dir = runner.ensure_dir("frequency_band_forecast")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the channel-to-band partition once. Channel counts default to
    # the v1 dataset schema (43 + 44 = 87).
    partition: BandPartition = build_band_partition(
        fhr_phase_min_freq=fhr_phase_min_freq, fs=fs,
    )
    json_path, csv_path = partition.write(output_dir)
    logger.info(
        f"frequency_band_forecast: band partition written to "
        f"{json_path.name} / {csv_path.name}"
    )

    # Collect every nonempty partition. We always proceed to inference
    # even if some partitions only have one nonempty label (e.g. by_kind
    # often misses ph_h3); the empty labels are silently omitted from
    # both the loop and the summary so plots stay clean.
    nonempty_by_partition: Dict[str, Tuple[str, ...]] = {}
    partition_idx_by_partition: Dict[str, Dict[str, np.ndarray]] = {}
    for pname in _PARTITION_NAMES:
        non_empty = partition.nonempty_partition(pname)
        nonempty_by_partition[pname] = non_empty
        partition_idx_by_partition[pname] = {
            label: partition.partition_idx(pname)[label]
            for label in non_empty
        }

    if not nonempty_by_partition[_LEGACY_DUPLICATED]:
        logger.warning(
            "frequency_band_forecast: every clinical_4band band is "
            "empty — aborting"
        )
        return {"error": "empty band partition"}

    # ----- Per-sample / per-horizon / per-anchor accumulation ---------
    # Long-format row buffers keyed by partition name so we can write
    # one CSV per partition at the end.
    per_sample_rows: Dict[str, List[Dict[str, Any]]] = {
        p: [] for p in _PARTITION_NAMES
    }
    per_horizon_rows: Dict[str, List[Dict[str, Any]]] = {
        p: [] for p in _PARTITION_NAMES
    }
    per_anchor_rows: Dict[str, List[Dict[str, Any]]] = {
        p: [] for p in _PARTITION_NAMES
    }
    per_channel_rows: List[Dict[str, Any]] = []
    processed = 0

    # Reused metadata per channel (kind / band / refined_band / octave /
    # primary freq / harmonic ratio) — looked up once and joined into
    # every per-channel record.
    channel_meta_df: pd.DataFrame = partition.channel_metadata
    channel_meta_lookup: Dict[int, Dict[str, Any]] = {
        int(row["channel"]): {
            "kind": row["kind"],
            "band": row["band"],
            "refined_band": row["refined_band"],
            "octave": row["octave"],
            "freq_hz_primary": float(row["freq_hz_primary"]),
            "freq_hz_secondary": float(row["freq_hz_secondary"]),
            "harmonic_ratio": float(row["harmonic_ratio"]),
        }
        for _, row in channel_meta_df.iterrows()
    }

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            outputs = runner.forward(batch)
            y_plus = runner.build_future_target(batch)
            warmup = int(runner.warmup_steps)

            # Per-partition band metrics.
            partition_metrics: Dict[
                str, Dict[str, Dict[str, Any]],
            ] = {}
            for pname in _PARTITION_NAMES:
                idx = partition_idx_by_partition[pname]
                if not idx:
                    partition_metrics[pname] = {}
                    continue
                partition_metrics[pname] = compute_band_forecast_metrics(
                    outputs["mu_full"], y_plus,
                    runner.warmup_steps, runner.horizon,
                    partition_idx=idx, return_per_anchor=True,
                )

            # Per-channel metrics (single call, reused across plots / CSV).
            per_channel = compute_per_channel_forecast_metrics(
                outputs["mu_full"], y_plus,
                runner.warmup_steps, runner.horizon,
            )
            per_channel_mse = per_channel["mse_per_channel"].detach().cpu().numpy()
            per_channel_r2 = per_channel["r2_per_channel"].detach().cpu().numpy()

            batch_size = int(outputs["mu_full"].size(0))
            sample_meta: List[Dict[str, Any]] = []
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

            # -------- Per-partition: build long-format rows --------
            for pname in _PARTITION_NAMES:
                non_empty = nonempty_by_partition[pname]
                if not non_empty:
                    continue
                n_channels_by_label = {
                    label: int(
                        partition_idx_by_partition[pname][label].size
                    ) for label in non_empty
                }
                bucket_metrics = partition_metrics[pname]
                for label in non_empty:
                    if label not in bucket_metrics:
                        continue
                    m = bucket_metrics[label]
                    mse_total = m["mse_total"].detach().cpu().numpy()
                    r2_total = m["r2_total"].detach().cpu().numpy()
                    mse_per_horizon = m["mse_per_horizon"].detach().cpu().numpy()
                    mse_per_anchor = (
                        m["mse_per_anchor"].detach().cpu().numpy()
                        if "mse_per_anchor" in m else None
                    )
                    n_ch = n_channels_by_label[label]

                    for idx_local, meta in enumerate(sample_meta):
                        if idx_local >= mse_total.shape[0]:
                            break
                        per_sample_rows[pname].append({
                            **meta,
                            "band": label,
                            "n_channels": n_ch,
                            "mse_total": float(mse_total[idx_local]),
                            "r2_total": float(r2_total[idx_local]),
                        })
                        for h in range(mse_per_horizon.shape[1]):
                            per_horizon_rows[pname].append({
                                **meta,
                                "band": label,
                                "h": int(h),
                                "mse": float(mse_per_horizon[idx_local, h]),
                            })
                        if mse_per_anchor is not None:
                            for t_local in range(mse_per_anchor.shape[1]):
                                per_anchor_rows[pname].append({
                                    **meta,
                                    "band": label,
                                    "t": int(warmup + t_local),
                                    "mse": float(
                                        mse_per_anchor[idx_local, t_local]
                                    ),
                                })

            # -------- Per-channel rows --------
            n_channels = int(per_channel_mse.shape[1])
            for idx_local, meta in enumerate(sample_meta):
                if idx_local >= per_channel_mse.shape[0]:
                    break
                for ch in range(n_channels):
                    info = channel_meta_lookup.get(ch, {})
                    per_channel_rows.append({
                        **meta,
                        "channel": int(ch),
                        "kind": info.get("kind"),
                        "band": info.get("band"),
                        "refined_band": info.get("refined_band"),
                        "octave": info.get("octave"),
                        "freq_hz_primary": info.get("freq_hz_primary"),
                        "freq_hz_secondary": info.get("freq_hz_secondary"),
                        "harmonic_ratio": info.get("harmonic_ratio"),
                        "mse_total": float(per_channel_mse[idx_local, ch]),
                        "r2_total": float(per_channel_r2[idx_local, ch]),
                    })

            processed += n_kept
            if max_samples and processed >= max_samples:
                break

    if not per_sample_rows[_LEGACY_DUPLICATED]:
        logger.warning(
            "frequency_band_forecast: no samples produced metrics "
            "(empty loader or all samples exhausted on warmup/horizon)"
        )
        return {
            "error": "no samples",
            "band_partition_json": str(json_path),
            "band_channel_map_csv": str(csv_path),
        }

    # ----- Per-partition CSVs + plots ---------------------------------
    per_partition_summary: Dict[str, Dict[str, Any]] = {}
    per_partition_paths: Dict[str, Dict[str, str]] = {}

    for pname in _PARTITION_NAMES:
        rows_s = per_sample_rows[pname]
        if not rows_s:
            continue
        sub_dir = output_dir / pname
        sub_dir.mkdir(parents=True, exist_ok=True)

        per_sample_df = pd.DataFrame(rows_s)
        per_horizon_df = pd.DataFrame(per_horizon_rows[pname])
        per_anchor_df = pd.DataFrame(per_anchor_rows[pname])

        per_sample_csv = sub_dir / "per_sample.csv"
        per_horizon_csv = sub_dir / "per_horizon.csv"
        per_anchor_csv = sub_dir / "per_anchor.csv"
        per_sample_df.to_csv(per_sample_csv, index=False)
        per_horizon_df.to_csv(per_horizon_csv, index=False)
        per_anchor_df.to_csv(per_anchor_csv, index=False)

        non_empty = nonempty_by_partition[pname]
        n_channels_by_label = {
            label: int(partition_idx_by_partition[pname][label].size)
            for label in non_empty
        }
        _emit_partition_plots(
            per_sample_df, per_horizon_df, per_anchor_df, sub_dir,
            n_channels_by_label=n_channels_by_label,
            decim_step_seconds=decim_step_seconds,
            title_suffix=pname,
        )

        by_label, by_label_and_class = _summarise_partition(
            per_sample_df, non_empty, n_channels_by_label,
        )
        per_partition_summary[pname] = {
            "by_band": by_label,
            "by_band_and_class": by_label_and_class,
            "n_unique_samples": int(
                per_sample_df[["guid", "epoch"]].drop_duplicates().shape[0]
            ),
            "n_records_per_sample": int(per_sample_df.shape[0]),
            "labels": list(non_empty),
            "n_channels_per_label": n_channels_by_label,
        }
        per_partition_paths[pname] = {
            "per_sample_csv": str(per_sample_csv),
            "per_horizon_csv": str(per_horizon_csv),
            "per_anchor_csv": str(per_anchor_csv),
        }

        # Back-compat: duplicate the clinical_4band CSVs at the top level
        # so existing TE_Calculated / per_class_breakdown consumers still
        # find them. Plots stay only in the subdirectory to avoid clutter.
        if pname == _LEGACY_DUPLICATED:
            per_sample_df.to_csv(output_dir / "per_sample.csv", index=False)
            per_horizon_df.to_csv(output_dir / "per_horizon.csv", index=False)
            per_anchor_df.to_csv(output_dir / "per_anchor.csv", index=False)

    # ----- Per-channel CSV + plots ------------------------------------
    per_channel_summary: Dict[str, Any] = {}
    per_channel_paths: Dict[str, str] = {}
    if per_channel_rows:
        per_channel_dir = output_dir / "per_channel"
        per_channel_dir.mkdir(parents=True, exist_ok=True)

        per_channel_df = pd.DataFrame(per_channel_rows)
        per_channel_csv = per_channel_dir / "per_channel_forecast.csv"
        per_channel_df.to_csv(per_channel_csv, index=False)
        per_channel_paths["per_channel_csv"] = str(per_channel_csv)

        # Mean MSE per channel — top / bottom diagnostics.
        agg = per_channel_df.groupby("channel", as_index=False).agg(
            kind=("kind", "first"),
            band=("band", "first"),
            refined_band=("refined_band", "first"),
            octave=("octave", "first"),
            freq_hz_primary=("freq_hz_primary", "first"),
            harmonic_ratio=("harmonic_ratio", "first"),
            mean_mse=("mse_total", "mean"),
            mean_r2=("r2_total", "mean"),
        )
        worst_csv = per_channel_dir / "worst_channels_by_kind.csv"
        worst_rows: List[pd.DataFrame] = []
        for kind, sub in agg.groupby("kind"):
            top = sub.sort_values("mean_mse", ascending=False).head(10).copy()
            top["rank_in_kind"] = np.arange(1, len(top) + 1)
            worst_rows.append(top)
        if worst_rows:
            pd.concat(worst_rows, ignore_index=True).to_csv(
                worst_csv, index=False,
            )
            per_channel_paths["worst_channels_csv"] = str(worst_csv)

        _safe_plot(
            "per_channel_mse_vs_freq", plot_per_channel_mse_vs_freq,
            per_channel_df, channel_meta_df,
            per_channel_dir / "mse_vs_freq.pdf",
            band_hz_ranges=partition.band_hz_ranges,
        )
        classes = unique_labels_in(per_channel_df.get("label"))
        if len(classes) >= 2:
            _safe_plot(
                "per_channel_mse_vs_freq_by_class",
                plot_per_channel_mse_vs_freq,
                per_channel_df, channel_meta_df,
                per_channel_dir / "mse_vs_freq_by_class.pdf",
                band_hz_ranges=partition.band_hz_ranges,
                by_class=True,
            )
        _safe_plot(
            "phase_harmonic_mse", plot_phase_harmonic_mse,
            per_channel_df, channel_meta_df,
            per_channel_dir / "mse_vs_harmonic_ratio.pdf",
        )

        # Aggregate diagnostics for summary.json.
        worst5 = agg.sort_values("mean_mse", ascending=False).head(5)
        best5 = agg.sort_values("mean_mse", ascending=True).head(5)
        phase_kind_mse: Dict[str, float] = {}
        for kind in ("ph_diag", "ph_h2", "ph_h3", "ph_other"):
            sub = agg[agg["kind"] == kind]
            if not sub.empty:
                phase_kind_mse[kind] = float(np.nanmean(sub["mean_mse"]))
        per_channel_summary = {
            "n_channels": int(agg.shape[0]),
            "worst_5_by_mse": worst5.to_dict(orient="records"),
            "best_5_by_mse": best5.to_dict(orient="records"),
            "phase_kind_mse": phase_kind_mse,
        }

    # ----- Combined summary.json --------------------------------------
    legacy_summary = per_partition_summary.get(_LEGACY_DUPLICATED, {})
    legacy_per_band = legacy_summary.get("by_band", {})
    log_line = ", ".join(
        f"{b}={legacy_per_band[b]['mean_mse']:.4f}"
        for b in legacy_per_band
    )
    if log_line:
        logger.info(f"frequency_band_forecast (clinical_4band): {log_line}")

    summary: Dict[str, Any] = {
        # Legacy top-level keys mirror the clinical_4band view so existing
        # readers see the historic shape.
        "n_unique_samples": legacy_summary.get("n_unique_samples", 0),
        "n_records_per_sample": legacy_summary.get("n_records_per_sample", 0),
        "bands": legacy_summary.get("labels", []),
        "n_channels_per_band": legacy_summary.get("n_channels_per_label", {}),
        "by_band": legacy_per_band,
        "by_band_and_class": legacy_summary.get("by_band_and_class", {}),
        "fhr_phase_min_freq": float(fhr_phase_min_freq),
        "fs": float(fs),
        "decim_step_seconds": float(decim_step_seconds),
        # New: complete per-partition view + per-channel diagnostics.
        "partitions": per_partition_summary,
        "per_channel": per_channel_summary,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    summary["band_partition_json"] = str(json_path)
    summary["band_channel_map_csv"] = str(csv_path)
    summary["summary_json"] = str(summary_path)
    summary["partition_paths"] = per_partition_paths
    summary["per_channel_paths"] = per_channel_paths
    # Legacy top-level paths (back-compat duplicates of clinical_4band).
    if _LEGACY_DUPLICATED in per_partition_paths:
        summary["per_sample_csv"] = str(output_dir / "per_sample.csv")
        summary["per_horizon_csv"] = str(output_dir / "per_horizon.csv")
        summary["per_anchor_csv"] = str(output_dir / "per_anchor.csv")
    return summary


__all__ = ["run_frequency_band_forecast_analysis"]
