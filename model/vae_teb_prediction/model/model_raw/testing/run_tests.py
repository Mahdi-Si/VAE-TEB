"""Entry point for the VAE-TEB Lag-Attentive v1 testing pipeline.

Typical usage — edit the ``__main__`` block to set paths, then run
``python -m model.vae_teb_prediction.model.model_raw.testing.run_tests``. All
configurable knobs flow through :func:`run_full_test_pipeline`.

Example:
    >>> from testing.run_tests import run_full_test_pipeline
    >>> results = run_full_test_pipeline(
    ...     checkpoint_path="best.ckpt",
    ...     data_path=None,
    ...     config_path="model/vae_teb_prediction/model/config_lag_attn_v1.yaml",
    ... )
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch
import yaml
from loguru import logger

# Ensure project root is on sys.path when run as a module.
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Testing components
from model.vae_teb_prediction.model.model_raw.testing.analyses import (
    run_anchor_position_analysis,
    run_attention_diagnostics,
    run_calibration_analysis,
    run_causal_te_validation,
    run_class_separation_analysis,
    run_cmi_comparison,
    run_dataset_stats_analysis,
    run_encoder_probe,
    run_forecast_quality_analysis,
    run_raw_forecast_analysis,
    run_histogram_analysis,
    run_horizon_error_profile,
    run_kld_lag_diagnostics,
    run_kld_pca_analysis,
    run_latent_distribution_analysis,
    run_latent_space_visualization,
    run_per_class_breakdown,
    run_residual_usage_analysis,
    run_sample_diagnostics,
    run_te_lag_class_analysis,
    run_trajectory_analysis,
    run_up_effect_analysis,
    run_uplift_analysis,
)
from model.vae_teb_prediction.model.model_raw.testing.base import TestRunner
from model.vae_teb_prediction.model.model_raw.testing.visualizers_interactive import (
    plot_metrics_comparison_interactive,
)

# Data loading
from hdf5_dataset.hdf5_dataset import (
    build_guid_filtered_dataloader,
    create_optimized_dataloader,
)


# ---------------------------------------------------------------------------
# Phase 1 process-pool helpers (one subgroup per GPU, used by
# :func:`run_full_test_pipeline_by_subgroup` when ``gpu_ids`` selects more
# than one GPU). Defined at module scope so they are picklable for the
# ``spawn``-context :class:`concurrent.futures.ProcessPoolExecutor`.
# ---------------------------------------------------------------------------


def _phase1_worker_init(gpu_queue: Any) -> None:
    """ProcessPoolExecutor initializer: pin this worker to one GPU.

    Pulls a single GPU id from ``gpu_queue`` and exports
    ``CUDA_VISIBLE_DEVICES`` in this child process. From the worker's
    perspective the assigned physical GPU then appears as ``cuda:0``,
    so downstream code can pass ``device="cuda:0"`` and it will land
    on the right card regardless of which physical id was claimed.

    The export happens here, before any CUDA call in this process —
    the module-level ``import torch`` at the top of this file does not
    initialise CUDA (it only imports the package), so the env-var
    change is honoured by the first ``torch.device('cuda:0')`` /
    ``model.to(device)`` call inside the task.
    """
    import os
    try:
        gpu_id = gpu_queue.get_nowait()
    except Exception:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))


def _run_phase1_subgroup_task(
    sg_name: str,
    sg_paths: List[str],
    sg_out: str,
    pipeline_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """ProcessPoolExecutor task: run one subgroup's full Phase 1 pipeline.

    Imports ``run_full_test_pipeline`` lazily so any
    ``CUDA_VISIBLE_DEVICES`` set by :func:`_phase1_worker_init` is in
    place by the time the runner constructs its first CUDA tensor.
    """
    import os
    from loguru import logger as _worker_logger

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    _worker_logger.info(
        f"[phase1 worker pid={os.getpid()} cuda_visible={visible}] "
        f"starting subgroup {sg_name!r}"
    )
    try:
        result = run_full_test_pipeline(
            data_path=list(sg_paths),
            output_dir=sg_out,
            **pipeline_kwargs,
        )
        _worker_logger.info(
            f"[phase1 worker pid={os.getpid()}] "
            f"subgroup {sg_name!r} complete."
        )
        return result if isinstance(result, dict) else {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        _worker_logger.error(
            f"[phase1 worker pid={os.getpid()}] "
            f"subgroup {sg_name!r} failed: {exc!r}"
        )
        return {"error": str(exc)}


def run_full_test_pipeline(
    checkpoint_path: Optional[str],
    data_path: Optional[Union[str, List[str]]],
    output_dir: Optional[str] = None,
    stats_path: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    skip_trajectory: bool = False,
    skip_attention: bool = False,
    skip_forecast_heatmaps: bool = False,
    skip_kld_pca: bool = False,
    skip_per_class_breakdown: bool = False,
    skip_interactive: bool = False,
    analysis_samples: int = 10,
    min_epochs_per_guid: int = 10,
    max_guids: Optional[int] = None,
    normalize_fields: Optional[Sequence[str]] = None,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    skip_up_effect: bool = False,
    skip_calibration: bool = False,
    skip_cmi_comparison: bool = False,
    empirical_te_csv: Optional[str] = None,
    skip_frequency_band: bool = False,
    skip_causal_te: bool = False,
    single_class_mode: bool = False,
    keep_kld_trajectory_only: bool = False,
    loader_override: Optional[Any] = None,
    guid_loader_override: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the full lag-attn v1 testing pipeline end-to-end.

    Args:
        checkpoint_path: Path to the checkpoint file (defaults to
            ``model_config.core_model_checkpoint`` from ``config_path``).
        data_path: Single HDF5 path or a list of paths (defaults to
            ``dataset_config.vae_test_datasets`` from ``config_path``).
        output_dir: Base output directory (defaults to a timestamped
            folder under ``general_config.folders_config.out_dir_base``).
        stats_path: Path to the per-channel normalisation stats HDF5.
        config_path: Path to the YAML config used to train the checkpoint.
            **Required** — the runner needs it to build the model.
        device: Torch device string (auto-detected if None).
        max_samples: Cap on samples for the aggregate (overall-behaviour)
            analyses — histogram, forecast_quality, horizon / anchor
            error, uplift, residual_usage, attention, te_lag, encoder
            probe, latent distribution, latent space, kld_pca, and class
            separation. Pass ``None`` to process **every** sample in
            the loaded test set (recommended for final reporting runs so
            the loader reaches every HDF5 file and every class is
            represented). Per-sample diagnostic PDFs are capped
            separately via ``analysis_samples``.
        batch_size: Test-time batch size (defaults to config).
        num_workers: DataLoader worker count (defaults to config).
        skip_trajectory: Skip the per-GUID trajectory analysis.
        skip_attention: Skip attention + TE lag class analyses.
        skip_forecast_heatmaps: Skip the per-sample diagnostic PDFs.
        skip_kld_pca: Skip the per-dim KL PCA analysis.
        skip_up_effect: Skip inference-time UP perturbation tests.
        skip_calibration: Skip the predictive-distribution calibration report.
        skip_cmi_comparison: Skip the neural + empirical CMI corroboration of
            ``K_raw`` (G11).
        empirical_te_csv: Optional path to a precomputed IDTxl empirical-TE CSV
            (with ``guid`` and ``ite_valid`` columns) joined into the CMI
            comparison at the patient level; a logged skip when absent.
        skip_frequency_band: Skip the frequency-band-stratified forecast
            quality analysis.
        skip_per_class_breakdown: Skip the per-class CSV/plot breakdown
            (a pure post-processor over the pooled CSVs).
        single_class_mode: When ``True``, the pipeline is being driven for
            a single subgroup that is intrinsically single-class (Phase 1
            of the per-subgroup driver, ``run_full_test_pipeline_by_subgroup``).
            Three downstream behaviours change: (a) the Step-0 probe
            "only 1 class / 1 file" warnings are downgraded to INFO, (b)
            the ``class_separation`` analysis is skipped (results entry is
            ``{"status": "skipped_single_class"}``), (c) the
            ``per_class_breakdown`` post-processor is skipped. Defaults to
            ``False`` so every existing caller behaves identically.
        skip_interactive: Skip Plotly interactive plots.
        analysis_samples: Number of per-sample diagnostic PDFs to emit.
        min_epochs_per_guid: Minimum epochs per GUID for trajectory
            analysis.
        max_guids: Optional cap on trajectory GUIDs.
        normalize_fields: Fields to apply per-channel normalisation to.
        dataset_kwargs: Additional constructor kwargs for
            ``CombinedHDF5Dataset``.
        loader_override: Optional pre-built standard test DataLoader. When
            supplied, the pipeline skips ``_create_dataloader`` and uses
            this loader instead — letting non-HDF5 datasets (e.g. the
            synthetic ``.npz`` benchmarks in
            ``model_experiment/synthetic/``) drive the same analysis suite
            without touching the HDF5 plumbing. ``data_path``,
            ``stats_path``, ``normalize_fields`` and ``dataset_kwargs``
            become unused in this mode. Defaults to ``None`` so every
            existing caller behaves identically.
        guid_loader_override: Optional pre-built per-GUID DataLoader for the
            trajectory step. When supplied, ``_create_guid_dataloader`` is
            skipped. Pass ``None`` together with ``skip_trajectory=True``
            for datasets without GUID-organised history (e.g. synthetic).

    Returns:
        Dict with one entry per analysis step.
    """
    if config_path is None:
        raise ValueError(
            "config_path is required: TestRunner needs it to build the "
            "SeqVaeLagAttnV1 model from model_config.VAE_model.*"
        )

    checkpoint_path_resolved, output_dir_resolved = _resolve_runner_settings(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        config_path=config_path,
    )

    data_paths: List[str] = []
    if isinstance(data_path, str):
        data_paths = [data_path]
    elif data_path is not None:
        data_paths = list(data_path)

    # When a pre-built loader is supplied (e.g. for synthetic .npz benchmarks)
    # every HDF5-specific resolver is bypassed: paths, stats, normalize_fields
    # and the dataset-construction kwargs are all irrelevant. We still resolve
    # batch_size / num_workers from the config so logging is consistent.
    using_loader_override = loader_override is not None
    if using_loader_override:
        stats_path_resolved: Optional[str] = None
        normalize_fields_resolved: Optional[Sequence[str]] = None
        dataset_kwargs_resolved: Dict[str, Any] = {}
        batch_size_resolved = batch_size if batch_size is not None else 0
        num_workers_resolved = num_workers if num_workers is not None else 0
    else:
        (
            data_paths,
            stats_path_resolved,
            batch_size_resolved,
            num_workers_resolved,
            normalize_fields_resolved,
            dataset_kwargs_resolved,
        ) = _resolve_dataloader_settings(
            data_paths=data_paths,
            stats_path=stats_path,
            batch_size=batch_size,
            num_workers=num_workers,
            normalize_fields=normalize_fields,
            dataset_kwargs=dataset_kwargs,
            config_path=config_path,
        )

        if not data_paths:
            raise ValueError(
                "No test data provided. Pass data_path or supply config_path "
                "with dataset_config.vae_test_datasets, or hand in a pre-built "
                "loader via loader_override."
            )

    device_obj = torch.device(
        device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    logger.info(f"Checkpoint: {checkpoint_path_resolved}")
    if using_loader_override:
        logger.info("Data:       <loader_override supplied — HDF5 path resolution skipped>")
    else:
        logger.info(f"Data:       {data_paths}")
    logger.info(f"Output:     {output_dir_resolved}")
    logger.info(f"Stats:      {stats_path_resolved}")
    logger.info(f"Config:     {config_path}")
    logger.info(f"Device:     {device_obj}")
    logger.info(f"Batch:      {batch_size_resolved}")

    # ----- Step 1: Create TestRunner -----
    logger.info("Loading model from checkpoint...")
    runner = TestRunner.from_checkpoint(
        checkpoint_path=checkpoint_path_resolved,
        output_dir=output_dir_resolved,
        config_path=config_path,
        device=device_obj,
    )

    # ----- Step 2: Create DataLoaders -----
    if using_loader_override:
        logger.info(
            "Using caller-supplied standard test dataloader "
            f"(dataset size = {len(getattr(loader_override, 'dataset', []))})."
        )
        standard_loader = loader_override
    else:
        logger.info("Creating standard test dataloader...")
        standard_loader = _create_dataloader(
            data_paths,
            batch_size_resolved,
            stats_path_resolved,
            normalize_fields=normalize_fields_resolved,
            num_workers=num_workers_resolved,
            dataset_kwargs=dataset_kwargs_resolved,
        )

    guid_loader = None
    if not skip_trajectory:
        if guid_loader_override is not None:
            logger.info(
                "Using caller-supplied GUID dataloader for trajectory analysis."
            )
            guid_loader = guid_loader_override
        elif using_loader_override:
            logger.info(
                "skip_trajectory=False but loader_override supplied without "
                "guid_loader_override; trajectory analysis will be skipped "
                "(no HDF5 paths to build a per-GUID loader from)."
            )
            guid_loader = None
        else:
            logger.info("Creating GUID-based dataloader for trajectory analysis...")
            try:
                _, guid_loader = _create_guid_dataloader(
                    data_paths,
                    stats_path=stats_path_resolved,
                    min_epochs_per_guid=min_epochs_per_guid,
                    max_guids=max_guids,
                    normalize_fields=normalize_fields_resolved,
                    num_workers=num_workers_resolved,
                    dataset_kwargs=dataset_kwargs_resolved,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to build GUID dataloader: {exc}")
                guid_loader = None

    results: Dict[str, Any] = {}

    # Sentinel used to mean "process every sample in the test set" for the
    # aggregate analyses. The downstream analyses break their collection
    # loops with ``if max_samples and processed >= max_samples`` so any
    # integer larger than the actual dataset size terminates at dataset
    # exhaustion rather than at the cap. Using a plain int (not None)
    # keeps the per-analysis signatures unchanged.
    _FULL_DATASET_CAP = 10_000_000

    def _cap(user_max: Optional[int]) -> int:
        """Resolve the per-analysis sample cap.

        - ``user_max is None``: process every sample (returns the
          ``_FULL_DATASET_CAP`` sentinel, larger than any realistic
          test set).
        - ``user_max`` is a positive int: honor it exactly.
        """
        return int(user_max) if user_max is not None else _FULL_DATASET_CAP

    # ------------------------------------------------------------------
    # STEP 0: definitive dataloader probe + one-pass latent capture.
    #
    # This step is intentionally FIRST — before any other analysis can
    # poison the DataLoader worker pool or the HDF5 FIFO cache. We
    # combine two jobs into one loader iteration:
    #
    #   1) Probe: per-file / per-label / per-target counts, so we can
    #      immediately hard-fail if the dataset layer isn't delivering
    #      what the config says.
    #   2) Latent capture: run the model forward on every batch and
    #      store the per-sample time-averaged latent ``z_mean``
    #      (24 floats / sample ≈ 96 bytes). This eliminates the second
    #      loader iteration that class_separation used to need — which
    #      was the proven source of the "Only 1 class(es) found" bug:
    #      the loader was silently truncating by the time the 12th
    #      analysis ran, after 11 prior iterations had populated/
    #      thrashed the workers' FIFO caches.
    #
    # Memory cost: N × 24 float32 ≈ 1 MB per 10 000 samples. Irrelevant.
    # Time cost: one model forward per sample — which every previous
    # single-pass analysis did anyway, so no net slowdown.
    # ------------------------------------------------------------------
    import numpy as np
    from collections import defaultdict
    from model.vae_teb_prediction.model.model_raw.testing.collectors import (
        _extract_epoch as _probe_epoch,
        _extract_guid as _probe_guid,
        _extract_label as _probe_label,
    )

    probe: Dict[str, Any] = {
        "n_samples_in_dataset": int(len(getattr(standard_loader, "dataset", []))),
        "n_batches": 0,
        "n_samples_seen": 0,
        "per_file_counts": defaultdict(int),
        "per_label_counts": defaultdict(int),
        "raw_target_first_nonzero_hist": defaultdict(int),
        "sample_index": [],   # List[Tuple[label, source_file_basename, epoch, guid]]
    }
    probe_z_means: List[np.ndarray] = []   # per-sample (d_z,) float32 arrays

    logger.info("=" * 60)
    logger.info("Step 0: probing test dataloader + capturing z_mean ...")
    try:
        with runner.inference_mode():
            # Use runner.iter_batches so tensors land on the runner's
            # device without us moving them manually. max_samples=None
            # iterates the whole loader.
            for batch in runner.iter_batches(standard_loader, None):
                probe["n_batches"] += 1
                batch_size = int(batch.fhr_st.size(0)) if hasattr(batch, "fhr_st") else 0
                if batch_size == 0:
                    continue

                # Forward pass — capture only z_mean, nothing else.
                try:
                    outputs = runner.forward(batch)
                    z = outputs.get("z")
                    if z is not None:
                        # (B, T, d_z) -> (B, d_z) on CPU
                        z_mean_batch = z.mean(dim=1).detach().cpu().numpy()
                    else:
                        z_mean_batch = None
                    del outputs
                except Exception as exc:
                    logger.error(f"probe forward failed on batch {probe['n_batches']}: {exc}")
                    z_mean_batch = None

                for idx in range(batch_size):
                    # source_file_basename — set by HDF5Dataset.__getitem__
                    src = getattr(batch, "source_file_basename", None)
                    if src is not None:
                        try:
                            src_val = src[idx] if hasattr(src, "__getitem__") else str(src)
                        except Exception:
                            src_val = "?"
                        if isinstance(src_val, bytes):
                            src_val = src_val.decode("utf-8", errors="replace")
                        src_val = str(src_val)
                    else:
                        src_val = "<no source_file_basename>"
                    probe["per_file_counts"][src_val] += 1

                    lab = _probe_label(batch, idx)
                    key = "None" if lab is None else int(lab)
                    probe["per_label_counts"][key] += 1

                    # Raw-target first-non-zero (truncation diagnostic).
                    tgt_attr = getattr(batch, "target", None)
                    if tgt_attr is not None:
                        try:
                            raw = tgt_attr[idx]
                            if hasattr(raw, "detach"):
                                raw = raw.detach().cpu().numpy()
                            else:
                                raw = np.asarray(raw)
                            nz = raw[raw > 0]
                            first_nz = float(nz[0]) if nz.size else 0.0
                        except Exception:
                            first_nz = float("nan")
                        bucket = round(first_nz, 2)
                        probe["raw_target_first_nonzero_hist"][bucket] += 1

                    ep = _probe_epoch(batch, idx)
                    guid = _probe_guid(batch, idx)
                    probe["sample_index"].append((
                        key,
                        src_val,
                        float(ep) if ep is not None else float("nan"),
                        guid,
                    ))
                    if z_mean_batch is not None:
                        probe_z_means.append(z_mean_batch[idx].astype(np.float32))
                    else:
                        probe_z_means.append(np.array([], dtype=np.float32))
                    probe["n_samples_seen"] += 1
    except Exception as exc:  # noqa: BLE001
        logger.error(f"loader probe failed mid-iteration: {exc}")

    # Convert defaultdicts to plain dicts for clean logging / JSON.
    probe["per_file_counts"] = dict(probe["per_file_counts"])
    probe["per_label_counts"] = dict(probe["per_label_counts"])
    probe["raw_target_first_nonzero_hist"] = dict(
        probe["raw_target_first_nonzero_hist"]
    )

    logger.info(
        f"Probe summary: dataset reports "
        f"{probe['n_samples_in_dataset']} samples, "
        f"loader yielded {probe['n_batches']} batches "
        f"= {probe['n_samples_seen']} samples"
    )
    logger.info("Probe per-file counts:")
    for fname, cnt in sorted(probe["per_file_counts"].items()):
        logger.info(f"    {fname:40s}  {cnt:>8d}")
    logger.info(
        "Probe per-label counts: "
        + ", ".join(f"{k}={v}" for k, v in sorted(
            probe["per_label_counts"].items(),
            key=lambda kv: (kv[0] == "None", kv[0]),
        ))
    )
    logger.info(
        "Probe raw-target first-non-zero histogram (truncation check): "
        + ", ".join(f"{k}={v}" for k, v in sorted(
            probe["raw_target_first_nonzero_hist"].items()
        ))
    )

    # Persist the probe summary so it's recoverable from disk.
    try:
        probe_path = Path(output_dir_resolved) / "loader_probe.json"
        with open(probe_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "n_samples_in_dataset": probe["n_samples_in_dataset"],
                    "n_batches": probe["n_batches"],
                    "n_samples_seen": probe["n_samples_seen"],
                    "per_file_counts": probe["per_file_counts"],
                    "per_label_counts": {
                        str(k): v for k, v in probe["per_label_counts"].items()
                    },
                    "raw_target_first_nonzero_hist": {
                        str(k): v for k, v in
                        probe["raw_target_first_nonzero_hist"].items()
                    },
                },
                fh, indent=2,
            )
        logger.info(f"Probe summary saved to {probe_path}")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to write loader_probe.json: {exc}")

    results["loader_probe"] = {
        k: v for k, v in probe.items() if k != "sample_index"
    }

    # Hard-fail signals — much louder than a deep "Only 1 class" later.
    distinct_files = len(probe["per_file_counts"])
    distinct_labels = sum(
        1 for k in probe["per_label_counts"] if k in (1, 2, 3)
    )
    if single_class_mode:
        # Phase 1 of the per-subgroup driver: each subgroup is a single
        # HDF5 file with a single outcome class. The "only 1 file / 1
        # class" probe signals are *expected*, not errors.
        logger.info(
            f"loader probe (single_class_mode): "
            f"distinct_files={distinct_files}, distinct_labels={distinct_labels}"
        )
    else:
        if distinct_files <= 1:
            logger.error(
                f"loader probe: only {distinct_files} HDF5 file yielded "
                f"samples (config lists multiple). Per-class analyses will "
                f"be degenerate. Check dataset path/filtering."
            )
        if distinct_labels < 2:
            logger.error(
                f"loader probe: only {distinct_labels} canonical clinical "
                f"class(es) {{1,2,3}} found in target. _extract_label may be "
                f"truncating fractional weights to 0, or the dataset's "
                f"target field has unexpected encoding."
            )

    def _step(name: str, fn, *args, **kwargs) -> None:
        try:
            logger.info("=" * 60)
            logger.info(f"Running {name} ...")
            results[name] = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{name} failed: {exc}")
            results[name] = {"error": str(exc)}

    # 1. Dataset statistics (cheap, model-agnostic).
    _step(
        "dataset_stats",
        run_dataset_stats_analysis,
        standard_loader,
        Path(output_dir_resolved) / "dataset_stats",
    )

    # For the aggregate (overall-behaviour) analyses, the cap is taken
    # from the user's ``max_samples``. ``max_samples=None`` → process
    # every sample in the loaded test set (see ``_cap`` above).
    aggregate_cap = _cap(max_samples)

    # Heavy collectors retain full per-sample forward tensors
    # (``collect_predictions`` ≈ 13 MB / sample, ``collect_attention_maps``
    # ≈ 750 KB / sample). Letting them iterate the entire test set blows
    # past available RAM and silently kills DataLoader workers, after
    # which downstream steps see batches from only the first HDF5 file
    # (the famous "Only 1 class(es) found" symptom). The numbers below
    # are statistically sufficient for their respective analyses (latent
    # scatter, linear probe, attention heatmaps) and orders of magnitude
    # smaller than typical test sets.
    HEAVY_PRED_CAP = min(aggregate_cap, 2000)   # collect_predictions consumers
    HEAVY_ATTN_CAP = min(aggregate_cap, 2000)   # collect_attention_maps consumers
    PROBE_CAP = min(aggregate_cap, 5000)       # linear-probe / latent stats

    # 2. Histogram (cheap, establishes baseline metrics).
    _step(
        "histogram",
        run_histogram_analysis,
        runner, standard_loader, aggregate_cap,
    )

    # 3. Raw-waveform forecast quality (raw port): waveform VAF/MSE/SNR/R^2 + multi-scale low-pass
    # and the raw overlay / per-horizon / heatmap plots. Replaces the feature-domain
    # ``forecast_quality`` (per-channel over the 87-ch target), which has no raw analogue.
    _step(
        "raw_forecast",
        run_raw_forecast_analysis,
        runner, standard_loader, aggregate_cap,
    )

    # 3b. (Removed in the raw port, S6-T01c) The scattering frequency-band-stratified forecast
    # quality step is dropped -- the raw target is a single-block waveform with no band partition.

    # 4. Horizon error profile.
    _step(
        "horizon_error",
        run_horizon_error_profile,
        runner, standard_loader, aggregate_cap,
    )

    # 5. Anchor position analysis.
    _step(
        "anchor_error",
        run_anchor_position_analysis,
        runner, standard_loader, aggregate_cap,
    )

    # 6. Uplift.
    _step(
        "uplift",
        run_uplift_analysis,
        runner, standard_loader, aggregate_cap,
    )

    # 7. UP perturbation effect (no-retrain source ablation).
    if not skip_up_effect:
        _step(
            "up_effect",
            run_up_effect_analysis,
            runner, standard_loader, min(aggregate_cap, 1000),
        )

    # 7b. Calibration of the learned predictive distribution (G10). Needs `logvar_full`;
    # degrades to a logged skip on a model that does not emit it.
    if not skip_calibration:
        _step(
            "calibration",
            run_calibration_analysis,
            runner, standard_loader, aggregate_cap,
        )

    # 7c. Neural + empirical CMI corroboration of K_raw (G11). Needs the encoder states and
    # the raw KL; degrades to a logged skip on a model that does not emit them. The empirical-TE
    # column is populated only when `empirical_te_csv` is supplied (else logged-skipped).
    if not skip_cmi_comparison:
        _step(
            "cmi_comparison",
            run_cmi_comparison,
            runner, standard_loader, aggregate_cap,
            empirical_te_csv=empirical_te_csv,
        )

    # 8. Residual usage.
    _step(
        "residual_usage",
        run_residual_usage_analysis,
        runner, standard_loader, aggregate_cap,
    )

    # 9-10. Attention + TE lag class (gated).
    if not skip_attention:
        # Capped: collect_attention_maps retains per-anchor (T,L), (T,M),
        # (L,) numpy arrays per sample (~750 KB each). 2 000 samples is
        # already plenty for stable per-anchor histograms.
        _step(
            "attention",
            run_attention_diagnostics,
            runner, standard_loader, HEAVY_ATTN_CAP,
        )
        _step(
            "te_lag",
            run_te_lag_class_analysis,
            runner, standard_loader, aggregate_cap,
        )

    # 10. Encoder probe (linear classifier — 5 000 samples is more than
    # enough for stable AUC).
    _step(
        "encoder_probe",
        run_encoder_probe,
        runner, standard_loader, PROBE_CAP,
    )

    # 11. Latent distribution & space visualization.
    # Capped: ``run_latent_space_visualization`` calls ``collect_predictions``
    # which retains the full ~13 MB per-sample forward output. Use the
    # heavy-pred cap so this step doesn't poison the dataloader workers
    # before later steps run.
    _step(
        "latent_distribution",
        run_latent_distribution_analysis,
        runner, standard_loader, PROBE_CAP,
    )
    _step(
        "latent_space",
        run_latent_space_visualization,
        runner, standard_loader, HEAVY_PRED_CAP,
    )

    # 12. Class separation (uses time-averaged latent features).
    #
    # PURE POST-PROCESSOR — does NOT iterate the dataloader.
    #
    # All per-sample data needed (z_mean + label + source_file + epoch)
    # was captured in the Step 0 probe pass. By the time we reach here,
    # 11 other analyses have run, each with its own loader iteration —
    # which has historically poisoned the dataloader workers and
    # truncated the iterable to the first file's worth of samples
    # (~1000 records, all class 3=HIE), producing the
    # "Only 1 class(es) found" symptom.
    #
    # Reading from the probe's in-memory data eliminates that failure
    # mode entirely: the data is already on the heap, no I/O, no model
    # forward, no worker process involvement.
    #
    # In ``single_class_mode`` (Phase 1 of the per-subgroup driver) the
    # collected samples are intrinsically single-class, so this analysis
    # is degenerate. Skip and record the skip status — Phase 2 produces
    # the cross-subgroup equivalent.
    if single_class_mode:
        logger.info(
            "class_separation: skipped (single_class_mode — Phase 1 of "
            "the per-subgroup driver). Cross-subgroup separation is "
            "covered by Phase 2."
        )
        results["class_separation"] = {"status": "skipped_single_class"}
    else:
        try:
            import pandas as pd

            probe_index = probe.get("sample_index", [])
            if not probe_index or not probe_z_means:
                raise RuntimeError(
                    "class_separation: Step 0 probe captured no samples — "
                    "check the probe output for HDF5 / dataset issues"
                )
            if len(probe_index) != len(probe_z_means):
                raise RuntimeError(
                    f"class_separation: probe index ({len(probe_index)}) and "
                    f"z_means ({len(probe_z_means)}) length mismatch"
                )

            X_rows: List[np.ndarray] = []
            labels: List[int] = []
            epochs: List[float] = []
            per_file_kept: Dict[str, int] = defaultdict(int)
            dropped_padonly = 0
            dropped_other = 0

            for (lab_key, src, ep, _guid), zm in zip(probe_index, probe_z_means):
                if zm.size == 0:
                    # Forward failed for this batch in Step 0 — skip.
                    dropped_other += 1
                    continue
                if lab_key is None or lab_key == "None":
                    dropped_padonly += 1
                    continue
                try:
                    lab_int = int(lab_key)
                except (TypeError, ValueError):
                    dropped_other += 1
                    continue
                if lab_int not in (1, 2, 3):
                    dropped_other += 1
                    continue
                X_rows.append(zm)
                labels.append(lab_int)
                epochs.append(float(ep) if ep == ep else float("nan"))  # NaN-safe
                per_file_kept[src] += 1

            per_file_kept = dict(per_file_kept)
            logger.info(
                f"class_separation (from probe): kept={len(X_rows)}, "
                f"dropped_pad_only={dropped_padonly}, "
                f"dropped_other_label={dropped_other}"
            )
            logger.info(
                "class_separation per-file z_mean coverage: "
                + ", ".join(
                    f"{k}={v}" for k, v in sorted(per_file_kept.items())
                )
            )

            if X_rows:
                X = np.asarray(X_rows, dtype=float)
                lab_arr = np.asarray(labels)
                unique_labs, counts = np.unique(lab_arr, return_counts=True)
                logger.info(
                    f"class_separation: collected {len(lab_arr)} samples "
                    f"across {len(unique_labs)} class(es): "
                    + ", ".join(
                        f"{int(u)}={int(c)}" for u, c in zip(unique_labs, counts)
                    )
                )
                if len(unique_labs) < 2:
                    logger.error(
                        "class_separation: only one class present in the "
                        "collected sample set. The standard test loader reads "
                        "HDF5 files sequentially with shuffle=False, so a "
                        "small max_samples can miss the later class files. "
                        "Re-run with max_samples=None (or a larger cap) so "
                        "all test HDF5 files are covered."
                    )
                    results["class_separation"] = {
                        "error": "single-class sample set",
                        "n_samples": int(len(lab_arr)),
                        "unique_labels": [int(u) for u in unique_labs],
                    }
                else:
                    df_cols: Dict[str, Any] = {
                        f"z{i}": X[:, i] for i in range(X.shape[1])
                    }
                    df_cols["label"] = lab_arr
                    ep_arr = np.asarray(epochs, dtype=float)
                    df_cols["hours_before"] = -ep_arr / 3600.0
                    latent_df = pd.DataFrame(df_cols)
                    _step(
                        "class_separation",
                        run_class_separation_analysis,
                        latent_df,
                        Path(output_dir_resolved) / "class_separation",
                    )
            else:
                logger.warning("class_separation: no latent samples collected.")
                results["class_separation"] = {"error": "no samples"}
        except Exception as exc:  # noqa: BLE001
            logger.error(f"class_separation failed: {exc}")
            results["class_separation"] = {"error": str(exc)}

    # 13. Trajectory analysis (gated).
    if not skip_trajectory and guid_loader is not None:
        _step(
            "trajectory",
            run_trajectory_analysis,
            runner, guid_loader,
            time_range_hours=12.0,
            min_epochs_per_guid=min_epochs_per_guid,
            keep_kld_trajectory_only=keep_kld_trajectory_only,
        )

    # 14. Sample diagnostics (gated).
    if not skip_forecast_heatmaps and analysis_samples > 0:
        # Cap analysis_samples by max_samples when set. Use ``is None``
        # explicitly so an explicit ``max_samples=0`` request does not get
        # coerced back to ``analysis_samples`` by ``or``.
        diag_cap = (
            analysis_samples if max_samples is None
            else min(analysis_samples, max_samples)
        )
        _step(
            "sample_diagnostics",
            run_sample_diagnostics,
            runner, standard_loader, diag_cap,
        )
        # 14b. Per-sample KLD + lag-attention diagnostic figures.
        _step(
            "kld_lag_diagnostics",
            run_kld_lag_diagnostics,
            runner, standard_loader, diag_cap,
        )

    # 14c. PCA on per-dim KL trajectory.
    if not skip_kld_pca:
        _step(
            "kld_pca",
            run_kld_pca_analysis,
            runner, standard_loader, aggregate_cap,
        )

    # 14d. Per-class breakdown of pooled CSVs (post-processor; must be last
    # of the analysis steps so all input CSVs already exist).
    #
    # Skipped in ``single_class_mode`` (Phase 1 of the per-subgroup
    # driver) because every input CSV has only one class — the
    # post-processor would emit one trivially-non-empty class folder and
    # nothing to overlay. Phase 2 produces the cross-subgroup overlays
    # instead.
    if single_class_mode:
        logger.info(
            "per_class_breakdown: skipped (single_class_mode — Phase 1 "
            "of the per-subgroup driver). Cross-subgroup overlays are "
            "covered by Phase 2."
        )
        results["per_class_breakdown"] = {"status": "skipped_single_class"}
    elif not skip_per_class_breakdown:
        _step(
            "per_class_breakdown",
            run_per_class_breakdown,
            runner.output_dir,
        )

    # 14e. Causal-TE validation suite. Runs after every other analysis so
    # the CSV-driven Tests 1, 2, 3, 9 can read upstream artifacts; Tests 4
    # and 10 share a single ``collect_predictions`` pass capped at the
    # ``HEAVY_PRED_CAP`` ceiling. Skipped via ``skip_causal_te=True``.
    if not skip_causal_te:
        _step(
            "causal_te_validation",
            run_causal_te_validation,
            runner, standard_loader,
            output_dir=Path(output_dir_resolved) / "causal_te_validation",
            max_samples=HEAVY_PRED_CAP,
            histogram_csv=Path(output_dir_resolved) / "histograms" / "histogram_metrics.csv",
            up_effect_dir=Path(output_dir_resolved) / "up_effect",
        )

    # 15. Interactive metrics comparison.
    if not skip_interactive and isinstance(results.get("histogram"), object):
        try:
            import pandas as pd
            hist = results.get("histogram")
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                plot_metrics_comparison_interactive(
                    hist,
                    Path(output_dir_resolved) / "metrics_comparison.html",
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"metrics_comparison_interactive failed: {exc}")

    # 16. Summary JSON.
    _save_summary(results, Path(output_dir_resolved))
    logger.info(f"Testing complete! Results saved to {output_dir_resolved}")
    return results


def run_full_test_pipeline_by_subgroup(
    subgroups: Optional[Mapping[str, Sequence[str]]] = None,
    *,
    checkpoint_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    stats_path: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    skip_trajectory: bool = False,
    skip_attention: bool = False,
    skip_forecast_heatmaps: bool = False,
    skip_kld_pca: bool = False,
    skip_interactive: bool = False,
    analysis_samples: int = 10,
    min_epochs_per_guid: int = 10,
    max_guids: Optional[int] = None,
    normalize_fields: Optional[Sequence[str]] = None,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    skip_up_effect: bool = False,
    skip_calibration: bool = False,
    skip_cmi_comparison: bool = False,
    empirical_te_csv: Optional[str] = None,
    skip_frequency_band: bool = False,
    skip_causal_te: bool = False,
    skip_phase2: bool = False,
    only_subgroups: Optional[Sequence[str]] = None,
    gpu_ids: Optional[Sequence[int]] = None,
    keep_kld_trajectory_only: bool = False,
) -> Dict[str, Any]:
    """Run the testing pipeline per outcome subgroup, then compare cross-subgroup.

    Two-phase driver around :func:`run_full_test_pipeline`:

    * **Phase 1** — for each entry in ``subgroups`` (a mapping
      ``{subgroup_name: [hdf5_paths]}``) the standard pipeline runs once
      with ``single_class_mode=True`` and ``output_dir=<base>/phase1/<name>/``.
      Each subgroup is intrinsically single-class so ``class_separation``
      and ``per_class_breakdown`` are skipped (degenerate on one class).
    * **Phase 2** — if ``len(subgroups) >= 2`` (and ``skip_phase2`` is
      ``False``), the CSV-driven post-processor
      :func:`cross_subgroup_breakdown.run_cross_subgroup_breakdown` reads
      every Phase 1 folder and emits cross-subgroup overlays + statistical
      comparisons under ``<base>/phase2/cross_subgroup/``.

    Phase 2 is skipped automatically when only one subgroup is supplied
    (no comparison to make). Subgroup failures are isolated: one
    subgroup raising does not abort the others, and Phase 2 still runs
    against whatever Phase 1 folders exist on disk.

    Args:
        subgroups: Mapping ``{name: [hdf5_paths]}``. When ``None`` the
            mapping is resolved via
            :func:`analyses.subgroup_utils.resolve_subgroups` from the
            YAML key ``dataset_config.vae_test_subgroups`` (preferred) or
            from ``dataset_config.vae_test_datasets`` as a single
            ``{"all": [...]}`` fallback.
        checkpoint_path: As in :func:`run_full_test_pipeline`.
        output_dir: Base output directory under which ``phase1/`` and
            ``phase2/`` are created. When ``None`` the same timestamped
            path as the single-pipeline run is used (``out_dir_base/
            <tag>/<run_date>/test_results``).
        stats_path: As in :func:`run_full_test_pipeline`.
        config_path: As in :func:`run_full_test_pipeline`.
        device: As in :func:`run_full_test_pipeline`.
        max_samples: Cap applied uniformly to every Phase 1 invocation.
        batch_size: As in :func:`run_full_test_pipeline`.
        num_workers: As in :func:`run_full_test_pipeline`.
        skip_trajectory: As in :func:`run_full_test_pipeline`.
        skip_attention: As in :func:`run_full_test_pipeline`.
        skip_forecast_heatmaps: As in :func:`run_full_test_pipeline`.
        skip_kld_pca: As in :func:`run_full_test_pipeline`.
        skip_interactive: As in :func:`run_full_test_pipeline`.
        analysis_samples: As in :func:`run_full_test_pipeline`.
        min_epochs_per_guid: As in :func:`run_full_test_pipeline`.
        max_guids: As in :func:`run_full_test_pipeline`.
        normalize_fields: As in :func:`run_full_test_pipeline`.
        dataset_kwargs: As in :func:`run_full_test_pipeline`.
        skip_up_effect: As in :func:`run_full_test_pipeline`.
        skip_frequency_band: As in :func:`run_full_test_pipeline`.
        skip_phase2: When ``True``, skip the cross-subgroup post-processor
            even if ``len(subgroups) >= 2``. Useful when you want to run
            Phase 2 manually later.
        only_subgroups: Optional iterable of subgroup names to filter the
            mapping before iteration (resume convenience). Phase 2 still
            runs against whatever ``phase1/<name>/`` folders exist on
            disk after iteration completes.
        gpu_ids: Optional list of physical GPU ids to use for Phase 1.
            When the list has $\\ge 2$ entries the subgroup loop is
            dispatched via a ``spawn``-context
            :class:`concurrent.futures.ProcessPoolExecutor` with one
            worker process per GPU; each worker is pinned to its GPU
            via ``CUDA_VISIBLE_DEVICES`` and runs every subgroup it is
            assigned sequentially on that GPU. The pool's ``__exit__``
            joins all workers so Phase 2 only ever starts after every
            subgroup has finished writing its outputs. When ``gpu_ids``
            is ``None`` or contains a single id the original sequential
            loop is used (no behaviour change for single-GPU users).

    Returns:
        ``{"phase1": {name: results_or_error_dict, ...},
           "phase2": phase2_results_or_status,
           "output_dir": str(base_dir)}``.
    """
    if config_path is None and subgroups is None:
        raise ValueError(
            "Either subgroups or config_path is required. Pass an "
            "explicit subgroups dict, or supply config_path so "
            "vae_test_subgroups / vae_test_datasets can be read."
        )

    # Lazy imports — these modules are added by the same refactor and
    # may not exist yet in older checkouts. Importing here keeps
    # ``run_full_test_pipeline`` callable on its own.
    from model.vae_teb_prediction.model.model_raw.testing.analyses.subgroup_utils import (
        resolve_subgroups,
    )

    resolved_subgroups = resolve_subgroups(
        explicit=subgroups,
        config_path=config_path,
    )
    if not resolved_subgroups:
        raise ValueError(
            "No subgroups resolved. Pass `subgroups=...` explicitly, "
            "or set `dataset_config.vae_test_subgroups` (or the legacy "
            "`vae_test_datasets`) in the YAML config."
        )

    if only_subgroups is not None:
        wanted = set(only_subgroups)
        filtered = {n: paths for n, paths in resolved_subgroups.items() if n in wanted}
        missing = wanted.difference(filtered)
        if missing:
            logger.warning(
                f"only_subgroups: {sorted(missing)} not present in the "
                f"resolved subgroup map; ignoring."
            )
        resolved_subgroups = filtered  # type: ignore[assignment]
        if not resolved_subgroups:
            raise ValueError(
                "only_subgroups filtered the subgroup map down to zero "
                "entries — nothing to run."
            )

    # Compute the timestamped base output dir exactly once so every
    # subgroup writes under the same parent. Reuse the single-pipeline
    # resolver for parity with existing run layouts.
    _, base_path = _resolve_runner_settings(
        checkpoint_path=checkpoint_path or "(deferred)",
        output_dir=output_dir,
        config_path=config_path,
    )
    base_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(
        f"Subgroup-mode pipeline: {len(resolved_subgroups)} subgroup(s) "
        f"=> {base_path}"
    )
    logger.info(f"Subgroups: {list(resolved_subgroups.keys())}")
    logger.info("=" * 70)

    phase1_results: Dict[str, Any] = {}

    # Decide whether to dispatch Phase 1 in parallel across multiple
    # GPUs. With $\le 1$ GPU listed, fall through to the original
    # sequential loop (zero behaviour change for single-GPU runs).
    gpu_ids_list: List[int] = (
        [int(g) for g in gpu_ids] if gpu_ids is not None else []
    )
    if len(gpu_ids_list) >= 2 and torch.cuda.is_available():
        # Trim to actual visible device count if the caller over-listed.
        n_visible = torch.cuda.device_count()
        if max(gpu_ids_list) >= n_visible:
            logger.warning(
                f"gpu_ids contains ids >= cuda device_count={n_visible}; "
                f"trimming to visible ids."
            )
            gpu_ids_list = [g for g in gpu_ids_list if g < n_visible]
    use_parallel = len(gpu_ids_list) >= 2

    if use_parallel:
        # ----- Parallel path: one subgroup per GPU at a time -----
        # Each worker process is pinned to a single physical GPU via
        # ``CUDA_VISIBLE_DEVICES`` and sees its card as ``cuda:0``.
        # Phase 2 below the ``with`` block is guaranteed to run after
        # *every* worker has joined, because
        # :class:`ProcessPoolExecutor`'s ``__exit__`` calls
        # ``shutdown(wait=True)``.
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        n_workers = min(len(gpu_ids_list), len(resolved_subgroups))
        logger.info("\n" + "#" * 70)
        logger.info(
            f"# Phase 1 (parallel): dispatching {len(resolved_subgroups)} "
            f"subgroup(s) across {n_workers} GPU worker(s) "
            f"gpu_ids={gpu_ids_list[:n_workers]}"
        )
        logger.info("#" * 70)

        ctx = mp.get_context("spawn")
        gpu_queue = ctx.Queue()
        for gid in gpu_ids_list[:n_workers]:
            gpu_queue.put(int(gid))

        # Static kwargs every worker shares. ``device='cuda:0'`` is
        # correct because ``CUDA_VISIBLE_DEVICES`` masks the real id.
        parallel_kwargs: Dict[str, Any] = dict(
            checkpoint_path=checkpoint_path,
            stats_path=stats_path,
            config_path=str(config_path) if config_path is not None else None,
            device="cuda:0",
            max_samples=max_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            skip_trajectory=skip_trajectory,
            skip_attention=skip_attention,
            skip_forecast_heatmaps=skip_forecast_heatmaps,
            skip_kld_pca=skip_kld_pca,
            skip_per_class_breakdown=True,   # belt-and-braces
            skip_interactive=skip_interactive,
            analysis_samples=analysis_samples,
            min_epochs_per_guid=min_epochs_per_guid,
            max_guids=max_guids,
            normalize_fields=(
                list(normalize_fields) if normalize_fields is not None else None
            ),
            dataset_kwargs=(
                dict(dataset_kwargs) if dataset_kwargs is not None else None
            ),
            skip_up_effect=skip_up_effect,
            skip_calibration=skip_calibration,
            skip_cmi_comparison=skip_cmi_comparison,
            empirical_te_csv=empirical_te_csv,
            skip_frequency_band=skip_frequency_band,
            skip_causal_te=skip_causal_te,
            single_class_mode=True,
            keep_kld_trajectory_only=keep_kld_trajectory_only,
        )

        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_phase1_worker_init,
            initargs=(gpu_queue,),
        ) as executor:
            future_to_sg: Dict[Any, str] = {}
            for sg_name, sg_paths in resolved_subgroups.items():
                sg_out = base_path / "phase1" / sg_name
                sg_out.mkdir(parents=True, exist_ok=True)
                logger.info(
                    f"# Phase 1 (parallel): queued subgroup {sg_name!r} "
                    f"({len(list(sg_paths))} file(s)) -> {sg_out}"
                )
                fut = executor.submit(
                    _run_phase1_subgroup_task,
                    sg_name,
                    list(sg_paths),
                    str(sg_out),
                    parallel_kwargs,
                )
                future_to_sg[fut] = sg_name

            for fut in as_completed(future_to_sg):
                sg_name = future_to_sg[fut]
                try:
                    result = fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        f"# Phase 1 subgroup {sg_name!r} worker raised: {exc}"
                    )
                    phase1_results[sg_name] = {"error": str(exc)}
                    continue
                if isinstance(result, dict) and "error" in result:
                    logger.error(
                        f"# Phase 1 subgroup {sg_name!r} failed: "
                        f"{result['error']}"
                    )
                else:
                    logger.info(
                        f"# Phase 1 subgroup {sg_name!r} complete."
                    )
                phase1_results[sg_name] = result
        # Executor's ``with`` block exited => every worker has joined
        # and every Phase 1 subgroup has flushed its outputs to disk.
        logger.info(
            f"# Phase 1 (parallel) complete: "
            f"{len(phase1_results)}/{len(resolved_subgroups)} subgroup(s) "
            f"finished."
        )
    else:
        # ----- Sequential path (single GPU / CPU; original behaviour) -----
        for sg_name, sg_paths in resolved_subgroups.items():
            sg_out = base_path / "phase1" / sg_name
            logger.info("\n" + "#" * 70)
            logger.info(
                f"# Phase 1: subgroup {sg_name!r} "
                f"({len(list(sg_paths))} file(s))"
            )
            logger.info(f"# Output: {sg_out}")
            logger.info("#" * 70)
            try:
                phase1_results[sg_name] = run_full_test_pipeline(
                    checkpoint_path=checkpoint_path,
                    data_path=list(sg_paths),
                    output_dir=str(sg_out),
                    stats_path=stats_path,
                    config_path=config_path,
                    device=device,
                    max_samples=max_samples,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    skip_trajectory=skip_trajectory,
                    skip_attention=skip_attention,
                    skip_forecast_heatmaps=skip_forecast_heatmaps,
                    skip_kld_pca=skip_kld_pca,
                    skip_per_class_breakdown=True,   # belt-and-braces
                    skip_interactive=skip_interactive,
                    analysis_samples=analysis_samples,
                    min_epochs_per_guid=min_epochs_per_guid,
                    max_guids=max_guids,
                    normalize_fields=normalize_fields,
                    dataset_kwargs=dataset_kwargs,
                    skip_up_effect=skip_up_effect,
                    skip_calibration=skip_calibration,
                    skip_cmi_comparison=skip_cmi_comparison,
                    empirical_te_csv=empirical_te_csv,
                    skip_frequency_band=skip_frequency_band,
                    skip_causal_te=skip_causal_te,
                    single_class_mode=True,
                    keep_kld_trajectory_only=keep_kld_trajectory_only,
                )
                logger.info(f"# Phase 1 subgroup {sg_name!r} complete.")
            except Exception as exc:  # noqa: BLE001
                logger.error(f"# Phase 1 subgroup {sg_name!r} failed: {exc}")
                phase1_results[sg_name] = {"error": str(exc)}

    # ----- Phase 2: cross-subgroup post-processor -----
    phase2_results: Any = None
    n_subgroups = len(resolved_subgroups)
    if skip_phase2:
        logger.info("Phase 2 skipped (skip_phase2=True).")
        phase2_results = {"status": "skipped_user"}
    elif n_subgroups < 2:
        logger.info("Phase 2 skipped (only 1 subgroup — no comparison to make).")
        phase2_results = {"status": "skipped_single_subgroup"}
    else:
        from model.vae_teb_prediction.model.model_raw.testing.analyses.cross_subgroup_breakdown import (
            run_cross_subgroup_breakdown,
        )
        phase2_dir = base_path / "phase2" / "cross_subgroup"
        logger.info("\n" + "#" * 70)
        logger.info(f"# Phase 2: cross-subgroup comparison")
        logger.info(f"# Inputs : {base_path / 'phase1'}")
        logger.info(f"# Output : {phase2_dir}")
        logger.info("#" * 70)
        try:
            phase2_results = run_cross_subgroup_breakdown(
                phase1_root=base_path / "phase1",
                output_dir=phase2_dir,
            )
            logger.info("# Phase 2 complete.")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"# Phase 2 failed: {exc}")
            phase2_results = {"error": str(exc)}

    # Persist a top-level digest so the run is recoverable from disk.
    summary_path = base_path / "subgroup_summary.json"
    try:
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "subgroups": list(resolved_subgroups.keys()),
                    "n_subgroups": n_subgroups,
                    "phase1_status": {
                        n: ("error" if isinstance(r, dict) and "error" in r else "ok")
                        for n, r in phase1_results.items()
                    },
                    "phase2_status": (
                        phase2_results.get("status")
                        if isinstance(phase2_results, dict) and "status" in phase2_results
                        else ("error" if isinstance(phase2_results, dict)
                              and "error" in phase2_results else "ok")
                    ),
                    "output_dir": str(base_path),
                },
                fh, indent=2,
            )
        logger.info(f"Subgroup summary saved to {summary_path}")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to write subgroup_summary.json: {exc}")

    return {
        "phase1": phase1_results,
        "phase2": phase2_results,
        "output_dir": str(base_path),
    }


def _create_dataloader(
    data_path: Union[str, Path, Sequence[Union[str, Path]]],
    batch_size: int,
    stats_path: Optional[str] = None,
    normalize_fields: Optional[Sequence[str]] = None,
    num_workers: int = 0,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Create a standard DataLoader for the test dataset.

    Args:
        data_path: HDF5 path or list of paths.
        batch_size: Batch size.
        stats_path: Optional normalisation stats HDF5 path.
        normalize_fields: Optional list of fields to normalise.
        num_workers: DataLoader worker count.
        dataset_kwargs: Extra constructor kwargs for
            ``CombinedHDF5Dataset``.

    Returns:
        A PyTorch DataLoader that yields batched ``AttributeDict`` samples.
    """
    paths = list(data_path) if isinstance(data_path, (list, tuple)) else [data_path]
    resolved_kwargs = {} if dataset_kwargs is None else dict(dataset_kwargs)
    if "pin_memory" not in resolved_kwargs:
        resolved_kwargs["pin_memory"] = True

    # CRITICAL: force num_workers=0 for the test loader.
    #
    # ``create_optimized_dataloader`` hard-codes
    #   ``persistent_workers=True if num_workers > 0 else False``
    # (hdf5_dataset.py:1047). With persistent_workers=True + spawn
    # multiprocessing + a multi-file HDF5 dataset, PyTorch workers enter a
    # degraded state after the FIRST full loader iteration: the second
    # iteration silently terminates early — in our testing runs, at
    # exactly the length of file 0's index range (~1000 samples, all of
    # class 3/HIE).
    #
    # This breaks every analysis that runs after Step 0 (which does the
    # first full pass). Symptoms:
    #   * histogram/forecast_quality/uplift/etc. all "n=1000" regardless
    #     of the cap we pass
    #   * per_class_breakdown finds "n_classes=1" because all 1000 are HIE
    #   * encoder_probe says "not enough data" (one-class subset)
    #
    # The fix is num_workers=0: single-process, no persistent workers,
    # deterministic iteration. Slower than 4 workers, but correct.
    # Users who want parallel loading should set
    # ``dataset_kwargs.persistent_workers=False`` on their training side;
    # the test pipeline stays single-process.
    forced_num_workers = 0
    if num_workers and num_workers > 0:
        logger.warning(
            f"Test loader: forcing num_workers=0 (config requested "
            f"{num_workers}). Multi-worker HDF5 loading is unreliable "
            f"across multiple loader iterations — see _create_dataloader "
            f"for the full diagnosis."
        )

    loader = create_optimized_dataloader(
        hdf5_files=[str(p) for p in paths],
        batch_size=batch_size,
        num_workers=forced_num_workers,
        shuffle=False,
        stats_path=stats_path,
        normalize_fields=normalize_fields,
        rank=0,
        world_size=1,
        **resolved_kwargs,
    )
    logger.info(
        f"Loaded {len(loader.dataset)} test samples "
        f"(num_workers=0, persistent_workers=False)"
    )
    return loader


def _create_guid_dataloader(
    data_path: Union[str, Path, Sequence[Union[str, Path]]],
    stats_path: Optional[str] = None,
    min_epochs_per_guid: int = 3,
    max_guids: Optional[int] = None,
    normalize_fields: Optional[Sequence[str]] = None,
    num_workers: Optional[int] = None,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Create a GUID-based DataLoader: each batch holds one patient."""
    paths = list(data_path) if isinstance(data_path, (list, tuple)) else [data_path]
    resolved_kwargs = {} if dataset_kwargs is None else dict(dataset_kwargs)
    if "pin_memory" not in resolved_kwargs:
        resolved_kwargs["pin_memory"] = True

    # Same num_workers=0 override as _create_dataloader — see the
    # comment there for the full explanation. TL;DR: persistent workers
    # + multi-file HDF5 + multiple loader iterations silently truncates
    # subsequent passes to one file's worth of samples. num_workers=0 is
    # slower but correct.
    if num_workers and num_workers > 0:
        logger.warning(
            f"GUID test loader: forcing num_workers=0 (config requested "
            f"{num_workers})."
        )
    loader_overrides: Dict[str, Any] = {
        "num_workers": 0,
        "persistent_workers": False,
    }

    eligible_guids, loader = build_guid_filtered_dataloader(
        dataset_paths=[str(p) for p in paths],
        min_samples=min_epochs_per_guid,
        max_guids=max_guids,
        sampler_shuffle=False,
        stats_path=stats_path,
        normalize_fields=normalize_fields,
        dataloader_overrides=loader_overrides,
        **resolved_kwargs,
    )
    logger.info(
        f"GUID loader: {len(eligible_guids)} patients "
        f"(>= {min_epochs_per_guid} epochs, num_workers=0)"
    )
    return eligible_guids, loader


def _load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config or {}


def _resolve_runner_settings(
    *,
    checkpoint_path: Optional[str],
    output_dir: Optional[str],
    config_path: Optional[Union[str, Path]],
) -> tuple[Path, Path]:
    """Resolve checkpoint and output directory paths from config + overrides."""
    resolved_checkpoint = checkpoint_path
    resolved_output = output_dir

    if config_path is not None:
        config = _load_config(config_path)
        model_cfg = config.get("model_config", {}) or {}
        folders_cfg = config.get("general_config", {}).get("folders_config", {}) or {}

        if not resolved_checkpoint:
            resolved_checkpoint = model_cfg.get("core_model_checkpoint")

        if resolved_output is None:
            base_dir = folders_cfg.get("out_dir_base")
            if base_dir:
                now = datetime.now()
                run_date = now.strftime("%Y-%m-%d--[%H-%M-%S]") + f"--{now.microsecond:06d}-"
                experiment_tag = config.get("general_config", {}).get("tag", "test")
                tag_dir = Path(base_dir) / experiment_tag
                timestamped_dir = tag_dir / run_date
                resolved_output = str(timestamped_dir / "test_results")

    if not resolved_checkpoint:
        raise ValueError(
            "checkpoint_path is required unless config_path provides "
            "model_config.core_model_checkpoint."
        )

    if not resolved_output:
        resolved_output = "test_results"

    return Path(resolved_checkpoint), Path(resolved_output)


def _resolve_dataloader_settings(
    *,
    data_paths: List[str],
    stats_path: Optional[str],
    batch_size: Optional[int],
    num_workers: Optional[int],
    normalize_fields: Optional[Sequence[str]],
    dataset_kwargs: Optional[Dict[str, Any]],
    config_path: Optional[Union[str, Path]],
) -> tuple[List[str], Optional[str], int, int, Optional[Sequence[str]], Dict[str, Any]]:
    """Resolve dataloader knobs from config + explicit overrides."""
    resolved_paths = list(data_paths) if data_paths else []
    resolved_stats = stats_path
    resolved_batch_size = batch_size
    resolved_workers = num_workers
    resolved_normalize_fields = normalize_fields
    resolved_kwargs = {} if dataset_kwargs is None else dict(dataset_kwargs)

    if config_path is not None:
        config = _load_config(config_path)
        dataset_cfg = config.get("dataset_config", {}) or {}
        dataloader_cfg = dataset_cfg.get("dataloader_config", {}) or {}

        if not resolved_paths:
            resolved_paths = list(dataset_cfg.get("vae_test_datasets", []) or [])
        if resolved_stats is None:
            # Accept both spellings for backwards compatibility.
            resolved_stats = dataset_cfg.get("stat_path") or dataset_cfg.get("stats_path")
        if resolved_batch_size is None:
            resolved_batch_size = (
                config.get("general_config", {})
                .get("batch_size", {})
                .get("test")
            )
        if resolved_workers is None:
            resolved_workers = dataloader_cfg.get("num_workers", 0)
        if resolved_normalize_fields is None:
            resolved_normalize_fields = dataloader_cfg.get("normalize_fields")

        config_dataset_kwargs = dataloader_cfg.get("dataset_kwargs", {}) or {}
        merged_kwargs = dict(config_dataset_kwargs)
        merged_kwargs.update(resolved_kwargs)
        resolved_kwargs = merged_kwargs

    if resolved_batch_size is None:
        resolved_batch_size = 32
    if resolved_workers is None:
        resolved_workers = 0

    return (
        resolved_paths,
        resolved_stats,
        resolved_batch_size,
        resolved_workers,
        resolved_normalize_fields,
        resolved_kwargs,
    )


def _save_summary(results: Dict[str, Any], output_dir: Path) -> None:
    """Persist a compact summary JSON with the headline metrics."""
    import pandas as pd

    summary_path = output_dir / "test_summary.json"
    summary: Dict[str, Any] = {}

    hist = results.get("histogram")
    if isinstance(hist, pd.DataFrame) and not hist.empty:
        summary["metrics"] = {
            "n_samples": int(len(hist)),
            "feat_mse_mean": float(hist["feat_mse_total"].mean())
            if "feat_mse_total" in hist.columns else None,
            "feat_r2_mean": float(hist["feat_r2_total"].mean())
            if "feat_r2_total" in hist.columns else None,
            "uplift_rel_mean": float(hist["uplift_rel"].mean())
            if "uplift_rel" in hist.columns else None,
            "residual_ratio_mean": float(hist["residual_ratio"].mean())
            if "residual_ratio" in hist.columns else None,
            "kld_mean": float(hist["kld_mean"].mean())
            if "kld_mean" in hist.columns else None,
            "kld_sum_mean": float(hist["kld_sum"].mean())
            if "kld_sum" in hist.columns else None,
            "kld_l2_mean": float(hist["kld_l2"].mean())
            if "kld_l2" in hist.columns else None,
        }

    if isinstance(results.get("calibration"), dict):
        # G10 headline: proper scoring rules plus the coverage error at each nominal level.
        # `nll_gain_over_constant <= 0` means the learned variance head beats nothing.
        summary["calibration"] = {
            k: v for k, v in results["calibration"].items()
            if isinstance(v, (int, float, str))
        }

    if isinstance(results.get("cmi_comparison"), dict):
        # G11 headline: rank correlations of the neural CMI bounds (and empirical TE) with
        # K_raw. Positive rho corroborates K_raw as a source-specific information measure.
        summary["cmi_comparison"] = {
            k: v for k, v in results["cmi_comparison"].items()
            if isinstance(v, (int, float, str, bool))
        }

    if isinstance(results.get("attention"), dict):
        summary["attention"] = {
            k: v for k, v in results["attention"].items()
            if isinstance(v, (int, float, str))
        }

    if isinstance(results.get("residual_usage"), dict):
        summary["residual_usage"] = {
            k: v for k, v in results["residual_usage"].items()
            if isinstance(v, (int, float))
        }

    if isinstance(results.get("up_effect"), dict):
        summary["up_effect"] = {
            k: v for k, v in results["up_effect"].items()
            if isinstance(v, (int, float, str))
        }

    if isinstance(results.get("dataset_stats"), dict):
        ds = results["dataset_stats"]
        summary["dataset_stats"] = {
            "n_samples": ds.get("n_samples"),
            "n_guids": ds.get("n_guids"),
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_path}")


def quick_test(
    checkpoint_path: Optional[str],
    data_path: Optional[str],
    output_dir: str = "quick_test_results",
    stats_path: Optional[str] = None,
    n_samples: int = 100,
    config_path: Optional[Union[str, Path]] = None,
    num_workers: Optional[int] = None,
    normalize_fields: Optional[Sequence[str]] = None,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    loader_override: Optional[Any] = None,
) -> Dict[str, Any]:
    """Fast path: run a trimmed pipeline with slow analyses disabled.

    Skips the trajectory and per-sample heatmap steps so the pipeline can be validated end-to-end
    in a minute or two. Pass ``loader_override`` (e.g.
    :func:`conftest.make_tiny_eval_loader`) to drive it off an in-memory loader with no HDF5
    dependency -- the raw-port fixture-eval path (S7-T02).
    """
    return run_full_test_pipeline(
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        output_dir=output_dir,
        stats_path=stats_path,
        max_samples=n_samples,
        skip_trajectory=True,
        skip_attention=False,
        skip_forecast_heatmaps=True,
        skip_interactive=True,
        analysis_samples=0,
        config_path=config_path,
        num_workers=num_workers,
        normalize_fields=normalize_fields,
        dataset_kwargs=dataset_kwargs,
        loader_override=loader_override,
    )


# ----- Example usage -----
if __name__ == "__main__":
    # Edit these paths for your setup and run:
    #   python -m model.vae_teb_prediction.model.model_raw.testing.run_tests
    CHECKPOINT: Optional[str] = None  # Use config's core_model_checkpoint if None
    DATA: Optional[str] = None  # Use config's vae_test_datasets if None
    STATS: Optional[str] = None  # Use config's dataset_config.stat_path if None
    CONFIG: Optional[str] = (
        "model/vae_teb_prediction/model/model_raw/testing/config_raw_v4_testing.yaml"
    )
    OUTPUT: Optional[str] = None  # Use config's folders_config.out_dir_base if None

    # Dispatch on whether the YAML config requests subgroup mode.
    # ``vae_test_subgroups`` (or ``vae_test_fold_dir``) → two-phase driver.
    # Otherwise fall back to the legacy single-pass pipeline.
    _subgroup_map = {}
    _gpu_ids: Optional[List[int]] = None
    if CONFIG is not None:
        try:
            from model.vae_teb_prediction.model.model_raw.testing.analyses.subgroup_utils import (
                resolve_subgroups,
            )
            _subgroup_map = resolve_subgroups(config_path=CONFIG)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Subgroup resolution failed ({exc!r}); falling back to "
                f"the legacy single-pass pipeline."
            )
        # Optional Phase 1 multi-GPU dispatch. Read
        # ``dataset_config.test_gpu_ids`` (preferred) or, as a
        # convenience fallback, ``trainer.cuda_devices`` from the
        # testing config. When the resolved list has $\ge 2$ entries
        # ``run_full_test_pipeline_by_subgroup`` runs each subgroup on
        # its own GPU via a process pool; with $\le 1$ entry the
        # original sequential loop is used.
        # Also read ``dataset_config.keep_kld_trajectory_only`` (default
        # ``False``) so the YAML drives the trajectory-slim-mode flag.
        _keep_kld_trajectory_only = False
        _empirical_te_csv: Optional[str] = None
        try:
            with open(CONFIG, "r", encoding="utf-8") as _fh:
                _cfg = yaml.safe_load(_fh) or {}
            _ds_cfg = _cfg.get("dataset_config", {}) or {}
            _trainer_cfg = _cfg.get("trainer", {}) or {}
            _model_cfg = _cfg.get("model_config", {}) or {}
            # Optional empirical-TE CSV (IDTxl) for the G11 CMI comparison; may live under
            # either dataset_config or model_config. Absent -> the empirical column is skipped.
            _empirical_te_csv = _ds_cfg.get("empirical_te_csv") or _model_cfg.get(
                "empirical_te_csv"
            )
            _raw_ids = _ds_cfg.get("test_gpu_ids")
            if _raw_ids is None:
                _raw_ids = _trainer_cfg.get("cuda_devices")
            if isinstance(_raw_ids, (list, tuple)) and _raw_ids:
                _gpu_ids = [int(g) for g in _raw_ids]
                logger.info(
                    f"__main__: resolved Phase 1 gpu_ids={_gpu_ids} "
                    f"from config."
                )
            _keep_kld_trajectory_only = bool(
                _ds_cfg.get("keep_kld_trajectory_only", False)
            )
            if _keep_kld_trajectory_only:
                logger.info(
                    "__main__: keep_kld_trajectory_only=True — trajectory "
                    "step will emit only the population-level KLD-vs-time "
                    "plots."
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Failed to read test_gpu_ids from config ({exc!r}); "
                f"falling back to single-GPU sequential dispatch."
            )

    if len(_subgroup_map) >= 2:
        logger.info(
            f"__main__: dispatching to run_full_test_pipeline_by_subgroup "
            f"({len(_subgroup_map)} subgroups, "
            f"gpu_ids={_gpu_ids if _gpu_ids else 'sequential'})."
        )
        results = run_full_test_pipeline_by_subgroup(
            subgroups=_subgroup_map,
            checkpoint_path=CHECKPOINT,
            output_dir=OUTPUT,
            stats_path=STATS,
            config_path=CONFIG,
            max_samples=None,
            analysis_samples=400,
            gpu_ids=_gpu_ids,
            empirical_te_csv=_empirical_te_csv,
            keep_kld_trajectory_only=_keep_kld_trajectory_only,
        )
    else:
        logger.info(
            "__main__: dispatching to run_full_test_pipeline (single-pass)."
        )
        results = run_full_test_pipeline(
            checkpoint_path=CHECKPOINT,
            data_path=DATA,
            output_dir=OUTPUT,
            stats_path=STATS,
            config_path=CONFIG,
            max_samples=None,
            analysis_samples=400,
            empirical_te_csv=_empirical_te_csv,
            keep_kld_trajectory_only=_keep_kld_trajectory_only,
        )

    # Headline summary (single-pass results only — Phase 1 results live
    # under ``results["phase1"][<subgroup>]`` in subgroup mode and we
    # don't print a per-subgroup table from here).
    hist = results.get("histogram") if isinstance(results, dict) else None
    try:
        import pandas as pd
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            print("\n=== Lag-Attn V1 Test Results ===")
            print(f"Samples:          {len(hist)}")
            print(f"feat_mse_total:   {hist['feat_mse_total'].mean():.6f}")
            print(f"feat_r2_total:    {hist['feat_r2_total'].mean():.4f}")
            print(f"uplift_rel:       {hist['uplift_rel'].mean():.4f}")
            print(f"residual_ratio:   {hist['residual_ratio'].mean():.4f}")
            print(f"kld_mean:         {hist['kld_mean'].mean():.4f}")
        elif isinstance(results, dict) and "phase1" in results:
            print("\n=== Subgroup-mode test complete ===")
            print(f"Output dir: {results.get('output_dir')}")
            print(f"Subgroups:  {list(results['phase1'].keys())}")
    except Exception:
        pass
