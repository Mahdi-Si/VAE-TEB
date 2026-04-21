"""Entry point for the VAE-TEB Lag-Attentive v1 testing pipeline.

Typical usage — edit the ``__main__`` block to set paths, then run
``python -m model.vae_teb_prediction.testing.run_tests``. All
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
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import yaml
from loguru import logger

# Ensure project root is on sys.path when run as a module.
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Testing components
from model.vae_teb_prediction.testing.analyses import (
    run_anchor_position_analysis,
    run_attention_diagnostics,
    run_class_separation_analysis,
    run_dataset_stats_analysis,
    run_encoder_probe,
    run_forecast_quality_analysis,
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
    run_uplift_analysis,
)
from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.visualizers_interactive import (
    plot_metrics_comparison_interactive,
)

# Data loading
from hdf5_dataset.hdf5_dataset import (
    build_guid_filtered_dataloader,
    create_optimized_dataloader,
)


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
        skip_per_class_breakdown: Skip the per-class CSV/plot breakdown
            (a pure post-processor over the pooled CSVs).
        skip_interactive: Skip Plotly interactive plots.
        analysis_samples: Number of per-sample diagnostic PDFs to emit.
        min_epochs_per_guid: Minimum epochs per GUID for trajectory
            analysis.
        max_guids: Optional cap on trajectory GUIDs.
        normalize_fields: Fields to apply per-channel normalisation to.
        dataset_kwargs: Additional constructor kwargs for
            ``CombinedHDF5Dataset``.

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
            "No test data provided. Pass data_path or supply config_path with "
            "dataset_config.vae_test_datasets."
        )

    device_obj = torch.device(
        device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    logger.info(f"Checkpoint: {checkpoint_path_resolved}")
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

    # 3. Forecast quality.
    _step(
        "forecast_quality",
        run_forecast_quality_analysis,
        runner, standard_loader, aggregate_cap,
    )

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

    # 7. Residual usage.
    _step(
        "residual_usage",
        run_residual_usage_analysis,
        runner, standard_loader, aggregate_cap,
    )

    # 8-9. Attention + TE lag class (gated).
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
    # IMPORTANT 1 — sample coverage: the standard test loader uses
    # ``shuffle=False`` and reads the configured HDF5 files sequentially.
    # Each file in the lag-attn v1 pipeline carries a single outcome class
    # (hie_no_cs, acidosis_no_cs, healthy_no_bg_no_cs, ...), so capping
    # ``max_samples`` too aggressively here exhausts the first file before
    # reaching the second — producing the misleading "Only 1 class(es)
    # found" error.
    #
    # IMPORTANT 2 — memory: we used to call ``collect_predictions`` here,
    # which retains the full forward output (``mu_full``, ``mu_base``,
    # ``delta_src``, ``y_plus``, ``attn``, ``te_lag``, raw ``fhr``/``up``,
    # ...) per sample — ~13 MB per record. With ``max_samples=None`` and
    # ten thousand+ samples in the test list that explodes to ~130 GB and
    # the OS reaps the process (exit code 137 / SIGKILL).
    #
    # Class separation only needs the per-sample time-averaged latent
    # ``z`` (24 floats) and the integer class label, so we now stream the
    # loader directly and accumulate just those — ~32 bytes per record
    # instead of 13 MB. Same statistical content, ~400 000× less RAM.
    try:
        import numpy as np
        import pandas as pd
        import torch as _torch_local
        from model.vae_teb_prediction.testing.collectors import (
            _extract_epoch,
            _extract_label,
        )

        X_rows: List[np.ndarray] = []
        labels: List[int] = []
        epochs: List[float] = []
        raw_label_counter: Dict[Any, int] = {}    # diagnostic: every label seen
        n_batches = 0
        n_seen = 0
        n_dropped_padonly = 0
        n_dropped_other = 0
        processed = 0
        with runner.inference_mode():
            for batch in runner.iter_batches(standard_loader, max_samples):
                outputs = runner.forward(batch)
                z = outputs.get("z")
                if z is None:
                    continue
                # (B, T, d_z) -> per-sample mean over time -> (B, d_z).
                z_mean = z.mean(dim=1).detach().cpu().numpy()
                batch_size = int(z_mean.shape[0])
                n_batches += 1
                for idx in range(batch_size):
                    if max_samples and processed >= max_samples:
                        break
                    lab = _extract_label(batch, idx)
                    n_seen += 1
                    raw_label_counter[lab] = raw_label_counter.get(lab, 0) + 1
                    # Drop pad-only (label None) and unknown class ids so
                    # they don't create a spurious zero-valued class.
                    if lab is None:
                        n_dropped_padonly += 1
                        processed += 1
                        continue
                    if int(lab) not in (1, 2, 3):
                        n_dropped_other += 1
                        processed += 1
                        continue
                    X_rows.append(z_mean[idx].astype(np.float32))
                    labels.append(int(lab))
                    ep = _extract_epoch(batch, idx)
                    epochs.append(float(ep) if ep is not None else float("nan"))
                    processed += 1
                if max_samples and processed >= max_samples:
                    break
                # Free per-batch tensors immediately.
                del outputs, z, z_mean
                if hasattr(_torch_local, "cuda") and _torch_local.cuda.is_available():
                    _torch_local.cuda.empty_cache()

        logger.info(
            f"class_separation streaming: n_batches={n_batches}, "
            f"n_seen={n_seen}, kept={len(X_rows)}, "
            f"dropped_pad_only={n_dropped_padonly}, "
            f"dropped_other_label={n_dropped_other}"
        )
        logger.info(
            f"class_separation raw labels seen: "
            + ", ".join(
                f"{k!r}={v}" for k, v in sorted(
                    raw_label_counter.items(),
                    key=lambda kv: (kv[0] is None, kv[0]),
                )
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
        )

    # 14. Sample diagnostics (gated).
    if not skip_forecast_heatmaps and analysis_samples > 0:
        _step(
            "sample_diagnostics",
            run_sample_diagnostics,
            runner, standard_loader, min(analysis_samples, max_samples or analysis_samples),
        )
        # 14b. Per-sample KLD + lag-attention diagnostic figures.
        _step(
            "kld_lag_diagnostics",
            run_kld_lag_diagnostics,
            runner, standard_loader, min(analysis_samples, max_samples or analysis_samples),
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
    if not skip_per_class_breakdown:
        _step(
            "per_class_breakdown",
            run_per_class_breakdown,
            runner.output_dir,
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

    loader = create_optimized_dataloader(
        hdf5_files=[str(p) for p in paths],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        stats_path=stats_path,
        normalize_fields=normalize_fields,
        rank=0,
        world_size=1,
        **resolved_kwargs,
    )
    logger.info(f"Loaded {len(loader.dataset)} test samples")
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

    loader_overrides: Dict[str, Any] = {}
    if num_workers is not None:
        loader_overrides["num_workers"] = num_workers

    eligible_guids, loader = build_guid_filtered_dataloader(
        dataset_paths=[str(p) for p in paths],
        min_samples=min_epochs_per_guid,
        max_guids=max_guids,
        sampler_shuffle=False,
        stats_path=stats_path,
        normalize_fields=normalize_fields,
        dataloader_overrides=loader_overrides if loader_overrides else None,
        **resolved_kwargs,
    )
    logger.info(
        f"GUID loader: {len(eligible_guids)} patients (>= {min_epochs_per_guid} epochs)"
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
) -> Dict[str, Any]:
    """Fast path: run a trimmed pipeline with slow analyses disabled.

    Skips the trajectory, attention, and per-sample heatmap steps so the
    pipeline can be validated end-to-end in a minute or two.
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
    )


# ----- Example usage -----
if __name__ == "__main__":
    # Edit these paths for your setup and run:
    #   python -m model.vae_teb_prediction.testing.run_tests
    CHECKPOINT: Optional[str] = None  # Use config's core_model_checkpoint if None
    DATA: Optional[str] = None  # Use config's vae_test_datasets if None
    STATS: Optional[str] = None  # Use config's dataset_config.stat_path if None
    CONFIG: Optional[str] = "model/vae_teb_prediction/model/config_lag_attn_v1.yaml"
    OUTPUT: Optional[str] = None  # Use config's folders_config.out_dir_base if None

    results = run_full_test_pipeline(
        checkpoint_path=CHECKPOINT,
        data_path=DATA,
        output_dir=OUTPUT,
        stats_path=STATS,
        config_path=CONFIG,
        max_samples=None,  # Process all samples
        analysis_samples=400,
    )

    # Headline summary
    hist = results.get("histogram")
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
    except Exception:
        pass
