"""Single-fold training entry point for ``guid_cls_v1`` (PRD §9).

Flow:

1. (Optional) auto-run :func:`precompute_fold_latents` when
   ``vae.freeze_vae=True`` and the cache for this fold is missing.
2. Build train/val/test :class:`GuidSequenceDataset`.
3. Compute per-fold inverse-frequency class weights at the GUID level.
4. Wrap train/val/test splits in DataLoaders; train uses true bucket-specific
   batch sizing.
5. Instantiate :class:`GuidOutcomeClassifier` (auto-detecting
   ``d_model_vae``/``d_z`` from the train cache attrs) and the
   :class:`PlGuidClassifier` Lightning wrapper.
6. Fit with Lightning (``ModelCheckpoint`` monitors ``val/total_loss``).
7. Resolve the best checkpoint via ``checkpoint_callback.best_model_path``
   and run a targeted ``trainer.validate(ckpt_path=...)`` to recover the
   accuracy / macro-F1 / AUROC at the chosen step.
8. Save ``fold_results.json`` and return the result dict.

The function is structured so that :func:`run_kfold_parallel` can call it
inside a ``ProcessPoolExecutor`` subprocess.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from hdf5_dataset.length_bucket_sampler import VariableBatchBucketSampler
from model.vae_teb_prediction.new_classifier.guid_cls_v1.callbacks import (
    TwoStageVaeUnfreeze,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.collate import (
    guid_sequence_collate_fn,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_classifier import (
    GuidClassifierConfig,
    GuidOutcomeClassifier,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_dataset import (
    GuidSequenceDataset,
    LiveGuidSequenceDataset,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.lightning_module import (
    PlGuidClassifier,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.logging_utils import (
    FoldLogHandle,
    attach_fold_log_sinks,
    detach_fold_log_sinks,
    dump_json,
    to_json_safe,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.losses import (
    LossWeights,
    class_prior_bin,
    class_priors_3,
    estimate_class_weights_3,
    estimate_class_weights_bin,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.precompute_latents import (
    _file_signature,
    _resolve_run_dir,
    build_vae_from_config,
    compute_checkpoint_sha256,
    get_fold_partition_files,
    precompute_fold_latents,
)


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load the YAML config from disk."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _bucket_batch_sizes(general_cfg: Dict[str, Any]) -> List[Tuple[Tuple[int, int], int]]:
    """Parse the ``batch_size_by_bucket`` map into ordered (range, bs) tuples.

    Args:
        general_cfg: The ``general_config`` dict from the YAML.

    Returns:
        Ordered list of ``((lo, hi), batch_size)`` tuples; default fallback
        is the PRD bucket layout when missing.
    """
    raw = general_cfg.get("batch_size_by_bucket", {}) or {}
    buckets: List[Tuple[Tuple[int, int], int]] = []
    if raw:
        for key, bs in raw.items():
            lo_s, hi_s = str(key).split("_")
            buckets.append(((int(lo_s), int(hi_s)), int(bs)))
        buckets.sort(key=lambda x: x[0][0])
    if not buckets:
        buckets = [
            ((1, 5), 16),
            ((6, 12), 12),
            ((13, 20), 8),
            ((21, 40), 4),
        ]
    return buckets


def _build_collate(
    cls_cfg: Dict[str, Any],
):
    """Bind the model's relative-time-bias config into the collate.

    The model's bias table is sized at ``n_rel_buckets``; the collate
    must produce bucket indices in the same range. ``rel_bucket_d_max``
    sets the saturation horizon. Both are read from
    ``model_config.classifier``.
    """
    from functools import partial as _partial  # noqa: WPS433

    return _partial(
        guid_sequence_collate_fn,
        rel_time_num_buckets=int(cls_cfg.get("n_rel_buckets", 32)),
        rel_time_d_max=float(cls_cfg.get("rel_bucket_d_max", 40.0)),
    )


def _make_train_dataloader(
    dataset: GuidSequenceDataset,
    *,
    bucket_batch_sizes: Sequence[Tuple[Tuple[int, int], int]],
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
    mp_context: str,
    seed: int,
    cls_cfg: Dict[str, Any],
) -> DataLoader:
    """Wrap the train dataset in a length-bucketed DataLoader.

    Uses a true variable-size batch sampler so each length bucket gets the
    batch size specified in ``bucket_batch_sizes``.

    Args:
        dataset: GuidSequenceDataset.
        bucket_batch_sizes: Ordered ``[((lo, hi), bs), ...]``.
        num_workers: DataLoader workers.
        prefetch_factor: Per-worker prefetch.
        persistent_workers: Whether workers persist across epochs.
        pin_memory: Pin DataLoader output tensors.
        mp_context: ``"spawn"`` (CUDA-safe) or ``"fork"``.
        seed: Sampler RNG seed.
        cls_cfg: ``model_config.classifier`` dict; supplies the
            relative-time bias parameters consumed by the collate.

    Returns:
        Configured DataLoader.
    """
    batch_sampler = VariableBatchBucketSampler(
        lengths=dataset.guid_lengths,
        bucket_batch_sizes=bucket_batch_sizes,
        shuffle=True,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=_build_collate(cls_cfg),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context=mp_context if num_workers > 0 else None,
        persistent_workers=persistent_workers and num_workers > 0,
        pin_memory=pin_memory,
        # Surface a stuck worker as a real exception instead of a silent
        # 6 h fold-timeout. Default ``timeout=0`` blocks indefinitely; 300 s
        # is well above the sub-second normal batch latency, so this only
        # triggers on a pathological NFS / h5py / OOM stall.
        timeout=300 if num_workers > 0 else 0,
    )


def _make_eval_dataloader(
    dataset: GuidSequenceDataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    mp_context: str,
    cls_cfg: Dict[str, Any],
) -> DataLoader:
    """Sequential DataLoader for val / test (no sampler shuffling).

    Like the train loader, binds the relative-time bias config into the
    collate so the dataset's bucket indices match the model.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_build_collate(cls_cfg),
        drop_last=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context=mp_context if num_workers > 0 else None,
        persistent_workers=False,
        pin_memory=pin_memory,
        # Same rationale as the train loader: convert a silent NFS / h5py /
        # OOM stall into a real ``RuntimeError`` after 300 s instead of
        # hanging for the entire fold timeout.
        timeout=300 if num_workers > 0 else 0,
    )


def _build_callbacks(
    *,
    fold_dir: Path,
    monitor: str,
    save_top_k: int,
    es_patience: int,
    es_enabled: bool,
    logging_cfg: Optional[Dict[str, Any]] = None,
):
    """Build the standard callback list (ModelCheckpoint + EarlyStopping +
    repo-shared LossPlot/Hparam/Metrics callbacks +
    :class:`TrainingDiagnosticsCallback` when ``logging.diagnostics_callback.enabled``
    is True).
    """
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint  # noqa: WPS433
    from train.callbacks import (  # noqa: WPS433
        HyperparameterLoggingCallback,
        LossPlotCallback,
        MetricsLoggingCallback,
    )

    ckpt_dir = fold_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_results_dir = fold_dir / "train_results"
    train_results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = fold_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Use a fixed ``best`` filename so Lightning overwrites a single file
    # in-place each time a better checkpoint is produced. This avoids two
    # historical pitfalls of metric-templated names:
    #   * Some Lightning versions don't replace ``/`` with ``_`` when
    #     rendering ``{val/total_loss:.4f}``, which spawned an unintended
    #     ``val/`` subdirectory.
    #   * Multi-file ``save_top_k`` listings with metric-encoded names
    #     accumulated stale checkpoints across runs.
    # The end result is always exactly one file: ``checkpoints/best.ckpt``
    # (plus optionally ``last.ckpt`` if ``save_last=True``).
    filename_template = "best"

    callbacks: List[Any] = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            # On-disk filename is exactly ``best.ckpt``. Lightning overwrites
            # it whenever a better checkpoint is produced. ``train_fold``
            # still reads ``ckpt_cb.best_model_path`` for the authoritative
            # path; the evaluator goes through that path directly.
            filename=filename_template,
            monitor=monitor,
            mode="min",
            save_top_k=int(save_top_k),
            save_last=False,
            auto_insert_metric_name=False,
        ),
        LossPlotCallback(output_dir=str(train_results_dir)),
        HyperparameterLoggingCallback(output_dir=str(train_results_dir)),
        MetricsLoggingCallback(),
    ]
    if es_enabled:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode="min",
                patience=int(es_patience),
            )
        )

    # Per-fold diagnostics callback (grad / weight norms + per-class val
    # metrics + epoch_summary.jsonl). Opt-out via
    # ``logging.diagnostics_callback.enabled: false`` — defaults are on.
    diag_cfg = (logging_cfg or {}).get("diagnostics_callback", {}) or {}
    if bool(diag_cfg.get("enabled", True)):
        from model.vae_teb_prediction.new_classifier.guid_cls_v1.diagnostics_callback import (  # noqa: WPS433
            TrainingDiagnosticsCallback,
        )

        summary_name = str(
            (logging_cfg or {}).get(
                "epoch_summary_jsonl", "epoch_summary.jsonl"
            )
        )
        callbacks.append(
            TrainingDiagnosticsCallback(
                log_dir=logs_dir,
                log_grad_norm=bool(diag_cfg.get("log_grad_norm", True)),
                log_weight_norm=bool(diag_cfg.get("log_weight_norm", True)),
                log_param_group_lrs=bool(diag_cfg.get("log_param_group_lrs", True)),
                log_per_class_metrics=bool(diag_cfg.get("log_per_class_metrics", True)),
                log_calibration=bool(diag_cfg.get("log_calibration", True)),
                epoch_summary_filename=summary_name,
            )
        )
    return callbacks


def _resolve_paths(
    config: Dict[str, Any], fold_id: int, output_dir_override: Optional[str]
) -> Tuple[Path, Path]:
    """Pick the run directory and per-fold subdirectory.

    Args:
        config: Parsed YAML.
        fold_id: 1-based fold id.
        output_dir_override: Optional explicit run-dir override.

    Returns:
        ``(run_dir, fold_dir)``.
    """
    run_dir = (
        Path(output_dir_override).resolve()
        if output_dir_override
        else _resolve_run_dir(config, None)
    )
    fold_dir = run_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, fold_dir


def _latent_stats_cache_dir(config: Dict[str, Any]) -> Path:
    """Resolve the shared latent-stats cache directory.

    Lives outside the per-run output directory so the cache survives
    re-runs of the same experiment tag (the cost of computing
    ``fit_latent_stats`` is identical across re-runs as long as the
    inputs match — checkpoint, train file partition, window, etc.).
    """
    base = Path(config["general_config"]["folders_config"]["out_dir_base"]).resolve()
    return base / "_latent_stats_cache"


def _latent_stats_cache_key(
    *,
    config: Dict[str, Any],
    fold_id: int,
    train_files: Sequence[str],
    vae_checkpoint_sha: str,
) -> str:
    """SHA256 over every input that determines the running stats output.

    Mirrors the cache-input-signature philosophy of
    :func:`precompute_latents.build_cache_input_summary`. Any change to
    the train file set, normalization config, window, VAE weights, VAE
    architecture, chunk size, or batch cap invalidates the cache.
    """
    payload = {
        "fold_id": int(fold_id),
        "vae_checkpoint_sha256": str(vae_checkpoint_sha),
        "vae_model_kwargs": dict(config.get("vae", {}).get("model_kwargs", {}) or {}),
        "vae_chunk_size": int(config.get("vae", {}).get("vae_chunk_size", 32)),
        "fit_latent_stats_max_batches": config.get("vae", {}).get(
            "fit_latent_stats_max_batches"
        ),
        "dataset": {
            "epoch_min": config.get("dataset_config", {}).get("epoch_min"),
            "epoch_max_rule": config.get("dataset_config", {}).get(
                "epoch_max_rule", "cross_delivery"
            ),
            "trim_minutes": float(
                config.get("dataset_config", {}).get("trim_minutes", 1.0)
            ),
            "normalize_fields": list(
                config.get("dataset_config", {}).get("normalize_fields") or []
            ),
            "stats_file": _file_signature(
                config.get("dataset_config", {}).get("stats_path")
            ),
            "train_files": [_file_signature(p) for p in train_files],
        },
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_or_fit_latent_stats(
    vae_module: Any,
    *,
    train_underlying: Any,
    config: Dict[str, Any],
    fold_id: int,
    train_files: Sequence[str],
    device: torch.device,
) -> int:
    """Cache-aware wrapper around :meth:`vae.fit_latent_stats`.

    On a cache hit, the persisted ``(mean, var, count)`` tensors are
    copied straight into the VAE's running buffers and the expensive
    encoder pass is skipped. On a miss, the function calls
    ``fit_latent_stats`` as usual and persists the result.

    The cache key is computed from every input that affects the result;
    any change invalidates it. Misuse-resistant: a corrupted / unreadable
    cache file falls back to recomputation rather than failing.

    Args:
        vae_module: The :class:`SeqVaeLagAttnV1` instance whose running
            buffers will be populated.
        train_underlying: The raw segment view of the train dataset
            (passed straight to :class:`DataLoader`; same object the
            previous direct call used).
        config: Parsed YAML config.
        fold_id: Fold id (folded into the cache key).
        train_files: List of HDF5 paths that make up the train partition.
        device: Compute device.

    Returns:
        Number of segment-position samples used for the stats fit
        (matches the legacy :meth:`fit_latent_stats` return).
    """
    from hdf5_dataset.hdf5_dataset import attribute_dict_collate  # noqa: WPS433

    vae_ckpt_path = str(Path(config["vae"]["checkpoint"]).resolve())
    try:
        vae_ckpt_sha = compute_checkpoint_sha256(vae_ckpt_path)
    except Exception as exc:
        logger.warning(
            f"[fold {fold_id}] could not hash VAE checkpoint for cache key "
            f"({exc}); skipping latent-stats cache for this fold."
        )
        vae_ckpt_sha = ""

    cache_key = _latent_stats_cache_key(
        config=config,
        fold_id=fold_id,
        train_files=train_files,
        vae_checkpoint_sha=vae_ckpt_sha,
    )
    cache_dir = _latent_stats_cache_dir(config)
    cache_file = cache_dir / f"{cache_key}.npz"

    if vae_ckpt_sha and cache_file.exists():
        try:
            payload = np.load(str(cache_file))
            mean_t = torch.from_numpy(np.asarray(payload["mean"]))
            var_t = torch.from_numpy(np.asarray(payload["var"]))
            count_val = int(np.asarray(payload["count"]).item())
            with torch.no_grad():
                vae_module.mu_post_running_mean.copy_(
                    mean_t.to(vae_module.mu_post_running_mean.device)
                )
                vae_module.mu_post_running_var.copy_(
                    var_t.to(vae_module.mu_post_running_var.device)
                )
                vae_module.mu_post_running_count.copy_(
                    torch.tensor(
                        count_val,
                        dtype=vae_module.mu_post_running_count.dtype,
                        device=vae_module.mu_post_running_count.device,
                    )
                )
            logger.info(
                f"[fold {fold_id}] vae.fit_latent_stats: loaded from cache "
                f"({cache_file.name}, n={count_val})"
            )
            return count_val
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                f"[fold {fold_id}] latent-stats cache read failed "
                f"({cache_file.name}: {exc}); recomputing."
            )

    stats_loader = DataLoader(
        train_underlying,
        batch_size=int(config["vae"].get("vae_chunk_size", 32)),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=attribute_dict_collate,
    )
    n_stats = vae_module.fit_latent_stats(
        stats_loader,
        max_batches=config["vae"].get("fit_latent_stats_max_batches"),
        device=device,
    )

    if vae_ckpt_sha:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                str(cache_file),
                mean=vae_module.mu_post_running_mean.detach().cpu().numpy(),
                var=vae_module.mu_post_running_var.detach().cpu().numpy(),
                count=np.int64(n_stats),
            )
            logger.info(
                f"[fold {fold_id}] vae.fit_latent_stats: wrote cache "
                f"({cache_file.name}, n={n_stats})"
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                f"[fold {fold_id}] latent-stats cache write failed "
                f"({cache_file}: {exc})"
            )

    return n_stats


def train_fold(
    *,
    fold_id: int,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    gpu_id: int = 0,
    output_dir_override: Optional[str] = None,
    auto_precompute: bool = True,
) -> Dict[str, Any]:
    """Train one fold end-to-end.

    Args:
        fold_id: 1-based fold identifier.
        config: Parsed YAML config dict; takes precedence over ``config_path``.
        config_path: Path to YAML config; required when ``config`` is None.
        gpu_id: Logical GPU id (after ``CUDA_VISIBLE_DEVICES`` is set in the
            calling subprocess this should usually be 0).
        output_dir_override: Optional override for the run output dir.
        auto_precompute: When True (default), missing precompute caches are
            built before training.

    Returns:
        Dict with keys ``fold_id``, ``status``, ``best_checkpoint_path``,
        ``best_val_total_loss``, ``best_val_macro_f1``, ``best_val_auroc``,
        ``train_seconds``, plus a copy of the resolved config.
    """
    if config is None:
        if config_path is None:
            raise ValueError("Either config or config_path must be provided")
        config = _load_config(config_path)

    seed = int(config["general_config"].get("seed", 42)) + int(fold_id)
    import lightning as L  # noqa: WPS433

    L.seed_everything(seed, workers=True)

    run_dir, fold_dir = _resolve_paths(config, fold_id, output_dir_override)

    # Per-fold logging sinks. Attach BEFORE the first informational log so
    # ``run_dir`` / ``fold_dir`` resolution lands in ``fold.log`` too.
    # The matching ``detach_fold_log_sinks`` lives in the ``finally``
    # block of the outer try/except wrapping the rest of this function.
    logging_cfg: Dict[str, Any] = config.get("logging", {}) or {}
    log_sink_handle: Optional[FoldLogHandle] = None
    try:
        log_sink_handle = attach_fold_log_sinks(
            fold_dir,
            log_level=str(logging_cfg.get("log_level", "INFO")),
            capture_stdout_stderr=bool(
                logging_cfg.get("capture_stdout_stderr", True)
            ),
            filename=str(logging_cfg.get("fold_log_filename", "fold.log")),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            f"[fold {fold_id}] attach_fold_log_sinks failed ({exc}); "
            "continuing without persistent log file."
        )

    try:
        return _train_fold_impl(
            fold_id=fold_id,
            config=config,
            gpu_id=gpu_id,
            output_dir_override=output_dir_override,
            auto_precompute=auto_precompute,
            run_dir=run_dir,
            fold_dir=fold_dir,
            logging_cfg=logging_cfg,
            seed=seed,
        )
    finally:
        detach_fold_log_sinks(log_sink_handle)


def _train_fold_impl(
    *,
    fold_id: int,
    config: Dict[str, Any],
    gpu_id: int,
    output_dir_override: Optional[str],
    auto_precompute: bool,
    run_dir: Path,
    fold_dir: Path,
    logging_cfg: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """Inner body of :func:`train_fold` (after sink attach).

    Split out so the public ``train_fold`` can wrap the body in a
    try/finally that guarantees the per-fold loguru sink is removed and
    the original ``sys.stdout`` / ``sys.stderr`` are restored even if
    training raises mid-run. The signature is internal — call
    :func:`train_fold` instead.
    """
    import lightning as L  # noqa: WPS433

    del output_dir_override  # already resolved into ``run_dir`` / ``fold_dir``
    logger.info(f"[fold {fold_id}] run_dir={run_dir} fold_dir={fold_dir}")

    # ------------------------------------------------------------------
    # Precompute caches (frozen-VAE path) / VAE checkpoint load (live)
    # ------------------------------------------------------------------
    freeze_vae = bool(config["vae"].get("freeze_vae", True))
    cls_cfg = config["model_config"]["classifier"]
    ds_cfg = config["dataset_config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_module = None
    # Compute and cache the VAE checkpoint SHA-256 once so ``setup.json``
    # has a reproducibility fingerprint regardless of which path
    # (freeze / live) we take. Defensive against a missing or
    # unreadable checkpoint — the SHA is best-effort metadata, not a
    # correctness gate.
    vae_ckpt_path_str = str(Path(config["vae"]["checkpoint"]).resolve())
    try:
        vae_checkpoint_sha = compute_checkpoint_sha256(vae_ckpt_path_str)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            f"[fold {fold_id}] could not hash VAE checkpoint for setup.json "
            f"({exc}); proceeding without SHA fingerprint."
        )
        vae_checkpoint_sha = ""
    if freeze_vae:
        if auto_precompute:
            # Per-partition epoch_min override (§ evaluation.epoch_min_test):
            # train always uses dataset_config.epoch_min; val/test widen to
            # the eval window when configured. Keeping val in lock-step with
            # test means the threshold-search operating point is calibrated
            # on the same distribution test will use. Matches the live-VAE
            # equivalent below.
            eval_cfg = config.get("evaluation", {}) or {}
            epoch_min_test_cfg = eval_cfg.get("epoch_min_test")
            epoch_min_overrides: Optional[Dict[str, int]] = None
            if epoch_min_test_cfg is not None:
                epoch_min_overrides = {
                    "val": int(epoch_min_test_cfg),
                    "test": int(epoch_min_test_cfg),
                }
            precompute_fold_latents(
                config=config,
                fold_id=fold_id,
                output_root=run_dir,
                device=device,
                epoch_min_overrides=epoch_min_overrides,
            )
    else:
        logger.info(
            f"[fold {fold_id}] live-VAE path enabled — loading checkpoint "
            f"and building raw-segment datasets"
        )
        vae_module = build_vae_from_config(config, device)
        # ``build_vae_from_config`` sets eval() + requires_grad=False on every
        # param. Keep params frozen for stage 1, but flip back into ``train()``
        # mode so encoder dropout stays on as documented in §12.2 ("freeze
        # parameters", not "freeze stochasticity"). The TwoStageVaeUnfreeze
        # callback flips ``requires_grad=True`` on the documented subset at
        # the stage-1 → stage-2 boundary.
        vae_module.train()

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    if freeze_vae:
        cache_root = (
            run_dir
            / config.get("precompute", {}).get("out_subdir", "precomputed_latents")
            / f"fold_{fold_id}"
        )
        train_ds = GuidSequenceDataset(
            cache_root / "train.hdf5",
            warmup_left=int(cls_cfg["warmup_left"]),
            warmup_right=int(cls_cfg["warmup_right"]),
            min_samples_per_guid=int(ds_cfg["min_samples_per_guid"]),
            min_valid_weight_fraction=float(ds_cfg["min_valid_weight_fraction"]),
            cross_delivery_censoring=bool(cls_cfg["cross_delivery_censoring"]),
        )
        val_ds = GuidSequenceDataset(
            cache_root / "val.hdf5",
            warmup_left=int(cls_cfg["warmup_left"]),
            warmup_right=int(cls_cfg["warmup_right"]),
            min_samples_per_guid=int(ds_cfg["min_samples_per_guid"]),
            min_valid_weight_fraction=float(ds_cfg["min_valid_weight_fraction"]),
            cross_delivery_censoring=bool(cls_cfg["cross_delivery_censoring"]),
        )
    else:
        kfold_base_path = ds_cfg["kfold_base_path"]
        test_mode = ds_cfg.get("test_mode")
        train_files = get_fold_partition_files(
            kfold_base_path, fold_id, "train", test_mode=test_mode
        )
        val_files = get_fold_partition_files(
            kfold_base_path, fold_id, "val", test_mode=test_mode
        )
        live_T = int(config["vae"]["model_kwargs"]["sequence_length"])
        live_warmup_left = int(cls_cfg["warmup_left"])
        live_warmup_right = int(cls_cfg["warmup_right"])
        live_min_samples = int(ds_cfg["min_samples_per_guid"])
        live_min_w = float(ds_cfg["min_valid_weight_fraction"])
        live_cross_censor = bool(cls_cfg["cross_delivery_censoring"])
        live_epoch_min = ds_cfg.get("epoch_min")
        # Per-partition window split (§ evaluation.epoch_min_test):
        # train always uses the dataset window; val (and test, at eval
        # time) widen to the eval window when configured. Keeping val
        # in lock-step with test means the threshold-search operating
        # point is calibrated on the same distribution test will use.
        eval_cfg = config.get("evaluation", {}) or {}
        live_epoch_min_val = eval_cfg.get("epoch_min_test", live_epoch_min)
        if live_epoch_min_val is None:
            live_epoch_min_val = live_epoch_min
        live_trim = float(ds_cfg.get("trim_minutes", 1.0))
        live_stats_path = ds_cfg.get("stats_path")
        live_normalize_fields = ds_cfg.get("normalize_fields")
        live_d_model_vae = int(config["vae"]["model_kwargs"]["d_model"])
        live_d_z = int(config["vae"]["model_kwargs"]["d_z"])
        train_ds = LiveGuidSequenceDataset(
            train_files,
            T=live_T,
            warmup_left=live_warmup_left,
            warmup_right=live_warmup_right,
            min_samples_per_guid=live_min_samples,
            min_valid_weight_fraction=live_min_w,
            cross_delivery_censoring=live_cross_censor,
            epoch_min=live_epoch_min,
            trim_minutes=live_trim,
            stats_path=live_stats_path,
            normalize_fields=live_normalize_fields,
            d_model_vae=live_d_model_vae,
            d_z=live_d_z,
        )
        val_ds = LiveGuidSequenceDataset(
            val_files,
            T=live_T,
            warmup_left=live_warmup_left,
            warmup_right=live_warmup_right,
            min_samples_per_guid=live_min_samples,
            min_valid_weight_fraction=live_min_w,
            cross_delivery_censoring=live_cross_censor,
            epoch_min=live_epoch_min_val,
            trim_minutes=live_trim,
            stats_path=live_stats_path,
            normalize_fields=live_normalize_fields,
            d_model_vae=live_d_model_vae,
            d_z=live_d_z,
        )
        if live_epoch_min_val != live_epoch_min:
            logger.info(
                f"[fold {fold_id}] live-VAE val window widened: "
                f"train epoch_min={live_epoch_min} -> "
                f"val epoch_min={live_epoch_min_val} "
                f"(via evaluation.epoch_min_test)"
            )

        # ------------------------------------------------------------------
        # Latent stats: populate vae.mu_post_running_{mean,var,count} once
        # so the live tokenizer can z-score mu_post / mu_prior with the
        # same stats the precompute path bakes into the cache attrs.
        #
        # ``_load_or_fit_latent_stats`` checks a content-addressed cache
        # under ``out_dir_base/_latent_stats_cache/`` and only calls the
        # expensive encoder pass on a miss. The default
        # ``vae.fit_latent_stats_max_batches`` (500) caps the pass so even
        # a cold cache finishes in seconds rather than minutes — running
        # mean/var on a d_z=24 vector converges within ~200 batches.
        # ------------------------------------------------------------------
        # ``vae_checkpoint_sha`` is already captured at the top of this
        # function (works for both freeze and live VAE paths), so we
        # don't need ``_load_or_fit_latent_stats`` to surface it.
        n_stats = _load_or_fit_latent_stats(
            vae_module,
            train_underlying=train_ds._underlying,
            config=config,
            fold_id=fold_id,
            train_files=train_files,
            device=device,
        )
        logger.info(
            f"[fold {fold_id}] vae.fit_latent_stats finished on "
            f"{n_stats} samples"
        )

    # ------------------------------------------------------------------
    # Per-head enable flags (YAML: ``model_config.classifier.heads``).
    # Legacy configs without the ``heads`` block default to both enabled
    # so existing runs are unaffected.
    # ------------------------------------------------------------------
    heads_cfg = cls_cfg.get("heads", {}) or {}
    enable_three_class_head = bool(
        (heads_cfg.get("three_class") or {}).get("enabled", True)
    )
    enable_binary_head = bool(
        (heads_cfg.get("binary") or {}).get("enabled", True)
    )
    if not (enable_three_class_head or enable_binary_head):
        raise ValueError(
            f"[fold {fold_id}] both classifier heads are disabled in "
            "model_config.classifier.heads — at least one head must be "
            "enabled to train."
        )
    logger.info(
        f"[fold {fold_id}] head gating: "
        f"three_class={enable_three_class_head} binary={enable_binary_head}"
    )

    # ------------------------------------------------------------------
    # Class weights from the *train* fold only — mode-aware (§18.17.3 B).
    # Skip the disabled head: passing ``None`` into ``PlGuidClassifier``
    # is the documented "no class weighting" path on that branch.
    # ------------------------------------------------------------------
    train_cfg_for_loss = (config.get("training", {}) or {}).get("loss", {}) or {}
    class_weight_mode = str(
        train_cfg_for_loss.get("class_weight_mode", "none")
    ).lower()
    train_labels_3 = train_ds.get_guid_labels_3class()
    train_labels_bin = train_ds.get_guid_labels_binary()
    cls_w_3: Optional[torch.Tensor] = (
        estimate_class_weights_3(train_labels_3, mode=class_weight_mode)
        if enable_three_class_head
        else None
    )
    cls_w_bin: Optional[torch.Tensor] = (
        estimate_class_weights_bin(train_labels_bin, mode=class_weight_mode)
        if enable_binary_head
        else None
    )
    logger.info(
        f"[fold {fold_id}] class_weight_mode={class_weight_mode!r} "
        f"3-class={cls_w_3.tolist() if cls_w_3 is not None else 'disabled'} "
        f"binary={cls_w_bin.tolist() if cls_w_bin is not None else 'disabled'}"
    )

    # Empirical priors used for head bias init (§18.17.3 D). Only
    # computed for enabled heads; the bias-init step downstream skips
    # the disabled head whether or not the prior tensor is present.
    prior_3: Optional[torch.Tensor] = (
        class_priors_3(train_labels_3) if enable_three_class_head else None
    )
    prior_bin: Optional[torch.Tensor] = (
        class_prior_bin(train_labels_bin) if enable_binary_head else None
    )
    logger.info(
        f"[fold {fold_id}] train priors: "
        f"3-class={prior_3.tolist() if prior_3 is not None else 'disabled'} "
        f"binary_pos={f'{float(prior_bin):.4f}' if prior_bin is not None else 'disabled'}"
    )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    dl_cfg = config.get("dataloader_config", {}) or {}
    bucket_bs = _bucket_batch_sizes(config["general_config"])
    cls_cfg_for_collate = config["model_config"]["classifier"]
    train_loader = _make_train_dataloader(
        train_ds,
        bucket_batch_sizes=bucket_bs,
        num_workers=int(dl_cfg.get("num_workers", 4)),
        prefetch_factor=int(dl_cfg.get("prefetch_factor", 4)),
        persistent_workers=bool(dl_cfg.get("persistent_workers", True)),
        pin_memory=bool(dl_cfg.get("pin_memory", False)),
        mp_context=str(dl_cfg.get("mp_context", "spawn")),
        seed=seed,
        cls_cfg=cls_cfg_for_collate,
    )
    eval_batch = max(bs for _, bs in bucket_bs)
    val_loader = _make_eval_dataloader(
        val_ds,
        batch_size=eval_batch,
        num_workers=int(dl_cfg.get("num_workers", 4)),
        prefetch_factor=int(dl_cfg.get("prefetch_factor", 4)),
        pin_memory=bool(dl_cfg.get("pin_memory", False)),
        mp_context=str(dl_cfg.get("mp_context", "spawn")),
        cls_cfg=cls_cfg_for_collate,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    head_hidden_raw = cls_cfg.get("head_hidden_dim")
    classifier_cfg = GuidClassifierConfig(
        d_model_vae=int(train_ds.d_model_vae),
        d_z=int(train_ds.d_z),
        d_seg=int(cls_cfg.get("d_seg", 192)),
        d_model=int(cls_cfg.get("d_model", 256)),
        n_layers=int(cls_cfg.get("n_layers", 3)),
        n_heads=int(cls_cfg.get("n_heads", 4)),
        d_head=int(cls_cfg.get("d_head", 64)),
        d_ff=int(cls_cfg.get("d_ff", 512)),
        n_rel_buckets=int(cls_cfg.get("n_rel_buckets", 32)),
        num_classes_multi=int(cls_cfg.get("num_classes_multi", 3)),
        head_hidden_dim=int(head_hidden_raw) if head_hidden_raw is not None else None,
        causal=bool(cls_cfg.get("causal", True)),
        c_meta_dim=5,
        te_summary_dim=6,
        late_window_steps=75,
        dropout=float(cls_cfg.get("dropout", 0.1)),
        enable_three_class_head=enable_three_class_head,
        enable_binary_head=enable_binary_head,
    )
    classifier = GuidOutcomeClassifier(classifier_cfg)
    if not freeze_vae:
        # Attach the VAE submodule so ``forward`` dispatches to
        # ``live_forward`` and ``build_optimizer`` produces a two-group
        # AdamW (low-LR group activated when params unfreeze at stage 2).
        classifier.vae = vae_module
        classifier.vae_chunk_size = int(config["vae"].get("vae_chunk_size", 32))

    train_cfg = config.get("training", {}) or {}
    loss_cfg = train_cfg.get("loss", {}) or {}
    # Head bias init from the empirical class prior (§18.17.3 D). On by
    # default — pairs with ``class_weight_mode: none`` to put position 0
    # at the prior at step 0 instead of forcing the optimizer to climb
    # out of a uniform-init basin under bursty rare-class gradients.
    # The init helper already no-ops on a None prior, but we also skip
    # the entire call when neither head is enabled-and-present (cheap
    # defence against an empty info-log).
    head_bias_init_3: Optional[List[float]] = None
    head_bias_init_bin: Optional[float] = None
    if bool(loss_cfg.get("head_bias_init_from_prior", True)) and (
        prior_3 is not None or prior_bin is not None
    ):
        classifier.outcome_head.init_class_bias_from_prior(
            prior_3=prior_3,
            prior_bin=prior_bin,
        )
        bias3_str = (
            f"log({prior_3.tolist()})" if prior_3 is not None else "disabled"
        )
        bias_bin_str = (
            f"logit({float(prior_bin):.4f})"
            if prior_bin is not None
            else "disabled"
        )
        logger.info(
            f"[fold {fold_id}] head bias-init from prior: "
            f"head_3.bias={bias3_str} head_bin.bias={bias_bin_str}"
        )
        # Snapshot the actual bias values that landed on the head so
        # ``setup.json`` is a complete record of the step-0 state.
        # ``init_class_bias_from_prior`` writes ``log(prior + ε)`` /
        # ``logit(prior + ε)`` — pulling the values back off the head
        # records the post-clamp result rather than recomputing.
        head = classifier.outcome_head
        if prior_3 is not None:
            head_3_module = getattr(head, "head_3", None)
            bias_3 = getattr(head_3_module, "bias", None) if head_3_module is not None else None
            if bias_3 is not None:
                try:
                    head_bias_init_3 = bias_3.detach().cpu().tolist()
                except Exception:  # pragma: no cover - defensive
                    head_bias_init_3 = None
        if prior_bin is not None:
            head_bin_module = getattr(head, "head_bin", None)
            bias_bin = getattr(head_bin_module, "bias", None) if head_bin_module is not None else None
            if bias_bin is not None:
                try:
                    head_bias_init_bin = float(bias_bin.detach().cpu().item())
                except Exception:  # pragma: no cover - defensive
                    head_bias_init_bin = None

    loss_weights = LossWeights(
        lambda_3=float(loss_cfg.get("lambda_3", 1.0)),
        lambda_2=float(loss_cfg.get("lambda_2", 0.5)),
        gamma_vae=0.0 if freeze_vae else float(loss_cfg.get("gamma_vae", 0.1)),
        lambda_sp=0.0 if freeze_vae else float(loss_cfg.get("lambda_sp", 1e-4)),
        position_weight_alpha=float(loss_cfg.get("position_weight_alpha", 0.0)),
        loss_warmup_positions=int(loss_cfg.get("loss_warmup_positions", 0)),
        enable_three_class=enable_three_class_head,
        enable_binary=enable_binary_head,
    )
    pl_model = PlGuidClassifier(
        base_model=classifier,
        loss_weights=loss_weights,
        class_weights_3=cls_w_3,
        class_weights_bin=cls_w_bin,
        segment_dropout_p=float(
            train_cfg.get("segment_dropout", {}).get("p", 0.1)
        ),
        segment_dropout_enabled=bool(
            train_cfg.get("segment_dropout", {}).get("enabled", True)
        ),
        rel_num_buckets=int(cls_cfg.get("n_rel_buckets", 32)),
        rel_d_max=float(cls_cfg.get("rel_bucket_d_max", 40.0)),
        lr=float(train_cfg.get("optimizer", {}).get("classifier", {}).get("lr", 1e-3)),
        lr_milestones=train_cfg.get("scheduler", {}).get("milestones") or [100],
        lr_gamma=float(train_cfg.get("scheduler", {}).get("gamma", 0.1)),
        lr_warmup_steps=int(train_cfg.get("lr_warmup_steps", 0)),
        weight_decay=float(
            train_cfg.get("optimizer", {}).get("classifier", {}).get("weight_decay", 1e-4)
        ),
        vae_lr=float(train_cfg.get("optimizer", {}).get("vae", {}).get("lr", 1e-5)),
    )

    # ------------------------------------------------------------------
    # Lightning Trainer
    # ------------------------------------------------------------------
    callbacks = _build_callbacks(
        fold_dir=fold_dir,
        monitor=str(train_cfg.get("monitor", "val/total_loss")),
        save_top_k=int(train_cfg.get("checkpoint", {}).get("save_top_k", 1)),
        es_patience=int(train_cfg.get("early_stopping", {}).get("patience", 50)),
        es_enabled=bool(train_cfg.get("early_stopping", {}).get("enabled", True)),
        logging_cfg=logging_cfg,
    )
    # Stage-1 → stage-2 boundary epoch is also needed by ``min_epochs``
    # below so EarlyStopping cannot terminate the run before stage 2 has
    # had a chance to fire.
    stage1_epochs: Optional[int] = None
    if not freeze_vae:
        two_stage_cfg = train_cfg.get("two_stage", {}) or {}
        stage1_epochs = int(two_stage_cfg.get("stage1_epochs", 100))
        stage_transitions_path = (
            fold_dir / "logs" / "stage_transitions.jsonl"
        )
        callbacks.append(
            TwoStageVaeUnfreeze(
                stage1_epochs=stage1_epochs,
                gamma_vae_stage2=float(loss_cfg.get("gamma_vae", 0.1)),
                lambda_sp_stage2=float(loss_cfg.get("lambda_sp", 1e-4)),
                log_path=stage_transitions_path,
            )
        )
        logger.info(
            f"[fold {fold_id}] TwoStageVaeUnfreeze callback registered "
            f"(stage1_epochs={stage1_epochs}, transitions log -> "
            f"{stage_transitions_path})"
        )

    # Live-VAE runs MUST reach stage 2: pin ``min_epochs = stage1_epochs + 1``
    # so EarlyStopping cannot fire during stage 1 even if val/total_loss
    # plateaus early. The ``+1`` ensures stage 2 gets at least one epoch
    # before EarlyStopping is allowed to evaluate it (and the
    # :class:`TwoStageVaeUnfreeze` callback also resets EarlyStopping's
    # ``best_score`` / ``wait_count`` at the boundary so stage 2 starts
    # with a fresh patience window).
    max_epochs_cfg = int(config["general_config"].get("epochs", 200))
    min_epochs: Optional[int] = (
        (stage1_epochs + 1) if (not freeze_vae and stage1_epochs is not None) else None
    )
    if min_epochs is not None and min_epochs > max_epochs_cfg:
        # Lightning silently caps min_epochs at max_epochs, which would
        # make the run finish at ``epochs`` without ever entering stage
        # 2. Fail loudly so the misconfiguration is caught at fold start.
        raise ValueError(
            f"two_stage.stage1_epochs={stage1_epochs} requires "
            f"min_epochs={min_epochs}, but general_config.epochs="
            f"{max_epochs_cfg} is smaller — stage 2 would never run. "
            "Either lower stage1_epochs or raise epochs."
        )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [int(gpu_id)] if accelerator == "gpu" else 1

    # CSV logger: per-step + per-epoch scalars persisted to
    # ``fold_{k}/logs/metrics.csv``. Disable via
    # ``logging.csv_logger.enabled: false``. ``name=""`` / ``version=""``
    # keeps Lightning from inserting an extra ``lightning_logs/version_0``
    # layer — the file lands directly under ``logs/``.
    pl_logger: Any = False
    csv_cfg = (logging_cfg or {}).get("csv_logger", {}) or {}
    if bool(csv_cfg.get("enabled", True)):
        from lightning.pytorch.loggers import CSVLogger  # noqa: WPS433

        pl_logger = CSVLogger(
            save_dir=str(fold_dir / "logs"),
            name="",
            version="",
            flush_logs_every_n_steps=int(
                csv_cfg.get("flush_logs_every_n_steps", 50)
            ),
        )

    trainer = L.Trainer(
        max_epochs=max_epochs_cfg,
        min_epochs=min_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=str(train_cfg.get("precision", "32-true")),
        gradient_clip_val=float(train_cfg.get("grad_clip_val", 1.0)),
        gradient_clip_algorithm=str(train_cfg.get("grad_clip_algorithm", "norm")),
        callbacks=callbacks,
        logger=pl_logger,
        log_every_n_steps=1,
        accumulate_grad_batches=int(
            config["general_config"].get("accumulate_grad_batches", 1)
        ),
        benchmark=bool(config.get("advanced_config", {}).get("trainer", {}).get("benchmark", True)),
        deterministic=bool(
            config.get("advanced_config", {}).get("trainer", {}).get("deterministic", False)
        ),
        num_sanity_val_steps=0,
    )

    # Snapshot the resolved config alongside the run.
    (fold_dir / "config.yaml").write_text(
        yaml.dump(config, sort_keys=False, default_flow_style=False), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Setup snapshot — written once, just before fit, so post-mortem
    # debugging has a single file with every input that determined the
    # run (sizes, class balance, priors, weights, optimizer cfg, head
    # bias-init values, VAE checkpoint SHA, trainer cfg). Distinct from
    # ``config.yaml`` (resolved config tree) and ``fold_results.json``
    # (training outputs) — see plan §4.4.
    # ------------------------------------------------------------------
    started = datetime.now(timezone.utc)
    setup_filename = str(
        (logging_cfg or {}).get("setup_json", "setup.json")
    )
    setup_path = fold_dir / "logs" / setup_filename
    try:
        setup_snapshot = _build_setup_snapshot(
            fold_id=fold_id,
            seed=seed,
            started_utc=started,
            run_dir=run_dir,
            fold_dir=fold_dir,
            config=config,
            train_ds=train_ds,
            val_ds=val_ds,
            train_labels_3=train_labels_3,
            train_labels_bin=train_labels_bin,
            class_weight_mode=class_weight_mode,
            cls_w_3=cls_w_3,
            cls_w_bin=cls_w_bin,
            prior_3=prior_3,
            prior_bin=prior_bin,
            head_bias_init_3=head_bias_init_3,
            head_bias_init_bin=head_bias_init_bin,
            enable_three_class_head=enable_three_class_head,
            enable_binary_head=enable_binary_head,
            freeze_vae=freeze_vae,
            vae_checkpoint_sha=vae_checkpoint_sha,
            vae_checkpoint_path=vae_ckpt_path_str,
            classifier_cfg=classifier_cfg,
            loss_weights=loss_weights,
            train_cfg=train_cfg,
            bucket_bs=bucket_bs,
            dl_cfg=dl_cfg,
            stage1_epochs=stage1_epochs,
            min_epochs=min_epochs,
            max_epochs=max_epochs_cfg,
            accelerator=accelerator,
            devices=devices,
        )
        dump_json(setup_path, setup_snapshot)
        logger.info(f"[fold {fold_id}] setup snapshot -> {setup_path}")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            f"[fold {fold_id}] failed to write setup snapshot to {setup_path}: {exc}"
        )

    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    train_seconds = (datetime.now(timezone.utc) - started).total_seconds()

    # ------------------------------------------------------------------
    # Best checkpoint resolution
    # ------------------------------------------------------------------
    best_path = ""
    best_score: Optional[float] = None
    ckpt_cb = next(
        (cb for cb in trainer.callbacks if cb.__class__.__name__ == "ModelCheckpoint"), None
    )
    if ckpt_cb is not None:
        best_path = str(ckpt_cb.best_model_path or "")
        if ckpt_cb.best_model_score is not None:
            best_score = float(ckpt_cb.best_model_score.item())

    best_metrics: Dict[str, float] = {}
    if best_path:
        try:
            val_results = trainer.validate(
                pl_model, dataloaders=val_loader, ckpt_path=best_path, verbose=False
            )
            if val_results:
                best_metrics = {
                    str(k): float(v) for k, v in val_results[0].items()
                }
        except Exception as exc:  # pragma: no cover - validation may fail in degenerate runs
            logger.warning(f"[fold {fold_id}] best-ckpt validation failed: {exc}")

    result = {
        "fold_id": int(fold_id),
        "status": "ok",
        "best_checkpoint_path": best_path,
        "best_val_total_loss": best_score,
        "train_seconds": train_seconds,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "best_metrics": best_metrics,
        "config": config,
    }
    (fold_dir / "fold_results.json").write_text(
        json.dumps(_to_json_safe(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info(
        f"[fold {fold_id}] training done in {train_seconds:.1f}s; "
        f"best val/total_loss={best_score} ckpt={best_path}"
    )
    return result


def _to_json_safe(obj: Any) -> Any:
    """Compatibility shim — delegates to :func:`logging_utils.to_json_safe`.

    Kept under the original private name so callers inside this module
    can keep their existing call sites; new callers should use
    :func:`logging_utils.to_json_safe` directly.
    """
    return to_json_safe(obj)


def _class_distribution(labels: Sequence[int], num_classes: int) -> List[int]:
    """Per-class GUID counts.

    Args:
        labels: Sequence of integer class labels (GUID level).
        num_classes: Total number of classes (3 for 3-class, 2 for binary).

    Returns:
        Length-``num_classes`` list where entry ``i`` is the number of
        GUIDs labelled class ``i``.
    """
    counts = [0] * int(num_classes)
    for lab in labels:
        idx = int(lab)
        if 0 <= idx < num_classes:
            counts[idx] += 1
    return counts


def _build_setup_snapshot(
    *,
    fold_id: int,
    seed: int,
    started_utc: datetime,
    run_dir: Path,
    fold_dir: Path,
    config: Dict[str, Any],
    train_ds: Any,
    val_ds: Any,
    train_labels_3: Sequence[int],
    train_labels_bin: Sequence[int],
    class_weight_mode: str,
    cls_w_3: Optional[torch.Tensor],
    cls_w_bin: Optional[torch.Tensor],
    prior_3: Optional[torch.Tensor],
    prior_bin: Optional[torch.Tensor],
    head_bias_init_3: Optional[List[float]],
    head_bias_init_bin: Optional[float],
    enable_three_class_head: bool,
    enable_binary_head: bool,
    freeze_vae: bool,
    vae_checkpoint_sha: str,
    vae_checkpoint_path: str,
    classifier_cfg: GuidClassifierConfig,
    loss_weights: LossWeights,
    train_cfg: Dict[str, Any],
    bucket_bs: List[Tuple[Tuple[int, int], int]],
    dl_cfg: Dict[str, Any],
    stage1_epochs: Optional[int],
    min_epochs: Optional[int],
    max_epochs: int,
    accelerator: str,
    devices: Any,
) -> Dict[str, Any]:
    """Build the structured per-fold setup snapshot.

    The output is what eventually lands in ``fold_{k}/logs/setup.json``.
    Captures every input that determines the training trajectory so
    a post-mortem diff between two runs can identify what changed
    without re-loading the config.
    """
    # GUID-level segment counts — ``guid_lengths`` is exposed on both
    # ``GuidSequenceDataset`` (precomputed cache) and
    # ``LiveGuidSequenceDataset`` (live VAE path).
    def _seg_counts(dataset: Any) -> Tuple[int, int]:
        n_guids = int(len(dataset))
        n_segments = 0
        try:
            n_segments = int(sum(dataset.guid_lengths))
        except Exception:  # pragma: no cover - defensive
            n_segments = -1
        return n_guids, n_segments

    train_n_guids, train_n_segments = _seg_counts(train_ds)
    val_n_guids, val_n_segments = _seg_counts(val_ds)

    snapshot: Dict[str, Any] = {
        "fold_id": int(fold_id),
        "seed": int(seed),
        "started_utc": started_utc.isoformat(),
        "run_dir": str(run_dir),
        "fold_dir": str(fold_dir),
        "freeze_vae": bool(freeze_vae),
        "vae": {
            "checkpoint_path": str(vae_checkpoint_path),
            "checkpoint_sha256": str(vae_checkpoint_sha or ""),
            "freeze_vae": bool(freeze_vae),
            "vae_chunk_size": int(config.get("vae", {}).get("vae_chunk_size", 32)),
            "fit_latent_stats_max_batches": config.get("vae", {}).get(
                "fit_latent_stats_max_batches"
            ),
        },
        "dataset": {
            "train_n_guids": int(train_n_guids),
            "train_n_segments": int(train_n_segments),
            "val_n_guids": int(val_n_guids),
            "val_n_segments": int(val_n_segments),
            "class_distribution_3class": _class_distribution(train_labels_3, 3),
            "class_distribution_binary": _class_distribution(train_labels_bin, 2),
            "min_samples_per_guid": int(
                config["dataset_config"].get("min_samples_per_guid", 3)
            ),
            "min_valid_weight_fraction": float(
                config["dataset_config"].get("min_valid_weight_fraction", 0.1)
            ),
            "trim_minutes": float(
                config["dataset_config"].get("trim_minutes", 1.0)
            ),
            "epoch_min": config["dataset_config"].get("epoch_min"),
            "epoch_max_rule": config["dataset_config"].get("epoch_max_rule"),
            "normalize_fields": list(
                config["dataset_config"].get("normalize_fields") or []
            ),
        },
        "heads": {
            "three_class_enabled": bool(enable_three_class_head),
            "binary_enabled": bool(enable_binary_head),
        },
        "class_weights": {
            "mode": str(class_weight_mode),
            "three_class": (
                cls_w_3.detach().cpu().tolist() if cls_w_3 is not None else None
            ),
            "binary": (
                cls_w_bin.detach().cpu().tolist() if cls_w_bin is not None else None
            ),
        },
        "class_priors": {
            "three_class": (
                prior_3.detach().cpu().tolist() if prior_3 is not None else None
            ),
            "binary_pos": (
                float(prior_bin.item()) if prior_bin is not None else None
            ),
        },
        "head_bias_init": {
            "head_3": head_bias_init_3,
            "head_bin": head_bias_init_bin,
        },
        "loss_weights": {
            "lambda_3": float(loss_weights.lambda_3),
            "lambda_2": float(loss_weights.lambda_2),
            "gamma_vae": float(loss_weights.gamma_vae),
            "lambda_sp": float(loss_weights.lambda_sp),
            "position_weight_alpha": float(loss_weights.position_weight_alpha),
            "loss_warmup_positions": int(loss_weights.loss_warmup_positions),
            "enable_three_class": bool(loss_weights.enable_three_class),
            "enable_binary": bool(loss_weights.enable_binary),
        },
        "classifier_config": {
            "d_model_vae": int(classifier_cfg.d_model_vae),
            "d_z": int(classifier_cfg.d_z),
            "d_seg": int(classifier_cfg.d_seg),
            "d_model": int(classifier_cfg.d_model),
            "n_layers": int(classifier_cfg.n_layers),
            "n_heads": int(classifier_cfg.n_heads),
            "d_head": int(classifier_cfg.d_head),
            "d_ff": int(classifier_cfg.d_ff),
            "n_rel_buckets": int(classifier_cfg.n_rel_buckets),
            "num_classes_multi": int(classifier_cfg.num_classes_multi),
            "head_hidden_dim": classifier_cfg.head_hidden_dim,
            "causal": bool(classifier_cfg.causal),
            "dropout": float(classifier_cfg.dropout),
        },
        "optimizer": {
            "classifier_lr": float(
                train_cfg.get("optimizer", {}).get("classifier", {}).get("lr", 1e-3)
            ),
            "classifier_weight_decay": float(
                train_cfg.get("optimizer", {}).get("classifier", {}).get("weight_decay", 1e-4)
            ),
            "vae_lr": float(
                train_cfg.get("optimizer", {}).get("vae", {}).get("lr", 1e-5)
            ),
            "vae_weight_decay": float(
                train_cfg.get("optimizer", {}).get("vae", {}).get("weight_decay", 0.0)
            ),
            "scheduler_milestones": list(
                train_cfg.get("scheduler", {}).get("milestones") or []
            ),
            "scheduler_gamma": float(
                train_cfg.get("scheduler", {}).get("gamma", 0.1)
            ),
            "lr_warmup_steps": int(train_cfg.get("lr_warmup_steps", 0)),
        },
        "trainer": {
            "max_epochs": int(max_epochs),
            "min_epochs": min_epochs,
            "accelerator": str(accelerator),
            "devices": devices,
            "precision": str(train_cfg.get("precision", "32-true")),
            "grad_clip_val": float(train_cfg.get("grad_clip_val", 1.0)),
            "grad_clip_algorithm": str(train_cfg.get("grad_clip_algorithm", "norm")),
            "accumulate_grad_batches": int(
                config["general_config"].get("accumulate_grad_batches", 1)
            ),
            "benchmark": bool(
                config.get("advanced_config", {}).get("trainer", {}).get("benchmark", True)
            ),
            "deterministic": bool(
                config.get("advanced_config", {}).get("trainer", {}).get("deterministic", False)
            ),
            "stage1_epochs": stage1_epochs,
        },
        "dataloader": {
            "bucket_batch_sizes": [
                {"range": list(rng), "batch_size": int(bs)} for rng, bs in bucket_bs
            ],
            "num_workers": int(dl_cfg.get("num_workers", 4)),
            "prefetch_factor": int(dl_cfg.get("prefetch_factor", 4)),
            "persistent_workers": bool(dl_cfg.get("persistent_workers", True)),
            "pin_memory": bool(dl_cfg.get("pin_memory", False)),
            "mp_context": str(dl_cfg.get("mp_context", "spawn")),
        },
    }
    return snapshot


__all__ = ["train_fold"]
