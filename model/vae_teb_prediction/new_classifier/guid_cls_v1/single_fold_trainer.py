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
from model.vae_teb_prediction.new_classifier.guid_cls_v1.collate import (
    guid_sequence_collate_fn,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_classifier import (
    GuidClassifierConfig,
    GuidOutcomeClassifier,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_dataset import (
    GuidSequenceDataset,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.lightning_module import (
    PlGuidClassifier,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.losses import (
    LossWeights,
    estimate_inverse_frequency_class_weights_3,
    estimate_inverse_frequency_class_weights_bin,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.precompute_latents import (
    _resolve_run_dir,
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
    )


def _build_callbacks(
    *,
    fold_dir: Path,
    monitor: str,
    save_top_k: int,
    es_patience: int,
    es_enabled: bool,
):
    """Build the standard callback list (ModelCheckpoint + EarlyStopping +
    repo-shared LossPlot/Hparam/Metrics callbacks).
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

    callbacks = [
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
    logger.info(f"[fold {fold_id}] run_dir={run_dir} fold_dir={fold_dir}")

    # ------------------------------------------------------------------
    # Precompute caches (frozen-VAE path)
    # ------------------------------------------------------------------
    freeze_vae = bool(config["vae"].get("freeze_vae", True))
    if not freeze_vae:
        # Live-VAE / two-stage fine-tune mode (PRD §8.4 + Phase 7 of the
        # implementation roadmap) is not yet wired. The supporting
        # infrastructure is already in place:
        #   * LossWeights.gamma_vae and lambda_sp control the stage-2 VAE
        #     auxiliary loss + L2 sparsity terms.
        #   * PlGuidClassifier.build_optimizer creates two parameter groups
        #     when a ``vae`` attribute exists on the classifier, applying
        #     ``training.optimizer.vae.lr`` (default 1e-5) to that group.
        # To finish Phase 7 you need: (1) a ``LiveGuidSequenceDataset`` that
        # returns raw per-segment fields per GUID (fhr_st / fhr_ph / up_st /
        # up_ph / weight / target / epoch / tlo / sso / cs_label / bg_label),
        # (2) a ``LiveGuidOutcomeClassifier`` that owns a ``SeqVaeLagAttnV1``
        # module and routes raw batches through ``vae.encode_only`` before
        # the existing tokenizer/transformer/heads, (3) a Lightning callback
        # that toggles ``requires_grad`` on the VAE encoder + adapters +
        # prior + posterior heads at ``training.two_stage.stage1_epochs``
        # and refreshes the optimizer.
        raise NotImplementedError(
            "freeze_vae=False (live VAE + two-stage fine-tune) is scaffolded "
            "but not yet wired end-to-end. See the comment block in "
            "single_fold_trainer.train_fold for the implementation outline. "
            "Set vae.freeze_vae=true to use the precompute path."
        )
    if auto_precompute:
        precompute_fold_latents(
            config=config,
            fold_id=fold_id,
            output_root=run_dir,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    cache_root = (
        run_dir
        / config.get("precompute", {}).get("out_subdir", "precomputed_latents")
        / f"fold_{fold_id}"
    )
    train_ds = GuidSequenceDataset(
        cache_root / "train.hdf5",
        warmup_left=int(config["model_config"]["classifier"]["warmup_left"]),
        warmup_right=int(config["model_config"]["classifier"]["warmup_right"]),
        min_samples_per_guid=int(config["dataset_config"]["min_samples_per_guid"]),
        min_valid_weight_fraction=float(
            config["dataset_config"]["min_valid_weight_fraction"]
        ),
        cross_delivery_censoring=bool(
            config["model_config"]["classifier"]["cross_delivery_censoring"]
        ),
    )
    val_ds = GuidSequenceDataset(
        cache_root / "val.hdf5",
        warmup_left=int(config["model_config"]["classifier"]["warmup_left"]),
        warmup_right=int(config["model_config"]["classifier"]["warmup_right"]),
        min_samples_per_guid=int(config["dataset_config"]["min_samples_per_guid"]),
        min_valid_weight_fraction=float(
            config["dataset_config"]["min_valid_weight_fraction"]
        ),
        cross_delivery_censoring=bool(
            config["model_config"]["classifier"]["cross_delivery_censoring"]
        ),
    )

    # ------------------------------------------------------------------
    # Class weights from the *train* fold only
    # ------------------------------------------------------------------
    cls_w_3 = estimate_inverse_frequency_class_weights_3(
        train_ds.get_guid_labels_3class()
    )
    cls_w_bin = estimate_inverse_frequency_class_weights_bin(
        train_ds.get_guid_labels_binary()
    )
    logger.info(
        f"[fold {fold_id}] class weights: 3-class={cls_w_3.tolist()} "
        f"binary={cls_w_bin.tolist()}"
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
    cls_cfg = config["model_config"]["classifier"]
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
    )
    classifier = GuidOutcomeClassifier(classifier_cfg)

    train_cfg = config.get("training", {}) or {}
    loss_cfg = train_cfg.get("loss", {}) or {}
    loss_weights = LossWeights(
        lambda_3=float(loss_cfg.get("lambda_3", 1.0)),
        lambda_2=float(loss_cfg.get("lambda_2", 0.5)),
        gamma_vae=0.0 if freeze_vae else float(loss_cfg.get("gamma_vae", 0.1)),
        lambda_sp=0.0 if freeze_vae else float(loss_cfg.get("lambda_sp", 1e-4)),
        position_weight_alpha=float(loss_cfg.get("position_weight_alpha", 0.0)),
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
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [int(gpu_id)] if accelerator == "gpu" else 1
    trainer = L.Trainer(
        max_epochs=int(config["general_config"].get("epochs", 200)),
        accelerator=accelerator,
        devices=devices,
        precision=str(train_cfg.get("precision", "32-true")),
        gradient_clip_val=float(train_cfg.get("grad_clip_val", 1.0)),
        gradient_clip_algorithm=str(train_cfg.get("grad_clip_algorithm", "norm")),
        callbacks=callbacks,
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

    started = datetime.now(timezone.utc)
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
    """Convert numpy/torch scalars and unsupported types to JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.item())
        return obj.tolist()
    return obj


__all__ = ["train_fold"]
