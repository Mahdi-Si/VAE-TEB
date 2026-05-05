"""Per-fold precompute of frozen-VAE latent caches for ``guid_cls_v1``.

For each fold ``k`` and partition ``p ∈ {train, val, test}`` this script:

1. Builds a segment-level :class:`CombinedHDF5Dataset` over the fold's HDF5
   files with the same filters the classifier will use.
2. Loads :class:`SeqVaeLagAttnV1` with the pretrained checkpoint.
3. (For the train partition only) runs ``vae.fit_latent_stats`` to obtain
   exact per-fold ``mu_post`` mean / variance.
4. Iterates the segment loader and calls ``vae.encode_only`` with
   ``sample_z=False``. Collects per-segment ``h_y`` (fp16), ``mu_prior``,
   ``mu_post`` (raw), ``kld_per_t`` and ``mean_alpha = attn_weights.mean(-2)``
   (fp16).
5. Groups results by GUID (sorted by epoch) and writes one HDF5 per partition
   under ``<run_dir>/precomputed_latents/fold_{k}/``.

The cache is *idempotent*: a partition is reused only when the stored cache
input signature matches the current precompute inputs (checkpoint, VAE config,
source HDF5 file identities, stats file identity, and dataset filters).

CLI::

    python -m model.vae_teb_prediction.new_classifier.guid_cls_v1.precompute_latents \
        --config <path>/config_guid_cls_v1.yaml --fold 3
    python -m model.vae_teb_prediction.new_classifier.guid_cls_v1.precompute_latents \
        --config <path>/config_guid_cls_v1.yaml --all-folds
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from hdf5_dataset.hdf5_dataset import (
    AttributeDict,
    CombinedHDF5Dataset,
    attribute_dict_collate,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1
from train.graph_models_utils import load_checkpoint_strict


SCHEMA_VERSION = "v1"
DEFAULT_LOAD_FIELDS: Tuple[str, ...] = (
    "fhr_st",
    "fhr_ph",
    "up_st",
    "up_ph",
    "weight",
    "target",
    "epoch",
    "cs_label",
    "bg_label",
    "guid",
    "time_from_labor_onset",
    "second_stage_onset",
)


def _json_safe(obj: Any) -> Any:
    """Recursively convert config/file metadata into stable JSON primitives."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj.resolve())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _file_signature(path: Optional[str | Path]) -> Dict[str, Any]:
    """Cheap file identity for cache invalidation.

    Uses resolved path + file size + mtime_ns rather than hashing the full file.
    This is strong enough to catch practical changes in the source HDF5s / stats
    file without incurring the cost of hashing multi-GB datasets.
    """
    if path is None:
        return {"path": None, "exists": False}
    resolved = Path(path).resolve()
    exists = resolved.exists()
    sig: Dict[str, Any] = {
        "path": str(resolved),
        "exists": bool(exists),
    }
    if exists:
        stat = resolved.stat()
        sig["size"] = int(stat.st_size)
        sig["mtime_ns"] = int(stat.st_mtime_ns)
    return sig


def build_cache_input_summary(
    *,
    config: Dict[str, Any],
    fold_id: int,
    partition: str,
    files: Sequence[str],
    checkpoint_sha256: str,
    epoch_min_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Return the semantic inputs that determine one cache's contents.

    Args:
        epoch_min_override: When provided, this value (instead of
            ``dataset_config.epoch_min``) is baked into the signature and
            recorded in the cache. Used by the per-partition window-split
            path (``evaluation.epoch_min_test``) so val/test caches can
            be invalidated independently of the train cache when the eval
            window changes.
    """
    ds_cfg = config.get("dataset_config", {}) or {}
    vae_cfg = config.get("vae", {}) or {}
    precompute_cfg = config.get("precompute", {}) or {}
    effective_epoch_min = (
        epoch_min_override
        if epoch_min_override is not None
        else ds_cfg.get("epoch_min")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "fold_id": int(fold_id),
        "partition": str(partition),
        "vae_checkpoint_sha256": str(checkpoint_sha256),
        "vae_checkpoint_path": str(Path(vae_cfg["checkpoint"]).resolve()),
        "vae_model_kwargs": _json_safe(dict(vae_cfg.get("model_kwargs", {}))),
        "dataset": {
            "epoch_min": effective_epoch_min,
            "epoch_max_rule": ds_cfg.get("epoch_max_rule", "cross_delivery"),
            "trim_minutes": float(ds_cfg.get("trim_minutes", 1.0)),
            "normalize_fields": _json_safe(
                list(
                    ds_cfg.get("normalize_fields")
                    or ["fhr_st", "fhr_ph", "up_st", "up_ph"]
                )
            ),
            "stats_file": _file_signature(ds_cfg.get("stats_path")),
            "source_files": [_file_signature(p) for p in files],
        },
        "precompute_storage": {
            "h_y_dtype": str(precompute_cfg.get("h_y_dtype", "float16")),
            "mean_alpha_dtype": str(precompute_cfg.get("mean_alpha_dtype", "float16")),
        },
    }


def build_cache_input_signature(
    *,
    config: Dict[str, Any],
    fold_id: int,
    partition: str,
    files: Sequence[str],
    checkpoint_sha256: str,
    epoch_min_override: Optional[int] = None,
) -> Tuple[str, str]:
    """Stable signature used to decide whether an existing cache is reusable."""
    summary = build_cache_input_summary(
        config=config,
        fold_id=fold_id,
        partition=partition,
        files=files,
        checkpoint_sha256=checkpoint_sha256,
        epoch_min_override=epoch_min_override,
    )
    summary_json = json.dumps(_json_safe(summary), sort_keys=True, separators=(",", ":"))
    signature = hashlib.sha256(summary_json.encode("utf-8")).hexdigest()
    return signature, summary_json


def compute_checkpoint_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 hex digest of a checkpoint file.

    Args:
        path: Absolute path to the checkpoint file.
        chunk_size: Read chunk size in bytes (1 MiB default).

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_fold_partition_files(
    kfold_base_path: str,
    fold_id: int,
    partition: str,
    test_mode: Optional[str] = None,
) -> List[str]:
    """Resolve the HDF5 file list for one (fold, partition).

    Mirrors the auto-detection rule used by
    ``new_classifier.kfold_classifier_trainer.get_fold_datasets``:
    test partition can live either at ``fold_K/test/`` (augmented) or at
    ``base/test/`` (holdout); train and val are always under ``fold_K/``.

    Args:
        kfold_base_path: Root k-fold directory.
        fold_id: 1-based fold identifier.
        partition: ``"train"``, ``"val"`` or ``"test"``.
        test_mode: ``None`` (auto), ``"holdout"`` or ``"augmented"``.

    Returns:
        Sorted list of HDF5 file paths.

    Raises:
        FileNotFoundError: If the resolved directory does not exist or is
            empty.
    """
    base = Path(kfold_base_path)
    fold_dir = base / f"fold_{fold_id}"

    if partition in ("train", "val"):
        partition_dir = fold_dir / partition
    elif partition == "test":
        if test_mode == "holdout":
            partition_dir = base / "test"
        elif test_mode == "augmented":
            partition_dir = fold_dir / "test"
        else:
            local = fold_dir / "test"
            shared = base / "test"
            if local.exists() and any(local.glob("*.hdf5")):
                partition_dir = local
            elif shared.exists() and any(shared.glob("*.hdf5")):
                partition_dir = shared
            else:
                raise FileNotFoundError(
                    f"No test HDF5 files for fold {fold_id} at {local} or {shared}"
                )
    else:
        raise ValueError(f"partition must be train/val/test, got {partition!r}")

    if not partition_dir.exists():
        raise FileNotFoundError(f"Partition directory missing: {partition_dir}")

    files = sorted(str(p) for p in partition_dir.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(
            f"No HDF5 files found under {partition_dir} for fold {fold_id}/{partition}"
        )
    return files


def build_vae_from_config(
    config: Dict[str, Any],
    device: torch.device,
) -> SeqVaeLagAttnV1:
    """Instantiate ``SeqVaeLagAttnV1`` and load the pretrained checkpoint.

    Args:
        config: Parsed classifier config (top-level dict).
        device: Target device for the loaded model.

    Returns:
        Frozen, eval-mode VAE on ``device``.

    Raises:
        ValueError: If the config is missing the VAE section or checkpoint.
    """
    vae_cfg = config.get("vae")
    if not vae_cfg:
        raise ValueError("config['vae'] is required")
    ckpt_path = vae_cfg.get("checkpoint")
    if not ckpt_path:
        raise ValueError("config['vae']['checkpoint'] is required")

    model_kwargs = dict(vae_cfg.get("model_kwargs", {}))
    # Tuples are not YAML-native; coerce list back to tuple for model API.
    if isinstance(model_kwargs.get("logvar_clamp"), list):
        lv = model_kwargs["logvar_clamp"]
        model_kwargs["logvar_clamp"] = (float(lv[0]), float(lv[1]))

    logger.info(
        "Building SeqVaeLagAttnV1 with kwargs: "
        + ", ".join(f"{k}={v}" for k, v in model_kwargs.items())
    )
    model = SeqVaeLagAttnV1(**model_kwargs)
    load_checkpoint_strict(model=model, checkpoint=ckpt_path)
    logger.info(f"Loaded VAE weights from {ckpt_path}")

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_segment_dataset(
    files: Sequence[str],
    config: Dict[str, Any],
    *,
    cache_size: int = 0,
    epoch_min_override: Optional[int] = None,
) -> CombinedHDF5Dataset:
    """Construct a segment-level :class:`CombinedHDF5Dataset` for precompute.

    Applies only the filters that are safe at the segment level (epoch_min,
    cross-delivery, normalisation, trim). The ``min_samples_per_guid`` and
    ``min_valid_weight_fraction`` filters are deferred to the GUID-level
    dataset.

    Args:
        files: HDF5 file paths.
        config: Parsed classifier config.
        cache_size: Per-segment cache size (0 disables — preferred during
            precompute since each segment is read exactly once).
        epoch_min_override: When provided, used in place of
            ``dataset_config.epoch_min``. Plumbed through from
            ``precompute_fold_latents``' ``epoch_min_overrides`` so that
            train/val/test caches can be built with different windows
            (e.g. train on last 6h, val+test on last 12h).

    Returns:
        Configured dataset ready to feed a DataLoader.
    """
    ds_cfg = config.get("dataset_config", {})
    epoch_min = (
        epoch_min_override
        if epoch_min_override is not None
        else ds_cfg.get("epoch_min")
    )
    epoch_max_rule = ds_cfg.get("epoch_max_rule", "cross_delivery")
    epoch_max = -1260.0 if epoch_max_rule == "cross_delivery" else None
    trim_minutes = float(ds_cfg.get("trim_minutes", 1.0))
    stats_path = ds_cfg.get("stats_path")
    normalize_fields = ds_cfg.get("normalize_fields") or [
        "fhr_st",
        "fhr_ph",
        "up_st",
        "up_ph",
    ]

    return CombinedHDF5Dataset(
        paths=list(files),
        load_fields=list(DEFAULT_LOAD_FIELDS),
        epoch_min=epoch_min,
        epoch_max=epoch_max,
        cache_size=cache_size,
        pin_memory=False,
        stats_path=stats_path,
        normalize_fields=list(normalize_fields),
        trim_minutes=trim_minutes,
    )


def make_segment_dataloader(
    dataset: CombinedHDF5Dataset,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Wrap the segment dataset in a sequential DataLoader for precompute.

    Args:
        dataset: The segment-level dataset.
        batch_size: Forward-pass batch size on the VAE.
        num_workers: DataLoader worker count.

    Returns:
        DataLoader yielding ``AttributeDict`` batches collated by
        :func:`attribute_dict_collate`.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=attribute_dict_collate,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        pin_memory=False,
    )


def _build_u_stream(batch: AttributeDict, use_up_st: bool) -> torch.Tensor:
    """Stack ``up_st`` and ``up_ph`` (or just ``up_ph``) as ``(B, T, C_u)``."""
    up_ph = batch.up_ph
    if use_up_st:
        up_st = batch.up_st
        return torch.cat([up_st, up_ph], dim=-1)
    return up_ph


def _segment_keep_mask(epochs: np.ndarray) -> np.ndarray:
    """Return boolean mask of segments that satisfy the cross-delivery rule.

    ``epoch + 1260 <= 0`` is the strict pre-delivery filter from PRD §4.3.
    The segment dataset already filters via ``epoch_max=-1260``, so this is a
    safety check.
    """
    return (epochs + 1260.0) <= 0.0


def precompute_partition(
    *,
    vae: SeqVaeLagAttnV1,
    files: Sequence[str],
    config: Dict[str, Any],
    cache_path: Path,
    fold_id: int,
    partition: str,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 2,
    train_stats: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None,
    fit_latent_stats_max_batches: Optional[int] = None,
    vae_checkpoint_sha256_override: Optional[str] = None,
    vae_checkpoint_path_override: Optional[str] = None,
    epoch_min_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Precompute one partition's latent cache and write it to ``cache_path``.

    Args:
        vae: Frozen VAE on ``device``.
        files: HDF5 source files.
        config: Parsed classifier config.
        cache_path: Output HDF5 cache path.
        fold_id: Fold identifier for attrs.
        partition: ``"train"`` / ``"val"`` / ``"test"``.
        device: Forward-pass device.
        batch_size: Segment batch size during VAE forward.
        num_workers: DataLoader worker count.
        train_stats: Optional ``(mu_post_mean, mu_post_var, count)``. When
            ``partition == "train"`` and this is ``None``, this function will
            run :func:`vae.fit_latent_stats` and read the result from the
            VAE's running buffers. When ``partition`` is ``val``/``test`` the
            train stats must be supplied so they can be copied into the cache
            attrs.
        fit_latent_stats_max_batches: Cap for the stats fit pass; ``None``
            uses the entire training loader.
        vae_checkpoint_sha256_override: Optional pre-computed SHA-256 to
            record in the cache attrs and signature in place of hashing
            ``config['vae']['checkpoint']``. Used by the live-VAE eval path
            where the source weights live inside the classifier checkpoint
            and the original VAE checkpoint may not be reachable at
            evaluation time.
        vae_checkpoint_path_override: Optional path string to record alongside
            the SHA override (e.g. the classifier ``best.ckpt``).
        epoch_min_override: Optional per-partition ``epoch_min`` override.
            When set, used in place of ``dataset_config.epoch_min`` for
            both segment-level filtering and the cache signature so this
            partition's cache can be invalidated independently of the
            others when only the eval window changes.

    Returns:
        Manifest dict describing the cache (counts, files, stats summary).
    """
    use_up_st = bool(config["vae"]["model_kwargs"].get("use_up_st", True))
    if vae_checkpoint_sha256_override is not None:
        ckpt_sha256 = str(vae_checkpoint_sha256_override)
        ckpt_path = (
            str(vae_checkpoint_path_override)
            if vae_checkpoint_path_override is not None
            else str(config.get("vae", {}).get("checkpoint", ""))
        )
    else:
        ckpt_path = config["vae"]["checkpoint"]
        ckpt_sha256 = compute_checkpoint_sha256(ckpt_path)
    cache_input_signature, cache_input_summary_json = build_cache_input_signature(
        config=config,
        fold_id=fold_id,
        partition=partition,
        files=files,
        checkpoint_sha256=ckpt_sha256,
        epoch_min_override=epoch_min_override,
    )
    d_z = int(config["vae"]["model_kwargs"]["d_z"])
    d_model_vae = int(config["vae"]["model_kwargs"]["d_model"])
    L = int(config["vae"]["model_kwargs"]["max_lag"]) + 1
    T = int(config["vae"]["model_kwargs"]["sequence_length"])
    warmup_period = int(config["vae"]["model_kwargs"]["warmup_period"])

    h_y_dtype_str = str(config.get("precompute", {}).get("h_y_dtype", "float16"))
    mean_alpha_dtype_str = str(
        config.get("precompute", {}).get("mean_alpha_dtype", "float16")
    )
    np_h_y_dtype = np.float16 if h_y_dtype_str == "float16" else np.float32
    np_mean_alpha_dtype = (
        np.float16 if mean_alpha_dtype_str == "float16" else np.float32
    )
    compression = config.get("precompute", {}).get("compression", "gzip")
    compression_level = int(config.get("precompute", {}).get("compression_level", 4))

    dataset = build_segment_dataset(
        files, config, cache_size=0, epoch_min_override=epoch_min_override
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"Segment dataset is empty for fold {fold_id}/{partition} (files={files})"
        )

    loader = make_segment_dataloader(
        dataset, batch_size=batch_size, num_workers=num_workers
    )

    # Fail-fast: confirm the HDF5 actually contains the fields the VAE
    # forward needs. ``CombinedHDF5Dataset`` silently skips missing fields,
    # so without this check we'd lose 5-10 minutes of encoding time before
    # ``_build_u_stream`` raised ``AttributeError`` mid-loop.
    #
    # Inspect the first sample directly rather than holding a live DataLoader
    # iterator: the live iterator would spawn worker processes we'd then need
    # to tear down before the main loop (fragile with ``persistent_workers``).
    try:
        _peek_sample = dataset[0]
    except Exception as exc:
        raise RuntimeError(
            f"Segment dataset for fold {fold_id}/{partition} failed to "
            f"produce a sample: {exc}"
        ) from exc
    required = ["fhr_st", "fhr_ph", "up_ph", "weight", "target", "epoch", "guid"]
    if use_up_st:
        required.append("up_st")
    missing = [f for f in required if not hasattr(_peek_sample, f)]
    if missing:
        available = [
            k for k in dict(_peek_sample).keys() if not k.startswith("source_")
        ]
        raise RuntimeError(
            f"Required fields {missing} are missing from the HDF5 batch for "
            f"fold {fold_id}/{partition}. Available fields: {available}. "
            f"Re-create the cache with these fields, or set "
            f"vae.use_up_st=false if up_st is intentionally absent."
        )
    del _peek_sample

    # Compute train-partition latent stats *inline* during the encoding loop
    # so gap-region timesteps (weight ≈ 0) can be excluded. The prior VAE
    # ``fit_latent_stats`` helper only masks warmup steps — not gaps — and
    # therefore biased ``(mu_post_mean, mu_post_var)`` toward whatever the
    # posterior produced on invalid regions. By accumulating stats only over
    # timesteps where ``weight > 0.5`` (and past the warmup period) we get a
    # physiologically meaningful baseline that matches ``hat_w``'s
    # classifier-time validity definition.
    computing_train_stats = partition == "train" and train_stats is None
    warmup_steps = int(warmup_period) if warmup_period else 0
    sum_x = torch.zeros(d_z, dtype=torch.float64, device=device)
    sum_xx = torch.zeros(d_z, dtype=torch.float64, device=device)
    stats_count = torch.zeros((), dtype=torch.float64, device=device)
    stats_batches_seen = 0
    if computing_train_stats:
        logger.info(
            f"[fold {fold_id}/{partition}] fitting weight-masked latent stats "
            f"inline during encoding (warmup={warmup_steps})"
        )

    # Per-GUID buckets (lists of per-segment numpy arrays / scalars).
    per_guid: Dict[str, Dict[str, List[Any]]] = defaultdict(
        lambda: {
            "h_y": [],
            "mu_prior": [],
            "mu_post": [],
            "kld_per_t": [],
            "mean_alpha": [],
            "weight": [],
            "epoch": [],
            "time_from_labor_onset": [],
            "second_stage_onset": [],
            "cs_label": [],
            "bg_label": [],
            "target": [],
        }
    )

    total_segments = 0
    skipped_cross_delivery = 0
    with torch.no_grad():
        for batch in loader:
            y_st = batch.fhr_st.to(device, non_blocking=True)
            y_ph = batch.fhr_ph.to(device, non_blocking=True)
            u_stream = _build_u_stream(batch, use_up_st).to(
                device, non_blocking=True
            )

            enc = vae.encode_only(y_st, y_ph, u_stream, sample_z=False)
            kld_btd = vae.kld_tensor(
                mu_prior=enc["mu_prior"],
                logvar_prior=enc["logvar_prior"],
                mu_post=enc["mu_post"],
                logvar_post=enc["logvar_post"],
                mask_warmup=False,
            )                                          # (B, T, d_z)
            kld_per_t = kld_btd.sum(dim=-1)            # (B, T)
            mean_alpha = enc["attn_weights"].mean(dim=-2)  # (B, T, L)

            # Accumulate weight-masked mu_post stats on the train partition.
            # Stops after ``fit_latent_stats_max_batches`` to honour the
            # legacy cap, though accumulating over the full loader is
            # generally preferred.
            if computing_train_stats and (
                fit_latent_stats_max_batches is None
                or stats_batches_seen < int(fit_latent_stats_max_batches)
            ):
                weight_dev = batch.weight.to(device, non_blocking=True)  # (B, T)
                valid = weight_dev > 0.5                                  # (B, T)
                if warmup_steps > 0:
                    t_idx = torch.arange(
                        valid.size(-1), device=device
                    )
                    valid = valid & (t_idx >= warmup_steps)[None, :]
                flat_mu = enc["mu_post"][valid].double()                   # (N_valid, d_z)
                if flat_mu.numel() > 0:
                    sum_x += flat_mu.sum(dim=0)
                    sum_xx += (flat_mu * flat_mu).sum(dim=0)
                    stats_count += flat_mu.size(0)
                stats_batches_seen += 1

            h_y = enc["target_state"].cpu().numpy().astype(np_h_y_dtype)
            mu_prior = enc["mu_prior"].cpu().numpy().astype(np.float32)
            mu_post = enc["mu_post"].cpu().numpy().astype(np.float32)
            kld_arr = kld_per_t.cpu().numpy().astype(np.float32)
            mean_alpha_arr = mean_alpha.cpu().numpy().astype(np_mean_alpha_dtype)
            weight = batch.weight.cpu().numpy().astype(np.float32)
            # Store target as float32 rather than int8. The schema defines
            # ``target = class_id * weight`` where ``weight in [0, 1]`` — any
            # segment with partial weights would be silently truncated by an
            # int8 cast (e.g. weight=0.5 * class=3 -> 1.5 -> int8=1, which
            # collapses to a wrong class id). The dataset rounds this back to
            # the canonical integer class id at load time.
            target_per_t = batch.target.cpu().numpy().astype(np.float32)
            epochs = (
                batch.epoch.cpu().numpy().astype(np.float64)
                if isinstance(batch.epoch, torch.Tensor)
                else np.asarray(batch.epoch, dtype=np.float64)
            )
            tlo = batch.time_from_labor_onset
            tlo_arr = (
                tlo.cpu().numpy().astype(np.float32)
                if isinstance(tlo, torch.Tensor)
                else np.asarray(tlo, dtype=np.float32)
            )
            sso = getattr(batch, "second_stage_onset", None)
            if sso is None:
                sso_arr = np.full(len(epochs), np.nan, dtype=np.float32)
            elif isinstance(sso, torch.Tensor):
                sso_arr = sso.cpu().numpy().astype(np.float32)
            else:
                sso_arr = np.asarray(sso, dtype=np.float32)

            cs_arr = (
                batch.cs_label.cpu().numpy().astype(np.uint8)
                if isinstance(batch.cs_label, torch.Tensor)
                else np.asarray(batch.cs_label, dtype=np.uint8)
            )
            bg_arr = (
                batch.bg_label.cpu().numpy().astype(np.uint8)
                if isinstance(batch.bg_label, torch.Tensor)
                else np.asarray(batch.bg_label, dtype=np.uint8)
            )
            guids = batch.guid
            if isinstance(guids, torch.Tensor):
                guids = [g.item() if hasattr(g, "item") else str(g) for g in guids]

            keep_mask = _segment_keep_mask(epochs)
            for i, keep in enumerate(keep_mask):
                if not keep:
                    skipped_cross_delivery += 1
                    continue
                guid = guids[i]
                bucket = per_guid[guid]
                bucket["h_y"].append(h_y[i])
                bucket["mu_prior"].append(mu_prior[i])
                bucket["mu_post"].append(mu_post[i])
                bucket["kld_per_t"].append(kld_arr[i])
                bucket["mean_alpha"].append(mean_alpha_arr[i])
                bucket["weight"].append(weight[i])
                bucket["epoch"].append(float(epochs[i]))
                bucket["time_from_labor_onset"].append(float(tlo_arr[i]))
                bucket["second_stage_onset"].append(float(sso_arr[i]))
                bucket["cs_label"].append(int(cs_arr[i]))
                bucket["bg_label"].append(int(bg_arr[i]))
                bucket["target"].append(target_per_t[i])
                total_segments += 1

    logger.info(
        f"[fold {fold_id}/{partition}] encoded {total_segments} segments "
        f"across {len(per_guid)} GUIDs; "
        f"skipped {skipped_cross_delivery} cross-delivery segments"
    )

    # Finalise inline stats (train partition only).
    if computing_train_stats:
        n_total = int(stats_count.item())
        if n_total == 0:
            raise RuntimeError(
                f"[fold {fold_id}/{partition}] inline latent-stats pass "
                "aggregated zero valid timesteps. Check that the training "
                "loader is not empty and that segments have any weight>0.5 "
                "steps past the warmup period."
            )
        mean_t = (sum_x / stats_count).float().cpu()
        var_t = ((sum_xx / stats_count) - (sum_x / stats_count) ** 2).clamp_min(
            0.0
        ).float().cpu()
        # Mirror the stats into the VAE's running buffers so any code that
        # queries ``vae.normalize_latent`` / ``vae.mu_post_running_*`` later
        # sees the same numbers written to the cache.
        vae.mu_post_running_mean.copy_(mean_t.to(vae.mu_post_running_mean.device))
        vae.mu_post_running_var.copy_(var_t.to(vae.mu_post_running_var.device))
        vae.mu_post_running_count.copy_(
            torch.tensor(n_total, dtype=vae.mu_post_running_count.dtype,
                         device=vae.mu_post_running_count.device)
        )
        train_stats = (mean_t, var_t, n_total)
        logger.info(
            f"[fold {fold_id}/{partition}] inline latent stats: "
            f"n={n_total} samples, mean range "
            f"[{float(mean_t.min()):+.3f}, {float(mean_t.max()):+.3f}], "
            f"var range [{float(var_t.min()):.3f}, {float(var_t.max()):.3f}]"
        )

    # Sanity: shape integrity
    if train_stats is not None:
        mean_buf, var_buf, count_buf = train_stats
    else:
        raise RuntimeError(
            "train_stats is required for val/test partitions; "
            "precompute the train partition first."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with h5py.File(tmp_path, "w", libver="latest") as fh:
        fh.attrs["schema_version"] = SCHEMA_VERSION
        fh.attrs["vae_checkpoint_sha256"] = ckpt_sha256
        fh.attrs["vae_checkpoint_path"] = ckpt_path
        fh.attrs["use_up_st"] = use_up_st
        fh.attrs["d_z"] = d_z
        fh.attrs["d_model"] = d_model_vae
        fh.attrs["L"] = L
        fh.attrs["T"] = T
        fh.attrs["warmup_period"] = warmup_period
        fh.attrs["partition"] = partition
        fh.attrs["fold_id"] = fold_id
        fh.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        fh.attrs["cache_input_signature"] = cache_input_signature
        fh.attrs["cache_input_summary_json"] = cache_input_summary_json
        fh.attrs["mu_post_mean"] = mean_buf.numpy().astype(np.float32)
        fh.attrs["mu_post_var"] = var_buf.numpy().astype(np.float32)
        fh.attrs["latent_stats_count"] = int(count_buf)
        fh.attrs["total_segments"] = total_segments
        fh.attrs["skipped_cross_delivery"] = skipped_cross_delivery

        guids_grp = fh.create_group("guids")
        for guid, bucket in per_guid.items():
            order = np.argsort(np.asarray(bucket["epoch"], dtype=np.float64))
            grp = guids_grp.create_group(guid)
            grp.attrs["S"] = int(len(order))

            def _stack(name: str, dtype: Any) -> np.ndarray:
                """Stack per-segment arrays in chronological order."""
                arrs = [np.asarray(bucket[name][i], dtype=dtype) for i in order]
                return np.stack(arrs, axis=0)

            grp.create_dataset(
                "h_y",
                data=_stack("h_y", np_h_y_dtype),
                compression=compression,
                compression_opts=compression_level,
            )
            grp.create_dataset(
                "mu_prior", data=_stack("mu_prior", np.float32)
            )
            grp.create_dataset(
                "mu_post", data=_stack("mu_post", np.float32)
            )
            grp.create_dataset(
                "kld_per_t", data=_stack("kld_per_t", np.float32)
            )
            grp.create_dataset(
                "mean_alpha",
                data=_stack("mean_alpha", np_mean_alpha_dtype),
                compression=compression,
                compression_opts=compression_level,
            )
            grp.create_dataset("weight", data=_stack("weight", np.float32))
            # Written as float32 to preserve ``class_id * weight`` semantics.
            # :class:`GuidSequenceDataset` rounds to the canonical class id.
            grp.create_dataset("target", data=_stack("target", np.float32))
            grp.create_dataset(
                "epoch",
                data=np.asarray(
                    [bucket["epoch"][i] for i in order], dtype=np.float64
                ),
            )
            grp.create_dataset(
                "time_from_labor_onset",
                data=np.asarray(
                    [bucket["time_from_labor_onset"][i] for i in order],
                    dtype=np.float32,
                ),
            )
            grp.create_dataset(
                "second_stage_onset",
                data=np.asarray(
                    [bucket["second_stage_onset"][i] for i in order],
                    dtype=np.float32,
                ),
            )
            grp.create_dataset(
                "cs_label",
                data=np.asarray(
                    [bucket["cs_label"][i] for i in order], dtype=np.uint8
                ),
            )
            grp.create_dataset(
                "bg_label",
                data=np.asarray(
                    [bucket["bg_label"][i] for i in order], dtype=np.uint8
                ),
            )

    tmp_path.replace(cache_path)
    logger.info(f"[fold {fold_id}/{partition}] wrote cache -> {cache_path}")

    return {
        "cache_path": str(cache_path),
        "fold_id": fold_id,
        "partition": partition,
        "num_guids": len(per_guid),
        "num_segments": total_segments,
        "skipped_cross_delivery": skipped_cross_delivery,
        "vae_checkpoint_sha256": ckpt_sha256,
        "mu_post_mean_summary": [
            float(mean_buf.min()),
            float(mean_buf.max()),
        ],
        "mu_post_var_summary": [
            float(var_buf.min()),
            float(var_buf.max()),
        ],
        "latent_stats_count": int(count_buf),
    }


def precompute_fold_latents(
    config: Dict[str, Any],
    fold_id: int,
    output_root: Path,
    *,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    num_workers: int = 2,
    force: bool = False,
    epoch_min_overrides: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Precompute caches for all three partitions of a single fold.

    Skips a partition cache when one already exists with a matching cache
    input signature (idempotent).

    Args:
        config: Parsed classifier config dict.
        fold_id: 1-based fold identifier.
        output_root: Directory in which ``precomputed_latents/fold_{fold_id}/``
            will be created.
        device: Compute device. Defaults to ``cuda:0`` when available.
        batch_size: VAE forward batch size in segments.
        num_workers: DataLoader worker count.
        force: When True, ignore existing caches and recompute.
        epoch_min_overrides: Optional mapping ``{partition: epoch_min}``
            applied per partition in place of ``dataset_config.epoch_min``.
            Plumbed into both segment-level filtering and the cache input
            signature so each partition's cache invalidates independently
            when only its window changes. Use to widen val/test relative
            to train (e.g. ``{"val": -43200, "test": -43200}``).

    Returns:
        Manifest dict with one entry per partition.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ds_cfg = config["dataset_config"]
    kfold_base_path = ds_cfg["kfold_base_path"]
    test_mode = ds_cfg.get("test_mode")

    out_subdir = config.get("precompute", {}).get(
        "out_subdir", "precomputed_latents"
    )
    fold_dir = output_root / out_subdir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    expected_sha = compute_checkpoint_sha256(config["vae"]["checkpoint"])

    manifest: Dict[str, Any] = {
        "fold_id": fold_id,
        "vae_checkpoint_sha256": expected_sha,
        "vae_checkpoint_path": config["vae"]["checkpoint"],
        "partitions": {},
    }

    overrides = dict(epoch_min_overrides or {})

    def _override_for(partition: str) -> Optional[int]:
        val = overrides.get(partition)
        return int(val) if val is not None else None

    partitions: List[str] = ["train", "val", "test"]
    partition_files: Dict[str, List[str]] = {
        p: get_fold_partition_files(
            kfold_base_path, fold_id, p, test_mode=test_mode
        )
        for p in partitions
    }
    expected_signatures: Dict[str, Tuple[str, str]] = {
        p: build_cache_input_signature(
            config=config,
            fold_id=fold_id,
            partition=p,
            files=partition_files[p],
            checkpoint_sha256=expected_sha,
            epoch_min_override=_override_for(p),
        )
        for p in partitions
    }

    # Decide whether each partition needs (re)computing.
    needs_compute: Dict[str, bool] = {}
    for p in partitions:
        cache_path = fold_dir / f"{p}.hdf5"
        if force or not cache_path.exists():
            needs_compute[p] = True
            continue
        expected_signature, _summary_json = expected_signatures[p]
        try:
            with h5py.File(cache_path, "r", libver="latest") as fh:
                cached_sha = fh.attrs.get("vae_checkpoint_sha256", "")
                cached_signature = fh.attrs.get("cache_input_signature", "")
        except (OSError, KeyError):
            cached_sha = ""
            cached_signature = ""
        if cached_sha != expected_sha or cached_signature != expected_signature:
            reasons = []
            if cached_sha != expected_sha:
                reasons.append("checkpoint SHA mismatch")
            if cached_signature != expected_signature:
                reasons.append("cache input signature mismatch")
            logger.warning(
                f"[fold {fold_id}/{p}] {'; '.join(reasons)} — recomputing"
            )
            needs_compute[p] = True
        else:
            needs_compute[p] = False
            manifest["partitions"][p] = {
                "cache_path": str(cache_path),
                "skipped": True,
                "reason": "matching cache signature",
            }

    if not any(needs_compute.values()):
        logger.info(f"[fold {fold_id}] all caches up-to-date; nothing to do")
        manifest_path = fold_dir / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        return manifest

    vae = build_vae_from_config(config, device)

    train_stats: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None

    # Train MUST be precomputed first because val/test reuse its mu_post stats.
    if needs_compute["train"]:
        manifest["partitions"]["train"] = precompute_partition(
            vae=vae,
            files=partition_files["train"],
            config=config,
            cache_path=fold_dir / "train.hdf5",
            fold_id=fold_id,
            partition="train",
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            train_stats=None,
            fit_latent_stats_max_batches=config["vae"].get(
                "fit_latent_stats_max_batches"
            ),
            epoch_min_override=_override_for("train"),
        )
        train_stats = (
            vae.mu_post_running_mean.detach().cpu().clone(),
            vae.mu_post_running_var.detach().cpu().clone(),
            int(vae.mu_post_running_count.item()),
        )
    else:
        # Pull train stats from the existing train cache so val/test reuse them.
        with h5py.File(fold_dir / "train.hdf5", "r", libver="latest") as fh:
            train_stats = (
                torch.from_numpy(np.asarray(fh.attrs["mu_post_mean"])),
                torch.from_numpy(np.asarray(fh.attrs["mu_post_var"])),
                int(fh.attrs["latent_stats_count"]),
            )
        logger.info(
            f"[fold {fold_id}] reused train mu_post stats from existing cache"
        )

    for partition in ("val", "test"):
        if not needs_compute[partition]:
            continue
        manifest["partitions"][partition] = precompute_partition(
            vae=vae,
            files=partition_files[partition],
            config=config,
            cache_path=fold_dir / f"{partition}.hdf5",
            fold_id=fold_id,
            partition=partition,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            train_stats=train_stats,
            epoch_min_override=_override_for(partition),
        )

    manifest_path = fold_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
    logger.info(f"[fold {fold_id}] manifest -> {manifest_path}")

    return manifest


def _resolve_run_dir(config: Dict[str, Any], explicit: Optional[str]) -> Path:
    """Pick the output run directory.

    Args:
        config: Parsed classifier config.
        explicit: Optional CLI-provided override.

    Returns:
        Absolute output directory under which precomputed caches will live.
    """
    if explicit:
        return Path(explicit).resolve()
    base = Path(config["general_config"]["folders_config"]["out_dir_base"]).resolve()
    tag = str(config["general_config"].get("tag", "guid_cls_v1_run"))
    return base / tag


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector for testing.

    Returns:
        Exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Precompute SeqVaeLagAttnV1 latents per fold for guid_cls_v1"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--fold", type=int, help="Single fold id (1-based)")
    grp.add_argument("--all-folds", action="store_true", help="Run all configured folds")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override run output dir (defaults to out_dir_base/tag)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Compute device",
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute even if cache exists"
    )
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    output_root = _resolve_run_dir(config, args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if args.all_folds:
        num_folds = int(config["dataset_config"]["num_folds"])
        fold_ids = config["dataset_config"].get("fold_ids") or list(
            range(1, num_folds + 1)
        )
    else:
        fold_ids = [int(args.fold)]

    for fold_id in fold_ids:
        precompute_fold_latents(
            config=config,
            fold_id=fold_id,
            output_root=output_root,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
