"""Benchmark: on-the-fly VAE encoding vs pre-computed latents (Task 7.6).

Trains fold_1 for a configurable number of epochs in two modes and reports
wall-clock time per epoch and peak GPU memory.

Usage::

    python -m model.vae_teb_prediction.guid_classifier.benchmark_precompute \\
        --config model/vae_teb_prediction/guid_classifier/config_temporal.yaml \\
        --epochs 10 \\
        --device cuda:0

The script:
  1. Pre-computes latents for fold_1 (if not already cached).
  2. Trains fold_1 for N epochs with on-the-fly VAE encoding.
  3. Trains fold_1 for N epochs with pre-computed latents.
  4. Prints a comparison table.

Expected result: pre-computed latents reduce epoch time by >50%.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from loguru import logger


def _run_training_epochs(
    config: Dict[str, Any],
    fold_id: int,
    n_epochs: int,
    use_precomputed: bool,
    precomputed_dir: str,
    device: str,
) -> Dict[str, Any]:
    """Train for n_epochs and measure timing / memory.

    Args:
        config: Full config dict.
        fold_id: Fold number.
        n_epochs: Number of epochs to train.
        use_precomputed: Whether to use pre-computed latents.
        precomputed_dir: Directory containing pre-computed HDF5 files.
        device: CUDA device string.

    Returns:
        Dict with timing and memory statistics.
    """
    from model.vae_teb_prediction.kfold_classifier_trainer import get_fold_datasets
    from model.vae_teb_prediction.guid_classifier.length_bucket_sampler import (
        create_bucketed_sequence_dataloader,
    )
    from model.vae_teb_prediction.guid_classifier.precompute_latents import (
        create_precomputed_sequence_dataloader,
    )
    from model.vae_teb_prediction.guid_classifier.temporal_classification_model import (
        TemporalVaeClassifier,
    )
    from model.vae_teb_prediction.vae_teb_model_prediction import SeqVae
    from train.graph_models_utils import load_checkpoint_strict

    dataset_cfg = config.get("dataset_config", {})
    kfold_base_path = dataset_cfg["kfold_base_path"]
    fold_datasets = get_fold_datasets(kfold_base_path, fold_id)

    dl_cfg = dataset_cfg.get("dataloader_config", {})
    ds_kwargs = dict(dl_cfg.get("dataset_kwargs", {}))
    stat_path = dataset_cfg.get("stat_path")
    normalize_fields = dl_cfg.get("normalize_fields")
    bucket_cfg = dataset_cfg.get("bucket_sampler", {})

    batch_size = config["general_config"]["batch_size"]["train"]

    common_dl_kwargs = dict(
        num_workers=0,  # Single-process for fair comparison.
        segment_duration=dl_cfg.get("segment_duration", 1200.0),
        guid_cache_size=dl_cfg.get("guid_cache_size", 128),
        stats_path=stat_path,
        normalize_fields=normalize_fields,
        seed=42,
        **ds_kwargs,
    )

    # --- Create dataloader ------------------------------------------------ #
    mode_label = "precomputed" if use_precomputed else "on-the-fly"
    logger.info("Creating {} dataloader...", mode_label)

    if use_precomputed:
        precomputed_path = os.path.join(
            precomputed_dir, f"precomputed_fold_{fold_id}_train.hdf5",
        )
        vae_ckpt = config["model_config"]["vae_checkpoint"]
        train_loader, _ = create_precomputed_sequence_dataloader(
            precomputed_path=precomputed_path,
            hdf5_files=fold_datasets["train"],
            batch_size=batch_size,
            bucket_ranges=bucket_cfg.get("bucket_ranges"),
            shuffle=bucket_cfg.get("shuffle", True),
            vae_checkpoint=vae_ckpt,
            **common_dl_kwargs,
        )
    else:
        train_loader, _ = create_bucketed_sequence_dataloader(
            hdf5_files=fold_datasets["train"],
            batch_size=batch_size,
            bucket_ranges=bucket_cfg.get("bucket_ranges"),
            shuffle=bucket_cfg.get("shuffle", True),
            **common_dl_kwargs,
        )

    # --- Create model ----------------------------------------------------- #
    model_cfg = config.get("model_config", {})
    vae_checkpoint = model_cfg["vae_checkpoint"]

    vae_model = SeqVae()
    load_checkpoint_strict(vae_model, checkpoint=vae_checkpoint)

    seg_cfg = model_cfg.get("segment_encoder", {})
    lstm_cfg = model_cfg.get("temporal_lstm", {})
    feat_cfg = model_cfg.get("temporal_features", {})
    head_cfg = model_cfg.get("classifier_head", {})

    model = TemporalVaeClassifier(
        vae_model=vae_model,
        segment_encoder_type=seg_cfg.get("type", "mean_pool"),
        d_seg=seg_cfg.get("d_seg", 64),
        temporal_lstm_hidden=lstm_cfg.get("hidden_dim", 128),
        temporal_lstm_layers=lstm_cfg.get("num_layers", 2),
        temporal_lstm_dropout=lstm_cfg.get("dropout", 0.1),
        gap_encoding=model_cfg.get("gap_encoding", "concat"),
        position_embed_dim=(
            feat_cfg.get("segment_index", {}).get("embed_dim", 0)
            if feat_cfg.get("segment_index", {}).get("enabled", False)
            else 0
        ),
        max_position_index=feat_cfg.get("segment_index", {}).get("max_index", 40),
        tlo_enabled=feat_cfg.get("time_from_labor_onset", {}).get("enabled", False),
        num_classes=head_cfg.get("num_classes", 2),
        classifier_dropout=head_cfg.get("dropout", 0.1),
        mlp_multiplier=head_cfg.get("mlp_multiplier", 2.0),
        vae_chunk_size=model_cfg.get("vae_chunk_size", 32),
        use_posterior=model_cfg.get("use_posterior", True),
        freeze_vae=model_cfg.get("freeze_vae", True),
        cnn_kernel=seg_cfg.get("cnn_kernel", 7),
    )

    gpu_device = torch.device(device)
    model = model.to(gpu_device)
    model.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["general_config"]["lr"],
    )

    # --- Training loop ---------------------------------------------------- #
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(gpu_device)

    epoch_times = []
    for epoch in range(n_epochs):
        t0 = time.perf_counter()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            # Move tensors to device.
            batch_gpu = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_gpu[k] = v.to(gpu_device)
                else:
                    batch_gpu[k] = v

            optimizer.zero_grad()
            outputs = model(batch_gpu)
            loss_dict = model.compute_loss(outputs, batch_gpu)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        dt = time.perf_counter() - t0
        epoch_times.append(dt)
        avg_loss = total_loss / max(n_batches, 1)
        logger.info(
            "[{}] Epoch {}/{}: loss={:.4f}, time={:.2f}s",
            mode_label, epoch + 1, n_epochs, avg_loss, dt,
        )

    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated(gpu_device) / (1024 ** 2)

    return {
        "mode": mode_label,
        "n_epochs": n_epochs,
        "epoch_times": epoch_times,
        "mean_epoch_time": sum(epoch_times) / len(epoch_times),
        "total_time": sum(epoch_times),
        "peak_gpu_memory_mb": peak_mem_mb,
    }


def main() -> None:
    """CLI entry point for the benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark on-the-fly vs pre-computed latent training.",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config_temporal.yaml.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs per mode.",
    )
    parser.add_argument(
        "--fold_id", type=int, default=1,
        help="Fold ID to benchmark.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="CUDA device.",
    )
    parser.add_argument(
        "--precomputed_dir", type=str, default=None,
        help="Directory with pre-computed HDF5 files. Auto-computes if absent.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # --- Pre-compute latents if needed ------------------------------------ #
    precomputed_dir = args.precomputed_dir
    if precomputed_dir is None:
        out_base = (
            config.get("general_config", {})
            .get("folders_config", {})
            .get("out_dir_base", ".")
        )
        precomputed_dir = os.path.join(
            out_base, "precomputed_latents", f"fold_{args.fold_id}",
        )

    precomputed_path = os.path.join(
        precomputed_dir, f"precomputed_fold_{args.fold_id}_train.hdf5",
    )
    if not os.path.exists(precomputed_path):
        logger.info("Pre-computed latents not found — generating...")
        from model.vae_teb_prediction.guid_classifier.precompute_latents import (
            precompute_fold_latents,
        )

        precompute_fold_latents(
            fold_id=args.fold_id,
            kfold_base_path=config["dataset_config"]["kfold_base_path"],
            vae_checkpoint=config["model_config"]["vae_checkpoint"],
            output_dir=precomputed_dir,
            config=config,
            device=args.device,
        )
    else:
        logger.info("Using existing pre-computed latents: {}", precomputed_path)

    # --- Run benchmark ---------------------------------------------------- #
    logger.info("=" * 60)
    logger.info("BENCHMARK: On-the-fly VAE encoding ({} epochs)", args.epochs)
    logger.info("=" * 60)
    result_online = _run_training_epochs(
        config, args.fold_id, args.epochs,
        use_precomputed=False,
        precomputed_dir=precomputed_dir,
        device=args.device,
    )

    # Clear GPU cache between runs.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info("BENCHMARK: Pre-computed latents ({} epochs)", args.epochs)
    logger.info("=" * 60)
    result_precomputed = _run_training_epochs(
        config, args.fold_id, args.epochs,
        use_precomputed=True,
        precomputed_dir=precomputed_dir,
        device=args.device,
    )

    # --- Print comparison table ------------------------------------------- #
    speedup = result_online["mean_epoch_time"] / max(
        result_precomputed["mean_epoch_time"], 1e-6,
    )

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Metric':<30} {'On-the-fly':>15} {'Pre-computed':>15}")
    print("-" * 60)
    print(
        f"{'Mean epoch time (s)':<30} "
        f"{result_online['mean_epoch_time']:>15.2f} "
        f"{result_precomputed['mean_epoch_time']:>15.2f}"
    )
    print(
        f"{'Total time (s)':<30} "
        f"{result_online['total_time']:>15.2f} "
        f"{result_precomputed['total_time']:>15.2f}"
    )
    print(
        f"{'Peak GPU memory (MB)':<30} "
        f"{result_online['peak_gpu_memory_mb']:>15.1f} "
        f"{result_precomputed['peak_gpu_memory_mb']:>15.1f}"
    )
    print(
        f"{'Speedup':<30} "
        f"{'1.00x':>15} "
        f"{speedup:>14.2f}x"
    )
    print("=" * 60)

    if speedup > 1.5:
        print(f"PASS: Pre-computed latents are {speedup:.1f}x faster (>1.5x).")
    else:
        print(
            f"NOTE: Speedup is {speedup:.1f}x. Expected >1.5x. "
            "The VAE forward pass may not be the bottleneck at this batch size."
        )


if __name__ == "__main__":
    main()
