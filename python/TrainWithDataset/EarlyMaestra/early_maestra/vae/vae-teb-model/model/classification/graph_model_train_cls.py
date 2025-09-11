import os
import sys
import json
import csv
import yaml
from datetime import datetime
from typing import List, Dict, Any, Optional
import multiprocessing as mp

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.strategies import DDPStrategy

from loguru import logger

from hdf5_dataset.hdf5_dataset import create_optimized_dataloader
from vae_teb_model import SeqVaeTebClassifier
try:
    from model.classification.pytorch_lightning_modules_cls import (
        LightSeqVaeTebClassifier,
        LossPlotCallback,
        ClassificationMetricsCallback,
        MemoryMonitorCallback,
        HyperparameterLoggingCallback,
    )
except ImportError:
    from pytorch_lightning_modules_cls import (
        LightSeqVaeTebClassifier,
        LossPlotCallback,
        ClassificationMetricsCallback,
        MemoryMonitorCallback,
        HyperparameterLoggingCallback,
    )


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d--[%H-%M]")


def discover_hdf5_files(root_dir: str, split: str) -> List[str]:
    """Collect all .hdf5 files under a split directory (train/val/test)."""
    base = os.path.join(root_dir, split)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Split directory not found: {base}")
    files = [os.path.join(base, f) for f in os.listdir(base) if f.lower().endswith(".hdf5")]
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under: {base}")
    return sorted(files)


def build_dataloaders_for_fold(
    fold_root: str,
    batch_sizes: Dict[str, int],
    num_workers: int,
    stats_path: str,
    normalize_fields: List[str],
    dataset_kwargs: Dict[str, Any],
    rank: int,
    world_size: int,
):
    train_files = discover_hdf5_files(fold_root, "train")
    val_files = discover_hdf5_files(fold_root, "val")
    test_files = discover_hdf5_files(fold_root, "test")

    common_kwargs = dict(
        rank=rank,
        world_size=world_size,
        stats_path=stats_pat
        normalize_fields=normalize_fields,
        **dataset_kwargs,
    )

    train_loader = create_optimized_dataloader(
        hdf5_files=train_files,
        batch_size=batch_sizes.get("train", 32),
        num_workers=num_workers,
        shuffle=True,
        **common_kwargs,
    )
    val_loader = create_optimized_dataloader(
        hdf5_files=val_files,
        batch_size=batch_sizes.get("val", batch_sizes.get("train", 32)),
        num_workers=max(0, num_workers // 2),
        shuffle=False,
        **common_kwargs,
    )
    test_loader = create_optimized_dataloader(
        hdf5_files=test_files,
        batch_size=batch_sizes.get("test", batch_sizes.get("val", 32)),
        num_workers=max(0, num_workers // 2),
        shuffle=False,
        **common_kwargs,
    )
    return train_loader, val_loader, test_loader


def maybe_compile_vae(model: SeqVaeTebClassifier, compile_flag: bool):
    if not compile_flag:
        return model
    try:
        model.vae_model = torch.compile(
            model.vae_model,
            mode="max-autotune-no-cudagraphs",
            fullgraph=False,
            dynamic=True,
        )
        logger.info("Compiled VAE submodule with torch.compile (max-autotune-no-cudagraphs)")
    except Exception as e:
        logger.warning(f"torch.compile failed for VAE: {e}; proceeding without compile")
    return model


def train_one_fold(
    fold_root: str,
    out_dir: str,
    cfg: Dict[str, Any],
    rank: int = 0,
    world_size: int = 1,
    devices_override: Optional[List[int]] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots"); os.makedirs(plots_dir, exist_ok=True)
    metrics_dir = os.path.join(out_dir, "metrics"); os.makedirs(metrics_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)

    # Data config
    data_cfg = cfg.get("data_config", {})
    stats_path = data_cfg.get("stats_path")
    normalize_fields = data_cfg.get("normalize_fields", ["fhr", "up", "fhr_st", "fhr_ph", "fhr_up_ph"])
    dataset_kwargs = data_cfg.get("dataset_kwargs", {})
    batch_sizes = data_cfg.get("batch_size", {"train": 32, "val": 64, "test": 64})
    # Force single-process data loading to avoid multiprocessing issues
    num_workers = 0

    # Build dataloaders
    train_loader, val_loader, test_loader = build_dataloaders_for_fold(
        fold_root=fold_root,
        batch_sizes=batch_sizes,
        num_workers=num_workers,
        stats_path=stats_path,
        normalize_fields=normalize_fields,
        dataset_kwargs=dataset_kwargs,
        rank=rank,
        world_size=world_size,
    )

    # Model config
    model_cfg = cfg.get("classifier_config", {})
    num_classes = int(model_cfg.get("num_classes", 2))
    filters = int(model_cfg.get("filters", 32))
    depth = int(model_cfg.get("depth", 6))
    dropout = float(model_cfg.get("dropout", 0.2))
    use_attention = bool(model_cfg.get("use_attention", True))
    freeze_vae_flag = bool(model_cfg.get("freeze_vae", True))
    pretrained_vae_ckpt = model_cfg.get("pretrained_vae_ckpt")
    class_weights = model_cfg.get("class_weights")
    vae_loss_weight = float(model_cfg.get("vae_loss_weight", 0.0))
    compile_vae = bool(model_cfg.get("compile_vae", False))

    # Instantiate classifier
    classifier = SeqVaeTebClassifier(
        num_classes=num_classes,
        classifier_filters=filters,
        classifier_depth=depth,
        classifier_dropout=dropout,
        use_attention=use_attention,
        freeze_vae=freeze_vae_flag,
        pretrained_vae_path=pretrained_vae_ckpt,
        class_weights=torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None,
    )

    # Lightning module
    train_cfg = cfg.get("training_config", {})
    lr = float(train_cfg.get("lr", 1e-3))
    lr_milestones = train_cfg.get("lr_milestones", [])
    compile_classifier = bool(train_cfg.get("compile_classifier", False))
    compile_mode = str(train_cfg.get("compile_mode", "max-autotune-no-cudagraphs"))

    lm = LightSeqVaeTebClassifier(
        model=classifier,
        lr=lr,
        lr_milestones=lr_milestones,
        vae_loss_weight=vae_loss_weight,
        freeze_vae=freeze_vae_flag,
        class_weights=class_weights,
        compile_classifier=compile_classifier,
        compile_vae=compile_vae,
        compile_mode=compile_mode,
    )

    # Callbacks
    callbacks = []
    callbacks.append(LossPlotCallback(output_dir=plots_dir, plot_frequency=int(cfg.get("plot_frequency", 1))))
    callbacks.append(ClassificationMetricsCallback(output_dir=metrics_dir, class_names=train_cfg.get("class_names")))
    callbacks.append(HyperparameterLoggingCallback())
    # VAE recon plots only make sense if we are fine-tuning VAE or using aux VAE loss
    from model.classification.pytorch_lightning_modules_cls import ReconstructionPlotCallback
    callbacks.append(ReconstructionPlotCallback(output_dir=os.path.join(plots_dir, "vae"), plot_every_epoch=int(cfg.get("plot_frequency", 1))))
    callbacks.append(MemoryMonitorCallback(threshold_gb=12.0, log_frequency=200))

    # Early stopping and checkpoints
    # Early stopping strictly on classification validation loss
    es_cfg = cfg.get("early_stopping", {"enabled": True, "patience": 20, "min_delta": 0.0})
    if es_cfg.get("enabled", True):
        callbacks.append(
            EarlyStopping(
                monitor="val/cls_loss",
                mode="min",
                patience=int(es_cfg.get("patience", 20)),
                min_delta=float(es_cfg.get("min_delta", 0.0)),
            )
        )
    # Checkpointing can also default to classification loss for model selection
    ckpt_cfg = cfg.get("checkpoint", {"mode": "min", "save_top_k": 2})
    callbacks.append(
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-{epoch}",
            monitor=ckpt_cfg.get("monitor", "val/cls_loss"),
            mode=ckpt_cfg.get("mode", "min"),
            save_top_k=int(ckpt_cfg.get("save_top_k", 2)),
        )
    )

    # Trainer
    # Devices: allow override (e.g., from parallel scheduler using CUDA_VISIBLE_DEVICES)
    devices = devices_override if devices_override is not None else (train_cfg.get("devices", [0]) if torch.cuda.is_available() else "auto")
    strategy = DDPStrategy(find_unused_parameters=False) if (isinstance(devices, list) and len(devices) > 1) else "auto"
    precision = train_cfg.get("precision", "16-mixed")
    max_epochs = int(train_cfg.get("epochs", 50))
    acc_batches = int(train_cfg.get("accumulate_grad_batches", 1))

    resume_ckpt = train_cfg.get("resume_ckpt")

    trainer = L.Trainer(
        max_epochs=max_epochs,
        precision=precision,
        callbacks=callbacks,
        default_root_dir=out_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        strategy=strategy,
        accumulate_grad_batches=acc_batches,
        log_every_n_steps=50,
    )

    # Fit and test (resume if provided)
    if resume_ckpt:
        logger.info(f"Resuming training from checkpoint: {resume_ckpt}")
        trainer.fit(lm, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_ckpt)
    else:
        trainer.fit(lm, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lm, dataloaders=test_loader, ckpt_path="best")

    # After testing, export predictions CSV for the test set using best checkpoint
    try:
        # Find best checkpoint path
        best_ckpt = None
        for cb in callbacks:
            if isinstance(cb, ModelCheckpoint):
                best_ckpt = cb.best_model_path
                break
        if not best_ckpt:
            if hasattr(trainer, "checkpoint_callback") and trainer.checkpoint_callback:
                best_ckpt = trainer.checkpoint_callback.best_model_path

        export_ckpt = best_ckpt if (best_ckpt and os.path.exists(best_ckpt)) else None
        out_csv = os.path.join(out_dir, "test_predictions.csv")
        export_test_predictions(
            export_ckpt=export_ckpt,
            base_model_cfg=model_cfg,
            lm_current=lm,
            test_loader=test_loader,
            out_csv_path=out_csv,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        logger.info(f"Saved test predictions to {out_csv}")
    except Exception as e:
        logger.warning(f"Failed to export test predictions CSV: {e}")


def export_test_predictions(
    export_ckpt: Optional[str],
    base_model_cfg: Dict[str, Any],
    lm_current: "LightSeqVaeTebClassifier",
    test_loader,
    out_csv_path: str,
    device: str = "cpu",
):
    """Run inference on test_loader and write CSV with requested columns."""
    # Build a fresh classifier matching training config
    num_classes = int(base_model_cfg.get("num_classes", 2))
    filters = int(base_model_cfg.get("filters", 32))
    depth = int(base_model_cfg.get("depth", 6))
    dropout = float(base_model_cfg.get("dropout", 0.2))
    use_attention = bool(base_model_cfg.get("use_attention", True))
    freeze_vae_flag = bool(base_model_cfg.get("freeze_vae", True))
    pretrained_vae_ckpt = base_model_cfg.get("pretrained_vae_ckpt")
    class_weights = base_model_cfg.get("class_weights")

    classifier = SeqVaeTebClassifier(
        num_classes=num_classes,
        classifier_filters=filters,
        classifier_depth=depth,
        classifier_dropout=dropout,
        use_attention=use_attention,
        freeze_vae=freeze_vae_flag,
        pretrained_vae_path=pretrained_vae_ckpt,
        class_weights=torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None,
    )

    if export_ckpt and os.path.exists(export_ckpt):
        lm = LightSeqVaeTebClassifier.load_from_checkpoint(export_ckpt, model=classifier, strict=False)
    else:
        lm = lm_current

    lm.eval().to(device)

    header = [
        "guid",
        "epoch",
        "cs_label",
        "bg_label",
        "target",
        "target_model",
        "probability",
    ]

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        with torch.no_grad():
            for batch in test_loader:
                y_st = batch.fhr_st.to(device)
                y_ph = batch.fhr_ph.to(device)
                x_ph = batch.fhr_up_ph.to(device)

                # Metadata fields may be tensors/lists/bytes
                guid = getattr(batch, "guid", None)
                epoch = getattr(batch, "epoch", None)
                cs_label = getattr(batch, "cs_label", None)
                bg_label = getattr(batch, "bg_label", None)
                target_ts = getattr(batch, "target", None)

                out = lm.model(y_st, y_ph, x_ph, labels=None, return_latent=False)
                probs = out["probabilities"].detach().cpu()
                preds = out["predictions"].detach().cpu()

                # Dataset target scalar = max over time (1..C)
                if target_ts is not None:
                    if torch.is_tensor(target_ts):
                        target_vals = target_ts.max(dim=1).values.detach().cpu().long()
                    else:
                        import numpy as np
                        tt = np.asarray(target_ts)
                        target_vals = torch.from_numpy(tt.max(axis=1)).long()
                else:
                    target_vals = torch.zeros(preds.shape[0], dtype=torch.long)

                B = preds.shape[0]
                for i in range(B):
                    pred = int(preds[i].item())
                    prob = float(probs[i, pred].item())

                    def norm_val(v, idx):
                        try:
                            if v is None:
                                return ""
                            vi = v[idx]
                            if torch.is_tensor(vi):
                                vi = vi.detach().cpu()
                                if vi.dtype == torch.uint8:
                                    return vi.numpy().tobytes().decode("utf-8", errors="ignore")
                                try:
                                    return vi.item()
                                except Exception:
                                    return str(vi)
                            if isinstance(vi, bytes):
                                return vi.decode("utf-8", errors="ignore")
                            return str(vi)
                        except Exception:
                            return ""

                    row = [
                        norm_val(guid, i),
                        norm_val(epoch, i),
                        norm_val(cs_label, i),
                        norm_val(bg_label, i),
                        int(target_vals[i].item()),
                        pred,
                        f"{prob:.6f}",
                    ]
                    writer.writerow(row)


def _run_fold_process(fold_path: str, fold_out: str, cfg: Dict[str, Any], assigned_gpus: List[int]):
    # Isolate GPUs for this process
    if torch.cuda.is_available() and assigned_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in assigned_gpus)
        local_devices = list(range(len(assigned_gpus)))
    else:
        local_devices = "auto"
    train_one_fold(fold_path, fold_out, cfg, rank=0, world_size=1, devices_override=local_devices if isinstance(local_devices, list) else None)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m model.classification.graph_model_train_cls <config_yaml>")
        sys.exit(1)
    cfg_path = sys.argv[1]
    cfg = load_yaml(cfg_path)

    # Root for folds and output
    folds_root = cfg.get("folds_root")
    if not folds_root or not os.path.isdir(folds_root):
        raise FileNotFoundError(f"folds_root directory not found: {folds_root}")

    tag = cfg.get("tag", "seqvae_teb_cls")
    out_base = cfg.get("out_dir_base", os.path.join(os.getcwd(), "results_cls"))
    session_dir = os.path.join(out_base, f"{timestamp_tag()}--{tag}")
    os.makedirs(session_dir, exist_ok=True)

    # Discover fold directories
    fold_dirs = sorted([d for d in os.listdir(folds_root) if d.startswith("fold_")])
    if not fold_dirs:
        raise RuntimeError(f"No folds found under {folds_root}; expected directories like 'fold_1', 'fold_2', ...")

    # Save config snapshot
    with open(os.path.join(session_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Parallel execution config
    par_cfg = cfg.get("parallel", {})
    folds_include = par_cfg.get("folds")  # e.g., [1,3,5] or ["fold_1","fold_3"]
    if folds_include:
        norm = set([f"fold_{int(x)}" if isinstance(x, (int, str)) and str(x).isdigit() else str(x) for x in folds_include])
        fold_dirs = [d for d in fold_dirs if d in norm]
        if not fold_dirs:
            raise RuntimeError("No matching folds from 'parallel.folds'")

    max_parallel = int(par_cfg.get("max_parallel", 1))
    devices_pool = par_cfg.get("devices_pool", list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [])
    gpus_per_run = int(par_cfg.get("gpus_per_run", 1)) if devices_pool else 0
    fold_devices_map = par_cfg.get("fold_devices_map", {})  # e.g., {"fold_1":[0],"fold_2":[1,2]}

    # Helper: assign devices per fold
    def assign_devices_for_fold(fd: str) -> List[int]:
        if str(fd) in fold_devices_map:
            return list(map(int, fold_devices_map[str(fd)]))
        if not devices_pool or gpus_per_run <= 0:
            return []
        # Round-robin chunking without overlap best-effort
        # Build chunks once
        chunks = [devices_pool[i:i+gpus_per_run] for i in range(0, len(devices_pool), gpus_per_run)]
        # Simple deterministic mapping by index
        idx = fold_dirs.index(fd) % len(chunks)
        return chunks[idx]

    if max_parallel <= 1:
        # Sequential
        for fd in fold_dirs:
            fold_path = os.path.join(folds_root, fd)
            logger.info(f"==== Training fold (sequential): {fd} ====")
            fold_out = os.path.join(session_dir, fd)
            assigned = assign_devices_for_fold(fd)
            _run_fold_process(fold_path, fold_out, cfg, assigned)
        return

    # Parallel: limit concurrency by GPU capacity
    if devices_pool and gpus_per_run > 0:
        capacity = max(1, len(devices_pool) // gpus_per_run)
        max_workers = min(max_parallel, capacity)
    else:
        max_workers = max_parallel

    logger.info(f"Launching up to {max_workers} parallel folds using non-daemonic processes ...")
    ctx = mp.get_context("spawn")
    procs = []

    def _start_for_fd(fd: str):
        fold_path = os.path.join(folds_root, fd)
        fold_out = os.path.join(session_dir, fd)
        assigned = assign_devices_for_fold(fd)
        logger.info(f"Starting {fd} on GPUs {assigned if assigned else 'CPU'}")
        p = ctx.Process(target=_run_fold_process, args=(fold_path, fold_out, cfg, assigned), daemon=False)
        p.start()
        return p

    # Launch with concurrency limit
    for fd in fold_dirs:
        while len([p for p in procs if p.is_alive()]) >= max_workers:
            # Reap finished processes
            for p in list(procs):
                if not p.is_alive():
                    p.join()
                    procs.remove(p)
            if len([p for p in procs if p.is_alive()]) >= max_workers:
                import time; time.sleep(0.5)
        procs.append(_start_for_fd(fd))

    # Wait for all to complete
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
