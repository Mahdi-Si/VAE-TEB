import lightning as L
import torch
import torch.nn as nn
from typing import Optional, Sequence, List
from lightning.pytorch.callbacks import Callback

import os
import json
import numpy as np
from loguru import logger
try:
    from utils.plot_utils import plot_model_analysis
    _HAVE_PLOT_UTILS = True
except Exception:
    _HAVE_PLOT_UTILS = False
try:
    from sklearn.metrics import (
        roc_auc_score,
        roc_curve,
        average_precision_score,
        precision_recall_curve,
        confusion_matrix,
        accuracy_score,
        precision_recall_fscore_support,
    )
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAVE_PLOTLY = True
except Exception:
    _HAVE_PLOTLY = False

from vae_teb_model import SeqVaeTebClassifier


class LightSeqVaeTebClassifier(L.LightningModule):
    """
    LightningModule for training SeqVaeTebClassifier (classification-only).

    - Freezes or fine-tunes the underlying VAE based on flags.
    - Optionally adds auxiliary VAE reconstruction loss (from classifier.compute_loss).
    - Logs classification loss and accuracy across train/val/test.
    """

    def __init__(
        self,
        model: Optional[SeqVaeTebClassifier] = None,
        lr: float = 1e-3,
        lr_milestones: Optional[Sequence[int]] = None,
        vae_loss_weight: float = 0.0,
        freeze_vae: Optional[bool] = None,
        class_weights: Optional[Sequence[float]] = None,
        # Compilation options
        compile_classifier: bool = False,
        compile_vae: bool = False,
        compile_mode: str = "max-autotune-no-cudagraphs",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])  # prevent checkpoint bloat

        # Build or accept an existing classifier
        self.model = model if model is not None else SeqVaeTebClassifier()

        # Optionally override freeze setting
        if freeze_vae is not None:
            if freeze_vae:
                self.model.freeze_vae = True
                self.model.freeze_vae_parameters()
                logger.info("VAE parameters frozen for classification training")
            else:
                self.model.freeze_vae = False
                self.model.unfreeze_vae_parameters()
                logger.info("VAE parameters unfrozen for end-to-end training")

        # Optional class weights for imbalanced classification
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
            self.model.classification_criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            logger.info(f"Applied class weights: {class_weights}")

        # Whether to include auxiliary VAE loss during training
        self.use_aux_vae_loss = vae_loss_weight is not None and vae_loss_weight > 0.0
        self._val_logits: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_logits: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []
        self._compiled = False

    # -------------------- compilation --------------------
    def _compile_if_requested(self):
        if self._compiled:
            return
        mode = getattr(self.hparams, "compile_mode", "max-autotune-no-cudagraphs")
        try:
            if getattr(self.hparams, "compile_classifier", False):
                self.model = torch.compile(self.model, mode=mode, fullgraph=False, dynamic=True)
                logger.info(f"Compiled classifier with torch.compile mode={mode}")
            elif getattr(self.hparams, "compile_vae", False):
                self.model.vae_model = torch.compile(self.model.vae_model, mode=mode, fullgraph=False, dynamic=True)
                logger.info(f"Compiled VAE submodule with torch.compile mode={mode}")
            self._compiled = True
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}; proceeding without compilation")

    def setup(self, stage: Optional[str] = None):
        # Compile after checkpoint weights are loaded (safe for load_from_checkpoint and resume)
        self._compile_if_requested()

    def on_fit_start(self) -> None:
        """Ensure optimizer LR and freeze flags match current hparams when resuming from checkpoint.

        When resuming with Trainer.fit(..., ckpt_path=...), Lightning restores optimizer state
        from the checkpoint, including learning rates. This method enforces the LR from the
        current config (self.hparams.lr) to override any checkpoint LR to keep config authoritative.
        """
        try:
            if hasattr(self, 'trainer') and self.trainer is not None and self.trainer.optimizers:
                for opt in self.trainer.optimizers:
                    for pg in opt.param_groups:
                        pg['lr'] = float(getattr(self.hparams, 'lr', pg.get('lr', 1e-3)))
                logger.info(f"Overrode optimizer LR from checkpoint with config LR={self.hparams.lr}")
        except Exception as e:
            logger.warning(f"Failed to override optimizer LR on fit start: {e}")

    # -------------------- utilities --------------------
    @staticmethod
    def _extract_labels(batch) -> torch.Tensor:
        """Extract binary labels from masked per-timestep targets.
        
        Following cls_implementation.md for binary classification:
        - Dataset stores 1=HEALTHY, 2=ACIDOSIS, 3=HIE at valid timesteps and 0 when masked
        - Binary mapping: HEALTHY (1) -> 0, ACIDOSIS (2) + HIE (3) -> 1
        - Recommended: label = int(target.max()) since invalid timesteps are 0
        """
        if not hasattr(batch, "target"):
            raise RuntimeError("Batch missing 'target' for classification labels")
        
        # Extract max value per sample (following cls_implementation.md recommendation)
        raw_labels = batch.target.max(dim=1).values.long()  # Shape: (batch_size,) values in {0, 1, 2, 3}
        
        # Binary classification mapping:
        # 0 (masked) -> 0 (default to healthy)
        # 1 (HEALTHY) -> 0 
        # 2 (ACIDOSIS) -> 1
        # 3 (HIE) -> 1
        binary_labels = torch.where(raw_labels >= 2, 1, 0)  # ACIDOSIS + HIE = 1, others = 0
        
        return binary_labels

    def _forward_with_optional_vae_loss(self, batch):
        y_st, y_ph, x_ph = batch.fhr_st, batch.fhr_ph, batch.fhr_up_ph
        labels = self._extract_labels(batch)

        if self.use_aux_vae_loss and hasattr(batch, "fhr") and batch.fhr is not None:
            out = self.model.compute_loss(
                y_st=y_st,
                y_ph=y_ph,
                x_ph=x_ph,
                labels=labels,
                y_raw=batch.fhr,
                compute_vae_loss=True,
                vae_loss_weight=self.hparams.vae_loss_weight,
            )
            cls_loss = out["classification_loss"]
            vae_loss = out["vae_loss"]
            total = out["total_loss"]  # classification + weighted VAE
        else:
            out = self.model(y_st, y_ph, x_ph, labels=labels, return_latent=False)
            cls_loss = out["classification_loss"]
            vae_loss = torch.tensor(0.0, device=cls_loss.device if cls_loss is not None else "cpu")
            total = cls_loss
        return out, labels, cls_loss, vae_loss, total
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Minimal cleanup after each training batch."""
        del batch

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        """Minimal cleanup after each validation batch."""
        del batch

    # -------------------- steps --------------------
    def training_step(self, batch, batch_idx):
        out, labels, cls_loss, vae_loss, total = self._forward_with_optional_vae_loss(batch)
        preds = out["predictions"]
        acc = (preds == labels).float().mean()

        self.log("train/loss", total, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/cls_loss", cls_loss, on_step=True, on_epoch=True, prog_bar=False)
        if self.use_aux_vae_loss:
            self.log("train/vae_loss", vae_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return total

    def validation_step(self, batch, batch_idx):
        out, labels, cls_loss, vae_loss, total = self._forward_with_optional_vae_loss(batch)
        preds = out["predictions"]
        acc = (preds == labels).float().mean()

        self.log("val/loss", total, on_epoch=True, prog_bar=True)
        self.log("val/cls_loss", cls_loss, on_epoch=True, prog_bar=False)
        if self.use_aux_vae_loss:
            self.log("val/vae_loss", vae_loss, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)
        # accumulate for metrics callback
        self._val_logits.append(out["logits"].detach().float().cpu())
        self._val_labels.append(labels.detach().cpu())
        return total

    def test_step(self, batch, batch_idx):
        out, labels, cls_loss, vae_loss, total = self._forward_with_optional_vae_loss(batch)
        preds = out["predictions"]
        acc = (preds == labels).float().mean()

        self.log("test/loss", total, on_epoch=True, prog_bar=True)
        self.log("test/cls_loss", cls_loss, on_epoch=True, prog_bar=False)
        if self.use_aux_vae_loss:
            self.log("test/vae_loss", vae_loss, on_epoch=True, prog_bar=False)
        self.log("test/acc", acc, on_epoch=True, prog_bar=True)
        # accumulate for metrics callback
        self._test_logits.append(out["logits"].detach().float().cpu())
        self._test_labels.append(labels.detach().cpu())
        return total

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_st, y_ph, x_ph = batch.fhr_st, batch.fhr_ph, batch.fhr_up_ph
        out = self.model(y_st, y_ph, x_ph, labels=None, return_latent=False)
        return {"predictions": out["predictions"], "probabilities": out["probabilities"]}

    # -------------------- VAE control --------------------
    def freeze_vae(self):
        self.model.freeze_vae_parameters()
        self.model.freeze_vae = True

    def unfreeze_vae(self):
        self.model.unfreeze_vae_parameters()
        self.model.freeze_vae = False

    # -------------------- optimizers --------------------
    def configure_optimizers(self):
        """Configure optimizers following cls_implementation.md recommendations."""
        # Use AdamW with weight decay as recommended in the documentation
        opt = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=1e-4,
            eps=1e-8,
            betas=(0.9, 0.95)  # Slightly higher β2 for better convergence
        )
        
        if self.hparams.lr_milestones:
            from torch.optim.lr_scheduler import MultiStepLR
            sch = MultiStepLR(opt, milestones=list(self.hparams.lr_milestones), gamma=0.1)
            logger.info(f"Using MultiStepLR scheduler with milestones: {self.hparams.lr_milestones}")
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
        return opt


class MemoryMonitorCallback(Callback):
    """
    Callback to monitor GPU memory usage and automatically clear cache when needed.
    Optimized for multi-GPU training with reduced monitoring frequency.
    """

    def __init__(self, threshold_gb=12.0, log_frequency=200):
        """
        Args:
            threshold_gb (float): GPU memory threshold in GB above which cache is cleared.
            log_frequency (int): Frequency (in batches) to log memory usage.
        """
        super().__init__()
        self.threshold_gb = threshold_gb
        self.log_frequency = log_frequency
        self.batch_count = 0

    def _log_memory_usage(self, prefix=""):
        """Log current GPU memory usage for all devices."""
        if torch.cuda.is_available():
            total_allocated = 0.0
            device_count = torch.cuda.device_count()
            for device_id in range(device_count):
                allocated = torch.cuda.memory_allocated(device_id) / 1024 ** 3  # GB
                reserved = torch.cuda.memory_reserved(device_id) / 1024 ** 3  # GB
                logger.info(f"{prefix} GPU {device_id}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                total_allocated += allocated
            return total_allocated
        return 0.0

    def _clear_memory_if_needed(self):
        """Clear GPU memory on all devices if usage exceeds threshold."""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            cleared_any = False
            for device_id in range(device_count):
                allocated = torch.cuda.memory_allocated(device_id) / 1024 ** 3  # GB
                if allocated > self.threshold_gb:
                    logger.warning(
                        f"GPU {device_id} memory usage ({allocated:.2f}GB) exceeds threshold ({self.threshold_gb}GB). Clearing cache...")
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()
                    cleared_any = True
            return cleared_any
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Monitor memory after each training batch."""
        self.batch_count += 1

        # Log memory usage periodically
        if self.batch_count % self.log_frequency == 0:
            self._log_memory_usage(f"Train batch {batch_idx}")

        # Clear memory if needed
        self._clear_memory_if_needed()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Monitor memory after each validation batch."""
        # Clear memory if needed during validation
        self._clear_memory_if_needed()

    def on_train_epoch_start(self, trainer, pl_module):
        """Log memory at the start of each epoch."""
        self._log_memory_usage(f"Epoch {trainer.current_epoch} start")

    def on_train_epoch_end(self, trainer, pl_module):
        """Log usage at the end of each epoch - reduced cache clearing for multi-GPU."""
        self._log_memory_usage(f"Epoch {trainer.current_epoch} end")
        # Only clear cache at epoch end, not during training for better multi-GPU performance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LossPlotCallback(Callback):
    """Tracks and plots classification and optional VAE losses over epochs.

    Saves Plotly HTML plots to `output_dir` every `plot_frequency` epochs.
    """

    def __init__(self, output_dir: str, plot_frequency: int = 10, max_history_size: int = 1000):
        super().__init__()
        self.output_dir = output_dir
        self.plot_frequency = plot_frequency
        self.max_history_size = max_history_size
        self.history = {
            "epoch": [],
            "train/loss": [],
            "train/cls_loss": [],
            "train/vae_loss": [],
            "train/acc": [],
            "val/loss": [],
            "val/cls_loss": [],
            "val/vae_loss": [],
            "val/acc": [],
            "hyperparams/lr": [],
        }
        os.makedirs(self.output_dir, exist_ok=True)

    def _to_float(self, x):
        try:
            return float(x.item() if hasattr(x, "item") else x)
        except Exception:
            return float("nan")

    def _trim(self):
        if len(self.history["epoch"]) > self.max_history_size:
            trim = len(self.history["epoch"]) - self.max_history_size
            for k in self.history:
                self.history[k] = self.history[k][trim:]

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        self.history["epoch"].append(epoch)
        for key in [
            "train/loss", "train/cls_loss", "train/vae_loss", "train/acc",
            "val/loss", "val/cls_loss", "val/vae_loss", "val/acc",
        ]:
            self.history[key].append(self._to_float(metrics.get(key)))

        # record LR
        try:
            lr = trainer.optimizers[0].param_groups[0]["lr"]
        except Exception:
            lr = float("nan")
        self.history["hyperparams/lr"].append(self._to_float(lr))

        self._trim()

        if (epoch + 1) % self.plot_frequency != 0 or not trainer.is_global_zero:
            return

        if not _HAVE_PLOTLY:
            logger.warning("Plotly not available; skipping loss plots.")
            return

        # Plot losses and accuracy
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Losses", "Accuracy"))
        # Loss traces
        for key in ["train/loss", "train/cls_loss", "train/vae_loss", "val/loss", "val/cls_loss", "val/vae_loss"]:
            if any(v is not None and not np.isnan(v) for v in self.history[key]):
                fig.add_trace(go.Scatter(x=self.history["epoch"], y=self.history[key], mode="lines+markers", name=key), row=1, col=1)
        # Accuracy traces
        for key in ["train/acc", "val/acc"]:
            if any(v is not None and not np.isnan(v) for v in self.history[key]):
                fig.add_trace(go.Scatter(x=self.history["epoch"], y=self.history[key], mode="lines+markers", name=key), row=1, col=2)

        fig.update_layout(title="Training/Validation Metrics", template="plotly_white")
        path = os.path.join(self.output_dir, "classification_metrics_epoch.html")
        try:
            fig.write_html(path)
            logger.info(f"Saved loss/accuracy plot to {path}")
        finally:
            del fig


class ClassificationMetricsCallback(Callback):
    """Computes ROC, PR curves, confusion matrix, and summary metrics.

    Expects LightSeqVaeTebClassifier to accumulate `_val_logits/_val_labels` and `_test_logits/_test_labels`.
    Saves plots and JSON metrics to `output_dir` at the end of each epoch.
    """

    def __init__(self, output_dir: str, class_names: Optional[Sequence[str]] = None):
        super().__init__()
        self.output_dir = output_dir
        self.class_names = list(class_names) if class_names is not None else None
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _softmax_probs(logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def _compute_and_save(self, trainer: L.Trainer, split: str, epoch: int, logits: torch.Tensor, labels: torch.Tensor):
        if logits.numel() == 0:
            return
        y_true = labels.numpy()
        y_logits = logits.numpy()
        y_prob = self._softmax_probs(y_logits)
        y_pred = y_prob.argmax(axis=1)
        n_classes = y_prob.shape[1]
        class_names = self.class_names or [f"C{i}" for i in range(n_classes)]

        metrics = {}

        # Basic metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred)) if _HAVE_SKLEARN else float((y_true == y_pred).mean())
        if _HAVE_SKLEARN:
            pr_micro = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
            pr_macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
            metrics.update({
                "precision_micro": float(pr_micro[0]),
                "recall_micro": float(pr_micro[1]),
                "f1_micro": float(pr_micro[2]),
                "precision_macro": float(pr_macro[0]),
                "recall_macro": float(pr_macro[1]),
                "f1_macro": float(pr_macro[2]),
            })

        # Confusion matrix
        if _HAVE_SKLEARN:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
            metrics["confusion_matrix"] = cm.tolist()
            if _HAVE_PLOTLY:
                fig_cm = go.Figure(data=go.Heatmap(z=cm, x=class_names, y=class_names, colorscale="Blues"))
                fig_cm.update_layout(title=f"{split.upper()} Confusion Matrix (epoch {epoch})", xaxis_title="Predicted", yaxis_title="True")
                cm_path = os.path.join(self.output_dir, f"{split}_confusion_matrix_epoch_{epoch}.html")
                try:
                    fig_cm.write_html(cm_path)
                    logger.info(f"Saved {split} confusion matrix to {cm_path}")
                finally:
                    del fig_cm

        # ROC / AUC and PR / AUPRC
        if _HAVE_SKLEARN:
            try:
                if n_classes == 2:
                    auc_roc = roc_auc_score(y_true, y_prob[:, 1])
                    auc_pr = average_precision_score(y_true, y_prob[:, 1])
                    metrics["auroc_macro"] = float(auc_roc)
                    metrics["auprc_macro"] = float(auc_pr)
                    if _HAVE_PLOTLY:
                        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                        prec, rec, _ = precision_recall_curve(y_true, y_prob[:, 1])
                        fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC", "PR"))
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC={auc_roc:.3f}"), row=1, col=1)
                        fig.add_trace(go.Scatter(x=rec, y=prec, name=f"AUPRC={auc_pr:.3f}"), row=1, col=2)
                        fig.update_layout(title=f"{split.upper()} Curves (epoch {epoch})", template="plotly_white")
                        path = os.path.join(self.output_dir, f"{split}_roc_pr_epoch_{epoch}.html")
                        try:
                            fig.write_html(path)
                            logger.info(f"Saved {split} ROC/PR plots to {path}")
                        finally:
                            del fig
                else:
                    # one-vs-rest
                    from sklearn.preprocessing import label_binarize
                    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
                    aucs = []
                    auprs = []
                    if _HAVE_PLOTLY:
                        fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC", "PR"))
                    for c in range(n_classes):
                        try:
                            auc_c = roc_auc_score(y_true_bin[:, c], y_prob[:, c])
                            apr_c = average_precision_score(y_true_bin[:, c], y_prob[:, c])
                        except Exception:
                            auc_c, apr_c = float("nan"), float("nan")
                        aucs.append(auc_c)
                        auprs.append(apr_c)
                        if _HAVE_PLOTLY:
                            try:
                                fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
                                prec, rec, _ = precision_recall_curve(y_true_bin[:, c], y_prob[:, c])
                                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{class_names[c]} AUC={auc_c:.3f}"), row=1, col=1)
                                fig.add_trace(go.Scatter(x=rec, y=prec, name=f"{class_names[c]} AUPRC={apr_c:.3f}"), row=1, col=2)
                            except Exception:
                                pass
                    metrics["auroc_macro"] = float(np.nanmean(aucs))
                    metrics["auprc_macro"] = float(np.nanmean(auprs))
                    if _HAVE_PLOTLY:
                        fig.update_layout(title=f"{split.upper()} Curves (epoch {epoch})", template="plotly_white")
                        path = os.path.join(self.output_dir, f"{split}_roc_pr_epoch_{epoch}.html")
                        try:
                            fig.write_html(path)
                            logger.info(f"Saved {split} ROC/PR plots to {path}")
                        finally:
                            del fig
            except Exception as e:
                logger.warning(f"Failed to compute ROC/PR metrics for {split}: {e}")

        # Save JSON
        json_path = os.path.join(self.output_dir, f"{split}_metrics_epoch_{epoch}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved {split} metrics JSON to {json_path}")
        except Exception as e:
            logger.warning(f"Could not save metrics JSON: {e}")

        # Log to Lightning logger if available
        try:
            if trainer is not None and getattr(trainer, "logger", None) is not None:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        trainer.logger.log_metrics({f"{split}/{k}": float(v)}, step=epoch)
        except Exception:
            pass

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: LightSeqVaeTebClassifier):
        if not trainer.is_global_zero:
            return
        if len(pl_module._val_logits) == 0:
            return
        logits = torch.cat(pl_module._val_logits, dim=0)
        labels = torch.cat(pl_module._val_labels, dim=0)
        # clear buffers
        pl_module._val_logits.clear()
        pl_module._val_labels.clear()
        self._compute_and_save(trainer, "val", trainer.current_epoch, logits, labels)

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: LightSeqVaeTebClassifier):
        if not trainer.is_global_zero:
            return
        if len(pl_module._test_logits) == 0:
            return
        logits = torch.cat(pl_module._test_logits, dim=0)
        labels = torch.cat(pl_module._test_labels, dim=0)
        # clear buffers
        pl_module._test_logits.clear()
        pl_module._test_labels.clear()
        self._compute_and_save(trainer, "test", trainer.current_epoch, logits, labels)


class ReconstructionPlotCallback(Callback):
    """Plots raw vs reconstructed FHR/UP, latent z, and losses using utils.plot_utils.

    Active only when the VAE is being trained (aux VAE loss enabled) or fine-tuned (VAE unfrozen).
    """

    def __init__(self, output_dir: str, plot_every_epoch: int = 5, max_samples: int = 1):
        super().__init__()
        self.output_dir = output_dir
        self.plot_every_epoch = plot_every_epoch
        self.max_samples = max_samples
        os.makedirs(self.output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: "LightSeqVaeTebClassifier"):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch
        if (epoch % self.plot_every_epoch) != 0:
            return
        if not _HAVE_PLOT_UTILS:
            logger.warning("plot_utils not available; skipping reconstruction plots")
            return

        # Gate: only when VAE is being trained/fine-tuned
        if not (getattr(pl_module, "use_aux_vae_loss", False) or (hasattr(pl_module, "model") and getattr(pl_module.model, "freeze_vae", True) is False)):
            return

        # Acquire a validation batch
        try:
            if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
                val_loader = trainer.datamodule.val_dataloader()
            else:
                val_loader = trainer.val_dataloaders
                if isinstance(val_loader, list):
                    val_loader = val_loader[0]
            batch = next(iter(val_loader))
        except Exception as e:
            logger.warning(f"Could not sample a validation batch for plotting: {e}")
            return

        # Move tensors to device
        device = pl_module.device
        to_dev = lambda t: t.to(device) if t is not None else None
        y_st = to_dev(getattr(batch, "fhr_st", None))
        y_ph = to_dev(getattr(batch, "fhr_ph", None))
        x_ph = to_dev(getattr(batch, "fhr_up_ph", None))
        y_raw = to_dev(getattr(batch, "fhr", None))
        up_raw = to_dev(getattr(batch, "up", None)) if hasattr(batch, "up") else None

        if any(t is None for t in [y_st, y_ph, x_ph, y_raw]):
            logger.warning("Missing required fields in batch for reconstruction plotting; need fhr_st, fhr_ph, fhr_up_ph, fhr")
            return

        # Forward through the bare VAE
        vae = pl_module.model.vae_model
        vae.eval()
        with torch.no_grad():
            fwd = vae(y_st=y_st, y_ph=y_ph, x_ph=x_ph)

            # Latent z (B, L, D) -> (D, L) for first sample
            z = fwd.get("z")
            if z is None:
                logger.warning("VAE forward did not return 'z'; skipping plot")
                return

            # Reconstructions
            mu_pr = fwd.get("mu_pr")
            logvar_pr = fwd.get("logvar_pr")

            # Means across timesteps if needed
            mu_means = None
            logvar_means = None
            if mu_pr is not None:
                if mu_pr.dim() == 3:
                    # (B, N, C) -> (B, C)
                    try:
                        _, mu_means_t = vae.get_predictions(mu_pr)
                        mu_means = mu_means_t
                    except Exception:
                        mu_means = mu_pr.mean(dim=1)
                elif mu_pr.dim() == 2:
                    mu_means = mu_pr
            if logvar_pr is not None:
                if logvar_pr.dim() == 3:
                    try:
                        _, logvar_means_t = vae.get_predictions(logvar_pr)
                        logvar_means = logvar_means_t
                    except Exception:
                        logvar_means = logvar_pr.mean(dim=1)
                elif logvar_pr.dim() == 2:
                    logvar_means = logvar_pr

            # Optional KLD tensor for visualization
            kld_tensor = None
            kld_mean = None
            try:
                mu_prior = fwd.get("mu_prior"); logvar_prior = fwd.get("logvar_prior")
                mu_post = fwd.get("mu_post"); logvar_post = fwd.get("logvar_post")
                if all(v is not None for v in [mu_prior, logvar_prior, mu_post, logvar_post]):
                    kld_full = vae._kld_loss(mu_prior, logvar_prior, mu_post, logvar_post, reduce_mean=False)
                    # Expect (B, L, D)
                    kld_tensor = kld_full
                    kld_mean = kld_full.mean(dim=2).mean(dim=0)  # (L,)
            except Exception:
                pass

            # Optional VAE loss for display
            loss_dict = None
            try:
                loss = vae.compute_loss(forward_outputs=fwd, y_st=y_st, y_ph=y_ph, y_raw=y_raw, compute_kld_loss=True, beta=1.0)
                loss_dict = {
                    "mse_loss": float(loss.get("mse_loss", torch.tensor(0.0)).item()),
                    "nll_loss": float(loss.get("nll_loss", torch.tensor(0.0)).item()),
                    "kld_loss": float(loss.get("kld_loss", torch.tensor(0.0)).item()),
                    "total_loss": float(loss.get("total_loss", torch.tensor(0.0)).item()),
                }
            except Exception:
                pass

            # Select first sample
            idx = 0
            def _get_1d(t):
                return t[idx].detach().float().cpu().numpy() if t is not None else None
            def _get_2d(t):
                return t[idx].detach().float().cpu().numpy().T if t is not None else None

            y_raw_norm_np = _get_1d(y_raw)
            up_raw_norm_np = _get_1d(up_raw)
            mu_means_np = _get_1d(mu_means)
            logvar_means_np = _get_1d(logvar_means)
            mu_pr_np = _get_2d(mu_pr) if (mu_pr is not None and mu_pr.dim()==3) else _get_1d(mu_pr)
            logvar_pr_np = _get_2d(logvar_pr) if (logvar_pr is not None and logvar_pr.dim()==3) else _get_1d(logvar_pr)
            z_np = _get_2d(z)
            kld_tensor_np = _get_2d(kld_tensor)
            kld_mean_np = kld_mean.detach().float().cpu().numpy() if kld_mean is not None else None

            # Plot
            plot_model_analysis(
                output_dir=self.output_dir,
                y_raw_normalized=y_raw_norm_np,
                up_raw_normalized=up_raw_norm_np,
                fhr_st=_get_2d(y_st),
                fhr_ph=_get_2d(y_ph),
                fhr_up_ph=_get_2d(x_ph),
                mu_pr_means=mu_means_np,
                log_var_means=logvar_means_np,
                mu_pr=mu_pr_np,
                logvar_pr=logvar_pr_np,
                latent_z=z_np,
                kld_tensor=kld_tensor_np,
                kld_mean_over_channels=kld_mean_np,
                loss_dict=loss_dict,
                epoch=epoch,
                batch_idx=idx,
                training_mode=True,
            )


class HyperparameterLoggingCallback(Callback):
    """
    Callback to track and log hyperparameters like learning rate for classification training.
    """
    def __init__(self):
        super().__init__()
        self.history = {
            "epoch": [],
            "lr": [],
            "vae_loss_weight": []
        }

    def on_train_epoch_start(self, trainer, pl_module):
        """Log hyperparameters at the start of each epoch"""
        if trainer.is_global_zero:  # Only log on main process for multi-GPU
            epoch = trainer.current_epoch
            
            # Get current learning rate
            lr = 0.0
            try:
                if hasattr(trainer, 'optimizers') and trainer.optimizers:
                    optimizer = trainer.optimizers[0] if isinstance(trainer.optimizers, list) else trainer.optimizers
                    lr = optimizer.param_groups[0]['lr']
            except (IndexError, AttributeError):
                pass
            
            # Get VAE loss weight if available
            vae_loss_weight = getattr(pl_module.hparams, 'vae_loss_weight', 0.0)
            
            # Store in history
            self.history["epoch"].append(epoch)
            self.history["lr"].append(lr)
            self.history["vae_loss_weight"].append(vae_loss_weight)
            
            # Log to trainer for tensorboard/wandb
            pl_module.log('hyperparams/lr', lr, on_epoch=True, logger=True)
            pl_module.log('hyperparams/vae_loss_weight', vae_loss_weight, on_epoch=True, logger=True)
            
            logger.info(f"Epoch {epoch}: lr={lr:.6f}, vae_loss_weight={vae_loss_weight:.4f}")
