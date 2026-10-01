r"""Training-framework integration of the neural classifier (SPEC §10.5, §10.8, §10.10).

:class:`ClassifierTask` (``LightningModelBase``) and :class:`ClassifierTrainer` (``GraphModelBase``) follow the
``lag_attn_rws`` family pattern; one trainer per unit = (fold, seed, kind). :func:`train_unit` is steps 1-3 of the
§10.10.1 outer loop (unit config, fit, teardown); :func:`score_split` scores a split from ``best.ckpt`` (raw,
prior-corrected logits); :func:`unit_calibration` / :func:`calibrate_frames` are §10.8. Callbacks, in registration
order (§10.10.2): :class:`GuidEpochMetricsCallback` (first: F4), ``MetricsLoggingCallback`` +
``MetricsHistoryCsvCallback``, ``LossPlotCallback``, ``HyperparameterLoggingCallback``, :class:`ClassifierPlotCallback`,
``ModelCheckpoint("best")``, then from the base ``EarlyStopping``, ``LearningRateMonitor`` (switched to ``step``),
``EMAWeightAveraging`` (inserted after it when ``train.ema``) and ``MLflowRunLoggingCallback``.

Framework gotchas (§10.10.7): F1 ``compile_model=False``; F2/F3 the optimizer and scheduler overrides; F4 the callback
order; F5 ``cuda_devices: []`` forces CPU in :meth:`ClassifierTrainer._build_trainer_kwargs`; F6 every output dir is
the unit dir; F7 :func:`_teardown`; F10 ``log_model: false`` (``config.unit_config``); F11 ``ClassifierNet`` has no
``model``/``net``/``network``/``module`` submodule, so ``_clean_state_dict`` only strips the task's ``model.`` /
``_orig_model.`` prefixes; F13 own loaders (``data.make_loader``); F14 ``train/*`` CSV cells are last-step values.

EMA: validation and the train-subset monitor see the EMA weights (the callback swaps them in for the validation
epoch), and ``best.ckpt``'s ``state_dict`` holds them (``current_model_state`` the raw ones), so :func:`score_split`
scores exactly the weights that were selected.
"""
from __future__ import annotations

import atexit
import dataclasses
import gc
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import Callback, EMAWeightAveraging, LearningRateMonitor, ModelCheckpoint
from loguru import logger
from scipy.optimize import minimize
from scipy.special import log_expit, logit, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from hdf5_dataset.hdf5_dataset import AttributeDict, attribute_dict_collate
from teb_vae.classifier import metrics
from teb_vae.classifier.config import (
    Classifier, Config, LabelsCfg, TrainCfg, frozen_baseline, selection_monitor, unit_config, unit_dir,
)
from teb_vae.classifier.data import (
    OnlineReader, SegmentDataset, UnitData, build_unit, collate_segments, make_loader, without_indicators,
)
from teb_vae.classifier.losses import class_weights, compute_loss, coral_log_probs, criterion, prior_offset
from teb_vae.classifier.model import ClassifierNet, aggregate, n_params, running_aggregate
from teb_vae.classifier.sources import OnlineFeatures, VaeSource
from teb_vae.classifier.thresholds import fpr_threshold
from teb_vae.lag_attn.eval.report import json_safe
from train.callbacks import (
    HyperparameterLoggingCallback, LossPlotCallback, MetricsHistoryCsvCallback, MetricsLoggingCallback,
)
from train.graph_model_base import GraphModelBase
from train.graph_models_utils import check_model_class, load_checkpoint_strict
from train.pl_model_base import LightningModelBase
from utils.custom_logger import setup_logging
from utils.mlflow_utils import log_artifact_to_mlflow

NAN = float("nan")
#: FPR cap of the monitoring-only quantile threshold (``val/sens_at_fpr30``, ``confusion_bin@q30``).
MONITOR_ALPHA = 0.3
ONLINE_POSITIONS = (1, 3, 5)

#: §10.10.3, the always-present columns of ``metrics_history.csv``; :func:`tracked_metrics` adds the conditional ones.
TRACKED_METRICS: Tuple[str, ...] = (
    *(f"train/{m}" for m in ("total_loss", "loss_final", "loss_positions", "loss_bag", "loss_segment", "loss_aux3",
                             "acc_bin", "grad_norm", "grad_clip_frac", "weight_norm", "attn_entropy",
                             "mean_num_segments", "guid_auroc", "guid_logloss")),
    *(f"val/{m}" for m in ("total_loss", "loss_final", "loss_positions", "loss_bag", "loss_segment", "loss_aux3",
                           "guid_logloss", "guid_auroc", "guid_auprc", "guid_pauc30", "guid_brier", "guid_ece",
                           "seg_auroc", "sens_at_fpr30", "online_auroc_last", "online_auroc_pos1",
                           "online_auroc_pos3", "score_mean_pos", "score_mean_neg")),
    "lr",
)


def has_three_class(labels: LabelsCfg) -> bool:
    """A 3-class probability output exists: the aux head, or the (multiclass | ordinal) main head of ``three_class``."""
    return labels.aux_3class_weight > 0 or labels.task == "three_class"


def tracked_metrics(c: Classifier, advanced: Mapping[str, Any]) -> Tuple[str, ...]:
    """:data:`TRACKED_METRICS` with the §10.10.3 conditional additions (3-class, spike breaker) of this unit; the
    clip fraction only when clipping is on (it is never logged otherwise)."""
    clip = (advanced.get("trainer") or {}).get("gradient_clip_val")
    out = [m for m in TRACKED_METRICS if m != "train/grad_clip_frac" or (clip is not None and float(clip) > 0)]
    if has_three_class(c.labels):  # §10.10.2 3-class row
        out += ["val/macro_f1", "val/auroc_macro", *(f"val/{m}_class{k}" for m in ("precision", "recall", "f1",
                                                                                   "auroc_ovr") for k in range(3))]
    if (advanced.get("spike_breaker") or {}).get("enabled", False):
        out += ["train/spike_skipped", "train/spike_ema_loss"]
    if c.train.regime == "cotrain":  # §10.10.3 cotrain row, plus the gated selection monitor (§10.10.2 #12)
        out += ["train/vae_total_loss", "train/vae_kld", "train/vae_nll", "train/l2sp",
                *(f"val/{m}" for m in GATE_METRICS), "val/gate_ok", selection_monitor(advanced, "cotrain")[0]]
    return tuple(out)


# ---- scoring (shared by validation, the train-subset monitor and score_split) --------------------------------
def _entropy(a: torch.Tensor) -> torch.Tensor:
    return -(a * a.clamp_min(1e-12).log()).sum(-1)


#: Per-segment pooling-attention scalars of the prediction tables (§11.12 E5; never per-step arrays), NaN where the
#: model has no such weights: see :func:`_attention`.
ATTN_COLUMNS = ["attn_late_mass", "attn_centroid", "seq_attn_final"]


def _attention(a: torch.Tensor, valid: torch.Tensor, seq_attn: Optional[torch.Tensor],
               mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """:data:`ATTN_COLUMNS` of the real segments (M rows): step weights ``a`` and ``valid`` (M, T') placed on the
    segment's valid span, 0 at its first valid step (the warm-up boundary the model saw) to 1 at its last:
    ``attn_centroid`` is the weighted mean position, ``attn_late_mass`` the mass on the late half (position >= 0.5).
    ``seq_attn_final``: the weight the GUID's final position gives each segment (``attention_mil``'s (B, N, N)
    ``seq_attn``; the last True of ``mask`` (B, N) is the final position), else NaN."""
    t = torch.arange(a.shape[-1], device=a.device, dtype=a.dtype).expand_as(a)
    lo = t.masked_fill(~valid, a.shape[-1]).amin(-1, keepdim=True)
    hi = t.masked_fill(~valid, -1).amax(-1, keepdim=True)
    pos = ((t - lo) / (hi - lo).clamp_min(1)).clamp(0, 1)
    out = {"attn_late_mass": (a * (pos >= 0.5)).sum(-1), "attn_centroid": (a * pos).sum(-1)}
    if seq_attn is None:
        out["seq_attn_final"] = torch.full_like(out["attn_centroid"], NAN)
    else:
        last = (mask * torch.arange(mask.shape[1], device=mask.device)).amax(1)
        out["seq_attn_final"] = seq_attn[torch.arange(len(last), device=mask.device), last][mask]
    return out


def _rows(net: ClassifierNet, out: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor],
          offset: torch.Tensor) -> Dict[str, np.ndarray]:
    """One batch as per-segment rows: frame ``row``, prior-corrected ``logit_seg`` (NaN without a segment-local
    head) and ``logit_online`` (sequence scope), step ``attn_entropy``, the :data:`ATTN_COLUMNS` scalars
    (:func:`_attention`), and ``p3``: the 3-class probabilities of the main head on ``three_class`` (softmax | CORAL),
    else of the aux head, if any. A CORAL head adds ``ord_score`` (its g); a multiclass one in sequence scope adds
    ``p3_seg`` (the segment-local head's, for its calibrated logit)."""
    seq = net.scope == "sequence"
    mask = batch["seg_mask"] if seq else torch.ones_like(batch["row"], dtype=torch.bool)
    off = offset.to(out["step_attn"].device)
    seg_head = net.head_seg if seq else net.head
    r = {"row": batch["row"][mask], "attn_entropy": _entropy(out["step_attn"][mask]),
         **_attention(out["step_attn"][mask], batch["step_mask"][mask], out.get("seq_attn"), mask)}
    r["logit_seg"] = (seg_head.score(out["seg"] - off)[mask] if "seg" in out
                      else torch.full_like(r["attn_entropy"], NAN))
    if seq:
        r["logit_online"] = net.head.score(out["pos"] - off)[mask]
    head, o = net.head, out["pos" if seq else "seg"] - off
    aux = out.get("aux3_pos" if seq else "aux3")
    if head.kind == "ordinal":
        r["ord_score"], r["p3"] = (o[..., 0] - head.biases()[0])[mask], coral_log_probs(o).exp()[mask]
    elif head.kind == "multiclass" and o.shape[-1] == 3:
        r["p3"] = o.softmax(-1)[mask]
        if seq and "seg" in out:
            r["p3_seg"] = (out["seg"] - off).softmax(-1)[mask]
    elif aux is not None:
        r["p3"] = aux.softmax(-1)[mask]
    return {k: v.detach().float().cpu().numpy() for k, v in r.items()}


def _aggregates(seg: pd.DataFrame, col: str, aggregators: Sequence[str], tau: float
                ) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    """Segment scope (§11.2): the online score, the running primary aggregate of ``col`` per GUID (aligned with
    ``seg``, whose rows are in ``seg_pos`` order per GUID), and ``{aggregator: GUID score}`` indexed by guid."""
    by = seg.groupby("guid", sort=False)[col]
    online = by.transform(lambda s: running_aggregate(s.to_numpy(copy=True), aggregators[0], tau))
    return online, {agg: by.agg(lambda s: aggregate(s.to_numpy(copy=True), agg, tau)) for agg in aggregators}


def _frames(frame: pd.DataFrame, parts: Sequence[Mapping[str, np.ndarray]], *, scope: str,
            aggregators: Sequence[str], tau: float, causal: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Buffered rows -> ``(segments, guids)``: one row per scored segment (frame columns + scores) and per GUID
    (``score_final``: sequence final position | segment-scope primary aggregator; ``score_<agg>`` for the other
    aggregators in segment scope; ``p_c<k>`` at the final position | mean over the GUID's segments; ``ord_score``
    read like ``score_final``). Segment rows carry ``p_c<k>``, ``ord_score`` and ``pseg_c<k>`` (from ``p3_seg``).
    ``causal=False`` (:attr:`ClassifierNet.causal`): ``logit_online``, ``p_c<k>`` and ``ord_score`` are NaN on every
    segment row, since a non-causal position sees later segments (§9.1); the GUID's ``score_final``, ``p_c<k>`` and
    ``ord_score`` still read the final position (the segment-local head's ``pseg_c<k>`` stay: they are causal)."""
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    probs = {name: cat.pop(key) for key, name in (("p3", "p_c"), ("p3_seg", "pseg_c")) if key in cat}
    seg = frame.iloc[cat.pop("row").astype(int)].assign(
        **cat, **{f"{name}{k}": P[:, k] for name, P in probs.items() for k in range(3)}).sort_index()
    by = seg.groupby("guid", sort=False)
    if scope == "segment":
        seg["logit_online"], per_guid = _aggregates(seg, "logit_seg", aggregators, tau)
    last = seg.groupby("guid", sort=False).tail(1).set_index("guid")
    gd = last[["fold", "split", "y", "class_code"]].assign(n_segments=by.size(), score_final=last["logit_online"])
    if scope == "segment":
        for agg in aggregators[1:]:
            gd[f"score_{agg}"] = per_guid[agg]
    p_cols = [c for c in seg if c.startswith("p_c")]
    if p_cols:
        gd[p_cols] = last[p_cols] if scope == "sequence" else by[p_cols].mean()
    if "ord_score" in seg:  # CORAL g, read like score_final = g + b_1 (the aggregators commute with a shift)
        gd["ord_score"] = (last["ord_score"] if scope == "sequence"
                           else _aggregates(seg, "ord_score", aggregators[:1], tau)[1][aggregators[0]])
    if not causal:  # every position output: the online logit, its class probabilities and CORAL g
        seg[["logit_online", *p_cols, *(["ord_score"] if "ord_score" in seg else [])]] = NAN
    return seg.reset_index(drop=True), gd.reset_index()


@torch.no_grad()
def _score(net: ClassifierNet, loader: Any, offset: torch.Tensor, device: Any,
           features: Optional[OnlineFeatures] = None) -> List[Dict[str, np.ndarray]]:
    """Buffered rows of ``loader`` in eval mode; ``features`` (the online backbone) rewrites ``x``/``attn`` first."""
    modules = [m for m in (net, features) if m is not None]
    was_training = [m.training for m in modules]
    for m in modules:
        m.eval()
    parts = []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        if features is not None:
            batch = features(batch)
        parts.append(_rows(net, net(batch), batch, offset))
    for m, was in zip(modules, was_training):
        m.train(was)
    return parts


def _auroc(y: Any, s: Any) -> float:
    y, s = np.asarray(y) > 0, np.asarray(s, dtype=float)
    y, s = y[~np.isnan(s)], s[~np.isnan(s)]
    return float(roc_auc_score(y, s)) if 0 < y.sum() < y.size else NAN


def guid_metrics(seg: pd.DataFrame, gd: pd.DataFrame, alpha: float = MONITOR_ALPHA) -> Dict[str, Any]:
    """§10.10.2 monitoring metrics from the engine's formulas (``metrics.threshold_free`` et al.), plus the
    ``epoch_summary.jsonl`` confusions and supports. Scores are prior-corrected, uncalibrated logits."""
    y, s = (gd["y"].to_numpy() > 0).astype(int), gd["score_final"].to_numpy(float)
    tf = metrics.threshold_free(y, s, alpha=alpha)
    thr = fpr_threshold(s[y == 0], alpha, "empirical")["threshold"] if (y == 0).any() else NAN
    at_q = metrics.thresholded(y, s, thr)
    on = seg.assign(k=seg.groupby("guid", sort=False).cumcount() + 1)
    m: Dict[str, Any] = {
        "val/guid_logloss": tf["logloss"], "val/guid_auroc": tf["auroc"], "val/guid_auprc": tf["auprc"],
        "val/guid_pauc30": tf[f"pauc@{alpha:g}"], "val/guid_brier": tf["brier"], "val/guid_ece": tf["ece"],
        "val/seg_auroc": _auroc(seg["y"], seg["logit_seg"]), "val/sens_at_fpr30": at_q["sens"],
        "val/score_mean_pos": float(s[y == 1].mean()) if y.any() else NAN,
        "val/score_mean_neg": float(s[y == 0].mean()) if (y == 0).any() else NAN,
        "val/prevalence": tf["prevalence"], "val/online_auroc_last": _auroc(y, s),
        **{f"val/online_auroc_pos{k}": _auroc(on.loc[on["k"] == k, "y"], on.loc[on["k"] == k, "logit_online"])
           for k in ONLINE_POSITIONS},
    }
    p_cols = [c for c in gd if c.startswith("p_c")]
    extra: Dict[str, Any] = {
        "confusion_bin@0.5": {k: v for k, v in metrics.thresholded(y, s, 0.0).items() if k in ("tp", "fp", "tn", "fn")},
        "confusion_bin@q30": {k: at_q[k] for k in ("tp", "fp", "tn", "fn", "threshold")},
        "support": {str(k): int(v) for k, v in gd["y"].value_counts().sort_index().items()},
        "n_val_guids": int(len(gd)),
    }
    if p_cols:
        y3, P = gd["class_code"].to_numpy(int) - 1, gd[p_cols].to_numpy(float)
        m |= {f"val/{k.replace('_c', '_class')}": v for k, v in metrics.multiclass(y3, P).items()}
        extra["confusion_3class"] = confusion_matrix(y3, P.argmax(1), labels=[0, 1, 2]).tolist()
    return m | extra


# ---- the Lightning module ----------------------------------------------------------------------------------------
class ClassifierTask(LightningModelBase):
    """Lightning task of :class:`~teb_vae.classifier.model.ClassifierNet` (§10.10.1).

    Every ctor kwarg is JSON-able (hparams / MLflow params). ``classifier_kwargs`` rebuilds the net without a config;
    ``prior_offset`` (K_out floats, §10.2) is subtracted from raw head outputs before any GUID score is formed.
    ``checkpoint_extras`` (scaler, fingerprint, channels) is set by the trainer and written by
    :meth:`on_save_checkpoint`; it is not a hyperparameter because it is large.
    """

    GRAD_NORM_LOG_EVERY_N_STEPS = 25

    def __init__(self, base_model: nn.Module, *, lr: float, weight_decay: float,
                 spike_breaker: Optional[Dict[str, Any]] = None, classifier_kwargs: Dict[str, Any],
                 train_cfg: Dict[str, Any], class_weights: List[float], prior_offset: List[float],
                 loss_alpha: Optional[float] = None) -> None:
        super().__init__(base_model, lr=lr, weight_decay=weight_decay, spike_breaker=spike_breaker,
                         compile_model=False)  # F1: GUID lengths vary
        self.save_hyperparameters("classifier_kwargs", "train_cfg", "class_weights", "prior_offset", "loss_alpha")
        model_cfg = classifier_kwargs["model_cfg"]
        self.scope, self.lse_tau = model_cfg["scope"], float(model_cfg["lse_tau"])
        self.labels_cfg = LabelsCfg.model_validate(classifier_kwargs["labels_cfg"])
        self.train_cfg = TrainCfg.model_validate(train_cfg)
        self.priors_main = (classifier_kwargs.get("priors") or {}).get("main")
        self.offset = torch.tensor(prior_offset, dtype=torch.float32)
        self.checkpoint_extras: Dict[str, Any] = {}
        self.val_buffer: List[Dict[str, np.ndarray]] = []
        self.grad_norms: List[float] = []
        self._last_out: Optional[Dict[str, torch.Tensor]] = None
        #: The online regimes' :class:`~teb_vae.classifier.sources.OnlineFeatures` (set by the trainer after
        #: construction, so it is no hyperparameter); None under ``frozen_cached``.
        self.backbone: Optional[OnlineFeatures] = None

    def compute_loss_and_metrics(self, batch, batch_idx: int, stage: str):
        if self.backbone is not None:
            batch = self.backbone(batch)
        out = self._last_out = self.model(batch)
        total, parts = compute_loss(out, batch, scope=self.scope, labels_cfg=self.labels_cfg,
                                    train_cfg=self.train_cfg, class_weights=self.hparams.class_weights,
                                    priors=self.priors_main, lse_tau=self.lse_tau, alpha=self.hparams.loss_alpha)
        if "vae_loss" in batch:  # a co-training step (§10.6): λ_cls L_cls + λ_vae L_vae + λ_sp ‖θ - θ₀‖²
            cw, l2sp = self.train_cfg.cotrain, self.backbone.l2sp()
            parts |= {"loss_cls": total.detach(), "l2sp": l2sp.detach(), **batch["vae_metrics"]}
            total = cw.cls_weight * total + cw.vae_weight * batch["vae_loss"] + cw.l2sp * l2sp
        net, y = self.orig_model, batch["y"]
        if self.scope == "sequence":
            mask = batch["seg_mask"]
            score = out["pos_score"][torch.arange(len(y), device=y.device), mask.sum(1) - 1]
            valid, attn, n_seg = batch["step_mask"][mask], out["step_attn"][mask], mask.sum(1).float().mean()
        else:
            score, valid, attn = out["seg_score"], batch["step_mask"], out["step_attn"]
            n_seg = len(y) / len(batch["guid"].unique())  # a float: the base logs it on the module device
        pos = y > 0
        head = net.head
        metrics_ = {
            "total_loss": total, "main_loss": total, **parts,
            "acc_bin": ((score > 0) == pos).float().mean(), "mean_num_segments": n_seg,
            "frac_masked_steps": 1.0 - valid.float().mean(), "attn_entropy": _entropy(attn).mean(),
            "attn_max_weight": attn.amax(-1).mean(),
            "logit_mean_pos": score[pos].mean() if bool(pos.any()) else None,
            "logit_mean_neg": score[~pos].mean() if bool((~pos).any()) else None,
            "head_bias": head.biases()[0] if head.kind == "ordinal" else head.score(head.out.bias),
        }
        if "seq_attn" in out:
            metrics_["seq_attn_entropy"] = _entropy(out["seq_attn"])[batch["seg_mask"]].mean()
        return total, metrics_

    def on_validation_epoch_start(self) -> None:
        self.val_buffer = []

    def validation_step(self, batch, batch_idx):
        loss = super().validation_step(batch, batch_idx)
        self.val_buffer.append(_rows(self.orig_model, self._last_out, batch, self.offset))
        return loss

    # -- optimisation (F2, F3) --
    def configure_param_groups(self):
        """Decay vs no-decay; no-decay = biases (incl. head, CORAL ``bias_*``, ``time_bias``), norms, embeddings.
        A trainable backbone adds its ``train.unfreeze`` allowlist as a third group at ``train.backbone_lr``, trainable
        now or not (LPFT flips it at stage 2 without touching the optimizer or the scheduler's per-group LRs), with no
        weight decay: AdamW's decay would pull pretrained weights toward 0, not toward themselves."""
        net = self.orig_model
        embeddings = {f"{n}.weight" for n, m in net.named_modules() if isinstance(m, nn.Embedding)}
        decay, no_decay = [], []
        for name, p in net.named_parameters():
            if p.requires_grad:
                (no_decay if p.ndim < 2 or "bias" in name or name in embeddings else decay).append(p)
        groups = [{"params": decay, "weight_decay": float(self.hparams.weight_decay)},
                  {"params": no_decay, "weight_decay": 0.0}]
        if self.backbone is not None and self.backbone.source.unfreeze:
            groups.append({"params": [p for _, p in self.backbone.source.allowlist()],
                           "lr": float(self.train_cfg.backbone_lr), "weight_decay": 0.0})
        return groups

    def build_optimizer(self, trainable_params):
        return torch.optim.AdamW(list(trainable_params), lr=float(self.hparams.lr), eps=1e-8,
                                 betas=tuple(self.train_cfg.optimizer.betas))

    def build_lr_scheduler(self, optimizer):
        """Per-step linear warm-up, then cosine to ``min_lr_frac`` over ``estimated_stepping_batches``."""
        warm, floor = self.train_cfg.schedule.warmup_steps, self.train_cfg.schedule.min_lr_frac
        total = float(self.trainer.estimated_stepping_batches)

        def factor(step: int) -> float:
            if step < warm:
                return (step + 1) / warm
            done = min(1.0, (step - warm) / max(1.0, total - warm)) if math.isfinite(total) else 0.0
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * done))

        return {"scheduler": LambdaLR(optimizer, factor), "interval": "step", "frequency": 1}

    # -- monitoring --
    def _on_train_epoch_start_hook(self) -> None:
        self.grad_norms = []

    def on_before_optimizer_step(self, optimizer) -> None:
        """Pre-clip ``train/grad_norm`` and ``train/grad_clip_frac`` every 25 steps and on the last batch
        (``lag_attn_rws/task.py:216``). First, the §10.1 guard on an online backbone (``VaeSource.check_frozen``)."""
        if self.backbone is not None:
            self.backbone.source.check_frozen()
        trainer = self.trainer
        if not (trainer.is_last_batch or trainer.global_step % self.GRAD_NORM_LOG_EVERY_N_STEPS == 0):
            return
        norms = [torch.linalg.vector_norm(p.grad.detach()) for p in self.parameters() if p.grad is not None]
        if not norms:
            return
        grad_norm = torch.linalg.vector_norm(torch.stack(norms))
        self.grad_norms.append(float(grad_norm))
        self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=True, logger=True)
        clip = trainer.gradient_clip_val
        if clip is not None and float(clip) > 0.0:
            self.log("train/grad_clip_frac", (grad_norm > float(clip)).to(grad_norm.dtype), on_step=True,
                     on_epoch=True, logger=True)

    def on_train_epoch_end(self) -> None:
        weights = torch.stack([torch.linalg.vector_norm(p.detach()) for p in self.orig_model.parameters()])
        self.log("train/weight_norm", torch.linalg.vector_norm(weights), on_epoch=True, logger=True)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)  # model_class stamp
        checkpoint["classifier_kwargs"] = dict(self.hparams.classifier_kwargs)
        checkpoint.update(self.checkpoint_extras)


# ---- callbacks ---------------------------------------------------------------------------------------------------
class GuidEpochMetricsCallback(Callback):
    """#1 (F4): GUID-level validation metrics logged in the callback's ``on_validation_epoch_end``, which runs before
    the module's, so CSV row ``e`` holds epoch ``e``. Also the train-subset overfitting monitor (every
    ``train_eval_every`` epochs from epoch 0) and one ``epoch_summary.jsonl`` line per epoch. The collated val frames
    are left on ``pl_module.val_result`` for :class:`ClassifierPlotCallback`."""

    def __init__(self, *, val_frame: pd.DataFrame, output_dir: Any, scope: str, aggregators: Sequence[str],
                 tau: float, regime: str, train_loader: Any = None, train_frame: Optional[pd.DataFrame] = None,
                 train_eval_every: int = 5) -> None:
        super().__init__()
        self.val_frame, self.path = val_frame, Path(output_dir) / "epoch_summary.jsonl"
        self.kw = dict(scope=scope, aggregators=list(aggregators), tau=tau)
        self.regime, self.train_loader, self.train_frame = regime, train_loader, train_frame
        self.every = max(1, int(train_eval_every))

    def on_fit_start(self, trainer, pl_module) -> None:
        if trainer.is_global_zero:
            self.path.write_text("")  # a re-run unit starts a fresh summary

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or not pl_module.val_buffer:
            return
        kw = self.kw | {"causal": pl_module.orig_model.causal}
        seg, gd = _frames(self.val_frame, pl_module.val_buffer, **kw)
        result = guid_metrics(seg, gd)
        epoch = trainer.current_epoch
        if self.train_loader is not None and epoch % self.every == 0:
            parts = _score(pl_module.orig_model, self.train_loader, pl_module.offset, pl_module.device,
                           pl_module.backbone)
            train = guid_metrics(*_frames(self.train_frame, parts, **kw))
            result |= {"train/guid_auroc": train["val/guid_auroc"], "train/guid_logloss": train["val/guid_logloss"]}
        for name, value in result.items():
            if "/" in name:
                pl_module.log(name, float(value), on_step=False, on_epoch=True, logger=True)
        pl_module.val_result = (seg, gd, result["confusion_bin@q30"]["threshold"])
        pl_module.val_metrics = result  # read by PreservationGateCallback, registered right after this one
        if trainer.is_global_zero:
            norms = pl_module.grad_norms
            line = {"epoch": epoch, "global_step": trainer.global_step, **result,
                    "grad_norm_mean_epoch": float(np.mean(norms)) if norms else None,
                    "grad_norm_max_epoch": float(np.max(norms)) if norms else None,
                    "lr": trainer.optimizers[0].param_groups[0]["lr"], "stage": self.regime}
            with self.path.open("a") as fh:
                fh.write(json.dumps(json_safe(line)) + "\n")


class ClassifierPlotCallback(Callback):
    """#5: one diagnostics page per ``every_n_epochs`` (rank 0, not in sanity; a failure only warns): val GUID ROC
    and PR, score histograms by class, reliability (equal-mass bins), confusion at the val-quantile FPR-cap
    threshold, pooling-attention entropy. The only figure text is the panel titles and ``epoch N``.
    ``classifier_diagnostics/epoch{E:04d}_diagnostics.<fmt>``."""

    def __init__(self, output_dir: Any, every_n_epochs: int = 5, file_format: str = "pdf",
                 mlflow_logger: Any = None) -> None:
        super().__init__()
        self.dir = Path(output_dir) / "classifier_diagnostics"
        self.every, self.fmt, self._mlflow_logger = max(1, int(every_n_epochs)), file_format.lstrip("."), mlflow_logger

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch
        if not trainer.is_global_zero or trainer.sanity_checking or (epoch + 1) % self.every:
            return
        import matplotlib.pyplot as plt

        try:
            path = self._plot(*pl_module.val_result, epoch)
            log_artifact_to_mlflow(self._mlflow_logger, path, trainer)
        except Exception as exc:  # noqa: BLE001 - a figure is never worth a fit
            plt.close("all")
            logger.warning(f"ClassifierPlotCallback failed at epoch {epoch}: {exc}")

    def _plot(self, seg: pd.DataFrame, gd: pd.DataFrame, thr: float, epoch: int) -> Path:
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import precision_recall_curve, roc_curve

        y, s = (gd["y"].to_numpy() > 0).astype(int), gd["score_final"].to_numpy(float)
        fig, ax = plt.subplots(2, 3, figsize=(13, 8))
        fpr, tpr, _ = roc_curve(y, s)
        ax[0, 0].plot(fpr, tpr)
        ax[0, 0].plot([0, 1], [0, 1], ":", color="grey")
        ax[0, 0].set(title=f"ROC (AUROC {_auroc(y, s):.3f})", xlabel="FPR", ylabel="TPR")
        prec, rec, _ = precision_recall_curve(y, s)
        ax[0, 1].plot(rec, prec)
        ax[0, 1].axhline(y.mean(), ls=":", color="grey")
        ax[0, 1].set(title="PR", xlabel="recall", ylabel="precision")
        for k, name in ((0, "negative"), (1, "positive")):
            ax[0, 2].hist(s[y == k], bins=20, alpha=0.6, label=name)
        ax[0, 2].axvline(thr, color="k", ls="--", label="threshold")
        ax[0, 2].set(title="GUID score by class", xlabel="logit")
        ax[0, 2].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))  # outside: never over the histograms
        frac, mean_p = calibration_curve(y, 1 / (1 + np.exp(-s)), n_bins=max(2, min(10, len(y) // 5)),
                                         strategy="quantile")
        ax[1, 0].plot(mean_p, frac, "o-")
        ax[1, 0].plot([0, 1], [0, 1], ":", color="grey")
        ax[1, 0].set(title="reliability", xlabel="predicted", ylabel="observed")
        cm = confusion_matrix(y, (s > thr).astype(int), labels=[0, 1])
        ax[1, 1].imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax[1, 1].text(j, i, str(v), ha="center", va="center")
        ax[1, 1].set(title="confusion", xlabel="predicted", ylabel="true", xticks=[0, 1], yticks=[0, 1])
        ax[1, 2].hist(seg["attn_entropy"], bins=30)
        ax[1, 2].set(title="pooling attention entropy", xlabel="nats")
        fig.suptitle(f"epoch {epoch}")
        fig.tight_layout()
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"epoch{epoch:04d}_diagnostics.{self.fmt}"
        fig.savefig(path)
        plt.close(fig)
        return path


class LpftUnfreezeCallback(Callback):
    """#11 (§10.1 ``lpft``, Kumar 2022): stage 1 trains the head alone for ``head_epochs`` epochs, with early stopping
    suspended (patience ∞) so it runs its full length; at the first epoch of stage 2 the backbone allowlist starts
    requiring gradients (its optimizer group at ``train.backbone_lr`` exists from the start), early stopping is reset
    (``wait_count`` 0, ``best_score`` ±∞, patience restored) and one line goes to ``stage_transitions.jsonl``.
    ``train/stage`` (1 | 2) is logged every epoch. ``best.ckpt`` keeps competing across stages, so a stage-1 head is a
    candidate as the frozen epoch is (§10.6). ``ponytail:`` the cosine schedule runs over both stages, so stage 2
    starts below ``backbone_lr``; a per-stage schedule if fine-tuning looks LR-starved."""

    def __init__(self, *, head_epochs: int, output_dir: Any) -> None:
        super().__init__()
        self.head_epochs, self.path, self.patience = int(head_epochs), Path(output_dir) / "stage_transitions.jsonl", {}

    def on_fit_start(self, trainer, pl_module) -> None:
        if trainer.is_global_zero:
            self.path.write_text("")
        for cb in trainer.early_stopping_callbacks:
            self.patience[id(cb)], cb.patience = cb.patience, math.inf

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        source, stage = pl_module.backbone.source, 1 + (trainer.current_epoch >= self.head_epochs)
        if stage == 2 and not any(p.requires_grad for _, p in source.allowlist()):
            for _, p in source.allowlist():
                p.requires_grad_(True)
            pl_module.backbone.train(pl_module.training)  # the allowlisted modules now train
            for cb in trainer.early_stopping_callbacks:
                cb.patience = self.patience.get(id(cb), cb.patience)
                cb.wait_count, cb.best_score = 0, torch.tensor(math.inf if cb.mode == "min" else -math.inf)
            n = sum(p.numel() for _, p in source.allowlist())
            line = {"epoch": trainer.current_epoch, "stage": 2, "n_params_trainable": n_params(pl_module),
                    "n_params_frozen": sum(p.numel() for p in pl_module.parameters()) - n_params(pl_module),
                    "n_backbone_trainable": n, "prefixes": list(source.unfreeze),
                    "backbone_lr": float(pl_module.train_cfg.backbone_lr), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if trainer.is_global_zero:
                with self.path.open("a") as fh:
                    fh.write(json.dumps(line) + "\n")
            logger.info(f"LPFT stage 2 at epoch {trainer.current_epoch}: {n} backbone parameters under "
                        f"{list(source.unfreeze)} now train at lr {pl_module.train_cfg.backbone_lr}")
        pl_module.log("train/stage", float(stage), on_step=False, on_epoch=True, logger=True)


#: ``val/<name>`` of the preservation gate (§10.6); ``forecast_mse_rel`` is the gated one.
GATE_METRICS = ("forecast_mse_rel", "kld_active_frac", "logvar_prior_floor_frac", "delta_mu_sat_frac")
#: Validation segments of the gate's fixed subset (seeded by the unit).
GATE_SEGMENTS = 64


@torch.no_grad()
def preservation(source: VaeSource, rows: Mapping[str, Any], chunk: int) -> Dict[str, float]:
    """The VAE's preservation readouts on ``rows`` (collated raw HDF5 rows) at its current weights, eval mode, dense
    geometry: ``forecast_mse`` (the full branch decoded from μ^q, not a draw, scored by the model's own ``compute_loss``
    under ``likelihood='mse'`` on its own target and forecast mask, so every family is scored by its own code),
    ``kld_active_frac`` and ``logvar_prior_floor_frac`` from the same call, and the forward's ``delta_mu_sat_frac``.
    Row-weighted means over chunks. The pilot's ``preservation_pass`` (the same readouts) does not import at HEAD and is
    trf_cfs-only; ``ponytail:`` its anchor-pooled means and preserved-window support are left out."""
    model, task = source.model, source.task
    modes = {m: m.training for m in model.modules()}
    model.eval()
    model._reparameterize_shared = lambda mu_p, _lv_p, mu_q, _lv_q: (mu_p, mu_q)  # mean decode, both branches
    device, n, sums = next(model.parameters()).device, len(rows["weight"]), dict.fromkeys(GATE_METRICS[1:], 0.0)
    try:
        for i in range(0, n, chunk):
            batch = task.transfer_batch_to_device(AttributeDict({k: v[i:i + chunk] for k, v in rows.items()}), device, 0)
            out = model(*task._build_forward_inputs(batch))
            target, weight = task._build_raw_target(batch)
            m = model.compute_loss(out, target, weight=weight, beta=0.0, likelihood="mse")["metrics"]
            k = len(batch["weight"]) / n
            for name, value in (("forecast_mse", m["nll_full_block"]), ("kld_active_frac", m["kld_active_frac"]),
                                ("logvar_prior_floor_frac", m["logvar_prior_floor_frac"]),
                                ("delta_mu_sat_frac", out["delta_mu_sat_frac"])):
                sums[name] = sums.get(name, 0.0) + k * float(value)
    finally:
        del model._reparameterize_shared
        for module, mode in modes.items():
            module.train(mode)
    return sums


class PreservationGateCallback(Callback):
    """#12 (§10.6, ``cotrain``): the preservation gate. At fit start, :func:`preservation` of the pretrained VAE on the
    fixed validation ``rows`` is the baseline; every validation epoch (on the weights validation sees, EMA included)
    it logs ``val/forecast_mse_rel`` (relative forecast MSE increase), ``val/kld_active_frac``,
    ``val/logvar_prior_floor_frac``, ``val/delta_mu_sat_frac``, ``val/gate_ok`` (``forecast_mse_rel ≤ tolerance``) and
    ``<monitor>_gated``: ``monitor`` (:meth:`_monitored`: ``pl_module.val_metrics``, so this callback follows
    :class:`GuidEpochMetricsCallback`) when the gate holds, else ±∞, so ``best.ckpt`` never holds a failing epoch
    and a unit whose every epoch failed is ``failed`` (:func:`train_unit`). The frozen epoch is always a candidate as the
    ``frozen`` unit. One line per epoch (the baseline first) goes to ``preservation.jsonl``."""

    def __init__(self, rows: Mapping[str, Any], *, tolerance: float, monitor: str, mode: str, chunk: int,
                 output_dir: Any) -> None:
        super().__init__()
        self.rows, self.tolerance, self.monitor, self.mode, self.chunk = rows, float(tolerance), monitor, mode, chunk
        self.path, self.baseline = Path(output_dir) / "preservation.jsonl", None

    def _write(self, trainer, line: Mapping[str, Any]) -> None:
        if trainer.is_global_zero:
            with self.path.open("a") as fh:
                fh.write(json.dumps(json_safe(dict(line))) + "\n")

    def on_fit_start(self, trainer, pl_module) -> None:
        self.baseline = preservation(pl_module.backbone.source, self.rows, self.chunk)
        if trainer.is_global_zero:
            self.path.write_text("")
        self._write(trainer, {"epoch": "frozen", **self.baseline})

    def _monitored(self, trainer, pl_module) -> float:
        """This epoch's ``monitor``: the GUID metrics of :class:`GuidEpochMetricsCallback`, else any metric already
        logged this validation epoch (e.g. ``val/total_loss``)."""
        value = pl_module.val_metrics.get(self.monitor)
        value = trainer.callback_metrics.get(self.monitor) if value is None else value
        if value is None:
            raise KeyError(f"preservation gate: the monitor {self.monitor!r} was not logged this validation epoch")
        return float(value)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or not hasattr(pl_module, "val_metrics"):
            return
        now = preservation(pl_module.backbone.source, self.rows, self.chunk)
        rel = now.pop("forecast_mse") / self.baseline["forecast_mse"] - 1.0
        ok = rel <= self.tolerance
        fail = math.inf if self.mode == "min" else -math.inf
        values = {f"val/{name}": value for name, value in {"forecast_mse_rel": rel, **now, "gate_ok": float(ok)}.items()}
        values[f"{self.monitor}_gated"] = self._monitored(trainer, pl_module) if ok else fail
        for name, value in values.items():
            pl_module.log(name, value, on_step=False, on_epoch=True, logger=True)
        self._write(trainer, {"epoch": trainer.current_epoch, **values})


# ---- the experiment driver ---------------------------------------------------------------------------------------
def prior_correction(c: Classifier, unit: UnitData) -> List[float]:
    """§10.2 offset (K_out floats) subtracted from raw head outputs: ``log w`` when ``weighted_*`` uses class
    weights, ``-τ log π`` for ``logit_adjusted``, plus ``log(n_0/n_k)`` for ``sampler: class_balanced`` (it trains
    under a uniform class prior, as inverse weighting does); zeros otherwise (and for ordinal heads)."""
    loss, k = c.train.loss, len(unit.class_counts)
    k_out, adjusted = (k if c.labels.head == "multiclass" else 1), loss.name == "logit_adjusted"
    offsets = []
    if adjusted or loss.name.startswith("weighted"):
        weights = [1.0] * k if adjusted else class_weights(unit.class_counts, loss.weighting, loss.beta_en).tolist()
        offsets.append(prior_offset(weights, unit.priors["main"] if adjusted else None,
                                    loss.logit_adjust_tau if adjusted else 0.0))
    if c.train.sampler == "class_balanced":
        offsets.append(prior_offset(class_weights(unit.class_counts, "inverse")))
    if c.labels.head == "ordinal" or not offsets:
        return [0.0] * k_out
    off = sum(offsets)
    return [0.0, off][-k_out:] if isinstance(off, float) else off.tolist()


def online_backbone(c: Classifier, unit: UnitData, *, trainable: bool) -> OnlineFeatures:
    """The online regimes' backbone (§10.1): the configured :class:`~teb_vae.classifier.sources.VaeSource` behind the
    unit's frozen train scaler (fitted on the cache = the frozen encoder, which is also the encoder at the start of
    every P7 training stage: LPFT's stage 2 starts from stage 1's untouched backbone, so no refit is needed). With
    ``trainable``, the ``train.unfreeze`` allowlist (off until stage 2 under ``lpft``); otherwise frozen (scoring).

    ``cotrain`` (trainable): the joint objective (§10.6, :class:`~teb_vae.classifier.sources.OnlineFeatures`) with β
    pinned ``constant`` at the checkpoint's final value (the VAE task is never attached to a trainer, so its epoch is
    0 and a warm-up schedule would train at its start value). The VAE task's own optimizer and LR warm-up are never
    built: the classifier's optimizer and schedule govern every parameter. ``ponytail:`` for the same reason the
    task's tile phase ignores the epoch (one tiling per segment); attach the epoch if tiling diversity matters."""
    source = VaeSource(c.source, unfreeze=c.train.unfreeze if trainable else (), sample_z=c.source.vae.sample_z_train)
    if c.train.regime == "lpft":
        for _, p in source.allowlist():
            p.requires_grad_(False)
    joint = trainable and c.train.regime == "cotrain"
    if joint:
        source.task.hparams["beta_schedule"] = {"kind": "constant", "value": source.task._resolve_beta(1 << 30)}
    return OnlineFeatures(source, unit.scaler, n_values=unit.n_values, chunk=c.train.cotrain.vae_chunk,
                          cotrain=c.train.cotrain if joint else None)


class ClassifierTrainer(GraphModelBase):
    """One unit = (fold, seed, kind). Every output dir is the unit dir (F6); loaders from ``data.py`` (F13)."""

    TASK_CLS = ClassifierTask
    CHECKPOINT_STEM = "classifier"
    TRACKED_METRICS = TRACKED_METRICS
    PLOT_CONFIG_KEY = "classifier_plotting"

    def __init__(self, unit_config_path: Any, *, unit_dir: Any, unit: UnitData,
                 source: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(config_file_path=str(unit_config_path))
        self.unit, self.unit_dir, self.source = unit, Path(unit_dir), dict(source or {})
        self.output_base_dir = str(self.unit_dir)
        self.base_folder = self.unit_dir.name
        self.train_results_dir = str(self.unit_dir / "train_results")
        self.test_results_dir = self.aux_dir = self.tensorboard_dir = str(self.unit_dir)
        self.model_checkpoint_dir = str(self.unit_dir / "model_checkpoints")
        self.seed = int(self.config["general_config"]["seed"])
        self.reader: Optional[OnlineReader] = None  # online regimes: the raw HDF5 rows behind the cached ones

    def _validate_tracking_uri(self, uri: str) -> bool:
        """The base predates MLflow 3's database default: without this a ``sqlite://`` URI is rejected and the child
        run silently lands in ``./mlflow.db`` instead of beside its parent."""
        return uri.startswith(("sqlite://", "postgresql://", "mysql://")) or super()._validate_tracking_uri(uri)

    def create_model(self) -> None:
        c, u = self.unit.cfg, self.unit
        kwargs = dict(n_values=u.n_values, n_attn=u.n_attn, n_ctx=u.n_ctx, model_cfg=c.model.model_dump(mode="json"),
                      labels_cfg=c.labels.model_dump(mode="json"), priors=u.priors, n_cov=u.n_cov,
                      fusion=c.context.fusion)
        weights = class_weights(u.class_counts, c.train.loss.weighting, c.train.loss.beta_en).tolist()
        alpha = metrics.primary_alpha(c.eval.model_dump(mode="json"))  # pauc's FPR range (§10.2)
        criterion(c.train.loss, c.labels.head, weights, u.priors["main"], alpha)  # fail before the first step
        self.pl_model = self.TASK_CLS(
            ClassifierNet(**kwargs), lr=self.lr, weight_decay=c.train.optimizer.weight_decay,
            spike_breaker=self.config["advanced_config"].get("spike_breaker"), classifier_kwargs=kwargs,
            train_cfg=c.train.model_dump(mode="json"), class_weights=weights, prior_offset=prior_correction(c, u),
            loss_alpha=alpha)
        self.pl_model.checkpoint_extras = {
            "source_fingerprint": self.source.get("fingerprint"),
            "scaler": {"channels": list(u.scaler.channels), "center": u.scaler.center.tolist(),
                       "scale": u.scaler.scale.tolist(), "keep": u.scaler.keep.tolist(), "record": u.scaler.record},
            "labels": c.labels.model_dump(mode="json"),
            "feature_channels": {"values": list(u.channels), "attn": list(u.attn_channels),
                                 "context": list(u.context_columns),
                                 "covariates": [] if u.covariates is None else list(u.covariates.columns)},
            "covariates": None if u.covariates is None else list(u.covariates.variables),
        }
        if c.train.regime != "frozen_cached":
            self.pl_model.backbone = online_backbone(c, u, trainable=True)
            self.reader = OnlineReader(self.pl_model.backbone.source, u)
        self.apply_config_hyperparameters({"lr": self.lr}, self.pl_model)

    def setup_record(self) -> Dict[str, Any]:
        """``setup.json`` (§10.10.4)."""
        c, u, net, hp = self.unit.cfg, self.unit, self.pl_model.orig_model, self.pl_model.hparams
        counts = {s: {"n_segments": len(f), "n_guids": int(f["guid"].nunique()),
                      "guids_per_class": {str(k): int(v) for k, v in
                                          f.groupby("guid")["y"].first().value_counts().sort_index().items()}}
                  for s, f in u.frames.items()}
        backbone = self.pl_model.backbone
        total = sum(p.numel() for p in self.pl_model.parameters())  # classifier + VAE (online regimes)
        trainable = n_params(net) + (0 if backbone is None else sum(p.numel() for _, p in backbone.source.allowlist()))
        fingerprint = self.source.get("fingerprint") or {}
        return {
            "fold": u.fold, "shuffle_seed": u.shuffle_seed, "seed": self.seed, "splits": counts,
            "train_priors": u.priors, "class_counts": u.class_counts, "class_weights": hp.class_weights,
            "prior_offset": hp.prior_offset,
            "head_bias_init": (net.head.biases() if net.head.kind == "ordinal" else net.head.out.bias).tolist(),
            "source_fingerprint_hash": self.source.get("fingerprint_hash"), "cache_dir": self.source.get("cache_dir"),
            "vae_checkpoint_sha256": fingerprint.get("checkpoint_sha256"),
            "feature_channels": self.pl_model.checkpoint_extras["feature_channels"],
            "dropped_channels": u.scaler.record.get("dropped", []),
            "optimizer": c.train.optimizer.model_dump(mode="json"),
            "schedule": c.train.schedule.model_dump(mode="json"),
            "trainer": self.config["advanced_config"].get("trainer", {}),
            "dataloader": {"scope": c.model.scope, "batch_guids": c.train.batch_guids,
                           "batch_segments": c.train.batch_segments, "sampler": c.train.sampler,
                           "segment_dropout": c.train.segment_dropout, "num_workers": c.run.num_workers},
            "params": {"trainable": trainable, "frozen": total - trainable},
            "regime": {"name": c.train.regime, "unfreeze": list(c.train.unfreeze), "backbone_lr": c.train.backbone_lr,
                       "lpft_head_epochs": c.train.lpft_head_epochs, "vae_chunk": c.train.cotrain.vae_chunk,
                       "sample_z_train": c.source.vae.sample_z_train},
        }

    def _callbacks(self) -> List[Callback]:
        c, u, adv = self.unit.cfg, self.unit, self.config["advanced_config"]
        cbs_cfg = adv.get("callbacks", {}) or {}
        ckpt, plot = cbs_cfg.get("model_checkpoint", {}) or {}, cbs_cfg.get(self.PLOT_CONFIG_KEY, {}) or {}
        n_eval = int(plot.get("train_eval_guids", 256))
        train_loader = train_frame = None
        if n_eval > 0:
            guids = u.frames["train"]["guid"].unique()
            pick = np.random.default_rng(self.seed).choice(guids, min(n_eval, len(guids)), replace=False)
            train_frame = u.frames["train"][u.frames["train"]["guid"].isin(pick)].reset_index(drop=True)
            sub = dataclasses.replace(u, frames={"train_eval": train_frame})
            train_loader = make_loader(sub, "train_eval", train=False, seed=self.seed, reader=self.reader)
        history = MetricsLoggingCallback(tracked_metrics=tracked_metrics(c, adv))
        callbacks: List[Callback] = [
            GuidEpochMetricsCallback(val_frame=u.frames["val"], output_dir=self.train_results_dir, scope=c.model.scope,
                                     aggregators=c.model.segment_aggregators, tau=c.model.lse_tau,
                                     regime=c.train.regime, train_loader=train_loader, train_frame=train_frame,
                                     train_eval_every=plot.get("train_eval_every", 5)),
            history,
            MetricsHistoryCsvCallback(source=history, output_dir=self.train_results_dir),
            LossPlotCallback(output_dir=self.train_results_dir, plot_frequency=self.plot_every_epoch,
                             metric_filters=("*/total_loss", "*/loss_*", "val/guid_*"),
                             mlflow_logger=self.mlflow_logger),
            HyperparameterLoggingCallback(tracked_keys=("lr",), output_dir=self.train_results_dir,
                                          plot_frequency=self.plot_every_epoch, mlflow_logger=self.mlflow_logger),
        ]
        if plot.get("enabled", False):
            callbacks.append(ClassifierPlotCallback(self.train_results_dir, plot.get("every_n_epochs", 5),
                                                    plot.get("file_format", "pdf"), self.mlflow_logger))
        if c.train.regime == "lpft":
            callbacks.append(LpftUnfreezeCallback(head_epochs=c.train.lpft_head_epochs,
                                                  output_dir=self.train_results_dir))
        if c.train.regime == "cotrain":  # #12, right after #1: it reads the epoch's val metrics and logs before #2
            val = u.frames["val"]
            pick = np.sort(np.random.default_rng(self.seed).choice(len(val), min(GATE_SEGMENTS, len(val)),
                                                                   replace=False))
            callbacks.insert(1, PreservationGateCallback(
                attribute_dict_collate(self.reader.bind(val)(pick)), tolerance=c.train.cotrain.gates["forecast_mse_rel"],
                monitor=selection_monitor(adv)[0], mode=selection_monitor(adv)[1], chunk=c.train.cotrain.vae_chunk,
                output_dir=self.train_results_dir))
        # §10.10.2 #6: best.ckpt is chosen on the early-stopping monitor (cotrain: its gated twin, #12)
        monitor, mode = selection_monitor(adv, c.train.regime)
        callbacks.append(ModelCheckpoint(dirpath=self.model_checkpoint_dir, filename="best", save_top_k=1,
                                         monitor=monitor, mode=mode, auto_insert_metric_name=False,
                                         save_last=ckpt.get("save_last", False)))
        return callbacks

    def _build_trainer_kwargs(self, callbacks, model=None) -> dict:
        """Base kwargs, then: LR monitor per step with EMA right after it (§10.10.2 #8-9), no distributed sampler
        (custom batch sampler), CPU when ``cuda_devices`` is empty (F5)."""
        kw = super()._build_trainer_kwargs(callbacks, model=model)
        cbs = [LearningRateMonitor(logging_interval="step") if isinstance(cb, LearningRateMonitor) else cb
               for cb in kw["callbacks"]]
        ema = self.unit.cfg.train.ema
        if ema is not None:
            at = next(i for i, cb in enumerate(cbs) if isinstance(cb, LearningRateMonitor)) + 1
            # use_buffers=False: the net has no buffers, and an online backbone's are constants, some integer or bool
            # (index maps), which an EMA lerp would round away.
            cbs.insert(at, EMAWeightAveraging(decay=ema, use_buffers=False))
        kw.update(callbacks=cbs, use_distributed_sampler=False)
        if not self.cuda_devices:
            kw.update(accelerator="cpu", devices=1)
            kw.pop("strategy", None)
        return kw

    def train_model(self, train_loader=None, validation_loader=None) -> Dict[str, Any]:
        """Fit on the unit's loaders; returns the ``fold_results.json`` record of ``best.ckpt``."""
        u = self.unit
        train_loader = train_loader or make_loader(u, "train", train=True, seed=self.seed, reader=self.reader)
        validation_loader = validation_loader or make_loader(u, "val", train=False, seed=self.seed, reader=self.reader)
        trainer = self.build_trainer(self._callbacks(), model=self.pl_model)
        trainer.fit(self.pl_model, train_loader, validation_loader)
        cb = trainer.checkpoint_callback
        if not cb.best_model_path:
            raise RuntimeError(f"no best checkpoint was written under {self.model_checkpoint_dir}")
        epoch = int(torch.load(cb.best_model_path, map_location="cpu", weights_only=False)["epoch"])
        lines = [json.loads(line) for line in
                 (Path(self.train_results_dir) / "epoch_summary.jsonl").read_text().splitlines()]
        record = {"best_ckpt": cb.best_model_path, "monitor": cb.monitor, "best_score": float(cb.best_model_score),
                  "best_epoch": epoch, "epochs_run": trainer.current_epoch,
                  "best_val": next((line for line in lines if line["epoch"] == epoch), None)}
        if not math.isfinite(record["best_score"]):  # cotrain: every epoch failed the preservation gate (§10.6)
            record["gate_failed_every_epoch"] = True  # train_unit fails the unit: no epoch is selectable
        return record


# ---- the unit (outer loop steps 1-3) ------------------------------------------------------------------------------
#: This process's own log pair, relative to the run dir, which :func:`_teardown` re-installs after each unit: the run's
#: ``run.log`` in the main process; a fold process (``run.fold_job``, SPEC §14.4) points it at its fold's files, so it
#: never writes (or rotates) the run-level log another process owns.
PROCESS_LOGS: Tuple[str, str] = ("run.log", "run.jsonl")


def _teardown(gm: Optional[ClassifierTrainer], run_dir: Path) -> None:
    """F7: the base leaks every driver (atexit upload), its system-metrics monitor, and replaces loguru's sinks."""
    if gm is not None:
        gm.upload_run_logs()
        atexit.unregister(gm.upload_run_logs)
        monitor = getattr(gm, "_system_metrics_monitor", None)
        if monitor is not None:
            monitor.finish()
        gm.__dict__.pop("pl_model", None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    setup_logging(file_path=str(run_dir / PROCESS_LOGS[0]), json_path=str(run_dir / PROCESS_LOGS[1]),
                  compression=None)


def train_unit(cfg: Config, run_dir: Any, manifest: Mapping[str, Any], *, fold: int, seed: int, kind: str,
               unit: Optional[UnitData] = None, mlflow_parent_id: Optional[str] = None) -> Dict[str, Any]:
    """Train one unit (§10.10.1 steps 1-3) and return its ``fold_results.json`` record.

    Writes under ``unit_dir(run_dir, fold, seed, kind)``: ``model_checkpoints/{resolved_config.yaml, best.ckpt,
    last.ckpt}``, ``train_results/*``, ``setup.json``, ``scaler.json``, ``fold_results.json``. ``kind="shuffled"``
    trains on train labels permuted with ``seed + 1000·fold``, ``kind="noind"`` on the ``no_indicator`` context
    (:func:`~teb_vae.classifier.data.without_indicators`), ``kind="frozen"`` on the cache under
    :func:`~teb_vae.classifier.config.frozen_baseline` (an online regime's frozen baseline, §10.1), as the shuffled
    control is ("one seed, frozen regime", §10.9.3). A failure is recorded (``status: failed`` + traceback) and
    re-raised only with ``run.fail_fast``; a cotrain unit whose every epoch failed the preservation gate is one (§10.6:
    no selectable checkpoint), so it is never locked or predicted.
    """
    run_dir = Path(run_dir)
    if kind in ("frozen", "shuffled"):
        cfg = frozen_baseline(cfg)
        unit = None if unit is None else dataclasses.replace(unit, cfg=cfg.classifier)
    shuffle_seed = seed + 1000 * fold if kind == "shuffled" else None  # §10.7: a fresh permutation per fold
    if unit is not None and (unit.fold, unit.shuffle_seed) != (fold, shuffle_seed):
        raise ValueError(f"unit is fold {unit.fold} / shuffle_seed {unit.shuffle_seed}; this {kind} unit needs "
                         f"fold {fold} / shuffle_seed {shuffle_seed}")
    out = unit_dir(run_dir, fold, seed, kind)
    (out / "model_checkpoints").mkdir(parents=True, exist_ok=True)
    for stale in (out / "model_checkpoints").glob("*.ckpt"):  # a re-run unit must not inherit best-v1.ckpt naming
        stale.unlink()
    config_path = out / "model_checkpoints" / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(unit_config(cfg, run_dir=run_dir, fold=fold, seed=seed, kind=kind,
                                                      mlflow_parent_id=mlflow_parent_id), sort_keys=False))
    record: Dict[str, Any] = {"fold": fold, "seed": seed, "kind": kind, "unit_dir": str(out)}
    gm, started, error = None, time.perf_counter(), None
    try:
        unit = unit or build_unit(without_indicators(cfg.classifier) if kind == "noind" else cfg, run_dir,
                                  manifest["source"], fold, shuffle_seed=shuffle_seed)
        gm = ClassifierTrainer(config_path, unit_dir=out, unit=unit, source=manifest.get("source"))
        gm.setup_config()
        record["mlflow_run_id"] = getattr(gm.mlflow_logger, "run_id", None)  # the child run (§10.10.5)
        gm.create_model()
        unit.scaler.save(out / "scaler.json")
        (out / "setup.json").write_text(json.dumps(json_safe(gm.setup_record()), indent=2))
        record |= gm.train_model()
        if record.get("gate_failed_every_epoch"):  # §10.6: only gate-passing epochs are selectable; none is
            raise RuntimeError(f"preservation gate failed at every epoch ({record['monitor']} is {record['best_score']} "
                               f"throughout): no selectable checkpoint; the frozen unit is the baseline")
        record["status"] = "done"
    except Exception as exc:
        logger.exception(f"unit fold {fold} seed {seed} {kind} failed")
        record |= {"status": "failed", "error": traceback.format_exc()}
        error = exc
    record["train_seconds"] = round(time.perf_counter() - started, 2)
    (out / "fold_results.json").write_text(json.dumps(json_safe(record), indent=2))
    _teardown(gm, run_dir)
    del gm
    logger.info(f"unit fold {fold} seed {seed} {kind}: {record['status']} in {record['train_seconds']} s"
                + (f", best {record['monitor']} {record['best_score']:.4f} at epoch {record['best_epoch']}"
                   if record["status"] == "done" else ""))
    if error is not None and cfg.classifier.run.fail_fast:
        raise error
    return record


# ---- prediction --------------------------------------------------------------------------------------------------
BACKBONE = "backbone."


def load_net(ckpt_path: Any) -> Tuple[ClassifierNet, Dict[str, Any]]:
    """``(net, hyper_parameters)`` from a unit checkpoint (``state_dict`` = the EMA weights when EMA is on); an online
    unit's ``backbone.*`` weights are left to :func:`backbone_state`."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    check_model_class(blob, ClassifierNet.__name__)
    net = ClassifierNet(**blob["classifier_kwargs"])
    state = {k: v for k, v in blob["state_dict"].items() if not k.startswith(BACKBONE)}
    if load_checkpoint_strict(net, dict(blob, state_dict=state)) is None:
        raise RuntimeError(f"{ckpt_path} does not align with ClassifierNet(**classifier_kwargs)")
    return net.eval(), blob["hyper_parameters"]


def backbone_state(ckpt_path: Any) -> Dict[str, torch.Tensor]:
    """The :class:`~teb_vae.classifier.sources.OnlineFeatures` state of an online unit's checkpoint (the fine-tuned
    VAE and the scaler), ``{}`` for a cached unit."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {k[len(BACKBONE):]: v for k, v in blob["state_dict"].items() if k.startswith(BACKBONE)}


def score_split(unit_dir_or_task: Any, unit: UnitData, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``(segments, guids)`` of ``split``: one row per retained segment / GUID, RAW (uncalibrated) logits after the
    §10.2 prior correction, from ``<unit_dir>/model_checkpoints/best.ckpt`` (or a live :class:`ClassifierTask`).

    Segments: ``unit.frames[split]`` columns + ``logit_seg``, ``logit_online``, ``attn_entropy`` (+ ``p_c0..2``;
    ``ord_score`` for a CORAL head; ``pseg_c0..2`` for a sequence-scope multiclass segment head, see :func:`_rows`).
    GUIDs: ``guid, fold, split, y, class_code, n_segments, score_final`` (+ ``score_<agg>`` in segment scope,
    ``p_c0..2``, ``ord_score``).

    An online unit (its checkpoint holds ``backbone.*``, whatever ``unit.cfg`` says) is scored online: the configured
    VAE rebuilt frozen, its weights and scaler replaced by the checkpoint's (:func:`backbone_state`).
    """
    if isinstance(unit_dir_or_task, ClassifierTask):
        net, hp, features = unit_dir_or_task.orig_model, unit_dir_or_task.hparams, unit_dir_or_task.backbone
    else:
        net, hp, features = _unit_model(unit_dir_or_task, unit)
    device = unit.cfg.run.device if torch.cuda.is_available() else "cpu"
    reader = None if features is None else OnlineReader(features.source, unit)
    parts = _score(net.to(device), make_loader(unit, split, train=False, seed=0, reader=reader),
                   torch.tensor(hp["prior_offset"], dtype=torch.float32), device,
                   None if features is None else features.to(device))
    m = unit.cfg.model
    return _frames(unit.frames[split], parts, scope=m.scope, aggregators=m.segment_aggregators, tau=m.lse_tau,
                   causal=net.causal)


def _unit_model(unit_dir: Any, unit: UnitData) -> Tuple[ClassifierNet, Dict[str, Any], Optional[OnlineFeatures]]:
    """``(net, hyper_parameters, backbone | None)`` of ``<unit_dir>/model_checkpoints/best.ckpt``. An online unit (its
    checkpoint holds ``backbone.*``, whatever ``unit.cfg`` says) gets the configured VAE rebuilt frozen, its weights and
    scaler replaced by the checkpoint's (:func:`backbone_state`)."""
    ckpt = Path(unit_dir) / "model_checkpoints" / "best.ckpt"
    (net, hp), state, features = load_net(ckpt), backbone_state(ckpt), None
    if state:
        features = online_backbone(unit.cfg, unit, trainable=False)
        features.load_state_dict(state)
    return net, hp, features


# ---- E6 attribution (§11.12; hand-written Integrated Gradients: captum is not a dependency) ------------------------
#: ``predictions/attribution.parquet`` columns: per GUID and channel group, the summed |IG| and signed IG over its
#: segments, steps and channels, and the GUID's score at the input and at the baseline (``f_x - f_0`` = Σ signed).
ATTRIBUTION_COLUMNS = ["fold", "seed", "split", "guid", "y", "class_code", "channel_group", "abs", "signed", "f_x",
                       "f_0"]


def guid_scores(net: ClassifierNet, batch: Mapping[str, torch.Tensor], offset: torch.Tensor, *,
                aggregator: str, tau: float) -> torch.Tensor:
    """The raw prior-corrected GUID score ``score_final`` of every GUID in ``batch``, differentiable in ``x``/``attn``:
    sequence scope the head's alarm logit at the last real position; segment scope ``aggregator`` over the segments'
    alarm logits, grouped by ``batch["guid"]`` (whole GUIDs per batch: :func:`attribute_split`)."""
    out = net(batch)
    if net.scope == "sequence":
        pos = net.head.score(out["pos"] - offset)
        return pos[torch.arange(len(pos), device=pos.device), batch["seg_mask"].sum(1) - 1]
    s, codes = net.head.score(out["seg"] - offset), batch["guid"]
    return torch.stack([aggregate(s[codes == g], aggregator, tau) for g in torch.unique_consecutive(codes)])


def integrated_gradients(net: ClassifierNet, batch: Mapping[str, torch.Tensor], offset: torch.Tensor, *,
                         n_steps: int, aggregator: str, tau: float
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrated Gradients (Sundararajan 2017) of :func:`guid_scores` w.r.t. the scaled features ``[x || attn]`` on the
    straight path from the baseline 0 (the train-fold mean, §8.5) to the input, by the midpoint rule over ``n_steps``;
    context and covariates stay at their values. Returns ``(attributions like [x || attn], f(x), f(0))``; per GUID the
    attributions sum to ``f(x) - f(0)`` up to the quadrature error (completeness)."""
    keys = [k for k in ("x", "attn") if k in batch]
    x = {k: batch[k].float() for k in keys}
    total = {k: torch.zeros_like(v) for k, v in x.items()}
    for a in (torch.arange(n_steps, dtype=torch.float32) + 0.5) / n_steps:
        xa = {k: (float(a) * v).requires_grad_() for k, v in x.items()}
        f = guid_scores(net, {**batch, **xa}, offset, aggregator=aggregator, tau=tau)
        for k, g in zip(keys, torch.autograd.grad(f.sum(), [xa[k] for k in keys])):
            total[k] += g
    with torch.no_grad():
        f_x = guid_scores(net, batch, offset, aggregator=aggregator, tau=tau)
        f_0 = guid_scores(net, {**batch, **{k: torch.zeros_like(v) for k, v in x.items()}}, offset,
                          aggregator=aggregator, tau=tau)
    return torch.cat([x[k] * total[k] / n_steps for k in keys], -1), f_x, f_0


def _whole_guid_batches(codes: np.ndarray, size: int) -> List[List[int]]:
    """Frame-order item batches of whole GUIDs (``codes`` contiguous per GUID), about ``size`` items each."""
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    batches, cur = [], []
    for s, e in zip(starts, np.r_[starts[1:], len(codes)]):
        if cur and len(cur) + e - s > size:
            batches.append(cur)
            cur = []
        cur += range(int(s), int(e))
    return batches + ([cur] if cur else [])


def attribute_split(unit_dir: Any, unit: UnitData, split: str, n_steps: int) -> pd.DataFrame:
    """E6 rows (:data:`ATTRIBUTION_COLUMNS` less ``fold/seed/split``) of ``split`` from the unit's ``best.ckpt``: the
    :func:`integrated_gradients` of each GUID's ``score_final`` (raw, prior-corrected; 3-class: the collapsed alarm
    logit) w.r.t. the features the head reads (after the backbone of an online unit), summed per channel group (the
    channel name before ``[``) over segments and steps. Segment scope batches whole GUIDs, since the aggregator couples
    their segments."""
    c, m = unit.cfg, unit.cfg.model
    net, hp, features = _unit_model(unit_dir, unit)
    device = c.run.device if torch.cuda.is_available() else "cpu"
    net, off = net.to(device).eval(), torch.tensor(hp["prior_offset"], dtype=torch.float32, device=device)
    features = None if features is None else features.to(device).eval()
    reader = None if features is None else OnlineReader(features.source, unit)
    if m.scope == "sequence":
        loader = make_loader(unit, split, train=False, seed=0, reader=reader)
    else:
        ds = SegmentDataset(unit, split, read=None if reader is None else reader.bind(unit.frames[split]))
        loader = torch.utils.data.DataLoader(ds, batch_sampler=_whole_guid_batches(ds.codes, c.train.batch_segments),
                                             collate_fn=collate_segments)
    names = pd.Index([*unit.channels, *unit.attn_channels]).str.split("[").str[0]
    acc: Dict[str, list] = {"guid": [], "abs": [], "signed": [], "f_x": [], "f_0": []}
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        if features is not None:
            with torch.no_grad():
                batch = features(batch)
        attr, f_x, f_0 = integrated_gradients(net, batch, off, n_steps=n_steps, aggregator=m.segment_aggregators[0],
                                              tau=m.lse_tau)
        if m.scope == "sequence":  # (B, N, T, C) -> per GUID (B, C)
            A, S, codes = attr.abs().sum((1, 2)), attr.sum((1, 2)), batch["guid"]
        else:  # (M, T, C) -> per GUID in batch order, as guid_scores groups them
            codes, inv = torch.unique_consecutive(batch["guid"], return_inverse=True)
            A, S = (attr.new_zeros(len(codes), attr.shape[-1]).index_add_(0, inv, a.sum(1)) for a in (attr.abs(), attr))
        for key, v in (("abs", A), ("signed", S), ("f_x", f_x), ("f_0", f_0)):
            acc[key].append(v.detach().cpu().numpy())
        acc["guid"].append(loader.dataset.guids[codes.cpu().numpy()])
    net.cpu()
    if not acc["guid"]:
        return pd.DataFrame(columns=ATTRIBUTION_COLUMNS[3:])
    guids = np.concatenate(acc["guid"]).astype(str)
    per = {key: pd.DataFrame(np.concatenate(acc[key]), index=guids, columns=names).T.groupby(level=0, sort=False).sum().T
           .stack() for key in ("abs", "signed")}  # (guid, channel_group) -> summed over the group's channels
    out = pd.concat(per, axis=1).rename_axis(["guid", "channel_group"]).reset_index()
    out = out.merge(pd.DataFrame({"guid": guids, "f_x": np.concatenate(acc["f_x"]), "f_0": np.concatenate(acc["f_0"])}),
                    on="guid")
    first = unit.frames[split].groupby("guid", sort=False)[["y", "class_code"]].first()
    out = out.merge(first, left_on="guid", right_index=True)
    return out[["guid", "y", "class_code", "channel_group", "abs", "signed", "f_x", "f_0"]]


# ---- calibration (§10.8) ------------------------------------------------------------------------------------------
P3 = [f"p_c{k}" for k in range(3)]
LOG_T_BOUNDS = (-math.log(100.0), math.log(100.0))


def _coral_logp(o: np.ndarray) -> np.ndarray:
    """:func:`~teb_vae.classifier.losses.coral_log_probs` on numpy cumulative logits (n, K-1) -> (n, K)."""
    return coral_log_probs(torch.as_tensor(o, dtype=torch.float64)).numpy()


def fit_calibration(guid_logits: Any, y: Any, method: str, *, p3: Any = None, ord_score: Any = None) -> Dict[str, Any]:
    """Fit on val GUID-level (final-position) prior-corrected predictions.

    Binary (no ``p3`` / ``ord_score``): the scalar logits against ``y > 0``. ``temperature``: T > 0 by L-BFGS on the
    NLL of ``s / T`` over ``log T`` within [1/100, 100], as the 3-class T below; ``platt``: ``a·s + b`` (unpenalised
    logistic regression); ``none``.

    3-class, ``y`` the class 0..K-1; the record gains ``head``. ``p3`` (n, K) -> ``multiclass``: one ``temperature``
    over the K logits, L-BFGS on the NLL of ``softmax(log p3 / T)`` (= ``softmax(logits / T)``). ``ord_score`` (n,), the
    CORAL g -> ``ordinal``: ``scale`` a > 0 on g (``temperature`` = 1/a) and K-1 refit ordered ``offsets`` b', by
    L-BFGS on the cumulative-link NLL of ``P(Y > k) = σ(a·g + b'_k)``, so every cut-point keeps g's ranking; ``shift``
    = b'_1 - a·b_1 (b_1 = ``guid_logits - g``, the head's first bias) maps any raw alarm logit ``s = g + b_1`` to
    ``a·s + shift = logit P_cal(Y ≥ 1)``. ``none`` records the head only. The 3-class T (1/a) stays within
    [1/100, 100]: a val set its score separates drives the MLE to T -> 0 (``ponytail:`` a hard bound; a prior on
    log T if real folds hit it).
    """
    s, n = np.asarray(guid_logits, dtype=np.float64), int(np.size(guid_logits))
    if p3 is not None or ord_score is not None:
        head, t = ("multiclass" if p3 is not None else "ordinal"), np.asarray(y, dtype=np.int64)
        if method == "none":
            return {"method": "none", "head": head}
        if method != "temperature":
            raise ValueError(f"calibration.method {method!r} does not fit a {head} head (temperature | none)")
        rows = np.arange(n)
        if head == "multiclass":
            lp = np.log(np.clip(np.asarray(p3, dtype=np.float64), 1e-300, None))

            def nll(v: np.ndarray) -> float:
                z = lp / np.exp(v[0])
                return -float(np.mean(z[rows, t] - logsumexp(z, axis=1)))

            res = minimize(nll, np.zeros(1), method="L-BFGS-B", bounds=[LOG_T_BOUNDS])
            return {"method": "temperature", "head": head, "temperature": float(np.exp(res.x[0])),
                    "nll": float(res.fun), "n": n}
        g = np.asarray(ord_score, dtype=np.float64)

        def unpack(v: np.ndarray) -> Tuple[float, np.ndarray]:  # b'_k = b'_1 - Σ_{j<k} softplus(δ_j): ordered
            return float(np.exp(v[0])), v[1] - np.r_[0.0, np.cumsum(np.logaddexp(0.0, v[2:]))]

        def nll(v: np.ndarray) -> float:
            a, b = unpack(v)
            return -float(np.mean(_coral_logp(a * g[:, None] + b)[rows, t]))

        # start: a = 1, b'_k = logit P̂(Y > k) - mean g (3 classes: three_class is the only ordinal task)
        b0 = logit(np.clip([(t > k).mean() for k in range(2)], 1e-3, 1 - 1e-3)) - g.mean()
        start = np.r_[0.0, b0[0], np.log(np.expm1(np.maximum(b0[0] - b0[1], 1e-3)))]
        res = minimize(nll, start, method="L-BFGS-B", bounds=[LOG_T_BOUNDS] + [(None, None)] * (start.size - 1))
        a, b = unpack(res.x)
        return {"method": "temperature", "head": head, "scale": a, "temperature": 1.0 / a, "offsets": b.tolist(),
                "shift": float(b[0] - a * np.median(s - g)), "nll": float(res.fun), "n": n}
    t = (np.asarray(y) > 0).astype(np.float64)
    if method == "none":
        return {"method": "none"}
    if method == "temperature":
        def nll(v: np.ndarray) -> float:
            z = s / np.exp(v[0])
            return -float(np.mean(t * log_expit(z) + (1 - t) * log_expit(-z)))

        res = minimize(nll, np.zeros(1), method="L-BFGS-B", bounds=[LOG_T_BOUNDS])
        return {"method": "temperature", "temperature": float(np.exp(res.x[0])), "nll": float(res.fun),
                "n": int(s.size)}
    if method == "platt":
        lr = LogisticRegression(C=np.inf, max_iter=1000).fit(s[:, None], t)
        return {"method": "platt", "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0]), "n": int(s.size)}
    raise ValueError(f"unknown calibration.method {method!r}")


def unit_calibration(c: Classifier, guids: pd.DataFrame) -> Dict[str, Any]:
    """:func:`fit_calibration` of one unit on its val GUID rows (:func:`score_split`): ``score_final`` against ``y``;
    on ``three_class``, the class probabilities (multiclass) or CORAL g (ordinal) against ``class_code - 1``, and in
    segment scope the record adds ``segment_scope`` (aggregators, τ), how :func:`calibrate_frames` rebuilds a
    multiclass head's online and GUID scores from its calibrated segment logits."""
    if c.labels.task != "three_class":
        return fit_calibration(guids["score_final"], guids["y"], c.calibration.method)
    kw = {"ord_score": guids["ord_score"]} if c.labels.head == "ordinal" else {"p3": guids[P3]}
    cal = fit_calibration(guids["score_final"], guids["class_code"] - 1, c.calibration.method, **kw)
    if c.model.scope == "segment":
        cal["segment_scope"] = {"aggregators": list(c.model.segment_aggregators), "lse_tau": c.model.lse_tau}
    return cal


def apply_calibration(logits: Any, cal: Mapping[str, Any]) -> Any:
    """Calibrated alarm logits (same shape; NaN stays NaN). Ordinal: ``scale·s + shift``. A multiclass head's alarm
    logit is no function of itself under T: :func:`calibrate_frames` rebuilds it from calibrated probabilities."""
    logits = np.asarray(logits, dtype=np.float64)
    if cal["method"] == "none":
        return logits
    if cal.get("head") == "ordinal":
        return cal["scale"] * logits + cal["shift"]
    if cal.get("head") == "multiclass":
        raise ValueError("a multiclass alarm logit is calibrated through its class probabilities (calibrate_frames)")
    if cal["method"] == "temperature":
        return logits / cal["temperature"]
    if cal["method"] == "platt":
        return cal["a"] * logits + cal["b"]
    return logits


#: Stored calibrated class probabilities lie in [P_EPS, 1 - P_EPS], so every ``logit(p_c<k>_cal)`` (OvR thresholds and
#: analyses) is finite. ``ponytail:`` a clip ties OvR scores beyond ±27.6 logits; store log-probabilities if that bites.
P_EPS = 1e-12


def calibrated_logp(p3: Any, cal: Mapping[str, Any], ord_score: Any = None) -> np.ndarray:
    """Calibrated class log-probabilities (n, K) of a 3-class record, in log space so that no temperature underflows
    them: ``log softmax(log p3 / T)`` (multiclass), ``log coral(scale·g + offsets)`` from ``ord_score`` (ordinal),
    ``log p3`` under ``none``. An exact 0 in the stored float32 ``p3`` reads as 1e-300. NaN rows stay NaN."""
    lp = np.log(np.clip(np.asarray(p3, dtype=np.float64), 1e-300, None))
    if cal["method"] == "none":
        return lp
    if cal["head"] == "ordinal":
        g = np.asarray(ord_score, dtype=np.float64)
        return _coral_logp(cal["scale"] * g[:, None] + np.asarray(cal["offsets"]))
    z = lp / cal["temperature"]
    return z - logsumexp(z, axis=1, keepdims=True)


def calibrate_probs(p3: Any, cal: Mapping[str, Any], ord_score: Any = None) -> np.ndarray:
    """Calibrated class probabilities (n, K): :func:`calibrated_logp`, exponentiated and clipped to :data:`P_EPS`."""
    return np.clip(np.exp(calibrated_logp(p3, cal, ord_score)), P_EPS, 1.0 - P_EPS)


def _alarm_lp(lp: np.ndarray) -> np.ndarray:
    """``logit P(Y ≥ 1)`` of class log-probabilities (n, K): ``logsumexp_{k≥1} lp_k - lp_0``, finite whenever lp is."""
    return logsumexp(lp[:, 1:], axis=1) - lp[:, 0]


def alarm_logit(P: Any) -> np.ndarray:
    """The collapsed adverse score of class probabilities (n, K): ``log Σ_{k≥1} p_k - log p_0 = logit P(Y ≥ 1)``."""
    with np.errstate(divide="ignore"):
        return _alarm_lp(np.log(np.asarray(P, dtype=np.float64)))


def calibrate_frames(seg: pd.DataFrame, gd: pd.DataFrame, cal: Mapping[str, Any]
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``(seg, gd)`` plus the ``*_cal`` twin of every score: ``logit_seg``, ``logit_online``, GUID ``score_*``; a
    3-class record (``cal["head"]``) adds ``p_c<k>_cal`` on every row and ``ord_score`` (NaN for a multiclass head).

    Binary and ordinal alarm logits go through :func:`apply_calibration` (increasing and affine, so it commutes with
    the segment aggregators, bar lse, as in P3). A multiclass one is rebuilt from the calibrated probabilities of the
    output it came from (``p_c*``; ``pseg_c*`` for a sequence-scope segment head), so ``σ(score_cal) = 1 - p_c0_cal``;
    in segment scope (``cal["segment_scope"]``) the online and GUID scores re-aggregate ``logit_seg_cal``. NaN stays
    NaN. ``ponytail:`` segment-scope GUID probabilities are segment means, calibrated as one row (an ordinal GUID from
    its ``ord_score``, read like ``score_final``), so they are not the mean of the segments' ``p_c*_cal``.
    """
    scores, head = [c for c in gd if c.startswith("score_")], cal.get("head")
    if head is not None:  # log space throughout, so a small T never makes an alarm logit infinite
        lp_seg, lp_gd = (calibrated_logp(f[P3], cal, f.get("ord_score")) for f in (seg, gd))
        seg, gd = (f.assign(ord_score=f["ord_score"] if "ord_score" in f else NAN,
                            **dict(zip([f"{c}_cal" for c in P3], np.clip(np.exp(lp), P_EPS, 1.0 - P_EPS).T)))
                   for f, lp in ((seg, lp_seg), (gd, lp_gd)))
    if head != "multiclass" or cal["method"] == "none":
        return (seg.assign(**{f"{c}_cal": apply_calibration(seg[c], cal) for c in ("logit_seg", "logit_online")}),
                gd.assign(**{f"{c}_cal": apply_calibration(gd[c], cal) for c in scores}))
    online = _alarm_lp(lp_seg)
    agg = cal.get("segment_scope")
    if agg is not None:
        seg = seg.assign(logit_seg_cal=online)
        seg["logit_online_cal"], per_guid = _aggregates(seg, "logit_seg_cal", agg["aggregators"], agg["lse_tau"])
        names = dict(zip(["score_final", *(f"score_{a}" for a in agg["aggregators"][1:])], agg["aggregators"]))
        return seg, gd.assign(**{f"{c}_cal": gd["guid"].map(per_guid[names[c]]) for c in scores})
    pseg = [f"pseg_c{k}" for k in range(3)]
    seg_cal = _alarm_lp(calibrated_logp(seg[pseg], cal)) if set(pseg) <= set(seg) else NAN
    return (seg.assign(logit_seg_cal=np.where(seg["logit_seg"].isna(), NAN, seg_cal),
                       logit_online_cal=np.where(seg["logit_online"].isna(), NAN, online)),
            gd.assign(score_final_cal=_alarm_lp(lp_gd)))
