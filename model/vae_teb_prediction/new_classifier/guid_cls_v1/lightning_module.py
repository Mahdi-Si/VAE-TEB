"""PyTorch Lightning wrapper for ``GuidOutcomeClassifier`` (causal AR).

* Bypasses ``LightningModelBase``'s automatic ``torch.compile`` (variable-N
  batches plus the live-VAE branch trip dynamic recompilation).
* Causal-AR design: every position contributes to the loss in one forward
  pass per GUID. Prefix-length sampling has been removed — short prefixes
  are now naturally trained by the per-position loss.
* Keeps :func:`apply_segment_dropout` as a regulariser (consistent with the
  PRD's segment-dropout intent).
* Validation metrics use the last-valid position's prediction as the
  single GUID summary so ``binary_auroc`` / ``macro_f1`` semantics match
  the natural "final clinical decision" at the end of the recording.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from loguru import logger

from train.pl_model_base import LightningModelBase, MetricDict

from model.vae_teb_prediction.new_classifier.guid_cls_v1.collate import (
    build_relative_time_bucket_index,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_classifier import (
    GuidOutcomeClassifier,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.losses import (
    GuidClassifierLoss,
    LossWeights,
)


_SECONDS_PER_HOUR = 3600.0
_SEGMENT_DURATION_SEC = 1200.0


def _recompute_delta_features(
    batch: Dict[str, torch.Tensor],
    *,
    rel_num_buckets: int,
    rel_d_max: float,
) -> None:
    """Recompute Δt-derived features after ``segment_mask`` has changed.

    When :func:`apply_segment_dropout` flips a non-terminal segment's mask to
    False, the gap between that segment's neighbours effectively widens — but
    the collate's precomputed ``delta_t_hours``, ``cum_monitor_hours``,
    ``gap_ratio`` and ``rel_bucket_idx`` still reflect the pre-dropout
    ordering. This helper rebuilds those tensors from the raw ``epoch``
    tensor and the current ``segment_mask`` so the transformer's
    relative-time bias sees the wider gap between surviving neighbours.

    ``c_meta`` is *not* touched: its columns are TLO/SSO statistics
    (segment-intrinsic, unaffected by dropout) — the cumulative / Δt / κ
    summaries are no longer part of ``c_meta`` (PRD §4.4).

    Operates in-place on ``batch``. Padded / dropped positions retain their
    default (zero) values since the attention mask hides them anyway.

    Args:
        batch: Collated batch dict. Must contain ``segment_mask`` and
            ``epoch``.
        rel_num_buckets: Number of relative-time bias buckets (must match
            the transformer's bias-table size).
        rel_d_max: Δt saturation horizon in 20-min slots.
    """
    seg_mask: torch.Tensor = batch["segment_mask"]
    epoch: torch.Tensor = batch["epoch"]
    if epoch.dtype not in (torch.float32, torch.float64):
        epoch = epoch.float()
    B, N = seg_mask.shape
    device = seg_mask.device

    delta_t = torch.zeros(B, N, dtype=torch.float32, device=device)
    cum_h = torch.zeros(B, N, dtype=torch.float32, device=device)
    gap_ratio = torch.zeros(B, N, dtype=torch.float32, device=device)

    # Per-row reconstruction from surviving positions. ``epoch`` at padded /
    # zeroed positions is ignored because ``segment_mask`` gates the inner
    # condition. The outer loop is B*N — for the typical batch shapes in this
    # project (B ≤ 16, N ≤ 40) the cost is negligible.
    seg_mask_cpu = seg_mask.detach().cpu()
    epoch_cpu = epoch.detach().cpu()
    for b in range(B):
        prev_epoch: Optional[float] = None
        cum = 0.0
        for j in range(N):
            if not bool(seg_mask_cpu[b, j].item()):
                continue
            ep = float(epoch_cpu[b, j].item())
            if prev_epoch is None:
                dt_sec = 0.0
            else:
                dt_sec = ep - prev_epoch
            dt_h = dt_sec / _SECONDS_PER_HOUR
            delta_t[b, j] = dt_h
            cum += dt_h
            cum_h[b, j] = cum
            gap_ratio[b, j] = max(0.0, dt_sec / _SEGMENT_DURATION_SEC - 1.0)
            prev_epoch = ep

    batch["delta_t_hours"] = delta_t
    batch["cum_monitor_hours"] = cum_h
    batch["gap_ratio"] = gap_ratio
    batch["rel_bucket_idx"] = build_relative_time_bucket_index(
        cum_h, num_buckets=rel_num_buckets, d_max=rel_d_max
    )
    # ``c_meta`` carries only TLO/SSO (5-d, all segment-intrinsic), so
    # nothing in it depends on Δt — no rewrite needed. The Δt / cum_h /
    # κ summaries used to live at c_meta[..., 0..2] but were dropped from
    # the feature surface (PRD §4.4) because they are biased by the
    # quality filter on ``epoch[0]``.


def apply_segment_dropout(
    batch: Dict[str, torch.Tensor],
    *,
    p: float = 0.1,
    rng: Optional[torch.Generator] = None,
    rel_num_buckets: int = 32,
    rel_d_max: float = 40.0,
) -> Dict[str, torch.Tensor]:
    """With probability ``p``, drop non-terminal valid segments from a GUID.

    The dropped position is masked out via ``segment_mask`` (the per-segment
    tensors themselves are left in place — the transformer ignores them via
    the mask). After the drop, :func:`_recompute_delta_features` is called to
    rebuild ``delta_t_hours`` / ``cum_monitor_hours`` / ``gap_ratio`` /
    ``rel_bucket_idx`` from the raw ``epoch`` tensor, so the transformer's
    relative-time bias sees the *widened* gap between surviving neighbours
    (matching the "model sees a wider gap" semantics of PRD §8.3).
    ``c_meta`` (TLO/SSO only) is segment-intrinsic and unaffected by
    dropout.

    Args:
        batch: Collated batch dict.
        p: Per-segment drop probability (default 0.1).
        rng: Optional RNG.
        rel_num_buckets: Number of relative-time bias buckets — must match
            the transformer's bias-table size.
        rel_d_max: Relative-time bias saturation horizon (in 20-min slots).

    Returns:
        Possibly-modified copy of ``batch`` with consistent Δt features.
    """
    if p <= 0.0:
        return batch
    seg_mask: torch.Tensor = batch["segment_mask"]
    num_seg: torch.Tensor = batch["num_segments"]
    B, N = seg_mask.shape
    n_per = num_seg.tolist()
    rand = torch.rand(B, N, generator=rng)
    new_mask = seg_mask.clone()
    for b in range(B):
        n = int(n_per[b])
        if n <= 1:
            continue
        # Non-terminal valid segments only.
        for j in range(n - 1):
            if seg_mask[b, j] and rand[b, j].item() < p:
                new_mask[b, j] = False
    if torch.equal(new_mask, seg_mask):
        return batch
    new_batch: Dict[str, Any] = {
        k: (v.clone() if isinstance(v, torch.Tensor) else list(v) if isinstance(v, list) else v)
        for k, v in batch.items()
    }
    new_batch["segment_mask"] = new_mask
    new_batch["num_segments"] = new_mask.sum(dim=-1).long()
    # Rebuild Δt-derived features from the raw ``epoch`` tensor so neighbours
    # of dropped segments see the correct widened gap.
    _recompute_delta_features(
        new_batch,
        rel_num_buckets=int(rel_num_buckets),
        rel_d_max=float(rel_d_max),
    )
    return new_batch


def _macro_f1(probs_3: torch.Tensor, target_3: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """Macro-averaged F1 from class probabilities."""
    preds = probs_3.argmax(dim=-1)
    f1_per_class: List[torch.Tensor] = []
    for c in range(num_classes):
        tp = ((preds == c) & (target_3 == c)).sum().float()
        fp = ((preds == c) & (target_3 != c)).sum().float()
        fn = ((preds != c) & (target_3 == c)).sum().float()
        denom = (2 * tp + fp + fn).clamp_min(1.0)
        f1_per_class.append(2 * tp / denom)
    return torch.stack(f1_per_class).mean()


def _binary_auroc(probs_pos: torch.Tensor, target_bin: torch.Tensor) -> torch.Tensor:
    """Mann-Whitney U binary AUROC computed on rank scores.

    Avoids importing scikit-learn at training time. Returns 0.5 when only
    one class is present.
    """
    target = target_bin.long()
    pos = probs_pos[target == 1]
    neg = probs_pos[target == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return probs_pos.new_tensor(0.5)
    all_scores = torch.cat([pos, neg])
    ranks = all_scores.argsort().argsort().float() + 1.0
    pos_ranks = ranks[: pos.numel()]
    n_pos = float(pos.numel())
    n_neg = float(neg.numel())
    auc = (pos_ranks.sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return auc.clamp(0.0, 1.0)


def _last_valid_idx(segment_mask: torch.Tensor) -> torch.Tensor:
    """Return the index of the last True position per row.

    Args:
        segment_mask: ``(B, N)`` bool — True for valid segments.

    Returns:
        ``(B,)`` long tensor — index in ``[0, N-1]`` of the last True per row.
        Rows with no valid positions get index 0 (callers should guard with
        the row's ``num_segments`` count).
    """
    mask_f = segment_mask.float()
    pos = mask_f.cumsum(dim=-1) * mask_f
    return pos.argmax(dim=-1).clamp(min=0)


def _gather_last_valid_per_position(
    per_pos: torch.Tensor, segment_mask: torch.Tensor
) -> torch.Tensor:
    """Gather the per-position output at each row's last valid position.

    Args:
        per_pos: ``(B, N)`` or ``(B, N, C)`` per-position tensor.
        segment_mask: ``(B, N)`` bool.

    Returns:
        ``(B,)`` if input was ``(B, N)``, else ``(B, C)``.
    """
    last_idx = _last_valid_idx(segment_mask)              # (B,)
    batch_idx = torch.arange(per_pos.shape[0], device=per_pos.device)
    return per_pos[batch_idx, last_idx]


class PlGuidClassifier(LightningModelBase):
    """Lightning wrapper around :class:`GuidOutcomeClassifier` (causal AR).

    Args:
        base_model: A :class:`GuidOutcomeClassifier` instance.
        loss_weights: Loss-component weights.
        class_weights_3: Optional length-3 class weights.
        class_weights_bin: Optional length-2 binary weights.
        segment_dropout_p: Per-segment dropout probability for training.
        segment_dropout_enabled: Toggle for segment dropout.
        rel_num_buckets: Number of relative-time bias buckets — must match
            the transformer's bias-table size.
        rel_d_max: Relative-time bias saturation horizon (in 20-min slots).
        lr: Classifier-group learning rate.
        lr_milestones: Optional MultiStepLR milestones (epoch units).
        lr_warmup_steps: Number of optimizer **steps** of linear warmup
            from 0 to ``lr``. ``0`` disables warmup. Standard transformer
            recipe is 500–2000 steps; needed when AdamW(0.9, 0.95) sees
            bursty class-imbalanced gradients early in training.
        weight_decay: AdamW weight decay.
        vae_lr: Optional separate LR for VAE parameters (live-VAE only).
        compute_macro_f1: Whether to log macro-F1 at val/test epoch end.
        compute_binary_auroc: Whether to log binary AUROC at val/test epoch end.
    """

    prog_bar_metrics: Tuple[str, ...] = ("total_loss", "ce_3", "bce_bin")

    def __init__(
        self,
        base_model: GuidOutcomeClassifier,
        *,
        loss_weights: Optional[LossWeights] = None,
        class_weights_3: Optional[torch.Tensor] = None,
        class_weights_bin: Optional[torch.Tensor] = None,
        segment_dropout_p: float = 0.1,
        segment_dropout_enabled: bool = True,
        rel_num_buckets: int = 32,
        rel_d_max: float = 40.0,
        lr: float = 1e-3,
        lr_milestones: Optional[Iterable[int]] = (100,),
        lr_warmup_steps: int = 0,
        weight_decay: float = 1e-4,
        vae_lr: float = 1e-5,
        compute_macro_f1: bool = True,
        compute_binary_auroc: bool = True,
    ) -> None:
        super().__init__(
            base_model=base_model,
            lr=lr,
            lr_milestones=list(lr_milestones) if lr_milestones else None,
            weight_decay=weight_decay,
            module_name="PlGuidClassifierV2",
        )
        # Stored as attribute (not in hparams) so we can read it in
        # ``build_lr_scheduler`` without adding more arguments to the base.
        self.lr_warmup_steps = int(lr_warmup_steps)
        # Bypass torch.compile for variable-shape batches.
        if getattr(base_model, "no_compile", False):
            self.model = self._orig_model
            logger.info(
                "PlGuidClassifierV2: torch.compile bypassed "
                "(GuidOutcomeClassifier.no_compile=True)"
            )

        self.loss_weights = loss_weights or LossWeights()
        self.loss = GuidClassifierLoss(
            weights=self.loss_weights,
            class_weights_3=class_weights_3,
            class_weights_bin=class_weights_bin,
        )
        self.segment_dropout_p = float(segment_dropout_p)
        self.segment_dropout_enabled = bool(segment_dropout_enabled)
        self.rel_num_buckets = int(rel_num_buckets)
        self.rel_d_max = float(rel_d_max)
        self.vae_lr = float(vae_lr)
        self.compute_macro_f1 = bool(compute_macro_f1)
        self.compute_binary_auroc = bool(compute_binary_auroc)

        # Per-epoch validation buffers (cleared at ``on_validation_epoch_start``).
        # Test-time metrics are produced by
        # :mod:`evaluate_guid_classifier` via the per-position inference
        # pass — ``trainer.test`` is never called from ``train_fold`` —
        # so we do not maintain a separate test-side buffer here.
        self._val_probs_3: List[torch.Tensor] = []
        self._val_target_3: List[torch.Tensor] = []
        self._val_probs_bin: List[torch.Tensor] = []
        self._val_target_bin: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Loss / metrics dispatch
    # ------------------------------------------------------------------

    def compute_loss_and_metrics(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, stage: str
    ) -> Tuple[torch.Tensor, MetricDict]:
        """Run forward + loss + metrics for one batch.

        Args:
            batch: Collated batch dict.
            batch_idx: Batch index within the epoch.
            stage: ``train`` / ``val`` / ``test``.

        Returns:
            ``(total_loss, metrics_dict)`` consumed by
            :class:`LightningModelBase`.
        """
        del batch_idx  # unused
        if stage == "train" and self.segment_dropout_enabled:
            batch = apply_segment_dropout(
                batch,
                p=self.segment_dropout_p,
                rel_num_buckets=self.rel_num_buckets,
                rel_d_max=self.rel_d_max,
            )
        outputs = self.model(batch)

        # Live-VAE auxiliary terms. Computed only when the live forward
        # populated ``vae_outputs`` and at least one of the multipliers
        # is non-zero (i.e. stage 2 is active or the user explicitly
        # turned them on for ablation). Stage 1 keeps ``gamma_vae`` and
        # ``lambda_sp`` at zero — :class:`TwoStageVaeUnfreeze` toggles
        # them at the boundary.
        vae_loss_scalar = None
        sparsity_scalar = None
        if "vae_outputs" in outputs and (
            self.loss_weights.gamma_vae > 0.0
            or self.loss_weights.lambda_sp > 0.0
        ):
            vae_loss_scalar, sparsity_scalar = self._compute_live_aux_terms(
                outputs["vae_outputs"]
            )
        components = self.loss(
            outputs=outputs,
            batch=batch,
            vae_loss=vae_loss_scalar,
            sparsity_term=sparsity_scalar,
        )
        total = components["total_loss"]

        seg_mask = batch["segment_mask"]
        target_3 = batch["label_3"].long()
        target_bin = batch["label_bin"].float()

        with torch.no_grad():
            # Single-GUID summary uses the **last-valid** position so that
            # ``acc_3`` / ``acc_bin`` and the val buffers semantically reflect
            # "the model's prediction at the end of the recording" — the
            # natural final clinical decision.
            last_prob_3 = _gather_last_valid_per_position(outputs["prob_3"], seg_mask)
            last_prob_bin = _gather_last_valid_per_position(outputs["prob_bin"], seg_mask)
            preds_3 = last_prob_3.argmax(dim=-1)
            acc_3 = (preds_3 == target_3).float().mean()
            preds_bin = (last_prob_bin >= 0.5).float()
            acc_bin = (preds_bin == target_bin).float().mean()
            mean_num_segments = batch["num_segments"].float().mean()

        metrics: MetricDict = {
            "total_loss": total,
            "ce_3": components["ce_3"],
            "bce_bin": components["bce_bin"],
            "acc_3": acc_3,
            "acc_bin": acc_bin,
            "mean_num_segments": mean_num_segments,
        }
        if "vae_loss" in components:
            metrics["vae_loss"] = components["vae_loss"]
        if "sparsity" in components:
            metrics["sparsity"] = components["sparsity"]

        # Buffer last-valid probabilities for end-of-epoch macro-F1 / AUROC.
        if stage == "val":
            self._buffer_val_outputs(last_prob_3, last_prob_bin, target_3, target_bin)
        return total, metrics

    def _compute_live_aux_terms(
        self,
        vae_outputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Build the per-batch VAE-aux KL + L2-anchor terms (live path).

        The KL is reduced as a per-step weighted mean over the
        classifier-time mask ``hat_w_v`` (same definition the cached
        path uses for the per-segment summary). The L2 anchor is
        ``Σ ‖θ − θ⁽⁰⁾‖₂²`` over the unfrozen VAE submodules — only
        evaluated once :class:`TwoStageVaeUnfreeze` has snapshotted
        ``θ⁽⁰⁾`` at stage 2 start.

        Args:
            vae_outputs: The ``vae_outputs`` sub-dict produced by
                :meth:`GuidOutcomeClassifier.live_forward`.

        Returns:
            ``(vae_loss, sparsity_or_None)``. ``vae_loss`` is always a
            scalar tensor; ``sparsity`` is ``None`` when no anchor was
            recorded yet.
        """
        kld_per_t = vae_outputs["kld_per_t"]                  # (M, T)
        hat_w_v = vae_outputs["hat_w_v"]                      # (M, T)
        weight_sum = hat_w_v.sum().clamp_min(1.0)
        vae_loss = (kld_per_t * hat_w_v).sum() / weight_sum

        sparsity: Optional[torch.Tensor] = None
        theta0 = getattr(self, "_vae_theta0", None)
        names = getattr(self, "_vae_unfreeze_names", None)
        if (
            theta0 is not None
            and names
            and self.loss_weights.lambda_sp > 0.0
        ):
            base = getattr(self, "_orig_model", self)
            vae = getattr(base, "vae", None)
            if vae is not None:
                accum = kld_per_t.new_zeros(())
                for name in names:
                    p0 = theta0.get(name)
                    if p0 is None:
                        continue
                    p = self._lookup_param(vae, name)
                    if p is None or not p.requires_grad:
                        continue
                    diff = p - p0.to(device=p.device, dtype=p.dtype)
                    accum = accum + diff.pow(2).sum()
                sparsity = accum
        return vae_loss, sparsity

    @staticmethod
    def _lookup_param(
        vae: torch.nn.Module, qualified_name: str
    ) -> Optional[torch.nn.Parameter]:
        """Resolve ``submodule.param.path`` against the VAE module tree."""
        try:
            obj: Any = vae
            for part in qualified_name.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, torch.nn.Parameter):
                return obj
        except AttributeError:
            return None
        return None

    def _buffer_val_outputs(
        self,
        last_prob_3: torch.Tensor,
        last_prob_bin: torch.Tensor,
        target_3: torch.Tensor,
        target_bin: torch.Tensor,
    ) -> None:
        """Accumulate per-batch last-valid probabilities for epoch metrics."""
        self._val_probs_3.append(last_prob_3.detach().cpu())
        self._val_target_3.append(target_3.detach().cpu())
        self._val_probs_bin.append(last_prob_bin.detach().cpu())
        self._val_target_bin.append(target_bin.detach().cpu())

    def on_validation_epoch_start(self) -> None:
        self._val_probs_3.clear()
        self._val_target_3.clear()
        self._val_probs_bin.clear()
        self._val_target_bin.clear()

    def on_validation_epoch_end(self) -> None:
        if not self._val_probs_3:
            return
        probs_3 = torch.cat(self._val_probs_3)
        target_3 = torch.cat(self._val_target_3)
        probs_bin = torch.cat(self._val_probs_bin)
        target_bin = torch.cat(self._val_target_bin)
        if self.compute_macro_f1:
            self.log("val/macro_f1", _macro_f1(probs_3, target_3), sync_dist=True)
        if self.compute_binary_auroc:
            self.log("val/binary_auroc", _binary_auroc(probs_bin, target_bin), sync_dist=True)

    # ------------------------------------------------------------------
    # Optimizer (supports two parameter groups when VAE is unfrozen)
    # ------------------------------------------------------------------

    def build_optimizer(self, trainable_params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
        """AdamW with optional separate VAE parameter group.

        Splits the trainable parameters by name prefix ``vae.``: anything
        whose ``id`` matches a parameter in ``self._orig_model.vae`` (when
        the live-VAE path is enabled) lands in a low-LR group; everything
        else uses ``self.hparams.lr``.
        """
        trainable_params = list(trainable_params)
        if not trainable_params:
            return torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-3)

        vae_module = getattr(self._orig_model, "vae", None)
        if vae_module is not None:
            vae_param_ids = {id(p) for p in vae_module.parameters() if p.requires_grad}
            cls_params = [p for p in trainable_params if id(p) not in vae_param_ids]
            vae_params = [p for p in trainable_params if id(p) in vae_param_ids]
            param_groups = [
                {
                    "params": cls_params,
                    "lr": float(self.hparams.lr),
                    "weight_decay": float(self.hparams.weight_decay),
                }
            ]
            if vae_params:
                param_groups.append(
                    {
                        "params": vae_params,
                        "lr": float(self.vae_lr),
                        "weight_decay": 0.0,
                    }
                )
            logger.info(
                f"PlGuidClassifier optimizer: classifier params={len(cls_params)}, "
                f"vae params={len(vae_params)}"
            )
            # AdamW betas (0.9, 0.95). Reverted from (0.9, 0.999) after
            # the latter caused early-training divergence: with extreme
            # class-imbalance, rare-class GUIDs produce bursty gradients
            # across batches, and β₂=0.999 averages over too long a window
            # to capture each burst — AdamW then behaves like signed-SGD
            # during the burst and over-shoots. β₂=0.95 adapts the second
            # moment fast enough to scale each burst correctly.
            return torch.optim.AdamW(param_groups, eps=1e-8, betas=(0.9, 0.95))
        return torch.optim.AdamW(
            trainable_params,
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
            eps=1e-8,
            betas=(0.9, 0.95),
        )

    def build_lr_scheduler(self, optimizer):
        """Linear warmup followed by ``MultiStepLR`` decay (step-level).

        When ``lr_warmup_steps == 0`` this falls back to the base class's
        epoch-level ``MultiStepLR``. Otherwise:

        * Linear ramp from ``lr / warmup_steps`` to ``lr`` over the first
          ``warmup_steps`` optimizer steps (≈ first epoch for this
          dataset's batch count).
        * Hand off to ``MultiStepLR`` at the configured epoch milestones,
          converted to step units via
          ``trainer.estimated_stepping_batches / max_epochs``.

        Lightning is told to step the scheduler every optimizer step
        (``interval='step'``) because the warmup and the milestones must
        live on the same time axis under :class:`SequentialLR`.
        """
        warmup_steps = int(getattr(self, "lr_warmup_steps", 0))
        if warmup_steps <= 0:
            return super().build_lr_scheduler(optimizer)

        from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR

        epoch_milestones = list(getattr(self.hparams, "lr_milestones", None) or [])
        gamma = float(getattr(self.hparams, "lr_gamma", 0.1))

        try:
            total_steps = int(self.trainer.estimated_stepping_batches)
            max_epochs = max(1, int(self.trainer.max_epochs or 1))
            steps_per_epoch = max(1, total_steps // max_epochs)
        except Exception:
            steps_per_epoch = 100

        warmup = LinearLR(
            optimizer,
            start_factor=1.0 / float(warmup_steps),
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        # Convert epoch milestones to absolute step counts. Add the warmup
        # offset because SequentialLR re-bases each child scheduler at the
        # boundary, so the second scheduler's "step 0" corresponds to the
        # global step ``warmup_steps``.
        step_milestones = [int(m) * int(steps_per_epoch) for m in epoch_milestones]
        if step_milestones:
            decay = MultiStepLR(optimizer, milestones=step_milestones, gamma=gamma)
        else:
            # No epoch milestones configured: keep LR flat after warmup.
            decay = MultiStepLR(optimizer, milestones=[10**9], gamma=1.0)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, decay],
            milestones=[int(warmup_steps)],
        )
        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }


__all__ = [
    "PlGuidClassifier",
    "apply_segment_dropout",
]
