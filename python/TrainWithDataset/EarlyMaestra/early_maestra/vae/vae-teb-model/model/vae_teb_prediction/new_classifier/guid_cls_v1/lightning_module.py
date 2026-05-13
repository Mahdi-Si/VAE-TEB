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


def _per_class_prf1(
    probs_3: torch.Tensor, target_3: torch.Tensor, num_classes: int = 3
) -> Dict[str, torch.Tensor]:
    """Per-class precision / recall / F1 + macro-F1 + confusion matrix.

    Args:
        probs_3: ``(B, C)`` class probabilities.
        target_3: ``(B,)`` integer class targets.
        num_classes: Number of classes; default 3.

    Returns:
        Dict with keys:

        * ``precision`` / ``recall`` / ``f1`` — ``(C,)`` float tensors,
          one entry per class. Zero when a class is absent from the
          buffer (denominator clamped to 1).
        * ``macro_f1`` — mean of ``f1``.
        * ``confusion`` — ``(C, C)`` long tensor, ``[true, pred]``.
        * ``support`` — ``(C,)`` long tensor of target counts per class.
    """
    preds = probs_3.argmax(dim=-1)
    precision = probs_3.new_zeros(num_classes)
    recall = probs_3.new_zeros(num_classes)
    f1 = probs_3.new_zeros(num_classes)
    support = torch.zeros(num_classes, dtype=torch.long)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for true_c in range(num_classes):
        true_mask = target_3 == true_c
        support[true_c] = int(true_mask.sum().item())
        for pred_c in range(num_classes):
            confusion[true_c, pred_c] = int(
                ((preds == pred_c) & true_mask).sum().item()
            )
    for c in range(num_classes):
        tp = ((preds == c) & (target_3 == c)).sum().float()
        fp = ((preds == c) & (target_3 != c)).sum().float()
        fn = ((preds != c) & (target_3 == c)).sum().float()
        precision[c] = tp / (tp + fp).clamp_min(1.0)
        recall[c] = tp / (tp + fn).clamp_min(1.0)
        denom = (2 * tp + fp + fn).clamp_min(1.0)
        f1[c] = 2 * tp / denom
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": f1.mean(),
        "confusion": confusion,
        "support": support,
    }


def _macro_f1(probs_3: torch.Tensor, target_3: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """Macro-averaged F1 from class probabilities (back-compat shim)."""
    return _per_class_prf1(probs_3, target_3, num_classes=num_classes)["macro_f1"]


def _binary_brier(probs_pos: torch.Tensor, target_bin: torch.Tensor) -> torch.Tensor:
    """Brier score $\\frac{1}{N}\\sum (p_i - y_i)^2$ for the binary head."""
    target = target_bin.float()
    return (probs_pos - target).pow(2).mean()


def _expected_calibration_error(
    probs_pos: torch.Tensor,
    target_bin: torch.Tensor,
    n_bins: int = 10,
) -> torch.Tensor:
    """Equal-width Expected Calibration Error (ECE) for binary probabilities.

    Bins probabilities into ``n_bins`` equal-width buckets in $[0, 1]$,
    computes the mean confidence and mean accuracy within each bucket,
    and returns the sample-weighted mean absolute gap. Returns 0 when
    only one bin is occupied (degenerate case where ECE is undefined).
    """
    target = target_bin.float()
    n = float(target.numel())
    if n <= 0:
        return probs_pos.new_tensor(0.0)
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs_pos.device)
    ece = probs_pos.new_tensor(0.0)
    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (probs_pos >= lo) & (probs_pos <= hi)
        else:
            in_bin = (probs_pos >= lo) & (probs_pos < hi)
        count = in_bin.float().sum()
        if count.item() <= 0:
            continue
        avg_conf = probs_pos[in_bin].mean()
        avg_acc = target[in_bin].mean()
        ece = ece + (count / n) * (avg_conf - avg_acc).abs()
    return ece


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
        lr_gamma: float = 0.1,
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
        # Stored as attributes (not in hparams) so we can read them in
        # ``build_lr_scheduler`` without adding more arguments to the base.
        # ``lr_gamma`` was previously hard-coded to 0.1 by a defaulted
        # ``getattr(self.hparams, "lr_gamma", 0.1)`` that the YAML never
        # populated; routing it as an explicit attribute here means the
        # ``training.scheduler.gamma`` YAML key is now actually honoured.
        self.lr_warmup_steps = int(lr_warmup_steps)
        self.lr_gamma = float(lr_gamma)
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

        # Populated at the end of every validation epoch with a dict
        # consumed by :class:`TrainingDiagnosticsCallback` (confusion
        # matrix, per-class support counts, calibration scalars). Set to
        # ``None`` when no validation has run yet.
        self._last_val_summary: Optional[Dict[str, Any]] = None

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

        # Head flags come from the loss weights (single source of truth that
        # the trainer also propagates into the model).
        enable_three_class = bool(self.loss_weights.enable_three_class)
        enable_binary = bool(self.loss_weights.enable_binary)

        last_prob_3: Optional[torch.Tensor] = None
        last_prob_bin: Optional[torch.Tensor] = None
        acc_3: Optional[torch.Tensor] = None
        acc_bin: Optional[torch.Tensor] = None

        with torch.no_grad():
            # Single-GUID summary uses the **last-valid** position so that
            # ``acc_3`` / ``acc_bin`` and the val buffers semantically reflect
            # "the model's prediction at the end of the recording" — the
            # natural final clinical decision.
            if enable_three_class and "prob_3" in outputs:
                last_prob_3 = _gather_last_valid_per_position(outputs["prob_3"], seg_mask)
                preds_3 = last_prob_3.argmax(dim=-1)
                acc_3 = (preds_3 == target_3).float().mean()
            if enable_binary and "prob_bin" in outputs:
                last_prob_bin = _gather_last_valid_per_position(outputs["prob_bin"], seg_mask)
                preds_bin = (last_prob_bin >= 0.5).float()
                acc_bin = (preds_bin == target_bin).float().mean()
            mean_num_segments = batch["num_segments"].float().mean()

        metrics: MetricDict = {
            "total_loss": total,
            "mean_num_segments": mean_num_segments,
        }
        if "ce_3" in components:
            metrics["ce_3"] = components["ce_3"]
        if "bce_bin" in components:
            metrics["bce_bin"] = components["bce_bin"]
        if acc_3 is not None:
            metrics["acc_3"] = acc_3
        if acc_bin is not None:
            metrics["acc_bin"] = acc_bin
        if "vae_loss" in components:
            metrics["vae_loss"] = components["vae_loss"]
        if "sparsity" in components:
            metrics["sparsity"] = components["sparsity"]

        # Effective-position counters (train stage only — at val these
        # are unmasked-position-count-equivalents and not informative).
        # ``loss_warmup_positions`` excludes the first K valid positions
        # of every row from the loss reduction; the count here reflects
        # the *kept* set, so the CSV shows whether each epoch saw the
        # rare class in actual loss-driving positions.
        if stage == "train":
            with torch.no_grad():
                pos_total, pos_per_class = self._count_loss_positions(
                    seg_mask, target_3, enable_three_class
                )
                metrics["loss_positions_total"] = pos_total
                if pos_per_class is not None:
                    for c, count in enumerate(pos_per_class):
                        metrics[f"loss_positions_class{c}"] = count

        # Buffer last-valid probabilities for end-of-epoch macro-F1 / AUROC,
        # honouring the per-head gate.
        if stage == "val":
            self._buffer_val_outputs(last_prob_3, last_prob_bin, target_3, target_bin)
        return total, metrics

    def _count_loss_positions(
        self,
        seg_mask: torch.Tensor,
        target_3: torch.Tensor,
        enable_three_class: bool,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Count per-batch positions that contribute to the loss reduction.

        Mirrors the ``loss_warmup_positions`` skip applied inside
        :meth:`GuidClassifierLoss._two_level_mean` so the reported count
        equals the count actually used by the gradient. The per-class
        breakdown uses the GUID-level 3-class label broadcast to every
        kept position (the same broadcast the loss uses).

        Args:
            seg_mask: ``(B, N)`` segment validity mask.
            target_3: ``(B,)`` long GUID-level class id.
            enable_three_class: When False, the per-class breakdown is
                skipped (returned ``None``).

        Returns:
            ``(positions_total, positions_per_class)``. ``positions_total``
            is always a scalar tensor; ``positions_per_class`` is a
            length-3 list of scalar tensors when ``enable_three_class``
            is True, else ``None``.
        """
        k_warm = int(self.loss_weights.loss_warmup_positions)
        if k_warm > 0:
            rank_one_based = seg_mask.long().cumsum(dim=-1)
            keep_mask = (rank_one_based > k_warm) & seg_mask
        else:
            keep_mask = seg_mask
        keep_float = keep_mask.float()
        positions_total = keep_float.sum()
        per_class: Optional[List[torch.Tensor]] = None
        if enable_three_class:
            per_class = []
            for c in range(3):
                row_match = (target_3 == c).float().unsqueeze(1)    # (B, 1)
                per_class.append((keep_float * row_match).sum())
        return positions_total, per_class

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
        last_prob_3: Optional[torch.Tensor],
        last_prob_bin: Optional[torch.Tensor],
        target_3: torch.Tensor,
        target_bin: torch.Tensor,
    ) -> None:
        """Accumulate per-batch last-valid probabilities for epoch metrics.

        Args:
            last_prob_3: 3-class probabilities at the last-valid position,
                or ``None`` when the 3-class head is disabled.
            last_prob_bin: Binary probabilities at the last-valid position,
                or ``None`` when the binary head is disabled.
            target_3: GUID-level 3-class targets.
            target_bin: GUID-level binary targets.
        """
        if last_prob_3 is not None:
            self._val_probs_3.append(last_prob_3.detach().cpu())
            self._val_target_3.append(target_3.detach().cpu())
        if last_prob_bin is not None:
            self._val_probs_bin.append(last_prob_bin.detach().cpu())
            self._val_target_bin.append(target_bin.detach().cpu())

    def on_validation_epoch_start(self) -> None:
        self._val_probs_3.clear()
        self._val_target_3.clear()
        self._val_probs_bin.clear()
        self._val_target_bin.clear()

    def on_validation_epoch_end(self) -> None:
        # Skip per-head metrics when the buffer for that head is empty
        # (either the head was disabled this run or no batches were
        # processed). Each head is independent — if one is disabled and
        # the other enabled, we still compute the enabled head's metric.
        summary: Dict[str, Any] = {
            "epoch": int(getattr(self.trainer, "current_epoch", -1)),
            "global_step": int(getattr(self.trainer, "global_step", 0)),
        }
        if self.compute_macro_f1 and self._val_probs_3:
            probs_3 = torch.cat(self._val_probs_3)
            target_3 = torch.cat(self._val_target_3)
            prf = _per_class_prf1(probs_3, target_3)
            self.log("val/macro_f1", prf["macro_f1"], sync_dist=True)
            # Per-class P/R/F1 as separate scalars so each lands in its
            # own CSV column. Indexed by class id (0=healthy, 1=acidosis,
            # 2=HIE) — see PRD §4.1.
            for c in range(prf["f1"].numel()):
                self.log(f"val/precision_class{c}", prf["precision"][c], sync_dist=True)
                self.log(f"val/recall_class{c}", prf["recall"][c], sync_dist=True)
                self.log(f"val/f1_class{c}", prf["f1"][c], sync_dist=True)
            summary["n_val_guids_3"] = int(target_3.numel())
            summary["confusion_3class"] = prf["confusion"].tolist()
            summary["support_3class"] = prf["support"].tolist()
            summary["precision_per_class"] = prf["precision"].detach().cpu().tolist()
            summary["recall_per_class"] = prf["recall"].detach().cpu().tolist()
            summary["f1_per_class"] = prf["f1"].detach().cpu().tolist()
            summary["macro_f1"] = float(prf["macro_f1"].item())
        if self.compute_binary_auroc and self._val_probs_bin:
            probs_bin = torch.cat(self._val_probs_bin)
            target_bin = torch.cat(self._val_target_bin)
            self.log("val/binary_auroc", _binary_auroc(probs_bin, target_bin), sync_dist=True)
            brier = _binary_brier(probs_bin, target_bin)
            ece = _expected_calibration_error(probs_bin, target_bin)
            self.log("val/brier", brier, sync_dist=True)
            self.log("val/ece", ece, sync_dist=True)
            summary["n_val_guids_bin"] = int(target_bin.numel())
            summary["bin_positive_count"] = int(target_bin.sum().item())
            summary["bin_negative_count"] = int((target_bin.numel() - target_bin.sum().item()))
            summary["brier"] = float(brier.item())
            summary["ece"] = float(ece.item())
        # Expose the structured summary so
        # :class:`TrainingDiagnosticsCallback` can persist it to
        # ``epoch_summary.jsonl`` without re-running the buffer math.
        self._last_val_summary = summary

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
        # ``lr_gamma`` is set as an attribute in __init__ rather than via
        # ``self.hparams`` because the base ``LightningModelBase`` does not
        # advertise this knob; the YAML key is now plumbed through the
        # explicit ``lr_gamma`` constructor arg.
        gamma = float(getattr(self, "lr_gamma", 0.1))

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
        # Convert epoch milestones to absolute step counts, then SUBTRACT
        # the warmup offset because :class:`SequentialLR` re-bases each
        # child scheduler at the transition: when the warmup phase ends,
        # the decay scheduler's ``last_epoch`` is reset to 0, so its
        # milestones are interpreted relative to that boundary, not to
        # the global step counter. Without the subtraction, milestone
        # ``M`` fires at global step ``warmup_steps + M`` — i.e. one
        # warmup window late — which silently delayed every LR drop.
        # ``max(1, ...)`` guards against milestones that fall inside the
        # warmup window (those should fire as soon as warmup ends).
        step_milestones = [
            max(1, int(m) * int(steps_per_epoch) - int(warmup_steps))
            for m in epoch_milestones
        ]
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
