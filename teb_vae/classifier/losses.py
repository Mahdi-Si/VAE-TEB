"""Classifier losses and their composition (SPEC §10.2-§10.3, Appendix B).

* :func:`loss_fn` validates ``train.loss`` against ``labels.head`` and returns the per-element loss
  ``ℓ(logits (..., K_out), y (...)) -> (...)``: ``bce | weighted_bce | focal | logit_adjusted`` (binary),
  ``ce | weighted_ce | focal_ce`` (multiclass), ``coral | cumulative_link`` (ordinal: Σ_k BCE on the K-1
  cumulative logits, or the proportional-odds NLL ``-log P(Y = y)`` of :func:`coral_log_probs`).
* :func:`set_loss`: the batch-level ``auc_margin | pauc`` (binary; ``(s, y, w) -> scalar``), no per-element form.
  :func:`criterion` is the one entry for both kinds (``(s, y, w) -> scalar``); ``train.py`` calls it once at setup so
  a bad config fails before the first step.
* :func:`class_weights` (GUID-level train counts; normalised to Σ_k w_k = K, Cui 2019) and
  :func:`prior_offset` (Saerens: subtract from raw logits before calibration).
* :func:`compute_loss` composes the terms (§10.3). Masks are applied by multiplication; every
  division is guarded, so empty GUIDs / padded rows / Σω = 0 give 0, never NaN.
  - segment scope: ``loss_positions = Σ w·ℓ / Σ w`` (``w`` already GUID-normalised) + λ₃ · the aux
    head's same term. The positions term has weight 1 here (it is the only main term).
  - sequence scope: λ_final·final + λ_pos·positions + λ_bag·bag + λ_seg·segment + λ₃·(the same four
    terms for the aux heads). ``positions``/``segment`` average over GUIDs with Σω > 0 only. The bag is
    ``τ·log mean_n exp(s_n/τ)`` over the segment-local head (``loss_weights.bag_head: segment``, else the position
    head; ``position`` pools the per-position head, whose running max the committed decision reads), per logit.
    ``strategy: mil`` turns the positions term off and the bag on (weight ``λ_bag``, or 1 if that is 0).
  - The aux 3-class head always uses plain ``ce`` (same label smoothing); ``y3 = -1`` is masked out.
  - ``parts`` holds the unweighted main terms and the λ-weighted aux composite (before λ₃).
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from teb_vae.classifier.config import HEAD_LOSSES, LabelsCfg, LossCfg, TrainCfg

#: Batch-level (pairwise) binary losses: no per-element form, so :func:`criterion` routes them to :func:`set_loss`.
SET_LOSSES = ("auc_margin", "pauc")
PARTS = ("loss_final", "loss_positions", "loss_bag", "loss_segment", "loss_aux3")

Loss = Callable[[Tensor, Tensor], Tensor]
#: ``(scores (..., K_out), targets (...), weights (...)) -> scalar``: one loss term over a weighted population.
Crit = Callable[[Tensor, Tensor, Tensor], Tensor]


def class_weights(counts: Sequence[float], scheme: str = "none", beta: float = 0.999) -> Tensor:
    """Per-class weights from GUID-level train counts, normalised so Σ_k w_k = K."""
    n = torch.as_tensor(counts, dtype=torch.float64)
    if scheme == "none":
        w = torch.ones_like(n)
    elif scheme == "inverse":
        w = 1 / n
    elif scheme == "sqrt_inverse":
        w = n.rsqrt()
    elif scheme == "effective_number":
        w = (1 - beta) / (1 - beta ** n)
    else:
        raise ValueError(f"unknown train.loss.weighting {scheme!r}")
    return (w * len(w) / w.sum()).float()


def prior_offset(weights: Sequence[float], priors: Optional[Sequence[float]] = None,
                 tau: float = 0.0) -> Any:
    """Offset to subtract from raw logits before calibration (Saerens 2002; Appendix B).

    ``log w`` undoes class weighting; ``- τ·log π`` undoes ``logit_adjusted`` (pass ``priors`` and
    ``train.loss.logit_adjust_tau``; weights all 1 then). Binary: the float ``log(w₁/w₀) - τ·log(π₁/π₀)``
    for ``s ← s - offset``; K > 2: the per-class vector for ``logits ← logits - offset``.
    """
    off = torch.as_tensor(weights, dtype=torch.float64).log()
    if priors is not None:
        off = off - tau * torch.as_tensor(priors, dtype=torch.float64).log()
    return float(off[1] - off[0]) if len(off) == 2 else off.float()


def loss_fn(cfg: LossCfg, head: str, class_weights: Optional[Sequence[float]] = None,
            priors: Optional[Sequence[float]] = None) -> Loss:
    """Validate ``train.loss`` for ``labels.head`` and return the per-element loss ℓ(logits, y)."""
    name, eps, gamma = cfg.name, cfg.label_smoothing, cfg.focal_gamma
    if name in SET_LOSSES:
        raise ValueError(f"train.loss.name={name!r} is batch-level (pairwise): it has no per-element form; use "
                         f"losses.criterion")
    if name not in HEAD_LOSSES[head]:
        raise ValueError(f"train.loss.name={name!r} does not fit labels.head={head!r}; "
                         f"expected one of {HEAD_LOSSES[head]}")
    if name == "focal" and cfg.focal_alpha is None:
        raise ValueError("train.loss.name=focal needs an explicit train.loss.focal_alpha (α for the "
                         "positive class; torchvision's 0.25 down-weights positives, SPEC §10.2)")
    if name.startswith("weighted") and class_weights is None:
        raise ValueError(f"train.loss.name={name!r} needs class_weights")
    if name == "logit_adjusted" and priors is None:
        raise ValueError("train.loss.name=logit_adjusted needs the train-fold priors")
    w = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
    shift = 0.0 if priors is None else cfg.logit_adjust_tau * math.log(priors[1] / priors[0])

    def binary(o: Tensor, y: Tensor) -> Tensor:
        s = o[..., 0] + (shift if name == "logit_adjusted" else 0.0)
        t = y.float() * (1 - eps) + eps / 2
        loss = F.binary_cross_entropy_with_logits(s, t, reduction="none")
        if name == "focal":
            p = torch.sigmoid(s)
            p_t = p * t + (1 - p) * (1 - t)
            loss = loss * (1 - p_t) ** gamma * (cfg.focal_alpha * t + (1 - cfg.focal_alpha) * (1 - t))
        if name == "weighted_bce":
            loss = loss * w.to(loss)[y.long()]
        return loss

    def multiclass(o: Tensor, y: Tensor) -> Tensor:
        y = y.long()
        loss = F.cross_entropy(o.flatten(0, -2), y.flatten(), reduction="none", label_smoothing=eps)
        loss = loss.view(y.shape)
        if name == "weighted_ce":
            loss = loss * w.to(loss)[y]
        if name == "focal_ce":
            loss = loss * (1 - o.log_softmax(-1).gather(-1, y[..., None])[..., 0].exp()) ** gamma
        return loss

    def coral(o: Tensor, y: Tensor) -> Tensor:
        k = torch.arange(o.shape[-1], device=o.device)
        t = (y.long()[..., None] > k).float() * (1 - eps) + eps / 2
        return F.binary_cross_entropy_with_logits(o, t, reduction="none").sum(-1)

    def cumulative_link(o: Tensor, y: Tensor) -> Tensor:
        lp = coral_log_probs(o)
        nll = -lp.gather(-1, y.long()[..., None])[..., 0]
        return (1 - eps) * nll - eps * lp.mean(-1)  # label smoothing as in F.cross_entropy

    return {"binary": binary, "multiclass": multiclass,
            "ordinal": cumulative_link if name == "cumulative_link" else coral}[head]


def set_loss(name: str, alpha: Optional[float] = None, margin: float = 1.0) -> Crit:
    """``auc_margin`` | ``pauc`` over one weighted population (§10.2), on ``h = σ(s)`` as in LibAUC; written from the
    papers. With ``w`` = 0 rows dropped, P = the positives and N the negatives (weighted expectations E₊, E₋):

    * ``auc_margin`` (AUC-M, Yuan et al. 2021, eq. 3): ``E₊(h - a)² + E₋(h - b)² + (m - a + b)₊²`` with a = E₊h and
      b = E₋h, i.e. the closed-form inner min over a, b and max over α ≥ 0 of the paper's min-max form, per batch
      (plain AdamW then optimises it; PESG's stochastic ascent on α needs no optimizer of its own).
    * ``pauc`` (one-way pAUC over FPR ≤ ``alpha``, Zhu et al. 2022): the CVaR estimator the paper's SOPA optimises,
      mean over P × (the top k = ⌈alpha·|N|⌉ negatives by score) of the squared hinge ``(m - (h_i - h_j))₊²``; the
      top-k set is the same for every positive because the surrogate decreases in ``h_i - h_j``. Pairs weigh
      ``w_i·w_j``.

    A population without both classes gives exactly 0 (a finite loss with a zero gradient). ``ponytail:`` the
    per-batch closed forms are biased by O(1/batch) against the paper's running a, b, α; learnable ones (and PESG) if
    small class-balanced batches look noisy."""
    if name == "pauc" and alpha is None:
        raise ValueError("train.loss.name=pauc needs alpha (the primary policy's FPR cap)")

    def fn(s: Tensor, y: Tensor, w: Tensor) -> Tensor:
        h, y, w = torch.sigmoid(s[..., 0]).flatten(), y.flatten() > 0.5, w.flatten().to(s.dtype)
        wp, wn = w * y, w * ~y
        both = ((wp.sum() > 0) & (wn.sum() > 0)).to(h.dtype)
        if name == "auc_margin":
            mp, mn = (wp * h).sum() / wp.sum().clamp(min=1e-12), (wn * h).sum() / wn.sum().clamp(min=1e-12)
            var = (wp * (h - mp) ** 2).sum() / wp.sum().clamp(min=1e-12) + (wn * (h - mn) ** 2).sum() / wn.sum().clamp(
                min=1e-12)
            return both * (var + F.relu(margin - (mp - mn)) ** 2)
        k = min(len(h), max(1, math.ceil(alpha * int((wn > 0).sum()))))
        top, idx = h.masked_fill(wn <= 0, -1.0).topk(k)  # σ > -1: a non-negative is never picked over a negative
        pair = wp[:, None] * wn[idx][None, :]  # 0 for a non-negative pick
        hinge = F.relu(margin - (h[:, None] - top[None, :])) ** 2
        return both * (pair * hinge).sum() / pair.sum().clamp(min=1e-12)

    return fn


def criterion(cfg: LossCfg, head: str, class_weights: Optional[Sequence[float]] = None,
              priors: Optional[Sequence[float]] = None, alpha: Optional[float] = None) -> Crit:
    """The loss term ``(s, y, w) -> Σ w·ℓ / Σ w`` of :func:`loss_fn` (0 when Σ w = 0), or :func:`set_loss` for
    ``auc_margin | pauc`` (``alpha``: the primary policy's FPR cap, ``pauc`` only)."""
    if cfg.name in SET_LOSSES:
        if head != "binary":
            raise ValueError(f"train.loss.name={cfg.name!r} is binary only; labels.head={head!r}")
        return set_loss(cfg.name, alpha)
    ell = loss_fn(cfg, head, class_weights, priors)
    return lambda s, y, w: _wmean(ell(s, y), w)


def coral_log_probs(o: Tensor) -> Tensor:
    """Class log-probabilities (..., K) of the K-1 ordered cumulative logits ``o_k = logit P(Y > k)`` (Appendix B):
    ``log(σ(o_{k-1}) - σ(o_k))`` with ``o_{-1} = +inf``, ``o_{K-1} = -inf``, in the stable form
    ``log σ(hi) + log σ(-lo) + log(-expm1(lo - hi))``; finite gradients at the padding."""
    hi = F.pad(o, (1, 0), value=math.inf)
    lo = F.pad(o, (0, 1), value=-math.inf)
    return F.logsigmoid(hi) + F.logsigmoid(-lo) + torch.log(-torch.expm1(lo - hi))


def _wmean(v: Tensor, w: Tensor) -> Tensor:
    """Σ w·v / Σ w; 0 when Σ w = 0."""
    return (v * w).sum() / w.sum().clamp(min=1e-12)


def masked_lse(s: Tensor, mask: Tensor, tau: float) -> Tensor:
    """Bag score τ·log mean_n exp(s_n/τ) over valid positions: (B, N, K), (B, N) -> (B, K); 0 if empty."""
    x = (s / tau).masked_fill(~mask[..., None], torch.finfo(s.dtype).min)
    r = tau * (x.logsumexp(1) - mask.sum(1, keepdim=True).clamp(min=1).log())
    return torch.where(mask.any(1, keepdim=True), r, torch.zeros_like(r))


def _sequence_terms(crit: Crit, pos: Tensor, seg: Optional[Tensor], y: Tensor, g: Tensor, mask: Tensor,
                    w: Tensor, w_pos: Tensor, lam: Dict[str, float], tau: float,
                    bag_head: str = "segment") -> Dict[str, Tensor]:
    """The four sequence-scope terms of one head family. pos/seg (B, N, K), GUID target y (B,), GUID
    weight g (B,) (0 = unlabeled), seg_mask (B, N), raw ω (B, N) for the segment-local term and ``w_pos`` (ω with
    ``k_warm`` applied, §6.5) for the per-position term. Disabled terms are 0. Per-segment terms weigh ω / Σ_n ω, so
    every GUID with Σω > 0 counts once (GUID-equal, §6.5). ``bag_head`` (``loss_weights.bag_head``): the bag pools
    the segment-local head when present (``segment``) or the per-position head (``position``, §10.3)."""
    zero = pos.new_zeros(())
    b, n = mask.shape
    has = g * mask.any(1)

    def per_guid(weight: Tensor) -> Tensor:
        wz = weight * mask * g[:, None]
        return wz / wz.sum(1, keepdim=True).clamp(min=1e-12)

    yy = y[:, None].expand(b, n)
    last = (mask.sum(1) - 1).clamp(min=0)
    bag_src = pos if (seg is None or bag_head == "position") else seg
    return {
        "final": crit(pos[torch.arange(b, device=pos.device), last], y, has) if lam["final"] else zero,
        "positions": crit(pos, yy, per_guid(w_pos)) if lam["positions"] else zero,
        "bag": crit(masked_lse(bag_src, mask, tau), y, has) if lam["bag"] else zero,
        "segment": crit(seg, yy, per_guid(w)) if lam["segment"] and seg is not None else zero,
    }


def compute_loss(out: Dict[str, Tensor], batch: Dict[str, Tensor], *, scope: str, labels_cfg: Any,
                 train_cfg: Any, class_weights: Optional[Sequence[float]] = None,
                 priors: Optional[Sequence[float]] = None, lse_tau: float = 1.0, alpha: Optional[float] = None
                 ) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Total loss and its ``parts`` (always the five :data:`PARTS` keys) for one batch (§10.3).

    ``class_weights``: main-head weights (``weighted_*``); ``priors``: main-head train priors
    (``logit_adjusted``); ``lse_tau``: ``model.lse_tau`` for the bag; ``alpha``: the FPR cap of ``pauc``.
    """
    lab, tr = LabelsCfg.model_validate(labels_cfg), TrainCfg.model_validate(train_cfg)
    crit = criterion(tr.loss, lab.head, class_weights, priors, alpha)
    lam3 = lab.aux_3class_weight
    ce3 = criterion(tr.loss.model_copy(update={"name": "ce"}), "multiclass") if lam3 else None
    y3 = batch["y3"].clamp(min=0)
    g3 = (batch["y3"] >= 0).float()

    if scope == "segment":
        w = batch["w"]
        pos = crit(out["seg"], batch["y"], w)
        aux = ce3(out["aux3"], y3, w * g3) if lam3 else pos.new_zeros(())
        zero = pos.new_zeros(())
        return pos + lam3 * aux, dict(zip(PARTS, (zero, pos, zero, zero, aux)))

    mil = lab.strategy == "mil"
    lw = tr.loss_weights
    lam = {"final": lw.final, "positions": 0.0 if mil else lw.positions,
           "bag": lw.bag or float(mil), "segment": lw.segment}
    mask, w = batch["seg_mask"], batch["w"]
    w_pos = batch.get("w_pos", w)  # k_warm drops positions from the per-position term only (data.GuidDataset)
    main = _sequence_terms(crit, out["pos"], out.get("seg"), batch["y"], torch.ones_like(g3), mask, w, w_pos,
                           lam, lse_tau, lw.bag_head)
    aux = (_sequence_terms(ce3, out["aux3_pos"], out.get("aux3_seg"), y3, g3, mask, w, w_pos, lam, lse_tau,
                           lw.bag_head)
           if lam3 else {key: torch.zeros_like(value) for key, value in main.items()})
    loss_aux3 = sum(lam[key] * aux[key] for key in lam)
    total = sum(lam[key] * main[key] for key in lam) + lam3 * loss_aux3
    parts = dict(zip(PARTS, (main["final"], main["positions"], main["bag"], main["segment"], loss_aux3)))
    return total, parts
