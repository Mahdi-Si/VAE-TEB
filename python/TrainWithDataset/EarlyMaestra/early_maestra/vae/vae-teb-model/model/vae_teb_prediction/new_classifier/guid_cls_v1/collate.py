"""Padded GUID-batch collate for ``guid_cls_v1``.

Pads variable-``N`` GUID samples (one per :meth:`GuidSequenceDataset.__getitem__`)
to the per-batch maximum and bundles all per-segment / per-step tensors into
``(B, N_max, ...)`` shape, plus a boolean ``segment_mask`` and the precomputed
relative-time bucket index used by :class:`RelativeTimeTransformer`.

The model is **not** allowed to consume raw ``epoch`` (per PRD §3.3 leakage
rule). The collate keeps ``epoch`` in the batch dictionary purely so that the
evaluation pipeline can emit per-prefix CSV rows; the model code path must
ignore that key.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import torch


def build_relative_time_bucket_index(
    cum_monitor_hours: torch.Tensor,
    *,
    num_buckets: int = 32,
    d_max: float = 40.0,
    segment_duration_sec: float = 1200.0,
) -> torch.Tensor:
    """Compute pairwise log-bucketised Δt indices.

    The score for time-pair ``(i, j)`` of GUID ``b`` is
    ``log(1 + |cum_h[b, i] - cum_h[b, j]| / step_h)``, mapped onto
    ``[0, num_buckets - 1]`` via ``log(1 + d_max)``.

    Args:
        cum_monitor_hours: ``(B, N)`` cumulative monitoring time, in hours.
        num_buckets: Number of buckets (default 32, PRD §7.1).
        d_max: Maximum Δt in 20-min slots; pairs farther than this saturate
            into the last bucket (default 40 ≈ 13 h).
        segment_duration_sec: Nominal segment stride; converts hours → slots
            (default 1200 s = 20 min).

    Returns:
        ``(B, N, N)`` int64 tensor with values in ``[0, num_buckets - 1]``.
    """
    if cum_monitor_hours.dim() != 2:
        raise ValueError(
            f"cum_monitor_hours must be (B, N); got shape {tuple(cum_monitor_hours.shape)}"
        )
    step_h = segment_duration_sec / 3600.0
    diff = (cum_monitor_hours.unsqueeze(2) - cum_monitor_hours.unsqueeze(1)).abs()
    dt_slots = diff / step_h
    log_pair = torch.log1p(dt_slots)
    log_dmax = math.log1p(d_max)
    bucket_f = (num_buckets - 1) * log_pair / log_dmax
    bucket_idx = bucket_f.floor().clamp_(min=0, max=num_buckets - 1).to(torch.long)
    return bucket_idx


def _pad_seg_tensor(
    samples: Sequence[Dict[str, Any]],
    key: str,
    target_n: int,
    *,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Right-pad a per-segment tensor along the segment axis.

    Args:
        samples: Batch of GUID dicts.
        key: Tensor key inside each sample.
        target_n: ``N_max`` to pad to.
        fill_value: Padding constant.

    Returns:
        ``(B, target_n, ...)`` tensor.
    """
    out: List[torch.Tensor] = []
    for s in samples:
        t = s[key]
        if t.dim() == 0:
            t = t.unsqueeze(0)
        n = t.shape[0]
        if n == target_n:
            out.append(t)
            continue
        pad_shape = (target_n - n,) + tuple(t.shape[1:])
        pad = torch.full(pad_shape, fill_value=fill_value, dtype=t.dtype)
        out.append(torch.cat([t, pad], dim=0))
    return torch.stack(out, dim=0)


def guid_sequence_collate_fn(
    batch: List[Dict[str, Any]],
    *,
    rel_time_num_buckets: int = 32,
    rel_time_d_max: float = 40.0,
) -> Dict[str, Any]:
    """Collate function for :class:`GuidSequenceDataset`.

    Args:
        batch: List of per-GUID sample dicts produced by the dataset.
        rel_time_num_buckets: Number of relative-time bias buckets (PRD §7.1).
        rel_time_d_max: Saturation horizon for the bias buckets (PRD §7.1).

    Returns:
        Dict with the following padded keys (B = batch size, N = max segment
        count, T = within-segment length, channels per VAE / model layout):

        * ``h_y``: ``(B, N, T, d_model_vae)``
        * ``mu_prior_norm`` / ``mu_post_norm``: ``(B, N, T, d_z)``
        * ``kld_per_t``: ``(B, N, T)``
        * ``mean_alpha``: ``(B, N, T, L)``
        * ``weight`` / ``hat_w``: ``(B, N, T)``
        * ``target_per_t``: ``(B, N, T)`` int64, padded with ``-1``
        * ``c_meta``: ``(B, N, 5)`` — TLO/SSO only; spans/quality dims
          excluded by design (see :mod:`guid_dataset` and PRD §4.4).
        * ``cum_monitor_hours`` / ``gap_ratio`` / ``delta_t_hours``: ``(B, N)``
        * ``cs_label`` / ``bg_label``: ``(B, N)`` bool
        * ``time_from_labor_onset`` / ``second_stage_onset`` / ``epoch``:
          ``(B, N)``
        * ``segment_mask``: ``(B, N)`` bool — True for valid segments.
        * ``rel_bucket_idx``: ``(B, N, N)`` int64 — log-bucketised Δt.
        * ``label_3``: ``(B,)`` int64
        * ``label_bin``: ``(B,)`` float32
        * ``num_segments``: ``(B,)`` int64
        * ``guid``: ``List[str]``
    """
    if not batch:
        raise ValueError("guid_sequence_collate_fn received an empty batch")

    B = len(batch)
    N = max(int(s["num_segments"]) for s in batch)

    out: Dict[str, Any] = {}
    out["h_y"] = _pad_seg_tensor(batch, "h_y", N)
    out["mu_prior_norm"] = _pad_seg_tensor(batch, "mu_prior_norm", N)
    out["mu_post_norm"] = _pad_seg_tensor(batch, "mu_post_norm", N)
    out["kld_per_t"] = _pad_seg_tensor(batch, "kld_per_t", N)
    out["mean_alpha"] = _pad_seg_tensor(batch, "mean_alpha", N)
    out["weight"] = _pad_seg_tensor(batch, "weight", N)
    out["hat_w"] = _pad_seg_tensor(batch, "hat_w", N)
    out["target_per_t"] = _pad_seg_tensor(batch, "target_per_t", N, fill_value=-1)
    out["c_meta"] = _pad_seg_tensor(batch, "c_meta", N)
    out["cum_monitor_hours"] = _pad_seg_tensor(batch, "cum_monitor_hours", N)
    out["gap_ratio"] = _pad_seg_tensor(batch, "gap_ratio", N)
    out["delta_t_hours"] = _pad_seg_tensor(batch, "delta_t_hours", N)
    out["cs_label"] = _pad_seg_tensor(batch, "cs_label", N).to(torch.bool)
    out["bg_label"] = _pad_seg_tensor(batch, "bg_label", N).to(torch.bool)
    out["time_from_labor_onset"] = _pad_seg_tensor(
        batch, "time_from_labor_onset", N, fill_value=float("nan")
    )
    out["second_stage_onset"] = _pad_seg_tensor(
        batch, "second_stage_onset", N, fill_value=float("nan")
    )
    out["epoch"] = _pad_seg_tensor(batch, "epoch", N, fill_value=0.0)

    # segment_mask
    seg_mask = torch.zeros(B, N, dtype=torch.bool)
    for i, s in enumerate(batch):
        seg_mask[i, : int(s["num_segments"])] = True
    out["segment_mask"] = seg_mask

    # Relative-time bucket index, computed once per batch.
    out["rel_bucket_idx"] = build_relative_time_bucket_index(
        out["cum_monitor_hours"],
        num_buckets=rel_time_num_buckets,
        d_max=rel_time_d_max,
    )

    out["label_3"] = torch.tensor(
        [int(s["label_3"]) for s in batch], dtype=torch.long
    )
    out["label_bin"] = torch.tensor(
        [float(s["label_bin"]) for s in batch], dtype=torch.float32
    )
    out["num_segments"] = torch.tensor(
        [int(s["num_segments"]) for s in batch], dtype=torch.long
    )
    out["guid"] = [str(s["guid"]) for s in batch]

    return out


__all__ = ["guid_sequence_collate_fn", "build_relative_time_bucket_index"]
