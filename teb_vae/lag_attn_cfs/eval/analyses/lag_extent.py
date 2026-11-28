r"""Where on the lag axis the source starts and stops mattering, per head, per class, through labour.

Every other lag readout of this pipeline reduces a profile to a location (argmax, centroid,
median). This analysis asks for the **extent**: the earliest and the latest lag that carries the
coupling, read three ways that answer different questions, and placed on the two axes the
clinical question is asked on (class, hours before delivery) and on the contraction cycle.

**1. Where the attention looks (observational, per head).** ``1.5-entmax`` gives exact zeros, so
each head at each anchor has a *support*: its first and last lag with weight above
:data:`SUPPORT_EPS`. Beside it, the lags where the cumulative weight reaches $10/50/90\,\%$ and
the centroid. Reported per head, never averaged over heads first: heads own disjoint latent groups,
and in practice they own disjoint lag windows too.

**2. Where the forecast needs the source (interventional, two truncation families).** Keys are
removed through the model's own running-gap removal mask (``_attend_and_pose`` with an extended
``removed``), so the target encoder and the prior are the matched forward's and only the attention,
the posterior and the decode are re-run. Each arm is decoded at its latent **mean** (the
``mean_pred_gap`` estimator) so there is no sampling noise between arms.

* *near family* keeps lags $\le k$. Its gain share $s_n(k)$ rises from the no-key arm (the gate
  closes, $q = p$) to the full forward. ``effect_mid_lag`` and ``effect_end_lag`` are the lags at
  which it first reaches $50\,\%$ and $90\,\%$: the latest lag that still adds gain.
* *far family* keeps lags $\ge k$. Its share $s_f(k)$ falls from $1$. ``effect_start_lag`` is the
  lag at which it falls below $90\,\%$: only $10\,\%$ of the effect lies at nearer lags.

The window ``[effect_start_lag, effect_end_lag]`` is the interventional answer to "which lags
does the forecast use". **Removal renormalises** the entmax over the remaining keys, so an arm is
"the model reading only these lags", not "these lags' share of the sum"; KL in particular is not
monotone under it (renormalised mass inflates the near keys), which is why the shares are taken
on the forecast gain. Both families are also resolved by horizon step $\tau$. The query is one
vector per anchor serving the whole horizon, so a physiological delay $d$ needs lag
$\ell = d - \tau - 1$ at step $\tau$: a model that has learned the delay needs *far* lags at
early $\tau$ and lag $0$ late (slope $-1$ against $\tau$); a recency filter is flat (slope $0$).
``lag_extent_horizon.csv`` tables both beside the prediction from the data's own delay.

**3. Whether the attention follows the contraction (event-locked).** For every anchor with a
detected contraction behind it, the lags covering that contraction's body are
$[t - s_{\mathrm{end}},\ t - s_{\mathrm{on}}]$. A model that tracks the event moves its weight
along that diagonal as the contraction ages (centroid slope $1$ against age); a fixed filter puts
on it only what its time-invariant profile happens to (the *null*: the run's pooled per-head
profile summed over the same lags). One more arm removes exactly those keys, so the forecast's
dependence on the contraction is measured, not inferred, and its **start and end** in contraction
age are where that loss first and last exceeds half its peak. The UA keys are causal filter
outputs whose memory spans minutes, so a key *after* the contraction still carries it: the
knockout measures reading the contraction's own steps, not every trace of it.

**4. The data's own delay (reference).** Decelerations on the raw FHR (bpm) are paired with the
most recent contraction peak within :data:`MAX_DELAY_S`, and against a null that pairs each
segment's decelerations with another segment's contractions. The excess of matched over null is
the "correct lag" the model could be judged against.

Everything is aggregated per recording first (each recording's anchor mean or its own truncation
curve), then by class and by :data:`cohort.TRAJECTORY_BIN_HOURS` windows before delivery. Two
readouts are tested per window across classes (Kruskal-Wallis, Holm over windows):
``effect_end_lag`` and the dominant head's ``att_last``. The rest is descriptive.

The analysis reaches for the model (``context.task``/``context.loader``) and records a skip
without them; ``caps.lag_extent`` bounds the segments (a seeded draw inside
``max_hours_before_delivery``). Lags are stored-coefficient time (:data:`lag_axis.GROUP_DELAY_NOTE`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn_cfs.eval import cohort, events, lag_axis, traces
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels, stats as shared_stats
from teb_vae.lag_attn_cfs.eval.dataset_rows import dataset_index_map, subset_loader
from teb_vae.lag_attn_cfs.eval.frames import grouped_frame_entry
from teb_vae.lag_attn_cfs.eval.metrics import (
    anchor_support,
    batch_field,
    batch_size_of,
    forecast_likelihood_terms,
    forward_kwargs,
    masked_raw_block_per_horizon_step,
    model_inputs,
)
from teb_vae.lag_attn_rws.nets.controls import _attend_and_pose

ANALYSIS_DIRNAME = "lag_extent"
CAP_KEY = "lag_extent"
SEED_OFFSET = 7_919

#: A weight at or below this is outside a head's support. entmax zeros are exact; the floor only
#: absorbs float16-level noise.
SUPPORT_EPS = 1e-4
#: Lags of the truncation grid (both families share it); log-spaced, so the near lags, where the
#: profile changes fastest, are resolved one by one.
GRID_POINTS = 14
#: Below this recording-level gain (nats/anchor) there is no effect to locate, and the shares are NaN.
MIN_EFFECT_NATS = 0.25
#: The share levels that name the effect's start, middle and end.
START_LEVEL, MID_LEVEL, END_LEVEL = 0.9, 0.5, 0.9
#: Contraction-age bins of the event-locked tables, in seconds, and the range they cover.
AGE_BIN_S, AGE_MAX_S = 8.0, 240.0
#: Decelerations further than this after a contraction peak are not paired with it.
MAX_DELAY_S = 120.0
DELAY_BIN_S = 4.0
#: The two readouts tested across classes per time window.
TESTED = ("effect_end_lag", "dominant_att_last")
MIN_RECORDINGS_PER_GROUP = 3

ANCHORS_FILENAME = "lag_extent_anchors.parquet"
ARMS_FILENAME = "lag_extent_arms.npz"
PER_RECORDING_FILENAME = "lag_extent_per_recording.csv"
HEADS_FILENAME = "lag_extent_heads.csv"
TRUNCATION_FILENAME = "lag_extent_truncation.csv"
HORIZON_FILENAME = "lag_extent_horizon.csv"
TRAJECTORY_FILENAME = "lag_extent_trajectory.csv"
SIGNIFICANCE_FILENAME = "lag_extent_trajectory_significance.csv"
EVENT_FILENAME = "lag_extent_event_locked.csv"
TRACKING_FILENAME = "lag_extent_tracking.csv"
DELAYS_FILENAME = "lag_extent_delays.csv"
FIGURES = ("lag_extent_heads", "lag_extent_reach", "lag_extent_events")


# =============================================================================
# Pure lag-axis arithmetic
# =============================================================================
def lag_grid(n_lags: int, n_points: int = GRID_POINTS) -> np.ndarray:
    """Log-spaced integer lags in ``[0, n_lags - 1]``, always holding both ends."""
    grid = np.unique(np.round(np.geomspace(1, n_lags, n_points)).astype(int) - 1)
    return np.unique(np.concatenate([[0], grid, [n_lags - 1]]))


def support_extent(weights: np.ndarray, eps: float = SUPPORT_EPS) -> Tuple[np.ndarray, np.ndarray]:
    """First and last lag (last axis) with weight above ``eps``; NaN where none is."""
    inside = weights > eps
    any_ = inside.any(-1)
    n = weights.shape[-1]
    first = np.where(any_, inside.argmax(-1), np.nan)
    last = np.where(any_, n - 1 - inside[..., ::-1].argmax(-1), np.nan)
    return first.astype(np.float64), last.astype(np.float64)


def mass_quantile(weights: np.ndarray, level: float) -> np.ndarray:
    """The first lag at which the normalised cumulative weight reaches ``level``; NaN on zero mass."""
    total = weights.sum(-1, keepdims=True)
    cumulative = np.cumsum(weights, -1) / np.where(total > 0, total, 1.0)
    out = (cumulative >= level - 1e-9).argmax(-1).astype(np.float64)
    return np.where(total[..., 0] > 0, out, np.nan)


def crossing(grid: np.ndarray, share: np.ndarray, level: float, *, rising: bool) -> np.ndarray:
    """Interpolated lag where a share curve (last axis, on ``grid``) first crosses ``level``.

    ``rising`` looks for the first point at or above the level (the near family); otherwise the
    first point below it (the far family), interpolated back to the crossing. A curve that never
    crosses gives the last grid lag (rising) or NaN (falling: it never dropped); a NaN curve NaN.
    """
    share = np.asarray(share, dtype=np.float64)
    hit = share >= level if rising else share < level
    found = hit.any(-1)
    j = hit.argmax(-1)
    prev = np.clip(j - 1, 0, None)
    take = lambda a, i: np.take_along_axis(a, i[..., None], -1)[..., 0]  # noqa: E731
    s0, s1 = take(share, prev), take(share, j)
    g = np.broadcast_to(grid.astype(np.float64), share.shape)
    g0, g1 = take(g, prev), take(g, j)
    frac = np.where(np.abs(s1 - s0) > 1e-12, (level - s0) / np.where(s1 == s0, 1.0, s1 - s0), 0.0)
    value = np.where(j > 0, g0 + np.clip(frac, 0.0, 1.0) * (g1 - g0), g1)
    fallback = float(grid[-1]) if rising else np.nan
    value = np.where(found, value, fallback)
    return np.where(np.isnan(share).any(-1), np.nan, value)


def effect_lags(gain_none: Any, gain_near: np.ndarray, gain_far: np.ndarray, grid: np.ndarray, *, min_effect: float = MIN_EFFECT_NATS) -> Dict[str, Any]:
    """Shares and start/mid/end lags from mean gains of the two truncation families.

    ``gain_near[..., i]`` keeps lags ``<= grid[i]`` and ``gain_far[..., i]`` keeps ``>= grid[i]``;
    ``gain_near[..., -1]`` is the full forward and ``gain_none`` the no-key arm. A curve whose full
    gain exceeds the no-key arm by less than ``min_effect`` has no effect to locate and gives NaN.
    """
    full = gain_near[..., -1]
    span = full - np.asarray(gain_none)
    ok = span >= min_effect
    denom = np.where(ok, span, np.nan)[..., None]
    near = (gain_near - np.asarray(gain_none)[..., None]) / denom
    far = (gain_far - np.asarray(gain_none)[..., None]) / denom
    return {
        "near_share": near, "far_share": far,
        "effect_start_lag": crossing(grid, far, START_LEVEL, rising=False),
        "effect_mid_lag": crossing(grid, near, MID_LEVEL, rising=True),
        "effect_end_lag": crossing(grid, near, END_LEVEL, rising=True),
    }


def contraction_lags(onset_raw: np.ndarray, end_raw: np.ndarray, anchors: np.ndarray, decimation: int) -> Dict[str, np.ndarray]:
    r"""For each anchor, the most recent contraction at or before it and the lags covering its body.

    On the causal-feature step grid, stored step $t$ sits at trimmed raw sample $Dt$ and reads only
    samples $\le Dt$. A contraction spanning raw indices $[i_{\mathrm{on}}, i_{\mathrm{end}}]$ is
    therefore visible at anchor $t$ once $Dt \ge i_{\mathrm{on}}$, and its body spans decimated
    steps $[\lceil i_{\mathrm{on}} / D \rceil, \lceil i_{\mathrm{end}} / D \rceil]$.

    Args:
        onset_raw: Contraction onset indices in raw samples.
        end_raw: Contraction end indices in raw samples.
        anchors: Anchor indices on the decimated step grid.
        decimation: Raw samples per decimated step $D$.

    Returns:
        Mapping with ``age_s`` (anchor causal sample $Dt$ minus onset in seconds), ``lo``/``hi``
        (inclusive lags of the body, ``lo`` is $0$ while the contraction is still running) and
        ``index`` ($-1$ where no contraction is behind).
    """
    anchors = np.asarray(anchors, dtype=np.int64)
    endpoint = decimation * anchors
    order = np.argsort(onset_raw)
    onset_raw, end_raw = np.asarray(onset_raw)[order], np.asarray(end_raw)[order]
    k = np.searchsorted(onset_raw, endpoint, side="right") - 1
    has = k >= 0
    kk = np.clip(k, 0, None)
    on_step = (
        ((onset_raw[kk] + decimation - 1) // decimation)
        if onset_raw.size else np.zeros_like(anchors)
    )
    end_step = (
        ((end_raw[kk] + decimation - 1) // decimation)
        if end_raw.size else np.zeros_like(anchors)
    )
    lo = anchors - np.minimum(end_step, anchors)
    hi = anchors - on_step
    age = (endpoint - (onset_raw[kk] if onset_raw.size else 0)) / events.FS_RAW
    return {
        "age_s": np.where(has, age, np.nan),
        "lo": np.where(has, lo, -1), "hi": np.where(has, hi, -1),
        "index": np.where(has, k, -1),
    }


def window_mass(weights: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Sum of ``weights[..., l]`` over ``lo <= l <= hi`` per row; NaN where ``lo < 0``."""
    lags = np.arange(weights.shape[-1])
    inside = (lags >= lo[:, None]) & (lags <= hi[:, None])
    if weights.ndim == 3:
        inside = inside[:, None, :]
    total = (weights * inside).sum(-1)
    return np.where((lo >= 0)[(...,) + (None,) * (total.ndim - 1)], total, np.nan)


def paired_delays(peaks_raw: np.ndarray, nadirs_raw: np.ndarray, *, max_delay_s: float = MAX_DELAY_S) -> np.ndarray:
    """Seconds from each deceleration nadir's most recent contraction peak, within ``max_delay_s``."""
    peaks = np.sort(np.asarray(peaks_raw, dtype=np.float64))
    nadirs = np.asarray(nadirs_raw, dtype=np.float64)
    if peaks.size == 0 or nadirs.size == 0:
        return np.empty(0)
    k = np.searchsorted(peaks, nadirs, side="right") - 1
    delay = (nadirs - peaks[np.clip(k, 0, None)]) / events.FS_RAW
    return delay[(k >= 0) & (delay <= max_delay_s)]


# =============================================================================
# Collection: one forward, the truncation arms and the events, per batch
# =============================================================================
def _gather(tensor: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Gather a dense ``(B, T, ...)`` tensor at the decoded anchors ``(B, A)``."""
    index = anchors.to(torch.long).view(*anchors.shape, *([1] * (tensor.dim() - 2)))
    return tensor.gather(1, index.expand(-1, -1, *tensor.shape[2:]))


def _per_tau(model: Any, mu_latent: torch.Tensor, anchors: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor, persistence: Optional[torch.Tensor], likelihood: str) -> torch.Tensor:
    """Mean-decoded block score of one latent mean, ``(B, A, H)``."""
    mu, logvar = model.decoder(_gather(mu_latent, anchors), persistence=persistence)
    return masked_raw_block_per_horizon_step(
        mu, target, mask, likelihood=likelihood, logvar=logvar, **forecast_likelihood_terms(model)
    )


@torch.no_grad()
def collect_batch(task: Any, batch: Any, *, grid: np.ndarray, raw_scales: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    """Run one batch: the matched forward, every truncation arm, the contraction arm and the events."""
    model = task.orig_model
    likelihood = str(task.hparams.get("likelihood", "gaussian_nll"))
    y_st, y_ph, u_stream, target_features, weight = model_inputs(task, batch)
    outputs = model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1,
                    **forward_kwargs(task, batch, u_stream))
    anchors = outputs["anchor_index"]
    mask, _coverage, _ = anchor_support(model, weight, outputs)
    contributing = mask.sum(-1) > 0
    target = model._build_forecast_target(target_features, anchors)
    persistence = outputs.get("persistence")
    B, T = anchors.shape[0], outputs["mu_prior"].shape[1]
    L = int(outputs["attn_weights"].shape[-1])
    device = anchors.device

    base = _per_tau(model, outputs["mu_prior"], anchors, target, mask, persistence, likelihood)
    full = _per_tau(model, outputs["mu_post"], anchors, target, mask, persistence, likelihood)
    removed0 = outputs.get("source_key_removed")
    if removed0 is None:
        removed0 = torch.zeros(B, T, L, dtype=torch.bool, device=device)

    def arm(extra: torch.Tensor) -> torch.Tensor:
        """Gain per (B, A, H) with the extra keys removed at every step."""
        _, mu_q, _, _ = _attend_and_pose(model, outputs, outputs["source_state"], removed=removed0 | extra)
        return base - _per_tau(model, mu_q, anchors, target, mask, persistence, likelihood)

    lags = torch.arange(L, device=device)
    expand = lambda keep: (~keep)[None, None].expand(B, T, L)  # noqa: E731
    gains = {"none": arm(expand(torch.zeros(L, dtype=torch.bool, device=device)))}
    gains["near"] = torch.stack([arm(expand(lags <= int(k))) for k in grid], -1)
    gains["far"] = torch.stack([arm(expand(lags >= int(k))) for k in grid], -1)

    # -- events on the raw traces ------------------------------------------------------------
    decimation = int(model.geometry.decimation)
    anchors_np = anchors.cpu().numpy()
    up = batch_field(batch, "up")
    fhr = batch_field(batch, "fhr")
    up_valid = batch_field(batch, "up_valid")
    w_np = batch_field(batch, "weight").detach().cpu().double().numpy().reshape(B, -1)
    ctr = {key: np.full(anchors_np.shape, np.nan if key == "age_s" else -1, dtype=np.float64)
           for key in ("age_s", "lo", "hi")}
    peaks, nadirs = [], []
    for i in range(B):
        if not isinstance(up, torch.Tensor):
            peaks.append(np.empty(0)), nadirs.append(np.empty(0))
            continue
        up_i = up[i].detach().cpu().double().numpy().reshape(-1)
        valid = events.raw_validity(w_np[i], decimation=decimation, raw_len=up_i.size)
        if isinstance(up_valid, torch.Tensor):
            valid = valid & up_valid[i].detach().cpu().numpy().reshape(-1)[: up_i.size].astype(bool)
        found = events.detect_contractions(up_i, valid=valid)
        geo = contraction_lags(found["onset_raw"], found["end_raw"], anchors_np[i], decimation)
        for key in ctr:
            ctr[key][i] = geo[key]
        peaks.append(np.asarray(found["peak_raw"], dtype=np.float64))
        if isinstance(fhr, torch.Tensor):
            bpm, _ = traces.physical_raw(fhr[i].detach().cpu().numpy(), "fhr", raw_scales, valid)
            ok = np.isfinite(bpm)
            nadirs.append(np.asarray(events.detect_decelerations(np.nan_to_num(bpm), valid=ok)["nadir_raw"], dtype=np.float64))
        else:
            nadirs.append(np.empty(0))

    # The contraction-knockout arm: per anchor, the keys covering its most recent contraction.
    lo = torch.as_tensor(ctr["lo"], device=device)
    hi = torch.as_tensor(ctr["hi"], device=device)
    in_window = (lo >= 0) & (lo <= L - 1)
    body = (lags[None, None] >= lo[..., None]) & (lags[None, None] <= hi[..., None]) & in_window[..., None]
    extra = torch.zeros(B, T, L, dtype=torch.bool, device=device)
    extra.scatter_(1, anchors.long()[..., None].expand(-1, -1, L), body)
    gains["contraction"] = arm(extra)

    attn = _gather(outputs["attn_weights"], anchors)
    return {
        "guid": [str(g) for g in (batch_field(batch, "guid") or [None] * B)],
        "epoch": batch_field(batch, "epoch").detach().cpu().double().numpy().reshape(-1),
        "labels": labels.batch_labels(batch, B),
        "anchor": anchors_np,
        "contributing": contributing.cpu().numpy(),
        "attn": attn.float().cpu().numpy(),
        "kl_head": _gather(outputs["kld_per_t_per_head"], anchors).float().cpu().numpy(),
        "kl_map": _gather(outputs["source_kl_lag_map"], anchors).float().cpu().numpy(),
        "gain_full": (base - full).double().cpu().numpy(),
        "gains": {k: v.double().cpu().numpy() for k, v in gains.items()},
        "ctr": ctr,
        "peaks": peaks, "nadirs": nadirs,
    }


def reduce_batch(out: Dict[str, Any], *, step_s: float) -> Tuple[pd.DataFrame, np.ndarray, Tuple[Dict[str, np.ndarray], Dict[str, int]], Dict[str, np.ndarray]]:
    """Flatten one batch to per-anchor rows (contributing anchors only), the arm-gain matrix and
    the per-segment per-tau arm sums.
    """
    keep = out["contributing"]
    seg, col = np.nonzero(keep)
    attn = out["attn"][seg, col]  # n, M, L
    M = attn.shape[1]
    rows: Dict[str, Any] = {
        "guid": np.asarray(out["guid"])[seg],
        "epoch": out["epoch"][seg],
        labels.CLASS_COLUMN: np.asarray(out["labels"][labels.CLASS_COLUMN], dtype=object)[seg],
        labels.SUBGROUP_COLUMN: np.asarray(out["labels"][labels.SUBGROUP_COLUMN], dtype=object)[seg],
        "anchor": out["anchor"][seg, col],
        "gain": out["gain_full"][seg, col].sum(-1),
        "kl": out["kl_head"][seg, col].sum(-1),
    }
    first, last = support_extent(attn)
    lag_idx = np.arange(attn.shape[-1])
    for m in range(M):
        rows[f"kl_h{m}"] = out["kl_head"][seg, col, m]
        rows[f"att_first_h{m}"], rows[f"att_last_h{m}"] = first[:, m], last[:, m]
        for level in (0.1, 0.5, 0.9):
            rows[f"att_q{int(level * 100)}_h{m}"] = mass_quantile(attn[:, m], level)
        rows[f"att_centroid_h{m}"] = (attn[:, m] * lag_idx).sum(-1) / np.maximum(attn[:, m].sum(-1), 1e-12)
    kl_map = np.clip(out["kl_map"][seg, col], 0.0, None)
    for level in (0.1, 0.5, 0.9):
        rows[f"klmap_q{int(level * 100)}"] = mass_quantile(kl_map, level)
    lo, hi = out["ctr"]["lo"][seg, col].astype(int), out["ctr"]["hi"][seg, col].astype(int)
    rows["ctr_age_s"] = out["ctr"]["age_s"][seg, col]
    rows["ctr_lo"], rows["ctr_hi"] = lo, hi
    mass = window_mass(attn, lo, hi)  # n, M
    for m in range(M):
        rows[f"ctr_mass_h{m}"] = mass[:, m]
    rows["ctr_kl_mass"] = window_mass(kl_map, lo, hi)
    gains = out["gains"]
    arm_matrix = np.concatenate([
        gains["none"][seg, col].sum(-1)[:, None],
        gains["near"][seg, col].sum(-2),
        gains["far"][seg, col].sum(-2),
        gains["contraction"][seg, col].sum(-1)[:, None],
    ], axis=1).astype(np.float32)
    rows["ctr_loss"] = rows["gain"] - arm_matrix[:, -1]
    frame = pd.DataFrame(rows)
    frame["lag_step_s"] = step_s
    # Per segment, per arm, per tau: sum over contributing anchors (and the count), for the
    # horizon-resolved reach.
    w = keep[..., None].astype(np.float64)
    per_tau = {
        "none": (gains["none"] * w).sum(1),
        "near": (gains["near"] * w[..., None]).sum(1),
        "far": (gains["far"] * w[..., None]).sum(1),
        "count": keep.sum(1),
        "class": np.asarray(out["labels"][labels.CLASS_COLUMN], dtype=object),
    }
    classes = np.asarray(rows[labels.CLASS_COLUMN], dtype=object)
    attn_sums = {"all": attn.sum(0)}
    for name in {str(c) for c in classes if c is not None}:
        attn_sums[name] = attn[classes.astype(str) == name].sum(0)
    counts = {name: int((classes.astype(str) == name).sum()) if name != "all" else len(classes) for name in attn_sums}
    return frame, arm_matrix, (attn_sums, counts), per_tau


# =============================================================================
# Reductions
# =============================================================================
def _arm_columns(grid: np.ndarray) -> List[str]:
    return ["none"] + [f"near_{k}" for k in grid] + [f"far_{k}" for k in grid] + ["contraction"]


def curve_readouts(arms: pd.DataFrame, grid: np.ndarray) -> Dict[str, Any]:
    """Start/mid/end lags of one population's mean arm gains (a DataFrame of arm columns)."""
    mean = arms.mean(0)
    near = np.array([mean[f"near_{k}"] for k in grid])
    far = np.array([mean[f"far_{k}"] for k in grid])
    lags = effect_lags(mean["none"], near, far, grid)
    return {key: float(lags[key]) for key in ("effect_start_lag", "effect_mid_lag", "effect_end_lag")}


def per_group_readouts(anchors: pd.DataFrame, arms: pd.DataFrame, keys: Sequence[str], value_columns: Sequence[str], grid: np.ndarray) -> pd.DataFrame:
    """Per group: anchor means of ``value_columns`` and the truncation readouts of the group's own curve."""
    carried = list(dict.fromkeys(list(keys) + ["guid", "epoch"] + list(value_columns)))
    joined = pd.concat([anchors[carried].reset_index(drop=True),
                        arms.reset_index(drop=True)], axis=1)
    records = []
    for key, part in joined.groupby(list(keys), sort=True, dropna=True):
        record = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        record["n_anchors"] = int(len(part))
        record["n_segments"] = int(part[["guid", "epoch"]].drop_duplicates().shape[0])
        record.update(part[list(value_columns)].mean().to_dict())
        record.update(curve_readouts(part[arms.columns], grid))
        records.append(record)
    return pd.DataFrame(records)


def head_table(anchors: pd.DataFrame, n_heads: int, step_s: float) -> pd.DataFrame:
    """Per head and class (and all): KL share and the attention extent, as medians of recording means."""
    rows = []
    cohorts = [("all", anchors)] + [(str(c), p) for c, p in anchors.groupby(labels.CLASS_COLUMN)]
    for name, part in cohorts:
        rec = part.groupby("guid").mean(numeric_only=True)
        kl_total = rec[[f"kl_h{m}" for m in range(n_heads)]].sum(1)
        for m in range(n_heads):
            row = {"cohort": name, "head": m, "n_recordings": int(len(rec)),
                   "kl_nats": float(rec[f"kl_h{m}"].mean()),
                   "kl_share": float((rec[f"kl_h{m}"] / kl_total.replace(0, np.nan)).mean())}
            for stat in ("first", "last", "q10", "q50", "q90", "centroid"):
                values = rec[f"att_{stat}_h{m}"]
                row[f"att_{stat}_lag_median"] = float(values.median())
                row[f"att_{stat}_lag_q25"] = float(values.quantile(0.25))
                row[f"att_{stat}_lag_q75"] = float(values.quantile(0.75))
            row["att_last_s_median"] = row["att_last_lag_median"] * step_s
            rows.append(row)
    return pd.DataFrame(rows)


def event_locked(anchors: pd.DataFrame, n_heads: int, pooled_profile: np.ndarray, *, step_s: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per contraction-age bin and class: attention on the contraction vs the fixed-filter null,
    the KL on it, the knockout loss and the gain; plus per head the tracking slope and index.
    """
    frame = anchors[np.isfinite(anchors["ctr_age_s"]) & (anchors["ctr_age_s"] < AGE_MAX_S)].copy()
    lo, hi = frame["ctr_lo"].to_numpy(), frame["ctr_hi"].to_numpy()
    null = window_mass(np.broadcast_to(pooled_profile, (len(frame),) + pooled_profile.shape), lo, hi)
    for m in range(n_heads):
        frame[f"ctr_null_h{m}"] = null[:, m]
    frame["age_bin_s"] = (np.floor(frame["ctr_age_s"] / AGE_BIN_S) + 0.5) * AGE_BIN_S
    values = (["gain", "kl", "ctr_loss", "ctr_kl_mass"] + [f"ctr_mass_h{m}" for m in range(n_heads)]
              + [f"ctr_null_h{m}" for m in range(n_heads)] + [f"att_centroid_h{m}" for m in range(n_heads)])
    rows = []
    for cohort_name, part in [("all", frame)] + [(str(c), p) for c, p in frame.groupby(labels.CLASS_COLUMN)]:
        # Recording means within each bin, then the mean over recordings.
        per = part.groupby(["age_bin_s", "guid"])[values].mean().groupby("age_bin_s")
        table = per.mean()
        table["n_recordings"] = per.size()
        table["cohort"] = cohort_name
        rows.append(table.reset_index())
    binned = pd.concat(rows, ignore_index=True)
    for m in range(n_heads):
        binned[f"tracking_index_h{m}"] = binned[f"ctr_mass_h{m}"] - binned[f"ctr_null_h{m}"]

    tracking = []
    for cohort_name, part in [("all", frame)] + [(str(c), p) for c, p in frame.groupby(labels.CLASS_COLUMN)]:
        in_window = part[part["ctr_age_s"] <= step_s * pooled_profile.shape[-1]]
        age_steps = in_window["ctr_age_s"].to_numpy() / step_s
        prof = binned[binned["cohort"] == cohort_name].sort_values("age_bin_s")
        loss = prof["ctr_loss"].to_numpy()
        start = end = np.nan
        if loss.size and np.nanmax(loss) > 0:
            above = np.flatnonzero(loss >= 0.5 * np.nanmax(loss))
            start, end = float(prof["age_bin_s"].iloc[above[0]]), float(prof["age_bin_s"].iloc[above[-1]])
        for m in range(n_heads):
            cen = in_window[f"att_centroid_h{m}"].to_numpy()
            ok = np.isfinite(cen) & np.isfinite(age_steps)
            slope = float(np.polyfit(age_steps[ok], cen[ok], 1)[0]) if ok.sum() > 10 else np.nan
            tracking.append({
                "cohort": cohort_name, "head": m, "n_anchors": int(ok.sum()),
                "centroid_slope_vs_age": slope,
                "mass_on_contraction": float(part[f"ctr_mass_h{m}"].mean()),
                "null_mass": float(part[f"ctr_null_h{m}"].mean()),
                "tracking_index": float((part[f"ctr_mass_h{m}"] - part[f"ctr_null_h{m}"]).mean()),
                "knockout_start_age_s": start, "knockout_end_age_s": end,
                "knockout_peak_loss_nats": float(np.nanmax(loss)) if loss.size else np.nan,
            })
    return binned, pd.DataFrame(tracking)


def horizon_table(per_tau: Dict[str, Any], grid: np.ndarray, *, delay_steps_median: float) -> pd.DataFrame:
    """Per class and horizon step: the truncation readouts of the pooled per-tau curve, beside the
    lag a fixed physiological delay would need there, ``max(d - tau - 1, 0)``."""
    rows = []
    classes = np.asarray(per_tau["class"], dtype=object)
    for name in ["all"] + sorted({str(c) for c in classes if c is not None}):
        sel = np.ones(classes.size, bool) if name == "all" else (classes.astype(str) == name)
        count = per_tau["count"][sel].sum()
        if count == 0:
            continue
        none = per_tau["none"][sel].sum(0) / count  # H
        near = per_tau["near"][sel].sum(0) / count  # H, K
        far = per_tau["far"][sel].sum(0) / count
        # Pooled over every anchor of the cohort, and per step rather than per block, so the floor
        # is only that there is a positive effect to locate.
        lags = effect_lags(none, near, far, grid, min_effect=1e-6)
        for tau in range(none.shape[0]):
            rows.append({
                "cohort": name, "horizon_step": tau, "gain_nats": float(near[tau, -1]),
                "effect_start_lag": float(lags["effect_start_lag"][tau]),
                "effect_mid_lag": float(lags["effect_mid_lag"][tau]),
                "effect_end_lag": float(lags["effect_end_lag"][tau]),
                "physiological_lag": float(max(delay_steps_median - tau - 1, 0)) if np.isfinite(delay_steps_median) else np.nan,
            })
    table = pd.DataFrame(rows)
    if not table.empty:
        slopes = {}
        for name, part in table.groupby("cohort"):
            ok = np.isfinite(part["effect_end_lag"])
            slopes[name] = float(np.polyfit(part["horizon_step"][ok], part["effect_end_lag"][ok], 1)[0]) if ok.sum() > 2 else np.nan
        table["end_lag_slope_vs_tau"] = table["cohort"].map(slopes)
    return table


def delay_table(delays: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """The matched and null delay histograms per class, and the summary record."""
    frame = pd.DataFrame(delays, columns=["guid", labels.CLASS_COLUMN, "kind", "delay_s"])
    edges = np.arange(0.0, MAX_DELAY_S + DELAY_BIN_S, DELAY_BIN_S)
    rows, summary = [], {}
    for name, part in [("all", frame)] + [(str(c), p) for c, p in frame.groupby(labels.CLASS_COLUMN)]:
        matched = part.loc[part["kind"] == "matched", "delay_s"].to_numpy()
        null = part.loc[part["kind"] == "null", "delay_s"].to_numpy()
        hm = np.histogram(matched, edges)[0] / max(len(part["guid"].unique()), 1)
        hn = np.histogram(null, edges)[0] / max(len(part["guid"].unique()), 1)
        for i in range(len(edges) - 1):
            rows.append({"cohort": name, "delay_lo_s": edges[i], "delay_hi_s": edges[i + 1],
                         "matched_per_recording": hm[i], "null_per_recording": hn[i], "excess": hm[i] - hn[i]})
        excess = hm - hn
        summary[name] = {
            "n_matched": int(matched.size), "n_null": int(null.size),
            "median_matched_s": float(np.median(matched)) if matched.size else np.nan,
            "median_null_s": float(np.median(null)) if null.size else np.nan,
            "excess_mode_s": float(edges[int(np.argmax(excess))] + DELAY_BIN_S / 2) if excess.size and excess.max() > 0 else np.nan,
            "excess_median_s": _weighted_median(edges[:-1] + DELAY_BIN_S / 2, np.clip(excess, 0, None)),
        }
    return pd.DataFrame(rows), summary


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if weights.sum() <= 0:
        return float("nan")
    cumulative = np.cumsum(weights) / weights.sum()
    return float(values[np.searchsorted(cumulative, 0.5)])


def trajectories(anchors: pd.DataFrame, arms: pd.DataFrame, grid: np.ndarray, dominant: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per (class, window, recording) readouts, their trajectory rows and the tested windows."""
    binned = cohort.add_time_bins(anchors.assign(dominant_att_last=anchors[f"att_last_h{dominant}"],
                                                 dominant_att_q90=anchors[f"att_q90_h{dominant}"]))
    arms_b = arms.loc[binned.index]
    per = per_group_readouts(
        binned, arms_b, [labels.CLASS_COLUMN, cohort.BIN_COLUMN, cohort.BIN_CENTER_COLUMN, "guid"],
        ["gain", "kl", "dominant_att_last", "dominant_att_q90", "klmap_q90"], grid,
    ).rename(columns={labels.CLASS_COLUMN: "group"})
    metrics = ["gain", "kl", "dominant_att_last", "dominant_att_q90", "klmap_q90",
               "effect_start_lag", "effect_mid_lag", "effect_end_lag"]
    rows = []
    for metric in metrics:
        rows.extend(cohort.trajectory_rows(per, metric, metric=metric))
    tests = []
    for metric in TESTED:
        windows: Dict[Any, Dict[str, np.ndarray]] = {}
        for b, part in per.groupby(cohort.BIN_COLUMN, sort=True):
            groups = {}
            for g in labels.ordered_groups(part["group"].unique(), labels.CLASS_COLUMN):
                v = part.loc[part["group"] == g, metric].to_numpy(dtype=np.float64)
                v = v[np.isfinite(v)]
                if v.size >= MIN_RECORDINGS_PER_GROUP:
                    groups[g] = v
            windows[int(b)] = groups
        result = shared_stats.windowed_group_comparisons(windows)
        for record in result["per_window"]:
            tests.append({"metric": metric, **{k: v for k, v in record.items() if not isinstance(v, dict)}})
    return per, pd.DataFrame(rows), pd.DataFrame(tests)


# =============================================================================
# Figures
# =============================================================================
def _class_colors(names: Sequence[str]) -> Dict[str, str]:
    colors = figures.group_colors([n for n in names if n != "all"])
    colors["all"] = "#272727"
    return colors


def build_heads_figure(anchors: pd.DataFrame, attn_by_class: Dict[str, np.ndarray], heads: pd.DataFrame, step_s: float) -> Any:
    """Per head: the mean attention over lags per class, its support band, and the KL share."""
    n_heads = next(iter(attn_by_class.values())).shape[0]
    figure, axes = figures.new_figure(n_heads, 2, height_per_row=1.9, width=10.0)
    colors = _class_colors(list(attn_by_class))
    seconds = np.arange(next(iter(attn_by_class.values())).shape[-1]) * step_s
    for m in range(n_heads):
        ax = axes[m, 0]
        for name, prof in attn_by_class.items():
            ax.plot(seconds, prof[m], color=colors[name], lw=1.2, label=name)
        row = heads[(heads["cohort"] == "all") & (heads["head"] == m)].iloc[0]
        ax.axvspan(row["att_first_lag_median"] * step_s, row["att_last_lag_median"] * step_s, color="0.85", zorder=0)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(f"head {m}: mean attention (shade: median support), KL share {row['kl_share']:.2f}", fontsize=8)
        ax.set_xlabel("lag (s)")
        if m == 0:
            ax.legend(fontsize=7)
        ax = axes[m, 1]
        cls = heads[(heads["head"] == m) & (heads["cohort"] != "all")]
        x = np.arange(len(cls))
        for stat, offset in (("first", -0.2), ("q50", 0.0), ("last", 0.2)):
            ax.errorbar(x + offset, cls[f"att_{stat}_lag_median"] * step_s,
                        yerr=[(cls[f"att_{stat}_lag_median"] - cls[f"att_{stat}_lag_q25"]) * step_s,
                              (cls[f"att_{stat}_lag_q75"] - cls[f"att_{stat}_lag_median"]) * step_s],
                        fmt="o", ms=3, label=stat)
        ax.set_xticks(x, cls["cohort"])
        ax.set_ylabel("lag (s)")
        ax.set_title(f"head {m}: support start / median / end per recording (median, IQR)", fontsize=8)
        if m == 0:
            ax.legend(fontsize=7)
    figures.caveat_note(figure)
    return figure


def build_reach_figure(truncation: pd.DataFrame, horizon: pd.DataFrame, trajectory: pd.DataFrame, grid: np.ndarray, step_s: float) -> Any:
    """Truncation share curves per class; reach by horizon step vs the physiological prediction;
    the readouts against hours before delivery."""
    figure, axes = figures.new_figure(3, 2, height_per_row=2.3, width=10.0)
    names = list(truncation["cohort"])
    colors = _class_colors(names)
    for _, row in truncation.iterrows():
        near = [row[f"near_share_{k}"] for k in grid]
        far = [row[f"far_share_{k}"] for k in grid]
        axes[0, 0].plot(grid * step_s, near, "-o", ms=2, color=colors[row["cohort"]], label=row["cohort"])
        axes[0, 1].plot(grid * step_s, far, "-o", ms=2, color=colors[row["cohort"]], label=row["cohort"])
    axes[0, 0].set_title("keep lags <= k: share of the source gain", fontsize=8)
    axes[0, 1].set_title("keep lags >= k: share of the source gain", fontsize=8)
    for ax in axes[0]:
        ax.axhline(0.9, ls="--", color="0.5", lw=0.8)
        ax.set_xlabel("k (s)")
        ax.legend(fontsize=7)
    for name, part in horizon.groupby("cohort"):
        axes[1, 0].plot(part["horizon_step"] * step_s, part["effect_end_lag"] * step_s, color=colors.get(name, "k"), label=f"{name} end (90%)")
        axes[1, 0].plot(part["horizon_step"] * step_s, part["effect_start_lag"] * step_s, ":", color=colors.get(name, "k"))
    if not horizon.empty:
        allp = horizon[horizon["cohort"] == "all"]
        axes[1, 0].plot(allp["horizon_step"] * step_s, allp["physiological_lag"] * step_s, "--", color="0.4", label="max(d - tau, 0), d = data delay")
    axes[1, 0].set_title("lag window the forecast uses, by horizon step (solid end, dotted start)", fontsize=8)
    axes[1, 0].set_xlabel("horizon (s)")
    axes[1, 0].set_ylabel("lag (s)")
    axes[1, 0].legend(fontsize=6)
    for name, part in horizon.groupby("cohort"):
        axes[1, 1].plot(part["horizon_step"] * step_s, part["gain_nats"], color=colors.get(name, "k"), label=name)
    axes[1, 1].set_title("source gain by horizon step (nats)", fontsize=8)
    axes[1, 1].set_xlabel("horizon (s)")
    for ax, metric, title in ((axes[2, 0], "effect_end_lag", "latest lag the forecast needs (90%)"),
                              (axes[2, 1], "dominant_att_last", "dominant head: last lag in support")):
        part = trajectory[trajectory["metric"] == metric]
        for name, cell in part.groupby("group"):
            cell = cell.sort_values("bin_center_h")
            ax.plot(-cell["bin_center_h"], cell["median"] * step_s, "-o", ms=2, color=colors.get(name, "k"), label=name)
            ax.fill_between(-cell["bin_center_h"], cell["q25"] * step_s, cell["q75"] * step_s, color=colors.get(name, "k"), alpha=0.12)
        ax.set_title(f"{title}, median and IQR over recordings", fontsize=8)
        ax.set_xlabel("hours relative to delivery")
        ax.set_ylabel("lag (s)")
        ax.legend(fontsize=7)
    figures.caveat_note(figure)
    return figure


def build_events_figure(binned: pd.DataFrame, delays: pd.DataFrame, tracking: pd.DataFrame, dominant: int) -> Any:
    """Event-locked: attention on the contraction vs the null, the knockout loss and the gain by
    contraction age; the data's contraction-to-deceleration delay against its null."""
    figure, axes = figures.new_figure(2, 2, height_per_row=2.4, width=10.0)
    colors = _class_colors(list(binned["cohort"].unique()))
    for name, part in binned.groupby("cohort"):
        part = part.sort_values("age_bin_s")
        c = colors.get(name, "k")
        axes[0, 0].plot(part["age_bin_s"], part[f"ctr_mass_h{dominant}"], color=c, label=name)
        axes[0, 0].plot(part["age_bin_s"], part[f"ctr_null_h{dominant}"], ":", color=c)
        axes[0, 1].plot(part["age_bin_s"], part["ctr_loss"], color=c, label=name)
        axes[1, 0].plot(part["age_bin_s"], part["gain"], color=c, label=name)
    axes[0, 0].set_title(f"head {dominant}: attention on the contraction (dotted: fixed-filter null)", fontsize=8)
    axes[0, 1].set_title("forecast gain lost when the contraction's keys are removed (nats)", fontsize=8)
    row = tracking[(tracking["cohort"] == "all") & (tracking["head"] == dominant)]
    if len(row) and np.isfinite(row["knockout_start_age_s"].iloc[0]):
        axes[0, 1].axvspan(row["knockout_start_age_s"].iloc[0], row["knockout_end_age_s"].iloc[0], color="0.9", zorder=0)
    axes[1, 0].set_title("total source gain (nats)", fontsize=8)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("seconds since contraction onset")
        ax.legend(fontsize=7)
    for name, part in delays.groupby("cohort"):
        axes[1, 1].plot(part["delay_lo_s"] + DELAY_BIN_S / 2, part["excess"], color=colors.get(name, "k"), label=name)
    axes[1, 1].axhline(0, color="0.5", lw=0.8)
    axes[1, 1].set_title("data: decelerations per recording after a contraction peak, matched minus null", fontsize=8)
    axes[1, 1].set_xlabel("peak to nadir (s)")
    axes[1, 1].legend(fontsize=7)
    return figure


# =============================================================================
# The analysis
# =============================================================================
def _skip(reason: str) -> Dict[str, Any]:
    logger.warning(f"{ANALYSIS_DIRNAME}: skipped -- {reason}")
    return {"n_samples": None, "composition": {}, "plan": {"capped": True, "skipped": True, "reason": reason},
            "skipped": True, "reason": reason, "files": []}


def choose_indices(loader: Any, eval_config: Dict[str, Any]) -> List[int]:
    """A seeded draw of dataset indices inside ``max_hours_before_delivery``, ascending."""
    index_map = cohort.within_horizon_index(dataset_index_map(loader), eval_config.get("max_hours_before_delivery"))
    indices = np.array(sorted(index_map.values()), dtype=np.int64)
    if indices.size == 0:
        indices = np.arange(len(loader.dataset))
    cap = (eval_config.get("caps") or {}).get(CAP_KEY)
    if cap is not None and indices.size > int(cap):
        rng = np.random.default_rng(int(eval_config.get("seed", 0)) + SEED_OFFSET)
        indices = np.sort(rng.choice(indices, size=int(cap), replace=False))
    return [int(i) for i in indices]


def run_lag_extent_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect and write the lag-extent readouts. See the module docstring."""
    task, loader = getattr(context, "task", None), getattr(context, "loader", None)
    if task is None or loader is None:
        return _skip("the truncation arms re-attend through the model, and this pass built no model")
    model = task.orig_model
    if not hasattr(model, "attend_lagged_source"):
        return _skip("the model has no lag attention to truncate")
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    L = int(model.lag_attn.L)
    grid = lag_grid(L)
    step_s = float(model.geometry.decimation) / events.FS_RAW
    raw_scales = traces.raw_signal_scales(getattr(context, "config", None), loader)
    indices = choose_indices(loader, eval_config)
    batch_size = max(1, int(getattr(loader, "batch_size", None) or 1))

    frames: List[pd.DataFrame] = []
    arm_rows: List[np.ndarray] = []
    attn_sum: Dict[str, np.ndarray] = {}
    attn_count: Dict[str, int] = {}
    tau: Dict[str, List[Any]] = {"none": [], "near": [], "far": [], "count": [], "class": []}
    delays: List[Dict[str, Any]] = []
    previous_peaks: Optional[np.ndarray] = None
    task._stage = "test"
    for batch in subset_loader(loader, indices, batch_size=batch_size):
        batch = task.transfer_batch_to_device(batch, task.device, dataloader_idx=0)
        out = collect_batch(task, batch, grid=grid, raw_scales=raw_scales)
        frame, arms, (sums, counts), per_tau = reduce_batch(out, step_s=step_s)
        frames.append(frame)
        arm_rows.append(arms)
        for name, value in sums.items():
            attn_sum[name] = attn_sum.get(name, 0.0) + value
            attn_count[name] = attn_count.get(name, 0) + counts[name]
        for key in tau:
            tau[key].append(per_tau[key])
        cls = out["labels"][labels.CLASS_COLUMN]
        for i, guid in enumerate(out["guid"]):
            for d in paired_delays(out["peaks"][i], out["nadirs"][i]):
                delays.append({"guid": guid, labels.CLASS_COLUMN: cls[i], "kind": "matched", "delay_s": d})
            # Null: this segment's decelerations against the previous segment's contractions.
            if previous_peaks is not None:
                for d in paired_delays(previous_peaks, out["nadirs"][i]):
                    delays.append({"guid": guid, labels.CLASS_COLUMN: cls[i], "kind": "null", "delay_s": d})
            previous_peaks = out["peaks"][i]
        del out

    if not frames:
        return _skip("no segment was drawn")
    anchors = pd.concat(frames, ignore_index=True)
    arms = pd.DataFrame(np.concatenate(arm_rows), columns=_arm_columns(grid))
    per_tau_all = {k: (np.concatenate(v) if k != "class" else np.concatenate([np.asarray(x, dtype=object) for x in v]))
                   for k, v in tau.items()}
    attn_by_class = {name: attn_sum[name] / max(attn_count[name], 1) for name in attn_sum}
    pooled_profile = attn_by_class["all"]
    n_heads = int(pooled_profile.shape[0])
    anchors.to_parquet(directory / ANCHORS_FILENAME, index=False)
    np.savez_compressed(directory / ARMS_FILENAME, arms=arms.to_numpy(np.float32), columns=np.array(arms.columns),
                        grid=grid, **{f"attention_{k}": v for k, v in attn_by_class.items()})

    heads = head_table(anchors, n_heads, step_s)
    heads.to_csv(directory / HEADS_FILENAME, index=False)
    dominant = int(heads[heads["cohort"] == "all"].sort_values("kl_share").iloc[-1]["head"])

    values = ["gain", "kl", f"att_last_h{dominant}", f"att_q90_h{dominant}", "klmap_q10", "klmap_q90", "ctr_loss"]
    per_recording = per_group_readouts(anchors, arms, ["guid", labels.CLASS_COLUMN, labels.SUBGROUP_COLUMN], values, grid)
    per_recording.to_csv(directory / PER_RECORDING_FILENAME, index=False)

    trunc_rows = []
    for name, sel in [("all", np.ones(len(anchors), bool))] + [
            (str(c), (anchors[labels.CLASS_COLUMN] == c).to_numpy()) for c in labels.ordered_groups(anchors[labels.CLASS_COLUMN].dropna().unique(), labels.CLASS_COLUMN)]:
        mean = arms[sel].mean(0)
        near = np.array([mean[f"near_{k}"] for k in grid])
        far = np.array([mean[f"far_{k}"] for k in grid])
        lags = effect_lags(mean["none"], near, far, grid)
        row = {"cohort": name, "n_anchors": int(sel.sum()), "gain_none": float(mean["none"]), "gain_full": float(near[-1])}
        row.update({f"near_share_{k}": float(v) for k, v in zip(grid, lags["near_share"])})
        row.update({f"far_share_{k}": float(v) for k, v in zip(grid, lags["far_share"])})
        row.update({key: float(lags[key]) for key in ("effect_start_lag", "effect_mid_lag", "effect_end_lag")})
        trunc_rows.append(row)
    truncation = pd.DataFrame(trunc_rows)
    truncation.to_csv(directory / TRUNCATION_FILENAME, index=False)

    delay_hist, delay_summary = delay_table(delays)
    delay_hist.to_csv(directory / DELAYS_FILENAME, index=False)
    d_med = delay_summary.get("all", {}).get("excess_median_s", np.nan)
    horizon = horizon_table(per_tau_all, grid, delay_steps_median=d_med / step_s if np.isfinite(d_med) else np.nan)
    horizon.to_csv(directory / HORIZON_FILENAME, index=False)

    binned, tracking = event_locked(anchors, n_heads, pooled_profile, step_s=step_s)
    binned.to_csv(directory / EVENT_FILENAME, index=False)
    tracking.to_csv(directory / TRACKING_FILENAME, index=False)

    _per_window, trajectory, tests = trajectories(anchors, arms, grid, dominant)
    trajectory.to_csv(directory / TRAJECTORY_FILENAME, index=False)
    tests.to_csv(directory / SIGNIFICANCE_FILENAME, index=False)

    written = []
    for name, builder in (
        ("lag_extent_heads", lambda: build_heads_figure(anchors, attn_by_class, heads, step_s)),
        ("lag_extent_reach", lambda: build_reach_figure(truncation, horizon, trajectory, grid, step_s)),
        ("lag_extent_events", lambda: build_events_figure(binned, delay_hist, tracking, dominant)),
    ):
        try:
            written.append(Path(figures.render_figure(builder(), directory / name)).name)
        except Exception as error:  # noqa: BLE001 - one figure is not worth the tables
            logger.warning(f"{ANALYSIS_DIRNAME}: figure {name} failed: {error}")

    all_trunc = truncation[truncation["cohort"] == "all"].iloc[0]
    all_track = tracking[(tracking["cohort"] == "all") & (tracking["head"] == dominant)].iloc[0]
    horizon_all = horizon[horizon["cohort"] == "all"]
    headline = {
        "dominant_head": dominant,
        "dominant_head_kl_share": float(heads[(heads["cohort"] == "all") & (heads["head"] == dominant)]["kl_share"].iloc[0]),
        "lag0_gain_share": float(all_trunc[f"near_share_{grid[0]}"]),
        "effect_start_s": float(all_trunc["effect_start_lag"]) * step_s,
        "effect_mid_s": float(all_trunc["effect_mid_lag"]) * step_s,
        "effect_end_s": float(all_trunc["effect_end_lag"]) * step_s,
        "end_lag_slope_vs_tau": float(horizon_all["end_lag_slope_vs_tau"].iloc[0]) if len(horizon_all) else np.nan,
        "dominant_tracking_slope": float(all_track["centroid_slope_vs_age"]),
        "dominant_tracking_index": float(all_track["tracking_index"]),
        "contraction_knockout_start_s": float(all_track["knockout_start_age_s"]),
        "contraction_knockout_end_s": float(all_track["knockout_end_age_s"]),
        "data_delay_excess_median_s": float(d_med) if np.isfinite(d_med) else None,
        "n_significant_windows": {m: int(tests[(tests["metric"] == m)]["significant"].fillna(False).astype(bool).sum())
                                  for m in TESTED} if not tests.empty else {},
    }
    logger.info(f"{ANALYSIS_DIRNAME}: {headline}")
    by_class = anchors.groupby(labels.CLASS_COLUMN)["guid"].nunique().to_dict()
    n_segments = int(anchors[["guid", "epoch"]].drop_duplicates().shape[0])
    return {
        "n_samples": n_segments,
        "composition": {"n_recordings": int(anchors["guid"].nunique()),
                        "n_recordings_by_class": {str(k): int(v) for k, v in by_class.items()},
                        "n_anchors": int(len(anchors))},
        # Always capped: the segments are a draw from the dataset, not the collected population.
        "plan": {"capped": True,
                 "cap": (eval_config.get("caps") or {}).get(CAP_KEY), "lag_grid": [int(k) for k in grid],
                 "n_arms": int(arms.shape[1]), "support_eps": SUPPORT_EPS, "min_effect_nats": MIN_EFFECT_NATS,
                 "decode": "latent mean", "removal": "keys removed through the model's running-gap mask; entmax renormalises"},
        "headline": headline,
        "delays": delay_summary,
        "caveat": lag_axis.GROUP_DELAY_NOTE,
        "grouped_frames": [grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, ["gain", "effect_end_lag", f"att_last_h{dominant}"])],
        "files": [ANCHORS_FILENAME, ARMS_FILENAME, PER_RECORDING_FILENAME, HEADS_FILENAME, TRUNCATION_FILENAME,
                  HORIZON_FILENAME, TRAJECTORY_FILENAME, SIGNIFICANCE_FILENAME, EVENT_FILENAME, TRACKING_FILENAME,
                  DELAYS_FILENAME, *written],
    }
