"""Threshold policies chosen on validation, and the basis populations they are chosen on (SPEC §11.3).

Scores are calibrated logits; the decision rule is **alarm iff score > thr** (strict). Every basis
reduces to one score vector per class, and every FPR-cap policy to :func:`fpr_threshold` on the
negatives of that vector. There is no binary search, no epoch filling, and no silent fallback to
0.5 (A1, A7, A8): a policy that cannot be computed raises.

The pilot's ``select_threshold`` (``latent_pilot/evaluate.py:1337``) is ported, not imported: that
module imports torch and the VAE nets at load time, and its predicate is ``>=``. The tie rule is
kept (the lowest threshold among the maxima).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import binom

BASES = ("guid_final", "instantaneous", "committed_cumulative", "committed_overall", "segment")


def np_order_k(n0: int, alpha: float, delta: float) -> int:
    """Neyman-Pearson umbrella order statistic (Tong, Feng & Li 2018), 1-based.

    ``k* = min{k : sum_{j>=k} C(n0,j)(1-alpha)^j alpha^(n0-j) <= delta}``; ``thr = v_(k*)`` then
    gives P(population FPR > alpha) <= delta.

    Raises:
        ValueError: If ``n0 < ceil(log delta / log(1 - alpha))`` (no k exists).
    """
    n_min = math.ceil(math.log(delta) / math.log(1.0 - alpha))
    k = np.arange(1, n0 + 1)
    ok = k[binom.sf(k - 1, n0, 1.0 - alpha) <= delta]
    if n0 < n_min or ok.size == 0:
        raise ValueError(
            f"NP umbrella needs n0 >= {n_min} negatives for alpha={alpha}, delta={delta}; got {n0}"
        )
    return int(ok[0])


def _clean(scores: Any) -> np.ndarray:
    """Float vector; NaN is a failure, never a silent 'no alarm' (A7)."""
    v = np.asarray(scores, dtype=np.float64).reshape(-1)
    if np.isnan(v).any():
        raise ValueError(f"{int(np.isnan(v).sum())} NaN score(s) in a threshold basis")
    return v


def fpr_threshold(
    neg_scores: Any, alpha: float, method: str, delta: float = 0.05, allow_fallback: bool = False
) -> Dict[str, Any]:
    """FPR-cap threshold on the negative scores of a basis.

    ``empirical``: the smallest candidate ``thr`` in the scores with ``#{v > thr}/n0 <= alpha``
    (maximal sensitivity under the cap). ``np_umbrella``: ``thr = v_(k*)``. Ties at ``thr`` are
    not alarmed (strict ``>``); ``tie_frac`` is the share of negatives equal to ``thr``. ``-inf``
    entries (unmonitored negatives) are legal and never alarm.

    Returns:
        ``{threshold, method, alpha, delta, k, n_neg, tie_frac, fallback}``.

    Raises:
        ValueError: On empty or NaN scores, alpha outside (0, 1), an unknown method, or too few
            negatives for NP without ``allow_fallback``.
    """
    v = np.sort(_clean(neg_scores))
    if v.size == 0 or not 0.0 < alpha < 1.0:
        raise ValueError(f"fpr_threshold needs negatives and 0 < alpha < 1; got n0={v.size}, alpha={alpha}")
    requested, k, fallback = method, None, False
    if method == "np_umbrella":
        try:
            k = np_order_k(v.size, alpha, delta)
        except ValueError:
            if not allow_fallback:
                raise
            method, fallback = "empirical", True
    if method == "np_umbrella":
        thr = v[k - 1]
    elif method == "empirical":
        n_above = v.size - np.searchsorted(v, v, side="right")
        thr = v[np.argmax(n_above <= math.floor(alpha * v.size + 1e-9))]  # v[-1] always qualifies
    else:
        raise ValueError(f"unknown fpr_cap method {method!r}")
    return {
        "threshold": float(thr), "method": method, "alpha": float(alpha),
        "delta": float(delta) if requested == "np_umbrella" else None, "k": k,
        "n_neg": int(v.size), "tie_frac": float(np.mean(v == thr)), "fallback": fallback,
    }


def _strict_roc(y: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(candidates, tpr, fpr) of the rule ``s > thr`` at every distinct split, incl. ``-inf``."""
    if not 0 < y.sum() < y.size:
        raise ValueError(f"threshold policy needs both classes; got {int(y.sum())} of {y.size} positive")
    cand = np.unique(np.r_[-np.inf, s])
    pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    tpr = 1.0 - np.searchsorted(pos, cand, side="right") / pos.size
    fpr = 1.0 - np.searchsorted(neg, cand, side="right") / neg.size
    return cand, tpr, fpr


def youden(y: Any, score: Any) -> Dict[str, Any]:
    """argmax (TPR - FPR); ties go to the lower threshold (pilot semantics)."""
    y, s = np.asarray(y, dtype=np.int64).reshape(-1), _clean(score)
    cand, tpr, fpr = _strict_roc(y, s)
    j = tpr - fpr
    thr = cand[j >= j.max() - 1e-12].min()
    return {"threshold": float(thr), "method": "youden", "tie_frac": float(np.mean(s == thr))}


def sens_target(y: Any, score: Any, beta: float, allow_fallback: bool = False) -> Dict[str, Any]:
    """Smallest-FPR threshold with TPR >= beta; among equal FPR the lower (more sensitive) one.

    A positive scored ``-inf`` (not yet monitored on a ``committed_overall`` basis) never alarms, so TPR tops out at
    the share of finite positives. A ``beta`` above that raises, unless ``allow_fallback``: then the most sensitive
    threshold, flagged ``fallback`` (as the NP umbrella's fallback)."""
    y, s = np.asarray(y, dtype=np.int64).reshape(-1), _clean(score)
    cand, tpr, fpr = _strict_roc(y, s)
    ok, fallback = tpr >= beta - 1e-12, False
    if not ok.any():
        if not allow_fallback:
            raise ValueError(f"sens_target beta={beta} is unreachable: at most {tpr.max():.3f} of the positives can "
                             f"alarm (the others score -inf, i.e. are not yet monitored)")
        ok, fallback = tpr >= tpr.max(), True
    thr = cand[ok & (fpr == fpr[ok].min())].min()
    return {"threshold": float(thr), "method": "sens_target", "tie_frac": float(np.mean(s == thr)),
            "fallback": fallback}


def fixed(p: float) -> Dict[str, Any]:
    """A given calibrated probability, as a logit threshold."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"fixed threshold needs 0 < p < 1; got {p}")
    return {"threshold": float(logit(p)), "method": "fixed"}


def apply_threshold(score: Any, thr: float) -> np.ndarray:
    """Alarm decisions ``score > thr``; NaN scores raise rather than read as 'no alarm'."""
    return _clean(score) > float(thr)


#: Evaluation time axes (SPEC §11.5; CONTRACT "Evaluation time axes"), and why a GUID drops off one.
AXES = ("to_delivery", "from_onset", "rel_second_stage", "position", "elapsed")
AXIS_REASON = {"from_onset": "no_tlo", "rel_second_stage": "unknown_second_stage"}


def clock(segments: pd.DataFrame, axis: str) -> pd.Series:
    """Clock of each segment on ``axis`` (increasing with real time; NaN = the GUID is ineligible).

    ``to_delivery`` ``t_end_s/3600`` (h, negative); ``from_onset`` ``tlo_end_h``; ``rel_second_stage``
    ``ss_rel_h`` (at segment start) plus the segment's duration; ``position`` ``seg_pos + 1``;
    ``elapsed`` hours since the GUID's first retained segment end. Threshold selection and every
    time-resolved metric read this one function, so they cannot disagree.
    """
    s = segments
    if axis == "to_delivery":
        c = s["t_end_s"] / 3600.0
    elif axis == "from_onset":
        c = s["tlo_end_h"]
    elif axis == "rel_second_stage":
        c = s["ss_rel_h"] + (s["t_end_s"] - s["epoch_s"]) / 3600.0
    elif axis == "position":
        c = s["seg_pos"] + 1.0
    elif axis == "elapsed":
        c = (s["t_end_s"] - s.groupby("guid")["t_end_s"].transform("min")) / 3600.0
    else:
        raise ValueError(f"unknown axis {axis!r}; expected one of {AXES}")
    return c.astype(np.float64)


def snapshot_window(axis: str, *, bin_h: float, staleness_h: Optional[float]) -> float:
    """Width ``w`` of the instantaneous population at a point ``c*`` of ``axis`` (§11.5.1): a GUID's snapshot is its
    last segment with clock <= c*, kept iff its clock > c* - w. On ``to_delivery`` a point is a checkpoint, so w is
    ``eval.snapshot_max_staleness_h``; the other axes are read on the bin grid, w = ``bin_h`` (``position``: 1).
    Threshold selection (:func:`basis_frame`) and the evaluation's checkpoints (``metrics._points``) both take w here,
    so an instantaneous threshold is chosen on the population it is reported on."""
    if axis == "position":
        return 1.0
    if axis != "to_delivery":
        return float(bin_h)
    if staleness_h is None:
        raise ValueError("an instantaneous point on to_delivery is a checkpoint: pass eval.snapshot_max_staleness_h "
                         "(staleness_h), the window its evaluation uses (§11.5.1)")
    return float(staleness_h)


def cstar(at: Any, axis: str = "to_delivery") -> float:
    """Evaluation point ``c*`` of ``at``: ``+inf`` at ``end``; ``-at`` on ``to_delivery`` (hours
    before delivery), ``at`` itself on every other axis (hours, or a position)."""
    return math.inf if at == "end" else -float(at) if axis == "to_delivery" else float(at)


def latch_state(exceed: np.ndarray, gi: np.ndarray, k: int = 1, n: int = 1) -> np.ndarray:
    """Latched alarm per row; rows sorted by GUID code ``gi``, ``exceed`` = raw ``s > thr`` (rows on its last axis).

    The alarm fires at a row once ``>= k`` of the GUID's last ``n`` rows exceed, and holds from then
    on. ``k = n = 1`` is the latch: alarmed by row i <=> some row <= i exceeded <=> ``r(i) > thr``.
    """
    idx = np.arange(gi.size)
    first = np.flatnonzero(np.r_[True, gi[1:] != gi[:-1]]) if gi.size else np.zeros(0, np.int64)
    start = np.repeat(first, np.diff(np.r_[first, gi.size]))
    pad = np.zeros(exceed.shape[:-1] + (1,), np.int64)
    C = np.concatenate([pad, np.cumsum(exceed, axis=-1)], axis=-1)
    fire = C[..., idx + 1] - C[..., np.maximum(idx + 1 - n, start)] >= k
    F = np.concatenate([pad, np.cumsum(fire, axis=-1)], axis=-1)
    return F[..., idx + 1] - F[..., start] > 0


def _online(segments: pd.DataFrame, axis: str = "to_delivery") -> pd.DataFrame:
    """Segments sorted by (guid, seg_pos) with clock ``c`` (:func:`clock`), online score ``s`` and running max ``r``.

    ``s`` is ``logit_online_cal``; a model without an online score (column absent or all NaN) uses
    the running max of ``logit_seg_cal`` (§11.2 running aggregator). Partial NaN raises.
    """
    seg = segments.sort_values(["guid", "seg_pos"], kind="stable")
    if "logit_online_cal" in seg and seg["logit_online_cal"].notna().any():
        s = seg["logit_online_cal"]
    else:
        s = pd.Series(_clean(seg["logit_seg_cal"]), index=seg.index).groupby(seg["guid"]).cummax()
    s = pd.Series(_clean(s), index=seg.index)
    return seg.assign(c=clock(seg, axis), s=s, r=s.groupby(seg["guid"]).cummax())


def axis_eligible(seg: pd.DataFrame, axis: str) -> Tuple[pd.Series, Dict[str, int]]:
    """Row mask of the GUIDs with a clock on every segment (``seg`` from :func:`_online`), and the
    excluded GUID count by reason (empty when none is excluded; L14)."""
    ok = seg["c"].notna().groupby(seg["guid"]).transform("all").astype(bool)
    n = int(seg.loc[~ok, "guid"].nunique())
    return ok, ({AXIS_REASON.get(axis, "missing_clock"): n} if n else {})


def basis_frame(
    segments: pd.DataFrame, guids: pd.DataFrame, basis: str, *, at: Any = "end", bin_h: float = 0.5,
    axis: str = "to_delivery", staleness_h: Optional[float] = None,
) -> pd.DataFrame:
    """``(unit, guid, y, score)`` of the basis population for one model/seed/fold/split group (§11.3).

    ``unit`` is the counting unit: the GUID, or ``guid@seg_pos`` for the ``segment`` basis.
    Clock: :func:`clock` on ``axis``; ``at='end'`` counts every segment, otherwise ``c* = cstar(at, axis)``
    (``to_delivery``: ``at=h`` is ``c* = -h``). GUIDs without a clock on ``axis`` (no TLO, unknown
    second stage) are excluded from the time bases; ``frame.attrs['excluded']`` counts them by reason.

    * ``guid_final``: ``score_final_cal`` of every GUID in ``guids``.
    * ``committed_cumulative``: ``r_g(n*)`` of GUIDs whose first segment ends by ``c*``.
    * ``committed_overall``: as cumulative, plus ``-inf`` for GUIDs not yet monitored.
    * ``instantaneous``: ``s_g`` at the last segment in ``(c* - w, c*]``, one per GUID present (w:
      :func:`snapshot_window`, the evaluation's own window); at ``end``, each GUID's last segment.
    * ``segment``: ``logit_seg_cal`` of every ``in_eval_window`` segment (``at`` must be ``end``).
    """
    def frame(unit: Any, guid: Any, y: Any, score: Any, excluded: Optional[Dict[str, int]] = None) -> pd.DataFrame:
        f = pd.DataFrame({"unit": pd.Series(np.asarray(unit), dtype=str),
                          "guid": pd.Series(np.asarray(guid), dtype=str),
                          "y": np.asarray(y, dtype=np.int64), "score": np.asarray(score, dtype=np.float64)})
        f.attrs["excluded"] = excluded or {}
        return f

    if basis == "guid_final":
        return frame(guids["guid"], guids["guid"], guids["y"], _clean(guids["score_final_cal"]))
    if basis == "segment":
        if at != "end":
            raise ValueError("segment basis uses the eval window; `at` must be 'end'")
        win = segments[segments["in_eval_window"].astype(bool)]
        return frame(win["guid"].astype(str) + "@" + win["seg_pos"].astype(str), win["guid"], win["y"],
                     _clean(win["logit_seg_cal"]))
    if basis not in BASES:
        raise ValueError(f"unknown basis {basis!r}; expected one of {BASES}")
    seg = _online(segments, axis)
    ok, excluded = axis_eligible(seg, axis)
    seg = seg[ok]
    c = cstar(at, axis)
    if basis == "instantaneous":
        snap = seg if at == "end" else seg[(seg["c"] <= c) & (
            seg["c"] > c - snapshot_window(axis, bin_h=bin_h, staleness_h=staleness_h))]
        last = snap.drop_duplicates("guid", keep="last")
        return frame(last["guid"], last["guid"], last["y"], last["s"], excluded)
    last = seg[seg["c"] <= c].drop_duplicates("guid", keep="last").set_index("guid")
    if basis == "committed_cumulative":
        return frame(last.index, last.index, last["y"], last["r"], excluded)
    every = seg.drop_duplicates("guid").set_index("guid")["y"]
    return frame(every.index, every.index, every, last["r"].reindex(every.index, fill_value=-np.inf), excluded)


def basis_scores(
    segments: pd.DataFrame, guids: pd.DataFrame, basis: str, *, at: Any = "end", bin_h: float = 0.5,
    axis: str = "to_delivery", staleness_h: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(y, score)`` arrays of :func:`basis_frame`."""
    f = basis_frame(segments, guids, basis, at=at, bin_h=bin_h, axis=axis, staleness_h=staleness_h)
    return f["y"].to_numpy(np.int64), f["score"].to_numpy(np.float64)


def drop_last_minutes(segments: pd.DataFrame, minutes: float) -> pd.DataFrame:
    """Segments ending more than ``minutes`` before delivery (``eval.exclude_last_min``, §11.5.1).

    The one filter threshold selection and every evaluation population share, so the population a
    threshold is chosen on is the population it is evaluated on.
    """
    return segments if not minutes else segments[segments["t_end_s"] <= -60.0 * float(minutes)]


def guid_level_policies(policies: Sequence[Mapping[str, Any]]) -> Tuple[list, list]:
    """``(policies, skipped_ids)`` for a model with no causal per-position score (the shortcut, a
    non-causal sequence model): the ``at: end`` policies on basis ``guid_final`` (identical at end),
    the others skipped (CONTRACT)."""
    kept = [dict(p) | {"basis": "guid_final"} for p in policies if p.get("at", "end") == "end"]
    return kept, [p["id"] for p in policies if p.get("at", "end") != "end"]


def policy_threshold(pol: Mapping[str, Any], y: np.ndarray, s: np.ndarray) -> Dict[str, Any]:
    """One policy's threshold record on a basis vector (``y`` 0/1, ``s`` scores): the dispatch of
    :func:`select_thresholds`, shared with the refit bootstrap (``metrics.refit_rows``)."""
    kind = pol["policy"]
    if kind == "fpr_cap":
        return fpr_threshold(s[y == 0], pol["alpha"], pol["method"], pol.get("delta", 0.05),
                             pol.get("allow_fallback", False))
    if kind == "youden":
        return youden(y, s)
    if kind == "sens_target":
        return sens_target(y, s, pol["beta"], pol.get("allow_fallback", False))
    if kind == "fixed":
        return fixed(pol["value"]) | {"tie_frac": float(np.mean(s == logit(pol["value"])))}
    raise ValueError(f"policy {pol['id']}: unknown policy {kind!r}")


def select_thresholds(
    val_segments: pd.DataFrame, val_guids: pd.DataFrame, policies: Sequence[Mapping[str, Any]], *, bin_h: float,
    exclude_last_min: float = 0.0, staleness_h: Optional[float] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Every policy's threshold on validation, in the CONTRACT ``thresholds.json`` shape.

    A policy is ``{id, policy: fpr_cap|youden|sens_target|fixed, basis, axis, at, alpha, method,
    delta, allow_fallback, beta (sens_target), value (fixed)}``. Level is ``segment`` for the segment
    basis, ``guid`` otherwise. ``val_sens/val_fpr/val_spec`` are on the same basis population.
    ``axis`` defaults to ``to_delivery``; when it excludes GUIDs (no TLO / unknown second stage) the
    record gains ``excluded: {reason: n_guids}``. ``exclude_last_min`` drops the last minutes of every
    segment basis first (:func:`drop_last_minutes`). ``staleness_h`` (``eval.snapshot_max_staleness_h``) is the
    instantaneous window at a ``to_delivery`` point (:func:`snapshot_window`); such a policy without it raises.
    """
    val_segments = drop_last_minutes(val_segments, exclude_last_min)
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for pol in policies:
        basis, at, axis = pol.get("basis", "committed_overall"), pol.get("at", "end"), pol.get("axis", "to_delivery")
        f = basis_frame(val_segments, val_guids, basis, at=at, bin_h=bin_h, axis=axis, staleness_h=staleness_h)
        y, s = f["y"].to_numpy(np.int64), f["score"].to_numpy(np.float64)
        if pol["policy"] == "fpr_cap" and basis == "segment" and pol["method"] != "empirical":
            raise ValueError(f"policy {pol['id']}: segment basis allows only method 'empirical'")
        res = policy_threshold(pol, y, s)
        alarm = s > res["threshold"]
        fpr = float(alarm[y == 0].mean()) if (y == 0).any() else math.nan
        out.setdefault("segment" if basis == "segment" else "guid", {})[pol["id"]] = {
            "threshold": res["threshold"], "basis": basis, "axis": axis, "at": at,
            "alpha": res.get("alpha"), "delta": res.get("delta"), "method": res["method"],
            "k": res.get("k"), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
            "val_sens": float(alarm[y == 1].mean()) if (y == 1).any() else math.nan,
            "val_fpr": fpr, "val_spec": 1.0 - fpr, "tie_frac": res["tie_frac"],
            "fallback": res.get("fallback", False),
        } | ({"excluded": f.attrs["excluded"]} if f.attrs["excluded"] else {})
    return out


def ovr_thresholds(
    val_segments: pd.DataFrame, val_guids: pd.DataFrame, policies: Sequence[Mapping[str, Any]], *, bin_h: float,
    exclude_last_min: float = 0.0, staleness_h: Optional[float] = None,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """Per-class one-vs-rest thresholds of a 3-class model (§11.3 T5): ``{str(k): select_thresholds(...)}`` for class
    k = 0 healthy, 1 acidosis, 2 hie (``class_code - 1``), every policy on its own basis.

    Class k's rows are :func:`ovr_view`'s. A model with no causal per-segment class probability (a non-causal
    sequence model) is thresholded like the shortcut: :func:`guid_level_policies`, the others recorded as skipped, as
    ``run.lock_unit`` does for its binary score. Written as ``thresholds.json["ovr"]``. A policy that cannot be
    computed raises (no fallback but the policy's own).
    """
    segs = [ovr_view(val_segments, k, OVR_SEGMENT_SCORES) for k in range(3)]
    skipped: list = []
    if all(s[list(OVR_SEGMENT_SCORES)].isna().all(axis=None) for s in segs):
        policies, skipped = guid_level_policies(policies)
    out = {}
    for k, s in enumerate(segs):
        out[str(k)] = select_thresholds(s, ovr_view(val_guids, k, ("score_final_cal",)), policies, bin_h=bin_h,
                                        exclude_last_min=exclude_last_min, staleness_h=staleness_h)
        for pid in skipped:
            out[str(k)].setdefault("guid", {})[pid] = {"skipped": "guid-level model"}
    return out


#: The segment-row scores a one-vs-rest view replaces (:func:`ovr_view`).
OVR_SEGMENT_SCORES = ("logit_online_cal", "logit_seg_cal")


def ovr_view(frame: pd.DataFrame, k: int, cols: Sequence[str]) -> pd.DataFrame:
    """Class k's one-vs-rest view of prediction rows (§11.3 T5), shared by selection and evaluation: target
    ``class_code - 1 == k``, and ``cols`` (where scored) replaced by ``logit(p_c<k>_cal)``: per GUID
    ``score_final_cal``, per segment row :data:`OVR_SEGMENT_SCORES`. A segment row's ``p_c<k>_cal`` is the class probability after segment n
    (the online output; in segment scope segment n's own, whose running max the committed bases latch), so a row with
    no causal online score (``logit_online_cal`` NaN: a non-causal sequence model, whose positions saw later segments,
    §9.1) gets no OvR score at all."""
    x = logit(frame[f"p_c{k}_cal"].to_numpy(np.float64))
    if "logit_online_cal" in cols:
        x = np.where(frame["logit_online_cal"].isna(), np.nan, x)
    return frame.assign(y=(frame["class_code"] - 1 == k).astype(np.int64),
                        **{c: np.where(frame[c].isna(), np.nan, x) for c in cols})
