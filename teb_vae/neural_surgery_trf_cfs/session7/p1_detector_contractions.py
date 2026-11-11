r"""Phase 1: audit of the shipped contraction detector, as the training reward runs it.

The training reward is $w_k = 1$ for $o \le k \le o + 15$ steps after every detected onset $o$,
from :func:`teb_vae.lag_attn_cfs.eval.events.detect_contractions` on the bridged ``up`` under the
joint validity (step weight above 0 and no UA quality bit in 2299). This script measures that
detector on recordings of split code 0 and 2 only, and writes ``out/detectors/contractions.json``
and panel figures.

Measured, each with a recording-level bootstrap (2000 resamples, 95% percentile interval):

* **Rates.** Detections per 10 min of eligible time (the $21\,$min a peak may occupy after the
  $30\,$s edge rule), by cohort and by hours before delivery; durations, onset-to-peak times and
  peak-to-peak intervals, with the share of implausible values.
* **Overlap consistency.** Neighbouring segments share $120\,$s of identical signal, which leaves
  no interior once a $60\,$s margin is taken from both edges. The script therefore stitches each
  run of neighbours into one exact trace and re-cuts $22\,$min windows at offsets of $330\,$s and
  $660\,$s. Detections whose peak lies at least $60\,$s inside both windows are matched by peak
  within $\pm 30\,$s. Agreement is the Dice share $2M / (n_1 + n_2)$. The native $120\,$s overlap
  is reported too, with a $30\,$s margin (the detector's own edge rule).
* **Reference agreement.** Recall and precision against the detector-free reference of
  :mod:`detectors7` (peak match within $\pm 30\,$s), the onset error
  $o_{\mathrm{shipped}} - o_{\mathrm{ref}}$, and the cause of every miss.
* **Validity rule.** The share of pre-drop detections whose span touches an invalid sample, the
  invalid time they hold, and the reward mass the rule removes.
* **Where the UA mask fires.** The invalid rate by contraction phase (before, rise, apex, fall,
  between contractions), to test whether the mask flags contraction apexes.
* **Onset against the reward window.** Onset-to-peak time, the share of peaks inside the
  $[o, o + 15]$ step window, and the share of valid steps at weight 1.

Edit :data:`RUN_ARGS` and run this file from the IDE, or pass ``--codes``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import warnings

import numpy as np
from scipy.signal._peak_finding_utils import PeakPropertyWarning

warnings.simplefilter("ignore", PeakPropertyWarning)

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs.session7 import detectors7 as D  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import p1_detector_io as IO  # noqa: E402
from teb_vae.lag_attn_cfs.eval import events as EV  # noqa: E402

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {"codes": [0, 2], "n_panels": 24, "seed": 7}

FS = 4
EDGE = int(EV.CONTRACTION_EDGE_S * FS)
TOL = 30 * FS
REACH = 15


# =============================================================================
# A stage-by-stage replica of the shipped detector, for miss attribution
# =============================================================================
def shipped_stages(up: np.ndarray, valid: Optional[np.ndarray]) -> Dict[str, Any]:
    r"""Replay :func:`detect_contractions` and keep every intermediate peak set.

    Stages: ``fp`` (``find_peaks`` output), ``edge`` (after the $30\,$s edge rule), ``rise``
    (after the onset-rise rule) and ``final`` (after the gap drop). Also the peaks the
    prominence rule alone removes (``fp_noprom``: ``find_peaks`` with distance and width only)
    and those width removes (``fp_nowidth``: distance and prominence only).

    Args:
        up: The UA trace.
        valid: The joint validity.

    Returns:
        The stage arrays, plus ``onset``/``end`` of the ``rise`` stage and ``sigma``.
    """
    from scipy.signal import find_peaks

    smooth = EV._smooth(EV.fill_gaps(up, valid), window=max(int(round(EV.CONTRACTION_SMOOTH_S * FS)), 5))
    sigma = float(np.nanstd(smooth)) or 1.0
    dist = int(EV.CONTRACTION_MIN_DISTANCE_S * FS)
    width = int(EV.CONTRACTION_MIN_WIDTH_S * FS)
    prom = EV.CONTRACTION_PROMINENCE_SIGMA * sigma
    peaks, props = find_peaks(smooth, distance=dist, prominence=prom, width=width)
    noprom, _ = find_peaks(smooth, distance=dist, width=width)
    nowidth, _ = find_peaks(smooth, distance=dist, prominence=prom)
    anyp, _ = find_peaks(smooth, prominence=prom)
    proms = props["prominences"]
    keep = (peaks >= EDGE) & (peaks < up.size - EDGE)
    edge_p, edge_prom = peaks[keep], proms[keep]
    if edge_p.size:
        on, en = EV._event_flanks(smooth, edge_p, edge_prom, walk_seconds=EV.CONTRACTION_WALK_S, fs=FS)
        rising = smooth[edge_p] - smooth[on] >= EV.ONSET_MIN_RISE_FRAC * edge_prom
    else:
        on = en = edge_p
        rising = np.zeros(0, dtype=bool)
    rise_p, rise_on, rise_en = edge_p[rising], on[rising], en[rising]
    final = EV.drop_events_overlapping_gaps({"onset_raw": rise_on, "peak_raw": rise_p, "end_raw": rise_en},
                                            valid, span_keys=("onset_raw", "end_raw"))
    return {"fp": peaks, "fp_noprom": noprom, "fp_nowidth": nowidth, "fp_nodist": anyp, "edge": edge_p,
            "rise": rise_p, "rise_on": rise_on, "rise_en": rise_en, "final": final, "sigma": sigma,
            "smooth": smooth}


def near(x: np.ndarray, p: int, tol: int = TOL) -> bool:
    """True when some element of ``x`` lies within ``tol`` of ``p``."""
    return bool(np.asarray(x).size) and bool(np.min(np.abs(np.asarray(x) - p)) <= tol)


def miss_reason(st: Dict[str, Any], p: int) -> str:
    """Name the first shipped stage that lost a reference peak at ``p``.

    Args:
        st: Output of :func:`shipped_stages`.
        p: Reference peak index.

    Returns:
        One of ``'prominence_sigma'``, ``'width'``, ``'distance_merge'``, ``'not_a_peak'``,
        ``'edge'``, ``'onset_rise_rule'``, ``'gap_drop'``, ``'displaced'``.
    """
    if near(st["final"]["peak_raw"], p, 2 * TOL):
        return "displaced"
    if near(st["rise"], p):
        return "gap_drop"
    if near(st["edge"], p):
        return "onset_rise_rule"
    if near(st["fp"], p):
        return "edge"
    if near(st["fp_noprom"], p):
        return "prominence_sigma"
    if near(st["fp_nowidth"], p):
        return "width"
    if near(st["fp_nodist"], p):
        return "distance_merge"
    return "not_a_peak"


# =============================================================================
# Per-segment collection
# =============================================================================
def collect(seg: IO.Segments) -> Dict[str, Any]:
    """Run the shipped detector, its pre-drop variants and the reference on every segment.

    Args:
        seg: The loaded rows.

    Returns:
        Per-segment lists of event dictionaries.
    """
    out: Dict[str, List[Any]] = {k: [] for k in ("ship", "ship_off", "pre", "st", "ref", "ref12", "ref30")}
    for i in range(len(seg)):
        up, v = seg.up[i], seg.up_valid[i]
        st = shipped_stages(up, v)
        ship = IO.shipped_contractions(up, v)
        assert np.array_equal(st["final"]["peak_raw"], ship["peak_raw"]), "replica diverged"
        out["st"].append(st)
        out["ship"].append(ship)
        out["ship_off"].append(IO.shipped_contractions(up, seg.step_valid[i]))
        out["pre"].append(EV.detect_contractions(EV.fill_gaps(up, v), fs=FS, valid=None))
        out["ref"].append(D.reference_contractions(up, v))
        out["ref12"].append(D.reference_contractions(up, v, theta=12.0))
        out["ref30"].append(D.reference_contractions(up, v, theta=30.0))
    return out


def eligible(ev: Dict[str, np.ndarray], n: int = IO.SEG_RAW, need_valid: bool = True) -> np.ndarray:
    """Mask of reference events the shipped detector could report: peak off the edges, span valid."""
    p = ev["peak_raw"]
    m = (p >= EDGE) & (p < n - EDGE)
    if need_valid:
        m &= ev["invalid_in_span"] == 0
    return m


# =============================================================================
# Analyses
# =============================================================================
def hour_bin(h: float) -> str:
    """Label of the hours-before-delivery bin of ``h``."""
    edges = IO.HOUR_BINS
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= h < hi:
            return f"{lo:g}-{hi:g}h"
    return f">{edges[-1]:g}h"


def rates(seg: IO.Segments, events: Sequence[Dict[str, np.ndarray]], ref: bool = False,
          rows: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Detections per 10 min of eligible time, overall, by cohort and by hours before delivery."""
    rows = np.arange(len(seg)) if rows is None else rows
    n = np.array([((events[i]["peak_raw"] >= EDGE) & (events[i]["peak_raw"] < IO.SEG_RAW - EDGE)).sum()
                  if ref else events[i]["peak_raw"].size for i in rows], dtype=float)
    den = np.full(n.size, (IO.SEG_S - 2 * EV.CONTRACTION_EDGE_S) / 600.0)
    rec = seg.guid[rows]
    res: Dict[str, Any] = {"all": IO.ratio_bootstrap(rec, n, den)}
    for c in np.unique(seg.cohort[rows]):
        m = seg.cohort[rows] == c
        res[f"cohort={c}"] = IO.ratio_bootstrap(rec[m], n[m], den[m])
    hb = np.array([hour_bin(h) for h in seg.hours_before()[rows]])
    for b in sorted(set(hb), key=lambda s: float(s.split("-")[0].lstrip(">"))):
        m = hb == b
        res[f"hours={b}"] = IO.ratio_bootstrap(rec[m], n[m], den[m])
    per10 = n / den
    res["share_segments_rate_gt5"] = IO.ratio_bootstrap(rec, per10 > 5, np.ones(n.size))
    res["share_segments_rate_lt1"] = IO.ratio_bootstrap(rec, per10 < 1, np.ones(n.size))
    return res


def shapes(seg: IO.Segments, events: Sequence[Dict[str, np.ndarray]], rows: np.ndarray) -> Dict[str, Any]:
    """Duration, onset-to-peak and peak-to-peak interval distributions and implausible shares."""
    dur, otp, ivl, rd, ri = [], [], [], [], []
    for i in rows:
        ev = events[i]
        d = (ev["end_raw"] - ev["onset_raw"]) / FS
        dur += d.tolist()
        otp += ((ev["peak_raw"] - ev["onset_raw"]) / FS).tolist()
        rd += [seg.guid[i]] * d.size
        p = np.sort(ev["peak_raw"])
        if p.size > 1:
            iv = np.diff(p) / FS
            ivl += iv.tolist()
            ri += [seg.guid[i]] * iv.size
    dur, otp, ivl = map(np.asarray, (dur, otp, ivl))
    q = lambda x: {f"p{k}": float(np.percentile(x, k)) for k in (5, 25, 50, 75, 95)} if x.size else {}
    return {
        "duration_s": q(dur), "onset_to_peak_s": q(otp), "peak_interval_s": q(ivl),
        "median_duration": IO.stat_bootstrap(rd, dur, IO.median),
        "median_onset_to_peak": IO.stat_bootstrap(rd, otp, IO.median),
        "median_interval": IO.stat_bootstrap(ri, ivl, IO.median),
        "share_duration_lt20": IO.stat_bootstrap(rd, dur, IO.share(lambda x: x < 20)),
        "share_duration_gt180": IO.stat_bootstrap(rd, dur, IO.share(lambda x: x > 180)),
        "share_duration_40_90": IO.stat_bootstrap(rd, dur, IO.share(lambda x: (x >= 40) & (x <= 90))),
        "share_interval_lt60": IO.stat_bootstrap(ri, ivl, IO.share(lambda x: x < 60)),
        "share_interval_lt90": IO.stat_bootstrap(ri, ivl, IO.share(lambda x: x < 90)),
    }


def reference_agreement(seg: IO.Segments, col: Dict[str, Any], rows: np.ndarray, key: str = "ref",
                        ship_key: str = "ship", with_reasons: bool = True) -> Dict[str, Any]:
    """Recall, precision and onset error of a detector against a reference event set."""
    rec_r, hit_r, rec_ra, hit_ra, rec_p, hit_p, rec_o = [], [], [], [], [], [], []
    oerr, perr, eerr = [], [], []
    reasons: Dict[str, int] = {}
    fp_rise = []
    for i in rows:
        ref, ship = col[key][i], col[ship_key][i]
        el = eligible(ref)
        ela = eligible(ref, need_valid=False)
        pairs = IO.match_peaks(ref["peak_raw"], ship["peak_raw"], TOL)
        mr = {a for a, _ in pairs}
        ms = {b for _, b in pairs}
        for a in np.where(ela)[0]:
            rec_ra.append(seg.guid[i])
            hit_ra.append(a in mr)
        for a in np.where(el)[0]:
            rec_r.append(seg.guid[i])
            hit_r.append(a in mr)
            if a not in mr and with_reasons:
                r = miss_reason(col["st"][i], int(ref["peak_raw"][a]))
                reasons[r] = reasons.get(r, 0) + 1
        for b in range(ship["peak_raw"].size):
            rec_p.append(seg.guid[i])
            hit_p.append(b in ms)
            if b not in ms:
                u, base = D.ua_rise(seg.up[i], seg.up_valid[i])
                fp_rise.append(float((u - base)[ship["peak_raw"][b]]))
        for a, b in pairs:
            if not el[a]:
                continue
            rec_o.append(seg.guid[i])
            oerr.append((ship["onset_raw"][b] - ref["onset_raw"][a]) / FS)
            perr.append((ship["peak_raw"][b] - ref["peak_raw"][a]) / FS)
            eerr.append((ship["end_raw"][b] - ref["end_raw"][a]) / FS)
    oerr = np.asarray(oerr)
    out = {
        "recall_eligible": IO.ratio_bootstrap(rec_r, hit_r, np.ones(len(hit_r))),
        "recall_all_offedge": IO.ratio_bootstrap(rec_ra, hit_ra, np.ones(len(hit_ra))),
        "precision": IO.ratio_bootstrap(rec_p, hit_p, np.ones(len(hit_p))),
        "onset_error_median": IO.stat_bootstrap(rec_o, oerr, IO.median),
        "onset_error_iqr": IO.stat_bootstrap(rec_o, oerr, IO.iqr),
        "onset_error_pct": {f"p{k}": float(np.percentile(oerr, k)) for k in (5, 25, 50, 75, 95)} if oerr.size else {},
        "peak_error_iqr": IO.stat_bootstrap(rec_o, perr, IO.iqr),
        "end_error_median": IO.stat_bootstrap(rec_o, eerr, IO.median),
        "end_error_iqr": IO.stat_bootstrap(rec_o, eerr, IO.iqr),
        "n_ref_eligible": len(hit_r), "n_ship": len(hit_p),
    }
    if with_reasons:
        tot = max(sum(reasons.values()), 1)
        out["miss_reasons"] = {k: {"n": v, "share": v / tot} for k, v in sorted(reasons.items(), key=lambda t: -t[1])}
        fp_rise = np.asarray(fp_rise)
        out["unmatched_shipped_rise_pct"] = (
            {f"p{k}": float(np.percentile(fp_rise, k)) for k in (10, 25, 50, 75, 90)} if fp_rise.size else {})
        out["unmatched_shipped_share_rise_lt12"] = float(np.mean(fp_rise < 12)) if fp_rise.size else None
    return out


def validity_rule(seg: IO.Segments, col: Dict[str, Any], rows: np.ndarray) -> Dict[str, Any]:
    r"""Drop-rule audit: share of pre-drop detections touching an invalid UA sample, and the cost.

    Pre-drop detections are the shipped detector run on the gap-filled trace without dropping.
    The drop share is split into "touches a step-invalid sample" (dropped also with the UA rule
    off) and "touches only a UA-invalid sample".
    """
    rec, dropped_ua, inv_s, rec_d, bits = [], [], [], [], {}
    names = {v: k for k, v in IO.S.UP_BITS.items()}
    removed = added = 0.0
    rec_m, m_on, m_off, m_rm = [], [], [], []
    for i in rows:
        pre = col["pre"][i]
        cj = np.concatenate([[0], np.cumsum(~seg.up_valid[i])])
        cs = np.concatenate([[0], np.cumsum(~seg.step_valid[i])])
        o, e = pre["onset_raw"], pre["end_raw"]
        nj = cj[e + 1] - cj[o]
        ns = cs[e + 1] - cs[o]
        for k in range(o.size):
            if ns[k] > 0:
                continue
            rec.append(seg.guid[i])
            dropped_ua.append(nj[k] > 0)
            if nj[k] > 0:
                inv_s.append(nj[k] / FS)
                rec_d.append(seg.guid[i])
                q = seg.up_quality[i][o[k]:e[k] + 1][~seg.up_valid[i][o[k]:e[k] + 1]]
                for bit, name in names.items():
                    c = int(((q & bit) != 0).sum())
                    if c:
                        bits[name] = bits.get(name, 0) + c
        w_on = IO.onset_weight(col["ship"][i]["onset_raw"])
        w_off = IO.onset_weight(col["ship_off"][i]["onset_raw"])
        keep_step = ns == 0
        keep_joint = nj == 0
        w_ps = IO.onset_weight(o[keep_step])
        w_pj = IO.onset_weight(o[keep_joint])
        valid_steps = seg.weight[i] > 0
        rec_m.append(seg.guid[i])
        m_on.append(float((w_on * valid_steps).sum()))
        m_off.append(float((w_off * valid_steps).sum()))
        m_rm.append(float(((w_ps - w_pj) * valid_steps).sum()))
        removed += float(((w_off == 1) & (w_on == 0) & valid_steps).sum())
        added += float(((w_off == 0) & (w_on == 1) & valid_steps).sum())
    inv_s = np.asarray(inv_s)
    tot_bits = max(sum(bits.values()), 1)
    return {
        "share_predrop_dropped_by_ua_rule": IO.ratio_bootstrap(rec, dropped_ua, np.ones(len(rec))),
        "n_predrop_events_step_valid": len(rec),
        "dropped_invalid_seconds_pct": {f"p{k}": float(np.percentile(inv_s, k)) for k in (10, 25, 50, 75, 90)} if inv_s.size else {},
        "share_dropped_with_le8s_invalid": IO.stat_bootstrap(rec_d, inv_s, IO.share(lambda x: x <= 8)),
        "share_dropped_with_le4s_invalid": IO.stat_bootstrap(rec_d, inv_s, IO.share(lambda x: x <= 4)),
        "invalid_bits_in_dropped_spans": {k: v / tot_bits for k, v in sorted(bits.items(), key=lambda t: -t[1])},
        "reward_mass_removed_by_drop_only": IO.ratio_bootstrap(rec_m, m_rm, _pre_step_mass(seg, col, rows)),
        "reward_mass_on_vs_off_ratio": IO.ratio_bootstrap(rec_m, m_on, m_off),
        "steps_removed_on_vs_off_share_of_off": removed / max(sum(m_off), 1.0),
        "steps_added_on_vs_off_share_of_off": added / max(sum(m_off), 1.0),
    }


def invalid_by_phase(seg: IO.Segments, col: Dict[str, Any], rows: np.ndarray) -> Dict[str, Any]:
    r"""UA-invalid rate by contraction phase, on the reference events at $\theta = 12$.

    Phases: ``pre`` ($30\,$s before the onset), ``rise`` (onset to $8\,$s before the peak),
    ``apex`` (peak $\pm 8\,$s), ``fall`` ($8\,$s after the peak to the end), and ``between``
    (samples outside every event widened by $30\,$s before its onset). A higher rate at the apex
    than between contractions means the quality mask flags contraction apexes.

    Returns:
        Per phase the pooled invalid rate with a recording bootstrap, the apex-to-between ratio,
        and the bit shares of invalid apex samples.
    """
    names = {v: k for k, v in IO.S.UP_BITS.items()}
    acc: Dict[str, List[List[float]]] = {k: [[], [], []] for k in ("pre", "rise", "apex", "fall", "between")}
    bits: Dict[str, int] = {}
    for i in rows:
        v = seg.up_valid[i]
        ref = col["ref12"][i]
        inside = np.zeros(IO.SEG_RAW, dtype=bool)
        tally = {k: [0, 0] for k in acc}
        for o, p, e in zip(ref["onset_raw"], ref["peak_raw"], ref["end_raw"]):
            inside[max(0, o - 120):e + 1] = True
            for k, (a, b) in {"pre": (o - 120, o), "rise": (o, p - 32), "apex": (p - 32, p + 33),
                              "fall": (p + 33, e + 1)}.items():
                a, b = max(a, 0), min(b, IO.SEG_RAW)
                if b > a:
                    tally[k][0] += int((~v[a:b]).sum())
                    tally[k][1] += b - a
                    if k == "apex":
                        q = seg.up_quality[i][a:b][~v[a:b]]
                        for bit, name in names.items():
                            c = int(((q & bit) != 0).sum())
                            if c:
                                bits[name] = bits.get(name, 0) + c
        tally["between"] = [int((~v[~inside]).sum()), int((~inside).sum())]
        for k, (n_inv, n_all) in tally.items():
            acc[k][0].append(seg.guid[i])
            acc[k][1].append(n_inv)
            acc[k][2].append(n_all)
    out: Dict[str, Any] = {k: IO.ratio_bootstrap(r, a, b) for k, (r, a, b) in acc.items()}
    ratio_num = np.asarray(acc["apex"][1], float)
    out["apex_over_between_rate"] = (out["apex"]["mean"] / out["between"]["mean"]) if out["between"]["mean"] else None
    rec = np.asarray(acc["apex"][0])
    out["apex_minus_between_rate"] = IO.ratio_diff_bootstrap(rec, acc["between"][1], acc["between"][2],
                                                             rec, ratio_num, acc["apex"][2])
    tot = max(sum(bits.values()), 1)
    out["apex_invalid_bits"] = {k: v / tot for k, v in sorted(bits.items(), key=lambda t: -t[1])}
    return out


def _pre_step_mass(seg: IO.Segments, col: Dict[str, Any], rows: np.ndarray) -> np.ndarray:
    """Reward mass per segment of the pre-drop detections that survive the step-validity drop."""
    out = []
    for i in rows:
        pre = col["pre"][i]
        cs = np.concatenate([[0], np.cumsum(~seg.step_valid[i])])
        keep = (cs[pre["end_raw"] + 1] - cs[pre["onset_raw"]]) == 0
        out.append(float((IO.onset_weight(pre["onset_raw"][keep]) * (seg.weight[i] > 0)).sum()))
    return np.asarray(out)


def reward_window(seg: IO.Segments, col: Dict[str, Any], rows: np.ndarray) -> Dict[str, Any]:
    """Where the shipped onset and peak sit against the $[o, o + 15]$ step reward window."""
    rec, inwin, rec_s, w1, nvalid, cover, rec_c = [], [], [], [], [], [], []
    for i in rows:
        ev = col["ship"][i]
        os_, ps = ev["onset_raw"] // IO.DEC, ev["peak_raw"] // IO.DEC
        es = ev["end_raw"] // IO.DEC
        for k in range(os_.size):
            rec.append(seg.guid[i])
            inwin.append(os_[k] <= ps[k] <= os_[k] + REACH)
            rec_c.append(seg.guid[i])
            cover.append(min(REACH + 1, es[k] - os_[k] + 1) / max(es[k] - os_[k] + 1, 1))
        vs = seg.weight[i] > 0
        w = IO.onset_weight(ev["onset_raw"])
        rec_s.append(seg.guid[i])
        w1.append(float((w * vs).sum()))
        nvalid.append(float(vs.sum()))
    return {
        "share_peaks_inside_reward_window": IO.ratio_bootstrap(rec, inwin, np.ones(len(rec))),
        "share_valid_steps_weight1": IO.ratio_bootstrap(rec_s, w1, nvalid),
        "share_of_contraction_span_covered_by_window": IO.stat_bootstrap(rec_c, cover, lambda x: float(np.mean(x))),
    }


# =============================================================================
# Overlap consistency
# =============================================================================
def recut_consistency(seg: IO.Segments, detector, offsets=(330, 660), margin_s: float = 60.0,
                      native: bool = True) -> Dict[str, Any]:
    """Overlap consistency of ``detector(up, valid)`` on re-cut windows of stitched runs.

    Args:
        seg: The loaded rows.
        detector: Callable returning ``onset_raw``/``peak_raw``/``end_raw``.
        offsets: Re-cut offsets in seconds.
        margin_s: Interior margin from each window edge, in seconds.
        native: Also score the native 120 s overlap with a 30 s margin.

    Returns:
        Agreement and onset/peak/end disagreement per offset.
    """
    res: Dict[str, Any] = {}
    plans = [(off, margin_s) for off in offsets] + ([(1200, 30.0)] if native else [])
    for off, mg in plans:
        rec_a, num, den, rec_o, od, pd, ed = [], [], [], [], [], [], []
        for run in IO.runs(seg):
            up = IO.stitch(seg.up, run, IO.SEG_RAW, 4800)
            va = IO.stitch(seg.up_valid, run, IO.SEG_RAW, 4800)
            cache: Dict[int, Dict[str, np.ndarray]] = {}

            def det(a: int) -> Dict[str, np.ndarray]:
                if a not in cache:
                    cache[a] = detector(up[a:a + IO.SEG_RAW], va[a:a + IO.SEG_RAW])
                return cache[a]

            starts = range(0, up.size - IO.SEG_RAW - off * FS + 1, 4800)
            for a in starts:
                b = a + off * FS
                lo, hi = b + int(mg * FS), a + IO.SEG_RAW - int(mg * FS)
                if hi <= lo:
                    continue
                e1, e2 = det(a), det(b)
                p1 = e1["peak_raw"] + a
                p2 = e2["peak_raw"] + b
                m1 = (p1 >= lo) & (p1 < hi)
                m2 = (p2 >= lo) & (p2 < hi)
                # Match against the full other set so a peak just across the interior line counts.
                pairs = IO.match_peaks(p1, p2, TOL)
                matched = [(x, y) for x, y in pairs if m1[x] or m2[y]]
                n1, n2 = int(m1.sum()), int(m2.sum())
                mm = sum(1 for x, y in matched if m1[x] and m2[y]) + 0.5 * sum(
                    1 for x, y in matched if m1[x] != m2[y])
                rec_a.append(seg.guid[run[0]])
                num.append(2 * mm)
                den.append(n1 + n2)
                for x, y in matched:
                    if m1[x] and m2[y]:
                        rec_o.append(seg.guid[run[0]])
                        od.append((e1["onset_raw"][x] + a - e2["onset_raw"][y] - b) / FS)
                        pd.append((p1[x] - p2[y]) / FS)
                        ed.append((e1["end_raw"][x] + a - e2["end_raw"][y] - b) / FS)
        od, pd, ed = map(np.asarray, (od, pd, ed))
        key = "native_120s_margin30" if off == 1200 else f"offset_{off}s_margin{int(mg)}"
        res[key] = {
            "agreement": IO.ratio_bootstrap(rec_a, num, den),
            "onset_diff_iqr": IO.stat_bootstrap(rec_o, od, IO.iqr),
            "onset_absdiff_median": IO.stat_bootstrap(rec_o, np.abs(od), IO.median),
            "onset_share_identical": IO.stat_bootstrap(rec_o, od, IO.share(lambda x: np.abs(x) < 0.5)),
            "peak_diff_iqr": IO.stat_bootstrap(rec_o, pd, IO.iqr),
            "end_diff_iqr": IO.stat_bootstrap(rec_o, ed, IO.iqr),
            "end_absdiff_median": IO.stat_bootstrap(rec_o, np.abs(ed), IO.median),
            "n_pairs": len(num), "n_events": float(np.sum(den)),
        }
    return res


# =============================================================================
# Panels
# =============================================================================
def panels(seg: IO.Segments, rows: np.ndarray, events: Dict[str, Sequence[Dict[str, np.ndarray]]],
           styles: Dict[str, Dict[str, Any]], path_stem: Path, per_page: int = 6,
           weight_key: Optional[str] = None, title: str = "") -> List[str]:
    """Save stacked UA panels with validity shading and every named event set.

    Args:
        seg: The loaded rows.
        rows: Row indices to draw.
        events: ``{name: per-row event list}``; each set gets its own band at the top.
        styles: ``{name: {'color', 'band'}}``, ``band`` in $[0, 1)$ the band's lower edge.
        path_stem: Output path stem; pages get ``_<k>.png``.
        per_page: Panels per page.
        weight_key: Name of the event set whose onsets draw the reward weight as a shaded strip.
        title: Figure title.

    Returns:
        The saved paths.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(IO.SEG_RAW) / FS / 60.0
    paths = []
    for page in range(int(np.ceil(rows.size / per_page))):
        sub = rows[page * per_page:(page + 1) * per_page]
        fig, axs = plt.subplots(sub.size, 1, figsize=(22, 3.4 * sub.size))
        axs = np.atleast_1d(axs)
        for ax, i in zip(axs, sub):
            up, v = seg.up[i], seg.up_valid[i]
            u, b = D.ua_rise(up, v)
            ax.plot(t, up, color="0.65", lw=0.6, label="up (stored)")
            ax.plot(t, u, color="k", lw=1.1, label="band-limited")
            ax.plot(t, b, color="tab:green", lw=1.0, label="baseline")
            ax.plot(t, b + D.REF_THETA, color="tab:green", lw=0.6, ls=":", label="baseline + theta")
            ax.fill_between(t, 0, 1, where=~v, transform=ax.get_xaxis_transform(), color="red", alpha=0.25,
                            lw=0, label="UA invalid")
            if weight_key is not None:
                w = np.repeat(IO.onset_weight(events[weight_key][i]["onset_raw"]), IO.DEC)
                ax.fill_between(t, 0, 0.06, where=w > 0, transform=ax.get_xaxis_transform(), color="tab:purple",
                                alpha=0.6, lw=0, label=f"reward weight ({weight_key})")
            for name, evs in events.items():
                ev = evs[i]
                sty = styles[name]
                y0 = sty["band"]
                for o, p, e in zip(ev["onset_raw"], ev["peak_raw"], ev["end_raw"]):
                    ax.axvspan(o / 240, e / 240, ymin=y0, ymax=y0 + 0.07, color=sty["color"], alpha=0.45, lw=0)
                    ax.axvline(p / 240, color=sty["color"], lw=1.0, ls=sty.get("ls", "-"))
                    ax.plot([o / 240], [0.0], transform=ax.get_xaxis_transform(), marker="^", ms=8,
                            color=sty["color"], clip_on=False)
                ax.text(1.002, y0 + 0.03, f"{name} ({ev['peak_raw'].size})", transform=ax.transAxes,
                        color=sty["color"], fontsize=9, va="center")
            ax.set_xlim(0, IO.SEG_S / 60.0)
            ax.set_title(f"{seg.cohort[i]}  {seg.guid[i][:8]}  code {seg.code[i]}  "
                         f"{seg.hours_before()[i]:.1f} h before delivery  UA valid {v.mean():.0%}",
                         fontsize=10, loc="left")
            ax.set_ylabel("toco units")
        axs[0].legend(loc="upper left", ncol=6, fontsize=8)
        axs[-1].set_xlabel("minutes into segment (triangles = onsets, lines = peaks, bars = onset..end)")
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 0.95, 0.98))
        p = Path(f"{path_stem}_{page + 1}.png")
        fig.savefig(p, dpi=55)
        plt.close(fig)
        paths.append(str(p))
    return paths


def pick_panel_rows(seg: IO.Segments, n: int, seed: int, codes: Sequence[int]) -> np.ndarray:
    """Draw ``n`` rows balanced over cohorts and spread over time to delivery."""
    rng = np.random.default_rng(seed)
    pool = np.where(np.isin(seg.code, list(codes)))[0]
    cohorts = list(np.unique(seg.cohort[pool]))
    out: List[int] = []
    while len(out) < n:
        for c in cohorts:
            cand = pool[(seg.cohort[pool] == c) & ~np.isin(pool, out)]
            if cand.size and len(out) < n:
                out.append(int(rng.choice(cand)))
    out_arr = np.array(out)
    return out_arr[np.argsort(seg.hours_before()[out_arr])]


# =============================================================================
# Main
# =============================================================================
def main(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the contraction audit and write the JSON and panels."""
    IO.OUT_DIR.mkdir(parents=True, exist_ok=True)
    seg = IO.load_segments(tuple(args.codes), with_fhr=False)
    col = collect(seg)
    report: Dict[str, Any] = {"n_segments": len(seg), "n_recordings": int(np.unique(seg.guid).size)}
    for label, codes in (("code0", [0]), ("code2", [2]), ("code0+2", [0, 2])):
        rows = np.where(np.isin(seg.code, codes))[0]
        if rows.size == 0:
            continue
        sub = seg.subset(rows)
        clean = np.where(sub.up_valid.mean(1) >= 0.95)[0]
        block = {
            "rates_shipped": rates(seg, col["ship"], rows=rows),
            "rates_shipped_clean_segments": rates(sub, [col["ship"][r] for r in rows], rows=clean),
            "rates_reference": rates(seg, col["ref"], ref=True, rows=rows),
            "shapes_shipped": shapes(seg, col["ship"], rows),
            "shapes_reference": shapes(seg, [dict(r, peak_raw=r["peak_raw"]) for r in col["ref"]], rows),
            "reference_theta20": reference_agreement(seg, col, rows),
            "reference_theta12": reference_agreement(seg, col, rows, key="ref12", with_reasons=False),
            "reference_theta30": reference_agreement(seg, col, rows, key="ref30", with_reasons=False),
            "validity_rule": validity_rule(seg, col, rows),
            "invalid_by_phase": invalid_by_phase(seg, col, rows),
            "reward_window": reward_window(seg, col, rows),
        }
        report[label] = block
        print(label, "done", flush=True)
    report["overlap_shipped"] = recut_consistency(seg, IO.shipped_contractions)
    report["overlap_reference"] = recut_consistency(seg, D.reference_contractions)
    prow = pick_panel_rows(seg, args.n_panels, args.seed, args.codes)
    report["panels"] = panels(
        seg, prow, {"shipped": col["ship"], "reference": col["ref"]},
        {"shipped": {"color": "tab:blue", "band": 0.92}, "reference": {"color": "tab:orange", "band": 0.84, "ls": "--"}},
        IO.OUT_DIR / "panels_contractions", weight_key="shipped",
        title="Shipped contraction detector (blue) vs detector-free reference (orange); codes 0/2")
    report["panel_rows"] = [{"cohort": seg.cohort[i], "guid": seg.guid[i][:8], "hours": float(seg.hours_before()[i])}
                            for i in prow]
    path = IO.OUT_DIR / "contractions.json"
    path.write_text(json.dumps(IO.to_jsonable(report), indent=1))
    print("wrote", path)
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line, defaulting to :data:`RUN_ARGS`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--codes", type=int, nargs="+", default=RUN_ARGS["codes"])
    parser.add_argument("--n-panels", dest="n_panels", type=int, default=RUN_ARGS["n_panels"])
    parser.add_argument("--seed", type=int, default=RUN_ARGS["seed"])
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
