r"""Phase 1: audit of the shipped deceleration detector against the NICHD definition.

:func:`teb_vae.lag_attn_cfs.eval.events.detect_decelerations` runs here on the stored bridged FHR
in bpm under the FHR validity (step weight above 0, no invalid quality bit, and a non-zero value).
Recordings of split code 0 and 2 only. Writes ``out/detectors/decels.json`` and panel figures.

**NICHD definition used.** A deceleration is a decrease of at least $15\,$bpm below the baseline
that lasts at least $15\,$s and less than $2\,$min from onset to return. A decrease of $2$ to
$10\,$min is *prolonged*. With respect to a contraction: *early* is gradual (onset to nadir
$\ge 30\,$s) with the nadir coinciding with the contraction peak, operationalised as
$|t_{\mathrm{nadir}} - t_{\mathrm{peak}}| \le 10\,$s; *late* is gradual with the nadir after the
peak ($> 10\,$s); *variable* is abrupt (onset to nadir $< 30\,$s). The association window is
$t_{\mathrm{nadir}} - t_{\mathrm{peak}} \in [-30, +60]\,$s, nearest peak first.

**Baselines.** Three estimates, for the sensitivity test, all on a $1\,$Hz smoothed trace of the
valid samples:

* ``iter10``: a centred $10\,$min rolling median, recomputed twice after excluding samples more
  than $15\,$bpm from the current estimate (the NICHD baseline excludes periodic changes);
* ``med5``: a plain centred $5\,$min rolling median;
* ``p60_10``: a centred $10\,$min rolling 60th percentile (biased upwards).

**The reference.** A detector-free deceleration is a run of at least $15\,$s where the smoothed
trace is at least $15\,$bpm below ``iter10`` (runs closer than $5\,$s merged). Its nadir is the run
minimum; its onset and end are where the trace is back within $5\,$bpm of the baseline or at a
local extremum, walking out from the run, at most $120\,$s.

Edit :data:`RUN_ARGS` and run this file from the IDE, or pass ``--codes``.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs.session7 import detectors7 as D  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import p1_detector_io as IO  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7.p1_detector_contractions import (  # noqa: E402
    hour_bin, pick_panel_rows)

warnings.filterwarnings("ignore")

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {"codes": [0, 2], "n_panels": 24, "seed": 11}

FS = 4
TOL = 30 * FS
EDGE = 30 * FS


# =============================================================================
# Baselines and the reference
# =============================================================================
def fhr_baselines(fhr: np.ndarray, valid: np.ndarray) -> Dict[str, np.ndarray]:
    r"""Three FHR baseline estimates on the $4\,$Hz grid; see the module docstring.

    Args:
        fhr: FHR in bpm.
        valid: Its validity.

    Returns:
        ``{'iter10', 'med5', 'p60_10', 'smooth'}``; ``smooth`` is the $1\,$Hz-smoothed trace
        interpolated back to $4\,$Hz (gap-filled).
    """
    import pandas as pd

    x = np.where(valid, fhr, np.nan).reshape(-1, FS)
    with np.errstate(all="ignore"):
        x1 = np.nanmean(x, axis=1)
    s = pd.Series(x1)
    sm = s.rolling(5, center=True, min_periods=1).mean()
    out: Dict[str, np.ndarray] = {}
    b = sm.rolling(600, center=True, min_periods=60).median()
    for _ in range(2):
        keep = sm.where((sm - b).abs() <= 15)
        b = keep.rolling(600, center=True, min_periods=60).median()
    out["iter10"] = b.to_numpy()
    out["med5"] = sm.rolling(300, center=True, min_periods=30).median().to_numpy()
    out["p60_10"] = sm.rolling(600, center=True, min_periods=60).quantile(0.6).to_numpy()
    grid = np.arange(fhr.size) / FS
    t1 = np.arange(x1.size) + 0.375
    res = {}
    for k, v in out.items():
        ok = np.isfinite(v)
        res[k] = np.interp(grid, t1[ok], v[ok]) if ok.any() else np.full(fhr.size, np.nan)
    smv = sm.to_numpy()
    ok = np.isfinite(smv)
    res["smooth"] = np.interp(grid, t1[ok], smv[ok]) if ok.any() else np.full(fhr.size, np.nan)
    return res


def reference_decels(smooth: np.ndarray, base: np.ndarray, valid: np.ndarray) -> Dict[str, np.ndarray]:
    """Detector-free NICHD-style decelerations; see the module docstring.

    Args:
        smooth: The smoothed FHR at 4 Hz.
        base: The baseline at 4 Hz.
        valid: FHR validity.

    Returns:
        ``{'onset_raw', 'nadir_raw', 'end_raw', 'depth_bpm', 'invalid_in_span'}``.
    """
    below = (base - smooth) >= 15.0
    idx = np.flatnonzero(np.diff(np.concatenate([[0], below.astype(int), [0]])))
    starts, stops = idx[::2], idx[1::2]
    merged: List[List[int]] = []
    for a, b in zip(starts, stops):
        if merged and a - merged[-1][1] < 5 * FS:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    on, nd, en, dp = [], [], [], []
    walk = 120 * FS
    for a, b in merged:
        if b - a < 15 * FS:
            continue
        n = a + int(np.argmin(smooth[a:b]))
        # Both walks climb back towards the baseline: continue while the trace is still more than
        # 5 bpm below it and still rising away from the nadir.
        o = a
        while o > max(0, a - walk) and base[o - 1] - smooth[o - 1] > 5.0 and smooth[o - 1] >= smooth[o] - 0.05:
            o -= 1
        e = b - 1
        while e < min(smooth.size - 1, b - 1 + walk) and base[e + 1] - smooth[e + 1] > 5.0 and smooth[e + 1] >= smooth[e] - 0.05:
            e += 1
        on.append(o)
        nd.append(n)
        en.append(e)
        dp.append(float(base[n] - smooth[n]))
    on, nd, en = (np.asarray(v, dtype=np.int64) for v in (on, nd, en))
    cum = np.concatenate([[0], np.cumsum(~valid)])
    inv = cum[en + 1] - cum[on] if on.size else np.zeros(0, dtype=np.int64)
    return {"onset_raw": on, "nadir_raw": nd, "end_raw": en, "depth_bpm": np.asarray(dp), "invalid_in_span": inv}


# =============================================================================
# Collection
# =============================================================================
def collect(seg: IO.Segments) -> Dict[str, Any]:
    """Shipped decelerations, baselines, reference decelerations and contractions per segment."""
    out: Dict[str, List[Any]] = {k: [] for k in ("ship", "base", "ref", "ctx_ship", "ctx_ref")}
    for i in range(len(seg)):
        f, v = seg.fhr[i], seg.fhr_valid[i]
        ship = IO.shipped_decelerations(f, v)
        bl = fhr_baselines(f, v)
        out["ship"].append(ship)
        out["base"].append({k: bl[k] for k in ("iter10", "med5", "p60_10", "smooth")})
        out["ref"].append(reference_decels(bl["smooth"], bl["iter10"], v))
        out["ctx_ship"].append(IO.shipped_contractions(seg.up[i], seg.up_valid[i]))
        out["ctx_ref"].append(D.reference_contractions(seg.up[i], seg.up_valid[i]))
    return out


def nichd_flags(ev: Dict[str, np.ndarray], bl: Dict[str, np.ndarray], key: str) -> Dict[str, np.ndarray]:
    """Depth below a baseline at the nadir, duration, and NICHD qualification per event."""
    n = ev["nadir_raw"]
    depth = bl[key][n] - bl["smooth"][n] if n.size else np.zeros(0)
    dur = (ev["end_raw"] - ev["onset_raw"]) / FS
    return {"depth": depth, "dur": dur, "nichd": (depth >= 15) & (dur >= 15) & (dur < 120),
            "prolonged": (depth >= 15) & (dur >= 120)}


def classify(ev: Dict[str, np.ndarray], ctx: Dict[str, np.ndarray]) -> List[str]:
    """NICHD timing class of every deceleration against the contraction peaks ``ctx``."""
    out = []
    peaks = ctx["peak_raw"]
    for o, n in zip(ev["onset_raw"], ev["nadir_raw"]):
        abrupt = (n - o) / FS < 30
        cls = "unassociated"
        if peaks.size:
            d = (n - peaks) / FS
            ok = (d >= -30) & (d <= 60)
            if ok.any():
                dd = d[ok][np.argmin(np.abs(d[ok]))]
                if abrupt:
                    cls = "variable"
                elif abs(dd) <= 10:
                    cls = "early"
                elif dd > 10:
                    cls = "late"
                else:
                    cls = "gradual_nadir_before_peak"
        if cls == "unassociated" and abrupt:
            cls = "variable_unassociated"
        out.append(cls)
    return out


def delta_to_peak(ev: Dict[str, np.ndarray], ctx: Dict[str, np.ndarray], window_s: float = 120.0) -> np.ndarray:
    """Nadir minus the nearest contraction peak within ``window_s``, in seconds (NaN when none)."""
    out = []
    for n in ev["nadir_raw"]:
        d = (n - ctx["peak_raw"]) / FS
        d = d[np.abs(d) <= window_s]
        out.append(d[np.argmin(np.abs(d))] if d.size else np.nan)
    return np.asarray(out)


# =============================================================================
# Analyses
# =============================================================================
def rates(seg: IO.Segments, events: Sequence[Dict[str, np.ndarray]], rows: np.ndarray,
          select=None) -> Dict[str, Any]:
    """Events per 10 min of the eligible window, by cohort and hours before delivery."""
    n = []
    for i in rows:
        ev = events[i]
        p = ev["nadir_raw"]
        m = (p >= EDGE) & (p < IO.SEG_RAW - EDGE)
        if select is not None:
            m &= select(i)
        n.append(m.sum())
    n = np.asarray(n, dtype=float)
    den = np.full(n.size, (IO.SEG_S - 60) / 600.0)
    rec = seg.guid[rows]
    res: Dict[str, Any] = {"all": IO.ratio_bootstrap(rec, n, den)}
    for c in np.unique(seg.cohort[rows]):
        m = seg.cohort[rows] == c
        res[f"cohort={c}"] = IO.ratio_bootstrap(rec[m], n[m], den[m])
    hb = np.array([hour_bin(h) for h in seg.hours_before()[rows]])
    for b in sorted(set(hb), key=lambda s: float(s.split("-")[0])):
        m = hb == b
        res[f"hours={b}"] = IO.ratio_bootstrap(rec[m], n[m], den[m])
    return res


def analyse(seg: IO.Segments, col: Dict[str, Any], rows: np.ndarray) -> Dict[str, Any]:
    """All deceleration measurements on ``rows``."""
    res: Dict[str, Any] = {}
    res["rates_shipped"] = rates(seg, col["ship"], rows)
    res["rates_reference"] = rates(seg, col["ref"], rows)
    flags = {k: [nichd_flags(col["ship"][i], col["base"][i], k) for i in rows] for k in ("iter10", "med5", "p60_10")}
    recs = np.concatenate([[seg.guid[i]] * col["ship"][i]["nadir_raw"].size for i in rows]) if rows.size else np.zeros(0)
    prom = np.concatenate([col["ship"][i]["depth_bpm"] for i in rows])
    dur = np.concatenate([f["dur"] for f in flags["iter10"]])
    otn = np.concatenate([(col["ship"][i]["nadir_raw"] - col["ship"][i]["onset_raw"]) / FS for i in rows])
    q = lambda x: {f"p{k}": float(np.nanpercentile(x, k)) for k in (5, 25, 50, 75, 95)} if np.isfinite(x).any() else {}
    res["shipped_prominence_bpm"] = q(prom)
    res["shipped_duration_s"] = q(dur)
    res["shipped_onset_to_nadir_s"] = q(otn)
    res["share_prominence_lt15"] = IO.stat_bootstrap(recs, prom, IO.share(lambda x: x < 15))
    res["share_duration_lt15"] = IO.stat_bootstrap(recs, dur, IO.share(lambda x: x < 15))
    res["share_duration_ge120"] = IO.stat_bootstrap(recs, dur, IO.share(lambda x: x >= 120))
    for k, fl in flags.items():
        depth = np.concatenate([f["depth"] for f in fl])
        nichd = np.concatenate([f["nichd"] for f in fl]).astype(float)
        res[f"baseline={k}"] = {
            "depth_below_baseline_bpm": q(depth),
            "share_nichd": IO.stat_bootstrap(recs, nichd, lambda x: float(np.mean(x))),
            "share_depth_lt15": IO.stat_bootstrap(recs, depth, IO.share(lambda x: x < 15)),
            "share_nadir_above_baseline": IO.stat_bootstrap(recs, depth, IO.share(lambda x: x < 0)),
        }
        res[f"rates_shipped_nichd_{k}"] = rates(
            seg, col["ship"], rows,
            select=lambda i, k=k: nichd_flags(col["ship"][i], col["base"][i], k)["nichd"])
    # Baseline sensitivity: the paired change of the NICHD share between baselines.
    a = np.concatenate([f["nichd"] for f in flags["iter10"]]).astype(float)
    for k in ("med5", "p60_10"):
        b = np.concatenate([f["nichd"] for f in flags[k]]).astype(float)
        res[f"nichd_share_delta_{k}_minus_iter10"] = IO.S.paired_bootstrap(recs, b - a)
        res[f"nichd_flag_flip_rate_{k}_vs_iter10"] = IO.stat_bootstrap(recs, np.abs(b - a), lambda x: float(np.mean(x)))
    # Reference agreement.
    rr, hr, rp, hp, ro, oerr, nerr = [], [], [], [], [], [], []
    for i in rows:
        ref, ship = col["ref"][i], col["ship"][i]
        el = (ref["nadir_raw"] >= EDGE) & (ref["nadir_raw"] < IO.SEG_RAW - EDGE) & (ref["invalid_in_span"] == 0)
        el &= (ref["end_raw"] - ref["onset_raw"]) / FS < 120
        pairs = IO.match_peaks(ref["nadir_raw"], ship["nadir_raw"], TOL)
        mr, ms = {x for x, _ in pairs}, {y for _, y in pairs}
        for x in np.where(el)[0]:
            rr.append(seg.guid[i])
            hr.append(x in mr)
        fl = nichd_flags(ship, col["base"][i], "iter10")["nichd"]
        for y in range(ship["nadir_raw"].size):
            rp.append(seg.guid[i])
            hp.append(y in ms)
        for x, y in pairs:
            if el[x]:
                ro.append(seg.guid[i])
                oerr.append((ship["onset_raw"][y] - ref["onset_raw"][x]) / FS)
                nerr.append((ship["nadir_raw"][y] - ref["nadir_raw"][x]) / FS)
    res["reference"] = {
        "recall": IO.ratio_bootstrap(rr, hr, np.ones(len(hr))),
        "precision_vs_reference": IO.ratio_bootstrap(rp, hp, np.ones(len(hp))),
        "onset_error_median": IO.stat_bootstrap(ro, oerr, IO.median),
        "onset_error_iqr": IO.stat_bootstrap(ro, oerr, IO.iqr),
        "nadir_error_iqr": IO.stat_bootstrap(ro, nerr, IO.iqr),
        "n_ref": len(hr), "n_ship": len(hp),
    }
    # Timing against contractions.
    for cname in ("ctx_ship", "ctx_ref"):
        recs_c, cls_all, dl, rec_d, recs_n, cls_n = [], [], [], [], [], []
        for i in rows:
            ship = col["ship"][i]
            cl = classify(ship, col[cname][i])
            fl = nichd_flags(ship, col["base"][i], "iter10")["nichd"]
            d = delta_to_peak(ship, col[cname][i])
            recs_c += [seg.guid[i]] * len(cl)
            cls_all += cl
            dl += d.tolist()
            rec_d += [seg.guid[i]] * d.size
            recs_n += [seg.guid[i]] * int(fl.sum())
            cls_n += [c for c, f in zip(cl, fl) if f]
        block: Dict[str, Any] = {}
        for lab, rc, cl in (("all_shipped", recs_c, cls_all), ("nichd_qualified", recs_n, cls_n)):
            cl = np.asarray(cl)
            block[lab] = {c: IO.ratio_bootstrap(rc, cl == c, np.ones(cl.size)) for c in
                          ("early", "late", "variable", "gradual_nadir_before_peak", "unassociated",
                           "variable_unassociated")}
        dl = np.asarray(dl)
        block["nadir_minus_peak_s"] = q(dl)
        block["nadir_minus_peak_median"] = IO.stat_bootstrap(rec_d, dl, IO.median)
        block["share_with_peak_within_120s"] = float(np.mean(np.isfinite(dl))) if dl.size else None
        res[f"timing_vs_{cname}"] = block
    return res


# =============================================================================
# Overlap consistency
# =============================================================================
def recut(seg: IO.Segments, offsets=(330, 660), margin_s: float = 60.0) -> Dict[str, Any]:
    """Re-cut overlap consistency of the shipped deceleration detector; see the contraction script."""
    res: Dict[str, Any] = {}
    plans = [(off, margin_s) for off in offsets] + [(1200, 30.0)]
    for off, mg in plans:
        rec_a, num, den, rec_o, od, nd = [], [], [], [], [], []
        for run in IO.runs(seg):
            f = IO.stitch(seg.fhr, run, IO.SEG_RAW, 4800)
            va = IO.stitch(seg.fhr_valid, run, IO.SEG_RAW, 4800)
            cache: Dict[int, Dict[str, np.ndarray]] = {}

            def det(a: int) -> Dict[str, np.ndarray]:
                if a not in cache:
                    cache[a] = IO.shipped_decelerations(f[a:a + IO.SEG_RAW], va[a:a + IO.SEG_RAW])
                return cache[a]

            for a in range(0, f.size - IO.SEG_RAW - off * FS + 1, 4800):
                b = a + off * FS
                lo, hi = b + int(mg * FS), a + IO.SEG_RAW - int(mg * FS)
                if hi <= lo:
                    continue
                e1, e2 = det(a), det(b)
                p1, p2 = e1["nadir_raw"] + a, e2["nadir_raw"] + b
                m1, m2 = (p1 >= lo) & (p1 < hi), (p2 >= lo) & (p2 < hi)
                pairs = [(x, y) for x, y in IO.match_peaks(p1, p2, TOL) if m1[x] or m2[y]]
                mm = sum(1 for x, y in pairs if m1[x] and m2[y]) + 0.5 * sum(1 for x, y in pairs if m1[x] != m2[y])
                rec_a.append(seg.guid[run[0]])
                num.append(2 * mm)
                den.append(int(m1.sum() + m2.sum()))
                for x, y in pairs:
                    if m1[x] and m2[y]:
                        rec_o.append(seg.guid[run[0]])
                        od.append((e1["onset_raw"][x] + a - e2["onset_raw"][y] - b) / FS)
                        nd.append((p1[x] - p2[y]) / FS)
        key = "native_120s_margin30" if off == 1200 else f"offset_{off}s_margin{int(mg)}"
        res[key] = {"agreement": IO.ratio_bootstrap(rec_a, num, den),
                    "onset_diff_iqr": IO.stat_bootstrap(rec_o, od, IO.iqr),
                    "onset_absdiff_median": IO.stat_bootstrap(rec_o, np.abs(od), IO.median),
                    "nadir_diff_iqr": IO.stat_bootstrap(rec_o, nd, IO.iqr),
                    "n_events": float(np.sum(den))}
    return res


# =============================================================================
# Panels
# =============================================================================
def panels(seg: IO.Segments, col: Dict[str, Any], rows: np.ndarray, stem: Path, per_page: int = 6,
           name: str = "shipped") -> List[str]:
    """FHR panels with baseline, NICHD line, validity, detected and reference decelerations, and UA.

    Args:
        seg: The loaded rows.
        col: Collection with ``ship`` (the detector drawn), ``base``, ``ref`` and ``ctx_ship``.
        rows: Rows to draw.
        stem: Output path stem.
        per_page: Segments per page.
        name: Name of the detector drawn in blue, for the titles.

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
        fig, axs = plt.subplots(2 * sub.size, 1, figsize=(22, 4.6 * sub.size),
                                gridspec_kw={"height_ratios": [3, 1] * sub.size})
        for k, i in enumerate(sub):
            ax, ax2 = axs[2 * k], axs[2 * k + 1]
            f, v = seg.fhr[i], seg.fhr_valid[i]
            bl = col["base"][i]
            ax.plot(t, np.where(v, f, np.nan), color="0.6", lw=0.5)
            ax.plot(t, bl["smooth"], color="k", lw=0.9)
            ax.plot(t, bl["iter10"], color="tab:green", lw=1.2, label="baseline iter10")
            ax.plot(t, bl["iter10"] - 15, color="tab:green", lw=0.7, ls=":", label="baseline - 15")
            ax.plot(t, bl["med5"], color="tab:olive", lw=0.7, ls="--", label="baseline med5")
            ax.fill_between(t, 0, 1, where=~v, transform=ax.get_xaxis_transform(), color="red", alpha=0.2, lw=0)
            ship = col["ship"][i]
            fl = nichd_flags(ship, bl, "iter10")
            cl = classify(ship, col["ctx_ship"][i])
            for j, (o, n, e) in enumerate(zip(ship["onset_raw"], ship["nadir_raw"], ship["end_raw"])):
                c = "tab:blue" if fl["nichd"][j] else "tab:cyan"
                ax.axvspan(o / 240, e / 240, ymin=0.92, ymax=0.99, color=c, alpha=0.5, lw=0)
                ax.axvline(n / 240, color=c, lw=0.9)
                ax.text(n / 240, 0.86, cl[j][:4], transform=ax.get_xaxis_transform(), fontsize=7, color=c, ha="center")
            ref = col["ref"][i]
            for o, n, e in zip(ref["onset_raw"], ref["nadir_raw"], ref["end_raw"]):
                ax.axvspan(o / 240, e / 240, ymin=0.0, ymax=0.06, color="tab:orange", alpha=0.6, lw=0)
                ax.axvline(n / 240, color="tab:orange", lw=0.9, ls="--")
            lo = np.nanpercentile(np.where(v, f, np.nan), 1) if v.any() else 60
            ax.set_ylim(max(50, lo - 10), min(210, np.nanmax(bl["iter10"]) + 30) if np.isfinite(bl["iter10"]).any() else 200)
            ax.set_xlim(0, 22)
            ax.set_title(f"{seg.cohort[i]}  {seg.guid[i][:8]}  code {seg.code[i]}  {seg.hours_before()[i]:.1f} h  "
                         f"{name} {ship['nadir_raw'].size} (NICHD {int(fl['nichd'].sum())}, dark blue)  "
                         f"reference {ref['nadir_raw'].size} (orange)  FHR valid {v.mean():.0%}", fontsize=10, loc="left")
            ax.set_ylabel("bpm")
            ax2.plot(t, seg.up[i], color="0.4", lw=0.7)
            ax2.fill_between(t, 0, 1, where=~seg.up_valid[i], transform=ax2.get_xaxis_transform(), color="red", alpha=0.2, lw=0)
            for p in col["ctx_ship"][i]["peak_raw"]:
                ax2.axvline(p / 240, color="tab:purple", lw=1)
                ax.axvline(p / 240, color="tab:purple", lw=0.6, alpha=0.6)
            ax2.set_xlim(0, 22)
            ax2.set_ylabel("UA")
        axs[0].legend(loc="lower left", fontsize=8, ncol=3)
        axs[-1].set_xlabel(f"minutes (purple = contraction peaks used for timing; text = NICHD timing class of the {name} events)")
        fig.tight_layout()
        p = Path(f"{stem}_{page + 1}.png")
        fig.savefig(p, dpi=55)
        plt.close(fig)
        paths.append(str(p))
    return paths


def main(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the deceleration audit and write the JSON and panels."""
    IO.OUT_DIR.mkdir(parents=True, exist_ok=True)
    seg = IO.load_segments(tuple(args.codes), with_fhr=True)
    col = collect(seg)
    report: Dict[str, Any] = {}
    for label, codes in (("code0", [0]), ("code2", [2])):
        rows = np.where(np.isin(seg.code, codes))[0]
        if rows.size:
            report[label] = analyse(seg, col, rows)
            print(label, "done", flush=True)
    report["overlap_shipped"] = recut(seg)
    prow = pick_panel_rows(seg, args.n_panels, args.seed, args.codes)
    report["panels"] = panels(seg, col, prow, IO.OUT_DIR / "panels_decels")
    path = IO.OUT_DIR / "decels.json"
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
