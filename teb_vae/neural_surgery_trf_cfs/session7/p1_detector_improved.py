r"""Phase 1: tune the v7 detectors on split code 0 and validate them on split code 2.

The script compares the shipped contraction detector with
:func:`detectors7.detect_contractions_v7` on the same segments and with the same tests as
:mod:`p1_detector_contractions`:

* a small grid of the v7 rise threshold and validity rule on code 0 (the chosen point is the
  module default), scored by recall against the reference at $\theta = 20$ and precision against
  the reference at $\theta = 12$;
* rates, durations and intervals; recall, precision and onset error against the reference, with
  paired recording-level differences (v7 minus shipped); re-cut overlap consistency; the share
  of reward mass; panel figures of code-2 segments;
* the causal UA activity of :func:`detectors7.ua_activity`: a causality check, the share of
  valid steps above $0.5$, and its confusion table against both binary rewards;
* the NICHD deceleration detector :func:`detectors7.detect_decelerations_v7`: rates, depth and
  duration, overlap consistency, and timing classes against the v7 contractions.

**Circularity.** v7 measures its rise against the same kind of rolling low-quantile baseline as
the reference, so part of its precision gain against the reference is built in. The tests that
do not share that construction are overlap consistency, plausibility, the paired recall on
reference events with short invalid flags, and the panels. The v7 deceleration detector IS the
reference rule, so its reference agreement is not reported as evidence.

Edit :data:`RUN_ARGS` and run this file from the IDE, or pass ``--skip-grid``.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs.session7 import detectors7 as D  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import p1_detector_io as IO  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import p1_detector_contractions as C  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import p1_detector_decels as DC  # noqa: E402

warnings.filterwarnings("ignore")

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {"skip_grid": False, "n_panels": 24, "seed": 23}

FS = 4
TOL = 30 * FS
EDGE = 30 * FS


def v7(up: np.ndarray, valid: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """The v7 contraction detector at its defaults, on stored units."""
    return D.detect_contractions_v7(up, valid=valid)


# =============================================================================
# Collection
# =============================================================================
def collect(seg: IO.Segments) -> Dict[str, List[Any]]:
    """Run every detector and the activity on every segment."""
    out: Dict[str, List[Any]] = {k: [] for k in ("ship", "v7", "v7_nodrop", "v7_strict", "ref", "ref12", "act",
                                               "dec_v7", "base")}
    for i in range(len(seg)):
        up, v = seg.up[i], seg.up_valid[i]
        out["ship"].append(IO.shipped_contractions(up, v))
        out["v7"].append(v7(up, v))
        out["v7_nodrop"].append(D.detect_contractions_v7(D.fill_invalid(up, v), valid=None))
        out["v7_strict"].append(D.detect_contractions_v7(up, valid=v, tolerant_validity=False))
        out["ref"].append(D.reference_contractions(up, v))
        out["ref12"].append(D.reference_contractions(up, v, theta=12.0))
        out["act"].append(D.ua_activity(up, v))
        f, fv = seg.fhr[i], seg.fhr_valid[i]
        out["dec_v7"].append(D.detect_decelerations_v7(f, valid=fv))
        sm, bl = D.fhr_baseline(f, fv)
        out["base"].append({"smooth": sm, "iter10": bl, "med5": bl})
    return out


# =============================================================================
# Tests
# =============================================================================
def grid(seg: IO.Segments) -> List[Dict[str, Any]]:
    """Code-0 grid of the v7 rise threshold and validity rule."""
    rows = np.where(seg.code == 0)[0]
    r20 = [D.reference_contractions(seg.up[i], seg.up_valid[i]) for i in rows]
    r12 = [D.reference_contractions(seg.up[i], seg.up_valid[i], theta=12.0) for i in rows]
    res = []
    for rise in (10.0, 12.0, 15.0, 18.0):
        for tol in (True, False):
            hr = nr = hp = npk = 0
            for k, i in enumerate(rows):
                ev = D.detect_contractions_v7(seg.up[i], valid=seg.up_valid[i], rise=rise, tolerant_validity=tol)
                ref = r20[k]
                m = (ref["peak_raw"] >= EDGE) & (ref["peak_raw"] < IO.SEG_RAW - EDGE)
                hit = {a for a, _ in IO.match_peaks(ref["peak_raw"], ev["peak_raw"], TOL)}
                hr += sum(a in hit for a in np.where(m)[0])
                nr += int(m.sum())
                hp += len(IO.match_peaks(r12[k]["peak_raw"], ev["peak_raw"], TOL))
                npk += ev["peak_raw"].size
            r, p = hr / max(nr, 1), hp / max(npk, 1)
            res.append({"rise": rise, "tolerant": tol, "recall_ref20_offedge": r, "precision_ref12": p,
                        "f1": 2 * r * p / max(r + p, 1e-12), "per10min": npk / rows.size / 2.1})
    return res


def paired_recall(seg: IO.Segments, col: Dict[str, List[Any]], rows: np.ndarray, valid_only: bool,
                  max_invalid_s: Optional[float] = None) -> Dict[str, Any]:
    """Recall of shipped and v7 on the same reference events, and the paired difference."""
    rec, a, b = [], [], []
    for i in rows:
        ref = col["ref"][i]
        m = (ref["peak_raw"] >= EDGE) & (ref["peak_raw"] < IO.SEG_RAW - EDGE)
        if valid_only:
            m &= ref["invalid_in_span"] == 0
        if max_invalid_s is not None:
            m &= ref["invalid_in_span"] <= max_invalid_s * FS
        hs = {x for x, _ in IO.match_peaks(ref["peak_raw"], col["ship"][i]["peak_raw"], TOL)}
        hv = {x for x, _ in IO.match_peaks(ref["peak_raw"], col["v7"][i]["peak_raw"], TOL)}
        for x in np.where(m)[0]:
            rec.append(seg.guid[i])
            a.append(float(x in hs))
            b.append(float(x in hv))
    a, b = np.asarray(a), np.asarray(b)
    return {"shipped": IO.ratio_bootstrap(rec, a, np.ones(a.size)), "v7": IO.ratio_bootstrap(rec, b, np.ones(b.size)),
            "v7_minus_shipped": IO.S.paired_bootstrap(rec, b - a), "n_events": int(a.size)}


def precision(seg: IO.Segments, col: Dict[str, List[Any]], rows: np.ndarray, ref_key: str) -> Dict[str, Any]:
    """Precision of both detectors against a reference set, and the recording-paired difference."""
    arms = {}
    for det in ("ship", "v7"):
        rec, hit = [], []
        for i in rows:
            ev = col[det][i]
            ms = {y for _, y in IO.match_peaks(col[ref_key][i]["peak_raw"], ev["peak_raw"], TOL)}
            for y in range(ev["peak_raw"].size):
                rec.append(seg.guid[i])
                hit.append(float(y in ms))
        arms[det] = (rec, np.asarray(hit))
    (ra, ha), (rb, hb) = arms["ship"], arms["v7"]
    return {"shipped": IO.ratio_bootstrap(ra, ha, np.ones(ha.size)), "v7": IO.ratio_bootstrap(rb, hb, np.ones(hb.size)),
            "v7_minus_shipped": IO.ratio_diff_bootstrap(ra, ha, np.ones(ha.size), rb, hb, np.ones(hb.size))}


def rate_diff(seg: IO.Segments, col: Dict[str, List[Any]], rows: np.ndarray) -> Dict[str, Any]:
    """Per-segment rate difference v7 minus shipped (paired by recording)."""
    ns = np.array([col["ship"][i]["peak_raw"].size for i in rows], float)
    nv = np.array([col["v7"][i]["peak_raw"].size for i in rows], float)
    return IO.S.paired_bootstrap(seg.guid[rows], (nv - ns) / 2.1)


def walk_bound(col: Dict[str, List[Any]], rows: np.ndarray, key: str, walk_s: float, seg: IO.Segments) -> Dict[str, Any]:
    """Share of onsets and ends sitting at the walk bound."""
    rec, a, b = [], [], []
    for i in rows:
        ev = col[key][i]
        rec += [seg.guid[i]] * ev["peak_raw"].size
        a += ((ev["peak_raw"] - ev["onset_raw"]) >= walk_s * FS).tolist()
        b += ((ev["end_raw"] - ev["peak_raw"]) >= walk_s * FS).tolist()
    return {"onset_at_bound": IO.ratio_bootstrap(rec, a, np.ones(len(a))),
            "end_at_bound": IO.ratio_bootstrap(rec, b, np.ones(len(b)))}


def drop_share(seg: IO.Segments, col: Dict[str, List[Any]], rows: np.ndarray) -> Dict[str, Any]:
    """Share of v7 pre-drop detections removed by the tolerant and by the strict validity rule."""
    rec, nd, nt, ns = [], [], [], []
    for i in rows:
        rec.append(seg.guid[i])
        nd.append(col["v7_nodrop"][i]["peak_raw"].size)
        nt.append(col["v7"][i]["peak_raw"].size)
        ns.append(col["v7_strict"][i]["peak_raw"].size)
    nd, nt, ns = (np.asarray(x, float) for x in (nd, nt, ns))
    return {"tolerant": IO.ratio_bootstrap(rec, nd - nt, nd), "strict": IO.ratio_bootstrap(rec, nd - ns, nd)}


def weights(seg: IO.Segments, col: Dict[str, List[Any]], rows: np.ndarray) -> Dict[str, Any]:
    """Reward mass of both binary weights, their confusion, and the activity against both."""
    rec, ws, wv, nvs = [], [], [], []
    cells: Dict[str, float] = {}
    act_rows: Dict[str, List[float]] = {k: [] for k in ("rec", "a_hi", "n", "ws1_hi", "ws1", "ws0_hi", "ws0",
                                                         "wv1_hi", "wv1", "wv0_hi", "wv0", "a_ws1", "a_ws0")}
    for i in rows:
        vs = seg.weight[i] > 0
        s = IO.onset_weight(col["ship"][i]["onset_raw"]) > 0
        v = IO.onset_weight(col["v7"][i]["onset_raw"]) > 0
        a = col["act"][i][: vs.size]
        hi = a > 0.5
        rec.append(seg.guid[i])
        ws.append(float((s & vs).sum()))
        wv.append(float((v & vs).sum()))
        nvs.append(float(vs.sum()))
        for name, m in (("ship1_v71", s & v), ("ship1_v70", s & ~v), ("ship0_v71", ~s & v), ("ship0_v70", ~s & ~v),
                        ("ship1_act_hi", s & hi), ("ship1_act_lo", s & ~hi), ("ship0_act_hi", ~s & hi),
                        ("ship0_act_lo", ~s & ~hi), ("v71_act_hi", v & hi), ("v71_act_lo", v & ~hi),
                        ("v70_act_hi", ~v & hi), ("v70_act_lo", ~v & ~hi)):
            cells[name] = cells.get(name, 0.0) + float((m & vs).sum())
        act_rows["rec"].append(seg.guid[i])
        act_rows["a_hi"].append(float((hi & vs).sum()))
        act_rows["n"].append(float(vs.sum()))
        for tag, w in (("ws", s), ("wv", v)):
            act_rows[f"{tag}1_hi"].append(float((hi & w & vs).sum()))
            act_rows[f"{tag}1"].append(float((w & vs).sum()))
            act_rows[f"{tag}0_hi"].append(float((hi & ~w & vs).sum()))
            act_rows[f"{tag}0"].append(float((~w & vs).sum()))
        act_rows["a_ws1"].append(float((a * (s & vs)).sum()))
        act_rows["a_ws0"].append(float((a * (~s & vs)).sum()))
    tot = sum(nvs)
    r = act_rows["rec"]
    return {
        "share_valid_steps_weight1_shipped": IO.ratio_bootstrap(rec, ws, nvs),
        "share_valid_steps_weight1_v7": IO.ratio_bootstrap(rec, wv, nvs),
        "reward_mass_v7_over_shipped": IO.ratio_bootstrap(rec, wv, ws),
        "confusion_share_of_valid_steps": {k: v / tot for k, v in cells.items()},
        "activity_share_valid_steps_gt05": IO.ratio_bootstrap(r, act_rows["a_hi"], act_rows["n"]),
        "activity_gt05_given_shipped1": IO.ratio_bootstrap(r, act_rows["ws1_hi"], act_rows["ws1"]),
        "activity_gt05_given_shipped0": IO.ratio_bootstrap(r, act_rows["ws0_hi"], act_rows["ws0"]),
        "activity_gt05_given_v71": IO.ratio_bootstrap(r, act_rows["wv1_hi"], act_rows["wv1"]),
        "activity_gt05_given_v70": IO.ratio_bootstrap(r, act_rows["wv0_hi"], act_rows["wv0"]),
        "shipped1_given_activity_gt05": IO.ratio_bootstrap(r, act_rows["ws1_hi"], act_rows["a_hi"]),
        "mean_activity_given_shipped1": IO.ratio_bootstrap(r, act_rows["a_ws1"], act_rows["ws1"]),
        "mean_activity_given_shipped0": IO.ratio_bootstrap(r, act_rows["a_ws0"], act_rows["ws0"]),
    }


def activity_causality(seg: IO.Segments, rows: np.ndarray, n: int = 20) -> Dict[str, float]:
    """Largest change of the activity at steps $< k$ when everything after step $k$ is removed."""
    worst = 0.0
    for i in rows[:n]:
        full = D.ua_activity(seg.up[i], seg.up_valid[i])
        for k in (40, 150, 300):
            part = D.ua_activity(seg.up[i][: 16 * k], seg.up_valid[i][: 16 * k])
            worst = max(worst, float(np.max(np.abs(full[:k] - part[:k]))))
    return {"max_abs_change": worst, "n_segments": int(min(n, rows.size))}


def recut_generic(seg: IO.Segments, signal: np.ndarray, valid: np.ndarray, detector: Callable,
                  peak_key: str, offsets=(330, 660), margin_s: float = 60.0) -> Dict[str, Any]:
    """Re-cut overlap consistency of any detector on any stitched signal; see the contraction script."""
    res: Dict[str, Any] = {}
    for off in offsets:
        rec_a, num, den, rec_o, od = [], [], [], [], []
        for run in IO.runs(seg):
            x = IO.stitch(signal, run, IO.SEG_RAW, 4800)
            va = IO.stitch(valid, run, IO.SEG_RAW, 4800)
            cache: Dict[int, Dict[str, np.ndarray]] = {}
            for a in range(0, x.size - IO.SEG_RAW - off * FS + 1, 4800):
                b = a + off * FS
                for s in (a, b):
                    if s not in cache:
                        cache[s] = detector(x[s:s + IO.SEG_RAW], va[s:s + IO.SEG_RAW])
                e1, e2 = cache[a], cache[b]
                lo, hi = b + int(margin_s * FS), a + IO.SEG_RAW - int(margin_s * FS)
                p1, p2 = e1[peak_key] + a, e2[peak_key] + b
                m1, m2 = (p1 >= lo) & (p1 < hi), (p2 >= lo) & (p2 < hi)
                pairs = [(i, j) for i, j in IO.match_peaks(p1, p2, TOL) if m1[i] or m2[j]]
                mm = sum(1 for i, j in pairs if m1[i] and m2[j]) + 0.5 * sum(1 for i, j in pairs if m1[i] != m2[j])
                rec_a.append(seg.guid[run[0]])
                num.append(2 * mm)
                den.append(int(m1.sum() + m2.sum()))
                for i, j in pairs:
                    if m1[i] and m2[j]:
                        rec_o.append(seg.guid[run[0]])
                        od.append((e1["onset_raw"][i] + a - e2["onset_raw"][j] - b) / FS)
        res[f"offset_{off}s_margin{int(margin_s)}"] = {
            "agreement": IO.ratio_bootstrap(rec_a, num, den),
            "onset_diff_iqr": IO.stat_bootstrap(rec_o, od, IO.iqr),
            "onset_absdiff_median": IO.stat_bootstrap(rec_o, np.abs(od), IO.median),
            "n_events": float(np.sum(den))}
    return res


def decel_block(seg: IO.Segments, col: Dict[str, List[Any]], rows: np.ndarray) -> Dict[str, Any]:
    """Rates, shapes and timing classes of the v7 decelerations against the v7 contractions."""
    out: Dict[str, Any] = {"rates_v7": DC.rates(seg, col["dec_v7"], rows)}
    rec, dep, dur, pro, cls_all, rec_c, dl, rec_d = [], [], [], [], [], [], [], []
    for i in rows:
        ev = col["dec_v7"][i]
        rec += [seg.guid[i]] * ev["nadir_raw"].size
        dep += ev["depth_bpm"].tolist()
        dur += ((ev["end_raw"] - ev["onset_raw"]) / FS).tolist()
        pro += ev["prolonged"].tolist()
        cl = DC.classify(ev, col["v7"][i])
        cls_all += cl
        rec_c += [seg.guid[i]] * len(cl)
        d = DC.delta_to_peak(ev, col["v7"][i])
        dl += d.tolist()
        rec_d += [seg.guid[i]] * d.size
    q = lambda x: {f"p{k}": float(np.nanpercentile(x, k)) for k in (5, 25, 50, 75, 95)} if len(x) else {}
    cls_all = np.asarray(cls_all)
    out.update({
        "depth_bpm": q(dep), "duration_s": q(dur),
        "share_prolonged": IO.ratio_bootstrap(rec, pro, np.ones(len(pro))),
        "timing_classes": {c: IO.ratio_bootstrap(rec_c, cls_all == c, np.ones(cls_all.size)) for c in
                           ("early", "late", "variable", "gradual_nadir_before_peak", "unassociated",
                            "variable_unassociated")},
        "nadir_minus_peak_s": q(np.asarray(dl)),
        "nadir_minus_peak_median": IO.stat_bootstrap(rec_d, dl, IO.median),
    })
    return out


# =============================================================================
# Panels
# =============================================================================
def panels(seg: IO.Segments, col: Dict[str, List[Any]], rows: np.ndarray, stem: Path, per_page: int = 6) -> List[str]:
    """UA panels: shipped, v7 and reference events, both reward weights, and the causal activity."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(IO.SEG_RAW) / FS / 60.0
    ts = (np.arange(330) * 16 + 15) / FS / 60.0
    sty = {"ship": ("tab:blue", 0.92, "-"), "v7": ("tab:red", 0.84, "-"), "ref": ("tab:orange", 0.76, "--")}
    paths = []
    for page in range(int(np.ceil(rows.size / per_page))):
        sub = rows[page * per_page:(page + 1) * per_page]
        fig, axs = plt.subplots(sub.size, 1, figsize=(22, 3.6 * sub.size))
        axs = np.atleast_1d(axs)
        for ax, i in zip(axs, sub):
            up, v = seg.up[i], seg.up_valid[i]
            u, b = D.ua_rise(up, v)
            ax.plot(t, up, color="0.7", lw=0.6)
            ax.plot(t, u, color="k", lw=1.0)
            ax.plot(t, b, color="tab:green", lw=0.9)
            ax.fill_between(t, 0, 1, where=~v, transform=ax.get_xaxis_transform(), color="red", alpha=0.25, lw=0)
            for k, (key, y0) in enumerate((("ship", 0.0), ("v7", 0.05))):
                w = np.repeat(IO.onset_weight(col[key][i]["onset_raw"]), 16) > 0
                ax.fill_between(t, y0, y0 + 0.045, where=w, transform=ax.get_xaxis_transform(),
                                color=sty[key][0], alpha=0.5, lw=0)
            for key, (c, y0, ls) in sty.items():
                ev = col[key][i]
                for o, p, e in zip(ev["onset_raw"], ev["peak_raw"], ev["end_raw"]):
                    ax.axvspan(o / 240, e / 240, ymin=y0, ymax=y0 + 0.07, color=c, alpha=0.45, lw=0)
                    ax.axvline(p / 240, color=c, lw=0.9, ls=ls)
                    ax.plot([o / 240], [y0 - 0.01], transform=ax.get_xaxis_transform(), marker="v", ms=6, color=c)
                ax.text(1.002, y0 + 0.03, f"{key} ({ev['peak_raw'].size})", transform=ax.transAxes, color=c, fontsize=9)
            ax2 = ax.twinx()
            ax2.plot(ts, col["act"][i], color="tab:purple", lw=1.0)
            ax2.set_ylim(-0.05, 3.2)
            ax2.set_yticks([0, 0.5, 1])
            ax2.axhline(0.5, color="tab:purple", lw=0.5, ls=":")
            ax.set_xlim(0, 22)
            ax.set_title(f"{seg.cohort[i]}  {seg.guid[i][:8]}  code {seg.code[i]}  {seg.hours_before()[i]:.1f} h  "
                         f"UA valid {v.mean():.0%}  (strips at bottom: blue = shipped reward, red = v7 reward; "
                         f"purple line = causal activity, right axis)", fontsize=9, loc="left")
        axs[-1].set_xlabel("minutes (v = onsets; bars = onset..end; blue shipped, red v7, orange reference)")
        fig.tight_layout(rect=(0, 0, 0.95, 1))
        p = Path(f"{stem}_{page + 1}.png")
        fig.savefig(p, dpi=55)
        plt.close(fig)
        paths.append(str(p))
    return paths


# =============================================================================
# Main
# =============================================================================
def main(args: argparse.Namespace) -> Dict[str, Any]:
    """Tune on code 0, validate on code 2, and write ``out/detectors/improved.json``."""
    IO.OUT_DIR.mkdir(parents=True, exist_ok=True)
    seg = IO.load_segments((0, 2), with_fhr=True)
    report: Dict[str, Any] = {"v7_params": {k: getattr(D, k) for k in dir(D) if k.startswith("V7_")},
                              "activity_params": {"theta": D.ACT_THETA, "slope": D.ACT_SLOPE,
                                                  "lowpass_hz": D.ACT_LOWPASS_HZ, "base_window_s": D.ACT_BASE_WINDOW_S}}
    if not args.skip_grid:
        report["grid_code0"] = grid(seg)
        print("grid done", flush=True)
    col = collect(seg)
    print("collected", flush=True)
    for label, code in (("code0", 0), ("code2", 2)):
        rows = np.where(seg.code == code)[0]
        sub = seg.subset(rows)
        report[label] = {
            "rates_shipped": C.rates(seg, col["ship"], rows=rows),
            "rates_v7": C.rates(seg, col["v7"], rows=rows),
            "rate_v7_minus_shipped_per10min": rate_diff(seg, col, rows),
            "shapes_shipped": C.shapes(seg, col["ship"], rows),
            "shapes_v7": C.shapes(seg, col["v7"], rows),
            "walk_bound_shipped": walk_bound(col, rows, "ship", 60.0, seg),
            "walk_bound_v7": walk_bound(col, rows, "v7", 120.0, seg),
            "recall_ref20_offedge_all": paired_recall(seg, col, rows, valid_only=False),
            "recall_ref20_offedge_le8s_invalid": paired_recall(seg, col, rows, valid_only=False, max_invalid_s=8.0),
            "recall_ref20_valid_span": paired_recall(seg, col, rows, valid_only=True),
            "precision_ref20": precision(seg, col, rows, "ref"),
            "precision_ref12": precision(seg, col, rows, "ref12"),
            "v7_vs_ref20": C.reference_agreement(seg, {"ref": col["ref"], "ship": col["v7"]}, rows, with_reasons=False),
            "v7_validity_drop_share": drop_share(seg, col, rows),
            "weights_and_activity": weights(seg, col, rows),
            "overlap_v7": C.recut_consistency(sub, v7, native=True),
            "decel_v7": decel_block(seg, col, rows),
            "decel_v7_overlap": recut_generic(sub, sub.fhr, sub.fhr_valid,
                                              lambda f, fv: D.detect_decelerations_v7(f, valid=fv), "nadir_raw"),
        }
        print(label, "done", flush=True)
    report["activity_causality"] = activity_causality(seg, np.where(seg.code == 0)[0])
    prow = C.pick_panel_rows(seg, args.n_panels, args.seed, [2])
    report["panels_contractions_v7"] = panels(seg, col, prow, IO.OUT_DIR / "panels_v7_code2")
    drow = C.pick_panel_rows(seg, 12, args.seed + 1, [2])
    dcol = {"ship": col["dec_v7"], "base": col["base"],
            "ref": [{"onset_raw": np.zeros(0, int), "nadir_raw": np.zeros(0, int), "end_raw": np.zeros(0, int)}] * len(seg),
            "ctx_ship": col["v7"]}
    report["panels_decels_v7"] = DC.panels(seg, dcol, drow, IO.OUT_DIR / "panels_decels_v7_code2", name="v7")
    path = IO.OUT_DIR / "improved.json"
    path.write_text(json.dumps(IO.to_jsonable(report), indent=1))
    print("wrote", path)
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line, defaulting to :data:`RUN_ARGS`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-grid", dest="skip_grid", action="store_true", default=RUN_ARGS["skip_grid"])
    parser.add_argument("--n-panels", dest="n_panels", type=int, default=RUN_ARGS["n_panels"])
    parser.add_argument("--seed", type=int, default=RUN_ARGS["seed"])
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
