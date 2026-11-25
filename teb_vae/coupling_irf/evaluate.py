"""Acceptance bench of CouplingIRF: tests A1..A7 on one step cache (one validation fold, or all of it).

    python -m teb_vae.coupling_irf.evaluate --ckpt tmp/coupling_irf/runs/v2/model.pt \
        --cache tmp/coupling_irf/steps_train.npz --fold 0 --out tmp/coupling_irf/runs/v2/eval_val [--limit N]

- One row = one recording (data.stitch, as in train.py); anchors are the steps 60 .. L - 31 of its timeline.
  A "valid" anchor has its UA covered (out["ua_ok"]). The IRF is the ungated h; its delay (irf_delay) is read
  only at valid anchors with gate a > 0.
- Unit = recording (GUID): anchors are pooled within the recording (mean; median for delays), then statistics
  run across recordings. CIs: percentile bootstrap over recordings, 1,000 draws, seed 0.
- Seconds are on the stored clock (UA is stored 20 s early): recorded delay = stored delay - 20 s.
- The anchor at recording step p is -(t0 + 4 p) seconds before delivery (t0 < 0; data.stitch).
- Cache step t is the mean of the 4 s that END at epoch + 60 + 4 t (data.py): its centre is at step coordinate
  t - 15/32 of the event tables (checked: UA crest in the cache - step_peak = +0.47 steps).
"""
import argparse
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import guid_folds, load_steps, stitch
from .model import FHR_SCALE, STEP_S, UA_SCALE, CouplingIRF, gains, irf_delay, plant
from .train import batches, take, to_device

warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices are expected (recordings without a valid anchor)
SEED, N_BOOT, OFFSET = 0, 1000, 15 / 32
MODES = ("expl", "fore")
PLANT = ([(d, 20.0, 2.0, 1.0) for d in (2, 6, 10, 14, 18, 22)] + [(10, x, 2.0, 1.0) for x in (5.0, 10.0, 40.0)]
         + [(10, 20.0, s, 1.0) for s in (1.5, 4.0)] + [(10, 25.0, 2.0, 0.3)])   # (delay steps, depth bpm, sigma steps, share)
SHIFTS = (-4, -2, 0, 2, 4)
CTX = {"all anchors": (0, 1e9), "15-30 min of context": (225, 450), "30-60 min": (450, 900), "> 60 min": (900, 1e9)}  # steps
TIME_BINS = (0, 30, 60, 120, 240)                                    # minutes before delivery
QUANT = ("gain_expl", "gain_fore", "spec_expl", "a", "delay")
mean, med = (lambda v: v.mean(0)), (lambda v: np.median(v, 0))


def boot(x, stat=mean):
    """stat over recording-level rows of x (rows with a NaN dropped) and its percentile-bootstrap CI."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x) if x.ndim == 1 else np.isfinite(x).all(1)]
    if len(x) < 3:
        return {"est": None, "lo": None, "hi": None, "n": len(x)}
    idx = np.random.default_rng(SEED).integers(0, len(x), (N_BOOT, len(x)))
    lo, hi = np.nanpercentile(np.array([stat(x[i]) for i in idx]), [2.5, 97.5], axis=0)
    return {"est": stat(x), "lo": lo, "hi": hi, "n": len(x)}


def corr(x, rank=False):
    """Pearson (or Spearman, average ranks for ties) correlation of the two columns of x."""
    x = pd.DataFrame(x)
    return (x.rank() if rank else x).corr().iloc[0, 1] if len(x) > 2 else np.nan


def auc(score, label):
    """P(a positive scores above a negative), ties count 1/2. NaN when a class is empty."""
    r, n1 = pd.Series(score).rank().to_numpy(), label.sum()
    n0 = len(label) - n1
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def nadir(trace):
    """Index (parabolic refinement) and value of the minimum of each row of trace (n, W). The index is NaN
    when the minimum is not negative or sits on the window edge."""
    i, n = trace.argmin(1), trace.shape[1]
    y0, y1, y2 = (np.take_along_axis(trace, np.clip(i + k, 0, n - 1)[:, None], 1)[:, 0] for k in (-1, 0, 1))
    curv = y0 - 2 * y1 + y2
    return np.where((i > 0) & (i < n - 1) & (y1 < 0), i + 0.5 * (y0 - y2) / np.where(curv > 0, curv, np.inf), np.nan), y1


def fmt(v):
    if isinstance(v, dict):
        return "not enough data" if v["est"] is None else f"{fmt(v['est'])} [{fmt(v['lo'])}, {fmt(v['hi'])}]"
    if isinstance(v, (bool, np.bool_)):
        return "PASS" if v else "FAIL"
    if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
        return "not enough data"
    return f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v)


def table(rows):
    cols = list(rows[0])
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
                     + ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows])


def pick_donors(guid, sub, rng):
    """Per recording, a random other recording of the same subgroup; -1 if the subgroup has one recording."""
    donor = np.full(len(guid), -1)
    for s in np.unique(sub):
        idx = np.flatnonzero(sub == s)
        for _ in range(50):                                          # redraw the rows that drew themselves
            todo = idx[donor[idx] < 0]
            cand = rng.choice(idx, len(todo))
            ok = guid[cand] != guid[todo]
            donor[todo[ok]] = cand[ok]
    return donor


def shift(ua, ok, s):
    """ua'(t) = ua(t - s); the exposed edge repeats the edge value and is marked invalid."""
    src, T = torch.arange(ua.shape[1], device=ua.device) - s, ua.shape[1]
    return ua[:, src.clamp(0, T - 1)], ok[:, src.clamp(0, T - 1)] * ((src >= 0) & (src < T))


@torch.no_grad()
def collect(model, rec, donor, ev, device):
    """All passes, batch of recordings by batch. Returns numpy arrays with one row per recording (own_* / wrong_*
    are (G, L) per timeline step, NaN where the step is no valid anchor) and one row per contraction of `ev`."""
    (G, Lmax), A0, H, nan = rec["fhr"].shape, model.first_anchor, model.H, float("nan")
    data, store, acc = to_device(rec, device), {}, defaultdict(list)
    don = torch.as_tensor(np.maximum(donor, 0), device=device)       # dummy donor 0 where none; blanked in main
    win, base = torch.arange(-8, 33, device=device), torch.arange(-23, -8, device=device)
    where = torch.zeros(G, dtype=torch.long, device=device)          # row of a recording in the current batch
    gen = torch.Generator().manual_seed(SEED)
    share = lambda x, m: torch.where(m, x.float(), nan).nanmean(1)

    def run(st, ua, ok):
        out = model.ua_side(st, ua, ok)
        a = out["a"].masked_fill(~out["ua_ok"], nan)
        return out, {**gains(model, st, out), "a": a, "delay": irf_delay(out["h"], model.knots)[0].masked_fill(~(a > 0), nan)}

    def put(k, x, width=None):                                       # rows ib of a (G, width, ...) array
        x = x.float().cpu().numpy()
        store.setdefault(k, np.full((G, width or x.shape[1], *x.shape[2:]), np.nan, np.float32))[ib, :x.shape[1]] = x

    for n, idx in enumerate(batches(data["length"], 12 if device == "cuda" else 4)):
        ib, idx = idx.numpy(), idx.to(device)
        b = take(data, idx)
        L = b["fhr"].shape[1]
        where[idx] = torch.arange(len(idx), device=device)
        p = torch.arange(A0, L - H, device=device)                   # anchor steps
        ctx = [(p >= lo) & (p < hi) for lo, hi in CTX.values()]
        by_ctx = lambda x: torch.stack([torch.where(m, x, nan).nanmedian(1).values for m in ctx], -1)   # (B, A) -> (B, 4)
        st = model.fhr_side(b["fhr"], b["fhr_sd"], b["fhr_ok"])
        out, own = run(st, b["ua"], b["ua_ok"])                       # A1
        ua_d, ok_d = data["ua"][don[idx], :L], data["ua_ok"][don[idx], :L]   # donor cropped / padded (ua_ok = 0) to L
        out_w, wrong = run(st, ua_d, ok_d)                            # A2
        for k in own:
            put("own_" + k, F.pad(own[k], (A0, H), value=nan), Lmax), put("wrong_" + k, F.pad(wrong[k], (A0, H), value=nan), Lmax)
        open_ = (own["a"] > 0)[..., None]                             # A7 kernel: sum of h over the anchors with a > 0
        put("hsum", (out["h"] * open_).sum(1)), put("hn", open_.sum(1))
        # A5: predicted (ungated h) and observed response around each contraction peak, IRF of the anchor at the peak
        c = ev[np.isin(ev.gid.to_numpy(), ib)]
        row, tl = (torch.tensor(c[k].to_numpy(), device=device) for k in ("gid", "tl"))
        row, at = where[row][:, None], tl[:, None]
        trace = (out["h"][row[:, 0], tl - A0][:, None] * out["Z"][row, at + win]).sum(-1)       # (n, 41), units of 20 bpm
        vb, vw = st["v"][row, at + base], st["v"][row, at + win]
        obs = st["yn"][row, at + win] - (st["yn"][row, at + base] * vb).sum(1, keepdim=True) / vb.sum(1, keepdim=True)
        for k, v in (("c_idx", torch.tensor(c.index.to_numpy(), device=device)), ("trace", trace), ("c_a", own["a"][row[:, 0], tl - A0]),
                     ("c_amp", (obs * trace * vw).sum(1) / (trace * trace * vw).sum(1))):       # least-squares scale of obs on trace
            acc[k].append(v)
        # A3: plant a response to the donor UA; depth 0 on the first batch must reproduce the wrong-UA pass
        pl = []
        for delay, depth, sigma, sh in PLANT + ([(10, 0.0, 2.0, 1.0)] if n == 0 else []):
            fhr_p = plant(b["fhr"], out_w["u"], torch.full_like(b["fhr"][:, 0], delay), sigma, depth, sh, gen)
            res = run(model.fhr_side(fhr_p, b["fhr_sd"], b["fhr_ok"]), ua_d, ok_d)[1]
            if depth == 0:
                check = max((res[k] - wrong[k]).abs().nan_to_num().max().item() for k in MODES)
                continue
            err = (res["delay"] - delay).abs() * STEP_S               # seconds, at valid anchors with a > 0
            pl.append(torch.cat([by_ctx(err), torch.stack([
                share(err <= 4, torch.isfinite(err)), share(err <= 8, torch.isfinite(err)), share(res["a"] > 0, torch.isfinite(res["a"])),
                res["a"].nanmean(1), res["expl"].nanmean(1), res["fore"].nanmean(1), res["delay"].nanmedian(1).values], -1)], -1))
        put("pl", torch.stack(pl, 1))                                 # (B, conditions, 4 + 7)
        put("shift", torch.stack([by_ctx((own["delay"] if s == 0 else run(st, *shift(b["ua"], b["ua_ok"], s))[1]["delay"])
                                         - own["delay"]) for s in SHIFTS], 1))                 # A4: (B, shifts, 4) steps
        print(f"  batch {n + 1}: {len(idx)} recordings of up to {L} steps", flush=True)
    order = torch.cat(acc.pop("c_idx")).argsort()
    return {**store, **{k: torch.cat(v)[order].float().cpu().numpy() for k, v in acc.items()}}, check


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fold", type=int, default=None, help="validation fold of guid_folds; omit for every recording")
    p.add_argument("--limit", type=int, default=None, help="keep the first N recordings (development)")
    p.add_argument("--events", default="tmp/lag_analysis_369/cache/events/contractions.parquet")
    args = p.parse_args()
    t0, out_dir = time.time(), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = load_steps(args.cache)
    if args.fold is not None:
        d = {k: v[guid_folds(d["guid"]) == args.fold] for k, v in d.items()}
    rec, seg = stitch(d)
    rec = {k: v[:args.limit] for k, v in rec.items()}
    G, Lmax = len(rec["guid"]), int(rec["length"].max())
    rec.update({k: v[:, :Lmax] for k, v in rec.items() if v.ndim == 2})
    ck = torch.load(args.ckpt, map_location=device)
    model = CouplingIRF(**ck["kwargs"]).to(device).eval()
    model.load_state_dict(ck["model"])
    A0, sub = model.first_anchor, rec["subgroup"]
    donor = pick_donors(rec["guid"], sub, np.random.default_rng(SEED))
    has = donor >= 0

    # contractions (A5): p = cache step of the peak inside its segment, tl = the same on the recording's timeline;
    # the window p-8..p+32 and the 15 steps before it must lie inside the segment, and tl must be an anchor
    c = pd.read_parquet(args.events, columns=["guid", "split", "subgroup", "row", "step_peak", "edge", "uncertain",
                                              "censored", "in_window", "d_id", "d_type", "lag_peak_nadir"])
    c = c[(c.split == Path(args.cache).stem.split("_")[-1]) & c.in_window & ~(c.edge | c.uncertain | c.censored)]
    c = c.merge(pd.DataFrame({"subgroup": d["subgroup"], "row": d["row"], "gid": seg["g"], "pos": seg["pos"], "guid_cache": d["guid"]}),
                on=["subgroup", "row"])
    c["p"] = np.round(c.step_peak + OFFSET).astype(int)
    c["tl"] = c.pos + c.p
    ev = c[(c.gid < G) & c.p.between(23, 265) & (c.tl >= A0 + 5)].reset_index(drop=True)

    r, check = collect(model, rec, donor, ev, device)
    for k in r:
        if k.startswith("wrong_") or k == "pl":
            r[k][~has] = np.nan                                      # no donor -> no wrong-UA / planted result
    P = np.arange(Lmax)[None]
    valid = np.isfinite(r["own_a"])

    def stats(m):
        """One row per recording: summaries over its anchors selected by the mask m (G, L)."""
        f = lambda x, fn=np.nanmean: fn(np.where(m, x, np.nan), 1)
        t = {"a": f(r["own_a"]), "open": f(np.where(valid, r["own_a"] > 0, np.nan)), "a_wrong": f(r["wrong_a"]),
             "open_wrong": f(np.where(np.isfinite(r["wrong_a"]), r["wrong_a"] > 0, np.nan)), "delay": f(r["own_delay"], np.nanmedian) * STEP_S}
        for k in MODES:
            t.update({"gain_" + k: f(r["own_" + k]), "wrong_" + k: f(r["wrong_" + k]), "spec_" + k: f(r["own_" + k] - r["wrong_" + k])})
        return pd.DataFrame(t)

    Gs = stats(True)
    desc = {"gain expl (nats/cell)": ("gain_expl", mean), "gain fore (nats/cell)": ("gain_fore", mean),
            "mean gate a": ("a", mean), "median IRF delay (s)": ("delay", med)}
    describe = lambda g, cols=desc: {"GUIDs": int(g.gain_fore.notna().sum()), **{k: boot(g[c], st) for k, (c, st) in cols.items()}}
    groups = [("all", Gs)] + [(s, Gs[sub == s]) for s in sorted(set(sub))]
    R = {"meta": {"ckpt": args.ckpt, "epochs_trained": len(ck.get("log", [])), "cache": args.cache, "fold": args.fold, "guids": G,
                  "valid_anchors": int(valid.sum()), "device": device, "clock": "stored (recorded delay = stored delay - 20 s)"}}

    # ---- A1 own UA
    cols = {**dict(list(desc.items())[:2]), "valid anchors with a > 0": ("open", mean)}
    rows = [{"group": n, **describe(g, cols)} for n, g in groups]
    lows = [rows[0][k]["lo"] for k in list(desc)[:2]]
    R["A1"] = {"title": "coupling with the own UA", "pass": None if None in lows else bool(min(lows) > 0),
               "note": "Pass: both CIs of the row 'all' are above 0.", "tables": {"by subgroup": rows}}

    # ---- A2 wrong UA
    rows = []
    for m in MODES:
        own, wrong = boot(Gs["gain_" + m]), boot(Gs["wrong_" + m])
        rows.append({"mode": m, "own gain": own, "wrong-UA gain": wrong, "specific gain (own - wrong)": boot(Gs["spec_" + m]),
                     "pass": None if None in (own["est"], wrong["lo"]) else bool(wrong["lo"] >= -0.1 * own["est"])})
    gates = [{"gate a": what, "own UA": boot(Gs[key]), "wrong UA": boot(Gs[key + "_wrong"])}
             for what, key in (("mean", "a"), ("share of valid anchors with a > 0", "open"))]
    R["A2"] = {"title": "wrong UA (donor: other GUID, same subgroup)", "tables": {"gain (nats/cell)": rows, "gates": gates},
               "pass": None if None in [x["pass"] for x in rows] else all(x["pass"] for x in rows),
               "note": f"{int(has.sum())} of {G} recordings have a donor (aligned at step 0, cut or padded to the own length). "
                       "Pass: CI lower bound of the wrong-UA gain >= -0.1 x own gain, both modes."}

    # ---- A3 planted delay
    pl, names = r["pl"], list(CTX)                                   # (G, conditions, 4 errors by context + 7)
    cond = [{"delay (s)": dl * STEP_S, "depth (bpm)": dp, "sigma (s)": sg * STEP_S, "responding share": sh} for dl, dp, sg, sh in PLANT]
    rows = [{**cond[k], "median abs error (s)": boot(pl[:, k, 0], med), **{lab: np.nanmean(pl[:, k, 4 + j]) for j, lab in enumerate(
        ("within 4 s", "within 8 s", "detected (a > 0)", "gate a", "gain expl", "gain fore"))}} for k in range(len(PLANT))]
    by_ctx = [{**cond[k], **{names[j]: boot(pl[:, k, j], med) for j in (1, 2, 3)}} for k in range(len(PLANT))]
    e20 = [x["median abs error (s)"]["est"] for x in rows[:6]]
    R["A3"] = {"title": "planted delay", "pass": None if None in e20 else bool(max(e20) <= 4), "depth0_check_max_abs_diff": check,
               "note": "Delays are read at all valid anchors with a > 0. Error: median over the anchors of a recording, then median "
                       "over recordings; 'within' = share of those anchors; 'detected' = share of the valid anchors with a > 0. Depth "
                       "= response to the donor recording's largest contraction. Pass: median error <= 4 s for the six delays at "
                       f"depth 20. Self-check: planting depth 0 changes the wrong-UA gain by {check:.2g} nats (must be 0).",
               "tables": {"by condition": rows, "median abs error (s) by context before the anchor": by_ctx}}

    # ---- A4 shifted UA
    slope = lambda v: np.polyfit(SHIFTS, np.median(v, 0), 1)[0]
    slopes = [{"anchors": lab, "slope (expected -1)": boot(r["shift"][:, :, j], slope)} for j, lab in enumerate(names)]
    rows = [{"shift s (s)": s * STEP_S, "expected change (s)": -s * STEP_S,
             **{lab: boot(r["shift"][:, k, j] * STEP_S, med) for j, lab in enumerate(names)}} for k, s in enumerate(SHIFTS)]
    q, est = np.isfinite(r["own_delay"]), slopes[0]["slope (expected -1)"]["est"]
    R["A4"] = {"title": "shifted UA", "pass": None if est is None else bool(-1.2 <= est <= -0.8),
               "qualified_anchors": int(q.sum()), "qualified_guids": int(q.any(1).sum()),
               "note": f"{int(q.sum())} of {int(valid.sum())} valid anchors in {int(q.any(1).sum())} of {G} recordings qualified (a > 0 and a "
                       "readable delay, unshifted). Change = median over the anchors of a recording that stay readable, then median "
                       "over recordings; slope over the recordings readable at all shifts. Pass: slope of 'all anchors' in [-1.2, -0.8].",
               "tables": {"slope of the change on s": slopes, "median change of the IRF delay (s)": rows}}

    # ---- A5 real contractions
    spearman = lambda v: corr(v, rank=True)
    arg, low = nadir(r["trace"])
    ev["pred"] = STEP_S * (arg - 8 + ev.p - OFFSET - ev.step_peak)             # predicted nadir - peak, seconds
    ev["depth"], ev["amp"], ev["has"] = np.maximum(-low, 0) * FHR_SCALE, r["c_amp"], ev.d_id >= 0   # predicted drop, bpm (0 = none)
    ev["abs_err"] = (ev.pred - ev.lag_peak_nadir).abs()
    ev, n_all = ev[r["c_a"] > 0], len(ev)
    pair, amp = ev.dropna(subset=["pred", "lag_peak_nadir"]), ev.dropna(subset=["amp"])
    within = lambda t, col: boot([auc(g[col].to_numpy(), g.has.to_numpy()) for _, g in t.groupby("gid")])
    summary = {
        "contractions": n_all, "with a > 0 at the peak (used below)": len(ev), "their GUIDs": ev.gid.nunique(),
        "with a paired deceleration": int(ev.has.sum()), "GUID of the table = GUID of the cache": (ev.guid == ev.guid_cache).mean(),
        "predicted trace has a nadir in the window": ev.pred.notna().mean(), "paired, timed and predicted": len(pair),
        "observed lag inside -32..128 s": pair.lag_peak_nadir.between(-32, 128).mean(),
        "Spearman over contractions": spearman(pair[["pred", "lag_peak_nadir"]]),
        "Spearman over GUID medians": boot(pair.groupby("gid")[["pred", "lag_peak_nadir"]].median().to_numpy(), spearman),
        "median abs error (s)": boot(pair.groupby("gid").abs_err.median(), med),
        "AUC of predicted depth for 'has a deceleration', pooled": auc(ev.depth.to_numpy(), ev.has.to_numpy()),
        "AUC of predicted depth, within GUID": within(ev, "depth"),
        "measured amplitude: median with / without a deceleration": f"{fmt(amp.amp[amp.has].median())} / {fmt(amp.amp[~amp.has].median())}",
        "AUC of measured amplitude for 'has a deceleration', pooled": auc(amp.amp.to_numpy(), amp.has.to_numpy()),
        "AUC of measured amplitude, within GUID": within(amp, "amp")}
    types = [{"d_type": t, "contractions": len(g), "GUIDs": g.gid.nunique(),
              "predicted median (s)": boot(g.groupby("gid").pred.median(), med),
              "observed median (s)": boot(g.groupby("gid").lag_peak_nadir.median(), med)} for t, g in pair.groupby("d_type")]
    R["A5"] = {"title": "real contractions", "pass": None,
               "note": "Predicted trace = (h . Z) in steps peak-8..peak+32 with the ungated IRF h of the anchor at the peak. Predicted "
                       "delay = its nadir minus the contraction peak (stored s); depth = its drop below 0 (bpm). Measured amplitude = "
                       "least-squares scale of the observed FHR (minus its mean over the 15 steps before the window, valid steps) on "
                       "the trace: about 1 = the typical response, 0 = none. No threshold.",
               "tables": {"summary": [{"quantity": k, "value": v} for k, v in summary.items()],
                          **({"by deceleration type (GUID medians)": types} if types else {})}}

    # ---- A6 split-half reliability
    rank, n_valid = valid.cumsum(1), valid.sum(1)
    long = n_valid >= 600
    splits = {"first half vs second half of the valid anchors": (valid & (rank <= n_valid[:, None] // 2), valid & (rank > n_valid[:, None] // 2)),
              "even vs odd 10-minute blocks": (P // 150 % 2 == 0, P // 150 % 2 == 1)}
    R["A6"] = {"title": "split-half reliability", "tables": {},
               "note": f"{int(long.sum())} recordings with >= 600 valid anchors. Pass: Pearson of the expl gain, first against "
                       f"second half, > 0.45. The IRF at an anchor is fitted on the last {model.context_steps * STEP_S / 60:.0f} min, so "
                       "the halves share context (the blocks more than the halves)."}
    for name, halves in splits.items():
        h1, h2 = (stats(m)[long] for m in halves)
        R["A6"]["tables"][name] = [{"recording-level quantity": k, "Pearson": boot(np.c_[h1[k], h2[k]], corr),
                                    "Spearman": boot(np.c_[h1[k], h2[k]], spearman)} for k in QUANT]
    est = R["A6"]["tables"][next(iter(splits))][0]["Pearson"]["est"]
    R["A6"]["pass"] = None if est is None else bool(est > 0.45)

    # ---- A7 descriptive
    tmin = -(rec["t0"][:, None] + STEP_S * P) / 60                    # minutes before delivery
    K = r["hsum"] / r["hn"] @ model.basis.cpu().numpy() * FHR_SCALE / UA_SCALE   # (G, delays 0..120 s), bpm per toco unit
    kcls = np.array([s.split("_")[0] for s in sub])
    R["A7"] = {"title": "descriptive", "pass": None, "note": "IRF delay: valid anchors with a > 0. No threshold.",
               "kernel_bpm_per_toco_unit": {c: boot(K[kcls == c]) for c in ("healthy", "acidosis", "hie")},
               "tables": {"by time to delivery": [{"minutes before delivery": f"{a}-{b}", **describe(stats((tmin >= a) & (tmin < b)))}
                                                  for a, b in zip(TIME_BINS, TIME_BINS[1:])],
                          "by subgroup": [{"group": n, **describe(g)} for n, g in groups]}}

    # ---- figures
    fig, ax = plt.subplots(figsize=(5, 4))
    lo, mid, hi = np.nanpercentile(pl[:, :6, -1].astype(float) * STEP_S, [25, 50, 75], axis=0)   # recording medians, six delays
    x = np.array([c[0] for c in PLANT[:6]]) * STEP_S
    ax.errorbar(x, mid, yerr=[mid - lo, hi - mid], fmt="o", capsize=3, label="median and IQR over recordings")
    ax.plot(x, x, "k--", lw=0.8, label="identity")
    ax.set(xlabel="planted delay (stored s; depth 20 bpm, sigma 8 s)", ylabel="IRF delay read back (stored s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "A3_planted_delay.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    lag = np.arange(K.shape[1]) * STEP_S
    for i, (c, kb) in enumerate(R["A7"]["kernel_bpm_per_toco_unit"].items()):
        if kb["est"] is not None:                                    # a class without 3 recordings is not drawn
            ax.fill_between(lag, kb["lo"], kb["hi"], alpha=0.2, color=f"C{i}")
            ax.plot(lag, kb["est"], color=f"C{i}", label=f"{c} ({kb['n']} GUIDs)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="delay after UA (stored s)", ylabel="mean IRF h (bpm per toco unit per 4-s step)")
    ax.secondary_xaxis("top", functions=(lambda v: v - 20, lambda v: v + 20)).set_xlabel("delay after UA (recorded s)")
    ax.legend(title="valid anchors with gate a > 0; band = 95 % CI")
    fig.tight_layout()
    fig.savefig(out_dir / "A7_kernel_by_class.png", dpi=150)

    # ---- write (json: numpy -> lists, NaN -> null)
    R["meta"]["seconds"] = round(time.time() - t0, 1)
    js = json.dumps(R, default=lambda o: o.tolist())
    (out_dir / "acceptance.json").write_text(json.dumps(json.loads(js, parse_constant=lambda _: None), indent=1))
    tests = {k: v for k, v in R.items() if k != "meta"}
    md = [f"# Acceptance of CouplingIRF\n\n`{args.ckpt}` ({R['meta']['epochs_trained']} epochs in its log) on `{args.cache}`, "
          f"fold {args.fold}: {G} recordings (GUIDs), {int(valid.sum())} valid anchors. Unit = recording; brackets = 95 % bootstrap "
          "CI over recordings. Seconds are on the stored clock (recorded delay = stored delay - 20 s). Gains are nats per cell.\n",
          table([{"test": k, "what": v["title"], "verdict": "no threshold" if k in ("A5", "A7") else v["pass"]}
                 for k, v in tests.items()])]
    for k, v in tests.items():
        md.append(f"\n## {k} {v['title']}\n\n{v['note']}")
        md += [f"\n{cap}\n\n{table(rows)}" for cap, rows in v["tables"].items()]
    (out_dir / "ACCEPTANCE.md").write_text("\n".join(md) + "\n")
    print(md[1], f"\n{R['meta']['seconds']} s -> {out_dir}")


if __name__ == "__main__":
    main()
