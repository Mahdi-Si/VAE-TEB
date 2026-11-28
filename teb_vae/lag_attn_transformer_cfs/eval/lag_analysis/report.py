"""Statistics, tables and figures of the lag analysis, from `<out>/steps/*.npz` alone.

Unit: the recording (GUID). An estimate is the equal-weight mean of the subgroup means of the GUID
values (`pooled`), as registered in the study. Interval: percentile interval of a GUID-cluster bootstrap,
2,000 draws, stratified by subgroup, fixed seed.
Differs from the study: the study used BCa intervals (widened by a t factor for strata under 50 GUIDs)
and, for donor rows, a two-way bootstrap over recipient and donor GUIDs. The intervals here are a
little narrower than those for donor rows.
"""
from __future__ import annotations

import json
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np

from .core import CLASSES, STEP_S, TIME_BINS, class_of
from .steps import AGES, GATE_OPEN, KAPPA_SHIFTS, L0, PLANT_STEPS, SHIFTS

N_BOOT = 2000
MIN_EVENTS = 5
BIN_NAMES = [f"{int(a)}-{int(b)} min" for a, b in zip(TIME_BINS[:-1], TIME_BINS[1:])]
CLOCK = "Seconds are on the stored clock of the shards."

# Reference: checkpoint 369, lag study. Value, 95 % interval, n GUIDs, row of REF_SOURCE. The study's block
# was the classes acidosis + HIE (here `adverse`) unless the row says otherwise. Provenance only.
REF_SOURCE = "tmp/lag_analysis_369/REPORT.md"
REF = {
    "gain.sg_rw": (0.01316, [0.01066, 0.01627], 580, "F1 / R04"),
    "gain.sg_surB_rw": (0.0130, [0.0108, 0.0161], None, "F1 / R04 control"),
    "shift.F8": (0.885, [0.866, 0.911], 570, "F5 / R33"),
    "shift.F32": (0.132, [0.055, 0.201], 570, "F5 / R33"),
    "shift.W50": (19.5, [17.8, 21.5], 570, "F5 / R33"),
    "shift.delta0": (-3.2, [-4.3, -1.8], None, "section 4.2 (other onset detector, all classes)"),
    "shift.kappa_h0": (0.084, [0.081, 0.088], 591, "F3 / R01"),
    "shift.kappa_h1": (0.063, [0.061, 0.066], 591, "F3 / R01"),
    "shift.kappa_h2": (0.161, [0.156, 0.168], 591, "F3 / R01"),
    "shift.kappa_h3": (0.326, [0.319, 0.333], 591, "F3 / R01"),
    "heads.kl_share_main": (0.963, [0.961, 0.966], 482, "F2 / R03"),
    "heads.shapley_main": (0.994, [0.993, 0.995], 470, "F2 / R11 (K = 8 samples)"),
    "heads.elag_h0": (3.8, None, None, "F10"), "heads.elag_h1": (4.5, None, None, "F10"),
    "heads.elag_h2": (20.7, None, None, "F10"), "heads.elag_h3": (4.4, None, None, "F10"),
    "heads.short_mass_main": (0.90, None, None, "F10"),
    "heads.short_loss": (0.0125, [0.0102, 0.0156], None, "F10 / R08"),
    "heads.short_loss_share": (0.95, None, None, "F10 / R08"),
    "heads.drift_main": (-0.019, [-0.207, 0.229], 592, "F10 / R07"),
    "clamp.pos": (0.998, [0.968, 1.048], 138, "F4 / R44"),
    "clamp.static": (0.953, [0.911, 1.012], 138, "F4 / R44"),
    "clamp.rev": (0.13, None, 138, "F4 / R44, upper bound"),
    "clock.omega_gate": (0.873, [0.840, 0.907], 588, "F8 / R34"),
    "clock.omega_kl": (0.803, [0.769, 0.839], 588, "F8 / R34"),
    "clock.omega_lag": (0.609, [0.566, 0.651], 588, "F8 / R34"),
    "clock.dip": (-0.0062, [-0.0087, -0.0042], 445, "F8 / R38"),
    "clock.dip_vs_own": (-0.0060, [-0.0084, -0.0035], 445, "F8 / R38 control"),
    "events.peak": (27.8, [24.9, 44.3], 445, "F12 / R37"),
    "events.first_open": (16.0, None, None, "section 4.1 (other onset detector, onsets about 5 s later; gate threshold 0.37)"),
    "events.never_open": (0.14, [0.10, 0.18], None, "section 4.1, range over strata (gate threshold 0.37)"),
    "groups.slow": (0.072, [0.014, 0.138], 137, "F6 / R50"),
    "groups.gate_open.hie-acidosis": (-0.0001, [-0.022, 0.022], None, "F15 / R32 (gate threshold 0.37, covariate-adjusted)"),
    "raw.nadir": (28.0, [27.0, 29.0], None, "section 1 and R28, all classes (other detector, 1-s grid)"),
    "raw.depth": (-4.4, None, None, "R28 binding text, at the nadir"),
}
ARMS = (  # first matching prefix names the arm(s) a statistic is computed from
    ("gain.sg_surB", "own UA minus surB"), ("gain.sg", "own UA minus surA"), ("gain.gain_matched", "own UA"),
    ("gain.gain_surA", "surA"), ("gain.gain_surB", "surB"), ("gain.gain_null", "zeroed UA"),
    ("gain.kl_specific", "own UA minus surA"), ("gain.kl_matched", "own UA"), ("gain.kl_surA", "surA"),
    ("gain.kl_surB", "surB"), ("gain.kl_null", "zeroed UA"),
    ("shift.kappa", "own UA and shifted own UA (attention)"), ("shift.", "shifted own UA / own UA"),
    ("heads.shapley", "own UA minus surA, heads switched off"), ("heads.loo", "own UA minus surA, heads switched off"),
    ("heads.only", "own UA minus surA, heads switched off"), ("heads.short_loss", "own UA minus surA, short lags removed"),
    ("heads.long_loss", "own UA, long lags removed"), ("heads.kl_null", "zeroed UA / own UA"), ("heads.", "own UA"),
    ("clamp.", "own UA with clamped attention / own UA"),
    ("clock.dip_vs_own", "surA minus own UA, donor's clock"), ("clock.dip", "surA, donor's clock"),
    ("clock.gate_rise", "surA, donor's clock"), ("clock.", "surA, own UA and the donor's own run"),
    ("events.peak_specific", "own UA minus surA"), ("events.", "own UA"),
    ("groups.slow", "own UA with slow channels from surA"), ("groups.sg", "own UA minus surA"), ("groups.", "own UA"),
    ("irf.plant", "IRF head, response planted after surA's UA"), ("irf.r_", "IRF head and lag attention, own UA"),
    ("irf.", "IRF head, own UA"), ("raw.", "raw signals, no model"),
)
HEADLINE = ["gain.sg_rw", "gain.sg_all", "gain.sg_outside_rw", "gain.sg_surB_rw", "shift.F8", "shift.F32", "shift.W50", "shift.kappa_h0", "shift.kappa_h1",
            "shift.kappa_h2", "shift.kappa_h3", "heads.kl_share_main", "heads.shapley_main", "heads.short_loss_share",
            "clamp.pos", "clamp.static", "clamp.rev", "clock.omega_gate", "clock.omega_kl", "clock.omega_lag",
            "clock.dip", "events.peak", "events.peak_0_180", "groups.slow", "raw.nadir", "raw.depth"]
READ = {
    "gain": "The specific gain is what the right UA adds over a wrong one. A value near zero with an interval "
            "that covers zero means the model does not use the recording's own UA for its forecast.",
    "shift": "F is the share of the gain kept when the own UA is moved in time. W50 is the shift that loses "
             "half of it; delta0 is the centre of the tuning curve. A tracking index of 0 is a fixed filter over "
             "lags, 1 an attention that moves with the UA sample.",
    "heads": "The main head is the one with the largest KL. A Shapley share near 1 means it alone carries the "
             "specific gain. A lag index is which stored UA sample is read, not a delay in seconds.",
    "clamp": "A share near 1 for the position-only or the static profile means the forecast does not need the "
             "part of the attention that reads the UA content. The reversed profile is the control: it must lose gain.",
    "clock": "Omega_U near 1 means the anchor-to-anchor course of a quantity follows the UA that is fed, also when "
             "it is another recording's. The dip is the fall of the gain after the donor's contractions.",
    "events": "The gain peak is the age after contraction onset at which own UA helps most. A peak inside 0-60 s "
              "is what the reward window of the model's objective produces; it is not by itself a physiological delay.",
    "groups": "Class rows compare GUID means between classes inside each delivery mode. The healthy placebo "
              "(with against without a blood-gas sample) shows how much a difference of selection alone gives.",
    "irf": "The gate of the IRF head should open with the own UA and stay shut with a wrong one. The gaps are "
           "the forecast gain of the head's mean shift. The planted rows check that a known delay is read back.",
    "raw": "A model-free yardstick: where the FHR is lowest after the UA peak of a contraction, in the raw "
           "signals. The model's timings can be read against it.",
}
SCOPE = ("Not in scope of this pipeline: the injected-contraction sweep (R13-R16), the per-dimension latent "
         "ablation, the clinician packet and the outcome-AUROC models (R24). Also not ported: the onset-aligned "
         "surrogate (R04b), the gate-forcing rows (R12, R39), the synthetic-UA and lag-window sweeps (R45-R47, R52), "
         "the covariate-adjusted class estimator E with permutation tests, the pre-registered verdict rules and the "
         "multiplicity correction.")
DESCRIPTIVE = (
    "All numbers describe the evaluated set as a whole. No number was selected on one part of the data and confirmed on "
    "another, and nothing was fitted on one part and applied to another. Choices that could otherwise leak: "
    "(1) the position-only attention profile comes from the model's parameters with the keys zeroed, no data; "
    "(2) the static profile needs a mean query: it is averaged over a seeded half of the recordings and applied to the other "
    "half, in both directions, so no recording shapes the profile it is tested with; "
    "(3) the gate counts as open at 0.5, the midpoint of its sigmoid, not at a data quantile; "
    "(4) short lags are lags 0-14, the UA of the last 60 s (the length of the objective's reward window), a constant; "
    "(5) slow UA channels are those with a filter delay of 100 s or more, read from the shard attributes; "
    "(6) surrogate donors are paired inside the evaluated set by time to delivery and UA valid fraction; "
    "(7) the main head is the head with the largest KL on the evaluated set. Its rows carry the suffix `_main`; every "
    "per-head row (`_h0` ...) is given too, so that choice can be redone.")


# ----------------------------------------------------------------------------- pooled statistics
def pooled(v, W, H):
    """Equal-weight mean over subgroups of the W-weighted mean of GUID values v (n,) or (n, P); NaN left out."""
    v = np.asarray(v, float)
    one = v.ndim == 1
    v = v.reshape(len(v), int(np.prod(v.shape[1:])))          # not -1: an empty class has no recording
    ok = np.isfinite(v)
    v0 = np.where(ok, v, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        parts = [((W * H[:, k]) @ v0) / ((W * H[:, k]) @ ok) for k in range(H.shape[1])]
        out = np.nanmean(np.stack(parts), 0) if parts else np.full((len(W), v.shape[1]), np.nan)
    return out[:, 0] if one else out


def wmedian(v, W):
    r"""Weighted median of the finite values v (n,) under each row of weights W (draws, n).

    With cumulative weights $c_j$ of the sorted values and total $c_n$, the median is the midpoint
    $\frac{1}{2}(v_{(j_-)} + v_{(j_+)})$ of the first order statistic with $c_j \ge c_n / 2$ and the first
    with $c_j > c_n / 2$, so equal weights on an even count give $\frac{1}{2}(v_{(n/2)} + v_{(n/2+1)})$.

    Returns:
        (draws,) medians; NaN for a row without positive weight, or everywhere when no value is finite.
    """
    ok = np.isfinite(v)
    v, W = v[ok], W[:, ok]
    if not v.size:
        return np.full(len(W), np.nan)
    o = np.argsort(v)
    vs, c = v[o], np.cumsum(W[:, o], 1)
    half = c[:, -1:] / 2
    mid = 0.5 * (vs[(c >= half).argmax(1)] + vs[(c > half).argmax(1)])
    return np.where(c[:, -1] > 0, mid, np.nan)


def parabolic_peak(Y, x):
    """Y (draws, bins) -> location and height of the 3-point parabolic peak around the maximum.

    The vertex offset is applied only where the parabola opens downwards and both neighbours are
    finite; otherwise the maximum bin itself is returned. A row whose maximum bin is not finite (for
    example a row without a finite entry) gives (NaN, NaN).
    """
    Y = np.atleast_2d(np.asarray(Y, float))
    i = np.clip(np.argmax(np.where(np.isfinite(Y), Y, -np.inf), 1), 1, Y.shape[1] - 2)
    r = np.arange(len(Y))
    y0, y1, y2 = Y[r, i - 1], Y[r, i], Y[r, i + 1]
    den = y0 - 2 * y1 + y2
    fit = (den < 0) & np.isfinite(y0) & np.isfinite(y2)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(fit, np.clip(0.5 * (y0 - y2) / np.where(fit, den, -1.0), -1, 1), 0.0)
        height = np.where(fit, y1 - 0.25 * (y0 - y2) * d, y1)
    found = np.isfinite(y1)
    return np.where(found, x[i] + d * (x[1] - x[0]), np.nan), np.where(found, height, np.nan)


def w50(Y, xs):
    """First crossing of 0.5 by the line through (0, 1) and (xs, Y) per draw; the last x if never
    crossed; NaN for a draw without a finite share."""
    empty = ~np.isfinite(np.asarray(Y, float)).any(1)
    Y = np.column_stack([np.ones(len(Y)), Y])
    xs = np.r_[0.0, xs]
    out, done = np.full(len(Y), xs[-1]), np.zeros(len(Y), bool)
    for k in range(1, Y.shape[1]):
        with np.errstate(invalid="ignore", divide="ignore"):
            t = xs[k - 1] + (Y[:, k - 1] - 0.5) / (Y[:, k - 1] - Y[:, k]) * (xs[k] - xs[k - 1])
        hit = ~done & (Y[:, k] < 0.5)
        out[hit], done = t[hit], done | hit
    out[empty] = np.nan
    return out


def shapley(v):
    """Exact Shapley values over M heads; v (..., 2**M) indexed by the bitmask of heads switched off."""
    M = int(np.log2(v.shape[-1]))
    full, phi = 2 ** M - 1, np.zeros(v.shape[:-1] + (M,))
    for h in range(M):
        others = [m for m in range(M) if m != h]
        for k in range(M):
            for S in combinations(others, k):
                on = sum(1 << m for m in S)
                phi[..., h] += factorial(k) * factorial(M - k - 1) / factorial(M) * (v[..., full ^ (on | 1 << h)] - v[..., full ^ on])
    return phi


class View:
    """The GUIDs of one step inside one block, with their bootstrap weights (row 0 = the sample itself)."""

    def __init__(self, d, mask=None, seed=0):
        mask = np.ones(len(d["guid"]), bool) if mask is None else mask
        self.d = {k: v[mask] for k, v in d.items() if isinstance(v, np.ndarray) and v.shape[:1] == mask.shape}
        self.sub = self.d["subgroup"]
        self.n = int(mask.sum())
        names, code = np.unique(self.sub, return_inverse=True)
        self.H = np.eye(len(names))[code] if self.n else np.zeros((0, 1))
        rng = np.random.default_rng(seed)
        W = np.ones((N_BOOT + 1, self.n), np.float32)
        for k in range(len(names)):
            idx = np.flatnonzero(code == k)
            W[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), N_BOOT)
        self.W = W

    def __getitem__(self, k):
        return self.d[k].astype(float)

    def keep(self, m):
        d = dict(self.d)
        return View(d, np.asarray(m, bool))

    def pool(self, v):
        return pooled(v, self.W, self.H)

    def ratio(self, a, b):
        return self.pool(a) / self.pool(b)

    def median(self, v):
        parts = [wmedian(np.where(self.H[:, k] > 0, v, np.nan), self.W) for k in range(self.H.shape[1])]
        return np.nanmean(np.stack(parts), 0)


def nfin(*vs):
    ok = np.ones(len(vs[0]), bool)
    for v in vs:
        v = np.asarray(v, float)
        ok &= np.isfinite(v.reshape(len(v), int(np.prod(v.shape[1:])))).all(1)
    return int(ok.sum())


class Rows(list):
    """The statistics of a run. Each row: id, step, block, estimate, 95 % interval, n, unit, definition."""

    def put(self, sid, block, vals, n, unit, definition, note=None):
        vals = np.asarray(vals, float)
        fin = vals[1:][np.isfinite(vals[1:])]
        ci = [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))] if fin.size > 10 else None
        ref = REF.get(sid)
        arm = next(a for pre, a in ARMS if sid.startswith(pre))
        if sid.startswith("irf.") and ("_w" in sid.split(".")[1]):
            arm = "IRF head, surA's UA"
        row = {"id": sid, "step": sid.split(".")[0], "arm": arm, "block": block, "estimate": float(vals[0]), "ci95": ci,
               "n_guids": int(n), "unit": unit, "definition": definition, "note": note,
               "reference": None if ref is None else {"label": "reference: checkpoint 369, lag study", "value": ref[0], "ci95": ref[1],
                                                      "n_guids": ref[2], "source": f"{REF_SOURCE}, {ref[3]}"}}
        self.append(row)
        return row


def blocks_of(d):
    """Block name -> GUID mask: `adverse` = classes acidosis + HIE, `healthy` = the two healthy classes, `all`.
    `adverse` and `healthy` are left out when they hold no recording; `all` is always there."""
    cls = np.array([class_of(s)[0] for s in d["subgroup"]], object)
    if any(c is None for c in cls):
        return {"all": np.ones(len(cls), bool)}, None, None
    cs = np.array([class_of(s)[1] for s in d["subgroup"]], bool)
    bl = {"adverse": np.isin(cls, ("acidosis", "hie")), "healthy": np.isin(cls, ("healthy_bg", "healthy_no_bg"))}
    return {**{b: m for b, m in bl.items() if m.any()}, "all": np.ones(len(cls), bool)}, cls, cs


def main_head(steps):
    for s in ("heads", "clock"):
        if s in steps:
            v = View(steps[s])
            return int(np.nanargmax(v.pool(v["kl" if s == "heads" else "klh"])[0]))
    return None


# ----------------------------------------------------------------------------- statistics per step
def stats_gain(d, R, C):
    bl, _, _ = blocks_of(d)
    for b, m in bl.items():
        v = View(d, m)
        sg, sga, sgb = v["am_rw"] - v["as_rw"], v["am_all"] - v["as_all"], v["bm_rw"] - v["bs_rw"]
        R.put("gain.sg_rw", b, v.pool(sg), nfin(sg), "nats per cell",
              "Specific gain: gain with own UA minus gain with another recording's UA (surA), both under the joint "
              "key mask, on reward-window anchors (0-60 s after a contraction onset) with at least one key and at "
              "least half of the scored cells valid; K shared latent samples.")
        R.put("gain.sg_all", b, v.pool(sga), nfin(sga), "nats per cell", "The same on all anchors with at least one key.")
        if "am_out" in d:
            for w, k, txt in (("outside_rw", "out", "anchors outside the reward window (more than 60 s after the latest "
                                                    "contraction onset, or with no onset before)"),
                              ("post_60_180", "post", "anchors 60-180 s after the latest contraction onset"),
                              ("quiet", "quiet", "anchors more than 180 s after the latest contraction onset, or with no onset before")):
                x = v[f"am_{k}"] - v[f"as_{k}"]
                R.put(f"gain.sg_{w}", b, v.pool(x), nfin(x), "nats per cell", f"The same specific gain (surA) on {txt}.")
            sgb_all = v["bm_all"] - v["bs_all"]
            R.put("gain.sg_surB_all", b, v.pool(sgb_all), nfin(sgb_all), "nats per cell", "Specific gain against surB on all anchors with at least one key.")
        R.put("gain.sg_surB_rw", b, v.pool(sgb), nfin(sgb), "nats per cell",
              "Specific gain against the same recording's UA at least 40 min away (surB), reward-window anchors; "
              "recordings with such a segment only.")
        for k, name, txt in (("am_rw", "matched", "own UA, joint mask"), ("as_rw", "surA", "another recording's UA"),
                             ("bs_rw", "surB", "the same recording's UA at least 40 min away"), ("g0_rw", "null", "zeroed UA")):
            R.put(f"gain.gain_{name}_rw", b, v.pool(v[k]), nfin(v[k]), "nats per cell",
                  f"Forecast gain (base NLL minus full NLL per valid scored cell) with {txt}, reward-window anchors.")
            ka = k.replace("_rw", "_all")
            if ka in d:
                R.put(f"gain.gain_{name}_all", b, v.pool(v[ka]), nfin(v[ka]), "nats per cell",
                      f"The same forecast gain with {txt}, all anchors with at least one key.")
        for j, name in enumerate(("scattering", "phase")):
            x = v["am_blk"][:, j] - v["as_blk"][:, j]
            R.put(f"gain.sg_rw_{name}", b, v.pool(x), nfin(x), "nats per cell", f"Specific gain on the {name} target channels only.")
        if "kl" in d:
            for k, name, txt in (("kl", "matched", "own UA"), ("a_kls", "surA", "another recording's UA (joint mask)"),
                                 ("b_kls", "surB", "the same recording's UA 40 min away (joint mask)"), ("kl0", "null", "zeroed UA")):
                R.put(f"gain.kl_{name}", b, v.pool(v[k]), nfin(v[k]), "nats per anchor",
                      f"KL(posterior || prior) per anchor with {txt}, anchors with at least one key.")
            R.put("gain.kl_specific_ratio", b, v.ratio(v["a_klm"] - v["a_kls"], v["a_klm"]), nfin(v["a_klm"], v["a_kls"]), "share",
                  "R02: (KL matched - KL surA) / KL matched, pooled; 0 means a wrong UA costs the full KL.")
        tau = v.pool(v["am_tau"] - v["as_tau"])
        C[f"gain.sg_by_horizon.{b}"] = curve(STEP_S * (1 + np.arange(tau.shape[1])), tau, "s ahead of the anchor", "nats per cell")


def stats_shift(d, R, C):
    bl, cls, _ = blocks_of(d)
    sec = STEP_S * np.array(SHIFTS, float)
    pos, neg = [SHIFTS.index(s) for s in sorted(x for x in SHIFTS if x > 0)], [SHIFTS.index(-s) for s in sorted(x for x in SHIFTS if x > 0)]
    xs = sec[pos]

    def tuning(v, tag=""):
        F = v.pool(v["f_num" + tag]) / v.pool(v["f_den" + tag])[:, None]
        pair = 0.5 * (F[:, pos] + F[:, neg])
        return F, pair, w50(pair, xs), 0.5 * (w50(F[:, pos], xs) - w50(F[:, neg], xs))

    for b, m in bl.items():
        v = View(d, m).keep(np.isfinite(d["f_den"][m]) & np.isfinite(d["f_num"][m]).all(1))
        F, pair, W, d0 = tuning(v)
        for j, x in enumerate(xs):
            R.put(f"shift.F{int(x)}", b, pair[:, j], v.n, "share",
                  f"Share of the gain kept when the own UA is moved by {int(x)} s: mean of the two directions of pooled gain "
                  "under the shift / pooled gain unshifted. Gain at the latent means, reward-window anchors that are "
                  "clean (all keys valid and warm) in every shifted window.")
        R.put("shift.W50", b, W, v.n, "s", "Shift at which half of the gain is lost: first crossing of 0.5 by the line through the measured shares.")
        R.put("shift.delta0", b, d0, v.n, "s", "Centre of the tuning curve: (W50 of delays - W50 of advances) / 2. Positive: the gain tolerates a delayed UA better.")
        if "f_den_all" in d:
            va = View(d, m).keep(np.isfinite(d["f_den_all"][m]) & np.isfinite(d["f_num_all"][m]).all(1))
            _, pair_a, W_a, d0_a = tuning(va, "_all")
            for j, x in enumerate(xs):
                R.put(f"shift.F{int(x)}_all", b, pair_a[:, j], va.n, "share",
                      f"The share of the gain kept under a shift of {int(x)} s on every clean anchor, not only the reward window.")
            R.put("shift.W50_all", b, W_a, va.n, "s", "W50 on every clean anchor.")
            R.put("shift.delta0_all", b, d0_a, va.n, "s", "Tuning centre on every clean anchor.")
        C[f"shift.tuning.{b}"] = curve(np.r_[sec[:len(sec) // 2], 0.0, sec[len(sec) // 2:]], np.column_stack([F[:, :len(sec) // 2], np.ones(len(F)), F[:, len(sec) // 2:]]),
                                       "shift of the own UA, s (positive = UA delayed)", "share of the gain kept")
        if "k_num" in d:
            for h in range(d["k_num"].shape[1]):
                with np.errstate(invalid="ignore", divide="ignore"):
                    k = View(d, m)["k_num"][:, h] / View(d, m)["k_den"][:, h]
                R.put(f"shift.kappa_h{h}", b, View(d, m).pool(k), nfin(k), "index",
                      f"Tracking index of head {h} (R01): projection of the attention change under a shifted UA on the change "
                      "a pure tracker would show; sums over clean anchors and the shifts of 8, 16 and 32 s in both directions.")
    if cls is not None:
        for c in CLASSES:
            v = View(d, cls == c).keep(np.isfinite(d["f_den"][cls == c]))
            _, _, W, d0 = tuning(v)
            R.put(f"shift.W50.{c}", c, W, v.n, "s", f"W50 in {c} recordings.")
            R.put(f"shift.delta0.{c}", c, d0, v.n, "s", f"Tuning centre in {c} recordings.")


def stats_heads(d, R, C):
    bl, _, _ = blocks_of(d)
    hs = int(np.nanargmax(View(d).pool(d["kl"])[0]))
    M = d["kl"].shape[1]
    for b, m in bl.items():
        v = View(d, m)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = v["kl"] / v["kl"].sum(1, keepdims=True)
        sg = v["ho_m"] - v["ho_s"]
        phi, full = shapley(sg), sg[:, 0]
        for h in range(M):
            tag = [f"h{h}"] + (["main"] if h == hs else [])
            for t in tag:
                R.put(f"heads.kl_share_{t}", b, v.median(share[:, h]), nfin(share[:, h]), "share",
                      f"KL of head {h} / total KL per recording (anchors with at least one key); median over recordings.")
                R.put(f"heads.shapley_{t}", b, v.ratio(phi[:, h], full), nfin(phi[:, h], full), "share",
                      f"Share of the specific gain carried by head {h}: exact Shapley value over the {2 ** M} head subsets "
                      "(a head is off when its posterior dimensions are set to the prior's; gate kept). "
                      "Differs from the study: gain at the latent means, not K samples.")
            R.put(f"heads.elag_h{h}", b, v.pool(v["elag"][:, h]), nfin(v["elag"][:, h]), "steps of 4 s",
                  f"Expected lag of head {h} on clean anchors (all keys valid and warm).")
            R.put(f"heads.short_mass_h{h}", b, v.pool(v["m_short"][:, h]), nfin(v["m_short"][:, h]), "share",
                  f"Attention mass of head {h} on lags 0-{L0}, clean anchors.")
        R.put("heads.main_head", b, np.r_[hs, np.full(N_BOOT, hs)], v.n, "head index", "Head with the largest pooled KL.")
        R.put("heads.short_mass_main", b, v.pool(v["m_short"][:, hs]), nfin(v["m_short"][:, hs]), "share", f"Attention mass of the main head on lags 0-{L0}.")
        R.put("heads.loo_main", b, 1 - v.ratio(sg[:, 1 << hs], full), nfin(full), "share", "Share of the specific gain lost when only the main head is switched off.")
        R.put("heads.only_main", b, v.ratio(sg[:, (2 ** M - 1) ^ (1 << hs)], full), nfin(full), "share", "Share of the specific gain kept when only the main head is left on.")
        R.put("heads.kl_null_ratio_main", b, v.ratio(v["kl0"][:, hs], v["kl"][:, hs]), nfin(v["kl0"][:, hs], v["kl"][:, hs]), "share", "KL of the main head with zeroed UA / with own UA.")
        R.put("heads.drift_main", b, v.pool(v["drift"][:, hs]), nfin(v["drift"][:, hs]), "steps over the span",
              "R07: within-segment slope of the main head's expected lag on the anchor index times the clean span.")
        if "short_loss" in d:
            R.put("heads.short_loss", b, v.pool(v["short_loss"]), nfin(v["short_loss"]), "nats per cell",
                  f"R08: specific gain lost when lags 0-{L0} are removed for every head (removal mask, gate recomputed), reward-window anchors, K samples.")
            R.put("heads.short_loss_share", b, v.ratio(v["short_loss"], v["sg"]), nfin(v["short_loss"]), "share", "The same as a share of the specific gain.")
            R.put("heads.long_loss_share", b, v.ratio(v["long_loss"], v["g_rw"]), nfin(v["long_loss"]), "share",
                  f"Share of the matched gain lost when lags above {L0} are removed (own mask).")
    return hs


def stats_clamp(d, R, C):
    for b, m in blocks_of(d)[0].items():
        v = View(d, m)
        for tag, where in (("", "reward-window anchors"), ("_all", "all anchors with at least one key")):
            if "c_m" + tag not in d:
                continue
            for mode, txt in (("pos", "its position-only profile (keys zeroed: query x lag embedding + lag bias)"),
                              ("static", "its static profile (position term at the mean query)"),
                              ("rev", "the static profile with lags 0-14 in reversed order (control)")):
                R.put(f"clamp.{mode}{tag}", b, v.ratio(v[f"c_{mode}{tag}"], v["c_m" + tag]), nfin(v[f"c_{mode}{tag}"]), "share",
                      f"Share of the matched gain kept when every head's attention is replaced by {txt}; gate recomputed; "
                      f"{where}, K samples.")


def _window(s, n, lo, hi):
    m = (AGES >= lo) & (AGES <= hi)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s[..., m].sum(-1) / np.where(n[..., m].sum(-1) > 0, n[..., m].sum(-1), np.nan)


def stats_clock(d, R, C, hs):
    hs = int(np.nanargmax(View(d).pool(d["klh"])[0])) if hs is None else hs
    for b, m in blocks_of(d)[0].items():
        v = View(d, m)
        for q, name, txt in (("gate", "gate", "gate logit"), ("kl", "kl", "log KL of the main head"), ("elag", "lag", "expected lag of the main head")):
            pick = (lambda x: x) if q == "gate" else (lambda x: x[:, hs])
            ou, of = pick(v[f"rU_{q}"]) - pick(v[f"r0_{q}"]), pick(v[f"rF_{q}"]) - pick(v[f"r0_{q}"])
            R.put(f"clock.omega_{name}", b, v.pool(ou), nfin(ou), "correlation",
                  f"Ownership by the fed UA of the {txt} (R34): correlation over the clean anchors of a segment between the "
                  "run with the donor's UA and the donor's own run, minus the correlation between the two own runs; mean over the recording's pairs.")
            R.put(f"clock.omegaF_{name}", b, v.pool(of), nfin(of), "correlation",
                  "The same for the recipient's FHR: correlation between the donor-UA run and the recipient's own run, minus the floor.")
        ok = (v["d_ev"] >= MIN_EVENTS).all(1)
        w = v.keep(ok)
        T = [_window(w["d_s"][:, k], w["d_n"][:, k // 2], 16, 48) - _window(w["d_s"][:, k], w["d_n"][:, k // 2], -60, -8) for k in range(4)]
        R.put("clock.dip", b, w.pool(T[1]), nfin(T[1]), "nats per cell",
              "R38: gain with the donor's UA, mean over 16-48 s after the donor's contraction onsets minus the mean over "
              f"60-8 s before them; recordings with at least {MIN_EVENTS} events on both clocks.")
        R.put("clock.dip_vs_own", b, w.pool(T[1] - T[0]), nfin(T[1], T[0]), "nats per cell", "Control: the same dip minus that of the own-UA gain on the donor's clock.")
        R.put("clock.gate_rise", b, w.pool(T[3]), nfin(T[3]), "gate units", "Rise of the gate with the donor's UA after the donor's onsets (same windows).")
        with np.errstate(invalid="ignore", divide="ignore"):
            C[f"clock.donor_clock.{b}"] = curve(AGES, np.stack([w.pool(w["d_s"][:, k] / w["d_n"][:, k // 2]) for k in (1, 0)], -1),
                                                "s after the donor's contraction onset", "gain, nats per cell", ["donor UA", "own UA"])


def stats_events(d, R, C):
    for b, m in blocks_of(d)[0].items():
        v = View(d, m).keep((d["e_ev"][m] >= MIN_EVENTS).all(1))
        s, n = v["e_s"], v["e_n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            cur = lambda c, k: v.pool(s[:, c, k] / n[:, c, min(k // 2, 1)])       # noqa: E731
            own, null, sur = cur(0, 0), cur(1, 0), cur(0, 1)
            gate = [cur(0, 2), cur(1, 2)]
            kl = [cur(0, 4), cur(1, 4)]
        for hi, tag in ((60, ""), (180, "_0_180")):   # 0-60 s as in the lag study; 0-180 s so a later peak is not cut at 60 s
            win = (AGES >= 0) & (AGES <= hi)
            loc, height = parabolic_peak(own[:, win], AGES[win])
            R.put("events.peak" + tag, b, loc, v.n, "s after onset",
                  f"R37: location of the parabolic peak over 0-{hi} s of the pooled onset-locked matched gain (joint mask, 4-s age "
                  f"bins); onsets = the contraction detector's onsets; recordings with at least {MIN_EVENTS} events on the own and "
                  "on the donor's clock.")
            R.put("events.peak_height_over_null" + tag, b, height - np.nanmax(null, 1), v.n, "nats per cell",
                  f"Peak height (0-{hi} s) minus the maximum of the null curve (the same anchors locked on the donor's onsets).")
            loc_s, _ = parabolic_peak((own - sur)[:, win], AGES[win])
            R.put("events.peak_specific" + tag, b, loc_s, v.n, "s after onset",
                  f"The same location over 0-{hi} s for the specific gain (matched minus surA).")
        C[f"events.gain.{b}"] = curve(AGES, np.stack([own, sur, null], -1), "s after contraction onset", "gain, nats per cell",
                                      ["own UA", "donor UA", "own UA on the donor's clock (null)"])
        C[f"events.gate.{b}"] = curve(AGES, np.stack(gate, -1), "s after contraction onset", "gate", ["own clock", "donor's clock (null)"])
        if np.nanmax(np.abs(kl[0])) > 0:
            C[f"events.kl.{b}"] = curve(AGES, np.stack(kl, -1), "s after contraction onset", "KL, nats per anchor", ["own clock", "donor's clock (null)"])
        v = View(d, m)
        for c, tag, txt in ((0, "", "own contraction onsets"), (1, "_sham", "the donor's onsets (sham)")):
            h, cnt = v["fo_hist"][:, c], v["fo_n"][:, c]
            cum = np.cumsum(v.pool(h) / v.pool(h).sum(1, keepdims=True), 1)
            k = (cum >= 0.5).argmax(1)
            prev = np.where(k > 0, cum[np.arange(len(cum)), np.maximum(k - 1, 0)], 0.0)
            med = STEP_S * (k + (0.5 - prev) / np.maximum(cum[np.arange(len(cum)), k] - prev, 1e-12))
            R.put(f"events.first_open{tag}", b, med, int((cnt[:, 1] > 0).sum()), "s after onset",
                  f"Median time from {txt} to the first anchor with gate >= {GATE_OPEN} (own UA), among onsets before which "
                  "the gate was shut for 20 s, with a follow-up of at least 60 s, and after which it opens within 180 s; "
                  "interpolated inside the 4-s bin.")
            R.put(f"events.never_open{tag}", b, v.pool(cnt[:, 2]) / v.pool(cnt[:, 0]), int((cnt[:, 0] > 0).sum()), "share",
                  f"Share of such onsets ({txt}) after which the gate does not open (follow-up of at least 60 s).")


def _contrast(v, x, cls, cs, a, b):
    """Mean over the delivery modes of (mean of class a - mean of class b), per bootstrap draw, and the
    number of recordings with a finite value in the modes where both classes have one (0 if none)."""
    parts, n = [], 0
    for mode in (True, False):
        ma, mb = (cls == a) & (cs == mode), (cls == b) & (cs == mode)
        if nfin(x[ma]) and nfin(x[mb]):
            n += nfin(x[ma | mb])
            with np.errstate(invalid="ignore", divide="ignore"):
                fa = (v.W[:, ma] @ np.nan_to_num(x[ma])) / (v.W[:, ma] @ np.isfinite(x[ma]))
                fb = (v.W[:, mb] @ np.nan_to_num(x[mb])) / (v.W[:, mb] @ np.isfinite(x[mb]))
            parts.append(fa - fb)
    return (np.nanmean(np.stack(parts), 0) if parts else np.full(len(v.W), np.nan)), n


def stats_groups(d, R, C, hs):
    bl, cls, cs = blocks_of(d)
    v = View(d)
    qs = {"sg": (v["gm"] - v["gs"], "nats per cell", "specific gain on reward-window anchors"),
          "gate_open": (v["open"], "share", f"share of anchors with gate >= {GATE_OPEN}")}
    binq = {"sg": d["gm@bin"] - d["gs@bin"], "gate_open": d["open@bin"]}
    if "kl" in d:
        qs["kl"] = (v["kl"], "nats per anchor", "KL per anchor")
        binq["kl"] = d["kl@bin"]
        if hs is not None:     # no heads or clock step: the main head is unknown, so its lag rows are left out
            qs["elag"] = (v["elag"][:, hs], "steps of 4 s", "expected lag of the main head on clean anchors")
            binq["elag"] = d["elag@bin"][:, :, hs]
    for q, (x, unit, txt) in qs.items():
        for j, name in enumerate(BIN_NAMES):
            xb = binq[q][:, j].astype(float)
            R.put(f"groups.{q}.{name.replace(' ', '')}", "all", v.pool(xb), nfin(xb), unit, f"{txt}, segments {name} before delivery; pooled over subgroups.")
        if cls is None:
            continue
        for c in CLASSES:
            w = View(d, cls == c)
            R.put(f"groups.{q}.{c}", c, w.pool(x[cls == c]), nfin(x[cls == c]), unit, f"{txt}, {c} recordings (equal weight of the two delivery modes).")
        for mode, name in ((True, "cs"), (False, "no_cs")):
            w = View(d, cs == mode)
            R.put(f"groups.{q}.{name}", name, w.pool(x[cs == mode]), nfin(x[cs == mode]), unit, f"{txt}, delivery mode {name} (equal weight of the classes).")
        for a, b in (("hie", "healthy_bg"), ("acidosis", "healthy_bg"), ("hie", "acidosis"), ("healthy_bg", "healthy_no_bg")):
            R.put(f"groups.{q}.{a}-{b}", "contrast", *_contrast(v, x, cls, cs, a, b), unit,
                  f"Difference of {txt}: {a} minus {b}, mean of the differences inside each delivery mode. "
                  + ("Placebo for how recordings were selected. " if b == "healthy_no_bg" else "") + "Differs from the study: no covariate adjustment.")
    if "g_slow" in d:
        for b, m in bl.items():
            w = View(d, m)
            R.put("groups.slow", b, w.ratio(w["gm"] - w["g_slow"], w["gm"] - w["gs"]), nfin(w["g_slow"]), "share",
                  "R50: share of the specific gain lost when the slow UA scattering channels (filter delay of 100 s or more) "
                  "are taken from the donor, values and flags; joint mask, reward-window anchors.")


def stats_irf(d, R, C):
    for b, m in blocks_of(d)[0].items():
        v = View(d, m)
        for tag, txt in (("", "own UA"), ("_w", "another recording's UA")):
            go = np.nanmean(v["open" + tag], 1)
            R.put(f"irf.gate_open{tag}", b, v.pool(go), nfin(go), "share",
                  f"Share of (anchor, head channel) cells with an open IRF gate, {txt}; anchors with UA cover and enough valid cells.")
            for nm, name in (("delta", "pred_gap_irf"), ("delta_expl", "pred_gap_expl")):
                R.put(f"irf.{name}{tag}", b, v.pool(v[f"gap_{nm}{tag}"]), nfin(v[f"gap_{nm}{tag}"]), "nats per cell",
                      f"Gain of the head's {'forecast' if nm == 'delta' else 'explain'}-mode mean shift on the base forecast, {txt}; "
                      "base NLL minus shifted NLL per valid scored cell, K shared samples.")
                R.put(f"irf.{name}{tag}_per_anchor", b, v.pool(v[f"gapsum_{nm}{tag}"]), nfin(v[f"gapsum_{nm}{tag}"]), "nats per anchor", "The same summed over the anchor's valid cells.")
        C[f"irf.open_by_channel.{b}"] = curve(np.arange(d["open"].shape[1]), np.stack([v.pool(v["open"]), v.pool(v["open_w"])], -1),
                                              "head channel (index into the head's channels)", "share of anchors with an open gate", ["own UA", "wrong UA"])
        dl = v["delay"]
        R.put("irf.open_level", b, v.pool(v["open0"]), nfin(v["open0"]), "share", "Share of anchors with an open gate on the level channel, own UA.")
        for qn, q in (("median", 50), ("q25", 25), ("q75", 75)):
            vals = np.array([np.nanpercentile(dl, q)] + [np.nanpercentile(np.repeat(dl, w.astype(int)), q) if w.sum() else np.nan
                                                        for w in v.W[1:201]]) if np.isfinite(dl).any() else np.full(2, np.nan)
            R.put(f"irf.delay_{qn}", b, vals, nfin(dl), "s",
                  f"{qn} over recordings of the level-channel IRF delay (most negative point of the IRF, anchors with an open gate; "
                  "segment median, then mean over the recording's segments). Includes the channel's filter latency. Interval from 200 draws. "
                  "Recordings weighted equally: a quantile over recordings, not the subgroup-equal mean.")
        a, c = d["delay@half"][m][:, 0].astype(float), d["delay@half"][m][:, 1].astype(float)
        ok = np.isfinite(a) & np.isfinite(c)

        def rel(w):
            w = w[ok]
            ma, mc = (w * a[ok]).sum() / w.sum(), (w * c[ok]).sum() / w.sum()
            r = (w * (a[ok] - ma) * (c[ok] - mc)).sum() / np.sqrt((w * (a[ok] - ma) ** 2).sum() * (w * (c[ok] - mc) ** 2).sum())
            return 2 * r / (1 + r)
        if ok.sum() > 10:
            R.put("irf.delay_reliability", b, np.array([rel(w) for w in v.W]), int(ok.sum()), "reliability",
                  "Split-half reliability of the per-recording delay: correlation between odd and even segments, Spearman-Brown corrected. "
                  "Recordings weighted equally, not subgroup-equal.")
        for j, st in enumerate(PLANT_STEPS):
            R.put(f"irf.plant_err_{int(STEP_S * st)}s", b, v.median(v["plant_err"][:, j]), nfin(v["plant_err"][:, j]), "s",
                  f"Planted response {int(STEP_S * st)} s after the donor's UA on the level channel (one normalised unit deep): "
                  "read delay minus planted delay, median over anchors with an open gate, then over recordings.")
            R.put(f"irf.plant_open_{int(STEP_S * st)}s", b, v.pool(v["plant_open"][:, j]), nfin(v["plant_open"][:, j]), "share", "Share of anchors where the gate opens on the planted response.")
        if "e0" in d:
            for e, etxt in (("e0", "level-channel evidence z / k, cut to [0, 3]"), ("e1", "share of head channels with an open gate")):
                for q, qtxt in (("gate", "the lag-attention gate"), ("sg", "the specific gain")):
                    x, y = v[e], v[q]
                    okk = np.isfinite(x) & np.isfinite(y)

                    def corr(w, x=x, y=y, okk=okk):
                        w = w[okk]
                        mx, my = (w * x[okk]).sum() / w.sum(), (w * y[okk]).sum() / w.sum()
                        return (w * (x[okk] - mx) * (y[okk] - my)).sum() / np.sqrt((w * (x[okk] - mx) ** 2).sum() * (w * (y[okk] - my) ** 2).sum())
                    R.put(f"irf.r_{e}_{q}_recordings", b, np.array([corr(w) for w in v.W]), int(okk.sum()), "correlation",
                          f"Across recordings: Pearson correlation of the GUID means of the head's evidence ({etxt}) and {qtxt}. "
                          "Recordings weighted equally, not subgroup-equal.")
                    R.put(f"irf.r_{e}_{q}_anchors", b, v.pool(v[f"r_{e}_{q}"]), nfin(v[f"r_{e}_{q}"]), "correlation",
                          f"Across anchors: the same correlation over the anchors of a segment (at least 40), mean over segments and recordings.")


def stats_raw(d, R, C):
    bl, cls, _ = blocks_of(d)
    win = (AGES >= 0) & (AGES <= 45)

    def nadir(v, s, n):
        with np.errstate(invalid="ignore", divide="ignore"):
            cur = v.pool(s / n)
        loc, depth = parabolic_peak(-cur[:, win], AGES[win])
        return cur, loc, -depth

    groups = [("all", bl["all"], d["r_s"], d["r_n"], d["r_ev"])]
    if cls is not None:
        groups += [(c, cls == c, d["r_s"], d["r_n"], d["r_ev"]) for c in CLASSES]
    groups += [(name.replace(" ", ""), bl["all"], d["r_s@bin"][:, j], d["r_n@bin"][:, j], d["r_ev@bin"][:, j]) for j, name in enumerate(BIN_NAMES)]
    for name, m, s, n, ev in groups:
        v = View({"guid": d["guid"], "subgroup": d["subgroup"], "s": s, "n": n}, m & (ev >= MIN_EVENTS))
        if v.n < 5:
            continue
        cur, loc, depth = nadir(v, v["s"][:, 0], v["n"][:, 0])
        sid = "" if name == "all" else "." + name
        R.put("raw.nadir" + sid, name, loc, v.n, "s after the UA peak",
              "R28 on the step grid: location of the minimum (3-point parabola) over 0-45 s of the pooled FHR level around the "
              f"UA peak of every contraction of the model's contraction detector; FHR level = mean of the valid raw samples of a 4-s step, minus the segment's median; "
              f"recordings with at least {MIN_EVENTS} contractions. Differs from the study: its own detector, a 1-s grid and a running baseline.")
        R.put("raw.depth" + sid, name, depth, v.n, "bpm", "FHR level at that minimum, relative to the segment's median.")
        R.put("raw.fall" + sid, name, depth - np.nanmax(cur[:, AGES <= 0], 1), v.n, "bpm",
              "The same minimum minus the highest point of the curve in the 60 s before the UA peak (no baseline needed).")
        if name == "all":
            with np.errstate(invalid="ignore", divide="ignore"):
                C["raw.fhr.all"] = curve(AGES, cur, "s after the UA peak", "FHR level minus segment median, bpm")
                C["raw.ua.all"] = curve(AGES, v.pool(v["s"][:, 1] / v["n"][:, 1]), "s after the UA peak", "UA level, toco units")


def _clean(x):
    """NaN and infinity -> None, so the JSON is strict."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return None if isinstance(x, float) and not np.isfinite(x) else x


def curve(x, Y, xlabel, ylabel, names=None):
    """A pooled curve with its bootstrap band, for the JSON and the figures. Y (draws, bins[, series])."""
    Y = np.asarray(Y, float)
    Y = Y[..., None] if Y.ndim == 2 else Y
    lo, hi = np.nanpercentile(Y[1:], [2.5, 97.5], 0)
    r = lambda a: np.round(a.T, 6).tolist()        # noqa: E731  NaN stays a gap; `_clean` writes it as null
    return {"x": np.asarray(x, float).tolist(), "y": r(Y[0]), "lo": r(lo), "hi": r(hi), "xlabel": xlabel, "ylabel": ylabel,
            "series": names or [""]}


# ----------------------------------------------------------------------------- outputs
def guid_table(steps, path):
    """One row per recording with every per-recording scalar (arrays up to 16 values wide are spread;
    curves and the per-bin / per-half copies stay in `steps/`)."""
    import pandas as pd

    frames = []
    for s, d in steps.items():
        cols = {"guid": d["guid"], "subgroup": d["subgroup"]}
        for k, v in d.items():
            if not isinstance(v, np.ndarray) or v.shape[:1] != d["guid"].shape or k in ("guid", "subgroup") or "@" in k or v.dtype.kind not in "fiu":
                continue
            flat = v.reshape(len(v), -1)
            if flat.shape[1] <= 16:
                for j in range(flat.shape[1]):
                    cols[f"{s}.{k}" + (f"[{j}]" if flat.shape[1] > 1 else "")] = flat[:, j]
        frames.append(pd.DataFrame(cols).set_index(["guid", "subgroup"]))
    pd.concat(frames, axis=1).reset_index().to_csv(path, index=False, float_format="%.5g")


def _fmt(x, unit=""):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.1f}" if (unit + " ").startswith("s ") else f"{x:.3g}" if abs(x) >= 1 else f"{x:.4f}" if abs(x) >= 0.01 else f"{x:.5f}"


def _ci(c, unit=""):
    return "" if c is None else f"[{_fmt(c[0], unit)}, {_fmt(c[1], unit)}]"


def verdict(row):
    """Agreement with the reference: the interval covers the reference value, or the two intervals overlap."""
    ref, ci = row["reference"], row["ci95"]
    if ref is None or ci is None:
        return ""
    if ci[0] <= ref["value"] <= ci[1]:
        return "covers"
    if ref["ci95"] and ci[0] <= ref["ci95"][1] and ref["ci95"][0] <= ci[1]:
        return "intervals overlap"
    return "differs"


def _yerr(row):
    """Error-bar deltas of a row with a finite estimate, never negative (a percentile interval need not
    contain the sample estimate); zero without an interval."""
    est, ci = row["estimate"], row["ci95"] or [row["estimate"]] * 2
    return [[max(0.0, est - ci[0])], [max(0.0, ci[1] - est)]]


def figures(curves, rows, out, hb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    made = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=150)
        plt.close(fig)
        made.append(name)

    def draw(ax, c, shade=None):
        x = np.array(c["x"])
        for j, name in enumerate(c["series"]):
            ax.plot(x, c["y"][j], label=name or None, lw=1.6)
            ax.fill_between(x, c["lo"][j], c["hi"][j], alpha=0.2)
        if shade:
            ax.axvspan(*shade, color="0.85", zorder=0, label="reward window of the objective")
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlabel(c["xlabel"])
        ax.set_ylabel(c["ylabel"])
        if any(c["series"]) or shade:
            ax.legend(fontsize=7)

    by = {(r["id"], r["block"]): r for r in rows}

    def bars(ax, ids, labels, ylabel, ref=True):
        rs = [by.get((i, hb)) for i in ids]
        keep = [(r, l) for r, l in zip(rs, labels) if r is not None and np.isfinite(r["estimate"])]
        for k, (r, l) in enumerate(keep):
            ax.errorbar(k, r["estimate"], yerr=_yerr(r), fmt="o", color="C0", capsize=3)
            if ref and r["reference"]:
                ax.plot(k + 0.2, r["reference"]["value"], "s", color="C1", ms=4)
        ax.set_xticks(range(len(keep)), [l for _, l in keep], fontsize=7, rotation=20)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_ylabel(ylabel)
        return len(keep)

    for name, title, shade in (("gain.sg_by_horizon", "Specific gain by forecast horizon", None),
                               ("shift.tuning", "Gain kept against the shift of the own UA", None),
                               ("clock.donor_clock", "Gain on the donor's contraction clock", None),
                               ("events.gain", "Gain locked on contraction onset", (0, 60)),
                               ("events.gate", "Gate (own UA) locked on contraction onset", (0, 60)),
                               ("events.kl", "KL locked on contraction onset", (0, 60)),
                               ("irf.open_by_channel", "IRF gate open share by head channel", None)):
        c = curves.get(f"{name}.{hb}")
        if c:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            draw(ax, c, shade)
            ax.set_title(f"{title} ({hb} block)", fontsize=9)
            save(fig, name.replace(".", "_") + ".png")
    if "raw.fhr.all" in curves:
        fig, axs = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
        draw(axs[0], curves["raw.fhr.all"])
        draw(axs[1], curves["raw.ua.all"])
        axs[0].set_xlabel("")
        axs[0].set_title("Raw signals around the UA peak of a contraction (all recordings)", fontsize=9)
        save(fig, "raw_peak_locked.png")
    for name, ids, labels, ylabel in (
            ("gain_arms.png", ["gain.gain_matched_rw", "gain.gain_surA_rw", "gain.gain_surB_rw", "gain.gain_null_rw", "gain.sg_rw",
                               "gain.sg_surB_rw", "gain.sg_all", "gain.sg_outside_rw"],
             ["own UA", "surA", "surB", "zeroed UA", "specific (surA)", "specific (surB)", "specific, all anchors",
              "specific, outside reward window"], "gain, nats per cell"),
            ("heads.png", [f"heads.{k}_h{h}" for k in ("kl_share", "shapley") for h in range(8)],
             [f"{k} h{h}" for k in ("KL share", "gain share") for h in range(8)], "share"),
            ("clamp.png", ["clamp.pos", "clamp.static", "clamp.rev"], ["position only", "static", "reversed (control)"], "share of the gain kept"),
            ("clock_ownership.png", [f"clock.omega{f}_{q}" for q in ("gate", "kl", "lag") for f in ("", "F")],
             [f"{q}: {o}" for q in ("gate", "KL", "lag") for o in ("fed UA", "FHR")], "ownership (correlation)")):
        fig, ax = plt.subplots(figsize=(6, 3.6))
        if bars(ax, ids, labels, ylabel):
            ax.set_title(f"{name[:-4].replace('_', ' ')} ({hb} block; orange = reference: checkpoint 369, lag study)", fontsize=8)
            save(fig, name)
        else:
            plt.close(fig)
    cl = [r for r in rows if r["step"] == "groups" and r["block"] in CLASSES and r["id"].split(".")[1] in ("sg", "gate_open")]
    if cl:
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.4))
        for ax, q, yl in zip(axs, ("sg", "gate_open"), ("specific gain, nats per cell", "gate open share")):
            rs = [r for r in cl if r["id"].split(".")[1] == q and np.isfinite(r["estimate"])]
            for k, r in enumerate(rs):
                ax.errorbar(k, r["estimate"], yerr=_yerr(r), fmt="o", capsize=3, color="C0")
            ax.set_xticks(range(len(rs)), [r["block"] for r in rs], fontsize=7, rotation=20)
            ax.set_ylabel(yl)
        axs[0].set_title("By class (GUID means, 95 % bootstrap interval)", fontsize=9)
        save(fig, "groups_class.png")
    return made


def build(out, info):
    """Read every step file of `<out>/steps` and write the four report outputs. Returns the rows."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # empty groups give NaN, which the tables show as n/a
        return _build(Path(out), info)


def _build(out, info):
    steps = {p.stem: dict(np.load(p, allow_pickle=False)) for p in sorted((out / "steps").glob("*.npz"))}
    R, C = Rows(), {}
    hs = main_head(steps)
    for name, fn in (("gain", stats_gain), ("shift", stats_shift), ("heads", stats_heads), ("clamp", stats_clamp),
                     ("clock", lambda d, R, C: stats_clock(d, R, C, hs)), ("events", stats_events),
                     ("groups", lambda d, R, C: stats_groups(d, R, C, hs)), ("irf", stats_irf), ("raw", stats_raw)):
        if name in steps:
            fn(steps[name], R, C)
    hb = "adverse" if any(r["block"] == "adverse" and r["n_guids"] > 0 for r in R) else "all"
    info = dict(info, notes=list(info.get("notes", [])))
    if hs is None and "kl" in steps.get("groups", {}):
        info["notes"].append("groups: the main head's expected-lag rows are left out: no heads or clock step names the main head.")
    (out / "lag_analysis.json").write_text(json.dumps(_clean({
        "about": "Key = '<statistic id>@<block>'. Blocks: adverse = classes acidosis + HIE, healthy = healthy_bg + healthy_no_bg, "
                 "all = every recording, a class or delivery-mode name, or contrast. Each entry: estimate, ci95 (percentile GUID-cluster "
                 "bootstrap), n_guids, unit, step, arm, definition, reference (checkpoint 369, lag study; provenance only).",
        "clock": CLOCK, "descriptive": DESCRIPTIVE, "headline_block": hb, "main_head": hs,
        "statistics": {f"{r['id']}@{r['block']}": r for r in R}, "curves": C,
        "steps_run": sorted(steps), "steps_skipped": info.get("skipped", {}), "notes": info.get("notes", [])}), indent=1))
    if steps:
        guid_table(steps, out / "guid_table.csv")
    made = figures(C, R, out, hb)
    by = {(r["id"], r["block"]): r for r in R}
    sk = info.get("skipped", {})
    L = [f"# Lag analysis: {Path(info.get('checkpoint', {}).get('path', '?')).name}", "",
         "## How to read this folder", "",
         "- `LAG_ANALYSIS.md` (this file): one section per step with a table, the definitions and how to read them.",
         "- `lag_analysis.json`: every statistic under the key `<id>@<block>` with estimate, 95 % interval, n recordings, unit, step, "
         "arm, one-sentence definition and the reference value; `curves` holds the plotted curves with their bands. Two folders "
         "from different checkpoints have the same keys.",
         "- `guid_table.csv`: one row per recording with every per-recording scalar (`<step>.<name>`), to redo any statistic.",
         "- `figures/*.png`: the curves and comparisons; `run_info.json`: checkpoint hash, model variant, shards, arguments, run times.",
         "- `steps/*.npz`: the per-recording arrays each step wrote (also curves per age bin). Not needed to read the results.",
         f"- Model variant: {info.get('variant', '?')}. Recordings: {info.get('n_guids', '?')}; segments: {info.get('n_segments', '?')}; "
         f"latent samples K = {info.get('args', {}).get('k_samples', '?')}.",
         f"- Steps in this folder: {', '.join(f'`{k}`' for k in steps) or 'none'}. Not run: "
         + ("; ".join(f"`{k}` ({v})" for k, v in sk.items()) or "none") + ".",
         "- Unit of analysis: the recording (GUID). A GUID value is the mean over the recording's segments of the segment mean over "
         "eligible anchors. An estimate is the equal-weight mean of the subgroup means (subgroup = shard name). Intervals: 95 % "
         f"percentile GUID-cluster bootstrap, {N_BOOT} draws, stratified by subgroup. The lag study used BCa intervals and a two-way "
         "bootstrap for donor rows, so its intervals are a little wider.",
         "- Units: gains in nats per valid scored forecast cell; KL in nats per anchor; lags in steps of 4 s; times in seconds.",
         f"- {CLOCK}",
         "- Arms: own UA (matched); surA = UA of another recording of the same subgroup; surB = UA of the same recording at least "
         "40 min away; zeroed UA. Reward window = anchors 0-60 s after a contraction onset of the model's contraction detector.",
         f"- Blocks: `adverse` = recordings of the classes acidosis and HIE; `healthy` = healthy_bg and healthy_no_bg; `all`. "
         f"Headline block of this run: `{hb}`.",
         f"- Reference column = \"reference: checkpoint 369, lag study\" (`{REF_SOURCE}`), given as provenance. \"covers\" means "
         "this run's interval contains the reference value.",
         f"- {DESCRIPTIVE}"] + [f"- Note: {n}" for n in info.get("notes", [])] + [
         "", "## Headline numbers", "", "| Statistic | This run | 95 % interval | n | Reference | Reference interval | Agreement |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for sid in HEADLINE:
        r = by.get((sid, hb)) or by.get((sid, "all"))
        if r:
            ref = r["reference"] or {"value": None, "ci95": None}
            L.append(f"| `{sid}` ({r['unit']}) | {_fmt(r['estimate'], r['unit'])} | {_ci(r['ci95'], r['unit'])} | {r['n_guids']} | "
                     f"{_fmt(ref['value'], r['unit']) if ref['value'] is not None else ''} | {_ci(ref['ci95'], r['unit'])} | {verdict(r)} |")
    for name in ("gain", "shift", "heads", "clamp", "clock", "events", "groups", "irf", "raw"):
        rs = [r for r in R if r["step"] == name]
        if not rs:
            continue
        L += ["", f"## {name}", "", READ[name], "", "| Statistic | Block | Estimate | 95 % interval | n | Unit | Reference |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in rs:
            ref = r["reference"]
            L.append(f"| `{r['id']}` | {r['block']} | {_fmt(r['estimate'], r['unit'])} | {_ci(r['ci95'], r['unit'])} | {r['n_guids']} | {r['unit']} | "
                     + (f"{_fmt(ref['value'], r['unit'])} {_ci(ref['ci95'], r['unit'])} ({ref['source']})" if ref and r["block"] in (hb, "all") else "") + " |")
        L += ["", "Definitions:", ""]
        seen = set()
        for r in rs:
            key = r["id"] if r["block"] in (hb, "all", "contrast") or r["block"] in CLASSES else None
            if key and key not in seen and r["definition"] not in seen:
                seen |= {key, r["definition"]}
                L.append(f"- `{r['id']}`: {r['definition']}")
        figs = [f for f in made if f.startswith(name)]
        L += [""] + [f"![{f}](figures/{f})" for f in figs]
    L += ["", "## Steps not run", ""] + ([f"- `{k}`: {v}" for k, v in sk.items()] or ["- none"])
    L += ["", "## Not in scope", "", SCOPE, ""]
    (out / "LAG_ANALYSIS.md").write_text("\n".join(L), encoding="utf-8")
    return R
