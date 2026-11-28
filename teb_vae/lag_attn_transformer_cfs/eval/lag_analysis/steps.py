"""The measurement steps. Each `step_<name>(ctx, acc)` adds one batch to the step's accumulator.

What is kept is small: per segment, a sum and a count over an anchor mask (`acc.mean`), one value
(`acc.seg`) or a sum (`acc.sum`). `Acc.finish` turns them into per-recording (GUID) arrays:
GUID value = mean over the recording's segments of the segment mean over eligible anchors (PREREG B);
sums are added over the recording's segments. Definitions follow `tmp/lag_analysis_369/design/PREREG.md`;
the row is named beside each block.
"""
from __future__ import annotations

import numpy as np
import torch

from . import core

SHIFTS = (-8, -4, -2, -1, 1, 2, 4, 8)     # steps of 4 s; positive delays the UA
KAPPA_SHIFTS = (-8, -4, -2, 2, 4, 8)      # the six shifts of the tracking index and of the clean-anchor rule (as in the lag study)
L0 = core.REWARD_STEPS - 1                # "short" lags 0-14: the UA of the last 60 s, the length of the reward window
GATE_OPEN = 0.5                           # the gate is a sigmoid; "open" = at or above its midpoint (no data needed)
AGES = np.arange(-60.0, 181.0, 4.0)       # 61 bins of 4 s centred on -60 .. 180 s after an event
PLANT_STEPS = (2, 6, 10, 14, 18, 22)      # planted delays: 8, 24, 40, 56, 72, 88 s
SLOW_DELAY_S = 100.0                      # "slow" UA scattering channels: filter delay of 100 s or more
LAG_STEPS = ("heads", "clamp", "clock")   # steps that need the lag attention
DONOR_STEPS = ("gain", "heads", "clock", "events", "groups", "irf")
SPLIT_BY = {"groups": ("@bin",), "irf": ("@half",)}   # steps whose GUID means are also kept per time bin / per half


class Acc:
    """Per-segment accumulators of one step."""

    def __init__(self, n, by=()):
        self.n, self.s, self.c, self.binned, self.by = n, {}, {}, set(), ("",) + tuple(by)

    def _add(self, name, pos, s, c):
        s = s.detach().double().cpu().numpy() if torch.is_tensor(s) else np.asarray(s, float)
        if name not in self.s:
            self.s[name] = np.zeros((self.n,) + s.shape[1:])
            self.c[name] = None if c is None else np.zeros((self.n,) + s.shape[1:])
        self.s[name][pos] += s
        if c is not None:
            self.c[name][pos] += c.detach().double().cpu().numpy() if torch.is_tensor(c) else c

    def mean(self, name, pos, x, m):
        """x (B, A, ...) averaged over the anchors of m (B, A); non-finite values are left out."""
        ok = m.reshape(m.shape + (1,) * (x.dim() - 2)) & torch.isfinite(x)
        self._add(name, pos, torch.where(ok, x, torch.zeros_like(x)).sum(1), ok.sum(1))

    def seg(self, name, pos, x):
        """One value per segment, x (B, ...); NaN = none."""
        x = torch.as_tensor(x)
        ok = torch.isfinite(x)
        self._add(name, pos, torch.where(ok, x, torch.zeros_like(x)), ok)

    def sum(self, name, pos, s, by_bin=False):
        """A per-segment sum; `by_bin` also keeps it per time-to-delivery bin (`<key>@bin`)."""
        self._add(name, pos, s, None)
        if by_bin:
            self.binned.add(name)

    def finish(self, gidx, G, tbin, half):
        """Per-GUID arrays, float32. Where the step asks for it (`by`), mean keys also get `<key>@bin`
        (G, 4 time-to-delivery bins, ...) or `<key>@half` (G, 2: odd / even segments of the recording)."""
        out = {}
        for k, s in self.s.items():
            if self.c[k] is None:
                out[k] = np.zeros((G,) + s.shape[1:])
                np.add.at(out[k], gidx, s)
                if k in self.binned:
                    g = np.zeros((G * 4,) + s.shape[1:])
                    np.add.at(g, (gidx * 4 + tbin)[tbin >= 0], s[tbin >= 0])
                    out[k + "@bin"] = g.reshape((G, 4) + s.shape[1:])
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                seg = np.where(self.c[k] > 0, s / np.where(self.c[k] > 0, self.c[k], 1), np.nan)
            for tag, code, n in (("", gidx, G), ("@bin", np.where(tbin >= 0, gidx * 4 + tbin, -1), G * 4),
                                 ("@half", gidx * 2 + half, G * 2)):
                if tag not in self.by:
                    continue
                ok = np.isfinite(seg) & (code >= 0).reshape((-1,) + (1,) * (seg.ndim - 1))
                gs, gc = np.zeros((n,) + seg.shape[1:]), np.zeros((n,) + seg.shape[1:])
                np.add.at(gs, np.maximum(code, 0), np.where(ok, seg, 0.0))
                np.add.at(gc, np.maximum(code, 0), ok)
                with np.errstate(invalid="ignore", divide="ignore"):
                    v = np.where(gc > 0, gs / np.where(gc > 0, gc, 1), np.nan)
                out[k + tag] = v if not tag else v.reshape((G, n // G) + seg.shape[1:])
        return {k: v.astype(np.float32) for k, v in out.items()}


def lock(onsets, vals, mask, n_bins=len(AGES)):
    """Event-locked sums of one segment: for every onset (anchor index) and anchor of `mask` (A,), add
    vals (S, A) to the bin of the anchor's age, 4 (anchor - onset) s. Returns sums (S, bins), counts
    (bins,) and the number of events that put at least one anchor in a bin."""
    s, n = np.zeros((len(vals), n_bins)), np.zeros(n_bins)
    if not len(onsets):
        return s, n, 0
    age = core.STEP_S * (np.arange(mask.shape[0])[None] - np.asarray(onsets, float)[:, None])
    b = np.floor((age - (AGES[0] - 2.0)) / 4.0).astype(int)
    e, a = np.nonzero((b >= 0) & (b < n_bins) & mask[None])
    np.add.at(n, b[e, a], 1.0)
    for k, v in enumerate(vals):
        np.add.at(s[k], b[e, a], v[a])
    return s, n, int(np.unique(e).size)


def _np(x):
    return x.detach().float().cpu().numpy()


def _logit(g, lo=1e-4):
    g = g.clamp(lo, 1 - lo)
    return torch.log(g / (1 - g))


def _elag(alpha, ak):
    """Expected lag per head (B, A, M) in steps; NaN where no key is left."""
    lags = torch.arange(alpha.shape[-1], device=alpha.device, dtype=alpha.dtype)
    return (alpha * lags).sum(-1).masked_fill(~ak[..., None], torch.nan)


def _own(ctx):
    """Masks of the own-UA run: >= 1 key, enough valid cells, reward window, clean."""
    E, own = ctx.E, ctx.ua("own")
    ak = ctx.any_key(own["km"], own["rm"])
    return ak, E["okc"], E["reward"]


# ----------------------------------------------------------------------------- 1. gain (R04, R02)
def step_gain(ctx, acc):
    p, E = ctx.pos, ctx.E
    ak, okc, rw = _own(ctx)
    g, _ = ctx.gm
    acc.mean("g_rw", p, g, rw & ak & okc)                        # matched gain, own mask
    acc.mean("g_all", p, g, ak & okc)
    P0 = ctx.arm(ctx.ua("null"))
    g0 = ctx.gain(P0, ak & okc)[0]
    acc.mean("g0_rw", p, g0, rw & ak & okc)
    acc.mean("g0_all", p, g0, ak & okc)
    acc.mean("gate", p, ctx.matched["gate"], ak)
    if ctx.run["lag"]:
        acc.mean("kl", p, ctx.matched["kl"].sum(-1), ak)        # KL per anchor, nats
        acc.mean("kl0", p, P0["kl"].sum(-1), ak)
    # Anchor windows by the age since the latest own onset: the reward window (0-60 s), its complement,
    # and the complement split into 60-180 s after an onset and quiet (later, or no onset before).
    age = ctx.since_onset()
    windows = {"rw": rw, "all": torch.ones_like(rw), "out": ~rw,
               "post": (age > core.REWARD_STEPS) & (age <= 3 * core.REWARD_STEPS), "quiet": age > 3 * core.REWARD_STEPS}
    for which, m, tag in (("A", ctx.mA, "a"), ("B", ctx.mB, "b")):
        D = ctx.donor(which)
        mj = D["ak"] & m[:, None]
        (gJ, cJ), (gX, cX) = D["gJ"], D["gX"]
        for w in (windows if which == "A" else ("rw", "all")):  # matched and surrogate under the joint mask
            acc.mean(f"{tag}m_{w}", p, gJ, mj & windows[w] & okc)
            acc.mean(f"{tag}s_{w}", p, gX, mj & windows[w] & okc)
        if ctx.run["lag"]:
            acc.mean(f"{tag}_klm", p, D["PJ"]["kl"].sum(-1), mj)
            acc.mean(f"{tag}_kls", p, D["PX"]["kl"].sum(-1), mj)
        if which == "A":
            for nm, cell in (("m", cJ), ("s", cX)):
                tau, blk = ctx.gain_tau(cell)
                acc.mean(f"a{nm}_tau", p, tau, mj & rw & okc)   # by horizon step
                acc.mean(f"a{nm}_blk", p, blk, mj & rw & okc)   # by block: scattering, phase


# ----------------------------------------------------------------------------- 2. shift (R33, R01)
def step_shift(ctx, acc):
    p, E, own = ctx.pos, ctx.E, ctx.ua("own")
    ak, okc, rw = _own(ctx)
    lag = ctx.run["lag"]
    P, call = {}, ctx.clean(own["km"], own["rm"]) if lag else torch.ones_like(ak)
    for d in SHIFTS:
        ua = ctx.shifted(own, d)
        P[d] = ctx.arm(ua)
        if d in KAPPA_SHIFTS:                                    # eligible: clean in every shifted window
            call = call & (ctx.clean(ua["km"], ua["rm"], d) if lag else ua["inside"])
    sel_all = okc & ak & call
    sel = rw & sel_all
    num = torch.stack([ctx.gain(P[d], sel_all, mean=True)[0] for d in SHIFTS], -1)   # gain at latent means
    den = ctx.gain(ctx.matched, sel_all, mean=True)[0]
    for tag, m in (("", sel), ("_all", sel_all)):               # reward-window anchors; every clean anchor
        acc.mean("f_num" + tag, p, num, m)
        acc.mean("f_den" + tag, p, den, m)
    if lag:                                                      # tracking index: every clean anchor
        kn = [core.kappa_terms(ctx.matched["alpha"], P[d]["alpha"], d) for d in KAPPA_SHIFTS]
        w = call[..., None]
        acc.sum("k_num", p, sum(torch.where(w, torch.nan_to_num(n), 0.0).sum(1) for n, _ in kn))
        acc.sum("k_den", p, sum(torch.where(w, torch.nan_to_num(dn), 0.0).sum(1) for _, dn in kn))


# ----------------------------------------------------------------------------- 3. heads (R03, R11, R07, R08)
def step_heads(ctx, acc):
    p, E, own = ctx.pos, ctx.E, ctx.ua("own")
    ak, okc, rw = _own(ctx)
    M, dev = ctx.run["model"].lag_attn.num_heads, ctx.run["device"]
    Pm = ctx.matched
    acc.mean("kl", p, Pm["kl"], ak)                              # KL per head
    acc.mean("kl0", p, ctx.arm(ctx.ua("null"))["kl"], ak)
    c0 = ctx.clean(own["km"], own["rm"])
    el = _elag(Pm["alpha"], ak)
    acc.mean("elag", p, el, c0)
    acc.mean("m_short", p, Pm["alpha"][..., :L0 + 1].sum(-1), c0)
    # R07: slope of the expected lag on the anchor index within the segment, times the clean span
    x = torch.arange(E["A"], device=dev, dtype=el.dtype)[None, :, None]
    w = (c0[..., None] & torch.isfinite(el)).to(el.dtype)
    n = w.sum(1).clamp(min=1)
    xc, y = (x - (w * x).sum(1, keepdim=True) / n[:, None]) * w, torch.nan_to_num(el)
    span = (torch.where(w > 0, x, -1e9).amax(1) - torch.where(w > 0, x, 1e9).amin(1))
    slope = (xc * y).sum(1) / (xc ** 2).sum(1)
    acc.seg("drift", p, torch.where(w.sum(1) >= 20, slope * span, torch.nan))
    # R11: every subset of heads switched off (its posterior dims set to the prior's), gate kept
    D = ctx.donor("A")
    sel = rw & D["ak"] & okc & ctx.mA[:, None]
    dims = E["mu_p"].shape[-1] // M
    for nm, P in (("m", D["PJ"]), ("s", D["PX"])):
        G = []
        for subset in range(2 ** M):
            off = torch.tensor([(subset >> m) & 1 for m in range(M)], dtype=torch.bool, device=dev).repeat_interleave(dims)
            G.append(ctx.gain({"mu": torch.where(off, E["mu_p"], P["mu"]), "lv": P["lv"]}, sel, mean=True)[0])
        acc.mean(f"ho_{nm}", p, torch.stack(G, -1), sel)
    # R08: lags removed for every head through the removal mask (gate recomputed)
    if own["rm"] is not None:
        lagi = torch.arange(own["rm"].shape[-1], device=dev)
        rws = rw & ak & okc
        Ps = ctx.arm(own, rm=own["rm"] | (lagi > L0))            # long lags removed, own mask
        acc.mean("long_loss", p, ctx.gm[0] - ctx.gain(Ps, rws)[0], rws)
        acc.mean("g_rw", p, ctx.gm[0], rws)
        cut = D["rm"] | (lagi <= L0)                             # short lags removed, joint mask
        sgc = ctx.gain(ctx.arm(own, D["km"], cut), sel)[0] - ctx.gain(ctx.arm(ctx.ua("A"), D["km"], cut), sel)[0]
        sg = D["gJ"][0] - D["gX"][0]
        acc.mean("short_loss", p, sg - sgc, sel)
        acc.mean("sg", p, sg, sel)


# ----------------------------------------------------------------------------- 4. clamp (R44)
def step_clamp(ctx, acc):
    p, own = ctx.pos, ctx.ua("own")
    ak, okc, rw = _own(ctx)
    sel_all = ak & okc
    sel = rw & sel_all
    acc.mean("c_m", p, ctx.gm[0], sel)
    acc.mean("c_m_all", p, ctx.gm[0], sel_all)
    # mean query of the other half of the recordings (cross-fit, so a row never shapes its own profile)
    qbar = ctx.run["qbar"][1 - ctx.run["ghalf"][p]][:, None]
    for mode in ("pos", "static", "rev"):
        with core.clamp(ctx.run["model"], mode, qbar):
            P = ctx.arm(own)
        g = ctx.gain(P, sel_all)[0]
        acc.mean(f"c_{mode}", p, g, sel)
        acc.mean(f"c_{mode}_all", p, g, sel_all)


# ----------------------------------------------------------------------------- 5. clock (R34, R38)
def step_clock(ctx, acc):
    p, E, run = ctx.pos, ctx.E, ctx.run
    D = ctx.donor("A")
    own, don = ctx.ua("own"), ctx.ua("A")
    # the donor's own run: its FHR with its UA
    with torch.no_grad():
        od = run["model"](*run["task"]._build_forward_inputs(ctx.ab))
    sl = E["sl"]
    akd = ctx.any_key(don["km"], don["rm"])
    ak = ctx.any_key(own["km"], own["rm"])
    cj = ctx.clean(D["km"], D["rm"]) & ctx.mA[:, None]
    acc.mean("klh", p, ctx.matched["kl"], ak)                    # KL per head, to name the main head
    mu_p, lv_p = od["mu_prior"][:, sl], od["logvar_prior"][:, sl]
    mu_q, lv_q = od["mu_post"][:, sl], od["logvar_post"][:, sl]
    kd = 0.5 * ((mu_q - mu_p) ** 2 / lv_p.exp() + lv_p - lv_q + (lv_q - lv_p).exp() - 1.0)
    kd = kd.view(*kd.shape[:-1], D["PX"]["kl"].shape[-1], -1).sum(-1)
    nan = lambda x, m: x.masked_fill(~m, torch.nan)                               # noqa: E731
    series = {"gate": lambda el, kl, g: _logit(g), "kl": lambda el, kl, g: torch.log(kl + 0.01),
              "elag": lambda el, kl, g: el}
    x = {"ii": (_elag(ctx.matched["alpha"], ak), ctx.matched["kl"], nan(ctx.matched["gate"], ak)),
         "ij": (_elag(D["PX"]["alpha"], D["ak"]), D["PX"]["kl"], nan(D["PX"]["gate"], D["ak"])),
         "jj": (_elag(od["attn_weights"][:, sl], akd), kd, nan(od["event_gate"][:, sl, 0] if "event_gate" in od else torch.ones_like(kd[..., 0]), akd))}
    for q, f in series.items():
        v = {k: f(*x[k]) for k in x}
        for name, a, b in (("rU", "ij", "jj"), ("rF", "ij", "ii"), ("r0", "ii", "jj")):
            if q == "gate":
                r = core.pearson(v[a], v[b], cj)
            else:
                r = torch.stack([core.pearson(v[a][..., m], v[b][..., m], cj) for m in range(v[a].shape[-1])], -1)
            acc.seg(f"{name}_{q}", p, r)
    # R38: gains and gate under donor UA, locked on the donor's contraction onsets
    mj = _np(D["ak"] & ctx.mA[:, None]) > 0
    okc = _np(E["okc"]) > 0
    vals = [_np(t) for t in (D["gJ"][0], D["gX"][0], D["PJ"]["gate"], D["PX"]["gate"])]
    S, N, ev = np.zeros((ctx.B, 4, len(AGES))), np.zeros((ctx.B, 2, len(AGES))), np.zeros((ctx.B, 2))
    for i in range(ctx.B):
        for c, ons in enumerate((ctx.onsets("own")[i], ctx.onsets("A")[i])):
            ev[i, c] = lock(ons, [], mj[i] & okc[i])[2]
        on = ctx.onsets("A")[i]
        S[i, :2], N[i, 0], _ = lock(on, [np.nan_to_num(v[i]) for v in vals[:2]], mj[i] & okc[i])
        S[i, 2:], N[i, 1], _ = lock(on, [v[i] for v in vals[2:]], mj[i])
    acc.sum("d_s", p, S)
    acc.sum("d_n", p, N)
    acc.sum("d_ev", p, ev)


# ----------------------------------------------------------------------------- 6. events (R37, R17)
def first_opening(onsets, gate, ak, next_cap=45):
    """Per segment: histogram of the gate's first-opening latency after each incident onset (bins of 4 s,
    4 .. 180 s) and counts (incident, opened, never). Incident: the anchors 20 s before the onset up to
    the onset exist, have a key and a shut gate, and the follow-up is at least 60 s (15 anchors).
    Follow-up: the anchors after the onset up to 180 s, the next onset or the segment end. An onset with
    a shorter follow-up is right-censored and counts nowhere, opened or not. Never: no open anchor in
    the follow-up."""
    hist, cnt = np.zeros(next_cap), np.zeros(3)
    op, A = (gate >= GATE_OPEN) & ak, len(gate)
    ons = np.sort(np.asarray(onsets, int))
    for j, o in enumerate(ons):
        if o - 5 < 0 or not ak[o - 5:o + 1].all() or op[o - 5:o + 1].any():
            continue
        end = min(A, o + next_cap + 1, ons[j + 1] if j + 1 < len(ons) else A)
        fol = op[o + 1:end]
        if len(fol) < 15:
            continue
        if fol.any():
            cnt[:2] += 1
            hist[int(fol.argmax())] += 1
        else:
            cnt[[0, 2]] += 1
    return hist, cnt


def step_events(ctx, acc):
    p, E = ctx.pos, ctx.E
    D = ctx.donor("A")
    lagm = ctx.run["lag"]
    ak = _np(_own(ctx)[0]) > 0
    mj = _np(D["ak"] & ctx.mA[:, None]) > 0
    okc = _np(E["okc"]) > 0
    kl = [_np(P["kl"].sum(-1)) if lagm else np.zeros_like(okc, float) for P in (D["PJ"], D["PX"])]
    gains = [np.nan_to_num(_np(t)) for t in (D["gJ"][0], D["gX"][0])]
    other = [_np(D["PJ"]["gate"]), _np(D["PX"]["gate"])] + kl
    g_own = _np(ctx.matched["gate"])
    S, N, ev = np.zeros((ctx.B, 2, 6, len(AGES))), np.zeros((ctx.B, 2, 2, len(AGES))), np.zeros((ctx.B, 2))
    FH, FN = np.zeros((ctx.B, 2, 45)), np.zeros((ctx.B, 2, 3))
    for i in range(ctx.B):
        for c, ons in enumerate((ctx.onsets("own")[i], ctx.onsets("A")[i] if ctx.mA[i] else [])):
            S[i, c, :2], N[i, c, 0], ev[i, c] = lock(ons, [v[i] for v in gains], mj[i] & okc[i])
            S[i, c, 2:], N[i, c, 1], _ = lock(ons, [v[i] for v in other], mj[i])
            FH[i, c], FN[i, c] = first_opening(ons, g_own[i], ak[i])
    acc.sum("e_s", p, S)      # clock (own, donor) x series (matched gain, surA gain, matched gate, surA gate, matched KL, surA KL)
    acc.sum("e_n", p, N)      # clock x (gain anchors, gate / KL anchors)
    acc.sum("e_ev", p, ev)
    acc.sum("fo_hist", p, FH)  # clock (own onsets, donor onsets as sham) x latency bin, own-UA gate
    acc.sum("fo_n", p, FN)


# ----------------------------------------------------------------------------- 7. groups (class, delivery, time; R50)
def step_groups(ctx, acc):
    p, E, own = ctx.pos, ctx.E, ctx.ua("own")
    ak, okc, rw = _own(ctx)
    D = ctx.donor("A")
    sel = D["ak"] & ctx.mA[:, None] & rw & okc
    acc.mean("gm", p, D["gJ"][0], sel)
    acc.mean("gs", p, D["gX"][0], sel)
    acc.mean("gate", p, ctx.matched["gate"], ak)
    acc.mean("open", p, (ctx.matched["gate"] >= GATE_OPEN).float(), ak)
    if ctx.run["lag"]:
        acc.mean("kl", p, ctx.matched["kl"].sum(-1), ak)
        acc.mean("elag", p, _elag(ctx.matched["alpha"], ak), ctx.clean(own["km"], own["rm"]))
        slow = ctx.run["slow"]
        if slow is not None:                                     # R50: slow UA channels taken from the donor
            Pn = ctx.arm(ctx.swapped(own, ctx.ua("A"), slow), D["km"], D["rm"])
            acc.mean("g_slow", p, ctx.gain(Pn, sel)[0], sel)


# ----------------------------------------------------------------------------- 8. irf
def step_irf(ctx, acc):
    from teb_vae.coupling_irf.model import irf_delay, plant

    p, E, run, model = ctx.pos, ctx.E, ctx.run, ctx.run["model"]
    ch = model.irf_channel_index
    own, don = ctx.ua("own"), ctx.ua("A")
    okc, mA = E["okc"], ctx.mA[:, None]
    r, w = ctx.irf(own), ctx.irf(don)
    for tag, d, m in (("", r, okc), ("_w", w, okc & mA)):
        cov = m & d["ua_ok"]
        acc.mean("open" + tag, p, (d["a"] > 0)[..., ch].float(), cov)             # per head channel
        for nm in ("delta", "delta_expl"):
            g = ctx.gain({"mu": None, "delta": d[nm]}, m)[0]
            acc.mean(f"gap_{nm}{tag}", p, g, m)                                   # nats per cell
            acc.mean(f"gapsum_{nm}{tag}", p, g * E["n"], m)                       # nats per anchor
    c0 = int(ch[0])
    delay = core.STEP_S * irf_delay(r["h"][..., c0])[0]                           # s, stored clock
    opened = (r["a"][..., c0] > 0) & r["ua_ok"] & okc & torch.isfinite(delay)
    acc.seg("delay", p, torch.nanmedian(delay.masked_fill(~opened, torch.nan), 1).values)
    acc.mean("open0", p, (r["a"][..., c0] > 0).float(), okc & r["ua_ok"])
    # a response planted on the level channel at a known delay after the donor's UA
    keep = model.target_gate.keep_index if model.target_gate is not None else torch.arange(E["feats"].shape[-1])
    col = int(keep[c0])
    u = torch.where(don["lvl_ok"], don["lvl"], torch.full_like(don["lvl"], float("inf")))
    u = torch.where(don["lvl_ok"], don["lvl"] - u.amin(-1, keepdim=True), torch.zeros_like(don["lvl"]))
    err, share = [], []
    for d in PLANT_STEPS:
        feats = E["feats"].clone()
        feats[..., col] = plant(feats[..., col], u, torch.full((ctx.B,), float(d), device=u.device), depth=1.0)
        q = ctx.irf(don, feats)
        got = core.STEP_S * (irf_delay(q["h"][..., c0])[0] - d)
        op = (q["a"][..., c0] > 0) & q["ua_ok"] & okc & mA
        err.append(torch.nanmedian(got.masked_fill(~(op & torch.isfinite(got)), torch.nan), 1).values)
        share.append(torch.where((q["ua_ok"] & okc & mA).any(1), (op.float().sum(1) / (q["ua_ok"] & okc & mA).sum(1).clamp(min=1)), torch.nan))
    acc.seg("plant_err", p, torch.stack(err, -1))
    acc.seg("plant_open", p, torch.stack(share, -1))
    if run["lag"]:   # hybrid: the head's evidence against the lag-attention path
        ak = _own(ctx)[0]
        e0 = (r["z"][..., c0] / model.irf_head.gate_k).clamp(0, 3)
        e1 = (r["a"] > 0)[..., ch].float().mean(-1)
        D = ctx.donor("A")
        sg, gate = D["gJ"][0] - D["gX"][0], ctx.matched["gate"]
        mj = D["ak"] & mA & okc
        for nm, e in (("e0", e0), ("e1", e1)):
            acc.mean(nm, p, e, ak)
            acc.seg(f"r_{nm}_gate", p, core.pearson(e, gate, ak))
            acc.seg(f"r_{nm}_sg", p, core.pearson(e, sg, mj))
        acc.mean("gate", p, gate, ak)
        acc.mean("sg", p, sg, mj)


# ----------------------------------------------------------------------------- 9. raw (R28)
def step_raw(ctx, acc):
    """Model-free: FHR level around the UA peak of every detected contraction, on the step grid."""
    from teb_vae.lag_attn_cfs.nets.irf_source import ua_step_level

    run, b = ctx.run, ctx.rb
    dec = int(run["model"].geometry.decimation)
    T = b.weight.shape[1]
    lvl, ok = ua_step_level(b.fhr, b.fhr_valid if hasattr(b, "fhr_valid") else torch.ones_like(b.fhr, dtype=torch.bool), dec, T)
    ua, uok = ua_step_level(b.up, b.up_valid if hasattr(b, "up_valid") else torch.ones_like(b.up, dtype=torch.bool), dec, T)
    if not hasattr(b, "fhr_valid"):                                               # zero-padded gaps carry weight 0
        ok = ok & (b.weight > 0)
    (fm, fs), (um, us) = run["scale"]["fhr"], run["scale"]["up"]
    lvl, ua = _np(lvl) * fs + fm, _np(ua) * us + um                              # bpm and toco units
    ok, uok = _np(ok) > 0, _np(uok) > 0
    S, N, ev = np.zeros((ctx.B, 2, len(AGES))), np.zeros((ctx.B, 2, len(AGES))), np.zeros(ctx.B)
    centre = (dec * np.arange(T) - (dec - 1) / 2) / dec                          # step centres, in steps
    for i, e in enumerate(ctx.detected[1]):
        if not ok[i].any():
            continue
        dev = lvl[i] - np.median(lvl[i][ok[i]])                                  # bpm from the segment's median
        for pk in np.asarray(e["peak_raw"], float) / dec:
            bins = np.floor((core.STEP_S * (centre - pk) - (AGES[0] - 2.0)) / 4.0).astype(int)
            for k, (v, m) in enumerate(((dev, ok[i]), (ua[i], uok[i]))):
                use = m & (bins >= 0) & (bins < len(AGES))
                np.add.at(S[i, k], bins[use], v[use])
                np.add.at(N[i, k], bins[use], 1.0)
            ev[i] += 1
    acc.sum("r_s", ctx.pos, S, by_bin=True)     # (FHR deviation, UA level) x age bin
    acc.sum("r_n", ctx.pos, N, by_bin=True)
    acc.sum("r_ev", ctx.pos, ev, by_bin=True)


STEPS = {"gain": step_gain, "shift": step_shift, "heads": step_heads, "clamp": step_clamp, "clock": step_clock,
         "events": step_events, "groups": step_groups, "irf": step_irf, "raw": step_raw}
