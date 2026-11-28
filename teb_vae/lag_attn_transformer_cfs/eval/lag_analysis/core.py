"""Model, data, donors and the per-batch arms of the lag analysis.

Ported from the study of checkpoint 369 (`tmp/lag_analysis_369`: `common.py`, `capture/replay.py`,
`phase2/arms/arms.py`, `phase2/arms/donors.py`). Differences from the study are marked "Differs:".

Terms. A *segment* is one shard row (T steps of 4 s). An *anchor* is a step the model forecasts from.
An *arm* is one forecast of the recipient's FHR with some UA: its own (matched), another recording's
(surA), the same recording's at least 40 min away (surB), zeroed UA (null), or a changed copy of its own.
Only the UA side is re-run for an arm. The prior, the FHR state, the persistence term, the target and
the latent noise always come from the recipient. Shapes: B rows, T steps, A anchors, M heads, L lags
(index = lag in steps, 0 = the anchor's own step), H horizon steps, C target channels.
"""
from __future__ import annotations

import copy
import hashlib
import math
import zlib
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import torch

STEP_S = 4.0                  # seconds per step
MIN_CELL_SHARE = 0.5          # an anchor counts when at least this share of its scored cells is valid
REWARD_STEPS = 15             # reward window: onset <= anchor <= onset + 15 steps (0-60 s)
B_MIN_S = 2400.0              # surB: same recording, start at least 40 min away
TIGHT, WIDE = (20.0, 0.05), (40.0, 0.10)   # surA calipers: minutes to delivery, UA valid fraction
MAX_EDGES = 40                # caliper-eligible donors kept per recipient (speed)
CLASSES = ("acidosis", "hie", "healthy_bg", "healthy_no_bg")
SUBGROUPS = tuple(f"{c}_{m}" for c in CLASSES for m in ("cs", "no_cs"))
TIME_BINS = (0.0, 30.0, 60.0, 120.0, 240.0)    # minutes before delivery


# ----------------------------------------------------------------------------- checkpoint and data
def load_task(checkpoint, device):
    """The model of the checkpoint inside its task, in eval mode (the eval package's own loader)."""
    from teb_vae.lag_attn_cfs.eval.probe import load_task as load
    from teb_vae.lag_attn_transformer_cfs.eval.binding import TRF_CFS_BINDING

    return load(Path(checkpoint), device, binding=TRF_CFS_BINDING)


def variant(model) -> dict:
    """What the checkpoint has: a lag attention, an IRF head, or both (hybrid)."""
    lag, irf = hasattr(model, "lag_attn"), getattr(model, "irf_head", None) is not None
    name = "hybrid (lag attention + IRF readout)" if lag and irf else "lag attention" if lag else "IRF only"
    return {"lag": lag, "irf": irf, "name": name}


def loader_config(config_path, checkpoint) -> dict:
    """`dataset_config.dataloader_config` of the resolved config: `--config`, else the file beside the
    checkpoint. The checkpoint itself does not store the loader settings."""
    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_cfs.eval.probe import resolved_config_for

    try:
        path = config_path or resolved_config_for(Path(checkpoint))
    except FileNotFoundError as err:
        raise SystemExit(
            "No resolved config found. Pass --config <resolved_config.yaml>. Needed keys: "
            "dataset_config.dataloader_config.normalize_fields and .dataset_kwargs (load_fields, "
            f"trim_minutes, ua_step_mask, fhr_step_mask, emit_quality_mask). ({err})")
    return load_config(str(path))["dataset_config"]["dataloader_config"]


def open_dataset(loader_cfg, shards, stats):
    """The model's loader dataset (settings of the resolved config) on the given shards, without its sample cache."""
    from hdf5_dataset.hdf5_dataset import CombinedHDF5Dataset

    kw = copy.deepcopy(loader_cfg["dataset_kwargs"])
    kw["cache_size"] = 0
    if kw.get("load_fields") is not None:      # the raw traces, the weight and the segment key are always needed
        kw["load_fields"] = list(dict.fromkeys(list(kw["load_fields"]) + ["fhr", "up", "weight", "guid", "epoch"]))
    return CombinedHDF5Dataset(paths=list(shards), stats_path=str(stats),
                               normalize_fields=loader_cfg["normalize_fields"], **kw)


def subgroup_of(path) -> str:
    """Shard file stem, or the folder name when only that is one of the eight subgroup names."""
    p = Path(path)
    return p.parent.name if p.stem not in SUBGROUPS and p.parent.name in SUBGROUPS else p.stem


def class_of(subgroup):
    """(class, delivered by caesarean section) of a subgroup name, or (None, None)."""
    if subgroup not in SUBGROUPS:
        return None, None
    cs = not subgroup.endswith("_no_cs")
    return subgroup[: -3 if cs else -6], cs


def build_index(ds) -> dict:
    """One entry per dataset row, from the shards' 1-D fields only: guid, epoch (s, delivery = 0),
    subgroup, UA valid fraction."""
    files, rows = np.asarray(ds.index_map).T
    guid, epoch, frac = np.empty(len(files), object), np.zeros(len(files)), np.full(len(files), np.nan)
    for f, path in enumerate(ds.paths):
        m = files == f
        with h5py.File(path, "r") as h:
            guid[m] = h["guid"].asstr()[:][rows[m]]
            epoch[m] = h["epoch"][:][rows[m]]
            if "up_valid_frac" in h:
                frac[m] = h["up_valid_frac"][:][rows[m]]
    sub = np.array([subgroup_of(p) for p in ds.paths], object)[files]
    # A recording in two subgroups would get one subgroup silently and stall `select_rows`.
    pairs = sorted(set(zip(guid, sub)))
    clash = sorted({g for (g, _), (h, _) in zip(pairs, pairs[1:]) if g == h})
    if clash:
        raise ValueError(f"{len(clash)} GUID(s) appear in more than one subgroup (shard), e.g. {clash[:3]}; "
                         "each recording must belong to one subgroup.")
    return {"guid": guid, "epoch": epoch, "up_valid_frac": frac, "subgroup": sub, "file": files}


def _rank(text, seed) -> int:
    return int.from_bytes(hashlib.blake2b(f"{text}|{seed}|lag".encode(), digest_size=8).digest(), "big")


def select_rows(ix, max_guids, seed) -> np.ndarray:
    """Dataset rows of the recordings used: all, or `max_guids` of them taken in turn from each
    subgroup in a seeded hash order (equal allocation while a subgroup has recordings left)."""
    pairs = sorted(set(zip(ix["subgroup"], ix["guid"])))
    if max_guids and max_guids < len(pairs):
        by = {}
        for s, g in sorted(pairs, key=lambda p: _rank(p[1], seed)):
            by.setdefault(s, []).append(g)
        keep, depth = set(), 0
        while len(keep) < max_guids:
            for s in sorted(by):
                if depth < len(by[s]) and len(keep) < max_guids:
                    keep.add(by[s][depth])
            depth += 1
        return np.flatnonzero(np.isin(ix["guid"], sorted(keep)))
    return np.arange(len(ix["guid"]))


# ----------------------------------------------------------------------------- donors
def _match(t, v, g, rec, don, cal, rng):
    """Seeded maximum matching of recipients to donors of another recording inside the calipers.
    Returns the donor position per recipient (-1 = none). A donor is used at most once."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    out = np.full(len(rec), -1)
    if not len(rec) or not len(don):
        return out
    pr, don_p = rng.permutation(len(rec)), rng.permutation(don)
    order = don_p[np.argsort(t[don_p], kind="stable")]
    col = np.empty(len(t), int)
    col[don_p] = np.arange(len(don_p))
    ri, ci = [], []
    for i, r in enumerate(rec[pr]):
        cand = order[np.searchsorted(t[order], t[r] - cal[0], "left"):np.searchsorted(t[order], t[r] + cal[0], "right")]
        ok = g[cand] != g[r]
        if np.isfinite(v[r]):
            ok &= np.abs(v[cand] - v[r]) <= cal[1]
        cand = cand[ok]
        if len(cand) > MAX_EDGES:
            cand = rng.choice(cand, MAX_EDGES, replace=False)
        ri.append(np.full(len(cand), i))
        ci.append(col[cand])
    graph = csr_matrix((np.ones(sum(map(len, ri)), np.int8), (np.concatenate(ri), np.concatenate(ci))),
                       shape=(len(rec), len(don_p)))
    m = maximum_bipartite_matching(graph, perm_type="column")
    out[pr] = np.where(m >= 0, don_p[np.maximum(m, 0)], -1)
    return out


def assign_donors(ix, rows, seed):
    """surA and surB donor row for every recipient row (-1 = none).

    surA: another recording of the same subgroup, time to delivery within 20 min and UA valid fraction
    within 0.05; recipients left over are matched once more at 40 min and 0.10 among unused donors.
    Donors come from all rows of the shards, also outside `--max-guids`.
    Differs: the study also matched the contraction count (+-1, +-2) of its own detector, which needs a
    pass over the raw UA. Its seeds differ too, so the pairs are not the study's pairs.
    surB: the same recording, start at least 40 min away, the nearest such segment (seeded tie break).
    """
    a, b = np.full(len(rows), -1), np.full(len(rows), -1)
    t, v, g = ix["epoch"] / 60.0, ix["up_valid_frac"], np.unique(ix["guid"], return_inverse=True)[1]
    where = {r: i for i, r in enumerate(rows)}
    for sub in np.unique(ix["subgroup"][rows]):
        rng = np.random.default_rng([zlib.crc32(str(sub).encode()), seed, 369])
        don = np.flatnonzero(ix["subgroup"] == sub)
        rec = np.array([r for r in don if r in where])
        d = _match(t, v, g, rec, don, TIGHT, rng)
        left = d < 0
        d[left] = _match(t, v, g, rec[left], np.setdiff1d(don, d[~left]), WIDE, rng)
        a[[where[r] for r in rec]] = d
        for r in rec:
            same = don[(g[don] == g[r]) & (np.abs(ix["epoch"][don] - ix["epoch"][r]) >= B_MIN_S)]
            if len(same):
                dist = np.abs(ix["epoch"][same] - ix["epoch"][r])
                b[where[r]] = rng.choice(same[dist == dist.min()])
    return a, b


class Triplets(torch.utils.data.Dataset):
    """(recipient, surA donor, surB donor) items through the dataset's own `__getitem__`. A missing
    donor, or `donors=False`, returns the recipient in its place."""

    def __init__(self, ds, rows, a, b, donors=True):
        self.ds, self.rows, self.a, self.b, self.donors = ds, list(rows), list(a), list(b), donors

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.ds[self.rows[i]]
        if not self.donors:
            return r, r, r
        return r, (self.ds[self.a[i]] if self.a[i] >= 0 else r), (self.ds[self.b[i]] if self.b[i] >= 0 else r)


def _collate3(items):
    from hdf5_dataset.hdf5_dataset import attribute_dict_collate

    return tuple(attribute_dict_collate([it[k] for it in items]) for k in range(3))


def make_loader(ds, rows, a, b, batch_size, num_workers, donors=True):
    return torch.utils.data.DataLoader(
        Triplets(ds, rows, a, b, donors), batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=_collate3, multiprocessing_context="spawn" if num_workers else None,
        prefetch_factor=2 if num_workers else None)


# ----------------------------------------------------------------------------- small tensor helpers
def roll(x, shift):
    """out[:, t] = x[:, t - shift] along dim 1, and inside[t] = (0 <= t - shift < T)."""
    T = x.shape[1]
    idx = torch.arange(T, device=x.device) - shift
    return x[:, idx.clamp(0, T - 1)], (idx >= 0) & (idx < T)


def pearson(x, y, m, min_n=40):
    """Pearson correlation over the anchors of each row on mask m; NaN below `min_n` anchors."""
    m = m & torch.isfinite(x) & torch.isfinite(y)
    n = m.sum(1)
    x, y = torch.where(m, x, 0.0), torch.where(m, y, 0.0)
    xc = torch.where(m, x - (x.sum(1) / n.clamp(min=1))[:, None], 0.0)
    yc = torch.where(m, y - (y.sum(1) / n.clamp(min=1))[:, None], 0.0)
    r = (xc * yc).sum(1) / torch.sqrt((xc ** 2).sum(1) * (yc ** 2).sum(1))
    return torch.where(n >= min_n, r, torch.nan)


def eps_for(guid, epoch, K, anchors, d_z, seed=0):
    """Antithetic latent noise (K, A, D) of one segment, the same for the base branch and every arm.
    With seed 0 this is the study's rule (`capture/replay.py: EPS_SEED_RULE`)."""
    key = f"{guid}|{math.floor(float(epoch))}|eps369" + (f"|{seed}" if seed else "")
    s = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big") % (2 ** 63)
    # ceil(K / 2) draws, so an odd K >= 3 still gets K rows; even K and K = 1 are unchanged.
    e = torch.randn((K + 1) // 2, anchors, d_z, generator=torch.Generator("cpu").manual_seed(s))
    return torch.cat([e, -e])[:K]


def kappa_terms(alpha, alpha_s, delta):
    """Tracking index terms of PREREG R01 per (B, A, M). On I = {l: 0 <= l, l + delta <= L - 1}, each
    renormalised on I: f = alpha (a fixed filter keeps it), r(l) = alpha(l + delta) (a tracker moves
    with the UA), a' = the attention of the shifted run. num = <a' - f, r - f>, den = <r - f, r - f>."""
    def norm(x):
        s = x.sum(-1, keepdim=True)
        return torch.where(s > 0, x / s.clamp(min=1e-30), torch.zeros_like(x))

    L = alpha.shape[-1]
    lo, hi = max(0, -delta), L - max(0, delta)
    f, r, ap = norm(alpha[..., lo:hi]), norm(alpha[..., lo + delta:hi + delta]), norm(alpha_s[..., lo:hi])
    return ((ap - f) * (r - f)).sum(-1), ((r - f) ** 2).sum(-1)


@contextmanager
def clamp(model, mode, qbar):
    """Replace the content-dependent attention of every head by a profile that does not read the UA.

    `pos`: the keys are zeroed, so the score is the position term (query x lag embedding + lag bias)
    plus the validity bias: the study's alpha^pos. `static`: the query is also replaced by the mean
    query `qbar`, so every anchor gets one profile: alpha^static. `rev`: static with the lag order
    reversed inside lags 0-14 (the control). Values, masks, gate and posterior run as in the forward.
    """
    la = model.lag_attn
    hooks = [la.W_k.register_forward_hook(lambda m, i, o: torch.zeros_like(o))]
    if mode != "pos":
        hooks.append(la.W_q.register_forward_hook(lambda m, i, o: qbar.to(o.dtype).expand_as(o)))
    bias = getattr(la, "lag_score_bias", None)                  # absent under some lag-bias initialisations
    saved = la.lag_embeddings.detach().clone(), None if bias is None else bias.detach().clone()
    span = min(15, la.L)
    try:
        if mode == "rev":
            with torch.no_grad():
                la.lag_embeddings[:span] = saved[0][:span].flip(0)
                if bias is not None:
                    bias[:, :span] = saved[1][:, :span].flip(-1)
        yield
    finally:
        for h in hooks:
            h.remove()
        with torch.no_grad():
            la.lag_embeddings.copy_(saved[0])
            if bias is not None:
                bias.copy_(saved[1])


def detect(task, batch):
    """The contraction detector of the model's objective on each row: (reward (B, T) bool, list of event
    dicts in raw samples). It reads the raw UA where the FHR weight and the UA mask are valid."""
    from teb_vae.lag_attn_rws import event_weight as EW

    dec = int(task.orig_model.geometry.decimation)
    up, w = batch.up.double().cpu().numpy(), batch.weight.double().cpu().numpy()
    uv = getattr(batch, "up_valid", None)
    uv = None if uv is None else uv.cpu().numpy()
    reward, events = np.zeros(w.shape, bool), []
    for i in range(len(w)):
        ev = EW._detect_events(up[i], w[i], decimation=dec, detector=task._event_detector(),
                               raw_valid=None if uv is None else uv[i])
        events.append(ev)
        for o in (np.asarray(ev["onset_raw"], np.int64) + dec - 1) // dec:
            reward[i, o:o + REWARD_STEPS + 1] = True
    return torch.from_numpy(reward), events


# ----------------------------------------------------------------------------- one batch
class Ctx:
    """One batch: the recipient's forward, then arms computed on demand and kept for the other steps."""

    def __init__(self, run, batches, pos):
        self.run, self.pos, self._memo = run, pos, {}
        task, model, dev = run["task"], run["model"], run["device"]
        self.rb, self.ab, self.bb = (task.transfer_batch_to_device(b, dev, 0) for b in batches)
        self.B = len(self.rb.epoch)
        t = lambda x: torch.as_tensor(np.asarray(x), device=dev)              # noqa: E731
        self.mA, self.mB = t(run["donor_a"][pos] >= 0), t(run["donor_b"][pos] >= 0)

    def memo(self, key, fn):
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    @property
    def detected(self):
        """(reward (B, T), events per row) of the recipient, from the model's contraction detector."""
        return self.memo("detected", lambda: detect(self.run["task"], self.rb))

    # ---- the recipient's forward and what scoring needs (capture/replay.py: encode)
    @property
    def E(self):
        return self.memo("E", self._encode)

    def _encode(self):
        from teb_vae.lag_attn_rws.nets.raw_masks import VALID_THRESHOLD

        task, model, b = self.run["task"], self.run["model"], self.rb
        inputs = list(task._build_forward_inputs(b))
        with torch.no_grad():
            out = model(*inputs)
        anchors = out["anchor_index"]
        a0, A = int(anchors[0, 0]), anchors.shape[1]
        assert bool((anchors == torch.arange(a0, a0 + A, device=anchors.device)).all()), "anchors are not dense"
        sl = slice(a0, a0 + A)
        feats = torch.cat([inputs[0], inputs[1]], -1)
        weight = task._target_quality_weight(b, b.weight)
        if hasattr(model, "scored_weight"):        # the forecast clock's pooled validity, as the objective scores
            weight = model.scored_weight(weight)
        valid = (weight >= VALID_THRESHOLD).to(feats.dtype)
        future = valid[:, 1:].unfold(1, model.horizon, 1)[:, sl]                      # steps t + 1 + tau
        step_valid = valid[:, sl, None] * future                                      # (B, A, H)
        like = model.forecast_likelihood_kwargs()
        cm = like.get("cell_mask")
        cells = step_valid[..., None] * (1.0 if cm is None else cm)
        cells = cells.expand(*step_valid.shape, out["mu_base"].shape[-1])
        n = cells.sum((-2, -1))
        scored = float(cells.shape[-1] * cells.shape[-2] if cm is None else cm.sum())
        reward, events = self.detected
        inputs += [None] * (10 - len(inputs))
        E = {"out": out, "inputs": inputs, "sl": sl, "A": A, "a0": a0, "feats": feats, "like": like,
             "target": model._build_forecast_target(feats, anchors), "step_valid": step_valid,
             "pers": out.get("persistence"), "cells": cells, "n": n, "okc": n >= MIN_CELL_SHARE * scored,
             "reward": reward.to(n.device)[:, sl], "events": events,
             "mu_p": out["mu_prior"][:, sl], "lv_p": out["logvar_prior"][:, sl]}
        E["eps"] = torch.stack([eps_for(g, e, self.run["K"], A, model.d_z, self.run["seed"])
                                for g, e in zip(b.guid, b.epoch.tolist())], 1).to(n.device)   # (K, B, A, D)
        return E

    # ---- the UA side (phase2/arms/arms.py: ua_side, null_state, rolled)
    def _ua_from(self, inputs, h=None, km=None):
        model = self.run["model"]
        u, v, rm = inputs[2], inputs[5], inputs[6]
        ua = {"u": u, "v": v, "rm": None if rm is None else rm.bool(), "lvl": inputs[7], "lvl_ok": inputs[8],
              "q": None, "h": h, "km": km}
        if not self.run["lag"]:
            return ua
        with torch.no_grad():
            if v is not None and getattr(model, "source_mask_mode", "hard") in ("soft", "input"):
                ua["q"] = model.source_quality_input(v, u.dtype)
            if h is None:
                source = u if model.source_gate is None else model.source_gate(u)
                ua["h"] = model.encode_source_kv(source, ua["q"])
                ua["km"] = None if v is None else model.source_key_mask(
                    v, model.build_lag_mask(u.shape[1], u.device), u.dtype).bool()
        return ua

    def ua(self, which):
        """UA of 'own', 'A' (surA donor), 'B' (surB donor) or 'null' (own UA values zeroed, flags kept)."""
        def build():
            task, model, o = self.run["task"], self.run["model"], self.E["out"]
            if which == "own":
                km = o.get("source_key_mask")
                return self._ua_from(self.E["inputs"], o.get("source_state"), None if km is None else km.bool())
            if which == "null":
                own = dict(self.ua("own"))
                if self.run["lag"]:
                    z = torch.zeros_like(own["u"])
                    with torch.no_grad():
                        own["h"] = model.encode_source_kv(z if model.source_gate is None else model.source_gate(z), own["q"])
                if own["lvl"] is not None:
                    own["lvl"] = torch.zeros_like(own["lvl"])
                return own
            inputs = list(task._build_forward_inputs(self.ab if which == "A" else self.bb))
            return self._ua_from(inputs + [None] * (10 - len(inputs)))
        return self.memo(("ua", which), build)

    def shifted(self, ua, shift):
        """UA delayed by `shift` steps (h'[s] = h[s - shift]), masks and UA level moved with it. An
        anchor whose shifted UA anchor lies outside the segment loses all its keys."""
        out = dict(ua)
        if ua["h"] is not None:
            out["h"], inside = roll(ua["h"], shift)
            if ua["km"] is not None:
                out["km"] = roll(ua["km"], shift)[0] & inside[None, :, None]
            if ua["rm"] is not None:
                out["rm"] = roll(ua["rm"], shift)[0] | ~inside[None, :, None]
        if ua["lvl"] is not None:
            out["lvl"], inside = roll(ua["lvl"], shift)
            out["lvl_ok"] = roll(ua["lvl_ok"], shift)[0] & inside[None]
        out["inside"] = inside[self.E["sl"]][None].expand(self.B, -1)
        return out

    def swapped(self, own, don, chans):
        """Own UA with the declared source channels `chans` (bool, c_u) taken from the donor, values
        and flags (round2/sweeps/arms2.py: hybrid)."""
        pick = lambda x, y: torch.where(chans.view(1, 1, -1), x, y)       # noqa: E731
        inputs = list(self.E["inputs"])
        inputs[2] = pick(don["u"], own["u"])
        inputs[5] = None if own["v"] is None else pick(don["v"], own["v"])
        return self._ua_from(inputs)

    def joint(self, x, y):
        """Joint mask of two UA: both keys valid, a key removed in either is removed (PREREG B)."""
        km = None if x["km"] is None else x["km"] & y["km"]
        rm = None if x["rm"] is None else x["rm"] | y["rm"]
        return km, rm

    def any_key(self, km, rm):
        """(B, A): at least one key is left for the attention."""
        if not self.run["lag"]:
            return torch.ones(self.B, self.E["A"], dtype=torch.bool, device=self.run["device"])
        model = self.run["model"]
        kept = model.build_lag_mask(self.E["feats"].shape[1], self.run["device"]).bool()[None]
        if km is not None and getattr(model, "source_mask_mode", "hard") == "hard":
            kept = kept & km
        if rm is not None:
            kept = kept & ~rm
        return kept.any(-1)[:, self.E["sl"]].expand(self.B, -1)

    def clean(self, km, rm, delta=0):
        r"""(B, A) clean anchors: all L keys valid, none removed, all warm in UA time, and the UA anchor
        $t - \delta$ of a shift $\delta$ inside the segment."""
        E, model = self.E, self.run["model"]
        combined = getattr(model, "_combined_source_steps", None)
        waits = combined() if callable(combined) else getattr(model, "source_warmup_steps", None)
        warm = max(waits or (0,)) + model.lag_attn.L - 1
        src = torch.arange(E["a0"], E["a0"] + E["A"], device=self.run["device"]) - delta
        ok = ((src >= warm) & (src < E["feats"].shape[1]))[None]
        if km is not None:
            ok = ok & km[:, E["sl"]].all(-1)
        if rm is not None:
            ok = ok & ~rm[:, E["sl"]].any(-1)
        return ok.expand(self.B, -1)

    # ---- posterior of an arm (the model's own attention, gate and posterior head)
    def arm(self, ua, km="ua", rm="ua"):
        """Posterior of the recipient with this UA under the masks (default: the UA's own).

        Lag attention: latent mean and log-variance at the anchors, attention (B, A, M, L), gate (B, A),
        KL per head (B, A, M). IRF only: the posterior is the prior and the head gives a mean shift
        `delta` (B, A, H, C); `gate` is the share of the head's channels whose gate is open.
        """
        model, E, o = self.run["model"], self.E, self.E["out"]
        km, rm = (ua["km"] if km == "ua" else km), (ua["rm"] if rm == "ua" else rm)
        with torch.no_grad():
            if not self.run["lag"]:
                d = self.irf(ua)
                return {"mu": None, "delta": d["delta"], "gate": (d["a"] > 0)[..., model.irf_channel_index].float().mean(-1),
                        "kl": None, "alpha": None}
            query = torch.cat([o["mu_prior"], o["logvar_prior"]], -1) if model.query_uses_logvar else o["mu_prior"]
            ev = None
            if getattr(model, "irf_gate_evidence", False):
                ev = model.irf_source(o["irf_target"], ua["lvl"], ua["lvl_ok"], o["irf_step_valid"],
                                      o["anchor_index"])["evidence"].detach()
            att = model.attend_lagged_source(model.query_proj(query), ua["h"], km, removed=rm, evidence=ev)
            mu, lv = model.posterior_head(o["target_state"], att.heads, o["mu_prior"], o["raw_logvar_prior"], gate=att.gate)
        sl = E["sl"]
        gate = torch.ones_like(mu[:, sl, 0]) if att.event_gate is None else att.event_gate[:, sl, 0]
        return {"mu": mu[:, sl], "lv": lv[:, sl], "delta": None, "gate": gate, "alpha": att.alpha[:, sl],
                "kl": self.kl(mu[:, sl], lv[:, sl])}

    def kl(self, mu_q, lv_q):
        """Closed-form KL(q || p) per head (B, A, M); head m owns the m-th block of latent dimensions."""
        mu_p, lv_p, M = self.E["mu_p"], self.E["lv_p"], self.run["model"].lag_attn.num_heads
        k = 0.5 * ((mu_q - mu_p) ** 2 / lv_p.exp() + lv_p - lv_q + (lv_q - lv_p).exp() - 1.0)
        return k.view(*k.shape[:-1], M, -1).sum(-1)

    def irf(self, ua, feats=None):
        """The IRF head for one UA against the recipient's (or the given) target features."""
        E = self.E
        with torch.no_grad():
            return self.run["model"].irf_source(E["feats"] if feats is None else feats, ua["lvl"], ua["lvl_ok"],
                                                E["inputs"][9], E["out"]["anchor_index"])

    @property
    def matched(self):
        """The forward's own posterior (own UA, own masks)."""
        def build():
            o, sl = self.E["out"], self.E["sl"]
            if not self.run["lag"]:
                return self.arm(self.ua("own"))
            gate = o.get("event_gate")
            gate = torch.ones_like(o["mu_post"][:, sl, 0]) if gate is None else gate[:, sl, 0]
            return {"mu": o["mu_post"][:, sl], "lv": o["logvar_post"][:, sl], "delta": None, "gate": gate,
                    "alpha": o["attn_weights"][:, sl], "kl": self.kl(o["mu_post"][:, sl], o["logvar_post"][:, sl])}
        return self.memo("matched", build)

    # ---- scoring (capture/replay.py: decode_scores, gain_k8; phase2/arms/arms.py: Scorer)
    def _score(self, mu_f, lv_f, target, step_valid):
        """Log score per cell (nats, higher is better) under the model's own forecast density: scored
        cells, AR(1) residual and Student-t as configured; no horizon, channel or event weight."""
        from teb_vae.lag_attn_rws.nets.losses import raw_sample_score

        s = -raw_sample_score(mu_f, target, likelihood=str(self.run["task"].hparams.get("likelihood", "gaussian_nll")),
                              logvar=lv_f, step_mask=step_valid, **self.E["like"])
        return s * step_valid[..., None]

    def _base(self, k):
        """Base branch (prior latent) forecast and score for noise sample k (None = at the mean)."""
        def build():
            E = self.E
            z = E["mu_p"] if k is None else E["mu_p"] + E["eps"][k] * torch.exp(0.5 * E["lv_p"])
            with torch.no_grad():
                mu_f, lv_f = self.run["model"].decoder(z, persistence=E["pers"])
                s = self._score(mu_f, lv_f, E["target"], E["step_valid"])
            return (mu_f, lv_f, s) if self.run["irf"] else (None, None, s)   # forecasts kept for mean shifts only
        return self.memo(("base", k), build)

    def gain(self, P, sel=None, mean=False):
        """Forecast gain of an arm = log score(full) - log score(base) per valid scored cell, averaged
        over the K shared noise samples (`mean=True`: at the latent means).

        Returns the per-anchor gain (B, A) in nats per cell (NaN without a valid cell or outside `sel`)
        and the per-cell gains (B, A, H, C). With `sel` (B, A) only those anchors are decoded.
        """
        E, model = self.E, self.run["model"]
        full = sel is None
        b, a = (None, None) if full else sel.nonzero(as_tuple=True)
        pick = (lambda x: x) if full else (lambda x: None if x is None else x[b, a][None])
        acc = torch.zeros_like(E["target"] if full else E["target"][b, a][None])
        ks = [None] if mean else range(E["eps"].shape[0])
        with torch.no_grad():
            for k in ks if acc.numel() else []:
                base = self._base(k)
                if P["mu"] is None:
                    mu_f, lv_f = pick(base[0]), pick(base[1])
                else:
                    z = pick(P["mu"])
                    if k is not None:
                        z = z + pick(E["eps"][k]) * torch.exp(0.5 * pick(P["lv"]))
                    mu_f, lv_f = model.decoder(z, persistence=pick(E["pers"]))
                if P.get("delta") is not None:
                    mu_f = mu_f + pick(P["delta"])
                acc += self._score(mu_f, lv_f, pick(E["target"]), pick(E["step_valid"])) - pick(base[2])
        acc /= len(ks)
        if not full:
            cell = torch.zeros_like(E["target"])
            cell[b, a] = acc[0]
            acc = cell
        g = (acc * E["cells"]).sum((-2, -1)) / E["n"].clamp(min=1)
        g = g.masked_fill(E["n"] == 0, torch.nan)
        return (g if full else g.masked_fill(~sel, torch.nan)), acc

    def gain_tau(self, cell):
        """Per-cell gains (B, A, H, C) -> mean over valid cells per horizon step (B, A, H) and per
        channel block (B, A, 2): scattering, phase."""
        E, model = self.E, self.run["model"]
        keep = model.target_gate.keep_index if model.target_gate is not None else torch.arange(cell.shape[-1], device=cell.device)
        first = (keep < model.TARGET_BLOCK_SPLIT).to(cell.dtype)
        blocks = torch.stack([first, 1.0 - first], -1)                              # (C, 2)
        n_tau = E["cells"].sum(-1)
        n_blk = E["cells"].sum(-2) @ blocks
        tau = (cell * E["cells"]).sum(-1) / n_tau.clamp(min=1)
        blk = ((cell * E["cells"]).sum(-2) @ blocks) / n_blk.clamp(min=1)
        return tau.masked_fill(n_tau == 0, torch.nan), blk.masked_fill(n_blk == 0, torch.nan)

    @property
    def gm(self):
        """Gain of the matched arm on every anchor: (per anchor, per cell)."""
        return self.memo("gm", lambda: self.gain(self.matched))

    # ---- donor arms under the joint mask (phase2/arms/run_arms.py: donor_arm, mj_gain)
    def donor(self, which, reward_only=False):
        """Donor arm 'A' or 'B'. PX: the donor's UA; PJ: the own UA; both under the joint mask.
        `ak`: anchors with a key left; `gX`, `gJ`: their gains (per anchor, per cell). With
        `reward_only` the gains are decoded on reward-window anchors only."""
        def build():
            E, own, don = self.E, self.ua("own"), self.ua(which)
            km, rm = self.joint(own, don)
            D = {"km": km, "rm": rm, "PX": self.arm(don, km, rm), "ak": self.any_key(km, rm)}
            sel = None
            if reward_only:
                sel = E["reward"] & D["ak"] & E["okc"] & (self.mB if which == "B" else self.mA)[:, None]
            D["gX"] = self.gain(D["PX"], sel)
            if not self.run["lag"]:
                D["PJ"], D["gJ"] = self.matched, self.gm
                return D
            D["PJ"] = self.arm(own, km, rm)
            # The matched gain where the joint mask equals the own mask (the same latent), decoded only
            # where the mask differs.
            diff = torch.zeros_like(D["ak"])
            for x, y in ((km, own["km"]), (rm, own["rm"])):
                if x is not None:
                    diff |= (x != y).any(-1)[:, E["sl"]]
            g, cell = (t.clone() for t in self.gm)
            d = diff if sel is None else diff & sel
            g2, cell2 = self.gain(D["PJ"], d)
            g[d], cell[d] = g2[d], cell2[d]
            D["gJ"] = (g if sel is None else g.masked_fill(~sel, torch.nan)), cell
            return D
        return self.memo(("donor", which, reward_only), build)

    def onsets(self, which="own"):
        """Contraction onsets per row as anchor indices ('own' or donor 'A'), from the detector's
        `onset_raw` on the step grid of the reward mask. Every onset is kept: also one before the first
        anchor (its later anchors still have an age) and one less than 64 s after the previous one."""
        def build():
            events = self.detected[1] if which == "own" else detect(self.run["task"], self.ab)[1]
            dec = int(self.run["model"].geometry.decimation)
            return [
                ((np.asarray(e["onset_raw"], np.int64) + dec - 1) // dec - self.E["a0"]).astype(float)
                for e in events
            ]
        return self.memo(("onsets", which), build)

    def since_onset(self):
        """(B, A) steps from the latest own onset at or before each anchor; infinity without one."""
        def build():
            out = np.full((self.B, self.E["A"]), np.inf)
            a = np.arange(self.E["A"])
            for i, ons in enumerate(self.onsets("own")):
                ons = np.sort(ons)
                k = np.searchsorted(ons, a, "right") - 1
                out[i] = np.where(k >= 0, a - ons[np.maximum(k, 0)], np.inf) if len(ons) else np.inf
            return torch.as_tensor(out, device=self.run["device"])
        return self.memo("since_onset", build)
