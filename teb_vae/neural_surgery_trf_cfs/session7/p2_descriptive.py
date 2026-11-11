r"""D2: descriptive mechanism tables of the epoch-424 s6 checkpoint, from the per-anchor dump.

Only split codes 0 and 2 enter any number; the script drops every other row before it computes
anything. Each headline number is reported for three groups: ``pooled``, ``unseen`` (acidosis and
HIE, never in training) and ``possibly_seen`` (healthy, membership unknown).

**Population.** Anchors in the KL support (``kl_support = 1``), which is the set the objective
charges and the evaluation reads.

**Intervals.** Every number carries a recording-level bootstrap 95% interval with 2000 resamples.
A mean of a per-anchor quantity (or a weighted share) uses :func:`s7common.paired_bootstrap`. A
statistic built from several sums (a ratio of sums, a correlation, a held-out $R^2$, a variance
share, a difference between two classes) resamples recordings once and recomputes the statistic
from the resampled per-recording sums (:func:`boot_sums`), so two arms compared on the same
resample stay paired at the recording level.

**Sections** (all written to ``out/phase2/REPORT_descriptive.json``):

1. *Gate.* Shares of anchors with $g \ge 0.2$ and $g \ge 0.8$; the share of total KL on gate-on
   anchors; the share of gate-on KL at lag 0; $g$ against seconds since onset, contraction phase
   and UA activity; $G$, KL and $g$ per anchor class (fully valid, $\rho$-only invalid, short
   bridge $\le 16$ s, long gap $> 16$ s).
2. *Attention.* The lag profile per head, overall, per phase and per activity bin; expected and
   argmax lag; how the attention argmax tracks the lag of the last UA peak in the window; where
   the attention mass and the KL sit relative to the contraction peak on the stored timeline.
3. *Latent and KL.* KL by head, lag and dimension; active dimensions; cross-fitted ridge probes of
   $\mu^q - \mu^p$ and $\mu^p$ (fit on code 0, scored on code 2, and the reverse).
4. *Read-outs.* Source against target adapter RMS; decoder log-variance floor occupancy; AR(1)
   innovation $z$-scores, PIT, interval coverage and kurtosis; implied marginal coverage; the
   latent share of the forecast-mean variance; a cross-fitted ridge from $\mu^p$ to the targets.

Stand-in normalisation statistics are in use, so absolute nats are approximate; the shares,
ratios and differences are the quantities to read.

Edit :data:`RUN_ARGS` and run this file from the IDE, or pass the options on the command line.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import norm  # noqa: E402

from teb_vae.lag_attn_rws.nets.losses import ar_innovation  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {
    "dump_dir": str(S.OUT / "dump"),
    "features": str(S.OUT / "phase2" / "anchor_features.parquet"),
    "out_dir": str(S.OUT / "phase2"),
    "n_boot": 2000,
    "ridge_alpha": 1.0,
    "max_ridge_anchors": 40000,
    "chunk": 4000,
}

CODES = (S.FIT_CODE, S.SELECT_CODE)
GROUPS = ("pooled", "unseen", "possibly_seen")
PHASES = ("rise", "peak", "fall", "between")
HEAD_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
GROUP_COLORS = {"pooled": "#2a78d6", "unseen": "#eb6834", "possibly_seen": "#1baf7a"}
PHASE_COLORS = {"rise": "#2a78d6", "peak": "#eb6834", "fall": "#1baf7a", "between": "#8a8a85"}
LW = 2.8
N_LAGS = 38
N_ST = 32
LEVELS = (0.5, 0.9, 0.95)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})


# --------------------------------------------------------------------------------------------
# Bootstrap helpers
# --------------------------------------------------------------------------------------------
def pb(rec: np.ndarray, x: np.ndarray, weight: Optional[np.ndarray] = None, n_boot: int = 2000) -> Dict[str, float]:
    """Recording-level bootstrap of a (weighted) mean, via :func:`s7common.paired_bootstrap`.

    Args:
        rec: Recording id per row.
        x: Value per row.
        weight: Optional row weight.
        n_boot: Resamples.

    Returns:
        ``{'mean', 'lo', 'hi', 'n_rec', 'n_rows'}``.
    """
    if len(x) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_rec": 0, "n_rows": 0}
    out = S.paired_bootstrap(rec, x, weight=weight, n_resamples=n_boot)
    out.pop("p_le0", None)
    return out


def boot_sums(rec: np.ndarray, cols: Dict[str, np.ndarray], fn: Callable[[Dict[str, np.ndarray]], Any],
              n_boot: int = 2000, seed: int = 0) -> Dict[str, Any]:
    r"""Recording-level bootstrap of a statistic of column sums.

    Every column is summed per recording, $s_{r,k}$. A resample draws recordings with replacement;
    with $W_{b,r}$ the number of times recording $r$ is drawn in resample $b$, the resampled sums
    are $W s$, and ``fn`` maps the sums to the statistic.

    Args:
        rec: Recording id per row.
        cols: Column name to a ``(N,)`` or ``(N, k)`` array.
        fn: Maps ``{name: sums}`` (sums with a leading resample axis, or none) to a scalar or array.
        n_boot: Resamples.
        seed: Generator seed.

    Returns:
        ``{'value', 'lo', 'hi', 'n_rec'}``; arrays become lists.
    """
    _, inv = np.unique(rec, return_inverse=True)
    n_rec = int(inv.max()) + 1 if inv.size else 0
    if n_rec == 0:
        return {"value": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_rec": 0}
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(n_rec))
    sums = {}
    for name, x in cols.items():
        x = np.asarray(x, dtype=np.float64)
        flat = x.reshape(len(x), -1)[order]
        sums[name] = (np.add.reduceat(flat, starts, axis=0), x.shape[1:])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_rec, size=(n_boot, n_rec))
    weights = np.zeros((n_boot, n_rec))
    np.add.at(weights, (np.repeat(np.arange(n_boot), n_rec), draws.ravel()), 1.0)
    full = {k: a.sum(0).reshape(s) for k, (a, s) in sums.items()}
    boot = {k: (weights @ a).reshape((n_boot, *s)) for k, (a, s) in sums.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.asarray(fn(full), dtype=np.float64)
        stats = np.asarray(fn(boot), dtype=np.float64)
    lo, hi = np.nanpercentile(stats, [2.5, 97.5], axis=0)
    conv = (lambda v: float(v)) if value.ndim == 0 else (lambda v: np.asarray(v).tolist())
    return {"value": conv(value), "lo": conv(lo), "hi": conv(hi), "n_rec": n_rec}


def ratio(num: str, den: str) -> Callable[[Dict[str, np.ndarray]], np.ndarray]:
    """Statistic ``sum(num) / sum(den)`` for :func:`boot_sums`."""
    return lambda s: s[num] / s[den]


def corr_fn(s: Dict[str, np.ndarray]) -> np.ndarray:
    """Pearson correlation from the sums ``n, x, y, xx, yy, xy``."""
    n = s["n"]
    cov = s["xy"] / n - s["x"] * s["y"] / n ** 2
    vx = s["xx"] / n - (s["x"] / n) ** 2
    vy = s["yy"] / n - (s["y"] / n) ** 2
    return cov / np.sqrt(vx * vy)


def corr_cols(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """Columns for :func:`corr_fn`."""
    return {"n": np.ones_like(x), "x": x, "y": y, "xx": x * x, "yy": y * y, "xy": x * y}


def r2_cols(y: np.ndarray, pred: np.ndarray) -> Dict[str, np.ndarray]:
    """Columns for :func:`r2_fn`: ``n, y, yy, sse`` (``y`` may be 2-D, one column per output)."""
    return {"n": np.ones(len(y)), "y": y, "yy": y * y, "sse": (y - pred) ** 2}


def r2_fn(s: Dict[str, np.ndarray]) -> np.ndarray:
    r"""Held-out $R^2 = 1 - \mathrm{SSE}/\mathrm{SST}$ per output from the sums of :func:`r2_cols`."""
    n = s["n"]
    n = n[..., None] if s["y"].ndim > n.ndim else n
    sst = s["yy"] - s["y"] ** 2 / n
    return 1.0 - s["sse"] / sst


def r2_pooled_fn(s: Dict[str, np.ndarray]) -> np.ndarray:
    r"""$R^2$ pooled over outputs, $1 - \sum \mathrm{SSE} / \sum \mathrm{SST}$."""
    n = s["n"][..., None]
    sst = (s["yy"] - s["y"] ** 2 / n).sum(-1)
    return 1.0 - s["sse"].sum(-1) / sst


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
def load(args: Dict[str, Any]) -> Dict[str, Any]:
    """Load the dump rows of codes 0 and 2 that lie in the KL support, joined with the features.

    Args:
        args: The run arguments.

    Returns:
        ``{'df': frame, 'arr': {name: array}, 'rows': dump row indices, 'schema': schema}``.
    """
    dump = Path(args["dump_dir"])
    schema = json.loads((dump / "schema.json").read_text())
    df = pd.read_parquet(dump / "anchors.parquet")
    df["row"] = np.arange(len(df))
    df = df[df["split_code"].isin(CODES)]
    assert set(df["split_code"].unique()) <= set(CODES)
    df = df[df["kl_support"] > 0.5]
    feats = pd.read_parquet(args["features"])
    df = df.merge(feats, on="row", how="left", validate="one_to_one")
    df = df.reset_index(drop=True)
    rows = df["row"].to_numpy()
    arr = {}
    for name in ("key_mask", "attn", "invalid_mass", "mu_prior", "mu_post", "kl_dim", "kl_head", "kl_lag",
                 "quality_inv_share"):
        arr[name] = np.asarray(np.load(dump / "arrays" / f"{name}.npy", mmap_mode="r")[rows], dtype=np.float32)
    df["group_unseen"] = df["membership"] == "unseen"
    return {"df": df, "arr": arr, "rows": rows, "schema": schema}


def group_masks(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Row masks of the three reporting groups.

    Args:
        df: The anchor frame.

    Returns:
        ``{'pooled', 'unseen', 'possibly_seen'}`` boolean masks.
    """
    unseen = (df["membership"] == "unseen").to_numpy()
    return {"pooled": np.ones(len(df), bool), "unseen": unseen, "possibly_seen": ~unseen}


def anchor_class(df: pd.DataFrame) -> np.ndarray:
    """Anchor class by key validity and raw UA validity over the 152-s key window.

    Returns:
        ``'i_valid'`` (all 38 keys usable), ``'ii_rho_only'`` (a key unusable, raw UA clean),
        ``'iii_short_bridge'`` (raw invalid total $\\le 16$ s), ``'iv_long_gap'`` ($> 16$ s).
    """
    out = np.where(df["n_keys_usable"] >= N_LAGS - 0.5, "i_valid",
                   np.where(df["raw_inv_s_win"] <= 0, "ii_rho_only",
                            np.where(df["raw_inv_s_win"] <= 16, "iii_short_bridge", "iv_long_gap")))
    return out


def activity_bins(x: np.ndarray, n: int = 5) -> (np.ndarray, List[float]):
    """Quantile bins of the detector-free UA activity measure.

    Args:
        x: ``ua_level60`` per anchor.
        n: Number of bins.

    Returns:
        ``(bin index, edges)``; bin $-1$ where ``x`` is not finite.
    """
    edges = np.nanquantile(x, np.linspace(0, 1, n + 1))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n - 1)
    return np.where(np.isfinite(x), idx, -1), edges.tolist()


# --------------------------------------------------------------------------------------------
# Section 1: gate
# --------------------------------------------------------------------------------------------
def section_gate(d: Dict[str, Any], nb: int) -> Dict[str, Any]:
    """Gate shares, KL concentration and the gate by contraction timing and anchor class."""
    df, arr = d["df"], d["arr"]
    rec = df["guid"].to_numpy()
    g = df["gate"].to_numpy()
    kl = df["kl"].to_numpy()
    G = df["G"].to_numpy()
    lag0 = arr["kl_lag"][:, 0].astype(np.float64)
    on = (g >= 0.2).astype(np.float64)
    res: Dict[str, Any] = {"by_group": {}, "onset_curve": {}, "phase": {}, "activity": {}, "classes": {}}
    for name, m in group_masks(df).items():
        r = rec[m]
        res["by_group"][name] = {
            "n_anchors": int(m.sum()), "n_rec": int(len(np.unique(r))),
            "share_g_ge_0.2": pb(r, on[m], n_boot=nb),
            "share_g_ge_0.8": pb(r, (g[m] >= 0.8).astype(float), n_boot=nb),
            "mean_g": pb(r, g[m], n_boot=nb),
            "kl_share_on_gate_ge_0.2": pb(r, on[m], weight=kl[m], n_boot=nb),
            "kl_share_on_gate_ge_0.8": pb(r, (g[m] >= 0.8).astype(float), weight=kl[m], n_boot=nb),
            "lag0_share_of_gate_on_kl": boot_sums(r, {"a": lag0[m] * on[m], "b": kl[m] * on[m]}, ratio("a", "b"), nb),
            "lag0_share_of_all_kl": boot_sums(r, {"a": lag0[m], "b": kl[m]}, ratio("a", "b"), nb),
            "mean_kl": pb(r, kl[m], n_boot=nb),
            "mean_G": pb(r, G[m], n_boot=nb),
            "mean_G_samp": pb(r, df["G_samp"].to_numpy()[m], n_boot=nb),
            "logit_terms": {
                "sd_logit_heads": float(df["gate_logit_heads"].to_numpy()[m].std()),
                "sd_logit_invalid": float(df["gate_logit_invalid"].to_numpy()[m].std()),
                "mean_logit_invalid": float(df["gate_logit_invalid"].to_numpy()[m].mean()),
            },
        }
    # g against seconds since onset.
    since = df["sec_since_onset"].to_numpy()
    edges = [0, 10, 20, 30, 45, 60, 90, 120, 180, 300, np.inf]
    labels = [f"{a}-{b}" for a, b in zip(edges[:-1], edges[1:])] + ["none"]
    bin_idx = np.where(np.isfinite(since), np.clip(np.searchsorted(edges, since, side="right") - 1, 0, len(edges) - 2),
                       len(edges) - 1)
    for name, m in group_masks(df).items():
        rows = []
        for b, lab in enumerate(labels):
            sel = m & (bin_idx == b)
            rows.append({"bin_s": lab, "n": int(sel.sum()), "g": pb(rec[sel], g[sel], n_boot=nb),
                         "kl": pb(rec[sel], kl[sel], n_boot=nb), "G": pb(rec[sel], G[sel], n_boot=nb)})
        res["onset_curve"][name] = rows
    # Contraction phase and activity bins.
    phase = df["phase"].to_numpy()
    act, act_edges = activity_bins(df["ua_level60"].to_numpy())
    res["activity_edges_level60"] = act_edges
    for name, m in group_masks(df).items():
        res["phase"][name] = {ph: {"share": float((phase[m] == ph).mean()),
                                   "g": pb(rec[m & (phase == ph)], g[m & (phase == ph)], n_boot=nb),
                                   "kl": pb(rec[m & (phase == ph)], kl[m & (phase == ph)], n_boot=nb),
                                   "G": pb(rec[m & (phase == ph)], G[m & (phase == ph)], n_boot=nb)}
                              for ph in PHASES}
        res["activity"][name] = [{"bin": b, "g": pb(rec[m & (act == b)], g[m & (act == b)], n_boot=nb),
                                  "kl": pb(rec[m & (act == b)], kl[m & (act == b)], n_boot=nb),
                                  "G": pb(rec[m & (act == b)], G[m & (act == b)], n_boot=nb),
                                  "share_on": pb(rec[m & (act == b)], on[m & (act == b)], n_boot=nb)}
                                 for b in range(5)]
    # Phase by activity cross-table of g (pooled).
    res["phase_x_activity_g_pooled"] = {ph: [float(g[(phase == ph) & (act == b)].mean()) if ((phase == ph) & (act == b)).any()
                                             else None for b in range(5)] for ph in PHASES}
    # Anchor classes.
    cls = anchor_class(df)
    names = ("i_valid", "ii_rho_only", "iii_short_bridge", "iv_long_gap")
    for name, m in group_masks(df).items():
        block = {}
        for c in names:
            sel = m & (cls == c)
            block[c] = {"share": float(sel.sum() / max(m.sum(), 1)), "n": int(sel.sum()),
                        "n_rec": int(len(np.unique(rec[sel]))),
                        "g": pb(rec[sel], g[sel], n_boot=nb), "kl": pb(rec[sel], kl[sel], n_boot=nb),
                        "G": pb(rec[sel], G[sel], n_boot=nb),
                        "share_g_ge_0.2": pb(rec[sel], on[sel], n_boot=nb)}
        # Differences against class i, resampling recordings jointly.
        for c in names[1:]:
            cols = {}
            for k, c2 in (("a", c), ("b", "i_valid")):
                sel = (m & (cls == c2)).astype(float)
                cols[f"n{k}"] = sel
                cols[f"G{k}"] = G * sel
                cols[f"g{k}"] = g * sel
                cols[f"kl{k}"] = kl * sel
            cols = {k: v[m] for k, v in cols.items()}
            block[c]["minus_i"] = {q: boot_sums(rec[m], cols, (lambda q: lambda s: s[f"{q}a"] / s["na"] - s[f"{q}b"] / s["nb"])(q), nb)
                                   for q in ("G", "g", "kl")}
        res["classes"][name] = block
    return res


# --------------------------------------------------------------------------------------------
# Section 2: attention
# --------------------------------------------------------------------------------------------
def section_attention(d: Dict[str, Any], nb: int) -> Dict[str, Any]:
    """Lag profiles, expected and argmax lag, peak tracking and the profile relative to the peak."""
    df, arr = d["df"], d["arr"]
    rec = df["guid"].to_numpy()
    alpha = arr["attn"]  # (N, 4, L)
    lags = np.arange(N_LAGS, dtype=np.float64)
    expected = (alpha * lags).sum(-1)  # steps
    argmax = alpha.argmax(-1).astype(np.float64)
    phase = df["phase"].to_numpy()
    act, _ = activity_bins(df["ua_level60"].to_numpy())
    res: Dict[str, Any] = {"profile": {}, "lag_seconds": {}, "peak_tracking": {}, "relative_to_peak": {}}
    for name, m in group_masks(df).items():
        res["profile"][name] = {"overall": alpha[m].mean(0).tolist(),
                                "phase": {ph: alpha[m & (phase == ph)].mean(0).tolist() for ph in PHASES
                                          if (m & (phase == ph)).any()},
                                "activity": {int(b): alpha[m & (act == b)].mean(0).tolist() for b in range(5)}}
        res["lag_seconds"][name] = {f"head{h}": {
            "expected_s": pb(rec[m], S.STEP_S * expected[m, h], n_boot=nb),
            "argmax_s": pb(rec[m], S.STEP_S * argmax[m, h], n_boot=nb),
            "share_argmax_lag0": pb(rec[m], (argmax[m, h] == 0).astype(float), n_boot=nb),
            "argmax_hist": np.bincount(argmax[m, h].astype(int), minlength=N_LAGS).tolist(),
            "expected_by_phase_s": {ph: float(S.STEP_S * expected[m & (phase == ph), h].mean()) for ph in PHASES
                                    if (m & (phase == ph)).any()},
        } for h in range(4)}
    # Peak tracking: anchors with a detected peak inside the 38-step window.
    peak_lag = df["peak_lag_last"].to_numpy()
    has = np.isfinite(peak_lag)
    for name, m in group_masks(df).items():
        sel = m & has
        r = rec[sel]
        block = {"n": int(sel.sum()), "n_rec": int(len(np.unique(r)))}
        p = peak_lag[sel]
        p_hist = np.bincount(p.astype(int), minlength=N_LAGS) / max(len(p), 1)
        for h in range(4):
            a = argmax[sel, h]
            e = expected[sel, h]
            a_hist = np.bincount(a.astype(int), minlength=N_LAGS) / max(len(a), 1)
            chance = float(sum(a_hist[i] * p_hist[j] for i in range(N_LAGS) for j in range(N_LAGS) if abs(i - j) <= 2))
            block[f"head{h}"] = {
                "corr_argmax_peaklag": boot_sums(r, corr_cols(a, p), corr_fn, nb),
                "corr_expected_peaklag": boot_sums(r, corr_cols(e, p), corr_fn, nb),
                "share_argmax_within_2": pb(r, (np.abs(a - p) <= 2).astype(float), n_boot=nb),
                "share_within_2_chance_independent": chance,
                "mean_argmax_minus_peaklag_s": pb(r, S.STEP_S * (a - p), n_boot=nb),
            }
        res["peak_tracking"][name] = block
    # Profiles on the axis "key time minus peak time" (steps): rel = peak_lag - l.
    kl_lag = arr["kl_lag"]
    kl = df["kl"].to_numpy()
    g = df["gate"].to_numpy()
    rel_axis = np.arange(-(N_LAGS - 1), N_LAGS)
    for name, m in group_masks(df).items():
        sel = np.flatnonzero(m & has)
        p = peak_lag[sel].astype(int)
        att_sum = np.zeros((4, rel_axis.size))
        kl_sum = np.zeros(rel_axis.size)
        cnt = np.zeros(rel_axis.size)
        for l in range(N_LAGS):
            k = p - l + (N_LAGS - 1)
            np.add.at(cnt, k, 1.0)
            np.add.at(kl_sum, k, kl_lag[sel, l])
            for h in range(4):
                np.add.at(att_sum[h], k, alpha[sel, h, l])
        mean_att = att_sum / np.maximum(cnt, 1)
        block = {"rel_steps": rel_axis.tolist(), "count": cnt.tolist(),
                 "attention_per_key": mean_att.tolist(), "kl_per_key": (kl_sum / np.maximum(cnt, 1)).tolist()}
        r = rec[sel]
        for h in range(4):
            rel_mean = (alpha[sel, h, :] * (p[:, None] - np.arange(N_LAGS)[None])).sum(-1)
            block[f"head{h}_mass_weighted_rel_s"] = pb(r, S.STEP_S * rel_mean, n_boot=nb)
            post = (alpha[sel, h, :] * ((p[:, None] - np.arange(N_LAGS)[None]) > 0)).sum(-1)
            block[f"head{h}_mass_after_peak"] = pb(r, post, n_boot=nb)
            atpk = (alpha[sel, h, :] * (np.abs(p[:, None] - np.arange(N_LAGS)[None]) <= 2)).sum(-1)
            block[f"head{h}_mass_within_8s_of_peak"] = pb(r, atpk, n_boot=nb)
        rel_kl = (kl_lag[sel] * (p[:, None] - np.arange(N_LAGS)[None])).sum(-1)
        block["kl_weighted_rel_s"] = boot_sums(r, {"a": S.STEP_S * rel_kl, "b": kl[sel]}, ratio("a", "b"), nb)
        res["relative_to_peak"][name] = block
    # Anchor-level KL and gate against the anchor's time relative to the nearest peak.
    past = df["sec_since_peak"].to_numpy()
    fut = S.STEP_S * df["steps_to_next_peak"].to_numpy()
    nearest = np.where(np.isfinite(past) & (~np.isfinite(fut) | (past <= fut)), past, -fut)
    edges = np.arange(-120, 244, 8.0)
    centers = (edges[:-1] + edges[1:]) / 2
    curves = {}
    for name, m in group_masks(df).items():
        b = np.searchsorted(edges, nearest, side="right") - 1
        ok = m & np.isfinite(nearest) & (b >= 0) & (b < centers.size)
        curves[name] = {"center_s": centers.tolist(),
                        "kl": [float(kl[ok & (b == i)].mean()) if (ok & (b == i)).any() else None for i in range(centers.size)],
                        "g": [float(g[ok & (b == i)].mean()) if (ok & (b == i)).any() else None for i in range(centers.size)],
                        "G": [float(df["G"].to_numpy()[ok & (b == i)].mean()) if (ok & (b == i)).any() else None
                              for i in range(centers.size)]}
        # Where the anchor-level KL mass sits relative to the peak, with an interval.
        kk = ok & (np.abs(nearest) <= 120)
        curves[name]["kl_weighted_anchor_rel_s"] = boot_sums(rec[kk], {"a": kl[kk] * nearest[kk], "b": kl[kk]},
                                                             ratio("a", "b"), nb)
        curves[name]["uniform_anchor_rel_s"] = float(nearest[kk].mean())
    res["anchor_relative_to_nearest_peak"] = curves
    return res


# --------------------------------------------------------------------------------------------
# Section 3: latent and KL
# --------------------------------------------------------------------------------------------
PROBE_GROUPS = {
    "UA": ["ua_lvl_l0", "ua_lvl_l30", "ua_lvl_l60", "ua_slope_l0", "ua_slope_l30", "ua_slope_l60",
           "log_since_onset", "no_onset"],
    "FHR": ["fhr_level", "fhr_change60", "fhr_var60"],
    "clock": ["sec_to_delivery", "anchor_step", "inv_key_share"],
}


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float):
    r"""Closed-form ridge on standardised inputs, $\hat\beta = (X^\top X + \lambda I)^{-1} X^\top y$.

    The penalty grows with the row count, $\lambda = 10^{-3}\,\alpha\, n$, so its strength relative
    to $X^\top X$ does not depend on how many anchors enter the fit.

    Args:
        x: Inputs $(n, p)$.
        y: Outputs $(n, k)$.
        alpha: Ridge strength $\alpha$.

    Returns:
        A function mapping new inputs to predictions.
    """
    mx, sx = x.mean(0), x.std(0) + 1e-9
    my = y.mean(0)
    xs = (x - mx) / sx
    a = xs.T @ xs + alpha * len(xs) * 1e-3 * np.eye(xs.shape[1])
    beta = np.linalg.solve(a, xs.T @ (y - my))
    return lambda z: ((z - mx) / sx) @ beta + my


def crossfit(x: np.ndarray, y: np.ndarray, code: np.ndarray, alpha: float) -> (np.ndarray, Dict[str, Any]):
    """Fit on code 0 and predict code 2, and the reverse.

    Args:
        x: Inputs.
        y: Outputs.
        code: Split code per row.
        alpha: Ridge strength.

    Returns:
        ``(held-out predictions, per-direction pooled R^2)``.
    """
    pred = np.zeros_like(y)
    info = {}
    for fit, test in ((0, 2), (2, 0)):
        f = ridge_fit(x[code == fit], y[code == fit], alpha)
        pred[code == test] = f(x[code == test])
        yt = y[code == test]
        sst = ((yt - yt.mean(0)) ** 2).sum(0)
        info[f"fit{fit}_score{test}_r2_per_output"] = (1 - ((yt - pred[code == test]) ** 2).sum(0) / sst).tolist()
    return pred, info


def section_latent(d: Dict[str, Any], nb: int, alpha: float) -> Dict[str, Any]:
    """KL decompositions, active dimensions and cross-fitted ridge probes."""
    df, arr = d["df"], d["arr"]
    rec = df["guid"].to_numpy()
    kl_dim = arr["kl_dim"].astype(np.float64)
    dmu = (arr["mu_post"] - arr["mu_prior"]).astype(np.float64)
    res: Dict[str, Any] = {"by_group": {}}
    for name, m in group_masks(df).items():
        r = rec[m]
        tot = df["kl"].to_numpy()[m]
        res["by_group"][name] = {
            "kl_share_by_head": boot_sums(r, {"a": arr["kl_head"][m].astype(np.float64), "b": tot},
                                          lambda s: s["a"] / s["b"][..., None], nb),
            "kl_share_by_lag": (arr["kl_lag"][m].sum(0) / tot.sum()).tolist(),
            "kl_share_by_dim": (kl_dim[m].sum(0) / tot.sum()).tolist(),
            "mean_kl_per_dim": kl_dim[m].mean(0).tolist(),
            "var_dmu_per_dim": dmu[m].var(0).tolist(),
            "active_kl_gt_0.01": np.flatnonzero(kl_dim[m].mean(0) > 0.01).tolist(),
            "active_var_dmu_gt_0.01": np.flatnonzero(dmu[m].var(0) > 0.01).tolist(),
        }
        bg = res["by_group"][name]
        bg["n_active_kl"] = len(bg["active_kl_gt_0.01"])
        bg["n_active_var"] = len(bg["active_var_dmu_gt_0.01"])
        # Share of KL on the top-k dimensions.
        order = np.argsort(kl_dim[m].sum(0))[::-1]
        bg["kl_share_top4_dims"] = boot_sums(r, {"a": kl_dim[m][:, order[:4]].sum(1), "b": tot}, ratio("a", "b"), nb)
        bg["kl_top_dims"] = order[:8].tolist()
    # Probes.
    feat = pd.DataFrame({
        **{c: df[c] for c in ("ua_lvl_l0", "ua_lvl_l30", "ua_lvl_l60", "ua_slope_l0", "ua_slope_l30", "ua_slope_l60",
                               "fhr_level", "fhr_change60", "fhr_var60", "sec_to_delivery", "anchor_step")},
    })
    since = df["sec_since_onset"].to_numpy()
    feat["log_since_onset"] = np.where(np.isfinite(since), np.log1p(np.nan_to_num(since)), np.log1p(1200.0))
    feat["no_onset"] = (~np.isfinite(since)).astype(float)
    feat["inv_key_share"] = 1.0 - df["n_keys_usable"].to_numpy() / N_LAGS
    pooled = res["by_group"]["pooled"]
    active = sorted(set(pooled["active_kl_gt_0.01"]) | set(pooled["active_var_dmu_gt_0.01"]))
    res["probe_dims"] = active
    ok = np.isfinite(feat.to_numpy()).all(1)
    code = df["split_code"].to_numpy()
    probes: Dict[str, Any] = {}
    for tname, tmat in (("dmu", dmu), ("mu_prior", arr["mu_prior"].astype(np.float64))):
        y = tmat[:, active]
        for gname, cols in list(PROBE_GROUPS.items()) + [("all", sum(PROBE_GROUPS.values(), []))]:
            x = feat[cols].to_numpy()
            pred, info = crossfit(x[ok], y[ok], code[ok], alpha)
            entry = {"directions": info}
            for name, m in group_masks(df).items():
                mm = m[ok]
                r = rec[ok][mm]
                entry[name] = {"r2_per_dim": boot_sums(r, r2_cols(y[ok][mm], pred[mm]), r2_fn, nb),
                               "r2_pooled_dims": boot_sums(r, r2_cols(y[ok][mm], pred[mm]), r2_pooled_fn, nb)}
            probes[f"{tname}/{gname}"] = entry
    res["probes"] = probes
    res["probe_n_rows"] = int(ok.sum())
    return res


# --------------------------------------------------------------------------------------------
# Section 4: read-outs
# --------------------------------------------------------------------------------------------
def section_readouts(d: Dict[str, Any], nb: int, args: Dict[str, Any]) -> Dict[str, Any]:
    """Adapter RMS, floor occupancy, innovation calibration, latent share and the ridge to targets."""
    df = d["df"]
    rec_all = df["guid"].to_numpy()
    res: Dict[str, Any] = {"adapter": {}}
    for name, m in group_masks(df).items():
        r = rec_all[m]
        res["adapter"][name] = {
            "src_out_rms": pb(r, df["src_adapter_rms"].to_numpy()[m], n_boot=nb),
            "tgt_out_rms": pb(r, df["tgt_adapter_rms"].to_numpy()[m], n_boot=nb),
            "src_over_tgt_out": boot_sums(r, {"a": df["src_adapter_rms"].to_numpy()[m], "b": df["tgt_adapter_rms"].to_numpy()[m]},
                                          ratio("a", "b"), nb),
            "src_prenorm_rms": pb(r, df["src_adapter_prenorm_rms"].to_numpy()[m], n_boot=nb),
            "tgt_prenorm_rms": pb(r, df["tgt_adapter_prenorm_rms"].to_numpy()[m], n_boot=nb),
            "src_over_tgt_prenorm": boot_sums(r, {"a": df["src_adapter_prenorm_rms"].to_numpy()[m],
                                                  "b": df["tgt_adapter_prenorm_rms"].to_numpy()[m]}, ratio("a", "b"), nb),
            "h_y_rms": pb(r, df["h_y_rms"].to_numpy()[m], n_boot=nb),
            "floor_st_full_anchor_mean": pb(r, df["floor_st_full"].to_numpy()[m], n_boot=nb),
            "floor_ph_full_anchor_mean": pb(r, df["floor_ph_full"].to_numpy()[m], n_boot=nb),
        }
    # Per-cell analyses on the subsample.
    schema = d["schema"]
    dump = Path(args["dump_dir"])
    cell = df[df["cell_row"] >= 0].reset_index(drop=True)
    crow = cell["cell_row"].to_numpy()
    order = np.argsort(crow)
    cell, crow = cell.iloc[order].reset_index(drop=True), crow[order]
    cells = {k: np.load(dump / "cells" / f"{k}.npy", mmap_mode="r") for k in
             ("mu_full", "lv_full", "mu_base", "lv_base", "mu_z0", "mu_zbar", "target", "step_mask")}
    ar = torch.tensor(schema["ar_coef"], dtype=torch.float64)
    cmask = np.asarray(schema["cell_mask"], dtype=np.float64)  # (H, C)
    H, C = cmask.shape
    _, inv = np.unique(cell["guid"].to_numpy(), return_inverse=True)
    n_rec = int(inv.max()) + 1
    unseen_rec = np.zeros(n_rec, bool)
    unseen_rec[inv[(cell["membership"] == "unseen").to_numpy()]] = True
    blocks = {"st": slice(0, N_ST), "ph": slice(N_ST, None)}
    qz = norm.ppf(0.5 + np.asarray(LEVELS) / 2)
    # Per-recording accumulators: branch (0 base, 1 full) x block x ...
    acc = {
        "floor": np.zeros((n_rec, 2, 2, H)), "n": np.zeros((n_rec, 2, 2, H)),
        "z2_floor": np.zeros((n_rec, 2, 2)), "n_floor": np.zeros((n_rec, 2, 2)),
        "z2_free": np.zeros((n_rec, 2, 2)), "n_free": np.zeros((n_rec, 2, 2)),
        "pit": np.zeros((n_rec, 2, 2, 10)), "cover": np.zeros((n_rec, 2, 2, len(LEVELS))),
        "mom": np.zeros((n_rec, 2, 2, 5)),
        "mcover": np.zeros((n_rec, 2, 2, H, len(LEVELS))),
        # Variance-share sums per cell: x (z = mu^p), L0 = x - mu_z0, Lb = x - mu_zbar, R0 = mu_z0,
        # Lq = mu_full - mu_z0.
        "v_s1": np.zeros((n_rec, 5, H, C)), "v_s2": np.zeros((n_rec, 5, H, C)), "v_n": np.zeros(n_rec),
    }
    chunk = int(args["chunk"])
    for lo in range(0, len(crow), chunk):
        idx = crow[lo:lo + chunk]
        ri = inv[lo:lo + chunk]
        tgt = np.asarray(cells["target"][idx], dtype=np.float64)
        sm = np.asarray(cells["step_mask"][idx], dtype=np.float64)
        M = sm[..., None] * cmask[None]
        for bi_, (mu_k, lv_k) in enumerate((("mu_base", "lv_base"), ("mu_full", "lv_full"))):
            mu = np.asarray(cells[mu_k][idx], dtype=np.float64)
            lv = np.asarray(cells[lv_k][idx], dtype=np.float64)
            e, lve = ar_innovation(torch.from_numpy(tgt - mu), ar, torch.from_numpy(sm), torch.from_numpy(lv))
            z = (e * torch.exp(-0.5 * lve)).numpy()
            floor = (lv <= -5.99).astype(np.float64)
            # Implied marginal variance of the AR(1) residual, V_tau = s2_tau + phi^2 V_{tau-1}.
            s2 = np.exp(lv)
            phi2 = (ar.numpy() ** 2)[None]
            V = np.empty_like(s2)
            V[:, 0] = s2[:, 0]
            for t in range(1, H):
                V[:, t] = s2[:, t] + phi2 * V[:, t - 1]
            zm = (tgt - mu) / np.sqrt(V)
            for bj, (bname, sl) in enumerate(blocks.items()):
                Mb = M[..., sl]
                zb, fb, zmb = z[..., sl], floor[..., sl], zm[..., sl]
                np.add.at(acc["floor"][:, bi_, bj], ri, (fb * Mb).sum(-1))
                np.add.at(acc["n"][:, bi_, bj], ri, Mb.sum(-1))
                np.add.at(acc["z2_floor"][:, bi_, bj], ri, (zb ** 2 * fb * Mb).sum((-1, -2)))
                np.add.at(acc["n_floor"][:, bi_, bj], ri, (fb * Mb).sum((-1, -2)))
                np.add.at(acc["z2_free"][:, bi_, bj], ri, (zb ** 2 * (1 - fb) * Mb).sum((-1, -2)))
                np.add.at(acc["n_free"][:, bi_, bj], ri, ((1 - fb) * Mb).sum((-1, -2)))
                u = norm.cdf(zb)
                pit_bin = np.clip((u * 10).astype(int), 0, 9)
                for k in range(10):
                    np.add.at(acc["pit"][:, bi_, bj, k], ri, ((pit_bin == k) * Mb).sum((-1, -2)))
                for k, q in enumerate(qz):
                    np.add.at(acc["cover"][:, bi_, bj, k], ri, ((np.abs(zb) <= q) * Mb).sum((-1, -2)))
                    np.add.at(acc["mcover"][:, bi_, bj, :, k], ri, ((np.abs(zmb) <= q) * Mb).sum(-1))
                for p in range(5):
                    np.add.at(acc["mom"][:, bi_, bj, p], ri, ((zb ** p) * Mb).sum((-1, -2)))
        x = np.asarray(cells["mu_base"][idx], dtype=np.float64)
        r0 = np.asarray(cells["mu_z0"][idx], dtype=np.float64)
        rb = np.asarray(cells["mu_zbar"][idx], dtype=np.float64)
        xf = np.asarray(cells["mu_full"][idx], dtype=np.float64)
        for k, v in enumerate((x, x - r0, x - rb, r0, xf - r0)):
            np.add.at(acc["v_s1"][:, k], ri, v)
            np.add.at(acc["v_s2"][:, k], ri, v * v)
        np.add.at(acc["v_n"], ri, 1.0)
        print(f"  cells {lo + len(idx)}/{len(crow)}", flush=True)

    rng = np.random.default_rng(0)
    out: Dict[str, Any] = {"n_cell_anchors": int(len(crow))}
    for gname in GROUPS:
        keep = np.ones(n_rec, bool) if gname == "pooled" else (unseen_rec if gname == "unseen" else ~unseen_rec)
        recs = np.flatnonzero(keep)
        draws = recs[rng.integers(0, len(recs), size=(nb, len(recs)))]
        W = np.zeros((nb, n_rec))
        np.add.at(W, (np.repeat(np.arange(nb), len(recs)), draws.ravel()), 1.0)
        w_full = keep.astype(float)

        def stat(fn: Callable[[Dict[str, np.ndarray]], np.ndarray], keys: Sequence[str]) -> Dict[str, Any]:
            full = {k: np.tensordot(w_full, acc[k], axes=(0, 0)) for k in keys}
            boot = {k: np.tensordot(W, acc[k], axes=(1, 0)) for k in keys}
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.asarray(fn(full))
                b = np.asarray(fn(boot))
            lo_, hi_ = np.nanpercentile(b, [2.5, 97.5], axis=0)
            return {"value": v.tolist(), "lo": lo_.tolist(), "hi": hi_.tolist(), "n_rec": int(keep.sum())}

        def kurt(s):
            n, m1, m2, m3, m4 = (s["mom"][..., p] for p in range(5))
            mu = m1 / n
            c2 = m2 / n - mu ** 2
            c4 = m4 / n - 4 * mu * m3 / n + 6 * mu ** 2 * m2 / n - 3 * mu ** 4
            return c4 / c2 ** 2 - 3.0

        def var_share(k_num: int, k_den: int, block: slice):
            def fn(s):
                n = s["v_n"][..., None, None]
                var = s["v_s2"] / n[..., None, :, :] - (s["v_s1"] / n[..., None, :, :]) ** 2
                vm = var * cmask
                num = vm[..., k_num, :, block].sum((-1, -2))
                den = vm[..., k_den, :, block].sum((-1, -2))
                return num / den
            return fn

        block_out = {}
        for bj, bname in enumerate(blocks):
            sl = blocks[bname]
            b = {}
            for bi_, br in enumerate(("base", "full")):
                b[br] = {
                    "floor_share": stat(lambda s: s["floor"][..., bi_, bj, :].sum(-1) / s["n"][..., bi_, bj, :].sum(-1), ["floor", "n"]),
                    "floor_share_by_tau": stat(lambda s: s["floor"][..., bi_, bj, :] / s["n"][..., bi_, bj, :], ["floor", "n"]),
                    "mean_z2_floor_cells": stat(lambda s: s["z2_floor"][..., bi_, bj] / s["n_floor"][..., bi_, bj], ["z2_floor", "n_floor"]),
                    "mean_z2_nonfloor_cells": stat(lambda s: s["z2_free"][..., bi_, bj] / s["n_free"][..., bi_, bj], ["z2_free", "n_free"]),
                    "pit_hist": stat(lambda s: s["pit"][..., bi_, bj, :] / s["pit"][..., bi_, bj, :].sum(-1, keepdims=True), ["pit"]),
                    "coverage_innov": {f"{int(100 * L)}": stat((lambda k: lambda s: s["cover"][..., bi_, bj, k] / s["n"][..., bi_, bj, :].sum(-1))(k),
                                                               ["cover", "n"]) for k, L in enumerate(LEVELS)},
                    "coverage_marginal": {f"{int(100 * L)}": stat((lambda k: lambda s: s["mcover"][..., bi_, bj, :, k].sum(-1) / s["n"][..., bi_, bj, :].sum(-1))(k),
                                                                  ["mcover", "n"]) for k, L in enumerate(LEVELS)},
                    "coverage_marginal_by_tau_95": stat(lambda s: s["mcover"][..., bi_, bj, :, 2] / s["n"][..., bi_, bj, :], ["mcover", "n"]),
                    "coverage_marginal_by_tau_50": stat(lambda s: s["mcover"][..., bi_, bj, :, 0] / s["n"][..., bi_, bj, :], ["mcover", "n"]),
                    "excess_kurtosis_innov": stat(lambda s: kurt({"mom": s["mom"][..., bi_, bj, :]}), ["mom"]),
                }
            b["latent_share"] = {
                "var_latent_z0_over_total_base": stat(var_share(1, 0, sl), ["v_s1", "v_s2", "v_n"]),
                "var_latent_zbar_over_total_base": stat(var_share(2, 0, sl), ["v_s1", "v_s2", "v_n"]),
                "var_skip_z0_over_total_base": stat(var_share(3, 0, sl), ["v_s1", "v_s2", "v_n"]),
                "var_latent_q_z0_over_total_base": stat(var_share(4, 0, sl), ["v_s1", "v_s2", "v_n"]),
            }
            block_out[bname] = b
        out[gname] = block_out
    res["cells"] = out
    res["ridge_mu_prior_to_target"] = ridge_targets(cell, crow, cells, cmask, args, nb)
    return res


def ridge_targets(cell: pd.DataFrame, crow: np.ndarray, cells: Dict[str, np.ndarray], cmask: np.ndarray,
                  args: Dict[str, Any], nb: int) -> Dict[str, Any]:
    r"""Cross-fitted ridge from $\mu^p$ to the scored target, per block, with model references.

    Rows: subsample anchors with a fully observed horizon, capped per code. Outputs: the cells of
    the block inside the cell mask. The model's own mean forecasts (base and full) and the $z = 0$
    decode are scored on the same rows for reference.

    Returns:
        Pooled held-out $R^2$ per block and group, with intervals.
    """
    dump = Path(args["dump_dir"])
    mu_prior = np.load(dump / "arrays" / "mu_prior.npy", mmap_mode="r")
    sm = np.asarray(cells["step_mask"][crow]).all(1)
    rng = np.random.default_rng(1)
    pick = []
    for c in CODES:
        idx = np.flatnonzero(sm & (cell["split_code"].to_numpy() == c))
        if len(idx) > args["max_ridge_anchors"]:
            idx = np.sort(rng.choice(idx, args["max_ridge_anchors"], replace=False))
        pick.append(idx)
    pick = np.sort(np.concatenate(pick))
    x = np.asarray(mu_prior[cell["row"].to_numpy()[pick]], dtype=np.float64)
    code = cell["split_code"].to_numpy()[pick]
    rec = cell["guid"].to_numpy()[pick]
    unseen = (cell["membership"] == "unseen").to_numpy()[pick]
    out: Dict[str, Any] = {"n_rows": int(len(pick))}
    for bname, sl in (("st", slice(0, N_ST)), ("ph", slice(N_ST, None))):
        keep = cmask[:, sl].reshape(-1) > 0
        def flat(name):
            return np.asarray(cells[name][crow[pick]], dtype=np.float64)[..., sl].reshape(len(pick), -1)[:, keep]
        y = flat("target")
        pred, _ = crossfit(x, y, code, float(args["ridge_alpha"]))
        refs = {"ridge_mu_prior": pred, "model_base_mean": flat("mu_base"), "model_full_mean": flat("mu_full"),
                "decode_z0": flat("mu_z0")}
        bo = {}
        for gname, m in (("pooled", np.ones(len(pick), bool)), ("unseen", unseen), ("possibly_seen", ~unseen)):
            bo[gname] = {k: boot_sums(rec[m], r2_cols(y[m], v[m]), r2_pooled_fn, nb) for k, v in refs.items()}
            # Ridge on the target minus the z = 0 decode: what the latent could add to the skip path.
            resid = y - refs["decode_z0"]
            pr, _ = crossfit(x, resid, code, float(args["ridge_alpha"]))
            bo[gname]["ridge_mu_prior_on_residual_after_z0"] = boot_sums(rec[m], r2_cols(resid[m], pr[m]), r2_pooled_fn, nb)
        out[bname] = bo
    return out


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------
def fig_gate(rep: Dict[str, Any], path: Path) -> None:
    """Gate against onset age, UA activity and anchor class."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for gname in GROUPS:
        rows = rep["gate"]["onset_curve"][gname]
        x = np.arange(len(rows))
        y = [r["g"]["mean"] for r in rows]
        lo = [r["g"]["lo"] for r in rows]
        hi = [r["g"]["hi"] for r in rows]
        ax[0].plot(x, y, lw=LW, color=GROUP_COLORS[gname], label=gname, marker="o", ms=7)
        ax[0].fill_between(x, lo, hi, color=GROUP_COLORS[gname], alpha=0.15)
        ax[0].set_xticks(x, [r["bin_s"] for r in rows], rotation=45)
        act = rep["gate"]["activity"][gname]
        ax[1].plot(range(5), [a["g"]["mean"] for a in act], lw=LW, color=GROUP_COLORS[gname], marker="o", ms=7,
                   label=f"{gname}: g")
        ax[1].plot(range(5), [a["share_on"]["mean"] for a in act], lw=LW, ls="-", color=GROUP_COLORS[gname],
                   alpha=0.5, marker="s", ms=7, label=f"{gname}: share g>=0.2")
    ax[0].set_xlabel("seconds since last detected contraction onset")
    ax[0].set_ylabel("event gate g (mean, 95% CI)")
    ax[0].set_title("Gate against contraction age")
    ax[0].legend()
    ax[1].set_xlabel("UA activity quintile (60-s level above baseline)")
    ax[1].set_title("Gate against detector-free UA activity")
    ax[1].legend(fontsize=9)
    names = ("i_valid", "ii_rho_only", "iii_short_bridge", "iv_long_gap")
    cl = rep["gate"]["classes"]["pooled"]
    xs = np.arange(len(names))
    for k, (q, col) in enumerate((("g", "#2a78d6"), ("G", "#eb6834"))):
        vals = [cl[n][q]["mean"] for n in names]
        err = np.array([[cl[n][q]["mean"] - cl[n][q]["lo"], cl[n][q]["hi"] - cl[n][q]["mean"]] for n in names]).T
        ax[2].bar(xs + (k - 0.5) * 0.38, vals, width=0.36, color=col, yerr=err, capsize=4,
                  label="mean g" if q == "g" else "mean G (nats/anchor)")
    ax[2].set_xticks(xs, [f"{n}\n{100 * cl[n]['share']:.1f}%" for n in names])
    ax[2].set_title("Gate and source gain by anchor class (pooled)")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_attention(rep: Dict[str, Any], path: Path) -> None:
    """Lag profile per head by contraction phase and by activity quintile."""
    att = rep["attention"]["profile"]["pooled"]
    fig, ax = plt.subplots(2, 4, figsize=(20, 8.5), sharex=True)
    lags_s = S.STEP_S * np.arange(N_LAGS)
    for h in range(4):
        for ph, prof in att["phase"].items():
            ax[0, h].plot(lags_s, np.asarray(prof)[h], lw=LW, color=PHASE_COLORS[ph], label=ph)
        ax[0, h].plot(lags_s, np.asarray(att["overall"])[h], lw=LW, color="black", ls="--", label="all")
        ax[0, h].set_title(f"head {h}: lag profile by contraction phase")
        cmap = plt.get_cmap("Blues")
        for b, prof in att["activity"].items():
            ax[1, h].plot(lags_s, np.asarray(prof)[h], lw=LW, color=cmap(0.35 + 0.15 * int(b)), label=f"activity Q{int(b) + 1}")
        ax[1, h].set_title(f"head {h}: by UA activity quintile")
        ax[1, h].set_xlabel("lag (s, feature grid)")
    ax[0, 0].set_ylabel("mean attention weight")
    ax[1, 0].set_ylabel("mean attention weight")
    ax[0, 0].legend()
    ax[1, 0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_peak(rep: Dict[str, Any], path: Path) -> None:
    """Attention, per-key KL and anchor KL relative to the contraction peak."""
    rel = rep["attention"]["relative_to_peak"]["pooled"]
    x = S.STEP_S * np.asarray(rel["rel_steps"])
    cnt = np.asarray(rel["count"])
    ok = cnt >= 200
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
    for h in range(4):
        ax[0].plot(x[ok], np.asarray(rel["attention_per_key"])[h][ok], lw=LW, color=HEAD_COLORS[h], label=f"head {h}")
    ax[0].axvline(0, color="grey", lw=1.5)
    ax[0].set_xlabel("key time minus peak time (s); >0 = key after the peak")
    ax[0].set_ylabel("mean attention per key")
    ax[0].set_title("Attention relative to the UA peak")
    ax[0].legend()
    ax[1].plot(x[ok], np.asarray(rel["kl_per_key"])[ok], lw=LW, color="#e87ba4")
    ax[1].axvline(0, color="grey", lw=1.5)
    ax[1].set_xlabel("key time minus peak time (s)")
    ax[1].set_ylabel("KL attributed to the key (nats)")
    ax[1].set_title("Per-lag KL relative to the UA peak")
    cur = rep["attention"]["anchor_relative_to_nearest_peak"]
    for gname in GROUPS:
        c = cur[gname]
        ax[2].plot(c["center_s"], [np.nan if v is None else v for v in c["kl"]], lw=LW, color=GROUP_COLORS[gname],
                   label=f"{gname}: KL")
    ax2 = ax[2].twinx()
    ax2.plot(cur["pooled"]["center_s"], [np.nan if v is None else v for v in cur["pooled"]["g"]], lw=LW,
             color="black", ls="--", label="pooled: g")
    ax2.set_ylabel("event gate g")
    ax2.grid(False)
    ax[2].axvline(0, color="grey", lw=1.5)
    ax[2].set_xlabel("anchor time minus nearest peak (s)")
    ax[2].set_ylabel("KL per anchor (nats)")
    ax[2].set_title("Anchor KL and gate around the UA peak")
    h1, l1 = ax[2].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[2].legend(h1 + h2, l1 + l2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_latent(rep: Dict[str, Any], path: Path) -> None:
    """KL by dimension and lag, and the probe R^2 table."""
    lat = rep["latent"]["by_group"]["pooled"]
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.4))
    share = np.asarray(lat["kl_share_by_dim"])
    order = np.argsort(share)[::-1]
    colors = [HEAD_COLORS[d // 16] for d in order]
    ax[0].bar(np.arange(64), share[order], color=colors)
    ax[0].set_xticks(np.arange(0, 64, 4), [str(d) for d in order[::4]])
    ax[0].set_xlabel("latent dimension (sorted; colour = head)")
    ax[0].set_ylabel("share of total KL")
    ax[0].set_title("KL by dimension (pooled)")
    for gname in GROUPS:
        ax[1].plot(S.STEP_S * np.arange(N_LAGS), rep["latent"]["by_group"][gname]["kl_share_by_lag"], lw=LW,
                   color=GROUP_COLORS[gname], label=gname)
    ax[1].set_xlabel("lag (s)")
    ax[1].set_ylabel("share of KL")
    ax[1].set_title("KL by lag")
    ax[1].legend()
    dims = rep["latent"]["probe_dims"]
    names = ["UA", "FHR", "clock", "all"]
    mat = np.array([[rep["latent"]["probes"][f"dmu/{n}"]["pooled"]["r2_per_dim"]["value"][k] for n in names]
                    for k in range(len(dims))]) if dims else np.zeros((1, 4))
    im = ax[2].imshow(mat.T, aspect="auto", cmap="Blues", vmin=0, vmax=max(0.05, float(np.nanmax(mat))))
    ax[2].set_yticks(range(4), names)
    ax[2].set_xticks(range(len(dims)), [str(d) for d in dims], rotation=90, fontsize=8)
    ax[2].set_xlabel("active dimension")
    ax[2].set_title("held-out R^2 of ridge probes on mu_q - mu_p")
    ax[2].grid(False)
    fig.colorbar(im, ax=ax[2])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_readouts(rep: Dict[str, Any], path: Path) -> None:
    """Floor occupancy by horizon, PIT histograms and marginal coverage by horizon."""
    cells = rep["readouts"]["cells"]["pooled"]
    fig, ax = plt.subplots(1, 4, figsize=(22, 5.2))
    tau = np.arange(1, 31)
    for bname, col in (("st", "#2a78d6"), ("ph", "#eb6834")):
        for br, ls in (("full", "-"), ("base", "-")):
            v = cells[bname][br]["floor_share_by_tau"]["value"]
            ax[0].plot(tau, v, lw=LW, color=col, alpha=1.0 if br == "full" else 0.45, label=f"{bname} {br}")
        pit = np.asarray(cells[bname]["full"]["pit_hist"]["value"])
        ax[1 if bname == "st" else 2].bar(np.arange(10) / 10 + 0.05, pit, width=0.09, color=col)
        ax[1 if bname == "st" else 2].axhline(0.1, color="black", lw=1.5)
        ax[1 if bname == "st" else 2].set_title(f"PIT of AR(1) innovations, {bname}, full branch")
        ax[3].plot(tau, cells[bname]["full"]["coverage_marginal_by_tau_95"]["value"], lw=LW, color=col, label=f"{bname} 95%")
        ax[3].plot(tau, cells[bname]["full"]["coverage_marginal_by_tau_50"]["value"], lw=LW, color=col, alpha=0.45,
                   label=f"{bname} 50%")
    ax[0].set_xlabel("horizon step")
    ax[0].set_ylabel("share of scored cells with logvar <= -5.99")
    ax[0].set_title("Decoder log-variance floor occupancy")
    ax[0].legend()
    ax[1].set_xlabel("PIT")
    ax[2].set_xlabel("PIT")
    ax[3].axhline(0.95, color="black", lw=1.5)
    ax[3].axhline(0.5, color="black", lw=1.5)
    ax[3].set_xlabel("horizon step")
    ax[3].set_ylabel("coverage")
    ax[3].set_title("Implied marginal coverage (AR(1)), full branch")
    ax[3].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
#: Reading notes stored with the report, so the numbers are not read without their limits.
NOTES = [
    "The contraction detector drops every event that overlaps an invalid UA sample, so anchors near raw gaps "
    "fall into 'between' more often; the anchor-class table does not depend on the detector.",
    "The per-key profiles relative to the peak mix the fixed lag template of each head with the peak timing: a "
    "key at +148 s after the peak exists only at lag 0 with the peak 37 steps back. Read the anchor-level curve "
    "(KL and gate against anchor time minus nearest peak) for timing.",
    "Peak timing uses the stored UP/FHR timeline as it is; lags are on the 4-s feature grid.",
    "The prior adapter-RMS claim (9.6 against 0.98) came from an older checkpoint with an unrecorded definition; "
    "here RMS is over the 128 model channels of the adapter output at the anchor step.",
    "Stand-in normalisation statistics: absolute nats are approximate.",
]


def prior_claims(rep: Dict[str, Any]) -> Dict[str, Any]:
    """Verdicts on the prior claims, pooled group, with the new number and interval.

    A claim is confirmed when its value lies inside the 95% interval or within 2 percentage points of the
    estimate for a share.

    Args:
        rep: The report with the gate and read-out sections.

    Returns:
        ``{claim: {'prior', 'new', 'verdict'}}``.
    """
    gate = rep["gate"]["by_group"]["pooled"]
    adapter = rep["readouts"]["adapter"]["pooled"]

    def verdict(prior: float, est: Dict[str, float], tol: float = 0.02) -> str:
        inside = est["lo"] <= prior <= est["hi"] or abs(est["mean"] - prior) <= tol
        return "confirmed" if inside else "refuted"

    out = {
        "gate_on_share_0.43": {"prior": 0.43, "new": gate["share_g_ge_0.2"],
                               "verdict": verdict(0.43, gate["share_g_ge_0.2"])},
        "gate_on_kl_share_0.977": {"prior": 0.977, "new": gate["kl_share_on_gate_ge_0.2"],
                                   "verdict": verdict(0.977, gate["kl_share_on_gate_ge_0.2"], 0.005)},
    }
    lag0 = gate["lag0_share_of_gate_on_kl"]
    lag0m = {"mean": lag0["value"], "lo": lag0["lo"], "hi": lag0["hi"]}
    out["gate_on_kl_at_lag0_0.48"] = {"prior": 0.48, "new": lag0m, "verdict": verdict(0.48, lag0m)}
    ratio_ = adapter["src_over_tgt_out"]
    out["adapter_rms_src_9.6_vs_tgt_0.98"] = {
        "prior": {"src": 9.6, "tgt": 0.98, "ratio": 9.6 / 0.98},
        "new": {"src": adapter["src_out_rms"], "tgt": adapter["tgt_out_rms"], "ratio": ratio_},
        "verdict": "refuted (source not larger than target at this checkpoint; definition of the prior number unknown)"
        if ratio_["hi"] < 2.0 else "confirmed",
    }
    return out


def build_parser() -> argparse.ArgumentParser:
    """Command-line options; every default comes from :data:`RUN_ARGS`.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump-dir", default=RUN_ARGS["dump_dir"])
    parser.add_argument("--features", default=RUN_ARGS["features"])
    parser.add_argument("--out-dir", default=RUN_ARGS["out_dir"])
    parser.add_argument("--n-boot", type=int, default=RUN_ARGS["n_boot"])
    parser.add_argument("--ridge-alpha", type=float, default=RUN_ARGS["ridge_alpha"])
    parser.add_argument("--max-ridge-anchors", type=int, default=RUN_ARGS["max_ridge_anchors"])
    parser.add_argument("--chunk", type=int, default=RUN_ARGS["chunk"])
    parser.add_argument("--only", nargs="*", default=None,
                        help="Sections to recompute (gate attention latent readouts); others are read back.")
    return parser


def main(args: Dict[str, Any]) -> None:
    """Compute every section, write the report and the figures.

    Args:
        args: The run arguments; see :data:`RUN_ARGS`.
    """
    out_dir = Path(args["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "REPORT_descriptive.json"
    rep: Dict[str, Any] = json.loads(report_path.read_text()) if (args.get("only") and report_path.exists()) else {}
    t0 = time.time()
    d = load(args)
    nb = int(args["n_boot"])
    df = d["df"]
    rep["population"] = {
        "codes": list(CODES), "kl_support_only": True, "n_anchors": int(len(df)),
        "n_recordings": {g: int(df.loc[m, "guid"].nunique()) for g, m in group_masks(df).items()},
        "n_recordings_by_cohort_code": {f"{c}/{k}": int(v) for (c, k), v in df.groupby(["cohort", "split_code"])["guid"].nunique().items()},
        "phase_shares": df["phase"].value_counts(normalize=True).round(4).to_dict(),
        "stand_in_statistics": True,
    }
    todo = args.get("only") or ["gate", "attention", "latent", "readouts"]
    if "gate" in todo:
        rep["gate"] = section_gate(d, nb)
        print(f"gate done {time.time() - t0:.0f} s", flush=True)
    if "attention" in todo:
        rep["attention"] = section_attention(d, nb)
        print(f"attention done {time.time() - t0:.0f} s", flush=True)
    if "latent" in todo:
        rep["latent"] = section_latent(d, nb, float(args["ridge_alpha"]))
        print(f"latent done {time.time() - t0:.0f} s", flush=True)
    if "readouts" in todo:
        rep["readouts"] = section_readouts(d, nb, args)
        print(f"readouts done {time.time() - t0:.0f} s", flush=True)
    if all(k in rep for k in ("gate", "readouts")):
        rep["prior_claims"] = prior_claims(rep)
    rep["notes"] = NOTES
    report_path.write_text(json.dumps(rep, indent=1, default=float))
    for name, fn in (("fig_gate.png", fig_gate), ("fig_attention_profiles.png", fig_attention),
                     ("fig_attention_peak.png", fig_peak), ("fig_latent.png", fig_latent),
                     ("fig_readouts.png", fig_readouts)):
        try:
            fn(rep, out_dir / name)
        except KeyError as err:  # a section that was not computed in this run
            print(f"skip {name}: missing {err}")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(vars(build_parser().parse_args()))
