r"""T4: the model-level cost of the bridge and of the strict flags on synthetic gaps.

For every synthetic segment of T2 the network runs on the base segment's own batch (built by the
loader, so the target, the weights, the step mask and the dense anchor set are those of the
evaluation pass). Only the source stream $u$ (input 3) and its validity $v$ (input 6) change:

* ``true``: the true UA coefficients, every flag valid;
* ``strict``: the bridged coefficients, flags as the loader computes them from the rebuilt bits
  and $\rho$ (step $k$, channel $c$ bad when a gap sample lies in raw samples $16k-15 \dots 16k$ or
  $\rho_c[k] > 0.1$), the current behaviour;
* ``tolerant``: the bridged coefficients, every flag valid;
* ``flags``: the true coefficients with the strict flags (the cost of the flags alone);
* ``band_fast``, ``band_mid``, ``band_slow``: true coefficients except that one band group's
  channels take the bridged values, every flag valid.

The coefficients are normalised by the loader's own ``_normalize_data`` and trimmed to the served
300 steps; for the stored base segment this reproduces the loader's stream exactly.

**Mean decoding.** Both branches are decoded at the latent means through
``model.decoder(mu.gather(1, idx), persistence=out['persistence'])`` as the forward does:
$\hat y^{\mathrm{base}} = D(\mu^p_t)$ and $\hat y^{\mathrm{full}} = D(\mu^q_t)$. Each branch is scored
by ``p0_dump.score_branch`` (``raw_sample_score`` under the model's AR(1) residual, cell mask and
forecast step mask). The source gain is $G = \mathrm{NLL}_{\mathrm{base}} - \mathrm{NLL}_{\mathrm{full}}$.
$\mathrm{NLL}_{\mathrm{base}}$ does not read the source; the script records its largest change
across variants as a check.

**Affected anchors.** Anchor $t$ (model step; stored raw sample $240 + 16t$) is affected when its
38-step key window $[t-37, t]$ holds at least one step at which the strict rule marks at least
one kept channel bad in the ``strict`` run. Its timing class relative to the gap that most
recently started before it: ``inside`` (anchor sample within the gap), ``0-60`` s, ``60-152`` s
or ``later`` after the gap end.

**Costs.** On affected anchors, $c_{\mathrm{tol}} = (\bar G_{\mathrm{true}} - \bar G_{\mathrm{tol}})/\bar G_{\mathrm{true}}$
and $c_{\mathrm{strict}}$, $c_{\mathrm{flags}}$, $c_{\mathrm{band}}$ likewise (ratios of means). Their
intervals resample recordings with the scheme of :func:`s7common.paired_bootstrap` (2000
resamples, seed 0) and recompute the ratio of sums; the mean differences in nats use
:func:`s7common.paired_bootstrap` itself.

**Tolerance.** A class is tolerable when $c_{\mathrm{tol}} < 10\%$ and its upper bound $< 20\%$
($X = 10\%$, fixed before any result). The tolerable length of a class is the largest length $L$
such that every grid length $\le L$ is tolerable.

**Causality caveat.** A bridged value inside a gap $[s, e)$ reads raw samples up to $e + 16$.
An anchor inside the gap under ``tolerant`` therefore reads UA from after the anchor; its gain is
not available to a causal deployment.

Edit :data:`RUN_ARGS` and run this file from the IDE.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import p1_common as C  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7.p0_dump import gather_steps, score_branch  # noqa: E402

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {"device": "cuda:0", "rows_per_forward": 32, "max_base": None, "n_boot": 2000}

TRIM = 15
N_LAGS = 38
X_TOL = 0.10
X_HI = 0.20
VARIANTS = ("strict", "tolerant", "flags", "band_fast", "band_mid", "band_slow")
#: Stream index of each kept channel (stream = 36 up_st then 10 up_ph).
STREAM_KEPT = np.array(C.KEPT_ST + [36 + i for i in C.KEPT_PH])


def strict_valid_full(gap: np.ndarray, rho_st: np.ndarray, rho_ph: np.ndarray) -> np.ndarray:
    """The loader's ``ua_step_mask`` (bits ``invalid``, $\\varepsilon = 0.1$) on all 46 channels.

    Args:
        gap: ``(N,)`` invalid raw samples (the injected gaps; the base is clean).
        rho_st: ``(36, T)`` ``uint8``.
        rho_ph: ``(10, T)`` ``uint8``.

    Returns:
        ``(300, 46)`` bool, ``True`` = usable, trimmed.
    """
    steps = rho_st.shape[-1]
    padded = np.concatenate([np.zeros(S.RAW_PER_STEP - 1, dtype=bool), gap])
    step_bad = padded[: steps * S.RAW_PER_STEP].reshape(steps, S.RAW_PER_STEP).any(axis=1)
    rho = np.concatenate([rho_st, rho_ph], axis=0).astype(np.float64) / 255.0
    bad = step_bad[:, None] | (rho.T > C.RHO_EPS)
    return ~bad[TRIM:steps - TRIM]


def normalise_stream(ds: Any, up_st: np.ndarray, up_ph: np.ndarray) -> np.ndarray:
    """Loader normalisation of both blocks, trimmed, as the ``(300, 46)`` stream.

    Args:
        ds: The dataset.
        up_st: ``(36, T)``.
        up_ph: ``(10, T)``.

    Returns:
        ``(300, 46)`` float32.
    """
    st = ds._normalize_data("up_st", torch.from_numpy(np.ascontiguousarray(up_st[:, TRIM:-TRIM], dtype=np.float32)))
    ph = ds._normalize_data("up_ph", torch.from_numpy(np.ascontiguousarray(up_ph[:, TRIM:-TRIM], dtype=np.float32)))
    return torch.cat([st.T, ph.T], dim=-1).numpy()


def ratio_boot(rec: np.ndarray, num: np.ndarray, den: np.ndarray, n_boot: int, seed: int = 0) -> Dict[str, float]:
    """Recording-level bootstrap of $\\sum num / \\sum den$ (the resampling of ``paired_bootstrap``).

    Args:
        rec: Recording per row.
        num: Numerator per row.
        den: Denominator per row.
        n_boot: Resamples.
        seed: Seed.

    Returns:
        ``{'ratio', 'lo', 'hi', 'n_rec', 'n_rows'}``.
    """
    keep = np.isfinite(num) & np.isfinite(den)
    rec, num, den = rec[keep], num[keep], den[keep]
    if rec.size == 0:
        return {"ratio": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_rec": 0, "n_rows": 0}
    _, inv = np.unique(rec, return_inverse=True)
    n_rec = int(inv.max()) + 1
    a = np.bincount(inv, weights=num, minlength=n_rec)
    b = np.bincount(inv, weights=den, minlength=n_rec)
    draws = np.random.default_rng(seed).integers(0, n_rec, size=(n_boot, n_rec))
    ratios = a[draws].sum(1) / b[draws].sum(1)
    return {"ratio": float(a.sum() / b.sum()), "lo": float(np.percentile(ratios, 2.5)),
            "hi": float(np.percentile(ratios, 97.5)), "n_rec": n_rec, "n_rows": int(rec.size),
            "den_mean": float(b.sum() / rec.size)}


@torch.no_grad()
def run(device: str, rows_per_forward: int, max_base: Any) -> pd.DataFrame:
    """Run every variant of every synthetic segment and collect per-anchor rows.

    Args:
        device: Torch device.
        rows_per_forward: Batch rows per forward.
        max_base: Limit on base segments (``None`` = all).

    Returns:
        The per-anchor table (one row per synthetic segment and anchor, columns per variant).
    """
    from train.data_module import GraphDataModule

    dev, cfg, task = S.load_model(device=device)
    model = task.orig_model.eval()
    lk = model.forecast_likelihood_kwargs()
    loader = GraphDataModule(cfg).test_dataloader()
    ds, collate = loader.dataset, loader.collate_fn
    event_mask = bool(task.hparams.get("source_event_mask", False))
    norm = lambda p: os.path.normcase(os.path.normpath(str(p)))  # noqa: E731
    file_of = {norm(p): i for i, p in enumerate(ds.paths)}
    where = {(fi, int(si)): k for k, (fi, si) in enumerate(ds.index_map)}

    synth = pd.read_parquet(C.OUT1 / "t2_synth.parquet")
    gaps = pd.read_parquet(C.OUT1 / "t2_injections.parquet")
    base = pd.read_parquet(C.OUT1 / "t2_base.parquet")
    true = dict(np.load(C.OUT1 / "t2_base.npz"))
    sets = {name: dict(np.load(C.OUT1 / f"t2_{name}.npz")) for name in ("grid", "mix")}
    pos_in_set = {name: {int(s): i for i, s in enumerate(arr["synth_id"])} for name, arr in sets.items()}
    meta = C.channel_meta()
    group_stream = {g: STREAM_KEPT[meta["group"] == g] for g in ("fast", "mid", "slow")}

    rows: List[Dict[str, Any]] = []
    checks = {"nll_base_maxabs_across_variants": 0.0, "true_vs_loader_stream_maxabs": 0.0,
              "true_vs_loader_G_maxabs": 0.0, "stored_valid_all_true_share": [], "source_event_mask": event_mask}
    t0 = time.time()
    bases = base if max_base is None else base.head(int(max_base))
    for bi, brec in enumerate(bases.itertuples()):
        b = int(brec.base_id)
        k = where[(file_of[norm(S.SHARDS[brec.cohort])], int(brec.row))]
        item = ds[k]
        u_true = normalise_stream(ds, true["up_st"][b], true["up_ph"][b])
        jobs: List[Tuple[str, int, str, np.ndarray, np.ndarray]] = [("base", -1, "true", u_true, np.ones((300, 46), bool))]
        for _, srow in synth[synth.base_id == b].iterrows():
            sid = int(srow.synth_id)
            arr = sets[srow["set"]]
            i = pos_in_set[srow["set"]][sid]
            gap_mask = np.unpackbits(arr["invalid"][i])[: C.N_RAW].astype(bool)
            u_br = normalise_stream(ds, arr["up_st"][i], arr["up_ph"][i])
            v_strict = strict_valid_full(gap_mask, arr["up_st_rho"][i], arr["up_ph_rho"][i])
            ones = np.ones_like(v_strict)
            jobs += [(srow["set"], sid, "strict", u_br, v_strict), (srow["set"], sid, "tolerant", u_br, ones),
                     (srow["set"], sid, "flags", u_true, v_strict)]
            for g, idx in group_stream.items():
                u_g = u_true.copy()
                u_g[:, idx] = u_br[:, idx]
                jobs.append((srow["set"], sid, f"band_{g}", u_g, ones))
        results: Dict[Tuple[int, str], Dict[str, np.ndarray]] = {}
        strict_bad_window: Dict[int, np.ndarray] = {}
        for start in range(0, len(jobs), rows_per_forward):
            chunk = jobs[start:start + rows_per_forward]
            batch = task.transfer_batch_to_device(collate([item] * len(chunk)), dev, 0)
            torch.manual_seed(0)
            inputs, out0, target, step_mask, anchors, anchor_valid = common.forward_parts(task, model, batch)
            if bi == 0 and start == 0:
                checks["true_vs_loader_stream_maxabs"] = float((inputs[2][0].cpu() - torch.from_numpy(u_true)).abs().max())
            checks["stored_valid_all_true_share"].append(float(inputs[5][0].float().mean()))
            new = list(inputs)
            new[2] = torch.from_numpy(np.stack([j[3] for j in chunk])).to(dev)
            valid = torch.from_numpy(np.stack([j[4] for j in chunk])).to(dev)
            if event_mask:
                inside = task._source_event_steps(batch, batch.weight)
                valid = valid & inside.to(dev)[..., None]
            new[5] = valid
            torch.manual_seed(0)
            out = model(*new)
            gidx = anchors[:, :, None].expand(-1, -1, model.d_z)
            mu_b, lv_b = model.decoder(out["mu_prior"].gather(1, gidx), persistence=out["persistence"])
            mu_f, lv_f = model.decoder(out["mu_post"].gather(1, gidx), persistence=out["persistence"])
            sb = score_branch(mu_b, lv_b, target, step_mask, lk)
            sf = score_branch(mu_f, lv_f, target, step_mask, lk)
            if bi == 0 and start == 0:
                mu_b0, lv_b0 = model.decoder(out0["mu_prior"].gather(1, gidx), persistence=out0["persistence"])
                mu_f0, lv_f0 = model.decoder(out0["mu_post"].gather(1, gidx), persistence=out0["persistence"])
                g0 = score_branch(mu_b0, lv_b0, target, step_mask, lk)["nll"] - score_branch(mu_f0, lv_f0, target, step_mask, lk)["nll"]
                checks["true_vs_loader_G_maxabs"] = float((g0[0] - (sb["nll"][0] - sf["nll"][0])).abs().max())
            scored = ((sb["n_st"] + sb["n_ph"]) > 0) & anchor_valid
            res = {"anchor": anchors.cpu().numpy(), "keep": scored.cpu().numpy(),
                   "nll_base": sb["nll"].cpu().numpy(), "nll_full": sf["nll"].cpu().numpy(),
                   "kl": gather_steps(out["kld_per_t"], anchors).cpu().numpy(),
                   "gate": gather_steps(out["event_gate"], anchors)[..., 0].cpu().numpy(),
                   "inv_mass": gather_steps(out["source_invalid_mass"], anchors).mean(-1).cpu().numpy()}
            step_bad_any = ~valid[:, :, STREAM_KEPT].all(-1)
            cs = torch.cat([torch.zeros_like(step_bad_any[:, :1], dtype=torch.float32),
                            torch.cumsum(step_bad_any.float(), 1)], 1)
            lo = (anchors - (N_LAGS - 1)).clamp_min(0)
            win_bad = (gather_steps(cs, anchors + 1) - gather_steps(cs, lo)).cpu().numpy()
            for r, job in enumerate(chunk):
                results[(job[1], job[2])] = {key: val[r] for key, val in res.items()}
                if job[2] == "strict":
                    strict_bad_window[job[1]] = win_bad[r]
        ref = results[(-1, "true")]
        if not ref["keep"].any():
            checks.setdefault("bases_without_scored_anchor", []).append(b)
            continue
        for (sid, var), res in results.items():
            if sid >= 0:
                checks["nll_base_maxabs_across_variants"] = max(checks["nll_base_maxabs_across_variants"],
                                                                float(np.abs(res["nll_base"] - ref["nll_base"])[ref["keep"]].max()))
        for sid in sorted({s for s, _ in results if s >= 0}):
            srow = synth.loc[synth.synth_id == sid].iloc[0]
            gp = gaps[gaps.synth_id == sid].sort_values("start")
            keep = ref["keep"]
            anc = ref["anchor"]
            raw_t = 240 + 16 * anc
            for a in np.flatnonzero(keep):
                n_a = int(raw_t[a])
                before = gp[gp.start <= n_a]
                if before.empty:
                    last = None
                    timing = "before"
                else:
                    last = before.iloc[-1]
                    if n_a < int(last.stop):
                        timing = "inside"
                    else:
                        d = (n_a - int(last.stop)) / S.FS
                        timing = "0-60" if d < 60 else ("60-152" if d < 152 else "later")
                row = {"set": srow["set"], "synth_id": int(sid), "base_id": b, "rec": f"{srow.cohort}/{srow.guid}",
                       "cohort": srow.cohort, "anchor": int(anc[a]), "len_s": float(last.len_s) if last is not None else np.nan,
                       "position": last.position if last is not None else "none",
                       "code": last.dominant if last is not None else "none", "timing": timing,
                       "affected": bool(strict_bad_window[sid][a] > 0), "n_bad_steps_window": float(strict_bad_window[sid][a]),
                       "nll_base": float(ref["nll_base"][a]), "nll_full_true": float(ref["nll_full"][a]),
                       "G_true": float(ref["nll_base"][a] - ref["nll_full"][a]), "kl_true": float(ref["kl"][a]),
                       "gate_true": float(ref["gate"][a])}
                for var in VARIANTS:
                    res = results[(sid, var)]
                    row[f"nll_full_{var}"] = float(res["nll_full"][a])
                    row[f"G_{var}"] = float(res["nll_base"][a] - res["nll_full"][a])
                    row[f"kl_{var}"] = float(res["kl"][a])
                    row[f"gate_{var}"] = float(res["gate"][a])
                    row[f"inv_mass_{var}"] = float(res["inv_mass"][a])
                rows.append(row)
        if bi % 10 == 0:
            print(f"base {bi + 1}/{len(bases)}: {len(jobs)} rows, {time.time() - t0:.0f} s", flush=True)
    checks["stored_valid_all_true_share"] = float(np.mean(checks["stored_valid_all_true_share"]))
    (C.OUT1 / "t4_checks.json").write_text(json.dumps(checks, indent=1))
    frame = pd.DataFrame(rows)
    frame.to_parquet(C.OUT1 / "t4_anchors.parquet")
    print("checks", checks)
    return frame


def costs(part: pd.DataFrame, n_boot: int) -> Dict[str, Any]:
    """Cost ratios and mean changes for one class of affected anchors.

    Args:
        part: Affected anchors of one class.
        n_boot: Resamples.

    Returns:
        ``{'n', 'n_rec', 'G_true', 'c_tol', 'c_strict', 'c_flags', 'c_band_*', 'dG_*', 'dkl_*', 'dgate_*', 'dnll_*'}``.
    """
    rec = part.rec.values
    out: Dict[str, Any] = {"n": int(len(part)), "n_rec": int(part.rec.nunique()),
                           "G_true": S.paired_bootstrap(rec, part.G_true.values, n_resamples=n_boot)}
    for var in VARIANTS:
        name = {"tolerant": "tol"}.get(var, var)
        out[f"c_{name}"] = ratio_boot(rec, (part.G_true - part[f"G_{var}"]).values, part.G_true.values, n_boot)
        out[f"dG_{name}"] = S.paired_bootstrap(rec, (part.G_true - part[f"G_{var}"]).values, n_resamples=n_boot)
        if var in ("strict", "tolerant", "flags"):
            out[f"dkl_{name}"] = S.paired_bootstrap(rec, (part[f"kl_{var}"] - part.kl_true).values, n_resamples=n_boot)
            out[f"dgate_{name}"] = S.paired_bootstrap(rec, (part[f"gate_{var}"] - part.gate_true).values, n_resamples=n_boot)
            out[f"dnll_full_{name}"] = S.paired_bootstrap(rec, (part[f"nll_full_{var}"] - part.nll_full_true).values,
                                                          n_resamples=n_boot)
    return out


def tolerable(c: Dict[str, float]) -> bool:
    """Pre-registered rule: ratio < 10% and upper bound < 20%."""
    return bool(np.isfinite(c["ratio"]) and c["ratio"] < X_TOL and c["hi"] < X_HI)


def max_tolerable(verdicts: Dict[float, bool]) -> Any:
    """Largest length $L$ such that every length $\\le L$ is tolerable (``0`` when the shortest fails)."""
    best = 0
    for length in sorted(verdicts):
        if not verdicts[length]:
            break
        best = length
    return best


def summarise(frame: pd.DataFrame, n_boot: int) -> Dict[str, Any]:
    """Tables of T4 and the tolerance table.

    Args:
        frame: Per-anchor table.
        n_boot: Resamples.

    Returns:
        The summary dictionary.
    """
    out: Dict[str, Any] = {}
    grid = frame[(frame["set"] == "grid") & frame.affected]
    mix = frame[(frame["set"] == "mix") & frame.affected]
    out["n_affected"] = {"grid": int(len(grid)), "mix": int(len(mix))}
    out["affected_by_timing_grid"] = frame[frame["set"] == "grid"].groupby("timing").affected.mean().to_dict()
    out["grid_len_pos"] = {f"{int(L)}s/{p}": costs(part, n_boot) for (L, p), part in grid.groupby(["len_s", "position"])}
    out["grid_len"] = {f"{int(L)}s": costs(part, n_boot) for L, part in grid.groupby("len_s")}
    out["grid_len_timing"] = {f"{int(L)}s/{t}": costs(part, n_boot) for (L, t), part in grid.groupby(["len_s", "timing"])}
    out["grid_len_pos_excl_inside"] = {f"{int(L)}s/{p}": costs(part, n_boot)
                                       for (L, p), part in grid[grid.timing != "inside"].groupby(["len_s", "position"])}
    mixc = mix.assign(lc=mix.len_s.map(C.len_class))
    out["mix_code"] = {code: costs(part, n_boot) for code, part in mixc.groupby("code") if part.rec.nunique() >= 3}
    out["mix_code_len"] = {f"{code}/{lc}": costs(part, n_boot) for (code, lc), part in mixc.groupby(["code", "lc"])
                           if part.rec.nunique() >= 3}
    out["mix_len"] = {lc: costs(part, n_boot) for lc, part in mixc.groupby("lc") if part.rec.nunique() >= 3}
    out["mix_timing"] = {t: costs(part, n_boot) for t, part in mixc.groupby("timing")}
    # tolerance table
    tol: Dict[str, Any] = {"rule": "c_tol < 10% and CI upper < 20%; max length = largest L with all L' <= L tolerable"}
    pos_table = {}
    for pos in ("rise", "peak", "fall", "between"):
        verdict = {float(L): tolerable(out["grid_len_pos"][f"{int(L)}s/{pos}"]["c_tol"])
                   for L in sorted(grid.len_s.unique()) if f"{int(L)}s/{pos}" in out["grid_len_pos"]}
        pos_table[pos] = {"per_length": verdict, "max_tolerable_s": max_tolerable(verdict)}
    tol["by_position"] = pos_table
    band_table = {}
    for g in ("fast", "mid", "slow"):
        verdict = {float(L.rstrip("s")): tolerable(v[f"c_band_{g}"]) for L, v in out["grid_len"].items()}
        band_table[g] = {"per_length": verdict, "max_tolerable_s": max_tolerable(verdict),
                         "c_by_length": {L: v[f"c_band_{g}"] for L, v in out["grid_len"].items()}}
    tol["by_band_group"] = band_table
    all_verdict = {float(L.rstrip("s")): tolerable(v["c_tol"]) for L, v in out["grid_len"].items()}
    tol["all_positions_pooled"] = {"per_length": all_verdict, "max_tolerable_s": max_tolerable(all_verdict)}
    code_table = {}
    edges = {f"<={e}": float(e) for e in C.LEN_EDGES}
    edges[f">{C.LEN_EDGES[-1]}"] = 1e9
    for code in mixc.code.unique():
        verdict = {edges[lc]: tolerable(out["mix_code_len"][f"{code}/{lc}"]["c_tol"])
                   for lc in edges if f"{code}/{lc}" in out["mix_code_len"]}
        code_table[code] = {"per_length_class": {str(k): v for k, v in verdict.items()},
                            "max_tolerable_s": max_tolerable(verdict) if verdict else None,
                            "pooled": out["mix_code"].get(code, {}).get("c_tol")}
    tol["by_code_mix"] = code_table
    out["tolerance"] = tol
    # kill criterion: c_tol >= 10% at 4-8 s in every position
    kill = all(out["grid_len_pos"][f"{L}s/{p}"]["c_tol"]["ratio"] >= X_TOL
               for L in (4, 8) for p in ("rise", "peak", "fall", "between") if f"{L}s/{p}" in out["grid_len_pos"])
    out["kill_criterion_met"] = bool(kill)
    return out


def figures(summary: Dict[str, Any]) -> None:
    """Cost curves by length and position (thick solid lines).

    Args:
        summary: Output of :func:`summarise`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"rise": "#1f77b4", "peak": "#d62728", "fall": "#2ca02c", "between": "#9467bd"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    for ax, key, title in zip(axes, ("c_tol", "c_strict", "c_flags"),
                              ("bridged UA, flags valid (c_tol)", "bridged UA, strict flags (c_strict)", "true UA, strict flags (c_flags)")):
        for pos, col in colors.items():
            pts = [(int(k.split("s/")[0]), v[key]) for k, v in summary["grid_len_pos"].items() if k.endswith(f"/{pos}")]
            pts.sort()
            xs = [p[0] for p in pts]
            ax.plot(xs, [100 * p[1]["ratio"] for p in pts], "-o", lw=3, color=col, label=pos)
            ax.fill_between(xs, [100 * p[1]["lo"] for p in pts], [100 * p[1]["hi"] for p in pts], color=col, alpha=0.12)
        ax.axhline(10, color="k", lw=2, alpha=0.5)
        ax.axhline(0, color="k", lw=1, alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("gap length (s)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("lost share of source gain G on affected anchors (%)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(C.OUT1 / "t4_costs.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for g, col in (("fast", "#ff7f0e"), ("mid", "#17becf"), ("slow", "#8c564b")):
        pts = sorted((int(k.rstrip("s")), v[f"c_band_{g}"]) for k, v in summary["grid_len"].items())
        xs = [p[0] for p in pts]
        ax.plot(xs, [100 * p[1]["ratio"] for p in pts], "-o", lw=3, color=col, label=f"{g} only bridged")
        ax.fill_between(xs, [100 * p[1]["lo"] for p in pts], [100 * p[1]["hi"] for p in pts], color=col, alpha=0.12)
    pts = sorted((int(k.rstrip("s")), v["c_tol"]) for k, v in summary["grid_len"].items())
    ax.plot([p[0] for p in pts], [100 * p[1]["ratio"] for p in pts], "-o", lw=3, color="k", label="all bridged")
    ax.axhline(10, color="k", lw=2, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("gap length (s)")
    ax.set_ylabel("c (%) on affected anchors")
    ax.set_title("cost by band group (flags valid)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.OUT1 / "t4_band_costs.png", dpi=130)
    plt.close(fig)


def main(device: str, rows_per_forward: int, max_base: Any, n_boot: int) -> Dict[str, Any]:
    """Run, summarise and draw.

    Args:
        device: Torch device.
        rows_per_forward: Rows per forward.
        max_base: Base limit.
        n_boot: Resamples.

    Returns:
        The summary.
    """
    frame = run(device, rows_per_forward, max_base)
    summary = summarise(frame, n_boot)
    summary["checks"] = json.loads((C.OUT1 / "t4_checks.json").read_text())
    (C.OUT1 / "t4_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    figures(summary)
    for key, v in summary["grid_len_pos"].items():
        print(key, "n", v["n"], "G", round(v["G_true"]["mean"], 3), "c_tol", round(v["c_tol"]["ratio"], 3),
              [round(v["c_tol"]["lo"], 3), round(v["c_tol"]["hi"], 3)], "c_strict", round(v["c_strict"]["ratio"], 3),
              "c_flags", round(v["c_flags"]["ratio"], 3))
    print(json.dumps(summary["tolerance"], indent=1, default=float)[:3000])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=RUN_ARGS["device"])
    parser.add_argument("--rows-per-forward", type=int, default=RUN_ARGS["rows_per_forward"])
    parser.add_argument("--max-base", type=int, default=RUN_ARGS["max_base"])
    parser.add_argument("--n-boot", type=int, default=RUN_ARGS["n_boot"])
    a = parser.parse_args()
    main(a.device, a.rows_per_forward, a.max_base, a.n_boot)
