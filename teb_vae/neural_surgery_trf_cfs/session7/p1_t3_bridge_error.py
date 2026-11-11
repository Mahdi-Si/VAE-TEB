r"""T3: the error of the builder's bridge on raw UA (level 1) and on the coefficients (level 2).

**Level 1, raw UA.** For each injected gap $[s, e)$ with true trace $x$ and bridged trace $\hat x$:

* $\mathrm{RMSE} = \sqrt{\tfrac{1}{e-s}\sum_{n=s}^{e-1} (\hat x_n - x_n)^2}$, in TOCO units and as a share
  of the reference contraction's amplitude $A = x[p] - x[o]$ (peak minus onset value). The
  reference contraction is the grid's target contraction, or for an observed-mix gap the
  contraction that holds the gap's middle (else the nearest one);
* the peak error inside the reference contraction $[o, e_c]$:
  $\Delta_{\mathrm{amp}} = (\max \hat x - \max x)/A$ and the shift of the arg-max in seconds;
* the detector's view: the contraction detector is re-run on $\hat x$ and the event whose peak
  is nearest the true peak (within 60 s) gives the onset, peak and end timing errors; no match
  counts as a miss.

**Level 2, coefficients.** The kept channels (st0--31, ph0--9) of true and bridged sources are
put through the loader's own normalisation (log for ``up_st`` channels $\ge 1$, asinh for
``up_ph``, then the per-channel standardisation of the stand-in statistics). This is the space
the model reads. The error of channel $c$ at step $k$ is

$$\epsilon_c[k] = \frac{|\hat u_c[k] - u_c[k]|}{\sigma_c},$$

with $\sigma_c$ the standard deviation of $u_c$ over the served steps of the clean base segments.
Time is counted from the gap end: step $k$ is $\tau = 16k - e$ raw samples after it. A channel is
*off* at step $k$ when $\epsilon_c[k] > 0.1$; its off duration is the last off step after the gap
end, in seconds (``censored`` when the off stretch reaches the segment end). It is compared with
its $\rho$ duration, the last step after the gap end with rebuilt $\rho_c > 0.1$.

**Statistics.** Every mean comes with a recording-level bootstrap interval
(:func:`s7common.paired_bootstrap`, 2000 resamples; a one-arm mean is the paired mean against 0).
Only code-0/2 recordings enter (T2 draws its base segments from them).

Edit :data:`RUN_ARGS` and run this file from the IDE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import p1_common as C  # noqa: E402

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {"n_boot": 2000}

TRIM = 15
SERVED = slice(TRIM, C.N_STEPS - TRIM)
OFF_EPS = 0.10
MATCH_S = 60.0


def loader_normaliser():
    """The test dataset of the run's configuration, for its ``_normalize_data``.

    Returns:
        The dataset object (no model is loaded).
    """
    from teb_vae.neural_surgery_trf_cfs import common
    from train.data_module import GraphDataModule

    cfg = common.merge_eval_overrides(common.load_config(str(S.CONFIG)), str(S.OVERRIDES))
    common.force_single_process_loader(cfg)
    return GraphDataModule(cfg).test_dataloader().dataset


def normalise(ds: Any, up_st: np.ndarray, up_ph: np.ndarray) -> np.ndarray:
    """Normalise full-length blocks as the loader does and keep the 42 model channels.

    The loader normalises the trimmed window; the transform is elementwise per channel, so the
    full 330-step grid is normalised the same way and trimmed by the caller.

    Args:
        ds: The dataset (for its statistics).
        up_st: ``(B, 36, T)``.
        up_ph: ``(B, 10, T)``.

    Returns:
        ``(B, 42, T)`` float32 in the model's input space.
    """
    import torch

    out = np.zeros((up_st.shape[0], 42, up_st.shape[-1]), dtype=np.float32)
    for i in range(up_st.shape[0]):
        st = ds._normalize_data("up_st", torch.from_numpy(np.ascontiguousarray(up_st[i], dtype=np.float32))).numpy()
        ph = ds._normalize_data("up_ph", torch.from_numpy(np.ascontiguousarray(up_ph[i], dtype=np.float32))).numpy()
        out[i] = np.concatenate([st[C.KEPT_ST], ph[C.KEPT_PH]], axis=0)
    return out


def boot(df: pd.DataFrame, col: str, n_boot: int) -> Dict[str, float]:
    """Recording-level bootstrap of the mean of one column.

    Args:
        df: Rows with a ``rec`` column.
        col: Column.
        n_boot: Resamples.

    Returns:
        :func:`s7common.paired_bootstrap` result.
    """
    sub = df[np.isfinite(df[col].astype(float))]
    return S.paired_bootstrap(sub.rec.values, sub[col].astype(float).values, n_resamples=n_boot)


def ref_contraction(events: Dict[str, np.ndarray], centre: int) -> int:
    """Index of the contraction holding ``centre``, else the nearest one by peak.

    Args:
        events: Detector output on the true trace.
        centre: Raw sample.

    Returns:
        Event index, or ``-1`` when there is none.
    """
    if len(events["peak_raw"]) == 0:
        return -1
    inside = np.flatnonzero((events["onset_raw"] <= centre) & (events["end_raw"] >= centre))
    if inside.size:
        return int(inside[0])
    return int(np.argmin(np.abs(events["peak_raw"] - centre)))


def level1(synth: pd.DataFrame, gaps: pd.DataFrame, base: pd.DataFrame, true: Dict[str, np.ndarray],
           bridged: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    """Raw-UA errors per gap.

    Args:
        synth: Synthetic segment table.
        gaps: Injection table.
        base: Base segment table.
        true: ``t2_base.npz`` arrays.
        bridged: ``{'grid': npz, 'mix': npz}`` arrays.

    Returns:
        One row per gap.
    """
    events_true = {int(b): C.detect(true["up"][i], true["weight"][i]) for i, b in enumerate(true["base_id"])}
    rows: List[Dict[str, Any]] = []
    for name, arr in bridged.items():
        pos_of = {int(s): i for i, s in enumerate(arr["synth_id"])}
        for sid, i in pos_of.items():
            srow = synth.loc[synth.synth_id == sid].iloc[0]
            b = int(srow.base_id)
            x = true["up"][b].astype(np.float64)
            xb = arr["up"][i].astype(np.float64)
            ev_t = events_true[b]
            ev_b = C.detect(arr["up"][i], true["weight"][b])
            brec = base.loc[base.base_id == b].iloc[0]
            for _, gp in gaps[gaps.synth_id == sid].iterrows():
                s0, s1 = int(gp.start), int(gp.stop)
                if name == "grid":
                    o, p, e = int(brec.onset), int(brec.peak), int(brec.end)
                else:
                    k = ref_contraction(ev_t, (s0 + s1) // 2)
                    if k < 0:
                        continue
                    o, p, e = int(ev_t["onset_raw"][k]), int(ev_t["peak_raw"][k]), int(ev_t["end_raw"][k])
                amp = float(x[p] - x[o])
                rmse = float(np.sqrt(np.mean((xb[s0:s1] - x[s0:s1]) ** 2)))
                win = slice(o, e + 1)
                amp_err = float((xb[win].max() - x[win].max()) / amp) if amp > 0 else np.nan
                tpk = float((np.argmax(xb[win]) - np.argmax(x[win])) / S.FS)
                touches = s0 <= e and s1 > o
                d_on = d_pk = d_end = np.nan
                miss = 1.0
                if len(ev_b["peak_raw"]):
                    j = int(np.argmin(np.abs(ev_b["peak_raw"] - p)))
                    if abs(int(ev_b["peak_raw"][j]) - p) <= MATCH_S * S.FS:
                        miss = 0.0
                        d_on = (int(ev_b["onset_raw"][j]) - o) / S.FS
                        d_pk = (int(ev_b["peak_raw"][j]) - p) / S.FS
                        d_end = (int(ev_b["end_raw"][j]) - e) / S.FS
                rows.append({"set": name, "synth_id": sid, "gap": int(gp.gap), "rec": f"{srow.cohort}/{srow.guid}",
                             "cohort": srow.cohort, "len_s": float(gp.len_s), "position": gp.position,
                             "code": gp.dominant, "amp": amp, "rmse": rmse, "rmse_rel": rmse / amp if amp > 0 else np.nan,
                             "touches_ref": bool(touches), "peak_amp_err_rel": amp_err if touches else 0.0,
                             "peak_amp_abs_err_rel": abs(amp_err) if touches else 0.0,
                             "peak_shift_s": tpk if touches else 0.0, "abs_peak_shift_s": abs(tpk) if touches else 0.0,
                             "det_miss": miss, "det_onset_err_s": d_on, "det_peak_err_s": d_pk, "det_end_err_s": d_end,
                             "abs_det_onset_err_s": abs(d_on), "abs_det_peak_err_s": abs(d_pk),
                             "abs_det_end_err_s": abs(d_end)})
    return pd.DataFrame(rows)


def level2(synth: pd.DataFrame, gaps: pd.DataFrame, true_n: np.ndarray, sigma: np.ndarray,
           grid: Dict[str, np.ndarray], grid_n: np.ndarray, meta: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Coefficient errors per grid segment and channel.

    Args:
        synth: Synthetic table.
        gaps: Injection table.
        true_n: ``(n_base, 42, T)`` normalised true source.
        sigma: ``(42,)`` channel standard deviations.
        grid: ``t2_grid.npz``.
        grid_n: ``(n_grid, 42, T)`` normalised bridged source.
        meta: Channel metadata.

    Returns:
        ``{'rows': DataFrame (synth x channel), 'curves': (n_grid, 42, L) errors vs steps after gap end}``.
    """
    horizon = C.N_STEPS
    rows: List[Dict[str, Any]] = []
    curves = np.full((len(grid["synth_id"]), 42, 200), np.nan, dtype=np.float32)
    rho = np.concatenate([grid["up_st_rho"][:, C.KEPT_ST], grid["up_ph_rho"][:, C.KEPT_PH]], axis=1).astype(np.float32) / 255.0
    for i, sid in enumerate(grid["synth_id"]):
        srow = synth.loc[synth.synth_id == sid].iloc[0]
        gp = gaps[gaps.synth_id == sid].iloc[0]
        b = int(srow.base_id)
        err = np.abs(grid_n[i] - true_n[b]) / sigma[:, None]
        k_end = -(-int(gp.stop) // S.RAW_PER_STEP)
        k_start = int(gp.start) // S.RAW_PER_STEP
        after = err[:, k_end:]
        n_after = after.shape[1]
        curves[i, :, :min(200, n_after)] = after[:, :200]
        off = after > OFF_EPS
        rbad = rho[i][:, k_end:] > C.RHO_EPS
        for c in range(42):
            last_off = int(n_after - np.argmax(off[c][::-1])) if off[c].any() else 0
            last_rho = int(n_after - np.argmax(rbad[c][::-1])) if rbad[c].any() else 0
            rows.append({"synth_id": int(sid), "rec": f"{srow.cohort}/{srow.guid}", "len_s": float(srow.len_s),
                         "position": srow.position, "channel": meta["name"][c], "group": meta["group"][c],
                         "warmup_s": float(meta["warmup_steps"][c] * S.STEP_S),
                         "off_s": last_off * S.STEP_S, "off_censored": bool(last_off == n_after and n_after > 0),
                         "rho_s": last_rho * S.STEP_S, "rho_censored": bool(last_rho == n_after and n_after > 0),
                         "follow_s": n_after * S.STEP_S,
                         "max_err_in_gap": float(err[c, k_start:k_end].max()) if k_end > k_start else 0.0,
                         "max_err_after": float(after[c].max()) if n_after else np.nan,
                         "mean_err_0_60": float(after[c, :15].mean()),
                         "mean_err_60_152": float(after[c, 15:38].mean()),
                         "mean_err_152_300": float(after[c, 38:75].mean()) if n_after > 38 else np.nan,
                         "off_within_window": bool(last_off * S.STEP_S <= meta["warmup_steps"][c] * S.STEP_S)})
    return {"rows": pd.DataFrame(rows), "curves": curves}


def figures(l1: pd.DataFrame, l2: pd.DataFrame, curves: np.ndarray, grid_synth: pd.DataFrame, true: Dict[str, np.ndarray],
            grid: Dict[str, np.ndarray], gaps: pd.DataFrame, base: pd.DataFrame) -> None:
    """Write the T3 figures (thick solid lines).

    Args:
        l1: Level-1 table.
        l2: Level-2 table.
        curves: Level-2 error curves.
        grid_synth: Grid rows of the synthetic table, in ``t2_grid`` order.
        true: Base arrays.
        grid: Grid arrays.
        gaps: Injection table.
        base: Base table.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"rise": "#1f77b4", "peak": "#d62728", "fall": "#2ca02c", "between": "#9467bd"}
    g1 = l1[l1["set"] == "grid"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for pos, col in colors.items():
        sub = g1[g1.position == pos].groupby("len_s")
        axes[0].plot(sub.rmse_rel.mean().index, 100 * sub.rmse_rel.mean().values, "-o", lw=3, color=col, label=pos)
        axes[1].plot(sub.peak_amp_err_rel.mean().index, 100 * sub.peak_amp_err_rel.mean().values, "-o", lw=3, color=col, label=pos)
        axes[2].plot(sub.abs_det_onset_err_s.median().index, sub.abs_det_onset_err_s.median().values, "-o", lw=3, color=col, label=pos)
    for ax, title, yl in zip(axes, ("bridge RMSE inside the gap", "peak amplitude error (bridged - true)", "detector onset error"),
                             ("% of contraction amplitude", "% of contraction amplitude", "median |error| (s)")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("gap length (s)")
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].axhline(10, color="k", lw=2, ls="-", alpha=0.4)
    axes[1].axhline(-20, color="k", lw=2, ls="-", alpha=0.4)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(C.OUT1 / "t3_level1_raw.png", dpi=130)
    plt.close(fig)

    # example traces
    fig, axes = plt.subplots(3, 4, figsize=(18, 9), sharey=False)
    b0 = int(grid_synth.base_id.value_counts().index[0])
    for r, length in enumerate((16.0, 32.0, 64.0)):
        for c, pos in enumerate(("rise", "peak", "fall", "between")):
            ax = axes[r, c]
            hit = grid_synth[(grid_synth.base_id == b0) & (grid_synth.len_s == length) & (grid_synth.position == pos)]
            if hit.empty:
                ax.set_axis_off()
                continue
            i = int(hit.index[0])
            sid = int(hit.synth_id.iloc[0])
            gp = gaps[gaps.synth_id == sid].iloc[0]
            t = np.arange(C.N_RAW) / S.FS
            lo, hi = max(int(gp.start) - 600, 0), min(int(gp.stop) + 600, C.N_RAW)
            ax.plot(t[lo:hi], true["up"][b0][lo:hi], "-", lw=3, color="#444444", label="true")
            ax.plot(t[lo:hi], grid["up"][i][lo:hi], "-", lw=3, color=colors[pos], label="bridged")
            ax.axvspan(gp.start / S.FS, gp.stop / S.FS, color=colors[pos], alpha=0.15)
            ax.set_title(f"{pos}, {int(length)} s gap")
            ax.set_xlabel("time in segment (s)")
            if c == 0:
                ax.set_ylabel("UA (TOCO units)")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(C.OUT1 / "t3_examples.png", dpi=120)
    plt.close(fig)

    # level 2 curves per group and length
    gcolors = {"fast": "#ff7f0e", "mid": "#17becf", "slow": "#8c564b"}
    meta = C.channel_meta()
    lengths = sorted(grid_synth.len_s.unique())
    fig, axes = plt.subplots(1, len(lengths), figsize=(4 * len(lengths), 4.2), sharey=True)
    tt = np.arange(curves.shape[2]) * S.STEP_S
    for ax, length in zip(np.atleast_1d(axes), lengths):
        idx = np.flatnonzero(grid_synth.len_s.values == length)
        for g, col in gcolors.items():
            ch = np.flatnonzero(meta["group"] == g)
            med = np.nanmedian(curves[idx][:, ch, :].reshape(-1, curves.shape[2]), axis=0)
            q90 = np.nanpercentile(curves[idx][:, ch, :].reshape(-1, curves.shape[2]), 90, axis=0)
            ax.plot(tt, med, "-", lw=3, color=col, label=f"{g} median")
            ax.plot(tt, q90, "-", lw=1.5, color=col, alpha=0.6)
        ax.axhline(OFF_EPS, color="k", lw=2, alpha=0.4)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 10)
        ax.set_title(f"gap {int(length)} s")
        ax.set_xlabel("time since gap end (s)")
    np.atleast_1d(axes)[0].set_ylabel("|bridged - true| / sd  (model input space)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.OUT1 / "t3_level2_curves.png", dpi=120)
    plt.close(fig)

    # off duration vs rho duration per channel, by length
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.6))
    for length, col in zip((8.0, 16.0, 32.0, 64.0), ("#1f77b4", "#2ca02c", "#d62728", "#9467bd")):
        sub = l2[l2.len_s == length].groupby("channel", sort=False)
        names = list(meta["name"])
        axes[0].plot(range(42), [sub.off_s.median()[n] for n in names], "-", lw=3, color=col, label=f"off >10% sd, {int(length)} s")
        axes[0].plot(range(42), [sub.rho_s.median()[n] for n in names], ":", lw=2.5, color=col)
        axes[1].plot(range(42), [sub.max_err_after.median()[n] for n in names], "-", lw=3, color=col, label=f"{int(length)} s")
    for ax in axes:
        ax.set_xticks(range(42))
        ax.set_xticklabels(meta["name"], rotation=90, fontsize=7)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("seconds after gap end (median)")
    axes[0].set_title("solid: error > 10% sd; dotted: rho > 0.1")
    axes[1].set_ylabel("max error after gap / sd (median)")
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.OUT1 / "t3_level2_channels.png", dpi=120)
    plt.close(fig)


def main(n_boot: int) -> Dict[str, Any]:
    """Compute both levels, the tables and the figures.

    Args:
        n_boot: Bootstrap resamples.

    Returns:
        The summary dictionary.
    """
    synth = pd.read_parquet(C.OUT1 / "t2_synth.parquet")
    gaps = pd.read_parquet(C.OUT1 / "t2_injections.parquet")
    base = pd.read_parquet(C.OUT1 / "t2_base.parquet")
    true = dict(np.load(C.OUT1 / "t2_base.npz"))
    grid = dict(np.load(C.OUT1 / "t2_grid.npz"))
    mix = dict(np.load(C.OUT1 / "t2_mix.npz"))
    meta = C.channel_meta()

    l1 = level1(synth, gaps, base, true, {"grid": grid, "mix": mix})
    l1.to_parquet(C.OUT1 / "t3_level1.parquet")

    ds = loader_normaliser()
    true_n = normalise(ds, true["up_st"], true["up_ph"])
    grid_n = normalise(ds, grid["up_st"], grid["up_ph"])
    sigma = true_n[:, :, SERVED].transpose(1, 0, 2).reshape(42, -1).std(axis=1)
    sigma_raw = np.concatenate([true["up_st"][:, C.KEPT_ST], true["up_ph"][:, C.KEPT_PH]], axis=1)[:, :, SERVED]
    sigma_raw = sigma_raw.transpose(1, 0, 2).reshape(42, -1).std(axis=1)
    l2 = level2(synth, gaps, true_n, sigma, grid, grid_n, meta)
    l2["rows"].to_parquet(C.OUT1 / "t3_level2.parquet")
    np.savez_compressed(C.OUT1 / "t3_level2_curves.npz", curves=l2["curves"], synth_id=grid["synth_id"],
                        sigma=sigma, sigma_raw=sigma_raw, names=meta["name"])
    grid_synth = synth.set_index("synth_id").loc[grid["synth_id"]].reset_index()
    figures(l1, l2["rows"], l2["curves"], grid_synth, true, grid, gaps, base)

    out: Dict[str, Any] = {"sigma_model_space": dict(zip(meta["name"], sigma.tolist())), "off_eps_sd": OFF_EPS}
    g1 = l1[l1["set"] == "grid"]
    t1: Dict[str, Any] = {}
    for (length, pos), part in g1.groupby(["len_s", "position"]):
        key = f"{int(length)}s/{pos}"
        t1[key] = {"n": int(len(part)), "n_rec": int(part.rec.nunique()),
                   "rmse_units": boot(part, "rmse", n_boot), "rmse_rel_amp": boot(part, "rmse_rel", n_boot),
                   "peak_amp_err_rel": boot(part, "peak_amp_err_rel", n_boot),
                   "peak_amp_abs_err_rel": boot(part, "peak_amp_abs_err_rel", n_boot),
                   "abs_peak_shift_s": boot(part, "abs_peak_shift_s", n_boot),
                   "det_miss_share": boot(part, "det_miss", n_boot),
                   "abs_det_onset_err_s": boot(part, "abs_det_onset_err_s", n_boot),
                   "abs_det_peak_err_s": boot(part, "abs_det_peak_err_s", n_boot),
                   "abs_det_end_err_s": boot(part, "abs_det_end_err_s", n_boot)}
    out["level1_grid"] = t1
    m1 = l1[l1["set"] == "mix"]
    out["level1_mix_by_code"] = {code: {"n": int(len(p)), "rmse_rel_amp": boot(p, "rmse_rel", n_boot),
                                        "len_s_median": float(p.len_s.median()),
                                        "peak_amp_abs_err_rel": boot(p, "peak_amp_abs_err_rel", n_boot)}
                                 for code, p in m1.groupby("code") if len(p) >= 5}
    out["level1_mix_by_len_class"] = {lc: {"n": int(len(p)), "rmse_rel_amp": boot(p, "rmse_rel", n_boot)}
                                      for lc, p in m1.assign(lc=m1.len_s.map(C.len_class)).groupby("lc")}
    r2 = l2["rows"]
    t2: Dict[str, Any] = {}
    for (length, pos, grp), part in r2.groupby(["len_s", "position", "group"]):
        key = f"{int(length)}s/{pos}/{grp}"
        t2[key] = {"n_rows": int(len(part)), "off_s": boot(part, "off_s", n_boot), "rho_s": boot(part, "rho_s", n_boot),
                   "off_censored_share": float(part.off_censored.mean()), "rho_censored_share": float(part.rho_censored.mean()),
                   "max_err_after": boot(part, "max_err_after", n_boot),
                   "mean_err_0_60": boot(part, "mean_err_0_60", n_boot),
                   "mean_err_60_152": boot(part, "mean_err_60_152", n_boot),
                   "share_off_within_window": float(part.off_within_window.mean())}
    out["level2_grid"] = t2
    t2b: Dict[str, Any] = {}
    for (length, grp), part in r2.groupby(["len_s", "group"]):
        t2b[f"{int(length)}s/{grp}"] = {"off_s_median": float(part.off_s.median()), "rho_s_median": float(part.rho_s.median()),
                                        "off_s_q90": float(part.off_s.quantile(0.9)),
                                        "max_err_after_median": float(part.max_err_after.median()),
                                        "max_err_after_q90": float(part.max_err_after.quantile(0.9)),
                                        "mean_err_60_152_median": float(part.mean_err_60_152.median()),
                                        "share_off_within_window": float(part.off_within_window.mean()),
                                        "off_censored_share": float(part.off_censored.mean()),
                                        "share_rho_longer_than_off": float((part.rho_s > part.off_s).mean())}
    out["level2_by_len_group"] = t2b
    out["level2_per_channel_by_len"] = {f"{int(length)}s": part.groupby("channel", sort=False)[["off_s", "rho_s", "max_err_after"]].median().to_dict()
                                        for length, part in r2.groupby("len_s")}
    (C.OUT1 / "t3_summary.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=RUN_ARGS["n_boot"])
    res = main(parser.parse_args().n_boot)
    for key, val in res["level1_grid"].items():
        print(key, "rmse%", round(100 * val["rmse_rel_amp"]["mean"], 1), "peak%", round(100 * val["peak_amp_err_rel"]["mean"], 1),
              "miss", round(val["det_miss_share"]["mean"], 2), "onset", round(val["abs_det_onset_err_s"]["mean"], 1))
    for key, val in res["level2_by_len_group"].items():
        print(key, {k: round(v, 3) for k, v in val.items()})
