r"""Analyse the physiology probe: linear probes, source gain by physiology, attention vs UA, figures.

Originally ``analyze2.py``. Reads the output of
:mod:`~teb_vae.neural_surgery_trf_cfs.probes.probe_physiology` and writes ``analysis2.json`` and
four figures (calibration, block ablation, source gain by quintile, persistence and latent paths).

* **Linear probes.** Ridge regressions, fitted on recordings with an even identifier and scored on
  the odd ones, read $y_t$ and the change $y_{t+1+\tau} - y_t$ from $\mu^p_t$, from $h^y_t$, from
  the last six target steps and from both. Pooled $R^2 = 1 - \mathrm{SSE}/\mathrm{SST}$ is
  reported per block.
* **Source gain.** The posterior-mean gain per anchor is binned by quintile of the UA amplitude
  and of the FHR drop, with a recording-level bootstrap interval.
* **Attention against the UA peak.** The argmax lag of each head is compared with the lag of the
  UA peak at strong-contraction anchors, against a shuffled reference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402

RUN_ARGS = {
    "in_dir": common.RESULTS / "session2" / "probe_physiology",
    "fig_dir": common.RESULTS / "session2" / "figures",
    "boot": 1000,
    "seed": 0,
}
S1, S2 = "#2a78d6", "#eb6834"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def block_r2(P, Y, ref_mean):
    r"""Pooled $R^2$ per block against a reference mean.

    Args:
        P: Predictions $(N, C)$.
        Y: Targets $(N, C)$.
        ref_mean: The per-channel mean of the training targets $(C,)$.

    Returns:
        ``{"st": R2, "ph": R2}``.
    """
    out = {}
    for name, sl in [("st", slice(0, common.N_ST)), ("ph", slice(common.N_ST, None))]:
        sse = ((Y[:, sl] - P[:, sl]) ** 2).sum()
        sst = ((Y[:, sl] - ref_mean[sl]) ** 2).sum()
        out[name] = float(1 - sse / sst)
    return out


def ridge_r2(Xtr, Ytr, Xte, Yte, lam=1.0):
    r"""Fit a standardised ridge regression on the training rows and score it on the test rows.

    Args:
        Xtr: Training features $(N_{tr}, F)$.
        Ytr: Training targets $(N_{tr}, C)$.
        Xte: Test features.
        Yte: Test targets.
        lam: Ridge strength; the penalty is $10^{-3}\,\lambda\, N_{tr}$ times the identity.

    Returns:
        The pooled test $R^2$ per block, see :func:`block_r2`.
    """
    mx, my = Xtr.mean(0), Ytr.mean(0)
    sx = Xtr.std(0) + 1e-6
    A = (Xtr - mx) / sx
    W = np.linalg.solve(A.T @ A + lam * len(A) * 1e-3 * np.eye(A.shape[1]), A.T @ (Ytr - my))
    P = ((Xte - mx) / sx) @ W + my
    return block_r2(P, Yte, Ytr.mean(0))


def boot_mean(x, groups, n, rng):
    """Mean of ``x`` with a recording-level bootstrap 95% interval.

    Args:
        x: Per-anchor values.
        groups: Recording identifier per anchor.
        n: Number of bootstrap resamples.
        rng: A NumPy generator.

    Returns:
        ``(mean, lo, hi)``.
    """
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=x)
    cnts = np.bincount(inv)
    draws = rng.integers(0, len(ug), size=(n, len(ug)))
    m = sums[draws].sum(1) / cnts[draws].sum(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def linear_probes(P):
    """Run every linear probe of the latent and the encoder state.

    Args:
        P: The ``probe_feats.npz`` arrays.

    Returns:
        A dictionary of pooled $R^2$ values per probe and horizon step.
    """
    tr = (P["guid"] % 2) == 0
    te = ~tr
    lp = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
          "y_now_from_mu_prior": ridge_r2(P["mu_prior"][tr], P["y_now"][tr], P["mu_prior"][te], P["y_now"][te]),
          "y_now_from_h_y": ridge_r2(P["h_y"][tr], P["y_now"][tr], P["h_y"][te], P["y_now"][te])}
    for k, tau in enumerate((0, 9, 29)):
        ok = P["ok"][:, k] > 0
        t_, e_ = tr & ok, te & ok
        D = P[f"y_fut{tau}"] - P["y_now"]
        both = np.c_[P["y_past"], P["mu_prior"]]
        lp[f"delta_tau{tau}"] = {
            "from_mu_prior": ridge_r2(P["mu_prior"][t_], D[t_], P["mu_prior"][e_], D[e_]),
            "from_h_y": ridge_r2(P["h_y"][t_], D[t_], P["h_y"][e_], D[e_]),
            "from_past6_linear": ridge_r2(P["y_past"][t_], D[t_], P["y_past"][e_], D[e_]),
            "from_past6_plus_mu": ridge_r2(both[t_], D[t_], both[e_], D[e_]),
            "decoder_mean": block_r2(P[f"mb{tau}"][e_] - P["y_now"][e_], D[e_], D[t_].mean(0)),
        }
    return lp


def source_physiology(R, boot, rng):
    """Bin the source gain by UA amplitude and FHR drop; compare attention with the UA peak lag.

    Args:
        R: The ``anchors.npz`` arrays.
        boot: Bootstrap resamples.
        rng: A NumPy generator.

    Returns:
        A dictionary of binned gains, a 2-by-2 UA-by-drop table, correlations and the attention
        comparison.
    """
    g = R["guid"]
    dmse = (R["se_base_st"] - R["se_full_st"]) / np.maximum(R["n_st"], 1)
    src = {"anchors": int(len(g)), "recordings": int(len(np.unique(g))),
           "gap_mean_all": boot_mean(R["gap_mean"], g, boot, rng),
           "gap_sample_all": boot_mean(R["gap_sample"], g, boot, rng),
           "frac_gap_mean_pos": float((R["gap_mean"] > 0).mean()),
           "dmse_st_all": boot_mean(dmse, g, boot, rng),
           "mse_base_st": float(R["se_base_st"].sum() / R["n_st"].sum())}
    for key in ("ua_amp", "fhr_drop"):
        q = np.quantile(R[key], [0, .2, .4, .6, .8, 1])
        bins = []
        for i in range(5):
            m = (R[key] >= q[i]) & (R[key] <= q[i + 1])
            bins.append({"lo": float(q[i]), "hi": float(q[i + 1]), "n": int(m.sum()),
                         "gap_mean": boot_mean(R["gap_mean"][m], g[m], boot, rng),
                         "kl": float(R["kl"][m].mean()), "gate": float(R["gate"][m].mean()),
                         "open": float(R["open"][m].mean()), "dmse_st": float(dmse[m].mean())})
        src[f"by_{key}"] = bins
    hi_ua = R["ua_amp"] >= np.quantile(R["ua_amp"], 0.8)
    deep = R["fhr_drop"] >= np.quantile(R["fhr_drop"], 0.8)
    src["ua_x_drop"] = {
        nm: {"n": int(m.sum()), "gap_mean": boot_mean(R["gap_mean"][m], g[m], boot, rng),
             "kl": float(R["kl"][m].mean()), "dmse_st": float(dmse[m].mean())}
        for nm, m in [("ua_hi&drop_deep", hi_ua & deep), ("ua_hi&drop_shallow", hi_ua & ~deep),
                      ("ua_lo&drop_deep", ~hi_ua & deep), ("ua_lo&drop_shallow", ~hi_ua & ~deep)]}
    for a_, b_ in [("kl", "ua_amp"), ("kl", "ua_now"), ("gate", "ua_now"), ("kl", "fhr_drop"),
                   ("gap_mean", "fhr_drop")]:
        src[f"corr_{a_}_{b_}"] = float(np.corrcoef(R[a_], R[b_])[0, 1])
    m = hi_ua & (R["open"] > 0)
    att = {}
    for h in range(4):
        am, pk = R[f"argmax{h}"][m], R["ua_peak_lag"][m]
        perm = rng.permutation(len(pk))
        att[f"head{h}"] = {"corr": float(np.corrcoef(am, pk)[0, 1]),
                           "corr_shuffled": float(np.corrcoef(am, pk[perm])[0, 1]),
                           "exact_match": float((am == pk).mean()),
                           "within2": float((np.abs(am - pk) <= 2).mean()),
                           "within2_shuffled": float((np.abs(am - pk[perm]) <= 2).mean())}
    src["attn_vs_ua_peak"] = att
    return src


def figures(res, src, figd):
    """Write the four figures of the analysis.

    Args:
        res: The ``probe2.json`` content.
        src: The output of :func:`source_physiology`.
        figd: Output directory.
    """
    cal = res["calibration"]
    tau = np.arange(1, len(cal["tau"]) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for b, (nm, col) in enumerate([("fhr_st", S1), ("fhr_ph", S2)]):
        z2 = np.array(cal["innov_z2"][b])
        ok = np.isfinite(z2)
        ax[0].plot(tau[ok], z2[ok], color=col, label=nm)
        ax[0].plot(tau[ok], np.array(cal["floor_z2"][b])[ok], color=col, ls=":", label=f"{nm}, floor cells")
        ax[1].plot(tau[ok], np.array(cal["innov_cov90"][b])[ok], color=col, label=f"{nm} innovation")
        ax[1].plot(tau[ok], np.array(cal["marg_cov90"][b])[ok], color=col, ls="--", label=f"{nm} marginal")
        ax[2].plot(tau[ok], np.array(cal["floor_frac"][b])[ok], color=col, label=nm)
    ax[0].axhline(1, color=INK2, lw=1)
    ax[0].set_yscale("log")
    ax[0].set(title="Mean $z^2$ of AR(1) innovations (base)", xlabel="horizon step $\\tau$")
    ax[1].axhline(0.9, color=INK2, lw=1)
    ax[1].set(title="90% interval coverage", xlabel="horizon step $\\tau$")
    ax[2].set(title="Share of cells at the log-variance floor", xlabel="horizon step $\\tau$")
    for x in ax:
        x.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figd / "06_calibration.png", dpi=150)
    plt.close(fig)

    ab = res["ablations"]
    names = [k for k in ab if k != "reference"]
    vals = np.array([ab[k]["d_nll_base"] for k in names])
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(np.array(names)[order], vals[order], color=S1, height=0.7)
    ax.set_xscale("symlog", linthresh=10)
    ax.set(title="Increase in base NLL (nats per anchor) when one residual branch is zeroed",
           xlabel="$\\Delta$ NLL, nats per anchor (symlog)")
    ax.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(figd / "07_block_ablation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for i, (key, lab) in enumerate([("ua_amp", "UA amplitude in last 2 min (stand-in units)"),
                                    ("fhr_drop", "FHR drop over the next 120 s (bpm)")]):
        bins = src[f"by_{key}"]
        x = np.arange(5)
        mval = [b_["gap_mean"][0] for b_ in bins]
        lo = [b_["gap_mean"][0] - b_["gap_mean"][1] for b_ in bins]
        hi = [b_["gap_mean"][2] - b_["gap_mean"][0] for b_ in bins]
        ax[i].bar(x, mval, color=S1, width=0.6, yerr=[lo, hi], ecolor=INK2, capsize=3)
        ax[i].set_xticks(x, [f"{b_['lo']:.0f}–{b_['hi']:.0f}" for b_ in bins], fontsize=7)
        ax[i].set(title=f"Source gain per anchor by quintile of {key}", xlabel=lab,
                  ylabel="NLL(base) − NLL(full, post. mean), nats")
        ax[i].axhline(0, color=INK2, lw=1)
    fig.tight_layout()
    fig.savefig(figd / "08_source_vs_physiology.png", dpi=150)
    plt.close(fig)

    pa = res["paths"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for b, (nm, col) in enumerate([("fhr_st", S1), ("fhr_ph", S2)]):
        mp, mf = np.array(pa["mse_persistence_path_only"][b]), np.array(pa["mse_full_mean"][b])
        ok = np.isfinite(mf)
        ax[0].plot(tau[ok], mp[ok], color=col, ls="--", label=f"{nm}: $w_\\tau y_t$ only")
        ax[0].plot(tau[ok], mf[ok], color=col, label=f"{nm}: full mean")
        ax[1].plot(tau[ok], np.array(pa["corr_latent_vs_residual"][b])[ok], color=col, label=nm)
    ax[0].set(title="MSE of the base mean and of its persistence path", xlabel="horizon step $\\tau$")
    ax[1].set(title="corr(latent path, target − persistence path)", xlabel="horizon step $\\tau$")
    for x in ax:
        x.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figd / "09_paths.png", dpi=150)
    plt.close(fig)


def main(args):
    """Run the analysis and write ``analysis2.json`` and the figures.

    Args:
        args: The run arguments; see :data:`RUN_ARGS`.
    """
    rng = np.random.default_rng(args["seed"])
    ind, figd = Path(args["in_dir"]), Path(args["fig_dir"])
    figd.mkdir(parents=True, exist_ok=True)
    res = json.load(open(ind / "probe2.json"))
    R = dict(np.load(ind / "anchors.npz"))
    P = dict(np.load(ind / "probe_feats.npz"))
    report = {"linear_probes": linear_probes(P), "source": source_physiology(R, args["boot"], rng),
              "ablations": res["ablations"],
              "calibration_summary": {k: res["calibration"][k] for k in ("floor_z2_all", "nonfloor_z2_all")}}
    json.dump(report, open(ind / "analysis2.json", "w"), indent=1)
    figures(res, report["source"], figd)
    print(json.dumps({k: v for k, v in report.items() if k != "ablations"}, indent=1)[:9000])


if __name__ == "__main__":
    main(RUN_ARGS)
