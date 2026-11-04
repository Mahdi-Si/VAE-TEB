r"""Turn the session-1 probe output into tables and figures (session 1).

Reads ``probe.json``, ``probe_cells.npz`` and ``probe_anchors.npz`` from
:mod:`~teb_vae.neural_surgery_trf_cfs.probes.probe_layers` and the run's metrics history, and
writes ``summary_blocks.json``, ``per_channel.csv``, ``per_horizon.csv`` and five figures:
training history, per-cell maps, latent statistics, attention profiles and horizon skill.

Block skill against persistence is $1 - \mathrm{MSE}/\mathrm{MSE}_{\mathrm{pers}}$. The fitted
Gaussian reference NLL per cell is $\tfrac12(\log 2\pi + \log \mathrm{MSE} + 1)$, the NLL of an
oracle homoscedastic variance per $(\tau, c)$.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402

RUN_ARGS = {
    "probe_dir": common.RESULTS / "session1" / "probe_out",
    "out_dir": common.RESULTS / "session1",
    "fig_dir": common.RESULTS / "session1" / "figures",
    "history": common.HISTORY,
}
BLUE, ORANGE, AQUA, YELLOW, MAG = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"axes.facecolor": SURF, "figure.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": "#e4e3df",
                     "grid.linewidth": 0.6, "lines.linewidth": 2, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def main(args):
    """Build the tables and figures from the session-1 probe output.

    Args:
        args: The run arguments; see :data:`RUN_ARGS`.
    """
    P = Path(args["probe_dir"])
    S = Path(args["out_dir"])
    FIG = Path(args["fig_dir"])
    FIG.mkdir(parents=True, exist_ok=True)
    HIST = args["history"]
    r = json.load(open(P / "probe.json"))
    c = dict(np.load(P / "probe_cells.npz"))
    a = dict(np.load(P / "probe_anchors.npz"))
    H, C = r["meta"]["H"], r["meta"]["C"]
    n_st = 32
    cnt = np.where(c["cnt"] > 0, c["cnt"], np.nan)

    def per(key):
        return c[key] / cnt

    mse_f, mse_b, mse_p, mse_c, mse_pp, mse_fm = (per(k) for k in
                                                    ["se_full", "se_base", "se_pers", "se_clim", "se_ppath", "se_fullmean"])
    var_t = per("tsq") - per("tsum") ** 2
    out = {}

    # ------------------------------------------------ block-level skill
    def block(mask_c):
        sl = (slice(None), mask_c)
        tot = lambda k: np.nansum(c[k][sl])  # noqa: E731
        n = np.nansum(c["cnt"][sl])
        return {
            "cells": n,
            "mse_full": tot("se_full") / n, "mse_full_postmean": tot("se_fullmean") / n,
            "mse_base": tot("se_base") / n, "mse_persistence": tot("se_pers") / n,
            "mse_climatology": tot("se_clim") / n, "mse_persistence_path_only": tot("se_ppath") / n,
            "skill_vs_persistence_full": 1 - tot("se_full") / tot("se_pers"),
            "skill_vs_persistence_base": 1 - tot("se_base") / tot("se_pers"),
            "nll_full_per_cell": tot("nll_full") / n, "nll_base_per_cell": tot("nll_base") / n,
            "nll_full_fact_per_cell": tot("nll_full_fact") / n,
            "floor_frac": tot("floor_full") / n,
            "z2_full": tot("z2_full") / n, "z2_base": tot("z2_base") / n,
            "cov90_full": tot("cov90_full") / n, "cov90_base": tot("cov90_base") / n,
            "floor_cells_mse_over_floor_var": (tot("se_full_floor") / max(tot("cnt_floor"), 1)) / np.exp(-6),
        }
    chan = np.arange(C)
    out["block_all"] = block(chan >= 0)
    out["block_st"] = block(chan < n_st)
    out["block_ph"] = block(chan >= n_st)

    # fitted Gaussian persistence/climatology reference NLL per cell (oracle variance per (tau, c))
    def fitted_nll(mse):
        return 0.5 * (np.log(2 * np.pi) + np.log(mse) + 1)
    w = c["cnt"]
    for name, m in [("persistence", mse_p), ("climatology", mse_c), ("model_full_mean", mse_f)]:
        v = np.nansum(fitted_nll(m) * w) / np.nansum(w)
        out[f"fitted_gauss_nll_per_cell_{name}"] = v

    # per-channel table
    rows = []
    for k in range(C):
        sl = (slice(None), k)
        n = c["cnt"][sl].sum()
        rows.append({
            "chan": k, "block": "st" if k < n_st else "ph",
            "scored_H": int((c["cnt"][sl] > 0).sum()),
            "skill_full_vs_pers": 1 - c["se_full"][sl].sum() / c["se_pers"][sl].sum(),
            "skill_full_vs_clim": 1 - c["se_full"][sl].sum() / c["se_clim"][sl].sum(),
            "skill_tau0": 1 - c["se_full"][0, k] / c["se_pers"][0, k],
            "skill_tau_last": 1 - c["se_full"][int((c["cnt"][sl] > 0).sum()) - 1, k]
            / c["se_pers"][int((c["cnt"][sl] > 0).sum()) - 1, k],
            "gap_nats_per_cell": (c["nll_base"][sl].sum() - c["nll_full"][sl].sum()) / n,
            "nll_full_per_cell": c["nll_full"][sl].sum() / n,
            "floor_frac": c["floor_full"][sl].sum() / n,
            "z2": c["z2_full"][sl].sum() / n,
            "cov90": c["cov90_full"][sl].sum() / n,
            "ar_phi": r["static"]["ar_phi"][k],
            "target_var": np.nansum(var_t[:, k] * c["cnt"][:, k]) / n,
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(S / "per_channel.csv", index=False)

    # horizon curves
    hz = pd.DataFrame({
        "tau": np.arange(H),
        "skill_st": 1 - np.nansum(c["se_full"][:, :n_st], 1) / np.nansum(c["se_pers"][:, :n_st], 1),
        "skill_ph": 1 - np.nansum(c["se_full"][:, n_st:], 1) / np.nansum(c["se_pers"][:, n_st:], 1),
        "skill_st_base": 1 - np.nansum(c["se_base"][:, :n_st], 1) / np.nansum(c["se_pers"][:, :n_st], 1),
        "gap_per_cell": (np.nansum(c["nll_base"], 1) - np.nansum(c["nll_full"], 1)) / np.nansum(c["cnt"], 1),
        "floor_frac": np.nansum(c["floor_full"], 1) / np.nansum(c["cnt"], 1),
        "z2": np.nansum(c["z2_full"], 1) / np.nansum(c["cnt"], 1),
        "cov90": np.nansum(c["cov90_full"], 1) / np.nansum(c["cnt"], 1),
    })
    hz.to_csv(S / "per_horizon.csv", index=False)

    with open(S / "summary_blocks.json", "w") as f:
        json.dump(out, f, indent=1, default=float)

    # ------------------------------------------------ figures
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
    df = pd.read_csv(HIST)
    ax[0, 0].plot(df.epoch, df["val/nll_full_block"], color=BLUE, label="val full")
    ax[0, 0].plot(df.epoch, df["val/nll_base_block"], color=ORANGE, label="val base")
    ax[0, 0].set_ylim(-1650, -1450); ax[0, 0].set_title("Validation block NLL (nats per anchor)"); ax[0, 0].legend()
    ax[0, 1].plot(df.epoch, df["val/pred_gap"], color=BLUE, label="pred_gap (sampled)")
    ax[0, 1].plot(df.epoch, df["val/pred_gap_mean"], color=ORANGE, label="pred_gap at means")
    ax[0, 1].plot(df.epoch, df["val/source_conditioned_kl_raw"], color=AQUA, label="KL")
    ax[0, 1].plot(df.epoch, df["val/kld_source_null"], color=YELLOW, label="KL, source values removed")
    ax[0, 1].set_ylim(-2, 12); ax[0, 1].set_title("Source benefit vs. KL (nats per anchor)"); ax[0, 1].legend(fontsize=7)
    ax[0, 2].plot(df.epoch, df["val/shuffle_penalty"], color=BLUE)
    ax[0, 2].set_ylim(0, 30); ax[0, 2].set_title("Shuffle penalty (nats per anchor)")
    ax[1, 0].plot(df.epoch, df["val/logvar_full_floor_frac"], color=BLUE, label="decoder log-var at floor")
    ax[1, 0].plot(df.epoch, df["val/logvar_prior_floor_frac"], color=ORANGE, label="prior log-var at floor")
    ax[1, 0].plot(df.epoch, df["val/kld_active_frac"], color=AQUA, label="active latent dims")
    ax[1, 0].set_title("Fractions"); ax[1, 0].legend(fontsize=7)
    ax[1, 1].plot(df.epoch, df["train/grad_norm"], color=BLUE)
    ax[1, 1].set_title("Pre-clip gradient norm")
    ax[1, 2].plot(df.epoch, df["val/pred_gap_tau_first"], color=BLUE, label="tau first")
    ax[1, 2].plot(df.epoch, df["val/pred_gap_tau_last"], color=ORANGE, label="tau last")
    ax[1, 2].plot(df.epoch, df["val/pred_gap_ph"], color=AQUA, label="phase block")
    ax[1, 2].set_ylim(-1.5, 0.5); ax[1, 2].set_title("Gap splits (nats per anchor)"); ax[1, 2].legend(fontsize=7)
    for x in ax.flat:
        x.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(FIG / "01_training_history.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    sk = 1 - mse_f / mse_p
    im = ax[0].imshow(sk.T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1, origin="lower")
    ax[0].axhline(n_st - 0.5, color=INK, lw=0.8)
    ax[0].set_title("MSE skill vs. persistence, 1 - MSE/MSE_pers"); ax[0].set_xlabel("horizon step tau"); ax[0].set_ylabel("kept channel (st below line, ph above)")
    fig.colorbar(im, ax=ax[0])
    im = ax[1].imshow((c["floor_full"] / cnt).T, aspect="auto", cmap="Blues", vmin=0, vmax=1, origin="lower")
    ax[1].axhline(n_st - 0.5, color=INK, lw=0.8); ax[1].set_title("Share of cells with decoder log-var at floor (-6)"); ax[1].set_xlabel("tau")
    fig.colorbar(im, ax=ax[1])
    im = ax[2].imshow(np.log10(c["z2_full"] / cnt).T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    ax[2].axhline(n_st - 0.5, color=INK, lw=0.8); ax[2].set_title("log10 E[z^2] of AR innovations (0 = calibrated)"); ax[2].set_xlabel("tau")
    fig.colorbar(im, ax=ax[2])
    fig.tight_layout(); fig.savefig(FIG / "02_cells.png", dpi=130); plt.close(fig)

    lat = r["latent"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    d = np.arange(len(lat["kl_per_dim"]))
    order = np.argsort(lat["kl_per_dim"])[::-1]
    ax[0].bar(d, np.array(lat["kl_per_dim"])[order], color=BLUE)
    ax[0].set_yscale("log"); ax[0].set_title("Source KL per latent dim (sorted), nats per anchor")
    ax[1].scatter(lat["prior_mu_var_across_anchors"], lat["decoder_jacobian_rms"], s=14, color=BLUE)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("Var of prior mean across anchors"); ax[1].set_ylabel("decoder Jacobian RMS")
    ax[1].set_title("Which latent dims carry history information")
    ax[2].scatter(lat["prior_logvar_mean"], lat["kl_per_dim"], s=14, color=ORANGE)
    ax[2].set_yscale("log"); ax[2].set_xlabel("mean prior log-variance"); ax[2].set_ylabel("KL per dim")
    ax[2].set_title("Prior width vs. KL")
    fig.tight_layout(); fig.savefig(FIG / "03_latent.png", dpi=130); plt.close(fig)

    la = r["lag_attention"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    cols = [BLUE, ORANGE, AQUA, YELLOW]
    for h, p in enumerate(la["mean_profile"]):
        ax[0].plot(np.arange(len(p)) * 4, p, color=cols[h], label=f"head {h}")
    ax[0].set_xlabel("lag (s, feature grid)"); ax[0].set_title("Mean lag-attention profile"); ax[0].legend()
    for h, p in enumerate(la["profile_high_kl"]):
        ax[1].plot(np.arange(len(p)) * 4, p, color=cols[h])
    ax[1].set_xlabel("lag (s)"); ax[1].set_title("Profile at anchors with KL > 5 nats")
    for i, p in r["encoder"]["self_attention_backprofile"].items():
        ax[2].plot(np.arange(len(p)) * 4, p, label=f"block {i}")
    ax[2].set_yscale("log"); ax[2].set_xlabel("look-back (s)"); ax[2].set_title("Encoder self-attention vs. look-back"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "04_attention.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(hz.tau, hz.skill_st, color=BLUE, label="scattering block")
    ax[0].plot(hz.tau, hz.skill_ph, color=ORANGE, label="phase block")
    ax[0].axhline(0, color=INK2, lw=0.8); ax[0].set_title("MSE skill vs. persistence by horizon"); ax[0].set_xlabel("tau (4 s steps)"); ax[0].legend()
    ax[1].hist(np.clip(a["gap"], -80, 80), bins=80, color=BLUE)
    ax[1].set_title("Per-anchor pred_gap (base - full NLL), nats"); ax[1].axvline(0, color=INK)
    fig.tight_layout(); fig.savefig(FIG / "05_horizon_gap.png", dpi=130); plt.close(fig)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(RUN_ARGS)
