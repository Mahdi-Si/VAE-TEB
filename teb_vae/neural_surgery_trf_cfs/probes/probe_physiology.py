r"""Second-pass probe: block ablations, latent probes, calibration, source physiology and paths.

Originally ``probe2.py``. One pass over the evaluation shard measures:

1. **Per-block ablation.** One residual branch at a time is set to zero (the ``LayerScale`` of each
   target-encoder block, the ``out_proj`` of each horizon attention block, each horizon refine
   block, the prior clock and the horizon embedding), and the base and posterior-mean NLL are
   re-scored with the training likelihood.
2. **Linear-probe features.** $\mu^p_t$, $h^y_t$, the current target $y_t$, the last six target
   steps and the targets at $\tau \in \{0, 9, 29\}$ are stored for
   :mod:`~teb_vae.neural_surgery_trf_cfs.probes.analyze_physiology`.
3. **Calibration.** Under the AR(1) residual density the innovation is
   $e_\tau = r_\tau - \phi_c\, r_{\tau-1}$ with variance $s^2_\tau$; the implied marginal residual
   variance is $V_\tau = s^2_\tau + \phi_c^2 V_{\tau-1}$. Both $z^2$ and 90% coverage are
   accumulated per horizon step, split by block and by floor cells ($\ell \le -5.9$).
4. **Source physiology per anchor.** The posterior-mean gain
   $\mathrm{NLL}_{\mathrm{base}} - \mathrm{NLL}_{\mathrm{full}}(\mu^q)$, the KL, the event gate, the
   UA amplitude over the last 120 s, the lag of the UA peak, and the FHR drop and rise over the
   next 120 s.
5. **Persistence path versus latent path.** The base mean is
   $\mu_{\tau,c} = w_{\tau,c}\, y_{t,c} + f_\theta(z)_{\tau,c}$; both parts are compared with the
   target.
"""
from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from teb_vae.lag_attn_rws.nets.losses import ar_innovation, raw_sample_score  # noqa: E402
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask  # noqa: E402
from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402

RUN_ARGS = {
    "checkpoint": common.CHECKPOINT,
    "config": common.CONFIG,
    "overrides": None,               # None: write the default local override file
    "stats": common.STATS,
    "out_dir": common.RESULTS / "session2" / "probe_physiology",
    "max_batches": None,             # None: the whole shard
    "ablation_batches": 12,          # batches used for the per-block ablations
    "probe_every": 5,                # keep every k-th anchor for the linear-probe features
    "seed": 0,
}
FLOOR = -5.9
RAW_PER_STEP = 16


def build_ablations(model):
    """List the residual branches to ablate and patch the refine stack so one block can be skipped.

    Args:
        model: The network. Its ``decoder.core.refine.forward`` is replaced by an equivalent
            forward that skips the residual add of the block named in the returned state.

    Returns:
        ``(ablations, refine_off)``: a mapping from name to either a list of parameters to zero or
        ``("refine", index)``, and the mutable state ``{"idx": None}`` that selects the skipped
        refine block.
    """
    enc = model.target_encoder
    core = model.decoder.core
    ablations = {"reference": []}
    for i, blk in enumerate(enc.conv_blocks):
        ablations[f"enc.conv{i}"] = [blk.layer_scale.weight]
    for i, blk in enumerate(enc.attention_blocks):
        ablations[f"enc.attn{i}.attn"] = [blk.attn_scale.weight]
        ablations[f"enc.attn{i}.ffn"] = [blk.ffn_scale.weight]
    for i, blk in enumerate(core.attention):
        ablations[f"hzn.attn{i}"] = [blk.out_proj.weight]
    ablations["prior.clock"] = [model.prior_head.clock_proj.weight, model.prior_head.clock_proj.bias]
    ablations["hzn.embedding"] = [core.horizon_embedding]
    refine_off = {"idx": None}

    def refine_forward(x, cond=None):
        r = core.refine
        for index, block in enumerate(r.blocks):
            y = block["conv"](x)
            y = torch.nn.functional.gelu(block["norm"](y))
            if r.film is not None and cond is not None:
                gamma, beta = r.film[index](cond).chunk(2, dim=-1)
                y = y * (1.0 + gamma[..., None]) + beta[..., None]
            if refine_off["idx"] != index:
                x = x + y
        return x

    core.refine.forward = refine_forward
    for i in range(len(core.refine.blocks)):
        ablations[f"hzn.refine{i}"] = ("refine", i)
    return ablations, refine_off


def zero_param(p):
    """Zero a parameter in place.

    Args:
        p: The parameter.

    Returns:
        A closure that restores the saved value.
    """
    saved = p.detach().clone()
    p.data.zero_()
    return lambda: p.data.copy_(saved)


def main(args):
    """Run the probe and write ``probe2.json``, ``anchors.npz`` and ``probe_feats.npz``.

    Args:
        args: The run arguments; see :data:`RUN_ARGS`.
    """
    torch.manual_seed(args["seed"])
    out = Path(args["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    dev, cfg, task = common.setup(args["checkpoint"], args["config"], args["overrides"])
    model = task.orig_model
    with h5py.File(args["stats"], "r") as s:
        fhr_mu, fhr_sd = float(s["fhr/mean"][()]), float(s["fhr/std"][()])
        up_mu, up_sd = float(s["up/mean"][()]), float(s["up/std"][()])

    lk = model.forecast_likelihood_kwargs()
    H, C, dz = int(model.horizon), int(lk["cell_mask"].shape[1]), int(model.d_z)
    phi = lk["ar_coef"]
    dec = model.decoder
    w_pers = dec.persistence_weight.detach()
    st_mask = torch.zeros(C, dtype=torch.bool, device=dev)
    st_mask[:common.N_ST] = True
    L = int(model.lag_attn.lag_score_bias.shape[1])
    ablations, refine_off = build_ablations(model)
    abl_acc = {k: np.zeros(6) for k in ablations}
    abl_n = {k: 0.0 for k in ablations}

    dd = dict(device=dev, dtype=torch.float64)
    cal = {k: torch.zeros(H, 2, **dd) for k in ["n", "z2", "c90", "n_floor", "z2_floor", "c90_floor",
                                               "m_c90", "m_z2", "m_c50"]}
    pit_bins = 20
    pit = torch.zeros(2, pit_bins, **dd)
    mpit = torch.zeros(2, pit_bins, **dd)
    path = {k: torch.zeros(H, 2, **dd) for k in ["n", "var_tgt", "var_p", "var_f", "cov_fp", "se_p", "se_full",
                                                 "cov_f_resid", "var_resid"]}
    rows = {k: [] for k in ["guid", "sample", "anchor", "gap_mean", "gap_sample", "kl", "gate", "open",
                            "ua_amp", "ua_peak_lag", "fhr_drop", "fhr_rise", "se_base_st", "se_full_st",
                            "n_st", "argmax0", "argmax1", "argmax2", "argmax3", "ua_now"]}
    probe = {k: [] for k in ["mu_prior", "h_y", "y_now", "y_past", "y_fut0", "y_fut9", "y_fut29",
                             "mb0", "mb9", "mb29", "ok", "guid"]}
    sample_offset = 0
    t0 = time.time()

    for bi, batch in enumerate(common.batches(task, cfg, dev, args["max_batches"])):
        with torch.no_grad():
            inputs = task._build_forward_inputs(batch)
            tf, weight = task._build_raw_target(batch)
            weight = task._target_quality_weight(batch, weight)
            o = model(*inputs)
            w = model.scored_weight(weight)
            anchors, av = o["anchor_index"], o["anchor_valid"]
            tgt = model._build_forecast_target(tf, anchors)
            smask, _ = forecast_mask(w, model.geometry, coverage_floor=model.coverage_floor,
                                     anchors=anchors, anchor_valid=av)
            smask = smask.to(tgt.dtype)
            M = smask[..., None] * lk["cell_mask"]
            B, A = anchors.shape
            pers = o["persistence"]
            gi = anchors[:, :, None].expand(-1, -1, dz)
            mu_fm, lv_fm = dec(o["mu_post"].gather(1, gi), persistence=pers)
            mu_b, lv_b = o["mu_base"], o["logvar_base"]

            def nll(mu, lv):
                return (raw_sample_score(mu, tgt, likelihood="gaussian_nll", logvar=lv,
                                         cell_mask=lk["cell_mask"], ar_coef=phi, step_mask=smask) * M)

            nb = nll(mu_b, lv_b).sum((-1, -2))
            nfm = nll(mu_fm, lv_fm).sum((-1, -2))
            nfs = nll(o["mu_full"], o["logvar_full"]).sum((-1, -2))

            # calibration of the base branch, innovation level and implied marginal
            e, le = ar_innovation(tgt - mu_b, phi, smask, lv_b)
            z = e * torch.exp(-0.5 * le)
            fl = (lv_b <= FLOOR).to(M.dtype) * M
            s2 = torch.exp(lv_b)
            V = torch.zeros_like(s2)
            V[..., 0, :] = s2[..., 0, :]
            for tau in range(1, H):
                V[..., tau, :] = s2[..., tau, :] + phi ** 2 * V[..., tau - 1, :]
            zm = (tgt - mu_b) / V.sqrt()
            normal = torch.distributions.Normal(0.0, 1.0)
            u_inn = normal.cdf(z.clamp(-8, 8))
            u_mar = normal.cdf(zm.clamp(-8, 8))
            for bidx, blk in enumerate([st_mask, ~st_mask]):
                Mb = M * blk.to(M.dtype)
                Fb = fl * blk.to(M.dtype)
                cal["n"][:, bidx] += Mb.double().sum((0, 1, 3))
                cal["z2"][:, bidx] += (z ** 2 * Mb).double().sum((0, 1, 3))
                cal["c90"][:, bidx] += ((z.abs() < 1.6449) * Mb).double().sum((0, 1, 3))
                cal["n_floor"][:, bidx] += Fb.double().sum((0, 1, 3))
                cal["z2_floor"][:, bidx] += (z ** 2 * Fb).double().sum((0, 1, 3))
                cal["c90_floor"][:, bidx] += ((z.abs() < 1.6449) * Fb).double().sum((0, 1, 3))
                cal["m_c90"][:, bidx] += ((zm.abs() < 1.6449) * Mb).double().sum((0, 1, 3))
                cal["m_c50"][:, bidx] += ((zm.abs() < 0.6745) * Mb).double().sum((0, 1, 3))
                cal["m_z2"][:, bidx] += (zm.clamp(-30, 30) ** 2 * Mb).double().sum((0, 1, 3))
                sel = Mb > 0
                pit[bidx] += torch.histc(u_inn[sel].float(), bins=pit_bins, min=0, max=1).double()
                mpit[bidx] += torch.histc(u_mar[sel].float(), bins=pit_bins, min=0, max=1).double()
                p_path = w_pers * pers[..., None, :]
                f_path = mu_b - p_path
                resid = tgt - p_path
                for name, val in [("n", Mb), ("var_tgt", tgt ** 2 * Mb), ("var_p", p_path ** 2 * Mb),
                                  ("var_f", f_path ** 2 * Mb), ("cov_fp", f_path * p_path * Mb),
                                  ("se_p", (tgt - p_path) ** 2 * Mb), ("se_full", (tgt - mu_b) ** 2 * Mb),
                                  ("cov_f_resid", f_path * resid * Mb), ("var_resid", resid ** 2 * Mb)]:
                    path[name][:, bidx] += val.double().sum((0, 1, 3))

            # raw-signal events per anchor, in bpm and in stand-in UA units
            fhr = batch["fhr"].reshape(B, -1) * fhr_sd + fhr_mu
            up = batch["up"].reshape(B, -1) * up_sd + up_mu
            k16 = torch.ones(1, 1, 16, device=dev) / 16
            fhr_s = torch.nn.functional.conv1d(fhr[:, None], k16, padding=8)[:, 0, : fhr.shape[1]]
            up_s = torch.nn.functional.conv1d(up[:, None], k16, padding=8)[:, 0, : up.shape[1]]
            up_step = up.reshape(B, -1, RAW_PER_STEP).mean(-1)
            end = (anchors + 1) * RAW_PER_STEP
            n_raw = fhr.shape[1]

            def window(sig, start_off, stop_off):
                idx = (end[..., None] + torch.arange(start_off, stop_off, device=dev)).clamp(0, n_raw - 1)
                return sig.gather(1, idx.reshape(B, -1)).reshape(B, A, -1)

            past_fhr = window(fhr_s, -240, 0)
            fut_fhr = window(fhr_s, 0, H * RAW_PER_STEP)
            base_fhr = past_fhr.median(-1).values
            fhr_drop = base_fhr - fut_fhr.min(-1).values
            fhr_rise = fut_fhr.max(-1).values - base_fhr
            up_long = window(up_s, -1200, 0)
            ua_amp = window(up_s, -480, 0).max(-1).values - up_long.quantile(0.1, dim=-1)
            ua_now = window(up_s, -16, 0).mean(-1) - up_long.quantile(0.1, dim=-1)
            lag_idx = anchors[..., None] - torch.arange(L, device=dev)
            up_lags = up_step.gather(1, lag_idx.clamp_min(0).reshape(B, -1)).reshape(B, A, L)
            ua_peak_lag = up_lags.argmax(-1)

            gate = o["event_gate"][..., 0].gather(1, anchors)
            opn = o["source_open"].gather(1, anchors).float()
            kl = o["kld_per_t"].gather(1, anchors)
            alpha = o["attn_weights"]
            am = alpha.gather(1, anchors[:, :, None, None].expand(-1, -1, alpha.shape[2], L)).argmax(-1)
            Mst = M * st_mask.to(M.dtype)
            se_b = ((tgt - mu_b) ** 2 * Mst).sum((-1, -2))
            se_f = ((tgt - mu_fm) ** 2 * Mst).sum((-1, -2))
            nst = Mst.sum((-1, -2))
            contrib = (smask.sum(-1) > 0) & av
            # A stable per-recording identifier (the original run used Python's salted hash).
            gid = torch.tensor([zlib.crc32(str(g).encode()) % (2 ** 31) for g in batch["guid"]], device=dev)
            gidA = gid[:, None].expand(B, A)
            sampA = (torch.arange(B, device=dev)[:, None] + sample_offset).expand(B, A)
            for key, val in [("guid", gidA), ("sample", sampA), ("anchor", anchors), ("gap_mean", nb - nfm),
                             ("gap_sample", nb - nfs), ("kl", kl), ("gate", gate), ("open", opn),
                             ("ua_amp", ua_amp), ("ua_peak_lag", ua_peak_lag), ("fhr_drop", fhr_drop),
                             ("fhr_rise", fhr_rise), ("se_base_st", se_b), ("se_full_st", se_f),
                             ("n_st", nst), ("argmax0", am[..., 0]), ("argmax1", am[..., 1]),
                             ("argmax2", am[..., 2]), ("argmax3", am[..., 3]), ("ua_now", ua_now)]:
                rows[key].append(val[contrib].float().cpu())

            # linear-probe features, every probe_every-th anchor
            ev = args["probe_every"]
            sub = contrib.clone()
            sub[:, (torch.arange(A, device=dev) % ev) != (bi % ev)] = False
            target_all = model.target_gate(torch.cat(inputs[:2], -1))

            def at(x):
                return x.gather(1, anchors[:, :, None].expand(-1, -1, x.shape[-1]))[sub]
            probe["mu_prior"].append(at(o["mu_prior"]).cpu())
            probe["h_y"].append(at(o["target_state"]).cpu())
            probe["y_now"].append(at(target_all).cpu())
            past = torch.cat([target_all.gather(1, (anchors - k).clamp_min(0)[:, :, None].expand(-1, -1, C))
                              for k in range(6)], -1)
            probe["y_past"].append(past[sub].cpu())
            for tau in (0, 9, 29):
                probe[f"y_fut{tau}"].append(tgt[:, :, tau][sub].cpu())
                probe[f"mb{tau}"].append(mu_b[:, :, tau][sub].cpu())
            probe["ok"].append(torch.stack([smask[:, :, tau] for tau in (0, 9, 29)], -1)[sub].cpu())
            probe["guid"].append(gidA[sub].cpu())

            # per-block ablations
            if bi < args["ablation_batches"]:
                for name, spec in ablations.items():
                    restore = []
                    if isinstance(spec, tuple):
                        refine_off["idx"] = spec[1]
                    else:
                        restore = [zero_param(p) for p in spec if p is not None]
                    oa = model(*inputs)
                    mfa, lfa = dec(oa["mu_post"].gather(1, gi), persistence=pers)
                    nba = nll(oa["mu_base"], oa["logvar_base"]).sum((-1, -2))[contrib].sum().item()
                    nfa = nll(mfa, lfa).sum((-1, -2))[contrib].sum().item()
                    err = (tgt - oa["mu_base"]) ** 2
                    Mst_ = M * st_mask.to(M.dtype)
                    Mph_ = M * (~st_mask).to(M.dtype)
                    abl_acc[name] += np.array([nba, nfa, (err * Mst_).sum().item(), Mst_.sum().item(),
                                               (err * Mph_).sum().item(), Mph_.sum().item()])
                    abl_n[name] += contrib.sum().item()
                    refine_off["idx"] = None
                    for r in restore:
                        r()
        sample_offset += B
        if bi % 10 == 0:
            print(f"batch {bi} {time.time() - t0:.0f}s", flush=True)

    res = {}
    abl = {}
    for name in ablations:
        a, n = abl_acc[name], abl_n[name]
        abl[name] = {"nll_base_per_anchor": a[0] / n, "nll_fullmean_per_anchor": a[1] / n,
                     "mse_base_st": a[2] / a[3], "mse_base_ph": a[4] / a[5], "anchors": n}
    ref = abl["reference"]
    for v in abl.values():
        v["d_nll_base"] = v["nll_base_per_anchor"] - ref["nll_base_per_anchor"]
        v["d_nll_fullmean"] = v["nll_fullmean_per_anchor"] - ref["nll_fullmean_per_anchor"]
        v["gap_mean"] = v["nll_base_per_anchor"] - v["nll_fullmean_per_anchor"]
    res["ablations"] = abl
    c = {k: v.cpu().numpy() for k, v in cal.items()}
    res["calibration"] = {
        "tau": list(range(H)),
        "innov_z2": (c["z2"] / c["n"]).T.tolist(),
        "innov_cov90": (c["c90"] / c["n"]).T.tolist(),
        "floor_frac": (c["n_floor"] / c["n"]).T.tolist(),
        "floor_z2": (c["z2_floor"] / np.maximum(c["n_floor"], 1)).T.tolist(),
        "floor_cov90": (c["c90_floor"] / np.maximum(c["n_floor"], 1)).T.tolist(),
        "marg_cov90": (c["m_c90"] / c["n"]).T.tolist(),
        "marg_cov50": (c["m_c50"] / c["n"]).T.tolist(),
        "marg_z2": (c["m_z2"] / c["n"]).T.tolist(),
        "pit_innov": (pit / pit.sum(-1, keepdim=True)).cpu().numpy().tolist(),
        "pit_marg": (mpit / mpit.sum(-1, keepdim=True)).cpu().numpy().tolist(),
        "nonfloor_z2_all": float(((c["z2"] - c["z2_floor"]).sum() / (c["n"] - c["n_floor"]).sum())),
        "floor_z2_all": float(c["z2_floor"].sum() / c["n_floor"].sum()),
    }
    p = {k: v.cpu().numpy() for k, v in path.items()}
    res["paths"] = {
        "share_var_persistence": (p["var_p"] / p["var_tgt"]).T.tolist(),
        "share_var_latent": (p["var_f"] / p["var_tgt"]).T.tolist(),
        "corr_latent_vs_residual": (p["cov_f_resid"] / np.sqrt(p["var_f"] * p["var_resid"])).T.tolist(),
        "mse_persistence_path_only": (p["se_p"] / p["n"]).T.tolist(),
        "mse_full_mean": (p["se_full"] / p["n"]).T.tolist(),
        "mse_climatology_zero": (p["var_tgt"] / p["n"]).T.tolist(),
    }
    np.savez(out / "anchors.npz", **{k: torch.cat(v).numpy() for k, v in rows.items()})
    np.savez(out / "probe_feats.npz", **{k: torch.cat(v).numpy() for k, v in probe.items()})
    with open(out / "probe2.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main(RUN_ARGS)
