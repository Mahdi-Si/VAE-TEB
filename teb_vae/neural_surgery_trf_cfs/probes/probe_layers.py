r"""Layer-level probe of the SeqVaeLagAttnTrfCfs epoch-524 checkpoint (session 1).

Reuses the evaluation pipeline's own task loader, data loader, forecast mask and scoring function,
so every number is computed with the training definitions. One pass over the shard accumulates:

* static weights: the persistence weights $w_{\tau,c}$, the AR(1) coefficients
  $\phi_c = \tanh(a_c)$, the lag score bias, ``LayerScale`` magnitudes and parameter counts;
* target-encoder activation RMS per block and per residual branch, and self-attention entropy,
  look-back distance and look-back profile per block and head;
* per-cell $(\tau, c)$ MSE of the full, posterior-mean, base, persistence and zero forecasts, the
  AR(1) and factorised NLL, the floor fraction, innovation $z^2$ and 90% coverage;
* latent statistics per dimension (prior mean variance, prior variance, KL, posterior shift) and
  the decoder Jacobian RMS per latent dimension;
* lag-attention profiles, entropies and argmax histograms per head, overall and at high-KL anchors;
* forward ablations of the source (closed, shuffled, zeroed, shifted by 15 steps) and decoder-only
  ablations of the latent and of the persistence input.

Writes ``probe.json``, ``probe_cells.npz`` and ``probe_anchors.npz``.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from teb_vae.lag_attn.config import load_config  # noqa: E402
from teb_vae.lag_attn_cfs.eval.config_schema import (  # noqa: E402
    force_single_process_loader,
    merge_eval_overrides,
)
from teb_vae.lag_attn_cfs.eval.probe import load_task, read_checkpoint  # noqa: E402
from teb_vae.lag_attn_rws.nets.losses import ar_innovation, raw_sample_score  # noqa: E402
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask  # noqa: E402
from teb_vae.lag_attn_transformer_cfs.eval.binding import TRF_CFS_BINDING  # noqa: E402
from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402
from train.data_module import GraphDataModule  # noqa: E402

RUN_ARGS = {
    "checkpoint": str(common.CHECKPOINT),
    "config": str(common.CONFIG),
    "overrides": None,            # None: write the default local override file
    "out_dir": str(common.RESULTS / "session1" / "probe_out"),
    "max_batches": None,          # dense per-cell statistics over this many batches (None = all)
    "ablation_batches": 24,       # batches used for the forward ablations
    "attn_map_batches": 4,        # batches used for encoder self-attention maps
    "sensitivity_batches": 4,     # batches used for decoder latent sensitivity
    "seed": 0,
}

LOG2PI = math.log(2 * math.pi)
FLOOR = -5.9


def nll_cells(mu, tgt, logvar, lk, step_mask, *, ar=True):
    """Per-cell Gaussian NLL with the model's cell mask and, optionally, AR(1) innovations.

    Args:
        mu: Forecast mean $(B, A, H, C)$.
        tgt: Target $(B, A, H, C)$.
        logvar: Forecast log-variance $(B, A, H, C)$.
        lk: The model's ``forecast_likelihood_kwargs()``.
        step_mask: Per-step validity $(B, A, H)$.
        ar: Score AR(1) innovations when ``True``, the factorised density when ``False``.

    Returns:
        The unsummed per-cell score $(B, A, H, C)$.
    """
    return raw_sample_score(
        mu, tgt, likelihood="gaussian_nll", logvar=logvar,
        cell_mask=lk["cell_mask"], ar_coef=lk["ar_coef"] if ar else None,
        step_mask=step_mask if ar else None,
    )


def main(args):
    """Run the probe over the shard and write the JSON and NPZ outputs.

    Args:
        args: The run arguments; see :data:`RUN_ARGS`.
    """
    torch.manual_seed(args["seed"])
    out_dir = Path(args["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    overrides = args["overrides"] or str(common.write_overrides())
    config = merge_eval_overrides(load_config(args["config"]), overrides)
    force_single_process_loader(config)
    blob = read_checkpoint(args["checkpoint"])
    task = load_task(args["checkpoint"], device, blob=blob, binding=TRF_CFS_BINDING)
    model = task.orig_model
    model.eval()
    loader = GraphDataModule(config).test_dataloader()

    lk = model.forecast_likelihood_kwargs()
    H, C = int(model.horizon), int(lk["cell_mask"].shape[1])
    dz = int(model.d_z)
    heads = int(model.lag_attn.lag_score_bias.shape[0])
    L = int(model.lag_attn.lag_score_bias.shape[1])
    cell_mask = lk["cell_mask"].double()

    # ---------------------------------------------------------------- static weights
    static = {}
    dec = model.decoder
    if getattr(dec, "persistence_weight", None) is not None:
        static["persistence_weight"] = dec.persistence_weight.detach().cpu().numpy()
    static["ar_phi"] = torch.tanh(model.target_ar_logit).detach().cpu().numpy()
    static["lag_score_bias"] = model.lag_attn.lag_score_bias.detach().cpu().numpy()
    static["lag_embed_norm"] = model.lag_attn.lag_embeddings.detach().norm(dim=-1).cpu().numpy()
    ls = {}
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "LayerScale":
            ls[name] = float(mod.weight.detach().abs().mean())
    static["layer_scale_mean_abs"] = ls
    static["channel_weight"] = (
        None if getattr(model, "target_channel_weight", None) is None
        else model.target_channel_weight.detach().cpu().numpy()
    )
    static["cell_mask"] = lk["cell_mask"].detach().cpu().numpy()
    gate_lin = None
    for name, mod in model.posterior_head.named_modules():
        if "gate" in name and isinstance(mod, torch.nn.Linear):
            gate_lin = (name, float(mod.bias.detach().mean()) if mod.bias is not None else None,
                        float(mod.weight.detach().norm()))
    static["event_gate_linear"] = gate_lin
    static["n_params"] = int(sum(p.numel() for p in model.parameters()))
    static["module_params"] = {
        n: int(sum(p.numel() for p in m.parameters()))
        for n, m in model.named_children()
    }
    print("decoder:", dec)

    # ---------------------------------------------------------------- hooks on the target encoder
    enc = model.target_encoder
    hook_acc = {}

    def add(key, value):
        hook_acc[key] = hook_acc.get(key, 0.0) + float(value)

    def rms(x):
        return x.pow(2).mean(-1).sqrt()  # (B, T)

    anchor_sl = slice(int(model.warmup_period), int(model.anchor_ceiling))

    def make_pre(name):
        def pre(mod, inp):
            x = inp[0][:, anchor_sl]
            add(f"{name}/in_rms", rms(x).mean())
            add(f"{name}/n", 1)
        return pre

    def make_post(name):
        def post(mod, inp, out):
            o = out[:, anchor_sl]
            add(f"{name}/out_rms", rms(o).mean())
        return post

    handles = []
    for i, blk in enumerate(enc.conv_blocks):
        handles.append(blk.register_forward_pre_hook(make_pre(f"conv{i}")))
        handles.append(blk.register_forward_hook(make_post(f"conv{i}")))
        handles.append(blk.layer_scale.register_forward_hook(make_post(f"conv{i}/branch")))
    for i, blk in enumerate(enc.attention_blocks):
        handles.append(blk.register_forward_pre_hook(make_pre(f"attn{i}")))
        handles.append(blk.register_forward_hook(make_post(f"attn{i}")))
        handles.append(blk.attn_scale.register_forward_hook(make_post(f"attn{i}/attn_branch")))
        handles.append(blk.ffn_scale.register_forward_hook(make_post(f"attn{i}/ffn_branch")))
    handles.append(model.target_adapter.register_forward_hook(make_post("target_adapter")))
    handles.append(model.source_adapter.register_forward_hook(make_post("source_adapter")))

    # self-attention maps: entropy and mean look-back distance per block and head
    attn_maps = {}
    attn_state = {"on": False}

    def make_attn_pre(i):
        def pre(mod, inp):
            if not attn_state["on"]:
                return
            x = inp[0]
            q, k, _ = mod._project(mod.norm(x))
            T = q.shape[-2]
            s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
            s = s.masked_fill(~causal, float("-inf"))
            p = s.softmax(-1)[:, :, anchor_sl]  # (B, h, A, T)
            ent = -(p.clamp_min(1e-12).log() * p).sum(-1)
            t = torch.arange(T, device=x.device)[anchor_sl]
            dist = (p * (t[:, None] - torch.arange(T, device=x.device)[None, :]).clamp_min(0)).sum(-1)
            self_w = p.gather(-1, t[None, None, :, None].expand(p.shape[0], p.shape[1], -1, 1))[..., 0]
            within10 = torch.stack(
                [p[:, :, a, max(0, int(tt) - 9): int(tt) + 1].sum(-1) for a, tt in enumerate(t)], -1)
            rec = attn_maps.setdefault(i, {"ent": 0, "dist": 0, "self": 0, "local10": 0, "n": 0,
                                           "profile": torch.zeros(T, device=x.device, dtype=torch.float64)})
            rec["ent"] = rec["ent"] + ent.mean((0, 2)).double()
            rec["dist"] = rec["dist"] + dist.mean((0, 2)).double()
            rec["self"] = rec["self"] + self_w.mean((0, 2)).double()
            rec["local10"] = rec["local10"] + within10.mean((0, 2)).double()
            # mean attention as a function of look-back distance, averaged over anchors and heads
            back = torch.zeros(T, device=x.device, dtype=torch.float64)
            for a, tt in enumerate(t):
                row = p[:, :, a, : int(tt) + 1].mean((0, 1)).double().flip(0)
                back[: row.shape[0]] += row
            rec["profile"] += back / len(t)
            rec["n"] += 1
        return pre

    for i, blk in enumerate(enc.attention_blocks):
        handles.append(blk.attn.register_forward_pre_hook(make_attn_pre(i)))

    # ---------------------------------------------------------------- accumulators
    dd = dict(device=device, dtype=torch.float64)
    Z = lambda *s: torch.zeros(*s, **dd)  # noqa: E731
    acc = {k: Z(H, C) for k in [
        "cnt", "se_full", "se_base", "se_pers", "se_clim", "se_ppath", "se_fullmean",
        "nll_full", "nll_base", "nll_full_fact", "nll_base_fact", "lv_full", "floor_full",
        "z2_full", "cov90_full", "z2_base", "tsum", "tsq", "abs_full", "cov90_base",
        "se_full_floor", "cnt_floor"]}
    lat = {k: Z(dz) for k in ["n", "mup_s", "mup_ss", "lvp_s", "kl_s", "dmu_ss", "lvq_s",
                               "mupost_s", "mupost_ss"]}
    att = {"prof": Z(heads, L), "ent": Z(heads), "zero": Z(heads), "argmax": Z(heads, L),
           "n": Z(1), "kl_head": Z(heads), "prof_hikl": Z(heads, L), "n_hikl": Z(1)}
    per_anchor = {k: [] for k in ["gap", "gap_mean", "kl", "gate", "open", "nll_full", "nll_base",
                                   "cov", "anchor", "sample"]}
    abl = {}
    sens = {"jac": Z(dz), "n": 0}
    hy_feats, hu_feats, z_feats = [], [], []
    sample_offset = 0
    t0 = time.time()

    def score_branch(mu, lv, tgt, smask, M):
        nll = nll_cells(mu, tgt, lv, lk, smask) * M
        return nll.sum((-1, -2))  # (B, A)

    for bi, batch in enumerate(loader):
        if args["max_batches"] is not None and bi >= args["max_batches"]:
            break
        batch = task.transfer_batch_to_device(batch, device, 0)
        with torch.no_grad():
            attn_state["on"] = bi < args["attn_map_batches"]
            inputs = task._build_forward_inputs(batch)
            target_features, weight = task._build_raw_target(batch)
            weight = task._target_quality_weight(batch, weight)
            out = model(*inputs)
            attn_state["on"] = False
            w = model.scored_weight(weight)
            anchors, av = out["anchor_index"], out["anchor_valid"]
            tgt = model._build_forecast_target(target_features, anchors)
            smask, cov = forecast_mask(w, model.geometry, coverage_floor=model.coverage_floor,
                                       anchors=anchors, anchor_valid=av)
            smask = smask.to(tgt.dtype)
            M = smask[..., None] * lk["cell_mask"]  # (B, A, H, C)
            Md = M.double()
            B, A = anchors.shape

            mu_f, lv_f, mu_b, lv_b = out["mu_full"], out["logvar_full"], out["mu_base"], out["logvar_base"]
            pers = out["persistence"]  # (B, A, C)
            ppath = dec.persistence_weight * pers[..., None, :]
            # full branch at the posterior mean (no sampling noise)
            gi = anchors[:, :, None].expand(-1, -1, dz)
            mu_fm, lv_fm = dec(out["mu_post"].gather(1, gi), persistence=pers)

            nf = nll_cells(mu_f, tgt, lv_f, lk, smask) * M
            nb = nll_cells(mu_b, tgt, lv_b, lk, smask) * M
            nfm = (nll_cells(mu_fm, tgt, lv_fm, lk, smask) * M).sum((-1, -2))
            nff = nll_cells(mu_f, tgt, lv_f, lk, smask, ar=False) * M
            nbf = nll_cells(mu_b, tgt, lv_b, lk, smask, ar=False) * M
            ef, lef = ar_innovation(tgt - mu_f, lk["ar_coef"], smask, lv_f)
            eb, leb = ar_innovation(tgt - mu_b, lk["ar_coef"], smask, lv_b)
            zf = ef * torch.exp(-0.5 * lef)
            zb = eb * torch.exp(-0.5 * leb)

            acc["cnt"] += Md.sum((0, 1))
            acc["se_full"] += (((tgt - mu_f) ** 2) * M).double().sum((0, 1))
            acc["se_fullmean"] += (((tgt - mu_fm) ** 2) * M).double().sum((0, 1))
            acc["se_base"] += (((tgt - mu_b) ** 2) * M).double().sum((0, 1))
            acc["se_pers"] += (((tgt - pers[..., None, :]) ** 2) * M).double().sum((0, 1))
            acc["se_clim"] += ((tgt ** 2) * M).double().sum((0, 1))
            acc["se_ppath"] += (((tgt - ppath) ** 2) * M).double().sum((0, 1))
            acc["abs_full"] += ((tgt - mu_f).abs() * M).double().sum((0, 1))
            acc["nll_full"] += nf.double().sum((0, 1))
            acc["nll_base"] += nb.double().sum((0, 1))
            acc["nll_full_fact"] += nff.double().sum((0, 1))
            acc["nll_base_fact"] += nbf.double().sum((0, 1))
            acc["lv_full"] += (lv_f * M).double().sum((0, 1))
            fl = (lv_f <= FLOOR).to(M.dtype) * M
            acc["floor_full"] += fl.double().sum((0, 1))
            acc["se_full_floor"] += (((tgt - mu_f) ** 2) * fl).double().sum((0, 1))
            acc["cnt_floor"] += fl.double().sum((0, 1))
            acc["z2_full"] += ((zf ** 2) * M).double().sum((0, 1))
            acc["z2_base"] += ((zb ** 2) * M).double().sum((0, 1))
            acc["cov90_full"] += ((zf.abs() < 1.6449).to(M.dtype) * M).double().sum((0, 1))
            acc["cov90_base"] += ((zb.abs() < 1.6449).to(M.dtype) * M).double().sum((0, 1))
            acc["tsum"] += (tgt * M).double().sum((0, 1))
            acc["tsq"] += ((tgt ** 2) * M).double().sum((0, 1))

            # per-anchor records (only anchors that contribute)
            contrib = (smask.sum(-1) > 0) & av
            kl_t = out["kld_per_t"]
            kl_a = kl_t.gather(1, anchors)
            gate_a = out["event_gate"][..., 0].gather(1, anchors)
            open_a = out["source_open"].gather(1, anchors).float()
            nfa, nba = nf.sum((-1, -2)), nb.sum((-1, -2))
            sel = contrib
            per_anchor["gap"].append((nba - nfa)[sel].cpu())
            per_anchor["gap_mean"].append((nba - nfm)[sel].cpu())
            per_anchor["kl"].append(kl_a[sel].cpu())
            per_anchor["gate"].append(gate_a[sel].cpu())
            per_anchor["open"].append(open_a[sel].cpu())
            per_anchor["nll_full"].append(nfa[sel].cpu())
            per_anchor["nll_base"].append(nba[sel].cpu())
            per_anchor["cov"].append(cov[sel].cpu())
            per_anchor["anchor"].append(anchors[sel].cpu())
            per_anchor["sample"].append(
                (torch.arange(B, device=device)[:, None].expand(B, A) + sample_offset)[sel].cpu())
            sample_offset += B

            # latent statistics over contributing anchors
            def at(x):
                return x.gather(1, anchors[:, :, None].expand(-1, -1, x.shape[-1]))[sel]
            mup, lvp = at(out["mu_prior"]), at(out["logvar_prior"])
            mq, lvq = at(out["mu_post"]), at(out["logvar_post"])
            kl_btd = model.kld_tensor(mu_prior=out["mu_prior"], logvar_prior=out["logvar_prior"],
                                      mu_post=out["mu_post"], logvar_post=out["logvar_post"])
            kld = at(kl_btd)
            lat["n"] += mup.shape[0]
            lat["mup_s"] += mup.double().sum(0)
            lat["mup_ss"] += (mup.double() ** 2).sum(0)
            lat["lvp_s"] += lvp.double().sum(0)
            lat["kl_s"] += kld.double().sum(0)
            lat["dmu_ss"] += ((mq - mup).double() ** 2).sum(0)
            lat["lvq_s"] += lvq.double().sum(0)
            lat["mupost_s"] += mq.double().sum(0)
            lat["mupost_ss"] += (mq.double() ** 2).sum(0)
            if bi < 30:
                z_feats.append(mup[::7].cpu())
                hy_feats.append(at(out["target_state"])[::7].cpu())
                hu_feats.append(at(out["source_state"])[::7].cpu())

            # lag attention at contributing, source-open anchors
            alpha = out["attn_weights"]  # (B, T, h, L)
            al = alpha.gather(1, anchors[:, :, None, None].expand(-1, -1, heads, L))[sel]
            op = open_a[sel] > 0
            al = al[op]
            att["prof"] += al.double().sum(0)
            att["ent"] += (-(al.clamp_min(1e-12).log() * al).sum(-1)).double().sum(0)
            att["zero"] += (al <= 0).double().mean(-1).sum(0)
            am = al.argmax(-1)
            att["argmax"] += F.one_hot(am, L).double().sum(0)
            att["n"] += al.shape[0]
            klh = out["kld_per_t_per_head"].gather(
                1, anchors[:, :, None].expand(-1, -1, heads))[sel][op]
            att["kl_head"] += klh.double().sum(0)
            hi = kl_a[sel][op] > 5.0
            att["prof_hikl"] += al[hi].double().sum(0)
            att["n_hikl"] += hi.sum()

            # persistence-fitted reference is computed after the loop from se_pers

            # ------------------------------------------------ ablations
            if bi < args["ablation_batches"]:
                def run(kw):
                    inp = list(inputs)
                    for idx, val in kw.items():
                        inp[idx] = val
                    o = model(*inp)
                    return (score_branch(o["mu_full"], o["logvar_full"], tgt, smask, M)[sel].sum().item(),
                            score_branch(o["mu_base"], o["logvar_base"], tgt, smask, M)[sel].sum().item(),
                            o["kld_per_t"].gather(1, anchors)[sel].sum().item())
                n_sel = float(sel.sum())
                ref = (nfa[sel].sum().item(), nba[sel].sum().item(), kl_a[sel].sum().item())
                y_st, y_ph, u, ph, st, sv = inputs
                rows = {
                    "reference": ref,
                    "full_at_post_mean": (nfm[sel].sum().item(), ref[1], ref[2]),
                    "source_closed": run({5: torch.zeros_like(sv)}),
                    "source_shuffled": run({2: u.roll(1, 0), 5: sv.roll(1, 0)}),
                    "source_zero_values": run({2: torch.zeros_like(u)}),
                    "source_shift_+15steps": run({2: u.roll(15, 1), 5: sv.roll(15, 1)}),
                    "source_shift_-15steps(future)": run({2: u.roll(-15, 1), 5: sv.roll(-15, 1)}),
                }
                # decoder-only ablations at the base latent
                zp = out["z_prior"].gather(1, gi)
                zmean = (lat["mup_s"] / lat["n"]).to(zp.dtype)
                for name, (zz, pp) in {
                    "decode_prior_mean(ref base)": (zp, pers),
                    "decode_latent_const_mean": (zmean.expand_as(zp), pers),
                    "decode_latent_shuffled": (zp.roll(1, 0), pers),
                    "decode_no_persistence": (zp, torch.zeros_like(pers)),
                    "decode_latent_const_no_persistence": (zmean.expand_as(zp), torch.zeros_like(pers)),
                }.items():
                    m_, l_ = dec(zz, persistence=pp)
                    rows[name] = (score_branch(m_, l_, tgt, smask, M)[sel].sum().item(), float("nan"),
                                  float("nan"))
                for k, v in rows.items():
                    a = abl.setdefault(k, [0.0, 0.0, 0.0, 0.0])
                    a[0] += v[0]; a[1] += v[1]; a[2] += v[2]; a[3] += n_sel

            # ------------------------------------------------ decoder latent sensitivity
            if bi < args["sensitivity_batches"]:
                zp = out["mu_prior"].gather(1, gi)[sel]  # (N, dz)
                m0, _ = dec(zp[:, None], persistence=pers[sel][:, None])
                eps = 0.05
                for d in range(dz):
                    zz = zp.clone()
                    zz[:, d] += eps
                    m1, _ = dec(zz[:, None], persistence=pers[sel][:, None])
                    sens["jac"][d] += ((m1 - m0) / eps).pow(2).mean().double()
                sens["n"] += 1
        if bi % 10 == 0:
            print(f"batch {bi} {time.time() - t0:.1f}s", flush=True)

    for h in handles:
        h.remove()

    # ---------------------------------------------------------------- assemble
    res = {"static": {}, "per_cell": {}, "latent": {}, "lag_attention": {}, "encoder": {},
           "ablations": {}, "per_anchor_summary": {}}
    for k, v in static.items():
        res["static"][k] = v.tolist() if isinstance(v, np.ndarray) else v
    cell = {k: v.cpu().numpy() for k, v in acc.items()}
    np.savez(out_dir / "probe_cells.npz", **cell)
    n_samples = sample_offset
    cnt = cell["cnt"]
    res["meta"] = {"samples": n_samples, "H": H, "C": C, "dz": dz, "heads": heads, "L": L,
                   "scored_cells": float(cnt.sum())}

    lt = {k: v.cpu().numpy() for k, v in lat.items()}
    n = lt["n"][0]
    mean = lt["mup_s"] / n
    res["latent"] = {
        "prior_mu_var_across_anchors": (lt["mup_ss"] / n - mean ** 2).tolist(),
        "prior_var_mean": np.exp(lt["lvp_s"] / n).tolist(),
        "prior_logvar_mean": (lt["lvp_s"] / n).tolist(),
        "kl_per_dim": (lt["kl_s"] / n).tolist(),
        "delta_mu_rms": np.sqrt(lt["dmu_ss"] / n).tolist(),
        "post_logvar_mean": (lt["lvq_s"] / n).tolist(),
        "decoder_jacobian_rms": (np.sqrt(sens["jac"].cpu().numpy() / max(sens["n"], 1))).tolist(),
    }
    at_ = {k: v.cpu().numpy() for k, v in att.items()}
    na = at_["n"][0]
    res["lag_attention"] = {
        "mean_profile": (at_["prof"] / na).tolist(),
        "mean_entropy": (at_["ent"] / na).tolist(),
        "zero_frac": (at_["zero"] / na).tolist(),
        "argmax_hist": (at_["argmax"] / na).tolist(),
        "kl_per_head": (at_["kl_head"] / na).tolist(),
        "profile_high_kl": (at_["prof_hikl"] / max(at_["n_hikl"][0], 1)).tolist(),
        "n_high_kl": float(at_["n_hikl"][0]),
        "n_open": float(na),
    }
    enc_rec = {}
    for key, val in hook_acc.items():
        enc_rec[key] = val
    nb_ = {}
    for key in list(enc_rec):
        if key.endswith("/n"):
            nb_[key[:-2]] = enc_rec[key]
    enc_out = {}
    for key, val in enc_rec.items():
        if key.endswith("/n"):
            continue
        base = key.split("/")[0]
        denom = nb_.get(base) or nb_.get(key.rsplit("/", 1)[0]) or (bi + 1)
        enc_out[key] = val / denom
    res["encoder"]["rms"] = enc_out
    res["encoder"]["self_attention"] = {
        str(i): {k: (v / r["n"]).tolist() if k != "n" else r["n"]
                 for k, v in r.items() if k not in ("profile",)}
        for i, r in attn_maps.items()
    }
    res["encoder"]["self_attention_backprofile"] = {
        str(i): (r["profile"] / r["n"]).cpu().numpy()[:140].tolist() for i, r in attn_maps.items()}

    def eff_rank(feats):
        X = torch.cat(feats).double()
        X = X - X.mean(0)
        s = torch.linalg.svdvals(X)
        p = (s ** 2) / (s ** 2).sum()
        return {"participation_ratio": float((s ** 2).sum() ** 2 / (s ** 4).sum()),
                "entropy_rank": float(torch.exp(-(p * p.clamp_min(1e-20).log()).sum())),
                "top10_var": p[:10].tolist(), "n": int(X.shape[0])}
    res["encoder"]["h_y_rank"] = eff_rank(hy_feats)
    res["encoder"]["h_u_rank"] = eff_rank(hu_feats)
    res["latent"]["mu_prior_rank"] = eff_rank(z_feats)

    for k, (f, b, kl, ns) in abl.items():
        res["ablations"][k] = {"nll_full_per_anchor": f / ns, "nll_base_per_anchor": b / ns,
                               "kl_per_anchor": kl / ns, "anchors": ns}
    pa = {k: torch.cat(v).numpy() for k, v in per_anchor.items()}
    np.savez(out_dir / "probe_anchors.npz", **pa)
    res["per_anchor_summary"] = {
        "anchors": int(pa["gap"].shape[0]),
        "nll_full_mean": float(pa["nll_full"].mean()), "nll_base_mean": float(pa["nll_base"].mean()),
        "gap_mean": float(pa["gap"].mean()), "gap_at_post_mean": float(pa["gap_mean"].mean()),
        "gap_q": np.quantile(pa["gap"], [0.01, 0.1, 0.5, 0.9, 0.99]).tolist(),
        "frac_gap_positive": float((pa["gap"] > 0).mean()),
        "kl_mean": float(pa["kl"].mean()),
        "kl_q": np.quantile(pa["kl"], [0.1, 0.5, 0.9, 0.99]).tolist(),
        "gate_mean": float(pa["gate"].mean()),
        "gate_q": np.quantile(pa["gate"], [0.1, 0.5, 0.9]).tolist(),
        "open_frac": float(pa["open"].mean()),
        "corr_kl_gap": float(np.corrcoef(pa["kl"], pa["gap"])[0, 1]),
        "corr_kl_gapmean": float(np.corrcoef(pa["kl"], pa["gap_mean"])[0, 1]),
    }
    with open(out_dir / "probe.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main(RUN_ARGS)
