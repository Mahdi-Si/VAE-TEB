"""Train the coupling model on the 4-s step cache (see data.py). One GPU, data held in memory.

    python -m teb_vae.coupling_irf.train --cache tmp/coupling_irf/steps_train.npz --out tmp/coupling_irf/runs/v2

One row of a batch is one recording (all its segments on one timeline). Each batch is used twice:
  own    the recording's own UA        -> the UA response may lower the forecast loss
  wrong  another recording's UA        -> any response it adds raises the loss
Validation also plants a synthetic response to the wrong UA at a known delay and reads it back.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import guid_folds, load_steps, stitch
from .model import CouplingIRF, cell_masks, irf_delay, plant

FIELDS = ("fhr", "fhr_sd", "fhr_ok", "ua", "ua_ok")


def to_device(rec, device, keep=None):
    keep = slice(None) if keep is None else keep
    t = {k: torch.as_tensor(rec[k][keep], device=device) for k in FIELDS}
    t["length"] = torch.as_tensor(rec["length"][keep], device=device)
    return t


def take(data, idx):
    """Rows idx of every array, cut to the longest recording among them."""
    L = int(data["length"][idx].max())
    return {k: (v[idx][:, :L] if k in FIELDS else v[idx]) for k, v in data.items()}


def batches(length, size, generator=None):
    """Batches of recordings of similar length (so little padding is computed); batch order shuffled."""
    order = torch.argsort(length.cpu() + (torch.rand(len(length), generator=generator) * 200 if generator else 0))
    out = [order[i:i + size] for i in range(0, len(order), size)]
    return [out[i] for i in torch.randperm(len(out), generator=generator)] if generator else out


def full_loss(model, st, out):
    """Mean NLL of the full forecast over the scored cells of both modes; also the gain in each mode."""
    nll0, total, gain = model.nll(st).detach(), 0.0, {}
    for mode, m in cell_masks(st, out).items():
        nll = model.nll(st, out["d_" + mode])
        n = m.sum().clamp_min(1)
        total = total + (nll * m).sum() / n
        gain[mode] = (((nll0 - nll) * m).sum() / n).item()
    return total, gain


def wrong_ua(b):
    """Another recording's UA for every row (the batch rolled by one)."""
    return b["ua"].roll(1, 0), b["ua_ok"].roll(1, 0)


@torch.no_grad()
def validate(model, data, batch=16, seed=0):
    """Held-out recordings: base loss, gain with own and wrong UA, and planted delays read back."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    acc = {}
    for idx in batches(data["length"], batch):
        if len(idx) < 2:
            continue
        b = take(data, idx.to(data["length"].device))
        st, own = model(b)
        ua_w, ok_w = wrong_ua(b)
        wrong = model.ua_side(st, ua_w, ok_w)
        m0 = st["mask"]
        row = {"nll_base": ((model.nll(st) * m0).sum() / m0.sum().clamp_min(1)).item()}
        for name, s, o in (("own", st, own), ("wrong", st, wrong)):
            row.update({f"gain_{mode}_{name}": v for mode, v in full_loss(model, s, o)[1].items()})
            row[f"a_{name}"] = o["a"][o["ua_ok"]].mean().item()
            row[f"open_{name}"] = (o["a"][o["ua_ok"]] > 0).float().mean().item()
        u_w = model.ua_features(ua_w, ok_w)["u"]
        for share in (1.0, 0.3):                              # a response to every contraction / to 30 % of them
            delay = 2 + 22 * torch.rand(len(idx), generator=gen).to(u_w.device)
            fhr_p = plant(b["fhr"], u_w, delay, sigma=2.0, depth=25.0, share=share, generator=gen)
            st_p = model.fhr_side(fhr_p, b["fhr_sd"], b["fhr_ok"])
            pl = model.ua_side(st_p, ua_w, ok_w)
            read, _ = irf_delay(pl["h"])
            err = (read - delay[:, None]).abs()[pl["ua_ok"] & (pl["a"] > 0)]
            err = err[torch.isfinite(err)]
            tag = f"plant{int(share * 100)}"
            row[f"{tag}_err_s"] = 4 * err.median().item() if len(err) else float("nan")
            row[f"{tag}_within4s"] = (err <= 1).float().mean().item() if len(err) else float("nan")
            row[f"{tag}_open"] = (pl["a"][pl["ua_ok"]] > 0).float().mean().item()
            row[f"{tag}_gain_expl"] = full_loss(model, st_p, pl)[1]["expl"]
        for k, v in row.items():
            if v == v:                                        # skip NaN (no readable delay in this batch)
                s, w = acc.get(k, (0.0, 0))
                acc[k] = (s + v * len(idx), w + len(idx))
    model.train()
    return {k: s / w for k, (s, w) in acc.items()}


@torch.no_grad()
def calibrate_k(model, data, false_open=0.05, n=480, batch=12, seed=0):
    """Set k so that, with another recording's UA, the gate opens at `false_open` of the valid anchors.
    The gate is open where slope / se > k, so k is a quantile of that ratio under wrong UA."""
    model.eval()
    pick = torch.randperm(len(data["length"]), generator=torch.Generator().manual_seed(seed))[:n].to(data["length"].device)
    sub = {k: v[pick] for k, v in data.items()}
    z = []
    for idx in batches(sub["length"], batch):
        if len(idx) < 2:
            continue
        b = take(sub, idx.to(pick.device))
        o = model.ua_side(model.fhr_side(b["fhr"], b["fhr_sd"], b["fhr_ok"]), *wrong_ua(b))
        z.append((o["slope"] / o["se"])[o["ua_ok"]][::4].cpu())
    k = torch.quantile(torch.cat(z), 1 - false_open).clamp_min(0.1).item()
    model.k_raw.fill_(math.log(math.expm1(k)))
    model.train()
    return k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="tmp/coupling_irf/steps_train.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--w-wrong", type=float, default=1.0)
    p.add_argument("--context-steps", type=int, default=900)
    p.add_argument("--val-fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--false-open", type=float, default=0.05, help="share of anchors where a wrong UA may open the gate")
    p.add_argument("--calibrate-only", action="store_true", help="load <out>/model.pt and only recalibrate k")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rec, _ = stitch(load_steps(args.cache))
    fold = guid_folds(rec["guid"])
    tr, va = to_device(rec, device, fold != args.val_fold), to_device(rec, device, fold == args.val_fold)
    kwargs = dict(context_steps=args.context_steps)
    model = CouplingIRF(**kwargs).to(device)
    scalars = [model.log_lambda, model.k_raw, model.rho_logit, model.eta_logit]
    rest = [q for q in model.parameters() if all(q is not s for s in scalars)]
    opt = torch.optim.AdamW([{"params": rest, "lr": args.lr, "weight_decay": 1e-4},
                             {"params": scalars, "lr": 10 * args.lr, "weight_decay": 0.0}])
    gen = torch.Generator().manual_seed(args.seed)
    steps = args.epochs * len(batches(tr["length"], args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, 10 * args.lr], total_steps=steps, pct_start=0.1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    if args.calibrate_only:
        ckpt = torch.load(out / "model.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        log, args.epochs = ckpt["log"], 0
    for epoch in range(args.epochs):
        t0 = time.time()
        for idx in batches(tr["length"], args.batch, gen):
            if len(idx) < 2:
                continue
            b = take(tr, idx.to(device))
            st, own = model(b)
            wrong = model.ua_side(st, *wrong_ua(b))
            m0 = st["mask"]
            loss = (model.nll(st) * m0).sum() / m0.sum().clamp_min(1) \
                + full_loss(model, st, own)[0] + args.w_wrong * full_loss(model, st, wrong)[0]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        row = {"epoch": epoch, "seconds": round(time.time() - t0, 1), **validate(model, va),
               "lambda_ridge": model.log_lambda[0].exp().item(), "lambda_smooth": model.log_lambda[1].exp().item(),
               "k": torch.nn.functional.softplus(model.k_raw).item()}
        log.append(row)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        torch.save({"model": model.state_dict(), "kwargs": kwargs, "args": vars(args), "log": log}, out / "model.pt")
    # k as trained balances own gain against wrong-UA loss; fix it instead by a false-opening rate
    k = calibrate_k(model, tr, args.false_open)
    row = {"epoch": "calibrated", "k": k, "false_open": args.false_open, **validate(model, va)}
    log.append(row)
    print(json.dumps({k_: (round(v, 5) if isinstance(v, float) else v) for k_, v in row.items()}), flush=True)
    torch.save({"model": model.state_dict(), "kwargs": kwargs, "args": vars(args), "log": log}, out / "model.pt")
    (out / "log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
