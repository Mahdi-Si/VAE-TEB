r"""Round 4: undo the shared-clamp side effect on the latent variances, then match q's variance to p's.

Originally ``surgery4.py``. Input: the original and the round-3 checkpoints. Output: v4.

**Why this round exists.** ``logvar_clamp`` is one constructor keyword. The decoder and both latent
heads read it. Rounds 1 to 3 lowered the decoder floor from $-6$ to $-11$ and saved the new value,
so on reload every latent log-variance $\ell = lo + (hi - lo)\,\sigma(r)$ was re-mapped as well.
The mean KL rose from about 1.8 to 10.5 nats per anchor.

**Step 1: latent affine.** Per latent dimension $d$, fit $r' = a_d r + b_d$ so that
$-11 + 14\,\sigma(r') \approx -6 + 9\,\sigma(r)$. The fit is ordinary least squares on the target
$r'^{*} = \operatorname{logit}((\ell_{\mathrm{old}} + 11)/14)$, pooled over the prior's raw values
and the posterior's gate-mixed raw values of the original model. One map serves both heads,
because it commutes with the gate mix:
$$g\,(a r^q + b) + (1 - g)\,(a r^p + b) = a\,(g r^q + (1 - g)\,r^p) + b.$$
The map folds exactly into the last linear layers of both heads.

**Step 2: posterior variance match.** Shift the posterior head's raw log-variance by $c_d$; after
the mix the effective shift is $g\,c_d$. The shift is chosen per dimension by grid search to
minimise the mean KL on cached latents. Because $\mathrm{KL}(q \Vert p)$ is minimised over
$\sigma_q^2$ at $\sigma_q^2 = \sigma_p^2$, the full branch then gets the same latent spread under
Monte Carlo scoring that the prior gives the base branch. Step 2 is kept only when the test-half
MC gap improves and the full MC score does not get worse by more than 0.5 nats.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from teb_vae.lag_attn_rws.nets.losses import raw_sample_score  # noqa: E402
from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.surgery import round1_closed_form as S  # noqa: E402

RUN_ARGS = {
    "config": common.CONFIG,
    "overrides": None,
    "original": common.CHECKPOINT,
    "checkpoint": common.checkpoint_variant("v3"),
    "out_ckpt": common.checkpoint_variant("v4"),
    "out_dir": common.RESULTS / "surgery" / "round4",
    "every": 3,          # keep every third anchor in the latent cache
    "mc_k": 8,           # latent draws per anchor for the MC predictive score
    "eval_batches": None,
}


@torch.no_grad()
def latent_cache(task, cfg, dev, every, codes=(0,)):
    """Cache the raw latent log-variances, the means, the gate and the KL at anchors.

    Args:
        task: The loaded task.
        cfg: The merged configuration.
        dev: Device.
        every: Keep every ``every``-th anchor.
        codes: Split codes to keep; ``(0,)`` is the fit split.

    Returns:
        ``(D, (lo, hi))``. ``D`` holds the prior raw log-variance ``rp``, the posterior gate-mixed
        raw log-variance ``rmix`` (recovered by the logit of the bounded value), ``mup``, ``muq``,
        the gate ``g`` and the model KL ``kl``. ``(lo, hi)`` is the latent clamp.
    """
    model = task.orig_model
    lo, hi = model.posterior_head.logvar_clamp
    out = {k: [] for k in ("rp", "rmix", "mup", "muq", "g", "kl")}
    for bi, batch in enumerate(common.batches(task, cfg, dev)):
        split = common.split_codes(batch, dev)
        _, o, tgt, sm, anc, av = common.forward_parts(task, model, batch)
        A = anc.shape[1]
        keep = av & (sm.sum(-1) > 0) & torch.isin(split, torch.tensor(codes, device=dev))[:, None]
        keep[:, (torch.arange(A, device=dev) % every) != (bi % every)] = False

        def at(x):
            return x.gather(1, anc[:, :, None].expand(-1, -1, x.shape[-1]))
        x = ((at(o["logvar_post"]) - lo) / (hi - lo)).clamp(1e-6, 1 - 1e-6)
        vals = {"rp": at(o["raw_logvar_prior"]), "rmix": torch.log(x) - torch.log1p(-x),
                "mup": at(o["mu_prior"]), "muq": at(o["mu_post"]), "g": at(o["event_gate"]),
                "kl": at(o["kld_per_t"][..., None])}
        for k, v in vals.items():
            out[k].append(v[keep].cpu())
    return {k: torch.cat(v).to(dev) for k, v in out.items()}, (lo, hi)


def fit_affine(r, old, new):
    r"""Per-dimension OLS of $\operatorname{logit}((\ell_{\mathrm{old}} - lo_n)/\Delta_n)$ on $r$.

    Args:
        r: Raw log-variances of the original model $(N, d_z)$.
        old: The original clamp $(lo, hi)$.
        new: The loaded (shared) clamp $(lo, hi)$.

    Returns:
        ``(a, b, err)``: slope and intercept per dimension, and the absolute log-variance error
        statistics ``max_abs``, ``mean_abs`` and ``q99_abs``.
    """
    lo_o, hi_o = old
    lo_n, hi_n = new
    l_old = lo_o + (hi_o - lo_o) * torch.sigmoid(r)
    x = ((l_old - lo_n) / (hi_n - lo_n)).clamp(1e-6, 1 - 1e-6)
    t = torch.log(x) - torch.log1p(-x)
    rm, tm = r.mean(0), t.mean(0)
    a = ((r - rm) * (t - tm)).mean(0) / ((r - rm) ** 2).mean(0).clamp_min(1e-12)
    b = tm - a * rm
    l_new = lo_n + (hi_n - lo_n) * torch.sigmoid(a * r + b)
    err = (l_new - l_old).abs()
    return a, b, {"max_abs": float(err.max()), "mean_abs": float(err.mean()),
                  "q99_abs": float(err.flatten().kthvalue(int(0.99 * err.numel())).values)}


def fold_latent_affine(model, a, b):
    r"""Fold $r' = a r + b$ into the prior and the posterior log-variance heads.

    The prior head is a ``ResidualMLP`` with output ``body(x) + skip_proj(x)`` and no final
    activation, so both last linear maps are scaled by $a$ and $b$ goes into ``skip_proj.bias``.
    The posterior has one linear head per latent head group of $d_z / n_{\mathrm{heads}}$
    dimensions.

    Args:
        model: The network, edited in place.
        a: Slope per latent dimension.
        b: Intercept per latent dimension.
    """
    with torch.no_grad():
        ph = model.prior_head.logvar_prior_head
        last = ph.body[len(ph.body) - 1]
        for lin in (last, ph.skip_proj):
            lin.weight.mul_(a[:, None])
            lin.bias.mul_(a)
        ph.skip_proj.bias.add_(b)
        heads = model.posterior_head.logvar_post_head
        k = a.numel() // len(heads)
        for i, lin in enumerate(heads):
            sl = slice(i * k, (i + 1) * k)
            lin.weight.mul_(a[sl, None])
            lin.bias.mul_(a[sl]).add_(b[sl])


def kl_with_post_shift(D, c, lo, hi):
    r"""Per-dimension KL when the posterior raw log-variance is shifted by $g\,c$.

    Args:
        D: Latent cache from :func:`latent_cache`.
        c: Shift per dimension, or a scalar.
        lo: Latent floor.
        hi: Latent ceiling.

    Returns:
        The KL $(N, d_z)$.
    """
    span = hi - lo
    lp = lo + span * torch.sigmoid(D["rp"])
    lq = lo + span * torch.sigmoid(D["rmix"] + D["g"] * c)
    return 0.5 * (lp - lq + (torch.exp(lq) + (D["muq"] - D["mup"]) ** 2) * torch.exp(-lp) - 1)


def solve_post_shift(D, lo, hi):
    """Grid search (161 points in $[-4, 4]$) of the KL-optimal posterior shift per dimension.

    Args:
        D: Latent cache.
        lo: Latent floor.
        hi: Latent ceiling.

    Returns:
        The shift per dimension $(d_z,)$.
    """
    grid = torch.linspace(-4, 4, 161, device=D["rp"].device)
    c = torch.zeros(D["rp"].shape[-1], device=grid.device)
    for d in range(c.numel()):
        sub = {k: (v if k == "g" else v[:, d:d + 1]) for k, v in D.items() if k != "kl"}
        vals = torch.stack([kl_with_post_shift(sub, float(x), lo, hi).mean() for x in grid])
        c[d] = grid[int(vals.argmin())]
    return c


def fold_post_shift(model, c):
    """Add the shift to the bias of each posterior log-variance head.

    Args:
        model: The network, edited in place.
        c: Shift per latent dimension.
    """
    with torch.no_grad():
        heads = model.posterior_head.logvar_post_head
        k = c.numel() // len(heads)
        for i, lin in enumerate(heads):
            lin.bias.add_(c[i * k:(i + 1) * k])


@torch.no_grad()
def mc_gap(task, model, cfg, dev, K, limit=None, seed=0):
    r"""Test-half Monte Carlo predictive scores with common random numbers.

    This is the evaluation's estimator: per anchor,
    $$-\log \tfrac1K \sum_{k=1}^{K} \exp(-\mathrm{NLL}(z_k)),$$
    with $z_k = \mu + \sigma \epsilon_k$ drawn from the prior (base) or the posterior (full), and
    the same $\epsilon_k$ for both branches.

    Args:
        task: The loaded task.
        model: The network.
        cfg: The merged configuration.
        dev: Device.
        K: Number of draws.
        limit: Optional number of batches.
        seed: Generator seed.

    Returns:
        Per-anchor means of ``base``, ``full``, ``base_mean``, ``full_mean`` and ``kl``, plus
        ``mc_gap = base - full`` and ``mean_gap = base_mean - full_mean``.
    """
    lk = model.forecast_likelihood_kwargs()
    gen = torch.Generator(device=dev).manual_seed(seed)
    tot = {"base": 0.0, "full": 0.0, "base_mean": 0.0, "full_mean": 0.0, "n": 0.0, "kl": 0.0}
    for batch in common.batches(task, cfg, dev, limit):
        split = common.split_codes(batch, dev)
        _, o, tgt, sm, anc, av = common.forward_parts(task, model, batch)
        M = sm[..., None] * lk["cell_mask"]
        c = (sm.sum(-1) > 0) & av & (split % 2 == 1)[:, None]

        def at(x):
            return x.gather(1, anc[:, :, None].expand(-1, -1, x.shape[-1]))

        def score(z):
            mu, lv = model.decoder(z, persistence=o["persistence"])
            return (raw_sample_score(mu, tgt, likelihood="gaussian_nll", logvar=lv, cell_mask=lk["cell_mask"],
                                     ar_coef=lk["ar_coef"], step_mask=sm) * M).sum((-1, -2))
        mp, lp, mq, lq = at(o["mu_prior"]), at(o["logvar_prior"]), at(o["mu_post"]), at(o["logvar_post"])
        db, df = [], []
        for _ in range(K):
            eps = torch.randn(mp.shape, generator=gen, device=dev)
            db.append(score(mp + torch.exp(0.5 * lp) * eps))
            df.append(score(mq + torch.exp(0.5 * lq) * eps))
        mb = -(torch.logsumexp(-torch.stack(db), 0) - math.log(K))
        mf = -(torch.logsumexp(-torch.stack(df), 0) - math.log(K))
        tot["base"] += float(mb[c].sum())
        tot["full"] += float(mf[c].sum())
        tot["base_mean"] += float(score(mp)[c].sum())
        tot["full_mean"] += float(score(mq)[c].sum())
        tot["kl"] += float(o["kld_per_t"].gather(1, anc)[c].sum())
        tot["n"] += float(c.sum())
    n = tot.pop("n")
    r = {k: v / n for k, v in tot.items()}
    r["mc_gap"] = r["base"] - r["full"]
    r["mean_gap"] = r["base_mean"] - r["full_mean"]
    return r


def main(a):
    """Fit and fold the latent affine, try the posterior shift, evaluate and save v4.

    Args:
        a: The run arguments; see :data:`RUN_ARGS`.
    """
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = {}
    dev, cfg, task0 = S.setup({**a, "checkpoint": a["original"]})
    D0, old = latent_cache(task0, cfg, dev, a["every"])
    log["original_kl_fit"] = float(D0["kl"].mean())
    log["original_mc_limit20"] = mc_gap(task0, task0.orig_model.eval(), cfg, dev, a["mc_k"], limit=20)
    print("original", old, "KL", log["original_kl_fit"], log["original_mc_limit20"], flush=True)
    del task0
    torch.cuda.empty_cache()

    _, _, task = S.setup(a)
    model = task.orig_model.eval()
    new = tuple(model.posterior_head.logvar_clamp)
    assert tuple(model.prior_head.logvar_clamp) == new
    r = torch.cat([D0["rp"], D0["rmix"]])
    A, B, err = fit_affine(r, old, new)
    log["affine"] = {"old": old, "new": new, "err": err, "a": A.tolist(), "b": B.tolist()}
    print("affine fit error", err, flush=True)
    log["v3_mc_limit20"] = mc_gap(task, model, cfg, dev, a["mc_k"], limit=20)
    print("v3", log["v3_mc_limit20"], flush=True)
    fold_latent_affine(model, A, B)
    D1, _ = latent_cache(task, cfg, dev, a["every"])
    log["step1_kl_fit"] = float(D1["kl"].mean())
    log["step1_mc_limit20"] = mc_gap(task, model, cfg, dev, a["mc_k"], limit=20)
    print("step1 KL", log["step1_kl_fit"], log["step1_mc_limit20"], flush=True)

    c = solve_post_shift(D1, *new)
    log["post_shift"] = c.tolist()
    log["post_shift_kl_fit_formula"] = [float(kl_with_post_shift(D1, torch.zeros_like(c), *new).sum(-1).mean()),
                                        float(kl_with_post_shift(D1, c, *new).sum(-1).mean())]
    print("post shift KL formula", log["post_shift_kl_fit_formula"], flush=True)
    step1 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    fold_post_shift(model, c)
    log["step2_mc_limit20"] = mc_gap(task, model, cfg, dev, a["mc_k"], limit=20)
    print("step2", log["step2_mc_limit20"], flush=True)

    s1, s2 = log["step1_mc_limit20"], log["step2_mc_limit20"]
    keep2 = s2["mc_gap"] > s1["mc_gap"] and s2["full"] <= s1["full"] + 0.5
    log["step2_kept"] = bool(keep2)
    if not keep2:
        model.load_state_dict(step1)
    log["test"] = S.evaluate(task, model, cfg, dev, a["eval_batches"])
    log["test_mc"] = mc_gap(task, model, cfg, dev, a["mc_k"], a["eval_batches"])
    print("test", log["test"], log["test_mc"], flush=True)

    S.save_checkpoint(model, a["checkpoint"], a["out_ckpt"], "surgery_v4",
                      {"source": str(a["checkpoint"]), "latent_affine_err": err, "post_shift_kept": bool(keep2)})
    json.dump(log, open(out / "v4.json", "w"), indent=1, default=str)
    print("saved", a["out_ckpt"], f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(RUN_ARGS)
