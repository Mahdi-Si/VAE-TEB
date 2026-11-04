r"""Round 3: Fisher-scoring re-solve of the decoder variance head, on top of the round-2 checkpoint.

Originally ``surgery3.py``.

1. ``logvar_head`` is re-solved per channel $c$ by Fisher scoring (IRLS) on fixed decoder features
   $f$. The raw output is $\rho = W_c f + b_c$ and the log-variance is
   $\ell = lo + (hi - lo)\,\sigma(\rho)$. The objective is the AR(1) innovation NLL
   $\tfrac12 \sum (\ell + e^2 e^{-\ell})$. No layer below the head changes.
2. ``mean_head`` and the persistence weights (GLS), $\phi_c$ (WLS) and the variance head are fitted
   on the base and the posterior-mean features stacked, because both branches use one decoder.
   Each floor $lo \in \{-9, -10, -11, -12\}$ is tried; the select set picks the floor.
3. The KL-optimal shift $b_d$ of the prior's raw log-variance is measured exactly from cached
   latents. In ``independent`` logvar mode the posterior raw log-variance is the mix
   $g\,r^q + (1 - g)\,r^p$, so the shift moves it by $(1 - g)\,b_d$.

The shift lowers the mean KL but makes the sampled base NLL worse by about 98 nats per anchor,
because the prior variance shrinks below the spread the decoder needs. This script therefore
reports the shift and its test effect and does **not** keep it in the saved checkpoint.

The decoder features are cached on the CPU in half precision; only the active set moves to the
GPU. Every sixth anchor is kept to stay inside GPU memory.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.surgery import round1_closed_form as S  # noqa: E402

RUN_ARGS = {
    "config": common.CONFIG,
    "overrides": None,
    "checkpoint": common.checkpoint_variant("neuron"),
    "out_ckpt": common.checkpoint_variant("v3"),
    "out_dir": common.RESULTS / "surgery" / "round3",
    "round1_json": common.RESULTS / "surgery" / "round1" / "surgery.json",
    "anchor_every": 6,
    "floors": [-9.0, -10.0, -11.0, -12.0],
    "irls_iters": 8,
    "irls_ridge": 1e-4,
    "rounds": 2,
    "cache_batches": None,
    "eval_batches": None,
    "keep_prior_shift": False,   # True reproduces the rejected variant (sampled NLL about 98 nats worse)
}


# ------------------------------------------------------------------------------ cache
@torch.no_grad()
def build_cache(task, cfg, dev, every, limit=None):
    """Cache decoder features of both branches and the latent quantities, on the CPU.

    Args:
        task: The loaded task.
        cfg: The merged configuration.
        dev: Device.
        every: Keep every ``every``-th fully observed anchor.
        limit: Optional number of batches.

    Returns:
        ``{"fit": D, "sel": D}``. ``D`` holds base features ``fb`` and posterior-mean features
        ``ff`` (half precision), target ``y``, persistence ``p``, prior mean ``mup``, prior raw
        log-variance ``rlp``, posterior mean ``muq`` and log-variance ``lvq``, event gate ``g`` and
        the model KL ``kl``.
    """
    model = task.orig_model
    dec = model.decoder
    keys = ("fb", "ff", "y", "p", "mup", "rlp", "muq", "lvq", "g", "kl")
    out = {s: {k: [] for k in keys} for s in ("fit", "sel")}
    for bi, batch in enumerate(common.batches(task, cfg, dev, limit)):
        split = common.split_codes(batch, dev)
        _, o, tgt, sm, anc, av = common.forward_parts(task, model, batch)
        A = anc.shape[1]
        keep = av & (sm.amin(-1) > 0)
        keep[:, (torch.arange(A, device=dev) % every) != (bi % every)] = False

        def at(x):
            return x.gather(1, anc[:, :, None].expand(-1, -1, x.shape[-1]))
        fb = dec.core.decode(dec.proj(at(o["mu_prior"])))
        ff = dec.core.decode(dec.proj(at(o["mu_post"])))
        vals = {"fb": fb.half(), "ff": ff.half(), "y": tgt, "p": o["persistence"], "mup": at(o["mu_prior"]),
                "rlp": at(o["raw_logvar_prior"]), "muq": at(o["mu_post"]), "lvq": at(o["logvar_post"]),
                "g": at(o["event_gate"]), "kl": at(o["kld_per_t"][..., None])}
        for name, code in (("fit", 0), ("sel", 2)):
            m = keep & (split[:, None] == code)
            for k, v in vals.items():
                out[name][k].append(v[m].cpu())
        if bi % 20 == 0:
            print(f"cache batch {bi}", flush=True)
    return {s: {k: torch.cat(v) for k, v in d.items()} for s, d in out.items()}


def gpu(D, dev):
    """Move every tensor of a cache dictionary to a device.

    Args:
        D: The cache dictionary.
        dev: Target device.

    Returns:
        A new dictionary on ``dev``.
    """
    return {k: v.to(dev) for k, v in D.items()}


def stacked(D):
    """Stack the base and the posterior-mean rows; one head decodes both.

    Args:
        D: A cache dictionary with ``fb``, ``ff``, ``y`` and ``p``.

    Returns:
        A head dataset ``{"f", "y", "p"}`` with twice the rows.
    """
    return {"f": torch.cat([D["fb"], D["ff"]]), "y": torch.cat([D["y"], D["y"]]),
            "p": torch.cat([D["p"], D["p"]])}


def base_only(D):
    """Select the base rows only.

    Args:
        D: A cache dictionary.

    Returns:
        A head dataset ``{"f", "y", "p"}``.
    """
    return {"f": D["fb"], "y": D["y"], "p": D["p"]}


# ------------------------------------------------------------------------------ head with a free variance layer
class Head3(S.Head):
    r"""The round-1 head with the affine variance map folded into the variance layer.

    ``S.Head`` keeps the variance as $a_c (W_v f + b_v)_c + c_c$. Here $a$ and $c$ are folded into
    $W_v$ and $b_v$ so that IRLS updates all variance weights at once. The class reads the
    features from ``D["f"]`` directly, in chunks, because the cache is in half precision.
    """

    def __init__(self, model):
        """Copy the head tensors and fold the variance affine.

        Args:
            model: The network whose decoder head is copied.
        """
        super().__init__(model)
        self.Wv = self.a[:, None] * self.Wv
        self.bv = self.a * self.bv + self.c
        self.a = torch.ones_like(self.a)
        self.c = torch.zeros_like(self.c)

    def mean(self, D, chunk=4000):
        r"""Forecast mean $W f + b + w \odot y_t$ in float64.

        Args:
            D: A head dataset.
            chunk: Rows per matrix product.

        Returns:
            The mean $(N, H, C)$.
        """
        out = []
        W, b = self.W.float(), self.b.float()
        for s in range(0, D["f"].shape[0], chunk):
            f = D["f"][s:s + chunk].float()
            out.append(f @ W.T + b + self.w.float()[None] * D["p"][s:s + chunk].float()[:, None, :])
        return torch.cat(out).double()

    def raw(self, D, chunk=4000):
        r"""Raw variance output $\rho = W_v f + b_v$.

        Args:
            D: A head dataset.
            chunk: Rows per matrix product.

        Returns:
            $\rho$ $(N, H, C)$ in float32.
        """
        Wv, bv = self.Wv.float(), self.bv.float()
        return torch.cat([D["f"][s:s + chunk].float() @ Wv.T + bv for s in range(0, D["f"].shape[0], chunk)])

    def logvar(self, D):
        r"""Bounded log-variance $lo + (hi - lo)\,\sigma(\rho)$.

        Args:
            D: A head dataset.

        Returns:
            The log-variance $(N, H, C)$ in float64.
        """
        return (self.lo + (self.hi - self.lo) * torch.sigmoid(self.raw(D))).double()


def innov_sq(head, D):
    r"""Squared AR(1) innovations $e_\tau^2$ with $e_1 = r_1$ and $e_\tau = r_\tau - \phi r_{\tau-1}$.

    Args:
        head: The head.
        D: A head dataset.

    Returns:
        $e^2$ $(N, H, C)$ in float32.
    """
    r = D["y"].double() - head.mean(D)
    e = r.clone()
    e[:, 1:] = r[:, 1:] - head.phi * r[:, :-1]
    return (e ** 2).float()


def irls_variance(head, D, cell_mask, iters, ridge, chunk=3000):
    r"""Fisher scoring for the variance weights $(W_{v,c}, b_{v,c})$ of each channel.

    With $s = \sigma(\rho)$ and $\Delta = hi - lo$, the per-cell gradient of the NLL with respect to
    $\rho$ is $\tfrac12 (1 - e^2 e^{-\ell})\,\Delta s (1 - s)$ and the expected Fisher information
    is $\tfrac12 (\Delta s (1 - s))^2$. Each step solves the ridge-damped Fisher system and accepts
    the first step size in $\{1, 0.5, 0.25, 0.1\}$ that lowers the NLL.

    Args:
        head: A :class:`Head3`, edited in place.
        D: A head dataset.
        cell_mask: The model's $(H, C)$ cell mask; only valid horizons of each channel are used.
        iters: Maximum Fisher steps per channel.
        ridge: Damping relative to the mean diagonal of the Fisher matrix.
        chunk: Rows per accumulation step.
    """
    e2 = innov_sq(head, D)
    N, H, F = D["f"].shape
    span = head.hi - head.lo
    for c in range(e2.shape[-1]):
        h = int(cell_mask[:, c].sum())
        theta = torch.cat([head.Wv[c], head.bv[c:c + 1]]).double()

        def nll(th):
            tot = 0.0
            for s in range(0, N, chunk):
                X = D["f"][s:s + chunk, :h].float()
                rho = X @ th[:F].float() + float(th[F])
                lv = head.lo + span * torch.sigmoid(rho)
                tot += float((lv + e2[s:s + chunk, :h, c] * torch.exp(-lv)).double().sum())
            return 0.5 * tot / N

        cur = nll(theta)
        for _ in range(iters):
            G = torch.zeros(F + 1, F + 1, dtype=torch.float64, device=e2.device)
            g = torch.zeros(F + 1, dtype=torch.float64, device=e2.device)
            for s in range(0, N, chunk):
                f = D["f"][s:s + chunk, :h].float().reshape(-1, F)
                X = torch.cat([f, torch.ones(f.shape[0], 1, device=f.device)], 1)
                rho = X @ theta.float()
                sg = torch.sigmoid(rho)
                lv = head.lo + span * sg
                ds = span * sg * (1 - sg)
                u = e2[s:s + chunk, :h, c].reshape(-1) * torch.exp(-lv)
                grad_row = 0.5 * (1 - u) * ds
                fish_row = 0.5 * ds ** 2
                G += (X * fish_row[:, None]).T.double() @ X.double()
                g += (X * grad_row[:, None]).sum(0).double()
            reg = ridge * G.diagonal().mean() * torch.ones(F + 1, dtype=torch.float64, device=G.device)
            step = torch.linalg.solve(G + torch.diag(reg), g)
            for alpha in (1.0, 0.5, 0.25, 0.1):
                cand = theta - alpha * step
                val = nll(cand)
                if val < cur:
                    theta, cur = cand, val
                    break
            else:
                break
        head.Wv[c], head.bv[c] = theta[:F], theta[F]


def coverage(head, D, cell_mask):
    r"""Fraction of valid cells with $|z| < k$ for $k = 1, 2, 3$, where $z = e\,e^{-\ell/2}$.

    Args:
        head: The head.
        D: A head dataset.
        cell_mask: The $(H, C)$ cell mask.

    Returns:
        ``{"cov1", "cov2", "cov3"}``; the Gaussian values are 0.683, 0.954 and 0.997.
    """
    e2 = innov_sq(head, D).double()
    lv = head.logvar(D)
    z2 = e2 * torch.exp(-lv)
    m = cell_mask.expand_as(z2) > 0
    return {f"cov{k}": float((z2[m] < k * k).double().mean()) for k in (1, 2, 3)}


def fit_heads(model, Fs, cm, lo, lam, a):
    """Alternate IRLS variance, GLS mean and WLS $\\phi$ solves at one floor.

    Args:
        model: The network that provides the starting head.
        Fs: The fit head dataset.
        cm: The cell mask.
        lo: The decoder log-variance floor.
        lam: The relative ridge chosen in round 1.
        a: The run arguments.

    Returns:
        The fitted :class:`Head3`.
    """
    h = Head3(model)
    h.lo = lo
    irls_variance(h, Fs, cm, a["irls_iters"], a["irls_ridge"])
    for _ in range(a["rounds"]):
        S.solve_mean(h, Fs, cm, lam)
        S.solve_phi(h, Fs, cm)
        irls_variance(h, Fs, cm, a["irls_iters"], a["irls_ridge"])
    return h


# ------------------------------------------------------------------------------ prior variance vs KL
def latent_kl(D, b, lo, hi):
    r"""Per-dimension KL after the prior's raw log-variance is shifted by $b$.

    The prior log-variance becomes $\ell^p = lo + \Delta\,\sigma(r^p + b)$. The posterior raw
    log-variance is recovered from the cached bounded value by the logit and moved by
    $(1 - g)\,b$. The Gaussian KL is
    $$\mathrm{KL} = \tfrac12\left(\ell^p - \ell^q + (e^{\ell^q} + (\mu^q - \mu^p)^2)\,e^{-\ell^p} - 1\right).$$

    Args:
        D: Latent cache with ``mup``, ``rlp``, ``muq``, ``lvq`` and ``g``.
        b: Shift per latent dimension, or a scalar.
        lo: Latent log-variance floor.
        hi: Latent log-variance ceiling.

    Returns:
        The KL $(N, d_z)$.
    """
    span = hi - lo
    lp = lo + span * torch.sigmoid(D["rlp"] + b)
    x = ((D["lvq"] - lo) / span).clamp(1e-6, 1 - 1e-6)
    rmix = torch.log(x) - torch.log1p(-x) + (1 - D["g"]) * b
    lq = lo + span * torch.sigmoid(rmix)
    return 0.5 * (lp - lq + (torch.exp(lq) + (D["muq"] - D["mup"]) ** 2) * torch.exp(-lp) - 1)


def solve_prior_shift(D, lo, hi, grid=None):
    """Grid search of the KL-optimal shift per latent dimension.

    Args:
        D: Latent cache.
        lo: Latent log-variance floor.
        hi: Latent log-variance ceiling.
        grid: Candidate shifts; default 161 points in $[-4, 4]$.

    Returns:
        The best shift per dimension $(d_z,)$.
    """
    grid = torch.linspace(-4, 4, 161, device=D["mup"].device) if grid is None else grid
    dz = D["mup"].shape[-1]
    best = torch.zeros(dz, device=grid.device)
    for d in range(dz):
        vals = torch.stack([latent_kl({k: (v if k == "g" else v[:, d:d + 1]) for k, v in D.items()},
                                      float(b_), lo, hi).mean()
                            for b_ in grid])
        best[d] = grid[int(vals.argmin())]
    return best


# ------------------------------------------------------------------------------ driver
def main(a):
    """Fit the variance head at every floor, measure the prior shift, evaluate and save.

    Args:
        a: The run arguments; see :data:`RUN_ARGS`.
    """
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dev, cfg, task = S.setup(a)
    model = task.orig_model.eval()
    log = {"loaded_clamp": list(model.decoder.logvar_clamp)}
    cm = model.forecast_likelihood_kwargs()["cell_mask"].double()
    lam = json.load(open(a["round1_json"]))["chosen"]["lam"]
    C = build_cache(task, cfg, dev, a["anchor_every"], a["cache_batches"])
    print("cache", {s: int(d["y"].shape[0]) for s, d in C.items()}, f"{time.time() - t0:.0f}s", flush=True)
    Sel_b = gpu(base_only(C["sel"]), dev)
    Sel_f = gpu({"f": C["sel"]["ff"], "y": C["sel"]["y"], "p": C["sel"]["p"]}, dev)

    def report(h):
        rb, rf = S.summary(h, Sel_b, cm), S.summary(h, Sel_f, cm)
        return {"base": rb, "full": rf, "both": rb["nll_per_anchor"] + rf["nll_per_anchor"],
                **coverage(h, Sel_b, cm)}

    log["start"] = report(Head3(model))
    print("start", log["start"], flush=True)

    candidates = {}
    for lo in a["floors"]:
        for name, make in (("stacked", stacked), ("base", base_only)):
            if name == "base" and lo != a["floors"][0]:
                continue
            Fs = gpu(make(C["fit"]), dev)
            h = fit_heads(model, Fs, cm, lo, lam, a)
            rep = report(h)
            candidates[(lo, name)] = (rep["both"], h)
            log[f"lo={lo},fit={name}"] = rep
            print(f"lo={lo} fit={name}", rep, f"{time.time() - t0:.0f}s", flush=True)
            del Fs
            torch.cuda.empty_cache()
    key = min(candidates, key=lambda k: candidates[k][0])
    head = candidates[key][1]
    log["chosen_head"] = {"lo": key[0], "fit": key[1]}
    if candidates[key][0] >= log["start"]["both"]:
        print("no head candidate beats the start; keeping the loaded head", flush=True)
        head = None

    lat_lo, lat_hi = model.posterior_head.logvar_clamp
    log["latent_clamps"] = [list(model.prior_head.logvar_clamp), list(model.posterior_head.logvar_clamp)]
    del Sel_b, Sel_f
    torch.cuda.empty_cache()
    L = {k: C["fit"][k].float().to(dev) for k in ("mup", "rlp", "muq", "lvq", "g")}
    kl_model = C["fit"]["kl"].float().to(dev)
    Ls = {k: C["sel"][k].float().to(dev) for k in ("mup", "rlp", "muq", "lvq", "g")}
    zero = torch.zeros(model.d_z, device=dev)
    b = solve_prior_shift(L, lat_lo, lat_hi)
    log["kl_prior_shift"] = {
        "fit_before": float(latent_kl(L, zero, lat_lo, lat_hi).sum(-1).mean()),
        "fit_after": float(latent_kl(L, b, lat_lo, lat_hi).sum(-1).mean()),
        "sel_before": float(latent_kl(Ls, zero, lat_lo, lat_hi).sum(-1).mean()),
        "sel_after": float(latent_kl(Ls, b, lat_lo, lat_hi).sum(-1).mean()),
        "shift": b.tolist(),
        "check_model_kl_vs_formula": [float(latent_kl(L, zero, lat_lo, lat_hi).sum(-1).mean()),
                                      float(kl_model.mean())],
    }
    print("prior shift KL", {k: v for k, v in log["kl_prior_shift"].items() if k != "shift"}, flush=True)
    del C, L, Ls, kl_model
    torch.cuda.empty_cache()

    log["test_start"] = S.evaluate(task, model, cfg, dev, a["eval_batches"])
    print("test start", log["test_start"], flush=True)
    if head is not None:
        S.apply_head(model, head)
        log["test_head"] = S.evaluate(task, model, cfg, dev, a["eval_batches"])
        print("test head", log["test_head"], flush=True)
    bias = model.prior_head.logvar_prior_head.skip_proj.bias
    with torch.no_grad():
        bias.add_(b)
    log["test_head_prior"] = S.evaluate(task, model, cfg, dev, a["eval_batches"])
    print("test head+prior", log["test_head_prior"], flush=True)
    if not a["keep_prior_shift"]:
        with torch.no_grad():
            bias.sub_(b)
        log["prior_shift_kept"] = False

    S.save_checkpoint(model, a["checkpoint"], a["out_ckpt"], "surgery_v3",
                      {"source": str(a["checkpoint"]), **log["chosen_head"],
                       "prior_shift_kept": a["keep_prior_shift"]},
                      clamp=model.decoder.logvar_clamp)
    json.dump(log, open(out / "v3.json", "w"), indent=1, default=str)
    print("saved", a["out_ckpt"], f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(RUN_ARGS)
