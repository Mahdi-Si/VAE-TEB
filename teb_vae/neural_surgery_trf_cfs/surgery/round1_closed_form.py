r"""Round 1: closed-form and derivative-free surgery of the decoder's last layers.

Originally ``surgery.py``. Every edit acts on the fixed decoder feature
$f_\tau = \mathrm{core}(\mathrm{proj}(z))$, so no gradient reaches any other layer:

* **Mean.** $\mu_{\tau,c} = W_c f_\tau + b_c + w_{\tau,c}\, y_{t,c}$ is linear in
  $(W_c, b_c, w_{\cdot,c})$. Under the AR(1) innovation likelihood
  $e_\tau = r_\tau - \phi_c\, r_{\tau-1}$, $r = y - \mu$, the NLL is quadratic in those
  parameters, so generalised least squares with weights $e^{-\ell_{\tau,c}}$ gives the exact
  optimum given the variances and $\phi_c$. A ridge term $\lambda\,\overline{\mathrm{diag}}$ acts on
  the feature columns only.
* **AR coefficient.** Given residuals and variances,
  $\hat\phi_c = \sum \omega\, r_\tau r_{\tau-1} / \sum \omega\, r_{\tau-1}^2$, capped at $0.9999$
  and written back as $a_c = \operatorname{atanh}(\hat\phi_c)$.
* **Variance.** $\ell = lo + (hi - lo)\,\sigma(a_c\, \rho + c_c)$ with $\rho$ the raw head
  output. $(a_c, c_c)$ fold into ``logvar_head`` exactly; they are chosen per channel by a coarse
  then a fine grid search, and the floor $lo$ by the select set.

The fit uses recordings with split code 0, the choices use code 2, and the reported test numbers
use codes 1 and 3. The round as run used the coarse variance grid $a \in [0.3, 2.5]$ in 23 points;
later rounds widened it to $[0.05, 2.5]$ in 50 points through ``a_lo`` and ``a_points``.

This module also holds the shared pieces of every later round: :class:`Head`, the three solvers,
:func:`apply_head` and the test-half :func:`evaluate`.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from teb_vae.lag_attn_rws.nets.losses import ar_innovation, raw_sample_score  # noqa: E402
from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402

RUN_ARGS = {
    "checkpoint": common.CHECKPOINT,
    "config": common.CONFIG,
    "overrides": None,
    "out_ckpt": common.checkpoint_variant("surgery"),
    "out_dir": common.RESULTS / "surgery" / "round1",
    "anchor_every": 4,          # cache every k-th anchor of the fit and select recordings
    "rounds": 2,
    "floors": [-6.0, -7.0, -8.0, -9.0, -10.0, -12.0],
    "ridge": [1e-4, 1e-3, 1e-2],
    "a_lo": 0.3,                # coarse variance-scale grid: 0.3..2.5 in 23 points (as run)
    "a_points": 23,
    "eval_batches": None,
    "cache_batches": None,
}
LOG2PI = math.log(2 * math.pi)

# Re-exported so later rounds can reach every shared helper through this module.
batches = common.batches
forward_parts = common.forward_parts
guid_hash = common.guid_hash


def setup(a):
    """Load the configuration and the checkpoint named in a run-argument dictionary.

    Args:
        a: Run arguments with ``checkpoint``, ``config`` and optionally ``overrides``.

    Returns:
        ``(device, config, task)``, see :func:`teb_vae.neural_surgery_trf_cfs.common.setup`.
    """
    return common.setup(a["checkpoint"], a["config"], a.get("overrides"))


# ---------------------------------------------------------------------------------- cache
@torch.no_grad()
def build_cache(task, cfg, dev, every, limit=None):
    """Cache decoder features, raw log-variance, targets and persistence on fit and select anchors.

    Only anchors whose whole horizon is observed are kept, so the AR(1) recursion has no gap.

    Args:
        task: The loaded task.
        cfg: The merged configuration.
        dev: Device.
        every: Keep every ``every``-th anchor.
        limit: Optional batch limit.

    Returns:
        ``{"fit": D, "sel": D}`` with ``D = {"f": (N, H, F) half, "r": (N, H, C), "y": (N, H, C),
        "p": (N, C)}``.
    """
    model = task.orig_model
    dec = model.decoder
    cache = {"fit": {k: [] for k in "f r y p".split()}, "sel": {k: [] for k in "f r y p".split()}}
    for bi, batch in enumerate(batches(task, cfg, dev, limit)):
        split = common.split_codes(batch, dev)
        if not bool(((split == 0) | (split == 2)).any()):
            continue
        _, o, tgt, sm, anc, av = forward_parts(task, model, batch)
        A = anc.shape[1]
        keep = av & (sm.amin(-1) > 0)
        keep[:, (torch.arange(A, device=dev) % every) != (bi % every)] = False
        gi = anc[:, :, None].expand(-1, -1, model.d_z)
        feat = dec.core.decode(dec.proj(o["mu_prior"].gather(1, gi)))
        raw = dec.logvar_head(feat)
        for name, code in (("fit", 0), ("sel", 2)):
            m = keep & (split[:, None] == code)
            for k, v in (("f", feat), ("r", raw), ("y", tgt), ("p", o["persistence"])):
                cache[name][k].append(v[m].half() if k == "f" else v[m])
        if bi % 20 == 0:
            print(f"cache batch {bi}", flush=True)
    return {s: {k: torch.cat(v) for k, v in d.items()} for s, d in cache.items()}


# ---------------------------------------------------------------------------------- head state
class Head:
    """The decoder's last-layer parameters, edited in closed form and written back at the end.

    Attributes:
        W, b: ``mean_head`` weight $(C, F)$ and bias $(C,)$.
        w: Persistence weights $(H, C)$.
        Wv, bv: ``logvar_head`` weight and bias.
        a, c: Per-channel affine on the raw log-variance, folded into ``Wv, bv`` on write-back.
        lo, hi: The log-variance bound.
        phi: AR(1) coefficients $(C,)$.
    """

    def __init__(self, model):
        """Copy the current last-layer parameters of a model.

        Args:
            model: The network.
        """
        d = model.decoder
        self.W = d.mean_head.weight.detach().double().clone()
        self.b = d.mean_head.bias.detach().double().clone()
        self.w = d.persistence_weight.detach().double().clone()
        self.Wv = d.logvar_head.weight.detach().double().clone()
        self.bv = d.logvar_head.bias.detach().double().clone()
        self.a = torch.ones_like(self.bv)
        self.c = torch.zeros_like(self.bv)
        self.lo, self.hi = d.logvar_clamp
        self.phi = torch.tanh(model.target_ar_logit.detach().double()).clone()

    def mean(self, D):
        """Forecast mean on a cache.

        Args:
            D: A cache with ``f`` and ``p``.

        Returns:
            $(N, H, C)$ in float64.
        """
        f = D["f"].double()
        return f @ self.W.T + self.b + self.w[None] * D["p"].double()[:, None, :]

    def logvar(self, D):
        """Forecast log-variance on a cache, from the cached raw head output.

        Args:
            D: A cache with ``r``.

        Returns:
            $(N, H, C)$ in float64.
        """
        r = D["r"].double() * self.a + self.c
        return self.lo + (self.hi - self.lo) * torch.sigmoid(r)


def cell_nll(head, D, cell_mask):
    """Per-cell AR(1) innovation NLL on fully observed horizons.

    Args:
        head: A :class:`Head`.
        D: A cache.
        cell_mask: Scored cells $(H, C)$.

    Returns:
        The masked per-cell NLL $(N, H, C)$.
    """
    r = D["y"].double() - head.mean(D)
    lv = head.logvar(D)
    e = r.clone()
    e[:, 1:] = r[:, 1:] - head.phi * r[:, :-1]
    return 0.5 * (LOG2PI + lv + e ** 2 * torch.exp(-lv)) * cell_mask


def summary(head, D, cell_mask, n_st=common.N_ST):
    """NLL per scored cell of each block and per anchor.

    Args:
        head: A :class:`Head`.
        D: A cache.
        cell_mask: Scored cells $(H, C)$.
        n_st: Number of leading ``fhr_st`` channels.

    Returns:
        ``{"nll_st", "nll_ph", "nll_per_anchor"}``.
    """
    nll = cell_nll(head, D, cell_mask)
    m = cell_mask.expand_as(nll)
    st = torch.zeros(cell_mask.shape[-1], dtype=torch.bool, device=nll.device)
    st[:n_st] = True
    return {"nll_st": float(nll[..., st].sum() / m[..., st].sum()),
            "nll_ph": float(nll[..., ~st].sum() / m[..., ~st].sum()),
            "nll_per_anchor": float(nll.sum() / nll.shape[0])}


# ---------------------------------------------------------------------------------- the three solves
def solve_mean(head, D, cell_mask, lam_rel, chunk=3000):
    r"""GLS for $(W_c, b_c, w_{\cdot,c})$ per channel under the AR(1) innovation likelihood.

    The design row at step $\tau$ is $x_\tau = [f_\tau, 1, y_{t,c}\, \mathbf e_\tau]$; the
    innovation transform is $\tilde x_\tau = x_\tau - \phi_c x_{\tau-1}$ and
    $\tilde y_\tau = y_\tau - \phi_c y_{\tau-1}$, with $\tilde x_0 = x_0$. Gram matrices are built
    in float32 per chunk and accumulated in float64.

    Args:
        head: The :class:`Head` to update in place.
        D: The fit cache.
        cell_mask: Scored cells $(H, C)$; channel $c$ uses its first $H_c$ steps.
        lam_rel: Ridge strength relative to the mean diagonal of the feature block.
        chunk: Anchors per Gram chunk.
    """
    N, H, F = D["f"].shape
    C = D["y"].shape[-1]
    Hc = cell_mask.sum(0).long()
    lv = head.logvar(D)
    for c in range(C):
        h = int(Hc[c])
        P = F + 1 + h
        G = torch.zeros(P, P, dtype=torch.float64, device=D["f"].device)
        g = torch.zeros(P, dtype=torch.float64, device=D["f"].device)
        phi = float(head.phi[c])
        for s in range(0, N, chunk):
            f = D["f"][s:s + chunk, :h].float()
            n = f.shape[0]
            p = D["p"][s:s + chunk, c].float()
            X = torch.cat([f, torch.ones(n, h, 1, dtype=f.dtype, device=f.device),
                           torch.diag_embed(p[:, None].expand(n, h))], -1)
            y = D["y"][s:s + chunk, :h, c].float()
            Xt, yt = X.clone(), y.clone()
            Xt[:, 1:] = X[:, 1:] - phi * X[:, :-1]
            yt[:, 1:] = y[:, 1:] - phi * y[:, :-1]
            om = torch.exp(-lv[s:s + chunk, :h, c]).float()
            Xw = Xt * om[..., None]
            G += torch.einsum("nhp,nhq->pq", Xw, Xt).double()
            g += torch.einsum("nhp,nh->p", Xw, yt).double()
        reg = torch.zeros(P, dtype=torch.float64, device=G.device)
        reg[:F] = lam_rel * G.diagonal()[:F].mean()
        sol = torch.linalg.solve(G + torch.diag(reg), g)
        head.W[c], head.b[c], head.w[:h, c] = sol[:F], sol[F], sol[F + 1:]


def solve_phi(head, D, cell_mask, cap=0.9999):
    r"""Weighted least squares for $\phi_c$ given the residuals and the variances.

    Args:
        head: The :class:`Head` to update in place.
        D: The fit cache.
        cell_mask: Scored cells; a pair of steps counts only when both are scored.
        cap: Bound on $|\phi_c|$, so that $\operatorname{atanh}$ stays finite.
    """
    r = D["y"].double() - head.mean(D)
    om = torch.exp(-head.logvar(D))
    m = cell_mask[1:] * cell_mask[:-1]
    num = (om[:, 1:] * r[:, 1:] * r[:, :-1] * m).sum((0, 1))
    den = (om[:, 1:] * r[:, :-1] ** 2 * m).sum((0, 1))
    head.phi = (num / den).clamp(-cap, cap)


def solve_variance(head, D, cell_mask, max_cells=400_000, a_lo=0.3, a_points=23):
    """Per-channel $(a_c, c_c)$ by a coarse then a fine grid search on the innovation NLL.

    The floor ``head.lo`` is fixed by the caller. Each channel uses at most ``max_cells`` cells.

    Args:
        head: The :class:`Head` to update in place.
        D: The fit cache.
        cell_mask: Scored cells.
        max_cells: Random subsample size per channel.
        a_lo: Lower end of the coarse scale grid, whose upper end is 2.5.
        a_points: Number of coarse scale points (23 in round 1, 50 from round 2 on).
    """
    r = D["y"].double() - head.mean(D)
    e = r.clone()
    e[:, 1:] = r[:, 1:] - head.phi * r[:, :-1]
    e2 = e ** 2
    raw = D["r"].double()
    dev = raw.device
    for c in range(raw.shape[-1]):
        h = int(cell_mask[:, c].sum())
        x, q = raw[:, :h, c].reshape(-1), e2[:, :h, c].reshape(-1)
        if x.numel() > max_cells:
            idx = torch.randperm(x.numel(), device=dev)[:max_cells]
            x, q = x[idx], q[idx]
        grids = [(torch.linspace(a_lo, 2.5, a_points), torch.linspace(-4.0, 4.0, 81))]
        best = (float("inf"), 1.0, 0.0)
        for _ in range(2):
            ga, gc = grids[-1]
            gc = gc.double().to(dev)
            for av in ga.tolist():
                lv = head.lo + (head.hi - head.lo) * torch.sigmoid(av * x[None] + gc[:, None])
                tot = (lv + q[None] * torch.exp(-lv)).mean(1)
                k = int(tot.argmin())
                if float(tot[k]) < best[0]:
                    best = (float(tot[k]), av, float(gc[k]))
            grids.append((torch.linspace(best[1] - 0.1, best[1] + 0.1, 11),
                          torch.linspace(best[2] - 0.1, best[2] + 0.1, 21)))
        head.a[c], head.c[c] = best[1], best[2]


# ---------------------------------------------------------------------------------- write back
def apply_head(model, head):
    """Write a :class:`Head` back into the model, folding the variance affine into the weights.

    Sets ``decoder.logvar_clamp`` in memory only. The checkpoint's ``logvar_clamp`` keyword is
    shared with both latent heads; round 4 handles that side effect.

    Args:
        model: The network, edited in place.
        head: The solved head.
    """
    d = model.decoder
    with torch.no_grad():
        d.mean_head.weight.copy_(head.W.float())
        d.mean_head.bias.copy_(head.b.float())
        d.persistence_weight.copy_(head.w.float())
        d.logvar_head.weight.copy_((head.a[:, None] * head.Wv).float())
        d.logvar_head.bias.copy_((head.a * head.bv + head.c).float())
        model.target_ar_logit.copy_(torch.atanh(head.phi).float())
    d.logvar_clamp = (float(head.lo), float(head.hi))


def save_checkpoint(model, source, out, extra_key, extra, clamp=None):
    """Copy every model tensor into a copy of the source checkpoint and save it.

    Args:
        model: The edited network.
        source: The checkpoint to copy.
        out: Output path; must differ from ``source``.
        extra_key: Name of the provenance record written into the checkpoint.
        extra: The provenance record.
        clamp: Optional new ``logvar_clamp`` for ``model_kwargs``.
    """
    assert Path(out).resolve() != Path(source).resolve(), "refusing to overwrite the source"
    ck = torch.load(source, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    for pre in ("model.", "_orig_model."):
        for k, v in model.state_dict().items():
            if pre + k in sd:
                sd[pre + k] = v.detach().cpu().clone()
    if clamp is not None:
        for kw in (ck["model_kwargs"], ck["hyper_parameters"]["model_kwargs"]):
            kw["logvar_clamp"] = [float(clamp[0]), float(clamp[1])]
    ck[extra_key] = extra
    torch.save(ck, out)


# ---------------------------------------------------------------------------------- test evaluation
@torch.no_grad()
def evaluate(task, model, cfg, dev, limit=None, source_all_valid=False):
    """Test-half metrics with the model's own likelihood over every scored anchor.

    Reports per block: NLL per scored cell of the base, posterior-mean and sampled full branches;
    MSE overall, at $\\tau = 0$ and at $\\tau = 29$; innovation $z^2$; implied marginal 90% coverage.
    Overall: NLL per anchor, the two source gaps, the KL per anchor and the innovation coverage and
    relative tail error at 1, 2 and 3 sigma.

    Args:
        task: The loaded task.
        model: The network to score.
        cfg: The merged configuration.
        dev: Device.
        limit: Optional batch limit.
        source_all_valid: Mark every UA step valid before the forward.

    Returns:
        A flat dictionary of floats.
    """
    lk = model.forecast_likelihood_kwargs()
    C = lk["cell_mask"].shape[1]
    st = torch.zeros(C, dtype=torch.bool, device=dev)
    st[:common.N_ST] = True
    acc = {}

    def add(k, v):
        acc[k] = acc.get(k, 0.0) + float(v)

    dec = model.decoder
    for batch in batches(task, cfg, dev, limit):
        odd = common.split_codes(batch, dev) % 2 == 1
        if not bool(odd.any()):
            continue
        inputs, o, tgt, sm, anc, av = forward_parts(task, model, batch)
        if source_all_valid:
            inputs = list(inputs)
            inputs[5] = torch.ones_like(inputs[5])
            o = model(*inputs)
        gi = anc[:, :, None].expand(-1, -1, model.d_z)
        mf, lf = dec(o["mu_post"].gather(1, gi), persistence=o["persistence"])
        M = sm[..., None] * lk["cell_mask"] * odd[:, None, None, None].to(sm.dtype)
        contrib = (sm.sum(-1) > 0) & av & odd[:, None]
        for name, (mu, lv) in {"base": (o["mu_base"], o["logvar_base"]), "fullmean": (mf, lf),
                               "fullsample": (o["mu_full"], o["logvar_full"])}.items():
            n = raw_sample_score(mu, tgt, likelihood="gaussian_nll", logvar=lv, cell_mask=lk["cell_mask"],
                                 ar_coef=lk["ar_coef"], step_mask=sm) * M
            add(f"nll_{name}_st", (n * st).sum())
            add(f"nll_{name}_ph", (n * ~st).sum())
            if name == "fullsample":
                continue
            se = (tgt - mu) ** 2 * M
            add(f"se_{name}_st", (se * st).sum())
            add(f"se_{name}_ph", (se * ~st).sum())
            add(f"se_{name}_st_t0", (se[:, :, 0] * st).sum())
            add(f"se_{name}_ph_t0", (se[:, :, 0] * ~st).sum())
            add(f"se_{name}_st_t29", (se[:, :, 29] * st).sum())
            if name == "base":
                e, le = ar_innovation(tgt - mu, lk["ar_coef"], sm, lv)
                z2 = e ** 2 * torch.exp(-le)
                add("z2_st", (z2 * M * st).sum())
                add("z2_ph", (z2 * M * ~st).sum())
                for k_ in (1, 2, 3):
                    add(f"cov{k_}", ((z2 < k_ * k_).to(M.dtype) * M).sum())
                s2 = torch.exp(lv)
                V = torch.zeros_like(s2)
                V[..., 0, :] = s2[..., 0, :]
                for t in range(1, s2.shape[-2]):
                    V[..., t, :] = s2[..., t, :] + lk["ar_coef"] ** 2 * V[..., t - 1, :]
                cov = (((tgt - mu) ** 2) < 1.6449 ** 2 * V).to(M.dtype) * M
                add("mcov90_st", (cov * st).sum())
                add("mcov90_ph", (cov * ~st).sum())
        add("n_st", (M * st).sum())
        add("n_ph", (M * ~st).sum())
        add("n_st_t0", (M[:, :, 0] * st).sum())
        add("n_ph_t0", (M[:, :, 0] * ~st).sum())
        add("n_st_t29", (M[:, :, 29] * st).sum())
        add("anchors", contrib.sum())
        add("kl", o["kld_per_t"].gather(1, anc)[contrib].sum())
    r = {}
    for blk in ("st", "ph"):
        for name in ("base", "fullmean", "fullsample"):
            r[f"nll_{name}_{blk}"] = acc[f"nll_{name}_{blk}"] / acc[f"n_{blk}"]
        for name in ("base", "fullmean"):
            r[f"mse_{name}_{blk}"] = acc[f"se_{name}_{blk}"] / acc[f"n_{blk}"]
            r[f"mse_{name}_{blk}_t0"] = acc[f"se_{name}_{blk}_t0"] / acc[f"n_{blk}_t0"]
        r[f"z2_base_{blk}"] = acc[f"z2_{blk}"] / acc[f"n_{blk}"]
        r[f"marg_cov90_base_{blk}"] = acc[f"mcov90_{blk}"] / acc[f"n_{blk}"]
    r["mse_base_st_t29"] = acc["se_base_st_t29"] / acc["n_st_t29"]
    r["mse_fullmean_st_t29"] = acc["se_fullmean_st_t29"] / acc["n_st_t29"]
    na = acc["anchors"]
    for name in ("base", "fullmean", "fullsample"):
        r[f"nll_{name}_per_anchor"] = (acc[f"nll_{name}_st"] + acc[f"nll_{name}_ph"]) / na
    r["gap_mean_per_anchor"] = r["nll_base_per_anchor"] - r["nll_fullmean_per_anchor"]
    r["gap_sample_per_anchor"] = r["nll_base_per_anchor"] - r["nll_fullsample_per_anchor"]
    r["anchors"] = na
    r["kl_per_anchor"] = acc["kl"] / na
    nominal = {1: 0.6826894921370857, 2: 0.9544997361036417, 3: 0.9973002039367398}
    ncell = acc["n_st"] + acc["n_ph"]
    for k_ in (1, 2, 3):
        obs = acc[f"cov{k_}"] / ncell
        r[f"innov_cov_{k_}sigma"] = obs
        r[f"rel_tail_err_{k_}sigma"] = abs((1 - obs) - (1 - nominal[k_])) / (1 - nominal[k_])
    return r


# ---------------------------------------------------------------------------------- driver
def main(a):
    """Choose the floor and the ridge strength, solve, evaluate the test half and save.

    Args:
        a: The run arguments; see :data:`RUN_ARGS`.
    """
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dev, cfg, task = setup(a)
    model = task.orig_model
    cm = model.forecast_likelihood_kwargs()["cell_mask"].double()
    cache = build_cache(task, cfg, dev, a["anchor_every"], a.get("cache_batches"))
    Fit, Sel = cache["fit"], cache["sel"]
    print("cache", {s: int(d["y"].shape[0]) for s, d in cache.items()}, f"{time.time() - t0:.0f}s", flush=True)
    log = {"original_sel": summary(Head(model), Sel, cm)}

    def var(h):
        solve_variance(h, Fit, cm, a_lo=a["a_lo"], a_points=a["a_points"])

    def trial(lo, lam):
        h = Head(model)
        h.lo = lo
        var(h)
        solve_mean(h, Fit, cm, lam)
        solve_phi(h, Fit, cm)
        var(h)
        s = summary(h, Sel, cm)
        log[f"lo={lo},lam={lam}"] = s
        print(f"lo={lo} lam={lam}", s, f"{time.time() - t0:.0f}s", flush=True)
        return s["nll_per_anchor"]

    lo = min(a["floors"], key=lambda v: trial(v, 1e-3))
    lam = min(a["ridge"], key=lambda v: trial(lo, v))
    h = Head(model)
    h.lo = lo
    var(h)
    for rnd in range(a["rounds"]):
        solve_mean(h, Fit, cm, lam)
        solve_phi(h, Fit, cm)
        var(h)
        log[f"round{rnd}"] = summary(h, Sel, cm)
    for name, steps in {"variance_only": ("v",), "phi+variance": ("p", "v"), "mean+variance": ("m", "v")}.items():
        g = Head(model)
        g.lo = lo
        for st_ in steps:
            {"v": lambda: solve_variance(g, Fit, cm, a_lo=a["a_lo"], a_points=a["a_points"]), "p": lambda: solve_phi(g, Fit, cm),
             "m": lambda: solve_mean(g, Fit, cm, lam)}[st_]()
        if "v" in steps:
            solve_variance(g, Fit, cm, a_lo=a["a_lo"], a_points=a["a_points"])
        log[f"edit:{name}"] = summary(g, Sel, cm)
    log["chosen"] = {"lo": lo, "lam": lam, "phi": h.phi.tolist(), "a": h.a.tolist(), "c": h.c.tolist()}
    del cache, Fit, Sel
    torch.cuda.empty_cache()

    log["test_original"] = evaluate(task, model, cfg, dev, a["eval_batches"])
    apply_head(model, h)
    log["test_surgery"] = evaluate(task, model, cfg, dev, a["eval_batches"])
    log["test_surgery_ua_all_valid"] = evaluate(task, model, cfg, dev, a["eval_batches"], source_all_valid=True)
    save_checkpoint(model, a["checkpoint"], a["out_ckpt"], "surgery",
                    {"source": str(a["checkpoint"]), "lo": lo, "ridge": lam,
                     "edits": "GLS mean head + persistence, WLS AR(1) phi, per-channel variance affine"},
                    clamp=(lo, h.hi))
    json.dump(log, open(out / "surgery.json", "w"), indent=1)
    print("saved", a["out_ckpt"], f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(RUN_ARGS)
