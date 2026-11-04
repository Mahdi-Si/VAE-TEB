r"""Round 2: neuron-level, gradient-free edits on top of the round-1 checkpoint.

Originally ``neuron_surgery.py``. Every edit is a direct change of existing weights and is exact,
so no hook survives into the saved model.

**Stage B (full forward; encoder and source path).** A coordinate search over scalar edits:
the lag-attention temperature (scale of ``query_proj``), a shift of the event-gate bias, a scale of
each ``delta_mu_head`` and of all four together, a scale of the ``fhr_ph`` columns of
``target_adapter.linear``, and a gain on each target-encoder residual branch (``LayerScale``). An
edit is accepted when it lowers the search-set score (base plus posterior-mean NLL per anchor) by
more than 0.5 nats; the whole stage is kept only when the select set also improves. On the
epoch-524 checkpoint the select set did not improve, so the stage was reverted.

**Stage A (decoder only, cached latents).** Each unit is removed alone and scored on 6000 search
anchors: a horizon-refine channel $j$ (FiLM fold: rows $j$ and $d + j$ of the FiLM generator set
so that $1 + \gamma_j = 0$ and $\beta_j = 0$), a hidden unit of the decoder ``proj`` MLP (its
column in the next linear map set to zero), or a horizon attention head (its ``out_proj`` input
slice set to zero). Units that save more than 0.5 nats are sorted, and the prefix length is chosen
on the select set. Refine-block and horizon-attention gains are then tried; a FiLM gain $g$ is the
exact fold $\gamma' = g\gamma + g - 1$, $\beta' = g\beta$.

**Head re-solve.** The round-1 solvers run again on the edited features (two rounds, the floor
and ridge chosen in round 1, the wide variance grid).
"""
from __future__ import annotations

import json
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
    "checkpoint": common.checkpoint_variant("surgery"),
    "out_ckpt": common.checkpoint_variant("neuron"),
    "out_dir": common.RESULTS / "surgery" / "round2",
    "round1_json": common.RESULTS / "surgery" / "round1" / "surgery.json",
    "search_batches": 16,
    "anchor_every": 4,
    "unit_eval_anchors": 6000,
    "min_unit_gain": 0.5,       # nats per anchor a single removal must save on the search set
}


# ------------------------------------------------------------------------------ stage B
@torch.no_grad()
def score_full(task, model, cfg, dev, n_batches):
    """Mean (base + posterior-mean full) NLL per anchor and mean KL, on search and select anchors.

    Args:
        task: The loaded task.
        model: The network.
        cfg: The merged configuration.
        dev: Device.
        n_batches: Number of leading batches to score.

    Returns:
        ``{"search": {...}, "select": {...}}`` with ``nll2``, ``kl`` and the anchor count ``n``.
    """
    lk = model.forecast_likelihood_kwargs()
    dec = model.decoder
    tot = {"search": [0.0, 0.0, 0.0], "select": [0.0, 0.0, 0.0]}
    for batch in common.batches(task, cfg, dev, n_batches):
        split = common.split_codes(batch, dev)
        _, o, tgt, sm, anc, av = common.forward_parts(task, model, batch)
        gi = anc[:, :, None].expand(-1, -1, model.d_z)
        mf, lf = dec(o["mu_post"].gather(1, gi), persistence=o["persistence"])
        M = sm[..., None] * lk["cell_mask"]

        def nll(mu, lv):
            return (raw_sample_score(mu, tgt, likelihood="gaussian_nll", logvar=lv, cell_mask=lk["cell_mask"],
                                     ar_coef=lk["ar_coef"], step_mask=sm) * M).sum((-1, -2))
        both = nll(o["mu_base"], o["logvar_base"]) + nll(mf, lf)
        kl = o["kld_per_t"].gather(1, anc)
        c = (sm.sum(-1) > 0) & av
        for name, code in (("search", 0), ("select", 2)):
            m = c & (split[:, None] == code)
            tot[name][0] += float(both[m].sum())
            tot[name][1] += float(kl[m].sum())
            tot[name][2] += float(m.sum())
    return {k: {"nll2": v[0] / max(v[2], 1), "kl": v[1] / max(v[2], 1), "n": v[2]} for k, v in tot.items()}


def scaled(params, factor, cols=None, add=None):
    """Edit tensors in place by a scale, a column scale or an additive shift.

    Args:
        params: Tensors to edit.
        factor: Multiplicative factor (ignored when ``add`` is given).
        cols: Optional column selection; only those columns are scaled.
        add: Optional additive shift instead of a scale.

    Returns:
        A closure that restores the saved values.
    """
    saved = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p in params:
            if add is not None:
                p.add_(add)
            elif cols is not None:
                p[:, cols] *= factor
            else:
                p.mul_(factor)

    def restore():
        with torch.no_grad():
            for p, s in zip(params, saved):
                p.copy_(s)
    return restore


def stage_b_edits(model):
    """List the scalar edits of the encoder and the source path.

    Args:
        model: The network.

    Returns:
        A list of ``(name, make)`` pairs; ``make()`` applies the edit and returns its restore
        closure.
    """
    ph = model.target_adapter.linear.weight
    edits = [(f"query_temp*{s}", lambda s=s: scaled([model.query_proj.weight, model.query_proj.bias], s))
             for s in (0.5, 0.75, 1.5, 2.0)]
    gate_b = model.posterior_head.event_gate_head.bias
    edits += [(f"gate_bias{d:+}", lambda d=d: scaled([gate_b], None, add=d)) for d in (-2.0, -1.0, 1.0, 2.0)]
    heads = list(model.posterior_head.delta_mu_head)
    edits += [(f"delta_mu_all*{k}", lambda k=k: scaled([p for h in heads for p in h.parameters()], k))
              for k in (0.5, 0.75, 1.25, 1.5)]
    for i, h in enumerate(heads):
        edits += [(f"delta_mu_head{i}*{k}", lambda h=h, k=k: scaled(list(h.parameters()), k)) for k in (0.5, 1.5)]
    edits += [(f"ph_input*{s}", lambda s=s: scaled([ph], s, cols=slice(common.N_ST, None))) for s in (0.0, 0.5, 2.0)]
    enc = model.target_encoder
    for i, blk in enumerate(enc.conv_blocks):
        edits += [(f"enc.conv{i}*{g}", lambda blk=blk, g=g: scaled([blk.layer_scale.weight], g)) for g in (0.8, 1.25)]
    for i, blk in enumerate(enc.attention_blocks):
        for part in ("attn_scale", "ffn_scale"):
            m_ = getattr(blk, part)
            edits += [(f"enc.attn{i}.{part}*{g}", lambda m_=m_, g=g: scaled([m_.weight], g)) for g in (0.8, 1.25)]
    return edits


def stage_b(task, model, cfg, dev, a, log):
    """Coordinate search over the scalar edits; one accepted edit per group at most.

    Args:
        task: The loaded task.
        model: The network, edited in place.
        cfg: The merged configuration.
        dev: Device.
        a: The run arguments.
        log: Result dictionary, filled in place.

    Returns:
        ``True`` when the accepted edits also improve the select set.
    """
    nb = a["search_batches"]
    base = score_full(task, model, cfg, dev, nb)
    log["stageB_start"] = base
    cur = base["search"]["nll2"]
    accepted = []
    for name, make in stage_b_edits(model):
        restore = make()
        s = score_full(task, model, cfg, dev, nb)
        d = s["search"]["nll2"] - cur
        log[f"B:{name}"] = {"d_search": d, "d_select": s["select"]["nll2"] - base["select"]["nll2"], **s}
        group = name.split("*")[0].split("+")[0].split("-")[0]
        if d < -0.5 and not any(x.startswith(group) for x in accepted):
            cur = s["search"]["nll2"]
            accepted.append(name)
        else:
            restore()
    end = score_full(task, model, cfg, dev, nb)
    log["stageB_accepted"] = accepted
    log["stageB_end"] = end
    print("stage B end", end, "accepted", accepted, flush=True)
    return end["select"]["nll2"] < base["select"]["nll2"]


# ------------------------------------------------------------------------------ stage A
@torch.no_grad()
def cache_latents(task, model, cfg, dev, every):
    """Cache prior and posterior means, persistence and targets on fully observed anchors.

    Args:
        task: The loaded task.
        model: The network.
        cfg: The merged configuration.
        dev: Device.
        every: Keep every ``every``-th anchor.

    Returns:
        ``{"fit": D, "sel": D}`` with ``D = {"zb", "zf", "p", "y"}``.
    """
    out = {s: {k: [] for k in ("zb", "zf", "p", "y")} for s in ("fit", "sel")}
    for bi, batch in enumerate(common.batches(task, cfg, dev)):
        split = common.split_codes(batch, dev)
        _, o, tgt, sm, anc, av = common.forward_parts(task, model, batch)
        A = anc.shape[1]
        keep = av & (sm.amin(-1) > 0)
        keep[:, (torch.arange(A, device=dev) % every) != (bi % every)] = False
        gi = anc[:, :, None].expand(-1, -1, model.d_z)
        for name, code in (("fit", 0), ("sel", 2)):
            m = keep & (split[:, None] == code)
            out[name]["zb"].append(o["mu_prior"].gather(1, gi)[m])
            out[name]["zf"].append(o["mu_post"].gather(1, gi)[m])
            out[name]["p"].append(o["persistence"][m])
            out[name]["y"].append(tgt[m])
    return {s: {k: torch.cat(v) for k, v in d.items()} for s, d in out.items()}


@torch.no_grad()
def score_dec(model, D, n=None, chunk=2000):
    """Mean (base + posterior-mean) AR(1) NLL per fully observed anchor, decoder only.

    The constant $\\tfrac12 \\log 2\\pi$ per cell is omitted; only differences are used.

    Args:
        model: The network.
        D: A latent cache from :func:`cache_latents`.
        n: Optional number of leading anchors to score.
        chunk: Anchors per decoder call.

    Returns:
        The score per anchor.
    """
    lk = model.forecast_likelihood_kwargs()
    cm, phi = lk["cell_mask"], lk["ar_coef"]
    N = D["y"].shape[0] if n is None else min(n, D["y"].shape[0])
    tot = 0.0
    for s in range(0, N, chunk):
        sl = slice(s, min(s + chunk, N))
        for z in (D["zb"][sl], D["zf"][sl]):
            mu, lv = model.decoder(z[:, None], persistence=D["p"][sl][:, None])
            r = D["y"][sl] - mu[:, 0]
            e = r.clone()
            e[:, 1:] = r[:, 1:] - phi * r[:, :-1]
            lv = lv[:, 0]
            tot += float((0.5 * (lv + e ** 2 * torch.exp(-lv)) * cm).sum())
    return tot / N


def scaled_rows(lin, rows, biases):
    """Set rows of a linear layer to zero weight and a given bias.

    Args:
        lin: The linear layer.
        rows: Row indices.
        biases: The bias value per row.

    Returns:
        A closure that restores the saved weight and bias.
    """
    saved = (lin.weight.detach().clone(), lin.bias.detach().clone())
    with torch.no_grad():
        for r, b in zip(rows, biases):
            lin.weight[r] = 0.0
            lin.bias[r] = b

    def restore():
        with torch.no_grad():
            lin.weight.copy_(saved[0])
            lin.bias.copy_(saved[1])
    return restore


def film_mask(film, j):
    r"""Remove refine channel $j$ exactly: $1 + \gamma_j = 0$ and $\beta_j = 0$ for every condition.

    Args:
        film: The block's FiLM generator, ``Linear(d, 2d)`` with $[\gamma, \beta]$ outputs.
        j: Channel index.

    Returns:
        The restore closure.
    """
    d = film.out_features // 2
    return scaled_rows(film, [j, d + j], [-1.0, 0.0])


def film_gain(film, g):
    r"""Scale a refine block's residual branch by $g$: $\gamma' = g\gamma + g - 1$, $\beta' = g\beta$.

    Args:
        film: The block's FiLM generator.
        g: The gain.

    Returns:
        The restore closure.
    """
    d = film.out_features // 2
    saved = (film.weight.detach().clone(), film.bias.detach().clone())
    with torch.no_grad():
        film.weight.mul_(g)
        film.bias[:d] = g * film.bias[:d] + g - 1.0
        film.bias[d:] = g * film.bias[d:]

    def restore():
        with torch.no_grad():
            film.weight.copy_(saved[0])
            film.bias.copy_(saved[1])
    return restore


def decoder_units(model):
    """List every removable decoder unit.

    Args:
        model: The network.

    Returns:
        A list of ``(name, make)`` pairs; ``make()`` removes the unit and returns its restore
        closure.
    """
    dec = model.decoder
    core = dec.core
    units = []
    for b, film in enumerate(core.refine.film):
        units += [(f"refine{b}.ch{j}", lambda film=film, j=j: film_mask(film, j))
                  for j in range(film.out_features // 2)]
    body = dec.proj.body
    for li in (3, 6, 9):
        nxt = body[li]
        units += [(f"proj.body{li}.in{j}", lambda nxt=nxt, j=j: scaled([nxt.weight], 0.0, cols=[j]))
                  for j in range(nxt.in_features)]
    for i, blk in enumerate(core.attention):
        dh = blk.d_head
        units += [(f"hzn.attn{i}.head{h}", lambda blk=blk, h=h, dh=dh:
                   scaled([blk.out_proj.weight], 0.0, cols=slice(h * dh, (h + 1) * dh)))
                  for h in range(blk.num_heads)]
    return units


def stage_a(model, D, a, log):
    """Single-unit removal scan, prefix selection on the select set, then branch gains.

    Args:
        model: The network, edited in place.
        D: The latent cache.
        a: The run arguments.
        log: Result dictionary, filled in place.
    """
    core = model.decoder.core
    n = a["unit_eval_anchors"]
    s0 = score_dec(model, D["fit"], n)
    log["stageA_start"] = {"fit": s0, "sel": score_dec(model, D["sel"])}
    gains = []
    t0 = time.time()
    units = decoder_units(model)
    for k, (name, make) in enumerate(units):
        restore = make()
        gains.append((score_dec(model, D["fit"], n) - s0, name, make))
        restore()
        if k % 200 == 0:
            print(f"  unit {k}/{len(units)} {time.time() - t0:.0f}s", flush=True)
    gains.sort(key=lambda x: x[0])
    log["stageA_top_units"] = [(nm, d) for d, nm, _ in gains[:40]]
    log["stageA_unit_delta_hist"] = {
        "n_units": len(gains),
        "n_helpful": sum(d < -a["min_unit_gain"] for d, _, _ in gains),
        "n_harmless(|d|<0.5)": sum(abs(d) < 0.5 for d, _, _ in gains),
        "worst_removal": (gains[-1][1], gains[-1][0]),
    }
    cand = [g for g in gains if g[0] < -a["min_unit_gain"]]
    best_k, best_sel = 0, log["stageA_start"]["sel"]
    for k in sorted({k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, len(cand)) if 0 < k <= len(cand)}):
        rest = [make() for _, _, make in cand[:k]]
        sel = score_dec(model, D["sel"])
        log[f"stageA_prefix{k}"] = {"fit": score_dec(model, D["fit"], n), "sel": sel}
        if sel < best_sel - 0.1:
            best_k, best_sel = k, sel
        for r in reversed(rest):
            r()
    for _, _, make in cand[:best_k]:
        make()
    log["stageA_removed"] = [nm for _, nm, _ in cand[:best_k]]
    cur_fit, cur_sel = score_dec(model, D["fit"], n), score_dec(model, D["sel"])
    for b, film in enumerate(core.refine.film):
        for g in (0.8, 0.9, 1.1, 1.25):
            restore = film_gain(film, g)
            f_, s_ = score_dec(model, D["fit"], n), score_dec(model, D["sel"])
            if f_ < cur_fit - 0.3 and s_ < cur_sel - 0.1:
                cur_fit, cur_sel = f_, s_
                log.setdefault("stageA_gains", []).append((f"refine{b}.gain", g, f_, s_))
                break
            restore()
    for i, blk in enumerate(core.attention):
        for g in (0.5, 0.8, 1.25, 1.5):
            restore = scaled([blk.residual_gain], g)
            f_, s_ = score_dec(model, D["fit"], n), score_dec(model, D["sel"])
            if f_ < cur_fit - 0.3 and s_ < cur_sel - 0.1:
                cur_fit, cur_sel = f_, s_
                log.setdefault("stageA_gains", []).append((f"hzn.attn{i}.gain", g, f_, s_))
                break
            restore()
    log["stageA_end"] = {"fit": cur_fit, "sel": cur_sel}


# ------------------------------------------------------------------------------ driver
def main(a):
    """Run stage B, stage A and the head re-solve; evaluate the test half; save.

    Args:
        a: The run arguments; see :data:`RUN_ARGS`.
    """
    out = Path(a["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dev, cfg, task = S.setup(a)
    model = task.orig_model
    log = {"loaded_logvar_clamp": list(model.decoder.logvar_clamp)}
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if not stage_b(task, model, cfg, dev, a, log):
        model.load_state_dict(snapshot)
        log["stageB_reverted"] = True
    D = cache_latents(task, model, cfg, dev, a["anchor_every"])
    stage_a(model, D, a, log)
    del D
    torch.cuda.empty_cache()

    chosen = json.load(open(a["round1_json"]))["chosen"]
    cm = model.forecast_likelihood_kwargs()["cell_mask"].double()
    cache = S.build_cache(task, cfg, dev, a["anchor_every"])
    h = S.Head(model)
    h.lo = chosen["lo"]
    log["head_before"] = S.summary(h, cache["sel"], cm)
    for _ in range(2):
        S.solve_mean(h, cache["fit"], cm, chosen["lam"])
        S.solve_phi(h, cache["fit"], cm)
        S.solve_variance(h, cache["fit"], cm, a_lo=0.05, a_points=50)
    log["head_after"] = S.summary(h, cache["sel"], cm)
    del cache
    torch.cuda.empty_cache()
    S.apply_head(model, h)
    log["test_neuron"] = S.evaluate(task, model, cfg, dev)
    log["test_neuron_ua_all_valid"] = S.evaluate(task, model, cfg, dev, source_all_valid=True)
    S.save_checkpoint(model, a["checkpoint"], a["out_ckpt"], "neuron_surgery",
                      {"source": str(a["checkpoint"]), "stageB": log.get("stageB_accepted"),
                       "removed_units": log.get("stageA_removed"), "gains": log.get("stageA_gains")},
                      clamp=model.decoder.logvar_clamp)
    json.dump(log, open(out / "neuron.json", "w"), indent=1, default=str)
    print("saved", a["out_ckpt"], f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(RUN_ARGS)
