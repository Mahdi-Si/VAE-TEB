r"""Locate the 3-sigma innovation excess and test whether a global variance inflation can pass.

Run inline in session 2 on the v3 checkpoint, before round 4. For the base branch, the
standardised squared innovation is $z^2 = e^2 e^{-\ell}$, where $e$ is the AR(1) innovation. The
script reports:

* the rate of $z^2 > 9$ by horizon $\tau$ and by channel block (``fhr_st`` and ``fhr_ph``);
* how the exceedances concentrate over anchors (share held by the top 1%, 5% and 10%);
* the rate on anchors with fully observed horizons and on anchors with gaps;
* the inflation table. For a global variance factor $k$, the tail rate at $j\sigma$ is the fraction
  of cells with $z^2 > j^2 k$. The relative tail error is
  $\mathrm{rel}_j = |\mathrm{tail}_j - \mathrm{nominal}_j| / \mathrm{nominal}_j$, with nominal
  rates 0.3173, 0.0455 and 0.0027. The change of NLL per cell is
  $$\Delta\mathrm{NLL} = \tfrac12\left(\log k + \overline{z^2}\,(1/k - 1)\right).$$

The calibration verdict needs $\mathrm{rel}_j \le 0.5$ for $j = 1, 2, 3$. On the epoch-524 model no
single $k$ passes: a $k$ large enough for the 3-sigma tail makes the 1-sigma tail far too thin.
That is the signature of heavy-tailed innovations, not of a wrong scale.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from teb_vae.lag_attn_rws.nets.losses import ar_innovation  # noqa: E402
from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402

RUN_ARGS = {
    "checkpoint": common.checkpoint_variant("v3"),
    "config": common.CONFIG,
    "overrides": None,
    "batches": 20,
    "ks": [1, 1.25, 1.5, 2, 2.5, 3, 4, 5],
    "out": common.RESULTS / "surgery" / "tail_diagnostic.json",
}
NOMINAL = {1: 0.31731, 2: 0.0455, 3: 0.0027}


@torch.no_grad()
def main(a):
    """Collect $z^2$ over the first batches and write the exceedance and inflation tables.

    Args:
        a: The run arguments; see :data:`RUN_ARGS`.
    """
    dev, cfg, task = common.setup(a["checkpoint"], a["config"], a["overrides"])
    m = task.orig_model.eval()
    lk = m.forecast_likelihood_kwargs()
    cm = lk["cell_mask"]
    H, C = cm.shape
    acc = {k: torch.zeros(n, device=dev) for k, n in (("tau", H), ("tau_n", H), ("ch", C), ("ch_n", C))}
    per_anchor, zs = [], []
    gap = [0.0, 0.0, 0.0, 0.0]
    for batch in common.batches(task, cfg, dev, a["batches"]):
        _, o, tgt, sm, anc, av = common.forward_parts(task, m, batch)
        e, le = ar_innovation(tgt - o["mu_base"], lk["ar_coef"], sm, o["logvar_base"])
        M = sm[..., None] * cm
        z2 = e ** 2 * torch.exp(-le)
        ex = (z2 > 9).float() * M
        acc["tau"] += ex.sum((0, 1, 3))
        acc["tau_n"] += M.sum((0, 1, 3))
        acc["ch"] += ex.sum((0, 1, 2))
        acc["ch_n"] += M.sum((0, 1, 2))
        c = (sm.sum(-1) > 0) & av
        per_anchor.append(ex.sum((-1, -2))[c].cpu())
        full = (sm.amin(-1) > 0) & c
        gap[0] += float(ex.sum((-1, -2))[full].sum())
        gap[1] += float(M.sum((-1, -2))[full].sum())
        gap[2] += float(ex.sum((-1, -2))[c & ~full].sum())
        gap[3] += float(M.sum((-1, -2))[c & ~full].sum())
        zs.append(z2[M > 0].float().cpu())

    pa = torch.cat(per_anchor).numpy()
    srt = np.sort(pa)[::-1]
    rate_tau = (acc["tau"] / acc["tau_n"].clamp_min(1)).cpu().numpy()
    n_st = common.N_ST
    out = {
        "rate3_by_tau": rate_tau.tolist(),
        "rate3_st": float(acc["ch"][:n_st].sum() / acc["ch_n"][:n_st].sum()),
        "rate3_ph": float(acc["ch"][n_st:].sum() / acc["ch_n"][n_st:].sum()),
        "share_top_anchors": {q: float(srt[:int(len(srt) * q)].sum() / srt.sum()) for q in (0.01, 0.05, 0.1)},
        "anchors_zero_exceedance": float((pa == 0).mean()),
        "rate3_full_horizon": gap[0] / max(gap[1], 1),
        "rate3_gapped_horizon": gap[2] / max(gap[3], 1),
    }
    z2 = torch.cat(zs).numpy()
    table = []
    for k in a["ks"]:
        tail = [float((z2 > j * j * k).mean()) for j in (1, 2, 3)]
        rel = [abs(tail[j - 1] - NOMINAL[j]) / NOMINAL[j] for j in (1, 2, 3)]
        dnll = 0.5 * (math.log(k) + float(z2.mean()) * (1 / k - 1))
        table.append({"k": k, "tail": tail, "rel": rel, "dnll_per_cell": dnll, "pass": max(rel) <= 0.5})
    out["inflation"] = table
    out["n_cells"] = int(len(z2))
    for row in table:
        print(f"k={row['k']:4.2f} tail={np.round(row['tail'], 4)} rel={np.round(row['rel'], 2)} "
              f"dNLL={row['dnll_per_cell']:+.3f} pass={row['pass']}")
    Path(a["out"]).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a["out"], "w"), indent=1, default=str)


if __name__ == "__main__":
    main(RUN_ARGS)
