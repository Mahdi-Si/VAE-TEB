r"""Fix the Student-$t$ degrees of freedom from the epoch-424 innovations of the fit recordings.

The full branch is decoded at the posterior mean on split code 0. For every scored cell the
script takes the standardised AR(1) innovation $z = \varepsilon / \sigma$ under the model's own
variance, separately for the scattering and the phase block, and for cells on and off the
decoder's log-variance floor. It reports:

* the Gaussian tail shares at 1, 2 and 3 standard deviations and the excess kurtosis;
* the maximum-likelihood fit of a scaled Student-$t$ density, $z / c \sim t_\nu$, on a grid of
  $\nu$ with the scale $c$ solved per $\nu$. The fitted $\nu$ is the value that the ``studentt``
  pilot arm uses, and $c$ says how far the model's variance is from the fitted scale.

The log-likelihood gain per cell over the Gaussian density with the model's own variance is the
most that a change of density can buy before any retraining.

Edit :data:`RUN_ARGS` and run this file from the IDE.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from scipy import optimize, stats

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.lag_attn_rws.nets.losses import ar_innovation  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7_interventions as I  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7_pilot as P  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {"device": "cuda:0", "codes": [0], "max_cells": 2_000_000,
                            "nu_grid": [2.5, 3, 4, 5, 6, 8, 10, 15, 30]}
N_ST = 32


def fit_student(z: np.ndarray, nu_grid: List[float]) -> Dict[str, Any]:
    r"""Maximum-likelihood scaled Student-$t$ fit of standardised innovations.

    Args:
        z: Standardised innovations.
        nu_grid: Candidate degrees of freedom.

    Returns:
        ``{'nu', 'scale', 'gain_per_cell', 'by_nu'}`` where ``gain_per_cell`` is the mean
        log-likelihood gain in nats over the standard Gaussian density of $z$.
    """
    gauss = float(np.mean(stats.norm.logpdf(z)))
    by_nu: Dict[str, Any] = {}
    best = (-np.inf, None, None)
    for nu in nu_grid:
        result = optimize.minimize_scalar(lambda log_c: -np.mean(stats.t.logpdf(z / math.exp(log_c), nu)) + log_c,
                                          bounds=(-3.0, 3.0), method="bounded")
        value = -float(result.fun)
        by_nu[str(nu)] = {"scale": float(math.exp(result.x)), "gain_per_cell": value - gauss}
        if value > best[0]:
            best = (value, nu, float(math.exp(result.x)))
    return {"nu": best[1], "scale": best[2], "gain_per_cell": best[0] - gauss, "by_nu": by_nu}


@torch.no_grad()
def main(device: str, codes: List[int], max_cells: int, nu_grid: List[float]) -> Dict[str, Any]:
    """Collect the innovations and write the tail report.

    Args:
        device: Torch device string.
        codes: Split codes to read.
        max_cells: Cells kept per group for the fit.
        nu_grid: Candidate degrees of freedom.

    Returns:
        The report dictionary.
    """
    dev, cfg, task = S.load_model(device=device)
    model = task.orig_model
    store = P.SegmentStore(cfg, P.guids_with_code(codes))
    lk = model.forecast_likelihood_kwargs()
    floor = float(model.decoder.logvar_clamp[0])
    groups: Dict[str, List[np.ndarray]] = {key: [] for key in ("st", "ph", "st_floor", "st_free", "ph_floor", "ph_free")}
    for raw in store.ordered(32):
        m = I.matched_forward(task, model, task.transfer_batch_to_device(raw, dev, 0), anchor_stride=5)
        mu, logvar = model.decoder(I.at_anchors(m.out["mu_post"], m.anchors), persistence=m.out.get("persistence"))
        innovation, innovation_logvar = ar_innovation(m.target - mu, lk["ar_coef"], m.step_mask, logvar)
        z = innovation * torch.exp(-0.5 * innovation_logvar)
        counted = (lk["cell_mask"][None, None] * m.step_mask[..., None]) > 0
        on_floor = logvar <= floor + 1e-3
        for block, cols in (("st", slice(0, N_ST)), ("ph", slice(N_ST, None))):
            keep = counted[..., cols]
            groups[block].append(z[..., cols][keep].cpu().numpy())
            groups[f"{block}_floor"].append(z[..., cols][keep & on_floor[..., cols]].cpu().numpy())
            groups[f"{block}_free"].append(z[..., cols][keep & ~on_floor[..., cols]].cpu().numpy())
    rng = np.random.default_rng(0)
    report: Dict[str, Any] = {"codes": codes, "floor": floor}
    for name, parts in groups.items():
        z = np.concatenate(parts).astype(np.float64)
        entry: Dict[str, Any] = {"n_cells": int(z.size)}
        if z.size > max_cells:
            z = rng.choice(z, max_cells, replace=False)
        if z.size > 1000:
            entry.update({"mean_z2": float(np.mean(z ** 2)), "excess_kurtosis": float(stats.kurtosis(z)),
                          **{f"tail{k}": float(np.mean(np.abs(z) > k)) for k in (1, 2, 3)},
                          "student": fit_student(z, nu_grid)})
        report[name] = entry
        if "student" in entry:
            print(f"{name:9s} n={entry['n_cells']:9d} z2={entry['mean_z2']:.3f} kurt={entry['excess_kurtosis']:.2f} "
                  f"tails={entry['tail1']:.3f}/{entry['tail2']:.4f}/{entry['tail3']:.5f} "
                  f"nu={entry['student']['nu']} scale={entry['student']['scale']:.3f} "
                  f"gain/cell={entry['student']['gain_per_cell']:.4f}")
    out = S.OUT / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    (out / "innovation_tails.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=RUN_ARGS["device"])
    args = parser.parse_args()
    main(args.device, list(RUN_ARGS["codes"]), int(RUN_ARGS["max_cells"]), list(RUN_ARGS["nu_grid"]))
