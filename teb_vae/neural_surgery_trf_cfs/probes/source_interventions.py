r"""Does the source gain come from the UA values or from the UA validity-mask pattern?

Originally ``probe3.py``. The posterior-mean gain
$\Delta = \mathrm{NLL}_{\mathrm{base}} - \mathrm{NLL}_{\mathrm{full}}(\mu^q)$ is measured under five
source interventions and binned by quintile of the FHR drop over the next 120 s:

* ``reference``: the recording's own UA and validity mask;
* ``zero_values``: UA values set to zero, mask kept;
* ``mask_all_valid``: UA values kept, every step marked valid;
* ``zero_values_all_valid``: both;
* ``shuffled_values_keep_mask``: another recording's UA values, own mask.

Finding on the epoch-524 checkpoint: the gain is carried by the UA values (zeroed or shuffled values
turn it negative), and marking every step valid raises it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from teb_vae.lag_attn_rws.nets.losses import raw_sample_score  # noqa: E402
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask  # noqa: E402
from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402

RUN_ARGS = {
    "checkpoint": common.CHECKPOINT,
    "config": common.CONFIG,
    "overrides": None,
    "stats": common.STATS,
    "max_batches": 40,
    "out": common.RESULTS / "session2" / "probe3.json",
}
ARMS = ["reference", "zero_values", "mask_all_valid", "zero_values_all_valid", "shuffled_values_keep_mask"]


def variants(inputs):
    """Build the five source interventions from one batch's forward inputs.

    Args:
        inputs: ``(y_st, y_ph, u, phase, stride, source_valid)`` as the task builds them.

    Returns:
        A mapping from arm name to the edited input list.
    """
    y_st, y_ph, u, ph, st, sv = inputs
    return {"reference": list(inputs),
            "zero_values": [y_st, y_ph, torch.zeros_like(u), ph, st, sv],
            "mask_all_valid": [y_st, y_ph, u, ph, st, torch.ones_like(sv)],
            "zero_values_all_valid": [y_st, y_ph, torch.zeros_like(u), ph, st, torch.ones_like(sv)],
            "shuffled_values_keep_mask": [y_st, y_ph, u.roll(1, 0), ph, st, sv]}


def main(args):
    """Measure the gain per arm and per FHR-drop quintile and write the JSON result.

    Args:
        args: The run arguments; see :data:`RUN_ARGS`.
    """
    torch.manual_seed(0)
    dev, cfg, task = common.setup(args["checkpoint"], args["config"], args["overrides"])
    m = task.orig_model
    dec, lk = m.decoder, m.forecast_likelihood_kwargs()
    with h5py.File(args["stats"]) as s:
        fm, fs = float(s["fhr/mean"][()]), float(s["fhr/std"][()])
    rec = {k: [] for k in ARMS + ["drop"]}
    for batch in common.batches(task, cfg, dev, args["max_batches"]):
        with torch.no_grad():
            inp = task._build_forward_inputs(batch)
            tf, w = task._build_raw_target(batch)
            ws = m.scored_weight(task._target_quality_weight(batch, w))
            for k, v in variants(inp).items():
                o = m(*v)
                anc, av = o["anchor_index"], o["anchor_valid"]
                tgt = m._build_forecast_target(tf, anc)
                sm, _ = forecast_mask(ws, m.geometry, coverage_floor=m.coverage_floor, anchors=anc, anchor_valid=av)
                sm = sm.to(tgt.dtype)
                M = sm[..., None] * lk["cell_mask"]
                gi = anc[:, :, None].expand(-1, -1, m.d_z)

                def nll(mu, lv):
                    return (raw_sample_score(mu, tgt, likelihood="gaussian_nll", logvar=lv, cell_mask=lk["cell_mask"],
                                             ar_coef=lk["ar_coef"], step_mask=sm) * M).sum((-1, -2))
                mf, lf = dec(o["mu_post"].gather(1, gi), persistence=o["persistence"])
                c = (sm.sum(-1) > 0) & av
                rec[k].append((nll(o["mu_base"], o["logvar_base"]) - nll(mf, lf))[c].cpu())
            B, A = anc.shape
            fhr = batch["fhr"].reshape(B, -1) * fs + fm
            fhr = torch.nn.functional.conv1d(fhr[:, None], torch.ones(1, 1, 16, device=dev) / 16,
                                             padding=8)[:, 0, :fhr.shape[1]]
            end = (anc + 1) * 16

            def seg(o0, o1):
                idx = (end[..., None] + torch.arange(o0, o1, device=dev)).clamp(0, fhr.shape[1] - 1)
                return fhr.gather(1, idx.reshape(B, -1)).reshape(B, A, -1)
            rec["drop"].append((seg(-240, 0).median(-1).values - seg(0, 480).min(-1).values)[c].cpu())
    R = {k: torch.cat(v).numpy() for k, v in rec.items()}
    q = np.quantile(R["drop"], [0, .2, .4, .6, .8, 1])
    out = {"n": int(len(R["drop"]))}
    for k in ARMS:
        out[k] = {"all": float(R[k].mean())}
        for i in range(5):
            s_ = (R["drop"] >= q[i]) & (R["drop"] <= q[i + 1])
            out[k][f"drop_q{i + 1}({q[i]:.0f}..{q[i + 1]:.0f})"] = float(R[k][s_].mean())
    Path(args["out"]).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args["out"], "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(RUN_ARGS)
