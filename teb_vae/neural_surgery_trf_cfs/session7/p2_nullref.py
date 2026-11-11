r"""E2.7: a posterior defined relative to the source-null posterior, as an intervention.

At epoch 424 the posterior moves off the prior by about 1 nat of KL even when every UA value is
zero. That part of the KL is not information from the source. The null-referenced posterior
removes it by construction. With $q$ the matched posterior and $q^{\varnothing}$ the posterior
rebuilt against a zeroed source stream with the same validity flags,

$$\mu' = \mu^p + (\mu^q - \mu^{q,\varnothing}), \qquad
  \log\sigma'^2 = \log\sigma^{2,p} + (\log\sigma^{2,q} - \log\sigma^{2,q,\varnothing}).$$

A zeroed source then gives $q' = p$ exactly, so the source-null KL is 0 and every nat of
$\mathrm{KL}(q' \Vert p)$ depends on the UA values.

The script scores the matched posterior, the null posterior and the null-referenced posterior on
the same anchors with mean decoding, and reports the source gain and the KL on all anchors, on
rewarded anchors and on unrewarded anchors, for split codes 0 and 2.

Edit :data:`RUN_ARGS` and run this file from the IDE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.lag_attn_rws.event_weight import contraction_anchor_weight  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7_interventions as I  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {"device": "cuda:0", "anchor_stride": 5, "limit": None, "collect": True,
                            "codes": [0, 2]}
OUT = S.OUT / "phase2"


@torch.no_grad()
def collect(device: str, anchor_stride: int, limit: Any) -> pd.DataFrame:
    """Score the matched, the null and the null-referenced posterior; one row per valid anchor.

    Args:
        device: Torch device string.
        anchor_stride: Spacing between decoded anchors.
        limit: Number of batches, or ``None``.

    Returns:
        The per-anchor table.
    """
    dev, cfg, task = S.load_model(device=device)
    model = task.orig_model
    lo, hi = (float(v) for v in model.logvar_clamp)
    detector = task._event_detector()
    reach = int(task.hparams.get("event_weight_reach_steps"))
    cohort_of = S.cohort_of_guids()
    frames: List[pd.DataFrame] = []
    for index, batch in enumerate(S.iter_batches(task, cfg, dev, limit=limit)):
        m = I.matched_forward(task, model, batch, anchor_stride=anchor_stride)
        anchors, out = m.anchors, m.out
        matched = I.reattend(model, m)
        null = I.reattend(model, m, source=torch.zeros_like(m.source))
        mu_ref = out["mu_prior"] + matched["mu_post"] - null["mu_post"]
        logvar_ref = (out["logvar_prior"] + matched["logvar_post"] - null["logvar_post"]).clamp(lo, hi)
        valid = m.anchor_valid.cpu().numpy().reshape(-1)
        columns: Dict[str, np.ndarray] = {}

        def put(name: str, tensor: torch.Tensor) -> None:
            columns[name] = tensor.detach().float().cpu().numpy().reshape(-1)[valid]

        put("nll_base", I.block_nll(model, m, out["mu_prior"]))
        for name, mu, logvar in (("matched", matched["mu_post"], matched["logvar_post"]),
                                 ("null", null["mu_post"], null["logvar_post"]),
                                 ("nullref", mu_ref, logvar_ref)):
            put(f"nll_{name}", I.block_nll(model, m, mu))
            put(f"kl_{name}", I.anchor_kl(model, m, mu, logvar))
        put("g_matched", I.at_anchors(matched["gate_learned"], anchors)[..., 0])
        put("g_null", I.at_anchors(null["gate_learned"], anchors)[..., 0])
        _, weight = task._build_raw_target(batch)
        omega = contraction_anchor_weight(batch["up"], weight, decimation=S.RAW_PER_STEP, reach_steps=reach,
                                          floor=0.0, detector=detector, raw_valid=batch["up_valid"].to(torch.bool))
        put("omega_trained", I.at_anchors(omega, anchors))
        count = anchors.shape[1]
        guids = [S.guid_text(g) for g in batch["guid"]]
        columns["guid"] = np.repeat(np.asarray(guids, dtype=object), count)[valid]
        frames.append(pd.DataFrame(columns))
        if index % 20 == 0:
            print(f"batch {index}", flush=True)
    table = pd.concat(frames, ignore_index=True)
    table["cohort"] = table["guid"].map(cohort_of)
    table["code"] = table["guid"].map(S.guid_hash)
    return table


def main(device: str, anchor_stride: int, limit: Any, do_collect: bool, codes: List[int]) -> Dict[str, Any]:
    """Collect the table (optional), write and print the summary.

    Args:
        device: Torch device string.
        anchor_stride: Spacing between decoded anchors.
        limit: Number of batches, or ``None``.
        do_collect: Run the model; ``False`` reads the stored table.
        codes: Split codes the summary may read.

    Returns:
        The summary dictionary.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "nullref_anchors.parquet"
    if do_collect:
        collect(device, anchor_stride, limit).to_parquet(path)
    table = pd.read_parquet(path)
    table = table[table["code"].isin(codes)]
    report: Dict[str, Any] = {}
    for member, rows in (("all", table), ("unseen", table[table["cohort"] != "healthy"]),
                         ("possibly_seen", table[table["cohort"] == "healthy"])):
        for scope, part in (("all", rows), ("omega1", rows[rows["omega_trained"] > 0.5]),
                            ("omega0", rows[rows["omega_trained"] <= 0.5])):
            guid = part["guid"].to_numpy()
            entry = {"n": int(len(part))}
            for name in ("matched", "null", "nullref"):
                entry[f"gain_{name}"] = S.paired_bootstrap(guid, (part["nll_base"] - part[f"nll_{name}"]).to_numpy())
                entry[f"kl_{name}"] = float(part[f"kl_{name}"].mean())
            entry["dnll_nullref_vs_matched"] = S.paired_bootstrap(guid, (part["nll_nullref"] - part["nll_matched"]).to_numpy())
            entry["dkl_nullref_vs_matched"] = S.paired_bootstrap(guid, (part["kl_nullref"] - part["kl_matched"]).to_numpy())
            entry["net_matched"] = S.paired_bootstrap(guid, (part["nll_base"] - part["nll_matched"] - part["kl_matched"]).to_numpy())
            entry["net_nullref"] = S.paired_bootstrap(guid, (part["nll_base"] - part["nll_nullref"] - part["kl_nullref"]).to_numpy())
            report[f"{member}/{scope}"] = entry
            print(f"{member + '/' + scope:22s} n={entry['n']:6d} | G matched {S.format_ci(entry['gain_matched'], 2).split(' (')[0]} "
                  f"null {S.format_ci(entry['gain_null'], 2).split(' (')[0]} nullref {S.format_ci(entry['gain_nullref'], 2).split(' (')[0]} | "
                  f"KL {entry['kl_matched']:.2f} / {entry['kl_null']:.2f} / {entry['kl_nullref']:.2f} | "
                  f"G-KL matched {entry['net_matched']['mean']:.2f} nullref {entry['net_nullref']['mean']:.2f}")
    (OUT / "nullref_summary.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=RUN_ARGS["device"])
    parser.add_argument("--anchor-stride", type=int, default=RUN_ARGS["anchor_stride"])
    parser.add_argument("--limit", type=int, default=RUN_ARGS["limit"])
    parser.add_argument("--no-collect", action="store_true", default=not RUN_ARGS["collect"])
    args = parser.parse_args()
    main(args.device, args.anchor_stride, args.limit, not args.no_collect, list(RUN_ARGS["codes"]))
