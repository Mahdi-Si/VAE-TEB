r"""E2.4: the tolerance rule of :mod:`s7_tolerance` applied to the epoch-424 weights, no training.

For each tolerance $T_{\mathrm{tol}}$ the script replaces the adapter flags and the key mask by
their tolerant versions, re-runs the source pathway and scores the full branch on the same
anchors as the matched forward. The per-anchor table holds, per arm, the NLL, the KL and the
gate, and the covariates that the pre-registered comparisons need: the reward as trained and
without the UA rule, the invalid raw time in the lag window, and whether a short run is running
at the anchor.

The table goes to ``out/phase2/tolerance_anchors.parquet`` and the summary on split codes 0 and
2 to ``out/phase2/tolerance_summary.json``. Edit :data:`RUN_ARGS` and run this file from the IDE.
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
from teb_vae.neural_surgery_trf_cfs.session7 import s7_tolerance as T  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7.p2_surgery import window_invalid_seconds  # noqa: E402

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {
    "device": "cuda:0",
    "anchor_stride": 5,
    "limit": None,
    "collect": True,
    "tolerances_s": [8, 16, 32, 64],
    "codes": [0, 2],
}
OUT = S.OUT / "phase2"


def channel_windows(model: Any) -> torch.Tensor:
    """Causal window of each kept UA channel in steps, from the builder's stored warm-up.

    Args:
        model: The network.

    Returns:
        ``(C_u,)`` long tensor.
    """
    keep = model.source_gate.keep_index.cpu().numpy()
    return torch.as_tensor(T.declared_windows()[keep], dtype=torch.long)


@torch.no_grad()
def collect(device: str, anchor_stride: int, limit: Any, tolerances: List[int]) -> pd.DataFrame:
    r"""Run the matched forward and every tolerance arm; one row per valid anchor.

    Args:
        device: Torch device string.
        anchor_stride: Spacing between decoded anchors.
        limit: Number of batches, or ``None``.
        tolerances: The $T_{\mathrm{tol}}$ values in seconds.

    Returns:
        The per-anchor table.
    """
    dev, cfg, task = S.load_model(device=device)
    model = task.orig_model
    windows = channel_windows(model)
    detector = task._event_detector()
    reach = int(task.hparams.get("event_weight_reach_steps"))
    cohort_of = S.cohort_of_guids()
    frames: List[pd.DataFrame] = []
    for index, batch in enumerate(S.iter_batches(task, cfg, dev, limit=limit)):
        m = I.matched_forward(task, model, batch, anchor_stride=anchor_stride)
        anchors = m.anchors
        up_valid = batch["up_valid"].to(torch.bool)
        valid = m.anchor_valid.cpu().numpy().reshape(-1)
        columns: Dict[str, np.ndarray] = {}

        def put(name: str, tensor: torch.Tensor) -> None:
            columns[name] = tensor.detach().float().cpu().numpy().reshape(-1)[valid]

        put("nll_base", I.block_nll(model, m, m.out["mu_prior"]))
        arms = {"matched": I.reattend(model, m)}
        ones = torch.ones_like(m.quality)
        arms["flags_valid"] = I.reattend(model, m, quality=ones,
                                         key_mask=m.lag_mask.to(torch.bool)[None].expand_as(m.key_mask))
        for tol in tolerances:
            flags = T.tolerant_flags(m.quality > 0.5, up_valid, windows, float(tol)).to(m.quality.dtype)
            arms[f"tol{tol}"] = I.reattend(model, m, quality=flags,
                                           key_mask=I.key_mask_from_flags(model, flags, m.lag_mask))
            put(f"running{tol}", I.at_anchors(T.running_steps(up_valid, float(tol)).float(), anchors))
            put(f"flag_share{tol}", 1.0 - I.at_anchors(flags, anchors).mean(dim=-1))
        removed = T.removed_keys(up_valid, int(m.lag_mask.shape[-1]))
        arms["hardonly"] = I.reattend(model, m, removed=removed)
        put("keys_removed", I.at_anchors(removed & m.lag_mask.to(torch.bool)[None], anchors).float().sum(dim=-1))
        for tol in tolerances:
            flags = T.tolerant_flags(m.quality > 0.5, up_valid, windows, float(tol),
                                     guard_running=False).to(m.quality.dtype)
            arms[f"tolhard{tol}"] = I.reattend(model, m, quality=flags, removed=removed,
                                               key_mask=I.key_mask_from_flags(model, flags, m.lag_mask))
        for name, arm in arms.items():
            put(f"nll_{name}", I.block_nll(model, m, arm["mu_post"]))
            put(f"kl_{name}", I.anchor_kl(model, m, arm["mu_post"], arm["logvar_post"]))
            put(f"g_{name}", I.at_anchors(arm["gate_learned"], anchors)[..., 0])
        put("keys_invalid", I.at_anchors(m.lag_mask.to(torch.bool)[None] & ~m.key_mask, anchors).float().sum(dim=-1))
        put("raw_invalid_s_window", window_invalid_seconds(up_valid, anchors, 38))
        put("inside_gap", (~up_valid).gather(1, (anchors + 1) * S.RAW_PER_STEP - 1).float())
        _, weight = task._build_raw_target(batch)
        for name, raw_valid in (("omega_trained", up_valid), ("omega_free", None)):
            omega = contraction_anchor_weight(batch["up"], weight, decimation=S.RAW_PER_STEP, reach_steps=reach,
                                              floor=0.0, detector=detector, raw_valid=raw_valid)
            put(name, I.at_anchors(omega, anchors))
        for tol in tolerances:
            omega = contraction_anchor_weight(batch["up"], weight, decimation=S.RAW_PER_STEP, reach_steps=reach,
                                              floor=0.0, detector=detector,
                                              raw_valid=T.tolerant_reward_validity(up_valid, float(tol)))
            omega = omega * (~T.running_steps(up_valid, float(tol))).to(omega.dtype)
            put(f"omega_tol{tol}", I.at_anchors(omega, anchors))
        count = anchors.shape[1]
        guids = [S.guid_text(g) for g in batch["guid"]]
        columns["guid"] = np.repeat(np.asarray(guids, dtype=object), count)[valid]
        columns["epoch"] = np.repeat(batch["epoch"].float().cpu().numpy(), count)[valid]
        columns["anchor"] = anchors.cpu().numpy().reshape(-1)[valid]
        frames.append(pd.DataFrame(columns))
        if index % 20 == 0:
            print(f"batch {index}", flush=True)
    table = pd.concat(frames, ignore_index=True)
    table["cohort"] = table["guid"].map(cohort_of)
    table["code"] = table["guid"].map(S.guid_hash)
    return table


def summarise(table: pd.DataFrame, codes: List[int], tolerances: List[int]) -> Dict[str, Any]:
    """Compute the pre-registered comparisons of E2.4.

    Args:
        table: The per-anchor table.
        codes: Split codes the summary may read.
        tolerances: The tolerance arms.

    Returns:
        The summary dictionary.
    """
    table = table[table["code"].isin(codes)]
    dropped = (table["omega_trained"] <= 0.5) & (table["omega_free"] > 0.5)
    has_gap = table["raw_invalid_s_window"] > 0
    scopes = {
        "all": np.ones(len(table), dtype=bool),
        "omega_trained": (table["omega_trained"] > 0.5).to_numpy(),
        "dropped_ended": (dropped & has_gap & (table["inside_gap"] == 0)).to_numpy(),
        "dropped_inside": (dropped & (table["inside_gap"] == 1)).to_numpy(),
        "inside_gap_any": (table["inside_gap"] == 1).to_numpy(),
        "omega_free0_gap": ((table["omega_free"] <= 0.5) & has_gap).to_numpy(),
    }
    report: Dict[str, Any] = {"n_anchors": int(len(table)), "n_recordings": int(table["guid"].nunique())}
    for member, rows_mask in (("all", np.ones(len(table), dtype=bool)),
                              ("unseen", (table["cohort"] != "healthy").to_numpy()),
                              ("possibly_seen", (table["cohort"] == "healthy").to_numpy())):
        block: Dict[str, Any] = {}
        for scope, mask in scopes.items():
            rows = table[mask & rows_mask]
            entry: Dict[str, Any] = {"n": int(len(rows)), "g_matched": float(rows["g_matched"].mean()),
                                     "gain_matched": S.paired_bootstrap(rows["guid"].to_numpy(),
                                                                        (rows["nll_base"] - rows["nll_matched"]).to_numpy())}
            for arm in ["flags_valid", "hardonly"] + [f"tolhard{t}" for t in tolerances]:
                entry[f"dnll_{arm}"] = S.paired_bootstrap(rows["guid"].to_numpy(),
                                                          (rows[f"nll_{arm}"] - rows["nll_matched"]).to_numpy())
                entry[f"g_{arm}"] = float(rows[f"g_{arm}"].mean())
                entry[f"kl_{arm}"] = float(rows[f"kl_{arm}"].mean())
            block[scope] = entry
        block["reward_share"] = {"trained": float(table.loc[rows_mask, "omega_trained"].mean()),
                                 "free": float(table.loc[rows_mask, "omega_free"].mean()),
                                 **{f"tol{t}": float(table.loc[rows_mask, f"omega_tol{t}"].mean()) for t in tolerances}}
        report[member] = block
    return report


def main(device: str, anchor_stride: int, limit: Any, do_collect: bool, tolerances: List[int],
         codes: List[int]) -> Dict[str, Any]:
    """Collect the table (optional), write and print the summary.

    Args:
        device: Torch device string.
        anchor_stride: Spacing between decoded anchors.
        limit: Number of batches, or ``None``.
        do_collect: Run the model; ``False`` reads the stored table.
        tolerances: The tolerance arms in seconds.
        codes: Split codes the summary may read.

    Returns:
        The summary dictionary.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tolerance_anchors.parquet"
    if do_collect:
        collect(device, anchor_stride, limit, tolerances).to_parquet(path)
    report = summarise(pd.read_parquet(path), codes, tolerances)
    (OUT / "tolerance_summary.json").write_text(json.dumps(report, indent=1))
    for member in ("all", "unseen", "possibly_seen"):
        print("==", member, "reward share", {k: round(v, 4) for k, v in report[member]["reward_share"].items()})
        for scope, entry in report[member].items():
            if scope == "reward_share":
                continue
            line = f"  {scope:16s} n={entry['n']:6d} G={entry['gain_matched']['mean']:.2f} g={entry['g_matched']:.2f} |"
            for arm in ["flags_valid", "hardonly"] + [f"tolhard{t}" for t in tolerances]:
                line += f" {arm}: {S.format_ci(entry['dnll_' + arm], 2).split(' (')[0]} g={entry['g_' + arm]:.2f} |"
            print(line)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=RUN_ARGS["device"])
    parser.add_argument("--anchor-stride", type=int, default=RUN_ARGS["anchor_stride"])
    parser.add_argument("--limit", type=int, default=RUN_ARGS["limit"])
    parser.add_argument("--no-collect", action="store_true", default=not RUN_ARGS["collect"])
    args = parser.parse_args()
    main(args.device, args.anchor_stride, args.limit, not args.no_collect,
         list(RUN_ARGS["tolerances_s"]), list(RUN_ARGS["codes"]))
