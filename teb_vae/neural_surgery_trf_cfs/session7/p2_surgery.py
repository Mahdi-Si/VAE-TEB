r"""Phase 2 surgery: gate clamps, validity-flag variants and source controls on the same anchors.

One matched forward per batch, then the source pathway is re-run under each arm
(:mod:`s7_interventions`). Every arm is scored on the same anchors with mean decoding, so a
difference between two arms is a paired difference.

**Arms.**

* ``matched``: the checkpoint as trained (per-channel flags from $\rho \le 0.1$).
* ``gate1`` / ``gate0``: the event gate clamped to 1 or 0.
* ``flags_valid``: every flag valid and every key usable.
* ``flags_slow_valid`` / ``flags_fastmid_valid``: the flags of the slow third of the UA channels
  (by causal warm-up) set valid, or the flags of the other two thirds.
* ``flags_any``: the all-channel rule written into every channel.
* ``null`` / ``null_valid``: UA values zeroed, flags kept or set valid.
* ``shuffle``: UA values and flags of another recording of the batch.
* ``shift15`` / ``shift38``: the UA stream delayed by 15 or 38 steps (60 s, 152 s), flags moved
  with it.

**Anchor covariates.** The strict key mask, the raw UA validity over the 152-s lag window and
over the last 60 s, and the reward $\omega$ built two ways: as trained (a contraction that
touches an invalid raw sample is dropped) and without that rule.

The table goes to ``out/phase2/surgery_anchors.parquet``. :func:`summarise` then reports each
pre-registered comparison on split codes 0 and 2 with recording-level bootstrap intervals.

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
RUN_ARGS: Dict[str, Any] = {
    "device": "cuda:0",
    "anchor_stride": 5,
    "limit": None,              # number of batches; None reads every shard
    "collect": True,            # False: only re-summarise an existing table
    "codes": [0, 2],            # split codes the summary may read
}

OUT = S.OUT / "phase2"
ARMS = ["matched", "gate1", "gate0", "flags_valid", "flags_slow_valid", "flags_fastmid_valid",
        "flags_any", "null", "null_valid", "shuffle", "shift15", "shift38"]


def delay(values: torch.Tensor, steps: int) -> torch.Tensor:
    """Delay a per-step tensor along the step axis, repeating the first step at the start.

    Args:
        values: ``(B, T, ...)``.
        steps: Delay in steps; position $t$ of the result holds position $t - $ ``steps``.

    Returns:
        The delayed tensor, same shape.
    """
    head = values[:, :1].expand(-1, steps, *values.shape[2:])
    return torch.cat([head, values[:, :-steps]], dim=1)


def window_invalid_seconds(up_valid: torch.Tensor, anchors: torch.Tensor, steps: int,
                           decimation: int = S.RAW_PER_STEP) -> torch.Tensor:
    r"""Seconds of invalid raw UA in the last ``steps`` feature steps up to each anchor.

    The window of anchor $t$ covers raw samples $[D(t - \text{steps} + 1),\, D(t + 1))$.

    Args:
        up_valid: Raw UA validity ``(B, L)``, ``True`` = valid.
        anchors: Anchor steps ``(B, A)``.
        steps: Window length in feature steps.
        decimation: Raw samples per feature step $D$.

    Returns:
        ``(B, A)`` seconds.
    """
    bad = (~up_valid).to(torch.float32)
    cumulative = torch.cat([bad.new_zeros(bad.shape[0], 1), bad.cumsum(dim=1)], dim=1)
    end = (anchors + 1) * decimation
    start = (end - steps * decimation).clamp_min(0)
    return (cumulative.gather(1, end) - cumulative.gather(1, start)) / S.FS


@torch.no_grad()
def collect(device: str, anchor_stride: int, limit: Any) -> pd.DataFrame:
    """Run every arm on every batch and return the per-anchor table.

    Args:
        device: Torch device string.
        anchor_stride: Spacing between decoded anchors.
        limit: Number of batches, or ``None``.

    Returns:
        One row per valid anchor.
    """
    dev, cfg, task = S.load_model(device=device)
    model = task.orig_model
    cohort_of = S.cohort_of_guids()
    waits = np.asarray(model.source_warmup_steps)
    order = np.argsort(waits, kind="stable")
    slow = torch.zeros(len(waits), dtype=torch.bool, device=dev)
    slow[torch.as_tensor(order[-(len(waits) // 3):].copy(), device=dev)] = True
    detector = task._event_detector()
    reach = int(task.hparams.get("event_weight_reach_steps"))
    frames: List[pd.DataFrame] = []
    for index, batch in enumerate(S.iter_batches(task, cfg, dev, limit=limit)):
        m = I.matched_forward(task, model, batch, anchor_stride=anchor_stride)
        out, anchors = m.out, m.anchors
        ones = torch.ones_like(m.quality)
        open_keys = m.lag_mask.to(torch.bool)[None].expand(m.key_mask.shape[0], -1, -1)
        step_ok = m.quality.min(dim=-1).values                                   # (B, T)
        guids = [S.guid_text(g) for g in batch["guid"]]
        size = len(guids)
        roll = torch.roll(torch.arange(size, device=dev), shifts=size // 2)
        donor_differs = torch.tensor([guids[i] != guids[int(roll[i])] for i in range(size)], device=dev)

        arms: Dict[str, Dict[str, torch.Tensor]] = {
            "matched": I.reattend(model, m),
            "gate1": I.reattend(model, m, gate_fn=torch.ones_like),
            "gate0": I.reattend(model, m, gate_fn=torch.zeros_like),
            "flags_valid": I.reattend(model, m, quality=ones, key_mask=open_keys),
            "flags_slow_valid": I.reattend(
                model, m, quality=torch.where(slow[None, None], ones, m.quality),
                key_mask=I.key_mask_from_flags(model, torch.where(slow[None, None], ones, m.quality), m.lag_mask)),
            "flags_fastmid_valid": I.reattend(
                model, m, quality=torch.where(slow[None, None], m.quality, ones),
                key_mask=I.key_mask_from_flags(model, torch.where(slow[None, None], m.quality, ones), m.lag_mask)),
            "flags_any": I.reattend(model, m, quality=step_ok[..., None].expand_as(m.quality)),
            "null": I.reattend(model, m, source=torch.zeros_like(m.source)),
            "null_valid": I.reattend(model, m, source=torch.zeros_like(m.source), quality=ones, key_mask=open_keys),
            "shuffle": I.reattend(model, m, source=m.source[roll], quality=m.quality[roll], key_mask=m.key_mask[roll]),
        }
        for steps in (15, 38):
            moved = delay(m.quality, steps)
            arms[f"shift{steps}"] = I.reattend(model, m, source=delay(m.source, steps), quality=moved,
                                               key_mask=I.key_mask_from_flags(model, moved, m.lag_mask))

        columns: Dict[str, np.ndarray] = {}
        valid = m.anchor_valid.cpu().numpy().reshape(-1)

        def put(name: str, tensor: torch.Tensor) -> None:
            columns[name] = tensor.detach().float().cpu().numpy().reshape(-1)[valid]

        nll_base = I.block_nll(model, m, out["mu_prior"], per_block=True)
        put("nll_base", nll_base[..., 0]); put("nll_base_st", nll_base[..., 1]); put("nll_base_ph", nll_base[..., 2])
        for name, arm in arms.items():
            nll = I.block_nll(model, m, arm["mu_post"], per_block=(name == "matched"))
            if name == "matched":
                put("nll_matched_st", nll[..., 1]); put("nll_matched_ph", nll[..., 2])
                nll = nll[..., 0]
            if name == "shuffle":
                nll = torch.where(donor_differs[:, None], nll, torch.full_like(nll, float("nan")))
            put(f"nll_{name}", nll)
            put(f"kl_{name}", I.anchor_kl(model, m, arm["mu_post"], arm["logvar_post"]))
            put(f"g_{name}", I.at_anchors(arm["gate_learned"], anchors)[..., 0])
        put("invalid_mass", I.at_anchors(arms["matched"]["invalid_mass"], anchors).mean(dim=-1))
        put("keys_invalid", I.at_anchors(open_keys & ~m.key_mask, anchors).float().sum(dim=-1))
        put("flag_invalid_share", 1.0 - I.at_anchors(m.quality, anchors).mean(dim=-1))
        put("flag_invalid_share_slow", 1.0 - I.at_anchors(m.quality, anchors)[..., slow].mean(dim=-1))
        up_valid = batch["up_valid"].to(torch.bool)
        put("raw_invalid_s_window", window_invalid_seconds(up_valid, anchors, 38))
        put("raw_invalid_s_60", window_invalid_seconds(up_valid, anchors, 15))
        put("inside_gap", (~up_valid).gather(1, (anchors + 1) * S.RAW_PER_STEP - 1).float())

        _, weight = task._build_raw_target(batch)
        for name, raw_valid in (("omega_trained", up_valid), ("omega_free", None)):
            omega = contraction_anchor_weight(batch["up"], weight, decimation=S.RAW_PER_STEP,
                                              reach_steps=reach, floor=0.0, detector=detector,
                                              raw_valid=raw_valid)
            put(name, I.at_anchors(omega, anchors))
        count = anchors.shape[1]
        columns["guid"] = np.repeat(np.asarray(guids, dtype=object), count)[valid]
        columns["epoch"] = np.repeat(batch["epoch"].float().cpu().numpy(), count)[valid]
        columns["anchor"] = anchors.cpu().numpy().reshape(-1)[valid]
        frames.append(pd.DataFrame(columns))
        if index % 20 == 0:
            print(f"batch {index}: {sum(len(f) for f in frames)} anchors", flush=True)
    table = pd.concat(frames, ignore_index=True)
    table["cohort"] = table["guid"].map(cohort_of)
    table["code"] = table["guid"].map(S.guid_hash)
    table["seconds_to_delivery"] = -table["epoch"] - (table["anchor"] + 1) * S.STEP_S
    return table


def anchor_class(table: pd.DataFrame) -> pd.Series:
    """Label each anchor by the validity of its lag window.

    Args:
        table: The per-anchor table.

    Returns:
        A series of ``'valid'``, ``'rho_only'``, ``'short_bridge'`` or ``'long_gap'``.
    """
    label = np.where(table["keys_invalid"] == 0, "valid",
                     np.where(table["raw_invalid_s_window"] == 0, "rho_only",
                              np.where(table["raw_invalid_s_window"] <= 16, "short_bridge", "long_gap")))
    # A fully valid key window with invalid raw samples cannot occur under the strict rule; keep
    # the label honest if it does.
    label = np.where((table["keys_invalid"] == 0) & (table["raw_invalid_s_window"] > 0), "valid_raw_gap", label)
    return pd.Series(label, index=table.index)


def paired(table: pd.DataFrame, arm: str, reference: str = "matched", column: str = "nll") -> Dict[str, float]:
    """Recording-level bootstrap of a paired per-anchor difference between two arms.

    Args:
        table: Rows to use.
        arm: The intervention arm.
        reference: The reference arm.
        column: ``'nll'``, ``'kl'`` or ``'g'``.

    Returns:
        The :func:`s7common.paired_bootstrap` dictionary for ``arm - reference``.
    """
    return S.paired_bootstrap(table["guid"].to_numpy(),
                              (table[f"{column}_{arm}"] - table[f"{column}_{reference}"]).to_numpy())


def summarise(table: pd.DataFrame, codes: List[int]) -> Dict[str, Any]:
    """Compute every pre-registered comparison on the allowed split codes.

    Args:
        table: The per-anchor table.
        codes: Split codes the summary may read.

    Returns:
        A nested dictionary of results.
    """
    table = table[table["code"].isin(codes)].copy()
    table["cls"] = anchor_class(table)
    table["gain"] = table["nll_base"] - table["nll_matched"]
    report: Dict[str, Any] = {"n_anchors": int(len(table)), "n_recordings": int(table["guid"].nunique()),
                              "codes": list(codes)}
    membership = {"all": table, "unseen": table[table["cohort"] != "healthy"],
                  "possibly_seen": table[table["cohort"] == "healthy"]}
    for member, rows in membership.items():
        block: Dict[str, Any] = {"n_anchors": int(len(rows))}
        block["class_share"] = rows["cls"].value_counts(normalize=True).round(4).to_dict()
        block["omega_share"] = {"trained": float(rows["omega_trained"].mean()), "free": float(rows["omega_free"].mean())}
        block["gate_on_share"] = float((rows["g_matched"] >= 0.2).mean())
        block["gate_open_share"] = float((rows["g_matched"] >= 0.8).mean())
        block["kl_share_gate_on"] = float(rows.loc[rows["g_matched"] >= 0.2, "kl_matched"].sum() / rows["kl_matched"].sum())
        gain_all = S.paired_bootstrap(rows["guid"].to_numpy(), rows["gain"].to_numpy())
        block["gain"] = gain_all
        block["kl"] = S.paired_bootstrap(rows["guid"].to_numpy(), rows["kl_matched"].to_numpy())
        # Controls (E2.3) on all anchors and on rewarded anchors.
        for scope_name, scope in (("all", rows), ("omega1", rows[rows["omega_trained"] > 0.5])):
            controls: Dict[str, Any] = {"gain": S.paired_bootstrap(scope["guid"].to_numpy(), scope["gain"].to_numpy())}
            for arm in ("null", "null_valid", "shuffle", "shift15", "shift38", "gate0", "gate1", "flags_valid"):
                keep = scope[np.isfinite(scope[f"nll_{arm}"])]
                controls[f"gain_{arm}"] = S.paired_bootstrap(
                    keep["guid"].to_numpy(), (keep["nll_base"] - keep[f"nll_{arm}"]).to_numpy())
                controls[f"kl_{arm}"] = S.paired_bootstrap(keep["guid"].to_numpy(), keep[f"kl_{arm}"].to_numpy())
            controls["kl_flag_part"] = S.paired_bootstrap(
                scope["guid"].to_numpy(), (scope["kl_null"] - scope["kl_null_valid"]).to_numpy())
            block[f"controls_{scope_name}"] = controls
        # Per class x omega: descriptive values and paired arm differences (E2.1, E2.2).
        per_class: Dict[str, Any] = {}
        for cls in ("valid", "rho_only", "short_bridge", "long_gap"):
            for omega_name, mask in (("omega1", rows["omega_trained"] > 0.5),
                                     ("omega0", rows["omega_trained"] <= 0.5),
                                     ("dropped", (rows["omega_trained"] <= 0.5) & (rows["omega_free"] > 0.5))):
                cell = rows[(rows["cls"] == cls) & mask]
                if cell["guid"].nunique() < 5:
                    continue
                entry: Dict[str, Any] = {
                    "n_anchors": int(len(cell)), "n_rec": int(cell["guid"].nunique()),
                    "g": float(cell["g_matched"].mean()), "kl": float(cell["kl_matched"].mean()),
                    "gain": S.paired_bootstrap(cell["guid"].to_numpy(), cell["gain"].to_numpy()),
                }
                for arm in ARMS[1:]:
                    keep = cell[np.isfinite(cell[f"nll_{arm}"])]
                    entry[f"dnll_{arm}"] = paired(keep, arm)
                    entry[f"dg_{arm}"] = float((keep[f"g_{arm}"] - keep["g_matched"]).mean())
                    entry[f"dkl_{arm}"] = float((keep[f"kl_{arm}"] - keep["kl_matched"]).mean())
                per_class[f"{cls}/{omega_name}"] = entry
        block["per_class"] = per_class
        # Clock test against time to delivery.
        bins = [0, 900, 1800, 3600, 7200, 14400, 1e9]
        rows = rows.assign(ttd=pd.cut(rows["seconds_to_delivery"], bins))
        block["by_time_to_delivery"] = {
            str(key): {"n": int(len(grp)), "kl": float(grp["kl_matched"].mean()), "kl_null": float(grp["kl_null"].mean()),
                       "kl_null_valid": float(grp["kl_null_valid"].mean()), "gain": float(grp["gain"].mean()),
                       "g": float(grp["g_matched"].mean()), "flag_invalid_share": float(grp["flag_invalid_share"].mean()),
                       "share_not_valid_class": float((grp["cls"] != "valid").mean())}
            for key, grp in rows.groupby("ttd", observed=True)}
        report[member] = block
    return report


def main(device: str, anchor_stride: int, limit: Any, do_collect: bool, codes: List[int]) -> Dict[str, Any]:
    """Collect the table (optional) and write the summary.

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
    path = OUT / "surgery_anchors.parquet"
    if do_collect:
        collect(device, anchor_stride, limit).to_parquet(path)
    report = summarise(pd.read_parquet(path), codes)
    (OUT / "surgery_summary.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=RUN_ARGS["device"])
    parser.add_argument("--anchor-stride", type=int, default=RUN_ARGS["anchor_stride"])
    parser.add_argument("--limit", type=int, default=RUN_ARGS["limit"])
    parser.add_argument("--no-collect", action="store_true", default=not RUN_ARGS["collect"])
    args = parser.parse_args()
    main(args.device, args.anchor_stride, args.limit, not args.no_collect, list(RUN_ARGS["codes"]))
