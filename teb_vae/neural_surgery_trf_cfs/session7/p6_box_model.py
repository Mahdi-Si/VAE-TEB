r"""E8.2: the model-level value of a masked box (epoch 424, intervention, no training).

For every segment that holds a box event of :mod:`p6_box_events`, the network runs on the
segment's own loader batch. Only the UA stream $u$ (input 3) and its validity $v$ (input 6)
change, and only at the box runs; every other invalid run keeps its stored values and flags.

* ``strict``: stored values (the bridge crosses the box) and stored flags. This is rung 6.
* ``erased``: stored values, flags recomputed with the box runs removed from the invalid mask.
  The event is gone and the flags say "valid".
* ``raw``: values rebuilt by the builder's own chain with the box runs removed from the invalid
  mask, so the raw box reaches the transform; the flags of ``erased``.
* ``slew``: as ``raw``, but the raw UA of each box run passes a causal slew limiter first,
  $y[n] = y[n-1] + \operatorname{clip}(x[n] - y[n-1], \pm 1)$ per sample (4 units/s). The limiter
  starts at the run start and stops once $y$ meets $x$ again after the run.

The flags follow the loader's strict rule on the changed mask: step $k$ of channel $c$ is bad when
an invalid sample lies in raw samples $16k-15 \dots 16k$ or $\rho_c[k] > 0.1$, with $\rho$ rebuilt
from the changed mask and the level shifts that lie outside the box runs.

**Metric.** Both branches are decoded at the latent means and scored as in :mod:`p1_t4_model`.
The source gain is $G = \mathrm{NLL}_{\text{base}} - \mathrm{NLL}_{\text{full}}$. An anchor is affected
when its 38-step key window holds at least one step of a box run. Its timing class refers to the
box run that most recently started before it: ``inside``, ``0-60`` s, ``60-152`` s or ``later``
after the run end.

**Caveats.** The network never trained on a box, so a positive difference is a lower bound.
Inside a run, ``erased`` reads the first valid sample after the run through the bridge, so it has
a look-ahead advantage over ``raw`` and ``slew``.

Outputs go to ``out/phase6``: ``box_model_anchors.parquet`` and ``box_model.json``.

Edit :data:`RUN_ARGS` and run this file from the IDE.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import p1_common as C  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7.p0_dump import gather_steps, score_branch  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7.p1_rebuild import Rebuilder, builder_inputs  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7.p1_t4_model import (  # noqa: E402
    N_LAGS, STREAM_KEPT, TRIM, normalise_stream, strict_valid_full)
from teb_vae.neural_surgery_trf_cfs.session7.p6_box_events import OUT6  # noqa: E402

#: Run-button inputs. ``max_segments`` limits the run for a smoke test (``None`` = all).
RUN_ARGS: Dict[str, Any] = {"device": "cuda:0", "max_segments": None, "n_boot": 2000}

VARIANTS = ("strict", "erased", "raw", "slew")
#: Slew limit in UA units per raw sample (4 Hz).
SLEW_PER_SAMPLE = 1.0


def slew_limit(x: np.ndarray, runs: List[Tuple[int, int]]) -> np.ndarray:
    r"""Replace each box run by a causal slew-limited copy of the raw UA.

    Args:
        x: Raw UA ``(N,)``.
        runs: Box runs ``[(start, stop), ...]``, ``stop`` exclusive.

    Returns:
        A copy of ``x`` in which, from each run start, $y[n] = y[n-1] + \operatorname{clip}(x[n] -
        y[n-1], \pm 1)$ until $y$ meets $x$ after the run end.
    """
    y = x.copy()
    for a, b in runs:
        level = y[a - 1]
        n = a
        while n < x.size:
            level = level + float(np.clip(x[n] - level, -SLEW_PER_SAMPLE, SLEW_PER_SAMPLE))
            if n >= b and abs(level - x[n]) < 1e-9:
                break
            y[n] = level
            n += 1
    return y


def box_step_mask(runs: List[Tuple[int, int]], n_raw: int, steps: int) -> np.ndarray:
    """Steps of the trimmed grid that hold a raw sample of a box run.

    Args:
        runs: Box runs in raw samples.
        n_raw: Raw samples of the segment.
        steps: Steps of the untrimmed grid.

    Returns:
        ``(steps - 2 * TRIM,)`` bool.
    """
    ind = np.zeros(n_raw, dtype=bool)
    for a, b in runs:
        ind[a:b] = True
    padded = np.concatenate([np.zeros(S.RAW_PER_STEP - 1, dtype=bool), ind])
    hit = padded[: steps * S.RAW_PER_STEP].reshape(steps, S.RAW_PER_STEP).any(axis=1)
    return hit[TRIM:steps - TRIM]


@torch.no_grad()
def run(device: str, max_segments: Any) -> pd.DataFrame:
    """Run the four variants of every box segment and collect per-anchor rows.

    Args:
        device: Torch device.
        max_segments: Limit on segments (``None`` = all).

    Returns:
        The per-anchor table.
    """
    from train.data_module import GraphDataModule

    dev, cfg, task = S.load_model(device=device)
    model = task.orig_model.eval()
    lk = model.forecast_likelihood_kwargs()
    loader = GraphDataModule(cfg).test_dataloader()
    ds, collate = loader.dataset, loader.collate_fn
    event_mask = bool(task.hparams.get("source_event_mask", False))
    norm = lambda p: os.path.normcase(os.path.normpath(str(p)))  # noqa: E731
    file_of = {norm(p): i for i, p in enumerate(ds.paths)}
    where = {(fi, int(si)): k for k, (fi, si) in enumerate(ds.index_map)}
    rb = Rebuilder(device)

    events = pd.read_parquet(OUT6 / "box_events.parquet")
    segments = list(events.groupby(["cohort", "row"], sort=True))
    if max_segments is not None:
        segments = segments[: int(max_segments)]
    rows: List[Dict[str, Any]] = []
    checks = {"rebuild_stored_maxabs": 0.0, "loader_flag_mismatch_share": [], "nll_base_maxabs_across_variants": 0.0,
              "strict_vs_loader_G_maxabs": 0.0, "segments": len(segments), "segments_without_scored_anchor": 0,
              "source_event_mask": event_mask}
    # The loader opens the shards in single-writer-multiple-reader mode; a second handle on the
    # same file must use the same mode.
    handles = {cohort: h5py.File(path, "r", libver="latest", swmr=True) for cohort, path in S.SHARDS.items()}
    t0 = time.time()
    try:
        for si, ((cohort, row), part) in enumerate(segments):
            handle = handles[cohort]
            row = int(row)
            x = handle["up_raw"][row].astype(np.float64)
            bits = handle["up_quality"][row].astype(np.int64)
            invalid = (bits & S.UP_INVALID_BITS) != 0
            runs = [(int(a), int(b)) for a, b in zip(part["start"], part["stop"])]
            reduced = invalid.copy()
            for a, b in runs:
                reduced[a:b] = False
            shifts = [p for p in builder_inputs(handle, row, rb.cfg)["steps"]
                      if not any(a - S.FS <= p <= b + S.FS for a, b in runs)]
            built = rb.rebuild(np.stack([x, slew_limit(x, runs)]), np.stack([reduced, reduced]), shifts=[shifts, shifts])
            if si < 5:
                stored = rb.rebuild(x[None], invalid[None], shifts=[builder_inputs(handle, row, rb.cfg)["steps"]])
                checks["rebuild_stored_maxabs"] = max(checks["rebuild_stored_maxabs"],
                                                      float(np.abs(stored["up_st"][0] - handle["up_st"][row]).max()))
            v_reduced = strict_valid_full(reduced, built["up_st_rho"][0], built["up_ph_rho"][0])
            v_stored = strict_valid_full(invalid, handle["up_st_rho"][row], handle["up_ph_rho"][row])
            u_raw = normalise_stream(ds, built["up_st"][0], built["up_ph"][0])
            u_slew = normalise_stream(ds, built["up_st"][1], built["up_ph"][1])

            item = ds[where[(file_of[norm(S.SHARDS[cohort])], row)]]
            batch = task.transfer_batch_to_device(collate([item] * len(VARIANTS)), dev, 0)
            torch.manual_seed(0)
            inputs, out0, target, step_mask, anchors, anchor_valid = common.forward_parts(task, model, batch)
            u_stored = inputs[2][0].cpu().numpy()
            loader_valid = inputs[5][0].cpu().numpy().astype(bool)
            checks["loader_flag_mismatch_share"].append(
                float((loader_valid[:, STREAM_KEPT] != v_stored[:, STREAM_KEPT]).mean()))
            streams = {"strict": u_stored, "erased": u_stored, "raw": u_raw, "slew": u_slew}
            flags = {"strict": v_stored, "erased": v_reduced, "raw": v_reduced, "slew": v_reduced}
            new = list(inputs)
            new[2] = torch.from_numpy(np.stack([streams[v] for v in VARIANTS]).astype(np.float32)).to(dev)
            valid = torch.from_numpy(np.stack([flags[v] for v in VARIANTS])).to(dev)
            if event_mask:
                valid = valid & task._source_event_steps(batch, batch.weight).to(dev)[..., None]
            new[5] = valid
            torch.manual_seed(0)
            out = model(*new)
            gidx = anchors[:, :, None].expand(-1, -1, model.d_z)
            mu_b, lv_b = model.decoder(out["mu_prior"].gather(1, gidx), persistence=out["persistence"])
            mu_f, lv_f = model.decoder(out["mu_post"].gather(1, gidx), persistence=out["persistence"])
            sb = score_branch(mu_b, lv_b, target, step_mask, lk)
            sf = score_branch(mu_f, lv_f, target, step_mask, lk)
            mu_b0, lv_b0 = model.decoder(out0["mu_prior"].gather(1, gidx), persistence=out0["persistence"])
            mu_f0, lv_f0 = model.decoder(out0["mu_post"].gather(1, gidx), persistence=out0["persistence"])
            g0 = (score_branch(mu_b0, lv_b0, target, step_mask, lk)["nll"]
                  - score_branch(mu_f0, lv_f0, target, step_mask, lk)["nll"])
            keep = (((sb["n_st"] + sb["n_ph"]) > 0) & anchor_valid)[0].cpu().numpy()
            if not keep.any():
                checks["segments_without_scored_anchor"] += 1
                continue
            nll_b, nll_f = sb["nll"].cpu().numpy(), sf["nll"].cpu().numpy()
            gain = nll_b - nll_f
            checks["strict_vs_loader_G_maxabs"] = max(checks["strict_vs_loader_G_maxabs"],
                                                      float(np.abs(g0[0].cpu().numpy() - gain[0])[keep].max()))
            checks["nll_base_maxabs_across_variants"] = max(checks["nll_base_maxabs_across_variants"],
                                                            float(np.abs(nll_b - nll_b[:1])[:, keep].max()))
            kl = gather_steps(out["kld_per_t"], anchors).cpu().numpy()
            gate = gather_steps(out["event_gate"], anchors)[..., 0].cpu().numpy()
            hit = torch.from_numpy(box_step_mask(runs, x.size, C.N_STEPS)).to(dev)[None]
            cs = torch.cat([torch.zeros_like(hit[:, :1], dtype=torch.float32), torch.cumsum(hit.float(), 1)], 1)
            lo = (anchors[:1] - (N_LAGS - 1)).clamp_min(0)
            window = (gather_steps(cs, anchors[:1] + 1) - gather_steps(cs, lo))[0].cpu().numpy()
            anc = anchors[0].cpu().numpy()
            starts = np.array([a for a, _ in runs])
            order = np.argsort(starts)
            for a_i in np.flatnonzero(keep):
                n_a = 240 + 16 * int(anc[a_i])
                before = order[starts[order] <= n_a]
                if before.size == 0:
                    timing, stage2, length = "before", False, np.nan
                else:
                    last = part.iloc[int(before[-1])]
                    stage2, length = bool(last["stage2"]), float(last["len_s"])
                    if n_a < int(last["stop"]):
                        timing = "inside"
                    else:
                        d = (n_a - int(last["stop"])) / S.FS
                        timing = "0-60" if d < 60 else ("60-152" if d < 152 else "later")
                rec = {"cohort": cohort, "rec": f"{cohort}/{part['guid'].iloc[0]}", "row": row, "anchor": int(anc[a_i]),
                       "timing": timing, "affected": bool(window[a_i] > 0), "stage2": stage2, "len_s": length,
                       "nll_base": float(nll_b[0, a_i])}
                for vi, var in enumerate(VARIANTS):
                    rec[f"G_{var}"] = float(gain[vi, a_i])
                    rec[f"kl_{var}"] = float(kl[vi, a_i])
                    rec[f"gate_{var}"] = float(gate[vi, a_i])
                rows.append(rec)
            if si % 25 == 0:
                print(f"segment {si + 1}/{len(segments)}, {time.time() - t0:.0f} s", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    checks["loader_flag_mismatch_share"] = float(np.mean(checks["loader_flag_mismatch_share"]))
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT6 / "box_model_anchors.parquet")
    (OUT6 / "box_model_checks.json").write_text(json.dumps(checks, indent=1))
    print("checks", checks)
    return frame


def summarise(frame: pd.DataFrame, n_boot: int) -> Dict[str, Any]:
    """Mean gains and paired differences on the affected anchors.

    Args:
        frame: The per-anchor table.
        n_boot: Bootstrap resamples.

    Returns:
        The summary dictionary.
    """
    pairs = (("raw", "erased"), ("raw", "strict"), ("slew", "raw"), ("slew", "erased"), ("slew", "strict"),
             ("erased", "strict"))
    scopes = {"affected": frame["affected"]}
    for timing in ("inside", "0-60", "60-152", "later"):
        scopes[f"affected/{timing}"] = frame["affected"] & (frame["timing"] == timing)
    scopes["affected/after_0-152"] = frame["affected"] & frame["timing"].isin(["0-60", "60-152"])
    scopes["affected/second_stage"] = frame["affected"] & frame["stage2"]
    scopes["affected/first_stage"] = frame["affected"] & ~frame["stage2"]
    for cohort in sorted(frame["cohort"].unique()):
        scopes[f"affected/{cohort}"] = frame["affected"] & (frame["cohort"] == cohort)
    scopes["not_affected"] = ~frame["affected"]
    out: Dict[str, Any] = {}
    for name, keep in scopes.items():
        part = frame[keep]
        if part.empty:
            continue
        entry: Dict[str, Any] = {"n_anchors": int(len(part)), "n_rec": int(part["rec"].nunique())}
        for var in VARIANTS:
            entry[f"G_{var}"] = float(part[f"G_{var}"].mean())
            entry[f"kl_{var}"] = float(part[f"kl_{var}"].mean())
            entry[f"gate_{var}"] = float(part[f"gate_{var}"].mean())
        for a, b in pairs:
            entry[f"G_{a}-G_{b}"] = S.paired_bootstrap(part["rec"].to_numpy(), (part[f"G_{a}"] - part[f"G_{b}"]).to_numpy(),
                                                       n_resamples=n_boot)
        out[name] = entry
    return out


def main(device: str, max_segments: Any, n_boot: int) -> Dict[str, Any]:
    """Run the intervention and write the summary.

    Args:
        device: Torch device.
        max_segments: Limit on segments (``None`` = all).
        n_boot: Bootstrap resamples.

    Returns:
        The summary dictionary.
    """
    OUT6.mkdir(parents=True, exist_ok=True)
    frame = run(device, max_segments)
    summary = summarise(frame, n_boot)
    (OUT6 / "box_model.json").write_text(json.dumps(summary, indent=1))
    for name, entry in summary.items():
        line = f"{name}: n={entry['n_anchors']} rec={entry['n_rec']} | " + " ".join(
            f"G_{v}={entry[f'G_{v}']:.2f}" for v in VARIANTS)
        print(line)
        for key in ("G_raw-G_erased", "G_raw-G_strict", "G_slew-G_raw", "G_slew-G_strict", "G_erased-G_strict"):
            print(f"   {key}: {S.format_ci(entry[key], 2)}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default=RUN_ARGS["device"])
    parser.add_argument("--max-segments", type=int, default=RUN_ARGS["max_segments"])
    parser.add_argument("--n-boot", type=int, default=RUN_ARGS["n_boot"])
    args = parser.parse_args()
    main(args.device, args.max_segments, args.n_boot)
