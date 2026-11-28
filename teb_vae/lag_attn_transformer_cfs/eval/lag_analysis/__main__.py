"""Command line of the lag analysis: one pass over the shards for every step that is not done yet."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# An IDE Run button runs this file as a script: put the repository root on sys.path (it is four levels
# up) so the absolute imports below resolve.
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from teb_vae.lag_attn_rws.eval.send_pack import pack_for_sending  # noqa: E402
from teb_vae.lag_attn_transformer_cfs.eval.lag_analysis import core, report  # noqa: E402
from teb_vae.lag_attn_transformer_cfs.eval.lag_analysis.steps import (  # noqa: E402
    DONOR_STEPS, LAG_STEPS, SLOW_DELAY_S, SPLIT_BY, STEPS, Acc,
)


def parse(argv=None):
    p = argparse.ArgumentParser(prog="python -m teb_vae.lag_attn_transformer_cfs.eval.lag_analysis", description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--shards", required=True, nargs="+", help="shard files or globs; all of them are evaluated as one set")
    p.add_argument("--stats", required=True, help="normalisation statistics the model was fitted with")
    p.add_argument("--out", required=True)
    p.add_argument("--config", help="resolved config of the run (loader settings); default: the file beside the checkpoint")
    p.add_argument("--steps", default=",".join(STEPS), help="comma-separated subset of: " + ", ".join(STEPS))
    p.add_argument("--max-guids", type=int, default=0, help="0 = every recording")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--k-samples", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="redo steps whose file exists; also drops saved progress")
    p.add_argument("--save-every", type=int, default=100, help="batches between two saves of the progress file")
    p.add_argument("--stop-after-batches", type=int, default=0, help="debug: stop the pass as a kill would, after this many batches")
    return p.parse_args(argv)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_progress(path, meta, accs, secs, batches, segments, elapsed):
    """Write the per-segment sums and counts of every pending step for the segments done so far, with
    what identifies the run. Written under a temporary name and renamed, so a kill cannot corrupt it."""
    arrays = {}
    for s, acc in accs.items():
        for k, v in acc.s.items():
            arrays[f"{s}|s|{k}"] = v[:segments]
            if acc.c[k] is not None:
                arrays[f"{s}|c|{k}"] = acc.c[k][:segments]
    state = {"meta": meta, "batches": batches, "segments": segments, "seconds": secs, "elapsed": elapsed,
             "binned": {s: sorted(acc.binned) for s, acc in accs.items()}}
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, state=json.dumps(state), **arrays)
    os.replace(tmp, path)


def load_progress(path, meta, accs):
    """Fill the accumulators from the progress file if it belongs to this run. Returns its state, or
    None with a printed reason."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            state = json.loads(str(z["state"]))
            diff = sorted(k for k in meta if state["meta"].get(k) != meta[k])
            if diff:
                print(f"progress file ignored: it was made with another {', '.join(diff)}", flush=True)
                return None
            for name in z.files:
                if name == "state":
                    continue
                s, kind, k = name.split("|", 2)
                full = np.zeros((accs[s].n,) + z[name].shape[1:])
                full[:state["segments"]] = z[name]
                (accs[s].s if kind == "s" else accs[s].c)[k] = full
            for s, acc in accs.items():
                acc.binned = set(state["binned"][s])
                acc.c.update({k: None for k in acc.s if k not in acc.c})
    except (OSError, ValueError, KeyError) as err:
        print(f"progress file ignored: it cannot be read ({err})", flush=True)
        for acc in accs.values():                 # a partial read must not be counted again by the fresh pass
            acc.s.clear()
            acc.c.clear()
            acc.binned = set()
        return None
    return state


def code_version():
    """Hash of the modules that define the numbers, so progress and step files of other code are never
    continued. This file counts up to its `RUN_ARGS` block: editing the Run-button values is not a code change."""
    here = Path(__file__).parent
    cli = (here / "__main__.py").read_bytes().split(b"\nRUN_ARGS = {")[0]
    return hashlib.sha256((here / "core.py").read_bytes() + (here / "steps.py").read_bytes() + cli).hexdigest()[:12]


def slow_channels(model, shard, device):
    """Declared source channels with a filter delay of 100 s or more in the scattering block, from the
    shard attribute `causal_delay_s` (a property of the filter bank, not of the data). None if unknown."""
    with h5py.File(shard, "r") as h:
        delay = h["up_st"].attrs.get("causal_delay_s") if "up_st" in h else None
        n_ph = h["up_ph"].shape[1] if "up_ph" in h else 0
    if delay is None or not getattr(model, "use_up_st", False):
        return None
    mask = torch.zeros(len(delay) + n_ph, dtype=torch.bool, device=device)
    mask[: len(delay)] = torch.as_tensor(np.asarray(delay) >= SLOW_DELAY_S)
    return mask


def mean_query(run, ds, rows, ghalf, batch_size):
    """Mean attention query of every head, separately over the two seeded halves of the recordings, from
    up to 256 evenly spaced segments. The static clamp gives a recording the profile of the other half."""
    task, model = run["task"], run["model"]
    pick = np.unique(np.linspace(0, len(rows) - 1, min(256, len(rows))).astype(int))
    loader = core.make_loader(ds, rows[pick], rows[pick], rows[pick], batch_size, 0, donors=False)
    got = []
    hook = model.lag_attn.W_q.register_forward_hook(lambda m, i, o: got.append(o))
    tot, cnt = None, np.zeros(2)
    for k, (b, _, _) in enumerate(loader):
        b = task.transfer_batch_to_device(b, run["device"], 0)
        with torch.no_grad():
            out = model(*task._build_forward_inputs(b))
        q = got.pop()[:, out["anchor_index"][0]].mean(1)                    # (B, M * d)
        half = ghalf[pick[k * batch_size:k * batch_size + len(q)]]
        tot = torch.zeros(2, q.shape[1], device=q.device) if tot is None else tot
        for h in (0, 1):
            tot[h] += q[torch.as_tensor(half == h, device=q.device)].sum(0)
            cnt[h] += (half == h).sum()
    hook.remove()
    q = tot / torch.as_tensor(np.maximum(cnt, 1), device=tot.device, dtype=tot.dtype)[:, None]
    return q if cnt.min() > 0 else q[[int(cnt.argmax())] * 2]              # one half empty (tiny runs): use the other


def main(argv=None):
    args = parse(argv)
    t_start = time.time()
    out = Path(args.out)
    (out / "steps").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    shards = sorted({f for pat in args.shards for f in (glob.glob(pat) or [pat])})
    task = core.load_task(args.checkpoint, device)
    model = task.orig_model
    var = core.variant(model)
    loader_cfg = core.loader_config(args.config, args.checkpoint)
    ds = core.open_dataset(loader_cfg, shards, args.stats)
    ix = core.build_index(ds)
    rows = core.select_rows(ix, args.max_guids, args.seed)
    guids, gidx = np.unique(ix["guid"][rows], return_inverse=True)
    sub_of = np.empty(len(guids), object)
    sub_of[gidx] = ix["subgroup"][rows]
    classes_known = all(core.class_of(s)[0] is not None for s in sub_of)

    # per-segment covariates: time-to-delivery bin of the anchor range, odd / even segment of the recording
    geo = model.geometry
    T = geo.raw_len // geo.decimation
    centre = 60.0 * float(ds.trim_minutes if getattr(ds, "trim_minutes", None) else 0.0) + core.STEP_S * (geo.warmup + T - geo.horizon - 1) / 2
    minutes = -(ix["epoch"][rows] + centre) / 60.0
    tbin = np.digitize(minutes, core.TIME_BINS) - 1
    tbin[(tbin < 0) | (tbin > 3)] = -1
    half = np.zeros(len(rows), int)
    for g in range(len(guids)):
        idx = np.flatnonzero(gidx == g)
        half[idx[np.argsort(ix["epoch"][rows][idx])]] = np.arange(len(idx)) % 2
    ghalf = np.array([core._rank(g, args.seed) % 2 for g in guids])[gidx]     # seeded half of the recording

    ck = Path(args.checkpoint)
    # Everything that changes a number: checkpoint, rows, shards, stats, loader settings, K, seed and code.
    loader_key = json.dumps(loader_cfg, sort_keys=True, default=str)
    key = hashlib.sha256(f"{sha256(ck)}|{','.join(map(str, rows))}|{sorted(shards)}|{sha256(args.stats)}|{loader_key}|"
                         f"{args.k_samples}|{args.seed}|{code_version()}".encode()).hexdigest()[:16]
    want = [s for s in STEPS if s in args.steps.split(",")]
    unknown = set(args.steps.split(",")) - set(STEPS)
    if unknown:
        raise SystemExit(f"unknown step(s) {sorted(unknown)}; steps: {', '.join(STEPS)}")
    # Every step file in the folder, requested or not, goes into one report: all must be of this run.
    for f in sorted((out / "steps").glob("*.npz")):
        with np.load(f) as z:
            same = "key" in z.files and str(z["key"]) == key
        if not same:
            if not args.force:
                raise SystemExit(f"{f} was written for another checkpoint, shard set, stats, loader setting, code or "
                                 "setting; one report cannot mix them. Use --force (deletes it) or another --out.")
            print(f"{f.name}: written by another run, deleted (--force)", flush=True)
            f.unlink()
    skipped, pending = {}, []
    for s in STEPS:
        f = out / "steps" / f"{s}.npz"
        if s not in want:
            if not f.exists():
                skipped[s] = "not requested (--steps)"
        elif s in LAG_STEPS and not var["lag"]:
            skipped[s] = "needs the lag attention; this checkpoint has none (source_path = irf)"
        elif s == "irf" and not var["irf"]:
            skipped[s] = "needs the IRF head; this checkpoint has none"
        elif f.exists() and not args.force:
            print(f"[{s}] done before, skipped (use --force to redo)")
        else:
            pending.append(s)
    notes = []
    if not classes_known:
        notes.append("Shard names are not the eight subgroup names: class, delivery-mode and block rows are left out; every row is on block `all`.")

    info_path = out / "run_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() and not args.force else {}
    seconds = dict(info.get("seconds_per_step", {}))
    if pending:
        run = {"task": task, "model": model, "device": device, "K": args.k_samples, "seed": args.seed, **var,
               "ghalf": ghalf, "qbar": None, "slow": None, "scale": {}}
        donors = any(s in DONOR_STEPS for s in pending)
        run["donor_a"], run["donor_b"] = core.assign_donors(ix, rows, args.seed) if donors else (np.full(len(rows), -1),) * 2
        with h5py.File(args.stats, "r") as h:
            for f in ("fhr", "up"):
                # normalize_fields None: every field with statistics is normalised (CombinedHDF5Dataset).
                norm = f in h and (ds.normalize_fields is None or f in ds.normalize_fields)
                run["scale"][f] = (float(h[f]["mean"][()]), float(h[f]["std"][()])) if norm else (0.0, 1.0)
        if var["lag"]:
            run["slow"] = slow_channels(model, shards[0], device)
            if run["slow"] is None and "groups" in pending:
                notes.append("groups: slow-channel swap left out (no `causal_delay_s` attribute on `up_st`).")
        if "clamp" in pending:
            run["qbar"] = mean_query(run, ds, rows, ghalf, args.batch_size)
        accs = {s: Acc(len(rows), SPLIT_BY.get(s, ())) for s in pending}
        secs = dict.fromkeys(pending, 0.0)
        # Crash-safe progress. The batch order is the row order, the donors depend on the seed and the
        # rows, and the latent noise on the seed and the segment, so a resumed pass repeats nothing random.
        progress = out / "steps" / "progress.state"
        meta = {"checkpoint, shard set, stats, loader settings, --max-guids, --k-samples or --seed": key, "--batch-size": args.batch_size,
                "step list": pending, "code version": code_version()}
        if args.force and progress.exists():
            progress.unlink()
        state = load_progress(progress, meta, accs) or {"batches": 0, "segments": 0, "seconds": secs, "elapsed": 0.0}
        k0, done, before = state["batches"], state["segments"], state["elapsed"]
        secs = dict(state["seconds"])
        total = -(-len(rows) // args.batch_size)
        if k0:
            print(f"resuming after batch {k0}/{total} ({done}/{len(rows)} segments) from {progress}", flush=True)
        loader = core.make_loader(ds, rows[done:], run["donor_a"][done:], run["donor_b"][done:], args.batch_size, args.num_workers, donors)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        for k, batches in enumerate(loader, k0 + 1):
            n = len(batches[0].epoch)
            ctx = core.Ctx(run, batches, np.arange(done, done + n))
            for s in pending:
                t = time.perf_counter()
                STEPS[s](ctx, accs[s])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                secs[s] += time.perf_counter() - t
            done += n
            if k % args.save_every == 0 or k == total:
                el = time.time() - t0
                if k < total:
                    save_progress(progress, meta, accs, secs, k, done, before + el)
                print(f"batch {k}/{total} ({done}/{len(rows)} segments), {before + el:.0f} s elapsed, "
                      f"about {el / (k - k0) * (total - k):.0f} s left", flush=True)
            if k == args.stop_after_batches:
                raise SystemExit(f"stopped after batch {k} (--stop-after-batches); run the same command to resume")
        for s in pending:
            arrays = accs[s].finish(gidx, len(guids), tbin, half)
            np.savez_compressed(out / "steps" / f"{s}.npz", guid=guids.astype(str), subgroup=sub_of.astype(str), key=key, **arrays)
            seconds[s] = round(secs[s], 1)
        progress.unlink(missing_ok=True)
        seconds["loading and bookkeeping"] = round(before + time.time() - t0 - sum(secs.values()), 1)
        info["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated(device) / 2 ** 20) if device.type == "cuda" else None
        info["donors"] = {"surA_matched_share": float((run["donor_a"] >= 0).mean()), "surB_matched_share": float((run["donor_b"] >= 0).mean())}

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    mk = task.hparams.get("model_kwargs") or {}
    counts = np.bincount(ix["file"], minlength=len(ds.paths))
    info.update({
        "checkpoint": {"path": str(ck), "bytes": ck.stat().st_size, "sha256": sha256(ck)},
        "variant": var["name"], "run_key": key,
        "model_kwargs": {k: mk.get(k) for k in ("sequence_length", "horizon", "d_z", "max_lag", "num_heads", "source_path", "irf_readout",
                                                 "irf_gate_evidence", "irf_gate_k", "irf_channels", "source_mask_mode", "source_running_gap_keys",
                                                 "posterior_event_gate", "lag_kv_source", "use_entmax", "forecast_student_nu", "persistence_lags",
                                                 "base_decode", "forecast_ar_residual", "forecast_ar_max", "decoder_logvar_clamp", "coverage_floor")},
        # The objective's settings that tell two checkpoints apart, from the model kwargs or else the task's hparams.
        "task_settings": {k: mk[k] if k in mk else task.hparams.get(k) for k in (
            "base_decode", "forecast_ar_residual", "forecast_ar_max", "decoder_logvar_clamp", "coverage_floor", "event_weight_floor",
            "event_weight_ua_valid", "source_valid_tolerance_s", "target_quality_mask", "freeze_target_path", "lambda_full_wrong")},
        "stats_sha256": sha256(args.stats), "code_version": code_version(),
        "git_commit": commit,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform()},
        "shards": [{"path": str(p), "subgroup": core.subgroup_of(p), "rows": int(c)} for p, c in zip(ds.paths, counts)],
        "stats": str(args.stats), "n_guids": int(len(guids)), "n_segments": int(len(rows)),
        "guids_per_subgroup": {str(s): int((sub_of == s).sum()) for s in np.unique(sub_of)},
        "args": {k: v for k, v in vars(args).items()}, "skipped": skipped, "notes": notes,
        "started": info.get("started") or time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
        "seconds_per_step": seconds,
        "seconds_note": "An arm shared by several steps is counted in the first step that uses it.",
    })
    if pending or "ended" not in info:
        info["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info_path.write_text(json.dumps(info, indent=1, default=str))
    rows_out = report.build(out, info)
    print(f"{len(rows_out)} statistics written to {out}")
    print(f"data without figures copied to {pack_for_sending(out)}")
    return 0


#: Arguments for a run without command-line arguments (an IDE Run button). Any argument given on the
#: command line replaces all of these. Keys are the command-line options with "_" for "-"; None leaves
#: an option at its default. Paths are the production server's.
RUN_ARGS = {
    # The checkpoint to analyse. Run once for the new run's best checkpoint and once for the
    # checkpoint it started from (epoch 369 or 400), with the same shards and stats.
    "checkpoint": "/data/deid/isilon/MS_model/q3_2026/lag_attn_transformer_cfs_new/2026-11-18--[12-40]-lag_attn_trf_cfs_revised/model_checkpoints/lag-attn-trf-cfs-epoch=369.ckpt",
    # Shard files or globs, evaluated together as one set. These are the training run's validation
    # shards (healthy only). Add acidosis and HIE shards to fill the "adverse" block of the report.
    "shards": ["/data1/fetal-heart-tracing/HDF5_Datasets/2026-11-18/pre_training_dataset/test/*/shard_*.hdf5"],
    # The normalisation statistics the checkpoint was trained with.
    "stats": "/data1/fetal-heart-tracing/HDF5_Datasets/2026-11-18/stats.hdf5",
    # Output folder. None = <run folder>/lag_analysis/<checkpoint name>, beside model_checkpoints;
    # the copy without figures goes to the same path + "_send".
    "out": None,
    # Resolved training config; None = the resolved_config.yaml beside the checkpoint.
    "config": None,
    "steps": None,        # None = all steps; or e.g. "gain,shift,events"
    "max_guids": None,    # None = every recording; a small number for a quick try
    "batch_size": None,   # None = 16
    "num_workers": None,  # None = 2
    "k_samples": None,    # None = 8 latent samples
    "device": None,       # None = cuda if available
    "force": False,       # True = redo steps whose result file exists
}


def run_args_argv(run_args):
    """The command-line arguments that ``run_args`` stands for (see :data:`RUN_ARGS`)."""
    args = dict(run_args)
    if args.get("out") is None:
        checkpoint = Path(args["checkpoint"])
        args["out"] = str(checkpoint.parent.parent / "lag_analysis" / checkpoint.stem)
    argv = []
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv += [flag, *map(str, value)]
        else:
            argv += [flag, str(value)]
    return argv


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or run_args_argv(RUN_ARGS)))
