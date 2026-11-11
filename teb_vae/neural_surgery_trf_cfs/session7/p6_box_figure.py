r"""E8 figure: the FHR and the UA around a masked box, a clean contraction and a random time.

The figure reads the event table of :mod:`p6_box_events` and draws:

* top left: the mean FHR change $f(t_0 + \tau) - \bar f_{[t_0 - 60, t_0 - 10]}$ for
  $\tau \in [-60, 150]$ s, per group. Each curve is the mean over recordings of the per-recording
  mean, and its band is $\pm 1.96$ standard errors over recordings;
* top right: the mean raw UA and the mean stored (bridged) UA around the same events, which shows
  what the bridge removes;
* bottom: eight box events of eight recordings, with the raw UA, the stored UA, the invalid mask
  and the valid FHR.

The output is ``out/phase6/box_figure.png``.

Edit :data:`RUN_ARGS` and run this file from the IDE.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7.p6_box_events import FS, OUT6  # noqa: E402

#: Run-button inputs. ``per_recording`` caps the events of the two large groups per recording.
RUN_ARGS: Dict[str, Any] = {"per_recording": 40, "seed": 3}

BEFORE_S, AFTER_S = 60, 150
GROUPS = {"box": ("masked box", "#d0473c"), "clean": ("clean contraction", "#2a6fbb"),
          "random_quiet": ("random quiet time", "#7a7a7a")}


def collect(per_recording: int, seed: int) -> Dict[str, Dict[str, np.ndarray]]:
    """Event-aligned traces per group, averaged within each recording.

    Args:
        per_recording: Largest number of events of one group taken from one recording.
        seed: Seed of the subsample.

    Returns:
        ``{group: {'fhr', 'raw', 'stored'}}``, each ``(recordings, samples)`` with ``NaN`` where a
        recording has no valid sample.
    """
    from hdf5_dataset import signal_quality as sq

    resp = pd.read_parquet(OUT6 / "fhr_response.parquet")
    resp = resp[resp["group"].isin(GROUPS) & resp["dip"].notna()]
    resp = resp.groupby(["rec", "group"], group_keys=False).apply(
        lambda part: part.sample(min(len(part), per_recording), random_state=seed))
    span = np.arange(-BEFORE_S * FS, AFTER_S * FS)
    sums: Dict[Any, Dict[str, np.ndarray]] = {}
    counts: Dict[Any, Dict[str, np.ndarray]] = {}
    for cohort, part in resp.groupby("cohort"):
        with h5py.File(S.SHARDS[cohort], "r") as handle:
            for row, events in part.groupby("row"):
                row = int(row)
                f = handle["fhr_raw"][row].astype(np.float64)
                f[(f == 0) | ((handle["fhr_quality"][row].astype(np.int64) & sq.FHR_INVALID) != 0)] = np.nan
                raw = handle["up_raw"][row].astype(np.float64)
                raw[(raw > 127) | (raw < 0)] = np.nan
                stored = handle["up"][row].astype(np.float64)
                for event in events.itertuples():
                    index = int(event.t0) + span
                    ok = (index >= 0) & (index < f.size)
                    level = np.nanmean(f[int(event.t0) - 60 * FS:int(event.t0) - 10 * FS])
                    key = (event.group, event.rec)
                    if key not in sums:
                        sums[key] = {k: np.zeros(span.size) for k in ("fhr", "raw", "stored")}
                        counts[key] = {k: np.zeros(span.size) for k in ("fhr", "raw", "stored")}
                    for name, trace in (("fhr", f - level), ("raw", raw), ("stored", stored)):
                        values = np.full(span.size, np.nan)
                        values[ok] = trace[index[ok]]
                        good = np.isfinite(values)
                        sums[key][name][good] += values[good]
                        counts[key][name][good] += 1
    out: Dict[str, Dict[str, List[np.ndarray]]] = {g: {"fhr": [], "raw": [], "stored": []} for g in GROUPS}
    for (group, _), total in sums.items():
        for name in total:
            n = counts[(group, _)][name]
            out[group][name].append(np.where(n > 0, total[name] / np.maximum(n, 1), np.nan))
    return {g: {name: np.asarray(v) for name, v in fields.items()} for g, fields in out.items()}


def main(per_recording: int, seed: int) -> Path:
    """Draw the figure.

    Args:
        per_recording: Largest number of events of one group taken from one recording.
        seed: Seed of the subsample and of the example choice.

    Returns:
        The path of the figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces = collect(per_recording, seed)
    tau = np.arange(-BEFORE_S * FS, AFTER_S * FS) / FS
    fig = plt.figure(figsize=(20, 12))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.5, 1, 1], hspace=0.38, wspace=0.28)
    ax = fig.add_subplot(grid[0, :2])
    for group, (label, color) in GROUPS.items():
        values = traces[group]["fhr"]
        mean = np.nanmean(values, axis=0)
        error = 1.96 * np.nanstd(values, axis=0) / np.sqrt(np.maximum(np.isfinite(values).sum(axis=0), 1))
        ax.plot(tau, mean, color=color, lw=2, label=f"{label} ({values.shape[0]} recordings)")
        ax.fill_between(tau, mean - error, mean + error, color=color, alpha=0.2, lw=0)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("time from the event start (s)")
    ax.set_ylabel("FHR change from the 50 s before (bpm)")
    ax.set_title("FHR after the event: mean over recordings, band = 95% interval")
    ax.legend(loc="lower left")
    ax = fig.add_subplot(grid[0, 2:])
    for group, (label, color) in GROUPS.items():
        if group == "random_quiet":
            continue
        ax.plot(tau, np.nanmean(traces[group]["raw"], axis=0), color=color, lw=2, label=f"{label}: raw UA")
        ax.plot(tau, np.nanmean(traces[group]["stored"], axis=0), color=color, lw=2, ls="--",
                label=f"{label}: stored UA (bridged)")
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("time from the event start (s)")
    ax.set_ylabel("UA (units)")
    ax.set_title("UA around the event: what the bridge removes")
    ax.legend(loc="upper right")

    events = pd.read_parquet(OUT6 / "box_events.parquet")
    pick = events.groupby("rec").head(1).sample(8, random_state=seed)
    for k, event in enumerate(pick.itertuples()):
        ax = fig.add_subplot(grid[1 + k // 4, k % 4])
        with h5py.File(S.SHARDS[event.cohort], "r") as handle:
            raw = handle["up_raw"][event.row].astype(np.float64)
            stored = handle["up"][event.row]
            bits = handle["up_quality"][event.row].astype(np.int64)
            f = handle["fhr_raw"][event.row].astype(np.float64)
            fq = handle["fhr_quality"][event.row].astype(np.int64)
        lo, hi = max(int(event.start) - 120 * FS, 0), min(int(event.stop) + 150 * FS, raw.size)
        t = (np.arange(lo, hi) - event.start) / FS
        ax.fill_between(t, 0, 130, where=(bits[lo:hi] & S.UP_INVALID_BITS) != 0, color="orange", alpha=0.25,
                        label="UA invalid")
        ax.plot(t, raw[lo:hi], color="#d0473c", lw=1.3, label="UA raw")
        ax.plot(t, stored[lo:hi], color="#2a6fbb", lw=1.3, label="UA stored")
        ax.set_ylim(-5, 135)
        twin = ax.twinx()
        twin.plot(t, np.where((fq[lo:hi] != 0) | (f[lo:hi] == 0), np.nan, f[lo:hi]), color="k", lw=0.8)
        twin.set_ylim(60, 200)
        ax.set_title(f"{event.cohort}, {event.len_s:.0f} s, {event.height:.0f} units above the rest", fontsize=10)
        if k == 0:
            ax.legend(fontsize=8, loc="upper left")
        if k % 4 == 0:
            ax.set_ylabel("UA (units)")
        if k % 4 == 3:
            twin.set_ylabel("FHR (bpm), black")
        if k >= 4:
            ax.set_xlabel("time from the box start (s)")
    path = OUT6 / "box_figure.png"
    fig.savefig(path, dpi=80, bbox_inches="tight")
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-recording", type=int, default=RUN_ARGS["per_recording"])
    parser.add_argument("--seed", type=int, default=RUN_ARGS["seed"])
    args = parser.parse_args()
    main(args.per_recording, args.seed)
