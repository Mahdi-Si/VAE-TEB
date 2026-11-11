r"""D2 inputs: per-anchor contraction phase, detector-free UA activity and UA/FHR state.

The script reads the dump's ids (``out/dump/anchors.parquet``) and, for every segment of split
codes 0 and 2 only, the stored 22-min rows of the shard: the bridged ``up``, ``up_quality`` and the
bridged ``fhr``. It writes ``out/phase2/anchor_features.parquet`` with one row per dump row of
those codes; the column ``row`` is the dump row index.

**Timeline.** The loader serves stored raw samples $[240, 5040)$. Anchor $t$ sits at trimmed raw
sample $r_t = 16t$ (the last sample of its 4-s step), so its stored raw index is $240 + 16t$. A
raw sample $p$ (trimmed) belongs to model step $\lceil p/16 \rceil$.

**Contraction phase.** :func:`teb_vae.lag_attn_cfs.eval.events.detect_contractions` runs on the
full stored row, with the raw validity $(q \,\&\, 2299) = 0$; the detector drops every event that
overlaps an invalid sample. For an event (onset $o$, peak $p$, end $e$) that contains $r_t$:

* ``peak`` if $|r_t - p| \le 40$ samples (10 s);
* ``rise`` if $o \le r_t < p - 40$; ``fall`` if $p + 40 < r_t \le e$;
* ``between`` when no event contains $r_t$.

**Detector-free UA activity.** With the robust baseline $b$ (10th percentile of the served
bridged ``up``) and spread $s = \max(P_{90} - P_{10}, 1)$,

$$A^{\mathrm{lvl}}_t = \frac{\overline{u}_{[r_t - 239,\, r_t]} - b}{s}, \qquad
  A^{\mathrm{slope}}_t = \frac{\hat\beta_{[r_t - 239,\, r_t]}}{s}\ \text{per minute},$$

where $\hat\beta$ is the least-squares slope over the last 60 s.

**State features** (probes of the latent): UA level (mean over the 4-s step, in $s$ units above
$b$) and 30-s slope at lags 0, 30 and 60 s; seconds since the last onset and peak; FHR level (mean
over the last 16 s, bpm), its 60-s change, and its detrended standard deviation over 60 s.

Edit :data:`RUN_ARGS` and run this file from the IDE, or pass the options on the command line.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.lag_attn_cfs.eval import events  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {
    "dump_dir": str(S.OUT / "dump"),
    "out": str(S.OUT / "phase2" / "anchor_features.parquet"),
    "codes": [0, 2],
}

TRIM_RAW = 240
PEAK_HALF = 40  # 10 s at 4 Hz
PHASES = ("rise", "peak", "fall", "between")


def window_slope(x: np.ndarray, ends: np.ndarray, width: int) -> np.ndarray:
    r"""Least-squares slope (units per sample) of $x$ over the ``width`` samples ending at each end.

    Args:
        x: 1-D trace.
        ends: Inclusive end indices.
        width: Window length in samples.

    Returns:
        One slope per end; NaN where the window leaves the trace.
    """
    cs = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(x.size, dtype=np.float64)
    cs_ix = np.concatenate([[0.0], np.cumsum(idx * x)])
    lo = ends - width + 1
    ok = lo >= 0
    lo_c, hi_c = np.clip(lo, 0, None), ends + 1
    n = float(width)
    sx = cs[hi_c] - cs[lo_c]
    sxi = cs_ix[hi_c] - cs_ix[lo_c]
    si = (lo_c + ends) * n / 2.0
    def sq_sum(m: np.ndarray) -> np.ndarray:
        m = m.astype(np.float64)
        return m * (m + 1) * (2 * m + 1) / 6.0

    sii = sq_sum(ends) - sq_sum(lo_c - 1)
    den = sii - si * si / n
    slope = (sxi - si * sx / n) / np.where(den > 0, den, np.nan)
    return np.where(ok, slope, np.nan)


def window_mean(x: np.ndarray, ends: np.ndarray, width: int) -> np.ndarray:
    """Mean of ``x`` over the ``width`` samples ending at each end (NaN outside the trace).

    Args:
        x: 1-D trace.
        ends: Inclusive end indices.
        width: Window length in samples.

    Returns:
        One mean per end.
    """
    cs = np.concatenate([[0.0], np.cumsum(x)])
    lo = ends - width + 1
    ok = lo >= 0
    out = (cs[ends + 1] - cs[np.clip(lo, 0, None)]) / float(width)
    return np.where(ok, out, np.nan)


def segment_features(up: np.ndarray, quality: np.ndarray, fhr: np.ndarray, steps: np.ndarray) -> Dict[str, np.ndarray]:
    """Every feature of the anchors of one segment.

    Args:
        up: Stored bridged UA row (5280 samples).
        quality: Stored ``up_quality`` row.
        fhr: Stored bridged FHR row (bpm).
        steps: Anchor steps on the trimmed grid.

    Returns:
        Feature arrays, one entry per anchor.
    """
    valid = (quality.astype(np.int64) & S.UP_INVALID_BITS) == 0
    try:
        ev = events.detect_contractions(up, fs=S.FS, valid=valid)
    except Exception:  # a row the detector cannot read has no events
        ev = {"onset_raw": np.zeros(0, int), "peak_raw": np.zeros(0, int), "end_raw": np.zeros(0, int)}
    onset = np.asarray(ev["onset_raw"], dtype=np.int64) - TRIM_RAW
    peak = np.asarray(ev["peak_raw"], dtype=np.int64) - TRIM_RAW
    end = np.asarray(ev["end_raw"], dtype=np.int64) - TRIM_RAW
    order = np.argsort(peak)
    onset, peak, end = onset[order], peak[order], end[order]
    r = (S.RAW_PER_STEP * steps).astype(np.int64)
    n = r.size
    phase = np.full(n, "between", dtype=object)
    amp = np.full(n, np.nan)
    for o, p, e in zip(onset, peak, end):
        inside = (r >= o) & (r <= e)
        phase[inside & (np.abs(r - p) <= PEAK_HALF)] = "peak"
        phase[inside & (r < p - PEAK_HALF)] = "rise"
        phase[inside & (r > p + PEAK_HALF)] = "fall"
        amp[inside] = up[p + TRIM_RAW] - up[o + TRIM_RAW]

    def since(marks: np.ndarray) -> np.ndarray:
        if marks.size == 0:
            return np.full(n, np.nan)
        srt = np.sort(marks)
        k = np.searchsorted(srt, r, side="right") - 1
        return np.where(k >= 0, (r - srt[np.clip(k, 0, None)]) / S.FS, np.nan)

    peak_step = np.ceil(peak / S.RAW_PER_STEP).astype(np.int64)
    lag_last = np.full(n, np.nan)
    lag_prev = np.full(n, np.nan)
    rel_next = np.full(n, np.nan)  # steps until the next peak (future), for reference only
    for i, t in enumerate(steps):
        lags = t - peak_step
        inwin = np.sort(lags[(lags >= 0) & (lags <= 37)])
        if inwin.size:
            lag_last[i] = inwin[0]
        if inwin.size > 1:
            lag_prev[i] = inwin[1]
        ahead = -lags[lags < 0]
        if ahead.size:
            rel_next[i] = ahead.min()

    served = up[TRIM_RAW:-TRIM_RAW].astype(np.float64)
    base = float(np.percentile(served, 10))
    spread = max(float(np.percentile(served, 90) - base), 1.0)
    u = (served - base) / spread
    f = fhr[TRIM_RAW:-TRIM_RAW].astype(np.float64)
    out: Dict[str, np.ndarray] = {
        "phase": phase,
        "contraction_amp": amp,
        "sec_since_onset": since(onset),
        "sec_since_peak": since(peak),
        "peak_lag_last": lag_last,
        "peak_lag_prev": lag_prev,
        "steps_to_next_peak": rel_next,
        "n_contractions_seg": np.full(n, onset.size),
        "ua_level60": window_mean(u, r, 240),
        "ua_slope60": window_slope(u, r, 240) * 240.0,
        "ua_base": np.full(n, base),
        "ua_spread": np.full(n, spread),
    }
    for lag in (0, 30, 60):
        e = r - lag * S.FS
        out[f"ua_lvl_l{lag}"] = window_mean(u, e, 16)
        out[f"ua_slope_l{lag}"] = window_slope(u, e, 120) * 240.0
    lvl_now = window_mean(f, r, 64)
    lvl_ago = window_mean(f, r - 240, 64)
    out["fhr_level"] = lvl_now
    out["fhr_change60"] = lvl_now - lvl_ago
    slope = window_slope(f, r, 240)
    mean60 = window_mean(f, r, 240)
    var = np.array([np.var(f[max(x - 239, 0): x + 1] - (mean60[i] + slope[i] * (np.arange(max(x - 239, 0), x + 1) - (x - 119.5))))
                    for i, x in enumerate(r)])
    out["fhr_var60"] = np.sqrt(var)
    return out


def build_parser() -> argparse.ArgumentParser:
    """Command-line options; every default comes from :data:`RUN_ARGS`.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump-dir", default=RUN_ARGS["dump_dir"])
    parser.add_argument("--out", default=RUN_ARGS["out"])
    parser.add_argument("--codes", type=int, nargs="+", default=RUN_ARGS["codes"])
    return parser


def main(args: Dict[str, Any]) -> None:
    """Build the feature table.

    Args:
        args: The run arguments; see :data:`RUN_ARGS`.
    """
    ids = pd.read_parquet(Path(args["dump_dir"]) / "anchors.parquet",
                          columns=["cohort", "shard_row", "anchor_step", "split_code"])
    ids["row"] = np.arange(len(ids))
    ids = ids[ids["split_code"].isin(args["codes"])]
    parts = []
    t0 = time.time()
    for cohort, path in S.SHARDS.items():
        sub = ids[ids["cohort"] == cohort]
        with h5py.File(path, "r") as handle:
            for k, (shard_row, group) in enumerate(sub.groupby("shard_row", sort=True)):
                feats = segment_features(handle["up"][shard_row], handle["up_quality"][shard_row],
                                         handle["fhr"][shard_row], group["anchor_step"].to_numpy())
                frame = pd.DataFrame(feats)
                frame.insert(0, "row", group["row"].to_numpy())
                parts.append(frame)
        print(f"{cohort}: {sub['shard_row'].nunique()} segments, {time.time() - t0:.0f} s", flush=True)
    table = pd.concat(parts, ignore_index=True).sort_values("row").reset_index(drop=True)
    Path(args["out"]).parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args["out"], index=False)
    print(table["phase"].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main(vars(build_parser().parse_args()))
