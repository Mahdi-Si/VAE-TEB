r"""E8.1: masked boxes near contractions, and the FHR response that follows them.

Some UA traces jump shortly after a contraction onset, hold a flat top and fall back to the rest
level. The quality mask marks the whole box as ``TRANSIENT``, and the bridge then replaces it with
a monotone line. This probe selects those runs from the raw UA and asks a model-free question:
does the FHR respond after a box as it responds after a clean contraction?

**Box event.** A maximal invalid run $[a, b)$ of ``up_quality`` on split codes 0 and 2 with

1. ``TRANSIENT`` on more than 90% of its samples, no other invalid bit and no raw value outside
   $[0, 127]$;
2. length $16 < L \le 64$ s;
3. height $h = \operatorname{median}(x[a..b)) - r \ge 10$ units, where the rest level $r$ is the 10th
   percentile of the valid raw UA within 180 s on both sides;
4. a return: the median valid raw UA from 5 to 30 s after the run is at most $r + 0.5\,h$;
5. at least 60 s of the segment on both sides.

**FHR response.** With $f$ the valid raw FHR, $\tilde f$ its centred 10-s median and
$\bar f_0 = \operatorname{mean} f$ over $[t_0 - 60, t_0 - 10]$ s, the dip and the absolute response are

$$D = \bar f_0 - \min_{\tau \in [0, 90]} \tilde f(t_0 + \tau), \qquad
A = \operatorname{mean}_{\tau \in [0, 90]} \lvert f(t_0 + \tau) - \bar f_0 \rvert .$$

An event counts when at least 70% of the FHR samples are valid in both windows. The minimum makes
$D$ positive for any window, so every statement compares a group with the random times.

**Groups.** ``box`` ($t_0$ = run start), ``clean`` (a detected contraction with no invalid UA
sample in its span; $t_0$ = onset), ``random_quiet`` (a time at least 90 s from every contraction
span and every invalid run) and ``random_any`` (a time with no invalid UA within 90 s; reported
only). A group difference is the mean over recordings of the per-recording mean difference; its
interval resamples recordings (:func:`s7common.paired_bootstrap`).

**Common-cause check.** A movement of the mother can disturb both sensors at once. The probe
reports the share of invalid FHR samples in $[t_0, t_0 + 90]$ s per group, and repeats the box
comparison on events with at least 95% valid FHR.

Outputs go to ``out/phase6``: ``box_events.parquet`` (every segment instance of a box run),
``fhr_response.parquet`` and ``box_fhr_response.json``.

Edit :data:`RUN_ARGS` and run this file from the IDE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import p1_common as C  # noqa: E402

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {"n_boot": 2000, "seed": 0}

#: Output folder of the box probes.
OUT6 = S.OUT / "phase6"
FS = int(S.FS)
#: Box rule (seconds and UA units); see the module docstring.
MIN_LEN_S, MAX_LEN_S = 16.0, 64.0
MIN_HEIGHT = 10.0
RETURN_SHARE = 0.5
CONTEXT_S = 180
MARGIN_S = 60
#: Response windows in seconds relative to $t_0$.
BASE_WIN = (-60, -10)
RESP_WIN = (0, 90)
MIN_VALID = 0.70
STRICT_VALID = 0.95
QUIET_S = 90
TRANSIENT = S.UP_BITS["TRANSIENT"]


def box_runs(x: np.ndarray, bits: np.ndarray) -> List[Dict[str, float]]:
    """Find the box events of one segment.

    Args:
        x: Raw UA ``(N,)``.
        bits: ``up_quality`` ``(N,)``.

    Returns:
        One dictionary per box run with its bounds and shape statistics.
    """
    inv = (bits & S.UP_INVALID_BITS) != 0
    other = S.UP_INVALID_BITS & ~TRANSIENT
    found: List[Dict[str, float]] = []
    for a, b in C.segment_runs(inv):
        length = (b - a) / FS
        if not (MIN_LEN_S < length <= MAX_LEN_S) or a < MARGIN_S * FS or b > x.size - MARGIN_S * FS:
            continue
        seg = bits[a:b]
        inside = x[a:b]
        if ((seg & TRANSIENT) != 0).mean() <= 0.9 or ((seg & other) != 0).any():
            continue
        if ((inside > 127) | (inside < 0)).any():
            continue
        lo, hi = max(a - CONTEXT_S * FS, 0), min(b + CONTEXT_S * FS, x.size)
        ctx = np.concatenate([x[lo:a][~inv[lo:a]], x[b:hi][~inv[b:hi]]])
        post = x[b + 5 * FS:b + 30 * FS][~inv[b + 5 * FS:b + 30 * FS]]
        if ctx.size < 200 or post.size < 20:
            continue
        rest = float(np.percentile(ctx, 10))
        height = float(np.median(inside)) - rest
        if height < MIN_HEIGHT or float(np.median(post)) > rest + RETURN_SHARE * height:
            continue
        found.append({"start": int(a), "stop": int(b), "len_s": float(length), "rest": rest, "height": height,
                      "ceiling_share": float(np.isin(inside, [100.0, 127.0]).mean()),
                      "pre_level": float(x[a - 1]) - rest})
    return found


def fhr_response(f: np.ndarray, smooth: np.ndarray, t0: int) -> Optional[Dict[str, float]]:
    """Dip and absolute response of the FHR after raw sample ``t0``.

    Args:
        f: Raw FHR ``(N,)`` with ``NaN`` at invalid samples.
        smooth: Its centred 10-s median.
        t0: Reference raw sample.

    Returns:
        ``{'dip', 'abs', 'valid_resp', 'valid_base'}``, or ``None`` when a window leaves the
        segment.
    """
    b0, b1 = t0 + BASE_WIN[0] * FS, t0 + BASE_WIN[1] * FS
    r0, r1 = t0 + RESP_WIN[0] * FS, t0 + RESP_WIN[1] * FS
    if b0 < 0 or r1 > f.size:
        return None
    base, resp = f[b0:b1], f[r0:r1]
    v_base, v_resp = float(np.isfinite(base).mean()), float(np.isfinite(resp).mean())
    out = {"valid_resp": v_resp, "valid_base": v_base, "dip": np.nan, "abs": np.nan}
    if v_base >= MIN_VALID and v_resp >= MIN_VALID and np.isfinite(smooth[r0:r1]).any():
        level = float(np.nanmean(base))
        out["dip"] = level - float(np.nanmin(smooth[r0:r1]))
        out["abs"] = float(np.nanmean(np.abs(resp - level)))
    return out


def scan(seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select the box events and measure the FHR response of every group.

    Args:
        seed: Seed of the random-time draws.

    Returns:
        ``(events, responses)``: every segment instance of a box run, and one row per measured
        event of every group.
    """
    from hdf5_dataset import signal_quality as sq

    rng = np.random.default_rng(seed)
    events: List[Dict[str, Any]] = []
    rows_out: List[Dict[str, Any]] = []
    for cohort in S.SHARDS:
        rows = C.split_rows(cohort)
        with h5py.File(S.SHARDS[cohort], "r") as handle:
            data = {k: handle[k][rows] for k in ("up_raw", "up", "up_quality", "weight", "epoch", "fhr_raw",
                                                 "fhr_quality", "second_stage_onset")}
            guids = [S.guid_text(g) for g in handle["guid"][rows]]
        for j, row in enumerate(rows):
            x = data["up_raw"][j].astype(np.float64)
            bits = data["up_quality"][j].astype(np.int64)
            inv = (bits & S.UP_INVALID_BITS) != 0
            f = data["fhr_raw"][j].astype(np.float64)
            f[(f == 0) | ((data["fhr_quality"][j].astype(np.int64) & sq.FHR_INVALID) != 0)] = np.nan
            smooth = pd.Series(f).rolling(10 * FS, center=True, min_periods=5 * FS).median().to_numpy()
            epoch = float(data["epoch"][j])
            since_stage2 = float(data["second_stage_onset"][j])
            rec = f"{cohort}/{guids[j]}"
            key = {"cohort": cohort, "guid": guids[j], "rec": rec, "row": int(row)}

            def add(group: str, t0: int, extra: Optional[Dict[str, Any]] = None) -> None:
                got = fhr_response(f, smooth, t0)
                if got is not None:
                    rows_out.append({**key, "group": group, "t0": int(t0), "abs_t0": epoch + t0 / FS,
                                     **got, **(extra or {})})

            boxes = box_runs(x, bits)
            for box in boxes:
                stage2 = bool(np.isfinite(since_stage2) and box["start"] / FS + since_stage2 >= 0)
                events.append({**key, **box, "abs_start": epoch + box["start"] / FS, "stage2": stage2})
                add("box", box["start"], {"len_s": box["len_s"], "height": box["height"],
                                          "ceiling_share": box["ceiling_share"], "stage2": stage2})
            det = C.detect(data["up"][j], data["weight"][j])
            busy = inv.copy()
            for o, e in zip(det["onset_raw"], det["end_raw"]):
                o, e = int(o), int(e)
                busy[max(o, 0):e + 1] = True
                if not inv[max(o, 0):e + 1].any():
                    add("clean", o)
            near_busy = np.convolve(busy.astype(np.float64), np.ones(2 * QUIET_S * FS + 1), mode="same") > 0
            near_inv = np.convolve(inv.astype(np.float64), np.ones(2 * QUIET_S * FS + 1), mode="same") > 0
            grid = np.arange(MARGIN_S * FS, x.size - 120 * FS, 30 * FS) + int(rng.integers(0, 30 * FS))
            grid = grid[grid < x.size - 120 * FS]
            for t0 in grid:
                if not near_busy[t0]:
                    add("random_quiet", int(t0))
                if not near_inv[t0]:
                    add("random_any", int(t0))
    ev = pd.DataFrame(events)
    resp = pd.DataFrame(rows_out)
    resp["slot"] = (resp["abs_t0"] / 10.0).round().astype(np.int64)
    resp = resp.drop_duplicates(subset=["rec", "group", "slot"]).drop(columns="slot").reset_index(drop=True)
    return ev, resp


def compare(resp: pd.DataFrame, a: str, b: str, column: str, n_boot: int) -> Dict[str, float]:
    """Recording-level difference of one response column between two groups.

    Args:
        resp: The response table.
        a: First group.
        b: Second group.
        column: ``'dip'`` or ``'abs'``.
        n_boot: Bootstrap resamples.

    Returns:
        The :func:`s7common.paired_bootstrap` dictionary of group ``a`` minus group ``b``, with the
        two group means.
    """
    part = resp[resp[column].notna()]
    mean = part.groupby(["rec", "group"])[column].mean().unstack("group")
    both = mean[[a, b]].dropna() if {a, b} <= set(mean.columns) else pd.DataFrame(columns=[a, b])
    out = S.paired_bootstrap(both.index.to_numpy(), (both[a] - both[b]).to_numpy(), n_resamples=n_boot)
    out.update({f"mean_{a}": float(both[a].mean()) if len(both) else float("nan"),
                f"mean_{b}": float(both[b].mean()) if len(both) else float("nan")})
    return out


def ratio(resp: pd.DataFrame, reference: str, column: str, n_boot: int) -> Dict[str, float]:
    r"""Share of the clean-contraction effect that the box events reach.

    The statistic is $\sum_r (\bar D^{\text{box}}_r - \bar D^{\text{ref}}_r) / \sum_r (\bar D^{\text{clean}}_r -
    \bar D^{\text{ref}}_r)$ over the recordings $r$ that hold all three groups.

    Args:
        resp: The response table.
        reference: The random group.
        column: ``'dip'`` or ``'abs'``.
        n_boot: Bootstrap resamples.

    Returns:
        ``{'ratio', 'lo', 'hi', 'n_rec'}``.
    """
    mean = resp[resp[column].notna()].groupby(["rec", "group"])[column].mean().unstack("group")
    if not {"box", "clean", reference} <= set(mean.columns):
        return {"ratio": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_rec": 0}
    full = mean[["box", "clean", reference]].dropna()
    num = (full["box"] - full[reference]).to_numpy()
    den = (full["clean"] - full[reference]).to_numpy()
    draws = np.random.default_rng(0).integers(0, len(full), size=(n_boot, len(full)))
    ratios = num[draws].sum(1) / den[draws].sum(1)
    return {"ratio": float(num.sum() / den.sum()), "lo": float(np.percentile(ratios, 2.5)),
            "hi": float(np.percentile(ratios, 97.5)), "n_rec": int(len(full))}


def main(n_boot: int, seed: int) -> Dict[str, Any]:
    """Run the selection and the model-free test, and write the outputs.

    Args:
        n_boot: Bootstrap resamples.
        seed: Seed of the random-time draws.

    Returns:
        The summary dictionary.
    """
    OUT6.mkdir(parents=True, exist_ok=True)
    events, resp = scan(seed)
    events.to_parquet(OUT6 / "box_events.parquet")
    resp.to_parquet(OUT6 / "fhr_response.parquet")
    unique = events.drop_duplicates(subset=["rec", "abs_start"])
    summary: Dict[str, Any] = {
        "split": "codes 0 and 2",
        "events": {"segment_instances": int(len(events)), "unique": int(len(unique)),
                   "recordings": int(unique["rec"].nunique()),
                   "per_cohort": unique.groupby("cohort").size().to_dict(),
                   "len_s_quartiles": [float(q) for q in unique["len_s"].quantile([0.25, 0.5, 0.75])],
                   "height_quartiles": [float(q) for q in unique["height"].quantile([0.25, 0.5, 0.75])],
                   "share_ceiling_ge_0.3": float((unique["ceiling_share"] >= 0.3).mean()),
                   "share_second_stage": float(unique["stage2"].mean())},
        "groups": {}, "comparisons": {}, "ratio_of_clean_effect": {}, "strata": {},
    }
    for group, part in resp.groupby("group"):
        ok = part["dip"].notna()
        summary["groups"][group] = {"n_events": int(len(part)), "n_measured": int(ok.sum()),
                                    "n_rec": int(part.loc[ok, "rec"].nunique()),
                                    "dip_mean": float(part.loc[ok, "dip"].mean()),
                                    "abs_mean": float(part.loc[ok, "abs"].mean()),
                                    "fhr_invalid_share_response": float(1.0 - part["valid_resp"].mean())}
    for column in ("dip", "abs"):
        for a, b in (("box", "random_quiet"), ("clean", "random_quiet"), ("box", "clean"),
                     ("box", "random_any"), ("clean", "random_any")):
            summary["comparisons"][f"{column}:{a}-{b}"] = compare(resp, a, b, column, n_boot)
        for ref in ("random_quiet", "random_any"):
            summary["ratio_of_clean_effect"][f"{column}:{ref}"] = ratio(resp, ref, column, n_boot)
    strict = resp[resp["valid_resp"] >= STRICT_VALID]
    summary["common_cause"] = {
        "fhr_invalid_share_box_minus_clean": summary["groups"]["box"]["fhr_invalid_share_response"]
        - summary["groups"]["clean"]["fhr_invalid_share_response"],
        "dip:box-random_quiet@95pct_valid": compare(strict, "box", "random_quiet", "dip", n_boot),
        "dip:box-random_any@95pct_valid": compare(strict, "box", "random_any", "dip", n_boot),
        "dip:clean-random_any@95pct_valid": compare(strict, "clean", "random_any", "dip", n_boot),
    }
    box = resp[resp["group"] == "box"]
    others = resp[resp["group"] != "box"]
    strata = {"ceiling_ge_0.3": box["ceiling_share"] >= 0.3, "ceiling_lt_0.3": box["ceiling_share"] < 0.3,
              "second_stage": box["stage2"].astype(bool), "first_stage": ~box["stage2"].astype(bool),
              "len_16_32": box["len_s"] <= 32, "len_32_64": box["len_s"] > 32}
    for name, keep in strata.items():
        part = pd.concat([box[keep], others])
        summary["strata"][name] = {"n_box": int(keep.sum()),
                                   "dip:box-random_any": compare(part, "box", "random_any", "dip", n_boot),
                                   "dip:box-clean": compare(part, "box", "clean", "dip", n_boot)}
    (OUT6 / "box_fhr_response.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=RUN_ARGS["n_boot"])
    parser.add_argument("--seed", type=int, default=RUN_ARGS["seed"])
    args = parser.parse_args()
    main(args.n_boot, args.seed)
