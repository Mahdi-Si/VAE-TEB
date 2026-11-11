r"""T1: the table of every invalid UA run in the code-0/2 recordings, and its summaries.

An invalid run is a maximal run of raw samples with ``up_quality & 2299 != 0``. For each run the
script records its cohort, GUID, segment row, start, length, the bits inside it (all bits and the
dominant invalid bit, i.e. the invalid bit on the most samples), its contraction phase (the phase
of its middle sample; see :mod:`p1_common`) and how long the strict rule keeps the source unusable
after the run ends.

**Contamination after a run, isolated.** For a run $[s, e)$ alone, the contamination of kept
channel $c$ at step $k$ is

$$\rho_c[k] = \mathcal{C}_c(16k - s) - \mathcal{C}_c(16k - e),$$

where $\mathcal{C}_c(j) = \sum_{i \le j} w_c[i]$ is the cumulative magnitude kernel of the builder
(``ContaminationBank.cumulative``; $\mathcal{C}_c(j) = 0$ for $j < 0$ and $1$ past the kernel). A
run that starts at sample 0 also covers the edge-padded history, so there
$\rho_c[k] = 1 - \mathcal{C}_c(16k - e)$. The script evaluates this on an open-ended step grid, so
the duration is not cut by the segment end. Step $k$ is "after the run" when $16k \ge e$. The
duration of channel $c$ is the number of steps from the first step after the run to the last step
with $\rho_c[k] > 0.1$. The strict duration adds the bit window (step $k$ reads raw samples
$16k-15 \dots 16k$) and takes the maximum over the kept channels.

**Contamination after a run, stored.** The script also counts the consecutive steps after the run
that the strict rule marks bad on the stored fields. This count includes other runs and the step
term of level shifts, and the segment end cuts it.

**Overlap.** Consecutive segments of one recording overlap by 120 s, so a run near a segment edge
can appear in two segments. Counts are per segment; a de-duplicated count keys a run by its GUID
and absolute start and stop time (``epoch`` + offset).

Edit :data:`RUN_ARGS` and run this file from the IDE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import p1_common as C  # noqa: E402

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {"device": "cuda:0"}

INVALID_NAMES = [name for name, bit in S.UP_BITS.items() if bit & S.UP_INVALID_BITS]


def cumulative_kernels(device: str) -> np.ndarray:
    """The builder's cumulative magnitude kernels of the 42 kept channels.

    Args:
        device: Torch device for the builder chain.

    Returns:
        ``(42, H + 1)`` float64, padded with 1 past each kernel.
    """
    from teb_vae.neural_surgery_trf_cfs.session7.p1_rebuild import Rebuilder

    rb = Rebuilder(device)
    st = rb.rho_banks["up_st"].cumulative[C.KEPT_ST]
    ph = rb.rho_banks["up_ph"].cumulative[C.KEPT_PH]
    width = max(st.shape[1], ph.shape[1])
    pad = lambda a: np.concatenate([a, np.ones((a.shape[0], width - a.shape[1]))], axis=1)  # noqa: E731
    return np.concatenate([pad(st), pad(ph)], axis=0), rb


def isolated_rho(cum: np.ndarray, start: int, stop: int, steps: np.ndarray) -> np.ndarray:
    r"""Contamination of one isolated run on the given steps (see the module docstring).

    Args:
        cum: ``(C, H + 1)`` cumulative kernels.
        start: First raw sample of the run.
        stop: One past the last raw sample.
        steps: Step indices $k$.

    Returns:
        ``(C, K)`` $\rho$.
    """
    h = cum.shape[1] - 1

    def big_c(lag: np.ndarray) -> np.ndarray:
        out = np.zeros((cum.shape[0], lag.size))
        ok = lag >= 0
        out[:, ok] = cum[:, np.minimum(lag[ok], h)]
        return out

    now = steps * S.RAW_PER_STEP
    upper = np.ones((cum.shape[0], steps.size)) if start == 0 else big_c(now - start)
    return np.clip(upper - big_c(now - stop), 0.0, 1.0)


def after_durations(cum: np.ndarray, start: int, stop: int) -> Dict[str, Any]:
    """Steps after a run during which each kept channel has $\\rho > 0.1$, isolated run.

    Args:
        cum: ``(C, H + 1)`` cumulative kernels.
        start: First raw sample.
        stop: One past the last raw sample.

    Returns:
        ``{'per_channel': (42,) int, 'strict': int, 'onset_per_channel': (42,) int}``; the onset is
        the first step after the run with $\\rho > 0.1$ (``-1`` if none).
    """
    k_end = -(-stop // S.RAW_PER_STEP)
    horizon = (cum.shape[1] - 1) // S.RAW_PER_STEP + 3
    steps = np.arange(k_end, k_end + horizon)
    bad = isolated_rho(cum, start, stop, steps) > C.RHO_EPS
    any_bad = bad.any(axis=1)
    last = np.where(any_bad, bad.shape[1] - np.argmax(bad[:, ::-1], axis=1), 0)
    onset = np.where(any_bad, np.argmax(bad, axis=1), -1)
    bit_step = 1 if (S.RAW_PER_STEP * k_end - (S.RAW_PER_STEP - 1)) <= stop - 1 else 0
    return {"per_channel": last.astype(int), "strict": int(max(last.max(), bit_step)),
            "onset_per_channel": onset.astype(int)}


def stored_after(bad_steps: np.ndarray, stop: int) -> Dict[str, int]:
    """Consecutive strictly-bad steps after a run on the stored fields.

    Args:
        bad_steps: ``(T,)`` any-channel strict bad flag.
        stop: One past the run's last raw sample.

    Returns:
        ``{'steps', 'censored'}``; ``censored`` is 1 when the bad stretch reaches the segment end.
    """
    k = -(-stop // S.RAW_PER_STEP)
    n = 0
    while k + n < bad_steps.size and bad_steps[k + n]:
        n += 1
    return {"steps": n, "censored": int(k + n >= bad_steps.size)}


def scan(device: str) -> Dict[str, Any]:
    """Build the run and contraction tables and write the summaries.

    Args:
        device: Torch device of the builder chain.

    Returns:
        The summary dictionary.
    """
    cum, rb = cumulative_kernels(device)
    meta = C.channel_meta()
    # Check the analytic isolated rho against the builder's bank on a few runs.
    check = []
    for start, stop in ((1000, 1016), (2000, 2128), (0, 40), (4000, 4512)):
        ind = np.zeros(C.N_RAW, dtype=bool)
        ind[start:stop] = True
        bank = np.concatenate([rb.rho_banks["up_st"](ind)[C.KEPT_ST], rb.rho_banks["up_ph"](ind)[C.KEPT_PH]])
        ana = isolated_rho(cum, start, stop, np.arange(C.N_STEPS))
        check.append(float(np.abs(bank - ana).max()))
    run_rows: List[Dict[str, Any]] = []
    con_rows: List[Dict[str, Any]] = []
    seg_rows: List[Dict[str, Any]] = []
    chan_dur: List[np.ndarray] = []
    chan_onset: List[np.ndarray] = []
    for cohort in S.SHARDS:
        rows = C.split_rows(cohort)
        with h5py.File(S.SHARDS[cohort], "r") as handle:
            data = {f: handle[f][rows] for f in ("up", "up_quality", "weight", "epoch", "up_st_rho", "up_ph_rho")}
            guids = [S.guid_text(g) for g in handle["guid"][rows]]
        for j, row in enumerate(rows):
            bits = data["up_quality"][j].astype(np.int64)
            invalid = (bits & S.UP_INVALID_BITS) != 0
            events = C.detect(data["up"][j], data["weight"][j])
            phase, cidx = C.phase_labels(C.N_RAW, events)
            epoch = float(data["epoch"][j])
            runs = C.segment_runs(invalid)
            seg_rows.append({"cohort": cohort, "guid": guids[j], "row": int(row), "n_runs": len(runs),
                             "invalid_s": float(invalid.sum()) / S.FS, "n_contractions": len(events["peak_raw"])})
            strict_any = C.strict_step_bad(bits, data["up_st_rho"][j], data["up_ph_rho"][j]).any(axis=1)
            for start, stop in runs:
                seg_bits = bits[start:stop]
                counts = {name: int(((seg_bits & S.UP_BITS[name]) != 0).sum()) for name in S.UP_BITS}
                inv_counts = {n: counts[n] for n in INVALID_NAMES}
                dominant = max(inv_counts, key=lambda n: (inv_counts[n], -S.UP_BITS[n]))
                mid = (start + stop - 1) // 2
                ph_share = np.bincount(phase[start:stop], minlength=4) / (stop - start)
                iso = after_durations(cum, start, stop)
                sto = stored_after(strict_any, stop)
                chan_dur.append(iso["per_channel"])
                chan_onset.append(iso["onset_per_channel"])
                group_max = {g: int(iso["per_channel"][meta["group"] == g].max()) for g in ("fast", "mid", "slow")}
                run_rows.append({
                    "cohort": cohort, "guid": guids[j], "row": int(row), "epoch": epoch,
                    "start": int(start), "stop": int(stop), "len_s": (stop - start) / S.FS,
                    "abs_start_s": epoch + start / S.FS, "abs_stop_s": epoch + stop / S.FS,
                    "bits_present": "|".join(n for n in S.UP_BITS if counts[n]),
                    "invalid_bits_present": "|".join(n for n in INVALID_NAMES if counts[n]),
                    "n_invalid_codes": sum(1 for n in INVALID_NAMES if counts[n]),
                    "dominant": dominant,
                    "position": C.PHASES[int(phase[mid])],
                    "contraction": int(cidx[mid]),
                    "share_between": ph_share[0], "share_rise": ph_share[1], "share_peak": ph_share[2],
                    "share_fall": ph_share[3],
                    "touches_peak": bool(ph_share[2] > 0),
                    "at_seg_start": start == 0, "at_seg_end": stop == C.N_RAW,
                    "strict_after_iso_steps": iso["strict"],
                    "fast_after_iso_steps": group_max["fast"], "mid_after_iso_steps": group_max["mid"],
                    "slow_after_iso_steps": group_max["slow"],
                    "strict_after_stored_steps": sto["steps"], "strict_after_stored_censored": sto["censored"],
                    "len_class": C.len_class((stop - start) / S.FS),
                })
            for i, (o, p, e) in enumerate(zip(events["onset_raw"], events["peak_raw"], events["end_raw"])):
                inside = invalid[int(o):int(e) + 1]
                b = bits[int(o):int(e) + 1][inside]
                dom = ""
                if b.size:
                    cnt = {n: int(((b & S.UP_BITS[n]) != 0).sum()) for n in INVALID_NAMES}
                    dom = max(cnt, key=lambda n: (cnt[n], -S.UP_BITS[n]))
                con_rows.append({"cohort": cohort, "guid": guids[j], "row": int(row), "onset": int(o),
                                 "peak": int(p), "end": int(e), "dur_s": (int(e) - int(o)) / S.FS,
                                 "abs_peak_s": epoch + int(p) / S.FS,
                                 "amplitude": float(data["up"][j][int(p)] - data["up"][j][int(o)]),
                                 "invalid_s": float(inside.sum()) / S.FS,
                                 "touches_invalid": bool(inside.any()), "dominant": dom})
    runs_df = pd.DataFrame(run_rows)
    con_df = pd.DataFrame(con_rows)
    seg_df = pd.DataFrame(seg_rows)
    C.OUT1.mkdir(parents=True, exist_ok=True)
    runs_df.to_parquet(C.OUT1 / "t1_runs.parquet")
    con_df.to_parquet(C.OUT1 / "t1_contractions.parquet")
    seg_df.to_parquet(C.OUT1 / "t1_segments.parquet")
    np.savez_compressed(C.OUT1 / "t1_run_channel_after.npz", after_steps=np.stack(chan_dur),
                        onset_steps=np.stack(chan_onset), names=meta["name"], group=meta["group"],
                        warmup_steps=meta["warmup_steps"], delay_s=meta["delay_s"])
    summary = summarise(runs_df, con_df, seg_df, np.stack(chan_dur), np.stack(chan_onset), meta)
    summary["isolated_rho_check_max_abs_vs_bank"] = check
    (C.OUT1 / "t1_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def summarise(runs: pd.DataFrame, cons: pd.DataFrame, segs: pd.DataFrame, chan_dur: np.ndarray,
              chan_onset: np.ndarray, meta: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Headline numbers of T1.

    Args:
        runs: Run table.
        cons: Contraction table.
        segs: Segment table.
        chan_dur: ``(n_runs, 42)`` isolated post-run bad steps per channel.
        chan_onset: ``(n_runs, 42)`` first post-run bad step per channel.
        meta: Channel metadata.

    Returns:
        A JSON-ready dictionary.
    """
    out: Dict[str, Any] = {}
    dedup = runs.drop_duplicates(["cohort", "guid", "abs_start_s", "abs_stop_s"])
    out["n_segments"] = int(len(segs))
    out["n_recordings"] = int(segs[["cohort", "guid"]].drop_duplicates().shape[0])
    out["n_runs_per_segment"] = int(len(runs))
    out["n_runs_dedup"] = int(len(dedup))
    out["runs_per_segment"] = {"mean": float(segs.n_runs.mean()), "share_segments_with_run": float((segs.n_runs > 0).mean()),
                               "q50": float(segs.n_runs.median()), "q90": float(segs.n_runs.quantile(0.9)),
                               "max": int(segs.n_runs.max())}
    out["note_overlap"] = ("segments of one recording overlap by 120 s; per-segment counts double-count runs in "
                           "the overlap; the dedup count keys runs by guid + absolute start/stop")
    out["invalid_time_share_all_samples"] = float(segs.invalid_s.sum() / (len(segs) * C.N_RAW / S.FS))
    by_code = {}
    for code, g in runs.groupby("dominant"):
        by_code[code] = {"n": int(len(g)), "n_dedup": int(len(dedup[dedup.dominant == code])),
                         "total_invalid_h": float(g.len_s.sum() / 3600),
                         "share_of_runs": float(len(g) / len(runs)),
                         "share_of_invalid_time": float(g.len_s.sum() / runs.len_s.sum()),
                         "len_q": {q: float(g.len_s.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)}}
    out["by_dominant_code"] = by_code
    out["bit_presence_share_of_runs"] = {n: float(runs.invalid_bits_present.str.contains(n).mean()) for n in INVALID_NAMES}
    out["share_runs_multi_code"] = float((runs.n_invalid_codes > 1).mean())
    out["length"] = {"q": {q: float(runs.len_s.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
                     "share_le": {str(e): float((runs.len_s <= e).mean()) for e in C.LEN_EDGES},
                     "share_le_dedup": {str(e): float((dedup.len_s <= e).mean()) for e in C.LEN_EDGES},
                     "invalid_time_share_in_runs_le": {str(e): float(runs.len_s[runs.len_s <= e].sum() / runs.len_s.sum())
                                                        for e in C.LEN_EDGES}}
    out["edge_runs_share"] = {"at_seg_start": float(runs.at_seg_start.mean()), "at_seg_end": float(runs.at_seg_end.mean())}
    out["position_mix_all"] = runs.position.value_counts(normalize=True).to_dict()
    out["position_mix_by_code"] = {c: g.position.value_counts(normalize=True).to_dict() for c, g in runs.groupby("dominant")}
    out["position_mix_by_len_class"] = {c: g.position.value_counts(normalize=True).to_dict() for c, g in runs.groupby("len_class")}
    out["position_by_code_counts"] = pd.crosstab(runs.dominant, runs.position).to_dict()
    # contraction-level
    n_con = len(cons)
    touched = cons[cons.touches_invalid]
    dcons = cons.assign(key=(cons.abs_peak_s / 4).round()).drop_duplicates(["cohort", "guid", "key"])
    dtouched = dcons[dcons.touches_invalid]
    out["contractions"] = {
        "n_per_segment": int(n_con), "n_dedup": int(len(dcons)),
        "share_touch_invalid": float(len(touched) / max(n_con, 1)),
        "share_touch_invalid_dedup": float(len(dtouched) / max(len(dcons), 1)),
        "of_touched_share_invalid_le_8s": float((touched.invalid_s <= 8).mean()),
        "of_touched_share_invalid_le_8s_dedup": float((dtouched.invalid_s <= 8).mean()),
        "of_touched_invalid_s_q": {q: float(touched.invalid_s.quantile(q)) for q in (0.25, 0.5, 0.75, 0.9)},
        "of_touched_dominant": touched.dominant.value_counts(normalize=True).to_dict(),
        "amplitude_q": {q: float(cons.amplitude.quantile(q)) for q in (0.1, 0.5, 0.9)},
        "dur_s_q": {q: float(cons.dur_s.quantile(q)) for q in (0.1, 0.5, 0.9)},
        "per_cohort_share_touch": cons.groupby("cohort").touches_invalid.mean().to_dict(),
    }
    # post-run contamination
    step_s = S.STEP_S
    post: Dict[str, Any] = {}
    for lc, g in runs.groupby("len_class"):
        idx = g.index.values
        post[lc] = {
            "n": int(len(g)),
            "strict_iso_s_q": {q: float(g.strict_after_iso_steps.quantile(q) * step_s) for q in (0.25, 0.5, 0.75, 0.9)},
            "fast_iso_s_median": float(g.fast_after_iso_steps.median() * step_s),
            "mid_iso_s_median": float(g.mid_after_iso_steps.median() * step_s),
            "slow_iso_s_median": float(g.slow_after_iso_steps.median() * step_s),
            "stored_strict_s_median": float(g.strict_after_stored_steps.median() * step_s),
            "stored_censored_share": float(g.strict_after_stored_censored.mean()),
            "per_channel_iso_s_median": {n: float(np.median(chan_dur[idx, c]) * step_s) for c, n in enumerate(meta["name"])},
            "per_channel_first_bad_s_median": {n: float(np.median(np.where(chan_onset[idx, c] >= 0, chan_onset[idx, c], np.nan)) * step_s)
                                               if (chan_onset[idx, c] >= 0).any() else None for c, n in enumerate(meta["name"])},
        }
    out["post_run_contamination_by_len_class"] = post
    short = runs.len_s <= 8
    out["short_gap_le8_slow_claim"] = {
        "n": int(short.sum()),
        "slow_group_iso_s_median": float(runs.slow_after_iso_steps[short].median() * step_s),
        "slow_group_iso_s_q90": float(runs.slow_after_iso_steps[short].quantile(0.9) * step_s),
        "strict_iso_s_median": float(runs.strict_after_iso_steps[short].median() * step_s),
        "share_strict_iso_ge_60s": float((runs.strict_after_iso_steps[short] * step_s >= 60).mean()),
        "share_strict_iso_ge_120s": float((runs.strict_after_iso_steps[short] * step_s >= 120).mean()),
        "share_strict_iso_driven_by_slow": float((runs.slow_after_iso_steps[short] >= runs.strict_after_iso_steps[short]).mean()),
    }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=RUN_ARGS["device"])
    result = scan(parser.parse_args().device)
    print(json.dumps({k: result[k] for k in ("n_segments", "n_recordings", "n_runs_per_segment", "n_runs_dedup",
                                              "length", "contractions", "short_gap_le8_slow_claim",
                                              "isolated_rho_check_max_abs_vs_bank")}, indent=1, default=float))
