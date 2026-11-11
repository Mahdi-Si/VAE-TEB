r"""T2: inject synthetic invalid runs into clean UA and rebuild the source with the builder's code.

**Base segments.** Code-0/2 segments whose UA has no invalid sample, whose mean step weight is at
least :data:`RUN_ARGS` ``min_weight``, and with at least one detected contraction whose rise and
fall each last at least 20 s. At most ``per_recording`` segments per recording, drawn at random.
The *true* source of a base segment is the builder chain on its raw UA with an empty invalid mask
(this equals the stored source to $2.4\times10^{-7}$; see :mod:`p1_rebuild`).

**Gap rebuild.** The invalid mask is set directly to the injected gap; the raw UA keeps its true
values (the bridge never reads an invalid sample except through its neighbours). Level shifts
come from the builder's own ``_span_shifts`` with the gap marked ``TRANSIENT``. The bridged
source is :meth:`p1_rebuild.Rebuilder.rebuild` of (raw UA, gap mask, shifts).

**Grid set.** One gap per synthetic segment, length $L \in \{4, 8, 16, 32, 64, 128\}$ s, centred in
one phase of the base segment's chosen contraction (onset $o$, peak $p$, end $e$):
``peak`` at $p$; ``rise`` at the middle of $[o, p - 10\,\mathrm{s})$; ``fall`` at the middle of
$(p + 10\,\mathrm{s}, e]$; ``between`` at the middle of the longest stretch outside every
contraction. The phase of a gap is the phase of its middle sample, as in T1. The model decodes
the dense anchors $t \in [134, 270)$ of the trimmed grid, i.e. the stored raw samples
$240 + 16t \in [2384, 4544]$; the 38-key lag window of the first anchor starts at raw sample
1777. A grid gap must lie in the raw samples $[1792, 3936]$ (448--984 s), so anchors exist
inside the gap and for 152 s after its end. A cell that does not fit is skipped. Each base
segment uses the contraction that fits the most grid cells (ties: the one nearest the middle of
that span), and the longest between-stretch whose centre lies in the span.

**Observed-mix set.** Per base segment, ``mix_reps`` synthetic segments. Each draws its number of
gaps from the observed runs-per-segment distribution (segments with at least one run), and each
gap resamples one observed run (its dominant code, length and phase) from the T1 table. The gap
is placed with its middle at a random sample of that phase, anywhere in the raw samples
$[400, 4880]$ (the natural spread over the segment); gaps keep 4 s apart. Only anchors decoded by
the model see them.

Outputs in ``out/phase1/``: ``t2_injections.parquet`` (one row per gap), ``t2_synth.parquet``
(one row per synthetic segment), ``t2_base.npz`` (true arrays), ``t2_grid.npz`` and
``t2_mix.npz`` (bridged arrays, row order of ``t2_synth``).

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
RUN_ARGS: Dict[str, Any] = {"device": "cuda:0", "min_weight": 0.98, "per_recording": 2, "max_base": 160,
                            "mix_reps": 2, "seed": 11}

GRID_LEN_S = (4, 8, 16, 32, 64, 128)
GRID_POS = ("rise", "peak", "fall", "between")
LO_RAW = 1792
HI_RAW = 3936
MIX_LO = 400
MIX_HI = C.N_RAW - 400
MIN_FLANK = int(20 * S.FS)
HALF_PEAK = int(C.PEAK_HALF_S * S.FS)


def pick_base(min_weight: float, per_recording: int, max_base: int, rng: np.random.Generator) -> pd.DataFrame:
    """Choose the clean base segments and their target contraction.

    Args:
        min_weight: Smallest mean step weight.
        per_recording: Most segments per recording.
        max_base: Most base segments overall.
        rng: Random generator.

    Returns:
        One row per base segment with its contraction ``onset``, ``peak``, ``end`` and the centre of
        the longest between-stretch.
    """
    rows: List[Dict[str, Any]] = []
    for cohort in S.SHARDS:
        idx = C.split_rows(cohort)
        with h5py.File(S.SHARDS[cohort], "r") as handle:
            q = handle["up_quality"][idx]
            w = handle["weight"][idx]
            clean = ~((q.astype(np.int64) & S.UP_INVALID_BITS) != 0).any(axis=1) & (w.mean(axis=1) >= min_weight)
            cand = idx[clean]
            ups = handle["up"][cand] if cand.size else np.zeros((0, C.N_RAW))
            ws = w[clean]
            guids = [S.guid_text(g) for g in handle["guid"][cand]] if cand.size else []
        for j, row in enumerate(cand):
            ev = C.detect(ups[j], ws[j])
            phase, _ = C.phase_labels(C.N_RAW, ev)
            ok = [(int(o), int(p), int(e)) for o, p, e in zip(ev["onset_raw"], ev["peak_raw"], ev["end_raw"])
                  if p - HALF_PEAK - o >= MIN_FLANK and e - p - HALF_PEAK >= MIN_FLANK]
            if not ok:
                continue
            between = C.segment_runs(phase == 0)
            between = [(a, b) for a, b in between if LO_RAW <= (a + b) // 2 <= HI_RAW]
            if not between:
                continue
            a, b = max(between, key=lambda r: r[1] - r[0])
            mid_span = (LO_RAW + HI_RAW) // 2

            def n_fit(t):
                rec = pd.Series({"onset": t[0], "peak": t[1], "end": t[2], "between_centre": (a + b) // 2})
                return sum(place(centre_of(pos, rec), int(L * S.FS)) is not None
                           for L in GRID_LEN_S for pos in GRID_POS)

            o, p, e = max(ok, key=lambda t: (n_fit(t), -abs(t[1] - mid_span)))
            rows.append({"cohort": cohort, "guid": guids[j], "row": int(row), "onset": o, "peak": p, "end": e,
                         "between_centre": (a + b) // 2, "between_len_s": (b - a) / S.FS,
                         "n_contractions": int(len(ev["peak_raw"])),
                         "amplitude": float(ups[j][p] - ups[j][o])})
    base = pd.DataFrame(rows)
    base = base.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    base = base.groupby(["cohort", "guid"], sort=False).head(per_recording)
    base = base.head(max_base).sort_values(["cohort", "row"]).reset_index(drop=True)
    base["base_id"] = np.arange(len(base))
    return base


def centre_of(pos: str, rec: pd.Series) -> int:
    """Centre sample of a grid gap.

    Args:
        pos: Phase name.
        rec: Base-segment row.

    Returns:
        Raw sample index.
    """
    o, p, e = int(rec.onset), int(rec.peak), int(rec.end)
    if pos == "peak":
        return p
    if pos == "rise":
        return (o + p - HALF_PEAK) // 2
    if pos == "fall":
        return (p + HALF_PEAK + 1 + e) // 2
    return int(rec.between_centre)


def place(centre: int, length_raw: int) -> Optional[Tuple[int, int]]:
    """Gap ``[start, stop)`` centred at ``centre``, or ``None`` when it leaves the allowed span."""
    start = centre - length_raw // 2
    stop = start + length_raw
    if start < LO_RAW or stop > HI_RAW:
        return None
    return start, stop


def main(device: str, min_weight: float, per_recording: int, max_base: int, mix_reps: int, seed: int) -> Dict[str, Any]:
    """Build the base, grid and observed-mix sets and write them.

    Args:
        device: Torch device of the builder chain.
        min_weight: Smallest mean step weight of a base segment.
        per_recording: Most base segments per recording.
        max_base: Most base segments.
        mix_reps: Observed-mix segments per base segment.
        seed: Random seed.

    Returns:
        A short summary.
    """
    from teb_vae.neural_surgery_trf_cfs.session7.p1_rebuild import Rebuilder

    rng = np.random.default_rng(seed)
    rb = Rebuilder(device)
    base = pick_base(min_weight, per_recording, max_base, rng)
    print(f"base segments: {len(base)}", flush=True)
    raw = np.zeros((len(base), C.N_RAW))
    bits = np.zeros((len(base), C.N_RAW), dtype=np.uint16)
    phases = np.zeros((len(base), C.N_RAW), dtype=np.int8)
    weights = np.zeros((len(base), C.N_STEPS), dtype=np.float32)
    for cohort, g in base.groupby("cohort"):
        with h5py.File(S.SHARDS[cohort], "r") as handle:
            r = g.row.values
            raw[g.index.values] = handle["up_raw"][r].astype(np.float64)
            bits[g.index.values] = handle["up_quality"][r]
            weights[g.index.values] = handle["weight"][r]
    for i in range(len(base)):
        ev = C.detect(rb.bridge(raw[i:i + 1], np.zeros((1, C.N_RAW), bool))[0], weights[i])
        phases[i], _ = C.phase_labels(C.N_RAW, ev)
    true = rb.rebuild(raw, np.zeros_like(raw, dtype=bool), [[] for _ in range(len(base))])
    np.savez_compressed(C.OUT1 / "t2_base.npz", base_id=base.base_id.values, up=true["up"], up_st=true["up_st"],
                        up_ph=true["up_ph"], up_raw=raw.astype(np.float32), weight=weights, phase=phases)

    runs = pd.read_parquet(C.OUT1 / "t1_runs.parquet")
    segs = pd.read_parquet(C.OUT1 / "t1_segments.parquet")
    n_runs_pool = segs.n_runs[segs.n_runs > 0].values
    synth: List[Dict[str, Any]] = []
    gaps: List[Dict[str, Any]] = []
    sid = 0
    for _, rec in base.iterrows():
        for length in GRID_LEN_S:
            for pos in GRID_POS:
                span = place(centre_of(pos, rec), int(length * S.FS))
                if span is None:
                    continue
                mid_phase = C.PHASES[int(phases[rec.base_id][(span[0] + span[1] - 1) // 2])]
                synth.append({"synth_id": sid, "set": "grid", "base_id": int(rec.base_id), "cohort": rec.cohort,
                              "guid": rec.guid, "row": int(rec.row), "n_gaps": 1, "len_s": float(length),
                              "position": pos, "position_check": mid_phase})
                gaps.append({"synth_id": sid, "gap": 0, "start": span[0], "stop": span[1], "len_s": float(length),
                             "position": pos, "dominant": "GRID"})
                sid += 1
    for _, rec in base.iterrows():
        ph = phases[rec.base_id]
        for _ in range(mix_reps):
            n_gap = int(rng.choice(n_runs_pool))
            taken = np.zeros(C.N_RAW, dtype=bool)
            mine: List[Dict[str, Any]] = []
            for _attempt in range(n_gap * 20):
                if len(mine) >= n_gap:
                    break
                src = runs.iloc[int(rng.integers(len(runs)))]
                length_raw = int(src.stop - src.start)
                where = np.flatnonzero(ph == C.PHASES.index(src.position))
                where = where[(where - length_raw // 2 >= MIX_LO) & (where - length_raw // 2 + length_raw <= MIX_HI)]
                if where.size == 0:
                    continue
                centre = int(rng.choice(where))
                start = centre - length_raw // 2
                stop = start + length_raw
                pad = int(4 * S.FS)
                if taken[max(start - pad, 0):stop + pad].any():
                    continue
                taken[start:stop] = True
                mine.append({"start": start, "stop": stop, "len_s": length_raw / S.FS, "position": src.position,
                             "dominant": src.dominant})
            if not mine:
                continue
            mine.sort(key=lambda d: d["start"])
            for k, gp in enumerate(mine):
                gaps.append({"synth_id": sid, "gap": k, **gp})
            synth.append({"synth_id": sid, "set": "mix", "base_id": int(rec.base_id), "cohort": rec.cohort,
                          "guid": rec.guid, "row": int(rec.row), "n_gaps": len(mine),
                          "len_s": float(sum(g["len_s"] for g in mine)), "position": "mix", "position_check": "mix"})
            sid += 1
    synth_df = pd.DataFrame(synth)
    gaps_df = pd.DataFrame(gaps)
    out: Dict[str, Any] = {"n_base": int(len(base)), "n_base_recordings": int(base[["cohort", "guid"]].drop_duplicates().shape[0])}
    for name in ("grid", "mix"):
        part = synth_df[synth_df["set"] == name]
        masks = np.zeros((len(part), C.N_RAW), dtype=bool)
        shifts: List[List[int]] = []
        for i, s in enumerate(part.synth_id.values):
            for _, gp in gaps_df[gaps_df.synth_id == s].iterrows():
                masks[i, int(gp.start):int(gp.stop)] = True
            b = int(part.base_id.values[i])
            shifts.append(rb.gap_shifts(raw[b], bits[b], masks[i]))
        print(f"rebuilding {name}: {len(part)} segments", flush=True)
        built = rb.rebuild(raw[part.base_id.values], masks, shifts)
        synth_df.loc[part.index, "n_shifts"] = [len(s) for s in shifts]
        np.savez_compressed(C.OUT1 / f"t2_{name}.npz", synth_id=part.synth_id.values, invalid=np.packbits(masks, axis=1),
                            **built)
        out[f"n_{name}"] = int(len(part))
        out[f"{name}_with_span_shift"] = int(sum(1 for s in shifts if s))
    base.to_parquet(C.OUT1 / "t2_base.parquet")
    synth_df.to_parquet(C.OUT1 / "t2_synth.parquet")
    gaps_df.to_parquet(C.OUT1 / "t2_injections.parquet")
    grid = synth_df[synth_df["set"] == "grid"]
    out["grid_cells"] = grid.groupby(["len_s", "position"]).size().unstack().to_dict()
    out["grid_position_check_agree"] = float((grid.position == grid.position_check).mean())
    out["base_by_cohort"] = base.cohort.value_counts().to_dict()
    out["mix_gaps_by_code"] = gaps_df[gaps_df.dominant != "GRID"].dominant.value_counts().to_dict()
    (C.OUT1 / "t2_summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in RUN_ARGS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    a = parser.parse_args()
    main(a.device, a.min_weight, a.per_recording, a.max_base, a.mix_reps, a.seed)
