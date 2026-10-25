r"""Profile the raw CTG signals and show what the quality step changes. Read only.

The script reads a sample of records the way ``create_new_pipeline.py`` reads them, twice: once
as the builder does today (with ``MimoSequence.interpolate_input`` and ``_sanitize_signals``),
and once with the MIMO fill turned off, which gives the **pre-fill** signals. On the pre-fill
signals it runs :mod:`hdf5_dataset.signal_quality`: the per-sample codes, the level correction,
the bridge, and the contamination $\rho_c[k]$ of every stored coefficient. It writes no dataset.

The MIMO fill is turned off by patching the method on its class for the duration of one read.
Nothing in the ``mimo`` packages changes on disk.

Outputs in ``output_dir``:

* ``REPORT.md``: what the quality step changes, how the mask did, and the figure list;
* ``segments.csv``: one row per profiled segment;
* ``ua_code_runs.csv``: one row per UA run at 100, 127, 128 or 255;
* ``channel_validity.csv``: per subgroup and channel, the share of coefficients with
  $\rho \le \varepsilon$ after the warm-up;
* ``records.csv``, ``summary.json``, ``run_config.json``;
* ``figures/``: the summary figures;
* ``examples/``: one page per example segment, and ``examples.pdf`` with all of them.

Run it on the production box, where the records and the MIMO adaptor live. Edit
:data:`RUN_ARGS` and press Run, or pass the flags of :func:`build_parser`.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Repository root: ``hdf5_dataset/new_pipeline/profile_signal_quality.py`` -> up three.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Launched with the Run button, this file's own directory goes on sys.path instead of the
# repository root, and neither the quality module nor the launch helper can be found.
if not __package__ and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hdf5_dataset import signal_quality as sq  # noqa: E402
from teb_vae.lag_attn_rws.eval.launch import missing_required, resolve_launch_args  # noqa: E402

logger = logging.getLogger(__name__)

#: Raw samples per stored step.
DECIMATION = 16
#: UA values whose runs are listed one by one.
CODE_VALUES = (100.0, 127.0, 128.0, 255.0)
#: The four blocks whose coefficient validity is measured.
BLOCKS = ("up_st", "up_ph", "fhr_st", "fhr_ph")
SUBGROUP_ORDER = (
    "hie_no_cs", "hie_cs", "acidosis_no_cs", "acidosis_cs",
    "healthy_bg_no_cs", "healthy_bg_cs", "healthy_no_bg_no_cs", "healthy_no_bg_cs",
)
CLASS_OF = {
    "hie_no_cs": "HIE", "hie_cs": "HIE",
    "acidosis_no_cs": "Acidosis", "acidosis_cs": "Acidosis",
    "healthy_bg_no_cs": "Healthy, BG", "healthy_bg_cs": "Healthy, BG",
    "healthy_no_bg_no_cs": "Healthy, no BG", "healthy_no_bg_cs": "Healthy, no BG",
}
#: Example categories: name -> (title, segments.csv column ranked high to low).
EXAMPLE_CATEGORIES = {
    "code_128": ("UA invalid code (128 / 255)", "share_code"),
    "censored": ("UA clamped crest (censored)", "ua_censored"),
    "ceiling_artifact": ("UA ceiling entered by a jump", "n_ceiling_artifact_runs"),
    "step": ("UA baseline step", "n_steps"),
    "transient": ("UA transient", "ua_transient"),
    "hold": ("UA hold", "ua_hold"),
    "floor": ("UA floor", "ua_floor"),
    "mimo_overwrite": ("MIMO fill overwrote valid samples", "fill_overwritten_valid"),
    "fhr_repaired": ("FHR repair", "fhr_invalid_share"),
    "clean": ("Clean UA with contractions (control)", "n_contr_clean"),
}


def _pipeline() -> Any:
    """Import ``create_new_pipeline`` on first use; it needs the production packages."""
    from hdf5_dataset.new_pipeline import create_new_pipeline

    return create_new_pipeline


# =================================================================================================
# Reading
# =================================================================================================
@contextlib.contextmanager
def mimo_fill(enabled: bool) -> Iterator[None]:
    """Turn ``MimoSequence.interpolate_input`` off for the body of the ``with`` block.

    The patch is on the class, because ``prepare_data`` replaces the instance twice before it
    calls the method. The original method is restored on exit, also after an exception.

    Args:
        enabled: Keep the fill (``True``) or replace it with the identity (``False``).
    """
    cnp = _pipeline()
    module = sys.modules[cnp.EarlyMaestraMimoAdaptor.__module__]
    cls = module.MimoSequence
    original = cls.interpolate_input
    if not enabled:
        cls.interpolate_input = lambda self: self
    try:
        yield
    finally:
        cls.interpolate_input = original


def load_record(record_path: str) -> Dict[str, Any]:
    """Read one record as the builder does, with and without the MIMO fill.

    Args:
        record_path: The ``.mat`` file.

    Returns:
        A dict with ``fhr_now``/``up_now`` (filled and sanitised, as stored today),
        ``fhr_pre``/``up_pre`` (no fill, no sanitising), ``weight`` ``(n, 330)``,
        ``domain_start`` ``(n,)``, ``present`` ``(n, N)``, and ``gate`` and ``gate_prefill``:
        the builder's gate decision on today's signals and on the pre-fill signals (``kept``,
        ``low_weight``, ``flat`` or ``post_delivery`` per segment). Duplicates are already removed.

    Raises:
        ValueError: If the two reads disagree on the segment list.
    """
    cnp = _pipeline()
    with mimo_fill(True):
        fhr_now, up_now, ds_now, w_now = cnp._run_mimo_pipeline(
            record_path, cnp.MIN_DOMAIN_START_DATASET, verbose=False)
    with mimo_fill(False):
        fhr_pre, up_pre, ds_pre, w_pre = cnp._run_mimo_pipeline(
            record_path, cnp.MIN_DOMAIN_START_DATASET, verbose=False)
    if list(ds_now) != list(ds_pre) or fhr_now.shape != fhr_pre.shape:
        raise ValueError(f"the reads with and without the fill disagree on the segments of "
                         f"{record_path}")
    w_now, w_pre = np.asarray(w_now), np.asarray(w_pre)
    if not np.array_equal(w_now, w_pre):
        raise ValueError(f"the reads with and without the fill disagree on the weights of "
                         f"{record_path}")
    fhr_now, up_now = cnp._sanitize_signals(fhr_now, up_now)
    keep, _, _ = cnp.deduplicate_segments(list(ds_now), w_now)
    keep = list(keep)
    fhr_now, up_now = fhr_now[keep], up_now[keep]
    # A non-finite pre-fill sample is a gap like a zero: it is not present, and it reads as 0.
    fhr_pre = np.nan_to_num(np.asarray(fhr_pre, dtype=np.float64)[keep], nan=0.0, posinf=0.0, neginf=0.0)
    up_pre = np.nan_to_num(np.asarray(up_pre, dtype=np.float64)[keep], nan=0.0, posinf=0.0, neginf=0.0)
    weight = w_now[keep]
    domain_start = np.asarray(ds_now, dtype=np.float64)[keep]
    n = fhr_pre.shape[1]
    present = np.repeat(weight > 0, DECIMATION, axis=1)[:, :n] & (fhr_pre != 0)
    rows = range(fhr_now.shape[0])
    return {"fhr_now": np.asarray(fhr_now, dtype=np.float64), "up_now": np.asarray(up_now, dtype=np.float64),
            "fhr_pre": fhr_pre, "up_pre": up_pre, "weight": weight, "domain_start": domain_start,
            "present": present,
            "gate": [_gate(cnp, fhr_now[i], up_now[i], weight[i], domain_start[i]) for i in rows],
            "gate_prefill": [_gate(cnp, fhr_pre[i], up_pre[i], weight[i], domain_start[i]) for i in rows]}


def _gate(cnp: Any, fhr: np.ndarray, up: np.ndarray, weight: np.ndarray, domain_start: float) -> str:
    """Apply the builder's segment gates, in the builder's order, to one segment.

    Returns:
        ``low_weight``, ``flat``, ``post_delivery`` or ``kept``.
    """
    if np.mean(weight) < cnp.WEIGHT_THRESHOLD:
        return "low_weight"
    fhr_lens = [b - a + 1 for a, b in cnp.find_flat_regions(fhr, tolerance=cnp.FLAT_TOLERANCE)]
    up_lens = [b - a + 1 for a, b in cnp.find_flat_regions(up, tolerance=cnp.FLAT_TOLERANCE)]
    if (max(fhr_lens, default=0) > 480 or max(up_lens, default=0) > 1200
            or sum(v for v in fhr_lens if v >= 240) > 1200):
        return "flat"
    return "post_delivery" if domain_start >= 0 else "kept"


def _series(segdat: Any) -> Tuple[int, np.ndarray]:
    """Lay one ``segdat`` channel on an absolute 4 Hz index grid.

    Args:
        segdat: A channel's ``segdat`` struct from ``efmStruct``.

    Returns:
        ``(first_index, values)``: the grid index of ``values[0]``, and the values with $-1$
        where the channel has no sample.
    """
    incr = float(segdat.domainIncr)
    starts = np.atleast_1d(np.asarray(segdat.domainStart, dtype=np.float64))
    onsets = np.atleast_1d(np.asarray(segdat.parameterSet.onsetPointer, dtype=np.int64)) - 1
    lengths = np.atleast_1d(np.asarray(segdat.parameterSet.segLength, dtype=np.int64))
    data = np.asarray(segdat.dataSet, dtype=np.float64).ravel()
    first = np.round(starts / incr).astype(np.int64)
    origin = int(first.min())
    values = np.full(int((first + lengths).max() - origin), -1.0)
    for g, onset, length in zip(first, onsets, lengths):
        values[g - origin:g - origin + length] = data[onset:onset + length]
    return origin, values


def _lookup(origin: int, values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Read ``values`` at absolute grid indices, $-1$ outside the channel."""
    local = index - origin
    out = np.full(index.shape, -1.0)
    inside = (local >= 0) & (local < values.size)
    out[inside] = values[local[inside]]
    return out


def raw_fhr_repaired(record_path: str, domain_start: np.ndarray, fhr_pre: np.ndarray,
                     present: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Mark the FHR samples that the upstream repair wrote, from the raw channel.

    The adaptor reads the last of ``Repaired HR1``, ``Repaired HR2`` and ``Repaired external
    FECG`` present. The raw channel is the same name without ``Repaired``. Each segment is first
    placed on the repaired channel's grid (the offset with the most exact matches, within
    $\\pm 4$ samples), then the raw channel is read at one record-wide offset within $\\pm 1$
    sample. A present sample is repaired if the raw value there is absent, 0, 255, or different.

    Args:
        record_path: The ``.mat`` file.
        domain_start: Segment start times in seconds.
        fhr_pre: Pre-fill FHR ``(n, N)``.
        present: Present samples ``(n, N)``.

    Returns:
        ``(repaired, info)``: a boolean ``(n, N)`` array or ``None`` when the raw channel is
        missing, and the offsets and match rates for ``records.csv``.
    """
    from scipy.io import loadmat

    mat = loadmat(record_path, squeeze_me=True, struct_as_record=False)["efmStruct"]
    signals = {sig.measure: sig for sig in np.atleast_1d(mat.signals)}
    chosen = [k for k in ("Repaired HR1", "Repaired HR2", "Repaired external FECG") if k in signals]
    if not chosen:
        return None, {"raw_fhr": "no repaired channel"}
    raw_key = chosen[-1].replace("Repaired ", "")
    if raw_key not in signals:
        return None, {"raw_fhr": f"no {raw_key!r} channel"}
    rep_origin, rep_values = _series(signals[chosen[-1]].segdat)
    raw_origin, raw_values = _series(signals[raw_key].segdat)

    n_seg, n = fhr_pre.shape
    grids = []
    placed = 0
    for i in range(n_seg):
        base = int(np.floor(domain_start[i] / 0.25)) + np.arange(n)
        best, best_hits = 0, -1
        for offset in range(-4, 5):
            hits = int(np.sum(_lookup(rep_origin, rep_values, base + offset)[present[i]]
                              == fhr_pre[i][present[i]]))
            if hits > best_hits:
                best, best_hits = offset, hits
        grids.append(base + best)
        placed += best_hits
    raw_offset, raw_hits = 0, -1
    for offset in (-1, 0, 1):
        hits = sum(int(np.sum(_lookup(raw_origin, raw_values, grids[i] + offset)[present[i]]
                              == fhr_pre[i][present[i]])) for i in range(n_seg))
        if hits > raw_hits:
            raw_offset, raw_hits = offset, hits
    repaired = np.zeros((n_seg, n), dtype=bool)
    for i in range(n_seg):
        raw = _lookup(raw_origin, raw_values, grids[i] + raw_offset)
        repaired[i] = present[i] & ((raw <= 0) | (raw >= 255) | (raw != fhr_pre[i]))
    total = max(int(present.sum()), 1)
    return repaired, {"raw_fhr": raw_key, "raw_offset": raw_offset,
                      "repaired_grid_match": placed / total, "raw_match": raw_hits / total}


# =================================================================================================
# Transform kit
# =================================================================================================
def build_kit(transform: str, leg_alignment: str, phase_operator: str) -> Dict[str, Any]:
    """Build the causal bank, the channel plan and the contamination kernels once.

    Args:
        transform: Must be ``'causal'``: the contamination needs one-sided kernels.
        leg_alignment: Phase leg alignment of the build.
        phase_operator: Phase operator of the build.

    Returns:
        A dict with ``masks`` (from ``compute_scattering_masks``), ``kernels`` (block -> list),
        ``warmup`` (block -> steps per channel), ``hz`` (block -> channel frequency in Hz) and
        ``pairs``.
    """
    import torch

    cnp = _pipeline()
    if transform != cnp.CAUSAL:
        raise ValueError("the contamination needs the causal transform; set transform='causal'")
    masks = cnp.compute_scattering_masks(
        cnp.SIGNAL_LENGTH, scattering_T=16, device=torch.device("cpu"), transform=transform,
        leg_alignment=leg_alignment, phase_operator=phase_operator)
    bank, plan = masks["causal_bank"], masks["channel_plan"]
    pairs = {"fhr_ph": cnp._selection_pairs(masks["fhr_ph_selection"]),
             "up_ph": cnp._selection_pairs(masks["up_ph_selection"])}
    resolve = sys.modules[type(bank).__module__].resolve_leg_alignment
    pair_shift = None
    if leg_alignment != "none":
        pair_shift = {block: resolve(bank, pairs[block], leg_alignment)[0] for block in pairs}
    kept = {block: np.asarray(plan[block].kept) for block in BLOCKS}
    kernels = sq.magnitude_kernels(bank.psi, bank.phi, kept, pairs, pair_shift)
    hz = {}
    for block in BLOCKS:
        if block.endswith("_st"):
            hz[block] = np.array([0.0 if c == 0 else float(bank.hz[c - 1]) for c in kept[block]])
        else:
            hz[block] = np.array([float(bank.hz[int(pairs[block][c][0])]) for c in kept[block]])
    return {"masks": masks, "kernels": kernels, "pairs": pairs,
            "warmup": {block: np.asarray(plan[block].warmup_steps) for block in BLOCKS},
            "hz": hz, "leg_alignment": leg_alignment, "phase_operator": phase_operator}


_WORKER: Dict[str, Any] = {}


def _init_worker(kernels: Mapping[str, List[np.ndarray]], warmup: Mapping[str, np.ndarray],
                 epsilon: float, read_raw_fhr: bool, n_signal: int) -> None:
    """Pool initializer: build the contamination banks once per process."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _WORKER.update(
        banks={block: sq.ContaminationBank(kernels[block], n_signal, DECIMATION) for block in BLOCKS},
        warmup={block: np.asarray(warmup[block]) for block in BLOCKS},
        epsilon=float(epsilon), read_raw_fhr=bool(read_raw_fhr))


# =================================================================================================
# Per record
# =================================================================================================
def _code_runs(x: np.ndarray, present: np.ndarray, bits: np.ndarray) -> List[Dict[str, Any]]:
    """One row per run of a UA code value, with how it was entered and what the rules decided."""
    lag = 4
    rows = []
    for value in CODE_VALUES:
        for start, stop in sq.runs(present & (x == value)):
            entry = float(x[start - 1]) if start > 0 and present[start - 1] else np.nan
            jump = float(value - x[start - lag]) if start >= lag and present[start - lag] else np.nan
            exit_value = float(x[stop]) if stop < x.size and present[stop] else np.nan
            head = int(bits[start])
            if head & sq.UaBit.CODE:
                decided = "code"
            elif head & (sq.UaBit.TRANSIENT | sq.UaBit.STEP):
                decided = "artifact"
            elif head & sq.UaBit.CENSORED:
                decided = "censored"
            else:
                decided = "other"
            rows.append({"value": value, "start_s": start / 4.0, "length_s": (stop - start) / 4.0,
                         "entry_value": entry, "entry_jump": jump, "exit_value": exit_value,
                         "decided": decided})
    return rows


def _coefficient_validity(rho: np.ndarray, warmup: np.ndarray, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per channel: valid coefficients after the warm-up, and the post-warm-up step count.

    Args:
        rho: ``(B, C, T)`` contamination.
        warmup: ``(C,)`` warm-up steps.
        epsilon: Validity threshold.

    Returns:
        ``(valid, total)``: ``(B, C)`` counts each.
    """
    steps = np.arange(rho.shape[-1])
    after = steps[None, :] >= warmup[:, None]
    valid = ((rho <= epsilon) & after[None]).sum(axis=-1)
    total = np.broadcast_to(after.sum(axis=-1)[None, :], valid.shape)
    return valid, total


def profile_record(record_path: str, subgroup: str) -> Dict[str, Any]:
    """Profile every segment of one record.

    Args:
        record_path: The ``.mat`` file.
        subgroup: Its subgroup.

    Returns:
        ``{"record": {...}, "segments": [...], "runs": [...], "channels": {block: (valid, total)}}``
        where ``valid`` and ``total`` are per-channel sums over the record's kept segments.
    """
    from teb_vae.lag_attn_rws.eval.events import detect_contractions, raw_validity

    guid = Path(record_path).stem
    started = time.time()
    record: Dict[str, Any] = {"guid": guid, "subgroup": subgroup, "record_path": record_path,
                              "error": ""}
    try:
        rec = load_record(record_path)
    except Exception as exc:  # noqa: BLE001 -- reported per record
        record["error"] = f"{type(exc).__name__}: {exc}"
        return {"record": record, "segments": [], "runs": [], "channels": {}}

    profiled = [i for i, ds in enumerate(rec["domain_start"]) if ds < 0]
    record.update(n_segments=len(rec["gate"]), n_kept=rec["gate"].count("kept"),
                  n_kept_prefill=rec["gate_prefill"].count("kept"))
    repaired = None
    if _WORKER["read_raw_fhr"] and profiled:
        try:
            repaired, info = raw_fhr_repaired(record_path, rec["domain_start"], rec["fhr_pre"],
                                              rec["present"])
            record.update(info)
        except Exception as exc:  # noqa: BLE001
            record["raw_fhr"] = f"failed: {type(exc).__name__}: {exc}"

    results = []
    for i in profiled:
        r = sq.clean_segment(rec["fhr_pre"][i], rec["up_pre"][i], rec["present"][i],
                             None if repaired is None else repaired[i])
        results.append(r)
    segments, runs_rows = [], []
    channels: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if not results:
        record["seconds"] = round(time.time() - started, 1)
        return {"record": record, "segments": [], "runs": [], "channels": {}}

    up_bad = np.stack([r.up_invalid for r in results])
    fhr_bad = np.stack([r.fhr_invalid for r in results])
    validity = {}
    for block in BLOCKS:
        rho = _WORKER["banks"][block](up_bad if block.startswith("up") else fhr_bad)
        validity[block] = _coefficient_validity(rho, _WORKER["warmup"][block], _WORKER["epsilon"])

    n = rec["fhr_pre"].shape[1]
    for k, (i, r) in enumerate(zip(profiled, results)):
        present = rec["present"][i]
        x_pre, x_now = rec["up_pre"][i], rec["up_now"][i]
        f_pre, f_now = rec["fhr_pre"][i], rec["fhr_now"][i]
        up_changed = x_now != x_pre
        fhr_changed = f_now != f_pre
        overwritten = (up_changed & (x_pre > 1e-4)) | (fhr_changed & (f_pre > 1e-4))
        row: Dict[str, Any] = {
            "guid": guid, "subgroup": subgroup, "segment": i, "domain_start": rec["domain_start"][i],
            "gate": rec["gate"][i], "kept": rec["gate"][i] == "kept",
            "gate_prefill": rec["gate_prefill"][i], "kept_prefill": rec["gate_prefill"][i] == "kept",
            "weight_mean": float(rec["weight"][i].mean()), "present_share": float(present.mean()),
            "fill_up_changed": int(up_changed.sum()), "fill_fhr_changed": int(fhr_changed.sum()),
            "fill_overwritten_valid": int((overwritten & present).sum()),
            "fill_fhr_overwritten_s": float((fhr_changed & (f_pre > 1e-4) & present).sum() / 4.0),
            "fill_up_negative": int((x_now < 0).sum()),
            "fill_up_all_zero": bool(np.all(x_now[present] == 0)) if present.any() else False,
        }
        xp = x_pre[present]
        row.update({
            "share_0": float(np.mean(xp == 0)), "share_100": float(np.mean(xp == 100)),
            "share_101_126": float(np.mean((xp > 100) & (xp < 127))),
            "share_127": float(np.mean(xp == 127)), "share_128": float(np.mean(xp == 128)),
            "share_255": float(np.mean(xp == 255)), "share_code": float(np.mean(xp > 127.5)),
        })
        row.update({f"ua_{name}": v for name, v in sq.bit_shares(r.up_bits, sq.UaBit, present).items()})
        row.update({f"fhr_{name}": v for name, v in sq.bit_shares(r.fhr_bits, sq.FhrBit, present).items()})
        row["ua_valid_share"] = float(np.mean(~r.up_invalid[present])) if present.any() else 0.0
        row["fhr_valid_share"] = float(np.mean(~r.fhr_invalid[present])) if present.any() else 0.0
        row["fhr_invalid_share"] = 1.0 - row["fhr_valid_share"]
        row["n_steps"] = len(r.steps)
        transient_runs = sq.runs((r.up_bits & int(sq.UaBit.TRANSIENT)) != 0)
        row["n_transient_runs"] = len(transient_runs)
        row["n_short_transient_runs"] = sum(1 for a, b in transient_runs if b - a <= 40)
        for block in BLOCKS:
            valid, total = validity[block]
            row[f"coef_valid_{block}"] = float(valid[k].sum() / max(int(total[k].sum()), 1))
        if repaired is not None:
            rep = repaired[i] & present
            heuristic = (r.fhr_bits & int(sq.FhrBit.HOLD | sq.FhrBit.LINEAR | sq.FhrBit.JUMP)) != 0
            row["fhr_repaired_n"] = int(rep.sum())
            row["fhr_heuristic_n"] = int((heuristic & present).sum())
            row["fhr_heuristic_hits"] = int((heuristic & rep).sum())

        # Contractions: the detector the training weight uses, on today's UA and on the clean UA.
        valid_now = raw_validity(rec["weight"][i], decimation=DECIMATION, raw_len=n)
        now = detect_contractions(x_now, valid=valid_now)
        clean = detect_contractions(r.up_clean, valid=present & ~r.up_invalid)
        on_bad = 0
        for onset, end in zip(now["onset_raw"], now["end_raw"]):
            if np.mean(r.up_invalid[int(onset):int(end) + 1]) > 0.5:
                on_bad += 1
        row.update(n_contr_now=int(now["onset_raw"].size), n_contr_clean=int(clean["onset_raw"].size),
                   n_contr_now_on_invalid=on_bad)

        seg_runs = _code_runs(x_pre, present, r.up_bits)
        row["n_ceiling_artifact_runs"] = sum(
            1 for run in seg_runs if run["value"] in (100.0, 127.0) and run["decided"] == "artifact")
        for run in seg_runs:
            run.update(guid=guid, subgroup=subgroup, segment=i, kept=row["kept"])
        runs_rows.extend(seg_runs)
        segments.append(row)

    kept_rows = np.array([rec["gate"][i] == "kept" for i in profiled])
    for block in BLOCKS:
        valid, total = validity[block]
        channels[block] = (valid[kept_rows].sum(axis=0), total[kept_rows].sum(axis=0))
    record["seconds"] = round(time.time() - started, 1)
    return {"record": record, "segments": segments, "runs": runs_rows, "channels": channels}


def _profile_task(task: Tuple[str, str]) -> Dict[str, Any]:
    """Process-pool entry point."""
    return profile_record(*task)


# =================================================================================================
# Record sample
# =================================================================================================
def select_records(screening_csv_path: Optional[str], record_paths: Optional[Sequence[str]],
                   per_subgroup: int, eligible_only: bool, seed: int) -> pd.DataFrame:
    """Choose the records to profile.

    Args:
        screening_csv_path: ``guid_screening_results.csv`` of a pipeline run.
        record_paths: Explicit ``.mat`` paths; used instead of the CSV when given.
        per_subgroup: Records drawn per subgroup from the CSV.
        eligible_only: Draw only records the pipeline found eligible (``eligible_2h`` for the
            unhealthy subgroups, ``eligible_3h`` for the healthy ones).
        seed: Seed of the draw.

    Returns:
        A frame with ``record_path`` and ``subgroup``.
    """
    if record_paths:
        cnp = _pipeline()
        rows = []
        for path in record_paths:
            parts = Path(path).parts
            subgroup = next((cnp.FOLDER_TO_SUBGROUP[p] for p in parts if p in cnp.FOLDER_TO_SUBGROUP),
                            "unknown")
            rows.append({"record_path": str(path), "subgroup": subgroup})
        return pd.DataFrame(rows)
    df = pd.read_csv(screening_csv_path, dtype={"guid": str, "record_path": str, "subgroup": str})
    df = df[~df["error"].astype(str).str.lower().isin(["true", "1"])]
    if eligible_only:
        cnp = _pipeline()
        healthy = df["subgroup"].isin(list(cnp.HEALTHY_SUBGROUPS))
        ok = np.where(healthy, df["eligible_3h"].astype(str).str.lower() == "true",
                      df["eligible_2h"].astype(str).str.lower() == "true")
        df = df[ok]
    picks = []
    for subgroup, group in df.groupby("subgroup", sort=True):
        take = min(int(per_subgroup), len(group))
        picks.append(group.sample(n=take, random_state=seed + sum(map(ord, str(subgroup)))))
    return pd.concat(picks)[["record_path", "subgroup"]].reset_index(drop=True)


# =================================================================================================
# Figures
# =================================================================================================
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
CLASS_COLORS = {"HIE": "#2a78d6", "Acidosis": "#eb6834", "Healthy, BG": "#1baf7a",
                "Healthy, no BG": "#eda100"}


def _style() -> Any:
    """Import matplotlib headless and apply the report style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 8.5,
        "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "grid.color": GRID, "grid.linewidth": 0.6,
    })
    return plt


def _subgroups(df: pd.DataFrame) -> List[str]:
    """Subgroups present in ``df``, in the fixed order."""
    present = set(df["subgroup"])
    return [s for s in SUBGROUP_ORDER if s in present] + sorted(present - set(SUBGROUP_ORDER))


def _grouped_bars(ax: Any, table: pd.DataFrame, colors: Sequence[str], ylabel: str,
                  percent: bool = True, log: bool = False) -> None:
    """Draw ``table`` (rows = x categories, columns = series) as grouped bars with a legend."""
    n_rows, n_cols = table.shape
    width = 0.8 / max(n_cols, 1)
    x = np.arange(n_rows)
    for j, column in enumerate(table.columns):
        values = table[column].to_numpy(dtype=float) * (100.0 if percent else 1.0)
        ax.bar(x + (j - (n_cols - 1) / 2) * width, values, width=width * 0.92, color=colors[j],
               label=str(column), edgecolor=SURFACE, linewidth=0.5)
    ax.set_xticks(x, [str(i).replace("_", " ") for i in table.index], rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("symlog", linthresh=0.01 if percent else 0.1)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    # Outside on the right: constrained layout reserves the room, so it never meets the title.
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5)


def figure_code_values(plt: Any, seg: pd.DataFrame, out: Path) -> str:
    """Share of present UA samples at each code value, per subgroup (kept segments)."""
    kept = seg[seg["kept"]]
    cols = {"share_100": "= 100", "share_101_126": "101–126", "share_127": "= 127",
            "share_128": "= 128", "share_255": "= 255"}
    table = pd.DataFrame({label: kept.groupby("subgroup").apply(
        lambda g, c=c: np.average(g[c], weights=g["present_share"].clip(lower=1e-9)), include_groups=False)
        for c, label in cols.items()}).reindex(_subgroups(kept))
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    _grouped_bars(ax, table, CATEGORICAL, "share of present UA samples (%)", log=True)
    ax.set_title("UA values at or above 100, by subgroup (pre-fill, kept segments)")
    path = out / "01_ua_code_values.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def figure_code_runs(plt: Any, runs_df: pd.DataFrame, out: Path) -> str:
    """Run length against entry jump for each code value, coloured by the rule's decision."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), layout="constrained", sharey=True)
    colors = {"censored": CATEGORICAL[2], "artifact": CATEGORICAL[1], "code": CATEGORICAL[0],
              "other": MUTED}
    for ax, value in zip(axes, CODE_VALUES):
        sub = runs_df[(runs_df["value"] == value) & runs_df["kept"]]
        for decided, color in colors.items():
            part = sub[sub["decided"] == decided]
            if len(part):
                ax.scatter(part["length_s"], part["entry_jump"], s=9, color=color, alpha=0.6,
                           edgecolors="none", label=f"{decided} ({len(part)})")
        ax.set_xscale("log")
        ax.set_title(f"UA = {int(value)}  ({len(sub)} runs)")
        ax.set_xlabel("run length (s)")
        ax.axhline(15, color=AXIS, linewidth=0.8, linestyle="--")
        ax.grid(True)
        ax.set_axisbelow(True)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("rise in the 1 s before the run (units)")
    fig.suptitle("How each UA code run starts: a jump (≥ 15, dashed) or a smooth rise",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    path = out / "02_ua_code_runs.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def figure_fill(plt: Any, seg: pd.DataFrame, out: Path) -> str:
    """What the MIMO fill does to the kept segments, per subgroup."""
    kept = seg[seg["kept"]]
    g = kept.groupby("subgroup")
    table = pd.DataFrame({
        "UA changed": g["fill_up_changed"].apply(lambda s: np.mean(s > 0)),
        "FHR changed": g["fill_fhr_changed"].apply(lambda s: np.mean(s > 0)),
        "valid samples overwritten": g["fill_overwritten_valid"].apply(lambda s: np.mean(s > 0)),
        "UA negative after fill": g["fill_up_negative"].apply(lambda s: np.mean(s > 0)),
        "UA all zero after fill": g["fill_up_all_zero"].apply(np.mean),
    }).reindex(_subgroups(kept))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained",
                             gridspec_kw={"width_ratios": [3, 2]})
    _grouped_bars(axes[0], table, CATEGORICAL, "share of kept segments (%)")
    axes[0].set_title("Segments the MIMO fill changes")
    affected = kept[kept["fill_overwritten_valid"] > 0]
    order = [s for s in _subgroups(kept) if s in set(affected["subgroup"])]
    if order:
        data = [affected.loc[affected["subgroup"] == s, "fill_overwritten_valid"] / 4.0 for s in order]
        axes[1].boxplot(data, vert=True, widths=0.6, showfliers=True,
                        medianprops={"color": INK}, boxprops={"color": CATEGORICAL[0]},
                        whiskerprops={"color": CATEGORICAL[0]}, capprops={"color": CATEGORICAL[0]},
                        flierprops={"markersize": 2, "markeredgecolor": MUTED})
        axes[1].set_xticks(np.arange(1, len(order) + 1), [s.replace("_", " ") for s in order],
                           rotation=30, ha="right")
        axes[1].set_yscale("log")
    axes[1].set_ylabel("valid samples overwritten per segment (s)")
    axes[1].set_title("How much valid signal it overwrites")
    axes[1].grid(axis="y")
    path = out / "03_mimo_fill.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def figure_bits(plt: Any, seg: pd.DataFrame, out: Path, prefix: str, flags: Any, title: str,
                name: str) -> str:
    """Heatmap of the share of present samples carrying each bit, subgroup by bit."""
    from matplotlib.colors import LinearSegmentedColormap

    kept = seg[seg["kept"]]
    columns = [f"{prefix}_{f.name.lower()}" for f in flags if f.name != "MISSING"]
    table = kept.groupby("subgroup")[columns].apply(
        lambda g: pd.Series(np.average(g, axis=0, weights=kept.loc[g.index, "present_share"].clip(lower=1e-9)),
                            index=columns)).reindex(_subgroups(kept)) * 100.0
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(1.1 * len(columns) + 3, 0.42 * len(table) + 1.6), layout="constrained")
    values = table.to_numpy()
    image = ax.imshow(np.log10(values + 0.01), cmap=cmap, aspect="auto", vmin=-2, vmax=2)
    for (r, c), v in np.ndenumerate(values):
        color = SURFACE if np.log10(v + 0.01) > 0.6 else INK
        ax.text(c, r, f"{v:.2f}" if v < 10 else f"{v:.1f}", ha="center", va="center", fontsize=7.5, color=color)
    ax.set_xticks(np.arange(len(columns)), [c.split("_", 1)[1] for c in columns])
    ax.set_yticks(np.arange(len(table)), [s.replace("_", " ") for s in table.index])
    ax.set_title(title)
    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    bar.set_ticks([-2, -1, 0, 1, 2], labels=["0.01", "0.1", "1", "10", "100"])
    bar.set_label("% of present samples")
    path = out / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def figure_valid_share(plt: Any, seg: pd.DataFrame, out: Path) -> str:
    """Per-segment valid share of UA and FHR by subgroup: the shortcut check."""
    kept = seg[seg["kept"]]
    order = _subgroups(kept)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained", sharey=True)
    for ax, column, title in ((axes[0], "ua_valid_share", "UA"), (axes[1], "fhr_valid_share", "FHR")):
        data = [kept.loc[kept["subgroup"] == s, column] * 100 for s in order]
        colors = [CLASS_COLORS.get(CLASS_OF.get(s, ""), MUTED) for s in order]
        parts = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True,
                           medianprops={"color": INK})
        for patch, color in zip(parts["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor(color)
        means = [d.mean() if len(d) else np.nan for d in data]
        ax.scatter(np.arange(1, len(order) + 1), means, marker="D", s=14, color=INK, zorder=3,
                   label="mean")
        ax.set_xticks(np.arange(1, len(order) + 1), [s.replace("_", " ") for s in order],
                      rotation=30, ha="right")
        ax.set_title(f"{title}: valid share of present samples per kept segment")
        ax.grid(axis="y")
        ax.legend(loc="lower left", fontsize=7.5)
    axes[0].set_ylabel("valid samples (%)")
    path = out / "06_valid_share_by_subgroup.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def figure_channels(plt: Any, chan: pd.DataFrame, out: Path, epsilon: float) -> str:
    """Share of valid coefficients per channel against the channel frequency, by class."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, block in zip(axes.ravel(), BLOCKS):
        sub = chan[chan["block"] == block]
        for cls, color in CLASS_COLORS.items():
            part = sub[sub["class"] == cls]
            if not len(part):
                continue
            agg = part.groupby(["channel", "hz"], as_index=False)[["valid", "total"]].sum()
            agg = agg[agg["hz"] > 0].sort_values("hz")
            share = 100 * agg["valid"] / agg["total"].clip(lower=1)
            if block.endswith("_st"):
                ax.plot(agg["hz"], share, color=color, linewidth=1.6, marker="o", markersize=2.5,
                        label=cls)
            else:
                # Several phase pairs share a slower leg, so a line would zig-zag between them.
                ax.scatter(agg["hz"], share, color=color, s=14, alpha=0.8, edgecolors="none",
                           label=cls)
        ax.set_xscale("log")
        ax.set_title(block)
        ax.set_xlabel("channel centre frequency (Hz; slower leg for phase)")
        ax.set_ylabel(f"coefficients with ρ ≤ {epsilon:g} (%)")
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.axvspan(0.004, 0.05, color=GRID, alpha=0.5, linewidth=0)
    axes[0, 0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle("Usable coefficients after the warm-up, per channel (gray band: contraction band)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    path = out / "07_coefficient_validity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def figure_contractions(plt: Any, seg: pd.DataFrame, out: Path) -> str:
    """Detected contractions per kept segment, today against after cleaning."""
    kept = seg[seg["kept"]]
    g = kept.groupby("subgroup")
    table = pd.DataFrame({
        "today (stored UA)": g["n_contr_now"].mean(),
        "today, on invalid UA": g["n_contr_now_on_invalid"].mean(),
        "after cleaning": g["n_contr_clean"].mean(),
    }).reindex(_subgroups(kept))
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    _grouped_bars(ax, table, CATEGORICAL[:3], "contractions per kept segment", percent=False)
    ax.set_title("Contractions found by the training-weight detector")
    path = out / "08_contractions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def figure_fhr_repair(plt: Any, seg: pd.DataFrame, out: Path) -> Optional[str]:
    """Share of FHR samples the upstream repair wrote, and how many the rules alone catch."""
    if "fhr_repaired_n" not in seg.columns or seg["fhr_repaired_n"].isna().all():
        return None
    kept = seg[seg["kept"]].dropna(subset=["fhr_repaired_n"])
    g = kept.groupby("subgroup")
    present = g.apply(lambda d: (d["present_share"] * 5280).sum(), include_groups=False)
    table = pd.DataFrame({
        "repaired (raw channel)": g["fhr_repaired_n"].sum() / present,
        "rules: recall": g["fhr_heuristic_hits"].sum() / g["fhr_repaired_n"].sum().clip(lower=1),
        "rules: precision": g["fhr_heuristic_hits"].sum() / g["fhr_heuristic_n"].sum().clip(lower=1),
    }).reindex(_subgroups(kept))
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    _grouped_bars(ax, table, CATEGORICAL[:3], "%")
    ax.set_title("FHR repair: exact share, and how much the hold/ramp/jump rules find")
    path = out / "09_fhr_repair.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def figure_gates(plt: Any, seg: pd.DataFrame, out: Path) -> str:
    """The builder's gates on today's signals and on the pre-fill signals, per subgroup."""
    order = _subgroups(seg)
    g = seg.groupby("subgroup")
    table = pd.DataFrame({
        "kept today": g["kept"].mean(),
        "kept without the fill": g["kept_prefill"].mean(),
        "flat today": g["gate"].apply(lambda s: np.mean(s == "flat")),
        "flat without the fill": g["gate_prefill"].apply(lambda s: np.mean(s == "flat")),
        "low weight": g["gate"].apply(lambda s: np.mean(s == "low_weight")),
    }).reindex(order)
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    _grouped_bars(ax, table, CATEGORICAL, "share of pre-delivery segments (%)")
    ax.set_title("The builder's segment gates, today and without the MIMO fill")
    path = out / "10_gates.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


# =================================================================================================
# Examples
# =================================================================================================
def choose_examples(seg: pd.DataFrame, n_examples: int, seed: int) -> List[Tuple[str, Dict[str, Any]]]:
    """Pick example segments per category: half the strongest, half at random, one per record.

    Args:
        seg: ``segments.csv`` with ``record_path`` joined in.
        n_examples: Segments per category.
        seed: Seed of the random half.

    Returns:
        ``[(category, row), ...]``.
    """
    rng = np.random.default_rng(seed)
    kept = seg[seg["kept"]]
    picks = []
    for category, (_, column) in EXAMPLE_CATEGORIES.items():
        if column not in kept.columns:
            continue
        pool = kept[kept[column] > 0]
        if category == "clean":
            pool = pool[pool["ua_valid_share"] >= 0.999]
        if pool.empty:
            continue
        pool = pool.sort_values(column, ascending=False).drop_duplicates("guid")
        top = pool.head((n_examples + 1) // 2)
        rest = pool.drop(top.index)
        extra = rest.iloc[rng.permutation(len(rest))[: n_examples - len(top)]] if len(rest) else rest
        for _, row in pd.concat([top, extra]).iterrows():
            picks.append((category, row.to_dict()))
    return picks


def _display(block: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-channel robust z-score of ``asinh`` values, with the reference's centre and spread."""
    a, r = np.arcsinh(block), np.arcsinh(reference)
    centre = np.median(r, axis=1, keepdims=True)
    spread = np.subtract(*np.percentile(r, [75, 25], axis=1))[:, None]
    return np.clip((a - centre) / np.where(spread > 1e-9, spread, 1.0), -3, 3)


def _bit_strip(ax: Any, bits: np.ndarray, flags: Any, t: np.ndarray, title: str) -> None:
    """One lane per bit, filled where the bit is set."""
    names = [f for f in flags]
    for lane, flag in enumerate(names):
        for start, stop in sq.runs((bits & int(flag)) != 0):
            ax.axvspan(t[start], t[min(stop, t.size - 1)], ymin=(len(names) - lane - 1) / len(names),
                       ymax=(len(names) - lane) / len(names),
                       color=CATEGORICAL[lane % len(CATEGORICAL)], linewidth=0)
    ax.set_yticks((len(names) - np.arange(len(names)) - 0.5) / len(names),
                  [f.name.lower() for f in names], fontsize=6.5)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=8.5)
    ax.grid(axis="x")


def plot_example(plt: Any, category: str, row: Mapping[str, Any], rec: Dict[str, Any],
                 r: sq.CleanResult, rho: Dict[str, np.ndarray], coeffs: Dict[str, Dict[str, np.ndarray]],
                 kit: Dict[str, Any], epsilon: float) -> Any:
    """Draw one example page: signals, bits, and the coefficients before and after."""
    i = int(row["segment"])
    n = rec["fhr_pre"].shape[1]
    t = np.arange(n) / 4.0
    steps_t = np.arange(n // DECIMATION) * DECIMATION / 4.0
    fig = plt.figure(figsize=(12, 17), layout="constrained")
    grid = fig.add_gridspec(9, 1, height_ratios=[1.3, 1.3, 0.8, 1.2, 0.6, 1.4, 1.4, 1.4, 1.4])
    title = EXAMPLE_CATEGORIES[category][0]
    fig.suptitle(f"{title} · {row['subgroup']} · {row['guid']} · segment {i} · "
                 f"domain start {row['domain_start']:.0f} s · gate {row['gate']}",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)

    ax = fig.add_subplot(grid[0])
    ax.plot(t, rec["up_now"][i], color=CATEGORICAL[1], linewidth=0.9, label="stored today (after MIMO fill)")
    ax.plot(t, rec["up_pre"][i], color=CATEGORICAL[0], linewidth=0.7, label="before the fill")
    for level in (100, 128):
        ax.axhline(level, color=AXIS, linewidth=0.8, linestyle="--")
    top = max(130.0, float(np.nanmax([rec["up_now"][i].max(), rec["up_pre"][i].max()])) + 5)
    ax.set_ylim(min(-5.0, float(rec["up_now"][i].min()) - 5), top)
    ax.set_ylabel("UA (units)")
    ax.set_title("UA, full range (dashed: 100 and 128)")
    ax.legend(loc="upper right", fontsize=7.5, ncols=2)
    ax.grid(True)

    ax = fig.add_subplot(grid[1], sharex=ax)
    ax.plot(t, r.up_clean, color=CATEGORICAL[2], linewidth=0.9, label="cleaned")
    invalid = r.up_invalid
    ax.fill_between(t, 0, 1, where=invalid, transform=ax.get_xaxis_transform(), color=GRID,
                    linewidth=0, label="invalid (bridged)")
    ax.set_ylabel("UA (units)")
    ax.set_title(f"UA after cleaning · valid {100 * row['ua_valid_share']:.1f}% · steps {row['n_steps']}")
    ax.legend(loc="upper right", fontsize=7.5, ncols=2)
    ax.grid(True)

    _bit_strip(fig.add_subplot(grid[2], sharex=ax), r.up_bits, sq.UaBit, t, "UA quality bits")

    ax = fig.add_subplot(grid[3], sharex=ax)
    ax.plot(t, rec["fhr_now"][i], color=CATEGORICAL[1], linewidth=0.8, label="stored today")
    ax.plot(t, r.fhr_clean, color=CATEGORICAL[2], linewidth=0.8, label="cleaned")
    present = rec["present"][i]
    if present.any():
        lo = float(np.percentile(r.fhr_clean[present], 0.5)) - 15
        hi = float(np.percentile(r.fhr_clean[present], 99.5)) + 15
        ax.set_ylim(max(lo, 30), min(hi, 240))
    ax.set_ylabel("FHR (bpm)")
    ax.set_title(f"FHR · valid {100 * row['fhr_valid_share']:.1f}%")
    ax.legend(loc="upper right", fontsize=7.5, ncols=2)
    ax.grid(True)

    _bit_strip(fig.add_subplot(grid[4], sharex=ax), r.fhr_bits, sq.FhrBit, t, "FHR quality bits")

    from matplotlib.colors import LinearSegmentedColormap

    def stacked(which: str, prefix: str) -> np.ndarray:
        return np.concatenate([coeffs[which][f"{prefix}_st"], coeffs[which][f"{prefix}_ph"]])

    source_now, source_new = stacked("now", "up"), stacked("clean", "up")
    target_now, target_new = stacked("now", "fhr"), stacked("clean", "fhr")
    rho_source = np.concatenate([rho["up_st"], rho["up_ph"]])
    warm = np.concatenate([kit["warmup"]["up_st"], kit["warmup"]["up_ph"]])
    # Each coefficient panel is coloured on the cleaned block's scale, so "today" and "after"
    # differ only where the data differ. The dashed line separates the _st and _ph channels.
    panels = (
        (source_now, source_new, coeffs["now"]["up_st"].shape[0],
         "UA coefficients today (up_st above the dashed line, up_ph below)"),
        (source_new, source_new, coeffs["now"]["up_st"].shape[0],
         "UA coefficients after cleaning (same colour scale)"),
        (rho_source, None, coeffs["now"]["up_st"].shape[0],
         f"UA contamination ρ (white: ρ ≤ {epsilon:g}, usable; yellow: warm-up end)"),
        (target_now if category == "fhr_repaired" else target_new, target_new,
         coeffs["now"]["fhr_st"].shape[0],
         ("FHR coefficients today" if category == "fhr_repaired" else
          "FHR coefficients after cleaning") + " (fhr_st above, fhr_ph below)"),
    )
    rho_cmap = LinearSegmentedColormap.from_list("rho", ["#ffffff", "#ffffff", "#ee8f8e", "#b52d2d"])
    for k, (block, reference, split, label) in enumerate(panels):
        ax = fig.add_subplot(grid[5 + k], sharex=fig.axes[0])
        extent = (steps_t[0], steps_t[-1] + DECIMATION / 4.0, block.shape[0], 0)
        if reference is None:
            image = ax.imshow(block, aspect="auto", cmap=rho_cmap, vmin=0, vmax=1, extent=extent,
                              interpolation="nearest")
            ax.plot(warm * DECIMATION / 4.0, np.arange(block.shape[0]) + 0.5, color=CATEGORICAL[3],
                    linewidth=1.0)
        else:
            image = ax.imshow(_display(block, reference), aspect="auto", cmap="viridis", vmin=-3,
                              vmax=3, extent=extent, interpolation="nearest")
        ax.axhline(split, color=INK if reference is None else SURFACE, linewidth=0.8, linestyle="--")
        ax.set_ylabel("channel")
        ax.set_title(label)
        fig.colorbar(image, ax=ax, shrink=0.9, pad=0.01)
    ax.set_xlabel("time in segment (s)")
    return fig


def render_examples(picks: Sequence[Tuple[str, Dict[str, Any]]], kit: Dict[str, Any],
                    out: Path, epsilon: float, read_raw_fhr: bool, device: Optional[str]) -> List[str]:
    """Recompute the chosen segments and draw one page each, plus ``examples.pdf``."""
    import torch
    from matplotlib.backends.backend_pdf import PdfPages

    cnp = _pipeline()
    plt = _style()
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    masks = kit["masks"]
    torch_bank = cnp.CausalTorchBank(masks["causal_bank"], dev, n_signal=cnp.SIGNAL_LENGTH)
    banks = {block: sq.ContaminationBank(kit["kernels"][block], cnp.SIGNAL_LENGTH, DECIMATION)
             for block in BLOCKS}
    names: List[str] = []
    cache: Dict[str, Tuple[Dict[str, Any], Optional[np.ndarray]]] = {}
    with PdfPages(out / "examples.pdf") as pdf:
        for number, (category, row) in enumerate(picks, start=1):
            path = row["record_path"]
            if path not in cache:
                rec = load_record(path)
                repaired = None
                if read_raw_fhr:
                    try:
                        repaired, _ = raw_fhr_repaired(path, rec["domain_start"], rec["fhr_pre"],
                                                       rec["present"])
                    except Exception:  # noqa: BLE001 -- the page still draws without it
                        repaired = None
                cache = {path: (rec, repaired)}
            rec, repaired = cache[path]
            i = int(row["segment"])
            r = sq.clean_segment(rec["fhr_pre"][i], rec["up_pre"][i], rec["present"][i],
                                 None if repaired is None else repaired[i])
            rho = {block: banks[block](r.up_invalid if block.startswith("up") else r.fhr_invalid)
                   for block in BLOCKS}
            pair = (np.stack([rec["fhr_now"][i], r.fhr_clean]).astype(np.float32),
                    np.stack([rec["up_now"][i], r.up_clean]).astype(np.float32))
            blocks, failures = cnp._transform_causal_record(
                torch_bank, pair[0], pair[1], kit["pairs"]["fhr_ph"], kit["pairs"]["up_ph"],
                masks["channel_plan"], 2, kit["leg_alignment"], kit["phase_operator"])
            if failures or blocks[0] is None or blocks[1] is None:
                logger.warning(f"example {category} {row['guid']}/{i}: transform failed {failures}")
                continue
            coeffs = {"now": blocks[0], "clean": blocks[1]}
            fig = plot_example(plt, category, row, rec, r, rho, coeffs, kit, epsilon)
            name = f"{number:02d}_{category}_{row['guid']}_seg{i}.png"
            fig.savefig(out / name, dpi=110)
            pdf.savefig(fig)
            plt.close(fig)
            names.append(name)
    return names


# =================================================================================================
# Report
# =================================================================================================
def _pct(value: float) -> str:
    """Format a share as a percentage."""
    return "n/a" if value is None or not np.isfinite(value) else f"{100 * value:.2f}%"


def write_report(out: Path, seg: pd.DataFrame, runs_df: pd.DataFrame, records: pd.DataFrame,
                 figures: Sequence[str], examples: Sequence[str], cfg: sq.QualityConfig,
                 epsilon: float, run_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Write ``REPORT.md`` and return the summary written to ``summary.json``."""
    kept = seg[seg["kept"]]
    order = _subgroups(kept) if len(kept) else []
    by = kept.groupby("subgroup")
    per_subgroup = pd.DataFrame({
        "records": kept.groupby("subgroup")["guid"].nunique(),
        "kept segments": by.size(),
        "UA code share": by["share_code"].mean(),
        "UA = 100 share": by["share_100"].mean(),
        "fill overwrote valid (segments)": by["fill_overwritten_valid"].apply(lambda s: np.mean(s > 0)),
        "valid FHR overwritten per segment (s)": by["fill_fhr_overwritten_s"].mean(),
        "UA valid": by["ua_valid_share"].mean(),
        "short UA transients per segment": by["n_short_transient_runs"].mean(),
        "FHR valid": by["fhr_valid_share"].mean(),
        "up_st usable": by["coef_valid_up_st"].mean(),
        "up_ph usable": by["coef_valid_up_ph"].mean(),
        "contractions today": by["n_contr_now"].mean(),
        "on invalid UA": by["n_contr_now_on_invalid"].mean(),
        "contractions after": by["n_contr_clean"].mean(),
    }).reindex(order)
    g = seg.groupby("subgroup")
    gates = pd.DataFrame({
        "pre-delivery segments": g.size(),
        "kept today": g["kept"].sum(),
        "kept without the fill": g["kept_prefill"].sum(),
        "flat today": g["gate"].apply(lambda s: int(np.sum(s == "flat"))),
        "flat without the fill": g["gate_prefill"].apply(lambda s: int(np.sum(s == "flat"))),
        "dropped only because of the fill": g.apply(
            lambda d: int(np.sum((d["gate"] == "flat") & d["kept_prefill"])), include_groups=False),
    }).reindex(_subgroups(seg))

    run_table = (runs_df[runs_df["kept"]].groupby(["value", "decided"]).agg(
        runs=("length_s", "size"), median_length_s=("length_s", "median"),
        median_entry_jump=("entry_jump", "median")).reset_index()) if len(runs_df) else pd.DataFrame()

    def fmt_table(df: pd.DataFrame, percent: Sequence[str]) -> str:
        lines = ["| " + " | ".join(["subgroup"] + list(df.columns)) + " |",
                 "|" + "---|" * (len(df.columns) + 1)]
        for idx, values in df.iterrows():
            cells = []
            for col in df.columns:
                v = values[col]
                cells.append(_pct(v) if col in percent else (f"{v:.2f}" if isinstance(v, float) else str(v)))
            lines.append("| " + " | ".join([str(idx)] + cells) + " |")
        return "\n".join(lines)

    percent_cols = ["UA code share", "UA = 100 share", "fill overwrote valid (segments)",
                    "UA valid", "FHR valid", "up_st usable", "up_ph usable"]
    errors = records[records["error"].astype(str) != ""] if "error" in records else pd.DataFrame()
    lines = [
        "# Signal quality profile",
        "",
        f"Created {time.strftime('%Y-%m-%d %H:%M')}. Quality rules version {sq.QUALITY_VERSION}. "
        f"Records profiled: {len(records)} ({len(errors)} failed). Segments profiled: {len(seg)}; "
        f"kept by today's gates: {len(kept)}. Contamination threshold $\\varepsilon$ = {epsilon:g}.",
        "",
        "The tables use the segments that today's gates keep. Every number comes from this run; the",
        "CSV files hold the per-segment and per-run values.",
        "",
    ]
    if run_config.get("eligible_only") and not run_config.get("record_paths"):
        lines += [
            "The records were drawn from the eligible records only. The prescreen uses the same flat",
            "gates, so it has already dropped records whose flats the MIMO fill made: the gate table",
            "below undercounts the effect. Run once with `eligible_only` off to see all of it.",
            "",
        ]
    lines += [
        "## What the quality step changes",
        "",
        "1. The MIMO fill (`MimoSequence.interpolate_input`) is turned off during the read. Its",
        "   effect on today's data is in figure 03 and in the `fill_*` columns of `segments.csv`.",
        "2. Every sample gets UA and FHR quality bits (`hdf5_dataset/signal_quality.py`).",
        "3. The UA level is corrected after each baseline step (`SHIFTED` bit).",
        "4. Every invalid span is bridged by a monotone cubic between its valid neighbours.",
        "5. Each stored coefficient gets its contamination $\\rho$ from the causal kernel of its channel.",
        "",
        "## Results by subgroup",
        "",
        fmt_table(per_subgroup, percent_cols) if len(per_subgroup) else "No kept segments.",
        "",
        "## Segment gates, today and without the MIMO fill",
        "",
        "The builder's gates run on today's signals and again on the pre-fill signals. A segment",
        "\"dropped only because of the fill\" fails the flat-FHR or flat-UA gate today but passes",
        "every gate without the fill: the flat run that fails it was written by the fill.",
        "",
        fmt_table(gates, []),
        "",
        "## UA code runs (kept segments)",
        "",
        run_table.to_markdown(index=False) if len(run_table) and _has_tabulate() else run_table.to_string(index=False),
        "",
        "## Figures",
        "",
    ]
    lines += [f"- `figures/{name}`" for name in figures]
    lines += ["", "## Examples", "", "One page per segment in `examples/`, all of them in "
              "`examples/examples.pdf`.", ""]
    lines += [f"- `examples/{name}`" for name in examples]
    lines += ["", "## Thresholds", "", "| name | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in asdict(cfg).items()]
    if len(errors):
        lines += ["", "## Failed records", ""]
        lines += [f"- {row.guid}: {row.error}" for row in errors.itertuples()]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "quality_version": sq.QUALITY_VERSION, "epsilon": epsilon,
        "records": int(len(records)), "failed_records": int(len(errors)),
        "segments": int(len(seg)), "kept_segments": int(len(kept)),
        "per_subgroup": json.loads(per_subgroup.to_json(orient="index")) if len(per_subgroup) else {},
        "run_config": dict(run_config),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _has_tabulate() -> bool:
    """Whether ``DataFrame.to_markdown`` can run."""
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True


# =================================================================================================
# Main
# =================================================================================================
def main(
    output_dir: str,
    screening_csv_path: Optional[str] = None,
    record_paths: Optional[Sequence[str]] = None,
    records_per_subgroup: Optional[int] = None,
    eligible_only: Optional[bool] = None,
    seed: Optional[int] = None,
    workers: Optional[int] = None,
    read_raw_fhr: Optional[bool] = None,
    n_examples: Optional[int] = None,
    epsilon: Optional[float] = None,
    transform: Optional[str] = None,
    leg_alignment: Optional[str] = None,
    phase_operator: Optional[str] = None,
    device: Optional[str] = None,
) -> int:
    """Profile a sample of records and write the report.

    Args:
        output_dir: A directory for the report. Existing files in it are overwritten.
        screening_csv_path: ``guid_screening_results.csv`` to draw the records from.
        record_paths: Explicit ``.mat`` paths, used instead of the CSV.
        records_per_subgroup: Records drawn per subgroup. Default: 40.
        eligible_only: Draw only eligible records. Default: on.
        seed: Seed of the draw and of the random examples. Default: 0.
        workers: Processes; 1 runs in this process. Default: 1.
        read_raw_fhr: Read the raw FHR channel for the exact repair mask. Default: on.
        n_examples: Example pages per category. Default: 4.
        epsilon: Contamination threshold $\\varepsilon$. Default: 0.1.
        transform: Must be ``'causal'``. Default: ``'causal'``.
        leg_alignment: Default: ``'envelope'``.
        phase_operator: Default: ``'integer_harmonic_v1'``.
        device: Torch device for the example transforms. Default: CUDA if present.

    Returns:
        The process exit code.
    """
    per_subgroup = int(records_per_subgroup or 40)
    eligible_only = True if eligible_only is None else bool(eligible_only)
    seed = 0 if seed is None else int(seed)
    workers = int(workers or 1)
    read_raw_fhr = True if read_raw_fhr is None else bool(read_raw_fhr)
    n_examples = 4 if n_examples is None else int(n_examples)
    epsilon = 0.1 if epsilon is None else float(epsilon)
    transform = transform or "causal"
    leg_alignment = leg_alignment or "envelope"
    phase_operator = phase_operator or "integer_harmonic_v1"

    out = Path(output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cfg = sq.DEFAULT_CONFIG
    run_config = {"output_dir": str(out), "screening_csv_path": screening_csv_path,
                  "record_paths": list(record_paths) if record_paths else None,
                  "records_per_subgroup": per_subgroup, "eligible_only": eligible_only,
                  "seed": seed, "workers": workers, "read_raw_fhr": read_raw_fhr,
                  "n_examples": n_examples, "epsilon": epsilon, "transform": transform,
                  "leg_alignment": leg_alignment, "phase_operator": phase_operator,
                  "quality_version": sq.QUALITY_VERSION, "quality_config": asdict(cfg)}
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

    sample = select_records(screening_csv_path, record_paths, per_subgroup, eligible_only, seed)
    logger.info(f"profiling {len(sample)} records "
                f"({sample.groupby('subgroup').size().to_dict()}) with {workers} worker(s)")
    kit = build_kit(transform, leg_alignment, phase_operator)
    cnp = _pipeline()
    init_args = (kit["kernels"], kit["warmup"], epsilon, read_raw_fhr, cnp.SIGNAL_LENGTH)
    tasks = list(zip(sample["record_path"], sample["subgroup"]))

    results: List[Dict[str, Any]] = []
    started = time.time()
    if workers <= 1:
        _init_worker(*init_args)
        for k, task in enumerate(tasks, start=1):
            results.append(_profile_task(task))
            _log_progress(k, len(tasks), results[-1], started)
    else:
        import multiprocessing as mp

        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                 initializer=_init_worker, initargs=init_args) as pool:
            futures = [pool.submit(_profile_task, task) for task in tasks]
            for k, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                _log_progress(k, len(tasks), results[-1], started)

    records = pd.DataFrame([r["record"] for r in results])
    seg = pd.DataFrame([row for r in results for row in r["segments"]])
    runs_df = pd.DataFrame([row for r in results for row in r["runs"]])
    if seg.empty:
        records.to_csv(out / "records.csv", index=False)
        raise SystemExit(f"no segment was profiled; see {out / 'records.csv'}")
    seg = seg.merge(records[["guid", "record_path"]], on="guid", how="left")
    if runs_df.empty:
        runs_df = pd.DataFrame(columns=["value", "start_s", "length_s", "entry_value", "entry_jump",
                                        "exit_value", "decided", "guid", "subgroup", "segment", "kept"])
    chan_rows = []
    for r in results:
        subgroup = r["record"]["subgroup"]
        for block, (valid, total) in r["channels"].items():
            for c in range(len(valid)):
                chan_rows.append({"subgroup": subgroup, "class": CLASS_OF.get(subgroup, subgroup),
                                  "block": block, "channel": c, "hz": float(kit["hz"][block][c]),
                                  "valid": int(valid[c]), "total": int(total[c])})
    chan = pd.DataFrame(chan_rows)
    chan_summary = (chan.groupby(["subgroup", "class", "block", "channel", "hz"], as_index=False)
                    [["valid", "total"]].sum()) if len(chan) else chan
    if len(chan_summary):
        chan_summary["valid_share"] = chan_summary["valid"] / chan_summary["total"].clip(lower=1)

    records.to_csv(out / "records.csv", index=False)
    seg.to_csv(out / "segments.csv", index=False)
    runs_df.to_csv(out / "ua_code_runs.csv", index=False)
    chan_summary.to_csv(out / "channel_validity.csv", index=False)

    plt = _style()
    figures_dir = out / "figures"
    figures = [
        figure_code_values(plt, seg, figures_dir),
        figure_code_runs(plt, runs_df, figures_dir),
        figure_fill(plt, seg, figures_dir),
        figure_bits(plt, seg, figures_dir, "ua", sq.UaBit, "UA quality bits: % of present samples "
                    "(kept segments)", "04_ua_bits.png"),
        figure_bits(plt, seg, figures_dir, "fhr", sq.FhrBit, "FHR quality bits: % of present "
                    "samples (kept segments)", "05_fhr_bits.png"),
        figure_valid_share(plt, seg, figures_dir),
        figure_channels(plt, chan_summary, figures_dir, epsilon) if len(chan_summary) else None,
        figure_contractions(plt, seg, figures_dir),
        figure_fhr_repair(plt, seg, figures_dir),
        figure_gates(plt, seg, figures_dir),
    ]
    figures = [f for f in figures if f]

    examples: List[str] = []
    if n_examples > 0:
        picks = choose_examples(seg, n_examples, seed)
        logger.info(f"drawing {len(picks)} example pages")
        examples = render_examples(picks, kit, out / "examples", epsilon, read_raw_fhr, device)
    write_report(out, seg, runs_df, records, figures, examples, cfg, epsilon, run_config)
    logger.info(f"wrote the report to {out / 'REPORT.md'}")
    return 0


def _log_progress(k: int, total: int, result: Mapping[str, Any], started: float) -> None:
    """Log one finished record."""
    record = result["record"]
    status = f"ERROR {record['error']}" if record.get("error") else (
        f"{len(result['segments'])} segments in {record.get('seconds', 0)} s")
    elapsed = time.time() - started
    logger.info(f"[{k}/{total}] {record['subgroup']} {record['guid']}: {status} "
                f"(elapsed {elapsed / 60:.1f} min)")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser. No flag has a default, so :data:`RUN_ARGS` fills the gaps.

    Returns:
        The parser.
    """
    p = argparse.ArgumentParser(description="Profile the CTG signal quality (read only).")
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--screening-csv-path", dest="screening_csv_path")
    p.add_argument("--record-paths", dest="record_paths", nargs="+")
    p.add_argument("--records-per-subgroup", dest="records_per_subgroup", type=int)
    p.add_argument("--eligible-only", dest="eligible_only", action=argparse.BooleanOptionalAction,
                   default=None)
    p.add_argument("--seed", dest="seed", type=int)
    p.add_argument("--workers", dest="workers", type=int)
    p.add_argument("--read-raw-fhr", dest="read_raw_fhr", action=argparse.BooleanOptionalAction,
                   default=None)
    p.add_argument("--n-examples", dest="n_examples", type=int)
    p.add_argument("--epsilon", dest="epsilon", type=float)
    p.add_argument("--transform", dest="transform")
    p.add_argument("--leg-alignment", dest="leg_alignment")
    p.add_argument("--phase-operator", dest="phase_operator")
    p.add_argument("--device", dest="device")
    return p


def _cli(argv: Optional[List[str]] = None) -> int:
    """Merge the command line over :data:`RUN_ARGS` and run.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If no output directory, or neither a screening CSV nor record paths, is set.
    """
    values, sources = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    refusal = missing_required(values, ("output_dir",))
    if refusal:
        raise SystemExit(refusal)
    if not values.get("screening_csv_path") and not values.get("record_paths"):
        raise SystemExit("set screening_csv_path or record_paths, in RUN_ARGS or on the command line")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("resolved arguments: " + ", ".join(
        f"{key}={values[key]!r} (from {sources[key]})" for key in sorted(values)))
    return main(**values)


#: **Edit the values below and press Run.** A flag on the command line overrides its entry.
RUN_ARGS: Dict[str, Any] = {
    # ================================ FILL IN: the paths ================================
    # str -- a directory for the report. Required. Files in it are overwritten.
    "output_dir": r"/data1/fetal-heart-tracing/HDF5_Datasets/signal_quality_profile",

    # str | None -- guid_screening_results.csv of a pipeline run; records are drawn from it.
    "screening_csv_path": r"/data1/fetal-heart-tracing/HDF5_Datasets/new_pipeline_6h_causal_int/guid_screening_results.csv",

    # list[str] | None -- explicit .mat paths; used instead of the CSV when set.
    "record_paths": None,

    # ================================ The sample ================================
    # int | None -- records drawn per subgroup.  (None -> 40)
    "records_per_subgroup": 40,
    # bool | None -- only records the pipeline found eligible.  (None -> True)
    "eligible_only": True,
    # int | None -- seed of the draw and of the random example pages.  (None -> 0)
    "seed": 0,
    # int | None -- processes; each reads and profiles whole records.  (None -> 1)
    "workers": 8,

    # ================================ The quality step ================================
    # bool | None -- read the raw FHR channel for the exact repair mask.  (None -> True)
    "read_raw_fhr": True,
    # float | None -- a coefficient is usable when its contamination is at most this.  (None -> 0.1)
    "epsilon": 0.1,
    # The transform the contamination kernels and the example coefficients are built with.
    # Same values as RUN_ARGS in create_new_pipeline.py.
    "transform": "causal",
    "leg_alignment": "envelope",
    "phase_operator": "integer_harmonic_v1",

    # ================================ Output ================================
    # int | None -- example pages per category.  (None -> 4)
    "n_examples": 4,
    # str | None -- torch device for the example transforms.  (None -> CUDA if present)
    "device": None,
}


if __name__ == "__main__":
    sys.exit(_cli())
