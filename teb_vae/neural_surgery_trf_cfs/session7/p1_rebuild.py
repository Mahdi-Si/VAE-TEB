r"""Rebuild the model's UA source from a raw UA trace and an invalid mask, with the builder's code.

The dataset builder turns one segment's pre-fill UA $x$ into the stored source in four steps, and
:class:`Rebuilder` calls the same functions in the same order:

1. ``signal_quality.bridge(x, invalid)``: a one-interval monotone cubic across every invalid span;
2. ``create_new_pipeline._sanitize_signals``: clip to $[-50, 500]$, cast to ``float32``;
3. ``causal_scattering_torch.transform_batch_numpy`` with the build's causal bank, channel plan,
   phase pairs, leg alignment and phase operator (``compute_scattering_masks`` with
   ``transform='causal'``): the blocks ``up_st`` $(36, 330)$ and ``up_ph`` $(10, 330)$;
4. ``signal_quality.ContaminationBank`` on the invalid indicator, combined by a maximum with
   :meth:`~hdf5_dataset.signal_quality.ContaminationBank.step_term` of the segment's level shifts,
   then stored as $\mathrm{round}(255\,\rho)$.

The transform is segment-local: each 5280-sample segment is transformed alone, with the edge pad
of its first sample as history. So a rebuild from the stored ``up_raw`` and the stored invalid
mask can reproduce the stored fields exactly, and :func:`validate` checks that it does.

**Level shifts.** The step term needs the builder's ``StepEvent`` list, which ``ua_quality``
returns and the file does not store. For the validation, :func:`builder_inputs` re-runs
``ua_quality`` with the inputs the builder gave it (``present`` from ``weight`` and ``fhr_raw``,
the FHR ``HOLD`` bit, the second-stage mask) and takes its events. For an injected gap,
:func:`gap_shifts` marks the gap as ``TRANSIENT`` in the stored bits and calls the builder's own
``_span_shifts``: a shift across the gap gets the step term the builder would give it.

**Future samples in a bridged value.** For an interior span $[s, e)$ the bridge reads
$y_0 = x[s-1]$, $y_1 = x[e]$, the end slope $m_0$ over $x[s-1-16 \dots s-1]$ and the end slope
$m_1$ over $x[e \dots e+16]$ (4 s at 4 Hz; set to 0 when that window touches an invalid sample).
Every bridged sample therefore depends on the raw samples up to $e + 16$, i.e. up to 4 s after the
first valid sample that follows the gap. A span at the segment start holds $x[e]$.

Edit :data:`RUN_ARGS` and run this file from the IDE to run the validation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import p1_common as C  # noqa: E402

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {"device": "cuda:0", "n_clean": 120, "n_dirty": 120, "seed": 7}


class Rebuilder:
    """The builder's UA chain (bridge, sanitise, causal transform, contamination) for one device.

    Args:
        device: Torch device string of the transform.
        n_signal: Raw samples per segment.
    """

    def __init__(self, device: str = "cuda:0", n_signal: int = C.N_RAW) -> None:
        import torch

        import hdf5_dataset.new_pipeline.create_new_pipeline as P
        from hdf5_dataset import signal_quality as sq

        self.P, self.sq = P, sq
        with h5py.File(S.SHARDS["acidosis"], "r") as handle:
            attrs = dict(handle.attrs)
        self.leg_alignment = str(attrs["causal_leg_alignment"])
        self.phase_operator = str(attrs["causal_phase_operator"])
        self.file_digest = str(attrs["quality_kernel_digest"])
        self.constants = {
            "kernel_taps": [int(attrs["causal_kernel_taps"]), int(P.CAUSAL_KERNEL_TAPS)],
            "warmup_quantile": [float(attrs["causal_warmup_quantile"]), float(P.CAUSAL_WARMUP_QUANTILE)],
            "gammatone_order": [int(attrs["gammatone_order"]), int(P.GAMMATONE_ORDER)],
        }
        self.cfg = sq.DEFAULT_CONFIG
        for key, value in sq.DEFAULT_CONFIG.as_attrs().items():
            if key in attrs and key not in ("quality_kernel_digest",):
                if not np.allclose(np.asarray(attrs[key], dtype=float), np.asarray(value, dtype=float)):
                    raise ValueError(f"quality attribute {key} differs: file {attrs[key]} vs code {value}")
        self.device = torch.device(device)
        self.n_signal = int(n_signal)
        masks = P.compute_scattering_masks(n_signal, S.RAW_PER_STEP, self.device, transform="causal",
                                           leg_alignment=self.leg_alignment,
                                           phase_operator=self.phase_operator)
        P.attach_signal_quality(masks, self.cfg)
        self.masks = masks
        self.code_digest = masks["signal_quality"]["kernel_digest"]
        self.torch_bank = P.CausalTorchBank(masks["causal_bank"], self.device, n_signal=n_signal)
        self.target_pairs = P._selection_pairs(masks["fhr_ph_selection"])
        self.source_pairs = P._selection_pairs(masks["up_ph_selection"])
        self.rho_banks = {block: sq.ContaminationBank(masks["signal_quality"]["kernels"][block],
                                                      n_signal, P.QUALITY_DECIMATION)
                          for block in P.CAUSAL_BLOCKS}

    def bridge(self, up_raw: np.ndarray, invalid: np.ndarray) -> np.ndarray:
        """Bridge and sanitise a batch of raw UA traces as the builder does.

        Args:
            up_raw: ``(B, N)`` pre-fill UA.
            invalid: ``(B, N)`` bool invalid mask.

        Returns:
            ``(B, N)`` ``float32`` bridged UA (the stored ``up``).
        """
        up_raw = np.atleast_2d(np.asarray(up_raw, dtype=np.float64))
        invalid = np.atleast_2d(np.asarray(invalid, dtype=bool))
        bridged = np.stack([self.sq.bridge(up_raw[j], invalid[j], self.cfg) for j in range(up_raw.shape[0])])
        dummy = np.zeros_like(bridged)
        _, up = self.P._sanitize_signals(dummy, bridged)
        return up

    def transform(self, up: np.ndarray, batch: int = 64) -> Dict[str, np.ndarray]:
        """Causal transform of bridged UA traces.

        Args:
            up: ``(B, N)`` bridged UA.
            batch: Segments per forward pass.

        Returns:
            ``{'up_st': (B, 36, T), 'up_ph': (B, 10, T)}`` ``float32``.
        """
        up = np.atleast_2d(np.asarray(up, dtype=np.float32))
        outs: Dict[str, List[np.ndarray]] = {"up_st": [], "up_ph": []}
        for i in range(0, up.shape[0], batch):
            part = up[i:i + batch]
            got = self.P.transform_batch_numpy(
                self.torch_bank, np.zeros_like(part), part, self.target_pairs, self.source_pairs,
                plan=self.masks["channel_plan"], leg_alignment=self.leg_alignment,
                phase_operator=self.phase_operator)
            for block in outs:
                outs[block].append(np.asarray(got[block], dtype=np.float32))
        return {block: np.concatenate(v, axis=0) for block, v in outs.items()}

    def rho(self, invalid: np.ndarray, shifts: Optional[Sequence[Sequence[int]]] = None) -> Dict[str, np.ndarray]:
        r"""Contamination $\rho$ of both UA blocks, stored-style (``uint8``, $255\rho$ rounded).

        Args:
            invalid: ``(B, N)`` invalid mask.
            shifts: Per segment, the raw positions of its level shifts (``StepEvent.at``).

        Returns:
            ``{'up_st_rho': (B, 36, T), 'up_ph_rho': (B, 10, T)}`` ``uint8``.
        """
        invalid = np.atleast_2d(np.asarray(invalid, dtype=bool))
        out: Dict[str, np.ndarray] = {}
        for block in ("up_st", "up_ph"):
            bank = self.rho_banks[block]
            rho = bank(invalid)
            if shifts is not None:
                for j, positions in enumerate(shifts):
                    if len(positions):
                        np.maximum(rho[j], bank.step_term(list(positions)), out=rho[j])
            out[f"{block}_rho"] = np.rint(rho * self.P.QUALITY_RHO_SCALE).astype(np.uint8)
        return out

    def rebuild(self, up_raw: np.ndarray, invalid: np.ndarray,
                shifts: Optional[Sequence[Sequence[int]]] = None) -> Dict[str, np.ndarray]:
        """Map raw UA plus an invalid mask to every source field the model reads.

        Args:
            up_raw: ``(B, N)`` raw (pre-fill) UA.
            invalid: ``(B, N)`` bool invalid mask.
            shifts: Per segment, raw positions of level shifts for the step term.

        Returns:
            ``{'up', 'up_st', 'up_ph', 'up_st_rho', 'up_ph_rho'}``.
        """
        up = self.bridge(up_raw, invalid)
        out = {"up": up}
        out.update(self.transform(up))
        out.update(self.rho(invalid, shifts))
        return out

    def gap_shifts(self, up_raw: np.ndarray, bits: np.ndarray, gap: np.ndarray) -> List[int]:
        """Level shifts the builder would find across an injected gap.

        The gap is marked ``TRANSIENT`` in the segment's stored bits; the builder's
        ``_span_shifts`` then judges it on the true values around it.

        Args:
            up_raw: ``(N,)`` raw UA of a clean segment.
            bits: ``(N,)`` stored ``up_quality``.
            gap: ``(N,)`` bool injected gap.

        Returns:
            Raw positions of the shifts (``StepEvent.at``).
        """
        sq = self.sq
        x = np.asarray(up_raw, dtype=np.float64)
        q = np.asarray(bits, dtype=sq.UA_DTYPE).copy()
        q[gap] |= sq.UA_DTYPE(sq.UaBit.TRANSIENT)
        present = (q & sq.UA_DTYPE(sq.UaBit.MISSING)) == 0
        code = present & ((x > self.cfg.code_threshold) | (x < 0.0))
        ceiling = present & ~code & np.isin(x, np.asarray(self.cfg.ceiling_values, dtype=np.float64))
        events = sq._span_shifts(x, present & ~code & ~ceiling, q, [], self.cfg)
        return [int(e.at) for e in events]


def builder_inputs(handle: h5py.File, row: int, cfg: Any) -> Dict[str, Any]:
    """Re-create the inputs ``ua_quality`` had at build time and re-run it.

    Args:
        handle: Open shard.
        row: Segment row.
        cfg: The quality thresholds.

    Returns:
        ``{'bits', 'steps'}``: the re-derived bits and the level-shift positions.
    """
    from hdf5_dataset import signal_quality as sq

    up_raw = handle["up_raw"][row].astype(np.float64)
    weight = handle["weight"][row]
    fhr_raw = handle["fhr_raw"][row].astype(np.float64)
    present = np.repeat(weight > 0, S.RAW_PER_STEP)[: up_raw.size] & (fhr_raw != 0)
    hold = (handle["fhr_quality"][row] & int(sq.FhrBit.HOLD)) != 0
    epoch = float(handle["epoch"][row])
    onset = epoch - float(handle["second_stage_onset"][row])
    stage2 = (epoch + np.arange(up_raw.size) / cfg.fs >= onset) if np.isfinite(onset) else None
    bits, steps = sq.ua_quality(up_raw, present, hold, cfg, stage2)
    return {"bits": bits, "steps": [int(e.at) for e in steps]}


def _err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))


def validate(device: str, n_clean: int, n_dirty: int, seed: int) -> Dict[str, Any]:
    """Rebuild stored segments from ``up_raw`` + stored invalid mask and compare every field.

    Args:
        device: Torch device.
        n_clean: Segments with no invalid UA sample.
        n_dirty: Segments with at least one invalid UA sample.
        seed: Seed of the segment draw.

    Returns:
        The validation report (also written to ``out/phase1/rebuild_validation.json``).
    """
    t0 = time.time()
    rb = Rebuilder(device)
    rng = np.random.default_rng(seed)
    picks = []
    for cohort in S.SHARDS:
        rows = C.split_rows(cohort)
        with h5py.File(S.SHARDS[cohort], "r") as handle:
            q = handle["up_quality"][:]
        dirty = ((q[rows] & S.UP_INVALID_BITS) != 0).any(axis=1)
        share = {"healthy": 0.45, "acidosis": 0.45, "hie": 0.10}[cohort]
        for flag, n in ((False, n_clean), (True, n_dirty)):
            pool = rows[dirty == flag]
            take = rng.choice(pool, size=min(len(pool), int(round(n * share))), replace=False)
            picks += [(cohort, int(r), bool(flag)) for r in sorted(take)]
    fields = ("up", "up_st", "up_ph", "up_st_rho", "up_ph_rho")
    errs: Dict[str, Dict[str, List[float]]] = {f: {"max": [], "median": []} for f in fields}
    errs_dirty: Dict[str, List[float]] = {f: [] for f in fields}
    bit_agree, n_shift_segments, rho_no_shift = [], 0, []
    for cohort in S.SHARDS:
        sel = sorted(r for c, r, _ in picks if c == cohort)
        if not sel:
            continue
        with h5py.File(S.SHARDS[cohort], "r") as handle:
            up_raw = handle["up_raw"][sel].astype(np.float64)
            bits = handle["up_quality"][sel]
            invalid = (bits.astype(np.int64) & S.UP_INVALID_BITS) != 0
            derived = [builder_inputs(handle, r, rb.cfg) for r in sel]
            stored = {f: handle[f][sel] for f in fields}
        shifts = [d["steps"] for d in derived]
        n_shift_segments += sum(1 for s in shifts if s)
        bit_agree += [float((d["bits"] == b).mean()) for d, b in zip(derived, bits)]
        got = rb.rebuild(up_raw, invalid, shifts)
        plain = rb.rho(invalid, None)
        for f in fields:
            e = _err(got[f], stored[f]).reshape(len(sel), -1)
            errs[f]["max"] += e.max(axis=1).tolist()
            errs[f]["median"] += np.median(e, axis=1).tolist()
            errs_dirty[f] += [float(v) for v, d in zip(e.max(axis=1), invalid.any(axis=1)) if d]
        for f in ("up_st_rho", "up_ph_rho"):
            rho_no_shift += _err(plain[f], stored[f]).reshape(len(sel), -1).max(axis=1).tolist()
    scale = {}
    for cohort in S.SHARDS:
        with h5py.File(S.SHARDS[cohort], "r") as handle:
            scale[cohort] = {f: float(np.abs(handle[f][:50]).max()) for f in ("up", "up_st", "up_ph")}
    report: Dict[str, Any] = {
        "n_segments": len(picks),
        "n_clean": sum(1 for p in picks if not p[2]),
        "n_dirty": sum(1 for p in picks if p[2]),
        "per_field": {f: {"max_abs_err": float(np.max(errs[f]["max"])),
                          "median_abs_err": float(np.median(errs[f]["median"])),
                          "share_segments_exact": float(np.mean(np.asarray(errs[f]["max"]) == 0.0)),
                          "max_abs_err_dirty_only": float(np.max(errs_dirty[f])) if errs_dirty[f] else None}
                      for f in fields},
        "rho_units": "uint8 counts of round(255 rho); 1 count = 0.0039",
        "value_scale_first50": scale,
        "re_derived_bits_agreement": {"mean": float(np.mean(bit_agree)), "min": float(np.min(bit_agree)),
                                      "share_exact": float(np.mean(np.asarray(bit_agree) == 1.0))},
        "segments_with_level_shifts": n_shift_segments,
        "rho_max_err_without_step_term": float(np.max(rho_no_shift)),
        "kernel_digest": {"file": rb.file_digest, "code": rb.code_digest,
                          "equal": rb.file_digest == rb.code_digest},
        "builder_constants_file_vs_code": rb.constants,
        "leg_alignment": rb.leg_alignment, "phase_operator": rb.phase_operator,
        "procedure": ("bridge(up_raw, stored invalid mask) -> _sanitize_signals -> transform_batch_numpy; "
                      "rho = ContaminationBank(stored invalid mask) max step_term(steps from re-run ua_quality)"),
        "seconds": time.time() - t0,
    }
    C.OUT1.mkdir(parents=True, exist_ok=True)
    (C.OUT1 / "rebuild_validation.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["per_field"], indent=1))
    print("digest", report["kernel_digest"], "bits agree", report["re_derived_bits_agreement"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=RUN_ARGS["device"])
    parser.add_argument("--n-clean", type=int, default=RUN_ARGS["n_clean"])
    parser.add_argument("--n-dirty", type=int, default=RUN_ARGS["n_dirty"])
    parser.add_argument("--seed", type=int, default=RUN_ARGS["seed"])
    args = parser.parse_args()
    validate(args.device, args.n_clean, args.n_dirty, args.seed)
