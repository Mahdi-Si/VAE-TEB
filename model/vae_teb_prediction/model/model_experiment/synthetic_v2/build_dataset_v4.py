r"""Sprint 2: the raw ``.npz`` cache build for ``synthetic_v4`` (no scattering).

A fork of the ``synthetic_v2`` three-stage build (:mod:`build_dataset_v2`) that renders the
concentrated cell grid straight to the raw $4\,\mathrm{Hz}$ waveform and caches it for the
raw-input model :class:`SeqVaeRawV4`. The scattering transform is deleted; weight synthesis is
added; the per-channel scattering z-score is replaced by a single **global scalar** z-score per
stream.

Build stages (mirroring v2, minus scattering):

* **Stage 1** (:func:`build_split_parts_v4`): per ``(cell, split)`` generate $n$ raw pairs via
  :func:`generate_split_raw_v4`, synthesize a decimated validity ``weight`` $(n, 330)$, and write a
  fingerprinted, resumable **physical** (un-normalised) part ``.npz``.
* **Stage 2** (:func:`_fit_norm_stats_v4`): fit one scalar $(\mu, \sigma)$ per stream over the
  **train** parts only, and persist ``norm_stats.npz``.
* **Stage 3** (:func:`assemble_split_v4` + :func:`write_cache_v4` + :func:`_write_meta_v4`):
  z-score each part with the pooled stats, stamp per-sample provenance, pool all cells, apply one
  seeded row-permutation per split, and write ``{split}.npz`` + ``meta.json``.

The model consumes **loader-normalised** input (``SeqVaeRawV4``'s ``fhr_mean``/``fhr_std`` default
to identity because the loader already normalises), and :meth:`SeqVaeRawV4.compute_loss` builds the
forecast target from the same ``fhr_raw`` it is handed -- so the cache stores the **normalised**
``fhr``/``up`` and the model both encodes and predicts in that one space. ``norm_stats.npz`` is
retained only to denormalise to bpm/mmHg for grading overlays.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.cells_v4 import (
    CellV4,
    enumerate_cells_v4,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import (
    cell_seed,
    generate_cell_raw,
    geometry,
    resolve_cache_dir,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v4 import (
    StageContextV4,
    StageSpecV4,
    register_stage_v4,
)

logger = logging.getLogger(__name__)

#: Geometry constants (from the reused model geometry so a change propagates automatically).
RAW_LEN: int = int(geometry.RAW_LEN)      # 5280
DECIMATION: int = int(geometry.D)         # 16
T_TILDE: int = int(geometry.T_TILDE)      # 330 = RAW_LEN // DECIMATION

#: Per-split shuffle offset (mirrors ``build_dataset_v2._SPLIT_SEED_OFFSET`` so the three splits
#: draw independent permutations off one ``seeds.shuffle``).
_SPLIT_SEED_OFFSET_V4: Dict[str, int] = {"train": 0, "val": 1, "test": 2}

#: The stream fields fit + z-scored (a single global scalar per stream).
_STREAM_FIELDS: Tuple[str, ...] = ("fhr", "up")


# ===========================================================================
# S2-T02: weight synthesis (+ optional gap planting)
# ===========================================================================
def synth_weight(
    n: int,
    t_tilde: Optional[int] = None,
    *,
    gap_frac: float = 0.0,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Synthesize the decimated validity ``weight`` $(n, \tilde T)$.

    Synthetic data has no acquisition gaps, so the default is all-ones. With ``gap_frac > 0`` a
    fraction of rows receive one contiguous zero-run (length in $[1, \tilde T/2]$), to exercise the
    raw-mask / NaN-safe path end to end. The length must equal ``raw_len / decimation`` ($=330$) or
    the model's :func:`frontend_mask` refuses it.

    Args:
        n: Number of rows.
        t_tilde: Decimated grid length; defaults to :data:`T_TILDE` ($330$).
        gap_frac: Fraction of rows that get a planted contiguous zero-run ($0$ = all-ones).
        rng: A :class:`numpy.random.Generator` (deterministic gap placement).

    Returns:
        A ``float32`` array $(n, \tilde T)$ in $\{0, 1\}$.
    """
    t = int(T_TILDE if t_tilde is None else t_tilde)
    n = int(n)
    weight = np.ones((n, t), dtype=np.float32)
    if gap_frac and float(gap_frac) > 0.0:
        n_gap = int(round(n * float(gap_frac)))
        if n_gap > 0:
            rows = rng.choice(n, size=min(n_gap, n), replace=False)
            max_len = max(1, t // 2)
            for r in np.atleast_1d(rows):
                glen = int(rng.integers(1, max_len + 1))
                start = int(rng.integers(0, t - glen + 1))
                weight[int(r), start:start + glen] = 0.0
    return weight


# ===========================================================================
# S2-T01: raw generation adapter
# ===========================================================================
def generate_split_raw_v4(
    cell: CellV4,
    n: int,
    split: str,
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw_v4",
    render_mode: Optional[str] = None,
    base_seed: int = 0,
) -> Dict[str, Any]:
    r"""Generate $n$ raw FHR/UP pairs for one cell (seed-deterministic, no scattering).

    A thin wrapper over :func:`generate_cell_raw` (mirroring
    :func:`build_dataset_v2.generate_pilot_samples`) that stamps the cell's solved coupling and
    seeds the generator with :func:`cell_seed` so ``(cell, split)`` units never share an RNG stream.

    Args:
        cell: The solved :class:`CellV4` to generate.
        n: Number of samples.
        split: One of ``train`` / ``val`` / ``test`` (selects the seed offset).
        config: The parsed ``config_synth_v4.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        render_mode: Optional override for ``raw.render_mode`` (else the benchmark's value).
        base_seed: Umbrella DGP seed; the per-cell/per-split seed is ``cell_seed(base_seed,
            cell_id, split)``.

    Returns:
        The :func:`generate_cell_raw` dict (``fhr_raw`` $(n, 5280)$, ``up_raw`` $(n, 5280)$,
        ``true_lag_tt`` $(n, 330)$, ``sample_delay``, ``latents``, ``meta``).
    """
    bench = config["benchmarks"][benchmark]
    mode = str(render_mode if render_mode is not None else bench["raw"].get("render_mode", "direct"))
    seed = cell_seed(int(base_seed), int(cell.cell_id), split)
    return generate_cell_raw(
        int(n),
        B=float(cell.B_y_scalar),
        D=int(cell.D),
        config=config,
        benchmark=benchmark,
        seed=seed,
        te_inj=float(cell.te_block_realised),
        render_mode=mode,
        lag_mode="fixed",
    )


# ===========================================================================
# S2-T04: atomic part I/O + fingerprint resume
# ===========================================================================
def _part_path_v4(parts_dir: Path, split: str, cell_id: int) -> Path:
    r"""Path of the Stage-1 physical part for one ``(cell, split)`` unit."""
    return parts_dir / f"{split}_cell{int(cell_id):03d}.npz"


def _savez_atomic_v4(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    r"""Write an uncompressed (mmap-able) ``.npz`` atomically (temp file + :func:`os.replace`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(str(tmp), **arrays)
    os.replace(str(tmp), str(path))


def _write_json_atomic_v4(path: Path, obj: Any) -> None:
    r"""Write ``obj`` as pretty JSON atomically (temp file + :func:`os.replace`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)
    os.replace(str(tmp), str(path))


def _part_fingerprint_v4(
    cell: CellV4,
    n: int,
    split: str,
    config: Dict[str, Any],
    *,
    benchmark: str,
    render_mode: str,
    base_seed: int,
    gap_frac: float = 0.0,
) -> str:
    r"""A resumable fingerprint over every generation-affecting input for one part.

    A change to ``n``, the cell coupling, the seed, the render mode, the planted-gap fraction
    ``gap_frac``, or any raw-render knob remaps the fingerprint and forces regeneration; an unchanged
    request reuses the existing part.
    """
    raw = config["benchmarks"][benchmark]["raw"]
    seed = cell_seed(int(base_seed), int(cell.cell_id), split)
    parts: List[str] = [
        "v4",
        f"n={int(n)}",
        f"cid={int(cell.cell_id)}",
        f"D={int(cell.D)}",
        f"B={float(cell.B_y_scalar):.10g}",
        f"te={float(cell.te_block_realised):.10g}",
        f"seed={int(seed)}",
        f"render={render_mode}",
        f"gap_frac={float(gap_frac):.10g}",
        f"fs={raw.get('fs')}",
        f"n_raw={raw.get('n_raw')}",
        f"f_pulse={raw.get('f_pulse')}",
        f"fhrv_notch={bool(raw.get('fhrv_notch_enabled'))}",
    ]
    if render_mode == "direct":
        parts.append(f"one_sided={bool((raw.get('direct') or {}).get('one_sided', False))}")
    else:
        parts.append(f"am_offset={raw.get('am_offset_ratio')}")
    fhrv = raw.get("fhrv_band_power", {}) or {}
    parts.append("fhrv_power=" + ",".join(f"{k}={fhrv[k]}" for k in sorted(fhrv)))
    return "|".join(parts)


def _read_part_fingerprint_v4(path: Path) -> Optional[str]:
    r"""Read a part's stored ``_fingerprint`` string, or ``None`` if absent/unreadable."""
    try:
        with np.load(path) as existing:
            return str(existing["_fingerprint"].item())
    except Exception:  # noqa: BLE001 -- a corrupt/legacy part is simply regenerated
        return None


def build_split_parts_v4(
    cells: Sequence[CellV4],
    split: str,
    n: int,
    *,
    config: Dict[str, Any],
    benchmark: str = "G1_raw_v4",
    render_mode: str,
    base_seed: int,
    parts_dir: Path,
    resume: bool = True,
    gap_frac: float = 0.0,
) -> List[Path]:
    r"""Stage 1: generate + weight-synthesize each ``(cell, split)`` to a physical part (resumable).

    For each cell, generate $n$ raw pairs (:func:`generate_split_raw_v4`), synthesize the decimated
    ``weight`` (:func:`synth_weight`, seeded off the same ``cell_seed`` so gaps are deterministic),
    and atomically write a part ``.npz`` holding **physical** ``fhr``/``up``, the ``weight``, the
    ground-truth ``true_lag_tt``, and the ``_fingerprint``. A part whose fingerprint matches is
    skipped (crash-safe resume).

    Args:
        cells: The cells to build for this split.
        split: One of ``train`` / ``val`` / ``test``.
        n: Samples per cell.
        config: The parsed ``config_synth_v4.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        render_mode: The raw render mode threaded from :func:`build_all_v4`.
        base_seed: Umbrella DGP seed.
        parts_dir: Directory for the per-unit part files.
        resume: Skip a unit whose part fingerprint matches (default ``True``).
        gap_frac: Optional fraction of rows with a planted validity gap (default ``0.0``).

    Returns:
        The list of part paths (one per cell), in cell order.
    """
    paths: List[Path] = []
    for cell in cells:
        pp = _part_path_v4(parts_dir, split, cell.cell_id)
        paths.append(pp)
        fingerprint = _part_fingerprint_v4(
            cell, int(n), split, config, benchmark=benchmark,
            render_mode=render_mode, base_seed=base_seed, gap_frac=gap_frac,
        )
        if resume and pp.is_file() and _read_part_fingerprint_v4(pp) == fingerprint:
            logger.info("build_split_parts_v4: skip existing %s", pp.name)
            continue
        if resume and pp.is_file():
            logger.info("build_split_parts_v4: regenerating %s (fingerprint changed)", pp.name)
        out = generate_split_raw_v4(
            cell, int(n), split, config, benchmark=benchmark,
            render_mode=render_mode, base_seed=base_seed,
        )
        w_rng = np.random.default_rng(cell_seed(int(base_seed), int(cell.cell_id), split))
        weight = synth_weight(int(n), rng=w_rng, gap_frac=gap_frac)
        part = {
            "fhr": np.asarray(out["fhr_raw"], dtype=np.float32),
            "up": np.asarray(out["up_raw"], dtype=np.float32),
            "weight": np.asarray(weight, dtype=np.float32),
            "true_lag_tt": np.asarray(out["true_lag_tt"], dtype=np.int16),
            "_fingerprint": np.array(fingerprint),
        }
        _savez_atomic_v4(pp, part)
        logger.info("build_split_parts_v4: wrote %s (n=%d)", pp.name, part["fhr"].shape[0])
    return paths


# ===========================================================================
# S2-T03b: pooled scalar normalisation stats
# ===========================================================================
def _fit_norm_stats_v4(
    cells: Sequence[CellV4],
    split: str,
    *,
    parts_dir: Path,
) -> Dict[str, Dict[str, float]]:
    r"""Stage 2: one global scalar $(\mu, \sigma)$ per stream pooled over one split's parts.

    Accumulates $\sum x$ / $\sum x^2$ / count over every ``fhr`` / ``up`` value of the split's parts
    (streamed, so the whole pool need not be held in RAM), then reduces to a scalar mean/std per
    stream. A degenerate (constant) stream floors ``std`` to $1$ so the z-score never divides by $0$.

    Args:
        cells: The cells whose parts contribute (normally the ``train`` cells).
        split: The split whose parts to pool (normally ``train``).
        parts_dir: Directory holding the Stage-1 parts.

    Returns:
        ``{field: {'mean': float, 'std': float}}`` for each stream.
    """
    acc: Dict[str, Dict[str, float]] = {
        f: {"sum": 0.0, "sumsq": 0.0, "count": 0.0} for f in _STREAM_FIELDS
    }
    for cell in cells:
        pp = _part_path_v4(parts_dir, split, cell.cell_id)
        with np.load(pp) as part:
            for field in _STREAM_FIELDS:
                arr = np.asarray(part[field], dtype=np.float64)
                acc[field]["sum"] += float(arr.sum())
                acc[field]["sumsq"] += float((arr * arr).sum())
                acc[field]["count"] += float(arr.size)

    stats: Dict[str, Dict[str, float]] = {}
    for field, a in acc.items():
        count = max(a["count"], 1.0)
        mean = a["sum"] / count
        var = max(a["sumsq"] / count - mean * mean, 0.0)
        std = float(np.sqrt(var))
        stats[field] = {"mean": float(mean), "std": (std if std > 0.0 else 1.0)}
    return stats


def _save_norm_stats_v4(path: Path, stats: Dict[str, Dict[str, float]]) -> None:
    r"""Persist scalar norm stats to ``norm_stats.npz`` (flat ``<field>_mean`` / ``<field>_std``)."""
    flat: Dict[str, np.ndarray] = {}
    for field, sd in stats.items():
        flat[f"{field}_mean"] = np.asarray(sd["mean"], dtype=np.float32)
        flat[f"{field}_std"] = np.asarray(sd["std"], dtype=np.float32)
    _savez_atomic_v4(path, flat)


# ===========================================================================
# S2-T03a: assemble (normalise + stamp + pool + shuffle) one split
# ===========================================================================
def assemble_split_v4(
    cells: Sequence[CellV4],
    split: str,
    *,
    parts_dir: Path,
    stats: Dict[str, Dict[str, float]],
    config: Dict[str, Any],
    benchmark: str = "G1_raw_v4",
) -> Dict[str, np.ndarray]:
    r"""Stage 3: normalise each part, stamp provenance, pool all cells, and shuffle one split.

    For each cell: load its physical part, z-score ``fhr``/``up`` with the pooled scalar ``stats``,
    stamp the per-sample provenance (``sample_te_true`` $=\mathrm{TE}_{\mathrm{inj}}$,
    ``sample_delay`` $= D$, ``sample_cell_id``, ``sample_held_out`` $= 0$, ``sample_raw_index``),
    and a label-only zeros ``target`` $(n, 330)$. Concatenate every cell and apply one seeded
    row-aligned permutation (``seeds.shuffle`` + split offset) so a ``shuffle=False`` consumer still
    sees mixed cells and features stay aligned to provenance.

    Args:
        cells: The cells to assemble.
        split: One of ``train`` / ``val`` / ``test``.
        parts_dir: Directory holding the Stage-1 parts.
        stats: The pooled scalar norm stats from :func:`_fit_norm_stats_v4`.
        config: The parsed ``config_synth_v4.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.

    Returns:
        The pooled, shuffled cache arrays: ``fhr``/``up`` $(N, 5280)$, ``weight``/``target``
        $(N, 330)$, ``true_lag_tt`` $(N, 330)$, and the five ``sample_*`` provenance arrays $(N,)$.
    """
    fhr_mean = float(stats["fhr"]["mean"])
    fhr_std = float(stats["fhr"]["std"])
    up_mean = float(stats["up"]["mean"])
    up_std = float(stats["up"]["std"])

    fhr_l, up_l, w_l, lag_l = [], [], [], []
    te_l, delay_l, cell_l, held_l, rawidx_l = [], [], [], [], []

    for cell in cells:
        pp = _part_path_v4(parts_dir, split, cell.cell_id)
        with np.load(pp) as part:
            fhr = (np.asarray(part["fhr"], dtype=np.float32) - fhr_mean) / fhr_std
            up = (np.asarray(part["up"], dtype=np.float32) - up_mean) / up_std
            weight = np.asarray(part["weight"], dtype=np.float32)
            lag = np.asarray(part["true_lag_tt"], dtype=np.int16)
        n_c = int(fhr.shape[0])
        fhr_l.append(np.ascontiguousarray(fhr))
        up_l.append(np.ascontiguousarray(up))
        w_l.append(weight)
        lag_l.append(lag)
        te_l.append(np.full(n_c, cell.te_block_realised, dtype=np.float32))
        delay_l.append(np.full(n_c, cell.D, dtype=np.int16))
        cell_l.append(np.full(n_c, cell.cell_id, dtype=np.int16))
        held_l.append(np.zeros(n_c, dtype=np.int8))
        rawidx_l.append(np.arange(n_c, dtype=np.int32))

    arrays: Dict[str, np.ndarray] = {
        "fhr": np.concatenate(fhr_l, axis=0),
        "up": np.concatenate(up_l, axis=0),
        "weight": np.concatenate(w_l, axis=0),
        "true_lag_tt": np.concatenate(lag_l, axis=0),
        "sample_te_true": np.concatenate(te_l, axis=0),
        "sample_delay": np.concatenate(delay_l, axis=0),
        "sample_cell_id": np.concatenate(cell_l, axis=0),
        "sample_held_out": np.concatenate(held_l, axis=0),
        "sample_raw_index": np.concatenate(rawidx_l, axis=0),
    }
    n_total = int(arrays["fhr"].shape[0])
    t_tilde = int(arrays["weight"].shape[1])
    arrays["target"] = np.zeros((n_total, t_tilde), dtype=np.float32)

    shuffle_seed = int(config.get("seeds", {}).get("shuffle", 0))
    rng = np.random.default_rng(shuffle_seed + _SPLIT_SEED_OFFSET_V4[split])
    perm = rng.permutation(n_total)
    for key in list(arrays.keys()):
        arrays[key] = np.ascontiguousarray(arrays[key][perm])
    return arrays


def write_cache_v4(out_dir: Path, splits: Dict[str, Dict[str, np.ndarray]]) -> Path:
    r"""Write each split's assembled arrays to ``{split}.npz`` atomically.

    Args:
        out_dir: Destination cache directory.
        splits: ``{split: arrays}`` from :func:`assemble_split_v4`.

    Returns:
        ``out_dir``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, arrays in splits.items():
        _savez_atomic_v4(out_dir / f"{split}.npz", arrays)
        logger.info("write_cache_v4: [%s] n=%d -> %s.npz", split, int(arrays["fhr"].shape[0]), split)
    return out_dir


def _write_meta_v4(
    out_dir: Path,
    *,
    cells: Sequence[CellV4],
    dropped: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    benchmark: str,
    render_mode: str,
    n_per_split: Dict[str, int],
) -> None:
    r"""Write the shared ``meta.json`` (geometry, render mode, pooled TE, per-cell manifest)."""
    bench = config["benchmarks"][benchmark]
    data = bench["data"]
    experiment = config.get("experiment", {})
    horizon = int(data["horizon"])
    te_pooled = float(np.mean([c.te_block_realised for c in cells])) if cells else 0.0

    manifest: List[Dict[str, Any]] = [
        {
            "cell_id": int(c.cell_id),
            "target_te": float(c.target_te),
            "D": int(c.D),
            "B_y_scalar": float(c.B_y_scalar),
            "te_block_realised": float(c.te_block_realised),
            "is_null": bool(c.is_null),
            "level_index": int(c.level_index),
            "replicate": int(c.replicate),
        }
        for c in cells
    ]

    meta: Dict[str, Any] = {
        "benchmark": benchmark,
        "tag": str(experiment.get("tag", benchmark)),
        "data_tag": str(experiment.get("data_tag", experiment.get("tag", benchmark))),
        "render_mode": render_mode,
        "te_true": te_pooled,
        "te_per_step": (te_pooled / horizon if horizon else 0.0),
        "sequence_length": int(data.get("sequence_length", T_TILDE)),
        "raw_len": int(bench["raw"]["n_raw"]),
        "decimation": int(DECIMATION),
        "crop": int(geometry.CROP),
        "horizon": horizon,
        "K_history": int(data.get("K_history", 0)),
        "n_per_cell": {s: int(v) for s, v in n_per_split.items()},
        "grid": {
            "target_te_grid": [float(t) for t in bench["mix"]["target_te_grid"]],
            "lag_grid": [int(d) for d in bench["mix"]["lag_grid"]],
            "lag_mode": str(bench["mix"].get("lag_mode", "fixed")),
        },
        "seeds": dict(config.get("seeds", {})),
        "cells": manifest,
        "dropped": [dict(d) for d in dropped],
    }
    _write_json_atomic_v4(out_dir / "meta.json", meta)
    logger.info(
        "write_meta_v4: te_true(pooled)=%.4f nats  cells=%d  render_mode=%s",
        te_pooled, len(cells), render_mode,
    )


# ===========================================================================
# S2-T05: full build driver + stage registration
# ===========================================================================
def build_all_v4(
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw_v4",
    pilot: bool = False,
    resume: bool = True,
    out_dir: Optional[Path] = None,
    grid_override: Optional[Dict[str, Any]] = None,
    n_override: Optional[Dict[str, int]] = None,
) -> Path:
    r"""Build the full v4 raw cache: enumerate $\to$ Stage 1 $\to$ Stage 2 $\to$ Stage 3 $\to$ write.

    Mirrors :func:`build_dataset_v2.build_all` minus the scattering transform. ``pilot`` selects the
    small ``eval.realizability.pilot`` grid; ``grid_override`` / ``n_override`` force a specific
    (tiny) grid / sample count for tests. The build is deterministic and resumable (Stage-1 parts
    under ``out_dir/_parts/``). The ``render_mode`` is read from the (arm-resolved) benchmark raw
    block, so an ``am_carrier`` arm builds the am cache under its own ``data_tag``.

    Args:
        config: The parsed ``config_synth_v4.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        pilot: Use the pilot grid + a single pilot ``n_per_cell`` for every split.
        resume: Reuse existing Stage-1 parts (default ``True``).
        out_dir: Cache directory override (defaults to :func:`resolve_cache_dir`).
        grid_override: ``{'target_te_grid', 'lag_grid', 'cells_per_level'}`` to force a grid.
        n_override: ``{'train', 'val', 'test'}`` sample counts to force.

    Returns:
        The cache directory holding ``train/val/test.npz``, ``meta.json`` and ``norm_stats.npz``.
    """
    bench = config["benchmarks"][benchmark]
    mix = bench["mix"]
    render_mode = str(bench["raw"].get("render_mode", "direct"))
    seeds = config.get("seeds", {})
    base_seed = int(seeds.get("dgp", seeds.get("base_seed", 0)))

    if grid_override is not None:
        cells, dropped = enumerate_cells_v4(
            config, benchmark=benchmark,
            target_te_grid=grid_override.get("target_te_grid"),
            lag_grid=grid_override.get("lag_grid"),
            cells_per_level=grid_override.get("cells_per_level"),
        )
    elif pilot:
        pil = bench["eval"]["realizability"]["pilot"]
        cells, dropped = enumerate_cells_v4(
            config, benchmark=benchmark,
            target_te_grid=pil.get("target_te_grid"), lag_grid=pil.get("lag_grid"),
        )
    else:
        cells, dropped = enumerate_cells_v4(config, benchmark=benchmark)

    if n_override is not None:
        n_per_split = {s: int(n_override[s]) for s in ("train", "val", "test")}
    elif pilot:
        pn = int(bench["eval"]["realizability"]["pilot"].get("n_per_cell", 384))
        n_per_split = {"train": pn, "val": max(1, pn // 2), "test": max(1, pn // 2)}
    else:
        n_per_split = {
            "train": int(mix["n_per_cell_train"]),
            "val": int(mix["n_per_cell_val"]),
            "test": int(mix["n_per_cell_test"]),
        }

    if not cells:
        raise ValueError(f"build_all_v4: no cells to build (all dropped: {dropped}); check the grid.")

    out_dir = resolve_cache_dir(config, benchmark=benchmark) if out_dir is None else Path(out_dir)
    parts_dir = out_dir / "_parts"
    logger.info(
        "build_all_v4: %d cells, render_mode=%s, n_per_split=%s -> %s",
        len(cells), render_mode, n_per_split, out_dir,
    )

    # Stage 1: generate + weight-synthesize every (cell, split) to physical parts (resumable).
    for split in ("train", "val", "test"):
        build_split_parts_v4(
            cells, split, n_per_split[split], config=config, benchmark=benchmark,
            render_mode=render_mode, base_seed=base_seed, parts_dir=parts_dir, resume=resume,
        )

    # Stage 2: fit the pooled scalar normaliser ONCE on the train parts.
    stats = _fit_norm_stats_v4(cells, "train", parts_dir=parts_dir)
    _save_norm_stats_v4(out_dir / "norm_stats.npz", stats)

    # Stage 3: normalise + stamp + pool + shuffle each split, then write the cache + meta.
    splits: Dict[str, Dict[str, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        splits[split] = assemble_split_v4(
            cells, split, parts_dir=parts_dir, stats=stats, config=config, benchmark=benchmark,
        )
    write_cache_v4(out_dir, splits)
    _write_meta_v4(
        out_dir, cells=cells, dropped=dropped, config=config, benchmark=benchmark,
        render_mode=render_mode, n_per_split=n_per_split,
    )
    return out_dir


def run_build_dataset_v4(ctx: StageContextV4) -> int:
    r"""``build`` stage: build the raw ``.npz`` cache for the active (arm-resolved) config.

    Returns:
        ``0`` on success.
    """
    out_dir = build_all_v4(ctx.config, benchmark=ctx.benchmark, pilot=ctx.pilot)
    print(f"[build] wrote cache -> {out_dir}")
    for split in ("train", "val", "test"):
        p = out_dir / f"{split}.npz"
        if p.is_file():
            with np.load(p) as npz:
                print(f"[build] {split}.npz  n={npz['fhr'].shape[0]}  "
                      f"fhr={tuple(npz['fhr'].shape)}  weight={tuple(npz['weight'].shape)}")
    return 0


# ===========================================================================
# S3-T04: untrimmed on-demand raw regenerator (deferred-CMI / overlay source)
# ===========================================================================
def make_raw_provider_v4(
    config: Dict[str, Any],
    split: str,
    *,
    benchmark: str = "G1_raw_v4",
    cache_dir: Optional[Path] = None,
) -> Any:
    r"""Build a memoised **untrimmed** raw regenerator keyed by ``(cell_id, raw_index)``.

    The untrimmed analog of :func:`build_dataset_v2.make_raw_provider`: it regenerates the exact
    **physical** raw $4\,\mathrm{Hz}$ pair for any cached row from ``sample_raw_index`` +
    :func:`cell_seed` determinism, serving the full $5280$ samples (no $4800$-sample trim). Because
    a row's amplitude depends on the ``n`` passed to :func:`generate_cell_raw`, the per-cell ``n`` is
    taken from the **cache's own** ``sample_cell_id`` counts (cache-authoritative), and the seed /
    render mode from ``meta.json`` -- so regeneration reproduces the cached (pre-normalisation) row.

    Args:
        config: The parsed ``config_synth_v4.yaml`` tree.
        split: The cache split the rows come from (``train`` / ``val`` / ``test``).
        benchmark: Active benchmark key under ``benchmarks``.
        cache_dir: Override for the cache directory (defaults to :func:`resolve_cache_dir`).

    Returns:
        A total callable ``provider(cell_id, raw_index) -> (fhr_5280, up_5280)`` of ``float32``
        physical arrays; on any failure it returns a NaN-filled pair of length ``raw_len`` rather
        than raising. The function carries a ``window_length`` attribute ($= 5280$).

    Raises:
        FileNotFoundError \\ ValueError \\ KeyError: At construction, when ``meta.json`` or the
            split ``.npz`` (with ``sample_cell_id``) is absent/unusable.
    """
    cdir = resolve_cache_dir(config, benchmark=benchmark) if cache_dir is None else Path(cache_dir)

    meta_path = cdir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"make_raw_provider_v4: no meta.json under {cdir}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    cells_by_id: Dict[int, CellV4] = {}
    for c in meta.get("cells", []) or []:
        cid = int(c["cell_id"])
        b = float(c.get("B_y_scalar", 0.0))
        cells_by_id[cid] = CellV4(
            cell_id=cid, target_te=float(c.get("target_te", 0.0)), D=int(c["D"]),
            B_y_scalar=b, te_block_realised=float(c.get("te_block_realised", 0.0)),
            is_null=bool(c.get("is_null", b == 0.0)),
            level_index=int(c.get("level_index", 0)), replicate=int(c.get("replicate", 0)),
        )
    if not cells_by_id:
        raise ValueError(f"make_raw_provider_v4: meta.json under {cdir} has no cells")

    seeds_meta = meta.get("seeds", {}) or {}
    base_seed = int(seeds_meta.get("dgp", seeds_meta.get("base_seed", 0)))
    render_mode = str(meta.get("render_mode", "direct"))
    raw_len = int(meta.get("raw_len", config["benchmarks"][benchmark]["raw"]["n_raw"]))

    split_npz = cdir / f"{split}.npz"
    if not split_npz.is_file():
        raise FileNotFoundError(f"make_raw_provider_v4: no {split}.npz under {cdir}")
    with np.load(split_npz) as npz:
        if "sample_cell_id" not in npz.files:
            raise KeyError(f"make_raw_provider_v4: {split}.npz lacks sample_cell_id")
        cell_ids = np.asarray(npz["sample_cell_id"]).astype(np.int64)
    n_by_cell = {int(cid): int(np.count_nonzero(cell_ids == cid)) for cid in np.unique(cell_ids)}

    cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    nan_win = np.full(raw_len, np.nan, dtype=np.float32)

    def provider(cell_id: int, raw_index: int) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return the untrimmed (fhr, up) physical pair for one row; a NaN pair on any failure."""
        cid, ri = int(cell_id), int(raw_index)
        try:
            if cid not in cells_by_id or cid not in n_by_cell:
                raise KeyError(f"unknown cell_id {cid}")
            if cid not in cache:
                out = generate_split_raw_v4(
                    cells_by_id[cid], int(n_by_cell[cid]), split, config,
                    benchmark=benchmark, render_mode=render_mode, base_seed=base_seed,
                )
                cache[cid] = (
                    np.ascontiguousarray(out["fhr_raw"], dtype=np.float32),
                    np.ascontiguousarray(out["up_raw"], dtype=np.float32),
                )
            fhr_all, up_all = cache[cid]
            if not 0 <= ri < fhr_all.shape[0]:
                raise IndexError(f"raw_index {ri} out of range for cell {cid} (n={fhr_all.shape[0]})")
            return fhr_all[ri].copy(), up_all[ri].copy()
        except Exception as exc:  # noqa: BLE001 -- raw is a diagnostic nicety; degrade to blank
            logger.warning("make_raw_provider_v4: raw unavailable for (cell=%d, row=%d): %s",
                           cid, ri, exc)
            return nan_win.copy(), nan_win.copy()

    provider.window_length = raw_len  # type: ignore[attr-defined]
    return provider


register_stage_v4(StageSpecV4(
    name="build",
    run=run_build_dataset_v4,
    order=30,
    model_dependent=False,
    fatal=True,
    help="build the raw .npz cache (train/val/test) with weight synthesis + resume",
))
