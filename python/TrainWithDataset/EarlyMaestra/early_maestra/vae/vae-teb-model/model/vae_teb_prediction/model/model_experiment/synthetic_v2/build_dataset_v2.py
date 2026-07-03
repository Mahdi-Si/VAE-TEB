r"""Cell enumeration, coupling solve, and pilot generation for ``synthetic_v2`` (Sprint 3/4).

This module owns the *cell grid*: the product of the requested injected block
transfer entropies (``mix.target_te_grid``, nats) and source→target lags
(``mix.lag_grid``, decimated steps, fixed per cell). Each non-null cell's coupling
$B$ is solved with the ported inverter
:func:`analytic_te.B_y_for_mean_te_block_state_space`, so the achieved block TE is
the exact injected-TE label $\mathrm{TE}_{\mathrm{inj}}$ (§9). The ``target_te = 0``
cell is a **null** ($B = 0$, $\mathrm{TE}_{\mathrm{inj}} = 0$) — the calibration
intercept anchor and the null control (§9.3).

Sprint 3 implements enumeration + the coupling solve + in-memory pilot generation
(:func:`generate_pilot_samples`), reusing :func:`raw_generators.generate_cell_raw`
so the Sprint 4 cache build shares the same generation path. The single source and
single target mean there is **no** ``M`` (informative-channel) axis and — for the
initial build — a **fixed** per-cell lag only (``lag_mode: fixed``); ``band`` mode is
deferred (the inverter retains the band-averaging math for later).

See ``SYNTHETIC_V2_RAW_TE_PIPELINE_EXPLAINED.md`` §6, §9, §17 and
``SYNTHETIC_V2_SPEC_AND_SPRINTS.md`` Sprints 3–4.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .analytic_te import (
    B_y_for_mean_te_block_state_space,
    snr_per_step_for_te_block,
)
from .raw_generators import generate_cell_raw

logger = logging.getLogger(__name__)

#: This module's directory; used to resolve the relative ``paths.data_dir`` cache root.
_MODULE_DIR = Path(__file__).resolve().parent

#: The four model-facing feature fields, in cache order.
_FIELD_NAMES: Tuple[str, ...] = ("fhr_st", "fhr_ph", "up_st", "up_ph")

# Deterministic per-(cell, split) seed offsets (ported from
# ``synthetic/mixed_dataset.build_mixed_split``): the umbrella DGP seed is
# combined with a split-specific stride and the cell id so train / val / test
# and distinct cells never share a stream, while a cache stays reproducible from
# its seeds alone.
_SPLIT_SEED_OFFSET: Dict[str, int] = {"train": 0, "val": 1, "test": 2}
_SPLIT_SEED_STRIDE: int = 100_003
_CELL_SEED_STRIDE: int = 101


@dataclass(frozen=True)
class CellV2:
    r"""One $(\texttt{target\_te}, D)$ cell of the single-pathway v2 grid.

    Unlike v1's ``MixCell`` there is no ``M`` (informative-channel count) and no
    lag ``band`` — v2 has exactly one source, one target, and a fixed per-cell lag.

    Attributes:
        cell_id: Stable index of the cell within the pool (assigned in enumeration
            order; null cells included).
        target_te: The *requested* injected block TE in nats ($\ge 0$; ``0`` is the
            null anchor).
        D: The fixed source→target lag $D \ge 1$ in decimated steps.
        B_y_scalar: The solved coupling $B$ (the inverter's ``B_y_scalar``; ``0`` for
            a null cell).
        te_block_realised: The *achieved* block TE at ``B_y_scalar`` — the exact
            $\mathrm{TE}_{\mathrm{inj}}$ label the cell is graded against. Equals the
            requested ``target_te`` within the inverter tolerance; ``0`` for a null.
    """

    cell_id: int
    target_te: float
    D: int
    B_y_scalar: float
    te_block_realised: float


def solve_cell_coupling(
    config: Dict[str, Any],
    target_te: float,
    delay: int,
    *,
    benchmark: str = "G1_raw",
) -> Dict[str, Any]:
    r"""Solve the coupling $B$ for a cell authored by ``(target_te, D)`` (§9.1).

    Reads the single-pathway latent spec (``benchmarks.<benchmark>.data``), the
    inverter knobs (``mix.inverter``), and the Monte-Carlo seed
    (``seeds.inverter_mc``) from ``config`` and calls the ported inverter with a
    fixed lag ($d_{\min} = d_{\max} = D$). The per-step SNR uses
    $M = \mathrm{len(oscillators)}$ ($= 1$ in v2), matching the $M$ the inverter's
    ``te_block`` already reflects.

    This is the single owner of the inverter call: :func:`run_pipeline_v2.solve_te`
    and :func:`enumerate_cells_v2` both delegate here so the CLI demo and the build
    solve identically.

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        target_te: Target injected block TE in nats ($\ge 0$; ``0`` is a null cell).
        delay: Fixed source→target lag $D \ge 1$ in decimated steps.
        benchmark: Active benchmark key under ``benchmarks``.

    Returns:
        The inverter's result dict augmented with ``snr_per_step``: keys ``B_y``,
        ``B_y_scalar``, ``te_block``, ``te_per_step``, ``n_iter``, ``snr_per_step``.

    Raises:
        ValueError: If the inverter's bracket $[\texttt{lo}, \texttt{hi}]$ does not
            contain ``target_te`` (an *unsolvable* cell; the caller decides whether
            to drop it).
    """
    bench = config["benchmarks"][benchmark]
    data = bench["data"]
    inverter = bench["mix"]["inverter"]
    seed = int(config.get("seeds", {}).get("inverter_mc", 0))
    oscillators = [tuple(spec) for spec in data["oscillators"]]
    horizon = int(data["horizon"])

    solution = B_y_for_mean_te_block_state_space(
        target_te_block=float(target_te),
        delay_min=int(delay),
        delay_max=int(delay),
        oscillators=oscillators,
        target_ar=float(data["target_ar"]),
        sigma2_y=float(data["sigma2_y"]),
        sigma2_eta=float(data["sigma2_eta"]),
        H=horizon,
        K_history=int(data["K_history"]),
        n_samples=int(inverter["n_samples"]),
        lo=float(inverter["lo"]),
        hi=float(inverter["hi"]),
        tol=float(inverter["tol"]),
        max_iter=int(inverter["max_iter"]),
        seed=seed,
    )
    solution["snr_per_step"] = snr_per_step_for_te_block(
        solution["te_block"], horizon, len(oscillators)
    )
    return solution


def enumerate_cells_v2(
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw",
    target_te_grid: Optional[Sequence[float]] = None,
    lag_grid: Optional[Sequence[int]] = None,
) -> Tuple[List[CellV2], List[Dict[str, Any]]]:
    r"""Enumerate and solve the cell grid (§9.3).

    Crosses ``target_te_grid`` with ``lag_grid`` (fixed lag). For each pair:

    * ``target_te == 0`` → a **null** cell ($B = 0$, ``te_block_realised = 0``),
      kept without invoking the inverter.
    * ``target_te > 0`` → solve the coupling via :func:`solve_cell_coupling`. The
      inverter Monte-Carlo is memoised on ``(round(target_te, 9), D)`` so identical
      authored cells solve once. A cell whose bracket misses the target (the
      inverter raises :class:`ValueError`) is **logged and dropped** — collected in
      the returned ``dropped`` list rather than aborting enumeration.

    ``cell_id`` is assigned in enumeration order over the *kept* cells (dropped cells
    do not consume an id), so the pool's ids are contiguous.

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        target_te_grid: Override for ``mix.target_te_grid`` (e.g. the pilot grid); the
            config value is used when ``None``.
        lag_grid: Override for ``mix.lag_grid``; the config value is used when ``None``.

    Returns:
        ``(cells, dropped)`` — the kept :class:`CellV2` list (contiguous ``cell_id``)
        and a list of ``{'target_te', 'D', 'reason'}`` dicts for unsolvable cells.

    Raises:
        ValueError: If ``mix.lag_mode`` is not ``fixed`` (``band`` mode is deferred).
    """
    bench = config["benchmarks"][benchmark]
    mix = bench["mix"]
    lag_mode = str(mix.get("lag_mode", "fixed"))
    if lag_mode != "fixed":
        raise ValueError(
            f"enumerate_cells_v2: only lag_mode 'fixed' is implemented in the "
            f"initial build, got {lag_mode!r} (band mode is deferred; see "
            "SYNTHETIC_V2_SPEC_AND_SPRINTS.md non-goals)."
        )
    te_grid = list(mix["target_te_grid"] if target_te_grid is None else target_te_grid)
    lags = list(mix["lag_grid"] if lag_grid is None else lag_grid)

    cells: List[CellV2] = []
    dropped: List[Dict[str, Any]] = []
    solved_cache: Dict[Tuple[float, int], Dict[str, Any]] = {}
    next_id = 0

    for target_te in te_grid:
        target_te = float(target_te)
        for delay in lags:
            delay = int(delay)
            if target_te == 0.0:
                cells.append(
                    CellV2(cell_id=next_id, target_te=0.0, D=delay,
                           B_y_scalar=0.0, te_block_realised=0.0)
                )
                next_id += 1
                continue

            key = (round(target_te, 9), delay)
            solution = solved_cache.get(key)
            if solution is None:
                try:
                    solution = solve_cell_coupling(
                        config, target_te, delay, benchmark=benchmark
                    )
                except ValueError as exc:
                    reason = str(exc)
                    logger.warning(
                        "enumerate_cells_v2: dropping unsolvable cell "
                        "(target_te=%g, D=%d): %s", target_te, delay, reason
                    )
                    dropped.append(
                        {"target_te": target_te, "D": delay, "reason": reason}
                    )
                    continue
                solved_cache[key] = solution

            cells.append(
                CellV2(
                    cell_id=next_id,
                    target_te=target_te,
                    D=delay,
                    B_y_scalar=float(solution["B_y_scalar"]),
                    te_block_realised=float(solution["te_block"]),
                )
            )
            next_id += 1

    logger.info(
        "enumerate_cells_v2: %d cells kept (%d null, %d signal), %d dropped.",
        len(cells),
        sum(1 for c in cells if c.target_te == 0.0),
        sum(1 for c in cells if c.target_te != 0.0),
        len(dropped),
    )
    return cells, dropped


def cell_seed(base_seed: int, cell_id: int, split: str) -> int:
    r"""Deterministic generation seed for one ``(cell, split)`` unit.

    Combines the umbrella DGP seed with a split-specific stride and the cell id so
    no two ``(cell, split)`` units share an RNG stream while the whole pool remains
    reproducible from ``base_seed`` alone (ported from ``mixed_dataset``).

    Args:
        base_seed: Umbrella DGP seed (``seeds.dgp``).
        cell_id: The cell's stable id.
        split: One of ``train`` / ``val`` / ``test``.

    Returns:
        A non-negative integer seed.

    Raises:
        ValueError: If ``split`` is not a recognised split name.
    """
    if split not in _SPLIT_SEED_OFFSET:
        raise ValueError(
            f"cell_seed: unknown split {split!r} (expected one of "
            f"{sorted(_SPLIT_SEED_OFFSET)})."
        )
    return (
        int(base_seed)
        + _SPLIT_SEED_OFFSET[split] * _SPLIT_SEED_STRIDE
        + int(cell_id) * _CELL_SEED_STRIDE
    )


def generate_pilot_samples(
    cell: CellV2,
    n: int,
    split: str,
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw",
    base_seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> Dict[str, Any]:
    r"""Generate ``n`` raw FHR/UP pairs for one cell (§5–§7).

    A thin, seed-deterministic wrapper over :func:`raw_generators.generate_cell_raw`
    that stamps the cell's solved coupling and label. **Shared with the Sprint 4
    cache build** so pilot and full-build generation cannot drift.

    Args:
        cell: The solved :class:`CellV2` to generate.
        n: Number of samples to generate.
        split: One of ``train`` / ``val`` / ``test`` (selects the seed offset).
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        base_seed: Umbrella DGP seed; ``seeds.dgp`` (falling back to ``base_seed``)
            is used when ``None``.
        render_mode: Optional override for ``raw.render_mode`` (only ``am_carrier`` in
            the initial build).

    Returns:
        The :func:`raw_generators.generate_cell_raw` dict (``fhr_raw``, ``up_raw``,
        ``true_lag_tt``, ``latents``, ``meta``), with ``meta['cell_id']`` added.
    """
    seeds = config.get("seeds", {})
    if base_seed is None:
        base_seed = int(seeds.get("dgp", seeds.get("base_seed", 0)))
    seed = cell_seed(base_seed, cell.cell_id, split)
    raw = generate_cell_raw(
        int(n),
        B=cell.B_y_scalar,
        D=cell.D,
        config=config,
        benchmark=benchmark,
        seed=seed,
        te_inj=cell.te_block_realised,
        render_mode=render_mode,
    )
    raw["meta"]["cell_id"] = int(cell.cell_id)
    return raw


def make_raw_provider(
    config: Dict[str, Any],
    split: str,
    *,
    benchmark: str = "G1_raw",
    cache_dir: Optional[Path] = None,
) -> Callable[[int, int], Tuple[np.ndarray, np.ndarray]]:
    r"""Build a memoised raw-waveform regenerator keyed by ``(cell_id, raw_index)``.

    The v2 cache stores only the $300$-step scattering features, not the raw $4\,\mathrm{Hz}$
    FHR/UP waveforms (§17). The per-sample diagnostic figure's first panel nonetheless wants the
    raw traces. Because generation is fully seed-deterministic (:func:`generate_pilot_samples`
    via :func:`cell_seed`) and the cache stamps each row's within-cell index
    (``sample_raw_index``), the exact raw pair for any shuffled cache row can be regenerated on
    demand from ``(cell_id, raw_index)`` alone — nothing raw needs to be persisted.

    The returned closure regenerates each cell's ``n_per_cell`` raw pairs **once** (cached by
    ``cell_id``) and indexes the requested row, returning the analysis-window slice
    ``[TRIM_STEPS·DECIMATION : n_raw − TRIM_STEPS·DECIMATION]`` — the $4800$-sample /
    $20\,\mathrm{min}$ window aligned to the $300$-step feature grid. Only the handful of samples
    actually plotted (:func:`run_pipeline_v2.run_test_plots` stops at ``analysis_samples``)
    trigger a regeneration, so at most a few cells are ever rebuilt.

    **Fully cache-authoritative.** Because :func:`raw_generators.generate_cell_raw` scales the
    AM envelope by the *batch-pooled* latent std, a row's raw amplitude depends on the ``n``
    passed to it — so regenerating with a different ``n`` than the build used would silently
    rescale the waveform. This provider therefore takes the per-cell ``n`` from the **cache's own
    row counts** (``sample_cell_id`` bincount), the solved $B$ / seeds / render mode / ``n_raw``
    (window geometry) from ``meta.json``, and **refuses to build** (raises) when either is missing
    — :func:`run_pipeline_v2.run_test_plots` then simply omits the raw panel rather than plotting a
    waveform inconsistent with the row's cached features. The live ``config`` is used only for the
    generation-invariant latent/raw physics (oscillators, ``f_pulse``, band powers), which the
    normal build → plot workflow does not change without a rebuild.

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        split: The cache split the rows come from (``test`` / ``val`` / ``train``); selects the
            generation seed offset and the split's ``.npz``.
        benchmark: Active benchmark key under ``benchmarks``.
        cache_dir: Override for the cache directory holding ``meta.json`` + ``<split>.npz``
            (defaults to :func:`resolve_cache_dir`).

    Returns:
        A **total** callable ``provider(cell_id, raw_index) -> (fhr_win, up_win)`` returning two
        1-D ``float32`` arrays of length ``n_raw - 2·TRIM_STEPS·DECIMATION`` in physical units
        (bpm / mmHg); on any failure (unknown cell / out-of-range row / regen error) it returns a
        NaN-filled window of the same length rather than raising, so a batch never ends up with
        inconsistent keys. The returned function carries a ``window_length`` attribute.

    Raises:
        FileNotFoundError / ValueError / KeyError: At construction, when ``meta.json`` or the
            split ``.npz`` (with ``sample_cell_id``) is absent/unusable — raw is then unavailable.
    """
    from .raw_generators import DECIMATION
    from .scattering_adapter import TRIM_STEPS

    cdir = resolve_cache_dir(config, benchmark=benchmark) if cache_dir is None else Path(cache_dir)

    meta_path = cdir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"make_raw_provider: no meta.json under {cdir}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    cells_by_id: Dict[int, CellV2] = {}
    for c in meta.get("cells", []) or []:
        cid = int(c["cell_id"])
        cells_by_id[cid] = CellV2(
            cell_id=cid, target_te=float(c.get("target_te", 0.0)), D=int(c["D"]),
            B_y_scalar=float(c.get("B_y_scalar", 0.0)),
            te_block_realised=float(c.get("te_block_realised", c.get("te_inj", 0.0))),
        )
    if not cells_by_id:
        raise ValueError(f"make_raw_provider: meta.json under {cdir} has no cells")
    seeds_meta = meta.get("seeds", {}) or {}
    base_seed = int(seeds_meta.get("dgp", seeds_meta.get("base_seed", 0)))
    render_mode = meta.get("render_mode")
    # Window geometry from the manifest's raw block (authoritative), not the live config.
    n_raw = int(((meta.get("raw") or {}).get("n_raw"))
                or config["benchmarks"][benchmark]["raw"]["n_raw"])
    win = slice(int(TRIM_STEPS) * int(DECIMATION), n_raw - int(TRIM_STEPS) * int(DECIMATION))
    win_len = int(win.stop - win.start)

    # Per-cell sample count straight from the cache: the number of rows stamped with each
    # cell_id equals the n the build passed to generate_cell_raw for that (cell, split), so the
    # pooled-std AM amplitude (and thus the row's raw) is reproduced exactly.
    split_npz = cdir / f"{split}.npz"
    if not split_npz.is_file():
        raise FileNotFoundError(f"make_raw_provider: no {split}.npz under {cdir}")
    with np.load(split_npz) as npz:
        if "sample_cell_id" not in npz.files:
            raise KeyError(f"make_raw_provider: {split}.npz lacks sample_cell_id")
        cell_ids = np.asarray(npz["sample_cell_id"]).astype(np.int64)
    n_by_cell = {int(cid): int(np.count_nonzero(cell_ids == cid))
                 for cid in np.unique(cell_ids)}

    cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    nan_win = np.full(win_len, np.nan, dtype=np.float32)

    def provider(cell_id: int, raw_index: int) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return the (fhr, up) analysis window for one row; a NaN window on any failure."""
        cid, ri = int(cell_id), int(raw_index)
        try:
            if cid not in cells_by_id or cid not in n_by_cell:
                raise KeyError(f"unknown cell_id {cid}")
            if cid not in cache:
                raw = generate_pilot_samples(
                    cells_by_id[cid], int(n_by_cell[cid]), split, config,
                    benchmark=benchmark, base_seed=base_seed, render_mode=render_mode,
                )
                # Retain only the trimmed analysis window (drop the 15-step/end edges).
                cache[cid] = (
                    np.ascontiguousarray(raw["fhr_raw"][:, win], dtype=np.float32),
                    np.ascontiguousarray(raw["up_raw"][:, win], dtype=np.float32),
                )
            fhr_all, up_all = cache[cid]
            if not 0 <= ri < fhr_all.shape[0]:
                raise IndexError(
                    f"raw_index {ri} out of range for cell {cid} (n={fhr_all.shape[0]})"
                )
            return fhr_all[ri].copy(), up_all[ri].copy()
        except Exception as exc:  # noqa: BLE001 -- raw is a plotting nicety; degrade to blank
            logger.warning("make_raw_provider: raw unavailable for (cell=%d, row=%d): %s",
                           cid, ri, exc)
            return nan_win.copy(), nan_win.copy()

    provider.window_length = win_len  # type: ignore[attr-defined]
    return provider


# ===========================================================================
# Sprint 4: the full build (generate -> scatter -> normalise -> cache)
# ===========================================================================
#
# The build is three staged passes so pooled normalisation and per-(cell, split)
# resumability coexist (see SYNTHETIC_V2_SPEC_AND_SPRINTS.md Sprint 4):
#
#   Stage 1 (build_split_parts): generate + scatter each (cell, split) to an
#     UN-normalised channels-first part ``.npz`` under ``_parts/`` -- the resume
#     checkpoint. Skipped if the part already exists.
#   Stage 2 (fit_pool_stats): accumulate per-channel mean/std over the TRAIN parts
#     (or load a real fold's ``stats.hdf5``). Fitting the normaliser ONCE on the
#     pooled train split (not per cell) keeps the same physical scattering value
#     mapping to the same z-score across cells -- the model needs a consistent
#     ch-0 baseline / absolute scale. TE is z-score-invariant (§12) so frac_Phi is
#     unaffected.
#   Stage 3 (assemble_split -> write_cache_v2): normalise each part with the pooled
#     stats, probe frac_Phi at build N, stamp §17 provenance, pool all cells, apply
#     one shared row-aligned permutation, and write ``{split}.npz`` + ``meta.json``.


def resolve_cache_dir(config: Dict[str, Any], *, benchmark: str = "G1_raw") -> Path:
    r"""Resolve the ``<data_dir>/<benchmark>/<tag>/`` cache directory for this run.

    Mirrors the v1 ``datamodule_synth`` convention (``data_root / benchmark / tag``)
    and :func:`run_pipeline_v2._results_dir`: a relative ``paths.data_dir`` resolves
    against this module's directory, and the leaf is ``experiment.tag``.

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key (also the middle path component).

    Returns:
        The cache directory as an absolute :class:`Path` (not created).
    """
    tag = str(config.get("experiment", {}).get("tag", benchmark))
    # Expand ``~`` and ``$VAR`` / ``${VAR}`` (matching the v1 ``resolve_user_path``) BEFORE
    # the absolute check, so an env-/home-relative ``data_dir`` (e.g. ``${SCRATCH}/te_cache``
    # on the prod box) resolves to the intended mount rather than a literal directory joined
    # under the source tree.
    raw_dir = os.path.expanduser(os.path.expandvars(
        str(config.get("paths", {}).get("data_dir", "./data"))
    ))
    data_dir = Path(raw_dir)
    if not data_dir.is_absolute():
        data_dir = _MODULE_DIR / data_dir
    return data_dir / benchmark / tag


def _part_path(parts_dir: Path, split: str, cell_id: int) -> Path:
    r"""Path of the Stage-1 un-normalised part for one ``(cell, split)`` unit."""
    return parts_dir / f"{split}_cell{int(cell_id):03d}.npz"


def _part_fingerprint(
    cell: CellV2, n: int, split: str, config: Dict[str, Any], *, benchmark: str,
    render_mode: Optional[str],
) -> str:
    r"""Stable fingerprint of everything that determines a Stage-1 part's contents.

    A part on disk is only a valid resume target if it was generated for the *same*
    request. The fingerprint therefore captures the sample count, the cell's identity and
    solved coupling, the deterministic generation seed, and the generation-affecting raw
    knobs. On resume a part whose stored fingerprint differs (e.g. after bumping
    ``n_per_cell`` or widening the grid so ``cell_id`` -> cell mapping shifted) is
    regenerated rather than silently reused (guards the "scale up by re-running" workflow).

    Args:
        cell: The cell the part is for.
        n: Requested samples for this ``(cell, split)``.
        split: One of ``train`` / ``val`` / ``test``.
        config: The parsed config tree.
        benchmark: Active benchmark key.
        render_mode: The effective ``render_mode`` override (``None`` -> config default).

    Returns:
        A ``|``-joined fingerprint string stored inside the part ``.npz``.
    """
    bench = config["benchmarks"][benchmark]
    raw = bench["raw"]
    seeds = config.get("seeds", {})
    base_seed = int(seeds.get("dgp", seeds.get("base_seed", 0)))
    seed = cell_seed(base_seed, cell.cell_id, split)
    render = render_mode if render_mode is not None else raw.get("render_mode")
    # FHRV dressing is baked into the raw waveform before the transform, so a change to the
    # notch flag, the FHRV band powers, or Q (which sets the notch width) must invalidate a
    # cached part. Represent fhrv_band_power deterministically (sorted key=value pairs).
    fhrv_power = raw.get("fhrv_band_power", {}) or {}
    fhrv_power_str = ",".join(f"{k}={float(v):.10g}" for k, v in sorted(fhrv_power.items()))
    return "|".join([
        "v2",
        f"n={int(n)}",
        f"cid={int(cell.cell_id)}",
        f"D={int(cell.D)}",
        f"B={float(cell.B_y_scalar):.10g}",
        f"te={float(cell.te_block_realised):.10g}",
        f"seed={int(seed)}",
        f"render={render}",
        f"f_pulse={raw.get('f_pulse')}",
        f"fs={raw.get('fs')}",
        f"n_raw={raw.get('n_raw')}",
        f"fhrv_notch={bool(raw.get('fhrv_notch_enabled', True))}",
        f"Q={bench.get('scattering', {}).get('Q')}",
        f"fhrv_power={fhrv_power_str}",
    ])


def _read_part_fingerprint(path: Path) -> Optional[str]:
    r"""Return the fingerprint stored in a part ``.npz``, or ``None`` if absent/unreadable.

    Reads only the small ``_fingerprint`` member (``NpzFile`` is lazy per-member), so this
    does not load the multi-hundred-MB feature arrays.
    """
    try:
        with np.load(path) as existing:
            if "_fingerprint" not in existing.files:
                return None
            return str(existing["_fingerprint"].item())
    except Exception:  # noqa: BLE001 -- a corrupt/partial part is treated as "regenerate"
        return None


def _savez_atomic(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    r"""Write an uncompressed ``.npz`` atomically (temp file + ``os.replace``).

    ``np.savez`` (ZIP **stored**, not deflated) keeps the archive memory-mappable by
    :class:`dataset_v2.SyntheticTEDatasetV2`. Writing to a sibling temp path and then
    renaming means a crash mid-write never leaves a half-written cache/part that a
    resume would mistake for complete.

    Args:
        path: Destination ``.npz`` path.
        arrays: Mapping of array name to numpy array.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # The temp name must end in ``.npz`` -- ``np.savez`` appends ``.npz`` when the given
    # path lacks it, which would otherwise write to ``<name>.tmp.npz`` and defeat the
    # rename below.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(str(tmp), **arrays)
    os.replace(str(tmp), str(path))


def _write_json_atomic(path: Path, obj: Any) -> None:
    r"""Write ``obj`` as pretty JSON atomically (temp file + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)
    os.replace(str(tmp), str(path))


def build_split_parts(
    cells: Sequence[CellV2],
    split: str,
    n: int,
    *,
    config: Dict[str, Any],
    adapter: Any,
    parts_dir: Path,
    benchmark: str = "G1_raw",
    resume: bool = True,
    render_mode: Optional[str] = None,
) -> List[Path]:
    r"""Stage 1: generate + scatter each ``(cell, split)`` to an un-normalised part (S4-T02c).

    For each cell, generates ``n`` raw pairs via :func:`generate_pilot_samples` (seeded
    deterministically by :func:`cell_seed`), runs the real scattering transform
    (:meth:`scattering_adapter.ScatteringAdapter.transform_raw`) to **un-normalised**
    channels-first fields $(n, C, 300)$, trims ``true_lag_tt`` to the feature grid
    ``[trim : trim + T]`` ($=$ ``latent[15:315]``, §3), and writes an atomic part
    ``.npz``. A part that already exists is **skipped** (crash-safe resume), so a
    re-run continues where it stopped and reproduces the identical final cache.

    Args:
        cells: The cells to build for this split.
        split: One of ``train`` / ``val`` / ``test``.
        n: Samples per cell.
        config: The parsed ``config_synth_v2.yaml`` tree.
        adapter: A live :class:`scattering_adapter.ScatteringAdapter`.
        parts_dir: Directory for the per-unit part files.
        benchmark: Active benchmark key under ``benchmarks``.
        resume: Skip a unit whose part already exists (default ``True``).
        render_mode: Optional ``raw.render_mode`` override.

    Returns:
        The list of part paths (one per cell), in cell order.
    """
    paths: List[Path] = []
    for cell in cells:
        pp = _part_path(parts_dir, split, cell.cell_id)
        paths.append(pp)
        fingerprint = _part_fingerprint(
            cell, int(n), split, config, benchmark=benchmark, render_mode=render_mode
        )
        # Resume only when an existing part was generated for the SAME request: a
        # fingerprint mismatch (bumped n_per_cell, changed grid so cell_id remaps, or
        # altered render knobs) forces regeneration instead of silently reusing a stale,
        # wrong-sized / wrong-content part.
        if resume and pp.is_file() and _read_part_fingerprint(pp) == fingerprint:
            logger.info("build_split_parts: skip existing %s", pp.name)
            continue
        if resume and pp.is_file():
            logger.info(
                "build_split_parts: regenerating %s (fingerprint changed)", pp.name
            )
        raw = generate_pilot_samples(
            cell, int(n), split, config, benchmark=benchmark, render_mode=render_mode
        )
        fields_cf = adapter.transform_raw(raw["fhr_raw"], raw["up_raw"])  # (n, C, 300)
        t_out = int(fields_cf["fhr_st"].shape[2])
        lo = int(adapter.trim)
        true_lag = np.asarray(
            raw["true_lag_tt"][:, lo : lo + t_out], dtype=np.int16
        )
        part = {name: np.asarray(fields_cf[name], dtype=np.float32) for name in _FIELD_NAMES}
        part["true_lag_tt"] = true_lag
        part["_fingerprint"] = np.array(fingerprint)
        _savez_atomic(pp, part)
        logger.info(
            "build_split_parts: wrote %s (n=%d)", pp.name, part["fhr_st"].shape[0]
        )
    return paths


def fit_pool_stats(
    cells: Sequence[CellV2],
    split: str,
    *,
    adapter: Any,
    parts_dir: Path,
) -> Dict[str, Dict[str, np.ndarray]]:
    r"""Stage 2: per-channel normalisation stats pooled over one split's parts (S4-T02a).

    When ``norm_stats_source == 'real_fold'`` the fold's ``stats.hdf5`` is loaded and
    returned unchanged. Otherwise (``synthetic_pool``, the default) the per-channel
    mean/std are accumulated **incrementally** ($O(C)$ memory) over the split's parts,
    on the *transformed* features (post log / asinh) exactly as
    :func:`scattering_adapter.compute_norm_stats` does, so the whole pool cannot be held
    in RAM at once. Pass ``split='train'`` to fit on the training pool and apply the
    same stats to val/test.

    Args:
        cells: The cells whose parts contribute to the stats.
        split: The split whose parts to pool (normally ``train``).
        adapter: A live :class:`scattering_adapter.ScatteringAdapter` (its
            ``norm_stats_source`` / ``real_fold_stats_path`` govern the source).
        parts_dir: Directory holding the Stage-1 parts.

    Returns:
        A dict ``{field: {'mean': (C,), 'std': (C,)}}`` (float32).
    """
    from .scattering_adapter import (  # local: pulls the transformed-feature helper
        LOG_EPSILON,
        _transform_field,
        load_real_fold_stats,
    )

    if adapter.norm_stats_source == "real_fold":
        return load_real_fold_stats(adapter.real_fold_stats_path)

    acc: Dict[str, Dict[str, Any]] = {}
    for cell in cells:
        pp = _part_path(parts_dir, split, cell.cell_id)
        with np.load(pp) as part:
            for field in _FIELD_NAMES:
                arr = np.asarray(part[field])
                transformed = _transform_field(arr, field, LOG_EPSILON)  # (n, C, T) f64
                s = transformed.sum(axis=(0, 2))
                sq = (transformed * transformed).sum(axis=(0, 2))
                count = int(arr.shape[0]) * int(arr.shape[2])
                if field not in acc:
                    acc[field] = {"sum": s, "sumsq": sq, "count": count}
                else:
                    acc[field]["sum"] += s
                    acc[field]["sumsq"] += sq
                    acc[field]["count"] += count

    stats: Dict[str, Dict[str, np.ndarray]] = {}
    for field, a in acc.items():
        count = max(int(a["count"]), 1)
        mean = a["sum"] / count
        var = np.maximum(a["sumsq"] / count - mean * mean, 0.0)
        stats[field] = {
            "mean": mean.astype(np.float32),
            "std": np.sqrt(var).astype(np.float32),
        }
    return stats


def _save_stats(path: Path, stats: Dict[str, Dict[str, np.ndarray]]) -> None:
    r"""Persist pooled normalisation stats to ``norm_stats.npz`` (``<field>_<stat>`` keys)."""
    flat: Dict[str, np.ndarray] = {}
    for field, sd in stats.items():
        flat[f"{field}_mean"] = np.asarray(sd["mean"], dtype=np.float32)
        flat[f"{field}_std"] = np.asarray(sd["std"], dtype=np.float32)
    _savez_atomic(path, flat)


def assemble_split(
    cells: Sequence[CellV2],
    split: str,
    *,
    config: Dict[str, Any],
    parts_dir: Path,
    stats: Dict[str, Dict[str, np.ndarray]],
    coupled: Dict[str, Any],
    benchmark: str = "G1_raw",
) -> Tuple[Dict[str, np.ndarray], Dict[int, Dict[str, Any]]]:
    r"""Stage 3: normalise -> probe -> stamp -> pool -> shuffle one split (S4-T02a/b).

    For each cell, loads its Stage-1 part, normalises with the pooled ``stats``, runs the
    §14.3 :func:`eval_v2.measure_te_scat` probe on the coupled pulse-shape channels at
    build $N$ (a probe failure is non-fatal -> ``te_scat = 0``), stamps the §17
    per-sample provenance, then concatenates every cell and applies one shared
    row-aligned permutation seeded from ``seeds.shuffle`` (so features and provenance
    stay aligned and a ``shuffle=False`` consumer still sees mixed cells).

    Args:
        cells: The cells to assemble.
        split: One of ``train`` / ``val`` / ``test``.
        config: The parsed ``config_synth_v2.yaml`` tree.
        parts_dir: Directory holding the Stage-1 parts.
        stats: The pooled normalisation stats from :func:`fit_pool_stats`.
        coupled: The ``coupled_channel_indices`` dict.
        benchmark: Active benchmark key under ``benchmarks``.

    Returns:
        ``(arrays, cell_probe)`` — the pooled, shuffled cache arrays (four fields +
        ``weight`` + ``true_lag_tt`` + the seven ``sample_*`` provenance arrays, incl.
        ``sample_raw_index`` for deterministic raw regeneration), and a
        ``{cell_id: {'te_scat', 'frac_phi', 'n_used'}}`` map of the per-cell probe result.
    """
    from .scattering_adapter import normalise_fields  # local: numpy-only, but co-located with torch
    from .eval_v2 import measure_te_scat  # local: avoids the eval<->build circular import

    per_field: Dict[str, List[np.ndarray]] = {name: [] for name in _FIELD_NAMES}
    lag_list: List[np.ndarray] = []
    te_true, te_scat_a, frac_a, delay_a, cell_a, held_a, weight_a = [], [], [], [], [], [], []
    raw_idx_a: List[np.ndarray] = []  # within-cell row index -> deterministic raw regeneration key
    cell_probe: Dict[int, Dict[str, Any]] = {}

    for cell in cells:
        pp = _part_path(parts_dir, split, cell.cell_id)
        with np.load(pp) as part:
            cf = {name: np.asarray(part[name]) for name in _FIELD_NAMES}
            true_lag = np.asarray(part["true_lag_tt"], dtype=np.int16)
        normed_cf = normalise_fields(cf, stats)
        fields = {
            name: np.ascontiguousarray(np.transpose(normed_cf[name], (0, 2, 1)))
            for name in _FIELD_NAMES
        }  # model-facing (n, T, C)
        n_c = int(fields["fhr_st"].shape[0])
        t_c = int(fields["fhr_st"].shape[1])

        # frac_Phi probe at build N (non-fatal: a degenerate tiny pool must not abort).
        try:
            probe = measure_te_scat(fields, cell, coupled, config=config, benchmark=benchmark)
            te_scat = probe["te_scat"]
            frac = probe["frac_phi"]
            n_used = int(probe["n_used"])
        except Exception as exc:  # noqa: BLE001 -- diagnostics only, never abort the build
            logger.warning(
                "assemble_split: frac_Phi probe failed for cell %d (%s); stamping 0.",
                cell.cell_id, exc,
            )
            te_scat, frac, n_used = float("nan"), None, 0
        te_scat_v = float(te_scat) if np.isfinite(te_scat) else 0.0
        frac_v = float(frac) if (frac is not None and np.isfinite(frac)) else 0.0
        cell_probe[int(cell.cell_id)] = {
            "te_scat": te_scat_v,
            "frac_phi": (None if frac is None else frac_v),
            "n_used": n_used,
        }

        for name in _FIELD_NAMES:
            per_field[name].append(fields[name])
        lag_list.append(true_lag)
        te_true.append(np.full(n_c, cell.te_block_realised, dtype=np.float32))
        te_scat_a.append(np.full(n_c, te_scat_v, dtype=np.float32))
        frac_a.append(np.full(n_c, frac_v, dtype=np.float32))
        delay_a.append(np.full(n_c, cell.D, dtype=np.int16))
        cell_a.append(np.full(n_c, cell.cell_id, dtype=np.int16))
        held_a.append(np.zeros(n_c, dtype=np.int8))
        # Within-cell row index (0..n_c-1), in the pre-shuffle generation order. This is the
        # deterministic key that lets a plot re-generate the exact raw waveform for a shuffled
        # cache row via ``make_raw_provider`` (cell parts are read in cell_id order and each row
        # is the i-th sample of ``generate_pilot_samples(cell, n_c, split)``).
        raw_idx_a.append(np.arange(n_c, dtype=np.int32))
        weight_a.append(np.ones((n_c, t_c), dtype=np.float32))

    arrays: Dict[str, np.ndarray] = {
        name: np.concatenate(per_field[name], axis=0) for name in _FIELD_NAMES
    }
    arrays["weight"] = np.concatenate(weight_a, axis=0)
    arrays["true_lag_tt"] = np.concatenate(lag_list, axis=0)
    arrays["sample_te_true"] = np.concatenate(te_true, axis=0)
    arrays["sample_te_scat"] = np.concatenate(te_scat_a, axis=0)
    arrays["sample_frac_phi"] = np.concatenate(frac_a, axis=0)
    arrays["sample_delay"] = np.concatenate(delay_a, axis=0)
    arrays["sample_cell_id"] = np.concatenate(cell_a, axis=0)
    arrays["sample_held_out"] = np.concatenate(held_a, axis=0)
    arrays["sample_raw_index"] = np.concatenate(raw_idx_a, axis=0)

    # One shared row-aligned permutation (ported from mixed_dataset.build_mixed_split):
    # a small split offset off the shuffle seed keeps train/val/test independent.
    shuffle_seed = int(config.get("seeds", {}).get("shuffle", 0))
    rng = np.random.default_rng(shuffle_seed + _SPLIT_SEED_OFFSET[split])
    perm = rng.permutation(int(arrays["fhr_st"].shape[0]))
    for key in list(arrays.keys()):
        arrays[key] = np.ascontiguousarray(arrays[key][perm])
    return arrays, cell_probe


def write_cache_v2(
    out_dir: Path,
    splits: Dict[str, Dict[str, np.ndarray]],
    *,
    cells: Sequence[CellV2],
    dropped: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    adapter: Any,
    coupled: Dict[str, Any],
    cell_probe: Dict[int, Dict[str, Any]],
    n_per_split: Dict[str, int],
    benchmark: str = "G1_raw",
) -> Path:
    r"""Write the split ``.npz`` caches and the shared ``meta.json`` (§17, S4-T02b).

    The model-facing tensors match the existing synthetic contract; the provenance is v2
    (no ``sample_M`` / band fields, plus ``sample_te_scat`` / ``sample_frac_phi``). The
    top-level ``te_true`` is the pooled mean of the cells' $\mathrm{TE}_{\mathrm{inj}}$ and
    ``true_lag_band`` is the union $\{0, \dots, \max D - 1\}$; the authoritative per-cell
    ground truth (incl. the train-split ``te_scat_measured`` / ``frac_phi``) lives in the
    ``cells`` manifest.

    Args:
        out_dir: Destination cache directory.
        splits: ``{split: arrays}`` from :func:`assemble_split`.
        cells: The kept cells.
        dropped: The unsolvable cells dropped during enumeration.
        config: The parsed ``config_synth_v2.yaml`` tree.
        adapter: The live scattering adapter (for channel counts).
        coupled: The coupled-channel index dict.
        cell_probe: The **train-split** per-cell probe results (te_scat / frac_phi).
        n_per_split: ``{split: n_per_cell}`` used for this build.
        benchmark: Active benchmark key under ``benchmarks``.

    Returns:
        ``out_dir`` (the cache directory that now holds the splits + ``meta.json``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, arrays in splits.items():
        _savez_atomic(out_dir / f"{split}.npz", arrays)
        logger.info("write_cache_v2: [%s] n=%d -> %s.npz",
                    split, int(arrays["fhr_st"].shape[0]), split)

    bench = config["benchmarks"][benchmark]
    data = bench["data"]
    mix = bench["mix"]
    horizon = int(data["horizon"])
    T = int(data["sequence_length"])
    c_y_st = int(adapter.scattering_channels)
    c_y_ph = int(adapter.fhr_ph_channels)
    c_u_st = int(adapter.scattering_channels)
    c_u_ph = int(adapter.up_ph_channels)
    max_delay = max((int(c.D) for c in cells), default=1)
    te_pooled = float(np.mean([c.te_block_realised for c in cells])) if cells else 0.0

    cell_manifest: List[Dict[str, Any]] = []
    for c in cells:
        probe = cell_probe.get(int(c.cell_id), {})
        cell_manifest.append({
            "cell_id": int(c.cell_id),
            "target_te": float(c.target_te),
            "D": int(c.D),
            "B_y_scalar": float(c.B_y_scalar),
            "te_block_realised": float(c.te_block_realised),
            "te_scat_measured": probe.get("te_scat"),
            "frac_phi": probe.get("frac_phi"),
        })

    meta: Dict[str, Any] = {
        "benchmark": benchmark,
        "tag": str(config.get("experiment", {}).get("tag", benchmark)),
        "render_mode": str(bench["raw"].get("render_mode", "am_carrier")),
        "te_true": te_pooled,
        "te_per_step": te_pooled / horizon if horizon else 0.0,
        "true_lag_band": list(range(int(max_delay))),
        "horizon": horizon,
        "sequence_length": T,
        "clean_anchor_range": [int(max_delay) - 1, T - horizon],
        "c_y": c_y_st + c_y_ph,
        "c_u": c_u_st + c_u_ph,
        "channel_map": {
            "fhr_st": [0, c_y_st],
            "fhr_ph": [c_y_st, c_y_st + c_y_ph],
            "up_st": [0, c_u_st],
            "up_ph": [c_u_st, c_u_st + c_u_ph],
        },
        "coupled_channel": {k: (float(v) if k in ("hz", "xi") else int(v))
                            for k, v in coupled.items()},
        "n_per_cell": {s: int(n) for s, n in n_per_split.items()},
        "grid": {
            "target_te_grid": [float(t) for t in mix["target_te_grid"]],
            "lag_grid": [int(d) for d in mix["lag_grid"]],
            "lag_mode": str(mix.get("lag_mode", "fixed")),
        },
        "raw": bench["raw"],
        "scattering": bench["scattering"],
        "seeds": dict(config.get("seeds", {})),
        "cells": cell_manifest,
        "dropped": [dict(d) for d in dropped],
    }
    _write_json_atomic(out_dir / "meta.json", meta)
    logger.info(
        "write_cache_v2: meta.json te_true(pooled)=%.4f nats  lag_band=0..%d  cells=%d",
        te_pooled, int(max_delay) - 1, len(cells),
    )
    return out_dir


def build_all(
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw",
    pilot: bool = False,
    resume: bool = True,
    out_dir: Optional[Path] = None,
    adapter: Optional[Any] = None,
    grid_override: Optional[Dict[str, Any]] = None,
    n_override: Optional[Dict[str, int]] = None,
) -> Path:
    r"""Build the full v2 cache: enumerate -> Stage 1 -> Stage 2 -> Stage 3 -> write (S4).

    Orchestrates the three staged passes over all splits. ``pilot`` selects the small
    ``eval.realizability.pilot`` grid; otherwise the locked ``mix`` grid is used with
    ``mix.n_per_cell_{train,val,test}``. The build is deterministic (seeds) and resumable
    (Stage-1 parts under ``out_dir/_parts/`` are reused).

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        pilot: Use the pilot grid + a single ``n_per_cell`` for every split.
        resume: Reuse existing Stage-1 parts (default ``True``).
        out_dir: Cache directory override (defaults to :func:`resolve_cache_dir`).
        adapter: A prebuilt :class:`scattering_adapter.ScatteringAdapter` to reuse
            (the filter bank is expensive; tests share one). Built here when ``None``.
        grid_override: ``{'target_te_grid', 'lag_grid'}`` to force a specific grid
            (tests use a tiny one); overrides ``pilot`` / config.
        n_override: ``{'train', 'val', 'test'}`` sample counts to force (tests use tiny N).

    Returns:
        The cache directory holding ``train/val/test.npz``, ``meta.json`` and
        ``norm_stats.npz``.
    """
    bench = config["benchmarks"][benchmark]
    mix = bench["mix"]

    if grid_override is not None:
        cells, dropped = enumerate_cells_v2(
            config, benchmark=benchmark,
            target_te_grid=grid_override.get("target_te_grid"),
            lag_grid=grid_override.get("lag_grid"),
        )
    elif pilot:
        pil = bench["eval"]["realizability"]["pilot"]
        cells, dropped = enumerate_cells_v2(
            config, benchmark=benchmark,
            target_te_grid=pil["target_te_grid"], lag_grid=pil["lag_grid"],
        )
    else:
        cells, dropped = enumerate_cells_v2(config, benchmark=benchmark)

    if n_override is not None:
        n_per_split = {s: int(n_override[s]) for s in ("train", "val", "test")}
    elif pilot:
        pn = int(bench["eval"]["realizability"]["pilot"]["n_per_cell"])
        n_per_split = {"train": pn, "val": max(1, pn // 2), "test": max(1, pn // 2)}
    else:
        n_per_split = {
            "train": int(mix["n_per_cell_train"]),
            "val": int(mix["n_per_cell_val"]),
            "test": int(mix["n_per_cell_test"]),
        }

    if not cells:
        raise ValueError(
            f"build_all: no cells to build (all dropped: {dropped}); check the grid."
        )

    out_dir = resolve_cache_dir(config, benchmark=benchmark) if out_dir is None else Path(out_dir)
    parts_dir = out_dir / "_parts"

    if adapter is None:
        from .scattering_adapter import ScatteringAdapter
        adapter = ScatteringAdapter(config, benchmark=benchmark)
    coupled = adapter.coupled_channel_indices()

    logger.info(
        "build_all: %d cells, n_per_split=%s -> %s", len(cells), n_per_split, out_dir
    )

    # Stage 1: generate + scatter every (cell, split) to un-normalised parts (resumable).
    for split in ("train", "val", "test"):
        build_split_parts(
            cells, split, n_per_split[split], config=config, adapter=adapter,
            parts_dir=parts_dir, benchmark=benchmark, resume=resume,
        )

    # Stage 2: fit the pooled normaliser ONCE on the train parts (or load a real fold).
    stats = fit_pool_stats(cells, "train", adapter=adapter, parts_dir=parts_dir)
    _save_stats(out_dir / "norm_stats.npz", stats)

    # Stage 3: normalise + probe + stamp + pool + shuffle each split, then write.
    splits: Dict[str, Dict[str, np.ndarray]] = {}
    train_cell_probe: Dict[int, Dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        arrays, probe = assemble_split(
            cells, split, config=config, parts_dir=parts_dir,
            stats=stats, coupled=coupled, benchmark=benchmark,
        )
        splits[split] = arrays
        if split == "train":
            train_cell_probe = probe

    write_cache_v2(
        out_dir, splits, cells=cells, dropped=dropped, config=config, adapter=adapter,
        coupled=coupled, cell_probe=train_cell_probe, n_per_split=n_per_split,
        benchmark=benchmark,
    )
    return out_dir
