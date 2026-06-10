r"""``mixed_dataset`` -- build a heterogeneous (mixed-population) G1 cache.

Every other v2 benchmark caches a **single** $(M, \mathrm{TE}, \text{lag})$
setting per dataset and the sweep machinery trains one model per cell. The
``G1_mix`` experiment asks the complementary question: train **one** model on a
**single heterogeneous pool** that mixes informative-channel counts $M$, target
block transfer entropies $\mathrm{TE}$, and lag bands, then recover the true
KLD / lag / TE **per sub-population** and test extrapolation to held-out
$(M, \mathrm{TE}, \text{lag})$ triples (see :mod:`mixed_eval`).

This module solves and generates one cell per grid triple and concatenates the
samples into one cache under ``data/G1_mix/<tag>/``, stamping **per-sample
provenance** (``sample_te_true`` / ``sample_M`` / ``sample_delay_min`` /
``sample_delay_max`` / ``sample_band_id`` / ``sample_cell_id`` /
``sample_held_out``) so :class:`dataset.SyntheticTEDataset` can expose each
sample's own cell TE and grouping keys. All cells share the G1 AR(2)-oscillator
DGP (:func:`generators.gen_state_space_oscillator`) with the within-signal
random-walk lag.

Design notes:
    * **TE held fixed across $M$.** Per Section 3.1.1 the $M$ informative
      channels are independent, so a cell's per-channel target is
      $\mathrm{TE}/M$ and the cell block TE is the per-channel solve $\times M$.
      The coupling $B_y$ is solved once per distinct
      $(d_{\min}, d_{\max}, \mathrm{TE}/M)$ via
      :func:`analytic_te.B_y_for_mean_te_block_state_space` (memoised --
      $M$-independent because the solve uses the single oscillator spec) and
      reused across all three splits.
    * **MC-floor trim.** At high $M$ and low $\mathrm{TE}$ the per-channel
      target falls below the Monte-Carlo bias floor of the TE estimator; those
      cells are trimmed (and a trimmed *held-out* triple is a hard error).
    * **Per-cell standardisation.** The generator z-scores per channel within a
      cell; the pool is **not** re-standardised (z-scoring is an affine,
      TE-invariant map and each cell is already $\approx \mathcal{N}(0,1)$).

Run modes (Decision V2-D8): both a CLI and an edit-and-run ``RUN_CONFIG``,
auto-detected from whether any command-line argument is present.

    * CLI::

        python -m ...synthetic.mixed_dataset --tag G1_mix_base
        python -m ...synthetic.mixed_dataset --tag G1_mix_base --holdout

    * Edit-and-run: edit ``RUN_CONFIG`` in ``__main__`` and run the file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from model.vae_teb_prediction.model.model_experiment.synthetic.analytic_te import (
    B_y_for_mean_te_block_state_space,
    mean_te_block_state_space_over_delays,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.build_dataset import (
    _resolve_channel_decomp,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.generators import (
    _decomp_to_json,
    _make_channel_layout,
    gen_state_space_oscillator,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (
    apply_path_overrides,
    load_config,
    resolve_active_benchmark,
    resolve_user_path,
)

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"
_BENCHMARK = "G1_mix"
_SPLIT_FILES = ("train.npz", "val.npz", "test.npz", "meta.json")
_HOLDOUT_SUFFIX = "_holdout"
# Independent per-split / per-cell seed offsets so train / val / test and the
# distinct cells never share an RNG stream.
_SPLIT_OFFSET = {"train": 0, "val": 1, "test": 2}


# =============================================================================
# Multi-core build helpers
# =============================================================================
#
# The two CPU-bound build phases -- the per-distinct-key coupling solves
# (:func:`enumerate_mix_cells`) and the per-(cell, split) sample generation
# (:func:`build_mixed_split`) -- are embarrassingly parallel: every unit uses a
# **local** ``np.random.default_rng(seed)`` (no global RNG state) with a
# deterministic, order-independent seed, and the results are reassembled in a
# fixed order. Running them across processes therefore yields **byte-identical**
# output (see ``test_deterministic_build``). Processes (not threads) are used
# because the generator / solver do real Python-level work that holds the GIL.


def _resolve_build_workers(mix: Dict[str, Any], n_tasks: int) -> int:
    r"""Resolve the worker count for a build phase from ``mix.build_workers``.

    The environment variable ``SYNTH_BUILD_WORKERS`` overrides the config when
    set (e.g. ``SYNTH_BUILD_WORKERS=1`` forces serial builds). This is useful for
    debugging and for test runners where the OS process-spawn semantics (Windows
    ``spawn`` under pytest) make a process pool flaky; the production Linux box
    (``fork``) is unaffected and can leave it unset to use all cores.

    Args:
        mix: The ``benchmarks.G1_mix.mix`` sub-block. ``build_workers`` may be
            ``"auto"`` / ``0`` / ``None`` (use ``os.cpu_count() - 1``) or a
            positive int cap. Absent ⇒ ``"auto"``.
        n_tasks: Number of independent tasks in this phase. The result is capped
            at ``n_tasks`` so a 1-task phase always runs serially.

    Returns:
        The clamped worker count in ``[1, n_tasks]``. ``1`` means run serially.
    """
    raw = os.environ.get("SYNTH_BUILD_WORKERS")
    if raw is None or raw == "":
        raw = mix.get("build_workers", "auto")
    if raw in (None, 0, "auto", "Auto", "AUTO"):
        workers = max(1, (os.cpu_count() or 1) - 1)
    else:
        workers = int(raw)
    return max(1, min(workers, max(1, n_tasks)))


def _parallel_or_serial(
    fn: Callable[[Any], Any], arg_list: List[Any], workers: int
) -> List[Any]:
    r"""Map ``fn`` over ``arg_list``, preserving input order.

    Runs serially when ``workers <= 1`` or there is at most one task (so the tiny
    test builds and the monkeypatched unit tests never spawn a process pool);
    otherwise fans out over a :class:`ProcessPoolExecutor`. Results are returned
    in submission order regardless of completion order, which is what keeps the
    pooled output byte-identical to the serial build.

    Args:
        fn: A **module-level** (picklable) callable taking one picklable arg.
        arg_list: The per-task argument objects.
        workers: Worker count from :func:`_resolve_build_workers`.

    Returns:
        ``[fn(a) for a in arg_list]`` (computed in parallel when ``workers > 1``).
    """
    if workers <= 1 or len(arg_list) <= 1:
        return [fn(a) for a in arg_list]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, a) for a in arg_list]
        return [f.result() for f in futures]


@dataclass(frozen=True)
class MixCell:
    r"""One $(M, \mathrm{TE}, \text{lag-band})$ cell of the mixture grid.

    Attributes:
        cell_id: Stable index into the (kept) cell list, stamped into every
            sample's ``sample_cell_id``.
        M: Informative-channel count.
        target_te: Nominal cell block TE in nats (the grid value).
        band: Lag-band name (a key of ``mix.lag_bands``).
        delay_min: Lag-band floor $d_{\min}$.
        delay_max: Lag-band ceiling $d_{\max}$ (band $= \{0,\dots,d_{\max}-1\}$).
        per_channel_target: $\mathrm{TE}/M$, the inverter target.
        B_y_scalar: Uniform target loading solved for ``per_channel_target``.
        te_block_realised: Per-channel achieved mean block TE from the solve.
        te_cell_realised: Cell block TE $=$ ``te_block_realised`` $\times M$ --
            the canonical per-sample ``sample_te_true`` for this cell.
        band_idx: Lag-band index (enumeration order of ``mix.lag_bands``).
        held_out: ``True`` if this triple is in ``mix.holdout``.
    """

    cell_id: int
    M: int
    target_te: float
    band: str
    delay_min: int
    delay_max: int
    per_channel_target: float
    B_y_scalar: float
    te_block_realised: float
    te_cell_realised: float
    band_idx: int
    held_out: bool


def _mix_block(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the ``benchmarks.G1_mix.mix`` sub-block.

    Args:
        config: The parsed config (``benchmarks`` must carry ``G1_mix``).

    Returns:
        The ``mix`` sub-block dict.

    Raises:
        KeyError: If the ``G1_mix`` benchmark or its ``mix`` block is absent.
    """
    try:
        return config["benchmarks"][_BENCHMARK]["mix"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "mixed_dataset: config is missing benchmarks.G1_mix.mix "
            f"({exc})."
        ) from exc


def verify_holdout_marginals(
    holdout: List[Tuple[int, float, str]],
    m_grid: List[int],
    target_te_grid: List[float],
    band_names: List[str],
) -> None:
    r"""Assert every held-out triple is interior and leaves trained marginals.

    Each held-out value (its $M$, $\mathrm{TE}$ and band) must (i) be a valid
    grid axis value and (ii) still appear in at least one *trained* cell, so the
    model can interpolate to the held-out triple. Fails loud otherwise -- a bad
    holdout config that removes an entire marginal would silently turn the
    extrapolation test into an impossible one.

    Args:
        holdout: List of ``(M, target_te, band)`` triples to hold out.
        m_grid: The $M$ axis.
        target_te_grid: The block-TE axis.
        band_names: The lag-band names.

    Raises:
        ValueError: If any held-out value is off-grid, or if holding out the
            triples would remove every trained cell carrying some marginal
            value.
    """
    holdout_set = {(int(m), float(t), str(b)) for m, t, b in holdout}
    full = [
        (int(m), float(t), str(b))
        for m in m_grid for t in target_te_grid for b in band_names
    ]
    trained = [c for c in full if c not in holdout_set]
    for (m, t, b) in holdout_set:
        if m not in {int(x) for x in m_grid}:
            raise ValueError(f"holdout M={m} is not in m_grid={m_grid}.")
        if t not in {float(x) for x in target_te_grid}:
            raise ValueError(
                f"holdout target_te={t} is not in target_te_grid="
                f"{target_te_grid}."
            )
        if b not in set(band_names):
            raise ValueError(f"holdout band={b!r} is not in {band_names}.")
        if not any(c[0] == m for c in trained):
            raise ValueError(f"holdout removes every trained cell with M={m}.")
        if not any(c[1] == t for c in trained):
            raise ValueError(
                f"holdout removes every trained cell with target_te={t}."
            )
        if not any(c[2] == b for c in trained):
            raise ValueError(
                f"holdout removes every trained cell with band={b!r}."
            )


def _solve_cell_b_y(
    delay_min: int,
    delay_max: int,
    per_channel_target: float,
    *,
    oscillators: List[Tuple[float, float]],
    target_ar: float,
    sigma2_y: float,
    sigma2_eta: Any,
    horizon: int,
    K_history: Optional[int],
    inverter_cfg: Dict[str, Any],
    cache: Dict[Tuple[int, int, float], Dict[str, float]],
) -> Dict[str, float]:
    r"""Solve (memoised) the uniform $B_y$ for a per-channel block-TE target.

    The solve is **$M$-independent** -- it uses the single oscillator spec and
    targets the per-channel TE $\mathrm{TE}/M$ -- so it is memoised on
    ``(delay_min, delay_max, round(per_channel_target, 9))`` and reused across
    every $M$ that shares the same per-channel target and across all three
    splits.

    Args:
        delay_min, delay_max: The cell's lag band.
        per_channel_target: $\mathrm{TE}/M$ in nats.
        oscillators: Single-spec oscillator list ``[(r, omega)]``.
        target_ar, sigma2_y, sigma2_eta, horizon, K_history: G1 DGP params
            forwarded to the inverter.
        inverter_cfg: ``n_samples`` / ``lo`` / ``hi`` / ``tol`` / ``max_iter``.
        cache: Memoisation dict (mutated in place).

    Returns:
        ``{"B_y_scalar": float, "te_block": float}`` -- the per-channel solve.
    """
    key = (int(delay_min), int(delay_max), round(float(per_channel_target), 9))
    if key in cache:
        return cache[key]
    sol = B_y_for_mean_te_block_state_space(
        target_te_block=float(per_channel_target),
        delay_min=int(delay_min),
        delay_max=int(delay_max),
        oscillators=oscillators,
        target_ar=float(target_ar),
        sigma2_y=float(sigma2_y),
        sigma2_eta=sigma2_eta,
        H=int(horizon),
        K_history=K_history,
        n_samples=int(inverter_cfg.get("n_samples", 12_000)),
        lo=float(inverter_cfg.get("lo", 1e-4)),
        hi=float(inverter_cfg.get("hi", 10.0)),
        tol=float(inverter_cfg.get("tol", 5e-3)),
        max_iter=int(inverter_cfg.get("max_iter", 30)),
    )
    out = {"B_y_scalar": float(sol["B_y_scalar"]), "te_block": float(sol["te_block"])}
    cache[key] = out
    return out


def _solve_key_task(task: Dict[str, Any]) -> Dict[str, Any]:
    r"""Solve one distinct $(d_{\min}, d_{\max}, \mathrm{TE}/M)$ key (pool worker).

    Module-level + picklable so :func:`enumerate_mix_cells` can fan the distinct
    solves over a :class:`ProcessPoolExecutor`. Mirrors the inverter call inside
    :func:`_solve_cell_b_y` exactly, but catches the near-floor bracket miss and
    reports it (rather than memoising) so the assembly pass can drop / hard-fail
    the cell with the same message the serial loop used.

    Args:
        task: A dict carrying ``key`` (the ``(d_min, d_max, per_channel)`` tuple),
            ``delay_min`` / ``delay_max`` / ``per_channel_target`` and the shared
            DGP params (``oscillators`` / ``target_ar`` / ``sigma2_y`` /
            ``sigma2_eta`` / ``horizon`` / ``K_history`` / ``inverter_cfg``).

    Returns:
        ``{"key", "ok": True, "B_y_scalar", "te_block"}`` on success, else
        ``{"key", "ok": False, "reason": "solver bracket failed: ..."}``.
    """
    inverter_cfg = task["inverter_cfg"]
    try:
        sol = B_y_for_mean_te_block_state_space(
            target_te_block=float(task["per_channel_target"]),
            delay_min=int(task["delay_min"]),
            delay_max=int(task["delay_max"]),
            oscillators=task["oscillators"],
            target_ar=float(task["target_ar"]),
            sigma2_y=float(task["sigma2_y"]),
            sigma2_eta=task["sigma2_eta"],
            H=int(task["horizon"]),
            K_history=task["K_history"],
            n_samples=int(inverter_cfg.get("n_samples", 12_000)),
            lo=float(inverter_cfg.get("lo", 1e-4)),
            hi=float(inverter_cfg.get("hi", 10.0)),
            tol=float(inverter_cfg.get("tol", 5e-3)),
            max_iter=int(inverter_cfg.get("max_iter", 30)),
        )
    except ValueError as exc:  # bracket miss near the MC floor
        return {"key": task["key"], "ok": False,
                "reason": f"solver bracket failed: {exc}"}
    return {
        "key": task["key"], "ok": True,
        "B_y_scalar": float(sol["B_y_scalar"]),
        "te_block": float(sol["te_block"]),
    }


def enumerate_mix_cells(
    config: Dict[str, Any],
    *,
    workers: int = 1,
) -> Tuple[List[MixCell], List[Dict[str, Any]]]:
    r"""Enumerate, solve, and trim the mixture grid.

    Crosses ``mix.m_grid`` $\times$ ``mix.target_te_grid`` $\times$
    ``mix.lag_bands``. For each cell the per-channel target $\mathrm{TE}/M$ is
    checked against the per-band Monte-Carlo floor (cells at/below
    ``floor * te_floor_margin`` are trimmed) and, if reachable, the coupling
    $B_y$ is solved by the inverter. Held-out triples (``mix.holdout``) are
    flagged but kept (they form the extrapolation cache).

    The solves run in three passes so they can be parallelised without changing
    the result: (1) a cheap serial pass that decides per ``(m, t, band)`` whether
    the cell is floor-trimmed or needs a solve and collects the **distinct**
    ``(d_min, d_max, TE/M)`` solve keys (this replaces the in-loop memo); (2) a
    parallel pass that solves the distinct keys over a process pool; (3) a cheap
    serial pass that assembles ``(cells, dropped)`` in the original nested order
    with identical drop / held-out-error / message semantics.

    Args:
        config: The parsed config carrying ``benchmarks.G1_mix``.
        workers: Process-pool size for the distinct-key solves. ``1`` (default)
            runs serially -- which the monkeypatched unit tests rely on, since a
            spawned worker would import a fresh module without their patch.

    Returns:
        ``(cells, dropped)`` -- the kept :class:`MixCell` list (stable
        ``cell_id`` order) and a list of trimmed-cell dicts.

    Raises:
        ValueError: If a held-out triple is trimmed below the MC floor, or if
            :func:`verify_holdout_marginals` fails.
    """
    mix = _mix_block(config)
    data = config["data"]
    horizon = int(config["model"]["horizon"])

    m_grid = [int(m) for m in mix["m_grid"]]
    target_te_grid = [float(t) for t in mix["target_te_grid"]]
    lag_bands: Dict[str, List[int]] = {
        str(name): [int(rng[0]), int(rng[1])]
        for name, rng in mix["lag_bands"].items()
    }
    band_names = list(lag_bands.keys())
    band_idx = {name: i for i, name in enumerate(band_names)}
    holdout = [(int(m), float(t), str(b)) for m, t, b in mix.get("holdout", [])]
    holdout_set = set(holdout)
    verify_holdout_marginals(holdout, m_grid, target_te_grid, band_names)

    oscillators = [tuple(pair) for pair in data["oscillators"]]
    if len(oscillators) != 1:
        raise ValueError(
            "mixed_dataset: G1_mix expects a single oscillator spec (it is "
            f"tiled to each cell's M); got {len(oscillators)}."
        )
    target_ar = float(data["target_ar"])
    sigma2_y = float(data["sigma2_y"])
    sigma2_eta = data["sigma2_eta"]
    K_history = None if data.get("K_history") is None else int(data["K_history"])
    inverter_cfg = dict(mix.get("inverter", {}))
    te_n_samples = int(inverter_cfg.get("n_samples", data.get("te_n_samples", 12_000)))
    floor_margin = float(mix.get("te_floor_margin", 1.5))

    # Per-band MC bias floor (probed once at the solver's lower bracket). It is
    # M-independent (the solve always uses the single oscillator spec), so a
    # cell whose per-channel target TE/M lands at/below floor*margin is
    # unreachable by the bisection bracket and is trimmed.
    band_floor: Dict[str, float] = {}
    for name, (dmin, dmax) in lag_bands.items():
        band_floor[name] = float(mean_te_block_state_space_over_delays(
            delay_min=dmin, delay_max=dmax, oscillators=oscillators,
            target_ar=target_ar, B_y=1e-4, sigma2_y=sigma2_y,
            sigma2_eta=sigma2_eta, H=horizon, K_history=K_history,
            n_samples=te_n_samples,
        ))

    # Shared solve params (identical across keys -- the solve is M-independent).
    solve_common = dict(
        oscillators=oscillators, target_ar=target_ar, sigma2_y=sigma2_y,
        sigma2_eta=sigma2_eta, horizon=horizon, K_history=K_history,
        inverter_cfg=inverter_cfg,
    )

    # --- Pass 1 (serial, cheap): decide floor-trim vs needs-solve, in order, and
    # collect the distinct (d_min, d_max, TE/M) solve keys. ---------------------
    plan: List[Tuple[Any, ...]] = []
    distinct_tasks: Dict[Tuple[int, int, float], Dict[str, Any]] = {}
    for m in m_grid:
        for t_target in target_te_grid:
            for band in band_names:
                dmin, dmax = lag_bands[band]
                per_channel = float(t_target) / int(m)
                min_per_channel = band_floor[band] * floor_margin
                held = (int(m), float(t_target), band) in holdout_set
                meta = {
                    "M": int(m), "target_te": float(t_target), "band": band,
                    "dmin": dmin, "dmax": dmax,
                    "per_channel": per_channel, "held": held,
                }
                if float(t_target) == 0.0:
                    # Zero-coupling NULL cell: $B_y = 0$, true block TE $= 0$
                    # exactly. Bypass the MC-floor trim AND the bisection inverter
                    # (which cannot bracket a target of $0$) -- the assembly pass
                    # fabricates the $B_y = 0$ solve directly. A null must be
                    # trained (it anchors the calibration intercept $\alpha$), so a
                    # held-out TE$=0$ triple is a hard error.
                    if held:
                        raise ValueError(
                            f"zero-coupling null cell (M={m}, TE=0.0, "
                            f"band={band}) must be trained, not held out -- "
                            f"remove it from mix.holdout."
                        )
                    plan.append(("zero", meta))
                elif per_channel <= min_per_channel:
                    meta["reason"] = (
                        f"per-channel target {per_channel:.4g} <= MC floor*"
                        f"margin {min_per_channel:.4g}"
                    )
                    plan.append(("floor", meta))
                else:
                    key = (int(dmin), int(dmax), round(float(per_channel), 9))
                    if key not in distinct_tasks:
                        distinct_tasks[key] = {
                            "key": key, "delay_min": dmin, "delay_max": dmax,
                            "per_channel_target": per_channel, **solve_common,
                        }
                    plan.append(("solve", key, meta))

    # --- Pass 2 (parallel): solve the distinct keys over a process pool. --------
    key_order = list(distinct_tasks.keys())
    solve_results = _parallel_or_serial(
        _solve_key_task, [distinct_tasks[k] for k in key_order], workers
    )
    solved: Dict[Tuple[int, int, float], Dict[str, Any]] = {
        r["key"]: r for r in solve_results
    }

    # --- Pass 3 (serial, cheap): assemble in the original nested order. ---------
    cells: List[MixCell] = []
    dropped: List[Dict[str, Any]] = []
    next_id = 0
    for entry in plan:
        if entry[0] == "floor":
            meta = entry[1]
            reason: Optional[str] = meta["reason"]
            res = None
        elif entry[0] == "zero":
            # Synthetic solve for the zero-coupling null cell (no inverter call).
            meta = entry[1]
            reason = None
            res = {"ok": True, "B_y_scalar": 0.0, "te_block": 0.0}
        else:
            _, key, meta = entry
            res = solved[key]
            reason = None if res["ok"] else res["reason"]
        if reason is not None:
            if meta["held"]:
                raise ValueError(
                    f"held-out cell (M={meta['M']}, TE={meta['target_te']}, "
                    f"band={meta['band']}) is unreachable: {reason}. Pick a "
                    f"held-out triple above the MC floor."
                )
            dropped.append({
                "M": meta["M"], "target_te": meta["target_te"],
                "band": meta["band"], "per_channel_target": meta["per_channel"],
                "floor": band_floor[meta["band"]], "reason": reason,
            })
            print(
                f"[mix] drop cell M={meta['M']} TE={meta['target_te']:g} "
                f"band={meta['band']}: {reason}"
            )
            continue
        assert res is not None  # a non-floor, non-failed entry has a solve
        te_block = res["te_block"]
        cells.append(MixCell(
            cell_id=next_id, M=meta["M"], target_te=meta["target_te"],
            band=meta["band"], delay_min=meta["dmin"], delay_max=meta["dmax"],
            per_channel_target=meta["per_channel"],
            B_y_scalar=res["B_y_scalar"],
            te_block_realised=te_block,
            te_cell_realised=te_block * meta["M"],
            band_idx=band_idx[meta["band"]], held_out=meta["held"],
        ))
        next_id += 1
    return cells, dropped


def _gen_cell_split(
    cell: MixCell,
    n: int,
    seed: int,
    *,
    data: Dict[str, Any],
    c_y: int,
    c_u: int,
    horizon: int,
    te_n_samples: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    r"""Generate one $(\text{cell}, \text{split})$ via the G1 generator.

    Tiles the single oscillator spec / coupling to the cell's $M$ channels and
    resolves the channel decomposition at $M$ (reusing
    :func:`build_dataset._resolve_channel_decomp`).

    The generator also estimates a realised ``meta.te_true`` by Monte Carlo, but
    the mixed cache **discards** it -- each sample's ``sample_te_true`` is the
    inverter-solved ``cell.te_cell_realised`` (a single canonical TE per cell,
    consistent across splits). The simulated sample data does **not** depend on
    ``te_n_samples`` (only the discarded TE estimate does), so a small budget
    here is purely a build-time saving.

    Args:
        cell: The cell to generate.
        n: Number of samples for this split.
        seed: Generator RNG seed (distinct per cell and split).
        data: The ``G1_mix`` data block (DGP constants + ``channel_decomp``).
        c_y, c_u: Native channel counts.
        horizon: Forecast horizon $H$.
        te_n_samples: MC budget for the generator's (discarded) ``meta.te_true``.

    Returns:
        ``(Y, U, meta)`` -- ``Y`` $(n, T, c_y)$, ``U`` $(n, T, c_u)$ numpy
        ``float32`` and the generator ``meta`` dict (carrying ``true_lag_tt``).
    """
    decomp = _resolve_channel_decomp({**data, "M": cell.M}, c_y, c_u, "G1")
    Y, U, meta = gen_state_space_oscillator(
        n=n,
        T=int(data["sequence_length"]),
        oscillators=[tuple(data["oscillators"][0])] * cell.M,
        target_ar=float(data["target_ar"]),
        B_y=[cell.B_y_scalar] * cell.M,
        sigma2_y=float(data["sigma2_y"]),
        sigma2_eta=data["sigma2_eta"],
        M=cell.M,
        delay_min=cell.delay_min,
        delay_max=cell.delay_max,
        delay_walk=bool(data.get("delay_walk", True)),
        delay_walk_step_prob=float(data.get("delay_walk_step_prob", 0.02)),
        c_y=c_y,
        c_u=c_u,
        horizon=horizon,
        K_history=None if data.get("K_history") is None else int(data["K_history"]),
        standardize=bool(data.get("standardize", True)),
        randomize_channel_layout=bool(data.get("randomize_channel_layout", False)),
        seed=seed,
        te_n_samples=te_n_samples,
        channel_decomp=decomp,
    )
    return Y.numpy(), U.numpy(), meta


def _gen_cell_task(
    task: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    r"""Generate one $(\text{cell}, \text{split})$ (pool worker).

    Module-level + picklable wrapper around :func:`_gen_cell_split` so
    :func:`build_mixed_split` can fan the per-cell generation over a
    :class:`ProcessPoolExecutor`. The cell ``seed`` is computed in the parent
    (deterministic, order-independent) and passed in, so the output is identical
    to the serial build regardless of worker scheduling.

    Args:
        task: A dict with ``cell`` / ``n`` / ``seed`` and the shared
            ``data`` / ``c_y`` / ``c_u`` / ``horizon`` / ``te_n_samples`` args.

    Returns:
        ``(Y, U, meta)`` as returned by :func:`_gen_cell_split`.
    """
    return _gen_cell_split(
        task["cell"], task["n"], task["seed"],
        data=task["data"], c_y=task["c_y"], c_u=task["c_u"],
        horizon=task["horizon"], te_n_samples=task["te_n_samples"],
    )


def build_mixed_split(
    cells: List[MixCell],
    split: str,
    n_per_cell: int,
    *,
    data: Dict[str, Any],
    c_y: int,
    c_u: int,
    horizon: int,
    te_n_samples: int,
    base_seed: int,
    held_out_cache: bool,
    split_channels: Dict[str, int],
    workers: int = 1,
) -> Dict[str, np.ndarray]:
    r"""Generate every cell's split, concatenate, and shuffle the pool.

    Args:
        cells: The cells to generate for this split.
        split: Split name (``train`` / ``val`` / ``test``).
        n_per_cell: Samples per cell.
        data: The ``G1_mix`` data block.
        c_y, c_u: Native channel counts.
        horizon: Forecast horizon.
        te_n_samples: Generator MC budget.
        base_seed: Mixture base seed (``mix.build_seed``).
        held_out_cache: Stamped into every ``sample_held_out`` of this cache.
        split_channels: ``{c_y_st, c_y_ph, c_u_st, c_u_ph}``.
        workers: Process-pool size for the per-cell generation. ``1`` (default)
            runs serially. Each cell's seed is computed here (deterministic,
            order-independent), so any worker count yields identical output.

    Returns:
        A dict of stacked, shuffled arrays: the five native fields, optional
        ``true_lag_tt``, and the seven ``sample_*`` provenance arrays.
    """
    c_y_st = split_channels["c_y_st"]
    c_y_ph = split_channels["c_y_ph"]
    c_u_st = split_channels["c_u_st"]
    c_u_ph = split_channels["c_u_ph"]
    T = int(data["sequence_length"])
    split_off = _SPLIT_OFFSET[split]

    # Generate every cell (parallel over a process pool when workers > 1), then
    # assemble in cell order so the pooled output is byte-identical to serial.
    gen_tasks = [
        {
            "cell": cell, "n": n_per_cell,
            "seed": int(base_seed) + split_off * 100_003 + cell.cell_id * 101,
            "data": data, "c_y": c_y, "c_u": c_u,
            "horizon": horizon, "te_n_samples": te_n_samples,
        }
        for cell in cells
    ]
    gen_results = _parallel_or_serial(_gen_cell_task, gen_tasks, workers)

    fhr_st, fhr_ph, up_st, up_ph = [], [], [], []
    lag_tt: List[np.ndarray] = []
    s_te, s_M, s_dmin, s_dmax, s_band, s_cell = [], [], [], [], [], []
    have_lag_tt = True
    for cell, (Y_np, U_np, meta) in zip(cells, gen_results):
        fhr_st.append(np.ascontiguousarray(Y_np[..., :c_y_st]))
        fhr_ph.append(np.ascontiguousarray(Y_np[..., c_y_st:c_y_st + c_y_ph]))
        up_st.append(np.ascontiguousarray(U_np[..., :c_u_st]))
        up_ph.append(np.ascontiguousarray(U_np[..., c_u_st:c_u_st + c_u_ph]))
        tlt = meta.get("true_lag_tt")
        if tlt is None:
            have_lag_tt = False
        else:
            lag_tt.append(np.asarray(tlt, dtype=np.int16))
        s_te.append(np.full(n_per_cell, cell.te_cell_realised, dtype=np.float32))
        s_M.append(np.full(n_per_cell, cell.M, dtype=np.int16))
        s_dmin.append(np.full(n_per_cell, cell.delay_min, dtype=np.int16))
        s_dmax.append(np.full(n_per_cell, cell.delay_max, dtype=np.int16))
        s_band.append(np.full(n_per_cell, cell.band_idx, dtype=np.int8))
        s_cell.append(np.full(n_per_cell, cell.cell_id, dtype=np.int16))

    arrays: Dict[str, np.ndarray] = {
        "fhr_st": np.concatenate(fhr_st, axis=0),
        "fhr_ph": np.concatenate(fhr_ph, axis=0),
        "up_st": np.concatenate(up_st, axis=0),
        "up_ph": np.concatenate(up_ph, axis=0),
        "sample_te_true": np.concatenate(s_te, axis=0),
        "sample_M": np.concatenate(s_M, axis=0),
        "sample_delay_min": np.concatenate(s_dmin, axis=0),
        "sample_delay_max": np.concatenate(s_dmax, axis=0),
        "sample_band_id": np.concatenate(s_band, axis=0),
        "sample_cell_id": np.concatenate(s_cell, axis=0),
    }
    n_total = arrays["fhr_st"].shape[0]
    arrays["weight"] = np.ones((n_total, T), dtype=np.float32)
    arrays["sample_held_out"] = np.full(
        n_total, 1 if held_out_cache else 0, dtype=np.int8
    )
    if have_lag_tt and lag_tt:
        arrays["true_lag_tt"] = np.concatenate(lag_tt, axis=0)

    # One shared permutation so every array stays row-aligned; mixing cells in
    # the pooled order also helps any shuffle=False consumer (val / test).
    rng = np.random.default_rng(int(base_seed) + split_off)
    perm = rng.permutation(n_total)
    for key in list(arrays.keys()):
        arrays[key] = np.ascontiguousarray(arrays[key][perm])
    return arrays


def _manifest(
    cells: List[MixCell],
    *,
    n_per_cell: Dict[str, int],
    held_out_cache: bool,
    mix: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the mixture manifest stored under ``meta['mixture']``.

    Args:
        cells: The cells included in this cache.
        n_per_cell: ``{split: n}`` sample counts per cell.
        held_out_cache: Whether this is the held-out (test-only) cache.
        mix: The ``mix`` config sub-block.

    Returns:
        A JSON-serialisable manifest dict.
    """
    return {
        "held_out_cache": bool(held_out_cache),
        "standardize_mode": "per_cell",
        "n_per_cell": dict(n_per_cell),
        "m_grid": [int(m) for m in mix["m_grid"]],
        "target_te_grid": [float(t) for t in mix["target_te_grid"]],
        "lag_bands": {str(k): [int(v[0]), int(v[1])] for k, v in mix["lag_bands"].items()},
        "holdout": [[int(m), float(t), str(b)] for m, t, b in mix.get("holdout", [])],
        "cells": [asdict(c) for c in cells],
    }


def write_mixed_cache(
    out_dir: Path,
    splits: Dict[str, Dict[str, np.ndarray]],
    *,
    cells: List[MixCell],
    tag: str,
    held_out_cache: bool,
    data: Dict[str, Any],
    model: Dict[str, Any],
    mix: Dict[str, Any],
    split_channels: Dict[str, int],
    representative_decomp: Dict[str, Any],
    channel_layout: Dict[str, Any],
) -> None:
    r"""Write the split ``.npz`` files and the shared ``meta.json``.

    The top-level ``te_true`` is the (sample-count-weighted) pooled mean of the
    per-cell TEs and ``true_lag_band`` is the union $\{0,\dots,\max d_{\max}-1\}$
    -- both split-invariant scalars that :class:`dataset.SyntheticTEDataset`
    reads at init. The authoritative per-cell ground truth lives in
    ``meta['mixture']`` and per sample in the ``sample_*`` arrays.

    Args:
        out_dir: Destination cache directory.
        splits: ``{split: arrays}`` from :func:`build_mixed_split`.
        cells: The cells in this cache.
        tag: Cache tag (subdirectory name).
        held_out_cache: Whether this is the held-out cache.
        data, model, mix: Config blocks.
        split_channels: Native channel counts.
        representative_decomp: ``channel_decomp`` at $\max M$ (conservative
            informative union for visualisation only).
        channel_layout: Matching ``channel_layout`` at $\max M$.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, arrays in splits.items():
        np.savez(str(out_dir / f"{split}.npz"), **arrays)
        print(f"  [{split:5s}] n={arrays['fhr_st'].shape[0]:6d}  ->  {split}.npz")

    T = int(data["sequence_length"])
    horizon = int(model["horizon"])
    max_delay_max = max(c.delay_max for c in cells)
    # Pooled mean TE weighted by per-cell sample counts (equal here, so a plain
    # mean over cells -- kept explicit in case n_per_cell becomes per-cell).
    te_pooled = float(np.mean([c.te_cell_realised for c in cells])) if cells else 0.0
    n_per_cell = {
        "train": int(mix.get("n_per_cell_train", 0)),
        "val": int(mix.get("n_per_cell_val", 0)),
        "test": int(mix.get("n_per_cell_test", 0)),
    }
    meta: Dict[str, Any] = {
        "benchmark": _BENCHMARK,
        "tag": tag,
        "te_true": te_pooled,
        "te_per_step": te_pooled / horizon,
        "true_lag_band": list(range(int(max_delay_max))),
        "horizon": horizon,
        "sequence_length": T,
        "clean_anchor_range": [int(max_delay_max) - 1, T - horizon],
        "c_y": split_channels["c_y_st"] + split_channels["c_y_ph"],
        "c_u": split_channels["c_u_st"] + split_channels["c_u_ph"],
        "channel_map": {
            "fhr_st": [0, split_channels["c_y_st"]],
            "fhr_ph": [split_channels["c_y_st"],
                       split_channels["c_y_st"] + split_channels["c_y_ph"]],
            "up_st": [0, split_channels["c_u_st"]],
            "up_ph": [split_channels["c_u_st"],
                      split_channels["c_u_st"] + split_channels["c_u_ph"]],
        },
        "channel_decomp": representative_decomp,
        "channel_layout": channel_layout,
        "mixture": _manifest(
            cells, n_per_cell=n_per_cell, held_out_cache=held_out_cache, mix=mix,
        ),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    band_hi = max_delay_max - 1
    print(
        f"  [meta ] te_true(pooled)={te_pooled:.4f} nats  "
        f"lag_band=0..{band_hi}  cells={len(cells)}  ->  meta.json"
    )


def build_g1_mix(
    config: Dict[str, Any], *, force: bool = False, holdout: bool = False,
    extrap_m: Optional[int] = None,
) -> Path:
    r"""Build the in-mix, held-out, or **M-extrapolation** ``G1_mix`` cache.

    Three test caches share this builder:

    * ``holdout=False, extrap_m=None`` -- the in-mix pool (all three splits from
      the trained cells), ``data/G1_mix/<tag>/``.
    * ``holdout=True`` -- the test-only held-out cache (the held-out cells, an
      *interpolation* extrapolation test), ``data/G1_mix/<tag>_holdout/``.
    * ``extrap_m=<M>`` -- a test-only cache at an informative-channel count $M$
      **outside** the trained ``m_grid`` (a genuine extrapolation across the
      channel-dilution axis), ``data/G1_mix/<tag>_extrap_m<M>/``. The grid is
      rebuilt at the single $M$ with no holdout, and every sample is stamped
      ``sample_held_out=1`` so :mod:`mixed_eval` scores it on the in-mix
      calibration exactly like the held-out cache.

    Args:
        config: The parsed config carrying ``benchmarks.G1_mix``. The active
            ``experiment.tag`` names the base cache.
        force: Regenerate even when a complete cache already exists.
        holdout: Build the test-only held-out cache from the held-out cells.
        extrap_m: If set, build the test-only $M$-extrapolation cache at this
            (untrained) $M$. Mutually exclusive with ``holdout``.

    Returns:
        The cache directory.

    Raises:
        ValueError: On channel-count mismatch, an unreachable held-out cell, or
            ``holdout`` and ``extrap_m`` both set.
    """
    if holdout and extrap_m is not None:
        raise ValueError("build_g1_mix: pass at most one of holdout / extrap_m.")
    if extrap_m is not None:
        # M-extrapolation: rebuild the grid at a single untrained M, no holdout.
        config = deepcopy(config)
        _mix_block(config)["m_grid"] = [int(extrap_m)]
        _mix_block(config)["holdout"] = []
    # Test-only caches: the held-out cache and every extrapolation cache.
    test_only = holdout or (extrap_m is not None)

    exp = config["experiment"]
    data = config["data"]
    model = config["model"]
    mix = _mix_block(config)
    base_tag = str(exp["tag"])
    if extrap_m is not None:
        tag = f"{base_tag}_extrap_m{int(extrap_m)}"
    else:
        tag = base_tag + _HOLDOUT_SUFFIX if holdout else base_tag

    c_y_st, c_y_ph = int(data["c_y_st"]), int(data["c_y_ph"])
    c_u_st, c_u_ph = int(data["c_u_st"]), int(data["c_u_ph"])
    c_y, c_u = c_y_st + c_y_ph, c_u_st + c_u_ph
    if c_y != int(model["c_y"]) or c_u != int(model["c_u"]):
        raise ValueError(
            f"channel mismatch: data gives c_y={c_y}, c_u={c_u} but model "
            f"gives c_y={model['c_y']}, c_u={model['c_u']}."
        )
    split_channels = {
        "c_y_st": c_y_st, "c_y_ph": c_y_ph, "c_u_st": c_u_st, "c_u_ph": c_u_ph,
    }

    data_root = resolve_user_path(config["paths"]["data_dir"])
    out_dir = data_root / _BENCHMARK / tag
    # Test-only caches (holdout / extrapolation) never carry train/val splits,
    # so requiring the full _SPLIT_FILES would force a rebuild on every run.
    required = ("test.npz", "meta.json") if test_only else _SPLIT_FILES
    if not force and all((out_dir / f).is_file() for f in required):
        print(f"cache exists, skipping: {out_dir}  (use --force to rebuild)")
        return out_dir

    horizon = int(model["horizon"])
    # Generation MC budget for the generator's *discarded* meta.te_true. The
    # sample data is independent of it (see ``_gen_cell_split``), so a small
    # value is a pure build-time saving; the accurate per-cell TE comes from the
    # inverter (``mix.inverter.n_samples``) via ``te_cell_realised``.
    gen_te_n_samples = int(mix.get("gen_te_n_samples", data.get("te_n_samples", 2_000)))
    base_seed = int(mix.get("build_seed", exp.get("seed", 0)))

    print(f"building {_BENCHMARK} (tag '{tag}', holdout={holdout}) -> {out_dir}")
    # Worker count for the distinct-key solves (upper-bounded by the full grid;
    # ``_parallel_or_serial`` / ``_resolve_build_workers`` clamp to the real task
    # count, so a 1-cell grid or ``build_workers: 1`` stays serial).
    n_grid = (
        len(mix["m_grid"]) * len(mix["target_te_grid"]) * len(mix["lag_bands"])
    )
    solve_workers = _resolve_build_workers(mix, n_grid)
    print(f"[mix] enumerating + solving cells (workers={solve_workers}) ...")
    all_cells, dropped = enumerate_mix_cells(config, workers=solve_workers)
    cells = [c for c in all_cells if c.held_out == holdout]
    # Re-id so cell_id is contiguous within this cache (manifest + provenance
    # stay self-consistent; cross-cache identity is the (M, TE, band) triple).
    cells = [_reindex(c, i) for i, c in enumerate(cells)]
    if not cells:
        raise ValueError(
            f"{_BENCHMARK}: no {'held-out' if holdout else 'trained'} cells to "
            f"build (dropped {len(dropped)} below the MC floor)."
        )
    print(
        f"[mix] {len(cells)} cells "
        f"({'held-out' if holdout else 'trained'}), {len(dropped)} dropped."
    )

    # Conservative informative union (for visualisation only): the decomposition
    # at max M so the 'te' index block spans every cell's informative channels.
    max_M = max(c.M for c in cells)
    rep_decomp = _resolve_channel_decomp({**data, "M": max_M}, c_y, c_u, "G1")

    if test_only:
        split_sizes = {"test": int(mix.get("n_per_cell_test", 0))}
    else:
        split_sizes = {
            "train": int(mix.get("n_per_cell_train", 0)),
            "val": int(mix.get("n_per_cell_val", 0)),
            "test": int(mix.get("n_per_cell_test", 0)),
        }
    gen_workers = _resolve_build_workers(mix, len(cells))
    splits: Dict[str, Dict[str, np.ndarray]] = {}
    for split, n_pc in split_sizes.items():
        if n_pc <= 0:
            continue
        print(
            f"[mix] generating split '{split}' "
            f"(n_per_cell={n_pc}, workers={gen_workers}) ..."
        )
        splits[split] = build_mixed_split(
            cells, split, n_pc, data=data, c_y=c_y, c_u=c_u, horizon=horizon,
            te_n_samples=gen_te_n_samples, base_seed=base_seed,
            held_out_cache=test_only, split_channels=split_channels,
            workers=gen_workers,
        )

    write_mixed_cache(
        out_dir, splits, cells=cells, tag=tag, held_out_cache=test_only,
        data=data, model=model, mix=mix, split_channels=split_channels,
        representative_decomp=_decomp_to_json(rep_decomp),
        channel_layout=_make_channel_layout(rep_decomp, c_y, c_u),
    )
    print(f"done: {out_dir}")
    return out_dir


def _reindex(cell: MixCell, new_id: int) -> MixCell:
    """Return a copy of ``cell`` with ``cell_id`` replaced by ``new_id``."""
    return MixCell(**{**asdict(cell), "cell_id": int(new_id)})


def _apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Apply CLI / in-file overrides in place (``tag`` + path overrides).

    Args:
        config: The parsed config (mutated in place).
        overrides: Mapping with optional ``tag`` / ``data_dir`` / ``results_dir``.
    """
    if overrides.get("tag") is not None:
        config["experiment"]["tag"] = overrides["tag"]
    apply_path_overrides(config, overrides)


def main() -> None:
    """CLI entry point: parse arguments, load config, build the mixed cache."""
    parser = argparse.ArgumentParser(
        description="Build the G1_mix mixed-population synthetic cache."
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                        help="path to config_synth.yaml")
    parser.add_argument("--tag", type=str, default=None,
                        help="override experiment.tag (cache subdir name)")
    parser.add_argument("--holdout", action="store_true",
                        help="build the test-only held-out extrapolation cache")
    parser.add_argument("--extrap-m", type=int, default=None, dest="extrap_m",
                        help="build a test-only M-extrapolation cache at this "
                             "(untrained) informative-channel count M; with "
                             "--extrap-m all builds every mix.holdout_m value")
    parser.add_argument("--extrap-m-all", action="store_true", dest="extrap_m_all",
                        help="build every mix.holdout_m extrapolation cache")
    parser.add_argument("--force", action="store_true",
                        help="regenerate even if a complete cache exists")
    parser.add_argument("--data-dir", type=str, default=None, dest="data_dir",
                        help="override paths.data_dir")
    parser.add_argument("--results-dir", type=str, default=None,
                        dest="results_dir", help="override paths.results_dir")
    args = parser.parse_args()

    config = load_config(args.config)
    config["experiment"]["benchmark"] = _BENCHMARK
    # Re-resolve so config['data'] / loss reflect the G1_mix block.
    resolve_active_benchmark(config)
    _apply_overrides(config, vars(args))
    if args.extrap_m_all:
        for m_val in _mix_block(config).get("holdout_m", []):
            build_g1_mix(config, force=args.force, extrap_m=int(m_val))
    elif args.extrap_m is not None:
        build_g1_mix(config, force=args.force, extrap_m=int(args.extrap_m))
    else:
        build_g1_mix(config, force=args.force, holdout=args.holdout)


if __name__ == "__main__":
    # ----- dual-mode launch (Decision V2-D8) ---------------------------------
    # CLI mode when any --flag is present; otherwise the RUN_CONFIG dict below.
    CONFIG_PATH = _DEFAULT_CONFIG
    RUN_CONFIG = {
        "tag": "G1_mix_base",   # cache subdir under data/G1_mix/
        "holdout": False,       # True -> build the held-out test-only cache
        "extrap_m": None,       # int -> build the M-extrapolation cache at this M
        "extrap_m_all": False,  # True -> build every mix.holdout_m extrap cache
        "force": False,         # True -> rebuild even if a complete cache exists
        "data_dir": None,       # None -> config paths.data_dir
        "results_dir": None,    # None -> config paths.results_dir
    }

    if len(sys.argv) > 1:
        main()
    else:
        config = load_config(CONFIG_PATH)
        config["experiment"]["benchmark"] = _BENCHMARK
        resolve_active_benchmark(config)
        _apply_overrides(config, RUN_CONFIG)
        if RUN_CONFIG["extrap_m_all"]:
            for m_val in _mix_block(config).get("holdout_m", []):
                build_g1_mix(config, force=RUN_CONFIG["force"], extrap_m=int(m_val))
        elif RUN_CONFIG["extrap_m"] is not None:
            build_g1_mix(config, force=RUN_CONFIG["force"],
                         extrap_m=int(RUN_CONFIG["extrap_m"]))
        else:
            build_g1_mix(config, force=RUN_CONFIG["force"],
                         holdout=RUN_CONFIG["holdout"])
