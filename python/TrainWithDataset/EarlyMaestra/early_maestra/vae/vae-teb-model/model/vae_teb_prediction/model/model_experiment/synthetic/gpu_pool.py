r"""Task-parallel multi-GPU training scheduler for the synthetic TE sweeps.

The synthetic experiment trains many **independent** models -- the A-sweep
(``a_grid`` x ``m_grid``, ~21 cells), the $\beta$ rate-distortion sweep
(~9 cells), the hyper-parameter probes, the Benchmark-B $\rho$-null diagnostic
and the directionality pair. Because the cells are independent, the cheapest
and safest way to use a multi-GPU box is **task parallelism**: run one model per
GPU and dispatch cells to whichever GPU is free -- no ``DistributedDataParallel``,
no change to the training / loss / KL code.

How it works:
    * The main process enumerates the cells for the requested ``mode`` and
      builds every needed dataset **once, serially** (so workers never race on
      :func:`build_dataset.build_dataset`).
    * It then maintains a pool of at most ``len(gpus)`` worker subprocesses.
      Each worker is this same module re-invoked with ``--worker``; it is
      pinned to one physical GPU via ``CUDA_VISIBLE_DEVICES`` and trains exactly
      one cell by calling :func:`train_minimal.train`.
    * Cell run-tags reuse the *exact* tag helpers of :mod:`evaluate_te`,
      :mod:`beta_sweep` and :mod:`directionality`, so every checkpoint lands
      where the matching evaluation harness already looks.

This scheduler is **train-only** -- it deliberately does not evaluate. After it
finishes, run the normal evaluation harness (``evaluate_te`` / ``beta_sweep`` /
``lag_recovery`` / ``directionality`` / ``final_report``); every checkpoint is
already on disk so they each just score what exists. Cells whose ``final.ckpt``
already exists are skipped unless ``--force`` is passed, so an interrupted run
resumes cleanly.

Run modes (project convention -- Decision D9 in
``synthetic_te_validation_plan.md``): this file supports **both** a CLI and an
edit-and-run ``__main__``, auto-detected from whether any command-line argument
is present.

    * CLI mode (any ``--flag`` passed)::

        python -m ...synthetic.gpu_pool --mode a_sweep --gpus 0,1,2,3,4,5,6
        python -m ...synthetic.gpu_pool --mode beta   --gpus 0,1,2,3,4,5,6
        python -m ...synthetic.gpu_pool --mode hp --axis lambda_base --gpus 0,1
        python -m ...synthetic.gpu_pool --mode rho_null --gpus 0,1,2,3
        python -m ...synthetic.gpu_pool --mode directionality --gpus 0,1
        [--config PATH] [--benchmark B] [--epochs N] [--seed S]
        [--no-build] [--force] [--dry-run]

    * Edit-and-run mode (no arguments) -- edit the ``RUN_CONFIG`` dict in the
      ``__main__`` block, then run the file directly (IDE / notebook)::

        python -m ...synthetic.gpu_pool

The ``--gpus`` list is a *slot* list: repeating an index (``0,0,1,1``) packs two
concurrent models onto that GPU -- useful when the GPUs are large enough to hold
two models at once.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from model.vae_teb_prediction.model.model_experiment.synthetic import (
    beta_sweep as bs,
)
from model.vae_teb_prediction.model.model_experiment.synthetic import (
    build_dataset as bd,
)
from model.vae_teb_prediction.model.model_experiment.synthetic import (
    directionality as dr,
)
from model.vae_teb_prediction.model.model_experiment.synthetic import (
    evaluate_te as ev,
)
from model.vae_teb_prediction.model.model_experiment.synthetic import (
    train_minimal as tm,
)

# ``synthetic/`` package dir, its parent ``model_experiment/`` and the repo
# root (``teb_vae_model/``) -- the worker subprocess runs ``python -m`` from the
# repo root so the ``model.*`` package chain resolves.
_PKG_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _PKG_DIR.parent
_REPO_ROOT = _PKG_DIR.parents[4]
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"
_MODULE = "model.vae_teb_prediction.model.model_experiment.synthetic.gpu_pool"

# Seconds between poll sweeps of the running-worker set.
_POLL_SECONDS = 5.0

_VALID_MODES = ("a_sweep", "beta", "hp", "rho_null", "directionality")

# Per-cell summary CSV columns (one row per training cell).
_SUMMARY_FIELDS = [
    "mode", "label", "benchmark", "data_tag", "run_tag",
    "gpu", "status", "returncode", "seconds", "ckpt", "log",
]


@dataclass
class TrainCell:
    """One independent training job dispatched to a single GPU.

    Attributes:
        benchmark: Benchmark identifier (``A`` / ``B`` / ``C`` / ``E`` / ``G``)
            the cell trains under.
        data_tag: Cache tag of the dataset to train on
            (``data/<benchmark>/<data_tag>/``).
        run_tag: Results subdirectory for this cell
            (``results/<benchmark>/<run_tag>/``). Reuses the exact tag the
            matching evaluation harness expects.
        label: Short human-readable description for console logs.
        patches: Dotted-path config overrides applied by the worker before
            training, e.g. ``{"loss.kld_beta": 3e-4, "optim.epochs": 60}``.
    """

    benchmark: str
    data_tag: str
    run_tag: str
    label: str
    patches: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Cell enumeration (one function per mode)
# =============================================================================

def _cells_a_sweep(config: Dict[str, Any], build: bool) -> List[TrainCell]:
    """Enumerate the A-sweep (``a_grid`` x ``m_grid``) training cells.

    Reuses :func:`evaluate_te.enumerate_sweep` and the ``_setting_tags`` helpers
    so the trained checkpoints land where :func:`evaluate_te.run_sweep` looks.
    Handles the Gaussian (A / B), XOR (C) and two-lag (E) sweep kinds.

    Args:
        config: The parsed, benchmark-resolved config.
        build: If True, generate each cell's dataset (idempotent).

    Returns:
        One :class:`TrainCell` per sweep setting.
    """
    benchmark = str(config["experiment"]["benchmark"])
    cells: List[TrainCell] = []
    for setting in ev.enumerate_sweep(config):
        kind = str(setting.get("kind", "gaussian"))
        m = int(setting["M"])
        if kind == "xor":
            q = float(setting["q"])
            data_tag, run_tag = ev._setting_tags_xor(benchmark, q, m)
            label = f"q={q:g} M={m}"
            if build:
                ev._ensure_dataset_xor(config, data_tag, q, m)
        elif kind == "two_lag":
            data_tag = str(config["experiment"]["tag"])
            run_tag = "sweep_two_lag"
            label = f"two-lag M={m}"
            if build:
                ev._ensure_dataset_simple(config, data_tag)
        else:  # gaussian
            a = float(setting["a"])
            data_tag, run_tag = ev._setting_tags(benchmark, a, m)
            label = f"a={a:g} M={m}"
            if build:
                ev._ensure_dataset(config, data_tag, a, m)
        cells.append(TrainCell(benchmark, data_tag, run_tag, label))
    return cells


def _cells_beta(config: Dict[str, Any], build: bool) -> List[TrainCell]:
    """Enumerate the $\\beta$ rate-distortion sweep cells.

    Every cell trains the same fixed $(a, M)$ dataset with a different
    ``loss.kld_beta``; run-tags reuse :func:`beta_sweep._cell_tags` so
    :func:`beta_sweep.run_sweep` finds the checkpoints.

    Args:
        config: The parsed config.
        build: If True, generate the fixed dataset (idempotent).

    Returns:
        One :class:`TrainCell` per $\\beta$ in ``beta_sweep.grid``.
    """
    benchmark = str(config["experiment"]["benchmark"])
    bsconf = config.get("beta_sweep", {}) or {}
    grid = [float(v) for v in (bsconf.get("grid") or [])]
    if not grid:
        raise ValueError("beta mode needs a populated beta_sweep.grid")
    data_tag = bs._fixed_data_tag(config)
    if build:
        bs._ensure_fixed_dataset(config, data_tag, build_missing=True)
    epochs = bsconf.get("epochs")
    cells: List[TrainCell] = []
    for value in grid:
        _, run_tag = bs._cell_tags("beta", value, "beta_sweep")
        patches: Dict[str, Any] = {"loss.kld_beta": value}
        if epochs is not None:
            patches["optim.epochs"] = int(epochs)
        cells.append(
            TrainCell(benchmark, data_tag, run_tag, f"beta={value:g}", patches)
        )
    return cells


def _cells_hp(
    config: Dict[str, Any], axis: Optional[str], build: bool
) -> List[TrainCell]:
    """Enumerate a secondary hyper-parameter probe's training cells.

    Args:
        config: The parsed config.
        axis: One of :data:`beta_sweep._HP_AXES`
            (``lambda_base`` / ``d_z`` / ``warmup_period``).
        build: If True, generate the fixed dataset (idempotent).

    Returns:
        One :class:`TrainCell` per swept value.

    Raises:
        ValueError: If ``axis`` is not a valid HP axis or its grid is empty.
    """
    if axis not in bs._HP_AXES:
        raise ValueError(
            f"hp mode needs --axis one of {bs._HP_AXES}, got {axis!r}"
        )
    benchmark = str(config["experiment"]["benchmark"])
    bsconf = config.get("beta_sweep", {}) or {}
    grid = list((bsconf.get("hp_probes", {}) or {}).get(axis, []) or [])
    if not grid:
        raise ValueError(f"hp mode needs a populated beta_sweep.hp_probes.{axis}")
    data_tag = bs._fixed_data_tag(config)
    if build:
        bs._ensure_fixed_dataset(config, data_tag, build_missing=True)
    section, field_name = bs._AXIS_FIELD[axis]
    epochs = bsconf.get("epochs")
    cells: List[TrainCell] = []
    for value in grid:
        _, run_tag = bs._cell_tags(axis, value, f"hp_{axis}")
        patches: Dict[str, Any] = {f"{section}.{field_name}": value}
        if epochs is not None:
            patches["optim.epochs"] = int(epochs)
        cells.append(
            TrainCell(benchmark, data_tag, run_tag, f"{axis}={value:g}", patches)
        )
    return cells


def _cells_rho_null(config: Dict[str, Any], build: bool) -> List[TrainCell]:
    """Enumerate the Benchmark-B $\\rho$-null diagnostic cells.

    One zero-transfer ($a=0$) Benchmark-B dataset per $\\rho$; the run-tags
    mirror :func:`evaluate_te.run_rho_null_check`.

    Args:
        config: The parsed config.
        build: If True, generate each $\\rho$-null dataset (idempotent).

    Returns:
        One :class:`TrainCell` per $\\rho$ in ``benchmarks.B.rho_null.rho_grid``.
    """
    cfg = deepcopy(config)
    cfg["experiment"]["benchmark"] = "B"
    tm.resolve_active_benchmark(cfg)
    rho_grid = list((cfg.get("rho_null", {}) or {}).get("rho_grid", []))
    if not rho_grid:
        raise ValueError(
            "rho_null mode needs benchmarks.B.rho_null.rho_grid"
        )
    cells: List[TrainCell] = []
    for rho in rho_grid:
        token = "r" + bs._fmt_token(float(rho))
        data_tag = f"benchmark_B_rho_null_{token}"
        run_tag = f"rho_null/{token}"
        if build:
            dcfg = deepcopy(cfg)
            bd._apply_overrides(
                dcfg, {"tag": data_tag, "a": 0.0, "rho": float(rho)}
            )
            bd.build_dataset(dcfg, force=False)
        # a=0 is baked into the dataset; patch data.a so the train-time
        # provenance check does not warn.
        cells.append(
            TrainCell("B", data_tag, run_tag, f"rho={rho:g}", {"data.a": 0.0})
        )
    return cells


def _cells_directionality(
    config: Dict[str, Any], build: bool
) -> List[TrainCell]:
    """Enumerate the forward / reverse directionality training cells.

    Reuses :data:`directionality._DIRECTIONS` so the two checkpoints land where
    :func:`directionality.run_directionality` looks.

    Args:
        config: The parsed config.
        build: If True, generate the forward (A) and reverse (G) datasets.

    Returns:
        Two :class:`TrainCell` objects (forward Benchmark-A, reverse Benchmark-G).
    """
    cells: List[TrainCell] = []
    for label, benchmark, data_tag, run_tag in dr._DIRECTIONS:
        if build:
            dcfg = deepcopy(config)
            dcfg["experiment"]["benchmark"] = benchmark
            tm.resolve_active_benchmark(dcfg)
            bd._apply_overrides(dcfg, {"tag": data_tag})
            bd.build_dataset(dcfg, force=False)
        cells.append(TrainCell(benchmark, data_tag, run_tag, label))
    return cells


_ENUMERATORS = {
    "a_sweep": lambda cfg, build, axis: _cells_a_sweep(cfg, build),
    "beta": lambda cfg, build, axis: _cells_beta(cfg, build),
    "hp": lambda cfg, build, axis: _cells_hp(cfg, axis, build),
    "rho_null": lambda cfg, build, axis: _cells_rho_null(cfg, build),
    "directionality": lambda cfg, build, axis: _cells_directionality(cfg, build),
}


# =============================================================================
# The subprocess pool
# =============================================================================

def _ckpt_path(config: Dict[str, Any], cell: TrainCell) -> Path:
    """Resolve the ``final.ckpt`` path a cell will write.

    Args:
        config: The parsed config (for ``paths.results_dir``).
        cell: The training cell.

    Returns:
        ``results/<benchmark>/<run_tag>/final.ckpt``.
    """
    return ev._results_root(config) / cell.benchmark / cell.run_tag / "final.ckpt"


def _worker_command(cell: TrainCell, config_path: Path) -> List[str]:
    """Build the ``python -m ...gpu_pool --worker`` argv for one cell.

    The dotted-path config ``patches`` are *not* passed on the command line --
    they travel via the ``GPU_POOL_PATCHES`` environment variable (see
    :func:`run_pool`) so a JSON payload with spaces / quotes never has to
    survive shell or Windows command-line quoting.

    Args:
        cell: The training cell.
        config_path: Path to ``config_synth.yaml`` the worker reloads.

    Returns:
        The argument list for :class:`subprocess.Popen`.
    """
    return [
        sys.executable, "-u", "-m", _MODULE, "--worker",
        "--config", str(config_path),
        "--benchmark", cell.benchmark,
        "--data-tag", cell.data_tag,
        "--run-tag", cell.run_tag,
        "--device", "cuda:0",
    ]


def run_pool(
    config: Dict[str, Any],
    cells: Sequence[TrainCell],
    gpus: Sequence[int],
    config_path: Path,
    out_dir: Path,
    *,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Train ``cells`` across ``gpus``, one model per GPU slot.

    Cells whose ``final.ckpt`` already exists are skipped unless ``force`` is
    set. Each worker is pinned to one GPU via ``CUDA_VISIBLE_DEVICES`` and its
    full stdout / stderr is captured to ``out_dir/logs/<run_tag>.log``.

    Args:
        config: The parsed config (for resolving checkpoint paths).
        cells: The training cells to run.
        gpus: GPU slot list -- a repeated index packs two workers onto that GPU.
        config_path: Path to ``config_synth.yaml`` workers reload.
        out_dir: Directory for the per-cell logs.
        force: If True, retrain cells even when their checkpoint exists.

    Returns:
        One result dict per cell (``status`` in ``ok`` / ``failed`` / ``skipped``).
    """
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    pending: List[TrainCell] = []
    for cell in cells:
        if not force and _ckpt_path(config, cell).is_file():
            print(f"[skip] {cell.run_tag}: final.ckpt already exists")
            results.append({
                "label": cell.label, "benchmark": cell.benchmark,
                "data_tag": cell.data_tag, "run_tag": cell.run_tag,
                "gpu": "", "status": "skipped", "returncode": "",
                "seconds": "", "ckpt": str(_ckpt_path(config, cell)), "log": "",
            })
        else:
            pending.append(cell)

    slots: List[int] = list(gpus)
    running: Dict[subprocess.Popen, Dict[str, Any]] = {}
    try:
        while pending or running:
            while pending and slots:
                gpu = slots.pop(0)
                cell = pending.pop(0)
                log_path = logs_dir / (
                    cell.run_tag.replace("/", "__").replace("\\", "__") + ".log"
                )
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["PYTHONPATH"] = (
                    str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
                )
                # Per-cell config patches travel via the environment, not argv,
                # so JSON spaces / quotes never hit shell quoting.
                env["GPU_POOL_PATCHES"] = json.dumps(cell.patches)
                log_file = open(log_path, "w", encoding="utf-8")
                proc = subprocess.Popen(
                    _worker_command(cell, config_path),
                    cwd=str(_REPO_ROOT), env=env,
                    stdout=log_file, stderr=subprocess.STDOUT,
                )
                running[proc] = {
                    "cell": cell, "gpu": gpu, "t0": time.time(),
                    "log_file": log_file, "log_path": log_path,
                }
                print(f"[gpu {gpu}] START  {cell.run_tag}  ({cell.label})")
            if not running:
                break
            time.sleep(_POLL_SECONDS)
            for proc in list(running):
                rc = proc.poll()
                if rc is None:
                    continue
                info = running.pop(proc)
                info["log_file"].close()
                slots.append(info["gpu"])
                secs = time.time() - info["t0"]
                cell = info["cell"]
                ok = rc == 0
                print(
                    f"[gpu {info['gpu']}] {'DONE ' if ok else 'FAIL '} "
                    f"{cell.run_tag}  {secs:.0f}s  rc={rc}"
                    + ("" if ok else f"  -> see {info['log_path']}")
                )
                results.append({
                    "label": cell.label, "benchmark": cell.benchmark,
                    "data_tag": cell.data_tag, "run_tag": cell.run_tag,
                    "gpu": info["gpu"],
                    "status": "ok" if ok else "failed",
                    "returncode": rc, "seconds": round(secs, 1),
                    "ckpt": str(_ckpt_path(config, cell)) if ok else "",
                    "log": str(info["log_path"]),
                })
    except KeyboardInterrupt:
        print("\n[gpu-pool] interrupted -- terminating running workers ...")
        for proc in running:
            proc.terminate()
        for proc in running:
            try:
                proc.wait(timeout=10)
            except Exception:  # pragma: no cover - defensive
                proc.kill()
        for info in running.values():
            info["log_file"].close()
        raise
    return results


# =============================================================================
# Output
# =============================================================================

def _write_summary(
    results: List[Dict[str, Any]],
    mode: str,
    gpus: Sequence[int],
    wall_seconds: float,
    out_dir: Path,
) -> None:
    """Write the per-cell ``summary.csv`` and an aggregate ``summary.json``.

    Args:
        results: The per-cell result dicts from :func:`run_pool`.
        mode: The sweep mode that was run.
        gpus: The GPU slot list used.
        wall_seconds: Total wall-clock time of the pool run.
        out_dir: Destination directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        for row in results:
            writer.writerow({"mode": mode, **{k: row.get(k) for k in
                                              _SUMMARY_FIELDS if k != "mode"}})
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_failed = sum(1 for r in results if r["status"] == "failed")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump({
            "created": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "gpus": list(gpus),
            "wall_seconds": round(wall_seconds, 1),
            "n_cells": len(results),
            "n_ok": n_ok, "n_failed": n_failed, "n_skipped": n_skipped,
            "cells": results,
        }, fh, indent=2)


def _warn_unavailable_gpus(gpus: Sequence[int]) -> None:
    """Print a pre-flight warning for GPU indices the box does not expose.

    A worker pinned to a non-existent GPU fails cleanly (the pool records it
    and moves on), but warning up front saves a wasted run -- e.g. passing
    ``[0,1,2,3,4,5,6]`` on a 4-GPU box.

    Args:
        gpus: The requested GPU slot list.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            print("[gpu-pool] WARNING: torch reports no CUDA device -- every "
                  "worker will fail (this scheduler trains on GPUs).")
            return
        n_dev = torch.cuda.device_count()
        bad = sorted({int(g) for g in gpus if int(g) >= n_dev})
        if bad:
            print(f"[gpu-pool] WARNING: GPU indices {bad} are >= the visible "
                  f"device count ({n_dev}); cells scheduled on them will fail.")
    except Exception:  # pragma: no cover - defensive (never block the run)
        pass


# =============================================================================
# Library entry point (Decision D9)
# =============================================================================

def run_gpu_pool(
    config: Dict[str, Any],
    config_path: Path,
    *,
    mode: str,
    gpus: Sequence[int],
    axis: Optional[str] = None,
    build: bool = True,
    force: bool = False,
    epochs: Optional[int] = None,
    seed: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Enumerate, build and train one sweep mode across multiple GPUs.

    This is the reusable entry point. It is train-only: run the matching
    evaluation harness afterwards.

    Args:
        config: The parsed, benchmark-resolved ``config_synth.yaml``.
        config_path: Path to that config file (workers reload it).
        mode: One of :data:`_VALID_MODES`.
        gpus: GPU slot list (one model per slot; repeats pack a GPU).
        axis: HP axis -- required for ``mode="hp"``, ignored otherwise.
        build: If True, generate every needed dataset before training.
        force: If True, retrain cells whose checkpoint already exists.
        epochs: Optional global ``optim.epochs`` override applied to every cell
            (handy for a fast end-to-end shake-out).
        seed: Optional global ``experiment.seed`` override applied to every cell.
        dry_run: If True, print the enumerated cells and exit without building
            datasets or training.

    Returns:
        A results dict: ``mode``, ``cells`` (per-cell result rows), ``out_dir``,
        ``n_ok`` / ``n_failed`` / ``n_skipped``.

    Raises:
        ValueError: On an unknown ``mode``.
    """
    if mode not in _ENUMERATORS:
        raise ValueError(
            f"unknown mode {mode!r}; expected one of {_VALID_MODES}"
        )
    if not dry_run:
        _warn_unavailable_gpus(gpus)
    out_dir = ev._results_root(config) / "gpu_pool" / mode
    cells = _ENUMERATORS[mode](config, build and not dry_run, axis)

    # Global per-cell patches (applied on top of each cell's own patches).
    for cell in cells:
        if epochs is not None:
            cell.patches["optim.epochs"] = int(epochs)
        if seed is not None:
            cell.patches["experiment.seed"] = int(seed)

    print(
        f"[gpu-pool] mode={mode}  {len(cells)} cell(s)  "
        f"GPU slots={list(gpus)}  build={build}  force={force}"
    )
    for cell in cells:
        extra = f"  patches={cell.patches}" if cell.patches else ""
        print(f"  - {cell.run_tag:42s} [{cell.benchmark}] {cell.label}{extra}")

    if dry_run:
        print("[gpu-pool] dry run -- no datasets built, no training launched.")
        return {
            "mode": mode, "cells": [], "out_dir": str(out_dir),
            "n_ok": 0, "n_failed": 0, "n_skipped": 0,
        }

    t0 = time.time()
    results = run_pool(
        config, cells, gpus, config_path, out_dir, force=force,
    )
    wall = time.time() - t0
    _write_summary(results, mode, gpus, wall, out_dir)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_failed = sum(1 for r in results if r["status"] == "failed")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    print(
        f"\n[gpu-pool] done: {n_ok} ok, {n_failed} failed, {n_skipped} skipped"
        f"  wall={wall:.0f}s"
    )
    for row in results:
        if row["status"] == "failed":
            print(f"  FAILED {row['run_tag']}  -> {row['log']}")
    print(
        f"[gpu-pool] summary -> {out_dir / 'summary.csv'}\n"
        f"           next: run the evaluation harness "
        f"(evaluate_te / beta_sweep / lag_recovery / directionality)."
    )
    return {
        "mode": mode, "cells": results, "out_dir": str(out_dir),
        "n_ok": n_ok, "n_failed": n_failed, "n_skipped": n_skipped,
    }


# =============================================================================
# Worker mode
# =============================================================================

def _worker_main(args: argparse.Namespace) -> None:
    """Train exactly one cell (run as a ``--worker`` subprocess).

    Reloads the config, forces the cell's benchmark, applies the dotted-path
    patches from the ``GPU_POOL_PATCHES`` environment variable and trains via
    :func:`train_minimal.train`. The GPU is already isolated by the
    ``CUDA_VISIBLE_DEVICES`` the pool set, so ``--device`` is ``cuda:0`` inside
    this process.

    Args:
        args: The parsed worker arguments (``config``, ``benchmark``,
            ``data_tag``, ``run_tag``, ``device``).

    Raises:
        KeyError: If a patch names a config section that does not exist.
    """
    config = tm.load_config(Path(args.config))
    config["experiment"]["benchmark"] = args.benchmark
    tm.resolve_active_benchmark(config)

    patches: Dict[str, Any] = json.loads(
        os.environ.get("GPU_POOL_PATCHES") or "{}"
    )
    for dotted, value in patches.items():
        section, _, field_name = dotted.partition(".")
        if not field_name or section not in config \
                or not isinstance(config[section], dict):
            raise KeyError(f"gpu_pool worker: invalid patch path {dotted!r}")
        config[section][field_name] = value

    config["runtime"]["device"] = args.device
    tm.train(config, overrides={
        "data_tag": args.data_tag, "run_tag": args.run_tag,
    })


# =============================================================================
# CLI
# =============================================================================

def _parse_gpus(spec: Optional[str], config: Dict[str, Any]) -> List[int]:
    """Resolve the GPU slot list from ``--gpus`` or the config.

    Args:
        spec: A ``--gpus`` string like ``"0,1,2,3,4,5,6"``, or ``None``.
        config: The parsed config (``runtime.gpus`` / ``runtime.cuda_device``
            fallback).

    Returns:
        A non-empty list of GPU indices (slots; repeats allowed).

    Raises:
        ValueError: If no GPU can be resolved.
    """
    if spec:
        gpus = [int(x) for x in str(spec).split(",") if str(x).strip() != ""]
    else:
        runtime = config.get("runtime", {}) or {}
        cfg_gpus = runtime.get("gpus")
        if cfg_gpus:
            gpus = [int(g) for g in cfg_gpus]
        else:
            gpus = [int(runtime.get("cuda_device", 0))]
    if not gpus:
        raise ValueError("no GPUs resolved -- pass --gpus or set runtime.gpus")
    return gpus


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        The parsed :class:`argparse.Namespace`.
    """
    p = argparse.ArgumentParser(
        description="Task-parallel multi-GPU training scheduler for the "
                    "synthetic transfer-entropy sweeps (one model per GPU)."
    )
    p.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="path to config_synth.yaml",
    )
    p.add_argument(
        "--mode", type=str, default=None, choices=list(_VALID_MODES),
        help="which sweep to train (default: a_sweep)",
    )
    p.add_argument(
        "--axis", type=str, default=None, choices=list(bs._HP_AXES),
        help="hp mode: which hyper-parameter to sweep",
    )
    p.add_argument(
        "--gpus", type=str, default=None,
        help="comma-separated GPU slot list, e.g. 0,1,2,3,4,5,6 "
             "(repeats pack a GPU); defaults to runtime.gpus",
    )
    p.add_argument(
        "--benchmark", type=str, default=None,
        help="override experiment.benchmark",
    )
    p.add_argument(
        "--epochs", type=int, default=None,
        help="global optim.epochs override applied to every cell",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="global experiment.seed override applied to every cell",
    )
    p.add_argument(
        "--no-build", action="store_true", dest="no_build",
        help="assume datasets are already cached; skip dataset generation",
    )
    p.add_argument(
        "--force", action="store_true",
        help="retrain cells even when their final.ckpt already exists",
    )
    p.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="print the enumerated cells and exit (no build, no training)",
    )
    # --- worker mode (internal: the pool re-invokes this module) -------------
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--data-tag", type=str, default=None, dest="data_tag",
                   help=argparse.SUPPRESS)
    p.add_argument("--run-tag", type=str, default=None, dest="run_tag",
                   help=argparse.SUPPRESS)
    p.add_argument("--device", type=str, default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point: parse args, then dispatch to worker or pool mode.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    args = parse_args(argv)
    if args.worker:
        _worker_main(args)
        return
    config = tm.load_config(args.config)
    if args.benchmark:
        config["experiment"]["benchmark"] = args.benchmark
        tm.resolve_active_benchmark(config)
    gpus = _parse_gpus(args.gpus, config)
    run_gpu_pool(
        config, args.config, mode=args.mode or "a_sweep", gpus=gpus,
        axis=args.axis, build=not args.no_build, force=args.force,
        epochs=args.epochs, seed=args.seed, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    # =========================================================================
    # How to run this script  (project convention -- Decision D9)
    # -------------------------------------------------------------------------
    # Two equivalent modes, auto-detected from the command line:
    #
    #   * CLI mode      -- launched with any --flag -> argparse `main()`.
    #   * EDIT-AND-RUN  -- launched with NO arguments -> the `RUN_CONFIG` dict
    #                      below is used. Edit it and run the file directly;
    #                      no terminal flags required.
    #
    # This scheduler only TRAINS. After it finishes, run the evaluation harness
    # (evaluate_te / beta_sweep / lag_recovery / directionality / final_report)
    # -- every checkpoint is already on disk.
    # =========================================================================

    CONFIG_PATH = _DEFAULT_CONFIG

    RUN_CONFIG = {
        "mode": "a_sweep",         # a_sweep | beta | hp | rho_null | directionality
        "axis": "lambda_base",     # hp mode: lambda_base | d_z | warmup_period
        "gpus": None,              # e.g. [0,1,2,3,4,5,6]; None -> runtime.gpus
        "benchmark": None,         # None -> config experiment.benchmark
        "epochs": None,            # global optim.epochs override (None -> config)
        "seed": None,              # global experiment.seed override
        "no_build": False,         # True -> assume datasets already cached
        "force": False,            # True -> retrain even if final.ckpt exists
        "dry_run": False,          # True -> just list the cells and exit
    }

    if len(sys.argv) > 1:
        main()                              # CLI mode -- argparse
    else:
        config = tm.load_config(CONFIG_PATH)
        if RUN_CONFIG["benchmark"]:
            config["experiment"]["benchmark"] = RUN_CONFIG["benchmark"]
            tm.resolve_active_benchmark(config)
        gpus = _parse_gpus(
            ",".join(str(g) for g in RUN_CONFIG["gpus"])
            if RUN_CONFIG["gpus"] else None,
            config,
        )
        run_gpu_pool(
            config, CONFIG_PATH, mode=RUN_CONFIG["mode"], gpus=gpus,
            axis=RUN_CONFIG["axis"], build=not RUN_CONFIG["no_build"],
            force=RUN_CONFIG["force"], epochs=RUN_CONFIG["epochs"],
            seed=RUN_CONFIG["seed"], dry_run=RUN_CONFIG["dry_run"],
        )
