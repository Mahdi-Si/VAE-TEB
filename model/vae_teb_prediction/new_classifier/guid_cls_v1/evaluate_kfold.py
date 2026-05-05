"""K-fold evaluation orchestrator for ``guid_cls_v1``.

Standalone counterpart to :mod:`kfold_trainer`: runs
:func:`evaluate_single_fold` (per-fold inference → threshold search →
three-metric-types → 3-class diagnostics) over a set of already-trained
folds, then calls :func:`aggregate_results` for the cross-fold summary.

Two flavours of "skip work" are supported:

* ``regenerate_predictions=False`` (default) — reuse the cached
  ``validation_predictions_raw.csv`` / ``test_predictions_raw.csv`` if
  they exist, only running inference when they are missing. The
  threshold search, time-binned metrics, per-class plots, and ROC plots
  are *always* recomputed because they are cheap.
* ``aggregate_only=True`` — skip per-fold evaluation entirely and only
  recompute the cross-fold aggregation. Useful when you want to refresh
  the aggregator output after editing a plotting helper.

CLI flag fallbacks: every CLI flag falls back to its corresponding YAML
config field when not supplied. The config keys consumed are:

* ``general_config.cuda_devices`` (default ``[0]``)
* ``dataset_config.fold_ids`` then ``dataset_config.num_folds``
  (default ``[1..num_folds]``)
* ``evaluation.regenerate_predictions`` (default ``False``)
* ``evaluation.aggregate_only`` (default ``False``)

Parallelism: by default each fold is evaluated sequentially in-process
(evaluation is fast enough that subprocess overhead dominates the
benefit). Pass ``--parallel`` to run folds in subprocesses with
round-robin GPU pinning when multiple GPUs are configured.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import queue
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from loguru import logger


# ---------------------------------------------------------------------------
# Config helpers (mirrors kfold_trainer for parity)
# ---------------------------------------------------------------------------


def _load_config(path: str) -> Dict[str, Any]:
    """Parse the YAML config from disk."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_run_dir_lazy(config: Dict[str, Any]) -> Path:
    """Build the run directory from config (mirrors precompute_latents)."""
    base = Path(config["general_config"]["folders_config"]["out_dir_base"]).resolve()
    tag = str(config["general_config"].get("tag", "guid_cls_v1_run"))
    return base / tag


def _git_sha(cwd: Optional[str] = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return ""


def _resolve_fold_ids(
    config: Dict[str, Any],
    cli_fold_ids: Optional[Sequence[int]],
    run_dir: Path,
) -> List[int]:
    """CLI > config > on-disk fold_* discovery > [1..num_folds]."""
    if cli_fold_ids is not None:
        return [int(f) for f in cli_fold_ids]
    cfg_fold_ids = config.get("dataset_config", {}).get("fold_ids")
    if cfg_fold_ids:
        return [int(f) for f in cfg_fold_ids]
    discovered: List[int] = []
    for child in sorted(run_dir.glob("fold_*")):
        if not child.is_dir():
            continue
        try:
            discovered.append(int(child.name.split("_")[-1]))
        except ValueError:
            continue
    if discovered:
        return discovered
    num_folds = int(config.get("dataset_config", {}).get("num_folds", 10))
    return list(range(1, num_folds + 1))


# ---------------------------------------------------------------------------
# Per-fold evaluation entry point (subprocess-friendly)
# ---------------------------------------------------------------------------


def _run_eval_one_fold_subprocess(
    fold_id: int,
    gpu_id: int,
    config_path: str,
    run_dir_str: str,
    regenerate_predictions: bool,
) -> Dict[str, Any]:
    """Subprocess entry point: pin GPU, then evaluate one fold.

    Args:
        fold_id: 1-based fold id.
        gpu_id: Physical GPU id assigned to this fold.
        config_path: Path to YAML config.
        run_dir_str: Run directory containing ``fold_<id>/``.
        regenerate_predictions: When True, force re-running inference.

    Returns:
        Result dict matching :func:`evaluate_single_fold`'s return,
        with ``status`` set to ``"failed"`` on exception.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Lazy imports — must come AFTER CUDA_VISIBLE_DEVICES is set.
    from model.vae_teb_prediction.new_classifier.guid_cls_v1.evaluate_guid_classifier import (  # noqa: WPS433
        evaluate_single_fold,
    )

    run_dir = Path(run_dir_str)
    fold_dir = run_dir / f"fold_{int(fold_id)}"
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        # Try to read fold_results.json to recover the authoritative best ckpt.
        best_ckpt: Optional[str] = None
        fold_results_path = fold_dir / "fold_results.json"
        if fold_results_path.exists():
            try:
                fr = json.loads(fold_results_path.read_text(encoding="utf-8"))
                best_ckpt = fr.get("best_checkpoint_path") or None
            except Exception as exc:
                logger.warning(
                    f"[fold {fold_id}] could not parse fold_results.json: {exc}"
                )
        result = evaluate_single_fold(
            fold_dir=fold_dir,
            config=config,
            regenerate_predictions=bool(regenerate_predictions),
            best_checkpoint_path=best_ckpt,
        )
        result.setdefault("fold_id", int(fold_id))
        result.setdefault("status", "ok")
        result["physical_gpu_id"] = int(gpu_id)
        return result
    except Exception as exc:  # pragma: no cover — reported via JSON
        return {
            "fold_id": int(fold_id),
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "physical_gpu_id": int(gpu_id),
        }


def _eval_worker_entry(
    result_queue: Any,
    fold_id: int,
    gpu_id: int,
    config_path: str,
    run_dir_str: str,
    regenerate_predictions: bool,
) -> None:
    """Child-process entry point that forwards one fold result to a queue."""
    result = _run_eval_one_fold_subprocess(
        fold_id=fold_id,
        gpu_id=gpu_id,
        config_path=config_path,
        run_dir_str=run_dir_str,
        regenerate_predictions=regenerate_predictions,
    )
    result_queue.put(result)


# ---------------------------------------------------------------------------
# Sequential and parallel drivers
# ---------------------------------------------------------------------------


def _run_eval_sequential(
    *,
    fold_ids: Sequence[int],
    cuda_devices: Sequence[int],
    config_path: str,
    run_dir: Path,
    regenerate_predictions: bool,
) -> Dict[int, Dict[str, Any]]:
    """Run all folds sequentially in this process.

    Pins ``CUDA_VISIBLE_DEVICES`` to ``cuda_devices[0]`` once before the
    first torch import so every fold lands on the same GPU. If multiple
    GPUs were configured but parallel mode was not requested, the extras
    are ignored — callers that want one fold per GPU should pass
    ``--parallel``.
    """
    sequential_gpu = int(cuda_devices[0]) if cuda_devices else 0
    if len(cuda_devices) > 1 and len(fold_ids) > 1:
        logger.info(
            f"sequential evaluation pins all folds to physical GPU "
            f"{sequential_gpu} (drop --parallel ignored). cuda_devices="
            f"{list(cuda_devices)}; pass --parallel to spread folds."
        )
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(sequential_gpu))

    # Lazy import after CUDA_VISIBLE_DEVICES so the eager torch.cuda init
    # picks up the right device.
    from model.vae_teb_prediction.new_classifier.guid_cls_v1.evaluate_guid_classifier import (  # noqa: WPS433
        evaluate_single_fold,
    )

    config = _load_config(config_path)
    results: Dict[int, Dict[str, Any]] = {}
    for fid in fold_ids:
        fold_dir = run_dir / f"fold_{int(fid)}"
        if not fold_dir.exists():
            logger.error(f"[fold {fid}] missing dir {fold_dir}; skipping")
            results[int(fid)] = {
                "fold_id": int(fid),
                "status": "failed",
                "error": f"missing fold dir {fold_dir}",
            }
            continue

        # Try to recover the authoritative best ckpt from fold_results.json.
        best_ckpt: Optional[str] = None
        fr_path = fold_dir / "fold_results.json"
        if fr_path.exists():
            try:
                fr = json.loads(fr_path.read_text(encoding="utf-8"))
                best_ckpt = fr.get("best_checkpoint_path") or None
            except Exception as exc:
                logger.warning(f"[fold {fid}] could not parse fold_results.json: {exc}")

        logger.info(f"--- evaluating fold {fid} on physical GPU {sequential_gpu} ---")
        try:
            result = evaluate_single_fold(
                fold_dir=fold_dir,
                config=config,
                regenerate_predictions=bool(regenerate_predictions),
                best_checkpoint_path=best_ckpt,
            )
            result.setdefault("fold_id", int(fid))
            result.setdefault("status", "ok")
            result["physical_gpu_id"] = sequential_gpu
            results[int(fid)] = result
        except Exception as exc:  # pragma: no cover - per-fold isolation
            logger.error(f"[fold {fid}] evaluation failed: {exc}")
            results[int(fid)] = {
                "fold_id": int(fid),
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "physical_gpu_id": sequential_gpu,
            }
    return results


def _run_eval_parallel(
    *,
    fold_ids: Sequence[int],
    cuda_devices: Sequence[int],
    max_parallel: int,
    config_path: str,
    run_dir: Path,
    regenerate_predictions: bool,
    fold_timeout_seconds: float,
) -> Dict[int, Dict[str, Any]]:
    """Run evaluation in spawn subprocesses, round-robin'd over GPUs."""
    ctx = multiprocessing.get_context("spawn")
    pending: List[Tuple[int, int]] = [
        (int(fid), int(cuda_devices[i % len(cuda_devices)]))
        for i, fid in enumerate(fold_ids)
    ]
    active: Dict[int, Dict[str, Any]] = {}
    results: Dict[int, Dict[str, Any]] = {}

    while pending or active:
        while pending and len(active) < max_parallel:
            fid, gpu = pending.pop(0)
            result_queue = ctx.Queue()
            proc = ctx.Process(
                target=_eval_worker_entry,
                args=(
                    result_queue,
                    fid,
                    gpu,
                    config_path,
                    str(run_dir),
                    bool(regenerate_predictions),
                ),
            )
            proc.start()
            active[fid] = {
                "process": proc,
                "queue": result_queue,
                "gpu": gpu,
                "started_monotonic": time.monotonic(),
            }
            logger.info(f"submitted eval for fold {fid} -> physical GPU {gpu}")

        progressed = False
        for fid in list(active.keys()):
            slot = active[fid]
            proc = slot["process"]
            result_queue = slot["queue"]

            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                result = None

            if result is not None:
                proc.join(timeout=1.0)
                results[fid] = result
                logger.info(
                    f"fold {fid} eval finished with status={result.get('status', '?')}"
                )
                result_queue.close()
                del active[fid]
                progressed = True
                continue

            if not proc.is_alive():
                proc.join(timeout=1.0)
                results[fid] = {
                    "fold_id": fid,
                    "status": "failed",
                    "error": (
                        f"eval subprocess exited without returning a result "
                        f"(exitcode={proc.exitcode})"
                    ),
                    "physical_gpu_id": int(slot["gpu"]),
                }
                logger.error(results[fid]["error"])
                result_queue.close()
                del active[fid]
                progressed = True
                continue

            elapsed = time.monotonic() - float(slot["started_monotonic"])
            if elapsed > fold_timeout_seconds:
                logger.error(
                    f"fold {fid} eval exceeded timeout of "
                    f"{fold_timeout_seconds / 3600.0:.2f}h; terminating"
                )
                proc.terminate()
                proc.join(timeout=10.0)
                if proc.is_alive():  # pragma: no cover - defensive
                    proc.kill()
                    proc.join(timeout=5.0)
                results[fid] = {
                    "fold_id": fid,
                    "status": "failed",
                    "error": (
                        f"eval exceeded timeout of "
                        f"{fold_timeout_seconds / 3600.0:.2f} hours"
                    ),
                    "physical_gpu_id": int(slot["gpu"]),
                }
                result_queue.close()
                del active[fid]
                progressed = True

        if pending or active:
            time.sleep(0.5 if progressed else 1.0)

    return results


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def evaluate_kfold(
    *,
    config_path: str,
    output_dir_override: Optional[str] = None,
    fold_ids: Optional[Sequence[int]] = None,
    cuda_devices: Optional[Sequence[int]] = None,
    regenerate_predictions: Optional[bool] = None,
    aggregate_only: Optional[bool] = None,
    parallel: bool = False,
    max_parallel: Optional[int] = None,
    fold_timeout_hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate (or just aggregate) one or more already-trained folds.

    Args:
        config_path: Path to YAML config (consumed for fallbacks, model
            shape, dataset paths, evaluation knobs).
        output_dir_override: Optional run-dir override (defaults to
            ``out_dir_base / tag``).
        fold_ids: Subset of fold ids; falls back to
            ``dataset_config.fold_ids`` then ``[1..num_folds]``.
        cuda_devices: GPU id list; falls back to
            ``general_config.cuda_devices``.
        regenerate_predictions: When True, force re-running inference even
            if cached prediction CSVs exist. Falls back to
            ``evaluation.regenerate_predictions``.
        aggregate_only: When True, skip per-fold evaluation entirely and
            only run cross-fold aggregation. Falls back to
            ``evaluation.aggregate_only``.
        parallel: When True, run folds in spawn subprocesses with
            round-robin GPU pinning. Default False (sequential).
        max_parallel: Cap on concurrent folds in parallel mode; defaults
            to ``general_config.max_parallel_folds``.
        fold_timeout_hours: Per-fold timeout in parallel mode; defaults
            to ``general_config.fold_timeout_hours``.

    Returns:
        Dict with the per-fold result list, the cross-fold aggregation
        summary, and execution metadata. Also writes
        ``evaluation_kfold_results.json`` under the run dir.
    """
    config = _load_config(config_path)
    eval_cfg = config.get("evaluation", {}) or {}

    # CLI > config fallback for boolean flags.
    if regenerate_predictions is None:
        regenerate_predictions = bool(eval_cfg.get("regenerate_predictions", False))
    if aggregate_only is None:
        aggregate_only = bool(eval_cfg.get("aggregate_only", False))

    run_dir = (
        Path(output_dir_override).resolve()
        if output_dir_override
        else _resolve_run_dir_lazy(config)
    )
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")

    cuda_devices = (
        list(cuda_devices)
        if cuda_devices is not None
        else list(config["general_config"].get("cuda_devices", [0]))
    )
    if not cuda_devices:
        cuda_devices = [0]

    fids = _resolve_fold_ids(config, fold_ids, run_dir)
    started = datetime.now(timezone.utc)
    logger.info(
        f"evaluate_kfold: folds={fids} gpus={cuda_devices} "
        f"regenerate_predictions={regenerate_predictions} "
        f"aggregate_only={aggregate_only} parallel={parallel} "
        f"run_dir={run_dir}"
    )

    per_fold_results: List[Dict[str, Any]] = []
    if aggregate_only:
        logger.info("aggregate_only=True: skipping per-fold evaluation")
    else:
        if parallel and len(fids) > 1 and len(cuda_devices) > 0:
            mp = int(
                max_parallel
                if max_parallel is not None
                else config["general_config"].get(
                    "max_parallel_folds", len(cuda_devices)
                )
            )
            mp = max(1, mp)
            timeout_h = float(
                fold_timeout_hours
                if fold_timeout_hours is not None
                else config["general_config"].get("fold_timeout_hours", 6.0)
            )
            results = _run_eval_parallel(
                fold_ids=fids,
                cuda_devices=cuda_devices,
                max_parallel=mp,
                config_path=config_path,
                run_dir=run_dir,
                regenerate_predictions=bool(regenerate_predictions),
                fold_timeout_seconds=timeout_h * 3600.0,
            )
        else:
            results = _run_eval_sequential(
                fold_ids=fids,
                cuda_devices=cuda_devices,
                config_path=config_path,
                run_dir=run_dir,
                regenerate_predictions=bool(regenerate_predictions),
            )
        per_fold_results = [results[fid] for fid in sorted(results.keys())]

    # ------------------------------------------------------------------
    # Cross-fold aggregation. Best-effort — a partial sweep should still
    # produce summary plots / JSON for the folds that did succeed.
    # ------------------------------------------------------------------
    aggregated: Optional[Dict[str, Any]] = None
    successful_fids = (
        fids
        if aggregate_only
        else [int(r["fold_id"]) for r in per_fold_results if r.get("status") == "ok"]
    )
    if successful_fids:
        try:
            from model.vae_teb_prediction.new_classifier.guid_cls_v1.aggregate_results import (  # noqa: WPS433
                aggregate_results,
            )

            decision_time_hours_cfg = (
                (config.get("evaluation", {}) or {}).get("decision_time_hours")
            )
            aggregated = aggregate_results(
                run_dir=run_dir,
                fold_ids=successful_fids,
                decision_time_hours=(
                    float(decision_time_hours_cfg)
                    if decision_time_hours_cfg is not None
                    else None
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"aggregate_results failed: {exc}")
    else:
        logger.warning("no successful folds; skipping cross-fold aggregation")

    finished = datetime.now(timezone.utc)
    metadata = {
        "config_path": str(config_path),
        "git_sha": _git_sha(),
        "run_dir": str(run_dir),
        "fold_ids": [int(f) for f in fids],
        "regenerate_predictions": bool(regenerate_predictions),
        "aggregate_only": bool(aggregate_only),
        "parallel": bool(parallel),
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "wall_seconds": (finished - started).total_seconds(),
    }
    summary: Dict[str, Any] = {
        "metadata": metadata,
        "per_fold_results": per_fold_results,
        "aggregated_summary": aggregated,
        "n_folds": len(fids),
        "n_successful": (
            len(fids) if aggregate_only else sum(
                1 for r in per_fold_results if r.get("status") == "ok"
            )
        ),
    }
    out_json = run_dir / "evaluation_kfold_results.json"
    out_json.write_text(
        json.dumps(_to_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info(
        f"evaluate_kfold: done in {(finished - started).total_seconds():.1f}s; "
        f"summary -> {out_json}"
    )
    return summary


def _to_json_safe(obj: Any) -> Any:
    """Convert numpy / torch / dataclass values to JSON-serialisable scalars."""
    import numpy as np  # local import keeps import cost low

    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Examples:
        # Evaluate every fold present (uses cached predictions when available)
        python -m model.vae_teb_prediction.new_classifier.guid_cls_v1.evaluate_kfold \\
            --config path/to/config_guid_cls_v1.yaml

        # Force re-running inference for a specific subset of folds
        python -m model.vae_teb_prediction.new_classifier.guid_cls_v1.evaluate_kfold \\
            --config path/to/config_guid_cls_v1.yaml \\
            --fold-ids 1 3 5 \\
            --regenerate-predictions

        # Skip per-fold work, only refresh the cross-fold aggregation
        python -m model.vae_teb_prediction.new_classifier.guid_cls_v1.evaluate_kfold \\
            --config path/to/config_guid_cls_v1.yaml --aggregate-only

        # Run folds in parallel across configured GPUs
        python -m model.vae_teb_prediction.new_classifier.guid_cls_v1.evaluate_kfold \\
            --config path/to/config_guid_cls_v1.yaml --parallel
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run guid_cls_v1 evaluation across already-trained folds and "
            "produce the cross-fold aggregation. Reuses cached prediction "
            "CSVs by default — pass --regenerate-predictions to force a "
            "fresh inference pass."
        )
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the run output directory (defaults to out_dir_base/tag)",
    )
    parser.add_argument(
        "--fold-ids",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Subset of fold ids to evaluate. Falls back to "
            "dataset_config.fold_ids, then to fold_* dirs found on disk, "
            "then to [1..num_folds]."
        ),
    )
    parser.add_argument(
        "--cuda-devices",
        type=int,
        nargs="*",
        default=None,
        help="Override GPU id list (defaults to general_config.cuda_devices)",
    )
    parser.add_argument(
        "--regenerate-predictions",
        action="store_true",
        default=None,
        help=(
            "Force re-running inference even if cached prediction CSVs "
            "exist. Without this flag the orchestrator falls back to "
            "evaluation.regenerate_predictions (default False) — i.e. it "
            "reuses cached CSVs when present."
        ),
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        default=None,
        help=(
            "Skip per-fold evaluation entirely and only run the cross-fold "
            "aggregation. Falls back to evaluation.aggregate_only (default "
            "False)."
        ),
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help=(
            "Run folds in spawn subprocesses with round-robin GPU pinning. "
            "Default is sequential, since per-fold evaluation is fast."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help=(
            "Cap on concurrent folds in --parallel mode (defaults to "
            "general_config.max_parallel_folds)."
        ),
    )
    parser.add_argument(
        "--fold-timeout-hours",
        type=float,
        default=None,
        help=(
            "Per-fold wall-clock timeout in --parallel mode (defaults to "
            "general_config.fold_timeout_hours)."
        ),
    )
    args = parser.parse_args(argv)

    evaluate_kfold(
        config_path=args.config,
        output_dir_override=args.output_dir,
        fold_ids=args.fold_ids,
        cuda_devices=args.cuda_devices,
        regenerate_predictions=args.regenerate_predictions,
        aggregate_only=args.aggregate_only,
        parallel=bool(args.parallel),
        max_parallel=args.max_parallel,
        fold_timeout_hours=args.fold_timeout_hours,
    )
    return 0


__all__ = ["evaluate_kfold", "main"]


if __name__ == "__main__":
    sys.exit(main())
