"""K-fold parallel orchestrator for ``guid_cls_v1`` (PRD §10).

Runs each fold in its own subprocess (``spawn`` start method) with a
dedicated GPU id round-robin'd from ``general_config.cuda_devices``. Each
subprocess sets ``CUDA_VISIBLE_DEVICES`` *before* importing torch.

After all folds complete, fold-level results are aggregated into
``kfold_results.json``, summarised into ``kfold_summary.json``, and an
``execution_metadata.json`` is written.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
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


def _resolve_run_dir_lazy(config: Dict[str, Any]) -> Path:
    """Build the run directory from config, mirroring precompute_latents."""
    base = Path(config["general_config"]["folders_config"]["out_dir_base"]).resolve()
    tag = str(config["general_config"].get("tag", "guid_cls_v1_run"))
    return base / tag


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _git_sha(cwd: Optional[str] = None) -> str:
    """Return the short git SHA, or empty string if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return ""


def _run_one_fold_subprocess(
    fold_id: int,
    gpu_id: int,
    config_path: str,
    output_dir_override: Optional[str],
) -> Dict[str, Any]:
    """Subprocess entry-point for a single fold.

    Sets ``CUDA_VISIBLE_DEVICES`` before any torch import, then delegates to
    :func:`train_fold`. All exceptions are captured into a structured
    result dict so the parent process can survive partial failures.

    Args:
        fold_id: 1-based fold id.
        gpu_id: Physical GPU id assigned to this fold.
        config_path: Path to YAML config.
        output_dir_override: Optional run-dir override.

    Returns:
        Result dict (matches :func:`train_fold` schema, with ``status`` set
        to ``"failed"`` on exception).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Lazy imports: must come AFTER the CUDA_VISIBLE_DEVICES env var is set.
    from pathlib import Path

    from model.vae_teb_prediction.new_classifier.guid_cls_v1.evaluate_guid_classifier import (  # noqa: WPS433
        evaluate_single_fold,
    )
    from model.vae_teb_prediction.new_classifier.guid_cls_v1.single_fold_trainer import (  # noqa: WPS433
        train_fold,
    )

    try:
        result = train_fold(
            fold_id=fold_id,
            config_path=config_path,
            gpu_id=0,                           # logical id after CUDA_VISIBLE_DEVICES masking
            output_dir_override=output_dir_override,
            auto_precompute=True,
        )
        # Now run evaluation immediately so the parent process gets a single
        # fold_results.json + evaluation/ tree.
        run_dir = Path(output_dir_override) if output_dir_override else None
        if run_dir is None:
            cfg = result.get("config", {})
            base = Path(cfg["general_config"]["folders_config"]["out_dir_base"]).resolve()
            tag = str(cfg["general_config"].get("tag", "guid_cls_v1_run"))
            run_dir = base / tag
        fold_dir = run_dir / f"fold_{int(fold_id)}"
        # Pass the exact checkpoint path Lightning promoted during training.
        # This bypasses ``find_best_checkpoint``'s directory scan and guarantees
        # the evaluator scores the same checkpoint the ModelCheckpoint callback
        # chose as best (avoids silent selection of a top-k-but-not-best file
        # when filename parsing fails).
        best_ckpt = result.get("best_checkpoint_path") or None
        eval_result = evaluate_single_fold(
            fold_dir=fold_dir,
            config=result.get("config", {}),
            regenerate_predictions=False,
            best_checkpoint_path=best_ckpt,
        )
        result["evaluation"] = eval_result
        result["physical_gpu_id"] = int(gpu_id)
        return result
    except Exception as exc:  # pragma: no cover - reported via fold_results.json
        tb = traceback.format_exc()
        return {
            "fold_id": int(fold_id),
            "status": "failed",
            "error": str(exc),
            "traceback": tb,
            "physical_gpu_id": int(gpu_id),
        }


def _fold_worker_entry(
    result_queue: Any,
    fold_id: int,
    gpu_id: int,
    config_path: str,
    output_dir_override: Optional[str],
) -> None:
    """Child-process entry point that forwards one fold result to a queue."""
    result = _run_one_fold_subprocess(
        fold_id=fold_id,
        gpu_id=gpu_id,
        config_path=config_path,
        output_dir_override=output_dir_override,
    )
    result_queue.put(result)


def _summarise_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean/std across folds for the headline metrics.

    Args:
        results: List of per-fold result dicts.

    Returns:
        Summary dict suitable for ``kfold_summary.json``.
    """
    successes = [r for r in results if r.get("status") == "ok"]
    failures = [r for r in results if r.get("status") != "ok"]

    def _stats(key: str) -> Dict[str, Optional[float]]:
        vals: List[float] = []
        for r in successes:
            v = r.get("best_metrics", {}).get(key)
            if v is None:
                v = r.get(key)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        if len(vals) == 1:
            return {"mean": float(vals[0]), "std": 0.0, "n": 1}
        import statistics

        return {
            "mean": float(statistics.mean(vals)),
            "std": float(statistics.stdev(vals)),
            "n": int(len(vals)),
        }

    summary: Dict[str, Any] = {
        "n_total": len(results),
        "n_successful": len(successes),
        "n_failed": len(failures),
        "metrics": {
            "val/total_loss": _stats("val/total_loss"),
            "val/macro_f1": _stats("val/macro_f1"),
            "val/binary_auroc": _stats("val/binary_auroc"),
            "val/acc_3": _stats("val/acc_3"),
            "val/acc_bin": _stats("val/acc_bin"),
            "best_val_total_loss": _stats("best_val_total_loss"),
        },
        "train_seconds": _stats("train_seconds"),
        "failed_fold_ids": sorted(int(r["fold_id"]) for r in failures),
    }
    return summary


def _execution_metadata(
    *,
    config_path: str,
    cuda_devices: Sequence[int],
    fold_ids: Sequence[int],
    started: datetime,
    finished: datetime,
) -> Dict[str, Any]:
    """Capture environment metadata for reproducibility."""
    return {
        "config_path": str(config_path),
        "git_sha": _git_sha(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_available": _cuda_available(),
        "cuda_version": _cuda_version(),
        "cuda_devices_requested": list(int(g) for g in cuda_devices),
        "fold_ids_run": list(int(f) for f in fold_ids),
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "wall_seconds": (finished - started).total_seconds(),
    }


def _cuda_available() -> bool:
    try:
        import torch  # noqa: WPS433

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cuda_version() -> str:
    try:
        import torch  # noqa: WPS433

        return str(torch.version.cuda or "")
    except Exception:
        return ""


def run_kfold_parallel(
    *,
    config_path: str,
    output_dir_override: Optional[str] = None,
    fold_ids: Optional[Sequence[int]] = None,
    cuda_devices: Optional[Sequence[int]] = None,
    max_parallel: Optional[int] = None,
    sequential: bool = False,
    fold_timeout_hours: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Run all (or selected) folds in parallel subprocesses.

    Args:
        config_path: Path to YAML config.
        output_dir_override: Optional run-dir override.
        fold_ids: Optional list of fold ids; defaults to ``range(1, num_folds + 1)``.
        cuda_devices: Optional override for the GPU ids; defaults to the
            list in ``general_config.cuda_devices``.
        max_parallel: Cap on concurrent folds; defaults to
            ``general_config.max_parallel_folds``.
        sequential: When True, force ``max_parallel=1`` (handy for debugging).
        fold_timeout_hours: Wall-clock timeout per fold.

    Returns:
        List of per-fold result dicts (sorted by ``fold_id``). Also writes
        ``kfold_results.json``, ``kfold_summary.json`` and
        ``execution_metadata.json`` under the run directory.
    """
    config = _load_config(config_path)
    run_dir = (
        Path(output_dir_override).resolve()
        if output_dir_override
        else _resolve_run_dir_lazy(config)
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    cuda_devices = list(cuda_devices) if cuda_devices is not None else list(
        config["general_config"].get("cuda_devices", [0])
    )
    if not cuda_devices:
        cuda_devices = [0]
    max_parallel = int(
        max_parallel
        if max_parallel is not None
        else config["general_config"].get("max_parallel_folds", len(cuda_devices))
    )
    if sequential:
        max_parallel = 1
    if fold_timeout_hours is None:
        fold_timeout_hours = float(
            config["general_config"].get("fold_timeout_hours", 6.0)
        )
    else:
        fold_timeout_hours = float(fold_timeout_hours)

    if fold_ids is None:
        cfg_fold_ids = config["dataset_config"].get("fold_ids")
        if cfg_fold_ids:
            fold_ids = list(int(f) for f in cfg_fold_ids)
        else:
            num_folds = int(config["dataset_config"].get("num_folds", 10))
            fold_ids = list(range(1, num_folds + 1))
    else:
        fold_ids = list(int(f) for f in fold_ids)

    started = datetime.now(timezone.utc)
    freeze_vae = bool(config.get("vae", {}).get("freeze_vae", True))
    logger.info(
        f"run_kfold_parallel: folds={fold_ids} gpus={cuda_devices} "
        f"max_parallel={max_parallel} run_dir={run_dir} "
        f"freeze_vae={freeze_vae}"
    )

    results: Dict[int, Dict[str, Any]] = {}

    if max_parallel <= 1 or len(fold_ids) == 1:
        # Sequential path runs in-process. CUDA_VISIBLE_DEVICES can only
        # be honoured BEFORE torch initialises in a process; once torch
        # has bound to a GPU, env-var changes are silently ignored. So in
        # sequential mode we pin every fold to ``cuda_devices[0]`` and
        # set the env var exactly once. If the user wants to run
        # different folds on different GPUs, they must use the parallel
        # branch (each subprocess is a fresh interpreter).
        sequential_gpu = int(cuda_devices[0])
        if len(cuda_devices) > 1 and len(fold_ids) > 1:
            logger.warning(
                "sequential mode pins all folds to physical GPU "
                f"{sequential_gpu}; cuda_devices={cuda_devices} extra "
                "entries are ignored. Use parallel mode (drop --sequential) "
                "to spread folds across GPUs."
            )
        os.environ.setdefault(
            "CUDA_VISIBLE_DEVICES", str(sequential_gpu)
        )
        for fid in fold_ids:
            logger.info(
                f"--- fold {fid} on physical GPU {sequential_gpu} "
                f"(freeze_vae={freeze_vae}) ---"
            )
            results[int(fid)] = _run_one_fold_subprocess(
                fold_id=int(fid),
                gpu_id=sequential_gpu,
                config_path=config_path,
                output_dir_override=str(run_dir),
            )
            # Mirror the parallel branch: surface per-fold status to the
            # console immediately. Otherwise an exception inside ``train_fold``
            # / ``evaluate_single_fold`` is captured silently into the result
            # dict, the loop "finishes" in seconds, and the user has no clue
            # anything went wrong until they open kfold_results.json.
            status = results[int(fid)].get("status", "unknown")
            if status == "ok":
                logger.info(f"fold {fid} finished with status={status}")
            else:
                err = results[int(fid)].get("error", "<no error message>")
                tb = results[int(fid)].get("traceback", "")
                logger.error(
                    f"fold {fid} FAILED: {err}\n"
                    f"--- traceback (head) ---\n{tb[-2000:] if tb else '<no traceback>'}"
                )
    else:
        ctx = multiprocessing.get_context("spawn")
        pending: List[Tuple[int, int]] = [
            (int(fid), int(cuda_devices[i % len(cuda_devices)]))
            for i, fid in enumerate(fold_ids)
        ]
        active: Dict[int, Dict[str, Any]] = {}
        timeout_seconds = fold_timeout_hours * 3600.0

        while pending or active:
            while pending and len(active) < max_parallel:
                fid, gpu = pending.pop(0)
                result_queue = ctx.Queue()
                proc = ctx.Process(
                    target=_fold_worker_entry,
                    args=(result_queue, fid, gpu, config_path, str(run_dir)),
                )
                proc.start()
                active[fid] = {
                    "process": proc,
                    "queue": result_queue,
                    "gpu": gpu,
                    "started_monotonic": time.monotonic(),
                }
                logger.info(
                    f"submitted fold {fid} -> physical GPU {gpu} "
                    f"(freeze_vae={freeze_vae})"
                )

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
                    status = results[fid].get("status", "unknown")
                    logger.info(f"fold {fid} finished with status={status}")
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
                            f"fold subprocess exited without returning a result "
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
                if elapsed > timeout_seconds:
                    logger.error(
                        f"fold {fid} exceeded per-fold timeout of "
                        f"{fold_timeout_hours:.2f}h; terminating subprocess"
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
                            f"fold exceeded timeout of {fold_timeout_hours:.2f} hours"
                        ),
                        "physical_gpu_id": int(slot["gpu"]),
                    }
                    result_queue.close()
                    del active[fid]
                    progressed = True

            if pending or active:
                time.sleep(0.5 if progressed else 1.0)

    finished = datetime.now(timezone.utc)
    ordered = [results[fid] for fid in sorted(results.keys())]

    # Persist results
    (run_dir / "kfold_results.json").write_text(
        json.dumps(_to_json_safe(ordered), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = _summarise_results(ordered)
    (run_dir / "kfold_summary.json").write_text(
        json.dumps(_to_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metadata = _execution_metadata(
        config_path=config_path,
        cuda_devices=cuda_devices,
        fold_ids=fold_ids,
        started=started,
        finished=finished,
    )
    (run_dir / "execution_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Cross-fold aggregation. Best-effort: a partial sweep should still
    # produce summary plots / JSON for the folds that did succeed.
    if any(r.get("status") == "ok" for r in ordered):
        try:
            from model.vae_teb_prediction.new_classifier.guid_cls_v1.aggregate_results import (  # noqa: WPS433
                aggregate_results,
            )

            decision_time_hours_cfg = (
                (config.get("evaluation", {}) or {}).get("decision_time_hours")
            )
            aggregate_results(
                run_dir=run_dir,
                fold_ids=None,
                decision_time_hours=(
                    float(decision_time_hours_cfg)
                    if decision_time_hours_cfg is not None
                    else None
                ),
            )
        except Exception as exc:  # pragma: no cover - aggregation should not block training
            logger.warning(f"aggregate_results failed: {exc}")

    logger.info(
        f"k-fold sweep done in {(finished - started).total_seconds():.1f}s; "
        f"results -> {run_dir / 'kfold_results.json'}"
    )
    return ordered


def _to_json_safe(obj: Any) -> Any:
    """Convert tensors / arrays / dataclasses to JSON-serialisable values."""
    import numpy as np  # local import avoids hard numpy dep at module load

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
    return obj


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector for testing.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Run k-fold guid_cls_v1 training in parallel across GPUs"
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
        help="Subset of fold ids to run (defaults to all configured folds)",
    )
    parser.add_argument(
        "--cuda-devices",
        type=int,
        nargs="*",
        default=None,
        help="Override CUDA device ids (round-robin per fold)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Max concurrent folds (defaults to config value)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Force sequential execution (for debugging)",
    )
    parser.add_argument(
        "--fold-timeout-hours",
        type=float,
        default=None,
        help="Per-fold wall-clock timeout (defaults to general_config.fold_timeout_hours)",
    )
    args = parser.parse_args(argv)

    run_kfold_parallel(
        config_path=args.config,
        output_dir_override=args.output_dir,
        fold_ids=args.fold_ids,
        cuda_devices=args.cuda_devices,
        max_parallel=args.max_parallel,
        sequential=bool(args.sequential),
        fold_timeout_hours=(
            float(args.fold_timeout_hours)
            if args.fold_timeout_hours is not None
            else None
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
