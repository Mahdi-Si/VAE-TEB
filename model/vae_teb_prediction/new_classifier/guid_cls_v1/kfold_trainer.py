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


class KfoldFoldFailure(RuntimeError):
    """One or more folds did not return ``status="ok"``.

    Raised at the end of :func:`run_kfold_parallel` after results,
    summary and metadata JSON have been persisted (and best-effort
    aggregation has run on whatever folds did succeed). The orchestrator
    used to silently swallow per-fold exceptions into
    ``kfold_results.json`` and exit 0 — that masked real failures from
    CI / shell scripts. This exception now bubbles all the way out so a
    failing sweep produces a non-zero exit code.

    Attributes:
        failed_fold_ids: Sorted list of ints — fold ids whose status was
            not ``"ok"``.
        results_path: Absolute path to the per-fold ``kfold_results.json``
            (which contains the captured ``error`` / ``traceback`` for
            every failed fold).
    """

    def __init__(
        self,
        failed_fold_ids: Sequence[int],
        results_path: Path,
        first_traceback: str,
    ) -> None:
        self.failed_fold_ids = sorted(int(f) for f in failed_fold_ids)
        self.results_path = Path(results_path)
        msg = (
            f"{len(self.failed_fold_ids)} fold(s) failed: "
            f"{self.failed_fold_ids}. See {self.results_path} for full "
            f"per-fold error/traceback."
        )
        if first_traceback:
            msg += f"\n--- first failed-fold traceback (tail) ---\n{first_traceback[-2000:]}"
        super().__init__(msg)


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


def _attach_subprocess_raw_log(
    *,
    config_path: str,
    fold_id: int,
    output_dir_override: Optional[str],
) -> Any:
    """Open ``fold_{k}/logs/subprocess_raw.log`` and dup stdout/stderr into it.

    Runs early in :func:`_run_one_fold_subprocess` (after CUDA env var
    setup, before any torch import). Captures everything the OS-level
    file descriptors 1 / 2 see, including C-level prints from torch /
    h5py and ``faulthandler`` dumps on SIGTERM that loguru's
    Python-side sink cannot reach.

    **Subprocess-only.** The redirect is *permanent* for the current
    process — ``os.dup2`` rewires fd 1 / 2 and the fallback path
    rebinds ``sys.stdout`` / ``sys.stderr``; neither is restored on
    return. In sequential mode this function would otherwise be called
    in the **main** process and would steal the user's terminal for
    the rest of the sweep. The guard at the top of the function
    detects ``MainProcess`` via :mod:`multiprocessing` and short-circuits
    with ``None`` so sequential ``--sequential`` and direct
    :func:`train_fold` invocations keep their console output intact.
    The per-fold loguru file sink attached by :func:`train_fold` itself
    still captures Python-level output in that path.

    The function is defensive: any failure to locate the fold-dir or
    open the file is logged via ``print`` (loguru is not yet attached
    at this point in the lifecycle) and the function returns ``None``
    so the caller proceeds without raw capture.

    Args:
        config_path: Path to the YAML config (used to locate ``out_dir``).
        fold_id: 1-based fold id.
        output_dir_override: Optional run-dir override.

    Returns:
        The opened file handle, or ``None`` if attach failed or was
        skipped because we're running in the main process. The handle
        is *not* explicitly closed — the subprocess exit closes it.
    """
    import multiprocessing as _mp_check  # noqa: WPS433

    if _mp_check.current_process().name == "MainProcess":
        # Sequential mode runs ``_run_one_fold_subprocess`` directly in
        # the main process. A permanent stdout/stderr redirect there
        # would (a) replace the user's terminal for the rest of the
        # sweep, fold after fold, and (b) point the parent's kfold
        # status messages into the LAST fold's raw log. The loguru
        # file sink attached by ``train_fold`` itself still captures
        # all Python-level output to ``fold.log`` in this path, so
        # we don't lose observability — just the belt-and-braces
        # C-level capture.
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        if output_dir_override:
            run_dir = Path(output_dir_override).resolve()
        else:
            run_dir = _resolve_run_dir_lazy(cfg)
        fold_logs_dir = run_dir / f"fold_{int(fold_id)}" / "logs"
        fold_logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = fold_logs_dir / "subprocess_raw.log"
        raw_log = open(str(log_path), "a", encoding="utf-8", buffering=1)
        # Tee both streams: dup the OS-level fd so even C-level writes
        # are captured. We don't restore on exit because the subprocess
        # is about to die anyway.
        try:
            os.dup2(raw_log.fileno(), sys.stdout.fileno())
            os.dup2(raw_log.fileno(), sys.stderr.fileno())
        except (OSError, AttributeError):  # pragma: no cover - defensive
            # ``sys.stdout`` / ``sys.stderr`` may not have a fileno()
            # under some pytest captures / Windows multiprocessing
            # configurations. Fall back to attribute replacement so at
            # least Python-side writes are captured.
            sys.stdout = raw_log  # type: ignore[assignment]
            sys.stderr = raw_log  # type: ignore[assignment]
        return raw_log
    except Exception as exc:  # pragma: no cover - defensive
        try:
            print(
                f"[fold {fold_id}] _attach_subprocess_raw_log failed: {exc}",
                file=sys.__stderr__,
            )
        except Exception:
            pass
        return None


def _append_progress_log(
    *,
    run_dir: Path,
    result: Dict[str, Any],
) -> None:
    """Append a one-line summary of a fold's result to ``kfold_progress.log``.

    The progress log lives at ``run_dir/kfold_progress.log`` and gives
    the user an at-a-glance view of which folds have finished, their
    status, and headline metrics — without having to open each fold's
    ``fold_results.json``.

    Args:
        run_dir: Sweep-wide run directory.
        result: One per-fold result dict as returned by
            :func:`_run_one_fold_subprocess`.
    """
    try:
        ts = datetime.now(timezone.utc).isoformat()
        fold_id = int(result.get("fold_id", -1))
        status = str(result.get("status", "unknown"))
        bm = result.get("best_metrics") or {}
        train_seconds = result.get("train_seconds")
        train_seconds_str = (
            f"{float(train_seconds):.1f}s" if train_seconds is not None else "n/a"
        )
        snippet_parts = [
            f"{ts}",
            f"fold={fold_id}",
            f"status={status}",
            f"train={train_seconds_str}",
        ]
        for key in (
            "val/total_loss",
            "val/macro_f1",
            "val/binary_auroc",
            "val/acc_3",
            "val/acc_bin",
        ):
            val = bm.get(key)
            if val is None:
                continue
            try:
                snippet_parts.append(f"{key}={float(val):.4f}")
            except (TypeError, ValueError):
                continue
        if status != "ok":
            err = result.get("error")
            if err:
                snippet_parts.append(f"error={str(err)[:160]!r}")
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(
            str(run_dir / "kfold_progress.log"), "a", encoding="utf-8"
        ) as fh:
            fh.write(" | ".join(snippet_parts) + "\n")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"_append_progress_log failed: {exc}")


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
    # Disable HDF5 advisory file locking inside each fold subprocess. The
    # per-fold latent caches live on a shared NFS/isilon mount; with
    # default locking enabled, 8 folds × N DataLoader workers all calling
    # ``h5py.File(..., "r", swmr=True)`` can deadlock on the NFS lock
    # manager (the documented HDF5 workaround for NFS hangs — see HDF
    # Group FAQ). All accesses reachable from training are read-only, so
    # disabling advisory locks is safe. ``setdefault`` lets the user
    # override via shell env if a different storage backend ever needs
    # the locks back.
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    # Belt-and-braces stdout/stderr capture for the subprocess. The
    # primary log file is the loguru ``fold.log`` attached by
    # ``train_fold``; this raw capture catches anything that escapes
    # loguru (C-level prints from torch/h5py, faulthandler dumps on
    # SIGTERM, raw ``print()`` calls from third-party libraries).
    # Best-effort — failure to open the raw log MUST NOT prevent the
    # fold from running.
    raw_log_handle = _attach_subprocess_raw_log(
        config_path=config_path,
        fold_id=fold_id,
        output_dir_override=output_dir_override,
    )

    # Install a faulthandler that dumps a Python stack of every thread
    # to stderr on SIGTERM. When the orchestrator's watchdog terminates a
    # hung fold, this puts the exact deadlock site (h5py / DataLoader /
    # queue / anywhere in Python) into the fold log instead of leaving a
    # silent ``terminated`` exit code. Registered here — before any
    # torch/h5py import — so the handler is installed even if the hang
    # happens during library initialisation.
    import faulthandler
    import signal
    import sys

    faulthandler.enable(file=sys.stderr, all_threads=True)
    # ``faulthandler.register`` is POSIX-only (Windows has no equivalent
    # signal-driven dump). The training cluster is Linux, so this runs;
    # the ``hasattr`` guard keeps a Windows dev box from crashing here.
    if hasattr(faulthandler, "register") and hasattr(signal, "SIGTERM"):
        faulthandler.register(  # type: ignore[attr-defined]
            signal.SIGTERM, file=sys.stderr, all_threads=True, chain=False
        )

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
    # Make sure the feeder thread finishes flushing the pickled result
    # before the child exits — without close()+join_thread() a large
    # result dict could leave a half-written pipe and the parent would
    # see ``proc.is_alive()`` flip to False with no message waiting.
    result_queue.close()
    result_queue.join_thread()


def _summarise_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean/std across folds for the headline metrics.

    Args:
        results: List of per-fold result dicts.

    Returns:
        Summary dict suitable for ``kfold_summary.json``.
    """
    successes = [r for r in results if r.get("status") == "ok"]
    failures = [r for r in results if r.get("status") not in ("ok", "skipped")]
    skipped = [r for r in results if r.get("status") == "skipped"]

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
        "n_skipped": len(skipped),
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
        "skipped_fold_ids": sorted(int(r["fold_id"]) for r in skipped),
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
    fail_fast: bool = False,
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
        fail_fast: When True, stop scheduling new folds and terminate
            any active subprocesses as soon as a fold returns a non-ok
            status. Per-fold ``kfold_results.json`` is still written for
            whatever ran. The function still raises
            :class:`KfoldFoldFailure` afterwards — fail_fast only changes
            *whether the rest of the sweep is also run*. Default False
            (let every fold finish, then raise once).

    Returns:
        List of per-fold result dicts (sorted by ``fold_id``). Also writes
        ``kfold_results.json``, ``kfold_summary.json`` and
        ``execution_metadata.json`` under the run directory.

    Raises:
        :class:`KfoldFoldFailure` after persisting results and running
        best-effort aggregation if any fold did not return
        ``status="ok"``. This causes the CLI / parent process to exit
        non-zero instead of silently reporting "done" with failed folds
        buried in JSON.
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
            _append_progress_log(run_dir=run_dir, result=results[int(fid)])
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
                if fail_fast:
                    logger.error(
                        f"fail_fast=True — aborting sweep after fold {fid}; "
                        "remaining folds will not be run."
                    )
                    # Record skipped folds so they show up in
                    # ``kfold_results.json`` (and so the failed-fold list
                    # at the end is complete).
                    remaining = [
                        int(rid) for rid in fold_ids
                        if int(rid) not in results
                    ]
                    for skip_fid in remaining:
                        results[int(skip_fid)] = {
                            "fold_id": int(skip_fid),
                            "status": "skipped",
                            "error": (
                                "fold not run because fail_fast aborted the "
                                "sweep after an earlier failure"
                            ),
                            "physical_gpu_id": sequential_gpu,
                        }
                    break
    else:
        ctx = multiprocessing.get_context("spawn")
        pending: List[Tuple[int, int]] = [
            (int(fid), int(cuda_devices[i % len(cuda_devices)]))
            for i, fid in enumerate(fold_ids)
        ]
        active: Dict[int, Dict[str, Any]] = {}
        timeout_seconds = fold_timeout_hours * 3600.0
        # Set when ``fail_fast`` and the first non-ok fold result arrives.
        # Drains ``pending`` (so we stop scheduling new folds) and
        # terminates everything in ``active`` (so we stop wasting GPU
        # time on folds that are about to be discarded anyway).
        aborting = False

        while pending or active:
            while pending and len(active) < max_parallel and not aborting:
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
                    _append_progress_log(run_dir=run_dir, result=result)
                    status = results[fid].get("status", "unknown")
                    if status == "ok":
                        logger.info(f"fold {fid} finished with status={status}")
                    else:
                        # Surface the captured error + traceback to the
                        # console — otherwise the parallel branch silently
                        # swallows them into kfold_results.json and the user
                        # has no idea why the fold died.
                        err = results[fid].get("error", "<no error message>")
                        tb = results[fid].get("traceback", "")
                        logger.error(
                            f"fold {fid} FAILED: {err}\n"
                            f"--- traceback (tail) ---\n"
                            f"{tb[-2000:] if tb else '<no traceback>'}"
                        )
                        if fail_fast and not aborting:
                            aborting = True
                            logger.error(
                                f"fail_fast=True — aborting sweep after fold "
                                f"{fid}; {len(pending)} pending fold(s) will "
                                f"be skipped, {len(active) - 1} active "
                                f"fold(s) will be terminated."
                            )
                    result_queue.close()
                    # Best-effort: wait for the queue's background feeder
                    # thread to finish flushing on the *parent* side too.
                    # The child already calls ``join_thread()`` after its
                    # ``put()`` (see :func:`_fold_worker_entry`); this pair
                    # is the documented multiprocessing.Queue shutdown
                    # contract.
                    try:
                        result_queue.join_thread()
                    except Exception:  # pragma: no cover - defensive
                        pass
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
                    _append_progress_log(run_dir=run_dir, result=results[fid])
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
                    _append_progress_log(run_dir=run_dir, result=results[fid])
                    result_queue.close()
                    del active[fid]
                    progressed = True

            # fail_fast bookkeeping: record any pending folds we will not
            # run, and forcibly terminate everything still active.
            if aborting and pending:
                for skip_fid, skip_gpu in pending:
                    results[int(skip_fid)] = {
                        "fold_id": int(skip_fid),
                        "status": "skipped",
                        "error": (
                            "fold not run because fail_fast aborted the sweep "
                            "after an earlier failure"
                        ),
                        "physical_gpu_id": int(skip_gpu),
                    }
                pending.clear()
            if aborting and active:
                for fid in list(active.keys()):
                    slot = active[fid]
                    proc = slot["process"]
                    result_queue = slot["queue"]
                    logger.error(
                        f"fail_fast: terminating active fold {fid} "
                        f"(physical GPU {slot['gpu']})"
                    )
                    proc.terminate()
                    proc.join(timeout=10.0)
                    if proc.is_alive():  # pragma: no cover - defensive
                        proc.kill()
                        proc.join(timeout=5.0)
                    results[fid] = {
                        "fold_id": fid,
                        "status": "failed",
                        "error": "fold terminated by fail_fast abort",
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

    # Fail loudly if any fold did not return status="ok". Per-fold
    # results, summary and aggregation have already been persisted, so
    # raising here only changes the *exit code* — partial outputs are
    # still on disk for the user to inspect. Without this raise, the
    # CLI would exit 0 even with every fold failed, which masks the
    # failure from CI and shell scripts.
    failed = [r for r in ordered if r.get("status") != "ok"]
    if failed:
        first_tb = next(
            (str(r.get("traceback", "")) for r in failed if r.get("traceback")),
            "",
        )
        raise KfoldFoldFailure(
            failed_fold_ids=[int(r["fold_id"]) for r in failed],
            results_path=run_dir / "kfold_results.json",
            first_traceback=first_tb,
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
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Abort the sweep as soon as any fold fails (terminate active "
            "subprocesses, skip pending folds). Default: let every fold "
            "finish, then raise once at the end."
        ),
    )
    args = parser.parse_args(argv)

    try:
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
            fail_fast=bool(args.fail_fast),
        )
    except KfoldFoldFailure as exc:
        # The exception has already logged everything that matters
        # (per-fold tracebacks, summary). Print the final fault summary
        # and exit non-zero so CI / shell scripts pick up the failure.
        logger.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
