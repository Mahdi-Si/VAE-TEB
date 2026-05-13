"""Persistent logging helpers for ``guid_cls_v1``.

The training pipeline historically wrote every log line to ``stderr``
only — nothing was persisted to disk. This module bundles the tiny set
of helpers needed by :mod:`single_fold_trainer`,
:mod:`evaluate_guid_classifier`, :mod:`kfold_trainer` and
:mod:`diagnostics_callback` to produce a self-contained per-fold log
bundle under ``fold_{k}/logs/``.

The public surface is intentionally small:

* :func:`attach_fold_log_sinks` / :func:`detach_fold_log_sinks` — add /
  remove the per-fold loguru file sink and the ``sys.stdout`` /
  ``sys.stderr`` tee, returning an opaque handle that the caller passes
  straight back into ``detach_fold_log_sinks`` on teardown.
* :func:`dump_json` / :func:`append_jsonl` — small filesystem helpers
  used by the diagnostics callback and the trainer to write
  ``setup.json`` / ``epoch_summary.jsonl`` /
  ``stage_transitions.jsonl`` / ``evaluation_summary.jsonl``.
* :func:`to_json_safe` — moved from :mod:`kfold_trainer` so both the
  trainer and the new logging path can share a single implementation
  for ``np.float``/``np.ndarray``/``NaN`` coercion.

Design notes:

* The loguru file sink is added with ``enqueue=True`` so it is safe in a
  multiprocessing / spawn context (each fold subprocess attaches its
  own sink in its own interpreter; the parent process never writes to
  the per-fold sink).
* The ``sys.stdout`` / ``sys.stderr`` tee preserves the original
  streams as the *first* writer, so console output is untouched. Loguru's
  default sink stores its target stream at sink-add time, so the
  default ``stderr`` sink continues writing to the *real* stderr rather
  than the tee — this avoids duplicating loguru lines in ``fold.log``.
* Non-loguru output (``print()``, Python tracebacks, Lightning's
  progress bar, ``faulthandler`` dumps) flows through the tee and lands
  in ``fold.log`` as a raw fallback alongside the formatted loguru
  stream.
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, IO, Mapping, Optional

from loguru import logger


_DEFAULT_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{name}:{function}:{line} - {message}"
)


def _ensure_path(path: Any) -> Path:
    """Coerce ``str`` / ``Path`` to ``Path`` while preserving ``Path`` identity."""
    return path if isinstance(path, Path) else Path(path)


class _TeeStream:
    """Write-only tee that forwards every ``write`` / ``flush`` to two sinks.

    Used to wrap ``sys.stdout`` / ``sys.stderr`` so anything that bypasses
    loguru (raw ``print()``, Lightning's progress bar, low-level C
    tracebacks routed through :mod:`faulthandler`) still ends up in
    ``fold.log`` alongside the formatted loguru stream.

    The original stream stays the *primary* sink, so console UX is
    unchanged. Writes to the secondary file sink are guarded by a lock
    because multi-threaded code (DataLoader workers, Lightning's
    internals) can call ``write`` concurrently.
    """

    def __init__(self, primary: IO[str], secondary: IO[str]) -> None:
        self._primary = primary
        self._secondary = secondary
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        # Write to the original stream first so the console latency is
        # unchanged. The secondary write is best-effort: if the file
        # handle has been closed (e.g. because ``detach`` already ran on
        # a parallel thread), swallow the error rather than crash the
        # caller.
        primary_written = self._primary.write(data)
        with self._lock:
            try:
                self._secondary.write(data)
            except (ValueError, OSError):  # pragma: no cover - defensive
                pass
        return primary_written

    def flush(self) -> None:
        try:
            self._primary.flush()
        except (ValueError, OSError):  # pragma: no cover - defensive
            pass
        with self._lock:
            try:
                self._secondary.flush()
            except (ValueError, OSError):  # pragma: no cover - defensive
                pass

    def isatty(self) -> bool:  # pragma: no cover - delegated
        return getattr(self._primary, "isatty", lambda: False)()

    def fileno(self) -> int:  # pragma: no cover - delegated
        return self._primary.fileno()

    # Lightning's progress bar uses ``encoding`` / ``buffer`` on stderr
    # for unicode-aware writes. Forward the attribute lookups to the
    # primary stream so the tee is a near-transparent wrapper.
    def __getattr__(self, item: str) -> Any:
        return getattr(self._primary, item)


@dataclass
class FoldLogHandle:
    """Opaque token returned by :func:`attach_fold_log_sinks`.

    Holds the loguru sink id, the file handle backing ``fold.log``, and
    the references to ``sys.stdout`` / ``sys.stderr`` *before* the tee
    was installed so :func:`detach_fold_log_sinks` can restore them.
    """

    sink_id: Optional[int]
    log_file: Optional[IO[str]]
    original_stdout: Optional[IO[str]]
    original_stderr: Optional[IO[str]]
    log_path: Path


def attach_fold_log_sinks(
    fold_dir: Any,
    *,
    log_level: str = "INFO",
    capture_stdout_stderr: bool = True,
    filename: str = "fold.log",
    log_format: str = _DEFAULT_LOG_FORMAT,
) -> FoldLogHandle:
    """Attach a per-fold loguru file sink and optional stdout/stderr tee.

    The on-disk path is ``{fold_dir}/logs/{filename}``. The directory is
    created if needed. The loguru default ``stderr`` sink is left
    untouched — console output continues to render exactly as before.

    Args:
        fold_dir: Per-fold output directory (created if missing).
        log_level: Minimum loguru level routed to the file sink.
        capture_stdout_stderr: When True, wrap ``sys.stdout`` /
            ``sys.stderr`` in a :class:`_TeeStream` whose secondary
            target is the same ``fold.log`` file. When False, only the
            loguru sink is added (useful in tests that don't want global
            stream mutation).
        filename: Filename under ``fold_dir/logs/``. Defaults to
            ``fold.log``.
        log_format: Loguru format string.

    Returns:
        :class:`FoldLogHandle` — must be passed to
        :func:`detach_fold_log_sinks` in a ``finally`` block so the sink
        is removed and the original streams are restored even if the
        wrapped code raises.
    """
    fold_path = _ensure_path(fold_dir)
    logs_dir = fold_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / filename

    # ``enqueue=True`` makes the sink multi-process-safe (each fold
    # subprocess attaches its own sink in its own interpreter; this flag
    # only matters when multiple threads inside *this* process log
    # concurrently — DataLoader workers occasionally do).
    sink_id = logger.add(
        str(log_path),
        level=log_level,
        format=log_format,
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    original_stdout: Optional[IO[str]] = None
    original_stderr: Optional[IO[str]] = None
    log_file: Optional[IO[str]] = None
    if capture_stdout_stderr:
        # The tee target is a *separate* file handle from the one loguru
        # owns. Loguru manages its sink file lifecycle internally; mixing
        # writes via the same handle is unsafe.
        log_file = open(str(log_path), "a", encoding="utf-8", buffering=1)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _TeeStream(original_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = _TeeStream(original_stderr, log_file)  # type: ignore[assignment]

    logger.info(
        f"[logging] fold log sinks attached: file={log_path} "
        f"tee_stdout_stderr={capture_stdout_stderr} level={log_level}"
    )
    return FoldLogHandle(
        sink_id=sink_id,
        log_file=log_file,
        original_stdout=original_stdout,
        original_stderr=original_stderr,
        log_path=log_path,
    )


def detach_fold_log_sinks(handle: Optional[FoldLogHandle]) -> None:
    """Tear down whatever :func:`attach_fold_log_sinks` installed.

    Safe to call with ``None`` (no-op), so the caller's ``finally``
    block can use a single unconditional call.

    Args:
        handle: Value previously returned by
            :func:`attach_fold_log_sinks`, or ``None`` if attach failed.
    """
    if handle is None:
        return
    # Restore original stdout/stderr *before* removing the loguru sink
    # so any final ``logger.info`` from the cleanup itself still lands
    # in the file.
    if handle.original_stdout is not None:
        try:
            sys.stdout = handle.original_stdout
        except Exception:  # pragma: no cover - defensive
            pass
    if handle.original_stderr is not None:
        try:
            sys.stderr = handle.original_stderr
        except Exception:  # pragma: no cover - defensive
            pass
    if handle.sink_id is not None:
        try:
            logger.remove(handle.sink_id)
        except (ValueError, KeyError):  # pragma: no cover - defensive
            pass
    if handle.log_file is not None:
        try:
            handle.log_file.flush()
            handle.log_file.close()
        except (ValueError, OSError):  # pragma: no cover - defensive
            pass


def to_json_safe(obj: Any) -> Any:
    """Convert tensors / arrays / dataclasses to JSON-serialisable values.

    Moved from :mod:`kfold_trainer` so both the trainer and the new
    logging path can share a single implementation. NaN / infinity float
    values are coerced to ``None`` so ``json.dumps`` does not produce
    non-JSON literals (``NaN`` / ``Infinity`` are valid JS but invalid
    JSON, breaking strict downstream parsers).

    Args:
        obj: Anything — recursively walked.

    Returns:
        A pure-python value tree of dicts / lists / scalars suitable for
        ``json.dumps``.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):  # NaN / Inf
            return None
        return obj
    # ``torch.Tensor`` is the other tensor-like we hit. Importing torch
    # at module load would force a torch dep on consumers that only want
    # the pure-python helpers, so import lazily.
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return to_json_safe(obj.detach().cpu().item())
            return to_json_safe(obj.detach().cpu().tolist())
    except ImportError:  # pragma: no cover - torch is always present in this repo
        pass
    return obj


def dump_json(path: Any, payload: Any, *, indent: int = 2) -> None:
    """Write ``payload`` to ``path`` as JSON, JSON-safety-coerced.

    Args:
        path: Destination file path; parent directory is created.
        payload: Value to serialise. Run through :func:`to_json_safe`.
        indent: ``json.dumps`` indent; default 2 for human readability.
    """
    dest = _ensure_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(to_json_safe(payload), indent=indent, sort_keys=True),
        encoding="utf-8",
    )


def append_jsonl(path: Any, record: Mapping[str, Any]) -> None:
    """Append one JSON object to ``path`` as a single line.

    The file is opened in append mode so concurrent writers from the
    same process serialise via the OS (line-buffered append on a single
    file handle is atomic for writes < PIPE_BUF on POSIX; on Windows
    the standard library serialises this too). Across processes,
    ``logs/`` is per-fold so contention is not possible.

    Args:
        path: Destination file path; parent directory is created.
        record: Mapping; coerced via :func:`to_json_safe`.
    """
    dest = _ensure_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(to_json_safe(dict(record)), sort_keys=True)
    with open(str(dest), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


__all__ = [
    "FoldLogHandle",
    "attach_fold_log_sinks",
    "detach_fold_log_sinks",
    "to_json_safe",
    "dump_json",
    "append_jsonl",
]
