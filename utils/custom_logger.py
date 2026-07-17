"""Loguru-based console + file logging, safe under Lightning's DDP launcher.

Three sinks, each with a distinct job:

* **console** (rank $0$ only) — the live view a human watches during a run.
* **``full.log``** (every rank) — the human-readable dump kept on disk.
* **``run.jsonl``** (every rank) — the same records as JSON Lines, for machine
  post-processing (``jq``, pandas, log shippers).

Design notes that are load-bearing, not stylistic:

**Why one file per rank.** ``strategy="ddp"`` re-executes the training script once
per GPU, so every rank independently calls :func:`setup_logging`. Loguru's
``enqueue=True`` is *not* a fix for that: it only synchronises processes that
**inherit** an already-constructed handler (fork / pickle), because the handler
nulls out its sink on pickling so exactly one process owns the file. Independently
spawned ranks each build their own queue, writer thread, and rotation state pointed
at the same inode — which loguru's maintainer describes as a configuration that
"may create corrupted logs", and which is the documented cause of rotation races
(loguru issues #220, #225, #1216). Suffixing the filename per rank gives every file
a single writer, which makes ``rotation``/``retention``/``compression`` correct
again and costs nothing. Rank $0$ keeps the unsuffixed name so ``tail -f full.log``
does the obvious thing. This mirrors Detectron2's ``setup_logger``.

**Why ``enqueue=False``.** With one writer per file it buys no safety, costs ~2.6x
per call, and can block the calling thread (i.e. the training step) once its
64 KB pipe fills.

**Why ``diagnose=False`` on disk.** ``diagnose`` renders the value of every local
variable in every traceback frame. A traceback raised inside ``training_step`` has
the input ``batch`` in scope, so on this repo's data that writes patient signal
into a plaintext log. Loguru's own docs: "This should be set to ``False`` in
production to avoid leaking sensitive data." It defaults to ``True``, so it must be
set explicitly. It stays available on the (non-persisted) rank-$0$ console.

lean-limit: the run is assumed to be launched by Lightning's ``ddp`` subprocess
launcher or a single process; under ``torchrun`` every rank is launched
independently and would need ``RANK`` exported by the job script (already handled
by :func:`resolve_global_rank`, which reads it when present).
"""

from __future__ import annotations

import inspect
import logging as _logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

# Rank env vars in Lightning's own precedence order (``_get_rank`` in
# ``lightning/fabric/utilities/rank_zero.py``). ``RANK`` is set by torchrun/SLURM;
# Lightning's own launcher sets only ``LOCAL_RANK``.
_RANK_ENV_KEYS = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")

# Rank-tagged so a merged view (``cat full.log*``) stays attributable.
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | rank{extra[rank]} | "
    "{module}:{function}:{line} - {message}\n"
)
_CONSOLE_FORMAT = (
    "{time:HH:mm:ss} | <level>{level: <8}</level> | {module}:{line} - {message}\n"
)

# Libraries that grab their own logging handler at import time and stop propagating
# to root. Lightning is the one that matters here — ``lightning/pytorch/__init__.py``
# does, at import:
#
#     if not _root_logger.hasHandlers():
#         _logger.addHandler(logging.StreamHandler())
#         _logger.propagate = False
#
# so unless root already had a handler when ``import lightning`` ran, every Trainer
# message ("GPU available", "LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES", checkpoint
# notices) goes to Lightning's private stderr handler and never reaches an
# InterceptHandler on root. Since imports land long before the run configures
# logging, that branch is the norm, not the exception.
_IMPORT_TIME_LOGGERS = (
    "lightning",
    "lightning.pytorch",
    "lightning.fabric",
    "pytorch_lightning",
)


def _reclaim_library_loggers() -> None:
    """Undo import-time logger grabs so records reach the root InterceptHandler.

    Drops each library's own handler (which would otherwise also double-print to
    the console) and restores ``propagate`` so the record reaches root, where the
    interception is installed. Safe when the library was never imported —
    ``getLogger`` just creates an inert placeholder.
    """
    for name in _IMPORT_TIME_LOGGERS:
        lib_logger = _logging.getLogger(name)
        lib_logger.handlers.clear()
        lib_logger.propagate = True


@dataclass(frozen=True)
class LoggingPaths:
    """Where :func:`setup_logging` actually wrote its sinks.

    Returned so the caller can upload the files (e.g. as MLflow artifacts) without
    re-deriving the rank suffix.

    Attributes:
        text_log: Path of the human-readable sink, or ``None`` if disabled.
        json_log: Path of the JSON Lines sink, or ``None`` if disabled.
        rank: The global rank this process resolved to.
    """

    text_log: Optional[str]
    json_log: Optional[str]
    rank: int


def resolve_global_rank() -> int:
    """Return this process's global rank from the launcher's environment.

    Reads the environment rather than ``torch.distributed`` because this runs at
    script start, long before a process group exists. Returns $0$ when no launcher
    variable is set — which is both the single-process case and, under Lightning's
    ``ddp`` launcher, the parent (it spawns children for ranks $1..N-1$ only and
    keeps rank $0$ for itself).

    Returns:
        The resolved global rank, or $0$ when no rank variable is present.
    """
    for key in _RANK_ENV_KEYS:
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue  # a malformed value must not crash logging setup
    return 0


def rank_suffixed_path(path: str, rank: int) -> str:
    """Append a ``.rank{n}`` suffix for non-zero ranks (Detectron2's scheme).

    Rank $0$ keeps the plain name so the common ``tail -f full.log`` works; other
    ranks sort adjacent to it and each own a file exclusively.
    """
    return path if rank == 0 else f"{path}.rank{rank}"


class InterceptHandler(_logging.Handler):
    """Route stdlib ``logging`` records into loguru, preserving level and traceback.

    This is what pulls Lightning's, MLflow's, and torch's own messages into the
    same sinks. Follows loguru's documented recipe: map the level by *name* with a
    numeric fallback, and walk back to the true caller frame — skipping both the
    stdlib ``logging`` internals and frozen ``importlib`` frames — so
    ``{module}:{function}:{line}`` names the real origin instead of ``logging``.
    """

    def emit(self, record: _logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno  # a custom numeric level loguru does not know

        frame, depth = inspect.currentframe(), 0
        while frame:
            filename = frame.f_code.co_filename
            is_logging = filename == _logging.__file__
            is_frozen = "importlib" in filename and "_bootstrap" in filename
            if depth > 0 and not (is_logging or is_frozen):
                break
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(
    *,
    log_to_file: bool = True,
    log_to_console: bool = True,
    file_path: str = "app.log",
    json_path: Optional[str] = None,
    file_level: str = "DEBUG",
    console_level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
    compression: Optional[str] = "zip",
    serialize: bool = False,
    backtrace: bool = True,
    diagnose: bool = False,
    console_diagnose: bool = False,
    console_rank_zero_only: bool = True,
    rank: Optional[int] = None,
) -> LoggingPaths:
    """Configure loguru's sinks for one process of a training run.

    Idempotent: it routes through ``logger.configure(handlers=...)``, which removes
    every previously installed handler (including loguru's default stderr sink,
    which is auto-installed in each freshly spawned rank). Calling it twice replaces
    the configuration rather than duplicating every line.

    Args:
        log_to_file: Write the human-readable ``file_path`` sink.
        log_to_console: Write a coloured stderr sink (subject to
            ``console_rank_zero_only``).
        file_path: Path of the human-readable sink; suffixed per rank.
        json_path: When set, also write JSON Lines here (suffixed per rank).
        file_level: Minimum level for both file sinks.
        console_level: Minimum level for the console sink.
        rotation: Size/time trigger for rotating the file sinks.
        retention: How long rotated files are kept.
        compression: Compression applied to rotated files (``None`` disables).
        serialize: Render the ``file_path`` sink as JSON too. Prefer ``json_path``,
            which keeps the human-readable sink human-readable.
        backtrace: Extend tracebacks beyond the catch point.
        diagnose: Render local-variable values in tracebacks **on the file sinks**.
            Defaults to ``False``: these files persist, and the values include the
            input batch. See the module docstring.
        console_diagnose: Same, for the non-persisted console sink.
        console_rank_zero_only: Suppress the console on ranks $> 0$; otherwise $N$
            ranks interleave on one terminal. The per-rank files still capture
            every rank, which is what a post-mortem of a hung rank needs.
        rank: Override the resolved global rank (testing seam).

    Returns:
        A :class:`LoggingPaths` naming the files actually written.
    """
    resolved_rank = resolve_global_rank() if rank is None else rank

    # Loguru types this as a TypedDict union; a plain mapping is the documented
    # call shape, so the dicts are built untyped and handed straight to configure().
    handlers: List[Dict[str, Any]] = []
    text_log: Optional[str] = None
    json_log: Optional[str] = None

    if log_to_console and (resolved_rank == 0 or not console_rank_zero_only):
        handlers.append(
            dict(
                sink=sys.stderr,
                level=console_level,
                format=_CONSOLE_FORMAT,
                colorize=True,
                backtrace=backtrace,
                diagnose=console_diagnose,
                enqueue=False,
            )
        )

    if log_to_file:
        text_log = rank_suffixed_path(file_path, resolved_rank)
        handlers.append(
            dict(
                sink=text_log,
                level=file_level,
                format=_FILE_FORMAT,
                rotation=rotation,
                retention=retention,
                compression=compression,
                serialize=serialize,
                backtrace=backtrace,
                diagnose=diagnose,
                enqueue=False,
            )
        )

    if json_path:
        json_log = rank_suffixed_path(json_path, resolved_rank)
        handlers.append(
            dict(
                sink=json_log,
                level=file_level,
                format=_FILE_FORMAT,  # renders record["text"]; the record tree is separate
                rotation=rotation,
                retention=retention,
                compression=compression,
                serialize=True,
                backtrace=backtrace,
                diagnose=diagnose,
                enqueue=False,
            )
        )

    # ``extra`` lands verbatim in every serialized record, so the JSONL is
    # rank-attributable without parsing the rendered text.
    # type: ignore -- loguru's stubs narrow ``handlers`` to a TypedDict union, but the
    # documented runtime contract is a plain mapping of add() kwargs.
    logger.configure(handlers=handlers, extra={"rank": resolved_rank})  # type: ignore[arg-type]

    # Re-point stdlib logging at loguru. force=True clears handlers any library
    # installed via basicConfig at import time (otherwise interception silently
    # loses to whoever configured logging first); level=0 lets the loguru sinks own
    # filtering.
    _logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    _reclaim_library_loggers()

    # Lightning's rank_zero_warn (and torch deprecations) go through warnings.warn,
    # not logging, so without this they bypass the file sinks entirely.
    # The False/True pair is deliberate: captureWarnings(True) is a no-op once it has
    # stashed the original hook, so if anything restored warnings.showwarning in the
    # meantime (a warnings.catch_warnings block is enough) a bare re-arm would leave
    # the bridge uninstalled. Clearing first forces a genuine re-install.
    _logging.captureWarnings(False)
    _logging.captureWarnings(True)

    return LoggingPaths(text_log=text_log, json_log=json_log, rank=resolved_rank)
