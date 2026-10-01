r"""Fold-parallel execution (SPEC §14.4): one spawned process per fold, one device slot per process.

``run.devices`` lists the slots, e.g. ``[cuda:0, cuda:1, …, cuda:7]``. :func:`run_folds` starts the next pending fold
whenever a slot frees up, so $K$ folds on $G$ slots finish in $\lceil K/G \rceil$ rounds at most. A device may repeat
(``[cuda:0, cuda:0]``) to run two folds on one GPU; the frozen-feature classifier is small enough for that.

Each fold runs in a fresh ``spawn`` process: its own CUDA context, its own loguru sinks, and a crash (OOM kill,
segfault, a deadlock the operator kills) takes only that fold down. The process

- sends its stdout and stderr, C-level writes included, to one file (Lightning progress bars, h5py and torch warnings
  and a Python traceback on an exception all land there; the terminal stays readable with eight folds running);
- disables HDF5 advisory file locking (``HDF5_USE_FILE_LOCKING=FALSE`` unless the shell set it). Many processes, each
  with loader workers, opening the same files on an NFS mount can deadlock in the NFS lock manager; every access a fold
  process makes is a read (the cache and the shards), so the locks protect nothing here. The previous classifier
  pipeline hit exactly this;
- makes its device the current CUDA device (``torch.cuda.set_device``), so nothing it runs can allocate on
  ``cuda:0`` by default.

What a fold process may write is the caller's contract: ``run.py`` lets it write only under its own fold's
directories, and the parent is the only writer of run-level files (``stage_state.json``, ``manifest.json``,
``kfold_progress.log``, ``predictions/``).
"""
from __future__ import annotations

import faulthandler
import importlib
import multiprocessing
import os
import pickle
import sys
import time
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from loguru import logger


def run_folds(target: str, folds: Sequence[int], devices: Sequence[str], args: Sequence[Any],
              log_path: Callable[[int], Path], on_exit: Optional[Callable[[int, int], None]] = None) -> Dict[int, int]:
    """Run ``target(fold, device, *args)`` once per fold, in a fresh process, at most one process per device slot.

    Args:
        target: ``"package.module:function"``, imported inside the fold process (a string, so the parent's
            ``__main__`` never has to be importable by that name).
        folds: Fold ids, started in this order.
        devices: Device slots, each ``cpu`` or ``cuda:<k>``; a repeated device takes one fold per repetition.
        args: Further positional arguments of ``target``. They are pickled to ``<log stem>.args.pkl`` beside the log
            and read back by the fold process, so keep them plain data (dicts, strings, numbers).
        log_path: Maps a fold to the file its process's stdout and stderr are appended to.
        on_exit: Called in this process as ``on_exit(fold, exit_code)`` as soon as each fold process has ended,
            before the next fold starts on the freed slot.

    Returns:
        ``{fold: exit code}``, 0 iff ``target`` returned.

    Raises:
        ValueError: If ``devices`` is empty.
    """
    if not devices:
        raise ValueError("run_folds needs at least one device slot")
    ctx = multiprocessing.get_context("spawn")  # fork is unsafe once the parent has touched CUDA
    pending, free, running, codes = list(folds), list(devices), {}, {}
    try:
        while pending or running:
            while pending and free:
                fold, device = pending.pop(0), free.pop(0)
                path = Path(log_path(fold))
                path.parent.mkdir(parents=True, exist_ok=True)
                # the arguments go through a file: pickled into the spawn pipe, anything above the pipe buffer (a few
                # KB on Windows, 64 KB on Linux) blocks start() until the child has imported torch, which staggers
                # every fold's start by that import time
                args_path = path.with_name(f"{path.stem}.args.pkl")
                args_path.write_bytes(pickle.dumps(tuple(args)))
                proc = ctx.Process(target=_fold_process, args=(target, fold, device, str(args_path), str(path)),
                                   name=f"classifier-fold{fold}")
                proc.start()
                running[proc.sentinel] = (proc, fold, device, time.perf_counter())
                logger.info(f"fold {fold}: {target} on {device} (pid {proc.pid}); output in {path}")
            for sentinel in wait(list(running)):
                proc, fold, device, started = running.pop(sentinel)
                proc.join()
                codes[fold] = proc.exitcode
                free.append(device)
                (logger.info if proc.exitcode == 0 else logger.error)(
                    f"fold {fold}: {target} on {device} ended with exit code {proc.exitcode} after "
                    f"{time.perf_counter() - started:.0f} s")
                if on_exit is not None:
                    on_exit(fold, proc.exitcode)
    finally:  # an interrupt or an on_exit error must not leave orphaned fold processes holding GPUs
        for proc, fold, _, _ in running.values():
            logger.warning(f"fold {fold}: terminating its process (pid {proc.pid})")
            proc.terminate()
            proc.join()
    return codes


def _fold_process(target: str, fold: int, device: str, args_path: str, log_path: str) -> None:
    """Body of a fold process: output to ``log_path``, HDF5 locking off, ``device`` current, then ``target`` on the
    arguments pickled at ``args_path`` (deleted once read)."""
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # HDF5 reads it at each file open: set before any
    out = open(log_path, "a", buffering=1)
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.flush()
    os.dup2(out.fileno(), 1)  # C-level writes (torch, h5py, CUDA) as well as Python's
    os.dup2(out.fileno(), 2)
    faulthandler.enable(out, all_threads=True)  # a hard crash leaves every thread's stack in the log
    print(f"fold {fold}: {target} on {device}, pid {os.getpid()}", flush=True)
    if device != "cpu":
        import torch

        torch.cuda.set_device(device)
    args = pickle.loads(Path(args_path).read_bytes())
    Path(args_path).unlink()
    module, _, name = target.partition(":")
    getattr(importlib.import_module(module), name)(fold, device, *args)
