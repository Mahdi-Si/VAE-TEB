r"""Run a list of classifier configs one after another, then compare the finished runs (SPEC §13.3, §17).

    python -m teb_vae.classifier.sweep --configs teb_vae/classifier/configs/regularised.yaml \
        teb_vae/classifier/configs/src_mu_prior.yaml ... \
        [--set classifier.data.kfold_root=/path ...] [--devices cuda:0,...] [--stage all] [--folds 1,2] \
        [--out DIR] [--tag batch1]

or edit :data:`RUN_ARGS` at the bottom and press Run. Each config runs in its own process
(``python -m teb_vae.classifier.run --config <cfg> --stage <stage> --run-dir <out_root>/<stamp>-<run.name>`` with
every ``--set`` override), so one run's state never leaks into the next. A run that did not reach its prediction and
metrics tables is recorded and left out of the comparison; the rest go to ``run.py compare`` (the first finished run
is the reference; at most six runs per comparison, so a longer list is compared in groups that share the reference)
into ``--out`` (default ``<out_root>/compare-<stamp>-<tag>``). ``sweep.json`` there maps every config to its run
directory, exit code and wall time. The process exit code is the largest of the runs' and the comparisons'.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if not __package__:  # run as a file (IDE Run button): the repo root first, never this directory
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _SCRIPT_DIR]
    if _REPO_ROOT in sys.path:
        sys.path.remove(_REPO_ROOT)
    sys.path.insert(0, _REPO_ROOT)

from loguru import logger  # noqa: E402

from teb_vae.classifier.config import load, resolve_path  # noqa: E402
from teb_vae.classifier.run import compare  # noqa: E402
from teb_vae.lag_attn_cfs.eval.launch import resolve_launch_args  # noqa: E402

#: ``compare`` takes 2-6 runs (§11.8): a longer sweep is compared in groups of the reference plus this many others.
COMPARE_OTHERS = 5
#: A run enters the comparison when these exist (its ``evaluate`` finished), whatever its exit code.
FINISHED = ("predictions/guids.parquet", "evaluation/tables/metrics.parquet")


def finished(run_dir: Path) -> bool:
    """``run_dir`` holds the prediction and metrics tables ``compare`` reads."""
    return all((run_dir / f).is_file() for f in FINISHED)


def main(*, configs: Sequence[str], overrides: Optional[List[str]] = None, stage: str = "all",
         folds: Optional[str] = None, device: Optional[str] = None, devices: Optional[str] = None,
         out: Optional[str] = None, tag: str = "sweep") -> int:
    """Run every config (in order), then compare the finished runs; returns the largest exit code.

    Args:
        configs: YAML paths, relative to the repository root or absolute; the first is the comparison's reference.
        overrides: ``classifier.x.y=value`` strings passed to every run (paths, out_root, num_workers, ...).
        stage: The stage each run executes (``all`` for a full run).
        folds: Comma-separated fold ids overriding ``run.folds`` in every run.
        device: ``run.device`` override (not digested).
        devices: Comma-separated device slots (fold-parallel train and predict, §14.4).
        out: The comparison directory; default ``<out_root>/compare-<stamp>-<tag>`` under the first config's
            ``run.out_root``.
        tag: A short name for the comparison directory.
    """
    if not configs:
        raise ValueError("sweep needs at least one config")
    overrides, stamp = list(overrides or []), time.strftime("%Y-%m-%d--%H-%M-%S")
    records: List[Dict[str, Any]] = []
    out_root = resolve_path(load(resolve_path(configs[0]), overrides).classifier.run.out_root)
    for cfg_path in configs:
        cfg = load(resolve_path(cfg_path), overrides + ([f"classifier.run.folds=[{folds}]"] if folds else []))
        root = resolve_path(cfg.classifier.run.out_root)
        run_dir = root / f"{time.strftime('%Y-%m-%d--%H-%M-%S')}-{cfg.classifier.run.name}"
        cmd = [sys.executable, "-m", "teb_vae.classifier.run", "--config", str(cfg_path), "--stage", stage,
               "--run-dir", str(run_dir)]
        for o in overrides:
            cmd += ["--set", o]
        for flag, value in (("--folds", folds), ("--device", device), ("--devices", devices)):
            if value:
                cmd += [flag, value]
        logger.info(f"sweep: {cfg_path} -> {run_dir}")
        started = time.perf_counter()
        code = subprocess.run(cmd, cwd=_REPO_ROOT).returncode
        records.append({"config": str(cfg_path), "run_dir": str(run_dir), "exit_code": int(code),
                        "finished": finished(run_dir), "seconds": round(time.perf_counter() - started, 1)})
        (logger.info if code == 0 else logger.warning)(f"sweep: {cfg_path} exit code {code} in {records[-1]['seconds']} s")
    out_dir = Path(out) if out else out_root / f"compare-{stamp}-{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep.json").write_text(json.dumps(records, indent=2))
    runs = [Path(r["run_dir"]) for r in records if r["finished"]]
    codes = [r["exit_code"] for r in records]
    if len(runs) < 2:
        logger.warning(f"sweep: {len(runs)} finished run(s); nothing to compare (sweep.json in {out_dir})")
        return max(codes, default=0)
    reference, others = runs[0], runs[1:]
    for i in range(0, len(others), COMPARE_OTHERS):
        group = [reference, *others[i:i + COMPARE_OTHERS]]
        target = out_dir if len(others) <= COMPARE_OTHERS else out_dir / f"group_{i // COMPARE_OTHERS + 1}"
        logger.info(f"sweep: compare {[g.name for g in group]} -> {target}")
        codes.append(int(compare(group, target)))
    return max(codes)


def build_parser() -> argparse.ArgumentParser:
    """Every default is None so :data:`RUN_ARGS` fills what the command line leaves out."""
    parser = argparse.ArgumentParser(description="Run several classifier configs and compare them (SPEC §17).")
    parser.add_argument("--configs", nargs="+", metavar="YAML", help="configs in run order; the first is the reference")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY.PATH=VALUE")
    parser.add_argument("--stage", choices=["cohort", "extract", "train", "predict", "evaluate", "report", "verify", "all"])
    parser.add_argument("--folds", help="comma-separated fold ids, e.g. 1,2,3")
    parser.add_argument("--device")
    parser.add_argument("--devices", help="fold-parallel slots, e.g. cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--out", help="comparison directory (default <out_root>/compare-<stamp>-<tag>)")
    parser.add_argument("--tag")
    return parser


def _cli(argv: Optional[List[str]] = None) -> int:
    values, _ = resolve_launch_args(build_parser(), RUN_ARGS, sys.argv[1:] if argv is None else list(argv))
    return main(**{key: value for key, value in values.items() if value is not None})


#: Arguments for an argument-less launch (the IDE Run button): edit the values below and press Run. On a command-line
#: launch the command line wins per key; a key left at None here is simply not set.
RUN_ARGS: Dict[str, Any] = {
    # Configs in run order (relative to the repo root). The first is the comparison's reference. This list is the
    # first batch of the 2026-10-07 plan (SPEC §17): the regularised baseline, then one source, representation,
    # context and label ablation each (lab_time_matched: the stage confound, added 2026-10-08); every other
    # configs/*.yaml of §13.3 can be appended (the adapt_*.yaml ones are slow).
    "configs": [
        "teb_vae/classifier/configs/regularised.yaml",
        "teb_vae/classifier/configs/src_mu_prior.yaml",
        "teb_vae/classifier/configs/src_mu_prior_target_state.yaml",
        "teb_vae/classifier/configs/src_st_ph.yaml",
        "teb_vae/classifier/configs/ctx_off.yaml",
        "teb_vae/classifier/configs/lab_time_matched.yaml",
    ],
    # None or a list of "classifier.<path>=<value>" strings applied to every run: the paths a real run needs,
    #   ["classifier.data.kfold_root=/data/.../k_fold_cross_validation_dataset",
    #    "classifier.source.vae.checkpoint=/runs/.../model_checkpoints/best.ckpt",
    #    "classifier.source.hdf5.stats_path=/data/.../stats.hdf5",
    #    "classifier.run.out_root=/data/.../classifier"]
    "overrides": None,
    # The stage every run executes: "all" runs (or resumes) everything.
    "stage": "all",
    # None = every fold of each config; a comma-separated subset applies to all runs, e.g. "1,2,3".
    "folds": None,
    # None keeps run.device; "cpu" or "cuda:<k>" (extract; train and predict too while `devices` is None).
    "device": None,
    # None runs folds serially; a comma-separated list of device slots runs one fold process per slot, e.g.
    # "cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7".
    "devices": None,
    # None writes the comparison to <out_root>/compare-<stamp>-<tag>; a path writes it there.
    "out": None,
    # A short name for the comparison directory.
    "tag": "batch1",
}


if __name__ == "__main__":
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        os.chdir(_REPO_ROOT)
    sys.exit(_cli())
