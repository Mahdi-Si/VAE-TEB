r"""The evaluation command line for the patch cell: a checkpoint in, ``summary.json`` out.

.. code-block:: bash

    python -m teb_vae.lag_attn_transformer_patch.eval.run --checkpoint <path> [--output-dir <dir>]
        [--overrides teb_vae/lag_attn_transformer_patch/eval/configs/planted_overrides.yaml]

Every line of the pipeline is the shared CFS runner's (:func:`teb_vae.lag_attn_cfs.eval.run.main`)
under :data:`~teb_vae.lag_attn_transformer_patch.eval.binding.TRF_PATCH_BINDING`. This module owns the
parser (``--only``/``--skip`` name *this* binding's registry) and :data:`RUN_ARGS`. The run contract
is the shared one: the checkpoint's resolved config plus the override delta, a single-process
fixed-seed loader, a dense forward, tables collected once per run directory, every analysis
failure-isolated, and ``--output-dir <finished run>`` re-running the analyses with no checkpoint.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

#: Repository root: ``teb_vae/lag_attn_transformer_patch/eval/run.py`` -> up four.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Launched as a script (an IDE's Run button), this directory goes on sys.path instead of the root.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from loguru import logger  # noqa: E402

from teb_vae.lag_attn_cfs.eval import run as shared_run  # noqa: E402
from teb_vae.lag_attn_transformer_patch.eval.binding import TRF_PATCH_BINDING  # noqa: E402

RESULTS_DIRNAME = shared_run.RESULTS_DIRNAME
SUMMARY_FILENAME = shared_run.SUMMARY_FILENAME


def analysis_registry() -> Dict[str, Any]:
    """This binding's merged registry, in run order (shared, ported, new, then ``cross_subgroup``)."""
    return shared_run.merged_analysis_functions(TRF_PATCH_BINDING)


#: Selectable analysis names, in run order. Derived, never a literal.
ANALYSES: Tuple[str, ...] = tuple(analysis_registry())


def main(*args: Any, **kwargs: Any) -> int:
    """Evaluate a patch checkpoint, or re-read a finished run. Arguments are the shared runner's.

    Returns:
        The process exit code: non-zero when any step failed (not the sanity flag; see ``verify``).
    """
    kwargs.setdefault("binding", TRF_PATCH_BINDING)
    return shared_run.main(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    """The shared flags, this package's ``prog`` and this binding's registry in the help text."""
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_attn_transformer_patch.eval.run",
        description=f"Evaluate a trained {TRF_PATCH_BINDING.model_cls.__name__} checkpoint.",
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint to evaluate. Required unless --output-dir is a finished run.")
    parser.add_argument("--output-dir", dest="output_dir", default=None,
                        help="Run directory. Default: a timestamped directory under "
                             "out_dir_base/<tag>-eval.")
    parser.add_argument("--overrides", default=None,
                        help="Override delta merged over the checkpoint's resolved config. Default: "
                             "this package's configs/eval_overrides.yaml.")
    parser.add_argument("--device", default=None, help="Torch device. Default: cuda:0, else cpu.")
    parser.add_argument("--num-samples", dest="num_samples", type=int, default=None,
                        help="Monte Carlo draws per anchor. Default: eval_config.num_mc_samples.")
    parser.add_argument("--max-batches", dest="max_batches", type=int, default=None,
                        help="Stop after this many batches. For a smoke run only.")
    selectable = ", ".join(ANALYSES)
    parser.add_argument("--only", default=None,
                        help=f"Comma-separated analyses to run exclusively. One or more of: {selectable}.")
    parser.add_argument("--skip", default=None,
                        help=f"Comma-separated analyses to skip. One or more of: {selectable}.")
    return parser


def _cli(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and run. Returns the process exit code."""
    values, sources = shared_run.resolve_arguments(argv, run_args=RUN_ARGS, parser=build_parser())
    if values["checkpoint"] is None and not shared_run._finished_run(values["output_dir"]):
        raise SystemExit(
            "--checkpoint is required unless --output-dir names a finished run directory."
        )
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        # Shard paths inside a resolved config are repo-root-relative for the fixture runs.
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)
    logger.info(
        "resolved arguments: "
        + ", ".join(f"{key}={values[key]!r} (from {sources[key]})" for key in sorted(values))
    )
    return main(**values, argument_sources=sources)


#: Values for arguments absent from the command line (an IDE's Run button), keyed by argparse
#: ``dest``. Fill ``checkpoint`` (or a finished ``output_dir``). Run settings belong in the
#: override delta, which is dumped into the run directory, not here.
RUN_ARGS: Dict[str, Any] = {
    "checkpoint": None,
    "output_dir": None,
    "overrides": None,
    "device": None,
    "num_samples": None,
    "max_batches": None,
    "only": None,
    "skip": None,
}


if __name__ == "__main__":
    sys.exit(_cli())
