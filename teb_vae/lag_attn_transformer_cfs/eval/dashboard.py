r"""The run dashboard's entry point for this cell: a finished run directory in, ``dashboard.html`` out.

.. code-block:: bash

    python -m teb_vae.lag_attn_transformer_cfs.eval.dashboard --output-dir <run directory>

The page is built by :mod:`teb_vae.lag_attn_cfs.eval.run_dashboard`, which reads the tables of a
results directory and nothing else. Those tables have one layout for both cfs cells, because both
cells run one pipeline, so this module supplies a ``prog=`` string and its own :data:`RUN_ARGS`
and defines no panel. The evaluation runner writes the same page at the end of every pass; use
this entry point to rebuild the page without a checkpoint, for example after a change to a panel.

Edit :data:`RUN_ARGS` at the bottom of the file and press the IDE's Run button, or give the run
directory on the command line.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

#: Repository root: ``teb_vae/lag_attn_transformer_cfs/eval/dashboard.py`` -> up four.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Launched as a script (an IDE's Run button) this file's own directory goes on sys.path instead
# of the repository root, and every absolute import below fails before __main__ is reached.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from loguru import logger  # noqa: E402

from teb_vae.lag_attn_cfs.eval import run_dashboard as shared_dashboard  # noqa: E402

#: Re-exported so a caller of this entry point need not reach into the cfs cell for the name of
#: the file it writes.
DASHBOARD_FILENAME = shared_dashboard.DASHBOARD_FILENAME


def main(output_dir: Any) -> Dict[str, Any]:
    """Write the dashboard of one finished run of this cell.

    Args:
        output_dir: The run directory given to ``eval/run.py`` as ``output_dir``, or its
            ``eval_results`` folder.

    Returns:
        The record of :func:`teb_vae.lag_attn_cfs.eval.run_dashboard.write_dashboard`: the file,
        the number of cards, and the cards whose input the directory does not hold.
    """
    return shared_dashboard.write_dashboard(shared_dashboard.resolve_results_dir(output_dir))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser. Every ``dest`` is also a valid :data:`RUN_ARGS` key."""
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_attn_transformer_cfs.eval.dashboard",
        description="Write the interactive dashboard of a finished evaluation run.",
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="The run directory given to eval/run.py as --output-dir, or its eval_results folder.",
    )
    return parser


def _cli(argv: Optional[List[str]] = None) -> int:
    """Parse arguments, fall back to :data:`RUN_ARGS`, and write the page."""
    values = vars(build_parser().parse_args(argv))
    unknown = sorted(set(RUN_ARGS) - set(values))
    if unknown:
        raise SystemExit(f"RUN_ARGS has keys that are not arguments: {unknown}")
    sources = {key: "cli" if values[key] is not None else "RUN_ARGS" for key in values}
    for key, value in RUN_ARGS.items():
        if values.get(key) is None:
            values[key] = value
    if values["output_dir"] is None:
        raise SystemExit("--output-dir is required: name a finished evaluation run directory.")
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        # A relative run directory is read from the repository root, as eval/run.py reads it.
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)
    logger.info(f"resolved arguments: output_dir={values['output_dir']!r} (from {sources['output_dir']})")
    record = main(values["output_dir"])
    return 0 if record["n_panels"] else 1


#: Values used for arguments absent from the command line, which is the IDE's Run button.
#:
#: ``output_dir`` is the run directory the evaluation wrote into: the same value as the
#: ``output_dir`` of ``eval/run.py``. A relative path is read from the repository root.
RUN_ARGS: Dict[str, Any] = {
    "output_dir": "tmp/checkpoint-2026-11-09/eval",
}


if __name__ == "__main__":
    sys.exit(_cli())
