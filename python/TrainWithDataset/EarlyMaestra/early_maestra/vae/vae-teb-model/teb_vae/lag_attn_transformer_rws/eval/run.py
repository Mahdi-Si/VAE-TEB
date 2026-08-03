r"""The evaluation command line for the conv-Transformer model: a checkpoint in, ``summary.json`` out.

.. code-block:: bash

    python -m teb_vae.lag_attn_transformer_rws.eval.run --checkpoint <path> [--output-dir <dir>]

**Every line of the pipeline behind this is the sibling's.** ``teb_vae.lag_attn_rws.eval.run.main``
takes the model it is evaluating as a :class:`~teb_vae.lag_attn_rws.eval.binding.ModelBinding`, so
what this module supplies is :data:`~teb_vae.lag_attn_transformer_rws.eval.binding.TRF_BINDING` and
a ``prog=`` string. It is not a thin wrapper for tidiness: the two models exist to be compared, and
a second copy of the runner is how two things that must stay comparable stop being comparable --
the first fix to an analysis lands on one side, and the two summaries quietly stop meaning the same
thing.

What the sibling's module docstring says about a run therefore holds here verbatim: the resolved
config beside the checkpoint is the authority and the committed delta is merged over it, the loader
is forced single-process and fixed-seed shuffled, the forward pass happens once per run directory
and writes durable tables, every analysis runs inside a failure-isolating wrapper while preflight
deliberately does not, and a checkpoint is not required for a re-run against a finished directory.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

#: Repository root: ``teb_vae/lag_attn_transformer_rws/eval/run.py`` -> up four.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Launched as a script (an IDE's Run button) this file's own directory goes on sys.path instead
# of the repository root, and every absolute import below fails before __main__ is reached.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from loguru import logger  # noqa: E402

from teb_vae.lag_attn_rws.eval import run as shared_run  # noqa: E402
from teb_vae.lag_attn_transformer_rws.eval.binding import TRF_BINDING  # noqa: E402

#: Re-exported so a caller of this entry point does not have to reach into the sibling for the
#: names its own output directory is described by.
RESULTS_DIRNAME = shared_run.RESULTS_DIRNAME
SUMMARY_FILENAME = shared_run.SUMMARY_FILENAME
STEPS_FILENAME = shared_run.STEPS_FILENAME
LOG_FILENAME = shared_run.LOG_FILENAME

#: Steps that always run and are **not** selectable. The sibling's, unchanged: an unskippable step
#: describes the *data* rather than the model, so a model-specific addition here would be a
#: category error.
UNSKIPPABLE_ANALYSES: Dict[str, Any] = shared_run.UNSKIPPABLE_ANALYSES


def analysis_registry() -> Dict[str, Any]:
    """Return this model's analyses: the shared registry, then this binding's own.

    Derived on every call rather than frozen at import, so the help text below, the selection
    ``main`` makes and the ``summary.json`` record all read one mapping -- and so an analysis
    registered on the binding appears in all three by being registered in one place.

    Returns:
        The merged registry, in run order.
    """
    return shared_run.merged_analysis_functions(TRF_BINDING)


#: Selectable analysis names, in run order. Derived, never a literal: a hand-written list is one
#: that goes stale silently the first time the shared registry gains an entry.
ANALYSES: Tuple[str, ...] = tuple(analysis_registry())


def main(*args: Any, **kwargs: Any) -> int:
    """Evaluate a conv-Transformer checkpoint, or re-read a finished run of one.

    Delegates to :func:`teb_vae.lag_attn_rws.eval.run.main` with this model's binding. Every
    argument is that function's; see its docstring for what each one shapes.

    Args:
        *args: Positional arguments for the shared runner: ``checkpoint``, ``output_dir``.
        **kwargs: Keyword arguments for the shared runner. ``binding`` is supplied here and may be
            overridden only by a caller that means to evaluate a different model through this
            entry point -- which the offline-re-run tests do, to prove no model is built.

    Returns:
        The process exit code: non-zero when any step failed.
    """
    kwargs.setdefault("binding", TRF_BINDING)
    return shared_run.main(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser: the sibling's flags, this package's name.

    The flags are enumerated here rather than borrowed from the sibling's parser because
    ``--only`` and ``--skip`` interpolate *this* model's registry into their help text, and a
    borrowed parser would name the sibling's.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_attn_transformer_rws.eval.run",
        description=f"Evaluate a trained {TRF_BINDING.model_cls.__name__} checkpoint.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to the checkpoint to evaluate. Required unless --output-dir names a finished "
             "run whose tables the analyses can be re-run against.",
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="Run directory. Default: a timestamped directory under out_dir_base/<tag>-eval.",
    )
    parser.add_argument(
        "--overrides", default=None,
        help="Evaluation override delta merged over the checkpoint's own resolved config. "
             "Default: this package's committed eval_overrides.yaml.",
    )
    parser.add_argument(
        "--device", default=None, help="Torch device. Default: cuda:0 when available, else cpu."
    )
    parser.add_argument(
        "--num-samples", dest="num_samples", type=int, default=None,
        help="Monte Carlo draws per anchor. Default: eval_config.num_mc_samples.",
    )
    parser.add_argument(
        "--max-batches", dest="max_batches", type=int, default=None,
        help="Stop after this many batches. For a smoke run only.",
    )
    selectable = ", ".join(ANALYSES)
    parser.add_argument(
        "--only", default=None,
        help=f"Comma-separated analyses to run exclusively. Default: all of them. One or more "
             f"of: {selectable}.",
    )
    parser.add_argument(
        "--skip", default=None,
        help=f"Comma-separated analyses to skip. Default: skip none. One or more of: "
             f"{selectable}.",
    )
    return parser


def _cli(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and run. Returns the process exit code."""
    values = vars(build_parser().parse_args(argv))
    if values["checkpoint"] is None and not shared_run._finished_run(values["output_dir"]):
        raise SystemExit(
            "--checkpoint is required unless --output-dir names a finished run directory: the "
            "analyses read the tables the collection pass wrote, and there are none here to read."
        )
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        # The shard paths inside a resolved config are repo-root-relative for the tiny variant,
        # and a relative path resolved against an arbitrary working directory surfaces as "no
        # samples match the specified filters" with no mention of the real cause.
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)
    sources = {key: ("cli" if value is not None else "default") for key, value in values.items()}
    logger.info(
        "resolved arguments: "
        + ", ".join(f"{key}={values[key]!r} (from {sources[key]})" for key in sorted(values))
    )
    return main(**values, argument_sources=sources)


if __name__ == "__main__":
    sys.exit(_cli())
