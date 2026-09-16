r"""The evaluation command line for this cell: a checkpoint in, a run directory out.

.. code-block:: bash

    python -m teb_vae.lag_slot_transformer_cfs.eval.run --checkpoint <path> [--output-dir <dir>]

From an IDE's Run button, with no command line: fill in ``RUN_ARGS`` at the bottom of this file.

**The runner is the family's.** :func:`teb_vae.lag_attn_cfs.eval.run.main` takes the model it is
evaluating as a :class:`~teb_vae.lag_attn_cfs.eval.binding.ModelBinding`, and what this module
supplies is :data:`~teb_vae.lag_slot_transformer_cfs.eval.binding.LAG_RESIDUAL_BINDING` and a
``prog=`` string. So a run of this cell is a run of the family's pipeline: the preflight guards and
their recovery table, the loader probe, the durable tables and the offline re-run against a
finished directory, every table-driven analysis of the family with its by-class and by-subgroup
variants, the headline, the sanity block and the artifact manifest -- and a directory of this cell
is read down the same layout as a lag-attentive cell's.

**What is this cell's own is declared on the binding rather than written here.** The collection
pass, because the shared one is written against the lag-attention forward and this architecture
computes none of what it reads; and thirteen analyses -- the arms, the lag suppression, the
resolved axes, the four that read the lag structure off this cell's own sidecars, three of the
family's own reused where the columns are the same quantities, and this cell's own pages, traces
and attributions under the family's three names. The six analyses the family has that read an
attention distribution or a per-lag divergence allocation are absent, and the summary names each
with the tensor it would have needed and the analysis of this cell that asks its question.

The one thing this module owns is the **registry it names in its own help text**: ``--only`` and
``--skip`` interpolate the analyses *this* binding resolves to, which is why the parser is
enumerated here rather than borrowed.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

#: Repository root: ``teb_vae/lag_slot_transformer_cfs/eval/run.py`` -> up four.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Launched as a script (an IDE's Run button) this file's own directory goes on sys.path instead
# of the repository root, and every absolute import below fails before __main__ is reached.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from loguru import logger  # noqa: E402

from teb_vae.lag_attn_cfs.eval import run as shared_run  # noqa: E402
from teb_vae.lag_attn_cfs.eval.launch import missing_required  # noqa: E402
from teb_vae.lag_slot_transformer_cfs.eval.binding import LAG_RESIDUAL_BINDING  # noqa: E402

#: Re-exported so a caller of this entry point does not have to reach into the family's cell for
#: the names its own output directory is described by.
RESULTS_DIRNAME = shared_run.RESULTS_DIRNAME
SUMMARY_FILENAME = shared_run.SUMMARY_FILENAME
STEPS_FILENAME = shared_run.STEPS_FILENAME
LOG_FILENAME = shared_run.LOG_FILENAME

#: Steps that always run and are **not** selectable. The family's, unchanged: the target channel
#: map describes the *data* -- the shards' own ``sel_*`` provenance and per-block causal
#: attributes -- and this cell's attributions join their band tables through it.
UNSKIPPABLE_ANALYSES: Dict[str, Any] = shared_run.UNSKIPPABLE_ANALYSES


def analysis_registry() -> Dict[str, Any]:
    """Return this model's analyses: the shared registry less the exclusions, then this cell's own.

    Derived on every call rather than frozen at import, so the help text below, the selection
    ``main`` makes and the ``summary.json`` record all read one mapping.

    Returns:
        The merged registry, in run order.
    """
    return shared_run.merged_analysis_functions(LAG_RESIDUAL_BINDING)


#: Selectable analysis names, in run order. Derived, never a literal: a hand-written list is one
#: that goes stale silently the first time the shared registry gains an entry.
ANALYSES: Tuple[str, ...] = tuple(analysis_registry())


def main(*args: Any, **kwargs: Any) -> int:
    """Evaluate a lag-residual checkpoint, or re-read a finished run of one.

    Delegates to :func:`teb_vae.lag_attn_cfs.eval.run.main` with this model's binding. Every
    argument is that function's; see its docstring for what each one shapes.

    ``--checkpoint`` is the one value the pass cannot proceed without unless ``--output-dir``
    names a finished run whose tables the analyses re-read, and it is enforced **here**, after
    the launch dict has been consulted, rather than by argparse -- which fires before that dict
    is ever read and would make the Run button unusable whatever the dict said.

    Args:
        *args: Positional arguments for the shared runner: ``checkpoint``, ``output_dir``.
        **kwargs: Keyword arguments for the shared runner. ``binding`` is supplied here and may be
            overridden only by a caller that means to evaluate a different model through this
            entry point.

    Returns:
        The process exit code: ``2`` on a refusal, non-zero when any step failed, ``0`` otherwise.
        **Not** the sanity block's warning flag -- that asymmetry is why ``verify.py`` exists
        separately.
    """
    checkpoint = args[0] if args else kwargs.get("checkpoint")
    output_dir = args[1] if len(args) > 1 else kwargs.get("output_dir")
    if checkpoint is None and not shared_run._finished_run(output_dir):
        refusal = missing_required({"checkpoint": checkpoint}, ("checkpoint",))
        logger.error(
            f"{refusal} Alternatively name a finished run directory with --output-dir, whose "
            f"tables the analyses re-read with no model built."
        )
        return 2
    kwargs.setdefault("binding", LAG_RESIDUAL_BINDING)
    return shared_run.main(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser: the family's flags, this package's name and registry.

    The flags are enumerated here rather than borrowed from the family's parser because ``--only``
    and ``--skip`` interpolate *this* model's registry into their help text, and a borrowed parser
    would name the other cell's. No ``required=True`` and no non-``None`` default anywhere below:
    the first fires before the launch dict is consulted, and the second would make that key's
    launch-dict entry unreachable.

    Returns:
        The parser. Every ``dest`` is also a valid :data:`RUN_ARGS` key.
    """
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_slot_transformer_cfs.eval.run",
        description=f"Evaluate a trained {LAG_RESIDUAL_BINDING.model_cls.__name__} checkpoint.",
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
    # Derived from the registry rather than restated, so ``--help`` names exactly what is
    # registered today. An unskippable step is NOT valid for either flag: it always runs and is
    # not selectable, so naming it raises rather than being read as a typo.
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
    """Parse arguments, merge with :data:`RUN_ARGS`, and run. Returns the process exit code."""
    # The family's resolver, handed this package's parser: the merge rule, the source record and
    # the unknown-key refusal are one implementation, while the usage line and the ``--only`` help
    # name this model's entry point and this model's own registry.
    values, sources = shared_run.resolve_arguments(
        argv, run_args=RUN_ARGS, parser=build_parser()
    )
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        # The shard paths inside a resolved config are repo-root-relative for the tiny variant,
        # and a relative path resolved against an arbitrary working directory surfaces as "no
        # samples match the specified filters" with no mention of the real cause.
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)
    logger.info(
        "resolved arguments: "
        + ", ".join(f"{key}={values[key]!r} (from {sources[key]})" for key in sorted(values))
    )
    return main(**values, argument_sources=sources)


#: Values used for arguments absent from the command line -- i.e. an IDE's Run button.
#:
#: Keyed by argparse ``dest``. Resolution is per key, so varying only the checkpoint works without
#: editing anything else here, and a key that is not an argparse ``dest`` raises at startup.
#:
#: **Running this file directly needs exactly one of two things filled in below**: ``checkpoint``,
#: or an ``output_dir`` naming a finished run whose tables the analyses re-read. With both left
#: ``None`` the run refuses at startup, because there is then neither a model to collect the tables
#: with nor tables to read. Nothing else is required: the working directory is moved to the
#: repository root for you, and every other value falls back to the merged configuration.
#:
#: Do not add run settings here. The seed, the caps, the draw count and the lag bands belong in the
#: override delta (``configs/eval_overrides.yaml``), which is dumped into the run directory as the
#: durable record; a value injected from Python would appear in no artifact and could not be
#: recovered from the output afterwards.
RUN_ARGS: Dict[str, Any] = {
    "checkpoint": None,
    "output_dir": None,
    "overrides": None,
    "device": None,
    "num_samples": None,
    "max_batches": None,
    # Which analyses run. Both keys take a comma-separated string of registered names and both
    # default to ``None``, which runs **every** one of them, in registry order. ``band_partition``
    # runs regardless and is not selectable by either key. An unknown name raises at startup,
    # before the checkpoint is loaded, so a misspelling costs a parse rather than a first pass
    # over the shards.
    "only": None,
    "skip": None,
}


if __name__ == "__main__":
    sys.exit(_cli())
