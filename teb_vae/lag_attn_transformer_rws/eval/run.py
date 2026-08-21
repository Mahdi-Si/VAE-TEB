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
    # The sibling's resolver, handed this package's parser: the merge rule, the source record and
    # the unknown-key refusal are one implementation, while the usage line and the ``--only`` help
    # name this model's entry point and this model's own registry.
    values, sources = shared_run.resolve_arguments(
        argv, run_args=RUN_ARGS, parser=build_parser()
    )
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
#: Do not add run settings here. The seed, the caps and the draw count belong in the override
#: delta (``configs/eval_overrides.yaml``), which is dumped into the run directory as the durable
#: record; a value injected from Python would appear in no artifact. In particular
#: ``caps.encoder_attention`` lives there -- absent, this model's own analysis records a skip.
RUN_ARGS: Dict[str, Any] = {
    "checkpoint": None,
    "output_dir": None,
    "overrides": None,
    "device": None,
    "num_samples": None,
    "max_batches": None,
    # Which analyses run. Both keys take a comma-separated string of the names below --
    # ``"forecast,encoder_attention"`` -- and both default to ``None``, which runs **every** one of
    # them, in the order listed. `band_partition` runs regardless and is not selectable by either
    # key; naming it raises rather than being read as a typo. An unknown name raises at startup
    # too, before the checkpoint is loaded, so a misspelling costs a parse rather than a first pass
    # over the shards.
    #
    # The first seventeen are the sibling's, one implementation shared by both models; the
    # eighteenth is this architecture's own and runs last, because the merge appends it after the
    # shared registry rather than reordering a run order both models share. `band_partition` makes
    # nineteen steps in a full run and is the one that is not on this list.
    #
    #   forecast:          Is the forecast any good. Skill against persistence, climatology and
    #                      the segment's own mean, in nats and in bpm, resolved by horizon step.
    #   coupling:          What the source added. `pred_gap` per recording in both estimators,
    #                      with a paired Wilcoxon, bootstrap intervals and the positive fraction.
    #   perm_control:      Is it *this* recording's source. The GUID-aware shuffle control, whose
    #                      verdict is three losses and deliberately not the KL.
    #   latent:            The per-dimension KL spectrum and active dimensions -- plus the
    #                      prior-variance-pinned detector that catches an inflated coupling number.
    #   lag_kl:            Where in the past the source informed the future. The per-lag KL
    #                      attribution in its raw, support-corrected and untruncated forms.
    #   attention:         The *lag cross-attention* per head, and its entropy against the ceiling
    #                      truncated lag support actually allows rather than against log L.
    #   calibration:       Is the decoder's learned variance the spread of its own errors. PIT,
    #                      coverage, CRPS. An `mse` checkpoint records a skip.
    #   residual:          How far apart the two forecasts are, in bpm, and the two latent-drift
    #                      quantities behind them.
    #   coherence:         The forecast in the frequency domain, resolved by lead time: coherence,
    #                      spectral gain, phase, and an exact split of the residual spectrum.
    #   distributions:     The shape of each metric over 20-minute segments, by cohort. Histograms
    #                      at both levels; descriptive only, and deliberately tests nothing.
    #   trajectory:        The readouts against time -- within one segment, and assembled across a
    #                      whole delivery on the absolute time axis.
    #   time_to_delivery:  The readouts binned on a 0.5 h grid of time before delivery,
    #                      class-stratified, with Holm across windows.
    #   second_stage:      The same two readouts on the *other* clinical clock -- signed hours from
    #                      the onset of the second stage, negative before it and positive after --
    #                      over the recordings that have a recorded onset. Its Holm family is this
    #                      clock's own, and it declares itself capped because it scores a subset.
    #   events:            What the raw target unlocks. Deceleration forecast skill, the
    #                      contraction-triggered response, and contraction-conditioned coupling.
    #   sufficiency:       What the latent bottleneck costs, against an evaluation-only oracle
    #                      decoder. The one analysis whose cost is a training loop, not a forward.
    #   samples:           Per-recording diagnostic PDF pages -- a stratified draw, plus the
    #                      extremes of each headline metric. Needs a checkpoint; skips without one.
    #   cross_subgroup:    Do the cohorts actually differ. Kruskal, Holm, then Mann-Whitney, over
    #                      the per-recording CSVs the analyses above it wrote, so it runs last of
    #                      the shared ones.
    #   encoder_attention: **This model's own.** What the two *encoder* self-attentions attend to:
    #                      per-head entropy against its truncation-aware ceiling, attention mass by
    #                      temporal distance, and the measured source reach against the lag range.
    #                      Runs its own bounded pass, so it needs a checkpoint and a loader, and it
    #                      needs `caps.encoder_attention` in the override delta -- absent, it
    #                      records a skip naming the key. Read it beside `attention`, which
    #                      profiles a different mechanism entirely.
    "only": None,
    "skip": None,
}


if __name__ == "__main__":
    sys.exit(_cli())
