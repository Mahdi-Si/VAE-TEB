r"""What the evaluation split actually yields, before a run pays for a model.

.. code-block:: bash

    python -m teb_vae.lag_attn_transformer_rws.eval.probe \
        --config <run>/model_checkpoints/resolved_config.yaml

The probe is entirely model-independent -- it reads the loader, not the network -- so all of it is
the sibling's. What this module supplies is the default override delta, which is *not* shared: the
two packages ship two committed deltas, and probing a config against the wrong one would report the
wrong split's composition and say nothing about it.

``eval_overrides.yaml`` names this command in its header as the way to check a config before paying
for a run, which is the reason the entry point exists at all rather than the sibling's being used
directly.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

#: Repository root: ``teb_vae/lag_attn_transformer_rws/eval/probe.py`` -> up four.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Launched as a script (an IDE's Run button) this file's own directory goes on sys.path instead
# of the repository root, and every absolute import below fails before __main__ is reached.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from loguru import logger  # noqa: E402

from teb_vae.lag_attn_rws.eval import probe as shared_probe  # noqa: E402
from teb_vae.lag_attn_transformer_rws.eval.binding import TRF_BINDING  # noqa: E402

#: Re-exported so a caller does not reach into the sibling for the name the probe writes under.
PROBE_FILENAME = shared_probe.PROBE_FILENAME


def main(
    config_path: Any,
    *,
    overrides: Optional[Any] = None,
    max_batches: Optional[int] = None,
    output_dir: Optional[Any] = None,
) -> Dict[str, Any]:
    """Load a config, merge this package's evaluation overrides over it, and probe the split.

    Args:
        config_path: Run config to probe; normally the ``resolved_config.yaml`` beside the
            checkpoints.
        overrides: Override delta. Defaults to this package's committed one, which is the only
            thing about the probe that is not shared.
        max_batches: Stop after this many batches. A prefix over the concatenated shards.
        output_dir: Where to write ``loader_probe.json``; nothing is written when omitted.

    Returns:
        The probe record.
    """
    return shared_probe.main(
        config_path,
        overrides=TRF_BINDING.overrides_path if overrides is None else overrides,
        max_batches=max_batches,
        output_dir=output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser: the sibling's flags, this package's name."""
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_attn_transformer_rws.eval.probe",
        description="Report what the evaluation split actually yields. No model, no checkpoint.",
    )
    parser.add_argument(
        "--config", required=True,
        help="Run config to probe; normally the resolved_config.yaml beside the checkpoints.",
    )
    parser.add_argument(
        "--overrides", default=None,
        help="Evaluation override delta merged over the config. Default: this package's "
             "committed eval_overrides.yaml.",
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help=f"Directory to write {PROBE_FILENAME} into. Default: report only.",
    )
    parser.add_argument(
        "--max-batches", dest="max_batches", type=int, default=None,
        help="Stop after this many batches. A prefix over the concatenated shards; smoke only.",
    )
    return parser


def _cli(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and run. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        # Shard paths inside a config are repo-root-relative for the tiny variant, and a relative
        # path resolved against an arbitrary working directory surfaces as "no samples match the
        # specified filters" with no mention of the real cause.
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)
    record = main(
        args.config,
        overrides=args.overrides,
        max_batches=args.max_batches,
        output_dir=args.output_dir,
    )
    print(shared_probe.format_cohort_table(record))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
