r"""The acceptance gate for one run of this cell, its arm table, and the cross-cell comparison.

Three entry points, one module, all of them reading files a finished run left behind and nothing
else -- no model, no shard, no ``torch``, no GPU. A summary produced on the production box can
therefore be checked on any machine the file can be copied to.

**The gate**::

    python -m teb_vae.lag_attn_transformer_cfs.eval.verify <run>/eval_results/summary.json

Delegated to :mod:`teb_vae.lag_attn_cfs.eval.verify` in full. The criteria are the run's own exit
code, the weight-space load check, the named ``pred_gap`` column, the model's ten acceptance
verdicts and the sanity block -- every one of which is a property of the shared objective, the
shared target domain and the shared readout registry rather than of either encoder. A second copy
here would be a second set of thresholds for two cells that exist to be compared under one.

**The tables**::

    python -m teb_vae.lag_attn_transformer_cfs.eval.verify --runs <dir> --out RESULTS_arms.md

What is local is which axis this cell was swept on, and the comparison the package exists to make.
This cell ships **one** ``sweep_*.yaml`` arm -- ``anchor_stride`` -- against the conv-LSTM cell's
four, so its own sweep section is one table; beside it the cfs cell's cross-cell table puts runs of
*both* architectures side by side, keyed by the ``model_class`` each run recorded in its own
``run_context``.

The cfs cell's four sourcing rules hold here unchanged and are inherited rather than restated: arms
are keyed by the value read from each run's own ``resolved_config.yaml`` and never from a directory
name, the training series come from each run's own ``metrics_history.csv``, an arm that is collapsed
or whose verdict is not computable is **marked rather than dropped**, and everything numeric comes
from the summary's headline block.

:data:`~teb_vae.lag_attn_cfs.eval.verify.SELECTION_RULE` travels into the emitted document with the
cross-cell table, because the rule is what stops that table being read backwards: a stronger target
prior lowers the source-conditioned KL *without the coupling having weakened*, so KL magnitude ranks
nothing, and an arm whose base reconstruction is worse than the comparison cell's has not earned a
reading on ``pred_gap`` at all.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from teb_vae.lag_attn_cfs.eval import launch, verify as shared

#: The filenames and headline columns the shared module already owns. Bound rather than restated:
#: the two packages must disagree about none of them, and a binding cannot drift.
SUMMARY_FILENAME = shared.SUMMARY_FILENAME
RESOLVED_CONFIG_FILENAME = shared.RESOLVED_CONFIG_FILENAME
METRICS_HISTORY_FILENAME = shared.METRICS_HISTORY_FILENAME
ACTIVE_FRAC_COLUMN = shared.ACTIVE_FRAC_COLUMN
PRED_GAP_COLUMN = shared.PRED_GAP_COLUMN
D_BASE_COLUMN = shared.D_BASE_COLUMN
KL_COLUMN = shared.KL_COLUMN
CLOCK_MARGIN_COLUMN = shared.CLOCK_MARGIN_COLUMN
ANCHORS_PER_SAMPLE_COLUMN = shared.ANCHORS_PER_SAMPLE_COLUMN

#: Where ``--runs`` writes the tables when no destination is named. Applied after the launch merge
#: rather than as the parser's default, so a path in :data:`RUN_ARGS` is reachable.
DEFAULT_ARMS_OUT = "RESULTS_arms.md"

#: The one axis this cell ships an arm for. The other three the cfs cell sweeps -- the floor, the
#: horizon and the decoder depth -- are properties of the shared target-domain half rather than of
#: these encoders, so this package ships no arm on them and renders no section for them: a table
#: whose every row read ``(absent)`` would be a sweep nobody ran, printed as though somebody had.
SWEPT_ANCHOR_STRIDE = shared.SWEPT_ANCHOR_STRIDE


# =============================================================================
# The gate
# =============================================================================
def main(summary_path: Any, json_out: Optional[Any] = None) -> int:
    """Verify one run against the pre-registered criteria and print the report.

    Args:
        summary_path: Path to a run's ``summary.json``.
        json_out: Optional path to write the machine-readable report to.

    Returns:
        The process exit code: non-zero when any criterion failed. That asymmetry is the point of
        the gate -- the runner's own exit code reports whether a step raised, and a run whose
        every step completed while its sanity block failed exits 0 there and 1 here.
    """
    return shared.main(summary_path, json_out)


# =============================================================================
# The tables
# =============================================================================
def collect_arm(summary_path: Path, runs_dir: Path) -> Dict[str, Any]:
    """Assemble one run's record. The cfs cell's collector, unchanged.

    Every fact it reads -- the headline block, the resolved config, the training series, the
    collapse verdict and the recorded ``model_class`` -- is model-independent, and the cross-cell
    table's whole premise is that the two cells' rows were assembled the same way.

    Args:
        summary_path: The run's ``summary.json``.
        runs_dir: The directory the scan started from, for the display path.

    Returns:
        The arm record.
    """
    return shared.collect_arm(summary_path, runs_dir)


def build_arm_tables(arms: Sequence[Dict[str, Any]]) -> str:
    """Render this cell's sweep table and the cross-cell comparison as one document.

    Every arm appears in the sweep table, keyed by its value of that table's axis -- the cfs cell's
    rule and for its reason: which arms belong to which sweep is a fact about the directory an
    operator handed in, not something this module can infer, and a reader filtering a table by the
    other columns' defaults loses nothing while a module guessing wrong loses a row.

    Args:
        arms: The collected arm records.

    Returns:
        The whole document as markdown text.
    """
    lines: List[str] = [
        "# Arm comparison",
        "",
        f"Generated by `python -m teb_vae.lag_attn_transformer_cfs.eval.verify --runs ...` from "
        f"{len(arms)} finished run(s). Rows are keyed by the swept value read from each run's own "
        f"`{RESOLVED_CONFIG_FILENAME}`, never from its directory name. `pred_gap` here is the "
        f"`{PRED_GAP_COLUMN}` headline column. Epochs, the final `{ACTIVE_FRAC_COLUMN}` and the "
        f"collapse verdict come from each run's training `{METRICS_HISTORY_FILENAME}`.",
        "",
        "## Arm inventory",
        "",
    ]
    lines += shared.build_arm_inventory(arms)

    lines += ["", "## Anchor tiling sweep (`anchor_stride`)", ""]
    lines += [
        "The one axis this cell ships an arm for. The stride is a *training* setting: every run "
        f"here was evaluated at the dense anchor set, so `{ANCHORS_PER_SAMPLE_COLUMN}` should read "
        "identically down this column and a row that does not is a run scored over another "
        "population.",
        "",
    ]
    lines += shared._table(
        ["`anchor_stride`", f"`{ANCHORS_PER_SAMPLE_COLUMN}`", "Parameters", f"`{D_BASE_COLUMN}`",
         "`pred_gap`", f"`{KL_COLUMN}`", f"`{CLOCK_MARGIN_COLUMN}`", "Collapsed?", "Run"],
        shared._sweep_rows(
            arms, SWEPT_ANCHOR_STRIDE,
            lambda arm: [
                shared._headline_cell(arm, ANCHORS_PER_SAMPLE_COLUMN),
                shared._render(shared._or_absent(arm["n_parameters"])),
                shared._headline_cell(arm, D_BASE_COLUMN),
                shared._headline_cell(arm, PRED_GAP_COLUMN),
                shared._headline_cell(arm, KL_COLUMN),
                shared._headline_cell(arm, CLOCK_MARGIN_COLUMN),
                shared._collapsed_cell(arm),
            ],
        ),
    )

    lines += shared.build_cross_cell_table(arms)

    incomplete = [arm for arm in arms if arm["incomplete"]]
    if incomplete:
        lines += ["", "## Incomplete runs", ""]
        lines += [f"- `{arm['run']}`: " + "; ".join(arm["incomplete"]) for arm in incomplete]
    lines.append("")
    return "\n".join(lines)


def compare_arms(runs_dir: Any, out_path: Any) -> int:
    """Scan a directory of finished runs and write this cell's table and the cross-cell one.

    The scan, the empty-directory refusal, the collection and the console line are the cfs cell's;
    what this package supplies is :func:`build_arm_tables`, the one thing that genuinely differs.

    Args:
        runs_dir: Directory scanned recursively for ``summary.json`` files, one per run. Runs of
            **both** cfs cells are expected: the cross-cell table is the reason this command takes
            a directory rather than a list of this cell's arms.
        out_path: Where the markdown document is written.

    Returns:
        The process exit code: non-zero when no summary was found -- an empty comparison is an
        operator error, not an empty result.
    """
    return shared.compare_arms(runs_dir, out_path, build_arm_tables)


# =============================================================================
# Command line
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for both entry points."""
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_attn_transformer_cfs.eval.verify",
        description=(
            "Check a completed eval run against the acceptance criteria, or build this cell's "
            "arm table and the cross-cell comparison from a directory of finished runs."
        ),
    )
    parser.add_argument(
        "summary", nargs="?", default=None,
        help="Path to a run's summary.json (the acceptance gate).",
    )
    parser.add_argument(
        "--json-out", dest="json_out", default=None,
        help="Gate only: also write the machine-readable report here.",
    )
    parser.add_argument(
        "--runs", default=None,
        help="Directory of finished runs, of either or both cfs cells; emits the tables instead "
             "of the gate.",
    )
    parser.add_argument(
        # Defaulted to None rather than to the filename, so a value in RUN_ARGS can win: the
        # merge treats any non-None parsed value as having come from the command line, and an
        # argparse default would therefore make the launch dict's entry unreachable. The real
        # default is applied after the merge.
        "--out", default=None,
        help=f"Tables only: where to write them. Default: {DEFAULT_ARMS_OUT}.",
    )
    return parser


def _cli(argv: Optional[List[str]] = None) -> int:
    """Dispatch between the gate and the tables. Returns the process exit code."""
    values, _sources = launch.resolve_launch_args(build_parser(), RUN_ARGS, argv)
    out = values["out"] or DEFAULT_ARMS_OUT
    if values["runs"] is not None:
        if values["summary"] is not None:
            print("give either a summary path or --runs, not both.")
            return 2
        return compare_arms(values["runs"], out)
    if values["summary"] is None:
        print(
            "a summary path is required unless --runs names a directory of finished runs. Pass "
            "one, or -- to launch this file from an IDE's Run button -- set 'summary' or 'runs' "
            "in RUN_ARGS near the bottom of this module."
        )
        return 2
    return main(values["summary"], values["json_out"])


#: Values used for arguments absent from the command line -- i.e. an IDE's Run button.
#:
#: Keyed by argparse ``dest``. Resolution is per key, so a flag overrides one value and leaves the
#: rest standing, and a key that is not an argparse ``dest`` raises at startup.
#:
#: **Running this file directly needs exactly one of ``summary`` or ``runs``**, which is the same
#: choice the two entry points are: ``summary`` points at one run's ``summary.json`` and gates it,
#: exiting non-zero when a sanity check failed; ``runs`` points at a directory of finished runs of
#: **either or both** cfs cells and writes this cell's arm table and the cross-cell comparison to
#: ``out`` instead. Setting both is refused, because it would be ambiguous which one was meant.
#:
#: This entry point reads only files a run left behind -- no checkpoint, no shard, no ``torch``.
RUN_ARGS: Dict[str, Any] = {
    "summary": None,
    "json_out": None,
    "runs": None,
    "out": None,
}


if __name__ == "__main__":
    sys.exit(_cli())
