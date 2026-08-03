r"""The acceptance gate for one run, this model's arm tables, and the cross-model comparison.

Three entry points, one module, all of them reading files a finished run left behind and nothing
else -- no model, no shard, no ``torch``, no GPU. The layering test walks this module's imports with
``torch`` on its forbidden list, so a summary produced on the production box can be checked on any
machine the file can be copied to.

**The gate**::

    python -m teb_vae.lag_attn_transformer_rws.eval.verify <run>/eval_results/summary.json

Delegated to :mod:`teb_vae.lag_attn_rws.eval.verify` in full. The criteria are the run's own exit
code, the weight-space load check, the named ``pred_gap`` column, the model's acceptance verdicts
and the sanity block -- every one of which is a property of the shared objective and the shared
readout registry rather than of either encoder. A second copy here would be a second set of
thresholds for two models that exist to be compared under one.

**The tables**::

    python -m teb_vae.lag_attn_transformer_rws.eval.verify --runs <dir-of-runs> --out RESULTS_arms.md

What is local is which axes this model was swept on, and the comparison the package exists to make.
The arm tables cover the twelve shipped ``sweep_*.yaml`` arms -- the source-window family, the two
depth families, the stem arm, the feed-forward width and the reach budget -- and the cross-model
table puts runs of *both* architectures side by side, keyed by the ``model_class`` each run recorded
in its own ``run_context``.

The sibling's three sourcing rules hold here unchanged and are inherited rather than restated:
arms are keyed by the value read from each run's own ``resolved_config.yaml`` and never from a
directory name, the training series come from each run's own ``metrics_history.csv``, and an arm
that is collapsed or incomplete is **marked rather than dropped**. Everything numeric comes from
the summary's headline block, which is the one surface the reporting layer promises to keep
resolvable.

The cross-model table carries :data:`SELECTION_RULE` in the emitted document rather than only in
this docstring, because the rule is what stops the table being read backwards: a stronger target
prior lowers the source-conditioned KL *without the coupling having weakened*, so KL magnitude
ranks nothing, and an arm whose base reconstruction is worse than the comparison model's has not
earned a reading on ``pred_gap`` at all.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from teb_vae.lag_attn_rws.eval import launch, verify as shared

#: The filenames and headline columns the shared module already owns. Bound rather than restated:
#: the two packages must disagree about none of them, and a binding cannot drift.
SUMMARY_FILENAME = shared.SUMMARY_FILENAME
RESOLVED_CONFIG_FILENAME = shared.RESOLVED_CONFIG_FILENAME
METRICS_HISTORY_FILENAME = shared.METRICS_HISTORY_FILENAME
ACTIVE_FRAC_COLUMN = shared.ACTIVE_FRAC_COLUMN
PRED_GAP_COLUMN = shared.PRED_GAP_COLUMN

#: Where ``--runs`` writes the arm tables when no destination is named. Applied after the launch
#: merge rather than as the parser's default, so a path in ``RUN_ARGS`` is reachable.
DEFAULT_ARMS_OUT = "RESULTS_arms.md"

#: The headline columns the tables read, named here so a rename in the reporting seam fails a test
#: rather than silently emptying a column. ``tests/test_eval_verify.py`` pins each one against the
#: registry that produces it.
D_BASE_COLUMN = "d_base_mc_nats"
D_FULL_COLUMN = "d_full_mc_nats"
KL_COLUMN = "source_conditioned_kl_raw_nats"
LAG_PEAK_COLUMN = "kl_argmax_lag_step"
ACTIVE_DIMS_COLUMN = "kl_active_dims"

#: The measured source reach, from this package's own ``encoder_attention`` analysis. It is what
#: gives the window family a **measured** x-axis beside its configured one: a window is what the
#: arm was given, and these are what its encoder used.
REACH_MEDIAN_COLUMN = "encoder_attention_source_reach_median_steps"
REACH_P95_COLUMN = "encoder_attention_source_reach_p95_steps"


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
# This model's sweep axes
# =============================================================================
#: Dotted paths into a run's resolved config, one per sweep axis this model ships arms for. The
#: value at the path is the arm's key in its table.
#:
#: ``source_attention_window`` needs the ``_ABSENT`` distinction more than any axis the sibling
#: sweeps: its ``null`` is a **value** -- the unbounded source encoder, the arm the whole locality
#: family is measured against -- so an arm file that lost the key entirely must not key identically
#: to the one that deliberately sets it to nothing.
SWEPT_WINDOW = ("model_config", "VAE_model", "source_attention_window")
SWEPT_TARGET_BLOCKS = ("model_config", "VAE_model", "target_attention_blocks")
SWEPT_SOURCE_BLOCKS = ("model_config", "VAE_model", "source_attention_blocks")
SWEPT_D_FF = ("model_config", "VAE_model", "encoder_d_ff")

#: The reach budget's path is the sibling's, because the budget is a property of the shared feature
#: bank rather than of either encoder.
SWEPT_REACH = shared.SWEPT_REACH

#: The encoder knobs, rendered together: each shipped architecture arm flips one or two of them
#: against the baseline, and a row naming all seven is readable without knowing in advance which
#: one this arm flipped. The stem arm is the pair of empty kernel and dilation lists.
ARCHITECTURE_KEYS: Tuple[Tuple[str, ...], ...] = (
    ("model_config", "VAE_model", "encoder_conv_kernels"),
    ("model_config", "VAE_model", "encoder_conv_dilations"),
    ("model_config", "VAE_model", "encoder_num_heads"),
    ("model_config", "VAE_model", "encoder_d_ff"),
    ("model_config", "VAE_model", "target_attention_blocks"),
    ("model_config", "VAE_model", "source_attention_blocks"),
    ("model_config", "VAE_model", "source_attention_window"),
)


# =============================================================================
# Cross-model comparison
# =============================================================================
#: Where a run records which architecture produced it: the ``model_class`` the collection pass
#: copies out of the checkpoint's own stamp into ``summary.json``'s ``run_context``. The stamp is
#: written nowhere else a finished run keeps -- the dumped config carries every constructor keyword
#: and not the class they build.
MODEL_CLASS_PATH: Tuple[str, ...] = ("run_context", "model_class")

#: The comparison model, and this one. Restated as strings rather than imported from the two
#: bindings, which name ``torch`` modules; ``tests/test_eval_verify.py`` pins both against the
#: bindings so a class rename fails there rather than emptying this table.
BASELINE_MODEL_CLASS = "SeqVaeLagAttnRws"
THIS_MODEL_CLASS = "SeqVaeLagAttnTrfRws"

#: How a run that recorded no class is keyed. A run evaluated before the class was recorded, or
#: re-run offline against a finished directory with no checkpoint, genuinely does not know what
#: produced it -- and a row guessed from the directory name would be the one error this table's
#: keying rule exists to prevent.
UNRECORDED_MODEL = "(unrecorded)"

#: The selection rule, emitted into the document rather than only stated here. A table of two
#: architectures' KLs invites exactly the ranking this forbids.
SELECTION_RULE = (
    "Do not select on KL magnitude. A stronger target prior lowers the source-conditioned KL "
    "without the coupling having weakened, so a smaller KL is not a worse model and a larger one "
    "is not a better one. Select on KL that comes with source-specific predictive gain, and treat "
    f"a competitive `{D_BASE_COLUMN}` as a precondition: an arm whose base reconstruction is worse "
    f"than the comparison model's has not earned a reading on `pred_gap` at all."
)

#: The marker a row worse than the baseline on $D_0$ carries, and its footnote. A marked cell
#: rather than a dropped row: a suppressed arm reads as an arm that was never run.
_D0_MARKER = "[^d0]"
_D0_FOOTNOTE = (
    f"{_D0_MARKER}: this run's `{D_BASE_COLUMN}` is worse (higher) than the best "
    f"`{BASELINE_MODEL_CLASS}` run's, so its `pred_gap` is marked rather than ranked. A base "
    f"branch that reconstructs worse changes what the gap is a difference *of*, and a gap read "
    f"against a weaker baseline is not the same measurement as one read against a competitive "
    f"one."
)


def _or_absent(value: Any) -> Any:
    """Return ``value``, or the shared absence sentinel when it is ``None``.

    The tables distinguish "the run did not record this" from "the run recorded nothing here", and
    :func:`teb_vae.lag_attn_rws.eval.verify._render` spells the two differently.
    """
    return shared._ABSENT if value is None else value


def collect_arm(summary_path: Path, runs_dir: Path) -> Dict[str, Any]:
    """Assemble one run's record: the shared arm facts plus what the cross-model table adds.

    The shared collector reads the headline, the resolved config, the training series and the
    collapse verdict, all of which are model-independent. Two facts it does not read are needed
    here and are taken from the same summary in a second parse: which architecture produced the
    run, and whether the run's own ``argmax_lag`` check found the lag peak readable. A second
    parse of one JSON file is cheaper than a parameter threaded through the shared collector for
    one caller's benefit.

    Args:
        summary_path: The run's ``summary.json``.
        runs_dir: The directory the scan started from, for the display path.

    Returns:
        The shared arm record with ``model_class`` and ``lag_peak_check`` added. Nothing here
        raises on a partial run: an arm with problems is a row with its problems named.
    """
    arm = shared.collect_arm(summary_path, runs_dir)

    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    model_class = shared._dig(summary, *MODEL_CLASS_PATH)
    arm["model_class"] = str(model_class) if model_class else UNRECORDED_MODEL
    if not model_class:
        arm["incomplete"].append(
            "the summary's run_context records no model_class, so which architecture produced "
            "this run is not readable from the run"
        )
    # The run's own verdict on whether its lag peak means anything: an argmax pinned at 0 never
    # looked back, and one at the largest attainable lag is against the window edge. Re-read
    # rather than re-derived, for the reason the gate re-reads the model's verdicts.
    arm["lag_peak_check"] = shared._dig(summary, "results", "sanity", "checks", "argmax_lag")
    return arm


def _lag_peak_cell(arm: Dict[str, Any]) -> str:
    """Render the lag peak and the run's own readability verdict on it, never a bare number.

    A peak quoted without that verdict is the misreading the check exists to prevent: an argmax
    is defined on a flat profile and on a censored one exactly as it is on a real peak.
    """
    peak = shared._render(_or_absent(arm["headline"].get(LAG_PEAK_COLUMN)))
    check = arm.get("lag_peak_check")
    if not isinstance(check, dict) or not check:
        return f"{peak} (not checked)"
    verdict = str(check.get("verdict"))
    if verdict == "pass":
        return f"{peak} (inside the window)"
    if verdict == "fail":
        return f"{peak} (**degenerate**: {check.get('detail', 'no detail recorded')})"
    return f"{peak} (inconclusive: {check.get('detail', 'no detail recorded')})"


def baseline_d_base(arms: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Return the comparison model's best (lowest) $D_0$ across the collected runs.

    Best rather than mean or latest: the rule asks whether an arm's base reconstruction is
    competitive with what the comparison architecture *can* do, and the strongest baseline run in
    the directory is the honest answer to that.

    Args:
        arms: The collected arm records.

    Returns:
        The lowest finite ``d_base_mc_nats`` among runs whose ``model_class`` is the comparison
        model's, or ``None`` when the directory holds no such run -- in which case no row is
        flagged, because there is nothing to be worse than.
    """
    values = [
        value for value in (_finite_d_base(arm) for arm in arms if
                            arm["model_class"] == BASELINE_MODEL_CLASS)
        if value is not None
    ]
    return min(values) if values else None


def _finite_d_base(arm: Dict[str, Any]) -> Optional[float]:
    """Return this run's $D_0$ as a finite float, or ``None`` when it is not one.

    A non-finite $D_0$ is not a worse reconstruction, it is an unusable one -- the run's own
    ``headline_finite`` check is where that gets reported -- so it neither sets the threshold nor
    trips it.
    """
    value = arm["headline"].get(D_BASE_COLUMN)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _pred_gap_cell(arm: Dict[str, Any], baseline: Optional[float]) -> str:
    """Render ``pred_gap``, marked when this row's $D_0$ has not earned the reading."""
    cell = shared._headline_cell(arm, PRED_GAP_COLUMN)
    d_base = _finite_d_base(arm)
    earned = baseline is None or d_base is None or d_base <= baseline
    return cell if earned else f"{cell} {_D0_MARKER}"


def build_cross_model_table(arms: Sequence[Dict[str, Any]]) -> List[str]:
    r"""Render the comparison this package exists to make, as markdown lines.

    One row per run, keyed by the architecture that produced it, carrying what the architecture
    question is actually asked with: the parameter count the two must be compared per, $D_0$ and
    $D_1$, the coupling gap, the unfloored KL, the lag peak with its readability verdict, the
    final active fraction, the collapse verdict and the epoch count.

    Args:
        arms: The collected arm records.

    Returns:
        The section's lines, heading included.
    """
    baseline = baseline_d_base(arms)
    ordered = sorted(
        arms,
        key=lambda arm: (
            # The baseline architecture first, so the row the rule is stated against is the row a
            # reader meets first; everything else alphabetically by class, then by run.
            0 if arm["model_class"] == BASELINE_MODEL_CLASS else 1,
            arm["model_class"],
            arm["run"],
        ),
    )
    classes = sorted({arm["model_class"] for arm in arms})

    lines: List[str] = [
        "",
        "## Cross-model comparison",
        "",
        f"Rows keyed by the `model_class` each run recorded in its own `run_context`, never by "
        f"directory name. The comparison baseline is `{BASELINE_MODEL_CLASS}`; "
        f"`{THIS_MODEL_CLASS}` replaces both history encoders and changes nothing else, so every "
        f"number below is the same objective's.",
        "",
        SELECTION_RULE,
        "",
    ]
    if len(classes) < 2:
        lines += [
            f"**Only one architecture is present here** ({', '.join(classes) or 'none'}). The "
            f"table is emitted anyway -- a comparison with one side missing is a fact about the "
            f"directory that was handed in, and dropping the table would report it as a fact "
            f"about the models.",
            "",
        ]
    if UNRECORDED_MODEL in classes:
        lines += [
            f"`{UNRECORDED_MODEL}` rows are runs whose `run_context` carries no `model_class`: "
            f"evaluated before the class was recorded, or re-run offline against a finished "
            f"directory with no checkpoint to read the stamp from. They are keyed as unknown "
            f"rather than guessed from a directory name.",
            "",
        ]

    rows = [
        [
            arm["model_class"],
            shared._render(_or_absent(arm["n_parameters"])),
            shared._headline_cell(arm, D_BASE_COLUMN),
            shared._headline_cell(arm, D_FULL_COLUMN),
            _pred_gap_cell(arm, baseline),
            shared._headline_cell(arm, KL_COLUMN),
            _lag_peak_cell(arm),
            shared._render(_or_absent(arm["final_kld_active_frac"])),
            shared._collapsed_cell(arm),
            shared._render(_or_absent(arm["last_epoch"])),
            arm["run"],
        ]
        for arm in ordered
    ]
    lines += shared._table(
        ["Model", "Parameters", f"`{D_BASE_COLUMN}` ($D_0$)", f"`{D_FULL_COLUMN}` ($D_1$)",
         f"`{PRED_GAP_COLUMN}`", f"`{KL_COLUMN}`", "Lag peak", f"`{ACTIVE_FRAC_COLUMN}` (final)",
         "Collapsed?", "Epochs", "Run"],
        rows,
    )
    # The footnote ships only where a marker did, so a table with nothing flagged does not carry a
    # rule about flagging. The marker can appear only when a baseline run was found.
    if any(_D0_MARKER in row[4] for row in rows):
        lines += ["", _D0_FOOTNOTE]
    return lines


# =============================================================================
# The arm tables
# =============================================================================
def build_arm_tables(arms: Sequence[Dict[str, Any]]) -> str:
    """Render this model's sweep tables and the cross-model comparison as one document.

    Every arm appears in every sweep table, keyed by its value of that table's axis -- the
    sibling's rule and for its reason: which arms belong to which sweep is a fact about the
    directory an operator handed in, not something this module can infer, and a reader filtering a
    table by the other columns' defaults loses nothing while a module guessing wrong loses a row.

    Args:
        arms: The collected arm records.

    Returns:
        The whole document as markdown text.
    """
    lines: List[str] = [
        "# Arm comparison",
        "",
        f"Generated by `python -m teb_vae.lag_attn_transformer_rws.eval.verify --runs ...` from "
        f"{len(arms)} finished run(s). Rows are keyed by the swept value read from each run's own "
        f"`{RESOLVED_CONFIG_FILENAME}`, never from its directory name; every arm appears in every "
        f"sweep table under its value of that table's axis. `pred_gap` here is the "
        f"`{PRED_GAP_COLUMN}` headline column. Epochs, the final `{ACTIVE_FRAC_COLUMN}` and the "
        f"collapse verdict come from each run's training `{METRICS_HISTORY_FILENAME}`.",
        "",
        "## Arm inventory",
        "",
    ]
    lines += shared._table(
        ["Run", "Model", "Checkpoint epoch", "Trained epochs", "Parameters", "Eval exit",
         "Status"],
        [
            [
                arm["run"],
                arm["model_class"],
                shared._render(_or_absent(arm["train_epoch"])),
                shared._render(_or_absent(arm["last_epoch"])),
                shared._render(_or_absent(arm["n_parameters"])),
                shared._render(_or_absent(arm["exit_code"])),
                ("incomplete: " + "; ".join(arm["incomplete"])) if arm["incomplete"] else "ok",
            ]
            for arm in sorted(arms, key=lambda record: record["run"])
        ],
    )

    lines += ["", "## Source window sweep (`source_attention_window`)", ""]
    lines += [
        f"The configured window beside what the encoder measurably used: the two reach columns "
        f"come from this package's `encoder_attention` analysis, which reports the mass-weighted "
        f"distance quantiles composed with the stem's receptive field. They are `(missing)` on a "
        f"run whose `caps.encoder_attention` was unset, because that analysis records a skip "
        f"rather than a zero. `null` is the unbounded arm, which is a value and not an absence.",
        "",
    ]
    lines += shared._table(
        ["`source_attention_window`", "Measured reach (median steps)",
         "Measured reach (p95 steps)", f"`{D_BASE_COLUMN}`", "`pred_gap`", f"`{KL_COLUMN}`",
         "Collapsed?", "Run"],
        shared._sweep_rows(
            arms, SWEPT_WINDOW,
            lambda arm: [
                shared._headline_cell(arm, REACH_MEDIAN_COLUMN),
                shared._headline_cell(arm, REACH_P95_COLUMN),
                shared._headline_cell(arm, D_BASE_COLUMN),
                shared._headline_cell(arm, PRED_GAP_COLUMN),
                shared._headline_cell(arm, KL_COLUMN),
                shared._collapsed_cell(arm),
            ],
        ),
    )

    for heading, axis in (
        ("Target encoder depth sweep (`target_attention_blocks`)", SWEPT_TARGET_BLOCKS),
        ("Source encoder depth sweep (`source_attention_blocks`)", SWEPT_SOURCE_BLOCKS),
        ("Feed-forward width sweep (`encoder_d_ff`)", SWEPT_D_FF),
    ):
        lines += ["", f"## {heading}", ""]
        lines += shared._table(
            ["Value", "Parameters", f"`{D_BASE_COLUMN}`", "`pred_gap`", f"`{KL_COLUMN}`",
             "Collapsed?", "Run"],
            shared._sweep_rows(
                arms, axis,
                lambda arm: [
                    shared._render(_or_absent(arm["n_parameters"])),
                    shared._headline_cell(arm, D_BASE_COLUMN),
                    shared._headline_cell(arm, PRED_GAP_COLUMN),
                    shared._headline_cell(arm, KL_COLUMN),
                    shared._collapsed_cell(arm),
                ],
            ),
        )

    lines += ["", "## Reach budget sweep (`causal_reach_budget_s`)", ""]
    lines += shared._table(
        ["`causal_reach_budget_s`", "Channels kept (tgt/src)", f"`{D_BASE_COLUMN}`",
         "`pred_gap`", f"`{KL_COLUMN}`", "Collapsed?", "Run"],
        shared._sweep_rows(
            arms, SWEPT_REACH,
            lambda arm: [
                shared._channels_kept_cell(arm),
                shared._headline_cell(arm, D_BASE_COLUMN),
                shared._headline_cell(arm, PRED_GAP_COLUMN),
                shared._headline_cell(arm, KL_COLUMN),
                shared._collapsed_cell(arm),
            ],
        ),
    )

    lines += ["", "## Encoder architecture arms", ""]
    lines += shared._table(
        ["Knobs", "Parameters", f"`{D_BASE_COLUMN}`", "`pred_gap`", f"`{KL_COLUMN}`",
         f"`{ACTIVE_DIMS_COLUMN}`", "Collapsed?", "Run"],
        [
            [
                ", ".join(
                    f"{path[-1]}={shared._render(shared._dig_config(arm['config'], path))}"
                    for path in ARCHITECTURE_KEYS
                ),
                shared._render(_or_absent(arm["n_parameters"])),
                shared._headline_cell(arm, D_BASE_COLUMN),
                shared._headline_cell(arm, PRED_GAP_COLUMN),
                shared._headline_cell(arm, KL_COLUMN),
                shared._headline_cell(arm, ACTIVE_DIMS_COLUMN),
                shared._collapsed_cell(arm),
                arm["run"],
            ]
            for arm in sorted(arms, key=lambda record: record["run"])
        ],
    )

    lines += build_cross_model_table(arms)

    incomplete = [arm for arm in arms if arm["incomplete"]]
    if incomplete:
        lines += ["", "## Incomplete runs", ""]
        lines += [f"- `{arm['run']}`: " + "; ".join(arm["incomplete"]) for arm in incomplete]
    lines.append("")
    return "\n".join(lines)


def compare_arms(runs_dir: Any, out_path: Any) -> int:
    """Scan a directory of finished runs and write the arm and cross-model tables.

    Args:
        runs_dir: Directory scanned recursively for ``summary.json`` files, one per run. Runs of
            **both** architectures are expected: the cross-model table is the reason this command
            takes a directory rather than a list of this model's arms.
        out_path: Where the markdown document is written.

    Returns:
        The process exit code: non-zero when no summary was found -- an empty comparison is an
        operator error, not an empty result.
    """
    runs_dir = Path(str(runs_dir))
    summaries = sorted(runs_dir.rglob(SUMMARY_FILENAME))
    if not summaries:
        print(
            f"no {SUMMARY_FILENAME} found under {runs_dir}; --runs expects a directory holding "
            f"finished evaluation runs, one summary per arm."
        )
        return 1

    arms = [collect_arm(path, runs_dir) for path in summaries]
    out = Path(str(out_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_arm_tables(arms), encoding="utf-8")

    by_class = sorted({arm["model_class"] for arm in arms})
    n_incomplete = sum(1 for arm in arms if arm["incomplete"])
    print(
        f"wrote {out}: {len(arms)} run(s) over {len(by_class)} model class(es) "
        f"({', '.join(by_class)}), {n_incomplete} incomplete"
    )
    return 0


# =============================================================================
# Command line
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for both entry points."""
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_attn_transformer_rws.eval.verify",
        description=(
            "Check a completed eval run against the acceptance criteria, or build this model's "
            "arm tables and the cross-model comparison from a directory of finished runs."
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
        help="Directory of finished runs, of either or both models; emits the tables instead of "
             "the gate.",
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
#: **either or both** models and writes the arm tables and the cross-model comparison to ``out``
#: instead. Setting both is refused, because it would be ambiguous which one was meant.
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
