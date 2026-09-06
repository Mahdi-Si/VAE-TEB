r"""The offline acceptance gate, the arm tables, and the cross-cell comparison.

Three entry points, one module, all of them reading files a finished run left behind and nothing
else -- no model, no shard, no ``torch``, no GPU. ``tests/test_eval_self_contained.py`` walks this
module's imports with ``torch`` **and this package's own ``binding``** on its forbidden list, so a
summary produced on the production box can be checked on any machine the file can be copied to.
The binding is on that list because it names a model class, and importing it would drag the whole
numeric stack in behind a module whose one property is that it needs none of it.

**The gate**::

    python -m teb_vae.lag_attn_cfs.eval.verify <run>/eval_results/summary.json

A first run against a genuinely trained checkpoint is a verification, and a verification whose
criteria are read off the output by eye is a search for reassurance among a hundred numbers. The
criteria are therefore encoded here, ahead of the run, and checked mechanically. Each prints
``PASS``, ``FAIL`` or ``INCONCLUSIVE`` with the numbers behind it, and the process exits non-zero
when any failed. ``INCONCLUSIVE`` means the run did not carry what the criterion needs -- an
analysis that was skipped, a control that could not run, a threshold that ships unset -- which is a
different outcome from the criterion being met and is **never counted as a pass**: a report with
inconclusive rows says so, so a partially-verified run cannot be mistaken for a fully verified one.

That last case is this cell's standing one rather than an edge case.
``eval_config.clock_margin_min_nats`` ships unset, so ``coupling_exceeds_availability_clock`` is
INCONCLUSIVE on every run until somebody sets it from measured data -- and the measurement reaches
the arm tables regardless, through the ``coupling_minus_clock_nats`` headline scalar, which is what
makes setting the threshold from data possible at all.

Which ``pred_gap`` the gate reads is stated in the machinery rather than in prose:
:data:`PRED_GAP_COLUMN` names the Monte Carlo marginalised headline column, the report carries it
under ``pred_gap_column_read``, and the criterion that reads it says so in its detail. The
single-draw training-path column beside it is the objective-parity check, not a second answer.

The model's own acceptance verdicts (``results.verdicts``) are re-read here rather than
re-derived: each becomes one criterion whose outcome is the verdict's recorded status. That is
deliberate -- the verdict arithmetic lives in the readout module beside the numbers it judges,
and the gate's job is to refuse a run whose verdicts failed, not to re-litigate them. Ten of them
here against the raw pipeline's eight, the two extra being the ones only a causal cell can have.

**The tables**::

    python -m teb_vae.lag_attn_cfs.eval.verify --runs <dir-of-runs> --out RESULTS_arms.md

Reads every finished ``summary.json`` under a directory and emits one table per swept axis -- the
four shipped ``sweep_*.yaml`` arms -- plus the cross-cell table putting ``SeqVaeLagAttnCfs`` beside
``SeqVaeLagAttnTrfCfs``, so the encoder edge is readable in one place. Four sourcing rules, each
closing a specific failure:

* **Arms are keyed by the swept value read from each run's own** ``resolved_config.yaml``,
  never from a directory name: a renamed directory must not relabel a measurement.
* **The training series come from each run's** ``metrics_history.csv`` -- the epoch count, the
  final ``val/kld_active_frac`` and the collapse verdict live there, not in any evaluation
  output, and :func:`~teb_vae.lag_attn_rws.collapse.is_collapsed` consumes the per-epoch tail
  rather than a point reading. The CSV is found from the summary's own checkpoint path.
* **A collapsed arm is marked, never omitted, and an arm missing either series is reported as
  unknown rather than as healthy.** A sweep table that quietly dropped its failures would read as
  a sweep that had none, and a blank ``Collapsed?`` cell reads as "checked, and fine".
* **Everything numeric comes from the summary's headline block**, which is the one surface the
  reporting layer promises to keep resolvable -- a number not registered there is deliberately
  invisible to this module.

**Two arms at different horizons are not compared on a loss level**, and
:data:`HORIZON_LEVEL_RULE` says so in the emitted document rather than only here: the block a score
is per-anchor over is $H \cdot C_{\mathrm{keep}}$ coefficients, so a run at a longer horizon scores a
proportionally wider block and its nats are larger for that reason alone.

There is deliberately **no cross-target table against** ``lag_attn_fs``. The blocks differ (a
budget-gated $H \cdot C_{\mathrm{keep}}$ against the two-sided cell's full channel set, and the
horizons may differ as well), so a level comparison would invite
exactly the reading both ``DESIGN.md`` records forbid.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from teb_vae.lag_attn_rws.collapse import is_collapsed
from teb_vae.lag_attn_cfs.eval import launch

#: Criterion outcomes, in decreasing order of goodness. ``INCONCLUSIVE`` never counts as a pass.
PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"

#: Where ``--runs`` writes the tables when no destination is named. Applied after the launch merge
#: rather than as the parser's default, so a path in :data:`RUN_ARGS` is reachable.
DEFAULT_ARMS_OUT = "RESULTS_arms.md"

#: The ``pred_gap`` column this module reads: the Monte Carlo marginalised headline,
#: $D = -[\operatorname{logsumexp}_r(-D_r) - \log K]$ differenced between the two branches. The
#: single-draw ``pred_gap_train_path_nats`` beside it is the objective-parity column and is
#: deliberately not read here.
PRED_GAP_COLUMN = "pred_gap_mc_nats"

#: The summary filename a run writes. Restated rather than imported: the reporting seam pulls in
#: ``numpy`` and the logging stack, and this module must stay importable with neither.
#: ``tests/test_eval_verify.py`` pins it equal to the seam's.
SUMMARY_FILENAME = "summary.json"

#: The resolved-config filename a run dumps beside its summary. Restated for the same reason --
#: the canonical constant lives in ``trainer``, which the layering forbids here -- and pinned by
#: the same test.
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"

#: The training history CSV a fit leaves in its ``train_results`` directory.
METRICS_HISTORY_FILENAME = "metrics_history.csv"

#: The metrics CSV columns the collapse criterion and the epoch count are read from.
EPOCH_COLUMN = "epoch"
KL_SERIES_COLUMN = "val/source_conditioned_kl_raw"
ACTIVE_FRAC_COLUMN = "val/kld_active_frac"

#: The model's acceptance verdicts, each promoted to one gate criterion. Restated rather than
#: imported from the readout module, which pulls in ``torch`` and the whole network;
#: ``tests/test_eval_verify.py`` pins this equal to ``metrics.PROMOTED_VERDICTS`` so a registry
#: change fails a test instead of silently dropping a criterion from the gate.
#:
#: Ten, against the raw pipeline's eight. The two at the end are the ones only a causal cell can
#: have: ``coupling_exceeds_availability_clock`` separates the coupling readout from a
#: deterministic availability clock no permutation control can see, and ``anchor_geometry_intact``
#: says the population every number above was computed over is the one the configuration states.
CFS_VERDICTS: Tuple[str, ...] = (
    "predictive_improvement",
    "source_margin_positive",
    "source_specificity",
    "prior_carries_target_state",
    "latent_not_collapsed",
    "prior_variance_not_pinned",
    "decoder_variance_not_pinned",
    "calibration_near_nominal",
    "coupling_exceeds_availability_clock",
    "anchor_geometry_intact",
)


def _dig(blob: Any, *path: Any) -> Any:
    """Follow a key path into nested dicts, returning ``None`` if any step is missing."""
    node = blob
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _result(verdict: str, detail: str, **numbers: Any) -> Dict[str, Any]:
    """Build one criterion's record."""
    return {"verdict": verdict, "detail": detail, **numbers}


def _from_sanity_check(summary: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Re-read one of the run's own sanity checks as a gate criterion.

    Delegated rather than recomputed: the run holds the tables and the probe record the check
    was computed from, and this module deliberately does not. The sanity block records the
    outcome without moving the exit code -- that asymmetry is why this gate exists -- so
    re-reading it here is what turns a warning into a refusal.

    Args:
        summary: The parsed summary.
        name: The check's key under ``results.sanity.checks``.

    Returns:
        The criterion record, ``INCONCLUSIVE`` when the run recorded no such check.
    """
    record = _dig(summary, "results", "sanity", "checks", name)
    if not record:
        return _result(INCONCLUSIVE, f"the run recorded no {name!r} sanity check")
    verdict = str(record.get("verdict"))
    mapped = PASS if verdict == "pass" else (FAIL if verdict == "fail" else INCONCLUSIVE)
    return _result(mapped, str(record.get("detail", "")))


# =============================================================================
# The criteria
# =============================================================================
def check_exit_code(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Every step completed without raising."""
    code, failed = summary.get("exit_code"), summary.get("failed") or []
    if code is None:
        return _result(INCONCLUSIVE, "the summary carries no exit code")
    return _result(
        PASS if int(code) == 0 and not failed else FAIL,
        "every step completed" if int(code) == 0 and not failed
        else f"{len(failed)} step(s) failed: {failed}",
        exit_code=int(code), failed=failed,
    )


def check_per_file_counts(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Every configured shard contributed samples. Delegated to the run's population check."""
    return _from_sanity_check(summary, "per_file_counts")


def check_weights_loaded(summary: Dict[str, Any]) -> Dict[str, Any]:
    """The weight-space load check passed -- the zero-initialised tensors are off zero."""
    preflight = summary.get("preflight") or {}
    if preflight.get("skipped"):
        return _result(
            INCONCLUSIVE,
            "this pass built no model and found no earlier preflight record, so the "
            "weight-space load check never ran",
        )
    record = _dig(summary, "preflight", "checks", "weights_loaded")
    if not record:
        return _result(INCONCLUSIVE, "the run recorded no weight-space load check")
    passed = bool(record.get("passed"))
    witnesses = record.get("witnesses_with_evidence") or []
    return _result(
        PASS if passed else FAIL,
        f"the checkpoint's zero-initialised tensors carry loaded weights "
        f"(evidence: {witnesses})" if passed
        else "the zero-initialised tensors are still at zero -- the checkpoint may never have "
             "loaded",
        witnesses_with_evidence=witnesses,
        max_abs_weight=record.get("max_abs_weight"),
    )


def check_headline_pred_gap(summary: Dict[str, Any]) -> Dict[str, Any]:
    """The headline coupling readout resolved to a finite number.

    This is the criterion that fixes which of the two ``pred_gap`` columns the gate means:
    :data:`PRED_GAP_COLUMN`, the Monte Carlo marginalised score. Finiteness only -- the *sign*
    is the ``predictive_improvement`` verdict's question, and asking it twice under two names
    would let the two answers drift.
    """
    value = _dig(summary, "results", "headline", PRED_GAP_COLUMN)
    if value is None:
        return _result(
            INCONCLUSIVE,
            f"the headline carries no {PRED_GAP_COLUMN} -- the collection pass did not report",
            column_read=PRED_GAP_COLUMN,
        )
    finite = isinstance(value, (int, float)) and math.isfinite(float(value))
    return _result(
        PASS if finite else FAIL,
        f"{PRED_GAP_COLUMN} = {value!r} (the Monte Carlo marginalised column; the train-path "
        f"column beside it is the parity check and is not read here)",
        column_read=PRED_GAP_COLUMN,
        value=value if finite else str(value),
    )


def check_headline_finite(summary: Dict[str, Any]) -> Dict[str, Any]:
    """No headline scalar is non-finite. Delegated to the run's own sanity block."""
    return _from_sanity_check(summary, "headline_finite")


def check_sanity_block(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Every in-run self-consistency check passed or was inconclusive."""
    sanity = _dig(summary, "results", "sanity")
    if not sanity:
        return _result(INCONCLUSIVE, "the run recorded no sanity block")
    failed = sanity.get("failed") or []
    return _result(
        PASS if not failed else FAIL,
        "no sanity check failed" if not failed
        else f"{len(failed)} sanity check(s) failed: {failed}",
        failed=failed, n_inconclusive=sanity.get("n_inconclusive"),
    )


def verdict_criterion(name: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Build the gate criterion for one of the model's own acceptance verdicts.

    The outcome is the verdict's recorded status, verbatim: the arithmetic that decided it lives
    in the readout module beside the numbers it judges, and a second implementation here would be
    a second chance to disagree. A verdict the run did not report is ``INCONCLUSIVE``.

    Args:
        name: The verdict's registry name.

    Returns:
        The criterion callable.
    """

    def check(summary: Dict[str, Any]) -> Dict[str, Any]:
        by_name = {
            str(verdict.get("name")): verdict
            for verdict in (_dig(summary, "results", "verdicts") or [])
            if isinstance(verdict, dict)
        }
        record = by_name.get(name)
        if record is None:
            return _result(INCONCLUSIVE, f"the run reported no {name!r} verdict")
        status = str(record.get("status"))
        mapped = status if status in (PASS, FAIL, INCONCLUSIVE) else INCONCLUSIVE
        return _result(
            mapped,
            str(record.get("detail", "")),
            criterion=record.get("criterion"),
            values=record.get("values"),
        )

    check.__name__ = f"check_verdict_{name}"
    return check


#: The criteria, in report order. A registry rather than a docstring somebody keeps in step with
#: the code: the five structural checks bracket the model's own verdicts, so the report reads
#: "did the run complete and load" -> "what did the model's criteria say" -> "do the numbers
#: hold together".
CRITERIA: Tuple[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]], ...] = (
    ("exit_code", check_exit_code),
    ("per_file_counts", check_per_file_counts),
    ("weights_loaded", check_weights_loaded),
    ("headline_pred_gap", check_headline_pred_gap),
    *((f"verdict_{name}", verdict_criterion(name)) for name in CFS_VERDICTS),
    ("headline_finite", check_headline_finite),
    ("sanity_block", check_sanity_block),
)


def verify(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Run every criterion against a loaded summary.

    Args:
        summary: The parsed ``summary.json``.

    Returns:
        Per-criterion records plus the counts, the overall ``passed`` flag, and
        ``pred_gap_column_read`` naming which of the two ``pred_gap`` columns the gate reads.
        ``passed`` requires no failure; an inconclusive criterion does not block it, but is
        reported so a run checked against a partial set is never mistaken for a fully verified
        one.
    """
    results = {name: check(summary) for name, check in CRITERIA}
    failed = [name for name, record in results.items() if record["verdict"] == FAIL]
    inconclusive = [name for name, record in results.items() if record["verdict"] == INCONCLUSIVE]
    return {
        "criteria": results,
        "failed": failed,
        "inconclusive": inconclusive,
        "n_passed": len(results) - len(failed) - len(inconclusive),
        "passed": not failed,
        "pred_gap_column_read": PRED_GAP_COLUMN,
    }


def format_report(report: Dict[str, Any]) -> str:
    """Render the verification as a console table.

    Args:
        report: The output of :func:`verify`.

    Returns:
        The table as a multi-line string.
    """
    lines = ["", "=" * 78, "run verification against the pre-registered criteria", "=" * 78]
    for name, record in report["criteria"].items():
        lines.append(f"  [{record['verdict']:>12s}] {name}")
        lines.append(f"                 {record['detail']}")
    lines.append("-" * 78)
    lines.append(
        f"  {report['n_passed']} passed, {len(report['failed'])} failed, "
        f"{len(report['inconclusive'])} inconclusive"
    )
    if report["inconclusive"]:
        lines.append(
            f"  note: {report['inconclusive']} could not be evaluated from this run, so the "
            f"verification is partial."
        )
    lines.append(f"  pred_gap column read: {report['pred_gap_column_read']}")
    lines.append(f"  VERDICT: {'PASS' if report['passed'] else 'FAIL'}")
    lines.append("=" * 78)
    return "\n".join(lines)


def main(summary_path: Any, json_out: Optional[Any] = None) -> int:
    """Verify one run and print the report.

    Args:
        summary_path: Path to a run's ``summary.json``.
        json_out: Optional path to write the machine-readable report to.

    Returns:
        The process exit code: non-zero when any criterion failed. That asymmetry is the point of
        the gate -- the runner's own exit code reports whether a step raised, and a run whose
        every step completed while its sanity block failed exits 0 there and 1 here.
    """
    path = Path(str(summary_path))
    with open(path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    report = verify(summary)
    report["summary_path"] = str(path)
    print(format_report(report))

    if json_out is not None:
        with open(str(json_out), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {json_out}")
    return 0 if report["passed"] else 1


# =============================================================================
# The sweep axes
# =============================================================================
#: Dotted paths into a run's resolved config, one per shipped ``sweep_*.yaml`` arm. The value at
#: the path is the arm's key in its table; a path the config does not carry keys the row
#: ``(absent)``, which is itself information -- an arm file that dropped its swept key is not the
#: arm it claims to be.
#:
#: Four, and each moves something different about what a number *means* rather than only about how
#: well the model does: the tiling decides how many anchors a forward scores, the floor decides
#: which anchors exist at all, the horizon decides how wide the scored block is, and the decoder
#: depth is the one arm that moves capacity alone.
SWEPT_ANCHOR_STRIDE = ("model_config", "VAE_model", "anchor_stride")
SWEPT_WARMUP_PERIOD = ("model_config", "VAE_model", "warmup_period")
SWEPT_HORIZON = ("model_config", "VAE_model", "horizon")
SWEPT_HORIZON_DEPTH = ("model_config", "VAE_model", "horizon_depth")

#: The headline columns the tables read, named here so a rename in the reporting seam fails a test
#: rather than silently emptying a column. ``tests/test_eval_verify.py`` pins each one against the
#: registry that produces it -- the first five against the shared registry, the last three against
#: the binding's own extras.
D_BASE_COLUMN = "d_base_mc_nats"
D_FULL_COLUMN = "d_full_mc_nats"
KL_COLUMN = "source_conditioned_kl_raw_nats"
ACTIVE_DIMS_COLUMN = "kl_active_dims"
LAG_PEAK_COLUMN = "kl_argmax_lag_step"
CLOCK_MARGIN_COLUMN = "coupling_minus_clock_nats"
ANCHORS_PER_SAMPLE_COLUMN = "anchors_per_sample"
TARGET_WARM_FRAC_COLUMN = "target_warm_frac"

#: Sentinel distinguishing "the key is not in the config" from an explicit ``null``. No shipped
#: cfs arm sets one of these axes to ``null``, but the distinction is kept rather than folded away:
#: a mis-built arm file that lost its swept key must not key identically to one that set it.
_ABSENT = object()

#: Emitted into the horizon section of the document rather than only stated here, because the
#: comparison it forbids is the one a reader makes by reflex. A block score is per anchor over
#: $H \cdot C_{\mathrm{keep}}$ coefficients, so a run at a longer horizon scores a proportionally
#: wider block and its nats are larger for that reason alone; and the anchor count falls with
#: $H$, so the two arms are not even scored over the same population of anchors.
HORIZON_LEVEL_RULE = (
    "Do not compare two horizons on a loss level. A block score is per anchor over H*C_keep "
    "target coefficients, so an arm at a longer horizon scores a proportionally wider block and "
    "its nats are larger for that reason alone -- and anchors live in [F, T - H), so the two arms "
    "are scored over different anchor counts as well. What is comparable across this axis is the "
    "SIGN and the ORDERING of `pred_gap`, and the scale-free percentage columns beside it."
)

#: The selection rule for the cross-cell table, emitted into the document for the same reason. A
#: table of two architectures' KLs invites exactly the ranking this forbids.
SELECTION_RULE = (
    "Do not select on KL magnitude. A stronger target prior lowers the source-conditioned KL "
    "without the coupling having weakened, so a smaller KL is not a worse model and a larger one "
    "is not a better one. Select on KL that comes with source-specific predictive gain, and treat "
    f"a competitive `{D_BASE_COLUMN}` as a precondition: an arm whose base reconstruction is worse "
    f"than the comparison cell's has not earned a reading on `pred_gap` at all."
)


def _dig_config(config: Any, path: Sequence[str]) -> Any:
    """Follow a dotted path into a config, returning :data:`_ABSENT` when any step is missing."""
    node = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return _ABSENT
        node = node[key]
    return node


def _render(value: Any) -> str:
    """Render one table cell: compact numbers, ``null`` spelt out, absence marked."""
    if value is _ABSENT:
        return "(absent)"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _or_absent(value: Any) -> Any:
    """Return ``value``, or the absence sentinel when it is ``None``.

    The tables distinguish "the run did not record this" from "the run recorded nothing here", and
    :func:`_render` spells the two differently.
    """
    return _ABSENT if value is None else value


def read_metrics_history(csv_path: Path) -> Dict[str, Any]:
    """Read the training series the arm tables and the collapse criterion consume.

    The stdlib reader rather than ``pandas``, deliberately: this module's one property is that it
    runs where nothing is installed, and three float columns do not justify a dataframe.

    Args:
        csv_path: Path to a run's ``metrics_history.csv``.

    Returns:
        The final epoch number, the two per-epoch validation series in epoch order, and the
        final active fraction. Missing columns yield empty series rather than raising -- the
        caller reports what it could not compute.
    """
    epochs: List[float] = []
    kl_series: List[float] = []
    active_series: List[float] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for column, series in (
                (EPOCH_COLUMN, epochs),
                (KL_SERIES_COLUMN, kl_series),
                (ACTIVE_FRAC_COLUMN, active_series),
            ):
                raw = (row.get(column) or "").strip()
                if raw:
                    try:
                        series.append(float(raw))
                    except ValueError:
                        pass
    return {
        "last_epoch": int(epochs[-1]) if epochs else None,
        "kl_raw_per_epoch": kl_series,
        "kld_active_frac_per_epoch": active_series,
        "final_kld_active_frac": active_series[-1] if active_series else None,
    }


def find_metrics_history(summary: Dict[str, Any]) -> Optional[Path]:
    """Locate a run's training CSV from the checkpoint path its summary records.

    The training entry point writes ``train_results/metrics_history.csv`` beside the
    ``model_checkpoints`` directory the checkpoint lives in; the run root and the checkpoint
    directory are tried as well, so a CSV copied around by hand is still found.

    Args:
        summary: The parsed summary, read for its ``checkpoint`` path.

    Returns:
        The CSV path, or ``None`` when the summary names no checkpoint or no candidate exists.
    """
    checkpoint = summary.get("checkpoint")
    if not checkpoint:
        return None
    checkpoint_path = Path(str(checkpoint))
    run_root = checkpoint_path.parent.parent
    for candidate in (
        run_root / "train_results" / METRICS_HISTORY_FILENAME,
        run_root / METRICS_HISTORY_FILENAME,
        checkpoint_path.parent / METRICS_HISTORY_FILENAME,
    ):
        if candidate.is_file():
            return candidate
    return None


# =============================================================================
# Cross-cell comparison
# =============================================================================
#: Where a run records which architecture produced it: the ``model_class`` the collection pass
#: copies out of the checkpoint's own stamp into ``summary.json``'s ``run_context``. The stamp is
#: written nowhere else a finished run keeps -- the dumped config carries every constructor keyword
#: and not the class they build.
MODEL_CLASS_PATH: Tuple[str, ...] = ("run_context", "model_class")

#: This cell, and the comparison cell. Restated as strings rather than imported from the two
#: bindings, which name ``torch`` modules; ``tests/test_eval_verify.py`` pins both against the
#: bindings so a class rename fails there rather than emptying this table.
BASELINE_MODEL_CLASS = "SeqVaeLagAttnCfs"
COMPARISON_MODEL_CLASS = "SeqVaeLagAttnTrfCfs"

#: How a run that recorded no class is keyed. A run evaluated before the class was recorded, or
#: re-run offline against a finished directory with no checkpoint, genuinely does not know what
#: produced it -- and a row guessed from the directory name would be the one error this table's
#: keying rule exists to prevent.
UNRECORDED_MODEL = "(unrecorded)"

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


def collect_arm(summary_path: Path, runs_dir: Path) -> Dict[str, Any]:
    """Assemble everything the tables need about one finished run.

    Args:
        summary_path: The run's ``summary.json``.
        runs_dir: The directory the scan started from, for the display path.

    Returns:
        The arm record: the headline numbers, the swept config values, the training series, the
        collapse verdict, the architecture that produced the run, and an ``incomplete`` list
        naming what could not be read. Nothing here raises on a partial run -- an arm with
        problems is a row with its problems named.
    """
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    try:
        display = summary_path.parent.relative_to(runs_dir).as_posix() or "."
    except ValueError:
        display = str(summary_path.parent)

    incomplete: List[str] = []

    config: Dict[str, Any] = {}
    config_path = summary_path.parent / RESOLVED_CONFIG_FILENAME
    if config_path.is_file():
        # Lazy so the single-run gate stays a stdlib parse; the tables are the one consumer of the
        # resolved config and therefore the one place PyYAML is needed.
        import yaml

        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    else:
        incomplete.append(f"no {RESOLVED_CONFIG_FILENAME} beside the summary")

    history: Dict[str, Any] = {}
    csv_path = find_metrics_history(summary)
    if csv_path is not None:
        history = read_metrics_history(csv_path)
    else:
        incomplete.append(
            f"no {METRICS_HISTORY_FILENAME} found from the summary's checkpoint path"
        )

    # Both series, not just the KL one. Clause 2 of the criterion reads the final active fraction,
    # so a CSV carrying only the KL column would silently answer with clause 1 alone -- and a
    # one-clause answer rendered as "no" is a verdict the run did not support. Unknown instead,
    # with what was missing named.
    collapsed: Optional[bool] = None
    d_z = _dig_config(config, ("model_config", "VAE_model", "d_z"))
    has_series = bool(history.get("kl_raw_per_epoch")) and bool(
        history.get("kld_active_frac_per_epoch")
    )
    if has_series and d_z is not _ABSENT and d_z is not None:
        collapsed = is_collapsed(
            history["kl_raw_per_epoch"],
            history["kld_active_frac_per_epoch"],
            int(d_z),
        )
    elif csv_path is not None:
        incomplete.append(
            "collapse verdict not computable: the CSV carries no "
            f"{KL_SERIES_COLUMN!r} or {ACTIVE_FRAC_COLUMN!r} series, or the config no d_z"
        )

    headline = _dig(summary, "results", "headline") or {}
    run_context = summary.get("run_context") or {}
    # The run's own verdict on whether its lag peak means anything: an argmax pinned at 0 never
    # looked back, and one at the largest attainable lag is against the window edge. Re-read
    # rather than re-derived, for the reason the gate re-reads the model's verdicts.
    lag_peak_check = _dig(summary, "results", "sanity", "checks", "argmax_lag")
    model_class = _dig(summary, *MODEL_CLASS_PATH)
    if not model_class:
        incomplete.append(
            "the summary's run_context records no model_class, so which architecture produced "
            "this run is not readable from the run"
        )
    return {
        "run": display,
        "summary_path": str(summary_path),
        "config": config,
        "headline": headline,
        "exit_code": summary.get("exit_code"),
        "n_parameters": run_context.get("n_parameters"),
        "train_epoch": run_context.get("train_epoch"),
        "model_class": str(model_class) if model_class else UNRECORDED_MODEL,
        "lag_peak_check": lag_peak_check,
        "last_epoch": history.get("last_epoch"),
        "final_kld_active_frac": history.get("final_kld_active_frac"),
        "collapsed": collapsed,
        "incomplete": incomplete,
    }


def _collapsed_cell(arm: Dict[str, Any]) -> str:
    """Render the ``Collapsed?`` cell: a verdict, never a blank.

    A collapsed arm is **marked** -- the criterion fired and that is the finding -- and an arm
    whose verdict could not be computed says why, because a blank cell in a sweep table reads as
    "checked and healthy".
    """
    if arm["collapsed"] is True:
        return "**collapsed**"
    if arm["collapsed"] is False:
        return "no"
    return "unknown: " + ("; ".join(arm["incomplete"]) or "verdict not computable")


def _sort_key(value: Any) -> Tuple[int, Any]:
    """Order rows by swept value: numbers first in numeric order, then the rest by rendering."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return (1, _render(value))
    return (0, float(value))


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    """Render one markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _headline_cell(arm: Dict[str, Any], column: str) -> str:
    """Render one headline number, ``(missing)`` when the run did not carry it."""
    value = arm["headline"].get(column)
    if value is None:
        return "(missing)"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.6g}"
    return str(value)


#: The source-pathway verdict family, rendered as one cell in registry order.
#:
#: Four here rather than the raw pipeline's three, and together rather than as four columns because
#: they are read as a family: FAIL / PASS / FAIL / PASS is a real state -- no predictive gain, a
#: positive source margin, therefore no specificity, and a coupling that still clears the
#: availability clock -- and four separate columns invite reading the first one alone.
_VERDICT_FAMILY: Tuple[str, ...] = (
    "predictive_improvement",
    "source_margin_positive",
    "source_specificity",
    "coupling_exceeds_availability_clock",
)


def _verdict_family_cell(arm: Dict[str, Any]) -> str:
    """Render the four source-pathway verdicts as one cell, in registry order."""
    return " / ".join(
        str(arm["headline"].get(f"verdict_{name}") or "?") for name in _VERDICT_FAMILY
    )


def _lag_peak_cell(arm: Dict[str, Any]) -> str:
    """Render the lag peak and the run's own readability verdict on it, never a bare number.

    A peak quoted without that verdict is the misreading the check exists to prevent: an argmax is
    defined on a flat profile and on a censored one exactly as it is on a real peak. The axis is
    stored-coefficient time and the composed group delay reaches the same order as the lag search
    itself, which is why the step index travels rather than a duration.
    """
    peak = _render(_or_absent(arm["headline"].get(LAG_PEAK_COLUMN)))
    check = arm.get("lag_peak_check")
    if not isinstance(check, dict) or not check:
        return f"{peak} (not checked)"
    verdict = str(check.get("verdict"))
    if verdict == "pass":
        return f"{peak} (inside the window)"
    if verdict == "fail":
        return f"{peak} (**degenerate**: {check.get('detail', 'no detail recorded')})"
    return f"{peak} (inconclusive: {check.get('detail', 'no detail recorded')})"


def _sweep_rows(
    arms: Sequence[Dict[str, Any]],
    key_path: Sequence[str],
    cells: Callable[[Dict[str, Any]], List[str]],
) -> List[List[str]]:
    """Build one sweep table's rows: every arm, keyed by its value at ``key_path``, sorted."""
    keyed = [(arm, _dig_config(arm["config"], key_path)) for arm in arms]
    keyed.sort(key=lambda pair: (_sort_key(pair[1]), pair[0]["run"]))
    return [[_render(value), *cells(arm), arm["run"]] for arm, value in keyed]


def build_arm_inventory(arms: Sequence[Dict[str, Any]]) -> List[str]:
    """Render the inventory section: one row per run, with what could not be read named."""
    return _table(
        ["Run", "Model", "Checkpoint epoch", "Trained epochs", "Parameters", "Eval exit",
         "Status"],
        [
            [
                arm["run"],
                arm["model_class"],
                _render(_or_absent(arm["train_epoch"])),
                _render(_or_absent(arm["last_epoch"])),
                _render(_or_absent(arm["n_parameters"])),
                _render(_or_absent(arm["exit_code"])),
                ("incomplete: " + "; ".join(arm["incomplete"])) if arm["incomplete"] else "ok",
            ]
            for arm in sorted(arms, key=lambda record: record["run"])
        ],
    )


def build_sweep_tables(arms: Sequence[Dict[str, Any]]) -> List[str]:
    """Render this cell's four sweep sections as markdown lines.

    Every arm appears in every sweep table, keyed by its value of that table's axis: which arms
    belong to which sweep is a fact about the run *directory* an operator handed in, not something
    this module infers, and a reader filtering a table by the other columns' defaults loses nothing
    while a module guessing wrong loses a row. The run column is identification only -- the key is
    always the config value.

    Args:
        arms: The collected arm records.

    Returns:
        The four sections' lines, headings included.
    """
    lines: List[str] = []

    # The tiling arm. `anchors_per_sample` is the column the arm exists to move, and it is the
    # evaluation's own measured count rather than the configured stride: the evaluation always
    # decodes densely, so this column says whether the two arms were scored over one population.
    lines += ["", "## Anchor tiling sweep (`anchor_stride`)", ""]
    lines += [
        "The stride is a *training* setting: every run here was evaluated at the dense anchor set, "
        f"so `{ANCHORS_PER_SAMPLE_COLUMN}` should read identically down this column and a row that "
        "does not is a run scored over another population.",
        "",
    ]
    lines += _table(
        ["`anchor_stride`", f"`{ANCHORS_PER_SAMPLE_COLUMN}`", f"`{D_BASE_COLUMN}`", "`pred_gap`",
         f"`{KL_COLUMN}`", f"`{CLOCK_MARGIN_COLUMN}`", "Collapsed?", "Run"],
        _sweep_rows(
            arms, SWEPT_ANCHOR_STRIDE,
            lambda arm: [
                _headline_cell(arm, ANCHORS_PER_SAMPLE_COLUMN),
                _headline_cell(arm, D_BASE_COLUMN),
                _headline_cell(arm, PRED_GAP_COLUMN),
                _headline_cell(arm, KL_COLUMN),
                _headline_cell(arm, CLOCK_MARGIN_COLUMN),
                _collapsed_cell(arm),
            ],
        ),
    )

    # The floor arm. It moves the anchor population deliberately, so the anchor count is the
    # column that says what the policy cost -- and the warm fraction is what says the channels
    # survived it, which is the axis the floor does NOT buy anything on.
    lines += ["", "## Anchor floor sweep (`warmup_period`)", ""]
    lines += [
        "The floor is a policy choice rather than a validity boundary above B - 1 (the slowest "
        "kept channel's warm-up less one): it keeps the "
        f"identical channel set -- `{TARGET_WARM_FRAC_COLUMN}` reads 1.0 either way -- and buys "
        "its policy with anchors. So the two columns are read together, and a difference in the "
        "warm fraction means the checkpoint predates the constructor's pairing refusal rather "
        "than that the floor did something.",
        "",
    ]
    lines += _table(
        ["`warmup_period`", f"`{ANCHORS_PER_SAMPLE_COLUMN}`", f"`{TARGET_WARM_FRAC_COLUMN}`",
         f"`{D_BASE_COLUMN}`", "`pred_gap`", f"`{KL_COLUMN}`", "Collapsed?", "Run"],
        _sweep_rows(
            arms, SWEPT_WARMUP_PERIOD,
            lambda arm: [
                _headline_cell(arm, ANCHORS_PER_SAMPLE_COLUMN),
                _headline_cell(arm, TARGET_WARM_FRAC_COLUMN),
                _headline_cell(arm, D_BASE_COLUMN),
                _headline_cell(arm, PRED_GAP_COLUMN),
                _headline_cell(arm, KL_COLUMN),
                _collapsed_cell(arm),
            ],
        ),
    )

    # The horizon arm, and the one section that carries a refusal rather than a reading rule: the
    # block a score is per-anchor over is H * C_keep, so the levels are not comparable at all.
    lines += ["", "## Horizon sweep (`horizon`)", ""]
    lines += [HORIZON_LEVEL_RULE, ""]
    lines += _table(
        ["`horizon`", f"`{ANCHORS_PER_SAMPLE_COLUMN}`", f"`{D_BASE_COLUMN}` (not comparable)",
         "`pred_gap` (not comparable)", "`pred_gap_rmse_pct`", "`pred_gap_mc_likelihood_pct`",
         "Collapsed?", "Run"],
        _sweep_rows(
            arms, SWEPT_HORIZON,
            lambda arm: [
                _headline_cell(arm, ANCHORS_PER_SAMPLE_COLUMN),
                _headline_cell(arm, D_BASE_COLUMN),
                _headline_cell(arm, PRED_GAP_COLUMN),
                _headline_cell(arm, "pred_gap_rmse_pct"),
                _headline_cell(arm, "pred_gap_mc_likelihood_pct"),
                _collapsed_cell(arm),
            ],
        ),
    )

    # The decoder-depth arm: the only one of the four that moves capacity and nothing about what
    # is scored, so the parameter count is the column it is read against.
    lines += ["", "## Decoder depth sweep (`horizon_depth`)", ""]
    lines += _table(
        ["`horizon_depth`", "Parameters", f"`{D_BASE_COLUMN}`", "`pred_gap`", f"`{KL_COLUMN}`",
         f"`{ACTIVE_DIMS_COLUMN}`", "Collapsed?", "Run"],
        _sweep_rows(
            arms, SWEPT_HORIZON_DEPTH,
            lambda arm: [
                _render(_or_absent(arm["n_parameters"])),
                _headline_cell(arm, D_BASE_COLUMN),
                _headline_cell(arm, PRED_GAP_COLUMN),
                _headline_cell(arm, KL_COLUMN),
                _headline_cell(arm, ACTIVE_DIMS_COLUMN),
                _collapsed_cell(arm),
            ],
        ),
    )
    return lines


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


def baseline_d_base(arms: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Return the baseline cell's best (lowest) $D_0$ across the collected runs.

    Best rather than mean or latest: the rule asks whether an arm's base reconstruction is
    competitive with what the baseline architecture *can* do, and the strongest baseline run in
    the directory is the honest answer to that.

    Args:
        arms: The collected arm records.

    Returns:
        The lowest finite ``d_base_mc_nats`` among runs whose ``model_class`` is
        :data:`BASELINE_MODEL_CLASS`, or ``None`` when the directory holds no such run -- in which
        case no row is flagged, because there is nothing to be worse than.
    """
    values = [
        value
        for value in (
            _finite_d_base(arm) for arm in arms if arm["model_class"] == BASELINE_MODEL_CLASS
        )
        if value is not None
    ]
    return min(values) if values else None


def _pred_gap_cell(arm: Dict[str, Any], baseline: Optional[float]) -> str:
    """Render ``pred_gap``, marked when this row's $D_0$ has not earned the reading."""
    cell = _headline_cell(arm, PRED_GAP_COLUMN)
    d_base = _finite_d_base(arm)
    earned = baseline is None or d_base is None or d_base <= baseline
    return cell if earned else f"{cell} {_D0_MARKER}"


def build_cross_cell_table(arms: Sequence[Dict[str, Any]]) -> List[str]:
    r"""Render the encoder edge: both cfs cells' runs side by side, as markdown lines.

    One row per run, keyed by the architecture that produced it, carrying what the encoder question
    is actually asked with: the parameter count the two must be compared per, $D_0$ and $D_1$, the
    coupling gap, the unfloored KL, the availability-clock margin, the lag peak with the run's own
    readability verdict on it, the four source-pathway verdicts, the final active fraction, the
    collapse verdict and the epoch count.

    The lag peak earns its column here rather than in any sweep table: replacing both history
    encoders is the change most likely to move *where in the past* the source informed the
    forecast, and it is the one column whose bare number is a misreading -- an argmax is defined on
    a flat profile and on a censored one exactly as it is on a real peak.

    ``coupling_minus_clock_nats`` is a column here whatever the threshold says, which is the whole
    reason it is a headline scalar: the verdict beside it ships INCONCLUSIVE, and the threshold is
    meant to be set from the spread this column shows.

    Args:
        arms: The collected arm records.

    Returns:
        The section's lines, heading included.
    """
    baseline = baseline_d_base(arms)
    ordered = sorted(
        arms,
        key=lambda arm: (
            # The baseline cell first, so the row the rule is stated against is the row a reader
            # meets first; everything else alphabetically by class, then by run.
            0 if arm["model_class"] == BASELINE_MODEL_CLASS else 1,
            arm["model_class"],
            arm["run"],
        ),
    )
    classes = sorted({arm["model_class"] for arm in arms})

    lines: List[str] = [
        "",
        "## Cross-cell comparison",
        "",
        f"Rows keyed by the `model_class` each run recorded in its own `run_context`, never by "
        f"directory name. The baseline cell is `{BASELINE_MODEL_CLASS}`; "
        f"`{COMPARISON_MODEL_CLASS}` replaces both history encoders and changes nothing else -- "
        f"the same objective, the same target domain, the same anchor tiling and the same warm-up "
        f"budget -- so every number below is the same objective's.",
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
            _render(_or_absent(arm["n_parameters"])),
            _headline_cell(arm, D_BASE_COLUMN),
            _headline_cell(arm, D_FULL_COLUMN),
            _pred_gap_cell(arm, baseline),
            _headline_cell(arm, KL_COLUMN),
            _headline_cell(arm, CLOCK_MARGIN_COLUMN),
            _lag_peak_cell(arm),
            _verdict_family_cell(arm),
            _render(_or_absent(arm["final_kld_active_frac"])),
            _collapsed_cell(arm),
            _render(_or_absent(arm["last_epoch"])),
            arm["run"],
        ]
        for arm in ordered
    ]
    lines += _table(
        ["Model", "Parameters", f"`{D_BASE_COLUMN}` ($D_0$)", f"`{D_FULL_COLUMN}` ($D_1$)",
         f"`{PRED_GAP_COLUMN}`", f"`{KL_COLUMN}`", f"`{CLOCK_MARGIN_COLUMN}`", "Lag peak",
         "Verdicts", f"`{ACTIVE_FRAC_COLUMN}` (final)", "Collapsed?", "Epochs", "Run"],
        rows,
    )
    # The footnote ships only where a marker did, so a table with nothing flagged does not carry a
    # rule about flagging. The marker can appear only when a baseline run was found.
    if any(_D0_MARKER in row[4] for row in rows):
        lines += ["", _D0_FOOTNOTE]
    return lines


def build_arm_tables(arms: Sequence[Dict[str, Any]]) -> str:
    """Render this cell's sweep tables and the cross-cell comparison as one document.

    Args:
        arms: The collected arm records.

    Returns:
        The whole document as markdown text.
    """
    lines: List[str] = [
        "# Arm comparison",
        "",
        f"Generated by `python -m teb_vae.lag_attn_cfs.eval.verify --runs ...` from "
        f"{len(arms)} finished run(s). Rows are keyed by the swept value read from each run's own "
        f"`{RESOLVED_CONFIG_FILENAME}`, never from its directory name; every arm appears in every "
        f"sweep table under its value of that table's axis. `pred_gap` here is the "
        f"`{PRED_GAP_COLUMN}` headline column. Epochs, the final `{ACTIVE_FRAC_COLUMN}` and the "
        f"collapse verdict come from each run's training `{METRICS_HISTORY_FILENAME}`.",
        "",
        "## Arm inventory",
        "",
    ]
    lines += build_arm_inventory(arms)
    lines += build_sweep_tables(arms)
    lines += build_cross_cell_table(arms)

    incomplete = [arm for arm in arms if arm["incomplete"]]
    if incomplete:
        lines += ["", "## Incomplete runs", ""]
        lines += [f"- `{arm['run']}`: " + "; ".join(arm["incomplete"]) for arm in incomplete]
    lines.append("")
    return "\n".join(lines)


def compare_arms(
    runs_dir: Any,
    out_path: Any,
    build_document: Callable[[Sequence[Dict[str, Any]]], str] = build_arm_tables,
) -> int:
    """Scan a directory of finished runs and write the arm and cross-cell tables.

    Args:
        runs_dir: Directory scanned recursively for ``summary.json`` files, one per run. Runs of
            **both** cfs cells are expected: the cross-cell table is the reason this command takes
            a directory rather than a list of this cell's arms.
        out_path: Where the markdown document is written.
        build_document: Renders the collected arms into markdown. Parameterised for exactly one
            reason: the second cfs cell ships one sweep arm against this cell's four, so its
            document has one sweep section and the same cross-cell table -- and everything around
            the rendering (the scan, the empty-directory refusal, the collection and the console
            line) is identical. A second copy of *that* is what this parameter removes.

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
    out.write_text(build_document(arms), encoding="utf-8")

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
        prog="python -m teb_vae.lag_attn_cfs.eval.verify",
        description=(
            "Check a completed eval run against the acceptance criteria, or build this cell's "
            "arm tables and the cross-cell comparison from a directory of finished runs."
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
#: **either or both** cfs cells and writes the arm tables and the cross-cell comparison to ``out``
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
