r"""The offline acceptance gate, and the arm tables built from finished runs.

Two entry points, one module, both reading files a finished run left behind and nothing else --
no model, no shard, no ``torch``, no GPU. A summary produced on the production box can therefore
be checked on any machine the file can be copied to, and the layering test enforces the property
directly: this module's imports are walked with ``torch`` on its forbidden list.

**The gate**::

    python -m teb_vae.lag_attn_rws.eval.verify <run>/eval_results/summary.json

A first run against a genuinely trained checkpoint is a verification, and a verification whose
criteria are read off the output by eye is a search for reassurance among a hundred numbers. The
criteria are therefore encoded here, ahead of the run, and checked mechanically. Each prints
``PASS``, ``FAIL`` or ``INCONCLUSIVE`` with the numbers behind it, and the process exits non-zero
when any failed. ``INCONCLUSIVE`` means the run did not carry what the criterion needs -- an
analysis that was skipped, a control that could not run -- which is a different outcome from the
criterion being met and is **never counted as a pass**: a report with inconclusive rows says so,
so a partially-verified run cannot be mistaken for a fully verified one.

Which ``pred_gap`` the gate reads is stated in the machinery rather than in prose:
:data:`PRED_GAP_COLUMN` names the Monte Carlo marginalised headline column, the report carries it
under ``pred_gap_column_read``, and the criterion that reads it says so in its detail. The
single-draw training-path column beside it is the objective-parity check, not a second answer.

The model's own acceptance verdicts (``results.verdicts``) are re-read here rather than
re-derived: each becomes one criterion whose outcome is the verdict's recorded status. That is
deliberate -- the verdict arithmetic lives in the readout module beside the numbers it judges,
and the gate's job is to refuse a run whose verdicts failed, not to re-litigate them.

**The arm tables**::

    python -m teb_vae.lag_attn_rws.eval.verify --runs <dir-of-runs> --out RESULTS_arms.md

Reads every finished ``summary.json`` under a directory and emits the calibration study's four
tables -- the KL-weight sweep, the latent width sweep, the reach budget sweep and the
architecture arms -- so the results document is filled by arithmetic rather than transcription.
Three sourcing rules, each closing a specific failure:

* **Arms are keyed by the swept value read from each run's own** ``resolved_config.yaml``,
  never from a directory name: a renamed directory must not relabel a measurement.
* **The training series come from each run's** ``metrics_history.csv`` -- the epoch count, the
  final ``val/kld_active_frac`` and the collapse verdict live there, not in any evaluation
  output, and :func:`~teb_vae.lag_attn_rws.collapse.is_collapsed` consumes the per-epoch tail
  rather than a point reading. The CSV is found from the summary's own checkpoint path.
* **A collapsed arm is marked, never omitted, and an arm missing its CSV is reported as
  incomplete rather than silently rowless.** A sweep table that quietly dropped its failures
  would read as a sweep that had none.

Everything numeric in the tables comes from the summary's **headline block**, which is the one
surface the reporting layer promises to keep resolvable -- a number not registered there is
deliberately invisible to this module.
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

#: Criterion outcomes, in decreasing order of goodness. ``INCONCLUSIVE`` never counts as a pass.
PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"

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
RWS_VERDICTS: Tuple[str, ...] = (
    "predictive_improvement",
    "source_specificity",
    "prior_carries_target_state",
    "latent_not_collapsed",
    "prior_variance_not_pinned",
    "decoder_variance_not_pinned",
    "calibration_near_nominal",
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
    *((f"verdict_{name}", verdict_criterion(name)) for name in RWS_VERDICTS),
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
        The process exit code: non-zero when any criterion failed.
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
# Arm comparison
# =============================================================================
#: Dotted paths into a run's resolved config, one per sweep axis. The value at the path is the
#: arm's key in its table; a path the config does not carry keys the row ``(absent)``, which is
#: itself information -- an arm file that dropped its swept key is not the arm it claims to be.
SWEPT_BETA = ("model_config", "VAE_model", "beta_schedule", "end")
SWEPT_DZ = ("model_config", "VAE_model", "d_z")
SWEPT_REACH = ("model_config", "VAE_model", "causal_reach_budget_s")

#: The architecture A/B knobs, rendered together: each arm flips one against the baseline, and a
#: row naming all of them is readable without knowing in advance which one this arm flipped.
ARCHITECTURE_KEYS: Tuple[Tuple[str, ...], ...] = (
    ("model_config", "VAE_model", "encoder_extra_kernel"),
    ("model_config", "VAE_model", "conv_norm_groups"),
    ("model_config", "VAE_model", "query_uses_logvar"),
    ("model_config", "VAE_model", "horizon_depth"),
    ("model_config", "VAE_model", "horizon_embed_std"),
    ("model_config", "VAE_model", "head_init_calibration"),
    ("model_config", "VAE_model", "a_head_gain"),
)

#: Sentinel distinguishing "the key is not in the config" from an explicit ``null``: the reach
#: sweep's baseline arm *sets* ``causal_reach_budget_s: null``, and folding the two together
#: would key a mis-built arm identically to the deliberate unguarded one.
_ABSENT = object()


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


def collect_arm(summary_path: Path, runs_dir: Path) -> Dict[str, Any]:
    """Assemble everything the tables need about one finished run.

    Args:
        summary_path: The run's ``summary.json``.
        runs_dir: The directory the scan started from, for the display path.

    Returns:
        The arm record: the headline numbers, the swept config values, the training series, the
        collapse verdict, and an ``incomplete`` list naming what could not be read. Nothing here
        raises on a partial run -- an arm with problems is a row with its problems named.
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
        # Lazy so the single-run gate stays a stdlib parse; the arm tables are the one consumer
        # of the resolved config and therefore the one place PyYAML is needed.
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

    collapsed: Optional[bool] = None
    d_z = _dig_config(config, SWEPT_DZ)
    if history.get("kl_raw_per_epoch") and d_z is not _ABSENT and d_z is not None:
        collapsed = is_collapsed(
            history["kl_raw_per_epoch"],
            history.get("kld_active_frac_per_epoch") or [],
            int(d_z),
        )
    elif csv_path is not None:
        incomplete.append(
            "collapse verdict not computable: the CSV carries no "
            f"{KL_SERIES_COLUMN!r} series or the config no d_z"
        )

    headline = _dig(summary, "results", "headline") or {}
    run_context = summary.get("run_context") or {}
    return {
        "run": display,
        "summary_path": str(summary_path),
        "config": config,
        "headline": headline,
        "exit_code": summary.get("exit_code"),
        "n_parameters": run_context.get("n_parameters"),
        "train_epoch": run_context.get("train_epoch"),
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
    return "incomplete: " + ("; ".join(arm["incomplete"]) or "verdict not computable")


def _channels_kept_cell(arm: Dict[str, Any]) -> str:
    """Render the reach sweep's ``Channels kept (tgt/src)`` cell.

    The declared ``c_y`` / ``c_u`` are the *shard's* stream widths, which two startup guards force
    to match for every arm that can train at all -- so rendering them here would print the same
    pair in every row of a sweep whose entire subject is how many channels survive the budget, and
    say the budget pruned nothing. The surviving counts are the record the trainer resolved and
    wrote beside the config; the declared widths stand in only when that record is absent, which
    is the unguarded arm, where the two genuinely coincide.
    """
    kept = ("model_config", "resolved_causal_budget")
    target = _dig_config(arm["config"], kept + ("target_channels_kept",))
    source = _dig_config(arm["config"], kept + ("source_channels_kept",))
    if target is _ABSENT or target is None:
        target = _dig_config(arm["config"], ("model_config", "VAE_model", "c_y"))
    if source is _ABSENT or source is None:
        source = _dig_config(arm["config"], ("model_config", "VAE_model", "c_u"))
    return f"{_render(target)} / {_render(source)}"


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


def _sweep_rows(
    arms: Sequence[Dict[str, Any]],
    key_path: Sequence[str],
    cells: Callable[[Dict[str, Any]], List[str]],
) -> List[List[str]]:
    """Build one sweep table's rows: every arm, keyed by its value at ``key_path``, sorted."""
    keyed = [(arm, _dig_config(arm["config"], key_path)) for arm in arms]
    keyed.sort(key=lambda pair: (_sort_key(pair[1]), pair[0]["run"]))
    return [[_render(value), *cells(arm), arm["run"]] for arm, value in keyed]


def build_arm_tables(arms: Sequence[Dict[str, Any]]) -> str:
    """Render the calibration study's tables from the collected arm records.

    Every arm appears in every sweep table, keyed by its value of that table's swept axis: which
    arms belong to which sweep is a fact about the run *directory* an operator handed in, not
    something this module infers, and a reader filtering a table by the other columns' defaults
    loses nothing while a module guessing wrong loses a row. The run column is identification
    only -- the key is always the config value.

    Args:
        arms: The collected arm records.

    Returns:
        The whole document as markdown text.
    """
    lines: List[str] = [
        "# Arm comparison",
        "",
        f"Generated by `python -m teb_vae.lag_attn_rws.eval.verify --runs ...` from "
        f"{len(arms)} finished run(s). Rows are keyed by the swept value read from each run's "
        f"own `{RESOLVED_CONFIG_FILENAME}`, never from its directory name; every arm appears in "
        f"every sweep table under its value of that table's axis. `pred_gap` here is the "
        f"`{PRED_GAP_COLUMN}` headline column. Epochs, the final `{ACTIVE_FRAC_COLUMN}` and the "
        f"collapse verdict come from each run's training `{METRICS_HISTORY_FILENAME}`.",
        "",
        "## Arm inventory",
        "",
    ]
    lines += _table(
        ["Run", "Checkpoint epoch", "Trained epochs", "Parameters", "Eval exit", "Status"],
        [
            [
                arm["run"],
                _render(arm["train_epoch"] if arm["train_epoch"] is not None else _ABSENT),
                _render(arm["last_epoch"] if arm["last_epoch"] is not None else _ABSENT),
                _render(arm["n_parameters"] if arm["n_parameters"] is not None else _ABSENT),
                _render(arm["exit_code"] if arm["exit_code"] is not None else _ABSENT),
                ("incomplete: " + "; ".join(arm["incomplete"])) if arm["incomplete"] else "ok",
            ]
            for arm in sorted(arms, key=lambda record: record["run"])
        ],
    )

    lines += ["", "## KL-weight sweep (`beta_schedule.end`)", ""]
    lines += _table(
        ["`beta_schedule.end`", "Epochs", "`pred_gap`", "`source_conditioned_kl_raw`",
         "`kld_active_frac` (final)", "Collapsed?", "Run"],
        _sweep_rows(
            arms, SWEPT_BETA,
            lambda arm: [
                _render(arm["last_epoch"] if arm["last_epoch"] is not None else _ABSENT),
                _headline_cell(arm, PRED_GAP_COLUMN),
                _headline_cell(arm, "source_conditioned_kl_raw_nats"),
                _render(
                    arm["final_kld_active_frac"]
                    if arm["final_kld_active_frac"] is not None else _ABSENT
                ),
                _collapsed_cell(arm),
            ],
        ),
    )

    lines += ["", "## Latent width sweep (`d_z`)", ""]
    lines += _table(
        ["`d_z`", "`d_base_mc_nats`", "Active dims", "`pred_gap`", "Collapsed?", "Run"],
        _sweep_rows(
            arms, SWEPT_DZ,
            lambda arm: [
                _headline_cell(arm, "d_base_mc_nats"),
                _headline_cell(arm, "kl_active_dims"),
                _headline_cell(arm, PRED_GAP_COLUMN),
                _collapsed_cell(arm),
            ],
        ),
    )

    lines += ["", "## Reach budget sweep (`causal_reach_budget_s`)", ""]
    lines += _table(
        ["`causal_reach_budget_s`", "Channels kept (tgt/src)", "`d_base_mc_nats`",
         "`pred_gap`", "`source_conditioned_kl_raw`", "Collapsed?", "Run"],
        _sweep_rows(
            arms, SWEPT_REACH,
            lambda arm: [
                _channels_kept_cell(arm),
                _headline_cell(arm, "d_base_mc_nats"),
                _headline_cell(arm, PRED_GAP_COLUMN),
                _headline_cell(arm, "source_conditioned_kl_raw_nats"),
                _collapsed_cell(arm),
            ],
        ),
    )

    lines += ["", "## Architecture arms", ""]
    architecture_rows = []
    for arm in sorted(arms, key=lambda record: record["run"]):
        knobs = ", ".join(
            f"{path[-1]}={_render(_dig_config(arm['config'], path))}"
            for path in ARCHITECTURE_KEYS
        )
        architecture_rows.append([
            knobs,
            _render(arm["n_parameters"] if arm["n_parameters"] is not None else _ABSENT),
            _headline_cell(arm, "d_base_mc_nats"),
            _headline_cell(arm, PRED_GAP_COLUMN),
            _headline_cell(arm, "source_conditioned_kl_raw_nats"),
            _render(
                arm["final_kld_active_frac"]
                if arm["final_kld_active_frac"] is not None else _ABSENT
            ),
            _collapsed_cell(arm),
            arm["run"],
        ])
    lines += _table(
        ["Knobs", "Parameters", "`d_base_mc_nats`", "`pred_gap`",
         "`source_conditioned_kl_raw`", "`kld_active_frac` (final)", "Collapsed?", "Run"],
        architecture_rows,
    )

    incomplete = [arm for arm in arms if arm["incomplete"]]
    if incomplete:
        lines += ["", "## Incomplete runs", ""]
        lines += [
            f"- `{arm['run']}`: " + "; ".join(arm["incomplete"]) for arm in incomplete
        ]
    lines.append("")
    return "\n".join(lines)


def compare_arms(runs_dir: Any, out_path: Any) -> int:
    """Scan a directory of finished runs and write the arm tables.

    Args:
        runs_dir: Directory scanned recursively for ``summary.json`` files, one per arm.
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
    document = build_arm_tables(arms)
    out = Path(str(out_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")

    n_incomplete = sum(1 for arm in arms if arm["incomplete"])
    print(f"wrote {out}: {len(arms)} arm(s), {n_incomplete} incomplete")
    return 0


# =============================================================================
# Command line
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for both entry points."""
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_attn_rws.eval.verify",
        description=(
            "Check a completed eval run against the acceptance criteria, or build the arm "
            "tables from a directory of finished runs."
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
        help="Directory of finished runs; emits the arm comparison instead of the gate.",
    )
    parser.add_argument(
        "--out", default="RESULTS_arms.md",
        help="Arm comparison only: where to write the tables.",
    )
    return parser


def _cli(argv: Optional[List[str]] = None) -> int:
    """Dispatch between the gate and the arm comparison. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    if args.runs is not None:
        if args.summary is not None:
            print("give either a summary path or --runs, not both.")
            return 2
        return compare_arms(args.runs, args.out)
    if args.summary is None:
        print("a summary path is required unless --runs names a directory of finished runs.")
        return 2
    return main(args.summary, args.json_out)


if __name__ == "__main__":
    sys.exit(_cli())
