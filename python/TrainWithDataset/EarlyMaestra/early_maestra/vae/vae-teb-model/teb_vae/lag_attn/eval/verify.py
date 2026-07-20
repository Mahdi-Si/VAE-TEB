r"""Check a completed run's ``summary.json`` against the acceptance criteria for a real run.

A first run against a genuinely trained checkpoint is a *verification*, and a verification whose
criteria are read off the output by eye is not one -- it is a search for reassurance among a
hundred numbers. The criteria are therefore encoded here, ahead of the run, and checked
mechanically:

.. code-block:: bash

    python -m teb_vae.lag_attn.eval.verify <run>/eval_results/summary.json

Each criterion prints ``PASS``, ``FAIL`` or ``INCONCLUSIVE`` with the number behind it, and the
process exits non-zero if any failed. ``INCONCLUSIVE`` means the run did not carry what the
criterion needs -- an analysis that was skipped, or a split with no labels -- which is a
different outcome from the criterion being met and is never counted as a pass.

Two of these deserve their reasoning stated, because both are easy to tighten into a wrong test.

**Source specificity is required to *resolve*, not to come back ``source_specific``.**
``influential_not_specific`` is a real finding about a checkpoint, and treating it as a failure
would be exactly the mistake the prediction-space criterion exists to prevent. Only
``undetermined`` -- the pipeline could not tell -- is a failure here.

**Coverage is checked against $0.9545$, not $0.95$.** That is the two-sided mass of a
$\pm 2\sigma$ band; $0.95$ is $\pm 1.96\sigma$. The half-point difference is large enough to read
as a real miscalibration, so the nominal is taken from the run's own report rather than written
down again here.

This module reads a summary and nothing else. It loads no model, opens no shard, and needs no
GPU, so a run produced on the production box can be checked anywhere the file can be copied to.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

#: Verdicts, in decreasing order of goodness.
PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"

#: Tolerance on central-interval coverage against its nominal, in absolute probability.
COVERAGE_TOLERANCE = 0.05

#: The fraction of samples that must show a positive uplift for the pathway to be doing work.
UPLIFT_MAJORITY = 0.5


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


# ---------------------------------------------------------------------------
# The criteria
# ---------------------------------------------------------------------------
def check_exit_code(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Every step completed without raising."""
    code, failed = summary.get("exit_code"), summary.get("failed") or []
    if code is None:
        return _result(INCONCLUSIVE, "the summary carries no exit code")
    return _result(
        PASS if int(code) == 0 and not failed else FAIL,
        "every step completed" if int(code) == 0 else f"{len(failed)} step(s) failed: {failed}",
        exit_code=int(code), failed=failed,
    )


def check_per_file_counts(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Every configured shard contributed samples."""
    per_file = _dig(summary, "results", "probe", "per_file")
    if not per_file:
        return _result(INCONCLUSIVE, "the run recorded no per-file counts")
    empty = sorted(name for name, count in per_file.items() if not count)
    return _result(
        PASS if not empty else FAIL,
        f"all {len(per_file)} shard(s) contributed" if not empty
        else f"{len(empty)} shard(s) contributed nothing: {empty}",
        per_file=per_file,
    )


def check_weights_loaded(summary: Dict[str, Any]) -> Dict[str, Any]:
    """The weight-space load check passed -- the delta heads are off their zero init."""
    passed = _dig(summary, "results", "preflight", "checks", "weights_loaded", "passed")
    if passed is None:
        return _result(INCONCLUSIVE, "the run recorded no weight-space load check")
    return _result(
        PASS if bool(passed) else FAIL,
        "the checkpoint's delta heads differ from their zero initialisation" if passed
        else "the delta heads are still at zero init -- the checkpoint may never have loaded",
    )


def check_uplift(summary: Dict[str, Any]) -> Dict[str, Any]:
    """The residual pathway helps, on a majority of samples."""
    fraction = _dig(summary, "results", "uplift", "positive_fraction")
    relative = _dig(summary, "results", "headline", "uplift_rel")
    if fraction is None:
        return _result(INCONCLUSIVE, "the uplift analysis did not report")
    ok = float(fraction) > UPLIFT_MAJORITY and (relative is None or float(relative) > 0.0)
    return _result(
        PASS if ok else FAIL,
        f"{float(fraction):.1%} of samples show a positive uplift"
        + ("" if relative is None else f", mean relative uplift {float(relative):.4g}"),
        positive_fraction=float(fraction),
        uplift_rel=None if relative is None else float(relative),
    )


def check_kld_active_frac(summary: Dict[str, Any]) -> Dict[str, Any]:
    r"""The latent is in use: ``kld_active_frac`` lies in $(0, 1]$."""
    value = _dig(summary, "results", "headline", "kld_active_frac")
    if value is None:
        return _result(INCONCLUSIVE, "the latent analysis did not report an active fraction")
    value = float(value)
    return _result(
        PASS if 0.0 < value <= 1.0 else FAIL,
        f"{value:.3f} of latent dimensions are active" if 0.0 < value <= 1.0
        else f"kld_active_frac = {value:.3f} is outside (0, 1] -- the latent is collapsed or the "
             f"reading is wrong",
        kld_active_frac=value,
    )


def check_specificity_resolves(summary: Dict[str, Any]) -> Dict[str, Any]:
    """The specificity verdict resolved to something.

    Resolving is the bar, not returning ``source_specific`` -- see the module docstring.
    """
    verdict = _dig(summary, "results", "source_specificity", "verdict")
    if verdict is None:
        return _result(INCONCLUSIVE, "the permutation control did not report")
    resolved = str(verdict) != "undetermined"
    return _result(
        PASS if resolved else FAIL,
        f"specificity resolved to {verdict!r}" if resolved
        else "specificity is undetermined -- too few samples to derange, or the control did not "
             "run",
        specificity=str(verdict),
    )


def check_coverage(summary: Dict[str, Any]) -> Dict[str, Any]:
    r"""Central-interval coverage is within tolerance of its nominal.

    The nominal is read from the run rather than restated, so the $\pm 2\sigma$ figure of
    $0.9545$ cannot be compared against $0.95$ by accident.
    """
    coverage = _dig(summary, "results", "calibration", "coverage")
    if not coverage:
        # Inconclusive, not a failure: a checkpoint trained under another objective has no
        # learned predictive variance, so there is nothing to calibrate and nothing is wrong.
        skipped = _dig(summary, "results", "calibration", "skipped")
        return _result(
            INCONCLUSIVE,
            "the calibration analysis was skipped -- it needs likelihood='gaussian_nll' and "
            "sigma_obs='learned'" if skipped else "the calibration analysis did not report",
        )
    record = coverage.get("2sigma") or {}
    gap = record.get("gap")
    if gap is None:
        return _result(INCONCLUSIVE, "the run reported no 2-sigma coverage gap", coverage=coverage)
    within = abs(float(gap)) <= COVERAGE_TOLERANCE
    return _result(
        PASS if within else FAIL,
        f"2-sigma coverage {float(record.get('observed', float('nan'))):.4f} against nominal "
        f"{float(record.get('nominal', float('nan'))):.4f} (gap {float(gap):+.4f}, tolerance "
        f"+/-{COVERAGE_TOLERANCE})",
        gap=float(gap), tolerance=COVERAGE_TOLERANCE,
    )


def check_headline_finite(summary: Dict[str, Any]) -> Dict[str, Any]:
    """No headline scalar is non-finite. Delegated to the run's own sanity block."""
    record = _dig(summary, "results", "sanity", "checks", "headline_finite")
    if not record:
        return _result(INCONCLUSIVE, "the run recorded no headline-finite check")
    verdict = str(record.get("verdict"))
    return _result(
        PASS if verdict == "pass" else (INCONCLUSIVE if verdict == "inconclusive" else FAIL),
        str(record.get("detail", "")),
        non_finite=record.get("non_finite"),
    )


def check_sanity_block(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Every machine-checked sanity verdict passed or was inconclusive."""
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


#: The criteria, in report order. Registered here so the list is the contract rather than a
#: docstring somebody has to keep in step with the code.
CRITERIA: Tuple[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]], ...] = (
    ("exit_code", check_exit_code),
    ("per_file_counts", check_per_file_counts),
    ("weights_loaded", check_weights_loaded),
    ("uplift_positive", check_uplift),
    ("kld_active_frac", check_kld_active_frac),
    ("specificity_resolves", check_specificity_resolves),
    ("coverage_near_nominal", check_coverage),
    ("headline_finite", check_headline_finite),
    ("sanity_block", check_sanity_block),
)


def verify(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Run every criterion against a loaded summary.

    Args:
        summary: The parsed ``summary.json``.

    Returns:
        Per-criterion records plus the counts and an overall ``passed`` flag. ``passed`` requires
        no failure; an inconclusive criterion does not block it, but is reported so a run checked
        against a partial set is never mistaken for a fully verified one.
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


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_attn.eval.verify",
        description="Check a completed eval run against the acceptance criteria.",
    )
    parser.add_argument("summary", help="Path to the run's summary.json.")
    parser.add_argument(
        "--json-out", dest="json_out", default=None,
        help="Also write the machine-readable report here.",
    )
    return parser


if __name__ == "__main__":
    _args = build_parser().parse_args()
    sys.exit(main(_args.summary, _args.json_out))
