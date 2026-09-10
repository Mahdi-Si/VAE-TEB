r"""The acceptance gate: read a finished ``summary.json`` and say what it does and does not support.

Run from the repository root:

.. code-block:: bash

    python -m teb_vae.lag_slot_transformer_cfs.eval.verify \
        --summary output/<run>-eval/<stamp>/eval_results/summary.json

From an IDE's Run button, with no command line: fill in ``RUN_ARGS`` at the bottom of this file.

**Deliberately stdlib only.** It imports no ``torch``, no model and no binding, and it must stay
that way: a gate that costs a numeric stack to answer "did this run measure what it claims" is one
nobody runs, and it has to be runnable against a summary copied off the box that produced it.

**What it checks, and what it refuses to decide.** Four of its verdicts are *structural* and can
genuinely fail: the reference arms whose margins are exactly zero by construction, the absence of
any attention-shaped column, the presence of the qualification the lag readouts must be read with,
and the two summaries -- equal-recording and anchor-weighted -- that must both be reported. Those
are properties of the pipeline and a run that violates one is broken.

The **predictive gap is reported and not gated**, and that is the whole point of shipping it this
way. Where the boundary of an acceptable gap sits is what the first real runs are supposed to
measure; a provisional threshold would decide a pass or a fail on exactly the run that was going to
answer the question, and nobody reading the output could tell a healthy model failing from a broken
one passing. The verdict is INCONCLUSIVE with the measured gap and both interval ends beside it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: Repository root: ``teb_vae/lag_slot_transformer_cfs/eval/verify.py`` -> up four.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Launched as a script, this file's own directory goes on sys.path instead of the repository root.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from teb_vae.lag_attn_cfs.eval.launch import (  # noqa: E402
    missing_required,
    resolve_launch_args,
)

#: Names no artifact of this architecture may carry, anywhere at any depth.
#:
#: Not a style rule. This model computes no attention distribution and no per-lag allocation of the
#: divergence, and a column carrying a proposal norm under one of these names would be read as an
#: attention allocation by every reader and every downstream table -- which is the specific claim
#: the design proves cannot be made. The gate is a name check because that is the form the mistake
#: takes: the tensor would be the right shape and the wrong quantity.
FORBIDDEN_KEYS: Sequence[str] = (
    "attn_weights",
    "attended_source_heads",
    "attention_lag_map",
    "attention_lag_profile",
    "kld_per_t_per_head",
    "source_kl_lag_map",
)

#: Phrases the lag readouts' qualification must still contain. Pinned here rather than compared
#: against the constant that produced it, so a summary written by an older run is checked against
#: what the caveat has to *say* rather than against today's wording.
REQUIRED_QUALIFICATION_PHRASES: Sequence[str] = (
    "which stored source time",
    "fitted computation",
    "physiological delay",
)

#: How far a margin that is exactly zero by construction may drift before it is a defect, in nats
#: per anchor. Zero exactly is what the arithmetic gives -- the reference arms share their latent
#: parameters with the arm they are measured against, bitwise -- so this is a floating-point
#: allowance and not a tolerance on a measurement.
EXACT_MARGIN_TOLERANCE = 1e-9


def _verdict(name: str, status: str, detail: str, **numbers: Any) -> Dict[str, Any]:
    """Assemble one verdict record.

    Args:
        name: The verdict's name.
        status: ``'PASS'``, ``'FAIL'`` or ``'INCONCLUSIVE'``.
        detail: One sentence a reader can act on.
        **numbers: The measurements the verdict was reached from, so a reader never has to take
            the status on trust.

    Returns:
        The record.
    """
    return {"name": name, "status": status, "detail": detail, **numbers}


def _walk_keys(value: Any) -> List[str]:
    """Every mapping key in a nested structure, at any depth.

    Args:
        value: The parsed summary, or any part of it.

    Returns:
        The keys, in traversal order and with duplicates kept.
    """
    found: List[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            found.append(str(key))
            found.extend(_walk_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_walk_keys(item))
    return found


def check_reference_arms(summary: Mapping[str, Any]) -> Dict[str, Any]:
    r"""The two margins that are zero by construction, and the one identity between arms.

    Suppressing an **empty** band removes nothing, so its arm shares the matched full branch's
    latent parameters bitwise and its margin is exactly zero. Suppressing **every** band leaves a
    sum over no lags, so that arm is the target-only prior and its margin equals the base-minus-full
    gap. A drift in either is a defect report: it means the intervention path and the forward path
    are two computations that nearly agree rather than one.

    Args:
        summary: The parsed summary.

    Returns:
        The verdict.
    """
    bands = (summary.get("lag_readouts") or {}).get("band_suppression") or {}
    controls = summary.get("source_controls") or {}
    headline = summary.get("headline") or {}
    empty = (bands.get("none") or {}).get("margin_nats")
    every = (bands.get("all") or {}).get("margin_nats")
    silence = controls.get("silence_margin_nats")
    gap = (headline.get("pred_gap") or {}).get("point")

    if is_target_only(summary):
        return _verdict(
            "reference_arms_are_exact",
            "INCONCLUSIVE",
            "this is a target-only checkpoint: it has no source pathway, so there is no "
            "intervention path to pin. Its gap is exactly zero by construction rather than by "
            "measurement, and the check has nothing to say about it.",
            pred_gap_nats=gap,
        )
    if empty is None:
        return _verdict(
            "reference_arms_are_exact",
            "INCONCLUSIVE",
            "the run recorded no empty-band reference arm, so the intervention path was never "
            "pinned against the forward path.",
            empty_band_margin_nats=empty,
        )
    failures: List[str] = []
    if abs(float(empty)) > EXACT_MARGIN_TOLERANCE:
        failures.append(
            f"the empty band removes nothing and its margin should be exactly zero, and is {empty}"
        )
    # Both are the target-only prior scored against the matched full branch, so they must agree
    # with each other and with the gap. Compared as a triple rather than pairwise, because two of
    # the three agreeing while the third drifts is the informative case.
    for label, value in (("all-band", every), ("silence", silence)):
        if value is None or gap is None:
            continue
        if abs(float(value) - float(gap)) > EXACT_MARGIN_TOLERANCE:
            failures.append(
                f"the {label} arm reproduces the prior, so its margin should equal the "
                f"base-minus-full gap of {gap}, and is {value}"
            )
    status = "FAIL" if failures else "PASS"
    detail = (
        "; ".join(failures)
        if failures
        else "the empty-band arm is exactly the matched forward and the all-band and silence arms "
        "are exactly the target-only prior, so every margin between them is a difference of "
        "predictions."
    )
    return _verdict(
        "reference_arms_are_exact",
        status,
        detail,
        empty_band_margin_nats=empty,
        all_band_margin_nats=every,
        silence_margin_nats=silence,
        pred_gap_nats=gap,
    )


def is_target_only(summary: Mapping[str, Any]) -> bool:
    """Whether the scored checkpoint was built with no source pathway.

    The first thing a reader of two summaries has to establish, because it decides what every other
    block can possibly say: a target-only arm's gap is exactly zero by construction, its lag
    readouts are empty because there was nothing to read, and both facts would otherwise look like
    measurements that came out flat.

    Args:
        summary: The parsed summary.

    Returns:
        ``True`` for a target-only checkpoint.
    """
    return bool((summary.get("arm") or {}).get("source_disabled", False))


def check_no_attention_shaped_column(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """No artifact key names a tensor this architecture does not compute.

    Args:
        summary: The parsed summary.

    Returns:
        The verdict.
    """
    keys = set(_walk_keys(summary))
    # The excluded-analysis block names them on purpose, as the reason each analysis was removed,
    # so it is exempted rather than making the gate unpassable on a correct run.
    keys -= set((summary.get("excluded_analyses") or {}).keys())
    offenders = sorted(keys & set(FORBIDDEN_KEYS))
    return _verdict(
        "no_attention_shaped_column",
        "FAIL" if offenders else "PASS",
        (
            f"the summary carries {offenders}, which name quantities this architecture does not "
            f"compute; a proposal norm under one of those names reads as an attention allocation."
            if offenders
            else "no key names an attention distribution or a per-lag divergence allocation."
        ),
        offending_keys=offenders,
    )


def check_qualification_present(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """The lag readouts carry the caveat they must be read with.

    Asserted on the written artifact rather than on the constant that produced it. A caveat that
    lives only in the code is one tidy-up away from being absent from the thing a reader opens.

    Args:
        summary: The parsed summary.

    Returns:
        The verdict.
    """
    text = str((summary.get("lag_readouts") or {}).get("qualification") or "")
    missing = [phrase for phrase in REQUIRED_QUALIFICATION_PHRASES if phrase not in text]
    return _verdict(
        "lag_qualification_present",
        "FAIL" if missing else "PASS",
        (
            f"the lag readouts' qualification is absent or no longer states: {missing}."
            if missing
            else "the lag readouts carry the qualification that suppression measures reliance on "
            "a fitted parameterisation rather than a unique contribution or a delay."
        ),
        missing_phrases=missing,
    )


def check_both_summaries_reported(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Equal-recording and anchor-weighted figures are both present.

    They differ whenever recordings contribute unequal numbers of anchors, which is always, and a
    run carrying one of them leaves a reader unable to separate a real effect from a length effect.

    Args:
        summary: The parsed summary.

    Returns:
        The verdict.
    """
    equal_recording = "pred_gap" in (summary.get("headline") or {})
    anchor_weighted = "pred_gap" in (summary.get("anchor_weighted") or {})
    both = equal_recording and anchor_weighted
    return _verdict(
        "both_summaries_reported",
        "PASS" if both else "FAIL",
        (
            "the gap is reported both as an equal-recording mean, which the interval is built "
            "over, and as an anchor-weighted mean."
            if both
            else "one of the two summaries is missing; they differ whenever recordings contribute "
            "unequal anchor counts."
        ),
        equal_recording=equal_recording,
        anchor_weighted=anchor_weighted,
    )


def check_excluded_analyses_recorded(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """The analyses this architecture cannot produce are named, with their reasons.

    Args:
        summary: The parsed summary.

    Returns:
        The verdict.
    """
    excluded = summary.get("excluded_analyses") or {}
    unexplained = sorted(name for name, reason in excluded.items() if not str(reason).strip())
    status = "PASS" if excluded and not unexplained else "FAIL"
    return _verdict(
        "excluded_analyses_recorded",
        status,
        (
            "every analysis this architecture cannot produce is named in the summary with the "
            "tensor it would have needed."
            if status == "PASS"
            else f"the summary records no exclusion reasons for {unexplained or 'any analysis'}, "
            f"so a reader finding fewer columns than a sibling has nothing to read."
        ),
        excluded=sorted(excluded),
    )


def report_predictive_gap(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """The measured gap and its interval, reported and deliberately not gated.

    Args:
        summary: The parsed summary.

    Returns:
        The verdict, always INCONCLUSIVE, with the measurement beside it.
    """
    record = (summary.get("headline") or {}).get("pred_gap") or {}
    point, lo, hi = record.get("point"), record.get("lo"), record.get("hi")
    if is_target_only(summary):
        return _verdict(
            "predictive_gap_measured",
            "INCONCLUSIVE",
            "this is a target-only checkpoint, so its gap is exactly zero by construction and is "
            "not a measurement of anything. What this run contributes is its own predictive score, "
            "which is the number a source-conditioned candidate has to be read against.",
            pred_gap_nats=point,
            target_only_nll=((summary.get("headline") or {}).get("nll_base") or {}).get("point"),
            n_recordings=record.get("n"),
        )
    return _verdict(
        "predictive_gap_measured",
        "INCONCLUSIVE",
        (
            f"the source-conditioned branch scores {point} nats per anchor better than the "
            f"target-only branch, interval [{lo}, {hi}] over {record.get('n')} recordings. This "
            f"ships ungated on purpose: where an acceptable boundary sits is what the first real "
            f"runs measure, and a guessed threshold would decide the answer on the run that was "
            f"going to supply it. A positive gap against the INTERNAL base is also not sufficient "
            f"on its own -- it must be read beside an independently trained target-only "
            f"comparator, which no single run of this pass produces."
        ),
        pred_gap_nats=point,
        ci_lo=lo,
        ci_hi=hi,
        n_recordings=record.get("n"),
    )


def check_against_reference(
    summary: Mapping[str, Any], reference: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Read the candidate's source-conditioned score against an independently trained baseline.

    **This is the comparison the internal gap cannot make.** A candidate's own base branch is
    trained jointly with its source pathway, so it can degrade during that training -- and an
    improved internal gap then measures the degradation rather than the source. An independently
    trained target-only predictor, frozen and never touched by the joint run, is the only thing
    that separates the two.

    Reported and not gated, on the same ground as the gap itself: where an acceptable margin sits
    is what the first real runs measure. What the verdict does say, and says as a FAIL, is when the
    candidate's own base branch has fallen behind the frozen reference -- because that is not a
    threshold question. It means the joint training made the baseline worse, and any gap measured
    against it is measuring that.

    Args:
        summary: The candidate's parsed summary.
        reference: An independently trained target-only run's parsed summary, or ``None``.

    Returns:
        The verdict.
    """
    if reference is None:
        return _verdict(
            "candidate_against_external_reference",
            "INCONCLUSIVE",
            "no reference summary was given. A gap against a candidate's own base branch cannot "
            "distinguish a source that helped from a base that got worse during joint training; "
            "pass an independently trained target-only run to make that distinction.",
        )
    if not is_target_only(reference):
        return _verdict(
            "candidate_against_external_reference",
            "FAIL",
            "the summary passed as the reference is not a target-only run. A source-conditioned "
            "model cannot serve as the baseline its own architecture is measured against.",
        )

    headline = summary.get("headline") or {}
    candidate_full = (headline.get("nll_full") or {}).get("point")
    candidate_base = (headline.get("nll_base") or {}).get("point")
    external = ((reference.get("headline") or {}).get("nll_base") or {}).get("point")
    if candidate_full is None or external is None or candidate_base is None:
        return _verdict(
            "candidate_against_external_reference",
            "INCONCLUSIVE",
            "one of the two runs did not report a predictive score to compare.",
            candidate_full_nll=candidate_full,
            reference_nll=external,
        )

    # Scores are negative log likelihoods, so lower is better and an improvement is a negative
    # difference. Both are stated as improvements so the two read in the same direction.
    external_gain = float(external) - float(candidate_full)
    base_drift = float(external) - float(candidate_base)
    degraded = base_drift < 0.0
    return _verdict(
        "candidate_against_external_reference",
        "FAIL" if degraded else "INCONCLUSIVE",
        (
            f"the candidate's own base branch scores {candidate_base} against the frozen "
            f"reference's {external}, so joint training left the baseline WORSE by "
            f"{-base_drift} nats per anchor. Any gap measured against that base is measuring the "
            f"degradation, whatever its sign."
            if degraded
            else f"the candidate's source-conditioned branch improves on the frozen reference by "
            f"{external_gain} nats per anchor, and its own base branch is within "
            f"{base_drift} of that reference rather than behind it. Ungated: where an acceptable "
            f"margin sits is what the first real runs measure."
        ),
        candidate_full_nll=candidate_full,
        candidate_base_nll=candidate_base,
        reference_nll=external,
        improvement_over_reference_nats=external_gain,
        base_minus_reference_nats=base_drift,
    )


#: The gate's checks, in report order: the structural ones that can fail, then the measurement.
CHECKS = (
    check_reference_arms,
    check_no_attention_shaped_column,
    check_qualification_present,
    check_both_summaries_reported,
    check_excluded_analyses_recorded,
    report_predictive_gap,
)


def verify(
    summary: Mapping[str, Any], reference: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Run every check against one parsed summary, and against a reference when one is given.

    Args:
        summary: The parsed summary.
        reference: An independently trained target-only run's summary, or ``None``.

    Returns:
        ``{'verdicts': [...], 'failed': [...], 'passed': bool}``.
    """
    verdicts = [check(summary) for check in CHECKS]
    verdicts.append(check_against_reference(summary, reference))
    failed = [record["name"] for record in verdicts if record["status"] == "FAIL"]
    return {"verdicts": verdicts, "failed": failed, "passed": not failed}


def main(summary: Optional[str] = None, reference: Optional[str] = None) -> int:
    """Read a summary and print its verdicts.

    Args:
        summary: Path to the ``summary.json``. Required, enforced here rather than by argparse.
        reference: Path to an independently trained target-only run's summary, or ``None``. Without
            it the candidate's gap is read against its own base branch alone, which cannot
            distinguish a source that helped from a base that got worse.

    Returns:
        ``0`` when nothing failed, ``1`` on a failure, ``2`` on a refusal.
    """
    refusal = missing_required({"summary": summary}, ("summary",))
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 2

    with open(str(summary), "r", encoding="utf-8") as handle:
        parsed = json.load(handle)
    baseline = None
    if reference is not None:
        with open(str(reference), "r", encoding="utf-8") as handle:
            baseline = json.load(handle)

    result = verify(parsed, baseline)
    for record in result["verdicts"]:
        print(f"{record['status']:<13} {record['name']}: {record['detail']}")
    return 0 if result["passed"] else 1


#: Values used when the module is launched with no command line -- i.e. an IDE's Run button. Keyed
#: by argparse ``dest``; a flag always wins over the entry here, per key.
#:
#: ``summary`` MUST be filled in for this file to run at all. Neither entry is a setting: both name
#: an artifact to read, which is what keeps every verdict a property of the runs rather than of how
#: the gate was invoked.
RUN_ARGS: Dict[str, Any] = {
    # REQUIRED. Path to a finished summary.json, repo-root-relative or absolute.
    "summary": None,
    # Optional. A frozen, independently trained target-only run's summary.json. Without it the
    # candidate's gap is read against its own base branch alone, and that comparison cannot tell a
    # source that helped from a base that got worse during joint training.
    "reference": None,
}


def build_parser() -> argparse.ArgumentParser:
    """Build this entry point's own parser.

    One argument, and it carries no ``required=True`` and no non-``None`` default for the reasons
    the run entry point's parser records: the first fires before the launch dict is read, and the
    second would make the dict entry unreachable while the operator edited it.

    Returns:
        The parser, whose ``dest`` set is also the valid key set for :data:`RUN_ARGS`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_slot_transformer_cfs.eval.verify",
        description="Verify a finished evaluation summary.",
    )
    parser.add_argument("--summary", default=None, help="Path to summary.json.")
    parser.add_argument(
        "--reference",
        default=None,
        help="A frozen target-only run's summary.json, to read the candidate against.",
    )
    return parser


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    """Parse, merge with :data:`RUN_ARGS`, and run.

    Args:
        argv: Command-line arguments, or ``None`` for ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    values, sources = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        os.chdir(_REPO_ROOT)
    print(f"argument sources: {sources}", file=sys.stderr)
    return main(**values)


if __name__ == "__main__":
    sys.exit(_cli())
