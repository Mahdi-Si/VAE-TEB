r"""The acceptance protocol: several seeds, one predeclaration, and a partition nothing has read.

Run from the repository root:

.. code-block:: bash

    python -m teb_vae.lag_slot_transformer_cfs.eval.acceptance \
        --runs output/development-evals \
        --reference output/reference-eval/eval_results/summary.json \
        --output acceptance.json

From an IDE's Run button, with no command line: fill in ``RUN_ARGS`` at the bottom of this file.

**What this reads and what it refuses to read.** Its inputs are finished evaluation directories --
a ``summary.json`` and the per-recording table beside it -- and one frozen target-only run to read
them against. It rebuilds no model, opens no shard and imports no numeric stack beyond the array
library its interval already uses, which is what lets it run against artifacts copied off the box
that produced them.

**Why one run is not evidence.** A single fit's held-out gap is a draw from the optimiser as much
as a property of the architecture, so every comparison here is read over several training seeds of
one arm: the per-recording values are averaged across an arm's seeds and the interval is drawn over
recordings **once**, around that average. Resampling recordings inside each seed and averaging the
three intervals would report the spread of one seed rather than the spread the comparison has.

**Why the pairing is per recording.** Two arms scored on the same split under the same evaluation
seed saw the same recordings and the same draws, so their difference is available per recording and
its interval is drawn over the differences. An interval on each arm's own score would be far wider
than the interval on their difference, and reading two overlapping intervals as "no difference" is
the specific error the paired form removes.

**What multiplicity is applied to, and what it is not.** The five primary comparisons are declared
in the plan before any arm was trained, and each is reported at the nominal level. The lag bands
are a **search**: the band carrying the largest margin is chosen on the same data its interval is
built from. Those are additionally reported at a family-adjusted level whose family is the bands
actually searched, which covers every member of the family at once -- and a member selected out of
a simultaneously covered family keeps that coverage however it was chosen.

**The reserved partition.** A confirmation is a second set of runs on recordings that no run used
to choose an architecture, a hyperparameter or a threshold has ever been scored on. This pass does
not take that on trust: it compares the recordings the two sets actually scored and fails when they
intersect. Two evaluation directories can point at different files and still overlap -- a fold's
train partition contains another fold's test recordings -- so the file paths are reported and the
recordings are what decide.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: Repository root: ``teb_vae/lag_slot_transformer_cfs/eval/acceptance.py`` -> up four.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Launched as a script, this file's own directory goes on sys.path instead of the repository root.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from teb_vae.lag_attn.eval.report import json_safe  # noqa: E402
from teb_vae.lag_attn.eval.stats import bootstrap_ci  # noqa: E402
from teb_vae.lag_attn_cfs.eval.launch import (  # noqa: E402
    missing_required,
    resolve_launch_args,
)
from teb_vae.lag_slot_transformer_cfs.eval import verify  # noqa: E402

#: The committed predeclaration, used when no other is named.
DEFAULT_PLAN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "acceptance_plan.yaml")

#: The two files an evaluation directory is recognised by.
#:
#: Named here rather than imported from the pass that writes them, for the reason the gate pins the
#: qualification's phrases rather than the constant that produced them: this module reads artifacts
#: that a possibly older run wrote, and importing the writer would pull a numeric stack into a pass
#: whose whole point is not needing one. The suite asserts the two agree.
SUMMARY_FILENAME = "summary.json"
PER_RECORDING_FILENAME = "per_recording.csv"

#: The latent-probe artifact. Found anywhere under the same root and attached to the run whose
#: checkpoint it names, rather than to the directory it happens to sit in.
PROBE_FILENAME = "latent_probes.json"

#: Keys the plan may carry. Closed, matching the evaluation schema's discipline: an unrecognised
#: key raises and names the valid set, because nothing reads a misspelled one and a plan that
#: silently dropped its seed minimum would still produce a record.
PLAN_KEYS = frozenset({"declared_on", "revision", "protocol", "primary_comparisons", "exploratory_bands"})

#: Keys the plan's protocol block may carry.
PROTOCOL_KEYS = frozenset(
    {
        "minimum_training_seeds",
        "primary_draws",
        "stability_draws",
        "bootstrap_resamples",
        "bootstrap_seed",
        "confidence",
    }
)

#: Keys one declared comparison may carry.
COMPARISON_KEYS = frozenset({"name", "left", "right", "column", "isolates"})

#: The arm each combination of constructor leaves is. Keyed by
#: ``(source_stem, lag_fusion, mean_only_residual, source_values_withheld, source_scalar_lift)``
#: for a source-conditioned checkpoint; a target-only one is decided by its own leaf before this
#: mapping is consulted, since it reports no stem and no fusion to key on.
#:
#: A combination that is not here is a legitimate arm this protocol does not compare -- a scalar
#: lift, say -- and is reported with its leaves rather than guessed at or dropped.
ARM_LEAVES: Mapping[Tuple[Any, ...], str] = {
    ("pointwise", "local", False, False, False): "candidate",
    ("pointwise", "local", True, False, False): "mean_only",
    ("pointwise", "local", False, True, False): "capacity_control",
    ("pointwise", "attention", False, False, False): "pointwise_attention",
    ("conv", "attention", False, False, False): "attention_reference",
}

#: The source controls, and the column each one's margin is taken from.
#:
#: The silence arm is here although its margin is the gap by construction: an arm whose value is a
#: verified identity belongs in the table that verifies it, and a reader who finds it missing
#: cannot tell a passing identity from an unrun one.
CONTROL_COLUMNS: Mapping[str, str] = {
    "silence": "nll_silence",
    "replace_zeros": "nll_replace:zeros",
    "replace_constant": "nll_replace:constant",
    "permute": "nll_permute",
}

#: Prefix marking a per-recording column as one band's suppressed score.
SUPPRESSION_COLUMN_PREFIX = "nll_suppress:"

#: The two suppression arms that are identities rather than members of the search: the empty band
#: removes nothing and reproduces the matched arm, and the full band removes everything and
#: reproduces the prior. Neither is a lag the search ranges over, and including either in the
#: family would widen every other band's adjusted interval to pay for a constant.
IDENTITY_BANDS = frozenset({"none", "all"})


def _verdict(name: str, status: str, detail: str, **numbers: Any) -> Dict[str, Any]:
    """Assemble one verdict record, in the shape the single-run gate reports.

    Args:
        name: The verdict's name.
        status: ``'PASS'``, ``'FAIL'`` or ``'INCONCLUSIVE'``.
        detail: One sentence a reader can act on.
        **numbers: The measurements it was reached from.

    Returns:
        The record.
    """
    return {"name": name, "status": status, "detail": detail, **numbers}


# =============================================================================
# The predeclaration
# =============================================================================
def load_plan(path: Optional[str] = None) -> Dict[str, Any]:
    """Read the predeclaration and stamp it with the digest of the bytes that were read.

    The digest is the mechanism. Nothing here prevents the file being edited between two
    campaigns; what it does is make an acceptance record name the exact text its comparisons,
    bands and seed minimum came from, so a record produced under a revised plan cannot be mistaken
    for one produced under the original.

    Args:
        path: The plan to read, or ``None`` for the committed one.

    Returns:
        The parsed plan, with ``digest`` and ``path`` added.

    Raises:
        ValueError: If the plan carries an unknown key, omits a required block, or declares a
            comparison against an arm this package does not build.
    """
    resolved = DEFAULT_PLAN_PATH if path is None else str(path)
    raw = open(resolved, "rb").read()
    parsed = yaml.safe_load(raw.decode("utf-8")) or {}
    if not isinstance(parsed, Mapping):
        raise ValueError(f"the acceptance plan must be a mapping, got {type(parsed).__name__}.")

    unknown = sorted(set(parsed) - PLAN_KEYS)
    if unknown:
        raise ValueError(
            f"unknown acceptance plan key(s): {', '.join(repr(key) for key in unknown)}. "
            f"Valid keys are: {', '.join(sorted(PLAN_KEYS))}."
        )
    protocol = parsed.get("protocol") or {}
    missing = sorted(PROTOCOL_KEYS - set(protocol))
    if missing:
        raise ValueError(
            f"the acceptance plan's protocol block omits {', '.join(missing)}. Every setting a "
            f"verdict depends on is declared rather than defaulted, so a record cannot be produced "
            f"under a value nobody wrote down."
        )
    extra = sorted(set(protocol) - PROTOCOL_KEYS)
    if extra:
        raise ValueError(
            f"unknown protocol key(s): {', '.join(extra)}. Valid keys are: "
            f"{', '.join(sorted(PROTOCOL_KEYS))}."
        )

    known_arms = set(ARM_LEAVES.values()) | {"target_only"}
    for entry in parsed.get("primary_comparisons") or []:
        unknown_fields = sorted(set(entry) - COMPARISON_KEYS)
        if unknown_fields:
            raise ValueError(
                f"comparison {entry.get('name')!r} carries unknown field(s): "
                f"{', '.join(unknown_fields)}."
            )
        for side in ("left", "right"):
            if entry.get(side) not in known_arms:
                raise ValueError(
                    f"comparison {entry.get('name')!r} names {entry.get(side)!r} as its {side} "
                    f"arm, which is not one this package builds. Known arms: "
                    f"{', '.join(sorted(known_arms))}."
                )
    return {
        **parsed,
        "path": resolved,
        "digest": hashlib.sha256(raw).hexdigest()[:16],
    }


# =============================================================================
# The evidence
# =============================================================================
def arm_of(summary: Mapping[str, Any]) -> str:
    """Which arm produced this summary, resolved from its own recorded leaves.

    From the leaves rather than from the directory the run was written into: a directory is named
    by whoever launched the run and a leaf is what the model was constructed with, and only one of
    the two is wrong when they disagree.

    Args:
        summary: The parsed summary.

    Returns:
        The arm's name, or ``'unrecognised'`` prefixed with the leaves that were found.
    """
    arm = summary.get("arm") or {}
    if bool(arm.get("source_disabled")):
        return "target_only"
    key = (
        arm.get("source_stem"),
        arm.get("lag_fusion"),
        bool(arm.get("mean_only_residual")),
        bool(arm.get("source_values_withheld")),
        bool(arm.get("source_scalar_lift")),
    )
    return ARM_LEAVES.get(key, f"unrecognised:{key}")


def read_table(path: str) -> Dict[str, Dict[str, float]]:
    """Read one run's per-recording table.

    Args:
        path: The CSV the scoring pass wrote.

    Returns:
        ``{recording: {column: value}}``, with unparseable cells dropped rather than coerced.
    """
    table: Dict[str, Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            guid = str(row.pop("guid", "")).strip()
            if not guid:
                continue
            values: Dict[str, float] = {}
            for name, cell in row.items():
                try:
                    values[str(name)] = float(cell)
                except (TypeError, ValueError):
                    continue
            table[guid] = values
    return table


def discover_runs(root: str) -> List[Dict[str, Any]]:
    """Every evaluation directory under a root, as the records this pass compares.

    A directory qualifies when it holds a summary. The per-recording table beside it is what every
    interval here is built from, and a run without one is discovered and reported with the table
    marked absent rather than read at its point estimates: an interval over recordings needs the
    recordings.

    Args:
        root: The directory to walk.

    Returns:
        One record per run, sorted by path so two invocations report in the same order.
    """
    summaries: List[Tuple[str, Dict[str, Any]]] = []
    probes: Dict[str, Any] = {}
    for directory, _subdirectories, files in sorted(os.walk(str(root))):
        if SUMMARY_FILENAME in files:
            with open(os.path.join(directory, SUMMARY_FILENAME), "r", encoding="utf-8") as handle:
                summaries.append((directory, json.load(handle)))
        if PROBE_FILENAME in files:
            with open(os.path.join(directory, PROBE_FILENAME), "r", encoding="utf-8") as handle:
                artifact = json.load(handle)
            probes[str((artifact.get("run") or {}).get("checkpoint", ""))] = artifact

    # A probe artifact is attached to a run by the CHECKPOINT both name, not by the directory it
    # was written into. The two passes write independently and either can be pointed anywhere, so
    # a directory convention would silently attach nothing the first time an operator gave the
    # probe its own output directory -- and an arm would then report an unprobed latent.
    found: List[Dict[str, Any]] = []
    for directory, summary in summaries:
        table_path = os.path.join(directory, PER_RECORDING_FILENAME)
        found.append(
            {
                "directory": directory,
                "summary": summary,
                "table": read_table(table_path) if os.path.isfile(table_path) else None,
                "probes": probes.get(str((summary.get("run") or {}).get("checkpoint", ""))),
                "arm": arm_of(summary),
                "training_seed": (summary.get("run") or {}).get("training_seed"),
                "training_tag": (summary.get("run") or {}).get("training_tag"),
                "eval_seed": (summary.get("run") or {}).get("seed"),
                "draws": (summary.get("draws") or {}).get("num_mc_samples"),
                "split": summary.get("scored_split") or {},
            }
        )
    return sorted(found, key=lambda record: record["directory"])


def run_descriptor(run: Mapping[str, Any]) -> Dict[str, Any]:
    """The identity of one run, as the record lists it.

    Args:
        run: A discovered run.

    Returns:
        The descriptor.
    """
    split = run.get("split") or {}
    return {
        "directory": run["directory"],
        "arm": run["arm"],
        "training_seed": run.get("training_seed"),
        "training_tag": run.get("training_tag"),
        "eval_seed": run.get("eval_seed"),
        "num_mc_samples": run.get("draws"),
        "split_label": split.get("label"),
        "recording_digest": split.get("recording_digest"),
        "n_recordings": split.get("n_recordings"),
        "has_per_recording_table": run.get("table") is not None,
        "has_latent_probes": run.get("probes") is not None,
    }


def scoring_mismatch(runs: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """Whether these runs were scored under settings that make their difference a measurement.

    Two runs compared per recording must have seen the same recordings under the same evaluation
    seed and the same draw count, or their difference carries the difference between two draw sets
    and two estimators as well as the difference between two models.

    Args:
        runs: The runs about to be compared.

    Returns:
        A sentence naming the first mismatch, or ``None``.
    """
    for field in ("eval_seed", "draws"):
        values = {record.get(field) for record in runs}
        if len(values) > 1:
            return (
                f"the runs were scored at more than one {field}: {sorted(map(str, values))}. A "
                f"difference across two of these is a difference of estimators as well as of "
                f"models."
            )
    labels = {(record.get("split") or {}).get("label") for record in runs}
    if len(labels) > 1:
        return (
            f"the runs scored more than one split: {sorted(map(str, labels))}. A difference "
            f"across two splits is a difference of cohorts."
        )
    return None


# =============================================================================
# The arithmetic
# =============================================================================
def seed_average(
    runs: Sequence[Mapping[str, Any]], column: str
) -> Dict[str, float]:
    """Average one column across an arm's training seeds, per recording.

    Only recordings every seed scored. A recording one seed missed would otherwise be represented
    by the seeds that saw it, and the average would be over a different number of fits in different
    rows of the same vector.

    Args:
        runs: One arm's runs, one per training seed.
        column: The per-recording column to average.

    Returns:
        ``{recording: mean over seeds}``.
    """
    tables = [record["table"] for record in runs if record.get("table")]
    if not tables:
        return {}
    shared = set(tables[0])
    for table in tables[1:]:
        shared &= set(table)
    averaged: Dict[str, float] = {}
    for guid in shared:
        values = [table[guid].get(column) for table in tables]
        if any(value is None or not np.isfinite(value) for value in values):
            continue
        averaged[guid] = float(np.mean(values))
    return averaged


def interval(
    values: Mapping[str, float],
    *,
    plan: Mapping[str, Any],
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """A recording-level bootstrap interval over one per-recording vector, with its spread.

    The interval is the mean's; the quantiles beside it are the recordings' own distribution, and
    the two are different statements. A mean whose interval excludes zero can sit on a population
    where a third of the recordings have the opposite sign, and a protocol that reported only the
    interval would describe that population as uniform.

    Args:
        values: ``{recording: value}``.
        plan: The loaded plan, for the resamples and the seed.
        confidence: Coverage, or ``None`` for the plan's nominal level.

    Returns:
        The bootstrap record with a ``spread`` block added.
    """
    protocol = plan["protocol"]
    sample = [float(value) for value in values.values()]
    record = bootstrap_ci(
        sample,
        confidence=float(protocol["confidence"] if confidence is None else confidence),
        resamples=int(protocol["bootstrap_resamples"]),
        seed=int(protocol["bootstrap_seed"]),
    )
    finite = np.asarray([value for value in sample if np.isfinite(value)], dtype=np.float64)
    record["spread"] = (
        {}
        if finite.size == 0
        else {
            "min": float(finite.min()),
            "q25": float(np.quantile(finite, 0.25)),
            "median": float(np.quantile(finite, 0.5)),
            "q75": float(np.quantile(finite, 0.75)),
            "max": float(finite.max()),
            "fraction_negative": float((finite < 0.0).mean()),
        }
    )
    return record


def paired_difference(
    left: Mapping[str, float], right: Mapping[str, float]
) -> Dict[str, float]:
    """The per-recording difference of two seed-averaged vectors, on the recordings both hold.

    Args:
        left: The left-hand vector.
        right: The right-hand vector.

    Returns:
        ``{recording: left - right}``, empty when the two share no recording.
    """
    return {guid: left[guid] - right[guid] for guid in sorted(set(left) & set(right))}


def searched_bands(runs: Sequence[Mapping[str, Any]]) -> List[str]:
    """The lag bands these runs actually suppressed, excluding the two identity arms.

    Args:
        runs: One arm's runs.

    Returns:
        The band names, sorted.
    """
    names: set = set()
    for record in runs:
        for row in (record.get("table") or {}).values():
            names |= {
                column[len(SUPPRESSION_COLUMN_PREFIX):]
                for column in row
                if column.startswith(SUPPRESSION_COLUMN_PREFIX)
            }
            # One row names every column the table has; the rest would repeat it.
            break
    return sorted(names - IDENTITY_BANDS)


# =============================================================================
# The blocks a record carries
# =============================================================================
def arm_evidence(runs: Sequence[Mapping[str, Any]], *, plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Group the runs by arm at the plan's primary draw count, and say which arms qualify.

    Args:
        runs: Every discovered run.
        plan: The loaded plan.

    Returns:
        ``{arm: {'runs', 'training_seeds', 'meets_minimum'}}``.
    """
    minimum = int(plan["protocol"]["minimum_training_seeds"])
    primary = int(plan["protocol"]["primary_draws"])
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for record in runs:
        if record.get("table") is None or int(record.get("draws") or -1) != primary:
            continue
        grouped.setdefault(record["arm"], []).append(record)

    evidence: Dict[str, Any] = {}
    for arm, members in sorted(grouped.items()):
        # One run per training seed. Two runs of one seed are two scorings of one fit, and counting
        # them as two seeds would let a rescored checkpoint stand in for a repeated experiment.
        by_seed: Dict[Any, Mapping[str, Any]] = {}
        for record in members:
            by_seed.setdefault(record.get("training_seed"), record)
        evidence[arm] = {
            "runs": [run_descriptor(record) for record in by_seed.values()],
            "training_seeds": sorted(str(seed) for seed in by_seed),
            "n_training_seeds": len(by_seed),
            "meets_minimum": len(by_seed) >= minimum,
            "_members": list(by_seed.values()),
        }
    return evidence


def primary_comparison_block(
    evidence: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> Dict[str, Any]:
    """Every declared comparison, read over the seeds each arm has.

    Args:
        evidence: The grouped arm evidence.
        plan: The loaded plan.

    Returns:
        ``{comparison name: record}``.
    """
    minimum = int(plan["protocol"]["minimum_training_seeds"])
    block: Dict[str, Any] = {}
    for entry in plan.get("primary_comparisons") or []:
        name, left, right = entry["name"], entry["left"], entry["right"]
        column = entry.get("column", "nll_full")
        record: Dict[str, Any] = {
            "left_arm": left,
            "right_arm": right,
            "column": column,
            "isolates": entry.get("isolates", ""),
        }
        if left not in evidence or right not in evidence:
            record["status"] = "NO_EVIDENCE"
            record["detail"] = (
                f"no run of {left if left not in evidence else right} was found at the declared "
                f"draw count, so this comparison has nothing to read."
            )
            block[name] = record
            continue

        members = list(evidence[left]["_members"]) + list(evidence[right]["_members"])
        mismatch = scoring_mismatch(members)
        if mismatch is not None:
            record["status"] = "UNMATCHED"
            record["detail"] = mismatch
            block[name] = record
            continue

        differences = paired_difference(
            seed_average(evidence[left]["_members"], column),
            seed_average(evidence[right]["_members"], column),
        )
        if not differences:
            record["status"] = "NO_SHARED_RECORDINGS"
            record["detail"] = (
                "the two arms share no scored recording, so no paired difference exists. Two "
                "runs on two partitions are the usual cause and they cannot be compared this way."
            )
            block[name] = record
            continue

        enough = evidence[left]["meets_minimum"] and evidence[right]["meets_minimum"]
        record.update(
            {
                "status": "READ" if enough else "BELOW_SEED_MINIMUM",
                "n_training_seeds": {
                    left: evidence[left]["n_training_seeds"],
                    right: evidence[right]["n_training_seeds"],
                },
                "difference_nats": interval(differences, plan=plan),
                "detail": (
                    "a negative difference favours the left arm, since every column compared "
                    "here is a negative log density in nats per anchor."
                    if enough
                    else f"read, and below the declared minimum of {minimum} training seeds. The "
                    f"interval is over recordings and says nothing about the spread across fits."
                ),
            }
        )
        block[name] = record
    return block


def internal_and_reference_block(
    evidence: Mapping[str, Any],
    reference: Optional[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
) -> Dict[str, Any]:
    """Each arm's own gap, and its two comparisons against the frozen reference.

    The gap against an arm's own base branch and the margin against an independently trained
    predictor answer different questions, and the second exists because the first cannot separate
    a source that helped from a base that got worse while it was being trained. The base drift is
    the third number and it is the one that can fail.

    Args:
        evidence: The grouped arm evidence.
        reference: The frozen target-only run, or ``None``.
        plan: The loaded plan.

    Returns:
        ``{arm: record}``.
    """
    block: Dict[str, Any] = {}
    reference_base = (
        {} if reference is None else seed_average([reference], "nll_base")
    )
    for arm, group in evidence.items():
        members = group["_members"]
        record: Dict[str, Any] = {
            "n_training_seeds": group["n_training_seeds"],
            "internal_gap_nats": interval(seed_average(members, "pred_gap"), plan=plan),
            "divergence_per_anchor": interval(
                seed_average(members, "kld_per_anchor"), plan=plan
            ),
        }
        if arm == "target_only":
            record["note"] = (
                "a target-only arm's gap is exactly zero by construction and is not a "
                "measurement. What it contributes is its own predictive score."
            )
        if reference_base:
            mismatch = scoring_mismatch(list(members) + [reference])
            if mismatch is not None:
                record["against_reference"] = {"status": "UNMATCHED", "detail": mismatch}
            else:
                record["against_reference"] = {
                    "status": "READ",
                    "full_minus_reference_nats": interval(
                        paired_difference(seed_average(members, "nll_full"), reference_base),
                        plan=plan,
                    ),
                    "base_minus_reference_nats": interval(
                        paired_difference(seed_average(members, "nll_base"), reference_base),
                        plan=plan,
                    ),
                    "detail": (
                        "both are differences of negative log densities against the frozen "
                        "reference, so negative favours this arm. The second is the one that "
                        "decides whether the first means anything: a base branch that fell "
                        "behind the reference during joint training makes every gap measured "
                        "against it a measurement of that."
                    ),
                }
        else:
            record["against_reference"] = {
                "status": "NO_REFERENCE",
                "detail": (
                    "no frozen target-only run was given, so no comparison against an "
                    "independently trained predictor was made."
                ),
            }
        block[arm] = record
    return block


def control_block(
    evidence: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> Dict[str, Any]:
    """Each arm's source controls, as intervals rather than as differences of point estimates.

    Args:
        evidence: The grouped arm evidence.
        plan: The loaded plan.

    Returns:
        ``{arm: {control: record}}``.
    """
    block: Dict[str, Any] = {}
    for arm, group in evidence.items():
        members = group["_members"]
        matched = seed_average(members, "nll_full")
        controls: Dict[str, Any] = {}
        for name, column in CONTROL_COLUMNS.items():
            values = seed_average(members, column)
            if not values:
                controls[name] = {
                    "status": "NOT_RUN",
                    "detail": (
                        "this arm ran no such intervention, which is a different statement from "
                        "an intervention that changed nothing."
                    ),
                }
                continue
            controls[name] = {
                "status": "READ",
                "margin_nats": interval(paired_difference(values, matched), plan=plan),
            }
        block[arm] = {
            "controls": controls,
            "detail": (
                "each margin is the intervened arm's score less the matched one's, so a POSITIVE "
                "margin means the fitted model predicts worse without what the intervention "
                "removed. The silence arm is an identity: its margin equals the gap exactly."
            ),
        }
    return block


def band_block(
    evidence: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> Dict[str, Any]:
    """The lag-band search, at the nominal level and at the family-adjusted one.

    Args:
        evidence: The grouped arm evidence.
        plan: The loaded plan.

    Returns:
        ``{arm: record}``.
    """
    declared = list(plan.get("exploratory_bands") or [])
    nominal = float(plan["protocol"]["confidence"])
    block: Dict[str, Any] = {}
    for arm, group in evidence.items():
        members = group["_members"]
        names = searched_bands(members)
        if not names:
            block[arm] = {
                "status": "NOT_SEARCHED",
                "detail": "this arm suppressed no lag band, so there was no search to correct.",
            }
            continue
        undeclared = sorted(set(names) - set(declared))
        family = len(names)
        adjusted = 1.0 - (1.0 - nominal) / float(family)
        matched = seed_average(members, "nll_full")
        bands: Dict[str, Any] = {}
        for name in names:
            margins = paired_difference(
                seed_average(members, f"{SUPPRESSION_COLUMN_PREFIX}{name}"), matched
            )
            if not margins:
                bands[name] = {"status": "NO_SHARED_RECORDINGS"}
                continue
            bands[name] = {
                "status": "READ",
                "margin_nats": interval(margins, plan=plan),
                "margin_nats_family_adjusted": interval(
                    margins, plan=plan, confidence=adjusted
                ),
            }
        read = {
            name: record
            for name, record in bands.items()
            if record.get("status") == "READ"
            and np.isfinite(record["margin_nats"].get("point", float("nan")))
        }
        peak = (
            None
            if not read
            else max(read, key=lambda name: read[name]["margin_nats"]["point"])
        )
        block[arm] = {
            "status": "READ",
            "declared_bands": declared,
            "searched_bands": names,
            "undeclared_bands": undeclared,
            "family_size": family,
            "nominal_confidence": nominal,
            "family_adjusted_confidence": adjusted,
            "peak_band": peak,
            "bands": bands,
            "detail": (
                "the peak band was chosen on the same recordings its interval is built from, so "
                "the family-adjusted interval is the one a claim about it rests on. The margins "
                "do not decompose the gap and are not normalised to it, and a band's margin is a "
                "property of this fitted parameterisation rather than a physiological delay or a "
                "unique contribution of the stored source times it names."
            ),
        }
    return block


def stability_block(
    runs: Sequence[Mapping[str, Any]], *, plan: Mapping[str, Any]
) -> Dict[str, Any]:
    """How far each arm's gap moves with the Monte Carlo draw count.

    Reported per arm over whichever declared draw counts were run. The negative logarithm of an
    average likelihood is upward biased at finite draws and two branches' biases need not cancel,
    so a gap that moves with the draw count is a gap whose reading depends on the estimator.

    Args:
        runs: Every discovered run, at every draw count.
        plan: The loaded plan.

    Returns:
        ``{arm: record}``.
    """
    declared = [int(value) for value in plan["protocol"]["stability_draws"]]
    grouped: Dict[str, Dict[int, List[Mapping[str, Any]]]] = {}
    for record in runs:
        if record.get("table") is None:
            continue
        draws = int(record.get("draws") or -1)
        if draws not in declared:
            continue
        grouped.setdefault(record["arm"], {}).setdefault(draws, []).append(record)

    block: Dict[str, Any] = {}
    for arm, by_draws in sorted(grouped.items()):
        points = {}
        for draws in sorted(by_draws):
            values = seed_average(by_draws[draws], "pred_gap")
            points[str(draws)] = None if not values else float(np.mean(list(values.values())))
        measured = [value for value in points.values() if value is not None]
        block[arm] = {
            "gap_by_draw_count": points,
            "declared_draw_counts": declared,
            "missing_draw_counts": [
                draws for draws in declared if draws not in by_draws
            ],
            "range_nats": (
                None if len(measured) < 2 else float(max(measured) - min(measured))
            ),
        }
    return block


def calibration_block(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    """Each arm's mixture coverage, averaged over its seeds.

    Both branches, because a coverage statement about the source-conditioned branch alone cannot
    say whether the source improved it or whether the observation model was already miscalibrated
    without it.

    Args:
        evidence: The grouped arm evidence.

    Returns:
        ``{arm: {branch: {'pit_mean', 'coverage'}}}``.
    """
    block: Dict[str, Any] = {}
    for arm, group in evidence.items():
        branches: Dict[str, Any] = {}
        for branch in ("base", "full"):
            blocks = [
                (record["summary"].get("calibration") or {}).get(branch) or {}
                for record in group["_members"]
            ]
            populated = [entry for entry in blocks if entry.get("coverage")]
            if not populated:
                continue
            levels = sorted(populated[0]["coverage"])
            branches[branch] = {
                "pit_mean": float(
                    np.mean([float(entry.get("pit_mean", float("nan"))) for entry in populated])
                ),
                "coverage": {
                    level: float(
                        np.mean([float(entry["coverage"][level]) for entry in populated])
                    )
                    for level in levels
                },
                "n_training_seeds": len(populated),
            }
        block[arm] = branches
    return block


def probe_block(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    """Each arm's latent probes, averaged over the seeds that ran them.

    Absent rather than zero where no probe artifact was written: a latent nothing probed and a
    latent a probe found nothing in are the two readings this block must keep apart.

    Args:
        evidence: The grouped arm evidence.

    Returns:
        ``{arm: record}``.
    """
    block: Dict[str, Any] = {}
    for arm, group in evidence.items():
        artifacts = [record["probes"] for record in group["_members"] if record.get("probes")]
        if not artifacts:
            block[arm] = {
                "status": "NOT_RUN",
                "detail": (
                    "no latent-probe artifact was found beside this arm's summaries. Whether the "
                    "latent carries the future is a separate measurement from whether the "
                    "predictive score improved, and this record makes no claim about it."
                ),
            }
            continue
        names = sorted(artifacts[0].get("probes") or {})
        block[arm] = {
            "status": "READ",
            "n_training_seeds": len(artifacts),
            "r2": {
                name: float(
                    np.mean(
                        [
                            float((entry.get("probes") or {}).get(name, {}).get("r2", np.nan))
                            for entry in artifacts
                        ]
                    )
                )
                for name in names
            },
            "detail": (
                "the fraction of the anchor-relative future block linearly readable from each "
                "readout on held-out recordings. A lower bound on what the latent holds, not a "
                "measurement of what the decoder uses."
            ),
        }
    return block


# =============================================================================
# The verdicts
# =============================================================================
def check_runs_are_sound(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Every run in the evidence passes its own single-run gate.

    The protocol is built on top of that gate rather than beside it: an acceptance record assembled
    from a run whose reference arms drifted, or whose lag readouts lost their qualification, would
    be a careful aggregation of a broken measurement.

    Args:
        runs: The discovered runs.

    Returns:
        The verdict.
    """
    broken: Dict[str, List[str]] = {}
    for record in runs:
        result = verify.verify(record["summary"])
        if result["failed"]:
            broken[record["directory"]] = result["failed"]
    if broken:
        return _verdict(
            "every_run_passes_its_own_gate",
            "FAIL",
            "one or more runs in this evidence fail the single-run gate, so the protocol is "
            "aggregating a measurement that is already known to be wrong.",
            failures=broken,
        )
    return _verdict(
        "every_run_passes_its_own_gate",
        "PASS",
        f"all {len(runs)} runs pass the structural checks of the single-run gate.",
        n_runs=len(runs),
    )


def check_seed_minimum(
    evidence: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> Dict[str, Any]:
    """Every arm with evidence has at least the declared number of training seeds.

    Args:
        evidence: The grouped arm evidence.
        plan: The loaded plan.

    Returns:
        The verdict.
    """
    minimum = int(plan["protocol"]["minimum_training_seeds"])
    counts = {arm: group["n_training_seeds"] for arm, group in evidence.items()}
    short = {arm: count for arm, count in counts.items() if count < minimum}
    if not counts:
        return _verdict(
            "training_seeds",
            "INCONCLUSIVE",
            f"no run was found at the declared primary draw count of "
            f"{plan['protocol']['primary_draws']}.",
            training_seeds=counts,
        )
    if short:
        return _verdict(
            "training_seeds",
            "INCONCLUSIVE",
            f"{', '.join(sorted(short))} carry fewer than the declared minimum of {minimum} "
            f"training seeds, so their comparisons are read and are not evidence of a stable "
            f"effect. One fit's held-out result is a draw from the optimiser as much as a "
            f"property of the architecture.",
            training_seeds=counts,
            minimum=minimum,
        )
    return _verdict(
        "training_seeds",
        "PASS",
        f"every arm carries at least {minimum} training seeds at the declared draw count.",
        training_seeds=counts,
        minimum=minimum,
    )


def check_declared_bands(bands: Mapping[str, Any]) -> Dict[str, Any]:
    """No band was searched that the plan does not declare.

    A search whose family grew after its peak was seen is a search with no multiplicity at all,
    and it is invisible in the numbers it produces.

    Args:
        bands: The band block.

    Returns:
        The verdict.
    """
    offenders = {
        arm: record["undeclared_bands"]
        for arm, record in bands.items()
        if record.get("undeclared_bands")
    }
    if offenders:
        return _verdict(
            "bands_were_predeclared",
            "FAIL",
            "a band outside the plan's declared list was searched, so the family the correction "
            "is applied over was not fixed before the peak was seen.",
            undeclared=offenders,
        )
    return _verdict(
        "bands_were_predeclared",
        "PASS",
        "every searched band is one the plan declares, so the family the correction covers was "
        "fixed before any margin was read.",
        searched={arm: record.get("searched_bands", []) for arm, record in bands.items()},
    )


def check_base_not_degraded(block: Mapping[str, Any]) -> Dict[str, Any]:
    """No arm's own base branch fell behind the frozen reference.

    Stricter than the single-run gate in one way and weaker in another, both deliberate. That gate
    compares two point estimates from two runs and fails on the sign alone, which is the only
    comparison one summary supports. Here the drift has a paired interval over recordings, so the
    verdict fails on a **confident** drift and reports one whose interval spans zero as what it is:
    a point estimate on the wrong side of zero and a measurement that cannot tell.

    Args:
        block: The internal-and-reference block.

    Returns:
        The verdict.
    """
    confident: Dict[str, Any] = {}
    unclear: Dict[str, Any] = {}
    for arm, record in block.items():
        against = record.get("against_reference") or {}
        drift = against.get("base_minus_reference_nats")
        if not drift or not np.isfinite(drift.get("point", float("nan"))):
            continue
        # Scores are negative log densities: a POSITIVE difference means this arm's base branch
        # predicts worse than the reference.
        if float(drift["point"]) <= 0.0:
            continue
        if float(drift.get("lo", float("-inf"))) > 0.0:
            confident[arm] = drift
        else:
            unclear[arm] = drift
    if confident:
        return _verdict(
            "base_not_degraded",
            "FAIL",
            "one or more arms' own base branches score confidently worse than the frozen "
            "reference, so joint training left the baseline behind and any gap measured against "
            "it is measuring that.",
            degraded={arm: drift["point"] for arm, drift in confident.items()},
        )
    if unclear:
        return _verdict(
            "base_not_degraded",
            "INCONCLUSIVE",
            "one or more arms' base branches sit on the worse side of the frozen reference by a "
            "margin whose interval spans zero. Not a failure and not a clearance: more seeds or "
            "more recordings are what separates the two.",
            unclear={arm: drift["point"] for arm, drift in unclear.items()},
        )
    return _verdict(
        "base_not_degraded",
        "PASS",
        "no arm's base branch scores worse than the frozen reference, so a gap measured against "
        "it is not measuring a degraded baseline.",
    )


def check_confirmation_partition(
    selection: Sequence[Mapping[str, Any]], confirmation: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """The confirmation runs scored recordings no selection run has ever been scored on.

    Decided on the recordings rather than on the file paths. Two evaluation directories can name
    different shards and still overlap -- a cross-validation fold's train partition holds another
    fold's test recordings -- so the paths are reported and the identifiers decide.

    Args:
        selection: The runs every choice was made on.
        confirmation: The runs on the reserved partition, possibly empty.

    Returns:
        The verdict.
    """
    if not confirmation:
        return _verdict(
            "confirmation_partition",
            "INCONCLUSIVE",
            "no confirmation runs were given. Every number in this record comes from the runs "
            "that architecture, hyperparameters and thresholds were chosen on, so it is a "
            "development result and confirms nothing.",
        )
    chosen_on: set = set()
    for record in selection:
        chosen_on |= set(record.get("table") or {})
    reserved: set = set()
    for record in confirmation:
        reserved |= set(record.get("table") or {})
    shared = sorted(chosen_on & reserved)
    labels = {
        "selection": sorted({str((r.get("split") or {}).get("label")) for r in selection}),
        "confirmation": sorted({str((r.get("split") or {}).get("label")) for r in confirmation}),
    }
    if shared:
        return _verdict(
            "confirmation_partition",
            "FAIL",
            f"{len(shared)} recording(s) appear in both sets, so the partition the confirmation "
            f"runs on is not one the selection has never seen.",
            n_shared_recordings=len(shared),
            example_shared=shared[:5],
            labels=labels,
        )
    return _verdict(
        "confirmation_partition",
        "PASS",
        f"the {len(reserved)} confirmation recordings and the {len(chosen_on)} selection "
        f"recordings are disjoint, so the confirmation reads a partition no choice was made on.",
        n_selection_recordings=len(chosen_on),
        n_confirmation_recordings=len(reserved),
        labels=labels,
    )


# =============================================================================
# The pass
# =============================================================================
def analyse(
    runs: Sequence[Mapping[str, Any]],
    reference: Optional[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
) -> Dict[str, Any]:
    """Everything one set of runs supports, under one predeclaration.

    Args:
        runs: The discovered runs.
        reference: The frozen target-only run, or ``None``.
        plan: The loaded plan.

    Returns:
        The analysis block, with the private member lists stripped.
    """
    evidence = arm_evidence(runs, plan=plan)
    bands = band_block(evidence, plan=plan)
    block = {
        "arms": {
            arm: {key: value for key, value in group.items() if not key.startswith("_")}
            for arm, group in evidence.items()
        },
        "primary_comparisons": primary_comparison_block(evidence, plan=plan),
        "per_arm": internal_and_reference_block(evidence, reference, plan=plan),
        "source_controls": control_block(evidence, plan=plan),
        "exploratory_bands": bands,
        "monte_carlo_stability": stability_block(runs, plan=plan),
        "calibration": calibration_block(evidence),
        "latent_probes": probe_block(evidence),
        "n_runs": len(runs),
    }
    block["verdicts"] = [
        check_runs_are_sound(runs),
        check_seed_minimum(evidence, plan=plan),
        check_declared_bands(bands),
        check_base_not_degraded(block["per_arm"]),
    ]
    return block


def read_reference(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load the frozen target-only run, and refuse a source-conditioned one in its place.

    Args:
        path: The reference summary's path, or ``None``.

    Returns:
        The run record, or ``None``.

    Raises:
        ValueError: If the named run is not target-only, or has no per-recording table.
    """
    if path is None:
        return None
    directory = os.path.dirname(os.path.abspath(str(path)))
    with open(str(path), "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not verify.is_target_only(summary):
        raise ValueError(
            f"{path} is not a target-only run. A source-conditioned model cannot serve as the "
            f"baseline its own architecture is measured against."
        )
    table_path = os.path.join(directory, PER_RECORDING_FILENAME)
    if not os.path.isfile(table_path):
        raise ValueError(
            f"{table_path} is missing. The reference is compared per recording, so its "
            f"per-recording table is what the comparison is built from."
        )
    return {
        "directory": directory,
        "summary": summary,
        "table": read_table(table_path),
        "probes": None,
        "arm": arm_of(summary),
        "training_seed": (summary.get("run") or {}).get("training_seed"),
        "training_tag": (summary.get("run") or {}).get("training_tag"),
        "eval_seed": (summary.get("run") or {}).get("seed"),
        "draws": (summary.get("draws") or {}).get("num_mc_samples"),
        "split": summary.get("scored_split") or {},
    }


def assess(
    runs: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    reference: Optional[Mapping[str, Any]] = None,
    confirmation: Sequence[Mapping[str, Any]] = (),
    confirmation_reference: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the whole record: the development answer, the confirmation, and the verdicts.

    Args:
        runs: The runs every choice was made on.
        plan: The loaded plan.
        reference: The frozen target-only run those were scored against, or ``None``.
        confirmation: The runs on the reserved partition, possibly empty.
        confirmation_reference: The frozen reference scored on that partition, or ``None``.

    Returns:
        The acceptance record.
    """
    record: Dict[str, Any] = {
        "plan": {
            "path": plan.get("path"),
            "digest": plan.get("digest"),
            "declared_on": plan.get("declared_on"),
            "revision": plan.get("revision"),
            "protocol": plan.get("protocol"),
            "note": (
                "the digest is of the plan file as it was read. A comparison, a band or a seed "
                "minimum changed after a result was seen is invisible in the numbers and changes "
                "this value, which is the only thing that makes the declaration checkable."
            ),
        },
        "runs": [run_descriptor(record) for record in runs],
        "reference": None if reference is None else run_descriptor(reference),
        "selection": analyse(runs, reference, plan=plan),
    }
    if confirmation:
        record["confirmation_runs"] = [run_descriptor(entry) for entry in confirmation]
        record["confirmation"] = analyse(
            confirmation, confirmation_reference or reference, plan=plan
        )
    record["verdicts"] = [
        *record["selection"]["verdicts"],
        check_confirmation_partition(runs, list(confirmation)),
    ]
    record["failed"] = [
        entry["name"] for entry in record["verdicts"] if entry["status"] == "FAIL"
    ]
    record["passed"] = not record["failed"]
    return record


def main(
    runs: Optional[str] = None,
    reference: Optional[str] = None,
    confirmation: Optional[str] = None,
    confirmation_reference: Optional[str] = None,
    plan: Optional[str] = None,
    output: Optional[str] = None,
    sources: Optional[Mapping[str, str]] = None,
) -> int:
    """Read the evidence under one predeclaration and report what it supports.

    Args:
        runs: Root holding the evaluation directories every choice was made on. Required,
            enforced here rather than by argparse.
        reference: The frozen target-only run's ``summary.json``, or ``None``.
        confirmation: Root holding the reserved partition's evaluation directories, or ``None``.
        confirmation_reference: The frozen reference scored on the reserved partition, or ``None``.
        plan: The predeclaration to read, or ``None`` for the committed one.
        output: Where to write the record, or ``None`` to print it.
        sources: Where each launch value came from, recorded in the output.

    Returns:
        ``0`` when nothing failed, ``1`` on a failure, ``2`` on a refusal.
    """
    refusal = missing_required({"runs": runs}, ("runs",))
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 2

    declaration = load_plan(plan)
    discovered = discover_runs(str(runs))
    if not discovered:
        print(
            f"no evaluation directory under {runs} carries a {SUMMARY_FILENAME}. This pass reads "
            f"finished evaluation directories rather than checkpoints.",
            file=sys.stderr,
        )
        return 2

    record = assess(
        discovered,
        plan=declaration,
        reference=read_reference(reference),
        confirmation=[] if confirmation is None else discover_runs(str(confirmation)),
        confirmation_reference=read_reference(confirmation_reference),
    )
    record["run"] = {"argument_sources": dict(sources or {})}

    text = json.dumps(json_safe(record), indent=2)
    if output is not None:
        with open(str(output), "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"wrote {output}", file=sys.stderr)
    else:
        print(text)
    for entry in record["verdicts"]:
        print(f"{entry['status']:<13} {entry['name']}: {entry['detail']}", file=sys.stderr)
    return 0 if record["passed"] else 1


#: Values used when the module is launched with no command line -- i.e. an IDE's Run button. Keyed
#: by argparse ``dest``; a flag always wins over the entry here, per key.
#:
#: ``runs`` MUST be filled in for this file to run at all. Every entry names an ARTIFACT rather than
#: a setting, which is what keeps each verdict a property of the runs: everything that decides one
#: -- the seed minimum, the draw count, the resamples, the comparisons, the bands -- is in the
#: predeclaration and travels into the record with its digest.
RUN_ARGS: Dict[str, Any] = {
    # REQUIRED. Root holding the evaluation directories every choice was made on.
    "runs": None,
    # The frozen, independently trained target-only run's summary.json.
    "reference": None,
    # Root holding the reserved partition's evaluation directories, once they exist.
    "confirmation": None,
    # The frozen reference scored on that same reserved partition.
    "confirmation_reference": None,
    # An alternative predeclaration, or None for the committed one.
    "plan": None,
    # Where to write the record, or None to print it.
    "output": None,
}


def build_parser() -> argparse.ArgumentParser:
    """Build this entry point's own parser.

    No ``required=True`` and no non-``None`` default, for the reasons the other entry points'
    parsers record: the first fires before the launch dict is read, and the second makes that key's
    entry unreachable while the operator edits it.

    Returns:
        The parser, whose ``dest`` set is also the valid key set for :data:`RUN_ARGS`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_slot_transformer_cfs.eval.acceptance",
        description="Read several evaluation runs under one predeclaration.",
    )
    parser.add_argument("--runs", default=None, help="Root of the evaluation directories.")
    parser.add_argument(
        "--reference", default=None, help="A frozen target-only run's summary.json."
    )
    parser.add_argument(
        "--confirmation", default=None, help="Root of the reserved partition's evaluations."
    )
    parser.add_argument(
        "--confirmation-reference",
        default=None,
        help="The frozen reference scored on the reserved partition.",
    )
    parser.add_argument("--plan", default=None, help="An alternative predeclaration.")
    parser.add_argument("--output", default=None, help="Where to write the record.")
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
    return main(**values, sources=sources)


if __name__ == "__main__":
    sys.exit(_cli())
