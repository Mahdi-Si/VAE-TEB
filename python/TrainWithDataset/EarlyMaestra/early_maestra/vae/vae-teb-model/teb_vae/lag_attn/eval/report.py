r"""``summary.json`` and the fail-soft step wrapper.

An eval run is a sequence of largely independent analyses over one checkpoint, and it can take
hours. One analysis raising should not discard the ten that already succeeded -- but it must
also not be possible to mistake the resulting run for a clean one. So each step runs inside
:meth:`Report.step`, which captures the failure and continues; every captured failure is
re-logged at ``ERROR`` when the summary is written; and the process exits non-zero.

Two details are load-bearing.

**The full traceback is captured, not ``str(exc)``.** For an unattended multi-hour run the
traceback is the entire debugging surface -- ``KeyError: 'mu_full'`` alone says nothing about
which of a dozen call sites produced it.

**``KeyboardInterrupt`` and ``SystemExit`` are not caught.** They inherit from
``BaseException``, not ``Exception``, so a bare ``except Exception`` already lets them through
-- this is stated because it is the property that makes Ctrl-C work, and a well-meant
``except BaseException`` would silently take it away and turn an interrupt into a "failed step"
that the run then continues past.
"""
from __future__ import annotations

import json
import math
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from loguru import logger

#: File written into the run directory.
SUMMARY_FILENAME = "summary.json"

#: Top-level ``results`` keys every completed run carries, whatever it found. A key that is
#: *absent* rather than null means an analysis did not reach the summary at all, which is a
#: different failure from an analysis that ran and had nothing to report -- so the schema is
#: asserted rather than left to whatever the run happened to produce.
REQUIRED_RESULT_KEYS: tuple = (
    "arguments",
    "artifacts",
    "checkpoint",
    "config",
    "eval_config",
    "geometry",
    "headline",
    "numerics",
    "objective",
    "output_dir",
    "preflight",
    "sanity",
)

#: Headline scalars, as ``(name, path into results)``. Flattened onto ``results['headline']`` so
#: a reader -- or a ``pandas`` merge across two runs -- does not need to know which analysis
#: produced which number. A path that does not resolve yields ``None`` rather than raising: an
#: analysis that failed or was skipped legitimately has no headline.
HEADLINE_SCALARS: tuple = (
    ("feat_mse", ("forecast", "mean_feat_mse_total")),
    ("feat_r2", ("forecast", "mean_feat_r2_total")),
    ("uplift_rel", ("uplift", "mean_uplift_rel")),
    ("uplift_positive_frac", ("uplift", "positive_fraction")),
    ("residual_ratio", ("residual", "mean_residual_ratio")),
    ("kld_mean", ("latent", "mean_kld_mean")),
    ("kld_active_frac", ("latent", "diagnostics", "kld_active_frac_masked")),
    ("median_argmax_lag", ("attention", "median_argmax_lag")),
    ("attention_entropy_nats", ("attention", "mean_entropy_nats")),
    ("nll_gain", ("calibration", "mean_nll_gain")),
    ("crps", ("calibration", "mean_crps")),
)

#: Verdicts promoted alongside the scalars, as ``(name, path)``. Strings, so they are reported
#: rather than range-checked.
HEADLINE_VERDICTS: tuple = (
    ("collapse", ("collapse", "verdict")),
    ("source_specificity", ("source_specificity", "verdict")),
    ("te_lag_map", ("te_lag", "te_lag_map_label")),
)


def _dig(results: Dict[str, Any], path: tuple) -> Any:
    """Follow a key path into nested dicts, returning ``None`` if any step is missing.

    Args:
        results: The accumulated results.
        path: Successive dict keys.

    Returns:
        The value, or ``None`` when the path does not resolve to one.
    """
    node: Any = results
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def json_safe(value: Any) -> Any:
    r"""Recursively convert a value into something ``json.dump`` accepts without ``allow_nan``.

    Applied *before* serialisation rather than passed as ``default=``, because ``default`` is
    consulted only for types the encoder does not recognise -- and ``float('nan')`` is
    recognised. Left alone, ``json.dump`` emits the bare token ``NaN``, which Python reads back
    but which is not valid JSON and which every strict parser rejects.

    Non-finite floats therefore become ``null``. That loses the NaN/Inf distinction, and it is
    the right trade: ``summary.json`` is read by humans and by ``pandas``, both of which handle
    ``null`` natively, and a metric that is NaN is reported as such by the analysis that
    produced it rather than by its serialised form.

    A ``torch.Tensor`` is converted through its own ``tolist``. The framework is reached for
    lazily and only when it is *already* imported: a tensor cannot exist unless ``torch`` is in
    ``sys.modules``, so the guard costs one dict lookup, and this module stays importable -- and
    testable -- on a box with no torch installed.

    Args:
        value: Any value destined for the summary.

    Returns:
        A structure of dicts, lists, strings, bools, ints, floats and ``None``.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    # Anything else -- a tensor, a dataclass, a device -- is recorded by its repr rather than
    # dropped, so an unexpected type shows up in the output instead of failing the write at the
    # end of a multi-hour run.
    return str(value)


def build_headline(results: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the run's headline scalars and verdicts out of the per-analysis blocks.

    Args:
        results: The accumulated results.

    Returns:
        Name to value, with ``None`` wherever the producing analysis did not report.
    """
    headline: Dict[str, Any] = {
        name: _dig(results, path) for name, path in HEADLINE_SCALARS
    }
    headline.update({name: _dig(results, path) for name, path in HEADLINE_VERDICTS})
    return headline


def build_manifest(output_dir: Any, since: Optional[float] = None) -> Dict[str, Any]:
    """List every file the run emitted, with its size.

    Not bookkeeping. It is what lets the documentation test assert that every emitted figure has
    a ``FIGURE_GUIDE.md`` entry without hardcoding a filename list -- a hardcoded list would pass
    by construction and would stop covering the moment an analysis gained a figure.

    Written before ``summary.json`` itself, so the summary is the one file the manifest cannot
    list; that is stated here rather than left as a puzzle for whoever diffs the two.

    Args:
        output_dir: The results directory.
        since: Run start time as a POSIX timestamp. Files older than it are a *previous* run's,
            which happens whenever ``--output-dir`` names a directory twice -- the default
            timestamped path cannot collide, but an explicit one can. Counting them would
            attribute another run's figures to this one and would make the documentation test
            cover files this run never produced. ``None`` lists everything.

    Returns:
        ``files`` -- relative POSIX paths to sizes in bytes -- plus counts, the figure subset,
        and how many stale files were excluded.
    """
    root = Path(output_dir)
    files: Dict[str, int] = {}
    n_stale = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == SUMMARY_FILENAME:
            continue
        try:
            stat = path.stat()
        except OSError:
            # Raced away between the walk and the stat. A vanished file is not worth losing the
            # summary over -- see Report.finalise.
            continue
        if since is not None and stat.st_mtime < float(since):
            n_stale += 1
            continue
        files[path.relative_to(root).as_posix()] = int(stat.st_size)
    figures = sorted(name for name in files if name.endswith(".pdf"))
    return {
        "files": files,
        "n_files": len(files),
        "figures": figures,
        "n_figures": len(figures),
        "n_excluded_stale": n_stale,
        "note": f"{SUMMARY_FILENAME} is excluded: the manifest is built before it is written.",
    }


def build_coverage(results: Dict[str, Any], analyses: List[str]) -> Dict[str, Any]:
    """Record how many samples each analysis actually saw, and warn when they disagree.

    Effective-$n$ per analysis is what reveals two analyses having run on different populations
    -- a forecast scored over the whole split beside an uplift scored over a capped draw of one
    shard reconcile with each other only by coincidence, and nothing else in the output shows it.

    Args:
        results: The accumulated results.
        analyses: The analyses that were selected.

    Returns:
        Per-analysis ``n_samples`` and ``composition``, plus any warnings raised.
    """
    per_analysis: Dict[str, Any] = {}
    for name in analyses:
        block = results.get(name)
        if not isinstance(block, dict):
            continue
        per_analysis[name] = {
            "n_samples": block.get("n_samples"),
            "composition": block.get("composition") or {},
            "capped": bool((block.get("plan") or {}).get("capped", False)),
        }

    warnings: List[str] = []
    populations = {
        name: record["n_samples"]
        for name, record in per_analysis.items()
        if record["n_samples"] is not None and not record["capped"]
    }
    distinct = set(populations.values())
    if len(distinct) > 1:
        warnings.append(
            f"uncapped analyses ran on different sample populations: {populations}. Metrics "
            f"from two of them do not describe the same set of recordings."
        )
    return {"per_analysis": per_analysis, "warnings": warnings}


def check_inert_caps(eval_config: Dict[str, Any]) -> List[str]:
    r"""Warn about caps that ``max_samples`` has already made unreachable.

    A cap below ``max_samples`` still bites; one above it never fires, and an operator who set
    ``caps.attention: 2000`` believing it bounded memory has in fact been bounded by
    ``max_samples`` all along -- so the cap they are tuning does nothing.

    Args:
        eval_config: The validated ``eval_config`` block.

    Returns:
        One warning per inert cap, empty when none is.
    """
    limit = eval_config.get("max_samples")
    if limit is None:
        return []
    return [
        f"eval_config.caps.{name} = {cap} is inert: max_samples = {limit} already bounds the "
        f"pass below it, so this cap never fires."
        for name, cap in (eval_config.get("caps") or {}).items()
        if cap is not None and int(cap) >= int(limit)
    ]


# ---------------------------------------------------------------------------
# Grouped emission
#
# One helper, used by every analysis that produces a per-sample frame, emitting the by-class and
# by-subgroup variants of that frame and its headline distributions.
#
# Here rather than in each analysis because the *policy* is shared and the policy is the hard
# part: when there are fewer than two groups there is nothing to compare and the grouped variant
# must be a recorded skip rather than a one-violin figure that looks like a result; a group column
# that is entirely unlabelled is the ordinary case on the pretraining split and must not raise;
# and the two grouping axes differ only in which column they read. Eleven copies of that would be
# eleven chances to get one of them subtly different.
#
# The pooled output is never touched. A grouped variant is written *beside* it, so a run over a
# single-class split produces exactly what it produced before this existed.
# ---------------------------------------------------------------------------
#: Quantiles reported per (group, metric), beside the mean and the count. The quartiles rather
#: than a standard deviation: these distributions are routinely skewed, and the summary should
#: describe the same shape the violin draws.
_GROUP_QUANTILES: tuple = (0.25, 0.5, 0.75)


def summarise_by_group(
    frame: Any, group_column: str, value_columns: List[str]
) -> Any:
    """Aggregate a per-sample frame into one row per (group, metric).

    Long form, not wide: a long table merges across two runs with no column renaming, and it does
    not change shape when an analysis gains a metric.

    Args:
        frame: The per-sample frame.
        group_column: The column to group by.
        value_columns: The metrics to summarise.

    Returns:
        A ``DataFrame`` with columns ``group``, ``metric``, ``n``, ``mean``, ``q25``, ``median``,
        ``q75``. Non-finite values are excluded from every statistic and from ``n``.
    """
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for group, block in frame.groupby(group_column, dropna=True):
        for metric in value_columns:
            if metric not in block:
                continue
            values = np.asarray(block[metric], dtype=np.float64)
            finite = values[np.isfinite(values)]
            record: Dict[str, Any] = {
                "group": str(group),
                "metric": metric,
                # Counted over the finite values, so a group of NaNs reports n = 0 rather than a
                # mean of NaN over a population that looks healthy.
                "n": int(finite.size),
                "mean": float(finite.mean()) if finite.size else float("nan"),
            }
            for quantile, name in zip(_GROUP_QUANTILES, ("q25", "median", "q75")):
                record[name] = (
                    float(np.quantile(finite, quantile)) if finite.size else float("nan")
                )
            rows.append(record)
    return pd.DataFrame(rows)


def emit_grouped_variants(
    frame: Any,
    directory: Any,
    *,
    value_columns: List[str],
    group_columns: Optional[List[str]] = None,
    stem: str = "per_sample",
    references: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Write the by-group CSV and figure for each grouping axis, or record why not.

    Args:
        frame: The analysis's per-sample frame, already carrying the group columns that
            :func:`~teb_vae.lag_attn.eval.collectors.collect_metrics` attaches.
        directory: The analysis's output directory.
        value_columns: The headline metrics to resolve by group. Columns absent from the frame
            are skipped rather than raising, so an analysis can name a metric it only sometimes
            produces.
        group_columns: The grouping axes. ``None`` uses
            :data:`~teb_vae.lag_attn.eval.labels.GROUP_COLUMNS`.
        stem: Filename stem, yielding ``<stem>_by_<group>.csv`` and ``.pdf``.
        references: Optional metric-to-reference-value mapping, drawn as a horizontal line.

    Returns:
        Grouping axis to a record: either the paths written and the per-group counts, or
        ``skipped`` with the reason. Never raises -- a grouped variant is an addition to a run,
        and an analysis whose pooled output succeeded must not be marked failed because its split
        turned out to hold one cohort.
    """
    from teb_vae.lag_attn.eval import figures, labels

    axes_to_emit = list(group_columns if group_columns is not None else labels.GROUP_COLUMNS)
    present = [column for column in value_columns if column in getattr(frame, "columns", [])]
    directory = Path(directory)
    emitted: Dict[str, Any] = {}

    for group_column in axes_to_emit:
        if group_column not in getattr(frame, "columns", []):
            emitted[group_column] = {
                "skipped": True,
                "reason": f"the per-sample frame carries no {group_column!r} column",
            }
            continue
        if not present:
            emitted[group_column] = {
                "skipped": True,
                "reason": f"none of the requested metrics {value_columns} is on the frame",
            }
            continue

        groups = labels.distinct_groups(list(frame[group_column]))
        if len(groups) < 2:
            # One group is the pooled output under another name, and a single violin invites a
            # comparison there is nothing to compare against. Expected on the pretraining split,
            # which is healthy-only by construction.
            emitted[group_column] = {
                "skipped": True,
                "reason": (
                    f"{len(groups)} distinct {group_column} value(s) in this split, so there is "
                    f"nothing to compare; the pooled output stands"
                ),
                "groups": groups,
            }
            continue

        directory.mkdir(parents=True, exist_ok=True)
        summary = summarise_by_group(frame, group_column, present)
        csv_path = directory / f"{stem}_by_{group_column}.csv"
        summary.to_csv(csv_path, index=False)

        values_by_metric = {
            metric: {
                group: np.asarray(
                    frame.loc[frame[group_column].astype(str) == group, metric], dtype=np.float64
                )
                for group in groups
            }
            for metric in present
        }
        figure, _ = figures.grouped_violin_figure(
            values_by_metric, groups,
            title_prefix=f"by {group_column}: ", references=references,
        )
        try:
            pdf_path = figures.render_to_pdf(figure, directory / f"{stem}_by_{group_column}.pdf")
        finally:
            figures.plt.close(figure)

        emitted[group_column] = {
            "skipped": False,
            "groups": groups,
            "n_per_group": {
                group: int((frame[group_column].astype(str) == group).sum()) for group in groups
            },
            "files": {"table": str(csv_path), "figure": str(pdf_path)},
        }
    return emitted


# ---------------------------------------------------------------------------
# Sanity checks
#
# Documented expectations, turned into asserted verdicts. Each returns a
# ``(verdict, detail, numbers)`` triple; ``inconclusive`` is a first-class outcome and means the
# run did not carry what the check needs -- which is different from the check having passed, and
# the distinction is why a bool would not do.
# ---------------------------------------------------------------------------
#: Verdict when a check cannot be evaluated from what the run produced.
INCONCLUSIVE = "inconclusive"


def _verdict(passed: bool, detail: str, **numbers: Any) -> Dict[str, Any]:
    """Build one check's record."""
    return {"verdict": "pass" if passed else "fail", "detail": detail, **numbers}


def _inconclusive(detail: str, **numbers: Any) -> Dict[str, Any]:
    """Build the record for a check the run could not evaluate."""
    return {"verdict": INCONCLUSIVE, "detail": detail, **numbers}


def check_per_file_counts(probe: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Every configured shard must have contributed samples."""
    per_file = (probe or {}).get("per_file")
    if not per_file:
        return _inconclusive("the loader probe recorded no per-file counts")
    empty = sorted(name for name, count in per_file.items() if not count)
    return _verdict(
        not empty,
        "every file contributed samples" if not empty
        else f"{len(empty)} file(s) contributed none: {empty}",
        n_files=len(per_file), empty_files=empty,
    )


def check_classes_present(probe: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Every clinical class the split should hold must appear.

    Inconclusive rather than failing when the labels are absent or uniformly unset: the
    pretraining split is healthy-only by construction and carries an all-zero ``target``, so a
    single class there is correct, not a coverage bug.

    The probe keys this counter by clinical class *name*, via ``labels.clinical_class_code``,
    which divides the per-step weight back out of the weight-scaled ``target``. It previously
    keyed on the raw stored value, so a recording whose first valid step was only partially
    valid contributed a key like ``'0.75'`` -- which this filter then counted as a second class.
    One such recording permanently defeated the check, and a genuinely single-class split, the
    coverage bug this exists to catch, reported "2 class(es) present".
    """
    counts = (probe or {}).get("per_target_class") or {}
    labelled = {key: value for key, value in counts.items() if key not in ("None", "0.0", "0")}
    if not labelled:
        return _inconclusive(
            "no labelled class in the split -- expected on the healthy-only pretraining split, "
            "where target is uniformly zero",
            observed=counts,
        )
    return _verdict(
        len(labelled) > 1,
        f"{len(labelled)} class(es) present" if len(labelled) > 1
        else f"only one class present: {sorted(labelled)}",
        observed=counts, n_classes=len(labelled),
    )


def check_argmax_lag(results: Dict[str, Any]) -> Dict[str, Any]:
    r"""The attention must select a lag rather than collapsing or spreading uniformly.

    Two degenerate readings, and they fail in opposite directions. An ``argmax_lag`` pinned at
    $0$ means the attention never looks back and the lag machinery is inert. An entropy at its
    ceiling means the weights are uniform, so the argmax is whichever lag won a rounding contest
    and the lag attribution is noise wearing a peak.

    **The ceiling is the attainable one, not $\log L$.** Causal masking gives anchor $t$ only
    $\min(t + 1, L)$ valid lags, so at the production geometry ($L = 91$, warmup $30$) sixty of
    the two hundred and forty supported anchors cannot reach $\log 91$ at all. Attention that is
    exactly uniform over every causally available lag -- the degenerate case this check exists to
    catch -- scores $4.398$ against $\log 91 = 4.511$, a ratio of $0.9749$. Divided by $\log L$
    the $0.99$ threshold could therefore **never fire**, and its 1% margin, justified below as
    floating-point slack, is twenty-four times smaller than that systematic gap.
    """
    attention = results.get("attention")
    if not isinstance(attention, dict):
        return _inconclusive("the attention analysis did not report")

    lag = attention.get("median_argmax_lag")
    entropy = attention.get("mean_entropy_nats")
    ceiling = attention.get("mean_attainable_entropy_nats")
    window_ceiling = attention.get("max_possible_entropy_nats")
    if lag is None or entropy is None or not ceiling:
        return _inconclusive("the attention analysis reported no lag or entropy", lag=lag)

    lag, entropy, ceiling = float(lag), float(entropy), float(ceiling)
    # 0.99 of the *attainable* ceiling rather than equality: the maximum is only attained by
    # exactly uniform weights, which floating point never produces, so an equality test could
    # never fire. Against log L instead of the attainable bound the reverse was true and the
    # branch could never fire either -- see the docstring.
    degenerate = lag <= 0.0
    uniform = entropy >= 0.99 * ceiling
    detail = (
        "argmax lag is neither pinned at 0 nor entropy-maximal"
        if not (degenerate or uniform)
        else "; ".join(
            part for part in (
                "argmax lag is pinned at 0, so the lag window is inert" if degenerate else "",
                "attention entropy is at its uniform ceiling, so the peak is not a selection"
                if uniform else "",
            ) if part
        )
    )
    return _verdict(
        not (degenerate or uniform), detail,
        median_argmax_lag=lag, mean_entropy_nats=entropy,
        # Both ceilings are recorded: the attainable one is what the verdict divides by, and the
        # window width is what a reader who checks the arithmetic against $L$ will expect to see.
        attainable_entropy_nats=ceiling,
        window_entropy_nats=float(window_ceiling) if window_ceiling else None,
        entropy_ratio=entropy / ceiling if ceiling else None,
    )


def check_headline_finite(headline: Dict[str, Any]) -> Dict[str, Any]:
    """No headline scalar may be non-finite.

    The pipeline's quietest failure mode: the metrics return ``NaN`` by design for a
    fully-masked sample, so a ``summary.json`` of nothing but nulls is a perfectly ordinary
    thing for a broken run to produce, and it exits $0$.

    ``None`` is not a failure here -- it means the producing analysis was skipped or did not
    report, which the step record already covers. A *number* that is not finite is.
    """
    offending = sorted(
        name for name, value in headline.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and not math.isfinite(float(value))
    )
    reported = sorted(name for name, value in headline.items() if value is not None)
    return _verdict(
        not offending,
        f"all {len(reported)} reported headline scalar(s) are finite" if not offending
        else f"non-finite headline scalar(s): {offending}",
        non_finite=offending, n_reported=len(reported),
    )


def check_target_not_truncated(probe: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    r"""The raw ``target`` values must show no integer-truncation artifact.

    ``target`` is the class label scaled by the per-step ``weight``, so where ``weight`` is
    fractional the stored target must be fractional too. If **no** value anywhere is fractional
    while ``weight`` is, the field was written through an integer dtype and the partially-valid
    steps have been rounded -- silently reassigning them to a neighbouring class or to none.

    Read from ``probe['target_values']``, which counts every step, **not** from
    ``per_target_class``, which records one value per recording -- its first nonzero step. That
    step sits in a full-weight region on almost every recording, so the class histogram reports
    "all integers" on perfectly healthy fractional-weight data and this check would fail every
    such run.

    Where ``weight`` is strictly binary an integer target is expected and truncation is not
    observable at all, so the check says so rather than passing vacuously.
    """
    values = (probe or {}).get("target_values") or {}
    weight = (probe or {}).get("weight") or {}
    if not values or "any_fractional" not in values:
        return _inconclusive("the loader probe recorded no raw target values")
    if "binary" not in weight:
        return _inconclusive("the loader probe recorded no weight distribution")
    if not int(values.get("n_nonzero", 0)):
        return _inconclusive(
            "every target value is zero -- expected on the healthy-only pretraining split",
            **values,
        )
    # A non-finite target is its own defect and would make the fractional test meaningless:
    # NaN != round(NaN), so a field full of NaN would count as "fractional" and pass.
    if bool(values.get("any_non_finite")):
        return _verdict(
            False,
            f"{values['n_non_finite']} target value(s) are non-finite, so the class label "
            f"cannot be recovered for those steps",
            **values,
        )
    if bool(weight["binary"]):
        return _inconclusive(
            "weight is strictly binary, so an integer target is expected and truncation is not "
            "observable",
            weight_binary=True, **values,
        )
    fractional = bool(values["any_fractional"])
    return _verdict(
        fractional,
        f"{values['n_fractional']} of {values['n_nonzero']} nonzero target value(s) are "
        f"fractional, as a fractional weight requires" if fractional
        else "no target value anywhere is fractional while weight is -- the field was written "
             "through an integer dtype and the partially-valid steps were rounded",
        weight_binary=False, **values,
    )


def build_sanity(results: Dict[str, Any], headline: Dict[str, Any]) -> Dict[str, Any]:
    """Run every sanity check and summarise the outcome.

    Args:
        results: The accumulated results, for the probe record and the attention block.
        headline: The flattened headline scalars.

    Returns:
        Per-check verdicts plus ``n_failed`` and an overall ``warning`` flag.
    """
    probe = results.get("probe") if isinstance(results.get("probe"), dict) else None
    checks = {
        "per_file_counts": check_per_file_counts(probe),
        "classes_present": check_classes_present(probe),
        "argmax_lag": check_argmax_lag(results),
        "headline_finite": check_headline_finite(headline),
        "target_not_truncated": check_target_not_truncated(probe),
    }
    failed = sorted(name for name, record in checks.items() if record["verdict"] == "fail")
    return {
        "checks": checks,
        "failed": failed,
        "n_failed": len(failed),
        "n_inconclusive": sum(
            1 for record in checks.values() if record["verdict"] == INCONCLUSIVE
        ),
        # Distinct from the process exit code, which reflects whether a *step* raised. A run can
        # complete every step cleanly and still be one nobody should draw a conclusion from.
        "warning": bool(failed),
    }


def format_console_table(results: Dict[str, Any], steps: List["StepRecord"]) -> str:
    """Render the end-of-run console summary.

    Args:
        results: The accumulated results.
        steps: The per-step records.

    Returns:
        The table as a multi-line string.
    """
    lines = ["", "=" * 72, "eval summary", "=" * 72]

    headline = results.get("headline") or {}
    for name, _ in HEADLINE_SCALARS:
        value = headline.get(name)
        # A path that resolved to a dict rather than a leaf is a bug in HEADLINE_SCALARS, but it
        # must show up as an odd-looking table row rather than as a TypeError that costs the run.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rendered = f"{float(value):.6g}"
        else:
            rendered = "-" if value is None else str(value)
        lines.append(f"  {name:<24s} {rendered}")
    for name, _ in HEADLINE_VERDICTS:
        lines.append(f"  {name:<24s} {headline.get(name) or '-'}")

    sanity = results.get("sanity") or {}
    lines.append("-" * 72)
    for name, record in (sanity.get("checks") or {}).items():
        lines.append(f"  [{record['verdict']:>12s}] {name}: {record['detail']}")
    if sanity.get("warning"):
        lines.append(f"  !! {sanity['n_failed']} sanity check(s) FAILED -- read them before "
                     f"quoting any number above")

    for warning in (results.get("coverage") or {}).get("warnings", []):
        lines.append(f"  !! {warning}")
    for warning in results.get("config_warnings") or []:
        lines.append(f"  !! {warning}")

    lines.append("-" * 72)
    for record in steps:
        lines.append(
            f"  {'ok ' if record.ok else 'FAIL'} {record.name:<20s} {record.elapsed_s:8.1f}s"
        )
    peak = results.get("max_memory_allocated_gb")
    if peak is not None:
        lines.append(f"  peak CUDA memory {float(peak):.2f} GB")
    lines.append("=" * 72)
    return "\n".join(lines)


@dataclass
class StepRecord:
    """Outcome of one analysis step."""

    name: str
    ok: bool
    elapsed_s: float
    error: Optional[str] = None
    traceback: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return the record as a plain dict for the summary."""
        record: Dict[str, Any] = {
            "name": self.name,
            "ok": self.ok,
            "elapsed_s": round(float(self.elapsed_s), 3),
        }
        if not self.ok:
            record["error"] = self.error
            record["traceback"] = self.traceback
        return record


@dataclass
class Report:
    """The run's accumulated results, its per-step outcomes, and the summary writer."""

    results: Dict[str, Any] = field(default_factory=dict)
    steps: List[StepRecord] = field(default_factory=list)

    # ------------------------------------------------------------------
    def set(self, key: str, value: Any) -> None:
        """Record a headline result under ``key``.

        Args:
            key: Result name, typically the analysis that produced it.
            value: Any JSON-safe-able value.
        """
        self.results[key] = value

    def step(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        r"""Run one analysis, capturing any failure and continuing.

        Args:
            name: Step name, used in the log line and in ``summary.json``.
            fn: The analysis callable.
            *args: Positional arguments for ``fn``.
            **kwargs: Keyword arguments for ``fn``.

        Returns:
            Whatever ``fn`` returned, or ``None`` if it raised.
        """
        logger.info(f"[{name}] starting")
        started = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the point of the wrapper; see the module docstring
            elapsed = time.perf_counter() - started
            self.steps.append(
                StepRecord(
                    name=name,
                    ok=False,
                    elapsed_s=elapsed,
                    error=f"{type(exc).__name__}: {exc}",
                    traceback="".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                )
            )
            logger.error(f"[{name}] FAILED after {elapsed:.1f}s: {type(exc).__name__}: {exc}")
            return None
        elapsed = time.perf_counter() - started
        self.steps.append(StepRecord(name=name, ok=True, elapsed_s=elapsed))
        logger.info(f"[{name}] done in {elapsed:.1f}s")
        return result

    # ------------------------------------------------------------------
    @property
    def failed_steps(self) -> List[StepRecord]:
        """The steps that raised, in the order they ran."""
        return [record for record in self.steps if not record.ok]

    def exit_code(self) -> int:
        """Return ``1`` when any step failed, else ``0``.

        An unattended run must not be mistaken for a clean one on the strength of having
        produced output files, which a partially failed run also does.
        """
        return 1 if self.failed_steps else 0

    def finalise(
        self,
        output_dir: Any,
        analyses: Optional[List[str]] = None,
        started_at: Optional[float] = None,
    ) -> None:
        """Assemble the derived blocks, immediately before the summary is written.

        Everything here is computed *from* what the analyses already reported rather than
        alongside them, so an analysis that failed costs its own block and nothing else. It runs
        as a method rather than inside :meth:`write` so a caller can inspect the result -- and so
        the console table can be printed from the same assembled state that reaches disk.

        **Each block is built under its own guard.** This runs after every analysis has
        completed, so anything raising here would abort before :meth:`write` and lose the entire
        run -- every result *and* every captured traceback -- to a failure in the bookkeeping.
        That is precisely the outcome :meth:`step` exists to prevent, and it would be perverse
        for the summariser to be the one place that does not honour it.

        Args:
            output_dir: The results directory, scanned for the artifact manifest.
            analyses: The analyses that were selected, for the coverage record. ``None`` uses
                whatever ``analyses_selected`` recorded.
            started_at: Run start as a POSIX timestamp, so the manifest can exclude a previous
                run's files when the output directory is reused.
        """
        selected = analyses if analyses is not None else self.results.get("analyses_selected")
        selected = list(selected or [])

        def _safe(name: str, builder: Callable[[], Any], fallback: Any) -> Any:
            try:
                return builder()
            except Exception as exc:  # noqa: BLE001 - see the docstring
                logger.error(
                    f"could not assemble the {name!r} summary block: {type(exc).__name__}: "
                    f"{exc}. The run's results and step records are unaffected."
                )
                return {**fallback, "error": f"{type(exc).__name__}: {exc}"}

        self.set("headline", _safe("headline", lambda: build_headline(self.results), {}))
        self.set(
            "coverage",
            _safe("coverage", lambda: build_coverage(self.results, selected),
                  {"per_analysis": {}, "warnings": []}),
        )
        self.set(
            "sanity",
            _safe("sanity", lambda: build_sanity(self.results, self.results["headline"]),
                  {"checks": {}, "failed": [], "n_failed": 0, "warning": True}),
        )
        self.set(
            "config_warnings",
            _safe("config_warnings",
                  lambda: check_inert_caps(self.results.get("eval_config") or {}), {}) or [],
        )
        # Last, so it sees every file the analyses wrote.
        self.set(
            "artifacts",
            _safe("artifacts", lambda: build_manifest(output_dir, since=started_at),
                  {"files": {}, "figures": [], "n_files": 0, "n_figures": 0}),
        )

        warnings = self.results["coverage"].get("warnings") or []
        if isinstance(self.results["config_warnings"], list):
            warnings = list(warnings) + self.results["config_warnings"]
        for warning in warnings:
            logger.warning(warning)
        for name in self.results["sanity"].get("failed") or []:
            logger.error(
                f"sanity check FAILED [{name}]: "
                f"{self.results['sanity']['checks'][name]['detail']}"
            )

    def console_table(self) -> str:
        """Render the end-of-run summary table from the assembled state.

        Guarded for the same reason :meth:`finalise` is: this is called between the last analysis
        and the summary write, and a formatting error here must not be what loses the run.
        """
        try:
            return format_console_table(self.results, self.steps)
        except Exception as exc:  # noqa: BLE001
            return f"(could not render the summary table: {type(exc).__name__}: {exc})"

    def write(self, output_dir: Any) -> Path:
        """Assemble and write ``summary.json``, re-logging every captured failure.

        The re-log is at the end deliberately: in a run whose log is tens of thousands of lines
        the original ``ERROR`` is long gone, and the operator's attention is on the tail.

        Args:
            output_dir: The run directory.

        Returns:
            The path written.
        """
        failed = self.failed_steps
        if failed:
            logger.error(f"{len(failed)} of {len(self.steps)} step(s) failed:")
            for record in failed:
                logger.error(f"  [{record.name}] {record.error}")
                logger.error(f"{record.traceback}")

        summary = {
            "results": self.results,
            "steps": [record.as_dict() for record in self.steps],
            "n_steps": len(self.steps),
            "n_failed": len(failed),
            "failed": [record.name for record in failed],
            "exit_code": self.exit_code(),
        }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / SUMMARY_FILENAME
        with open(path, "w", encoding="utf-8") as handle:
            # allow_nan=False so an unsanitised non-finite value raises here rather than
            # producing a file that only Python can read back.
            json.dump(json_safe(summary), handle, indent=2, allow_nan=False)
        logger.info(f"wrote {path}")
        return path
