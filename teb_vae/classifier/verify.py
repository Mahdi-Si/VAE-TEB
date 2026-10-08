r"""Offline acceptance gate over a classifier run (SPEC §11.15, criteria 1-12).

    python -m teb_vae.classifier.verify RUN_DIR [--json-out PATH]
    python -m teb_vae.classifier.verify --runs RUN_A RUN_B ...

Torch-free: reads ``<run>/evaluation/summary.json``, plus ``predictions/provenance.json``'s ``written_at``
and the ``## `` headings of ``<run>/summary.md`` (``on_disk``, criteria 1b and 12). The config that
:func:`FIGURE_REGISTRY` needs travels in ``results.config``. Each criterion returns PASS, FAIL or INCONCLUSIVE.
INCONCLUSIVE is never a pass, but it does not block. The exit code is 1 on any FAIL. The VAE
gate's verdict constants, helpers and its exit-code criterion are reused, since the summary envelope is
the same (``teb_vae/lag_attn_cfs/eval/verify.py``). The validity checks evaluate computes (cohort, NP overshoot,
controls, M7, missingness confound) arrive as ``results.sanity.checks`` and are re-read here, not re-derived.
``--runs`` prints a markdown table of headline metrics per run instead; its arm columns are the config leaves
whose values differ across the runs.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from teb_vae.classifier.metrics import BASELINES, P2_TABLES, primary_model
from teb_vae.classifier.report import FIGURE_REGISTRY, SUMMARY_MD, _cell, _ci
from teb_vae.lag_attn_cfs.eval.verify import (
    _ABSENT, FAIL, INCONCLUSIVE, PASS, _dig, _from_sanity_check, _render, _result, _table, check_exit_code,
)

#: The sanity checks criteria 3 and 9 re-read (written by the ``C`` and ``M`` analyses).
COHORT_CHECKS = ("cohort_disjoint", "cohort_exposure")
M7_CHECKS = ("m7_committed_overall_monotone", "m7_end_equals_running_max", "m7_last_equals_final")
#: The §11.11 section numbers ``summary.md`` must carry as ``## <n>. `` headings, in this order (criterion 12).
SUMMARY_SECTIONS = list(range(1, 13))


def _checks(summary: Dict[str, Any], names: Sequence[str], block: str) -> Dict[str, Any]:
    """Several sanity checks as one criterion: FAIL if any failed, PASS if all passed, else INCONCLUSIVE. A check
    that covered nothing (``n_checked`` 0) is inconclusive whatever it recorded: a vacuous pass is no pass."""
    checks = _dig(summary, "results", "sanity", "checks") or {}
    recs = {name: checks.get(name) for name in names}
    missing = [name for name, rec in recs.items() if not rec]
    if missing:
        return _result(INCONCLUSIVE, f"the run recorded no {missing} check(s); was block {block} skipped?")
    verdicts = {name: "inconclusive" if rec.get("n_checked") == 0 and rec["verdict"] == "pass" else rec["verdict"]
                for name, rec in recs.items()}
    detail = "; ".join(f"{name}: {rec['detail']}" for name, rec in recs.items())
    if "fail" in verdicts.values():
        return _result(FAIL, detail, verdicts=verdicts)
    return _result(PASS if set(verdicts.values()) == {"pass"} else INCONCLUSIVE, detail, verdicts=verdicts)


def _num(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None


def check_cohort(summary: Dict[str, Any]) -> Dict[str, Any]:
    """3: within-fold disjointness passed; no test GUID in the VAE's pretraining exposure (L2, L3)."""
    return _checks(summary, COHORT_CHECKS, "C")


def check_exit_codes(summary: Dict[str, Any]) -> Dict[str, Any]:
    """1: no evaluate step raised, and no report step raised (``results.report``, which report writes;
    INCONCLUSIVE before report has run)."""
    rec = check_exit_code(summary)
    rep = _dig(summary, "results", "report")
    if rec["verdict"] != PASS:
        return rec
    if rep is None:
        return _result(INCONCLUSIVE, "every evaluate step completed; report has not recorded a run since")
    failed = rep.get("failed") or []
    if rep.get("exit_code") or failed:
        return _result(FAIL, f"report: {len(failed)} step(s) failed: {failed[:5]}", report_failed=failed)
    return _result(PASS, "every evaluate and report step completed")


def check_evaluation_current(summary: Dict[str, Any]) -> Dict[str, Any]:
    """1b: the evaluation read the predictions now on disk: ``results.predictions_written_at`` equals
    ``predictions/provenance.json``'s ``written_at`` (``on_disk``, which :func:`load_summary` reads). A predict
    after the last full evaluate fails, since report and verify would read the older tables."""
    used, now = _dig(summary, "results", "predictions_written_at"), _dig(summary, "on_disk", "predictions_written_at")
    if used is None or now is None:
        return _result(INCONCLUSIVE, f"predictions written_at unknown (evaluation: {used}, on disk: {now})")
    if used != now:
        return _result(FAIL, f"the evaluation read predictions written at {used}, but predictions/ was rewritten at "
                             f"{now}; re-run --stage evaluate")
    return _result(PASS, f"the evaluation read the current predictions ({now})")


def check_units(summary: Dict[str, Any]) -> Dict[str, Any]:
    """2: no planned unit is missing (``results.missing_units``); INCONCLUSIVE under ``--allow-partial``, or when
    the evaluation recorded no unit list at all (never a vacuous PASS)."""
    missing = _dig(summary, "results", "missing_units")
    if missing is None:
        return _result(INCONCLUSIVE, "the evaluation recorded no missing_units (predictions/provenance.json lacks it)")
    if not missing:
        return _result(PASS, "every planned unit was trained and predicted")
    detail = f"{len(missing)} unit(s) missing: {missing[:5]}"
    if _dig(summary, "results", "allow_partial"):
        return _result(INCONCLUSIVE, f"{detail} (pooled with --allow-partial)", missing_units=missing)
    return _result(FAIL, detail, missing_units=missing)


def check_selection_lock(summary: Dict[str, Any]) -> Dict[str, Any]:
    """4: every unit's ``selection_lock.json`` was written before its test predictions (L5): ``lock_written_at <
    test_written_at`` for each of ``results.prediction_units`` (the predictions' provenance, which evaluate copies)."""
    units = _dig(summary, "results", "prediction_units")
    if not units:
        return _result(INCONCLUSIVE, "the evaluation recorded no prediction units and their lock timestamps")
    name = lambda u: f"{u.get('model_id')}|{u.get('seed')}|{u.get('fold')}"  # noqa: E731
    stamped = [u for u in units if u.get("lock_written_at") and u.get("test_written_at")]
    late = [name(u) for u in stamped
            if datetime.fromisoformat(u["lock_written_at"]) >= datetime.fromisoformat(u["test_written_at"])]
    unknown = [name(u) for u in units if u not in stamped]
    if late:
        return _result(FAIL, f"{len(late)} unit(s) locked at or after their test rows were written: {late[:5]}",
                       late=late)
    if unknown:
        return _result(INCONCLUSIVE, f"{len(unknown)} unit(s) without a lock or test timestamp: {unknown[:5]}")
    return _result(PASS, f"all {len(units)} units locked before their test rows were written")


def check_headline_finite(summary: Dict[str, Any]) -> Dict[str, Any]:
    """5: every headline entry has a finite pooled test value and a finite fold mean (FAIL), over
    finite folds only (a NaN fold, ``n_folds_nan``, is INCONCLUSIVE, named)."""
    head = _dig(summary, "results", "headline")
    if not head:
        return _result(FAIL, "the summary carries no headline metrics")
    bad = [key for key, rec in head.items()
           if not all(isinstance(rec.get(f), (int, float)) and math.isfinite(rec[f]) for f in ("test", "fold_mean"))]
    nan = [f"{key} ({rec['n_folds_nan']} of {rec.get('n_folds')} folds)" for key, rec in head.items()
           if rec.get("n_folds_nan")]
    if bad:
        return _result(FAIL, f"{len(bad)} of {len(head)} headline entries not finite: {bad[:5]}", n_bad=len(bad),
                       nan_folds=nan)
    if nan:
        return _result(INCONCLUSIVE, f"{len(nan)} headline entries average over NaN folds: {nan[:5]}", n_bad=0,
                       nan_folds=nan)
    return _result(PASS, f"all {len(head)} headline entries finite", n_bad=0)


def check_np_overshoot(summary: Dict[str, Any]) -> Dict[str, Any]:
    """6: every NP-policy pooled test FPR <= alpha + the 95% binomial tolerance on n_neg (``np_overshoot``, from
    T2); FAIL names the policy, and a fold that fell back to empirical voids the guarantee (INCONCLUSIVE)."""
    return _from_sanity_check(summary, "np_overshoot")


def check_shuffled_control(summary: Dict[str, Any]) -> Dict[str, Any]:
    """7: the shuffled-label control's pooled test AUROC <= 0.60 and its CI holds 0.5 (``shuffled_auroc``, B1)."""
    return _from_sanity_check(summary, "shuffled_auroc")


def check_model_vs_shortcut(summary: Dict[str, Any]) -> Dict[str, Any]:
    """8: the primary model's pooled test GUID AUROC exceeds the shortcut baseline's (FAIL otherwise; point
    estimates), INCONCLUSIVE where their bootstrap CIs overlap. The primary model is evaluate's
    (:func:`~teb_vae.classifier.metrics.primary_model`: ``model``'s seed ensemble, else its first seed, else the probe);
    every other non-baseline row (extra seeds, ``frozen``, ``noind``, ``*_covoff``, ``*_stage``) is a diagnostic, listed but never
    gated. Pooled, not the fold mean, since the CIs are the pooled ones."""
    auc = [r for r in (_dig(summary, "results", "headline") or {}).values()
           if (r.get("level"), r.get("policy_id"), r.get("metric")) == ("guid", "threshold_free", "auroc")]
    short = next((r for r in auc if r["model_id"] == "shortcut"), None)
    pm = primary_model(sorted({(r["model_id"], r["seed"]) for r in auc}))
    models = [r for r in auc if (r["model_id"], r["seed"]) == pm and pm[0] != "shortcut"]
    if short is None or not models:
        return _result(INCONCLUSIVE, "the headline carries no shortcut or no model GUID AUROC")
    info = [f"{r['model_id']}|{r['seed']} {_ci(r.get('test'), r.get('test_ci'))}" for r in auc
            if r["model_id"] not in BASELINES and r not in models]
    s, s_ci = _num(short.get("test")), [_num(x) for x in short.get("test_ci") or (None, None)]
    lose, close = [], []
    for r in models:
        v, ci = _num(r.get("test")), [_num(x) for x in r.get("test_ci") or (None, None)]
        if v is None or s is None:
            close.append(f"{r['model_id']}|{r['seed']} (AUROC undefined)")
        elif v <= s:
            lose.append(f"{r['model_id']}|{r['seed']}")
        elif None in (*ci, *s_ci) or ci[0] <= s_ci[1]:
            close.append(f"{r['model_id']}|{r['seed']} (CIs overlap or undefined)")
    detail = "pooled test GUID AUROC: " + "; ".join(f"{r['model_id']}|{r['seed']} {_ci(r.get('test'), r.get('test_ci'))}"
                                                    for r in [short, *models])
    detail += f"; diagnostic rows (not gated): {info}" if info else ""
    return _result(FAIL if lose else INCONCLUSIVE if close else PASS,
                   detail + (f"; not above the shortcut: {lose}" if lose else f"; inconclusive: {close}" if close else ""))


def check_m7(summary: Dict[str, Any]) -> Dict[str, Any]:
    """9: committed_overall is monotone, and the M7 end-point consistency checks pass (block M)."""
    return _checks(summary, M7_CHECKS, "M")


def check_missing_confound(summary: Dict[str, Any]) -> Dict[str, Any]:
    """10: no train-fold missing rate differs across classes above ``context.missing_confound_max``, or the
    ``no_indicator`` ablation (``model_id`` ``noind``) is present (``missing_confound``, from block C)."""
    return _from_sanity_check(summary, "missing_confound")


def check_expected_outputs(summary: Dict[str, Any]) -> Dict[str, Any]:
    """11: ``artifacts.figures`` (files written since evaluate started) covers ``FIGURE_REGISTRY(config)``;
    every P2 table is non-empty."""
    cfg = _dig(summary, "results", "config")
    if not cfg:
        return _result(INCONCLUSIVE, "the summary carries no config, so the figure registry is unknown")
    figures = {f.removeprefix("figures/").rsplit(".", 1)[0] for f in _dig(summary, "results", "artifacts", "figures") or []}
    missing = [stem for stem in FIGURE_REGISTRY(cfg) if stem not in figures]
    tables = _dig(summary, "results", "tables") or {}
    empty = [name for name in P2_TABLES if not tables.get(name)]
    ok = not missing and not empty
    return _result(PASS if ok else FAIL, "every registry figure and table present" if ok
                   else f"missing figures {missing[:5]} ({len(missing)}); empty/missing tables {empty}",
                   missing_figures=missing, empty_tables=empty)


def check_summary_sections(summary: Dict[str, Any]) -> Dict[str, Any]:
    """12: ``summary.md`` carries the §11.11 sections 1-12 as ``## <n>. `` headings, in order
    (``on_disk.summary_md_headings``, which :func:`load_summary` reads)."""
    heads = _dig(summary, "on_disk", "summary_md_headings")
    if heads is None:
        return _result(FAIL, f"no {SUMMARY_MD}; run --stage report")
    nums = [int(m.group(1)) for h in heads if (m := re.match(r"## (\d+)\. ", h))]
    missing = [n for n in SUMMARY_SECTIONS if n not in nums]
    if missing or nums != SUMMARY_SECTIONS:
        return _result(FAIL, f"{SUMMARY_MD} sections missing {missing}" if missing
                       else f"{SUMMARY_MD} sections out of order or repeated: {nums}", missing_sections=missing)
    return _result(PASS, f"{SUMMARY_MD} carries sections 1-12 in order")


#: The §11.15 criteria, in order.
CRITERIA: Tuple[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]], ...] = (
    ("1_exit_code", check_exit_codes),
    ("1b_evaluation_current", check_evaluation_current),
    ("2_units", check_units),
    ("3_cohort", check_cohort),
    ("4_selection_lock", check_selection_lock),
    ("5_headline_finite", check_headline_finite),
    ("6_np_overshoot", check_np_overshoot),
    ("7_shuffled_control", check_shuffled_control),
    ("8_model_vs_shortcut", check_model_vs_shortcut),
    ("9_m7_consistency", check_m7),
    ("10_missing_confound", check_missing_confound),
    ("11_expected_outputs", check_expected_outputs),
    ("12_summary_sections", check_summary_sections),
)


def verify(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Every criterion's record plus ``failed``, ``inconclusive`` and ``passed`` (no FAIL)."""
    results = {name: check(summary) for name, check in CRITERIA}
    failed = [n for n, r in results.items() if r["verdict"] == FAIL]
    return {"criteria": results, "failed": failed, "passed": not failed,
            "inconclusive": [n for n, r in results.items() if r["verdict"] == INCONCLUSIVE]}


def load_summary(run_dir: Any) -> Optional[Dict[str, Any]]:
    """``<run>/evaluation/summary.json`` with ``on_disk``: ``predictions/provenance.json``'s ``written_at`` (1b)
    and the ``## `` headings of ``summary.md`` (12, None without the file). None without a summary."""
    run = Path(run_dir)
    path, prov, md = run / "evaluation" / "summary.json", run / "predictions" / "provenance.json", run / SUMMARY_MD
    if not path.is_file():
        return None
    on_disk = {"predictions_written_at": json.loads(prov.read_text()).get("written_at") if prov.is_file() else None,
               "summary_md_headings": [x for x in md.read_text(encoding="utf-8").splitlines() if x.startswith("## ")]
               if md.is_file() else None}
    return json.loads(path.read_text()) | {"on_disk": on_disk}


def _leaves(cfg: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """A nested config's leaves as ``{dotted.key: value}`` (a list is one leaf)."""
    out: Dict[str, Any] = {}
    for k, v in cfg.items():
        out |= _leaves(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v}
    return out


def config_arms(configs: Sequence[Optional[Dict[str, Any]]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """``(arms, leaves)``: the dotted config leaves (:func:`_leaves`) whose values differ across ``configs``, sorted (a
    leaf one config lacks counts as absent), and each config's leaves. The arm columns of ``--runs`` and of ``compare``'s
    ``comparison.md`` (Q3)."""
    leaves = [_leaves(c or {}) for c in configs]
    return sorted(k for k in set().union(*leaves)
                  if len({json.dumps(f.get(k, "(absent)"), sort_keys=True, default=str) for f in leaves}) > 1), leaves


def runs_table(run_dirs: Sequence[Any]) -> str:
    """Markdown table of headline metrics, one row per run x model x seed: the GUID AUROC (fold mean ± SD, the
    primary; pooled test with CI), the primary policy's pooled test GUID sensitivity and FPR, and the run's gate
    verdict. The arm columns are the ``results.config`` leaves whose values differ across the runs."""
    summaries = {str(r): load_summary(r) for r in run_dirs}
    done = [r for r, s in summaries.items() if s]
    arms, lv = config_arms([_dig(summaries[r], "results", "config") for r in done])
    leaves = dict(zip(done, lv))
    header = ["run", *arms, "model", "seed", "AUROC fold mean ± SD", "AUROC pooled test [CI]",
              "sens (primary policy) pooled test [CI]", "FPR (primary policy) pooled test [CI]", "gate"]
    rows = []
    for r, s in summaries.items():
        if s is None:
            rows.append([r, *("-" for _ in arms), "-", "-", "-", "-", "-", "-", "no evaluation/summary.json"])
            continue
        head, primary, gate = _dig(s, "results", "headline") or {}, _dig(s, "results", "config", "eval",
                                                                          "primary_policy"), verify(s)
        verdict = "FAIL" if gate["failed"] else "PASS" + (f" ({len(gate['inconclusive'])} inconclusive)"
                                                          if gate["inconclusive"] else "")
        for m, sd in sorted({(h["model_id"], h["seed"]) for h in head.values()}) or [("-", "-")]:
            rec = {met: head.get(f"{m}|{sd}|guid|{pid}|{met}") or {}
                   for pid, met in (("threshold_free", "auroc"), (primary, "sens"), (primary, "fpr"))}
            rows.append([r, *(_render(leaves[r].get(k, _ABSENT)) for k in arms), m, sd,
                         f"{_cell(rec['auroc'].get('fold_mean'))} ± {_cell(rec['auroc'].get('fold_sd'))}",
                         *(_ci(rec[met].get("test"), rec[met].get("test_ci")) for met in ("auroc", "sens", "fpr")),
                         verdict])
    return "\n".join(_table(header, rows))


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry; returns 1 on any FAIL (or a missing summary), else 0. ``--runs`` prints :func:`runs_table`."""
    parser = argparse.ArgumentParser(description="Classifier verify gate (SPEC §11.15).")
    parser.add_argument("run_dir", nargs="?")
    parser.add_argument("--json-out")
    parser.add_argument("--runs", nargs="+", metavar="RUN_DIR",
                        help="print a markdown table of headline metrics per run instead of gating one run")
    args = parser.parse_args(argv)
    if (args.run_dir is None) == (args.runs is None):
        parser.error("give either RUN_DIR or --runs")
    if args.runs:
        print(runs_table(args.runs))
        return 0
    summary = load_summary(args.run_dir)
    path = Path(args.run_dir) / "evaluation" / "summary.json"
    if summary is None:
        print(f"FAIL: {path} does not exist; run --stage evaluate first")
        return 1
    report = verify(summary)
    for name, rec in report["criteria"].items():
        print(f"[{rec['verdict']:>12s}] {name}: {rec['detail']}")
    print(f"VERDICT: {'PASS' if report['passed'] else 'FAIL'}"
          + (f" (inconclusive: {report['inconclusive']})" if report["inconclusive"] else ""))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report | {"summary_path": str(path)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
