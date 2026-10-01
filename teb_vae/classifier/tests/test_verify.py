"""verify.py (T-V1): every §11.15 criterion hits PASS / FAIL / INCONCLUSIVE on a synthetic summary; a missing registry
figure and a missing summary.md section fail; a stale figure left by an earlier render never counts, and a failed report
step fails the gate; the sanity producers verify re-reads (T2 NP tolerance, block C confound); ``--runs``."""
from __future__ import annotations

import copy


from teb_vae.classifier import metrics as M
from teb_vae.classifier import report as R
from teb_vae.classifier import verify as V

CFG = {"run": {"folds": [1, 2]}, "eval": {"per_fold_figures": True, "primary_policy": "np30"}}
OK = {"verdict": "pass", "detail": "ok"}


def _head(m, sd, pid, metric, test, ci):
    return {f"{m}|{sd}|guid|{pid}|{metric}": {"model_id": m, "seed": sd, "level": "guid", "policy_id": pid,
                                                "metric": metric, "test": test, "test_ci": ci, "fold_mean": test}}


GOOD = {
    "exit_code": 0, "failed": [], "n_failed": 0,
    "results": {
        "config": CFG,
        "sanity": {"checks": {k: OK for k in (*V.COHORT_CHECKS, *V.M7_CHECKS, "np_overshoot", "shuffled_auroc",
                                               "missing_confound")}},
        "headline": {**_head("probe", "42", "threshold_free", "auroc", 0.9, [0.8, 0.95]),
                     **_head("model", "42", "threshold_free", "auroc", 0.92, [0.85, 0.97]),
                     **_head("model", "42", "np30", "sens", 0.8, [0.6, 0.9]),
                     **_head("shortcut", "na", "threshold_free", "auroc", 0.5, [0.35, 0.65])},
        "artifacts": {"figures": [f"figures/{s}.pdf" for s in R.FIGURE_REGISTRY(CFG)]},
        "tables": {name: 1 for name in M.P2_TABLES},
        "report": {"exit_code": 0, "n_steps": 40, "failed": []},
        "predictions_written_at": "2026-01-01T00:00:00+00:00", "missing_units": [], "allow_partial": False,
        "prediction_units": [{"model_id": "model", "seed": "42", "fold": 1, "lock_written_at": "2026-01-01T00:00:00+00:00",
                              "test_written_at": "2026-01-01T00:00:01+00:00"}],
    },
    # verify.load_summary reads these from predictions/provenance.json and summary.md
    "on_disk": {"predictions_written_at": "2026-01-01T00:00:00+00:00",
                "summary_md_headings": [f"## {n}. Section" for n in range(1, 13)]},
}
def _verdicts(summary):
    return {k: v["verdict"] for k, v in V.verify(summary)["criteria"].items()}


def _with(path, value):
    s = copy.deepcopy(GOOD)
    node = s
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return s


def test_good_summary_passes():
    assert set(_verdicts(GOOD).values()) == {V.PASS} and V.verify(GOOD)["passed"]
    assert [name for name, _ in V.CRITERIA] == ["1_exit_code", "1b_evaluation_current", "2_units", "3_cohort",
                                                "4_selection_lock", "5_headline_finite", "6_np_overshoot",
                                                "7_shuffled_control", "8_model_vs_shortcut", "9_m7_consistency",
                                                "10_missing_confound", "11_expected_outputs", "12_summary_sections"]


def _checks(name, rec):
    return ("results", "sanity", "checks", name), rec


def test_a_check_that_covered_nothing_is_never_a_pass():
    """A criterion that evaluated zero items is INCONCLUSIVE, never PASS: the producer (``_verdict`` over n = 0, a cohort
    with no (fold, guid) pair), a stored ``pass`` with ``n_checked`` 0, and a summary without ``missing_units``."""
    assert M._verdict(0, [], "series")["verdict"] == "inconclusive" and M._verdict(3, [], "series")["verdict"] == "pass"
    vacuous = OK | {"n_checked": 0}
    assert _verdicts(_with(*_checks("m7_last_equals_final", vacuous)))["9_m7_consistency"] == V.INCONCLUSIVE
    assert _verdicts(_with(*_checks("cohort_disjoint", vacuous)))["3_cohort"] == V.INCONCLUSIVE
    assert _verdicts(_with(("results", "missing_units"), None))["2_units"] == V.INCONCLUSIVE
    assert _verdicts(GOOD)["2_units"] == V.PASS  # an empty recorded list is a real pass
