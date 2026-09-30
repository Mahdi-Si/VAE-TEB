"""verify.py (T-V1): every §11.15 criterion hits PASS / FAIL / INCONCLUSIVE on a synthetic summary; a missing registry
figure and a missing summary.md section fail; a stale figure left by an earlier render never counts, and a failed report
step fails the gate; the sanity producers verify re-reads (T2 NP tolerance, block C confound); ``--runs``."""
from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest

from teb_vae.classifier import metrics as M
from teb_vae.classifier import report as R
from teb_vae.classifier import verify as V
from teb_vae.classifier.tests.test_eval_analyses import make_run

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
AUC = ("results", "headline", "model|42|guid|threshold_free|auroc")
UNIT = GOOD["results"]["prediction_units"][0]
FAILED = {"verdict": "fail", "detail": "bad"}
UNSURE = {"verdict": "inconclusive", "detail": "?"}


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


@pytest.mark.parametrize("path,value,expected", [
    (("exit_code",), 1, {"1_exit_code": V.FAIL}),
    (("failed",), ["R1"], {"1_exit_code": V.FAIL}),
    (("results", "report"), {"exit_code": 1, "failed": ["roc/roc_guid.pdf"]}, {"1_exit_code": V.FAIL}),
    (("results", "report"), None, {"1_exit_code": V.INCONCLUSIVE}),
    (("on_disk", "predictions_written_at"), "2026-01-02T00:00:00+00:00", {"1b_evaluation_current": V.FAIL}),
    (("results", "predictions_written_at"), None, {"1b_evaluation_current": V.INCONCLUSIVE}),
    (("results", "missing_units"), ["model|42|1"], {"2_units": V.FAIL}),
    (*_checks("cohort_exposure", {"verdict": "fail", "detail": "overlap"}), {"3_cohort": V.FAIL}),
    (*_checks("cohort_exposure", UNSURE), {"3_cohort": V.INCONCLUSIVE}),
    (("results", "prediction_units"), [UNIT | {"lock_written_at": "2026-01-01T00:00:02+00:00"}],
     {"4_selection_lock": V.FAIL}),
    (("results", "prediction_units"), [UNIT | {"lock_written_at": UNIT["test_written_at"]}], {"4_selection_lock": V.FAIL}),
    (("results", "prediction_units"), [UNIT | {"lock_written_at": None}], {"4_selection_lock": V.INCONCLUSIVE}),
    (("results", "prediction_units"), [], {"4_selection_lock": V.INCONCLUSIVE}),
    (("results", "headline", "probe|42|guid|threshold_free|auroc"), {"test": None, "fold_mean": 0.8},
     {"5_headline_finite": V.FAIL}),
    (("results", "headline"), {}, {"5_headline_finite": V.FAIL, "8_model_vs_shortcut": V.INCONCLUSIVE}),
    (("results", "headline", "probe|42|guid|threshold_free|auroc"), {"test": 0.9, "fold_mean": 0.88, "n_folds": 3,
                                                                     "n_folds_nan": 1}, {"5_headline_finite": V.INCONCLUSIVE}),
    (*_checks("np_overshoot", FAILED), {"6_np_overshoot": V.FAIL}),
    (*_checks("np_overshoot", UNSURE), {"6_np_overshoot": V.INCONCLUSIVE}),
    (*_checks("shuffled_auroc", FAILED), {"7_shuffled_control": V.FAIL}),
    (*_checks("shuffled_auroc", None), {"7_shuffled_control": V.INCONCLUSIVE}),  # no shuffled unit
    (AUC, GOOD["results"]["headline"][AUC[-1]] | {"test": 0.45, "test_ci": [0.3, 0.6]}, {"8_model_vs_shortcut": V.FAIL}),
    (AUC, GOOD["results"]["headline"][AUC[-1]] | {"test": 0.7, "test_ci": [0.55, 0.85]},
     {"8_model_vs_shortcut": V.INCONCLUSIVE}),  # above the shortcut, but the CIs overlap
    (AUC, GOOD["results"]["headline"][AUC[-1]] | {"test_ci": [None, None]}, {"8_model_vs_shortcut": V.INCONCLUSIVE}),
    (("results", "headline", "shortcut|na|guid|threshold_free|auroc"),  # no shortcut AUROC in the headline
     GOOD["results"]["headline"]["shortcut|na|guid|threshold_free|auroc"] | {"metric": "auprc"},
     {"8_model_vs_shortcut": V.INCONCLUSIVE}),
    (*_checks("m7_committed_overall_monotone", FAILED), {"9_m7_consistency": V.FAIL}),
    (*_checks("m7_end_equals_running_max", UNSURE), {"9_m7_consistency": V.INCONCLUSIVE}),  # k_of_n alarm rule
    (*_checks("m7_last_equals_final", None), {"9_m7_consistency": V.INCONCLUSIVE}),
    (*_checks("missing_confound", FAILED), {"10_missing_confound": V.FAIL}),
    (*_checks("missing_confound", UNSURE), {"10_missing_confound": V.INCONCLUSIVE}),
    (("results", "artifacts", "figures"), [f"figures/{s}.pdf" for s in R.FIGURE_REGISTRY(CFG)][1:],
     {"11_expected_outputs": V.FAIL}),
    (("results", "tables", "roc_points"), 0, {"11_expected_outputs": V.FAIL}),
    (("results", "config"), None, {"11_expected_outputs": V.INCONCLUSIVE}),
    (("on_disk", "summary_md_headings"), [f"## {n}. Section" for n in range(1, 13) if n != 7],
     {"12_summary_sections": V.FAIL}),
    (("on_disk", "summary_md_headings"), [f"## {n}. Section" for n in (1, 2, 3, 4, 5, 7, 6, 8, 9, 10, 11, 12)],
     {"12_summary_sections": V.FAIL}),
    (("on_disk", "summary_md_headings"), None, {"12_summary_sections": V.FAIL}),  # report never wrote summary.md
])
def test_criteria_verdicts(path, value, expected):
    got = _verdicts(_with(path, value))
    assert {k: got[k] for k in expected} == expected
    assert all(v == V.PASS for k, v in got.items() if k not in expected)
    assert V.verify(_with(path, value))["passed"] == (V.FAIL not in expected.values())


def test_missing_units_under_allow_partial_are_inconclusive():
    s = _with(("results", "missing_units"), ["model|42|1"])
    s["results"]["allow_partial"] = True
    assert _verdicts(s)["2_units"] == V.INCONCLUSIVE and V.verify(s)["passed"]


def test_model_vs_shortcut_falls_back_to_the_probe_and_names_the_loser():
    """A probe-only run compares the probe; every non-baseline model is compared, and a FAIL names the loser."""
    s = copy.deepcopy(GOOD)
    del s["results"]["headline"][AUC[-1]]
    assert V.check_model_vs_shortcut(s)["verdict"] == V.PASS  # probe 0.90 [0.80, 0.95] vs shortcut [.., 0.65]
    s["results"]["headline"] |= _head("model", "43", "threshold_free", "auroc", 0.4, [0.2, 0.6])
    rec = V.check_model_vs_shortcut(s)
    assert rec["verdict"] == V.FAIL and "not above the shortcut: ['model|43']" in rec["detail"]


def test_model_vs_shortcut_gates_the_primary_model_only():
    """Only the primary model (``model``'s ensemble, else its first seed) is gated; a diagnostic row below the shortcut
    (frozen baseline, noind ablation, covariates-off pass, another seed) is listed, never a FAIL."""
    s = copy.deepcopy(GOOD)
    for m, sd in (("noind", "42"), ("frozen", "42"), ("model_covoff", "42"), ("model", "43")):
        s["results"]["headline"] |= _head(m, sd, "threshold_free", "auroc", 0.4, [0.2, 0.6])
    rec = V.check_model_vs_shortcut(s)
    assert rec["verdict"] == V.PASS and "noind|42" in rec["detail"] and "not gated" in rec["detail"]
    s["results"]["headline"] |= _head("model", "ens", "threshold_free", "auroc", 0.45, [0.3, 0.6])
    rec = V.check_model_vs_shortcut(s)  # the ensemble is primary once present
    assert rec["verdict"] == V.FAIL and "not above the shortcut: ['model|ens']" in rec["detail"]


def test_main_exit_codes_json_out_and_summary_md(tmp_path):
    (tmp_path / "evaluation").mkdir()
    good = {k: v for k, v in GOOD.items() if k != "on_disk"}  # load_summary reads on_disk from the run dir
    (tmp_path / "evaluation" / "summary.json").write_text(json.dumps(good))
    assert V.main([str(tmp_path)]) == 1  # no summary.md yet: criterion 12
    (tmp_path / R.SUMMARY_MD).write_text("# run\n\n" + "\n\n".join(f"## {n}. x\ntext\n### sub" for n in range(1, 13)))
    assert V.main([str(tmp_path), "--json-out", str(tmp_path / "v.json")]) == 0  # no provenance: 1b inconclusive only
    assert json.loads((tmp_path / "v.json").read_text())["inconclusive"] == ["1b_evaluation_current"]
    (tmp_path / "predictions").mkdir()
    (tmp_path / "predictions" / "provenance.json").write_text(json.dumps({"written_at": GOOD["on_disk"]["predictions_written_at"]}))
    assert V.main([str(tmp_path), "--json-out", str(tmp_path / "v.json")]) == 0
    assert json.loads((tmp_path / "v.json").read_text())["passed"] is True
    missing_fig = copy.deepcopy(good)
    missing_fig["results"]["artifacts"]["figures"] = good["results"]["artifacts"]["figures"][:-1]
    (tmp_path / "evaluation" / "summary.json").write_text(json.dumps(missing_fig))
    assert V.main([str(tmp_path)]) == 1
    assert V.main([str(tmp_path / "nowhere")]) == 1
    (tmp_path / "evaluation" / "summary.json").write_text(json.dumps(good))
    (tmp_path / "predictions" / "provenance.json").write_text(json.dumps({"written_at": "2026-01-02T00:00:00+00:00"}))
    assert V.main([str(tmp_path), "--json-out", str(tmp_path / "v.json")]) == 1  # a predict after the evaluation
    assert json.loads((tmp_path / "v.json").read_text())["failed"] == ["1b_evaluation_current"]
    with pytest.raises(SystemExit):
        V.main([])


def test_runs_table_keys_arms_on_config_differences(tmp_path, capsys):
    for name, per_fold in (("a", True), ("b", False)):
        s = copy.deepcopy(GOOD)
        s["results"]["config"] = {**CFG, "eval": {**CFG["eval"], "per_fold_figures": per_fold}}
        (tmp_path / name / "evaluation").mkdir(parents=True)
        (tmp_path / name / "evaluation" / "summary.json").write_text(json.dumps(s))
    assert V.main(["--runs", str(tmp_path / "a"), str(tmp_path / "b"), str(tmp_path / "c")]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("| run | eval.per_fold_figures | model | seed | AUROC fold mean ± SD |")
    row = next(x for x in lines if x.startswith(f"| {tmp_path / 'a'} |") and "| model | 42 |" in x)
    assert "| true | model | 42 | 0.920 ± - | 0.920 [0.850, 0.970] | 0.800 [0.600, 0.900] | - |" in row
    assert "FAIL" in row  # no summary.md or provenance beside these summaries: criterion 12 fails
    assert any("| false | shortcut | na |" in x for x in lines) and "no evaluation/summary.json" in lines[-1]


def test_producers_np_tolerance_and_confound(tmp_path):
    """T2's NP record carries the 95% binomial tolerance on the pooled negatives; block C's confound check walks every
    flagged variable (has_tlo and covariates) and passes once the `noind` ablation is present."""
    from scipy.stats import binom

    run, cfg = make_run(tmp_path, folds=(1, 2), resamples=5)
    assert M.evaluate(run, cfg) == 0
    res = json.loads((run / "evaluation" / "summary.json").read_text())["results"]
    rec = res["T2"]["overshoot"]["probe|42|guid|np30"]
    n = rec["n_neg"]
    assert rec["alpha"] == 0.3 and rec["tolerance"] == pytest.approx(binom.ppf(0.95, n, 0.3) / n - 0.3)
    assert "tolerance" not in res["T2"]["overshoot"]["probe|42|guid|emp30"]  # empirical: no guarantee to test
    npo = res["sanity"]["checks"]["np_overshoot"]
    assert npo["verdict"] == ("fail" if any(r["pooled"] > r["tolerance"] + 1e-9 for r in res["T2"]["overshoot"].values()
                                            if "tolerance" in r and not r["fallback_folds"]) else "pass")
    assert res["sanity"]["checks"]["missing_confound"]["verdict"] == "pass"

    (run / "cohort" / "confound.json").write_text(json.dumps({
        "1": {"has_tlo": {"delta": 0.2, "flagged": True}},
        "2": {"has_tlo": {"delta": 0.01, "flagged": False}, "covariates": {"parity": {"delta": 0.3, "flagged": True}}}}))
    ctx = M.load_context(run, cfg)
    got = M.run_C(ctx, eval_config=ctx.cfg["eval"], out_dir=run / "evaluation")["sanity"]["missing_confound"]
    assert got["verdict"] == "fail" and got["flagged"] == ["1/has_tlo (delta 0.200)", "2/covariates/parity (delta 0.300)"]
    ctx.guids = ctx.guids.assign(model_id=M.NOIND)  # the no_indicator ablation's rows
    got = M.run_C(ctx, eval_config=ctx.cfg["eval"], out_dir=run / "evaluation")["sanity"]["missing_confound"]
    assert got["verdict"] == "pass" and got["ablation_present"]


def test_unknown_pretraining_exposure_is_never_clean(tmp_path):
    """L3: a fold whose pretraining or selection population is unknown (the checkpoint's resolved config names no shard
    list) makes block C's exposure check, and so criterion 3, INCONCLUSIVE; a known, disjoint one passes."""
    run, cfg = make_run(tmp_path, folds=(1, 2), resamples=5)
    ctx = M.load_context(run, cfg)
    known = {"known": True, "n_exposed": 0}
    for sel, want, crit in (({"known": False, "n_exposed": None}, "inconclusive", V.INCONCLUSIVE),
                            (known, "pass", V.PASS)):
        (run / "cohort" / "exposure.json").write_text(json.dumps({"applicable": True, "folds": {
            f: {"pretraining": known, "selection": sel, "test_overlap": []} for f in ("1", "2")}}))
        got = M.run_C(ctx, eval_config=ctx.cfg["eval"], out_dir=run / "evaluation")["sanity"]
        assert got["cohort_exposure"]["verdict"] == want
        assert V.check_cohort({"results": {"sanity": {"checks": got}}})["verdict"] == crit


def test_verify_is_torch_free():
    code = "import sys, teb_vae.classifier.verify; print('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "False"


def test_end_to_end_evaluate_report_verify(tmp_path):
    run, cfg = make_run(tmp_path, folds=(1, 2), resamples=5, overrides=["classifier.eval.figure_formats=[png]"])
    assert V.main([str(run)]) == 1  # no evaluation yet
    assert M.evaluate(run, cfg) == 0
    assert V.main([str(run)]) == 1  # evaluated, figures and summary.md not rendered yet
    assert R.report(run, cfg) == 0
    assert V.main([str(run)]) == 0
    got = V.verify(V.load_summary(run))["criteria"]
    assert got["12_summary_sections"]["verdict"] == V.PASS and got["8_model_vs_shortcut"]["verdict"] != V.FAIL


def test_nan_fold_is_named():
    rec = V.check_headline_finite(_with(("results", "headline", "probe|42|guid|np30|sens"),
                                        {"test": 0.8, "fold_mean": 0.7, "n_folds": 3, "n_folds_nan": 1}))
    assert rec["verdict"] == V.INCONCLUSIVE and "probe|42|guid|np30|sens (1 of 3 folds)" in rec["detail"]


def test_stale_figure_and_failed_redraw_fail_the_gate(tmp_path, monkeypatch):
    """A re-evaluated run whose report fails to redraw roc_guid: the figure left by the earlier render is
    older than the evaluation, so criterion 11 misses it, and the failed report step fails criterion 1."""
    import concurrent.futures

    stems = [R.ROC_GUID, R.PR_GUID]
    for mod in (R, V):
        monkeypatch.setattr(mod, "FIGURE_REGISTRY", lambda c: stems)

    def no_pool(*_, **__):  # render serially in this process, so the patched builder below applies
        raise OSError("no subprocesses here")

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", no_pool)
    run, cfg = make_run(tmp_path, folds=(1, 2), resamples=5, overrides=["classifier.eval.figure_formats=[png]"])
    assert M.evaluate(run, cfg) == 0 and R.report(run, cfg) == 0 and V.main([str(run)]) == 0
    assert M.evaluate(run, cfg) == 0

    def boom(*_, **__):
        raise RuntimeError("cannot redraw")

    monkeypatch.setitem(R._BUILDERS, R.ROC_GUID, boom)
    assert R.report(run, cfg) == 1
    assert (run / "evaluation" / "figures" / f"{R.ROC_GUID}.png").is_file()  # the stale copy is still on disk
    got = V.verify(json.loads((run / "evaluation" / "summary.json").read_text()))["criteria"]
    assert got["11_expected_outputs"]["verdict"] == V.FAIL and got["11_expected_outputs"]["missing_figures"] == [R.ROC_GUID]
    assert got["1_exit_code"]["verdict"] == V.FAIL and got["1_exit_code"]["report_failed"] == [f"{R.ROC_GUID}.png"]
    assert V.main([str(run)]) == 1


def test_file_clock_start_and_a_failed_manifest_is_recorded(tmp_path, monkeypatch):
    """The evaluate start is the file clock's (a host clock an hour ahead of the file server must not make this
    evaluation's own tables stale), and a report whose artifacts step raises records that failure, not the
    previous report's success."""
    import time

    from teb_vae.lag_attn.eval import report as seam
    from teb_vae.lag_attn.eval.report import Report

    run, cfg = make_run(tmp_path, folds=(1, 2), resamples=5)
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 3600.0)
    assert M.evaluate(run, cfg) == 0
    monkeypatch.undo()
    path = run / "evaluation" / "summary.json"
    summary = json.loads(path.read_text())
    assert "tables/metrics.parquet" in summary["results"]["artifacts"]["files"]

    summary["results"]["report"] = {"exit_code": 0, "n_steps": 40, "failed": []}  # an earlier report's record
    path.write_text(json.dumps(summary))

    def boom(*_, **__):
        raise OSError("manifest walk failed")

    monkeypatch.setattr(seam, "build_manifest", boom)
    rep = Report()
    rep.step("artifacts", R._refresh_artifacts, run / "evaluation", rep)
    assert [r.name for r in rep.failed_steps] == ["artifacts"]
    assert json.loads(path.read_text())["results"]["report"] == {"exit_code": 1, "n_steps": 0, "failed": ["artifacts"]}


def test_a_check_that_covered_nothing_is_never_a_pass():
    """A criterion that evaluated zero items is INCONCLUSIVE, never PASS: the producer (``_verdict`` over n = 0, a cohort
    with no (fold, guid) pair), a stored ``pass`` with ``n_checked`` 0, and a summary without ``missing_units``."""
    assert M._verdict(0, [], "series")["verdict"] == "inconclusive" and M._verdict(3, [], "series")["verdict"] == "pass"
    vacuous = OK | {"n_checked": 0}
    assert _verdicts(_with(*_checks("m7_last_equals_final", vacuous)))["9_m7_consistency"] == V.INCONCLUSIVE
    assert _verdicts(_with(*_checks("cohort_disjoint", vacuous)))["3_cohort"] == V.INCONCLUSIVE
    assert _verdicts(_with(("results", "missing_units"), None))["2_units"] == V.INCONCLUSIVE
    assert _verdicts(GOOD)["2_units"] == V.PASS  # an empty recorded list is a real pass
