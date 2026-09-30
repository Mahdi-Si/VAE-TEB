"""report.py: every FIGURE_REGISTRY stem is produced, and empty or all-NaN inputs never raise (SPEC §11.10-§11.11)."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from teb_vae.classifier import metrics as M
from teb_vae.classifier import report as R
from teb_vae.classifier.tests.test_eval_analyses import make_run

AXES = ("to_delivery", "from_onset", "rel_second_stage", "position", "elapsed")


@pytest.fixture(scope="module")
def reported(tmp_path_factory):
    run, cfg = make_run(tmp_path_factory.mktemp("report"), overrides=["classifier.eval.figure_formats=[png]"])
    assert M.evaluate(run, cfg) == 0
    # keys other stages write (MLflow status, missing units, the shuffled-control check): report shows them
    for path, edit in (("manifest.json", lambda d: d | {"mlflow": {"enabled": True, "status": "unreachable",
                                                                  "parent_run_id": None, "reason": "connection refused"}}),
                       ("predictions/provenance.json", lambda d: d | {"missing_units": ["model|42|2"]}),
                       ("evaluation/summary.json", lambda d: d["results"]["sanity"]["checks"].update(
                           shuffled_auroc={"verdict": "fail", "detail": "shuffled-label control pooled test AUROC 0.700"}) or d)):
        (run / path).write_text(json.dumps(edit(json.loads((run / path).read_text()))))
    return run, cfg, R.report(run, cfg)


def _missing(run, cfg, fmt="png"):
    return [s for s in R.FIGURE_REGISTRY(cfg) if not (run / "evaluation" / "figures" / f"{s}.{fmt}").is_file()]


def test_every_registry_stem_produced(reported):
    run, cfg, code = reported
    assert code == 0 and not _missing(run, cfg)
    stems = R.FIGURE_REGISTRY(cfg)
    folds = cfg.classifier.run.folds
    assert {"fold_1/roc/roc_guid", "fold_3/roc/roc_guid", "val/roc/roc_guid", "val/roc/pr_guid"} <= set(stems)
    # M1 for the primary and every `at: <hours>` policy on every axis; M6 and R7 per axis; the core set per fold and val
    m1 = {f"metric_types/metric_types_{a}_{p}" for a in AXES for p in ("np30", "inst30_1h", "cum30_1h", "ovr30_1h")}
    assert m1 <= set(stems) and not any(s.endswith(("_emp30", "_youden")) for s in stems)
    assert {f"metric_types/segment_instantaneous_{a}" for a in AXES} | {f"roc/auroc_vs_time_{a}" for a in AXES} <= set(stems)
    assert {"fold_2/roc/roc_checkpoints_cumulative", "fold_3/metric_types/metric_types_to_delivery_np30",
            "val/roc/roc_checkpoints_cumulative", "val/metric_types/metric_types_to_delivery_np30",
            "roc/roc_segment", "roc/pr_checkpoints", "thresholds/decision_horizon", "thresholds/threshold_stability",
            "thresholds/metric_type_comparison", "alarms/lead_time", "alarms/false_alarms"} <= set(stems)
    assert len(stems) == len(set(stems))
    # P6 blocks: their pooled stems, and the per-fold / val core set (§11.10: K1, X1, S2 of the class family)
    core = {"calibration/calibration_guid", "confusion/confusion_binary", "subgroups/subgroups_vs_time_to_delivery_np30"}
    assert core | {"subgroups/subgroup_forest", "heterogeneity/fold_forest", "calibration/decision_curve",
                   "roc/score_distributions"} <= set(stems)
    assert {f"{where}/{s}" for where in ("val", *(f"fold_{k}" for k in folds)) for s in core} <= set(stems)
    assert not any(s.startswith("multiclass/roc_ovr") for s in stems)  # 3-class figures need task: three_class
    no_fold = cfg.classifier.model_copy(update={"eval": cfg.classifier.eval.model_copy(update={"per_fold_figures": False})})
    assert not any(s.startswith("fold_") for s in R.FIGURE_REGISTRY(no_fold))
    figs = json.loads((run / "evaluation" / "summary.json").read_text())["results"]["artifacts"]["figures"]
    assert {f"figures/{s}.png" for s in stems} <= set(figs)  # report refreshed the manifest
    md = (run / "summary.md").read_text()
    heads = [line for line in md.splitlines() if line.startswith("## ")]
    assert [h.split(".")[0] for h in heads] == [f"## {n}" for n in range(1, 13)]  # §11.11, verify criterion 12
    sec = {n: md.split(f"\n## {n}. ")[1].split("\n## ")[0] for n in range(1, 13)}
    assert "| probe | 42 |" in sec[6] and "calib_slope" in sec[6]  # no neural unit, so no calibration.json rows
    assert "Not applicable: binary task" in sec[7] and "_pending" not in sec[9]
    assert "Internal validation only" in sec[11] and "NP guarantee at GUID level only" in sec[11]
    assert "| Calibration | yes |" in sec[12] and "| Performance with CIs | yes |" in sec[12]
    assert "WARNING: shortcut" in md and "optimistic" in md
    assert "- MLflow: unreachable (parent run: n/a; connection refused)" in md
    assert "- missing units (not trained or predicted): model|42|2" in md
    assert "- shortcut baseline: **fail**" in md
    assert "- shuffled-label control: **fail** (shuffled-label control pooled test AUROC 0.700)" in md
    s8 = md.split("\n## 8. ")[1].split("\n## ")[0]
    for mt in M.TYPES:
        assert f"| probe | 42 | {mt} | 1 h |" in s8 and f"| probe | 42 | {mt} | end |" in s8
    assert "snapshot AUROC [CI] (n+/n-)" in s8 and "- m7_committed_overall_monotone: **pass**" in s8
    s10 = sec[10].split("### Alarm and lead-time summary")[1].split("\n### ")[0]  # the A1-A5 part of §10
    assert "| probe | 42 | latch | np30 |" in s10 and "IQR" in s10 and "shortcut" not in s10  # no online score


def test_all_nan_time_resolved_panels_do_not_raise(reported):
    """Every P4 builder on time-resolved rows whose values and CIs are all NaN (built, not rendered)."""
    import matplotlib.pyplot as plt

    run, cfg, _ = reported
    c = M.classifier_cfg(cfg)
    T = R.load_tables(run)
    T["tr"][["value", "ci_lo", "ci_hi"]] = np.nan
    T["alarms"][["lead_time_h", "time_to_first_alarm_h"]] = np.nan
    builders = R._figure_set(c)
    for stem in [s for s in builders if s.startswith(("metric_types/", "alarms/"))
                 or s in (R.ROC_CHECKPOINTS_SNAPSHOT, R.DECISION_HORIZON, R.THRESHOLD_STABILITY,
                          R.METRIC_TYPE_COMPARISON, R.AUROC_VS_TIME.format(axis="to_delivery"))]:
        plt.close(builders[stem](T, c))
    assert "no online score" in R._reasons(json.dumps({"no_online_score": 3}))


def test_serial_fallback_and_fail_soft_figures(reported, monkeypatch):
    """A pool that cannot start renders serially; a raising figure comes back as its traceback, which
    ``report`` re-raises under ``Report.step`` (a failed step, exit 1)."""
    import concurrent.futures

    run, cfg, _ = reported

    def no_pool(*_, **__):
        raise OSError("no subprocesses here")

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", no_pool)
    monkeypatch.setattr(R, "FIGURE_REGISTRY", lambda c: ["cohort/cohort_overview", "no/such_stem"])
    out = R._render_all(run, M.classifier_cfg(cfg))
    assert [n for n, _ in out] == ["cohort/cohort_overview.png", "no/such_stem.png"]
    assert out[0][1] is None and "KeyError" in out[1][1]
    with pytest.raises(RuntimeError, match="KeyError"):
        R._raise(out[1][1])


def test_empty_inputs_do_not_raise(reported, tmp_path):
    src, full, _ = reported
    run = shutil.copytree(src, tmp_path / "run")
    # every builder, fewer copies of it (other axes and the per-fold sets reuse the same builders)
    cfg = full.classifier.model_copy(update={"eval": full.classifier.eval.model_copy(
        update={"time_axes": ["to_delivery", "rel_second_stage"], "per_fold_figures": False})})
    empty = tmp_path / "empty"
    empty.mkdir()
    assert R.report(empty, cfg) == 0 and not _missing(empty, cfg)  # no tables at all
    for name in (*M.P2_TABLES, "alarms"):  # tables with their columns but zero rows
        p = run / "evaluation" / "tables" / f"{name}.parquet"
        pd.read_parquet(p).iloc[:0].to_parquet(p)
    p = run / "evaluation" / "tables" / "inclusion.csv"
    pd.read_csv(p).iloc[:0].to_csv(p, index=False)
    for name in ("segments", "guids"):
        p = run / "cohort" / f"{name}.parquet"
        pd.read_parquet(p).iloc[:0].to_parquet(p)
    p = run / "predictions" / "guids.parquet"
    pd.read_parquet(p).iloc[:0].to_parquet(p)
    assert R.report(run, cfg) == 0 and not _missing(run, cfg)


def _no_pool(*_, **__):
    raise OSError("no subprocesses here")


def test_missing_roc_points_draw_the_empty_note(reported):
    """R1 failed (no roc_points.parquet) while metrics exist: every ROC panel shows EMPTY_NOTE, never a KeyError."""
    import matplotlib.pyplot as plt

    run, cfg, _ = reported
    c, T = M.classifier_cfg(cfg), R.load_tables(run)
    T["roc"] = pd.DataFrame()
    for fig in (R._roc_guid(T, c), R._roc_guid(T, c, fold="1"), R._roc_guid(T, c, level="segment", variant="segment"),
                R._roc_checkpoints(T, c, kind="committed_overall")):
        assert R._seam().EMPTY_NOTE in [t.get_text() for ax in fig.axes for t in ax.texts]
        plt.close(fig)


def test_pooled_false_alarm_histogram_counts_each_guid_once(reported):
    """The shared test GUID sits in every fold's test split; the pooled histogram counts it once, as the
    pooled A1 rows its legend quotes (data.shared_test_policy)."""
    import matplotlib.pyplot as plt

    run, cfg, _ = reported
    c, T = M.classifier_cfg(cfg), R.load_tables(run)
    end = T["tr"][(T["tr"]["level"] == "alarm") & (T["tr"]["split"] == "test") & (T["tr"]["fold"] == "pooled")
                  & (T["tr"]["point"] == "end") & (T["tr"]["subgroup_value"] == "latch") & (T["tr"]["metric"] == "event_fpr")]
    want = {r.policy_id: round(r.value * r.n_neg) for r in end.itertuples()}
    a = T["alarms"]
    shared = (a["guid"] == "g000") & (a["split"] == "test") & (a["rule"] == "latch")
    assert a.loc[shared, "fold"].nunique() == 3 and not a.loc[shared, "alarmed"].any()
    a.loc[shared, ["alarmed", "time_to_first_alarm_h"]] = [True, 1.0]  # now it false-alarms in all 3 folds
    fig = R._alarms(T, c, what="false_alarms")
    got = {lab.split(" (n = ")[0]: int(lab.split(" (n = ")[1].rstrip(")"))
           for ax in fig.axes for lab in ax.get_legend_handles_labels()[1] if " (n = " in lab}
    plt.close(fig)
    assert got == {k: v + 1 for k, v in want.items()}  # once, not once per fold


def test_m1_title_names_the_policy_axis(reported):
    import matplotlib.pyplot as plt

    run, cfg, _ = reported
    c = M.classifier_cfg(cfg)
    c["eval"]["thresholds"] = [*c["eval"]["thresholds"], {"id": "onset2", "policy": "fpr_cap", "alpha": 0.3,
                                                           "basis": "committed_overall", "axis": "from_onset", "at": 2.0}]
    fig = R._metric_types(R.load_tables(run), c, axis="from_onset", pid="onset2")
    title = fig._suptitle.get_text()
    plt.close(fig)
    assert "2 h after labour onset" in title and "before delivery" not in title


def test_report_workers_zero_renders_serially_without_a_pool(reported, monkeypatch):
    """``run.report_workers: 0`` renders in this process (one copy of the tables: each worker holds its own, ~3-5 GB
    at full scale), never starting a forkserver."""
    import multiprocessing

    run, cfg, _ = reported
    monkeypatch.setattr(multiprocessing, "get_context", lambda *a: pytest.fail("a pool was started"))
    monkeypatch.setattr(R, "FIGURE_REGISTRY", lambda c: [R.COHORT_OVERVIEW, R.ROC_GUID])
    c = M.classifier_cfg(cfg)
    c["run"]["report_workers"] = 0
    assert [tb for _, tb in R._render_all(run, c)] == [None, None]


def test_worker_loads_the_tables_once_per_report(reported, monkeypatch):
    import concurrent.futures

    run, cfg, _ = reported
    calls, real = [], R.load_tables
    monkeypatch.setattr(R, "load_tables", lambda d: calls.append(d) or real(d))
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _no_pool)
    monkeypatch.setattr(R, "FIGURE_REGISTRY", lambda c: [R.COHORT_OVERVIEW, R.ROC_GUID, R.PR_GUID])
    for n in (1, 2):  # a later report call reloads (its tables may have been re-evaluated)
        assert all(tb is None for _, tb in R._render_all(run, M.classifier_cfg(cfg))) and len(calls) == n


def test_k3_legend_reads_the_primary_seeds_slope_and_intercept():
    """K3's legend shows the primary model's own (model_id, seed) S1 slope and intercept, never another seed's."""
    import matplotlib.pyplot as plt

    from teb_vae.classifier.config import load

    c = M.classifier_cfg(load())
    base = dict(split="test", fold="pooled", subgroup="cs", model_id="model")
    k3 = [base | dict(analysis="K3", seed=sd, subgroup_value=v, t=t, value=t, ci_lo=t, ci_hi=t, n_pos=5, n_neg=5)
          for sd in ("43", "42") for v in ("cs_pos", "cs_neg") for t in (0.2, 0.6)]
    s1 = [base | dict(analysis="S1", seed=sd, subgroup_value=v, metric=m, value=val)
          for sd, val in (("43", 9.0), ("42", 1.25)) for v in ("cs_pos", "cs_neg")
          for m in ("calib_slope", "calib_intercept")]  # seed 43's rows first: a model_id-only match took them
    fig = R._calibration_subgroups({"subgroups": pd.DataFrame(k3 + s1)}, c)
    labels = [t for ax in fig.axes for t in ax.get_legend_handles_labels()[1]]
    plt.close(fig)
    assert labels and all("slope 1.250, intercept 1.250" in t for t in labels), labels


def test_page_geometry_keeps_a_zero_trim():
    """E2 pages place steps at ``trim_minutes`` from the segment start: a recorded 0 is a 0-s offset, not 60 s."""
    assert R.page_geometry({"trim_minutes": 0.0, "time_pool": 2}) == (8.0, 0.0)
    assert R.page_geometry({"trim_minutes": 1.0}) == (4.0, 60.0)
    assert R.page_geometry({}) == (4.0, 60.0)  # unrecorded: the production trim
