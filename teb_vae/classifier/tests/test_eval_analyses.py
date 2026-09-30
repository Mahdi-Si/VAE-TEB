"""P2 evaluation pipeline: the analysis registry end to end, T-E6 and T-E7 (SPEC §11.12-§11.14, §15).

:func:`make_run` fabricates a CONTRACT-shaped run directory (config, cohort tables, predictions,
thresholds, provenance) with a planted signal; ``test_report`` and ``test_verify`` reuse it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
import yaml

from teb_vae.classifier import metrics as M

CLASS = {0: ("healthy", 1, "healthy_bg_no_cs"), 1: ("acidosis", 2, "acidosis_no_cs"), 3: ("hie", 3, "hie_cs")}


def _split_of(i: int, fold: int) -> str:
    if i == 0:  # the shared test GUID: in every fold's test split
        return "test"
    return "test" if i % 5 == fold - 1 else "val" if i % 5 == fold % 5 else "train"


def make_run(root: Path, *, folds: Sequence[int] = (1, 2, 3), n_guids: int = 100, resamples: int = 20,
             overrides: Sequence[str] = (), seed: int = 0) -> Tuple[Path, Any]:
    """A run dir with ``config.resolved.yaml``, ``manifest.json``, ``cohort/*`` and ``predictions/*``.

    GUID ``i`` is healthy for even i, acidosis for i % 4 == 1, HIE otherwise; 8-12 segments on a
    660-s grid ending in the last 20 min. Positives' segment logits are shifted up (more so in the
    last hour), so AUROC is high; the shortcut baseline scores GUID length with a weak signal.
    """
    from teb_vae.classifier.config import digest, load
    from teb_vae.classifier.thresholds import select_thresholds

    cfg = load(overrides=[f"classifier.run.folds=[{','.join(map(str, folds))}]",
                          f"classifier.eval.bootstrap.resamples={resamples}", *overrides])
    run, rng = Path(root) / "run", np.random.default_rng(seed)
    (run / "predictions").mkdir(parents=True)
    (run / "cohort").mkdir()
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))

    segs = []
    for i in range(n_guids):
        name, code, sub = CLASS[i % 4 if i % 2 else 0]
        n, last = 8 + i % 5, -rng.uniform(60, 1200)
        t_end = last - 660.0 * np.arange(n)[::-1]
        y = int(code > 1)
        segs.append(pd.DataFrame({
            "guid": f"g{i:03d}", "seg_pos": np.arange(n), "slot": np.arange(n), "t_end_s": t_end,
            "epoch_s": t_end - 1260.0, "class_code": code, "clinical_class": name, "source_file": sub, "subgroup": sub,
            "y": y, "cs": sub.endswith("_cs") and "no_cs" not in sub, "bg": True, "has_tlo": i % 3 > 0, "has_ss": i % 2 == 0,
            "tlo_end_s": np.where(i % 3 > 0, t_end + 6 * 3600.0, np.nan), "ss_rel_s": np.nan, "stage": "unknown",
            "valid_frac": 1.0, "n_valid_steps": 300,
            "base": rng.normal(0, 1, n) + rng.normal(0, 0.7) + 1.5 * y * (1 + (t_end > -3600)) / 1.25,
        }))
    base = pd.concat(segs, ignore_index=True)
    base["hours_to_delivery"] = -base["t_end_s"] / 3600
    base["shared_test"] = base["guid"] == "g000"
    idx = base["guid"].str[1:].astype(int)

    cseg, cgd, pseg, pgd, thr = [], [], [], [], {}
    policies = [p.model_dump() for p in cfg.classifier.eval.thresholds]
    for fold in folds:
        split = idx.map(lambda i: _split_of(i, fold))
        s = base.assign(fold=fold, split=split, in_eval_window=True, label_weight=1.0)
        cseg.append(s)
        for m, sd in (("probe", "42"), ("shortcut", "na")):
            p = s[s["split"] != "train"].assign(model_id=m, seed=sd, run_id="run")
            if m == "probe":
                p["logit_seg"] = p["base"] + rng.normal(0, 0.3, len(p))
                p["logit_online"] = p.groupby(["split", "guid"])["logit_seg"].transform(lambda v: v.expanding().mean())
            else:
                p["logit_seg"] = p["logit_online"] = np.nan
            p["logit_seg_cal"], p["logit_online_cal"] = p["logit_seg"], p["logit_online"]
            p["tlo_end_h"], p["ss_rel_h"] = p["tlo_end_s"] / 3600, np.nan
            g = p.groupby(["split", "guid"], as_index=False).agg(
                **{c: (c, "first") for c in ("run_id", "model_id", "seed", "fold", "class_code", "y", "cs", "bg",
                                             "clinical_class", "subgroup", "has_tlo", "has_ss", "shared_test")},
                n_segments=("seg_pos", "size"), first_epoch_s=("epoch_s", "min"), last_t_end_s=("t_end_s", "max"),
                score_final=("logit_online", "last"))
            if m == "shortcut":
                g["score_final"] = 0.2 * g["n_segments"] + 0.8 * g["y"] + rng.normal(0, 1, len(g))
            g["score_final_cal"] = g["score_final"]
            pseg.append(p.drop(columns=["base", "source_file", "tlo_end_s", "ss_rel_s", "slot"]).assign(slot=p["slot"]))
            pgd.append(g)
            vs, vg = p[p["split"] == "val"], g[g["split"] == "val"]
            L = cfg.classifier.eval.exclude_last_min
            if m == "probe":
                thr[f"{m}|{sd}|{fold}"] = select_thresholds(
                    vs, vg, policies, bin_h=0.5, exclude_last_min=L,
                    staleness_h=cfg.classifier.eval.snapshot_max_staleness_h)
            else:
                end = [q | {"basis": "guid_final"} for q in policies if q["at"] == "end"]
                t = select_thresholds(vs, vg, end, bin_h=0.5, exclude_last_min=L)
                t["guid"].update({q["id"]: {"skipped": "guid-level model"} for q in policies if q["at"] != "end"})
                thr[f"{m}|{sd}|{fold}"] = t

    seg = pd.concat(cseg, ignore_index=True).drop(columns=["base", "subgroup", "has_tlo", "has_ss", "n_valid_steps", "shared_test"])
    seg["excluded"], seg["exclusion_reason"] = False, ""
    seg.loc[seg.index[::97], ["excluded", "exclusion_reason"]] = [True, "low_valid_frac"]
    seg["ds_index"], seg["guid_norm"] = seg.index, seg["guid"].str.upper()
    gd = (pd.concat(cseg).groupby(["fold", "split", "guid"], as_index=False)
          .agg(**{c: (c, "first") for c in ("source_file", "class_code", "clinical_class", "y", "cs", "bg", "has_tlo",
                                            "has_ss", "shared_test")},
               n_segments=("seg_pos", "size"), first_epoch_s=("epoch_s", "min"), last_t_end_s=("t_end_s", "max")))
    gd["excluded"], gd["exclusion_reason"], gd["guid_norm"] = False, "", gd["guid"].str.upper()
    seg.to_parquet(run / "cohort" / "segments.parquet", index=False)
    gd.to_parquet(run / "cohort" / "guids.parquet", index=False)
    pd.concat([f.assign(level=lvl, status=f["exclusion_reason"].replace("", "retained"))
               .groupby(["fold", "split", "class_code", "source_file", "level", "status"]).size().rename("n").reset_index()
               for f, lvl in ((gd, "guid"), (seg, "segment"))]).to_csv(run / "cohort" / "fold_summary.csv", index=False)
    (run / "cohort" / "exposure.json").write_text(json.dumps({"applicable": False, "note": "hdf5 source"}))
    (run / "cohort" / "confound.json").write_text(json.dumps(
        {str(f): {"has_tlo": {"missing_rate_by_class": {}, "delta": 0.01, "flagged": False}} for f in folds}))

    pd.concat(pseg, ignore_index=True).to_parquet(run / "predictions" / "segments.parquet", index=False)
    pd.concat(pgd, ignore_index=True).to_parquet(run / "predictions" / "guids.parquet", index=False)
    (run / "predictions" / "thresholds.json").write_text(json.dumps(thr))
    (run / "predictions" / "provenance.json").write_text(json.dumps(
        {"config_digest": digest(cfg), "run_id": "run", "written_at": "now", "units": [], "missing_units": []}))
    (run / "manifest.json").write_text(json.dumps(
        {"config_digest": digest(cfg), "software": {"revision": "deadbeef"}, "cohort": {"sentinel_guids": []}}))
    return run, cfg


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    run, cfg = make_run(tmp_path_factory.mktemp("eval"))
    code = M.evaluate(run, cfg)
    return run, cfg, code, json.loads((run / "evaluation" / "summary.json").read_text())


def test_registry_end_to_end(evaluated):
    run, cfg, code, summary = evaluated
    assert code == 0 and summary["exit_code"] == 0 and summary["n_failed"] == 0, summary["failed"]
    assert [s["name"] for s in summary["steps"]] == [*M.ANALYSES, "tables"]
    res = summary["results"]
    for key in ("headline", "sanity", "artifacts", "config_digest", "run_context", "config", *M.ANALYSES):
        assert key in res
    tables = {n: pd.read_parquet(run / "evaluation" / "tables" / f"{n}.parquet") for n in M.P2_TABLES}
    assert all(len(t) for t in tables.values()) and res["tables"] == {n: len(t) for n, t in tables.items()}
    assert list(tables["metrics"].columns) == list(M.METRIC_COLUMNS)
    for f in ("dataset_summary.json", "label_cross_table.csv", "count_spread.csv"):
        assert (run / "cohort" / f).is_file()
    assert len(pd.read_csv(run / "cohort" / "label_cross_table.csv").query("fold == 'all'")) == 12
    assert json.loads((run / "evaluation" / "steps.json").read_text())[-1]["name"] == "tables"

    met = tables["metrics"]
    probe = res["headline"]["probe|42|guid|threshold_free|auroc"]
    assert probe["test"] > 0.85 and probe["n_folds"] == 3 and probe["primary"] == "fold_mean"
    assert res["headline"]["probe|42|guid|np30|sens"]["primary"] == "pooled"
    assert not any(k.startswith("shortcut|na|segment") for k in res["headline"])  # no segment scores
    # val rows on the basis population reproduce the selection-time rates in thresholds.json
    t = tables["thresholds"].query("model_id == 'probe' and fold == '1' and policy_id == 'emp30'").iloc[0]
    v = met.query("model_id == 'probe' and split == 'val' and fold == '1' and policy_id == 'emp30' and point == 'n/a'")
    assert v.set_index("metric").loc[["sens", "fpr"], "value"].tolist() == pytest.approx([t.val_sens, t.val_fpr])
    # pooled rows: primary policy, plus the other shared-test policy as a sensitivity row on test
    sens = met.query("model_id == 'probe' and fold == 'pooled' and policy_id == 'np30' and metric == 'sens'")
    assert set(sens["split"]) == {"val", "test"} and sens["subgroup"].fillna("").tolist().count("shared_test_policy") == 1
    assert (met.query("policy_id == 'oracle'")["metric"].unique() == [f"tpr@fpr{f:g}" for f in (0.05, 0.1, 0.15, 0.2, 0.3)]).all()
    assert set(tables["thresholds"].query("model_id == 'shortcut'")["skipped"].dropna()) == {"guid-level model"}
    assert res["sanity"]["checks"]["cohort_disjoint"]["verdict"] == "pass"
    assert res["sanity"]["checks"]["cohort_exposure"]["verdict"] == "pass"
    assert res["C"]["C12"]["post_warmup_coverage_frac"] is None and res["C"]["C12"]["stride_s"] == 660


def test_fail_soft_step_is_recorded(tmp_path, monkeypatch):
    """T-E7: one analysis raising -> n_failed = 1, the others complete, exit 1, traceback kept."""
    run, cfg = make_run(tmp_path, folds=(1, 2), resamples=5)

    def boom(ctx, **_):
        raise RuntimeError("planted failure")

    monkeypatch.setitem(M.ANALYSES, "R9", boom)
    assert M.evaluate(run, cfg, skip=["C"]) == 1
    summary = _partial_summary(run)
    assert summary["n_failed"] == 1 and summary["failed"] == ["R9"] and summary["exit_code"] == 1
    failed = next(s for s in summary["steps"] if s["name"] == "R9")
    assert "planted failure" in failed["traceback"] and "Traceback" in failed["traceback"]
    assert {s["name"] for s in summary["steps"] if s["ok"]} == set(M.ANALYSES) - {"C", "R9"} | {"tables"}
    with pytest.raises(ValueError, match="unknown analyses"):
        M.evaluate(run, cfg, only=["nope"])


def _partial_summary(run: Path) -> dict:
    (part,) = (run / "evaluation").glob("partial_*")
    return json.loads((part / "summary.json").read_text())


def test_partial_evaluation_never_touches_the_canonical_outputs(tmp_path):
    """--only/--skip write evaluation/partial_<stamp>/; a full re-run backs up summary, steps and tables."""
    run, cfg = make_run(tmp_path, folds=(1, 2), resamples=5)
    ev = run / "evaluation"
    assert M.evaluate(run, cfg) == 0
    canonical = {p: p.read_bytes() for p in (ev / "summary.json", ev / "steps.json", *(ev / "tables").iterdir())}
    assert M.evaluate(run, cfg, only=["metrics", "R1"]) == 0
    assert {p: p.read_bytes() for p in canonical} == canonical and not list(ev.glob("*.bak.*"))
    part = _partial_summary(run)
    assert part["results"]["partial"] and part["results"]["analyses_selected"] == ["metrics", "R1"]
    (pdir,) = ev.glob("partial_*")
    assert len(pd.read_parquet(pdir / "tables" / "metrics.parquet")) < len(pd.read_parquet(ev / "tables" / "metrics.parquet"))
    assert M.evaluate(run, cfg) == 0
    (tables_bak,) = ev.glob("tables.bak.*")
    assert len(list(ev.glob("summary.bak.*.json"))) == len(list(ev.glob("steps.bak.*.json"))) == 1
    assert (tables_bak / "metrics.parquet").read_bytes() == canonical[ev / "tables" / "metrics.parquet"]
    assert not json.loads((ev / "summary.json").read_text())["results"]["partial"]


def test_exclude_last_min_is_one_filter(tmp_path):
    """eval.exclude_last_min: the P2 committed_overall@end rows equal the P4 end rows, and the evaluated val
    FPR equals the selection-time val FPR in thresholds.json (every population is drop_last_minutes')."""
    run, cfg = make_run(tmp_path, folds=(1, 2), resamples=5, overrides=["classifier.eval.exclude_last_min=30"])
    assert M.evaluate(run, cfg, only=["metrics", "T1", "M"]) == 0
    (pdir,) = (run / "evaluation").glob("partial_*")
    met = pd.read_parquet(pdir / "tables" / "metrics.parquet")
    met = met[met["subgroup"].isna() & met["metric"].isin(["tp", "fp", "sens", "fpr"])]
    key = ["model_id", "seed", "split", "fold", "policy_id", "metric"]
    p2 = met[(met["level"] == "guid") & (met["point"] == "n/a") & (met["policy_basis"] == "committed_overall")
             & met["t"].isna()]
    p4 = met[(met["level"] == "online") & (met["axis"] == "to_delivery") & (met["point"] == "end")
             & (met["metric_type"] == "committed_overall") & met["policy_id"].isin(p2["policy_id"])]
    both = p2.merge(p4, on=key, suffixes=("_p2", "_p4"))
    assert len(both) == len(p2) == len(p4) > 0 and set(both["fold"]) == {"1", "2", "pooled"}
    assert both["value_p2"].tolist() == pytest.approx(both["value_p4"].tolist(), nan_ok=True)
    thr = pd.read_parquet(pdir / "tables" / "thresholds.parquet").query("model_id == 'probe' and skipped.isna()")
    val = met[(met["split"] == "val") & (met["metric"] == "fpr") & (met["point"] == "n/a") & (met["model_id"] == "probe")]
    got = val.set_index(["fold", "policy_id"])["value"]
    want = thr.set_index(["fold", "policy_id"])["val_fpr"]
    assert got.reindex(want.index).tolist() == pytest.approx(want.tolist())
    inc = pd.read_csv(pdir / "tables" / "inclusion.csv").query("analysis == 'exclude_last_min'")
    assert all(json.loads(r)["segments_dropped"] > 0 for r in inc["reasons"])


def test_recorded_stride_wins(tmp_path):
    """C12 reads the stride the cohort stage recorded (data.stride_s may differ from the inferred mode)."""
    run, cfg = make_run(tmp_path, folds=(1, 2), resamples=5)
    man = json.loads((run / "manifest.json").read_text())
    (run / "manifest.json").write_text(json.dumps(man | {"cohort": man["cohort"] | {"stride_s": 600.0}}))
    assert M.evaluate(run, cfg, only=["C"]) == 0
    assert _partial_summary(run)["results"]["C"]["C12"]["stride_s"] == 600.0
    assert M.infer_stride(pd.read_parquet(run / "cohort" / "segments.parquet")) == 660.0


def test_no_valid_steps_exclusions_reach_inclusion_and_summary(tmp_path):
    """The per-fold ``no_valid_steps`` record of ``baselines/fold_<k>/*_fit.json`` becomes L14 inclusion rows (per fold
    and split) and a cohort-flow line in summary.md."""
    from teb_vae.classifier import report as R

    run, cfg = make_run(tmp_path, folds=(1, 2), resamples=5)
    (run / "baselines" / "fold_1").mkdir(parents=True)
    rec = {"segments": {"train": 3, "val": 1, "test": 0}, "guids": {"train": 1, "val": 0, "test": 0},
           "min_segments": {"train": 0, "val": 1, "test": 0}}
    (run / "baselines" / "fold_1" / "probe_fit.json").write_text(json.dumps({"fold": 1, "no_valid_steps": rec}))
    assert M.evaluate(run, cfg) == 0
    inc = pd.read_csv(run / "evaluation" / "tables" / "inclusion.csv").query("analysis == 'no_valid_steps'")
    got = inc.set_index("split")
    assert list(inc["fold"].astype(str).unique()) == ["1"] and set(got.index) == {"train", "val", "test"}
    assert got.loc["train", "n_excluded"] == 1 and got.loc["val", "n_excluded"] == 1
    assert json.loads(got.loc["train", "reasons"])["segments"] == 3
    summary = json.loads((run / "evaluation" / "summary.json").read_text())
    assert summary["results"]["C"]["no_valid_steps"] == {"segments": 4, "guids": 2}
    md = R.summary_md(run, R.load_tables(run), R.classifier_cfg(cfg)).read_text()
    assert "excluded 4 segment(s) and 2 GUID(s)" in md


def test_patient_clusters_widen_the_intervals(tmp_path):
    """Every GUID doubled into a second GUID of the same patient: clustering on the patient reproduces the
    single-GUID interval (a patient's two GUIDs are drawn together); clustering on the GUIDs narrows it."""
    run, cfg = make_run(tmp_path, folds=(1, 2), resamples=400)
    pred = run / "predictions"
    orig = {name: pd.read_parquet(pred / f"{name}.parquet") for name in ("segments", "guids")}
    widths = {}
    for variant in ("single", "patient", "guid"):
        for name, df in orig.items():
            df = df.assign(patient=df["guid"])
            if variant != "single":
                df = pd.concat([df, df.assign(guid=df["guid"] + "b")], ignore_index=True)
            (df.drop(columns="patient") if variant == "guid" else df).to_parquet(pred / f"{name}.parquet", index=False)
        assert M.evaluate(run, cfg, only=["metrics"]) == 0
        (pdir,) = (run / "evaluation").glob("partial_*")
        met = pd.read_parquet(pdir / "tables" / "metrics.parquet").query(
            "model_id == 'probe' and level == 'guid' and split == 'test' and fold == 'pooled' and subgroup.isna()")
        pdir.rename(pdir.with_name(variant))
        widths[variant] = {m: float((q["ci_hi"] - q["ci_lo"]).iloc[0]) for m, q in (
            ("auroc", met.query("metric == 'auroc'")), ("sens", met.query("metric == 'sens' and policy_id == 'np30'")))}
    assert widths["patient"]["auroc"] == pytest.approx(widths["single"]["auroc"])
    assert widths["patient"]["sens"] == pytest.approx(widths["single"]["sens"], rel=0.2)
    # AUROC ~0.98 sits near its ceiling, so its interval shrinks less than 1/sqrt 2
    assert widths["guid"]["auroc"] < widths["patient"]["auroc"] and widths["guid"]["sens"] < widths["patient"]["sens"] / 1.2


def test_headline_counts_nan_folds():
    rows = [dict.fromkeys(M.METRIC_COLUMNS) | dict(model_id="p", seed="1", level="guid", split="test", fold=f,
                                                    metric="auroc", value=v, point="n/a")
            for f, v in (("1", 0.8), ("2", np.nan), ("pooled", 0.75))]
    h = M.headline(pd.DataFrame(rows), 0.3)["p|1|guid|threshold_free|auroc"]
    assert h["fold_mean"] == 0.8 and h["n_folds"] == 2 and h["n_folds_nan"] == 1


def test_provenance_digest_mismatch_refuses(tmp_path):
    run, cfg = make_run(tmp_path, folds=(1, 2))
    prov = json.loads((run / "predictions" / "provenance.json").read_text())
    (run / "predictions" / "provenance.json").write_text(json.dumps(prov | {"config_digest": "0" * 64}))
    with pytest.raises(ValueError, match="config digest"):
        M.evaluate(run, cfg)


def test_vertical_average_and_pooled_roc():
    """T-E6: step-read vertical average (no extrapolation) and the pooled ROC on synthetic folds."""
    from sklearn.metrics import roc_curve

    f1 = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], drop_intermediate=False)[:2]  # (0,.5) -> (.5,.5) -> (.5,1)
    f2 = roc_curve([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], drop_intermediate=False)[:2]   # perfect
    va = M.vertical_average([f1, f2], grid=np.array([0.0, 0.25, 0.5, 1.0])).set_index("fpr")
    assert va["tpr"].tolist() == pytest.approx([0.75, 0.75, 1.0, 1.0])
    assert va.loc[0.25, "tpr_sd"] == pytest.approx(np.std([0.5, 1.0], ddof=1))
    assert va.loc[0.25, ["tpr_min", "tpr_max"]].tolist() == [0.5, 1.0] and (va["n_folds"] == 2).all()
    # a diagonal (tied) segment reads as its lower-left corner, never an interpolated TPR
    fpr, tpr, _ = roc_curve([0, 1, 0, 1], [0.5, 0.5, 0.1, 0.9], drop_intermediate=False)
    assert M.tpr_at(fpr, tpr, np.array([0.25, 0.5, 1.0])).tolist() == [0.5, 1.0, 1.0]
    # pooled ROC = the ROC of the de-duplicated concatenation (shared GUID kept once, lowest fold)
    units = pd.DataFrame({"fold": [1, 1, 1, 2, 2, 2], "unit": ["a", "b", "s", "c", "d", "s"],
                          "y": [0, 1, 0, 0, 1, 0], "score": [0.1, 0.9, 0.2, 0.3, 0.8, 0.95],
                          "shared_test": [False, False, True, False, False, True]})
    first = M.pool_rows(units, "first_fold", "test")
    assert first["unit"].tolist() == ["a", "b", "s", "c", "d"] and first.query("unit == 's'")["fold"].item() == 1
    assert M.pool_rows(units, "exclude", "test")["unit"].tolist() == ["a", "b", "c", "d"]
    assert M.pool_rows(units.assign(shared_test=False), "first_fold", "val")["unit"].tolist() == ["a", "b", "s", "c", "d"]


def test_shuffled_control_flagged(tmp_path):
    """B3: a 'shuffled' control that still ranks (here: the planted-signal probe renamed) fails its sanity check."""
    run, cfg = make_run(tmp_path, folds=(1, 2))
    pred = run / "predictions"
    for name in ("segments", "guids"):
        df = pd.read_parquet(pred / f"{name}.parquet")
        pd.concat([df, df[df["model_id"] == "probe"].assign(model_id="shuffled")]).to_parquet(pred / f"{name}.parquet")
    thr = json.loads((pred / "thresholds.json").read_text())
    thr |= {k.replace("probe|", "shuffled|", 1): v for k, v in thr.items() if k.startswith("probe|")}
    (pred / "thresholds.json").write_text(json.dumps(thr))
    assert M.evaluate(run, cfg, only=["metrics", "B1"]) == 0
    results = _partial_summary(run)["results"]
    assert results["sanity"]["checks"]["shuffled_auroc"]["verdict"] == "fail"
    assert results["missing_units"] == []
