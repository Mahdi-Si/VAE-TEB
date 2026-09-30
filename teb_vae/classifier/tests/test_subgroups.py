"""T-E5: block S subgroups (SPEC §11.6, §11.12 S1-S7): membership known answers for every §11.6.1 family, the metric a
population reports (healthy-only: specificity, unhealthy-only: sensitivity), the power guard, S2 against the M engine,
and the S6 test pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from teb_vae.classifier import metrics as M
from teb_vae.classifier.tests.test_eval_analyses import make_run


def _toy():
    """Four GUIDs, hand-built: three in fold 1 test (healthy a, acidosis b, hie c), one healthy in fold 1 val (d)."""
    g = pd.DataFrame({
        "fold": 1, "split": ["test", "test", "test", "val"], "guid": list("abcd"),
        "clinical_class": ["healthy", "acidosis", "hie", "healthy"], "class_code": [1, 2, 3, 1], "y": [0, 1, 1, 0],
        "subgroup": ["healthy_bg_cs", "acidosis_no_cs", "hie_cs", "healthy_no_bg_no_cs"],
        "cs": [True, False, True, False], "bg": [True, True, True, False], "has_tlo": [True, False, True, True],
        "shared_test": [True, False, False, False], "n_segments": [3, 5, 8, 2],
        "first_epoch_s": [-9000.0, -7200.0, -14400.0, -3000.0], "last_t_end_s": [-600.0, -5000.0, -60.0, -700.0]})
    rows = [("a", ["first", "straddle", "second"], [1.0, 2.0, 3.0], [2.0, 1.0, 0.2], 0.9),
            ("b", ["unknown"] * 5, [np.nan] * 5, [3.0, 2.5, 2.0, 1.8, 1.4], 0.5),
            ("c", ["first"] * 8, [5.0 + i for i in range(8)], [4.0 - 0.5 * i for i in range(8)], 1.0),
            ("d", ["first", "first"], [0.5, 0.7], [0.4, 0.2], 0.95)]
    seg = pd.concat([pd.DataFrame({"fold": 1, "split": "val" if x == "d" else "test", "guid": x, "seg_pos": range(len(st)),
                                   "stage": st, "tlo_end_h": tlo, "hours_to_delivery": htd, "valid_frac": vf})
                     for x, st, tlo, htd, vf in rows], ignore_index=True)
    cov = pd.DataFrame({"fold": 1, "split": "test", "guid": ["a", "b"], "variable": "parity", "available": [True, False]})
    return g, seg, cov


def test_membership_known_answers():
    g, seg, cov = _toy()
    fams = [*M.SUBGROUP_FAMILIES, "covariate:parity"]
    mem, cut = M.subgroup_members(g, seg, fams, covariates=cov)
    got = {(f, x): sorted(v) for (f, x), v in mem.groupby(["subgroup", "guid"])["subgroup_value"]}
    want = {
        "class": {"a": ["healthy"], "b": ["acidosis", "unhealthy"], "c": ["hie", "unhealthy"], "d": ["healthy"]},
        "source_file": {"a": ["healthy_bg_cs"], "b": ["acidosis_no_cs"], "c": ["hie_cs"], "d": ["healthy_no_bg_no_cs"]},
        "class_x_cs": {"a": ["healthy_cs_pos"], "b": ["acidosis_cs_neg", "unhealthy_cs_neg"],
                       "c": ["hie_cs_pos", "unhealthy_cs_pos"], "d": ["healthy_cs_neg"]},
        "healthy_x_bg": {"a": ["healthy_bg_pos"], "d": ["healthy_bg_neg"]},
        "healthy_bg_x_cs": {"a": ["healthy_bg_pos_cs_pos"], "d": ["healthy_bg_neg_cs_neg"]},
        "cs": {"a": ["cs_pos"], "b": ["cs_neg"], "c": ["cs_pos"], "d": ["cs_neg"]},
        "bg": {"a": ["bg_pos"], "b": ["bg_pos"], "c": ["bg_pos"], "d": ["bg_neg"]},
        "stage_last": {"a": ["second"], "b": ["unknown"], "c": ["first"], "d": ["first"]},
        "reached_second_stage": {"a": ["yes"], "b": ["unknown"], "c": ["no"], "d": ["no"]},
        "has_tlo": {"a": ["yes"], "b": ["no"], "c": ["yes"], "d": ["yes"]},
        # test values 1 | NaN | 5 -> cuts on (1, 5); d's 0.5 (val) reuses them
        "admission_tlo_tertile": {"a": ["T1"], "b": ["unknown"], "c": ["T3"], "d": ["T1"]},
        "labour_duration_tertile": {"a": ["T1"], "b": ["unknown"], "c": ["T3"], "d": ["T1"]},
        "n_segments_tertile": {"a": ["T1"], "b": ["T2"], "c": ["T3"], "d": ["T1"]},  # cuts 4.33, 6 on 3 | 5 | 8
        "span_tertile": {"a": ["T2"], "b": ["T1"], "c": ["T3"], "d": ["T1"]},  # 2.33 | 0.61 | 3.98 h
        "valid_frac_tertile": {"a": ["T2"], "b": ["T1"], "c": ["T3"], "d": ["T3"]},  # cuts 0.77, 0.93
        "late_coverage": {"a": ["yes"], "b": ["no"], "c": ["yes"], "d": ["yes"]},
        "shared_test": {"a": ["yes"], "b": ["no"], "c": ["no"], "d": ["no"]},
        "fold": {x: ["1"] for x in "abcd"},
        "covariate:parity": {"a": ["available"], "b": ["missing"]},
    }
    assert set(want) == set(fams)
    for f, per in want.items():
        assert {x: v for (ff, x), v in got.items() if ff == f} == per, f
    assert cut["n_segments_tertile"] == pytest.approx(np.quantile([3, 5, 8], [1 / 3, 2 / 3]))
    assert cut["admission_tlo_tertile"] == pytest.approx(np.quantile([1.0, 5.0], [1 / 3, 2 / 3]))
    # reused cut-points override the pooled-test ones
    assert set(M.subgroup_members(g, seg, ["n_segments_tertile"], cutpoints={"n_segments_tertile": [0.0, 1.0]})[0]
               ["subgroup_value"]) == {"T3"}
    # ordered by family, members worst first
    assert list(dict.fromkeys(mem["subgroup"])) == fams
    assert list(dict.fromkeys(mem.loc[mem["subgroup"] == "class", "subgroup_value"])) == ["hie", "acidosis", "unhealthy", "healthy"]
    # documented-empty families are never computed: the family list drops them, a direct request is refused
    cfg = {"eval": {"subgroup_families": ["class", "acidosis_x_bg", "hie_x_bg"]},
           "context": {"covariates": {"variables": [{"name": "parity"}]}}}
    assert M.subgroup_families(cfg) == ["class", "covariate:parity"]
    with pytest.raises(ValueError, match="acidosis_x_bg"):
        M.subgroup_members(g, seg, ["acidosis_x_bg"])


def test_member_order_is_worst_first():
    from teb_vae.lag_attn.eval.labels import CANONICAL_SUBGROUPS, ordered_groups

    assert M.member_order(["healthy", "hie", "acidosis"]) == ordered_groups(["healthy", "hie", "acidosis"], "clinical_class")
    assert M.member_order(CANONICAL_SUBGROUPS) == ordered_groups(CANONICAL_SUBGROUPS, "subgroup")
    assert M.member_order(["T3", "unknown", "T1", "T2"]) == ["T1", "T2", "T3", "unknown"]
    assert M.member_order(["10", "2", "1"]) == ["1", "2", "10"]


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    run, cfg = make_run(tmp_path_factory.mktemp("subgroups"), folds=(1, 2), resamples=20)
    ctx = M.load_context(run, cfg)
    (run / "evaluation" / "tables").mkdir(parents=True)
    return ctx


def _run(ctx, name, **ev):
    out = ctx.run_dir / "evaluation"
    return M.ANALYSES[name](ctx, eval_config=ctx.cfg["eval"] | ev, out_dir=out)


def test_population_decides_the_reported_metric_and_matches_the_whole(ctx):
    _run(ctx, "S1")
    s = pd.read_parquet(ctx.run_dir / "evaluation" / "tables" / "subgroups.parquet")
    rates = s[s["policy_id"].notna() & s["metric"].isin(["sens", "spec", "fpr", "tp", "fp", "tn", "fn", "ppv", "npv"])]
    by_pop = rates.groupby("population")["metric"].agg(set)
    assert by_pop["healthy_only"] == {"tn", "fp", "spec", "fpr"} and by_pop["unhealthy_only"] == {"tp", "fn", "sens"}
    assert by_pop["mixed"] == {"tp", "fp", "tn", "fn", "sens", "spec", "fpr", "ppv", "npv"}
    cls = s[s["subgroup"] == "class"].groupby("subgroup_value")["population"].first()
    assert cls.to_dict() == {"healthy": "healthy_only", "acidosis": "unhealthy_only", "hie": "unhealthy_only",
                             "unhealthy": "unhealthy_only"}
    assert s.loc[(s["population"] != "mixed") & (s["metric_type"] == "threshold_free"), "metric"].empty
    # the unhealthy member's sensitivity and the healthy member's specificity are the whole population's (S1 = P2)
    ents, pop = M.policy_populations(ctx, "probe", "42", "test")[("guid", "np30")]
    whole = M.pooled_confusion(pop.assign(guid=pop["unit"]), {f: e["threshold"] for f, e in ents.items()},
                               ctx.cfg["data"]["shared_test_policy"], score="score")
    q = s.query("model_id == 'probe' and split == 'test' and fold == 'pooled' and policy_id == 'np30' and subgroup == 'class'")
    v = q.set_index(["subgroup_value", "metric"])["value_raw"]
    assert v["unhealthy", "sens"] == pytest.approx(whole["sens"]) and v["healthy", "spec"] == pytest.approx(whole["spec"])
    assert v["unhealthy", "tp"] + v["healthy", "fp"] == whole["tp"] + whole["fp"]
    assert "shuffled" not in set(s["model_id"])  # the label-permutation control has no subgroups


def test_underpowered_flag(ctx):
    for n, expect in ((10_000, True), (0, False)):
        ctx.__dict__.pop("s_pops", None)
        _run(ctx, "S1", min_subgroup_n=n)
        s = pd.read_parquet(ctx.run_dir / "evaluation" / "tables" / "subgroups.parquet")
        r = s[s["metric"].isin(["sens", "spec", "auroc"]) & s["value_raw"].notna()]
        assert len(r) and (r["underpowered"] == expect).all()
        assert r["value"].isna().all() == expect and r["ci_lo"].isna().all() == expect
    s1 = s  # min_subgroup_n = 0: every powered mixed cell carries AUROC; counts are never guarded
    assert not s1.loc[s1["metric"] == "n_guids", "underpowered"].any()


def test_s2_class_rows_equal_the_whole_population_rows(ctx):
    m = _run(ctx, "M")["frame"]
    s2 = _run(ctx, "S2", min_subgroup_n=5)["frame"]  # M's min_bin_class_n
    assert set(s2.loc[s2["fold"] != "pooled", "subgroup"]) == {"class"}  # per fold: the core-set family only
    key = ["split", "fold", "axis", "metric_type", "t", "point"]
    whole = m[m["subgroup"].isna() & (m["model_id"] == "probe") & (m["policy_id"] == "np30") & (m["level"] == "online")]
    for member, metric, n in (("unhealthy", "tp", "n_pos"), ("healthy", "fp", "n_neg")):
        a = whole[whole["metric"] == metric]
        b = s2[(s2["subgroup"] == "class") & (s2["subgroup_value"] == member) & (s2["metric"] == metric)]
        j = a.merge(b, on=key, suffixes=("", "_s"))
        assert len(j) == len(b) > 100 and (j["value"] == j["value_s"]).all() and (j[n] == j[f"{n}_s"]).all()


def test_s6_family_tests(ctx):
    res = _run(ctx, "S6")
    t = pd.read_parquet(ctx.run_dir / "evaluation" / "tables" / "subgroup_tests.parquet")
    om = t[(t["kind"] == "omnibus") & (t["model_id"] == "probe")]
    assert res["n_omnibus"] == len(t[t["kind"] == "omnibus"])
    # Holm across every omnibus test of the model
    from teb_vae.lag_attn.eval.stats import holm_adjust

    assert om["p_holm"].tolist() == pytest.approx(holm_adjust(om["p_value"].tolist()), nan_ok=True)
    cls = om[om["subgroup"] == "class"].iloc[0]
    assert cls["stratum"] == "all" and cls["significant"] and cls["n_groups"] == 3  # planted signal separates classes
    pw = t[(t["kind"] == "pairwise") & (t["model_id"] == "probe") & (t["subgroup"] == "class")]
    assert [tuple(x) for x in pw[["left", "right"]].to_numpy()] == [("hie", "acidosis"), ("hie", "healthy"),
                                                                    ("acidosis", "healthy")]
    assert (pw.set_index("right").loc["healthy", "cliffs_delta"] > 0).all()  # severe first: delta > 0
    assert pw["p_holm"].tolist() == pytest.approx(holm_adjust(pw["p_value"].tolist()))
    # a group below MIN_GROUP_SIZE is excluded and recorded: the single shared test GUID
    st = om[om["subgroup"] == "shared_test"]
    assert any('"yes": 1' in e for e in st["excluded"])
    assert (t["alpha"] == M.S_ALPHA).all() and M.S_ALPHA == 0.05


def test_s2_power_guard_voids_only_the_thin_class(ctx):
    """A mixed member whose adverse GUIDs are below ``min_subgroup_n`` while its healthy ones are not keeps its FPR and
    specificity (with CIs); only its sensitivity is NaN. The guard used to void both classes' rates."""
    mn = 8
    s2 = _run(ctx, "S2", min_subgroup_n=mn)["frame"]
    key = ["model_id", "seed", "split", "fold", "subgroup", "subgroup_value", "policy_id", "axis", "metric_type", "t",
           "point"]
    wide = s2[s2["metric"].isin(["sens", "fpr", "spec"])].pivot_table(index=key, columns="metric", values="value",
                                                                        dropna=False)
    counts = s2[s2["metric"] == "fpr"].set_index(key)[["n_pos", "n_neg"]]
    thin = wide.join(counts, how="inner").query("n_pos < @mn and n_neg >= @mn and n_pos > 0")
    assert len(thin) > 10
    assert thin["sens"].isna().all() and thin["fpr"].notna().all() and np.allclose(thin["spec"], 1 - thin["fpr"])
