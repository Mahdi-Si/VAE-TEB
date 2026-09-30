"""P4 time-resolved evaluation: T-T3, T-T4 and the metric-type engine of metrics.py (SPEC §11.5, §15).

The hand-built 4-GUID table below is the known-answer example; ``make_run`` (test_eval_analyses) gives
the end-to-end run.
"""
from __future__ import annotations

import json
import math
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from teb_vae.classifier import metrics as M
from teb_vae.classifier import thresholds as T
from teb_vae.classifier.tests.test_eval_analyses import make_run

EV = {"checkpoints_h": [1.0], "snapshot_max_staleness_h": 1.0, "bin_h": 0.5, "min_bin_class_n": 0,
      "alarm_rule": {"kind": "latch"}, "exclude_last_min": 0}
BOOT = {"resamples": 50, "seed": 0}
SPEC = {  # guid: (y, stage, clocks in h (to_delivery), online scores); thr = 0.45
    "A": (0, "first", [-3.0, -2.0, -1.4, -1.2, -0.8], [0.1, 0.5, 0.3, 0.2, 0.9]),
    "B": (0, "second", [-0.6, -0.2], [0.3, 0.4]),
    "C": (1, "first", [-2.5, -1.4, -1.1], [0.2, 0.8, 0.6]),
    "D": (0, "first", [-5.0, -4.0], [0.7, 0.1]),
}


def _seg(spec=SPEC, fold=1):
    rows = [{"guid": g, "seg_pos": i, "t_end_s": c * 3600.0, "epoch_s": c * 3600.0 - 1260.0, "y": y, "fold": fold,
             "stage": st, "clinical_class": "hie" if y else "healthy", "logit_online_cal": s, "logit_seg_cal": s,
             "tlo_end_h": c + 8.0 if g != "B" else np.nan, "ss_rel_h": np.nan if g == "D" else c + 1.0}
            for g, (y, st, cs, ss) in spec.items() for i, (c, s) in enumerate(zip(cs, ss))]
    return pd.DataFrame(rows).sample(frac=1.0, random_state=0)  # order must not matter


def _group(seg, thr=0.45, rule=None):
    ents = {f: {"p": {"threshold": thr, "basis": "committed_overall"}} for f in seg["fold"].unique()}
    return M._prep(seg, ents, ["p"], {}, [], rule or EV["alarm_rule"])


def _rows(seg, ev=EV, axis="to_delivery", **kw):
    U = _group(seg, **kw)
    edges = M._grids([("1", U)], [axis], ev["bin_h"])[axis]
    pts = M._points(axis, edges, ev)
    V = M._view(U, axis, pts)
    frames, _ = M._mt_frames(U, V, pts, ev, alpha=0.3, boot=BOOT, rank_ci=np.zeros(len(pts), bool),
                             fold="1", axis=axis)
    return M._blocks_frame(frames), U, V, pts


def _get(df, mt, point, t, metric, stratum=None):
    q = df[(df["metric_type"] == mt) & (df["point"] == point) & (df["metric"] == metric)
           & (df["t"].isna() if t is None else np.isclose(df["t"], t))]
    q = q[q["subgroup_value"] == stratum] if stratum else q[q.get("subgroup", pd.Series(index=q.index)).isna()]
    assert len(q) == 1, q
    return q.iloc[0]


# --- T-T4: the three metric types on a hand-built 4-GUID example --------------------------------
def test_metric_types_known_answers():
    df, U, V, pts = _rows(_seg())
    # instantaneous, bin (-1.5, -1.0] (t = 1.25 h): A's snapshot is its LAST in-bin segment (-1.2, s .2):
    # raw and unlatched, so A does not fire although r_A = .5 > thr; one row per GUID (A has two in the bin)
    r = _get(df, "instantaneous", "bin", 1.25, "fpr")
    assert (r.value, r.n_neg, r.n_pos) == (0.0, 1, 1) and r.denominator == "bin_present"
    assert _get(df, "instantaneous", "bin", 1.25, "sens").value == 1.0  # C at -1.1: s .6
    # committed cumulative at the bin's right edge c* = -1 h: available A, C, D (B starts at -0.6), latched
    r = _get(df, "committed_cumulative", "bin", 1.0, "fpr")
    assert (r.value, r.n_neg, r.denominator) == (1.0, 2, "available")
    # committed overall at -1 h: all GUIDs, B not yet monitored = not alarmed
    r = _get(df, "committed_overall", "bin", 1.0, "fpr")
    assert (r.value, r.n_neg, r.denominator) == (pytest.approx(2 / 3), 3, "all")
    # the cumulative denominator grows, so it is not monotone: D alone at -4.5 h, A joins unalarmed at -2.5 h
    assert _get(df, "committed_cumulative", "bin", 4.5, "fpr").value == 1.0
    assert _get(df, "committed_cumulative", "bin", 2.5, "fpr").value == 0.5
    # committed overall is monotone and, at end, the running-max GUID decision: A .9, D .7 alarm; B .4 does not
    ov = df[(df["metric_type"] == "committed_overall") & (df["metric"] == "fp") & (df["point"] != "checkpoint")]
    ov = ov.assign(c=np.where(ov["point"] == "end", np.inf, -ov["t"])).sort_values("c")
    assert (np.diff(ov["value"]) >= 0).all() and ov["value"].iloc[-1] == 2
    runmax = pd.DataFrame(SPEC, index=["y", "st", "c", "s"]).T["s"].map(max) > 0.45
    assert _get(df, "committed_overall", "end", None, "fp").value == runmax[["A", "B", "D"]].sum()
    assert _get(df, "instantaneous", "end", None, "fp").value == 1  # last segments: A .9, B .4, D .1
    # checkpoint 1 h: snapshot within the 1 h staleness window; D's last segment (-4 h) is stale
    r = _get(df, "instantaneous", "checkpoint", 1.0, "fpr")
    assert (r.value, r.n_neg, r.n_pos) == (0.0, 1, 1)
    k = int(np.flatnonzero(pts["kind"] == "checkpoint")[0])
    assert (V.avail[:, k] & ~V.inst[:, k]).sum() == 1 and U.guid[V.avail[:, k] & ~V.inst[:, k]].tolist() == ["D"]
    # PPV/NPV only at checkpoints and end; threshold-free companions share the population
    assert set(df.loc[df["metric"] == "ppv", "point"]) == {"checkpoint", "end"}
    tf = df[(df["metric_type"] == "threshold_free") & (df["point"] == "end")]
    assert set(tf["denominator"]) == {"bin_present", "available"} and set(tf["metric"]) == {"auroc", "auprc", "pauc@0.3"}


def test_stage_strata_and_underpowered_bins():
    df, *_ = _rows(_seg())
    # stage of the snapshot / of n*: at end B's last segment is second stage, the others first
    r = _get(df, "instantaneous", "end", None, "fp", stratum="second")
    assert (r.value, r.n_neg, r.n_pos) == (0.0, 1, 0)
    assert _get(df, "committed_cumulative", "end", None, "fp", stratum="first").value == 2  # A, D
    assert not ((df["metric_type"] == "committed_overall") & (df["subgroup"] == "stage")).any()
    # min_bin_class_n = 2: the one-positive bin is NaN, marked underpowered; counts stay
    df, *_ = _rows(_seg(), ev=EV | {"min_bin_class_n": 2})
    assert math.isnan(_get(df, "instantaneous", "bin", 1.25, "sens").value)
    assert _get(df, "instantaneous", "bin", 1.25, "underpowered").value == 1.0
    assert _get(df, "instantaneous", "bin", 1.25, "tp").value == 1.0
    assert math.isnan(_get(df, "committed_overall", "end", None, "sens").value)  # 1 positive in total


def test_instantaneous_threshold_is_chosen_on_the_checkpoint_population():
    """An instantaneous policy at 1 h is selected on the population M reports at the 1 h checkpoint (the staleness
    window, thresholds.snapshot_window), not on a bin: E's last segment before -1 h ends at -1.8 h, inside (-2, -1]
    but outside the (-1.5, -1] bin, so both see negatives A and E and agree on the FPR."""
    seg = _seg(SPEC | {"E": (0, "first", [-3.0, -1.8, -0.5], [0.1, 0.9, 0.2])})
    pol = {"id": "p", "policy": "fpr_cap", "alpha": 0.3, "method": "empirical", "basis": "instantaneous", "at": 1.0}
    rec = T.select_thresholds(seg, None, [pol], bin_h=EV["bin_h"],
                              staleness_h=EV["snapshot_max_staleness_h"])["guid"]["p"]
    df, *_ = _rows(seg, thr=rec["threshold"])
    r = _get(df, "instantaneous", "checkpoint", 1.0, "fpr")
    assert (r.n_neg, r.value) == (rec["n_neg"], rec["val_fpr"]) == (2, 0.0)


def test_minus_inf_padding_gives_floor_alpha_n_all_alarms():
    """committed_overall basis at 1 h, then the engine at the 1 h checkpoint: floor(alpha * n_all) alarms."""
    rng = np.random.default_rng(0)
    spec = {f"n{i}": (0, "first", [-3.0 + 0.1 * i, -1.5], rng.normal(size=2).tolist()) for i in range(6)}
    spec |= {f"u{i}": (0, "first", [-0.6, -0.3], [9.0, 9.0]) for i in range(4)}  # monitored only after 1 h
    spec |= {"p0": (1, "first", [-2.0, -1.2], [0.0, 3.0])}
    seg = _seg(spec)
    y, s = T.basis_scores(seg, None, "committed_overall", at=1.0)
    thr = T.fpr_threshold(s[y == 0], 0.3, "empirical")["threshold"]
    df, *_ = _rows(seg, thr=thr)
    r = _get(df, "committed_overall", "checkpoint", 1.0, "fp")
    assert r.value == math.floor(0.3 * 10) and r.n_neg == 10
    assert _get(df, "committed_cumulative", "checkpoint", 1.0, "fpr").value == pytest.approx(3 / 6)


# --- T-T3: running-max latch, lead time, k_of_n ------------------------------------------------
def test_latch_is_running_max_and_lead_time_from_first_crossing():
    rng = np.random.default_rng(1)
    gi = np.repeat(np.arange(30), rng.integers(1, 9, 30))
    s = rng.normal(size=gi.size)
    r = pd.Series(s).groupby(gi).cummax().to_numpy()
    for thr in (-0.5, 0.0, 1.0):
        assert (T.latch_state(s > thr, gi) == (r > thr)).all()  # alarmed by n <=> r_g(n) > thr
    U = _group(_seg())
    A = M._alarm_arrays(U, U.latch)
    got = pd.DataFrame({k: A[k][0] for k in ("alarmed", "lead_time_h", "burden")}, index=U.guid)
    # A first crosses at -2.0 (s .5), C at -1.4, D at -5.0; B never; burden = segments at/after the alarm
    assert got["alarmed"].tolist() == [True, False, True, True]
    assert got["lead_time_h"].tolist()[0] == 2.0 and math.isnan(got.loc["B", "lead_time_h"])
    assert got.loc[["C", "D"], "lead_time_h"].tolist() == [1.4, 5.0]
    assert got["burden"].tolist() == [0.8, 0.0, pytest.approx(2 / 3), 1.0]


def test_k_of_n_windows_stay_within_the_guid():
    exceed = np.array([0, 1, 0, 1, 1, 0, 1, 1], bool)  # GUID 0: 6 rows, GUID 1: 2 rows
    gi = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    # k=2 of n=3: GUID 0 fires at row 3 ([1,0,1]) and holds; GUID 1's first row must not see GUID 0's tail
    assert T.latch_state(exceed, gi, k=2, n=3).astype(int).tolist() == [0, 0, 0, 1, 1, 1, 0, 1]
    U = _group(_seg(), rule={"kind": "k_of_n", "k": 2, "n": 2})
    A = M._alarm_arrays(U, U.la)
    # 2 consecutive exceedances: only C (-1.4 and -1.1) -> first alarm at -1.1 h
    assert A["alarmed"][0].tolist() == [False, False, True, False] and A["lead_time_h"][0][2] == 1.1


# --- clocks, exclusions, staleness, exclude_last_min ------------------------------------------
def test_axis_clocks_and_eligibility():
    seg = _seg().sort_values(["guid", "seg_pos"])
    a = seg[seg["guid"] == "A"]
    assert T.clock(a, "to_delivery").tolist() == [-3.0, -2.0, -1.4, -1.2, -0.8]
    assert T.clock(a, "from_onset").tolist() == pytest.approx([5.0, 6.0, 6.6, 6.8, 7.2])
    assert T.clock(a, "rel_second_stage").tolist() == pytest.approx([-2.0 + 0.35, -1.0 + 0.35, -0.4 + 0.35,
                                                                      -0.2 + 0.35, 0.2 + 0.35])
    assert T.clock(a, "position").tolist() == [1, 2, 3, 4, 5]
    assert T.clock(seg, "elapsed")[seg["guid"] == "A"].tolist() == pytest.approx([0.0, 1.0, 1.6, 1.8, 2.2])
    assert T.cstar(1.0) == -1.0 and T.cstar(2.0, "from_onset") == 2.0 and T.cstar("end", "position") == math.inf
    # axis-ineligible GUIDs leave every time basis, counted by reason; the policy record reports it
    f = T.basis_frame(seg, None, "committed_overall", at=6.5, axis="from_onset")
    assert sorted(f["guid"]) == ["A", "C", "D"] and f.attrs["excluded"] == {"no_tlo": 1}
    pol = {"id": "x", "policy": "fpr_cap", "alpha": 0.5, "method": "empirical", "basis": "committed_cumulative",
           "axis": "rel_second_stage", "at": 0.0}
    rec = T.select_thresholds(seg, None, [pol], bin_h=0.5)["guid"]["x"]
    assert rec["excluded"] == {"unknown_second_stage": 1} and rec["axis"] == "rel_second_stage"
    df, U, V, _ = _rows(_seg(), axis="from_onset")
    assert V.elig.tolist() == [True, False, True, True] and set(df["n_neg"].dropna()) <= {0, 1, 2}


def _ctx(seg, ev):
    seg = seg.assign(model_id="m", seed="1", split="test", shared_test=False)
    thr = {"m|1|1": {"guid": {"p": {"threshold": 0.45, "basis": "committed_overall"}}}}
    return SimpleNamespace(seg=seg, thr=thr, inclusion=[], cfg={
        "eval": ev | {"thresholds": [{"id": "p"}]}, "data": {"shared_test_policy": "first_fold"}})


def test_exclude_last_min_drops_late_segments_everywhere():
    ctx = _ctx(_seg(), EV | {"exclude_last_min": 30})
    (fold, U), (pooled, _) = M._tr_groups(ctx, "m", "1", "test")
    assert (fold, pooled) == ("1", "pooled") and U.o["t_end_s"].max() <= -1800  # B's -0.2 h segment is gone
    assert U.s[U.gi == np.flatnonzero(U.guid == "B")[0]].tolist() == [0.3]
    rec = next(r for r in ctx.inclusion if r["analysis"] == "exclude_last_min")
    assert json.loads(rec["reasons"])["segments_dropped"] == 1 and rec["n_excluded"] == 0


# --- vectorised bootstraps --------------------------------------------------------------------
def test_rank_stats_match_sklearn_and_roc_points():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(y * 0.8, 1.0), 1)  # ties
    o, st, _ = M._ranked(s)
    got = {k: v[0] for k, v in M._rank_stats(*M._curve(y, np.ones((1, y.size)), o, st), 0.3).items()}
    assert got["auroc"] == pytest.approx(roc_auc_score(y, s))
    assert got["auprc"] == pytest.approx(average_precision_score(y, s))
    assert got["pauc@0.3"] == pytest.approx(roc_auc_score(y, s, max_fpr=0.3))
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    r = M.roc_points(y, s)
    assert np.allclose(r["fpr"], fpr) and np.allclose(r["tpr"], tpr) and np.array_equal(r["thr"], thr)
    r = M.roc_points([0, 1, 0, 1], [-np.inf, 2.0, 1.0, -np.inf])  # R3: unmonitored GUIDs score -inf
    assert r["fpr"].tolist() == [0.0, 0.0, 0.5, 1.0] and r["tpr"].tolist() == [0.0, 0.5, 0.5, 1.0]


def test_bootstrap_auroc_matches_bootstrap_ci_distributionally():
    rng = np.random.default_rng(3)
    y_g = rng.integers(0, 2, 80)
    units = np.repeat(np.arange(80), 3)  # 3 rows per GUID: cluster draws
    y, s = np.repeat(y_g, 3), np.repeat(y_g * 0.7, 3) + rng.normal(size=240)
    fast = M.bootstrap_auroc(y, s, units, 2000, 7)
    slow = M.bootstrap_ci(lambda i: {"auroc": roc_auc_score(y[i], s[i])}, units, y, resamples=2000, seed=7)
    assert fast["metrics"]["auroc"]["value"] == pytest.approx(slow["metrics"]["auroc"]["value"])
    for side in ("ci_lo", "ci_hi"):
        assert fast["metrics"]["auroc"][side] == pytest.approx(slow["metrics"]["auroc"][side], abs=0.02)
    assert set(fast["record"]) == {"method", "resamples", "seed", "confidence", "n", "n_dropped"}
    assert fast["record"]["n"] == 80 and fast == M.bootstrap_auroc(y, s, units, 2000, 7)
    # GUID-level (one row per unit): the spread matches too
    g = M.bootstrap_auroc(y_g, s[::3], np.arange(80), 2000, 1)["metrics"]["auroc"]
    ref = M.bootstrap_ci(lambda i: {"a": roc_auc_score(y_g[i], s[::3][i])}, np.arange(80), y_g, resamples=2000, seed=1)
    assert g["ci_hi"] - g["ci_lo"] == pytest.approx(ref["metrics"]["a"]["ci_hi"] - ref["metrics"]["a"]["ci_lo"], abs=0.03)


def test_rate_ci_wilson_and_the_patient_cluster_fallback():
    """One decision per patient: Wilson, never zero-width at 0/n or n/n (§11.7). A patient holding two
    GUIDs of the cell: the patient-cluster bootstrap, which matches bootstrap_ci and is wider than Wilson."""
    five = [1] * 5 + [0] * 5
    lo, hi, nu = M.rate_ci(np.array([[0] * 10, five, [1] * 4 + [0] * 6], bool),
                           np.array([[1] * 10, five, [1] * 10], bool), np.zeros(10), None, resamples=100, seed=0)
    assert np.isnan(nu).all()  # Wilson: no draws
    assert lo[:2].tolist() == [0.0, M.wilson(5, 5)[0]] and hi[0] == pytest.approx(M.wilson(0, 10)[1]) and hi[0] > 0.2
    assert hi[1] == 1.0 and lo[1] < 0.6 and (lo[2], hi[2]) == pytest.approx(M.wilson(4, 10))
    assert np.isnan(M.rate_ci([[0, 0]], [[0, 0]], np.zeros(2), None, resamples=10, seed=0)[0]).all()
    rng = np.random.default_rng(4)
    hits = rng.random(40) < 0.3                 # 40 patients, each with two GUIDs sharing the decision
    num, pat = np.r_[hits, hits], np.r_[np.arange(40), np.arange(40)]
    lo, hi, nu = M.rate_ci(num[None], np.ones((1, 80), bool), np.zeros(80), pat, resamples=4000, seed=1)
    assert nu[0] == 0
    slow = M.bootstrap_ci(lambda i: {"r": num[i].mean()}, pat, np.zeros(80), resamples=4000, seed=1)["metrics"]["r"]
    assert (lo[0], hi[0]) == pytest.approx((slow["ci_lo"], slow["ci_hi"]), abs=0.03)
    w_guid = np.subtract(*M.wilson(num.sum(), 80)[::-1])
    assert hi[0] - lo[0] > 1.25 * w_guid       # GUID-level (Wilson on 80) would be ~1/sqrt 2 as wide
    # the same patients once each: Wilson; the per-row check keeps 1:1 rows on Wilson inside a clustered call
    lo, hi, _ = M.rate_ci(np.stack([num, num & (pat < 0)]), np.stack([np.ones(80, bool), np.arange(80) < 40]),
                          np.zeros(80), pat, resamples=200, seed=1)
    assert (lo[1], hi[1]) == pytest.approx(M.wilson(0, 40))


def test_rate_ci_counts_undefined_cluster_draws():
    """§11.7: a cluster-bootstrap draw with an empty denominator is counted, never silently dropped. 20 patients x 10
    segments, the policy firing on 3 segments of one positive patient: PPV = 1 is undefined whenever that patient is
    not drawn, and the count reaches the metric rows' ``n_boot_undefined``."""
    pat, y = np.repeat(np.arange(20), 10), np.repeat(np.arange(20) < 10, 10)
    fire = np.zeros(200, bool)
    fire[:3] = True  # patient 0 (positive)
    lo, hi, nu = M.rate_ci((fire & y)[None], fire[None], y, pat, resamples=2000, seed=0)
    assert (lo[0], hi[0]) == (1.0, 1.0) and 0 < nu[0] < 2000
    ci = M._prop_ci(fire, y, pat, 0.3, {"resamples": 2000, "seed": 0})
    assert ci["ppv"]["n_undefined"] == nu[0] and ci["fpr_overshoot"]["n_undefined"] == ci["fpr"]["n_undefined"]
    rows = M.metric_rows({"ppv": 1.0}, ci)
    assert rows[0]["n_boot_undefined"] == nu[0]


def test_alarm_stat_ci_clusters_on_patients():
    """A1/A4 means and medians: each GUID doubled under one patient keeps the single-GUID interval."""
    x = np.random.default_rng(0).exponential(1.0, 60)
    boot = dict(resamples=4000, seed=0)
    w = lambda r: r[2] - r[1]  # noqa: E731
    for stat in (np.mean, np.median):
        one, pat = M._stat_ci(x, stat, boot), M._stat_ci(np.r_[x, x], stat, boot, np.r_[np.arange(60), np.arange(60)])
        assert pat[0] == one[0] and w(pat) == pytest.approx(w(one), rel=0.05)
        assert w(M._stat_ci(np.r_[x, x], stat, boot)) < w(pat) / 1.2  # GUID draws would narrow it
    four = np.array([1.0, 2.0, 3.0, 4.0])  # the value takes the draws' (lower) median, never an interpolation
    assert M._stat_ci(four, np.median, boot)[0] == M._stat_ci(four, np.median, boot, np.arange(4))[0] == 2.0


def test_m7c_is_not_applicable_without_an_online_score(tmp_path):
    """A non-causal model with a segment head has no online score: _online falls back to the running max of its
    segment logits, which is not its score_final, so M7c records it as not applicable instead of failing. With no
    model left to check, M7c covered nothing: inconclusive, never a vacuous pass."""
    run, cfg = make_run(tmp_path, folds=(1, 2), resamples=5)
    pred = run / "predictions"
    seg, gd = pd.read_parquet(pred / "segments.parquet"), pd.read_parquet(pred / "guids.parquet")
    probe = seg["model_id"] == "probe"
    seg.loc[probe, ["logit_online", "logit_online_cal"]] = np.nan
    mean = seg[probe].groupby(["fold", "split", "guid"])["logit_seg_cal"].mean().rename("m")  # a pooled final score
    gd = gd.merge(mean.reset_index(), on=["fold", "split", "guid"], how="left")
    gd["score_final_cal"] = gd["m"].where(gd["model_id"] == "probe", gd["score_final_cal"])
    seg.to_parquet(pred / "segments.parquet", index=False)
    gd.drop(columns="m").to_parquet(pred / "guids.parquet", index=False)
    assert M.evaluate(run, cfg, only=["M"]) == 0
    (part,) = (run / "evaluation").glob("partial_*")
    rec = json.loads((part / "summary.json").read_text())["results"]["sanity"]["checks"]["m7_last_equals_final"]
    assert rec["verdict"] == "inconclusive" and rec["n_checked"] == 0 and rec["not_applicable"] == ["probe|42"], rec


def test_bootstrap_rank_is_bootstrap_auroc_plus_ap_and_pauc():
    rng = np.random.default_rng(5)
    y_g = rng.integers(0, 2, 60)
    y, s, units = np.repeat(y_g, 2), np.repeat(y_g * 0.8, 2) + rng.normal(size=120), np.repeat(np.arange(60), 2)
    r = M.bootstrap_rank(y, s, units, alpha=0.3, resamples=2000, seed=3)
    fast = M.bootstrap_auroc(y, s, units, 2000, 3)["metrics"]["auroc"]  # the same draws (float32 sums aside)
    assert r["metrics"]["auroc"] == {k: pytest.approx(v) for k, v in fast.items()}
    assert r["metrics"]["auroc"]["value"] == pytest.approx(roc_auc_score(y, s))
    slow = M.bootstrap_ci(lambda i: {"auprc": average_precision_score(y[i], s[i]),
                                     "pauc@0.3": roc_auc_score(y[i], s[i], max_fpr=0.3)}, units, y, resamples=2000, seed=3)
    for m in ("auprc", "pauc@0.3"):
        for side in ("ci_lo", "ci_hi"):
            assert r["metrics"][m][side] == pytest.approx(slow["metrics"][m][side], abs=0.03)
    assert r["metrics"]["auprc"]["value"] == pytest.approx(average_precision_score(y, s))
    assert r["record"]["n"] == 60 and r["record"]["n_dropped"] == 0


# --- end to end on make_run -------------------------------------------------------------------
@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    run, cfg = make_run(tmp_path_factory.mktemp("tr"), resamples=50)  # the smoke.yaml budget
    seg = pd.read_parquet(run / "predictions" / "segments.parquet")  # give it stages and a second-stage clock
    known = seg["guid"].str[1:].astype(int) % 2 == 0
    seg["ss_rel_h"] = np.where(known, seg["t_end_s"] / 3600.0 + 1.0, np.nan)
    seg["stage"] = np.where(~known, "unknown", np.where(seg["ss_rel_h"] + 0.35 <= 0, "first", "second"))
    seg.to_parquet(run / "predictions" / "segments.parquet", index=False)
    t0 = time.time()
    code = M.evaluate(run, cfg)
    return run, cfg, code, time.time() - t0, json.loads((run / "evaluation" / "summary.json").read_text())


def test_evaluate_end_to_end_with_time_resolved_analyses(evaluated):
    run, cfg, code, seconds, summary = evaluated
    assert code == 0 and summary["n_failed"] == 0, summary["failed"]
    assert seconds < 30, f"smoke-sized evaluate took {seconds:.1f} s"
    res, tables = summary["results"], run / "evaluation" / "tables"
    assert {"M", "A", "R2", "R5", "R8", "T3", "T4"} <= set(res)
    checks = res["sanity"]["checks"]
    for name in ("m7_committed_overall_monotone", "m7_end_equals_running_max", "m7_last_equals_final"):
        assert checks[name]["verdict"] == "pass" and checks[name]["n_checked"] > 0, checks[name]
    met = pd.read_parquet(tables / "metrics.parquet")
    assert list(met.columns) == list(M.METRIC_COLUMNS)
    assert set(met["point"]) == {"n/a", "checkpoint", "bin", "end"}
    on = met[met["level"] == "online"]
    ev = cfg.classifier.eval
    assert set(on["axis"]) == set(ev.time_axes)  # every axis, incl. rel_second_stage with half the GUIDs
    assert set(on["metric_type"]) == {*M.TYPES, "threshold_free"}
    # M's stage strata and R8's horizons, plus P6's S2: one time-resolved set per subgroup family
    families = set(M.subgroup_families(M.classifier_cfg(cfg)))
    assert set(on["subgroup"].dropna()) - families == {"stage", "decision_horizon"}
    assert set(on["fold"]) == {"1", "2", "3", "pooled"}
    assert set(on.loc[on["subgroup"] == "stage", "subgroup_value"]) == {"first", "second"}
    # T4: every policy x metric type x checkpoint is addressable
    cp = on[(on["point"] == "checkpoint") & on["subgroup"].isna() & (on["metric"] == "sens")]
    assert len(cp.groupby(["fold", "split", "policy_id", "metric_type", "t"])) == len(cp)
    assert set(cp["policy_id"]) == {p.id for p in ev.thresholds} and set(cp["t"]) == set(ev.checkpoints_h)
    assert res["T4"]["table"]["probe|42"]["test"]["np30"]["committed_overall"]["end"]["sens"][0] is not None
    # the P2 headline is untouched by the time-resolved rows
    assert set(res["headline"]) == {k for k in res["headline"] if k.split("|")[2] in ("guid", "segment")}
    # alarms
    al = pd.read_parquet(tables / "alarms.parquet")
    assert {"model_id", "seed", "fold", "split", "policy_id", "rule", "guid", "alarmed", "first_alarm_t_s",
            "lead_time_h", "burden"} <= set(al.columns)
    assert not al.duplicated(["model_id", "seed", "fold", "split", "policy_id", "rule", "guid"]).any()
    assert set(met.loc[met["level"] == "alarm", "metric"]) >= {"event_sens", "lead_time_median_h", "false_alarm_frac",
                                                                 "alarms_per_detection", "burden_neg_mean"}
    # R2-R5 variants, R8 ribbon, T3 perturbation, L14 inclusion
    roc = pd.read_parquet(tables / "roc_points.parquet")
    assert {"score_final", "committed_cumulative@1", "committed_overall@end", "snapshot@0.5", "segment",
            "segment:band"} <= set(roc["variant"])
    r8 = met[met["subgroup"] == "decision_horizon"]
    assert set(r8["t"]) == set(ev.decision_horizons_h) and set(r8["split"]) == {"val", "test"}
    assert (r8.query("split == 'val' and metric == 'fpr' and fold != 'pooled'")["value"] <= 0.3 + 1e-9).all()
    # rates at 0/n or n/n keep a Wilson width (one decision per GUID here): never a zero-width interval
    edge = on[on["metric"].isin(["sens", "fpr"]) & on["value"].isin([0.0, 1.0])]
    assert len(edge) and (edge["ci_hi"] - edge["ci_lo"] > 0).all()
    assert set(met.loc[met["subgroup"] == "threshold_perturbation", "subgroup_value"]) == {"-2", "-1", "+0", "+1", "+2"}
    assert res["T3"]["b"]["status"] == "off" and not (met["subgroup"] == "threshold_refit").any()  # default: no refit
    inc = pd.read_csv(tables / "inclusion.csv")
    tlo = inc.query("analysis == 'axis_eligibility' and axis == 'from_onset' and fold == 'pooled' and split == 'test'")
    assert (tlo["n_excluded"] > 0).all() and json.loads(tlo["reasons"].iloc[0]).keys() == {"no_tlo"}
    assert set(inc["analysis"]) >= {"axis_eligibility", "exclude_last_min", "pooled_dedupe", "staleness@1h", "time_resolved"}


# --- §11.7 variant b: the refit bootstrap (T3 b) ------------------------------------------------
REFIT_POLICIES = ("[{id: np30, policy: fpr_cap, alpha: 0.3, method: np_umbrella, delta: 0.05, basis: committed_overall},"
                  " {id: inst30_1h, policy: fpr_cap, alpha: 0.3, method: empirical, basis: instantaneous, at: 1.0},"
                  " {id: youden, policy: youden, basis: guid_final},"
                  " {id: sens80, policy: sens_target, beta: 0.8, basis: guid_final},"
                  " {id: p50, policy: fixed, value: 0.5, basis: guid_final}]")


def test_refit_threshold_reuses_the_selection_dispatch():
    rng = np.random.default_rng(0)
    y, s = np.r_[np.zeros(30, int), np.ones(10, int)], rng.normal(0, 1, 40)
    pol = {"id": "e", "policy": "fpr_cap", "alpha": 0.1, "method": "empirical"}
    w = np.ones(40)
    top = np.argsort(np.where(y == 0, s, -np.inf))[-5:]  # the 5 highest negatives, ascending
    assert M._refit_threshold(pol, y, s, w) == T.policy_threshold(pol, y, s)["threshold"] == s[top[-4]]  # 3 of 30 alarm
    w[top[-2:]] = 0  # a resample without the 2 highest: 28 negatives, 2 may alarm -> the 3rd highest remaining
    assert M._refit_threshold(pol, y, s, w) == s[top[-5]]
    assert np.isnan(M._refit_threshold(pol | {"method": "np_umbrella", "delta": 0.05, "alpha": 0.01}, y, s, w))


def test_refit_bootstrap_rows_and_figure(tmp_path):
    from teb_vae.classifier import report as R

    run, cfg = make_run(tmp_path, resamples=40, overrides=[
        "classifier.eval.bootstrap.refit_threshold=true", f"classifier.eval.thresholds={REFIT_POLICIES}"])
    assert M.evaluate(run, cfg) == 0
    summary = json.loads((run / "evaluation" / "summary.json").read_text())
    assert summary["results"]["T3"]["b"]["status"] == "done"
    met = pd.read_parquet(run / "evaluation" / "tables" / "metrics.parquet")
    rf = met[met["subgroup"] == "threshold_refit"]
    ids = {p.id for p in cfg.classifier.eval.thresholds}
    assert set(rf["split"]) == {"test"} and set(rf["subgroup_value"]) == {"refit"} and set(rf["level"]) == {"guid"}
    probe = rf[rf["model_id"] == "probe"]
    assert set(probe["policy_id"]) == ids and set(probe["fold"]) == {"1", "2", "3", "pooled"}
    assert set(probe.loc[probe["fold"] == "pooled", "metric"]) == {"sens", "spec", "fpr"}
    assert set(probe.loc[probe["fold"] != "pooled", "metric"]) == {"sens", "spec", "fpr", "threshold"}
    assert set(rf.loc[rf["model_id"] == "shortcut", "policy_id"]) == ids - {"inst30_1h"}  # GUID-level model

    # the point values are the fixed-threshold ones; only the intervals carry the refit noise
    keys = ["model_id", "fold", "policy_id", "metric"]
    fixed = met[met["subgroup"].isna() & (met["point"] == "n/a") & (met["split"] == "test") & (met["level"] == "guid")
                & met["metric"].isin(["sens", "fpr"])]
    both = rf[rf["metric"].isin(["sens", "fpr"])].merge(fixed, on=keys, suffixes=("", "_fixed"), validate="one_to_one")
    assert len(both) == (rf["metric"].isin(["sens", "fpr"])).sum()
    np.testing.assert_allclose(both["value"], both["value_fixed"])
    thr = rf[rf["metric"] == "threshold"]
    tj = json.loads((run / "predictions" / "thresholds.json").read_text())
    for r in thr.itertuples():
        assert r.value == pytest.approx(tj[f"{r.model_id}|{r.seed}|{r.fold}"]["guid"][r.policy_id]["threshold"])
    p50 = thr[thr["policy_id"] == "p50"]  # ponytail: a fixed policy keeps the fold's calibration: no threshold noise
    assert (p50["ci_lo"] == p50["value"]).all() and (p50["ci_hi"] == p50["value"]).all()
    yo = thr[thr["policy_id"] == "youden"]
    assert (yo["ci_hi"] > yo["ci_lo"]).any()  # re-selection on resampled val moves a rank policy
    rates = rf[rf["metric"] != "threshold"]
    assert (rates["ci_lo"] <= rates["ci_hi"]).all() and (rates["n_boot_undefined"] >= 0).all()

    import matplotlib.pyplot as plt

    fig = R._threshold_stability(R.load_tables(run), M.classifier_cfg(cfg))
    assert fig.axes[1].lines and fig.axes[1].get_title().endswith("(b) refit-bootstrap 95% interval per fold")
    plt.close(fig)


def test_valid_configs_without_to_delivery_or_subgroup_families_evaluate(tmp_path):
    """``eval.time_axes`` without ``to_delivery`` (T4's checkpoints live there) and an empty ``eval.subgroup_families``
    are valid configs: evaluate runs with exit 0, T4 and S1 recorded as not applicable instead of raising."""
    run, cfg = make_run(tmp_path, folds=(1, 2), resamples=5, overrides=[
        "classifier.eval.time_axes=[position]", "classifier.eval.subgroup_families=[]"])
    assert M.evaluate(run, cfg) == 0
    res = json.loads((run / "evaluation" / "summary.json").read_text())["results"]
    assert res["T4"]["plan"]["status"].startswith("not applicable") and res["T4"]["table"] == {}
    assert res["S1"]["plan"]["status"].startswith("not applicable")
