"""T-T1, T-T2 and the basis builders of thresholds.py (SPEC §11.3, §15)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import comb
from scipy.stats import norm

from teb_vae.classifier import thresholds as T


# --- T-T1: empirical FPR cap -------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_empirical_cap_is_valid_and_maximally_sensitive(seed):
    rng = np.random.default_rng(seed)
    neg = np.round(rng.normal(size=37), 1)  # rounding forces ties
    pos = np.round(rng.normal(1.0, size=23), 1)
    alpha = 0.3
    res = T.fpr_threshold(neg, alpha, "empirical")
    thr = res["threshold"]
    assert np.mean(neg > thr) <= alpha
    # brute force over every split: no valid threshold alarms more positives
    best = max(np.sum(pos > c) for c in np.r_[-np.inf, neg, pos] if np.mean(neg > c) <= alpha)
    assert np.sum(pos > thr) == best
    assert res["tie_frac"] == pytest.approx(np.mean(neg == thr))


def test_empirical_strict_rule_with_ties():
    neg = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], float)  # alpha*n0 = 3 alarms allowed
    res = T.fpr_threshold(neg, 0.3, "empirical")
    assert res["threshold"] == 1.0  # > 1 alarms the three 2s; > 0 would alarm 7
    assert res["tie_frac"] == 0.4
    assert T.apply_threshold(neg, res["threshold"]).sum() == 3
    # all tied: the only valid threshold alarms nobody
    assert T.fpr_threshold(np.ones(10), 0.3, "empirical")["threshold"] == 1.0


def test_fpr_threshold_raises_instead_of_falling_back():
    with pytest.raises(ValueError):
        T.fpr_threshold([0.1, np.nan, 0.3], 0.3, "empirical")
    with pytest.raises(ValueError):
        T.fpr_threshold([], 0.3, "empirical")
    with pytest.raises(ValueError):
        T.fpr_threshold([0.1, 0.2], 0.3, "bogus")
    with pytest.raises(ValueError):
        T.apply_threshold([0.0, np.nan], 0.5)


# --- T-T2: NP umbrella -------------------------------------------------------------------------
def _brute_k(n, alpha, delta):
    for k in range(1, n + 1):
        tail = sum(comb(n, j, exact=True) * (1 - alpha) ** j * alpha ** (n - j) for j in range(k, n + 1))
        if tail <= delta:
            return k
    return None


@pytest.mark.parametrize("alpha", [0.05, 0.15, 0.3])
@pytest.mark.parametrize("delta", [0.01, 0.05, 0.1])
@pytest.mark.parametrize("n", [9, 20, 59, 100, 250])
def test_np_order_k_matches_brute_force(n, alpha, delta):
    n_min = math.ceil(math.log(delta) / math.log(1 - alpha))
    if n < n_min:
        with pytest.raises(ValueError):
            T.np_order_k(n, alpha, delta)
    else:
        assert T.np_order_k(n, alpha, delta) == _brute_k(n, alpha, delta)


def test_np_minimum_n_and_fallback():
    assert T.np_order_k(9, 0.3, 0.05) == 9  # SPEC: n_min = 9 for alpha 0.3, delta 0.05
    with pytest.raises(ValueError):
        T.np_order_k(8, 0.3, 0.05)
    neg = np.arange(8.0)
    with pytest.raises(ValueError):
        T.fpr_threshold(neg, 0.3, "np_umbrella")
    res = T.fpr_threshold(neg, 0.3, "np_umbrella", allow_fallback=True)
    assert res["fallback"] and res["method"] == "empirical" and res["k"] is None


def test_np_population_fpr_guarantee_monte_carlo():
    n0, alpha, delta, reps = 60, 0.3, 0.05, 4000
    k = T.np_order_k(n0, alpha, delta)
    v = np.sort(np.random.default_rng(0).normal(size=(reps, n0)), axis=1)
    assert T.fpr_threshold(v[0], alpha, "np_umbrella")["threshold"] == v[0, k - 1]
    pop_fpr = norm.sf(v[:, k - 1])  # N(0,1) negatives: P(score > thr)
    assert np.mean(pop_fpr > alpha) <= delta + 0.015
    # the empirical rule overshoots far more often (Tong 2018: ~50%)
    emp = norm.sf(v[:, n0 - math.floor(alpha * n0) - 1])
    assert np.mean(emp > alpha) > 0.3


# --- other policies ----------------------------------------------------------------------------
def test_youden_ties_to_lower_threshold_and_sens_target():
    y = np.array([0, 0, 1, 0, 1, 1])
    s = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    # J: thr=1 -> 1-1/3=2/3 ; thr=3 -> 2/3-0 = 2/3 ; tie -> lower
    res = T.youden(y, s)
    assert res["threshold"] == 1.0
    assert T.sens_target(y, s, 1.0)["threshold"] == 1.0
    assert T.sens_target(y, s, 0.6)["threshold"] == 3.0
    assert T.fixed(0.5)["threshold"] == 0.0
    with pytest.raises(ValueError):
        T.youden(np.zeros(3), np.arange(3.0))


# --- basis builders on a hand-built 4-GUID table ----------------------------------------------
def _table(online=True):
    spec = {  # guid: (y, clocks in h (to_delivery), online scores)
        "A": (0, [-3.0, -2.0, -1.4, -1.2, -0.8], [0.1, 0.5, 0.3, 0.2, 0.9]),
        "B": (0, [-0.6, -0.2], [0.3, 0.4]),
        "C": (1, [-2.5, -1.4, -1.1], [0.2, 0.8, 0.6]),
        "D": (0, [-5.0, -4.0], [0.7, 0.1]),
    }
    rows = [
        {"guid": g, "seg_pos": i, "t_end_s": c * 3600.0, "y": y, "in_eval_window": c > -1.0,
         "logit_online_cal": s if online else np.nan, "logit_seg_cal": s}
        for g, (y, cs, ss) in spec.items() for i, (c, s) in enumerate(zip(cs, ss))
    ]
    seg = pd.DataFrame(rows).sample(frac=1.0, random_state=0)  # order must not matter
    guids = pd.DataFrame({"guid": list(spec), "y": [v[0] for v in spec.values()],
                          "score_final_cal": [0.9, 0.4, 0.6, 0.1]})
    return seg, guids


def _by_class(y, s):
    return sorted(s[y == 0].tolist()), sorted(s[y == 1].tolist())


def test_basis_committed_and_instantaneous_known_answers():
    seg, guids = _table()
    # c* = -1 h: available = A, C, D; r at n*: A .5, C .8, D .7; B not yet monitored
    assert _by_class(*T.basis_scores(seg, guids, "committed_cumulative", at=1.0)) == ([0.5, 0.7], [0.8])
    assert _by_class(*T.basis_scores(seg, guids, "committed_overall", at=1.0)) == ([-np.inf, 0.5, 0.7], [0.8])
    # checkpoint window (-2.0, -1.0] (staleness 1 h): A's last segment in it is -1.2 (0.2, not 0.3); C's is -1.1
    # (0.6); D's last (-4 h) is stale
    assert _by_class(*T.basis_scores(seg, guids, "instantaneous", at=1.0, staleness_h=1.0)) == ([0.2], [0.6])
    with pytest.raises(ValueError, match="staleness"):  # a to_delivery point needs the evaluation's window
        T.basis_scores(seg, guids, "instantaneous", at=1.0)
    # end: running max over the whole recording; instantaneous = last segment
    assert _by_class(*T.basis_scores(seg, guids, "committed_overall")) == ([0.4, 0.7, 0.9], [0.8])
    assert _by_class(*T.basis_scores(seg, guids, "instantaneous")) == ([0.1, 0.4, 0.9], [0.6])
    assert _by_class(*T.basis_scores(seg, guids, "guid_final")) == ([0.1, 0.4, 0.9], [0.6])
    assert _by_class(*T.basis_scores(seg, guids, "segment")) == ([0.3, 0.4, 0.9], [])


def test_basis_falls_back_to_running_max_of_segment_scores():
    seg, guids = _table(online=False)
    # instantaneous snapshot is now the running max: A at -1.2 -> max(.1,.5,.3,.2) = .5
    assert _by_class(*T.basis_scores(seg, guids, "instantaneous", at=1.0, staleness_h=1.0)) == ([0.5], [0.8])
    assert _by_class(*T.basis_scores(seg, guids, "committed_overall", at=1.0)) == ([-np.inf, 0.5, 0.7], [0.8])
    seg.loc[seg.index[0], "logit_online_cal"] = 1.0  # partial online score is a bug, not a fallback
    with pytest.raises(ValueError):
        T.basis_scores(seg, guids, "committed_overall")


def test_minus_inf_padding_allows_floor_alpha_n_all_alarms():
    for n_monitored, expected in [(12, 6), (4, 4)]:
        neg = np.r_[np.random.default_rng(1).normal(size=n_monitored), np.full(20 - n_monitored, -np.inf)]
        thr = T.fpr_threshold(neg, 0.3, "empirical")["threshold"]
        assert (neg > thr).sum() == expected  # floor(0.3 * 20) = 6, capped by the monitored count
    seg, guids = _table()
    y, s = T.basis_scores(seg, guids, "committed_overall", at=1.0)
    thr = T.fpr_threshold(s[y == 0], 0.34, "empirical")["threshold"]
    assert (s[y == 0] > thr).sum() == math.floor(0.34 * 3)


def test_select_thresholds_contract_shape():
    rng = np.random.default_rng(3)
    rows, grows = [], []
    for g in range(60):
        y = int(g < 25)
        n = int(rng.integers(3, 9))
        clocks = -np.sort(rng.uniform(0.1, 6.0, n))[::-1]
        on = np.cumsum(rng.normal(0.3 * y, 0.5, n))
        rows += [{"guid": f"g{g}", "seg_pos": i, "t_end_s": c * 3600, "y": y, "in_eval_window": True,
                  "logit_online_cal": o, "logit_seg_cal": o} for i, (c, o) in enumerate(zip(clocks, on))]
        grows.append({"guid": f"g{g}", "y": y, "score_final_cal": on[-1]})
    seg, guids = pd.DataFrame(rows), pd.DataFrame(grows)
    policies = [
        {"id": "np30", "policy": "fpr_cap", "alpha": 0.3, "method": "np_umbrella", "delta": 0.05,
         "basis": "committed_overall", "axis": "to_delivery", "at": "end"},
        {"id": "emp30", "policy": "fpr_cap", "alpha": 0.3, "method": "empirical", "basis": "committed_overall", "at": "end"},
        {"id": "inst30_1h", "policy": "fpr_cap", "alpha": 0.3, "method": "empirical", "basis": "instantaneous", "at": 1.0},
        {"id": "cum30_1h", "policy": "fpr_cap", "alpha": 0.3, "method": "empirical", "basis": "committed_cumulative", "at": 1.0},
        {"id": "youden", "policy": "youden", "basis": "guid_final"},
        {"id": "s80", "policy": "sens_target", "beta": 0.8, "basis": "guid_final"},
        {"id": "p50", "policy": "fixed", "value": 0.5, "basis": "guid_final"},
        {"id": "seg30", "policy": "fpr_cap", "alpha": 0.3, "method": "empirical", "basis": "segment"},
    ]
    out = T.select_thresholds(seg, guids, policies, bin_h=0.5, staleness_h=1.0)
    keys = {"threshold", "basis", "axis", "at", "alpha", "delta", "method", "k", "n_pos", "n_neg",
            "val_sens", "val_fpr", "val_spec", "tie_frac", "fallback"}
    assert set(out) == {"guid", "segment"} and set(out["segment"]) == {"seg30"}
    for level in out.values():
        for rec in level.values():
            assert set(rec) == keys
    g = out["guid"]
    assert g["np30"]["k"] == T.np_order_k(35, 0.3, 0.05) and g["np30"]["n_neg"] == 35
    assert g["np30"]["threshold"] >= g["emp30"]["threshold"]
    for pid in ("np30", "emp30", "inst30_1h", "cum30_1h"):
        assert g[pid]["val_fpr"] <= 0.3
        assert g[pid]["val_spec"] == pytest.approx(1 - g[pid]["val_fpr"])
    assert g["s80"]["val_sens"] >= 0.8
    assert g["p50"]["threshold"] == 0.0
    with pytest.raises(ValueError):  # NP is void on correlated segments
        T.select_thresholds(seg, guids, [policies[-1] | {"method": "np_umbrella"}], bin_h=0.5)
    with pytest.raises(ValueError):  # too few negatives for NP, no fallback
        T.select_thresholds(seg[seg.guid.isin([f"g{i}" for i in range(30)])],
                            guids[guids.guid.isin([f"g{i}" for i in range(30)])], policies[:1], bin_h=0.5)


def test_ovr_thresholds_per_class_known_answers():
    """T5: class k's thresholds are the binary ones on logit p_c<k>_cal against class_code - 1 == k; a policy that
    cannot be computed raises."""
    from scipy.special import logit

    rng = np.random.default_rng(4)
    rows, grows = [], []
    for g in range(45):
        code, n = 1 + g % 3, int(rng.integers(2, 6))
        P = rng.dirichlet([1, 1, 1], n)
        P[:, code - 1] += np.linspace(0.0, 1.0, n)  # the true class grows towards delivery
        P /= P.sum(1, keepdims=True)
        clocks = -np.sort(rng.uniform(0.1, 4.0, n))[::-1]
        rows += [{"guid": f"g{g}", "seg_pos": i, "t_end_s": c * 3600, "class_code": code, "y": int(code > 1),
                  "in_eval_window": True, "logit_online_cal": 0.0, "logit_seg_cal": 0.0,
                  **{f"p_c{k}_cal": p[k] for k in range(3)}} for i, (c, p) in enumerate(zip(clocks, P))]
        grows.append({"guid": f"g{g}", "class_code": code, "y": int(code > 1), "score_final_cal": 0.0,
                      **{f"p_c{k}_cal": P[-1, k] for k in range(3)}})
    seg, guids = pd.DataFrame(rows), pd.DataFrame(grows)
    policies = [{"id": "emp30", "policy": "fpr_cap", "alpha": 0.3, "method": "empirical",
                 "basis": "committed_overall", "at": "end"},
                {"id": "youden", "policy": "youden", "basis": "guid_final"}]
    out = T.ovr_thresholds(seg, guids, policies, bin_h=0.5)
    assert set(out) == {"0", "1", "2"}
    for k in range(3):
        y = (guids["class_code"] - 1 == k).to_numpy()
        e = out[str(k)]["guid"]
        assert (e["emp30"]["n_pos"], e["emp30"]["n_neg"]) == (15, 30) and e["emp30"]["val_fpr"] <= 0.3
        want = T.youden(y.astype(int), logit(guids[f"p_c{k}_cal"]))["threshold"]
        assert e["youden"]["threshold"] == pytest.approx(want)
        r = seg.assign(s=logit(seg[f"p_c{k}_cal"])).groupby("guid")["s"].max().loc[guids["guid"]].to_numpy()
        thr = T.fpr_threshold(r[~y], 0.3, "empirical")["threshold"]  # committed_overall at end = running max
        assert e["emp30"]["threshold"] == pytest.approx(thr)
    np30 = [{"id": "np30", "policy": "fpr_cap", "alpha": 0.3, "method": "np_umbrella", "delta": 0.05,
             "basis": "committed_overall", "at": "end"}]
    few = guids["guid"].isin([f"g{i}" for i in range(12)])  # class 0 vs rest: 8 negatives < 9 for NP
    with pytest.raises(ValueError, match="NP umbrella"):
        T.ovr_thresholds(seg[seg["guid"].isin(guids.loc[few, "guid"])], guids[few], np30, bin_h=0.5)


def test_sens_target_with_unmonitored_positives():
    """A positive scored -inf (not yet monitored on a committed_overall basis) never alarms: a reachable beta still
    selects; an unreachable one raises a named error (not numpy's), or with allow_fallback takes the most sensitive
    threshold, flagged."""
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.0, 1.0, 2.0, -np.inf, -np.inf, 1.5, 3.0])  # at most 2 of 4 positives can alarm
    assert T.sens_target(y, s, 0.5)["threshold"] == 1.0 and not T.sens_target(y, s, 0.5)["fallback"]
    with pytest.raises(ValueError, match="unreachable"):
        T.sens_target(y, s, 0.75)
    fb = T.sens_target(y, s, 0.75, allow_fallback=True)
    assert fb["fallback"] and fb["threshold"] == 1.0  # TPR 0.5, the maximum, at the smallest FPR
    pol = {"id": "s75", "policy": "sens_target", "beta": 0.75, "allow_fallback": True}
    assert T.policy_threshold(pol, y, s)["fallback"]


def _non_causal_ovr_rows(rng):
    """Prediction rows of a non-causal 3-class sequence model with a segment head: ``logit_online_cal`` NaN (its
    positions saw later segments), ``logit_seg_cal`` finite, and segment-row ``p_c<k>_cal`` from the non-causal
    position head."""
    rows, grows = [], []
    for g in range(45):
        code, n = 1 + g % 3, int(rng.integers(3, 7))
        P = rng.dirichlet([1, 1, 1], n)
        clocks = -np.sort(rng.uniform(0.1, 4.0, n))[::-1]
        rows += [{"guid": f"g{g}", "seg_pos": i, "t_end_s": c * 3600, "class_code": code, "y": int(code > 1),
                  "in_eval_window": True, "logit_online_cal": np.nan, "logit_seg_cal": 0.0,
                  **{f"p_c{k}_cal": p[k] for k in range(3)}} for i, (c, p) in enumerate(zip(clocks, P))]
        grows.append({"guid": f"g{g}", "class_code": code, "y": int(code > 1), "score_final_cal": 0.0,
                      **{f"p_c{k}_cal": P[-1, k] for k in range(3)}})
    return pd.DataFrame(rows), pd.DataFrame(grows)


def test_ovr_thresholds_of_a_non_causal_model_never_read_segment_probabilities():
    """Top-15 #5: a non-causal model's segment-row class probabilities have seen later segments, so its OvR thresholds
    are GUID-level only (the time-resolved policies recorded as skipped), and perturbing any segment's probabilities, as
    a later segment would through the non-causal aggregator, leaves every OvR threshold unchanged."""
    seg, guids = _non_causal_ovr_rows(np.random.default_rng(7))
    policies = [{"id": "emp30", "policy": "fpr_cap", "alpha": 0.3, "method": "empirical",
                 "basis": "committed_overall", "at": "end"},
                {"id": "inst30_1h", "policy": "fpr_cap", "alpha": 0.3, "method": "empirical",
                 "basis": "instantaneous", "at": 1.0},
                {"id": "youden", "policy": "youden", "basis": "guid_final"}]
    out = T.ovr_thresholds(seg, guids, policies, bin_h=0.5, staleness_h=1.0)
    early = seg["t_end_s"] <= -3600  # rows at or before c* = -1 h
    moved = seg.assign(**{f"p_c{k}_cal": np.where(early, 1 / 3, seg[f"p_c{k}_cal"]) for k in range(3)})
    assert T.ovr_thresholds(moved, guids, policies, bin_h=0.5, staleness_h=1.0) == out
    for k in range(3):
        g = out[str(k)]["guid"]
        assert g["inst30_1h"] == {"skipped": "guid-level model"} and g["emp30"]["basis"] == "guid_final"
        view = T.ovr_view(guids, k, ("score_final_cal",))  # at end: the final (whole-recording) score
        want = T.fpr_threshold(view.loc[view["y"] == 0, "score_final_cal"], 0.3, "empirical")["threshold"]
        assert g["emp30"]["threshold"] == pytest.approx(want)
    assert T.ovr_view(seg, 0, T.OVR_SEGMENT_SCORES)[list(T.OVR_SEGMENT_SCORES)].isna().all(axis=None)
