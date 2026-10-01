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
