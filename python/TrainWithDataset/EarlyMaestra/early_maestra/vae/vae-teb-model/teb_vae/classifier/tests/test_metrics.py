"""T-E1 to T-E4 for the metric engine of metrics.py (SPEC §11.4, §11.7, §15)."""
from __future__ import annotations


import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

from teb_vae.classifier import metrics as M

def _data(n=400, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(-0.5, 1.5, n)
    y = (rng.uniform(size=n) < expit(true_logit)).astype(int)
    return y, scale * true_logit


# --- T-E1: engine matches sklearn / known formulas ---------------------------------------------
@pytest.mark.parametrize("seed", range(3))
def test_threshold_free_matches_sklearn(seed):
    y, s = _data(seed=seed)
    p, alpha = expit(s), 0.3
    out = M.threshold_free(y, s, alpha=alpha)
    assert out["auroc"] == pytest.approx(roc_auc_score(y, s))
    assert out["auprc"] == pytest.approx(average_precision_score(y, s))
    assert out["prevalence"] == pytest.approx(y.mean())
    assert out["logloss"] == pytest.approx(log_loss(y, p))
    assert out["brier"] == pytest.approx(brier_score_loss(y, p))
    assert out["scaled_brier"] == pytest.approx(1 - out["brier"] / (y.mean() * (1 - y.mean())))
    # McClish pAUC from the raw partial area (Appendix B)
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    stop = np.searchsorted(fpr, alpha, side="right")
    t_a = np.interp(alpha, fpr[stop - 1:stop + 1], tpr[stop - 1:stop + 1])
    a_p = np.trapezoid(np.r_[tpr[:stop], t_a], np.r_[fpr[:stop], alpha])
    a_min, a_max = alpha**2 / 2, alpha
    assert out["pauc@0.3"] == pytest.approx(0.5 * (1 + (a_p - a_min) / (a_max - a_min)), abs=1e-9)
    # ECE: equal-mass bins, same bins as calibration_curve(strategy='quantile')
    frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
    counts = np.bincount(np.searchsorted(np.quantile(p, np.linspace(0, 1, 11))[1:-1], p), minlength=10)
    assert out["ece"] == pytest.approx(np.sum(counts[counts > 0] / y.size * np.abs(frac - mean_p)))


def test_calibration_known_answers():
    y, s = _data(n=20000, seed=1)
    good = M.threshold_free(y, s, alpha=0.1)
    assert good["calib_slope"] == pytest.approx(1.0, abs=0.05)
    assert good["calib_intercept"] == pytest.approx(0.0, abs=0.05)
    assert good["ici"] < 0.02 and good["e50"] <= good["e90"]
    over = M.threshold_free(y, 2.0 * s, alpha=0.1)  # overconfident logits
    assert over["calib_slope"] == pytest.approx(0.5, abs=0.03)
    assert over["ici"] > good["ici"]
    shifted = M.threshold_free(y, s - 1.0, alpha=0.1)  # under-predicting risk
    assert shifted["calib_intercept"] > 0.8


# --- T-E2: pooled confusion ---------------------------------------------------------------------
def test_pooled_confusion_sums_folds_and_dedupes_shared():
    rng = np.random.default_rng(5)
    frames = []
    for fold in (1, 2, 3):
        n = 30
        frames.append(pd.DataFrame({"fold": fold, "guid": [f"f{fold}_{i}" for i in range(n)],
                                    "y": rng.integers(0, 2, n), "score_final_cal": rng.normal(size=n),
                                    "shared_test": False}))
    shared = pd.DataFrame({"guid": ["s1", "s2"], "y": [0, 0], "shared_test": True})
    frames += [shared.assign(fold=f, score_final_cal=[5.0 * (f == 2), -5.0]) for f in (1, 2, 3)]
    rows = pd.concat(frames, ignore_index=True)
    thr = {1: 0.0, 2: 0.3, 3: -0.2}
    per_fold = [M.thresholded(g["y"], g["score_final_cal"], thr[f]) for f, g in rows[~rows.shared_test].groupby("fold")]
    ex = M.pooled_confusion(rows, thr, "exclude")
    for c in ("tp", "fp", "tn", "fn"):
        assert ex[c] == sum(d[c] for d in per_fold)
    assert ex["n_dropped"] == 6
    ff = M.pooled_confusion(rows, thr, "first_fold")  # fold 1 copies: s1 = 0 (not > 0), s2 = -5
    assert ff["tn"] == ex["tn"] + 2 and ff["fp"] == ex["fp"] and ff["n_dropped"] == 4
    bad = pd.concat([rows, rows[~rows.shared_test].head(1)])
    with pytest.raises(ValueError):
        M.pooled_confusion(bad, thr, "first_fold")


# --- T-E3: bootstrap ----------------------------------------------------------------------------
def test_bootstrap_reproducible_stratified_and_counts_undefined():
    y_g, s_g = _data(n=60, seed=6)
    guids = np.repeat([f"g{i}" for i in range(60)], 3)  # 3 segment rows per GUID (cluster)
    y, s = np.repeat(y_g, 3), np.repeat(s_g, 3) + np.random.default_rng(0).normal(0, 0.1, 180)
    seen = []

    def fn(i):
        seen.append((int(y[i].sum()), int(i.size)))
        return {"auroc": roc_auc_score(y[i], s[i]),
                "needs_g0": 1.0 if (i < 3).any() else np.nan,  # undefined when GUID g0 is not drawn
                "never": np.nan}

    a = M.bootstrap_ci(fn, guids, y, resamples=300, seed=11)
    sizes = seen[1:]
    b = M.bootstrap_ci(fn, guids, y, resamples=300, seed=11)
    assert a == b
    assert a["metrics"]["auroc"]["value"] == pytest.approx(roc_auc_score(y, s))
    assert a["metrics"]["auroc"]["ci_lo"] < a["metrics"]["auroc"]["value"] < a["metrics"]["auroc"]["ci_hi"]
    # stratified cluster draws: each keeps the class counts and whole GUIDs
    assert all(pos == y.sum() and n == y.size for pos, n in sizes[:300])
    und = a["metrics"]["needs_g0"]["n_undefined"]
    assert 0 < und < 300  # P(g0 absent) ~ (1 - 1/k)^k ~ 0.37
    assert a["metrics"]["never"]["n_undefined"] == 300 and np.isnan(a["metrics"]["never"]["ci_lo"])
    assert a["metrics"]["auroc"]["n_undefined"] == 0
    assert a["record"] == {"method": a["record"]["method"], "resamples": 300, "seed": 11,
                                "confidence": 0.95, "n": 60, "n_dropped": 300}
    c = M.bootstrap_ci(fn, guids, y, resamples=300, seed=12)
    assert c["metrics"]["auroc"] != a["metrics"]["auroc"]


# --- T-E4: prevalence re-weighting and friends ------------------------------------------------


def test_shuffled_interval_uses_the_fold_spread():
    """Criterion 7 (§11.15): the t interval over the folds, not the pooled bootstrap CI. The per-fold test AUROCs of
    run 2026-10-07--17-52's shuffled control hold 0.5 (its pooled CI [0.510, 0.539] did not); one finite fold falls
    back to the pooled value and CI."""
    folds = [0.488, 0.486, 0.601, 0.535, 0.603, 0.642, 0.609, 0.572, 0.407, 0.57]
    point, lo, hi = M.shuffled_interval(folds, 0.524, 0.510, 0.539)
    assert point == pytest.approx(0.5513) and lo == pytest.approx(0.4997, abs=1e-4) and hi == pytest.approx(0.6029,
                                                                                                         abs=1e-4)
    assert M.shuffled_interval([0.6, np.nan], 0.524, 0.510, 0.539) == (0.524, 0.510, 0.539)


# --- T-E5: row weights (§11.7 fold weighting) ----------------------------------------------------
def _weighted_case(n=300, seed=4):
    """Binary labels with tied scores, a 3-class split of the positives, class probabilities, and 2-row units."""
    y, s = _data(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    y3 = np.where(y == 1, rng.integers(1, 3, n), 0)
    return y, np.round(s, 1), y3, rng.dirichlet([1.0, 1.0, 1.0], n), np.repeat(np.arange(n // 2), 2)


def _points(r):
    return {m: v["value"] for m, v in r["metrics"].items()}


def test_weights_none_and_ones_agree():
    y, s, y3, P, units = _weighted_case()
    one, boot, fire, pos = np.ones(y.size), {"resamples": 50, "seed": 1}, s > 0.2, y == 1
    assert M.threshold_free(y, s, alpha=0.3, w=one) == M.threshold_free(y, s, alpha=0.3)
    assert M.thresholded(y, s, 0.2, w=one) == M.thresholded(y, s, 0.2)
    assert M.multiclass(y3, P, w=one) == M.multiclass(y3, P)
    pts = np.linspace(0.05, 0.9, 18)
    pd.testing.assert_frame_equal(M.net_benefit(y, expit(s), pts, w=one), M.net_benefit(y, expit(s), pts))
    r0, r1 = M.roc_points(y, s), M.roc_points(y, s, w=one)
    assert all(np.array_equal(r0[k], r1[k], equal_nan=True) for k in r0)
    assert _points(M.bootstrap_rank(y, s, units, alpha=0.3, w=one, **boot)) == _points(
        M.bootstrap_rank(y, s, units, alpha=0.3, **boot))
    num, den = np.stack([fire & pos, fire & ~pos]), np.stack([pos, ~pos])
    for u in (None, units):  # Wilson path, and the cluster bootstrap (two rows per unit)
        assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(
            M.rate_ci(num, den, y, u, w=one, **boot), M.rate_ci(num, den, y, u, **boot)))
    pop = pd.DataFrame({"y": y, "score": s, "patient": units})  # the `w` column hook of the row writers
    for rows in ((M._tf_rows(pop, 0.3, boot), M._tf_rows(pop.assign(w=1.0), 0.3, boot)),
                 (M._thr_rows(pop, 0.2, 0.3, boot), M._thr_rows(pop.assign(w=1.0), 0.2, 0.3, boot))):
        assert [r["value"] for r in rows[0]] == [r["value"] for r in rows[1]]
        assert type(rows[0][0]["n_pos"]) is int and type(rows[1][0]["n_pos"]) is int  # all-ones w: the unweighted path
        assert rows[0][0]["n_pos"] == rows[1][0]["n_pos"] and rows[0][0]["n_neg"] == rows[1][0]["n_neg"]


def test_integer_weights_equal_row_duplication():
    y, s, y3, P, units = _weighted_case()
    w = np.where(np.arange(y.size) % 3 == 0, 2.0, 1.0)
    d = np.repeat(np.arange(y.size), w.astype(int))  # each weight-2 row twice
    tf, td = M.threshold_free(y, s, alpha=0.3, w=w), M.threshold_free(y[d], s[d], alpha=0.3)
    for m in ("auroc", "auprc", "pauc@0.3", "prevalence", "logloss", "brier", "scaled_brier", "calib_intercept"):
        assert tf[m] == pytest.approx(td[m], abs=1e-9), m
    assert M.thresholded(y, s, 0.2, w=w) == pytest.approx(M.thresholded(y[d], s[d], 0.2), abs=1e-9)
    assert M._rates(s > 0.2, y == 1, w=w) == pytest.approx(M._rates(s[d] > 0.2, y[d] == 1), abs=1e-9)
    assert M.multiclass(y3, P, w=w) == pytest.approx(M.multiclass(y3[d], P[d]), abs=1e-9)
    pts = np.linspace(0.05, 0.9, 18)
    pd.testing.assert_frame_equal(M.net_benefit(y, expit(s), pts, w=w), M.net_benefit(y[d], expit(s[d]), pts),
                                  atol=1e-9, rtol=0)
    rw, rd = M.roc_points(y, s, w=w), M.roc_points(y[d], s[d])
    for k in rw:
        np.testing.assert_allclose(rw[k], rd[k], atol=1e-9, rtol=0)
    boot = {"resamples": 50, "seed": 1}
    assert _points(M.bootstrap_rank(y, s, units, alpha=0.3, w=w, **boot)) == pytest.approx(
        _points(M.bootstrap_rank(y[d], s[d], units[d], alpha=0.3, **boot)), abs=1e-9)
    assert _points(M.bootstrap_auroc(y, s, units, 50, 1, w=w)) == pytest.approx(
        _points(M.bootstrap_auroc(y[d], s[d], units[d], 50, 1)), abs=1e-9)


def test_zero_weights_equal_dropping_rows():
    y, s, y3, P, units = _weighted_case()
    w = np.random.default_rng(9).uniform(0.1, 1.0, y.size)
    w[::4], w[2:4] = 0.0, 0.0  # unit 1 loses both rows
    k, boot, pts = w > 0, {"resamples": 100, "seed": 3}, np.linspace(0.05, 0.9, 18)
    assert M.threshold_free(y, s, alpha=0.3, w=w) == M.threshold_free(y[k], s[k], alpha=0.3, w=w[k])
    assert M.threshold_free(y, s, alpha=0.3, w=k * 1.0) == M.threshold_free(y[k], s[k], alpha=0.3)
    assert M.thresholded(y, s, 0.2, w=w) == M.thresholded(y[k], s[k], 0.2, w=w[k])
    assert M.multiclass(y3, P, w=w) == M.multiclass(y3[k], P[k], w=w[k])
    pd.testing.assert_frame_equal(M.net_benefit(y, expit(s), pts, w=w), M.net_benefit(y[k], expit(s[k]), pts, w=w[k]))
    r0, r1 = M.roc_points(y, s, w=w), M.roc_points(y[k], s[k], w=w[k])
    assert all(np.array_equal(r0[c], r1[c], equal_nan=True) for c in r0)
    assert M.bootstrap_rank(y, s, units, alpha=0.3, w=w, **boot) == M.bootstrap_rank(y[k], s[k], units[k], alpha=0.3,
                                                                                    w=w[k], **boot)
    assert M.three_class_ci(y3, P, units, alpha=0.3, w=w, **boot) == M.three_class_ci(y3[k], P[k], units[k], alpha=0.3,
                                                                                    w=w[k], **boot)
    fire, pos = s > 0.2, y == 1
    num, den = np.stack([fire & pos, fire & ~pos]), np.stack([pos, ~pos])
    assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(
        M.rate_ci(num, den, y, units, w=w, **boot), M.rate_ci(num[:, k], den[:, k], y[k], units[k], w=w[k], **boot)))
    pop = pd.DataFrame({"y": y, "score": s, "patient": units, "w": w})
    for rows in (lambda x: M._thr_rows(x, 0.2, 0.3, boot), lambda x: M._tf_rows(x, 0.3, boot)):
        assert pd.DataFrame(rows(pop)).equals(pd.DataFrame(rows(pop[k].reset_index(drop=True))))


def test_kish_wilson():
    k, n = np.array([0, 3, 7, 10]), 10
    lo, hi = M.kish_wilson(np.arange(n)[None, :] < k[:, None], np.ones((k.size, n), bool), np.ones(n))
    wlo, whi = M.wilson(k, n)
    assert np.array_equal(lo, wlo) and np.array_equal(hi, whi)  # unit weights: plain Wilson of the counts
    w = np.r_[np.ones(5), np.full(5, 0.1)]  # half the rows at 0.1
    hit = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0], bool)
    n_eff, p = 5.5 ** 2 / 5.05, 2.1 / 5.5  # (sum w)^2 / sum w^2 = 5.99, not 10; sum w hit / sum w
    assert n_eff == pytest.approx(5.990099, abs=1e-6)
    elo, ehi = M.wilson(p * n_eff, n_eff)
    lo, hi = M.kish_wilson(hit, np.ones(10, bool), w)
    assert lo[0] == pytest.approx(elo, abs=1e-12) and hi[0] == pytest.approx(ehi, abs=1e-12)
    lo, hi, nu = M.rate_ci(hit, np.ones(10, bool), hit.astype(int), None, resamples=10, seed=0, w=w)  # Wilson path
    assert lo[0] == pytest.approx(elo, abs=1e-12) and hi[0] == pytest.approx(ehi, abs=1e-12) and np.isnan(nu[0])
    assert np.isnan(M.kish_wilson(hit, np.zeros(10, bool), w)[0][0])  # empty denominator


def test_weight_edge_cases():
    """A NaN score on a zero-weight row is dropped with the row (a NaN on a kept row still raises); a ``w`` column of
    ones takes the unweighted path (int counts), and an all-zero one leaves an empty population."""
    got = M.threshold_free([0, 1, 0], [0.0, 1.0, np.nan], alpha=0.3, w=[1.0, 1.0, 0.0])
    ref = M.threshold_free([0, 1], [0.0, 1.0], alpha=0.3)
    assert got["brier"] == pytest.approx(ref["brier"]) and got["logloss"] == pytest.approx(ref["logloss"])
    with pytest.raises(ValueError):
        M.threshold_free([0, 1, 0], [0.0, 1.0, np.nan], alpha=0.3, w=[1.0, 1.0, 1.0])
    f, w = M._weighted(pd.DataFrame({"y": [0, 1], "w": [1.0, 1.0]}))
    assert w is None and len(f) == 2
    f, w = M._weighted(pd.DataFrame({"y": [0, 1], "w": [0.0, 0.0]}))
    assert w is None and f.empty
