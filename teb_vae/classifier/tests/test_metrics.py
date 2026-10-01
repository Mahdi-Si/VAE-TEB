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
