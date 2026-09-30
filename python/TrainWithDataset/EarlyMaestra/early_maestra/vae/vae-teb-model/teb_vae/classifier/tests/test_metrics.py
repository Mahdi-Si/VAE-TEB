"""T-E1 to T-E4 for the metric engine of metrics.py (SPEC §11.4, §11.7, §15)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from teb_vae.classifier import metrics as M

BOOT = {"resamples": 20, "seed": 0}  # _three_class_rows CIs; the point values are what these tests check


def _data(n=400, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(-0.5, 1.5, n)
    y = (rng.uniform(size=n) < expit(true_logit)).astype(int)
    return y, scale * true_logit


def test_engine_is_torch_free():
    import subprocess
    code = "import sys, teb_vae.classifier.metrics, teb_vae.classifier.thresholds; print('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "False"


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


def test_ici_smooth_is_unpenalised():
    """An underconfident model (true logit 2s) at n = 300: the ICI smooth must not be L2-shrunk toward flat (C = 1
    reported about half the true ICI of about 0.10), and it fits without a convergence warning."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    ici, true = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for seed in range(5):
            rng = np.random.default_rng(seed)
            s = rng.normal(0.0, 1.5, 300)
            ici.append(M.threshold_free((rng.random(300) < expit(2 * s)).astype(int), s, alpha=0.3)["ici"])
            true.append(np.abs(expit(2 * s) - expit(s)).mean())
    assert np.mean(ici) == pytest.approx(np.mean(true), abs=0.02)


def test_threshold_free_single_class_is_nan():
    out = M.threshold_free(np.zeros(6, int), np.linspace(-1, 1, 6), alpha=0.3)
    for k in ("auroc", "auprc", "pauc@0.3", "scaled_brier", "calib_slope", "calib_intercept", "ici"):
        assert np.isnan(out[k])
    assert out["prevalence"] == 0.0 and np.isfinite(out["brier"])


def test_thresholded_known_answer_and_nan_rules():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    s = np.array([3.0, 2.0, 0.5, 2.0, 0.5, 0.0, -1.0, -2.0])
    out = M.thresholded(y, s, 0.5)  # strict: 0.5 does not alarm
    assert (out["tp"], out["fn"], out["fp"], out["tn"]) == (2, 1, 1, 4)
    assert out["sens"] == pytest.approx(2 / 3) and out["spec"] == pytest.approx(0.8)
    assert out["fpr"] == pytest.approx(0.2) and out["ppv"] == pytest.approx(2 / 3) and out["npv"] == pytest.approx(0.8)
    assert out["bal_acc"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert out["lr_pos"] == pytest.approx((2 / 3) / 0.2) and out["lr_neg"] == pytest.approx((1 / 3) / 0.8)
    assert out["f1"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))
    assert out["mcc"] == pytest.approx((2 * 4 - 1 * 1) / np.sqrt(3 * 3 * 5 * 5))
    none = M.thresholded(np.zeros(4, int), np.arange(4.0), 10.0)  # no positives, never fires
    assert np.isnan(none["sens"]) and np.isnan(none["bal_acc"]) and np.isnan(none["mcc"])
    assert none["spec"] == 1.0 and none["ppv"] == 0.0 and none["f1"] == 0.0


def test_multiclass_matches_sklearn():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 3, 300)
    P = rng.dirichlet([1, 1, 1], 300)
    P[np.arange(300), y] += 0.5
    P /= P.sum(1, keepdims=True)
    out = M.multiclass(y, P)
    pred = P.argmax(1)
    assert out["qwk"] == pytest.approx(cohen_kappa_score(y, pred, weights="quadratic"))
    assert out["auroc_hand_till"] == pytest.approx(roc_auc_score(y, P, multi_class="ovo"))
    assert out["auroc_macro"] == pytest.approx(np.mean([out[f"auroc_ovr_c{k}"] for k in range(3)]))
    assert out["bal_acc"] == pytest.approx(balanced_accuracy_score(y, pred))
    assert out["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))
    for k in range(3):
        assert out[f"auroc_ovr_c{k}"] == pytest.approx(roc_auc_score(y == k, P[:, k]))
        assert out[f"auprc_ovr_c{k}"] == pytest.approx(average_precision_score(y == k, P[:, k]))
        assert out[f"recall_c{k}"] == pytest.approx(recall_score(y, pred, labels=[k], average="macro"))
        assert out[f"precision_c{k}"] == pytest.approx(precision_score(y, pred, labels=[k], average="macro"))
        assert out[f"f1_c{k}"] == pytest.approx(f1_score(y, pred, labels=[k], average="macro"))
    # RPS known answer: one-hot correct = 0; certain two classes off = 1
    assert M.multiclass([0, 2], np.array([[1.0, 0, 0], [0, 0, 1.0]]))["rps"] == 0.0
    assert M.multiclass([0], np.array([[0, 0, 1.0]]))["rps"] == 1.0


def test_three_class_rows_match_sklearn_and_pool_by_summing():
    """T-E1, 3-class rows of run_metrics: sklearn values per fold, pooled confusion = summed folds (the shared test
    GUID once), the collapses = threshold_free on logit(p1 + p2) / logit p2, names that headline() never picks."""
    from types import SimpleNamespace

    rng = np.random.default_rng(5)
    n = 90
    y3 = np.r_[rng.integers(0, 3, n - 3), 0, 1, 2]
    P = rng.dirichlet([1, 1, 1], n)
    P[np.arange(n), y3] += 0.6
    P /= P.sum(1, keepdims=True)
    g = pd.DataFrame({"model_id": "model", "seed": "42", "split": "test", "fold": np.r_[np.repeat([1, 2, 3], 30)],
                      "guid": [f"g{i}" for i in range(n)], "class_code": y3 + 1, "shared_test": False,
                      **{f"p_c{k}_cal": P[:, k] for k in range(3)}})
    g.loc[g.index[:2], "shared_test"] = True
    g = pd.concat([g, g.iloc[:2].assign(fold=2)], ignore_index=True)  # the shared GUIDs again in fold 2
    no_segments = g.iloc[:0].assign(seg_pos=0, t_end_s=0.0, in_eval_window=True, logit_seg_cal=0.0)  # GUID level only
    ctx = SimpleNamespace(guids=g, seg=no_segments, prov={"run_id": "r"}, inclusion=[],
                          cfg={"labels": {"task": "three_class", "head": "ordinal", "strategy": "horizon_decay"},
                               "eval": {"exclude_last_min": 0}})
    df = pd.DataFrame(M._three_class_rows(ctx, "model", "42", "test", 0.3, "first_fold", BOOT), columns=M.METRIC_COLUMNS)
    assert set(df["head"]) == {"ordinal"} and set(df["task"]) == {"three_class"}
    assert set(df["metric_type"]) == {"threshold_free"} and set(df["policy_id"].dropna()) == {"argmax"}
    f1 = df[df["fold"] == "1"].set_index("metric")["value"]
    y, Q = y3[:30], P[:30]
    assert f1["auroc_hand_till"] == pytest.approx(roc_auc_score(y, Q, multi_class="ovo"))
    assert f1["qwk"] == pytest.approx(cohen_kappa_score(y, Q.argmax(1), weights="quadratic"))
    assert f1["confusion_t2_p0"] == ((y == 2) & (Q.argmax(1) == 0)).sum()
    assert f1["adverse_vs_healthy/auroc"] == pytest.approx(roc_auc_score(y > 0, Q[:, 1] + Q[:, 2]))
    assert f1["hie_vs_rest/auprc"] == pytest.approx(average_precision_score(y == 2, Q[:, 2]))
    pooled = df[df["fold"] == "pooled"].set_index("metric")["value"]
    cm = sum(confusion_matrix(y3[i:i + 30], P[i:i + 30].argmax(1), labels=[0, 1, 2]) for i in (0, 30, 60))
    assert all(pooled[f"confusion_t{i}_p{j}"] == cm[i, j] for i in range(3) for j in range(3))
    assert pooled["auroc_macro"] == pytest.approx(roc_auc_score(y3, P, multi_class="ovr"))
    assert M.headline(df, 0.3) == {}  # no 3-class name is a headline metric
    assert M._three_class_rows(ctx, "probe", "na", "test", 0.3, "first_fold", BOOT) == []  # no such model / no probs


def test_net_benefit_known_answer():
    y, p = np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.6, 0.1])
    nb = M.net_benefit(y, p, [0.5])
    assert nb["net_benefit"][0] == pytest.approx(1 / 4 - 1 / 4 * 1.0)
    assert nb["treat_all"][0] == pytest.approx(0.5 - 0.5)


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


def test_metric_rows_long_format():
    ci = {"auroc": {"ci_lo": 0.7, "ci_hi": 0.9, "n_undefined": 2}}
    rows = M.metric_rows({"auroc": 0.8, "tp": 3}, ci, fold=1, split="test", level="guid")
    df = pd.DataFrame(rows, columns=M.METRIC_COLUMNS)
    assert list(df.columns) == list(M.METRIC_COLUMNS) and len(df) == 2
    assert df.loc[0, "ci_lo"] == 0.7 and df.loc[0, "n_boot_undefined"] == 2 and df.loc[1, "value"] == 3.0
    with pytest.raises(ValueError):
        M.metric_rows({"auroc": 0.8}, not_a_column=1)


# --- T-E4: prevalence re-weighting and friends ------------------------------------------------
def test_prevalence_reweighting_known_answer():
    ppv, npv = M.adjust_ppv_npv(0.8, 0.9, 0.1)
    assert ppv == pytest.approx(0.08 / (0.08 + 0.09))
    assert npv == pytest.approx(0.81 / (0.81 + 0.02))
    # at the observed prevalence, it reproduces the observed PPV/NPV
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    s = np.array([3.0, 2.0, 0.5, 2.0, 0.5, 0.0, -1.0, -2.0])
    t = M.thresholded(y, s, 0.5)
    assert M.adjust_ppv_npv(t["sens"], t["spec"], y.mean()) == pytest.approx((t["ppv"], t["npv"]))


def test_prior_shift_and_wilson_known_answers():
    assert M.prior_shift(0.0, 0.5, 0.2) == pytest.approx(np.log(0.25))
    assert expit(M.prior_shift(np.log(0.3 / 0.7), 0.3, 0.3)) == pytest.approx(0.3)
    lo, hi = M.wilson(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert all(np.isnan(M.wilson(0, 0)))


def test_pooled_derived_rates_carry_cis(tmp_path):
    """Pooled OOF thresholded rows give the derived rates (balanced accuracy, MCC, LR+, LR-, F1) the same patient-cluster
    bootstrap CIs as the per-fold rows; they used to get none."""
    import json

    from teb_vae.classifier.tests.test_eval_analyses import make_run

    run, cfg = make_run(tmp_path, folds=(1, 2), resamples=30)
    ctx = M.load_context(run, cfg)
    rows = pd.DataFrame(M.run_metrics(ctx, eval_config=ctx.cfg["eval"], out_dir=run / "evaluation")["rows"])
    d = rows[rows["policy_id"].notna() & rows["metric"].isin(M._DERIVED) & (rows["model_id"] == "probe")]
    pooled, per_fold = d[d["fold"] == "pooled"], d[d["fold"] != "pooled"]
    for x in (pooled, per_fold):
        ok = x[x["metric"].isin(["bal_acc", "mcc"])]
        assert len(ok) and ok["ci_lo"].notna().all() and (ok["ci_lo"] <= ok["value"] + 1e-12).all()
    json.dumps(M.run_context(run))  # still JSON-able


def test_derived_rate_draws_match_confusion_rates():
    """The vectorised derived rates (every bootstrap draw at once) equal :func:`confusion_rates` count by count,
    NaN and 0 rules included (no positive, no negative, never fired)."""
    rng = np.random.default_rng(0)
    counts = np.r_[rng.integers(0, 6, (200, 4)), [[0, 3, 4, 0], [2, 0, 0, 3], [0, 0, 5, 4], [3, 0, 0, 0]]]
    got = M._derived(*counts.T)
    for i, (tp, fp, tn, fn) in enumerate(counts):
        want = M.confusion_rates(tp, fp, tn, fn)
        for k in M._DERIVED:
            assert np.allclose(got[k][i], want[k], equal_nan=True), (k, tp, fp, tn, fn)
    units = np.repeat(np.arange(40), 5)  # 40 patients x 5 segments: the cluster bootstrap path
    fire, pos = rng.random(200) < 0.4, np.repeat(rng.random(40) < 0.5, 5)
    ci = M._rate_cis(fire, pos, units, 0.3, dict(resamples=200, seed=0))
    for k in M._DERIVED:
        v = M._rates(fire, pos)[k]
        assert np.isfinite(ci[k]["ci_lo"]) and ci[k]["ci_lo"] <= v <= ci[k]["ci_hi"], k


def test_run_context_reads_the_vae_checkpoint_sha(tmp_path):
    """``run_context.checkpoint_sha256`` is the source fingerprint's VAE checkpoint SHA-256 (None for an hdf5 source)."""
    import json

    for source, want in (({"fingerprint_hash": "f", "fingerprint": {"kind": "vae", "checkpoint_sha256": "abc"}}, "abc"),
                         ({"fingerprint_hash": "f", "fingerprint": {"kind": "hdf5"}}, None)):
        (tmp_path / "manifest.json").write_text(json.dumps({"source": source}))
        ctx = M.run_context(tmp_path)
        assert ctx["checkpoint_sha256"] == want and ctx["source_fingerprint"] == "f"
