"""Fold weighting (SPEC §11.7) in metrics and report: ``metrics.fold_weights_of``, the fold band of the ``*_folds``
subgroup figures (``report.fold_rates``) and the report's own weighted per-fold curves (``report._fold_w``,
``_s_rows``, ``_reliability``, ``_ovr_curve``)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from teb_vae.classifier import report as R
from teb_vae.classifier.metrics import fold_weights_of, roc_points, wilson
from teb_vae.classifier.report import fold_rates

WEIGHTED = {"results": {"fold_weighting": "inverse_k"}}


def _frame():
    """Two folds' test rows: ``a`` and ``b`` in one fold each, ``s`` a shared test GUID in both."""
    return pd.DataFrame({"fold": ["1", "1", "2", "2"], "guid": ["a", "s", "b", "s"],
                         "shared_test": [False, True, False, True]})


def test_fold_weights_of_split_each_shared_guid_over_its_folds():
    """The row weights of the per-fold test populations: 1/K for a GUID in K folds, 0 under ``exclude``; val rows all
    weigh 1 (each fold's thresholds were selected on its unweighted val population)."""
    assert fold_weights_of(_frame(), "first_fold", "test").tolist() == [1.0, 0.5, 1.0, 0.5]
    assert fold_weights_of(_frame(), "exclude", "test").tolist() == [1.0, 0.0, 1.0, 0.0]
    assert fold_weights_of(_frame(), "first_fold", "val").tolist() == [1.0] * 4


def _rows(fold, tp, n_pos, fp, n_neg):
    """One fold's S2 count rows of member ``a`` at t = 1: ``tp`` and ``fp`` metric rows carrying the (weighted)
    ``n_pos`` / ``n_neg`` columns, as the per-fold rows of a weighted evaluation do."""
    return [dict(fold=fold, subgroup_value="a", t=1.0, metric=m, value=v, n_pos=n_pos, n_neg=n_neg)
            for m, v in (("tp", tp), ("fp", fp))]


def test_fold_rates_line_is_the_pooled_rate_inside_the_band():
    """Each fold's rate from its weighted counts (every fold with a GUID of the class counts, however thin); the line
    is Σ num / Σ den, so it lies within the fold min–max; specificity mirrors the FPR."""
    per = pd.DataFrame(_rows("1", 5, 10, 1.0, 10.0) + _rows("2", 1, 4, 3.0, 5.0)
                       + _rows("3", 0, 0, 0.5, 5.0))  # fold 3: no adverse GUID in the bin, so no sensitivity
    sens, rates = fold_rates(per, "sens")
    s = sens.loc[("a", 1.0)]
    assert (s["min"], s["max"], s["count"], s["n_median"]) == (0.25, 0.5, 2, 7.0)
    assert s["line"] == pytest.approx(6 / 14) and sorted(rates.loc["a"].tolist()) == [0.25, 0.5]
    f = fold_rates(per, "fpr")[0].loc[("a", 1.0)]
    assert (f["min"], f["max"], f["count"]) == (pytest.approx(0.1), pytest.approx(0.6), 3)
    assert f["line"] == pytest.approx(4.5 / 20) and f["min"] <= f["line"] <= f["max"]
    sp = fold_rates(per, "spec")[0].loc[("a", 1.0)]
    assert (sp["min"], sp["max"], sp["line"]) == (pytest.approx(0.4), pytest.approx(0.9), pytest.approx(1 - 4.5 / 20))
    assert fold_rates(per[per["metric"] != "tp"], "sens")[0].empty  # no numerator rows: no band


def test_report_follows_the_evaluations_fold_weighting():
    """Without ``results.fold_weighting`` (an older evaluation) the report's own per-fold curves stay unweighted and
    the ``*_folds`` figures get the pooled rows only; with it, the test rows weigh ``fold_weights_of`` (val stays
    unweighted) and keep every fold."""
    c = {"data": {"shared_test_policy": "first_fold"}, "eval": {"primary_policy": "p"}}
    assert R._fold_w({"summary": {}}, c, _frame(), "test").tolist() == [1.0] * 4
    assert R._fold_w({"summary": WEIGHTED}, c, _frame(), "test").tolist() == [1.0, 0.5, 1.0, 0.5]
    assert R._fold_w({"summary": WEIGHTED}, c, _frame(), "val").tolist() == [1.0] * 4
    assert R._fold_phrase({"summary": WEIGHTED}, c, "val") == ""
    tr = pd.DataFrame([dict(model_id="model", seed="42", fold=f, split="test", level="online", subgroup="class",
                            subgroup_value="healthy", axis="to_delivery", t=1.0, point="bin", policy_id="p", metric="fp",
                            value=1.0, n_pos=0.0, n_neg=4.0) for f in ("pooled", "1")])
    for summary, folds in (({}, ["pooled"]), (WEIGHTED, ["pooled", "1"])):
        d = R._s_rows({"tr": tr, "summary": summary}, c, "to_delivery", "test", None, ovr=False, all_folds=True)
        assert d["fold"].tolist() == folds and R._has_fold_counts(d) == (len(folds) > 1)


def test_weighted_reliability_and_ovr_curve():
    """A weight-w row counts as w rows: the reliability bins sum the weights (a zero weight drops the row), and an
    integer weight in the one-vs-rest curve equals repeating the row; unweighted, the curve is ``roc_points``'."""
    r = R._reliability([1, 0, 0, 1], [0.2, 0.4, 0.6, 0.8], [1.0, 0.5, 0.5, 1.0]).iloc[0]
    assert (r["n"], r["k"], r["pred"], r["obs"]) == (pytest.approx(3.0), 2.0, pytest.approx(0.5), pytest.approx(2 / 3))
    pd.testing.assert_frame_equal(R._reliability([1, 0, 0, 1], [0.2, 0.4, 0.6, 0.8], [1, 0, 1, 1]),
                                  R._reliability([1, 0, 1], [0.2, 0.6, 0.8]))
    f = pd.DataFrame({"class_code": [1, 2, 1, 3, 2, 1], "p_c1_cal": [0.1, 0.8, 0.3, 0.4, 0.6, 0.2]})
    y, s = (f["class_code"] - 1 == 1).to_numpy(), f["p_c1_cal"].to_numpy()
    ref = roc_points(y, s)
    fpr, tpr, _, prev = R._ovr_curve(f, 1, pr=False)
    np.testing.assert_allclose(fpr, ref["fpr"])
    np.testing.assert_allclose(tpr, ref["tpr"])
    assert prev == pytest.approx(2 / 6)
    rec, prec, _, _ = R._ovr_curve(f, 1, pr=True)
    ok = np.isfinite(ref["precision"])
    np.testing.assert_allclose(rec, ref["tpr"][ok])
    np.testing.assert_allclose(prec, ref["precision"][ok])
    rep = pd.concat([f, f.iloc[[3]]], ignore_index=True)  # row 3 (a negative) twice
    for pr in (False, True):
        a, b = R._ovr_curve(f.assign(w=[1, 1, 1, 2, 1, 1]), 1, pr), R._ovr_curve(rep, 1, pr)
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])
        assert a[2:] == pytest.approx(b[2:])
    assert (R._count_spec([3.0, 4.0]), R._count_spec([412.6])) == (".0f", ".1f")


def test_weighted_reliability_uses_kish_intervals():
    """The bin interval is Wilson at Kish's effective size, not at Σw: two GUIDs at weight 0.1 are two GUIDs
    ($n_\mathrm{eff} = 0.2^2/0.02 = 2$), not 0.2 of one."""
    r = R._reliability([0, 1], [0.3, 0.7], [0.1, 0.1]).iloc[0]
    lo, hi = wilson(1, 2)
    assert (r["n"], r["lo"], r["hi"]) == (pytest.approx(0.2), pytest.approx(lo), pytest.approx(hi))
