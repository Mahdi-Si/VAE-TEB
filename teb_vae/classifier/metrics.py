"""Metric engine and analysis registry for the classifier (SPEC §11.4, §11.5, §11.7, §11.9).

The engine is pure functions over numpy arrays (labels, calibrated logits, thresholds) and never
imports torch, so evaluation re-runs from the prediction tables alone. sklearn computes every metric
it has; the rest is a few lines of formula (Appendix B).

The pilot helpers (``latent_pilot/evaluate.py``: ``confusion_counts``, ``derived_rates``,
``paired_bootstrap``, ``metric_intervals``) are ported, not imported: that module loads torch and
the VAE nets at import time, its decision predicate is ``>=`` (the contract is strict ``>``), and
its bootstrap is hard-wired to one metric set. Their semantics are kept: rate NaN only when a class
count is zero (PPV/NPV/F1 are 0 when the rule never fires), outcome-stratified GUID-cluster
percentile draws, and undefined draws counted per metric.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit as _logit, ndtri
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    d2_brier_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

# --- engine ---

NAN = float("nan")

#: Columns of ``evaluation/tables/metrics.parquet``: §11.9 plus ``point`` (checkpoint | bin | end;
#: ``n/a`` for the P2 rows, which are read at a policy's own basis point), in order.
METRIC_COLUMNS: Tuple[str, ...] = (
    "run_id", "model_id", "seed", "fold", "split", "level", "task", "head", "label_strategy",
    "eval_window", "subgroup", "subgroup_value", "axis", "t", "point", "metric_type", "denominator",
    "policy_id", "policy_basis", "threshold", "metric", "value", "ci_lo", "ci_hi", "n_pos", "n_neg",
    "n_boot_undefined",
)


def _yl(y: Any, logit: Any) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    s = np.asarray(logit, dtype=np.float64).reshape(-1)
    if y.size != s.size or np.isnan(s).any():
        raise ValueError(f"{y.size} labels vs {s.size} scores ({int(np.isnan(s).sum())} NaN)")
    return y, s


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE over equal-mass bins, binned exactly as ``calibration_curve(strategy='quantile')``."""
    ids = np.searchsorted(np.quantile(p, np.linspace(0, 1, n_bins + 1))[1:-1], p)
    return float(np.abs(np.bincount(ids, weights=y - p)).sum() / y.size)


def threshold_free(y: Any, logit: Any, *, alpha: float) -> Dict[str, float]:
    """Ranking and calibration metrics of calibrated logits (§11.4).

    ``calib_intercept`` is calibration-in-the-large (slope fixed at 1, logit as offset);
    ``calib_slope`` the unpenalised logistic slope of y on the logit. ICI/E50/E90 use a cubic
    B-spline logistic smooth of y on the logit (Austin & Steyerberg, LOWESS-free). Ranking and
    recalibration metrics are NaN when a class is missing; logloss/brier/ece stay defined.
    """
    y, s = _yl(y, logit)
    p = expit(s)
    out = dict.fromkeys(("auroc", "auprc", "prevalence", f"pauc@{alpha:g}", "logloss", "brier",
                         "scaled_brier", "calib_intercept", "calib_slope", "ece", "ici", "e50", "e90"), NAN)
    if y.size == 0:
        return out
    out.update(prevalence=float(y.mean()), logloss=float(log_loss(y, p, labels=[0, 1])),
               brier=float(brier_score_loss(y, p, labels=[0, 1])), ece=_ece(y, p))
    if not 0 < y.sum() < y.size:
        return out
    b = float(np.abs(s).max()) + 40.0  # bracket: f(-b) < 0 < f(b) whenever both classes exist
    smooth = make_pipeline(SplineTransformer(n_knots=4, degree=3, include_bias=False),  # unpenalised, as the slope:
                           LogisticRegression(C=np.inf, max_iter=1000)).fit(s[:, None], y)  # L2 shrinks it flat
    gap = np.abs(smooth.predict_proba(s[:, None])[:, 1] - p)
    out.update({
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        f"pauc@{alpha:g}": float(roc_auc_score(y, s, max_fpr=alpha)),
        "scaled_brier": float(d2_brier_score(y, p)),
        "calib_intercept": float(brentq(lambda a: expit(a + s).sum() - y.sum(), -b, b)),
        "calib_slope": float(LogisticRegression(C=np.inf, max_iter=1000).fit(s[:, None], y).coef_[0, 0]),
        "ici": float(gap.mean()), "e50": float(np.quantile(gap, 0.5)), "e90": float(np.quantile(gap, 0.9)),
    })
    return out


def confusion_rates(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """Counts plus rates. NaN only when a class count is 0; PPV/NPV/F1 are 0 when never fired."""
    tp, fp, tn, fn = int(tp), int(fp), int(tn), int(fn)
    P, N, pp, pn = tp + fn, tn + fp, tp + fp, tn + fn

    def rate(a: float, d: float, empty: float = NAN) -> float:
        return a / d if d > 0 else empty

    sens, spec = rate(tp, P), rate(tn, N)
    den = math.sqrt(float(pp) * P * N * pn)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr_pos, lr_neg = float(np.float64(sens) / (1.0 - spec)), float((1.0 - np.float64(sens)) / spec)
    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn), "sens": sens, "spec": spec,
        "fpr": 1.0 - spec, "ppv": rate(tp, pp, 0.0), "npv": rate(tn, pn, 0.0),
        "bal_acc": (sens + spec) / 2.0,
        "mcc": NAN if P == 0 or N == 0 else rate(tp * tn - fp * fn, den, 0.0),
        "lr_pos": lr_pos, "lr_neg": lr_neg, "f1": rate(2 * tp, 2 * tp + fp + fn, 0.0),
    }


def thresholded(y: Any, logit: Any, thr: float) -> Dict[str, float]:
    """Confusion and rates of the rule ``logit > thr`` (§11.4); includes ``threshold``."""
    y, s = _yl(y, logit)
    tn, fp, fn, tp = confusion_matrix(y, (s > thr).astype(np.int64), labels=[0, 1]).ravel()
    return confusion_rates(tp, fp, tn, fn) | {"threshold": float(thr)}


def multiclass(y: Any, P: Any) -> Dict[str, float]:
    """3-class (K-class) metrics of probabilities ``P`` (n, K); argmax for the hard ones.

    RPS is normalised by K-1 (0 = perfect, 1 = worst). AUROC-type metrics (and the OvR AUPRC) are NaN
    when a needed class is absent; recall is NaN for an absent class, so ``bal_acc`` is too; precision
    and F1 are 0 for a class argmax never predicts (the binary PPV rule).
    """
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    P = np.asarray(P, dtype=np.float64)
    K, labels, pred = P.shape[1], list(range(P.shape[1])), P.argmax(1)
    full = len(np.unique(y)) == K
    out = {f"{name}_ovr_c{k}": float(fn(y == k, P[:, k])) if 0 < (y == k).sum() < y.size else NAN
           for name, fn in (("auroc", roc_auc_score), ("auprc", average_precision_score)) for k in labels}
    rec = recall_score(y, pred, labels=labels, average=None, zero_division=np.nan)
    prec = precision_score(y, pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    obs = (y[:, None] <= np.arange(K - 1)).astype(np.float64)
    out.update({
        "auroc_macro": float(roc_auc_score(y, P, multi_class="ovr", labels=labels)) if full else NAN,
        "auroc_hand_till": float(roc_auc_score(y, P, multi_class="ovo", labels=labels)) if full else NAN,
        "qwk": float(cohen_kappa_score(y, pred, labels=labels, weights="quadratic")),
        "rps": float(((np.cumsum(P, 1)[:, :-1] - obs) ** 2).sum(1).mean() / (K - 1)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "bal_acc": float(np.mean(rec)),
        **{f"{name}_c{k}": float(v[k]) for name, v in (("recall", rec), ("precision", prec), ("f1", f1))
           for k in labels},
    })
    return out


def net_benefit(y: Any, p: Any, pts: Sequence[float]) -> pd.DataFrame:
    """Decision curve ``NB = TP/N - FP/N * pt/(1-pt)`` (rule ``p > pt``) with treat-all/none."""
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    pts = np.asarray(pts, dtype=np.float64)
    fire, w, pi = np.asarray(p)[None, :] > pts[:, None], pts / (1.0 - pts), y.mean()
    tp, fp = (fire & (y == 1)).mean(1), (fire & (y == 0)).mean(1)
    return pd.DataFrame({"pt": pts, "net_benefit": tp - fp * w, "treat_all": pi - (1 - pi) * w, "treat_none": 0.0})


def wilson(k: Any, n: Any, confidence: float = 0.95) -> Tuple[Any, Any]:
    """Wilson interval for k of n, vectorised: scipy's ``binomtest(k, n).proportion_ci('wilson')``
    formula (Newcombe 1998; 0 at k = 0, 1 at k = n). NaN where n = 0; floats for scalar input."""
    k, n = np.broadcast_arrays(np.asarray(k, np.float64), np.asarray(n, np.float64))
    z = ndtri(0.5 + 0.5 * confidence)
    with np.errstate(invalid="ignore", divide="ignore"):
        p, den = k / n, 2.0 * (n + z * z)
        c, d = (2.0 * n * p + z * z) / den, z / den * np.sqrt(4.0 * n * p * (1.0 - p) + z * z)
    lo = np.where(n > 0, np.where(k == 0, 0.0, c - d), NAN)
    hi = np.where(n > 0, np.where(k == n, 1.0, c + d), NAN)
    return (float(lo), float(hi)) if lo.ndim == 0 else (lo, hi)


def adjust_ppv_npv(sens: float, spec: float, prevalence: float) -> Tuple[float, float]:
    """PPV/NPV at a reference prevalence via Bayes (§11.7)."""
    pi = prevalence
    return (sens * pi / (sens * pi + (1 - spec) * (1 - pi)),
            spec * (1 - pi) / (spec * (1 - pi) + (1 - sens) * pi))


def prior_shift(logit: Any, pi_from: float, pi_to: float) -> Any:
    """``logit p' = logit p - logit pi_from + logit pi_to``."""
    return np.asarray(logit, dtype=np.float64) - _logit(pi_from) + _logit(pi_to)


def pooled_confusion(
    guid_rows: pd.DataFrame, thresholds_by_fold: Mapping[Any, float], shared_test_policy: str,
    *, score: str = "score_final_cal",
) -> Dict[str, Any]:
    """Pooled OOF confusion: each row thresholded at its own fold's threshold, counts summed (§11.7).

    ``guid_rows`` needs ``fold, guid, y, shared_test`` and ``score``. Shared test GUIDs are kept once
    at the lowest fold (``first_fold``) or dropped (``exclude``); any other duplicate GUID raises.
    """
    rows = guid_rows.sort_values("fold", kind="stable")
    shared = rows["shared_test"].astype(bool)
    if shared_test_policy == "exclude":
        kept = rows[~shared]
    elif shared_test_policy == "first_fold":
        kept = rows[~(shared & rows["guid"].duplicated())]
    else:
        raise ValueError(f"unknown shared_test_policy {shared_test_policy!r}")
    if kept["guid"].duplicated().any():
        raise ValueError(f"non-shared GUID(s) repeated across folds: {kept['guid'][kept['guid'].duplicated()].tolist()[:5]}")
    y, s = _yl(kept["y"], kept[score])
    fire = s > kept["fold"].map(thresholds_by_fold).to_numpy(np.float64)
    pos = y == 1
    counts = confusion_rates(int((fire & pos).sum()), int((fire & ~pos).sum()),
                             int((~fire & ~pos).sum()), int((~fire & pos).sum()))
    return counts | {"n_dropped": int(len(rows) - len(kept)), "shared_test_policy": shared_test_policy}


def bootstrap_ci(
    fn: Callable[[np.ndarray], Mapping[str, float]], units: Any, y: Any, *,
    resamples: int = 2000, seed: int = 0, confidence: float = 0.95,
) -> Dict[str, Any]:
    """Outcome-stratified cluster percentile bootstrap (§11.7; pilot ``paired_bootstrap`` semantics).

    ``fn(idx)`` computes ``{metric: value}`` on the rows ``idx`` of the caller's aligned arrays, so
    one call can score several models on the same draws (paired). ``units`` names each row's cluster
    (GUID); units are stratified by the label set they carry and drawn with replacement within the
    stratum, all rows of a unit together. The point value is ``fn`` on all rows.

    Returns:
        ``{metrics: {m: {value, ci_lo, ci_hi, n_undefined}}, record: {method, resamples, seed,
        confidence, n, n_dropped}}``; ``n`` = units, ``n_dropped`` = draws with >= 1 undefined
        metric. Undefined (non-finite) draws are excluded per metric and counted, never hidden.
    """
    y = np.asarray(y).reshape(-1)
    codes, names = pd.factorize(np.asarray(units).reshape(-1), sort=True)
    rows = np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])
    strata: Dict[Tuple[Any, ...], list] = {}
    for u, r in enumerate(rows):
        strata.setdefault(tuple(np.unique(y[r]).tolist()), []).append(u)
    point = dict(fn(np.arange(codes.size)))
    rng = np.random.default_rng(seed)
    draws = {m: np.full(resamples, NAN) for m in point}
    for b in range(resamples):
        idx = np.concatenate([rows[members[j]] for members in strata.values()
                              for j in rng.integers(0, len(members), len(members))])
        for m, v in fn(idx).items():
            draws[m][b] = v
    return _percentile_ci(point, draws, method="guid-cluster outcome-stratified percentile bootstrap",
                          resamples=resamples, seed=seed, confidence=confidence, n=len(names))


def _percentile_ci(point: Mapping[str, float], draws: Mapping[str, np.ndarray], *, method: str, resamples: int,
                   seed: int, confidence: float, n: int) -> Dict[str, Any]:
    """:func:`bootstrap_ci`'s return shape from point values and per-metric draw vectors."""
    a = (1.0 - confidence) / 2.0
    bad = {m: ~np.isfinite(d) for m, d in draws.items()}
    metrics = {}
    for m, d in draws.items():
        ok = d[~bad[m]]
        lo, hi = np.quantile(ok, [a, 1 - a]) if ok.size else (NAN, NAN)
        metrics[m] = {"value": float(point[m]), "ci_lo": float(lo), "ci_hi": float(hi), "n_undefined": int(bad[m].sum())}
    return {"metrics": metrics, "record": {
        "method": method, "resamples": int(resamples), "seed": int(seed), "confidence": float(confidence), "n": int(n),
        "n_dropped": int(np.any(list(bad.values()), axis=0).sum()) if bad else 0,
    }}


def _ranked(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(order, starts, values)``: descending sort, start of each distinct-score group in it, the distinct scores."""
    order = np.argsort(-s, kind="stable")
    ss = s[order]
    starts = np.flatnonzero(np.concatenate(([True], ss[1:] != ss[:-1])))
    return order, starts, ss[starts]


def _curve(y: np.ndarray, W: np.ndarray, order: np.ndarray, starts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per row of weights ``W`` (draws, n): weighted positives and negatives of each distinct-score
    group, highest score first (:func:`_ranked`)."""
    Wo, pos = W[:, order], y[order] == 1
    return (np.add.reduceat(np.where(pos, Wo, 0.0), starts, axis=1),
            np.add.reduceat(np.where(pos, 0.0, Wo), starts, axis=1))


def _rank_stats(P: np.ndarray, N: np.ndarray, alpha: Optional[float] = None) -> Dict[str, np.ndarray]:
    """AUROC (Mann-Whitney, ties 1/2), AP and McClish pAUC@alpha per row of :func:`_curve` output;
    with unit weights exactly ``roc_auc_score``, ``average_precision_score``, ``roc_auc_score(max_fpr=alpha)``."""
    TP, FP = P.cumsum(1), N.cumsum(1)
    pt, nt = TP[:, -1:], FP[:, -1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {"auroc": (N * (TP - 0.5 * P)).sum(1) / (pt * nt)[:, 0],
               "auprc": (P * np.where(TP + FP > 0, TP / (TP + FP), 0.0)).sum(1) / pt[:, 0]}
        if alpha is not None:  # trapezoids of the ROC up to fpr = alpha, the crossing one cut linearly
            z = np.zeros_like(pt)
            x, v = np.concatenate([z, FP / nt], axis=1), np.concatenate([z, TP / pt], axis=1)
            x0, x1, y0, y1 = x[:, :-1], x[:, 1:], v[:, :-1], v[:, 1:]
            xe = np.minimum(x1, alpha)
            ye = y0 + np.where(x1 > x0, (xe - x0) / (x1 - x0), 1.0) * (y1 - y0)
            part = np.where(x0 < alpha, (xe - x0) * (y0 + ye) / 2.0, 0.0).sum(1)
            out[f"pauc@{alpha:g}"] = 0.5 * (1.0 + (part - alpha ** 2 / 2.0) / (alpha - alpha ** 2 / 2.0))
    return out


def _counts(m: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """(b, m) integer multiplicities of m units resampled with replacement b times (each row sums to m)."""
    d = rng.integers(0, m, (b, m)) + m * np.arange(b)[:, None]
    return np.bincount(d.ravel(), minlength=b * m).reshape(b, m)


def _unit_draws(y: np.ndarray, units: Any, resamples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(M, codes)``: outcome-stratified unit multiplicities (resamples, n_units) and each row's unit
    code. :func:`bootstrap_ci`'s scheme (units stratified by their label set, drawn with replacement
    within the stratum) expressed as weights."""
    codes, names = pd.factorize(np.asarray(units).reshape(-1), sort=True)
    lab = pd.DataFrame({"u": codes, "y": y}).groupby("u")["y"].agg(["min", "max"])
    strata = (2 * lab["min"] + lab["max"]).to_numpy()
    rng, M = np.random.default_rng(seed), np.zeros((resamples, len(names)), np.float32)
    for k in np.unique(strata):
        u = np.flatnonzero(strata == k)
        M[:, u] = _counts(u.size, resamples, rng)
    return M, codes


def _chunks(resamples: int, n: int, cells: int = 4_000_000) -> list:
    """Draw slices of at most ``cells`` draw x row weights (bounds memory)."""
    step = max(1, cells // max(n, 1))
    return [slice(b, b + step) for b in range(0, resamples, step)]


def bootstrap_auroc(y: Any, s: Any, units: Any = None, B: int = 2000, seed: int = 0, *,
                    confidence: float = 0.95) -> Dict[str, Any]:
    """Vectorised :func:`bootstrap_ci` of AUROC.

    Same resampling scheme (outcome-stratified cluster percentile bootstrap over ``units``), but each
    draw is a row of unit multiplicities and AUROC the weighted Mann-Whitney statistic: each positive
    weighs the cumulative weight of the negatives below it (ties 1/2), read off one sort of the
    negatives. B draws cost O(B n) array work instead of B sklearn calls. ``units=None``: every row is
    its own unit (GUID level), so each class's multiplicities are drawn directly (they are
    exchangeable within the class). Needs both classes.

    Returns:
        :func:`bootstrap_ci`'s ``{metrics: {"auroc": {value, ci_lo, ci_hi, n_undefined}}, record}``.
    """
    y, s = _yl(y, s)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    neg = neg[np.argsort(s[neg], kind="stable")]
    lt, le = np.searchsorted(s[neg], s[pos], side="left"), np.searchsorted(s[neg], s[pos], side="right")

    def auc(Wp: np.ndarray, Wn: np.ndarray) -> np.ndarray:
        C = np.concatenate([np.zeros((len(Wn), 1), Wn.dtype), np.cumsum(Wn, axis=1)], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (Wp * (C[:, lt] + C[:, le])).sum(1, dtype=np.float64) / (2.0 * Wp.sum(1, dtype=np.float64) * C[:, -1])

    point = {"auroc": float(auc(np.ones((1, pos.size)), np.ones((1, neg.size)))[0])}
    draws = np.empty(B)
    if units is None:
        rng, n = np.random.default_rng(seed), y.size
        for sl in _chunks(B, y.size):
            b = len(draws[sl])
            draws[sl] = auc(_counts(pos.size, b, rng), _counts(neg.size, b, rng))
    else:
        M, codes = _unit_draws(y, units, B, seed)
        n = M.shape[1]
        for sl in _chunks(B, y.size):
            draws[sl] = auc(M[sl][:, codes[pos]], M[sl][:, codes[neg]])
    return _percentile_ci(point, {"auroc": draws}, method="guid-cluster outcome-stratified percentile bootstrap "
                          "(vectorised weighted Mann-Whitney)", resamples=B, seed=seed, confidence=confidence, n=n)


def bootstrap_rank(y: Any, s: Any, units: Any, *, alpha: float, resamples: int, seed: int,
                   confidence: float = 0.95) -> Dict[str, Any]:
    """Vectorised :func:`bootstrap_ci` of AUROC, AUPRC and pAUC@alpha over cluster ``units``: each
    draw a row of unit multiplicities (:func:`_unit_draws`, so its AUROC draws are
    :func:`bootstrap_auroc`'s), the metrics the weighted rank statistics of one shared sort
    (:func:`_rank_stats`, exactly sklearn's with unit weights). Needs both classes."""
    y, s = _yl(y, s)
    order, starts, _ = _ranked(s)
    M, codes = _unit_draws(y, units, resamples, seed)
    point = {m: float(v[0]) for m, v in _rank_stats(*_curve(y, np.ones((1, y.size)), order, starts), alpha).items()}
    draws = {m: np.empty(resamples) for m in point}
    for sl in _chunks(resamples, y.size):
        for m, v in _rank_stats(*_curve(y, M[sl][:, codes], order, starts), alpha).items():
            draws[m][sl] = v
    return _percentile_ci(point, draws, method="cluster outcome-stratified percentile bootstrap (vectorised rank "
                          "statistics)", resamples=resamples, seed=seed, confidence=confidence, n=M.shape[1])


def rate_ci(num: Any, den: Any, y: Any, units: Any = None, *, resamples: int, seed: int,
            confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ci_lo, ci_hi, n_undefined)`` of each row's rate ``num[c].sum() / den[c].sum()`` (``(C, n)`` bool over
    rows, ``num`` within ``den``).

    Wilson (§11.7) where every cluster of ``units`` (None: each row its own) holds at most one ``den``
    row, i.e. one decision per cluster; its ``n_undefined`` is NaN (no draws). Elsewhere the decisions are not
    independent, so that row falls back to the outcome-stratified cluster percentile bootstrap (:func:`_unit_draws`
    on labels ``y``) of the ratio over all clusters; a draw with an empty denominator is undefined, left out of the
    percentiles and counted in ``n_undefined`` (§11.7: never silently dropped). NaN CIs where ``den`` is empty.
    """
    num, den = np.atleast_2d(num), np.atleast_2d(den)
    lo, hi = wilson(num.sum(1), den.sum(1), confidence)
    nu = np.full(len(num), NAN)
    if units is None:
        return lo, hi, nu
    codes = pd.factorize(np.asarray(units).reshape(-1), sort=True)[0]
    o = np.argsort(codes, kind="stable")
    starts = np.flatnonzero(np.r_[True, codes[o][1:] != codes[o][:-1]])

    def per_unit(a: np.ndarray) -> np.ndarray:
        return np.add.reduceat(a[:, o].astype(np.float32), starts, axis=1) if a.shape[1] else a[:, :0]

    D = per_unit(den)
    multi = (D > 1).any(1)
    if multi.any():
        M = _unit_draws(np.asarray(y).reshape(-1), units, resamples, seed)[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            d = (per_unit(num[multi]) @ M.T) / (D[multi] @ M.T)
        a = (1.0 - confidence) / 2.0
        lo[multi], hi[multi] = np.nanquantile(d, [a, 1.0 - a], axis=1)
        nu[multi] = np.isnan(d).sum(1)
    return lo, hi, nu


def roc_points(y: Any, s: Any) -> Dict[str, np.ndarray]:
    """``roc_curve(drop_intermediate=False)`` that also takes ``-inf`` scores (R3's not-yet-monitored
    GUIDs): ``fpr, tpr, thr`` (from ``+inf`` down) plus ``precision`` = TP/(TP+FP), NaN before any alarm."""
    y, s = _yl(y, s)
    order, starts, thr = _ranked(s)
    P, N = _curve(y, np.ones((1, y.size)), order, starts)
    TP, FP = np.r_[0.0, P[0].cumsum()], np.r_[0.0, N[0].cumsum()]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"fpr": FP / FP[-1], "tpr": TP / TP[-1], "thr": np.r_[np.inf, thr], "precision": TP / (TP + FP)}


def roc_band(y: Any, s: Any, units: Any, grid: np.ndarray, *, resamples: int, seed: int,
             confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Pointwise percentile band of the ROC step-read on ``grid`` (:func:`tpr_at`) under the
    outcome-stratified cluster bootstrap, vectorised as :func:`bootstrap_auroc` (R5)."""
    y, s = _yl(y, s)
    order, starts, _ = _ranked(s)
    M, codes = _unit_draws(y, units, resamples, seed)
    V = np.empty((resamples, grid.size))
    for sl in _chunks(resamples, y.size):
        P, N = _curve(y, M[sl][:, codes], order, starts)
        z = np.zeros((len(P), 1))
        TP, FP = np.c_[z, P.cumsum(1)], np.c_[z, N.cumsum(1)]
        V[sl] = [tpr_at(f, t, grid) for f, t in zip(FP / FP[:, -1:], TP / TP[:, -1:])]
    a = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(V, [a, 1 - a], axis=0)
    return lo, hi


def metric_rows(
    values: Mapping[str, float], ci: Optional[Mapping[str, Mapping[str, float]]] = None, **keys: Any
) -> list:
    """§11.9 long-table rows, one per metric. ``values`` is ``{metric: value}`` (e.g. from
    :func:`threshold_free`); ``ci`` is ``bootstrap_ci(...)['metrics']``; ``keys`` fill the other
    :data:`METRIC_COLUMNS` (unset ones are None, ``point`` is ``n/a``). Build with
    ``pd.DataFrame(rows, columns=METRIC_COLUMNS)``.
    """
    unknown = set(keys) - set(METRIC_COLUMNS)
    if unknown:
        raise ValueError(f"not §11.9 columns: {sorted(unknown)}")
    base, ci = dict.fromkeys(METRIC_COLUMNS) | {"point": "n/a"} | keys, ci or {}
    return [base | {"metric": m, "value": float(v), "ci_lo": ci.get(m, {}).get("ci_lo", NAN),
                    "ci_hi": ci.get(m, {}).get("ci_hi", NAN), "n_boot_undefined": ci.get(m, {}).get("n_undefined")}
            for m, v in values.items()]


# --- analyses ---
#
# The P2 analysis registry (§11.12 blocks C, R1/R9, T1/T2, B1, plus the §11.4/§11.7 metric rows),
# run fail-soft by :func:`evaluate` under the VAE's ``Report.step`` (§11.14). Evaluation reads only
# ``<run>/predictions/*``, ``<run>/cohort/*`` and the resolved config (manifest.json optionally, for
# provenance). Every table lands in ``<run>/evaluation/tables/``; ``report.py`` draws every figure
# from those tables afterwards (figures are never rendered here). Nothing below imports torch at
# module load; ``evaluate`` pulls the VAE step heartbeat (``report_seam``) lazily.

import json  # noqa: E402
import platform  # noqa: E402
from datetime import datetime  # noqa: E402
from importlib.metadata import version  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from loguru import logger  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from teb_vae.classifier.thresholds import basis_frame, drop_last_minutes  # noqa: E402
from teb_vae.lag_attn.eval.report import Report, build_manifest, json_safe  # noqa: E402

SPLITS = ("val", "test")
VAL_LABEL = "optimistic (used for selection)"
BASELINES = ("probe", "shortcut", "shuffled")
NOIND = "noind"  # model_id of the `no_indicator` ablation (§7.3.4), which verify criterion 10 accepts
SHUFFLED_WARN_AUROC = 0.60  # §10.9.3 / §11.15 criterion 7: a fixed leak flag, not a config key
CLASSES = ("healthy", "acidosis", "hie")
#: Tables §11.13 lists that P2 writes; verify criterion 11 requires each non-empty.
P2_TABLES = ("metrics", "thresholds", "roc_points")
#: FPR grid for the pooled bootstrap band and the vertical average (R1).
ROC_GRID = np.linspace(0.0, 1.0, 101)
#: Denominator of each metric-type basis (§11.5.1).
_DENOM = {"instantaneous": "bin_present", "committed_cumulative": "available", "committed_overall": "all"}
#: Single proportions get Wilson intervals (§11.7): metric -> (hits, misses).
_PROP = {"sens": ("tp", "fn"), "spec": ("tn", "fp"), "fpr": ("fp", "tn"), "ppv": ("tp", "fp"), "npv": ("tn", "fn")}
_DERIVED = ("bal_acc", "mcc", "lr_pos", "lr_neg", "f1")
_HEADLINE = ("auroc", "auprc", "sens", "spec", "fpr", "ppv", "npv")


def classifier_cfg(cfg: Any) -> Dict[str, Any]:
    """The ``classifier`` block as a plain dict, from a pydantic model or a (dumped) dict."""
    d = cfg.model_dump(mode="json") if hasattr(cfg, "model_dump") else dict(cfg)
    return d.get("classifier", d)


def primary_alpha(ev: Mapping[str, Any]) -> float:
    """alpha of the primary policy (pAUC range), 0.3 when it has none."""
    return next((p.get("alpha") for p in ev["thresholds"] if p["id"] == ev["primary_policy"]), None) or 0.3


def _sel(df: pd.DataFrame, **eq: Any) -> pd.DataFrame:
    """Rows where every ``column == value``; empty when ``df`` lacks a column (empty tables)."""
    if not set(eq) <= set(df.columns):
        return df.iloc[:0]
    mask = np.ones(len(df), dtype=bool)
    for k, v in eq.items():
        mask &= (df[k] == v).to_numpy()
    return df[mask]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.is_file() else None


def load_context(run_dir: Any, cfg: Any) -> SimpleNamespace:
    """Read the CONTRACT evaluation inputs; refuse predictions written under another config. ``thr`` holds each
    unit's binary levels; its per-class OvR thresholds (``thresholds.json["ovr"]``, §11.3 T5) are ``thr_ovr``.

    Raises:
        ValueError: If ``predictions/provenance.json``'s ``config_digest`` differs from the digest
            of ``<run>/config.resolved.yaml``.
    """
    from teb_vae.classifier.config import digest, load

    run_dir = Path(run_dir)
    pred = run_dir / "predictions"
    prov = json.loads((pred / "provenance.json").read_text())
    want = digest(load(run_dir / "config.resolved.yaml"))
    if prov.get("config_digest") != want:
        raise ValueError(f"predictions were written under config digest {str(prov.get('config_digest'))[:12]}, "
                         f"but {run_dir / 'config.resolved.yaml'} digests to {want[:12]}; re-run predict")
    thr = json.loads((pred / "thresholds.json").read_text())
    return SimpleNamespace(
        run_dir=run_dir, cfg=classifier_cfg(cfg), prov=prov, digest=want, rows=[], frames=[], inclusion=[], sanity={},
        seg=pd.read_parquet(pred / "segments.parquet"), guids=pd.read_parquet(pred / "guids.parquet"),
        thr={k: {lvl: v for lvl, v in d.items() if lvl != "ovr"} for k, d in thr.items()},
        thr_ovr={k: d["ovr"] for k, d in thr.items() if "ovr" in d},
    )


def _models(ctx: SimpleNamespace) -> list:
    return sorted(ctx.guids[["model_id", "seed"]].drop_duplicates().itertuples(index=False, name=None))


def _keys(ctx: SimpleNamespace, m: str, sd: str, **extra: Any) -> Dict[str, Any]:
    lab = ctx.cfg["labels"]
    return dict(run_id=ctx.prov.get("run_id"), model_id=m, seed=sd, task=lab["task"], head=lab["head"],
                label_strategy=lab["strategy"]) | extra


def cluster(df: pd.DataFrame) -> pd.Series:
    """Bootstrap cluster of each row: ``patient`` (``data.patient_map``, else the GUID) when the table
    carries it (CONTRACT), else the GUID (runs predicted before the column existed)."""
    return df["patient"].fillna(df["guid"]).astype(str) if "patient" in df.columns else df["guid"].astype(str)


def segments(ctx: SimpleNamespace, m: str, sd: str, split: str) -> pd.DataFrame:
    """Prediction segments of one model/seed/split after ``eval.exclude_last_min``, through
    :func:`drop_last_minutes`, the filter threshold selection applied, so every evaluation population
    (P2 and P4) is the one its threshold was chosen on. The drop is recorded per fold (L14), once."""
    cache = ctx.__dict__.setdefault("seg_cache", {})
    if (m, sd, split) not in cache:
        seg, L = _sel(ctx.seg, model_id=m, seed=sd, split=split), ctx.cfg["eval"]["exclude_last_min"]
        kept = cache[(m, sd, split)] = drop_last_minutes(seg, L)
        for fold, f in seg.groupby("fold"):
            n_in = kept.loc[kept["fold"] == fold, "guid"].nunique()
            _incl(ctx, "exclude_last_min", m, sd, split, str(fold), None, n_in, f["guid"].nunique() - n_in,
                  {"segments_dropped": int(len(f) - (kept["fold"] == fold).sum()), "exclude_last_min": L})
    return cache[(m, sd, split)]


def level_units(ctx: SimpleNamespace, m: str, sd: str, split: str) -> Dict[str, pd.DataFrame]:
    """Threshold-free populations ``{level: (fold, unit, guid, patient, y, score, shared_test)}``.

    GUID level scores ``score_final_cal`` (the offline score, all segments); segment level
    ``logit_seg_cal`` on ``in_eval_window`` rows of :func:`segments`, skipped for a model without
    segment scores (the shortcut). ``unit`` is the row identity pooling de-duplicates on; ``patient``
    the bootstrap cluster (:func:`cluster`).
    """
    g = _sel(ctx.guids, model_id=m, seed=sd, split=split)
    out = {"guid": g.assign(unit=g["guid"], score=g["score_final_cal"])}
    s = segments(ctx, m, sd, split)
    s = s[s["in_eval_window"].astype(bool)]
    if s["logit_seg_cal"].notna().any():
        out["segment"] = s.assign(unit=s["guid"] + "@" + s["seg_pos"].astype(str), score=s["logit_seg_cal"])
    return {k: v.assign(patient=cluster(v))[["fold", "unit", "guid", "patient", "y", "score", "shared_test"]]
            for k, v in out.items()}


def pool_rows(units: pd.DataFrame, policy: str, split: str) -> pd.DataFrame:
    """Pooled OOF rows, de-duplicated like :func:`pooled_confusion` (lowest fold kept, or dropped).

    On val the ``shared_test`` flag says nothing, so any unit repeated across folds counts as shared.
    """
    u = units.sort_values("fold", kind="stable")
    shared = u["unit"].duplicated(keep=False) if split == "val" else u["shared_test"].astype(bool)
    return u[~shared] if policy == "exclude" else u[~(shared & u["unit"].duplicated())]


def _tf_rows(u: pd.DataFrame, alpha: float, boot: Mapping[str, Any], **keys: Any) -> list:
    """Threshold-free rows of one population; patient-cluster bootstrap CIs on the ranking metrics."""
    y, s = u["y"].to_numpy(np.int64), u["score"].to_numpy(np.float64)
    ci = (bootstrap_rank(y, s, u["patient"], alpha=alpha, **boot)["metrics"] if 0 < y.sum() < y.size else None)
    return metric_rows(threshold_free(y, s, alpha=alpha), ci, metric_type="threshold_free", denominator="n/a",
                       n_pos=int(y.sum()), n_neg=int(y.size - y.sum()), **keys)


def _prop_ci(fire: np.ndarray, pos: np.ndarray, units: Any, alpha: Optional[float],
             boot: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    """CIs of the single proportions (:data:`_PROP`) of one population's decisions: :func:`rate_ci`
    (Wilson; the patient-cluster bootstrap where a patient holds two of the rows)."""
    lo, hi, nu = rate_ci(np.stack([fire & pos, ~fire & ~pos, fire & ~pos, fire & pos, ~fire & ~pos]),
                         np.stack([pos, ~pos, ~pos, fire, ~fire]), pos, units, **boot)
    ci = {m: {"ci_lo": float(a), "ci_hi": float(b), "n_undefined": float(u)} for m, a, b, u in zip(_PROP, lo, hi, nu)}
    if alpha is not None:
        ci["fpr_overshoot"] = ci["fpr"] | {k: ci["fpr"][k] - alpha for k in ("ci_lo", "ci_hi")}
    return ci


def _with_overshoot(vals: Dict[str, Any], alpha: Optional[float]) -> Dict[str, Any]:
    return vals | ({"fpr_overshoot": vals["fpr"] - alpha} if alpha is not None else {})


def _rates(fire: np.ndarray, pos: np.ndarray, i: Any = slice(None)) -> Dict[str, float]:
    """:func:`confusion_rates` of the decisions ``fire`` against ``pos`` on the rows ``i``."""
    f, p = fire[i], pos[i]
    return confusion_rates((f & p).sum(), (f & ~p).sum(), (~f & ~p).sum(), (~f & p).sum())


def _derived(tp: Any, fp: Any, tn: Any, fn: Any) -> Dict[str, np.ndarray]:
    """:data:`_DERIVED` of :func:`confusion_rates` on arrays of (weighted) counts, with its NaN and 0 rules."""
    tp, fp, tn, fn = (np.asarray(x, np.float64) for x in (tp, fp, tn, fn))
    P, N, pp, pn = tp + fn, tn + fp, tp + fp, tn + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        sens, spec, den = np.where(P > 0, tp / P, NAN), np.where(N > 0, tn / N, NAN), np.sqrt(pp * P * N * pn)
        return {"bal_acc": (sens + spec) / 2.0,
                "mcc": np.where((P == 0) | (N == 0), NAN, np.where(den > 0, (tp * tn - fp * fn) / den, 0.0)),
                "lr_pos": sens / (1.0 - spec), "lr_neg": (1.0 - sens) / spec,
                "f1": np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)}


def _rate_cis(fire: np.ndarray, pos: np.ndarray, units: Any, alpha: Optional[float],
              boot: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    """CIs of one population's decisions: :func:`_prop_ci` on the proportions, the outcome-stratified patient-cluster
    bootstrap on the derived rates (:data:`_DERIVED`), vectorised: each draw's confusion counts are its unit
    multiplicities (:func:`_unit_draws`) times the units' counts, so no statistic runs once per draw in Python."""
    M, codes = _unit_draws(pos.astype(np.int64), units, boot["resamples"], boot["seed"])
    counts = [M @ np.bincount(codes, weights=x, minlength=M.shape[1]) for x in
              (fire & pos, fire & ~pos, ~fire & ~pos, ~fire & pos)]
    point = {k: v for k, v in _rates(fire, pos).items() if k in _DERIVED}
    return _prop_ci(fire, pos, units, alpha, boot) | _percentile_ci(
        point, _derived(*counts), method="patient-cluster outcome-stratified percentile bootstrap (vectorised counts)",
        resamples=boot["resamples"], seed=boot["seed"], confidence=0.95, n=M.shape[1])["metrics"]


def _thr_rows(pop: pd.DataFrame, thr: float, alpha: Optional[float], boot: Mapping[str, Any], **keys: Any) -> list:
    """Per-fold thresholded rows with :func:`_rate_cis`."""
    s, pos = pop["score"].to_numpy(np.float64), pop["y"].to_numpy(np.int64) == 1
    fire = s > thr
    return metric_rows(_with_overshoot(_rates(fire, pos), alpha) | {"threshold": thr},
                       _rate_cis(fire, pos, pop["patient"], alpha, boot), threshold=thr, n_pos=int(pos.sum()),
                       n_neg=int((~pos).sum()), **keys)


def _pooled_ci(flagged: pd.DataFrame, thr_by_fold: Mapping[int, float], policy: str, split: str,
               alpha: Optional[float], boot: Mapping[str, Any], *, derived: bool = True) -> Dict[str, Dict[str, float]]:
    """:func:`_rate_cis` of the pooled OOF rows :func:`pooled_confusion` counts (:func:`pool_rows`), each
    fired at its own fold's threshold, so pooled derived rates carry CIs as the per-fold ones do; ``derived=False``:
    the proportions only (:func:`_prop_ci`, for callers that report no derived rate)."""
    kept = pool_rows(flagged, policy, split)
    fire = kept["score"].to_numpy(np.float64) > kept["fold"].map(thr_by_fold).to_numpy(np.float64)
    return (_rate_cis if derived else _prop_ci)(fire, kept["y"].to_numpy() == 1, kept["patient"], alpha, boot)


def policy_populations(ctx: SimpleNamespace, m: str, sd: str, split: str) -> Dict[Tuple[str, str], Tuple[dict, pd.DataFrame]]:
    """``{(level, policy_id): ({fold: thresholds entry}, population with fold, shared_test, patient)}``:
    each policy's basis population (§11.3) on :func:`segments`, the segments its threshold was chosen on."""
    ev, out = ctx.cfg["eval"], {}
    for key, per_level in ctx.thr.items():
        km, ks, kf = key.split("|")
        if (km, ks) != (m, sd):
            continue
        fold = int(kf)
        g = _sel(ctx.guids, model_id=m, seed=sd, fold=fold, split=split)
        s = _sel(segments(ctx, m, sd, split), fold=fold)
        shared, pat = g.set_index("guid")["shared_test"], cluster(g).set_axis(g["guid"])
        for level, pols in per_level.items():
            for pid, e in pols.items():
                if "skipped" in e:
                    continue
                pop = basis_frame(s, g, e["basis"], at=e["at"], bin_h=ev["bin_h"], axis=e.get("axis") or "to_delivery",
                                  staleness_h=ev["snapshot_max_staleness_h"])
                ents, frames = out.setdefault((level, pid), ({}, []))
                ents[fold] = e
                frames.append(pop.assign(fold=fold, shared_test=pop["guid"].map(shared).fillna(False).astype(bool),
                                         patient=pop["guid"].map(pat).fillna(pop["guid"])))
    return {k: (e, pd.concat(f, ignore_index=True)) for k, (e, f) in out.items()}


def _three_class_rows(ctx: SimpleNamespace, m: str, sd: str, split: str, alpha: float, policy: str,
                      boot: Mapping[str, Any]) -> list:
    """§11.4 3-class rows of one model/seed/split, per fold and pooled (:func:`pool_rows`), at GUID level and, for a
    model :func:`level_units` scores at segment level, at segment level: every ``in_eval_window`` segment of
    :func:`segments` with its row's probabilities (the output after segment n: segment-local in segment scope, the
    online output in sequence scope). Probabilities are the calibrated ``p_c0..2_cal`` (a binary task's aux head: its
    raw ``p_c0..2``, ``head='aux3'``) against ``class_code - 1`` (0 healthy, 1 acidosis, 2 hie). ``metric_type``
    threshold_free, ``denominator`` n/a, ``task`` the run's, ``head`` the one that made the probabilities. Metric names:

    * probabilities (``policy_id`` null): ``auroc_ovr_c{k}``, ``auprc_ovr_c{k}``, ``auroc_macro`` (OvR),
      ``auroc_hand_till`` (OvO), ``rps`` (:func:`multiclass`); a CORAL head (finite ``ord_score``) adds
      ``ordinal_auroc_adverse`` / ``ordinal_auroc_hie``, the AUROC of g at the cut-points y >= 1 and y >= 2 (X8);
    * argmax (``policy_id`` = ``policy_basis`` = ``argmax``): ``confusion_t{i}_p{j}`` (units of true class i
      predicted j; pooled = summed folds), ``recall_c{k}``, ``precision_c{k}``, ``f1_c{k}``, ``bal_acc``,
      ``macro_f1``, ``qwk``;
    * binary collapses of the 3-class output (``policy_id`` null), ``<collapse>/<name>`` for every
      :func:`threshold_free` name: ``adverse_vs_healthy/*`` (score ``logit(p_c1 + p_c2)``, the collapsed adverse
      score) and ``hie_vs_rest/*`` (score ``logit p_c2``), with their own ``n_pos``/``n_neg``.

    The other rows' ``n_pos``/``n_neg`` count adverse/healthy units. CIs: :func:`three_class_ci` (patient clusters)
    on every metric but the confusion counts (and, as :func:`_tf_rows`, the collapses' non-ranking metrics). Empty
    for a model without class probabilities.
    """
    lab, rows = ctx.cfg["labels"], []
    head = lab["head"] if lab["task"] == "three_class" else "aux3"
    g = _sel(ctx.guids, model_id=m, seed=sd, split=split)
    pops = {"guid": g.assign(unit=g["guid"])}
    s = segments(ctx, m, sd, split)
    s = s[s["in_eval_window"].astype(bool)]
    if s["logit_seg_cal"].notna().any():
        pops["segment"] = s.assign(unit=s["guid"] + "@" + s["seg_pos"].astype(str))
    for level, u in pops.items():
        cols = [f"p_c{k}_cal" for k in range(3)]
        cols = cols if set(cols) <= set(u.columns) else [f"p_c{k}" for k in range(3)]
        if not set(cols) <= set(u.columns) or not len(u) or u[cols].isna().any(axis=None):
            continue
        u = u.assign(patient=cluster(u))
        for fold, f in [*((str(k), x) for k, x in u.groupby("fold")), ("pooled", pool_rows(u, policy, split))]:
            y, P = f["class_code"].to_numpy(np.int64) - 1, f[cols].to_numpy(np.float64)
            o = f["ord_score"].to_numpy(np.float64) if "ord_score" in f and f["ord_score"].notna().all() else None
            k = _keys(ctx, m, sd, split=split, level=level, fold=fold, head=head, metric_type="threshold_free",
                      denominator="n/a", eval_window=lab["eval_window"] if level == "segment" else None)
            mc, cm = multiclass(y, P), confusion_matrix(y, P.argmax(1), labels=[0, 1, 2])
            ci = three_class_ci(y, P, f["patient"], alpha=alpha, ords=o, **boot)["metrics"]
            n = dict(n_pos=int((y > 0).sum()), n_neg=int((y == 0).sum()))
            prob = {x: v for x, v in mc.items() if x.startswith(("auroc", "auprc")) or x == "rps"}
            if o is not None:
                prob |= {f"ordinal_auroc_{name}": float(roc_auc_score(pos, o)) if 0 < pos.sum() < pos.size else NAN
                         for name, pos in (("adverse", y > 0), ("hie", y == 2))}
            argmax = {x: v for x, v in mc.items() if x not in prob} | {
                f"confusion_t{i}_p{j}": cm[i, j] for i in range(3) for j in range(3)}
            rows += metric_rows(prob, ci, **k, **n) + metric_rows(argmax, ci, policy_id="argmax", policy_basis="argmax",
                                                                  **k, **n)
            for name, pos, q in (("adverse_vs_healthy", y > 0, P[:, 1] + P[:, 2]), ("hie_vs_rest", y == 2, P[:, 2])):
                tf = threshold_free(pos, _logit(np.clip(q, 1e-12, 1 - 1e-12)), alpha=alpha)
                rows += metric_rows({f"{name}/{x}": v for x, v in tf.items()}, ci, n_pos=int(pos.sum()),
                                    n_neg=int((~pos).sum()), **k)
    return rows


def run_metrics(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """§11.4/§11.7 rows for val and test, per fold and pooled, per model/seed, GUID and segment level.

    Threshold-free with cluster-bootstrap CIs; per policy (on its own basis population) per fold with
    Wilson + bootstrap CIs, and pooled OOF via :func:`pooled_confusion` under
    ``data.shared_test_policy``, plus the other shared-test policy as a sensitivity row
    (``subgroup='shared_test_policy'``; H3 light). Val rows are the optimistic, selection-side copy.
    A model with class probabilities adds the 3-class rows of :func:`_three_class_rows` (with CIs).
    """
    ev, policy = eval_config, ctx.cfg["data"]["shared_test_policy"]
    other = {"first_fold": "exclude", "exclude": "first_fold"}[policy]
    alpha, boot = primary_alpha(ev), dict(resamples=ev["bootstrap"]["resamples"], seed=ev["bootstrap"]["seed"])
    win, rows = ctx.cfg["labels"]["eval_window"], []
    for m, sd in _models(ctx):
        for split in SPLITS:
            for level, units in level_units(ctx, m, sd, split).items():
                k = _keys(ctx, m, sd, split=split, level=level, eval_window=win if level == "segment" else None)
                for fold, u in units.groupby("fold"):
                    rows += _tf_rows(u, alpha, boot, fold=str(fold), **k)
                rows += _tf_rows(pool_rows(units, policy, split), alpha, boot, fold="pooled", **k)
            rows += _three_class_rows(ctx, m, sd, split, alpha, policy, boot)
            for (level, pid), (ents, pop) in policy_populations(ctx, m, sd, split).items():
                e = next(iter(ents.values()))
                a, basis = e.get("alpha"), e["basis"]
                k = _keys(ctx, m, sd, split=split, level=level, policy_id=pid, policy_basis=basis,
                          metric_type=basis if basis in _DENOM else None, denominator=_DENOM.get(basis, "n/a"),
                          axis=None if e["at"] == "end" else e["axis"], t=NAN if e["at"] == "end" else float(e["at"]),
                          eval_window=win if level == "segment" else None)
                for fold, p in pop.groupby("fold"):
                    rows += _thr_rows(p, ents[fold]["threshold"], a, boot, fold=str(fold), **k)
                flagged = pop if split == "test" else pop.assign(shared_test=pop["unit"].duplicated(keep=False))
                thr_by_fold = {f: x["threshold"] for f, x in ents.items()}
                for pol, sub in ((policy, {}), (other, {"subgroup": "shared_test_policy", "subgroup_value": other})):
                    if split == "val" and sub:
                        continue
                    r = pooled_confusion(flagged.assign(guid=flagged["unit"]), thr_by_fold, pol, score="score")
                    vals = _with_overshoot({x: r[x] for x in r if x not in ("n_dropped", "shared_test_policy")}, a)
                    rows += metric_rows(vals, _pooled_ci(flagged, thr_by_fold, pol, split, a, boot), fold="pooled",
                                        n_pos=r["tp"] + r["fn"], n_neg=r["tn"] + r["fp"], **k, **sub)
    return {"rows": rows, **_meta(ctx), "plan": {
        "models": [list(x) for x in _models(ctx)], "splits": list(SPLITS), "pauc_alpha": alpha,
        "shared_test_policy": policy, "sensitivity_policy": other, "val": VAL_LABEL,
        "exclude_last_min": ctx.cfg["eval"]["exclude_last_min"],
        "intervals": {"proportions": "wilson 95% (one decision per patient; else the patient-cluster bootstrap)",
                      "ranking_and_derived": {"method": "patient-cluster outcome-stratified percentile bootstrap",
                                              **boot, "confidence": 0.95}},
        "caveat": "naive CV confidence intervals under-cover across folds (Bates 2024)"}}


def _meta(ctx: SimpleNamespace) -> Dict[str, Any]:
    g = ctx.guids.drop_duplicates("guid")
    return {"n_guids": int(len(g)), "composition": {str(k): int(v) for k, v in g["clinical_class"].value_counts().items()}}


def tpr_at(fpr: np.ndarray, tpr: np.ndarray, x: Any) -> np.ndarray:
    """Highest TPR reachable at FPR <= x on a ROC curve: a step read, never an interpolation (A14)."""
    return tpr[np.searchsorted(fpr, x, side="right") - 1]


def vertical_average(curves: Sequence[Tuple[np.ndarray, np.ndarray]], grid: np.ndarray = ROC_GRID) -> pd.DataFrame:
    """Per-fold ROC read on ``grid``: mean, SD (ddof 1), min, max and ``n_folds`` per point."""
    V = np.array([tpr_at(f, t, grid) for f, t in curves]).reshape(len(curves), grid.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = V.std(0, ddof=1) if len(V) > 1 else np.full(grid.size, NAN)
    return pd.DataFrame({"fpr": grid, "tpr": V.mean(0), "tpr_sd": sd, "tpr_min": V.min(0), "tpr_max": V.max(0),
                         "n_folds": len(V)})


def _roc(y: Any, s: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return roc_curve(y, s, drop_intermediate=False)  # dropping collinear points would break the step read


def run_R1(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """R1: GUID final-score ROC per fold, pooled (+ patient-cluster bootstrap band on ``ROC_GRID``) and the
    vertical average across folds -> ``roc_points.parquet`` (fold, split, level, variant, fpr, tpr, thr, ...)."""
    policy, frames = ctx.cfg["data"]["shared_test_policy"], []
    boot = dict(resamples=eval_config["bootstrap"]["resamples"], seed=eval_config["bootstrap"]["seed"])
    for m, sd in _models(ctx):
        for split in SPLITS:
            u, key = level_units(ctx, m, sd, split)["guid"], dict(model_id=m, seed=sd, split=split, level="guid")
            curves = []
            for fold, f in u.groupby("fold"):
                if f["y"].nunique() == 2:
                    fpr, tpr, thr = _roc(f["y"], f["score"])
                    curves.append((fpr, tpr))
                    frames.append(pd.DataFrame({**key, "fold": str(fold), "variant": "score_final",
                                                "fpr": fpr, "tpr": tpr, "thr": thr}))
            p = pool_rows(u, policy, split)
            y, s = p["y"].to_numpy(np.int64), p["score"].to_numpy(np.float64)
            if not 0 < y.sum() < y.size:
                continue
            fpr, tpr, thr = _roc(y, s)
            lo, hi = roc_band(y, s, p["patient"], ROC_GRID, **boot)
            frames += [
                pd.DataFrame({**key, "fold": "pooled", "variant": "score_final", "fpr": fpr, "tpr": tpr, "thr": thr}),
                pd.DataFrame({**key, "fold": "pooled", "variant": "score_final:band", "fpr": ROC_GRID,
                              "tpr": tpr_at(fpr, tpr, ROC_GRID), "tpr_lo": lo, "tpr_hi": hi}),
            ]
            if curves:
                frames.append(vertical_average(curves).assign(**key, fold="mean", variant="score_final:vavg"))
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["model_id", "seed", "fold", "split", "level", "variant", "fpr", "tpr", "thr"])
    table.to_parquet(out_dir / "tables" / "roc_points.parquet", index=False)
    return {**_meta(ctx), "n_rows": int(len(table)), "plan": {
        "grid_points": int(ROC_GRID.size), "band": "patient-cluster bootstrap 95% (roc_band)",
        "vertical_average": "step read, no extrapolation (A14)"}}


def run_R9(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """R9: oracle test TPR at FPR <= f (``eval.report_tpr_at_fpr``), per fold and pooled; never a decision."""
    policy, rows = ctx.cfg["data"]["shared_test_policy"], []
    for m, sd in _models(ctx):
        u = level_units(ctx, m, sd, "test")["guid"]
        for fold, g in [*((str(f), x) for f, x in u.groupby("fold")), ("pooled", pool_rows(u, policy, "test"))]:
            if g["y"].nunique() < 2:
                continue
            fpr, tpr, _ = _roc(g["y"], g["score"])
            rows += metric_rows({f"tpr@fpr{f:g}": tpr_at(fpr, tpr, f) for f in eval_config["report_tpr_at_fpr"]},
                                fold=fold, n_pos=int(g["y"].sum()), n_neg=int((g["y"] == 0).sum()),
                                **_keys(ctx, m, sd, split="test", level="guid", policy_id="oracle", policy_basis="guid_final",
                                        metric_type="threshold_free", denominator="n/a"))
    return {"rows": rows, **_meta(ctx), "plan": {"label": "oracle (read off the test ROC; never used for decisions)"}}


T1_COLUMNS = ("model_id", "seed", "fold", "level", "policy_id", "class", "threshold", "basis", "axis", "at", "alpha",
              "delta", "method", "k", "n_pos", "n_neg", "val_sens", "val_fpr", "val_spec", "tie_frac", "fallback",
              "skipped")


def run_T1(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """T1: per fold x policy thresholds (from ``predictions/thresholds.json``) + cross-fold mean/SD/min/max; T5: the
    per-class OvR thresholds as extra rows with ``class`` k (0 healthy, 1 acidosis, 2 hie; null on binary rows)."""
    units = [(key, None, per_level) for key, per_level in ctx.thr.items()]
    units += [(key, int(k), per_level) for key, per in ctx.thr_ovr.items() for k, per_level in per.items()]
    recs = [{"model_id": m, "seed": sd, "fold": fold, "level": level, "policy_id": pid, "class": k, **e}
            for key, k, per_level in units for m, sd, fold in [key.split("|")]
            for level, pols in per_level.items() for pid, e in pols.items()]
    t = pd.DataFrame(recs).reindex(columns=T1_COLUMNS).astype({"class": "Int64"})
    t["at"] = t["at"].map(lambda v: None if v is None or v != v else str(v))
    for col in ("threshold", "alpha", "delta", "k", "val_sens", "val_fpr", "val_spec", "tie_frac"):
        t[col] = pd.to_numeric(t[col], errors="coerce").astype(float)
    t["fallback"] = t["fallback"].map(lambda v: bool(v) if isinstance(v, (bool, np.bool_)) else None)
    g = t.groupby(["model_id", "seed", "level", "policy_id", "class"], dropna=False)["threshold"]
    for stat in ("mean", "std", "min", "max"):
        t[f"thr_{stat}"] = g.transform(stat)
    t.to_parquet(out_dir / "tables" / "thresholds.parquet", index=False)
    return {**_meta(ctx), "plan": {"units": sorted(ctx.thr)}, "n_rows": int(len(t)),
            "n_skipped": int(t["skipped"].notna().sum()), "n_fallback": int((t["fallback"] == True).sum())}  # noqa: E712


def _rows_frame(ctx: SimpleNamespace) -> pd.DataFrame:
    """Every metric row so far: the dict rows plus the vectorised time-resolved frames."""
    frames = [f for f in (pd.DataFrame(ctx.rows, columns=list(METRIC_COLUMNS)), *ctx.frames) if len(f)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(METRIC_COLUMNS))
    return df.assign(n_boot_undefined=pd.to_numeric(df["n_boot_undefined"], errors="coerce"))


def run_T2(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """T2: FPR overshoot (test FPR - alpha) per fold and pooled, per FPR-cap policy; needs ``metrics``.

    An ``np_umbrella`` policy's record adds ``tolerance``, the 95% binomial tolerance on the pooled negatives
    (``binom.ppf(0.95, n_neg, alpha) / n_neg - alpha``), and the folds that fell back to empirical. Sanity
    ``np_overshoot`` (verify criterion 6) fails, naming the policy, when a pooled overshoot exceeds its tolerance,
    and is inconclusive where a fold fell back (no NP guarantee there).
    """
    if not ctx.rows:
        raise RuntimeError("T2 reads the metric rows; run the 'metrics' analysis first")
    df = _rows_frame(ctx)
    o = df[(df["metric"] == "fpr_overshoot") & (df["split"] == "test") & df["subgroup"].isna() & (df["point"] == "n/a")]
    alphas = {p["id"]: p["alpha"] for p in eval_config["thresholds"] if p.get("method") == "np_umbrella"}
    out, over, void = {}, [], []
    for (m, sd, lvl, pid), g in o.groupby(["model_id", "seed", "level", "policy_id"]):
        per, pooled = g[g["fold"] != "pooled"], g[g["fold"] == "pooled"]
        key, n = f"{m}|{sd}|{lvl}|{pid}", int(pooled["n_neg"].iloc[0]) if len(pooled) else 0
        rec = out[key] = {
            "per_fold": dict(zip(per["fold"], per["value"])), "mean": per["value"].mean(), "sd": per["value"].std(),
            "min": per["value"].min(), "max": per["value"].max(), "pooled": pooled["value"].iloc[0] if len(pooled) else None}
        if pid not in alphas or not n:
            continue
        a = alphas[pid]
        rec |= {"alpha": a, "n_neg": n, "tolerance": float(binom.ppf(0.95, n, a) / n - a),
                "fallback_folds": [f for f in per["fold"] if ctx.thr[f"{m}|{sd}|{f}"][lvl][pid].get("fallback")]}
        if rec["fallback_folds"]:
            void.append(f"{key} (folds {rec['fallback_folds']})")
        elif rec["pooled"] > rec["tolerance"] + 1e-9:  # FP counts: a 1-ulp gap at equality is not an overshoot
            over.append(f"{key}: overshoot {rec['pooled']:.3f} > {rec['tolerance']:.3f} (n_neg {n})")
    n_np = sum("tolerance" in r for r in out.values())
    sanity = {"np_overshoot": (
        {"verdict": "fail", "detail": f"NP-policy pooled test FPR above alpha + the 95% binomial tolerance: {over}"}
        if over else {"verdict": "inconclusive", "detail": "no np_umbrella policy with pooled test negatives"} if not n_np
        else {"verdict": "inconclusive", "detail": f"the NP guarantee is void where a fold fell back to empirical: {void}"}
        if void else {"verdict": "pass", "detail": f"all {n_np} NP-policy pooled test FPRs within alpha + the 95% "
                                                  "binomial tolerance"})}
    return {**_meta(ctx), "plan": {"definition": "test FPR - alpha on the policy's own basis population",
                                   "np_tolerance": "binom.ppf(0.95, n_neg, alpha) / n_neg - alpha, pooled test"},
            "overshoot": out, "sanity": sanity}


def run_B1(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """B1/B3: baseline table (AUROC with CI, sens at each policy) and the shortcut warning; needs ``metrics``."""
    if not ctx.rows:
        raise RuntimeError("B1 reads the metric rows; run the 'metrics' analysis first")
    df, warn_at = _rows_frame(ctx), ctx.cfg["baselines"]["shortcut_warn_auroc"]
    df = df[(df["level"] == "guid") & df["subgroup"].isna()]
    table, warnings, sanity = [], [], {}
    for m, sd in _models(ctx):
        if m not in BASELINES:
            continue
        au = _sel(df, model_id=m, seed=sd, metric="auroc")
        pooled = _sel(au, split="test", fold="pooled")
        test, val = _sel(au, split="test"), _sel(au, split="val")
        per, val_mean = test[test["fold"] != "pooled"]["value"], val[val["fold"] != "pooled"]["value"].mean()
        sens = _sel(df, model_id=m, seed=sd, metric="sens", split="test", fold="pooled")
        table.append({
            "model_id": m, "seed": sd,
            "auroc_test_pooled": pooled["value"].iloc[0] if len(pooled) else None,
            "auroc_test_ci": [pooled["ci_lo"].iloc[0], pooled["ci_hi"].iloc[0]] if len(pooled) else None,
            "auroc_test_fold_mean": per.mean(), "auroc_test_fold_sd": per.std(), "auroc_val_fold_mean": val_mean,
            "sens_test_pooled": {r.policy_id: {"value": r.value, "ci": [r.ci_lo, r.ci_hi]} for r in sens.itertuples()},
        })
        if m == "shortcut":
            over = bool(val_mean > warn_at)
            detail = f"shortcut (metadata-only) val AUROC {val_mean:.3f} {'>' if over else '<='} {warn_at}"
            verdict = "fail" if over else "pass" if np.isfinite(val_mean) else "inconclusive"
            sanity["shortcut_auroc"] = {"verdict": verdict, "detail": detail, "value": val_mean}
            if over:
                warnings.append(f"WARNING: {detail}: cohort construction lets length/missingness predict the label")
                logger.warning(warnings[-1])
        if m == "shuffled" and len(pooled):
            v, lo, hi = (float(pooled[k].iloc[0]) for k in ("value", "ci_lo", "ci_hi"))
            over = bool(v > SHUFFLED_WARN_AUROC or not lo <= 0.5 <= hi)
            detail = f"shuffled-label control pooled test AUROC {v:.3f} [{lo:.3f}, {hi:.3f}]"
            sanity["shuffled_auroc"] = {"verdict": "fail" if over else "pass", "detail": detail, "value": v}
            if over:
                warnings.append(f"WARNING: {detail}: > {SHUFFLED_WARN_AUROC} or CI excludes 0.5, a pipeline leak")
                logger.warning(warnings[-1])
    return {**_meta(ctx), "plan": {"baselines": list(BASELINES), "shortcut_warn_auroc": warn_at},
            "table": table, "warnings": warnings, "sanity": sanity}


# --- cohort (block C) ---

def infer_stride(seg: pd.DataFrame, manifest: Optional[Mapping[str, Any]] = None) -> Optional[float]:
    """The stride the cohort stage used (``manifest.cohort.stride_s``: inferred there, or ``data.stride_s``);
    for a run without that record, the mode of positive within-GUID epoch differences (s), None with no repeats."""
    recorded = ((manifest or {}).get("cohort") or {}).get("stride_s")
    if recorded is not None:
        return float(recorded)
    d = seg.sort_values("epoch_s").groupby(["fold", "split", "guid"])["epoch_s"].diff().round()
    d = d[d > 0]
    return float(d.mode().iloc[0]) if len(d) else None


def unique_cohort(run_dir: Any) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(segments incl. excluded, retained segments, retained GUIDs)``, one row per GUID/segment
    across folds (every fold re-partitions the same cohort)."""
    co = Path(run_dir) / "cohort"
    seg, gd = pd.read_parquet(co / "segments.parquet"), pd.read_parquet(co / "guids.parquet")
    kept = seg[~seg["excluded"].astype(bool)]
    # the reason is part of the key so a `duplicate_epoch` row survives next to the row it duplicates; any other
    # exclusion of a segment another split retains is split-local (data.train_epoch_min_s, L6), not the cohort's
    local = (seg["excluded"].astype(bool) & (seg["exclusion_reason"] != "duplicate_epoch")
             & pd.MultiIndex.from_frame(seg[["guid", "epoch_s"]]).isin(
                 pd.MultiIndex.from_frame(kept[["guid", "epoch_s"]])))
    return (seg[~local].drop_duplicates(["guid", "epoch_s", "exclusion_reason"]), kept.drop_duplicates(["guid", "epoch_s"]),
            gd[~gd["excluded"].astype(bool)].drop_duplicates("guid"))


def guid_timeline(seg: pd.DataFrame, gd: pd.DataFrame, stride: Optional[float]) -> pd.DataFrame:
    """Per GUID: class, span, contiguous runs, largest gap (h), late coverage (a segment ending in
    the last hour), admission TLO and labour duration (h; NaN without TLO) -- C9/C10."""
    s = seg.sort_values(["guid", "epoch_s"])
    step = s.groupby("guid")["epoch_s"].diff()
    gap_h = ((step - (stride or 0)) / 3600).where(step > 1.5 * (stride or 0))
    onset = (s["t_end_s"] - s["tlo_end_s"]).groupby(s["guid"]).first()
    g = s.groupby("guid")
    out = pd.DataFrame({
        "span_h": (g["t_end_s"].max() - g["epoch_s"].min()) / 3600, "n_runs": 1 + gap_h.notna().groupby(s["guid"]).sum(),
        "max_gap_h": gap_h.groupby(s["guid"]).max(), "late": g["t_end_s"].max() >= -3600,
        "admission_tlo_h": (g["epoch_s"].min() - onset) / 3600, "labour_h": -onset / 3600,
    })
    return gd.set_index("guid")[["clinical_class", "has_tlo", "has_ss"]].join(out, how="inner").reset_index()


def _describe(seg: pd.DataFrame, gd: pd.DataFrame) -> Dict[str, Any]:
    """C2 readout of one population."""
    n = seg.groupby("guid").size()
    h = seg["hours_to_delivery"]

    def hhmm(x: float) -> Optional[str]:
        return None if x != x else "%d:%02d" % divmod(int(round(x * 60)), 60)

    by = lambda frame, col: {str(k): int(v) for k, v in frame.groupby("clinical_class")[col].sum().items()}  # noqa: E731
    return {
        "n_guids": int(len(gd)), "n_segments": int(len(seg)),
        "segments_per_guid": {"min": n.min(), "max": n.max(), "mean": n.mean(), "median": n.median(), "sd": n.std()},
        "hours_to_delivery": {"min": h.min(), "max": h.max(), "min_hhmm": hhmm(h.min()), "max_hhmm": hhmm(h.max())},
        "unique_slots": int(seg["slot"].nunique()),
        "guids_by_class": {str(k): int(v) for k, v in gd["clinical_class"].value_counts().items()},
        "segments_by_class": {str(k): int(v) for k, v in seg["clinical_class"].value_counts().items()},
        "cs": {"guids": by(gd, "cs"), "segments": by(seg, "cs")}, "bg": {"guids": by(gd, "bg"), "segments": by(seg, "bg")},
        "has_tlo_by_class": by(gd, "has_tlo"), "has_ss_by_class": by(gd, "has_ss"),
    }


def _flagged(rec: Any, path: str = "") -> list:
    """``"<path> (delta d)"`` of every ``flagged`` entry of a confound record (``cohort/confound.json``: fold ->
    variable -> ``{missing_rate_by_class, delta, flagged}``), walked at any depth so has_tlo and covariates both count."""
    if not isinstance(rec, dict):
        return []
    if rec.get("flagged"):
        return [f"{path} (delta {rec.get('delta', NAN):.3f})"]
    return [x for k, v in rec.items() for x in _flagged(v, f"{path}/{k}" if path else str(k))]


def run_C(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Block C (C1-C12): cohort readouts from ``<run>/cohort/*`` written back into ``cohort/``.

    C2 ``dataset_summary.json``, C3 ``label_cross_table.csv``, C11 ``count_spread.csv``; C1 and C12
    re-read P0's ``fold_summary.csv``, ``exposure.json``, ``confound.json`` into the summary. C4-C10
    are figures (``report.py``); C9/C10 per-class numbers are returned here.
    """
    co = ctx.run_dir / "cohort"
    fs = pd.read_csv(co / "fold_summary.csv")
    seg, gd = pd.read_parquet(co / "segments.parquet"), pd.read_parquet(co / "guids.parquet")
    seg, gd = seg[~seg["excluded"].astype(bool)], gd[~gd["excluded"].astype(bool)]
    _, useg, ugd = unique_cohort(ctx.run_dir)
    man = _read_json(ctx.run_dir / "manifest.json") or {}
    sentinel = (man.get("cohort") or {}).get("sentinel_guids")

    # C2
    summary = {"overall": _describe(useg, ugd) | {"n_ss_sentinel_guids": None if sentinel is None else len(sentinel)},
               "per_fold_split": {f"{f}/{s}": _describe(_sel(seg, fold=f, split=s), g)
                                  for (f, s), g in gd.groupby(["fold", "split"])}}
    (co / "dataset_summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False))

    # C3: class x cs x bg, all 12 cells, per fold x split and over the unique cohort
    K = ["fold", "split", "clinical_class", "cs", "bg"]
    g_all = pd.concat([gd.astype({"fold": str}), ugd.assign(fold="all", split="all")], ignore_index=True)
    s_all = pd.concat([seg.astype({"fold": str}), useg.assign(fold="all", split="all")], ignore_index=True)
    cells = pd.MultiIndex.from_tuples(
        [(f, s, c, a, b) for f, s in g_all[["fold", "split"]].drop_duplicates().itertuples(index=False)
         for c in CLASSES for a in (False, True) for b in (False, True)], names=K)
    cross = pd.DataFrame({"n_guids": g_all.groupby(K)["guid"].nunique(), "n_segments": s_all.groupby(K).size()}
                         ).reindex(cells, fill_value=0).reset_index()
    cross.to_csv(co / "label_cross_table.csv", index=False)

    # C11: spread across folds of every C1 and C3 count
    per_fold = cross[cross["fold"] != "all"]
    stats = ["mean", "std", "min", "max"]
    spread = [per_fold.groupby(["split", "clinical_class", "cs", "bg"])[["n_guids", "n_segments"]].agg(stats),
              fs.groupby([c for c in fs.columns if c not in ("fold", "n")], dropna=False)[["n"]].agg(stats)]
    count_spread = pd.concat([x.set_axis(["_".join(c) for c in x.columns], axis=1).reset_index().assign(source=src)
                              for x, src in zip(spread, ("label_cross_table", "fold_summary"))], ignore_index=True)
    count_spread.to_csv(co / "count_spread.csv", index=False)

    # C9/C10 per class
    stride = infer_stride(seg, man)
    tl = guid_timeline(useg, ugd, stride)
    per_class = tl.groupby("clinical_class").agg(
        n_guids=("guid", "size"), runs_mean=("n_runs", "mean"), max_gap_h_median=("max_gap_h", "median"),
        late_coverage_frac=("late", "mean"), has_tlo_frac=("has_tlo", "mean"), has_ss_frac=("has_ss", "mean"),
        admission_tlo_h_median=("admission_tlo_h", "median"), labour_h_median=("labour_h", "median"))

    # C12: leakage and validity readouts
    src = ctx.cfg["source"]
    min_step = src["hdf5"]["min_step"] if src["kind"] == "hdf5" else None
    warm = min_step if isinstance(min_step, int) and not isinstance(min_step, bool) else None
    coverage = None if warm is None or not stride else min(1.0, 4.0 * (300 - warm) / stride)
    exposure, confound = _read_json(co / "exposure.json"), _read_json(co / "confound.json")
    splits_per_guid = pd.read_parquet(co / "guids.parquet").groupby(["fold", "guid"])["split"].nunique()
    leaks = splits_per_guid[splits_per_guid > 1].reset_index()[["fold", "guid"]].astype(str).values.tolist()
    overlap = {f: r.get("test_overlap") for f, r in ((exposure or {}).get("folds") or {}).items() if r.get("test_overlap")}
    unknown = sorted(f for f, r in ((exposure or {}).get("folds") or {}).items()  # L3: an unnamed shard list
                     if not all((r.get(k) or {}).get("known") for k in ("pretraining", "selection")))
    flagged, cap = _flagged(confound), ctx.cfg["context"]["missing_confound_max"]
    ablated = any(m == NOIND for m, _ in _models(ctx))
    sanity = {
        "cohort_disjoint": {"verdict": "fail" if leaks else "pass" if len(splits_per_guid) else "inconclusive",
                            "examples": leaks[:5], "n_checked": int(len(splits_per_guid)),
                            "detail": f"{len(leaks)} (fold, guid) pair(s) in more than one split" if leaks
                            else "train/val/test GUID-disjoint in every fold" if len(splits_per_guid)
                            else "no (fold, guid) pair to check"},
        "cohort_exposure": (
            {"verdict": "inconclusive", "detail": "no cohort/exposure.json"} if exposure is None else
            {"verdict": "fail", "detail": f"test GUIDs seen by the VAE in fold(s) {sorted(overlap)}"} if overlap else
            {"verdict": "inconclusive", "detail": f"pretraining exposure UNKNOWN in fold(s) {unknown}: the checkpoint's "
                                                  "resolved config names no vae_train_datasets / vae_test_datasets"}
            if unknown else
            {"verdict": "pass", "detail": exposure.get("note") or "no test GUID in the VAE's pretraining/selection shards"}),
        "missing_confound": {"flagged": flagged, "ablation_present": ablated, **(
            {"verdict": "inconclusive", "detail": "no cohort/confound.json, or it covers no fold"} if not confound else
            {"verdict": "pass", "detail": f"no train-fold missing rate differs across classes by more than {cap} "
                                          "(context.missing_confound_max)"} if not flagged else
            {"verdict": "pass", "detail": f"missing-rate gap across classes > {cap} in {flagged}; the no_indicator "
                                          f"ablation (model_id '{NOIND}') is present"} if ablated else
            {"verdict": "fail", "detail": f"missing-rate gap across classes > {cap} (context.missing_confound_max) in "
                                          f"{flagged}, and no no_indicator ablation (model_id '{NOIND}')"})},
    }
    # no_valid_steps: cohort segments baselines.fold_frame dropped for an all-masked cached step_mask (L14)
    nvs = {"segments": 0, "guids": 0}
    for fold in ctx.cfg["run"]["folds"]:
        rec = (_read_json(ctx.run_dir / "baselines" / f"fold_{fold}" / "probe_fit.json") or {}).get("no_valid_steps") or {}
        for split, n_seg in (rec.get("segments") or {}).items():
            n_none, n_short = ((rec.get(k) or {}).get(split, 0) for k in ("guids", "min_segments"))
            n_gd = int(((gd["fold"] == fold) & (gd["split"] == split)).sum())
            _incl(ctx, "no_valid_steps", "all", "all", split, str(fold), None, n_gd - n_none - n_short, n_none + n_short,
                  {"segments": n_seg, "guids_without_segments": n_none, "guids_below_min_segments": n_short})
            nvs = {"segments": nvs["segments"] + n_seg, "guids": nvs["guids"] + n_none + n_short}

    written = ["cohort/dataset_summary.json", "cohort/label_cross_table.csv", "cohort/count_spread.csv"]
    return {**_meta(ctx), "plan": {"written": written}, "no_valid_steps": nvs,
            "C1": {"fold_summary": fs.to_dict("records")}, "C2": summary["overall"],
            "C9_C10": {str(k): v for k, v in per_class.to_dict("index").items()},
            "C12": {"exposure": exposure, "confound": confound, "n_shared_test_guids": int(ugd["shared_test"].astype(bool).sum()),
                    "stride_s": stride, "warmup_steps": warm, "post_warmup_coverage_frac": coverage,
                    "coverage_note": None if coverage is not None
                    else "n/a: warm-up unknown (source.hdf5.min_step auto, or a VAE source)"},
            "sanity": sanity}


# --- time-resolved ---
#
# P4 (SPEC §11.5, §11.12 blocks M, A, R2-R8, T3, T4; L14). Per model/seed/split, every fold and the
# pooled OOF group (shared-test dedupe as :func:`pool_rows`) is prepared once (:func:`_tr_groups`):
# segments after ``eval.exclude_last_min``, sorted by (guid, seg_pos), with the online score s, its
# running max r and, per policy, the raw decision ``s > thr`` and the latched state (``eval.alarm_rule``).
# A time point is then one "last segment with clock <= c*" lookup per GUID (:func:`_view`): no epoch
# is ever filled and no point re-scans segments. Rates get Wilson CIs, or the patient-cluster bootstrap
# where a patient holds two GUIDs of the cell (:func:`rate_ci`), rankings the vectorised weighted
# Mann-Whitney (:func:`bootstrap_auroc`, clustered on the patient).
# GUID-level time-resolved rows are ``level='online'`` (§11.9), so they never mix with the P2 offline
# ``guid`` rows; their ``point`` is checkpoint, bin or end, and ``t`` is in the axis display units
# (to_delivery: hours before delivery; bins: the centre for instantaneous, the right edge c_hi for the
# committed types, where each is evaluated).

from teb_vae.classifier.thresholds import (  # noqa: E402
    AXIS_REASON, OVR_SEGMENT_SCORES, _online, clock, fpr_threshold, latch_state, ovr_view, policy_threshold,
    snapshot_window,
)

TYPES = ("instantaneous", "committed_cumulative", "committed_overall")
STAGE_CODES = {"first": 1, "second": 2}
_RANK = ("auroc", "auprc")
_ROC_P4 = ("committed_cumulative@", "committed_overall@", "snapshot@")


def _tr_ents(ctx: SimpleNamespace, m: str, sd: str, level: str) -> Dict[int, Dict[str, Any]]:
    """``{fold: {policy_id: thresholds.json entry}}`` of one model/seed and level; skipped entries dropped."""
    out = {}
    for key, per_level in ctx.thr.items():
        km, ks, kf = key.split("|")
        if (km, ks) == (m, sd):
            out[int(kf)] = {p: e for p, e in (per_level.get(level) or {}).items() if "skipped" not in e}
    return out


def _tr_pids(ev: Mapping[str, Any], ents: Mapping[int, Mapping[str, Any]]) -> list:
    """Config-ordered policy ids with a threshold in every fold."""
    return [p["id"] for p in ev["thresholds"] if ents and all(p["id"] in e for e in ents.values())]


def _prep(seg: pd.DataFrame, ents: Mapping, pids: list, seg_ents: Mapping, seg_pids: list,
          rule: Mapping) -> SimpleNamespace:
    """One group's arrays. Rows sorted by (guid, seg_pos) (``_online``); ``gi`` GUID code per row,
    ``first`` each GUID's first row; ``T`` (G, P) its fold's guid-level thresholds; per policy (P, n)
    ``ex`` = s > thr (raw), ``latch`` = r > thr and ``la`` the state under ``eval.alarm_rule``.
    ``T6``/``pids6`` add the segment-level policies (M6). ``pat`` (G,) each GUID's patient code
    (:func:`cluster`), None when every patient has one GUID in the group (the bootstraps' fast path)."""
    o = _online(seg)
    gi = pd.factorize(o["guid"])[0]
    first = np.flatnonzero(np.r_[True, gi[1:] != gi[:-1]])
    folds = o["fold"].to_numpy()[first]

    def table(e: Mapping, ids: list) -> np.ndarray:
        return np.array([[e[f][p]["threshold"] for p in ids] for f in folds], np.float64).reshape(first.size, len(ids))

    T = table(ents, pids)
    s, r = o["s"].to_numpy(np.float64), o["r"].to_numpy(np.float64)
    ex, latch = s > T[gi].T, r > T[gi].T
    la = latch if rule["kind"] == "latch" else latch_state(ex, gi, rule.get("k") or 1, rule.get("n") or 1)
    basis = {p: ents[folds[0]][p]["basis"] for p in pids} | {p: seg_ents[folds[0]][p]["basis"] for p in seg_pids}
    pat = pd.factorize(cluster(o).to_numpy()[first])[0]
    return SimpleNamespace(pat=None if pat.max(initial=-1) + 1 == pat.size else pat,
        o=o, gi=gi, first=first, s=s, r=r, y=o["y"].to_numpy()[first] == 1, guid=o["guid"].to_numpy()[first],
        st=o["stage"].map(STAGE_CODES).fillna(0).to_numpy(np.int64), T=T, T6=np.c_[T, table(seg_ents, seg_pids)],
        pids=pids, pids6=pids + seg_pids, basis=basis, ex=ex, latch=latch, la=la, clocks={})


def _clock(U: SimpleNamespace, axis: str) -> np.ndarray:
    if axis not in U.clocks:
        U.clocks[axis] = clock(U.o, axis).to_numpy(np.float64)
    return U.clocks[axis]


def _incl(ctx: SimpleNamespace, analysis: str, m: str, sd: str, split: str, fold: str, axis: Optional[str],
          n_included: int, n_excluded: int, reasons: Mapping[str, Any]) -> None:
    """One L14 ``inclusion.csv`` record."""
    ctx.inclusion.append(dict(analysis=analysis, model_id=m, seed=sd, split=split, fold=fold, axis=axis,
                              n_included=int(n_included), n_excluded=int(n_excluded), reasons=json.dumps(dict(reasons))))


def _tr_groups(ctx: SimpleNamespace, m: str, sd: str, split: str) -> Optional[list]:
    """``[(fold label, prepared group)]`` of one model/seed/split: every fold, then ``pooled`` (cached).

    Segments are :func:`segments` (after ``eval.exclude_last_min``, as threshold selection). The
    pooled group keeps each GUID once (:func:`pool_rows`, ``data.shared_test_policy``), each with its
    own fold's thresholds. None (recorded in ``inclusion``) for a model without an online or segment
    score, i.e. the GUID-level shortcut.
    """
    cache = ctx.__dict__.setdefault("tr_cache", {})
    if (m, sd, split) in cache:
        return cache[(m, sd, split)]
    ev, policy = ctx.cfg["eval"], ctx.cfg["data"]["shared_test_policy"]
    seg, kept = _sel(ctx.seg, model_id=m, seed=sd, split=split), segments(ctx, m, sd, split)
    ents, seg_ents = _tr_ents(ctx, m, sd, "guid"), _tr_ents(ctx, m, sd, "segment")
    pids, seg_pids = _tr_pids(ev, ents), _tr_pids(ev, seg_ents)
    if not len(kept) or not pids or seg[["logit_online_cal", "logit_seg_cal"]].isna().all().all():
        why = "no_segment_left" if len(seg) and not len(kept) else "no_online_score"
        _incl(ctx, "time_resolved", m, sd, split, "all", None, 0, seg["guid"].nunique(), {why: int(seg["guid"].nunique())})
        cache[(m, sd, split)] = None
        return None
    units = kept.drop_duplicates(["fold", "guid"])
    keep = pool_rows(units.assign(unit=units["guid"]), policy, split)[["fold", "guid"]]
    _incl(ctx, "pooled_dedupe", m, sd, split, "pooled", None, len(keep), len(units) - len(keep),
          {policy: len(units) - len(keep)})
    args = (ents, pids, seg_ents, seg_pids, ev["alarm_rule"])
    groups = [(str(f), _prep(x, *args)) for f, x in kept.groupby("fold")]
    groups.append(("pooled", _prep(kept.merge(keep, on=["fold", "guid"]), *args)))
    cache[(m, sd, split)] = groups
    return groups


def _grids(groups: list, axes: Sequence[str], bin_h: float) -> Dict[str, np.ndarray]:
    """Bin edges per axis, shared by every group of a model/seed/split: width ``bin_h`` aligned on 0,
    bins ``(lo, hi]``. ``position`` uses width 1 centred on each index (bin_h is in hours)."""
    out = {}
    for axis in axes:
        c = np.concatenate([_clock(U, axis) for f, U in groups if f != "pooled"])
        c = c[np.isfinite(c)]
        w, off = (1.0, 0.5) if axis == "position" else (float(bin_h), 0.0)
        out[axis] = (off + w * np.arange(np.ceil((c.min() - off) / w) - 1, np.ceil((c.max() - off) / w) + 1)
                     if c.size else np.zeros(0))
    return out


def _points(axis: str, edges: np.ndarray, ev: Mapping[str, Any]) -> pd.DataFrame:
    """Evaluation points of one axis: ``kind`` (bin | checkpoint | end), the instantaneous window
    ``(lo, hi]``, the committed point ``hi``, and display times ``t_inst``/``t_comm``.

    Bins: instantaneous on the bin, committed at its right edge. Checkpoints (to_delivery only):
    committed at ``c* = -h``, instantaneous within ``eval.snapshot_max_staleness_h`` of it (the window instantaneous
    threshold selection uses, ``thresholds.snapshot_window``). End: every segment.
    """
    sign = -1.0 if axis == "to_delivery" else 1.0
    lo, hi = list(edges[:-1]), list(edges[1:])
    kind = ["bin"] * len(lo)
    t_inst = [sign * (a + b) / 2.0 + 0.0 for a, b in zip(lo, hi)]
    t_comm = [sign * b + 0.0 for b in hi]
    if axis == "to_delivery":
        for h in ev["checkpoints_h"]:
            w = snapshot_window(axis, bin_h=ev["bin_h"], staleness_h=ev["snapshot_max_staleness_h"])
            kind.append("checkpoint"), lo.append(-h - w), hi.append(-h)
            t_inst.append(float(h)), t_comm.append(float(h))
    kind.append("end"), lo.append(-np.inf), hi.append(np.inf), t_inst.append(NAN), t_comm.append(NAN)
    return pd.DataFrame({"kind": kind, "lo": lo, "hi": hi, "t_inst": t_inst, "t_comm": t_comm})


def _view(U: SimpleNamespace, axis: str, pts: pd.DataFrame) -> Optional[SimpleNamespace]:
    """Per (GUID, point): ``row`` = n*, the last segment with clock <= hi; ``avail`` (monitoring had
    started); ``inst`` (n* is inside the instantaneous window, i.e. the snapshot); ``elig`` per GUID
    (a clock on every segment). One vectorised searchsorted over GUID-offset clocks. None if no GUID
    is eligible on the axis."""
    c = _clock(U, axis)
    elig = np.logical_and.reduceat(np.isfinite(c), U.first)
    if not elig.any():
        return None
    ce = c[elig[U.gi]]
    cmin, cmax = ce.min(), ce.max()
    W = cmax - cmin + 4.0
    key = U.gi * W + (np.where(elig[U.gi], c, cmin) - cmin + 2.0)
    if (np.diff(key) < 0).any():
        raise ValueError(f"the {axis} clock decreases within a GUID; seg_pos and the clock disagree")
    g = np.arange(U.first.size)[:, None]
    hi = np.clip(pts["hi"].to_numpy(np.float64), cmin - 1.0, cmax + 1.0)
    idx = np.searchsorted(key, g * W + (hi[None, :] - cmin + 2.0), side="right") - 1
    avail = (idx >= 0) & (U.gi[np.maximum(idx, 0)] == g) & elig[:, None]
    row = np.where(avail, idx, 0)
    inst = avail & (c[row] > pts["lo"].to_numpy(np.float64)[None, :])
    return SimpleNamespace(row=row, avail=avail, inst=inst, elig=elig, c=c)


#: Numeric columns of the metrics table; the rest hold strings or None.
_NUMERIC = ("t", "threshold", "value", "ci_lo", "ci_hi", "n_pos", "n_neg", "n_boot_undefined")


def _blocks_frame(blocks: list) -> pd.DataFrame:
    """One :data:`METRIC_COLUMNS` frame from column blocks ``{column: scalar | 1-D array}`` (``value``
    always an array). Building a DataFrame per small block costs more than the metrics themselves."""
    ns = np.array([len(b["value"]) for b in blocks], np.int64)
    ends, cols = np.cumsum(ns), {}
    for c in METRIC_COLUMNS:
        dt, vals = (np.float64 if c in _NUMERIC else object), [b.get(c) for b in blocks]
        arr = [i for i, v in enumerate(vals) if np.ndim(v)]
        fill = [NAN if (v is None or np.ndim(v)) and dt is np.float64 else None if np.ndim(v) else v for v in vals]
        out = np.repeat(np.array(fill, dt), ns)
        for i in arr:
            out[ends[i] - ns[i]:ends[i]] = np.asarray(vals[i], dt)
        cols[c] = out
    return pd.DataFrame(cols)


def _long(vals: Mapping[str, Tuple], *, pids: list, basis: list, thr: np.ndarray, t: np.ndarray, point: np.ndarray,
          n_pos: np.ndarray, n_neg: np.ndarray, **keys: Any) -> list:
    """Column blocks from ``{metric: (value, ci_lo, ci_hi, keep[, n_undefined])}`` of (P, K) arrays; ``keep`` masks
    rows; ``n_undefined`` (:func:`rate_ci`) fills ``n_boot_undefined``."""
    P, K = len(pids), len(t)
    col = dict(policy_id=np.repeat(np.asarray(pids, object), K), policy_basis=np.repeat(np.asarray(basis, object), K),
               threshold=np.repeat(thr, K), t=np.tile(t, P), point=np.tile(point, P), n_pos=np.tile(n_pos, P),
               n_neg=np.tile(n_neg, P))
    out = []
    for met, (v, lo, hi, keep, *nu) in vals.items():
        m = np.broadcast_to(keep, (P, K)).ravel()
        out.append(keys | {c: a[m] for c, a in col.items()} | {
            "metric": met, "value": np.broadcast_to(v, (P, K)).ravel()[m].astype(np.float64),
            "ci_lo": lo.ravel()[m], "ci_hi": hi.ravel()[m]}
            | ({"n_boot_undefined": np.broadcast_to(nu[0], (P, K)).ravel()[m]} if nu else {}))
    return out


def _rate_vals(hit: np.ndarray, pop: np.ndarray, y: np.ndarray, pat: Optional[np.ndarray], under: np.ndarray,
               live: np.ndarray, cp: np.ndarray, boot: Mapping[str, Any]) -> Dict[str, Tuple]:
    """``{metric: (value, ci_lo, ci_hi, keep)}`` of (P, K) alarm rates of the decisions ``hit`` (P, G, K)
    within the population ``pop`` (G, K): tp/fp always; sens/spec/fpr NaN when underpowered; PPV/NPV at
    checkpoints and end only. CIs: :func:`rate_ci` over the patients ``pat`` (Wilson; the cluster
    bootstrap in a cell where a patient holds two of its GUIDs)."""
    P, G, K = hit.shape
    yp, ok = y[:, None], live & ~under
    tp, fp = (hit & yp).sum(1), (hit & ~yp).sum(1)
    n_pos, n_neg = (pop & yp).sum(0), (pop & ~yp).sum(0)

    def ci(num: np.ndarray, den: np.ndarray, cols: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo, hi, nu = np.full((P, K), NAN), np.full((P, K), NAN), np.full((P, K), NAN)
        c = np.flatnonzero(cols)
        if c.size:
            rows = [np.broadcast_to(x, (P, G, K))[:, :, c].transpose(0, 2, 1).reshape(-1, G) for x in (num, den)]
            lo[:, c], hi[:, c], nu[:, c] = (x.reshape(P, c.size) for x in rate_ci(*rows, y, pat, **boot))
        return lo, hi, nu

    with np.errstate(invalid="ignore", divide="ignore"):  # confusion_rates: PPV/NPV 0 when never fired
        sens, fpr = np.where(ok, tp / n_pos, NAN), np.where(ok, fp / n_neg, NAN)
        tn, fn = n_neg - fp, n_pos - tp
        ppv, npv = np.where(tp + fp > 0, tp / (tp + fp), 0.0), np.where(tn + fn > 0, tn / (tn + fn), 0.0)
    (s_lo, s_hi, s_nu), (f_lo, f_hi, f_nu) = ci(hit & yp, pop & yp, ok), ci(hit & ~yp, pop & ~yp, ok)
    nan = np.full(tp.shape, NAN)
    out = {"tp": (tp, nan, nan, live), "fp": (fp, nan, nan, live), "sens": (sens, s_lo, s_hi, live, s_nu),
           "spec": (1.0 - fpr, 1.0 - f_hi, 1.0 - f_lo, live, f_nu), "fpr": (fpr, f_lo, f_hi, live, f_nu)}
    j = cp & ok
    if j.any():
        miss = pop & ~hit
        (p_lo, p_hi, p_nu), (n_lo, n_hi, n_nu) = ci(hit & yp, hit, j), ci(miss & ~yp, miss, j)
        out["ppv"] = (np.where(j, ppv, NAN), p_lo, p_hi, cp & live, p_nu)
        out["npv"] = (np.where(j, npv, NAN), n_lo, n_hi, cp & live, n_nu)
    return out


def _tf_block(y: np.ndarray, pop: np.ndarray, score: np.ndarray, *, names: Tuple[str, ...], under: np.ndarray,
              live: np.ndarray, t: np.ndarray, point: np.ndarray, rank_ci: np.ndarray, alpha: float,
              boot: Mapping[str, Any], n_pos: np.ndarray, n_neg: np.ndarray, units: Optional[np.ndarray] = None,
              **keys: Any) -> dict:
    """Threshold-free rows (``names`` of auroc, auprc, pauc@alpha) of each live point's population
    ``pop[:, k]`` scored ``score[:, k]``; an AUROC bootstrap CI where ``rank_ci``; NaN when
    underpowered or single-class."""
    cols: Dict[str, list] = {c: [] for c in ("t", "point", "n_pos", "n_neg", "metric", "value", "ci_lo", "ci_hi",
                                             "n_boot_undefined")}
    for k in np.flatnonzero(live):
        sel = pop[:, k]
        yk, sk = y[sel].astype(np.int64), score[sel, k]
        vals = dict.fromkeys(names, (NAN, NAN, NAN, NAN))
        if not under[k] and 0 < yk.sum() < yk.size:
            o, st, _ = _ranked(sk)
            vals = {m: (float(v[0]), NAN, NAN, NAN) for m, v in
                    _rank_stats(*_curve(yk, np.ones((1, yk.size)), o, st), alpha).items() if m in names}
            if rank_ci[k]:
                c = bootstrap_auroc(yk, sk, None if units is None else units[sel], boot["resamples"],
                                    boot["seed"])["metrics"]["auroc"]
                vals["auroc"] = (c["value"], c["ci_lo"], c["ci_hi"], c["n_undefined"])
        for m in names:
            for c, v in zip(cols, (t[k], point[k], n_pos[k], n_neg[k], m, *vals[m])):
                cols[c].append(v)
    return keys | {c: np.asarray(v, object if c in ("point", "metric") else np.float64) for c, v in cols.items()}


def _mt_frames(U: SimpleNamespace, V: SimpleNamespace, pts: pd.DataFrame, ev: Mapping[str, Any], *, alpha: float,
               boot: Mapping[str, Any], rank_ci: np.ndarray, **keys: Any) -> Tuple[list, Dict[str, np.ndarray]]:
    """§11.5.1 column blocks of one group on one axis: every policy under the three metric types (tp,
    fp, sens, spec, fpr; PPV/NPV at checkpoints/end), the threshold-free companions (snapshot AUROC,
    AUPRC, pAUC; cumulative AUROC), and the ``underpowered`` marker; all GUIDs and per stage (§11.5.4,
    the stage of the snapshot / of n*; committed_overall is not stratified: an unmonitored GUID has no
    stage at c*). AUROC CIs where ``rank_ci`` (stage strata: checkpoints and end only). Also returns
    the committed_overall counts for the M7 checks."""
    mn, yp = ev["min_bin_class_n"], U.y[:, None]
    kind = pts["kind"].to_numpy()
    cp = np.isin(kind, ("checkpoint", "end"))
    thr = U.T[0] if keys["fold"] != "pooled" else np.full(len(U.pids), NAN)
    dec_i, dec_c, st = U.ex[:, V.row], U.la[:, V.row] & V.avail, U.st[V.row]
    snap, cum = U.s[V.row], U.r[V.row]
    pops = [("instantaneous", None, V.inst, dec_i, snap), ("committed_cumulative", None, V.avail, dec_c, cum),
            ("committed_overall", None, np.broadcast_to(V.elig[:, None], V.avail.shape), dec_c, None)]
    pops += [(mt, name, pop & (st == code), dec, sc) for name, code in STAGE_CODES.items() for mt, _, pop, dec, sc in pops[:2]]
    blocks, overall = [], {}
    for mt, stratum, pop, dec, score in pops:
        n_pos, n_neg = (pop & yp).sum(0), (pop & ~yp).sum(0)
        live = n_pos + n_neg > 0
        if not live.any():
            continue
        under = (n_pos < mn) | (n_neg < mn)
        rates = _rate_vals(dec & pop, pop, U.y, U.pat, under, live, cp, boot)
        if mt == "committed_overall":
            overall = {"tp": rates["tp"][0], "fp": rates["fp"][0]}
        t = pts["t_inst" if mt == "instantaneous" else "t_comm"].to_numpy()
        k = keys | {"metric_type": mt, "denominator": _DENOM[mt]} | (
            {"subgroup": "stage", "subgroup_value": stratum} if stratum else {})
        cnt = dict(t=t, point=kind, n_pos=n_pos, n_neg=n_neg)
        blocks += _long(rates, pids=U.pids,
                        basis=[U.basis[p] for p in U.pids], thr=thr, **cnt, **k)
        blocks.append(k | {c: v[live] for c, v in cnt.items()} | {"metric": "underpowered", "value": under[live] * 1.0})
        if score is not None:
            names = (*_RANK, f"pauc@{alpha:g}") if mt == "instantaneous" else ("auroc",)
            blocks.append(_tf_block(U.y, pop, score, names=names, under=under, live=live, alpha=alpha, boot=boot,
                                    rank_ci=rank_ci if stratum is None else rank_ci & cp, units=U.pat,
                                    **cnt, **(k | {"metric_type": "threshold_free"})))
    return blocks, overall


def _seg_frames(U: SimpleNamespace, axis: str, edges: np.ndarray, ev: Mapping[str, Any], *, alpha: float,
                boot: Mapping[str, Any], ci: bool, **keys: Any) -> list:
    """M6 column blocks: segment-level instantaneous per bin (every segment in b, ``logit_seg_cal``,
    every policy incl. segment-level ones) and segment AUROC/AUPRC/pAUC per bin; all segments and per
    stage (the segment's own). Patient-cluster CIs when ``ci`` (stratum 'all'; :func:`rate_ci`). ``n_pos`` /
    ``n_neg`` count segments; underpowered counts GUIDs."""
    sc = U.o["logit_seg_cal"].to_numpy(np.float64)
    c = _clock(U, axis)
    b = np.searchsorted(edges, c, side="left") - 1
    ok = np.logical_and.reduceat(np.isfinite(c), U.first)[U.gi] & np.isfinite(sc) & (b >= 0) & (b < edges.size - 1)
    if not ok.any():
        return []
    nb, G, P = edges.size - 1, U.first.size, len(U.pids6)
    t = (-1.0 if axis == "to_delivery" else 1.0) * (edges[:-1] + edges[1:]) / 2.0 + 0.0
    thr = U.T6[0] if keys["fold"] != "pooled" else np.full(P, NAN)
    names, blocks = (*_RANK, f"pauc@{alpha:g}"), []
    for stratum, code in ((None, None), *STAGE_CODES.items()):
        R = np.flatnonzero(ok & ((U.st == code) if code else True))
        if not R.size:
            continue
        y, s, g, bb = U.y[U.gi[R]], sc[R], U.gi[R], b[R]
        cl = g if U.pat is None else U.pat[g]  # the bootstrap cluster of each segment
        hits = s[None, :] > U.T6[g].T
        n_pos, n_neg = np.bincount(bb, y, nb), np.bincount(bb, ~y, nb)
        tp = np.stack([np.bincount(bb, h & y, nb) for h in hits])
        fp = np.stack([np.bincount(bb, h & ~y, nb) for h in hits])
        pair = np.unique(bb * G + g)
        g_pos = np.bincount(pair // G, U.y[pair % G], nb)
        under = np.minimum(g_pos, np.bincount(pair // G, minlength=nb) - g_pos) < max(ev["min_bin_class_n"], 1)
        live = n_pos + n_neg > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            sens, fpr = np.where(under, NAN, tp / n_pos), np.where(under, NAN, fp / n_neg)
        s_lo, s_hi, f_lo, f_hi, s_nu, f_nu = (np.full((P, nb), NAN) for _ in range(6))
        tf = {m: np.full((4, nb), NAN) for m in names}
        for j, Rj in zip(*_split_by(bb, np.arange(R.size))):
            if under[j]:
                continue
            yj, sj, cj, hj = y[Rj], s[Rj], cl[Rj], hits[:, Rj]
            o, st, _ = _ranked(sj)
            for m, v in _rank_stats(*_curve(yj.astype(np.int64), np.ones((1, yj.size)), o, st), alpha).items():
                tf[m][0, j] = v[0]
            if ci and stratum is None:
                lo, hi, nu = rate_ci(np.r_[hj & yj, hj & ~yj],
                                     np.r_[np.broadcast_to(yj, hj.shape), np.broadcast_to(~yj, hj.shape)], yj, cj, **boot)
                s_lo[:, j], f_lo[:, j], s_hi[:, j], f_hi[:, j] = lo[:P], lo[P:], hi[:P], hi[P:]
                s_nu[:, j], f_nu[:, j] = nu[:P], nu[P:]
                r = bootstrap_auroc(yj, sj, cj, boot["resamples"], boot["seed"])["metrics"]["auroc"]
                tf["auroc"][1:, j] = r["ci_lo"], r["ci_hi"], r["n_undefined"]
        k = keys | {"axis": axis, "level": "segment", "denominator": "bin_present"} | (
            {"subgroup": "stage", "subgroup_value": stratum} if stratum else {})
        cnt = dict(t=t, point=np.full(nb, "bin", object), n_pos=n_pos, n_neg=n_neg)
        nan = np.full((P, nb), NAN)
        blocks += _long({"tp": (tp, nan, nan, live), "fp": (fp, nan, nan, live), "sens": (sens, s_lo, s_hi, live, s_nu),
                         "spec": (1.0 - fpr, 1.0 - f_hi, 1.0 - f_lo, live, f_nu), "fpr": (fpr, f_lo, f_hi, live, f_nu)},
                        pids=U.pids6, basis=[U.basis[p] for p in U.pids6], thr=thr, **cnt,
                        **(k | {"metric_type": "instantaneous"}))
        blocks.append(k | {c_: v[live] for c_, v in cnt.items()} | {"metric_type": "instantaneous",
                                                                    "metric": "underpowered", "value": under[live] * 1.0})
        L = np.flatnonzero(live)
        blocks.append(k | {c_: np.tile(v[L], len(names)) for c_, v in cnt.items()} | {
            "metric_type": "threshold_free", "metric": np.repeat(np.asarray(names, object), L.size),
            **{c_: np.concatenate([tf[m][i, L] for m in names])
               for i, c_ in enumerate(("value", "ci_lo", "ci_hi", "n_boot_undefined"))}})
    return blocks


def _split_by(b: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, list]:
    """``(values, [idx of each value])`` grouping ``idx`` by ``b``."""
    o = np.argsort(b, kind="stable")
    vals, starts = np.unique(b[o], return_index=True)
    return vals, np.split(idx[o], starts[1:])


def _verdict(n: int, bad: list, what: str) -> Dict[str, Any]:
    """A sanity check over ``n`` items: fail if any is ``bad``; a check that covered nothing is inconclusive, never a
    pass."""
    return {"verdict": "fail" if bad else "pass" if n else "inconclusive", "n_checked": int(n), "n_failed": len(bad),
            "examples": bad[:5], "detail": f"{len(bad)} of {n} {what} violated" if bad
            else f"all {n} {what} hold" if n else f"no {what} to check"}


def _boot(ev: Mapping[str, Any]) -> Dict[str, int]:
    return dict(resamples=ev["bootstrap"]["resamples"], seed=ev["bootstrap"]["seed"])


def run_M(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Block M (M1-M7) with R7 and the T4 table: every policy (whatever its basis) under the three
    metric types (§11.5.1) at ``eval.checkpoints_h`` (to_delivery) + end and on the ``eval.bin_h`` grid
    of every ``eval.time_axes`` axis, per fold and pooled, val and test, all GUIDs and per stage; the
    threshold-free companions; M6 for models with segment scores; M7 sanity; L14 inclusion records.

    Threshold-free bootstrap CIs are computed for the pooled group at every point and per fold at
    checkpoints and end; rates always carry binomial bootstrap CIs.
    """
    ev, alpha = eval_config, primary_alpha(eval_config)
    boot = _boot(ev)
    sentinel = set(((_read_json(ctx.run_dir / "manifest.json") or {}).get("cohort") or {}).get("sentinel_guids") or [])
    frames, mono, endq, last, n_mono, n_end, n_last, gap, no_online = [], [], [], [], 0, 0, 0, 0.0, set()
    for m, sd in _models(ctx):
        for split in SPLITS:
            groups = _tr_groups(ctx, m, sd, split)
            if groups is None:
                continue
            grids = _grids(groups, ev["time_axes"], ev["bin_h"])
            has_seg = any(U.o["logit_seg_cal"].notna().any() for _, U in groups)
            for fold, U in groups:
                keys = _keys(ctx, m, sd, split=split, fold=fold, level="online")
                for axis in ev["time_axes"]:
                    pts = _points(axis, grids[axis], ev)
                    V = _view(U, axis, pts)
                    n_el = int(V.elig.sum()) if V else 0
                    reasons = {AXIS_REASON.get(axis, "missing_clock"): U.first.size - n_el} if n_el < U.first.size else {}
                    if axis == "rel_second_stage" and reasons:
                        out = set(U.guid[~V.elig]) if V else set(U.guid)
                        reasons = {"ss_nan": len(out - sentinel), "ss_sentinel": len(out & sentinel)}
                    _incl(ctx, "axis_eligibility", m, sd, split, fold, axis, n_el, U.first.size - n_el, reasons)
                    if V is None:
                        continue
                    # ponytail: AUROC bootstrap per point is O(B n); pooled every point, per fold checkpoints/end
                    rank_ci = (pts["kind"] != "bin").to_numpy() | (fold == "pooled")
                    f, overall = _mt_frames(U, V, pts, ev, alpha=alpha, boot=boot, rank_ci=rank_ci, axis=axis, **keys)
                    frames += f
                    for k in np.flatnonzero(pts["kind"] == "checkpoint"):
                        n_in, n_av = int(V.inst[:, k].sum()), int(V.avail[:, k].sum())
                        _incl(ctx, f"staleness@{pts['t_comm'][k]:g}h", m, sd, split, fold, axis, n_in, n_av - n_in,
                              {"stale": n_av - n_in, "max_staleness_h": ev["snapshot_max_staleness_h"]})
                    # M7a: committed_overall is monotone non-decreasing in c*
                    o = np.argsort(pts["hi"].to_numpy(), kind="stable")
                    for p, pid in enumerate(U.pids):
                        n_mono += 1
                        if (np.diff(overall["tp"][p, o]) < 0).any() or (np.diff(overall["fp"][p, o]) < 0).any():
                            mono.append(f"{m}|{sd}|{split}|{fold}|{axis}|{pid}")
                    if axis == "to_delivery":  # M7b: at end, committed_overall = running-max GUID decision
                        runmax = np.maximum.reduceat(U.s, U.first)[:, None] > U.T
                        for p, pid in enumerate(U.pids):
                            n_end += 1
                            want = (int((runmax[:, p] & U.y).sum()), int((runmax[:, p] & ~U.y).sum()))
                            if ev["alarm_rule"]["kind"] == "latch" and want != (overall["tp"][p, -1], overall["fp"][p, -1]):
                                endq.append(f"{m}|{sd}|{split}|{fold}|{pid}")
                    if has_seg:
                        frames += _seg_frames(U, axis, grids[axis], ev, alpha=alpha, boot=boot,
                                              ci=fold == "pooled" and axis == "to_delivery", **keys)
                if fold == "pooled":
                    continue
                # M7c: instantaneous at the last segment = the score_final decision. Not applicable without an online
                # score: _online's running max of logit_seg_cal is not a non-causal model's score_final.
                segs = _sel(ctx.seg, model_id=m, seed=sd, split=split, fold=int(fold))
                if segs["logit_online_cal"].isna().all():
                    no_online.add(f"{m}|{sd}")
                    continue
                o = _online(segs).drop_duplicates("guid", keep="last")
                fin = _sel(ctx.guids, model_id=m, seed=sd, split=split, fold=int(fold)).set_index("guid")["score_final_cal"]
                s_last, s_fin = o["s"].to_numpy(np.float64), fin.reindex(o["guid"]).to_numpy(np.float64)
                gap = max(gap, float(np.nanmax(np.abs(s_last - s_fin), initial=0.0)))
                for p, pid in enumerate(U.pids):  # a 1e-14 float gap at a tied threshold is not a disagreement
                    n_last += 1
                    if (((s_last > U.T[0, p]) != (s_fin > U.T[0, p])) & ~(np.abs(s_last - s_fin) <= 1e-9)).any():
                        last.append(f"{m}|{sd}|{split}|{fold}|{pid}")
    frame = _blocks_frame(frames) if frames else None
    latch = ev["alarm_rule"]["kind"] == "latch"
    sanity = {
        "m7_committed_overall_monotone": _verdict(n_mono, mono, "committed_overall monotone series"),
        "m7_end_equals_running_max": _verdict(n_end, endq, "end committed_overall == running-max decision") if latch else
        {"verdict": "inconclusive", "detail": "the running-max identity holds under the latch rule only (k_of_n configured)"},
        "m7_last_equals_final": _verdict(n_last, last, "instantaneous-at-last-segment == score_final decision")
        | {"max_abs_score_gap": gap, "not_applicable": sorted(no_online)},  # models without an online score
    }
    return {"frame": frame, "sanity": sanity, **_meta(ctx), "n_rows": 0 if frame is None else int(len(frame)), "plan": {
        "levels": {"online": "GUID online score s_g(n), running max r_g(n)", "segment": "M6, logit_seg_cal"},
        "metric_types": list(TYPES), "axes": list(ev["time_axes"]), "checkpoints_h": list(ev["checkpoints_h"]),
        "bin_h": ev["bin_h"], "position_bin": 1, "staleness_h": ev["snapshot_max_staleness_h"],
        "exclude_last_min": ev["exclude_last_min"], "min_bin_class_n": ev["min_bin_class_n"],
        "alarm_rule": ev["alarm_rule"], "stages": list(STAGE_CODES),
        "t": "display units: to_delivery hours before delivery; bins: centre (instantaneous), right edge (committed)",
        "intervals": {"rates": {"method": "wilson 95% where each patient holds at most one GUID of the cell, else the "
                                          "outcome-stratified patient-cluster percentile bootstrap (rate_ci)",
                                **boot, "confidence": 0.95},
                      "ranking": {"method": "vectorised weighted Mann-Whitney patient-cluster bootstrap (AUROC)", **boot,
                                  "confidence": 0.95, "where": "pooled: every point (stage strata: checkpoints/end); "
                                  "per fold: checkpoints and end; AUPRC/pAUC point values"},
                      "segment": "M6 patient-cluster CIs (rate_ci): pooled, to_delivery, all segments"}}}


def _alarm_arrays(U: SimpleNamespace, la: np.ndarray) -> Dict[str, np.ndarray]:
    """(P, G) per-GUID alarm summary of a latched state (a suffix of each GUID's rows)."""
    n_g = np.diff(np.r_[U.first, U.gi.size])
    n_on = np.add.reduceat(la.astype(np.int64), U.first, axis=1)
    alarmed = n_on > 0
    t_end = U.o["t_end_s"].to_numpy(np.float64)
    first_t = np.where(alarmed, t_end[np.minimum(U.first + n_g - n_on, t_end.size - 1)], NAN)
    return {"alarmed": alarmed, "first_alarm_t_s": first_t, "lead_time_h": -first_t / 3600.0,
            "time_to_first_alarm_h": (first_t - t_end[U.first]) / 3600.0, "burden": n_on / n_g,
            "n_segments": np.broadcast_to(n_g, alarmed.shape)}


def _stat_ci(x: np.ndarray, stat: Callable, boot: Mapping[str, Any], units: Optional[np.ndarray] = None,
             confidence: float = 0.95) -> Tuple[float, float, float]:
    """(value, ci_lo, ci_hi) of ``stat`` (``np.mean`` or ``np.median``) over GUIDs: percentile bootstrap
    of the GUIDs, or with ``units`` of their patients (a patient's GUIDs weigh together). The median is the
    weighted lower median, for the value and every draw alike. NaN if empty."""
    if not x.size:
        return NAN, NAN, NAN
    B, rng = boot["resamples"], np.random.default_rng(boot["seed"])
    codes, names = pd.factorize(np.arange(x.size) if units is None else np.asarray(units))
    o = np.argsort(x, kind="stable")
    W = np.vstack([np.ones(x.size), _counts(len(names), B, rng)[:, codes[o]]])  # row 0: the sample itself
    C = np.cumsum(W, axis=1)
    d = (W @ x[o]) / C[:, -1] if stat is np.mean else x[o][np.argmax(C >= C[:, -1:] / 2.0, axis=1)]
    a = (1.0 - confidence) / 2.0
    return (float(d[0]), *map(float, np.quantile(d[1:], [a, 1 - a])))


def run_A(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Block A (A1-A5), per policy under the latch (and ``k_of_n`` when configured, A5):
    ``alarms.parquet`` (one row per model, seed, fold, split, policy, rule, GUID) and ``level='alarm'``
    rows per fold and pooled: event sensitivity/FPR, lead time and time-to-first-false-alarm median
    and IQR, alarm burden, alarms per detection, and the detection / false-alarm curves vs hours
    before delivery (bin right edges and checkpoints)."""
    ev = eval_config
    boot, rules = _boot(ev), ["latch"] + (["k_of_n"] if ev["alarm_rule"]["kind"] == "k_of_n" else [])
    tables, rows = [], []
    for m, sd in _models(ctx):
        for split in SPLITS:
            groups = _tr_groups(ctx, m, sd, split)
            if groups is None:
                continue
            edges = _grids(groups, ["to_delivery"], ev["bin_h"])["to_delivery"]
            cs = np.r_[edges[1:], -np.asarray(ev["checkpoints_h"], np.float64)]
            kinds = ["bin"] * (edges.size - 1 if edges.size else 0) + ["checkpoint"] * len(ev["checkpoints_h"])
            for fold, U in groups:
                G = U.first.size

                def pat(sel: np.ndarray, U: SimpleNamespace = U) -> Optional[np.ndarray]:
                    return None if U.pat is None else U.pat[sel]

                for rule in rules:
                    A = _alarm_arrays(U, U.latch if rule == "latch" else U.la)
                    thr = U.T[0] if fold != "pooled" else np.full(len(U.pids), NAN)
                    if fold != "pooled":
                        tables.append(pd.DataFrame({
                            "model_id": m, "seed": sd, "fold": int(fold), "split": split,
                            "policy_id": np.repeat(U.pids, G), "rule": rule, "guid": np.tile(U.guid, len(U.pids)),
                            "y": np.tile(U.y.astype(np.int64), len(U.pids)),
                            "clinical_class": np.tile(U.o["clinical_class"].to_numpy()[U.first], len(U.pids)),
                            "threshold": np.repeat(thr, G), **{c: v.ravel() for c, v in A.items()}}))
                    base = _keys(ctx, m, sd, split=split, fold=fold, level="alarm", metric_type="alarm", denominator="all",
                                 axis="to_delivery", subgroup="alarm_rule", subgroup_value=rule)
                    n_pos, n_neg = int(U.y.sum()), int((~U.y).sum())
                    for p, pid in enumerate(U.pids):
                        k = base | dict(policy_id=pid, policy_basis=U.basis[pid], threshold=thr[p], n_pos=n_pos, n_neg=n_neg)
                        al = A["alarmed"][p]
                        tp, fp = int((al & U.y).sum()), int((al & ~U.y).sum())
                        (s_lo, f_lo), (s_hi, f_hi), (s_nu, f_nu) = rate_ci(
                            np.stack([al & U.y, al & ~U.y]), np.stack([U.y, ~U.y]), U.y, U.pat, **boot)
                        stat = {"event_sens": (tp / n_pos if n_pos else NAN, s_lo, s_hi, s_nu),
                                "event_fpr": (fp / n_neg if n_neg else NAN, f_lo, f_hi, f_nu),
                                "alarms_per_detection": (fp / tp if tp else NAN, NAN, NAN),
                                "burden_pos_mean": _stat_ci(A["burden"][p][U.y], np.mean, boot, pat(U.y)),
                                "burden_neg_mean": _stat_ci(A["burden"][p][~U.y], np.mean, boot, pat(~U.y))}
                        for name, sel in (("lead_time", al & U.y), ("ttfa", al & ~U.y)):
                            x = A["lead_time_h" if name == "lead_time" else "time_to_first_alarm_h"][p][sel]
                            stat[f"{name}_median_h"] = _stat_ci(x, np.median, boot, pat(sel))
                            for q in (25, 75):
                                stat[f"{name}_q{q}_h"] = (float(np.quantile(x, q / 100)) if x.size else NAN, NAN, NAN)
                        rows += metric_rows({mm: v[0] for mm, v in stat.items()},
                                            {mm: {"ci_lo": v[1], "ci_hi": v[2], "n_undefined": (v[3:] or (None,))[0]}
                                             for mm, v in stat.items()}, point="end", t=NAN, **k)
                        c_first = A["first_alarm_t_s"][p] / 3600.0
                        hit = c_first[None, :] <= cs[:, None]  # alarmed by c* (NaN = never)
                        dp, dn = (hit & U.y).sum(1), (hit & ~U.y).sum(1)
                        lo, hi, nu = rate_ci(np.r_[hit & U.y, hit & ~U.y],
                                             np.repeat(np.stack([U.y, ~U.y]), len(cs), axis=0), U.y, U.pat, **boot)
                        (p_lo, n_lo), (p_hi, n_hi), (p_nu, n_nu) = np.split(lo, 2), np.split(hi, 2), np.split(nu, 2)
                        for i, (c_, kd) in enumerate(zip(cs, kinds)):
                            rows += metric_rows({"detection_frac": dp[i] / n_pos if n_pos else NAN,
                                                 "false_alarm_frac": dn[i] / n_neg if n_neg else NAN},
                                                {"detection_frac": {"ci_lo": p_lo[i], "ci_hi": p_hi[i], "n_undefined": p_nu[i]},
                                                 "false_alarm_frac": {"ci_lo": n_lo[i], "ci_hi": n_hi[i],
                                                                      "n_undefined": n_nu[i]}},
                                                point=kd, t=-c_ + 0.0, **k)
    alarms = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(
        columns=["model_id", "seed", "fold", "split", "policy_id", "rule", "guid", "y", "alarmed"])
    alarms.to_parquet(out_dir / "tables" / "alarms.parquet", index=False)
    return {"rows": rows, **_meta(ctx), "n_alarm_rows": int(len(alarms)), "plan": {
        "rules": rules, "alarmed": "latched state at the last segment (after exclude_last_min)",
        "lead_time_h": "hours before delivery of the first alarm", "time_to_first_alarm_h": "from the first segment end",
        "burden": "fraction of monitored segments at/after the first alarm", "curves": "alarmed by c* = -t, all GUIDs"}}


def _append_roc(out_dir: Path, frames: list, prefixes: Tuple[str, ...]) -> int:
    """Replace this analysis' ``variant`` rows in ``roc_points.parquet`` (R1 writes the file first)."""
    path = out_dir / "tables" / "roc_points.parquet"
    old = pd.read_parquet(path) if path.is_file() else pd.DataFrame(columns=["variant"])
    old = old[~old["variant"].astype(str).str.startswith(prefixes)]
    new = pd.concat([old, *frames], ignore_index=True) if frames else old
    new.to_parquet(path, index=False)
    return int(len(new) - len(old))


def run_R2(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """R2-R4 and R6 (PR at checkpoints): at every ``eval.checkpoints_h`` and end, the ROC of r(n*) over
    the GUIDs available (``committed_cumulative@<h>``), over all GUIDs with -inf for the unmonitored
    (``committed_overall@<h>``), and of the snapshot inside the staleness window (``snapshot@<h>``);
    per fold and pooled, val and test, ``level='online'``. ``precision`` next to ``tpr`` (= recall)
    is the PR curve."""
    ev, frames = eval_config, []
    for m, sd in _models(ctx):
        for split in SPLITS:
            for fold, U in _tr_groups(ctx, m, sd, split) or []:
                pts = _points("to_delivery", np.zeros(0), ev)
                V = _view(U, "to_delivery", pts)
                if V is None:
                    continue
                for k, (kind, t) in enumerate(zip(pts["kind"], pts["t_comm"])):
                    row = V.row[:, k]
                    for name, sel, score in (("committed_cumulative", V.avail[:, k], U.r[row]),
                                             ("committed_overall", V.elig, np.where(V.avail[:, k], U.r[row], -np.inf)),
                                             ("snapshot", V.inst[:, k], U.s[row])):
                        y = U.y[sel]
                        if 0 < y.sum() < y.size:
                            frames.append(pd.DataFrame({
                                "model_id": m, "seed": sd, "fold": fold, "split": split, "level": "online",
                                "variant": f"{name}@{'end' if kind == 'end' else f'{t:g}'}", "axis": "to_delivery", "t": t,
                                **roc_points(y, score[sel]), "n_pos": int(y.sum()), "n_neg": int((~y).sum())}))
    return {**_meta(ctx), "n_rows": _append_roc(out_dir, frames, _ROC_P4), "plan": {
        "variants": [f"{v}<h|end>" for v in _ROC_P4], "pr": "precision vs tpr (recall) of the same rows (R6)",
        "t": "hours before delivery (NaN at end)"}}


def run_R5(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """R5: segment-level ROC on the time-matched eval window (``in_eval_window`` rows, ``logit_seg_cal``),
    per fold and pooled (variant ``segment``), with the pooled GUID-cluster bootstrap band on
    ``ROC_GRID`` (``segment:band``)."""
    policy, boot, frames = ctx.cfg["data"]["shared_test_policy"], _boot(eval_config), []
    for m, sd in _models(ctx):
        for split in SPLITS:
            u = level_units(ctx, m, sd, split).get("segment")
            if u is None:
                continue
            key = dict(model_id=m, seed=sd, split=split, level="segment", eval_window=ctx.cfg["labels"]["eval_window"])
            for fold, f in [*((str(k), x) for k, x in u.groupby("fold")), ("pooled", pool_rows(u, policy, split))]:
                y = f["y"].to_numpy(np.int64)
                if not 0 < y.sum() < y.size:
                    continue
                r = roc_points(y, f["score"])
                frames.append(pd.DataFrame({**key, "fold": fold, "variant": "segment", **r,
                                            "n_pos": int(y.sum()), "n_neg": int(y.size - y.sum())}))
                if fold == "pooled":
                    lo, hi = roc_band(y, f["score"], f["patient"], ROC_GRID, **boot)
                    frames.append(pd.DataFrame({**key, "fold": fold, "variant": "segment:band", "fpr": ROC_GRID,
                                                "tpr": tpr_at(r["fpr"], r["tpr"], ROC_GRID), "tpr_lo": lo, "tpr_hi": hi}))
    return {**_meta(ctx), "n_rows": _append_roc(out_dir, frames, ("segment",)), "plan": {
        "window": ctx.cfg["labels"]["eval_window"], "band": "pooled patient-cluster bootstrap 95%, step read"}}


def _count_rows(fire: np.ndarray, y: np.ndarray, units: Any, boot: Mapping[str, Any], **keys: Any) -> list:
    """tp, fp, sens, spec, fpr rows of one decision per GUID row (``fire``, ``y`` bool); :func:`rate_ci` CIs."""
    tp, fp, n_pos, n_neg = int((fire & y).sum()), int((fire & ~y).sum()), int(y.sum()), int((~y).sum())
    (s_lo, f_lo), (s_hi, f_hi), (s_nu, f_nu) = rate_ci(np.stack([fire & y, fire & ~y]), np.stack([y, ~y]), y, units,
                                                       **boot)
    sens, fpr = tp / n_pos if n_pos else NAN, fp / n_neg if n_neg else NAN
    return metric_rows({"tp": tp, "fp": fp, "sens": sens, "spec": 1.0 - fpr, "fpr": fpr},
                       {"sens": {"ci_lo": s_lo, "ci_hi": s_hi, "n_undefined": s_nu},
                        "spec": {"ci_lo": 1.0 - f_hi, "ci_hi": 1.0 - f_lo, "n_undefined": f_nu},
                        "fpr": {"ci_lo": f_lo, "ci_hi": f_hi, "n_undefined": f_nu}}, n_pos=n_pos, n_neg=n_neg, **keys)


def run_R8(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """R8 decision-horizon ribbon: at each c* in ``eval.decision_horizons_h`` every FPR-cap policy's
    threshold is re-selected on VAL (committed_overall basis at c*, to_delivery, running max), then
    val and test sens/FPR are read at c* on the committed_overall population, per fold and pooled.
    Rows: ``level='online'``, ``subgroup='decision_horizon'``, ``subgroup_value=t=h``."""
    ev, policy = eval_config, ctx.cfg["data"]["shared_test_policy"]
    boot = _boot(ev)
    pols = [p for p in ev["thresholds"] if p["policy"] == "fpr_cap" and p["basis"] != "segment"]
    recs, failed = [], []
    for m, sd in _models(ctx):
        seg = _sel(ctx.seg, model_id=m, seed=sd)
        if not pols or seg[["logit_online_cal", "logit_seg_cal"]].isna().all().all():
            continue
        seg = pd.concat([segments(ctx, m, sd, sp) for sp in SPLITS])
        for fold, f in seg.groupby("fold"):
            by = {sp: f[f["split"] == sp] for sp in SPLITS}
            for h in ev["decision_horizons_h"]:
                pops = {sp: basis_frame(x, None, "committed_overall", at=h) for sp, x in by.items()}
                neg = pops["val"]["score"][pops["val"]["y"] == 0]
                for p in pols:
                    try:
                        thr = fpr_threshold(neg, p["alpha"], p["method"], p.get("delta") or 0.05,
                                            p.get("allow_fallback", False))["threshold"]
                    except ValueError as e:
                        failed.append(f"{m}|{sd}|{fold}|{p['id']}@{h:g}h: {e}")
                        continue
                    recs += [x.assign(model_id=m, seed=sd, fold=fold, split=sp, policy_id=p["id"], alpha=p["alpha"],
                                      h=float(h), threshold=thr, fired=x["score"] > thr) for sp, x in pops.items()]
    rows = []
    if recs:
        R = pd.concat(recs, ignore_index=True)
        flags = ctx.seg.drop_duplicates(["model_id", "seed", "fold", "split", "guid"])
        flags = flags.assign(patient=cluster(flags))[["model_id", "seed", "fold", "split", "guid", "shared_test", "patient"]]
        R = R.merge(flags, on=["model_id", "seed", "fold", "split", "guid"], how="left")
        for (m, sd, sp, pid, h), g in R.groupby(["model_id", "seed", "split", "policy_id", "h"], sort=False):
            k = _keys(ctx, m, sd, split=sp, level="online", metric_type="committed_overall", denominator="all",
                      axis="to_delivery", t=h, point="checkpoint", policy_id=pid, policy_basis="committed_overall",
                      subgroup="decision_horizon", subgroup_value=f"{h:g}")
            parts = [(str(fo), x, float(x["threshold"].iloc[0])) for fo, x in g.groupby("fold")]
            parts.append(("pooled", pool_rows(g.assign(unit=g["guid"]), policy, sp), NAN))
            for fold, x, thr in parts:
                rows += _count_rows(x["fired"].to_numpy(), x["y"].to_numpy() == 1, x["patient"], boot, fold=fold,
                                    threshold=thr, **k)
    return {"rows": rows, **_meta(ctx), "failed": failed, "plan": {
        "horizons_h": list(ev["decision_horizons_h"]), "policies": [p["id"] for p in pols],
        "basis": "committed_overall at c* (to_delivery, latch), re-selected on val per fold"}}


def _patient_draws(units: pd.DataFrame, patients: Any, resamples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(M, at)``: the outcome-stratified patient-cluster bootstrap of ``units`` (one row per GUID with ``y`` and
    ``patient``; :func:`_unit_draws`) as patient multiplicities ``M`` (resamples, patients), and each of ``patients``'s
    column in ``M``: row ``i`` of a basis population weighs ``M[b, at[i]]`` in draw ``b``."""
    names = units["patient"].astype(str).to_numpy()
    M = _unit_draws(units["y"].to_numpy(np.int64), names, resamples, seed)[0]
    at = pd.Index(np.unique(names)).get_indexer(np.asarray(patients, dtype=str))
    if (at < 0).any():
        raise ValueError("a basis row's patient is not among the resampled GUIDs")
    return M, at


def _refit_threshold(pol: Mapping[str, Any], y: np.ndarray, s: np.ndarray, w: np.ndarray) -> float:
    """``pol``'s threshold on the resampled basis (row ``i`` repeated ``w[i]`` times); NaN when it cannot be chosen."""
    idx = np.repeat(np.arange(y.size), w.astype(np.int64))
    try:
        return policy_threshold(pol, y[idx], s[idx])["threshold"]
    except ValueError:  # e.g. an NP cap whose resampled bin holds too few negatives: an undefined draw, counted
        return NAN


def _refit_rates(pop: pd.DataFrame, thr: np.ndarray, code: np.ndarray, M: np.ndarray, at: np.ndarray
                 ) -> Dict[str, np.ndarray]:
    """sens / spec / fpr per draw of the test basis rows ``pop`` fired at their fold's draw threshold ``thr[b, code]``
    (``thr`` (B, folds)), each row weighing its patient's multiplicity ``M[b, at]``; NaN where a threshold is."""
    y, s = pop["y"].to_numpy() == 1, pop["score"].to_numpy(np.float64)
    out = {k: np.full(len(thr), NAN) for k in ("sens", "fpr")}
    for sl in _chunks(len(thr), len(pop)):
        W, fire = M[sl][:, at], s > thr[sl][:, code]
        with np.errstate(invalid="ignore", divide="ignore"):
            out["sens"][sl] = (W * (fire & y)).sum(1) / (W * y).sum(1)
            out["fpr"][sl] = (W * (fire & ~y)).sum(1) / (W * ~y).sum(1)
    bad = np.isnan(thr[:, np.unique(code)]).any(1)
    for v in out.values():
        v[bad] = NAN
    return out | {"spec": 1.0 - out["fpr"]}


def refit_rows(ctx: SimpleNamespace, m: str, sd: str, pops: Mapping[str, Any], boot: Mapping[str, Any]) -> list:
    """§11.7 variant b, the refit bootstrap (``eval.bootstrap.refit_threshold``): test sens / spec / FPR of every policy
    (per fold and pooled) whose CIs carry the threshold's own sampling noise, plus per fold the chosen threshold with
    the percentile interval of its refits. Rows ``split='test'``, ``subgroup='threshold_refit'``, value ``refit``,
    keyed as T3 (c); ``pops`` is ``{split: policy_populations(...)}``.

    Replicate b resamples each fold's val GUIDs (outcome-stratified patient clusters, seed ``seed + fold``; a GUID drawn
    twice counts twice), re-selects every policy on that fold's resampled val basis population with the selection's
    own dispatch (:func:`~teb_vae.classifier.thresholds.policy_threshold`), then scores a resample of the fold's test
    GUIDs (seed ``seed + 10000 + fold``; pooled: of the pooled test GUIDs, seed ``seed + 20000``, each row at its own
    fold's replicate threshold). The point value is the fixed-threshold one (:func:`run_metrics`' rows).

    The val scores are the calibrated ones: binary temperature / Platt (a > 0) and the ordinal map are increasing per
    fold, so for a rank policy (fpr_cap, youden, sens_target) re-selecting on them is exactly refitting the calibration,
    then the threshold. ``ponytail:`` a ``fixed`` policy keeps the fold's calibration (its refit CI carries no
    calibration noise), and a multiclass alarm logit is re-ranked by a refitted T in general; refit the calibration
    per replicate (``train.fit_calibration`` on the raw val logits) if either becomes a primary policy. One threshold
    search per replicate, fold and policy (``ponytail:`` O(B) python calls; vectorise the empirical cap if final
    reports are slow). Per-class OvR thresholds (T5) are not refitted.
    """
    policy, pols = ctx.cfg["data"]["shared_test_policy"], {p["id"]: p for p in ctx.cfg["eval"]["thresholds"]}
    B, seed = boot["resamples"], boot["seed"]
    gd = {sp: level_units(ctx, m, sd, sp)["guid"] for sp in SPLITS}
    rows = []
    for (level, pid), (ents, vpop) in pops["val"].items():
        if (level, pid) not in pops["test"] or pid not in pols:
            continue
        tpop, folds = pops["test"][(level, pid)][1], sorted(ents)
        thr = np.empty((B, len(folds)))
        for j, fold in enumerate(folds):
            v = vpop[vpop["fold"] == fold]
            M, at = _patient_draws(_sel(gd["val"], fold=fold), v["patient"], B, seed + fold)
            y, s = v["y"].to_numpy(np.int64), v["score"].to_numpy(np.float64)
            thr[:, j] = [_refit_threshold(pols[pid], y, s, M[b, at]) for b in range(B)]
        e0 = next(iter(ents.values()))
        k = _keys(ctx, m, sd, split="test", level=level, policy_id=pid, policy_basis=e0["basis"],
                  metric_type=e0["basis"] if e0["basis"] in _DENOM else None, denominator=_DENOM.get(e0["basis"], "n/a"),
                  axis=None if e0["at"] == "end" else e0["axis"], t=NAN if e0["at"] == "end" else float(e0["at"]),
                  point="end" if e0["at"] == "end" else "checkpoint", subgroup="threshold_refit", subgroup_value="refit")
        chosen = np.array([ents[f]["threshold"] for f in folds])
        parts = [(f, tpop[tpop["fold"] == f], _sel(gd["test"], fold=f), seed + 10_000 + f) for f in folds]
        parts.append(("pooled", pool_rows(tpop, policy, "test"), pool_rows(gd["test"], policy, "test"), seed + 20_000))
        for fold, t, units, sd_ in parts:
            code = pd.Index(folds).get_indexer(t["fold"])
            M, at = _patient_draws(units, t["patient"], B, sd_)
            point = {x: float(v[0]) for x, v in _refit_rates(t, chosen[None], code, np.ones((1, M.shape[1])), at).items()}
            draws, j = _refit_rates(t, thr, code, M, at), None if fold == "pooled" else folds.index(fold)
            if j is not None:
                point, draws = point | {"threshold": float(chosen[j])}, draws | {"threshold": thr[:, j]}
            ci = _percentile_ci(point, draws, method="refit bootstrap (§11.7 b)", resamples=B, seed=sd_,
                                confidence=0.95, n=M.shape[1])["metrics"]
            n_pos = int((t["y"] == 1).sum())
            rows += metric_rows(point, ci, fold=str(fold), threshold=NAN if j is None else float(chosen[j]),
                                n_pos=n_pos, n_neg=int(len(t) - n_pos), **k)
    return rows


def run_T3(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """T3 threshold stability: (a) spread of the per-fold thresholds; (b) with
    ``eval.bootstrap.refit_threshold``, the refit bootstrap (:func:`refit_rows`: the refitted thresholds' spread and
    test rates whose CIs carry threshold noise); (c) val and test sens/FPR on each FPR-cap policy's basis population when
    its threshold moves by -2..+2 validation order statistics (sorted val basis negatives, ties
    included), per fold and pooled; rows ``subgroup='threshold_perturbation'``, value ``+j``."""
    ev, policy = eval_config, ctx.cfg["data"]["shared_test_policy"]
    kinds, boot = {p["id"]: p["policy"] for p in ev["thresholds"]}, _boot(ev)
    per: Dict[str, Dict[str, float]] = {}
    for key, per_level in ctx.thr.items():
        m, sd, fold = key.split("|")
        for level, pols in per_level.items():
            for pid, e in pols.items():
                if "skipped" not in e:
                    per.setdefault(f"{m}|{sd}|{level}|{pid}", {})[fold] = e["threshold"]
    spread = {k: {"per_fold": v, "mean": float(np.mean(list(v.values()))), "sd": float(np.std(list(v.values()), ddof=1))
                  if len(v) > 1 else NAN, "min": min(v.values()), "max": max(v.values()),
                  "range": max(v.values()) - min(v.values())} for k, v in per.items()}
    rows, refit = [], bool(ev["bootstrap"]["refit_threshold"])
    for m, sd in _models(ctx):
        pops = {sp: policy_populations(ctx, m, sd, sp) for sp in SPLITS}
        if refit:
            rows += refit_rows(ctx, m, sd, pops, boot)
        for (level, pid), (ents, vpop) in pops["val"].items():
            if kinds.get(pid) != "fpr_cap" or (level, pid) not in pops["test"]:
                continue
            thr_j: Dict[int, Dict[int, float]] = {j: {} for j in range(-2, 3)}
            for fold, e in ents.items():
                v = np.sort(vpop[(vpop["fold"] == fold) & (vpop["y"] == 0)]["score"].to_numpy(np.float64))
                k0 = int(e["k"]) - 1 if e.get("k") else int(np.searchsorted(v, e["threshold"], side="left"))
                for j in thr_j:
                    thr_j[j][fold] = float(v[np.clip(k0 + j, 0, v.size - 1)])
            e0 = next(iter(ents.values()))
            key = dict(level=level, policy_id=pid, policy_basis=e0["basis"],
                       metric_type=e0["basis"] if e0["basis"] in _DENOM else None,
                       denominator=_DENOM.get(e0["basis"], "n/a"), axis=None if e0["at"] == "end" else e0["axis"],
                       t=NAN if e0["at"] == "end" else float(e0["at"]), point="end" if e0["at"] == "end" else "checkpoint",
                       subgroup="threshold_perturbation")
            for sp in SPLITS:
                pop = pops[sp][(level, pid)][1]
                for j, thr in thr_j.items():
                    k = _keys(ctx, m, sd, split=sp, subgroup_value=f"{j:+d}", **key)
                    for fold, p in pop.groupby("fold"):
                        vals = thresholded(p["y"], p["score"], thr[fold])
                        ci = _prop_ci(p["score"].to_numpy(np.float64) > thr[fold], p["y"].to_numpy() == 1, p["patient"],
                                      None, boot)
                        rows += metric_rows({x: vals[x] for x in ("sens", "spec", "fpr")}, ci,
                                            fold=str(fold), threshold=thr[fold], n_pos=vals["tp"] + vals["fn"],
                                            n_neg=vals["tn"] + vals["fp"], **k)
                    flagged = pop if sp == "test" else pop.assign(shared_test=pop["unit"].duplicated(keep=False))
                    r = pooled_confusion(flagged.assign(guid=flagged["unit"]), thr, policy, score="score")
                    rows += metric_rows({x: r[x] for x in ("sens", "spec", "fpr")},
                                        _pooled_ci(flagged, thr, policy, sp, None, boot, derived=False), fold="pooled",
                                        n_pos=r["tp"] + r["fn"], n_neg=r["tn"] + r["fp"], **k)
    return {"rows": rows, **_meta(ctx), "a": spread,
            "b": ({"status": "done", "rows": "subgroup='threshold_refit' (test; per fold: the chosen threshold with its "
                   "refit interval)", "resamples": boot["resamples"], "seed": boot["seed"]} if refit else
                  {"status": "off", "reason": "eval.bootstrap.refit_threshold is false"}),
            "plan": {"b": "refit bootstrap (§11.7 b): per replicate, every policy re-selected on the fold's resampled "
                          "val GUIDs, then the fold's (pooled: all folds') resampled test GUIDs scored",
                     "c": "threshold = v_(k*+j), j in -2..2, on the sorted val basis negatives of each fold"}}


def run_T4(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """T4: every policy under all three metric types at every checkpoint and end, indexed from the M
    rows: ``table[model|seed][split][policy][metric_type][t|'end'][metric] = [value, ci_lo, ci_hi,
    n_pos, n_neg]`` (pooled, all stages); needs ``M``. Not applicable when ``eval.time_axes`` leaves out
    ``to_delivery``, the axis the checkpoints live on."""
    if "to_delivery" not in eval_config["time_axes"]:
        return {**_meta(ctx), "table": {}, "plan": {"status": "not applicable: eval.time_axes has no to_delivery "
                                                              "axis, where the checkpoints live"}}
    df = pd.concat(ctx.frames, ignore_index=True) if ctx.frames else pd.DataFrame(columns=list(METRIC_COLUMNS))
    d = df[(df["level"] == "online") & (df["axis"] == "to_delivery") & df["point"].isin(["checkpoint", "end"])
           & df["subgroup"].isna() & (df["fold"] == "pooled") & df["policy_id"].notna()
           & df["metric"].isin(["sens", "spec", "fpr", "ppv", "npv"])]
    if not len(d):
        raise RuntimeError("T4 reads the time-resolved rows; run the 'M' analysis first")
    table: Dict[str, Any] = {}
    for r in d.itertuples(index=False):
        at = "end" if r.point == "end" else f"{r.t:g}"
        (table.setdefault(f"{r.model_id}|{r.seed}", {}).setdefault(r.split, {}).setdefault(r.policy_id, {})
         .setdefault(r.metric_type, {}).setdefault(at, {}))[r.metric] = [r.value, r.ci_lo, r.ci_hi, int(r.n_pos), int(r.n_neg)]
    return {**_meta(ctx), "table": table, "plan": {
        "rows": "metrics.parquet: level online, axis to_delivery, point checkpoint|end, policy_id x metric_type x t"}}


#: The analyses, in run order (§11.14): P2, then P4. ``metrics`` precedes the blocks that read its
#: rows, R1 writes ``roc_points`` before R2/R5 append to it, and M precedes T4.
ANALYSES: Dict[str, Callable[..., Dict[str, Any]]] = {
    "C": run_C, "metrics": run_metrics, "R1": run_R1, "R9": run_R9, "T1": run_T1, "T2": run_T2, "B1": run_B1,
    "M": run_M, "A": run_A, "R2": run_R2, "R5": run_R5, "R8": run_R8, "T3": run_T3, "T4": run_T4,
}


def headline(df: pd.DataFrame, alpha: float) -> Dict[str, Dict[str, Any]]:
    """``results.headline``: per (model_id, seed, level, policy, metric) the pooled test value with CI,
    the per-fold test mean +/- SD over the finite folds (``n_folds_nan`` counts the others, which verify
    surfaces), and the (optimistic) validation values (§11.9)."""
    d = df[df["metric"].isin((*_HEADLINE, f"pauc@{alpha:g}")) & df["subgroup"].isna()
           & (df["policy_id"].fillna("") != "oracle") & (df["point"] == "n/a")]
    out = {}
    for (m, sd, lvl, pid, met), g in d.groupby(["model_id", "seed", "level", "policy_id", "metric"], dropna=False):
        pid = pid if isinstance(pid, str) else "threshold_free"
        t, v = g[g["split"] == "test"], g[g["split"] == "val"]
        pt, per = t[t["fold"] == "pooled"], t[t["fold"] != "pooled"]["value"]
        pv, vper = v[v["fold"] == "pooled"]["value"], v[v["fold"] != "pooled"]["value"]
        out[f"{m}|{sd}|{lvl}|{pid}|{met}"] = {
            "model_id": m, "seed": sd, "level": lvl, "policy_id": pid, "metric": met,
            "test": pt["value"].iloc[0] if len(pt) else None,
            "test_ci": [pt["ci_lo"].iloc[0], pt["ci_hi"].iloc[0]] if len(pt) else None,
            "fold_mean": per.mean(), "fold_sd": per.std(), "n_folds": int(per.size),
            "n_folds_nan": int((~np.isfinite(per.to_numpy(np.float64))).sum()),
            "val": pv.iloc[0] if len(pv) else None, "val_fold_mean": vper.mean(),
            "primary": "fold_mean" if met in ("auroc", f"pauc@{alpha:g}") else "pooled",
        }
    return out


def _write_tables(ctx: SimpleNamespace, rep: Report, out: Path) -> None:
    df = _rows_frame(ctx)
    df.to_parquet(out / "tables" / "metrics.parquet", index=False)
    rep.set("headline", headline(df, primary_alpha(ctx.cfg["eval"])))
    rep.set("tables", {name: int(len(pd.read_parquet(out / "tables" / f"{name}.parquet")))
                       for name in P2_TABLES if (out / "tables" / f"{name}.parquet").is_file()})
    if ctx.inclusion:  # L14
        pd.DataFrame(ctx.inclusion).to_csv(out / "tables" / "inclusion.csv", index=False)


def run_context(run_dir: Path) -> Dict[str, Any]:
    """Provenance for summary.json: manifest's git/software/fingerprint plus this process's versions.
    ``checkpoint_sha256`` is the VAE checkpoint's (the source fingerprint's); None for an hdf5 source."""
    man = _read_json(Path(run_dir) / "manifest.json") or {}
    src = man.get("source") or {}
    return {"software": man.get("software"), "versions_at_run": man.get("versions"),
            "source_fingerprint": src.get("fingerprint_hash"),
            "checkpoint_sha256": (src.get("fingerprint") or {}).get("checkpoint_sha256"),
            "evaluate_env": {"python": platform.python_version(),
                             **{p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy")}}}


def evaluate(run_dir: Any, cfg: Any, *, only: Optional[Sequence[str]] = None,
             skip: Optional[Sequence[str]] = None, allow_partial: bool = False) -> int:
    """Run the analyses fail-soft into ``<run>/evaluation/`` (§11.13-§11.14).

    Each analysis runs under ``Report.step``; ``steps.json`` is rewritten after each one. A full run
    renames the prior ``summary.json``/``steps.json`` to ``*.bak.<stamp>.json`` and ``tables/`` to
    ``tables.bak.<stamp>``. A partial run (``only``/``skip``) never touches those canonical outputs,
    which report and verify read: it writes the same tree to ``evaluation/partial_<stamp>/``.
    ``allow_partial`` (the run's ``--allow-partial``) is recorded for verify, as is the predictions'
    ``written_at``.

    Returns:
        1 if and only if a step raised, else 0.

    Raises:
        ValueError: Unknown ``only``/``skip`` names, or a provenance config-digest mismatch.
    """
    from teb_vae.lag_attn_cfs.eval.report_seam import write_steps

    unknown = (set(only or ()) | set(skip or ())) - set(ANALYSES)
    if unknown:
        raise ValueError(f"unknown analyses {sorted(unknown)}; known: {list(ANALYSES)}")
    names = [n for n in ANALYSES if (not only or n in only) and n not in (skip or ())]
    ctx = load_context(run_dir, cfg)
    stamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f")
    out = ctx.run_dir / "evaluation"
    if only or skip:
        out = out / f"partial_{stamp}"
        logger.warning(f"partial evaluation ({names}): writing to {out}; evaluation/tables, summary.json and "
                       "steps.json stay those of the last full evaluation")
    for prior in (out / "summary.json", out / "steps.json", out / "tables"):
        if prior.exists():
            prior.rename(out / f"{prior.stem}.bak.{stamp}{prior.suffix}")
            logger.warning(f"preserved the prior {prior.name} as {prior.stem}.bak.{stamp}{prior.suffix}")
    (out / "tables").mkdir(parents=True)

    # report lists only artifacts written since (stale figures): the new directory's mtime, on the clock that stamps them
    rep, started = Report(), (out / "tables").stat().st_mtime
    rep.set("evaluate_started_at", started)
    rep.set("partial", bool(only or skip))
    for name in names:
        res = rep.step(name, ANALYSES[name], ctx, eval_config=ctx.cfg["eval"], out_dir=out)
        if res is not None:
            ctx.rows += res.pop("rows", [])
            frame = res.pop("frame", None)
            if frame is not None:
                ctx.frames.append(frame)
            ctx.sanity.update(res.pop("sanity", {}))
            rep.set(name, res)
        write_steps(rep.steps, out)
    rep.step("tables", _write_tables, ctx, rep, out)
    write_steps(rep.steps, out)
    rep.set("missing_units", ctx.prov.get("missing_units"))  # None: unrecorded (verify: INCONCLUSIVE, not a pass)
    rep.set("allow_partial", allow_partial)  # verify: missing units FAIL, or INCONCLUSIVE under --allow-partial
    rep.set("predictions_written_at", ctx.prov.get("written_at"))  # verify: a later predict makes this stale
    rep.set("prediction_units", ctx.prov.get("units", []))  # verify 4: each unit's lock_written_at < test_written_at
    rep.set("sanity", {"checks": ctx.sanity, "failed": sorted(k for k, v in ctx.sanity.items() if v["verdict"] == "fail")})
    rep.set("config_digest", ctx.digest)
    rep.set("config", ctx.cfg)
    rep.set("labels", {"val": VAL_LABEL, "oracle": "tpr@fpr rows: read off the test ROC, never a decision",
                       "auroc": "per-fold mean +/- SD primary; pooled OOF AUROC secondary"})
    rep.set("arguments", {"run_dir": str(ctx.run_dir), "only": list(only or []), "skip": list(skip or [])})
    rep.set("analyses_selected", names)
    rep.set("run_context", run_context(ctx.run_dir))
    rep.set("artifacts", build_manifest(out, since=started))
    rep.write(out)
    return rep.exit_code()


# ==== P6 blocks. Each section below belongs to one block; register analyses with ``ANALYSES.update(...)`` (run order = section order) at its end. ====
# ---- block S (subgroups: S1-S7, R11, K3, X10) ----
#
# §11.6 and §11.12 block S, with R11 (subgroup ROC), K3 (calibration per subgroup) and X10 (per-class x subgroup).
# Membership (:func:`subgroup_members`) is computed once per evaluation from GUID-level fields and cached on the
# context. Each analysis restricts a population the engine above already built (``level_units``,
# ``policy_populations``, ``_tr_groups``) with a (cell, GUID) membership matrix (:func:`_member_matrix`; a *cell* is
# one member, i.e. one (subgroup, subgroup_value)), so no subgroup ever changes a threshold. Tables:
# ``subgroups.parquet`` (S1, S4, S5 and the K3 reliability points: §11.9 columns plus :data:`S_EXTRA`),
# ``subgroup_tests.parquet`` (S6) and ``subgroup_cutpoints.json``; the S2 and X10 time-resolved rows go to
# ``metrics.parquet``, the R11 curves to ``roc_points.parquet``. Scale (§11.10): S2, X10, R11 and K3 cover the primary
# model (:func:`primary_model`) only, S2 and the X10 bin grid the primary policy; the shuffled-label control gets no
# subgroup analysis (its labels are permuted, so its subgroups mean nothing).

from teb_vae.lag_attn.eval import stats as vae_stats  # noqa: E402

TERTILE_FAMILIES = ("admission_tlo_tertile", "labour_duration_tertile", "n_segments_tertile", "span_tertile",
                    "valid_frac_tertile")
#: The §11.6.1 families in table order; ``covariate:<name>`` families come from the declared covariates.
SUBGROUP_FAMILIES = ("class", "source_file", "class_x_cs", "healthy_x_bg", "healthy_bg_x_cs", "cs", "bg", "stage_last",
                     "reached_second_stage", "has_tlo", *TERTILE_FAMILIES, "late_coverage", "shared_test", "fold")
#: Families whose members hold one clinical class by definition (X10).
SINGLE_CLASS_FAMILIES = ("class", "source_file", "class_x_cs", "healthy_x_bg", "healthy_bg_x_cs")
#: Documented, never computed (§11.6.1); the report states why instead of drawing an empty line.
EMPTY_FAMILIES = {"acidosis_x_bg": "every acidosis GUID has bg = 1 (§2.3)", "hie_x_bg": "every HIE GUID has bg = 1 (§2.3)"}
#: Member-name tokens in display and test order (:func:`member_order`): worst cohort first as ``labels.ordered_groups``
#: (HIE, acidosis, healthy; the shard cells in reversed canonical order), positive before negative, stages in time
#: order, tertiles low to high, unknown last. S6 orients every pair by it, so Cliff's delta > 0 = the earlier member higher.
MEMBER_ORDER = ("hie", "acidosis", "unhealthy", "healthy", "bg", "cs", "pos", "neg", "first", "straddle", "second",
                "T1", "T2", "T3", "yes", "no", "available", "missing", "unknown")
_MEMBER_RANK = {t: i for i, t in enumerate(MEMBER_ORDER)}
S2_FAMILIES = ("class", "class_x_cs", "healthy_x_bg", "healthy_bg_x_cs", "stage_last", "has_tlo", *TERTILE_FAMILIES)
ROC_FAMILIES = ("cs", "bg", "stage_last", "has_tlo", *TERTILE_FAMILIES)  # R11: the mixed families
K3_FAMILIES = ("cs", "bg", "stage_last", "has_tlo")
#: S6 family-wise error rate: a module constant, not a config key (the VAE rationale, §11.6.2).
S_ALPHA = 0.05
#: ``subgroups.parquet`` columns beyond §11.9: the writing analysis, the member's population (healthy_only |
#: unhealthy_only | mixed, from the task targets its GUIDs carry), the power guard (value NaN, the estimate kept in
#: ``value_raw``) and S4's bootstrap p-values.
S_EXTRA = ("analysis", "population", "underpowered", "value_raw", "p_value", "p_holm")
RESTRICTED = "restricted_pair"


def subgroup_families(cfg: Mapping[str, Any]) -> list:
    """The run's families: ``eval.subgroup_families`` without the documented-empty ones (:data:`EMPTY_FAMILIES`), plus
    ``covariate:<name>`` per declared covariate (§7.3, S7). Empty for a config without them."""
    ev, cov = cfg.get("eval") or {}, ((cfg.get("context") or {}).get("covariates") or {}).get("variables") or []
    return list(dict.fromkeys([*(f for f in ev.get("subgroup_families") or [] if f not in EMPTY_FAMILIES),
                               *(f"covariate:{v['name']}" for v in cov)]))


def _member_key(v: Any) -> list:
    return [(_MEMBER_RANK.get(t, len(_MEMBER_RANK)), int(t) if t.isdigit() else 0, t) for t in str(v).split("_")]


def member_order(values: Any) -> list:
    """Distinct member names in :data:`MEMBER_ORDER` (worst cohort first)."""
    return sorted(dict.fromkeys(map(str, values)), key=_member_key)


def primary_model(models: Sequence[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
    """The (model_id, seed) the subgroup figures and the scale-limited analyses use: ``model``'s seed ensemble (§10.7),
    else ``model``, else the first model that is no baseline or control, else the first one (a probe-only run); None
    without models."""
    models = list(models)
    return (next((x for x in models if x == ("model", "ens")), None) or next((x for x in models if x[0] == "model"), None)
            or next((x for x in models if x[0] not in BASELINES), None) or (models[0] if models else None))


def subgroup_members(guids: pd.DataFrame, seg: pd.DataFrame, families: Sequence[str], *,
                     cutpoints: Optional[Mapping[str, Sequence[float]]] = None,
                     covariates: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, list]]:
    """§11.6.1 membership, per GUID from GUID-level fields (never the row-level target).

    Args:
        guids: prediction GUID rows (``predictions/guids.parquet`` columns ``fold, split, guid, clinical_class,
            subgroup`` (the shard), ``cs, bg, has_tlo, shared_test, n_segments, first_epoch_s, last_t_end_s``); any
            models, de-duplicated on (fold, split, guid).
        seg: their segment rows (``seg_pos, stage, tlo_end_h, hours_to_delivery, valid_frac``), de-duplicated on
            (fold, split, guid, seg_pos): stage_last and reached_second_stage (the trim-aware ``stage``), admission
            TLO and labour duration (first segment: ``tlo_end_h``, ``+ hours_to_delivery``), mean ``valid_frac``.
        families: family ids (:func:`subgroup_families`); ``covariate:<name>`` reads ``covariates``.
        cutpoints: ``{tertile family: [q1/3, q2/3]}`` to reuse (``subgroup_cutpoints.json``); a family it lacks gets
            the cut-points of the pooled test GUIDs, each GUID once (§11.6.1: descriptive strata, reused for val).
        covariates: ``cohort/covariate_availability.parquet`` rows (fold, split, guid, variable, available).

    Returns:
        ``(members, cutpoints)``. ``members`` has one row per (fold, split, guid, subgroup, subgroup_value) a GUID
        satisfies, ordered by family, then member (:func:`member_order`). ``unhealthy`` (acidosis or HIE) overlaps its
        classes in ``class`` and ``class_x_cs``. Tertiles are ``T1`` (<= q1/3), ``T2``, ``T3``, and ``unknown`` for a
        missing value (no TLO). Other members: ``cs_pos|cs_neg``, ``bg_pos|bg_neg``, ``healthy_bg_pos_cs_neg``, …,
        stages ``first|straddle|second|unknown``, ``yes|no`` (reached_second_stage also ``unknown``), the fold
        number, ``available|missing``.

    Raises:
        ValueError: an unknown family, or a documented-empty one (:data:`EMPTY_FAMILIES`).
    """
    bad = [f for f in families if f not in SUBGROUP_FAMILIES and not str(f).startswith("covariate:")]
    if bad:
        raise ValueError(f"unknown subgroup families {bad} (documented-empty, never computed: {sorted(EMPTY_FAMILIES)}); "
                         f"known: {list(SUBGROUP_FAMILIES)} and covariate:<name>")
    key = ["fold", "split", "guid"]
    g = guids.drop_duplicates(key).set_index(key)
    sd = seg.drop_duplicates([*key, "seg_pos"]).sort_values([*key, "seg_pos"])
    first, last = (sd.drop_duplicates(key, keep=k).set_index(key).reindex(g.index) for k in ("first", "last"))
    st = sd.assign(r=sd["stage"].isin(["straddle", "second"]), u=sd["stage"].eq("unknown")).groupby(key)
    reached = pd.Series(np.select([st["r"].any(), st["u"].all()], ["yes", "unknown"], "no"),
                        index=st.size().index).reindex(g.index)
    raw = {"admission_tlo_tertile": first["tlo_end_h"], "labour_duration_tertile": first["tlo_end_h"] + first["hours_to_delivery"],
           "n_segments_tertile": g["n_segments"].astype(float), "span_tertile": (g["last_t_end_s"] - g["first_epoch_s"]) / 3600.0,
           "valid_frac_tertile": sd.groupby(key)["valid_frac"].mean().reindex(g.index)}
    test, cut = g.index.get_level_values("split") == "test", {}
    for f, v in raw.items():
        x = v[test].groupby(level="guid").first().dropna().to_numpy(np.float64)
        cut[f] = np.quantile(x, [1 / 3, 2 / 3]).tolist() if x.size else []
    cutpoints = cut | dict(cutpoints or {})

    def tertile(v: pd.Series, cut: Sequence[float]) -> pd.Series:
        x, ok = v.to_numpy(np.float64), len(cut) == 2
        i = np.searchsorted(np.asarray(cut if ok else [0.0, 0.0], np.float64), np.nan_to_num(x), side="left")
        return pd.Series(np.where(np.isfinite(x) & ok, np.array(["T1", "T2", "T3"])[i], "unknown"), index=v.index)

    def flag(b: Any, yes: str = "yes", no: str = "no") -> pd.Series:
        return pd.Series(np.where(np.asarray(b, bool), yes, no), index=g.index)

    cls, healthy = g["clinical_class"].astype(str), g["clinical_class"].astype(str) == "healthy"
    unh = pd.Series("unhealthy", index=g.index).where(~healthy)
    cs, bg = flag(g["cs"], "cs_pos", "cs_neg"), flag(g["bg"], "bg_pos", "bg_neg")
    defs = {
        "class": [cls, unh], "source_file": [g["subgroup"].astype(str)], "class_x_cs": [cls + "_" + cs, unh + "_" + cs],
        "healthy_x_bg": [("healthy_" + bg).where(healthy)], "healthy_bg_x_cs": [("healthy_" + bg + "_" + cs).where(healthy)],
        "cs": [cs], "bg": [bg], "stage_last": [last["stage"]], "reached_second_stage": [reached],
        "has_tlo": [flag(g["has_tlo"])], "late_coverage": [flag(g["last_t_end_s"] >= -3600.0)],
        "shared_test": [flag(g["shared_test"])],
        "fold": [pd.Series(g.index.get_level_values("fold").astype(str), index=g.index)],
        **{f: [tertile(raw[f], cutpoints[f])] for f in TERTILE_FAMILIES},
    }
    if covariates is not None and len(covariates):
        cov = covariates.drop_duplicates([*key, "variable"]).set_index(key)
        for f in families:
            if f.startswith("covariate:"):
                a = cov.loc[cov["variable"] == f.split(":", 1)[1], "available"]
                defs[f] = [pd.Series(np.where(a.astype(bool), "available", "missing"), index=a.index).reindex(g.index)]
    parts = [s.rename("subgroup_value").dropna().astype(str).to_frame().assign(subgroup=f)
             for f in families for s in defs.get(f, [])]
    out = (pd.concat(parts).reset_index() if parts else pd.DataFrame(columns=[*key, "subgroup_value", "subgroup"]))
    rank, cell = {f: i for i, f in enumerate(families)}, list(zip(out["subgroup"], out["subgroup_value"]))
    pos = {c: i for i, c in enumerate(sorted(set(cell), key=lambda c: (rank[c[0]], _member_key(c[1]))))}
    out = out.iloc[np.argsort([pos[c] for c in cell], kind="stable")].reset_index(drop=True)
    return out[[*key, "subgroup", "subgroup_value"]].astype({"subgroup": str, "subgroup_value": str}), cutpoints


def _members(ctx: SimpleNamespace) -> pd.DataFrame:
    """The run's membership (:func:`subgroup_members` over every model's GUIDs), cached with ``ctx.cutpoints``."""
    if "members" not in ctx.__dict__:
        path = ctx.run_dir / "cohort" / "covariate_availability.parquet"
        ctx.members, ctx.cutpoints = subgroup_members(ctx.guids, ctx.seg, subgroup_families(ctx.cfg),
                                                      covariates=pd.read_parquet(path) if path.is_file() else None)
    return ctx.members


def _cells(ctx: SimpleNamespace) -> pd.DataFrame:
    """One row per cell in membership order: ``subgroup, subgroup_value, population`` (healthy_only | unhealthy_only |
    mixed: the task targets ``y`` its GUIDs carry over every fold and split) and ``class_mask`` (bit k set: it holds a
    GUID of class k = ``class_code - 1``)."""
    if "cells" not in ctx.__dict__:
        key = ["fold", "split", "guid"]
        x = _members(ctx).merge(ctx.guids.drop_duplicates(key)[[*key, "y", "class_code"]], on=key)
        g = x.assign(bit=np.left_shift(1, x["class_code"].to_numpy(np.int64) - 1)).groupby(
            ["subgroup", "subgroup_value"], sort=False)
        c = g["y"].agg(["min", "max"]).assign(class_mask=g["bit"].agg(np.bitwise_or.reduce))
        c["population"] = np.select([c["max"] == 0, c["min"] == 1], ["healthy_only", "unhealthy_only"], "mixed")
        ctx.cells = c.reset_index()[["subgroup", "subgroup_value", "population", "class_mask"]]
    return ctx.cells


def _member_matrix(ctx: SimpleNamespace, folds: Any, guids: Any) -> np.ndarray:
    """(C, n) membership of n distinct rows (fold, guid) in each :func:`_cells` cell."""
    if "member_index" not in ctx.__dict__:
        M, cells = _members(ctx), _cells(ctx)
        cid = pd.MultiIndex.from_frame(cells[["subgroup", "subgroup_value"]]).get_indexer(
            pd.MultiIndex.from_frame(M[["subgroup", "subgroup_value"]]))
        ctx.member_index = (cid, pd.MultiIndex.from_frame(M[["fold", "guid"]]))
    cid, rows = ctx.member_index
    guids = np.asarray(guids)
    at = pd.MultiIndex.from_arrays([np.asarray(folds), guids]).get_indexer(rows)
    out, ok = np.zeros((len(_cells(ctx)), guids.size), bool), at >= 0
    out[cid[ok], at[ok]] = True
    return out


def _policy_pops(ctx: SimpleNamespace, m: str, sd: str, split: str) -> Dict[Tuple[str, str], Tuple[dict, pd.DataFrame]]:
    """:func:`policy_populations`, cached for the block S analyses (S1, S4), without the frames' ``attrs`` (pandas deep-copies
    them on every operation)."""
    cache = ctx.__dict__.setdefault("s_pops", {})
    if (m, sd, split) not in cache:
        cache[(m, sd, split)] = pops = policy_populations(ctx, m, sd, split)
        for _, pop in pops.values():
            pop.attrs = {}
    return cache[(m, sd, split)]


def _rank_point(y: np.ndarray, s: np.ndarray, alpha: float) -> Dict[str, float]:
    """AUROC, AUPRC and pAUC@alpha point values (:func:`_rank_stats`: sklearn's; NaN with one class)."""
    o, st, _ = _ranked(np.asarray(s, np.float64))
    y = np.asarray(y, np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {m: float(v[0]) for m, v in _rank_stats(*_curve(y, np.ones((1, y.size)), o, st), alpha).items()}


def _s_models(ctx: SimpleNamespace) -> list:
    return [x for x in _models(ctx) if x[0] != "shuffled"]


def _folds(units: pd.DataFrame, policy: str, split: str) -> list:
    """``[(fold label, rows)]``: every fold, then the pooled OOF rows (:func:`pool_rows`)."""
    return [*((str(f), x) for f, x in units.groupby("fold")), ("pooled", pool_rows(units, policy, split))]


def _guid_col(ctx: SimpleNamespace, m: str, sd: str, split: str, x: pd.DataFrame, col: str) -> np.ndarray:
    """``col`` of ``predictions/guids.parquet`` for the rows (fold, guid) of ``x``."""
    g = _sel(ctx.guids, model_id=m, seed=sd, split=split).set_index(["fold", "guid"])[col]
    return g.reindex(pd.MultiIndex.from_frame(x[["fold", "guid"]])).to_numpy()


def _append_subgroups(out_dir: Path, frame: pd.DataFrame, analysis: str) -> int:
    """Replace ``analysis``' rows of ``subgroups.parquet`` with ``frame`` (§11.9 columns + :data:`S_EXTRA`)."""
    path, cols = out_dir / "tables" / "subgroups.parquet", [*METRIC_COLUMNS, *S_EXTRA]
    new = frame.reindex(columns=cols).assign(analysis=analysis)
    for c in ("value", "ci_lo", "ci_hi", "t", "threshold", "n_pos", "n_neg", "value_raw", "p_value", "p_holm", "n_boot_undefined"):
        new[c] = pd.to_numeric(new[c], errors="coerce").astype(float)
    new["underpowered"] = new["underpowered"].fillna(False).astype(bool)
    for c in (*(c for c in METRIC_COLUMNS if c not in _NUMERIC), "analysis", "population"):
        new[c] = new[c].astype(object).where(new[c].notna(), None)
    if path.is_file():
        old = pd.read_parquet(path)
        new = pd.concat([old[old["analysis"] != analysis], new], ignore_index=True)
    new.to_parquet(path, index=False)
    return int((new["analysis"] == analysis).sum())


def _s_frame(blocks: list) -> pd.DataFrame:
    """:func:`_blocks_frame` of column blocks plus the :data:`S_EXTRA` columns they carry (missing: None)."""
    if not blocks:
        return pd.DataFrame(columns=[*METRIC_COLUMNS, *S_EXTRA])
    f = _blocks_frame(blocks)
    for c in S_EXTRA:
        f[c] = np.concatenate([np.broadcast_to(np.asarray(b.get(c), object), len(b["value"])) for b in blocks])
    return f


def _s1_rates(ctx: SimpleNamespace, x: pd.DataFrame, fire: np.ndarray, mn: int, boot: Mapping[str, Any],
              **keys: Any) -> Dict[str, Any]:
    """S1 thresholded column block (:func:`_s_frame`) of one population ``x`` (fold, guid, y, patient) and its decisions ``fire``, per cell: what
    the cell's population reports (§11.6): healthy-only tn, fp, spec, fpr; unhealthy-only tp, fn, sens; mixed those
    plus PPV and NPV. Rates carry :func:`rate_ci` CIs (Wilson; the patient-cluster bootstrap where a patient holds two
    of the cell's rows); a rate whose class count is below ``mn`` (``eval.min_subgroup_n``) is NaN and
    ``underpowered``, its estimate in ``value_raw``. PPV/NPV are 0 when the rule never fires (:func:`confusion_rates`).
    ``keys`` fill the other columns."""
    cells, Mm = _cells(ctx), _member_matrix(ctx, x["fold"], x["guid"])
    pos, n = x["y"].to_numpy() == 1, len(x)
    num = np.stack([fire & pos, ~fire & ~pos, fire & ~pos, fire & pos, ~fire & ~pos])  # the _PROP order
    den = np.stack([pos, ~pos, ~pos, fire, ~fire])
    N, D = Mm[None] & num[:, None], Mm[None] & den[:, None]  # (5, C, n)
    k, d = N.sum(2), D.sum(2)
    lo, hi, nu = (v.reshape(5, -1) for v in rate_ci(N.reshape(-1, n), D.reshape(-1, n), pos, x["patient"], **boot))
    n_pos, n_neg = d[0], d[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(d > 0, k / d, NAN)
    rate[3:] = np.where(d[3:] > 0, rate[3:], 0.0)
    small = np.stack([n_pos < mn, n_neg < mn, n_neg < mn, (n_pos < mn) | (n_neg < mn), (n_pos < mn) | (n_neg < mn)])
    counts = np.stack([k[0], n_neg - k[1], k[1], n_pos - k[0]]).astype(np.float64)  # tp, fp, tn, fn
    names = np.array(["tp", "fp", "tn", "fn", *_PROP])
    nan4 = np.full(counts.shape, NAN)
    value, raw = np.r_[counts, np.where(small, NAN, rate)], np.r_[counts, rate]
    under = np.r_[np.zeros(counts.shape, bool), small]
    lo, hi, nu = (np.r_[nan4, np.where(small, NAN, v)] for v in (lo, hi, nu))
    show = {"healthy_only": {"tn", "fp", "spec", "fpr"}, "unhealthy_only": {"tp", "fn", "sens"}, "mixed": set(names)}
    keep = np.array([[nm in show[p] for p in cells["population"]] for nm in names]) & (n_pos + n_neg > 0)[None]
    j, c = np.nonzero(keep)
    return keys | {
        "subgroup": cells["subgroup"].to_numpy(object)[c], "subgroup_value": cells["subgroup_value"].to_numpy(object)[c],
        "population": cells["population"].to_numpy(object)[c], "metric": names[j].astype(object), "value": value[j, c],
        "ci_lo": lo[j, c], "ci_hi": hi[j, c], "n_boot_undefined": nu[j, c], "value_raw": raw[j, c],
        "underpowered": under[j, c],
        "n_pos": n_pos[c], "n_neg": n_neg[c]}


def run_S1(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """S1 ``subgroups.parquet``: every model but the shuffled control, val and test, per fold and pooled
    (:func:`pool_rows`), GUID level, per member (§11.6.2 items 1-3):

    * counts ``n_guids``, ``n_segments``, ``prevalence`` (``n_pos``/``n_neg`` adverse/healthy GUIDs);
    * mixed members, threshold-free on ``score_final_cal``: pooled test, every :func:`threshold_free` metric (AUROC,
      pAUC, AUPRC, Brier, calibration intercept and slope, …) with patient-cluster bootstrap CIs on the ranking ones;
      per fold and pooled val, AUROC, AUPRC and pAUC point values (:func:`_rank_point`);
    * per GUID-level policy, on its basis population at each fold's own threshold: :func:`_s1_rates`.

    A metric needing a class with fewer than ``eval.min_subgroup_n`` GUIDs is NaN with ``underpowered`` (estimate in
    ``value_raw``). Also writes ``subgroup_cutpoints.json``.
    """
    ev, policy = eval_config, ctx.cfg["data"]["shared_test_policy"]
    alpha, boot, mn = primary_alpha(ev), _boot(ev), ev["min_subgroup_n"]
    _members(ctx)
    cells = _cells(ctx)
    sub, val, popn = (cells[c].to_numpy(object) for c in ("subgroup", "subgroup_value", "population"))
    (out_dir / "tables" / "subgroup_cutpoints.json").write_text(json.dumps({
        "basis": "pooled test GUIDs, each once; reused for val (descriptive strata, never inputs)",
        "tertiles": ctx.cutpoints, "rule": "T1 <= q1/3 < T2 <= q2/3 < T3; unknown = no value",
        "documented_empty": EMPTY_FAMILIES}, indent=2))
    if not len(cells):
        return {**_meta(ctx), "n_rows": 0, "plan": {"status": "not applicable: no subgroup family "
                                                               "(eval.subgroup_families is empty, no covariates)"}}
    blocks, rows, rank = [], [], ("auroc", "auprc", f"pauc@{alpha:g}")
    for m, sd in _s_models(ctx):
        for split in SPLITS:
            k0 = _keys(ctx, m, sd, split=split, level="guid")
            for fold, x in _folds(level_units(ctx, m, sd, split)["guid"], policy, split):
                Mm, y, sc = _member_matrix(ctx, x["fold"], x["guid"]), x["y"].to_numpy(np.int64), x["score"].to_numpy(np.float64)
                n_pos, n_neg = (Mm & (y == 1)).sum(1), (Mm & (y == 0)).sum(1)
                nseg = Mm @ np.nan_to_num(_guid_col(ctx, m, sd, split, x, "n_segments").astype(np.float64))
                live = np.flatnonzero(n_pos + n_neg)
                with np.errstate(invalid="ignore", divide="ignore"):
                    counts = {"n_guids": n_pos + n_neg, "n_segments": nseg, "prevalence": n_pos / (n_pos + n_neg)}
                blocks.append(k0 | {
                    "fold": fold, "denominator": "n/a", "subgroup": np.tile(sub[live], 3), "subgroup_value": np.tile(val[live], 3),
                    "population": np.tile(popn[live], 3), "metric": np.repeat(np.array(list(counts), object), live.size),
                    "value": np.concatenate([v[live] for v in counts.values()]).astype(np.float64),
                    "n_pos": np.tile(n_pos[live], 3), "n_neg": np.tile(n_neg[live], 3)})
                for i in live[popn[live] == "mixed"]:
                    k = k0 | dict(fold=fold, subgroup=sub[i], subgroup_value=val[i])
                    ex, n, under = {"population": "mixed"}, dict(n_pos=int(n_pos[i]), n_neg=int(n_neg[i])), bool(min(n_pos[i], n_neg[i]) < mn)
                    if fold == "pooled" and split == "test" and not under:
                        rows += [r | ex | {"underpowered": False, "value_raw": r["value"]} for r in _tf_rows(x[Mm[i]], alpha, boot, **k)]
                        continue
                    pt = _rank_point(y[Mm[i]], sc[Mm[i]], alpha)
                    rows += [r | ex | {"underpowered": under, "value_raw": pt[r["metric"]]} for r in metric_rows(
                        {mm: NAN if under else pt[mm] for mm in rank}, metric_type="threshold_free", denominator="n/a", **n, **k)]
            for (level, pid), (ents, pop) in _policy_pops(ctx, m, sd, split).items():
                if level != "guid":
                    continue
                e = next(iter(ents.values()))
                basis, thr = e["basis"], {f: v["threshold"] for f, v in ents.items()}
                k = _keys(ctx, m, sd, split=split, level=level, policy_id=pid, policy_basis=basis,
                          metric_type=basis if basis in _DENOM else None, denominator=_DENOM.get(basis, "n/a"),
                          axis=None if e["at"] == "end" else e["axis"], t=NAN if e["at"] == "end" else float(e["at"]))
                for fold, x in _folds(pop, policy, split):
                    fire = x["score"].to_numpy(np.float64) > x["fold"].map(thr).to_numpy(np.float64)
                    blocks.append(_s1_rates(ctx, x, fire, mn, boot, **k, fold=fold,
                                            threshold=NAN if fold == "pooled" else thr[int(fold)]))
    table = pd.concat([f for f in (_s_frame(blocks), pd.DataFrame(rows)) if len(f)], ignore_index=True) \
        if rows or blocks else pd.DataFrame()
    n = _append_subgroups(out_dir, table, "S1")
    return {**_meta(ctx), "n_rows": n, "plan": {
        "families": subgroup_families(ctx.cfg), "documented_empty": EMPTY_FAMILIES, "cells": int(len(cells)),
        "models": [list(x) for x in _s_models(ctx)], "min_subgroup_n": mn, "cutpoints": ctx.cutpoints,
        "population": "healthy_only: spec/FPR; unhealthy_only: sens; mixed: both, PPV/NPV and threshold-free",
        "threshold_free": "pooled test: threshold_free + patient-cluster bootstrap (ranking CIs); per fold and pooled "
                          "val: AUROC/AUPRC/pAUC point values"}}


def _cell_time_blocks(U: SimpleNamespace, V: SimpleNamespace, pts: pd.DataFrame, Mm: np.ndarray, cells: pd.DataFrame,
                      pidx: list, has_pos: np.ndarray, has_neg: np.ndarray, ev: Mapping[str, Any],
                      boot: Mapping[str, Any], *, suffix: str = "", **keys: Any) -> list:
    """Column blocks (:func:`_blocks_frame`) of the §11.5.1 metric types restricted to each cell (rows of ``Mm``, (C, G)
    over ``U``'s GUIDs) under the policies ``U.pids[pidx]``, at the points ``pts`` (the :func:`_view` ``V``): tp and
    sens for cells ``has_pos``, fp, spec and fpr for cells ``has_neg`` (:func:`rate_ci` CIs), and the ``underpowered``
    marker (a needed class below ``eval.min_subgroup_n`` GUIDs at the point: that class's rates are NaN, the other
    class's stay). Metric names carry ``suffix`` (X10: ``_ovr_c<k>``)."""
    mn, kind, G = ev["min_subgroup_n"], pts["kind"].to_numpy(), U.first.size
    C, P, K = len(cells), len(pidx), len(pts)
    thr = U.T[0, pidx] if keys["fold"] != "pooled" else np.full(P, NAN)
    yp, hp, hn = U.y[None, None, :, None], has_pos[:, None, None], has_neg[:, None, None]
    ex, la = U.ex[pidx][:, V.row], U.la[pidx][:, V.row] & V.avail
    c, p, k = (a.ravel() for a in np.indices((C, P, K)))
    blocks = []
    for mt, pop, dec in (("instantaneous", V.inst, ex), ("committed_cumulative", V.avail, la),
                         ("committed_overall", np.broadcast_to(V.elig[:, None], V.avail.shape), la)):
        in_c = Mm[:, None, :, None] & pop[None, None]  # (C, 1, G, K)
        hit = in_c & dec[None]  # (C, P, G, K)
        n_pos, n_neg = (in_c & yp).sum(2), (in_c & ~yp).sum(2)
        tp, fp = (hit & yp).sum(2), (hit & ~yp).sum(2)
        live = n_pos + n_neg > 0
        u_pos, u_neg = hp & (n_pos < mn), hn & (n_neg < mn)  # per class: a thin class voids only its own rates
        under = u_pos | u_neg

        def ci(num: np.ndarray, den: np.ndarray) -> Tuple[np.ndarray, ...]:
            flat = [np.moveaxis(np.broadcast_to(a, hit.shape), 2, 3).reshape(-1, G) for a in (num, den)]
            return tuple(v.reshape(C, P, K) for v in rate_ci(*flat, U.y, U.pat, **boot))

        (s_lo, s_hi, s_nu), (f_lo, f_hi, f_nu) = ci(hit & yp, in_c & yp), ci(hit & ~yp, in_c & ~yp)
        with np.errstate(invalid="ignore", divide="ignore"):
            sens, fpr = np.where(u_pos, NAN, tp / n_pos), np.where(u_neg, NAN, fp / n_neg)
        s_lo, s_hi, s_nu = (np.where(u_pos, NAN, v) for v in (s_lo, s_hi, s_nu))
        f_lo, f_hi, f_nu = (np.where(u_neg, NAN, v) for v in (f_lo, f_hi, f_nu))
        vals = {"tp": (tp, NAN, NAN, live & hp, NAN), "sens": (sens, s_lo, s_hi, live & hp, s_nu),
                "fp": (fp, NAN, NAN, live & hn, NAN), "spec": (1.0 - fpr, 1.0 - f_hi, 1.0 - f_lo, live & hn, f_nu),
                "fpr": (fpr, f_lo, f_hi, live & hn, f_nu), "underpowered": (under * 1.0, NAN, NAN, live, NAN)}
        t = pts["t_inst" if mt == "instantaneous" else "t_comm"].to_numpy(np.float64)
        col = dict(subgroup=cells["subgroup"].to_numpy(object)[c], subgroup_value=cells["subgroup_value"].to_numpy(object)[c],
                   policy_id=np.asarray(U.pids, object)[pidx][p], threshold=thr[p], t=t[k], point=kind[k].astype(object),
                   n_pos=np.broadcast_to(n_pos, (C, P, K)).ravel(), n_neg=np.broadcast_to(n_neg, (C, P, K)).ravel())
        col["policy_basis"] = np.asarray([U.basis[x] for x in col["policy_id"]], object)
        for met, (v, lo, hi, keep, nu) in vals.items():
            sel = np.broadcast_to(keep, (C, P, K)).ravel()
            if sel.any():
                blocks.append(keys | {x: a[sel] for x, a in col.items()} | {
                    "metric_type": mt, "denominator": _DENOM[mt], "metric": met + suffix,
                    **{x: np.broadcast_to(a, (C, P, K)).ravel()[sel].astype(np.float64)
                       for x, a in (("value", v), ("ci_lo", lo), ("ci_hi", hi), ("n_boot_undefined", nu))}})
    return blocks


def _sub_view(V: SimpleNamespace, pts: pd.DataFrame, sel: np.ndarray) -> Tuple[SimpleNamespace, pd.DataFrame]:
    """The :func:`_view` and points restricted to the points ``sel``."""
    return (SimpleNamespace(row=V.row[:, sel], avail=V.avail[:, sel], inst=V.inst[:, sel], elig=V.elig, c=V.c),
            pts[sel].reset_index(drop=True))


def run_S2(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """S2: the three metric types vs time per member (:func:`_cell_time_blocks` on the :func:`run_M` groups and views),
    primary model and policy, every axis, bins, checkpoints and end: every family for pooled test; the ``class`` family
    for the per-fold test groups and for val (the §11.10 core set). ``level='online'`` rows in ``metrics.parquet``."""
    ev, pid, boot = eval_config, eval_config["primary_policy"], _boot(eval_config)
    pm, cells, frames = primary_model(_models(ctx)), _cells(ctx), []
    has_pos, has_neg = (cells["population"] != "healthy_only").to_numpy(), (cells["population"] != "unhealthy_only").to_numpy()
    for split in SPLITS if pm else ():
        groups = _tr_groups(ctx, *pm, split)
        if not groups or pid not in groups[0][1].pids:
            continue
        grids = _grids(groups, ev["time_axes"], ev["bin_h"])
        for fold, U in groups:
            full = split == "test" and fold == "pooled"
            Mm = _member_matrix(ctx, U.o["fold"].to_numpy()[U.first], U.guid) & (
                True if full else (cells["subgroup"] == "class").to_numpy()[:, None])
            keys = _keys(ctx, *pm, split=split, fold=fold, level="online")
            for axis in ev["time_axes"]:
                pts = _points(axis, grids[axis], ev)
                V = _view(U, axis, pts)
                if V is not None:
                    frames += _cell_time_blocks(U, V, pts, Mm, cells, [U.pids.index(pid)], has_pos, has_neg, ev, boot,
                                                axis=axis, **keys)
    frame = _blocks_frame(frames) if frames else None
    return {"frame": frame, **_meta(ctx), "n_rows": 0 if frame is None else int(len(frame)), "plan": {
        "model": pm and list(pm), "policy": pid, "axes": list(ev["time_axes"]), "min_subgroup_n": ev["min_subgroup_n"],
        "scope": "pooled test: every family; per-fold test and val: the class family (§11.10 core set)"}}


def _deltas(ctx: SimpleNamespace, x: pd.DataFrame, fire: Optional[np.ndarray], boot: Mapping[str, Any],
            mn: int) -> list:
    """S4 records per cell of the rows ``x`` (pooled test): the paired patient-cluster bootstrap (the same draws for the
    cell and its complement, everyone else in ``x``) of cell minus complement: ``delta_auroc`` of ``x.score``
    (``fire`` None; mixed cells) or ``delta_sens`` (cells with adverse GUIDs) and ``delta_spec`` (cells with healthy
    ones) of the decisions ``fire``. Percentile CI; two-sided bootstrap p = 2 min(P(d <= 0), P(d >= 0)). Underpowered
    (a needed class below ``mn`` in the cell or in its complement): NaN, no test, the estimate in ``value_raw``. A cell covering every row has no complement (skipped)."""
    cells, Mm = _cells(ctx), _member_matrix(ctx, x["fold"], x["guid"])
    y = x["y"].to_numpy(np.int64)
    pos, a = y == 1, (1.0 - boot.get("confidence", 0.95)) / 2.0
    M, codes = _unit_draws(y, x["patient"], boot["resamples"], boot["seed"])
    W = np.vstack([np.ones(M.shape[1], np.float32), M])[:, codes].astype(np.float64)  # row 0: the sample itself
    if fire is None:
        o, st, _ = _ranked(x["score"].to_numpy(np.float64))

    def stat(w: np.ndarray, met: str) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            if met == "delta_auroc":
                return _rank_stats(*_curve(y, w, o, st))["auroc"]
            hit, cls = (fire, pos) if met == "delta_sens" else (~fire, ~pos)
            return (w * (hit & cls)).sum(1) / (w * cls).sum(1)

    out = []
    for i, c in enumerate(cells.itertuples(index=False)):
        inside, n_pos, n_neg = Mm[i], int((Mm[i] & pos).sum()), int((Mm[i] & ~pos).sum())
        if not n_pos + n_neg or inside.all():
            continue
        mets = (["delta_auroc"] if c.population == "mixed" else []) if fire is None else \
            ["delta_sens"] * (c.population != "healthy_only") + ["delta_spec"] * (c.population != "unhealthy_only")
        c_pos, c_neg = int(pos.sum()) - n_pos, int((~pos).sum()) - n_neg  # the complement's
        for met in mets:
            need = {"delta_auroc": min(n_pos, n_neg, c_pos, c_neg), "delta_sens": min(n_pos, c_pos),
                    "delta_spec": min(n_neg, c_neg)}[met]
            rec = dict(subgroup=c.subgroup, subgroup_value=c.subgroup_value, population=c.population, metric=met,
                       n_pos=n_pos, n_neg=n_neg, underpowered=need < mn, value=NAN, ci_lo=NAN, ci_hi=NAN, p_value=NAN,
                       value_raw=NAN)
            if need:  # both sides hold the needed class(es): an estimate exists; the bootstrap only when powered
                w = W if need >= mn else W[:1]
                d = stat(w * inside, met) - stat(w * ~inside, met)
                dr, rec["value_raw"] = d[1:][np.isfinite(d[1:])], float(d[0])
                if dr.size:
                    rec |= dict(value=float(d[0]), ci_lo=float(np.quantile(dr, a)), ci_hi=float(np.quantile(dr, 1 - a)),
                                p_value=float(min(1.0, 2.0 * min((dr <= 0).mean(), (dr >= 0).mean()))),
                                n_boot_undefined=int(d.size - 1 - dr.size))
            out.append(rec)
    return out


def run_S4(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """S4 Δ vs complement (§11.6.2 item 5), pooled test, every model but the shuffled control: ΔAUROC of the GUID final
    score and Δsens/Δspec at the primary policy (its basis population, each fold's threshold) from a paired
    patient-cluster bootstrap (:func:`_deltas`); ``p_holm`` Holm-adjusted within each (family, metric). Rows of
    ``subgroups.parquet`` (``analysis`` S4)."""
    ev, policy = eval_config, ctx.cfg["data"]["shared_test_policy"]
    boot, mn, pid, recs = _boot(ev), ev["min_subgroup_n"], ev["primary_policy"], []
    for m, sd in _s_models(ctx):
        k = _keys(ctx, m, sd, split="test", level="guid", fold="pooled", denominator="n/a")
        u = pool_rows(level_units(ctx, m, sd, "test")["guid"], policy, "test")
        recs += [r | k | {"metric_type": "threshold_free"} for r in _deltas(ctx, u, None, boot, mn)]
        ents, pop = _policy_pops(ctx, m, sd, "test").get(("guid", pid), (None, None))
        if ents:
            x = pool_rows(pop, policy, "test")
            fire = x["score"].to_numpy(np.float64) > x["fold"].map({f: v["threshold"] for f, v in ents.items()}).to_numpy(np.float64)
            basis = next(iter(ents.values()))["basis"]
            recs += [r | k | dict(policy_id=pid, policy_basis=basis, metric_type=basis if basis in _DENOM else None)
                     for r in _deltas(ctx, x, fire, boot, mn)]
    table = pd.DataFrame(recs)
    if len(table):
        table["p_holm"] = table.groupby(["model_id", "seed", "subgroup", "metric"])["p_value"].transform(
            lambda p: pd.Series(vae_stats.holm_adjust(p.tolist()), index=p.index))
    return {**_meta(ctx), "n_rows": _append_subgroups(out_dir, table, "S4"), "plan": {
        "definition": "cell minus complement (every other GUID of the pooled test split), same bootstrap draws",
        "bootstrap": {"method": "paired outcome-stratified patient-cluster percentile bootstrap", **boot},
        "p": "two-sided bootstrap p = 2 min(P(d <= 0), P(d >= 0)); Holm within (family, metric)", "policy": pid}}


def run_S5(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """S5 restricted pairs (``eval.restricted_pairs``): per pair the GUIDs of its two classes, the more severe one
    (:data:`MEMBER_ORDER`) positive, ``subgroup_value`` ``<positive>_vs_<other>``; the AUROC of the GUID final score, per
    fold (point) and pooled (patient-cluster bootstrap CI), val and test, every model but the shuffled control. The
    subtype sensitivity at each policy is the S1 ``class`` row of the positive class, and the shared FPR the ``healthy``
    one (§11.6.1: identical across pairs, reported once); S2 carries them vs time."""
    ev, policy = eval_config, ctx.cfg["data"]["shared_test_policy"]
    boot, mn, rows = _boot(ev), ev["min_subgroup_n"], []
    for m, sd in _s_models(ctx):
        for split in SPLITS:
            k0 = _keys(ctx, m, sd, split=split, level="guid", subgroup=RESTRICTED, metric_type="threshold_free",
                       denominator="n/a")
            for fold, x in _folds(level_units(ctx, m, sd, split)["guid"], policy, split):
                cls = _guid_col(ctx, m, sd, split, x, "clinical_class").astype(str)
                for pair in ev["restricted_pairs"]:
                    hi_, lo_ = member_order(pair)
                    sel = np.isin(cls, [hi_, lo_])
                    yp, s = cls[sel] == hi_, x["score"].to_numpy(np.float64)[sel]
                    n_pos, n_neg = int(yp.sum()), int((~yp).sum())
                    r = dict(value=NAN, ci_lo=NAN, ci_hi=NAN, value_raw=NAN, underpowered=min(n_pos, n_neg) < mn)
                    if n_pos and n_neg:
                        r["value_raw"] = _rank_point(yp, s, 0.3)["auroc"]
                        if not r["underpowered"]:  # CIs pooled only (per fold: point values, as S1)
                            ci = bootstrap_auroc(yp, s, x["patient"].to_numpy()[sel], boot["resamples"],
                                                 boot["seed"])["metrics"]["auroc"] if fold == "pooled" else {}
                            r |= {"value": r["value_raw"], "ci_lo": ci.get("ci_lo", NAN), "ci_hi": ci.get("ci_hi", NAN)}
                    rows.append(k0 | r | dict(fold=fold, subgroup_value=f"{hi_}_vs_{lo_}", metric="auroc", n_pos=n_pos,
                                              n_neg=n_neg, population="mixed"))
    return {**_meta(ctx), "n_rows": _append_subgroups(out_dir, pd.DataFrame(rows), "S5"), "plan": {
        "pairs": [f"{a}_vs_{b}" for a, b in (member_order(p) for p in ev["restricted_pairs"])],
        "score": "score_final_cal (binary, or the collapsed adverse score of a 3-class model)",
        "subtype_sensitivity": "S1 rows subgroup=class (positive class); shared FPR: S1 healthy row"}}


def run_S6(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """S6 ``subgroup_tests.parquet`` (§11.6.2 item 7; the ``cross_subgroup.py`` pattern), pooled test GUID final scores,
    every model but the shuffled control. Per family and stratum (``class``: one test across its disjoint classes;
    every other family: per clinical class, the members restricted to it; the overlapping ``unhealthy*`` members are
    left out), Kruskal-Wallis over the members with >= ``MIN_GROUP_SIZE`` GUIDs (smaller ones excluded and recorded),
    Holm across every omnibus test of the model; for the significant ones pairwise Mann-Whitney + Cliff's delta in
    :data:`MEMBER_ORDER` (delta > 0: the earlier, more severe member scores higher), Holm within the (family,
    stratum). alpha = :data:`S_ALPHA`."""
    policy, recs = ctx.cfg["data"]["shared_test_policy"], []
    cells = _cells(ctx)
    for m, sd in _s_models(ctx):
        u = pool_rows(level_units(ctx, m, sd, "test")["guid"], policy, "test")
        cls, s = _guid_col(ctx, m, sd, "test", u, "clinical_class").astype(str), u["score"].to_numpy(np.float64)
        Mm, omni, samples = _member_matrix(ctx, u["fold"], u["guid"]), [], []
        for fam in dict.fromkeys(cells["subgroup"]):
            idx = [i for i in np.flatnonzero(cells["subgroup"] == fam) if not cells["subgroup_value"].iat[i].startswith("unhealthy")]
            for stratum in ["all"] if fam == "class" else CLASSES[::-1]:
                keep = np.ones(len(u), bool) if stratum == "all" else cls == stratum
                groups = {cells["subgroup_value"].iat[i]: s[Mm[i] & keep] for i in idx}
                groups = {g: v for g, v in groups.items() if v.size}
                if not groups:
                    continue
                usable = {g: v for g, v in groups.items() if v.size >= vae_stats.MIN_GROUP_SIZE}
                rec = vae_stats.kruskal_across_groups(usable)
                omni.append(dict(subgroup=fam, stratum=stratum, kind="omnibus", test=rec["test"], statistic=rec["statistic"],
                                 p_value=rec["p_value"], n_groups=rec["n_groups"], note=rec.get("note"),
                                 n_per_group=json.dumps(rec["n_per_group"]),
                                 excluded=json.dumps({g: int(v.size) for g, v in groups.items() if g not in usable})))
                samples.append(usable)
        for r, p in zip(omni, vae_stats.holm_adjust([r["p_value"] for r in omni])):
            r |= dict(p_holm=p, significant=bool(np.isfinite(p) and p < S_ALPHA))
        recs += [r | dict(model_id=m, seed=sd) for r in omni]
        for r, smp in zip(omni, samples):
            pairs = vae_stats.pairwise_comparisons(smp) if r["significant"] else []
            for q, p in zip(pairs, vae_stats.holm_adjust([q["p_value"] for q in pairs])):
                recs.append(dict(subgroup=r["subgroup"], stratum=r["stratum"], kind="pairwise", test=q["test"],
                                 left=q["left"], right=q["right"], n_left=q["n_left"], n_right=q["n_right"],
                                 statistic=q.get("u_statistic", NAN), p_value=q["p_value"], p_holm=p,
                                 significant=bool(np.isfinite(p) and p < S_ALPHA), cliffs_delta=q["cliffs_delta"],
                                 magnitude=q["magnitude"], note=q.get("note"), model_id=m, seed=sd))
    cols = ["model_id", "seed", "split", "subgroup", "stratum", "kind", "test", "left", "right", "n_groups", "n_per_group",
            "excluded", "n_left", "n_right", "statistic", "p_value", "p_holm", "significant", "cliffs_delta", "magnitude",
            "alpha", "min_group_size", "unit", "note"]
    table = pd.DataFrame(recs).reindex(columns=cols).assign(split="test", alpha=S_ALPHA, min_group_size=vae_stats.MIN_GROUP_SIZE,
                                                          unit="GUID (score_final_cal)")
    for c in ("model_id", "seed", "subgroup", "stratum", "kind", "test", "left", "right", "n_per_group", "excluded",
              "magnitude", "note"):
        table[c] = table[c].astype(object).where(table[c].notna(), None)
    table.to_parquet(out_dir / "tables" / "subgroup_tests.parquet", index=False)
    om = table[table["kind"] == "omnibus"]
    return {**_meta(ctx), "n_omnibus": int(len(om)), "n_significant": int(om["significant"].fillna(False).astype(bool).sum()),
            "plan": {"alpha": S_ALPHA, "min_group_size": vae_stats.MIN_GROUP_SIZE, "correction": "holm across the omnibus "
                     "tests of a model; holm within a (family, stratum) for the pairs", "order": list(MEMBER_ORDER)}}


def run_R11(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """R11: R1 per member of the configured mixed families (:data:`ROC_FAMILIES`) and per restricted pair, pooled test,
    primary model: ``roc_points`` variants ``subgroup:<family>=<member>`` (with ``n_pos``/``n_neg``) and, where both
    classes reach ``eval.min_subgroup_n``, ``subgroup:<family>=<member>:band`` (the patient-cluster bootstrap band on
    ``ROC_GRID``, :func:`roc_band`); restricted pairs use family ``restricted_pair``."""
    ev, policy, pm, frames = eval_config, ctx.cfg["data"]["shared_test_policy"], primary_model(_models(ctx)), []
    boot, mn, fams = _boot(ev), ev["min_subgroup_n"], [f for f in subgroup_families(ctx.cfg) if f in ROC_FAMILIES]
    if pm:
        u = pool_rows(level_units(ctx, *pm, "test")["guid"], policy, "test")
        cells, Mm = _cells(ctx), _member_matrix(ctx, u["fold"], u["guid"])
        y, s, pat = u["y"].to_numpy(np.int64), u["score"].to_numpy(np.float64), u["patient"].to_numpy()
        cls = _guid_col(ctx, *pm, "test", u, "clinical_class").astype(str)
        sets = [(f"{c.subgroup}={c.subgroup_value}", Mm[i], y) for i, c in enumerate(cells.itertuples()) if c.subgroup in fams]
        sets += [(f"{RESTRICTED}={a}_vs_{b}", np.isin(cls, [a, b]), (cls == a).astype(np.int64))
                 for a, b in (member_order(p) for p in ev["restricted_pairs"])]
        key = dict(model_id=pm[0], seed=pm[1], fold="pooled", split="test", level="guid")
        for name, sel, yy in sets:
            yc, sc = yy[sel], s[sel]
            n_pos, n_neg = int(yc.sum()), int(yc.size - yc.sum())
            if not (n_pos and n_neg):
                continue
            r = roc_points(yc, sc)
            frames.append(pd.DataFrame({**key, "variant": f"subgroup:{name}", **r, "n_pos": n_pos, "n_neg": n_neg}))
            if min(n_pos, n_neg) >= mn:
                lo, hi = roc_band(yc, sc, pat[sel], ROC_GRID, **boot)
                frames.append(pd.DataFrame({**key, "variant": f"subgroup:{name}:band", "fpr": ROC_GRID,
                                            "tpr": tpr_at(r["fpr"], r["tpr"], ROC_GRID), "tpr_lo": lo, "tpr_hi": hi}))
    return {**_meta(ctx), "n_rows": _append_roc(out_dir, frames, ("subgroup:",)), "plan": {
        "model": pm and list(pm), "families": fams, "pairs": list(ev["restricted_pairs"]),
        "band": "pooled patient-cluster bootstrap 95% where both classes reach eval.min_subgroup_n"}}


def run_K3(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """K3 reliability per member of the configured :data:`K3_FAMILIES`, pooled test, primary model: the calibrated GUID
    probability in equal-mass bins (at most 10, at least 5 GUIDs each; binned as :func:`_ece`) as ``subgroups.parquet``
    rows ``metric_type='calibration_curve'``, ``metric='observed'``: ``t`` the bin's mean predicted probability,
    ``value`` the observed fraction with its Wilson 95% interval, ``n_pos``/``n_neg`` the bin's GUIDs. Slope and
    intercept per member are S1's pooled test rows."""
    policy, pm, rows = ctx.cfg["data"]["shared_test_policy"], primary_model(_models(ctx)), []
    fams = [f for f in subgroup_families(ctx.cfg) if f in K3_FAMILIES]
    if pm:
        u = pool_rows(level_units(ctx, *pm, "test")["guid"], policy, "test")
        cells, Mm = _cells(ctx), _member_matrix(ctx, u["fold"], u["guid"])
        y, p = u["y"].to_numpy(np.float64), expit(u["score"].to_numpy(np.float64))
        k = _keys(ctx, *pm, split="test", level="guid", fold="pooled", metric_type="calibration_curve", denominator="n/a",
                  point="bin", metric="observed")
        for i, c in enumerate(cells.itertuples(index=False)):
            yc, pc = y[Mm[i]], p[Mm[i]]
            if c.subgroup not in fams or not yc.size:
                continue
            nb = int(np.clip(yc.size // 5, 1, 10))
            ids = np.searchsorted(np.quantile(pc, np.linspace(0, 1, nb + 1))[1:-1], pc)
            hit, n, pred = np.bincount(ids, yc, nb), np.bincount(ids, None, nb), np.bincount(ids, pc, nb)
            ok = n > 0
            lo, hi = wilson(hit[ok], n[ok])
            rows += [k | dict(subgroup=c.subgroup, subgroup_value=c.subgroup_value, population=c.population, t=a / m,
                              value=h / m, ci_lo=l_, ci_hi=h_, n_pos=h, n_neg=m - h)
                     for a, h, m, l_, h_ in zip(pred[ok], hit[ok], n[ok], lo, hi)]
    return {**_meta(ctx), "n_rows": _append_subgroups(out_dir, pd.DataFrame(rows), "K3"), "plan": {
        "model": pm and list(pm), "families": fams, "bins": "equal-mass, at most 10, at least 5 GUIDs each; Wilson 95%",
        "slope_intercept": "subgroups.parquet S1 rows calib_slope / calib_intercept (pooled test)"}}


def _ovr_groups(ctx: SimpleNamespace, m: str, sd: str, split: str, k: int) -> Optional[list]:
    """:func:`_tr_groups` of class k's one-vs-rest view (§11.3 T5, as ``thresholds.ovr_thresholds`` selected it): target
    ``class_code - 1 == k``, online and segment scores ``logit(p_c<k>_cal)``, the GUID-level thresholds of
    ``thresholds.json[unit]['ovr'][k]``. None without them or without an online/segment class probability."""
    ents = {int(key.split("|")[2]): {p: e for p, e in ((per.get(str(k)) or {}).get("guid") or {}).items() if "skipped" not in e}
            for key, per in ctx.thr_ovr.items() if key.split("|")[:2] == [m, sd]}
    pids, col, seg = _tr_pids(ctx.cfg["eval"], ents), f"p_c{k}_cal", segments(ctx, m, sd, split)
    if not len(seg) or not pids or col not in seg:
        return None
    scored = seg[["logit_online_cal", "logit_seg_cal"]].notna().any(axis=1)
    if not scored.any() or seg.loc[scored, col].isna().any():
        return None
    seg = ovr_view(seg, k, OVR_SEGMENT_SCORES)
    if seg[list(OVR_SEGMENT_SCORES)].isna().all(axis=None):  # no causal per-segment class probability
        return None
    units = seg.drop_duplicates(["fold", "guid"])
    keep = pool_rows(units.assign(unit=units["guid"]), ctx.cfg["data"]["shared_test_policy"], split)[["fold", "guid"]]
    args = (ents, pids, {}, [], ctx.cfg["eval"]["alarm_rule"])
    return [*((str(f), _prep(g, *args)) for f, g in seg.groupby("fold")),
            ("pooled", _prep(seg.merge(keep, on=["fold", "guid"]), *args))]


def run_X10(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """X10: X5 (b) per single-class family member (:data:`SINGLE_CLASS_FAMILIES`): each class k's one-vs-rest policies
    (:func:`_ovr_groups`) under the three metric types, restricted to the member (:func:`_cell_time_blocks`: ``sens``
    for members holding class k, ``spec``/``fpr`` for members holding another class), pooled test, primary model.
    Metric names end ``_ovr_c<k>``. Rows: every policy at the to_delivery checkpoints and end (the table), and the
    primary policy on the bin grid of every axis (the figure). Not applicable without OvR thresholds."""
    ev, boot, pm, frames = eval_config, _boot(eval_config), primary_model(_models(ctx)), []
    if not ctx.thr_ovr or pm is None:
        return {**_meta(ctx), "n_rows": 0, "plan": {"status": "not applicable: no per-class OvR thresholds (§11.3 T5)"}}
    cells = _cells(ctx)
    single = cells["subgroup"].isin(SINGLE_CLASS_FAMILIES).to_numpy()
    for k in range(3):
        groups = _ovr_groups(ctx, *pm, "test", k)
        if not groups:
            continue
        grids = _grids(groups, ev["time_axes"], ev["bin_h"])
        mask = cells["class_mask"].to_numpy(np.int64)
        has_pos, has_neg = (mask >> k) & 1 == 1, (mask & ~(1 << k)) != 0
        U = groups[-1][1]
        Mm = _member_matrix(ctx, U.o["fold"].to_numpy()[U.first], U.guid) & single[:, None]
        keys = _keys(ctx, *pm, split="test", fold="pooled", level="online")
        for axis in ev["time_axes"]:
            pts = _points(axis, grids[axis], ev)
            V = _view(U, axis, pts)
            if V is None:
                continue
            for sel, pidx in (((pts["kind"] == "bin").to_numpy(), [U.pids.index(ev["primary_policy"])]
                               if ev["primary_policy"] in U.pids else []),
                              ((pts["kind"] != "bin").to_numpy() & (axis == "to_delivery"), list(range(len(U.pids))))):
                if sel.any() and pidx:
                    frames += _cell_time_blocks(U, *_sub_view(V, pts, sel), Mm, cells, pidx, has_pos, has_neg, ev,
                                                boot, suffix=f"_ovr_c{k}", axis=axis, **keys)
    frame = _blocks_frame(frames) if frames else None
    return {"frame": frame, **_meta(ctx), "n_rows": 0 if frame is None else int(len(frame)), "plan": {
        "model": list(pm), "families": list(SINGLE_CLASS_FAMILIES), "classes": "k = class_code - 1 (0 healthy, 1 acidosis, 2 hie)",
        "rows": "every policy at to_delivery checkpoints/end; the primary policy on every axis' bins; pooled test"}}


P2_TABLES += ("subgroups", "subgroup_tests")
ANALYSES.update({"S1": run_S1, "S2": run_S2, "S4": run_S4, "S5": run_S5, "S6": run_S6, "R11": run_R11, "K3": run_K3,
                 "X10": run_X10})


# ---- block KH (calibration and heterogeneity: K1, K2, K4-K7, H1-H4, R10, R12, R13) ----
#
# Rows keep the §11.9 long format. The views that are not the run's calibrated, primary-pooling one carry a
# ``subgroup`` tag, so every ``subgroup.isna()`` reader (headline, B1, T2, summary.md) skips them: ``calibration`` /
# ``uncalibrated`` (K1), ``prevalence_shift`` (K5), ``fold_heterogeneity`` (H1) and ``shared_test_policy`` (H3, as
# ``run_metrics``' thresholded sensitivity rows). K7 rows are time-resolved (``level='online'``, like M). K6 and H3
# write tables, R10 appends to ``roc_points``, and R12 writes the snapshot table R12 and R13 are drawn from. K2, K4,
# H2 and H4 are drawn by report.py from these rows and the existing ones.

from scipy.stats import chi2  # noqa: E402

#: The :func:`threshold_free` calibration metrics K1/K5 recompute (a monotone map leaves the ranking ones unchanged).
CALIB_NAMES = ("calib_intercept", "calib_slope", "ece", "ici", "e50", "e90", "brier", "scaled_brier", "logloss")
#: K6 threshold probabilities p_t over [0.05, 0.6] (§11.4).
DC_PTS = np.round(np.linspace(0.05, 0.6, 56), 2)
#: H3 metrics compared under the two shared-test policies (pooled test, GUID level).
H3_METRICS = ("auroc", "auprc", "brier", "calib_intercept", "calib_slope", "sens", "spec", "fpr", "ppv", "npv")


def _folds_and_pooled(u: pd.DataFrame, policy: str, split: str) -> list:
    """``[(fold label, rows)]``: every fold, then the pooled rows (:func:`pool_rows`)."""
    return [*((str(f), x) for f, x in u.groupby("fold")), ("pooled", pool_rows(u, policy, split))]


def _guid_units(ctx: SimpleNamespace, m: str, sd: str, split: str, score: str) -> pd.DataFrame:
    """GUID rows of one model/seed/split as pooling units: ``unit`` = guid, ``score`` = the column ``score``, ``patient``."""
    g = _sel(ctx.guids, model_id=m, seed=sd, split=split)
    return g.assign(unit=g["guid"], score=g[score], patient=cluster(g))


def prior_shifted(test: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    """K5: each test GUID's calibrated logit moved from its fold's val prevalence to its fold's test prevalence,
    ``logit p' = logit p - logit pi_val + logit pi_test`` (:func:`prior_shift`, §11.7; GUID-level prevalences)."""
    pv, pt = (x.groupby("fold")["y"].mean().clip(1e-6, 1 - 1e-6) for x in (val, test))  # ponytail: a one-class fold
    return prior_shift(test["score_final_cal"], test["fold"].map(pv).to_numpy(), test["fold"].map(pt).to_numpy())


def _nb_band(y: np.ndarray, p: np.ndarray, units: Any, boot: Mapping[str, Any],
             confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Pointwise percentile band of :func:`net_benefit` over :data:`DC_PTS` under the outcome-stratified cluster
    bootstrap (:func:`_unit_draws`), each draw one weighted matrix product."""
    M, codes = _unit_draws(y.astype(np.int64), units, boot["resamples"], boot["seed"])
    fire, pos = p[:, None] > DC_PTS[None, :], (y == 1)[:, None]
    A, n_pt = np.c_[fire & pos, fire & ~pos].astype(np.float32), DC_PTS.size
    draws = []
    for sl in _chunks(boot["resamples"], y.size):
        W = M[sl][:, codes]
        r = (W @ A) / W.sum(1, keepdims=True)
        draws.append(r[:, :n_pt] - r[:, n_pt:] * (DC_PTS / (1.0 - DC_PTS)))
    a = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(np.vstack(draws), [a, 1 - a], axis=0)
    return lo, hi


def run_K(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """K1, K5 and K6 at GUID level; needs ``metrics`` (K5 re-weights its policy rows).

    * K1: the :data:`CALIB_NAMES` metrics of the uncalibrated ``score_final``, val and test, per fold and pooled
      (``subgroup='calibration'``, ``subgroup_value='uncalibrated'``); the calibrated ones are ``run_metrics``'.
    * K5 (test): the same metrics of the prior-shift-corrected score (:func:`prior_shifted`;
      ``subgroup='prevalence_shift'``, ``subgroup_value='prior_shift'``), the GUID prevalences ``pi_val`` and
      ``pi_test`` (``subgroup_value='prevalence'``), and with ``eval.reference_prevalence`` every GUID-level policy's
      ``ppv@pi_ref`` / ``npv@pi_ref`` from its test sens/spec (:func:`adjust_ppv_npv`; ``subgroup_value='pi_ref'``).
      PPV/NPV at the test prevalence are the observed ``ppv``/``npv`` rows.
    * K6 -> ``decision_curve.parquet``: :func:`net_benefit` of the calibrated probability over :data:`DC_PTS`, test,
      per fold and pooled, the pooled one with a patient-cluster bootstrap band (``nb_lo``, ``nb_hi``).
    """
    if not ctx.rows:
        raise RuntimeError("K reads the metric rows; run the 'metrics' analysis first")
    ev, policy = eval_config, ctx.cfg["data"]["shared_test_policy"]
    alpha, boot, pi_ref = primary_alpha(ev), _boot(ev), ev["reference_prevalence"]
    rows, curves = [], []

    def calib(u: pd.DataFrame) -> Dict[str, float]:
        tf = threshold_free(u["y"], u["score"], alpha=alpha)
        return {k: tf[k] for k in CALIB_NAMES}

    for m, sd in _models(ctx):
        units = {sp: _guid_units(ctx, m, sd, sp, "score_final") for sp in SPLITS}
        for sp, u in units.items():
            k = _keys(ctx, m, sd, split=sp, level="guid", metric_type="threshold_free", denominator="n/a",
                      subgroup="calibration", subgroup_value="uncalibrated")
            for fold, x in _folds_and_pooled(u, policy, sp):
                rows += metric_rows(calib(x), fold=fold, n_pos=int(x["y"].sum()), n_neg=int((x["y"] == 0).sum()), **k)
        test, val = units["test"], units["val"]
        if not len(test) or not len(val):
            continue
        shifted = test.assign(score=prior_shifted(test, val))
        k = _keys(ctx, m, sd, split="test", level="guid", metric_type="threshold_free", denominator="n/a",
                  subgroup="prevalence_shift")
        pi_val = {f: v["y"].mean() for f, v in _folds_and_pooled(val, policy, "val")}
        for fold, x in _folds_and_pooled(shifted, policy, "test"):
            n = dict(n_pos=int(x["y"].sum()), n_neg=int((x["y"] == 0).sum()))
            rows += metric_rows(calib(x), fold=fold, subgroup_value="prior_shift", **n, **k)
            rows += metric_rows({"pi_val": pi_val.get(fold, NAN), "pi_test": x["y"].mean()}, fold=fold,
                                subgroup_value="prevalence", **n, **k)
        for fold, x in _folds_and_pooled(test.assign(score=test["score_final_cal"]), policy, "test"):
            y, p = x["y"].to_numpy(np.int64), expit(x["score"].to_numpy(np.float64))
            nb = net_benefit(y, p, DC_PTS).assign(model_id=m, seed=sd, split="test", fold=fold, n_pos=int(y.sum()),
                                                  n_neg=int(y.size - y.sum()), nb_lo=NAN, nb_hi=NAN)
            if fold == "pooled" and y.size:
                nb["nb_lo"], nb["nb_hi"] = _nb_band(y, p, x["patient"], boot)
            curves.append(nb)
    if pi_ref is not None:
        df = _rows_frame(ctx)
        d = df[(df["level"] == "guid") & (df["split"] == "test") & df["subgroup"].isna() & (df["point"] == "n/a")
               & df["metric"].isin(["sens", "spec"]) & df["policy_id"].notna() & (df["policy_id"] != "oracle")]
        for (m, sd, pid, fold), g in d.groupby(["model_id", "seed", "policy_id", "fold"], sort=False):
            v = g.drop_duplicates("metric").set_index("metric")
            if {"sens", "spec"} <= set(v.index):
                with np.errstate(invalid="ignore", divide="ignore"):  # a rule that never fires: PPV 0/0
                    ppv, npv = adjust_ppv_npv(v.at["sens", "value"], v.at["spec", "value"], pi_ref)
                rows += metric_rows({"ppv@pi_ref": ppv, "npv@pi_ref": npv}, **_keys(
                    ctx, m, sd, split="test", level="guid", fold=fold, policy_id=pid, policy_basis=g["policy_basis"].iloc[0],
                    threshold=g["threshold"].iloc[0], metric_type=g["metric_type"].iloc[0],
                    denominator=g["denominator"].iloc[0], axis=g["axis"].iloc[0], t=g["t"].iloc[0],
                    subgroup="prevalence_shift", subgroup_value="pi_ref",
                    n_pos=g["n_pos"].iloc[0], n_neg=g["n_neg"].iloc[0]))
    table = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(
        columns=["pt", "net_benefit", "treat_all", "treat_none", "model_id", "seed", "split", "fold", "n_pos", "n_neg",
                 "nb_lo", "nb_hi"])
    table.to_parquet(out_dir / "tables" / "decision_curve.parquet", index=False)
    return {"rows": rows, **_meta(ctx), "plan": {
        "K1": "calibration metrics of the uncalibrated score_final (subgroup calibration/uncalibrated)",
        "K5": "logit p' = logit p - logit pi_val + logit pi_test per fold (GUID prevalences); PPV/NPV re-weighted to "
              f"eval.reference_prevalence = {pi_ref}", "K6": {"pts": [float(DC_PTS[0]), float(DC_PTS[-1]), DC_PTS.size],
                                                              "band": "pooled patient-cluster bootstrap 95%", **boot}}}


def i_squared(theta: Any, se: Any) -> Dict[str, float]:
    """Cochran's Q and Higgins' I² = max(0, (Q - df) / Q) of per-fold estimates ``theta`` with standard errors ``se``
    (inverse-variance weights, df = k - 1), with Q's chi-square p. Non-finite pairs are dropped; NaN with fewer than two
    folds or a zero SE (a fold whose bootstrap CI has zero width, e.g. AUROC 1 in every draw)."""
    theta, se = np.asarray(theta, np.float64), np.asarray(se, np.float64)
    ok = np.isfinite(theta) & np.isfinite(se)
    theta, se, out = theta[ok], se[ok], {"k": int(ok.sum())}
    if theta.size < 2 or (se <= 0).any():
        return out | dict.fromkeys(("q", "q_p", "i2", "pooled"), NAN)
    w = se ** -2.0
    mu = float((w * theta).sum() / w.sum())
    q, df = float((w * (theta - mu) ** 2).sum()), theta.size - 1
    return out | {"q": q, "q_p": float(chi2.sf(q, df)), "i2": max(0.0, (q - df) / q) if q > 0 else 0.0, "pooled": mu}


def run_H(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """H1 dispersion and H3; needs ``metrics``. H2 (seed spread) and H4 (val vs test) are drawn from existing rows.

    * H1: :func:`i_squared` of the per-fold GUID AUROC per model/seed/split, SE = bootstrap CI width / (2 z_0.975)
      (rows ``fold='pooled'``, ``subgroup='fold_heterogeneity'``, ``subgroup_value='auroc'``: ``i2``, ``cochran_q``,
      ``cochran_q_p``, ``fold_sd``); the per-fold values the forest draws are ``run_metrics``' rows.
    * H3: the pooled test threshold-free GUID rows under the shared-test policy that is not ``data.shared_test_policy``
      (``subgroup='shared_test_policy'``; its thresholded rows are ``run_metrics``'), and ``shared_test.parquet``:
      :data:`H3_METRICS` under ``first_fold`` and ``exclude`` side by side with n and the difference.
    """
    if not ctx.rows:
        raise RuntimeError("H reads the metric rows; run the 'metrics' analysis first")
    ev, policy = eval_config, ctx.cfg["data"]["shared_test_policy"]
    other = {"first_fold": "exclude", "exclude": "first_fold"}[policy]
    alpha, boot, z, rows, disp = primary_alpha(ev), _boot(ev), ndtri(0.975), [], {}
    for m, sd in _models(ctx):
        u = level_units(ctx, m, sd, "test")["guid"]
        rows += _tf_rows(pool_rows(u, other, "test"), alpha, boot, fold="pooled", subgroup="shared_test_policy",
                         subgroup_value=other, **_keys(ctx, m, sd, split="test", level="guid"))
    df = _rows_frame(ctx)
    base = df[(df["level"] == "guid") & (df["point"] == "n/a") & (df["metric_type"] == "threshold_free")]
    au = base[(base["metric"] == "auroc") & base["subgroup"].isna() & (base["fold"] != "pooled")]
    for (m, sd, sp), g in au.groupby(["model_id", "seed", "split"], sort=False):
        g = g.drop_duplicates("fold")  # ponytail: one binary AUROC per fold (a 3-class head's OvR rows are named apart)
        rec = i_squared(g["value"], (g["ci_hi"] - g["ci_lo"]) / (2 * z))
        rec["fold_sd"] = float(g["value"].std())
        disp[f"{m}|{sd}|{sp}"] = rec | {"per_fold": dict(zip(g["fold"], g["value"]))}
        rows += metric_rows({"i2": rec["i2"], "cochran_q": rec["q"], "cochran_q_p": rec["q_p"], "fold_sd": rec["fold_sd"]},
                            **_keys(ctx, m, sd, split=sp, level="guid", fold="pooled", metric_type="threshold_free",
                                    denominator="n/a", subgroup="fold_heterogeneity", subgroup_value="auroc"))
    h3 = pd.concat([df, pd.DataFrame(rows, columns=list(METRIC_COLUMNS))], ignore_index=True)
    h3 = h3[(h3["level"] == "guid") & (h3["split"] == "test") & (h3["fold"] == "pooled") & (h3["point"] == "n/a")
            & h3["metric"].isin(H3_METRICS) & (h3["policy_id"].fillna("") != "oracle")
            & (h3["subgroup"].isna() | (h3["subgroup"] == "shared_test_policy"))]
    h3 = h3.assign(shared=np.where(h3["subgroup"].isna(), policy, other), n=h3["n_pos"] + h3["n_neg"],
                   policy_id=h3["policy_id"].fillna("threshold_free"))
    pols = ["first_fold", "exclude"]
    table = h3.pivot_table(index=["model_id", "seed", "policy_id", "metric"], columns="shared", values=["value", "n"],
                           aggfunc="first").reindex(columns=pd.MultiIndex.from_product([["value", "n"], pols]))
    table.columns = [*pols, *(f"n_{p}" for p in pols)]
    table = table.reset_index().assign(delta=lambda t: t["exclude"] - t["first_fold"])
    table.to_parquet(out_dir / "tables" / "shared_test.parquet", index=False)
    n_shared = int(ctx.guids.loc[ctx.guids["split"] == "test"].drop_duplicates("guid")["shared_test"].astype(bool).sum())
    return {"rows": rows, **_meta(ctx), "H1": disp, "H3": {"n_shared_test_guids": n_shared, "rows": int(len(table))},
            "plan": {"H1": "Cochran's Q / Higgins' I^2 of the per-fold GUID AUROC, SE = bootstrap 95% CI width / 3.92",
                     "H3": f"pooled test under {policy} (primary) and {other} (sensitivity); shared_test.parquet"}}


def run_R10(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """R10: at every ``eval.checkpoints_h`` and end, the snapshot ROC (R4's population: the GUID's last segment in the
    staleness window) restricted to snapshots in the first stage and in the second stage (the snapshot segment's
    stage, §11.5.4), per fold and pooled, val and test -> ``roc_points`` variants ``snapshot_first@<h|end>`` and
    ``snapshot_second@<h|end>`` (``level='online'``)."""
    ev, frames = eval_config, []
    for m, sd in _models(ctx):
        for split in SPLITS:
            for fold, U in _tr_groups(ctx, m, sd, split) or []:
                pts = _points("to_delivery", np.zeros(0), ev)
                V = _view(U, "to_delivery", pts)
                if V is None:
                    continue
                for k, (kind, t) in enumerate(zip(pts["kind"], pts["t_comm"])):
                    st = U.st[V.row[:, k]]
                    for stage, code in STAGE_CODES.items():
                        sel = V.inst[:, k] & (st == code)
                        y = U.y[sel]
                        if 0 < y.sum() < y.size:
                            frames.append(pd.DataFrame({
                                "model_id": m, "seed": sd, "fold": fold, "split": split, "level": "online",
                                "variant": f"snapshot_{stage}@{'end' if kind == 'end' else f'{t:g}'}", "axis": "to_delivery",
                                "t": t, **roc_points(y, U.s[V.row[sel, k]]), "n_pos": int(y.sum()), "n_neg": int((~y).sum())}))
    prefixes = tuple(f"snapshot_{s}@" for s in STAGE_CODES)
    return {**_meta(ctx), "n_rows": _append_roc(out_dir, frames, prefixes), "plan": {
        "variants": [f"{p}<h|end>" for p in prefixes], "stage": "of the snapshot segment (first | second)"}}


def run_R12(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """R12/R13 input -> ``snapshots.parquet``: every pooled test GUID's snapshot (the calibrated online logit of its
    last segment in the window; :func:`_view`'s instantaneous population) at each ``eval.checkpoints_h`` (to_delivery,
    within the staleness window; ``point='checkpoint'``) and in each bin of every ``eval.time_axes`` axis
    (``point='bin'``): model_id, seed, split, fold (the GUID's), axis, point, t (M's display units), guid,
    clinical_class, y, score."""
    ev, frames = eval_config, []
    for m, sd in _models(ctx):
        groups = _tr_groups(ctx, m, sd, "test")
        if groups is None:
            continue
        grids, U = _grids(groups, ev["time_axes"], ev["bin_h"]), dict(groups)["pooled"]
        fold, cls = U.o["fold"].to_numpy()[U.first], U.o["clinical_class"].astype(str).to_numpy()[U.first]
        for axis in ev["time_axes"]:
            pts = _points(axis, grids[axis], ev)
            V = _view(U, axis, pts)
            if V is None:
                continue
            g, k = np.nonzero(V.inst & (pts["kind"] != "end").to_numpy()[None, :])
            frames.append(pd.DataFrame({
                "model_id": m, "seed": sd, "split": "test", "fold": fold[g].astype(str), "axis": axis,
                "point": pts["kind"].to_numpy()[k], "t": pts["t_inst"].to_numpy(np.float64)[k], "guid": U.guid[g],
                "clinical_class": cls[g], "y": U.y[g].astype(np.int64), "score": U.s[V.row[g, k]]}))
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["model_id", "seed", "split", "fold", "axis", "point", "t", "guid", "clinical_class", "y", "score"])
    table.to_parquet(out_dir / "tables" / "snapshots.parquet", index=False)
    return {**_meta(ctx), "n_rows": int(len(table)), "plan": {
        "population": "pooled test, each GUID once (data.shared_test_policy); snapshot = last segment in the window",
        "score": "calibrated online logit (the running segment aggregator without an online score)"}}


def run_K7(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """K7: Brier of the snapshot (the instantaneous population, as R7's snapshot AUROC) vs time on every axis, at the
    bins, checkpoints and end, per fold and pooled, val and test: ``brier`` of the calibrated alarm probability
    sigma(s) against y and, when the segments carry the 3-class ``p_c*_cal``, ``brier_c{k}`` of each calibrated class
    probability against 1[class = k] and ``brier_macro``, their mean. ``level='online'``, ``metric_type=
    'threshold_free'``, ``denominator='bin_present'``; NaN where a class has fewer than ``eval.min_bin_class_n`` GUIDs
    (M's rule). The pooled rows carry a patient-cluster bootstrap CI of the mean (one weighted product per axis)."""
    ev = eval_config
    boot, mn, blocks = _boot(ev), ev["min_bin_class_n"], []
    p3 = [f"p_c{k}_cal" for k in range(3)]
    a = (1.0 - 0.95) / 2.0
    for m, sd in _models(ctx):
        for split in SPLITS:
            groups = _tr_groups(ctx, m, sd, split)
            if groups is None:
                continue
            grids = _grids(groups, ev["time_axes"], ev["bin_h"])
            for fold, U in groups:
                P3 = (U.o[p3].to_numpy(np.float64) if set(p3) <= set(U.o) and U.o[p3].notna().all(axis=None) else None)
                cls = U.o["class_code"].to_numpy(np.int64)[U.first] - 1
                if fold == "pooled":
                    M, codes = _unit_draws(U.y.astype(np.int64), np.arange(U.y.size) if U.pat is None else U.pat,
                                           boot["resamples"], boot["seed"])
                for axis in ev["time_axes"]:
                    pts = _points(axis, grids[axis], ev)
                    V = _view(U, axis, pts)
                    if V is None:
                        continue
                    pop = V.inst
                    n_pos, n_neg = (pop & U.y[:, None]).sum(0), (pop & ~U.y[:, None]).sum(0)
                    live = n_pos + n_neg > 0
                    if not live.any():
                        continue
                    E = {"brier": (expit(U.s[V.row]) - U.y[:, None]) ** 2}
                    if P3 is not None:
                        E |= {f"brier_c{k}": (P3[V.row, k] - (cls == k)[:, None]) ** 2 for k in range(3)}
                        E["brier_macro"] = sum(E[f"brier_c{k}"] for k in range(3)) / 3.0
                    ok = live & (n_pos >= mn) & (n_neg >= mn)
                    popf = pop.astype(np.float64)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        vals = {k_: np.where(ok, (popf * e).sum(0) / popf.sum(0), NAN) for k_, e in E.items()}
                    lo = {k_: np.full(pop.shape[1], NAN) for k_ in E}
                    hi = {k_: np.full(pop.shape[1], NAN) for k_ in E}
                    if fold == "pooled" and ok.any():
                        X = np.concatenate([(popf * e)[:, ok] for e in E.values()] + [popf[:, ok]], axis=1).astype(np.float32)
                        D = np.vstack([M[sl][:, codes] @ X for sl in _chunks(boot["resamples"], U.y.size)])
                        K = int(ok.sum())
                        with np.errstate(invalid="ignore", divide="ignore"):
                            for i, k_ in enumerate(E):
                                r = D[:, i * K:(i + 1) * K] / D[:, -K:]
                                lo[k_][ok], hi[k_][ok] = np.nanquantile(r, [a, 1 - a], axis=0)
                    keys = _keys(ctx, m, sd, split=split, fold=fold, level="online", axis=axis, metric_type="threshold_free",
                                 denominator="bin_present")
                    cnt = dict(t=pts["t_inst"].to_numpy(np.float64)[live], point=pts["kind"].to_numpy(object)[live],
                               n_pos=n_pos[live], n_neg=n_neg[live])
                    blocks += [keys | cnt | {"metric": k_, "value": vals[k_][live], "ci_lo": lo[k_][live],
                                             "ci_hi": hi[k_][live]} for k_ in E]
    frame = _blocks_frame(blocks) if blocks else None
    return {"frame": frame, **_meta(ctx), "n_rows": 0 if frame is None else int(len(frame)), "plan": {
        "population": "instantaneous (snapshot within the bin / staleness window)", "min_bin_class_n": mn,
        "intervals": {"method": "pooled: patient-cluster outcome-stratified percentile bootstrap of the mean", **boot}}}


P2_TABLES += ("decision_curve", "snapshots")  # always non-empty: every model has test GUIDs and time-resolved rows
ANALYSES.update({"K": run_K, "H": run_H, "R10": run_R10, "R12": run_R12, "K7": run_K7})

# ---- block X (confusion and 3-class: X1-X9) ----
#
# §11.3-§11.5, §11.12 block X. The GUID- and segment-level 3-class rows are :func:`_three_class_rows` (``metrics``),
# their CIs one set of 3-class-stratified patient-cluster draws (:func:`three_class_ci`). ``X`` adds a ``three_class``
# run's time-resolved rows on the M engine's groups: the argmax snapshot rows (X3, X5a, X7; ``policy_id = argmax``)
# and, per class k, the M engine on the OvR score with the class-k OvR thresholds (X5b, X6; ``ovr_c<k>/<M metric>``).
# ``X9`` compares a multi-task binary run's head with its aux 3-class collapse. X1, X2, X4 and X8 are figures and
# summary tables over these rows (``report.py``).

from itertools import combinations  # noqa: E402


def confusion_stats(C: Any) -> Dict[str, np.ndarray]:
    """Argmax metrics of confusion matrices ``C`` (..., K, K; true x predicted counts, or bootstrap weights), as
    :func:`multiclass` defines them: ``recall_c{k}`` (NaN for an absent class), ``precision_c{k}`` and ``f1_c{k}`` (0
    when undefined), ``bal_acc``, ``macro_f1``, ``weighted_f1`` (true-support weights), ``top1_acc`` and ``qwk``
    (sklearn's quadratic-weighted kappa)."""
    C = np.asarray(C, np.float64)
    K = C.shape[-1]
    tp, true, pred = np.diagonal(C, axis1=-2, axis2=-1), C.sum(-1), C.sum(-2)
    n, w = true.sum(-1), (np.arange(K)[:, None] - np.arange(K)[None, :]) ** 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        rec, prec = tp / true, np.where(pred > 0, tp / pred, 0.0)
        f1 = np.where(true + pred > 0, 2.0 * tp / (true + pred), 0.0)
        expected = true[..., :, None] * pred[..., None, :] / n[..., None, None]
        out = {"top1_acc": tp.sum(-1) / n, "bal_acc": rec.mean(-1), "macro_f1": f1.mean(-1),
               "weighted_f1": (true * f1).sum(-1) / n,
               "qwk": 1.0 - (w * C).sum((-2, -1)) / (w * expected).sum((-2, -1))}
    return out | {f"{name}_c{k}": v[..., k] for name, v in (("recall", rec), ("precision", prec), ("f1", f1))
                  for k in range(K)}


def row_normalised(C: Any) -> np.ndarray:
    """Each confusion row (true class) over its total; NaN rows for an absent class."""
    C = np.asarray(C, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return C / C.sum(-1, keepdims=True)


def fold_mean_rownorm(mats: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """X2's second pooled view: the mean and SD (ddof 1) over folds of the row-normalised per-fold confusions, every
    fold weighing the same whatever its size (the pooled matrix is the summed counts); a fold without a class skips
    that row. NaN SD with one fold."""
    R = row_normalised(np.stack(mats))
    n = np.isfinite(R).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(R, 0) / n
        return mean, np.sqrt(np.nansum((R - mean) ** 2, 0) / (n - 1))


def _mc_stats(y: np.ndarray, P: np.ndarray, W: np.ndarray, *, alpha: float,
              ords: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Every metric :func:`three_class_ci` bounds, per row of unit weights ``W`` (draws, n) over rows ``y`` (0..K-1),
    ``P`` (n, K): OvR AUROC/AUPRC and the macro AUROC, Hand-Till (each class pair's two AUROCs on that pair's rows),
    RPS, :func:`confusion_stats` of the weighted argmax confusion, the collapses' AUROC/AUPRC/pAUC@alpha and, with
    CORAL scores ``ords``, the ordinal alarm-score AUROC at both cut-points. With unit weights these are exactly
    :func:`multiclass` and :func:`threshold_free` (the weighted rank statistics of :func:`_rank_stats`)."""
    K = P.shape[1]

    def rank(pos: np.ndarray, s: np.ndarray, sel: Any = slice(None)) -> Dict[str, np.ndarray]:
        o, st, _ = _ranked(s[sel])
        return _rank_stats(*_curve(pos[sel].astype(np.int64), W[:, sel], o, st), alpha)

    out: Dict[str, np.ndarray] = {}
    for k in range(K):
        r = rank(y == k, P[:, k])
        out[f"auroc_ovr_c{k}"], out[f"auprc_ovr_c{k}"] = r["auroc"], r["auprc"]
    out["auroc_macro"] = np.mean([out[f"auroc_ovr_c{k}"] for k in range(K)], axis=0)
    out["auroc_hand_till"] = np.mean([(rank(y == i, P[:, i], sel)["auroc"] + rank(y == j, P[:, j], sel)["auroc"]) / 2.0
                                      for i, j in combinations(range(K), 2) for sel in [(y == i) | (y == j)]], axis=0)
    obs = (y[:, None] <= np.arange(K - 1)).astype(np.float64)
    out["rps"] = (W @ (((np.cumsum(P, 1)[:, :-1] - obs) ** 2).sum(1) / (K - 1))) / W.sum(1)
    out |= confusion_stats((W @ np.eye(K * K)[y * K + P.argmax(1)]).reshape(-1, K, K))
    for name, pos, q in (("adverse_vs_healthy", y > 0, P[:, 1:].sum(1)), ("hie_vs_rest", y == K - 1, P[:, K - 1])):
        out |= {f"{name}/{m}": v for m, v in rank(pos, _logit(np.clip(q, 1e-12, 1 - 1e-12))).items()}
    if ords is not None:
        out |= {f"ordinal_auroc_{name}": rank(pos, ords)["auroc"] for name, pos in (("adverse", y > 0), ("hie", y == K - 1))}
    return out


def three_class_ci(y: Any, P: Any, units: Any, *, alpha: float, resamples: int, seed: int,
                   ords: Optional[np.ndarray] = None, confidence: float = 0.95) -> Dict[str, Any]:
    """:func:`bootstrap_ci` of every :func:`_mc_stats` metric, vectorised: outcome-stratified cluster draws over
    ``units`` (:func:`_unit_draws`, strata = the 3-class label sets), all metrics on the same draws."""
    y, P = np.asarray(y, np.int64).reshape(-1), np.asarray(P, np.float64)
    M, codes = _unit_draws(y, units, resamples, seed)
    point = {m: float(v[0]) for m, v in _mc_stats(y, P, np.ones((1, y.size)), alpha=alpha, ords=ords).items()}
    draws = {m: np.empty(resamples) for m in point}
    for sl in _chunks(resamples, y.size):
        for m, v in _mc_stats(y, P, M[sl][:, codes], alpha=alpha, ords=ords).items():
            draws[m][sl] = v
    return _percentile_ci(point, draws, method="cluster outcome-stratified percentile bootstrap (3-class label sets; "
                          "vectorised weighted statistics)", resamples=resamples, seed=seed, confidence=confidence,
                          n=M.shape[1])


#: The argmax snapshot metrics of X5a/X7 (:func:`_argmax_block`); ``recall_c{k}`` needs class k, the rest every class.
ARGMAX_TR = (*(f"recall_c{k}" for k in range(3)), *(f"f1_c{k}" for k in range(3)), "top1_acc", "macro_f1", "weighted_f1")
CONFUSION_NAMES = tuple(f"confusion_t{i}_p{j}" for i in range(3) for j in range(3))


def _argmax_block(U: SimpleNamespace, V: SimpleNamespace, pts: pd.DataFrame, P: np.ndarray, *, mn: int,
                  boot: Mapping[str, Any], **keys: Any) -> list:
    """X3/X5a/X7 column blocks of one group on one axis: at every point, the argmax of each present GUID's snapshot
    (its last segment in the instantaneous window, so one row per GUID: :func:`_view`) against its class
    (``P``: the class probabilities of ``U``'s rows). ``confusion_t{i}_p{j}`` counts, :data:`ARGMAX_TR` from them
    (:func:`confusion_stats`); a metric is NaN when a class it needs has fewer than ``mn`` GUIDs in the bin. Recall and
    top-1 accuracy carry :func:`rate_ci` CIs (Wilson; the patient-cluster bootstrap where a patient holds two GUIDs)."""
    y = U.o["class_code"].to_numpy(np.int64)[U.first] - 1
    pred, inst, npt = P.argmax(1)[V.row], V.inst, len(pts)
    C = np.bincount((np.arange(npt) * 9 + y[:, None] * 3 + pred)[inst], minlength=npt * 9).reshape(npt, 3, 3)
    st, n_c = confusion_stats(C), C.sum(2)
    under, live, nan = n_c < mn, n_c.sum(1) > 0, np.full(npt, NAN)
    hit = inst & (pred == y[:, None])
    vals = {f"confusion_t{i}_p{j}": (C[:, i, j], nan, nan, live) for i in range(3) for j in range(3)}
    for name in ARGMAX_TR:
        u = under[:, int(name[-1])] if name.startswith("recall") else under.any(1)
        den = inst & (y == int(name[-1]))[:, None] if name.startswith("recall") else inst
        lo, hi, nu = (rate_ci((hit & den).T, den.T, y, U.pat, **boot) if name.startswith(("recall", "top1"))
                      else (nan, nan, nan))
        vals[name] = (np.where(u, NAN, st[name]), np.where(u, NAN, lo), np.where(u, NAN, hi), live, np.where(u, NAN, nu))
    return _long(vals, pids=["argmax"], basis=["argmax"], thr=np.full(1, NAN), t=pts["t_inst"].to_numpy(),
                 point=pts["kind"].to_numpy(), n_pos=(inst & (y > 0)[:, None]).sum(0),
                 n_neg=(inst & (y == 0)[:, None]).sum(0), **keys)


def _sub_ctx(ctx: SimpleNamespace, **repl: Any) -> SimpleNamespace:
    """A copy of ``ctx`` with ``repl`` swapped in (e.g. transformed scores and their thresholds), fresh segment and
    group caches, and its own ``inclusion`` list (the parent already recorded these populations)."""
    return SimpleNamespace(**{k: v for k, v in vars(ctx).items() if k not in ("seg_cache", "tr_cache")}
                           | {"inclusion": []} | repl)


def _ovr_frames(ctx: SimpleNamespace, m: str, sd: str, split: str, ev: Mapping[str, Any], *, alpha: float,
                boot: Mapping[str, Any]) -> list:
    """X5b/X6 column blocks: per class k, :func:`_mt_frames` on every axis, per fold and pooled, with the OvR score
    ``logit p_c<k>_cal`` and target ``class_code - 1 == k`` (``thresholds.ovr_thresholds``' transform) and the class-k
    OvR threshold of the primary policy; metric names ``ovr_c<k>/<name>``. Empty without OvR thresholds."""
    ovr = {key: per for key, per in ctx.thr_ovr.items() if key.startswith(f"{m}|{sd}|")}
    if not ovr:
        return []
    # ponytail: the primary policy only (the figure's); every policy would triple M's rows per class
    primary = [p for p in ev["thresholds"] if p["id"] == ev["primary_policy"]]
    cfg = ctx.cfg | {"eval": ctx.cfg["eval"] | {"thresholds": primary}}
    seg, blocks = _sel(ctx.seg, model_id=m, seed=sd), []
    for k in range(3):
        sub = _sub_ctx(ctx, cfg=cfg, thr={key: per[str(k)] for key, per in ovr.items()},
                       seg=ovr_view(seg, k, OVR_SEGMENT_SCORES))
        groups = _tr_groups(sub, m, sd, split)
        if groups is None:
            continue
        grids = _grids(groups, ev["time_axes"], ev["bin_h"])
        for fold, U in groups:
            for axis in ev["time_axes"]:
                pts = _points(axis, grids[axis], ev)
                V = _view(U, axis, pts)
                if V is None:
                    continue
                f, _ = _mt_frames(U, V, pts, ev, alpha=alpha, boot=boot, axis=axis,
                                  rank_ci=(pts["kind"] != "bin").to_numpy() | (fold == "pooled"),
                                  **_keys(ctx, m, sd, split=split, fold=fold, level="online"))
                blocks += [b | {"metric": f"ovr_c{k}/{b['metric']}" if isinstance(b["metric"], str)
                                else np.array([f"ovr_c{k}/{name}" for name in b["metric"]], object)}
                           for b in f if b.get("subgroup") is None]  # stage strata dropped
    return blocks


def run_X(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """X3, X5-X7: a ``three_class`` run's time-resolved rows on every ``eval.time_axes`` axis, per fold and pooled,
    val and test, ``level='online'`` (the M engine's groups, :func:`_tr_groups`): the argmax snapshot rows
    (:func:`_argmax_block`) and the primary policy's per-class OvR rows (:func:`_ovr_frames`). Not applicable to a
    binary task; a model without calibrated class probabilities on its segments is skipped."""
    ev, task = eval_config, ctx.cfg["labels"]["task"]
    if task != "three_class":
        return {**_meta(ctx), "plan": {"applicable": False, "reason": f"labels.task {task} is not three_class"}}
    alpha, boot, cols, frames, skipped = primary_alpha(ev), _boot(ev), [f"p_c{k}_cal" for k in range(3)], [], []
    for m, sd in _models(ctx):
        for split in SPLITS:
            groups = _tr_groups(ctx, m, sd, split)
            if groups is None or not set(cols) <= set(ctx.seg.columns) or any(
                    U.o[cols].isna().any(axis=None) for _, U in groups):
                skipped.append(f"{m}|{sd}|{split}")
                continue
            grids = _grids(groups, ev["time_axes"], ev["bin_h"])
            for fold, U in groups:
                P = U.o[cols].to_numpy(np.float64)
                keys = _keys(ctx, m, sd, split=split, fold=fold, level="online", metric_type="instantaneous",
                             denominator="bin_present")
                for axis in ev["time_axes"]:
                    pts = _points(axis, grids[axis], ev)
                    V = _view(U, axis, pts)
                    if V is not None:
                        frames += _argmax_block(U, V, pts, P, mn=ev["min_bin_class_n"], boot=boot, axis=axis, **keys)
            frames += _ovr_frames(ctx, m, sd, split, ev, alpha=alpha, boot=boot)
    frame = _blocks_frame(frames) if frames else None
    return {"frame": frame, **_meta(ctx), "n_rows": 0 if frame is None else int(len(frame)), "skipped": skipped,
            "plan": {"argmax": "policy_id argmax, instantaneous (one snapshot per GUID per bin): confusion_t{i}_p{j}, "
                               + ", ".join(ARGMAX_TR), "min_bin_class_n": ev["min_bin_class_n"],
                     "ovr": f"ovr_c<k>/<M metric>: the M engine on logit p_c<k>_cal with the class-k OvR threshold of "
                            f"{ev['primary_policy']} (thresholds.json ovr); stage strata dropped",
                     "axes": list(ev["time_axes"])}}


def aux_collapse(f: pd.DataFrame, task: str) -> np.ndarray:
    """X9's aux score: ``logit`` of the aux head's (raw) probability of ``task``'s positive classes (``P(acidosis) +
    P(HIE)`` for adverse_vs_healthy, §6.3); NaN where the row has no aux output."""
    from teb_vae.classifier.config import TASKS

    q = sum(f[f"p_c{c - 1}"].to_numpy(np.float64) for c, t in TASKS[task].items() if t == 1)
    return _logit(np.clip(q, 1e-12, 1 - 1e-12))


def run_X9(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """X9 collapse consistency of a multi-task binary run (``labels.aux_3class_weight > 0``), per model with aux
    probabilities, fold and pooled, val and test, GUID level, metric names ``collapse/<name>``:

    * threshold-free: ``auroc_binary`` (the head's ``score_final_cal``) and ``auroc_aux`` (:func:`aux_collapse`) with
      patient-cluster bootstrap CIs, and ``spearman``, the rank correlation of the two GUID scores;
    * at the primary policy: the aux collapse gets its own threshold, chosen on val by that policy on its basis
      (:func:`~teb_vae.classifier.thresholds.select_thresholds` on the aux-scored val rows), and on each policy's
      basis population ``disagreement`` (Wilson CI) is the fraction of GUIDs whose two decisions differ,
      ``binary_only`` / ``aux_only`` the GUIDs only that score alarms.
    """
    from scipy.stats import spearmanr

    from teb_vae.classifier.thresholds import select_thresholds

    ev, lab = eval_config, ctx.cfg["labels"]
    cols = [f"p_c{k}" for k in range(3)]
    if lab["task"] in ("three_class", "cs_outcome") or not lab["aux_3class_weight"] > 0:
        return {**_meta(ctx), "plan": {"applicable": False, "reason": "not a multi-task binary run"}}
    pid, policy, boot, rows = ev["primary_policy"], ctx.cfg["data"]["shared_test_policy"], _boot(ev), []
    pol = [p for p in ev["thresholds"] if p["id"] == pid]
    for m, sd in _models(ctx):
        g, s = _sel(ctx.guids, model_id=m, seed=sd), _sel(ctx.seg, model_id=m, seed=sd)
        if not set(cols) <= set(g.columns) or g[cols].isna().any(axis=None):
            continue
        g = g.assign(aux=aux_collapse(g, lab["task"]))
        aux = np.where(s["logit_online_cal"].isna(), NAN, aux_collapse(s, lab["task"]))  # causal rows only (ovr_view)
        sub = _sub_ctx(ctx, guids=g.assign(score_final_cal=g["aux"]), seg=s.assign(
            **{c: np.where(s[c].isna(), NAN, aux) for c in OVR_SEGMENT_SCORES}))
        seg_aux = sub.seg[list(OVR_SEGMENT_SCORES)].notna().any(axis=None)
        # the model's own basis (a model without per-position scores selected on guid_final, guid_level_policies); a
        # time basis without a causal per-segment aux score (non-causal model) has no aux threshold: no comparison
        sub.thr = {key: select_thresholds(_sel(sub.seg, fold=int(key.split("|")[2]), split="val"),
                                          _sel(sub.guids, fold=int(key.split("|")[2]), split="val"),
                                          [pol[0] | {x: e[x] for x in ("basis", "axis", "at")}],
                                          bin_h=ev["bin_h"], exclude_last_min=ev["exclude_last_min"],
                                          staleness_h=ev["snapshot_max_staleness_h"])
                   for key, per in ctx.thr.items() if key.startswith(f"{m}|{sd}|")
                   for e in [per.get("guid", {}).get(pid, {"skipped": True})]
                   if "skipped" not in e and (seg_aux or e["basis"] == "guid_final")}
        for split in SPLITS:
            u = _sel(g, split=split)
            u = u.assign(unit=u["guid"], patient=cluster(u))
            for fold, f in [*((str(k), x) for k, x in u.groupby("fold")), ("pooled", pool_rows(u, policy, split))]:
                y, vals, ci = f["y"].to_numpy(np.int64), {}, {}
                for name, col in (("auroc_binary", "score_final_cal"), ("auroc_aux", "aux")):
                    r = (bootstrap_auroc(y, f[col], f["patient"], boot["resamples"], boot["seed"])["metrics"]["auroc"]
                         if 0 < y.sum() < y.size else {"value": NAN})
                    vals[f"collapse/{name}"], ci[f"collapse/{name}"] = r["value"], r
                vals["collapse/spearman"] = float(spearmanr(f["score_final_cal"], f["aux"]).statistic) if len(f) > 2 else NAN
                rows += metric_rows(vals, ci, n_pos=int(y.sum()), n_neg=int(y.size - y.sum()), **_keys(
                    ctx, m, sd, split=split, fold=fold, level="guid", metric_type="threshold_free", denominator="n/a"))
            (eb, pb), (ea, pa) = (policy_populations(x, m, sd, split).get(("guid", pid), ({}, None)) for x in (ctx, sub))
            if pb is None or pa is None:
                continue
            j = pb.merge(pa[["fold", "unit", "score"]], on=["fold", "unit"], suffixes=("", "_aux"))
            j = j.assign(**{d: j[col].to_numpy(np.float64) > j["fold"].map({f: e["threshold"] for f, e in ents.items()})
                            .to_numpy(np.float64) for d, col, ents in (("db", "score", eb), ("da", "score_aux", ea))})
            e0 = next(iter(eb.values()))
            k = _keys(ctx, m, sd, split=split, level="guid", policy_id=pid, policy_basis=e0["basis"],
                      metric_type=e0["basis"] if e0["basis"] in _DENOM else None, denominator=_DENOM.get(e0["basis"], "n/a"),
                      axis=None if e0["at"] == "end" else e0["axis"], t=NAN if e0["at"] == "end" else float(e0["at"]))
            for fold, f in [*((str(x), q) for x, q in j.groupby("fold")), ("pooled", pool_rows(j, policy, split))]:
                d, y = (f["db"] != f["da"]).to_numpy(), f["y"].to_numpy() == 1
                lo, hi, nu = rate_ci(d[None], np.ones((1, d.size), bool), y, f["patient"], **boot)
                rows += metric_rows({"collapse/disagreement": d.mean() if d.size else NAN,
                                     "collapse/binary_only": int((f["db"] & ~f["da"]).sum()),
                                     "collapse/aux_only": int((f["da"] & ~f["db"]).sum())},
                                    {"collapse/disagreement": {"ci_lo": float(lo[0]), "ci_hi": float(hi[0]),
                                                               "n_undefined": float(nu[0])}},
                                    fold=fold, threshold=eb[int(fold)]["threshold"] if fold != "pooled" else NAN,
                                    n_pos=int(y.sum()), n_neg=int((~y).sum()), **k)
    return {"rows": rows, **_meta(ctx), "plan": {
        "aux_score": "logit of the aux head's raw probability of the task's positive classes (p_c1 + p_c2 for "
                     "adverse_vs_healthy)", "binary_score": "score_final_cal",
        "policy": f"{pid}; the aux threshold re-selected on val by the same policy on the aux score, per fold",
        "intervals": {"auroc": "patient-cluster bootstrap", "disagreement": "wilson 95% (rate_ci)", **boot}}}


ANALYSES.update({"X": run_X, "X9": run_X9})

# ---- block E (errors and interpretation: E1-E5) ----
#
# §11.12 block E, read off the time-resolved groups (:func:`_tr_groups`: after ``eval.exclude_last_min``, each GUID
# with its fold's thresholds), so a GUID's score, alarm and first-alarm time are the ones block A counts. The GUID
# alarm score is the running max of the calibrated online score at its last segment (the running segment aggregator
# without an online score) and the decision the primary policy's latched alarm. Tables: ``errors.parquet`` (E1: every
# model with a per-position score but the shuffled control, per fold x split) and ``error_segments.parquet`` (the
# E3-E5 figures' input: the primary model's pooled test segments, each GUID once). ``report`` draws the E2 pages and
# the E3-E5 figures. E6 (:func:`run_E6`, ``eval.attribution.enabled``, default off) summarises the IG attributions the
# predict stage wrote (``predictions/attribution.parquet``: torch lives there, evaluate stays torch-free).

#: ``errors.parquet`` columns (E1).
ERROR_COLUMNS = ["model_id", "seed", "fold", "split", "kind", "rank", "guid", "patient", "clinical_class", "subgroup",
                 "stage", "y", "max_score", "policy_id", "threshold", "alarmed", "first_alarm_h", "n_segments", "span_h",
                 "valid_frac", "has_tlo"]
#: The pooling-attention scalars of the neural prediction rows (``train.ATTN_COLUMNS``; absent in older runs).
ATTN_SCALARS = ("attn_late_mass", "attn_centroid", "seq_attn_final")


def extreme_rows(frame: pd.DataFrame, column: str, *, per_tail: int) -> Dict[str, pd.DataFrame]:
    """``{'low', 'high'}``: the rows with the smallest and largest finite ``column``, ascending. Ported from
    ``teb_vae/lag_attn_cfs/eval/analyses/samples.py:753 extreme_rows`` (analysis modules are not imported): the tails
    are disjoint, so ``per_tail`` is lowered to half the finite rows, and both are empty below two finite rows."""
    if frame.empty or column not in frame.columns:
        return {"low": frame.head(0), "high": frame.head(0)}
    finite = frame[frame[column].notna()].sort_values(column, kind="stable")
    per_tail = min(int(per_tail), len(finite) // 2)
    if per_tail < 1:
        return {"low": finite.head(0), "high": finite.head(0)}
    return {"low": finite.head(per_tail), "high": finite.tail(per_tail)}


def top_errors(guids: pd.DataFrame, k: int) -> pd.DataFrame:
    """E1 of one (model, seed, fold, split) GUID frame (``y``, ``max_score``): the top-``k`` false positives, the healthy
    GUIDs (``y == 0``) with the highest alarm score, and false negatives, the adverse ones with the lowest, each the
    :func:`extreme_rows` tail of its own class (so ``k`` is capped at half the class); ``kind`` fp | fn and ``rank``
    (1 = the worst). The alarm decision is in ``alarmed``: a top-ranked GUID the policy got right is still listed."""
    fp = extreme_rows(guids[guids["y"] == 0], "max_score", per_tail=k)["high"].iloc[::-1]
    fn = extreme_rows(guids[guids["y"] == 1], "max_score", per_tail=k)["low"]
    return pd.concat([f.assign(kind=kind, rank=np.arange(1, len(f) + 1)) for kind, f in (("fp", fp), ("fn", fn))],
                     ignore_index=True)


def _guid_frame(U: SimpleNamespace, p: int) -> pd.DataFrame:
    """One row per GUID of a prepared group (:func:`_prep`) under policy index ``p``: the alarm score ``max_score``
    (r at the last segment), its fold's ``threshold``, the latched ``alarmed`` and ``first_alarm_h`` (hours before
    delivery; NaN = never), ``n_segments``, ``span_h`` (first epoch to last segment end), the mean ``valid_frac``, and
    the GUID's fields; ``stage`` is its last segment's."""
    o, n_g = U.o, np.diff(np.r_[U.first, U.gi.size])
    last = U.first + n_g - 1
    A = _alarm_arrays(U, U.latch)
    return pd.DataFrame({
        "fold": o["fold"].to_numpy()[U.first], "guid": U.guid.astype(str),
        "patient": cluster(o).to_numpy()[U.first].astype(str), "y": U.y.astype(np.int64),
        **{c: o[c].astype(str).to_numpy()[i] for c, i in (("clinical_class", U.first), ("subgroup", U.first),
                                                          ("stage", last))},
        "max_score": U.r[last], "threshold": U.T[:, p], "alarmed": A["alarmed"][p],
        "first_alarm_h": -A["first_alarm_t_s"][p] / 3600.0, "n_segments": n_g,
        "span_h": (o["t_end_s"].to_numpy(np.float64)[last] - o["epoch_s"].to_numpy(np.float64)[U.first]) / 3600.0,
        "valid_frac": np.add.reduceat(o["valid_frac"].to_numpy(np.float64), U.first) / n_g,
        "has_tlo": o["has_tlo"].to_numpy(bool)[U.first]})


def _error_segments(U: SimpleNamespace, p: int) -> pd.DataFrame:
    """The E3-E5 rows of a (pooled) group: every segment with its online ``score``, running max ``alarm_score``, the
    GUID's latched ``alarmed`` under policy index ``p`` and the :data:`ATTN_SCALARS` (NaN when absent)."""
    o = U.o
    return pd.DataFrame({
        "fold": o["fold"].to_numpy(), "guid": o["guid"].astype(str).to_numpy(), "seg_pos": o["seg_pos"].to_numpy(),
        "y": o["y"].to_numpy(np.int64), "clinical_class": o["clinical_class"].astype(str).to_numpy(),
        **{c: o[c].to_numpy(np.float64) for c in ("epoch_s", "t_end_s", "valid_frac")},
        "score": U.s, "alarm_score": U.r, "alarmed": U.latch[p][np.r_[U.first[1:], U.gi.size] - 1][U.gi],
        **{c: o[c].to_numpy(np.float64) if c in o else np.full(len(o), NAN) for c in ATTN_SCALARS}})


def run_E(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """E1 -> ``errors.parquet`` (:data:`ERROR_COLUMNS`): per model (not the shuffled control), split and fold, the
    :func:`top_errors` of ``eval.error_analysis.top_k`` under the primary policy; and the primary model's pooled test
    segments -> ``error_segments.parquet`` (:func:`_error_segments`), the E3-E5 figure input. A model without a
    per-position score (the shortcut) or without the primary policy is skipped, and listed."""
    ev = eval_config
    pid, k, prim = ev["primary_policy"], int(ev["error_analysis"]["top_k"]), primary_model(_s_models(ctx))
    frames, segs, skipped = [], [], []
    for m, sd in _s_models(ctx):
        for split in SPLITS:
            groups = _tr_groups(ctx, m, sd, split)
            if groups is None or pid not in groups[0][1].pids:
                skipped.append(f"{m}|{sd}|{split}")
                continue
            for fold, U in groups:
                p = U.pids.index(pid)
                if fold != "pooled":
                    frames.append(top_errors(_guid_frame(U, p), k).assign(model_id=m, seed=sd, split=split, policy_id=pid))
                elif (m, sd) == prim and split == "test":
                    segs.append(_error_segments(U, p).assign(model_id=m, seed=sd, split=split))
    errors = (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()).reindex(columns=ERROR_COLUMNS)
    errors.to_parquet(out_dir / "tables" / "errors.parquet", index=False)
    seg = pd.concat(segs, ignore_index=True) if segs else pd.DataFrame(columns=["model_id", "seed", "split", "guid"])
    seg.to_parquet(out_dir / "tables" / "error_segments.parquet", index=False)
    return {**_meta(ctx), "n_errors": int(len(errors)), "n_error_segments": int(len(seg)), "skipped": skipped,
            "plan": {"policy": pid, "top_k": k, "score": "running max of the calibrated online score at the last segment "
                     "(after eval.exclude_last_min)", "tails": "each class's own tail, at most half the class (extreme_rows)",
                     "error_segments": f"{'|'.join(prim or ('none',))}: pooled test, each GUID once "
                                       "(data.shared_test_policy)",
                     "E6": "run_E6 (eval.attribution.enabled)"}}


#: ``attribution.parquet`` (E6) columns: per fold (and ``pooled``) x channel group x class, the mean over GUIDs of
#: ``share`` (the group's fraction of the GUID's summed |IG|) and ``signed`` (its summed IG), with patient-cluster
#: bootstrap CIs.
ATTRIBUTION_TABLE = ["model_id", "seed", "fold", "split", "channel_group", "clinical_class", "metric", "value", "ci_lo",
                     "ci_hi", "n"]


def run_E6(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """E6 -> ``attribution.parquet`` (:data:`ATTRIBUTION_TABLE`): the primary model's test IG attributions (§11.12;
    ``run.attribution_fold``) per channel group and class, per fold and pooled (shared test GUIDs as
    ``data.shared_test_policy``). A seed ensemble's are its members' means per GUID (IG is linear in the score and
    ``ens`` averages the raw member scores). The IG completeness error (|Σ IG - (f(x) - f(0))| over |f(x) - f(0)|) is
    reported, not gated. Off (nothing written) unless ``eval.attribution.enabled``."""
    if not eval_config["attribution"]["enabled"]:
        return {**_meta(ctx), "status": "off", "plan": {"why": "eval.attribution.enabled is false"}}
    path = ctx.run_dir / "predictions" / "attribution.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing although eval.attribution.enabled: re-run --stage predict")
    a, prim = pd.read_parquet(path), primary_model(_s_models(ctx))
    if prim is None or prim[0] != "model" or a.empty:
        return {**_meta(ctx), "status": "no model rows", "primary": prim}
    a["seed"] = a["seed"].astype(str)
    if prim[1] != "ens":
        a = a[a["seed"] == prim[1]]
    a = a.groupby(["fold", "guid", "channel_group"], as_index=False)[["abs", "signed", "f_x", "f_0"]].mean()
    a["share"] = a["abs"] / a.groupby(["fold", "guid"])["abs"].transform("sum").replace(0.0, np.nan)
    g = _sel(ctx.guids, model_id=prim[0], seed=prim[1], split="test")
    g = g[["fold", "clinical_class", "shared_test"]].assign(guid=g["guid"].astype(str), patient=cluster(g).to_numpy())
    a = a.merge(g, on=["fold", "guid"], validate="many_to_one")
    per_guid = a.drop_duplicates(["fold", "guid"])
    gap = (per_guid["f_x"] - per_guid["f_0"]).to_numpy()
    total = a.groupby(["fold", "guid"], sort=False)["signed"].sum().to_numpy()
    err = np.abs(total - gap) / np.maximum(np.abs(gap), 1e-6)
    pooled = pool_rows(a.assign(unit=a["guid"] + "|" + a["channel_group"]), ctx.cfg["data"]["shared_test_policy"], "test")
    boot, recs = _boot(eval_config), []
    for fold, f in [*((str(k), x) for k, x in a.groupby("fold")), ("pooled", pooled)]:
        for (grp, cls), x in f.groupby(["channel_group", "clinical_class"]):
            for met in ("share", "signed"):
                v = x[met].to_numpy(np.float64)
                ok = np.isfinite(v)
                recs.append(dict(model_id=prim[0], seed=prim[1], fold=fold, split="test", channel_group=grp,
                                 clinical_class=cls, metric=met, n=int(ok.sum()),
                                 **dict(zip(("value", "ci_lo", "ci_hi"),
                                            _stat_ci(v[ok], np.mean, boot, x["patient"].to_numpy()[ok])))))
    pd.DataFrame(recs).reindex(columns=ATTRIBUTION_TABLE).to_parquet(out_dir / "tables" / "attribution.parquet",
                                                                     index=False)
    return {**_meta(ctx), "status": "done", "primary": list(prim), "n_rows": len(recs),
            "n_guids": int(len(per_guid)), "channel_groups": sorted(a["channel_group"].unique()),
            "completeness": {"median_rel_error": float(np.median(err)) if err.size else None,
                             "max_rel_error": float(np.max(err)) if err.size else None},
            "plan": {"method": "Integrated Gradients (hand-written; captum is not a dependency), baseline 0 = the "
                     "train-fold feature mean, midpoint rule", "n_steps": eval_config["attribution"]["n_steps"],
                     "target": "score_final (raw, prior-corrected; 3-class: the collapsed alarm logit)",
                     "intervals": "patient-cluster bootstrap of the GUID mean", **boot}}


P2_TABLES += ("errors",)  # non-empty whenever a model has an online or segment score: every neural and probe run
ANALYSES.update({"E": run_E, "E6": run_E6})


# ---- Δ vs frozen (§10.1: every non-frozen regime reads against its frozen baseline) ----
def run_VF(ctx: SimpleNamespace, *, eval_config: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Each ``model`` seed (and its ensemble) with a ``frozen`` unit at that seed (the online regimes), minus that
    baseline, on the pooled OOF test GUIDs under one paired patient-cluster bootstrap (:func:`paired_deltas`, block Q's
    machinery within one run): ΔAUROC of ``score_final_cal`` (with its DeLong p and the Nadeau-Bengio p over folds) and
    Δsens/Δspec at the primary policy, each model on its own basis population at its own fold thresholds. The ablation
    and diagnostic models (``noind``, ``*_covoff``) are not paired. Both sides keep only the (fold, guid) rows they share
    (a unit that failed under ``--allow-partial`` drops its fold from both), recorded as L14 ``inclusion`` rows.
    ``summary.md`` §4 prints ``table``."""
    models, pid = _models(ctx), eval_config["primary_policy"]
    kw = dict(mn=eval_config["min_subgroup_n"], **_boot(eval_config))
    table = []
    for m, sd in models:
        if m != "model" or ("frozen", sd) not in models:
            continue
        a, b = _q_side(ctx, "frozen", ("frozen", sd)), _q_side(ctx, m, (m, sd))
        shared = set(zip(a.guid["fold"], a.guid["guid"])) & set(zip(b.guid["fold"], b.guid["guid"]))
        both = lambda x: x[[k in shared for k in zip(x["fold"], x["guid"])]].reset_index(drop=True)  # noqa: E731
        n_out = len(a.guid) + len(b.guid) - 2 * len(shared)
        _incl(ctx, "VF", m, sd, "test", "pooled", None, len(shared), n_out, {"not_in_both_model_and_frozen": n_out})
        nb = nadeau_bengio_p([b.fold_auroc[f] - a.fold_auroc[f] for f in sorted(set(a.fold_auroc) & set(b.fold_auroc))])
        recs = [r | {"p_nb": nb} for r in paired_deltas(both(a.guid), both(b.guid), ["auroc"], **kw)]
        if pid in a.pols and pid in b.pols:
            recs += [r | {"policy_id": pid}
                     for r in paired_deltas(both(a.pols[pid][1]), both(b.pols[pid][1]), ["sens", "spec"], **kw)]
        table += [{"model_id": m, "seed": sd, "n_unpaired": n_out, **r} for r in recs]
    return {**_meta(ctx), "plan": {"reference": "frozen (same seed)", "policy": pid, "population": "pooled OOF test"},
            "table": table}


ANALYSES.update({"VF": run_VF})

# ---- block Q (model comparison: Q1-Q3) ----
#
# §11.8 and §11.12 block Q, run by ``run.py compare`` over 2-:data:`MAX_COMPARED` finished runs. It spans runs, so it is
# not an evaluate analysis and nothing here enters ``ANALYSES``. Runs pair only when their cohorts do
# (:func:`cohort_digest`). Each run contributes its primary model (:func:`primary_model`); the first run is the
# reference, and every delta is run minus reference. Q2 is the paired outcome-stratified patient-cluster bootstrap
# (:func:`paired_draws`) on the pooled OOF test rows (:func:`pool_rows` under each run's ``data.shared_test_policy``):
# both sides of a delta are read off the same resamples. The Q1/Q2 figures and ``comparison.md`` (Q3) are
# ``report.compare_report``'s.

import hashlib  # noqa: E402

#: Q1 draws one line-palette colour per run; compare refuses more runs.
MAX_COMPARED = 6
#: The ``cohort/guids.parquet`` fields that make two runs' GUID-level rows pair (:func:`cohort_digest`).
COHORT_KEY = ("fold", "split", "guid", "patient", "y")
#: ``comparison.parquet`` columns beyond §11.9: the compared and reference run (directory names) and the reference's
#: model, the member's population (as S4), the power guard (value NaN, the estimate in ``value_raw``), each side's point
#: value and the bootstrap p-values.
Q_EXTRA = ("run", "reference", "reference_model", "population", "underpowered", "value_raw", "run_value",
           "reference_value", "p_value", "p_holm", "p_delong", "p_nb")

from scipy.stats import norm, rankdata, t as student_t  # noqa: E402


def delong(y: Any, *scores: Any) -> Tuple[np.ndarray, np.ndarray]:
    """AUROCs of ``scores`` (each over the same rows, labels ``y``) and their DeLong covariance matrix, by the
    midrank structural components of Sun & Xu (2014): with m positives X and n negatives Y, the placement values
    V10(X_i) = (T_Z(X_i) - T_X(X_i)) / n and V01(Y_j) = 1 - (T_Z(Y_j) - T_Y(Y_j)) / m (T: midranks within X, Y and
    Z = X ∪ Y; ties count 1/2), AUC = mean V10, Cov = S10 / m + S01 / n with S the sample covariances (ddof 1).
    Needs at least two rows of each class."""
    y, S = np.asarray(y).reshape(-1) == 1, np.atleast_2d(np.asarray(scores, dtype=np.float64))
    X, Y = S[:, y], S[:, ~y]
    m, n = X.shape[1], Y.shape[1]
    tx, ty, tz = (rankdata(A, axis=1) for A in (X, Y, np.hstack([X, Y])))
    v10, v01 = (tz[:, :m] - tx) / n, 1.0 - (tz[:, m:] - ty) / m
    return v10.mean(1), np.atleast_2d(np.cov(v10)) / m + np.atleast_2d(np.cov(v01)) / n


def delong_p(y: Any, a: Any, b: Any) -> float:
    """Two-sided DeLong p of AUROC(b) - AUROC(a) on the same rows; 1 (0) when the difference has zero variance and is
    (is not) 0, NaN with fewer than two rows of a class."""
    y = np.asarray(y).reshape(-1) == 1
    if min(y.sum(), (~y).sum()) < 2:
        return NAN
    auc, cov = delong(y, a, b)
    d, var = auc[1] - auc[0], cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    return float(2.0 * norm.sf(abs(d) / np.sqrt(var))) if var > 1e-15 else float(d == 0)


def nadeau_bengio_p(d: Any) -> float:
    """Two-sided Nadeau-Bengio corrected resampled t-test of per-fold differences ``d`` (k-fold CV: variance factor
    1/k + n_test/n_train = 1/k + 1/(k-1), df k - 1; SPEC Appendix B). NaN below two finite folds; 1 (0) when every
    fold agrees and the mean is (is not) 0."""
    d = np.asarray(d, dtype=np.float64)
    d, k = d[np.isfinite(d)], int(np.isfinite(d).sum())
    if k < 2:
        return NAN
    se = np.sqrt((1.0 / k + 1.0 / (k - 1)) * d.var(ddof=1))
    return float(2.0 * student_t.sf(abs(d.mean()) / se, k - 1)) if se > 1e-15 else float(d.mean() == 0)


def cohort_digest(run_dir: Any) -> str:
    """SHA-256 of the retained (not ``excluded``) val and test :data:`COHORT_KEY` rows of
    ``<run>/cohort/guids.parquet``, sorted, with ``patient`` the bootstrap cluster (:func:`cluster`) and ``y`` the
    adverse outcome ``y > 0``.

    Equal digests mean the same val and test GUIDs of the same patients in the same folds, against the same binary
    outcome, so the runs' GUID-level rows pair. Label weights, labeling strategies, covariates and the train split
    (``data.train_epoch_min_s`` can drop train GUIDs) stay out, so their ablations compare. A ``three_class`` run
    (``y`` = class index 0-2) pairs with an ``adverse_vs_healthy`` one: its binary rows are the collapsed adverse score
    (§11.3). ``hie_vs_rest`` does not (another outcome).
    """
    g = pd.read_parquet(Path(run_dir) / "cohort" / "guids.parquet")
    g = g[~g["excluded"].astype(bool) & (g["split"] != "train")]
    key = g.assign(patient=cluster(g), y=(g["y"].astype(int) > 0).astype(int))[list(COHORT_KEY)]
    return hashlib.sha256(key.sort_values(["fold", "split", "guid"]).to_csv(index=False).encode()).hexdigest()


def paired_draws(frames: Sequence[pd.DataFrame], *, resamples: int, seed: int) -> list:
    """One outcome-stratified patient-cluster bootstrap shared by several populations (§11.8, paired).

    Per frame (rows with ``patient`` and ``y``), a ``(1 + resamples, rows)`` weight matrix: row 0 is the sample itself,
    row b each row's patient multiplicity in draw b. The draws cover the union of the frames' patients
    (:func:`_unit_draws`, strata by the labels a patient carries), so a statistic of one frame under row b is paired
    with every other frame's under row b.
    """
    allp = pd.concat([f[["patient", "y"]] for f in frames], ignore_index=True)
    M, codes = _unit_draws(allp["y"].to_numpy(np.int64), allp["patient"].astype(str), resamples, seed)
    M = np.vstack([np.ones((1, M.shape[1]), M.dtype), M])
    return [M[:, c].astype(np.float64) for c in np.split(codes, np.cumsum([len(f) for f in frames])[:-1])]


def _q_stat(f: pd.DataFrame, W: np.ndarray, inside: np.ndarray, met: str) -> np.ndarray:
    """``met`` of the rows ``inside`` of ``f`` under each weight row of ``W``: ``auroc`` of ``score``, ``sens`` /
    ``spec`` of the decisions ``fire``."""
    y = f["y"].to_numpy(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        if met == "auroc":
            o, st, _ = _ranked(f["score"].to_numpy(np.float64))
            return _rank_stats(*_curve(y, W * inside, o, st))["auroc"]
        fire = f["fire"].to_numpy(bool)
        hit, cls = (fire, y == 1) if met == "sens" else (~fire, y == 0)
        return (W @ (hit & cls & inside).astype(np.float64)) / (W @ (cls & inside).astype(np.float64))


def paired_deltas(ref: pd.DataFrame, run: pd.DataFrame, metrics: Sequence[str], *, resamples: int, seed: int, mn: int,
                  cells: Optional[pd.DataFrame] = None, member: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
                  confidence: float = 0.95) -> list:
    """Q2 records of ``run`` minus ``ref`` on two pooled OOF test populations under one :func:`paired_draws` bootstrap.

    Rows carry ``fold, guid, patient, y``, plus ``score`` for ``auroc`` and ``fire`` for ``sens``/``spec``. Records
    cover the whole population (``subgroup`` None, ``population`` mixed), then each cell of ``cells`` (``subgroup,
    subgroup_value, population``, membership ``member(frame)`` (C, rows)). A cell reports what its population allows,
    as S4 does (§11.6): ``delta_auroc`` for mixed cells, ``delta_sens`` where it holds adverse GUIDs, ``delta_spec``
    where it holds healthy ones.

    The point value is the sample (draw row 0), with a percentile CI and the two-sided bootstrap p = 2 min(P(d <= 0),
    P(d >= 0)). A ``delta_auroc`` record adds ``p_delong`` (:func:`delong_p`, on the cell's GUIDs) when ``ref`` and
    ``run`` hold the same (fold, guid) rows in the same order, else NaN; DeLong treats GUIDs as independent
    (``ponytail:`` no patient clustering; Obuchowski's clustered variance when a patient map groups GUIDs). Underpowered
    cells (a needed class below ``mn`` on either side) get NaN and no test, with the estimate in ``value_raw``. A cell
    empty on either side is skipped.
    """
    Wa, Wb = paired_draws([ref, run], resamples=resamples, seed=seed)
    ya, yb, a = ref["y"].to_numpy() == 1, run["y"].to_numpy() == 1, (1.0 - confidence) / 2.0
    keys = ["fold", "guid"]
    aligned = set(keys) <= set(ref) & set(run) and len(ref) == len(run) and (
        ref[keys].to_numpy() == run[keys].to_numpy()).all()
    todo = [(None, None, "mixed", np.ones(len(ref), bool), np.ones(len(run), bool))]
    if cells is not None and len(cells):
        Ma, Mb = member(ref), member(run)
        todo += [(c.subgroup, c.subgroup_value, c.population, Ma[i], Mb[i])
                 for i, c in enumerate(cells.itertuples(index=False))]
    skip = {"auroc": ("healthy_only", "unhealthy_only"), "sens": ("healthy_only",), "spec": ("unhealthy_only",)}
    out = []
    for sub, val, pop, ia, ib in todo:
        pa, na, pb, nb = (int(x.sum()) for x in (ia & ya, ia & ~ya, ib & yb, ib & ~yb))
        if not pa + na or not pb + nb:
            continue
        for met in (m for m in metrics if pop not in skip[m]):
            need = {"auroc": min(pa, na, pb, nb), "sens": min(pa, pb), "spec": min(na, nb)}[met]
            rows = slice(None) if need >= mn else slice(0, 1)  # the bootstrap only when powered
            ra, rb = _q_stat(ref, Wa[rows], ia, met), _q_stat(run, Wb[rows], ib, met)
            d = rb - ra
            dr = d[1:][np.isfinite(d[1:])]
            rec = dict(subgroup=sub, subgroup_value=val, population=pop, metric=f"delta_{met}", n_pos=pb, n_neg=nb,
                       underpowered=need < mn, value_raw=float(d[0]), run_value=float(rb[0]),
                       reference_value=float(ra[0]), value=NAN, ci_lo=NAN, ci_hi=NAN, p_value=NAN,
                       n_boot_undefined=None, p_delong=NAN)
            if met == "auroc" and aligned and need >= mn:
                rec["p_delong"] = delong_p(ya[ia], ref["score"].to_numpy(np.float64)[ia],
                                           run["score"].to_numpy(np.float64)[ia])
            if dr.size:
                rec |= dict(value=float(d[0]), ci_lo=float(np.quantile(dr, a)), ci_hi=float(np.quantile(dr, 1 - a)),
                            p_value=float(min(1.0, 2.0 * min((dr <= 0).mean(), (dr >= 0).mean()))),
                            n_boot_undefined=int(d.size - 1 - dr.size))
            out.append(rec)
    return out


def run_names(paths: Sequence[Any]) -> list:
    """Compared runs' names: their directory names, or the paths as given when two names coincide."""
    names = [Path(p).name for p in paths]
    return names if len(set(names)) == len(names) else [str(p) for p in paths]


def _q_side(ctx: SimpleNamespace, name: str, pm: Optional[Tuple[str, str]] = None) -> SimpleNamespace:
    """One compared run (``name``): its primary model's (or ``pm``'s) pooled OOF test rows, ``guid`` (:func:`level_units`,
    sorted by (fold, guid), so two runs' rows align for DeLong), its per-fold test GUID AUROC ``fold_auroc``
    (``score_final_cal``, for Nadeau-Bengio) and ``pols``, per GUID-level policy ``(thresholds entry, basis population
    with fire)`` at each fold's own threshold."""
    pm, st = pm or primary_model(_models(ctx)), ctx.cfg["data"]["shared_test_policy"]
    if pm is None:
        raise ValueError(f"{ctx.run_dir}: no prediction rows to compare")
    pols = {}
    for (level, pid), (ents, pop) in policy_populations(ctx, *pm, "test").items():
        if level == "guid":
            x = pool_rows(pop, st, "test")
            thr = x["fold"].map({f: e["threshold"] for f, e in ents.items()}).to_numpy(np.float64)
            pols[pid] = (next(iter(ents.values())), x.assign(fire=x["score"].to_numpy(np.float64) > thr))
    g = level_units(ctx, *pm, "test")["guid"]
    return SimpleNamespace(pm=pm, name=name, pols=pols,
                           fold_auroc={int(f): float(roc_auc_score(x["y"], x["score"])) if 0 < x["y"].sum() < len(x) else NAN
                                       for f, x in g.groupby("fold")},
                           guid=pool_rows(g, st, "test").sort_values(["fold", "guid"], ignore_index=True),
                           keys=_keys(ctx, *pm, split="test", level="guid", fold="pooled", point="n/a",
                                      denominator="n/a"))


def compare_runs(run_dirs: Sequence[Any], *, policy: Optional[str] = None) -> pd.DataFrame:
    """Q2 ``comparison.parquet`` rows (§11.9 columns + :data:`Q_EXTRA`) of 2-:data:`MAX_COMPARED` finished runs.

    The first run is the reference. For each other run, :func:`paired_deltas` of its primary model minus the
    reference's, on the pooled OOF test GUIDs, for the whole population and every member of the reference's subgroup
    families (:func:`subgroup_families`). Membership is the reference's, which holds for every run because the cohorts
    match. Rows:

    * ``delta_auroc`` of ``score_final_cal``, with the DeLong p (``p_delong``, :func:`paired_deltas`) and, on the whole
      population, the Nadeau-Bengio p of the per-fold test AUROC differences (``p_nb``, :func:`nadeau_bengio_p`, over
      the folds both runs have);
    * per GUID-level policy every run has (``policy``: that one only), ``delta_sens`` and ``delta_spec``, each run on
      its own basis population at its own fold thresholds.

    The bootstrap uses the reference's ``eval.bootstrap`` and the power guard its ``eval.min_subgroup_n``. ``p_holm`` is
    Holm within (run, policy, metric, family); the whole population is a family of its own. Runs are named by
    :func:`run_names`.

    Raises:
        ValueError: If fewer than 2 or more than :data:`MAX_COMPARED` runs are given; if the cohort digests differ (the
            runs do not pair); if the pooled test GUIDs differ (e.g. a source dropped GUIDs at predict, or the
            shared-test policies differ); or if ``policy`` is not a GUID-level policy of every run's primary model.
    """
    from teb_vae.classifier.config import load

    paths = [Path(r) for r in run_dirs]
    if not 2 <= len(paths) <= MAX_COMPARED:
        raise ValueError(f"compare takes 2 to {MAX_COMPARED} runs, got {len(paths)}")
    digests = {str(p): cohort_digest(p) for p in paths}
    if len(set(digests.values())) > 1:
        raise ValueError("cohort digests differ, so the runs' GUID-level rows do not pair: "
                         + ", ".join(f"{p} {d[:12]}" for p, d in digests.items()))
    ref, sides = None, []
    for p, name in zip(paths, run_names(paths)):  # one run's predictions at a time; the reference's context is kept
        ctx = load_context(p, load(p / "config.resolved.yaml"))
        ref = ref or ctx
        sides.append(_q_side(ctx, name))
    a, ev = sides[0], ref.cfg["eval"]
    for s in sides[1:]:
        ka, kb = set(zip(a.guid["fold"], a.guid["guid"])), set(zip(s.guid["fold"], s.guid["guid"]))
        if ka != kb:
            raise ValueError(f"{s.name}: {len(ka ^ kb)} pooled test GUIDs are not shared with the reference {a.name} "
                             f"(a GUID dropped at predict, or another data.shared_test_policy); the runs do not pair")
    common = [p["id"] for p in ev["thresholds"] if all(p["id"] in s.pols for s in sides)]
    if policy is not None and policy not in common:
        raise ValueError(f"policy {policy!r} is not a GUID-level policy of every run's primary model; shared: {common}")
    cells = _cells(ref) if subgroup_families(ref.cfg) else None
    kw = dict(mn=ev["min_subgroup_n"], cells=cells, member=lambda f: _member_matrix(ref, f["fold"], f["guid"]),
              **_boot(ev))
    recs = []
    for b in sides[1:]:
        k = b.keys | dict(run=b.name, reference=a.name, reference_model="|".join(a.pm))
        nb = nadeau_bengio_p([b.fold_auroc[f] - a.fold_auroc[f] for f in sorted(set(a.fold_auroc) & set(b.fold_auroc))])
        recs += [r | k | {"metric_type": "threshold_free", "p_nb": nb if r["subgroup"] is None else NAN}
                 for r in paired_deltas(a.guid, b.guid, ["auroc"], **kw)]
        for pid in [policy] if policy else common:
            (e, xb), xa = b.pols[pid], a.pols[pid][1]
            pk = dict(policy_id=pid, policy_basis=e["basis"], metric_type=e["basis"] if e["basis"] in _DENOM else None,
                      denominator=_DENOM.get(e["basis"], "n/a"), axis=None if e["at"] == "end" else e["axis"],
                      t=NAN if e["at"] == "end" else float(e["at"]))
            recs += [r | k | pk for r in paired_deltas(xa, xb, ["sens", "spec"], **kw)]
    t = pd.DataFrame(recs).reindex(columns=[*METRIC_COLUMNS, *Q_EXTRA])
    fam = [t[c].fillna("").astype(str) for c in ("run", "policy_id", "metric", "subgroup")]
    t["p_holm"] = t.groupby(fam)["p_value"].transform(
        lambda p: pd.Series(vae_stats.holm_adjust(p.tolist()), index=p.index))
    for c in (*_NUMERIC, "value_raw", "run_value", "reference_value", "p_value", "p_holm", "p_delong", "p_nb"):
        t[c] = pd.to_numeric(t[c], errors="coerce").astype(float)
    t["underpowered"] = t["underpowered"].fillna(False).astype(bool)
    return t
