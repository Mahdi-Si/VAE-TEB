"""GUID-clustered statistics helpers for the causal-TE validation suite.

Segments from the same patient (GUID) are *not* independent. Every
test in this submodule respects the cluster structure with one of two
mechanisms:

* **Random-intercept regressions** ($y_i = \\beta x_i + b_{\\mathrm{GUID}(i)} + \\varepsilon_i$)
  via :class:`statsmodels.regression.mixed_linear_model.MixedLM`, when
  available. Returns coefficient estimates with proper random-intercept
  inference.
* **Cluster bootstrap** when statsmodels is not installed: resample
  unique GUIDs with replacement, fit OLS on the resampled superset,
  and report percentile CIs.

The Wilcoxon helpers here are paired one-sample tests on the per-sample
delta (segment-level), with optional GUID bootstrap for CIs on the
median delta.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
except Exception:  # pragma: no cover - scipy is a hard dep elsewhere
    _scipy_wilcoxon = None  # type: ignore[assignment]

try:
    import statsmodels.formula.api as _smf  # type: ignore[import-not-found]
    _HAS_STATSMODELS = True
except Exception:  # pragma: no cover - documented optional dep
    _smf = None  # type: ignore[assignment]
    _HAS_STATSMODELS = False


# ---------------------------------------------------------------------------
# Wilcoxon
# ---------------------------------------------------------------------------


def paired_wilcoxon(
    deltas: np.ndarray,
    *,
    alternative: str = "greater",
) -> Dict[str, float]:
    """Run a one-sample Wilcoxon signed-rank test on per-sample deltas.

    Args:
        deltas: 1-D array of per-sample paired differences. NaN entries
            are dropped before testing.
        alternative: One of ``{"two-sided", "greater", "less"}``.

    Returns:
        Dict with ``n_pairs``, ``W`` (signed-rank statistic),
        ``p_value``, ``median_delta``, ``frac_positive`` (share of
        non-NaN deltas with $\\Delta > 0$). Returns NaN-filled fields
        when the test cannot run (n < 5 or scipy unavailable).
    """
    arr = np.asarray(deltas, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    out: Dict[str, float] = {
        "n_pairs": float(n),
        "W": float("nan"),
        "p_value": float("nan"),
        "median_delta": float(np.median(arr)) if n else float("nan"),
        "frac_positive": float(np.mean(arr > 0.0)) if n else float("nan"),
    }
    if n < 5 or _scipy_wilcoxon is None:
        return out
    try:
        # ``zero_method='wilcox'`` drops zero-deltas before ranking.
        result = _scipy_wilcoxon(arr, alternative=alternative, zero_method="wilcox")
        # Result behaves like a tuple ``(stat, pvalue)`` and (since SciPy 1.7)
        # also exposes ``.statistic`` / ``.pvalue`` attributes.
        try:
            stat = float(result.statistic)  # type: ignore[attr-defined]
            pval = float(result.pvalue)  # type: ignore[attr-defined]
        except AttributeError:
            stat, pval = float(result[0]), float(result[1])  # type: ignore[index]
        out["W"] = stat
        out["p_value"] = pval
    except Exception:
        # Degenerate input (all zeros, identical values). Leave NaN.
        pass
    return out


def holm_correction(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment of a vector of p-values.

    NaN p-values pass through unchanged.

    Args:
        p_values: Sequence of unadjusted p-values.

    Returns:
        Numpy array of Holm-adjusted p-values, same order as input.
    """
    p = np.asarray(p_values, dtype=np.float64)
    finite_mask = np.isfinite(p)
    out = p.copy()
    finite_idx = np.flatnonzero(finite_mask)
    m = int(finite_idx.size)
    if m == 0:
        return out
    sub = p[finite_idx]
    order = np.argsort(sub)
    adj = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        scaled = (m - rank) * sub[idx]
        running_max = max(running_max, scaled)
        adj[idx] = min(1.0, running_max)
    out[finite_idx] = adj
    return out


# ---------------------------------------------------------------------------
# GUID-cluster bootstrap
# ---------------------------------------------------------------------------


def guid_bootstrap_ci(
    values: np.ndarray,
    guids: np.ndarray,
    *,
    stat_fn: Callable[[np.ndarray], float] = np.nanmean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Cluster-percentile bootstrap CI grouped by GUID.

    Resampling protocol: draw $|\\mathrm{unique\\_guids}|$ GUIDs with
    replacement; for each draw, take **all** values from that GUID;
    apply ``stat_fn`` to the resulting concatenated array; repeat
    ``n_boot`` times. Percentile CI at ``[alpha/2, 1 - alpha/2]``.

    Args:
        values: Per-segment values (1-D, same length as ``guids``).
        guids: Per-segment GUID strings.
        stat_fn: Aggregator (default :func:`numpy.nanmean`). Receives a
            1-D numpy array.
        n_boot: Number of bootstrap iterations.
        alpha: Two-sided $\\alpha$ for the percentile CI.
        seed: RNG seed.

    Returns:
        ``(ci_low, ci_high)``. Returns ``(nan, nan)`` when fewer than
        two distinct GUIDs are available.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    g = np.asarray(guids).ravel()
    if v.size != g.size or v.size == 0:
        return (float("nan"), float("nan"))
    finite = np.isfinite(v) & (g != None)  # noqa: E711
    v = v[finite]
    g = g[finite]
    unique_guids, inverse = np.unique(g, return_inverse=True)
    n_g = int(unique_guids.size)
    if n_g < 2:
        return (float("nan"), float("nan"))

    # Pre-bucket per-guid index lists for fast resampling.
    buckets: List[np.ndarray] = [
        np.flatnonzero(inverse == j) for j in range(n_g)
    ]
    rng = np.random.default_rng(int(seed))
    scores = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        sampled = rng.integers(0, n_g, size=n_g)
        idx = np.concatenate([buckets[s] for s in sampled])
        if idx.size == 0:
            scores[b] = np.nan
            continue
        try:
            scores[b] = float(stat_fn(v[idx]))
        except Exception:
            scores[b] = np.nan
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size < 10:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(finite_scores, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(finite_scores, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


# ---------------------------------------------------------------------------
# OLS + cluster-bootstrap regression (statsmodels-free fallback)
# ---------------------------------------------------------------------------


def _design_matrix(
    df: pd.DataFrame,
    formula: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build a design matrix from a Patsy-style formula without Patsy.

    Supports the limited subset used by this submodule:
    ``y ~ x1 + x2 + ...`` with numeric columns and an automatic
    intercept. Categorical columns are NOT supported here — encode them
    upstream as one-hot if needed. NaN rows are dropped.

    Args:
        df: Source DataFrame.
        formula: ``"y ~ x1 + x2 + ..."``.

    Returns:
        Tuple ``(y, X, term_names)``. ``X`` includes an Intercept column.
    """
    if "~" not in formula:
        raise ValueError(f"formula must contain '~': got {formula!r}")
    y_part, x_part = (s.strip() for s in formula.split("~", 1))
    y_col = y_part
    x_cols = [s.strip() for s in x_part.split("+")] if x_part else []
    x_cols = [c for c in x_cols if c and c != "1"]
    needed = [y_col, *x_cols]
    raw = df.loc[:, needed].copy()
    for col in needed:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    sub = raw.dropna()
    y = np.asarray(sub[y_col].values, dtype=np.float64)
    n = int(y.size)
    if n == 0 or not x_cols:
        return y, np.ones((n, 1)), ["Intercept"]
    x_block = np.asarray(sub[x_cols].values, dtype=np.float64)
    X = np.column_stack([np.ones(n, dtype=np.float64), x_block])
    names = ["Intercept", *x_cols]
    return y, X, names


def _ols_fit(y: np.ndarray, X: np.ndarray) -> Optional[np.ndarray]:
    """Solve $\\arg\\min_\\beta \\|y - X\\beta\\|^2$ via numpy lstsq.

    Args:
        y: Response vector $(n,)$.
        X: Design matrix $(n, p)$.

    Returns:
        Coefficient vector $(p,)$, or ``None`` if the system is
        singular / ``y`` has insufficient observations.
    """
    if y.size <= X.shape[1]:
        return None
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        if not np.all(np.isfinite(beta)):
            return None
        return np.asarray(beta, dtype=np.float64)
    except np.linalg.LinAlgError:
        return None


def _cluster_bootstrap_regression(
    df: pd.DataFrame,
    formula: str,
    guid_col: str,
    *,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> Optional[Dict[str, Any]]:
    """OLS coefficient inference via GUID-clustered bootstrap.

    Args:
        df: Source DataFrame including ``guid_col``.
        formula: Patsy-style ``"y ~ x1 + x2 + ..."``.
        guid_col: Cluster key.
        n_boot: Bootstrap iterations.
        alpha: Two-sided $\\alpha$ for percentile CIs.
        seed: RNG seed.

    Returns:
        Dict with keys ``coefs`` (DataFrame), ``method``, ``n_obs``,
        ``n_guids``, or ``None`` on failure.
    """
    needed = list(_extract_columns(formula)) + [guid_col]
    sub = df.dropna(subset=needed).copy()
    if sub.empty:
        return None
    y, X, names = _design_matrix(sub, formula)
    if y.size <= X.shape[1]:
        return None
    point = _ols_fit(y, X)
    if point is None:
        return None

    g = sub[guid_col].to_numpy()
    unique_guids, inverse = np.unique(g, return_inverse=True)
    n_g = int(unique_guids.size)
    if n_g < 2:
        return None
    buckets = [np.flatnonzero(inverse == j) for j in range(n_g)]
    rng = np.random.default_rng(int(seed))
    boot = np.full((int(n_boot), point.size), np.nan, dtype=np.float64)
    for b in range(int(n_boot)):
        sampled = rng.integers(0, n_g, size=n_g)
        idx = np.concatenate([buckets[s] for s in sampled])
        if idx.size <= X.shape[1]:
            continue
        beta_b = _ols_fit(y[idx], X[idx])
        if beta_b is not None:
            boot[b] = beta_b

    rows = []
    for j, term in enumerate(names):
        col = boot[:, j]
        finite = col[np.isfinite(col)]
        if finite.size < 10:
            se = float("nan"); lo = float("nan"); hi = float("nan"); p = float("nan")
        else:
            se = float(np.std(finite, ddof=1))
            lo = float(np.percentile(finite, 100.0 * (alpha / 2.0)))
            hi = float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0)))
            # Two-sided p approximated as 2 * min(P(b<=0), P(b>=0)).
            p_left = float(np.mean(finite <= 0.0))
            p_right = float(np.mean(finite >= 0.0))
            p = float(min(1.0, 2.0 * min(p_left, p_right)))
        rows.append({
            "term": term,
            "estimate": float(point[j]),
            "se": se,
            "ci_low": lo,
            "ci_high": hi,
            "p_value": p,
        })
    coefs = pd.DataFrame(rows)
    return {
        "coefs": coefs,
        "method": "cluster_bootstrap",
        "n_obs": int(y.size),
        "n_guids": int(n_g),
    }


def _extract_columns(formula: str) -> Sequence[str]:
    """Return the column names referenced by a simple ``y ~ x1 + x2`` formula."""
    if "~" not in formula:
        return []
    y_part, x_part = (s.strip() for s in formula.split("~", 1))
    cols = [y_part]
    for s in x_part.split("+"):
        s = s.strip()
        if s and s != "1":
            cols.append(s)
    return cols


def _statsmodels_mixedlm(
    df: pd.DataFrame,
    formula: str,
    guid_col: str,
    *,
    alpha: float,
) -> Optional[Dict[str, Any]]:
    """Fit a random-intercept MixedLM via statsmodels (when installed)."""
    if not _HAS_STATSMODELS or _smf is None:
        return None
    needed = list(_extract_columns(formula)) + [guid_col]
    sub = df.dropna(subset=needed).copy()
    if sub.empty:
        return None
    try:
        n_g = int(sub[guid_col].nunique())  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if n_g < 2:
        return None
    try:
        model = _smf.mixedlm(formula, sub, groups=sub[guid_col])
        result = model.fit(method=["lbfgs"], reml=True)
    except Exception:
        return None
    if not getattr(result, "converged", True):
        return None

    rows = []
    z = float(_normal_quantile(1.0 - alpha / 2.0))
    for term in result.params.index:
        if term == "Group Var":
            continue
        est = float(result.params[term])
        se = float(result.bse.get(term, float("nan")))
        p = float(result.pvalues.get(term, float("nan")))
        lo = est - z * se if np.isfinite(se) else float("nan")
        hi = est + z * se if np.isfinite(se) else float("nan")
        rows.append({
            "term": str(term),
            "estimate": est,
            "se": se,
            "ci_low": float(lo),
            "ci_high": float(hi),
            "p_value": p,
        })
    return {
        "coefs": pd.DataFrame(rows),
        "method": "mixedlm",
        "n_obs": int(sub.shape[0]),
        "n_guids": n_g,
    }


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF via :func:`scipy.stats.norm.ppf` if available.

    Falls back to a rational approximation (Beasley-Springer-Moro) when
    scipy is missing — this keeps the helper usable in offline test
    environments.
    """
    try:
        from scipy.stats import norm  # local to avoid hard dep at import
        return float(norm.ppf(p))
    except Exception:
        # Rational approximation; max abs error ~ 4e-4 in [1e-3, 1-1e-3].
        if p <= 0.0 or p >= 1.0:
            return float("nan")
        # Beasley-Springer-Moro (Glasserman 2003 p. 67).
        a = (-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
             1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0)
        b = (-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
             6.680131188771972e1, -1.328068155288572e1)
        c = (-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
             -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0)
        d = (7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
             3.754408661907416e0)
        plow = 0.02425
        phigh = 1 - plow
        if p < plow:
            q = (-2 * np.log(p)) ** 0.5
            return float((((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                         ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
        if p <= phigh:
            q = p - 0.5
            r = q * q
            return float((((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
                         (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1))
        q = (-2 * np.log(1 - p)) ** 0.5
        return float(-(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                     ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))


def guid_clustered_mixedlm(
    formula: str,
    df: pd.DataFrame,
    guid_col: str = "guid",
    *,
    fallback: str = "cluster_bootstrap",
    alpha: float = 0.05,
    n_boot: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Random-intercept regression with a numpy-only fallback.

    Tries statsmodels' MixedLM first; on any failure (not installed,
    singular, non-convergent) falls back to OLS + GUID cluster bootstrap.
    The returned dict always has the same shape regardless of which
    path was taken; callers should consult ``method`` to know which.

    Args:
        formula: Patsy-style ``"y ~ x1 + x2 + ..."``. Categorical terms
            must be one-hot encoded upstream.
        df: Source DataFrame.
        guid_col: Cluster key (default ``"guid"``).
        fallback: ``"cluster_bootstrap"`` or ``"none"`` — when ``"none"``
            and statsmodels is unavailable, returns an empty result with
            ``method = "unavailable"``.
        alpha: Two-sided $\\alpha$ for CIs.
        n_boot: Bootstrap iterations (only used by the fallback).
        seed: RNG seed.

    Returns:
        Dict with ``coefs`` (DataFrame[term, estimate, se, ci_low, ci_high, p_value]),
        ``method`` (``"mixedlm"`` | ``"cluster_bootstrap"`` | ``"unavailable"``),
        ``n_obs``, ``n_guids``.
    """
    res = _statsmodels_mixedlm(df, formula, guid_col, alpha=alpha)
    if res is not None:
        return res
    if fallback == "cluster_bootstrap":
        boot = _cluster_bootstrap_regression(
            df, formula, guid_col, n_boot=n_boot, alpha=alpha, seed=seed,
        )
        if boot is not None:
            return boot
    return {
        "coefs": pd.DataFrame(
            columns=["term", "estimate", "se", "ci_low", "ci_high", "p_value"],
        ),
        "method": "unavailable",
        "n_obs": 0,
        "n_guids": 0,
    }


def has_statsmodels() -> bool:
    """Return ``True`` when ``statsmodels`` is importable in this environment."""
    return bool(_HAS_STATSMODELS)
