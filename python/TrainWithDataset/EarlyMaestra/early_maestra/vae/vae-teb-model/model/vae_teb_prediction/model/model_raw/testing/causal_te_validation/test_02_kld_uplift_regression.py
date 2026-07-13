"""Test 2 — KLD-uplift GUID-clustered regression.

Regresses per-sample relative uplift on the per-sample bottleneck
information $K_i$ with a random intercept per GUID:

$$
\\mathrm{uplift}^{\\mathrm{rel}}_i
=
\\beta_0 + \\beta_1 K_i + \\beta_2 E^{\\mathrm{base}}_i
+ \\beta_3 \\mathrm{label}_i + \\beta_4 \\mathrm{epoch}_i
+ b_{\\mathrm{GUID}(i)} + \\varepsilon_i.
$$

A negative-control regression $E^{\\mathrm{full}}_i$ on the same
predictors flags the failure mode "KLD just tracks difficulty".

Inputs (read-only): ``<output>/histograms/histogram_metrics.csv`` —
must contain columns ``guid, epoch, label, kld_mean, kld_sum,
base_mse_total, uplift_rel, feat_mse_total``.

Outputs: ``coefficients.csv`` with one row per coefficient term.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from model.vae_teb_prediction.model.model_raw.testing.causal_te_validation.statistics import (
    guid_clustered_mixedlm,
)


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric, NaN-trimmed copy with the predictor columns we need."""
    needed = [
        "guid", "label", "epoch",
        "kld_mean", "kld_sum",
        "base_mse_total", "feat_mse_total",
        "uplift_rel",
    ]
    keep = [c for c in needed if c in df.columns]
    sub = df.loc[:, keep].copy()
    for col in ("epoch", "kld_mean", "kld_sum", "base_mse_total",
                "feat_mse_total", "uplift_rel"):
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
    if "label" in sub.columns:
        sub["label"] = pd.to_numeric(sub["label"], errors="coerce")
    return sub


def _fit_one(
    df: pd.DataFrame,
    *,
    y: str,
    predictors: List[str],
    drop_label_if_single: bool = True,
) -> Dict[str, Any]:
    """Fit a single random-intercept regression with NaN-safe column drop."""
    use = []
    for c in predictors:
        if c not in df.columns:
            continue
        notna_any = bool(df[c].notna().any())
        if notna_any:
            use.append(c)
    if drop_label_if_single and "label" in use:
        try:
            n_unique_labels = int(df["label"].nunique(dropna=True))
        except (TypeError, ValueError):
            n_unique_labels = 0
        if n_unique_labels <= 1:
            use = [c for c in use if c != "label"]
    if not use:
        return {
            "coefs": pd.DataFrame(
                columns=["term", "estimate", "se", "ci_low", "ci_high", "p_value"],
            ),
            "method": "unavailable", "n_obs": 0, "n_guids": 0,
        }
    formula = f"{y} ~ " + " + ".join(use)
    return guid_clustered_mixedlm(formula, df, guid_col="guid")


def _coef_value(coefs: pd.DataFrame, term: str) -> Dict[str, float]:
    """Extract the coefficient row for ``term``, NaN-filling if missing."""
    if coefs.empty:
        return {"estimate": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_value": float("nan")}
    row = coefs[coefs["term"] == term]
    if row.empty:
        return {"estimate": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_value": float("nan")}
    r = row.iloc[0]
    return {
        "estimate": float(r["estimate"]),
        "ci_low": float(r["ci_low"]),
        "ci_high": float(r["ci_high"]),
        "p_value": float(r["p_value"]),
    }


def run(
    *,
    histogram_csv: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Fit the primary and negative-control KLD-uplift regressions.

    Args:
        histogram_csv: Path to ``<output>/histograms/histogram_metrics.csv``.
        output_dir: Path to ``<output>/causal_te_validation/kld_uplift_regression``.

    Returns:
        Dict with ``verdict``, ``evidence`` (consumed by
        ``decision_rules.verdict_test_02_kld_uplift``), ``csv_paths``,
        ``figure_paths``, ``method``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv = Path(histogram_csv)
    if not csv.is_file():
        return {
            "verdict": "missing",
            "evidence": {},
            "error": f"missing input: {csv}",
            "csv_paths": [],
            "figure_paths": [],
        }
    raw = pd.read_csv(csv)
    df = _prepare_features(raw)
    if df.empty or "guid" not in df.columns:
        return {
            "verdict": "missing",
            "evidence": {},
            "error": "histogram CSV is missing required columns or rows",
            "csv_paths": [],
            "figure_paths": [],
        }

    primary_predictors = ["kld_mean", "base_mse_total", "label", "epoch"]
    primary = _fit_one(
        df, y="uplift_rel", predictors=primary_predictors,
        drop_label_if_single=True,
    )
    primary_coefs = primary["coefs"]
    primary_coefs = primary_coefs.assign(
        regression="primary_uplift_vs_kld",
        method=primary["method"],
        n_obs=primary["n_obs"],
        n_guids=primary["n_guids"],
    )

    negctrl_predictors = ["kld_mean", "base_mse_total", "label"]
    negctrl = _fit_one(
        df, y="feat_mse_total", predictors=negctrl_predictors,
        drop_label_if_single=True,
    )
    negctrl_coefs = negctrl["coefs"].assign(
        regression="negctrl_efull_vs_kld",
        method=negctrl["method"],
        n_obs=negctrl["n_obs"],
        n_guids=negctrl["n_guids"],
    )

    coefs = pd.concat([primary_coefs, negctrl_coefs], ignore_index=True)
    csv_path = output_dir / "coefficients.csv"
    coefs.to_csv(csv_path, index=False)

    beta1 = _coef_value(primary_coefs, "kld_mean")
    gamma1 = _coef_value(negctrl_coefs, "kld_mean")
    beta1_positive = (
        np.isfinite(beta1["ci_low"]) and beta1["ci_low"] > 0.0
    )
    gamma1_positive = (
        np.isfinite(gamma1["ci_low"]) and gamma1["ci_low"] > 0.0
    )
    evidence = {
        "beta1_positive": bool(beta1_positive),
        "gamma1_positive": bool(gamma1_positive),
        "beta1_estimate": beta1["estimate"],
        "beta1_ci_low": beta1["ci_low"],
        "beta1_ci_high": beta1["ci_high"],
        "beta1_p_value": beta1["p_value"],
        "gamma1_estimate": gamma1["estimate"],
        "gamma1_ci_low": gamma1["ci_low"],
        "gamma1_ci_high": gamma1["ci_high"],
        "method": primary["method"],
        "n_obs": int(primary["n_obs"]),
        "n_guids": int(primary["n_guids"]),
    }

    from model.vae_teb_prediction.model.model_raw.testing.causal_te_validation.decision_rules import (
        verdict_test_02_kld_uplift,
    )
    verdict = verdict_test_02_kld_uplift(evidence)

    return {
        "verdict": verdict,
        "evidence": evidence,
        "csv_paths": [str(csv_path)],
        "figure_paths": [],
        "method": primary["method"],
        "df": df,  # passed to plots.py for the scatter; runner trims this off
                   # before json-serialising the summary dict.
    }


__all__ = ["run"]
