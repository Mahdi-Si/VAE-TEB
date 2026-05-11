"""Test 3 — Band-specific UP$\\to$FHR transfer regression.

Joins the per-band per-sample CSVs from
``frequency_band_forecast/<partition>/per_sample.csv`` (extended with
``mse_base, uplift_abs, uplift_rel`` columns by the patched
:mod:`model.vae_teb_prediction.testing.analyses.frequency_band_forecast`
analysis) to ``histograms/histogram_metrics.csv`` on
``(guid, epoch)`` for the per-sample bottleneck information $K_i$,
then fits one random-intercept regression per band $b$:

$$
\\mathrm{uplift}_{i,b}
=
\\alpha_{0,b}
+ \\alpha_{1,b} K_i
+ \\alpha_{2,b} E^{\\mathrm{base}}_{i,b}
+ b_{\\mathrm{GUID}(i)}
+ \\varepsilon_{i,b}.
$$

Spec partitions evaluated: ``clinical_4band`` (slow_baseline, decel,
variability, beat_to_beat) and ``clinical_7band`` (the canonical 7-tile
split — the primary partition for the spec's decision rule).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from model.vae_teb_prediction.testing.causal_te_validation.statistics import (
    guid_clustered_mixedlm,
    holm_correction,
)


_PARTITIONS: Tuple[str, ...] = ("clinical_4band", "clinical_7band")


def _load_band_per_sample(band_forecast_dir: Path, partition: str) -> pd.DataFrame:
    """Read one partition's ``per_sample.csv``; return empty DF if missing."""
    csv = Path(band_forecast_dir) / partition / "per_sample.csv"
    if not csv.is_file():
        return pd.DataFrame()
    return pd.read_csv(csv)


def _fit_band(
    df_band: pd.DataFrame,
    *,
    band: str,
    partition: str,
) -> Dict[str, Any]:
    """Fit one band's regression and return a tidy row dict.

    NaN-safe: if the band has fewer than 5 samples or fewer than 2
    GUIDs, a placeholder row with NaN coefficients is returned so the
    output table still has one row per band.
    """
    sub = df_band.dropna(subset=["uplift_rel", "kld_mean", "mse_base", "guid"])
    n_obs = int(len(sub))
    n_guids = int(sub["guid"].nunique()) if n_obs else 0
    placeholder = {
        "partition": partition,
        "band": band,
        "n_obs": n_obs,
        "n_guids": n_guids,
        "n_channels": int(sub["n_channels"].iloc[0]) if n_obs else 0,
        "alpha1_estimate": float("nan"),
        "alpha1_ci_low": float("nan"),
        "alpha1_ci_high": float("nan"),
        "alpha1_p_value": float("nan"),
        "alpha2_estimate": float("nan"),
        "alpha2_ci_low": float("nan"),
        "alpha2_ci_high": float("nan"),
        "alpha2_p_value": float("nan"),
        "method": "skipped",
    }
    if n_obs < 5 or n_guids < 2:
        return placeholder

    # Predictors: kld_mean (the bottleneck information $K_i$) and the
    # band's own baseline MSE so we control for band-specific difficulty.
    formula = "uplift_rel ~ kld_mean + mse_base"
    fit = guid_clustered_mixedlm(formula, sub, guid_col="guid")
    coefs = fit["coefs"]
    out = dict(placeholder)
    out["method"] = fit["method"]
    out["n_obs"] = int(fit["n_obs"]) or n_obs
    out["n_guids"] = int(fit["n_guids"]) or n_guids
    if not coefs.empty:
        a1 = coefs[coefs["term"] == "kld_mean"]
        a2 = coefs[coefs["term"] == "mse_base"]
        if not a1.empty:
            out["alpha1_estimate"] = float(a1["estimate"].iloc[0])
            out["alpha1_ci_low"] = float(a1["ci_low"].iloc[0])
            out["alpha1_ci_high"] = float(a1["ci_high"].iloc[0])
            out["alpha1_p_value"] = float(a1["p_value"].iloc[0])
        if not a2.empty:
            out["alpha2_estimate"] = float(a2["estimate"].iloc[0])
            out["alpha2_ci_low"] = float(a2["ci_low"].iloc[0])
            out["alpha2_ci_high"] = float(a2["ci_high"].iloc[0])
            out["alpha2_p_value"] = float(a2["p_value"].iloc[0])
    return out


def _evidence_from_results(results: pd.DataFrame) -> Dict[str, Any]:
    """Translate the per-band coefficient table into the verdict evidence dict."""
    if results.empty:
        return {
            "alpha1_deceleration_positive": False,
            "alpha1_early_decel_positive": False,
            "alpha1_late_decel_positive": False,
            "alpha1_variability_positive": False,
            "alpha1_nyquist_edge_positive": False,
            "alpha1_beat_to_beat_positive": False,
            "alpha1_all_bands_positive": False,
            "n_bands": 0,
        }
    # Use the clinical_7band rows for the primary verdict; fall back to
    # clinical_4band when 7band is empty (e.g. degenerate run).
    primary = results[results["partition"] == "clinical_7band"]
    if primary.empty:
        primary = results[results["partition"] == "clinical_4band"]

    def _band_pass(name: str) -> bool:
        sub = primary[primary["band"] == name]
        if sub.empty:
            return False
        ci_low = float(sub["alpha1_ci_low"].iloc[0])
        p_holm = float(sub["alpha1_p_holm"].iloc[0]) if "alpha1_p_holm" in sub.columns else float("nan")
        return bool(np.isfinite(ci_low) and ci_low > 0.0 and (np.isnan(p_holm) or p_holm < 0.05))

    bands_present = primary["band"].unique().tolist()
    n_pos = sum(1 for b in bands_present if _band_pass(b))
    return {
        "alpha1_deceleration_positive": _band_pass("deceleration"),
        "alpha1_early_decel_positive":  _band_pass("early_decel"),
        "alpha1_late_decel_positive":   _band_pass("late_decel"),
        "alpha1_variability_positive":  _band_pass("variability") or _band_pass("lf_var") or _band_pass("mf_var"),
        "alpha1_nyquist_edge_positive": _band_pass("nyquist_edge"),
        "alpha1_beat_to_beat_positive": _band_pass("beat_to_beat"),
        "alpha1_all_bands_positive":    bool(bands_present and n_pos == len(bands_present)),
        "n_bands": int(len(bands_present)),
        "n_bands_positive": int(n_pos),
    }


def run(
    *,
    histogram_csv: Path,
    band_forecast_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Fit per-band regressions and write the per-band coefficients CSV.

    Args:
        histogram_csv: ``<output>/histograms/histogram_metrics.csv``.
        band_forecast_dir: ``<output>/frequency_band_forecast``.
        output_dir: ``<output>/causal_te_validation/band_uplift_regression``.

    Returns:
        Dict with ``verdict``, ``evidence``, ``csv_paths``, ``figure_paths``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(histogram_csv).is_file():
        return {
            "verdict": "missing", "evidence": {},
            "error": f"missing input: {histogram_csv}",
            "csv_paths": [], "figure_paths": [],
        }
    hist = pd.read_csv(histogram_csv)
    if "kld_mean" not in hist.columns or "guid" not in hist.columns:
        return {
            "verdict": "missing", "evidence": {},
            "error": "histogram CSV missing required columns",
            "csv_paths": [], "figure_paths": [],
        }
    kld_keys = ["guid", "epoch", "kld_mean"] if "epoch" in hist.columns else ["guid", "kld_mean"]
    hist_kld = hist.loc[:, kld_keys].copy()
    for c in kld_keys:
        if c != "guid":
            hist_kld[c] = pd.to_numeric(hist_kld[c], errors="coerce")

    rows: List[Dict[str, Any]] = []
    for partition in _PARTITIONS:
        per_sample = _load_band_per_sample(band_forecast_dir, partition)
        if per_sample.empty:
            continue
        if "uplift_rel" not in per_sample.columns or "mse_base" not in per_sample.columns:
            # Old per_sample.csv (pre-extension); skip with a row-level note.
            rows.append({
                "partition": partition, "band": "<all>", "n_obs": 0,
                "n_guids": 0, "n_channels": 0,
                "alpha1_estimate": float("nan"), "alpha1_ci_low": float("nan"),
                "alpha1_ci_high": float("nan"), "alpha1_p_value": float("nan"),
                "alpha2_estimate": float("nan"), "alpha2_ci_low": float("nan"),
                "alpha2_ci_high": float("nan"), "alpha2_p_value": float("nan"),
                "method": "missing_columns",
            })
            continue

        # Cast numeric columns and drop rows with missing keys.
        for c in ("epoch", "uplift_rel", "mse_base", "uplift_abs",
                  "mse_total", "n_channels"):
            if c in per_sample.columns:
                per_sample[c] = pd.to_numeric(per_sample[c], errors="coerce")
        join_keys = [k for k in ("guid", "epoch") if k in per_sample.columns and k in hist_kld.columns]
        merged = per_sample.merge(hist_kld, on=join_keys, how="inner")
        if merged.empty:
            continue
        bands = sorted(b for b in merged["band"].dropna().unique().tolist())
        for band in bands:
            df_b = merged[merged["band"] == band]
            rows.append(_fit_band(df_b, band=band, partition=partition))

    if not rows:
        return {
            "verdict": "missing", "evidence": {},
            "error": "no per-band rows to fit",
            "csv_paths": [], "figure_paths": [],
        }

    coefs_df = pd.DataFrame(rows)
    # Holm correction across all (partition, band) cells per partition,
    # only over the alpha_1 (kld_mean) p-values.
    coefs_df["alpha1_p_holm"] = float("nan")
    for partition, sub in coefs_df.groupby("partition"):
        idx = sub.index
        p_adj = holm_correction(sub["alpha1_p_value"].to_numpy(dtype=np.float64))
        coefs_df.loc[idx, "alpha1_p_holm"] = p_adj

    csv_path = output_dir / "per_band_coefficients.csv"
    coefs_df.to_csv(csv_path, index=False)

    evidence = _evidence_from_results(coefs_df)
    from model.vae_teb_prediction.testing.causal_te_validation.decision_rules import (
        verdict_test_03_band_uplift,
    )
    verdict = verdict_test_03_band_uplift(evidence)

    return {
        "verdict": verdict,
        "evidence": evidence,
        "csv_paths": [str(csv_path)],
        "figure_paths": [],
    }


__all__ = ["run"]
