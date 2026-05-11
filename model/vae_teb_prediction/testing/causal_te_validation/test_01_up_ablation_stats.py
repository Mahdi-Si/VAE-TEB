"""Test 1 — Statistical layer on top of `analyses/up_effect.py` outputs.

The existing UP perturbation analysis already computes per-sample
forecast-degradation, KLD-drop, uplift-drop, and residual-drop deltas
for the `{zero, batch_permute, time_shuffle}` conditions vs `normal`.
This module adds the *inferential* layer the spec calls for:

* Paired one-sided Wilcoxon signed-rank tests on each
  `condition $\\times$ metric` cell, with Holm correction across the
  full $3 \\times 4$ grid.
* GUID-cluster percentile bootstrap CIs ($B = 1000$) on the median
  delta.
* A pass / fail-mode-A/B/C verdict per the decision rules in
  ``decision_rules.verdict_test_01_up_ablation``.

Inputs (read-only, must already exist):

* ``<output>/up_effect/per_sample.csv``
* ``<output>/up_effect/condition_deltas.csv``

Outputs:

* ``<output>/causal_te_validation/up_ablation_stats/wilcoxon_results.csv``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from model.vae_teb_prediction.testing.causal_te_validation.statistics import (
    guid_bootstrap_ci,
    holm_correction,
    paired_wilcoxon,
)


# Metric -> human-readable label and target column name in
# ``up_effect/condition_deltas.csv``.  Positive deltas mean the normal
# UP stream improved the metric (lower error / higher KLD / higher
# residual usage / higher uplift) relative to the perturbation.
_METRICS: Dict[str, str] = {
    "forecast_degradation":  "feat_mse_total_delta_vs_normal",
    "kld_sum_drop":          "kld_sum_drop_from_normal",
    "uplift_rel_drop":       "uplift_rel_drop_from_normal",
    "residual_ratio_drop":   "residual_ratio_drop_from_normal",
}

_CONDITIONS: List[str] = ["zero", "batch_permute", "time_shuffle"]


def _evidence_from_results(results: pd.DataFrame) -> Dict[str, Any]:
    """Translate the Wilcoxon table into the ``evidence`` dict expected by
    :func:`decision_rules.verdict_test_01_up_ablation`.

    A condition $\\times$ metric cell counts as "passing" when its
    Holm-adjusted $p$-value is below 0.05 *and* its median delta is
    strictly positive.
    """
    if results.empty:
        return {
            "deltaE_pass": False, "deltaK_pass": False, "deltaR_pass": False,
            "n_conditions": 0, "n_metrics": 0,
        }

    def _all_three_conditions_pass(metric: str) -> bool:
        sub = results[results["metric"] == metric]
        if len(sub) < 3:
            return False
        return bool(
            ((sub["p_holm"] < 0.05) & (sub["median_delta"] > 0)).sum() >= 3
        )

    return {
        "deltaE_pass": _all_three_conditions_pass("forecast_degradation"),
        "deltaK_pass": _all_three_conditions_pass("kld_sum_drop"),
        "deltaR_pass": _all_three_conditions_pass("residual_ratio_drop"),
        "deltaUplift_pass": _all_three_conditions_pass("uplift_rel_drop"),
        "n_conditions": int(results["condition"].nunique()),
        "n_metrics": int(results["metric"].nunique()),
    }


def run(
    *,
    up_effect_dir: Path,
    output_dir: Path,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Post-process ``up_effect`` CSVs into a Holm-corrected statistics table.

    Args:
        up_effect_dir: Path to ``<output>/up_effect`` (Phase-1 output).
        output_dir: Path to ``<output>/causal_te_validation/up_ablation_stats``.
        n_boot: GUID-cluster bootstrap iterations for the median-delta CI.
        seed: RNG seed.

    Returns:
        Dict with ``verdict``, ``evidence``, ``csv_paths``,
        ``figure_paths``, ``n_pairs_total``, and ``n_guids_total``.
        Returns an ``error`` dict when the upstream CSV is missing.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    delta_csv = Path(up_effect_dir) / "condition_deltas.csv"
    if not delta_csv.is_file():
        return {
            "verdict": "missing",
            "evidence": {},
            "error": f"missing input: {delta_csv}",
            "csv_paths": [],
            "figure_paths": [],
        }
    df = pd.read_csv(delta_csv)
    if df.empty:
        return {
            "verdict": "missing",
            "evidence": {},
            "error": f"empty input: {delta_csv}",
            "csv_paths": [],
            "figure_paths": [],
        }

    rows: List[Dict[str, Any]] = []
    p_values: List[float] = []
    cell_index: List[int] = []  # row-id within ``rows`` for back-mapping p-values.

    for cond in _CONDITIONS:
        sub_cond = df[df["condition"] == cond]
        if sub_cond.empty:
            continue
        for metric_short, metric_col in _METRICS.items():
            if metric_col not in sub_cond.columns:
                continue
            sub_metric = sub_cond.dropna(subset=[metric_col])
            deltas = sub_metric[metric_col].to_numpy(dtype=np.float64)
            wilc = paired_wilcoxon(deltas, alternative="greater")
            ci_low, ci_high = guid_bootstrap_ci(
                deltas,
                guids=sub_metric.get("guid", pd.Series(["__"] * len(sub_metric))).to_numpy(),
                stat_fn=np.nanmedian,
                n_boot=int(n_boot),
                seed=int(seed),
            )
            n_pairs = int(wilc.get("n_pairs", 0))
            n_guids_metric = int(
                sub_metric["guid"].nunique() if "guid" in sub_metric.columns else 0
            )
            rows.append({
                "condition": cond,
                "metric": metric_short,
                "metric_col": metric_col,
                "n_pairs": n_pairs,
                "n_guids": n_guids_metric,
                "median_delta": float(wilc["median_delta"]),
                "frac_positive": float(wilc["frac_positive"]),
                "W": float(wilc["W"]),
                "p_value": float(wilc["p_value"]),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            })
            p_values.append(float(wilc["p_value"]))
            cell_index.append(len(rows) - 1)

    if not rows:
        return {
            "verdict": "missing",
            "evidence": {},
            "error": "no condition x metric cells produced statistics",
            "csv_paths": [],
            "figure_paths": [],
        }

    p_holm = holm_correction(p_values)
    for cell_id, p_adj in zip(cell_index, p_holm):
        rows[cell_id]["p_holm"] = float(p_adj)

    results = pd.DataFrame(rows)
    csv_path = output_dir / "wilcoxon_results.csv"
    results.to_csv(csv_path, index=False)

    evidence = _evidence_from_results(results)
    n_pairs_total = int(results["n_pairs"].sum())
    n_guids_total = int(results["n_guids"].max() if not results.empty else 0)

    from model.vae_teb_prediction.testing.causal_te_validation.decision_rules import (
        verdict_test_01_up_ablation,
    )
    verdict = verdict_test_01_up_ablation(evidence)

    return {
        "verdict": verdict,
        "evidence": evidence,
        "csv_paths": [str(csv_path)],
        "figure_paths": [],
        "n_pairs_total": n_pairs_total,
        "n_guids_total": n_guids_total,
    }


__all__ = ["run"]
