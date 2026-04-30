"""Per-class + binary-by-underlying-class metric tests.

Validates ``add_perclass_clinical_columns``, ``compute_perclass_time_binned_metrics``,
and ``compute_binary_by_underlying_class`` against hand-rolled references on a
synthetic prediction DataFrame.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from model.vae_teb_prediction.new_classifier.guid_cls_v1.evaluate_3class_metrics import (  # noqa: E402
    add_perclass_clinical_columns,
    compute_binary_by_underlying_class,
    compute_perclass_time_binned_metrics,
)


def _make_df():
    """Build a synthetic per-position prediction DataFrame.

    Six GUIDs, each with one prediction at ``epoch = -3600 s`` (1h before
    delivery). Targets cover all three underlying classes so per-class
    metrics produce non-trivial values.
    """
    rows = []
    # (guid, target, predicted_class_3) tuples
    cases = [
        ("g1", 1, 0),  # healthy correct
        ("g2", 1, 1),  # healthy mispred → acidosis
        ("g3", 2, 1),  # acidosis correct
        ("g4", 2, 0),  # acidosis missed → healthy
        ("g5", 3, 2),  # hie correct
        ("g6", 3, 0),  # hie missed → healthy
    ]
    for guid, target, pred_3 in cases:
        prob_3 = [0.0, 0.0, 0.0]
        prob_3[pred_3] = 1.0
        prob_bin = float(pred_3 != 0)  # 1 if predicted unhealthy
        rows.append(
            {
                "guid": guid,
                "epoch": -3600.0,
                "target": target,
                "binary_target": int(target > 1),
                "predicted_class": int(prob_bin >= 0.5),
                "prob_class_0": 1.0 - prob_bin,
                "prob_class_1": prob_bin,
                "prob_healthy": prob_3[0],
                "prob_acidosis": prob_3[1],
                "prob_hie": prob_3[2],
                "predicted_class_3": pred_3,
                "position": 1,
                "prefix_length": 1,
                "cs_label": False,
                "bg_label": False,
                "tlo_hours": np.nan,
                "sso_hours": np.nan,
                "guid_binary_target": int(target > 1),
                "guid_class_3_target": target - 1,
                "clinical_pred": int(prob_bin >= 0.5),
            }
        )
    return pd.DataFrame(rows)


def test_add_perclass_clinical_columns_argmax_mode() -> None:
    """argmax mode: ``clinical_pred_<c> == (predicted_class_3 == c)``."""
    df = _make_df()
    out = add_perclass_clinical_columns(df, thresholds=None)
    # Healthy: predicted as 0 → 1 in two rows (g1, g4, g6)
    assert out["clinical_pred_healthy"].sum() == 3
    assert out["clinical_pred_acidosis"].sum() == 2  # g2, g3
    assert out["clinical_pred_hie"].sum() == 1       # g5
    # Binary targets per class (from `target == one_indexed_id`).
    assert out["binary_target_healthy"].sum() == 2   # g1, g2
    assert out["binary_target_acidosis"].sum() == 2  # g3, g4
    assert out["binary_target_hie"].sum() == 2       # g5, g6


def test_compute_perclass_metrics_match_hand_rolled() -> None:
    """Per-class sensitivity & FPR equal hand-rolled values in a single bin."""
    df = _make_df()
    df_pc = add_perclass_clinical_columns(df)
    # One bin covering 1.0±0.5 hours before delivery.
    bins = np.array([0.5, 1.5])

    # Acidosis class: positives = {g3, g4}. clinical_pred_acidosis is 1
    # for {g2, g3}. So TP = {g3} = 1, FN = {g4} = 1; sensitivity = 0.5.
    # Negatives = {g1, g2, g5, g6}. FP = {g2} = 1, TN = 3; FPR = 1/4 = 0.25.
    m_acid = compute_perclass_time_binned_metrics(
        df_pc, time_bins=bins, metric_type="instantaneous", class_name="acidosis"
    )
    assert len(m_acid) == 1
    assert pytest.approx(float(m_acid["sensitivity"].iloc[0])) == 0.5
    assert pytest.approx(float(m_acid["fpr"].iloc[0])) == 0.25

    # HIE: positives = {g5, g6}. clinical_pred_hie = {g5}. sens=0.5.
    # Negatives = {g1, g2, g3, g4}. FP=0 → FPR=0.
    m_hie = compute_perclass_time_binned_metrics(
        df_pc, time_bins=bins, metric_type="instantaneous", class_name="hie"
    )
    assert pytest.approx(float(m_hie["sensitivity"].iloc[0])) == 0.5
    assert pytest.approx(float(m_hie["fpr"].iloc[0])) == 0.0


def test_binary_by_underlying_class_filters_correctly() -> None:
    """Restrict to HEALTHY ∪ {acidosis} reduces dataset size and changes FPR."""
    df = _make_df()
    bins = np.array([0.5, 1.5])
    m_acid = compute_binary_by_underlying_class(
        df, time_bins=bins, metric_type="instantaneous", restrict_class="acidosis"
    )
    # Subset = {g1, g2, g3, g4}. Positives (binary_target=1) = {g3, g4} (acidosis).
    # In that subset, clinical_pred=1 corresponds to predicted_class>=1 i.e.
    # predicted_class_3 != 0 → {g2, g3}. TP = {g3} = 1. sens = 0.5.
    # Negatives = {g1, g2}. FP = {g2} = 1. FPR = 0.5.
    assert pytest.approx(float(m_acid["sensitivity"].iloc[0])) == 0.5
    assert pytest.approx(float(m_acid["fpr"].iloc[0])) == 0.5

    m_hie = compute_binary_by_underlying_class(
        df, time_bins=bins, metric_type="instantaneous", restrict_class="hie"
    )
    # Subset = {g1, g2, g5, g6}. Positives = {g5, g6}. clinical_pred=1 →
    # predicted_class_3 != 0 → {g2, g5}. TP = {g5} = 1. sens=0.5.
    # Negatives = {g1, g2}. FP = {g2}. FPR=0.5.
    assert pytest.approx(float(m_hie["sensitivity"].iloc[0])) == 0.5
    assert pytest.approx(float(m_hie["fpr"].iloc[0])) == 0.5
