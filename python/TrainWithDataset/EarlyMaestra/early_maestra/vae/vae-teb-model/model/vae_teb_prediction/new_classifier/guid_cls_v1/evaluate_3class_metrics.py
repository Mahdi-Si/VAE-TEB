"""Per-class + binary-by-underlying-class evaluation pipeline.

The legacy time-binned utilities in
:mod:`model.vae_teb_prediction.new_classifier.evaluate_classifier`
(``apply_clinical_decision_rule``, ``compute_instantaneous_metrics``,
``compute_committed_cumulative_metrics``,
``compute_committed_overall_metrics``, etc.) are **column-generic**: they
read ``binary_target`` / ``clinical_pred`` / ``prob_class_1`` from the
prediction DataFrame.

This module extends evaluation to the 3-class case **without modifying the
legacy module**: we add per-class clinical columns to the DataFrame, then
either call the legacy time-binned metrics with renamed columns, or call
the legacy CDR with a swapped probability column. Three new families of
output are produced per fold:

1. **Per-class time-binned metrics** — for each class
   ``c ∈ {healthy, acidosis, hie}``, compute sensitivity_c / FPR_c /
   specificity_c vs time before delivery using the same three metric
   types (instantaneous / committed_cumulative / committed_overall).
2. **Per-class subgroup stratification** — same metrics × CS/BG subgroups.
3. **Binary by underlying class** — restrict the dataset to
   HEALTHY ∪ {ACIDOSIS} or HEALTHY ∪ {HIE} and recompute the binary
   metrics. Tells us whether the binary classifier detects both unhealthy
   types equally.

Class-id convention (matches the dataset's `target` field):
- ``target == 1`` → HEALTHY
- ``target == 2`` → ACIDOSIS
- ``target == 3`` → HIE

The corresponding probability columns are ``prob_healthy``,
``prob_acidosis``, ``prob_hie``. The 0-based class id (used by
``predicted_class_3``) maps as ``0 → healthy``, ``1 → acidosis``,
``2 → hie``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from loguru import logger


# Class names + their related (1-indexed) target id and the prob column.
CLASS_INFO: Tuple[Tuple[str, int, str], ...] = (
    ("healthy", 1, "prob_healthy"),
    ("acidosis", 2, "prob_acidosis"),
    ("hie", 3, "prob_hie"),
)

# Columns that ``fill_missing_epochs`` does NOT propagate into filled rows
# (it only knows about the binary-side columns). Forward-fill these per
# GUID before any int-cast or argmax-style consumer touches them.
_THREE_CLASS_FFILL_COLS: Tuple[str, ...] = (
    "predicted_class_3",
    "prob_healthy",
    "prob_acidosis",
    "prob_hie",
)
_THREE_CLASS_FFILL_DEFAULTS: Mapping[str, float] = {
    "predicted_class_3": 0.0,  # default to "healthy" prediction at leading NaN
    "prob_healthy": 1.0,
    "prob_acidosis": 0.0,
    "prob_hie": 0.0,
}


def _ffill_three_class_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill 3-class columns left NaN by ``fill_missing_epochs``.

    Background: the legacy ``fill_missing_epochs`` was written for the
    binary pipeline and only propagates ``prob_class_0/1``, ``model_pred``,
    ``clinical_pred``, ``binary_target``, ``target``. Any 3-class columns
    on filled rows therefore arrive as NaN, which breaks ``.astype(int)``
    in :func:`add_perclass_clinical_columns` (the actual error reported in
    eval logs: ``Cannot convert non-finite values (NA or inf) to integer``).

    This helper performs the equivalent forward-fill for the 3-class
    columns within each GUID, sorted ascending in epoch (matching the
    fill order used by the legacy fn). Leading NaN rows (a GUID whose
    first epoch was synthesised from scratch) are filled with healthy
    defaults so they line up with the binary-side default of
    ``clinical_pred=0`` at the first epoch.

    Args:
        df: DataFrame post-CDR / post-fill_missing_epochs.

    Returns:
        Same DataFrame with the four 3-class columns forward-filled and
        NaN-free. Returns ``df`` unchanged if none of the columns need
        filling.
    """
    present = [c for c in _THREE_CLASS_FFILL_COLS if c in df.columns]
    if not present:
        return df
    needs_fill = df[present].isna().any().any()
    if not needs_fill:
        return df
    df = df.sort_values(["guid", "epoch"], ascending=[True, True], kind="mergesort").copy()
    df[present] = df.groupby("guid", sort=False)[present].ffill()
    fill_map = {k: _THREE_CLASS_FFILL_DEFAULTS[k] for k in present}
    df[present] = df[present].fillna(value=fill_map)
    return df


def _drop_filled_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out rows synthesised by :func:`fill_missing_epochs`.

    The legacy gap-filler injects placeholder rows for missing epochs so
    the binary-side committed metrics see a continuous decision curve;
    those rows have ``target=1`` (healthy) by construction and a
    forward-filled ``predicted_class_3``. The per-class
    forward-fill-against-binary metric machinery (the legacy time-binned
    fns) explicitly excludes them via ``is_filled==False``, but the
    *aggregate* 3-class metrics (top-1 accuracy, macro-F1, Brier,
    confusion-matrix evolution) read ``target`` and ``predicted_class_3``
    directly and would otherwise count synthesised rows as ground-truth
    healthy — inflating top-1 accuracy and skewing per-class F1 / Brier.

    The flag column is only present when the upstream gap filler ran;
    if it's absent (e.g. instantaneous-mode evaluation, which skips the
    fill pass), this helper is a no-op.

    Args:
        df: A predictions DataFrame potentially carrying an
            ``is_filled`` boolean column (see
            :func:`clinical_metrics_utils.fill_missing_epochs`).

    Returns:
        ``df`` with synthesised rows removed (or ``df`` unchanged when
        no flag column exists).
    """
    if "is_filled" not in df.columns:
        return df
    return df.loc[~df["is_filled"].astype(bool)].copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _legacy_utils() -> Dict[str, Any]:
    """Late-import the in-pipeline clinical-metric utilities.

    All helpers live in the sibling ``clinical_metrics_utils`` module — a
    self-contained fork of the legacy ``new_classifier/evaluate_classifier.py``
    with all legacy-model orchestration removed. The function is named
    ``_legacy_utils`` for historical reasons; it does not actually reach into
    any legacy ``vae_teb_prediction.classifier`` code.
    """
    from model.vae_teb_prediction.new_classifier.guid_cls_v1.clinical_metrics_utils import (  # noqa: WPS433
        annotate_decision_time,
        apply_clinical_decision_rule,
        compute_committed_cumulative_metrics,
        compute_committed_overall_metrics,
        compute_instantaneous_metrics,
        create_enhanced_subgroup_filters,
        ensure_committed_epochs_filled,
        ensure_epoch_hours,
        find_threshold_for_committed_cumulative_fpr_at_1h,
        find_threshold_for_committed_overall_fpr_at_1h,
        find_threshold_for_instantaneous_fpr_at_1h,
        plot_single_metric_type,
    )

    return dict(
        annotate_decision_time=annotate_decision_time,
        apply_clinical_decision_rule=apply_clinical_decision_rule,
        compute_committed_cumulative_metrics=compute_committed_cumulative_metrics,
        compute_committed_overall_metrics=compute_committed_overall_metrics,
        compute_instantaneous_metrics=compute_instantaneous_metrics,
        create_enhanced_subgroup_filters=create_enhanced_subgroup_filters,
        ensure_committed_epochs_filled=ensure_committed_epochs_filled,
        ensure_epoch_hours=ensure_epoch_hours,
        find_threshold_for_committed_cumulative_fpr_at_1h=find_threshold_for_committed_cumulative_fpr_at_1h,
        find_threshold_for_committed_overall_fpr_at_1h=find_threshold_for_committed_overall_fpr_at_1h,
        find_threshold_for_instantaneous_fpr_at_1h=find_threshold_for_instantaneous_fpr_at_1h,
        plot_single_metric_type=plot_single_metric_type,
    )


METRIC_TYPES: Tuple[str, ...] = ("instantaneous", "committed_cumulative", "committed_overall")

# Axis-mode identifiers accepted by the public entry points.
AXIS_MODES: Tuple[str, ...] = ("delivery", "sso")


def _legacy_metric_fn(name: str, utils: Mapping[str, Any]) -> Callable[..., pd.DataFrame]:
    return {
        "instantaneous": utils["compute_instantaneous_metrics"],
        "committed_cumulative": utils["compute_committed_cumulative_metrics"],
        "committed_overall": utils["compute_committed_overall_metrics"],
    }[name]


def _sso_metric_fn(name: str) -> Callable[..., pd.DataFrame]:
    """Return the matching SSO-axis metric function.

    Late-imports :mod:`sso_metrics_utils` so this module remains usable
    in environments without matplotlib (used by some unit tests).
    """
    from model.vae_teb_prediction.new_classifier.guid_cls_v1 import (  # noqa: WPS433
        sso_metrics_utils,
    )

    return {
        "instantaneous": sso_metrics_utils.compute_instantaneous_metrics_sso,
        "committed_cumulative": sso_metrics_utils.compute_committed_cumulative_metrics_sso,
        "committed_overall": sso_metrics_utils.compute_committed_overall_metrics_sso,
    }[name]


def _metric_fn_for_axis(
    name: str, axis_mode: str, utils: Mapping[str, Any]
) -> Callable[..., pd.DataFrame]:
    """Dispatch between delivery-axis and SSO-axis metric implementations.

    Args:
        name: One of :data:`METRIC_TYPES`.
        axis_mode: ``'delivery'`` or ``'sso'``.
        utils: Legacy utils mapping (only consumed in delivery mode).

    Returns:
        The selected metric-computation function.
    """
    if axis_mode == "sso":
        return _sso_metric_fn(name)
    return _legacy_metric_fn(name, utils)


# ---------------------------------------------------------------------------
# Per-class clinical columns
# ---------------------------------------------------------------------------


def add_perclass_clinical_columns(
    df: pd.DataFrame,
    *,
    thresholds: Optional[Mapping[str, float]] = None,
    apply_cdr: bool = True,
) -> pd.DataFrame:
    """Add per-class ``clinical_pred_<class>`` and ``binary_target_<class>``.

    Two prediction modes:
      * ``thresholds is None`` (default, primary): argmax over ``prob_3``.
        ``clinical_pred_<c> = (predicted_class_3 == c).astype(int)`` —
        each row predicts at most one class.
      * Per-class OvR thresholds (secondary diagnostic): for each class
        ``c``, ``clinical_pred_<c> = (prob_<c> >= thresholds[c]).astype(int)``.

    Args:
        df: Predictions DataFrame from :func:`run_inference_per_position`.
            Must contain ``prob_healthy``, ``prob_acidosis``, ``prob_hie``,
            ``predicted_class_3``, ``target`` (1-indexed).
        thresholds: Optional mapping ``{class_name: threshold}``. When set,
            uses OvR-threshold mode and applies the legacy clinical decision
            rule (forward-fill within GUID after first detection) per class.
        apply_cdr: When ``True`` (default) and thresholds are provided,
            apply :func:`apply_clinical_decision_rule` per class so the
            forward-fill semantics match the binary-side pipeline. Has no
            effect in argmax mode (argmax doesn't have a CDR analogue
            because it's not threshold-based).

    Returns:
        Copy of ``df`` with added columns:
            ``clinical_pred_healthy/acidosis/hie``,
            ``binary_target_healthy/acidosis/hie``.
    """
    df = _ffill_three_class_columns(df.copy())
    target = df["target"].astype(int).to_numpy()

    if thresholds is None:
        # Argmax mode (primary 3-class reporting).
        pred_3 = df["predicted_class_3"].astype(int).to_numpy()
        for class_name, target_id, _ in CLASS_INFO:
            class_zero_indexed = target_id - 1
            df[f"clinical_pred_{class_name}"] = (pred_3 == class_zero_indexed).astype(int)
            df[f"binary_target_{class_name}"] = (target == target_id).astype(int)
        return df

    # OvR-threshold mode (secondary). Forward-fill via the legacy CDR.
    utils = _legacy_utils() if apply_cdr else None
    for class_name, target_id, prob_col in CLASS_INFO:
        df[f"binary_target_{class_name}"] = (target == target_id).astype(int)
        thr = float(thresholds[class_name])
        df[f"_model_pred_{class_name}"] = (df[prob_col] >= thr).astype(int)
        if utils is not None:
            # Reuse the legacy CDR by swapping the probability column
            # and binary target it reads. The CDR overwrites
            # ``clinical_pred`` from ``model_pred = (prob_class_1 >=
            # threshold)`` and forward-fills per GUID, so the per-class
            # decision uses the same flag-and-stay semantics as the
            # binary head — preserving binary parity by construction.
            swap = df.copy()
            swap["prob_class_1"] = swap[prob_col]
            swap["binary_target"] = swap[f"binary_target_{class_name}"]
            swap = utils["apply_clinical_decision_rule"](swap, threshold=thr, verify=False)
            df[f"clinical_pred_{class_name}"] = swap["clinical_pred"].to_numpy()
        else:
            df[f"clinical_pred_{class_name}"] = df[f"_model_pred_{class_name}"]
        df = df.drop(columns=[f"_model_pred_{class_name}"])
    return df


# ---------------------------------------------------------------------------
# Per-class time-binned metrics
# ---------------------------------------------------------------------------


def compute_perclass_time_binned_metrics(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    metric_type: str,
    class_name: str,
    *,
    subgroup_filter: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
    axis_mode: str = "delivery",
) -> pd.DataFrame:
    """Per-class wrapper around the legacy time-binned binary metrics.

    Renames ``clinical_pred_<class>`` → ``clinical_pred`` and
    ``binary_target_<class>`` → ``binary_target`` in a temporary view of
    ``df`` and calls the binary metric fn matching ``axis_mode``.
    Returns the same columns the underlying fn returns
    (``bin_center``, ``sensitivity``, ``fpr``, ``specificity``, ``n_*``).

    Args:
        df: DataFrame from :func:`add_perclass_clinical_columns`.
        time_bins: Bin edges. Units are hours-before-delivery when
            ``axis_mode='delivery'`` and signed hours-from-SSO when
            ``axis_mode='sso'``.
        metric_type: One of ``METRIC_TYPES``.
        class_name: One of ``healthy``, ``acidosis``, ``hie``.
        subgroup_filter: Optional row-level boolean mask producer.
        axis_mode: ``'delivery'`` (default, uses ``epoch_hours``) or
            ``'sso'`` (uses ``t_rel_sso_hours``).

    Returns:
        Per-class metrics DataFrame; the columns match the legacy
        binary-side schema so :func:`plot_single_metric_type` (or
        :func:`sso_metrics_utils.plot_single_metric_type_sso`) can
        render the result without modification.
    """
    if metric_type not in METRIC_TYPES:
        raise ValueError(f"Unknown metric_type {metric_type!r}; expected one of {METRIC_TYPES}")
    if axis_mode not in AXIS_MODES:
        raise ValueError(f"Unknown axis_mode {axis_mode!r}; expected one of {AXIS_MODES}")
    utils = _legacy_utils()
    pred_col = f"clinical_pred_{class_name}"
    bt_col = f"binary_target_{class_name}"
    if pred_col not in df.columns or bt_col not in df.columns:
        raise KeyError(
            f"Missing per-class columns ({pred_col!r}, {bt_col!r}) — "
            "call add_perclass_clinical_columns first."
        )

    swap = df.copy()
    swap["clinical_pred"] = swap[pred_col]
    swap["binary_target"] = swap[bt_col]
    fn = _metric_fn_for_axis(metric_type, axis_mode, utils)
    return fn(swap, time_bins, subgroup_filter)


# ---------------------------------------------------------------------------
# Binary metrics restricted to one underlying unhealthy class
# ---------------------------------------------------------------------------


def compute_binary_by_underlying_class(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    metric_type: str,
    restrict_class: str,
    *,
    subgroup_filter: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
    axis_mode: str = "delivery",
) -> pd.DataFrame:
    """Binary metric on the subset HEALTHY ∪ {restrict_class}.

    Tells us whether the binary classifier detects (e.g.) ACIDOSIS as
    well as it detects HIE — a clinically critical question that the
    pooled binary metric obscures.

    Args:
        df: Full predictions DataFrame (must have ``target``,
            ``binary_target``, ``clinical_pred`` set by the CDR).
        time_bins: Bin edges. Units depend on ``axis_mode``.
        metric_type: One of ``METRIC_TYPES``.
        restrict_class: ``"acidosis"`` or ``"hie"`` — the unhealthy class
            to keep alongside HEALTHY.
        subgroup_filter: Optional row-level boolean mask producer.
        axis_mode: Selects the underlying metric function via
            :func:`_metric_fn_for_axis`.

    Returns:
        DataFrame with the same shape as the legacy binary metrics.
    """
    if restrict_class not in {"acidosis", "hie"}:
        raise ValueError(
            f"restrict_class must be 'acidosis' or 'hie', got {restrict_class!r}"
        )
    target_id = {name: tid for name, tid, _ in CLASS_INFO}[restrict_class]
    keep = df[(df["target"] == 1) | (df["target"] == target_id)].copy()
    if keep.empty:
        return pd.DataFrame()
    utils = _legacy_utils()
    fn = _metric_fn_for_axis(metric_type, axis_mode, utils)
    return fn(keep, time_bins, subgroup_filter)


# ---------------------------------------------------------------------------
# Per-class threshold search (secondary diagnostic)
# ---------------------------------------------------------------------------


_PERCLASS_THRESHOLD_FN_KEY: Mapping[str, str] = {
    "instantaneous": "find_threshold_for_instantaneous_fpr_at_1h",
    "committed_cumulative": "find_threshold_for_committed_cumulative_fpr_at_1h",
    "committed_overall": "find_threshold_for_committed_overall_fpr_at_1h",
}


def find_perclass_threshold_at_target_fpr(
    df_val: pd.DataFrame,
    class_name: str,
    *,
    target_fpr: float = 0.2,
    decision_time_hours: float = 1.0,
    metric_type: str = "committed_overall",
    max_gap_multiplier: Optional[float] = None,
    fallback_tolerance_hours: float = 0.5,
) -> Tuple[float, Dict[str, Any]]:
    """OvR-threshold search per class, mode-aware.

    Mirrors the binary head's per-mode threshold discipline: each mode
    (``instantaneous`` / ``committed_cumulative`` / ``committed_overall``)
    has its own legacy threshold-search fn, and we dispatch to the
    matching one. The legacy fn reads ``prob_class_1`` and
    ``binary_target``, so we temporarily swap in the per-class
    probability column and per-class binary target before calling it.

    This is the primary path for selecting per-class operating points
    when ``evaluation.three_class.threshold_search_default`` is true.

    Args:
        df_val: Validation prediction rows (output of
            :func:`run_inference_per_position`).
        class_name: One of ``"healthy"``, ``"acidosis"``, ``"hie"``.
        target_fpr: Target false-positive rate at the decision time.
        decision_time_hours: Hours before delivery at which to evaluate
            (passed to the legacy fn as ``time_window_hours``).
        metric_type: One of :data:`METRIC_TYPES`. Selects which legacy
            threshold-search fn to call so the per-class threshold is
            optimised under the same metric-type semantics that will
            later score the test fold.
        max_gap_multiplier: Optional gap-fill multiplier (legacy default).
        fallback_tolerance_hours: Decision-time fallback window
            (legacy default).

    Returns:
        Tuple ``(threshold, info_dict)``. ``info_dict`` is normalised to
        a fixed cross-mode schema:
        ``{metric_type, class_name, target_fpr, target_time_hours,
        actual_time_hours, fpr, sensitivity, specificity, n_positive,
        n_negative, _extra}``. Mode-specific keys (e.g.
        ``is_primary_metric`` from the committed-overall fn,
        ``n_positive_available`` from cumulative) are preserved under
        ``_extra`` for diagnostic use.
    """
    if metric_type not in _PERCLASS_THRESHOLD_FN_KEY:
        raise ValueError(
            f"Unknown metric_type {metric_type!r}; expected one of "
            f"{tuple(_PERCLASS_THRESHOLD_FN_KEY)}"
        )
    utils = _legacy_utils()
    target_id = {name: tid for name, tid, _ in CLASS_INFO}[class_name]
    prob_col = {name: pc for name, _, pc in CLASS_INFO}[class_name]

    swap = df_val.copy()
    swap["prob_class_1"] = swap[prob_col]
    swap["binary_target"] = (swap["target"] == target_id).astype(int)
    fn = utils[_PERCLASS_THRESHOLD_FN_KEY[metric_type]]
    threshold, raw_info = fn(
        swap,
        target_fpr=target_fpr,
        time_window_hours=decision_time_hours,
        max_gap_multiplier=max_gap_multiplier,
        fallback_tolerance_hours=fallback_tolerance_hours,
    )

    # The three legacy fns return ``info`` dicts with mode-specific keys
    # (``n_positive`` vs ``n_positive_available`` vs ``n_positive_total``,
    # plus ``is_primary_metric`` only on the overall mode). Normalise to
    # a single fixed schema so downstream consumers — ``aggregate_results``
    # and the per-fold ``perclass_thresholds_<mode>.json`` writer — can
    # rely on a stable layout. Mode-specific extras are preserved under
    # ``_extra`` for diagnostic deep-dives.
    info: Dict[str, Any] = {
        "metric_type": str(metric_type),
        "class_name": str(class_name),
        "target_fpr": float(target_fpr),
        "target_time_hours": float(decision_time_hours),
        "actual_time_hours": float(
            raw_info.get("actual_time_hours", decision_time_hours)
        ),
        "fpr": float(raw_info.get("fpr", float("nan"))),
        "sensitivity": float(raw_info.get("sensitivity", float("nan"))),
        "specificity": float(raw_info.get("specificity", float("nan"))),
        "n_positive": int(
            raw_info.get(
                "n_positive",
                raw_info.get(
                    "n_positive_available", raw_info.get("n_positive_total", 0)
                ),
            )
            or 0
        ),
        "n_negative": int(
            raw_info.get(
                "n_negative",
                raw_info.get(
                    "n_negative_available", raw_info.get("n_negative_total", 0)
                ),
            )
            or 0
        ),
    }
    extras_seen = {"actual_time_hours", "fpr", "sensitivity", "specificity"}
    extras = {k: v for k, v in raw_info.items() if k not in extras_seen}
    if extras:
        info["_extra"] = extras
    return float(threshold), info


# ---------------------------------------------------------------------------
# Plot helpers — reuse legacy plotters where possible
# ---------------------------------------------------------------------------


_CLASS_PALETTE: Mapping[str, str] = {
    "healthy": "#27ae60",
    "acidosis": "#f39c12",
    "hie": "#c0392b",
}


def _annotate_decision_time(ax, decision_time_hours: Optional[float]) -> None:
    """Re-export the shared decision-time annotation helper.

    Wrapped here so this module doesn't import the helper at import time
    (the legacy utils dispatcher is already lazy).
    """
    utils = _legacy_utils()
    annotate = utils.get("annotate_decision_time")
    if annotate is None:
        return
    annotate(ax, decision_time_hours=decision_time_hours)


def collect_perclass_long_format(
    metrics_per_class: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Stack per-class metric DataFrames into one long-format frame.

    Output columns: ``[bin_center, class, sensitivity, specificity, fpr,
    n_positive, n_negative, ...]`` (any extra columns produced by the
    legacy time-binned fns are passed through).

    Args:
        metrics_per_class: Mapping ``{class_name: metrics_df}`` (legacy
            wide-format per-class output).

    Returns:
        A long-format DataFrame; one row per (bin, class).
    """
    frames = []
    for class_name, metrics_df in metrics_per_class.items():
        if metrics_df is None or metrics_df.empty:
            continue
        m = metrics_df.copy()
        m.insert(0, "class", class_name)
        frames.append(m)
    if not frames:
        return pd.DataFrame(columns=["bin_center", "class"])
    return pd.concat(frames, ignore_index=True)


def _axis_style(axis_mode: str) -> Tuple[str, bool]:
    """Return (x_label, invert_x) for a given axis mode."""
    if axis_mode == "sso":
        return "Hours from second stage onset", False
    return "Hours Before Birth", True


def _maybe_add_sso_zero(ax, axis_mode: str) -> None:
    """Draw the SSO reference line when plotting against the SSO axis."""
    if axis_mode != "sso":
        return
    xlim = ax.get_xlim()
    lo, hi = min(xlim), max(xlim)
    if 0.0 < lo or 0.0 > hi:
        return
    ax.axvline(x=0.0, color="0.35", linestyle=":", linewidth=1.1, zorder=0)


def _maybe_add_dropped_footer(fig, axis_mode: str, n_dropped_guids: int) -> None:
    """Footer annotation reporting GUIDs dropped for the SSO eval."""
    if axis_mode != "sso" or n_dropped_guids <= 0:
        return
    fig.text(
        0.995, 0.005,
        f"Dropped {int(n_dropped_guids)} GUIDs (no second-stage onset)",
        fontsize=8, color="0.35", ha="right", va="bottom",
    )


def plot_perclass_panel_combined(
    metrics_per_class: Mapping[str, pd.DataFrame],
    *,
    output_path: Path,
    metric_type: str,
    decision_time_hours: Optional[float] = None,
    title_suffix: str = "",
    axis_mode: str = "delivery",
    n_dropped_guids: int = 0,
) -> None:
    """One PNG with three side-by-side panels (one per class).

    Each panel shows sensitivity, specificity, and FPR vs the chosen
    time axis (delivery-anchored hours before birth, or signed hours
    from SSO).

    Args:
        metrics_per_class: ``{class_name: per_class_metrics_df}``.
        output_path: Output PNG path.
        metric_type: Drives the panel title.
        decision_time_hours: When provided and ``axis_mode='delivery'``,
            draws a dashed vertical reference line at that x value.
        title_suffix: Extra suffix appended to the figure suptitle.
        axis_mode: ``'delivery'`` (default, inverted x-axis labelled
            "Hours Before Birth") or ``'sso'`` (natural x-axis labelled
            "Hours from second stage onset"; vertical zero-line marks
            SSO; figure footer reports dropped GUIDs).
        n_dropped_guids: SSO-mode footer count. Ignored in delivery mode.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    metric_label = {
        "instantaneous": "Instantaneous Decisions",
        "committed_cumulative": "Committed (Cumulative)",
        "committed_overall": "Committed (Overall)",
    }.get(metric_type, metric_type)
    x_label, invert_x = _axis_style(axis_mode)
    for ax, (class_name, _, _) in zip(axes, CLASS_INFO):
        metrics_df = metrics_per_class.get(class_name)
        if metrics_df is None or metrics_df.empty:
            ax.text(0.5, 0.5, f"{class_name}: no data", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        m = metrics_df.dropna(subset=["sensitivity", "specificity", "fpr"], how="all")
        # In SSO mode read left-to-right (ascending); in delivery mode the
        # subsequent ``invert_xaxis()`` displays high tau on the left so we
        # sort descending to match the legacy panel render order.
        m = m.sort_values("bin_center", ascending=(axis_mode == "sso"))
        x = m["bin_center"].to_numpy()
        if "sensitivity" in m.columns:
            ax.plot(x, m["sensitivity"], marker="o", linewidth=2.2,
                    label="Sensitivity", color="#2ecc71")
        if "specificity" in m.columns:
            ax.plot(x, m["specificity"], marker="s", linewidth=2.2,
                    label="Specificity", color="#3498db")
        if "fpr" in m.columns:
            ax.plot(x, m["fpr"], marker="^", linewidth=2.2,
                    label="FPR", color="#e74c3c")
        ax.set_title(class_name, color=_CLASS_PALETTE.get(class_name, "black"),
                     fontweight="bold")
        ax.set_xlabel(x_label)
        ax.set_ylim([0, 1.05])
        if invert_x:
            ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")
        if axis_mode == "delivery":
            _annotate_decision_time(ax, decision_time_hours)
        _maybe_add_sso_zero(ax, axis_mode)
    axes[0].set_ylabel("Metric value")
    suptitle = f"Per-class metrics vs time — {metric_label}"
    if axis_mode == "sso":
        suptitle += " (axis: SSO)"
    if title_suffix:
        suptitle += f" ({title_suffix})"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout()
    _maybe_add_dropped_footer(fig, axis_mode, n_dropped_guids)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {output_path.parent.name}/{output_path.name}")


def compute_perclass_subgroup_long_format(
    df: pd.DataFrame,
    *,
    time_bins: np.ndarray,
    metric_type: str,
    axis_mode: str = "delivery",
) -> pd.DataFrame:
    """Build a long-format DataFrame over (bin × class × subgroup).

    Replaces the per-fold tree of 12 (class × subgroup) ``metrics.csv``
    files with a single tidy table.

    Args:
        df: DataFrame from :func:`add_perclass_clinical_columns`.
        time_bins: Bin edges (hours-before-delivery for
            ``axis_mode='delivery'``; signed hours-from-SSO for
            ``axis_mode='sso'``).
        metric_type: One of ``METRIC_TYPES``.
        axis_mode: Selects the underlying metric function — see
            :func:`compute_perclass_time_binned_metrics`.

    Returns:
        DataFrame with columns
        ``[bin_center, class, subgroup, sensitivity, specificity, fpr,
        n_positive, n_negative]`` plus any extras from the legacy
        time-binned function.
    """
    utils = _legacy_utils()
    filters = utils["create_enhanced_subgroup_filters"]()
    frames = []
    for subgroup_name, filter_fn in filters.items():
        for class_name, _, _ in CLASS_INFO:
            try:
                metrics_df = compute_perclass_time_binned_metrics(
                    df,
                    time_bins=time_bins,
                    metric_type=metric_type,
                    class_name=class_name,
                    subgroup_filter=filter_fn,
                    axis_mode=axis_mode,
                )
            except Exception:  # noqa: BLE001
                continue
            if metrics_df is None or metrics_df.empty:
                continue
            m = metrics_df.copy()
            m.insert(0, "subgroup", subgroup_name)
            m.insert(0, "class", class_name)
            frames.append(m)
    if not frames:
        return pd.DataFrame(columns=["bin_center", "class", "subgroup"])
    return pd.concat(frames, ignore_index=True)


def plot_perclass_subgroup_long(
    long_df: pd.DataFrame,
    *,
    output_path: Path,
    metric_type: str,
    decision_time_hours: Optional[float] = None,
    axis_mode: str = "delivery",
    n_dropped_guids: int = 0,
) -> None:
    """Render the per-class × subgroup long-format DataFrame as one PNG.

    Layout: 3 columns (one per class), N rows (one per subgroup); each
    cell plots SE / SP / FPR vs the chosen time axis (delivery hours
    before birth or signed hours from SSO).

    Args:
        long_df: Output of
            :func:`compute_perclass_subgroup_long_format`.
        output_path: Output PNG path.
        metric_type: Display metric label.
        decision_time_hours: Vertical reference line drawn at this x
            value when ``axis_mode='delivery'``. Ignored in SSO mode.
        axis_mode: ``'delivery'`` or ``'sso'``.
        n_dropped_guids: SSO-mode footer count.
    """
    if long_df is None or long_df.empty:
        logger.warning(
            f"plot_perclass_subgroup_long: empty long_df, skipping {output_path.name}"
        )
        return
    subgroups = sorted(long_df["subgroup"].unique().tolist())
    n_rows = len(subgroups)
    if n_rows == 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(15, 3.4 * n_rows + 1),
        sharex=True, sharey=True,
        squeeze=False,
    )
    metric_label = {
        "instantaneous": "Instantaneous",
        "committed_cumulative": "Committed (Cumulative)",
        "committed_overall": "Committed (Overall)",
    }.get(metric_type, metric_type)
    x_label, invert_x = _axis_style(axis_mode)
    for r, subgroup_name in enumerate(subgroups):
        for c, (class_name, _, _) in enumerate(CLASS_INFO):
            ax = axes[r, c]
            sub = long_df[
                (long_df["subgroup"] == subgroup_name)
                & (long_df["class"] == class_name)
            ]
            if sub.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            sub = sub.sort_values("bin_center", ascending=(axis_mode == "sso"))
            x = sub["bin_center"].to_numpy()
            if "sensitivity" in sub.columns:
                ax.plot(x, sub["sensitivity"], marker="o", linewidth=2.0,
                        markersize=4, label="SE", color="#2ecc71")
            if "specificity" in sub.columns:
                ax.plot(x, sub["specificity"], marker="s", linewidth=2.0,
                        markersize=4, label="SP", color="#3498db")
            if "fpr" in sub.columns:
                ax.plot(x, sub["fpr"], marker="^", linewidth=2.0,
                        markersize=4, label="FPR", color="#e74c3c")
            ax.set_ylim([0, 1.05])
            if invert_x:
                ax.invert_xaxis()
            ax.grid(True, alpha=0.3)
            if r == 0:
                ax.set_title(class_name,
                             color=_CLASS_PALETTE.get(class_name, "black"),
                             fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{subgroup_name}\nMetric value", fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel(x_label)
            if r == 0 and c == 2:
                ax.legend(fontsize=8, loc="best")
            if axis_mode == "delivery":
                _annotate_decision_time(ax, decision_time_hours)
            _maybe_add_sso_zero(ax, axis_mode)
    suptitle = f"Per-class × subgroup metrics vs time — {metric_label}"
    if axis_mode == "sso":
        suptitle += " (axis: SSO)"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout()
    _maybe_add_dropped_footer(fig, axis_mode, n_dropped_guids)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {output_path.parent.name}/{output_path.name}")


def compute_binary_by_underlying_class_long_format(
    df: pd.DataFrame,
    *,
    time_bins: np.ndarray,
    metric_type: str,
    axis_mode: str = "delivery",
) -> pd.DataFrame:
    """Stack the binary-by-underlying-class metrics into one long-format frame.

    Output columns: ``[bin_center, restrict_class, sensitivity,
    specificity, fpr, ...]``.

    Args:
        df: Predictions DataFrame.
        time_bins: Bin edges (units depend on ``axis_mode``).
        metric_type: One of ``METRIC_TYPES``.
        axis_mode: ``'delivery'`` or ``'sso'``.
    """
    frames = []
    for restrict_class in ("acidosis", "hie"):
        m = compute_binary_by_underlying_class(
            df, time_bins=time_bins, metric_type=metric_type,
            restrict_class=restrict_class,
            axis_mode=axis_mode,
        )
        if m is None or m.empty:
            continue
        out = m.copy()
        out.insert(0, "restrict_class", restrict_class)
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["bin_center", "restrict_class"])
    return pd.concat(frames, ignore_index=True)


def plot_binary_by_underlying_class_long(
    long_df: pd.DataFrame,
    *,
    output_path: Path,
    metric_type: str,
    decision_time_hours: Optional[float] = None,
    axis_mode: str = "delivery",
    n_dropped_guids: int = 0,
) -> None:
    """Render the binary-by-underlying-class long-format DataFrame as one PNG.

    Two side-by-side panels (acidosis-only, hie-only); each shows SE/SP/FPR
    vs the chosen time axis.

    Args:
        long_df: Output of
            :func:`compute_binary_by_underlying_class_long_format`.
        output_path: PNG path.
        metric_type: Display metric label.
        decision_time_hours: Vertical reference line in delivery mode.
        axis_mode: ``'delivery'`` or ``'sso'``.
        n_dropped_guids: SSO-mode footer count.
    """
    if long_df is None or long_df.empty:
        logger.warning(
            f"plot_binary_by_underlying_class_long: empty long_df, "
            f"skipping {output_path.name}"
        )
        return
    classes = ["acidosis", "hie"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    metric_label = {
        "instantaneous": "Instantaneous",
        "committed_cumulative": "Committed (Cumulative)",
        "committed_overall": "Committed (Overall)",
    }.get(metric_type, metric_type)
    x_label, invert_x = _axis_style(axis_mode)
    for ax, restrict_class in zip(axes, classes):
        sub = long_df[long_df["restrict_class"] == restrict_class]
        if sub.empty:
            ax.text(0.5, 0.5, f"only_{restrict_class}: no data",
                    ha="center", va="center")
            continue
        sub = sub.sort_values("bin_center", ascending=(axis_mode == "sso"))
        x = sub["bin_center"].to_numpy()
        if "sensitivity" in sub.columns:
            ax.plot(x, sub["sensitivity"], marker="o", linewidth=2.2,
                    label="Sensitivity", color="#2ecc71")
        if "specificity" in sub.columns:
            ax.plot(x, sub["specificity"], marker="s", linewidth=2.2,
                    label="Specificity", color="#3498db")
        if "fpr" in sub.columns:
            ax.plot(x, sub["fpr"], marker="^", linewidth=2.2,
                    label="FPR", color="#e74c3c")
        ax.set_xlabel(x_label)
        ax.set_title(f"healthy vs {restrict_class}",
                     color=_CLASS_PALETTE.get(restrict_class, "black"),
                     fontweight="bold")
        ax.set_ylim([0, 1.05])
        if invert_x:
            ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="best")
        if axis_mode == "delivery":
            _annotate_decision_time(ax, decision_time_hours)
        _maybe_add_sso_zero(ax, axis_mode)
    axes[0].set_ylabel("Metric value")
    suptitle = f"Binary head — restricted to one underlying class — {metric_label}"
    if axis_mode == "sso":
        suptitle += " (axis: SSO)"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout()
    _maybe_add_dropped_footer(fig, axis_mode, n_dropped_guids)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {output_path.parent.name}/{output_path.name}")


# ---------------------------------------------------------------------------
# Per-class AUROC vs time (one-vs-rest)
# ---------------------------------------------------------------------------


def compute_perclass_auroc_vs_time(
    df: pd.DataFrame,
    *,
    time_bins: np.ndarray,
) -> pd.DataFrame:
    """One-vs-rest AUROC per class per time bin.

    For each ``(bin, class c)``: take the most-recent prefix per GUID
    that falls in the bin, then compute AUROC against the binary
    indicator ``guid_class_3_target == c``. Bins where ``y_true`` is
    all-positive or all-negative produce NaN.

    Args:
        df: Predictions DataFrame with ``guid``, ``epoch_hours``,
            ``prob_{healthy,acidosis,hie}``, ``guid_class_3_target``,
            and a sortable position column (``prefix_length`` or
            ``position``).
        time_bins: Bin edges in hours-before-delivery.

    Returns:
        Long-format DataFrame ``[bin_center, class, auroc, n_pos, n_neg, n]``.
    """
    from sklearn.metrics import roc_auc_score  # noqa: WPS433

    if "epoch_hours" not in df.columns:
        raise KeyError("df must have 'epoch_hours' (run ensure_epoch_hours first)")
    if "guid_class_3_target" not in df.columns:
        raise KeyError("df must have 'guid_class_3_target'")

    sort_col = "prefix_length" if "prefix_length" in df.columns else "position"
    df_sorted = df.sort_values(["guid", sort_col]).copy()
    centers = _bin_centers(time_bins)
    rows = []
    epoch_arr = df_sorted["epoch_hours"].astype(float).to_numpy()
    bin_idx = _bin_assign(epoch_arr, time_bins)
    df_sorted["_bin_idx"] = bin_idx
    for b, center in enumerate(centers):
        sub = df_sorted[df_sorted["_bin_idx"] == b]
        if sub.empty:
            for class_name, _, _ in CLASS_INFO:
                rows.append({
                    "bin_center": float(center),
                    "class": class_name,
                    "auroc": float("nan"),
                    "n_pos": 0,
                    "n_neg": 0,
                    "n": 0,
                })
            continue
        last = sub.groupby("guid", as_index=False).tail(1)
        target = last["guid_class_3_target"].astype(int).to_numpy()
        n = int(len(last))
        for class_name, target_id, prob_col in CLASS_INFO:
            zero_based_id = target_id - 1
            y = (target == zero_based_id).astype(int)
            n_pos = int(y.sum())
            n_neg = int(n - n_pos)
            if prob_col not in last.columns or n_pos == 0 or n_neg == 0:
                rows.append({
                    "bin_center": float(center),
                    "class": class_name,
                    "auroc": float("nan"),
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "n": n,
                })
                continue
            scores = last[prob_col].astype(float).to_numpy()
            valid = ~np.isnan(scores)
            if valid.sum() < 2 or y[valid].sum() == 0 or y[valid].sum() == valid.sum():
                rows.append({
                    "bin_center": float(center),
                    "class": class_name,
                    "auroc": float("nan"),
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "n": n,
                })
                continue
            try:
                auroc = float(roc_auc_score(y[valid], scores[valid]))
            except ValueError:
                auroc = float("nan")
            rows.append({
                "bin_center": float(center),
                "class": class_name,
                "auroc": auroc,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n": n,
            })
    return pd.DataFrame(rows)


def plot_perclass_auroc_vs_time(
    auroc_df: pd.DataFrame,
    *,
    output_path: Path,
    metric_type: str,
    decision_time_hours: Optional[float] = None,
) -> None:
    """Plot one line per class on a single axes; AUROC vs hours before birth."""
    if auroc_df is None or auroc_df.empty:
        logger.warning(
            f"plot_perclass_auroc_vs_time: empty df, skipping {output_path.name}"
        )
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for class_name, _, _ in CLASS_INFO:
        sub = auroc_df[auroc_df["class"] == class_name].sort_values(
            "bin_center", ascending=False
        )
        if sub.empty:
            continue
        ax.plot(
            sub["bin_center"].to_numpy(),
            sub["auroc"].to_numpy(),
            marker="o",
            linewidth=2.2,
            label=class_name,
            color=_CLASS_PALETTE.get(class_name, None),
        )
    ax.set_xlabel("Hours Before Birth", fontsize=13)
    ax.set_ylabel("AUROC (one-vs-rest)", fontsize=13)
    metric_label = {
        "instantaneous": "Instantaneous",
        "committed_cumulative": "Committed (Cumulative)",
        "committed_overall": "Committed (Overall)",
    }.get(metric_type, metric_type)
    ax.set_title(f"Per-class AUROC vs time — {metric_label}",
                 fontsize=14, fontweight="bold")
    ax.set_ylim([0.4, 1.02])
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1.0, label="chance")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    _annotate_decision_time(ax, decision_time_hours)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {output_path.parent.name}/{output_path.name}")


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_3class_evaluation_for_metric_type(
    df: pd.DataFrame,
    *,
    time_bins: np.ndarray,
    metric_type: str,
    eval_root: Path,
    thresholds: Optional[Mapping[str, float]] = None,
    df_val: Optional[pd.DataFrame] = None,
    target_fpr: float = 0.2,
    decision_time_hours: float = 1.0,
    threshold_search_kwargs: Optional[Mapping[str, Any]] = None,
    axis_mode: str = "delivery",
    n_dropped_guids: int = 0,
) -> Dict[str, Any]:
    """Compute and persist all per-class artefacts for one metric type.

    Output layout (per fold, written under ``eval_root``):

    * ``multiclass_head/per_class_vs_time/<mode>.csv`` — long-format
      ``(bin_center, class, sensitivity, specificity, fpr, ...)``.
    * ``multiclass_head/per_class_vs_time/<mode>_panel.png`` —
      combined 3-panel figure (one panel per class).
    * ``multiclass_head/per_class_subgroups_vs_time/<mode>.csv`` —
      long-format ``(bin_center, class, subgroup, ...)``.
    * ``multiclass_head/per_class_subgroups_vs_time/<mode>.png`` —
      grid of subgroup × class panels.
    * ``multiclass_head/auroc_vs_time/<mode>.{csv,png}`` — per-class
      one-vs-rest AUROC vs time.
    * ``multiclass_head/aggregate_vs_time/<mode>_top1_acc.{csv,png}``,
      ``..._macro_f1.{csv,png}``, ``..._brier_perclass.{csv,png}``.
    * ``multiclass_head/confusion_evolution/<mode>.png``.
    * ``binary_head/by_underlying_class_vs_time/<mode>.csv`` — long-format
      ``(bin_center, restrict_class, ...)``.
    * ``binary_head/by_underlying_class_vs_time/<mode>.png`` — combined plot.

    Per-class operating points for the ``clinical_pred_<class>`` columns
    are resolved in this order (highest priority first): explicit
    ``thresholds`` argument; per-class OvR threshold search on ``df_val``
    at ``target_fpr`` / ``decision_time_hours`` for this ``metric_type``;
    fallback argmax (``predicted_class_3 == c``).

    Note: argmax-derived diagnostics (top-1 accuracy, macro-F1, confusion
    evolution) read ``predicted_class_3`` directly so they're unaffected
    by which path produced the per-class clinical columns.

    Args:
        df: Predictions DataFrame after the legacy CDR has been applied.
        time_bins: Bin edges (legacy ``compute_time_bins``).
        metric_type: One of ``METRIC_TYPES``.
        eval_root: Per-fold ``evaluation/`` directory. ``binary_head/``
            and ``multiclass_head/`` are created beneath it.
        thresholds: Optional per-class OvR thresholds. When provided,
            short-circuits the search.
        df_val: Optional validation predictions DataFrame. When provided
            and ``thresholds is None``, the per-class threshold is
            searched here.
        target_fpr: Target FPR for the per-class threshold search.
        decision_time_hours: Hours before delivery at which thresholds
            are evaluated; also drawn as a vertical reference line on
            every delivery-axis vs-time plot via
            ``annotate_decision_time``. Ignored on SSO-axis plots.
        threshold_search_kwargs: Optional extras for the search
            (``max_gap_multiplier``, ``fallback_tolerance_hours``).
        axis_mode: ``'delivery'`` (default) renders the legacy
            delivery-anchored figures (x = hours before birth, inverted).
            ``'sso'`` renders SSO-anchored figures (x = signed hours
            from second-stage onset, natural orientation, vertical
            zero-line marks SSO). When ``'sso'``, the function emits
            only the per-class vs-time, per-class × subgroup, and
            binary-by-underlying-class artefacts; AUROC-vs-time,
            aggregate (top-1 / macro-F1 / Brier) and
            confusion-matrix-evolution outputs are delivery-axis only.
        n_dropped_guids: SSO-mode figure-footer annotation reporting
            how many GUIDs the caller pre-filtered for missing SSO.
            Ignored in delivery mode.

    Returns:
        Dict with keys:

        * ``perclass_metrics`` — long-format DataFrame
          ``(bin_center, class, ...)``.
        * ``perclass_subgroup_metrics`` — long-format DataFrame
          ``(bin_center, class, subgroup, ...)``.
        * ``binary_by_underlying_class`` — long-format DataFrame
          ``(bin_center, restrict_class, ...)``.
        * ``auroc_vs_time`` — DataFrame ``(bin_center, class, auroc, n_pos, n_neg, n)``.
        * ``top1_accuracy``, ``f1_scores``, ``brier_scores`` — DataFrames.
        * ``perclass_threshold_info`` — ``{class_name: {threshold, ...}}``
          for caller-side aggregation (e.g. writing the consolidated
          ``multiclass_head/perclass_thresholds.json`` file across modes).
    """
    multiclass_root = eval_root / "multiclass_head"
    binary_root = eval_root / "binary_head"
    multiclass_root.mkdir(parents=True, exist_ok=True)
    binary_root.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Any] = {"perclass_threshold_info": {}}

    if thresholds is None and df_val is not None:
        thresholds = {}
        extra = dict(threshold_search_kwargs or {})
        for class_name, _, _ in CLASS_INFO:
            try:
                thr, info = find_perclass_threshold_at_target_fpr(
                    df_val,
                    class_name=class_name,
                    metric_type=metric_type,
                    target_fpr=target_fpr,
                    decision_time_hours=decision_time_hours,
                    **extra,
                )
                thresholds[class_name] = float(thr)
                out["perclass_threshold_info"][class_name] = {
                    "threshold": float(thr), **info,
                }
            except Exception:  # noqa: BLE001 — fall back to argmax for this class
                logger.exception(
                    f"per-class threshold search failed for class={class_name!r} "
                    f"metric_type={metric_type!r} — falling back to argmax for this class"
                )
        if not thresholds:
            thresholds = None  # all classes failed — argmax path

    if axis_mode not in AXIS_MODES:
        raise ValueError(f"axis_mode must be one of {AXIS_MODES}, got {axis_mode!r}")

    df_pc = add_perclass_clinical_columns(df, thresholds=thresholds)

    # ---- per-class vs time ----
    metrics_per_class: Dict[str, pd.DataFrame] = {}
    for class_name, _, _ in CLASS_INFO:
        metrics_per_class[class_name] = compute_perclass_time_binned_metrics(
            df_pc,
            time_bins=time_bins,
            metric_type=metric_type,
            class_name=class_name,
            axis_mode=axis_mode,
        )
    long_perclass = collect_perclass_long_format(metrics_per_class)
    pc_dir = multiclass_root / "per_class_vs_time"
    pc_dir.mkdir(parents=True, exist_ok=True)
    long_perclass.to_csv(pc_dir / f"{metric_type}.csv", index=False)
    plot_perclass_panel_combined(
        metrics_per_class,
        output_path=pc_dir / f"{metric_type}_panel.png",
        metric_type=metric_type,
        decision_time_hours=decision_time_hours,
        axis_mode=axis_mode,
        n_dropped_guids=n_dropped_guids,
    )
    out["perclass_metrics"] = long_perclass

    # ---- per-class × subgroup vs time (long-format) ----
    long_subgroup = compute_perclass_subgroup_long_format(
        df_pc, time_bins=time_bins, metric_type=metric_type,
        axis_mode=axis_mode,
    )
    sg_dir = multiclass_root / "per_class_subgroups_vs_time"
    sg_dir.mkdir(parents=True, exist_ok=True)
    long_subgroup.to_csv(sg_dir / f"{metric_type}.csv", index=False)
    plot_perclass_subgroup_long(
        long_subgroup,
        output_path=sg_dir / f"{metric_type}.png",
        metric_type=metric_type,
        decision_time_hours=decision_time_hours,
        axis_mode=axis_mode,
        n_dropped_guids=n_dropped_guids,
    )
    out["perclass_subgroup_metrics"] = long_subgroup

    # ---- binary head: by underlying class (long-format) ----
    long_bin = compute_binary_by_underlying_class_long_format(
        df, time_bins=time_bins, metric_type=metric_type,
        axis_mode=axis_mode,
    )
    bin_dir = binary_root / "by_underlying_class_vs_time"
    bin_dir.mkdir(parents=True, exist_ok=True)
    long_bin.to_csv(bin_dir / f"{metric_type}.csv", index=False)
    plot_binary_by_underlying_class_long(
        long_bin,
        output_path=bin_dir / f"{metric_type}.png",
        metric_type=metric_type,
        decision_time_hours=decision_time_hours,
        axis_mode=axis_mode,
        n_dropped_guids=n_dropped_guids,
    )
    out["binary_by_underlying_class"] = long_bin

    # The AUROC-vs-time / aggregate-vs-time / confusion-evolution
    # blocks below all read ``epoch_hours`` directly via ``_bin_assign``
    # and use the legacy ``ensure_epoch_hours`` helper. They have no
    # signed-axis counterpart yet, so we only run them on the
    # delivery-anchored axis. SSO-axis equivalents can be added in a
    # follow-up if the per-class + subgroup curves prove insufficient.
    if axis_mode != "delivery":
        return out

    # ---- per-class AUROC vs time (NEW) ----
    try:
        auroc_df = compute_perclass_auroc_vs_time(df_pc, time_bins=time_bins)
        auroc_dir = multiclass_root / "auroc_vs_time"
        auroc_dir.mkdir(parents=True, exist_ok=True)
        auroc_df.to_csv(auroc_dir / f"{metric_type}.csv", index=False)
        plot_perclass_auroc_vs_time(
            auroc_df,
            output_path=auroc_dir / f"{metric_type}.png",
            metric_type=metric_type,
            decision_time_hours=decision_time_hours,
        )
        out["auroc_vs_time"] = auroc_df
    except Exception:  # noqa: BLE001
        logger.exception(f"auroc_vs_time failed for {metric_type}")

    # ---- aggregate: top1, macro-F1, brier (single curves, not per-class) ----
    try:
        agg_dir = multiclass_root / "aggregate_vs_time"
        agg_dir.mkdir(parents=True, exist_ok=True)
        top1_df = compute_topk_accuracy_vs_time(df_pc, time_bins=time_bins)
        f1_df = compute_macro_f1_vs_time(df_pc, time_bins=time_bins)
        brier_df = compute_perclass_brier_vs_time(df_pc, time_bins=time_bins)
        top1_df.to_csv(agg_dir / f"{metric_type}_top1_acc.csv", index=False)
        f1_df.to_csv(agg_dir / f"{metric_type}_macro_f1.csv", index=False)
        brier_df.to_csv(agg_dir / f"{metric_type}_brier_perclass.csv", index=False)
        plot_topk_accuracy_vs_time(
            top1_df, agg_dir / f"{metric_type}_top1_acc.png",
            title_suffix=metric_type,
            decision_time_hours=decision_time_hours,
        )
        plot_macro_f1_vs_time(
            f1_df, agg_dir / f"{metric_type}_macro_f1.png",
            title_suffix=metric_type,
            decision_time_hours=decision_time_hours,
        )
        plot_perclass_brier_vs_time(
            brier_df, agg_dir / f"{metric_type}_brier_perclass.png",
            title_suffix=metric_type,
            decision_time_hours=decision_time_hours,
        )
        out["top1_accuracy"] = top1_df
        out["f1_scores"] = f1_df
        out["brier_scores"] = brier_df
    except Exception:  # noqa: BLE001
        logger.exception(f"aggregate 3-class metrics failed for {metric_type}")

    # ---- confusion-matrix evolution ----
    try:
        ce_dir = multiclass_root / "confusion_evolution"
        ce_dir.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix_evolution(
            df_pc, time_bins=time_bins,
            output_path=ce_dir / f"{metric_type}.png",
        )
    except Exception:  # noqa: BLE001
        logger.exception(f"confusion_evolution failed for {metric_type}")

    return out


# ---------------------------------------------------------------------------
# Aggregate-style metrics over time (single curve, 3 classes pooled)
# ---------------------------------------------------------------------------


def _bin_assign(epoch_hours: np.ndarray, time_bins: np.ndarray) -> np.ndarray:
    """Assign each row to a time bin index (or -1 if outside)."""
    idx = np.digitize(epoch_hours, time_bins) - 1
    n_bins = len(time_bins) - 1
    idx[(idx < 0) | (idx >= n_bins)] = -1
    return idx


def _bin_centers(time_bins: np.ndarray) -> np.ndarray:
    return 0.5 * (time_bins[:-1] + time_bins[1:])


def compute_topk_accuracy_vs_time(
    df: pd.DataFrame, time_bins: np.ndarray
) -> pd.DataFrame:
    """Per-bin top-1 accuracy across all 3 classes pooled.

    Args:
        df: Predictions DataFrame from
            :func:`add_perclass_clinical_columns` (so ``predicted_class_3``
            and ``target`` are NaN-free) with ``epoch_hours`` (hours
            **before** birth, positive numbers).
        time_bins: Bin edges (output of legacy ``compute_time_bins``).

    Returns:
        DataFrame ``[bin_center, top1_accuracy, n]``.
    """
    if "epoch_hours" not in df.columns:
        raise KeyError("df must have 'epoch_hours' (run ensure_epoch_hours first)")
    df = _drop_filled_rows(_ffill_three_class_columns(df.copy()))
    target = df["target"].astype(int).to_numpy() - 1  # -> {0, 1, 2}
    pred = df["predicted_class_3"].astype(int).to_numpy()
    epoch = df["epoch_hours"].astype(float).to_numpy()
    bin_idx = _bin_assign(epoch, time_bins)
    centers = _bin_centers(time_bins)
    rows = []
    for b in range(len(centers)):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin_center": float(centers[b]), "top1_accuracy": float("nan"), "n": 0})
            continue
        acc = float((pred[mask] == target[mask]).mean())
        rows.append({"bin_center": float(centers[b]), "top1_accuracy": acc, "n": n})
    return pd.DataFrame(rows)


def compute_macro_f1_vs_time(
    df: pd.DataFrame, time_bins: np.ndarray
) -> pd.DataFrame:
    """Per-bin macro-F1, weighted-F1, and per-class F1 across 3 classes.

    Args:
        df: Predictions DataFrame from
            :func:`add_perclass_clinical_columns` with ``epoch_hours``.
        time_bins: Bin edges.

    Returns:
        DataFrame ``[bin_center, macro_f1, weighted_f1, f1_healthy,
        f1_acidosis, f1_hie, n]``.
    """
    if "epoch_hours" not in df.columns:
        raise KeyError("df must have 'epoch_hours'")
    df = _drop_filled_rows(_ffill_three_class_columns(df.copy()))
    target = df["target"].astype(int).to_numpy() - 1
    pred = df["predicted_class_3"].astype(int).to_numpy()
    epoch = df["epoch_hours"].astype(float).to_numpy()
    bin_idx = _bin_assign(epoch, time_bins)
    centers = _bin_centers(time_bins)
    rows = []
    for b in range(len(centers)):
        mask = bin_idx == b
        n = int(mask.sum())
        row: Dict[str, Any] = {"bin_center": float(centers[b]), "n": n}
        if n == 0:
            row.update(
                macro_f1=float("nan"),
                weighted_f1=float("nan"),
                f1_healthy=float("nan"),
                f1_acidosis=float("nan"),
                f1_hie=float("nan"),
            )
            rows.append(row)
            continue
        y_true = target[mask]
        y_pred = pred[mask]
        per_class_f1 = []
        per_class_support = []
        for c in (0, 1, 2):
            tp = int(((y_pred == c) & (y_true == c)).sum())
            fp = int(((y_pred == c) & (y_true != c)).sum())
            fn = int(((y_pred != c) & (y_true == c)).sum())
            denom = 2 * tp + fp + fn
            f1c = float(2 * tp / denom) if denom > 0 else float("nan")
            per_class_f1.append(f1c)
            per_class_support.append(int((y_true == c).sum()))
        finite_f1 = [v for v in per_class_f1 if v == v]  # filter NaN
        macro_f1 = float(np.mean(finite_f1)) if finite_f1 else float("nan")
        total_support = sum(per_class_support)
        if total_support > 0 and finite_f1:
            weighted_f1 = float(
                sum(
                    (s * v) for s, v in zip(per_class_support, per_class_f1) if v == v
                )
                / total_support
            )
        else:
            weighted_f1 = float("nan")
        row.update(
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            f1_healthy=per_class_f1[0],
            f1_acidosis=per_class_f1[1],
            f1_hie=per_class_f1[2],
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_perclass_brier_vs_time(
    df: pd.DataFrame, time_bins: np.ndarray
) -> pd.DataFrame:
    """Per-bin Brier score (lower is better) per class.

    Brier_c = mean over rows of (P(class=c) - 1{target==c})^2.

    Args:
        df: Predictions DataFrame from
            :func:`add_perclass_clinical_columns` with ``epoch_hours``.
        time_bins: Bin edges.

    Returns:
        DataFrame ``[bin_center, brier_healthy, brier_acidosis, brier_hie,
        brier_macro, n]``.
    """
    if "epoch_hours" not in df.columns:
        raise KeyError("df must have 'epoch_hours'")
    df = _drop_filled_rows(_ffill_three_class_columns(df.copy()))
    target = df["target"].astype(int).to_numpy()  # 1, 2, 3
    epoch = df["epoch_hours"].astype(float).to_numpy()
    bin_idx = _bin_assign(epoch, time_bins)
    centers = _bin_centers(time_bins)
    probs = {
        name: df[col].astype(float).to_numpy() for name, _, col in CLASS_INFO
    }
    rows = []
    for b in range(len(centers)):
        mask = bin_idx == b
        n = int(mask.sum())
        row: Dict[str, Any] = {"bin_center": float(centers[b]), "n": n}
        if n == 0:
            for name, _, _ in CLASS_INFO:
                row[f"brier_{name}"] = float("nan")
            row["brier_macro"] = float("nan")
            rows.append(row)
            continue
        per_class = []
        for name, target_id, _ in CLASS_INFO:
            y = (target[mask] == target_id).astype(float)
            p = probs[name][mask]
            valid = ~np.isnan(p)
            if not valid.any():
                row[f"brier_{name}"] = float("nan")
                continue
            brier = float(np.mean((p[valid] - y[valid]) ** 2))
            row[f"brier_{name}"] = brier
            per_class.append(brier)
        row["brier_macro"] = float(np.mean(per_class)) if per_class else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot helpers for the new aggregate / global metrics
# ---------------------------------------------------------------------------


def _line_vs_time(
    df: pd.DataFrame,
    *,
    y_cols: Sequence[Tuple[str, str, str]],
    output_path: Path,
    title: str,
    ylabel: str,
    ylim: Optional[Tuple[float, float]] = (0.0, 1.05),
    decision_time_hours: Optional[float] = None,
) -> None:
    """Generic single-axes vs-time plot, x-inverted (time before birth).

    Args:
        df: Long-form DataFrame with at least ``bin_center`` and the y-cols.
        y_cols: Sequence of ``(column_name, label, colour)`` triples.
        output_path: Output PNG path.
        title: Plot title.
        ylabel: Y-axis label.
        ylim: Optional ``(low, high)`` tuple (``None`` to autoscale).
        decision_time_hours: When provided, draws a dashed vertical
            reference line at that x value (via the shared annotator).
    """
    if df is None or df.empty:
        logger.warning(f"_line_vs_time: empty df, skipping {output_path.name}")
        return
    valid = df.dropna(subset=[c for c, _, _ in y_cols], how="all").sort_values(
        "bin_center", ascending=False
    )
    if valid.empty:
        logger.warning(f"_line_vs_time: no non-NaN rows for {output_path.name}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = valid["bin_center"].to_numpy()
    for col, label, colour in y_cols:
        if col not in valid.columns:
            continue
        ax.plot(x, valid[col].to_numpy(), marker="o", linewidth=2.5, markersize=6,
                label=label, color=colour)
    ax.set_xlabel("Hours Before Birth", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.invert_xaxis()
    ax.legend(fontsize=11, loc="best")
    _annotate_decision_time(ax, decision_time_hours)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {output_path.parent.name}/{output_path.name}")


def plot_topk_accuracy_vs_time(
    metrics_df: pd.DataFrame, output_path: Path,
    title_suffix: str = "",
    *,
    decision_time_hours: Optional[float] = None,
) -> None:
    """Top-1 accuracy line plot."""
    title = "Top-1 3-class Accuracy vs Time"
    if title_suffix:
        title += f" ({title_suffix})"
    _line_vs_time(
        metrics_df,
        y_cols=[("top1_accuracy", "Top-1 accuracy", "#2c3e50")],
        output_path=output_path,
        title=title,
        ylabel="Accuracy",
        decision_time_hours=decision_time_hours,
    )


def plot_macro_f1_vs_time(
    metrics_df: pd.DataFrame, output_path: Path,
    title_suffix: str = "",
    *,
    decision_time_hours: Optional[float] = None,
) -> None:
    """Macro-F1 + weighted-F1 + per-class F1 panel."""
    title = "F1 Scores vs Time"
    if title_suffix:
        title += f" ({title_suffix})"
    _line_vs_time(
        metrics_df,
        y_cols=[
            ("macro_f1", "Macro-F1", "#34495e"),
            ("weighted_f1", "Weighted-F1", "#7f8c8d"),
            ("f1_healthy", "F1 (Healthy)", "#27ae60"),
            ("f1_acidosis", "F1 (Acidosis)", "#f39c12"),
            ("f1_hie", "F1 (HIE)", "#c0392b"),
        ],
        output_path=output_path,
        title=title,
        ylabel="F1 Score",
        decision_time_hours=decision_time_hours,
    )


def plot_perclass_brier_vs_time(
    metrics_df: pd.DataFrame, output_path: Path,
    title_suffix: str = "",
    *,
    decision_time_hours: Optional[float] = None,
) -> None:
    """Per-class Brier score + macro Brier."""
    title = "Per-class Brier Score vs Time (lower is better)"
    if title_suffix:
        title += f" ({title_suffix})"
    _line_vs_time(
        metrics_df,
        y_cols=[
            ("brier_macro", "Macro Brier", "#34495e"),
            ("brier_healthy", "Healthy", "#27ae60"),
            ("brier_acidosis", "Acidosis", "#f39c12"),
            ("brier_hie", "HIE", "#c0392b"),
        ],
        output_path=output_path,
        title=title,
        ylabel="Brier Score",
        ylim=(0.0, None),  # type: ignore[arg-type]
        decision_time_hours=decision_time_hours,
    )


# ---------------------------------------------------------------------------
# Global (per-fold, not per-metric-type) 3-class diagnostics
# ---------------------------------------------------------------------------


def _last_per_guid(df: pd.DataFrame) -> pd.DataFrame:
    """Return the row with the largest prefix per GUID (final clinical decision)."""
    sort_col = "prefix_length" if "prefix_length" in df.columns else "position"
    return (
        df.sort_values(["guid", sort_col])
        .groupby("guid", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def plot_perclass_calibration(
    df: pd.DataFrame, output_dir: Path, *, n_bins: int = 12
) -> None:
    """Reliability diagrams (one per class) at the GUID level (last position).

    For each class ``c``, bin predictions by ``prob_c`` into ``n_bins``
    quantile bins and plot mean predicted prob (x) vs observed positive
    fraction (y). Diagonal = perfect calibration.

    Args:
        df: Full per-position predictions DataFrame.
        output_dir: Output directory (created if missing).
        n_bins: Number of equal-width probability bins (default 12).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    last = _last_per_guid(df)
    target = last["guid_class_3_target"].astype(int).to_numpy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    rows = []
    for ax, (name, target_id, prob_col) in zip(axes, CLASS_INFO):
        if prob_col not in last.columns:
            ax.set_title(f"{name}: prob col missing")
            continue
        probs = last[prob_col].astype(float).to_numpy()
        y = (target == (target_id - 1)).astype(int)
        valid = ~np.isnan(probs)
        probs = probs[valid]
        y = y[valid]
        if probs.size == 0:
            ax.set_title(f"{name}: no data")
            continue
        bin_idx = np.digitize(probs, edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        mean_pred = np.full(n_bins, np.nan)
        frac_pos = np.full(n_bins, np.nan)
        counts = np.zeros(n_bins, dtype=int)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.any():
                mean_pred[b] = float(probs[mask].mean())
                frac_pos[b] = float(y[mask].mean())
                counts[b] = int(mask.sum())
                rows.append({
                    "class": name,
                    "bin_low": float(edges[b]),
                    "bin_high": float(edges[b + 1]),
                    "mean_pred": float(mean_pred[b]),
                    "frac_positive": float(frac_pos[b]),
                    "n": int(counts[b]),
                })
        # Reliability line
        finite = ~np.isnan(mean_pred)
        ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.6, label="Perfect")
        ax.plot(mean_pred[finite], frac_pos[finite], marker="o", linewidth=2,
                color="#2c3e50", label="Observed")
        # Calibration ECE
        weights = counts.astype(float)
        weights = weights / max(weights.sum(), 1.0)
        ece = float(np.nansum(weights * np.abs(mean_pred - frac_pos)))
        # Histogram on twin axis
        ax2 = ax.twinx()
        ax2.bar(centers, counts, width=(1.0 / n_bins) * 0.9, alpha=0.18,
                color="#3498db")
        ax2.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"Predicted P({name})")
        ax.set_ylabel("Observed positive fraction")
        ax.set_title(f"{name} (ECE={ece:.3f}, n={int(counts.sum())})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("Per-class Reliability Diagrams (GUID-level, final position)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_perclass.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output_dir / "calibration_perclass.csv", index=False)
    logger.info(f"  Saved: {output_dir.name}/calibration_perclass.png")


def plot_perclass_pr_curves(df: pd.DataFrame, output_dir: Path) -> None:
    """Per-class precision-recall curves at the GUID level (final position).

    Three subplots (one per class) plus a combined plot.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score  # noqa: WPS433

    output_dir.mkdir(parents=True, exist_ok=True)
    last = _last_per_guid(df)
    target = last["guid_class_3_target"].astype(int).to_numpy()
    fig_combined, ax_combined = plt.subplots(figsize=(8, 6))
    palette = {"healthy": "#27ae60", "acidosis": "#f39c12", "hie": "#c0392b"}
    for name, target_id, prob_col in CLASS_INFO:
        if prob_col not in last.columns:
            continue
        probs = last[prob_col].astype(float).to_numpy()
        y = (target == (target_id - 1)).astype(int)
        valid = ~np.isnan(probs)
        probs = probs[valid]
        y = y[valid]
        if y.sum() == 0 or y.sum() == len(y):
            logger.warning(f"PR curve for {name}: degenerate (positives={int(y.sum())}, n={len(y)}); skipping")
            continue
        precision, recall, thresholds = precision_recall_curve(y, probs)
        ap = float(average_precision_score(y, probs))
        # Per-class PNG
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recall, precision, color=palette[name], linewidth=2.2,
                label=f"AP={ap:.3f}")
        ax.fill_between(recall, 0, precision, alpha=0.15, color=palette[name])
        baseline = float(y.mean())
        ax.axhline(baseline, linestyle="--", color="grey", alpha=0.7,
                   label=f"prevalence={baseline:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"PR curve ({name}, GUID-level)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / f"pr_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        # CSV dump
        pd.DataFrame({
            "precision": precision,
            "recall": recall,
            "threshold": np.append(thresholds, np.nan),  # sklearn returns N-1 thresholds
        }).to_csv(output_dir / f"pr_{name}.csv", index=False)
        # Combined
        ax_combined.plot(recall, precision, color=palette[name], linewidth=2.2,
                         label=f"{name} (AP={ap:.3f})")
        logger.info(f"  Saved: {output_dir.name}/pr_{name}.png")
    ax_combined.set_xlabel("Recall")
    ax_combined.set_ylabel("Precision")
    ax_combined.set_xlim(0, 1)
    ax_combined.set_ylim(0, 1.02)
    ax_combined.set_title("PR curves (one-vs-rest, GUID-level final position)")
    ax_combined.grid(True, alpha=0.3)
    ax_combined.legend(loc="best")
    fig_combined.tight_layout()
    fig_combined.savefig(output_dir / "pr_combined.png", dpi=150,
                         bbox_inches="tight")
    plt.close(fig_combined)
    logger.info(f"  Saved: {output_dir.name}/pr_combined.png")


def plot_confusion_matrix_evolution(
    df: pd.DataFrame, time_bins: np.ndarray, output_path: Path,
    *, max_panels: int = 8,
) -> None:
    """Row-normalised 3×3 confusion matrices across time-before-birth bins.

    Renders up to ``max_panels`` confusion matrices (chosen as evenly-spaced
    bins covering the range ``time_bins``), so the user can visualise how
    confusion shifts as we move closer to delivery.

    Args:
        df: Predictions DataFrame from
            :func:`add_perclass_clinical_columns` with ``epoch_hours``.
        time_bins: Bin edges.
        output_path: Output PNG path. Parent dir is created if missing.
        max_panels: Maximum number of CM panels to render (default 8).
    """
    if "epoch_hours" not in df.columns:
        raise KeyError("df must have 'epoch_hours'")
    df = _drop_filled_rows(_ffill_three_class_columns(df.copy()))
    target = df["target"].astype(int).to_numpy() - 1
    pred = df["predicted_class_3"].astype(int).to_numpy()
    epoch = df["epoch_hours"].astype(float).to_numpy()
    bin_idx = _bin_assign(epoch, time_bins)
    centers = _bin_centers(time_bins)
    n_bins = len(centers)
    keep = sorted(set(np.linspace(0, n_bins - 1, num=min(max_panels, n_bins), dtype=int).tolist()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = min(len(keep), 4)
    rows = int(np.ceil(len(keep) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.4))
    axes_flat = np.atleast_1d(axes).flatten()
    class_names = ["healthy", "acidosis", "hie"]
    for ax_i, b in enumerate(keep):
        ax = axes_flat[ax_i]
        mask = bin_idx == b
        if not mask.any():
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"~{centers[b]:.2f} h before birth")
            continue
        cm = np.zeros((3, 3), dtype=int)
        for t in (0, 1, 2):
            for p in (0, 1, 2):
                cm[t, p] = int(((target[mask] == t) & (pred[mask] == p)).sum())
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                        ha="center", va="center", fontsize=8,
                        color="black" if cm_norm[i, j] < 0.5 else "white")
        ax.set_xticks(range(3))
        ax.set_xticklabels(class_names, fontsize=8)
        ax.set_yticks(range(3))
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_title(f"~{centers[b]:.2f} h before birth (n={int(mask.sum())})",
                     fontsize=9)
    # Hide unused subplots
    for k in range(len(keep), len(axes_flat)):
        axes_flat[k].axis("off")
    fig.suptitle("3-class confusion evolution (row-normalised)", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {output_path.parent.name}/{output_path.name}")


def plot_perclass_probability_box(df: pd.DataFrame, output_dir: Path) -> None:
    """For each predicted-prob column, a violin/box plot stratified by true class.

    Adds context to the existing ``three_class_probability_hist.png`` by
    showing the median and IQR overlaid.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    last = _last_per_guid(df)
    target = last["guid_class_3_target"].astype(int).to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    palette = ["#27ae60", "#f39c12", "#c0392b"]
    class_names = ["healthy", "acidosis", "hie"]
    for ax, (name, _, prob_col) in zip(axes, CLASS_INFO):
        if prob_col not in last.columns:
            continue
        data = []
        labels = []
        for cls_id, cls_label in enumerate(class_names):
            mask = target == cls_id
            if mask.sum() == 0:
                continue
            vals = last.loc[mask, prob_col].astype(float).to_numpy()
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            data.append(vals)
            labels.append(f"{cls_label}\n(n={int(mask.sum())})")
        if not data:
            ax.set_title(f"P({name}): no data")
            continue
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, c in zip(bp["boxes"], palette[:len(data)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(f"Predicted P({name})")
        ax.set_xlabel("True class")
        ax.set_title(f"P({name}) by true class")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Per-class probability distribution by ground truth",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "prob_by_truth_box.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {output_dir.name}/prob_by_truth_box.png")


def run_3class_global_diagnostics(
    df: pd.DataFrame, output_dir: Path
) -> Dict[str, Any]:
    """Run all per-fold (non-time-binned) 3-class diagnostics in one go.

    These artefacts complement the per-metric-type ``per_class/`` and
    ``binary_by_underlying_class/`` outputs. They live under the existing
    ``three_class_diagnostics/`` directory and need to be computed only
    once per fold (they don't depend on which CDR threshold is in play).

    Args:
        df: Full per-position predictions DataFrame (from
            :func:`run_inference_per_position`).
        output_dir: ``evaluation/three_class_diagnostics/``.

    Returns:
        Dict ``{section: status}``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses: Dict[str, Any] = {}
    for name, fn in (
        ("calibration", lambda: plot_perclass_calibration(df, output_dir / "calibration")),
        ("pr_curves", lambda: plot_perclass_pr_curves(df, output_dir / "pr_curves")),
        ("prob_box", lambda: plot_perclass_probability_box(df, output_dir / "prob_distributions")),
    ):
        try:
            fn()
            statuses[name] = "ok"
        except Exception:  # noqa: BLE001
            logger.exception(f"3-class diagnostic '{name}' failed")
            statuses[name] = "failed"
    return statuses


def write_perclass_thresholds_json(
    df_val: pd.DataFrame,
    *,
    output_path: Path,
    target_fpr: float = 0.2,
    decision_time_hours: float = 1.0,
) -> Dict[str, Any]:
    """Run :func:`find_perclass_threshold_at_target_fpr` per class + save JSON.

    Args:
        df_val: Validation predictions DataFrame.
        output_path: Where to write ``perclass_thresholds.json``.
        target_fpr: Target FPR at the decision time.
        decision_time_hours: Decision time in hours before delivery.

    Returns:
        Dict carrying per-class thresholds and validation metrics.
    """
    perclass_info: Dict[str, Any] = {}
    for class_name, _, _ in CLASS_INFO:
        thr, info = find_perclass_threshold_at_target_fpr(
            df_val,
            class_name=class_name,
            target_fpr=target_fpr,
            decision_time_hours=decision_time_hours,
        )
        perclass_info[class_name] = {"threshold": float(thr), **info}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(perclass_info, indent=2, default=float))
    return perclass_info


__all__ = [
    "CLASS_INFO",
    "METRIC_TYPES",
    "add_perclass_clinical_columns",
    "collect_perclass_long_format",
    "compute_binary_by_underlying_class",
    "compute_binary_by_underlying_class_long_format",
    "compute_macro_f1_vs_time",
    "compute_perclass_auroc_vs_time",
    "compute_perclass_brier_vs_time",
    "compute_perclass_subgroup_long_format",
    "compute_perclass_time_binned_metrics",
    "compute_topk_accuracy_vs_time",
    "find_perclass_threshold_at_target_fpr",
    "plot_binary_by_underlying_class_long",
    "plot_confusion_matrix_evolution",
    "plot_macro_f1_vs_time",
    "plot_perclass_auroc_vs_time",
    "plot_perclass_brier_vs_time",
    "plot_perclass_calibration",
    "plot_perclass_panel_combined",
    "plot_perclass_pr_curves",
    "plot_perclass_probability_box",
    "plot_perclass_subgroup_long",
    "plot_topk_accuracy_vs_time",
    "run_3class_evaluation_for_metric_type",
    "run_3class_global_diagnostics",
    "write_perclass_thresholds_json",
]
