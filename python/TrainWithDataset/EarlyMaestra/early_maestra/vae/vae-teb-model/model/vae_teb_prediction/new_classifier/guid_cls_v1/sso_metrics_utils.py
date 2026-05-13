"""Second-stage-onset-referenced evaluation utilities for ``guid_cls_v1``.

This module is the SSO-axis sibling of :mod:`clinical_metrics_utils`. It
provides the same time-binned metric computations and figure renderers
that the per-fold evaluator already produces against ``epoch_hours``
(positive hours **before birth**, inverted x-axis), but it operates on
``t_rel_sso_hours`` — the **signed** hours-relative-to-second-stage-onset
axis (negative = before SSO, positive = after SSO).

Why a parallel module instead of parameterising the delivery-axis code?

1. **Inequality direction flips.** The "committed" metrics use
   ``epoch_hours >= bin_center`` to denote "segment observed at or before
   this $\\tau$" because $epoch\\_hours$ *decreases* with absolute time. For the
   SSO axis the natural reading is reversed: ``t_rel_sso_hours <= bin_center``
   means "observed at or before SSO time $\\tau$". Trying to express this
   with a single parameterised function obscures the per-axis semantics
   and risks silent off-by-one bugs.
2. **The "exclude post-delivery" bin filter is wrong for SSO.** The
   delivery-axis :func:`clinical_metrics_utils.compute_time_bins` drops
   ``epoch_hours <= 0`` because there is no meaningful data after
   delivery. The SSO axis spans both signs by construction and must keep
   the full range.
3. **X-axis orientation is opposite.** Delivery-axis plots
   ``ax.invert_xaxis()`` so birth sits on the right; SSO-axis plots want
   natural left-to-right reading with $t=0$ (SSO) marked by an
   ``axvline``.

Public surface (mirrors :mod:`clinical_metrics_utils`):

* :func:`ensure_t_rel_sso_hours` — guarantee the signed axis column is
  present, copying from ``sso_hours`` if necessary.
* :func:`filter_to_sso_eligible` — drop GUIDs that lack SSO end-to-end
  and return the dropped count for figure annotations.
* :func:`recompute_t_rel_sso_after_fill` — repair
  ``t_rel_sso_hours`` on rows synthesised by the legacy
  :func:`clinical_metrics_utils.fill_missing_epochs` (which only knows
  about the binary-side columns).
* :func:`compute_sso_time_bins` — uniform signed bin edges.
* :func:`compute_instantaneous_metrics_sso`,
  :func:`compute_committed_cumulative_metrics_sso`,
  :func:`compute_committed_overall_metrics_sso` — bin-level metric
  computations matching the legacy schema (``bin_center``,
  ``sensitivity``, ``specificity``, ``fpr``, ``n_*``).
* :func:`plot_single_metric_type_sso` — five-PNG renderer (sensitivity,
  sens+spec, sens+FPR, all-metrics, FPR-only).
* :func:`plot_subgroup_analysis_sso` — diagnosis / CS / BG / healthy
  4-way subgroup PNGs.

Output columns and figure styles are kept byte-compatible with the
delivery-axis variants so the cross-fold aggregator can treat both
trees uniformly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ``loguru`` is the project-standard logger but is not always installed
# in the lightweight CI environment used for schema-level tests. Fall
# back to the stdlib :mod:`logging` module so this file remains
# importable when ``loguru`` is missing.
try:
    from loguru import logger  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - exercised only in lightweight CI
    import logging

    logger = logging.getLogger(__name__)

# Matplotlib is imported lazily inside the plot functions so this module
# stays importable in lightweight environments (CI test runners that
# only check schema-level helpers) where matplotlib is not installed.


SSO_TIME_COL: str = "t_rel_sso_hours"
"""Canonical name of the signed SSO-axis column written by
:func:`evaluate_guid_classifier.run_inference_per_position` and consumed
by every function in this module."""


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def ensure_t_rel_sso_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame carries a ``t_rel_sso_hours`` column.

    Recent prediction CSVs write the column directly; older CSVs only
    carry ``sso_hours`` (the legacy alias with identical semantics). To
    keep the SSO-axis tools loadable against any prior run, this helper
    derives ``t_rel_sso_hours`` from ``sso_hours`` when it is missing,
    and is a no-op when it is already present.

    Args:
        df: Predictions DataFrame.

    Returns:
        Same DataFrame (copy) with a populated ``t_rel_sso_hours``
        column. ``NaN`` rows are preserved.

    Raises:
        KeyError: When neither ``t_rel_sso_hours`` nor ``sso_hours``
            are present.
    """
    if SSO_TIME_COL in df.columns:
        return df
    if "sso_hours" not in df.columns:
        raise KeyError(
            f"DataFrame is missing both '{SSO_TIME_COL}' and 'sso_hours'; "
            "cannot derive the SSO-referenced axis."
        )
    df = df.copy()
    df[SSO_TIME_COL] = df["sso_hours"].astype(float)
    logger.debug(
        f"ensure_t_rel_sso_hours: derived '{SSO_TIME_COL}' from 'sso_hours' "
        f"({len(df)} rows)"
    )
    return df


def filter_to_sso_eligible(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Drop GUIDs without a finite second-stage-onset timestamp.

    SSO is a per-recording (per-GUID) scalar that is either fully observed
    or entirely missing — there is no clinical scenario where one segment
    of a recording has SSO and another does not. To be defensive, this
    helper drops the whole GUID whenever **any** of its rows have NaN in
    :data:`SSO_TIME_COL` (this also handles the edge case of an upstream
    bug propagating partial NaNs into a GUID).

    Args:
        df: Predictions DataFrame with a populated :data:`SSO_TIME_COL`
            column (call :func:`ensure_t_rel_sso_hours` first when
            unsure).

    Returns:
        Tuple ``(filtered_df, stats)`` where ``stats`` carries:

        * ``n_dropped_guids``: number of GUIDs removed.
        * ``n_kept_guids``: number of GUIDs retained.
        * ``n_total_guids``: total GUIDs before filtering.
        * ``dropped_guids``: sorted list of dropped GUID strings (for
          the per-fold ``sso_filter_summary.json`` artefact).

    Raises:
        KeyError: When :data:`SSO_TIME_COL` is missing.
    """
    if SSO_TIME_COL not in df.columns:
        raise KeyError(
            f"filter_to_sso_eligible requires '{SSO_TIME_COL}'; "
            "call ensure_t_rel_sso_hours first."
        )
    if "guid" not in df.columns:
        raise KeyError("filter_to_sso_eligible requires a 'guid' column.")

    has_nan_per_guid = (
        df.groupby("guid", sort=False)[SSO_TIME_COL]
        .apply(lambda s: bool(s.isna().any()))
    )
    eligible = has_nan_per_guid[~has_nan_per_guid].index.tolist()
    dropped = sorted(has_nan_per_guid[has_nan_per_guid].index.tolist())

    stats: Dict[str, Any] = {
        "n_total_guids": int(len(has_nan_per_guid)),
        "n_kept_guids": int(len(eligible)),
        "n_dropped_guids": int(len(dropped)),
        "dropped_guids": [str(g) for g in dropped],
    }

    if not eligible:
        logger.warning(
            "filter_to_sso_eligible: all GUIDs dropped (no SSO available); "
            "returning empty DataFrame"
        )
        return df.iloc[0:0].copy(), stats

    out = df[df["guid"].isin(eligible)].copy()
    logger.info(
        f"filter_to_sso_eligible: kept {stats['n_kept_guids']} / "
        f"{stats['n_total_guids']} GUIDs "
        f"(dropped {stats['n_dropped_guids']} for missing SSO)"
    )
    return out, stats


def recompute_t_rel_sso_after_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Repair ``t_rel_sso_hours`` on rows synthesised by ``fill_missing_epochs``.

    The legacy gap-filler in :mod:`clinical_metrics_utils` only knows
    about the binary-side columns (``prob_class_*``, ``binary_target``,
    ``clinical_pred``, ``cs_label``, ``bg_label``) and leaves SSO / TLO
    columns NaN on its synthesised rows. Because the SSO offset is
    constant within a recording, ``t_rel_sso_hours`` can be reconstructed
    from any single non-NaN row of the same GUID via:

    $$
        \\text{offset\\_seconds}
            = \\mathrm{epoch} - \\mathrm{t\\_rel\\_sso\\_hours} \\cdot 3600,
    $$

    after which every row of the GUID gets

    $$
        \\mathrm{t\\_rel\\_sso\\_hours} =
            \\frac{\\mathrm{epoch} - \\mathrm{offset\\_seconds}}{3600}.
    $$

    GUIDs that have no non-NaN reference row (every row was synthesised,
    which would only happen if the GUID was 100% gap) are left untouched
    so :func:`filter_to_sso_eligible` can still drop them downstream.

    Args:
        df: DataFrame returned by :func:`fill_missing_epochs`. Must have
            ``guid``, ``epoch`` and either :data:`SSO_TIME_COL` or
            ``sso_hours``.

    Returns:
        Copy of ``df`` with ``t_rel_sso_hours`` populated on all rows
        whose GUID had at least one observed segment.
    """
    df = ensure_t_rel_sso_hours(df.copy())
    if "epoch" not in df.columns:
        return df

    # Per-GUID: first non-NaN (epoch, t_rel_sso_hours) yields the offset.
    sso_seconds = df[SSO_TIME_COL] * 3600.0
    offset_seconds = df["epoch"].astype(float) - sso_seconds
    df["_sso_offset_seconds_tmp_"] = offset_seconds
    per_guid_offset = (
        df.dropna(subset=["_sso_offset_seconds_tmp_"])
        .groupby("guid", sort=False)["_sso_offset_seconds_tmp_"]
        .first()
    )
    df = df.drop(columns=["_sso_offset_seconds_tmp_"])

    if per_guid_offset.empty:
        return df

    # Broadcast offsets back per row, then recompute t_rel_sso_hours
    # for any row whose value is currently NaN.
    mapped_offset = df["guid"].map(per_guid_offset)
    needs_fix = df[SSO_TIME_COL].isna() & mapped_offset.notna()
    if needs_fix.any():
        df.loc[needs_fix, SSO_TIME_COL] = (
            (df.loc[needs_fix, "epoch"].astype(float) - mapped_offset[needs_fix])
            / 3600.0
        )
        logger.debug(
            f"recompute_t_rel_sso_after_fill: filled "
            f"{int(needs_fix.sum())} rows from per-GUID SSO offsets"
        )
    return df


# ---------------------------------------------------------------------------
# Bin computation
# ---------------------------------------------------------------------------


def _round_minute(width_hours: float) -> float:
    """Round a bin width (in hours) to the nearest minute."""
    return float(round(width_hours * 60.0) / 60.0)


def _infer_segment_spacing_seconds(df: pd.DataFrame) -> float:
    """Robust per-GUID epoch-spacing inference (in seconds).

    Local mirror of
    :func:`clinical_metrics_utils.infer_epoch_interval_seconds` — kept
    here so :func:`compute_sso_time_bins` doesn't drag in the legacy
    module's heavy imports (notably ``torch``) when only schema-level
    helpers are required.

    Returns ``0.0`` when the spacing cannot be inferred (e.g. one
    segment per GUID).
    """
    if df is None or len(df) < 2 or "epoch" not in df.columns:
        return 0.0
    if "is_filled" in df.columns:
        df = df[df["is_filled"] == False]  # noqa: E712
        if len(df) < 2:
            return 0.0

    if "guid" not in df.columns:
        epochs = np.sort(np.unique(df["epoch"].to_numpy().astype(float)))
        return float(np.median(np.diff(epochs))) if epochs.size >= 2 else 0.0

    deltas: List[float] = []
    for _, g in df.groupby("guid", sort=False):
        ep = np.sort(np.unique(g["epoch"].to_numpy().astype(float)))
        if ep.size < 2:
            continue
        d = np.diff(ep)
        d = d[d > 0]
        if d.size:
            deltas.extend(np.round(d, 0).tolist())

    if len(deltas) >= 10:
        vc = pd.Series(deltas).value_counts()
        return float(vc.index[0])

    return float(np.median(deltas)) if deltas else 0.0


def compute_sso_time_bins(
    df: pd.DataFrame,
    *,
    bin_size_hours: Optional[float] = None,
    pad_bins: int = 1,
) -> np.ndarray:
    """Compute uniform signed bin edges across :data:`SSO_TIME_COL`.

    The bin width defaults to one segment spacing — inferred from the
    per-GUID epoch interval via
    :func:`clinical_metrics_utils.infer_epoch_interval_seconds` — which
    matches the granularity of the underlying observations and the
    delivery-axis :func:`clinical_metrics_utils.compute_time_bins`.

    Args:
        df: DataFrame carrying :data:`SSO_TIME_COL` and ``epoch``.
        bin_size_hours: Optional explicit bin width. Pass ``None``
            (the default) to auto-infer from segment spacing.
        pad_bins: Number of bin widths to pad on both ends of the
            observed range so the extreme observations have margin
            around them. Defaults to 1.

    Returns:
        Float-valued ``np.ndarray`` of bin edges, monotonically
        increasing, suitable for ``bins[:-1]`` / ``bins[1:]`` slicing
        in the metric functions below.
    """
    # Defensive: empty data falls back to a single zero-width-ish bin.
    if df is None or len(df) == 0 or SSO_TIME_COL not in df.columns:
        return np.array([-0.5, 0.5], dtype=float)

    sso = df[SSO_TIME_COL].astype(float).to_numpy()
    sso = sso[np.isfinite(sso)]
    if sso.size == 0:
        return np.array([-0.5, 0.5], dtype=float)

    if bin_size_hours is None:
        inferred_seconds = _infer_segment_spacing_seconds(df)
        bin_size_hours = (
            inferred_seconds / 3600.0 if inferred_seconds > 0 else (1.0 / 3.0)
        )
    bin_size_hours = _round_minute(bin_size_hours)
    if bin_size_hours <= 0:
        bin_size_hours = 1.0 / 3.0  # 20 min fallback (matches segment spacing).

    lo = float(np.floor(sso.min() / bin_size_hours) * bin_size_hours)
    hi = float(np.ceil(sso.max() / bin_size_hours) * bin_size_hours)
    lo -= pad_bins * bin_size_hours
    hi += pad_bins * bin_size_hours

    if hi <= lo:
        hi = lo + bin_size_hours

    # ``np.arange`` with a final +half-step ensures the closing edge is
    # included and avoids floating-point eat-the-last-bin pathologies.
    bins = np.arange(lo, hi + bin_size_hours / 2.0, bin_size_hours)
    return bins.astype(float)


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------


def _ensure_filled_for_sso_committed(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill (epoch-axis) and repair t_rel_sso_hours.

    The committed metrics expect a contiguous per-GUID timeline so the
    detection-state-stays-detected semantic is well defined across
    arbitrarily long gaps. This helper applies the existing
    epoch-axis gap fill (which preserves :func:`apply_clinical_decision_rule`
    semantics) and then re-derives :data:`SSO_TIME_COL` on the
    synthesised rows via :func:`recompute_t_rel_sso_after_fill`.

    When the legacy ``clinical_metrics_utils`` module cannot be imported
    (e.g. lightweight CI environments without ``torch``), the fill is
    skipped — the metric remains correct on data that has no gaps
    relative to delivery, which is the common case for screened
    eligibility-filtered test sets.

    Args:
        df: Post-CDR DataFrame with :data:`SSO_TIME_COL` present.

    Returns:
        Filled DataFrame ready for the SSO committed metric functions.
    """
    try:
        from model.vae_teb_prediction.new_classifier.guid_cls_v1.clinical_metrics_utils import (  # noqa: WPS433
            ensure_committed_epochs_filled,
        )
    except ModuleNotFoundError:  # pragma: no cover - lightweight CI only
        logger.debug(
            "_ensure_filled_for_sso_committed: clinical_metrics_utils "
            "unavailable; skipping epoch fill step"
        )
        return df

    filled = ensure_committed_epochs_filled(df)
    return recompute_t_rel_sso_after_fill(filled)


def compute_instantaneous_metrics_sso(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    subgroup_filter: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
) -> pd.DataFrame:
    """Instantaneous bin-level metrics on the SSO axis.

    Per signed time bin $\\tau$ ($\\tau$ in hours from SSO):

    $$
        \\mathrm{Sens}(\\tau) = \\frac{TP(\\tau)}{P(\\tau)}, \\quad
        \\mathrm{FPR}(\\tau) = \\frac{FP(\\tau)}{N(\\tau)},
    $$

    where every count is restricted to segments whose
    :data:`SSO_TIME_COL` falls inside ``[bin_start, bin_end)``.

    Args:
        df: Predictions DataFrame with :data:`SSO_TIME_COL`,
            ``binary_target`` and ``clinical_pred``.
        time_bins: Signed bin edges (typically from
            :func:`compute_sso_time_bins`).
        subgroup_filter: Optional row-level mask producer.

    Returns:
        DataFrame with the same schema as the delivery-axis
        :func:`clinical_metrics_utils.compute_instantaneous_metrics`
        (``bin_center``, ``sensitivity``, ``specificity``, ``fpr``,
        ``n_positive``, ``n_negative``, ``n_tp``, ``n_fp``, ``n_tn``,
        ``n_fn``).
    """
    if subgroup_filter is not None:
        df = df[subgroup_filter(df)].copy()
    if df is None or len(df) == 0:
        logger.warning(
            "compute_instantaneous_metrics_sso: empty dataframe after filter"
        )
        return pd.DataFrame()
    df = ensure_t_rel_sso_hours(df)

    sso = df[SSO_TIME_COL].astype(float).to_numpy()
    bt = df["binary_target"].astype(int).to_numpy()
    cp = df["clinical_pred"].astype(int).to_numpy()
    finite = np.isfinite(sso)

    results: List[Dict[str, Any]] = []
    for i in range(len(time_bins) - 1):
        bin_start, bin_end = float(time_bins[i]), float(time_bins[i + 1])
        bin_center = 0.5 * (bin_start + bin_end)
        mask = finite & (sso >= bin_start) & (sso < bin_end)
        if not mask.any():
            results.append({
                "bin_center": bin_center,
                "sensitivity": np.nan,
                "specificity": np.nan,
                "fpr": np.nan,
                "n_positive": 0,
                "n_negative": 0,
                "n_tp": 0,
                "n_fp": 0,
                "n_tn": 0,
                "n_fn": 0,
            })
            continue
        bt_b = bt[mask]
        cp_b = cp[mask]
        n_pos = int((bt_b == 1).sum())
        n_neg = int((bt_b == 0).sum())
        n_tp = int(((bt_b == 1) & (cp_b == 1)).sum())
        n_fp = int(((bt_b == 0) & (cp_b == 1)).sum())
        n_tn = int(((bt_b == 0) & (cp_b == 0)).sum())
        n_fn = int(((bt_b == 1) & (cp_b == 0)).sum())
        sens = float(n_tp / n_pos) if n_pos > 0 else np.nan
        fpr = float(n_fp / n_neg) if n_neg > 0 else np.nan
        spec = float(n_tn / n_neg) if n_neg > 0 else np.nan
        results.append({
            "bin_center": bin_center,
            "sensitivity": sens,
            "specificity": spec,
            "fpr": fpr,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_tp": n_tp,
            "n_fp": n_fp,
            "n_tn": n_tn,
            "n_fn": n_fn,
        })
    out = pd.DataFrame(results)
    logger.info(
        f"compute_instantaneous_metrics_sso: {len(out)} bins "
        f"({out['sensitivity'].notna().sum()} non-NaN)"
    )
    return out


def compute_committed_cumulative_metrics_sso(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    subgroup_filter: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
) -> pd.DataFrame:
    """Committed-decision cumulative metrics on the SSO axis.

    Per signed bin centre $\\tau$:

    * ``available_mask = t_rel_sso_hours <= bin_center`` — segments
      observed at or before SSO time $\\tau$.
    * A GUID is detected at $\\tau$ when *any* segment in its
      ``available`` window has ``clinical_pred == 1``.
    * Sensitivity / FPR use the **available** denominator (GUIDs whose
      recording reaches as far back as $\\tau$ in SSO time). The
      denominator therefore changes with $\\tau$, so sensitivity is not
      strictly monotonic — the cumulative-overall variant should be the
      primary reporting metric.

    Output schema matches
    :func:`clinical_metrics_utils.compute_committed_cumulative_metrics`.
    """
    if subgroup_filter is not None:
        df = df[subgroup_filter(df)].copy()
    if df is None or len(df) == 0:
        logger.warning(
            "compute_committed_cumulative_metrics_sso: empty dataframe after filter"
        )
        return pd.DataFrame()

    df = _ensure_filled_for_sso_committed(df)
    df = ensure_t_rel_sso_hours(df)
    guid_targets = df.groupby("guid")["binary_target"].first()

    results: List[Dict[str, Any]] = []
    sso_arr = df[SSO_TIME_COL].astype(float).to_numpy()
    finite = np.isfinite(sso_arr)
    guids_arr = df["guid"].to_numpy()
    cp_arr = df["clinical_pred"].astype(int).to_numpy()

    for i in range(len(time_bins) - 1):
        bin_start, bin_end = float(time_bins[i]), float(time_bins[i + 1])
        bin_center = 0.5 * (bin_start + bin_end)

        available = finite & (sso_arr <= bin_center)
        if not available.any():
            results.append({
                "bin_center": bin_center,
                "sensitivity": np.nan,
                "specificity": np.nan,
                "fpr": np.nan,
                "n_positive_available": 0,
                "n_negative_available": 0,
                "n_detected_positive": 0,
                "n_detected_negative": 0,
            })
            continue

        # Unique GUIDs in the available window.
        avail_guids = np.unique(guids_arr[available])
        avail_pos = [g for g in avail_guids if guid_targets.get(g, -1) == 1]
        avail_neg = [g for g in avail_guids if guid_targets.get(g, -1) == 0]
        n_pos_avail = len(avail_pos)
        n_neg_avail = len(avail_neg)

        # Per-GUID "detected by tau" check.
        detected_pos = 0
        for g in avail_pos:
            g_mask = available & (guids_arr == g)
            if g_mask.any() and (cp_arr[g_mask] == 1).any():
                detected_pos += 1
        detected_neg = 0
        for g in avail_neg:
            g_mask = available & (guids_arr == g)
            if g_mask.any() and (cp_arr[g_mask] == 1).any():
                detected_neg += 1

        sens = detected_pos / n_pos_avail if n_pos_avail > 0 else np.nan
        fpr = detected_neg / n_neg_avail if n_neg_avail > 0 else np.nan
        spec = (
            (n_neg_avail - detected_neg) / n_neg_avail
            if n_neg_avail > 0
            else np.nan
        )

        results.append({
            "bin_center": bin_center,
            "sensitivity": sens,
            "specificity": spec,
            "fpr": fpr,
            "n_positive_available": n_pos_avail,
            "n_negative_available": n_neg_avail,
            "n_detected_positive": detected_pos,
            "n_detected_negative": detected_neg,
        })

    out = pd.DataFrame(results)
    # Natural axis: ascending bin_center so the curve reads left-to-right.
    out = out.sort_values("bin_center", ascending=True).reset_index(drop=True)
    logger.info(
        f"compute_committed_cumulative_metrics_sso: {len(out)} bins "
        f"({out['sensitivity'].notna().sum()} non-NaN)"
    )
    return out


def compute_committed_overall_metrics_sso(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    subgroup_filter: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
) -> pd.DataFrame:
    """Committed-decision overall metrics on the SSO axis (PRIMARY).

    Identical to :func:`compute_committed_cumulative_metrics_sso` except
    the denominator is **fixed** at the total positive / negative GUID
    counts in the dataset. Sensitivity is therefore monotonically
    non-decreasing in :math:`\\tau` (sorted ascending).
    """
    if subgroup_filter is not None:
        df = df[subgroup_filter(df)].copy()
    if df is None or len(df) == 0:
        logger.warning(
            "compute_committed_overall_metrics_sso: empty dataframe after filter"
        )
        return pd.DataFrame()

    df = _ensure_filled_for_sso_committed(df)
    df = ensure_t_rel_sso_hours(df)
    guid_targets = df.groupby("guid")["binary_target"].first()
    all_pos = guid_targets[guid_targets == 1].index.tolist()
    all_neg = guid_targets[guid_targets == 0].index.tolist()
    n_pos_total = len(all_pos)
    n_neg_total = len(all_neg)

    logger.info(
        "compute_committed_overall_metrics_sso: FIXED denominators "
        f"P_total={n_pos_total}, N_total={n_neg_total}"
    )

    sso_arr = df[SSO_TIME_COL].astype(float).to_numpy()
    finite = np.isfinite(sso_arr)
    guids_arr = df["guid"].to_numpy()
    cp_arr = df["clinical_pred"].astype(int).to_numpy()

    results: List[Dict[str, Any]] = []
    for i in range(len(time_bins) - 1):
        bin_start, bin_end = float(time_bins[i]), float(time_bins[i + 1])
        bin_center = 0.5 * (bin_start + bin_end)

        available = finite & (sso_arr <= bin_center)
        # Pre-index segments-to-guid for available rows.
        detected_pos = 0
        for g in all_pos:
            g_mask = available & (guids_arr == g)
            if g_mask.any() and (cp_arr[g_mask] == 1).any():
                detected_pos += 1
        detected_neg = 0
        for g in all_neg:
            g_mask = available & (guids_arr == g)
            if g_mask.any() and (cp_arr[g_mask] == 1).any():
                detected_neg += 1

        avail_guids = np.unique(guids_arr[available]) if available.any() else np.array([])
        n_avail_pos = int(sum(1 for g in avail_guids if g in set(all_pos)))
        n_avail_neg = int(sum(1 for g in avail_guids if g in set(all_neg)))

        sens = detected_pos / n_pos_total if n_pos_total > 0 else np.nan
        fpr = detected_neg / n_neg_total if n_neg_total > 0 else np.nan
        spec = (
            (n_neg_total - detected_neg) / n_neg_total
            if n_neg_total > 0
            else np.nan
        )

        results.append({
            "bin_center": bin_center,
            "sensitivity": sens,
            "specificity": spec,
            "fpr": fpr,
            "n_positive_total": n_pos_total,
            "n_negative_total": n_neg_total,
            "n_detected_positive": detected_pos,
            "n_detected_negative": detected_neg,
            "n_available_positive": n_avail_pos,
            "n_available_negative": n_avail_neg,
        })

    out = pd.DataFrame(results)
    out = out.sort_values("bin_center", ascending=True).reset_index(drop=True)
    logger.info(
        f"compute_committed_overall_metrics_sso: PRIMARY {len(out)} bins "
        f"({out['sensitivity'].notna().sum()} non-NaN)"
    )

    # Monotonicity check (informational; sorted ascending => non-decreasing
    # expected).
    valid = out["sensitivity"].dropna()
    if len(valid) > 1:
        violations = int((valid.diff() < -1e-6).sum())
        if violations > 0:
            logger.debug(
                "compute_committed_overall_metrics_sso: "
                f"{violations} monotonicity-violation bins (sorted ascending)"
            )
    return out


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


_SSO_X_LABEL: str = "Hours from second stage onset"
_SSO_TITLE_SUFFIX: str = "axis: SSO"


def _annotate_sso_zero(ax: Any) -> None:
    """Draw a faint vertical reference line at $t_{\\mathrm{SSO}} = 0$."""
    xlim = ax.get_xlim()
    lo, hi = min(xlim), max(xlim)
    if 0.0 < lo or 0.0 > hi:
        return
    ax.axvline(
        x=0.0,
        color="0.35",
        linestyle=":",
        linewidth=1.1,
        zorder=0,
    )
    ylim = ax.get_ylim()
    ax.text(
        x=0.0,
        y=ylim[1],
        s=" SSO",
        fontsize=7,
        color="0.35",
        va="top",
        ha="left",
    )


def _add_dropped_guid_annotation(
    fig: Any,
    n_dropped: int,
    *,
    extra: Optional[str] = None,
) -> None:
    """Render a footer-style annotation reporting GUIDs dropped for SSO.

    Mirrors the delivery-side legend convention of carrying GUID counts
    without crowding the per-subgroup ``(N=...)`` chips. The footer is
    only rendered when at least one GUID was dropped — a clean plot
    with zero drops doesn't need the noise. Matches the gating logic
    in :func:`evaluate_3class_metrics._maybe_add_dropped_footer` so
    both SSO renderers behave consistently.

    Args:
        fig: Active figure.
        n_dropped: Count of GUIDs dropped because their SSO timestamp
            was missing. Non-positive values produce no annotation.
        extra: Optional additional descriptor (e.g. "per-fold sum:
            ...") rendered after the dropped count.
    """
    if int(n_dropped) <= 0:
        return
    import matplotlib.pyplot as plt  # noqa: F401, WPS433
    parts = [f"Dropped {int(n_dropped)} GUIDs (no second-stage onset)"]
    if extra:
        parts.append(extra)
    fig.text(
        0.995,
        0.005,
        " | ".join(parts),
        fontsize=8,
        color="0.35",
        ha="right",
        va="bottom",
    )


def _compose_title(metric_label: str, title_suffix: str) -> str:
    base = f" - {metric_label} ({_SSO_TITLE_SUFFIX}"
    if title_suffix:
        base += f"; {title_suffix.strip(' -:')}"
    base += ")"
    return base


def plot_single_metric_type_sso(
    metrics_df: pd.DataFrame,
    metric_type: str,
    output_dir: Path,
    title_suffix: str = "",
    *,
    n_dropped_guids: int = 0,
) -> None:
    """Render the five canonical metric-vs-time PNGs against the SSO axis.

    File names match the delivery-axis :func:`plot_single_metric_type`
    output so consumers (and the cross-fold aggregator) can locate the
    SSO equivalents by mirroring the directory layout. Differences from
    the delivery-axis variant:

    * X-axis is signed ``t_rel_sso_hours`` — no ``invert_xaxis``.
    * Vertical reference line at $t = 0$ marks SSO.
    * Figure footer annotates the number of GUIDs dropped for missing
      SSO timestamps.

    Args:
        metrics_df: Output of the SSO metric computations.
        metric_type: ``'instantaneous'`` / ``'committed_cumulative'`` /
            ``'committed_overall'``.
        output_dir: Directory to write PNGs into.
        title_suffix: Extra label rendered in every title (e.g. fold tag).
        n_dropped_guids: Dropped-GUID count (for the footer).
    """
    import matplotlib.pyplot as plt  # noqa: WPS433 - lazy to keep schema tests light
    output_dir.mkdir(parents=True, exist_ok=True)

    if metrics_df is None or len(metrics_df) == 0:
        logger.warning(f"plot_single_metric_type_sso: no data for {metric_type}")
        return

    valid_df = metrics_df[metrics_df["sensitivity"].notna()].copy()
    if len(valid_df) == 0:
        logger.warning(
            f"plot_single_metric_type_sso: no non-NaN bins for {metric_type}"
        )
        return

    valid_df = valid_df.sort_values("bin_center", ascending=True)
    x = valid_df["bin_center"].to_numpy()

    metric_labels = {
        "instantaneous": "Instantaneous Decisions",
        "committed_cumulative": "Committed Decisions (Cumulative)",
        "committed_overall": "Committed Decisions (Overall) - PRIMARY",
    }
    metric_label = metric_labels.get(metric_type, metric_type)

    colors = {
        "sensitivity": "#2ecc71",
        "specificity": "#3498db",
        "fpr": "#e74c3c",
    }

    def _finalize(ax, *, ylabel: str, title: str) -> None:
        ax.set_xlabel(_SSO_X_LABEL, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        _annotate_sso_zero(ax)

    # --- Plot 1: Sensitivity ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, valid_df["sensitivity"], marker="o", label="Sensitivity",
            color=colors["sensitivity"], linewidth=2.5, markersize=6)
    _finalize(ax, ylabel="Sensitivity",
              title=f"Sensitivity vs Time{_compose_title(metric_label, title_suffix)}")
    _add_dropped_guid_annotation(fig, n_dropped_guids)
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {output_dir.name}/sensitivity_vs_time.png")

    # --- Plot 2: Sensitivity + Specificity ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, valid_df["sensitivity"], marker="o", label="Sensitivity",
            color=colors["sensitivity"], linewidth=2.5, markersize=6)
    ax.plot(x, valid_df["specificity"], marker="s", label="Specificity",
            color=colors["specificity"], linewidth=2.5, markersize=6)
    _finalize(ax, ylabel="Metric Value",
              title=f"Sensitivity & Specificity vs Time{_compose_title(metric_label, title_suffix)}")
    _add_dropped_guid_annotation(fig, n_dropped_guids)
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_specificity_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {output_dir.name}/sensitivity_specificity_vs_time.png")

    # --- Plot 3: Sensitivity + FPR ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, valid_df["sensitivity"], marker="o", label="Sensitivity",
            color=colors["sensitivity"], linewidth=2.5, markersize=6)
    ax.plot(x, valid_df["fpr"], marker="^", label="FPR",
            color=colors["fpr"], linewidth=2.5, markersize=6)
    _finalize(ax, ylabel="Metric Value",
              title=f"Sensitivity & FPR vs Time{_compose_title(metric_label, title_suffix)}")
    _add_dropped_guid_annotation(fig, n_dropped_guids)
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_fpr_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {output_dir.name}/sensitivity_fpr_vs_time.png")

    # --- Plot 4: All Three Metrics ---
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(x, valid_df["sensitivity"], marker="o", label="Sensitivity",
            color=colors["sensitivity"], linewidth=2.5, markersize=7, linestyle="-")
    ax.plot(x, valid_df["specificity"], marker="s", label="Specificity",
            color=colors["specificity"], linewidth=2.5, markersize=7, linestyle="--")
    ax.plot(x, valid_df["fpr"], marker="^", label="FPR",
            color=colors["fpr"], linewidth=2.5, markersize=7, linestyle=":")
    ax.set_xlabel(_SSO_X_LABEL, fontsize=14)
    ax.set_ylabel("Metric Value", fontsize=14)
    ax.set_title(
        f"All Metrics vs Time{_compose_title(metric_label, title_suffix)}",
        fontsize=15, fontweight="bold",
    )
    ax.legend(fontsize=12, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    _annotate_sso_zero(ax)
    _add_dropped_guid_annotation(fig, n_dropped_guids)
    plt.tight_layout()
    plt.savefig(output_dir / "all_metrics_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {output_dir.name}/all_metrics_vs_time.png")

    # --- Plot 5: FPR Only ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, valid_df["fpr"], marker="^", label="FPR",
            color=colors["fpr"], linewidth=2.5, markersize=6)
    _finalize(ax, ylabel="FPR",
              title=f"FPR vs Time{_compose_title(metric_label, title_suffix)}")
    _add_dropped_guid_annotation(fig, n_dropped_guids)
    plt.tight_layout()
    plt.savefig(output_dir / "fpr_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {output_dir.name}/fpr_vs_time.png")


# ---------------------------------------------------------------------------
# Subgroup plots
# ---------------------------------------------------------------------------


def _select_compute_fn_sso(metric_type: str) -> Callable[..., pd.DataFrame]:
    return {
        "instantaneous": compute_instantaneous_metrics_sso,
        "committed_cumulative": compute_committed_cumulative_metrics_sso,
        "committed_overall": compute_committed_overall_metrics_sso,
    }[metric_type]


def plot_subgroup_analysis_sso(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    metric_type: str,
    subgroup_filters: Dict[str, Callable[[pd.DataFrame], pd.Series]],
    output_dir: Path,
    title_suffix: str = "",
    *,
    n_dropped_guids: int = 0,
) -> Dict[str, pd.DataFrame]:
    """SSO-axis sibling of :func:`clinical_metrics_utils.plot_subgroup_analysis`.

    Dispatches to the four SSO subgroup helpers
    (``_plot_diagnosis_comparison_sso`` / ``_plot_cs_stratification_sso`` /
    ``_plot_bg_stratification_sso`` / ``_plot_healthy_subgroups_sso``).
    Returns the per-subgroup metric DataFrames for caller-side
    persistence (parity with the delivery-axis function).

    Args:
        df: Predictions DataFrame (already SSO-eligible).
        time_bins: Signed bin edges.
        metric_type: One of the three metric-type names.
        subgroup_filters: Mapping from subgroup name to row-mask
            producer. Reuses
            :func:`clinical_metrics_utils.create_enhanced_subgroup_filters`.
        output_dir: Directory for per-metric-type subgroup PNGs.
        title_suffix: Extra title chip.
        n_dropped_guids: Footer annotation count.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"plot_subgroup_analysis_sso: computing {metric_type} for "
        f"{len(subgroup_filters)} subgroups"
    )

    compute_fn = _select_compute_fn_sso(metric_type)

    subgroup_metrics: Dict[str, pd.DataFrame] = {}
    for sg_name, sg_filter in subgroup_filters.items():
        try:
            sg_df = compute_fn(df, time_bins, sg_filter)
            if sg_df is not None and len(sg_df) > 0:
                subgroup_metrics[sg_name] = sg_df
        except Exception:
            logger.exception(
                f"plot_subgroup_analysis_sso: subgroup {sg_name!r} ({metric_type}) failed"
            )

    if not subgroup_metrics:
        logger.warning(
            f"plot_subgroup_analysis_sso: no valid subgroup metrics for {metric_type}"
        )
        return {}

    # Per-subgroup GUID counts for legend labels (parity with the
    # delivery-axis renderer).
    subgroup_guid_counts: Dict[str, int] = {}
    for sg_name, sg_filter in subgroup_filters.items():
        try:
            subgroup_guid_counts[sg_name] = int(df[sg_filter(df)]["guid"].nunique())
        except Exception:
            subgroup_guid_counts[sg_name] = 0

    _plot_diagnosis_comparison_sso(
        subgroup_metrics, metric_type, output_dir, title_suffix,
        subgroup_guid_counts=subgroup_guid_counts,
        n_dropped_guids=n_dropped_guids,
    )
    _plot_cs_stratification_sso(
        subgroup_metrics, metric_type, output_dir, title_suffix,
        subgroup_guid_counts=subgroup_guid_counts,
        n_dropped_guids=n_dropped_guids,
    )
    _plot_bg_stratification_sso(
        subgroup_metrics, metric_type, output_dir, title_suffix,
        subgroup_guid_counts=subgroup_guid_counts,
        n_dropped_guids=n_dropped_guids,
    )
    _plot_healthy_subgroups_sso(
        subgroup_metrics, metric_type, output_dir, title_suffix,
        subgroup_guid_counts=subgroup_guid_counts,
        n_dropped_guids=n_dropped_guids,
    )

    logger.info(f"plot_subgroup_analysis_sso: done for {metric_type}")
    return subgroup_metrics


def _plot_diagnosis_comparison_sso(
    subgroup_metrics: Dict[str, pd.DataFrame],
    metric_type: str,
    output_dir: Path,
    title_suffix: str,
    *,
    subgroup_guid_counts: Optional[Dict[str, int]] = None,
    n_dropped_guids: int = 0,
) -> None:
    import matplotlib.pyplot as plt  # noqa: WPS433
    diagnosis_groups = ["healthy", "acidosis", "hie", "unhealthy"]
    available = [g for g in diagnosis_groups if g in subgroup_metrics]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {
        "healthy": "#2ecc71",
        "acidosis": "#e74c3c",
        "hie": "#e67e22",
        "unhealthy": "#95a5a6",
    }
    for group in available:
        df_g = subgroup_metrics[group]
        valid = df_g[df_g["sensitivity"].notna()].sort_values(
            "bin_center", ascending=True
        )
        if valid.empty:
            continue
        n = subgroup_guid_counts.get(group, 0) if subgroup_guid_counts else 0
        label = f"{group.capitalize()} (N={n})" if n > 0 else group.capitalize()
        ax.plot(
            valid["bin_center"], valid["sensitivity"],
            marker="o", label=label, linewidth=2.5,
            color=colors.get(group), markersize=6,
        )
    ax.set_xlabel(_SSO_X_LABEL, fontsize=13)
    ax.set_ylabel("Sensitivity", fontsize=13)
    title = (
        f"Diagnosis Comparison - {metric_type.replace('_', ' ').title()} "
        f"({_SSO_TITLE_SUFFIX}"
    )
    if title_suffix:
        title += f"; {title_suffix.strip(' -:')}"
    title += ")"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    _annotate_sso_zero(ax)
    _add_dropped_guid_annotation(fig, n_dropped_guids)
    plt.tight_layout()
    plt.savefig(output_dir / "diagnosis_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {output_dir.name}/diagnosis_comparison.png")


def _plot_cs_stratification_sso(
    subgroup_metrics: Dict[str, pd.DataFrame],
    metric_type: str,
    output_dir: Path,
    title_suffix: str,
    *,
    subgroup_guid_counts: Optional[Dict[str, int]] = None,
    n_dropped_guids: int = 0,
) -> None:
    import matplotlib.pyplot as plt  # noqa: WPS433
    stratifications = [
        (["unhealthy_cs_pos", "unhealthy_cs_neg"], "Unhealthy by CS Status"),
        (["hie_cs_pos", "hie_cs_neg"], "HIE by CS Status"),
        (["acidosis_cs_pos", "acidosis_cs_neg"], "Acidosis by CS Status"),
    ]
    for groups, title_text in stratifications:
        available = [g for g in groups if g in subgroup_metrics]
        if not available:
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = {"pos": "#3498db", "neg": "#9b59b6"}
        for group in available:
            df_g = subgroup_metrics[group]
            valid = df_g[df_g["sensitivity"].notna()].sort_values(
                "bin_center", ascending=True
            )
            if valid.empty:
                continue
            base_label = "CS Positive" if "pos" in group else "CS Negative"
            n = subgroup_guid_counts.get(group, 0) if subgroup_guid_counts else 0
            label = f"{base_label} (N={n})" if n > 0 else base_label
            color = colors["pos"] if "pos" in group else colors["neg"]
            ax.plot(
                valid["bin_center"], valid["sensitivity"],
                marker="o", label=label, linewidth=2.5,
                color=color, markersize=6,
            )
        ax.set_xlabel(_SSO_X_LABEL, fontsize=13)
        ax.set_ylabel("Sensitivity", fontsize=13)
        title = (
            f"{title_text} - {metric_type.replace('_', ' ').title()} "
            f"({_SSO_TITLE_SUFFIX}"
        )
        if title_suffix:
            title += f"; {title_suffix.strip(' -:')}"
        title += ")"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        _annotate_sso_zero(ax)
        _add_dropped_guid_annotation(fig, n_dropped_guids)
        plt.tight_layout()
        filename = f"{groups[0].rsplit('_', 2)[0]}_cs_stratification.png"
        plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: {output_dir.name}/{filename}")


def _plot_bg_stratification_sso(
    subgroup_metrics: Dict[str, pd.DataFrame],
    metric_type: str,
    output_dir: Path,
    title_suffix: str,
    *,
    subgroup_guid_counts: Optional[Dict[str, int]] = None,
    n_dropped_guids: int = 0,
) -> None:
    import matplotlib.pyplot as plt  # noqa: WPS433
    bg_groups = ["acidosis_bg_pos", "acidosis_bg_neg"]
    available = [g for g in bg_groups if g in subgroup_metrics]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"pos": "#f39c12", "neg": "#16a085"}
    for group in available:
        df_g = subgroup_metrics[group]
        valid = df_g[df_g["sensitivity"].notna()].sort_values(
            "bin_center", ascending=True
        )
        if valid.empty:
            continue
        base_label = "BG Positive" if "pos" in group else "BG Negative"
        n = subgroup_guid_counts.get(group, 0) if subgroup_guid_counts else 0
        label = f"{base_label} (N={n})" if n > 0 else base_label
        color = colors["pos"] if "pos" in group else colors["neg"]
        ax.plot(
            valid["bin_center"], valid["sensitivity"],
            marker="o", label=label, linewidth=2.5,
            color=color, markersize=6,
        )
    ax.set_xlabel(_SSO_X_LABEL, fontsize=13)
    ax.set_ylabel("Sensitivity", fontsize=13)
    title = (
        f"Acidosis by BG Status - {metric_type.replace('_', ' ').title()} "
        f"({_SSO_TITLE_SUFFIX}"
    )
    if title_suffix:
        title += f"; {title_suffix.strip(' -:')}"
    title += ")"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    _annotate_sso_zero(ax)
    _add_dropped_guid_annotation(fig, n_dropped_guids)
    plt.tight_layout()
    plt.savefig(output_dir / "acidosis_bg_stratification.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {output_dir.name}/acidosis_bg_stratification.png")


def _plot_healthy_subgroups_sso(
    subgroup_metrics: Dict[str, pd.DataFrame],
    metric_type: str,
    output_dir: Path,
    title_suffix: str,
    *,
    subgroup_guid_counts: Optional[Dict[str, int]] = None,
    n_dropped_guids: int = 0,
) -> None:
    import matplotlib.pyplot as plt  # noqa: WPS433
    # Healthy by CS — uses specificity, not sensitivity (no positives).
    cs_groups = ["healthy_cs_pos", "healthy_cs_neg"]
    available_cs = [g for g in cs_groups if g in subgroup_metrics]
    if available_cs:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = {"pos": "#3498db", "neg": "#9b59b6"}
        for group in available_cs:
            df_g = subgroup_metrics[group]
            valid = df_g[df_g["specificity"].notna()].sort_values(
                "bin_center", ascending=True
            )
            if valid.empty:
                continue
            base_label = "CS Positive" if "pos" in group else "CS Negative"
            n = subgroup_guid_counts.get(group, 0) if subgroup_guid_counts else 0
            label = f"{base_label} (N={n})" if n > 0 else base_label
            color = colors["pos"] if "pos" in group else colors["neg"]
            ax.plot(
                valid["bin_center"], valid["specificity"],
                marker="o", label=label, linewidth=2.5,
                color=color, markersize=6,
            )
        ax.set_xlabel(_SSO_X_LABEL, fontsize=13)
        ax.set_ylabel("Specificity (Correctly Identified as Healthy)", fontsize=13)
        title = (
            f"Healthy by CS Status - {metric_type.replace('_', ' ').title()} "
            f"({_SSO_TITLE_SUFFIX}"
        )
        if title_suffix:
            title += f"; {title_suffix.strip(' -:')}"
        title += ")"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        _annotate_sso_zero(ax)
        _add_dropped_guid_annotation(fig, n_dropped_guids)
        plt.tight_layout()
        plt.savefig(output_dir / "healthy_cs_stratification.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: {output_dir.name}/healthy_cs_stratification.png")

    # Healthy by BG.
    bg_groups = ["healthy_bg_pos", "healthy_bg_neg"]
    available_bg = [g for g in bg_groups if g in subgroup_metrics]
    if available_bg:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = {"pos": "#f39c12", "neg": "#16a085"}
        for group in available_bg:
            df_g = subgroup_metrics[group]
            valid = df_g[df_g["specificity"].notna()].sort_values(
                "bin_center", ascending=True
            )
            if valid.empty:
                continue
            base_label = "BG Positive" if "pos" in group else "BG Negative"
            n = subgroup_guid_counts.get(group, 0) if subgroup_guid_counts else 0
            label = f"{base_label} (N={n})" if n > 0 else base_label
            color = colors["pos"] if "pos" in group else colors["neg"]
            ax.plot(
                valid["bin_center"], valid["specificity"],
                marker="o", label=label, linewidth=2.5,
                color=color, markersize=6,
            )
        ax.set_xlabel(_SSO_X_LABEL, fontsize=13)
        ax.set_ylabel("Specificity (Correctly Identified as Healthy)", fontsize=13)
        title = (
            f"Healthy by BG Status - {metric_type.replace('_', ' ').title()} "
            f"({_SSO_TITLE_SUFFIX}"
        )
        if title_suffix:
            title += f"; {title_suffix.strip(' -:')}"
        title += ")"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        _annotate_sso_zero(ax)
        _add_dropped_guid_annotation(fig, n_dropped_guids)
        plt.tight_layout()
        plt.savefig(output_dir / "healthy_bg_stratification.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: {output_dir.name}/healthy_bg_stratification.png")

    # Healthy BG x CS 4-way.
    combo_groups = [
        "healthy_bg_pos_cs_pos",
        "healthy_bg_pos_cs_neg",
        "healthy_bg_neg_cs_pos",
        "healthy_bg_neg_cs_neg",
    ]
    available_combo = [g for g in combo_groups if g in subgroup_metrics]
    if available_combo:
        fig, ax = plt.subplots(figsize=(14, 7))
        colors = ["#e74c3c", "#3498db", "#f39c12", "#9b59b6"]
        for i, group in enumerate(available_combo):
            df_g = subgroup_metrics[group]
            valid = df_g[df_g["specificity"].notna()].sort_values(
                "bin_center", ascending=True
            )
            if valid.empty:
                continue
            base_label = group.replace("healthy_", "").replace("_", " ").upper()
            n = subgroup_guid_counts.get(group, 0) if subgroup_guid_counts else 0
            label = f"{base_label} (N={n})" if n > 0 else base_label
            ax.plot(
                valid["bin_center"], valid["specificity"],
                marker="o", label=label, linewidth=2.5,
                color=colors[i % len(colors)], markersize=6,
            )
        ax.set_xlabel(_SSO_X_LABEL, fontsize=13)
        ax.set_ylabel("Specificity (Correctly Identified as Healthy)", fontsize=13)
        title = (
            f"Healthy BG x CS Combinations - {metric_type.replace('_', ' ').title()} "
            f"({_SSO_TITLE_SUFFIX}"
        )
        if title_suffix:
            title += f"; {title_suffix.strip(' -:')}"
        title += ")"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, loc="best", ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        _annotate_sso_zero(ax)
        _add_dropped_guid_annotation(fig, n_dropped_guids)
        plt.tight_layout()
        plt.savefig(output_dir / "healthy_bg_cs_combinations.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: {output_dir.name}/healthy_bg_cs_combinations.png")


# ---------------------------------------------------------------------------
# Subgroup long-format CSV writer (parity with the delivery side)
# ---------------------------------------------------------------------------


def persist_subgroup_long_csv(
    subgroup_metrics: Dict[str, pd.DataFrame],
    csv_path: Path,
) -> None:
    """Stack per-subgroup metric DataFrames into one long-format CSV.

    Mirrors the long-format produced by the delivery-side caller in
    :func:`evaluate_guid_classifier.evaluate_single_fold` so the
    cross-fold aggregator can read both trees with identical code.

    Args:
        subgroup_metrics: Mapping from subgroup name to per-subgroup
            metric DataFrame.
        csv_path: Output path.
    """
    rows: List[Dict[str, Any]] = []
    for sg_name, sg_df in (subgroup_metrics or {}).items():
        if sg_df is None or len(sg_df) == 0:
            continue
        for _, r in sg_df.iterrows():
            row: Dict[str, Any] = {"subgroup": sg_name}
            for col in (
                "bin_center",
                "sensitivity",
                "specificity",
                "fpr",
                "n_pos",
                "n_neg",
                "n",
                "n_positive",
                "n_negative",
                "n_positive_available",
                "n_negative_available",
                "n_positive_total",
                "n_negative_total",
                "n_detected_positive",
                "n_detected_negative",
            ):
                if col in r.index:
                    row[col] = r[col]
            rows.append(row)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# JSON serialisation helper for the per-fold summary
# ---------------------------------------------------------------------------


def write_sso_filter_summary(path: Path, stats: Dict[str, Any]) -> None:
    """Persist the dropped-GUID statistics for a fold's SSO eval.

    Args:
        path: Output JSON path.
        stats: Dict from :func:`filter_to_sso_eligible`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "SSO_TIME_COL",
    "ensure_t_rel_sso_hours",
    "filter_to_sso_eligible",
    "recompute_t_rel_sso_after_fill",
    "compute_sso_time_bins",
    "compute_instantaneous_metrics_sso",
    "compute_committed_cumulative_metrics_sso",
    "compute_committed_overall_metrics_sso",
    "plot_single_metric_type_sso",
    "plot_subgroup_analysis_sso",
    "persist_subgroup_long_csv",
    "write_sso_filter_summary",
]
